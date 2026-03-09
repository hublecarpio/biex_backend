"""
Lógica del Gatekeeper y enrutado condicional.
"""
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import GatekeeperEval
from app.services.supabase_api import SupabaseClient

logger = logging.getLogger(__name__)
cache_logger = logging.getLogger("cache_monitor")


def _log_cache_stats(response, context: str) -> None:
    """Loggea las estadísticas de Gemini Implicit Caching del response."""
    try:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return
        cached = getattr(metadata, "cached_content_token_count", 0) or 0
        total = getattr(metadata, "prompt_token_count", 0) or 0
        ratio = cached / total if total else 0
        cache_logger.info(
            "CACHE_STATS | context=%s | cached=%s | total_input=%s | ratio=%.2f%%",
            context, cached, total, ratio * 100,
        )
    except Exception:
        pass  # No romper el flujo por un error de logging

# ---------------------------------------------------------------------------
# Singleton LLM para el Gatekeeper — misma estrategia que nodes.py
# ---------------------------------------------------------------------------
_gatekeeper_llm_instance = None


def _get_llm():
    """Retorna la instancia singleton del LLM del Gatekeeper con fallback."""
    global _gatekeeper_llm_instance
    if _gatekeeper_llm_instance is None:
        settings = get_settings()
        primary = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        fallback = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        _gatekeeper_llm_instance = primary.with_fallbacks([fallback])
        logger.info("[gatekeeper] Instancia LLM creada (singleton).")
    return _gatekeeper_llm_instance



_gatekeeper_protocol_cache: str | None = None
_gatekeeper_protocol_cached_at: float = 0.0
_GATEKEEPER_PROTOCOL_TTL = 600.0  # 10 minutos

async def _get_gatekeeper_protocol() -> str:
    """Obtiene el protocolo del gatekeeper con cache TTL de 10 minutos."""
    global _gatekeeper_protocol_cache, _gatekeeper_protocol_cached_at
    now = time.monotonic()
    if (_gatekeeper_protocol_cache is not None
            and (now - _gatekeeper_protocol_cached_at) < _GATEKEEPER_PROTOCOL_TTL):
        return _gatekeeper_protocol_cache

    client = SupabaseClient()
    try:
        protocol = await client.get_active_protocol("gatekeeper")
        result = protocol or "Evalúa la comprensión y frustración del alumno."
        _gatekeeper_protocol_cache = result
        _gatekeeper_protocol_cached_at = now
        logger.info("[gatekeeper] Protocolo cacheado (%s chars, TTL 10min).", len(result))
        return result
    except Exception as e:
        logger.warning("[gatekeeper] Error obteniendo protocolo: %s", e)
        if _gatekeeper_protocol_cache:
            return _gatekeeper_protocol_cache
        return "Evalúa la comprensión y frustración del alumno."
    finally:
        await client.close()


def _get_default_gatekeeper_eval(reason: str) -> dict:
    """Retorna un dict con los valores por defecto cuando el LLM falla."""
    logger.warning("[gatekeeper] Usando default eval por: %s", reason)
    return {
        "gatekeeper_eval": {
            "comprension_score": 50.0,
            "frustracion_detectada": False,
            "frustracion_nivel": 0,
            "engagement_score": 50.0,
            "misconceptions": [],
            "recomendacion": "continuar",
            "justificacion": f"Fallback por defecto: {reason}",
        }
    }


async def evaluate_gatekeeper(state: GraphState) -> dict:
    """
    Evalúa al alumno (comprensión, frustración, engagement etc.)
    Retorna estado actualizado con "gatekeeper_eval".
    """
    messages = state.get("messages") or []
    if not messages:
        return _get_default_gatekeeper_eval("Sin mensajes")

    # Mantenemos lógica de enrutar o fallar si no hay HumanMessage al final
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return _get_default_gatekeeper_eval("Último mensaje no es HumanMessage")

    user_content = last.content
    text = user_content if isinstance(user_content, str) else str(user_content)
    if not text.strip():
        return _get_default_gatekeeper_eval("Mensaje vacío")

    logger.info("[gatekeeper] Evaluando mensaje del alumno (%s chars)...", len(text))

    # 1. Cargar el protocolo del gatekeeper
    gk_protocol = await _get_gatekeeper_protocol()

    # 2. Extraer Perfil del alumno
    starter_profile = state.get("starter_profile") or {}
    profile_data = starter_profile.get("profile_data") or {}
    
    edad = starter_profile.get("age", "No especificada")
    intereses = profile_data.get("interests", [])
    estilo = profile_data.get("learningStyle", [])

    # 3. Insights previos (los primeros 5)
    learner_insights = state.get("learner_insights") or []
    insights_texto = "\n".join(
        [f"- {i.get('insight_type', '')}: {i.get('insight_value', '')} (confianza: {i.get('confidence', '')})"
         for i in learner_insights[:5]]
    ) if learner_insights else "Ninguno"

    # 4. Estado de sesión
    session_state = state.get("session_state") or {}
    fase_actual = session_state.get("session_phase", "generativa")
    interacciones = session_state.get("interaction_count", 0)
    
    comp_history = session_state.get("comprehension_history", [])[-5:]
    zdp = session_state.get("zdp_level", 50.0)
    frust_history = session_state.get("frustration_history", [])[-3:]

    # 5. Últimos 15 mensajes (alineado con build_response_messages)
    last_msgs = messages[-15:]
    historial_str = ""
    for m in last_msgs:
        if isinstance(m, HumanMessage):
            rol = "ALUMNO"
        elif isinstance(m, AIMessage):
            rol = "SOFÍA"
        else:
            rol = "SISTEMA"
        ct = m.content if isinstance(m.content, str) else str(m.content)
        historial_str += f"{rol}: {ct}\n\n"

    # Construir prompt
    prompt = (
        f"{gk_protocol}\n\n"
        "## Perfil del Alumno\n"
        f"- Edad: {edad}\n"
        f"- Intereses: {intereses}\n"
        f"- Estilo de aprendizaje: {estilo}\n\n"
        "## Insights Previos (hasta 5)\n"
        f"{insights_texto}\n\n"
        "## Estado de Sesión\n"
        f"- Fase actual: {fase_actual}\n"
        f"- Interacciones: {interacciones}\n"
        f"- Historial Comprensión (últimos 5): {comp_history}\n"
        f"- ZDP Nivel: {zdp}\n"
        f"- Historial Frustración (últimos 3): {frust_history}\n\n"
        "## Últimos mensajes de la conversación\n"
        f"{historial_str}\n\n"
        "Evaluá el último mensaje del alumno considerando todo el contexto anterior.\n\n"
        "Respondé ÚNICAMENTE con un JSON válido con esta estructura exacta:\n"
        '{"comprension_score": <0-100>, "frustracion_detectada": <true|false>, '
        '"frustracion_nivel": <0-10>, "engagement_score": <0-100>, '
        '"misconceptions": ["<string>", ...], '
        '"recomendacion": "<continuar|intensificar|simplificar|vicario|socratico|metacognicion>", '
        '"justificacion": "<1 oración breve>", '
        '"topic": "<tema educativo principal o vacío si no hay>"}'
    )

    llm = _get_llm()

    try:
        raw_result = await llm.with_structured_output(GatekeeperEval, include_raw=True).ainvoke(prompt)
        eval_result: GatekeeperEval = raw_result["parsed"]
        _log_cache_stats(raw_result.get("raw"), "gatekeeper")
    except Exception as e:
        logger.error("[gatekeeper] Falló el LLM durante la evaluación: %s", e)
        return _get_default_gatekeeper_eval("Fallo LLM")


    logger.info(
        "[gatekeeper] Evaluación exitosa: comprensión=%.1f | frustración=%s | engagement=%.1f | rec=%s",
        eval_result.comprension_score,
        eval_result.frustracion_detectada,
        eval_result.engagement_score,
        eval_result.recomendacion,
    )

    try:
        resultado = eval_result.model_dump()
    except AttributeError:
        resultado = eval_result.dict()

    return {"gatekeeper_eval": resultado}
