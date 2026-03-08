"""
Nodos del grafo BIEX: setup, generativo, vicario, socrático, metacognición.

Optimizaciones de rendimiento aplicadas:
 - LLM singleton a nivel de módulo (evita re-instanciación por nodo/request)
 - Cache TTL para system_prompt (evita llamada HTTP en cada request)
 - asyncio.gather para paralelizar las 3 llamadas Supabase en node_setup
 - Generación de imágenes en asyncio background task (no bloquea la respuesta)
   → El state retorna images_job_id + images_pending para que el frontend
     pueda mostrar placeholders y hacer polling al endpoint de jobs.
"""
import asyncio
import ast
import json
import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import ImageTopicList
from app.services.image_jobs import complete_job, create_job, fail_job
from app.services.supabase_api import SupabaseClient
from app.utils.cache import get_pedagogical_context_cached

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton LLM — se construye UNA sola vez al importar el módulo.
# Usar get_llm() para acceder a la instancia cacheada.
# ---------------------------------------------------------------------------
_llm_instance: ChatGoogleGenerativeAI | None = None
_llm_image_topic_instance: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Retorna la instancia singleton del LLM principal con fallback."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        primary = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.7,
        )
        fallback = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=settings.gemini_api_key,
            temperature=0.7,
        )
        _llm_instance = primary.with_fallbacks([fallback])
        logger.info("[llm] Instancia LLM principal creada (singleton).")
    return _llm_instance


def _get_llm_image_topic() -> ChatGoogleGenerativeAI:
    """Retorna la instancia singleton del LLM auxiliar para extracción de temas."""
    global _llm_image_topic_instance
    if _llm_image_topic_instance is None:
        settings = get_settings()
        _llm_image_topic_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        logger.info("[llm] Instancia LLM de temas de imagen creada (singleton).")
    return _llm_image_topic_instance


# ---------------------------------------------------------------------------
# Cache TTL para el system_prompt — rara vez cambia, evita 1 HTTP por request
# ---------------------------------------------------------------------------
_system_prompt_cache: str | None = None
_system_prompt_cached_at: float = 0.0
_SYSTEM_PROMPT_TTL = 300.0  # 5 minutos


async def _get_system_prompt_cached(client: SupabaseClient) -> str:
    """
    Retorna el system_prompt desde cache si no expiró, o lo refresca desde Supabase.
    El TTL es de 5 minutos — cambios en Supabase se propagan en ese plazo.
    """
    global _system_prompt_cache, _system_prompt_cached_at
    now = time.monotonic()
    if _system_prompt_cache is not None and (now - _system_prompt_cached_at) < _SYSTEM_PROMPT_TTL:
        logger.info("[node_setup] system_prompt desde cache (%.0fs restantes).",
                    _SYSTEM_PROMPT_TTL - (now - _system_prompt_cached_at))
        return _system_prompt_cache

    logger.info("[node_setup] Refrescando system_prompt desde Supabase...")
    prompt = await client.get_system_prompt()
    if prompt:
        _system_prompt_cache = prompt
        _system_prompt_cached_at = now
    return prompt or "Eres un tutor educativo amable y claro."


# ---------------------------------------------------------------------------
# Helpers de contenido
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """
    Extrae texto plano del contenido de la respuesta del LLM.
    Maneja strings simples, listas de content blocks (type/text/extras),
    y representaciones en string de diccionarios o listas.
    """
    if isinstance(content, str):
        cleaned = content.strip()
        if (cleaned.startswith("[") and cleaned.endswith("]")) or (
            cleaned.startswith("{") and cleaned.endswith("}")
        ):
            try:
                parsed = ast.literal_eval(cleaned)
                content = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _build_system_content(system_prompt: str, starter_profile: dict) -> str:
    """Construye el contenido del system message con prompt y perfil."""
    profile_str = json.dumps(starter_profile, ensure_ascii=False, indent=2)
    return f"""{system_prompt}

## Perfil del alumno (usa esto para personalizar)
{profile_str}
"""


# ---------------------------------------------------------------------------
# Extracción de temas visuales (auxiliar — usa LLM propio)
# ---------------------------------------------------------------------------

async def _extract_image_topics(response_text: str) -> list[dict]:
    """
    Usa el LLM para extraer temas visuales de la respuesta del tutor.
    Retorna lista de dicts {"tema": ..., "descripcion": ...} o vacía si no aplica.
    """
    if not response_text.strip():
        return []

    structured_llm = _get_llm_image_topic().with_structured_output(ImageTopicList)

    prompt = (
        "Analiza la siguiente respuesta de un tutor educativo. "
        "Identifica los conceptos o elementos visuales concretos que se beneficiarían de una imagen ilustrativa "
        "(diagramas, partes de objetos, procesos, estructuras, etc.). "
        "Para cada tema visual, provee un nombre corto y una descripción detallada de 2-3 oraciones "
        "que ayude al generador de imágenes a crear algo preciso y educativo. "
        "Si la respuesta es puramente conversacional o no requiere imágenes, devuelve una lista vacía.\n\n"
        f"Respuesta del tutor:\n{response_text}"
    )

    try:
        result: ImageTopicList = await structured_llm.ainvoke(prompt)
        topics = [{"tema": t.tema, "descripcion": t.descripcion} for t in result.temas]
        if topics:
            logger.info("[image_topics] %s tema(s) detectados: %s", len(topics), [t["tema"] for t in topics])
        else:
            logger.info("[image_topics] No se detectaron temas visuales.")
        return topics
    except Exception as e:
        logger.warning("[image_topics] Error extrayendo temas: %s", e)
        return []


# ---------------------------------------------------------------------------
# Worker de imágenes en background
# ---------------------------------------------------------------------------

async def _generate_images_background(job_id: str, topics: list[dict]) -> None:
    """
    Task de asyncio que ejecuta la generación de imágenes en background.
    Actualiza el job store al completar o fallar.
    No bloquea la respuesta al cliente.
    """
    from app.services.image_service import generate_images  # import local para evitar ciclos
    try:
        urls = await generate_images(topics)
        await complete_job(job_id, urls)
    except Exception as e:
        logger.error("[images_bg] Error en job %s: %s", job_id, e)
        await fail_job(job_id, str(e))


# ---------------------------------------------------------------------------
# Nodo principal: setup
# ---------------------------------------------------------------------------

async def should_query_rag(message: str, client: SupabaseClient) -> bool:
    """Clasificador heurístico basado en LLM para decidir si hacer RAG."""
    if not message.strip():
        return False
        
    prompt = f"""Clasificá este mensaje de un alumno en una tutoría educativa.
Respondé SOLO con un JSON: {{"needs_rag": true}} o {{"needs_rag": false}}
Respondé true SOLO si el alumno está haciendo una pregunta sobre
contenido educativo de una materia o necesita información factual.
Respondé false si es: saludo, despedida, expresión emocional,
respuesta corta (sí/no/ok), o meta-conversación.
Mensaje: {message}"""

    try:
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        response = await llm.ainvoke(prompt)
        content = _extract_text(response.content if hasattr(response, "content") else response)
        
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        return parsed.get("needs_rag", True)
    except Exception as e:
        logger.warning("[should_query_rag] Error en clasificador RAG: %s. Default fallback a True.", e)
        return True


async def node_setup(state: GraphState) -> dict:
    t0 = time.perf_counter()
    messages: list = state.get("messages") or []
    user_id: str = state.get("user_id") or ""
    conversation_id: str = state.get("conversation_id") or ""

    logger.info("[node_setup] Iniciando para user_id=%s, conversation_id=%s", user_id, conversation_id)

    if not messages:
        logger.warning("[node_setup] No hay mensajes en el estado, retornando vacío.")
        return {
            "starter_profile": {}, 
            "system_prompt": "", 
            "rag_context": "",
            "session_state": {},
            "pedagogical_context": "",
            "fase_actual": "generativa",
            "learner_insights": []
        }

    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        user_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    else:
        user_text = ""

    logger.info("[node_setup] Consultando dependencias...")
    client = SupabaseClient()
    try:
        # c) Cargar pedagogical_context (con caché de memoria)
        pedagogical_context = await get_pedagogical_context_cached(client)

        # d) Obtener el resto en paralelo
        system_prompt_raw, starter_profile_obj, session_state, learner_insights = await asyncio.gather(
            _get_system_prompt_cached(client),
            client.get_starter_profile(user_id),
            client.get_session_state(conversation_id),
            client.get_learner_insights(user_id),
        )

        # e) Si session_state no existe, crear defaults
        if not session_state:
            session_state = {
               "conversation_id": conversation_id, 
               "user_id": user_id,
               "session_phase": "generativa", 
               "interaction_count": 0,
               "comprehension_history": [], 
               "frustration_history": [],
               "engagement_history": [], 
               "zdp_level": 50.0,
               "cognitive_resilience": 50.0, 
               "current_comprehension": 0.0,
               "topics_covered": [], 
               "misconceptions": [],
               "socratic_questions_answered": 0, 
               "socratic_correct_answers": 0,
               "gatekeeper_override": False, 
               "phase_transitions": [],
               "vicario_triggers": 0
            }

        # f) Obtener current_phase (referencia inicial)
        current_phase = session_state.get("session_phase", "generativa")

        # g) RAG condicional
        rag_context = ""
        needs_rag = await should_query_rag(user_text, client)
        if needs_rag:
            logger.info("[node_setup] Clasificador decidió MANTENER RAG para: '%s...'", user_text[:30].replace('\n', ' '))
            rag_context = await client.query_knowledge(user_text)
        else:
            logger.info("[node_setup] Clasificador decidió SALTAR RAG para: '%s...'", user_text[:30].replace('\n', ' '))

    finally:
        await client.close()

    starter_profile = starter_profile_obj.model_dump() if starter_profile_obj else {}
    elapsed = time.perf_counter() - t0
    
    logger.info(
        "[node_setup] Setup OK en %.2fs — RAG=%s",
        elapsed,
        "Ejecutado" if needs_rag else "Saltado",
    )

    # h) Retornar resultado
    return {
        "system_prompt": system_prompt_raw or "Eres un tutor educativo amable y claro.",
        "starter_profile": starter_profile,
        "session_state": session_state,
        "pedagogical_context": pedagogical_context,
        "rag_context": rag_context,
        "fase_actual": current_phase,
        "learner_insights": learner_insights or [],
    }


# ---------------------------------------------------------------------------
# Invocador central con imágenes en background
# ---------------------------------------------------------------------------

async def _invoke_with_images(llm, full_messages: list[BaseMessage], fase: str) -> dict:
    """
    Invoca el LLM, extrae el texto y lanza la generación de imágenes en background.

    El state retorna:
    - images_job_id: ID del job de background (None si no hay imágenes).
    - images_pending: cantidad de imágenes que se están generando.
    - image_urls: vacío (las URLs llegan via polling al job endpoint).

    El cliente consulta GET /api/v1/images/{images_job_id} para obtener las URLs
    cuando estén listas (status == "done").
    """
    t0 = time.perf_counter()
    response = await llm.ainvoke(full_messages)
    content = _extract_text(response.content if hasattr(response, "content") else response)
    llm_elapsed = time.perf_counter() - t0
    logger.info("[%s] LLM respondió en %.2fs — %s chars.", fase, llm_elapsed, len(content))

    # Extracción de temas — llamada LLM auxiliar rápida
    topics = await _extract_image_topics(content)

    images_job_id: str | None = None
    images_pending: int = 0

    if topics:
        # ✅ OPTIMIZACIÓN: las imágenes se generan en background, no bloqueamos la respuesta
        job_id = await create_job(images_pending=len(topics))
        asyncio.ensure_future(_generate_images_background(job_id, topics))
        images_job_id = job_id
        images_pending = len(topics)
        logger.info("[%s] Imágenes lanzadas en background — job_id=%s, pending=%s", fase, job_id, images_pending)

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": fase,
        "image_urls": [],          # vacío: las URLs llegan async vía job polling
        "images_job_id": images_job_id,
        "images_pending": images_pending,
    }


# ---------------------------------------------------------------------------
# Nodos de respuesta
# ---------------------------------------------------------------------------

async def node_generativo(state: GraphState) -> dict:
    """Genera respuesta educativa usando rag_context. Fase: generativa."""
    logger.info("[node_generativo] Generando respuesta...")
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}
    rag_context = state.get("rag_context") or ""

    system_content = _build_system_content(system_prompt, starter_profile)
    if rag_context:
        system_content += f"\n\n## Contexto de conocimiento (úsalo para fundamentar tu respuesta)\n{rag_context}"

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    return await _invoke_with_images(llm, full_messages, "generativa")


async def node_vicario(state: GraphState) -> dict:
    """Modo empatía / pensamiento en voz alta. Fase: vicaria."""
    logger.info("[node_vicario] Generando respuesta vicaria (empatía)...")
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += (
        "\n\nInstrucción: Responde en modo vicario: muestra empatía, piensa en voz alta "
        "y acompaña al alumno sin dar la respuesta directa. No uses el contexto RAG de forma "
        "rígida; prioriza el estado emocional y el perfil del alumno."
    )

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    return await _invoke_with_images(llm, full_messages, "vicaria")


async def node_socratico(state: GraphState) -> dict:
    """Solo hace preguntas de pensamiento crítico. Fase: socratica."""
    logger.info("[node_socratico] Generando preguntas socráticas...")
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += (
        "\n\nInstrucción: Responde únicamente con una o dos preguntas socráticas para guiar "
        "el pensamiento crítico del alumno. No des explicaciones largas ni la respuesta; "
        "solo preguntas que le hagan reflexionar."
    )

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    return await _invoke_with_images(llm, full_messages, "socratica")


async def node_metacognicion(state: GraphState) -> dict:
    """Evalúa la sesión al final (nodo de cierre)."""
    logger.info("[node_metacognicion] Cerrando sesión con fase metacognición.")
    return {"fase_actual": "metacognicion"}
