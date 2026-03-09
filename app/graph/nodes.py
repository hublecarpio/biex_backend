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
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import ImageTopicList
from app.services.image_jobs import complete_job, create_job, fail_job
from app.services.supabase_api import SupabaseClient
from app.utils.cache import get_pedagogical_context_cached

logger = logging.getLogger(__name__)
cache_logger = logging.getLogger("cache_monitor")


def _log_cache_stats(response, context: str) -> None:
    """Loggea las estadísticas de Gemini Implicit Caching del response."""
    try:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return

        if isinstance(metadata, dict):
            total = metadata.get("input_tokens", 0) or 0
            output = metadata.get("output_tokens", 0) or 0
            details = metadata.get("input_token_details") or {}
            cached = details.get("cache_read", 0) or 0 if isinstance(details, dict) else 0
        else:
            total = getattr(metadata, "input_tokens", 0) or getattr(metadata, "prompt_token_count", 0) or 0
            output = getattr(metadata, "output_tokens", 0) or getattr(metadata, "candidates_token_count", 0) or 0
            details = getattr(metadata, "input_token_details", None)
            if isinstance(details, dict):
                cached = details.get("cache_read", 0) or 0
            else:
                cached = getattr(metadata, "cached_content_token_count", 0) or 0

        ratio = cached / total if total else 0
        cache_logger.info(
            "CACHE_STATS | context=%s | cached=%s | total_input=%s | output=%s | ratio=%.2f%%",
            context, cached, total, output, ratio * 100,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Singleton LLM — se construye UNA sola vez al importar el módulo.
# Usar get_llm() para acceder a la instancia cacheada.
# ---------------------------------------------------------------------------
_llm_instance: ChatGoogleGenerativeAI | None = None
_llm_image_topic_instance: ChatGoogleGenerativeAI | None = None
_rag_classifier_llm: ChatGoogleGenerativeAI | None = None


def _get_rag_classifier_llm() -> ChatGoogleGenerativeAI:
    """Retorna la instancia singleton del clasificador RAG."""
    global _rag_classifier_llm
    if _rag_classifier_llm is None:
        settings = get_settings()
        _rag_classifier_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        logger.info("[llm] Instancia LLM clasificador RAG creada (singleton).")
    return _rag_classifier_llm


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
        "Eres un curador de contenido visual educativo. Analiza esta respuesta de un tutor "
        "y extrae SOLO conceptos que REQUIERAN una imagen para ser comprendidos correctamente.\n\n"
        "REGLAS ESTRICTAS:\n"
        "- Máximo 2 temas visuales por respuesta.\n"
        "- Solo elementos que un estudiante NO puede imaginar fácilmente por sí mismo.\n"
        "- Priorizar: diagramas de procesos, estructuras anatómicas/científicas, relaciones causa-efecto, "
        "comparaciones visuales, ciclos, mapas conceptuales.\n"
        "- NO generar imágenes para: definiciones simples, conceptos abstractos sin forma visual, "
        "listas, fechas, datos numéricos, emociones, motivación.\n"
        "- Cada descripción debe ser PRECISA y DETALLADA (2-3 oraciones) indicando: qué mostrar, "
        "qué etiquetas incluir, qué estilo visual usar (diagrama, ilustración, esquema, corte transversal, etc.).\n"
        "- Si la respuesta es conversacional, de acompañamiento emocional, o no tiene contenido visual "
        "que realmente necesite ilustración, devuelve una lista VACÍA.\n\n"
        f"Respuesta del tutor:\n{response_text[:1500]}"
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

def _detect_resource_suggestions(state: GraphState, fase: str) -> list[str]:
    """Detecta qué recursos multimedia serían útiles. MUY conservador para economizar costos.
    Solo sugiere cuando hay evidencia fuerte de que el recurso será valioso."""
    # NUNCA sugerir en fases emocionales o de evaluación
    if fase in ("vicario", "socratico"):
        return []

    session = state.get("session_state") or {}
    gk = state.get("gatekeeper_eval") or {}
    suggestions = []

    interaction_count = session.get("interaction_count", 0)
    topics_covered = session.get("topics_covered", [])
    comprension = gk.get("comprension_score", 50)

    # Mapa mental: solo cuando hay suficiente material consolidado
    if len(topics_covered) >= 5 and comprension >= 75 and interaction_count >= 10:
        suggestions.append("mind_map")

    # Fichas: después de una sesión sustancial con buena comprensión
    if interaction_count >= 14 and comprension >= 65 and len(topics_covered) >= 3:
        suggestions.append("fichas")

    # Video, Podcast e Informe: EXCLUSIVAMENTE en metacognición (cierre)
    if fase == "metacognicion":
        suggestions.append("video")
        suggestions.append("podcast")
        suggestions.append("informe")

    # Máximo 3 sugerencias para no saturar la UI
    suggestions = suggestions[:3]

    if suggestions:
        logger.info("[resource_suggestions] Sugerencias: %s (interacciones=%s, topics=%s, comp=%.0f)",
                    suggestions, interaction_count, len(topics_covered), comprension)

    return suggestions


async def _generate_images_background(job_id: str, topics: list[dict], conversation_id: str = "") -> None:
    """
    Task de asyncio que genera imágenes en background.
    Cuando las imágenes están listas, actualiza directamente el mensaje en Supabase
    para que el frontend lo reciba vía Realtime UPDATE (sin polling).
    """
    from app.services.image_service import generate_images  # import local para evitar ciclos
    try:
        urls = await generate_images(topics)
        await complete_job(job_id, urls)

        # Actualizar el mensaje en Supabase con las URLs resueltas
        if conversation_id:
            # Esperar a que la Edge Function haya guardado el mensaje
            await asyncio.sleep(8)
            client = SupabaseClient()
            try:
                updated = await client.update_message_images(conversation_id, job_id, urls)
                if not updated:
                    logger.warning("[images_bg] Mensaje no encontrado para job %s, reintentando en 10s...", job_id)
                    await asyncio.sleep(10)
                    updated = await client.update_message_images(conversation_id, job_id, urls)
                    if not updated:
                        logger.error("[images_bg] Mensaje no encontrado después de reintento — job %s", job_id)
                    else:
                        logger.info("[images_bg] Mensaje actualizado con %s imagen(es) — job %s (reintento)", len(urls), job_id)
                else:
                    logger.info("[images_bg] Mensaje actualizado con %s imagen(es) — job %s", len(urls), job_id)
            finally:
                await client.close()
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
        
    # Pre-filtro heurístico: mensajes cortos y comunes que nunca necesitan RAG
    lower = message.lower().strip()
    NO_RAG_PATTERNS = [
        "hola", "chau", "gracias", "dale", "ok", "si", "no", "sí",
        "bueno", "bien", "genial", "perfecto", "listo", "ya",
        "no sé", "no se", "no entiendo", "me aburro", "me cuesta",
        "jaja", "xd", "jeje",
    ]
    if len(lower) < 25 and any(lower.startswith(p) or lower == p for p in NO_RAG_PATTERNS):
        logger.info("[should_query_rag] Pre-filtro heurístico: NO RAG para '%s'", lower[:30])
        return False

    prompt = f'''Clasificá este mensaje de un alumno en una tutoría educativa.
Respondé SOLO con JSON: {{"needs_rag": true}} o {{"needs_rag": false}}

Respondé true si el alumno pregunta sobre contenido educativo de una materia
o necesita información factual para resolver un problema.
Respondé false si es saludo, despedida, expresión emocional, respuesta corta,
meta-conversación, o cualquier mensaje que no requiera buscar contenido.

Ejemplos:
- "¿qué es la fotosíntesis?" → {{"needs_rag": true}}
- "hola sofía" → {{"needs_rag": false}}
- "no entiendo nada" → {{"needs_rag": false}}
- "¿cuáles son las partes de la célula?" → {{"needs_rag": true}}
- "sí, creo que sí" → {{"needs_rag": false}}
- "¿y eso cómo se relaciona con la mitosis?" → {{"needs_rag": true}}
- "me aburro" → {{"needs_rag": false}}
- "explicame las leyes de newton" → {{"needs_rag": true}}
- "gracias, entendí" → {{"needs_rag": false}}
- "¿qué pasó en la revolución francesa?" → {{"needs_rag": true}}

Mensaje: "{message}"'''

    try:
        llm = _get_rag_classifier_llm()
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
        system_prompt_raw, starter_profile_obj, session_state, learner_insights, needs_rag = await asyncio.gather(
            _get_system_prompt_cached(client),
            client.get_starter_profile(user_id),
            client.get_session_state(conversation_id),
            client.get_learner_insights(user_id),
            should_query_rag(user_text, client),
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
        if needs_rag:
            logger.info("[node_setup] Clasificador decidió MANTENER RAG para: '%s...'", user_text[:30].replace('\n', ' '))
            rag_context = await client.query_knowledge(user_text)
        else:
            logger.info("[node_setup] Clasificador decidió SALTAR RAG para: '%s...'", user_text[:30].replace('\n', ' '))

    finally:
        await client.close()

    starter_profile = starter_profile_obj.model_dump(by_alias=True) if starter_profile_obj else {}
    elapsed = time.perf_counter() - t0
    
    is_new_session = not bool(session_state.get("interaction_count"))
    logger.info(
        "[node_setup] Setup OK en %.2fs — fase=%s | interacción=#%s | insights=%s | RAG=%s | sesión=%s",
        elapsed,
        current_phase,
        session_state.get("interaction_count", 0),
        len(learner_insights or []),
        "Ejecutado" if needs_rag else "Saltado",
        "NUEVA" if is_new_session else "existente",
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

async def _invoke_with_images(llm, full_messages: list[BaseMessage], fase: str, state: GraphState = None) -> dict:
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
    # Explicit cache como fallback del implicit
    cached_content_name = None
    if state and fase in ("generativa", "vicario", "socratico", "metacognicion"):
        try:
            from app.services.cache_manager import get_or_create_cache
            # Extraer contenido estático del SystemMessage
            system_msgs = [m for m in full_messages if isinstance(m, SystemMessage)]
            if system_msgs:
                static_content = system_msgs[0].content
                cached_content_name = await get_or_create_cache(static_content, fase)
        except Exception as e:
            logger.debug("[cache] Explicit cache no disponible: %s", e)

    if cached_content_name:
        try:
            from app.core.config import get_settings
            from langchain_google_genai import ChatGoogleGenerativeAI as _CachedLLM
            settings = get_settings()
            cached_llm = _CachedLLM(
                model="gemini-2.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.7,
                cached_content=cached_content_name,
            )
            history_only = [m for m in full_messages if not isinstance(m, SystemMessage)]
            response = await cached_llm.ainvoke(history_only)
            logger.info("[%s] Respuesta con EXPLICIT CACHE", fase)
        except Exception as e:
            logger.warning("[%s] Explicit cache falló, fallback a implicit: %s", fase, e)
            response = await llm.ainvoke(full_messages)
    else:
        response = await llm.ainvoke(full_messages)
    content = _extract_text(response.content if hasattr(response, "content") else response)
    llm_elapsed = time.perf_counter() - t0
    logger.info("[%s] LLM respondió en %.2fs — %s chars.", fase, llm_elapsed, len(content))

    # Gemini Implicit Caching — loggear cache hits para verificar descuentos
    _log_cache_stats(response, fase)

    images_job_id: str | None = None
    images_pending: int = 0

    # Solo extraer temas visuales en generativa
    # En socrático y vicario no se generan imágenes (ahorro de tokens)
    if fase == "generativa":
        # Guard: no extraer temas visuales en interacciones tempranas, comprensión baja,
        # o contenido corto (saludo, respuesta breve). Ahorra 1 llamada LLM por turno.
        session_check = state.get("session_state") or {} if state else {}
        skip_images = (
            session_check.get("interaction_count", 0) < 1
            or len(content) < 100
        )

        if skip_images:
            logger.info("[%s] Extracción de temas visuales saltada (guard: interacción temprana, baja comprensión, o respuesta corta).", fase)
            topics = []
        else:
            topics = await _extract_image_topics(content)

        if topics:
            # ✅ OPTIMIZACIÓN: las imágenes se generan en background, no bloqueamos la respuesta
            job_id = await create_job(images_pending=len(topics))
            conversation_id = state.get("conversation_id", "") if state else ""
            asyncio.ensure_future(_generate_images_background(job_id, topics, conversation_id))
            images_job_id = job_id
            images_pending = len(topics)
            logger.info("[%s] Imágenes lanzadas en background — job_id=%s, pending=%s",
                        fase, job_id, images_pending)
    else:
        logger.info("[%s] Extracción de temas visuales saltada (solo se ejecuta en generativa).", fase)

    # Detectar sugerencias de recursos multimedia
    suggested_resources = []
    if state:
        suggested_resources = _detect_resource_suggestions(state, fase)

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": fase,
        "image_urls": [],          # vacío: las URLs llegan async vía job polling
        "images_job_id": images_job_id,
        "images_pending": images_pending,
        "suggested_resources": suggested_resources,
    }


# ---------------------------------------------------------------------------
# Helper: construcción del prompt optimizado para Gemini Implicit Caching
# ---------------------------------------------------------------------------

def build_response_messages(state: GraphState) -> list[BaseMessage]:
    """Construye la lista de mensajes para el LLM con orden óptimo para caching.

    Orden:
      1. pedagogical_context  — estático, máxima reutilización en caché
      2. system_prompt + protocolo de fase — semi-estático
      3. Perfil del alumno + estado de sesión + RAG — dinámico
      4. Historial de conversación (últimos 15 mensajes)
    """
    profile = state.get("starter_profile") or {}
    profile_data = profile.get("profile_data") or {}
    if isinstance(profile_data, str):
        try:
            profile_data = json.loads(profile_data)
        except Exception:
            profile_data = {}

    session = state.get("session_state") or {}
    insights = state.get("learner_insights") or []
    gk = state.get("gatekeeper_eval") or {}

    # 1. Bloque estático (para implicit caching)
    static = state.get("pedagogical_context") or ""

    # 2. Bloque semi-estático
    semi_static = (
        f"{state.get('system_prompt', '')}\n\n"
        f"--- PROTOCOLO DE FASE: {state.get('fase_actual', 'generativa')} ---\n"
        f"{state.get('protocol_content', '')}"
    )

    # 3. Bloque dinámico
    if insights:
        insights_lines = [
            f"- {i.get('insight_type', '')}: {i.get('insight_value', '')}"  
            for i in insights[:5]
        ]
        insights_text = "\n".join(insights_lines)
    else:
        insights_text = "Primera sesión, sin observaciones previas."

    dynamic = (
        f"--- PERFIL DEL ALUMNO ---\n"
        f"Descripción: {profile_data.get('description', '')}\n"
        f"Edad: {profile.get('age', 'desconocida')}\n"
        f"Intereses: {', '.join(profile_data.get('interests', []))}\n"
        f"Dato único: {profile_data.get('uniqueData', '')}\n"
        f"Estilo de aprendizaje: {', '.join(profile_data.get('learningStyle', []))}\n"
        f"Preferencia de explicación: {', '.join(profile_data.get('explanationStyle', []))}\n\n"
        f"--- OBSERVACIONES DE SESIONES ANTERIORES ---\n"
        f"{insights_text}\n\n"
        f"--- ESTADO DE ESTA SESIÓN ---\n"
        f"Fase: {session.get('session_phase', 'generativa')}\n"
        f"Interacción #{session.get('interaction_count', 0) + 1}\n"
        f"Comprensión actual: {gk.get('comprension_score', 'sin evaluar')}\n"
        f"Engagement: {gk.get('engagement_score', 'sin evaluar')}\n"
        f"Tema actual: {session.get('current_topic', 'por definir')}\n"
        f"Temas cubiertos en esta sesión: {', '.join(session.get('topics_covered', [])) or 'ninguno'}\n"
    )

    rag = state.get("rag_context") or ""
    if rag:
        dynamic += f"\n--- CONTENIDO EDUCATIVO RELEVANTE ---\n{rag}\n"

    system_content = (
        static
        + "\n\n===== FIN DOCUMENTACIÓN PEDAGÓGICA =====\n\n"
        + semi_static
        + "\n\n===== CONTEXTO DE ESTA INTERACCIÓN =====\n\n"
        + dynamic
    )

    # Historial limitado a los últimos 20 mensajes
    history = list(state.get("messages") or [])[-20:]
    return [SystemMessage(content=system_content)] + history


# ---------------------------------------------------------------------------
# Nodos de respuesta
# ---------------------------------------------------------------------------

async def node_generativo(state: GraphState) -> dict:
    """Genera respuesta educativa. Fase: generativa."""
    logger.info("[node_generativo] Generando respuesta...")
    full_messages = build_response_messages(state)
    return await _invoke_with_images(_get_llm(), full_messages, "generativa", state)


async def node_vicario(state: GraphState) -> dict:
    """Modo empatía / pensamiento en voz alta. Fase: vicario."""
    logger.info("[node_vicario] Generando respuesta vicario...")
    full_messages = build_response_messages(state)
    return await _invoke_with_images(_get_llm(), full_messages, "vicario", state)


async def node_socratico(state: GraphState) -> dict:
    """Preguntas de pensamiento crítico. Fase: socratico."""
    logger.info("[node_socratico] Generando preguntas socráticas...")
    full_messages = build_response_messages(state)
    return await _invoke_with_images(_get_llm(), full_messages, "socratico", state)


async def node_metacognicion(state: GraphState) -> dict:
    """Cierre de sesión con reflexión metacognitiva."""
    logger.info("[node_metacognicion] Ejecutando nodo de metacognición...")
    full_messages = build_response_messages(state)
    return await _invoke_with_images(_get_llm(), full_messages, "metacognicion", state)


# ---------------------------------------------------------------------------
# Nodo supervisor: routing determinístico + carga de protocolo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Umbrales del supervisor pedagógico
# Ajustar estos valores con datos reales de alumnos.
# Cambiar un valor = 1 línea + deploy. Sin tablas, sin queries, sin LLM.
# ---------------------------------------------------------------------------
SUPERVISOR_THRESHOLDS = {
    # Generativa → Socrático
    "comprension_para_socratico": 70,       # Promedio mínimo de comprensión (0-100)
    "mensajes_para_promediar": 3,           # Cuántos últimos scores promediar
    "interacciones_minimas_generativa": 5,  # Mínimo de interacciones antes de poder pasar

    # Trigger y salida de Vicario
    "frustracion_trigger_vicario": 6,       # Nivel de frustración (0-10) para activar vicario
    "frustracion_salida_vicario": 3,        # Nivel de frustración (0-10) para salir de vicario

    # Socrático → Metacognición
    "socratico_correctas_para_meta": 3,     # Respuestas correctas en socrático para avanzar
    "socratico_comprension_correcta": 60,   # Score mínimo (0-100) para contar respuesta como correcta

    # Recalibración (vuelta a generativa)
    "comprension_recalibracion": 40,        # Comprensión por debajo de esto → recalibrar
    "frustracion_recalibracion": 7,         # Frustración por encima de esto → recalibrar

    # Metacognición
    "rubrica_minima_aprobacion": 50,        # Promedio de rúbrica por debajo → recalibrar
}


async def supervisor_decide(state: GraphState) -> dict:
    """Nodo determinístico que decide la siguiente fase pedagógica.
    También carga el protocolo correcto para la fase decidida.

    Los umbrales viven en SUPERVISOR_THRESHOLDS (inicio del archivo).
    """
    T = SUPERVISOR_THRESHOLDS  # alias corto para legibilidad
    session = dict(state.get("session_state") or {})
    gk = state.get("gatekeeper_eval") or {}
    current_phase = session.get("session_phase", "generativa")
    is_recalibration = False

    frustracion = gk.get("frustracion_nivel", 0)
    comprension = gk.get("comprension_score", 50)

    # Regla 1: Frustración alta → vicario (en cualquier fase)
    if frustracion >= T["frustracion_trigger_vicario"]:
        next_phase = "vicario"

    # Regla 2: Vicario + frustración resuelta → volver a generativa
    elif current_phase == "vicario" and frustracion <= T["frustracion_salida_vicario"]:
        next_phase = "generativa"

    # Regla 3: Override manual → socrático
    elif session.get("gatekeeper_override", False):
        next_phase = "socratico"

    # Regla 4: Generativa + comprensión alta + suficientes interacciones → socrático
    elif current_phase == "generativa":
        history = session.get("comprehension_history", [])
        n = T["mensajes_para_promediar"]
        ultimos = history[-n:] if history else []
        promedio = sum(ultimos) / len(ultimos) if ultimos else 0
        if (promedio >= T["comprension_para_socratico"]
                and session.get("interaction_count", 0) >= T["interacciones_minimas_generativa"]):
            next_phase = "socratico"
        else:
            next_phase = "generativa"

    # Regla 5: Socrático → metacognición o recalibración
    elif current_phase == "socratico":
        if session.get("socratic_correct_answers", 0) >= T["socratico_correctas_para_meta"]:
            next_phase = "metacognicion"
        elif comprension < T["comprension_recalibracion"] or frustracion >= T["frustracion_recalibracion"]:
            next_phase = "generativa"
            is_recalibration = True
        else:
            next_phase = "socratico"

    # Regla 6: Metacognición con rúbrica baja → recalibración
    elif current_phase == "metacognicion":
        rubric = session.get("rubric_scores", {})
        if rubric:
            promedio_rubric = sum(rubric.values()) / len(rubric)
            if promedio_rubric < T["rubrica_minima_aprobacion"]:
                next_phase = "generativa"
                is_recalibration = True
            else:
                next_phase = "metacognicion"
        else:
            next_phase = "metacognicion"

    # Default: mantener fase actual
    else:
        next_phase = current_phase

    logger.info(
        "[supervisor] Fase: %s → %s | recalibracion=%s",
        current_phase, next_phase, is_recalibration,
    )

    # Cargar el protocolo correcto para la fase decidida.
    # Si hay recalibración, usar el protocolo especial "recalibracion".
    protocol_phase = "recalibracion" if is_recalibration else next_phase
    client = SupabaseClient()
    try:
        protocol_content = await client.get_active_protocol(protocol_phase)
    except Exception as e:
        logger.warning("[supervisor] Error cargando protocolo '%s': %s", protocol_phase, e)
        protocol_content = None
    finally:
        await client.close()

    # Registrar transición si cambió de fase
    if next_phase != current_phase:
        transitions = list(session.get("phase_transitions", []))
        transitions.append({
            "from": current_phase,
            "to": next_phase,
            "interaction": session.get("interaction_count", 0),
            "reason": gk.get("recomendacion", "supervisor_rule"),
            "is_recalibration": is_recalibration,
        })
        session["phase_transitions"] = transitions

        # Reset contadores socrático al SALIR de socrático
        if current_phase == "socratico" and next_phase != "socratico":
            session["socratic_questions_answered"] = 0
            session["socratic_correct_answers"] = 0
            logger.info("[supervisor] Reset contadores socrático (salió de fase socrática).")

    session["session_phase"] = next_phase

    return {
        "fase_actual": next_phase,
        "session_state": session,
        "protocol_content": protocol_content or "",
    }


# ---------------------------------------------------------------------------
# Nodo de persistencia: guarda estado al final de cada turno
# ---------------------------------------------------------------------------

async def node_persist(state: GraphState) -> dict:
    """Persiste el estado de la sesión y la evaluación del gatekeeper.

    Se ejecuta DESPUÉS de cada nodo de respuesta.
    Nunca rompe el chat — cualquier error de persistencia se loggea y se ignora.
    """
    session = dict(state.get("session_state") or {})
    gk = state.get("gatekeeper_eval") or {}

    # Actualizar contadores
    session["interaction_count"] = session.get("interaction_count", 0) + 1
    session["last_interaction_time"] = datetime.now(timezone.utc).isoformat()
    session["session_phase"] = state.get("fase_actual", "generativa")

    # Historiales de comprensión (últimos 20)
    if gk.get("comprension_score") is not None:
        h = list(session.get("comprehension_history", []))
        h.append(gk["comprension_score"])
        session["comprehension_history"] = h[-20:]
        session["current_comprehension"] = gk["comprension_score"]

    # Historial de frustración (últimos 20)
    if gk.get("frustracion_nivel") is not None:
        h = list(session.get("frustration_history", []))
        h.append(gk["frustracion_nivel"])
        session["frustration_history"] = h[-20:]

    # Historial de engagement (últimos 20)
    if gk.get("engagement_score") is not None:
        h = list(session.get("engagement_history", []))
        h.append(gk["engagement_score"])
        session["engagement_history"] = h[-20:]

    # Misconceptions acumulados (últimos 10 únicos)
    if gk.get("misconceptions"):
        existing = list(session.get("misconceptions", []))
        for m in gk["misconceptions"]:
            if m not in existing:
                existing.append(m)
        session["misconceptions"] = existing[-10:]

    # Actualizar tema actual si el gatekeeper detectó uno
    detected_topic = gk.get("topic", "")
    if detected_topic:
        session["current_topic"] = detected_topic
        # Agregar a topics_covered si es nuevo
        covered = list(session.get("topics_covered", []))
        if detected_topic not in covered:
            covered.append(detected_topic)
            session["topics_covered"] = covered[-20:]  # últimos 20 temas

    # Contador de triggers vicario
    if state.get("fase_actual") == "vicario":
        session["vicario_triggers"] = session.get("vicario_triggers", 0) + 1
        logger.info(
            "[node_persist] Vicario trigger #%s acumulado.",
            session["vicario_triggers"],
        )

    # Calcular zdp_level basado en comprensión y frustración recientes
    comp_h = list(session.get("comprehension_history", []))[-5:]
    frust_h = list(session.get("frustration_history", []))[-5:]
    if comp_h:
        comp_avg = sum(comp_h) / len(comp_h)
        frust_avg = sum(frust_h) / len(frust_h) if frust_h else 0
        session["zdp_level"] = round(max(0.0, min(100.0, comp_avg - (frust_avg * 10))), 1)
        logger.info("[node_persist] ZDP level recalculado: %.1f (comp_avg=%.1f, frust_avg=%.1f)",
                    session["zdp_level"], comp_avg, frust_avg)

    # Contadores socrático: preguntas respondidas y respuestas correctas
    if state.get("fase_actual") == "socratico":
        session["socratic_questions_answered"] = session.get("socratic_questions_answered", 0) + 1
        # Si la comprensión del mensaje es >= 60, contar como respuesta correcta
        # Usamos el umbral configurado por si se ajusta después
        if gk.get("comprension_score", 0) >= SUPERVISOR_THRESHOLDS.get("socratico_comprension_correcta", 60):
            session["socratic_correct_answers"] = session.get("socratic_correct_answers", 0) + 1
            logger.info(
                "[node_persist] Socrático: respuesta correcta #%s (comprensión=%.1f) | respondidas=%s",
                session["socratic_correct_answers"],
                gk.get("comprension_score", 0),
                session["socratic_questions_answered"],
            )
        else:
            logger.info(
                "[node_persist] Socrático: respuesta NO correcta (comprensión=%.1f < %.0f) | respondidas=%s | correctas=%s",
                gk.get("comprension_score", 0),
                SUPERVISOR_THRESHOLDS.get("socratico_comprension_correcta", 60),
                session["socratic_questions_answered"],
                session.get("socratic_correct_answers", 0),
            )

    # Rúbrica de metacognición: usar comprensión y engagement como proxy
    # hasta implementar extracción de rúbrica del texto del LLM
    if state.get("fase_actual") == "metacognicion":
        comp = gk.get("comprension_score", 50)
        eng = gk.get("engagement_score", 50)
        # Proxy: factual y aplicación se basan en comprensión,
        # análisis y síntesis se basan en el promedio de ambos
        promedio = (comp + eng) / 2
        session["rubric_scores"] = {
            "factual": comp,
            "aplicacion": comp,
            "analisis": promedio,
            "sintesis": promedio,
        }
        logger.info(
            "[node_persist] Metacognición: rubric_scores generadas (proxy) — factual=%.0f, aplicacion=%.0f, analisis=%.0f, sintesis=%.0f",
            comp, comp, promedio, promedio,
        )

    client = SupabaseClient()
    try:
        await asyncio.gather(
            client.save_session_state(session),
            client.save_gatekeeper_evaluation({
                "conversation_id": state.get("conversation_id"),
                "user_id": state.get("user_id"),
                "comprehension_score": int(round(gk.get("comprension_score", 0))),
                "frustration_detected": gk.get("frustracion_detectada", False),
                "frustration_level": gk.get("frustracion_nivel", 0),
                "engagement_score": int(round(gk.get("engagement_score", 0))),
                "misconceptions": gk.get("misconceptions", []),
                "recommendation": gk.get("recomendacion", "continuar"),
                "justification": gk.get("justificacion", ""),
                "current_phase": state.get("fase_actual", "generativa"),
                "interaction_number": session["interaction_count"],
                "raw_response": gk,
            }),
            client.update_conversation_phase(
                state.get("conversation_id"),
                state.get("fase_actual", "generativa"),
            ),
        )

        # Escribir learner_insights al cierre de sesión (metacognición)
        if state.get("fase_actual") == "metacognicion":
            try:
                topics = list(session.get("topics_covered", []))
                comp_h = list(session.get("comprehension_history", []))
                avg_comp = sum(comp_h) / len(comp_h) if comp_h else 50

                # Insights de dominio por tema
                for topic in topics[-5:]:
                    await client.upsert_learner_insight({
                        "user_id": state.get("user_id"),
                        "conversation_id": state.get("conversation_id"),
                        "insight_type": "topic_mastery",
                        "insight_key": topic,
                        "insight_value": f"Comprensión promedio: {avg_comp:.0f}%. Temas en sesión: {len(topics)}. Interacciones: {session.get('interaction_count', 0)}.",
                        "confidence": min(avg_comp / 100, 1.0),
                    })

                # Insights de misconceptions
                for mc in list(session.get("misconceptions", []))[-5:]:
                    await client.upsert_learner_insight({
                        "user_id": state.get("user_id"),
                        "conversation_id": state.get("conversation_id"),
                        "insight_type": "misconception",
                        "insight_key": mc[:50],
                        "insight_value": mc,
                        "confidence": 0.7,
                    })

                # Insight de resiliencia (basado en triggers vicario)
                vicario_triggers = session.get("vicario_triggers", 0)
                if vicario_triggers > 0:
                    await client.upsert_learner_insight({
                        "user_id": state.get("user_id"),
                        "conversation_id": state.get("conversation_id"),
                        "insight_type": "emotional_pattern",
                        "insight_key": "frustration_frequency",
                        "insight_value": f"Activaciones vicario en sesión: {vicario_triggers}. ZDP final: {session.get('zdp_level', 50)}.",
                        "confidence": 0.6,
                    })

                logger.info("[node_persist] Learner insights escritos para metacognición — %s temas, %s misconceptions.",
                            len(topics[-5:]), len(list(session.get('misconceptions', []))[-5:]))
            except Exception as e:
                logger.error("[node_persist] Error escribiendo learner_insights: %s", e)

        logger.info(
            "[node_persist] Estado guardado — interacción #%s, fase=%s",
            session["interaction_count"],
            session["session_phase"],
        )
    except Exception as e:
        logger.error("[node_persist] Error persistiendo estado: %s", e)
    finally:
        await client.close()

    return {}


