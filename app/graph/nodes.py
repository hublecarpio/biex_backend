"""
Nodos del grafo BIEX: setup, generativo, vicario, socrático, metacognición.
"""
import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import ImageTopicList
from app.services.supabase_api import SupabaseClient
from app.services.image_service import generate_images

logger = logging.getLogger(__name__)


def _get_llm() -> ChatGoogleGenerativeAI:
    """Instancia el LLM Gemini con fallback: gemini-3-flash-preview -> gemini-2.5-flash."""
    settings = get_settings()
    primary = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=settings.gemini_api_key,
        temperature=0.7,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.7,
    )
    return primary.with_fallbacks([fallback])


def _extract_text(content) -> str:
    """
    Extrae texto plano del contenido de la respuesta del LLM.
    Maneja tanto strings simples como listas de content blocks (type/text/extras).
    """
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


async def _extract_image_topics(response_text: str) -> list[dict]:
    """
    Usa el LLM para extraer temas visuales de la respuesta del tutor.
    Retorna lista de dicts {"tema": ..., "descripcion": ...} o vacía si no aplica.
    """
    if not response_text.strip():
        return []

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # siempre el modelo estable para este paso auxiliar
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(ImageTopicList)

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
            logger.info("[image_topics] %s tema(s) visuales detectados: %s", len(topics), [t["tema"] for t in topics])
        else:
            logger.info("[image_topics] No se detectaron temas visuales.")
        return topics
    except Exception as e:
        logger.warning("[image_topics] Error extrayendo temas: %s", e)
        return []


async def node_setup(state: GraphState) -> dict:
    """
    Entry point: obtiene perfil, prompt maestro y contexto RAG.
    Guarda todo en el state.
    """
    messages: list = state.get("messages") or []
    user_id: str = state.get("user_id") or ""

    logger.info("[node_setup] Iniciando para user_id=%s", user_id)

    if not messages:
        logger.warning("[node_setup] No hay mensajes en el estado, retornando vacío.")
        return {
            "starter_profile": {},
            "system_prompt": "",
            "rag_context": "",
        }

    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        user_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    else:
        user_text = ""

    logger.info("[node_setup] Consultando Supabase (system_prompt, starter_profile, RAG)...")
    client = SupabaseClient()
    try:
        system_prompt = await client.get_system_prompt()
        starter_profile_obj = await client.get_starter_profile(user_id)
        rag_context = await client.query_knowledge(user_text)
    finally:
        await client.close()

    starter_profile = starter_profile_obj.model_dump() if starter_profile_obj else {}
    logger.info(
        "[node_setup] Supabase OK — system_prompt=%s chars | profile=%s | rag=%s",
        len(system_prompt) if system_prompt else 0,
        "OK" if starter_profile else "vacío",
        "OK" if rag_context else "vacío",
    )

    return {
        "starter_profile": starter_profile,
        "system_prompt": system_prompt or "Eres un tutor educativo amable y claro.",
        "rag_context": rag_context or "",
    }


async def _invoke_with_images(llm, full_messages: list[BaseMessage], fase: str) -> dict:
    """
    Invoca el LLM, extrae el texto limpio y genera imágenes si corresponde.
    Retorna dict con messages, fase_actual, e image_urls (para el state).
    """
    response = await llm.ainvoke(full_messages)
    content = _extract_text(response.content if hasattr(response, "content") else response)
    logger.info("[%s] Respuesta del LLM: %s chars.", fase, len(content))

    topics = await _extract_image_topics(content)
    image_urls = await generate_images(topics) if topics else []

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": fase,
        "image_urls": image_urls,
    }


async def node_generativo(state: GraphState) -> dict:
    """
    Genera respuesta educativa usando rag_context.
    Actualiza fase_actual a 'generativa'.
    """
    logger.info("[node_generativo] Generando respuesta en modo generativo...")
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
    """
    Modo empatía / pensamiento en voz alta. Usa el perfil, no RAG duro.
    Actualiza fase_actual a 'vicaria'.
    """
    logger.info("[node_vicario] Generando respuesta en modo vicario (empatía)...")
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += "\n\nInstrucción: Responde en modo vicario: muestra empatía, piensa en voz alta y acompaña al alumno sin dar la respuesta directa. No uses el contexto RAG de forma rígida; prioriza el estado emocional y el perfil del alumno."

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    return await _invoke_with_images(llm, full_messages, "vicaria")


async def node_socratico(state: GraphState) -> dict:
    """
    Solo hace preguntas de pensamiento crítico.
    Actualiza fase_actual a 'socratica'.
    """
    logger.info("[node_socratico] Generando preguntas socráticas...")
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += "\n\nInstrucción: Responde únicamente con una o dos preguntas socráticas para guiar el pensamiento crítico del alumno. No des explicaciones largas ni la respuesta; solo preguntas que le hagan reflexionar."

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    return await _invoke_with_images(llm, full_messages, "socratica")


async def node_metacognicion(state: GraphState) -> dict:
    """
    Evalúa la sesión al final (nodo de cierre).
    Por ahora solo actualiza fase; luego se puede implementar en background.
    """
    logger.info("[node_metacognicion] Cerrando sesión con fase metacognición.")
    return {"fase_actual": "metacognicion"}
