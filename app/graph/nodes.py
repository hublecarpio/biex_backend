"""
Nodos del grafo BIEX: setup, generativo, vicario, socrático, metacognición.
"""
import json
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.services.supabase_api import SupabaseClient


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


def _build_system_content(system_prompt: str, starter_profile: dict) -> str:
    """Construye el contenido del system message con prompt y perfil."""
    profile_str = json.dumps(starter_profile, ensure_ascii=False, indent=2)
    return f"""{system_prompt}

## Perfil del alumno (usa esto para personalizar)
{profile_str}
"""


async def node_setup(state: GraphState) -> dict:
    """
    Entry point: obtiene perfil, prompt maestro y contexto RAG.
    Guarda todo en el state.
    """
    messages: list = state.get("messages") or []
    user_id: str = state.get("user_id") or ""

    if not messages:
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

    client = SupabaseClient()
    try:
        system_prompt = await client.get_system_prompt()
        starter_profile_obj = await client.get_starter_profile(user_id)
        rag_context = await client.query_knowledge(user_text)
    finally:
        await client.close()

    starter_profile = starter_profile_obj.model_dump() if starter_profile_obj else {}

    return {
        "starter_profile": starter_profile,
        "system_prompt": system_prompt or "Eres un tutor educativo amable y claro.",
        "rag_context": rag_context or "",
    }


async def node_generativo(state: GraphState) -> dict:
    """
    Genera respuesta educativa usando rag_context.
    Actualiza fase_actual a 'generativa'.
    """
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}
    rag_context = state.get("rag_context") or ""

    system_content = _build_system_content(system_prompt, starter_profile)
    if rag_context:
        system_content += f"\n\n## Contexto de conocimiento (úsalo para fundamentar tu respuesta)\n{rag_context}"

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    response = await llm.ainvoke(full_messages)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": "generativa",
    }


async def node_vicario(state: GraphState) -> dict:
    """
    Modo empatía / pensamiento en voz alta. Usa el perfil, no RAG duro.
    Actualiza fase_actual a 'vicaria'.
    """
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += "\n\nInstrucción: Responde en modo vicario: muestra empatía, piensa en voz alta y acompaña al alumno sin dar la respuesta directa. No uses el contexto RAG de forma rígida; prioriza el estado emocional y el perfil del alumno."

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    response = await llm.ainvoke(full_messages)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": "vicaria",
    }


async def node_socratico(state: GraphState) -> dict:
    """
    Solo hace preguntas de pensamiento crítico.
    Actualiza fase_actual a 'socratica'.
    """
    messages = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    starter_profile = state.get("starter_profile") or {}

    system_content = _build_system_content(system_prompt, starter_profile)
    system_content += "\n\nInstrucción: Responde únicamente con una o dos preguntas socráticas para guiar el pensamiento crítico del alumno. No des explicaciones largas ni la respuesta; solo preguntas que le hagan reflexionar."

    llm = _get_llm()
    full_messages: list[BaseMessage] = [SystemMessage(content=system_content)] + list(messages)
    response = await llm.ainvoke(full_messages)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=content)],
        "fase_actual": "socratica",
    }


async def node_metacognicion(state: GraphState) -> dict:
    """
    Evalúa la sesión al final (nodo de cierre).
    Por ahora solo actualiza fase; luego se puede implementar en background.
    """
    return {"fase_actual": "metacognicion"}
