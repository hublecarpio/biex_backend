"""
Lógica del Gatekeeper y enrutado condicional.
"""
import logging

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import GatekeeperEval

logger = logging.getLogger(__name__)

# Umbral de comprensión para ir a socrático
COMPRENSION_THRESHOLD = 85.0


def _get_llm() -> ChatGoogleGenerativeAI:
    """Instancia el LLM Gemini con fallback: gemini-3-flash-preview -> gemini-2.5-flash."""
    settings = get_settings()
    primary = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    return primary.with_fallbacks([fallback])


async def evaluate_gatekeeper(state: GraphState) -> str:
    """
    Evalúa el último mensaje del alumno con LLM estructurado (GatekeeperEval).
    Retorna el nombre del nodo al que enrutar:
    - "node_socratico" si comprension_score >= 85
    - "node_vicario" si frustracion_detectada
    - "node_generativo" en otro caso
    """
    messages = state.get("messages") or []
    if not messages:
        logger.info("[gatekeeper] Sin mensajes, enrutando a node_generativo por defecto.")
        return "node_generativo"

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        logger.info("[gatekeeper] Último mensaje no es HumanMessage, enrutando a node_generativo.")
        return "node_generativo"

    user_content = last.content
    text = user_content if isinstance(user_content, str) else str(user_content)
    if not text.strip():
        logger.info("[gatekeeper] Mensaje vacío, enrutando a node_generativo.")
        return "node_generativo"

    logger.info("[gatekeeper] Evaluando mensaje del alumno (%s chars)...", len(text))

    llm = _get_llm()
    structured_llm = llm.with_structured_output(GatekeeperEval)

    prompt = (
        "Evalúa el siguiente mensaje de un alumno en una sesión de tutoría. "
        "Indica: (1) comprension_score: puntuación de 0 a 100 según si entendió el tema; "
        "(2) frustracion_detectada: true si detectas frustración, confusión o malestar.\n\n"
        f"Mensaje del alumno:\n{text}"
    )
    try:
        eval_result: GatekeeperEval = await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.warning("[gatekeeper] Error evaluando con LLM (%s), enrutando a node_generativo.", e)
        return "node_generativo"

    logger.info(
        "[gatekeeper] Evaluación: comprensión=%.1f | frustración=%s",
        eval_result.comprension_score,
        eval_result.frustracion_detectada,
    )

    if eval_result.comprension_score >= COMPRENSION_THRESHOLD:
        logger.info("[gatekeeper] -> node_socratico (comprensión alta)")
        return "node_socratico"
    if eval_result.frustracion_detectada:
        logger.info("[gatekeeper] -> node_vicario (frustración detectada)")
        return "node_vicario"

    logger.info("[gatekeeper] -> node_generativo")
    return "node_generativo"
