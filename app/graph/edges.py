"""
Lógica del Gatekeeper y enrutado condicional.
"""
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.graph.state import GraphState
from app.models.schemas import GatekeeperEval

# Umbral de comprensión para ir a socrático
COMPRENSION_THRESHOLD = 85.0


def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


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
        return "node_generativo"

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return "node_generativo"

    user_content = last.content
    text = user_content if isinstance(user_content, str) else str(user_content)
    if not text.strip():
        return "node_generativo"

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
    except Exception:
        return "node_generativo"

    if eval_result.comprension_score >= COMPRENSION_THRESHOLD:
        return "node_socratico"
    if eval_result.frustracion_detectada:
        return "node_vicario"
    return "node_generativo"
