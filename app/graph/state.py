"""
Estado del grafo LangGraph para BIEX.
"""
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Estado compartido del grafo del tutor cognitivo."""

    messages: Annotated[list, add_messages]
    user_id: str
    starter_profile: dict
    system_prompt: str
    fase_actual: str
    comprension_score: float
    frustracion_detectada: bool
    rag_context: str
