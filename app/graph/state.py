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
    conversation_id: str
    session_state: dict
    protocol_content: str
    gatekeeper_eval: dict
    pedagogical_context: str
    learner_insights: list
    starter_profile: dict
    system_prompt: str
    fase_actual: str
    comprension_score: float
    frustracion_detectada: bool
    rag_context: str
    image_urls: list        # URLs de imágenes (vacío si se procesan en background)
    images_job_id: str      # ID del job de generación de imágenes en background
    images_pending: int     # Cantidad de imágenes siendo generadas (0 si ya están listas)
    suggested_resources: list
