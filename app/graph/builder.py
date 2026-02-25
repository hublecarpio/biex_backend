"""
Ensamblaje y compilación del StateGraph de BIEX.
"""
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import GraphState
from app.graph.nodes import (
    node_setup,
    node_generativo,
    node_vicario,
    node_socratico,
)
from app.graph.edges import evaluate_gatekeeper


def build_graph(checkpointer=None):
    """
    Construye y compila el grafo.
    Si se pasa checkpointer (p. ej. MemorySaver()), el grafo soporta checkpointing.
    """
    builder = StateGraph(GraphState)

    # Nodos
    builder.add_node("node_setup", node_setup)
    builder.add_node("node_generativo", node_generativo)
    builder.add_node("node_vicario", node_vicario)
    builder.add_node("node_socratico", node_socratico)

    # Entrada
    builder.set_entry_point("node_setup")

    # node_setup -> gatekeeper (condicional)
    builder.add_conditional_edges("node_setup", evaluate_gatekeeper)

    # Rutas del gatekeeper hacia cada nodo de respuesta
    builder.add_edge("node_generativo", "__end__")
    builder.add_edge("node_vicario", "__end__")
    builder.add_edge("node_socratico", "__end__")

    # Compilar (opcional: checkpointer para persistir estado por sesión)
    memory = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=memory)
