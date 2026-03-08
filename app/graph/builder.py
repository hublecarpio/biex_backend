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
    node_metacognicion,
    node_persist,
    supervisor_decide,
)
from app.graph.edges import evaluate_gatekeeper


def build_graph():
    """
    Construye y compila el grafo BIEX.

    Flujo:
      node_setup → evaluate_gatekeeper → supervisor_decide
        → [generativo | vicario | socratico | metacognicion]
        → node_persist → END

    Usa MemorySaver in-process por request; el estado real se lee y escribe
    desde las tablas nativas de Supabase (session_state, gatekeeper_evaluations, etc.).
    """
    workflow = StateGraph(GraphState)

    # Nodos
    workflow.add_node("node_setup", node_setup)
    workflow.add_node("evaluate_gatekeeper", evaluate_gatekeeper)
    workflow.add_node("supervisor_decide", supervisor_decide)
    workflow.add_node("node_generativo", node_generativo)
    workflow.add_node("node_vicario", node_vicario)
    workflow.add_node("node_socratico", node_socratico)
    workflow.add_node("node_metacognicion", node_metacognicion)
    workflow.add_node("node_persist", node_persist)

    # Flujo lineal: setup → gatekeeper → supervisor
    workflow.set_entry_point("node_setup")
    workflow.add_edge("node_setup", "evaluate_gatekeeper")
    workflow.add_edge("evaluate_gatekeeper", "supervisor_decide")

    # Routing determinístico desde supervisor según fase_actual
    workflow.add_conditional_edges(
        "supervisor_decide",
        lambda state: state["fase_actual"],
        {
            "generativa": "node_generativo",
            "vicario": "node_vicario",
            "socratico": "node_socratico",
            "metacognicion": "node_metacognicion",
        },
    )

    # Todos los nodos de respuesta → persist → end
    workflow.add_edge("node_generativo", "node_persist")
    workflow.add_edge("node_vicario", "node_persist")
    workflow.add_edge("node_socratico", "node_persist")
    workflow.add_edge("node_metacognicion", "node_persist")
    workflow.add_edge("node_persist", "__end__")

    return workflow.compile(checkpointer=MemorySaver())
