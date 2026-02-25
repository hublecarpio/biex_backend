"""
BIEX Backend - Agente Tutor Cognitivo.
FastAPI + LangGraph + streaming SSE.
"""
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.graph.builder import build_graph
from app.graph.state import GraphState
from app.models.schemas import ChatRequest, ChatResponse


# --- Dependencia API Key ---
def verify_api_key(request: Request) -> None:
    key = request.headers.get("X-API-Key")
    settings = get_settings()
    if not settings.api_key_secret or key != settings.api_key_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --- App ---
app = FastAPI(
    title="BIEX API",
    description="Agente Tutor Cognitivo - Backend",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check sin autenticación."""
    return {"status": "ok"}


@app.post(
    "/api/v1/chat",
    dependencies=[Depends(verify_api_key)],
)
async def chat(request: ChatRequest):
    """
    Chat con el tutor. Si stream=False devuelve ChatResponse.
    Si stream=True devuelve StreamingResponse (SSE) con los tokens.
    """
    initial_state: GraphState = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
    }
    graph = build_graph()

    if not request.stream:
        result = await graph.ainvoke(initial_state)
        messages = result.get("messages") or []
        fase = result.get("fase_actual") or "generativa"
        response_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                response_text = m.content if isinstance(m.content, str) else str(m.content)
                break
        return ChatResponse(response=response_text, current_phase=fase)

    # Streaming: SSE con tokens
    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in graph.astream_events(
            initial_state,
            version="v2",
            include_types=["on_chat_model_stream"],
        ):
            kind = event.get("event")
            if kind != "on_chat_model_stream":
                continue
            data = event.get("data", {})
            if not isinstance(data, dict):
                continue
            chunk = data.get("chunk")
            if chunk is None:
                continue
            content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", "") or ""
            if isinstance(content, str) and content:
                yield f"data: {json.dumps({'token': content})}\n\n"
        # Envío de fase al final (opcional; el cliente puede obtenerla del último mensaje)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return EventSourceResponse(event_generator())
