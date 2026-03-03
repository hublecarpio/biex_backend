"""
BIEX Backend - Agente Tutor Cognitivo.
FastAPI + LangGraph + streaming SSE + memoria Postgres.
"""
import json
import logging
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.graph.builder import build_graph
from app.graph.state import GraphState
from app.models.schemas import ChatRequest, ChatResponseStructured
from app.utils.response_parser import parse_response_to_structured

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Dependencia API Key ---
def verify_api_key(request: Request) -> None:
    key = request.headers.get("X-API-Key")
    settings = get_settings()
    if not settings.api_key_secret or key != settings.api_key_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --- Exception Handler para debug ---
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = str(exc)
    tb = traceback.format_exc()
    logger.error("Excepción no manejada en %s: %s\n%s", request.url.path, error_msg, tb)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_message": error_msg, "traceback": tb.splitlines()}
    )


# --- Lifespan: Postgres checkpointer + grafo ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.database_url:
        logger.info("[lifespan] Conectando a Postgres para checkpointer LangGraph...")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            await checkpointer.setup()
            app.state.graph = build_graph(checkpointer)
            logger.info("[lifespan] Grafo compilado con checkpointer Postgres. App lista.")
            yield
    else:
        logger.info("[lifespan] DATABASE_URL no configurada, usando MemorySaver in-process.")
        app.state.graph = build_graph(None)
        logger.info("[lifespan] Grafo compilado con MemorySaver. App lista.")
        yield


# --- App ---
app = FastAPI(
    title="BIEX API",
    description="Agente Tutor Cognitivo - Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(Exception, global_exception_handler)


@app.get("/health")
async def health():
    """Health check sin autenticación."""
    return {"status": "ok"}


@app.post(
    "/api/v1/chat",
    dependencies=[Depends(verify_api_key)],
)
async def chat(http_request: Request, body: ChatRequest):
    """
    Chat con el tutor. Payload: mensaje, id_conversation, id_user, tipo_respuesta, stream.
    id_conversation = thread_id para memoria Postgres. Respuesta: mensajes, images, images_count, current_phase.
    """
    logger.info(
        "[chat] Request recibido — user_id=%s | conversation_id=%s | stream=%s | mensaje=%s chars",
        body.id_user,
        body.id_conversation,
        body.stream,
        len(body.mensaje),
    )

    initial_state: GraphState = {
        "messages": [HumanMessage(content=body.mensaje)],
        "user_id": body.id_user,
    }
    config = {"configurable": {"thread_id": body.id_conversation}}
    graph = http_request.app.state.graph

    if not body.stream:
        logger.info("[chat] Modo no-streaming, invocando grafo...")
        result = await graph.ainvoke(initial_state, config=config)
        messages = result.get("messages") or []
        fase = result.get("fase_actual") or "generativa"
        response_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                response_text = m.content if isinstance(m.content, str) else str(m.content)
                break
        parsed = parse_response_to_structured(response_text)
        logger.info(
            "[chat] Respuesta lista — fase=%s | segmentos=%s | imágenes=%s",
            fase,
            len(parsed["mensajes"]),
            parsed["images_count"],
        )
        return ChatResponseStructured(
            mensajes=parsed["mensajes"],
            images=parsed["images"],
            images_count=parsed["images_count"],
            current_phase=fase,
        )

    # Streaming: SSE con tokens; al final un evento con la respuesta estructurada
    logger.info("[chat] Modo streaming SSE iniciado.")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_text: list[str] = []
        async for event in graph.astream_events(
            initial_state,
            config=config,
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
                full_text.append(content)
                yield f"data: {json.dumps({'token': content})}\n\n"
        # Evento final con la respuesta estructurada
        parsed = parse_response_to_structured("".join(full_text))
        logger.info("[chat] Stream finalizado — total tokens acumulados: %s chars.", len("".join(full_text)))
        yield f"data: {json.dumps({'done': True, 'mensajes': parsed['mensajes'], 'images': parsed['images'], 'images_count': parsed['images_count']})}\n\n"

    return EventSourceResponse(event_generator())
