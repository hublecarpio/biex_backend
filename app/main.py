"""
BIEX Backend - Agente Tutor Cognitivo.
FastAPI + LangGraph + streaming SSE + memoria Postgres.

Endpoints:
 - POST /api/v1/chat          → Respuesta del tutor (streaming o no).
 - GET  /api/v1/images/{job_id} → Polling de imágenes generadas en background.
 - GET  /health               → Health check.
"""
import json
import logging
import traceback
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.graph.builder import build_graph
from app.graph.nodes import _extract_text
from app.graph.state import GraphState
from app.models.schemas import ChatRequest, ChatResponseStructured, ImageJobResponse
from app.services.image_jobs import get_job
from app.services.supabase_api import SupabaseClient
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


# --- Lifespan: Inicialización de Grafo ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[lifespan] Inicializando grafo BIEX (Memoria in-process)...")
    app.state.graph = build_graph()
    logger.info("[lifespan] Grafo compilado. App lista.")
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


# ---------------------------------------------------------------------------
# Endpoint de polling para imágenes generadas en background
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/images/{job_id}",
    response_model=ImageJobResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["images"],
    summary="Polling del estado de generación de imágenes",
    description=(
        "Consulta el estado de un job de generación de imágenes iniciado en background. "
        "El frontend debe hacer polling hasta que `status == 'done'`. "
        "Cuando `status == 'done'`, `urls` contiene las URLs finales de las imágenes."
    ),
)
async def get_image_job(job_id: str):
    """
    GET /api/v1/images/{job_id}

    Respuesta posible:
    - status: "pending" → imágenes aún siendo generadas, seguir haciendo polling.
    - status: "done"    → `urls` contiene las URLs listas. Detener polling.
    - status: "error"   → falló la generación. `error` contiene el motivo.
    - 404               → job_id inválido o expirado (TTL de 5 minutos).
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' no encontrado o expirado. TTL máximo: 5 minutos."
        )
    return ImageJobResponse(
        job_id=job.job_id,
        status=job.status,
        images_pending=job.images_pending if job.status == "pending" else 0,
        urls=job.urls,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# Endpoint principal de chat
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/chat",
    response_model=ChatResponseStructured,
    dependencies=[Depends(verify_api_key)],
    tags=["chat"],
    summary="Chat con el tutor cognitivo",
    description=(
        "Envía un mensaje al agente tutor. "
        "El historial se recupera automáticamente desde Supabase (tabla `messages`)."
    ),
)
async def chat(http_request: Request, body: ChatRequest):
    t_start = time.perf_counter()
    logger.info(
        "[chat] ═══ NUEVO TURNO ══════════════════════════════════════════"
    )
    logger.info(
        "[chat] Request recibido — user_id=%s | conversation_id=%s | stream=%s | mensaje=%s chars",
        body.id_user,
        body.id_conversation,
        body.stream,
        len(body.mensaje),
    )

    # Cargamos el historial de la conversación para dar contexto al grafo.
    # El guardado en BD lo maneja la Edge Function de Supabase, no este backend.
    supabase = SupabaseClient()
    try:
        raw_history = await supabase.get_conversation_history(body.id_conversation)
    finally:
        await supabase.close()

    messages = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("message", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Limitar historial a los últimos 19 mensajes + el mensaje nuevo = 20 total
    messages = messages[-19:]
    messages.append(HumanMessage(content=body.mensaje))
    logger.info(
        "[chat] Historial cargado — %s msgs de DB, total con nuevo: %s",
        len(raw_history),
        len(messages),
    )

    initial_state: GraphState = {
        "messages": messages,
        "user_id": body.id_user,
        "conversation_id": body.id_conversation,
    }
    
    # Thread_id local solo para memoria MemorySaver en runtime (opcional)
    config = {"configurable": {"thread_id": body.id_conversation}}
    graph = http_request.app.state.graph

    if not body.stream:
        logger.info("[chat] Modo no-streaming, invocando grafo...")
        result = await graph.ainvoke(initial_state, config=config)

        messages = result.get("messages") or []
        fase = result.get("fase_actual") or "generativa"
        image_urls_from_state: list[str] = result.get("image_urls") or []
        images_job_id: str | None = result.get("images_job_id")
        images_pending: int = result.get("images_pending") or 0

        response_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                response_text = _extract_text(m.content)
                break

        parsed = parse_response_to_structured(response_text)

        all_images = list(dict.fromkeys(image_urls_from_state + parsed["images"]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[chat] Respuesta lista en %.2fs — fase=%s | segmentos=%s | imágenes=%s | pending=%s | job_id=%s",
            elapsed,
            fase,
            len(parsed["mensajes"]),
            len(all_images),
            images_pending,
            images_job_id,
        )
        return ChatResponseStructured(
            mensajes=parsed["mensajes"],
            images=all_images,
            images_count=len(all_images) + images_pending,
            images_pending=images_pending,
            images_job_id=images_job_id,
            current_phase=fase,
        )

    # --- Modo Streaming SSE ---
    logger.info("[chat] Modo streaming SSE iniciado.")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_text: list[str] = []
        images_job_id: str | None = None
        images_pending: int = 0
        image_urls_from_state: list[str] = []
        fase = "generativa"

        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                data = event.get("data", {})
                if not isinstance(data, dict):
                    continue
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                content = (
                    chunk.get("content", "") if isinstance(chunk, dict)
                    else getattr(chunk, "content", "") or ""
                )
                extracted = _extract_text(content)
                if extracted:
                    full_text.append(extracted)
                    yield f"data: {json.dumps({'token': extracted})}\n\n"

            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    job_id = output.get("images_job_id")
                    if job_id:
                        images_job_id = job_id
                        images_pending = output.get("images_pending", 0)
                    urls = output.get("image_urls") or []
                    if urls:
                        image_urls_from_state = urls
                    
                    if "fase_actual" in output:
                        fase = output["fase_actual"]

        full_response = "".join(full_text)
        parsed = parse_response_to_structured(full_response)
        all_images = list(dict.fromkeys(image_urls_from_state + parsed["images"]))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[chat] Stream finalizado en %.2fs — %s chars | pending=%s | job_id=%s",
            elapsed, len(full_response), images_pending, images_job_id,
        )

        yield f"data: {json.dumps({'done': True, 'mensajes': parsed['mensajes'], 'images': all_images, 'images_count': len(all_images) + images_pending, 'images_pending': images_pending, 'images_job_id': images_job_id})}\n\n"

    return EventSourceResponse(event_generator())
