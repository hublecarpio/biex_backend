"""
Store en memoria para jobs de generación de imágenes en background.

Flujo:
 1. Cuando el nodo detecta que hay imágenes a generar, crea un job con create_job().
 2. La respuesta al cliente incluye images_job_id y images_pending (cantidad).
 3. El frontend hace polling a GET /api/v1/images/{job_id} hasta que status == "done".
 4. El worker background llama a complete_job() con las URLs resultantes.

NOTA: El store es in-process (dict en memoria). Para producción con múltiples workers
se debería usar Redis. Para un worker único (Uvicorn + 1 proceso) esto es suficiente.
Jobs se auto-limpian después de TTL_SECONDS para evitar memory leaks.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Tiempo en segundos antes de que un job completado sea eliminado del store
TTL_SECONDS = 300  # 5 minutos


@dataclass
class ImageJob:
    job_id: str
    status: Literal["pending", "done", "error"]
    images_pending: int  # cantidad esperada de imágenes
    urls: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None


# Store global — un dict simple es thread-safe para lecturas/escrituras atómicas en CPython
_jobs: dict[str, ImageJob] = {}
_lock = asyncio.Lock()


async def create_job(images_pending: int) -> str:
    """Registra un nuevo job y retorna el job_id."""
    job_id = str(uuid.uuid4())
    async with _lock:
        _jobs[job_id] = ImageJob(
            job_id=job_id,
            status="pending",
            images_pending=images_pending,
        )
    logger.info("[image_jobs] Job creado: %s (%s imágenes pendientes)", job_id, images_pending)
    return job_id


async def complete_job(job_id: str, urls: list[str]) -> None:
    """Marca el job como completado con las URLs resultantes."""
    async with _lock:
        job = _jobs.get(job_id)
        if job is None:
            logger.warning("[image_jobs] complete_job: job_id=%s no encontrado.", job_id)
            return
        job.status = "done"
        job.urls = urls
        job.completed_at = time.monotonic()
    logger.info("[image_jobs] Job completado: %s — %s URL(s)", job_id, len(urls))
    _schedule_cleanup(job_id)


async def fail_job(job_id: str, error: str) -> None:
    """Marca el job como fallido."""
    async with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = "error"
        job.error = error
        job.completed_at = time.monotonic()
    logger.warning("[image_jobs] Job fallido: %s — %s", job_id, error)
    _schedule_cleanup(job_id)


def get_job(job_id: str) -> ImageJob | None:
    """Retorna el job o None si no existe / ya fue limpiado."""
    return _jobs.get(job_id)


def _schedule_cleanup(job_id: str) -> None:
    """Programa la limpieza del job después de TTL_SECONDS."""
    async def _cleanup():
        await asyncio.sleep(TTL_SECONDS)
        _jobs.pop(job_id, None)
        logger.debug("[image_jobs] Job %s eliminado del store.", job_id)

    asyncio.ensure_future(_cleanup())
