"""
Gestión de explicit cache de Gemini para contenido estático.
Cachea SOLO pedagogical_context + system_prompt (no cambian entre turnos).
El explicit cache es fallback del implicit caching.
Implicit funciona automático cuando el prefijo del prompt se repite.
"""
import logging
import time
import os
import hashlib

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_genai_client = None
_active_cache: dict | None = None
_CACHE_TTL_SECONDS = "600s"


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _hash_content(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:16]


async def get_or_create_cache(
    static_content: str,
    model: str = "gemini-2.5-flash",
) -> str | None:
    """
    Crea o reutiliza un explicit cache para el contenido estático.
    Retorna el cache name o None si falla.
    El cache se reutiliza si el contenido no cambió (mismo hash) y no expiró.
    """
    global _active_cache

    if not static_content or len(static_content) < 500:
        return None

    content_hash = _hash_content(static_content)

    # Reutilizar cache existente si mismo hash y no expirado (margen 60s antes de TTL)
    if (_active_cache
            and _active_cache["hash"] == content_hash
            and (time.monotonic() - _active_cache["created_at"]) < 540):
        return _active_cache["cache_name"]

    # Crear nuevo cache
    try:
        client = _get_genai_client()

        # Eliminar cache viejo si existe
        if _active_cache:
            try:
                client.caches.delete(name=_active_cache["cache_name"])
            except Exception:
                pass

        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name="biex-static",
                system_instruction=static_content,
                ttl=_CACHE_TTL_SECONDS,
            ),
        )

        _active_cache = {
            "cache_name": cache.name,
            "hash": content_hash,
            "created_at": time.monotonic(),
        }

        logger.info(
            "[cache_manager] Explicit cache creado — name=%s hash=%s",
            cache.name, content_hash,
        )
        return cache.name

    except Exception as e:
        logger.warning("[cache_manager] Explicit cache falló: %s", e)
        return None
