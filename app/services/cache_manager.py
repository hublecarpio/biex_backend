"""
Gestión de explicit cache de Gemini para contenido estático.
Cachea system_prompt + pedagogical_context + protocolo de fase.
Fallback garantizado del implicit caching.
"""
import logging
import time
import os

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_genai_client = None
_cache_registry: dict[str, dict] = {}
_CACHE_TTL_SECONDS = "600s"


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


async def get_or_create_cache(
    static_content: str,
    phase: str,
    model: str = "gemini-2.5-flash",
) -> str | None:
    """
    Crea o reutiliza un explicit cache para el contenido estático.
    Retorna el cache name para cached_content, o None si falla.
    """
    if not static_content or len(static_content) < 500:
        return None

    content_hash = str(hash(static_content))[:12]
    cache_key = f"biex_{phase}_{content_hash}"

    existing = _cache_registry.get(cache_key)
    if existing and (time.monotonic() - existing["created_at"]) < 540:
        logger.debug("[cache_manager] Cache hit para fase=%s", phase)
        return existing["cache_name"]

    try:
        client = _get_genai_client()
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=f"biex-{phase}",
                system_instruction=static_content,
                ttl=_CACHE_TTL_SECONDS,
            ),
        )
        cache_name = cache.name
        _cache_registry[cache_key] = {
            "cache_name": cache_name,
            "created_at": time.monotonic(),
        }

        keys_to_remove = [
            k for k in _cache_registry
            if k.startswith(f"biex_{phase}_") and k != cache_key
        ]
        for k in keys_to_remove:
            old = _cache_registry.pop(k, None)
            if old:
                try:
                    client.caches.delete(name=old["cache_name"])
                except Exception:
                    pass

        logger.info("[cache_manager] Explicit cache creado fase=%s name=%s", phase, cache_name)
        return cache_name

    except Exception as e:
        logger.warning("[cache_manager] Explicit cache falló fase=%s: %s", phase, e)
        return None
