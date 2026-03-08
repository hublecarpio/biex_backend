import time
import logging

logger = logging.getLogger(__name__)

_pedagogical_cache = {"content": None, "loaded_at": 0}
PEDAGOGICAL_CACHE_TTL = 1800  # 30 minutos en segundos

async def get_pedagogical_context_cached(client) -> str:
    """Retorna la documentación pedagógica, cacheada en memoria."""
    now = time.monotonic()
    if (_pedagogical_cache["content"] is not None
            and (now - _pedagogical_cache["loaded_at"]) < PEDAGOGICAL_CACHE_TTL):
        return _pedagogical_cache["content"]

    try:
        content = await client.get_pedagogical_docs()
        _pedagogical_cache["content"] = content
        _pedagogical_cache["loaded_at"] = now
        logger.info(f"Pedagogical docs cache refreshed: {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"Error loading pedagogical docs: {e}")
        # Si falla pero hay cache viejo, usarlo
        if _pedagogical_cache["content"]:
            return _pedagogical_cache["content"]
        return ""
