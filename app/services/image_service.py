"""
Servicio de generación de imágenes vía webhook externo.
Llama al endpoint GENERATE_IMAGE_WEBHOOK con {"temas": [...]} y retorna las URLs.
"""
import logging

import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_image_client: httpx.AsyncClient | None = None

def _get_image_client() -> httpx.AsyncClient:
    global _image_client
    if _image_client is None or _image_client.is_closed:
        _image_client = httpx.AsyncClient(timeout=120.0)  # 2 minutos
    return _image_client


async def generate_images(temas: list[dict]) -> list[str]:
    """
    Llama al webhook de generación de imágenes.

    Args:
        temas: Lista de dicts con {"tema": str, "descripcion": str}

    Returns:
        Lista de URLs de imágenes generadas. Vacía si falla o no está configurado.
    """
    settings = get_settings()
    webhook_url = settings.generate_image_webhook
    if not webhook_url:
        logger.info("[image_service] GENERATE_IMAGE_WEBHOOK no configurado, salteando.")
        return []

    if not temas:
        return []

    payload = {"temas": temas}
    logger.info("[image_service] Llamando webhook con %s tema(s): %s", len(temas), [t.get("tema") for t in temas])

    try:
        client = _get_image_client()
        resp = await client.post(webhook_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # El webhook puede retornar {"urls": [...]} o directamente [...]
        if isinstance(data, list):
            urls = [item for item in data if isinstance(item, str)]
        elif isinstance(data, dict):
            urls = data.get("urls") or data.get("images") or []
            # Fallback: si n8n retorna URLs en campo "response" (formato legacy)
            if not urls and data.get("response"):
                import re
                found = re.findall(
                    r'https://minio\.biexedu\.com/n8nback/[A-Z0-9]+\.png',
                    str(data["response"]),
                    re.IGNORECASE
                )
                if found:
                    urls = found
                    logger.info("[image_service] URLs extraídas de campo 'response' (formato legacy): %s", len(urls))
            urls = [u for u in urls if isinstance(u, str)]
        else:
            urls = []

        logger.info("[image_service] %s imagen(es) recibida(s).", len(urls))
        return urls

    except Exception as e:
        logger.warning("[image_service] Falló la llamada al webhook: %s", e)
        return []
