"""
Cliente asíncrono para Supabase Edge Functions usando httpx.
"""
import logging

import httpx
from app.core.config import get_settings
from app.models.schemas import StarterProfile, StarterProfileResponse

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Cliente asíncrono para las funciones de Supabase."""

    def __init__(
        self,
        base_url: str | None = None,
        anon_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.supabase_url).rstrip("/")
        self.anon_key = anon_key or settings.supabase_anon_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.anon_key}",
                    "apikey": self.anon_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_system_prompt(self) -> str:
        """
        GET a /functions/v1/get-system-prompt.
        Retorna el prompt del sistema como string.
        """
        logger.info("[supabase] GET get-system-prompt...")
        client = await self._get_client()
        try:
            resp = await client.get("/functions/v1/get-system-prompt")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                result = data
            elif isinstance(data, dict) and "prompt" in data:
                result = data["prompt"]
            elif isinstance(data, dict) and "data" in data:
                result = data["data"] if isinstance(data["data"], str) else str(data["data"])
            else:
                result = str(data) if data else ""
            logger.info("[supabase] system_prompt obtenido (%s chars).", len(result))
            return result
        except Exception as e:
            logger.warning("[supabase] get_system_prompt falló: %s", e)
            return ""

    async def get_starter_profile(self, user_id: str) -> StarterProfile | None:
        """
        GET a /functions/v1/get-starter-profile?user_id={user_id}.
        Retorna el perfil inicial del alumno o None.
        """
        logger.info("[supabase] GET get-starter-profile para user_id=%s...", user_id)
        client = await self._get_client()
        try:
            resp = await client.get(
                "/functions/v1/get-starter-profile",
                params={"user_id": user_id},
            )
            resp.raise_for_status()
            body = resp.json()
            parsed = StarterProfileResponse(success=True, data=body.get("data", body))
            profile = parsed.get_starter_profile()
            logger.info("[supabase] starter_profile=%s", "OK" if profile else "no encontrado")
            return profile
        except Exception as e:
            logger.warning("[supabase] get_starter_profile falló: %s", e)
            return None

    async def query_knowledge(self, query: str) -> str:
        """
        POST a /functions/v1/query-knowledge.
        Payload: {"search_type": "semantic_ai", "query": query}.
        Retorna el valor de la clave "context" de la respuesta JSON, o "" si falla.
        """
        logger.info("[supabase] POST query-knowledge (query: %s chars)...", len(query))
        client = await self._get_client()
        try:
            resp = await client.post(
                "/functions/v1/query-knowledge",
                json={"search_type": "semantic_ai", "query": query},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "context" in data:
                ctx = data["context"]
                result = ctx if isinstance(ctx, str) else str(ctx)
                logger.info("[supabase] RAG context obtenido (%s chars).", len(result))
                return result
            logger.info("[supabase] RAG context vacío en la respuesta.")
            return ""
        except Exception as e:
            logger.warning("[supabase] query_knowledge falló: %s", e)
            return ""

    async def get_conversation_history(self, conversation_id: str) -> list[dict]:
        """
        GET a /rest/v1/messages.
        Retorna el historial de la conversación ordenado cronológicamente.
        """
        logger.info("[supabase] GET messages para conversation_id=%s...", conversation_id)
        client = await self._get_client()
        try:
            resp = await client.get(
                "/rest/v1/messages",
                params={
                    "conversation_id": f"eq.{conversation_id}",
                    "select": "*",
                    "order": "created_at.asc"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("[supabase] Historial recuperado: %s mensajes.", len(data))
            return data
        except Exception as e:
            logger.warning("[supabase] get_conversation_history falló: %s", e)
            return []

    async def save_message(self, user_id: str, conversation_id: str, role: str, message: str, metadata: dict | None = None) -> dict | None:
        """
        POST a /rest/v1/messages.
        Guarda un nuevo mensaje en la tabla nativa de Supabase.
        """
        logger.info("[supabase] POST message (role=%s, conversation_id=%s)...", role, conversation_id)
        client = await self._get_client()
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "message": message,
        }
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            resp = await client.post(
                "/rest/v1/messages",
                headers={"Prefer": "return=representation"},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                logger.debug("[supabase] Mensaje guardado correctamente (ID: %s).", data[0].get("id"))
                return data[0]
            return None
        except Exception as e:
            logger.warning("[supabase] save_message falló: %s", e)
            return None
