"""
Cliente asíncrono para Supabase Edge Functions usando httpx.
"""
import httpx
from app.core.config import get_settings
from app.models.schemas import StarterProfile, StarterProfileResponse


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
        client = await self._get_client()
        try:
            resp = await client.get("/functions/v1/get-system-prompt")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                return data
            if isinstance(data, dict) and "prompt" in data:
                return data["prompt"]
            if isinstance(data, dict) and "data" in data:
                return data["data"] if isinstance(data["data"], str) else str(data["data"])
            return str(data) if data else ""
        except Exception:
            return ""

    async def get_starter_profile(self, user_id: str) -> StarterProfile | None:
        """
        GET a /functions/v1/get-starter-profile?user_id={user_id}.
        Retorna el perfil inicial del alumno o None.
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                "/functions/v1/get-starter-profile",
                params={"user_id": user_id},
            )
            resp.raise_for_status()
            body = resp.json()
            parsed = StarterProfileResponse(success=True, data=body.get("data", body))
            return parsed.get_starter_profile()
        except Exception:
            return None

    async def query_knowledge(self, query: str) -> str:
        """
        POST a /functions/v1/query-knowledge.
        Payload: {"search_type": "semantic_ai", "query": query}.
        Retorna el valor de la clave "context" de la respuesta JSON, o "" si falla.
        """
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
                return ctx if isinstance(ctx, str) else str(ctx)
            return ""
        except Exception:
            return ""
