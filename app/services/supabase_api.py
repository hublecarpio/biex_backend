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
        self.service_role_key = settings.supabase_service_role_key
        self._client_anon: httpx.AsyncClient | None = None
        self._client_service: httpx.AsyncClient | None = None

    def _make_client(self, key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": "application/json",
            },
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Cliente con anon_key (para Edge Functions y lecturas públicas)."""
        if self._client_anon is None or self._client_anon.is_closed:
            self._client_anon = self._make_client(self.anon_key)
        return self._client_anon

    async def _get_service_client(self) -> httpx.AsyncClient:
        """Cliente con service_role_key (para escrituras en tablas con RLS)."""
        key = self.service_role_key or self.anon_key
        if self._client_service is None or self._client_service.is_closed:
            self._client_service = self._make_client(key)
        return self._client_service

    async def close(self) -> None:
        """Cierra los clientes HTTP."""
        for c in (self._client_anon, self._client_service):
            if c and not c.is_closed:
                await c.aclose()
        self._client_anon = None
        self._client_service = None

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
        client = await self._get_service_client()  # Usa service_role_key para bypassear RLS
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

    async def get_session_state(self, conversation_id: str) -> dict | None:
        """GET /rest/v1/session_state?conversation_id=eq.{conversation_id}&limit=1"""
        logger.info("[supabase] GET session_state para conversation_id=%s...", conversation_id)
        client = await self._get_service_client()
        try:
            resp = await client.get(
                "/rest/v1/session_state",
                params={"conversation_id": f"eq.{conversation_id}", "limit": "1"}
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.warning("[supabase] get_session_state falló: %s", e)
            return None

    async def save_session_state(self, session_data: dict) -> None:
        """POST /rest/v1/session_state con Prefer: resolution=merge-duplicates"""
        conversation_id = session_data.get("conversation_id")
        logger.info("[supabase] POST session_state para conversation_id=%s...", conversation_id)
        client = await self._get_service_client()
        try:
            resp = await client.post(
                "/rest/v1/session_state",
                headers={"Prefer": "resolution=merge-duplicates"},
                json=session_data
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[supabase] save_session_state falló: %s", e)

    async def get_active_protocol(self, phase: str) -> str | None:
        """GET /rest/v1/protocols?phase=eq.{phase}&is_active=eq.true&order=version.desc&limit=1"""
        logger.info("[supabase] GET active protocol para phase=%s...", phase)
        client = await self._get_client()
        try:
            resp = await client.get(
                "/rest/v1/protocols",
                params={
                    "phase": f"eq.{phase}",
                    "is_active": "eq.true",
                    "order": "version.desc",
                    "limit": "1"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get("content")
            return None
        except Exception as e:
            logger.warning("[supabase] get_active_protocol falló: %s", e)
            return None

    async def save_gatekeeper_evaluation(self, evaluation: dict) -> None:
        """POST /rest/v1/gatekeeper_evaluations"""
        logger.info("[supabase] POST gatekeeper_evaluations...")
        client = await self._get_service_client()
        try:
            resp = await client.post(
                "/rest/v1/gatekeeper_evaluations",
                json=evaluation
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[supabase] save_gatekeeper_evaluation falló: %s", e)

    async def get_pedagogical_docs(self) -> str:
        """GET /rest/v1/pedagogical_docs?is_active=eq.true&order=sort_order.asc"""
        logger.info("[supabase] GET pedagogical_docs...")
        client = await self._get_client()
        try:
            resp = await client.get(
                "/rest/v1/pedagogical_docs",
                params={"is_active": "eq.true", "order": "sort_order.asc"}
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                docs = []
                for doc in data:
                    title = doc.get("title", "Doc")
                    content = doc.get("content", "")
                    docs.append(f"## {title}\n{content}\n\n")
                return "".join(docs)
            return ""
        except Exception as e:
            logger.warning("[supabase] get_pedagogical_docs falló: %s", e)
            return ""

    async def get_learner_insights(self, user_id: str) -> list:
        """GET /rest/v1/learner_insights?user_id=eq.{user_id}&order=confidence.desc&limit=20"""
        logger.info("[supabase] GET learner_insights para user_id=%s...", user_id)
        client = await self._get_service_client()
        try:
            resp = await client.get(
                "/rest/v1/learner_insights",
                params={
                    "user_id": f"eq.{user_id}",
                    "order": "confidence.desc",
                    "limit": "20"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning("[supabase] get_learner_insights falló: %s", e)
            return []

    async def save_learner_insight(self, insight: dict) -> None:
        """POST /rest/v1/learner_insights"""
        logger.info("[supabase] POST learner_insights...")
        client = await self._get_service_client()
        try:
            resp = await client.post(
                "/rest/v1/learner_insights",
                json=insight
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[supabase] save_learner_insight falló: %s", e)

    async def update_conversation_phase(self, conversation_id: str, phase: str) -> None:
        """PATCH /rest/v1/conversations?id=eq.{conversation_id}"""
        logger.info("[supabase] PATCH conversations para conversation_id=%s, phase=%s...", conversation_id, phase)
        client = await self._get_service_client()
        try:
            resp = await client.patch(
                "/rest/v1/conversations",
                params={"id": f"eq.{conversation_id}"},
                json={"current_phase": phase}
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[supabase] update_conversation_phase falló: %s", e)
