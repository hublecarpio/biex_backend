"""
Esquemas Pydantic para request/response y modelos de Supabase.
"""
from pydantic import BaseModel, Field


# --- Chat API ---
class ChatRequest(BaseModel):
    """
    Request del endpoint de chat (payload desde n8n/webhook).
    id_conversation = thread_id para memoria Postgres.
    """

    mensaje: str
    id_conversation: str
    id_user: str
    tipo_respuesta: str = "informativa"  # ej. informativa, socratica, etc.
    stream: bool = False


class ChatResponse(BaseModel):
    """Response del endpoint de chat (modo no-streaming) - legacy."""

    response: str
    current_phase: str


class ChatResponseStructured(BaseModel):
    """
    Response estructurada para consumo en n8n u otros clientes.
    mensajes: segmentos de texto (sin markdown, sin URLs de imagen).
    images: URLs de imágenes Minio (https://minio.biexedu.com/n8nback/*.png).
    """

    mensajes: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    images_count: int = 0
    current_phase: str = "generativa"


class ImageTopic(BaseModel):
    """Un tema visual a generar como imagen."""
    tema: str = Field(description="Nombre corto del tema visual")
    descripcion: str = Field(description="Descripción detallada del tema para guiar al generador de imágenes (2-3 oraciones)")


class ImageTopicList(BaseModel):
    """Lista de temas visuales extraídos de la respuesta del tutor."""
    temas: list[ImageTopic] = Field(
        default_factory=list,
        description="Lista de temas visuales relevantes de la respuesta. Vacía si no hay nada visual que mostrar."
    )


# --- Gatekeeper (evaluación del alumno) ---
class GatekeeperEval(BaseModel):
    """Evaluación del gatekeeper: comprensión y frustración."""

    comprension_score: float = Field(ge=0.0, le=100.0, description="Puntuación de comprensión 0-100")
    frustracion_detectada: bool = Field(description="Si se detecta frustración en el mensaje")


# --- Supabase: Starter Profile ---
class ProfileData(BaseModel):
    """Datos de perfil del alumno dentro de starter_profile."""

    feelings: str = ""
    interests: list = Field(default_factory=list)
    unique_data: str = Field(alias="uniqueData", default="")
    learning_goal: list = Field(alias="learningGoal", default_factory=list)
    learning_style: list = Field(alias="learningStyle", default_factory=list)

    model_config = {"populate_by_name": True}


class UserProfile(BaseModel):
    """Perfil de usuario (name, email)."""

    name: str = ""
    email: str = ""


class StarterProfile(BaseModel):
    """Perfil inicial del alumno (user_id, age, profile_data)."""

    user_id: str = ""
    age: int = 0
    age_group: str = ""
    profile_data: ProfileData = Field(default_factory=ProfileData)


class StarterProfileResponse(BaseModel):
    """
    Respuesta de get-starter-profile.
    Mapea: {"success": bool, "data": {"starter_profile": {...}, "user_profile": {...}}}
    """

    success: bool = False
    data: dict = Field(default_factory=dict)

    def get_starter_profile(self) -> StarterProfile | None:
        """Extrae starter_profile de data."""
        sp = self.data.get("starter_profile")
        if sp is None:
            return None
        if isinstance(sp, dict):
            return StarterProfile(**sp)
        return sp

    def get_user_profile(self) -> UserProfile | None:
        """Extrae user_profile de data."""
        up = self.data.get("user_profile")
        if up is None:
            return None
        if isinstance(up, dict):
            return UserProfile(**up)
        return up
