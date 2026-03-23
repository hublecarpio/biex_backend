"""
Esquemas Pydantic para request/response y modelos de Supabase.
"""
from typing import Literal

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
    tipo_respuesta: str | None = None  # Deprecated: la decisión la toma el supervisor internamente
    stream: bool = False


class ChatResponse(BaseModel):
    """Response del endpoint de chat (modo no-streaming) - legacy."""

    response: str
    current_phase: str


class ChatResponseStructured(BaseModel):
    """
    Response estructurada para consumo en n8n u otros clientes.

    Campos de imágenes:
    - images: URLs disponibles de inmediato (vacío si el job está en background).
    - images_count: cantidad final de imágenes (incluyendo las pendientes).
    - images_pending: cantidad de imágenes siendo generadas en background. Si > 0,
      el frontend debe mostrar ese número de placeholders y hacer polling al endpoint
      GET /api/v1/images/{images_job_id} hasta que status == "done".
    - images_job_id: ID del job de background. None si las imágenes ya están en `images`.
    """

    mensajes: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    images_count: int = 0
    images_pending: int = Field(default=0, description="Imágenes siendo generadas en background")
    images_job_id: str | None = Field(default=None, description="Job ID para polling de imágenes")
    current_phase: str = "generativa"
    suggested_resources: list[str] = Field(
        default_factory=list,
        description="Recursos sugeridos: 'mind_map', 'fichas', 'video', 'podcast', 'informe'"
    )


class ImageJobResponse(BaseModel):
    """Respuesta del endpoint GET /api/v1/images/{job_id}."""

    job_id: str
    status: Literal["pending", "done", "error"]
    images_pending: int
    urls: list[str] = Field(default_factory=list)
    error: str | None = None


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
    frustracion_nivel: int = Field(ge=0, le=10, description="Nivel de frustración 0-10")
    engagement_score: float = Field(ge=0.0, le=100.0, description="Puntuación de engagement 0-100")
    misconceptions: list[str] = Field(default_factory=list, description="Lista de malentendidos detectados")
    recomendacion: Literal['continuar', 'intensificar', 'simplificar', 'vicario', 'socratico', 'metacognicion'] = Field(description="Recomendación pedagógica o táctica a seguir")
    justificacion: str = Field(description="Breve justificación de los scores y la recomendación")
    topic: str = Field(
        default="",
        description="Tema educativo principal del mensaje. "
        "Vacío si es saludo o mensaje sin tema educativo."
    )
    image_needed: bool = Field(
        default=False,
        description="True si el alumno pide explícitamente una imagen/visual/mapa/diagrama "
        "o si el contenido que no entiende es inherentemente visual (geografía, anatomía, "
        "procesos, estructuras) y una imagen ayudaría significativamente a la comprensión."
    )


# --- Supabase: Starter Profile ---
class ProfileData(BaseModel):
    """Datos de perfil del alumno — acepta todos los campos del formulario."""
    model_config = {"extra": "allow", "populate_by_name": True}

    feelings: str = ""
    interests: list = Field(default_factory=list)
    unique_data: str = Field(alias="uniqueData", default="")
    description: str = ""
    learning_goal: list = Field(alias="learningGoal", default_factory=list)
    learning_style: list = Field(alias="learningStyle", default_factory=list)
    explanation_style: list = Field(alias="explanationStyle", default_factory=list)
    content_preference: list = Field(alias="contentPreference", default_factory=list)
    study_time: str = Field(alias="studyTime", default="")
    challenges: list = Field(default_factory=list)
    language: str = ""
    autonomy_level: str = Field(alias="autonomyLevel", default="")
    learning_goals: str = Field(alias="learningGoals", default="")
    problem_approach: str = Field(alias="problemApproach", default="")
    session_duration: str = Field(alias="sessionDuration", default="")
    knowledge_context: str = Field(alias="knowledgeContext", default="")
    passionate_topics: list = Field(alias="passionateTopics", default_factory=list)
    challenge_tolerance: str = Field(alias="challengeTolerance", default="")
    communication_style: str = Field(alias="communicationStyle", default="")
    unique_characteristics: str = Field(alias="uniqueCharacteristics", default="")


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
