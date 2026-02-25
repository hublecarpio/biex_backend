"""
Esquemas Pydantic para request/response y modelos de Supabase.
"""
from pydantic import BaseModel, Field


# --- Chat API ---
class ChatRequest(BaseModel):
    """Request del endpoint de chat."""

    user_id: str
    message: str
    session_id: str
    stream: bool = False


class ChatResponse(BaseModel):
    """Response del endpoint de chat (modo no-streaming)."""

    response: str
    current_phase: str


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
