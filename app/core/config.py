"""
Configuración centralizada con pydantic-settings.
Manejo de variables de entorno para BIEX Backend.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables de entorno para la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_key_secret: str = ""

    # Gemini
    gemini_api_key: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""

    # Postgres (memoria / checkpointer LangGraph)
    # database_url ya no se usa porque pasamos al guardado nativo por REST

    # Image generation webhook
    generate_image_webhook: str = ""  # POST con {"temas": [...]}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Factory para obtener la configuración.
    Cacheada con lru_cache: Settings se instancia UNA sola vez por proceso,
    evitando releer el .env en cada request.
    """
    return Settings()
