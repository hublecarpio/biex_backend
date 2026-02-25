"""
Configuración centralizada con pydantic-settings.
Manejo de variables de entorno para BIEX Backend.
"""
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
    database_url: str = ""  # postgresql://user:pass@host:port/db


def get_settings() -> Settings:
    """Factory para obtener la configuración."""
    return Settings()
