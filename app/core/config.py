from functools import lru_cache
import json
from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Settings and Environment Variable Management.

    Utilizes Pydantic Settings (v2) to validate and provide type-safe
    configuration parameters across the microservice.
    """

    # --- Project Metadata ---
    PROJECT_NAME: str = Field(default="AI Agent Gateway", description="Name of the microservice")
    VERSION: str = Field(default="0.1.0", description="API Version")
    API_V1_STR: str = Field(default="/api/v1", description="Prefix for API v1 routes")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment (development, staging, production)")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # --- LLM & AI Engine (Google GenAI Studio / Gemini API) ---
    GEMINI_API_KEY: str = Field(
        ...,
        description="Google AI Studio / Gemini API Key for google-genai client authentication",
    )

    # --- Distributed Cache & Agent State Memory ---
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URI for agent session memory and caching",
    )

    # --- Django Monolith Integration & Security ---
    DJANGO_BACKEND_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of the transactional Django backend",
    )
    INTERNAL_API_SECRET: str = Field(
        ...,
        description="Shared secret key for authenticating internal service-to-service communication",
    )

    # --- CORS Configuration ---
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"],
        description="List of origins allowed to make Cross-Origin requests",
    )

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, value: Any) -> list[str]:
        """Parses CORS origins from JSON string, comma-separated string, or list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(origin).rstrip("/") for origin in parsed]
                except json.JSONDecodeError:
                    pass
            return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        elif isinstance(value, (list, tuple)):
            return [str(origin).rstrip("/") for origin in value]
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Provides a cached singleton instance of the application Settings."""
    return Settings()


# Global settings instance for direct imports
settings: Settings = get_settings()
