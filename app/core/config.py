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
        default="",
        description="Google AI Studio / Gemini API Key for google-genai client authentication",
    )
    DEFAULT_MODEL: str = Field(
        default="gemini-3.7-flash",
        description="Default Gemini LLM model identifier for inference",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        description="Maximum timeout in seconds for LLM generation requests",
    )
    LLM_MAX_RETRIES: int = Field(
        default=2,
        description="Maximum number of retry attempts for transient LLM errors",
    )
    LLM_INITIAL_RETRY_DELAY: float = Field(
        default=0.5,
        description="Initial delay in seconds before first retry in exponential backoff",
    )
    LLM_BACKOFF_FACTOR: float = Field(
        default=2.0,
        description="Multiplier factor for exponential backoff delay",
    )
    LLM_MAX_RETRY_DELAY: float = Field(
        default=2.0,
        description="Maximum ceiling delay in seconds between retries",
    )


    # --- Distributed Cache & Agent State Memory ---
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URI for agent session memory and caching",
    )
    SESSION_TTL_SECONDS: int = Field(
        default=86400,
        description="TTL in seconds for storing conversation history in memory",
    )

    # --- Django Monolith Integration & Security ---
    DJANGO_BACKEND_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of the transactional Django backend",
    )
    DJANGO_API_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        description="Timeout in seconds for internal Django HTTP API calls",
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
    @field_validator("GEMINI_API_KEY", mode="before")
    @classmethod
    def clean_gemini_api_key(cls, value: Any) -> str:
        """Sanitizes quotes/whitespace and checks alternative env var names."""
        import os
        val_str = str(value or "").strip().strip("'").strip('"').strip()
        if not val_str:
            val_str = (
                os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_KEY", "")
                or os.environ.get("GOOGLE_GENAI_API_KEY", "")
            ).strip().strip("'").strip('"').strip()
        return val_str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Provides a cached singleton instance of the application Settings."""
    return Settings()


# Global settings instance for direct imports
settings: Settings = get_settings()
