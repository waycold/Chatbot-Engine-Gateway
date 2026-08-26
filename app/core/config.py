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


    # --- Embeddings (RAG pipeline) ---
    EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-2",
        description="Primary Gemini embedding model (accepts up to 8192 input tokens)",
    )
    EMBEDDING_FALLBACK_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Secondary embedding model used when the primary model fails (2048 token limit)",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=768,
        description="Output dimensionality of the stored pgvector embeddings",
    )
    EMBEDDING_INPUT_MAX_CHARS: int = Field(
        default=6000,
        description=(
            "Defensive character truncation applied to THE TEXT BEING EMBEDDED before calling "
            "the primary embedding model: the product text on ingestion (RETRIEVAL_DOCUMENT) "
            "and the user's query on search (RETRIEVAL_QUERY). This is NOT a cap on chat "
            "prompts -- prompt/grounding text has its own budget, PROMPT_CONTEXT_MAX_CHARS."
        ),
    )
    EMBEDDING_FALLBACK_MAX_CHARS: int = Field(
        default=4500,
        description=(
            "Harder truncation for the fallback model, which only accepts 2048 tokens. "
            "The design doc suggested 2048*3=6144 chars, but that exceeds EMBEDDING_INPUT_MAX_CHARS "
            "(6000), so the 'truncate harder' rule would never actually fire. It is also an "
            "optimistic ratio: 3 chars/token holds for plain ASCII English, while accented "
            "Spanish product copy tokenizes closer to 2.2-2.6 chars/token. 4500 (~2.2 chars/token) "
            "keeps us safely under the 2048-token ceiling. Raise only with real tokenizer data. "
            "Recalibrate this number with `scripts/calibrate_token_ratio.py` instead of guessing."
        ),
    )
    PROMPT_CONTEXT_MAX_CHARS: int = Field(
        default=24000,
        description=(
            "Cap for grounding/context text injected into a chat prompt (retrieved catalog "
            "snippets, business context, tool output). A different budget with a different "
            "purpose than EMBEDDING_INPUT_MAX_CHARS -- chat models accept far larger inputs "
            "than the embedding models -- and deliberately NOT shared with it."
        ),
    )
    EMBEDDING_BATCH_LIMIT: int = Field(
        default=20,
        description="Maximum number of pending embedding tasks pulled per ingestion run",
    )

    # --- Function Calling / Tool Loop ---
    ENABLE_TOOL_CALLING: bool = Field(
        default=True,
        description="Enables the multi-turn Gemini function-calling loop for agents",
    )
    MAX_TOOL_ITERATIONS: int = Field(
        default=4,
        description="Maximum number of tool-call round trips before forcing a final answer",
    )

    # --- Authorization ---
    # NOTE: there is deliberately no configurable role list here. Django's `auth_user`
    # model has no `role` column; privilege is expressed exclusively by the native
    # `is_staff` / `is_superuser` booleans returned by the token validator. A settings
    # key holding magic role strings would be a second, divergent source of truth.

    # --- Auth token validation cache ---
    TOKEN_VALIDATION_CACHE_TTL_SECONDS: float = Field(
        default=20.0,
        description=(
            "Short-lived cache so one conversation turn does not re-validate the same token "
            "against Django several times (the dispatcher and the analytics agent each "
            "validate independently). Set to 0 to disable caching entirely."
        ),
    )
    TOKEN_VALIDATION_CACHE_MAX_ENTRIES: int = Field(
        default=512,
        description="Maximum number of validated tokens held in the in-process validation cache",
    )

    # --- Provider Resilience (OpenRouter — chat/function-calling ONLY, never embeddings) ---
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="OpenRouter API key; an empty value disables the chat fallback entirely",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL of the OpenRouter OpenAI-compatible API",
    )
    OPENROUTER_MODEL: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model identifier used for the chat/function-calling fallback",
    )
    OPENROUTER_TIMEOUT_SECONDS: float = Field(
        default=20.0,
        description="Timeout in seconds for the OpenRouter fallback HTTP request",
    )

    # --- Knowledge Base & Business Context ---
    ECOMMERCE_CONTEXT_PATH: str = Field(
        default="data/ecommerce_business_context.md",
        description="Path to Markdown business context file",
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
        # Defensive: never fall through to an implicit None for scalars/unknown types.
        if value is None:
            return []
        return [str(value).strip().rstrip("/")]

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
