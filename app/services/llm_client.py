"""LLM Client service wrapper for Google GenAI (Gemini / Google AI Studio)."""
from typing import Optional
from app.core.config import settings


class LLMClientService:
    """Wrapper service for Google GenAI SDK (`google-genai`).

    Handles client initialization and provides unified methods for
    streaming and standard completions.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        # Google GenAI client instance (genai.Client) will be initialized here.
        self._client = None
