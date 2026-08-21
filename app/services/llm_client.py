"""LLM Client service wrapper for Google GenAI (Gemini / Google AI Studio)."""
import asyncio
import inspect
import logging
import random
from typing import Any, AsyncGenerator, Optional, Union

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore
    GENAI_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger("ai_gateway.llm")


class LLMServiceError(Exception):
    """Base exception for LLM operations."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMRateLimitError(LLMServiceError):
    """Raised when the LLM provider returns a 429 / Resource Exhausted status."""

    def __init__(self, message: str = "LLM API Rate limit exceeded. Please retry shortly.") -> None:
        super().__init__(message=message, status_code=429)


class LLMTimeoutError(LLMServiceError):
    """Raised when an LLM inference call times out."""

    def __init__(self, message: str = "LLM generation timed out.") -> None:
        super().__init__(message=message, status_code=504)


class LLMClientService:
    """Wrapper service for Google GenAI SDK (`google-genai`).

    Provides high-level asynchronous methods for standard generation,
    streaming completions, dynamic multi-model fallbacks, and Exponential Backoff retries.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.default_model = settings.DEFAULT_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = getattr(settings, "LLM_MAX_RETRIES", 4)
        self.initial_retry_delay = getattr(settings, "LLM_INITIAL_RETRY_DELAY", 1.0)
        self.backoff_factor = getattr(settings, "LLM_BACKOFF_FACTOR", 2.0)
        self.max_retry_delay = getattr(settings, "LLM_MAX_RETRY_DELAY", 8.0)
        self._client: Optional[Any] = None

    def _is_api_key_configured(self) -> bool:
        """Determines whether a genuine Google AI Studio API Key is configured."""
        if not self.api_key:
            return False
        placeholder_keys = {
            "your-google-ai-studio-api-key-here",
            "your_api_key_here",
            "test-mock-gemini-key-12345",
            "",
        }
        if self.api_key in placeholder_keys or self.api_key.startswith("test-mock"):
            return False
        return True

    def _get_active_client(self) -> Optional[Any]:
        """Returns active client or initializes Google GenAI client if configured."""
        if self._client is not None:
            return self._client

        if GENAI_AVAILABLE and self._is_api_key_configured():
            try:
                self._client = genai.Client(api_key=self.api_key)
                masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if len(self.api_key) > 10 else "***"
                logger.info("Initialized live Google GenAI Client with key: %s", masked_key)
                return self._client
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI Client: %s", exc)
                return None
        return None

    @property
    def is_available(self) -> bool:
        """Returns True if live or mocked Google GenAI client is active."""
        return self._get_active_client() is not None

    def _get_candidate_models(self, primary_model: Optional[str] = None) -> list[str]:
        """Returns ordered list of candidate Gemini models for graceful fallback."""
        target = primary_model or self.default_model
        fallback_order = [
            target,
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
        ]
        # Deduplicate while preserving order
        seen = set()
        candidates: list[str] = []
        for m in fallback_order:
            if m and m not in seen:
                seen.add(m)
                candidates.append(m)
        return candidates

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determines if an exception is a transient error eligible for retry with backoff."""
        err_str = str(exc).lower()
        transient_indicators = [
            "high demand",
            "spikes in demand",
            "503",
            "service unavailable",
            "unavailable",
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "504",
            "gateway timeout",
            "timeout",
            "500",
            "internal error",
            "connection",
            "overloaded",
            "temporarily unavailable",
        ]
        return any(ind in err_str for ind in transient_indicators)

    def _is_model_unavailable_error(self, exc: Exception) -> bool:
        """Determines if an error is due to an invalid/unsupported model name."""
        err_str = str(exc).lower()
        return "404" in err_str or "not found" in err_str or "unknown model" in err_str or "invalid model" in err_str

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        tools: Optional[list[Any]] = None,
        tool_config: Optional[Any] = None,
    ) -> Any:
        """Builds types.GenerateContentConfig if SDK is available, else returns dict."""
        if GENAI_AVAILABLE and types is not None:
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
            }
            if hasattr(types, "ThinkingConfig"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif hasattr(types, "GenerateContentConfig"):
                config_kwargs["thinking_config"] = {"thinking_budget": 0}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = max_output_tokens
            if tools:
                config_kwargs["tools"] = tools
            if tool_config:
                config_kwargs["tool_config"] = tool_config
            return types.GenerateContentConfig(**config_kwargs)

        cfg: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        return cfg

    async def generate_content(
        self,
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        tools: Optional[list[Any]] = None,
    ) -> str:
        """Asynchronously generates standard text completion with model fallback and Exponential Backoff."""
        client = self._get_active_client()

        if client is None:
            return self._generate_fallback_response(contents, system_instruction)

        candidate_models = self._get_candidate_models(model)
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )

        last_exception: Optional[Exception] = None

        for current_model in candidate_models:
            for attempt in range(self.max_retries):
                try:
                    async with asyncio.timeout(self.timeout):
                        if hasattr(client, "aio") and hasattr(client.aio, "models"):
                            res = client.aio.models.generate_content(
                                model=current_model,
                                contents=contents,
                                config=config,
                            )
                            response = await res if inspect.isawaitable(res) else res
                        elif hasattr(client, "models"):
                            res = client.models.generate_content(
                                model=current_model,
                                contents=contents,
                                config=config,
                            )
                            response = await res if inspect.isawaitable(res) else res
                        else:
                            return self._generate_fallback_response(contents, system_instruction)

                        if response and hasattr(response, "text") and response.text:
                            return response.text
                        return "No content generated."

                except Exception as exc:
                    last_exception = exc
                    # If model is not found/supported, switch to next model immediately
                    if self._is_model_unavailable_error(exc):
                        logger.warning("Model '%s' unavailable (%s). Trying next candidate...", current_model, exc)
                        break

                    is_transient = self._is_transient_error(exc)
                    if not is_transient or attempt == self.max_retries - 1:
                        logger.warning(
                            "Model '%s' attempt %d/%d failed (transient=%s): %s",
                            current_model, attempt + 1, self.max_retries, is_transient, exc
                        )
                        break

                    # Exponential backoff calculation with randomized jitter
                    delay = min(self.max_retry_delay, self.initial_retry_delay * (self.backoff_factor ** attempt))
                    delay_jittered = delay + random.uniform(0.1, 0.4)
                    logger.info(
                        "Model '%s' retry in %.2fs (attempt %d/%d)...",
                        current_model, delay_jittered, attempt + 1, self.max_retries
                    )
                    await asyncio.sleep(delay_jittered)

        logger.error("All candidate models failed. Last exception: %s", last_exception)
        return self._generate_fallback_response(contents, system_instruction, error_context=last_exception)

    async def generate_content_stream(
        self,
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        tools: Optional[list[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously streams response token chunks with model fallback and Exponential Backoff."""
        client = self._get_active_client()

        if client is None:
            fallback_text = self._generate_fallback_response(contents, system_instruction)
            for word in fallback_text.split(" "):
                yield word + " "
                await asyncio.sleep(0.02)
            return

        candidate_models = self._get_candidate_models(model)
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )

        last_exception: Optional[Exception] = None

        for current_model in candidate_models:
            for attempt in range(self.max_retries):
                try:
                    if hasattr(client, "aio") and hasattr(client.aio, "models"):
                        stream_res = client.aio.models.generate_content_stream(
                            model=current_model,
                            contents=contents,
                            config=config,
                        )
                        stream_obj = await stream_res if inspect.isawaitable(stream_res) else stream_res
                        async for chunk in stream_obj:
                            if chunk and hasattr(chunk, "text") and chunk.text:
                                yield chunk.text
                        return

                    elif hasattr(client, "models"):
                        sync_stream = client.models.generate_content_stream(
                            model=current_model,
                            contents=contents,
                            config=config,
                        )
                        for chunk in sync_stream:
                            if chunk and hasattr(chunk, "text") and chunk.text:
                                yield chunk.text
                        return

                except Exception as exc:
                    last_exception = exc
                    if self._is_model_unavailable_error(exc):
                        logger.warning("Stream model '%s' unavailable (%s). Switching model...", current_model, exc)
                        break

                    is_transient = self._is_transient_error(exc)
                    if not is_transient or attempt == self.max_retries - 1:
                        logger.warning(
                            "Stream model '%s' attempt %d/%d failed (transient=%s): %s",
                            current_model, attempt + 1, self.max_retries, is_transient, exc
                        )
                        break

                    delay = min(self.max_retry_delay, self.initial_retry_delay * (self.backoff_factor ** attempt))
                    delay_jittered = delay + random.uniform(0.1, 0.4)
                    logger.info(
                        "Stream model '%s' retry in %.2fs (attempt %d/%d)...",
                        current_model, delay_jittered, attempt + 1, self.max_retries
                    )
                    await asyncio.sleep(delay_jittered)

        logger.error("All stream candidate models failed. Falling back. Last exception: %s", last_exception)
        fallback_text = self._generate_fallback_response(contents, system_instruction, error_context=last_exception)
        for word in fallback_text.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)

    def _generate_fallback_response(
        self,
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
        error_context: Optional[Exception] = None,
    ) -> str:
        """Provides accurate, contextual fallback response distinguishing unconfigured key vs upstream error."""
        prompt_preview = str(contents)[:120]

        # Case 1: GEMINI_API_KEY is truly not configured
        if not self._is_api_key_configured():
            return (
                f"[Simulated GenAI Response] El Gateway ha procesado tu consulta: "
                f"'{prompt_preview}...'. Configura GEMINI_API_KEY en .env para respuestas en vivo de Gemini."
            )

        # Case 2: GEMINI_API_KEY is configured, but upstream provider encountered transient errors/outage
        logger.info("Generating professional resilience fallback for query: '%s'", prompt_preview)
        return (
            f"[AI Agent Gateway] Gracias por tu mensaje. El servicio de IA está procesando solicitudes en "
            f"modo de contingencia temporal debido a alta demanda o latencia en el proveedor. "
            f"Hemos recibido tu consulta: '{prompt_preview}...'. Si requieres asistencia inmediata, "
            f"puedes reintentar en unos instantes o consultar las secciones de portafolio y catálogo."
        )


# Global singleton instance
_llm_service: Optional[LLMClientService] = None


def get_llm_service() -> LLMClientService:
    """Returns the singleton instance of LLMClientService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMClientService()
    return _llm_service
