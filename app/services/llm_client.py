"""LLM Client service wrapper for Google GenAI (Gemini / Google AI Studio).

Resilience note — OpenRouter is wired in below as a last-resort provider fallback for
CHAT and FUNCTION CALLING ONLY. It must NEVER be used to generate embeddings: vectors
produced by a different provider live in a different semantic space with a different
dimensionality, are not cosine-comparable with Gemini vectors, and mixing them inside
one pgvector index silently invalidates ranking. See `app/services/embeddings.py`,
which deliberately has no provider fallback at all.
"""
import asyncio
import inspect
import json
import logging
import random
from typing import Any, AsyncGenerator, Optional, Union

import httpx

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
            "gemini-3.1-pro-preview",
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
        thinking_budget: Optional[int] = None,
    ) -> Any:
        """Builds types.GenerateContentConfig if SDK is available, else returns dict."""
        if GENAI_AVAILABLE and types is not None:
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
            }
            if thinking_budget is not None and thinking_budget > 0:
                if hasattr(types, "ThinkingConfig"):
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
                elif hasattr(types, "GenerateContentConfig"):
                    config_kwargs["thinking_config"] = {"thinking_budget": thinking_budget}
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
            # No Gemini client at all: try OpenRouter before the canned response.
            # No-op unless OPENROUTER_API_KEY is configured.
            openrouter_text = await self._try_openrouter_fallback(contents, system_instruction, temperature)
            if openrouter_text:
                return openrouter_text
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

        # Fase 7: last-resort cross-provider fallback (chat only). No-op without a key.
        openrouter_text = await self._try_openrouter_fallback(contents, system_instruction, temperature)
        if openrouter_text:
            return openrouter_text

        return self._generate_fallback_response(contents, system_instruction, error_context=last_exception)

    async def generate_raw(
        self,
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[list[Any]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        """Returns the raw provider response object (or None if unavailable), for tool-call inspection.

        Unlike `generate_content`, this seam does not collapse the response to `.text`,
        so the multi-turn function-calling loop can read out `functionCall` parts and
        feed tool results back into the conversation.

        Args:
            contents: Prompt string or Gemini-shaped content list.
            system_instruction: Optional system prompt.
            model: Optional primary model override.
            temperature: Sampling temperature.
            tools: Optional tool/function declarations.
            max_output_tokens: Optional output cap.

        Returns:
            The raw SDK response object, or None when no candidate model succeeded or
            the client is unavailable. Callers should treat None as "no tool calls".
        """
        client = self._get_active_client()
        if client is None:
            logger.info("generate_raw called without an active GenAI client; returning None.")
            return None

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
                        elif hasattr(client, "models"):
                            res = client.models.generate_content(
                                model=current_model,
                                contents=contents,
                                config=config,
                            )
                        else:
                            return None
                        response = await res if inspect.isawaitable(res) else res

                        if response is not None:
                            return response
                        return None

                except Exception as exc:
                    last_exception = exc
                    if self._is_model_unavailable_error(exc):
                        logger.warning("Raw model '%s' unavailable (%s). Trying next candidate...", current_model, exc)
                        break

                    is_transient = self._is_transient_error(exc)
                    if not is_transient or attempt == self.max_retries - 1:
                        logger.warning(
                            "Raw model '%s' attempt %d/%d failed (transient=%s): %s",
                            current_model, attempt + 1, self.max_retries, is_transient, exc
                        )
                        break

                    delay = min(self.max_retry_delay, self.initial_retry_delay * (self.backoff_factor ** attempt))
                    delay_jittered = delay + random.uniform(0.1, 0.4)
                    logger.info(
                        "Raw model '%s' retry in %.2fs (attempt %d/%d)...",
                        current_model, delay_jittered, attempt + 1, self.max_retries
                    )
                    await asyncio.sleep(delay_jittered)

        logger.error("All candidate models failed in generate_raw. Last exception: %s", last_exception)
        return None

    @staticmethod
    def extract_function_calls(response: Any) -> list[dict[str, Any]]:
        """Returns [{"name": str, "args": dict}, ...] parsed from the response's candidates/parts.

        Parses an SDK object that cannot be imported in every environment, so every
        access is defensive: any malformed or missing level yields an empty list
        rather than an exception.

        Args:
            response: A raw provider response object (may be None).

        Returns:
            A list of parsed function calls; empty when there are none or on any error.
        """
        calls: list[dict[str, Any]] = []
        if response is None:
            return calls

        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return calls

            for candidate in candidates:
                if candidate is None:
                    continue
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", None)
                if not parts:
                    continue

                for part in parts:
                    if part is None:
                        continue
                    function_call = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                    if function_call is None:
                        continue

                    name = getattr(function_call, "name", None)
                    if name is None and isinstance(function_call, dict):
                        name = function_call.get("name")
                    if not name:
                        continue

                    raw_args = getattr(function_call, "args", None)
                    if raw_args is None and isinstance(function_call, dict):
                        raw_args = function_call.get("args")

                    args: dict[str, Any] = {}
                    if isinstance(raw_args, dict):
                        args = dict(raw_args)
                    elif isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args)
                            args = parsed if isinstance(parsed, dict) else {}
                        except (ValueError, TypeError):
                            args = {}
                    elif raw_args is not None:
                        # Proto/Struct-like mappings expose .items() but are not dicts.
                        try:
                            args = dict(raw_args)
                        except (TypeError, ValueError):
                            args = {}

                    calls.append({"name": str(name), "args": args})
        except Exception as exc:
            logger.warning("Failed to extract function calls from response: %s", exc)
            return []

        return calls

    @staticmethod
    def extract_text(response: Any) -> str:
        """Best-effort text extraction from a raw response (handles .text, and candidates→parts→text).

        Args:
            response: A raw provider response object (may be None).

        Returns:
            The concatenated text, or an empty string when none could be extracted.
        """
        if response is None:
            return ""

        try:
            direct_text = getattr(response, "text", None)
            if isinstance(direct_text, str) and direct_text.strip():
                return direct_text
        except Exception as exc:
            logger.debug("Response.text accessor raised: %s", exc)

        chunks: list[str] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                if candidate is None:
                    continue
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    if part is None:
                        continue
                    text = getattr(part, "text", None)
                    if text is None and isinstance(part, dict):
                        text = part.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
        except Exception as exc:
            logger.warning("Failed to extract text from response candidates: %s", exc)
            return ""

        return "".join(chunks)

    # ==============================================================================
    # OpenRouter Provider Fallback (Fase 7) — chat / function calling ONLY
    # ==============================================================================

    @staticmethod
    def _to_openai_messages(
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Translates Gemini-shaped contents into OpenAI chat messages.

        Gemini's `"model"` role maps to OpenAI's `"assistant"`; the system instruction
        becomes a leading `"system"` message.

        Args:
            contents: A prompt string, or a list of `{"role", "parts": [{"text"}]}` dicts.
            system_instruction: Optional system prompt.

        Returns:
            A list of `{"role", "content"}` dicts suitable for /chat/completions.
        """
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": str(system_instruction)})

        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
            return messages

        try:
            for entry in contents or []:
                if isinstance(entry, str):
                    messages.append({"role": "user", "content": entry})
                    continue

                if isinstance(entry, dict):
                    role = str(entry.get("role", "user") or "user")
                    parts = entry.get("parts") or []
                else:
                    role = str(getattr(entry, "role", "user") or "user")
                    parts = getattr(entry, "parts", None) or []

                texts: list[str] = []
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict):
                        if isinstance(part.get("text"), str):
                            texts.append(part["text"])
                    else:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            texts.append(text)

                content_text = "\n".join(t for t in texts if t)
                if not content_text:
                    continue

                messages.append({
                    "role": "assistant" if role == "model" else ("system" if role == "system" else "user"),
                    "content": content_text,
                })
        except Exception as exc:
            logger.warning("Failed to translate contents to OpenAI messages: %s", exc)
            messages.append({"role": "user", "content": str(contents)[:4000]})

        if not any(m["role"] == "user" for m in messages):
            messages.append({"role": "user", "content": str(contents)[:4000]})
        return messages

    async def _try_openrouter_fallback(
        self,
        contents: Union[str, list[Any]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Attempts one OpenRouter completion after all Gemini candidates are exhausted.

        Uses the already-present `httpx` dependency against the OpenAI-compatible
        `/chat/completions` route. Never used for embeddings (see module docstring).

        Args:
            contents: The same contents passed to the Gemini call.
            system_instruction: Optional system prompt.
            temperature: Sampling temperature.

        Returns:
            The assistant text, or None when the key is unset or the request failed.
        """
        api_key = getattr(settings, "OPENROUTER_API_KEY", "") or ""
        if not api_key.strip():
            # No-op by default: without a key the behaviour is exactly as before.
            return None

        base_url = str(getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        model_name = getattr(settings, "OPENROUTER_MODEL", "openai/gpt-4o-mini")
        timeout = float(getattr(settings, "OPENROUTER_TIMEOUT_SECONDS", 20.0))

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._to_openai_messages(contents, system_instruction),
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                if response.status_code != 200:
                    logger.warning(
                        "OpenRouter fallback returned HTTP %s: %s",
                        response.status_code, response.text[:200]
                    )
                    return None

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    logger.warning("OpenRouter fallback returned no choices.")
                    return None

                message = choices[0].get("message") or {}
                text = message.get("content")
                if isinstance(text, str) and text.strip():
                    logger.info("OpenRouter fallback served the response with model '%s'.", model_name)
                    return text
                logger.warning("OpenRouter fallback returned an empty message content.")
                return None
        except Exception as exc:
            logger.warning("OpenRouter fallback request failed: %s", exc)
            return None

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
            # No Gemini client at all: try OpenRouter before the canned response.
            # No-op unless OPENROUTER_API_KEY is configured.
            openrouter_text = await self._try_openrouter_fallback(contents, system_instruction, temperature)
            fallback_text = openrouter_text or self._generate_fallback_response(contents, system_instruction)
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

        # Fase 7: last-resort cross-provider fallback (chat only). No-op without a key.
        openrouter_text = await self._try_openrouter_fallback(contents, system_instruction, temperature)
        fallback_text = openrouter_text or self._generate_fallback_response(
            contents, system_instruction, error_context=last_exception
        )
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
