"""Text embedding service for the pgvector RAG pipeline (Fase 2).

Wraps the Google GenAI embedding models behind a small async surface used by both
the ingestion worker (documents) and the retrieval path (queries).

IMPORTANT — embeddings must NEVER be routed through OpenRouter (or any other
provider). A vector produced by a different provider lives in a different semantic
space, with a different dimensionality, and is NOT cosine-comparable with a Gemini
vector. Mixing providers inside a single pgvector index silently invalidates the
whole ranking. OpenRouter exists in this codebase strictly as a chat /
function-calling fallback (see `app/services/llm_client.py`); do not "helpfully"
extend it to this module.

Second invariant: this module NEVER fabricates a vector. Unlike `django_api.py`,
which returns realistic mock dicts while the Django endpoints are being built, a
fake embedding written into the index would corrupt search ranking permanently and
undetectably. Every failure path raises `EmbeddingServiceError` instead.
"""
import asyncio
import inspect
import logging
import math
import random
from typing import Any, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    GENAI_AVAILABLE = False

from app.core.config import settings
from app.services.llm_client import LLMClientService

logger = logging.getLogger("ai_gateway.embeddings")

# Asymmetric retrieval: documents and queries MUST be embedded with different task
# types. Using the same task type on both sides measurably degrades recall.
VALID_TASK_TYPES = frozenset({"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"})

TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class EmbeddingServiceError(Exception):
    """Raised when a text embedding cannot be produced by any configured model."""


def _is_transient_error(exc: Exception) -> bool:
    """Determines if an exception is transient, reusing the LLM client heuristic.

    Delegates to `LLMClientService._is_transient_error` so the keyword list stays a
    single source of truth across the LLM and embedding services.

    Args:
        exc: The exception raised by the provider SDK.

    Returns:
        True if the error looks transient and the call is worth retrying.
    """
    try:
        return LLMClientService(api_key="")._is_transient_error(exc)
    except Exception:  # pragma: no cover - defensive, heuristic must never raise
        return False


class EmbeddingService:
    """Async wrapper over the Google GenAI embedding models.

    Implements a two-model fallback chain (primary -> fallback) with exponential
    backoff and jitter around each attempt, defensive input truncation, and L2
    normalization of fallback-model vectors.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.fallback_model = settings.EMBEDDING_FALLBACK_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = getattr(settings, "LLM_MAX_RETRIES", 2)
        self.initial_retry_delay = getattr(settings, "LLM_INITIAL_RETRY_DELAY", 0.5)
        self.backoff_factor = getattr(settings, "LLM_BACKOFF_FACTOR", 2.0)
        self.max_retry_delay = getattr(settings, "LLM_MAX_RETRY_DELAY", 2.0)
        self._client: Optional[Any] = None
        # Reuses LLMClientService's placeholder-key detection ("test-mock...",
        # "your-google-ai-studio-api-key-here", ...) so both services agree on what
        # an "unconfigured" key means.
        self._key_probe = LLMClientService(api_key=self.api_key)

    def _is_api_key_configured(self) -> bool:
        """Returns True when a genuine (non-placeholder) Google AI Studio key is set."""
        return self._key_probe._is_api_key_configured()

    def _get_active_client(self) -> Optional[Any]:
        """Returns the active GenAI client, initializing it lazily when configured."""
        if self._client is not None:
            return self._client

        if GENAI_AVAILABLE and self._is_api_key_configured():
            try:
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI Embedding client (model=%s)", self.model)
                return self._client
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI Embedding client: %s", exc)
                return None
        return None

    @property
    def is_available(self) -> bool:
        """Returns True if the SDK is importable and a real API key is configured."""
        return self._get_active_client() is not None

    @staticmethod
    def l2_normalize(vector: list[float]) -> list[float]:
        """Scales a vector to unit L2 norm.

        Args:
            vector: The raw embedding values.

        Returns:
            The unit-norm vector, or the input unchanged when it is empty or its norm
            is zero (dividing by zero would produce NaNs and poison the index).
        """
        if not vector:
            return vector

        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm == 0.0 or not math.isfinite(norm):
            return vector
        return [float(value) / norm for value in vector]

    def _build_embed_config(self, task_type: str) -> Any:
        """Builds types.EmbedContentConfig if available, else a plain dict payload."""
        config_kwargs: dict[str, Any] = {
            "output_dimensionality": self.dimensions,
            "task_type": task_type,
        }
        if GENAI_AVAILABLE and types is not None and hasattr(types, "EmbedContentConfig"):
            try:
                return types.EmbedContentConfig(**config_kwargs)
            except Exception as exc:
                logger.debug("EmbedContentConfig construction failed (%s); using dict config.", exc)
        return config_kwargs

    @staticmethod
    def _extract_vector(result: Any) -> list[float]:
        """Extracts and coerces `result.embeddings[0].values` into a list[float].

        Args:
            result: The raw SDK embedding response.

        Raises:
            EmbeddingServiceError: If the response carries no usable vector.
        """
        embeddings = getattr(result, "embeddings", None)
        if not embeddings:
            raise EmbeddingServiceError("Embedding response contained no embeddings.")

        values = getattr(embeddings[0], "values", None)
        if values is None and isinstance(embeddings[0], dict):
            values = embeddings[0].get("values")
        if not values:
            raise EmbeddingServiceError("Embedding response contained an empty vector.")

        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise EmbeddingServiceError(f"Embedding vector is not numeric: {exc}") from exc

        if not vector:
            raise EmbeddingServiceError("Embedding response contained an empty vector.")
        return vector

    async def _embed_once(self, model: str, text: str, task_type: str) -> list[float]:
        """Performs a single embedding call against the given model."""
        client = self._get_active_client()
        if client is None:
            raise EmbeddingServiceError("Google GenAI embedding client is not available.")

        config = self._build_embed_config(task_type)

        async with asyncio.timeout(self.timeout):
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                res = client.aio.models.embed_content(model=model, contents=text, config=config)
            elif hasattr(client, "models"):
                res = client.models.embed_content(model=model, contents=text, config=config)
            else:
                raise EmbeddingServiceError("GenAI client exposes no embedding interface.")
            result = await res if inspect.isawaitable(res) else res

        return self._extract_vector(result)

    async def _embed_with_retries(self, model: str, text: str, task_type: str) -> list[float]:
        """Calls `_embed_once` with exponential backoff and jitter on transient errors.

        Args:
            model: Embedding model identifier.
            text: Already-truncated input text.
            task_type: One of `VALID_TASK_TYPES`.

        Returns:
            The raw (non-normalized) embedding vector.

        Raises:
            Exception: The last exception raised by the provider after all retries.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(max(1, self.max_retries)):
            try:
                return await self._embed_once(model=model, text=text, task_type=task_type)
            except Exception as exc:
                last_exception = exc
                is_transient = _is_transient_error(exc)
                if not is_transient or attempt == max(1, self.max_retries) - 1:
                    logger.warning(
                        "Embedding model '%s' attempt %d/%d failed (transient=%s): %s",
                        model, attempt + 1, max(1, self.max_retries), is_transient, exc,
                    )
                    break

                delay = min(self.max_retry_delay, self.initial_retry_delay * (self.backoff_factor ** attempt))
                delay_jittered = delay + random.uniform(0.1, 0.4)
                logger.info(
                    "Embedding model '%s' retry in %.2fs (attempt %d/%d)...",
                    model, delay_jittered, attempt + 1, max(1, self.max_retries),
                )
                await asyncio.sleep(delay_jittered)

        raise last_exception if last_exception else EmbeddingServiceError("Embedding failed without an exception.")

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        """Embeds a single text with the configured model fallback chain.

        Args:
            text: The raw text to embed. Must be non-empty after stripping.
            task_type: `RETRIEVAL_DOCUMENT` for catalog ingestion, `RETRIEVAL_QUERY`
                for what the user typed. Asymmetric by design.

        Returns:
            The embedding vector, L2-normalized only when produced by the fallback model.

        Raises:
            ValueError: If `task_type` is not in `VALID_TASK_TYPES` or the text is empty.
            EmbeddingServiceError: If the SDK/key is unavailable or both models failed.
        """
        # A wrong task_type is a programming error: never silently coerce it, because
        # symmetric task types measurably degrade retrieval recall.
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Expected one of: {sorted(VALID_TASK_TYPES)}."
            )

        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")

        if not self.is_available:
            raise EmbeddingServiceError(
                "Embedding service unavailable: google-genai SDK missing or GEMINI_API_KEY unconfigured. "
                "Refusing to fabricate a vector."
            )

        clean_text = text.strip()
        # Defense in depth: the producer upstream already truncates, but we never
        # trust that an upstream limit was actually respected.
        primary_text = clean_text[: settings.EMBEDDING_INPUT_MAX_CHARS]

        try:
            # gemini-embedding-2 auto-normalizes truncated output dimensions, so the
            # primary vector MUST NOT be normalized again here.
            return await self._embed_with_retries(
                model=self.model, text=primary_text, task_type=task_type
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary embedding model '%s' failed (%s). Retrying with fallback model '%s'.",
                self.model, primary_exc, self.fallback_model,
            )

        # The fallback model only accepts 2048 tokens, so truncate harder. Derived from
        # the already-truncated text, so the effective limit is the smaller of the two.
        fallback_text = primary_text[: settings.EMBEDDING_FALLBACK_MAX_CHARS]

        try:
            raw_vector = await self._embed_with_retries(
                model=self.fallback_model, text=fallback_text, task_type=task_type
            )
        except Exception as fallback_exc:
            raise EmbeddingServiceError(
                f"Both embedding models failed ('{self.model}', '{self.fallback_model}'): {fallback_exc}"
            ) from fallback_exc

        # CRITICAL: gemini-embedding-001 does NOT auto-normalize when the output
        # dimensionality is truncated. An un-normalized vector silently biases the
        # cosine distance computed on the Django/pgvector side, so normalize here.
        return self.l2_normalize(raw_vector)

    async def embed_document(self, text: str) -> list[float]:
        """Embeds a catalog document for ingestion (task_type=RETRIEVAL_DOCUMENT)."""
        return await self.embed_text(text=text, task_type=TASK_TYPE_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a user query for retrieval (task_type=RETRIEVAL_QUERY)."""
        return await self.embed_text(text=text, task_type=TASK_TYPE_QUERY)


# Global singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Returns the singleton instance of EmbeddingService."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
