"""Tests for the pgvector embedding service (Fase 2).

Every rule protected here has the same failure signature if it breaks: nothing raises,
nothing logs, and retrieval quality silently rots. A wrong `task_type` degrades recall
with no error. An un-normalized fallback vector biases every cosine distance in the
index. A fabricated vector poisons ranking permanently. These are exactly the bugs a
type checker and a smoke test cannot catch, so they are pinned explicitly.
"""
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.config import settings
from app.services.embeddings import (
    TASK_TYPE_DOCUMENT,
    TASK_TYPE_QUERY,
    VALID_TASK_TYPES,
    EmbeddingService,
    EmbeddingServiceError,
    get_embedding_service,
)
from tests.conftest import EMBEDDING_DIM, MockEmbeddingResponse, deterministic_vector


# ==============================================================================
# Helpers
# ==============================================================================

def _task_type_of(config: Any) -> Any:
    """Reads the task_type out of whichever config shape the service produced.

    `EmbeddingService._build_embed_config` returns a real `types.EmbedContentConfig`
    when google-genai is importable and a plain dict otherwise; both must be readable
    so this assertion works in a full venv and in a stripped environment alike.
    """
    task_type = getattr(config, "task_type", None)
    if task_type is None and isinstance(config, dict):
        task_type = config.get("task_type")
    return task_type


def _service_with_client(
    *,
    return_value: Any = None,
    side_effect: Any = None,
) -> tuple[EmbeddingService, AsyncMock]:
    """Builds an EmbeddingService whose GenAI client is a recording AsyncMock.

    Assigning `_client` directly short-circuits `_get_active_client`, so `is_available`
    is True without needing a real API key.

    Args:
        return_value: Canned response for every `embed_content` call.
        side_effect: Async callable driving per-call behaviour (used for the
            primary-fails-then-fallback scenarios).

    Returns:
        The service and the `embed_content` mock, for call-kwargs assertions.
    """
    embed_mock = AsyncMock()
    if side_effect is not None:
        embed_mock.side_effect = side_effect
    else:
        embed_mock.return_value = return_value if return_value is not None else MockEmbeddingResponse()
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.embed_content = embed_mock

    service = EmbeddingService(api_key="unit-test-embedding-key")
    service._client = client
    return service, embed_mock


def _l2_norm_squared(vector: list[float]) -> float:
    """Returns the squared L2 norm of a vector."""
    return sum(value * value for value in vector)


# ==============================================================================
# Pure maths: l2_normalize
# ==============================================================================

class TestL2Normalize:
    """Protects the vector normalization helper used on every fallback embedding."""

    def test_l2_normalize_known_vector(self) -> None:
        """Protects the normalization maths itself: [3,4] must scale to [0.6,0.8].

        A 3-4-5 triangle is the canonical case; getting it wrong (e.g. dividing by the
        sum instead of the norm) would still return plausible-looking small floats.
        """
        assert [round(value, 10) for value in EmbeddingService.l2_normalize([3.0, 4.0])] == [0.6, 0.8]

    def test_l2_normalize_produces_unit_norm(self) -> None:
        """Protects the invariant that a normalized vector has unit length."""
        normalized = EmbeddingService.l2_normalize([float(index % 13) + 0.5 for index in range(EMBEDDING_DIM)])
        assert abs(_l2_norm_squared(normalized) - 1.0) < 1e-9

    def test_l2_normalize_zero_vector_is_returned_unchanged(self) -> None:
        """Protects against division by zero writing NaNs into the pgvector index.

        A NaN row is not merely wrong: it makes every cosine comparison against it
        undefined, and Postgres will happily store it.
        """
        assert EmbeddingService.l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_l2_normalize_empty_vector_is_returned_unchanged(self) -> None:
        """Protects the empty-input guard so normalization never raises."""
        assert EmbeddingService.l2_normalize([]) == []

    def test_l2_normalize_does_not_mutate_its_input(self) -> None:
        """Protects callers from an in-place surprise on a vector they still hold."""
        original = [3.0, 4.0]
        EmbeddingService.l2_normalize(original)
        assert original == [3.0, 4.0]


# ==============================================================================
# Argument validation
# ==============================================================================

class TestEmbeddingArgumentValidation:
    """Protects the fail-fast contract on task_type and empty input."""

    def test_valid_task_types_are_exactly_the_asymmetric_pair(self) -> None:
        """Protects the asymmetric retrieval design from a third task type creeping in."""
        assert VALID_TASK_TYPES == {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}
        assert TASK_TYPE_DOCUMENT != TASK_TYPE_QUERY

    @pytest.mark.parametrize(
        "bad_task_type",
        ["SEMANTIC_SIMILARITY", "CLASSIFICATION", "retrieval_query", "RETRIEVAL", "", None],
    )
    @pytest.mark.asyncio
    async def test_invalid_task_type_raises_value_error(self, bad_task_type: Any) -> None:
        """Protects recall: an unsupported task_type must fail loudly, never be coerced.

        Silently defaulting an unknown task_type to one of the valid ones would produce
        a usable-looking vector that ranks badly forever.
        """
        service, embed_mock = _service_with_client()
        with pytest.raises(ValueError):
            await service.embed_text("un curso de fastapi", task_type=bad_task_type)
        embed_mock.assert_not_awaited()

    @pytest.mark.parametrize("bad_text", ["", "   ", "\n\t \r\n"])
    @pytest.mark.asyncio
    async def test_empty_or_whitespace_text_raises_value_error(self, bad_text: str) -> None:
        """Protects the index from blank documents, which embed to meaningless vectors."""
        service, embed_mock = _service_with_client()
        with pytest.raises(ValueError):
            await service.embed_text(bad_text, task_type=TASK_TYPE_QUERY)
        embed_mock.assert_not_awaited()


# ==============================================================================
# Task type propagation (highest-value assertions in this file)
# ==============================================================================

class TestAsymmetricTaskTypes:
    """Protects the asymmetric retrieval contract at the SDK boundary."""

    @pytest.mark.asyncio
    async def test_embed_document_sends_retrieval_document(self) -> None:
        """Protects recall: ingestion must reach the SDK as RETRIEVAL_DOCUMENT.

        Asserting on the exact string that reached the SDK — not merely on the wrapper
        method being called — is what makes this test meaningful: using the same task
        type on both sides of retrieval degrades recall with no error anywhere.
        """
        service, embed_mock = _service_with_client()
        await service.embed_document("Curso Avanzado de FastAPI & Microservicios")

        assert _task_type_of(embed_mock.await_args.kwargs["config"]) == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_embed_query_sends_retrieval_query(self) -> None:
        """Protects recall: the query side must reach the SDK as RETRIEVAL_QUERY."""
        service, embed_mock = _service_with_client()
        await service.embed_query("algo para aprender a construir microservicios")

        assert _task_type_of(embed_mock.await_args.kwargs["config"]) == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_document_and_query_task_types_actually_differ_at_the_sdk(self) -> None:
        """Protects against both wrappers collapsing onto one task type.

        Each individual assertion above would still pass if someone hard-coded a single
        task type in `_build_embed_config`; comparing the two recorded calls will not.
        """
        service, embed_mock = _service_with_client()
        await service.embed_document("texto del catálogo")
        document_task_type = _task_type_of(embed_mock.await_args.kwargs["config"])
        await service.embed_query("lo que escribió el usuario")
        query_task_type = _task_type_of(embed_mock.await_args.kwargs["config"])

        assert document_task_type != query_task_type
        assert {document_task_type, query_task_type} == VALID_TASK_TYPES

    @pytest.mark.asyncio
    async def test_configured_output_dimensionality_reaches_the_sdk(self) -> None:
        """Protects the index contract: vectors must be requested at the stored width."""
        service, embed_mock = _service_with_client()
        await service.embed_query("hola")

        config = embed_mock.await_args.kwargs["config"]
        dimensions = getattr(config, "output_dimensionality", None)
        if dimensions is None and isinstance(config, dict):
            dimensions = config.get("output_dimensionality")
        assert dimensions == settings.EMBEDDING_DIMENSIONS


# ==============================================================================
# Truncation
# ==============================================================================

class TestEmbeddingTruncation:
    """Protects the defensive input truncation before each provider call."""

    @pytest.mark.asyncio
    async def test_primary_input_is_truncated_to_embedding_input_max_chars(self) -> None:
        """Protects against provider 400s on oversized catalog descriptions.

        Asserted on the text that actually reached the mocked SDK, not on an internal
        variable, so a truncation applied to the wrong string would still fail.
        """
        service, embed_mock = _service_with_client()
        oversized = "á" * (settings.EMBEDDING_INPUT_MAX_CHARS + 4321)

        await service.embed_document(oversized)

        sent_text = embed_mock.await_args.kwargs["contents"]
        assert len(sent_text) == settings.EMBEDDING_INPUT_MAX_CHARS

    @pytest.mark.asyncio
    async def test_short_text_is_passed_through_untruncated_and_stripped(self) -> None:
        """Protects normal-sized inputs from being mangled by the truncation guard."""
        service, embed_mock = _service_with_client()
        await service.embed_query("  Curso de FastAPI  ")

        assert embed_mock.await_args.kwargs["contents"] == "Curso de FastAPI"


# ==============================================================================
# Model fallback chain and normalization
# ==============================================================================

class TestEmbeddingModelFallback:
    """Protects the primary -> fallback chain and its normalization asymmetry."""

    @pytest.mark.asyncio
    async def test_primary_success_returns_the_vector_unnormalized(self) -> None:
        """Protects against double normalization of the primary model's output.

        `gemini-embedding-2` already returns unit-norm vectors for truncated output
        dimensions. Re-normalizing is not merely redundant — it would mask the fallback
        model's genuine need for normalization, making this whole distinction untestable.
        """
        raw_vector = [3.0, 4.0]
        service, embed_mock = _service_with_client(return_value=MockEmbeddingResponse(raw_vector))

        result = await service.embed_query("consulta")

        assert result == raw_vector
        assert abs(_l2_norm_squared(result) - 1.0) > 1e-6, "primary vector must not be re-normalized"
        assert embed_mock.await_args.kwargs["model"] == settings.EMBEDDING_MODEL
        assert embed_mock.await_count == 1, "the fallback model must not be touched on success"

    @pytest.mark.asyncio
    async def test_fallback_model_is_used_truncates_harder_and_normalizes(self) -> None:
        """Protects the whole fallback contract in one path.

        The fallback model has a smaller input ceiling AND does not auto-normalize when
        the output dimensionality is truncated. An un-normalized vector in a cosine
        index silently biases every ranking it participates in, with no error at all.
        """
        calls: list[dict[str, Any]] = []

        async def embed_side_effect(**kwargs: Any) -> MockEmbeddingResponse:
            calls.append(kwargs)
            if kwargs["model"] == settings.EMBEDDING_MODEL:
                raise RuntimeError("primary embedding model refused the request")
            return MockEmbeddingResponse([3.0, 4.0])

        service, embed_mock = _service_with_client(side_effect=embed_side_effect)
        oversized = "z" * (settings.EMBEDDING_INPUT_MAX_CHARS + 9999)

        result = await service.embed_document(oversized)

        assert calls[0]["model"] == settings.EMBEDDING_MODEL
        assert calls[-1]["model"] == settings.EMBEDDING_FALLBACK_MODEL
        assert len(calls[-1]["contents"]) == settings.EMBEDDING_FALLBACK_MAX_CHARS
        assert settings.EMBEDDING_FALLBACK_MAX_CHARS < settings.EMBEDDING_INPUT_MAX_CHARS, (
            "the fallback limit must be genuinely stricter, or 'truncate harder' never fires"
        )
        assert abs(_l2_norm_squared(result) - 1.0) < 1e-6, "the fallback vector MUST be L2-normalized"
        assert _task_type_of(calls[-1]["config"]) == "RETRIEVAL_DOCUMENT", (
            "the fallback attempt must keep the caller's task type"
        )

    @pytest.mark.asyncio
    async def test_both_models_failing_raises_and_never_fabricates_a_vector(self) -> None:
        """Protects the never-fabricate invariant.

        `django_api.py` returns realistic mock dicts while Django is being built; this
        module must NOT follow that pattern. A fake embedding written into the index is
        undetectable after the fact and corrupts ranking permanently.
        """
        async def always_fail(**kwargs: Any) -> MockEmbeddingResponse:
            raise RuntimeError("provider refused the request")

        service, embed_mock = _service_with_client(side_effect=always_fail)
        returned: Any = "<sentinel: nothing was returned>"

        with pytest.raises(EmbeddingServiceError):
            returned = await service.embed_document("texto del catálogo")

        assert returned == "<sentinel: nothing was returned>"
        attempted_models = {call.kwargs["model"] for call in embed_mock.await_args_list}
        assert attempted_models == {settings.EMBEDDING_MODEL, settings.EMBEDDING_FALLBACK_MODEL}

    @pytest.mark.asyncio
    async def test_unavailable_service_raises_instead_of_returning_a_vector(self) -> None:
        """Protects against a missing SDK/API key silently producing a usable-looking vector."""
        service = EmbeddingService(api_key="test-mock-gemini-key-12345")
        service._client = None

        with pytest.raises(EmbeddingServiceError):
            await service.embed_query("consulta")

    @pytest.mark.asyncio
    async def test_empty_provider_response_raises_rather_than_returning_an_empty_vector(self) -> None:
        """Protects against writing a zero-length row when the provider returns nothing."""
        empty_response = MockEmbeddingResponse([])
        empty_response.embeddings = []
        service, _ = _service_with_client(return_value=empty_response)

        with pytest.raises(EmbeddingServiceError):
            await service.embed_query("consulta")


# ==============================================================================
# Wiring
# ==============================================================================

class TestEmbeddingServiceWiring:
    """Protects the module-level singleton and vector width used across the pipeline."""

    def test_get_embedding_service_is_a_singleton(self) -> None:
        """Protects against a new GenAI client being constructed per call."""
        assert get_embedding_service() is get_embedding_service()

    def test_service_dimensions_match_the_configured_index_width(self) -> None:
        """Protects the index contract shared with `upsert_embedding`'s dimension guard."""
        assert get_embedding_service().dimensions == settings.EMBEDDING_DIMENSIONS
        assert len(deterministic_vector()) == settings.EMBEDDING_DIMENSIONS
