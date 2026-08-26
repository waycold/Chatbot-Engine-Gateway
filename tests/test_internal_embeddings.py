"""Tests for the internal embeddings ingestion router (Fase 6).

Requires FastAPI, so this suite only runs in the project venv.

Two properties dominate here. First, these routes are service-to-service only: they
drain an outbox and write into the pgvector index, so an unauthenticated caller must
never reach them. Second, the ingestion loop must be poison-resistant — one product
whose text cannot be embedded must not stall every product queued behind it, which is
exactly what a naive `for` loop with a shared try/except would do.
"""
from typing import Any, Optional
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from app.api.v1 import internal_embeddings
from app.api.v1.internal_embeddings import (
    process_pending_embeddings,
    wake_embeddings_worker,
)
from app.core.config import settings
from app.services.django_api import DjangoAPIService
from app.services.embeddings import EmbeddingServiceError
from tests.conftest import EMBEDDING_DIM, deterministic_vector

WAKE_URL = f"{settings.API_V1_STR}/internal/embeddings/wake"
PROCESS_URL = f"{settings.API_V1_STR}/internal/embeddings/process-pending"
STATUS_URL = f"{settings.API_V1_STR}/internal/embeddings/status"


# ==============================================================================
# Doubles
# ==============================================================================

class RecordingEmbeddingService:
    """Embedding service double that records task types and can poison chosen texts."""

    def __init__(self, poison_substrings: Optional[list[str]] = None) -> None:
        self.poison_substrings = poison_substrings or []
        self.calls: list[dict[str, Any]] = []
        self.is_available = True

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        """Returns a vector unless the text was marked as poison."""
        self.calls.append({"text": text, "task_type": task_type})
        if any(needle in text for needle in self.poison_substrings):
            raise EmbeddingServiceError(f"cannot embed poisoned text: {text[:30]}")
        return deterministic_vector()

    async def embed_document(self, text: str) -> list[float]:
        """Ingestion half of the asymmetric pair."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Retrieval half of the asymmetric pair."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_QUERY")


class RecordingDjangoService:
    """Django double recording outbox reads, upserts and error marks."""

    def __init__(self, tasks: Optional[list[dict[str, Any]]] = None, upsert_error_for: Optional[set[str]] = None) -> None:
        self.tasks = tasks if tasks is not None else _default_tasks()
        self.upsert_error_for = upsert_error_for or set()
        self.pending_calls: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.marked_errors: list[dict[str, Any]] = []

    async def get_pending_embeddings(self, limit: int = 20) -> dict[str, Any]:
        """Serves one outbox batch."""
        self.pending_calls.append({"limit": limit})
        return {"status": "success", "count": len(self.tasks), "tasks": list(self.tasks)}

    async def upsert_embedding(self, **kwargs: Any) -> dict[str, Any]:
        """Records the upsert, optionally rejecting selected task ids."""
        self.upserts.append(kwargs)
        if kwargs.get("task_id") in self.upsert_error_for:
            return {"status": "error", "error": "índice rechazó el vector", "task_id": kwargs.get("task_id")}
        return {"status": "success", "task_id": kwargs.get("task_id"), "dimensions": len(kwargs.get("vector", []))}

    async def mark_embedding_error(self, task_id: str, error: str) -> dict[str, Any]:
        """Records the failure mark."""
        self.marked_errors.append({"task_id": task_id, "error": error})
        return {"status": "success", "task_id": task_id, "marked": "error"}


def _default_tasks() -> list[dict[str, Any]]:
    """Builds a four-task outbox batch where exactly one text is poisoned."""
    return [
        {"task_id": "emb_001", "item_id": 1, "text": "Servicio Cloud AI. Categoría: Servicios.", "content_hash": "sha256:a"},
        {"task_id": "emb_002", "item_id": 2, "text": "POISON Consultoría DevOps.", "content_hash": "sha256:b"},
        {"task_id": "emb_003", "item_id": 3, "text": "Curso FastAPI. Categoría: Cursos.", "content_hash": "sha256:c"},
        {"task_id": "emb_004", "item_id": 4, "text": "Módulo LLM. Categoría: Software.", "content_hash": "sha256:d"},
    ]


@pytest.fixture
def wired_services(monkeypatch: pytest.MonkeyPatch) -> tuple[RecordingDjangoService, RecordingEmbeddingService]:
    """Wires the router's two singleton getters to recording doubles.

    The router imports the getters into its own namespace, so they must be patched
    there rather than on the service modules.
    """
    django = RecordingDjangoService()
    embedder = RecordingEmbeddingService(poison_substrings=["POISON"])
    monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: django)
    monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: embedder)
    return django, embedder


# ==============================================================================
# Authentication
# ==============================================================================

class TestInternalEmbeddingsAuthentication:
    """Protects the service-to-service boundary on every ingestion route."""

    @pytest.mark.parametrize(
        "method,url",
        [("post", WAKE_URL), ("post", PROCESS_URL), ("get", STATUS_URL)],
    )
    def test_missing_secret_is_rejected(self, sync_client: TestClient, method: str, url: str) -> None:
        """Protects against a browser or scraper reaching the ingestion machinery.

        These endpoints trigger paid embedding calls and write into the vector index;
        an unauthenticated caller could both run up a bill and poison retrieval.
        """
        response = getattr(sync_client, method)(url)

        assert response.status_code == 401

    @pytest.mark.parametrize(
        "method,url",
        [("post", WAKE_URL), ("post", PROCESS_URL), ("get", STATUS_URL)],
    )
    def test_wrong_secret_is_rejected(
        self, sync_client: TestClient, invalid_internal_headers: dict[str, str], method: str, url: str
    ) -> None:
        """Protects against a stale or guessed secret being accepted."""
        response = getattr(sync_client, method)(url, headers=invalid_internal_headers)

        assert response.status_code == 401

    def test_valid_secret_is_accepted(
        self,
        sync_client: TestClient,
        valid_internal_headers: dict[str, str],
        wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService],
    ) -> None:
        """Protects the happy path so the auth tests cannot pass for the wrong reason."""
        response = sync_client.post(PROCESS_URL, headers=valid_internal_headers)

        assert response.status_code == 200
        assert response.json()["total"] == 4


# ==============================================================================
# /wake
# ==============================================================================

class TestWakeEndpoint:
    """Protects the fire-and-forget contract of the Django write webhook."""

    @pytest.mark.asyncio
    async def test_wake_queues_the_work_instead_of_awaiting_it(self) -> None:
        """Protects Django's request latency: /wake must NOT run the ingestion inline.

        Django calls this straight after a catalog write. Awaiting a whole embedding
        batch here would add seconds to a user-facing save, so the run has to be handed
        to BackgroundTasks. The handler is called directly (rather than through the test
        client, which drains background tasks before returning) so that "was queued, not
        awaited" is asserted rather than inferred from timing.
        """
        background_tasks = BackgroundTasks()

        result = await wake_embeddings_worker(background_tasks=background_tasks, _authorized=True)

        assert result == {"status": "accepted", "queued": True}
        assert len(background_tasks.tasks) == 1
        assert background_tasks.tasks[0].func is process_pending_embeddings

    def test_wake_returns_202_accepted(
        self,
        sync_client: TestClient,
        valid_internal_headers: dict[str, str],
        wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService],
    ) -> None:
        """Protects the status code Django's webhook client checks for."""
        response = sync_client.post(WAKE_URL, headers=valid_internal_headers)

        assert response.status_code == 202
        assert response.json() == {"status": "accepted", "queued": True}


# ==============================================================================
# process_pending_embeddings
# ==============================================================================

class TestProcessPendingEmbeddings:
    """Protects the ingestion loop's task type, isolation and reporting."""

    @pytest.mark.asyncio
    async def test_documents_are_embedded_with_retrieval_document(
        self, wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService]
    ) -> None:
        """Protects the ingestion half of the asymmetric retrieval pair.

        If ingestion used RETRIEVAL_QUERY the index would still fill up and every search
        would still return results — just measurably worse ones, with no error anywhere
        and no way to notice short of re-embedding the whole catalog.
        """
        _, embedder = wired_services

        await process_pending_embeddings()

        assert embedder.calls, "no embedding call was made at all"
        assert {call["task_type"] for call in embedder.calls} == {"RETRIEVAL_DOCUMENT"}

    @pytest.mark.asyncio
    async def test_one_poisoned_task_is_marked_and_the_batch_continues(
        self, wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService]
    ) -> None:
        """Protects the outbox from being permanently stalled behind one bad record.

        With a single shared try/except the first unembeddable product would abort the
        run, and — because it stays pending — every subsequent run would abort at the
        same record. The outbox would never drain again, silently.
        """
        django, embedder = wired_services

        summary = await process_pending_embeddings()

        assert summary["total"] == 4
        assert summary["processed"] == 3, "the three healthy tasks must still be ingested"
        assert summary["failed"] == 1
        assert summary["status"] == "partial"
        assert [mark["task_id"] for mark in django.marked_errors] == ["emb_002"]
        assert [upsert["task_id"] for upsert in django.upserts] == ["emb_001", "emb_003", "emb_004"]

    @pytest.mark.asyncio
    async def test_a_rejected_upsert_is_also_marked_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects against an index rejection being silently counted as a success.

        The embedding succeeded but the write did not; leaving the task un-marked would
        drop that product out of the index with a clean-looking run summary.
        """
        django = RecordingDjangoService(upsert_error_for={"emb_003"})
        embedder = RecordingEmbeddingService()
        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: django)
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: embedder)

        summary = await process_pending_embeddings()

        assert summary["processed"] == 3
        assert summary["failed"] == 1
        assert [mark["task_id"] for mark in django.marked_errors] == ["emb_003"]

    @pytest.mark.asyncio
    async def test_a_failing_mark_error_does_not_abort_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the batch when even the failure-recording call fails.

        Marking the failure is best-effort bookkeeping; letting it take down the run
        would turn a transient bookkeeping blip into a stalled outbox.
        """
        class MarkFails(RecordingDjangoService):
            async def mark_embedding_error(self, task_id: str, error: str) -> dict[str, Any]:
                self.marked_errors.append({"task_id": task_id, "error": error})
                raise RuntimeError("outbox unreachable")

        django = MarkFails()
        embedder = RecordingEmbeddingService(poison_substrings=["POISON"])
        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: django)
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: embedder)

        summary = await process_pending_embeddings()

        assert summary["processed"] == 3
        assert summary["failed"] == 1

    @pytest.mark.asyncio
    async def test_every_task_failing_reports_status_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the alerting signal: a fully failed run must not read as 'partial'."""
        django = RecordingDjangoService()
        embedder = RecordingEmbeddingService(poison_substrings=["."])  # every text contains a dot
        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: django)
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: embedder)

        summary = await process_pending_embeddings()

        assert summary["processed"] == 0
        assert summary["failed"] == 4
        assert summary["status"] == "error"
        assert len(django.marked_errors) == 4

    @pytest.mark.asyncio
    async def test_a_clean_run_reports_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Protects the healthy signal so the failure statuses are distinguishable."""
        django = RecordingDjangoService()
        embedder = RecordingEmbeddingService()
        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: django)
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: embedder)

        summary = await process_pending_embeddings()

        assert summary["status"] == "success"
        assert summary["processed"] == 4
        assert summary["failed"] == 0
        assert django.marked_errors == []

    @pytest.mark.asyncio
    async def test_an_outbox_read_failure_returns_an_error_summary_without_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the cron caller from a 500 when the outbox itself is unreachable."""
        class BrokenOutbox(RecordingDjangoService):
            async def get_pending_embeddings(self, limit: int = 20) -> dict[str, Any]:
                raise RuntimeError("outbox table unreachable")

        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: BrokenOutbox())
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: RecordingEmbeddingService())

        summary = await process_pending_embeddings()

        assert summary["status"] == "error"
        assert summary["total"] == 0
        assert "error" in summary

    @pytest.mark.asyncio
    async def test_batch_limit_defaults_to_the_configured_value(
        self, wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService]
    ) -> None:
        """Protects the batch bound that keeps a single run inside the request budget."""
        django, _ = wired_services

        await process_pending_embeddings()

        assert django.pending_calls[-1]["limit"] == settings.EMBEDDING_BATCH_LIMIT

    @pytest.mark.asyncio
    async def test_explicit_limit_overrides_the_default(
        self, wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService]
    ) -> None:
        """Protects the operator override used to drain a large backlog gradually."""
        django, _ = wired_services

        await process_pending_embeddings(limit=2)

        assert django.pending_calls[-1]["limit"] == 2

    @pytest.mark.asyncio
    async def test_the_upserted_vector_matches_the_configured_dimensionality(
        self, wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService]
    ) -> None:
        """Protects the contract between the embedding width and the pgvector column."""
        django, _ = wired_services

        await process_pending_embeddings()

        assert all(len(upsert["vector"]) == settings.EMBEDDING_DIMENSIONS for upsert in django.upserts)
        assert all(upsert["model_name"] == settings.EMBEDDING_MODEL for upsert in django.upserts)
        assert all(upsert["content_hash"] for upsert in django.upserts), (
            "the content hash must round-trip so Django can detect stale rows"
        )


# ==============================================================================
# upsert dimension guard
# ==============================================================================

class TestUpsertDimensionGuard:
    """Protects the index from wrong-width vectors, without spending an HTTP call."""

    @pytest.mark.asyncio
    async def test_wrong_dimension_vector_is_rejected_without_an_http_call(self) -> None:
        """Protects pgvector from a silently corrupting write.

        A wrong-width vector is either rejected by Postgres (a 500 on a background job
        nobody watches) or, worse, accepted into a differently-typed column where it
        distorts every ranking it participates in. Rejecting locally also avoids paying
        a network round trip to learn something checkable in one comparison.
        """
        service = DjangoAPIService()
        http_calls: list[Any] = []

        class TripwireClient:
            async def post(self, *args: Any, **kwargs: Any) -> Any:
                http_calls.append(args)
                raise AssertionError("an invalid vector must never be sent over the wire")

        async def get_client() -> Any:
            return TripwireClient()

        service.get_client = get_client  # type: ignore[assignment]

        result = await service.upsert_embedding(
            item_id=1, task_id="emb_001", vector=[0.1] * (EMBEDDING_DIM // 2),
            content_hash="sha256:a", model_name=settings.EMBEDDING_MODEL,
        )

        assert result["status"] == "error"
        assert str(settings.EMBEDDING_DIMENSIONS) in result["error"]
        assert result["task_id"] == "emb_001"
        assert http_calls == []

    @pytest.mark.asyncio
    async def test_empty_vector_is_rejected_without_an_http_call(self) -> None:
        """Protects against an empty list being written as a zero-length row."""
        service = DjangoAPIService()

        class TripwireClient:
            async def post(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("an empty vector must never be sent over the wire")

        async def get_client() -> Any:
            return TripwireClient()

        service.get_client = get_client  # type: ignore[assignment]

        result = await service.upsert_embedding(
            item_id=1, task_id="emb_001", vector=[], content_hash="sha256:a", model_name="m",
        )

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_correctly_sized_vector_is_accepted(self, mock_django_service: DjangoAPIService) -> None:
        """Protects the happy path so the guard cannot degenerate into blanket rejection."""
        result = await mock_django_service.upsert_embedding(
            item_id=3, task_id="emb_003", vector=deterministic_vector(),
            content_hash="sha256:c", model_name=settings.EMBEDDING_MODEL,
        )

        assert result["status"] == "success"


# ==============================================================================
# /status
# ==============================================================================

class TestEmbeddingsStatusEndpoint:
    """Protects the cheap operational-visibility endpoint."""

    def test_status_reports_the_outbox_depth(
        self,
        sync_client: TestClient,
        valid_internal_headers: dict[str, str],
        wired_services: tuple[RecordingDjangoService, RecordingEmbeddingService],
    ) -> None:
        """Protects the only signal an operator has that ingestion is keeping up."""
        response = sync_client.get(STATUS_URL, headers=valid_internal_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["pending"] == 4
        assert body["batch_limit"] == settings.EMBEDDING_BATCH_LIMIT
        assert body["model_name"] == settings.EMBEDDING_MODEL

    def test_status_degrades_instead_of_raising_when_the_outbox_is_down(
        self,
        sync_client: TestClient,
        valid_internal_headers: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Protects the health surface from becoming a 500 during the incident it reports."""
        class BrokenOutbox(RecordingDjangoService):
            async def get_pending_embeddings(self, limit: int = 20) -> dict[str, Any]:
                raise RuntimeError("outbox table unreachable")

        monkeypatch.setattr(internal_embeddings, "get_django_api_service", lambda: BrokenOutbox())
        monkeypatch.setattr(internal_embeddings, "get_embedding_service", lambda: RecordingEmbeddingService())

        response = sync_client.get(STATUS_URL, headers=valid_internal_headers)

        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert response.json()["pending"] is None
