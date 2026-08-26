"""Internal embeddings ingestion router (Fase 6).

Drains the Django embeddings outbox: pull pending tasks, embed each product's text with
the `RETRIEVAL_DOCUMENT` task type (the ingestion half of the asymmetric retrieval pair),
and write the resulting vector back into the pgvector index.

Every route is protected by the shared internal service secret: these endpoints are
service-to-service only and must never be reachable from a browser.
"""
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.core.config import settings
from app.core.security import verify_internal_api_secret
from app.services.django_api import get_django_api_service
from app.services.embeddings import get_embedding_service

logger = logging.getLogger("ai_gateway.api.internal_embeddings")

router = APIRouter(prefix="/internal/embeddings", tags=["Internal — Embeddings Ingestion"])


async def process_pending_embeddings(limit: Optional[int] = None) -> dict[str, Any]:
    """Drains one batch of the Django embeddings outbox.

    Each task is isolated in its own try/except: a single poison record (unembeddable
    text, wrong dimensionality, provider rejection) is marked as failed and the loop moves
    on, so it can never stall the whole outbox behind it. Marking the failure is itself
    guarded, because the mark-error call can fail too.

    Args:
        limit: Maximum number of pending tasks to pull. Defaults to
            `settings.EMBEDDING_BATCH_LIMIT`.

    Returns:
        A summary dict with `status`, `processed`, `failed`, `total` and `elapsed_ms`.
    """
    start_time = time.perf_counter()
    batch_limit = int(limit or settings.EMBEDDING_BATCH_LIMIT)

    django_service = get_django_api_service()
    embedding_service = get_embedding_service()

    processed = 0
    failed = 0
    tasks: list[dict[str, Any]] = []

    try:
        pending = await django_service.get_pending_embeddings(limit=batch_limit)
        tasks = list(pending.get("tasks") or []) if isinstance(pending, dict) else []
    except Exception as exc:
        logger.error("Failed to pull pending embedding tasks: %s", exc, exc_info=True)
        return {
            "status": "error",
            "processed": 0,
            "failed": 0,
            "total": 0,
            "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "error": str(exc),
        }

    logger.info("Embeddings ingestion run started: %d pending task(s).", len(tasks))

    for task in tasks:
        task_id = str(task.get("task_id", ""))
        try:
            item_id = int(task.get("item_id"))
            text = str(task.get("text") or "")

            # Ingestion side of the asymmetric pair: documents are RETRIEVAL_DOCUMENT.
            vector = await embedding_service.embed_document(text)

            result = await django_service.upsert_embedding(
                item_id=item_id,
                task_id=task_id,
                vector=vector,
                content_hash=str(task.get("content_hash") or ""),
                model_name=settings.EMBEDDING_MODEL,
            )

            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(str(result.get("error", "Upsert rejected the vector.")))

            processed += 1

        except Exception as exc:
            failed += 1
            logger.warning("Embedding task '%s' failed: %s", task_id, exc)
            try:
                await django_service.mark_embedding_error(task_id=task_id, error=str(exc))
            except Exception as mark_exc:
                # A failure to record the failure must not abort the remaining batch.
                logger.error("Could not mark embedding task '%s' as failed: %s", task_id, mark_exc)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    overall_status = "success" if failed == 0 else ("error" if processed == 0 and failed else "partial")

    logger.info(
        "Embeddings ingestion run finished: %d processed, %d failed in %.2f ms.",
        processed, failed, elapsed_ms,
    )

    return {
        "status": overall_status,
        "processed": processed,
        "failed": failed,
        "total": len(tasks),
        "elapsed_ms": elapsed_ms,
    }


@router.post(
    "/wake",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Wake the Embeddings Ingestion Worker",
    description=(
        "Webhook called by Django right after a catalog write. Queues the ingestion run as a "
        "background task and returns 202 immediately — the caller must never block on embedding work."
    ),
)
async def wake_embeddings_worker(
    background_tasks: BackgroundTasks,
    _authorized: bool = Depends(verify_internal_api_secret),
) -> dict[str, Any]:
    """Schedules an ingestion run without awaiting it.

    Args:
        background_tasks: FastAPI background task registry.
        _authorized: Result of the internal secret verification dependency.

    Returns:
        An acknowledgement payload.
    """
    background_tasks.add_task(process_pending_embeddings)
    logger.info("Embeddings ingestion wake-up received; run queued in the background.")
    return {"status": "accepted", "queued": True}


@router.post(
    "/process-pending",
    status_code=status.HTTP_200_OK,
    summary="Process the Pending Embeddings Batch",
    description=(
        "Synchronously drains one batch of the embeddings outbox and returns the run summary. "
        "This is the endpoint the GitHub Actions keep-alive cron hits every 10 minutes, which bounds "
        "index staleness even if a `/wake` webhook is lost to a cold start."
    ),
)
async def process_pending_endpoint(
    limit: Optional[int] = None,
    _authorized: bool = Depends(verify_internal_api_secret),
) -> dict[str, Any]:
    """Runs one ingestion batch and returns its summary.

    Args:
        limit: Optional override of the batch size.
        _authorized: Result of the internal secret verification dependency.

    Returns:
        The ingestion run summary dict.
    """
    return await process_pending_embeddings(limit=limit)


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Embeddings Outbox Status",
    description="Cheap operational visibility: how many tasks are currently pending ingestion.",
)
async def embeddings_status(
    _authorized: bool = Depends(verify_internal_api_secret),
) -> dict[str, Any]:
    """Reports the current depth of the embeddings outbox.

    Args:
        _authorized: Result of the internal secret verification dependency.

    Returns:
        A dict with `status`, `pending`, `batch_limit` and the active `model_name`.
    """
    django_service = get_django_api_service()
    embedding_service = get_embedding_service()

    try:
        pending = await django_service.get_pending_embeddings(limit=settings.EMBEDDING_BATCH_LIMIT)
        tasks = list(pending.get("tasks") or []) if isinstance(pending, dict) else []
        pending_count = int(pending.get("count", len(tasks))) if isinstance(pending, dict) else len(tasks)
    except Exception as exc:
        logger.warning("Could not read the embeddings outbox status: %s", exc)
        return {
            "status": "error",
            "pending": None,
            "batch_limit": settings.EMBEDDING_BATCH_LIMIT,
            "model_name": settings.EMBEDDING_MODEL,
            "embedding_service_available": embedding_service.is_available,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "pending": pending_count,
        "batch_limit": settings.EMBEDDING_BATCH_LIMIT,
        "model_name": settings.EMBEDDING_MODEL,
        "embedding_service_available": embedding_service.is_available,
    }
