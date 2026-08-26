"""Graceful degradation layer for catalog retrieval (Fase 5).

Orchestrates the RAG happy path — embed the user query with `RETRIEVAL_QUERY`, then
run a pgvector similarity search — and falls back to the legacy lexical engine when
anything in that chain fails. Callers of these helpers never see an exception: a
degraded answer beats no answer at all in the chat path.
"""
import logging
from typing import Any, Optional

from app.services.django_api import get_django_api_service
from app.services.embeddings import get_embedding_service

logger = logging.getLogger("ai_gateway.catalog_search")


async def _lexical_degrade(
    django_service: Any,
    query: str,
    reason: str,
    top_k: int = 8,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    in_stock_only: bool = True,
) -> dict[str, Any]:
    """Runs the lexical engine and tags the result as degraded.

    Args:
        django_service: The Django API service (real or injected mock).
        query: Text to search with the keyword engine.
        reason: Short Spanish explanation shown in `degraded_reason`.
        top_k: Maximum number of items to return.
        min_price: Inclusive lower price bound.
        max_price: Inclusive upper price bound.
        category: Category filter.
        brand: Brand filter.
        in_stock_only: When True, only returns items with available stock.

    Returns:
        The lexical result with `status="degraded"`, or an error dict when the
        lexical engine also fails.
    """
    try:
        result = await django_service.legacy_lexical_search(
            query=query,
            top_k=top_k,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )
        if not isinstance(result, dict):
            raise TypeError(f"legacy_lexical_search returned {type(result).__name__}, expected dict.")

        result["status"] = "degraded"
        result["degraded_reason"] = reason
        result["fallback_engine"] = "lexical"
        return result
    except Exception as exc:
        logger.error("Lexical fallback also failed for query '%s': %s", query, exc)
        return {
            "status": "error",
            "error": f"Búsqueda no disponible temporalmente: {exc}",
            "items": [],
        }


async def semantic_catalog_search_with_fallback(
    query: str,
    top_k: int = 8,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    in_stock_only: bool = True,
    embedding_service: Optional[Any] = None,
    django_service: Optional[Any] = None,
) -> dict[str, Any]:
    """Searches the catalog with pgvector, degrading to lexical search on any failure.

    Args:
        query: The natural language text the user typed.
        top_k: Maximum number of items to return.
        min_price: Inclusive lower price bound.
        max_price: Inclusive upper price bound.
        category: Category filter.
        brand: Brand filter.
        in_stock_only: When True, only returns items with available stock.
        embedding_service: Injectable embedding service (defaults to the singleton).
        django_service: Injectable Django API service (defaults to the singleton).

    Returns:
        The vector search response on success (`status="success"`), the lexical
        response tagged `status="degraded"` when the vector path failed, or
        `{"status": "error", "error": ..., "items": []}` when both engines failed.
        Items are passed through untouched: the canonical contract (`id`, `title`,
        `slug`, `price`, `stock`, `brand`, `category`) is shaped once inside
        `django_api._shape_catalog_item` and never remapped here.
        This function never raises.
    """
    django = django_service or get_django_api_service()
    embedder = embedding_service or get_embedding_service()

    if not query or not query.strip():
        return {
            "status": "error",
            "error": "La consulta de búsqueda está vacía.",
            "items": [],
        }

    clean_query = query.strip()
    degraded_reason: Optional[str] = None

    try:
        # Query side of the asymmetric pair: RETRIEVAL_QUERY, never RETRIEVAL_DOCUMENT.
        query_vector = await embedder.embed_query(clean_query)
    except Exception as exc:
        logger.warning("Query embedding failed for '%s': %s. Degrading to lexical search.", clean_query, exc)
        degraded_reason = "No se pudo generar el embedding de la consulta; se usó búsqueda por palabras clave."
        return await _lexical_degrade(
            django_service=django,
            query=clean_query,
            reason=degraded_reason,
            top_k=top_k,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )

    try:
        result = await django.vector_search(
            query_vector=query_vector,
            query_text=clean_query,
            top_k=top_k,
            min_price=min_price,
            max_price=max_price,
            category=category,
            brand=brand,
            in_stock_only=in_stock_only,
        )
        if not isinstance(result, dict):
            raise TypeError(f"vector_search returned {type(result).__name__}, expected dict.")

        # A non-success status is a failure too, not just a raised exception.
        if result.get("status") != "success":
            degraded_reason = (
                "La búsqueda vectorial respondió con estado "
                f"'{result.get('status')}'; se usó búsqueda por palabras clave."
            )
        else:
            return result
    except Exception as exc:
        logger.warning("Vector search failed for '%s': %s. Degrading to lexical search.", clean_query, exc)
        degraded_reason = "El motor vectorial no está disponible; se usó búsqueda por palabras clave."

    return await _lexical_degrade(
        django_service=django,
        query=clean_query,
        reason=degraded_reason or "Búsqueda vectorial no disponible.",
        top_k=top_k,
        min_price=min_price,
        max_price=max_price,
        category=category,
        brand=brand,
        in_stock_only=in_stock_only,
    )


async def find_similar_products_with_fallback(
    item_id: int,
    top_k: int = 5,
    exclude_out_of_stock: bool = True,
    django_service: Optional[Any] = None,
) -> dict[str, Any]:
    """Finds neighbours of a product, degrading to lexical search on any failure.

    Args:
        item_id: Primary key of the reference product.
        top_k: Maximum number of neighbours to return.
        exclude_out_of_stock: When True, drops neighbours with zero stock.
        django_service: Injectable Django API service (defaults to the singleton).

    Returns:
        The similarity response on success, a lexical response seeded from the
        reference product's title and tagged `status="degraded"` on failure, or an
        error dict when everything failed. Items are passed through untouched, in the
        canonical contract shaped by `django_api._shape_catalog_item`.
        This function never raises.
    """
    django = django_service or get_django_api_service()
    degraded_reason: Optional[str] = None

    try:
        result = await django.find_similar_products(
            item_id=item_id,
            top_k=top_k,
            exclude_out_of_stock=exclude_out_of_stock,
        )
        if not isinstance(result, dict):
            raise TypeError(f"find_similar_products returned {type(result).__name__}, expected dict.")

        if result.get("status") != "success":
            degraded_reason = (
                "El motor de similitud respondió con estado "
                f"'{result.get('status')}'; se usó búsqueda por palabras clave."
            )
        else:
            return result
    except Exception as exc:
        logger.warning("Vector similarity failed for item %s: %s. Degrading to lexical search.", item_id, exc)
        degraded_reason = "El motor de similitud vectorial no está disponible; se usó búsqueda por palabras clave."

    # Seed the lexical query with the reference product's own title. `title` is the
    # canonical field (Django's own); `name` is only read as the deprecated mirror, for
    # payloads produced before the canonical catalog contract landed.
    seed_query = f"producto {item_id}"
    try:
        verification = await django.verify_items(item_ids=[item_id])
        items = verification.get("items") or [] if isinstance(verification, dict) else []
        if items:
            reference_title = items[0].get("title") or items[0].get("name")
            if reference_title:
                seed_query = str(reference_title)
    except Exception as exc:
        logger.debug("Could not resolve reference item %s title for lexical seed: %s", item_id, exc)

    degraded = await _lexical_degrade(
        django_service=django,
        query=seed_query,
        reason=degraded_reason or "Similitud vectorial no disponible.",
        top_k=top_k,
        in_stock_only=exclude_out_of_stock,
    )

    if degraded.get("status") == "degraded":
        degraded["reference_item_id"] = item_id
        # The reference product is never its own recommendation.
        degraded["items"] = [item for item in degraded.get("items", []) if item.get("id") != item_id][:top_k]
        degraded["count"] = len(degraded["items"])
    return degraded
