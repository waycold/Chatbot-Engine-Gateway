"""Tests for the catalog retrieval degradation layer (Fase 5).

The contract these tests protect is blunt: `semantic_catalog_search_with_fallback` and
`find_similar_products_with_fallback` are called from inside the model's tool loop, and
an exception escaping either of them takes down the user's whole chat turn. A degraded
answer beats no answer — but a degraded answer that does not SAY it is degraded is
worse than both, because the shopper decides based on a catalog slice they believe is
complete. So every test here asserts two things: no exception escaped, and the
degradation is visible in the payload.
"""
from typing import Any
import pytest

from app.services.catalog_search import (
    find_similar_products_with_fallback,
    semantic_catalog_search_with_fallback,
)
from tests.conftest import (
    FailingEmbeddingService,
    FailingVectorDjangoService,
    StubEmbeddingService,
)


# ==============================================================================
# Local Django doubles
# ==============================================================================

class HealthyDjangoService:
    """Django double whose vector engine is healthy; its lexical engine must stay idle."""

    def __init__(self) -> None:
        self.vector_calls: list[dict[str, Any]] = []
        self.lexical_called = False

    async def vector_search(self, **kwargs: Any) -> dict[str, Any]:
        """Serves a healthy pgvector response, honouring price filters."""
        self.vector_calls.append(kwargs)
        items = [
            {"id": 3, "name": "Curso FastAPI", "category": "Cursos", "brand": "Academy Pro",
             "price": 49.99, "stock": 50, "in_stock": True, "similarity": 0.93},
            {"id": 4, "name": "Módulo LLM", "category": "Software", "brand": "GenAI Labs",
             "price": 89.00, "stock": 25, "in_stock": True, "similarity": 0.88},
        ]
        if kwargs.get("min_price") is not None:
            items = [item for item in items if item["price"] >= kwargs["min_price"]]
        if kwargs.get("max_price") is not None:
            items = [item for item in items if item["price"] <= kwargs["max_price"]]
        if kwargs.get("category"):
            items = [item for item in items if item["category"] == kwargs["category"]]
        if kwargs.get("brand"):
            items = [item for item in items if item["brand"] == kwargs["brand"]]
        if kwargs.get("in_stock_only"):
            items = [item for item in items if item["in_stock"]]
        return {
            "status": "success",
            "count": len(items),
            "items": items,
            "engine": "pgvector",
            "filters_applied": dict(kwargs),
        }

    async def find_similar_products(self, **kwargs: Any) -> dict[str, Any]:
        """Serves a healthy similarity response."""
        self.vector_calls.append(kwargs)
        return {
            "status": "success",
            "reference_item_id": kwargs.get("item_id"),
            "count": 1,
            "items": [{"id": 4, "name": "Módulo LLM", "similarity": 0.88}],
            "engine": "pgvector",
        }

    async def legacy_lexical_search(self, **kwargs: Any) -> dict[str, Any]:
        """Must never be reached while the vector engine is healthy."""
        self.lexical_called = True
        raise AssertionError("the lexical engine must not run when vector search succeeds")

    async def verify_items(self, **kwargs: Any) -> dict[str, Any]:
        """Resolves reference item names."""
        return {"status": "success", "items": [{"id": 3, "name": "Curso FastAPI"}], "not_found": []}


class SilentErrorDjangoService(FailingVectorDjangoService):
    """Vector engine that returns `{"status": "error"}` instead of raising."""

    async def vector_search(self, **kwargs: Any) -> dict[str, Any]:
        """Returns a non-success status without raising."""
        self.vector_calls.append(kwargs)
        return {"status": "error", "error": "vector index is rebuilding", "items": []}

    async def find_similar_products(self, **kwargs: Any) -> dict[str, Any]:
        """Returns a non-success status without raising."""
        self.vector_calls.append(kwargs)
        return {"status": "error", "error": "similarity index is rebuilding", "items": []}


class TotallyDownDjangoService(FailingVectorDjangoService):
    """Both engines unavailable — the worst case the caller must still survive."""

    async def legacy_lexical_search(self, **kwargs: Any) -> dict[str, Any]:
        """The keyword engine is down too."""
        self.lexical_calls.append(kwargs)
        raise RuntimeError("catalog database unreachable")


class GarbageDjangoService(FailingVectorDjangoService):
    """Vector engine returning a non-dict payload (a contract violation upstream)."""

    async def vector_search(self, **kwargs: Any) -> Any:
        """Returns a string where a dict is required."""
        self.vector_calls.append(kwargs)
        return "<html>502 Bad Gateway</html>"


# ==============================================================================
# semantic_catalog_search_with_fallback
# ==============================================================================

class TestSemanticCatalogSearchHappyPath:
    """Protects the non-degraded path from being accidentally marked degraded."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_success(self, mock_embedding_service: StubEmbeddingService) -> None:
        """Protects the healthy path: pgvector results pass through as status='success'."""
        django = HealthyDjangoService()

        result = await semantic_catalog_search_with_fallback(
            query="algo para aprender microservicios",
            embedding_service=mock_embedding_service,
            django_service=django,
        )

        assert result["status"] == "success"
        assert result["items"], "the happy path must return items"
        assert django.lexical_called is False

    @pytest.mark.asyncio
    async def test_happy_path_carries_no_degradation_markers(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects the disclosure contract from crying wolf.

        The e-commerce system prompt forces the reply to OPEN with a technical-problem
        warning whenever any tool reports `degraded`. Marking a healthy search as
        degraded would put that warning in front of every shopper.
        """
        result = await semantic_catalog_search_with_fallback(
            query="curso",
            embedding_service=mock_embedding_service,
            django_service=HealthyDjangoService(),
        )

        assert "degraded_reason" not in result
        assert "fallback_engine" not in result

    @pytest.mark.asyncio
    async def test_query_is_embedded_with_retrieval_query_task_type(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects the query half of the asymmetric retrieval pair.

        Embedding the query as RETRIEVAL_DOCUMENT would still return results — just
        measurably worse ones — with nothing in the logs to explain why.
        """
        await semantic_catalog_search_with_fallback(
            query="  algo para microservicios  ",
            embedding_service=mock_embedding_service,
            django_service=HealthyDjangoService(),
        )

        assert mock_embedding_service.calls[-1]["task_type"] == "RETRIEVAL_QUERY"
        assert mock_embedding_service.calls[-1]["text"] == "algo para microservicios"

    @pytest.mark.asyncio
    async def test_empty_query_is_rejected_without_calling_any_engine(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects against burning an embedding call on a whitespace-only query."""
        django = HealthyDjangoService()

        result = await semantic_catalog_search_with_fallback(
            query="   ", embedding_service=mock_embedding_service, django_service=django,
        )

        assert result["status"] == "error"
        assert result["items"] == []
        assert mock_embedding_service.calls == []
        assert django.vector_calls == []


class TestSemanticCatalogSearchFilters:
    """Protects the hard metadata filters that keep the model from over-promising."""

    @pytest.mark.parametrize(
        "filters",
        [
            {"min_price": 50.0},
            {"max_price": 60.0},
            {"category": "Cursos"},
            {"brand": "GenAI Labs"},
            {"in_stock_only": False},
            {"min_price": 10.0, "max_price": 99.0, "category": "Cursos", "brand": "Academy Pro"},
        ],
    )
    @pytest.mark.asyncio
    async def test_filters_are_forwarded_to_vector_search(
        self, mock_embedding_service: StubEmbeddingService, filters: dict[str, Any]
    ) -> None:
        """Protects filter forwarding: a dropped filter yields confidently wrong answers.

        If `max_price` is silently lost the model happily recommends a product outside
        the budget the customer just stated, and nothing anywhere reports an error.
        """
        django = HealthyDjangoService()

        await semantic_catalog_search_with_fallback(
            query="curso", embedding_service=mock_embedding_service, django_service=django, **filters,
        )

        forwarded = django.vector_calls[-1]
        for key, value in filters.items():
            assert forwarded[key] == value, f"filter '{key}' was not forwarded verbatim"

    @pytest.mark.asyncio
    async def test_filters_are_honoured_by_the_engine_response(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects against a vacuous forwarding test: the filter must change the result."""
        result = await semantic_catalog_search_with_fallback(
            query="curso",
            max_price=60.0,
            embedding_service=mock_embedding_service,
            django_service=HealthyDjangoService(),
        )

        assert result["items"], "the filtered search should still match something"
        assert all(item["price"] <= 60.0 for item in result["items"])

    @pytest.mark.asyncio
    async def test_top_k_is_forwarded(self, mock_embedding_service: StubEmbeddingService) -> None:
        """Protects the result-count cap that bounds prompt size and latency."""
        django = HealthyDjangoService()

        await semantic_catalog_search_with_fallback(
            query="curso", top_k=3, embedding_service=mock_embedding_service, django_service=django,
        )

        assert django.vector_calls[-1]["top_k"] == 3


class TestSemanticCatalogSearchDegradation:
    """Protects every branch of the degradation ladder."""

    @pytest.mark.asyncio
    async def test_embedding_failure_degrades_to_lexical(
        self,
        failing_embedding_service: FailingEmbeddingService,
        failing_django_service: FailingVectorDjangoService,
    ) -> None:
        """Protects against a dead embedding provider taking the catalog offline.

        This is the single most likely real incident: Gemini quota exhausted, and the
        shop must keep answering product questions from the keyword engine.
        """
        result = await semantic_catalog_search_with_fallback(
            query="algo para aprender fastapi",
            embedding_service=failing_embedding_service,
            django_service=failing_django_service,
        )

        assert result["status"] == "degraded"
        assert result["degraded_reason"], "a degraded result must explain itself to the model"
        assert result["fallback_engine"] == "lexical"
        assert result["items"], "degraded still means answered, not empty"
        assert failing_django_service.lexical_calls, "the lexical engine must have been used"

    @pytest.mark.asyncio
    async def test_vector_search_exception_degrades_to_lexical(
        self,
        mock_embedding_service: StubEmbeddingService,
        failing_django_service: FailingVectorDjangoService,
    ) -> None:
        """Protects against a raised pgvector error escaping into the chat turn."""
        result = await semantic_catalog_search_with_fallback(
            query="curso fastapi",
            embedding_service=mock_embedding_service,
            django_service=failing_django_service,
        )

        assert result["status"] == "degraded"
        assert result["fallback_engine"] == "lexical"
        assert result["items"]

    @pytest.mark.asyncio
    async def test_vector_search_error_status_also_degrades(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects against a SILENT failure passing straight through to the customer.

        An engine that answers `{"status": "error", "items": []}` without raising is the
        nastiest case: an early implementation would happily return that untouched, and
        the user would be told "no tenemos nada así" when in fact the index was down.
        """
        django = SilentErrorDjangoService()

        result = await semantic_catalog_search_with_fallback(
            query="curso fastapi", embedding_service=mock_embedding_service, django_service=django,
        )

        assert result["status"] == "degraded", "a non-success status must degrade, not pass through"
        assert "error" in result["degraded_reason"]
        assert result["items"], "the lexical engine still had results to serve"
        assert django.lexical_calls

    @pytest.mark.asyncio
    async def test_non_dict_vector_payload_degrades(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects against an HTML error page from a proxy being treated as results."""
        result = await semantic_catalog_search_with_fallback(
            query="curso", embedding_service=mock_embedding_service, django_service=GarbageDjangoService(),
        )

        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_both_engines_failing_returns_error_without_raising(
        self, mock_embedding_service: StubEmbeddingService
    ) -> None:
        """Protects the hard guarantee: this function NEVER raises.

        A raised exception here propagates through `execute_tool` into `run_tool_loop`,
        which abandons the tool loop entirely — so a database blip would silently strip
        the agent of all grounding rather than degrading one search.
        """
        django = TotallyDownDjangoService()

        result = await semantic_catalog_search_with_fallback(
            query="curso fastapi", embedding_service=mock_embedding_service, django_service=django,
        )

        assert result["status"] == "error"
        assert result["items"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_total_failure_with_failing_embedder_also_returns_error(
        self, failing_embedding_service: FailingEmbeddingService
    ) -> None:
        """Protects the compound failure mode: embeddings AND keyword search both down."""
        result = await semantic_catalog_search_with_fallback(
            query="curso",
            embedding_service=failing_embedding_service,
            django_service=TotallyDownDjangoService(),
        )

        assert result["status"] == "error"
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_filters_survive_the_degradation(
        self,
        failing_embedding_service: FailingEmbeddingService,
        failing_django_service: FailingVectorDjangoService,
    ) -> None:
        """Protects the customer's stated budget across the fallback boundary.

        Degrading is acceptable; degrading INTO an unfiltered result set is not — the
        model would then quote a product the customer explicitly ruled out.
        """
        result = await semantic_catalog_search_with_fallback(
            query="curso",
            max_price=60.0,
            category="Cursos",
            embedding_service=failing_embedding_service,
            django_service=failing_django_service,
        )

        assert result["status"] == "degraded"
        forwarded = failing_django_service.lexical_calls[-1]
        assert forwarded["max_price"] == 60.0
        assert forwarded["category"] == "Cursos"
        assert all(item["price"] <= 60.0 for item in result["items"])


# ==============================================================================
# find_similar_products_with_fallback
# ==============================================================================

class TestFindSimilarProductsFallback:
    """Protects the cross-sell path's identical degradation contract."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_success(self) -> None:
        """Protects the healthy similarity path from being marked degraded."""
        result = await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=HealthyDjangoService(),
        )

        assert result["status"] == "success"
        assert result["items"]

    @pytest.mark.asyncio
    async def test_similarity_exception_degrades_to_lexical(
        self, failing_django_service: FailingVectorDjangoService
    ) -> None:
        """Protects the cross-sell path when the similarity index is unavailable."""
        result = await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=failing_django_service,
        )

        assert result["status"] == "degraded"
        assert result["fallback_engine"] == "lexical"
        assert result["degraded_reason"]
        assert result["reference_item_id"] == 3

    @pytest.mark.asyncio
    async def test_similarity_error_status_also_degrades(self) -> None:
        """Protects against a silent non-success status passing through unflagged."""
        result = await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=SilentErrorDjangoService(),
        )

        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_reference_product_is_excluded_from_its_own_recommendations(
        self, failing_django_service: FailingVectorDjangoService
    ) -> None:
        """Protects against recommending the product the customer is already looking at.

        The lexical seed is the reference product's own name, so it is the single most
        likely keyword match — the exclusion has to be explicit.
        """
        result = await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=failing_django_service,
        )

        assert all(item["id"] != 3 for item in result["items"])
        assert result["count"] == len(result["items"])

    @pytest.mark.asyncio
    async def test_lexical_seed_uses_the_reference_product_name(
        self, failing_django_service: FailingVectorDjangoService
    ) -> None:
        """Protects recommendation quality in the degraded path.

        Seeding the keyword search with "producto 3" instead of the product's real name
        returns plausible-looking but unrelated items — degraded quality that nothing
        surfaces as an error.
        """
        await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=failing_django_service,
        )

        assert failing_django_service.verify_calls, "the reference item name must be resolved"
        assert failing_django_service.lexical_calls[-1]["query"] == "Curso FastAPI"

    @pytest.mark.asyncio
    async def test_total_failure_returns_error_without_raising(self) -> None:
        """Protects the never-raises guarantee for the similarity path too."""
        result = await find_similar_products_with_fallback(
            item_id=3, top_k=5, django_service=TotallyDownDjangoService(),
        )

        assert result["status"] == "error"
        assert result["items"] == []


# ==============================================================================
# The canonical catalog item contract
# ==============================================================================
# Every catalog method in `DjangoAPIService` funnels its items through
# `_shape_catalog_item`, so exactly one item shape reaches the agents, the tool loop
# and ultimately the chat widget. These tests pin that shape from the OUTSIDE -- via
# the public methods -- because a contract that only holds for the shaping helper is a
# contract that a new method can silently bypass.

import json as _json
from pathlib import Path

from app.services.django_api import (
    CATALOG_FIXTURE_PATH,
    DjangoAPIService,
    _FALLBACK_CATALOG_ITEMS,
    _load_catalog_fixture,
    _resolve_fixture_path,
    _shape_catalog_item,
    _slugify,
)
from app.services import django_api as _django_api_module

# The 7 fields the architecture team declared REQUIRED on every catalog item.
CANONICAL_REQUIRED_FIELDS = ("id", "title", "slug", "price", "stock", "brand", "category")

# Fields that travel alongside them, including the deprecated `name` mirror.
CANONICAL_COMPANION_FIELDS = ("currency", "description", "in_stock", "name")

# Fixture products deliberately kept out of stock, so `in_stock_only` has something
# real to exclude instead of filtering an empty set and passing vacuously.
OUT_OF_STOCK_IDS = {9, 12}


def _fixture_document() -> dict[str, Any]:
    """Reads data/catalog_fixture.json straight from disk, bypassing every cache."""
    resolved = _resolve_fixture_path(CATALOG_FIXTURE_PATH)
    assert resolved.is_file(), f"catalog fixture not found at {resolved}"
    with resolved.open("r", encoding="utf-8") as handle:
        return _json.load(handle)


def assert_canonical_item(item: Any, context: str = "") -> None:
    """Asserts one item satisfies the canonical contract, with correct types."""
    assert isinstance(item, dict), f"{context}: item is {type(item).__name__}, not a dict"

    for field in CANONICAL_REQUIRED_FIELDS:
        assert field in item, f"{context}: canonical field '{field}' missing from {sorted(item)}"

    assert isinstance(item["id"], int), f"{context}: id must be an int, got {item['id']!r}"
    assert isinstance(item["title"], str) and item["title"].strip(), f"{context}: empty title"
    assert isinstance(item["slug"], str) and item["slug"].strip(), f"{context}: empty slug"
    assert isinstance(item["price"], float), f"{context}: price must be a float, got {item['price']!r}"
    assert isinstance(item["stock"], int) and not isinstance(item["stock"], bool), (
        f"{context}: stock must be an int, got {item['stock']!r}"
    )
    assert item["brand"] is None or isinstance(item["brand"], str), f"{context}: bad brand"
    assert item["category"] is None or isinstance(item["category"], str), f"{context}: bad category"

    assert isinstance(item["currency"], str) and item["currency"], f"{context}: empty currency"
    assert isinstance(item["description"], str), f"{context}: description must be a str"
    assert item["in_stock"] is (item["stock"] > 0), f"{context}: in_stock disagrees with stock"
    assert item["name"] == item["title"], f"{context}: the deprecated `name` mirror drifted from `title`"


def assert_canonical_items(items: Any, context: str = "") -> None:
    """Asserts a non-empty list of canonical items."""
    assert isinstance(items, list) and items, f"{context}: expected a non-empty item list"
    for index, item in enumerate(items):
        assert_canonical_item(item, f"{context}[{index}]")


@pytest.fixture
def offline_service() -> DjangoAPIService:
    """A real DjangoAPIService pointed at a dead port, so every local fallback runs.

    That is the path under test: these tests are about the shape the gateway itself
    produces from the shared fixture, not about what a Django that does not exist yet
    would return.
    """
    return DjangoAPIService(base_url="http://127.0.0.1:9")


class TestCanonicalCatalogItemContract:
    """Every catalog-returning method must emit the same 7-field canonical item."""

    @pytest.mark.asyncio
    async def test_search_catalog_items_are_canonical(self, offline_service: DjangoAPIService) -> None:
        """`search_catalog` is the oldest method and the likeliest to drift."""
        assert_canonical_items(await offline_service.search_catalog(query="curso", limit=5), "search_catalog")

    @pytest.mark.asyncio
    async def test_search_catalog_without_a_query_is_canonical(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The unranked branch shapes its items too."""
        assert_canonical_items(await offline_service.search_catalog(limit=18), "search_catalog(all)")

    @pytest.mark.asyncio
    async def test_semantic_catalog_search_items_are_canonical(
        self, offline_service: DjangoAPIService
    ) -> None:
        """Plus the score field, which rides alongside and never replaces a canonical key."""
        response = await offline_service.semantic_catalog_search(query="curso de fastapi", top_k=5)

        assert_canonical_items(response["items"], "semantic_catalog_search")
        for item in response["items"]:
            assert isinstance(item["semantic_score"], float)

    @pytest.mark.asyncio
    async def test_vector_search_items_are_canonical(self, offline_service: DjangoAPIService) -> None:
        """The pgvector path."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, query_text="curso", top_k=5
        )

        assert_canonical_items(response["items"], "vector_search")
        for item in response["items"]:
            assert isinstance(item["similarity"], float)

    @pytest.mark.asyncio
    async def test_find_similar_products_items_are_canonical(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The 'more like this' path."""
        response = await offline_service.find_similar_products(item_id=3, top_k=5)

        assert_canonical_items(response["items"], "find_similar_products")

    @pytest.mark.asyncio
    async def test_verify_items_returns_full_canonical_items(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The stock/price re-check hands back FULL items; the caller re-renders from them."""
        response = await offline_service.verify_items(item_ids=[1, 3, 7])

        assert_canonical_items(response["items"], "verify_items")
        assert response["not_found"] == []

    @pytest.mark.asyncio
    async def test_legacy_lexical_search_items_are_canonical(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The keyword engine the whole Fase 5 fallback layer degrades onto."""
        response = await offline_service.legacy_lexical_search(query="curso", top_k=5)

        assert_canonical_items(response["items"], "legacy_lexical_search")

    @pytest.mark.asyncio
    async def test_every_method_emits_the_same_field_set_for_the_same_product(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The point of a canonical contract: one product, one shape, whatever produced it.

        Divergence here is what forces the widget into `item.get("title") or
        item.get("name")` defensive reads, and that is exactly how a missing `slug`
        turns into a /product/None/ 404 in front of a customer.
        """
        canonical_keys = set(CANONICAL_REQUIRED_FIELDS) | set(CANONICAL_COMPANION_FIELDS)

        by_method: dict[str, dict[str, Any]] = {}
        verified = await offline_service.verify_items(item_ids=[3])
        by_method["verify_items"] = verified["items"][0]
        for item in await offline_service.search_catalog(query="FastAPI", limit=18):
            if item["id"] == 3:
                by_method["search_catalog"] = item
        semantic = await offline_service.semantic_catalog_search(query="FastAPI", top_k=18)
        for item in semantic["items"]:
            if item["id"] == 3:
                by_method["semantic_catalog_search"] = item
        vector = await offline_service.vector_search(query_vector=[0.1] * 768, query_text="FastAPI", top_k=18)
        for item in vector["items"]:
            if item["id"] == 3:
                by_method["vector_search"] = item

        assert set(by_method) == {
            "verify_items", "search_catalog", "semantic_catalog_search", "vector_search",
        }, f"item 3 was not returned by every method: {sorted(by_method)}"

        for method, item in by_method.items():
            assert canonical_keys <= set(item), f"{method} is missing {canonical_keys - set(item)}"
            for field in CANONICAL_REQUIRED_FIELDS:
                assert item[field] == by_method["verify_items"][field], (
                    f"{method} disagrees with verify_items on '{field}'"
                )

    @pytest.mark.asyncio
    async def test_the_deprecated_name_mirror_is_still_emitted(
        self, offline_service: DjangoAPIService
    ) -> None:
        """Pins the deprecated mirror so its removal is a deliberate, visible change.

        The chat widget still reads `item.name`. Dropping the mirror silently would
        render every product card blank in production; this test makes that removal a
        red build with an explanation attached instead of a support ticket.
        """
        items = await offline_service.search_catalog(query="curso", limit=5)

        for item in items:
            assert "name" in item, (
                "the deprecated `name` mirror was removed. Django's field is `title`; "
                "`name` exists only so the current chat widget keeps rendering. Migrate "
                "the widget first, then delete this test in the same change."
            )
            assert item["name"] == item["title"]

    def test_shaping_is_idempotent(self) -> None:
        """Re-shaping an already-shaped item must be a no-op, not a slug regeneration."""
        raw = _load_catalog_fixture()[0]

        once = _shape_catalog_item(raw)
        twice = _shape_catalog_item(dict(once))

        assert once == twice

    def test_a_payload_without_a_slug_derives_one_rather_than_emitting_none(self) -> None:
        """A legacy payload must never produce `slug: None` -- /product/None/ is a 404."""
        shaped = _shape_catalog_item({"id": 99, "title": "Consultoría DevOps", "price": 1, "stock": 1})

        assert shaped["slug"] == "consultoria-devops"


class TestFixtureSlugInvariant:
    """The 404-prevention invariant the chat widget depends on.

    The widget builds product links as `/product/<slug>/`. A slug that is not the real
    one renders a 404 for the customer, and a 404 arriving from a chatbot recommendation
    is indistinguishable from the product not existing. So the fixture is the authority
    on slugs, and the fixture's own slugs must be internally consistent.
    """

    def test_every_fixture_slug_equals_the_slugify_of_its_own_title(self) -> None:
        """Iterates the fixture file itself, not the loaded/cached copy."""
        items = _fixture_document()["items"]
        assert len(items) == 18, f"expected 18 fixture products, found {len(items)}"

        mismatches = [
            (item["id"], item["title"], item["slug"], _slugify(item["title"]))
            for item in items
            if item["slug"] != _slugify(item["title"])
        ]

        assert mismatches == [], (
            "fixture slugs disagree with slugify(title); each of these renders "
            f"/product/<slug>/ as a 404 for the customer: {mismatches}"
        )

    def test_every_fixture_slug_is_unique(self) -> None:
        """Two products sharing a slug means one of them is unreachable."""
        slugs = [item["slug"] for item in _fixture_document()["items"]]

        assert len(slugs) == len(set(slugs)), (
            f"duplicate fixture slugs: {sorted({s for s in slugs if slugs.count(s) > 1})}"
        )

    def test_every_fixture_id_is_unique_and_an_int(self) -> None:
        """Ids key the verify/stock path; a duplicate silently drops a product."""
        ids = [item["id"] for item in _fixture_document()["items"]]

        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == len(set(ids))

    def test_every_fixture_item_carries_the_canonical_source_fields(self) -> None:
        """The fixture is the source of truth; it must be able to satisfy the contract."""
        for item in _fixture_document()["items"]:
            for field in CANONICAL_REQUIRED_FIELDS:
                assert field in item, f"fixture item {item.get('id')} lacks '{field}'"

    def test_the_declared_out_of_stock_products_really_are_out_of_stock(self) -> None:
        """Keeps the `in_stock_only` filter tests from passing vacuously."""
        by_id = {item["id"]: item for item in _fixture_document()["items"]}

        for item_id in OUT_OF_STOCK_IDS:
            assert by_id[item_id]["stock"] == 0, f"fixture item {item_id} is no longer out of stock"

    @pytest.mark.asyncio
    async def test_a_slug_the_catalog_does_not_carry_is_reported_not_found(
        self, offline_service: DjangoAPIService
    ) -> None:
        """`verify_items` must not rewrite an unknown slug into a neighbouring product."""
        response = await offline_service.verify_items(slugs=["curso-de-fastapi-que-no-existe"])

        assert response["items"] == []
        assert response["not_found"] == ["curso-de-fastapi-que-no-existe"]

    @pytest.mark.asyncio
    async def test_real_fixture_slugs_resolve(self, offline_service: DjangoAPIService) -> None:
        """The positive half: the real slugs must genuinely verify."""
        slugs = [item["slug"] for item in _fixture_document()["items"][:5]]

        response = await offline_service.verify_items(slugs=slugs)

        assert response["not_found"] == []
        assert [item["slug"] for item in response["items"]] == slugs


class TestCatalogFilters:
    """Each filter must genuinely narrow the result set, not merely be accepted."""

    @pytest.mark.asyncio
    async def test_in_stock_only_excludes_the_zero_stock_products(
        self, offline_service: DjangoAPIService
    ) -> None:
        """Recommending an out-of-stock product is a checkout dead end for the customer."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, in_stock_only=True
        )
        returned_ids = {item["id"] for item in response["items"]}

        assert returned_ids & OUT_OF_STOCK_IDS == set(), (
            f"out-of-stock products were recommended: {returned_ids & OUT_OF_STOCK_IDS}"
        )
        assert all(item["in_stock"] is True for item in response["items"])

    @pytest.mark.asyncio
    async def test_in_stock_only_false_includes_them(self, offline_service: DjangoAPIService) -> None:
        """The negative control: without the filter the same products come back."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, in_stock_only=False
        )
        returned_ids = {item["id"] for item in response["items"]}

        assert OUT_OF_STOCK_IDS <= returned_ids, (
            "in_stock_only=False changed nothing -- the filter is not being applied at all"
        )

    @pytest.mark.asyncio
    async def test_min_price_filters(self, offline_service: DjangoAPIService) -> None:
        """A price floor the customer stated must be honoured."""
        response = await offline_service.vector_search(query_vector=[0.1] * 768, top_k=50, min_price=100.0)

        assert response["items"], "min_price filtered everything -- the assertion would be vacuous"
        assert all(item["price"] >= 100.0 for item in response["items"])

    @pytest.mark.asyncio
    async def test_max_price_filters(self, offline_service: DjangoAPIService) -> None:
        """A budget the customer stated must be honoured."""
        response = await offline_service.vector_search(query_vector=[0.1] * 768, top_k=50, max_price=60.0)

        assert response["items"]
        assert all(item["price"] <= 60.0 for item in response["items"])

    @pytest.mark.asyncio
    async def test_price_bounds_are_inclusive_and_really_exclude(
        self, offline_service: DjangoAPIService
    ) -> None:
        """Pins the boundary AND proves the bound removes something."""
        unfiltered = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, in_stock_only=False
        )
        all_prices = sorted({item["price"] for item in unfiltered["items"]})
        assert len(all_prices) > 1

        bounded = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, in_stock_only=False, max_price=all_prices[0]
        )

        assert bounded["items"], "the inclusive bound dropped the item sitting exactly on it"
        assert all(item["price"] == all_prices[0] for item in bounded["items"])
        assert len(bounded["items"]) < len(unfiltered["items"])

    @pytest.mark.asyncio
    async def test_brand_filters(self, offline_service: DjangoAPIService) -> None:
        """A brand filter must return that brand and nothing else."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, brand="Academy Pro", in_stock_only=False
        )

        assert response["items"]
        assert {item["brand"] for item in response["items"]} == {"Academy Pro"}

    @pytest.mark.asyncio
    async def test_category_filters(self, offline_service: DjangoAPIService) -> None:
        """Same for category."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, category="Cursos", in_stock_only=False
        )

        assert response["items"]
        assert {item["category"] for item in response["items"]} == {"Cursos"}

    @pytest.mark.asyncio
    async def test_a_filter_matching_nothing_returns_empty_not_everything(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The dangerous failure mode: an unmatched filter falling back to the full catalog."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, brand="Marca Que No Existe"
        )

        assert response["items"] == []

    @pytest.mark.asyncio
    async def test_filters_compose(self, offline_service: DjangoAPIService) -> None:
        """Combined filters must AND, never OR."""
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50,
            category="Cursos", brand="Academy Pro", max_price=100.0, in_stock_only=True,
        )

        assert response["items"]
        for item in response["items"]:
            assert item["category"] == "Cursos"
            assert item["brand"] == "Academy Pro"
            assert item["price"] <= 100.0
            assert item["in_stock"] is True
            assert item["id"] not in OUT_OF_STOCK_IDS


class TestCatalogFacetsAgreeWithTheCatalog:
    """Facets and search results can never disagree.

    A facet is a filter value offered to the customer. Offering one that yields zero
    results is a dead end inside the conversation, and the reverse -- a brand that
    exists but is never offered -- hides inventory.
    """

    @pytest.mark.asyncio
    async def test_facets_are_exactly_the_distinct_fixture_values(
        self, offline_service: DjangoAPIService
    ) -> None:
        """Derived from the same fixture the search methods return items from."""
        items = _fixture_document()["items"]
        expected_brands = sorted({item["brand"] for item in items if item.get("brand")})
        expected_categories = sorted({item["category"] for item in items if item.get("category")})

        facets = await offline_service.get_catalog_facets(facet="both")

        assert facets["brands"] == expected_brands
        assert facets["categories"] == expected_categories

    @pytest.mark.asyncio
    async def test_every_offered_facet_value_yields_at_least_one_product(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The end-to-end guarantee, asserted through the search path rather than the data."""
        facets = await offline_service.get_catalog_facets(facet="both")

        for brand in facets["brands"]:
            response = await offline_service.vector_search(
                query_vector=[0.1] * 768, top_k=50, brand=brand, in_stock_only=False
            )
            assert response["items"], f"facet brand '{brand}' matches no product"

        for category in facets["categories"]:
            response = await offline_service.vector_search(
                query_vector=[0.1] * 768, top_k=50, category=category, in_stock_only=False
            )
            assert response["items"], f"facet category '{category}' matches no product"

    @pytest.mark.asyncio
    async def test_no_product_carries_a_brand_or_category_the_facets_omit(
        self, offline_service: DjangoAPIService
    ) -> None:
        """The other direction: hidden inventory is just as broken as a dead-end filter."""
        facets = await offline_service.get_catalog_facets(facet="both")
        response = await offline_service.vector_search(
            query_vector=[0.1] * 768, top_k=50, in_stock_only=False
        )

        assert {item["brand"] for item in response["items"]} <= set(facets["brands"])
        assert {item["category"] for item in response["items"]} <= set(facets["categories"])

    @pytest.mark.parametrize("facet,expected_keys", [
        ("category", {"categories"}), ("brand", {"brands"}), ("both", {"categories", "brands"}),
    ])
    @pytest.mark.asyncio
    async def test_only_the_requested_facet_is_returned(
        self, offline_service: DjangoAPIService, facet: str, expected_keys: set
    ) -> None:
        """A caller asking for one facet must not be handed the other."""
        response = await offline_service.get_catalog_facets(facet=facet)

        assert {"categories", "brands"} & set(response) == expected_keys

    @pytest.mark.asyncio
    async def test_an_invalid_facet_is_an_error_not_a_silent_both(
        self, offline_service: DjangoAPIService
    ) -> None:
        """A typo must surface, not quietly widen the answer."""
        response = await offline_service.get_catalog_facets(facet="colour")

        assert response["status"] == "error"
        assert "categories" not in response and "brands" not in response


class TestFixtureLoaderDegradation:
    """The service must never fail to start because a data file moved or was truncated."""

    @pytest.fixture(autouse=True)
    def _reset_fixture_cache(self) -> Any:
        """Restores the module-level fixture cache around every test in this class."""
        original = _django_api_module._catalog_fixture_cache
        _django_api_module._catalog_fixture_cache = None
        yield
        _django_api_module._catalog_fixture_cache = original

    @staticmethod
    def _load_from(monkeypatch: pytest.MonkeyPatch, path: Any) -> list[dict[str, Any]]:
        """Points the loader at `path` and returns what it loaded."""
        monkeypatch.setattr(_django_api_module, "_resolve_fixture_path", lambda _target: Path(path))
        return _load_catalog_fixture()

    def test_a_missing_file_degrades_to_the_builtin_catalog(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A moved data file must log a warning, not crash the process at import time."""
        loaded = self._load_from(monkeypatch, tmp_path / "definitely-not-here.json")

        assert loaded == _FALLBACK_CATALOG_ITEMS
        assert loaded, "the built-in fallback catalog is empty -- degrading yields nothing"

    @pytest.mark.parametrize("content,label", [
        ("{ this is not json", "truncated json"),
        ("", "empty file"),
        ('{"version": 1}', "no items key"),
        ('{"version": 1, "items": []}', "empty items list"),
        ('{"version": 1, "items": "curso"}', "items is a string"),
        ('{"version": 1, "items": [1, 2, 3]}', "items holds no objects"),
        ('["curso"]', "document is a list"),
        ('null', "document is null"),
    ])
    def test_a_malformed_file_degrades_to_the_builtin_catalog(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, content: str, label: str
    ) -> None:
        """Every malformed shape degrades; none of them raises."""
        broken = tmp_path / "broken.json"
        broken.write_text(content, encoding="utf-8")

        loaded = self._load_from(monkeypatch, broken)

        assert loaded == _FALLBACK_CATALOG_ITEMS, f"{label} did not degrade cleanly"

    def test_the_degradation_is_logged_as_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A silent degradation to 3 products would look like an empty catalog in prod."""
        records: list[str] = []
        monkeypatch.setattr(
            _django_api_module.logger, "warning",
            lambda message, *args, **kwargs: records.append(str(message) % args if args else str(message)),
        )

        self._load_from(monkeypatch, tmp_path / "gone.json")

        assert records, "degrading to the built-in catalog was not logged"
        assert "fixture" in records[0].lower()

    @pytest.mark.asyncio
    async def test_the_service_still_serves_canonical_items_while_degraded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The contract holds on the degraded path too, or the fallback is not a fallback."""
        self._load_from(monkeypatch, tmp_path / "gone.json")
        service = DjangoAPIService(base_url="http://127.0.0.1:9")

        assert_canonical_items(await service.search_catalog(limit=5), "degraded search_catalog")

    def test_the_builtin_fallback_items_satisfy_the_canonical_contract(self) -> None:
        """The last-resort catalog must not itself be malformed."""
        for item in _FALLBACK_CATALOG_ITEMS:
            assert_canonical_item(_shape_catalog_item(item), f"builtin id={item.get('id')}")
            assert item["slug"] == _slugify(item["title"]), (
                f"built-in item {item['id']} carries a slug that is not slugify(title)"
            )
