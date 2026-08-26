"""Pytest configuration and shared fixtures for AI Agent Gateway test suite."""
import os

# ==============================================================================
# TEST ENVIRONMENT PINNING  --  read this before changing a single line below
# ==============================================================================
# These MUST be DIRECT ASSIGNMENTS (`os.environ[...] = ...`), never
# `os.environ.setdefault(...)`, and they MUST stay at the very top of this module,
# above every `app.*` import.
#
# WHY `setdefault` IS WRONG HERE
# ------------------------------
# `setdefault` means "the developer's own environment wins". Every developer running
# this suite has a real `.env` for local work against a live Django --
# `ENVIRONMENT="development"`, `DEBUG=true`,
# `INTERNAL_API_SECRET="django-insecure-..."`. With `setdefault` those values survive,
# Pydantic Settings loads THEM, and the suite silently exercises a different
# application than the one it asserts about. That cost 16 failures once already:
#
#   * The 4 embeddings endpoint tests send `test-internal-secret-67890` while the app
#     compares it against the developer's real secret, so every request 401s where the
#     test expects 200. The failure reads `assert 401 == 200` -- it says nothing at all
#     about the environment, which is exactly why it is so expensive to diagnose.
#   * `DJANGO_BACKEND_URL` keeps its local value, producing `127.0.0.1 != test-django`.
#   * Worst of all, `ENVIRONMENT=development` + `DEBUG=true` re-arms the development
#     escape hatch in `DjangoAPIService.validate_user_token`. The 7 fail-closed
#     security tests in `test_tool_authorization.py` exist precisely to prove that
#     hatch is shut; leaking a dev environment into the run makes them fail for a
#     reason that looks like a product regression and is not.
#
# A test suite has no business inheriting ambient configuration. It pins it.
os.environ["PROJECT_NAME"] = "AI Agent Gateway - Test"
os.environ["VERSION"] = "0.1.0-test"
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEBUG"] = "false"
os.environ["GEMINI_API_KEY"] = "test-mock-gemini-key-12345"
os.environ["INTERNAL_API_SECRET"] = "test-internal-secret-67890"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["DJANGO_BACKEND_URL"] = "http://test-django:8000"
os.environ["BACKEND_CORS_ORIGINS"] = '["http://localhost:3000","http://test-frontend:3000"]'

# The values the entire suite is written against. Named constants so the tripwire
# below, and the fixtures further down, cannot drift away from the block above.
TEST_ENVIRONMENT = "testing"
TEST_INTERNAL_SECRET = "test-internal-secret-67890"
TEST_DJANGO_BACKEND_URL = "http://test-django:8000"

from typing import AsyncGenerator, Generator, Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

import sys

# `get_settings` is an `lru_cache`d singleton AND `app/core/config.py` binds a
# module-level `settings = get_settings()` at import time, which every other module
# then imports by value (`from app.core.config import settings`). Two different things
# therefore have to be right:
#
#   1. the cache, so the NEXT `get_settings()` rebuilds from the pinned environment;
#   2. the already-created global, which `cache_clear()` cannot touch at all.
#
# If `app.core.config` was imported before this conftest ran -- a pytest plugin, an
# IDE runner, a stray import in `tests/__init__.py` -- that global was built from the
# developer's environment and is now frozen into every `app.*` module. Detecting that
# case is the entire point of the tripwire below, so the config module is imported
# through `sys.modules` first, before the cache is cleared, to see what is really there.
_config_was_already_imported = "app.core.config" in sys.modules

from app.agents.analytics import _auth_status_var
from app.core import config as _config_module
from app.core.config import Settings, get_settings

get_settings.cache_clear()

# ------------------------------------------------------------------------------
# Environment-leak tripwire
# ------------------------------------------------------------------------------
# The `os.environ` assignments above are sufficient on their own for every key the
# suite depends on: `Settings`' `env_file=".env"` has LOWER precedence than a real
# process environment variable in pydantic-settings v2, so an overridden key can no
# longer be taken from a developer `.env`. What a developer `.env` can still do is
# supply keys we do NOT override (say `DEFAULT_MODEL` or `OPENROUTER_API_KEY`), and a
# future edit could always drop a key from the block above by accident.
#
# So this stays as a tripwire rather than as the fix: it fails the collection of the
# whole run, loudly and with the actual cause named, instead of letting 16 tests fail
# with `assert 401 == 200`. One clear error beats sixteen mysterious ones.
#
# It deliberately inspects `app.core.config.settings` -- the module-level global the
# production code actually reads -- and NOT a fresh `get_settings()`. A fresh call
# would happily rebuild clean values from the pinned environment and report all-green
# while every `app.*` module still held a leaked object.
_live_settings = _config_module.settings
if (
    _live_settings.ENVIRONMENT != TEST_ENVIRONMENT
    or _live_settings.INTERNAL_API_SECRET != TEST_INTERNAL_SECRET
    or _live_settings.DEBUG is not False
    or _live_settings.DJANGO_BACKEND_URL != TEST_DJANGO_BACKEND_URL
):
    raise RuntimeError(
        "Test environment leak detected: the live application settings object "
        "(app.core.config.settings) does not match the values pinned at the top of "
        "tests/conftest.py.\n"
        f"  ENVIRONMENT         = {_live_settings.ENVIRONMENT!r} (expected {TEST_ENVIRONMENT!r})\n"
        f"  DEBUG               = {_live_settings.DEBUG!r} (expected False)\n"
        f"  DJANGO_BACKEND_URL  = {_live_settings.DJANGO_BACKEND_URL!r} "
        f"(expected {TEST_DJANGO_BACKEND_URL!r})\n"
        f"  INTERNAL_API_SECRET matches the test value: "
        f"{_live_settings.INTERNAL_API_SECRET == TEST_INTERNAL_SECRET}\n"
        f"  app.core.config was imported before this conftest: {_config_was_already_imported}\n"
        "\nYour own environment reached the test run. Most likely causes: a `.env` in "
        "the repository root supplying a key this file does not override; an exported "
        "shell variable; or an `app.*` module imported before this conftest, which "
        "would have frozen a Settings built from your environment into every module "
        "that does `from app.core.config import settings` (get_settings.cache_clear() "
        "cannot undo that -- it only affects future calls).\n"
        "Running the suite as `ENVIRONMENT=testing DEBUG=false "
        "INTERNAL_API_SECRET=test-internal-secret-67890 pytest` is a workaround; "
        "removing the leak is the fix."
    )

from app.main import create_application
from app.schemas.payload import ChatRequest, ChatResponse, HealthResponse
from app.services.django_api import DjangoAPIService
from app.services.embeddings import EmbeddingServiceError
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService

# Dimensionality of the stored pgvector embeddings. Hard-coded here (rather than read
# from settings) so a test that asserts on dimensionality fails loudly if the setting
# is ever changed without the index being rebuilt.
EMBEDDING_DIM = 768


@pytest.fixture(autouse=True)
def reset_analytics_auth_context() -> Generator[None, None, None]:
    """Clears the request-scoped analytics auth ContextVar around every test.

    `AnalyticsAgent` publishes its staff verdict through a module-level ContextVar. In
    production each request runs in its own asyncio task and therefore its own Context
    copy, but tests share a process: without this reset, a test that authenticates a
    staff user could leave the variable set and silently grant privileges to the next
    test, turning a real authorization regression into a green suite.
    """
    _auth_status_var.set(None)
    yield
    _auth_status_var.set(None)


# ==============================================================================
# Settings & Application Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Returns application test settings singleton."""
    return get_settings()


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """Instantiates a fresh FastAPI application instance for each test."""
    application = create_application()
    return application


@pytest.fixture
def sync_client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Synchronous test client for standard request-response endpoint testing."""
    with TestClient(app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Asynchronous HTTP test client using ASGITransport for streaming and async endpoints."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ==============================================================================
# Google GenAI SDK (google-genai) Mocks
# ==============================================================================

class MockGenAIChunk:
    """Simulates a streamed chunk response from Google GenAI."""
    def __init__(self, text: str):
        self.text = text


class MockGenAIResponse:
    """Simulates a synchronous or full response from Google GenAI."""
    def __init__(self, text: str, total_tokens: int = 42):
        self.text = text
        self.usage_metadata = MagicMock()
        self.usage_metadata.total_token_count = total_tokens
        self.usage_metadata.prompt_token_count = 20
        self.usage_metadata.candidates_token_count = 22


class MockEmbeddingValues:
    """Simulates a single `result.embeddings[i]` entry from the GenAI embedding API."""

    def __init__(self, values: list[float]) -> None:
        self.values = values


class MockEmbeddingResponse:
    """Simulates the response of `client.aio.models.embed_content(...)`.

    The shape matters: `EmbeddingService._extract_vector` reads
    `result.embeddings[0].values`, so any drift in that path must fail the tests.
    """

    def __init__(self, values: list[float] | None = None) -> None:
        self.embeddings = [MockEmbeddingValues(values if values is not None else deterministic_vector())]


def deterministic_vector(dimensions: int = EMBEDDING_DIM) -> list[float]:
    """Builds a reproducible, non-constant embedding vector.

    A constant vector would make a cosine-ordering bug invisible, so the values vary
    with the index while staying fully deterministic across runs.

    Args:
        dimensions: Length of the vector to build.

    Returns:
        A list of `dimensions` floats in (0, 1].
    """
    return [((index % 97) + 1) / 100.0 for index in range(dimensions)]


@pytest.fixture
def mock_genai_chunks() -> list[MockGenAIChunk]:
    """Provides a sequence of stream chunks."""
    return [
        MockGenAIChunk("Hola, "),
        MockGenAIChunk("soy el agente "),
        MockGenAIChunk("especializado. "),
        MockGenAIChunk("¿En qué te puedo colaborar hoy?"),
    ]


@pytest.fixture
def mock_genai_client(mock_genai_chunks: list[MockGenAIChunk]) -> MagicMock:
    """Provides a mocked Google GenAI SDK Client."""
    mock_client = MagicMock()

    # Mock standard non-streaming generation
    mock_client.models.generate_content.return_value = MockGenAIResponse(
        text="Hola, soy el agente de IA. ¿En qué puedo ayudarte?"
    )

    # Mock async / sync streaming generator
    async def async_stream_generator(*args: Any, **kwargs: Any):
        for chunk in mock_genai_chunks:
            yield chunk

    def sync_stream_generator(*args: Any, **kwargs: Any):
        for chunk in mock_genai_chunks:
            yield chunk

    mock_client.models.generate_content_stream = MagicMock(side_effect=sync_stream_generator)
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MockGenAIResponse("Respuesta asíncrona de IA")
    )
    mock_client.aio.models.generate_content_stream = MagicMock(side_effect=async_stream_generator)

    # Embedding API (Fase 2). `EmbeddingService._embed_once` prefers the async surface
    # (`client.aio.models.embed_content`) and falls back to the sync one, so both are
    # provided. Call kwargs are recorded by the mock, which is what the truncation and
    # task_type assertions in tests/test_embeddings.py inspect.
    mock_client.aio.models.embed_content = AsyncMock(return_value=MockEmbeddingResponse())
    mock_client.models.embed_content = MagicMock(return_value=MockEmbeddingResponse())

    return mock_client


# ==============================================================================
# Redis Memory Service Mocks
# ==============================================================================

class MockRedisBackend:
    """In-memory dictionary simulating Redis key-value and list operations."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                self.ttls.pop(k, None)
                count += 1
        return count

    async def rpush(self, key: str, *values: str) -> int:
        if key not in self.store:
            self.store[key] = []
        if not isinstance(self.store[key], list):
            self.store[key] = [self.store[key]]
        self.store[key].extend(values)
        return len(self.store[key])

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        items = self.store.get(key, [])
        if not isinstance(items, list):
            return []
        if stop == -1:
            return items[start:]
        return items[start:stop + 1]

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self.store:
            self.ttls[key] = seconds
            return True
        return False

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


@pytest.fixture
def mock_redis_backend() -> MockRedisBackend:
    """In-memory mock for Redis client."""
    return MockRedisBackend()


@pytest.fixture
def mock_memory_service(mock_redis_backend: MockRedisBackend) -> RedisMemoryService:
    """Mocked RedisMemoryService using the in-memory mock backend."""
    service = RedisMemoryService(redis_url="redis://localhost:6379/15")
    service._redis_client = mock_redis_backend
    return service


# ==============================================================================
# Django Monolith API Service Mocks
# ==============================================================================

@pytest.fixture
def mock_django_http_client() -> AsyncMock:
    """Mock for httpx.AsyncClient used by DjangoAPIService."""
    client = AsyncMock(spec=httpx.AsyncClient)

    # Default mock responses for Django endpoints
    mock_token_verify_resp = MagicMock(spec=httpx.Response)
    mock_token_verify_resp.status_code = 200
    # Canonical Django envelope: `{"valid": true, "user": {...}}`. `auth_user` has no
    # `role` column, so privilege is the two native booleans and nothing else. This
    # default identity is a plain CUSTOMER: neither staff nor superuser.
    mock_token_verify_resp.json.return_value = {
        "valid": True,
        "user": {
            "id": 101,
            "username": "test_user",
            "email": "user@example.com",
            "is_staff": False,
            "is_superuser": False,
        },
    }

    mock_products_resp = MagicMock(spec=httpx.Response)
    mock_products_resp.status_code = 200
    mock_products_resp.json.return_value = {
        "total_found": 2,
        "limit": 5,
        # Canonical item contract: `title` is the real field (Django's column name);
        # `name` is only ever emitted BY the gateway as a deprecated mirror, so the
        # payloads Django hands us are keyed on `title` and carry all 7 canonical
        # fields. Slugs are taken verbatim from data/catalog_fixture.json -- a guessed
        # slug renders /product/<slug>/ as a 404 for the customer.
        "items": [
            {"id": 1, "title": "Servicio Cloud AI", "slug": "servicio-cloud-ai",
             "price": 49.99, "stock": 10, "brand": "Cloud Ops Studio", "category": "Servicios"},
            {"id": 2, "title": "Consultoría DevOps", "slug": "consultoria-devops",
             "price": 120.00, "stock": 5, "brand": "Cloud Ops Studio", "category": "Servicios"},
        ],
        "results": [
            {"id": 1, "title": "Servicio Cloud AI", "slug": "servicio-cloud-ai",
             "price": 49.99, "stock": 10, "brand": "Cloud Ops Studio", "category": "Servicios"},
            {"id": 2, "title": "Consultoría DevOps", "slug": "consultoria-devops",
             "price": 120.00, "stock": 5, "brand": "Cloud Ops Studio", "category": "Servicios"},
        ],
    }

    mock_metrics_resp = MagicMock(spec=httpx.Response)
    mock_metrics_resp.status_code = 200
    mock_metrics_resp.json.return_value = {
        "daily_active_users": 1520,
        "conversion_rate": 3.8,
        "total_revenue": 54200.00,
    }

    # --------------------------------------------------------------------------
    # RAG / pgvector internal endpoints (Fase 1-3). These do not exist in Django yet;
    # the shapes below mirror the contract `DjangoAPIService` parses so a future
    # contract drift on the Django side breaks a test instead of production.
    # --------------------------------------------------------------------------
    _rag_items = [
        {
            "id": 3, "slug": "curso-avanzado-de-fastapi-microservicios",
            "title": "Curso Avanzado de FastAPI & Microservicios", "category": "Cursos",
            "brand": "Academy Pro", "price": 49.99, "currency": "USD", "stock": 50,
            "in_stock": True, "description": "FastAPI, Pydantic v2 y Docker.",
        },
        {
            "id": 4, "slug": "modulo-de-integracion-llm-agentes-autonomos",
            "title": "Módulo de Integración LLM & Agentes Autónomos", "category": "Software",
            "brand": "GenAI Labs", "price": 89.00, "currency": "USD", "stock": 25,
            "in_stock": True, "description": "Orquesta agentes multi-rol.",
        },
    ]

    mock_vector_search_resp = MagicMock(spec=httpx.Response)
    mock_vector_search_resp.status_code = 200
    mock_vector_search_resp.json.return_value = {
        "status": "success",
        "query": "curso de microservicios",
        "top_k": 8,
        "count": len(_rag_items),
        "items": [{**item, "similarity": 0.93 - index * 0.05} for index, item in enumerate(_rag_items)],
        "engine": "pgvector",
    }

    mock_similar_resp = MagicMock(spec=httpx.Response)
    mock_similar_resp.status_code = 200
    mock_similar_resp.json.return_value = {
        "status": "success",
        "reference_item_id": 3,
        "count": 1,
        "items": [{**_rag_items[1], "similarity": 0.88}],
        "engine": "pgvector",
    }

    mock_pending_resp = MagicMock(spec=httpx.Response)
    mock_pending_resp.status_code = 200
    mock_pending_resp.json.return_value = {
        "status": "success",
        "count": 2,
        "tasks": [
            {
                "task_id": "emb_task_003", "item_id": 3,
                "text": "Curso Avanzado de FastAPI. Categoría: Cursos. FastAPI, Pydantic v2 y Docker.",
                "content_hash": "sha256:aaa",
            },
            {
                "task_id": "emb_task_004", "item_id": 4,
                "text": "Módulo de Integración LLM. Categoría: Software. Orquesta agentes multi-rol.",
                "content_hash": "sha256:bbb",
            },
        ],
    }

    mock_upsert_resp = MagicMock(spec=httpx.Response)
    mock_upsert_resp.status_code = 200
    mock_upsert_resp.json.return_value = {
        "status": "success", "task_id": "emb_task_003", "item_id": 3, "dimensions": EMBEDDING_DIM,
    }

    mock_mark_error_resp = MagicMock(spec=httpx.Response)
    mock_mark_error_resp.status_code = 200
    mock_mark_error_resp.json.return_value = {
        "status": "success", "task_id": "emb_task_003", "marked": "error",
    }

    mock_verify_resp = MagicMock(spec=httpx.Response)
    mock_verify_resp.status_code = 200
    mock_verify_resp.json.return_value = {
        "status": "success",
        "checked_at": "2026-08-25T00:00:00+00:00",
        "items": [
            {
                "id": item["id"], "slug": item["slug"], "title": item["title"],
                "brand": item["brand"], "category": item["category"],
                "price": item["price"], "currency": item["currency"],
                "stock": item["stock"], "in_stock": item["in_stock"],
            }
            for item in _rag_items
        ],
        "not_found": [],
    }

    mock_facets_resp = MagicMock(spec=httpx.Response)
    mock_facets_resp.status_code = 200
    mock_facets_resp.json.return_value = {
        "status": "success",
        "facet": "both",
        "categories": ["Cursos", "Servicios", "Software", "Templates"],
        "brands": ["Academy Pro", "Cloud Ops Studio", "DevKit", "GenAI Labs"],
    }

    async def mock_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        # RAG endpoints are matched first: `catalog/embeddings/pending` must not be
        # swallowed by the broader `catalog/search` / `products` branch below.
        if "catalog/embeddings/pending" in url:
            return mock_pending_resp
        if "catalog/facets" in url:
            return mock_facets_resp
        if "auth/validate-token" in url or "auth/verify" in url:
            return mock_token_verify_resp
        if "catalog/search" in url or "products" in url:
            return mock_products_resp
        if "analytics/metrics" in url or "metrics" in url:
            return mock_metrics_resp
        if "health" in url:
            return MagicMock(status_code=200, json=lambda: {"status": "healthy"})
        return MagicMock(status_code=404, json=lambda: {"detail": "Not Found"})

    async def mock_post(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        if "catalog/vector-search" in url:
            return mock_vector_search_resp
        if "catalog/embeddings/similar" in url:
            return mock_similar_resp
        if "catalog/embeddings/upsert" in url:
            return mock_upsert_resp
        if "catalog/embeddings/mark-error" in url:
            return mock_mark_error_resp
        if "catalog/items/verify" in url:
            return mock_verify_resp
        if "auth/validate-token" in url:
            return mock_token_verify_resp
        # Unchanged default: `semantic-search` and everything else still answer 201,
        # which DjangoAPIService treats as "not 200" and routes to its dev mock. Several
        # pre-existing tests depend on that fallback, so it must stay as it is.
        return MagicMock(status_code=201, json=lambda: {"status": "created"})

    client.get = AsyncMock(side_effect=mock_get)
    client.post = AsyncMock(side_effect=mock_post)
    return client


@pytest.fixture
def mock_django_service(mock_django_http_client: AsyncMock) -> DjangoAPIService:
    """Mocked DjangoAPIService returning the mock HTTP client."""
    service = DjangoAPIService(
        base_url=TEST_DJANGO_BACKEND_URL,
        internal_secret=TEST_INTERNAL_SECRET,
    )
    service.get_client = AsyncMock(return_value=mock_django_http_client)
    return service


# ==============================================================================
# Embedding & Retrieval Degradation Fixtures (Fase 2 & 5)
# ==============================================================================

class StubEmbeddingService:
    """Deterministic stand-in for `EmbeddingService` with the same async surface.

    Records the `(text, task_type)` of every call so a test can assert that the
    asymmetric retrieval contract was honoured — documents embedded with
    `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY`.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, Any]] = []
        self.is_available = True

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        """Returns the deterministic vector and records the call."""
        self.calls.append({"text": text, "task_type": task_type})
        return deterministic_vector(self.dimensions)

    async def embed_document(self, text: str) -> list[float]:
        """Document half of the asymmetric pair."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Query half of the asymmetric pair."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_QUERY")


class FailingEmbeddingService:
    """Embedding service double that always fails, to drive the degradation path."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or EmbeddingServiceError("Both embedding models failed (simulated).")
        self.calls: list[dict[str, Any]] = []
        self.is_available = False

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        """Always raises, never fabricates a vector."""
        self.calls.append({"text": text, "task_type": task_type})
        raise self.error

    async def embed_document(self, text: str) -> list[float]:
        """Always raises."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Always raises."""
        return await self.embed_text(text=text, task_type="RETRIEVAL_QUERY")


class FailingVectorDjangoService:
    """Django service double whose vector engine is down but whose lexical engine works.

    This is the exact production incident the Fase 5 fallback exists for: pgvector
    unavailable, keyword search still serving.
    """

    def __init__(self) -> None:
        self.vector_calls: list[dict[str, Any]] = []
        self.lexical_calls: list[dict[str, Any]] = []
        self.verify_calls: list[dict[str, Any]] = []

    async def vector_search(self, **kwargs: Any) -> dict[str, Any]:
        """Simulates the pgvector engine being unreachable."""
        self.vector_calls.append(kwargs)
        raise RuntimeError("pgvector engine unavailable (simulated)")

    async def find_similar_products(self, **kwargs: Any) -> dict[str, Any]:
        """Simulates the similarity engine being unreachable."""
        self.vector_calls.append(kwargs)
        raise RuntimeError("vector similarity engine unavailable (simulated)")

    async def legacy_lexical_search(self, **kwargs: Any) -> dict[str, Any]:
        """Serves a healthy keyword-search response honouring the requested filters."""
        self.lexical_calls.append(kwargs)
        items = [
            {"id": 3, "slug": "curso-fastapi", "title": "Curso FastAPI", "name": "Curso FastAPI",
             "category": "Cursos", "brand": "Academy Pro", "price": 49.99, "currency": "USD",
             "stock": 50, "in_stock": True, "match_score": 0.91},
            {"id": 4, "slug": "modulo-llm", "title": "Módulo LLM", "name": "Módulo LLM",
             "category": "Software", "brand": "GenAI Labs", "price": 89.00, "currency": "USD",
             "stock": 25, "in_stock": True, "match_score": 0.84},
        ]
        min_price = kwargs.get("min_price")
        max_price = kwargs.get("max_price")
        if min_price is not None:
            items = [item for item in items if item["price"] >= min_price]
        if max_price is not None:
            items = [item for item in items if item["price"] <= max_price]
        items = items[: kwargs.get("top_k") or 8]
        return {
            "status": "success",
            "query": kwargs.get("query"),
            "count": len(items),
            "items": items,
            "engine": "lexical",
            "filters_applied": dict(kwargs),
        }

    async def verify_items(self, **kwargs: Any) -> dict[str, Any]:
        """Resolves the reference product name used to seed the lexical fallback."""
        self.verify_calls.append(kwargs)
        return {
            "status": "success",
            "items": [{"id": 3, "title": "Curso FastAPI", "name": "Curso FastAPI"}],
            "not_found": [],
        }


@pytest.fixture
def mock_embedding_service() -> StubEmbeddingService:
    """Deterministic embedding service double (768-dim, records task types)."""
    return StubEmbeddingService()


@pytest.fixture
def failing_embedding_service() -> FailingEmbeddingService:
    """Embedding service double that always raises EmbeddingServiceError."""
    return FailingEmbeddingService()


@pytest.fixture
def failing_django_service() -> FailingVectorDjangoService:
    """Django double whose vector engine fails but whose lexical engine succeeds."""
    return FailingVectorDjangoService()


# ==============================================================================
# Sample Payloads & Headers Fixtures
# ==============================================================================

@pytest.fixture
def valid_internal_headers() -> dict[str, str]:
    """Valid headers for service-to-service internal communication."""
    return {"X-Internal-Secret": TEST_INTERNAL_SECRET}


@pytest.fixture
def invalid_internal_headers() -> dict[str, str]:
    """Invalid headers with bad secret."""
    return {"X-Internal-Secret": "incorrect-wrong-secret"}


@pytest.fixture
def sample_chat_request_payload() -> dict[str, Any]:
    """Standard chat request dictionary payload (stream=True by default)."""
    return {
        "agent_id": "portfolio",
        "session_id": "sess_test_qa_001",
        "message": "Hola, ¿podrías detallar tu experiencia en Python y Cloud?",
        "stream": True,
    }


@pytest.fixture
def sample_chat_request_non_stream_payload() -> dict[str, Any]:
    """Non-streaming chat request dictionary payload."""
    return {
        "agent_id": "ecommerce",
        "session_id": "sess_test_qa_002",
        "message": "Quisiera consultar los precios del catálogo.",
        "stream": False,
    }


@pytest.fixture
def sample_chat_request_obj(sample_chat_request_payload: dict[str, Any]) -> ChatRequest:
    """Validated ChatRequest Pydantic model instance."""
    return ChatRequest(**sample_chat_request_payload)


@pytest.fixture
def sample_chat_response_payload() -> dict[str, Any]:
    """Standard chat response dictionary payload."""
    return {
        "agent_id": "portfolio",
        "session_id": "sess_test_qa_001",
        "message": "Cuento con más de 6 años de experiencia en arquitecturas cloud y Python...",
        "metadata": {
            "model": "gemini-3.7-flash",
            "tokens_used": 128,
            "latency_ms": 245.5,
        },
    }
