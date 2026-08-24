"""Pytest configuration and shared fixtures for AI Agent Gateway test suite."""
import os
from typing import AsyncGenerator, Generator, Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

# Ensure test environment variables are loaded prior to any app/settings import
os.environ.setdefault("PROJECT_NAME", "AI Agent Gateway - Test")
os.environ.setdefault("VERSION", "0.1.0-test")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("GEMINI_API_KEY", "test-mock-gemini-key-12345")
os.environ.setdefault("INTERNAL_API_SECRET", "test-internal-secret-67890")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("DJANGO_BACKEND_URL", "http://test-django:8000")
os.environ.setdefault("BACKEND_CORS_ORIGINS", '["http://localhost:3000","http://test-frontend:3000"]')

from app.core.config import Settings, get_settings
from app.main import create_application
from app.schemas.payload import ChatRequest, ChatResponse, HealthResponse
from app.services.django_api import DjangoAPIService
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService


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
    mock_token_verify_resp.json.return_value = {
        "valid": True,
        "user_id": 101,
        "username": "test_user",
        "email": "user@example.com",
        "roles": ["customer", "analyst"],
    }

    mock_products_resp = MagicMock(spec=httpx.Response)
    mock_products_resp.status_code = 200
    mock_products_resp.json.return_value = {
        "total_found": 2,
        "limit": 5,
        "items": [
            {"id": 1, "name": "Servicio Cloud AI", "price": 49.99, "stock": 10},
            {"id": 2, "name": "Consultoría DevOps", "price": 120.00, "stock": 5},
        ],
        "results": [
            {"id": 1, "name": "Servicio Cloud AI", "price": 49.99, "stock": 10},
            {"id": 2, "name": "Consultoría DevOps", "price": 120.00, "stock": 5},
        ],
    }

    mock_metrics_resp = MagicMock(spec=httpx.Response)
    mock_metrics_resp.status_code = 200
    mock_metrics_resp.json.return_value = {
        "daily_active_users": 1520,
        "conversion_rate": 3.8,
        "total_revenue": 54200.00,
    }

    async def mock_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
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
        if "auth/validate-token" in url:
            return mock_token_verify_resp
        return MagicMock(status_code=201, json=lambda: {"status": "created"})

    client.get = AsyncMock(side_effect=mock_get)
    client.post = AsyncMock(side_effect=mock_post)
    return client


@pytest.fixture
def mock_django_service(mock_django_http_client: AsyncMock) -> DjangoAPIService:
    """Mocked DjangoAPIService returning the mock HTTP client."""
    service = DjangoAPIService(
        base_url="http://test-django:8000",
        internal_secret="test-internal-secret-67890",
    )
    service.get_client = AsyncMock(return_value=mock_django_http_client)
    return service


# ==============================================================================
# Sample Payloads & Headers Fixtures
# ==============================================================================

@pytest.fixture
def valid_internal_headers() -> dict[str, str]:
    """Valid headers for service-to-service internal communication."""
    return {"X-Internal-Secret": "test-internal-secret-67890"}


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
