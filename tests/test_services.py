"""Unit and Integration tests for Gateway Services (Django API, LLM Client, Redis Memory, Knowledge Base, and Security)."""
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException, status
import httpx
from app.core.security import verify_internal_api_secret
from app.services.django_api import DjangoAPIService
from app.services.knowledge_base import KnowledgeBaseService
from app.services.llm_client import LLMClientService
from app.services.memory import RedisMemoryService
from tests.conftest import MockGenAIChunk, MockGenAIResponse, MockRedisBackend


# ==============================================================================
# Security & Internal API Secret Tests
# ==============================================================================

class TestSecurityService:
    """Test suite for internal authentication and secret verification."""

    @pytest.mark.asyncio
    async def test_verify_internal_api_secret_success(self) -> None:
        """Verifies that matching internal secret passes validation."""
        result = await verify_internal_api_secret(x_internal_secret="test-internal-secret-67890")
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_internal_api_secret_missing(self) -> None:
        """Verifies that missing secret raises HTTP 401 Unauthorized."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_internal_api_secret(x_internal_secret=None)
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid or missing" in exc_info.value.detail
        assert exc_info.value.headers.get("WWW-Authenticate") == "X-Internal-Secret"

    @pytest.mark.asyncio
    async def test_verify_internal_api_secret_invalid(self) -> None:
        """Verifies that invalid secret raises HTTP 401 Unauthorized."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_internal_api_secret(x_internal_secret="wrong-secret-token")
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid or missing" in exc_info.value.detail


# ==============================================================================
# Knowledge Base Service Tests (Ticket BE-04)
# ==============================================================================

class TestKnowledgeBaseService:
    """Test suite for KnowledgeBaseService Markdown loading and caching."""

    @pytest.mark.asyncio
    async def test_load_existing_markdown_file(self) -> None:
        """Verifies loading the template ecommerce_business_context.md file."""
        service = KnowledgeBaseService()
        content = await service.get_ecommerce_context()
        assert len(content) > 100
        assert "Base de Conocimiento" in content or "Políticas" in content or "Envíos" in content

    @pytest.mark.asyncio
    async def test_load_non_existent_file_returns_fallback(self) -> None:
        """Verifies that a missing file path returns structured fallback context."""
        service = KnowledgeBaseService(default_ecommerce_path="data/non_existent_file_123.md")
        content = await service.get_ecommerce_context()
        assert "Información de Negocio y Políticas (Fallback)" in content
        assert "AI Solutions & E-Commerce Store" in content

    @pytest.mark.asyncio
    async def test_knowledge_base_caching(self) -> None:
        """Verifies in-memory caching across consecutive loads."""
        service = KnowledgeBaseService()
        service.clear_cache()
        content1 = await service.get_ecommerce_context()
        content2 = await service.get_ecommerce_context()
        assert content1 == content2


# ==============================================================================
# Django Monolith API Service Tests
# ==============================================================================

class TestDjangoAPIService:
    """Test suite for Django Monolith HTTP client communication service."""

    def test_django_api_service_default_init(self) -> None:
        """Verifies initialization with default configuration."""
        service = DjangoAPIService()
        assert service.base_url == "http://test-django:8000"
        assert service.internal_secret == "test-internal-secret-67890"
        assert service._headers["X-Internal-Secret"] == "test-internal-secret-67890"
        assert service._headers["Content-Type"] == "application/json"

    def test_django_api_service_custom_init(self) -> None:
        """Verifies custom base URL and secret overrides."""
        service = DjangoAPIService(
            base_url="https://api.monolith.company.com/v2/",
            internal_secret="custom-secret-key",
        )
        assert service.base_url == "https://api.monolith.company.com/v2"
        assert service.internal_secret == "custom-secret-key"
        assert service._headers["X-Internal-Secret"] == "custom-secret-key"

    @pytest.mark.asyncio
    async def test_django_api_service_get_client(self) -> None:
        """Verifies that get_client returns configured httpx.AsyncClient."""
        service = DjangoAPIService()
        client = await service.get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert str(client.base_url).rstrip("/") == "http://test-django:8000"
        assert client.headers.get("x-internal-secret") == "test-internal-secret-67890"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_django_api_service_mock_call(self, mock_django_service: DjangoAPIService) -> None:
        """Verifies interacting with mocked Django backend."""
        client = await mock_django_service.get_client()
        response = await client.get("/api/products/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["name"] == "Servicio Cloud AI"

    @pytest.mark.asyncio
    async def test_django_api_service_search_catalog_scoring(self) -> None:
        """Verifies search_catalog token scoring and match prioritization."""
        service = DjangoAPIService()
        results = await service.search_catalog(query="DevOps", limit=3)
        assert len(results) > 0
        assert "DevOps" in results[0]["name"] or "DevOps" in results[0]["description"]

    @pytest.mark.asyncio
    async def test_django_api_service_timeout_handling(self) -> None:
        """Verifies error handling when Django backend times out."""
        service = DjangoAPIService()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Connection timed out to Django")
        service.get_client = AsyncMock(return_value=mock_client)

        client = await service.get_client()
        with pytest.raises(httpx.TimeoutException) as exc_info:
            await client.get("/api/auth/verify/")
        assert "Connection timed out" in str(exc_info.value)


# ==============================================================================
# Google GenAI LLM Client Service Tests
# ==============================================================================

class TestLLMClientService:
    """Test suite for Google GenAI LLM Client Service."""

    def test_llm_client_default_init(self) -> None:
        """Verifies initialization with default Gemini API Key."""
        service = LLMClientService()
        assert service.api_key == "test-mock-gemini-key-12345"
        assert service._client is None

    def test_llm_client_custom_init(self) -> None:
        """Verifies initialization with explicit API Key."""
        service = LLMClientService(api_key="custom-gemini-key-999")
        assert service.api_key == "custom-gemini-key-999"

    def test_llm_client_candidate_models_fallback_order(self) -> None:
        """Verifies ordered list of candidate models for fallback."""
        service = LLMClientService()
        models = service._get_candidate_models("gemini-custom")
        assert models[0] == "gemini-custom"
        assert "gemini-3.7-flash" in models
        assert "gemini-3.5-flash-lite" in models

    def test_llm_client_fallback_distinguishes_missing_key_vs_configured_key(self) -> None:
        """Verifies that fallback message does not falsely claim unconfigured key when key is set."""
        # Unconfigured key scenario
        unconfigured_service = LLMClientService(api_key="your-google-ai-studio-api-key-here")
        resp_unconf = unconfigured_service._generate_fallback_response("Hola")
        assert "Configura GEMINI_API_KEY" in resp_unconf

        # Configured key with error scenario
        configured_service = LLMClientService(api_key="AIzaSyA1234567890abcdefghijklmnopqrstuv")
        resp_conf = configured_service._generate_fallback_response("Hola", error_context=Exception("503 Overloaded"))
        assert "Configura GEMINI_API_KEY" not in resp_conf
        assert "modo de contingencia temporal" in resp_conf

    @pytest.mark.asyncio
    async def test_llm_client_mock_completion(self, mock_genai_client: MagicMock) -> None:
        """Verifies mocked GenAI standard content generation."""
        service = LLMClientService()
        service._client = mock_genai_client

        # Call mock generate_content
        response = service._client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hola Gemini",
        )
        assert response.text == "Hola, soy el agente de IA. ¿En qué puedo ayudarte?"
        assert response.usage_metadata.total_token_count == 42

    @pytest.mark.asyncio
    async def test_llm_client_mock_streaming(self, mock_genai_client: MagicMock) -> None:
        """Verifies mocked GenAI streaming token delivery."""
        service = LLMClientService()
        service._client = mock_genai_client

        stream_gen = service._client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents="Dame una introducción",
        )

        tokens = [chunk.text for chunk in stream_gen]
        assert len(tokens) == 4
        assert tokens[0] == "Hola, "
        assert "".join(tokens) == "Hola, soy el agente especializado. ¿En qué te puedo colaborar hoy?"

    def test_llm_client_rate_limit_error_handling(self) -> None:
        """Verifies handling of Google AI Studio HTTP 429 Rate Limit exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")

        service = LLMClientService()
        service._client = mock_client

        with pytest.raises(Exception) as exc_info:
            service._client.models.generate_content(model="gemini-2.5-flash", contents="Test")
        assert "429 RESOURCE_EXHAUSTED" in str(exc_info.value)


# ==============================================================================
# Redis Memory Service Tests
# ==============================================================================

class TestRedisMemoryService:
    """Test suite for Redis Session & Context Memory Service."""

    def test_memory_service_default_init(self) -> None:
        """Verifies initialization with default Redis URL."""
        service = RedisMemoryService()
        assert service.redis_url == "redis://localhost:6379/15"
        assert service._redis_client is None

    def test_memory_service_custom_init(self) -> None:
        """Verifies initialization with custom Redis URL."""
        service = RedisMemoryService(redis_url="redis://custom-redis-host:6380/2")
        assert service.redis_url == "redis://custom-redis-host:6380/2"

    @pytest.mark.asyncio
    async def test_memory_service_session_cache_operations(self, mock_memory_service: RedisMemoryService) -> None:
        """Verifies setting, getting, and deleting session data in Redis mock."""
        client = mock_memory_service._redis_client
        assert client is not None

        # Set session context
        await client.set("session:sess_100:context", '{"user_name": "Facundo", "role": "admin"}', ex=3600)

        # Retrieve session context
        cached = await client.get("session:sess_100:context")
        assert cached == '{"user_name": "Facundo", "role": "admin"}'

        # Delete session
        deleted = await client.delete("session:sess_100:context")
        assert deleted == 1

        # Check key after delete
        cached_after = await client.get("session:sess_100:context")
        assert cached_after is None

    @pytest.mark.asyncio
    async def test_memory_service_conversation_history_list(self, mock_memory_service: RedisMemoryService) -> None:
        """Verifies appending and fetching chat history message list in Redis mock."""
        client = mock_memory_service._redis_client
        assert client is not None

        history_key = "history:sess_200"

        # Append messages
        await client.rpush(history_key, "User: Hola", "Assistant: Hola, ¿cómo estás?", "User: Necesito soporte")

        # Fetch all messages
        history = await client.lrange(history_key, 0, -1)
        assert len(history) == 3
        assert history[0] == "User: Hola"
        assert history[2] == "User: Necesito soporte"

    @pytest.mark.asyncio
    async def test_memory_service_redis_failure_resilience(self) -> None:
        """Verifies handling when Redis connection fails or disconnects."""
        failing_client = AsyncMock()
        failing_client.get.side_effect = ConnectionError("Redis connection refused")

        service = RedisMemoryService()
        service._redis_client = failing_client

        with pytest.raises(ConnectionError) as exc_info:
            await service._redis_client.get("test_key")
        assert "Redis connection refused" in str(exc_info.value)


# ==============================================================================
# Knowledge Base Service Tests (Ticket QA-03 / BE-04)
# ==============================================================================

class TestKnowledgeBaseService:
    """Test suite for Markdown Knowledge Base loading, caching, and fallback."""

    def test_knowledge_base_default_init(self) -> None:
        """Verifies KnowledgeBaseService initialization with default configuration."""
        service = KnowledgeBaseService()
        assert service.ecommerce_path == "data/ecommerce_business_context.md"
        assert len(service._cache) == 0

    def test_knowledge_base_custom_init(self) -> None:
        """Verifies KnowledgeBaseService initialization with custom path."""
        service = KnowledgeBaseService(default_ecommerce_path="custom/path/context.md")
        assert service.ecommerce_path == "custom/path/context.md"

    @pytest.mark.asyncio
    async def test_knowledge_base_load_existing_markdown_file(self) -> None:
        """Verifies loading official business context Markdown file from disk."""
        service = KnowledgeBaseService(default_ecommerce_path="data/ecommerce_business_context.md")
        content = await service.get_ecommerce_context()

        assert content is not None
        assert len(content) > 100
        assert "Políticas de Devolución" in content or "Garantía de Satisfacción" in content
        assert "Envíos y Entregas" in content
        assert "Métodos de Pago" in content

    @pytest.mark.asyncio
    async def test_knowledge_base_caching_and_ttl(self) -> None:
        """Verifies that loaded Markdown content is cached in-memory."""
        service = KnowledgeBaseService(default_ecommerce_path="data/ecommerce_business_context.md")
        service.clear_cache()
        assert len(service._cache) == 0

        # First call loads from disk and populates cache
        content1 = await service.get_ecommerce_context()
        assert len(service._cache) == 1
        assert "data/ecommerce_business_context.md" in service._cache

        # Second call should retrieve directly from cache
        content2 = await service.get_ecommerce_context()
        assert content1 == content2

        # Clear cache
        service.clear_cache()
        assert len(service._cache) == 0

    @pytest.mark.asyncio
    async def test_knowledge_base_fallback_when_file_not_found(self) -> None:
        """Verifies that missing Markdown file safely returns default fallback context."""
        service = KnowledgeBaseService(default_ecommerce_path="data/non_existent_file_9999.md")
        content = await service.get_ecommerce_context()

        assert content is not None
        assert "Información de Negocio y Políticas (Fallback)" in content
        assert "Garantías:" in content or "14 días" in content
        assert "Métodos de Pago:" in content

    @pytest.mark.asyncio
    async def test_knowledge_base_fallback_on_read_error(self) -> None:
        """Verifies that disk I/O errors gracefully degrade to fallback context."""
        service = KnowledgeBaseService(default_ecommerce_path="data/corrupted_file.md")
        with patch.object(service, "_sync_read_file", return_value=(None, 0.0)):
            content = await service.get_ecommerce_context()
            assert "Información de Negocio y Políticas (Fallback)" in content

    def test_get_knowledge_base_service_singleton(self) -> None:
        """Verifies singleton getter returns consistent instance."""
        from app.services.knowledge_base import get_knowledge_base_service
        s1 = get_knowledge_base_service()
        s2 = get_knowledge_base_service()
        assert s1 is s2

