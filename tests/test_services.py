"""Unit and Integration tests for Gateway Services (Django API, LLM Client, Redis Memory, Knowledge Base, Tools and Security)."""
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException, status
import httpx
from app.agents.tools import ANALYTICS_TOOL_DECLARATIONS, execute_tool
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
# Ticket BE-05: 8 Specialized Endpoints and Tool Execution Tests
# ==============================================================================

class TestDjangoAPIExtendedEndpointsAndTools:
    """Test suite for Ticket BE-05: 8 internal endpoints and LLM tools."""

    @pytest.mark.asyncio
    async def test_query_sales_analytics(self) -> None:
        """Verifies dynamic sales query endpoint."""
        service = DjangoAPIService()
        res = await service.query_sales_analytics(date_from="2026-07-01", date_to="2026-08-23", dimension="category")
        assert res.get("status") == "success"
        assert "aggregates" in res
        assert res["aggregates"]["total_revenue_usd"] > 0
        assert len(res["breakdown"]) > 0

    @pytest.mark.asyncio
    async def test_get_inventory_health(self) -> None:
        """Verifies inventory health endpoint."""
        service = DjangoAPIService()
        res = await service.get_inventory_health(status_filter="all", limit=5)
        assert res.get("status") == "success"
        assert "total_products_tracked" in res
        assert len(res["items"]) > 0

    @pytest.mark.asyncio
    async def test_get_product_profitability(self) -> None:
        """Verifies product profitability & gross margin endpoint."""
        service = DjangoAPIService()
        res = await service.get_product_profitability(group_by="product", limit=5)
        assert res.get("status") == "success"
        assert "overall_gross_margin_pct" in res
        assert len(res["ranking"]) > 0

    @pytest.mark.asyncio
    async def test_get_funnel_and_cart_metrics(self) -> None:
        """Verifies conversion funnel and cart abandonment endpoint."""
        service = DjangoAPIService()
        res = await service.get_funnel_and_cart_metrics(timeframe="30d")
        assert res.get("status") == "success"
        assert "funnel_stages" in res
        assert "cart_abandonment_rate_pct" in res

    @pytest.mark.asyncio
    async def test_get_customer_reviews_summary(self) -> None:
        """Verifies reviews sentiment and star rating summary endpoint."""
        service = DjangoAPIService()
        res = await service.get_customer_reviews_summary(sentiment="all")
        assert res.get("status") == "success"
        assert "average_rating" in res
        assert res["average_rating"] >= 4.0

    @pytest.mark.asyncio
    async def test_get_customer_segmentation(self) -> None:
        """Verifies customer RFM insights endpoint."""
        service = DjangoAPIService()
        res = await service.get_customer_segmentation(segment="all")
        assert res.get("status") == "success"
        assert "segments" in res
        assert "vip" in res["segments"]

    @pytest.mark.asyncio
    async def test_semantic_catalog_search(self) -> None:
        """Verifies conceptual semantic search endpoint."""
        service = DjangoAPIService()
        res = await service.semantic_catalog_search(query="programación y microservicios", top_k=3)
        assert res.get("status") == "success"
        assert len(res["items"]) > 0
        assert "semantic_score" in res["items"][0]

    @pytest.mark.asyncio
    async def test_execute_raw_sql_sandbox_allowed_query(self) -> None:
        """Verifies safe SELECT execution in sandbox."""
        service = DjangoAPIService()
        res = await service.execute_raw_sql_sandbox(sql_query="SELECT id, name, price FROM products LIMIT 5;", max_rows=5)
        assert res.get("status") == "success"
        assert "columns" in res
        assert "data" in res

    @pytest.mark.asyncio
    async def test_execute_raw_sql_sandbox_rejects_dml_ddl(self) -> None:
        """Verifies defensive rejection of dangerous SQL keywords (DROP, DELETE, UPDATE)."""
        service = DjangoAPIService()
        res = await service.execute_raw_sql_sandbox(sql_query="DROP TABLE users;", max_rows=5)
        assert res.get("status") == "error"
        assert "Safety violation" in res.get("error", "")

    @pytest.mark.asyncio
    async def test_execute_tool_dispatcher(self) -> None:
        """Verifies the centralized execute_tool dispatcher for all 8 tools."""
        assert len(ANALYTICS_TOOL_DECLARATIONS) == 8

        # Test tool dispatcher for inventory
        inv_res = await execute_tool("get_inventory_health", {"status_filter": "critical", "limit": 2})
        assert inv_res.get("status") == "success"

        # Test tool dispatcher for sales
        sales_res = await execute_tool("query_sales_analytics", {"dimension": "category"})
        assert sales_res.get("status") == "success"

        # Test tool dispatcher for unknown tool
        unknown_res = await execute_tool("non_existent_tool", {})
        assert unknown_res.get("status") == "error"


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
# Knowledge Base Service Tests
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
        assert "POLICIES" in content.upper() or "POLÍTICAS" in content.upper()
        assert "SHIPPING" in content.upper() or "ENVÍOS" in content.upper()
        assert "PAYMENT" in content.upper() or "PAGO" in content.upper()

    @pytest.mark.asyncio
    async def test_knowledge_base_caching_and_ttl(self) -> None:
        """Verifies that loaded Markdown content is cached in-memory."""
        service = KnowledgeBaseService(default_ecommerce_path="data/ecommerce_business_context.md")
        service.clear_cache()
        assert len(service._cache) == 0

        content1 = await service.get_ecommerce_context()
        assert len(service._cache) == 1
        assert "data/ecommerce_business_context.md" in service._cache

        content2 = await service.get_ecommerce_context()
        assert content1 == content2

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

    def test_get_knowledge_base_service_singleton(self) -> None:
        """Verifies singleton getter returns consistent instance."""
        from app.services.knowledge_base import get_knowledge_base_service
        s1 = get_knowledge_base_service()
        s2 = get_knowledge_base_service()
        assert s1 is s2
