"""Unit and integration tests for Agent Base, Concrete Agents, Analytics Multi-Tool Grounding and Agent Dispatcher."""
from typing import Any, AsyncGenerator
import pytest
from app.agents.analytics import AnalyticsAgent
from app.agents.base import BaseAgent
from app.agents.dispatcher import AgentDispatcher
from app.agents.ecommerce import EcommerceAgent
from app.schemas.payload import ChatRequest, ChatResponse
from app.services.knowledge_base import KnowledgeBaseService


# ==============================================================================
# Concrete Mock Agents for Testing
# ==============================================================================

class MockPortfolioAgent(BaseAgent):
    """Specialized agent for Portfolio, Skills, and Experience inquiries."""

    def __init__(self, agent_id: str = "portfolio") -> None:
        super().__init__(agent_id)

    async def process(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            agent_id=self.agent_id,
            session_id=request.session_id,
            message=f"[Portfolio] Información para: '{request.message}'",
            metadata={"agent": "portfolio", "model": "gemini-3.7-flash"},
        )

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        chunks = ["[Portfolio] ", "Detalle ", "de experiencia: ", request.message]
        for c in chunks:
            yield c


class MockEcommerceAgent(BaseAgent):
    """Specialized agent for Product Catalog, Pricing, and Cart management."""

    def __init__(self, agent_id: str = "ecommerce") -> None:
        super().__init__(agent_id)

    async def process(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            agent_id=self.agent_id,
            session_id=request.session_id,
            message=f"[Ecommerce] Catálogo consultado para: '{request.message}'",
            metadata={"agent": "ecommerce", "catalog_version": "v1"},
        )

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        chunks = ["[Ecommerce] ", "Productos ", "disponibles ", "en catálogo"]
        for c in chunks:
            yield c


class MockAnalyticsAgent(BaseAgent):
    """Specialized agent for KPI, Metrics, and Analytics reporting."""

    def __init__(self, agent_id: str = "analytics") -> None:
        super().__init__(agent_id)

    async def process(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            agent_id=self.agent_id,
            session_id=request.session_id,
            message=f"[Analytics] Métricas calculadas para: '{request.message}'",
            metadata={"agent": "analytics", "kpi_count": 5},
        )

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        chunks = ["[Analytics] ", "Total ", "conversiones: ", "98.5%"]
        for c in chunks:
            yield c


class FaultyAgent(BaseAgent):
    """Agent that simulates runtime execution failure for robustness testing."""

    async def process(self, request: ChatRequest) -> ChatResponse:
        raise RuntimeError("Agent LLM service timeout")

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        yield "Initial chunk"
        raise ConnectionResetError("Stream connection severed unexpectedly")


# ==============================================================================
# BaseAgent & Abstract Class Tests
# ==============================================================================

class TestBaseAgent:
    """Test suite for BaseAgent abstract contract."""

    def test_cannot_instantiate_abstract_base_agent(self) -> None:
        """Verifies that BaseAgent cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError) as exc_info:
            BaseAgent("abstract_test")  # type: ignore
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_concrete_agent_initialization(self) -> None:
        """Verifies that a properly implemented agent sets agent_id correctly."""
        agent = MockPortfolioAgent(agent_id="custom_portfolio")
        assert agent.agent_id == "custom_portfolio"


# ==============================================================================
# Ecommerce Agent Product Extraction & Hybrid Context Tests (Ticket BE-02 & BE-04)
# ==============================================================================

class TestEcommerceProductExtractionAndHybridContext:
    """Test suite for Ticket BE-02 & BE-04: Product extraction, policies & hybrid context."""

    def test_extract_search_terms_quoted_product(self) -> None:
        """Verifies extraction of product names inside quotes and stripped price tags."""
        agent = EcommerceAgent()
        msg = 'Hola, me interesa el producto "Servicio Cloud AI" ($49.99)...'
        terms = agent.extract_search_terms(msg)
        assert len(terms) > 0
        assert terms[0] == "Servicio Cloud AI"

    def test_extract_search_terms_pattern_match(self) -> None:
        """Verifies extraction when user uses phrase 'producto ...' without quotes."""
        agent = EcommerceAgent()
        msg = "Quisiera consultar disponibilidad del producto Curso Avanzado de FastAPI y Microservicios"
        terms = agent.extract_search_terms(msg)
        assert len(terms) > 0
        assert any("Curso Avanzado de FastAPI" in t for t in terms)

    def test_extract_search_terms_meaningful_tokens(self) -> None:
        """Verifies extraction of tokenized keywords when no patterns or quotes match."""
        agent = EcommerceAgent()
        msg = "Buenas tardes, ¿tienen servidores cloud disponibles?"
        terms = agent.extract_search_terms(msg)
        assert len(terms) > 0
        assert any("servidores" in t.lower() or "cloud" in t.lower() for t in terms)

    @pytest.mark.asyncio
    async def test_ecommerce_context_augmentation_with_hybrid_grounding(self) -> None:
        """Verifies that context augmentation generates structured hybrid context (Business Context + Database Grounding)."""
        agent = EcommerceAgent()
        req = ChatRequest(
            agent_id="ecommerce",
            session_id="sess_ecom_01",
            message='Hola, me interesa el producto "Consultoría DevOps" ($120.00)... ¿Cuáles son las políticas de devolución?',
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "Business Context & Policies" in context
        assert "Live Catalog / Database Grounding" in context
        assert "Consultoría DevOps" in context

    @pytest.mark.asyncio
    async def test_ecommerce_context_augmentation_with_reviews_inquiry(self) -> None:
        """Verifies that asking for reviews attaches the customer reviews summary block."""
        agent = EcommerceAgent()
        req = ChatRequest(
            agent_id="ecommerce",
            session_id="sess_ecom_02",
            message="¿Qué opiniones y reseñas tienen los clientes sobre sus cursos?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "Customer Reviews & Ratings Summary" in context
        assert "average_rating" in context or "total_reviews" in context


# ==============================================================================
# Analytics Agent Multi-Tool Grounding Tests (Ticket BE-05)
# ==============================================================================

class TestAnalyticsAgentMultiToolGrounding:
    """Test suite for Ticket BE-05: AnalyticsAgent routing across all 8 internal tools."""

    @pytest.mark.asyncio
    async def test_analytics_inventory_health_grounding(self) -> None:
        """Verifies routing to inventory health on stock inquiry."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_an_01",
            message="¿Cuál es el estado del stock crítico y los productos agotados en el inventario?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_inventory_health" in context
        assert "inventory_health" in context

    @pytest.mark.asyncio
    async def test_analytics_margins_and_profitability_grounding(self) -> None:
        """Verifies routing to margins and profitability on margin inquiry."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_an_02",
            message="¿Cuál es el margen de ganancia y rentabilidad por categoría?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_product_profitability" in context
        assert "margins_and_profitability" in context

    @pytest.mark.asyncio
    async def test_analytics_funnel_and_cart_metrics_grounding(self) -> None:
        """Verifies routing to funnel and abandoned carts on funnel inquiry."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_an_03",
            message="Muéstrame la tasa de abandono de carritos y la conversión del funnel.",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_funnel_and_cart_metrics" in context
        assert "funnel_and_cart_metrics" in context

    @pytest.mark.asyncio
    async def test_analytics_customer_rfm_segmentation_grounding(self) -> None:
        """Verifies routing to customer RFM insights on customer segment inquiry."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_an_04",
            message="¿Cuántos clientes VIP tenemos y cuál es su LTV promedio?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_customer_segmentation" in context
        assert "customer_segmentation" in context

    @pytest.mark.asyncio
    async def test_analytics_safe_sql_sandbox_grounding(self) -> None:
        """Verifies routing to Safe SQL Sandbox on SELECT query."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_an_05",
            message="SELECT id, name, price FROM products LIMIT 5;",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "execute_raw_sql_sandbox" in context
        assert "sql_results" in context


# ==============================================================================
# Agent Dispatcher Tests
# ==============================================================================

class TestAgentDispatcher:
    """Test suite for AgentDispatcher registry and routing."""

    def test_dispatcher_initial_state(self) -> None:
        """Verifies that a new dispatcher has an empty agent registry."""
        dispatcher = AgentDispatcher()
        assert len(dispatcher._agents) == 0

    def test_register_and_get_agents(self) -> None:
        """Verifies registering and retrieving distinct agent types."""
        dispatcher = AgentDispatcher()
        portfolio_agent = MockPortfolioAgent()
        ecommerce_agent = MockEcommerceAgent()
        analytics_agent = MockAnalyticsAgent()

        dispatcher.register("portfolio", portfolio_agent)
        dispatcher.register("ecommerce", ecommerce_agent)
        dispatcher.register("analytics", analytics_agent)

        assert dispatcher.get("portfolio") is portfolio_agent
        assert dispatcher.get("ecommerce") is ecommerce_agent
        assert dispatcher.get("analytics") is analytics_agent

    def test_get_unregistered_agent_raises_key_error(self) -> None:
        """Verifies that querying an unregistered agent ID raises a KeyError."""
        dispatcher = AgentDispatcher()
        with pytest.raises(KeyError) as exc_info:
            dispatcher.get("non_existent_agent")
        assert "Agent 'non_existent_agent' is not registered" in str(exc_info.value)

    def test_dispatcher_overwrite_agent(self) -> None:
        """Verifies that registering an existing agent_id replaces the old instance."""
        dispatcher = AgentDispatcher()
        agent_v1 = MockPortfolioAgent("portfolio")
        agent_v2 = MockPortfolioAgent("portfolio_v2")

        dispatcher.register("portfolio", agent_v1)
        assert dispatcher.get("portfolio") is agent_v1

        dispatcher.register("portfolio", agent_v2)
        assert dispatcher.get("portfolio") is agent_v2

    @pytest.mark.asyncio
    async def test_dispatched_agent_process(self, sample_chat_request_obj: ChatRequest) -> None:
        """Verifies end-to-end non-streaming message processing via dispatched agent."""
        dispatcher = AgentDispatcher()
        agent = MockPortfolioAgent()
        dispatcher.register("portfolio", agent)

        retrieved_agent = dispatcher.get("portfolio")
        response = await retrieved_agent.process(sample_chat_request_obj)

        assert isinstance(response, ChatResponse)
        assert response.agent_id == "portfolio"
        assert response.session_id == sample_chat_request_obj.session_id
        assert "Información para" in response.message

    @pytest.mark.asyncio
    async def test_dispatched_agent_streaming(self, sample_chat_request_obj: ChatRequest) -> None:
        """Verifies end-to-end streaming token emission via dispatched agent."""
        dispatcher = AgentDispatcher()
        agent = MockEcommerceAgent()
        dispatcher.register("ecommerce", agent)

        retrieved_agent = dispatcher.get("ecommerce")
        chunks = []
        async for chunk in retrieved_agent.process_stream(sample_chat_request_obj):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0] == "[Ecommerce] "
        assert "".join(chunks) == "[Ecommerce] Productos disponibles en catálogo"

    @pytest.mark.asyncio
    async def test_faulty_agent_isolation(self, sample_chat_request_obj: ChatRequest) -> None:
        """Verifies that an error in one agent does not corrupt the dispatcher or affect others."""
        dispatcher = AgentDispatcher()
        dispatcher.register("faulty", FaultyAgent("faulty"))
        dispatcher.register("healthy", MockPortfolioAgent("healthy"))

        # Faulty agent fails
        with pytest.raises(RuntimeError) as exc_info:
            await dispatcher.get("faulty").process(sample_chat_request_obj)
        assert "Agent LLM service timeout" in str(exc_info.value)

        # Healthy agent still operates normally
        healthy_res = await dispatcher.get("healthy").process(sample_chat_request_obj)
        assert healthy_res.agent_id == "healthy"


# ==============================================================================
# Analytics Agent Multi-Tool Intent Resolution & Tool Invocation Tests (Ticket QA-04)
# ==============================================================================

class TestAnalyticsAgentMultiToolResolution:
    """Test suite for AnalyticsAgent intent routing across the 8 specialized Django tools."""

    @pytest.mark.asyncio
    async def test_analytics_agent_sales_intent_resolution(self) -> None:
        """Verifies that sales/revenue inquiries invoke query_sales_analytics."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_sales_01",
            message="Quisiera ver el reporte de ventas e ingresos por categoría de este mes.",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "tool_invoked" in context
        assert "query_sales_analytics" in context
        assert "total_revenue_usd" in context or "sales_analytics" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_inventory_intent_resolution(self) -> None:
        """Verifies that stock/inventory inquiries invoke get_inventory_health."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_inv_01",
            message="¿Cuáles productos están en stock crítico o con riesgo de quiebre?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_inventory_health" in context
        assert "critical_stock_count" in context or "inventory_health" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_margins_intent_resolution(self) -> None:
        """Verifies that profitability/margin inquiries invoke get_product_profitability."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_margin_01",
            message="Mostrar análisis de rentabilidad, ganancias y margen porcentual por producto.",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_product_profitability" in context
        assert "overall_gross_margin_pct" in context or "margins_and_profitability" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_funnel_intent_resolution(self) -> None:
        """Verifies that conversion/cart inquiries invoke get_funnel_and_cart_metrics."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_funnel_01",
            message="Revisar tasa de conversión del embudo (funnel) y carritos abandonados.",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_funnel_and_cart_metrics" in context
        assert "cart_abandonment_rate_pct" in context or "funnel_and_cart_metrics" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_segmentation_intent_resolution(self) -> None:
        """Verifies that customer RFM/LTV inquiries invoke get_customer_segmentation."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_cust_01",
            message="Analizar métricas de clientes VIP y segmentos en riesgo de churn.",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_customer_segmentation" in context
        assert "vip" in context or "customer_segmentation" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_reviews_intent_resolution(self) -> None:
        """Verifies that customer satisfaction/reviews inquiries invoke get_customer_reviews_summary."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_rev_01",
            message="¿Cuál es el resumen de opiniones, reseñas y calificación promedio de clientes?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "get_customer_reviews_summary" in context
        assert "average_rating" in context or "reviews_summary" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_sql_sandbox_safe_intent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that SQL SELECT queries invoke execute_raw_sql_sandbox FOR STAFF.

        The token is now validated explicitly as staff instead of relying on the string
        "valid-analyst-token" happening to satisfy a permissive development fallback.
        That fallback was a security hole and has been closed, so without this stub the
        test would still pass — but through the BLOCKED branch, silently asserting the
        opposite of what its name claims. Pinning the staff identity keeps it testing the
        privileged path it was written for.
        """
        agent = AnalyticsAgent()

        async def staff_validation(token: str) -> dict[str, Any]:
            # The six normalized keys `validate_user_token` now always returns.
            # Privilege is Django's native `is_staff` / `is_superuser` booleans:
            # `auth_user` has no `role` column, and a role string authorizes nothing.
            return {
                "valid": True,
                "user_id": 1,
                "username": "admin_user",
                "is_staff": True,
                "is_superuser": False,
                "error": None,
            }

        monkeypatch.setattr(agent.django_service, "validate_user_token", staff_validation)

        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_sql_01",
            message="Ejecuta la consulta: SELECT id, name, price, stock FROM products WHERE stock > 5;",
            user_token="valid-analyst-token",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "execute_raw_sql_sandbox" in context
        assert "sandbox_mode" in context or "sql_results" in context
        # The privileged branch really ran: a blocked result would carry blocked=true.
        assert '"blocked": true' not in context.lower()

    @pytest.mark.asyncio
    async def test_analytics_agent_sql_sandbox_blocks_destructive_sql(self) -> None:
        """Verifies that destructive SQL commands in user prompts are blocked with safety error."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_sql_unsafe",
            message="Ejecuta: DROP TABLE users; SELECT * FROM products;",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "Safety violation" in context or "error" in context

    @pytest.mark.asyncio
    async def test_analytics_agent_auth_status_propagation(self) -> None:
        """Verifies that user_token is validated and attached to the analytics auth context."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_auth_01",
            message="Dame el balance general de ventas",
            user_token="valid-jwt-token-qa-test",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "auth_context" in context
        assert "authenticated" in context


# ==============================================================================
# Ecommerce Agent Advanced Tools & Reviews Integration Tests (Ticket QA-04)
# ==============================================================================

class TestEcommerceAgentAdvancedTools:
    """Test suite for EcommerceAgent reviews and semantic catalog tools."""

    @pytest.mark.asyncio
    async def test_ecommerce_agent_customer_reviews_inquiry(self) -> None:
        """Verifies that customer inquiries about opinions or reviews include reviews data."""
        agent = EcommerceAgent()
        req = ChatRequest(
            agent_id="ecommerce",
            session_id="sess_ecom_rev_01",
            message="¿Qué opiniones y calificaciones tienen los clientes sobre sus cursos y servicios?",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "Customer Reviews & Ratings Summary" in context
        assert "average_rating" in context or "total_reviews" in context

    @pytest.mark.asyncio
    async def test_ecommerce_agent_semantic_search_direct(self) -> None:
        """Verifies semantic search functionality directly on Django service."""
        from app.services.django_api import get_django_api_service
        service = get_django_api_service()
        results = await service.semantic_catalog_search(
            query="arquitectura de microservicios distribuida con docker",
            top_k=2,
        )
        assert results["status"] == "success"
        assert len(results["items"]) <= 2
        # `title` is the canonical field; `name` is the deprecated mirror the widget
        # still reads. Both must be present until the widget migrates.
        assert "title" in results["items"][0]
        assert "name" in results["items"][0]


# ==============================================================================
# Analytics Agent Temporal & Date Range Extraction Tests (Ticket QA-05)
# ==============================================================================

class TestAnalyticsAgentDateRangeExtraction:
    """Test suite for AnalyticsAgent._extract_date_range natural-language temporal parsing."""

    def test_extract_explicit_iso_date_range(self) -> None:
        """Verifies parsing of explicit ISO date ranges."""
        d_from, d_to = AnalyticsAgent._extract_date_range("ventas desde 2025-01-01 hasta 2025-03-31")
        assert d_from == "2025-01-01"
        assert d_to == "2025-03-31"

    def test_extract_named_month_and_year_spanish(self) -> None:
        """Verifies parsing of Spanish month + year phrases."""
        d_from, d_to = AnalyticsAgent._extract_date_range("reporte de ventas de diciembre de 2025")
        assert d_from == "2025-12-01"
        assert d_to == "2025-12-31"

        d_from2, d_to2 = AnalyticsAgent._extract_date_range("ingresos de febrero 2024")
        assert d_from2 == "2024-02-01"
        assert d_to2 == "2024-02-29"  # 2024 is a leap year

    def test_extract_named_month_and_year_english(self) -> None:
        """Verifies parsing of English month + year phrases."""
        d_from, d_to = AnalyticsAgent._extract_date_range("sales in january 2025")
        assert d_from == "2025-01-01"
        assert d_to == "2025-01-31"

    def test_extract_quarter_format(self) -> None:
        """Verifies parsing of quarter expressions."""
        d_from, d_to = AnalyticsAgent._extract_date_range("kpis del Q1 2025")
        assert d_from == "2025-01-01"
        assert d_to == "2025-03-31"

        d_from_q3, d_to_q3 = AnalyticsAgent._extract_date_range("tercer trimestre 2024")
        assert d_from_q3 == "2024-07-01"
        assert d_to_q3 == "2024-09-30"

    def test_extract_relative_year(self) -> None:
        """Verifies parsing of relative year keywords."""
        from datetime import date
        today = date.today()

        # Last year
        d_from_ly, d_to_ly = AnalyticsAgent._extract_date_range("balance del año pasado")
        assert d_from_ly == f"{today.year - 1}-01-01"
        assert d_to_ly == f"{today.year - 1}-12-31"

        # This year
        d_from_ty, d_to_ty = AnalyticsAgent._extract_date_range("ingresos de este año")
        assert d_from_ty == f"{today.year}-01-01"
        assert d_to_ty == f"{today.year}-12-31"

    def test_extract_relative_month(self) -> None:
        """Verifies parsing of relative month keywords."""
        d_from, d_to = AnalyticsAgent._extract_date_range("ventas de este mes")
        assert d_from is not None
        assert d_to is not None
        assert d_from.endswith("-01")

    def test_extract_no_dates_returns_none(self) -> None:
        """Verifies that non-temporal inquiries return (None, None)."""
        d_from, d_to = AnalyticsAgent._extract_date_range("balance general de productos")
        assert d_from is None
        assert d_to is None

    @pytest.mark.asyncio
    async def test_analytics_agent_temporal_context_injection(self) -> None:
        """Verifies that detected_date_range is injected into context_data when dates are present."""
        agent = AnalyticsAgent()
        req = ChatRequest(
            agent_id="analytics",
            session_id="sess_temp_01",
            message="Ingresos y ventas de diciembre de 2025 por categoría",
        )
        context = await agent.get_context_augmentation(req)
        assert context is not None
        assert "detected_date_range" in context
        assert "2025-12-01" in context
        assert "2025-12-31" in context


