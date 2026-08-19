"""Unit and integration tests for Agent Base, Concrete Agents, and Agent Dispatcher."""
from typing import AsyncGenerator
import pytest
from app.agents.base import BaseAgent
from app.agents.dispatcher import AgentDispatcher
from app.schemas.payload import ChatRequest, ChatResponse


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
            metadata={"agent": "portfolio", "model": "gemini-2.5-flash"},
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
