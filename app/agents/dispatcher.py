"""Agent Dispatcher and Intent Orchestrator."""
import logging
from typing import AsyncGenerator, Optional
from app.agents.analytics import AnalyticsAgent
from app.agents.base import BaseAgent
from app.agents.ecommerce import EcommerceAgent
from app.agents.portfolio import PortfolioAgent
from app.schemas.payload import AgentInfo, ChatRequest, ChatResponse

logger = logging.getLogger("ai_gateway.dispatcher")


class AgentDispatcher:
    """Registry, Intent Classifier and Dispatcher for Multi-Agent routing."""

    def __init__(self, auto_register: bool = False) -> None:
        self._agents: dict[str, BaseAgent] = {}
        if auto_register:
            self._register_default_agents()

    def _register_default_agents(self) -> None:
        """Instantiates and registers the core specialized agents."""
        self.register("portfolio", PortfolioAgent())
        self.register("ecommerce", EcommerceAgent())
        self.register("analytics", AnalyticsAgent())

    def register(self, agent_id: str, agent: BaseAgent) -> None:
        """Registers a specialized agent under a unique identifier."""
        self._agents[agent_id] = agent
        logger.info("Agent registered: id='%s', name='%s'", agent_id, agent.name)

    def get(self, agent_id: str) -> BaseAgent:
        """Retrieves a registered agent instance.

        Raises KeyError if the requested agent_id is not found.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' is not registered.")
        return self._agents[agent_id]

    def list_agents(self) -> list[AgentInfo]:
        """Returns metadata for all currently registered agents."""
        return [agent.get_info() for agent in self._agents.values()]

    def classify_intent(self, message: str) -> str:
        """Heuristic and keyword-based intent classifier for automatic routing.

        Routes message to 'portfolio', 'ecommerce', or 'analytics'.
        Defaults to 'portfolio' for general greeting/profile inquiries.
        """
        msg_lower = message.lower().strip()

        # E-Commerce signals
        ecommerce_keywords = [
            "producto", "productos", "comprar", "precio", "precios", "catálogo", "catalogo",
            "curso", "cursos", "stock", "cuanto cuesta", "cuánto cuesta", "costo", "costos",
            "tienda", "order", "buy", "price", "shop", "adquirir", "contratar servicio",
        ]
        if any(kw in msg_lower for kw in ecommerce_keywords):
            return "ecommerce"

        # Analytics signals
        analytics_keywords = [
            "métrica", "metrica", "métricas", "metricas", "kpi", "kpis", "analítica", "analitica",
            "estadística", "estadisticas", "estadísticas", "ventas del mes", "conversión",
            "conversion", "tráfico", "trafico", "rendimiento", "analytics", "dashboard", "reporte",
            "revenue", "usuarios activos",
        ]
        if any(kw in msg_lower for kw in analytics_keywords):
            return "analytics"

        # Default to portfolio
        return "portfolio"

    def resolve_agent(self, requested_id: Optional[str], message: str) -> BaseAgent:
        """Resolves target agent by requested ID or performs automatic intent classification."""
        if not requested_id or requested_id.lower() in ("auto", "default", "general"):
            target_id = self.classify_intent(message)
            logger.info("Auto-classified message intent to agent '%s'", target_id)
            return self.get(target_id)

        target_id = requested_id.lower().strip()
        if target_id in self._agents:
            return self._agents[target_id]

        logger.warning("Requested agent '%s' not found. Falling back to intent classification.", target_id)
        return self.get(self.classify_intent(message))

    async def dispatch(self, request: ChatRequest) -> ChatResponse:
        """Dispatches request to resolved agent for non-streamed inference."""
        agent = self.resolve_agent(request.agent_id, request.message)
        # Ensure request reflects the executing agent
        request.agent_id = agent.agent_id
        return await agent.process(request)

    async def dispatch_stream(self, request: ChatRequest) -> tuple[BaseAgent, AsyncGenerator[str, None]]:
        """Resolves target agent and returns the token generator for streaming."""
        agent = self.resolve_agent(request.agent_id, request.message)
        request.agent_id = agent.agent_id
        return agent, agent.process_stream(request)


# Global singleton instance
_dispatcher: Optional[AgentDispatcher] = None


def get_agent_dispatcher() -> AgentDispatcher:
    """Returns the singleton instance of AgentDispatcher with core agents registered."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AgentDispatcher(auto_register=True)
    return _dispatcher
