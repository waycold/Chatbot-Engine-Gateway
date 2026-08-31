"""Agent Dispatcher and Intent Orchestrator."""
import inspect
import logging
from typing import Any, AsyncGenerator, Optional
from app.agents.analytics import AnalyticsAgent, _is_staff, set_auth_status
from app.agents.base import BaseAgent, EventSink
from app.agents.ecommerce import EcommerceAgent
from app.agents.exceptions import AgentAuthorizationError
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

    async def _authorize_agent(self, agent: BaseAgent, request: ChatRequest) -> BaseAgent:
        """Enforces staff-gated routing, rejecting unauthorized analytics requests.

        This is the outermost authorization layer: an unauthenticated or non-staff
        caller never even reaches the analytics agent, let alone its tools. It
        complements — and does not replace — the per-agent schema gate and the
        `execute_tool` allowlist. Every failure mode (missing token, invalid token,
        validation error) fails closed and raises `AgentAuthorizationError` — there is
        no code path where an analytics request silently ends up executing on a
        different agent.

        Privilege is read from Django's native `auth_user` booleans (`is_staff` /
        `is_superuser`) exactly as `validate_user_token` normalized them; `auth_user` has
        no `role` column, so no role strings are consumed here.

        Args:
            agent: The agent resolved by intent classification or explicit request.
            request: The incoming chat request.

        Returns:
            The original agent, unchanged, once authorization succeeds.

        Raises:
            AgentAuthorizationError: 401 when no `user_token` was presented at all, 403
                when a token was presented but does not carry staff privilege (invalid
                token, non-staff account, or a validation failure — all fail closed).
        """
        if agent.agent_id != "analytics":
            return agent

        auth_status: dict[str, Any] = {
            "authenticated": False,
            "user_id": None,
            "username": None,
            "is_staff": False,
            "is_superuser": False,
        }

        if request.user_token:
            try:
                # The analytics agent validates this same token again a moment later, in
                # `get_context_augmentation`. That double validation is cheap now:
                # `DjangoAPIService.validate_user_token` caches successful validations
                # in-process for a few seconds, so both layers hit the cache, not Django.
                validation = await agent.django_service.validate_user_token(request.user_token)
                if isinstance(validation, dict) and validation.get("valid"):
                    auth_status = {
                        "authenticated": True,
                        "user_id": validation.get("user_id"),
                        "username": validation.get("username"),
                        "is_staff": validation.get("is_staff"),
                        "is_superuser": validation.get("is_superuser"),
                    }
            except Exception as exc:
                logger.warning("Token validation failed during agent authorization: %s", exc)

        if _is_staff(auth_status):
            # Publish the verdict into the request-scoped, server-side auth channel.
            set_auth_status(auth_status)
            return agent

        logger.warning(
            "Rejecting analytics request for session '%s': caller is not staff.",
            request.session_id,
        )
        if not request.user_token:
            raise AgentAuthorizationError(
                status_code=401,
                message="Analytics access requires authentication. Please sign in with a staff account.",
            )
        raise AgentAuthorizationError(
            status_code=403,
            message="Analytics access requires staff privileges. Your account does not have the required role.",
        )

    @staticmethod
    def _accepts_event_sink(stream_callable: Any) -> bool:
        """Determines whether an agent's `process_stream` accepts an `event_sink` kwarg.

        Agents outside this refactor (and test doubles) still define `process_stream(self,
        request)`; passing an unexpected keyword to those would raise TypeError.

        Args:
            stream_callable: The bound `process_stream` method to inspect.

        Returns:
            True when the callable declares an `event_sink` parameter or accepts **kwargs.
        """
        try:
            parameters = inspect.signature(stream_callable).parameters
        except (TypeError, ValueError):
            return False

        if "event_sink" in parameters:
            return True
        return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    async def dispatch(self, request: ChatRequest) -> ChatResponse:
        """Dispatches request to resolved agent for non-streamed inference."""
        agent = self.resolve_agent(request.agent_id, request.message)
        agent = await self._authorize_agent(agent, request)
        # Ensure request reflects the FINAL executing agent (post-authorization)
        request.agent_id = agent.agent_id
        return await agent.process(request)

    async def dispatch_stream(
        self,
        request: ChatRequest,
        event_sink: Optional[EventSink] = None,
    ) -> tuple[BaseAgent, AsyncGenerator[str, None]]:
        """Resolves target agent and returns the token generator for streaming.

        Args:
            request: The incoming chat request.
            event_sink: Optional tool-progress callback forwarded to agents that support
                it; agents that do not are called with the legacy single-argument form.

        Returns:
            A tuple of the final authorized agent and its token generator.
        """
        agent = self.resolve_agent(request.agent_id, request.message)
        agent = await self._authorize_agent(agent, request)
        request.agent_id = agent.agent_id

        if event_sink is not None and self._accepts_event_sink(agent.process_stream):
            return agent, agent.process_stream(request, event_sink=event_sink)
        return agent, agent.process_stream(request)


# Global singleton instance
_dispatcher: Optional[AgentDispatcher] = None


def get_agent_dispatcher() -> AgentDispatcher:
    """Returns the singleton instance of AgentDispatcher with core agents registered."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AgentDispatcher(auto_register=True)
    return _dispatcher
