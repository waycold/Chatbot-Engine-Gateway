"""Agent Dispatcher and Intent Orchestrator."""
import inspect
import logging
import re
from typing import Any, AsyncGenerator, Optional
from app.agents.analytics import AnalyticsAgent, _is_staff, set_auth_status
from app.agents.base import BaseAgent, EventSink
from app.agents.ecommerce import EcommerceAgent
from app.agents.exceptions import AgentAuthorizationError
from app.agents.portfolio import PortfolioAgent
from app.schemas.payload import AgentInfo, ChatRequest, ChatResponse

logger = logging.getLogger("ai_gateway.dispatcher")


# --- Intent classification keyword sets --------------------------------------------
#
# E-Commerce signals: catalog, pricing, purchase, shipping/stock inquiries.
ECOMMERCE_KEYWORDS: list[str] = [
    "producto", "productos", "comprar", "precio", "precios", "catálogo", "catalogo",
    "curso", "cursos", "stock", "cuanto cuesta", "cuánto cuesta", "costo", "costos",
    "tienda", "order", "buy", "price", "shop", "adquirir", "contratar servicio",
]

# Analytics signals: KPIs, sales/revenue, margins, rankings, funnels, segmentation.
ANALYTICS_KEYWORDS: list[str] = [
    "métrica", "metrica", "métricas", "metricas", "kpi", "kpis", "analítica", "analitica",
    "estadística", "estadisticas", "estadísticas", "ventas del mes", "conversión",
    "conversion", "tráfico", "trafico", "rendimiento", "analytics", "dashboard", "reporte",
    "revenue", "usuarios activos",
    # EN/ES terms for "best-selling" / ranking / margin phrasing that were previously
    # missing, causing these queries to fall through to "ecommerce" or "portfolio"
    # instead of "analytics" (diagnostico-plan-agentes-multi-agente.md, causa raíz A2/A3).
    "best-selling", "best seller", "best sellers", "bestseller", "bestsellers",
    "top selling", "top-selling", "top sales", "top products", "most sold",
    "highest revenue", "sales ranking", "product ranking",
    "rentabilidad", "márgenes", "margenes", "margen",
    "más vendido", "más vendidos", "mas vendido", "mas vendidos",
    "mejor vendido", "mejores vendidos", "ranking de productos",
    "embudo", "segmentación", "segmentacion", "trending",
    # Bare EN/ES domain terms already recognized as analytics triggers by
    # AnalyticsAgent.get_context_augmentation's own branch matching (analytics.py,
    # margins/customer-segmentation/sales branches) but previously absent from the
    # dispatcher's own keyword list, so a query using only these words (e.g. "margin
    # by category", "customer RFM") never reached the analytics agent in the first
    # place — closing this gap is a direct extension of the same causa raíz (A2/A3).
    # Deliberately NOT adding bare "customer"/"customers": those are too generic and
    # would tie with (or lose to) plain ecommerce/support phrasing like "I'm a
    # customer, where is my order?" — "rfm" alone already routes the canonical
    # "customer RFM" test case correctly without that false-positive risk.
    "margin", "margins", "rfm",
    "venta", "ventas", "vendido", "vendidos",
]

# High-confidence intent -> tool hints. This reinforces (never replaces) Gemini's own
# function-calling: `run_tool_loop` still receives the full ANALYTICS_TOOL_DECLARATIONS
# and decides for itself which tool(s) to call for a given turn. These hints exist so a
# caller that wants a low-latency, pre-model signal (e.g. AnalyticsAgent, or
# logging/telemetry) can know which tool a phrase is most likely asking for, without
# waiting on an LLM round trip. Patterns are matched case-insensitively against the raw
# message via `get_strong_tool_hint`.
STRONG_TOOL_HINTS: dict[str, str] = {
    r'\bbest[- ]?sell(ing|er)': "get_product_profitability",
    r'\bmás vendid': "get_product_profitability",
    r'\bmas vendid': "get_product_profitability",
    r'\btop\s+\d*\s*produc': "get_product_profitability",
}


def _matches_any(msg_lower: str, keywords: list[str]) -> bool:
    """Whole-word/phrase match: True if any keyword occurs in `msg_lower` bounded by
    word edges (`\\b`), never as a bare substring.

    This is what prevents false positives like "recurso" matching the "curso"
    ecommerce keyword — `\\bcurso\\b` does not match inside "recurso" because there is
    no word boundary between "re" and "curso".

    Args:
        msg_lower: The already-lowercased message to search.
        keywords: Candidate keywords/phrases to look for.

    Returns:
        True if any keyword matches as a whole word/phrase.
    """
    return any(re.search(rf'\b{re.escape(kw)}\b', msg_lower) for kw in keywords)


def get_strong_tool_hint(message: str) -> Optional[str]:
    """Returns the high-confidence tool name for a message, if any `STRONG_TOOL_HINTS`
    pattern matches.

    This is a reinforcement signal, not a routing decision or a tool-execution
    shortcut: it never calls a tool itself, and it does not gate or replace Gemini's
    function-calling inside `run_tool_loop`. It exists as a hook that `AnalyticsAgent`
    (or its context-augmentation logic) can consult when it wants a fast, pre-model
    guess at which tool a phrase is asking for — e.g. to bias a pre-fetch or to log
    routing confidence — without reworking the existing tool-selection logic.

    Args:
        message: The raw user message (case handled internally).

    Returns:
        The hinted tool name, or None when no pattern matches.
    """
    msg_lower = message.lower()
    for pattern, tool_name in STRONG_TOOL_HINTS.items():
        if re.search(pattern, msg_lower):
            return tool_name
    return None


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

        Routes message to 'portfolio', 'ecommerce', or 'analytics' using weighted
        keyword scoring with word-boundary matching (`_matches_any`) instead of naive
        substring containment — the previous `kw in msg_lower` check produced false
        positives such as "recurso" matching the "curso" ecommerce keyword.

        Scoring, not binary precedence: both keyword lists are always evaluated, and
        analytics wins whenever it has at least one hit and its hit count is greater
        than or equal to ecommerce's. This fixes messages with strong analytical
        signal that also contain a generic catalog word — e.g. "revenue de mis
        productos" previously matched "productos" first and returned early as
        "ecommerce", ignoring the explicit "revenue" signal. Ecommerce only wins when
        it has hits and analytics does not have an equal-or-greater count.

        This is a low-latency hint for the first-render UX and does not decide which
        tool actually executes — that remains Gemini's function-calling job inside
        `run_tool_loop`, optionally reinforced by `STRONG_TOOL_HINTS` /
        `get_strong_tool_hint`.

        Defaults to 'portfolio' for general greeting/profile inquiries with no hits
        in either list.
        """
        msg_lower = message.lower().strip()

        ecommerce_hits = sum(1 for kw in ECOMMERCE_KEYWORDS if _matches_any(msg_lower, [kw]))
        analytics_hits = sum(1 for kw in ANALYTICS_KEYWORDS if _matches_any(msg_lower, [kw]))

        if analytics_hits > 0 and analytics_hits >= ecommerce_hits:
            return "analytics"
        if ecommerce_hits > 0:
            return "ecommerce"

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
