"""Agents domain layer and dispatching mechanisms."""
from app.agents.analytics import AnalyticsAgent
from app.agents.base import BaseAgent, EventSink
from app.agents.dispatcher import AgentDispatcher, get_agent_dispatcher
from app.agents.ecommerce import EcommerceAgent
from app.agents.portfolio import PortfolioAgent
from app.agents.tools import (
    ALL_TOOL_DECLARATIONS,
    ANALYTICS_TOOL_DECLARATIONS,
    CATALOG_RAG_TOOL_DECLARATIONS,
    SQL_SANDBOX_TOOL_NAME,
    TOOL_PROGRESS_LABELS,
    execute_tool,
    get_tool_label,
)

__all__ = [
    "BaseAgent",
    "EventSink",
    "PortfolioAgent",
    "EcommerceAgent",
    "AnalyticsAgent",
    "AgentDispatcher",
    "get_agent_dispatcher",
    "ANALYTICS_TOOL_DECLARATIONS",
    "CATALOG_RAG_TOOL_DECLARATIONS",
    "ALL_TOOL_DECLARATIONS",
    "TOOL_PROGRESS_LABELS",
    "SQL_SANDBOX_TOOL_NAME",
    "get_tool_label",
    "execute_tool",
]
