"""Agent Dispatcher module for routing requests to specialized agents."""
from typing import Dict
from app.agents.base import BaseAgent


class AgentDispatcher:
    """Registry and dispatcher for specialized AI agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent_id: str, agent: BaseAgent) -> None:
        """Registers an agent instance under a specific identifier."""
        self._agents[agent_id] = agent

    def get(self, agent_id: str) -> BaseAgent:
        """Retrieves the agent instance corresponding to the agent_id."""
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' is not registered.")
        return self._agents[agent_id]
