"""Base Agent Abstract Definition."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.schemas.payload import ChatRequest, ChatResponse


class BaseAgent(ABC):
    """Abstract Base Class for specialized AI agents (Portfolio, Ecommerce, Analytics)."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        pass

    @abstractmethod
    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        pass
