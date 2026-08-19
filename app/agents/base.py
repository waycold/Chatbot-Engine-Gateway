"""Base Agent Abstract Definition and Common Execution Engine."""
from abc import ABC, abstractmethod
import logging
import time
from typing import Any, AsyncGenerator, Optional
from app.core.config import settings
from app.schemas.payload import AgentInfo, ChatRequest, ChatResponse
from app.services.django_api import DjangoAPIService, get_django_api_service
from app.services.llm_client import LLMClientService, get_llm_service
from app.services.memory import RedisMemoryService, get_memory_service

logger = logging.getLogger("ai_gateway.agent.base")


class BaseAgent(ABC):
    """Abstract Base Class for specialized AI agents.

    Provides common plumbing for memory retrieval, system prompt formatting,
    context grounding, LLM invocation (streaming & non-streaming), and memory persistence.
    """

    def __init__(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        llm_service: Optional[LLMClientService] = None,
        memory_service: Optional[RedisMemoryService] = None,
        django_service: Optional[DjangoAPIService] = None,
    ) -> None:
        self.agent_id = agent_id
        self.name = name or agent_id.capitalize()
        self.description = description or f"{agent_id} specialized agent"
        self.capabilities = capabilities or []
        self.llm_service = llm_service or get_llm_service()
        self.memory_service = memory_service or get_memory_service()
        self.django_service = django_service or get_django_api_service()

    def get_info(self) -> AgentInfo:
        """Returns the metadata descriptor for this agent."""
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
        )

    async def get_system_instruction(self, request: ChatRequest) -> str:
        """Constructs the specialized system instruction prompt for this agent."""
        return f"You are the {self.name}. Answer user questions accurately."

    async def get_context_augmentation(self, request: ChatRequest) -> Optional[str]:
        """Hook for child agents to inject external/RAG context (e.g. catalog, portfolio, analytics)."""
        return None

    async def build_conversation_contents(self, request: ChatRequest) -> list[dict[str, Any]]:
        """Retrieves session history and compiles formatted contents for Gemini inference.

        Returns a list of message dicts formatted for the LLM.
        """
        history = await self.memory_service.get_history(request.session_id, limit=8)
        contents: list[dict[str, Any]] = []

        for item in history:
            role = "user" if item.get("role") == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": item.get("content", "")}],
            })

        # Inject RAG context augmentation if present
        context_aug = await self.get_context_augmentation(request)
        user_message_text = request.message
        if context_aug:
            user_message_text = f"[Context / Grounding Data]:\n{context_aug}\n\n[User Query]:\n{request.message}"

        contents.append({
            "role": "user",
            "parts": [{"text": user_message_text}],
        })

        return contents

    async def save_turn(self, session_id: str, user_message: str, model_response: str) -> None:
        """Persists the turn (user message + model response) into session memory."""
        try:
            await self.memory_service.add_message(session_id=session_id, role="user", content=user_message)
            await self.memory_service.add_message(session_id=session_id, role="model", content=model_response)
        except Exception as exc:
            logger.warning("Failed to save conversation turn for session %s: %s", session_id, exc)

    async def _execute_process(self, request: ChatRequest) -> ChatResponse:
        """Helper to run standard non-streamed inference turn."""
        start_time = time.perf_counter()
        system_instruction = await self.get_system_instruction(request)
        contents = await self.build_conversation_contents(request)

        response_text = await self.llm_service.generate_content(
            contents=contents,
            system_instruction=system_instruction,
            model=settings.DEFAULT_MODEL,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Persist turn in short-term memory
        await self.save_turn(
            session_id=request.session_id,
            user_message=request.message,
            model_response=response_text,
        )

        return ChatResponse(
            agent_id=self.agent_id,
            session_id=request.session_id,
            message=response_text,
            metadata={
                "agent_name": self.name,
                "model": settings.DEFAULT_MODEL,
                "latency_ms": elapsed_ms,
            },
        )

    async def _execute_process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Helper to run streaming inference turn."""
        system_instruction = await self.get_system_instruction(request)
        contents = await self.build_conversation_contents(request)

        collected_tokens: list[str] = []

        async for token in self.llm_service.generate_content_stream(
            contents=contents,
            system_instruction=system_instruction,
            model=settings.DEFAULT_MODEL,
        ):
            collected_tokens.append(token)
            yield token

        # Persist full assembled response turn
        full_response = "".join(collected_tokens)
        if full_response:
            await self.save_turn(
                session_id=request.session_id,
                user_message=request.message,
                model_response=full_response,
            )

    @abstractmethod
    async def process(self, request: ChatRequest) -> ChatResponse:
        """Processes a chat request and returns a complete response."""
        pass

    @abstractmethod
    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat request and yields SSE chunk tokens."""
        pass
