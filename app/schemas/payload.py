"""Pydantic schemas and DTOs for request/response payloads."""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Input payload for agent chat interactions."""

    agent_id: str = Field(
        ...,
        description="Target agent identifier ('portfolio', 'ecommerce', 'analytics', 'auto')",
        min_length=1,
        examples=["portfolio"],
    )
    session_id: str = Field(
        ...,
        description="Unique user session or conversation ID for memory persistence",
        min_length=1,
        examples=["sess_abc12345"],
    )
    message: str = Field(
        ...,
        description="User input prompt or message to be processed by the agent",
        min_length=1,
        examples=["¿Podrías mostrarme los servicios disponibles en tu portafolio?"],
    )
    stream: bool = Field(
        default=True,
        description="Determines whether the response should be streamed via Server-Sent Events (SSE)",
        examples=[True],
    )
    user_token: Optional[str] = Field(
        default=None,
        description="Optional JWT bearer token for authenticated user actions (e.g. AnalyticsAgent)",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata or client-side context (e.g. user preferences, route)",
        examples=[{"source": "web_widget", "theme": "dark"}],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "agent_id": "portfolio",
                "session_id": "sess_abc12345",
                "message": "¿Cuáles son tus proyectos más destacados?",
                "stream": True,
                "user_token": None,
                "context": {"source": "portfolio_header"},
            }
        },
    )


class ChatResponse(BaseModel):
    """Standard non-streamed response payload for agent chat interactions."""

    agent_id: str = Field(
        ...,
        description="Identifier of the agent that produced the response",
        examples=["portfolio"],
    )
    session_id: str = Field(
        ...,
        description="Session identifier associated with the conversation",
        examples=["sess_abc12345"],
    )
    message: str = Field(
        ...,
        description="AI Agent text response content",
        examples=["Aquí tienes la información sobre mis proyectos principales..."],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (model, latency, tool calls, token usage)",
        examples=[{"model": "gemini-3.7-flash", "intent": "portfolio_skills", "latency_ms": 280}],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "agent_id": "portfolio",
                "session_id": "sess_abc12345",
                "message": "Cuento con amplia experiencia en Python, FastAPI, Django y arquitecturas de microservicios.",
                "metadata": {
                    "model": "gemini-3.7-flash",
                    "intent": "portfolio_skills",
                    "latency_ms": 245,
                },
            }
        },
    )


class StreamTokenChunk(BaseModel):
    """Event payload emitted over Server-Sent Events (SSE) stream."""

    token: str = Field(
        default="",
        description="Text token chunk yielded by the LLM stream",
        examples=["Hola"],
    )
    agent_id: str = Field(
        ...,
        description="Identifier of the executing agent",
        examples=["portfolio"],
    )
    session_id: str = Field(
        ...,
        description="Conversation session identifier",
        examples=["sess_abc12345"],
    )
    done: bool = Field(
        default=False,
        description="Flag indicating whether generation is complete",
        examples=[False],
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Final execution metadata attached to the closing chunk",
        examples=[{"total_tokens": 120, "finish_reason": "STOP"}],
    )


class AgentInfo(BaseModel):
    """Metadata describing a registered AI agent."""

    agent_id: str = Field(..., description="Unique agent identifier", examples=["portfolio"])
    name: str = Field(..., description="Human-readable agent name", examples=["Portfolio Agent"])
    description: str = Field(
        ...,
        description="Agent specialization description",
        examples=["Responde sobre CV, experiencia, habilidades y proyectos."],
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Key capabilities and tools handled by the agent",
        examples=[["cv_inquiry", "skills_breakdown", "project_showcase"]],
    )


class AgentListResponse(BaseModel):
    """Response containing the list of all registered specialized agents."""

    agents: list[AgentInfo] = Field(..., description="List of registered agents in the gateway")


class MessageHistoryItem(BaseModel):
    """Individual conversation turn record stored in session memory."""

    role: str = Field(..., description="Message role: 'user', 'model', or 'system'", examples=["user"])
    content: str = Field(..., description="Text content of the message", examples=["Hola"])
    timestamp: str = Field(..., description="ISO 8601 timestamp of message creation")


class SessionHistoryResponse(BaseModel):
    """Payload returning conversational history for a session."""

    session_id: str = Field(..., description="Session identifier")
    messages: list[MessageHistoryItem] = Field(default_factory=list, description="Historical messages")
    total_messages: int = Field(default=0, description="Total count of stored turns")


class ClearHistoryResponse(BaseModel):
    """Response returned upon clearing conversation memory."""

    status: str = Field(default="ok", description="Status code")
    session_id: str = Field(..., description="Target session identifier")
    message: str = Field(..., description="Result message")


class HealthResponse(BaseModel):
    """Payload returned by the microservice healthcheck endpoint."""

    status: str = Field(default="ok", description="Overall health status of the service", examples=["ok"])
    app_name: str = Field(..., description="Application name", examples=["AI Agent Gateway"])
    environment: str = Field(..., description="Active runtime environment", examples=["development"])
    version: str = Field(..., description="API Version", examples=["0.1.0"])
    redis_healthy: Optional[bool] = Field(
        default=None,
        description="Status of Redis connection (populated only in detailed checks)",
    )
    django_healthy: Optional[bool] = Field(
        default=None,
        description="Status of Django monolith connectivity (populated only in detailed checks)",
    )
