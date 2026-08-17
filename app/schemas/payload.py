"""Pydantic schemas and DTOs for request/response payloads."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Input payload for agent chat interactions."""

    agent_id: str = Field(
        ...,
        description="Target agent identifier (e.g., 'portfolio', 'ecommerce', 'analytics')",
        min_length=1,
        examples=["portfolio"],
    )
    session_id: str = Field(
        ...,
        description="Unique user session or conversation ID",
        min_length=1,
        examples=["sess_abc12345"],
    )
    message: str = Field(
        ...,
        description="User input prompt or message",
        min_length=1,
        examples=["¿Podrías mostrarme los servicios disponibles?"],
    )
    stream: bool = Field(
        default=True,
        description="Determines whether the response should be streamed via Server-Sent Events (SSE)",
        examples=[True],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "agent_id": "portfolio",
                "session_id": "sess_abc12345",
                "message": "¿Podrías mostrarme los servicios disponibles?",
                "stream": True,
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
        description="Session identifier associated with the response",
        examples=["sess_abc12345"],
    )
    message: str = Field(
        ...,
        description="AI Agent text response content",
        examples=["Aquí tienes la información solicitada..."],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (latency, token usage, tool invocations, etc.)",
        examples=[{"tokens_used": 150, "model": "gemini-2.5-flash"}],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "agent_id": "portfolio",
                "session_id": "sess_abc12345",
                "message": "Aquí tienes la información solicitada...",
                "metadata": {
                    "tokens_used": 150,
                    "model": "gemini-2.5-flash",
                },
            }
        },
    )


class HealthResponse(BaseModel):
    """Payload returned by the microservice healthcheck endpoint."""

    status: str = Field(default="ok", description="Overall health status of the service", examples=["ok"])
    app_name: str = Field(..., description="Application name", examples=["AI Agent Gateway"])
    environment: str = Field(..., description="Active runtime environment", examples=["development"])
    version: str = Field(..., description="API Version", examples=["0.1.0"])
