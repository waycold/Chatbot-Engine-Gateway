"""Integration and Contract tests for Chat Endpoints (Standard POST and SSE Streaming)."""
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import httpx
from app.agents.dispatcher import AgentDispatcher
from app.core.security import verify_internal_api_secret
from app.schemas.payload import ChatRequest, ChatResponse
from tests.test_agents import MockEcommerceAgent, MockPortfolioAgent


# ==============================================================================
# Sample App Setup with Chat Endpoints for Comprehensive Contract Validation
# ==============================================================================

def create_test_chat_router() -> APIRouter:
    """Creates a configured Chat Router with both standard and SSE streaming logic."""
    router = APIRouter(prefix="/chat", tags=["Chat & Agents"])
    dispatcher = AgentDispatcher()
    dispatcher.register("portfolio", MockPortfolioAgent("portfolio"))
    dispatcher.register("ecommerce", MockEcommerceAgent("ecommerce"))

    @router.post(
        "",
        summary="Unified Chat Endpoint (Standard & SSE Streaming)",
        response_model=ChatResponse,
    )
    async def chat_handler(request: ChatRequest):
        try:
            agent = dispatcher.get(request.agent_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{request.agent_id}' not found.",
            )

        if request.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                try:
                    async for chunk in agent.process_stream(request):
                        yield f"data: {json.dumps({'chunk': chunk, 'agent_id': request.agent_id})}\n\n"
                    yield f"data: {json.dumps({'event': 'done', 'session_id': request.session_id})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                response = await agent.process(request)
                return response
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent execution failed: {str(e)}",
                )

    return router


@pytest.fixture
def chat_app(app: FastAPI) -> FastAPI:
    """FastAPI app instance with registered test chat router for contract verification."""
    chat_router = create_test_chat_router()
    app.include_router(chat_router, prefix="/api/v1/test-chat")
    return app


@pytest.fixture
def chat_sync_client(chat_app: FastAPI) -> TestClient:
    """Synchronous test client for chat app."""
    return TestClient(chat_app, base_url="http://testserver")


@pytest.fixture
async def chat_async_client(chat_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Asynchronous test client for chat app."""
    transport = httpx.ASGITransport(app=chat_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ==============================================================================
# Non-Streaming Chat Endpoint Tests
# ==============================================================================

class TestStandardChatEndpoint:
    """Test suite for standard non-streamed POST /chat interactions."""

    def test_standard_chat_success(
        self,
        chat_sync_client: TestClient,
        sample_chat_request_non_stream_payload: dict,
    ) -> None:
        """Verifies that stream=False returns HTTP 200 with complete ChatResponse JSON."""
        response = chat_sync_client.post(
            "/api/v1/test-chat/chat",
            json=sample_chat_request_non_stream_payload,
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["agent_id"] == "ecommerce"
        assert data["session_id"] == sample_chat_request_non_stream_payload["session_id"]
        assert "[Ecommerce]" in data["message"]
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_standard_chat_async_success(
        self,
        chat_async_client: httpx.AsyncClient,
        sample_chat_request_non_stream_payload: dict,
    ) -> None:
        """Verifies asynchronous standard chat call returns valid ChatResponse."""
        response = await chat_async_client.post(
            "/api/v1/test-chat/chat",
            json=sample_chat_request_non_stream_payload,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "ecommerce"

    def test_standard_chat_unregistered_agent_404(self, chat_sync_client: TestClient) -> None:
        """Verifies that requesting an unknown agent returns HTTP 404 Not Found."""
        payload = {
            "agent_id": "unknown_agent_xyz",
            "session_id": "sess_001",
            "message": "Hola",
            "stream": False,
        }
        response = chat_sync_client.post("/api/v1/test-chat/chat", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


# ==============================================================================
# Streaming SSE Chat Endpoint Tests
# ==============================================================================

class TestStreamingChatEndpoint:
    """Test suite for Server-Sent Events (SSE) streaming chat interactions."""

    def test_streaming_chat_headers(
        self,
        chat_sync_client: TestClient,
        sample_chat_request_payload: dict,
    ) -> None:
        """Verifies that stream=True returns HTTP 200 with text/event-stream content type."""
        response = chat_sync_client.post(
            "/api/v1/test-chat/chat",
            json=sample_chat_request_payload,
        )
        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert response.headers.get("cache-control") == "no-cache"

    @pytest.mark.asyncio
    async def test_streaming_chat_async_sse_chunks(
        self,
        chat_async_client: httpx.AsyncClient,
        sample_chat_request_payload: dict,
    ) -> None:
        """Verifies parsing of incoming SSE chunk stream."""
        response = await chat_async_client.post(
            "/api/v1/test-chat/chat",
            json=sample_chat_request_payload,
        )
        assert response.status_code == status.HTTP_200_OK

        events = []
        raw_text = response.text
        lines = raw_text.strip().split("\n\n")

        for line in lines:
            if line.startswith("data: "):
                data_str = line.replace("data: ", "", 1)
                parsed = json.loads(data_str)
                events.append(parsed)

        # Check received events
        assert len(events) >= 2
        chunk_events = [e for e in events if "chunk" in e]
        assert len(chunk_events) > 0
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["session_id"] == sample_chat_request_payload["session_id"]


# ==============================================================================
# Payload Validation & Error Handling Tests (422, 500, Rate Limit)
# ==============================================================================

class TestChatValidationAndErrors:
    """Test suite for input validation, malformed payloads, and server errors."""

    @pytest.mark.parametrize(
        "invalid_payload,expected_loc",
        [
            ({"session_id": "s1", "message": "m1"}, "agent_id"),
            ({"agent_id": "portfolio", "message": "m1"}, "session_id"),
            ({"agent_id": "portfolio", "session_id": "s1"}, "message"),
            ({"agent_id": "", "session_id": "s1", "message": "m1"}, "agent_id"),
            ({"agent_id": "portfolio", "session_id": "", "message": "m1"}, "session_id"),
            ({"agent_id": "portfolio", "session_id": "s1", "message": ""}, "message"),
        ],
    )
    def test_chat_invalid_payload_returns_422(
        self,
        chat_sync_client: TestClient,
        invalid_payload: dict,
        expected_loc: str,
    ) -> None:
        """Verifies that missing or empty fields return HTTP 422 Unprocessable Entity."""
        response = chat_sync_client.post("/api/v1/test-chat/chat", json=invalid_payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        errors = response.json().get("detail", [])
        assert any(expected_loc in str(err.get("loc", [])) for err in errors)

    def test_chat_malformed_json_returns_422(self, chat_sync_client: TestClient) -> None:
        """Verifies that non-JSON body returns HTTP 422."""
        response = chat_sync_client.post(
            "/api/v1/test-chat/chat",
            content="invalid-plain-text-body",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_chat_agent_execution_error_500(self, chat_app: FastAPI) -> None:
        """Verifies that unhandled agent exceptions in non-streaming mode return HTTP 500."""
        # Attach a router with a failing agent
        failing_router = APIRouter(prefix="/failing-chat")
        
        @failing_router.post("")
        async def failing_endpoint(request: ChatRequest):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent processing internal failure: Out of memory",
            )

        chat_app.include_router(failing_router, prefix="/api/v1")

        transport = httpx.ASGITransport(app=chat_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/failing-chat",
                json={"agent_id": "portfolio", "session_id": "s1", "message": "test"},
            )
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "internal failure" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error_429(self, chat_app: FastAPI) -> None:
        """Verifies upstream Rate Limit / Quota Exceeded handling returning HTTP 429."""
        rate_limited_router = APIRouter(prefix="/rate-limited-chat")

        @rate_limited_router.post("")
        async def rate_limited_endpoint(request: ChatRequest):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Google GenAI Quota Exceeded (429 Resource Exhausted)",
            )

        chat_app.include_router(rate_limited_router, prefix="/api/v1")

        transport = httpx.ASGITransport(app=chat_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/rate-limited-chat",
                json={"agent_id": "portfolio", "session_id": "s1", "message": "test"},
            )
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "Quota Exceeded" in response.json()["detail"]
