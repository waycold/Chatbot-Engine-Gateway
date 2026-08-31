"""API-level contract tests for `AgentAuthorizationError` propagation.

Background: unauthorized analytics requests used to be silently downgraded to the
e-commerce agent and answered with HTTP 200. `AgentDispatcher._authorize_agent` now
raises `AgentAuthorizationError` (401 with no token, 403 with a non-staff token)
instead, and both chat entry points -- the JSON `POST /chat` endpoint and the SSE
`POST /chat/stream` endpoint -- must translate that into an explicit `HTTPException`
with the same status code and message, rather than a generic 500 or a silent 200.

These tests exercise the REAL `app.api.v1.chat` router (via the `sync_client` fixture,
which builds the actual FastAPI app), with only `get_agent_dispatcher` monkeypatched to
a stub dispatcher whose `dispatch` / `dispatch_stream` raise the real exception -- so
the assertion is on the router's own except-clause wiring, not on a reimplementation of
it.
"""
from typing import Any
import pytest
from fastapi.testclient import TestClient

from app.agents.exceptions import AgentAuthorizationError
from app.api.v1 import chat as chat_module
from app.core.config import settings

CHAT_URL = f"{settings.API_V1_STR}/chat"
STREAM_URL = f"{settings.API_V1_STR}/chat/stream"


class _RaisingDispatcher:
    """Dispatcher double whose entry points raise a canned AgentAuthorizationError."""

    def __init__(self, error: AgentAuthorizationError) -> None:
        self._error = error

    async def dispatch(self, request: Any) -> Any:
        raise self._error

    async def dispatch_stream(self, request: Any, event_sink: Any = None) -> Any:
        raise self._error


def install_raising_dispatcher(
    monkeypatch: pytest.MonkeyPatch, status_code: int, message: str,
) -> None:
    """Points the chat router at a dispatcher that always raises AgentAuthorizationError."""
    error = AgentAuthorizationError(status_code=status_code, message=message)
    monkeypatch.setattr(chat_module, "get_agent_dispatcher", lambda: _RaisingDispatcher(error))


def payload(agent_id: str = "analytics") -> dict[str, Any]:
    """Builds a minimal chat request body."""
    return {"agent_id": agent_id, "session_id": "sess_authz_qa", "message": "dame el reporte de ventas"}


class TestChatCompletionAuthorizationError:
    """Protects the JSON POST /chat endpoint's handling of AgentAuthorizationError."""

    def test_missing_token_returns_401(self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        install_raising_dispatcher(monkeypatch, 401, "Analytics access requires authentication.")

        response = sync_client.post(CHAT_URL, json=payload())

        assert response.status_code == 401
        assert response.json()["detail"] == "Analytics access requires authentication."

    def test_non_staff_token_returns_403(self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        install_raising_dispatcher(monkeypatch, 403, "Analytics access requires staff privileges.")

        response = sync_client.post(CHAT_URL, json=payload())

        assert response.status_code == 403
        assert response.json()["detail"] == "Analytics access requires staff privileges."


class TestChatStreamAuthorizationError:
    """Protects the SSE POST /chat/stream endpoint's handling of AgentAuthorizationError.

    `_authorize_agent` runs before the token generator is created, so the rejection
    surfaces as a plain HTTPException (never entering the StreamingResponse body).
    """

    def test_missing_token_returns_401(self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        install_raising_dispatcher(monkeypatch, 401, "Analytics access requires authentication.")

        response = sync_client.post(STREAM_URL, json={**payload(), "stream": True})

        assert response.status_code == 401
        assert response.json()["detail"] == "Analytics access requires authentication."

    def test_non_staff_token_returns_403(self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        install_raising_dispatcher(monkeypatch, 403, "Analytics access requires staff privileges.")

        response = sync_client.post(STREAM_URL, json={**payload(), "stream": True})

        assert response.status_code == 403
        assert response.json()["detail"] == "Analytics access requires staff privileges."
