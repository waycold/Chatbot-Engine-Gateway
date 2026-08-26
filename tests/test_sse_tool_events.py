"""Contract tests for the SSE stream after tool-progress multiplexing.

Requires FastAPI, so this suite only runs in the project venv.

`/api/v1/chat/stream` gained `event: tool_start` / `event: tool_end` frames, pushed
through the same `asyncio.Queue` as the token frames. Two things are asserted, and the
second matters more than the first:

  1. the new frames appear, are valid JSON, and are correctly ordered; and
  2. the PRE-EXISTING contract is byte-for-byte unchanged — anonymous `data: {...}`
     token frames, the final `done: true` chunk, and the trailing `data: [DONE]`.

The frontend's SSE reader was written against that original contract. An extra named
event is additive and safe for a compliant reader, but reordering, renaming, or
dropping the `[DONE]` sentinel would leave the widget spinning forever with a fully
successful HTTP 200 in the logs.
"""
import json
from typing import Any, AsyncGenerator, Optional
import pytest
from fastapi.testclient import TestClient

from app.agents.base import BaseAgent, EventSink
from app.agents.dispatcher import AgentDispatcher
from app.api.v1 import chat as chat_module
from app.core.config import settings
from app.schemas.payload import ChatRequest, ChatResponse

STREAM_URL = f"{settings.API_V1_STR}/chat/stream"


# ==============================================================================
# A scripted agent that emits a two-tool chain plus tokens
# ==============================================================================

class ToolChainAgent(BaseAgent):
    """Agent double that emits a realistic chained-tool stream through the event sink.

    Its `process_stream` accepts `event_sink` as a keyword-only argument, which is what
    `AgentDispatcher._accepts_event_sink` inspects before forwarding the sink.
    """

    def __init__(self, agent_id: str = "ecommerce", fail_midway: bool = False) -> None:
        super().__init__(agent_id=agent_id, name="Tool Chain Test Agent")
        self.fail_midway = fail_midway

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Unused by the streaming tests, but required by the abstract base."""
        return ChatResponse(agent_id=self.agent_id, session_id=request.session_id, message="ok")

    async def process_stream(
        self,
        request: ChatRequest,
        *,
        event_sink: Optional[EventSink] = None,
    ) -> AsyncGenerator[str, None]:
        """Emits two full tool cycles, then three answer tokens."""
        for tool_name, label in (
            ("semantic_catalog_search", "Buscando en el catálogo..."),
            ("check_stock_and_price", "Verificando stock y precio..."),
        ):
            if event_sink is not None:
                await event_sink("tool_start", {"tool": tool_name, "label": label})
                await event_sink("tool_end", {"tool": tool_name, "ok": True})

        if self.fail_midway:
            raise RuntimeError("el proveedor de modelos falló a mitad del stream")

        for token in ("El curso ", "de FastAPI ", "está disponible."):
            yield token


class ToollessAgent(BaseAgent):
    """Agent double with the legacy single-argument `process_stream` signature.

    Agents outside this refactor (and older test doubles) still define
    `process_stream(self, request)`. Passing them an unexpected keyword would raise a
    TypeError, so the dispatcher must detect the signature and call the legacy form.
    """

    def __init__(self, agent_id: str = "portfolio") -> None:
        super().__init__(agent_id=agent_id, name="Legacy Signature Agent")

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Unused by the streaming tests."""
        return ChatResponse(agent_id=self.agent_id, session_id=request.session_id, message="ok")

    async def process_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Yields plain tokens with no progress events at all."""
        for token in ("Hola ", "soy ", "el portafolio."):
            yield token


def install_dispatcher(monkeypatch: pytest.MonkeyPatch, agent: BaseAgent) -> AgentDispatcher:
    """Points the chat router at a dispatcher containing only the given agent."""
    dispatcher = AgentDispatcher()
    dispatcher.register(agent.agent_id, agent)
    monkeypatch.setattr(chat_module, "get_agent_dispatcher", lambda: dispatcher)
    return dispatcher


def payload(agent_id: str = "ecommerce", message: str = "¿tenés el curso de FastAPI?") -> dict[str, Any]:
    """Builds a streaming chat request payload."""
    return {"agent_id": agent_id, "session_id": "sess_sse_qa", "message": message, "stream": True}


# ==============================================================================
# SSE frame parsing
# ==============================================================================

def parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parses a raw SSE body into ordered `{"event", "data"}` frames.

    Written by hand rather than with an SSE library so the test asserts on the exact
    bytes the browser receives, including the anonymous-event default.
    """
    frames: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name: Optional[str] = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        frames.append({"event": event_name, "data": "\n".join(data_lines)})
    return frames


def stream_body(client: TestClient, body: dict[str, Any]) -> str:
    """POSTs to the stream endpoint and returns the raw SSE text."""
    response = client.post(STREAM_URL, json=body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    return response.text


# ==============================================================================
# The new tool-progress frames
# ==============================================================================

class TestToolProgressFrames:
    """Protects the newly multiplexed tool_start / tool_end frames."""

    def test_tool_start_and_tool_end_frames_are_present(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the reason these events exist: a 4-7s chain must not look frozen."""
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        names = [frame["event"] for frame in frames if frame["event"]]

        assert names.count("tool_start") == 2
        assert names.count("tool_end") == 2

    def test_tool_frames_carry_valid_json_with_the_expected_keys(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the payload contract the widget renders its progress rows from.

        A missing `label` renders a blank row; a non-JSON payload throws inside the
        client's `onmessage`, which in most browsers kills the whole EventSource.
        """
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))

        starts = [json.loads(frame["data"]) for frame in frames if frame["event"] == "tool_start"]
        ends = [json.loads(frame["data"]) for frame in frames if frame["event"] == "tool_end"]

        assert [start["tool"] for start in starts] == ["semantic_catalog_search", "check_stock_and_price"]
        assert all(isinstance(start["label"], str) and start["label"] for start in starts)
        assert [end["tool"] for end in ends] == ["semantic_catalog_search", "check_stock_and_price"]
        assert all(end["ok"] is True for end in ends)

    def test_tool_frames_carry_the_routing_identifiers(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects multi-session clients from attributing progress to the wrong chat."""
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        bodies = [json.loads(f["data"]) for f in frames if f["event"] in {"tool_start", "tool_end"}]

        assert all(body["session_id"] == "sess_sse_qa" for body in bodies)
        assert all(body["agent_id"] == "ecommerce" for body in bodies)

    def test_start_precedes_end_for_every_tool(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects per-tool ordering, which the shared single-consumer queue guarantees.

        Two independent writers into the response would race and could deliver an `end`
        before its `start`, leaving a progress row that never clears.
        """
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        sequence = [
            (frame["event"], json.loads(frame["data"])["tool"])
            for frame in frames
            if frame["event"] in {"tool_start", "tool_end"}
        ]

        open_tools: set[str] = set()
        for event_name, tool in sequence:
            if event_name == "tool_start":
                assert tool not in open_tools, f"'{tool}' started twice without ending"
                open_tools.add(tool)
            else:
                assert tool in open_tools, f"'{tool}' ended before it started"
                open_tools.remove(tool)

        assert not open_tools, f"tools left without a tool_end frame: {open_tools}"

    def test_tool_frames_precede_the_answer_tokens(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the user-visible sequence: progress first, then the answer.

        This is the property the single shared queue exists to guarantee; two separate
        writers could interleave progress rows into the middle of the prose.
        """
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        kinds = [
            "progress" if frame["event"] in {"tool_start", "tool_end"} else "data"
            for frame in frames
        ]
        last_progress = max(index for index, kind in enumerate(kinds) if kind == "progress")
        first_token = min(
            index
            for index, frame in enumerate(frames)
            if frame["event"] is None and frame["data"] != "[DONE]" and json.loads(frame["data"]).get("token")
        )

        assert last_progress < first_token


# ==============================================================================
# Backwards compatibility of the original contract
# ==============================================================================

class TestStreamBackwardsCompatibility:
    """Protects every part of the pre-existing SSE contract from the new frames."""

    def test_token_frames_are_still_anonymous_data_events(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the frontend's `onmessage` handler.

        Token frames must stay anonymous `data:` events. Giving them a name would route
        them to a named listener the widget does not register, and the chat would render
        nothing at all while the request still returned 200.
        """
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        token_frames = [
            json.loads(frame["data"])
            for frame in frames
            if frame["event"] is None and frame["data"] != "[DONE]"
        ]

        assert [chunk["token"] for chunk in token_frames if not chunk["done"]] == [
            "El curso ", "de FastAPI ", "está disponible.",
        ]
        assert all(chunk["session_id"] == "sess_sse_qa" for chunk in token_frames)
        assert all(chunk["agent_id"] == "ecommerce" for chunk in token_frames)

    def test_done_chunk_is_still_emitted_with_metadata(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the closing chunk clients use to stop their spinner."""
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        done_chunks = [
            json.loads(frame["data"])
            for frame in frames
            if frame["event"] is None and frame["data"] != "[DONE]" and json.loads(frame["data"])["done"]
        ]

        assert len(done_chunks) == 1
        assert done_chunks[0]["token"] == ""
        assert done_chunks[0]["metadata"]["total_tokens_yielded"] == 3
        assert "latency_ms" in done_chunks[0]["metadata"]

    def test_trailing_done_sentinel_is_the_last_frame(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the `data: [DONE]` sentinel, verbatim and last.

        Clients written against the OpenAI-style convention close the connection on
        this exact string. Losing it leaves the widget waiting forever on a stream the
        server considers finished.
        """
        install_dispatcher(monkeypatch, ToolChainAgent())

        raw = stream_body(sync_client, payload())
        frames = parse_sse(raw)

        assert raw.endswith("data: [DONE]\n\n")
        assert frames[-1] == {"event": None, "data": "[DONE]"}
        assert frames[-2]["event"] is None
        assert json.loads(frames[-2]["data"])["done"] is True

    def test_progress_frames_do_not_inflate_the_token_count(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the telemetry in the closing chunk from counting progress rows."""
        install_dispatcher(monkeypatch, ToolChainAgent())

        frames = parse_sse(stream_body(sync_client, payload()))
        done_chunk = json.loads(frames[-2]["data"])

        assert done_chunk["metadata"]["total_tokens_yielded"] == 3

    def test_stream_without_any_tool_calls_is_byte_compatible(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects the far more common no-tool turn from any new frames at all.

        Most portfolio and FAQ turns run no tools. Those streams must look exactly as
        they did before this change.
        """
        install_dispatcher(monkeypatch, ToollessAgent())

        frames = parse_sse(stream_body(sync_client, payload(agent_id="portfolio", message="hola")))

        assert all(frame["event"] is None for frame in frames), "no named events on a tool-less turn"
        assert frames[-1]["data"] == "[DONE]"
        assert json.loads(frames[-2]["data"])["done"] is True

    def test_legacy_agent_signature_is_still_supported(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects agents whose `process_stream` never learned about `event_sink`.

        Forwarding the sink unconditionally would raise TypeError for those agents and
        turn every one of their turns into a stream error.
        """
        install_dispatcher(monkeypatch, ToollessAgent())

        frames = parse_sse(stream_body(sync_client, payload(agent_id="portfolio", message="hola")))
        tokens = [
            json.loads(frame["data"])["token"]
            for frame in frames
            if frame["event"] is None and frame["data"] != "[DONE]"
        ]

        assert "".join(tokens) == "Hola soy el portafolio."


# ==============================================================================
# Failure inside the stream
# ==============================================================================

class TestStreamErrorHandling:
    """Protects the error contract when the agent fails mid-stream."""

    def test_mid_stream_failure_emits_an_error_event(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects clients from a stream that simply stops with no explanation.

        The response status is already 200 by the time the failure happens, so the only
        way to report it is an in-band `event: error` frame.
        """
        install_dispatcher(monkeypatch, ToolChainAgent(fail_midway=True))

        frames = parse_sse(stream_body(sync_client, payload()))
        error_frames = [json.loads(frame["data"]) for frame in frames if frame["event"] == "error"]

        assert len(error_frames) == 1
        assert error_frames[0]["status_code"] == 500
        assert error_frames[0]["done"] is True

    def test_progress_frames_emitted_before_a_failure_are_still_delivered(
        self, sync_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protects partial progress from being swallowed by the later failure.

        The events already queued describe work that genuinely happened; dropping them
        would leave the user with an unexplained error and no idea how far it got.
        """
        install_dispatcher(monkeypatch, ToolChainAgent(fail_midway=True))

        frames = parse_sse(stream_body(sync_client, payload()))
        names = [frame["event"] for frame in frames if frame["event"]]

        assert names.count("tool_start") == 2
        assert names[-1] == "error"
