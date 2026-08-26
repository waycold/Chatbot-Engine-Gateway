"""API v1 Chat router definition with streaming SSE and standard JSON endpoints."""
import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import StreamingResponse
from app.agents.dispatcher import get_agent_dispatcher
from app.schemas.payload import (
    AgentListResponse,
    ChatRequest,
    ChatResponse,
    ClearHistoryResponse,
    MessageHistoryItem,
    SessionHistoryResponse,
    StreamTokenChunk,
)
from app.services.llm_client import LLMRateLimitError, LLMServiceError, LLMTimeoutError
from app.services.memory import get_memory_service

logger = logging.getLogger("ai_gateway.api.chat")

router = APIRouter(prefix="/chat", tags=["Chat & Agents"])

# Internal queue message kinds multiplexed onto the single SSE consumer.
_KIND_TOKEN = "token"
_KIND_PROGRESS = "progress"
_KIND_END = "end"
_KIND_ERROR = "error"


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Standard Agent Chat Completion",
    description="Processes a user message and returns a complete, structured JSON response from the resolved agent.",
)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """Synchronous/standard chat endpoint."""
    dispatcher = get_agent_dispatcher()
    try:
        response = await dispatcher.dispatch(request)
        return response
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.message,
        ) from exc
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in chat_completion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the chat request.",
        ) from exc


@router.post(
    "/stream",
    summary="Real-time Agent Chat Streaming (SSE)",
    description="Streams LLM token chunks in real-time using Server-Sent Events (text/event-stream).",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream yielding token events and completion signal.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint.

    Chained tool calls take 4-7s warm and far longer when Render and Neon are both cold;
    without progress events the stream looks frozen to the client. Tokens and tool
    progress events are therefore multiplexed onto ONE queue with a single consumer, so
    their relative ordering is preserved and there is no interleaving race.
    """
    dispatcher = get_agent_dispatcher()

    # Created here (before dispatch) because the agent needs the sink up front. It binds
    # to the running loop, which is the same loop that will drain the StreamingResponse.
    event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def event_sink(event_name: str, payload: dict[str, Any]) -> None:
        """Enqueues a tool progress event emitted from inside the agent's tool loop."""
        await event_queue.put((_KIND_PROGRESS, (event_name, payload)))

    try:
        agent, token_generator = await dispatcher.dispatch_stream(request, event_sink=event_sink)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Failed to initialize stream dispatcher: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize stream: {exc}",
        ) from exc

    async def _pump_tokens() -> None:
        """Drains the agent's token generator onto the shared queue."""
        try:
            async for token in token_generator:
                await event_queue.put((_KIND_TOKEN, token))
            await event_queue.put((_KIND_END, None))
        except Exception as exc:
            # Re-raised by the single consumer so the existing error handling applies.
            await event_queue.put((_KIND_ERROR, exc))

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        token_count = 0
        pump_task = asyncio.create_task(_pump_tokens())

        try:
            while True:
                kind, payload = await event_queue.get()

                if kind == _KIND_TOKEN:
                    token_count += 1
                    chunk = StreamTokenChunk(
                        token=payload,
                        agent_id=agent.agent_id,
                        session_id=request.session_id,
                        done=False,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    continue

                if kind == _KIND_PROGRESS:
                    event_name, event_payload = payload
                    body = {
                        **event_payload,
                        "agent_id": agent.agent_id,
                        "session_id": request.session_id,
                    }
                    yield f"event: {event_name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
                    continue

                if kind == _KIND_ERROR:
                    raise payload

                break  # _KIND_END

            # Emit final completion event
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            done_chunk = StreamTokenChunk(
                token="",
                agent_id=agent.agent_id,
                session_id=request.session_id,
                done=True,
                metadata={
                    "total_tokens_yielded": token_count,
                    "latency_ms": elapsed_ms,
                    "agent_name": agent.name,
                },
            )
            yield f"data: {done_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except LLMRateLimitError as exc:
            err_data = json.dumps({"error": exc.message, "status_code": 429, "done": True})
            yield f"event: error\ndata: {err_data}\n\n"
        except LLMTimeoutError as exc:
            err_data = json.dumps({"error": exc.message, "status_code": 504, "done": True})
            yield f"event: error\ndata: {err_data}\n\n"
        except Exception as exc:
            logger.error("Error during active SSE stream: %s", exc)
            err_data = json.dumps({"error": str(exc), "status_code": 500, "done": True})
            yield f"event: error\ndata: {err_data}\n\n"
        finally:
            # A client disconnect leaves the pump mid-flight; never leak the task.
            if not pump_task.done():
                pump_task.cancel()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(sse_event_stream(), headers=headers, media_type="text/event-stream")


@router.get(
    "/agents",
    response_model=AgentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Registered Agents",
    description="Returns metadata and capabilities of all specialized agents registered in the Gateway.",
)
async def list_registered_agents() -> AgentListResponse:
    """Retrieves list of active agents."""
    dispatcher = get_agent_dispatcher()
    agents = dispatcher.list_agents()
    return AgentListResponse(agents=agents)


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Session History",
    description="Fetches recent conversational message turns for a given session ID.",
)
async def get_session_history(
    session_id: str = Path(..., description="Target session identifier"),
) -> SessionHistoryResponse:
    """Retrieves conversation history for a given session."""
    memory_service = get_memory_service()
    raw_history = await memory_service.get_history(session_id=session_id, limit=20)
    messages = [
        MessageHistoryItem(
            role=item.get("role", "user"),
            content=item.get("content", ""),
            timestamp=item.get("timestamp", ""),
        )
        for item in raw_history
    ]
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
    )


@router.delete(
    "/history/{session_id}",
    response_model=ClearHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear Session History",
    description="Removes all stored conversation history for the specified session ID.",
)
async def clear_session_history(
    session_id: str = Path(..., description="Target session identifier to clear"),
) -> ClearHistoryResponse:
    """Deletes conversation memory for a given session."""
    memory_service = get_memory_service()
    await memory_service.clear_history(session_id)
    return ClearHistoryResponse(
        status="ok",
        session_id=session_id,
        message=f"Conversation memory for session '{session_id}' has been cleared.",
    )
