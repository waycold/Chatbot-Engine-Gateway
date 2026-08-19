"""Redis memory management service for agent sessions and conversational context."""
from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger("ai_gateway.memory")


class InMemoryFallbackStore:
    """In-memory session history store used as fallback when Redis is unavailable."""

    def __init__(self, max_items_per_session: int = 50) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._max_items = max_items_per_session

    def get_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieves the most recent messages for a session."""
        history = self._store.get(session_id, [])
        return history[-limit:] if limit > 0 else history

    def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Appends a message to the in-memory session history."""
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(message)
        if len(self._store[session_id]) > self._max_items:
            self._store[session_id] = self._store[session_id][-self._max_items:]

    def clear(self, session_id: str) -> None:
        """Clears memory for a specific session."""
        self._store.pop(session_id, None)


class RedisMemoryService:
    """Service for storing and retrieving agent conversation history and session states.

    Uses Redis lists with JSON serialization and automatic key expiration (TTL).
    Features automatic in-memory fallback for local development or connection failures.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = settings.SESSION_TTL_SECONDS
        self._redis_client: Optional[Any] = None
        self._fallback_store = InMemoryFallbackStore()

    @property
    def is_connected(self) -> bool:
        """Returns True if a Redis client is attached."""
        return self._redis_client is not None

    async def init_pool(self) -> None:
        """Initializes the Redis connection pool and verifies connectivity."""
        try:
            self._redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            await self._redis_client.ping()
            logger.info("Connected to Redis session memory at %s", self.redis_url)
        except Exception as exc:
            self._redis_client = None
            logger.warning(
                "Could not connect to Redis at %s (%s). Falling back to in-memory store.",
                self.redis_url,
                exc,
            )

    async def close(self) -> None:
        """Closes the active Redis connection pool gracefully."""
        if self._redis_client is not None:
            try:
                if hasattr(self._redis_client, "aclose"):
                    await self._redis_client.aclose()
                logger.info("Redis memory connection closed.")
            except Exception as exc:
                logger.error("Error closing Redis connection: %s", exc)
            finally:
                self._redis_client = None

    def _get_key(self, session_id: str) -> str:
        """Generates namespaced Redis key for session history."""
        return f"gateway:session:{session_id}:messages"

    async def get_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieves conversational history for the specified session.

        Args:
            session_id: Unique session identifier.
            limit: Maximum number of recent messages to return.

        Returns:
            List of message dictionaries with 'role', 'content', and 'timestamp'.
        """
        if self._redis_client is None:
            return self._fallback_store.get_history(session_id, limit=limit)

        key = self._get_key(session_id)
        try:
            raw_messages = await self._redis_client.lrange(key, -limit, -1)
            messages: list[dict[str, Any]] = []
            for item in raw_messages:
                try:
                    if isinstance(item, str):
                        messages.append(json.loads(item))
                    elif isinstance(item, dict):
                        messages.append(item)
                except (json.JSONDecodeError, TypeError):
                    messages.append({"role": "user", "content": str(item), "timestamp": ""})
            return messages
        except Exception as exc:
            logger.warning("Redis lrange failed for key %s: %s. Using fallback store.", key, exc)
            return self._fallback_store.get_history(session_id, limit=limit)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        ttl: Optional[int] = None,
    ) -> None:
        """Appends a new turn message to the session's conversation history.

        Args:
            session_id: Session identifier.
            role: Message role ('user', 'model', or 'system').
            content: Text content of the message.
            ttl: Key expiration time in seconds (defaults to settings.SESSION_TTL_SECONDS).
        """
        item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Always update fallback store for consistency
        self._fallback_store.add_message(session_id, item)

        if self._redis_client is None:
            return

        key = self._get_key(session_id)
        key_ttl = ttl or self.default_ttl

        try:
            if hasattr(self._redis_client, "pipeline"):
                pipe = self._redis_client.pipeline()
                pipe.rpush(key, json.dumps(item, ensure_ascii=False))
                pipe.expire(key, key_ttl)
                await pipe.execute()
            else:
                await self._redis_client.rpush(key, json.dumps(item, ensure_ascii=False))
                if hasattr(self._redis_client, "expire"):
                    await self._redis_client.expire(key, key_ttl)
        except Exception as exc:
            logger.warning("Failed to persist message to Redis key %s: %s", key, exc)

    async def clear_history(self, session_id: str) -> None:
        """Deletes conversational history for the given session.

        Args:
            session_id: Unique session identifier.
        """
        self._fallback_store.clear(session_id)

        if self._redis_client is None:
            return

        key = self._get_key(session_id)
        try:
            await self._redis_client.delete(key)
        except Exception as exc:
            logger.warning("Failed to delete Redis key %s: %s", key, exc)

    async def health_check(self) -> bool:
        """Verifies operational status of the Redis connection."""
        if self._redis_client is None:
            return False
        try:
            pong = await self._redis_client.ping()
            return bool(pong)
        except Exception:
            return False


# Global singleton instance
_memory_service: Optional[RedisMemoryService] = None


def get_memory_service() -> RedisMemoryService:
    """Returns the singleton instance of RedisMemoryService."""
    global _memory_service
    if _memory_service is None:
        _memory_service = RedisMemoryService()
    return _memory_service
