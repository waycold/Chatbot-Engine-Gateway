"""Redis memory management service for agent sessions and conversational context."""
from typing import Optional
from app.core.config import settings


class RedisMemoryService:
    """Service for storing and retrieving agent conversation history and session states in Redis."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis_client = None
