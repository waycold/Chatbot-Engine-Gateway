"""HTTP client service for internal communication with the Django monolith."""
from typing import Optional
import httpx
from app.core.config import settings


class DjangoAPIService:
    """Async HTTP client for interacting with the core transactional Django backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        internal_secret: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or settings.DJANGO_BACKEND_URL).rstrip("/")
        self.internal_secret = internal_secret or settings.INTERNAL_API_SECRET
        self._headers = {
            "X-Internal-Secret": self.internal_secret,
            "Content-Type": "application/json",
        }

    async def get_client(self) -> httpx.AsyncClient:
        """Returns an async HTTP client configured with base URL and internal headers."""
        return httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=10.0)
