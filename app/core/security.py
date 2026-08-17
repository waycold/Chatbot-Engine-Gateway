"""Security and authentication utilities for internal and external communication."""
from typing import Annotated
from fastapi import Header, HTTPException, status
from app.core.config import settings


async def verify_internal_api_secret(
    x_internal_secret: Annotated[str | None, Header(alias="X-Internal-Secret")] = None,
) -> bool:
    """Verifies that the request carries a valid internal service secret header.

    Used to secure internal communication between the Django Monolith and this Gateway.
    """
    if not x_internal_secret or x_internal_secret != settings.INTERNAL_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal service secret.",
            headers={"WWW-Authenticate": "X-Internal-Secret"},
        )
    return True
