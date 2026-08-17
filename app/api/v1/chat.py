"""API v1 Chat router definition."""
from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["Chat & Agents"])

# Chat endpoints (SSE streaming & standard POST) will be registered here in subsequent phases.
