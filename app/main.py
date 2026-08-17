"""Main FastAPI application entrypoint for AI Agent Gateway."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.core.config import settings
from app.schemas.payload import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown events."""
    # Startup: Initialize service pools, connections or caches if needed
    yield
    # Shutdown: Cleanly close open connections, HTTP clients and Redis pools


def create_application() -> FastAPI:
    """Factory function to configure and instantiate the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="AI Agent Gateway microservice for Multi-Agent routing and LLM streaming.",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register API Routers
    application.include_router(chat_router, prefix=settings.API_V1_STR)

    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["Health & Monitoring"],
        summary="Service Health Check",
        description="Returns the current operational status and environment metadata of the microservice.",
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.PROJECT_NAME,
            environment=settings.ENVIRONMENT,
            version=settings.VERSION,
        )

    @application.get(
        "/",
        status_code=status.HTTP_200_OK,
        tags=["Root"],
        summary="Gateway Root Information",
    )
    async def root() -> dict[str, str]:
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": "/health",
        }

    return application


app: FastAPI = create_application()
