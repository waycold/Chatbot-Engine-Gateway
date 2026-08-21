"""Main FastAPI application entrypoint for AI Agent Gateway."""
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.agents.dispatcher import get_agent_dispatcher
from app.api.v1.chat import router as chat_router
from app.core.config import settings
from app.schemas.payload import HealthResponse
from app.services.django_api import get_django_api_service
from app.services.llm_client import get_llm_service
from app.services.memory import get_memory_service

# Configure root application logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("ai_gateway.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Initializing %s v%s in %s mode...", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)

    # Startup 1: Connect to Redis Session Memory
    memory_service = get_memory_service()
    await memory_service.init_pool()

    # Startup 2: Start Django HTTP Client
    django_service = get_django_api_service()
    await django_service.start()

    # Startup 3: Initialize LLM Service and Multi-Agent Dispatcher
    get_llm_service()
    get_agent_dispatcher()
    logger.info("Multi-Agent Gateway startup sequence completed successfully.")

    yield

    # Shutdown sequence
    logger.info("Initiating graceful shutdown...")
    await memory_service.close()
    await django_service.close()
    logger.info("All Gateway connections closed cleanly.")


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
            max_age=86400,
        )

    # Register API Routers
    application.include_router(chat_router, prefix=settings.API_V1_STR)

    # --- Keep-Alive & Health Check Endpoints ---
    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["Health & Monitoring"],
        summary="Ultralight Service Health Check",
        description="Returns immediate 200 OK without blocking external dependencies (ideal for Render keep-alive pings).",
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.PROJECT_NAME,
            environment=settings.ENVIRONMENT,
            version=settings.VERSION,
        )

    @application.get(
        "/ping",
        status_code=status.HTTP_200_OK,
        tags=["Health & Monitoring"],
        summary="Keep-Alive Ping Endpoint",
        description="Ultra-fast ping endpoint returning 'pong' to prevent server idle spin-down.",
    )
    async def ping() -> dict[str, str]:
        return {"ping": "pong"}

    @application.get(
        "/health/details",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["Health & Monitoring"],
        summary="Detailed Health Check",
        description="Verifies operational status of downstream dependencies (Redis, Django backend).",
    )
    async def health_details() -> HealthResponse:
        memory_service = get_memory_service()
        django_service = get_django_api_service()

        redis_ok = await memory_service.health_check()
        django_ok = await django_service.health_check()

        overall_status = "ok" if (redis_ok or django_ok) else "degraded"

        return HealthResponse(
            status=overall_status,
            app_name=settings.PROJECT_NAME,
            environment=settings.ENVIRONMENT,
            version=settings.VERSION,
            redis_healthy=redis_ok,
            django_healthy=django_ok,
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
            "chat_stream_url": f"{settings.API_V1_STR}/chat/stream",
            "chat_url": f"{settings.API_V1_STR}/chat",
        }

    return application


app: FastAPI = create_application()
