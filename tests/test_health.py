"""Unit and Integration tests for Health & Root metadata endpoints."""
import pytest
from fastapi import status
from fastapi.testclient import TestClient
import httpx
from app.schemas.payload import HealthResponse


class TestHealthEndpoint:
    """Test suite for the service healthcheck endpoint (/health)."""

    def test_health_check_sync(self, sync_client: TestClient) -> None:
        """Verifies that synchronous GET /health returns 200 OK and valid HealthResponse schema."""
        response = sync_client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "ok"
        assert "app_name" in data
        assert "environment" in data
        assert "version" in data

        # Validate strictly against Pydantic schema
        health_obj = HealthResponse.model_validate(data)
        assert health_obj.status == "ok"
        assert health_obj.version is not None

    @pytest.mark.asyncio
    async def test_health_check_async(self, async_client: httpx.AsyncClient) -> None:
        """Verifies that asynchronous GET /health via ASGITransport returns 200 OK."""
        response = await async_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_is_lightweight(self, sync_client: TestClient) -> None:
        """Verifies that /health is ultralightweight and does not require third-party dependencies."""
        # /health should respond immediately with JSON content type
        response = sync_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers.get("content-type", "")


class TestRootEndpoint:
    """Test suite for Gateway Root metadata endpoint (/)."""

    def test_root_endpoint_metadata(self, sync_client: TestClient) -> None:
        """Verifies that GET / returns gateway metadata, documentation link, and version."""
        response = sync_client.get("/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data.get("docs_url") == "/docs"
        assert data.get("health_url") == "/health"

    @pytest.mark.asyncio
    async def test_root_endpoint_async(self, async_client: httpx.AsyncClient) -> None:
        """Verifies asynchronous GET / response."""
        response = await async_client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data.get("docs_url") == "/docs"


class TestDocsAndOpenAPI:
    """Test suite for Swagger documentation and OpenAPI specification generation."""

    def test_docs_swagger_ui_available(self, sync_client: TestClient) -> None:
        """Verifies that Swagger UI HTML is served at /docs."""
        response = sync_client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        assert "swagger-ui" in response.text.lower() or "html" in response.headers.get("content-type", "")

    def test_openapi_json_schema(self, sync_client: TestClient) -> None:
        """Verifies that the OpenAPI JSON schema is generated correctly."""
        response = sync_client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/" in schema["paths"]


class TestCORSAndNotFound:
    """Test suite for CORS headers and error handling on undefined routes."""

    def test_cors_headers_on_allowed_origin(self, sync_client: TestClient) -> None:
        """Verifies that configured CORS origins receive Access-Control-Allow-Origin header."""
        allowed_origin = "http://localhost:3000"
        response = sync_client.get("/health", headers={"Origin": allowed_origin})
        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("access-control-allow-origin") == allowed_origin

    def test_cors_preflight_options(self, sync_client: TestClient) -> None:
        """Verifies that CORS preflight OPTIONS requests are handled with 200 OK."""
        response = sync_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_not_found_endpoint(self, sync_client: TestClient) -> None:
        """Verifies that non-existent routes return HTTP 404 Not Found."""
        response = sync_client.get("/non-existent-path")
        assert response.status_code == status.HTTP_404_NOT_FOUND
