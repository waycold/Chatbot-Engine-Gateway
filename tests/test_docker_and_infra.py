"""Tests for Dockerfile, Render Blueprint, Entrypoint, and Keep-Alive Infrastructure."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
import pytest
import yaml

from scripts.keep_alive import (
    build_parser,
    execute_keep_alive,
    ping_endpoint,
    PingResult,
)

BASE_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1. Dockerfile & .dockerignore Tests
# ==============================================================================
class TestDockerInfrastructure:
    """Validates Dockerfile multi-stage structure, security, and .dockerignore rules."""

    @pytest.fixture(scope="class")
    def dockerfile_content(self) -> str:
        """Reads Dockerfile content."""
        dockerfile_path = BASE_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist at repository root"
        return dockerfile_path.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def dockerignore_content(self) -> str:
        """Reads .dockerignore content."""
        dockerignore_path = BASE_DIR / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore must exist at repository root"
        return dockerignore_path.read_text(encoding="utf-8")

    def test_dockerfile_multi_stage_build(self, dockerfile_content: str) -> None:
        """Ensures Dockerfile employs multi-stage builds for minimal image size."""
        assert "AS builder" in dockerfile_content, "Dockerfile must define a builder stage"
        assert "AS runner" in dockerfile_content, "Dockerfile must define a runner/production stage"
        assert "python:3.11-slim" in dockerfile_content, "Must use Python 3.11-slim base image"

    def test_dockerfile_security_non_root_user(self, dockerfile_content: str) -> None:
        """Ensures container runs under a non-privileged appuser account."""
        assert "useradd" in dockerfile_content or "adduser" in dockerfile_content, (
            "Must create a non-root user"
        )
        assert "USER appuser" in dockerfile_content, "Must switch to non-root user"

    def test_dockerfile_environment_and_port(self, dockerfile_content: str) -> None:
        """Ensures environment variables and dynamic port handling are properly configured."""
        assert "PYTHONUNBUFFERED=1" in dockerfile_content, "PYTHONUNBUFFERED must be enabled"
        assert "PORT" in dockerfile_content, "Must define or support PORT environment variable"
        assert "EXPOSE 8000" in dockerfile_content, "Must expose default port 8000"

    def test_dockerfile_healthcheck_defined(self, dockerfile_content: str) -> None:
        """Ensures Dockerfile defines a container-level healthcheck."""
        assert "HEALTHCHECK" in dockerfile_content, "Must define container HEALTHCHECK instruction"
        assert "/health" in dockerfile_content, "Healthcheck must target /health endpoint"

    def test_dockerignore_excludes_critical_paths(self, dockerignore_content: str) -> None:
        """Ensures .dockerignore prevents sensitive or unnecessary files from entering image."""
        required_patterns = [
            ".git",
            "venv",
            "__pycache__",
            ".env",
            "tests",
            ".pytest_cache",
            "docs",
        ]
        for pattern in required_patterns:
            assert pattern in dockerignore_content, f".dockerignore must exclude '{pattern}'"


# ==============================================================================
# 2. Render Blueprint (render.yaml) Tests
# ==============================================================================
class TestRenderBlueprint:
    """Validates the render.yaml Infrastructure-as-Code manifest."""

    @pytest.fixture(scope="class")
    def render_config(self) -> dict[str, Any]:
        """Parses and validates render.yaml structure."""
        render_yaml_path = BASE_DIR / "render.yaml"
        assert render_yaml_path.exists(), "render.yaml must exist at repository root"
        content = render_yaml_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), "render.yaml must parse as a valid YAML mapping"
        return parsed

    def test_render_services_structure(self, render_config: dict[str, Any]) -> None:
        """Validates the presence of services and web service definition."""
        assert "services" in render_config, "render.yaml must declare a 'services' list"
        services = render_config["services"]
        assert isinstance(services, list) and len(services) >= 1, "Must declare at least one service"

        web_service = next((s for s in services if s.get("name") == "ai-agent-gateway"), None)
        assert web_service is not None, "Must declare service with name 'ai-agent-gateway'"
        assert web_service.get("type") == "web", "Service type must be 'web'"
        assert web_service.get("runtime") == "docker", "Service runtime must be 'docker'"
        assert web_service.get("healthCheckPath") == "/health", "healthCheckPath must be '/health'"

    def test_render_required_env_vars(self, render_config: dict[str, Any]) -> None:
        """Validates all required environment variables are mapped in render.yaml."""
        web_service = next(s for s in render_config["services"] if s.get("name") == "ai-agent-gateway")
        env_vars = web_service.get("envVars", [])
        env_keys = {item.get("key") for item in env_vars}

        expected_keys = {
            "GEMINI_API_KEY",
            "INTERNAL_API_SECRET",
            "DJANGO_BACKEND_URL",
            "REDIS_URL",
            "ENVIRONMENT",
            "DEBUG",
            "BACKEND_CORS_ORIGINS",
            "PYTHONUNBUFFERED",
        }
        missing_keys = expected_keys - env_keys
        assert not missing_keys, f"render.yaml missing environment variables: {missing_keys}"


# ==============================================================================
# 3. Entrypoint Script Tests
# ==============================================================================
class TestRenderEntrypointScript:
    """Validates the scripts/render_entrypoint.sh startup script."""

    @pytest.fixture(scope="class")
    def entrypoint_content(self) -> str:
        """Reads scripts/render_entrypoint.sh content."""
        script_path = BASE_DIR / "scripts" / "render_entrypoint.sh"
        assert script_path.exists(), "scripts/render_entrypoint.sh must exist"
        return script_path.read_text(encoding="utf-8")

    def test_entrypoint_shebang_and_safety(self, entrypoint_content: str) -> None:
        """Ensures proper shell shebang and fail-fast set -e flag."""
        assert entrypoint_content.startswith("#!/bin/sh") or entrypoint_content.startswith("#!/usr/bin/env bash")
        assert "set -e" in entrypoint_content, "Entrypoint must contain 'set -e' for strict error handling"

    def test_entrypoint_port_fallback(self, entrypoint_content: str) -> None:
        """Ensures default port fallback handling."""
        assert "PORT=" in entrypoint_content
        assert "8000" in entrypoint_content

    def test_entrypoint_uvicorn_execution_and_sse_opts(self, entrypoint_content: str) -> None:
        """Ensures exec uvicorn with SSE keep-alive timeout is present."""
        assert "exec uvicorn app.main:app" in entrypoint_content, "Must use 'exec uvicorn' for clean signal forwarding"
        assert "--timeout-keep-alive" in entrypoint_content, "Must configure --timeout-keep-alive for SSE stability"
        assert "--proxy-headers" in entrypoint_content, "Must configure --proxy-headers for Render reverse proxy"


# ==============================================================================
# 4. Keep-Alive Script Tests (scripts/keep_alive.py)
# ==============================================================================
class TestKeepAliveScript:
    """Unit tests for the keep_alive ping mechanism."""

    def test_build_parser_defaults(self) -> None:
        """Tests that CLI parser defines expected flags and defaults."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.endpoint == "/health"
        assert args.timeout == 10.0
        assert args.retries == 3
        assert args.backoff == 2.0
        assert not args.verbose

    def test_ping_endpoint_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests successful 200 OK ping."""
        mock_response_body = b'{"status": "ok", "app_name": "AI Agent Gateway"}'

        class MockHTTPResponse:
            def __init__(self) -> None:
                self.code = 200

            def getcode(self) -> int:
                return self.code

            def read(self) -> bytes:
                return mock_response_body

            def __enter__(self) -> MockHTTPResponse:
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: MockHTTPResponse())

        result: PingResult = ping_endpoint("http://localhost:8000/health", timeout=5.0)
        assert result.success is True
        assert result.status_code == 200
        assert result.latency_ms >= 0.0
        assert isinstance(result.response_data, dict)
        assert result.response_data.get("status") == "ok"
        assert result.error_message is None

    def test_ping_endpoint_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests handling of HTTP 503 error."""
        def mock_urlopen(req: Any, timeout: float) -> Any:
            fp = io.BytesIO(b"Service Unavailable")
            raise urllib.error.HTTPError(
                url="http://test.local/health",
                code=503,
                msg="Service Unavailable",
                hdrs={},  # type: ignore[arg-type]
                fp=fp,
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        result: PingResult = ping_endpoint("http://test.local/health", timeout=5.0)
        assert result.success is False
        assert result.status_code == 503
        assert "HTTP Error 503" in (result.error_message or "")

    def test_ping_endpoint_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests handling of network unreachable URLError."""
        def mock_urlopen(req: Any, timeout: float) -> Any:
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        result: PingResult = ping_endpoint("http://unreachable.local/health", timeout=2.0)
        assert result.success is False
        assert result.status_code == 0
        assert "Network/Connection Error" in (result.error_message or "")

    def test_execute_keep_alive_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests execute_keep_alive returns True when ping succeeds."""
        monkeypatch.setattr(
            "scripts.keep_alive.ping_endpoint",
            lambda url, timeout: PingResult(
                success=True,
                status_code=200,
                latency_ms=12.5,
                response_data={"status": "ok"},
            ),
        )

        success = execute_keep_alive("https://ai-agent-gateway.onrender.com", endpoint="/health")
        assert success is True

    def test_execute_keep_alive_retry_and_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests execute_keep_alive retries on transient error and succeeds."""
        call_count = 0

        def mock_ping(url: str, timeout: float) -> PingResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PingResult(
                    success=False,
                    status_code=502,
                    latency_ms=50.0,
                    response_data={},
                    error_message="Bad Gateway",
                )
            return PingResult(
                success=True,
                status_code=200,
                latency_ms=15.0,
                response_data={"ping": "pong"},
            )

        monkeypatch.setattr("scripts.keep_alive.ping_endpoint", mock_ping)
        monkeypatch.setattr("time.sleep", lambda s: None)

        success = execute_keep_alive(
            "https://ai-agent-gateway.onrender.com",
            endpoint="/ping",
            max_retries=3,
            backoff_factor=1.0,
        )
        assert success is True
        assert call_count == 2

    def test_execute_keep_alive_all_retries_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests execute_keep_alive returns False when all retries fail."""
        monkeypatch.setattr(
            "scripts.keep_alive.ping_endpoint",
            lambda url, timeout: PingResult(
                success=False,
                status_code=500,
                latency_ms=10.0,
                response_data={},
                error_message="Internal Server Error",
            ),
        )
        monkeypatch.setattr("time.sleep", lambda s: None)

        success = execute_keep_alive(
            "https://ai-agent-gateway.onrender.com",
            endpoint="/health",
            max_retries=2,
            backoff_factor=1.0,
        )
        assert success is False


# ==============================================================================
# 5. GitHub Actions Keep-Alive Workflow Tests
# ==============================================================================
class TestKeepAliveWorkflow:
    """Validates the .github/workflows/keep_alive.yml GitHub Action."""

    @pytest.fixture(scope="class")
    def workflow_config(self) -> dict[str, Any]:
        """Parses and validates keep_alive.yml workflow."""
        workflow_path = BASE_DIR / ".github" / "workflows" / "keep_alive.yml"
        assert workflow_path.exists(), ".github/workflows/keep_alive.yml must exist"
        content = workflow_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), "keep_alive.yml must parse as a valid YAML mapping"
        return parsed

    def test_workflow_triggers(self, workflow_config: dict[str, Any]) -> None:
        """Ensures workflow has cron schedule and manual workflow_dispatch triggers."""
        triggers = workflow_config.get("on") or workflow_config.get(True)
        assert triggers is not None, "Workflow must define trigger events"

        # Check schedule
        schedule = triggers.get("schedule", [])
        assert isinstance(schedule, list) and len(schedule) >= 1, "Must define cron schedule"
        cron_expr = schedule[0].get("cron", "")
        assert "*/10" in cron_expr or "*/12" in cron_expr or "*/14" in cron_expr, (
            "Cron expression must ping frequently enough to avoid 15m idle shutdown"
        )

        # Check workflow_dispatch
        assert "workflow_dispatch" in triggers, "Workflow must allow manual dispatch"

    def test_workflow_executes_keep_alive_script(self, workflow_config: dict[str, Any]) -> None:
        """Ensures workflow jobs run scripts/keep_alive.py."""
        jobs = workflow_config.get("jobs", {})
        assert "keep-alive-ping" in jobs, "Must define 'keep-alive-ping' job"

        job = jobs["keep-alive-ping"]
        steps = job.get("steps", [])
        run_steps = [s.get("run", "") for s in steps if "run" in s]

        assert any("scripts/keep_alive.py" in step for step in run_steps), (
            "Workflow must execute scripts/keep_alive.py in its steps"
        )
