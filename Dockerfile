# ==============================================================================
# Stage 1: Build & Dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Configure environment for clean dependency installation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy dependency definition
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# ==============================================================================
# Stage 2: Minimal Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runner

LABEL maintainer="Chatbot Engine Team" \
      service="ai-agent-gateway" \
      runtime="fastapi-python3.11"

WORKDIR /app

# Configure runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    ENVIRONMENT=production

# Install minimal runtime system utilities (curl for Docker healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Security: Create non-root group and user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code, business knowledge base, and deployment scripts
COPY --chown=appuser:appgroup app /app/app
COPY --chown=appuser:appgroup data /app/data
COPY --chown=appuser:appgroup scripts /app/scripts

# Grant execution permissions to entrypoint scripts
RUN chmod +x /app/scripts/*.sh || true

# Switch to non-root execution context
USER appuser

# Expose default port (Render overrides this dynamically via $PORT)
EXPOSE 8000

# Docker-level healthcheck against lightweight /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Execute startup entrypoint with graceful signal propagation and SSE streaming optimizations
CMD ["/bin/sh", "/app/scripts/render_entrypoint.sh"]
