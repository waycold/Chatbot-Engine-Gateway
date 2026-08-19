#!/bin/sh
# ==============================================================================
# Render Production Entrypoint Script for FastAPI AI Gateway
# ==============================================================================
set -e

# Default to port 8000 if PORT is not supplied by Render or Docker environment
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TIMEOUT_KEEP_ALIVE="${TIMEOUT_KEEP_ALIVE:-65}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "============================================================"
echo " Starting Chatbot-Engine-Gateway on ${HOST}:${PORT}"
echo " Environment: ${ENVIRONMENT:-production}"
echo " Keep-Alive Timeout: ${TIMEOUT_KEEP_ALIVE}s (Optimized for SSE)"
echo "============================================================"

# Pre-flight check: Warn if critical secrets are missing in production
if [ -z "$GEMINI_API_KEY" ]; then
    echo "[WARNING] GEMINI_API_KEY is not set. LLM inference calls will fail until configured." >&2
fi

if [ -z "$INTERNAL_API_SECRET" ]; then
    echo "[WARNING] INTERNAL_API_SECRET is not set. Monolith communication may be rejected." >&2
fi

# Execute Uvicorn via exec to preserve POSIX signal propagation (SIGTERM / SIGINT)
# for graceful FastAPI lifespan shutdown.
exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE" \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --log-level "$LOG_LEVEL"
