#!/usr/bin/env python3
"""Keep-Alive Ping Utility for Chatbot-Engine-Gateway.

This standalone, zero-dependency script sends periodic HTTP GET requests to the
gateway's lightweight `/health` or `/ping` endpoint to prevent free/eco tier
instances (e.g., Render Free Tier) from spinning down due to inactivity.

Features:
- Standard library only (`urllib.request`), zero external pip dependencies.
- Precise latency measurement (RTT in milliseconds).
- Configurable retries with exponential backoff.
- Clean structured logging and exit codes for CI/CD / Cron integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, NamedTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("keep_alive")


class PingResult(NamedTuple):
    """Encapsulates the outcome of a keep-alive ping attempt."""

    success: bool
    status_code: int
    latency_ms: float
    response_data: Dict[str, Any] | str
    error_message: str | None = None


def ping_endpoint(
    url: str,
    timeout: float = 10.0,
    user_agent: str = "Chatbot-Engine-Gateway-KeepAlive/1.0",
) -> PingResult:
    """Sends a single HTTP GET request to the target URL and measures latency.

    Args:
        url: Full URL to the health check or ping endpoint.
        timeout: Maximum request timeout in seconds.
        user_agent: Custom User-Agent header string.

    Returns:
        PingResult containing success status, HTTP code, latency (ms), and payload.
    """
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
        },
        method="GET",
    )

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = response.getcode()
            body_bytes = response.read()

            try:
                body_str = body_bytes.decode("utf-8")
                payload = json.loads(body_str)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = body_str

            return PingResult(
                success=(status_code == 200),
                status_code=status_code,
                latency_ms=round(latency_ms, 2),
                response_data=payload,
                error_message=None,
            )

    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return PingResult(
            success=False,
            status_code=e.code,
            latency_ms=round(latency_ms, 2),
            response_data={},
            error_message=f"HTTP Error {e.code}: {e.reason}",
        )
    except urllib.error.URLError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return PingResult(
            success=False,
            status_code=0,
            latency_ms=round(latency_ms, 2),
            response_data={},
            error_message=f"Network/Connection Error: {e.reason}",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return PingResult(
            success=False,
            status_code=0,
            latency_ms=round(latency_ms, 2),
            response_data={},
            error_message=f"Unexpected Error: {str(e)}",
        )


def execute_keep_alive(
    base_url: str,
    endpoint: str = "/health",
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> bool:
    """Executes a keep-alive routine with exponential backoff retries.

    Args:
        base_url: Base domain or service URL.
        endpoint: Endpoint path to ping (e.g., '/health' or '/ping').
        timeout: Request timeout in seconds.
        max_retries: Total number of attempts before marking as failed.
        backoff_factor: Exponential backoff delay multiplier.

    Returns:
        bool: True if ping was successful, False otherwise.
    """
    clean_base = base_url.rstrip("/")
    clean_endpoint = "/" + endpoint.lstrip("/")
    target_url = f"{clean_base}{clean_endpoint}"

    logger.info("Initiating Keep-Alive ping to: %s", target_url)

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt [%d/%d] sending request...", attempt, max_retries)
        result = ping_endpoint(target_url, timeout=timeout)

        if result.success:
            logger.info(
                "✅ [SUCCESS] Status: %d | Latency: %.2f ms | Response: %s",
                result.status_code,
                result.latency_ms,
                json.dumps(result.response_data) if isinstance(result.response_data, dict) else result.response_data,
            )
            return True

        logger.warning(
            "⚠️ [ATTEMPT FAILED] Attempt %d/%d failed: %s (Status: %d, Latency: %.2f ms)",
            attempt,
            max_retries,
            result.error_message,
            result.status_code,
            result.latency_ms,
        )

        if attempt < max_retries:
            delay = backoff_factor ** (attempt - 1)
            logger.info("Waiting %.1f seconds before retry...", delay)
            time.sleep(delay)

    logger.error("❌ [FAILURE] All %d keep-alive ping attempts failed for %s", max_retries, target_url)
    return False


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Lightweight Keep-Alive ping utility for Chatbot-Engine-Gateway on Render.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        "-u",
        default=os.getenv("GATEWAY_URL", os.getenv("TARGET_URL", "http://localhost:8000")),
        help="Target base URL of the gateway service (can also be set via GATEWAY_URL env var)",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default="/health",
        help="Target health/ping endpoint path (/health or /ping)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=3,
        help="Maximum retry attempts on failure",
    )
    parser.add_argument(
        "--backoff",
        "-b",
        type=float,
        default=2.0,
        help="Exponential backoff factor between retries",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    success = execute_keep_alive(
        base_url=args.url,
        endpoint=args.endpoint,
        timeout=args.timeout,
        max_retries=args.retries,
        backoff_factor=args.backoff,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
