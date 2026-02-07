# -*- coding: utf-8 -*-
"""
Auto-Instrumentation - Library instrumentors for GreenLang (OBS-003)

Enables automatic span creation for I/O libraries used across GreenLang:
FastAPI, httpx, psycopg / psycopg2, redis-py, Celery, and requests.

Each instrumentor is wrapped in a defensive try/except so the service
starts normally even when an instrumentor package is not installed.  Hooks
inject GreenLang-specific span attributes (tenant ID, peer service, etc.).

This module is called once during ``configure_tracing()`` and never needs
to be invoked directly by application code.

Example:
    >>> from greenlang.infrastructure.tracing_service.instrumentors import (
    ...     setup_instrumentors,
    ... )
    >>> setup_instrumentors(TracingConfig())

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from greenlang.infrastructure.tracing_service.config import TracingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def setup_instrumentors(config: TracingConfig) -> Dict[str, bool]:
    """Enable auto-instrumentation for all configured libraries.

    Each instrumentor that succeeds is logged at INFO.  Missing or failed
    instrumentors are logged at DEBUG or WARNING respectively.

    Args:
        config: Tracing configuration with ``instrument_*`` flags.

    Returns:
        A dict mapping library names to ``True`` (success) or ``False``.
    """
    results: Dict[str, bool] = {}

    if config.instrument_fastapi:
        results["fastapi"] = _instrument_fastapi(config)

    if config.instrument_httpx:
        results["httpx"] = _instrument_httpx()

    if config.instrument_psycopg:
        results["psycopg"] = _instrument_psycopg()

    if config.instrument_redis:
        results["redis"] = _instrument_redis()

    if config.instrument_celery:
        results["celery"] = _instrument_celery()

    if config.instrument_requests:
        results["requests"] = _instrument_requests()

    succeeded = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(
        "Auto-instrumentation complete: %d/%d libraries instrumented",
        succeeded,
        total,
    )
    return results


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

def _instrument_fastapi(config: TracingConfig) -> bool:
    """Instrument FastAPI with request/response hooks.

    Args:
        config: Tracing configuration (used for tenant_header setting).

    Returns:
        True if instrumentation succeeded.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument(
            excluded_urls="health,ready,ping,metrics,favicon.ico",
            server_request_hook=_fastapi_request_hook,
        )
        logger.info("FastAPI auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-fastapi not installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument FastAPI: %s", exc)
    return False


def _fastapi_request_hook(span: Any, scope: Dict[str, Any]) -> None:
    """Extract GreenLang-specific attributes from a FastAPI ASGI scope.

    Runs once per request inside the instrumentor's server span.

    Args:
        span: The server span created by the FastAPI instrumentor.
        scope: ASGI scope dict.
    """
    try:
        headers = dict(scope.get("headers", []))
        # Headers come as bytes in ASGI
        tenant_id = _extract_asgi_header(headers, b"x-tenant-id")
        if tenant_id:
            span.set_attribute("gl.tenant_id", tenant_id)

        request_id = _extract_asgi_header(headers, b"x-request-id")
        if request_id:
            span.set_attribute("gl.request_id", request_id)

        correlation_id = _extract_asgi_header(headers, b"x-correlation-id")
        if correlation_id:
            span.set_attribute("gl.correlation_id", correlation_id)

        # Route path for grouping
        path = scope.get("path", "")
        if path:
            span.set_attribute("http.route", path)
    except Exception:
        pass


def _extract_asgi_header(
    headers: Dict[bytes, bytes], key: bytes
) -> Optional[str]:
    """Extract a header value from ASGI-style byte-key dict.

    Args:
        headers: Dict mapping ``bytes`` header names to ``bytes`` values.
        key: The header name to look up (lowercase bytes).

    Returns:
        The decoded header value, or ``None``.
    """
    value = headers.get(key)
    if value is not None:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return None


# ---------------------------------------------------------------------------
# httpx
# ---------------------------------------------------------------------------

def _instrument_httpx() -> bool:
    """Instrument the httpx HTTP client library.

    Returns:
        True if instrumentation succeeded.
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(
            request_hook=_httpx_request_hook,
            response_hook=_httpx_response_hook,
        )
        logger.info("httpx auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-httpx not installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument httpx: %s", exc)
    return False


def _httpx_request_hook(span: Any, request: Any) -> None:
    """Enrich outgoing HTTP client spans with peer service information.

    Args:
        span: The client span.
        request: The httpx request object.
    """
    try:
        url = str(getattr(request, "url", ""))
        if url:
            parsed = urlparse(url)
            peer_service = _resolve_peer_service(parsed.hostname or "")
            if peer_service:
                span.set_attribute("peer.service", peer_service)
    except Exception:
        pass


def _httpx_response_hook(span: Any, request: Any, response: Any) -> None:
    """Record response-level details on client spans.

    Args:
        span: The client span.
        request: The httpx request object.
        response: The httpx response object.
    """
    try:
        status = getattr(response, "status_code", None)
        if status is not None:
            span.set_attribute("http.response.status_code", status)
            content_length = response.headers.get("content-length")
            if content_length:
                span.set_attribute(
                    "http.response.body.size", int(content_length)
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# psycopg / psycopg2
# ---------------------------------------------------------------------------

def _instrument_psycopg() -> bool:
    """Instrument psycopg2 (or psycopg3) for database tracing.

    Tries psycopg2 first (more common), then psycopg (v3).

    Returns:
        True if instrumentation succeeded.
    """
    # Try psycopg2 first
    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

        Psycopg2Instrumentor().instrument(
            enable_commenter=True,
            commenter_options={"db_driver": True, "opentelemetry_values": True},
        )
        logger.info("psycopg2 auto-instrumentation enabled")
        return True
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Failed to instrument psycopg2: %s", exc)

    # Try psycopg (v3)
    try:
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument(
            enable_commenter=True,
        )
        logger.info("psycopg (v3) auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "Neither psycopg2 nor psycopg instrumentors installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument psycopg: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

def _instrument_redis() -> bool:
    """Instrument redis-py for cache/session tracing.

    Returns:
        True if instrumentation succeeded.
    """
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument(
            request_hook=_redis_request_hook,
            response_hook=_redis_response_hook,
        )
        logger.info("Redis auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-redis not installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument Redis: %s", exc)
    return False


def _redis_request_hook(span: Any, instance: Any, args: Any, kwargs: Any) -> None:
    """Enrich Redis spans with command metadata.

    Args:
        span: The client span.
        instance: Redis client instance.
        args: Command arguments.
        kwargs: Command keyword arguments.
    """
    try:
        if args:
            command = str(args[0]).upper() if args[0] else "UNKNOWN"
            span.set_attribute("db.redis.command", command)
            # Detect key prefix for categorisation
            if len(args) > 1 and isinstance(args[1], (str, bytes)):
                key = args[1].decode("utf-8") if isinstance(args[1], bytes) else args[1]
                prefix = key.split(":")[0] if ":" in key else key
                span.set_attribute("db.redis.key_prefix", prefix)
    except Exception:
        pass


def _redis_response_hook(span: Any, instance: Any, response: Any) -> None:
    """Record Redis response metadata.

    Args:
        span: The client span.
        instance: Redis client instance.
        response: The Redis response.
    """
    try:
        if response is None:
            span.set_attribute("db.redis.hit", False)
        else:
            span.set_attribute("db.redis.hit", True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------

def _instrument_celery() -> bool:
    """Instrument Celery task execution.

    Returns:
        True if instrumentation succeeded.
    """
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
        logger.info("Celery auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-celery not installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument Celery: %s", exc)
    return False


# ---------------------------------------------------------------------------
# requests (urllib3-based)
# ---------------------------------------------------------------------------

def _instrument_requests() -> bool:
    """Instrument the ``requests`` HTTP library.

    Returns:
        True if instrumentation succeeded.
    """
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument()
        logger.info("requests auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-requests not installed; skipping"
        )
    except Exception as exc:
        logger.warning("Failed to instrument requests: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Peer-service resolver
# ---------------------------------------------------------------------------

# Maps hostname substrings to logical service names for peer.service tagging.
_PEER_SERVICE_MAP: Dict[str, str] = {
    "otel-collector": "otel-collector",
    "postgres": "postgresql",
    "redis": "redis",
    "kong": "api-gateway",
    "loki": "loki",
    "prometheus": "prometheus",
    "tempo": "tempo",
    "vault": "vault",
    "elasticsearch": "elasticsearch",
    "s3.amazonaws.com": "s3",
}


def _resolve_peer_service(hostname: str) -> Optional[str]:
    """Resolve a hostname to a logical peer service name.

    Args:
        hostname: The target hostname.

    Returns:
        A logical service name, or ``None`` if not recognised.
    """
    hostname_lower = hostname.lower()
    for pattern, service in _PEER_SERVICE_MAP.items():
        if pattern in hostname_lower:
            return service
    return None


__all__ = ["setup_instrumentors"]
