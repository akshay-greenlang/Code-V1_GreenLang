# -*- coding: utf-8 -*-
"""
TracingMiddleware - ASGI middleware for FastAPI request tracing (OBS-003)

Provides an ASGI middleware that:
1. Extracts incoming W3C TraceContext from request headers.
2. Creates a SERVER span for every non-health HTTP request.
3. Enriches the span with tenant ID, request ID, HTTP attributes.
4. Injects trace context into response headers for correlation.
5. Records response status code and duration.

The middleware is designed for **manual attachment** via ``configure_tracing()``
and complements (not replaces) the FastAPI auto-instrumentor.  When both are
active, this middleware provides the GreenLang-specific enrichment layer on
top of the auto-instrumentor's generic HTTP span.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.tracing_service.middleware import TracingMiddleware
    >>> app = FastAPI()
    >>> app.add_middleware(TracingMiddleware, service_name="api-service")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from greenlang.infrastructure.tracing_service.context import (
    extract_trace_context,
    inject_trace_context,
)
from greenlang.infrastructure.tracing_service.span_enrichment import (
    SpanEnricher,
    GL_TENANT_ID,
    GL_REQUEST_ID,
    GL_CORRELATION_ID,
    GL_USER_ID,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode, SpanKind

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Excluded paths
# ---------------------------------------------------------------------------

_EXCLUDED_PATHS: Set[str] = {
    "/health",
    "/ready",
    "/readyz",
    "/livez",
    "/healthz",
    "/ping",
    "/metrics",
    "/favicon.ico",
}


# ---------------------------------------------------------------------------
# TracingMiddleware (Starlette-style)
# ---------------------------------------------------------------------------

class TracingMiddleware:
    """ASGI middleware that wraps HTTP requests in OpenTelemetry SERVER spans.

    This middleware is compatible with Starlette / FastAPI's ``add_middleware``
    pattern.  It uses the ``pure ASGI`` interface so it works with any ASGI
    application.

    Attributes:
        app: The wrapped ASGI application.
        service_name: Service name used for the tracer scope.
        enricher: SpanEnricher instance for GreenLang attributes.
        excluded_paths: Paths to skip tracing for (health checks, etc.).
    """

    def __init__(
        self,
        app: Any,
        service_name: str = "api-service",
        enricher: Optional[SpanEnricher] = None,
        excluded_paths: Optional[Set[str]] = None,
    ) -> None:
        """Initialise the middleware.

        Args:
            app: The ASGI application to wrap.
            service_name: Logical service name for tracer scope.
            enricher: Optional SpanEnricher (auto-created if None).
            excluded_paths: Set of URL paths to skip.
        """
        self.app = app
        self.service_name = service_name
        self.enricher = enricher or SpanEnricher()
        self.excluded_paths = excluded_paths or _EXCLUDED_PATHS

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        """Process an ASGI request.

        Non-HTTP scopes (websocket, lifespan) are passed through.
        Health-check paths are passed through without tracing.

        Args:
            scope: ASGI scope dict.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Skip health checks and metrics endpoints
        if path in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        # Skip paths starting with excluded prefixes
        if any(path.startswith(p + "/") for p in self.excluded_paths if p != "/"):
            await self.app(scope, receive, send)
            return

        if not OTEL_AVAILABLE:
            await self.app(scope, receive, send)
            return

        # Extract incoming trace context from headers
        raw_headers = scope.get("headers", [])
        header_dict = _asgi_headers_to_dict(raw_headers)
        parent_ctx = extract_trace_context(header_dict)

        method: str = scope.get("method", "GET")
        span_name = f"{method} {path}"

        tracer = trace.get_tracer(self.service_name)

        start_kwargs: Dict[str, Any] = {"kind": SpanKind.SERVER}
        if parent_ctx is not None:
            start_kwargs["context"] = parent_ctx

        with tracer.start_as_current_span(span_name, **start_kwargs) as span:
            # -- Standard HTTP attributes
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", path)
            span.set_attribute("http.scheme", scope.get("scheme", "https"))
            span.set_attribute("net.host.name", _get_host(raw_headers))

            query = scope.get("query_string", b"")
            if query:
                span.set_attribute(
                    "http.query_string",
                    query.decode("utf-8", errors="replace"),
                )

            # -- GreenLang attributes from headers
            tenant_id = header_dict.get("x-tenant-id", "")
            if tenant_id:
                span.set_attribute(GL_TENANT_ID, tenant_id)

            request_id = header_dict.get("x-request-id", "")
            if request_id:
                span.set_attribute(GL_REQUEST_ID, request_id)

            correlation_id = header_dict.get("x-correlation-id", "")
            if correlation_id:
                span.set_attribute(GL_CORRELATION_ID, correlation_id)

            user_id = header_dict.get("x-user-id", "")
            if user_id:
                span.set_attribute(GL_USER_ID, user_id)

            # -- Track response status
            status_code: int = 500
            start_time = time.monotonic()

            async def send_wrapper(message: Dict[str, Any]) -> None:
                nonlocal status_code
                if message.get("type") == "http.response.start":
                    status_code = message.get("status", 500)
                    span.set_attribute("http.status_code", status_code)

                    if status_code >= 500:
                        span.set_status(StatusCode.ERROR, f"HTTP {status_code}")
                    elif status_code >= 400:
                        span.set_status(StatusCode.ERROR, f"HTTP {status_code}")

                    # Inject trace context into response headers
                    response_headers: List[List[bytes]] = list(
                        message.get("headers", [])
                    )
                    trace_headers: Dict[str, str] = {}
                    inject_trace_context(trace_headers)
                    for key, value in trace_headers.items():
                        response_headers.append(
                            [key.encode("utf-8"), value.encode("utf-8")]
                        )
                    message["headers"] = response_headers

                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise
            finally:
                duration_ms = (time.monotonic() - start_time) * 1000
                span.set_attribute("http.duration_ms", round(duration_ms, 2))
                span.set_attribute("http.status_code", status_code)


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _asgi_headers_to_dict(raw_headers: List[Tuple[bytes, bytes]]) -> Dict[str, str]:
    """Convert ASGI byte-tuple headers to a case-normalised string dict.

    ASGI headers arrive as ``[(b"host", b"example.com"), ...]``.  This
    converts them to ``{"host": "example.com", ...}`` with lowercased keys.

    Args:
        raw_headers: ASGI-style header list.

    Returns:
        Dict with lowercase string keys and decoded string values.
    """
    result: Dict[str, str] = {}
    for raw_key, raw_value in raw_headers:
        try:
            key = raw_key.decode("latin-1").lower()
            value = raw_value.decode("latin-1")
            result[key] = value
        except Exception:
            continue
    return result


def _get_host(raw_headers: List[Tuple[bytes, bytes]]) -> str:
    """Extract the Host header value from ASGI headers.

    Args:
        raw_headers: ASGI-style header list.

    Returns:
        The host string, or "unknown".
    """
    for raw_key, raw_value in raw_headers:
        try:
            if raw_key.decode("latin-1").lower() == "host":
                return raw_value.decode("latin-1")
        except Exception:
            continue
    return "unknown"


__all__ = ["TracingMiddleware"]
