"""
Structured Logging Middleware - INFRA-009

FastAPI/Starlette middleware that automatically logs HTTP request/response
cycles with correlation IDs, tenant isolation, and duration tracking. Every
request receives a unique ``request_id`` (from the ``X-Request-ID`` header
or generated as a UUID4), and the middleware binds this into the structlog
context so that every log entry emitted during request processing carries
the correlation ID.

Request flow:
    1. Extract or generate ``request_id`` from ``X-Request-ID`` header
    2. Extract ``trace_id`` from ``X-Trace-ID`` or ``traceparent`` header
    3. Extract ``tenant_id`` from ``X-Tenant-ID`` header or JWT payload
    4. Bind context variables for the duration of the request
    5. Log ``request_started`` at INFO (skip health/metrics endpoints)
    6. Call the next middleware/handler
    7. Log ``request_completed`` with ``status_code`` and ``duration_ms``
    8. Set ``X-Request-ID`` and ``X-Trace-ID`` response headers
    9. Clear context variables

Classes:
    - StructuredLoggingMiddleware: The middleware implementation.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.logging.middleware import StructuredLoggingMiddleware
    >>> app = FastAPI()
    >>> app.add_middleware(StructuredLoggingMiddleware)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any, Optional, Set

from greenlang.infrastructure.logging.config import LoggingConfig, get_config
from greenlang.infrastructure.logging.context import bind_context, clear_context

try:
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    BaseHTTPMiddleware = object  # type: ignore[misc, assignment]
    RequestResponseEndpoint = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    Response = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


# Paths to skip for request logging (health checks, probes, static assets)
_DEFAULT_SKIP_PATHS: Set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/livez",
    "/metrics",
    "/favicon.ico",
}


class StructuredLoggingMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Middleware that logs HTTP request/response cycles with correlation IDs.

    Automatically generates or propagates request IDs, extracts tenant
    context from headers or JWT claims, logs request start/complete events,
    and sets correlation headers on the response.

    Health check and metrics endpoints are skipped to avoid log noise.

    Attributes:
        config: The logging configuration.
        skip_paths: Set of URL paths to skip logging for.
    """

    def __init__(
        self,
        app: Any,
        config: Optional[LoggingConfig] = None,
        skip_paths: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the structured logging middleware.

        Args:
            app: The ASGI application.
            config: Optional logging configuration. Uses the singleton if None.
            skip_paths: Additional paths to skip logging for. Merged with
                the default skip paths.
        """
        super().__init__(app)
        self.config = config or get_config()

        self.skip_paths: Set[str] = set(_DEFAULT_SKIP_PATHS)
        if skip_paths:
            self.skip_paths.update(skip_paths)

        logger.info(
            "StructuredLoggingMiddleware initialized (skip_paths=%d)",
            len(self.skip_paths),
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process each request through the logging pipeline.

        Extracts correlation IDs, binds them to the structlog context,
        logs request start/complete events, and propagates IDs on the
        response headers.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response with correlation headers.
        """
        # Generate or extract correlation IDs
        request_id = self._get_or_generate_request_id(request)
        trace_id = self._extract_trace_id(request)
        tenant_id = self._extract_tenant_id(request)

        start_time = time.monotonic()
        path = request.url.path
        method = request.method
        skip_logging = self._should_skip_logging(path)

        # Bind context for the duration of this request
        context_vars: dict[str, Any] = {
            "request_id": request_id,
            "method": method,
            "path": path,
        }
        if trace_id:
            context_vars["trace_id"] = trace_id
        if tenant_id:
            context_vars["tenant_id"] = tenant_id

        # Extract client IP (may be redacted downstream by the processor)
        client_ip = self._get_client_ip(request)
        if client_ip:
            context_vars["client_ip"] = client_ip

        bind_context(**context_vars)

        try:
            # Log request start (skip for health/metrics endpoints)
            if not skip_logging:
                logger.info(
                    "request_started",
                    extra={
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "trace_id": trace_id or "",
                        "tenant_id": tenant_id or "",
                    },
                )

            # Call the next handler
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.monotonic() - start_time) * 1000

            # Log request completion
            if not skip_logging:
                log_method = (
                    logger.warning if response.status_code >= 400 else logger.info
                )
                log_method(
                    "request_completed",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2),
                        "method": method,
                        "path": path,
                    },
                )

            # Set correlation headers on the response
            response.headers["X-Request-ID"] = request_id
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as exc:
            # Log the failure
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_failed",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "duration_ms": round(duration_ms, 2),
                    "method": method,
                    "path": path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        finally:
            clear_context()

    # -- Private helpers -----------------------------------------------------

    def _get_or_generate_request_id(self, request: Request) -> str:
        """Extract request ID from header or generate a new UUID4.

        Args:
            request: The HTTP request.

        Returns:
            The request correlation ID string.
        """
        request_id = request.headers.get(
            self.config.correlation_header, ""
        ).strip()
        if not request_id:
            request_id = str(uuid.uuid4())
        return request_id

    def _extract_trace_id(self, request: Request) -> Optional[str]:
        """Extract trace ID from headers.

        Checks the configured ``trace_header`` first, then falls back to
        the W3C ``traceparent`` header. For ``traceparent``, extracts
        the trace-id portion (second field).

        Args:
            request: The HTTP request.

        Returns:
            The trace ID string, or None if not found.
        """
        # Check primary trace header
        trace_id = request.headers.get(
            self.config.trace_header, ""
        ).strip()
        if trace_id:
            return trace_id

        # Fall back to W3C traceparent: version-trace_id-parent_id-flags
        traceparent = request.headers.get("traceparent", "").strip()
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 2:
                return parts[1]

        return None

    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from header or JWT claims.

        Tries the ``X-Tenant-ID`` header first. If not found, attempts
        to decode the JWT payload from the Authorization header and
        extract ``tenant_id`` or ``org_id`` from claims.

        The JWT is decoded WITHOUT signature verification because the
        API gateway (Kong) has already verified the token.

        Args:
            request: The HTTP request.

        Returns:
            The tenant ID string, or None if not found.
        """
        # Try explicit header first
        tenant_id = request.headers.get("X-Tenant-ID", "").strip()
        if tenant_id:
            return tenant_id

        # Try JWT claims
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None

        token = auth_header[7:].strip()
        if not token:
            return None

        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None

            # Decode the payload (base64url without verification)
            payload_b64 = parts[1]
            # Add padding if needed
            padding_needed = 4 - len(payload_b64) % 4
            if padding_needed != 4:
                payload_b64 += "=" * padding_needed

            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            claims = json.loads(payload_bytes)

            tenant = claims.get("tenant_id") or claims.get("org_id")
            return str(tenant) if tenant else None

        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _get_client_ip(request: Request) -> Optional[str]:
        """Extract client IP address from the request.

        Checks ``X-Forwarded-For`` first (for proxied requests), then
        falls back to the direct client host.

        Args:
            request: The HTTP request.

        Returns:
            The client IP string, or None if unavailable.
        """
        forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            # X-Forwarded-For may contain multiple IPs; the first is the client
            return forwarded_for.split(",")[0].strip()

        if request.client:
            return request.client.host

        return None

    def _should_skip_logging(self, path: str) -> bool:
        """Determine whether to skip request logging for the given path.

        Args:
            path: The URL path of the request.

        Returns:
            True if the path should not produce request_started/completed logs.
        """
        normalized = path.rstrip("/")
        return normalized in self.skip_paths


__all__ = ["StructuredLoggingMiddleware"]
