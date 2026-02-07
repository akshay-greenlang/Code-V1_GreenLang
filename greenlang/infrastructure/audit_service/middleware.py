# -*- coding: utf-8 -*-
"""
Audit Middleware - Centralized Audit Logging Service (SEC-005)

FastAPI middleware that automatically captures HTTP request/response data
for audit logging. Integrates with the AuditService to emit structured
audit events with provenance tracking.

Features:
    - Request capture: path, method, headers (sanitized), client IP, user agent
    - Response capture: status code, content type, timing
    - User context extraction from request.state (post-auth middleware)
    - Sensitivity classification by path
    - Configurable path exclusions (health checks, metrics, static assets)
    - Sub-2ms overhead target for hot path
    - Async event emission to avoid blocking request processing

Security Compliance:
    - SOC 2 CC6.1 (Logical Access Control)
    - SOC 2 CC7.2 (System Monitoring)
    - ISO 27001 A.12.4 (Logging and Monitoring)
    - PCI DSS 10.2 (Audit Trail Requirements)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from greenlang.infrastructure.audit_service.audit_metrics import (
    AuditMetrics,
    get_audit_metrics,
)
from greenlang.infrastructure.audit_service.exclusions import (
    AuditExclusionRules,
    get_exclusion_rules,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Headers that should never be logged (security sensitive)
REDACTED_HEADERS: Set[str] = {
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-csrf-token",
    "x-xsrf-token",
    "proxy-authorization",
}

# Headers that are safe to log
SAFE_HEADERS: Set[str] = {
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "content-type",
    "content-length",
    "host",
    "origin",
    "referer",
    "user-agent",
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-real-ip",
    "x-request-id",
    "x-correlation-id",
    "x-tenant-id",
}

# Maximum request body size to capture (bytes)
MAX_BODY_CAPTURE_SIZE = 10_000

# Audit event type for API calls
API_EVENT_TYPE = "api.request"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AuditRequestContext:
    """Captured request context for audit logging.

    Attributes:
        request_id: Unique identifier for this request.
        timestamp: UTC timestamp when request was received.
        method: HTTP method (GET, POST, etc.).
        path: Request URL path.
        query_string: URL query parameters (may be redacted).
        headers: Sanitized headers dictionary.
        client_ip: Client IP address (respects X-Forwarded-For).
        user_agent: User-Agent header value.
        content_type: Request Content-Type header.
        content_length: Request body size in bytes.
    """

    request_id: str
    timestamp: datetime
    method: str
    path: str
    query_string: str
    headers: Dict[str, str]
    client_ip: str
    user_agent: str
    content_type: Optional[str]
    content_length: Optional[int]


@dataclass
class AuditResponseContext:
    """Captured response context for audit logging.

    Attributes:
        status_code: HTTP response status code.
        content_type: Response Content-Type header.
        content_length: Response body size in bytes.
        duration_ms: Request processing time in milliseconds.
    """

    status_code: int
    content_type: Optional[str]
    content_length: Optional[int]
    duration_ms: float


@dataclass
class AuditUserContext:
    """User context extracted from authenticated request.

    Attributes:
        user_id: Authenticated user ID.
        tenant_id: Tenant ID from auth context.
        email: User email (may be None).
        roles: List of user roles.
        auth_method: Authentication method used.
        session_id: Session ID if available.
    """

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    auth_method: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AuditEvent:
    """Complete audit event ready for logging.

    Attributes:
        event_id: Unique event identifier (UUID).
        event_type: Type classification (e.g., "api.request").
        sensitivity: Sensitivity level from path classification.
        request: Captured request context.
        response: Captured response context.
        user: User context from authentication.
        result: Outcome ("success", "failure", "denied").
        error_message: Error details if result is failure.
    """

    event_id: str
    event_type: str
    sensitivity: str
    request: AuditRequestContext
    response: AuditResponseContext
    user: AuditUserContext
    result: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the audit event.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "sensitivity": self.sensitivity,
            "result": self.result,
            "error_message": self.error_message,
            "request": {
                "request_id": self.request.request_id,
                "timestamp": self.request.timestamp.isoformat(),
                "method": self.request.method,
                "path": self.request.path,
                "query_string": self.request.query_string,
                "headers": self.request.headers,
                "client_ip": self.request.client_ip,
                "user_agent": self.request.user_agent,
                "content_type": self.request.content_type,
                "content_length": self.request.content_length,
            },
            "response": {
                "status_code": self.response.status_code,
                "content_type": self.response.content_type,
                "content_length": self.response.content_length,
                "duration_ms": self.response.duration_ms,
            },
            "user": {
                "user_id": self.user.user_id,
                "tenant_id": self.user.tenant_id,
                "email": self.user.email,
                "roles": self.user.roles,
                "auth_method": self.user.auth_method,
                "session_id": self.user.session_id,
            },
        }


# ---------------------------------------------------------------------------
# Audit Middleware
# ---------------------------------------------------------------------------


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware for automatic audit logging.

    Captures request/response data and emits structured audit events
    through the configured audit service. Designed for minimal overhead
    (<2ms) on the request hot path.

    The middleware:
    1. Checks if the path should be excluded (health checks, etc.)
    2. Captures request metadata at the start
    3. Passes through to the next middleware/handler
    4. Captures response metadata after handler returns
    5. Extracts user context from request.state.auth
    6. Classifies sensitivity based on path
    7. Emits audit event asynchronously

    Args:
        app: The ASGI application to wrap.
        audit_service: Optional AuditService instance for event logging.
            If None, events are logged to the standard logger.
        exclusion_rules: Optional AuditExclusionRules instance.
            Defaults to the module-level singleton.
        metrics: Optional AuditMetrics instance.
            Defaults to the module-level singleton.
        capture_request_body: Whether to capture request body content.
            Defaults to False for performance.
        capture_response_body: Whether to capture response body content.
            Defaults to False for performance.

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.infrastructure.audit_service.middleware import AuditMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(AuditMiddleware)
    """

    def __init__(
        self,
        app: Any,
        audit_service: Any = None,
        exclusion_rules: Optional[AuditExclusionRules] = None,
        metrics: Optional[AuditMetrics] = None,
        capture_request_body: bool = False,
        capture_response_body: bool = False,
    ) -> None:
        """Initialize the audit middleware.

        Args:
            app: The ASGI application.
            audit_service: Optional audit service for event emission.
            exclusion_rules: Path exclusion rules.
            metrics: Prometheus metrics collector.
            capture_request_body: Enable request body capture.
            capture_response_body: Enable response body capture.
        """
        super().__init__(app)
        self._audit_service = audit_service
        self._exclusion_rules = exclusion_rules or get_exclusion_rules()
        self._metrics = metrics or get_audit_metrics()
        self._capture_request_body = capture_request_body
        self._capture_response_body = capture_response_body

        logger.info(
            "AuditMiddleware initialized: capture_request_body=%s, "
            "capture_response_body=%s",
            capture_request_body,
            capture_response_body,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process a request through the audit pipeline.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The HTTP response.
        """
        # Fast path: skip excluded paths
        if self._exclusion_rules.should_exclude(request.url.path):
            return await call_next(request)

        # Start timing
        start_time = time.perf_counter()
        capture_start = start_time

        # Generate request ID if not present
        request_id = request.headers.get(
            "x-request-id",
            request.headers.get("x-correlation-id", str(uuid.uuid4())),
        )

        # Capture request context
        request_context = self._capture_request(request, request_id)

        capture_duration = time.perf_counter() - capture_start
        self._metrics.observe_latency("capture", capture_duration)

        # Process the request
        response: Optional[Response] = None
        error_message: Optional[str] = None
        result = "success"

        try:
            response = await call_next(request)
        except Exception as exc:
            # Log the exception but re-raise
            error_message = str(exc)
            result = "failure"
            logger.exception(
                "Request failed: request_id=%s path=%s error=%s",
                request_id,
                request.url.path,
                error_message,
            )
            raise

        finally:
            # Calculate total duration
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Capture response context
            response_context = self._capture_response(
                response,
                duration_ms,
            )

            # Determine result from status code if we have a response
            if response is not None:
                if response.status_code >= 500:
                    result = "failure"
                elif response.status_code in (401, 403):
                    result = "denied"
                elif response.status_code >= 400:
                    result = "failure"

            # Extract user context from request.state.auth
            enrich_start = time.perf_counter()
            user_context = self._extract_user_context(request)
            enrich_duration = time.perf_counter() - enrich_start
            self._metrics.observe_latency("enrich", enrich_duration)

            # Classify sensitivity
            sensitivity = self._exclusion_rules.get_sensitivity(
                request.url.path
            )

            # Create audit event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=API_EVENT_TYPE,
                sensitivity=sensitivity,
                request=request_context,
                response=response_context,
                user=user_context,
                result=result,
                error_message=error_message,
            )

            # Emit audit event (async, non-blocking)
            await self._emit_audit_event(audit_event)

            # Record metrics
            self._metrics.record_event(
                event_type=API_EVENT_TYPE,
                severity=self._severity_from_status(
                    response_context.status_code
                ),
                result=result,
            )

        return response

    # -------------------------------------------------------------------------
    # Request/Response capture
    # -------------------------------------------------------------------------

    def _capture_request(
        self,
        request: Request,
        request_id: str,
    ) -> AuditRequestContext:
        """Capture request metadata for audit logging.

        Args:
            request: The incoming HTTP request.
            request_id: The request correlation ID.

        Returns:
            Captured request context.
        """
        # Sanitize headers
        headers = self._sanitize_headers(dict(request.headers))

        # Get client IP (respecting reverse proxy headers)
        client_ip = self._extract_client_ip(request)

        # Get content info
        content_type = request.headers.get("content-type")
        content_length_str = request.headers.get("content-length")
        content_length = (
            int(content_length_str) if content_length_str else None
        )

        return AuditRequestContext(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            method=request.method,
            path=request.url.path,
            query_string=str(request.url.query) if request.url.query else "",
            headers=headers,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent", ""),
            content_type=content_type,
            content_length=content_length,
        )

    def _capture_response(
        self,
        response: Optional[Response],
        duration_ms: float,
    ) -> AuditResponseContext:
        """Capture response metadata for audit logging.

        Args:
            response: The HTTP response (may be None on exception).
            duration_ms: Request processing duration in milliseconds.

        Returns:
            Captured response context.
        """
        if response is None:
            return AuditResponseContext(
                status_code=500,
                content_type=None,
                content_length=None,
                duration_ms=duration_ms,
            )

        # Get content info from response headers
        content_type = response.headers.get("content-type")
        content_length_str = response.headers.get("content-length")
        content_length = (
            int(content_length_str) if content_length_str else None
        )

        return AuditResponseContext(
            status_code=response.status_code,
            content_type=content_type,
            content_length=content_length,
            duration_ms=duration_ms,
        )

    # -------------------------------------------------------------------------
    # User context extraction
    # -------------------------------------------------------------------------

    def _extract_user_context(self, request: Request) -> AuditUserContext:
        """Extract user context from request.state.auth.

        The auth context is expected to be populated by the authentication
        middleware (SEC-001) before this middleware runs.

        Args:
            request: The HTTP request with potential auth state.

        Returns:
            User context (may be empty for unauthenticated requests).
        """
        auth = getattr(request.state, "auth", None)
        if auth is None:
            return AuditUserContext()

        return AuditUserContext(
            user_id=getattr(auth, "user_id", None),
            tenant_id=getattr(auth, "tenant_id", None),
            email=getattr(auth, "email", None),
            roles=getattr(auth, "roles", []) or [],
            auth_method=getattr(auth, "auth_method", None),
            session_id=getattr(auth, "session_id", None),
        )

    # -------------------------------------------------------------------------
    # Audit event emission
    # -------------------------------------------------------------------------

    async def _emit_audit_event(self, event: AuditEvent) -> None:
        """Emit an audit event to the configured service.

        If no audit service is configured, logs to the standard logger.

        Args:
            event: The audit event to emit.
        """
        queue_start = time.perf_counter()

        try:
            if self._audit_service is not None:
                # Use the audit service's async log method
                await self._audit_service.log_api_event(
                    request_id=event.request.request_id,
                    event_type=event.event_type,
                    user_id=event.user.user_id,
                    tenant_id=event.user.tenant_id,
                    resource=event.request.path,
                    action=event.request.method,
                    result=event.result,
                    metadata=event.to_dict(),
                    sensitivity=event.sensitivity,
                )
            else:
                # Fallback to structured logging
                logger.info(
                    "audit_event",
                    extra={
                        "audit": event.to_dict(),
                    },
                )

            queue_duration = time.perf_counter() - queue_start
            self._metrics.observe_latency("queue", queue_duration)

        except Exception as exc:
            # Never fail the request due to audit logging errors
            logger.error(
                "Failed to emit audit event: event_id=%s error=%s",
                event.event_id,
                str(exc),
            )
            self._metrics.record_db_write_failure("emit_error")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers by redacting sensitive values.

        Args:
            headers: Raw headers dictionary.

        Returns:
            Headers with sensitive values redacted.
        """
        sanitized = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in REDACTED_HEADERS:
                sanitized[key] = "[REDACTED]"
            elif key_lower in SAFE_HEADERS:
                sanitized[key] = value
            else:
                # Include unknown headers but truncate long values
                if len(value) > 200:
                    sanitized[key] = value[:200] + "...[TRUNCATED]"
                else:
                    sanitized[key] = value
        return sanitized

    @staticmethod
    def _extract_client_ip(request: Request) -> str:
        """Extract client IP address respecting reverse proxies.

        Args:
            request: The HTTP request.

        Returns:
            Client IP address string.
        """
        # Check X-Forwarded-For header (set by reverse proxies)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"

    @staticmethod
    def _severity_from_status(status_code: int) -> str:
        """Map HTTP status code to log severity.

        Args:
            status_code: HTTP response status code.

        Returns:
            Severity level string.
        """
        if status_code < 400:
            return "info"
        elif status_code < 500:
            return "warning"
        else:
            return "error"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_audit_middleware(
    audit_service: Any = None,
    exclusion_rules: Optional[AuditExclusionRules] = None,
    metrics: Optional[AuditMetrics] = None,
    capture_request_body: bool = False,
    capture_response_body: bool = False,
) -> Callable:
    """Create an AuditMiddleware factory for use with FastAPI.

    This function returns a middleware class that can be added to FastAPI
    using ``app.add_middleware()``.

    Args:
        audit_service: Optional AuditService instance.
        exclusion_rules: Optional exclusion rules.
        metrics: Optional metrics collector.
        capture_request_body: Enable request body capture.
        capture_response_body: Enable response body capture.

    Returns:
        Middleware class with pre-configured settings.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     create_audit_middleware(audit_service=my_audit_service)
        ... )
    """

    class ConfiguredAuditMiddleware(AuditMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(
                app,
                audit_service=audit_service,
                exclusion_rules=exclusion_rules,
                metrics=metrics,
                capture_request_body=capture_request_body,
                capture_response_body=capture_response_body,
            )

    return ConfiguredAuditMiddleware


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "AuditRequestContext",
    "AuditResponseContext",
    "AuditUserContext",
    "AuditEvent",
    # Middleware
    "AuditMiddleware",
    "create_audit_middleware",
    # Constants
    "REDACTED_HEADERS",
    "SAFE_HEADERS",
    "API_EVENT_TYPE",
]
