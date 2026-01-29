"""
Audit middleware for GL Normalizer Service.

This module provides audit logging middleware that captures all API requests
and responses for compliance, debugging, and analytics. Supports structured
logging with correlation IDs.

Audit Events:
    - request_received: Incoming request logged
    - request_completed: Response sent with timing
    - request_failed: Error response with details

Features:
    - Request/response body capture (configurable)
    - Sensitive data masking
    - Correlation ID tracking
    - Performance metrics
    - Structured JSON logging

Usage:
    >>> from fastapi import FastAPI
    >>> from gl_normalizer_service.middleware.audit import AuditMiddleware
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(AuditMiddleware)
"""

import time
from datetime import datetime
from typing import Callable, Optional
from uuid import uuid4

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from gl_normalizer_service.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware for compliance and monitoring.

    Captures all API requests and responses with timing information,
    user context, and correlation IDs for tracing.

    Attributes:
        capture_body: Whether to capture request/response bodies
        sensitive_headers: Headers to mask in logs
        sensitive_fields: JSON fields to mask in body logs
        max_body_size: Maximum body size to capture (bytes)

    Example:
        >>> app.add_middleware(
        ...     AuditMiddleware,
        ...     capture_body=True,
        ...     max_body_size=10000
        ... )
    """

    def __init__(
        self,
        app,
        capture_body: bool = False,
        sensitive_headers: Optional[set[str]] = None,
        sensitive_fields: Optional[set[str]] = None,
        max_body_size: int = 10000,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize audit middleware.

        Args:
            app: ASGI application
            capture_body: Whether to capture request/response bodies
            sensitive_headers: Headers to mask (case-insensitive)
            sensitive_fields: JSON fields to mask in bodies
            max_body_size: Maximum body size to capture
            settings: Application settings
        """
        super().__init__(app)
        self.capture_body = capture_body
        self.max_body_size = max_body_size
        self.settings = settings or get_settings()

        # Default sensitive headers
        self.sensitive_headers = {h.lower() for h in (sensitive_headers or {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
        })}

        # Default sensitive fields
        self.sensitive_fields = sensitive_fields or {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "credentials",
        }

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request with audit logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler
        """
        # Generate or extract request ID
        request_id = request.headers.get(
            "X-Request-ID",
            f"req_{uuid4().hex[:12]}"
        )
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Extract request metadata
        audit_context = await self._build_audit_context(request, request_id)

        # Log request received
        logger.info(
            "request_received",
            **audit_context,
            event_type="request",
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Log successful response
            logger.info(
                "request_completed",
                **audit_context,
                event_type="response",
                status_code=response.status_code,
                duration_ms=duration_ms,
                content_length=response.headers.get("content-length", "unknown"),
            )

            # Add audit headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms}ms"

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Log error
            logger.error(
                "request_failed",
                **audit_context,
                event_type="error",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms,
                exc_info=True,
            )

            # Re-raise for error handlers
            raise

    async def _build_audit_context(
        self, request: Request, request_id: str
    ) -> dict:
        """
        Build audit context from request.

        Args:
            request: HTTP request
            request_id: Request correlation ID

        Returns:
            Audit context dictionary
        """
        context = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
        }

        # Add user context if available
        if hasattr(request.state, "user"):
            user = request.state.user
            context.update({
                "user_id": getattr(user, "id", None),
                "tenant_id": getattr(user, "tenant_id", None),
                "auth_method": getattr(request.state, "auth_method", None),
            })

        # Add masked headers
        context["headers"] = self._mask_headers(dict(request.headers))

        # Capture request body if enabled
        if self.capture_body and request.method in ("POST", "PUT", "PATCH"):
            body = await self._capture_body(request)
            if body:
                context["request_body"] = self._mask_sensitive_fields(body)

        return context

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP from request.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _mask_headers(self, headers: dict) -> dict:
        """
        Mask sensitive headers.

        Args:
            headers: Request headers

        Returns:
            Headers with sensitive values masked
        """
        masked = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                masked[key] = "[REDACTED]"
            else:
                masked[key] = value
        return masked

    async def _capture_body(self, request: Request) -> Optional[str]:
        """
        Capture request body for logging.

        Args:
            request: HTTP request

        Returns:
            Body content or None if too large
        """
        try:
            # Read body (this consumes it)
            body = await request.body()

            # Check size limit
            if len(body) > self.max_body_size:
                return f"[TRUNCATED: {len(body)} bytes]"

            # Try to decode as string
            try:
                return body.decode("utf-8")
            except UnicodeDecodeError:
                return f"[BINARY: {len(body)} bytes]"

        except Exception:
            return None

    def _mask_sensitive_fields(self, body: str) -> str:
        """
        Mask sensitive fields in JSON body.

        Args:
            body: Request body string

        Returns:
            Body with sensitive fields masked
        """
        import json
        import re

        # Try to parse as JSON
        try:
            data = json.loads(body)
            masked_data = self._mask_dict(data)
            return json.dumps(masked_data)
        except json.JSONDecodeError:
            # Not JSON, use regex masking
            for field in self.sensitive_fields:
                # Mask "field": "value" patterns
                pattern = rf'("{field}"\s*:\s*)"[^"]*"'
                body = re.sub(pattern, rf'\1"[REDACTED]"', body, flags=re.IGNORECASE)
            return body

    def _mask_dict(self, data: dict | list) -> dict | list:
        """
        Recursively mask sensitive fields in dict/list.

        Args:
            data: Data structure to mask

        Returns:
            Masked data structure
        """
        if isinstance(data, dict):
            return {
                k: "[REDACTED]" if k.lower() in self.sensitive_fields
                else self._mask_dict(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._mask_dict(item) for item in data]
        else:
            return data


class AuditLogger:
    """
    Structured audit logger for compliance events.

    Provides methods for logging specific audit events with
    consistent formatting and metadata.

    Example:
        >>> audit = AuditLogger()
        >>> audit.log_normalization(
        ...     request_id="req_123",
        ...     user_id="user_456",
        ...     input_value="100 kg",
        ...     output_value="0.1 metric_ton"
        ... )
    """

    def __init__(self, logger_name: str = "audit"):
        """
        Initialize audit logger.

        Args:
            logger_name: Logger name for filtering
        """
        self.logger = structlog.get_logger(logger_name)

    def log_authentication(
        self,
        request_id: str,
        method: str,
        success: bool,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Log authentication event.

        Args:
            request_id: Request correlation ID
            method: Authentication method (api_key, jwt)
            success: Whether authentication succeeded
            user_id: User ID if successful
            tenant_id: Tenant ID if successful
            reason: Failure reason if unsuccessful
        """
        self.logger.info(
            "authentication",
            audit_type="authentication",
            request_id=request_id,
            method=method,
            success=success,
            user_id=user_id,
            tenant_id=tenant_id,
            reason=reason,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def log_authorization(
        self,
        request_id: str,
        user_id: str,
        action: str,
        resource: str,
        allowed: bool,
        reason: Optional[str] = None,
    ) -> None:
        """
        Log authorization decision.

        Args:
            request_id: Request correlation ID
            user_id: User ID
            action: Requested action
            resource: Target resource
            allowed: Whether access was granted
            reason: Denial reason if not allowed
        """
        self.logger.info(
            "authorization",
            audit_type="authorization",
            request_id=request_id,
            user_id=user_id,
            action=action,
            resource=resource,
            allowed=allowed,
            reason=reason,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def log_normalization(
        self,
        request_id: str,
        audit_id: str,
        user_id: str,
        tenant_id: str,
        input_value: str,
        input_unit: str,
        output_value: float,
        output_unit: str,
        confidence: float,
        needs_review: bool,
        duration_ms: int,
    ) -> None:
        """
        Log normalization operation.

        Args:
            request_id: Request correlation ID
            audit_id: Normalization audit ID
            user_id: User ID
            tenant_id: Tenant ID
            input_value: Original input value
            input_unit: Original input unit
            output_value: Normalized output value
            output_unit: Canonical output unit
            confidence: Confidence score
            needs_review: Whether review is needed
            duration_ms: Processing duration
        """
        self.logger.info(
            "normalization",
            audit_type="normalization",
            request_id=request_id,
            audit_id=audit_id,
            user_id=user_id,
            tenant_id=tenant_id,
            input_value=input_value,
            input_unit=input_unit,
            output_value=output_value,
            output_unit=output_unit,
            confidence=confidence,
            needs_review=needs_review,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def log_batch_operation(
        self,
        request_id: str,
        user_id: str,
        tenant_id: str,
        operation: str,
        item_count: int,
        success_count: int,
        failed_count: int,
        duration_ms: int,
    ) -> None:
        """
        Log batch operation.

        Args:
            request_id: Request correlation ID
            user_id: User ID
            tenant_id: Tenant ID
            operation: Operation type (batch_normalize, job_create)
            item_count: Total items
            success_count: Successful items
            failed_count: Failed items
            duration_ms: Processing duration
        """
        self.logger.info(
            "batch_operation",
            audit_type="batch_operation",
            request_id=request_id,
            user_id=user_id,
            tenant_id=tenant_id,
            operation=operation,
            item_count=item_count,
            success_count=success_count,
            failed_count=failed_count,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def log_data_access(
        self,
        request_id: str,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> None:
        """
        Log data access for compliance.

        Args:
            request_id: Request correlation ID
            user_id: User ID
            tenant_id: Tenant ID
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            action: Access action (read, write, delete)
        """
        self.logger.info(
            "data_access",
            audit_type="data_access",
            request_id=request_id,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
