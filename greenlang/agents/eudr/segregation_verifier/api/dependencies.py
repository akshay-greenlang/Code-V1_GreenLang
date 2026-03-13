# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-010 Segregation Verifier

FastAPI dependency injection providers for authentication, authorization,
rate limiting, request validation, pagination, and service access.
All route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_sgv_service: Returns the Segregation Verifier service singleton (lazy init).
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - Validators: Request-level validation helpers.

Rate Limiter Tiers:
    - standard: 200 requests/minute (reads)
    - batch: 10 requests/minute (bulk imports)
    - report: 20 requests/minute (report generation)
    - write: 50 requests/minute (mutations)
    - export: 10 requests/minute (downloads)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Path, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OAuth2 / API Key security schemes
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False,
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# User model for auth context
# ---------------------------------------------------------------------------


class AuthUser(BaseModel):
    """Authenticated user context extracted from JWT or API key."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(default="", description="User email address")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    operator_id: str = Field(default="", description="Associated operator ID")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    permissions: List[str] = Field(
        default_factory=list, description="Granted permissions"
    )


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> AuthUser:
    """Extract and validate the current user from JWT token or API key.

    Checks request.state.auth first (populated by AuthenticationMiddleware
    from SEC-001). Falls back to manual token/api_key validation.

    Args:
        request: FastAPI request object.
        token: OAuth2 bearer token from Authorization header.
        api_key: API key from X-API-Key header.

    Returns:
        AuthUser with user identity, tenant, roles, and permissions.

    Raises:
        HTTPException: 401 if no valid credentials are provided.
    """
    # Check if AuthenticationMiddleware already populated auth context
    auth_ctx = getattr(request.state, "auth", None)
    if auth_ctx is not None:
        return AuthUser(
            user_id=getattr(auth_ctx, "user_id", ""),
            email=getattr(auth_ctx, "email", ""),
            tenant_id=getattr(auth_ctx, "tenant_id", "default"),
            operator_id=getattr(auth_ctx, "operator_id", ""),
            roles=getattr(auth_ctx, "roles", []),
            permissions=getattr(auth_ctx, "permissions", []),
        )

    # Fallback: validate token manually
    if token:
        try:
            from greenlang.infrastructure.auth_service.jwt_service import (
                decode_token,
            )

            payload = decode_token(token)
            return AuthUser(
                user_id=payload.get("sub", ""),
                email=payload.get("email", ""),
                tenant_id=payload.get("tenant_id", "default"),
                operator_id=payload.get("operator_id", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired JWT token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Fallback: validate API key manually
    if api_key:
        try:
            from greenlang.infrastructure.auth_service.api_key_service import (
                validate_api_key,
            )

            key_data = validate_api_key(api_key)
            return AuthUser(
                user_id=key_data.get("user_id", "api-user"),
                email=key_data.get("email", ""),
                tenant_id=key_data.get("tenant_id", "default"),
                operator_id=key_data.get("operator_id", ""),
                roles=key_data.get("roles", ["api-user"]),
                permissions=key_data.get("permissions", []),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# RBAC permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(permission: str) -> Callable:
    """Factory returning a FastAPI dependency that checks RBAC permissions.

    Uses wildcard matching: ``eudr-sgv:*`` grants all
    ``eudr-sgv:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-sgv:scp:create``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/scp")
        ... async def register_scp(
        ...     user: AuthUser = Depends(require_permission("eudr-sgv:scp:create"))
        ... ):
        ...     ...
    """

    async def _check_permission(
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        # Admin and platform_admin bypass all permission checks
        if "admin" in user.roles or "platform_admin" in user.roles:
            return user

        # Check exact match
        if permission in user.permissions:
            return user

        # Check wildcard patterns
        parts = permission.split(":")
        for i in range(len(parts)):
            wildcard = ":".join(parts[: i + 1]) + ":*"
            if wildcard in user.permissions:
                return user
        # Check top-level wildcard
        if parts[0] + ":*" in user.permissions:
            return user

        logger.warning(
            "Permission denied: user=%s permission=%s available=%s",
            user.user_id,
            permission,
            user.permissions,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required permission: {permission}",
        )

    return _check_permission


# ---------------------------------------------------------------------------
# Pagination parameters
# ---------------------------------------------------------------------------


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""

    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


def get_pagination(
    limit: int = Query(default=50, ge=1, le=1000, description="Results per page"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
) -> PaginationParams:
    """Extract pagination parameters from query string.

    Args:
        limit: Maximum number of results to return (1-1000).
        offset: Number of results to skip for pagination.

    Returns:
        PaginationParams with validated limit and offset.
    """
    return PaginationParams(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-endpoint)
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding-window rate limiter per user per endpoint.

    Tracks request timestamps and enforces a maximum request count
    within a rolling time window. Thread-safe for async usage.

    Attributes:
        max_requests: Maximum requests allowed per window.
        window_seconds: Rolling window duration in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    async def __call__(
        self,
        request: Request,
        user: AuthUser = Depends(get_current_user),
    ) -> None:
        """Check rate limit for the current user and endpoint.

        Args:
            request: FastAPI request object.
            user: Authenticated user.

        Raises:
            HTTPException: 429 if rate limit exceeded.
        """
        key = f"{user.user_id}:{request.url.path}"
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > cutoff
        ]

        if len(self._requests[key]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded: user=%s endpoint=%s limit=%d/%ds",
                user.user_id,
                request.url.path,
                self.max_requests,
                self.window_seconds,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {self.max_requests} requests "
                    f"per {self.window_seconds} seconds"
                ),
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        self._requests[key].append(now)


# ---------------------------------------------------------------------------
# Pre-configured rate limiter instances
# ---------------------------------------------------------------------------

_rate_limit_standard = RateLimiter(max_requests=200, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=50, window_seconds=60)
_rate_limit_batch = RateLimiter(max_requests=10, window_seconds=60)
_rate_limit_report = RateLimiter(max_requests=20, window_seconds=60)
_rate_limit_export = RateLimiter(max_requests=10, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_standard(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Standard rate limit: 200 requests/minute."""
    await _rate_limit_standard(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 50 requests/minute."""
    await _rate_limit_write(request, user)


async def rate_limit_batch(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Batch import rate limit: 10 requests/minute."""
    await _rate_limit_batch(request, user)


async def rate_limit_report(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Report generation rate limit: 20 requests/minute."""
    await _rate_limit_report(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export/download rate limit: 10 requests/minute."""
    await _rate_limit_export(request, user)


# ---------------------------------------------------------------------------
# Request validators
# ---------------------------------------------------------------------------


def validate_uuid_path(
    value: str,
    param_name: str = "id",
) -> str:
    """Validate a path parameter is a valid UUID or non-empty string.

    Args:
        value: The path parameter value to validate.
        param_name: Name for error messages.

    Returns:
        The validated value.

    Raises:
        HTTPException: 400 if the value is empty or invalid.
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{param_name} must be a non-empty string",
        )
    return value.strip()


def validate_scp_id(
    scp_id: str = Path(..., description="Segregation control point identifier"),
) -> str:
    """Validate scp_id path parameter."""
    return validate_uuid_path(scp_id, "scp_id")


def validate_facility_id(
    facility_id: str = Path(..., description="Facility identifier"),
) -> str:
    """Validate facility_id path parameter."""
    return validate_uuid_path(facility_id, "facility_id")


def validate_vehicle_id(
    vehicle_id: str = Path(..., description="Transport vehicle identifier"),
) -> str:
    """Validate vehicle_id path parameter."""
    return validate_uuid_path(vehicle_id, "vehicle_id")


def validate_line_id(
    line_id: str = Path(..., description="Processing line identifier"),
) -> str:
    """Validate line_id path parameter."""
    return validate_uuid_path(line_id, "line_id")


def validate_event_id(
    event_id: str = Path(..., description="Contamination event identifier"),
) -> str:
    """Validate event_id path parameter."""
    return validate_uuid_path(event_id, "event_id")


def validate_report_id(
    report_id: str = Path(..., description="Report identifier"),
) -> str:
    """Validate report_id path parameter."""
    return validate_uuid_path(report_id, "report_id")


def validate_job_id(
    job_id: str = Path(..., description="Batch job identifier"),
) -> str:
    """Validate batch job ID path parameter."""
    return validate_uuid_path(job_id, "job_id")


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# ---------------------------------------------------------------------------
# Success envelope response
# ---------------------------------------------------------------------------


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")


# ---------------------------------------------------------------------------
# Service singleton (lazy initialization with stub fallback)
# ---------------------------------------------------------------------------


class _SgvServiceStub:
    """Stub service for Segregation Verifier operations.

    Provides safe no-op methods when the actual service module
    is not yet initialized. Enables API startup and health checks
    without requiring full engine initialization.
    """

    def __init__(self) -> None:
        self._initialized = False
        logger.info("Segregation Verifier service stub initialized")

    @property
    def is_initialized(self) -> bool:
        """Whether the real service has been initialized."""
        return self._initialized


_sgv_service_instance: Optional[Any] = None


def get_sgv_service() -> Any:
    """Return the Segregation Verifier service singleton.

    Attempts to import and initialize the real service on first call.
    Falls back to a stub if the real service is not available (e.g.,
    during testing or early startup).

    Returns:
        Segregation Verifier service instance (real or stub).
    """
    global _sgv_service_instance

    if _sgv_service_instance is not None:
        return _sgv_service_instance

    try:
        from greenlang.agents.eudr.segregation_verifier import (
            SegregationVerifierEngine,
        )

        _sgv_service_instance = {
            "engine": SegregationVerifierEngine(),
        }
        logger.info("Segregation Verifier service engine initialized")
    except Exception as exc:
        logger.warning(
            "Could not initialize SGV service engine, using stub: %s",
            exc,
        )
        _sgv_service_instance = _SgvServiceStub()

    return _sgv_service_instance


def reset_sgv_service() -> None:
    """Reset the service singleton. Used in testing."""
    global _sgv_service_instance
    _sgv_service_instance = None


# ---------------------------------------------------------------------------
# Request ID injection
# ---------------------------------------------------------------------------


def get_request_id(request: Request) -> str:
    """Extract or generate a request correlation ID.

    Checks for X-Request-ID header first, then generates a new UUID4.

    Args:
        request: FastAPI request object.

    Returns:
        Request correlation ID string.
    """
    request_id = request.headers.get("X-Request-ID", "")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "ErrorResponse",
    "PaginationParams",
    "RateLimiter",
    "SuccessResponse",
    "_SgvServiceStub",
    "api_key_header",
    "get_current_user",
    "get_pagination",
    "get_request_id",
    "get_sgv_service",
    "oauth2_scheme",
    "rate_limit_batch",
    "rate_limit_export",
    "rate_limit_report",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "reset_sgv_service",
    "validate_event_id",
    "validate_facility_id",
    "validate_job_id",
    "validate_line_id",
    "validate_report_id",
    "validate_scp_id",
    "validate_uuid_path",
    "validate_vehicle_id",
]
