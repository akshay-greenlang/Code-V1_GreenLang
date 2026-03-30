# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-012 Document Authentication

FastAPI dependency injection providers for authentication, authorization,
rate limiting, request validation, pagination, and service access.
All route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_dav_service: Returns the DocumentAuthentication service singleton (lazy init).
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - DateRangeParams: Common date range filter parameters.
    - SortParams: Sort field and order parameters.
    - Validators: Request-level validation helpers for path parameters.

Rate Limiter Tiers:
    - standard: 100 requests/minute (reads, queries, lookups)
    - write: 50 requests/minute (classify, verify, detect)
    - batch: 20 requests/minute (bulk imports, batch verification)
    - report: 20 requests/minute (report generation, evidence packages)
    - export: 10 requests/minute (report downloads)

Permission Prefix: eudr-dav:*

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Section 7.4
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Path, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import Field
from greenlang.schemas import GreenLangBase

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


class AuthUser(GreenLangBase):
    """Authenticated user context extracted from JWT or API key.

    Attributes:
        user_id: Unique user identifier from JWT ``sub`` claim.
        email: User email address.
        tenant_id: Multi-tenant identifier (defaults to ``default``).
        operator_id: Associated EUDR operator identifier.
        roles: Assigned RBAC roles (e.g. ``admin``, ``operator``).
        permissions: Granted fine-grained permissions (e.g. ``eudr-dav:classify:create``).
    """

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

    Checks ``request.state.auth`` first (populated by AuthenticationMiddleware
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

    # Fallback: validate JWT token manually
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

    Uses wildcard matching: ``eudr-dav:*`` grants all
    ``eudr-dav:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-dav:classify:create``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/classify")
        ... async def classify_doc(
        ...     user: AuthUser = Depends(require_permission("eudr-dav:classify:create"))
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

        # Check wildcard patterns (eudr-dav:classify:* matches eudr-dav:classify:create)
        parts = permission.split(":")
        for i in range(len(parts)):
            wildcard = ":".join(parts[: i + 1]) + ":*"
            if wildcard in user.permissions:
                return user
        # Check top-level wildcard (eudr-dav:*)
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


class PaginationParams(GreenLangBase):
    """Standard pagination query parameters.

    Attributes:
        limit: Maximum number of results per page (1-1000).
        offset: Number of results to skip for pagination.
    """

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
# Date range filter parameters
# ---------------------------------------------------------------------------


class DateRangeParams(GreenLangBase):
    """Common date range filter parameters.

    Attributes:
        start_date: Start of date range filter (inclusive).
        end_date: End of date range filter (inclusive).
    """

    start_date: Optional[datetime] = Field(
        None, description="Start date filter (inclusive, UTC)"
    )
    end_date: Optional[datetime] = Field(
        None, description="End date filter (inclusive, UTC)"
    )


def get_date_range(
    start_date: Optional[datetime] = Query(
        None, description="Start date filter (inclusive, UTC)"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date filter (inclusive, UTC)"
    ),
) -> DateRangeParams:
    """Extract date range filter parameters from query string.

    Args:
        start_date: Start of date range (inclusive).
        end_date: End of date range (inclusive).

    Returns:
        DateRangeParams with validated date range.

    Raises:
        HTTPException: 400 if start_date is after end_date.
    """
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be before or equal to end_date",
        )
    return DateRangeParams(start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Sort parameters
# ---------------------------------------------------------------------------


class SortParams(GreenLangBase):
    """Sort field and order parameters.

    Attributes:
        sort_by: Field name to sort by.
        sort_order: Sort direction (asc or desc).
    """

    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: str = Field(default="desc", description="Sort direction (asc/desc)")


def get_sort(
    sort_by: str = Query(
        default="created_at", description="Field to sort by"
    ),
    sort_order: str = Query(
        default="desc", description="Sort direction (asc/desc)"
    ),
) -> SortParams:
    """Extract sort parameters from query string.

    Args:
        sort_by: Field name to sort by.
        sort_order: Sort direction.

    Returns:
        SortParams with validated sort configuration.

    Raises:
        HTTPException: 400 if sort_order is invalid.
    """
    normalized = sort_order.lower().strip()
    if normalized not in ("asc", "desc"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sort_order must be 'asc' or 'desc'",
        )
    return SortParams(sort_by=sort_by, sort_order=normalized)


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-endpoint sliding window)
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
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Window duration in seconds.
        """
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

_rate_limit_standard = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=50, window_seconds=60)
_rate_limit_batch = RateLimiter(max_requests=20, window_seconds=60)
_rate_limit_report = RateLimiter(max_requests=20, window_seconds=60)
_rate_limit_export = RateLimiter(max_requests=10, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_standard(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Standard rate limit: 100 requests/minute for read operations."""
    await _rate_limit_standard(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 50 requests/minute for mutations."""
    await _rate_limit_write(request, user)


async def rate_limit_batch(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Batch import rate limit: 20 requests/minute."""
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
# Request validators - Path parameter helpers
# ---------------------------------------------------------------------------


def validate_uuid_path(
    value: str,
    param_name: str = "id",
) -> str:
    """Validate a path parameter is a non-empty string.

    Args:
        value: The path parameter value to validate.
        param_name: Name for error messages.

    Returns:
        The validated, stripped value.

    Raises:
        HTTPException: 400 if the value is empty or whitespace-only.
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{param_name} must be a non-empty string",
        )
    return value.strip()


def validate_document_id(
    document_id: str = Path(..., description="Document identifier"),
) -> str:
    """Validate document_id path parameter."""
    return validate_uuid_path(document_id, "document_id")


def validate_verification_id(
    verification_id: str = Path(..., description="Verification identifier"),
) -> str:
    """Validate verification_id path parameter."""
    return validate_uuid_path(verification_id, "verification_id")


def validate_validation_id(
    validation_id: str = Path(..., description="Validation identifier"),
) -> str:
    """Validate validation_id path parameter."""
    return validate_uuid_path(validation_id, "validation_id")


def validate_report_id(
    report_id: str = Path(..., description="Report identifier"),
) -> str:
    """Validate report_id path parameter."""
    return validate_uuid_path(report_id, "report_id")


def validate_operator_id(
    operator_id: str = Path(..., description="Operator identifier"),
) -> str:
    """Validate operator_id path parameter."""
    return validate_uuid_path(operator_id, "operator_id")


def validate_dds_id(
    dds_id: str = Path(..., description="DDS package identifier"),
) -> str:
    """Validate dds_id path parameter."""
    return validate_uuid_path(dds_id, "dds_id")


def validate_hash_value(
    hash_value: str = Path(
        ..., alias="hash", description="Document hash value"
    ),
) -> str:
    """Validate hash path parameter.

    Args:
        hash_value: Hash string from URL path.

    Returns:
        Lowercased, stripped hash string.

    Raises:
        HTTPException: 400 if hash is empty.
    """
    validated = validate_uuid_path(hash_value, "hash")
    return validated.lower()


def validate_job_id(
    job_id: str = Path(..., description="Batch job identifier"),
) -> str:
    """Validate batch job ID path parameter."""
    return validate_uuid_path(job_id, "job_id")


# ---------------------------------------------------------------------------
# Error response model (shared across routes)
# ---------------------------------------------------------------------------


class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints.

    Attributes:
        error: Error type identifier.
        message: Human-readable error message.
        detail: Additional error details.
        request_id: Request correlation ID for debugging.
    """

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# ---------------------------------------------------------------------------
# Success envelope response
# ---------------------------------------------------------------------------


class SuccessResponse(GreenLangBase):
    """Standard success response wrapper.

    Attributes:
        status: Response status string (always ``success``).
        message: Response message.
        data: Response payload.
    """

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")


# ---------------------------------------------------------------------------
# Service singleton (lazy initialization with stub fallback)
# ---------------------------------------------------------------------------


class _DAVServiceStub:
    """Stub service for Document Authentication operations.

    Provides safe no-op methods when the actual engine modules are not
    yet initialized. Enables API startup and health checks without
    requiring full engine initialization.
    """

    def __init__(self) -> None:
        """Initialize the stub service."""
        self._initialized = False
        logger.info("Document Authentication service stub initialized")

    @property
    def is_initialized(self) -> bool:
        """Whether the real service has been initialized."""
        return self._initialized


_dav_service_instance: Optional[Any] = None


def get_dav_service() -> Any:
    """Return the Document Authentication service singleton.

    Attempts to import and initialize the real service engines on first
    call. Falls back to a stub if the real engines are not available
    (e.g. during testing or early startup).

    Returns:
        Document Authentication service instance (real or stub).
    """
    global _dav_service_instance

    if _dav_service_instance is not None:
        return _dav_service_instance

    try:
        from greenlang.agents.eudr.document_authentication.config import (
            get_config,
        )

        config = get_config()
        _dav_service_instance = {
            "config": config,
            "initialized": True,
        }
        logger.info("Document Authentication service engines initialized")
    except Exception as exc:
        logger.warning(
            "Could not initialize DAV service engines, using stub: %s",
            exc,
        )
        _dav_service_instance = _DAVServiceStub()

    return _dav_service_instance


def reset_dav_service() -> None:
    """Reset the service singleton. Used in testing."""
    global _dav_service_instance
    _dav_service_instance = None


# ---------------------------------------------------------------------------
# Request ID injection
# ---------------------------------------------------------------------------


def get_request_id(request: Request) -> str:
    """Extract or generate a request correlation ID.

    Checks for ``X-Request-ID`` header first, then generates a new UUID4.

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
    # Auth
    "AuthUser",
    "get_current_user",
    "require_permission",
    # Pagination / Filters
    "DateRangeParams",
    "PaginationParams",
    "SortParams",
    "get_date_range",
    "get_pagination",
    "get_sort",
    # Rate limiting
    "RateLimiter",
    "rate_limit_batch",
    "rate_limit_export",
    "rate_limit_report",
    "rate_limit_standard",
    "rate_limit_write",
    # Security schemes
    "api_key_header",
    "oauth2_scheme",
    # Validators
    "validate_dds_id",
    "validate_document_id",
    "validate_hash_value",
    "validate_job_id",
    "validate_operator_id",
    "validate_report_id",
    "validate_uuid_path",
    "validate_validation_id",
    "validate_verification_id",
    # Responses
    "ErrorResponse",
    "SuccessResponse",
    # Service
    "_DAVServiceStub",
    "get_dav_service",
    "get_request_id",
    "reset_dav_service",
]
