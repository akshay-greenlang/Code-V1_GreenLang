# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-015 Mobile Data Collector

FastAPI dependency injection providers for authentication, authorization,
rate limiting, request validation, pagination, file upload validation,
and service access. All route handlers inject these dependencies to
enforce JWT auth (SEC-001), RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_mdc_service: Returns the Mobile Data Collector service singleton.
    - get_current_operator_id: Extracts operator ID from auth context.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - DateRangeParams: Common date range filter parameters.
    - Validators: Request-level validation helpers for path parameters.

Rate Limiter Tiers (5):
    - read: 200 requests/minute (GET operations)
    - write: 100 requests/minute (POST/PUT operations)
    - upload: 50 requests/minute (photo/file uploads)
    - sync: 30 requests/minute (sync operations)
    - admin: 20 requests/minute (admin/fleet operations)

File Size Limits:
    - Photo upload: 10 MB per photo
    - Package export: 50 MB per package

Permission Prefix: eudr-mdc:*

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Header, Path, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum photo file size in bytes (10 MB).
MAX_PHOTO_SIZE_BYTES: int = 10 * 1024 * 1024

#: Maximum package export size in bytes (50 MB).
MAX_PACKAGE_SIZE_BYTES: int = 50 * 1024 * 1024

#: Allowed photo content types.
ALLOWED_PHOTO_CONTENT_TYPES: frozenset = frozenset({
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/heif",
})

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
    """Authenticated user context extracted from JWT or API key.

    Attributes:
        user_id: Unique user identifier from JWT ``sub`` claim.
        email: User email address.
        tenant_id: Multi-tenant identifier (defaults to ``default``).
        operator_id: Associated EUDR operator identifier.
        roles: Assigned RBAC roles (e.g. ``admin``, ``operator``).
        permissions: Granted fine-grained permissions (e.g. ``eudr-mdc:forms:create``).
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
# Operator ID extraction
# ---------------------------------------------------------------------------


async def get_current_operator_id(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> str:
    """Extract the current operator ID from auth context.

    Args:
        request: FastAPI request object.
        user: Authenticated user.

    Returns:
        Operator identifier string.

    Raises:
        HTTPException: 403 if no operator_id is associated.
    """
    if user.operator_id:
        return user.operator_id

    # Try request header fallback
    operator_id = request.headers.get("X-Operator-ID", "")
    if operator_id:
        return operator_id.strip()

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="No operator ID associated with this user. "
               "Set X-Operator-ID header or configure operator mapping.",
    )


# ---------------------------------------------------------------------------
# RBAC permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(permission: str) -> Callable:
    """Factory returning a FastAPI dependency that checks RBAC permissions.

    Uses wildcard matching: ``eudr-mdc:*`` grants all
    ``eudr-mdc:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-mdc:forms:create``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/forms")
        ... async def submit_form(
        ...     user: AuthUser = Depends(require_permission("eudr-mdc:forms:create"))
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

        # Check wildcard patterns (eudr-mdc:forms:* matches eudr-mdc:forms:create)
        parts = permission.split(":")
        for i in range(len(parts)):
            wildcard = ":".join(parts[: i + 1]) + ":*"
            if wildcard in user.permissions:
                return user
        # Check top-level wildcard (eudr-mdc:*)
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
    """Standard pagination query parameters.

    Attributes:
        page: Page number (1-based).
        page_size: Number of results per page (1-500).
    """

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=1, le=500, description="Results per page"
    )


def get_pagination(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=500, description="Results per page"
    ),
) -> PaginationParams:
    """Extract pagination parameters from query string.

    Args:
        page: Page number (1-based, default 1).
        page_size: Results per page (1-500, default 50).

    Returns:
        PaginationParams with validated page and page_size.
    """
    return PaginationParams(page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# Date range filter parameters
# ---------------------------------------------------------------------------


class DateRangeParams(BaseModel):
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
# Pre-configured rate limiter instances (5 tiers)
# ---------------------------------------------------------------------------

_rate_limit_read = RateLimiter(max_requests=200, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_upload = RateLimiter(max_requests=50, window_seconds=60)
_rate_limit_sync = RateLimiter(max_requests=30, window_seconds=60)
_rate_limit_admin = RateLimiter(max_requests=20, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_read(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Read rate limit: 200 requests/minute for GET operations."""
    await _rate_limit_read(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 100 requests/minute for POST/PUT operations."""
    await _rate_limit_write(request, user)


async def rate_limit_upload(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Upload rate limit: 50 requests/minute for photo/file uploads."""
    await _rate_limit_upload(request, user)


async def rate_limit_sync(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Sync rate limit: 30 requests/minute for sync operations."""
    await _rate_limit_sync(request, user)


async def rate_limit_admin(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Admin rate limit: 20 requests/minute for admin/fleet operations."""
    await _rate_limit_admin(request, user)


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


def validate_form_id(
    form_id: str = Path(..., description="Form submission identifier"),
) -> str:
    """Validate form_id path parameter."""
    return validate_uuid_path(form_id, "form_id")


def validate_capture_id(
    capture_id: str = Path(..., description="GPS capture identifier"),
) -> str:
    """Validate capture_id path parameter."""
    return validate_uuid_path(capture_id, "capture_id")


def validate_polygon_id(
    polygon_id: str = Path(..., description="Polygon trace identifier"),
) -> str:
    """Validate polygon_id path parameter."""
    return validate_uuid_path(polygon_id, "polygon_id")


def validate_photo_id(
    photo_id: str = Path(..., description="Photo evidence identifier"),
) -> str:
    """Validate photo_id path parameter."""
    return validate_uuid_path(photo_id, "photo_id")


def validate_conflict_id(
    conflict_id: str = Path(..., description="Sync conflict identifier"),
) -> str:
    """Validate conflict_id path parameter."""
    return validate_uuid_path(conflict_id, "conflict_id")


def validate_template_id(
    template_id: str = Path(..., description="Form template identifier"),
) -> str:
    """Validate template_id path parameter."""
    return validate_uuid_path(template_id, "template_id")


def validate_signature_id(
    signature_id: str = Path(..., description="Digital signature identifier"),
) -> str:
    """Validate signature_id path parameter."""
    return validate_uuid_path(signature_id, "signature_id")


def validate_package_id(
    package_id: str = Path(..., description="Data package identifier"),
) -> str:
    """Validate package_id path parameter."""
    return validate_uuid_path(package_id, "package_id")


def validate_device_id(
    device_id: str = Path(..., description="Device identifier"),
) -> str:
    """Validate device_id path parameter."""
    return validate_uuid_path(device_id, "device_id")


def validate_campaign_id(
    campaign_id: str = Path(..., description="Campaign identifier"),
) -> str:
    """Validate campaign_id path parameter."""
    return validate_uuid_path(campaign_id, "campaign_id")


# ---------------------------------------------------------------------------
# Device authentication header extraction
# ---------------------------------------------------------------------------


async def get_device_auth_header(
    x_device_id: Optional[str] = Header(
        None, description="Device identifier from device auth header"
    ),
    x_device_token: Optional[str] = Header(
        None, description="Device authentication token"
    ),
) -> Dict[str, Optional[str]]:
    """Extract device authentication headers.

    Args:
        x_device_id: Device identifier header.
        x_device_token: Device authentication token.

    Returns:
        Dictionary with device_id and device_token.
    """
    return {
        "device_id": x_device_id.strip() if x_device_id else None,
        "device_token": x_device_token.strip() if x_device_token else None,
    }


# ---------------------------------------------------------------------------
# File upload size validation
# ---------------------------------------------------------------------------


def validate_photo_file_size(file_size_bytes: int) -> int:
    """Validate photo file size is within the allowed limit.

    Args:
        file_size_bytes: Photo file size in bytes.

    Returns:
        Validated file size.

    Raises:
        HTTPException: 400 if file exceeds the 10 MB limit.
    """
    if file_size_bytes > MAX_PHOTO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Photo file size {file_size_bytes} bytes exceeds "
                f"maximum allowed size of {MAX_PHOTO_SIZE_BYTES} bytes "
                f"({MAX_PHOTO_SIZE_BYTES // (1024 * 1024)} MB)"
            ),
        )
    if file_size_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Photo file size must be greater than 0 bytes",
        )
    return file_size_bytes


def validate_package_file_size(file_size_bytes: int) -> int:
    """Validate package export file size is within the allowed limit.

    Args:
        file_size_bytes: Package file size in bytes.

    Returns:
        Validated file size.

    Raises:
        HTTPException: 400 if file exceeds the 50 MB limit.
    """
    if file_size_bytes > MAX_PACKAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Package file size {file_size_bytes} bytes exceeds "
                f"maximum allowed size of {MAX_PACKAGE_SIZE_BYTES} bytes "
                f"({MAX_PACKAGE_SIZE_BYTES // (1024 * 1024)} MB)"
            ),
        )
    return file_size_bytes


# ---------------------------------------------------------------------------
# Content type validation
# ---------------------------------------------------------------------------


def validate_photo_content_type(content_type: str) -> str:
    """Validate photo content type is an allowed image format.

    Args:
        content_type: MIME content type string.

    Returns:
        Validated, lowercased content type.

    Raises:
        HTTPException: 400 if content type is not an allowed photo format.
    """
    normalized = content_type.strip().lower()
    if normalized not in ALLOWED_PHOTO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported photo content type '{content_type}'. "
                f"Allowed types: {sorted(ALLOWED_PHOTO_CONTENT_TYPES)}"
            ),
        )
    return normalized


# ---------------------------------------------------------------------------
# Service singleton (lazy initialization with stub fallback)
# ---------------------------------------------------------------------------


class _MDCServiceStub:
    """Stub service for Mobile Data Collector operations.

    Provides safe no-op methods when the actual engine modules are not
    yet initialized. Enables API startup and health checks without
    requiring full engine initialization.
    """

    def __init__(self) -> None:
        """Initialize the stub service."""
        self._initialized = False
        logger.info("Mobile Data Collector service stub initialized")

    @property
    def is_initialized(self) -> bool:
        """Whether the real service has been initialized."""
        return self._initialized


_mdc_service_instance: Optional[Any] = None


def get_mdc_service() -> Any:
    """Return the Mobile Data Collector service singleton.

    Attempts to import and initialize the real service engines on first
    call. Falls back to a stub if the real engines are not available
    (e.g. during testing or early startup).

    Returns:
        Mobile Data Collector service instance (real or stub).
    """
    global _mdc_service_instance

    if _mdc_service_instance is not None:
        return _mdc_service_instance

    try:
        from greenlang.agents.eudr.mobile_data_collector.config import (
            get_config,
        )

        config = get_config()
        _mdc_service_instance = {
            "config": config,
            "initialized": True,
        }
        logger.info("Mobile Data Collector service engines initialized")
    except Exception as exc:
        logger.warning(
            "Could not initialize MDC service engines, using stub: %s",
            exc,
        )
        _mdc_service_instance = _MDCServiceStub()

    return _mdc_service_instance


def reset_mdc_service() -> None:
    """Reset the service singleton. Used in testing."""
    global _mdc_service_instance
    _mdc_service_instance = None


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
    "get_current_operator_id",
    "require_permission",
    # Pagination / Filters
    "DateRangeParams",
    "PaginationParams",
    "get_date_range",
    "get_pagination",
    # Rate limiting
    "RateLimiter",
    "rate_limit_admin",
    "rate_limit_read",
    "rate_limit_sync",
    "rate_limit_upload",
    "rate_limit_write",
    # Security schemes
    "api_key_header",
    "oauth2_scheme",
    # Validators
    "validate_campaign_id",
    "validate_capture_id",
    "validate_conflict_id",
    "validate_device_id",
    "validate_form_id",
    "validate_package_id",
    "validate_photo_id",
    "validate_polygon_id",
    "validate_signature_id",
    "validate_template_id",
    "validate_uuid_path",
    # File validation
    "validate_package_file_size",
    "validate_photo_content_type",
    "validate_photo_file_size",
    # Device auth
    "get_device_auth_header",
    # Constants
    "ALLOWED_PHOTO_CONTENT_TYPES",
    "MAX_PACKAGE_SIZE_BYTES",
    "MAX_PHOTO_SIZE_BYTES",
    # Service
    "_MDCServiceStub",
    "get_mdc_service",
    "get_request_id",
    "reset_mdc_service",
]
