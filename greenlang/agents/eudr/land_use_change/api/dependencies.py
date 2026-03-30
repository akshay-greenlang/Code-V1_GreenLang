# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-005 Land Use Change Detector

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and engine singletons. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_land_use_service: Returns the LandUseChangeService facade singleton.
    - get_classifier_engine: Returns the LandUseClassifier singleton.
    - get_transition_engine: Returns the TransitionDetector singleton.
    - get_trajectory_engine: Returns the TemporalTrajectoryAnalyzer singleton.
    - get_cutoff_engine: Returns the CutoffDateVerifier singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_coordinates: Coordinate validation dependency.
    - validate_date_range: Date range validation dependency.
    - validate_plot_id: Plot ID format validator.
    - validate_polygon_wkt: WKT polygon geometry validator.
    - get_request_id: Unique request ID generator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Query, Request, status
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
    """Authenticated user context extracted from JWT or API key."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(default="", description="User email address")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    operator_id: str = Field(
        default="", description="Associated operator ID"
    )
    roles: List[str] = Field(
        default_factory=list, description="Assigned roles"
    )
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

    # Fallback: validate API key
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
        detail=(
            "Authentication required. Provide Bearer token or "
            "X-API-Key header."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# RBAC permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(permission: str) -> Callable:
    """Factory returning a FastAPI dependency that checks RBAC permissions.

    Uses wildcard matching: ``eudr-luc:*`` grants all
    ``eudr-luc:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-luc:classification:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/classify")
        ... async def classify(
        ...     user: AuthUser = Depends(
        ...         require_permission("eudr-luc:classification:write")
        ...     )
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


class PaginationParams(GreenLangBase):
    """Standard pagination query parameters."""

    limit: int = Field(
        default=50, ge=1, le=1000, description="Results per page"
    )
    offset: int = Field(
        default=0, ge=0, description="Number of results to skip"
    )


def get_pagination(
    limit: int = Query(
        default=50, ge=1, le=1000, description="Results per page"
    ),
    offset: int = Query(
        default=0, ge=0, description="Number of results to skip"
    ),
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
# Sort parameters
# ---------------------------------------------------------------------------


class SortParams(GreenLangBase):
    """Standard sort query parameters."""

    sort_by: str = Field(
        default="created_at", description="Field to sort by"
    )
    sort_order: str = Field(
        default="desc", description="Sort order: asc or desc"
    )


def get_sort(
    sort_by: str = Query(
        default="created_at",
        description="Field to sort by",
    ),
    sort_order: str = Query(
        default="desc",
        description="Sort order: asc or desc",
    ),
) -> SortParams:
    """Extract sort parameters from query string.

    Args:
        sort_by: Field name to sort results by.
        sort_order: Sort direction (asc or desc).

    Returns:
        SortParams with validated sort_by and sort_order.

    Raises:
        HTTPException: 400 if sort_order is invalid.
    """
    sort_order = sort_order.lower().strip()
    if sort_order not in ("asc", "desc"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"sort_order must be 'asc' or 'desc', got '{sort_order}'"
            ),
        )
    return SortParams(sort_by=sort_by, sort_order=sort_order)


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

    def __init__(
        self, max_requests: int = 100, window_seconds: int = 60
    ):
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
                "Rate limit exceeded: user=%s endpoint=%s",
                user.user_id,
                request.url.path,
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


# Pre-configured rate limiter instances
_rate_limit_standard = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=30, window_seconds=60)
_rate_limit_heavy = RateLimiter(max_requests=10, window_seconds=60)
_rate_limit_export = RateLimiter(max_requests=5, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_standard(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Standard rate limit: 100 requests/minute."""
    await _rate_limit_standard(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 30 requests/minute."""
    await _rate_limit_write(request, user)


async def rate_limit_heavy(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Heavy operation rate limit: 10 requests/minute (batch analysis)."""
    await _rate_limit_heavy(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 5 requests/minute (report generation)."""
    await _rate_limit_export(request, user)


# ---------------------------------------------------------------------------
# Coordinate Validator
# ---------------------------------------------------------------------------


def validate_coordinates(
    latitude: float, longitude: float
) -> Tuple[float, float]:
    """Validate geographic coordinates are within valid ranges.

    Args:
        latitude: Latitude in decimal degrees (-90 to 90).
        longitude: Longitude in decimal degrees (-180 to 180).

    Returns:
        Tuple of (latitude, longitude).

    Raises:
        HTTPException: 400 if coordinates are out of range.
    """
    if not (-90.0 <= latitude <= 90.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"latitude must be between -90 and 90, got {latitude}"
            ),
        )
    if not (-180.0 <= longitude <= 180.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"longitude must be between -180 and 180, got {longitude}"
            ),
        )
    return latitude, longitude


# ---------------------------------------------------------------------------
# Date Range Validator
# ---------------------------------------------------------------------------


def validate_date_range(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Tuple[Optional[date], Optional[date]]:
    """Validate and parse a date range.

    Ensures date_to is not before date_from and neither date is in
    the future. Returns both dates unchanged if valid.

    Args:
        date_from: Range start date (optional).
        date_to: Range end date (optional).

    Returns:
        Tuple of (date_from, date_to) dates.

    Raises:
        HTTPException: 400 if date range is invalid.
    """
    if date_from and date_to and date_to < date_from:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"date_to ({date_to}) must be on or after "
                f"date_from ({date_from})"
            ),
        )

    today = date.today()
    if date_from and date_from > today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"date_from ({date_from}) must not be in the future"
            ),
        )
    if date_to and date_to > today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"date_to ({date_to}) must not be in the future"
            ),
        )

    return date_from, date_to


# ---------------------------------------------------------------------------
# WKT Polygon Validator
# ---------------------------------------------------------------------------


_WKT_POLYGON_RE = re.compile(
    r"^POLYGON\s*\(\s*\(.*\)\s*\)$",
    re.IGNORECASE | re.DOTALL,
)


def validate_polygon_wkt(wkt: str) -> str:
    """Validate that a WKT string is a valid POLYGON geometry.

    Checks basic structure and ensures the string matches WKT POLYGON
    format. Does not perform full topological validation.

    Args:
        wkt: WKT string to validate.

    Returns:
        Cleaned WKT string.

    Raises:
        HTTPException: 400 if the WKT is invalid.
    """
    wkt = wkt.strip()
    if not wkt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="polygon_wkt must not be empty",
        )
    if not _WKT_POLYGON_RE.match(wkt):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "polygon_wkt must be a valid WKT POLYGON, e.g. "
                "'POLYGON((-1.624 6.688, -1.623 6.689, "
                "-1.622 6.687, -1.624 6.688))'"
            ),
        )
    return wkt


# ---------------------------------------------------------------------------
# Plot ID Validator
# ---------------------------------------------------------------------------


def validate_plot_id(plot_id: str) -> str:
    """Validate plot ID format.

    Ensures the plot ID is non-empty, trimmed, and within the 200
    character limit.

    Args:
        plot_id: Plot identifier to validate.

    Returns:
        Cleaned plot ID string.

    Raises:
        HTTPException: 400 if the plot ID is invalid.
    """
    plot_id = plot_id.strip()
    if not plot_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plot_id must not be empty",
        )
    if len(plot_id) > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plot_id must not exceed 200 characters",
        )
    return plot_id


# ---------------------------------------------------------------------------
# Request ID Generator
# ---------------------------------------------------------------------------


def get_request_id() -> str:
    """Generate a unique request ID for correlation.

    Returns:
        Request ID string in format ``req-<16 hex chars>``.
    """
    return f"req-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Engine singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_classifier_engine = None
_transition_engine = None
_trajectory_engine = None
_cutoff_engine = None
_land_use_service = None


def get_classifier_engine():
    """Return the LandUseClassifier singleton.

    Lazily imports and instantiates the land use classifier engine
    using the global configuration. The singleton is cached for the
    lifetime of the process.

    Returns:
        LandUseClassifier instance configured from get_config().
    """
    global _classifier_engine
    if _classifier_engine is None:
        from greenlang.agents.eudr.land_use_change.land_use_classifier import (
            LandUseClassifier,
        )
        from greenlang.agents.eudr.land_use_change.config import get_config

        cfg = get_config()
        _classifier_engine = LandUseClassifier(config=cfg)
        logger.info("LandUseClassifier singleton initialized")
    return _classifier_engine


def get_transition_engine():
    """Return the TransitionDetector singleton.

    Lazily imports and instantiates the transition detection engine
    for detecting land use transitions between two time points.

    Returns:
        TransitionDetector instance configured from get_config().
    """
    global _transition_engine
    if _transition_engine is None:
        from greenlang.agents.eudr.land_use_change.transition_detector import (
            TransitionDetector,
        )
        from greenlang.agents.eudr.land_use_change.config import get_config

        cfg = get_config()
        _transition_engine = TransitionDetector(config=cfg)
        logger.info("TransitionDetector singleton initialized")
    return _transition_engine


def get_trajectory_engine():
    """Return the TemporalTrajectoryAnalyzer singleton.

    Lazily imports and instantiates the temporal trajectory analyzer
    for analysing multi-temporal land use change patterns.

    Returns:
        TemporalTrajectoryAnalyzer instance configured from get_config().
    """
    global _trajectory_engine
    if _trajectory_engine is None:
        from greenlang.agents.eudr.land_use_change.temporal_trajectory_analyzer import (
            TemporalTrajectoryAnalyzer,
        )
        from greenlang.agents.eudr.land_use_change.config import get_config

        cfg = get_config()
        _trajectory_engine = TemporalTrajectoryAnalyzer(config=cfg)
        logger.info("TemporalTrajectoryAnalyzer singleton initialized")
    return _trajectory_engine


def get_cutoff_engine():
    """Return the CutoffDateVerifier singleton.

    Lazily imports and instantiates the EUDR cutoff date verification
    engine that coordinates all analysis engines for compliance verdicts.

    Returns:
        CutoffDateVerifier instance configured from get_config().
    """
    global _cutoff_engine
    if _cutoff_engine is None:
        from greenlang.agents.eudr.land_use_change.cutoff_date_verifier import (
            CutoffDateVerifier,
        )
        from greenlang.agents.eudr.land_use_change.config import get_config

        cfg = get_config()
        _cutoff_engine = CutoffDateVerifier(config=cfg)
        logger.info("CutoffDateVerifier singleton initialized")
    return _cutoff_engine


def get_land_use_service():
    """Return the LandUseChangeService facade singleton.

    Lazily imports and instantiates the high-level service facade
    that coordinates all land use change analysis engines. This
    is the primary entry point for route handlers.

    Returns:
        LandUseChangeService instance configured from get_config().
    """
    global _land_use_service
    if _land_use_service is None:
        from greenlang.agents.eudr.land_use_change.service import (
            LandUseChangeService,
        )
        from greenlang.agents.eudr.land_use_change.config import get_config

        cfg = get_config()
        _land_use_service = LandUseChangeService(config=cfg)
        logger.info("LandUseChangeService singleton initialized")
    return _land_use_service


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(
        None, description="Additional error details"
    )
    request_id: Optional[str] = Field(
        None, description="Request correlation ID"
    )


# ---------------------------------------------------------------------------
# Success envelope response
# ---------------------------------------------------------------------------


class SuccessResponse(GreenLangBase):
    """Standard success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "ErrorResponse",
    "PaginationParams",
    "RateLimiter",
    "SortParams",
    "SuccessResponse",
    "api_key_header",
    "get_classifier_engine",
    "get_current_user",
    "get_cutoff_engine",
    "get_land_use_service",
    "get_pagination",
    "get_request_id",
    "get_sort",
    "get_trajectory_engine",
    "get_transition_engine",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_coordinates",
    "validate_date_range",
    "validate_plot_id",
    "validate_polygon_wkt",
]
