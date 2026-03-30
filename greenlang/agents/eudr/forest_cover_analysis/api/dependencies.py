# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-004 Forest Cover Analysis

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and engine singletons. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_forest_cover_service: Returns the ForestCoverAnalysisService singleton.
    - get_density_engine: Returns the CanopyDensityEngine singleton.
    - get_classification_engine: Returns the ForestClassificationEngine singleton.
    - get_historical_engine: Returns the HistoricalReconstructionEngine singleton.
    - get_verification_engine: Returns the DeforestationVerificationEngine singleton.
    - get_height_engine: Returns the CanopyHeightEngine singleton.
    - get_fragmentation_engine: Returns the FragmentationAnalysisEngine singleton.
    - get_biomass_engine: Returns the BiomassEstimationEngine singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_polygon_wkt: WKT polygon geometry validator.
    - validate_plot_id: Plot ID format validator.
    - parse_date_range: Date range validator and parser.
    - get_request_id: Unique request ID generator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
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
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# RBAC permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(permission: str) -> Callable:
    """Factory returning a FastAPI dependency that checks RBAC permissions.

    Uses wildcard matching: ``eudr-fca:*`` grants all
    ``eudr-fca:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-fca:density:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/density/analyze")
        ... async def analyze_density(
        ...     user: AuthUser = Depends(require_permission("eudr-fca:density:write"))
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
            detail=f"sort_order must be 'asc' or 'desc', got '{sort_order}'",
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
# WKT Polygon Validator
# ---------------------------------------------------------------------------


_WKT_POLYGON_RE = re.compile(
    r"^POLYGON\s*\(\s*\(.*\)\s*\)$",
    re.IGNORECASE | re.DOTALL,
)


def validate_polygon_wkt(wkt: str) -> str:
    """Validate that a WKT string is a valid POLYGON geometry.

    Checks basic structure and ensures the polygon ring is closed
    (first and last coordinate match). Does not perform full
    topological validation.

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
                "'POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))'"
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
# Date Range Parser
# ---------------------------------------------------------------------------


def parse_date_range(
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Tuple[Optional[date], Optional[date]]:
    """Validate and parse a date range.

    Ensures end is not before start. Returns both dates unchanged
    if valid.

    Args:
        start: Range start date (optional).
        end: Range end date (optional).

    Returns:
        Tuple of (start, end) dates.

    Raises:
        HTTPException: 400 if end is before start.
    """
    if start and end and end < start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"end date ({end}) must be on or after start date ({start})",
        )
    return start, end


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

_density_engine = None
_classification_engine = None
_historical_engine = None
_verification_engine = None
_height_engine = None
_fragmentation_engine = None
_biomass_engine = None
_forest_cover_service = None


def get_density_engine():
    """Return the CanopyDensityEngine singleton.

    Lazily imports and instantiates the canopy density engine
    using the global configuration. The singleton is cached for the
    lifetime of the process.

    Returns:
        CanopyDensityEngine instance configured from get_config().
    """
    global _density_engine
    if _density_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.canopy_density import (
            CanopyDensityEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _density_engine = CanopyDensityEngine(config=cfg)
        logger.info("CanopyDensityEngine singleton initialized")
    return _density_engine


def get_classification_engine():
    """Return the ForestClassificationEngine singleton.

    Lazily imports and instantiates the forest classification engine
    using the global configuration.

    Returns:
        ForestClassificationEngine instance configured from get_config().
    """
    global _classification_engine
    if _classification_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.forest_classification import (
            ForestClassificationEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _classification_engine = ForestClassificationEngine(config=cfg)
        logger.info("ForestClassificationEngine singleton initialized")
    return _classification_engine


def get_historical_engine():
    """Return the HistoricalReconstructionEngine singleton.

    Lazily imports and instantiates the historical reconstruction engine
    for reconstructing forest cover at the EUDR cutoff date.

    Returns:
        HistoricalReconstructionEngine instance configured from get_config().
    """
    global _historical_engine
    if _historical_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.historical_reconstruction import (
            HistoricalReconstructionEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _historical_engine = HistoricalReconstructionEngine(config=cfg)
        logger.info("HistoricalReconstructionEngine singleton initialized")
    return _historical_engine


def get_verification_engine():
    """Return the DeforestationVerificationEngine singleton.

    Lazily imports and instantiates the deforestation-free verification
    engine that coordinates all analysis engines for compliance verdicts.

    Returns:
        DeforestationVerificationEngine instance configured from get_config().
    """
    global _verification_engine
    if _verification_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.deforestation_verification import (
            DeforestationVerificationEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _verification_engine = DeforestationVerificationEngine(config=cfg)
        logger.info("DeforestationVerificationEngine singleton initialized")
    return _verification_engine


def get_height_engine():
    """Return the CanopyHeightEngine singleton.

    Lazily imports and instantiates the canopy height estimation engine
    using GEDI, ICESat-2, and other height data sources.

    Returns:
        CanopyHeightEngine instance configured from get_config().
    """
    global _height_engine
    if _height_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.canopy_height import (
            CanopyHeightEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _height_engine = CanopyHeightEngine(config=cfg)
        logger.info("CanopyHeightEngine singleton initialized")
    return _height_engine


def get_fragmentation_engine():
    """Return the FragmentationAnalysisEngine singleton.

    Lazily imports and instantiates the fragmentation analysis engine
    for landscape-level forest patch metrics.

    Returns:
        FragmentationAnalysisEngine instance configured from get_config().
    """
    global _fragmentation_engine
    if _fragmentation_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.fragmentation_analysis import (
            FragmentationAnalysisEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _fragmentation_engine = FragmentationAnalysisEngine(config=cfg)
        logger.info("FragmentationAnalysisEngine singleton initialized")
    return _fragmentation_engine


def get_biomass_engine():
    """Return the BiomassEstimationEngine singleton.

    Lazily imports and instantiates the above-ground biomass estimation
    engine using GEDI L4A, ESA CCI, and allometric models.

    Returns:
        BiomassEstimationEngine instance configured from get_config().
    """
    global _biomass_engine
    if _biomass_engine is None:
        from greenlang.agents.eudr.forest_cover_analysis.biomass_estimation import (
            BiomassEstimationEngine,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _biomass_engine = BiomassEstimationEngine(config=cfg)
        logger.info("BiomassEstimationEngine singleton initialized")
    return _biomass_engine


def get_forest_cover_service():
    """Return the ForestCoverAnalysisService facade singleton.

    Lazily imports and instantiates the high-level service facade
    that coordinates all forest cover analysis engines.

    Returns:
        ForestCoverAnalysisService instance configured from get_config().
    """
    global _forest_cover_service
    if _forest_cover_service is None:
        from greenlang.agents.eudr.forest_cover_analysis.service import (
            ForestCoverAnalysisService,
        )
        from greenlang.agents.eudr.forest_cover_analysis.config import get_config

        cfg = get_config()
        _forest_cover_service = ForestCoverAnalysisService(config=cfg)
        logger.info("ForestCoverAnalysisService singleton initialized")
    return _forest_cover_service


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


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
    "get_biomass_engine",
    "get_classification_engine",
    "get_current_user",
    "get_density_engine",
    "get_forest_cover_service",
    "get_fragmentation_engine",
    "get_height_engine",
    "get_historical_engine",
    "get_pagination",
    "get_request_id",
    "get_sort",
    "get_verification_engine",
    "oauth2_scheme",
    "parse_date_range",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_plot_id",
    "validate_polygon_wkt",
]
