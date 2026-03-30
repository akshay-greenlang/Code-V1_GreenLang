# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-006 Plot Boundary Manager

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and engine singletons. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_config: Returns the PlotBoundaryConfig singleton.
    - get_boundary_service: Returns the PlotBoundaryService facade singleton.
    - get_boundary_versioner: Returns the BoundaryVersioner engine singleton.
    - get_simplification_engine: Returns the SimplificationEngine singleton.
    - get_split_merge_engine: Returns the SplitMergeEngine singleton.
    - get_compliance_reporter: Returns the ComplianceReporter engine singleton.
    - get_overlap_detector: Returns the OverlapDetector engine singleton.
    - get_area_calculator: Returns the AreaCalculator engine singleton.
    - get_geometry_validator: Returns the GeometryValidator engine singleton.
    - get_export_engine: Returns the ExportEngine singleton.
    - validate_plot_id: Validates plot_id format (UUID).
    - validate_geometry_input: Ensures at least one geometry format is provided.
    - validate_commodity: Validates EUDR commodity string.
    - validate_country_iso: Validates ISO 3166-1 alpha-2 country code.
    - validate_date_param: Parses ISO 8601 date strings.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - rate_limit_standard: 100 requests/minute.
    - rate_limit_batch: 10 requests/minute.
    - rate_limit_export: 20 requests/minute.
    - rate_limit_scan: 5 requests/minute.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from datetime import date as date_cls
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import Field
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EUDR commodity and country constants
# ---------------------------------------------------------------------------

EUDR_COMMODITIES = frozenset({
    "palm_oil", "cocoa", "coffee", "soya", "rubber", "cattle", "wood",
})

# ISO 3166-1 alpha-2 pattern
_ISO_ALPHA2_PATTERN = re.compile(r"^[A-Z]{2}$")

# UUID v4 pattern (lowercase hex with dashes)
_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)

# General plot_id pattern: UUIDs, prefixed IDs, or alphanumeric with dashes
_PLOT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


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

    Uses wildcard matching: ``eudr-boundary:*`` grants all
    ``eudr-boundary:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-boundary:boundaries:write``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/boundaries")
        ... async def create_boundary(
        ...     user: AuthUser = Depends(require_permission("eudr-boundary:boundaries:write"))
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


# Pre-configured rate limiter instances
_rate_limit_standard = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=30, window_seconds=60)
_rate_limit_batch = RateLimiter(max_requests=10, window_seconds=60)
_rate_limit_export = RateLimiter(max_requests=20, window_seconds=60)
_rate_limit_scan = RateLimiter(max_requests=5, window_seconds=60)
_rate_limit_heavy = RateLimiter(max_requests=10, window_seconds=60)


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


async def rate_limit_batch(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Batch rate limit: 10 requests/minute."""
    await _rate_limit_batch(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 20 requests/minute."""
    await _rate_limit_export(request, user)


async def rate_limit_scan(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Scan rate limit: 5 requests/minute."""
    await _rate_limit_scan(request, user)


async def rate_limit_heavy(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Heavy operation rate limit: 10 requests/minute."""
    await _rate_limit_heavy(request, user)


# ---------------------------------------------------------------------------
# Input validation dependencies
# ---------------------------------------------------------------------------


def validate_plot_id(plot_id: str) -> str:
    """Validate a plot_id path parameter.

    Accepts UUIDs (with or without dashes) and alphanumeric identifiers
    up to 128 characters.

    Args:
        plot_id: Plot identifier from the URL path.

    Returns:
        Validated plot_id string.

    Raises:
        HTTPException: 400 if plot_id format is invalid.
    """
    if not plot_id or not plot_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="plot_id must not be empty",
        )

    plot_id = plot_id.strip()

    if len(plot_id) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"plot_id must be <= 128 characters, got {len(plot_id)}",
        )

    if not _PLOT_ID_PATTERN.match(plot_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "plot_id must contain only alphanumeric characters, "
                "hyphens, and underscores"
            ),
        )

    return plot_id


def validate_commodity(commodity: str) -> str:
    """Validate a commodity string is one of the 7 EUDR commodities.

    Args:
        commodity: Commodity name to validate.

    Returns:
        Normalized (lowercase) commodity string.

    Raises:
        HTTPException: 400 if commodity is not EUDR-regulated.
    """
    if not commodity or not commodity.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="commodity must not be empty",
        )

    normalized = commodity.strip().lower()
    if normalized not in EUDR_COMMODITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"commodity must be one of {sorted(EUDR_COMMODITIES)}, "
                f"got '{commodity}'"
            ),
        )

    return normalized


def validate_country_iso(country_iso: str) -> str:
    """Validate an ISO 3166-1 alpha-2 country code.

    Args:
        country_iso: Two-letter country code to validate.

    Returns:
        Normalized (uppercase) country code.

    Raises:
        HTTPException: 400 if country code format is invalid.
    """
    if not country_iso or not country_iso.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="country_iso must not be empty",
        )

    normalized = country_iso.strip().upper()
    if not _ISO_ALPHA2_PATTERN.match(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code, "
                f"got '{country_iso}'"
            ),
        )

    return normalized


def validate_date_param(date_str: str) -> datetime:
    """Parse and validate an ISO 8601 date string.

    Accepts date-only (YYYY-MM-DD) and datetime (YYYY-MM-DDTHH:MM:SSZ)
    formats.

    Args:
        date_str: ISO 8601 date string to parse.

    Returns:
        Parsed datetime object.

    Raises:
        HTTPException: 400 if date string format is invalid.
    """
    if not date_str or not date_str.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="date parameter must not be empty",
        )

    date_str = date_str.strip()

    # Try datetime formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try date-only format
    try:
        parsed = date_cls.fromisoformat(date_str[:10])
        return datetime(parsed.year, parsed.month, parsed.day)
    except (ValueError, IndexError):
        pass

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"date must be a valid ISO 8601 string "
            f"(YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ), got '{date_str}'"
        ),
    )


def validate_geometry_input(
    geometry: Optional[Any] = None,
    wkt: Optional[str] = None,
    kml: Optional[str] = None,
    plot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate that at least one geometry input is provided.

    Args:
        geometry: GeoJSON geometry object.
        wkt: WKT geometry string.
        kml: KML geometry string.
        plot_id: Existing plot_id reference.

    Returns:
        Dictionary with the provided input type and value.

    Raises:
        HTTPException: 400 if no geometry input is provided.
    """
    if geometry is not None:
        return {"type": "geojson", "value": geometry}
    if wkt is not None and wkt.strip():
        return {"type": "wkt", "value": wkt.strip()}
    if kml is not None and kml.strip():
        return {"type": "kml", "value": kml.strip()}
    if plot_id is not None and plot_id.strip():
        return {"type": "plot_id", "value": plot_id.strip()}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="At least one geometry input required: geometry (GeoJSON), wkt, kml, or plot_id",
    )


# ---------------------------------------------------------------------------
# Engine singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_boundary_service = None
_boundary_versioner = None
_simplification_engine = None
_split_merge_engine = None
_compliance_reporter = None
_overlap_detector = None
_area_calculator = None
_geometry_validator = None
_export_engine = None


def get_config():
    """Return the PlotBoundaryConfig singleton.

    Lazily loads the configuration from environment variables on first call.

    Returns:
        PlotBoundaryConfig instance.
    """
    from greenlang.agents.eudr.plot_boundary.config import get_config as _get_cfg

    return _get_cfg()


def get_boundary_service():
    """Return the PlotBoundaryService facade singleton.

    Lazily imports and instantiates the high-level boundary management
    service facade that coordinates all engines.

    Returns:
        PlotBoundaryService instance.
    """
    global _boundary_service
    if _boundary_service is None:
        try:
            from greenlang.agents.eudr.plot_boundary.service import (
                PlotBoundaryService,
            )

            cfg = get_config()
            _boundary_service = PlotBoundaryService(config=cfg)
            logger.info("PlotBoundaryService singleton initialized")
        except ImportError:
            logger.warning(
                "PlotBoundaryService not available; "
                "using stub implementation for development"
            )
            _boundary_service = _StubService()
    return _boundary_service


def get_boundary_versioner():
    """Return the BoundaryVersioner engine singleton.

    Lazily imports and instantiates the immutable boundary version
    management engine.

    Returns:
        BoundaryVersioner instance.
    """
    global _boundary_versioner
    if _boundary_versioner is None:
        try:
            from greenlang.agents.eudr.plot_boundary.boundary_versioner import (
                BoundaryVersioner,
            )

            cfg = get_config()
            _boundary_versioner = BoundaryVersioner(config=cfg)
            logger.info("BoundaryVersioner singleton initialized")
        except ImportError:
            logger.warning(
                "BoundaryVersioner not available; "
                "using stub implementation"
            )
            _boundary_versioner = _StubEngine("BoundaryVersioner")
    return _boundary_versioner


def get_simplification_engine():
    """Return the SimplificationEngine singleton.

    Lazily imports and instantiates the polygon simplification and
    generalization engine.

    Returns:
        SimplificationEngine instance.
    """
    global _simplification_engine
    if _simplification_engine is None:
        try:
            from greenlang.agents.eudr.plot_boundary.simplification_engine import (
                SimplificationEngine,
            )

            cfg = get_config()
            _simplification_engine = SimplificationEngine(config=cfg)
            logger.info("SimplificationEngine singleton initialized")
        except ImportError:
            logger.warning(
                "SimplificationEngine not available; "
                "using stub implementation"
            )
            _simplification_engine = _StubEngine("SimplificationEngine")
    return _simplification_engine


def get_split_merge_engine():
    """Return the SplitMergeEngine singleton.

    Lazily imports and instantiates the split/merge engine with
    genealogy tracking capabilities.

    Returns:
        SplitMergeEngine instance.
    """
    global _split_merge_engine
    if _split_merge_engine is None:
        try:
            from greenlang.agents.eudr.plot_boundary.split_merge_engine import (
                SplitMergeEngine,
            )

            cfg = get_config()
            _split_merge_engine = SplitMergeEngine(config=cfg)
            logger.info("SplitMergeEngine singleton initialized")
        except ImportError:
            logger.warning(
                "SplitMergeEngine not available; "
                "using stub implementation"
            )
            _split_merge_engine = _StubEngine("SplitMergeEngine")
    return _split_merge_engine


def get_compliance_reporter():
    """Return the ComplianceReporter engine singleton.

    Lazily imports and instantiates the multi-format export and
    compliance reporting engine.

    Returns:
        ComplianceReporter instance.
    """
    global _compliance_reporter
    if _compliance_reporter is None:
        try:
            from greenlang.agents.eudr.plot_boundary.compliance_reporter import (
                ComplianceReporter,
            )

            cfg = get_config()
            _compliance_reporter = ComplianceReporter(config=cfg)
            logger.info("ComplianceReporter singleton initialized")
        except ImportError:
            logger.warning(
                "ComplianceReporter not available; "
                "using stub implementation"
            )
            _compliance_reporter = _StubEngine("ComplianceReporter")
    return _compliance_reporter


def get_overlap_detector():
    """Return the OverlapDetector engine singleton.

    Lazily imports and instantiates the spatial overlap detection engine
    with R-tree indexing.

    Returns:
        OverlapDetector instance.
    """
    global _overlap_detector
    if _overlap_detector is None:
        try:
            from greenlang.agents.eudr.plot_boundary.overlap_detector import (
                OverlapDetector,
            )

            cfg = get_config()
            _overlap_detector = OverlapDetector(config=cfg)
            logger.info("OverlapDetector singleton initialized")
        except ImportError:
            logger.warning(
                "OverlapDetector not available; "
                "using stub implementation"
            )
            _overlap_detector = _StubEngine("OverlapDetector")
    return _overlap_detector


def get_area_calculator():
    """Return the AreaCalculator engine singleton.

    Lazily imports and instantiates the geodetic area calculation engine
    using the Karney algorithm on the WGS84 ellipsoid.

    Returns:
        AreaCalculator instance.
    """
    global _area_calculator
    if _area_calculator is None:
        try:
            from greenlang.agents.eudr.plot_boundary.area_calculator import (
                AreaCalculator,
            )

            cfg = get_config()
            _area_calculator = AreaCalculator(config=cfg)
            logger.info("AreaCalculator singleton initialized")
        except ImportError:
            logger.warning(
                "AreaCalculator not available; "
                "using stub implementation"
            )
            _area_calculator = _StubEngine("AreaCalculator")
    return _area_calculator


def get_geometry_validator():
    """Return the GeometryValidator engine singleton.

    Lazily imports and instantiates the geometry topology validation
    engine with auto-repair capabilities.

    Returns:
        GeometryValidator instance.
    """
    global _geometry_validator
    if _geometry_validator is None:
        try:
            from greenlang.agents.eudr.plot_boundary.geometry_validator import (
                GeometryValidator,
            )

            cfg = get_config()
            _geometry_validator = GeometryValidator(config=cfg)
            logger.info("GeometryValidator singleton initialized")
        except ImportError:
            logger.warning(
                "GeometryValidator not available; "
                "using stub implementation"
            )
            _geometry_validator = _StubEngine("GeometryValidator")
    return _geometry_validator


def get_export_engine():
    """Return the ExportEngine singleton.

    Lazily imports and instantiates the multi-format boundary export
    engine supporting GeoJSON, KML, Shapefile, EUDR XML, and more.

    Returns:
        ExportEngine instance.
    """
    global _export_engine
    if _export_engine is None:
        try:
            from greenlang.agents.eudr.plot_boundary.export_engine import (
                ExportEngine,
            )

            cfg = get_config()
            _export_engine = ExportEngine(config=cfg)
            logger.info("ExportEngine singleton initialized")
        except ImportError:
            logger.warning(
                "ExportEngine not available; "
                "using stub implementation"
            )
            _export_engine = _StubEngine("ExportEngine")
    return _export_engine


# ---------------------------------------------------------------------------
# Stub implementations for development
# ---------------------------------------------------------------------------


class _StubEngine:
    """Stub engine for development when real engines are not yet available."""

    def __init__(self, name: str = "StubEngine"):
        self.name = name
        logger.debug("Stub engine created: %s", name)


class _StubService:
    """Stub service facade for development when PlotBoundaryService is not available."""

    def __init__(self):
        logger.debug("StubService created for PlotBoundaryService")


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
    "EUDR_COMMODITIES",
    "PaginationParams",
    "RateLimiter",
    "SuccessResponse",
    "api_key_header",
    "get_area_calculator",
    "get_boundary_service",
    "get_boundary_versioner",
    "get_compliance_reporter",
    "get_config",
    "get_current_user",
    "get_export_engine",
    "get_geometry_validator",
    "get_overlap_detector",
    "get_pagination",
    "get_simplification_engine",
    "get_split_merge_engine",
    "oauth2_scheme",
    "rate_limit_batch",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_scan",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_commodity",
    "validate_country_iso",
    "validate_date_param",
    "validate_geometry_input",
    "validate_plot_id",
]
