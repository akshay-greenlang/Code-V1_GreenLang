# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-003 Satellite Monitoring

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and engine singletons. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_imagery_engine: Returns the ImageryAcquisitionEngine singleton.
    - get_spectral_calculator: Returns the SpectralIndexCalculator singleton.
    - get_baseline_manager: Returns the BaselineManager singleton.
    - get_change_detector: Returns the ForestChangeDetector singleton.
    - get_fusion_engine: Returns the multi-source fusion engine singleton.
    - get_cloud_filler: Returns the cloud gap filler engine singleton.
    - get_continuous_monitor: Returns the ContinuousMonitor singleton.
    - get_alert_generator: Returns the AlertGenerator singleton.
    - get_satellite_service: Returns the SatelliteMonitoringService facade.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
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

    Uses wildcard matching: ``eudr-satellite:*`` grants all
    ``eudr-satellite:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-satellite:imagery:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/imagery/search")
        ... async def search_scenes(
        ...     user: AuthUser = Depends(require_permission("eudr-satellite:imagery:read"))
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
    """Heavy operation rate limit: 10 requests/minute (satellite imagery downloads)."""
    await _rate_limit_heavy(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 5 requests/minute (evidence package generation)."""
    await _rate_limit_export(request, user)


# ---------------------------------------------------------------------------
# Engine singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_imagery_engine = None
_spectral_calculator = None
_baseline_manager = None
_change_detector = None
_fusion_engine = None
_cloud_filler = None
_continuous_monitor = None
_alert_generator = None
_satellite_service = None


def get_imagery_engine():
    """Return the ImageryAcquisitionEngine singleton.

    Lazily imports and instantiates the imagery acquisition engine
    using the global configuration. The singleton is cached for the
    lifetime of the process.

    Returns:
        ImageryAcquisitionEngine instance configured from get_config().
    """
    global _imagery_engine
    if _imagery_engine is None:
        from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
            ImageryAcquisitionEngine,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _imagery_engine = ImageryAcquisitionEngine(config=cfg)
        logger.info("ImageryAcquisitionEngine singleton initialized")
    return _imagery_engine


def get_spectral_calculator():
    """Return the SpectralIndexCalculator singleton.

    Lazily imports and instantiates the spectral index calculator
    using the global configuration.

    Returns:
        SpectralIndexCalculator instance configured from get_config().
    """
    global _spectral_calculator
    if _spectral_calculator is None:
        from greenlang.agents.eudr.satellite_monitoring.spectral_index_calculator import (
            SpectralIndexCalculator,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _spectral_calculator = SpectralIndexCalculator(config=cfg)
        logger.info("SpectralIndexCalculator singleton initialized")
    return _spectral_calculator


def get_baseline_manager():
    """Return the BaselineManager singleton.

    Lazily imports and instantiates the baseline management engine
    using the global configuration.

    Returns:
        BaselineManager instance configured from get_config().
    """
    global _baseline_manager
    if _baseline_manager is None:
        from greenlang.agents.eudr.satellite_monitoring.baseline_manager import (
            BaselineManager,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _baseline_manager = BaselineManager(config=cfg)
        logger.info("BaselineManager singleton initialized")
    return _baseline_manager


def get_change_detector():
    """Return the ForestChangeDetector singleton.

    Lazily imports and instantiates the forest change detection engine
    using the global configuration.

    Returns:
        ForestChangeDetector instance configured from get_config().
    """
    global _change_detector
    if _change_detector is None:
        from greenlang.agents.eudr.satellite_monitoring.forest_change_detector import (
            ForestChangeDetector,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _change_detector = ForestChangeDetector(config=cfg)
        logger.info("ForestChangeDetector singleton initialized")
    return _change_detector


def get_fusion_engine():
    """Return the multi-source FusionEngine singleton.

    Lazily imports and instantiates the multi-source data fusion engine
    using the global configuration and fusion weights.

    Returns:
        FusionEngine instance configured from get_config().
    """
    global _fusion_engine
    if _fusion_engine is None:
        from greenlang.agents.eudr.satellite_monitoring.fusion_engine import (
            FusionEngine,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _fusion_engine = FusionEngine(config=cfg)
        logger.info("FusionEngine singleton initialized")
    return _fusion_engine


def get_cloud_filler():
    """Return the CloudGapFiller singleton.

    Lazily imports and instantiates the cloud gap filler engine for
    handling cloudy imagery using temporal compositing and SAR data.

    Returns:
        CloudGapFiller instance configured from get_config().
    """
    global _cloud_filler
    if _cloud_filler is None:
        from greenlang.agents.eudr.satellite_monitoring.cloud_gap_filler import (
            CloudGapFiller,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _cloud_filler = CloudGapFiller(config=cfg)
        logger.info("CloudGapFiller singleton initialized")
    return _cloud_filler


def get_continuous_monitor():
    """Return the ContinuousMonitor singleton.

    Lazily imports and instantiates the continuous monitoring engine
    that manages scheduled monitoring executions and alert generation.

    Returns:
        ContinuousMonitor instance configured from get_config().
    """
    global _continuous_monitor
    if _continuous_monitor is None:
        from greenlang.agents.eudr.satellite_monitoring.continuous_monitor import (
            ContinuousMonitor,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _continuous_monitor = ContinuousMonitor(config=cfg)
        logger.info("ContinuousMonitor singleton initialized")
    return _continuous_monitor


def get_alert_generator():
    """Return the AlertGenerator singleton.

    Lazily imports and instantiates the alert generation engine for
    creating deforestation and degradation alerts from monitoring results.

    Returns:
        AlertGenerator instance configured from get_config().
    """
    global _alert_generator
    if _alert_generator is None:
        from greenlang.agents.eudr.satellite_monitoring.alert_generator import (
            AlertGenerator,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _alert_generator = AlertGenerator(config=cfg)
        logger.info("AlertGenerator singleton initialized")
    return _alert_generator


def get_satellite_service():
    """Return the SatelliteMonitoringService facade singleton.

    Lazily imports and instantiates the high-level service facade
    that coordinates all satellite monitoring engines.

    Returns:
        SatelliteMonitoringService instance configured from get_config().
    """
    global _satellite_service
    if _satellite_service is None:
        from greenlang.agents.eudr.satellite_monitoring.service import (
            SatelliteMonitoringService,
        )
        from greenlang.agents.eudr.satellite_monitoring.config import get_config

        cfg = get_config()
        _satellite_service = SatelliteMonitoringService(config=cfg)
        logger.info("SatelliteMonitoringService singleton initialized")
    return _satellite_service


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
    "SuccessResponse",
    "api_key_header",
    "get_alert_generator",
    "get_baseline_manager",
    "get_change_detector",
    "get_cloud_filler",
    "get_continuous_monitor",
    "get_current_user",
    "get_fusion_engine",
    "get_imagery_engine",
    "get_pagination",
    "get_satellite_service",
    "get_spectral_calculator",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
]
