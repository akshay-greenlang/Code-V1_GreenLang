# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-020 Deforestation Alert System

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_das_config: Returns the DeforestationAlertSystemConfig singleton.
    - get_satellite_detector: Returns the SatelliteChangeDetector singleton.
    - get_alert_generator: Returns the AlertGenerator singleton.
    - get_severity_classifier: Returns the SeverityClassifier singleton.
    - get_buffer_monitor: Returns the SpatialBufferMonitor singleton.
    - get_cutoff_verifier: Returns the CutoffDateVerifier singleton.
    - get_baseline_engine: Returns the HistoricalBaselineEngine singleton.
    - get_workflow_engine: Returns the AlertWorkflowEngine singleton.
    - get_compliance_assessor: Returns the ComplianceImpactAssessor singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_country_code: Validates country code path/query parameter.
    - validate_date_range: Validates start/end date query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

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

    Uses wildcard matching: ``eudr-deforestation-alert:*`` grants all
    ``eudr-deforestation-alert:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-deforestation-alert:satellite:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/satellite/sources")
        ... async def get_sources(
        ...     user: AuthUser = Depends(require_permission("eudr-deforestation-alert:satellite:read"))
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
    """Heavy operation rate limit: 10 requests/minute."""
    await _rate_limit_heavy(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 5 requests/minute."""
    await _rate_limit_export(request, user)


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
# Common query parameter validators
# ---------------------------------------------------------------------------

_COUNTRY_CODE_PATTERN = r"^[A-Z]{2}$"


def validate_country_code(
    country_code: str,
) -> str:
    """Validate and normalize a country code parameter.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Normalized (uppercase) country code.

    Raises:
        HTTPException: 400 if country code is invalid.
    """
    normalized = country_code.strip().upper()
    if len(normalized) != 2 or not normalized.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid country code: {country_code}. "
                f"Must be a two-letter ISO 3166-1 alpha-2 code."
            ),
        )
    return normalized


def validate_date_range(
    start_date: Optional[date] = Query(
        None,
        description="Start date for date range filter (YYYY-MM-DD)",
    ),
    end_date: Optional[date] = Query(
        None,
        description="End date for date range filter (YYYY-MM-DD)",
    ),
) -> Dict[str, Optional[date]]:
    """Validate that start_date <= end_date when both provided.

    Args:
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        Dictionary with 'start_date' and 'end_date' keys.

    Raises:
        HTTPException: 400 if start_date > end_date.
    """
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            ),
        )
    return {"start_date": start_date, "end_date": end_date}


# ---------------------------------------------------------------------------
# Engine singleton accessors
# ---------------------------------------------------------------------------

# Global singletons (lazily initialized)
_das_config: Optional[Any] = None
_satellite_detector: Optional[Any] = None
_alert_generator: Optional[Any] = None
_severity_classifier: Optional[Any] = None
_buffer_monitor: Optional[Any] = None
_cutoff_verifier: Optional[Any] = None
_baseline_engine: Optional[Any] = None
_workflow_engine: Optional[Any] = None
_compliance_assessor: Optional[Any] = None


def get_das_config() -> Any:
    """Return the DeforestationAlertSystemConfig singleton.

    Lazily initializes the configuration on first access using
    the ``get_config()`` factory from the config module.

    Returns:
        DeforestationAlertSystemConfig instance.

    Raises:
        HTTPException: 503 if configuration cannot be loaded.
    """
    global _das_config
    if _das_config is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system.config import (
                get_config,
            )

            _das_config = get_config()
            logger.info("DeforestationAlertSystemConfig initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize DeforestationAlertSystemConfig: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Deforestation Alert System configuration unavailable",
            )
    return _das_config


def get_satellite_detector() -> Any:
    """Return the SatelliteChangeDetector singleton.

    Lazily initializes the engine on first access.

    Returns:
        SatelliteChangeDetector instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _satellite_detector
    if _satellite_detector is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                SatelliteChangeDetector,
            )

            config = get_das_config()
            _satellite_detector = SatelliteChangeDetector(config)
            logger.info("SatelliteChangeDetector initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize SatelliteChangeDetector: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SatelliteChangeDetector unavailable",
            )
    return _satellite_detector


def get_alert_generator() -> Any:
    """Return the AlertGenerator singleton.

    Lazily initializes the engine on first access.

    Returns:
        AlertGenerator instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _alert_generator
    if _alert_generator is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                AlertGenerator,
            )

            config = get_das_config()
            _alert_generator = AlertGenerator(config)
            logger.info("AlertGenerator initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AlertGenerator: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AlertGenerator unavailable",
            )
    return _alert_generator


def get_severity_classifier() -> Any:
    """Return the SeverityClassifier singleton.

    Lazily initializes the engine on first access.

    Returns:
        SeverityClassifier instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _severity_classifier
    if _severity_classifier is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                SeverityClassifier,
            )

            config = get_das_config()
            _severity_classifier = SeverityClassifier(config)
            logger.info("SeverityClassifier initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize SeverityClassifier: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SeverityClassifier unavailable",
            )
    return _severity_classifier


def get_buffer_monitor() -> Any:
    """Return the SpatialBufferMonitor singleton.

    Lazily initializes the engine on first access.

    Returns:
        SpatialBufferMonitor instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _buffer_monitor
    if _buffer_monitor is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                SpatialBufferMonitor,
            )

            config = get_das_config()
            _buffer_monitor = SpatialBufferMonitor(config)
            logger.info("SpatialBufferMonitor initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize SpatialBufferMonitor: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SpatialBufferMonitor unavailable",
            )
    return _buffer_monitor


def get_cutoff_verifier() -> Any:
    """Return the CutoffDateVerifier singleton.

    Lazily initializes the engine on first access.

    Returns:
        CutoffDateVerifier instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _cutoff_verifier
    if _cutoff_verifier is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                CutoffDateVerifier,
            )

            config = get_das_config()
            _cutoff_verifier = CutoffDateVerifier(config)
            logger.info("CutoffDateVerifier initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CutoffDateVerifier: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CutoffDateVerifier unavailable",
            )
    return _cutoff_verifier


def get_baseline_engine() -> Any:
    """Return the HistoricalBaselineEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        HistoricalBaselineEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _baseline_engine
    if _baseline_engine is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                HistoricalBaselineEngine,
            )

            config = get_das_config()
            _baseline_engine = HistoricalBaselineEngine(config)
            logger.info("HistoricalBaselineEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize HistoricalBaselineEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HistoricalBaselineEngine unavailable",
            )
    return _baseline_engine


def get_workflow_engine() -> Any:
    """Return the AlertWorkflowEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AlertWorkflowEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _workflow_engine
    if _workflow_engine is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                AlertWorkflowEngine,
            )

            config = get_das_config()
            _workflow_engine = AlertWorkflowEngine(config)
            logger.info("AlertWorkflowEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AlertWorkflowEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AlertWorkflowEngine unavailable",
            )
    return _workflow_engine


def get_compliance_assessor() -> Any:
    """Return the ComplianceImpactAssessor singleton.

    Lazily initializes the engine on first access.

    Returns:
        ComplianceImpactAssessor instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _compliance_assessor
    if _compliance_assessor is None:
        try:
            from greenlang.agents.eudr.deforestation_alert_system import (
                ComplianceImpactAssessor,
            )

            config = get_das_config()
            _compliance_assessor = ComplianceImpactAssessor(config)
            logger.info("ComplianceImpactAssessor initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ComplianceImpactAssessor: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ComplianceImpactAssessor unavailable",
            )
    return _compliance_assessor


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
    "get_baseline_engine",
    "get_buffer_monitor",
    "get_compliance_assessor",
    "get_current_user",
    "get_cutoff_verifier",
    "get_das_config",
    "get_pagination",
    "get_satellite_detector",
    "get_severity_classifier",
    "get_workflow_engine",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_country_code",
    "validate_date_range",
]
