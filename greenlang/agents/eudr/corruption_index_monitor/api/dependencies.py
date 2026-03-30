# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-019 Corruption Index Monitor

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_cim_config: Returns the CorruptionIndexMonitorConfig singleton.
    - get_cpi_engine: Returns the CPIMonitorEngine singleton.
    - get_wgi_engine: Returns the WGIAnalyzerEngine singleton.
    - get_bribery_engine: Returns the BriberyRiskEngine singleton.
    - get_institutional_engine: Returns the InstitutionalQualityEngine singleton.
    - get_trend_engine: Returns the TrendAnalysisEngine singleton.
    - get_correlation_engine: Returns the DeforestationCorrelationEngine singleton.
    - get_alert_engine: Returns the AlertEngine singleton.
    - get_compliance_engine: Returns the ComplianceImpactEngine singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_country_code: Validates country code path/query parameter.
    - validate_year_range: Validates start/end year query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
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

    Uses wildcard matching: ``eudr-corruption-index:*`` grants all
    ``eudr-corruption-index:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-corruption-index:cpi:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/cpi/{country_code}/score")
        ... async def get_cpi_score(
        ...     user: AuthUser = Depends(require_permission("eudr-corruption-index:cpi:read"))
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

# Valid ISO 3166-1 alpha-2 pattern (basic validation)
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


def validate_year_range(
    start_year: Optional[int] = Query(
        None,
        ge=1995,
        le=2030,
        description="Start year for year range filter",
    ),
    end_year: Optional[int] = Query(
        None,
        ge=1995,
        le=2030,
        description="End year for year range filter",
    ),
) -> Dict[str, Optional[int]]:
    """Validate that start_year <= end_year when both provided.

    Args:
        start_year: Optional start year.
        end_year: Optional end year.

    Returns:
        Dictionary with 'start_year' and 'end_year' keys.

    Raises:
        HTTPException: 400 if start_year > end_year.
    """
    if start_year and end_year and start_year > end_year:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"start_year ({start_year}) must be <= end_year ({end_year})"
            ),
        )
    return {"start_year": start_year, "end_year": end_year}


# ---------------------------------------------------------------------------
# Engine singleton accessors
# ---------------------------------------------------------------------------

# Global singletons (lazily initialized)
_cim_config: Optional[Any] = None
_cpi_engine: Optional[Any] = None
_wgi_engine: Optional[Any] = None
_bribery_engine: Optional[Any] = None
_institutional_engine: Optional[Any] = None
_trend_engine: Optional[Any] = None
_correlation_engine: Optional[Any] = None
_alert_engine: Optional[Any] = None
_compliance_engine: Optional[Any] = None


def get_cim_config() -> Any:
    """Return the CorruptionIndexMonitorConfig singleton.

    Lazily initializes the configuration on first access using
    the ``get_config()`` factory from the config module.

    Returns:
        CorruptionIndexMonitorConfig instance.

    Raises:
        HTTPException: 503 if configuration cannot be loaded.
    """
    global _cim_config
    if _cim_config is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.config import (
                get_config,
            )

            _cim_config = get_config()
            logger.info("CorruptionIndexMonitorConfig initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize CorruptionIndexMonitorConfig: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Corruption Index Monitor configuration unavailable",
            )
    return _cim_config


def get_cpi_engine() -> Any:
    """Return the CPIMonitorEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CPIMonitorEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _cpi_engine
    if _cpi_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.cpi_monitor_engine import (
                CPIMonitorEngine,
            )

            config = get_cim_config()
            _cpi_engine = CPIMonitorEngine(config)
            logger.info("CPIMonitorEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CPIMonitorEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CPIMonitorEngine unavailable",
            )
    return _cpi_engine


def get_wgi_engine() -> Any:
    """Return the WGIAnalyzerEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        WGIAnalyzerEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _wgi_engine
    if _wgi_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.wgi_analyzer_engine import (
                WGIAnalyzerEngine,
            )

            config = get_cim_config()
            _wgi_engine = WGIAnalyzerEngine(config)
            logger.info("WGIAnalyzerEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize WGIAnalyzerEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WGIAnalyzerEngine unavailable",
            )
    return _wgi_engine


def get_bribery_engine() -> Any:
    """Return the BriberyRiskEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        BriberyRiskEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _bribery_engine
    if _bribery_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.bribery_risk_engine import (
                BriberyRiskEngine,
            )

            config = get_cim_config()
            _bribery_engine = BriberyRiskEngine(config)
            logger.info("BriberyRiskEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize BriberyRiskEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="BriberyRiskEngine unavailable",
            )
    return _bribery_engine


def get_institutional_engine() -> Any:
    """Return the InstitutionalQualityEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        InstitutionalQualityEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _institutional_engine
    if _institutional_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.institutional_quality_engine import (
                InstitutionalQualityEngine,
            )

            config = get_cim_config()
            _institutional_engine = InstitutionalQualityEngine(config)
            logger.info("InstitutionalQualityEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize InstitutionalQualityEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InstitutionalQualityEngine unavailable",
            )
    return _institutional_engine


def get_trend_engine() -> Any:
    """Return the TrendAnalysisEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        TrendAnalysisEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _trend_engine
    if _trend_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.trend_analysis_engine import (
                TrendAnalysisEngine,
            )

            config = get_cim_config()
            _trend_engine = TrendAnalysisEngine(config)
            logger.info("TrendAnalysisEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize TrendAnalysisEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TrendAnalysisEngine unavailable",
            )
    return _trend_engine


def get_correlation_engine() -> Any:
    """Return the DeforestationCorrelationEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        DeforestationCorrelationEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _correlation_engine
    if _correlation_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.deforestation_correlation_engine import (
                DeforestationCorrelationEngine,
            )

            config = get_cim_config()
            _correlation_engine = DeforestationCorrelationEngine(config)
            logger.info("DeforestationCorrelationEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize DeforestationCorrelationEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DeforestationCorrelationEngine unavailable",
            )
    return _correlation_engine


def get_alert_engine() -> Any:
    """Return the AlertEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AlertEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _alert_engine
    if _alert_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.alert_engine import (
                AlertEngine,
            )

            config = get_cim_config()
            _alert_engine = AlertEngine(config)
            logger.info("AlertEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AlertEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AlertEngine unavailable",
            )
    return _alert_engine


def get_compliance_engine() -> Any:
    """Return the ComplianceImpactEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        ComplianceImpactEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _compliance_engine
    if _compliance_engine is None:
        try:
            from greenlang.agents.eudr.corruption_index_monitor.engines.compliance_impact_engine import (
                ComplianceImpactEngine,
            )

            config = get_cim_config()
            _compliance_engine = ComplianceImpactEngine(config)
            logger.info("ComplianceImpactEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ComplianceImpactEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ComplianceImpactEngine unavailable",
            )
    return _compliance_engine


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
    "get_alert_engine",
    "get_bribery_engine",
    "get_cim_config",
    "get_compliance_engine",
    "get_correlation_engine",
    "get_cpi_engine",
    "get_current_user",
    "get_institutional_engine",
    "get_pagination",
    "get_trend_engine",
    "get_wgi_engine",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_country_code",
    "validate_year_range",
]
