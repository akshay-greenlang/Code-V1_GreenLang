# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-024 Third-Party Audit Manager

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_tam_config: Returns the ThirdPartyAuditManagerConfig singleton.
    - get_planning_engine: Returns the AuditPlanningSchedulingEngine singleton.
    - get_auditor_engine: Returns the AuditorRegistryQualificationEngine singleton.
    - get_execution_engine: Returns the AuditExecutionEngine singleton.
    - get_nc_engine: Returns the NonConformanceDetectionEngine singleton.
    - get_car_engine: Returns the CARManagementEngine singleton.
    - get_certification_engine: Returns the CertificationIntegrationEngine singleton.
    - get_reporting_engine: Returns the AuditReportingEngine singleton.
    - get_analytics_engine: Returns the AuditAnalyticsEngine singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date
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

    Uses wildcard matching: ``eudr-tam:*`` grants all
    ``eudr-tam:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-tam:audits:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/audits")
        ... async def list_audits(
        ...     user: AuthUser = Depends(require_permission("eudr-tam:audits:read"))
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
_rate_limit_evidence = RateLimiter(max_requests=20, window_seconds=60)
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


async def rate_limit_evidence(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Evidence upload rate limit: 20 requests/minute."""
    await _rate_limit_evidence(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 5 requests/minute."""
    await _rate_limit_export(request, user)


# ---------------------------------------------------------------------------
# Common query parameter validators
# ---------------------------------------------------------------------------


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
_tam_config: Optional[Any] = None
_planning_engine: Optional[Any] = None
_auditor_engine: Optional[Any] = None
_execution_engine: Optional[Any] = None
_nc_engine: Optional[Any] = None
_car_engine: Optional[Any] = None
_certification_engine: Optional[Any] = None
_reporting_engine: Optional[Any] = None
_analytics_engine: Optional[Any] = None


def get_tam_config() -> Any:
    """Return the ThirdPartyAuditManagerConfig singleton.

    Lazily initializes the configuration on first access using
    the ``get_config()`` factory from the config module.

    Returns:
        ThirdPartyAuditManagerConfig instance.

    Raises:
        HTTPException: 503 if configuration cannot be loaded.
    """
    global _tam_config
    if _tam_config is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager.config import (
                get_config,
            )

            _tam_config = get_config()
            logger.info("ThirdPartyAuditManagerConfig initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize ThirdPartyAuditManagerConfig: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Third-Party Audit Manager configuration unavailable",
            )
    return _tam_config


def get_planning_engine() -> Any:
    """Return the AuditPlanningSchedulingEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AuditPlanningSchedulingEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _planning_engine
    if _planning_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                AuditPlanningSchedulingEngine,
            )

            config = get_tam_config()
            _planning_engine = AuditPlanningSchedulingEngine(config)
            logger.info("AuditPlanningSchedulingEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AuditPlanningSchedulingEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AuditPlanningSchedulingEngine unavailable",
            )
    return _planning_engine


def get_auditor_engine() -> Any:
    """Return the AuditorRegistryQualificationEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AuditorRegistryQualificationEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _auditor_engine
    if _auditor_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                AuditorRegistryQualificationEngine,
            )

            config = get_tam_config()
            _auditor_engine = AuditorRegistryQualificationEngine(config)
            logger.info("AuditorRegistryQualificationEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AuditorRegistryQualificationEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AuditorRegistryQualificationEngine unavailable",
            )
    return _auditor_engine


def get_execution_engine() -> Any:
    """Return the AuditExecutionEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AuditExecutionEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _execution_engine
    if _execution_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                AuditExecutionEngine,
            )

            config = get_tam_config()
            _execution_engine = AuditExecutionEngine(config)
            logger.info("AuditExecutionEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AuditExecutionEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AuditExecutionEngine unavailable",
            )
    return _execution_engine


def get_nc_engine() -> Any:
    """Return the NonConformanceDetectionEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        NonConformanceDetectionEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _nc_engine
    if _nc_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                NonConformanceDetectionEngine,
            )

            config = get_tam_config()
            _nc_engine = NonConformanceDetectionEngine(config)
            logger.info("NonConformanceDetectionEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize NonConformanceDetectionEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NonConformanceDetectionEngine unavailable",
            )
    return _nc_engine


def get_car_engine() -> Any:
    """Return the CARManagementEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CARManagementEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _car_engine
    if _car_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                CARManagementEngine,
            )

            config = get_tam_config()
            _car_engine = CARManagementEngine(config)
            logger.info("CARManagementEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CARManagementEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CARManagementEngine unavailable",
            )
    return _car_engine


def get_certification_engine() -> Any:
    """Return the CertificationIntegrationEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CertificationIntegrationEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _certification_engine
    if _certification_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                CertificationIntegrationEngine,
            )

            config = get_tam_config()
            _certification_engine = CertificationIntegrationEngine(config)
            logger.info("CertificationIntegrationEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CertificationIntegrationEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CertificationIntegrationEngine unavailable",
            )
    return _certification_engine


def get_reporting_engine() -> Any:
    """Return the AuditReportingEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AuditReportingEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _reporting_engine
    if _reporting_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                AuditReportingEngine,
            )

            config = get_tam_config()
            _reporting_engine = AuditReportingEngine(config)
            logger.info("AuditReportingEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AuditReportingEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AuditReportingEngine unavailable",
            )
    return _reporting_engine


def get_analytics_engine() -> Any:
    """Return the AuditAnalyticsEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        AuditAnalyticsEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _analytics_engine
    if _analytics_engine is None:
        try:
            from greenlang.agents.eudr.third_party_audit_manager import (
                AuditAnalyticsEngine,
            )

            config = get_tam_config()
            _analytics_engine = AuditAnalyticsEngine(config)
            logger.info("AuditAnalyticsEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize AuditAnalyticsEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AuditAnalyticsEngine unavailable",
            )
    return _analytics_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "PaginationParams",
    "RateLimiter",
    "api_key_header",
    "get_analytics_engine",
    "get_auditor_engine",
    "get_car_engine",
    "get_certification_engine",
    "get_current_user",
    "get_execution_engine",
    "get_nc_engine",
    "get_pagination",
    "get_planning_engine",
    "get_reporting_engine",
    "get_tam_config",
    "oauth2_scheme",
    "rate_limit_evidence",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_country_code",
    "validate_date_range",
]
