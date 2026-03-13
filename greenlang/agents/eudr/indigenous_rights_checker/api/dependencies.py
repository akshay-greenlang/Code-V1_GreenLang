# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-021 Indigenous Rights Checker

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_irc_config: Returns the IndigenousRightsCheckerConfig singleton.
    - get_territory_manager: Returns the TerritoryManager engine singleton.
    - get_fpic_verifier: Returns the FPICVerifier engine singleton.
    - get_overlap_analyzer: Returns the OverlapAnalyzer engine singleton.
    - get_consultation_tracker: Returns the ConsultationTracker engine singleton.
    - get_violation_detector: Returns the ViolationDetector engine singleton.
    - get_community_registry: Returns the CommunityRegistry engine singleton.
    - get_compliance_reporter: Returns the ComplianceReporter engine singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_country_code: Validates country code path/query parameter.
    - validate_date_range: Validates start/end date query parameters.

Auth & RBAC Permissions (14 total):
    eudr-irc:territories:read
    eudr-irc:territories:create
    eudr-irc:territories:update
    eudr-irc:territories:delete
    eudr-irc:fpic:read
    eudr-irc:fpic:verify
    eudr-irc:overlap:read
    eudr-irc:overlap:analyze
    eudr-irc:consultations:read
    eudr-irc:consultations:create
    eudr-irc:violations:read
    eudr-irc:violations:detect
    eudr-irc:violations:resolve
    eudr-irc:compliance:read

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
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

    Uses wildcard matching: ``eudr-irc:*`` grants all
    ``eudr-irc:<resource>:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-irc:territories:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/territories")
        ... async def list_territories(
        ...     user: AuthUser = Depends(require_permission("eudr-irc:territories:read"))
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
_irc_config: Optional[Any] = None
_territory_manager: Optional[Any] = None
_fpic_verifier: Optional[Any] = None
_overlap_analyzer: Optional[Any] = None
_consultation_tracker: Optional[Any] = None
_violation_detector: Optional[Any] = None
_community_registry: Optional[Any] = None
_compliance_reporter: Optional[Any] = None


def get_irc_config() -> Any:
    """Return the IndigenousRightsCheckerConfig singleton.

    Lazily initializes the configuration on first access using
    the ``get_config()`` factory from the config module.

    Returns:
        IndigenousRightsCheckerConfig instance.

    Raises:
        HTTPException: 503 if configuration cannot be loaded.
    """
    global _irc_config
    if _irc_config is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker.config import (
                get_config,
            )

            _irc_config = get_config()
            logger.info("IndigenousRightsCheckerConfig initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize IndigenousRightsCheckerConfig: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Indigenous Rights Checker configuration unavailable",
            )
    return _irc_config


def get_territory_manager() -> Any:
    """Return the TerritoryManager engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        TerritoryManager instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _territory_manager
    if _territory_manager is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                TerritoryManager,
            )

            config = get_irc_config()
            _territory_manager = TerritoryManager(config)
            logger.info("TerritoryManager initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize TerritoryManager: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TerritoryManager unavailable",
            )
    return _territory_manager


def get_fpic_verifier() -> Any:
    """Return the FPICVerifier engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        FPICVerifier instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _fpic_verifier
    if _fpic_verifier is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                FPICVerifier,
            )

            config = get_irc_config()
            _fpic_verifier = FPICVerifier(config)
            logger.info("FPICVerifier initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize FPICVerifier: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FPICVerifier unavailable",
            )
    return _fpic_verifier


def get_overlap_analyzer() -> Any:
    """Return the OverlapAnalyzer engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        OverlapAnalyzer instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _overlap_analyzer
    if _overlap_analyzer is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                OverlapAnalyzer,
            )

            config = get_irc_config()
            _overlap_analyzer = OverlapAnalyzer(config)
            logger.info("OverlapAnalyzer initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize OverlapAnalyzer: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OverlapAnalyzer unavailable",
            )
    return _overlap_analyzer


def get_consultation_tracker() -> Any:
    """Return the ConsultationTracker engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        ConsultationTracker instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _consultation_tracker
    if _consultation_tracker is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                ConsultationTracker,
            )

            config = get_irc_config()
            _consultation_tracker = ConsultationTracker(config)
            logger.info("ConsultationTracker initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ConsultationTracker: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ConsultationTracker unavailable",
            )
    return _consultation_tracker


def get_violation_detector() -> Any:
    """Return the ViolationDetector engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        ViolationDetector instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _violation_detector
    if _violation_detector is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                ViolationDetector,
            )

            config = get_irc_config()
            _violation_detector = ViolationDetector(config)
            logger.info("ViolationDetector initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ViolationDetector: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ViolationDetector unavailable",
            )
    return _violation_detector


def get_community_registry() -> Any:
    """Return the CommunityRegistry engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CommunityRegistry instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _community_registry
    if _community_registry is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                CommunityRegistry,
            )

            config = get_irc_config()
            _community_registry = CommunityRegistry(config)
            logger.info("CommunityRegistry initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CommunityRegistry: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CommunityRegistry unavailable",
            )
    return _community_registry


def get_compliance_reporter() -> Any:
    """Return the ComplianceReporter engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        ComplianceReporter instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _compliance_reporter
    if _compliance_reporter is None:
        try:
            from greenlang.agents.eudr.indigenous_rights_checker import (
                ComplianceReporter,
            )

            config = get_irc_config()
            _compliance_reporter = ComplianceReporter(config)
            logger.info("ComplianceReporter initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ComplianceReporter: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ComplianceReporter unavailable",
            )
    return _compliance_reporter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "PaginationParams",
    "RateLimiter",
    "api_key_header",
    "get_community_registry",
    "get_compliance_reporter",
    "get_consultation_tracker",
    "get_current_user",
    "get_fpic_verifier",
    "get_irc_config",
    "get_overlap_analyzer",
    "get_pagination",
    "get_territory_manager",
    "get_violation_detector",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_country_code",
    "validate_date_range",
]
