# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-002 Geolocation Verification

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and engine singletons. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_coordinate_validator: Returns the CoordinateValidator engine singleton.
    - get_polygon_verifier: Returns the PolygonTopologyVerifier engine singleton.
    - get_protected_area_checker: Returns the ProtectedAreaChecker engine singleton.
    - get_deforestation_verifier: Returns the DeforestationCutoffVerifier engine singleton.
    - get_accuracy_scorer: Returns the AccuracyScoringEngine singleton.
    - get_temporal_analyzer: Returns the TemporalConsistencyAnalyzer singleton.
    - get_batch_pipeline: Returns the BatchVerificationPipeline singleton.
    - get_article9_reporter: Returns the Article9ComplianceReporter singleton.
    - get_verification_service: Returns the GeolocationVerificationService facade.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
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

    Uses wildcard matching: ``eudr-geolocation:*`` grants all
    ``eudr-geolocation:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-geolocation:coordinates:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/coordinates")
        ... async def validate_coordinates(
        ...     user: AuthUser = Depends(require_permission("eudr-geolocation:coordinates:write"))
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
# Engine singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_coordinator_validator = None
_polygon_verifier = None
_protected_area_checker = None
_deforestation_verifier = None
_accuracy_scorer = None
_temporal_analyzer = None
_batch_pipeline = None
_article9_reporter = None
_verification_service = None


def get_coordinate_validator():
    """Return the CoordinateValidator engine singleton.

    Lazily imports and instantiates the coordinate validation engine
    using the global configuration. The singleton is cached for the
    lifetime of the process.

    Returns:
        CoordinateValidator instance configured from get_config().
    """
    global _coordinator_validator
    if _coordinator_validator is None:
        from greenlang.agents.eudr.geolocation_verification.coordinate_validator import (
            CoordinateValidator,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _coordinator_validator = CoordinateValidator(config=cfg)
        logger.info("CoordinateValidator singleton initialized")
    return _coordinator_validator


def get_polygon_verifier():
    """Return the PolygonTopologyVerifier engine singleton.

    Lazily imports and instantiates the polygon topology verification
    engine using the global configuration.

    Returns:
        PolygonTopologyVerifier instance configured from get_config().
    """
    global _polygon_verifier
    if _polygon_verifier is None:
        from greenlang.agents.eudr.geolocation_verification.polygon_verifier import (
            PolygonTopologyVerifier,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _polygon_verifier = PolygonTopologyVerifier(config=cfg)
        logger.info("PolygonTopologyVerifier singleton initialized")
    return _polygon_verifier


def get_protected_area_checker():
    """Return the ProtectedAreaChecker engine singleton.

    Lazily imports and instantiates the protected area screening engine
    using the global configuration.

    Returns:
        ProtectedAreaChecker instance configured from get_config().
    """
    global _protected_area_checker
    if _protected_area_checker is None:
        from greenlang.agents.eudr.geolocation_verification.protected_area_checker import (
            ProtectedAreaChecker,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _protected_area_checker = ProtectedAreaChecker(config=cfg)
        logger.info("ProtectedAreaChecker singleton initialized")
    return _protected_area_checker


def get_deforestation_verifier():
    """Return the DeforestationCutoffVerifier engine singleton.

    Lazily imports and instantiates the deforestation cutoff verification
    engine using the global configuration.

    Returns:
        DeforestationCutoffVerifier instance configured from get_config().
    """
    global _deforestation_verifier
    if _deforestation_verifier is None:
        from greenlang.agents.eudr.geolocation_verification.deforestation_verifier import (
            DeforestationCutoffVerifier,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _deforestation_verifier = DeforestationCutoffVerifier(config=cfg)
        logger.info("DeforestationCutoffVerifier singleton initialized")
    return _deforestation_verifier


def get_accuracy_scorer():
    """Return the AccuracyScoringEngine singleton.

    Lazily imports and instantiates the accuracy scoring engine
    using the global configuration.

    Returns:
        AccuracyScoringEngine instance configured from get_config().
    """
    global _accuracy_scorer
    if _accuracy_scorer is None:
        from greenlang.agents.eudr.geolocation_verification.accuracy_scorer import (
            AccuracyScoringEngine,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _accuracy_scorer = AccuracyScoringEngine(config=cfg)
        logger.info("AccuracyScoringEngine singleton initialized")
    return _accuracy_scorer


def get_temporal_analyzer():
    """Return the TemporalConsistencyAnalyzer singleton.

    Lazily imports and instantiates the temporal consistency analysis
    engine using the global configuration.

    Returns:
        TemporalConsistencyAnalyzer instance configured from get_config().
    """
    global _temporal_analyzer
    if _temporal_analyzer is None:
        from greenlang.agents.eudr.geolocation_verification.temporal_analyzer import (
            TemporalConsistencyAnalyzer,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _temporal_analyzer = TemporalConsistencyAnalyzer(config=cfg)
        logger.info("TemporalConsistencyAnalyzer singleton initialized")
    return _temporal_analyzer


def get_batch_pipeline():
    """Return the BatchVerificationPipeline singleton.

    Lazily imports and instantiates the batch verification pipeline
    using the global configuration and all engine dependencies.

    Returns:
        BatchVerificationPipeline instance configured from get_config().
    """
    global _batch_pipeline
    if _batch_pipeline is None:
        from greenlang.agents.eudr.geolocation_verification.batch_pipeline import (
            BatchVerificationPipeline,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _batch_pipeline = BatchVerificationPipeline(config=cfg)
        logger.info("BatchVerificationPipeline singleton initialized")
    return _batch_pipeline


def get_article9_reporter():
    """Return the Article9ComplianceReporter singleton.

    Lazily imports and instantiates the Article 9 compliance reporting
    engine using the global configuration.

    Returns:
        Article9ComplianceReporter instance configured from get_config().
    """
    global _article9_reporter
    if _article9_reporter is None:
        from greenlang.agents.eudr.geolocation_verification.compliance_reporter import (
            Article9ComplianceReporter,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _article9_reporter = Article9ComplianceReporter(config=cfg)
        logger.info("Article9ComplianceReporter singleton initialized")
    return _article9_reporter


def get_verification_service():
    """Return the GeolocationVerificationService facade singleton.

    Lazily imports and instantiates the high-level verification service
    facade that coordinates all verification engines.

    Returns:
        GeolocationVerificationService instance configured from get_config().
    """
    global _verification_service
    if _verification_service is None:
        from greenlang.agents.eudr.geolocation_verification.service import (
            GeolocationVerificationService,
        )
        from greenlang.agents.eudr.geolocation_verification.config import get_config

        cfg = get_config()
        _verification_service = GeolocationVerificationService(config=cfg)
        logger.info("GeolocationVerificationService singleton initialized")
    return _verification_service


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# ---------------------------------------------------------------------------
# Success envelope response
# ---------------------------------------------------------------------------


class SuccessResponse(BaseModel):
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
    "get_accuracy_scorer",
    "get_article9_reporter",
    "get_batch_pipeline",
    "get_coordinate_validator",
    "get_current_user",
    "get_deforestation_verifier",
    "get_pagination",
    "get_polygon_verifier",
    "get_protected_area_checker",
    "get_temporal_analyzer",
    "get_verification_service",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
]
