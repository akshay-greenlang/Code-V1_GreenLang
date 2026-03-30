# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-025 Risk Mitigation Advisor

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_rma_service: Returns the RiskMitigationAdvisorSetup singleton.
    - RateLimiter helpers: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_uuid: Validates UUID path parameters.

RBAC Permissions (21 permissions, eudr-rma: prefix):
    eudr-rma:strategies:read, eudr-rma:strategies:execute,
    eudr-rma:plans:read, eudr-rma:plans:write, eudr-rma:plans:approve,
    eudr-rma:capacity:read, eudr-rma:capacity:manage,
    eudr-rma:library:read, eudr-rma:library:manage,
    eudr-rma:effectiveness:read,
    eudr-rma:monitoring:read, eudr-rma:monitoring:acknowledge,
    eudr-rma:optimization:execute, eudr-rma:optimization:read,
    eudr-rma:collaboration:participate, eudr-rma:collaboration:manage,
    eudr-rma:reports:read, eudr-rma:reports:generate, eudr-rma:reports:dds,
    eudr-rma:audit:read,
    eudr-rma:supplier-portal:access

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
"""

from __future__ import annotations

import logging
import time
import uuid
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

    Uses wildcard matching: ``eudr-rma:*`` grants all
    ``eudr-rma:<scope>:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-rma:strategies:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/strategies")
        ... async def list_strategies(
        ...     user: AuthUser = Depends(require_permission("eudr-rma:strategies:read"))
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
        limit: Maximum number of results per page (1-1000).
        offset: Number of results to skip for pagination.

    Returns:
        PaginationParams instance.
    """
    return PaginationParams(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-tenant)
# ---------------------------------------------------------------------------

_rate_limit_buckets: Dict[str, Dict[str, float]] = defaultdict(dict)


class RateLimiter:
    """Simple in-memory rate limiter per tenant.

    Uses a sliding-window counter to limit requests per minute.
    Production deployments use Redis-backed rate limiting via Kong.

    Args:
        requests_per_minute: Maximum requests per minute per tenant.
    """

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    async def __call__(
        self,
        request: Request,
        user: AuthUser = Depends(get_current_user),
    ) -> None:
        """Check rate limit for the current tenant.

        Raises:
            HTTPException: 429 if rate limit exceeded.
        """
        tenant_key = f"{user.tenant_id}:{request.url.path}"
        now = time.monotonic()
        window_start = now - 60.0

        # Clean old entries
        self._buckets[tenant_key] = [
            ts for ts in self._buckets[tenant_key] if ts > window_start
        ]

        if len(self._buckets[tenant_key]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests/minute",
                headers={"Retry-After": "60"},
            )

        self._buckets[tenant_key].append(now)


# Pre-configured rate limiters
rate_limit_standard = RateLimiter(requests_per_minute=100)
rate_limit_heavy = RateLimiter(requests_per_minute=30)
rate_limit_write = RateLimiter(requests_per_minute=50)
rate_limit_ml = RateLimiter(requests_per_minute=30)
rate_limit_optimize = RateLimiter(requests_per_minute=10)
rate_limit_report = RateLimiter(requests_per_minute=10)
rate_limit_upload = RateLimiter(requests_per_minute=20)


# ---------------------------------------------------------------------------
# Service singleton dependency
# ---------------------------------------------------------------------------


def get_rma_service() -> Any:
    """Get the RiskMitigationAdvisorSetup singleton.

    Returns the initialized service facade for engine access.
    Falls back to None if service not yet started.

    Returns:
        RiskMitigationAdvisorSetup instance or raises 503.
    """
    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.setup import get_service

        service = get_service()
        if service is None or not getattr(service, "_initialized", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk Mitigation Advisor service not initialized",
            )
        return service
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk Mitigation Advisor service not available",
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate a string is a valid UUID format.

    Args:
        value: String to validate.
        field_name: Field name for error message.

    Returns:
        The validated UUID string.

    Raises:
        HTTPException: 400 if invalid UUID format.
    """
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format for {field_name}: {value}",
        )


def validate_date_range(
    start_date: Optional[date] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date filter (YYYY-MM-DD)"),
) -> Dict[str, Optional[date]]:
    """Validate and return start/end date range.

    Args:
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        Dictionary with start_date and end_date.

    Raises:
        HTTPException: 400 if start_date > end_date.
    """
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"start_date ({start_date}) must be <= end_date ({end_date})",
        )
    return {"start_date": start_date, "end_date": end_date}
