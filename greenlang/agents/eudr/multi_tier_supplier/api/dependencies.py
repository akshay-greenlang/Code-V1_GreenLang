# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-008 Multi-Tier Supplier Tracker

FastAPI dependency injection providers for authentication, authorization,
rate limiting, service access, and input validation. All route handlers
inject these dependencies to enforce JWT auth (SEC-001), RBAC (SEC-002),
per-endpoint rate limits, and input constraints.

Dependencies:
    - get_current_user: Extracts and validates JWT or API key credentials.
    - require_permission: Factory returning RBAC permission check dependency.
    - get_supplier_service: Returns the Multi-Tier Supplier service singleton.
    - validate_supplier_id: Validates supplier ID format.
    - validate_commodity: Validates EUDR commodity.
    - validate_country_iso: Validates ISO 3166-1 alpha-2 country code.
    - Rate limiters: standard (200/min), batch (10/min), report (20/min).
    - PaginationParams: Standard pagination query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    EUDR_COMMODITIES,
)

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
    from SEC-001). Falls back to manual token or API key validation.

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

    # Fallback: validate JWT token manually
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

    Uses wildcard matching: ``eudr-mst:*`` grants all
    ``eudr-mst:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-mst:discover:write``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/discover")
        ... async def discover_suppliers(
        ...     user: AuthUser = Depends(require_permission("eudr-mst:discover:write"))
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
        """Initialize rate limiter.

        Args:
            max_requests: Max requests per window.
            window_seconds: Window duration in seconds.
        """
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
_rate_limit_standard = RateLimiter(max_requests=200, window_seconds=60)
_rate_limit_batch = RateLimiter(max_requests=10, window_seconds=60)
_rate_limit_report = RateLimiter(max_requests=20, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_standard(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Standard rate limit: 200 requests/minute."""
    await _rate_limit_standard(request, user)


async def rate_limit_batch(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Batch rate limit: 10 requests/minute."""
    await _rate_limit_batch(request, user)


async def rate_limit_report(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Report rate limit: 20 requests/minute."""
    await _rate_limit_report(request, user)


# ---------------------------------------------------------------------------
# Input validation dependencies
# ---------------------------------------------------------------------------


def validate_supplier_id(
    supplier_id: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="Supplier unique identifier",
    ),
) -> str:
    """Validate supplier ID from query or path parameter.

    Args:
        supplier_id: Supplier identifier string.

    Returns:
        Validated and stripped supplier ID.

    Raises:
        HTTPException: 400 if supplier ID is empty or too long.
    """
    supplier_id = supplier_id.strip()
    if not supplier_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier ID must not be empty",
        )
    return supplier_id


def validate_commodity(
    commodity: Optional[str] = Query(
        None,
        description=(
            "EUDR commodity: cattle, cocoa, coffee, oil_palm, rubber, soy, wood"
        ),
    ),
) -> Optional[str]:
    """Validate EUDR commodity from query parameter.

    Args:
        commodity: Optional commodity string to validate.

    Returns:
        Lowercase validated commodity or None.

    Raises:
        HTTPException: 400 if commodity is not in the EUDR list.
    """
    if commodity is None:
        return None

    commodity = commodity.lower().strip()
    if commodity not in EUDR_COMMODITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid EUDR commodity: '{commodity}'. "
                f"Valid values: {EUDR_COMMODITIES}"
            ),
        )
    return commodity


def validate_country_iso(
    country_iso: Optional[str] = Query(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    ),
) -> Optional[str]:
    """Validate ISO 3166-1 alpha-2 country code from query parameter.

    Args:
        country_iso: Optional two-letter country code.

    Returns:
        Uppercase validated country code or None.

    Raises:
        HTTPException: 400 if code is not two alphabetic characters.
    """
    if country_iso is None:
        return None

    country_iso = country_iso.upper().strip()
    if len(country_iso) != 2 or not country_iso.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid country code: '{country_iso}'. "
                "Must be a two-letter ISO 3166-1 alpha-2 code."
            ),
        )
    return country_iso


def validate_relationship_id(
    rel_id: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="Relationship unique identifier",
    ),
) -> str:
    """Validate relationship ID from query or path parameter.

    Args:
        rel_id: Relationship identifier string.

    Returns:
        Validated and stripped relationship ID.

    Raises:
        HTTPException: 400 if relationship ID is empty.
    """
    rel_id = rel_id.strip()
    if not rel_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Relationship ID must not be empty",
        )
    return rel_id


def validate_report_id(
    report_id: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="Report unique identifier",
    ),
) -> str:
    """Validate report ID from query or path parameter.

    Args:
        report_id: Report identifier string.

    Returns:
        Validated and stripped report ID.

    Raises:
        HTTPException: 400 if report ID is empty.
    """
    report_id = report_id.strip()
    if not report_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report ID must not be empty",
        )
    return report_id


def validate_batch_id(
    batch_id: str = Query(
        ...,
        min_length=1,
        max_length=100,
        description="Batch job unique identifier",
    ),
) -> str:
    """Validate batch job ID from query or path parameter.

    Args:
        batch_id: Batch job identifier string.

    Returns:
        Validated and stripped batch ID.

    Raises:
        HTTPException: 400 if batch ID is empty.
    """
    batch_id = batch_id.strip()
    if not batch_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch ID must not be empty",
        )
    return batch_id


def validate_severity(
    severity: Optional[str] = Query(
        None,
        description="Alert severity: critical, high, medium, low",
    ),
) -> Optional[str]:
    """Validate alert severity from query parameter.

    Args:
        severity: Optional severity string.

    Returns:
        Lowercase validated severity or None.

    Raises:
        HTTPException: 400 if severity is not valid.
    """
    if severity is None:
        return None

    severity = severity.lower().strip()
    allowed = {"critical", "high", "medium", "low"}
    if severity not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid severity: '{severity}'. "
                f"Valid values: {sorted(allowed)}"
            ),
        )
    return severity


def validate_compliance_status(
    compliance_status: Optional[str] = Query(
        None,
        description="Compliance status filter",
    ),
) -> Optional[str]:
    """Validate compliance status from query parameter.

    Args:
        compliance_status: Optional compliance status string.

    Returns:
        Lowercase validated status or None.

    Raises:
        HTTPException: 400 if status is not valid.
    """
    if compliance_status is None:
        return None

    compliance_status = compliance_status.lower().strip()
    allowed = {
        "compliant",
        "conditionally_compliant",
        "non_compliant",
        "unverified",
        "expired",
    }
    if compliance_status not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid compliance status: '{compliance_status}'. "
                f"Valid values: {sorted(allowed)}"
            ),
        )
    return compliance_status


# ---------------------------------------------------------------------------
# Service singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_supplier_service = None


def get_supplier_service():
    """Return the Multi-Tier Supplier service singleton.

    Lazily imports and instantiates the supplier tracking service
    using the global configuration. The singleton is cached for the
    lifetime of the process.

    Returns:
        MultiTierSupplierService instance configured from get_config().
    """
    global _supplier_service
    if _supplier_service is None:
        try:
            from greenlang.agents.eudr.multi_tier_supplier.setup import (
                MultiTierSupplierService,
            )
            from greenlang.agents.eudr.multi_tier_supplier.config import (
                get_config,
            )

            cfg = get_config()
            _supplier_service = MultiTierSupplierService(config=cfg)
            logger.info("MultiTierSupplierService singleton initialized")
        except ImportError:
            logger.warning(
                "MultiTierSupplierService not available, using stub. "
                "Ensure service module is implemented."
            )
            _supplier_service = _MultiTierSupplierServiceStub()
    return _supplier_service


class _MultiTierSupplierServiceStub:
    """Stub service for development when real service is not yet available.

    All methods raise NotImplementedError with a helpful message
    directing developers to implement the actual service module.
    """

    def __getattr__(self, name: str) -> Any:
        """Raise NotImplementedError for any method call."""
        def _stub(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"MultiTierSupplierService.{name}() is not yet implemented. "
                "Implement greenlang.agents.eudr.multi_tier_supplier.setup"
            )
        return _stub


# ---------------------------------------------------------------------------
# Error response model (convenience re-export)
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Auth
    "AuthUser",
    "get_current_user",
    "require_permission",
    "oauth2_scheme",
    "api_key_header",
    # Rate limiting
    "RateLimiter",
    "rate_limit_standard",
    "rate_limit_batch",
    "rate_limit_report",
    # Pagination
    "PaginationParams",
    "get_pagination",
    # Validation
    "validate_supplier_id",
    "validate_commodity",
    "validate_country_iso",
    "validate_relationship_id",
    "validate_report_id",
    "validate_batch_id",
    "validate_severity",
    "validate_compliance_status",
    # Service
    "get_supplier_service",
    # Response models
    "ErrorResponse",
    "SuccessResponse",
]
