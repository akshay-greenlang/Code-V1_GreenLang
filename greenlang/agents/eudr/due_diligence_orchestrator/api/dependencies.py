# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-026 Due Diligence Orchestrator

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_ddo_service: Returns the DueDiligenceOrchestratorService singleton.
    - get_workflow_engine: Returns the WorkflowDefinitionEngine singleton.
    - get_info_coordinator: Returns the InformationGatheringCoordinator singleton.
    - get_risk_coordinator: Returns the RiskAssessmentCoordinator singleton.
    - get_mitigation_coordinator: Returns the RiskMitigationCoordinator singleton.
    - get_quality_gate_engine: Returns the QualityGateEngine singleton.
    - get_state_manager: Returns the WorkflowStateManager singleton.
    - get_parallel_engine: Returns the ParallelExecutionEngine singleton.
    - get_error_manager: Returns the ErrorRecoveryManager singleton.
    - get_package_generator: Returns the DueDiligencePackageGenerator singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.

RBAC Permissions (19):
    eudr-ddo:workflows:read, eudr-ddo:workflows:create,
    eudr-ddo:workflows:manage, eudr-ddo:workflows:delete,
    eudr-ddo:templates:read, eudr-ddo:templates:manage,
    eudr-ddo:gates:read, eudr-ddo:gates:override,
    eudr-ddo:checkpoints:read, eudr-ddo:checkpoints:rollback,
    eudr-ddo:audit-trail:read,
    eudr-ddo:packages:read, eudr-ddo:packages:generate,
    eudr-ddo:packages:download,
    eudr-ddo:batch:manage,
    eudr-ddo:circuit-breakers:read, eudr-ddo:circuit-breakers:manage,
    eudr-ddo:dlq:read, eudr-ddo:dlq:manage

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
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

    Uses wildcard matching: ``eudr-ddo:*`` grants all
    ``eudr-ddo:<resource>:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-ddo:workflows:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/workflows")
        ... async def list_workflows(
        ...     user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read"))
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

    def __init__(self, max_requests: int = 200, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    async def __call__(
        self,
        request: Request,
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        """Check rate limit for the current user.

        Args:
            request: FastAPI request object.
            user: Authenticated user.

        Returns:
            AuthUser if within rate limit.

        Raises:
            HTTPException: 429 if rate limit exceeded.
        """
        key = f"{user.tenant_id}:{user.user_id}:{request.url.path}"
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Prune old entries
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > window_start
        ]

        if len(self._requests[key]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {self.max_requests} "
                    f"requests per {self.window_seconds}s"
                ),
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        self._requests[key].append(now)
        return user


# Pre-configured rate limiter instances
rate_limit_standard = RateLimiter(max_requests=200, window_seconds=60)
rate_limit_write = RateLimiter(max_requests=50, window_seconds=60)
rate_limit_package = RateLimiter(max_requests=20, window_seconds=60)
rate_limit_batch = RateLimiter(max_requests=10, window_seconds=60)


# ---------------------------------------------------------------------------
# Service / Engine dependency injection
# ---------------------------------------------------------------------------

_ddo_service_instance = None


def get_ddo_service():
    """Return the DueDiligenceOrchestratorService singleton.

    Lazily initializes the service on first call. Thread-safe
    through Python GIL for single-writer initialization.

    Returns:
        DueDiligenceOrchestratorService instance.
    """
    global _ddo_service_instance
    if _ddo_service_instance is None:
        from greenlang.agents.eudr.due_diligence_orchestrator.setup import (
            DueDiligenceOrchestratorService,
        )

        _ddo_service_instance = DueDiligenceOrchestratorService()
    return _ddo_service_instance


def get_workflow_engine():
    """Return the WorkflowDefinitionEngine from the DDO service."""
    return get_ddo_service()._workflow_engine


def get_info_coordinator():
    """Return the InformationGatheringCoordinator from the DDO service."""
    return get_ddo_service()._info_coordinator


def get_risk_coordinator():
    """Return the RiskAssessmentCoordinator from the DDO service."""
    return get_ddo_service()._risk_coordinator


def get_mitigation_coordinator():
    """Return the RiskMitigationCoordinator from the DDO service."""
    return get_ddo_service()._mitigation_coordinator


def get_quality_gate_engine():
    """Return the QualityGateEngine from the DDO service."""
    return get_ddo_service()._quality_gate_engine


def get_state_manager():
    """Return the WorkflowStateManager from the DDO service."""
    return get_ddo_service()._state_manager


def get_parallel_engine():
    """Return the ParallelExecutionEngine from the DDO service."""
    return get_ddo_service()._parallel_engine


def get_error_manager():
    """Return the ErrorRecoveryManager from the DDO service."""
    return get_ddo_service()._error_manager


def get_package_generator():
    """Return the DueDiligencePackageGenerator from the DDO service."""
    return get_ddo_service()._package_generator


# ---------------------------------------------------------------------------
# Common query parameter validators
# ---------------------------------------------------------------------------


def validate_commodity(
    commodity: Optional[str] = Query(
        default=None,
        description="EUDR commodity filter: cattle, cocoa, coffee, palm_oil, rubber, soya, wood",
    ),
) -> Optional[str]:
    """Validate commodity query parameter against EUDR-regulated commodities.

    Args:
        commodity: Optional commodity string.

    Returns:
        Validated commodity string or None.

    Raises:
        HTTPException: 400 if commodity is not a valid EUDR commodity.
    """
    valid_commodities = {
        "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
    }
    if commodity is not None and commodity not in valid_commodities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid commodity: {commodity}. "
                f"Valid values: {', '.join(sorted(valid_commodities))}"
            ),
        )
    return commodity


def validate_workflow_type(
    workflow_type: Optional[str] = Query(
        default=None,
        description="Workflow type filter: standard, simplified, custom",
    ),
) -> Optional[str]:
    """Validate workflow type query parameter.

    Args:
        workflow_type: Optional workflow type string.

    Returns:
        Validated workflow type string or None.

    Raises:
        HTTPException: 400 if workflow type is invalid.
    """
    valid_types = {"standard", "simplified", "custom"}
    if workflow_type is not None and workflow_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid workflow_type: {workflow_type}. "
                f"Valid values: {', '.join(sorted(valid_types))}"
            ),
        )
    return workflow_type


def validate_status_filter(
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Workflow status filter",
    ),
) -> Optional[str]:
    """Validate workflow status query parameter.

    Args:
        status_filter: Optional status string.

    Returns:
        Validated status string or None.

    Raises:
        HTTPException: 400 if status is invalid.
    """
    valid_statuses = {
        "created", "validating", "running", "paused", "quality_gate",
        "gate_failed", "resuming", "completing", "completed",
        "cancelled", "terminated",
    }
    if status_filter is not None and status_filter not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid status: {status_filter}. "
                f"Valid values: {', '.join(sorted(valid_statuses))}"
            ),
        )
    return status_filter


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(GreenLangBase):
    """Standardized API error response."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request correlation ID"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "ErrorResponse",
    "PaginationParams",
    "RateLimiter",
    "api_key_header",
    "get_current_user",
    "get_ddo_service",
    "get_error_manager",
    "get_info_coordinator",
    "get_mitigation_coordinator",
    "get_package_generator",
    "get_pagination",
    "get_parallel_engine",
    "get_quality_gate_engine",
    "get_risk_coordinator",
    "get_state_manager",
    "get_workflow_engine",
    "oauth2_scheme",
    "rate_limit_batch",
    "rate_limit_package",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_commodity",
    "validate_status_filter",
    "validate_workflow_type",
]
