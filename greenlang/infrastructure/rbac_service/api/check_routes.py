# -*- coding: utf-8 -*-
"""
RBAC Permission Check REST API Route - SEC-002

FastAPI APIRouter providing the permission check endpoint for the GreenLang
RBAC authorization layer.  This is the primary runtime endpoint used by
API gateways, middleware, and services to evaluate whether a user has the
required permission to perform an action on a resource.

Endpoint:
    POST /api/v1/rbac/check - Evaluate authorization for a user/resource/action.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.rbac_service.api.check_routes import check_router
    >>> app = FastAPI()
    >>> app.include_router(check_router)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CheckPermissionRequest(BaseModel):
        """Request schema for checking user authorization.

        Attributes:
            user_id: UUID of the user to check.
            resource: Resource being accessed (e.g. ``reports``, ``agents``).
            action: Action being performed (e.g. ``read``, ``write``, ``delete``).
            tenant_id: UUID of the tenant scope.
            context: Optional context for ABAC-style conditional evaluation.
        """

        model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
            json_schema_extra={
                "examples": [
                    {
                        "user_id": "u-analyst-01",
                        "resource": "reports",
                        "action": "read",
                        "tenant_id": "t-acme-corp",
                        "context": {"environment": "production"},
                    }
                ]
            },
        )

        user_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the user to check.",
        )
        resource: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="Resource being accessed (e.g. 'reports', 'agents').",
        )
        action: str = Field(
            ...,
            min_length=1,
            max_length=64,
            description="Action being performed (e.g. 'read', 'write', 'delete').",
        )
        tenant_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the tenant scope.",
        )
        context: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Optional ABAC-style context for conditional evaluation.",
        )

    class CheckPermissionResponse(BaseModel):
        """Response schema for permission check result.

        Attributes:
            allowed: Whether the action is authorized.
            matched_permissions: List of permissions that matched (for audit).
            evaluation_time_ms: Time taken to evaluate the permission in milliseconds.
            cache_hit: Whether the result was served from cache.
        """

        allowed: bool = Field(
            ..., description="Whether the action is authorized."
        )
        matched_permissions: List[str] = Field(
            default_factory=list,
            description="Permissions that matched the request.",
        )
        evaluation_time_ms: float = Field(
            default=0.0,
            ge=0.0,
            description="Evaluation duration in milliseconds.",
        )
        cache_hit: bool = Field(
            default=False,
            description="Whether the result was served from cache.",
        )


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


def _get_permission_service() -> Any:
    """FastAPI dependency that provides the PermissionService instance.

    Returns:
        The PermissionService singleton.

    Raises:
        HTTPException 503: If the service is not available.
    """
    try:
        from greenlang.infrastructure.rbac_service.permission_service import (
            PermissionService,
            get_permission_service,
        )

        return get_permission_service()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RBAC permission service is not available.",
        )


def _get_audit_logger() -> Any:
    """FastAPI dependency that provides the RBACAuditLogger instance."""
    from greenlang.infrastructure.rbac_service.rbac_audit import RBACAuditLogger

    return RBACAuditLogger()


def _get_metrics() -> Any:
    """FastAPI dependency that provides the RBACMetrics instance."""
    from greenlang.infrastructure.rbac_service.rbac_metrics import RBACMetrics

    return RBACMetrics()


# ---------------------------------------------------------------------------
# Helper: extract context from request headers
# ---------------------------------------------------------------------------


def _get_client_ip(request: Any) -> Optional[str]:
    """Extract client IP from request."""
    if hasattr(request, "client") and request.client:
        return request.client.host
    return None


def _get_correlation_id(request: Any) -> Optional[str]:
    """Extract correlation ID from request headers."""
    return request.headers.get("x-correlation-id") or request.headers.get(
        "x-request-id"
    )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from fastapi import Request

    check_router = APIRouter(
        prefix="/api/v1/rbac",
        tags=["RBAC Authorization Check"],
        responses={
            400: {"description": "Bad Request"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    # -- Permission Check --------------------------------------------------

    @check_router.post(
        "/check",
        response_model=CheckPermissionResponse,
        summary="Check permission",
        description=(
            "Evaluate whether a user has the required permission to perform "
            "an action on a resource within a tenant scope. This is the primary "
            "runtime authorization endpoint used by the API gateway and services."
        ),
        operation_id="check_permission",
    )
    async def check_permission(
        request: Request,
        body: CheckPermissionRequest,
        permission_service: Any = Depends(_get_permission_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> CheckPermissionResponse:
        """Check if a user has a specific permission.

        This endpoint evaluates the user's roles, inherited permissions,
        and any conditional (ABAC) rules to determine if the requested
        action is authorized.  Results may be served from cache for
        sub-millisecond latency.

        Args:
            request: The incoming HTTP request.
            body: Permission check request body.
            permission_service: Injected PermissionService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Returns:
            Authorization decision with matched permissions and timing.
        """
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        start_time = time.perf_counter()

        try:
            result = await permission_service.evaluate_permission(
                user_id=body.user_id,
                tenant_id=body.tenant_id,
                resource=body.resource,
                action=body.action,
                context=body.context or {},
            )

            elapsed_s = time.perf_counter() - start_time
            elapsed_ms = elapsed_s * 1000.0

            allowed = result.get("allowed", False)
            matched = result.get("matched_permissions", [])
            cache_hit = result.get("cache_hit", False)

            # Record metrics
            metrics.record_authorization(
                result="allowed" if allowed else "denied",
                resource=body.resource,
                action=body.action,
                tenant_id=body.tenant_id,
                duration_s=elapsed_s,
                cache_hit=cache_hit,
            )

            if cache_hit:
                metrics.record_cache_hit(body.tenant_id)
            else:
                metrics.record_cache_miss(body.tenant_id)

            # Audit log (fire-and-forget for authorization decisions)
            await audit.log_authorization_decision(
                tenant_id=body.tenant_id,
                user_id=body.user_id,
                resource=body.resource,
                action=body.action,
                allowed=allowed,
                matched_permissions=matched,
                evaluation_time_ms=elapsed_ms,
                cache_hit=cache_hit,
                ip_address=client_ip,
                correlation_id=correlation_id,
            )

            return CheckPermissionResponse(
                allowed=allowed,
                matched_permissions=matched,
                evaluation_time_ms=round(elapsed_ms, 3),
                cache_hit=cache_hit,
            )

        except Exception as exc:
            elapsed_s = time.perf_counter() - start_time

            # Record error metrics
            metrics.record_authorization(
                result="error",
                resource=body.resource,
                action=body.action,
                tenant_id=body.tenant_id,
                duration_s=elapsed_s,
                cache_hit=False,
            )

            logger.exception(
                "Permission check failed for user '%s' resource '%s' action '%s'",
                body.user_id,
                body.resource,
                body.action,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Permission evaluation failed due to an internal error.",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(check_router)
    except ImportError:
        pass  # auth_service not available

else:
    check_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - check_router is None")


__all__ = ["check_router"]
