# -*- coding: utf-8 -*-
"""
RBAC Permissions REST API Routes - SEC-002

FastAPI APIRouter providing REST endpoints for permission management within
the GreenLang RBAC authorization layer.  Supports listing all permissions,
granting permissions to roles, and revoking permissions from roles.

Endpoints:
    GET    /api/v1/rbac/permissions                                  - List permissions (paginated)
    POST   /api/v1/rbac/roles/{role_id}/permissions                  - Grant permission to role
    DELETE /api/v1/rbac/roles/{role_id}/permissions/{permission_id}   - Revoke permission from role

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.rbac_service.api.permissions_routes import permissions_router
    >>> app = FastAPI()
    >>> app.include_router(permissions_router)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class GrantPermissionRequest(BaseModel):
        """Request schema for granting a permission to a role.

        Attributes:
            permission_id: UUID of the permission to grant.
            effect: Permission effect (``allow`` or ``deny``).
            conditions: Optional JSON conditions for conditional permissions.
            scope: Permission scope (``global``, ``tenant``, ``resource``).
        """

        model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
            json_schema_extra={
                "examples": [
                    {
                        "permission_id": "perm-reports-read",
                        "effect": "allow",
                        "conditions": None,
                        "scope": "tenant",
                    }
                ]
            },
        )

        permission_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the permission to grant.",
        )
        effect: str = Field(
            default="allow",
            description="Permission effect: 'allow' or 'deny'.",
        )
        conditions: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Optional conditions for conditional permission grants (ABAC-style).",
        )
        scope: str = Field(
            default="tenant",
            description="Permission scope: 'global', 'tenant', or 'resource'.",
        )

        @field_validator("effect")
        @classmethod
        def validate_effect(cls, v: str) -> str:
            """Validate effect is 'allow' or 'deny'."""
            v_lower = v.strip().lower()
            if v_lower not in ("allow", "deny"):
                raise ValueError(
                    f"Invalid effect '{v}'. Must be 'allow' or 'deny'."
                )
            return v_lower

        @field_validator("scope")
        @classmethod
        def validate_scope(cls, v: str) -> str:
            """Validate scope is one of the allowed values."""
            allowed = {"global", "tenant", "resource"}
            v_lower = v.strip().lower()
            if v_lower not in allowed:
                raise ValueError(
                    f"Invalid scope '{v}'. Allowed: {sorted(allowed)}"
                )
            return v_lower

    class PermissionResponse(BaseModel):
        """Response schema for a single permission.

        Attributes:
            id: Permission UUID.
            resource: Resource identifier (e.g. ``reports``, ``agents``).
            action: Action identifier (e.g. ``read``, ``write``, ``delete``).
            description: Permission description.
            created_at: Creation timestamp.
        """

        model_config = ConfigDict(from_attributes=True)

        id: str = Field(..., description="Permission UUID.")
        resource: str = Field(..., description="Resource identifier.")
        action: str = Field(..., description="Action identifier.")
        description: str = Field(
            default="", description="Permission description."
        )
        created_at: Optional[datetime] = Field(
            default=None, description="Creation timestamp."
        )

    class PermissionListResponse(BaseModel):
        """Paginated list of permissions.

        Attributes:
            items: Permissions for this page.
            total: Total matching permissions.
            page: Current page number.
            page_size: Items per page.
            total_pages: Total pages.
            has_next: Whether there is a next page.
            has_prev: Whether there is a previous page.
        """

        items: List[PermissionResponse] = Field(
            ..., description="Permissions for this page."
        )
        total: int = Field(..., ge=0, description="Total matching permissions.")
        page: int = Field(..., ge=1, description="Current page number.")
        page_size: int = Field(..., ge=1, description="Items per page.")
        total_pages: int = Field(..., ge=0, description="Total pages.")
        has_next: bool = Field(..., description="Has next page.")
        has_prev: bool = Field(..., description="Has previous page.")

    class PermissionGrantResponse(BaseModel):
        """Response schema for a permission grant operation.

        Attributes:
            id: Grant record UUID.
            role_id: UUID of the role.
            permission_id: UUID of the permission.
            effect: Grant effect.
            conditions: Grant conditions.
            scope: Grant scope.
            granted_by: Actor who granted the permission.
            granted_at: Timestamp of the grant.
        """

        model_config = ConfigDict(from_attributes=True)

        id: str = Field(..., description="Grant record UUID.")
        role_id: str = Field(..., description="Role UUID.")
        permission_id: str = Field(..., description="Permission UUID.")
        effect: str = Field(default="allow", description="Grant effect.")
        conditions: Optional[Dict[str, Any]] = Field(
            default=None, description="Grant conditions."
        )
        scope: str = Field(default="tenant", description="Grant scope.")
        granted_by: Optional[str] = Field(
            default=None, description="Actor who granted."
        )
        granted_at: Optional[datetime] = Field(
            default=None, description="Grant timestamp."
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


def _get_role_service() -> Any:
    """FastAPI dependency that provides the RoleService instance.

    Returns:
        The RoleService singleton.

    Raises:
        HTTPException 503: If the service is not available.
    """
    try:
        from greenlang.infrastructure.rbac_service.role_service import (
            RoleService,
            get_role_service,
        )

        return get_role_service()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RBAC role service is not available.",
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


def _get_actor_id(request: Any) -> str:
    """Extract actor ID from request headers."""
    return request.headers.get("x-user-id", "anonymous")


def _get_tenant_id(request: Any) -> Optional[str]:
    """Extract tenant ID from request headers."""
    return request.headers.get("x-tenant-id")


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

    permissions_router = APIRouter(
        prefix="/api/v1/rbac",
        tags=["RBAC Permissions"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Not Found"},
            409: {"description": "Conflict"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -- List Permissions --------------------------------------------------

    @permissions_router.get(
        "/permissions",
        response_model=PermissionListResponse,
        summary="List permissions",
        description="Retrieve a paginated list of all available permissions, optionally filtered by resource.",
        operation_id="list_permissions",
    )
    async def list_permissions(
        request: Request,
        resource: Optional[str] = Query(
            None,
            description="Filter by resource type (e.g. 'agents', 'reports').",
        ),
        page: int = Query(1, ge=1, description="Page number (1-indexed)."),
        page_size: int = Query(
            50, ge=1, le=100, description="Items per page."
        ),
        permission_service: Any = Depends(_get_permission_service),
    ) -> PermissionListResponse:
        """List all available permissions with optional resource filter.

        Args:
            request: The incoming HTTP request.
            resource: Optional resource type filter.
            page: Page number.
            page_size: Items per page.
            permission_service: Injected PermissionService.

        Returns:
            Paginated list of permissions.
        """
        try:
            result = await permission_service.list_permissions(
                resource_filter=resource,
                page=page,
                page_size=page_size,
            )

            items = [
                PermissionResponse(**perm)
                for perm in result.get("items", [])
            ]
            total = result.get("total", 0)
            total_pages = (
                (total + page_size - 1) // page_size if total > 0 else 0
            )

            return PermissionListResponse(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1,
            )

        except Exception as exc:
            logger.exception("Failed to list permissions")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list permissions: {exc}",
            )

    # -- Grant Permission to Role ------------------------------------------

    @permissions_router.post(
        "/roles/{role_id}/permissions",
        response_model=PermissionGrantResponse,
        status_code=201,
        summary="Grant permission to role",
        description="Grant a permission to an RBAC role with optional conditions and scope.",
        operation_id="grant_permission_to_role",
    )
    async def grant_permission_to_role(
        request: Request,
        role_id: str,
        body: GrantPermissionRequest,
        role_service: Any = Depends(_get_role_service),
        permission_service: Any = Depends(_get_permission_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> PermissionGrantResponse:
        """Grant a permission to a role.

        Args:
            request: The incoming HTTP request.
            role_id: UUID of the role to grant the permission to.
            body: Grant permission request body.
            role_service: Injected RoleService.
            permission_service: Injected PermissionService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Returns:
            The created permission grant.

        Raises:
            HTTPException 404: If the role or permission does not exist.
            HTTPException 409: If the permission is already granted.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Verify role exists
        role = await role_service.get_role(role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        # Verify permission exists
        perm = await permission_service.get_permission(body.permission_id)
        if perm is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Permission '{body.permission_id}' not found.",
            )

        try:
            grant = await permission_service.grant_permission_to_role(
                role_id=role_id,
                permission_id=body.permission_id,
                effect=body.effect,
                conditions=body.conditions,
                scope=body.scope,
                granted_by=actor_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

        # Audit and metrics
        await audit.log_permission_granted(
            tenant_id=tenant_id,
            actor_id=actor_id,
            role_id=role_id,
            permission_id=body.permission_id,
            effect=body.effect,
            ip_address=client_ip,
            correlation_id=correlation_id,
        )

        logger.info(
            "API: Granted permission '%s' to role '%s' by actor %s",
            body.permission_id,
            role_id,
            actor_id,
        )

        return PermissionGrantResponse(**grant)

    # -- Revoke Permission from Role ---------------------------------------

    @permissions_router.delete(
        "/roles/{role_id}/permissions/{permission_id}",
        status_code=204,
        summary="Revoke permission from role",
        description="Revoke a previously granted permission from an RBAC role.",
        operation_id="revoke_permission_from_role",
    )
    async def revoke_permission_from_role(
        request: Request,
        role_id: str,
        permission_id: str,
        role_service: Any = Depends(_get_role_service),
        permission_service: Any = Depends(_get_permission_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> None:
        """Revoke a permission from a role.

        Args:
            request: The incoming HTTP request.
            role_id: UUID of the role.
            permission_id: UUID of the permission to revoke.
            role_service: Injected RoleService.
            permission_service: Injected PermissionService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Raises:
            HTTPException 404: If the role or permission grant does not exist.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Verify role exists
        role = await role_service.get_role(role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        result = await permission_service.revoke_permission_from_role(
            role_id, permission_id
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Permission grant '{permission_id}' not found for role '{role_id}'.",
            )

        # Audit
        await audit.log_permission_revoked(
            tenant_id=tenant_id,
            actor_id=actor_id,
            role_id=role_id,
            permission_id=permission_id,
            ip_address=client_ip,
            correlation_id=correlation_id,
        )

        logger.info(
            "API: Revoked permission '%s' from role '%s' by actor %s",
            permission_id,
            role_id,
            actor_id,
        )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(permissions_router)
    except ImportError:
        pass  # auth_service not available

else:
    permissions_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - permissions_router is None")


__all__ = ["permissions_router"]
