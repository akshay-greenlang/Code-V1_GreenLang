# -*- coding: utf-8 -*-
"""
RBAC Roles REST API Routes - SEC-002

FastAPI APIRouter providing REST endpoints for role management within the
GreenLang RBAC authorization layer.  Supports role CRUD, enable/disable,
hierarchy inspection, and role-permission listing.

Endpoints:
    GET    /api/v1/rbac/roles                       - List roles (paginated)
    POST   /api/v1/rbac/roles                       - Create role
    GET    /api/v1/rbac/roles/{role_id}              - Get role details
    PUT    /api/v1/rbac/roles/{role_id}              - Update role
    DELETE /api/v1/rbac/roles/{role_id}              - Delete role
    GET    /api/v1/rbac/roles/{role_id}/permissions  - List role permissions

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.rbac_service.api.roles_routes import roles_router
    >>> app = FastAPI()
    >>> app.include_router(roles_router)

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

    class CreateRoleRequest(BaseModel):
        """Request schema for creating a new RBAC role.

        Attributes:
            name: Unique role identifier (lowercase snake_case).
            display_name: Human-readable display name.
            description: Detailed role description.
            parent_role_id: UUID of the parent role for hierarchy inheritance.
            metadata: Arbitrary key-value metadata.
        """

        model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
            json_schema_extra={
                "examples": [
                    {
                        "name": "emissions_analyst",
                        "display_name": "Emissions Analyst",
                        "description": "Can view and analyze emissions data.",
                        "parent_role_id": None,
                        "metadata": {"department": "sustainability"},
                    }
                ]
            },
        )

        name: str = Field(
            ...,
            min_length=2,
            max_length=128,
            description="Unique role name. Lowercase alphanumeric with underscores and hyphens.",
        )
        display_name: str = Field(
            ...,
            min_length=1,
            max_length=256,
            description="Human-readable display name.",
        )
        description: str = Field(
            default="",
            max_length=2048,
            description="Detailed role description.",
        )
        parent_role_id: Optional[str] = Field(
            default=None,
            max_length=64,
            description="UUID of the parent role for hierarchy inheritance.",
        )
        metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="Arbitrary key-value metadata.",
        )

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Validate role name is lowercase with allowed characters."""
            import re

            if not re.match(r"^[a-z][a-z0-9_-]*$", v):
                raise ValueError(
                    f"Role name '{v}' must be lowercase, start with a letter, "
                    "and contain only alphanumeric, underscore, or hyphen characters."
                )
            return v

    class UpdateRoleRequest(BaseModel):
        """Request schema for updating an existing role.

        All fields are optional. Only provided fields are updated.

        Attributes:
            display_name: New display name.
            description: New description.
            parent_role_id: New parent role UUID.
            metadata: New metadata.
            is_enabled: Enable or disable the role.
        """

        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

        display_name: Optional[str] = Field(
            default=None,
            min_length=1,
            max_length=256,
            description="Updated display name.",
        )
        description: Optional[str] = Field(
            default=None,
            max_length=2048,
            description="Updated description.",
        )
        parent_role_id: Optional[str] = Field(
            default=None,
            max_length=64,
            description="Updated parent role UUID.",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Updated metadata.",
        )
        is_enabled: Optional[bool] = Field(
            default=None,
            description="Enable or disable the role.",
        )

    class RoleResponse(BaseModel):
        """Response schema for a single RBAC role.

        Attributes:
            id: Role UUID.
            name: Unique role identifier.
            display_name: Human-readable name.
            description: Role description.
            parent_role_id: Parent role UUID (if any).
            is_system: Whether this is a system-defined role.
            is_enabled: Whether the role is currently active.
            tenant_id: Tenant scope (None for global roles).
            metadata: Arbitrary metadata.
            created_at: Creation timestamp.
            updated_at: Last modification timestamp.
        """

        model_config = ConfigDict(from_attributes=True)

        id: str = Field(..., description="Role UUID.")
        name: str = Field(..., description="Unique role identifier.")
        display_name: str = Field(..., description="Human-readable name.")
        description: str = Field(default="", description="Role description.")
        parent_role_id: Optional[str] = Field(
            default=None, description="Parent role UUID."
        )
        is_system: bool = Field(
            default=False, description="Whether this is a system role."
        )
        is_enabled: bool = Field(
            default=True, description="Whether the role is active."
        )
        tenant_id: Optional[str] = Field(
            default=None, description="Tenant scope."
        )
        metadata: Dict[str, Any] = Field(
            default_factory=dict, description="Arbitrary metadata."
        )
        created_at: Optional[datetime] = Field(
            default=None, description="Creation timestamp."
        )
        updated_at: Optional[datetime] = Field(
            default=None, description="Last modification timestamp."
        )

    class RoleListResponse(BaseModel):
        """Paginated list of RBAC roles.

        Attributes:
            items: List of roles for the current page.
            total: Total number of matching roles.
            page: Current page number.
            page_size: Items per page.
            total_pages: Total number of pages.
            has_next: Whether there is a next page.
            has_prev: Whether there is a previous page.
        """

        items: List[RoleResponse] = Field(
            ..., description="Roles for this page."
        )
        total: int = Field(..., ge=0, description="Total matching roles.")
        page: int = Field(..., ge=1, description="Current page number.")
        page_size: int = Field(..., ge=1, description="Items per page.")
        total_pages: int = Field(..., ge=0, description="Total pages.")
        has_next: bool = Field(..., description="Has next page.")
        has_prev: bool = Field(..., description="Has previous page.")

    class RolePermissionsResponse(BaseModel):
        """Response schema listing permissions for a role.

        Attributes:
            role_id: UUID of the role.
            permissions: List of permission grants.
            include_inherited: Whether inherited permissions are included.
        """

        role_id: str = Field(..., description="Role UUID.")
        permissions: List[Dict[str, Any]] = Field(
            ..., description="Permission grants for this role."
        )
        include_inherited: bool = Field(
            default=True, description="Inherited permissions included."
        )


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


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
    """FastAPI dependency that provides the RBACAuditLogger instance.

    Returns:
        The RBACAuditLogger singleton.
    """
    from greenlang.infrastructure.rbac_service.rbac_audit import RBACAuditLogger

    return RBACAuditLogger()


def _get_metrics() -> Any:
    """FastAPI dependency that provides the RBACMetrics instance.

    Returns:
        The RBACMetrics singleton.
    """
    from greenlang.infrastructure.rbac_service.rbac_metrics import RBACMetrics

    return RBACMetrics()


# ---------------------------------------------------------------------------
# Helper: extract actor_id from request headers
# ---------------------------------------------------------------------------


def _get_actor_id(request: Any) -> str:
    """Extract the authenticated actor ID from the request.

    Looks for ``X-User-Id`` header (set by the auth gateway/middleware).
    Falls back to ``anonymous`` if not present.

    Args:
        request: The FastAPI Request object.

    Returns:
        Actor user ID string.
    """
    return request.headers.get("x-user-id", "anonymous")


def _get_tenant_id(request: Any) -> Optional[str]:
    """Extract the tenant ID from the request.

    Looks for ``X-Tenant-Id`` header (set by the auth gateway/middleware).

    Args:
        request: The FastAPI Request object.

    Returns:
        Tenant ID string or None.
    """
    return request.headers.get("x-tenant-id")


def _get_client_ip(request: Any) -> Optional[str]:
    """Extract the client IP from the request.

    Args:
        request: The FastAPI Request object.

    Returns:
        Client IP string or None.
    """
    if hasattr(request, "client") and request.client:
        return request.client.host
    return None


def _get_correlation_id(request: Any) -> Optional[str]:
    """Extract the correlation ID from the request.

    Args:
        request: The FastAPI Request object.

    Returns:
        Correlation ID string or None.
    """
    return request.headers.get("x-correlation-id") or request.headers.get(
        "x-request-id"
    )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from fastapi import Request

    roles_router = APIRouter(
        prefix="/api/v1/rbac/roles",
        tags=["RBAC Roles"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Role Not Found"},
            409: {"description": "Conflict"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -- List Roles --------------------------------------------------------

    @roles_router.get(
        "",
        response_model=RoleListResponse,
        summary="List roles",
        description="Retrieve a paginated list of RBAC roles, optionally filtered by tenant.",
        operation_id="list_roles",
    )
    async def list_roles(
        request: Request,
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant ID."
        ),
        include_system: bool = Query(
            True, description="Include system-defined roles."
        ),
        page: int = Query(1, ge=1, description="Page number (1-indexed)."),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page."
        ),
        role_service: Any = Depends(_get_role_service),
    ) -> RoleListResponse:
        """List roles with pagination and optional tenant filter.

        Args:
            request: The incoming HTTP request.
            tenant_id: Optional tenant ID filter.
            include_system: Whether to include system roles.
            page: Page number.
            page_size: Items per page.
            role_service: Injected RoleService.

        Returns:
            Paginated list of roles.
        """
        try:
            result = await role_service.list_roles(
                tenant_id=tenant_id,
                include_system=include_system,
                page=page,
                page_size=page_size,
            )

            items = [
                RoleResponse(**role) for role in result.get("items", [])
            ]
            total = result.get("total", 0)
            total_pages = (
                (total + page_size - 1) // page_size if total > 0 else 0
            )

            return RoleListResponse(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1,
            )

        except Exception as exc:
            logger.exception("Failed to list roles")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list roles: {exc}",
            )

    # -- Create Role -------------------------------------------------------

    @roles_router.post(
        "",
        response_model=RoleResponse,
        status_code=201,
        summary="Create a role",
        description="Create a new RBAC role definition.",
        operation_id="create_role",
    )
    async def create_role(
        request: Request,
        body: CreateRoleRequest,
        role_service: Any = Depends(_get_role_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> RoleResponse:
        """Create a new RBAC role.

        Args:
            request: The incoming HTTP request.
            body: Role creation request.
            role_service: Injected RoleService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Returns:
            The created role.

        Raises:
            HTTPException 409: If a role with the same name already exists.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        try:
            role = await role_service.create_role(
                name=body.name,
                display_name=body.display_name,
                description=body.description,
                parent_role_id=body.parent_role_id,
                tenant_id=tenant_id,
                created_by=actor_id,
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

        # Audit and metrics
        await audit.log_role_created(
            tenant_id=tenant_id,
            actor_id=actor_id,
            role_id=role.get("id", ""),
            role_name=body.name,
            ip_address=client_ip,
            correlation_id=correlation_id,
        )
        metrics.record_role_change("created", tenant_id or "global")

        logger.info(
            "API: Created role '%s' (id=%s) by actor %s",
            body.name,
            role.get("id"),
            actor_id,
        )

        return RoleResponse(**role)

    # -- Get Role ----------------------------------------------------------

    @roles_router.get(
        "/{role_id}",
        response_model=RoleResponse,
        summary="Get role details",
        description="Retrieve a specific RBAC role by its UUID.",
        operation_id="get_role",
    )
    async def get_role(
        role_id: str,
        role_service: Any = Depends(_get_role_service),
    ) -> RoleResponse:
        """Get a single role by UUID.

        Args:
            role_id: Role UUID.
            role_service: Injected RoleService.

        Returns:
            Role details.

        Raises:
            HTTPException 404: If the role does not exist.
        """
        role = await role_service.get_role(role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )
        return RoleResponse(**role)

    # -- Update Role -------------------------------------------------------

    @roles_router.put(
        "/{role_id}",
        response_model=RoleResponse,
        summary="Update role",
        description="Update an existing RBAC role.",
        operation_id="update_role",
    )
    async def update_role(
        request: Request,
        role_id: str,
        body: UpdateRoleRequest,
        role_service: Any = Depends(_get_role_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> RoleResponse:
        """Update an existing role.

        Args:
            request: The incoming HTTP request.
            role_id: Role UUID.
            body: Update request with fields to change.
            role_service: Injected RoleService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Returns:
            Updated role.

        Raises:
            HTTPException 404: If the role does not exist.
            HTTPException 403: If attempting to modify a system role.
            HTTPException 422: If no fields are provided.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Check role exists and fetch old state
        existing_role = await role_service.get_role(role_id)
        if existing_role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        # Prevent modification of system roles
        if existing_role.get("is_system", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify a system-defined role.",
            )

        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No fields provided for update.",
            )

        # Handle enable/disable through dedicated methods
        if "is_enabled" in updates:
            is_enabled = updates.pop("is_enabled")
            if is_enabled:
                await role_service.enable_role(role_id)
                await audit.log_role_enabled(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    role_id=role_id,
                    ip_address=client_ip,
                    correlation_id=correlation_id,
                )
            else:
                await role_service.disable_role(role_id)
                await audit.log_role_disabled(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    role_id=role_id,
                    ip_address=client_ip,
                    correlation_id=correlation_id,
                )

        # Apply remaining field updates if any
        if updates:
            try:
                updated_role = await role_service.update_role(
                    role_id, **updates
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc),
                )

            await audit.log_role_updated(
                tenant_id=tenant_id,
                actor_id=actor_id,
                role_id=role_id,
                old_value=existing_role,
                new_value=updated_role,
                ip_address=client_ip,
                correlation_id=correlation_id,
            )
            metrics.record_role_change("updated", tenant_id or "global")

            return RoleResponse(**updated_role)

        # If only is_enabled was changed, re-fetch
        refreshed = await role_service.get_role(role_id)
        return RoleResponse(**refreshed)

    # -- Delete Role -------------------------------------------------------

    @roles_router.delete(
        "/{role_id}",
        status_code=204,
        summary="Delete role",
        description="Delete an RBAC role. System roles cannot be deleted.",
        operation_id="delete_role",
    )
    async def delete_role(
        request: Request,
        role_id: str,
        role_service: Any = Depends(_get_role_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> None:
        """Delete a role by UUID.

        Args:
            request: The incoming HTTP request.
            role_id: Role UUID.
            role_service: Injected RoleService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Raises:
            HTTPException 404: If the role does not exist.
            HTTPException 403: If attempting to delete a system role.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Check role exists
        existing_role = await role_service.get_role(role_id)
        if existing_role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        # Prevent deletion of system roles
        if existing_role.get("is_system", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete a system-defined role.",
            )

        result = await role_service.delete_role(role_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        await audit.log_role_deleted(
            tenant_id=tenant_id,
            actor_id=actor_id,
            role_id=role_id,
            role_name=existing_role.get("name", ""),
            ip_address=client_ip,
            correlation_id=correlation_id,
        )
        metrics.record_role_change("deleted", tenant_id or "global")

        logger.info(
            "API: Deleted role '%s' (id=%s) by actor %s",
            existing_role.get("name"),
            role_id,
            actor_id,
        )

    # -- List Role Permissions ---------------------------------------------

    @roles_router.get(
        "/{role_id}/permissions",
        response_model=RolePermissionsResponse,
        summary="List role permissions",
        description="List all permissions assigned to a role, optionally including inherited permissions.",
        operation_id="list_role_permissions",
    )
    async def list_role_permissions(
        role_id: str,
        include_inherited: bool = Query(
            True, description="Include permissions inherited from parent roles."
        ),
        role_service: Any = Depends(_get_role_service),
        permission_service: Any = Depends(_get_permission_service),
    ) -> RolePermissionsResponse:
        """List permissions for a specific role.

        Args:
            role_id: Role UUID.
            include_inherited: Whether to include inherited permissions.
            role_service: Injected RoleService.
            permission_service: Injected PermissionService.

        Returns:
            List of permissions for the role.

        Raises:
            HTTPException 404: If the role does not exist.
        """
        # Verify role exists
        role = await role_service.get_role(role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{role_id}' not found.",
            )

        permissions = await permission_service.get_role_permissions(
            role_id, include_inherited=include_inherited
        )

        return RolePermissionsResponse(
            role_id=role_id,
            permissions=permissions,
            include_inherited=include_inherited,
        )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(roles_router)
    except ImportError:
        pass  # auth_service not available

else:
    roles_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - roles_router is None")


__all__ = ["roles_router"]
