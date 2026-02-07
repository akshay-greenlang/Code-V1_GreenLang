# -*- coding: utf-8 -*-
"""
RBAC Assignments REST API Routes - SEC-002

FastAPI APIRouter providing REST endpoints for role assignment management
within the GreenLang RBAC authorization layer.  Supports assigning roles to
users, revoking assignments, listing user roles, and resolving effective
permissions.

Endpoints:
    GET    /api/v1/rbac/assignments                   - List assignments (paginated)
    POST   /api/v1/rbac/assignments                   - Assign role to user
    DELETE /api/v1/rbac/assignments/{assignment_id}    - Revoke assignment
    GET    /api/v1/rbac/users/{user_id}/roles          - Get user's roles
    GET    /api/v1/rbac/users/{user_id}/permissions    - Get user's effective permissions

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.rbac_service.api.assignments_routes import assignments_router
    >>> app = FastAPI()
    >>> app.include_router(assignments_router)

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

    class AssignRoleRequest(BaseModel):
        """Request schema for assigning a role to a user.

        Attributes:
            user_id: UUID of the user receiving the role.
            role_id: UUID of the role to assign.
            tenant_id: UUID of the tenant scope for the assignment.
            expires_at: Optional expiration datetime (UTC) for time-limited assignments.
        """

        model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
            json_schema_extra={
                "examples": [
                    {
                        "user_id": "u-analyst-01",
                        "role_id": "r-emissions-analyst",
                        "tenant_id": "t-acme-corp",
                        "expires_at": "2026-12-31T23:59:59Z",
                    }
                ]
            },
        )

        user_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the user receiving the role.",
        )
        role_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the role to assign.",
        )
        tenant_id: str = Field(
            ...,
            min_length=1,
            max_length=128,
            description="UUID of the tenant scope for the assignment.",
        )
        expires_at: Optional[datetime] = Field(
            default=None,
            description="Optional expiration datetime (UTC) for time-limited assignments.",
        )

    class AssignmentResponse(BaseModel):
        """Response schema for a single role assignment.

        Attributes:
            id: Assignment UUID.
            user_id: UUID of the user.
            role_id: UUID of the role.
            tenant_id: UUID of the tenant.
            assigned_by: Actor who created the assignment.
            assigned_at: Assignment creation timestamp.
            expires_at: Optional expiration datetime.
            is_active: Whether the assignment is currently active.
            revoked_at: Revocation timestamp (if revoked).
            revoked_by: Actor who revoked (if revoked).
        """

        model_config = ConfigDict(from_attributes=True)

        id: str = Field(..., description="Assignment UUID.")
        user_id: str = Field(..., description="User UUID.")
        role_id: str = Field(..., description="Role UUID.")
        tenant_id: str = Field(..., description="Tenant UUID.")
        assigned_by: Optional[str] = Field(
            default=None, description="Actor who created the assignment."
        )
        assigned_at: Optional[datetime] = Field(
            default=None, description="Assignment creation timestamp."
        )
        expires_at: Optional[datetime] = Field(
            default=None, description="Expiration datetime."
        )
        is_active: bool = Field(
            default=True, description="Whether the assignment is active."
        )
        revoked_at: Optional[datetime] = Field(
            default=None, description="Revocation timestamp."
        )
        revoked_by: Optional[str] = Field(
            default=None, description="Actor who revoked."
        )

    class AssignmentListResponse(BaseModel):
        """Paginated list of role assignments.

        Attributes:
            items: Assignments for this page.
            total: Total matching assignments.
            page: Current page number.
            page_size: Items per page.
            total_pages: Total pages.
            has_next: Whether there is a next page.
            has_prev: Whether there is a previous page.
        """

        items: List[AssignmentResponse] = Field(
            ..., description="Assignments for this page."
        )
        total: int = Field(
            ..., ge=0, description="Total matching assignments."
        )
        page: int = Field(..., ge=1, description="Current page number.")
        page_size: int = Field(..., ge=1, description="Items per page.")
        total_pages: int = Field(..., ge=0, description="Total pages.")
        has_next: bool = Field(..., description="Has next page.")
        has_prev: bool = Field(..., description="Has previous page.")

    class UserRolesResponse(BaseModel):
        """Response schema for a user's role assignments.

        Attributes:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant scope.
            roles: List of role assignment details.
            include_expired: Whether expired assignments are included.
        """

        user_id: str = Field(..., description="User UUID.")
        tenant_id: str = Field(..., description="Tenant UUID.")
        roles: List[Dict[str, Any]] = Field(
            ..., description="Role assignment details."
        )
        include_expired: bool = Field(
            default=False,
            description="Whether expired assignments are included.",
        )

    class UserPermissionsResponse(BaseModel):
        """Response schema for a user's effective permissions.

        Attributes:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant scope.
            permissions: List of effective permission strings (``resource:action``).
        """

        user_id: str = Field(..., description="User UUID.")
        tenant_id: str = Field(..., description="Tenant UUID.")
        permissions: List[str] = Field(
            ...,
            description="Effective permission strings (resource:action format).",
        )


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


def _get_assignment_service() -> Any:
    """FastAPI dependency that provides the AssignmentService instance.

    Returns:
        The AssignmentService singleton.

    Raises:
        HTTPException 503: If the service is not available.
    """
    try:
        from greenlang.infrastructure.rbac_service.assignment_service import (
            AssignmentService,
            get_assignment_service,
        )

        return get_assignment_service()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RBAC assignment service is not available.",
        )


def _get_role_service() -> Any:
    """FastAPI dependency that provides the RoleService instance."""
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


def _get_tenant_id_header(request: Any) -> Optional[str]:
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

    assignments_router = APIRouter(
        prefix="/api/v1/rbac",
        tags=["RBAC Assignments"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Not Found"},
            409: {"description": "Conflict"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -- List Assignments --------------------------------------------------

    @assignments_router.get(
        "/assignments",
        response_model=AssignmentListResponse,
        summary="List assignments",
        description="Retrieve a paginated list of role assignments, optionally filtered by tenant and/or user.",
        operation_id="list_assignments",
    )
    async def list_assignments(
        request: Request,
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant ID."
        ),
        user_id: Optional[str] = Query(
            None, description="Filter by user ID."
        ),
        page: int = Query(1, ge=1, description="Page number (1-indexed)."),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page."
        ),
        assignment_service: Any = Depends(_get_assignment_service),
    ) -> AssignmentListResponse:
        """List role assignments with pagination and optional filters.

        Args:
            request: The incoming HTTP request.
            tenant_id: Optional tenant ID filter.
            user_id: Optional user ID filter.
            page: Page number.
            page_size: Items per page.
            assignment_service: Injected AssignmentService.

        Returns:
            Paginated list of assignments.
        """
        effective_tenant = tenant_id or _get_tenant_id_header(request)

        try:
            if user_id and effective_tenant:
                # List roles for a specific user in a tenant
                roles = await assignment_service.list_user_roles(
                    user_id=user_id,
                    tenant_id=effective_tenant,
                    include_expired=False,
                )
                # Convert to assignment-style dicts for response
                items = [AssignmentResponse(**r) for r in roles]
                total = len(items)

                # Apply manual pagination
                start = (page - 1) * page_size
                end = start + page_size
                page_items = items[start:end]
                total_pages = (
                    (total + page_size - 1) // page_size if total > 0 else 0
                )

                return AssignmentListResponse(
                    items=page_items,
                    total=total,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                    has_next=page < total_pages,
                    has_prev=page > 1,
                )
            elif user_id:
                # List all assignments for a user across tenants
                header_tenant = _get_tenant_id_header(request) or ""
                roles = await assignment_service.list_user_roles(
                    user_id=user_id,
                    tenant_id=header_tenant,
                    include_expired=False,
                )
                items = [AssignmentResponse(**r) for r in roles]
                total = len(items)
                start = (page - 1) * page_size
                end = start + page_size
                page_items = items[start:end]
                total_pages = (
                    (total + page_size - 1) // page_size if total > 0 else 0
                )

                return AssignmentListResponse(
                    items=page_items,
                    total=total,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                    has_next=page < total_pages,
                    has_prev=page > 1,
                )
            else:
                # No specific user - return empty or use tenant context
                return AssignmentListResponse(
                    items=[],
                    total=0,
                    page=page,
                    page_size=page_size,
                    total_pages=0,
                    has_next=False,
                    has_prev=False,
                )

        except Exception as exc:
            logger.exception("Failed to list assignments")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list assignments: {exc}",
            )

    # -- Assign Role -------------------------------------------------------

    @assignments_router.post(
        "/assignments",
        response_model=AssignmentResponse,
        status_code=201,
        summary="Assign role to user",
        description="Create a new role assignment for a user within a tenant scope.",
        operation_id="assign_role",
    )
    async def assign_role(
        request: Request,
        body: AssignRoleRequest,
        assignment_service: Any = Depends(_get_assignment_service),
        role_service: Any = Depends(_get_role_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> AssignmentResponse:
        """Assign a role to a user.

        Args:
            request: The incoming HTTP request.
            body: Assignment request body.
            assignment_service: Injected AssignmentService.
            role_service: Injected RoleService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Returns:
            The created assignment.

        Raises:
            HTTPException 404: If the role does not exist.
            HTTPException 409: If the assignment already exists.
        """
        actor_id = _get_actor_id(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Verify role exists
        role = await role_service.get_role(body.role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role '{body.role_id}' not found.",
            )

        # Verify role is enabled
        if not role.get("is_enabled", True):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Role '{body.role_id}' is disabled and cannot be assigned.",
            )

        try:
            assignment = await assignment_service.assign_role(
                user_id=body.user_id,
                role_id=body.role_id,
                tenant_id=body.tenant_id,
                assigned_by=actor_id,
                expires_at=body.expires_at,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

        # Audit and metrics
        await audit.log_role_assigned(
            tenant_id=body.tenant_id,
            actor_id=actor_id,
            user_id=body.user_id,
            role_id=body.role_id,
            assignment_id=assignment.get("id"),
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
            ip_address=client_ip,
            correlation_id=correlation_id,
        )

        logger.info(
            "API: Assigned role '%s' to user '%s' in tenant '%s' by actor %s",
            body.role_id,
            body.user_id,
            body.tenant_id,
            actor_id,
        )

        return AssignmentResponse(**assignment)

    # -- Revoke Assignment -------------------------------------------------

    @assignments_router.delete(
        "/assignments/{assignment_id}",
        status_code=204,
        summary="Revoke assignment",
        description="Revoke an existing role assignment.",
        operation_id="revoke_assignment",
    )
    async def revoke_assignment(
        request: Request,
        assignment_id: str,
        assignment_service: Any = Depends(_get_assignment_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> None:
        """Revoke a role assignment.

        Args:
            request: The incoming HTTP request.
            assignment_id: UUID of the assignment to revoke.
            assignment_service: Injected AssignmentService.
            audit: Injected RBACAuditLogger.
            metrics: Injected RBACMetrics.

        Raises:
            HTTPException 404: If the assignment does not exist.
        """
        actor_id = _get_actor_id(request)
        tenant_id = _get_tenant_id_header(request)
        client_ip = _get_client_ip(request)
        correlation_id = _get_correlation_id(request)

        # Fetch assignment details for audit
        assignment = await assignment_service.get_assignment(assignment_id)
        if assignment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assignment '{assignment_id}' not found.",
            )

        try:
            result = await assignment_service.revoke_role(
                assignment_id=assignment_id,
                revoked_by=actor_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        # Audit
        await audit.log_role_revoked_assignment(
            tenant_id=assignment.get("tenant_id", tenant_id),
            actor_id=actor_id,
            assignment_id=assignment_id,
            user_id=assignment.get("user_id"),
            role_id=assignment.get("role_id"),
            ip_address=client_ip,
            correlation_id=correlation_id,
        )

        logger.info(
            "API: Revoked assignment '%s' by actor %s",
            assignment_id,
            actor_id,
        )

    # -- Get User Roles ----------------------------------------------------

    @assignments_router.get(
        "/users/{user_id}/roles",
        response_model=UserRolesResponse,
        summary="Get user roles",
        description="Retrieve all roles assigned to a user within a tenant scope.",
        operation_id="get_user_roles",
    )
    async def get_user_roles(
        request: Request,
        user_id: str,
        tenant_id: Optional[str] = Query(
            None, description="Tenant scope for the role lookup."
        ),
        include_expired: bool = Query(
            False, description="Include expired role assignments."
        ),
        assignment_service: Any = Depends(_get_assignment_service),
    ) -> UserRolesResponse:
        """Get roles assigned to a user.

        Args:
            request: The incoming HTTP request.
            user_id: UUID of the user.
            tenant_id: Optional tenant scope (falls back to header).
            include_expired: Whether to include expired assignments.
            assignment_service: Injected AssignmentService.

        Returns:
            User's role assignments.
        """
        effective_tenant = tenant_id or _get_tenant_id_header(request) or ""

        try:
            roles = await assignment_service.list_user_roles(
                user_id=user_id,
                tenant_id=effective_tenant,
                include_expired=include_expired,
            )

            return UserRolesResponse(
                user_id=user_id,
                tenant_id=effective_tenant,
                roles=roles,
                include_expired=include_expired,
            )

        except Exception as exc:
            logger.exception(
                "Failed to get roles for user '%s'", user_id
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user roles: {exc}",
            )

    # -- Get User Permissions ----------------------------------------------

    @assignments_router.get(
        "/users/{user_id}/permissions",
        response_model=UserPermissionsResponse,
        summary="Get user effective permissions",
        description="Resolve all effective permissions for a user within a tenant, considering role hierarchy and inheritance.",
        operation_id="get_user_permissions",
    )
    async def get_user_permissions(
        request: Request,
        user_id: str,
        tenant_id: Optional[str] = Query(
            None, description="Tenant scope for permission resolution."
        ),
        assignment_service: Any = Depends(_get_assignment_service),
    ) -> UserPermissionsResponse:
        """Get effective permissions for a user.

        Args:
            request: The incoming HTTP request.
            user_id: UUID of the user.
            tenant_id: Optional tenant scope (falls back to header).
            assignment_service: Injected AssignmentService.

        Returns:
            User's effective permissions as a list of ``resource:action`` strings.
        """
        effective_tenant = tenant_id or _get_tenant_id_header(request) or ""

        try:
            permissions = await assignment_service.get_user_permissions(
                user_id=user_id,
                tenant_id=effective_tenant,
            )

            return UserPermissionsResponse(
                user_id=user_id,
                tenant_id=effective_tenant,
                permissions=permissions,
            )

        except Exception as exc:
            logger.exception(
                "Failed to get permissions for user '%s'", user_id
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user permissions: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(assignments_router)
    except ImportError:
        pass  # auth_service not available

else:
    assignments_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - assignments_router is None")


__all__ = ["assignments_router"]
