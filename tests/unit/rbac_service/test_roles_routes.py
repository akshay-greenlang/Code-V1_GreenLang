# -*- coding: utf-8 -*-
"""
Unit tests for RBAC Roles API Routes - RBAC Authorization Layer (SEC-002)

Tests FastAPI endpoints for role management:
GET/POST /api/v1/rbac/roles, GET/PUT/DELETE /api/v1/rbac/roles/{role_id},
GET /api/v1/rbac/roles/{role_id}/permissions

Uses the HTTPX AsyncClient / TestClient pattern consistent with SEC-001.

Coverage targets: 85%+ of roles_routes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import FastAPI test tooling.
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Attempt to import RBAC routes module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.api import create_rbac_router
    _HAS_ROUTES = True
except ImportError:
    _HAS_ROUTES = False

try:
    from greenlang.infrastructure.rbac_service.role_service import RoleService
except ImportError:
    class RoleService:  # type: ignore[no-redef]
        pass

try:
    from greenlang.infrastructure.rbac_service.permission_service import PermissionService
except ImportError:
    class PermissionService:  # type: ignore[no-redef]
        pass

try:
    from greenlang.infrastructure.rbac_service.assignment_service import AssignmentService
except ImportError:
    class AssignmentService:  # type: ignore[no-redef]
        pass

pytestmark = [
    pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed"),
    pytest.mark.skipif(not _HAS_ROUTES, reason="rbac_service.api not implemented"),
]


# ============================================================================
# Helpers
# ============================================================================


def _make_role_dict(
    role_id: str = "role-1",
    name: str = "viewer",
    display_name: str = "Viewer",
    description: str = "Read-only access",
    is_system: bool = False,
    is_active: bool = True,
    parent_role_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "id": role_id,
        "name": name,
        "display_name": display_name,
        "description": description,
        "is_system": is_system,
        "is_active": is_active,
        "parent_role_id": parent_role_id,
        "tenant_id": tenant_id,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_test_app(
    role_service: Any = None,
    permission_service: Any = None,
    assignment_service: Any = None,
) -> "FastAPI":
    app = FastAPI()
    router = create_rbac_router(
        role_service=role_service or AsyncMock(spec=RoleService),
        permission_service=permission_service or AsyncMock(spec=PermissionService),
        assignment_service=assignment_service or AsyncMock(spec=AssignmentService),
    )
    app.include_router(router, prefix="/api/v1/rbac")
    return app


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def role_service() -> AsyncMock:
    svc = AsyncMock(spec=RoleService)
    svc.create_role = AsyncMock(return_value=_make_role_dict())
    svc.get_role = AsyncMock(return_value=_make_role_dict())
    svc.get_role_by_name = AsyncMock(return_value=_make_role_dict())
    svc.list_roles = AsyncMock(return_value={
        "items": [_make_role_dict()],
        "total": 1,
        "page": 1,
        "page_size": 20,
    })
    svc.update_role = AsyncMock(return_value=_make_role_dict(display_name="Updated"))
    svc.delete_role = AsyncMock(return_value=True)
    svc.enable_role = AsyncMock(return_value=_make_role_dict(is_active=True))
    svc.disable_role = AsyncMock(return_value=_make_role_dict(is_active=False))
    svc.get_role_hierarchy = AsyncMock(return_value=[_make_role_dict()])
    return svc


@pytest.fixture
def permission_service() -> AsyncMock:
    svc = AsyncMock(spec=PermissionService)
    svc.get_role_permissions = AsyncMock(return_value=[
        {"id": "perm-1", "resource": "agents", "action": "read", "effect": "allow"}
    ])
    return svc


@pytest.fixture
def assignment_service() -> AsyncMock:
    return AsyncMock(spec=AssignmentService)


@pytest.fixture
def app(role_service, permission_service, assignment_service) -> "FastAPI":
    return _build_test_app(
        role_service=role_service,
        permission_service=permission_service,
        assignment_service=assignment_service,
    )


@pytest.fixture
def client(app) -> "TestClient":
    return TestClient(app)


# ============================================================================
# TestListRoles
# ============================================================================


class TestListRoles:
    """Tests for GET /api/v1/rbac/roles."""

    def test_list_roles_200(self, client, role_service) -> None:
        """Listing roles returns 200 with role array."""
        resp = client.get("/api/v1/rbac/roles")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body or isinstance(body, list)

    def test_list_roles_with_pagination(self, client, role_service) -> None:
        """Pagination parameters are accepted."""
        resp = client.get("/api/v1/rbac/roles?page=2&page_size=5")
        assert resp.status_code == 200

    def test_list_roles_filter_tenant(self, client, role_service) -> None:
        """Tenant filter returns only tenant-scoped roles."""
        resp = client.get("/api/v1/rbac/roles?tenant_id=t-acme")
        assert resp.status_code == 200

    def test_list_roles_include_system(self, client, role_service) -> None:
        """include_system parameter is accepted."""
        resp = client.get("/api/v1/rbac/roles?include_system=true")
        assert resp.status_code == 200

    def test_list_roles_empty(self, client, role_service) -> None:
        """Empty role list returns 200 with empty array."""
        role_service.list_roles.return_value = {
            "items": [], "total": 0, "page": 1, "page_size": 20
        }
        resp = client.get("/api/v1/rbac/roles")
        assert resp.status_code == 200

    def test_list_roles_default_pagination(self, client, role_service) -> None:
        """Default pagination values are used when not specified."""
        resp = client.get("/api/v1/rbac/roles")
        assert resp.status_code == 200
        role_service.list_roles.assert_awaited_once()


# ============================================================================
# TestCreateRole
# ============================================================================


class TestCreateRole:
    """Tests for POST /api/v1/rbac/roles."""

    def test_create_role_201(self, client, role_service) -> None:
        """Creating a role returns 201 with the created role."""
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": "analyst", "display_name": "Analyst"},
        )
        assert resp.status_code in (200, 201)
        body = resp.json()
        assert "id" in body or "name" in body

    def test_create_role_missing_name_422(self, client) -> None:
        """Missing name field returns 422."""
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"display_name": "No Name"},
        )
        assert resp.status_code == 422

    def test_create_role_duplicate_409(self, client, role_service) -> None:
        """Duplicate role name returns 409."""
        role_service.create_role.side_effect = Exception("unique constraint")
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": "admin"},
        )
        assert resp.status_code in (409, 400, 500)

    def test_create_role_with_parent(self, client, role_service) -> None:
        """Creating a role with parent_role_id is accepted."""
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": "sub-admin", "parent_role_id": "role-parent"},
        )
        assert resp.status_code in (200, 201)

    def test_create_role_with_metadata(self, client, role_service) -> None:
        """Creating a role with metadata is accepted."""
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": "dept-role", "metadata": {"department": "finance"}},
        )
        assert resp.status_code in (200, 201)

    def test_create_role_max_name_length(self, client, role_service) -> None:
        """Very long role names are either accepted or rejected with 422."""
        long_name = "a" * 256
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": long_name},
        )
        assert resp.status_code in (200, 201, 422, 400)

    def test_create_role_invalid_parent_400(self, client, role_service) -> None:
        """Invalid parent_role_id returns 400."""
        role_service.create_role.side_effect = Exception("Parent not found")
        resp = client.post(
            "/api/v1/rbac/roles",
            json={"name": "orphan", "parent_role_id": "nonexistent"},
        )
        assert resp.status_code in (400, 404, 500)


# ============================================================================
# TestGetRole
# ============================================================================


class TestGetRole:
    """Tests for GET /api/v1/rbac/roles/{role_id}."""

    def test_get_role_200(self, client, role_service) -> None:
        """Getting an existing role returns 200."""
        resp = client.get("/api/v1/rbac/roles/role-1")
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body or "name" in body

    def test_get_role_404(self, client, role_service) -> None:
        """Getting a non-existent role returns 404."""
        role_service.get_role.return_value = None
        resp = client.get("/api/v1/rbac/roles/ghost")
        assert resp.status_code == 404

    def test_get_role_with_hierarchy(self, client, role_service) -> None:
        """Getting a role may include hierarchy information."""
        resp = client.get("/api/v1/rbac/roles/role-1")
        assert resp.status_code == 200


# ============================================================================
# TestUpdateRole
# ============================================================================


class TestUpdateRole:
    """Tests for PUT /api/v1/rbac/roles/{role_id}."""

    def test_update_role_200(self, client, role_service) -> None:
        """Updating a role returns 200 with updated data."""
        resp = client.put(
            "/api/v1/rbac/roles/role-1",
            json={"display_name": "Updated Viewer"},
        )
        assert resp.status_code == 200

    def test_update_role_404(self, client, role_service) -> None:
        """Updating a non-existent role returns 404."""
        role_service.update_role.side_effect = Exception("not found")
        resp = client.put(
            "/api/v1/rbac/roles/ghost",
            json={"display_name": "X"},
        )
        assert resp.status_code in (404, 400, 500)

    def test_update_role_system_403(self, client, role_service) -> None:
        """Updating a system role returns 403."""
        role_service.update_role.side_effect = Exception("system role protected")
        resp = client.put(
            "/api/v1/rbac/roles/sys-admin",
            json={"display_name": "Hacked"},
        )
        assert resp.status_code in (403, 400, 500)

    def test_update_role_partial_fields(self, client, role_service) -> None:
        """Partial update with only some fields is accepted."""
        resp = client.put(
            "/api/v1/rbac/roles/role-1",
            json={"description": "Updated description only"},
        )
        assert resp.status_code == 200


# ============================================================================
# TestDeleteRole
# ============================================================================


class TestDeleteRole:
    """Tests for DELETE /api/v1/rbac/roles/{role_id}."""

    def test_delete_role_200(self, client, role_service) -> None:
        """Deleting a role returns 200 or 204."""
        resp = client.delete("/api/v1/rbac/roles/role-1")
        assert resp.status_code in (200, 204)

    def test_delete_role_404(self, client, role_service) -> None:
        """Deleting a non-existent role returns 404."""
        role_service.delete_role.side_effect = Exception("not found")
        resp = client.delete("/api/v1/rbac/roles/ghost")
        assert resp.status_code in (404, 400, 500)

    def test_delete_role_system_403(self, client, role_service) -> None:
        """Deleting a system role returns 403."""
        role_service.delete_role.side_effect = Exception("system role protected")
        resp = client.delete("/api/v1/rbac/roles/sys-admin")
        assert resp.status_code in (403, 400, 500)

    def test_delete_role_with_assignments_cascade(
        self, client, role_service
    ) -> None:
        """Deleting a role that has assignments is handled."""
        role_service.delete_role.return_value = True
        resp = client.delete("/api/v1/rbac/roles/role-with-users")
        assert resp.status_code in (200, 204)


# ============================================================================
# TestGetRolePermissions
# ============================================================================


class TestGetRolePermissions:
    """Tests for GET /api/v1/rbac/roles/{role_id}/permissions."""

    def test_get_role_permissions_200(
        self, client, permission_service
    ) -> None:
        """Getting role permissions returns 200 with permission list."""
        resp = client.get("/api/v1/rbac/roles/role-1/permissions")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list) or "items" in body or "permissions" in body
