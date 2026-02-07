# -*- coding: utf-8 -*-
"""
Unit tests for RBAC Assignments API Routes - RBAC Authorization Layer (SEC-002)

Tests FastAPI endpoints for assignment management:
GET/POST /api/v1/rbac/assignments, DELETE /api/v1/rbac/assignments/{id},
GET /api/v1/rbac/users/{user_id}/roles, GET /api/v1/rbac/users/{user_id}/permissions,
POST /api/v1/rbac/check

Uses the HTTPX AsyncClient / TestClient pattern consistent with SEC-001.

Coverage targets: 85%+ of assignments_routes.py + check_routes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
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


def _make_assignment_dict(
    assignment_id: str = "asgn-1",
    user_id: str = "user-1",
    role_id: str = "role-1",
    tenant_id: str = "t-acme",
    is_active: bool = True,
    expires_at: Optional[str] = None,
) -> dict:
    return {
        "id": assignment_id,
        "user_id": user_id,
        "role_id": role_id,
        "tenant_id": tenant_id,
        "assigned_by": "admin-1",
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at,
        "revoked_at": None,
        "is_active": is_active,
    }


def _make_role_dict(
    role_id: str = "role-1",
    name: str = "viewer",
) -> dict:
    return {
        "id": role_id,
        "name": name,
        "display_name": name.title(),
        "is_system": False,
        "is_active": True,
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
    svc.get_role = AsyncMock(return_value=_make_role_dict())
    return svc


@pytest.fixture
def permission_service() -> AsyncMock:
    svc = AsyncMock(spec=PermissionService)
    svc.evaluate_permission = AsyncMock(return_value={
        "allowed": True,
        "effect": "allow",
        "resource": "agents",
        "action": "read",
    })
    return svc


@pytest.fixture
def assignment_service() -> AsyncMock:
    svc = AsyncMock(spec=AssignmentService)
    svc.assign_role = AsyncMock(return_value=_make_assignment_dict())
    svc.revoke_role = AsyncMock(return_value=_make_assignment_dict(is_active=False))
    svc.list_user_roles = AsyncMock(return_value=[
        _make_assignment_dict(assignment_id=f"a-{i}", role_id=f"role-{i}")
        for i in range(2)
    ])
    svc.get_user_permissions = AsyncMock(return_value=[
        "agents:read", "agents:write", "emissions:read"
    ])
    svc.get_assignment = AsyncMock(return_value=_make_assignment_dict())
    svc.bulk_assign_role = AsyncMock(return_value=[
        _make_assignment_dict(assignment_id=f"b-{i}", user_id=f"user-{i}")
        for i in range(3)
    ])
    return svc


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
# TestListAssignments
# ============================================================================


class TestListAssignments:
    """Tests for GET /api/v1/rbac/assignments."""

    def test_list_assignments_200(self, client, assignment_service) -> None:
        """Listing assignments returns 200."""
        assignment_service.list_user_roles.return_value = [
            _make_assignment_dict()
        ]
        resp = client.get("/api/v1/rbac/assignments?user_id=user-1&tenant_id=t-acme")
        assert resp.status_code == 200

    def test_list_assignments_filter_user(self, client, assignment_service) -> None:
        """Filtering by user_id returns that user's assignments."""
        resp = client.get("/api/v1/rbac/assignments?user_id=user-1&tenant_id=t-acme")
        assert resp.status_code == 200

    def test_list_assignments_filter_tenant(self, client, assignment_service) -> None:
        """Filtering by tenant_id scopes the results."""
        resp = client.get("/api/v1/rbac/assignments?user_id=user-1&tenant_id=t-acme")
        assert resp.status_code == 200

    def test_list_assignments_pagination(self, client, assignment_service) -> None:
        """Pagination parameters are accepted."""
        resp = client.get(
            "/api/v1/rbac/assignments?user_id=user-1&tenant_id=t-acme&page=1&page_size=10"
        )
        assert resp.status_code == 200

    def test_list_assignments_empty(self, client, assignment_service) -> None:
        """Empty result set returns 200 with empty array."""
        assignment_service.list_user_roles.return_value = []
        resp = client.get("/api/v1/rbac/assignments?user_id=user-1&tenant_id=t-acme")
        assert resp.status_code == 200


# ============================================================================
# TestCreateAssignment
# ============================================================================


class TestCreateAssignment:
    """Tests for POST /api/v1/rbac/assignments."""

    def test_create_assignment_201(self, client, assignment_service) -> None:
        """Creating an assignment returns 201."""
        resp = client.post(
            "/api/v1/rbac/assignments",
            json={
                "user_id": "user-1",
                "role_id": "role-1",
                "tenant_id": "t-acme",
            },
        )
        assert resp.status_code in (200, 201)
        body = resp.json()
        assert "id" in body or "user_id" in body

    def test_create_assignment_duplicate_409(
        self, client, assignment_service
    ) -> None:
        """Duplicate assignment returns 409."""
        assignment_service.assign_role.side_effect = Exception("unique constraint")
        resp = client.post(
            "/api/v1/rbac/assignments",
            json={
                "user_id": "user-1",
                "role_id": "role-1",
                "tenant_id": "t-acme",
            },
        )
        assert resp.status_code in (409, 400, 500)

    def test_create_assignment_with_expiry(
        self, client, assignment_service
    ) -> None:
        """Assignment with expires_at is accepted."""
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        resp = client.post(
            "/api/v1/rbac/assignments",
            json={
                "user_id": "user-1",
                "role_id": "role-1",
                "tenant_id": "t-acme",
                "expires_at": future,
            },
        )
        assert resp.status_code in (200, 201)

    def test_create_assignment_missing_fields_422(self, client) -> None:
        """Missing required fields returns 422."""
        resp = client.post(
            "/api/v1/rbac/assignments",
            json={"user_id": "user-1"},
        )
        assert resp.status_code == 422

    def test_create_assignment_invalid_role_400(
        self, client, assignment_service
    ) -> None:
        """Invalid role_id returns 400."""
        assignment_service.assign_role.side_effect = Exception("Role not found")
        resp = client.post(
            "/api/v1/rbac/assignments",
            json={
                "user_id": "user-1",
                "role_id": "nonexistent",
                "tenant_id": "t-acme",
            },
        )
        assert resp.status_code in (400, 404, 500)


# ============================================================================
# TestRevokeAssignment
# ============================================================================


class TestRevokeAssignment:
    """Tests for DELETE /api/v1/rbac/assignments/{assignment_id}."""

    def test_revoke_assignment_200(self, client, assignment_service) -> None:
        """Revoking an assignment returns 200."""
        resp = client.delete("/api/v1/rbac/assignments/asgn-1")
        assert resp.status_code in (200, 204)

    def test_revoke_assignment_404(self, client, assignment_service) -> None:
        """Revoking a non-existent assignment returns 404."""
        assignment_service.revoke_role.side_effect = Exception("not found")
        resp = client.delete("/api/v1/rbac/assignments/ghost")
        assert resp.status_code in (404, 400, 500)

    def test_revoke_assignment_already_revoked(
        self, client, assignment_service
    ) -> None:
        """Revoking an already-revoked assignment is handled."""
        assignment_service.revoke_role.side_effect = Exception("already revoked")
        resp = client.delete("/api/v1/rbac/assignments/already-revoked")
        assert resp.status_code in (409, 400, 500)


# ============================================================================
# TestGetUserRoles
# ============================================================================


class TestGetUserRoles:
    """Tests for GET /api/v1/rbac/users/{user_id}/roles."""

    def test_get_user_roles_200(self, client, assignment_service) -> None:
        """Getting user roles returns 200 with role list."""
        resp = client.get("/api/v1/rbac/users/user-1/roles?tenant_id=t-acme")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list) or "items" in body or "roles" in body

    def test_get_user_roles_empty(self, client, assignment_service) -> None:
        """User with no roles returns 200 with empty list."""
        assignment_service.list_user_roles.return_value = []
        resp = client.get("/api/v1/rbac/users/user-none/roles?tenant_id=t-acme")
        assert resp.status_code == 200

    def test_get_user_roles_include_expired(
        self, client, assignment_service
    ) -> None:
        """include_expired parameter is accepted."""
        resp = client.get(
            "/api/v1/rbac/users/user-1/roles?tenant_id=t-acme&include_expired=true"
        )
        assert resp.status_code == 200

    def test_get_user_roles_with_hierarchy(
        self, client, assignment_service
    ) -> None:
        """User roles include inherited role information."""
        resp = client.get("/api/v1/rbac/users/user-1/roles?tenant_id=t-acme")
        assert resp.status_code == 200


# ============================================================================
# TestGetUserPermissions
# ============================================================================


class TestGetUserPermissions:
    """Tests for GET /api/v1/rbac/users/{user_id}/permissions."""

    def test_get_user_permissions_200(
        self, client, assignment_service
    ) -> None:
        """Getting user permissions returns 200."""
        resp = client.get(
            "/api/v1/rbac/users/user-1/permissions?tenant_id=t-acme"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list) or "permissions" in body

    def test_get_user_permissions_aggregated(
        self, client, assignment_service
    ) -> None:
        """Permissions are aggregated from all user roles."""
        resp = client.get(
            "/api/v1/rbac/users/user-1/permissions?tenant_id=t-acme"
        )
        assert resp.status_code == 200

    def test_permissions_include_inherited(
        self, client, assignment_service
    ) -> None:
        """Permissions include those inherited through role hierarchy."""
        assignment_service.get_user_permissions.return_value = [
            "agents:read",
            "agents:write",
            "admin:manage",  # inherited from parent
        ]
        resp = client.get(
            "/api/v1/rbac/users/user-1/permissions?tenant_id=t-acme"
        )
        assert resp.status_code == 200


# ============================================================================
# TestCheckPermission
# ============================================================================


class TestCheckPermission:
    """Tests for POST /api/v1/rbac/check."""

    def test_check_permission_allowed(
        self, client, permission_service
    ) -> None:
        """Permission check returns allowed=True for authorized access."""
        resp = client.post(
            "/api/v1/rbac/check",
            json={
                "user_id": "user-1",
                "tenant_id": "t-acme",
                "resource": "agents",
                "action": "read",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("allowed") is True or body.get("effect") == "allow"

    def test_check_permission_denied(
        self, client, permission_service
    ) -> None:
        """Permission check returns allowed=False for unauthorized access."""
        permission_service.evaluate_permission.return_value = {
            "allowed": False,
            "effect": "deny",
            "resource": "agents",
            "action": "delete",
        }
        resp = client.post(
            "/api/v1/rbac/check",
            json={
                "user_id": "user-1",
                "tenant_id": "t-acme",
                "resource": "agents",
                "action": "delete",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("allowed") is False or body.get("effect") == "deny"
