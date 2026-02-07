# -*- coding: utf-8 -*-
"""
Tests for Admin API Routes - JWT Authentication Service (SEC-001)

Covers:
    - Admin dependency (_require_admin) enforcement
    - All admin endpoints: users, sessions, audit log, lockouts
    - 401 when unauthenticated
    - 403 when non-admin
    - Correct response shapes for successful calls
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.auth_service.api.admin_routes import (
    AuditEventType,
    AuditLogResponse,
    LockoutListResponse,
    SessionListResponse,
    TerminateSessionResponse,
    UnlockResponse,
    RevokeTokensResponse,
    ForcePasswordResetResponse,
    DisableMFAResponse,
    UserDetailResponse,
    UserListResponse,
    UserStatus,
    router,
)


# ---------------------------------------------------------------------------
# Test app setup
# ---------------------------------------------------------------------------


def _make_app(auth: Any = None) -> FastAPI:
    """Create a minimal FastAPI app with the admin router mounted.

    A middleware-like hook sets ``request.state.auth`` so that the admin
    dependency can read it.
    """
    app = FastAPI()

    @app.middleware("http")
    async def inject_auth(request, call_next):
        request.state.auth = auth
        return await call_next(request)

    app.include_router(router)
    return app


def _admin_auth() -> MagicMock:
    """Create a mock auth context with admin privileges."""
    auth = MagicMock()
    auth.user_id = "admin-001"
    auth.tenant_id = "t-platform"
    auth.roles = ["admin"]
    auth.permissions = ["admin:*"]
    return auth


def _viewer_auth() -> MagicMock:
    """Create a mock auth context with only viewer privileges."""
    auth = MagicMock()
    auth.user_id = "viewer-001"
    auth.tenant_id = "t-acme"
    auth.roles = ["viewer"]
    auth.permissions = ["emissions:list"]
    return auth


# ---------------------------------------------------------------------------
# Tests: authentication and authorisation
# ---------------------------------------------------------------------------


class TestAdminAuthEnforcement:
    """Verify that admin endpoints reject unauthenticated/unprivileged users."""

    def test_returns_401_when_not_authenticated(self) -> None:
        app = _make_app(auth=None)
        client = TestClient(app)
        resp = client.get("/auth/admin/users")
        assert resp.status_code == 401

    def test_returns_403_when_not_admin(self) -> None:
        app = _make_app(auth=_viewer_auth())
        client = TestClient(app)
        resp = client.get("/auth/admin/users")
        assert resp.status_code == 403

    def test_returns_200_for_admin(self) -> None:
        app = _make_app(auth=_admin_auth())
        client = TestClient(app)
        resp = client.get("/auth/admin/users")
        assert resp.status_code == 200

    def test_super_admin_role_grants_access(self) -> None:
        auth = MagicMock()
        auth.user_id = "super-001"
        auth.roles = ["super_admin"]
        auth.permissions = []
        app = _make_app(auth=auth)
        client = TestClient(app)
        resp = client.get("/auth/admin/users")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: user management endpoints
# ---------------------------------------------------------------------------


class TestListUsers:
    """Tests for GET /auth/admin/users."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_returns_user_list(self) -> None:
        resp = self.client.get("/auth/admin/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data
        assert "total" in data
        assert isinstance(data["users"], list)

    def test_supports_pagination_params(self) -> None:
        resp = self.client.get("/auth/admin/users?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page_size"] == 5

    def test_supports_tenant_filter(self) -> None:
        resp = self.client.get("/auth/admin/users?tenant_id=t-acme")
        assert resp.status_code == 200

    def test_supports_status_filter(self) -> None:
        resp = self.client.get("/auth/admin/users?status=active")
        assert resp.status_code == 200


class TestGetUser:
    """Tests for GET /auth/admin/users/{user_id}."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_returns_user_details(self) -> None:
        resp = self.client.get("/auth/admin/users/usr-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr-123"
        assert "email" in data
        assert "roles" in data
        assert "permissions" in data

    def test_response_shape_matches_model(self) -> None:
        resp = self.client.get("/auth/admin/users/usr-456")
        data = resp.json()
        # Validate all required fields are present
        required_fields = {
            "user_id", "tenant_id", "status", "roles",
            "permissions", "mfa_enabled", "created_at", "updated_at",
        }
        assert required_fields.issubset(set(data.keys()))


class TestUnlockUser:
    """Tests for POST /auth/admin/users/{user_id}/unlock."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_unlocks_successfully(self) -> None:
        resp = self.client.post("/auth/admin/users/usr-locked/unlock")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr-locked"
        assert data["current_status"] == "active"
        assert data["previous_status"] == "locked"
        assert data["unlocked_by"] == "admin-001"

    def test_returns_401_when_not_authenticated(self) -> None:
        app = _make_app(auth=None)
        client = TestClient(app)
        resp = client.post("/auth/admin/users/usr-locked/unlock")
        assert resp.status_code == 401


class TestRevokeUserTokens:
    """Tests for POST /auth/admin/users/{user_id}/revoke-tokens."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_revokes_successfully(self) -> None:
        resp = self.client.post(
            "/auth/admin/users/usr-compromised/revoke-tokens"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr-compromised"
        assert data["revoked_by"] == "admin-001"
        assert "reason" in data

    def test_custom_reason(self) -> None:
        resp = self.client.post(
            "/auth/admin/users/usr-test/revoke-tokens?reason=security_incident"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reason"] == "security_incident"


class TestForcePasswordReset:
    """Tests for POST /auth/admin/users/{user_id}/force-password-reset."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_forces_reset_successfully(self) -> None:
        resp = self.client.post(
            "/auth/admin/users/usr-target/force-password-reset"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr-target"
        assert data["reset_token_sent"] is True
        assert data["forced_by"] == "admin-001"


class TestDisableMFA:
    """Tests for POST /auth/admin/users/{user_id}/disable-mfa."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_disables_mfa_successfully(self) -> None:
        resp = self.client.post(
            "/auth/admin/users/usr-mfa/disable-mfa"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr-mfa"
        assert data["disabled_by"] == "admin-001"
        assert isinstance(data["previous_mfa_methods"], list)


# ---------------------------------------------------------------------------
# Tests: session management endpoints
# ---------------------------------------------------------------------------


class TestListSessions:
    """Tests for GET /auth/admin/sessions."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_returns_session_list(self) -> None:
        resp = self.client.get("/auth/admin/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)

    def test_supports_user_filter(self) -> None:
        resp = self.client.get("/auth/admin/sessions?user_id=usr-123")
        assert resp.status_code == 200

    def test_supports_tenant_filter(self) -> None:
        resp = self.client.get("/auth/admin/sessions?tenant_id=t-acme")
        assert resp.status_code == 200


class TestTerminateSession:
    """Tests for DELETE /auth/admin/sessions/{session_id}."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_terminates_successfully(self) -> None:
        resp = self.client.delete("/auth/admin/sessions/sess-abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-abc"
        assert data["terminated_by"] == "admin-001"


# ---------------------------------------------------------------------------
# Tests: audit log endpoint
# ---------------------------------------------------------------------------


class TestQueryAuditLog:
    """Tests for GET /auth/admin/audit-log."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_returns_audit_entries(self) -> None:
        resp = self.client.get("/auth/admin/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total" in data

    def test_supports_user_filter(self) -> None:
        resp = self.client.get("/auth/admin/audit-log?user_id=usr-123")
        assert resp.status_code == 200

    def test_supports_event_type_filter(self) -> None:
        resp = self.client.get(
            "/auth/admin/audit-log?event_type=login_success"
        )
        assert resp.status_code == 200

    def test_supports_time_range(self) -> None:
        resp = self.client.get(
            "/auth/admin/audit-log"
            "?start=2026-01-01T00:00:00Z"
            "&end=2026-02-01T00:00:00Z"
        )
        assert resp.status_code == 200

    def test_supports_pagination(self) -> None:
        resp = self.client.get("/auth/admin/audit-log?limit=10&offset=5")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: lockout management endpoint
# ---------------------------------------------------------------------------


class TestListLockouts:
    """Tests for GET /auth/admin/lockouts."""

    def setup_method(self) -> None:
        self.app = _make_app(auth=_admin_auth())
        self.client = TestClient(self.app)

    def test_returns_lockout_list(self) -> None:
        resp = self.client.get("/auth/admin/lockouts")
        assert resp.status_code == 200
        data = resp.json()
        assert "lockouts" in data
        assert "total" in data
        assert isinstance(data["lockouts"], list)

    def test_supports_tenant_filter(self) -> None:
        resp = self.client.get("/auth/admin/lockouts?tenant_id=t-acme")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: model validation
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for Pydantic model validation."""

    def test_user_status_enum(self) -> None:
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.LOCKED.value == "locked"
        assert UserStatus.DISABLED.value == "disabled"

    def test_audit_event_type_enum(self) -> None:
        assert AuditEventType.LOGIN_SUCCESS.value == "login_success"
        assert AuditEventType.TOKEN_REVOKED.value == "token_revoked"
        assert AuditEventType.ACCOUNT_LOCKED.value == "account_locked"
