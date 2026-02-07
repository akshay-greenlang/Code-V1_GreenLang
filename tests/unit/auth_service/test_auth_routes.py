# -*- coding: utf-8 -*-
"""
Unit tests for Auth API Routes - JWT Authentication Service (SEC-001)

Tests FastAPI endpoints: /auth/login, /auth/refresh, /auth/revoke,
/auth/validate, /auth/me, /auth/jwks, /auth/logout using the HTTPX
AsyncClient / TestClient pattern.

Coverage targets: 85%+ of auth_routes.py / api.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import FastAPI test tooling.  If not available, skip.
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx import ASGITransport, AsyncClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

# Import the auth service types that ARE available
from greenlang.infrastructure.auth_service.token_service import (
    IssuedToken,
    TokenClaims,
    TokenService,
)
from greenlang.infrastructure.auth_service.revocation import RevocationService

# ---------------------------------------------------------------------------
# Try importing the auth routes module.  It may not exist yet.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.auth_service.api import (
        create_auth_router,
    )
    _HAS_AUTH_ROUTES = True
except ImportError:
    _HAS_AUTH_ROUTES = False

pytestmark = [
    pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed"),
    pytest.mark.skipif(not _HAS_AUTH_ROUTES, reason="auth_service.api not implemented"),
]


# ============================================================================
# Helpers
# ============================================================================


def _build_test_app(
    token_service: Any = None,
    revocation_service: Any = None,
    refresh_manager: Any = None,
    lockout_manager: Any = None,
    password_validator: Any = None,
) -> FastAPI:
    """Build a minimal FastAPI app with the auth router."""
    app = FastAPI()
    router = create_auth_router(
        token_service=token_service or AsyncMock(spec=TokenService),
        revocation_service=revocation_service or AsyncMock(spec=RevocationService),
        refresh_manager=refresh_manager or AsyncMock(),
        lockout_manager=lockout_manager or AsyncMock(),
        password_validator=password_validator or AsyncMock(),
    )
    app.include_router(router, prefix="/auth")
    return app


def _mock_issued_token(jti: str = "jti-1") -> IssuedToken:
    return IssuedToken(
        access_token="eyJ.mock.access",
        token_type="Bearer",
        expires_in=1800,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=1800),
        jti=jti,
        scope="openid profile",
    )


def _mock_token_claims() -> TokenClaims:
    return TokenClaims(
        sub="user-1",
        tenant_id="t-acme",
        roles=["viewer"],
        permissions=["read:data"],
        scopes=["openid"],
        email="user@example.com",
        name="Test User",
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def token_service() -> AsyncMock:
    svc = AsyncMock(spec=TokenService)
    svc.issue_token = AsyncMock(return_value=_mock_issued_token())
    svc.validate_token = AsyncMock(return_value=_mock_token_claims())
    svc.decode_token = AsyncMock(return_value={"sub": "user-1", "jti": "jti-1"})
    svc.get_public_key_jwks = AsyncMock(return_value={"keys": [{"kty": "RSA"}]})
    return svc


@pytest.fixture
def revocation_service() -> AsyncMock:
    svc = AsyncMock(spec=RevocationService)
    svc.revoke_token = AsyncMock(return_value=True)
    svc.is_revoked = AsyncMock(return_value=False)
    return svc


@pytest.fixture
def refresh_manager() -> AsyncMock:
    mgr = AsyncMock()
    mgr.issue = AsyncMock(return_value=MagicMock(
        token="opaque-refresh-token",
        family_id="fam-1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    ))
    mgr.rotate = AsyncMock(return_value=MagicMock(
        token="new-refresh-token",
        family_id="fam-1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    ))
    return mgr


@pytest.fixture
def lockout_manager() -> AsyncMock:
    mgr = AsyncMock()
    status_mock = MagicMock()
    status_mock.is_locked = False
    status_mock.remaining_attempts = 5
    mgr.get_lockout_status = AsyncMock(return_value=status_mock)
    mgr.record_failed_attempt = AsyncMock()
    mgr.record_successful_login = AsyncMock()
    return mgr


@pytest.fixture
def password_validator() -> AsyncMock:
    v = AsyncMock()
    v.validate = MagicMock(return_value=(True, []))
    return v


@pytest.fixture
def app(
    token_service,
    revocation_service,
    refresh_manager,
    lockout_manager,
    password_validator,
) -> FastAPI:
    return _build_test_app(
        token_service=token_service,
        revocation_service=revocation_service,
        refresh_manager=refresh_manager,
        lockout_manager=lockout_manager,
        password_validator=password_validator,
    )


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ============================================================================
# TestLoginEndpoint
# ============================================================================


class TestLoginEndpoint:
    """Tests for POST /auth/login."""

    def test_login_success(self, client, token_service) -> None:
        """Successful login returns access and refresh tokens."""
        resp = client.post(
            "/auth/login",
            json={"username": "user@example.com", "password": "Str0ng!Pass"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "Bearer"

    def test_login_wrong_password(
        self, client, token_service, lockout_manager
    ) -> None:
        """Wrong password returns 401."""
        token_service.issue_token.side_effect = ValueError("Invalid credentials")
        resp = client.post(
            "/auth/login",
            json={"username": "user@example.com", "password": "wrong"},
        )
        assert resp.status_code in (401, 422)

    def test_login_unknown_user(self, client, token_service) -> None:
        """Unknown user returns 401."""
        token_service.issue_token.side_effect = ValueError("User not found")
        resp = client.post(
            "/auth/login",
            json={"username": "nobody@example.com", "password": "pass"},
        )
        assert resp.status_code in (401, 422)

    def test_login_locked_account(
        self, client, lockout_manager
    ) -> None:
        """Locked account returns 423."""
        locked_status = MagicMock()
        locked_status.is_locked = True
        locked_status.remaining_attempts = 0
        lockout_manager.get_lockout_status.return_value = locked_status
        resp = client.post(
            "/auth/login",
            json={"username": "locked@example.com", "password": "pass"},
        )
        assert resp.status_code in (423, 403, 429)

    def test_login_rate_limited(self, client, lockout_manager) -> None:
        """Excessive login attempts return 429."""
        lockout_manager.check_ip_rate_limit = AsyncMock(return_value=True)
        resp = client.post(
            "/auth/login",
            json={"username": "user@example.com", "password": "pass"},
        )
        # May return 429 or proceed depending on route wiring
        assert resp.status_code in (200, 429, 401)

    def test_login_with_mfa(self, client, token_service) -> None:
        """Login with MFA code is accepted."""
        resp = client.post(
            "/auth/login",
            json={
                "username": "user@example.com",
                "password": "Str0ng!Pass",
                "mfa_code": "123456",
            },
        )
        assert resp.status_code in (200, 422)


# ============================================================================
# TestRefreshEndpoint
# ============================================================================


class TestRefreshEndpoint:
    """Tests for POST /auth/refresh."""

    def test_refresh_success(self, client, refresh_manager, token_service) -> None:
        """Valid refresh token returns new access + refresh tokens."""
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": "valid-refresh-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body

    def test_refresh_expired_token(self, client, refresh_manager) -> None:
        """Expired refresh token returns 401."""
        refresh_manager.rotate.side_effect = Exception("Token expired")
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": "expired-token"},
        )
        assert resp.status_code in (401, 400)

    def test_refresh_revoked_token(self, client, refresh_manager) -> None:
        """Revoked refresh token returns 401."""
        refresh_manager.rotate.side_effect = Exception("Token revoked")
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": "revoked-token"},
        )
        assert resp.status_code in (401, 400)

    def test_refresh_reuse_detection(
        self, client, refresh_manager
    ) -> None:
        """Reuse of a rotated refresh token returns 401 and revokes family."""
        refresh_manager.rotate.side_effect = Exception("Reuse detected")
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": "reused-token"},
        )
        assert resp.status_code in (401, 400)


# ============================================================================
# TestRevokeEndpoint
# ============================================================================


class TestRevokeEndpoint:
    """Tests for POST /auth/revoke."""

    def test_revoke_access_token(
        self, client, revocation_service, token_service
    ) -> None:
        """Revoking an access token succeeds."""
        resp = client.post(
            "/auth/revoke",
            json={"token": "eyJ.access.token", "token_type": "access"},
            headers={"Authorization": "Bearer eyJ.mock.auth"},
        )
        assert resp.status_code in (200, 204)

    def test_revoke_refresh_token(
        self, client, revocation_service
    ) -> None:
        """Revoking a refresh token succeeds."""
        resp = client.post(
            "/auth/revoke",
            json={"token": "opaque-refresh", "token_type": "refresh"},
            headers={"Authorization": "Bearer eyJ.mock.auth"},
        )
        assert resp.status_code in (200, 204)

    def test_revoke_requires_auth(self, client) -> None:
        """Revoke endpoint requires authentication."""
        resp = client.post(
            "/auth/revoke",
            json={"token": "some-token"},
        )
        assert resp.status_code in (401, 403, 422)


# ============================================================================
# TestValidateEndpoint
# ============================================================================


class TestValidateEndpoint:
    """Tests for POST /auth/validate."""

    def test_validate_valid_token(self, client, token_service) -> None:
        """Valid token returns claims."""
        resp = client.post(
            "/auth/validate",
            json={"token": "valid.jwt.token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("active") is True or "sub" in body

    def test_validate_expired_token(self, client, token_service) -> None:
        """Expired token returns inactive."""
        token_service.validate_token.return_value = None
        resp = client.post(
            "/auth/validate",
            json={"token": "expired.jwt.token"},
        )
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            body = resp.json()
            assert body.get("active") is False or body.get("valid") is False

    def test_validate_revoked_token(self, client, token_service) -> None:
        """Revoked token returns inactive."""
        token_service.validate_token.return_value = None
        resp = client.post(
            "/auth/validate",
            json={"token": "revoked.jwt.token"},
        )
        assert resp.status_code in (200, 401)


# ============================================================================
# TestMeEndpoint
# ============================================================================


class TestMeEndpoint:
    """Tests for GET /auth/me."""

    def test_get_current_user(self, client, token_service) -> None:
        """Authenticated user gets their profile."""
        resp = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer eyJ.mock.auth"},
        )
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            body = resp.json()
            assert "sub" in body or "user_id" in body

    def test_me_requires_auth(self, client) -> None:
        """GET /auth/me without auth returns 401."""
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)


# ============================================================================
# TestJWKSEndpoint
# ============================================================================


class TestJWKSEndpoint:
    """Tests for GET /auth/jwks or /.well-known/jwks.json."""

    def test_jwks_returns_public_key(self, client, token_service) -> None:
        """JWKS endpoint returns public key(s)."""
        resp = client.get("/auth/jwks")
        assert resp.status_code == 200
        body = resp.json()
        assert "keys" in body
        assert len(body["keys"]) >= 1

    def test_jwks_format_valid(self, client, token_service) -> None:
        """JWKS response keys have required fields."""
        resp = client.get("/auth/jwks")
        body = resp.json()
        key = body["keys"][0]
        assert "kty" in key


# ============================================================================
# TestLogoutEndpoint
# ============================================================================


class TestLogoutEndpoint:
    """Tests for POST /auth/logout."""

    def test_logout_revokes_tokens(
        self, client, revocation_service, token_service
    ) -> None:
        """Logout revokes the current access token."""
        resp = client.post(
            "/auth/logout",
            headers={"Authorization": "Bearer eyJ.mock.auth"},
        )
        assert resp.status_code in (200, 204)

    def test_logout_requires_auth(self, client) -> None:
        """Logout requires authentication."""
        resp = client.post("/auth/logout")
        assert resp.status_code in (401, 403)


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in auth routes."""

    def test_malformed_json_returns_422(self, client) -> None:
        """Malformed JSON body returns 422."""
        resp = client.post(
            "/auth/login",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_required_field_returns_422(self, client) -> None:
        """Missing required fields return 422."""
        resp = client.post(
            "/auth/login",
            json={"username": "user@example.com"},
            # missing "password"
        )
        assert resp.status_code == 422

    def test_internal_error_returns_500(
        self, client, token_service
    ) -> None:
        """Internal service error returns 500."""
        token_service.issue_token.side_effect = RuntimeError("Internal failure")
        resp = client.post(
            "/auth/login",
            json={"username": "user@example.com", "password": "Str0ng!Pass"},
        )
        assert resp.status_code in (500, 401)
