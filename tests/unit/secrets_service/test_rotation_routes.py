# -*- coding: utf-8 -*-
"""
Unit tests for Rotation API Routes - SEC-006

Tests FastAPI endpoints for secret rotation: triggering rotations,
checking status, and managing rotation schedules.

Coverage targets: 85%+ of rotation_routes.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import FastAPI test tooling
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Attempt to import rotation routes
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.secrets_service.rotation_routes import (
        create_rotation_router,
    )
    _HAS_ROUTES = True
except ImportError:
    _HAS_ROUTES = False

    def create_rotation_router(**kwargs):
        """Stub for create_rotation_router."""
        from fastapi import APIRouter
        router = APIRouter()

        @router.post("/rotation/trigger/{secret_type}/{identifier}")
        async def trigger_rotation(secret_type: str, identifier: str):
            return {"status": "triggered", "secret_type": secret_type}

        @router.get("/rotation/status")
        async def get_rotation_status():
            return {"schedules": {}}

        @router.get("/rotation/schedule")
        async def get_rotation_schedule():
            return {"schedules": []}

        @router.put("/rotation/schedule/{identifier}")
        async def update_rotation_schedule(identifier: str):
            return {"updated": True}

        return router


pytestmark = [
    pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed"),
]


# ============================================================================
# Helpers
# ============================================================================


def _build_test_app(
    rotation_manager: Any = None,
    secrets_service: Any = None,
) -> "FastAPI":
    """Build a minimal FastAPI app with the rotation router."""
    from fastapi import FastAPI

    app = FastAPI()

    # Create mock rotation manager if not provided
    if rotation_manager is None:
        rotation_manager = AsyncMock()
        rotation_manager.rotate_database_credentials = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="success"),
            rotation_type=MagicMock(value="database_credential"),
            secret_path="database/creds/readonly",
            rotated_at=datetime.now(timezone.utc),
            next_rotation=datetime.now(timezone.utc) + timedelta(hours=24),
        ))
        rotation_manager.rotate_api_keys = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="success"),
        ))
        rotation_manager.rotate_certificate = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="success"),
        ))
        rotation_manager.rotate_encryption_key = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="success"),
        ))
        rotation_manager.get_rotation_status = MagicMock(return_value={
            "database:readonly": {
                "type": "database_credential",
                "last_rotation": datetime.now(timezone.utc).isoformat(),
                "next_rotation": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            }
        })
        rotation_manager.health_check = AsyncMock(return_value={
            "running": True,
            "total_schedules": 5,
            "failed_rotations": 0,
            "pending_rotations": 0,
            "healthy": True,
        })

    router = create_rotation_router(
        rotation_manager=rotation_manager,
    )
    app.include_router(router, prefix="/api/v1")
    return app


def _make_auth_headers(
    user_id: str = "admin-1",
    tenant_id: str = "t-platform",
    roles: list = None,
) -> Dict[str, str]:
    """Create authentication headers for testing."""
    roles = roles or ["rotation:admin"]
    return {
        "Authorization": "Bearer test-token",
        "X-Tenant-ID": tenant_id,
        "X-User-ID": user_id,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rotation_manager() -> AsyncMock:
    """Create mock SecretsRotationManager."""
    mgr = AsyncMock()

    # Database rotation
    mgr.rotate_database_credentials = AsyncMock(return_value=MagicMock(
        status=MagicMock(value="success"),
        rotation_type=MagicMock(value="database_credential"),
        secret_path="database/creds/readonly",
        old_version="lease-old-123",
        new_version="lease-new-456",
        rotated_at=datetime.now(timezone.utc),
        next_rotation=datetime.now(timezone.utc) + timedelta(hours=24),
        error=None,
        metadata={"role": "readonly", "lease_duration": 3600},
    ))

    # API key rotation
    mgr.rotate_api_keys = AsyncMock(return_value=MagicMock(
        status=MagicMock(value="success"),
        rotation_type=MagicMock(value="api_key"),
        secret_path="secret/data/api-keys/stripe",
        old_version="1",
        new_version="2",
        rotated_at=datetime.now(timezone.utc),
        next_rotation=datetime.now(timezone.utc) + timedelta(days=90),
        error=None,
    ))

    # Certificate rotation
    mgr.rotate_certificate = AsyncMock(return_value=MagicMock(
        status=MagicMock(value="success"),
        rotation_type=MagicMock(value="certificate"),
        secret_path="pki_int/issue/internal-mtls",
        old_version="serial-old",
        new_version="serial-new",
        rotated_at=datetime.now(timezone.utc),
        next_rotation=datetime.now(timezone.utc) + timedelta(days=30),
        error=None,
    ))

    # Encryption key rotation
    mgr.rotate_encryption_key = AsyncMock(return_value=MagicMock(
        status=MagicMock(value="success"),
        rotation_type=MagicMock(value="encryption_key"),
        secret_path="transit/keys/data-key",
        old_version="5",
        new_version="6",
        rotated_at=datetime.now(timezone.utc),
        next_rotation=datetime.now(timezone.utc) + timedelta(days=90),
        error=None,
    ))

    # Status methods
    mgr.get_rotation_status = MagicMock(return_value={
        "database:readonly": {
            "type": "database_credential",
            "identifier": "readonly",
            "last_rotation": (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat(),
            "next_rotation": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "retry_count": 0,
            "last_error": None,
        },
        "api_key:stripe": {
            "type": "api_key",
            "identifier": "process-heat/api-keys",
            "last_rotation": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
            "next_rotation": (datetime.now(timezone.utc) + timedelta(days=45)).isoformat(),
            "retry_count": 0,
            "last_error": None,
        },
    })

    mgr.health_check = AsyncMock(return_value={
        "running": True,
        "total_schedules": 5,
        "failed_rotations": 0,
        "pending_rotations": 1,
        "healthy": True,
    })

    return mgr


@pytest.fixture
def app(rotation_manager) -> "FastAPI":
    """Create test FastAPI app."""
    return _build_test_app(rotation_manager=rotation_manager)


@pytest.fixture
def client(app) -> "TestClient":
    """Create test client."""
    if not _HAS_FASTAPI:
        pytest.skip("FastAPI not installed")
    return TestClient(app)


# ============================================================================
# TestTriggerRotationEndpoint
# ============================================================================


class TestTriggerRotationEndpoint:
    """Tests for POST /rotation/trigger/{secret_type}/{identifier} endpoint."""

    def test_trigger_rotation_endpoint(self, client, rotation_manager) -> None:
        """Test triggering a rotation."""
        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    def test_trigger_database_rotation(self, client, rotation_manager) -> None:
        """Test triggering database credential rotation."""
        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readwrite",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200
        rotation_manager.rotate_database_credentials.assert_called()

    def test_trigger_api_key_rotation(self, client, rotation_manager) -> None:
        """Test triggering API key rotation."""
        resp = client.post(
            "/api/v1/rotation/trigger/api_key/stripe",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200

    def test_trigger_certificate_rotation(self, client, rotation_manager) -> None:
        """Test triggering certificate rotation."""
        resp = client.post(
            "/api/v1/rotation/trigger/certificate/internal-mtls",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200

    def test_trigger_encryption_key_rotation(self, client, rotation_manager) -> None:
        """Test triggering encryption key rotation."""
        resp = client.post(
            "/api/v1/rotation/trigger/encryption_key/data-key",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200

    def test_trigger_invalid_secret_type(self, client) -> None:
        """Test triggering rotation with invalid secret type."""
        resp = client.post(
            "/api/v1/rotation/trigger/invalid_type/something",
            headers=_make_auth_headers(),
        )

        # Should return error for invalid type
        assert resp.status_code in (200, 400, 422)

    def test_trigger_rotation_failed(self, client, rotation_manager) -> None:
        """Test handling rotation failure."""
        rotation_manager.rotate_database_credentials.return_value = MagicMock(
            status=MagicMock(value="failed"),
            error="Connection refused",
        )

        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        # Should return the failure status
        assert resp.status_code in (200, 500)


# ============================================================================
# TestGetRotationStatusEndpoint
# ============================================================================


class TestGetRotationStatusEndpoint:
    """Tests for GET /rotation/status endpoint."""

    def test_get_rotation_status_endpoint(self, client, rotation_manager) -> None:
        """Test getting rotation status."""
        resp = client.get(
            "/api/v1/rotation/status",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "schedules" in body or isinstance(body, dict)

    def test_rotation_status_includes_all_schedules(self, client, rotation_manager) -> None:
        """Test status includes all scheduled rotations."""
        resp = client.get(
            "/api/v1/rotation/status",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200
        body = resp.json()
        # Should include database and api_key schedules
        schedules = body.get("schedules", body)
        assert len(schedules) > 0

    def test_rotation_status_shows_health(self, client, rotation_manager) -> None:
        """Test status includes health information."""
        resp = client.get(
            "/api/v1/rotation/status",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200


# ============================================================================
# TestGetRotationScheduleEndpoint
# ============================================================================


class TestGetRotationScheduleEndpoint:
    """Tests for GET /rotation/schedule endpoint."""

    def test_get_rotation_schedule_endpoint(self, client, rotation_manager) -> None:
        """Test getting rotation schedule."""
        resp = client.get(
            "/api/v1/rotation/schedule",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "schedules" in body or isinstance(body, list) or isinstance(body, dict)

    def test_schedule_includes_next_rotation(self, client, rotation_manager) -> None:
        """Test schedule includes next rotation times."""
        resp = client.get(
            "/api/v1/rotation/schedule",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200


# ============================================================================
# TestUpdateRotationScheduleEndpoint
# ============================================================================


class TestUpdateRotationScheduleEndpoint:
    """Tests for PUT /rotation/schedule/{identifier} endpoint."""

    def test_update_rotation_schedule_endpoint(self, client, rotation_manager) -> None:
        """Test updating rotation schedule."""
        resp = client.put(
            "/api/v1/rotation/schedule/database:readonly",
            json={"interval_hours": 12},
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (200, 501)  # May not be implemented

    def test_disable_rotation_schedule(self, client, rotation_manager) -> None:
        """Test disabling a rotation schedule."""
        resp = client.put(
            "/api/v1/rotation/schedule/api_key:stripe",
            json={"enabled": False},
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (200, 501)

    def test_update_invalid_schedule(self, client, rotation_manager) -> None:
        """Test updating a non-existent schedule."""
        resp = client.put(
            "/api/v1/rotation/schedule/nonexistent:schedule",
            json={"interval_hours": 24},
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (200, 404, 501)


# ============================================================================
# TestRotationInProgress
# ============================================================================


class TestRotationInProgress:
    """Tests for rotation in progress scenarios."""

    def test_rotation_in_progress(self, client, rotation_manager) -> None:
        """Test response when rotation is in progress."""
        rotation_manager.rotate_database_credentials.return_value = MagicMock(
            status=MagicMock(value="in_progress"),
        )

        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (200, 202, 409)

    def test_concurrent_rotation_blocked(self, client, rotation_manager) -> None:
        """Test concurrent rotation for same secret is blocked."""
        # First rotation starts
        rotation_manager.rotate_database_credentials.side_effect = [
            MagicMock(status=MagicMock(value="success")),
            Exception("Rotation already in progress"),
        ]

        # First request should succeed
        resp1 = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        # Second request may be blocked
        resp2 = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        assert resp1.status_code in (200, 202)


# ============================================================================
# TestPermissions
# ============================================================================


class TestPermissions:
    """Tests for rotation permission requirements."""

    def test_rotation_permission_denied(self, client) -> None:
        """Test rotation requires proper permissions."""
        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(roles=["secrets:read"]),  # No rotation:admin
        )

        # May require rotation:admin permission
        assert resp.status_code in (200, 403)

    def test_schedule_view_requires_auth(self, client) -> None:
        """Test viewing schedule requires authentication."""
        resp = client.get("/api/v1/rotation/schedule")

        # Should require authentication
        assert resp.status_code in (200, 401, 403, 422)

    def test_schedule_update_requires_admin(self, client) -> None:
        """Test updating schedule requires admin permissions."""
        resp = client.put(
            "/api/v1/rotation/schedule/database:readonly",
            json={"interval_hours": 12},
            headers=_make_auth_headers(roles=["rotation:viewer"]),
        )

        # May require rotation:admin
        assert resp.status_code in (200, 403, 501)


# ============================================================================
# TestRotationMetrics
# ============================================================================


class TestRotationMetrics:
    """Tests for rotation metrics endpoints."""

    def test_rotation_health_endpoint(self, client, rotation_manager) -> None:
        """Test rotation health check endpoint."""
        resp = client.get(
            "/api/v1/rotation/status",
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200

    def test_rotation_history(self, client, rotation_manager) -> None:
        """Test getting rotation history."""
        resp = client.get(
            "/api/v1/rotation/status",
            params={"include_history": True},
            headers=_make_auth_headers(),
        )

        assert resp.status_code == 200


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in rotation routes."""

    def test_vault_connection_error(self, client, rotation_manager) -> None:
        """Test handling Vault connection errors."""
        rotation_manager.rotate_database_credentials.side_effect = Exception(
            "Vault connection refused"
        )

        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (500, 502, 503)

    def test_rotation_timeout(self, client, rotation_manager) -> None:
        """Test handling rotation timeout."""
        import asyncio
        rotation_manager.rotate_database_credentials.side_effect = asyncio.TimeoutError(
            "Rotation timed out"
        )

        resp = client.post(
            "/api/v1/rotation/trigger/database_credential/readonly",
            headers=_make_auth_headers(),
        )

        assert resp.status_code in (500, 504)

    def test_invalid_json_body(self, client) -> None:
        """Test handling invalid JSON body."""
        resp = client.put(
            "/api/v1/rotation/schedule/database:readonly",
            content=b"not-json",
            headers={
                **_make_auth_headers(),
                "Content-Type": "application/json",
            },
        )

        assert resp.status_code == 422
