# -*- coding: utf-8 -*-
"""
Unit tests for Schema Migration API Router - AGENT-DATA-017
=============================================================

Comprehensive tests for the FastAPI router module providing 20 endpoints
under ``/api/v1/schema-migration``. Tests cover:

- Router availability and importability
- Schema CRUD endpoints (POST, GET, PUT, DELETE /schemas)
- Version management endpoints (POST, GET /versions)
- Change detection endpoints (POST /changes/detect, GET /changes)
- Compatibility check endpoints (POST /compatibility/check, GET /compatibility)
- Migration plan endpoints (POST /plans, GET /plans/{id})
- Migration execution endpoints (POST /execute, GET /executions/{id})
- Rollback endpoints (POST /rollback/{id})
- Pipeline orchestration endpoint (POST /pipeline)
- Health and statistics endpoints (GET /health, GET /stats)
- Error handling (ValueError -> 400, None -> 404, Exception -> 500, 422)
- Pagination parameter defaults and bounds

Target: ~80 tests, 85%+ coverage of router.py

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import importlib
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Conditional imports -- skip entire module if FastAPI is unavailable
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "/api/v1/schema-migration"


# ---------------------------------------------------------------------------
# Helper: build mock return values
# ---------------------------------------------------------------------------


def _schema_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal schema definition dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "namespace": "greenlang.emissions",
        "name": "emission_factors_v1",
        "schema_type": "json_schema",
        "owner": "platform-team",
        "tags": {"domain": "emissions"},
        "status": "active",
        "description": "Test schema",
        "definition_json": {"type": "object"},
        "created_at": "2026-02-01T00:00:00Z",
    }
    data.update(overrides)
    return data


def _version_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal schema version dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "schema_id": str(uuid.uuid4()),
        "version": "1.0.0",
        "definition_json": {"type": "object"},
        "changelog": "Initial version",
        "is_deprecated": False,
        "created_at": "2026-02-01T00:00:00Z",
    }
    data.update(overrides)
    return data


def _change_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal schema change dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "source_version_id": str(uuid.uuid4()),
        "target_version_id": str(uuid.uuid4()),
        "change_type": "added",
        "field_path": "user.salary",
        "severity": "non_breaking",
        "description": "Added salary field",
        "detected_at": "2026-02-01T00:00:00Z",
    }
    data.update(overrides)
    return data


def _compat_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal compatibility result dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "source_version_id": str(uuid.uuid4()),
        "target_version_id": str(uuid.uuid4()),
        "compatibility_level": "backward",
        "issues": [],
        "recommendations": [],
        "checked_at": "2026-02-01T00:00:00Z",
    }
    data.update(overrides)
    return data


def _plan_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal migration plan dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "source_schema_id": str(uuid.uuid4()),
        "target_schema_id": str(uuid.uuid4()),
        "source_version": "1.0.0",
        "target_version": "2.0.0",
        "steps": [],
        "status": "draft",
        "estimated_effort": "medium",
        "created_at": "2026-02-01T00:00:00Z",
    }
    data.update(overrides)
    return data


def _execution_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal migration execution dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "plan_id": str(uuid.uuid4()),
        "status": "completed",
        "current_step": 3,
        "total_steps": 3,
        "records_processed": 1000,
        "records_failed": 0,
        "started_at": "2026-02-01T00:00:00Z",
        "completed_at": "2026-02-01T00:00:01Z",
    }
    data.update(overrides)
    return data


def _rollback_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal rollback record dict with optional overrides."""
    data: Dict[str, Any] = {
        "id": overrides.pop("id", str(uuid.uuid4())),
        "execution_id": str(uuid.uuid4()),
        "rollback_type": "full",
        "status": "completed",
        "records_reverted": 500,
        "started_at": "2026-02-01T00:00:00Z",
        "completed_at": "2026-02-01T00:00:01Z",
    }
    data.update(overrides)
    return data


def _pipeline_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal pipeline result dict with optional overrides."""
    data: Dict[str, Any] = {
        "pipeline_id": str(uuid.uuid4()),
        "stages_completed": ["detect", "compatibility", "plan", "execute"],
        "stages_failed": [],
        "total_time_ms": 125.3,
    }
    data.update(overrides)
    return data


def _stats_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal statistics dict with optional overrides."""
    data: Dict[str, Any] = {
        "total_schemas": 10,
        "total_versions": 25,
        "total_changes": 40,
        "total_migrations": 5,
        "total_rollbacks": 1,
        "total_drift_events": 3,
        "schemas_by_type": {"json_schema": 8, "avro": 2},
        "schemas_by_status": {"active": 7, "deprecated": 3},
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a MagicMock service with sensible default return values.

    All 20 endpoint methods are configured to return simple dicts
    so the router can serialise responses without errors.
    """
    svc = MagicMock()

    # Schema CRUD
    svc.register_schema.return_value = _schema_dict()
    svc.list_schemas.return_value = [_schema_dict(), _schema_dict()]
    svc.get_schema.return_value = _schema_dict()
    svc.update_schema.return_value = _schema_dict()
    svc.delete_schema.return_value = True

    # Versions
    svc.create_version.return_value = _version_dict()
    svc.list_versions.return_value = [_version_dict()]
    svc.get_version.return_value = _version_dict()

    # Change detection
    svc.detect_changes.return_value = _change_dict()
    svc.list_changes.return_value = [_change_dict(), _change_dict()]

    # Compatibility
    svc.check_compatibility.return_value = _compat_dict()
    svc.list_compatibility_results.return_value = [_compat_dict()]

    # Migration plans
    svc.create_plan.return_value = _plan_dict()
    svc.get_plan.return_value = _plan_dict()

    # Migration execution
    svc.execute_migration.return_value = _execution_dict()
    svc.get_execution.return_value = _execution_dict()

    # Rollback
    svc.rollback_execution.return_value = _rollback_dict()

    # Pipeline
    svc.run_pipeline.return_value = _pipeline_dict()

    # Health & stats
    svc.health_check.return_value = {
        "status": "healthy",
        "engines": {"registry": "ok", "versioner": "ok"},
    }
    svc.get_stats.return_value = _stats_dict()

    return svc


@pytest.fixture
def client(mock_service: MagicMock) -> "TestClient":
    """Create a FastAPI TestClient wired to the mock service.

    The router is imported dynamically to benefit from the conftest
    stub that patches the greenlang.schema_migration package.
    """
    from greenlang.schema_migration.api.router import (
        create_schema_migration_router,
    )

    app = FastAPI()
    app.state.schema_migration_service = mock_service
    router = create_schema_migration_router()
    app.include_router(router)
    return TestClient(app)


# ===========================================================================
# TestRouterAvailability
# ===========================================================================


class TestRouterAvailability:
    """Verify the router module is importable and exposes expected symbols."""

    def test_module_importable(self):
        """The router module can be imported without errors."""
        mod = importlib.import_module(
            "greenlang.schema_migration.api.router"
        )
        assert mod is not None

    def test_fastapi_available_flag(self):
        """The FASTAPI_AVAILABLE flag is True when FastAPI is installed."""
        from greenlang.schema_migration.api.router import (
            FASTAPI_AVAILABLE as flag,
        )

        assert flag is True

    def test_create_schema_migration_router_callable(self):
        """create_schema_migration_router is a callable factory function."""
        from greenlang.schema_migration.api.router import (
            create_schema_migration_router,
        )

        assert callable(create_schema_migration_router)

    def test_router_module_level_instance(self):
        """The module exports a pre-built router instance."""
        from greenlang.schema_migration.api.router import router

        assert router is not None

    def test_all_exports(self):
        """__all__ contains the expected public symbols."""
        from greenlang.schema_migration.api.router import __all__

        assert "create_schema_migration_router" in __all__
        assert "router" in __all__
        assert "FASTAPI_AVAILABLE" in __all__


# ===========================================================================
# TestSchemaEndpoints
# ===========================================================================


class TestSchemaEndpoints:
    """Tests for the five schema CRUD endpoints (POST, GET list, GET detail,
    PUT update, DELETE)."""

    # --- POST /schemas ---

    def test_register_schema_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /schemas returns 201 with valid body."""
        body = {
            "namespace": "greenlang.test",
            "name": "test_schema",
            "schema_type": "json_schema",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["namespace"] == "greenlang.emissions"

    def test_register_schema_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /schemas passes body fields to service.register_schema."""
        body = {
            "namespace": "ns.test",
            "name": "my_schema",
            "schema_type": "avro",
            "definition": {"type": "record", "name": "X", "fields": []},
            "owner": "team-a",
            "tags": ["core"],
            "description": "A test schema",
        }
        client.post(f"{BASE_URL}/schemas", json=body)
        mock_service.register_schema.assert_called_once_with(
            namespace="ns.test",
            name="my_schema",
            schema_type="avro",
            definition={"type": "record", "name": "X", "fields": []},
            owner="team-a",
            tags=["core"],
            description="A test schema",
        )

    def test_register_schema_with_pydantic_model_return(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /schemas handles BaseModel return via model_dump."""
        mock_model = MagicMock()
        mock_model.model_dump.return_value = _schema_dict()
        # Make isinstance check for BaseModel return True
        from pydantic import BaseModel as PydanticBase

        mock_model.__class__ = type(
            "FakeModel", (PydanticBase,), {"__annotations__": {}}
        )
        mock_service.register_schema.return_value = mock_model
        body = {
            "namespace": "greenlang.test",
            "name": "test_schema",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 201

    def test_register_schema_missing_required_field_422(
        self, client: "TestClient"
    ):
        """POST /schemas returns 422 when namespace is missing."""
        body = {"name": "test_schema", "definition": {"type": "object"}}
        resp = client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 422

    def test_register_schema_empty_body_422(self, client: "TestClient"):
        """POST /schemas returns 422 with empty JSON body."""
        resp = client.post(f"{BASE_URL}/schemas", json={})
        assert resp.status_code == 422

    # --- GET /schemas ---

    def test_list_schemas_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas returns 200 with paginated envelope."""
        resp = client.get(f"{BASE_URL}/schemas")
        assert resp.status_code == 200
        data = resp.json()
        assert "schemas" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data

    def test_list_schemas_passes_filter_params(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas forwards query params to service.list_schemas."""
        client.get(
            f"{BASE_URL}/schemas",
            params={
                "namespace": "ns.test",
                "schema_type": "avro",
                "status": "active",
                "owner": "team-a",
                "tag": "core",
                "limit": 10,
                "offset": 5,
            },
        )
        mock_service.list_schemas.assert_called_once_with(
            namespace="ns.test",
            schema_type="avro",
            status="active",
            owner="team-a",
            tag="core",
            limit=10,
            offset=5,
        )

    def test_list_schemas_default_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas uses default limit=50 and offset=0."""
        client.get(f"{BASE_URL}/schemas")
        call_kwargs = mock_service.list_schemas.call_args.kwargs
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 0

    def test_list_schemas_count_matches_items(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas count field matches actual items returned."""
        mock_service.list_schemas.return_value = [
            _schema_dict(),
            _schema_dict(),
            _schema_dict(),
        ]
        resp = client.get(f"{BASE_URL}/schemas")
        data = resp.json()
        assert data["count"] == 3
        assert len(data["schemas"]) == 3

    # --- GET /schemas/{schema_id} ---

    def test_get_schema_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas/{id} returns 200 for an existing schema."""
        schema_id = "abc-123"
        resp = client.get(f"{BASE_URL}/schemas/{schema_id}")
        assert resp.status_code == 200
        mock_service.get_schema.assert_called_once_with(schema_id)

    def test_get_schema_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas/{id} returns 404 when service returns None."""
        mock_service.get_schema.return_value = None
        resp = client.get(f"{BASE_URL}/schemas/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # --- PUT /schemas/{schema_id} ---

    def test_update_schema_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """PUT /schemas/{id} returns 200 with valid partial update."""
        body = {"owner": "new-team", "status": "deprecated"}
        resp = client.put(f"{BASE_URL}/schemas/abc-123", json=body)
        assert resp.status_code == 200
        mock_service.update_schema.assert_called_once()

    def test_update_schema_passes_non_none_fields(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """PUT /schemas/{id} only passes non-None fields to service."""
        body = {"owner": "team-b"}
        client.put(f"{BASE_URL}/schemas/abc-123", json=body)
        call_kwargs = mock_service.update_schema.call_args.kwargs
        assert call_kwargs["schema_id"] == "abc-123"
        assert call_kwargs["updates"] == {"owner": "team-b"}

    def test_update_schema_empty_body_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """PUT /schemas/{id} returns 400 when all fields are null (no updates)."""
        body: Dict[str, Any] = {}
        resp = client.put(f"{BASE_URL}/schemas/abc-123", json=body)
        assert resp.status_code == 400
        assert "no update fields" in resp.json()["detail"].lower()

    def test_update_schema_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """PUT /schemas/{id} returns 404 when service returns None."""
        mock_service.update_schema.return_value = None
        body = {"owner": "team-new"}
        resp = client.put(f"{BASE_URL}/schemas/abc-123", json=body)
        assert resp.status_code == 404

    # --- DELETE /schemas/{schema_id} ---

    def test_delete_schema_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """DELETE /schemas/{id} returns 200 on successful soft delete."""
        resp = client.delete(f"{BASE_URL}/schemas/abc-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["schema_id"] == "abc-123"
        mock_service.delete_schema.assert_called_once_with("abc-123")

    def test_delete_schema_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """DELETE /schemas/{id} returns 404 when schema not found."""
        mock_service.delete_schema.return_value = False
        resp = client.delete(f"{BASE_URL}/schemas/no-such-id")
        assert resp.status_code == 404


# ===========================================================================
# TestVersionEndpoints
# ===========================================================================


class TestVersionEndpoints:
    """Tests for version management endpoints (POST, GET list, GET detail)."""

    # --- POST /versions ---

    def test_create_version_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /versions returns 201 with valid body."""
        body = {
            "schema_id": str(uuid.uuid4()),
            "definition": {"type": "object", "properties": {"x": {"type": "string"}}},
            "changelog_note": "Added field x",
        }
        resp = client.post(f"{BASE_URL}/versions", json=body)
        assert resp.status_code == 201

    def test_create_version_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /versions forwards body fields to service.create_version."""
        sid = str(uuid.uuid4())
        defn = {"type": "object"}
        body = {
            "schema_id": sid,
            "definition": defn,
            "changelog_note": "v2",
        }
        client.post(f"{BASE_URL}/versions", json=body)
        mock_service.create_version.assert_called_once_with(
            schema_id=sid,
            definition=defn,
            changelog_note="v2",
        )

    def test_create_version_missing_schema_id_422(
        self, client: "TestClient"
    ):
        """POST /versions returns 422 when schema_id is missing."""
        body = {"definition": {"type": "object"}}
        resp = client.post(f"{BASE_URL}/versions", json=body)
        assert resp.status_code == 422

    def test_create_version_missing_definition_422(
        self, client: "TestClient"
    ):
        """POST /versions returns 422 when definition is missing."""
        body = {"schema_id": "abc"}
        resp = client.post(f"{BASE_URL}/versions", json=body)
        assert resp.status_code == 422

    # --- GET /versions ---

    def test_list_versions_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions returns 200 with paginated envelope."""
        resp = client.get(f"{BASE_URL}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert "versions" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data

    def test_list_versions_filter_by_schema_id(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions passes schema_id filter to service."""
        sid = str(uuid.uuid4())
        client.get(f"{BASE_URL}/versions", params={"schema_id": sid})
        call_kwargs = mock_service.list_versions.call_args.kwargs
        assert call_kwargs["schema_id"] == sid

    def test_list_versions_filter_by_deprecated(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions passes deprecated filter to service."""
        client.get(
            f"{BASE_URL}/versions", params={"deprecated": "true"}
        )
        call_kwargs = mock_service.list_versions.call_args.kwargs
        assert call_kwargs["deprecated"] is True

    def test_list_versions_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions passes custom limit and offset."""
        client.get(
            f"{BASE_URL}/versions",
            params={"limit": 25, "offset": 10},
        )
        call_kwargs = mock_service.list_versions.call_args.kwargs
        assert call_kwargs["limit"] == 25
        assert call_kwargs["offset"] == 10

    # --- GET /versions/{version_id} ---

    def test_get_version_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions/{id} returns 200 for an existing version."""
        vid = "ver-abc"
        resp = client.get(f"{BASE_URL}/versions/{vid}")
        assert resp.status_code == 200
        mock_service.get_version.assert_called_once_with(vid)

    def test_get_version_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions/{id} returns 404 when service returns None."""
        mock_service.get_version.return_value = None
        resp = client.get(f"{BASE_URL}/versions/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# TestChangeDetectionEndpoints
# ===========================================================================


class TestChangeDetectionEndpoints:
    """Tests for change detection endpoints."""

    # --- POST /changes/detect ---

    def test_detect_changes_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /changes/detect returns 201 with valid body."""
        body = {
            "source_version_id": str(uuid.uuid4()),
            "target_version_id": str(uuid.uuid4()),
        }
        resp = client.post(f"{BASE_URL}/changes/detect", json=body)
        assert resp.status_code == 201

    def test_detect_changes_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /changes/detect forwards IDs to service.detect_changes."""
        src = str(uuid.uuid4())
        tgt = str(uuid.uuid4())
        body = {"source_version_id": src, "target_version_id": tgt}
        client.post(f"{BASE_URL}/changes/detect", json=body)
        mock_service.detect_changes.assert_called_once_with(
            source_version_id=src, target_version_id=tgt
        )

    def test_detect_changes_missing_source_422(
        self, client: "TestClient"
    ):
        """POST /changes/detect returns 422 without source_version_id."""
        body = {"target_version_id": str(uuid.uuid4())}
        resp = client.post(f"{BASE_URL}/changes/detect", json=body)
        assert resp.status_code == 422

    def test_detect_changes_missing_target_422(
        self, client: "TestClient"
    ):
        """POST /changes/detect returns 422 without target_version_id."""
        body = {"source_version_id": str(uuid.uuid4())}
        resp = client.post(f"{BASE_URL}/changes/detect", json=body)
        assert resp.status_code == 422

    # --- GET /changes ---

    def test_list_changes_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /changes returns 200 with paginated envelope."""
        resp = client.get(f"{BASE_URL}/changes")
        assert resp.status_code == 200
        data = resp.json()
        assert "changes" in data
        assert "count" in data

    def test_list_changes_with_filters(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /changes passes filter params to service.list_changes."""
        client.get(
            f"{BASE_URL}/changes",
            params={
                "schema_id": "sid",
                "severity": "breaking",
                "change_type": "removed",
                "limit": 20,
                "offset": 5,
            },
        )
        call_kwargs = mock_service.list_changes.call_args.kwargs
        assert call_kwargs["schema_id"] == "sid"
        assert call_kwargs["severity"] == "breaking"
        assert call_kwargs["change_type"] == "removed"
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 5


# ===========================================================================
# TestCompatibilityEndpoints
# ===========================================================================


class TestCompatibilityEndpoints:
    """Tests for compatibility check endpoints."""

    # --- POST /compatibility/check ---

    def test_check_compatibility_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /compatibility/check returns 201."""
        body = {
            "source_version_id": str(uuid.uuid4()),
            "target_version_id": str(uuid.uuid4()),
            "level": "full",
        }
        resp = client.post(f"{BASE_URL}/compatibility/check", json=body)
        assert resp.status_code == 201

    def test_check_compatibility_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /compatibility/check forwards all params to service."""
        src = str(uuid.uuid4())
        tgt = str(uuid.uuid4())
        body = {
            "source_version_id": src,
            "target_version_id": tgt,
            "level": "forward",
        }
        client.post(f"{BASE_URL}/compatibility/check", json=body)
        mock_service.check_compatibility.assert_called_once_with(
            source_version_id=src,
            target_version_id=tgt,
            level="forward",
        )

    def test_check_compatibility_default_level(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /compatibility/check uses 'backward' as default level."""
        body = {
            "source_version_id": str(uuid.uuid4()),
            "target_version_id": str(uuid.uuid4()),
        }
        client.post(f"{BASE_URL}/compatibility/check", json=body)
        call_kwargs = mock_service.check_compatibility.call_args.kwargs
        assert call_kwargs["level"] == "backward"

    def test_check_compatibility_missing_fields_422(
        self, client: "TestClient"
    ):
        """POST /compatibility/check returns 422 when required fields are missing."""
        resp = client.post(f"{BASE_URL}/compatibility/check", json={})
        assert resp.status_code == 422

    # --- GET /compatibility ---

    def test_list_compatibility_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /compatibility returns 200 with paginated envelope."""
        resp = client.get(f"{BASE_URL}/compatibility")
        assert resp.status_code == 200
        data = resp.json()
        assert "compatibility_results" in data
        assert "count" in data

    def test_list_compatibility_with_filters(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /compatibility passes filter params to service."""
        client.get(
            f"{BASE_URL}/compatibility",
            params={
                "schema_id": "sid",
                "result": "breaking",
                "limit": 15,
                "offset": 3,
            },
        )
        call_kwargs = mock_service.list_compatibility_results.call_args.kwargs
        assert call_kwargs["schema_id"] == "sid"
        assert call_kwargs["result_filter"] == "breaking"
        assert call_kwargs["limit"] == 15
        assert call_kwargs["offset"] == 3


# ===========================================================================
# TestMigrationEndpoints
# ===========================================================================


class TestMigrationEndpoints:
    """Tests for migration plan and execution endpoints."""

    # --- POST /plans ---

    def test_create_plan_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /plans returns 201."""
        body = {
            "source_schema_id": str(uuid.uuid4()),
            "target_schema_id": str(uuid.uuid4()),
            "source_version": "1.0.0",
            "target_version": "2.0.0",
        }
        resp = client.post(f"{BASE_URL}/plans", json=body)
        assert resp.status_code == 201

    def test_create_plan_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /plans forwards all body fields to service.create_plan."""
        src_id = str(uuid.uuid4())
        tgt_id = str(uuid.uuid4())
        body = {
            "source_schema_id": src_id,
            "target_schema_id": tgt_id,
            "source_version": "1.0.0",
            "target_version": "2.0.0",
        }
        client.post(f"{BASE_URL}/plans", json=body)
        mock_service.create_plan.assert_called_once_with(
            source_schema_id=src_id,
            target_schema_id=tgt_id,
            source_version="1.0.0",
            target_version="2.0.0",
        )

    def test_create_plan_optional_versions(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /plans allows null source_version and target_version."""
        body = {
            "source_schema_id": str(uuid.uuid4()),
            "target_schema_id": str(uuid.uuid4()),
        }
        resp = client.post(f"{BASE_URL}/plans", json=body)
        assert resp.status_code == 201
        call_kwargs = mock_service.create_plan.call_args.kwargs
        assert call_kwargs["source_version"] is None
        assert call_kwargs["target_version"] is None

    def test_create_plan_missing_required_422(self, client: "TestClient"):
        """POST /plans returns 422 without source_schema_id."""
        body = {"target_schema_id": str(uuid.uuid4())}
        resp = client.post(f"{BASE_URL}/plans", json=body)
        assert resp.status_code == 422

    # --- GET /plans/{plan_id} ---

    def test_get_plan_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /plans/{id} returns 200 for an existing plan."""
        pid = "plan-abc"
        resp = client.get(f"{BASE_URL}/plans/{pid}")
        assert resp.status_code == 200
        mock_service.get_plan.assert_called_once_with(pid)

    def test_get_plan_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /plans/{id} returns 404 when service returns None."""
        mock_service.get_plan.return_value = None
        resp = client.get(f"{BASE_URL}/plans/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # --- POST /execute ---

    def test_execute_migration_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /execute returns 201."""
        body = {"plan_id": str(uuid.uuid4()), "dry_run": False}
        resp = client.post(f"{BASE_URL}/execute", json=body)
        assert resp.status_code == 201

    def test_execute_migration_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /execute forwards plan_id and dry_run to service."""
        pid = str(uuid.uuid4())
        body = {"plan_id": pid, "dry_run": True}
        client.post(f"{BASE_URL}/execute", json=body)
        mock_service.execute_migration.assert_called_once_with(
            plan_id=pid, dry_run=True
        )

    def test_execute_migration_dry_run_default_false(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /execute defaults dry_run to False."""
        body = {"plan_id": str(uuid.uuid4())}
        client.post(f"{BASE_URL}/execute", json=body)
        call_kwargs = mock_service.execute_migration.call_args.kwargs
        assert call_kwargs["dry_run"] is False

    # --- GET /executions/{execution_id} ---

    def test_get_execution_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /executions/{id} returns 200 for an existing execution."""
        eid = "exec-abc"
        resp = client.get(f"{BASE_URL}/executions/{eid}")
        assert resp.status_code == 200
        mock_service.get_execution.assert_called_once_with(eid)

    def test_get_execution_404(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /executions/{id} returns 404 when service returns None."""
        mock_service.get_execution.return_value = None
        resp = client.get(f"{BASE_URL}/executions/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # --- POST /rollback/{execution_id} ---

    def test_rollback_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /rollback/{id} returns 201 on success."""
        body = {"to_checkpoint": 2}
        resp = client.post(f"{BASE_URL}/rollback/exec-abc", json=body)
        assert resp.status_code == 201

    def test_rollback_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /rollback/{id} forwards execution_id and checkpoint to service."""
        body = {"to_checkpoint": 5}
        client.post(f"{BASE_URL}/rollback/exec-xyz", json=body)
        mock_service.rollback_execution.assert_called_once_with(
            execution_id="exec-xyz", to_checkpoint=5
        )

    def test_rollback_null_checkpoint(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /rollback/{id} allows null to_checkpoint for full rollback."""
        body: Dict[str, Any] = {}
        client.post(f"{BASE_URL}/rollback/exec-abc", json=body)
        call_kwargs = mock_service.rollback_execution.call_args.kwargs
        assert call_kwargs["to_checkpoint"] is None


# ===========================================================================
# TestPipelineEndpoints
# ===========================================================================


class TestPipelineEndpoints:
    """Tests for the full pipeline orchestration endpoint."""

    def test_run_pipeline_201(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /pipeline returns 201."""
        body = {
            "source_schema_id": str(uuid.uuid4()),
            "target_schema_id": str(uuid.uuid4()),
        }
        resp = client.post(f"{BASE_URL}/pipeline", json=body)
        assert resp.status_code == 201

    def test_run_pipeline_calls_service(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /pipeline forwards all body fields to service.run_pipeline."""
        src = str(uuid.uuid4())
        tgt = str(uuid.uuid4())
        body = {
            "source_schema_id": src,
            "target_schema_id": tgt,
            "source_version": "1.0.0",
            "target_version": "2.0.0",
            "skip_compatibility": True,
            "skip_dry_run": True,
        }
        client.post(f"{BASE_URL}/pipeline", json=body)
        mock_service.run_pipeline.assert_called_once_with(
            source_schema_id=src,
            target_schema_id=tgt,
            source_version="1.0.0",
            target_version="2.0.0",
            skip_compatibility=True,
            skip_dry_run=True,
        )

    def test_run_pipeline_default_skip_flags(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /pipeline defaults skip_compatibility and skip_dry_run to False."""
        body = {
            "source_schema_id": str(uuid.uuid4()),
            "target_schema_id": str(uuid.uuid4()),
        }
        client.post(f"{BASE_URL}/pipeline", json=body)
        call_kwargs = mock_service.run_pipeline.call_args.kwargs
        assert call_kwargs["skip_compatibility"] is False
        assert call_kwargs["skip_dry_run"] is False

    def test_run_pipeline_optional_versions(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /pipeline allows null source_version and target_version."""
        body = {
            "source_schema_id": str(uuid.uuid4()),
            "target_schema_id": str(uuid.uuid4()),
        }
        client.post(f"{BASE_URL}/pipeline", json=body)
        call_kwargs = mock_service.run_pipeline.call_args.kwargs
        assert call_kwargs["source_version"] is None
        assert call_kwargs["target_version"] is None

    def test_run_pipeline_missing_required_422(self, client: "TestClient"):
        """POST /pipeline returns 422 without required schema IDs."""
        resp = client.post(f"{BASE_URL}/pipeline", json={})
        assert resp.status_code == 422


# ===========================================================================
# TestHealthAndStats
# ===========================================================================


class TestHealthAndStats:
    """Tests for health check and statistics endpoints."""

    # --- GET /health ---

    def test_health_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /health returns 200."""
        resp = client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200

    def test_health_contains_status(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /health response contains a 'status' key."""
        resp = client.get(f"{BASE_URL}/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_engines(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /health response contains an 'engines' key."""
        resp = client.get(f"{BASE_URL}/health")
        data = resp.json()
        assert "engines" in data

    def test_health_returns_unhealthy_on_exception(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /health returns 200 with 'unhealthy' status when service raises."""
        mock_service.health_check.side_effect = RuntimeError("DB down")
        resp = client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert "error" in data

    # --- GET /stats ---

    def test_stats_200(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /stats returns 200."""
        resp = client.get(f"{BASE_URL}/stats")
        assert resp.status_code == 200

    def test_stats_contains_counters(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /stats response contains expected counter fields."""
        resp = client.get(f"{BASE_URL}/stats")
        data = resp.json()
        assert "total_schemas" in data
        assert "total_versions" in data
        assert "total_changes" in data
        assert "total_migrations" in data
        assert "total_rollbacks" in data
        assert "schemas_by_type" in data
        assert "schemas_by_status" in data


# ===========================================================================
# TestErrorHandling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling across all endpoints."""

    # --- ValueError -> 400 ---

    def test_register_schema_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /schemas returns 400 when service raises ValueError."""
        mock_service.register_schema.side_effect = ValueError(
            "Invalid namespace"
        )
        body = {
            "namespace": "bad!",
            "name": "test",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 400
        assert "Invalid namespace" in resp.json()["detail"]

    def test_create_version_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /versions returns 400 when service raises ValueError."""
        mock_service.create_version.side_effect = ValueError(
            "Schema not found"
        )
        body = {
            "schema_id": "nonexistent",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/versions", json=body)
        assert resp.status_code == 400

    def test_detect_changes_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /changes/detect returns 400 when service raises ValueError."""
        mock_service.detect_changes.side_effect = ValueError(
            "Source version not found"
        )
        body = {
            "source_version_id": "bad",
            "target_version_id": "bad",
        }
        resp = client.post(f"{BASE_URL}/changes/detect", json=body)
        assert resp.status_code == 400

    def test_check_compatibility_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /compatibility/check returns 400 on ValueError."""
        mock_service.check_compatibility.side_effect = ValueError("bad level")
        body = {
            "source_version_id": "a",
            "target_version_id": "b",
        }
        resp = client.post(f"{BASE_URL}/compatibility/check", json=body)
        assert resp.status_code == 400

    def test_create_plan_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /plans returns 400 on ValueError."""
        mock_service.create_plan.side_effect = ValueError("Invalid schema IDs")
        body = {
            "source_schema_id": "a",
            "target_schema_id": "b",
        }
        resp = client.post(f"{BASE_URL}/plans", json=body)
        assert resp.status_code == 400

    def test_execute_migration_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /execute returns 400 on ValueError."""
        mock_service.execute_migration.side_effect = ValueError("Plan not approved")
        body = {"plan_id": "p1"}
        resp = client.post(f"{BASE_URL}/execute", json=body)
        assert resp.status_code == 400

    def test_rollback_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /rollback/{id} returns 400 on ValueError."""
        mock_service.rollback_execution.side_effect = ValueError(
            "Invalid checkpoint"
        )
        resp = client.post(
            f"{BASE_URL}/rollback/exec-1", json={"to_checkpoint": -1}
        )
        assert resp.status_code == 400

    def test_pipeline_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /pipeline returns 400 on ValueError."""
        mock_service.run_pipeline.side_effect = ValueError("Bad config")
        body = {
            "source_schema_id": "a",
            "target_schema_id": "b",
        }
        resp = client.post(f"{BASE_URL}/pipeline", json=body)
        assert resp.status_code == 400

    def test_update_schema_valueerror_400(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """PUT /schemas/{id} returns 400 on ValueError."""
        mock_service.update_schema.side_effect = ValueError("Invalid status")
        body = {"status": "unknown"}
        resp = client.put(f"{BASE_URL}/schemas/abc", json=body)
        assert resp.status_code == 400

    # --- Exception -> 500 ---

    def test_register_schema_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /schemas returns 500 on unexpected Exception."""
        mock_service.register_schema.side_effect = RuntimeError("DB error")
        body = {
            "namespace": "ns.test",
            "name": "x",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 500
        assert "DB error" in resp.json()["detail"]

    def test_list_schemas_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas returns 500 on unexpected Exception."""
        mock_service.list_schemas.side_effect = RuntimeError("DB down")
        resp = client.get(f"{BASE_URL}/schemas")
        assert resp.status_code == 500

    def test_get_schema_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas/{id} returns 500 on unexpected Exception."""
        mock_service.get_schema.side_effect = RuntimeError("timeout")
        resp = client.get(f"{BASE_URL}/schemas/abc")
        assert resp.status_code == 500

    def test_delete_schema_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """DELETE /schemas/{id} returns 500 on unexpected Exception."""
        mock_service.delete_schema.side_effect = RuntimeError("disk full")
        resp = client.delete(f"{BASE_URL}/schemas/abc")
        assert resp.status_code == 500

    def test_create_version_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """POST /versions returns 500 on unexpected Exception."""
        mock_service.create_version.side_effect = RuntimeError("boom")
        body = {
            "schema_id": "sid",
            "definition": {"type": "object"},
        }
        resp = client.post(f"{BASE_URL}/versions", json=body)
        assert resp.status_code == 500

    def test_list_versions_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions returns 500 on unexpected Exception."""
        mock_service.list_versions.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/versions")
        assert resp.status_code == 500

    def test_get_version_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions/{id} returns 500 on unexpected Exception."""
        mock_service.get_version.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/versions/abc")
        assert resp.status_code == 500

    def test_list_changes_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /changes returns 500 on unexpected Exception."""
        mock_service.list_changes.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/changes")
        assert resp.status_code == 500

    def test_list_compatibility_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /compatibility returns 500 on unexpected Exception."""
        mock_service.list_compatibility_results.side_effect = RuntimeError(
            "error"
        )
        resp = client.get(f"{BASE_URL}/compatibility")
        assert resp.status_code == 500

    def test_get_plan_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /plans/{id} returns 500 on unexpected Exception."""
        mock_service.get_plan.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/plans/abc")
        assert resp.status_code == 500

    def test_get_execution_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /executions/{id} returns 500 on unexpected Exception."""
        mock_service.get_execution.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/executions/abc")
        assert resp.status_code == 500

    def test_stats_exception_500(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /stats returns 500 on unexpected Exception."""
        mock_service.get_stats.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE_URL}/stats")
        assert resp.status_code == 500

    # --- 422 validation errors ---

    def test_post_schemas_no_json_body_422(self, client: "TestClient"):
        """POST /schemas returns 422 when no body is sent."""
        resp = client.post(
            f"{BASE_URL}/schemas",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_post_execute_missing_plan_id_422(self, client: "TestClient"):
        """POST /execute returns 422 without required plan_id."""
        resp = client.post(f"{BASE_URL}/execute", json={})
        assert resp.status_code == 422


# ===========================================================================
# TestPagination
# ===========================================================================


class TestPagination:
    """Tests for pagination defaults and bounds across list endpoints."""

    def test_schemas_default_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas defaults to limit=50, offset=0."""
        client.get(f"{BASE_URL}/schemas")
        kw = mock_service.list_schemas.call_args.kwargs
        assert kw["limit"] == 50
        assert kw["offset"] == 0

    def test_versions_default_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions defaults to limit=50, offset=0."""
        client.get(f"{BASE_URL}/versions")
        kw = mock_service.list_versions.call_args.kwargs
        assert kw["limit"] == 50
        assert kw["offset"] == 0

    def test_changes_default_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /changes defaults to limit=50, offset=0."""
        client.get(f"{BASE_URL}/changes")
        kw = mock_service.list_changes.call_args.kwargs
        assert kw["limit"] == 50
        assert kw["offset"] == 0

    def test_compatibility_default_pagination(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /compatibility defaults to limit=50, offset=0."""
        client.get(f"{BASE_URL}/compatibility")
        kw = mock_service.list_compatibility_results.call_args.kwargs
        assert kw["limit"] == 50
        assert kw["offset"] == 0

    def test_custom_limit_and_offset(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """Custom limit and offset are forwarded correctly."""
        client.get(
            f"{BASE_URL}/schemas", params={"limit": 200, "offset": 100}
        )
        kw = mock_service.list_schemas.call_args.kwargs
        assert kw["limit"] == 200
        assert kw["offset"] == 100

    def test_limit_lower_bound(self, client: "TestClient"):
        """limit=0 is rejected (minimum is 1)."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"limit": 0}
        )
        assert resp.status_code == 422

    def test_limit_upper_bound(self, client: "TestClient"):
        """limit=1001 is rejected (maximum is 1000)."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"limit": 1001}
        )
        assert resp.status_code == 422

    def test_offset_lower_bound(self, client: "TestClient"):
        """offset=-1 is rejected (minimum is 0)."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"offset": -1}
        )
        assert resp.status_code == 422

    def test_limit_at_maximum(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """limit=1000 is accepted (boundary)."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"limit": 1000}
        )
        assert resp.status_code == 200
        kw = mock_service.list_schemas.call_args.kwargs
        assert kw["limit"] == 1000

    def test_limit_at_minimum(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """limit=1 is accepted (boundary)."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"limit": 1}
        )
        assert resp.status_code == 200
        kw = mock_service.list_schemas.call_args.kwargs
        assert kw["limit"] == 1

    def test_pagination_in_response_envelope(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """Pagination values appear in the response envelope."""
        resp = client.get(
            f"{BASE_URL}/schemas", params={"limit": 25, "offset": 10}
        )
        data = resp.json()
        assert data["limit"] == 25
        assert data["offset"] == 10


# ===========================================================================
# TestServiceNotConfigured
# ===========================================================================


class TestServiceNotConfigured:
    """Tests for the case when schema_migration_service is not set on app state."""

    def test_no_service_503(self):
        """Endpoints return 503 when service is not configured on app state."""
        from greenlang.schema_migration.api.router import (
            create_schema_migration_router,
        )

        app = FastAPI()
        # Do NOT set app.state.schema_migration_service
        router = create_schema_migration_router()
        app.include_router(router)
        no_svc_client = TestClient(app)

        resp = no_svc_client.get(f"{BASE_URL}/health")
        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"].lower()

    def test_no_service_503_on_post(self):
        """POST endpoints return 503 when service is not configured."""
        from greenlang.schema_migration.api.router import (
            create_schema_migration_router,
        )

        app = FastAPI()
        router = create_schema_migration_router()
        app.include_router(router)
        no_svc_client = TestClient(app)

        body = {
            "namespace": "ns.test",
            "name": "x",
            "definition": {"type": "object"},
        }
        resp = no_svc_client.post(f"{BASE_URL}/schemas", json=body)
        assert resp.status_code == 503


# ===========================================================================
# TestRouterFactoryDirectService
# ===========================================================================


class TestRouterFactoryDirectService:
    """Tests for the router factory when a service is injected directly."""

    def test_direct_service_injection(self):
        """create_schema_migration_router(service=...) uses injected service."""
        from greenlang.schema_migration.api.router import (
            create_schema_migration_router,
        )

        mock_svc = MagicMock()
        mock_svc.health_check.return_value = {
            "status": "healthy",
            "engines": {},
        }

        app = FastAPI()
        router = create_schema_migration_router(service=mock_svc)
        app.include_router(router)
        direct_client = TestClient(app)

        resp = direct_client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        mock_svc.health_check.assert_called_once()

    def test_direct_service_ignores_app_state(self):
        """When service is injected directly, app.state is ignored."""
        from greenlang.schema_migration.api.router import (
            create_schema_migration_router,
        )

        direct_svc = MagicMock()
        direct_svc.get_stats.return_value = _stats_dict(total_schemas=42)

        state_svc = MagicMock()
        state_svc.get_stats.return_value = _stats_dict(total_schemas=0)

        app = FastAPI()
        app.state.schema_migration_service = state_svc
        router = create_schema_migration_router(service=direct_svc)
        app.include_router(router)
        direct_client = TestClient(app)

        resp = direct_client.get(f"{BASE_URL}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_schemas"] == 42
        direct_svc.get_stats.assert_called_once()
        state_svc.get_stats.assert_not_called()


# ===========================================================================
# TestEndpointResponseStructure
# ===========================================================================


class TestEndpointResponseStructure:
    """Tests verifying the shape of JSON responses from various endpoints."""

    def test_list_schemas_envelope_keys(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas envelope contains schemas, count, limit, offset."""
        resp = client.get(f"{BASE_URL}/schemas")
        keys = set(resp.json().keys())
        assert keys == {"schemas", "count", "limit", "offset"}

    def test_list_versions_envelope_keys(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /versions envelope contains versions, count, limit, offset."""
        resp = client.get(f"{BASE_URL}/versions")
        keys = set(resp.json().keys())
        assert keys == {"versions", "count", "limit", "offset"}

    def test_list_changes_envelope_keys(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /changes envelope contains changes, count, limit, offset."""
        resp = client.get(f"{BASE_URL}/changes")
        keys = set(resp.json().keys())
        assert keys == {"changes", "count", "limit", "offset"}

    def test_list_compatibility_envelope_keys(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /compatibility envelope contains compatibility_results, count, limit, offset."""
        resp = client.get(f"{BASE_URL}/compatibility")
        keys = set(resp.json().keys())
        assert keys == {"compatibility_results", "count", "limit", "offset"}

    def test_delete_schema_response_keys(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """DELETE /schemas/{id} returns {deleted, schema_id}."""
        resp = client.delete(f"{BASE_URL}/schemas/abc")
        keys = set(resp.json().keys())
        assert keys == {"deleted", "schema_id"}

    def test_empty_list_returns_zero_count(
        self, client: "TestClient", mock_service: MagicMock
    ):
        """GET /schemas returns count=0 when service returns empty list."""
        mock_service.list_schemas.return_value = []
        resp = client.get(f"{BASE_URL}/schemas")
        data = resp.json()
        assert data["count"] == 0
        assert data["schemas"] == []
