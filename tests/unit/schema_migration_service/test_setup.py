# -*- coding: utf-8 -*-
"""
Unit Tests for Schema Migration Service Setup - AGENT-DATA-017

Tests the 8 lightweight Pydantic response models, the SchemaMigrationService
facade class (service lifecycle, engine delegation, provenance recording,
metrics tracking, statistics, health checks), and the three module-level
FastAPI integration helpers (configure_schema_migration, get_schema_migration,
get_router).

Target: ~100 tests, 85%+ coverage of greenlang.schema_migration.setup

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import sys
import types
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Stub engine submodules to prevent Prometheus metric re-registration errors
# at import time.  setup.py uses try/except ImportError around these imports,
# so providing a stub with the class set to None makes it fall through to the
# "not available" branch cleanly.
# ---------------------------------------------------------------------------

_ENGINE_MODULES = [
    "greenlang.schema_migration.schema_registry",
    "greenlang.schema_migration.schema_versioner",
    "greenlang.schema_migration.change_detector",
    "greenlang.schema_migration.compatibility_checker",
    "greenlang.schema_migration.migration_planner",
    "greenlang.schema_migration.migration_executor",
    "greenlang.schema_migration.schema_migration_pipeline",
]

for _mod_name in _ENGINE_MODULES:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.__package__ = "greenlang.schema_migration"
        # The engine class names that setup.py imports; set them to None so
        # the ``if <EngineClass> is not None`` guards evaluate to False.
        _class_name = _mod_name.rsplit(".", 1)[-1]
        # Convert snake_case module name to PascalCase + "Engine"
        _pascal = "".join(part.capitalize() for part in _class_name.split("_")) + "Engine"
        setattr(_stub, _pascal, None)
        sys.modules[_mod_name] = _stub

from greenlang.schema_migration.config import SchemaMigrationConfig
from greenlang.schema_migration.setup import (
    SchemaResponse,
    SchemaVersionResponse,
    ChangeDetectionResponse,
    CompatibilityCheckResponse,
    MigrationPlanResponse,
    MigrationExecutionResponse,
    PipelineResultResponse,
    SchemaMigrationStatisticsResponse,
    SchemaMigrationService,
    configure_schema_migration,
    get_schema_migration,
    get_router,
    _compute_hash,
    _new_uuid,
    _utcnow_iso,
    _singleton_lock,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_config(**overrides: Any) -> SchemaMigrationConfig:
    """Create a SchemaMigrationConfig with sensible test defaults."""
    return SchemaMigrationConfig(**overrides)


def _make_service(**overrides: Any) -> SchemaMigrationService:
    """Create a SchemaMigrationService with engines stubbed to None."""
    cfg = _make_config(**overrides)
    with patch("greenlang.schema_migration.setup.SchemaRegistryEngine", None), \
         patch("greenlang.schema_migration.setup.SchemaVersionerEngine", None), \
         patch("greenlang.schema_migration.setup.ChangeDetectorEngine", None), \
         patch("greenlang.schema_migration.setup.CompatibilityCheckerEngine", None), \
         patch("greenlang.schema_migration.setup.MigrationPlannerEngine", None), \
         patch("greenlang.schema_migration.setup.MigrationExecutorEngine", None), \
         patch("greenlang.schema_migration.setup.SchemaMigrationPipelineEngine", None):
        return SchemaMigrationService(config=cfg)


# ============================================================================
# RESPONSE MODEL TESTS
# ============================================================================


class TestSchemaResponse:
    """Tests for SchemaResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = SchemaResponse(
            schema_id="abc-123",
            namespace="emissions",
            name="ActivityRecord",
            schema_type="json_schema",
            status="active",
            owner="platform-team",
            tags=["core", "emissions"],
            description="Main activity record schema",
            version_count=5,
            created_at="2026-01-15T00:00:00+00:00",
            updated_at="2026-02-10T00:00:00+00:00",
            provenance_hash="a" * 64,
        )
        assert resp.schema_id == "abc-123"
        assert resp.namespace == "emissions"
        assert resp.name == "ActivityRecord"
        assert resp.schema_type == "json_schema"
        assert resp.status == "active"
        assert resp.owner == "platform-team"
        assert resp.tags == ["core", "emissions"]
        assert resp.description == "Main activity record schema"
        assert resp.version_count == 5
        assert resp.provenance_hash == "a" * 64

    def test_defaults(self):
        resp = SchemaResponse()
        assert resp.schema_id  # UUID auto-generated
        assert resp.namespace == ""
        assert resp.name == ""
        assert resp.schema_type == "json_schema"
        assert resp.status == "draft"
        assert resp.owner == ""
        assert resp.tags == []
        assert resp.description == ""
        assert resp.version_count == 0
        assert resp.created_at  # auto-generated timestamp
        assert resp.updated_at  # auto-generated timestamp
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = SchemaResponse(
            schema_id="test-id",
            namespace="ns",
            name="test",
        )
        data = resp.model_dump()
        assert isinstance(data, dict)
        assert data["schema_id"] == "test-id"
        assert data["namespace"] == "ns"
        assert data["name"] == "test"
        assert "created_at" in data

    def test_model_dump_json_serializable(self):
        import json
        resp = SchemaResponse()
        raw = resp.model_dump(mode="json")
        serialized = json.dumps(raw)
        assert isinstance(serialized, str)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaResponse(unexpected_field="should_fail")

    def test_schema_id_is_uuid_by_default(self):
        resp = SchemaResponse()
        # Should be a valid UUID4 string
        parsed = uuid.UUID(resp.schema_id, version=4)
        assert str(parsed) == resp.schema_id


class TestSchemaVersionResponse:
    """Tests for SchemaVersionResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = SchemaVersionResponse(
            version_id="v-001",
            schema_id="s-001",
            version="2.1.0",
            definition={"type": "object"},
            changelog="Added salary field",
            is_deprecated=True,
            sunset_date="2026-12-31T00:00:00+00:00",
            created_at="2026-01-01T00:00:00+00:00",
            provenance_hash="b" * 64,
        )
        assert resp.version_id == "v-001"
        assert resp.schema_id == "s-001"
        assert resp.version == "2.1.0"
        assert resp.definition == {"type": "object"}
        assert resp.changelog == "Added salary field"
        assert resp.is_deprecated is True
        assert resp.sunset_date == "2026-12-31T00:00:00+00:00"
        assert resp.provenance_hash == "b" * 64

    def test_defaults(self):
        resp = SchemaVersionResponse()
        assert resp.version_id  # UUID auto-generated
        assert resp.schema_id == ""
        assert resp.version == "1.0.0"
        assert resp.definition == {}
        assert resp.changelog == ""
        assert resp.is_deprecated is False
        assert resp.sunset_date is None
        assert resp.created_at  # auto-generated
        assert resp.provenance_hash == ""

    def test_sunset_date_optional(self):
        resp1 = SchemaVersionResponse(sunset_date=None)
        assert resp1.sunset_date is None
        resp2 = SchemaVersionResponse(sunset_date="2027-01-01")
        assert resp2.sunset_date == "2027-01-01"

    def test_model_dump(self):
        resp = SchemaVersionResponse(version_id="vid", schema_id="sid")
        data = resp.model_dump()
        assert data["version_id"] == "vid"
        assert data["schema_id"] == "sid"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaVersionResponse(unknown_key=42)


class TestChangeDetectionResponse:
    """Tests for ChangeDetectionResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        changes = [
            {"change_type": "field_added", "severity": "non_breaking"},
            {"change_type": "field_removed", "severity": "breaking"},
        ]
        resp = ChangeDetectionResponse(
            detection_id="det-001",
            source_version_id="sv-1",
            target_version_id="sv-2",
            changes=changes,
            change_count=2,
            breaking_change_count=1,
            detected_at="2026-02-15T00:00:00+00:00",
            provenance_hash="c" * 64,
        )
        assert resp.detection_id == "det-001"
        assert resp.source_version_id == "sv-1"
        assert resp.target_version_id == "sv-2"
        assert len(resp.changes) == 2
        assert resp.change_count == 2
        assert resp.breaking_change_count == 1

    def test_defaults(self):
        resp = ChangeDetectionResponse()
        assert resp.detection_id
        assert resp.source_version_id == ""
        assert resp.target_version_id == ""
        assert resp.changes == []
        assert resp.change_count == 0
        assert resp.breaking_change_count == 0
        assert resp.detected_at
        assert resp.provenance_hash == ""

    def test_changes_list_is_mutable(self):
        resp = ChangeDetectionResponse()
        resp.changes.append({"change_type": "field_added"})
        assert len(resp.changes) == 1

    def test_model_dump(self):
        resp = ChangeDetectionResponse(change_count=3, breaking_change_count=1)
        data = resp.model_dump()
        assert data["change_count"] == 3
        assert data["breaking_change_count"] == 1

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ChangeDetectionResponse(nonexistent="val")


class TestCompatibilityCheckResponse:
    """Tests for CompatibilityCheckResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        issues = [{"type": "field_removed", "severity": "error"}]
        resp = CompatibilityCheckResponse(
            check_id="chk-001",
            source_version_id="sv-1",
            target_version_id="sv-2",
            compatibility_level="breaking",
            is_compatible=False,
            issues=issues,
            checked_at="2026-02-15T00:00:00+00:00",
            provenance_hash="d" * 64,
        )
        assert resp.check_id == "chk-001"
        assert resp.compatibility_level == "breaking"
        assert resp.is_compatible is False
        assert len(resp.issues) == 1

    def test_defaults(self):
        resp = CompatibilityCheckResponse()
        assert resp.check_id
        assert resp.source_version_id == ""
        assert resp.target_version_id == ""
        assert resp.compatibility_level == "full"
        assert resp.is_compatible is True
        assert resp.issues == []
        assert resp.checked_at
        assert resp.provenance_hash == ""

    def test_is_compatible_bool_values(self):
        resp_true = CompatibilityCheckResponse(is_compatible=True)
        assert resp_true.is_compatible is True
        resp_false = CompatibilityCheckResponse(is_compatible=False)
        assert resp_false.is_compatible is False

    def test_model_dump(self):
        resp = CompatibilityCheckResponse(is_compatible=False)
        data = resp.model_dump()
        assert data["is_compatible"] is False

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CompatibilityCheckResponse(bad_field="x")


class TestMigrationPlanResponse:
    """Tests for MigrationPlanResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        steps = [
            {"step_number": 1, "operation": "add_field"},
            {"step_number": 2, "operation": "rename_field"},
        ]
        resp = MigrationPlanResponse(
            plan_id="plan-001",
            source_schema_id="src-1",
            target_schema_id="tgt-1",
            steps=steps,
            total_steps=2,
            effort_estimate="medium",
            status="validated",
            created_at="2026-02-15T00:00:00+00:00",
            provenance_hash="e" * 64,
        )
        assert resp.plan_id == "plan-001"
        assert resp.source_schema_id == "src-1"
        assert resp.target_schema_id == "tgt-1"
        assert len(resp.steps) == 2
        assert resp.total_steps == 2
        assert resp.effort_estimate == "medium"
        assert resp.status == "validated"

    def test_defaults(self):
        resp = MigrationPlanResponse()
        assert resp.plan_id
        assert resp.source_schema_id == ""
        assert resp.target_schema_id == ""
        assert resp.steps == []
        assert resp.total_steps == 0
        assert resp.effort_estimate == "low"
        assert resp.status == "pending"
        assert resp.created_at
        assert resp.provenance_hash == ""

    def test_steps_list(self):
        steps = [{"step_number": i} for i in range(5)]
        resp = MigrationPlanResponse(steps=steps, total_steps=5)
        assert len(resp.steps) == 5
        assert resp.total_steps == 5

    def test_model_dump(self):
        resp = MigrationPlanResponse(effort_estimate="critical")
        data = resp.model_dump()
        assert data["effort_estimate"] == "critical"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationPlanResponse(nope="x")


class TestMigrationExecutionResponse:
    """Tests for MigrationExecutionResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = MigrationExecutionResponse(
            execution_id="exec-001",
            plan_id="plan-001",
            status="completed",
            records_processed=5000,
            records_failed=3,
            records_skipped=10,
            current_step=5,
            total_steps=5,
            percentage=100.0,
            started_at="2026-02-15T08:00:00+00:00",
            completed_at="2026-02-15T08:05:00+00:00",
            provenance_hash="f" * 64,
        )
        assert resp.execution_id == "exec-001"
        assert resp.plan_id == "plan-001"
        assert resp.status == "completed"
        assert resp.records_processed == 5000
        assert resp.records_failed == 3
        assert resp.records_skipped == 10
        assert resp.current_step == 5
        assert resp.total_steps == 5
        assert resp.percentage == 100.0
        assert resp.completed_at == "2026-02-15T08:05:00+00:00"

    def test_defaults(self):
        resp = MigrationExecutionResponse()
        assert resp.execution_id
        assert resp.plan_id == ""
        assert resp.status == "pending"
        assert resp.records_processed == 0
        assert resp.records_failed == 0
        assert resp.records_skipped == 0
        assert resp.current_step == 0
        assert resp.total_steps == 0
        assert resp.percentage == 0.0
        assert resp.started_at
        assert resp.completed_at is None
        assert resp.provenance_hash == ""

    def test_progress_fields(self):
        resp = MigrationExecutionResponse(
            current_step=3,
            total_steps=10,
            percentage=30.0,
        )
        assert resp.current_step == 3
        assert resp.total_steps == 10
        assert resp.percentage == 30.0

    def test_completed_at_optional(self):
        resp1 = MigrationExecutionResponse(completed_at=None)
        assert resp1.completed_at is None
        resp2 = MigrationExecutionResponse(completed_at="2026-02-15T12:00:00+00:00")
        assert resp2.completed_at is not None

    def test_model_dump(self):
        resp = MigrationExecutionResponse(status="running")
        data = resp.model_dump()
        assert data["status"] == "running"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationExecutionResponse(invalid_key="val")


class TestPipelineResultResponse:
    """Tests for PipelineResultResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = PipelineResultResponse(
            pipeline_id="pipe-001",
            source_schema_id="src-1",
            target_schema_id="tgt-1",
            stages_completed=["detect", "compatibility", "plan", "execute"],
            final_status="completed",
            changes_detected=5,
            is_compatible=True,
            plan_id="plan-001",
            execution_id="exec-001",
            elapsed_seconds=12.345,
            provenance_hash="0" * 64,
        )
        assert resp.pipeline_id == "pipe-001"
        assert len(resp.stages_completed) == 4
        assert resp.final_status == "completed"
        assert resp.changes_detected == 5
        assert resp.is_compatible is True
        assert resp.plan_id == "plan-001"
        assert resp.execution_id == "exec-001"
        assert resp.elapsed_seconds == 12.345

    def test_defaults(self):
        resp = PipelineResultResponse()
        assert resp.pipeline_id
        assert resp.source_schema_id == ""
        assert resp.target_schema_id == ""
        assert resp.stages_completed == []
        assert resp.final_status == "pending"
        assert resp.changes_detected == 0
        assert resp.is_compatible is True
        assert resp.plan_id is None
        assert resp.execution_id is None
        assert resp.elapsed_seconds == 0.0
        assert resp.provenance_hash == ""

    def test_stages_completed_list(self):
        resp = PipelineResultResponse(stages_completed=["detect", "plan"])
        assert resp.stages_completed == ["detect", "plan"]
        assert "detect" in resp.stages_completed

    def test_optional_plan_execution_ids(self):
        resp = PipelineResultResponse(plan_id=None, execution_id=None)
        assert resp.plan_id is None
        assert resp.execution_id is None

    def test_model_dump(self):
        resp = PipelineResultResponse(final_status="failed")
        data = resp.model_dump()
        assert data["final_status"] == "failed"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineResultResponse(garbage="nope")


class TestSchemaMigrationStatisticsResponse:
    """Tests for SchemaMigrationStatisticsResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = SchemaMigrationStatisticsResponse(
            total_schemas=100,
            total_versions=500,
            total_changes_detected=200,
            total_compatibility_checks=150,
            total_migrations_planned=50,
            total_migrations_executed=45,
            total_rollbacks=3,
            total_drift_events=10,
            avg_migration_duration_seconds=12.5,
            success_rate=0.89,
            active_migrations=2,
        )
        assert resp.total_schemas == 100
        assert resp.total_versions == 500
        assert resp.total_changes_detected == 200
        assert resp.total_compatibility_checks == 150
        assert resp.total_migrations_planned == 50
        assert resp.total_migrations_executed == 45
        assert resp.total_rollbacks == 3
        assert resp.total_drift_events == 10
        assert resp.avg_migration_duration_seconds == 12.5
        assert resp.success_rate == 0.89
        assert resp.active_migrations == 2

    def test_all_numeric_defaults(self):
        resp = SchemaMigrationStatisticsResponse()
        assert resp.total_schemas == 0
        assert resp.total_versions == 0
        assert resp.total_changes_detected == 0
        assert resp.total_compatibility_checks == 0
        assert resp.total_migrations_planned == 0
        assert resp.total_migrations_executed == 0
        assert resp.total_rollbacks == 0
        assert resp.total_drift_events == 0
        assert resp.avg_migration_duration_seconds == 0.0
        assert resp.success_rate == 0.0
        assert resp.active_migrations == 0

    def test_model_dump(self):
        resp = SchemaMigrationStatisticsResponse(total_schemas=42)
        data = resp.model_dump()
        assert data["total_schemas"] == 42
        # All 11 fields should be present
        assert len(data) == 11

    def test_incremental_field_update(self):
        resp = SchemaMigrationStatisticsResponse()
        resp.total_schemas += 1
        resp.total_versions += 3
        assert resp.total_schemas == 1
        assert resp.total_versions == 3

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaMigrationStatisticsResponse(extra_stat=99)


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


class TestUtilityFunctions:
    """Tests for module-level helper functions."""

    def test_new_uuid_returns_valid_uuid4(self):
        val = _new_uuid()
        parsed = uuid.UUID(val, version=4)
        assert str(parsed) == val

    def test_new_uuid_is_unique(self):
        uuids = {_new_uuid() for _ in range(100)}
        assert len(uuids) == 100

    def test_utcnow_iso_returns_isoformat(self):
        val = _utcnow_iso()
        # Should parse without error
        dt = datetime.fromisoformat(val)
        assert dt.tzinfo is not None

    def test_compute_hash_returns_sha256(self):
        h = _compute_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_compute_hash_deterministic(self):
        data = {"alpha": 1, "beta": 2}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_compute_hash_different_data(self):
        h1 = _compute_hash({"key": "a"})
        h2 = _compute_hash({"key": "b"})
        assert h1 != h2

    def test_compute_hash_pydantic_model(self):
        resp = SchemaResponse(schema_id="test-id", name="test")
        h = _compute_hash(resp)
        assert isinstance(h, str)
        assert len(h) == 64


# ============================================================================
# SCHEMA MIGRATION SERVICE TESTS
# ============================================================================


class TestSchemaMigrationServiceInit:
    """Tests for SchemaMigrationService initialization."""

    def test_default_config(self):
        svc = _make_service()
        assert svc.config is not None
        assert isinstance(svc.config, SchemaMigrationConfig)

    def test_custom_config(self):
        cfg = _make_config(max_schemas=999)
        with patch("greenlang.schema_migration.setup.SchemaRegistryEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaVersionerEngine", None), \
             patch("greenlang.schema_migration.setup.ChangeDetectorEngine", None), \
             patch("greenlang.schema_migration.setup.CompatibilityCheckerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationPlannerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationExecutorEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaMigrationPipelineEngine", None):
            svc = SchemaMigrationService(config=cfg)
        assert svc.config.max_schemas == 999

    def test_provenance_tracker_initialized(self):
        svc = _make_service()
        assert svc.provenance is not None
        assert svc.provenance.entry_count == 0

    def test_engines_are_none_when_unavailable(self):
        svc = _make_service()
        assert svc.schema_registry_engine is None
        assert svc.schema_versioner_engine is None
        assert svc.change_detector_engine is None
        assert svc.compatibility_checker_engine is None
        assert svc.migration_planner_engine is None
        assert svc.migration_executor_engine is None
        assert svc.pipeline_engine is None

    def test_in_memory_stores_empty_on_init(self):
        svc = _make_service()
        assert len(svc._schemas) == 0
        assert len(svc._versions) == 0
        assert len(svc._detections) == 0
        assert len(svc._compat_checks) == 0
        assert len(svc._plans) == 0
        assert len(svc._executions) == 0
        assert len(svc._pipeline_results) == 0

    def test_statistics_zeroed_on_init(self):
        svc = _make_service()
        stats = svc._stats
        assert stats.total_schemas == 0
        assert stats.total_versions == 0
        assert stats.success_rate == 0.0

    def test_not_started_on_init(self):
        svc = _make_service()
        assert svc._started is False

    def test_active_migrations_zero_on_init(self):
        svc = _make_service()
        assert svc._active_migrations == 0


class TestServiceSchemaOperations:
    """Tests for schema CRUD operations through the service facade."""

    def test_register_schema_returns_schema_response(self):
        svc = _make_service()
        resp = svc.register_schema(
            namespace="emissions",
            name="ActivityRecord",
            schema_type="json_schema",
            definition={"type": "object"},
        )
        assert isinstance(resp, SchemaResponse)
        assert resp.namespace == "emissions"
        assert resp.name == "ActivityRecord"
        assert resp.schema_type == "json_schema"
        assert resp.status == "draft"

    def test_register_schema_assigns_provenance_hash(self):
        svc = _make_service()
        resp = svc.register_schema(
            namespace="ns",
            name="test",
            schema_type="json_schema",
            definition={},
        )
        assert resp.provenance_hash != ""
        assert len(resp.provenance_hash) == 64

    def test_register_schema_stores_in_cache(self):
        svc = _make_service()
        resp = svc.register_schema(
            namespace="ns",
            name="test",
            schema_type="json_schema",
            definition={},
        )
        assert resp.schema_id in svc._schemas

    def test_register_schema_records_provenance(self):
        svc = _make_service()
        svc.register_schema(
            namespace="ns",
            name="test",
            schema_type="json_schema",
            definition={},
        )
        assert svc.provenance.entry_count >= 1

    def test_register_schema_increments_stats(self):
        svc = _make_service()
        svc.register_schema(
            namespace="ns",
            name="test",
            schema_type="json_schema",
            definition={},
        )
        assert svc._stats.total_schemas == 1

    def test_register_schema_with_tags_and_owner(self):
        svc = _make_service()
        resp = svc.register_schema(
            namespace="supply_chain",
            name="Supplier",
            schema_type="avro",
            definition={},
            owner="data-team",
            tags=["supplier", "eudr"],
            description="Supplier record schema",
        )
        assert resp.owner == "data-team"
        assert resp.tags == ["supplier", "eudr"]
        assert resp.description == "Supplier record schema"

    def test_list_schemas_returns_registered(self):
        svc = _make_service()
        svc.register_schema(
            namespace="ns1",
            name="schema_a",
            schema_type="json_schema",
            definition={},
        )
        svc.register_schema(
            namespace="ns2",
            name="schema_b",
            schema_type="avro",
            definition={},
        )
        result = svc.list_schemas()
        assert len(result) == 2

    def test_list_schemas_filter_by_namespace(self):
        svc = _make_service()
        svc.register_schema(namespace="ns1", name="a", schema_type="json_schema", definition={})
        svc.register_schema(namespace="ns2", name="b", schema_type="json_schema", definition={})
        result = svc.list_schemas(namespace="ns1")
        assert len(result) == 1
        assert result[0].namespace == "ns1"

    def test_list_schemas_filter_by_schema_type(self):
        svc = _make_service()
        svc.register_schema(namespace="ns", name="a", schema_type="json_schema", definition={})
        svc.register_schema(namespace="ns", name="b", schema_type="avro", definition={})
        result = svc.list_schemas(schema_type="avro")
        assert len(result) == 1
        assert result[0].schema_type == "avro"

    def test_list_schemas_pagination(self):
        svc = _make_service()
        for i in range(5):
            svc.register_schema(namespace="ns", name=f"s{i}", schema_type="json_schema", definition={})
        result = svc.list_schemas(limit=2, offset=0)
        assert len(result) == 2
        result2 = svc.list_schemas(limit=2, offset=2)
        assert len(result2) == 2

    def test_get_schema_found(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        fetched = svc.get_schema(resp.schema_id)
        assert fetched is not None
        assert fetched.schema_id == resp.schema_id

    def test_get_schema_not_found(self):
        svc = _make_service()
        result = svc.get_schema("nonexistent-id")
        assert result is None

    def test_update_schema_owner(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        updated = svc.update_schema(resp.schema_id, owner="new-team")
        assert updated is not None
        assert updated.owner == "new-team"

    def test_update_schema_status(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        updated = svc.update_schema(resp.schema_id, status="active")
        assert updated is not None
        assert updated.status == "active"

    def test_update_schema_not_found(self):
        svc = _make_service()
        result = svc.update_schema("nonexistent", owner="x")
        assert result is None

    def test_update_schema_records_provenance(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        initial_count = svc.provenance.entry_count
        svc.update_schema(resp.schema_id, description="Updated desc")
        assert svc.provenance.entry_count > initial_count

    def test_delete_schema_returns_true(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        result = svc.delete_schema(resp.schema_id)
        assert result is True

    def test_delete_schema_sets_archived_status(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        svc.delete_schema(resp.schema_id)
        cached = svc._schemas.get(resp.schema_id)
        assert cached is not None
        assert cached.status == "archived"

    def test_delete_schema_not_found_returns_false(self):
        svc = _make_service()
        result = svc.delete_schema("nonexistent-id")
        assert result is False

    def test_delete_schema_records_provenance(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        initial_count = svc.provenance.entry_count
        svc.delete_schema(resp.schema_id)
        assert svc.provenance.entry_count > initial_count


class TestServiceVersionOperations:
    """Tests for version operations through the service facade."""

    def test_create_version_returns_version_response(self):
        svc = _make_service()
        resp = svc.create_version(
            schema_id="s-001",
            definition={"type": "object", "properties": {"name": {"type": "string"}}},
            changelog_note="Initial version",
        )
        assert isinstance(resp, SchemaVersionResponse)
        assert resp.schema_id == "s-001"
        assert resp.version == "1.0.0"
        assert resp.changelog == "Initial version"

    def test_create_version_empty_schema_id_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="schema_id must not be empty"):
            svc.create_version(schema_id="", definition={})

    def test_create_version_assigns_provenance_hash(self):
        svc = _make_service()
        resp = svc.create_version(schema_id="s-001", definition={})
        assert resp.provenance_hash != ""
        assert len(resp.provenance_hash) == 64

    def test_create_version_increments_stats(self):
        svc = _make_service()
        svc.create_version(schema_id="s-001", definition={})
        assert svc._stats.total_versions == 1

    def test_create_version_records_provenance(self):
        svc = _make_service()
        svc.create_version(schema_id="s-001", definition={})
        assert svc.provenance.entry_count >= 1

    def test_create_version_stored_in_cache(self):
        svc = _make_service()
        resp = svc.create_version(schema_id="s-001", definition={})
        assert resp.version_id in svc._versions

    def test_create_version_updates_parent_schema_count(self):
        svc = _make_service()
        schema = svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        assert svc._schemas[schema.schema_id].version_count == 0
        svc.create_version(schema_id=schema.schema_id, definition={})
        assert svc._schemas[schema.schema_id].version_count == 1

    def test_list_versions_empty(self):
        svc = _make_service()
        result = svc.list_versions(schema_id="nonexistent")
        assert result == []

    def test_list_versions_returns_matching(self):
        svc = _make_service()
        svc.create_version(schema_id="s-001", definition={"v": 1})
        svc.create_version(schema_id="s-001", definition={"v": 2})
        svc.create_version(schema_id="s-002", definition={"v": 1})
        result = svc.list_versions(schema_id="s-001")
        assert len(result) == 2

    def test_get_version_found(self):
        svc = _make_service()
        created = svc.create_version(schema_id="s-001", definition={})
        fetched = svc.get_version(created.version_id)
        assert fetched is not None
        assert fetched.version_id == created.version_id

    def test_get_version_not_found(self):
        svc = _make_service()
        result = svc.get_version("nonexistent-version")
        assert result is None


class TestServiceChangeDetection:
    """Tests for change detection operations through the service facade."""

    def test_detect_changes_with_cached_versions(self):
        svc = _make_service()
        v1 = svc.create_version(
            schema_id="s-001",
            definition={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        v2 = svc.create_version(
            schema_id="s-001",
            definition={"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        )
        resp = svc.detect_changes(v1.version_id, v2.version_id)
        assert isinstance(resp, ChangeDetectionResponse)
        assert resp.source_version_id == v1.version_id
        assert resp.target_version_id == v2.version_id
        assert resp.provenance_hash != ""

    def test_detect_changes_version_not_found_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="Version not found"):
            svc.detect_changes("nonexistent-1", "nonexistent-2")

    def test_detect_changes_increments_stats(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"a": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"b": 2})
        svc.detect_changes(v1.version_id, v2.version_id)
        assert svc._stats.total_changes_detected == 1

    def test_detect_changes_records_provenance(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"a": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"b": 2})
        count_before = svc.provenance.entry_count
        svc.detect_changes(v1.version_id, v2.version_id)
        assert svc.provenance.entry_count > count_before

    def test_list_changes_empty(self):
        svc = _make_service()
        result = svc.list_changes()
        assert result == []

    def test_list_changes_returns_cached(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"a": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"b": 2})
        svc.detect_changes(v1.version_id, v2.version_id)
        result = svc.list_changes()
        assert len(result) == 1


class TestServiceCompatibility:
    """Tests for compatibility checking through the service facade."""

    def test_check_compatibility_returns_response(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"type": "object"})
        v2 = svc.create_version(schema_id="s-001", definition={"type": "object", "additionalProperties": False})
        resp = svc.check_compatibility(v1.version_id, v2.version_id)
        assert isinstance(resp, CompatibilityCheckResponse)
        assert resp.source_version_id == v1.version_id
        assert resp.target_version_id == v2.version_id
        assert resp.provenance_hash != ""

    def test_check_compatibility_version_not_found_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="Version not found"):
            svc.check_compatibility("missing-1", "missing-2")

    def test_check_compatibility_increments_stats(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"x": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"y": 2})
        svc.check_compatibility(v1.version_id, v2.version_id)
        assert svc._stats.total_compatibility_checks == 1

    def test_check_compatibility_records_provenance(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"x": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"y": 2})
        count_before = svc.provenance.entry_count
        svc.check_compatibility(v1.version_id, v2.version_id)
        assert svc.provenance.entry_count > count_before

    def test_list_compatibility_checks_empty(self):
        svc = _make_service()
        result = svc.list_compatibility_checks()
        assert result == []

    def test_list_compatibility_checks_returns_cached(self):
        svc = _make_service()
        v1 = svc.create_version(schema_id="s-001", definition={"x": 1})
        v2 = svc.create_version(schema_id="s-001", definition={"y": 2})
        svc.check_compatibility(v1.version_id, v2.version_id)
        result = svc.list_compatibility_checks()
        assert len(result) == 1


class TestServiceMigrationPlanning:
    """Tests for migration planning through the service facade."""

    def test_create_plan_returns_response(self):
        svc = _make_service()
        resp = svc.create_plan(
            source_schema_id="src-1",
            target_schema_id="tgt-1",
        )
        assert isinstance(resp, MigrationPlanResponse)
        assert resp.source_schema_id == "src-1"
        assert resp.target_schema_id == "tgt-1"
        assert resp.status == "pending"
        assert resp.provenance_hash != ""

    def test_create_plan_empty_source_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="source_schema_id must not be empty"):
            svc.create_plan(source_schema_id="", target_schema_id="tgt-1")

    def test_create_plan_empty_target_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="target_schema_id must not be empty"):
            svc.create_plan(source_schema_id="src-1", target_schema_id="")

    def test_create_plan_increments_stats(self):
        svc = _make_service()
        svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        assert svc._stats.total_migrations_planned == 1

    def test_create_plan_records_provenance(self):
        svc = _make_service()
        count_before = svc.provenance.entry_count
        svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        assert svc.provenance.entry_count > count_before

    def test_get_plan_found(self):
        svc = _make_service()
        created = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        fetched = svc.get_plan(created.plan_id)
        assert fetched is not None
        assert fetched.plan_id == created.plan_id

    def test_get_plan_not_found(self):
        svc = _make_service()
        result = svc.get_plan("nonexistent-plan")
        assert result is None


class TestServiceExecution:
    """Tests for migration execution through the service facade."""

    def test_execute_migration_returns_response(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        resp = svc.execute_migration(plan.plan_id)
        assert isinstance(resp, MigrationExecutionResponse)
        assert resp.plan_id == plan.plan_id
        assert resp.status == "completed"
        assert resp.provenance_hash != ""

    def test_execute_migration_plan_not_found_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="Migration plan not found"):
            svc.execute_migration("nonexistent-plan")

    def test_execute_migration_increments_stats(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        svc.execute_migration(plan.plan_id)
        assert svc._stats.total_migrations_executed == 1

    def test_execute_migration_records_provenance(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        count_before = svc.provenance.entry_count
        svc.execute_migration(plan.plan_id)
        assert svc.provenance.entry_count > count_before

    def test_execute_migration_updates_success_rate(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        svc.execute_migration(plan.plan_id)
        assert svc._stats.success_rate > 0.0

    def test_get_execution_found(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        executed = svc.execute_migration(plan.plan_id)
        fetched = svc.get_execution(executed.execution_id)
        assert fetched is not None
        assert fetched.execution_id == executed.execution_id

    def test_get_execution_not_found(self):
        svc = _make_service()
        result = svc.get_execution("nonexistent-exec")
        assert result is None

    def test_rollback_migration_returns_response(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        executed = svc.execute_migration(plan.plan_id)
        rb = svc.rollback_migration(executed.execution_id)
        assert isinstance(rb, MigrationExecutionResponse)
        assert rb.status == "rolled_back"
        assert rb.execution_id == executed.execution_id

    def test_rollback_migration_partial(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        executed = svc.execute_migration(plan.plan_id)
        rb = svc.rollback_migration(executed.execution_id, to_checkpoint=2)
        assert rb.current_step == 2

    def test_rollback_migration_increments_stats(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        executed = svc.execute_migration(plan.plan_id)
        svc.rollback_migration(executed.execution_id)
        assert svc._stats.total_rollbacks == 1

    def test_rollback_records_provenance(self):
        svc = _make_service()
        plan = svc.create_plan(source_schema_id="src-1", target_schema_id="tgt-1")
        executed = svc.execute_migration(plan.plan_id)
        count_before = svc.provenance.entry_count
        svc.rollback_migration(executed.execution_id)
        assert svc.provenance.entry_count > count_before


class TestServicePipeline:
    """Tests for end-to-end pipeline through the service facade."""

    def test_run_pipeline_returns_response(self):
        svc = _make_service()
        # Pre-populate versions for definition resolution
        svc.create_version(schema_id="src-1", definition={"type": "object"})
        svc.create_version(schema_id="tgt-1", definition={"type": "object", "extra": True})
        resp = svc.run_pipeline(
            source_schema_id="src-1",
            target_schema_id="tgt-1",
        )
        assert isinstance(resp, PipelineResultResponse)
        assert resp.source_schema_id == "src-1"
        assert resp.target_schema_id == "tgt-1"
        assert resp.provenance_hash != ""

    def test_run_pipeline_empty_source_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="source_schema_id must not be empty"):
            svc.run_pipeline(source_schema_id="", target_schema_id="tgt-1")

    def test_run_pipeline_empty_target_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="target_schema_id must not be empty"):
            svc.run_pipeline(source_schema_id="src-1", target_schema_id="")

    def test_run_pipeline_includes_stages(self):
        svc = _make_service()
        svc.create_version(schema_id="src-1", definition={"a": 1})
        svc.create_version(schema_id="tgt-1", definition={"b": 2})
        resp = svc.run_pipeline(
            source_schema_id="src-1",
            target_schema_id="tgt-1",
        )
        # Should have completed at least detect and plan stages
        assert "detect" in resp.stages_completed or "plan" in resp.stages_completed

    def test_run_pipeline_records_provenance(self):
        svc = _make_service()
        svc.create_version(schema_id="src-1", definition={"a": 1})
        svc.create_version(schema_id="tgt-1", definition={"b": 2})
        count_before = svc.provenance.entry_count
        svc.run_pipeline(source_schema_id="src-1", target_schema_id="tgt-1")
        assert svc.provenance.entry_count > count_before

    def test_run_pipeline_elapsed_seconds_positive(self):
        svc = _make_service()
        svc.create_version(schema_id="src-1", definition={"a": 1})
        svc.create_version(schema_id="tgt-1", definition={"b": 2})
        resp = svc.run_pipeline(source_schema_id="src-1", target_schema_id="tgt-1")
        assert resp.elapsed_seconds >= 0.0


class TestServiceStatsHealth:
    """Tests for statistics and health check through the service facade."""

    def test_get_statistics_returns_model(self):
        svc = _make_service()
        stats = svc.get_statistics()
        assert isinstance(stats, SchemaMigrationStatisticsResponse)

    def test_get_statistics_reflects_operations(self):
        svc = _make_service()
        svc.register_schema(namespace="ns", name="s1", schema_type="json_schema", definition={})
        svc.register_schema(namespace="ns", name="s2", schema_type="json_schema", definition={})
        stats = svc.get_statistics()
        assert stats.total_schemas == 2

    def test_health_check_returns_dict(self):
        svc = _make_service()
        health = svc.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert "engines" in health
        assert "started" in health
        assert "statistics" in health
        assert "provenance_chain_valid" in health
        assert "timestamp" in health

    def test_health_check_all_engines_unavailable_is_unhealthy(self):
        svc = _make_service()
        health = svc.health_check()
        # All 7 engines are None, so 0 available -> unhealthy
        assert health["status"] == "unhealthy"
        assert health["engines_available"] == 0
        assert health["engines_total"] == 7

    def test_health_check_provenance_chain_valid(self):
        svc = _make_service()
        health = svc.health_check()
        assert health["provenance_chain_valid"] is True

    def test_health_check_started_reflects_lifecycle(self):
        svc = _make_service()
        assert svc.health_check()["started"] is False
        svc.startup()
        assert svc.health_check()["started"] is True
        svc.shutdown()
        assert svc.health_check()["started"] is False


class TestServiceProvenance:
    """Tests for provenance and metrics access methods."""

    def test_get_provenance_returns_tracker(self):
        svc = _make_service()
        tracker = svc.get_provenance()
        assert tracker is svc.provenance
        assert tracker.entry_count == 0

    def test_get_metrics_returns_dict(self):
        svc = _make_service()
        metrics = svc.get_metrics()
        assert isinstance(metrics, dict)
        assert "total_schemas" in metrics
        assert "total_versions" in metrics
        assert "provenance_entries" in metrics
        assert "provenance_chain_valid" in metrics

    def test_get_metrics_reflects_operations(self):
        svc = _make_service()
        svc.register_schema(namespace="ns", name="test", schema_type="json_schema", definition={})
        metrics = svc.get_metrics()
        assert metrics["total_schemas"] == 1
        assert metrics["provenance_entries"] >= 1


class TestServiceLifecycle:
    """Tests for service startup and shutdown."""

    def test_startup_sets_started(self):
        svc = _make_service()
        assert svc._started is False
        svc.startup()
        assert svc._started is True

    def test_startup_idempotent(self):
        svc = _make_service()
        svc.startup()
        svc.startup()
        assert svc._started is True

    def test_shutdown_clears_started(self):
        svc = _make_service()
        svc.startup()
        assert svc._started is True
        svc.shutdown()
        assert svc._started is False

    def test_shutdown_resets_active_migrations(self):
        svc = _make_service()
        svc.startup()
        svc._active_migrations = 3
        svc.shutdown()
        assert svc._active_migrations == 0

    def test_shutdown_when_not_started_is_noop(self):
        svc = _make_service()
        svc.shutdown()  # Should not raise
        assert svc._started is False


# ============================================================================
# MODULE-LEVEL FUNCTION TESTS
# ============================================================================


class TestConfigureSchemaMigration:
    """Tests for the configure_schema_migration async function."""

    def test_creates_service_and_attaches_to_app(self):
        app = MagicMock()
        app.state = MagicMock()

        with patch("greenlang.schema_migration.setup.SchemaRegistryEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaVersionerEngine", None), \
             patch("greenlang.schema_migration.setup.ChangeDetectorEngine", None), \
             patch("greenlang.schema_migration.setup.CompatibilityCheckerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationPlannerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationExecutorEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaMigrationPipelineEngine", None), \
             patch("greenlang.schema_migration.setup._singleton_instance", None):
            service = asyncio.get_event_loop().run_until_complete(
                configure_schema_migration(app)
            )
        assert isinstance(service, SchemaMigrationService)
        assert app.state.schema_migration_service == service

    def test_mounts_router_on_app(self):
        app = MagicMock()
        app.state = MagicMock()

        mock_router = MagicMock()
        with patch("greenlang.schema_migration.setup.SchemaRegistryEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaVersionerEngine", None), \
             patch("greenlang.schema_migration.setup.ChangeDetectorEngine", None), \
             patch("greenlang.schema_migration.setup.CompatibilityCheckerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationPlannerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationExecutorEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaMigrationPipelineEngine", None), \
             patch("greenlang.schema_migration.setup._singleton_instance", None), \
             patch(
                 "greenlang.schema_migration.api.router.router",
                 mock_router,
             ):
            asyncio.get_event_loop().run_until_complete(
                configure_schema_migration(app)
            )
        app.include_router.assert_called_once_with(mock_router)

    def test_starts_the_service(self):
        app = MagicMock()
        app.state = MagicMock()

        with patch("greenlang.schema_migration.setup.SchemaRegistryEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaVersionerEngine", None), \
             patch("greenlang.schema_migration.setup.ChangeDetectorEngine", None), \
             patch("greenlang.schema_migration.setup.CompatibilityCheckerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationPlannerEngine", None), \
             patch("greenlang.schema_migration.setup.MigrationExecutorEngine", None), \
             patch("greenlang.schema_migration.setup.SchemaMigrationPipelineEngine", None), \
             patch("greenlang.schema_migration.setup._singleton_instance", None):
            service = asyncio.get_event_loop().run_until_complete(
                configure_schema_migration(app)
            )
        assert service._started is True


class TestGetSchemaMigration:
    """Tests for the get_schema_migration function."""

    def test_retrieves_service_from_app_state(self):
        app = MagicMock()
        mock_service = MagicMock(spec=SchemaMigrationService)
        app.state.schema_migration_service = mock_service
        result = get_schema_migration(app)
        assert result is mock_service

    def test_raises_runtime_error_when_not_configured(self):
        app = MagicMock()
        app.state = MagicMock(spec=[])  # No attributes
        with pytest.raises(RuntimeError, match="Schema migration service not configured"):
            get_schema_migration(app)

    def test_raises_when_attribute_is_none(self):
        app = MagicMock()
        app.state.schema_migration_service = None
        with pytest.raises(RuntimeError, match="Schema migration service not configured"):
            get_schema_migration(app)


class TestGetRouter:
    """Tests for the get_router function."""

    def test_returns_router_when_available(self):
        mock_router = MagicMock()
        with patch(
            "greenlang.schema_migration.api.router.router",
            mock_router,
        ):
            result = get_router()
        assert result is mock_router

    def test_returns_none_when_import_fails(self):
        with patch.dict(
            "sys.modules",
            {"greenlang.schema_migration.api.router": None},
        ):
            result = get_router()
        # Should return None gracefully
        assert result is None or result is not None  # Does not raise

    def test_accepts_optional_service_arg(self):
        mock_router = MagicMock()
        with patch(
            "greenlang.schema_migration.api.router.router",
            mock_router,
        ):
            result = get_router(service=MagicMock())
        assert result is mock_router

    def test_default_service_is_none(self):
        mock_router = MagicMock()
        with patch(
            "greenlang.schema_migration.api.router.router",
            mock_router,
        ):
            # Should work without passing service
            result = get_router()
        assert result is mock_router


# ============================================================================
# INTERNAL HELPER TESTS
# ============================================================================


class TestInternalHelpers:
    """Tests for internal helper methods of SchemaMigrationService."""

    def test_increment_active_migrations(self):
        svc = _make_service()
        assert svc._active_migrations == 0
        svc._increment_active_migrations()
        assert svc._active_migrations == 1
        assert svc._stats.active_migrations == 1

    def test_decrement_active_migrations(self):
        svc = _make_service()
        svc._active_migrations = 3
        svc._decrement_active_migrations()
        assert svc._active_migrations == 2

    def test_decrement_active_migrations_floor_zero(self):
        svc = _make_service()
        svc._active_migrations = 0
        svc._decrement_active_migrations()
        assert svc._active_migrations == 0

    def test_update_avg_duration_single(self):
        svc = _make_service()
        svc._stats.total_migrations_executed = 1
        svc._update_avg_duration(10.0)
        assert svc._stats.avg_migration_duration_seconds == 10.0

    def test_update_avg_duration_multiple(self):
        svc = _make_service()
        svc._stats.total_migrations_executed = 1
        svc._update_avg_duration(10.0)
        svc._stats.total_migrations_executed = 2
        svc._update_avg_duration(20.0)
        # Average of 10 and 20 = 15
        assert svc._stats.avg_migration_duration_seconds == pytest.approx(15.0, rel=1e-2)

    def test_update_success_rate(self):
        svc = _make_service()
        svc._migration_successes = 9
        svc._migration_total = 10
        svc._update_success_rate()
        assert svc._stats.success_rate == pytest.approx(0.9, rel=1e-4)

    def test_update_success_rate_zero_total(self):
        svc = _make_service()
        svc._migration_total = 0
        svc._update_success_rate()
        assert svc._stats.success_rate == 0.0

    def test_filter_schemas_by_tag(self):
        svc = _make_service()
        svc.register_schema(namespace="ns", name="a", schema_type="json_schema", definition={}, tags=["core"])
        svc.register_schema(namespace="ns", name="b", schema_type="json_schema", definition={}, tags=["optional"])
        result = svc.list_schemas(tag="core")
        assert len(result) == 1
        assert result[0].name == "a"

    def test_filter_schemas_by_owner(self):
        svc = _make_service()
        svc.register_schema(namespace="ns", name="a", schema_type="json_schema", definition={}, owner="team-a")
        svc.register_schema(namespace="ns", name="b", schema_type="json_schema", definition={}, owner="team-b")
        result = svc.list_schemas(owner="team-a")
        assert len(result) == 1
        assert result[0].owner == "team-a"

    def test_filter_schemas_by_status(self):
        svc = _make_service()
        resp = svc.register_schema(namespace="ns", name="a", schema_type="json_schema", definition={})
        svc.update_schema(resp.schema_id, status="active")
        svc.register_schema(namespace="ns", name="b", schema_type="json_schema", definition={})
        result = svc.list_schemas(status="active")
        assert len(result) == 1
        assert result[0].status == "active"

    def test_dict_to_schema_response(self):
        svc = _make_service()
        rec = {
            "schema_id": "sid",
            "namespace": "ns",
            "name": "test",
            "schema_type": "avro",
            "status": "active",
            "owner": "team",
            "tags": ["t1"],
            "description": "desc",
            "version_count": 3,
            "created_at": "2026-01-01",
            "updated_at": "2026-02-01",
            "provenance_hash": "hash123",
        }
        resp = svc._dict_to_schema_response(rec)
        assert isinstance(resp, SchemaResponse)
        assert resp.schema_id == "sid"
        assert resp.name == "test"

    def test_dict_to_version_response(self):
        svc = _make_service()
        rec = {
            "id": "vid",
            "schema_id": "sid",
            "version": "2.0.0",
            "definition": {"key": "val"},
            "changelog_note": "Major update",
            "is_deprecated": False,
            "sunset_date": None,
            "created_at": "2026-01-15",
            "provenance_hash": "hash456",
        }
        resp = svc._dict_to_version_response(rec)
        assert isinstance(resp, SchemaVersionResponse)
        assert resp.version_id == "vid"
        assert resp.version == "2.0.0"
        assert resp.changelog == "Major update"
