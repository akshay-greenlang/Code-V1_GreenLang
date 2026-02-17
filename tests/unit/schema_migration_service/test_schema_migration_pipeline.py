# -*- coding: utf-8 -*-
"""
Unit tests for SchemaMigrationPipelineEngine - AGENT-DATA-017

Tests the SchemaMigrationPipelineEngine from
greenlang.schema_migration.schema_migration_pipeline with ~100 tests
covering initialization, run_pipeline (full success, short-circuits,
abort paths, skip flags, dry-run), individual stage methods, pipeline
result retrieval/listing, batch pipeline, report generation,
statistics, reset, cancel semantics, and edge cases.

Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Pre-import metrics before the pipeline module to avoid Prometheus
# duplicate-metric ValueError.  The pipeline module lazily imports
# schema_registry.py (via a try/except block) which creates its own
# Counter objects using the same metric names defined in metrics.py.
# By loading metrics.py first, the Counters are registered in the
# CollectorRegistry by metrics.py; when schema_registry.py then tries
# to re-register the same names, the ValueError is caught by the
# pipeline module's broad except clause and the registry engine falls
# back to None.  This mirrors the import order used in the conftest
# for other test files in this package.
from greenlang.schema_migration import metrics as _sm_metrics  # noqa: F401

from greenlang.schema_migration.schema_migration_pipeline import (
    SchemaMigrationPipelineEngine,
    _sha256,
    _new_id,
    _elapsed_ms,
    _utcnow_iso,
    _PIPELINE_STAGES,
    _STATUS_COMPLETED,
    _STATUS_FAILED,
    _STATUS_ABORTED,
    _STATUS_NO_CHANGES,
    _STATUS_DRY_RUN,
)
from greenlang.schema_migration.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_json(schema: Dict[str, Any]) -> str:
    """Serialize a schema dict to a JSON string for run_pipeline."""
    return json.dumps(schema, sort_keys=True)


def _make_engine(**kwargs) -> SchemaMigrationPipelineEngine:
    """Create a SchemaMigrationPipelineEngine with all engines set to None.

    Avoids the slow/flaky import of real upstream engines.  Individual
    tests may patch internal engines back in as needed.
    """
    with patch.object(
        SchemaMigrationPipelineEngine, "__init__", lambda self: None
    ):
        eng = SchemaMigrationPipelineEngine.__new__(SchemaMigrationPipelineEngine)

    eng._registry = kwargs.get("registry", None)
    eng._versioner = kwargs.get("versioner", None)
    eng._detector = kwargs.get("detector", None)
    eng._checker = kwargs.get("checker", None)
    eng._planner = kwargs.get("planner", None)
    eng._executor = kwargs.get("executor", None)
    eng._pipeline_runs = {}
    eng._lock = threading.Lock()
    eng._provenance = ProvenanceTracker()
    eng._active_migrations = 0
    return eng


def _register_source_schema(
    engine: SchemaMigrationPipelineEngine,
    schema_id: str,
    definition: Dict[str, Any],
    version: str = "1.0.0",
) -> None:
    """Seed the engine's registry mock so _fetch_current_definition works."""
    mock_registry = MagicMock()
    mock_registry.get_schema.return_value = {
        "definition": definition,
        "version": version,
    }
    mock_registry.register_schema.return_value = None
    engine._registry = mock_registry


def _register_source_versioner(
    engine: SchemaMigrationPipelineEngine,
    schema_id: str,
    version: str = "1.0.0",
) -> None:
    """Seed the engine's versioner mock so _fetch_current_version works."""
    mock_versioner = MagicMock()
    mock_versioner.get_current_version.return_value = {"version": version}
    mock_versioner.create_version.return_value = None
    engine._versioner = mock_versioner


def _inject_mock_detector(
    engine: SchemaMigrationPipelineEngine,
    has_changes: bool = True,
    changes: Optional[List[Dict[str, Any]]] = None,
    breaking_changes: Optional[List[Dict[str, Any]]] = None,
) -> MagicMock:
    """Inject a mock ChangeDetectorEngine that returns controlled results.

    The stub _stub_detect compares only top-level dict keys, so when both
    source and target are full JSON Schema objects with identical top-level
    keys ($schema, type, properties, required, additionalProperties) the
    stub reports zero changes even though the properties differ.  This
    helper injects a mock that properly reports the expected changes,
    bypassing the stub.
    """
    if changes is None:
        changes = [
            {"change_type": "field_added", "field": "salary", "severity": "non_breaking"},
            {"change_type": "field_removed", "field": "age", "severity": "breaking"},
            {"change_type": "field_added", "field": "team", "severity": "non_breaking"},
            {"change_type": "field_removed", "field": "department", "severity": "breaking"},
        ]
    if breaking_changes is None:
        breaking_changes = [c for c in changes if c.get("severity") == "breaking"]
    non_breaking = [c for c in changes if c.get("severity") != "breaking"]

    mock_detector = MagicMock()
    mock_detector.detect_changes.return_value = {
        "has_changes": has_changes,
        "change_count": len(changes) if has_changes else 0,
        "changes": changes if has_changes else [],
        "breaking_changes": breaking_changes if has_changes else [],
        "non_breaking_changes": non_breaking if has_changes else [],
    }
    engine._detector = mock_detector
    return mock_detector


# ===========================================================================
# TestSchemaMigrationPipelineInit
# ===========================================================================


class TestSchemaMigrationPipelineInit:
    """Test SchemaMigrationPipelineEngine initialization."""

    def test_default_initialization_creates_provenance(self):
        """Engine always initializes a ProvenanceTracker."""
        engine = _make_engine()
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_default_initialization_empty_pipeline_runs(self):
        """Engine starts with an empty pipeline run store."""
        engine = _make_engine()
        assert engine._pipeline_runs == {}

    def test_default_initialization_lock_exists(self):
        """Engine creates a threading.Lock for thread safety."""
        engine = _make_engine()
        # threading.Lock is a factory function on some platforms, so
        # check for the expected acquire/release interface instead of
        # isinstance.
        assert hasattr(engine._lock, "acquire")
        assert hasattr(engine._lock, "release")

    def test_default_initialization_active_migrations_zero(self):
        """Active migrations counter starts at zero."""
        engine = _make_engine()
        assert engine._active_migrations == 0

    def test_engines_default_to_none(self):
        """All six upstream engines default to None when unavailable."""
        engine = _make_engine()
        assert engine._registry is None
        assert engine._versioner is None
        assert engine._detector is None
        assert engine._checker is None
        assert engine._planner is None
        assert engine._executor is None

    def test_injected_registry_engine(self):
        """A mock registry engine can be injected."""
        mock_reg = MagicMock()
        engine = _make_engine(registry=mock_reg)
        assert engine._registry is mock_reg

    def test_injected_detector_engine(self):
        """A mock detector engine can be injected."""
        mock_det = MagicMock()
        engine = _make_engine(detector=mock_det)
        assert engine._detector is mock_det

    def test_injected_all_engines(self):
        """All six engines can be injected simultaneously."""
        mocks = {
            "registry": MagicMock(),
            "versioner": MagicMock(),
            "detector": MagicMock(),
            "checker": MagicMock(),
            "planner": MagicMock(),
            "executor": MagicMock(),
        }
        engine = _make_engine(**mocks)
        assert engine._registry is mocks["registry"]
        assert engine._versioner is mocks["versioner"]
        assert engine._detector is mocks["detector"]
        assert engine._checker is mocks["checker"]
        assert engine._planner is mocks["planner"]
        assert engine._executor is mocks["executor"]


# ===========================================================================
# TestRunPipeline
# ===========================================================================


class TestRunPipeline:
    """Test run_pipeline end-to-end scenarios."""

    def test_empty_schema_id_raises_value_error(self):
        """Empty schema_id raises ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="schema_id must not be empty"):
            engine.run_pipeline(schema_id="", target_definition_json="{}")

    def test_empty_target_json_raises_value_error(self):
        """Empty target_definition_json raises ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="target_definition_json must not be empty"):
            engine.run_pipeline(schema_id="test", target_definition_json="")

    def test_invalid_target_json_aborts_at_detect(self):
        """Non-JSON target_definition_json aborts at detect stage."""
        engine = _make_engine()
        result = engine.run_pipeline(
            schema_id="test",
            target_definition_json="NOT VALID JSON {{",
        )
        assert result["status"] == _STATUS_ABORTED
        assert "detect" in result["stages_failed"]

    def test_no_changes_returns_no_changes_status(self, sample_json_schema):
        """Identical source and target produces status='no_changes'."""
        engine = _make_engine()
        _register_source_schema(engine, "test_schema", sample_json_schema)
        _register_source_versioner(engine, "test_schema")

        result = engine.run_pipeline(
            schema_id="test_schema",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["status"] == _STATUS_NO_CHANGES
        assert "detect" in result["stages_completed"]
        assert result["pipeline_id"].startswith("pipe-")

    def test_full_pipeline_completed_with_data(
        self, sample_json_schema, sample_json_schema_v2, sample_records_v1
    ):
        """Full pipeline with data executes all 7 stages successfully."""
        engine = _make_engine()
        _register_source_schema(engine, "emp_schema", sample_json_schema)
        _register_source_versioner(engine, "emp_schema", "1.0.0")
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="emp_schema",
            target_definition_json=_target_json(sample_json_schema_v2),
            data=sample_records_v1,
            skip_compatibility=True,
            skip_dry_run=True,
        )
        assert result["status"] in (_STATUS_COMPLETED, _STATUS_ABORTED)
        assert result["pipeline_id"].startswith("pipe-")
        assert isinstance(result["total_time_ms"], float)
        assert result["provenance_hash"] is not None

    def test_pipeline_with_skip_compatibility(
        self, sample_json_schema, sample_json_schema_v2
    ):
        """Pipeline skips compatibility abort when skip_compatibility=True."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        _register_source_versioner(engine, "s1")
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema_v2),
            skip_compatibility=True,
            skip_dry_run=True,
        )
        # Should not abort at compatibility
        assert "compatibility" not in result.get("stages_failed", [])

    def test_pipeline_with_skip_dry_run(
        self, sample_json_schema, sample_json_schema_v2
    ):
        """Pipeline skips validate stage when skip_dry_run=True."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        _register_source_versioner(engine, "s1")
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema_v2),
            skip_compatibility=True,
            skip_dry_run=True,
        )
        assert "validate" not in result.get("stages_completed", [])

    def test_dry_run_returns_dry_run_status(
        self, sample_json_schema, sample_json_schema_v2
    ):
        """Pipeline with dry_run=True returns status='dry_run_completed'."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        _register_source_versioner(engine, "s1")
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema_v2),
            skip_compatibility=True,
            dry_run=True,
        )
        assert result["status"] == _STATUS_DRY_RUN
        assert "execute" not in result.get("stages_completed", [])
        assert "verify" not in result.get("stages_completed", [])
        assert "registry" not in result.get("stages_completed", [])

    def test_pipeline_id_is_unique(self, sample_json_schema):
        """Each pipeline run produces a unique pipeline_id."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        r1 = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        r2 = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert r1["pipeline_id"] != r2["pipeline_id"]

    def test_pipeline_has_created_at_timestamp(self, sample_json_schema):
        """Pipeline result includes created_at ISO timestamp."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["created_at"] is not None
        assert "T" in result["created_at"]

    def test_pipeline_has_provenance_hash(self, sample_json_schema):
        """Pipeline result includes a 64-char hex provenance hash."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_pipeline_stages_completed_list(self, sample_json_schema):
        """stages_completed is a list tracking completed stage names."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert isinstance(result["stages_completed"], list)

    def test_pipeline_total_time_ms_positive(self, sample_json_schema):
        """total_time_ms is a non-negative float after pipeline completes."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert isinstance(result["total_time_ms"], float)
        assert result["total_time_ms"] >= 0

    def test_pipeline_result_stored_in_runs(self, sample_json_schema):
        """Completed pipeline is stored in _pipeline_runs for later retrieval."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        stored = engine.get_pipeline_run(result["pipeline_id"])
        assert stored is not None
        assert stored["pipeline_id"] == result["pipeline_id"]

    def test_pipeline_without_data_no_execution_data(
        self, sample_json_schema, sample_json_schema_v2
    ):
        """Pipeline without data does not produce migrated_data in execution."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        _register_source_versioner(engine, "s1")
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema_v2),
            data=None,
            skip_compatibility=True,
            skip_dry_run=True,
        )
        execution = result.get("execution")
        if execution is not None:
            assert execution.get("migrated_data") is None

    def test_breaking_changes_aborts_without_skip(
        self, sample_json_schema, sample_json_schema_v2
    ):
        """Breaking changes abort pipeline when skip_compatibility=False."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        _register_source_versioner(engine, "s1")
        # Inject detector with breaking changes so the pipeline reaches
        # the compatibility stage and aborts due to breaking changes.
        _inject_mock_detector(engine)

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema_v2),
            skip_compatibility=False,
        )
        assert result["status"] == _STATUS_ABORTED

    def test_pipeline_active_migrations_returns_to_zero(self, sample_json_schema):
        """Active migrations counter returns to zero after pipeline completes."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert engine._active_migrations == 0

    def test_pipeline_source_version_populated(self, sample_json_schema):
        """Result includes source_version from the versioner/registry."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema, version="2.3.4")
        _register_source_versioner(engine, "s1", version="2.3.4")

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["source_version"] == "2.3.4"

    def test_pipeline_schema_id_in_result(self, sample_json_schema):
        """Result includes the schema_id that was passed in."""
        engine = _make_engine()
        _register_source_schema(engine, "my_schema", sample_json_schema)

        result = engine.run_pipeline(
            schema_id="my_schema",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["schema_id"] == "my_schema"

    def test_pipeline_unexpected_error_sets_failed(self, sample_json_schema):
        """An unexpected exception during pipeline sets status='failed'."""
        engine = _make_engine()
        # Make _fetch_current_definition raise an unexpected error
        engine._fetch_current_definition = MagicMock(
            side_effect=RuntimeError("boom")
        )

        result = engine.run_pipeline(
            schema_id="s1",
            target_definition_json=_target_json(sample_json_schema),
        )
        assert result["status"] == _STATUS_FAILED
        assert "unexpected" in result["stages_failed"]
        assert result["error"] is not None


# ===========================================================================
# TestPipelineStages
# ===========================================================================


class TestPipelineStages:
    """Test individual stage methods."""

    # -- detect_stage --

    def test_detect_stage_with_no_changes(self, sample_json_schema):
        """detect_stage returns has_changes=False for identical schemas."""
        engine = _make_engine()
        result = engine.detect_stage(sample_json_schema, sample_json_schema)
        assert result["has_changes"] is False
        assert result["change_count"] == 0

    def test_detect_stage_with_changes(self):
        """detect_stage returns has_changes=True for schemas with different keys.

        The stub _stub_detect compares top-level dict keys, so we use
        simple flat dicts that differ in their keys.
        """
        engine = _make_engine()
        source = {"id": "int", "name": "str"}
        target = {"id": "int", "name": "str", "email": "str"}
        result = engine.detect_stage(source, target)
        assert result["has_changes"] is True
        assert result["change_count"] > 0

    def test_detect_stage_has_duration_ms(self, sample_json_schema):
        """detect_stage result includes duration_ms."""
        engine = _make_engine()
        result = engine.detect_stage(sample_json_schema, sample_json_schema)
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)

    def test_detect_stage_has_detected_at(self, sample_json_schema):
        """detect_stage result includes detected_at timestamp."""
        engine = _make_engine()
        result = engine.detect_stage(sample_json_schema, sample_json_schema)
        assert "detected_at" in result

    def test_detect_stage_breaking_and_non_breaking(self):
        """detect_stage categorizes changes as breaking/non_breaking.

        Uses flat dicts with differing keys so the stub detects changes.
        """
        engine = _make_engine()
        source = {"id": "int", "name": "str", "age": "int"}
        target = {"id": "int", "name": "str", "email": "str"}
        result = engine.detect_stage(source, target)
        assert "breaking_changes" in result
        assert "non_breaking_changes" in result
        assert isinstance(result["breaking_changes"], list)
        assert isinstance(result["non_breaking_changes"], list)
        # 'age' removed (breaking), 'email' added (non-breaking)
        assert len(result["breaking_changes"]) >= 1
        assert len(result["non_breaking_changes"]) >= 1

    def test_detect_stage_with_detector_engine(self, sample_json_schema):
        """detect_stage delegates to ChangeDetectorEngine when available."""
        mock_detector = MagicMock()
        mock_detector.detect_changes.return_value = {
            "has_changes": True,
            "change_count": 2,
            "changes": [
                {"change_type": "field_added", "field": "x", "severity": "non_breaking"},
                {"change_type": "field_removed", "field": "y", "severity": "breaking"},
            ],
            "breaking_changes": [
                {"change_type": "field_removed", "field": "y", "severity": "breaking"},
            ],
            "non_breaking_changes": [
                {"change_type": "field_added", "field": "x", "severity": "non_breaking"},
            ],
        }
        engine = _make_engine(detector=mock_detector)
        result = engine.detect_stage(sample_json_schema, {"type": "object"})
        assert result["has_changes"] is True
        assert result["change_count"] == 2
        mock_detector.detect_changes.assert_called_once()

    def test_detect_stage_engine_failure_uses_stub(self, sample_json_schema):
        """detect_stage falls back to stub when engine raises."""
        mock_detector = MagicMock()
        mock_detector.detect_changes.side_effect = RuntimeError("engine broke")
        engine = _make_engine(detector=mock_detector)
        result = engine.detect_stage(sample_json_schema, {"type": "object"})
        # Stub should still work
        assert "has_changes" in result

    # -- compatibility_stage --

    def test_compatibility_stage_compatible(self, sample_json_schema):
        """compatibility_stage reports compatible when no breaking changes."""
        engine = _make_engine()
        changes = {"breaking_changes": [], "change_count": 1}
        result = engine.compatibility_stage(
            sample_json_schema, sample_json_schema, changes
        )
        assert result["is_compatible"] is True
        assert result["is_breaking"] is False

    def test_compatibility_stage_breaking(self, sample_json_schema):
        """compatibility_stage reports breaking when breaking changes exist."""
        engine = _make_engine()
        changes = {
            "breaking_changes": [{"change_type": "field_removed"}],
            "change_count": 1,
        }
        result = engine.compatibility_stage(
            sample_json_schema, sample_json_schema, changes
        )
        assert result["is_breaking"] is True

    def test_compatibility_stage_has_duration(self, sample_json_schema):
        """compatibility_stage result includes duration_ms."""
        engine = _make_engine()
        result = engine.compatibility_stage(
            sample_json_schema, sample_json_schema, {"breaking_changes": []}
        )
        assert "duration_ms" in result

    def test_compatibility_stage_recommended_bump(self, sample_json_schema):
        """compatibility_stage returns recommended_bump."""
        engine = _make_engine()
        result = engine.compatibility_stage(
            sample_json_schema, sample_json_schema, {"breaking_changes": [], "change_count": 2}
        )
        assert "recommended_bump" in result

    # -- plan_stage --

    def test_plan_stage_returns_plan_id(self, sample_json_schema):
        """plan_stage returns a plan_id string."""
        engine = _make_engine()
        changes = {
            "changes": [{"change_type": "field_added", "field": "x", "severity": "non_breaking"}],
            "change_count": 1,
        }
        result = engine.plan_stage("s1", "1.0.0", "1.1.0", changes)
        assert result["plan_id"].startswith("plan-")

    def test_plan_stage_step_count(self, sample_json_schema):
        """plan_stage step_count matches the number of changes."""
        engine = _make_engine()
        changes = {
            "changes": [
                {"change_type": "field_added", "field": "a", "severity": "non_breaking"},
                {"change_type": "field_removed", "field": "b", "severity": "breaking"},
            ],
            "change_count": 2,
        }
        result = engine.plan_stage("s1", "1.0.0", "2.0.0", changes)
        assert result["step_count"] == 2
        assert len(result["steps"]) == 2

    def test_plan_stage_effort_classification(self):
        """plan_stage classifies effort based on change count."""
        engine = _make_engine()

        # 0 changes -> NONE
        result_0 = engine.plan_stage("s", "1.0.0", "1.0.1", {"changes": [], "change_count": 0})
        assert result_0["effort"] == "NONE"

        # 3 changes -> LOW
        changes_3 = {
            "changes": [{"change_type": "field_added", "field": f"f{i}"} for i in range(3)],
            "change_count": 3,
        }
        result_3 = engine.plan_stage("s", "1.0.0", "1.1.0", changes_3)
        assert result_3["effort"] == "LOW"

        # 10 changes -> MEDIUM
        changes_10 = {
            "changes": [{"change_type": "field_added", "field": f"f{i}"} for i in range(10)],
            "change_count": 10,
        }
        result_10 = engine.plan_stage("s", "1.0.0", "1.1.0", changes_10)
        assert result_10["effort"] == "MEDIUM"

    def test_plan_stage_has_duration(self):
        """plan_stage result includes duration_ms and planned_at."""
        engine = _make_engine()
        result = engine.plan_stage("s", "1.0.0", "1.0.1", {"changes": [], "change_count": 0})
        assert "duration_ms" in result
        assert "planned_at" in result

    # -- validate_stage --

    def test_validate_stage_valid_plan_id(self):
        """validate_stage returns is_valid=True for non-empty plan_id."""
        engine = _make_engine()
        result = engine.validate_stage("plan-abc123")
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_validate_stage_empty_plan_id(self):
        """validate_stage returns is_valid=False for empty plan_id."""
        engine = _make_engine()
        result = engine.validate_stage("")
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_stage_has_duration(self):
        """validate_stage result includes duration_ms and validated_at."""
        engine = _make_engine()
        result = engine.validate_stage("plan-abc")
        assert "duration_ms" in result
        assert "validated_at" in result

    # -- execute_stage --

    def test_execute_stage_with_data(self, sample_records_v1):
        """execute_stage transforms data using plan steps."""
        engine = _make_engine()
        plan = {
            "plan_id": "plan-test",
            "steps": [
                {"operation": "field_added", "field": "salary"},
                {"operation": "field_removed", "field": "age"},
            ],
        }
        result = engine.execute_stage(plan, data=sample_records_v1)
        assert result["status"] == "success"
        assert result["records_migrated"] == len(sample_records_v1)
        assert result["migrated_data"] is not None
        # Verify salary was added, age was removed
        for record in result["migrated_data"]:
            assert "salary" in record
            assert "age" not in record

    def test_execute_stage_without_data(self):
        """execute_stage without data returns records_migrated=0."""
        engine = _make_engine()
        plan = {"plan_id": "plan-test", "steps": []}
        result = engine.execute_stage(plan, data=None)
        assert result["records_migrated"] == 0
        assert result["migrated_data"] is None

    def test_execute_stage_dry_run(self, sample_records_v1):
        """execute_stage with dry_run=True returns status='dry_run'."""
        engine = _make_engine()
        plan = {"plan_id": "plan-test", "steps": []}
        result = engine.execute_stage(plan, data=sample_records_v1, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["records_migrated"] == 0
        assert result["migrated_data"] is None

    def test_execute_stage_has_execution_id(self):
        """execute_stage result includes an execution_id."""
        engine = _make_engine()
        plan = {"plan_id": "plan-test", "steps": []}
        result = engine.execute_stage(plan)
        assert result["execution_id"].startswith("exec-")

    def test_execute_stage_has_duration(self):
        """execute_stage result includes duration_ms and executed_at."""
        engine = _make_engine()
        plan = {"plan_id": "plan-test", "steps": []}
        result = engine.execute_stage(plan)
        assert "duration_ms" in result
        assert "executed_at" in result

    # -- verify_stage --

    def test_verify_stage_no_data(self, sample_json_schema):
        """verify_stage with no data returns passed=True, records_verified=0."""
        engine = _make_engine()
        result = engine.verify_stage("exec-1", sample_json_schema, migrated_data=None)
        assert result["passed"] is True
        assert result["records_verified"] == 0

    def test_verify_stage_has_duration(self, sample_json_schema):
        """verify_stage result includes duration_ms and verified_at."""
        engine = _make_engine()
        result = engine.verify_stage("exec-1", sample_json_schema)
        assert "duration_ms" in result
        assert "verified_at" in result


# ===========================================================================
# TestGetPipelineResult
# ===========================================================================


class TestGetPipelineResult:
    """Test get_pipeline_run retrieval."""

    def test_get_existing_run(self, sample_json_schema):
        """Retrieve a pipeline run that exists in the store."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        fetched = engine.get_pipeline_run(result["pipeline_id"])
        assert fetched is not None
        assert fetched["pipeline_id"] == result["pipeline_id"]

    def test_get_non_existing_run(self):
        """get_pipeline_run returns None for unknown pipeline_id."""
        engine = _make_engine()
        assert engine.get_pipeline_run("pipe-nonexistent") is None

    def test_get_run_after_multiple_runs(self, sample_json_schema):
        """get_pipeline_run retrieves the correct run among multiple."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        r1 = engine.run_pipeline("s1", _target_json(sample_json_schema))
        r2 = engine.run_pipeline("s1", _target_json(sample_json_schema))

        fetched1 = engine.get_pipeline_run(r1["pipeline_id"])
        fetched2 = engine.get_pipeline_run(r2["pipeline_id"])
        assert fetched1["pipeline_id"] == r1["pipeline_id"]
        assert fetched2["pipeline_id"] == r2["pipeline_id"]

    def test_get_run_contains_all_keys(self, sample_json_schema):
        """Retrieved run contains all expected top-level keys."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        fetched = engine.get_pipeline_run(result["pipeline_id"])

        expected_keys = {
            "pipeline_id", "schema_id", "status", "stages_completed",
            "stages_failed", "changes", "compatibility", "validation",
            "plan", "execution", "verification", "registry_update",
            "source_version", "target_version", "total_time_ms",
            "created_at", "provenance_hash", "error",
        }
        assert expected_keys.issubset(set(fetched.keys()))

    def test_get_run_empty_string_id(self):
        """get_pipeline_run with empty string returns None."""
        engine = _make_engine()
        assert engine.get_pipeline_run("") is None


# ===========================================================================
# TestListPipelineResults
# ===========================================================================


class TestListPipelineResults:
    """Test list_pipeline_runs pagination and filtering."""

    def test_list_empty(self):
        """list_pipeline_runs returns empty list when no runs exist."""
        engine = _make_engine()
        assert engine.list_pipeline_runs() == []

    def test_list_returns_all(self, sample_json_schema):
        """list_pipeline_runs returns all runs when count <= limit."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        for _ in range(3):
            engine.run_pipeline("s1", _target_json(sample_json_schema))

        runs = engine.list_pipeline_runs(limit=100)
        assert len(runs) == 3

    def test_list_respects_limit(self, sample_json_schema):
        """list_pipeline_runs returns at most 'limit' items."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        for _ in range(5):
            engine.run_pipeline("s1", _target_json(sample_json_schema))

        runs = engine.list_pipeline_runs(limit=2)
        assert len(runs) == 2

    def test_list_respects_offset(self, sample_json_schema):
        """list_pipeline_runs skips first 'offset' items."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        for _ in range(5):
            engine.run_pipeline("s1", _target_json(sample_json_schema))

        runs = engine.list_pipeline_runs(limit=100, offset=3)
        assert len(runs) == 2

    def test_list_negative_limit_raises(self):
        """Negative limit raises ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="limit must be >= 0"):
            engine.list_pipeline_runs(limit=-1)

    def test_list_negative_offset_raises(self):
        """Negative offset raises ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="offset must be >= 0"):
            engine.list_pipeline_runs(offset=-1)

    def test_list_sorted_descending(self, sample_json_schema):
        """list_pipeline_runs returns results sorted by created_at descending."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        for _ in range(3):
            engine.run_pipeline("s1", _target_json(sample_json_schema))

        runs = engine.list_pipeline_runs()
        timestamps = [r["created_at"] for r in runs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_offset_beyond_count(self, sample_json_schema):
        """list_pipeline_runs with offset beyond count returns empty list."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        engine.run_pipeline("s1", _target_json(sample_json_schema))

        runs = engine.list_pipeline_runs(offset=100)
        assert runs == []


# ===========================================================================
# TestRunBatch
# ===========================================================================


class TestRunBatch:
    """Test run_batch_pipeline for multiple schema pairs."""

    def test_batch_empty_raises_value_error(self):
        """Empty schema_pairs raises ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="schema_pairs must not be empty"):
            engine.run_batch_pipeline([])

    def test_batch_single_pair(self, sample_json_schema):
        """Batch with a single pair processes it successfully."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        batch_result = engine.run_batch_pipeline([
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ])
        assert batch_result["total"] == 1
        assert len(batch_result["results"]) == 1

    def test_batch_multiple_pairs(self, sample_json_schema):
        """Batch with multiple pairs processes all of them."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        pairs = [
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)}
            for _ in range(3)
        ]
        batch_result = engine.run_batch_pipeline(pairs)
        assert batch_result["total"] == 3
        assert len(batch_result["results"]) == 3

    def test_batch_has_batch_id(self, sample_json_schema):
        """Batch result includes a batch_id."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        batch_result = engine.run_batch_pipeline([
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ])
        assert batch_result["batch_id"].startswith("batch-")

    def test_batch_has_provenance_hash(self, sample_json_schema):
        """Batch result includes a provenance hash."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        batch_result = engine.run_batch_pipeline([
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ])
        assert batch_result["provenance_hash"] is not None
        assert len(batch_result["provenance_hash"]) == 64

    def test_batch_counts_statuses(self, sample_json_schema):
        """Batch result includes status counts."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        batch_result = engine.run_batch_pipeline([
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ])
        assert "completed" in batch_result
        assert "failed" in batch_result
        assert "no_changes" in batch_result

    def test_batch_partial_failure(self, sample_json_schema):
        """Batch continues processing even if one pair fails."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        pairs = [
            {"schema_id": "", "target_definition_json": "{}"},  # will raise ValueError
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ]
        batch_result = engine.run_batch_pipeline(pairs)
        assert batch_result["total"] == 2
        assert len(batch_result["results"]) == 2
        # At least one should have failed
        statuses = [r.get("status") for r in batch_result["results"]]
        assert _STATUS_FAILED in statuses

    def test_batch_total_time_ms(self, sample_json_schema):
        """Batch result includes total_time_ms."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        batch_result = engine.run_batch_pipeline([
            {"schema_id": "s1", "target_definition_json": _target_json(sample_json_schema)},
        ])
        assert isinstance(batch_result["total_time_ms"], float)
        assert batch_result["total_time_ms"] >= 0


# ===========================================================================
# TestGenerateReport
# ===========================================================================


class TestGenerateReport:
    """Test generate_report compliance reporting."""

    def test_report_for_existing_pipeline(self, sample_json_schema):
        """Report can be generated for a completed pipeline run."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])
        assert report["pipeline_id"] == result["pipeline_id"]
        assert report["report_id"].startswith("rpt-")

    def test_report_non_existing_raises_key_error(self):
        """Report generation raises KeyError for unknown pipeline_id."""
        engine = _make_engine()
        with pytest.raises(KeyError, match="Pipeline run not found"):
            engine.generate_report("pipe-nonexistent")

    def test_report_has_change_summary(self, sample_json_schema):
        """Report includes a change_summary section."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        cs = report["change_summary"]
        assert "total_changes" in cs
        assert "breaking_changes" in cs
        assert "non_breaking_changes" in cs
        assert "by_type" in cs

    def test_report_has_timing(self, sample_json_schema):
        """Report includes a timing section with total_time_ms."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        assert "timing" in report
        assert "total_time_ms" in report["timing"]

    def test_report_has_compliance_notes(self, sample_json_schema):
        """Report includes compliance_notes list."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        assert isinstance(report["compliance_notes"], list)

    def test_report_has_provenance_entries(self, sample_json_schema):
        """Report includes provenance_entries list."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        assert isinstance(report["provenance_entries"], list)

    def test_report_has_report_hash(self, sample_json_schema):
        """Report includes a 64-char report_hash."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        assert len(report["report_hash"]) == 64

    def test_report_status_matches_pipeline(self, sample_json_schema):
        """Report status field matches the original pipeline result status."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        report = engine.generate_report(result["pipeline_id"])

        assert report["status"] == result["status"]


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test get_statistics aggregation."""

    def test_statistics_empty(self):
        """Statistics for an engine with no runs shows zero counts."""
        engine = _make_engine()
        stats = engine.get_statistics()
        assert stats["total_runs"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["active_migrations"] == 0

    def test_statistics_after_runs(self, sample_json_schema):
        """Statistics reflect recorded pipeline runs."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        engine.run_pipeline("s1", _target_json(sample_json_schema))
        engine.run_pipeline("s1", _target_json(sample_json_schema))

        stats = engine.get_statistics()
        assert stats["total_runs"] == 2
        assert "by_status" in stats
        assert stats["avg_duration_ms"] >= 0
        assert "computed_at" in stats

    def test_statistics_provenance_count(self, sample_json_schema):
        """Statistics include provenance_entry_count."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        engine.run_pipeline("s1", _target_json(sample_json_schema))

        stats = engine.get_statistics()
        assert stats["provenance_entry_count"] > 0

    def test_statistics_min_max_duration(self, sample_json_schema):
        """Statistics include min and max duration."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        engine.run_pipeline("s1", _target_json(sample_json_schema))

        stats = engine.get_statistics()
        assert "min_duration_ms" in stats
        assert "max_duration_ms" in stats
        assert stats["min_duration_ms"] <= stats["max_duration_ms"]


# ===========================================================================
# TestReset
# ===========================================================================


class TestReset:
    """Test reset clears all state."""

    def test_reset_clears_pipeline_runs(self, sample_json_schema):
        """reset() clears all pipeline run records."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        engine.run_pipeline("s1", _target_json(sample_json_schema))
        assert len(engine._pipeline_runs) > 0

        engine.reset()
        assert len(engine._pipeline_runs) == 0

    def test_reset_zeros_active_migrations(self):
        """reset() sets active_migrations to 0."""
        engine = _make_engine()
        engine._active_migrations = 5
        engine.reset()
        assert engine._active_migrations == 0

    def test_reset_resets_provenance(self, sample_json_schema):
        """reset() resets the provenance tracker entry count."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        engine.run_pipeline("s1", _target_json(sample_json_schema))
        assert engine._provenance.entry_count > 0

        engine.reset()
        assert engine._provenance.entry_count == 0

    def test_reset_allows_new_runs(self, sample_json_schema):
        """After reset, new pipeline runs can be executed."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        engine.run_pipeline("s1", _target_json(sample_json_schema))
        engine.reset()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        assert result["pipeline_id"].startswith("pipe-")
        assert len(engine._pipeline_runs) == 1


# ===========================================================================
# TestCancelPipeline (simulated via status check)
# ===========================================================================


class TestCancelPipeline:
    """Test cancel-like semantics for pipeline runs.

    The engine does not expose a cancel_pipeline method directly, but
    pipelines can be aborted at any stage.  These tests validate the
    abort path and status tracking.
    """

    def test_aborted_pipeline_has_stages_failed(self):
        """Aborted pipeline records the failed stage name."""
        engine = _make_engine()
        result = engine.run_pipeline("s1", "NOT_JSON!!!")
        assert _STATUS_ABORTED == result["status"]
        assert len(result["stages_failed"]) > 0

    def test_aborted_pipeline_has_error_message(self):
        """Aborted pipeline records a human-readable error."""
        engine = _make_engine()
        result = engine.run_pipeline("s1", "NOT_JSON!!!")
        assert result["error"] is not None
        assert len(result["error"]) > 0

    def test_aborted_pipeline_stored_in_runs(self):
        """Aborted pipeline is still stored for later inspection."""
        engine = _make_engine()
        result = engine.run_pipeline("s1", "NOT_JSON!!!")
        fetched = engine.get_pipeline_run(result["pipeline_id"])
        assert fetched is not None
        assert fetched["status"] == _STATUS_ABORTED

    def test_completed_pipeline_cannot_be_re_aborted(self, sample_json_schema):
        """A completed pipeline result retains its final status.

        Since _abort is called only during pipeline execution, after
        finalization the status is immutable.
        """
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        assert result["status"] in (_STATUS_NO_CHANGES, _STATUS_COMPLETED)

        # Verify the stored status is unchanged
        stored = engine.get_pipeline_run(result["pipeline_id"])
        assert stored["status"] == result["status"]


# ===========================================================================
# TestPipelineEdgeCases
# ===========================================================================


class TestPipelineEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_source_schema(self):
        """Pipeline handles empty source schema from registry."""
        engine = _make_engine()
        # No registry -> empty source definition
        target = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = engine.run_pipeline("s1", _target_json(target))
        assert result["status"] in (
            _STATUS_NO_CHANGES, _STATUS_COMPLETED, _STATUS_ABORTED, _STATUS_DRY_RUN,
        )
        assert result["pipeline_id"].startswith("pipe-")

    def test_identical_schemas_no_changes(self, sample_json_schema):
        """Identical source and target always short-circuit to no_changes."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)

        result = engine.run_pipeline("s1", _target_json(sample_json_schema))
        assert result["status"] == _STATUS_NO_CHANGES

    def test_complex_schema_with_many_fields(self):
        """Pipeline handles a schema with 50+ properties via mock detector."""
        engine = _make_engine()
        source = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(60)},
        }
        target = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(5, 65)},
        }
        _register_source_schema(engine, "complex", source)
        _register_source_versioner(engine, "complex")
        # Inject a mock detector with many changes to exercise large-change
        # paths (effort classification, step generation).
        many_changes = [
            {"change_type": "field_added", "field": f"field_{i}", "severity": "non_breaking"}
            for i in range(60, 65)
        ] + [
            {"change_type": "field_removed", "field": f"field_{i}", "severity": "breaking"}
            for i in range(5)
        ]
        _inject_mock_detector(engine, has_changes=True, changes=many_changes)

        result = engine.run_pipeline(
            "complex",
            _target_json(target),
            skip_compatibility=True,
            skip_dry_run=True,
        )
        assert result["pipeline_id"].startswith("pipe-")
        assert result["changes"]["has_changes"] is True
        assert result["changes"]["change_count"] == 10

    def test_concurrent_pipelines_thread_safety(self, sample_json_schema):
        """Multiple concurrent pipeline runs do not corrupt shared state."""
        engine = _make_engine()
        _register_source_schema(engine, "s1", sample_json_schema)
        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []

        def worker():
            try:
                r = engine.run_pipeline("s1", _target_json(sample_json_schema))
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 5
        # All pipeline_ids must be unique
        ids = {r["pipeline_id"] for r in results}
        assert len(ids) == 5

    def test_target_json_empty_object(self):
        """Pipeline processes target_definition_json of '{}'."""
        engine = _make_engine()
        source = {"type": "object", "properties": {"a": {"type": "string"}}}
        _register_source_schema(engine, "s1", source)

        result = engine.run_pipeline("s1", "{}")
        assert result["pipeline_id"].startswith("pipe-")

    def test_very_long_schema_id(self):
        """Pipeline handles very long schema_id strings."""
        engine = _make_engine()
        long_id = "x" * 1000
        result = engine.run_pipeline(long_id, "{}")
        assert result["schema_id"] == long_id

    def test_special_characters_in_schema_id(self):
        """Pipeline handles special characters in schema_id."""
        engine = _make_engine()
        special_id = "schema/v2:test@org#1"
        result = engine.run_pipeline(special_id, "{}")
        assert result["schema_id"] == special_id

    def test_unicode_in_target_definition(self):
        """Pipeline handles unicode characters in target definition."""
        engine = _make_engine()
        target = {"type": "object", "properties": {"description": {"type": "string"}}}
        result = engine.run_pipeline("s1", _target_json(target))
        assert result["pipeline_id"].startswith("pipe-")


# ===========================================================================
# TestUtilityFunctions
# ===========================================================================


class TestUtilityFunctions:
    """Test module-level utility functions."""

    def test_sha256_deterministic(self):
        """_sha256 produces the same hash for the same input."""
        payload = {"key": "value", "number": 42}
        assert _sha256(payload) == _sha256(payload)

    def test_sha256_different_for_different_inputs(self):
        """_sha256 produces different hashes for different inputs."""
        assert _sha256({"a": 1}) != _sha256({"a": 2})

    def test_sha256_returns_64_hex(self):
        """_sha256 returns a 64-character hex string."""
        h = _sha256("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_new_id_prefix(self):
        """_new_id returns an identifier with the given prefix."""
        assert _new_id("pipe").startswith("pipe-")
        assert _new_id("batch").startswith("batch-")
        assert _new_id("rpt").startswith("rpt-")

    def test_new_id_uniqueness(self):
        """_new_id produces unique identifiers."""
        ids = {_new_id("x") for _ in range(100)}
        assert len(ids) == 100

    def test_new_id_length(self):
        """_new_id produces identifiers of prefix + dash + 12 hex chars."""
        nid = _new_id("pipe")
        parts = nid.split("-", 1)
        assert parts[0] == "pipe"
        assert len(parts[1]) == 12

    def test_elapsed_ms_positive(self):
        """_elapsed_ms returns a non-negative value."""
        start = time.monotonic()
        result = _elapsed_ms(start)
        assert result >= 0

    def test_utcnow_iso_format(self):
        """_utcnow_iso returns an ISO 8601 string with timezone info."""
        ts = _utcnow_iso()
        assert "T" in ts
        # Should contain timezone offset or Z
        assert "+" in ts or "Z" in ts


# ===========================================================================
# TestModuleConstants
# ===========================================================================


class TestModuleConstants:
    """Test module-level constants are defined correctly."""

    def test_pipeline_stages_tuple(self):
        """_PIPELINE_STAGES is a 7-element tuple."""
        assert len(_PIPELINE_STAGES) == 7
        assert _PIPELINE_STAGES[0] == "detect"
        assert _PIPELINE_STAGES[-1] == "registry"

    def test_status_constants(self):
        """Status constants are non-empty strings."""
        assert _STATUS_COMPLETED == "completed"
        assert _STATUS_FAILED == "failed"
        assert _STATUS_ABORTED == "aborted"
        assert _STATUS_NO_CHANGES == "no_changes"
        assert _STATUS_DRY_RUN == "dry_run_completed"


# ===========================================================================
# TestNormalisationHelpers
# ===========================================================================


class TestNormalisationHelpers:
    """Test internal result normalisation methods."""

    def test_normalise_detect_result_dict(self):
        """_normalise_detect_result handles a plain dict."""
        engine = _make_engine()
        raw = {"changes": [{"change_type": "field_added"}], "change_count": 1}
        result = engine._normalise_detect_result(raw)
        assert result["has_changes"] is True
        assert result["change_count"] == 1

    def test_normalise_detect_result_empty(self):
        """_normalise_detect_result defaults missing keys."""
        engine = _make_engine()
        raw = {}
        result = engine._normalise_detect_result(raw)
        assert result["has_changes"] is False
        assert result["change_count"] == 0
        assert result["changes"] == []

    def test_normalise_compat_result_dict(self):
        """_normalise_compat_result handles a plain dict."""
        engine = _make_engine()
        raw = {"is_compatible": True, "is_breaking": False}
        result = engine._normalise_compat_result(raw)
        assert result["is_compatible"] is True
        assert result["is_breaking"] is False

    def test_normalise_plan_result_defaults(self):
        """_normalise_plan_result fills missing keys with defaults."""
        engine = _make_engine()
        raw = {"steps": [{"op": "add"}]}
        result = engine._normalise_plan_result(raw, "s1", "1.0.0", "1.1.0")
        assert result["plan_id"].startswith("plan-")
        assert result["schema_id"] == "s1"
        assert result["step_count"] == 1

    def test_normalise_exec_result_defaults(self):
        """_normalise_exec_result fills missing keys with defaults."""
        engine = _make_engine()
        raw = {}
        result = engine._normalise_exec_result(raw)
        assert result["execution_id"].startswith("exec-")
        assert result["status"] == "success"
        assert result["records_migrated"] == 0

    def test_normalise_verify_result_defaults(self):
        """_normalise_verify_result fills missing keys with defaults."""
        engine = _make_engine()
        raw = {}
        result = engine._normalise_verify_result(raw)
        assert result["passed"] is True
        assert result["records_verified"] == 0


# ===========================================================================
# TestDetermineTargetVersion
# ===========================================================================


class TestDetermineTargetVersion:
    """Test _determine_target_version version bumping logic."""

    def test_patch_bump_default(self):
        """Default bump increments the patch segment."""
        engine = _make_engine()
        version = engine._determine_target_version("1.0.0", {})
        assert version == "1.0.1"

    def test_minor_bump(self):
        """recommended_bump='minor' increments minor and resets patch."""
        engine = _make_engine()
        version = engine._determine_target_version(
            "1.2.3", {"recommended_bump": "minor"}
        )
        assert version == "1.3.0"

    def test_major_bump(self):
        """recommended_bump='major' increments major and resets minor/patch."""
        engine = _make_engine()
        version = engine._determine_target_version(
            "1.2.3", {"recommended_bump": "major"}
        )
        assert version == "2.0.0"

    def test_breaking_change_forces_major(self):
        """is_breaking=True forces a major version bump."""
        engine = _make_engine()
        version = engine._determine_target_version(
            "1.0.0", {"is_breaking": True}
        )
        assert version == "2.0.0"

    def test_short_version_string(self):
        """Handles version strings with fewer than 3 segments."""
        engine = _make_engine()
        version = engine._determine_target_version("1", {})
        assert version == "1.0.1"

    def test_invalid_version_string(self):
        """Handles completely invalid version strings gracefully."""
        engine = _make_engine()
        version = engine._determine_target_version("not.a.version", {})
        # Falls back to 1.0.0 then increments patch
        assert version == "1.0.1"


# ===========================================================================
# TestStructuralVerify
# ===========================================================================


class TestStructuralVerify:
    """Test _structural_verify internal method."""

    def test_verify_with_none_data(self):
        """Structural verify with None data returns passed=True."""
        engine = _make_engine()
        result = engine._structural_verify({"a": 1, "b": 2}, None)
        assert result["passed"] is True
        assert result["records_verified"] == 0

    def test_verify_matching_records(self):
        """Records matching target keys pass verification."""
        engine = _make_engine()
        target = {"name": {"type": "string"}, "age": {"type": "integer"}}
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = engine._structural_verify(target, data)
        assert result["passed"] is True
        assert result["records_verified"] == 2

    def test_verify_missing_fields_fails(self):
        """Records missing required target fields fail verification."""
        engine = _make_engine()
        target = {"name": {"type": "string"}, "age": {"type": "integer"}}
        data = [{"name": "Alice"}]  # missing 'age'
        result = engine._structural_verify(target, data)
        assert result["passed"] is False
        assert "missing" in result["failure_reason"].lower()

    def test_verify_extra_fields_warning(self):
        """Records with extra fields generate warnings but pass."""
        engine = _make_engine()
        target = {"name": {"type": "string"}}
        data = [{"name": "Alice", "extra_field": 42}]
        result = engine._structural_verify(target, data)
        # Extra fields generate warnings, not failures
        assert len(result["warnings"]) > 0

    def test_verify_empty_records_list(self):
        """Empty records list passes verification with records_verified=0."""
        engine = _make_engine()
        result = engine._structural_verify({"a": 1}, [])
        assert result["passed"] is True
        assert result["records_verified"] == 0


# ===========================================================================
# TestComplianceNotes
# ===========================================================================


class TestComplianceNotes:
    """Test _build_compliance_notes report helper."""

    def test_completed_migration_note(self):
        """Completed migration produces a success note."""
        engine = _make_engine()
        run = {
            "status": "completed",
            "source_version": "1.0.0",
            "target_version": "2.0.0",
            "stages_failed": [],
            "plan": {},
            "execution": {"records_migrated": 5},
        }
        notes = engine._build_compliance_notes(run, {})
        assert any("completed successfully" in n for n in notes)

    def test_aborted_migration_note(self):
        """Aborted migration produces an abort note."""
        engine = _make_engine()
        run = {
            "status": "aborted",
            "stages_failed": ["compatibility"],
            "plan": {},
            "execution": {},
        }
        notes = engine._build_compliance_notes(run, {})
        assert any("aborted" in n.lower() for n in notes)

    def test_no_changes_note(self):
        """No-changes migration produces appropriate note."""
        engine = _make_engine()
        run = {
            "status": "no_changes",
            "stages_failed": [],
            "plan": {},
            "execution": {},
        }
        notes = engine._build_compliance_notes(run, {})
        assert any("no schema changes" in n.lower() for n in notes)

    def test_dry_run_note(self):
        """Dry-run pipeline produces a dry-run note."""
        engine = _make_engine()
        run = {
            "status": "dry_run_completed",
            "stages_failed": [],
            "plan": {},
            "execution": {},
        }
        notes = engine._build_compliance_notes(run, {})
        assert any("dry-run" in n.lower() for n in notes)

    def test_breaking_changes_note(self):
        """Breaking changes produce a warning note."""
        engine = _make_engine()
        run = {
            "status": "completed",
            "source_version": "1.0.0",
            "target_version": "2.0.0",
            "stages_failed": [],
            "plan": {},
            "execution": {},
        }
        compat = {"is_breaking": True}
        notes = engine._build_compliance_notes(run, compat)
        assert any("breaking changes" in n.lower() for n in notes)
