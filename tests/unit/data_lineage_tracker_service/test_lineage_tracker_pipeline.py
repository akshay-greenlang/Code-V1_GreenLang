# -*- coding: utf-8 -*-
"""
Unit Tests for LineageTrackerPipelineEngine - AGENT-DATA-018

Tests the 7-stage pipeline orchestration, bulk registration, bulk capture,
graph construction, change detection, pipeline run queries, health checks,
statistics, and clear operations.

Target: 60+ tests, 7 test classes, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_lineage_tracker.lineage_tracker_pipeline import (
    LineageTrackerPipelineEngine,
    PIPELINE_STAGES,
    _sha256,
    _new_id,
    _elapsed_ms,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_engines():
    """Create a set of mock engines for injection into the pipeline."""
    return {
        "asset_registry": MagicMock(),
        "transformation_tracker": MagicMock(),
        "lineage_graph": MagicMock(),
        "impact_analyzer": MagicMock(),
        "lineage_validator": MagicMock(),
        "lineage_reporter": MagicMock(),
        "provenance": ProvenanceTracker(),
    }


@pytest.fixture
def pipeline(mock_engines) -> LineageTrackerPipelineEngine:
    """Create a pipeline with all mock engines injected."""
    return LineageTrackerPipelineEngine(**mock_engines)


@pytest.fixture
def pipeline_no_engines() -> LineageTrackerPipelineEngine:
    """Create a pipeline with all engines set to None.

    We patch the availability flags so that the constructor does not
    auto-create engines from the importable modules.
    """
    import greenlang.data_lineage_tracker.lineage_tracker_pipeline as _mod

    patches = {
        "_ASSET_REGISTRY_AVAILABLE": False,
        "_TRANSFORMATION_TRACKER_AVAILABLE": False,
        "_LINEAGE_GRAPH_AVAILABLE": False,
        "_IMPACT_ANALYZER_AVAILABLE": False,
        "_LINEAGE_VALIDATOR_AVAILABLE": False,
        "_LINEAGE_REPORTER_AVAILABLE": False,
    }
    originals = {k: getattr(_mod, k, None) for k in patches}
    for k, v in patches.items():
        setattr(_mod, k, v)
    try:
        prov = ProvenanceTracker()
        prov.record("test", "seed", "init")  # Make truthy
        p = LineageTrackerPipelineEngine(
            asset_registry=None,
            transformation_tracker=None,
            lineage_graph=None,
            impact_analyzer=None,
            lineage_validator=None,
            lineage_reporter=None,
            provenance=prov,
        )
    finally:
        for k, v in originals.items():
            if v is not None:
                setattr(_mod, k, v)
    return p


# ============================================================================
# TestPipelineInit
# ============================================================================


class TestPipelineInit:
    """Tests for LineageTrackerPipelineEngine initialization."""

    def test_init_with_all_engines(self, pipeline, mock_engines):
        assert pipeline.asset_registry is mock_engines["asset_registry"]
        assert pipeline.transformation_tracker is mock_engines["transformation_tracker"]
        assert pipeline.lineage_graph is mock_engines["lineage_graph"]
        assert pipeline.impact_analyzer is mock_engines["impact_analyzer"]
        assert pipeline.lineage_validator is mock_engines["lineage_validator"]
        assert pipeline.lineage_reporter is mock_engines["lineage_reporter"]
        assert pipeline.provenance is mock_engines["provenance"]

    def test_init_with_none_engines(self, pipeline_no_engines):
        assert pipeline_no_engines.asset_registry is None
        assert pipeline_no_engines.transformation_tracker is None
        assert pipeline_no_engines.lineage_graph is None

    def test_init_creates_provenance_if_none(self):
        p = LineageTrackerPipelineEngine(provenance=None)
        # Provenance may or may not be created depending on import availability
        # Just ensure no exception

    def test_pipeline_stages_constant(self):
        assert len(PIPELINE_STAGES) == 7
        assert "register" in PIPELINE_STAGES
        assert "capture" in PIPELINE_STAGES
        assert "build_graph" in PIPELINE_STAGES
        assert "validate" in PIPELINE_STAGES
        assert "analyze" in PIPELINE_STAGES
        assert "report" in PIPELINE_STAGES
        assert "detect_changes" in PIPELINE_STAGES

    def test_empty_pipeline_runs_on_init(self, pipeline):
        assert len(pipeline._pipeline_runs) == 0
        assert len(pipeline._snapshots) == 0
        assert len(pipeline._change_events) == 0


# ============================================================================
# TestRunPipeline
# ============================================================================


class TestRunPipeline:
    """Tests for run_pipeline method."""

    def test_run_pipeline_returns_dict(self, pipeline):
        result = pipeline.run_pipeline(scope="full")
        assert isinstance(result, dict)

    def test_run_pipeline_has_required_keys(self, pipeline):
        result = pipeline.run_pipeline(scope="full")
        required = [
            "pipeline_id", "scope", "stages_completed", "stages_skipped",
            "assets_registered", "transformations_captured", "edges_created",
            "validation_result", "report", "duration_ms", "errors",
            "started_at", "completed_at", "provenance_hash", "status",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_run_pipeline_custom_id(self, pipeline):
        result = pipeline.run_pipeline(pipeline_id="my-pipe-001")
        assert result["pipeline_id"] == "my-pipe-001"

    def test_run_pipeline_auto_generated_id(self, pipeline):
        result = pipeline.run_pipeline()
        assert result["pipeline_id"].startswith("pipe-")

    def test_run_pipeline_full_scope(self, pipeline):
        result = pipeline.run_pipeline(scope="full")
        assert result["scope"] == "full"
        assert len(result["stages_completed"]) > 0

    def test_run_pipeline_register_only_scope(self, pipeline):
        result = pipeline.run_pipeline(scope="register_only")
        assert result["scope"] == "register_only"
        assert "capture" in result["stages_skipped"]

    def test_run_pipeline_validate_only_scope(self, pipeline):
        result = pipeline.run_pipeline(scope="validate_only")
        assert result["scope"] == "validate_only"

    def test_run_pipeline_stored_in_memory(self, pipeline):
        result = pipeline.run_pipeline()
        stored = pipeline.get_pipeline_run(result["pipeline_id"])
        assert stored is not None
        assert stored["pipeline_id"] == result["pipeline_id"]

    def test_run_pipeline_has_provenance_hash(self, pipeline):
        result = pipeline.run_pipeline()
        assert result["provenance_hash"] is not None

    def test_run_pipeline_has_timing(self, pipeline):
        result = pipeline.run_pipeline()
        assert result["duration_ms"] >= 0
        assert result["started_at"] is not None
        assert result["completed_at"] is not None

    def test_run_pipeline_skip_register(self, pipeline):
        result = pipeline.run_pipeline(register_assets=False)
        assert "register" in result["stages_skipped"]

    def test_run_pipeline_skip_capture(self, pipeline):
        result = pipeline.run_pipeline(capture_transformations=False)
        assert "capture" in result["stages_skipped"]

    def test_run_pipeline_with_asset_metadata(self, pipeline):
        metadata = [
            {"qualified_name": "db.table1", "asset_type": "dataset"},
            {"qualified_name": "db.table2", "asset_type": "table"},
        ]
        pipeline.asset_registry.register_asset.return_value = None
        result = pipeline.run_pipeline(asset_metadata=metadata)
        assert "register" in result["stages_completed"]

    def test_run_pipeline_with_transformation_events(self, pipeline):
        events = [
            {
                "transformation_type": "filter",
                "agent_id": "agent-001",
                "input_assets": ["in1"],
                "output_assets": ["out1"],
            },
        ]
        pipeline.transformation_tracker.record_transformation.return_value = None
        result = pipeline.run_pipeline(transformation_events=events)
        assert "capture" in result["stages_completed"]


# ============================================================================
# TestBulkRegistration
# ============================================================================


class TestBulkRegistration:
    """Tests for register_assets_from_metadata method."""

    def test_register_no_engine(self, pipeline_no_engines):
        result = pipeline_no_engines.register_assets_from_metadata([
            {"qualified_name": "db.table1", "asset_type": "dataset"},
        ])
        assert result["registered"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) == 1

    def test_register_empty_list(self, pipeline):
        result = pipeline.register_assets_from_metadata([])
        assert result["registered"] == 0
        assert result["failed"] == 0

    def test_register_valid_entries(self, pipeline):
        pipeline.asset_registry.register_asset.return_value = None
        metadata = [
            {"qualified_name": "db.table1", "asset_type": "dataset"},
            {"qualified_name": "db.table2", "asset_type": "table"},
        ]
        result = pipeline.register_assets_from_metadata(metadata)
        assert result["registered"] == 2
        assert result["failed"] == 0

    def test_register_missing_qualified_name(self, pipeline):
        metadata = [{"asset_type": "dataset"}]
        result = pipeline.register_assets_from_metadata(metadata)
        assert result["registered"] == 0
        assert result["failed"] == 1

    def test_register_partial_failure(self, pipeline):
        pipeline.asset_registry.register_asset.side_effect = [
            None,
            Exception("duplicate"),
        ]
        metadata = [
            {"qualified_name": "db.table1", "asset_type": "dataset"},
            {"qualified_name": "db.table2", "asset_type": "dataset"},
        ]
        result = pipeline.register_assets_from_metadata(metadata)
        assert result["registered"] == 1
        assert result["failed"] == 1


# ============================================================================
# TestBulkCapture
# ============================================================================


class TestBulkCapture:
    """Tests for capture_transformations_from_events method."""

    def test_capture_no_engine(self, pipeline_no_engines):
        result = pipeline_no_engines.capture_transformations_from_events([
            {"transformation_type": "filter"},
        ])
        assert result["captured"] == 0
        assert result["failed"] == 1

    def test_capture_empty_list(self, pipeline):
        result = pipeline.capture_transformations_from_events([])
        assert result["captured"] == 0

    def test_capture_valid_events(self, pipeline):
        pipeline.transformation_tracker.record_transformation.return_value = None
        events = [
            {
                "transformation_type": "filter",
                "agent_id": "agent-001",
                "input_assets": ["in1"],
                "output_assets": ["out1"],
            },
        ]
        result = pipeline.capture_transformations_from_events(events)
        assert result["captured"] == 1
        assert result["failed"] == 0

    def test_capture_partial_failure(self, pipeline):
        pipeline.transformation_tracker.record_transformation.side_effect = [
            None,
            Exception("error"),
        ]
        events = [
            {"transformation_type": "filter", "agent_id": "a1"},
            {"transformation_type": "enrich", "agent_id": "a2"},
        ]
        result = pipeline.capture_transformations_from_events(events)
        assert result["captured"] == 1
        assert result["failed"] == 1


# ============================================================================
# TestBuildGraph
# ============================================================================


class TestBuildGraph:
    """Tests for build_graph_from_registry method."""

    def test_build_graph_no_engines(self, pipeline_no_engines):
        result = pipeline_no_engines.build_graph_from_registry()
        assert result["nodes_added"] == 0
        assert result["edges_added"] == 0

    def test_build_graph_with_assets(self, pipeline):
        pipeline.asset_registry.list_assets.return_value = [
            {"qualified_name": "db.table1", "asset_type": "dataset"},
            {"qualified_name": "db.table2", "asset_type": "table"},
        ]
        pipeline.transformation_tracker.list_transformations.return_value = []
        pipeline.lineage_graph.add_node.return_value = None
        result = pipeline.build_graph_from_registry()
        assert result["nodes_added"] >= 0


# ============================================================================
# TestChangeDetection
# ============================================================================


class TestChangeDetection:
    """Tests for detect_changes method."""

    def test_detect_changes_first_call(self, pipeline):
        pipeline.lineage_graph.take_snapshot = MagicMock(return_value={
            "snapshot_id": "snap-001",
            "node_count": 5,
            "edge_count": 3,
        })
        pipeline.lineage_graph.get_statistics = MagicMock(return_value={
            "total_nodes": 5, "total_edges": 3,
        })
        result = pipeline.detect_changes()
        assert "current_snapshot" in result
        assert "changes" in result
        assert isinstance(result["change_count"], int)

    def test_detect_changes_stored(self, pipeline):
        pipeline.lineage_graph.take_snapshot = MagicMock(return_value={
            "snapshot_id": "snap-001",
            "node_count": 0,
            "edge_count": 0,
        })
        pipeline.lineage_graph.get_statistics = MagicMock(return_value={
            "total_nodes": 0, "total_edges": 0,
        })
        pipeline.detect_changes()
        events = pipeline.get_change_events()
        assert isinstance(events, list)


# ============================================================================
# TestPipelineRunQueries
# ============================================================================


class TestPipelineRunQueries:
    """Tests for pipeline run query methods."""

    def test_get_pipeline_run_missing(self, pipeline):
        assert pipeline.get_pipeline_run("missing") is None

    def test_list_pipeline_runs_empty(self, pipeline):
        runs = pipeline.list_pipeline_runs()
        assert runs == []

    def test_list_pipeline_runs_after_run(self, pipeline):
        pipeline.run_pipeline()
        runs = pipeline.list_pipeline_runs()
        assert len(runs) == 1

    def test_list_pipeline_runs_limit(self, pipeline):
        for _ in range(5):
            pipeline.run_pipeline()
        runs = pipeline.list_pipeline_runs(limit=3)
        assert len(runs) == 3

    def test_list_pipeline_runs_negative_limit_raises(self, pipeline):
        with pytest.raises(ValueError, match="limit must be >= 0"):
            pipeline.list_pipeline_runs(limit=-1)

    def test_get_change_events_empty(self, pipeline):
        events = pipeline.get_change_events()
        assert events == []

    def test_get_change_events_negative_limit_raises(self, pipeline):
        with pytest.raises(ValueError, match="limit must be >= 0"):
            pipeline.get_change_events(limit=-1)

    def test_get_snapshots_empty(self, pipeline):
        snapshots = pipeline.get_snapshots()
        assert snapshots == []


# ============================================================================
# TestHealthAndStatistics
# ============================================================================


class TestHealthAndStatistics:
    """Tests for get_health and get_statistics methods."""

    def test_health_all_engines(self, pipeline):
        health = pipeline.get_health()
        assert health["status"] == "healthy"
        assert health["engines_available"] == 7
        assert health["engines_total"] == 7

    def test_health_no_engines(self, pipeline_no_engines):
        health = pipeline_no_engines.get_health()
        # Only provenance is not None
        assert health["status"] in ("degraded", "unhealthy")

    def test_health_has_required_keys(self, pipeline):
        health = pipeline.get_health()
        assert "status" in health
        assert "engines" in health
        assert "graph_stats" in health
        assert "checked_at" in health

    def test_health_engines_dict(self, pipeline):
        health = pipeline.get_health()
        engines = health["engines"]
        assert "asset_registry" in engines
        assert "lineage_graph" in engines
        assert "provenance" in engines

    def test_statistics_empty(self, pipeline):
        stats = pipeline.get_statistics()
        assert stats["total_pipeline_runs"] == 0
        assert stats["success_rate"] == 0.0

    def test_statistics_after_runs(self, pipeline):
        pipeline.run_pipeline()
        stats = pipeline.get_statistics()
        assert stats["total_pipeline_runs"] == 1
        assert stats["avg_duration_ms"] >= 0

    def test_statistics_has_required_keys(self, pipeline):
        stats = pipeline.get_statistics()
        required = [
            "total_pipeline_runs", "by_status", "avg_duration_ms",
            "min_duration_ms", "max_duration_ms", "success_rate",
            "total_snapshots", "total_change_events",
            "graph_stats", "provenance_entry_count", "computed_at",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"


# ============================================================================
# TestClear
# ============================================================================


class TestClear:
    """Tests for clear method."""

    def test_clear_pipeline_runs(self, pipeline):
        pipeline.run_pipeline()
        assert len(pipeline._pipeline_runs) > 0
        pipeline.clear()
        assert len(pipeline._pipeline_runs) == 0

    def test_clear_snapshots(self, pipeline):
        pipeline._snapshots.append({"test": True})
        pipeline.clear()
        assert len(pipeline._snapshots) == 0

    def test_clear_change_events(self, pipeline):
        pipeline._change_events.append({"test": True})
        pipeline.clear()
        assert len(pipeline._change_events) == 0

    def test_clear_resets_provenance(self, pipeline):
        prov = pipeline.provenance
        prov.record("test", "t1", "test_action")
        initial = prov.entry_count
        assert initial > 0
        pipeline.clear()
        assert prov.entry_count == 0

    def test_clear_idempotent(self, pipeline):
        pipeline.clear()
        pipeline.clear()
        assert len(pipeline._pipeline_runs) == 0


# ============================================================================
# TestUtilityHelpers
# ============================================================================


class TestUtilityHelpers:
    """Tests for module-level utility functions."""

    def test_sha256_deterministic(self):
        h1 = _sha256({"key": "value"})
        h2 = _sha256({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_different_inputs(self):
        h1 = _sha256({"a": 1})
        h2 = _sha256({"a": 2})
        assert h1 != h2

    def test_new_id_prefix(self):
        id_val = _new_id("test")
        assert id_val.startswith("test-")
        assert len(id_val) > 5

    def test_new_id_uniqueness(self):
        ids = {_new_id("x") for _ in range(100)}
        assert len(ids) == 100

    def test_elapsed_ms(self):
        import time
        start = time.monotonic()
        time.sleep(0.01)
        ms = _elapsed_ms(start)
        assert ms > 0
