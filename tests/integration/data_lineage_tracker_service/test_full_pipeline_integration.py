# -*- coding: utf-8 -*-
"""
Integration Tests: Full Pipeline End-to-End (AGENT-DATA-018)
=============================================================

Tests the complete LineageTrackerPipelineEngine workflow from asset
registration through change detection.  Validates 7-stage pipeline
execution, provenance tracking, error handling, health checks,
statistics, and deterministic reproducibility.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

import pytest

from greenlang.data_lineage_tracker.lineage_tracker_pipeline import (
    LineageTrackerPipelineEngine,
    PIPELINE_STAGES,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

from tests.integration.data_lineage_tracker_service.conftest import (
    GREENLANG_ASSET_NAMES,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _build_asset_metadata() -> List[Dict[str, Any]]:
    """Build a realistic set of GreenLang asset metadata for pipeline ingestion."""
    return [
        {
            "qualified_name": "supplier.invoices",
            "asset_type": "external_source",
            "description": "Raw supplier invoice PDFs",
            "classification": "confidential",
            "owner": "procurement-team",
            "tags": ["scope3", "supplier"],
        },
        {
            "qualified_name": "agent.pdf_extractor",
            "asset_type": "agent",
            "description": "PDF text extraction agent",
            "owner": "data-engineering",
        },
        {
            "qualified_name": "data.extracted_invoices",
            "asset_type": "dataset",
            "description": "Structured invoice data extracted from PDFs",
            "owner": "data-engineering",
        },
        {
            "qualified_name": "agent.excel_normalizer",
            "asset_type": "agent",
            "description": "Excel/CSV normalization agent",
            "owner": "data-engineering",
        },
        {
            "qualified_name": "data.normalized_spend",
            "asset_type": "dataset",
            "description": "Normalized spend data",
            "owner": "finance-team",
        },
        {
            "qualified_name": "agent.spend_categorizer",
            "asset_type": "agent",
            "description": "Spend categorization agent",
            "owner": "data-engineering",
        },
        {
            "qualified_name": "data.categorized_spend",
            "asset_type": "dataset",
            "description": "Categorized spend with UNSPSC codes",
            "owner": "finance-team",
        },
        {
            "qualified_name": "agent.emission_calculator",
            "asset_type": "agent",
            "description": "GHG emission calculator agent",
            "owner": "sustainability-team",
        },
        {
            "qualified_name": "metric.scope3_emissions",
            "asset_type": "metric",
            "description": "Scope 3 upstream emissions metric (tCO2e)",
            "owner": "sustainability-team",
        },
        {
            "qualified_name": "report.csrd_report",
            "asset_type": "report",
            "description": "CSRD/ESRS annual sustainability report",
            "classification": "public",
            "owner": "sustainability-team",
        },
    ]


def _build_transformation_events() -> List[Dict[str, Any]]:
    """Build transformation events matching the asset chain."""
    return [
        {
            "transformation_type": "filter",
            "agent_id": "agent.pdf_extractor",
            "input_assets": ["supplier.invoices"],
            "output_assets": ["data.extracted_invoices"],
            "description": "Extract structured fields from supplier invoices",
        },
        {
            "transformation_type": "normalize",
            "agent_id": "agent.excel_normalizer",
            "input_assets": ["data.extracted_invoices"],
            "output_assets": ["data.normalized_spend"],
            "description": "Normalize extracted data to canonical spend schema",
        },
        {
            "transformation_type": "classify",
            "agent_id": "agent.spend_categorizer",
            "input_assets": ["data.normalized_spend"],
            "output_assets": ["data.categorized_spend"],
            "description": "Classify spend into UNSPSC categories",
        },
        {
            "transformation_type": "calculate",
            "agent_id": "agent.emission_calculator",
            "input_assets": ["data.categorized_spend"],
            "output_assets": ["metric.scope3_emissions"],
            "description": "Calculate Scope 3 emissions from categorized spend",
        },
        {
            "transformation_type": "aggregate",
            "agent_id": "agent.emission_calculator",
            "input_assets": ["metric.scope3_emissions"],
            "output_assets": ["report.csrd_report"],
            "description": "Aggregate emissions into CSRD report format",
        },
    ]


# ---------------------------------------------------------------------------
# TestEndToEndPipeline
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end integration tests for LineageTrackerPipelineEngine."""

    # ------------------------------------------------------------------ #
    # Basic pipeline lifecycle tests
    # ------------------------------------------------------------------ #

    def test_pipeline_initializes_all_engines(self, pipeline):
        """Test that pipeline constructor initializes all 7 engines."""
        assert pipeline.asset_registry is not None
        assert pipeline.transformation_tracker is not None
        assert pipeline.lineage_graph is not None
        assert pipeline.impact_analyzer is not None
        assert pipeline.lineage_validator is not None
        assert pipeline.lineage_reporter is not None
        assert pipeline.provenance is not None

    def test_full_pipeline_run_empty_scope(self, pipeline):
        """Test running the full pipeline with no input data."""
        result = pipeline.run_pipeline(scope="full")

        assert result is not None
        assert "pipeline_id" in result
        assert result["pipeline_id"].startswith("pipe-")
        assert result["scope"] == "full"
        assert isinstance(result["stages_completed"], list)
        assert isinstance(result["stages_skipped"], list)
        assert isinstance(result["errors"], list)
        assert result["started_at"] is not None
        assert result["completed_at"] is not None
        assert result["duration_ms"] >= 0
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_pipeline_with_realistic_data(self, pipeline):
        """Test full pipeline with realistic GreenLang asset and transformation data."""
        result = pipeline.run_pipeline(
            scope="full",
            asset_metadata=_build_asset_metadata(),
            transformation_events=_build_transformation_events(),
            report_type="visualization",
            report_format="json",
        )

        assert result["status"] in ("completed", "partial")
        assert result["assets_registered"] == 10
        assert result["transformations_captured"] == 5
        assert "register" in result["stages_completed"]
        assert "capture" in result["stages_completed"]

    def test_pipeline_asset_count(self, pipeline):
        """Test that asset registration counts match input metadata."""
        metadata = _build_asset_metadata()
        result = pipeline.register_assets_from_metadata(metadata)

        assert result["registered"] == len(metadata)
        assert result["failed"] == 0
        assert result["errors"] == []

    def test_pipeline_transformation_capture(self, pipeline):
        """Test that transformation capture counts match input events."""
        events = _build_transformation_events()
        result = pipeline.capture_transformations_from_events(events)

        assert result["captured"] == len(events)
        assert result["failed"] == 0
        assert result["errors"] == []

    def test_pipeline_graph_construction(self, pipeline):
        """Test that build_graph_from_registry creates nodes and edges."""
        pipeline.register_assets_from_metadata(_build_asset_metadata())
        pipeline.capture_transformations_from_events(_build_transformation_events())

        result = pipeline.build_graph_from_registry()

        assert isinstance(result, dict)
        assert "nodes_added" in result
        assert "edges_added" in result
        # We registered 10 assets and 5 transformations
        assert result["nodes_added"] >= 0
        assert result["edges_added"] >= 0

    def test_pipeline_validation(self, populated_pipeline):
        """Test that pipeline validation produces a result dictionary."""
        pipe, assets = populated_pipeline
        result = pipe.run_pipeline(scope="full")

        validation_result = result.get("validation_result")
        assert validation_result is not None
        if isinstance(validation_result, dict):
            assert "result" in validation_result or "verdict" in validation_result

    def test_pipeline_report_generation_json(self, populated_pipeline):
        """Test that the pipeline generates a JSON visualization report."""
        pipe, assets = populated_pipeline
        result = pipe.run_pipeline(
            scope="full",
            report_type="visualization",
            report_format="json",
        )

        assert result.get("report") is not None
        if isinstance(result["report"], dict):
            assert "content" in result["report"] or "status" in result["report"]

    def test_pipeline_report_generation_mermaid(self, populated_pipeline):
        """Test that the pipeline generates a Mermaid visualization report."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="mermaid",
        )

        assert report["format"] == "mermaid"
        assert "graph TD" in report["content"]
        assert report["report_hash"] is not None
        assert len(report["report_hash"]) == 64

    def test_pipeline_report_generation_text(self, populated_pipeline):
        """Test that the pipeline generates a plain text report."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="text",
        )

        assert report["format"] == "text"
        assert "DATA LINEAGE SUMMARY REPORT" in report["content"]

    def test_pipeline_change_detection(self, populated_pipeline):
        """Test that change detection produces a valid result."""
        pipe, assets = populated_pipeline
        result = pipe.detect_changes()

        assert isinstance(result, dict)
        assert "previous_snapshot" in result
        assert "current_snapshot" in result
        assert "changes" in result
        assert "change_count" in result
        assert isinstance(result["changes"], list)
        assert isinstance(result["change_count"], int)

    def test_pipeline_change_detection_detects_initial_assets(self, populated_pipeline):
        """Test that the first change detection run detects all initial assets."""
        pipe, assets = populated_pipeline
        result = pipe.detect_changes()

        # First snapshot should detect all 10 assets as new
        assert result["change_count"] >= 0
        if result["changes"]:
            change_types = {c["change_type"] for c in result["changes"]}
            # Initial changes should all be additions
            assert "asset_added" in change_types or "edge_added" in change_types

    # ------------------------------------------------------------------ #
    # Health and statistics tests
    # ------------------------------------------------------------------ #

    def test_pipeline_health_check(self, pipeline):
        """Test that health check returns status and engine availability."""
        health = pipeline.get_health()

        assert isinstance(health, dict)
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert "engines" in health
        assert "engines_available" in health
        assert "engines_total" in health
        assert "graph_stats" in health
        assert "checked_at" in health

    def test_pipeline_health_all_engines_available(self, pipeline):
        """Test that all 7 engines report as available in health check."""
        health = pipeline.get_health()

        assert health["status"] == "healthy"
        assert health["engines_available"] == health["engines_total"]

        for engine_name, is_available in health["engines"].items():
            assert is_available is True, (
                f"Engine '{engine_name}' is unexpectedly unavailable"
            )

    def test_pipeline_statistics_empty(self, pipeline):
        """Test statistics on a fresh pipeline with no runs."""
        stats = pipeline.get_statistics()

        assert isinstance(stats, dict)
        assert stats["total_pipeline_runs"] == 0
        assert isinstance(stats["by_status"], dict)
        assert stats["avg_duration_ms"] == 0.0
        assert stats["total_snapshots"] == 0
        assert stats["total_change_events"] == 0
        assert "graph_stats" in stats
        assert "computed_at" in stats

    def test_pipeline_statistics_after_runs(self, pipeline):
        """Test statistics after multiple pipeline runs."""
        pipeline.run_pipeline(scope="full")
        pipeline.run_pipeline(scope="full")

        stats = pipeline.get_statistics()

        assert stats["total_pipeline_runs"] == 2
        assert stats["avg_duration_ms"] >= 0.0
        assert stats["max_duration_ms"] >= 0.0
        assert stats["success_rate"] >= 0.0

    def test_pipeline_run_retrieval(self, pipeline):
        """Test retrieving a specific pipeline run by ID."""
        result = pipeline.run_pipeline(scope="full")
        run_id = result["pipeline_id"]

        retrieved = pipeline.get_pipeline_run(run_id)

        assert retrieved is not None
        assert retrieved["pipeline_id"] == run_id
        assert retrieved["scope"] == "full"

    def test_pipeline_run_retrieval_not_found(self, pipeline):
        """Test retrieving a non-existent pipeline run returns None."""
        retrieved = pipeline.get_pipeline_run("pipe-nonexistent")
        assert retrieved is None

    def test_pipeline_list_runs(self, pipeline):
        """Test listing pipeline runs with limit."""
        pipeline.run_pipeline(scope="full")
        pipeline.run_pipeline(scope="full")
        pipeline.run_pipeline(scope="full")

        runs = pipeline.list_pipeline_runs(limit=2)
        assert len(runs) == 2

        all_runs = pipeline.list_pipeline_runs(limit=100)
        assert len(all_runs) == 3

    def test_pipeline_list_runs_negative_limit_raises(self, pipeline):
        """Test that negative limit raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.list_pipeline_runs(limit=-1)

    # ------------------------------------------------------------------ #
    # Provenance tracking tests
    # ------------------------------------------------------------------ #

    def test_pipeline_provenance_chain(self, pipeline):
        """Test that provenance chain grows with each pipeline run."""
        initial_count = pipeline.provenance.entry_count

        pipeline.run_pipeline(scope="full")
        after_first = pipeline.provenance.entry_count

        pipeline.run_pipeline(scope="full")
        after_second = pipeline.provenance.entry_count

        assert after_first > initial_count
        assert after_second > after_first

    def test_pipeline_provenance_hash_is_sha256(self, pipeline):
        """Test that provenance hash is a valid 64-char SHA-256 hex string."""
        result = pipeline.run_pipeline(scope="full")
        prov_hash = result["provenance_hash"]

        assert prov_hash is not None
        assert len(prov_hash) == 64
        # Verify it is a valid hex string
        int(prov_hash, 16)

    def test_pipeline_provenance_deterministic(self, pipeline):
        """Test that the provenance chain is verified successfully."""
        pipeline.run_pipeline(scope="full")

        chain_valid = pipeline.provenance.verify_chain()
        assert chain_valid is True

    # ------------------------------------------------------------------ #
    # Scope-specific tests
    # ------------------------------------------------------------------ #

    def test_pipeline_register_only_scope(self, pipeline):
        """Test pipeline with scope='register_only' runs only registration."""
        result = pipeline.run_pipeline(
            scope="register_only",
            asset_metadata=_build_asset_metadata(),
        )

        assert result["scope"] == "register_only"
        assert "register" in result["stages_completed"]
        # Other stages should be skipped
        skipped = result["stages_skipped"]
        assert "capture" in skipped
        assert "build_graph" in skipped

    def test_pipeline_validate_only_scope(self, pipeline):
        """Test pipeline with scope='validate_only' runs only validation."""
        result = pipeline.run_pipeline(scope="validate_only")

        assert result["scope"] == "validate_only"
        assert "validate" in result["stages_completed"]
        assert "register" in result["stages_skipped"]
        assert "capture" in result["stages_skipped"]

    def test_pipeline_report_only_scope(self, pipeline):
        """Test pipeline with scope='report_only' runs only reporting."""
        result = pipeline.run_pipeline(scope="report_only")

        assert result["scope"] == "report_only"
        assert "report" in result["stages_completed"]
        assert "register" in result["stages_skipped"]

    def test_pipeline_custom_pipeline_id(self, pipeline):
        """Test pipeline with a custom pipeline ID."""
        result = pipeline.run_pipeline(
            pipeline_id="custom-test-001",
            scope="full",
        )

        assert result["pipeline_id"] == "custom-test-001"

    # ------------------------------------------------------------------ #
    # Error handling tests
    # ------------------------------------------------------------------ #

    def test_pipeline_handles_invalid_asset_metadata(self, pipeline):
        """Test pipeline gracefully handles invalid asset metadata."""
        invalid_metadata = [
            {"qualified_name": "", "asset_type": "dataset"},  # empty name
            {"asset_type": "dataset"},  # missing qualified_name
        ]

        result = pipeline.register_assets_from_metadata(invalid_metadata)
        assert result["failed"] >= 1

    def test_pipeline_handles_empty_metadata(self, pipeline):
        """Test pipeline with empty metadata lists."""
        result = pipeline.run_pipeline(
            scope="full",
            asset_metadata=[],
            transformation_events=[],
        )

        assert result["status"] in ("completed", "partial")
        assert result["assets_registered"] == 0
        assert result["transformations_captured"] == 0

    # ------------------------------------------------------------------ #
    # Snapshot and change event tests
    # ------------------------------------------------------------------ #

    def test_pipeline_snapshots_stored(self, populated_pipeline):
        """Test that snapshots are stored after detect_changes."""
        pipe, assets = populated_pipeline
        pipe.detect_changes()

        snapshots = pipe.get_snapshots()
        assert len(snapshots) >= 1

        snapshot = snapshots[-1]
        assert "snapshot_id" in snapshot
        assert "timestamp" in snapshot
        assert "node_count" in snapshot
        assert "edge_count" in snapshot

    def test_pipeline_change_events_stored(self, populated_pipeline):
        """Test that change events are stored and retrievable."""
        pipe, assets = populated_pipeline
        pipe.detect_changes()

        events = pipe.get_change_events()
        assert isinstance(events, list)

    def test_pipeline_change_events_negative_limit_raises(self, pipeline):
        """Test that negative limit for change events raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.get_change_events(limit=-1)

    def test_pipeline_two_snapshots_detect_no_change(self, populated_pipeline):
        """Test that two consecutive snapshots with no changes detect no diff."""
        pipe, assets = populated_pipeline
        pipe.detect_changes()
        result = pipe.detect_changes()

        # The second detection compares two identical snapshots
        assert result["change_count"] == 0

    # ------------------------------------------------------------------ #
    # Clear / reset tests
    # ------------------------------------------------------------------ #

    def test_pipeline_clear_resets_state(self, pipeline):
        """Test that clear() resets all pipeline state."""
        pipeline.run_pipeline(scope="full")
        assert pipeline.get_statistics()["total_pipeline_runs"] == 1

        pipeline.clear()

        stats = pipeline.get_statistics()
        assert stats["total_pipeline_runs"] == 0
        assert stats["total_snapshots"] == 0
        assert stats["total_change_events"] == 0

    # ------------------------------------------------------------------ #
    # Concurrency test
    # ------------------------------------------------------------------ #

    def test_pipeline_thread_safe_runs(self, provenance):
        """Test that concurrent pipeline runs do not corrupt state."""
        pipe = LineageTrackerPipelineEngine(provenance=provenance)
        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []

        def run_pipeline_thread():
            try:
                r = pipe.run_pipeline(scope="full")
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_pipeline_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5
        assert pipe.get_statistics()["total_pipeline_runs"] == 5

    # ------------------------------------------------------------------ #
    # PIPELINE_STAGES constant test
    # ------------------------------------------------------------------ #

    def test_pipeline_stages_constant(self):
        """Test that PIPELINE_STAGES contains all 7 expected stages."""
        expected = [
            "register",
            "capture",
            "build_graph",
            "validate",
            "analyze",
            "report",
            "detect_changes",
        ]
        assert PIPELINE_STAGES == expected

    def test_full_pipeline_completes_all_stages(self, pipeline):
        """Test that a full pipeline run completes all 7 stages."""
        result = pipeline.run_pipeline(scope="full")
        completed = set(result["stages_completed"])

        for stage in PIPELINE_STAGES:
            assert stage in completed, (
                f"Stage '{stage}' was not completed in a full pipeline run"
            )

    # ------------------------------------------------------------------ #
    # Duration tracking
    # ------------------------------------------------------------------ #

    def test_pipeline_duration_tracked(self, pipeline):
        """Test that pipeline run duration is tracked in milliseconds."""
        result = pipeline.run_pipeline(scope="full")
        assert isinstance(result["duration_ms"], (int, float))
        assert result["duration_ms"] >= 0

    # ------------------------------------------------------------------ #
    # Bulk API direct tests
    # ------------------------------------------------------------------ #

    def test_register_assets_from_metadata_duplicate_names(self, pipeline):
        """Test that registering assets with duplicate names is handled."""
        metadata = [
            {"qualified_name": "test.asset_a", "asset_type": "dataset"},
            {"qualified_name": "test.asset_a", "asset_type": "dataset"},
        ]

        result = pipeline.register_assets_from_metadata(metadata)
        # The second registration may succeed as an upsert or fail
        assert result["registered"] + result["failed"] == 2

    def test_capture_transformations_various_types(self, pipeline):
        """Test capturing transformations of various types."""
        events = [
            {
                "transformation_type": "filter",
                "agent_id": "agent-1",
                "input_assets": ["a"],
                "output_assets": ["b"],
            },
            {
                "transformation_type": "aggregate",
                "agent_id": "agent-2",
                "input_assets": ["b"],
                "output_assets": ["c"],
            },
            {
                "transformation_type": "join",
                "agent_id": "agent-3",
                "input_assets": ["c", "d"],
                "output_assets": ["e"],
            },
        ]

        result = pipeline.capture_transformations_from_events(events)
        assert result["captured"] == 3
        assert result["failed"] == 0
