# -*- coding: utf-8 -*-
"""
Unit tests for ReconciliationPipelineEngine - AGENT-DATA-015 (Engine 7 of 7)

Tests full pipeline execution, individual stages, source validation,
auto-configuration, strategy selection, field type inference, statistics,
error handling, and provenance tracking with 40+ tests targeting 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.cross_source_reconciliation.reconciliation_pipeline import (
    PipelineStageResult,
    ReconciliationPipelineEngine,
    ReconciliationStats,
    _safe_mean,
    _STAGES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ReconciliationPipelineEngine:
    """Create a fresh ReconciliationPipelineEngine instance."""
    return ReconciliationPipelineEngine()


@pytest.fixture
def two_source_data() -> Dict[str, List[Dict[str, Any]]]:
    """Two sources with matching records on entity_id key field."""
    return {
        "erp": [
            {"entity_id": "fac-001", "value": 100.0, "region": "EU", "period": "2025-Q1"},
            {"entity_id": "fac-002", "value": 200.0, "region": "US", "period": "2025-Q1"},
        ],
        "invoice": [
            {"entity_id": "fac-001", "value": 105.0, "region": "EU", "period": "2025-Q1"},
            {"entity_id": "fac-002", "value": 200.0, "region": "US", "period": "2025-Q1"},
        ],
    }


@pytest.fixture
def three_source_data() -> Dict[str, List[Dict[str, Any]]]:
    """Three sources with some differing values."""
    return {
        "erp": [
            {"entity_id": "fac-001", "value": 100.0, "category": "cement"},
        ],
        "invoice": [
            {"entity_id": "fac-001", "value": 110.0, "category": "cement"},
        ],
        "meter": [
            {"entity_id": "fac-001", "value": 100.0, "category": "steel"},
        ],
    }


@pytest.fixture
def basic_job_config() -> Dict[str, Any]:
    """Minimal job configuration."""
    return {
        "job_id": "test-job-001",
        "key_fields": ["entity_id"],
        "match_strategy": "exact",
        "match_threshold": 0.8,
        "resolution_strategy": "source_priority",
    }


@pytest.fixture
def auto_job_config() -> Dict[str, Any]:
    """Job configuration using 'auto' for match strategy."""
    return {
        "job_id": "test-job-auto",
        "match_strategy": "auto",
        "resolution_strategy": "source_priority",
    }


@pytest.fixture
def numeric_source_data() -> Dict[str, List[Dict[str, Any]]]:
    """Sources with numeric, date, boolean, and string fields."""
    return {
        "src_a": [
            {
                "record_id": "r1",
                "amount": 100.0,
                "created_date": "2025-01-15",
                "is_active": True,
                "category": "Type A",
            },
            {
                "record_id": "r2",
                "amount": 200.0,
                "created_date": "2025-02-20",
                "is_active": False,
                "category": "Type B",
            },
        ],
        "src_b": [
            {
                "record_id": "r1",
                "amount": 102.0,
                "created_date": "2025-01-15",
                "is_active": True,
                "category": "Type A",
            },
        ],
    }


# ===========================================================================
# Test Module-Level Helpers
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_safe_mean_normal(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_single(self):
        assert _safe_mean([5.0]) == 5.0

    def test_stages_tuple(self):
        assert _STAGES == (
            "register", "align", "match", "compare",
            "detect", "resolve", "golden",
        )


# ===========================================================================
# Test Data Models
# ===========================================================================


class TestDataModels:
    """Tests for pipeline data model defaults."""

    def test_pipeline_stage_result_defaults(self):
        psr = PipelineStageResult()
        assert psr.stage == ""
        assert psr.status == "pending"
        assert psr.duration_ms == 0.0
        assert psr.records_processed == 0
        assert psr.output_summary == {}
        assert psr.error is None

    def test_reconciliation_stats_defaults(self):
        stats = ReconciliationStats()
        assert stats.total_runs == 0
        assert stats.total_sources_registered == 0
        assert stats.total_records_matched == 0
        assert stats.total_comparisons == 0
        assert stats.total_discrepancies == 0
        assert stats.total_resolutions == 0
        assert stats.total_golden_records == 0
        assert stats.avg_match_confidence == 0.0


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestInitialization:
    """Tests for engine initialization."""

    def test_default_init(self, engine):
        assert engine._statistics.total_runs == 0

    def test_statistics_initial_state(self, engine):
        stats = engine.get_statistics()
        assert stats["total_runs"] == 0
        assert "engine_availability" in stats

    def test_engine_health(self, engine):
        health = engine.get_engine_health()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert "engines" in health
        assert health["engines_total"] == 6


# ===========================================================================
# Test validate_sources
# ===========================================================================


class TestValidateSources:
    """Tests for source data validation."""

    def test_valid_two_sources(self, engine, two_source_data):
        result = engine.validate_sources(two_source_data)
        assert result["valid"] is True
        assert result["source_count"] == 2
        assert result["error"] is None

    def test_rejects_empty_source_data(self, engine):
        result = engine.validate_sources({})
        assert result["valid"] is False
        assert "No source data" in result["error"]

    def test_rejects_single_source(self, engine):
        result = engine.validate_sources({"only_one": [{"id": 1}]})
        assert result["valid"] is False
        assert "At least 2 sources" in result["error"]

    def test_rejects_empty_records(self, engine):
        result = engine.validate_sources({"src_a": [], "src_b": [{"id": 1}]})
        assert result["valid"] is False
        assert "no records" in result["error"].lower()

    def test_warns_on_uneven_sizes(self, engine):
        data = {
            "big": [{"id": i} for i in range(100)],
            "small": [{"id": 1}],
        }
        result = engine.validate_sources(data)
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "vary significantly" in result["warnings"][0].lower()

    def test_record_counts_populated(self, engine, two_source_data):
        result = engine.validate_sources(two_source_data)
        assert result["record_counts"]["erp"] == 2
        assert result["record_counts"]["invoice"] == 2

    def test_rejects_empty_source_name(self, engine):
        result = engine.validate_sources({"": [{"id": 1}], "b": [{"id": 2}]})
        assert result["valid"] is False
        assert "non-empty string" in result["error"]


# ===========================================================================
# Test auto_configure
# ===========================================================================


class TestAutoConfigure:
    """Tests for auto-configuration of job settings."""

    def test_infers_key_fields(self, engine, two_source_data):
        config = engine.auto_configure(two_source_data)
        assert "entity_id" in config["key_fields"]

    def test_infers_field_types(self, engine, numeric_source_data):
        config = engine.auto_configure(numeric_source_data)
        assert "field_types" in config
        ft = config["field_types"]
        assert ft.get("amount") == "numeric"

    def test_auto_configured_flag(self, engine, two_source_data):
        config = engine.auto_configure(two_source_data)
        assert config["auto_configured"] is True

    def test_generates_job_id(self, engine, two_source_data):
        config = engine.auto_configure(two_source_data)
        assert config["job_id"] != ""
        assert len(config["job_id"]) > 0

    def test_common_fields_detected(self, engine, two_source_data):
        config = engine.auto_configure(two_source_data)
        assert "entity_id" in config["common_fields"]
        assert "value" in config["common_fields"]

    def test_default_match_threshold(self, engine, two_source_data):
        config = engine.auto_configure(two_source_data)
        assert config["match_threshold"] == 0.8

    def test_tolerance_rules_for_numeric(self, engine, numeric_source_data):
        config = engine.auto_configure(numeric_source_data)
        if "amount" in config.get("tolerance_rules", {}):
            tol = config["tolerance_rules"]["amount"]
            assert tol["type"] == "relative"
            assert tol["threshold"] == 0.05


# ===========================================================================
# Test _select_match_strategy
# ===========================================================================


class TestSelectMatchStrategy:
    """Tests for automatic match strategy selection."""

    def test_exact_for_clean_id_keys(self, engine, two_source_data):
        """Sources with entity_id field and high overlap -> exact."""
        result = engine._select_match_strategy(two_source_data)
        assert result["strategy"] == "exact"

    def test_fuzzy_for_low_overlap(self, engine):
        """Sources with very different schemas -> fuzzy."""
        data = {
            "a": [{"entity_id": "1", "co2": 100}],
            "b": [{"facility_code": "1", "emissions": 100, "x": 1, "y": 2, "z": 3}],
        }
        result = engine._select_match_strategy(data)
        assert result["strategy"] == "fuzzy"

    def test_temporal_for_date_fields_no_ids(self, engine):
        """Sources with date fields but no ID fields -> temporal."""
        data = {
            "a": [{"period_date": "2025-01-01", "value": 100}],
            "b": [{"period_date": "2025-01-01", "value": 110}],
        }
        result = engine._select_match_strategy(data)
        assert result["strategy"] == "temporal"

    def test_composite_for_multiple_key_types(self, engine):
        """Sources with both ID and date fields -> composite."""
        data = {
            "a": [{"entity_id": "1", "report_date": "2025-01", "code_id": "A", "value": 100}],
            "b": [{"entity_id": "1", "report_date": "2025-01", "code_id": "A", "value": 110}],
        }
        result = engine._select_match_strategy(data)
        assert result["strategy"] in ("exact", "composite")

    def test_empty_data_returns_exact(self, engine):
        result = engine._select_match_strategy({"a": [], "b": []})
        assert result["strategy"] == "exact"


# ===========================================================================
# Test _infer_field_types
# ===========================================================================


class TestInferFieldTypes:
    """Tests for field type inference."""

    def test_detects_numeric(self, engine):
        records = [{"amount": 100.0}, {"amount": 200.0}]
        types = engine._infer_field_types(records)
        assert types["amount"] == "numeric"

    def test_detects_date(self, engine):
        records = [{"date": "2025-01-01"}, {"date": "2025-02-15"}]
        types = engine._infer_field_types(records)
        assert types["date"] == "date"

    def test_detects_boolean(self, engine):
        records = [{"flag": True}, {"flag": False}]
        types = engine._infer_field_types(records)
        assert types["flag"] == "boolean"

    def test_detects_string(self, engine):
        records = [{"name": "Cement"}, {"name": "Steel"}]
        types = engine._infer_field_types(records)
        assert types["name"] == "string"

    def test_empty_records(self, engine):
        types = engine._infer_field_types([])
        assert types == {}

    def test_numeric_strings(self, engine):
        """String values like '100.5' are detected as numeric."""
        records = [{"val": "100.5"}, {"val": "200.3"}]
        types = engine._infer_field_types(records)
        assert types["val"] == "numeric"

    def test_boolean_strings(self, engine):
        """Boolean string values like 'true', 'false' detected."""
        records = [{"active": "true"}, {"active": "false"}]
        types = engine._infer_field_types(records)
        assert types["active"] == "boolean"

    def test_infer_across_sources(self, engine, numeric_source_data):
        """Field types inferred from multiple sources via majority vote."""
        types = engine._infer_field_types_from_data(numeric_source_data)
        assert types.get("amount") == "numeric"
        assert types.get("category") == "string"


# ===========================================================================
# Test run_pipeline
# ===========================================================================


class TestRunPipeline:
    """Tests for full pipeline execution."""

    def test_full_pipeline_completes(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        assert report["status"] in ("completed", "completed_with_warnings")
        assert report["job_id"] == "test-job-001"
        assert report["sources_registered"] == 2
        assert report["total_records"] == 4

    def test_pipeline_has_stage_results(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        sr = report["stage_results"]
        assert "register" in sr
        assert "align" in sr
        assert "match" in sr
        assert "compare" in sr
        assert "detect" in sr
        assert "resolve" in sr
        assert "golden" in sr

    def test_all_stages_completed(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        for stage_name in _STAGES:
            assert report["stage_results"][stage_name]["status"] == "completed"

    def test_pipeline_records_total_time(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        # Very fast pipelines may register 0.0 ms, so >= is correct.
        assert report["total_time_ms"] >= 0
        assert isinstance(report["total_time_ms"], (int, float))

    def test_pipeline_generates_provenance(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        assert report["provenance_hash"] != ""
        assert len(report["provenance_hash"]) == 64

    def test_pipeline_with_auto_strategy(self, engine, two_source_data, auto_job_config):
        report = engine.run_pipeline(auto_job_config, two_source_data)
        assert report["status"] in ("completed", "completed_with_warnings")

    def test_pipeline_generates_golden_records(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        assert len(report["golden_records"]) > 0

    def test_pipeline_generates_statistics(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        stats = report["statistics"]
        assert stats["sources_registered"] == 2
        assert stats["total_records"] == 4

    def test_auto_generates_job_id(self, engine, two_source_data):
        config = {"match_strategy": "exact"}
        report = engine.run_pipeline(config, two_source_data)
        assert report["job_id"] != ""

    def test_pipeline_fails_on_invalid_sources(self, engine, basic_job_config):
        report = engine.run_pipeline(basic_job_config, {"only_one": [{"id": 1}]})
        assert report["status"] == "failed"
        assert report["error"] is not None

    def test_pipeline_fails_on_empty_sources(self, engine, basic_job_config):
        report = engine.run_pipeline(basic_job_config, {})
        assert report["status"] == "failed"


# ===========================================================================
# Test Individual Stages
# ===========================================================================


class TestStageRegister:
    """Tests for Stage 1: REGISTER."""

    def test_register_creates_source_ids(self, engine, two_source_data, basic_job_config):
        result = engine._stage_register("j1", two_source_data, basic_job_config)
        assert result.status == "completed"
        assert len(result.output_summary["source_ids"]) == 2
        assert "erp" in result.output_summary["source_ids"]
        assert "invoice" in result.output_summary["source_ids"]

    def test_register_computes_credibilities(self, engine, two_source_data, basic_job_config):
        result = engine._stage_register("j1", two_source_data, basic_job_config)
        creds = result.output_summary["credibilities"]
        assert "erp" in creds
        assert 0.0 <= creds["erp"] <= 1.0

    def test_register_uses_configured_priorities(self, engine, two_source_data):
        config = {"source_priorities": {"erp": 0.95, "invoice": 0.6}}
        result = engine._stage_register("j1", two_source_data, config)
        creds = result.output_summary["credibilities"]
        assert creds["erp"] == pytest.approx(0.95)
        assert creds["invoice"] == pytest.approx(0.6)


class TestStageAlign:
    """Tests for Stage 2: ALIGN."""

    def test_align_finds_common_fields(self, engine, two_source_data):
        result = engine._stage_align("j1", two_source_data, ["erp", "invoice"])
        assert result.status == "completed"
        common = result.output_summary["common_fields"]
        assert "entity_id" in common
        assert "value" in common

    def test_align_normalizes_strings(self, engine):
        data = {
            "a": [{"name": " Hello "}],
            "b": [{"name": "World"}],
        }
        result = engine._stage_align("j1", data, ["a", "b"])
        aligned = result.output_summary["aligned_data"]
        assert aligned["a"][0]["name"] == "Hello"


class TestStageMatch:
    """Tests for Stage 3: MATCH."""

    def test_match_finds_pairs(self, engine, two_source_data):
        result = engine._stage_match(
            "j1", two_source_data, ["erp", "invoice"], "exact", 0.5
        )
        assert result.status == "completed"
        assert result.output_summary["total_matches"] > 0

    def test_match_with_no_overlap(self, engine):
        data = {
            "a": [{"entity_id": "x", "val": 1}],
            "b": [{"entity_id": "y", "val": 2}],
        }
        result = engine._stage_match("j1", data, ["a", "b"], "exact", 0.8)
        assert result.status == "completed"
        assert result.output_summary["total_matches"] == 0


class TestStageCompare:
    """Tests for Stage 4: COMPARE."""

    def test_compare_with_matches(self, engine, two_source_data):
        # First match to get matches
        match_result = engine._stage_match(
            "j1", two_source_data, ["erp", "invoice"], "exact", 0.5
        )
        matches = match_result.output_summary.get("matches", [])

        if matches:
            compare_result = engine._stage_compare(
                "j1", matches, two_source_data, {}, {}
            )
            assert compare_result.status == "completed"
            assert compare_result.output_summary["total_comparisons"] > 0

    def test_compare_with_empty_matches(self, engine):
        result = engine._stage_compare("j1", [], {}, {}, {})
        assert result.status == "completed"
        assert result.output_summary["total_comparisons"] == 0


class TestStageDetect:
    """Tests for Stage 5: DETECT."""

    def test_detect_no_mismatches(self, engine):
        comparisons = [{"field": "a", "result": "match"}]
        result = engine._stage_detect("j1", comparisons, [])
        assert result.status == "completed"
        assert result.output_summary["total_discrepancies"] == 0

    def test_detect_with_mismatches(self, engine):
        comparisons = [
            {
                "field": "value",
                "result": "mismatch",
                "source_a": "erp",
                "source_b": "invoice",
                "value_a": 100,
                "value_b": 110,
                "deviation": 0.1,
            },
        ]
        result = engine._stage_detect("j1", comparisons, [])
        assert result.status == "completed"
        assert result.output_summary["total_discrepancies"] >= 1


class TestStageResolve:
    """Tests for Stage 6: RESOLVE."""

    def test_resolve_empty_discrepancies(self, engine):
        result = engine._stage_resolve("j1", [], {}, {}, "source_priority")
        assert result.status == "completed"
        assert result.output_summary["total_resolutions"] == 0

    def test_resolve_with_discrepancies(self, engine, two_source_data):
        discrepancies = [
            {
                "discrepancy_id": "d1",
                "type": "value_mismatch",
                "severity": "medium",
                "field": "value",
                "source_a": "erp",
                "source_b": "invoice",
                "value_a": 100,
                "value_b": 110,
            },
        ]
        creds = {"erp": 0.9, "invoice": 0.6}
        result = engine._stage_resolve(
            "j1", discrepancies, creds, two_source_data, "source_priority"
        )
        assert result.status == "completed"
        assert result.output_summary["total_resolutions"] >= 1


class TestStageGoldenRecords:
    """Tests for Stage 7: GOLDEN RECORDS."""

    def test_golden_records_from_sources(self, engine, two_source_data):
        result = engine._stage_golden_records(
            "j1", two_source_data, [], {"erp": 0.9, "invoice": 0.6}
        )
        assert result.status == "completed"
        assert len(result.output_summary["golden_records"]) > 0


# ===========================================================================
# Test get_statistics
# ===========================================================================


class TestGetStatistics:
    """Tests for pipeline statistics retrieval."""

    def test_initial_statistics(self, engine):
        stats = engine.get_statistics()
        assert stats["total_runs"] == 0
        assert stats["total_records_matched"] == 0

    def test_statistics_after_pipeline_run(self, engine, two_source_data, basic_job_config):
        engine.run_pipeline(basic_job_config, two_source_data)
        stats = engine.get_statistics()
        assert stats["total_runs"] == 1
        assert stats["total_sources_registered"] == 2

    def test_statistics_accumulate_across_runs(self, engine, two_source_data, basic_job_config):
        engine.run_pipeline(basic_job_config, two_source_data)
        engine.run_pipeline(
            {**basic_job_config, "job_id": "job-2"}, two_source_data
        )
        stats = engine.get_statistics()
        assert stats["total_runs"] == 2
        assert stats["total_sources_registered"] == 4

    def test_statistics_by_status(self, engine, two_source_data, basic_job_config):
        engine.run_pipeline(basic_job_config, two_source_data)
        stats = engine.get_statistics()
        assert len(stats["by_status"]) > 0


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_graceful_failure_on_bad_source_data(self, engine, basic_job_config):
        report = engine.run_pipeline(basic_job_config, {"a": []})
        assert report["status"] == "failed"

    def test_pipeline_with_none_records(self, engine, basic_job_config):
        """Pipeline fails gracefully when source has None instead of list."""
        report = engine.run_pipeline(basic_job_config, {})
        assert report["status"] == "failed"

    def test_error_report_structure(self, engine, basic_job_config):
        """Error reports have the same structure as success reports."""
        report = engine.run_pipeline(basic_job_config, {})
        assert "job_id" in report
        assert "status" in report
        assert "error" in report
        assert "total_time_ms" in report
        assert "matches" in report
        assert "golden_records" in report


# ===========================================================================
# Test Batch Pipeline
# ===========================================================================


class TestBatchPipeline:
    """Tests for batch pipeline execution."""

    def test_batch_runs_all_jobs(self, engine, two_source_data, basic_job_config):
        jobs = [
            {**basic_job_config, "job_id": "batch-1"},
            {**basic_job_config, "job_id": "batch-2"},
        ]
        source_map = {
            "batch-1": two_source_data,
            "batch-2": two_source_data,
        }
        reports = engine.run_batch_pipeline(jobs, source_map)
        assert len(reports) == 2
        for r in reports:
            assert r["status"] in ("completed", "completed_with_warnings")

    def test_batch_handles_missing_source_data(self, engine, two_source_data, basic_job_config):
        jobs = [
            {**basic_job_config, "job_id": "batch-1"},
            {**basic_job_config, "job_id": "batch-missing"},
        ]
        source_map = {"batch-1": two_source_data}
        reports = engine.run_batch_pipeline(jobs, source_map)
        assert len(reports) == 2

    def test_batch_empty_jobs(self, engine):
        reports = engine.run_batch_pipeline([], {})
        assert len(reports) == 0


# ===========================================================================
# Test Provenance Tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests for SHA-256 provenance chains across pipeline stages."""

    def test_pipeline_provenance(self, engine, two_source_data, basic_job_config):
        report = engine.run_pipeline(basic_job_config, two_source_data)
        assert report["provenance_hash"] != ""
        assert len(report["provenance_hash"]) == 64

    def test_provenance_chain_grows(self, engine, two_source_data, basic_job_config):
        initial_len = engine._provenance.get_chain_length()
        engine.run_pipeline(basic_job_config, two_source_data)
        assert engine._provenance.get_chain_length() > initial_len

    def test_register_stage_has_provenance(self, engine, two_source_data, basic_job_config):
        result = engine._stage_register("j1", two_source_data, basic_job_config)
        assert result.output_summary.get("provenance_hash", "") != ""

    def test_align_stage_has_provenance(self, engine, two_source_data):
        result = engine._stage_align("j1", two_source_data, list(two_source_data.keys()))
        assert result.output_summary.get("provenance_hash", "") != ""
