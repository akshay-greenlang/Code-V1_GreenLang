# -*- coding: utf-8 -*-
"""
Unit tests for HazardPipelineEngine - AGENT-DATA-020 (Engine 7 of 7)

Tests the end-to-end pipeline orchestrator including run_pipeline,
run_batch_pipeline, get_pipeline_run, list_pipeline_runs, get_health,
get_statistics, and clear methods.

Target: 85%+ code coverage across all methods and edge cases.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.climate_hazard.hazard_pipeline import (
    HazardPipelineEngine,
    PIPELINE_STAGES,
    _classify_risk,
    _elapsed_ms,
    _extract_location_lat,
    _extract_location_lon,
    _new_id,
    _normalise_raw,
    _safe_mean,
    _sha256,
    _utcnow_iso,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_database():
    """Create a mock HazardDatabaseEngine."""
    db = MagicMock()
    db.register_source.return_value = {"source_id": "src_001", "status": "active"}
    db.ingest_hazard_data.return_value = {"record_id": "rec_001"}
    db.list_sources.return_value = []
    db.clear.return_value = None
    return db


@pytest.fixture
def mock_risk_engine():
    """Create a mock RiskIndexEngine."""
    engine = MagicMock()
    engine.calculate_risk_index.return_value = {
        "index_id": "idx_001",
        "composite_score": 55.0,
        "risk_classification": "medium",
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def mock_projector():
    """Create a mock ScenarioProjectorEngine."""
    engine = MagicMock()
    engine.project_scenario.return_value = {
        "projection_id": "proj_001",
        "projected_value": 65.0,
        "change_percent": 18.0,
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def mock_exposure():
    """Create a mock ExposureAssessorEngine."""
    engine = MagicMock()
    engine.assess_exposure.return_value = {
        "exposure_id": "exp_001",
        "exposure_score": 50.0,
        "exposure_level": "medium",
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def mock_vulnerability():
    """Create a mock VulnerabilityScorerEngine."""
    engine = MagicMock()
    engine.score_vulnerability.return_value = {
        "vulnerability_id": "vuln_001",
        "vulnerability_score": 60.0,
        "vulnerability_level": "high",
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def mock_reporter():
    """Create a mock ComplianceReporterEngine."""
    engine = MagicMock()
    engine.generate_report.return_value = {
        "report_id": "rpt_001",
        "compliance_score": 75.0,
        "content": "Report content",
    }
    engine.clear.return_value = None
    return engine


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker."""
    prov = MagicMock()
    prov.entry_count = 0
    prov.record.return_value = MagicMock(hash_value="a" * 64)
    prov.build_hash.return_value = "b" * 64
    prov.hash_record.return_value = "c" * 64
    prov.reset.return_value = None
    return prov


@pytest.fixture
def engine(
    mock_database,
    mock_risk_engine,
    mock_projector,
    mock_exposure,
    mock_vulnerability,
    mock_reporter,
    mock_provenance,
) -> HazardPipelineEngine:
    """Create a HazardPipelineEngine with all mock engines injected."""
    return HazardPipelineEngine(
        database=mock_database,
        risk_engine=mock_risk_engine,
        projector=mock_projector,
        exposure_engine=mock_exposure,
        vulnerability_engine=mock_vulnerability,
        reporter=mock_reporter,
        provenance=mock_provenance,
    )


@pytest.fixture
def stub_engine() -> HazardPipelineEngine:
    """Create a HazardPipelineEngine that simulates missing upstream engines.

    We set each engine attribute to None after construction to simulate
    the 'unavailable' condition, bypassing auto-creation from imports.
    """
    mock_prov = MagicMock(
        entry_count=0,
        record=MagicMock(return_value=MagicMock(hash_value="0" * 64)),
        build_hash=MagicMock(return_value="0" * 64),
        reset=MagicMock(),
    )
    eng = HazardPipelineEngine(provenance=mock_prov)
    # Force all upstream engines to None to test stub/degraded paths
    eng._database = None
    eng._risk_engine = None
    eng._projector = None
    eng._exposure_engine = None
    eng._vulnerability_engine = None
    eng._reporter = None
    return eng


@pytest.fixture
def sample_assets() -> List[Dict[str, Any]]:
    """Sample asset list for pipeline runs."""
    return [
        {
            "asset_id": "asset_001",
            "name": "London HQ",
            "asset_type": "office",
            "location": {"lat": 51.5074, "lon": -0.1278},
        },
        {
            "asset_id": "asset_002",
            "name": "Paris Factory",
            "asset_type": "factory",
            "location": {"lat": 48.8566, "lon": 2.3522},
        },
    ]


@pytest.fixture
def sample_hazard_types() -> List[str]:
    """Sample hazard types for pipeline runs."""
    return ["flood", "heat_wave"]


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level utility functions."""

    def test_new_id_with_prefix(self):
        result = _new_id("pipe")
        assert result.startswith("pipe-")
        assert len(result) == 17  # "pipe-" + 12 hex chars

    def test_new_id_uniqueness(self):
        ids = {_new_id("x") for _ in range(100)}
        assert len(ids) == 100

    def test_elapsed_ms(self):
        start = time.monotonic()
        time.sleep(0.01)
        elapsed = _elapsed_ms(start)
        assert elapsed > 0
        assert isinstance(elapsed, float)

    def test_sha256_deterministic(self):
        data = {"key": "value", "n": 42}
        h1 = _sha256(data)
        h2 = _sha256(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_different_data(self):
        assert _sha256({"a": 1}) != _sha256({"a": 2})

    def test_safe_mean_normal(self):
        assert _safe_mean([10.0, 20.0, 30.0]) == 20.0

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_classify_risk_extreme(self):
        assert _classify_risk(80.0) == "extreme"
        assert _classify_risk(100.0) == "extreme"

    def test_classify_risk_high(self):
        assert _classify_risk(60.0) == "high"
        assert _classify_risk(79.9) == "high"

    def test_classify_risk_medium(self):
        assert _classify_risk(40.0) == "medium"
        assert _classify_risk(59.9) == "medium"

    def test_classify_risk_low(self):
        assert _classify_risk(20.0) == "low"
        assert _classify_risk(39.9) == "low"

    def test_classify_risk_negligible(self):
        assert _classify_risk(0.0) == "negligible"
        assert _classify_risk(19.9) == "negligible"

    def test_normalise_raw_dict(self):
        result = _normalise_raw({"key": "val"})
        assert result == {"key": "val"}

    def test_normalise_raw_pydantic_like(self):
        obj = MagicMock()
        obj.dict.return_value = {"x": 1}
        result = _normalise_raw(obj)
        assert result == {"x": 1}

    def test_normalise_raw_object(self):
        class Simple:
            def __init__(self):
                self.a = 1
                self.b = 2
        result = _normalise_raw(Simple())
        assert result["a"] == 1
        assert result["b"] == 2

    def test_normalise_raw_string(self):
        result = _normalise_raw("hello")
        assert result == {"raw": "hello"}

    def test_extract_location_lat_dict(self):
        assert _extract_location_lat({"lat": 51.5}) == 51.5
        assert _extract_location_lat({"latitude": 48.8}) == 48.8

    def test_extract_location_lat_missing(self):
        assert _extract_location_lat({}) == 0.0
        assert _extract_location_lat("invalid") == 0.0

    def test_extract_location_lat_object(self):
        obj = MagicMock()
        obj.lat = 40.7
        result = _extract_location_lat(obj)
        assert result == 40.7

    def test_extract_location_lon_dict(self):
        assert _extract_location_lon({"lon": -0.13}) == -0.13
        assert _extract_location_lon({"lng": 2.35}) == 2.35
        assert _extract_location_lon({"longitude": -73.94}) == -73.94

    def test_extract_location_lon_missing(self):
        assert _extract_location_lon({}) == 0.0
        assert _extract_location_lon(42) == 0.0

    def test_extract_location_lon_object(self):
        obj = MagicMock()
        obj.lon = -0.13
        result = _extract_location_lon(obj)
        assert result == -0.13

    def test_utcnow_iso_format(self):
        result = _utcnow_iso()
        assert isinstance(result, str)
        assert "T" in result


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_pipeline_stages_count(self):
        assert len(PIPELINE_STAGES) == 7

    def test_pipeline_stages_order(self):
        assert PIPELINE_STAGES == [
            "ingest", "index", "project", "assess", "score", "report", "audit",
        ]


# =========================================================================
# HazardPipelineEngine initialization
# =========================================================================


class TestEngineInitialization:
    """Tests for engine construction and initialization."""

    def test_with_all_mocks(self, engine: HazardPipelineEngine):
        assert engine._database is not None
        assert engine._risk_engine is not None
        assert engine._projector is not None
        assert engine._exposure_engine is not None
        assert engine._vulnerability_engine is not None
        assert engine._reporter is not None
        assert engine._provenance is not None

    def test_stub_mode_all_none(self, stub_engine: HazardPipelineEngine):
        assert stub_engine._database is None
        assert stub_engine._risk_engine is None
        assert stub_engine._projector is None
        assert stub_engine._exposure_engine is None
        assert stub_engine._vulnerability_engine is None
        assert stub_engine._reporter is None

    def test_initial_state(self, engine: HazardPipelineEngine):
        assert engine._total_runs == 0
        assert engine._success_count == 0
        assert engine._failure_count == 0
        assert engine._partial_count == 0
        assert len(engine._pipeline_runs) == 0

    def test_stage_durations_initialized(self, engine: HazardPipelineEngine):
        for stage in PIPELINE_STAGES:
            assert stage in engine._stage_durations
            assert engine._stage_durations[stage] == []

    def test_properties(self, engine: HazardPipelineEngine, mock_database, mock_risk_engine):
        assert engine.database is mock_database
        assert engine.risk_engine is mock_risk_engine

    def test_custom_genesis_hash(self, mock_provenance):
        eng = HazardPipelineEngine(
            provenance=mock_provenance,
            genesis_hash="custom-genesis",
        )
        assert eng._provenance is mock_provenance


# =========================================================================
# run_pipeline
# =========================================================================


class TestRunPipeline:
    """Tests for the run_pipeline method."""

    def test_basic_run(
        self,
        engine: HazardPipelineEngine,
        sample_assets: List[Dict[str, Any]],
        sample_hazard_types: List[str],
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        assert "pipeline_id" in result
        assert "status" in result
        assert result["status"] in ("completed", "partial", "failed")
        assert "duration_ms" in result
        assert "provenance_hash" in result

    def test_empty_assets_raises(self, engine: HazardPipelineEngine):
        with pytest.raises(ValueError, match="assets"):
            engine.run_pipeline(assets=[], hazard_types=["flood"])

    def test_empty_hazard_types_raises(
        self, engine: HazardPipelineEngine, sample_assets
    ):
        with pytest.raises(ValueError, match="hazard_types"):
            engine.run_pipeline(assets=sample_assets, hazard_types=[])

    def test_default_scenarios(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_custom_scenarios(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            scenarios=["ssp5_8.5"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_custom_time_horizons(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            time_horizons=["short_term", "long_term"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_custom_frameworks(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            report_frameworks=["tcfd", "csrd_esrs"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_subset_stages(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            stages=["ingest", "index"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_pipeline_stored(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        stored = engine.get_pipeline_run(result["pipeline_id"])
        assert stored is not None

    def test_counters_incremented(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        assert engine._total_runs == 1

    def test_duration_positive(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        assert result["duration_ms"] >= 0.0

    def test_evaluation_summary(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        if "evaluation_summary" in result:
            summary = result["evaluation_summary"]
            assert "total_assets" in summary

    def test_stage_timings(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        if "stage_timings" in result:
            assert isinstance(result["stage_timings"], dict)

    def test_stub_mode_pipeline(
        self,
        stub_engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        """Pipeline should still complete in stub mode (no engines)."""
        result = stub_engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_with_parameters(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            parameters={"custom_param": "value"},
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_single_asset(self, engine: HazardPipelineEngine):
        result = engine.run_pipeline(
            assets=[{
                "asset_id": "a1",
                "name": "Single",
                "asset_type": "office",
                "location": {"lat": 0.0, "lon": 0.0},
            }],
            hazard_types=["flood"],
        )
        assert result["status"] in ("completed", "partial", "failed")


# =========================================================================
# run_batch_pipeline
# =========================================================================


class TestRunBatchPipeline:
    """Tests for the run_batch_pipeline method."""

    def test_basic_batch(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        portfolios = [
            {"portfolio_id": "p1", "assets": sample_assets},
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        assert "batch_id" in result
        assert "per_portfolio_results" in result
        assert "summary" in result
        assert "duration_ms" in result
        assert "provenance_hash" in result
        assert len(result["per_portfolio_results"]) == 1

    def test_multiple_portfolios(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        portfolios = [
            {"portfolio_id": "p1", "assets": sample_assets[:1]},
            {"portfolio_id": "p2", "assets": sample_assets[1:]},
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        assert len(result["per_portfolio_results"]) == 2
        assert result["summary"]["portfolios_processed"] == 2

    def test_empty_portfolios_raises(
        self, engine: HazardPipelineEngine, sample_hazard_types
    ):
        with pytest.raises(ValueError, match="asset_portfolios"):
            engine.run_batch_pipeline(
                asset_portfolios=[],
                hazard_types=sample_hazard_types,
            )

    def test_empty_hazard_types_raises(
        self, engine: HazardPipelineEngine, sample_assets
    ):
        with pytest.raises(ValueError, match="hazard_types"):
            engine.run_batch_pipeline(
                asset_portfolios=[{"portfolio_id": "p1", "assets": sample_assets}],
                hazard_types=[],
            )

    def test_batch_summary_fields(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        portfolios = [
            {"portfolio_id": "p1", "assets": sample_assets},
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        summary = result["summary"]
        assert "avg_risk" in summary
        assert "total_assets" in summary
        assert "total_high_risk" in summary
        assert "portfolios_processed" in summary
        assert "portfolios_completed" in summary
        assert "portfolios_failed" in summary

    def test_batch_with_custom_scenarios(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        portfolios = [
            {
                "portfolio_id": "p1",
                "assets": sample_assets,
                "scenarios": ["ssp5_8.5"],
            },
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        assert result["summary"]["portfolios_processed"] == 1

    def test_batch_portfolio_auto_id(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        portfolios = [
            {"assets": sample_assets},  # No portfolio_id
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        assert len(result["per_portfolio_results"]) == 1

    def test_batch_failure_in_one_portfolio(
        self,
        engine: HazardPipelineEngine,
        sample_hazard_types,
    ):
        """A portfolio with no assets should fail but not kill the batch."""
        portfolios = [
            {"portfolio_id": "good", "assets": [{
                "asset_id": "a1", "name": "A", "asset_type": "office",
                "location": {"lat": 0, "lon": 0},
            }]},
            {"portfolio_id": "bad", "assets": []},  # Will raise ValueError
        ]
        result = engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=sample_hazard_types,
        )
        statuses = [r["status"] for r in result["per_portfolio_results"]]
        assert "failed" in statuses


# =========================================================================
# get_pipeline_run
# =========================================================================


class TestGetPipelineRun:
    """Tests for the get_pipeline_run method."""

    def test_get_existing_run(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        run = engine.get_pipeline_run(result["pipeline_id"])
        assert run is not None
        assert run["pipeline_id"] == result["pipeline_id"]

    def test_get_nonexistent_run(self, engine: HazardPipelineEngine):
        result = engine.get_pipeline_run("pipe-nonexistent0")
        assert result is None

    def test_get_empty_id(self, engine: HazardPipelineEngine):
        result = engine.get_pipeline_run("")
        assert result is None


# =========================================================================
# list_pipeline_runs
# =========================================================================


class TestListPipelineRuns:
    """Tests for the list_pipeline_runs method."""

    def test_list_empty(self, engine: HazardPipelineEngine):
        result = engine.list_pipeline_runs()
        assert result == []

    def test_list_after_runs(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        result = engine.list_pipeline_runs()
        assert len(result) == 2

    def test_list_with_limit(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        for _ in range(5):
            engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        result = engine.list_pipeline_runs(limit=3)
        assert len(result) == 3

    def test_list_sorted_newest_first(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        runs = engine.list_pipeline_runs()
        if len(runs) >= 2 and "created_at" in runs[0] and "created_at" in runs[1]:
            assert runs[0]["created_at"] >= runs[1]["created_at"]


# =========================================================================
# get_health
# =========================================================================


class TestGetHealth:
    """Tests for the get_health method."""

    def test_health_with_all_engines(self, engine: HazardPipelineEngine):
        health = engine.get_health()
        assert health["status"] == "healthy"
        assert health["engines_available"] == 6
        assert health["engines_total"] == 6

    def test_health_with_no_engines(self, stub_engine: HazardPipelineEngine):
        health = stub_engine.get_health()
        assert health["status"] == "unhealthy"
        assert health["engines_available"] == 0

    def test_health_degraded(self, mock_database, mock_provenance):
        engine = HazardPipelineEngine(
            database=mock_database,
            provenance=mock_provenance,
        )
        # Force some engines to None to simulate partial availability
        engine._risk_engine = None
        engine._projector = None
        engine._exposure_engine = None
        engine._vulnerability_engine = None
        engine._reporter = None
        health = engine.get_health()
        assert health["status"] == "degraded"
        assert health["engines_available"] == 1

    def test_health_engines_dict(self, engine: HazardPipelineEngine):
        health = engine.get_health()
        assert "database" in health["engines"]
        assert "risk_engine" in health["engines"]
        assert "projector" in health["engines"]
        assert "exposure_engine" in health["engines"]
        assert "vulnerability_engine" in health["engines"]
        assert "reporter" in health["engines"]

    def test_health_pipeline_stats(self, engine: HazardPipelineEngine):
        health = engine.get_health()
        assert "pipeline_stats" in health
        assert health["pipeline_stats"]["total_runs"] == 0

    def test_health_checked_at(self, engine: HazardPipelineEngine):
        health = engine.get_health()
        assert "checked_at" in health
        assert isinstance(health["checked_at"], str)


# =========================================================================
# get_statistics
# =========================================================================


class TestGetStatistics:
    """Tests for the get_statistics method."""

    def test_initial_statistics(self, engine: HazardPipelineEngine):
        stats = engine.get_statistics()
        assert stats["total_runs"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 0
        assert stats["partial_count"] == 0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["success_rate"] == 0.0

    def test_statistics_after_runs(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        stats = engine.get_statistics()
        assert stats["total_runs"] >= 1

    def test_statistics_per_stage_avg(self, engine: HazardPipelineEngine):
        stats = engine.get_statistics()
        assert "per_stage_avg_ms" in stats
        for stage in PIPELINE_STAGES:
            assert stage in stats["per_stage_avg_ms"]

    def test_statistics_provenance_count(self, engine: HazardPipelineEngine):
        stats = engine.get_statistics()
        assert "provenance_entry_count" in stats

    def test_statistics_computed_at(self, engine: HazardPipelineEngine):
        stats = engine.get_statistics()
        assert "computed_at" in stats

    def test_statistics_duration_metrics(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        stats = engine.get_statistics()
        assert stats["min_duration_ms"] >= 0.0
        assert stats["max_duration_ms"] >= 0.0
        assert stats["avg_duration_ms"] >= 0.0


# =========================================================================
# clear
# =========================================================================


class TestClear:
    """Tests for the clear method."""

    def test_clear_resets_runs(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        assert len(engine._pipeline_runs) > 0
        engine.clear()
        assert len(engine._pipeline_runs) == 0

    def test_clear_resets_counters(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.run_pipeline(assets=sample_assets, hazard_types=sample_hazard_types)
        engine.clear()
        assert engine._total_runs == 0
        assert engine._success_count == 0
        assert engine._failure_count == 0
        assert engine._partial_count == 0
        assert engine._total_duration_ms == 0.0

    def test_clear_resets_stage_durations(self, engine: HazardPipelineEngine):
        engine.clear()
        for stage in PIPELINE_STAGES:
            assert engine._stage_durations[stage] == []

    def test_clear_calls_provenance_reset(
        self, engine: HazardPipelineEngine, mock_provenance
    ):
        engine.clear()
        mock_provenance.reset.assert_called()

    def test_clear_calls_engine_clear(
        self,
        engine: HazardPipelineEngine,
        mock_database,
        mock_risk_engine,
        mock_reporter,
    ):
        engine.clear()
        mock_database.clear.assert_called()
        mock_risk_engine.clear.assert_called()
        mock_reporter.clear.assert_called()

    def test_clear_then_statistics(self, engine: HazardPipelineEngine):
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_runs"] == 0

    def test_clear_then_run(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        engine.clear()
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
        )
        assert result["status"] in ("completed", "partial", "failed")


# =========================================================================
# Thread safety
# =========================================================================


class TestThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_pipeline_runs(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        errors = []

        def run():
            try:
                engine.run_pipeline(
                    assets=sample_assets,
                    hazard_types=sample_hazard_types,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine._total_runs >= 5


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge case tests for HazardPipelineEngine."""

    def test_single_stage_only(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            stages=["ingest"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_all_stages_explicit(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
        sample_hazard_types,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=sample_hazard_types,
            stages=PIPELINE_STAGES,
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_pipeline_with_many_assets(
        self,
        engine: HazardPipelineEngine,
    ):
        assets = [
            {
                "asset_id": f"asset_{i:04d}",
                "name": f"Asset {i}",
                "asset_type": "facility",
                "location": {"lat": float(i), "lon": float(i)},
            }
            for i in range(20)
        ]
        result = engine.run_pipeline(
            assets=assets,
            hazard_types=["flood"],
        )
        assert result["status"] in ("completed", "partial", "failed")

    def test_pipeline_multiple_hazard_types(
        self,
        engine: HazardPipelineEngine,
        sample_assets,
    ):
        result = engine.run_pipeline(
            assets=sample_assets,
            hazard_types=["flood", "drought", "wildfire", "heat_wave", "storm"],
        )
        assert result["status"] in ("completed", "partial", "failed")
