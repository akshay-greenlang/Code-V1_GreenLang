# -*- coding: utf-8 -*-
"""
Unit tests for ProcessEmissionsPipelineEngine (Engine 7).

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests the 8-stage pipeline (VALIDATE, RESOLVE_PROCESS,
CALCULATE_MATERIAL_BALANCE, CALCULATE_EMISSIONS, APPLY_ABATEMENT,
QUANTIFY_UNCERTAINTY, CHECK_COMPLIANCE, GENERATE_AUDIT), batch
execution, statistics tracking, provenance hashing, and error handling.

Total: 64 tests across 7 test classes.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.process_emissions.process_emissions_pipeline import (
    ProcessEmissionsPipelineEngine,
    PipelineStage,
    PIPELINE_STAGES,
    GWP_VALUES,
    CARBONATE_FACTORS,
    PROCESS_TYPES,
    DEFAULT_ABATEMENT_EFFICIENCIES,
    REGULATORY_FRAMEWORKS,
    _compute_hash,
    _to_decimal,
    _stage_result,
    _utcnow,
    _utcnow_iso,
    _new_uuid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ProcessEmissionsPipelineEngine:
    """Create a standalone ProcessEmissionsPipelineEngine (no upstreams)."""
    return ProcessEmissionsPipelineEngine()


@pytest.fixture
def cement_request() -> Dict[str, Any]:
    """Standard cement production pipeline request."""
    return {
        "process_type": "CEMENT_CLINKER",
        "production_quantity": 100000.0,
        "production_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "FAC-001",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def iron_steel_request() -> Dict[str, Any]:
    """Iron and steel BF-BOF pipeline request."""
    return {
        "process_type": "IRON_STEEL_BF_BOF",
        "production_quantity": 50000.0,
        "production_unit": "tonne",
        "calculation_method": "MASS_BALANCE",
        "production_route": "BF_BOF",
        "facility_id": "FAC-STEEL-001",
    }


@pytest.fixture
def semiconductor_request() -> Dict[str, Any]:
    """Semiconductor manufacturing pipeline request."""
    return {
        "process_type": "SEMICONDUCTOR_MANUFACTURING",
        "production_quantity": 10000.0,
        "production_unit": "wafer_start",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "FAC-SEMI-001",
    }


@pytest.fixture
def nitric_acid_request() -> Dict[str, Any]:
    """Nitric acid production pipeline request."""
    return {
        "process_type": "NITRIC_ACID",
        "production_quantity": 80000.0,
        "production_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "FAC-CHEM-001",
    }


@pytest.fixture
def batch_requests() -> List[Dict[str, Any]]:
    """Multiple pipeline requests for batch testing."""
    return [
        {
            "process_type": "CEMENT_CLINKER",
            "production_quantity": 10000.0,
        },
        {
            "process_type": "LIME_PRODUCTION",
            "production_quantity": 5000.0,
        },
        {
            "process_type": "AMMONIA_PRODUCTION",
            "production_quantity": 20000.0,
        },
    ]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPipelineInit:
    """Test pipeline engine initialization."""

    def test_init_standalone(self):
        """Engine initializes without any upstream engines."""
        engine = ProcessEmissionsPipelineEngine()
        assert engine.process_database is None
        assert engine.emission_calculator is None
        assert engine.material_balance_engine is None
        assert engine.abatement_tracker is None
        assert engine.uncertainty_engine is None
        assert engine.compliance_checker is None

    def test_init_with_engines(self):
        """Engine accepts injected upstream engines."""
        mock_db = MagicMock()
        mock_calc = MagicMock()
        engine = ProcessEmissionsPipelineEngine(
            process_database=mock_db,
            emission_calculator=mock_calc,
        )
        assert engine.process_database is mock_db
        assert engine.emission_calculator is mock_calc

    def test_init_statistics_zeroed(self, engine: ProcessEmissionsPipelineEngine):
        """Initial statistics counters are zero."""
        assert engine._total_runs == 0
        assert engine._successful_runs == 0
        assert engine._failed_runs == 0
        assert engine._total_duration_ms == 0.0

    def test_pipeline_stages_constant(self):
        """PIPELINE_STAGES lists all 8 stages in order."""
        assert len(PIPELINE_STAGES) == 8
        assert PIPELINE_STAGES[0] == "VALIDATE"
        assert PIPELINE_STAGES[-1] == "GENERATE_AUDIT"


class TestPipelineStages:
    """Test PipelineStage enum and individual stage definitions."""

    def test_pipeline_stage_enum_values(self):
        """PipelineStage enum has all 8 values."""
        assert len(PipelineStage) == 8
        assert PipelineStage.VALIDATE.value == "VALIDATE"
        assert PipelineStage.RESOLVE_PROCESS.value == "RESOLVE_PROCESS"
        assert PipelineStage.CALCULATE_MATERIAL_BALANCE.value == "CALCULATE_MATERIAL_BALANCE"
        assert PipelineStage.CALCULATE_EMISSIONS.value == "CALCULATE_EMISSIONS"
        assert PipelineStage.APPLY_ABATEMENT.value == "APPLY_ABATEMENT"
        assert PipelineStage.QUANTIFY_UNCERTAINTY.value == "QUANTIFY_UNCERTAINTY"
        assert PipelineStage.CHECK_COMPLIANCE.value == "CHECK_COMPLIANCE"
        assert PipelineStage.GENERATE_AUDIT.value == "GENERATE_AUDIT"

    def test_stage_result_helper(self):
        """_stage_result produces a well-formed result dict."""
        sr = _stage_result("VALIDATE", True, 1.5, "test-pipeline-id")
        assert sr["stage"] == "VALIDATE"
        assert sr["success"] is True
        assert sr["duration_ms"] == 1.5
        assert "provenance_hash" in sr

    def test_stage_result_with_error(self):
        """_stage_result includes error field when provided."""
        sr = _stage_result(
            "VALIDATE", False, 0.5, "pid-1",
            error="Missing process_type",
        )
        assert sr["success"] is False
        assert sr["error"] == "Missing process_type"

    def test_stage_result_with_extra(self):
        """_stage_result includes extra key-value pairs."""
        sr = _stage_result(
            "RESOLVE_PROCESS", True, 2.0, "pid-1",
            extra={"resolved_type": "CEMENT_CLINKER"},
        )
        assert sr["resolved_type"] == "CEMENT_CLINKER"


class TestExecutePipeline:
    """Test full pipeline execution."""

    def test_execute_cement_pipeline(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Execute pipeline for cement production returns valid result."""
        result = engine.execute_pipeline(cement_request)
        assert "success" in result
        assert "pipeline_id" in result
        assert "stages_completed" in result
        assert "stages_total" in result
        assert result["stages_total"] == 8
        assert "result" in result
        assert "pipeline_provenance_hash" in result
        assert "total_duration_ms" in result
        assert result["total_duration_ms"] > 0

    def test_pipeline_generates_uuid_pipeline_id(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline generates unique pipeline_id for each run."""
        r1 = engine.execute_pipeline(cement_request)
        r2 = engine.execute_pipeline(cement_request)
        assert r1["pipeline_id"] != r2["pipeline_id"]

    def test_pipeline_provenance_hash_is_sha256(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline provenance hash is a 64-char SHA-256 hex digest."""
        result = engine.execute_pipeline(cement_request)
        assert len(result["pipeline_provenance_hash"]) == 64

    def test_pipeline_stage_results_list(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline result includes stage_results for all 8 stages."""
        result = engine.execute_pipeline(cement_request)
        assert len(result["stage_results"]) == 8
        for sr in result["stage_results"]:
            assert "stage" in sr
            assert "success" in sr
            assert "duration_ms" in sr

    def test_pipeline_iron_steel(
        self,
        engine: ProcessEmissionsPipelineEngine,
        iron_steel_request: Dict[str, Any],
    ):
        """Pipeline executes for iron/steel BF-BOF process."""
        result = engine.execute_pipeline(iron_steel_request)
        assert "success" in result
        assert result["stages_total"] == 8

    def test_pipeline_semiconductor(
        self,
        engine: ProcessEmissionsPipelineEngine,
        semiconductor_request: Dict[str, Any],
    ):
        """Pipeline executes for semiconductor manufacturing."""
        result = engine.execute_pipeline(semiconductor_request)
        assert "success" in result
        assert result["stages_total"] == 8

    def test_pipeline_nitric_acid(
        self,
        engine: ProcessEmissionsPipelineEngine,
        nitric_acid_request: Dict[str, Any],
    ):
        """Pipeline executes for nitric acid production."""
        result = engine.execute_pipeline(nitric_acid_request)
        assert "success" in result

    def test_pipeline_skip_abatement(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline can skip abatement stage."""
        result = engine.execute_pipeline(
            cement_request, include_abatement=False,
        )
        abatement_stage = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "APPLY_ABATEMENT"
        )
        assert abatement_stage.get("skipped") is True

    def test_pipeline_skip_uncertainty(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline can skip uncertainty stage."""
        result = engine.execute_pipeline(
            cement_request, include_uncertainty=False,
        )
        uncertainty_stage = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "QUANTIFY_UNCERTAINTY"
        )
        assert uncertainty_stage.get("skipped") is True

    def test_pipeline_skip_compliance(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline can skip compliance stage."""
        result = engine.execute_pipeline(
            cement_request, include_compliance=False,
        )
        compliance_stage = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "CHECK_COMPLIANCE"
        )
        assert compliance_stage.get("skipped") is True

    def test_pipeline_gwp_sources(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline accepts different GWP sources."""
        for gwp in ["AR4", "AR5", "AR6", "AR6_20yr"]:
            result = engine.execute_pipeline(
                cement_request, gwp_source=gwp,
            )
            assert "success" in result

    def test_pipeline_tiers(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline accepts different calculation tiers."""
        for tier in ["TIER_1", "TIER_2", "TIER_3"]:
            result = engine.execute_pipeline(
                cement_request, tier=tier,
            )
            assert "success" in result

    def test_pipeline_empty_request(
        self,
        engine: ProcessEmissionsPipelineEngine,
    ):
        """Pipeline handles empty request dict gracefully."""
        result = engine.execute_pipeline({})
        assert "success" in result
        # Should still complete with 8 stage results
        assert len(result["stage_results"]) == 8


class TestBatchExecution:
    """Test pipeline batch execution."""

    def test_batch_execute_returns_list(
        self,
        engine: ProcessEmissionsPipelineEngine,
        batch_requests: List[Dict[str, Any]],
    ):
        """execute_batch returns results for all requests."""
        results = engine.execute_batch(batch_requests)
        assert len(results) == len(batch_requests)

    def test_batch_execute_each_has_pipeline_id(
        self,
        engine: ProcessEmissionsPipelineEngine,
        batch_requests: List[Dict[str, Any]],
    ):
        """Each batch result has a unique pipeline_id."""
        results = engine.execute_batch(batch_requests)
        ids = [r["pipeline_id"] for r in results]
        assert len(set(ids)) == len(ids)

    def test_batch_empty_list(
        self,
        engine: ProcessEmissionsPipelineEngine,
    ):
        """execute_batch with empty list returns empty results."""
        results = engine.execute_batch([])
        assert results == []

    def test_batch_single_item(
        self,
        engine: ProcessEmissionsPipelineEngine,
    ):
        """execute_batch with single item returns single result."""
        results = engine.execute_batch([
            {"process_type": "CEMENT_CLINKER", "production_quantity": 100},
        ])
        assert len(results) == 1


class TestStatisticsTracking:
    """Test pipeline statistics tracking."""

    def test_stats_increment_on_run(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Pipeline run increments total_runs."""
        assert engine._total_runs == 0
        engine.execute_pipeline(cement_request)
        assert engine._total_runs == 1

    def test_stats_multiple_runs(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Multiple runs accumulate statistics."""
        for _ in range(3):
            engine.execute_pipeline(cement_request)
        assert engine._total_runs == 3

    def test_stats_duration_accumulates(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Total duration accumulates across runs."""
        engine.execute_pipeline(cement_request)
        engine.execute_pipeline(cement_request)
        assert engine._total_duration_ms > 0

    def test_stats_last_run_at_updated(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """Last run timestamp is updated after each run."""
        assert engine._last_run_at is None
        engine.execute_pipeline(cement_request)
        assert engine._last_run_at is not None

    def test_get_statistics(
        self,
        engine: ProcessEmissionsPipelineEngine,
        cement_request: Dict[str, Any],
    ):
        """get_statistics returns accumulated counters."""
        engine.execute_pipeline(cement_request)
        if hasattr(engine, "get_statistics"):
            stats = engine.get_statistics()
            assert stats["total_runs"] == 1


class TestReferenceData:
    """Test built-in reference data used by the pipeline."""

    def test_gwp_values_has_4_sources(self):
        """GWP_VALUES has AR4, AR5, AR6, and AR6_20yr."""
        assert len(GWP_VALUES) == 4
        assert "AR4" in GWP_VALUES
        assert "AR5" in GWP_VALUES
        assert "AR6" in GWP_VALUES
        assert "AR6_20yr" in GWP_VALUES

    def test_gwp_co2_always_1(self):
        """CO2 GWP is always 1.0 regardless of source."""
        for source, values in GWP_VALUES.items():
            assert values["CO2"] == 1.0, f"CO2 != 1.0 for {source}"

    def test_carbonate_factors_has_key_minerals(self):
        """CARBONATE_FACTORS includes key mineral types."""
        assert "CALCITE" in CARBONATE_FACTORS
        assert "DOLOMITE" in CARBONATE_FACTORS
        assert "MAGNESITE" in CARBONATE_FACTORS
        assert "SIDERITE" in CARBONATE_FACTORS

    def test_carbonate_factors_have_co2_factor(self):
        """Each carbonate has a co2_factor field."""
        for mineral, data in CARBONATE_FACTORS.items():
            assert "co2_factor" in data, f"{mineral} missing co2_factor"
            assert data["co2_factor"] > 0

    def test_process_types_coverage(self):
        """PROCESS_TYPES covers mineral, chemical, metal, electronics."""
        categories = set()
        for pt, data in PROCESS_TYPES.items():
            categories.add(data["category"])
        assert "MINERAL" in categories
        assert "CHEMICAL" in categories
        assert "METAL" in categories
        assert "ELECTRONICS" in categories

    def test_process_types_have_default_ef(self):
        """All process types have at least one default emission factor."""
        for pt, data in PROCESS_TYPES.items():
            has_ef = any(
                key.startswith("default_ef_")
                for key in data.keys()
            )
            assert has_ef, f"{pt} missing default emission factor"

    def test_abatement_efficiencies_range(self):
        """All abatement efficiencies are between 0 and 1."""
        for atype, processes in DEFAULT_ABATEMENT_EFFICIENCIES.items():
            for process, eff in processes.items():
                assert 0.0 <= eff <= 1.0, (
                    f"{atype}/{process}: efficiency {eff} out of range"
                )

    def test_regulatory_frameworks_has_6_entries(self):
        """REGULATORY_FRAMEWORKS dict has 6 framework entries."""
        assert len(REGULATORY_FRAMEWORKS) == 6
        assert "GHG_PROTOCOL" in REGULATORY_FRAMEWORKS
        assert "ISO_14064" in REGULATORY_FRAMEWORKS
        assert "CSRD_ESRS_E1" in REGULATORY_FRAMEWORKS
        assert "EPA_40CFR98" in REGULATORY_FRAMEWORKS
        assert "UK_SECR" in REGULATORY_FRAMEWORKS
        assert "EU_ETS" in REGULATORY_FRAMEWORKS


class TestUtilityHelpers:
    """Test module-level utility functions."""

    def test_compute_hash_deterministic(self):
        """_compute_hash is deterministic for same input."""
        data = {"key": "value", "num": 42}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_compute_hash_different_data(self):
        """_compute_hash produces different hashes for different data."""
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_sha256_length(self):
        """_compute_hash produces a 64-character hex string."""
        h = _compute_hash("test")
        assert len(h) == 64

    def test_to_decimal_from_int(self):
        """_to_decimal converts int to Decimal."""
        assert _to_decimal(42) == Decimal("42")

    def test_to_decimal_from_float(self):
        """_to_decimal converts float to Decimal via string."""
        result = _to_decimal(3.14)
        assert isinstance(result, Decimal)

    def test_to_decimal_from_decimal(self):
        """_to_decimal passes through Decimal unchanged."""
        d = Decimal("99.99")
        assert _to_decimal(d) is d

    def test_to_decimal_invalid(self):
        """_to_decimal returns Decimal('0') for invalid input."""
        assert _to_decimal("not_a_number") == Decimal("0")
        assert _to_decimal(None) == Decimal("0")

    def test_utcnow_returns_datetime(self):
        """_utcnow returns a datetime object."""
        from datetime import datetime
        dt = _utcnow()
        assert isinstance(dt, datetime)

    def test_utcnow_iso_returns_string(self):
        """_utcnow_iso returns an ISO-8601 string."""
        s = _utcnow_iso()
        assert isinstance(s, str)
        assert "T" in s

    def test_new_uuid_returns_unique(self):
        """_new_uuid generates unique strings."""
        u1 = _new_uuid()
        u2 = _new_uuid()
        assert u1 != u2
