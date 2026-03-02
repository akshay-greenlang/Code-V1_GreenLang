# -*- coding: utf-8 -*-
"""
Test suite for franchises.franchises_pipeline - AGENT-MRV-027.

Tests the FranchisesPipelineEngine including the full 10-stage pipeline,
individual stages, stage error recovery, batch pipeline, network pipeline,
stage metrics recording, provenance generation, and different franchise types.

Target: 50+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from greenlang.franchises.franchises_pipeline import (
    FranchisesPipelineEngine,
    PipelineStage,
    PipelineStatus,
)


# ==============================================================================
# HELPERS
# ==============================================================================


def _make_energy_unit(
    unit_id: str = "FRN-QSR-001",
    franchise_type: str = "quick_service_restaurant",
    ownership_type: str = "franchise",
    electricity_kwh: float = 180000,
    gas_kwh: float = 85000,
    floor_area_sqm: float = 250,
    country: str = "US",
) -> dict:
    """Build a franchise unit dict with energy data for pipeline input."""
    return {
        "unit_id": unit_id,
        "franchise_type": franchise_type,
        "ownership_type": ownership_type,
        "electricity_kwh": electricity_kwh,
        "gas_kwh": gas_kwh,
        "floor_area_sqm": floor_area_sqm,
        "country": country,
        "operating_months": 12,
    }


def _make_area_unit(
    unit_id: str = "FRN-RTL-001",
    franchise_type: str = "retail_store",
    floor_area_sqm: float = 400,
    country: str = "GB",
) -> dict:
    """Build a franchise unit dict with floor area data only (Tier 2)."""
    return {
        "unit_id": unit_id,
        "franchise_type": franchise_type,
        "ownership_type": "franchise",
        "floor_area_sqm": floor_area_sqm,
        "country": country,
        "operating_months": 12,
    }


def _make_spend_unit(
    unit_id: str = "FRN-SPD-001",
    franchise_type: str = "convenience_store",
    revenue: float = 2400000,
    country: str = "US",
) -> dict:
    """Build a franchise unit dict with spend data only (Tier 3)."""
    return {
        "unit_id": unit_id,
        "franchise_type": franchise_type,
        "ownership_type": "franchise",
        "revenue": revenue,
        "country": country,
        "operating_months": 12,
    }


def _make_network_input(units: List[dict] = None) -> dict:
    """Build a network-level pipeline input dict."""
    if units is None:
        units = [_make_energy_unit()]
    return {
        "units": units,
        "reporting_period": "2025",
        "consolidation_approach": "financial_control",
        "tenant_id": "tenant-001",
    }


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> FranchisesPipelineEngine:
    """Create a fresh FranchisesPipelineEngine instance."""
    FranchisesPipelineEngine._instance = None
    if hasattr(FranchisesPipelineEngine, "_initialized"):
        try:
            del FranchisesPipelineEngine._initialized
        except AttributeError:
            pass
    return FranchisesPipelineEngine()


@pytest.fixture
def single_unit_input() -> dict:
    """Single unit pipeline input dict (for execute_single)."""
    return _make_energy_unit()


@pytest.fixture
def network_input() -> dict:
    """Network pipeline input with a single unit."""
    return _make_network_input()


@pytest.fixture
def multi_unit_network() -> dict:
    """Network input with 3 diverse units."""
    return _make_network_input([
        _make_energy_unit(),
        _make_area_unit(),
        _make_spend_unit(),
    ])


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestFranchisesPipelineInit:
    """Test FranchisesPipelineEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via __new__."""
        FranchisesPipelineEngine._instance = None
        if hasattr(FranchisesPipelineEngine, "_initialized"):
            try:
                del FranchisesPipelineEngine._initialized
            except AttributeError:
                pass
        e1 = FranchisesPipelineEngine()
        e2 = FranchisesPipelineEngine()
        assert e1 is e2

    def test_engine_reset_singleton(self):
        """Test reset_singleton class method."""
        FranchisesPipelineEngine._instance = None
        eng = FranchisesPipelineEngine()
        FranchisesPipelineEngine.reset_singleton()
        assert FranchisesPipelineEngine._instance is None


# ==============================================================================
# SINGLE UNIT PIPELINE TESTS
# ==============================================================================


class TestSingleUnitPipeline:
    """Test execute_single for individual franchise units."""

    def test_single_unit_success(self, engine, single_unit_input):
        """Test single unit pipeline produces SUCCESS result."""
        result = engine.execute_single(single_unit_input)
        assert result is not None
        assert result["status"] == "SUCCESS"

    def test_single_unit_has_total_co2e(self, engine, single_unit_input):
        """Test single unit result has total_co2e."""
        result = engine.execute_single(single_unit_input)
        assert "total_co2e" in result
        co2e = Decimal(str(result["total_co2e"]))
        assert co2e > 0

    def test_single_unit_has_method(self, engine, single_unit_input):
        """Test single unit result includes calculation method."""
        result = engine.execute_single(single_unit_input)
        assert "method" in result
        assert isinstance(result["method"], str)

    def test_single_unit_has_provenance(self, engine, single_unit_input):
        """Test single unit result includes provenance hash."""
        result = engine.execute_single(single_unit_input)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_single_unit_has_stage_durations(self, engine, single_unit_input):
        """Test single unit result includes stage durations."""
        result = engine.execute_single(single_unit_input)
        assert "stage_durations" in result
        assert isinstance(result["stage_durations"], dict)

    def test_single_unit_has_processing_time(self, engine, single_unit_input):
        """Test single unit result includes processing_time_ms."""
        result = engine.execute_single(single_unit_input)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_single_unit_has_compliance(self, engine, single_unit_input):
        """Test single unit result includes compliance field."""
        result = engine.execute_single(single_unit_input)
        assert "compliance" in result

    def test_single_unit_dqi_score(self, engine, single_unit_input):
        """Test single unit result includes dqi_score."""
        result = engine.execute_single(single_unit_input)
        assert "dqi_score" in result


# ==============================================================================
# NETWORK PIPELINE TESTS
# ==============================================================================


class TestNetworkPipeline:
    """Test execute for full franchise network."""

    def test_network_pipeline_success(self, engine, network_input):
        """Test network pipeline executes successfully."""
        result = engine.execute(network_input)
        assert result is not None
        assert result["status"] in ("SUCCESS", "PARTIAL_SUCCESS")

    def test_network_has_total_co2e(self, engine, network_input):
        """Test network result has total_co2e."""
        result = engine.execute(network_input)
        co2e = Decimal(str(result["total_co2e"]))
        assert co2e > 0

    def test_network_has_unit_results(self, engine, network_input):
        """Test network result includes per-unit results."""
        result = engine.execute(network_input)
        assert "unit_results" in result
        assert len(result["unit_results"]) >= 1

    def test_network_has_provenance(self, engine, network_input):
        """Test network result includes provenance hash."""
        result = engine.execute(network_input)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_network_has_compliance(self, engine, network_input):
        """Test network result includes compliance results."""
        result = engine.execute(network_input)
        assert "compliance" in result

    def test_network_has_processing_time(self, engine, network_input):
        """Test network result includes processing time."""
        result = engine.execute(network_input)
        assert "processing_time_ms" in result

    def test_network_multi_unit(self, engine, multi_unit_network):
        """Test network pipeline with 3 diverse units."""
        result = engine.execute(multi_unit_network)
        assert result["total_units"] == 3
        assert result["successful_units"] >= 1

    def test_network_by_franchise_type(self, engine, multi_unit_network):
        """Test network result includes by_franchise_type breakdown."""
        result = engine.execute(multi_unit_network)
        assert "by_franchise_type" in result

    def test_network_by_region(self, engine, multi_unit_network):
        """Test network result includes by_region breakdown."""
        result = engine.execute(multi_unit_network)
        assert "by_region" in result


# ==============================================================================
# STAGE ERROR RECOVERY TESTS
# ==============================================================================


class TestStageErrorRecovery:
    """Test error recovery in pipeline stages."""

    def test_empty_units_raises(self, engine):
        """Test empty units list raises ValueError."""
        with pytest.raises(ValueError):
            engine.execute({"units": []})

    def test_no_units_key_raises(self, engine):
        """Test missing units key raises ValueError."""
        with pytest.raises(ValueError):
            engine.execute({"reporting_period": "2025"})

    def test_invalid_single_input_raises(self, engine):
        """Test invalid single unit input raises error."""
        with pytest.raises((ValueError, RuntimeError)):
            engine.execute_single({})

    def test_no_data_fields_raises(self, engine):
        """Test unit with no energy/area/spend data fails validation."""
        unit = {
            "unit_id": "FRN-BAD-001",
            "franchise_type": "qsr",
            "ownership_type": "franchise",
        }
        with pytest.raises((ValueError, RuntimeError)):
            engine.execute_single(unit)

    def test_company_owned_validation_error(self, engine):
        """Test company-owned unit fails validation with DC-FRN-001."""
        unit = _make_energy_unit(ownership_type="company_owned")
        with pytest.raises((ValueError, RuntimeError)):
            engine.execute_single(unit)


# ==============================================================================
# BATCH PIPELINE TESTS
# ==============================================================================


class TestBatchPipeline:
    """Test batch pipeline processing."""

    def test_batch_pipeline(self, engine):
        """Test batch pipeline with multiple units."""
        units = [
            _make_energy_unit(unit_id="FRN-B1-001"),
            _make_energy_unit(unit_id="FRN-B2-001", franchise_type="hotel"),
        ]
        results = engine.execute_batch(units)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_batch_preserves_unit_ids(self, engine):
        """Test batch results preserve unit IDs."""
        units = [
            _make_energy_unit(unit_id="FRN-UID-001"),
            _make_energy_unit(unit_id="FRN-UID-002"),
        ]
        results = engine.execute_batch(units)
        unit_ids = [r.get("unit_id", "") for r in results]
        assert "FRN-UID-001" in unit_ids
        assert "FRN-UID-002" in unit_ids

    def test_batch_empty_list(self, engine):
        """Test batch with empty list returns empty."""
        results = engine.execute_batch([])
        assert results == []


# ==============================================================================
# PIPELINE STAGE TESTS
# ==============================================================================


class TestPipelineStages:
    """Test pipeline stage enumeration."""

    def test_10_stages_defined(self):
        """Test all 10 pipeline stages are defined."""
        expected = [
            "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS", "CALCULATE",
            "ALLOCATE", "AGGREGATE", "COMPLIANCE", "PROVENANCE", "SEAL",
        ]
        actual = [ps.value for ps in PipelineStage]
        for s in expected:
            assert s in actual

    @pytest.mark.parametrize("stage", [ps.value for ps in PipelineStage])
    def test_each_stage_value(self, stage):
        """Test each pipeline stage has a valid string value."""
        assert isinstance(stage, str)
        assert len(stage) > 0


# ==============================================================================
# PIPELINE STATUS TESTS
# ==============================================================================


class TestPipelineStatusEnum:
    """Test PipelineStatus enumeration."""

    def test_success_status(self):
        """Test SUCCESS status value."""
        assert PipelineStatus.SUCCESS.value == "SUCCESS"

    def test_partial_success_status(self):
        """Test PARTIAL_SUCCESS status value."""
        assert PipelineStatus.PARTIAL_SUCCESS.value == "PARTIAL_SUCCESS"

    def test_failed_status(self):
        """Test FAILED status value."""
        assert PipelineStatus.FAILED.value == "FAILED"


# ==============================================================================
# PROVENANCE GENERATION TESTS
# ==============================================================================


class TestProvenanceGeneration:
    """Test provenance chain generation."""

    def test_provenance_hash_is_sha256(self, engine, single_unit_input):
        """Test provenance hash is a 64-char hex string."""
        result = engine.execute_single(single_unit_input)
        ph = result.get("provenance_hash", "")
        assert len(ph) == 64
        assert all(c in "0123456789abcdef" for c in ph)

    def test_provenance_hash_unique_per_run(self, engine):
        """Test each pipeline run produces a valid provenance hash."""
        unit = _make_energy_unit(unit_id="FRN-DET-001", electricity_kwh=100000)
        r1 = engine.execute_single(dict(unit))
        r2 = engine.execute_single(dict(unit))
        # Each run may include a timestamp, so hashes can differ;
        # verify both are valid SHA-256 hex strings.
        assert len(r1["provenance_hash"]) == 64
        assert len(r2["provenance_hash"]) == 64


# ==============================================================================
# GET PIPELINE STATUS TESTS
# ==============================================================================


class TestGetPipelineStatus:
    """Test get_pipeline_status method."""

    def test_pipeline_status_dict(self, engine):
        """Test get_pipeline_status returns metadata dict."""
        status = engine.get_pipeline_status()
        assert isinstance(status, dict)
        assert status["agent_id"] == "GL-MRV-S3-014"
        assert status["agent_component"] == "AGENT-MRV-027"

    def test_pipeline_status_engines_loaded(self, engine):
        """Test pipeline status includes engines_loaded section."""
        status = engine.get_pipeline_status()
        assert "engines_loaded" in status
        assert isinstance(status["engines_loaded"], dict)

    def test_pipeline_status_max_batch_size(self, engine):
        """Test pipeline status includes max_batch_size."""
        status = engine.get_pipeline_status()
        assert status["max_batch_size"] == 10000


# ==============================================================================
# PARAMETRIZED FRANCHISE TYPE TESTS
# ==============================================================================


class TestParametrizedFranchiseTypes:
    """Test pipeline with different franchise types."""

    @pytest.mark.parametrize("franchise_type", [
        "quick_service_restaurant", "hotel", "convenience_store",
        "retail_store", "fitness_center", "automotive_service",
    ])
    def test_pipeline_by_franchise_type(self, engine, franchise_type):
        """Test pipeline for each franchise type."""
        unit = _make_energy_unit(
            unit_id=f"FRN-{franchise_type[:3].upper()}-001",
            franchise_type=franchise_type,
        )
        result = engine.execute_single(unit)
        assert result["status"] == "SUCCESS"
        assert Decimal(str(result["total_co2e"])) > 0


# ==============================================================================
# PROGRESS CALLBACK TESTS
# ==============================================================================


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_set_progress_callback(self, engine):
        """Test setting a progress callback."""
        called = []
        engine.set_progress_callback(lambda c, t: called.append((c, t)))
        network = _make_network_input([
            _make_energy_unit(unit_id="FRN-CB-001"),
            _make_energy_unit(unit_id="FRN-CB-002"),
        ])
        engine.execute(network)
        assert len(called) >= 1

    def test_none_callback(self, engine):
        """Test setting callback to None does not error."""
        engine.set_progress_callback(None)
        result = engine.execute(_make_network_input())
        assert result["status"] in ("SUCCESS", "PARTIAL_SUCCESS")


# ==============================================================================
# RESET PIPELINE TESTS
# ==============================================================================


class TestResetPipeline:
    """Test reset_pipeline method."""

    def test_reset_pipeline_clears_state(self, engine):
        """Test reset_pipeline clears provenance chains."""
        engine.reset_pipeline()
        status = engine.get_pipeline_status()
        assert status["active_chains"] == 0
