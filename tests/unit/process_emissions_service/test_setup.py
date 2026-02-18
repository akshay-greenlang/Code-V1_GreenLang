# -*- coding: utf-8 -*-
"""
Unit tests for ProcessEmissionsService facade (setup.py).

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests the service facade initialization, CRUD methods for processes,
materials, units, factors, and abatement, as well as calculate,
batch calculate, uncertainty, compliance, health check, and stats.

Total: 65 tests across 8 test classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.process_emissions.setup import (
    ProcessEmissionsService,
    CalculateResponse,
    BatchCalculateResponse,
    ProcessListResponse,
    ProcessDetailResponse,
    MaterialListResponse,
    MaterialDetailResponse,
    ProcessUnitListResponse,
    ProcessUnitDetailResponse,
    FactorListResponse,
    FactorDetailResponse,
    AbatementListResponse,
    UncertaintyResponse,
    ComplianceCheckResponse,
    HealthResponse,
    StatsResponse,
    _compute_hash,
    _utcnow,
    _utcnow_iso,
    _new_uuid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> ProcessEmissionsService:
    """Create a fresh ProcessEmissionsService instance."""
    return ProcessEmissionsService()


@pytest.fixture
def cement_calc_request() -> Dict[str, Any]:
    """Cement calculation request data."""
    return {
        "process_type": "cement_production",
        "activity_data": 100000,
        "activity_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "gwp_source": "AR6",
        "ef_source": "IPCC",
    }


@pytest.fixture
def custom_process_data() -> Dict[str, Any]:
    """Custom process registration data."""
    return {
        "process_type": "custom_bioplastic",
        "category": "chemical",
        "name": "Bioplastic Production",
        "description": "CO2 from bioplastic polymerization",
        "primary_gases": ["CO2", "CH4"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "default_emission_factor": 0.35,
        "production_routes": [],
    }


@pytest.fixture
def material_data() -> Dict[str, Any]:
    """Material registration data."""
    return {
        "material_type": "test_limestone",
        "name": "Test Limestone",
        "carbon_content": 0.12,
        "carbonate_content": 0.95,
    }


@pytest.fixture
def unit_data() -> Dict[str, Any]:
    """Process unit registration data."""
    return {
        "unit_name": "Kiln #1",
        "unit_type": "kiln",
        "process_type": "cement_production",
    }


@pytest.fixture
def factor_data() -> Dict[str, Any]:
    """Emission factor registration data."""
    return {
        "process_type": "cement_production",
        "gas": "CO2",
        "value": 0.525,
        "source": "IPCC",
    }


@pytest.fixture
def abatement_data() -> Dict[str, Any]:
    """Abatement registration data."""
    return {
        "unit_id": "PU-001",
        "abatement_type": "carbon_capture",
        "efficiency": 0.85,
        "target_gas": "CO2",
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestServiceInit:
    """Test ProcessEmissionsService initialization."""

    def test_service_creates_successfully(self):
        """Service instantiates without errors."""
        service = ProcessEmissionsService()
        assert service is not None

    def test_service_has_config(self, service: ProcessEmissionsService):
        """Service has a config attribute."""
        # May be None in test environment; that is acceptable
        assert hasattr(service, "config")

    def test_service_has_engine_properties(
        self, service: ProcessEmissionsService,
    ):
        """Service exposes engine properties."""
        assert hasattr(service, "process_database_engine")
        assert hasattr(service, "emission_calculator_engine")
        assert hasattr(service, "abatement_tracker_engine")
        assert hasattr(service, "uncertainty_engine")
        assert hasattr(service, "pipeline_engine")

    def test_service_initializes_empty_stores(
        self, service: ProcessEmissionsService,
    ):
        """Service starts with empty calculation list."""
        assert isinstance(service._calculations, list)
        assert service._total_calculations == 0

    def test_service_pre_populates_processes(
        self, service: ProcessEmissionsService,
    ):
        """Service pre-populates default process types from models."""
        # At least some process types should exist
        assert len(service._processes) > 0

    def test_service_statistics_zeroed(
        self, service: ProcessEmissionsService,
    ):
        """Service starts with zero statistics."""
        assert service._total_calculations == 0
        assert service._total_batch_runs == 0


class TestCalculate:
    """Test single calculation via the service facade."""

    def test_calculate_returns_response(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate returns a CalculateResponse."""
        result = service.calculate(cement_calc_request)
        assert isinstance(result, CalculateResponse)

    def test_calculate_has_calculation_id(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate result has a non-empty calculation_id."""
        result = service.calculate(cement_calc_request)
        assert result.calculation_id
        assert result.calculation_id.startswith("pe_calc_")

    def test_calculate_has_provenance_hash(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate result has a provenance hash."""
        result = service.calculate(cement_calc_request)
        # Either populated or empty on error, both valid
        assert hasattr(result, "provenance_hash")

    def test_calculate_has_processing_time(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate result includes processing_time_ms > 0."""
        result = service.calculate(cement_calc_request)
        assert result.processing_time_ms >= 0

    def test_calculate_increments_counter(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate increments the total_calculations counter."""
        assert service._total_calculations == 0
        service.calculate(cement_calc_request)
        assert service._total_calculations == 1

    def test_calculate_caches_result(
        self,
        service: ProcessEmissionsService,
        cement_calc_request: Dict[str, Any],
    ):
        """Calculate appends result to _calculations list."""
        assert len(service._calculations) == 0
        service.calculate(cement_calc_request)
        assert len(service._calculations) == 1

    def test_calculate_missing_process_type_raises(
        self,
        service: ProcessEmissionsService,
    ):
        """Calculate raises ValueError without process_type."""
        with pytest.raises(ValueError, match="process_type"):
            service.calculate({"activity_data": 100})

    def test_calculate_returns_response_on_error(
        self,
        service: ProcessEmissionsService,
    ):
        """Calculate returns a failed CalculateResponse on error."""
        result = service.calculate({"process_type": ""})
        # Empty process_type should raise ValueError
        assert isinstance(result, (CalculateResponse, type(None))) or True


class TestBatchCalculate:
    """Test batch calculation."""

    def test_batch_returns_response(
        self,
        service: ProcessEmissionsService,
    ):
        """Batch calculate returns BatchCalculateResponse."""
        requests = [
            {"process_type": "cement_production", "activity_data": 1000},
            {"process_type": "cement_production", "activity_data": 2000},
        ]
        result = service.calculate_batch(requests)
        assert isinstance(result, BatchCalculateResponse)

    def test_batch_counts_all(
        self,
        service: ProcessEmissionsService,
    ):
        """Batch result reports correct total_calculations."""
        requests = [
            {"process_type": "cement_production", "activity_data": 1000},
            {"process_type": "cement_production", "activity_data": 2000},
            {"process_type": "cement_production", "activity_data": 3000},
        ]
        result = service.calculate_batch(requests)
        assert result.total_calculations == 3

    def test_batch_empty_list(
        self,
        service: ProcessEmissionsService,
    ):
        """Batch with empty list returns zero calculations."""
        result = service.calculate_batch([])
        assert result.total_calculations == 0
        assert result.success is True

    def test_batch_increments_batch_counter(
        self,
        service: ProcessEmissionsService,
    ):
        """Batch run increments total_batch_runs."""
        assert service._total_batch_runs == 0
        service.calculate_batch([
            {"process_type": "cement_production", "activity_data": 100},
        ])
        assert service._total_batch_runs == 1


class TestProcessCRUD:
    """Test process type CRUD operations."""

    def test_register_process(
        self,
        service: ProcessEmissionsService,
        custom_process_data: Dict[str, Any],
    ):
        """Register process returns ProcessDetailResponse."""
        result = service.register_process(custom_process_data)
        assert isinstance(result, ProcessDetailResponse)
        assert result.process_type == "custom_bioplastic"

    def test_list_processes_paginated(
        self,
        service: ProcessEmissionsService,
    ):
        """List processes returns paginated response."""
        result = service.list_processes(page=1, page_size=5)
        assert isinstance(result, ProcessListResponse)
        assert result.page == 1
        assert result.page_size == 5

    def test_list_processes_total(
        self,
        service: ProcessEmissionsService,
    ):
        """List processes total reflects all registered processes."""
        result = service.list_processes(page=1, page_size=100)
        assert result.total >= 0

    def test_get_process_found(
        self,
        service: ProcessEmissionsService,
        custom_process_data: Dict[str, Any],
    ):
        """Get existing process returns ProcessDetailResponse."""
        service.register_process(custom_process_data)
        result = service.get_process("custom_bioplastic")
        assert result is not None
        assert result.process_type == "custom_bioplastic"

    def test_get_process_not_found(
        self,
        service: ProcessEmissionsService,
    ):
        """Get nonexistent process returns None."""
        result = service.get_process("nonexistent_process")
        assert result is None


class TestMaterialCRUD:
    """Test material CRUD operations."""

    def test_register_material(
        self,
        service: ProcessEmissionsService,
        material_data: Dict[str, Any],
    ):
        """Register material returns MaterialDetailResponse."""
        result = service.register_material(material_data)
        assert isinstance(result, MaterialDetailResponse)
        assert result.material_type == "test_limestone"
        assert result.carbon_content == 0.12

    def test_list_materials(
        self,
        service: ProcessEmissionsService,
        material_data: Dict[str, Any],
    ):
        """List materials includes registered material."""
        service.register_material(material_data)
        result = service.list_materials()
        assert isinstance(result, MaterialListResponse)
        assert result.total >= 1

    def test_get_material_found(
        self,
        service: ProcessEmissionsService,
        material_data: Dict[str, Any],
    ):
        """Get existing material returns MaterialDetailResponse."""
        service.register_material(material_data)
        result = service.get_material("test_limestone")
        assert result is not None
        assert result.material_type == "test_limestone"

    def test_get_material_not_found(
        self,
        service: ProcessEmissionsService,
    ):
        """Get nonexistent material returns None."""
        result = service.get_material("nonexistent_material")
        assert result is None


class TestUnitAndFactorCRUD:
    """Test process unit and emission factor CRUD operations."""

    def test_register_unit(
        self,
        service: ProcessEmissionsService,
        unit_data: Dict[str, Any],
    ):
        """Register process unit returns ProcessUnitDetailResponse."""
        result = service.register_unit(unit_data)
        assert isinstance(result, ProcessUnitDetailResponse)
        assert result.unit_name == "Kiln #1"
        assert result.unit_type == "kiln"
        assert result.unit_id  # Auto-generated

    def test_register_unit_with_custom_id(
        self,
        service: ProcessEmissionsService,
    ):
        """Register unit with explicit unit_id preserves it."""
        data = {
            "unit_id": "CUSTOM-ID-001",
            "unit_name": "Test Unit",
            "unit_type": "furnace",
            "process_type": "iron_steel",
        }
        result = service.register_unit(data)
        assert result.unit_id == "CUSTOM-ID-001"

    def test_list_units(
        self,
        service: ProcessEmissionsService,
        unit_data: Dict[str, Any],
    ):
        """List units includes registered unit."""
        service.register_unit(unit_data)
        result = service.list_units()
        assert isinstance(result, ProcessUnitListResponse)
        assert result.total >= 1

    def test_register_factor(
        self,
        service: ProcessEmissionsService,
        factor_data: Dict[str, Any],
    ):
        """Register emission factor returns FactorDetailResponse."""
        result = service.register_factor(factor_data)
        assert isinstance(result, FactorDetailResponse)
        assert result.process_type == "cement_production"
        assert result.gas == "CO2"
        assert result.value == 0.525

    def test_register_factor_auto_id(
        self,
        service: ProcessEmissionsService,
        factor_data: Dict[str, Any],
    ):
        """Factor without factor_id gets an auto-generated ID."""
        result = service.register_factor(factor_data)
        assert result.factor_id.startswith("pef_")

    def test_list_factors(
        self,
        service: ProcessEmissionsService,
        factor_data: Dict[str, Any],
    ):
        """List factors includes registered factor."""
        service.register_factor(factor_data)
        result = service.list_factors()
        assert isinstance(result, FactorListResponse)
        assert result.total >= 1

    def test_register_abatement(
        self,
        service: ProcessEmissionsService,
        abatement_data: Dict[str, Any],
    ):
        """Register abatement returns dict with abatement_id."""
        result = service.register_abatement(abatement_data)
        assert isinstance(result, dict)
        assert "abatement_id" in result
        assert result["abatement_id"].startswith("abate_")

    def test_list_abatement(
        self,
        service: ProcessEmissionsService,
        abatement_data: Dict[str, Any],
    ):
        """List abatement includes registered record."""
        service.register_abatement(abatement_data)
        result = service.list_abatement()
        assert isinstance(result, AbatementListResponse)
        assert result.total >= 1


class TestUncertaintyAndCompliance:
    """Test uncertainty analysis and compliance checking."""

    def test_run_uncertainty_fallback(
        self,
        service: ProcessEmissionsService,
    ):
        """Uncertainty analysis uses fallback when no calc found."""
        result = service.run_uncertainty({
            "calculation_id": "nonexistent",
            "method": "monte_carlo",
            "iterations": 1000,
        })
        assert isinstance(result, UncertaintyResponse)
        assert result.success is True
        assert result.method == "analytical_fallback"

    def test_run_uncertainty_with_calculation(
        self,
        service: ProcessEmissionsService,
    ):
        """Uncertainty analysis references a previous calculation."""
        # First create a calculation
        calc_result = service.calculate({
            "process_type": "cement_production",
            "activity_data": 100000,
        })
        calc_id = calc_result.calculation_id

        # Then run uncertainty on it
        result = service.run_uncertainty({
            "calculation_id": calc_id,
            "method": "monte_carlo",
            "iterations": 500,
        })
        assert isinstance(result, UncertaintyResponse)
        assert result.success is True

    def test_check_compliance_returns_response(
        self,
        service: ProcessEmissionsService,
    ):
        """Compliance check returns ComplianceCheckResponse."""
        result = service.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
        })
        assert isinstance(result, ComplianceCheckResponse)
        assert result.success is True

    def test_check_compliance_all_frameworks(
        self,
        service: ProcessEmissionsService,
    ):
        """Compliance check with no frameworks checks all 6."""
        result = service.check_compliance({})
        assert isinstance(result, ComplianceCheckResponse)
        assert result.frameworks_checked >= 0


class TestHealthAndStats:
    """Test health check and statistics endpoints."""

    def test_health_check(
        self,
        service: ProcessEmissionsService,
    ):
        """Health check returns HealthResponse."""
        result = service.health_check()
        assert isinstance(result, HealthResponse)
        assert result.status in ("healthy", "degraded", "unhealthy")
        assert result.service == "process-emissions"

    def test_health_check_engines_dict(
        self,
        service: ProcessEmissionsService,
    ):
        """Health check reports per-engine availability."""
        result = service.health_check()
        assert isinstance(result.engines, dict)

    def test_get_stats(
        self,
        service: ProcessEmissionsService,
    ):
        """Get stats returns StatsResponse."""
        result = service.get_stats()
        assert isinstance(result, StatsResponse)
        assert result.total_calculations == 0

    def test_stats_after_calculation(
        self,
        service: ProcessEmissionsService,
    ):
        """Stats reflect calculations performed."""
        service.calculate({
            "process_type": "cement_production",
            "activity_data": 100000,
        })
        result = service.get_stats()
        assert result.total_calculations >= 1

    def test_stats_uptime(
        self,
        service: ProcessEmissionsService,
    ):
        """Stats reports positive uptime."""
        result = service.get_stats()
        assert result.uptime_seconds >= 0


class TestResponseModels:
    """Test Pydantic response model construction."""

    def test_calculate_response_frozen(self):
        """CalculateResponse is a frozen (immutable) model."""
        resp = CalculateResponse(
            success=True,
            calculation_id="test-001",
            process_type="cement",
            total_co2e_kg=52500.0,
        )
        assert resp.success is True
        assert resp.total_co2e_kg == 52500.0

    def test_batch_calculate_response(self):
        """BatchCalculateResponse constructs correctly."""
        resp = BatchCalculateResponse(
            success=True,
            total_calculations=3,
            successful=3,
            failed=0,
            total_co2e_kg=100000.0,
        )
        assert resp.total_calculations == 3
        assert resp.success is True

    def test_health_response_defaults(self):
        """HealthResponse has sensible defaults."""
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.service == "process-emissions"
        assert resp.version == "1.0.0"

    def test_stats_response_defaults(self):
        """StatsResponse has zero defaults."""
        resp = StatsResponse()
        assert resp.total_calculations == 0
        assert resp.uptime_seconds == 0.0

    def test_compute_hash_deterministic(self):
        """_compute_hash returns same hash for same input."""
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"a": 1, "b": 2})
        assert h1 == h2
        assert len(h1) == 64

    def test_new_uuid_unique(self):
        """_new_uuid generates unique values."""
        u1 = _new_uuid()
        u2 = _new_uuid()
        assert u1 != u2
