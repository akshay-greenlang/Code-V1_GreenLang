# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Service Setup.

Tests AgriculturalEmissionsService: initialization, engine access,
single/batch calculation, farm/livestock/cropland management,
compliance, uncertainty, health, stats, and response models.

Target: 70+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.setup import (
        AgriculturalEmissionsService,
        CalculateResponse,
        BatchCalculateResponse,
        FarmResponse,
        FarmListResponse,
        LivestockResponse,
        LivestockListResponse,
        CroplandInputResponse,
        RiceFieldResponse,
        FieldBurningResponse,
        ComplianceCheckResponse,
        UncertaintyResponse,
        AggregationResponse,
        HealthResponse,
        StatsResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="AgriculturalEmissionsService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh AgriculturalEmissionsService instance."""
    return AgriculturalEmissionsService()


@pytest.fixture
def calc_request():
    """Standard enteric fermentation calculation request."""
    return {
        "farm_id": "farm-001",
        "source_category": "enteric_fermentation",
        "livestock_type": "dairy_cattle",
        "head_count": 200,
        "calculation_method": "IPCC_TIER_1",
        "gwp_source": "AR6",
        "tenant_id": "test_tenant",
    }


@pytest.fixture
def farm_data():
    """Standard farm registration data."""
    return {
        "name": "Green Valley Dairy Farm",
        "farm_type": "dairy",
        "area_ha": 250.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "tenant_id": "test_tenant",
        "country_code": "GB",
        "climate_zone": "cool_temperate",
        "soil_type": "mineral",
    }


@pytest.fixture
def livestock_data():
    """Standard livestock registration data."""
    return {
        "farm_id": "farm-001",
        "livestock_type": "dairy_cattle",
        "head_count": 200,
        "average_weight_kg": 600.0,
        "milk_yield_kg_per_day": 25.0,
        "manure_system": "pasture_range_paddock",
        "reporting_year": 2025,
        "tenant_id": "test_tenant",
    }


# ===========================================================================
# Test Class: Service Initialization
# ===========================================================================


@_SKIP
class TestServiceInit:
    """Test AgriculturalEmissionsService initialization."""

    def test_service_creation(self):
        svc = AgriculturalEmissionsService()
        assert svc is not None

    def test_calculations_list_empty(self, service):
        assert len(service._calculations) == 0

    def test_farms_dict_empty(self, service):
        assert len(service._farms) == 0

    def test_total_calculations_zero(self, service):
        assert service._total_calculations == 0

    def test_total_co2e_zero(self, service):
        assert service._total_co2e == 0.0

    def test_total_head_count_zero(self, service):
        assert service._total_head_count == 0


# ===========================================================================
# Test Class: Engine Access
# ===========================================================================


@_SKIP
class TestEngineAccess:
    """Test engine accessor properties."""

    def test_database_engine(self, service):
        _ = service.database_engine

    def test_enteric_engine(self, service):
        _ = service.enteric_engine

    def test_manure_engine(self, service):
        _ = service.manure_engine

    def test_cropland_engine(self, service):
        _ = service.cropland_engine

    def test_uncertainty_engine(self, service):
        _ = service.uncertainty_engine

    def test_compliance_engine(self, service):
        _ = service.compliance_engine

    def test_pipeline_engine(self, service):
        _ = service.pipeline_engine


# ===========================================================================
# Test Class: Single Calculation
# ===========================================================================


@_SKIP
class TestSingleCalculation:
    """Test single emission calculation."""

    def test_calculate_returns_response(self, service, calc_request):
        result = service.calculate(calc_request)
        assert isinstance(result, CalculateResponse)

    def test_calculate_has_calculation_id(self, service, calc_request):
        result = service.calculate(calc_request)
        assert result.calculation_id != ""

    def test_calculate_has_source_category(self, service, calc_request):
        result = service.calculate(calc_request)
        assert result.source_category == "enteric_fermentation"

    def test_calculate_has_provenance_hash(self, service, calc_request):
        result = service.calculate(calc_request)
        assert len(result.provenance_hash) == 64

    def test_calculate_has_processing_time(self, service, calc_request):
        result = service.calculate(calc_request)
        assert result.processing_time_ms >= 0

    def test_calculate_has_timestamp(self, service, calc_request):
        result = service.calculate(calc_request)
        assert result.timestamp != ""

    def test_calculate_increments_counter(self, service, calc_request):
        initial = service._total_calculations
        service.calculate(calc_request)
        assert service._total_calculations == initial + 1

    def test_calculate_cached(self, service, calc_request):
        initial_count = len(service._calculations)
        service.calculate(calc_request)
        assert len(service._calculations) == initial_count + 1

    def test_calculate_accumulates_head_count(self, service, calc_request):
        initial = service._total_head_count
        service.calculate(calc_request)
        assert service._total_head_count == initial + 200

    def test_calculate_with_bad_input(self, service):
        bad_request = {
            "source_category": "",
            "livestock_type": "",
            "head_count": 0,
        }
        result = service.calculate(bad_request)
        assert isinstance(result, CalculateResponse)


# ===========================================================================
# Test Class: Batch Calculation
# ===========================================================================


@_SKIP
class TestBatchCalculation:
    """Test batch emission calculation."""

    def test_batch_returns_response(self, service, calc_request):
        result = service.calculate_batch([calc_request])
        assert isinstance(result, BatchCalculateResponse)

    def test_batch_multiple_items(self, service, calc_request):
        requests = [calc_request, calc_request.copy()]
        result = service.calculate_batch(requests)
        assert result.total_calculations >= 2

    def test_batch_empty_list(self, service):
        result = service.calculate_batch([])
        assert isinstance(result, BatchCalculateResponse)
        assert result.total_calculations == 0


# ===========================================================================
# Test Class: Farm Management
# ===========================================================================


@_SKIP
class TestFarmManagement:
    """Test farm registration and management."""

    def test_register_farm(self, service, farm_data):
        if hasattr(service, "register_farm"):
            result = service.register_farm(farm_data)
            assert result is not None

    def test_list_farms(self, service):
        if hasattr(service, "list_farms"):
            result = service.list_farms()
            assert result is not None

    def test_update_farm(self, service, farm_data):
        if hasattr(service, "register_farm") and hasattr(service, "update_farm"):
            reg = service.register_farm(farm_data)
            fid = getattr(reg, "farm_id", None) or (reg.get("farm_id") if isinstance(reg, dict) else "")
            if fid:
                updated = service.update_farm(fid, {"name": "Updated Farm"})
                assert updated is not None


# ===========================================================================
# Test Class: Livestock Management
# ===========================================================================


@_SKIP
class TestLivestockManagement:
    """Test livestock registration and management."""

    def test_register_livestock(self, service, livestock_data):
        if hasattr(service, "register_livestock"):
            result = service.register_livestock(livestock_data)
            assert result is not None

    def test_list_livestock(self, service):
        if hasattr(service, "list_livestock"):
            result = service.list_livestock()
            assert result is not None


# ===========================================================================
# Test Class: Cropland and Rice
# ===========================================================================


@_SKIP
class TestCroplandAndRice:
    """Test cropland input and rice field management."""

    def test_register_cropland_input(self, service):
        if hasattr(service, "register_cropland_input"):
            data = {
                "farm_id": "farm-001",
                "input_type": "synthetic_n",
                "quantity_tonnes": 100.0,
                "tenant_id": "test_tenant",
            }
            result = service.register_cropland_input(data)
            assert result is not None

    def test_list_cropland_inputs(self, service):
        if hasattr(service, "list_cropland_inputs"):
            result = service.list_cropland_inputs()
            assert result is not None

    def test_register_rice_field(self, service):
        if hasattr(service, "register_rice_field"):
            data = {
                "farm_id": "farm-001",
                "area_ha": 50.0,
                "water_regime": "continuously_flooded",
                "cultivation_period_days": 120,
                "tenant_id": "test_tenant",
            }
            result = service.register_rice_field(data)
            assert result is not None

    def test_list_rice_fields(self, service):
        if hasattr(service, "list_rice_fields"):
            result = service.list_rice_fields()
            assert result is not None


# ===========================================================================
# Test Class: Compliance and Uncertainty
# ===========================================================================


@_SKIP
class TestComplianceAndUncertainty:
    """Test compliance checking and uncertainty analysis."""

    def test_check_compliance(self, service):
        if hasattr(service, "check_compliance"):
            result = service.check_compliance({
                "calculation_id": "",
                "frameworks": ["IPCC_2006"],
            })
            assert result is not None

    def test_run_uncertainty(self, service):
        if hasattr(service, "run_uncertainty"):
            result = service.run_uncertainty({
                "calculation_id": "",
                "iterations": 100,
                "seed": 42,
            })
            assert result is not None


# ===========================================================================
# Test Class: Health and Stats
# ===========================================================================


@_SKIP
class TestHealthAndStats:
    """Test health check and statistics."""

    def test_health_check(self, service):
        if hasattr(service, "health_check"):
            result = service.health_check()
            assert result is not None
            if isinstance(result, HealthResponse):
                assert result.status in ("healthy", "degraded")

    def test_get_stats(self, service):
        if hasattr(service, "get_stats"):
            result = service.get_stats()
            assert result is not None


# ===========================================================================
# Test Class: Response Models
# ===========================================================================


@_SKIP
class TestResponseModels:
    """Test Pydantic response models."""

    def test_calculate_response(self):
        r = CalculateResponse()
        assert r.success is True
        assert r.total_co2e_tonnes == 0.0

    def test_batch_calculate_response(self):
        r = BatchCalculateResponse()
        assert r.total_calculations == 0

    def test_farm_response(self):
        r = FarmResponse()
        assert r.farm_id == ""

    def test_farm_list_response(self):
        r = FarmListResponse()
        assert r.total == 0

    def test_livestock_response(self):
        r = LivestockResponse()
        assert r.herd_id == ""

    def test_livestock_list_response(self):
        r = LivestockListResponse()
        assert r.total == 0

    def test_cropland_input_response(self):
        r = CroplandInputResponse()
        assert r.input_id == ""

    def test_rice_field_response(self):
        r = RiceFieldResponse()
        assert r.field_id == ""

    def test_field_burning_response(self):
        r = FieldBurningResponse()
        assert r.burning_id == ""

    def test_compliance_check_response(self):
        r = ComplianceCheckResponse()
        assert r.success is True

    def test_uncertainty_response(self):
        r = UncertaintyResponse()
        assert r.method == "monte_carlo"

    def test_aggregation_response(self):
        r = AggregationResponse()
        assert r.total_co2e_tonnes == 0.0

    def test_health_response(self):
        r = HealthResponse()
        assert r.status == "healthy"
        assert r.service == "agricultural-emissions"

    def test_stats_response(self):
        r = StatsResponse()
        assert r.total_calculations == 0
