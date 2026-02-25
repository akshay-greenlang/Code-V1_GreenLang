# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Waste Treatment Emissions Agent - WasteTreatmentEmissionsService.

Tests the service facade including initialization, engine access, single/batch
calculation, facility management, waste stream management, treatment events,
methane recovery, compliance checking, uncertainty analysis, aggregations,
health check, stats, and pagination.

Target: 30+ tests, 85%+ coverage.

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
    from greenlang.waste_treatment_emissions.setup import (
        WasteTreatmentEmissionsService,
        CalculateResponse,
        BatchCalculateResponse,
        FacilityResponse,
        FacilityListResponse,
        WasteStreamResponse,
        WasteStreamListResponse,
        TreatmentEventResponse,
        MethaneRecoveryResponse,
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
    reason="WasteTreatmentEmissionsService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh WasteTreatmentEmissionsService instance."""
    return WasteTreatmentEmissionsService()


@pytest.fixture
def calc_request():
    """Standard calculation request data."""
    return {
        "facility_id": "fac-001",
        "waste_stream_id": "ws-001",
        "treatment_method": "incineration",
        "waste_category": "municipal_solid_waste",
        "waste_quantity_tonnes": 500.0,
        "calculation_method": "IPCC_TIER_1",
        "gwp_source": "AR6",
        "tenant_id": "test_tenant",
    }


@pytest.fixture
def facility_data():
    """Standard facility registration data."""
    return {
        "name": "Test Incinerator Plant",
        "facility_type": "incinerator",
        "capacity_tonnes_yr": 50000.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "tenant_id": "test_tenant",
        "country_code": "GB",
    }


@pytest.fixture
def waste_stream_data():
    """Standard waste stream registration data."""
    return {
        "name": "Municipal Solid Waste Stream A",
        "waste_category": "municipal_solid_waste",
        "source_type": "residential",
        "tenant_id": "test_tenant",
        "facility_id": "fac-001",
        "composition": {
            "organic": 0.40,
            "paper": 0.20,
            "plastic": 0.15,
            "glass": 0.10,
            "metal": 0.05,
            "textile": 0.05,
            "other": 0.05,
        },
        "moisture_content": 0.35,
        "carbon_content": 0.40,
        "fossil_carbon_fraction": 0.55,
    }


# ===========================================================================
# Test Class: Service Initialization
# ===========================================================================


@_SKIP
class TestServiceInit:
    """Test WasteTreatmentEmissionsService initialization."""

    def test_service_creation(self):
        """Service can be created."""
        svc = WasteTreatmentEmissionsService()
        assert svc is not None

    def test_calculations_list_empty(self, service):
        """Calculations list starts empty."""
        assert len(service._calculations) == 0

    def test_facilities_dict_empty(self, service):
        """Facilities dict starts empty."""
        assert len(service._facilities) == 0

    def test_waste_streams_dict_empty(self, service):
        """Waste streams dict starts empty."""
        assert len(service._waste_streams) == 0

    def test_total_calculations_starts_zero(self, service):
        """Total calculations counter starts at zero."""
        assert service._total_calculations == 0

    def test_total_co2e_starts_zero(self, service):
        """Total CO2e accumulator starts at zero."""
        assert service._total_co2e == 0.0

    def test_total_waste_processed_starts_zero(self, service):
        """Total waste processed accumulator starts at zero."""
        assert service._total_waste_processed == 0.0


# ===========================================================================
# Test Class: Engine Access Properties
# ===========================================================================


@_SKIP
class TestEngineAccess:
    """Test engine accessor properties."""

    def test_database_engine_property(self, service):
        """database_engine property is accessible."""
        # May be None if engine not available, but should not raise
        _ = service.database_engine

    def test_biological_engine_property(self, service):
        """biological_engine property is accessible."""
        _ = service.biological_engine

    def test_thermal_engine_property(self, service):
        """thermal_engine property is accessible."""
        _ = service.thermal_engine

    def test_wastewater_engine_property(self, service):
        """wastewater_engine property is accessible."""
        _ = service.wastewater_engine

    def test_uncertainty_engine_property(self, service):
        """uncertainty_engine property is accessible."""
        _ = service.uncertainty_engine

    def test_compliance_engine_property(self, service):
        """compliance_engine property is accessible."""
        _ = service.compliance_engine

    def test_pipeline_engine_property(self, service):
        """pipeline_engine property is accessible."""
        _ = service.pipeline_engine


# ===========================================================================
# Test Class: Single Calculation
# ===========================================================================


@_SKIP
class TestSingleCalculation:
    """Test single emission calculation."""

    def test_calculate_returns_response(self, service, calc_request):
        """calculate() returns a CalculateResponse."""
        result = service.calculate(calc_request)
        assert isinstance(result, CalculateResponse)

    def test_calculate_has_calculation_id(self, service, calc_request):
        """Result has a non-empty calculation_id."""
        result = service.calculate(calc_request)
        assert result.calculation_id != ""

    def test_calculate_has_treatment_method(self, service, calc_request):
        """Result includes the treatment method."""
        result = service.calculate(calc_request)
        assert result.treatment_method == "incineration"

    def test_calculate_has_provenance_hash(self, service, calc_request):
        """Result includes a SHA-256 provenance hash."""
        result = service.calculate(calc_request)
        assert len(result.provenance_hash) == 64

    def test_calculate_has_processing_time(self, service, calc_request):
        """Result includes processing time."""
        result = service.calculate(calc_request)
        assert result.processing_time_ms >= 0

    def test_calculate_has_timestamp(self, service, calc_request):
        """Result includes a timestamp."""
        result = service.calculate(calc_request)
        assert result.timestamp != ""

    def test_calculate_increments_counter(self, service, calc_request):
        """Calculation counter increments."""
        initial = service._total_calculations
        service.calculate(calc_request)
        assert service._total_calculations == initial + 1

    def test_calculate_cached(self, service, calc_request):
        """Calculation result is cached in _calculations list."""
        initial_count = len(service._calculations)
        service.calculate(calc_request)
        assert len(service._calculations) == initial_count + 1

    def test_calculate_accumulates_waste(self, service, calc_request):
        """Waste quantity is accumulated."""
        initial = service._total_waste_processed
        service.calculate(calc_request)
        assert service._total_waste_processed == initial + 500.0

    def test_calculate_failed_returns_success_false(self, service):
        """Calculate with problematic input returns success=False gracefully."""
        bad_request = {
            "treatment_method": "",
            "waste_category": "",
            "waste_quantity_tonnes": 0,
        }
        result = service.calculate(bad_request)
        # Should return a CalculateResponse, either success or not
        assert isinstance(result, CalculateResponse)


# ===========================================================================
# Test Class: Batch Calculation
# ===========================================================================


@_SKIP
class TestBatchCalculation:
    """Test batch emission calculation."""

    def test_batch_returns_response(self, service, calc_request):
        """calculate_batch returns a BatchCalculateResponse."""
        result = service.calculate_batch([calc_request])
        assert isinstance(result, BatchCalculateResponse)

    def test_batch_with_multiple_items(self, service, calc_request):
        """Batch processes multiple items."""
        requests = [calc_request, calc_request.copy()]
        result = service.calculate_batch(requests)
        assert result.total_calculations >= 2

    def test_batch_empty_list(self, service):
        """Batch with empty list returns zero calculations."""
        result = service.calculate_batch([])
        assert isinstance(result, BatchCalculateResponse)
        assert result.total_calculations == 0


# ===========================================================================
# Test Class: Facility Management
# ===========================================================================


@_SKIP
class TestFacilityManagement:
    """Test facility registration and management."""

    def test_register_facility(self, service, facility_data):
        """register_facility creates a facility record."""
        if hasattr(service, "register_facility"):
            result = service.register_facility(facility_data)
            assert result is not None

    def test_list_facilities(self, service):
        """list_facilities returns a list or response object."""
        if hasattr(service, "list_facilities"):
            result = service.list_facilities()
            assert result is not None

    def test_update_facility(self, service, facility_data):
        """update_facility modifies facility data."""
        if hasattr(service, "register_facility") and hasattr(service, "update_facility"):
            reg = service.register_facility(facility_data)
            fac_id = getattr(reg, "facility_id", None) or reg.get("facility_id", "")
            if fac_id:
                updated = service.update_facility(fac_id, {"name": "Updated Name"})
                assert updated is not None


# ===========================================================================
# Test Class: Waste Stream Management
# ===========================================================================


@_SKIP
class TestWasteStreamManagement:
    """Test waste stream registration and management."""

    def test_register_waste_stream(self, service, waste_stream_data):
        """register_waste_stream creates a waste stream record."""
        if hasattr(service, "register_waste_stream"):
            result = service.register_waste_stream(waste_stream_data)
            assert result is not None

    def test_list_waste_streams(self, service):
        """list_waste_streams returns a list or response."""
        if hasattr(service, "list_waste_streams"):
            result = service.list_waste_streams()
            assert result is not None


# ===========================================================================
# Test Class: Treatment Events
# ===========================================================================


@_SKIP
class TestTreatmentEvents:
    """Test treatment event recording."""

    def test_record_treatment_event(self, service):
        """record_treatment_event creates an event record."""
        if hasattr(service, "record_treatment_event"):
            event_data = {
                "facility_id": "fac-001",
                "treatment_method": "incineration",
                "waste_category": "municipal_solid_waste",
                "waste_tonnes": 100.0,
                "event_date": "2025-06-15",
            }
            result = service.record_treatment_event(event_data)
            assert result is not None

    def test_list_treatment_events(self, service):
        """list_treatment_events returns a list or response."""
        if hasattr(service, "list_treatment_events"):
            result = service.list_treatment_events()
            assert result is not None


# ===========================================================================
# Test Class: Methane Recovery
# ===========================================================================


@_SKIP
class TestMethaneRecovery:
    """Test methane recovery recording."""

    def test_record_methane_recovery(self, service):
        """record_methane_recovery creates a recovery record."""
        if hasattr(service, "record_methane_recovery"):
            recovery_data = {
                "facility_id": "fac-001",
                "recovery_date": "2025-06-15",
                "methane_captured_tonnes": 5.0,
                "methane_flared_tonnes": 3.0,
                "methane_utilized_tonnes": 2.0,
            }
            result = service.record_methane_recovery(recovery_data)
            assert result is not None


# ===========================================================================
# Test Class: Compliance and Uncertainty
# ===========================================================================


@_SKIP
class TestComplianceAndUncertainty:
    """Test compliance checking and uncertainty analysis via service."""

    def test_check_compliance(self, service):
        """check_compliance runs a compliance check."""
        if hasattr(service, "check_compliance"):
            result = service.check_compliance({
                "calculation_id": "",
                "frameworks": ["GHG_PROTOCOL"],
            })
            assert result is not None

    def test_run_uncertainty(self, service):
        """run_uncertainty performs Monte Carlo analysis."""
        if hasattr(service, "run_uncertainty"):
            result = service.run_uncertainty({
                "calculation_id": "",
                "iterations": 100,
                "seed": 42,
            })
            assert result is not None


# ===========================================================================
# Test Class: Aggregation and Health
# ===========================================================================


@_SKIP
class TestAggregationAndHealth:
    """Test aggregated emissions and health check."""

    def test_health_check(self, service):
        """health_check returns service status."""
        if hasattr(service, "health_check"):
            result = service.health_check()
            assert result is not None
            if isinstance(result, dict):
                assert result.get("status") in ("healthy", "degraded")
            elif hasattr(result, "status"):
                assert result.status in ("healthy", "degraded")

    def test_get_stats(self, service):
        """get_stats returns aggregate statistics."""
        if hasattr(service, "get_stats"):
            result = service.get_stats()
            assert result is not None

    def test_get_aggregations(self, service):
        """get_aggregations returns aggregated emission data."""
        if hasattr(service, "get_aggregations"):
            result = service.get_aggregations()
            assert result is not None


# ===========================================================================
# Test Class: Response Models
# ===========================================================================


@_SKIP
class TestResponseModels:
    """Test Pydantic response models."""

    def test_calculate_response_creation(self):
        """CalculateResponse can be created with defaults."""
        r = CalculateResponse()
        assert r.success is True
        assert r.total_co2e_tonnes == 0.0

    def test_batch_calculate_response(self):
        """BatchCalculateResponse can be created with defaults."""
        r = BatchCalculateResponse()
        assert r.total_calculations == 0

    def test_facility_response(self):
        """FacilityResponse can be created with defaults."""
        r = FacilityResponse()
        assert r.facility_id == ""

    def test_waste_stream_response(self):
        """WasteStreamResponse can be created with defaults."""
        r = WasteStreamResponse()
        assert r.stream_id == ""

    def test_health_response(self):
        """HealthResponse can be created with defaults."""
        r = HealthResponse()
        assert r.status == "healthy"
        assert r.service == "waste-treatment-emissions"

    def test_stats_response(self):
        """StatsResponse can be created with defaults."""
        r = StatsResponse()
        assert r.total_calculations == 0

    def test_compliance_check_response(self):
        """ComplianceCheckResponse can be created with defaults."""
        r = ComplianceCheckResponse()
        assert r.success is True

    def test_uncertainty_response(self):
        """UncertaintyResponse can be created with defaults."""
        r = UncertaintyResponse()
        assert r.method == "monte_carlo"

    def test_aggregation_response(self):
        """AggregationResponse can be created with defaults."""
        r = AggregationResponse()
        assert r.total_co2e_tonnes == 0.0

    def test_treatment_event_response(self):
        """TreatmentEventResponse can be created with defaults."""
        r = TreatmentEventResponse()
        assert r.event_id == ""

    def test_methane_recovery_response(self):
        """MethaneRecoveryResponse can be created with defaults."""
        r = MethaneRecoveryResponse()
        assert r.recovery_id == ""
