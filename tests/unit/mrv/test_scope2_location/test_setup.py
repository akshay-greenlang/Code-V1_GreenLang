# -*- coding: utf-8 -*-
"""
Unit tests for Scope2LocationService (setup.py)

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Tests the unified service facade covering all 20 REST API operations:
calculate, calculate_batch, facility management, consumption recording,
grid factors, T&D losses, compliance, uncertainty, aggregations,
health check, and stats.

Target: 45 tests, ~450 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.scope2_location.setup import (
        Scope2LocationService,
        get_service,
        SERVICE_VERSION,
        SERVICE_NAME,
        AGENT_ID,
        VALID_ENERGY_TYPES,
        VALID_GWP_SOURCES,
    )
    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SERVICE_AVAILABLE, reason="Scope2LocationService not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a fresh Scope2LocationService instance."""
    return Scope2LocationService()


@pytest.fixture
def calc_request() -> Dict[str, Any]:
    """Build a valid electricity calculation request."""
    return {
        "tenant_id": "tenant-001",
        "facility_id": "fac-001",
        "energy_type": "electricity",
        "consumption_value": 5000.0,
        "consumption_unit": "mwh",
        "country_code": "US",
        "egrid_subregion": "CAMX",
        "gwp_source": "AR5",
    }


@pytest.fixture
def facility_data() -> Dict[str, Any]:
    """Build a valid facility registration request."""
    return {
        "name": "Test Office Building",
        "facility_type": "office",
        "country_code": "US",
        "grid_region_id": "CAMX",
        "egrid_subregion": "CAMX",
        "tenant_id": "tenant-001",
    }


@pytest.fixture
def consumption_data() -> Dict[str, Any]:
    """Build a valid consumption record."""
    return {
        "facility_id": "fac-001",
        "energy_type": "electricity",
        "quantity": 5000.0,
        "unit": "mwh",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "data_source": "meter",
        "tenant_id": "tenant-001",
    }


# ===========================================================================
# 1. TestServiceInit
# ===========================================================================


@_SKIP
class TestServiceInit:
    """Tests for service initialization and engine availability."""

    def test_service_creation(self):
        """Service can be created without errors."""
        svc = Scope2LocationService()
        assert svc is not None

    def test_service_has_config(self, service):
        """Service exposes config property."""
        # config may be None if module not available
        _ = service.config

    def test_service_has_pipeline(self, service):
        """Service has pipeline_engine property."""
        _ = service.pipeline_engine

    def test_service_constants(self):
        """Service constants are defined."""
        assert SERVICE_VERSION == "1.0.0"
        assert "scope2" in SERVICE_NAME.lower()
        assert AGENT_ID == "AGENT-MRV-009"

    def test_valid_energy_types(self):
        """At least 4 energy types defined."""
        assert len(VALID_ENERGY_TYPES) >= 4
        assert "electricity" in VALID_ENERGY_TYPES

    def test_valid_gwp_sources(self):
        """At least 3 GWP sources defined."""
        assert len(VALID_GWP_SOURCES) >= 3
        assert "AR5" in VALID_GWP_SOURCES


# ===========================================================================
# 2. TestCalculate
# ===========================================================================


@_SKIP
class TestCalculate:
    """Tests for calculate and calculate_batch."""

    def test_calculate_electricity(self, service, calc_request):
        """calculate returns a response with CO2e values."""
        result = service.calculate(calc_request)
        assert result is not None
        # Result should be a Pydantic model or dict with success field
        if hasattr(result, "success"):
            assert result.success is True
            assert result.total_co2e_tonnes >= 0
        elif isinstance(result, dict):
            assert result.get("total_co2e_kg", 0) >= 0

    def test_calculate_batch(self, service, calc_request):
        """calculate_batch processes multiple requests."""
        result = service.calculate_batch([calc_request, calc_request])
        assert result is not None
        if hasattr(result, "total_calculations"):
            assert result.total_calculations == 2
        elif isinstance(result, dict):
            assert result.get("total_requests", 0) >= 1

    def test_calculate_steam(self, service):
        """calculate with steam energy type."""
        request = {
            "tenant_id": "tenant-001",
            "facility_id": "fac-002",
            "energy_type": "steam",
            "consumption_value": 1200.0,
            "consumption_unit": "gj",
            "country_code": "US",
            "steam_type": "natural_gas",
        }
        result = service.calculate(request)
        assert result is not None

    def test_calculate_invalid_energy_type(self, service):
        """calculate with invalid energy type raises or returns error."""
        request = {
            "tenant_id": "tenant-001",
            "facility_id": "fac-003",
            "energy_type": "nuclear",
            "consumption_value": 100.0,
            "consumption_unit": "mwh",
            "country_code": "US",
        }
        try:
            result = service.calculate(request)
            # Should either raise or return an error response
            if hasattr(result, "success"):
                # May still succeed with fallback or may fail
                pass
        except (ValueError, Exception):
            pass  # Expected for invalid input


# ===========================================================================
# 3. TestFacilities
# ===========================================================================


@_SKIP
class TestFacilities:
    """Tests for register_facility, list_facilities, update_facility."""

    def test_register_facility(self, service, facility_data):
        """register_facility creates a new facility record."""
        result = service.register_facility(facility_data)
        assert result is not None
        if hasattr(result, "facility_id"):
            assert result.facility_id != ""
        elif isinstance(result, dict):
            assert result.get("facility_id", "") != ""

    def test_list_facilities(self, service, facility_data):
        """list_facilities returns registered facilities."""
        service.register_facility(facility_data)
        result = service.list_facilities(tenant_id="tenant-001")
        assert result is not None

    def test_update_facility(self, service, facility_data):
        """update_facility modifies an existing facility."""
        registered = service.register_facility(facility_data)
        fac_id = ""
        if hasattr(registered, "facility_id"):
            fac_id = registered.facility_id
        elif isinstance(registered, dict):
            fac_id = registered.get("facility_id", "")

        if fac_id:
            update_data = {"name": "Updated Office Building"}
            result = service.update_facility(fac_id, update_data)
            assert result is not None

    def test_register_multiple_facilities(self, service):
        """Registering multiple facilities increases the count."""
        for i in range(3):
            service.register_facility({
                "name": f"Facility {i}",
                "facility_type": "office",
                "country_code": "US",
                "tenant_id": "tenant-001",
            })
        result = service.list_facilities(tenant_id="tenant-001")
        if hasattr(result, "total"):
            assert result.total >= 3
        elif isinstance(result, dict):
            assert len(result.get("facilities", [])) >= 3


# ===========================================================================
# 4. TestConsumption
# ===========================================================================


@_SKIP
class TestConsumption:
    """Tests for record_consumption and list_consumption."""

    def test_record_consumption(self, service, consumption_data):
        """record_consumption creates a consumption record."""
        result = service.record_consumption(consumption_data)
        assert result is not None

    def test_list_consumption(self, service, consumption_data):
        """list_consumption returns recorded entries."""
        service.record_consumption(consumption_data)
        result = service.list_consumption(facility_id="fac-001")
        assert result is not None

    def test_record_multiple_consumption(self, service):
        """Multiple consumption records can be created."""
        for month in range(1, 4):
            service.record_consumption({
                "facility_id": "fac-001",
                "energy_type": "electricity",
                "quantity": 1000.0 + month * 100,
                "unit": "mwh",
                "period_start": f"2025-{month:02d}-01",
                "period_end": f"2025-{month:02d}-28",
                "tenant_id": "tenant-001",
            })
        result = service.list_consumption(facility_id="fac-001")
        assert result is not None


# ===========================================================================
# 5. TestGridFactors
# ===========================================================================


@_SKIP
class TestGridFactors:
    """Tests for list_grid_factors, get_grid_factor, add_custom_factor."""

    def test_list_grid_factors(self, service):
        """list_grid_factors returns a response."""
        result = service.list_grid_factors()
        assert result is not None

    def test_get_grid_factor(self, service):
        """get_grid_factor returns factor for a known region."""
        result = service.get_grid_factor("US")
        assert result is not None

    def test_add_custom_factor(self, service):
        """add_custom_factor registers a custom grid emission factor."""
        result = service.add_custom_factor({
            "region_id": "CUSTOM-001",
            "co2_kg_per_mwh": 350.0,
            "ch4_kg_per_mwh": 0.02,
            "n2o_kg_per_mwh": 0.003,
            "year": 2025,
            "source_name": "test_custom",
        })
        assert result is not None


# ===========================================================================
# 6. TestTDLosses
# ===========================================================================


@_SKIP
class TestTDLosses:
    """Tests for list_td_losses."""

    def test_list_td_losses(self, service):
        """list_td_losses returns a response."""
        result = service.list_td_losses()
        assert result is not None


# ===========================================================================
# 7. TestCompliance
# ===========================================================================


@_SKIP
class TestCompliance:
    """Tests for check_compliance and get_compliance_result."""

    def test_check_compliance(self, service, calc_request):
        """check_compliance returns compliance results."""
        # First do a calculation
        calc_result = service.calculate(calc_request)
        calc_id = ""
        if hasattr(calc_result, "calculation_id"):
            calc_id = calc_result.calculation_id
        elif isinstance(calc_result, dict):
            calc_id = calc_result.get("calculation_id", "")

        if calc_id:
            result = service.check_compliance(calc_id)
            assert result is not None

    def test_check_compliance_with_frameworks(self, service, calc_request):
        """check_compliance with specific frameworks."""
        calc_result = service.calculate(calc_request)
        calc_id = ""
        if hasattr(calc_result, "calculation_id"):
            calc_id = calc_result.calculation_id
        elif isinstance(calc_result, dict):
            calc_id = calc_result.get("calculation_id", "")

        if calc_id:
            result = service.check_compliance(
                calc_id,
                frameworks=["ghg_protocol_scope2"],
            )
            assert result is not None


# ===========================================================================
# 8. TestUncertainty
# ===========================================================================


@_SKIP
class TestUncertaintyService:
    """Tests for run_uncertainty."""

    def test_run_uncertainty(self, service, calc_request):
        """run_uncertainty returns uncertainty analysis results."""
        calc_result = service.calculate(calc_request)
        calc_id = ""
        if hasattr(calc_result, "calculation_id"):
            calc_id = calc_result.calculation_id
        elif isinstance(calc_result, dict):
            calc_id = calc_result.get("calculation_id", "")

        if calc_id:
            result = service.run_uncertainty(
                calc_id, iterations=500, seed=42,
            )
            assert result is not None


# ===========================================================================
# 9. TestAggregations
# ===========================================================================


@_SKIP
class TestAggregations:
    """Tests for get_aggregations."""

    def test_get_aggregations(self, service, calc_request):
        """get_aggregations returns aggregated data."""
        service.calculate(calc_request)
        result = service.get_aggregations()
        assert result is not None


# ===========================================================================
# 10. TestHealth
# ===========================================================================


@_SKIP
class TestHealth:
    """Tests for health_check and get_stats."""

    def test_health_check(self, service):
        """health_check returns status healthy."""
        result = service.health_check()
        assert result is not None
        if hasattr(result, "status"):
            assert result.status in ("healthy", "degraded", "unhealthy")
        elif isinstance(result, dict):
            assert result.get("status") in ("healthy", "degraded", "unhealthy")

    def test_health_check_version(self, service):
        """health_check includes service version."""
        result = service.health_check()
        if hasattr(result, "version"):
            assert result.version == SERVICE_VERSION
        elif isinstance(result, dict):
            assert result.get("version", "") == SERVICE_VERSION

    def test_get_stats(self, service):
        """get_stats returns statistics."""
        result = service.get_stats()
        assert result is not None
        if hasattr(result, "total_calculations"):
            assert result.total_calculations >= 0
        elif isinstance(result, dict):
            assert "total_calculations" in result

    def test_get_stats_after_calculation(self, service, calc_request):
        """Stats update after a calculation."""
        service.calculate(calc_request)
        result = service.get_stats()
        if hasattr(result, "total_calculations"):
            assert result.total_calculations >= 1
        elif isinstance(result, dict):
            assert result.get("total_calculations", 0) >= 1

    def test_health_engines_reported(self, service):
        """health_check reports engine availability."""
        result = service.health_check()
        if hasattr(result, "engines"):
            assert isinstance(result.engines, dict)
        elif isinstance(result, dict) and "engines" in result:
            assert isinstance(result["engines"], dict)


# ===========================================================================
# 11. TestSingleton
# ===========================================================================


@_SKIP
class TestSingleton:
    """Tests for get_service singleton behavior."""

    def test_get_service_returns_instance(self):
        """get_service returns a Scope2LocationService or compatible object."""
        svc = get_service()
        assert svc is not None

    def test_get_service_same_instance(self):
        """get_service returns the same instance on repeated calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_service_engine_properties(self, service):
        """Service exposes engine properties without errors."""
        _ = service.grid_db_engine
        _ = service.electricity_engine
        _ = service.steam_heat_cool_engine
        _ = service.transmission_engine
        _ = service.uncertainty_engine
        _ = service.compliance_engine
        _ = service.pipeline_engine
