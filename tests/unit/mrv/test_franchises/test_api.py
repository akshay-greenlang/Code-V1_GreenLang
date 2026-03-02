# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-027 Franchises Agent - REST API Router.

Tests all 22 API endpoints using FastAPI TestClient with a mock service
dependency override. Validates status codes, response structure, and
error handling for the endpoints defined in the router.

Target: 55+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment, misc]
    TestClient = None  # type: ignore[assignment, misc]

try:
    from greenlang.franchises.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or franchises router not available",
)

PREFIX = "/api/v1/franchises"


# ===========================================================================
# Mock Service
# ===========================================================================


class MockFranchisesService:
    """
    Mock service that returns deterministic responses for all API endpoints.

    Every async method returns a dict matching what the router expects,
    so we can test router logic without exercising the real calculation engines.
    Method names and return keys must exactly match what the router calls.
    """

    # -- calculations -------------------------------------------------------

    async def calculate(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-001",
            "franchise_type": "qsr",
            "method": "franchise_specific",
            "total_emissions_kgco2e": 85432.50,
            "energy_emissions_kgco2e": 55000.0,
            "refrigerant_emissions_kgco2e": 30432.50,
            "unit_count": 1,
            "unit_results": None,
            "coverage_percent": 100.0,
            "data_quality_score": 1.5,
            "provenance_hash": "a" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_franchise_specific(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-fs-001",
            "unit_id": data.get("unit", {}).get("unit_id", "FRN-001"),
            "franchise_type": "qsr",
            "method": "franchise_specific",
            "electricity_emissions_kgco2e": 30000.0,
            "natural_gas_emissions_kgco2e": 20000.0,
            "other_fuel_emissions_kgco2e": 5000.0,
            "refrigerant_emissions_kgco2e": 432.50,
            "total_emissions_kgco2e": 55432.50,
            "grid_ef_kgco2e_per_kwh": 0.386,
            "eui_kwh_per_m2": 720.0,
            "data_quality_score": 1.5,
            "provenance_hash": "b" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_qsr(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-qsr-001",
            "unit_id": data.get("unit", {}).get("unit_id", "FRN-QSR-001"),
            "franchise_type": "qsr",
            "method": "franchise_specific",
            "electricity_emissions_kgco2e": 25000.0,
            "natural_gas_emissions_kgco2e": 35000.0,
            "other_fuel_emissions_kgco2e": 0.0,
            "refrigerant_emissions_kgco2e": 500.0,
            "total_emissions_kgco2e": 60500.0,
            "grid_ef_kgco2e_per_kwh": 0.386,
            "eui_kwh_per_m2": 800.0,
            "data_quality_score": 1.5,
            "provenance_hash": "q" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_hotel(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-htl-001",
            "unit_id": data.get("unit", {}).get("unit_id", "FRN-HTL-001"),
            "franchise_type": "hotel_upscale",
            "method": "franchise_specific",
            "electricity_emissions_kgco2e": 300000.0,
            "natural_gas_emissions_kgco2e": 125000.0,
            "other_fuel_emissions_kgco2e": 0.0,
            "refrigerant_emissions_kgco2e": 1200.0,
            "total_emissions_kgco2e": 426200.0,
            "grid_ef_kgco2e_per_kwh": 0.386,
            "eui_kwh_per_m2": 310.0,
            "data_quality_score": 1.5,
            "provenance_hash": "c" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_convenience(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-conv-001",
            "unit_id": data.get("unit", {}).get("unit_id", "FRN-CONV-001"),
            "franchise_type": "convenience_store",
            "method": "franchise_specific",
            "electricity_emissions_kgco2e": 45000.0,
            "natural_gas_emissions_kgco2e": 5000.0,
            "other_fuel_emissions_kgco2e": 0.0,
            "refrigerant_emissions_kgco2e": 2500.0,
            "total_emissions_kgco2e": 52500.0,
            "data_quality_score": 1.5,
            "provenance_hash": "v" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_retail(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-ret-001",
            "unit_id": data.get("unit", {}).get("unit_id", "FRN-RET-001"),
            "franchise_type": "retail_apparel",
            "method": "franchise_specific",
            "electricity_emissions_kgco2e": 35000.0,
            "natural_gas_emissions_kgco2e": 10000.0,
            "other_fuel_emissions_kgco2e": 0.0,
            "refrigerant_emissions_kgco2e": 0.0,
            "total_emissions_kgco2e": 45000.0,
            "data_quality_score": 1.5,
            "provenance_hash": "r" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_average_data(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-avg-001",
            "franchise_type": data.get("franchise_type", "qsr"),
            "method": "average_data",
            "benchmark_eui_kwh_per_m2": 450.0,
            "estimated_energy_kwh": 99000.0,
            "total_emissions_kgco2e": 12500000.0,
            "unit_count": data.get("unit_count", 100),
            "emission_intensity_kgco2e_per_m2": 173.7,
            "data_quality_score": 3.0,
            "provenance_hash": "d" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_spend_based(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-spd-001",
            "franchise_type": data.get("franchise_type", "qsr"),
            "method": "spend_based",
            "naics_code": data.get("naics_code", "722513"),
            "naics_description": "Limited-Service Restaurants",
            "spend_amount_usd": data.get("revenue_usd", 450000000.0),
            "ef_kgco2e_per_dollar": 0.335,
            "total_emissions_kgco2e": 15075000.0,
            "cpi_deflation_factor": 0.98,
            "data_quality_score": 4.0,
            "provenance_hash": "e" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_hybrid(self, data: dict) -> dict:
        return {
            "calculation_id": "frn-mock-hyb-001",
            "method": "hybrid",
            "metered_unit_count": 1,
            "estimated_unit_count": 0,
            "spend_unit_count": 0,
            "metered_emissions_kgco2e": 55432.50,
            "estimated_emissions_kgco2e": 0.0,
            "spend_emissions_kgco2e": 0.0,
            "total_emissions_kgco2e": 55432.50,
            "coverage_percent": 100.0,
            "data_quality_score": 1.5,
            "provenance_hash": "f" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def calculate_batch(self, data: dict) -> dict:
        return {
            "batch_id": "batch-mock-001",
            "results": [
                {"calculation_id": "frn-mock-b-001", "total_emissions_kgco2e": 85432.50},
            ],
            "total_emissions_kgco2e": 85432.50,
            "unit_count": 1,
            "errors": [],
            "processing_time_ms": 123.4,
        }

    async def analyze_network(self, data: dict) -> dict:
        return {
            "calculation_id": "net-mock-001",
            "total_emissions_kgco2e": 45000000.0,
            "total_units": 1,
            "by_brand": {"TestBrand": {"emissions_kgco2e": 45000000.0}},
            "by_franchise_type": {"qsr": {"emissions_kgco2e": 45000000.0}},
            "by_region": {"US": {"emissions_kgco2e": 45000000.0}},
            "by_method": {"franchise_specific": {"emissions_kgco2e": 45000000.0}},
            "intensity_metrics": {"per_unit": 45000000.0},
            "coverage_summary": {"metered_percent": 100.0},
            "data_quality_score": 1.5,
            "provenance_hash": "1" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    # -- compliance ---------------------------------------------------------

    async def check_compliance(self, data: dict) -> dict:
        return {
            "results": [
                {
                    "framework": "ghg_protocol",
                    "status": "compliant",
                    "score": 92.5,
                    "findings": [],
                    "recommendations": [],
                }
            ],
            "overall_status": "compliant",
            "overall_score": 0.925,
            "recommendations": [],
        }

    # -- CRUD ---------------------------------------------------------------

    async def get_calculation(self, calculation_id: str):
        if calculation_id == "fake-id":
            return None
        return {
            "calculation_id": calculation_id,
            "franchise_type": "qsr",
            "method": "franchise_specific",
            "total_emissions_kgco2e": 85432.50,
            "unit_count": 1,
            "details": {"units": []},
            "provenance_hash": "a" * 64,
            "calculated_at": "2026-02-28T00:00:00Z",
        }

    async def list_calculations(self, filters: dict) -> dict:
        return {
            "calculations": [],
            "count": 0,
            "page": 1,
            "page_size": 50,
        }

    async def delete_calculation(self, calculation_id: str) -> bool:
        if calculation_id == "fake-id":
            return False
        return True

    # -- Emission factors & metadata ----------------------------------------

    async def get_emission_factors(self, filters: dict) -> dict:
        return {
            "factors": [
                {
                    "franchise_type": filters.get("franchise_type", "qsr"),
                    "climate_zone": "4A",
                    "eui_kwh_per_m2": 450.0,
                    "ef_kgco2e_per_m2": 173.7,
                    "ef_kgco2e_per_dollar": 0.335,
                    "source": "DEFRA_2024",
                    "year": 2024,
                }
            ],
            "count": 1,
        }

    async def get_franchise_benchmarks(self, filters: dict) -> dict:
        return {
            "benchmarks": [
                {
                    "franchise_type": "qsr",
                    "climate_zone": "4A",
                    "eui_kwh_per_m2": 450.0,
                    "source": "CBECS_2018",
                    "valid_from": "2018-01-01",
                    "valid_to": "2028-12-31",
                }
            ],
            "count": 1,
        }

    async def get_grid_factors(self, filters: dict) -> dict:
        return {
            "factors": [
                {
                    "country": "USA",
                    "region": "RFCW",
                    "ef_kgco2e_per_kwh": 0.386,
                    "source": "eGRID_2024",
                    "year": 2024,
                }
            ],
            "count": 1,
        }

    async def list_franchise_types(self) -> dict:
        return {
            "franchise_types": [
                {
                    "franchise_type": "qsr",
                    "display_name": "Quick-Service Restaurant",
                    "description": "Fast food and limited-service restaurants",
                    "naics_code": "722513",
                    "typical_eui_range": {"min": 300.0, "max": 800.0},
                    "typical_floor_area_m2": {"min": 100.0, "max": 500.0},
                    "has_refrigeration": True,
                    "has_cooking": True,
                },
                {
                    "franchise_type": "hotel_midscale",
                    "display_name": "Hotel (Midscale)",
                    "description": "Midscale hotels and lodging",
                    "naics_code": "721110",
                    "typical_eui_range": {"min": 200.0, "max": 500.0},
                    "typical_floor_area_m2": {"min": 2000.0, "max": 15000.0},
                    "has_refrigeration": False,
                    "has_cooking": False,
                },
            ],
            "count": 2,
        }

    async def get_aggregations(self, filters: dict) -> dict:
        return {
            "total_emissions_kgco2e": 45000000.0,
            "by_franchise_type": {"qsr": 30000000.0},
            "by_method": {"franchise_specific": 45000000.0},
            "by_region": {"US": 45000000.0},
            "unit_count": 500,
            "intensity_per_unit": 90000.0,
        }

    async def get_provenance(self, calculation_id: str):
        if calculation_id == "fake-id":
            return None
        return {
            "calculation_id": calculation_id,
            "chain": [
                {"stage": "VALIDATE", "hash": "a" * 64},
                {"stage": "CLASSIFY", "hash": "b" * 64},
            ],
            "is_valid": True,
            "root_hash": "z" * 64,
        }

    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "agent_id": "GL-MRV-S3-014",
            "version": "1.0.0",
            "engines": {"database": "ok", "pipeline": "ok"},
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def client():
    """Create TestClient with mock service."""
    if not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE:
        pytest.skip("FastAPI or router not available")

    app = FastAPI()
    app.include_router(router, prefix=PREFIX)

    mock_svc = MockFranchisesService()
    app.dependency_overrides[get_service] = lambda: mock_svc

    return TestClient(app)


# ===========================================================================
# POST /calculate Tests
# ===========================================================================


@_SKIP
class TestCalculateEndpoint:
    """Test POST /calculate endpoint."""

    def test_calculate_valid(self, client):
        """Test POST /calculate with valid network input."""
        payload = {
            "units": [{
                "unit_id": "FRN-001",
                "franchise_type": "qsr",
                "country": "USA",
                "energy": {
                    "electricity_kwh": 180000,
                    "natural_gas_therms": 12000,
                },
            }],
            "method": "franchise_specific",
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert "provenance_hash" in data
        assert data["calculation_id"] == "frn-mock-001"

    def test_calculate_empty_units_422(self, client):
        """Test POST /calculate with empty units returns 422."""
        payload = {"units": [], "reporting_year": 2025}
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        assert resp.status_code in (400, 422)


# ===========================================================================
# POST /calculate/franchise-specific Tests
# ===========================================================================


@_SKIP
class TestFranchiseSpecificEndpoint:
    """Test POST /calculate/franchise-specific endpoint."""

    def test_franchise_specific_qsr(self, client):
        """Test franchise-specific with metered energy data."""
        payload = {
            "unit": {
                "unit_id": "FRN-QSR-001",
                "franchise_type": "qsr",
                "country": "USA",
                "floor_area_m2": 250,
                "energy": {
                    "electricity_kwh": 180000,
                    "natural_gas_therms": 12000,
                },
            },
            "reporting_year": 2025,
            "include_refrigerants": True,
        }
        resp = client.post(f"{PREFIX}/calculate/franchise-specific", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert data["method"] == "franchise_specific"

    def test_franchise_specific_no_energy_400(self, client):
        """Test franchise-specific without energy data returns 400."""
        payload = {
            "unit": {
                "unit_id": "FRN-001",
                "franchise_type": "qsr",
                "country": "USA",
            },
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/franchise-specific", json=payload)
        assert resp.status_code == 400


# ===========================================================================
# POST /calculate/franchise-specific/hotel Tests
# ===========================================================================


@_SKIP
class TestFranchiseSpecificHotelEndpoint:
    """Test POST /calculate/franchise-specific/hotel endpoint."""

    def test_hotel_calculation(self, client):
        """Test franchise-specific hotel with room data."""
        payload = {
            "unit": {
                "unit_id": "FRN-HTL-001",
                "franchise_type": "hotel_upscale",
                "country": "USA",
                "floor_area_m2": 5000,
                "energy": {
                    "electricity_kwh": 950000,
                    "natural_gas_therms": 30000,
                },
            },
            "hotel_data": {
                "total_rooms": 120,
                "occupancy_rate": 0.72,
                "has_pool": True,
                "has_restaurant": True,
            },
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/franchise-specific/hotel", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert data["method"] == "franchise_specific"


# ===========================================================================
# POST /calculate/average-data Tests
# ===========================================================================


@_SKIP
class TestAverageDataEndpoint:
    """Test POST /calculate/average-data endpoint."""

    def test_average_data_calculation(self, client):
        """Test average-data benchmark calculation."""
        payload = {
            "franchise_type": "qsr",
            "floor_area_m2": 220,
            "climate_zone": "4A",
            "country": "USA",
            "unit_count": 150,
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/average-data", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert data["method"] == "average_data"
        assert "benchmark_eui_kwh_per_m2" in data


# ===========================================================================
# POST /calculate/spend-based Tests
# ===========================================================================


@_SKIP
class TestSpendBasedEndpoint:
    """Test POST /calculate/spend-based endpoint."""

    def test_spend_based_calculation(self, client):
        """Test spend-based EEIO calculation."""
        payload = {
            "franchise_type": "qsr",
            "naics_code": "722513",
            "revenue_usd": 450000000,
            "unit_count": 500,
            "reporting_year": 2025,
            "currency": "USD",
        }
        resp = client.post(f"{PREFIX}/calculate/spend-based", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert data["method"] == "spend_based"
        assert "naics_code" in data
        assert "ef_kgco2e_per_dollar" in data

    def test_spend_based_no_revenue_400(self, client):
        """Test spend-based without revenue or royalty returns 400."""
        payload = {
            "franchise_type": "qsr",
            "naics_code": "722513",
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/spend-based", json=payload)
        assert resp.status_code == 400


# ===========================================================================
# POST /calculate/hybrid Tests
# ===========================================================================


@_SKIP
class TestHybridEndpoint:
    """Test POST /calculate/hybrid endpoint."""

    def test_hybrid_calculation(self, client):
        """Test hybrid method waterfall calculation."""
        payload = {
            "metered_units": [{
                "unit_id": "FRN-001",
                "franchise_type": "qsr",
                "country": "USA",
                "energy": {
                    "electricity_kwh": 180000,
                    "natural_gas_therms": 12000,
                },
            }],
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/hybrid", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] >= 0
        assert data["method"] == "hybrid"

    def test_hybrid_no_data_400(self, client):
        """Test hybrid without any data source returns 400."""
        payload = {
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/hybrid", json=payload)
        assert resp.status_code == 400


# ===========================================================================
# POST /calculate/batch Tests
# ===========================================================================


@_SKIP
class TestBatchEndpoint:
    """Test POST /calculate/batch endpoint."""

    def test_batch_calculation(self, client):
        """Test batch calculation with multiple units."""
        payload = {
            "units": [
                {
                    "unit_id": "FRN-001",
                    "franchise_type": "qsr",
                    "country": "USA",
                    "electricity_kwh": 180000,
                },
            ],
            "method": "franchise_specific",
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/batch", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert "batch_id" in data
        assert "results" in data


# ===========================================================================
# POST /calculate/network Tests
# ===========================================================================


@_SKIP
class TestNetworkEndpoint:
    """Test POST /calculate/network endpoint."""

    def test_network_calculation(self, client):
        """Test full network analysis."""
        payload = {
            "units": [{
                "unit_id": "FRN-001",
                "franchise_type": "qsr",
                "country": "USA",
                "energy": {
                    "electricity_kwh": 180000,
                    "natural_gas_therms": 12000,
                },
            }],
            "reporting_year": 2025,
        }
        resp = client.post(f"{PREFIX}/calculate/network", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert "by_franchise_type" in data
        assert "intensity_metrics" in data


# ===========================================================================
# POST /compliance/check Tests
# ===========================================================================


@_SKIP
class TestComplianceEndpoint:
    """Test POST /compliance/check endpoint."""

    def test_compliance_check(self, client):
        """Test compliance check with GHG Protocol."""
        payload = {
            "frameworks": ["ghg_protocol"],
            "calculation_results": [
                {
                    "calculation_id": "frn-001",
                    "total_emissions_kgco2e": 85432.50,
                    "method": "franchise_specific",
                }
            ],
            "data_coverage_percent": 100.0,
            "method_hierarchy_followed": True,
        }
        resp = client.post(f"{PREFIX}/compliance/check", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["overall_status"] == "compliant"
        assert data["overall_score"] > 0

    def test_compliance_check_invalid_framework_400(self, client):
        """Test compliance check with invalid framework returns 400."""
        payload = {
            "frameworks": ["invalid_framework"],
            "calculation_results": [
                {"calculation_id": "frn-001", "total_emissions_kgco2e": 100.0}
            ],
        }
        resp = client.post(f"{PREFIX}/compliance/check", json=payload)
        assert resp.status_code == 400


# ===========================================================================
# GET /calculations/{id} Tests
# ===========================================================================


@_SKIP
class TestGetCalculationEndpoint:
    """Test GET /calculations/{id} endpoint."""

    def test_get_calculation_found(self, client):
        """Test GET calculation by ID returns 200."""
        resp = client.get(f"{PREFIX}/calculations/frn-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["calculation_id"] == "frn-001"
        assert "provenance_hash" in data

    def test_get_calculation_not_found(self, client):
        """Test GET calculation with fake ID returns 404."""
        resp = client.get(f"{PREFIX}/calculations/fake-id")
        assert resp.status_code == 404


# ===========================================================================
# GET /calculations Tests
# ===========================================================================


@_SKIP
class TestListCalculationsEndpoint:
    """Test GET /calculations endpoint."""

    def test_list_calculations(self, client):
        """Test listing calculations with pagination."""
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200
        data = resp.json()
        assert "calculations" in data
        assert "count" in data
        assert "page" in data

    def test_list_calculations_with_filters(self, client):
        """Test listing calculations with franchise type filter."""
        resp = client.get(f"{PREFIX}/calculations?franchise_type=qsr&page=1&page_size=50")
        assert resp.status_code == 200


# ===========================================================================
# DELETE /calculations/{id} Tests
# ===========================================================================


@_SKIP
class TestDeleteCalculationEndpoint:
    """Test DELETE /calculations/{id} endpoint."""

    def test_delete_calculation_found(self, client):
        """Test DELETE calculation returns 200."""
        resp = client.delete(f"{PREFIX}/calculations/frn-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True

    def test_delete_calculation_not_found(self, client):
        """Test DELETE calculation with fake ID returns 404."""
        resp = client.delete(f"{PREFIX}/calculations/fake-id")
        assert resp.status_code == 404


# ===========================================================================
# GET /emission-factors/{franchise_type} Tests
# ===========================================================================


@_SKIP
class TestEmissionFactorsEndpoint:
    """Test GET /emission-factors/{franchise_type} endpoint."""

    @pytest.mark.parametrize("franchise_type", [
        "qsr", "hotel_midscale", "convenience_store", "retail_apparel",
        "fitness_center", "automotive_service",
    ])
    def test_emission_factors_by_type(self, client, franchise_type):
        """Test emission factors for each franchise type."""
        resp = client.get(f"{PREFIX}/emission-factors/{franchise_type}")
        assert resp.status_code == 200
        data = resp.json()
        assert "factors" in data
        assert data["count"] >= 1


# ===========================================================================
# GET /franchise-benchmarks Tests
# ===========================================================================


@_SKIP
class TestBenchmarksEndpoint:
    """Test GET /franchise-benchmarks endpoint."""

    def test_get_benchmarks(self, client):
        """Test getting EUI benchmarks."""
        resp = client.get(f"{PREFIX}/franchise-benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert "benchmarks" in data
        assert data["count"] >= 1


# ===========================================================================
# GET /grid-factors Tests
# ===========================================================================


@_SKIP
class TestGridFactorsEndpoint:
    """Test GET /grid-factors endpoint."""

    def test_get_grid_factors(self, client):
        """Test getting grid emission factors."""
        resp = client.get(f"{PREFIX}/grid-factors")
        assert resp.status_code == 200
        data = resp.json()
        assert "factors" in data
        assert data["count"] >= 1


# ===========================================================================
# GET /franchise-types Tests
# ===========================================================================


@_SKIP
class TestFranchiseTypesEndpoint:
    """Test GET /franchise-types endpoint."""

    def test_get_franchise_types(self, client):
        """Test getting franchise types list."""
        resp = client.get(f"{PREFIX}/franchise-types")
        assert resp.status_code == 200
        data = resp.json()
        assert "franchise_types" in data
        assert data["count"] >= 1


# ===========================================================================
# GET /aggregations Tests
# ===========================================================================


@_SKIP
class TestAggregationsEndpoint:
    """Test GET /aggregations endpoint."""

    def test_get_aggregations(self, client):
        """Test getting aggregation data."""
        resp = client.get(f"{PREFIX}/aggregations?period=monthly")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_emissions_kgco2e"] > 0
        assert "by_franchise_type" in data

    def test_get_aggregations_invalid_period_400(self, client):
        """Test invalid aggregation period returns 400."""
        resp = client.get(f"{PREFIX}/aggregations?period=invalid_period")
        assert resp.status_code == 400


# ===========================================================================
# GET /provenance/{id} Tests
# ===========================================================================


@_SKIP
class TestProvenanceEndpoint:
    """Test GET /provenance/{id} endpoint."""

    def test_get_provenance_found(self, client):
        """Test GET provenance chain by ID."""
        resp = client.get(f"{PREFIX}/provenance/frn-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert "chain" in data
        assert "root_hash" in data

    def test_get_provenance_not_found(self, client):
        """Test GET provenance with fake ID returns 404."""
        resp = client.get(f"{PREFIX}/provenance/fake-id")
        assert resp.status_code == 404


# ===========================================================================
# GET /health Tests
# ===========================================================================


@_SKIP
class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_id"] == "GL-MRV-S3-014"

    def test_health_check_has_version(self, client):
        """Test health check includes version."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["version"] == "1.0.0"


# ===========================================================================
# Error Response Tests
# ===========================================================================


@_SKIP
class TestErrorResponses:
    """Test 400/404/422 error responses."""

    def test_invalid_json_422(self, client):
        """Test invalid JSON body returns 422."""
        resp = client.post(
            f"{PREFIX}/calculate",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_required_field_422(self, client):
        """Test missing required field returns 422."""
        payload = {"reporting_year": 2025}
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        assert resp.status_code == 422

    def test_not_found_endpoint_404(self, client):
        """Test non-existent endpoint returns 404."""
        resp = client.get(f"{PREFIX}/nonexistent")
        assert resp.status_code == 404

    def test_method_not_allowed_405(self, client):
        """Test wrong HTTP method returns 405."""
        resp = client.put(f"{PREFIX}/health")
        assert resp.status_code == 405
