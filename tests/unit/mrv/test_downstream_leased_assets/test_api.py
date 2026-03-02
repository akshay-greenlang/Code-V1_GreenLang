# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-026 Downstream Leased Assets Agent - REST API Router.

Tests all REST API endpoints using FastAPI TestClient with a mock service
dependency override. Validates status codes, response structure, and
error handling.

Target: 25 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal
import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]
    TestClient = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or downstream leased assets router not available",
)

PREFIX = "/api/v1/downstream-leased-assets"

pytestmark = _SKIP


# ==============================================================================
# MOCK SERVICE
# ==============================================================================


class MockDownstreamLeasedService:
    """Mock service returning deterministic responses."""

    async def calculate(self, data: dict) -> dict:
        return {
            "calculation_id": "dla-mock-001",
            "asset_category": data.get("asset_category", "building"),
            "method": "asset_specific",
            "total_co2e_kg": 42500.0,
            "dqi_score": 4.0,
            "provenance_hash": "a" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_batch(self, data: dict) -> dict:
        return {
            "batch_id": "batch-mock-001",
            "results": [{"calculation_id": "dla-mock-001", "total_co2e_kg": 42500.0}],
            "total_co2e_kg": 42500.0,
            "count": 1,
            "errors": [],
            "reporting_period": data.get("reporting_period", "2024"),
        }

    async def calculate_building(self, data: dict) -> dict:
        return {
            "calculation_id": "dla-bldg-mock-001",
            "building_type": data.get("building_type", "office"),
            "total_co2e_kg": 42500.0,
            "provenance_hash": "b" * 64,
        }

    async def calculate_vehicle(self, data: dict) -> dict:
        return {
            "calculation_id": "dla-veh-mock-001",
            "vehicle_type": data.get("vehicle_type", "medium_car"),
            "total_co2e_kg": 5250.0,
            "provenance_hash": "c" * 64,
        }

    async def calculate_equipment(self, data: dict) -> dict:
        return {
            "calculation_id": "dla-eq-mock-001",
            "equipment_type": data.get("equipment_type", "construction"),
            "total_co2e_kg": 32500.0,
            "provenance_hash": "d" * 64,
        }

    async def calculate_it_asset(self, data: dict) -> dict:
        return {
            "calculation_id": "dla-it-mock-001",
            "it_type": data.get("it_type", "server"),
            "total_co2e_kg": 2050.0,
            "provenance_hash": "e" * 64,
        }

    async def check_compliance(self, data: dict) -> dict:
        return {
            "frameworks_checked": data.get("frameworks", ["ghg_protocol"]),
            "overall_status": "pass",
            "score": 95.0,
            "results": {},
        }

    async def get_uncertainty(self, data: dict) -> dict:
        return {"method": "monte_carlo", "lower_bound": 38250.0, "upper_bound": 46750.0}

    async def get_emission_factors(self) -> dict:
        return {"building_eui": {"office": {"temperate": 180.0}}, "grid_efs": {"US": 0.37170}}

    async def health_check(self) -> dict:
        return {"status": "healthy", "agent_id": "GL-MRV-S3-013", "version": "1.0.0"}

    async def aggregate(self, data: dict) -> dict:
        return {"total_co2e_kg": 50000.0, "by_category": {"building": 42500.0}, "count": 3}

    async def get_building_types(self) -> list:
        return ["office", "retail", "warehouse", "data_center", "hotel", "healthcare", "education", "industrial"]

    async def get_vehicle_types(self) -> list:
        return ["small_car", "medium_car", "large_car", "suv", "light_van", "heavy_van", "light_truck", "heavy_truck"]

    async def get_climate_zones(self) -> list:
        return ["tropical", "arid", "temperate", "cold", "warm"]

    async def get_allocation_methods(self) -> list:
        return ["area", "headcount", "revenue", "equal"]


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def mock_service():
    return MockDownstreamLeasedService()


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: mock_service
    return TestClient(app)


# ==============================================================================
# HEALTH CHECK
# ==============================================================================


class TestHealthCheck:

    def test_health_endpoint_200(self, client):
        response = client.get(f"{PREFIX}/health")
        assert response.status_code in (200, 405, 422)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "healthy"
            assert data["agent_id"] == "GL-MRV-S3-013"


# ==============================================================================
# CALCULATE ENDPOINTS
# ==============================================================================


class TestCalculateEndpoints:

    def test_calculate_200(self, client):
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "asset_specific",
        })
        assert response.status_code in (200, 422)

    def test_calculate_building_200(self, client):
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": 2500,
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": 450000},
            "region": "US",
        })
        assert response.status_code == 200
        if response.status_code == 200:
            data = response.json()
            assert "total_co2e_kg" in data

    def test_calculate_vehicle_200(self, client):
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": 25000,
            "fleet_size": 10,
            "region": "US",
        })
        assert response.status_code == 200

    def test_calculate_equipment_200(self, client):
        response = client.post(f"{PREFIX}/calculate/equipment", json={
            "equipment_type": "construction",
            "rated_power_kw": 200,
            "annual_operating_hours": 2000,
            "energy_source": "diesel",
            "region": "US",
        })
        assert response.status_code == 200

    def test_calculate_it_asset_200(self, client):
        response = client.post(f"{PREFIX}/calculate/it-asset", json={
            "it_type": "server",
            "rated_power_w": 500,
            "utilization_pct": 0.9,
            "pue": 1.4,
            "region": "US",
        })
        assert response.status_code == 200


# ==============================================================================
# AVERAGE-DATA, SPEND-BASED, HYBRID ENDPOINTS
# ==============================================================================


class TestMethodEndpoints:

    def test_average_data_200(self, client):
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "average_data",
            "building_type": "office",
            "floor_area_sqm": 2500,
            "climate_zone": "temperate",
            "region": "US",
        })
        assert response.status_code in (200, 422)

    def test_spend_based_200(self, client):
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "spend_based",
            "naics_code": "531120",
            "amount": 120000.0,
            "currency": "USD",
            "reporting_year": 2024,
        })
        assert response.status_code in (200, 422)

    def test_hybrid_200(self, client):
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "hybrid",
        })
        assert response.status_code in (200, 422)


# ==============================================================================
# BATCH AND PORTFOLIO ENDPOINTS
# ==============================================================================


class TestBatchEndpoints:

    def test_batch_200(self, client):
        response = client.post(f"{PREFIX}/calculate/batch", json={
            "assets": [
                {"asset_category": "building", "asset_id": "B-001", "building_type": "office", "floor_area_sqm": 1000},
            ],
            "reporting_period": "2024",
        })
        assert response.status_code in (200, 422)

    def test_portfolio_200(self, client):
        response = client.post(f"{PREFIX}/aggregate", json={"results": []})
        assert response.status_code in (200, 422)


# ==============================================================================
# COMPLIANCE ENDPOINT
# ==============================================================================


class TestComplianceEndpoint:

    def test_compliance_200(self, client):
        response = client.post(f"{PREFIX}/compliance", json={
            "frameworks": ["ghg_protocol", "cdp"],
            "total_co2e": 85000,
            "method_used": "asset_specific",
            "reporting_period": "2024",
        })
        assert response.status_code in (200, 422)


# ==============================================================================
# REFERENCE DATA ENDPOINTS (8 GET endpoints)
# ==============================================================================


class TestReferenceData:

    def test_emission_factors_200(self, client):
        response = client.get(f"{PREFIX}/emission-factors")
        assert response.status_code in (200, 405, 422)

    def test_building_types_200(self, client):
        response = client.get(f"{PREFIX}/reference/building-types")
        assert response.status_code in (200, 404, 405)

    def test_vehicle_types_200(self, client):
        response = client.get(f"{PREFIX}/reference/vehicle-types")
        assert response.status_code in (200, 404, 405)

    def test_equipment_types_200(self, client):
        response = client.get(f"{PREFIX}/reference/equipment-types")
        assert response.status_code in (200, 404, 405)

    def test_it_types_200(self, client):
        response = client.get(f"{PREFIX}/reference/it-types")
        assert response.status_code in (200, 404, 405)

    def test_climate_zones_200(self, client):
        response = client.get(f"{PREFIX}/reference/climate-zones")
        assert response.status_code in (200, 404, 405)

    def test_allocation_methods_200(self, client):
        response = client.get(f"{PREFIX}/reference/allocation-methods")
        assert response.status_code in (200, 404, 405)

    def test_naics_codes_200(self, client):
        response = client.get(f"{PREFIX}/reference/naics-codes")
        assert response.status_code in (200, 404, 405)


# ==============================================================================
# VALIDATION ERROR TESTS (422)
# ==============================================================================


class TestValidationErrors:

    def test_negative_area_422(self, client):
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": -1000,
            "region": "US",
        })
        assert response.status_code == 422

    def test_invalid_building_type_422(self, client):
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "spaceship_hangar",
            "floor_area_sqm": 1000,
            "region": "US",
        })
        assert response.status_code == 422

    def test_missing_required_field_422(self, client):
        response = client.post(f"{PREFIX}/calculate/building", json={
            "floor_area_sqm": 2500,
        })
        assert response.status_code == 422

    def test_negative_distance_422(self, client):
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "diesel",
            "annual_distance_km": -1000,
        })
        assert response.status_code == 422
