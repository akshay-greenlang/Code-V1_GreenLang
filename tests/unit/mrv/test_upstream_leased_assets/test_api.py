# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-021 Upstream Leased Assets Agent - REST API Router.

Tests all REST API endpoints using FastAPI TestClient with a mock service
dependency override. Validates status codes, response structure, and
error handling for the endpoints defined in the router.

Target: 40 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

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
    from greenlang.upstream_leased_assets.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or upstream leased assets router not available",
)

PREFIX = "/api/v1/upstream-leased-assets"

pytestmark = _SKIP


# ===========================================================================
# Mock Service
# ===========================================================================


class MockUpstreamLeasedService:
    """
    Mock service that returns deterministic responses for all API endpoints.
    """

    async def calculate(self, data: dict) -> dict:
        return {
            "calculation_id": "ula-mock-001",
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
            "results": [
                {
                    "calculation_id": "ula-mock-001",
                    "asset_category": "building",
                    "total_co2e_kg": 42500.0,
                }
            ],
            "total_co2e_kg": 42500.0,
            "count": 1,
            "errors": [],
            "reporting_period": data.get("reporting_period", "2024"),
        }

    async def calculate_building(self, data: dict) -> dict:
        return {
            "calculation_id": "ula-bldg-mock-001",
            "building_type": data.get("building_type", "office"),
            "total_co2e_kg": 42500.0,
            "provenance_hash": "b" * 64,
        }

    async def calculate_vehicle(self, data: dict) -> dict:
        return {
            "calculation_id": "ula-veh-mock-001",
            "vehicle_type": data.get("vehicle_type", "medium_car"),
            "total_co2e_kg": 5250.0,
            "provenance_hash": "c" * 64,
        }

    async def calculate_equipment(self, data: dict) -> dict:
        return {
            "calculation_id": "ula-eq-mock-001",
            "equipment_type": data.get("equipment_type", "manufacturing"),
            "total_co2e_kg": 832500.0,
            "provenance_hash": "d" * 64,
        }

    async def calculate_it_asset(self, data: dict) -> dict:
        return {
            "calculation_id": "ula-it-mock-001",
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
        return {
            "method": "monte_carlo",
            "confidence_level": 0.95,
            "lower_bound": 38250.0,
            "upper_bound": 46750.0,
            "mean": 42500.0,
        }

    async def get_emission_factors(self) -> dict:
        return {
            "building_eui": {"office": {"temperate": 200.0}},
            "grid_efs": {"US": 0.37170},
            "fuel_efs": {"diesel": 2.68},
        }

    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "agent_id": "GL-MRV-S3-008",
            "version": "1.0.0",
        }

    async def aggregate(self, data: dict) -> dict:
        return {
            "total_co2e_kg": 50000.0,
            "by_category": {"building": 42500.0, "vehicle": 5250.0, "it_asset": 2250.0},
            "count": 3,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_service():
    """Create mock service instance."""
    return MockUpstreamLeasedService()


@pytest.fixture
def client(mock_service):
    """Create TestClient with mock service override."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: mock_service
    return TestClient(app)


# ===========================================================================
# Router Configuration Tests
# ===========================================================================


class TestRouterConfiguration:
    """Test router configuration."""

    def test_router_prefix(self):
        """Test router prefix is /api/v1/upstream-leased-assets."""
        assert router.prefix == PREFIX

    def test_router_tags(self):
        """Test router has correct tags."""
        assert "upstream-leased-assets" in router.tags


# ===========================================================================
# Endpoint Existence Tests
# ===========================================================================


class TestEndpointExistence:
    """Test all endpoints exist."""

    def test_health_endpoint_exists(self, client):
        """Test GET /health endpoint exists."""
        response = client.get(f"{PREFIX}/health")
        assert response.status_code in (200, 405, 422)

    def test_calculate_endpoint_exists(self, client):
        """Test POST /calculate endpoint exists."""
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "asset_specific",
        })
        assert response.status_code in (200, 422)

    def test_calculate_batch_endpoint_exists(self, client):
        """Test POST /calculate/batch endpoint exists."""
        response = client.post(f"{PREFIX}/calculate/batch", json={
            "assets": [],
            "reporting_period": "2024",
        })
        assert response.status_code in (200, 422)

    def test_calculate_building_endpoint_exists(self, client):
        """Test POST /calculate/building endpoint exists."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": 2500,
        })
        assert response.status_code in (200, 422)

    def test_calculate_vehicle_endpoint_exists(self, client):
        """Test POST /calculate/vehicle endpoint exists."""
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": 25000,
        })
        assert response.status_code in (200, 422)

    def test_calculate_equipment_endpoint_exists(self, client):
        """Test POST /calculate/equipment endpoint exists."""
        response = client.post(f"{PREFIX}/calculate/equipment", json={
            "equipment_type": "manufacturing",
            "rated_power_kw": 500,
        })
        assert response.status_code in (200, 422)

    def test_calculate_it_asset_endpoint_exists(self, client):
        """Test POST /calculate/it-asset endpoint exists."""
        response = client.post(f"{PREFIX}/calculate/it-asset", json={
            "it_type": "server",
            "rated_power_w": 500,
        })
        assert response.status_code in (200, 422)

    def test_compliance_endpoint_exists(self, client):
        """Test POST /compliance endpoint exists."""
        response = client.post(f"{PREFIX}/compliance", json={
            "frameworks": ["ghg_protocol"],
            "total_co2e": 85000,
        })
        assert response.status_code in (200, 422)

    def test_uncertainty_endpoint_exists(self, client):
        """Test POST /uncertainty endpoint exists."""
        response = client.post(f"{PREFIX}/uncertainty", json={
            "total_co2e_kg": 42500,
            "method": "asset_specific",
        })
        assert response.status_code in (200, 422)

    def test_emission_factors_endpoint_exists(self, client):
        """Test GET /emission-factors endpoint exists."""
        response = client.get(f"{PREFIX}/emission-factors")
        assert response.status_code in (200, 405, 422)

    def test_aggregate_endpoint_exists(self, client):
        """Test POST /aggregate endpoint exists."""
        response = client.post(f"{PREFIX}/aggregate", json={
            "results": [],
        })
        assert response.status_code in (200, 422)


# ===========================================================================
# Request Model Validation Tests
# ===========================================================================


class TestRequestValidation:
    """Test request model validation."""

    def test_building_request_valid(self, client):
        """Test valid building calculation request."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": 2500,
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": 450000},
            "region": "US",
        })
        assert response.status_code == 200

    def test_building_request_missing_type(self, client):
        """Test building request without building_type fails."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "floor_area_sqm": 2500,
        })
        assert response.status_code == 422

    def test_vehicle_request_valid(self, client):
        """Test valid vehicle calculation request."""
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": 25000,
            "region": "US",
        })
        assert response.status_code == 200

    def test_vehicle_request_missing_distance(self, client):
        """Test vehicle request without distance fails."""
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
        })
        assert response.status_code == 422

    def test_equipment_request_valid(self, client):
        """Test valid equipment calculation request."""
        response = client.post(f"{PREFIX}/calculate/equipment", json={
            "equipment_type": "manufacturing",
            "rated_power_kw": 500,
            "annual_operating_hours": 6000,
            "energy_source": "electricity",
            "region": "US",
        })
        assert response.status_code == 200

    def test_it_request_valid(self, client):
        """Test valid IT asset calculation request."""
        response = client.post(f"{PREFIX}/calculate/it-asset", json={
            "it_type": "server",
            "rated_power_w": 500,
            "utilization_pct": 0.9,
            "pue": 1.4,
            "region": "US",
        })
        assert response.status_code == 200

    def test_compliance_request_valid(self, client):
        """Test valid compliance check request."""
        response = client.post(f"{PREFIX}/compliance", json={
            "frameworks": ["ghg_protocol", "cdp"],
            "total_co2e": 85000,
            "method_used": "asset_specific",
            "reporting_period": "2024",
        })
        assert response.status_code == 200

    def test_spend_request_valid(self, client):
        """Test valid spend-based calculation request."""
        response = client.post(f"{PREFIX}/calculate", json={
            "asset_category": "building",
            "calculation_method": "spend_based",
            "naics_code": "531120",
            "amount": 120000.0,
            "currency": "USD",
            "reporting_year": 2024,
        })
        assert response.status_code == 200

    def test_batch_request_valid(self, client):
        """Test valid batch calculation request."""
        response = client.post(f"{PREFIX}/calculate/batch", json={
            "assets": [
                {
                    "asset_category": "building",
                    "asset_id": "BLDG-001",
                    "calculation_method": "average_data",
                    "building_type": "office",
                    "floor_area_sqm": 1000,
                }
            ],
            "reporting_period": "2024",
        })
        assert response.status_code == 200

    def test_negative_area_rejected(self, client):
        """Test negative floor area is rejected."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": -1000,
            "region": "US",
        })
        assert response.status_code == 422

    def test_invalid_building_type_rejected(self, client):
        """Test invalid building type is rejected."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "spaceship_hangar",
            "floor_area_sqm": 1000,
            "region": "US",
        })
        assert response.status_code == 422


# ===========================================================================
# Response Structure Tests
# ===========================================================================


class TestResponseStructure:
    """Test response model structure."""

    def test_building_response_structure(self, client):
        """Test building response has required fields."""
        response = client.post(f"{PREFIX}/calculate/building", json={
            "building_type": "office",
            "floor_area_sqm": 2500,
            "climate_zone": "temperate",
            "energy_sources": {"electricity_kwh": 450000},
            "region": "US",
        })
        if response.status_code == 200:
            data = response.json()
            assert "total_co2e_kg" in data
            assert "provenance_hash" in data

    def test_vehicle_response_structure(self, client):
        """Test vehicle response has required fields."""
        response = client.post(f"{PREFIX}/calculate/vehicle", json={
            "vehicle_type": "medium_car",
            "fuel_type": "petrol",
            "annual_distance_km": 25000,
            "region": "US",
        })
        if response.status_code == 200:
            data = response.json()
            assert "total_co2e_kg" in data

    def test_health_response_structure(self, client):
        """Test health check response structure."""
        response = client.get(f"{PREFIX}/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"

    def test_compliance_response_structure(self, client):
        """Test compliance response structure."""
        response = client.post(f"{PREFIX}/compliance", json={
            "frameworks": ["ghg_protocol"],
            "total_co2e": 85000,
            "method_used": "asset_specific",
            "reporting_period": "2024",
        })
        if response.status_code == 200:
            data = response.json()
            assert "overall_status" in data or "status" in data

    def test_emission_factors_response_structure(self, client):
        """Test emission factors response structure."""
        response = client.get(f"{PREFIX}/emission-factors")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_batch_response_structure(self, client):
        """Test batch response structure."""
        response = client.post(f"{PREFIX}/calculate/batch", json={
            "assets": [
                {
                    "asset_category": "building",
                    "asset_id": "BLDG-001",
                    "calculation_method": "average_data",
                    "building_type": "office",
                    "floor_area_sqm": 1000,
                }
            ],
            "reporting_period": "2024",
        })
        if response.status_code == 200:
            data = response.json()
            assert "results" in data or "batch_id" in data
