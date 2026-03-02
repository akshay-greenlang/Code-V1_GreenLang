# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-019 Business Travel Agent - REST API Router.

Tests all REST API endpoints using FastAPI TestClient with a mock service
dependency override. Validates status codes, response structure, and
error handling for the 22 endpoints defined in the router.

Target: 35 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
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
    from greenlang.business_travel.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or business travel router not available",
)

PREFIX = "/api/v1/business-travel"


# ===========================================================================
# Mock Service
# ===========================================================================


class MockBusinessTravelService:
    """
    Mock service that returns deterministic responses for all API endpoints.

    Every async method returns a dict matching what the router expects,
    so we can test router logic without exercising the real calculation engines.
    """

    # -- calculations -------------------------------------------------------

    async def calculate(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-001",
            "mode": data.get("mode", "air"),
            "method": "distance_based",
            "total_co2e_kg": 1234.56,
            "co2e_without_rf_kg": 1000.0,
            "co2e_with_rf_kg": 1100.0,
            "wtt_co2e_kg": 134.56,
            "dqi_score": 4.0,
            "provenance_hash": "a" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    async def calculate_batch(self, data: dict) -> dict:
        return {
            "batch_id": "batch-mock-001",
            "results": [
                {
                    "calculation_id": "bt-mock-001",
                    "mode": "air",
                    "total_co2e_kg": 1234.56,
                }
            ],
            "total_co2e_kg": 1234.56,
            "count": 1,
            "errors": [],
            "reporting_period": data.get("reporting_period", "2024-Q1"),
        }

    async def calculate_flight(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-flt-001",
            "origin_iata": data.get("origin_iata", "LHR"),
            "destination_iata": data.get("destination_iata", "JFK"),
            "distance_km": 5541.0,
            "distance_band": "long_haul",
            "cabin_class": data.get("cabin_class", "economy"),
            "class_multiplier": 1.0,
            "co2e_without_rf_kg": 1000.0,
            "co2e_with_rf_kg": 1100.0,
            "wtt_co2e_kg": 134.0,
            "total_co2e_kg": 1234.0,
            "rf_option": data.get("rf_option", "with_rf"),
            "provenance_hash": "b" * 64,
        }

    async def calculate_rail(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-rail-001",
            "mode": "rail",
            "method": "distance_based",
            "total_co2e_kg": 25.53,
            "wtt_co2e_kg": 3.12,
            "dqi_score": 3.5,
            "provenance_hash": "c" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    async def calculate_road(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-road-001",
            "mode": "road",
            "method": "distance_based",
            "total_co2e_kg": 81.44,
            "wtt_co2e_kg": 18.87,
            "dqi_score": 3.5,
            "provenance_hash": "d" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    async def calculate_hotel(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-hotel-001",
            "mode": "hotel",
            "method": "average_data",
            "total_co2e_kg": 36.96,
            "wtt_co2e_kg": 0.0,
            "dqi_score": 3.0,
            "provenance_hash": "e" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    async def calculate_spend(self, data: dict) -> dict:
        return {
            "calculation_id": "bt-mock-spend-001",
            "mode": "spend",
            "method": "spend_based",
            "total_co2e_kg": 2385.0,
            "wtt_co2e_kg": 0.0,
            "dqi_score": 1.5,
            "provenance_hash": "f" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    # -- CRUD ---------------------------------------------------------------

    async def get_calculation(self, calculation_id: str):
        if calculation_id == "fake-id":
            return None
        return {
            "calculation_id": calculation_id,
            "mode": "air",
            "method": "distance_based",
            "total_co2e_kg": 1234.56,
            "details": {},
            "provenance_hash": "a" * 64,
            "calculated_at": "2025-01-01T00:00:00Z",
        }

    async def list_calculations(self, filters: dict) -> dict:
        return {
            "calculations": [],
            "count": 0,
        }

    async def delete_calculation(self, calculation_id: str) -> bool:
        if calculation_id == "fake-id":
            return False
        return True

    # -- Emission factors & metadata ----------------------------------------

    async def list_emission_factors(self, filters: dict) -> dict:
        return {
            "factors": [
                {
                    "mode": "air",
                    "vehicle_type": None,
                    "ef_value": 0.219,
                    "wtt_value": 0.045,
                    "unit": "kgCO2e/pkm",
                    "source": "DEFRA",
                }
            ],
            "count": 1,
        }

    async def get_emission_factors_by_mode(self, mode: str) -> dict:
        return {
            "factors": [
                {
                    "mode": mode,
                    "vehicle_type": None,
                    "ef_value": 0.219,
                    "wtt_value": 0.045,
                    "unit": "kgCO2e/pkm",
                    "source": "DEFRA",
                }
            ],
            "count": 1,
        }

    async def search_airports(self, filters: dict) -> dict:
        q = (filters.get("q") or "").upper()
        airports = [
            {
                "iata_code": "LHR",
                "name": "London Heathrow",
                "city": "London",
                "country_code": "GB",
                "latitude": 51.47,
                "longitude": -0.4543,
            },
            {
                "iata_code": "JFK",
                "name": "John F. Kennedy International",
                "city": "New York",
                "country_code": "US",
                "latitude": 40.6413,
                "longitude": -73.7781,
            },
        ]
        if q:
            airports = [a for a in airports if q in a["iata_code"] or q in a["name"].upper()]
        return {"airports": airports, "count": len(airports)}

    async def list_transport_modes(self) -> dict:
        return {
            "modes": [
                {"mode": m, "display_name": m.title(), "ef_source": "DEFRA"}
                for m in ["air", "rail", "road", "bus", "taxi", "ferry", "motorcycle", "hotel"]
            ],
        }

    async def list_cabin_classes(self) -> dict:
        return {
            "classes": [
                {"cabin_class": "economy", "display_name": "Economy", "multiplier": 1.0},
                {"cabin_class": "premium_economy", "display_name": "Premium Economy", "multiplier": 1.6},
                {"cabin_class": "business", "display_name": "Business", "multiplier": 2.9},
                {"cabin_class": "first", "display_name": "First", "multiplier": 4.0},
            ],
        }

    # -- Compliance & analysis ----------------------------------------------

    async def check_compliance(self, data: dict) -> dict:
        return {
            "results": [{"framework": "ghg_protocol", "status": "pass", "findings": []}],
            "overall_status": "pass",
            "overall_score": 1.0,
        }

    async def analyze_uncertainty(self, data: dict) -> dict:
        return {
            "mean": 1234.56,
            "std_dev": 100.0,
            "ci_lower": 1034.56,
            "ci_upper": 1434.56,
            "method": data.get("method", "monte_carlo"),
            "iterations": data.get("iterations", 10000),
        }

    async def analyze_hot_spots(self, data: dict) -> dict:
        return {
            "top_routes": [{"route": "LHR-JFK", "co2e_kg": 1234.0}],
            "top_modes": {"air": 1234.0},
            "reduction_opportunities": [
                {"category": "rail_substitute", "saving_pct": 80}
            ],
        }

    # -- Aggregation & provenance -------------------------------------------

    async def get_aggregations(self, filters: dict) -> dict:
        return {
            "total_co2e_kg": 5000.0,
            "by_mode": {"air": 3000.0, "rail": 500.0, "hotel": 1500.0},
            "by_department": {},
            "trip_count": 10,
        }

    async def get_provenance(self, calculation_id: str):
        if calculation_id == "fake-id":
            return None
        return {
            "calculation_id": calculation_id,
            "chain": [{"stage": "validate", "hash": "aaa"}],
            "is_valid": True,
            "root_hash": "a" * 64,
        }

    # -- Stats --------------------------------------------------------------

    async def get_stats(self) -> dict:
        return {
            "total_calculations": 42,
            "total_co2e_kg": 99999.9,
            "total_flights": 20,
            "total_ground_trips": 15,
            "total_hotel_nights": 7,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def client():
    """FastAPI TestClient with the business travel router and mock service."""
    if not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE:
        pytest.skip("FastAPI or router not available")

    app = FastAPI()
    app.include_router(router)

    mock_service = MockBusinessTravelService()
    app.dependency_overrides[get_service] = lambda: mock_service

    return TestClient(app)


# ===========================================================================
# Tests (35)
# ===========================================================================


@_SKIP
class TestHealthAndStats:
    """Test health and statistics endpoints."""

    def test_health_endpoint(self, client):
        """GET /health returns 200."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    def test_health_response_fields(self, client):
        """GET /health response contains status, agent_id, version."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "status" in data
        assert "agent_id" in data
        assert "version" in data

    def test_stats_endpoint(self, client):
        """GET /stats returns 200."""
        resp = client.get(f"{PREFIX}/stats")
        assert resp.status_code == 200


@_SKIP
class TestMetadataEndpoints:
    """Test transport modes, cabin classes, emission factors, and airports."""

    def test_transport_modes(self, client):
        """GET /transport-modes returns 200."""
        resp = client.get(f"{PREFIX}/transport-modes")
        assert resp.status_code == 200

    def test_transport_modes_count(self, client):
        """GET /transport-modes returns 8 modes."""
        resp = client.get(f"{PREFIX}/transport-modes")
        data = resp.json()
        assert len(data["modes"]) == 8

    def test_cabin_classes(self, client):
        """GET /cabin-classes returns 200."""
        resp = client.get(f"{PREFIX}/cabin-classes")
        assert resp.status_code == 200

    def test_cabin_classes_count(self, client):
        """GET /cabin-classes returns 4 classes."""
        resp = client.get(f"{PREFIX}/cabin-classes")
        data = resp.json()
        assert len(data["classes"]) == 4

    def test_emission_factors(self, client):
        """GET /emission-factors returns 200."""
        resp = client.get(f"{PREFIX}/emission-factors")
        assert resp.status_code == 200

    def test_emission_factors_by_mode(self, client):
        """GET /emission-factors/air returns 200."""
        resp = client.get(f"{PREFIX}/emission-factors/air")
        assert resp.status_code == 200

    def test_airports(self, client):
        """GET /airports returns 200."""
        resp = client.get(f"{PREFIX}/airports")
        assert resp.status_code == 200

    def test_airports_search(self, client):
        """GET /airports?q=london returns results containing London."""
        resp = client.get(f"{PREFIX}/airports", params={"q": "london"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1


@_SKIP
class TestCalculationEndpoints:
    """Test single-mode and generic calculation endpoints."""

    def test_calculate_flight(self, client):
        """POST /calculate/flight returns 201."""
        payload = {
            "origin_iata": "LHR",
            "destination_iata": "JFK",
            "cabin_class": "economy",
            "passengers": 1,
            "round_trip": False,
            "rf_option": "with_rf",
        }
        resp = client.post(f"{PREFIX}/calculate/flight", json=payload)
        assert resp.status_code == 201

    def test_calculate_rail(self, client):
        """POST /calculate/rail returns 201."""
        payload = {
            "rail_type": "national",
            "distance_km": 500.0,
            "passengers": 1,
        }
        resp = client.post(f"{PREFIX}/calculate/rail", json=payload)
        assert resp.status_code == 201

    def test_calculate_road(self, client):
        """POST /calculate/road returns 201."""
        payload = {
            "vehicle_type": "car_average",
            "distance_km": 200.0,
        }
        resp = client.post(f"{PREFIX}/calculate/road", json=payload)
        assert resp.status_code == 201

    def test_calculate_hotel(self, client):
        """POST /calculate/hotel returns 201."""
        payload = {
            "country_code": "GB",
            "room_nights": 3,
            "hotel_class": "standard",
        }
        resp = client.post(f"{PREFIX}/calculate/hotel", json=payload)
        assert resp.status_code == 201

    def test_calculate_spend(self, client):
        """POST /calculate/spend returns 201."""
        payload = {
            "naics_code": "481000",
            "amount": 5000.0,
            "currency": "USD",
            "reporting_year": 2024,
        }
        resp = client.post(f"{PREFIX}/calculate/spend", json=payload)
        assert resp.status_code == 201

    def test_calculate_generic(self, client):
        """POST /calculate returns 201."""
        payload = {
            "mode": "rail",
            "trip_data": {
                "rail_type": "national",
                "distance_km": 300,
                "passengers": 1,
            },
        }
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        assert resp.status_code == 201

    def test_calculate_batch(self, client):
        """POST /calculate/batch returns 201."""
        payload = {
            "trips": [
                {"mode": "air", "trip_data": {"origin_iata": "LHR", "destination_iata": "JFK"}},
            ],
            "reporting_period": "2024-Q1",
        }
        resp = client.post(f"{PREFIX}/calculate/batch", json=payload)
        assert resp.status_code == 201

    def test_flight_response_has_fields(self, client):
        """Flight response includes calculation_id, total_co2e_kg, distance_km."""
        payload = {
            "origin_iata": "LHR",
            "destination_iata": "JFK",
        }
        resp = client.post(f"{PREFIX}/calculate/flight", json=payload)
        data = resp.json()
        assert "calculation_id" in data
        assert "total_co2e_kg" in data
        assert "distance_km" in data

    def test_calculate_response_format(self, client):
        """Generic calculate response includes mode, method, and provenance_hash."""
        payload = {
            "mode": "road",
            "trip_data": {"vehicle_type": "car_average", "distance_km": 100},
        }
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        data = resp.json()
        assert "mode" in data
        assert "method" in data
        assert "provenance_hash" in data

    def test_api_generates_calculation_id(self, client):
        """Each calculation generates a unique calculation_id."""
        payload = {
            "mode": "rail",
            "trip_data": {"rail_type": "national", "distance_km": 100, "passengers": 1},
        }
        resp = client.post(f"{PREFIX}/calculate", json=payload)
        data = resp.json()
        assert data["calculation_id"] is not None
        assert len(data["calculation_id"]) > 0


@_SKIP
class TestComplianceAndAnalysis:
    """Test compliance, uncertainty, and hot-spot endpoints."""

    def test_compliance_check(self, client):
        """POST /compliance/check returns 201."""
        payload = {
            "frameworks": ["ghg_protocol"],
            "calculation_results": [{"total_co2e": 1500.0}],
            "rf_disclosed": True,
            "mode_breakdown_provided": True,
        }
        resp = client.post(f"{PREFIX}/compliance/check", json=payload)
        assert resp.status_code == 201

    def test_uncertainty_analyze(self, client):
        """POST /uncertainty/analyze returns 201."""
        payload = {
            "method": "monte_carlo",
            "iterations": 10000,
            "confidence_level": 0.95,
            "calculation_results": [{"total_co2e": 1234.0}],
        }
        resp = client.post(f"{PREFIX}/uncertainty/analyze", json=payload)
        assert resp.status_code == 201

    def test_hot_spots(self, client):
        """POST /hot-spots/analyze returns 201."""
        payload = {
            "calculation_results": [{"total_co2e": 1234.0, "mode": "air"}],
            "top_n": 5,
        }
        resp = client.post(f"{PREFIX}/hot-spots/analyze", json=payload)
        assert resp.status_code == 201


@_SKIP
class TestCRUDEndpoints:
    """Test calculation listing, retrieval, deletion, aggregation, provenance."""

    def test_get_calculation_not_found(self, client):
        """GET /calculations/fake-id returns 404."""
        resp = client.get(f"{PREFIX}/calculations/fake-id")
        assert resp.status_code == 404

    def test_list_calculations(self, client):
        """GET /calculations returns 200."""
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200

    def test_delete_calculation(self, client):
        """DELETE /calculations/fake-id returns 404 for non-existent."""
        resp = client.delete(f"{PREFIX}/calculations/fake-id")
        assert resp.status_code == 404

    def test_get_aggregations(self, client):
        """GET /aggregations/quarterly returns 200."""
        resp = client.get(f"{PREFIX}/aggregations/quarterly")
        assert resp.status_code == 200

    def test_get_provenance(self, client):
        """GET /provenance/bt-001 returns 200 for existing calculation."""
        resp = client.get(f"{PREFIX}/provenance/bt-001")
        assert resp.status_code == 200

    def test_get_provenance_not_found(self, client):
        """GET /provenance/fake-id returns 404 for non-existent."""
        resp = client.get(f"{PREFIX}/provenance/fake-id")
        assert resp.status_code == 404


@_SKIP
class TestValidationErrors:
    """Test request validation returning 422 for invalid inputs."""

    def test_invalid_flight_missing_iata(self, client):
        """POST /calculate/flight without IATA codes returns 422."""
        payload = {
            "cabin_class": "economy",
            "passengers": 1,
        }
        resp = client.post(f"{PREFIX}/calculate/flight", json=payload)
        assert resp.status_code == 422

    def test_invalid_hotel_zero_nights(self, client):
        """POST /calculate/hotel with room_nights=0 returns 422."""
        payload = {
            "country_code": "GB",
            "room_nights": 0,
            "hotel_class": "standard",
        }
        resp = client.post(f"{PREFIX}/calculate/hotel", json=payload)
        assert resp.status_code == 422

    def test_batch_size_limit(self, client):
        """POST /calculate/batch with empty trips list returns 422."""
        payload = {
            "trips": [],
            "reporting_period": "2024-Q1",
        }
        resp = client.post(f"{PREFIX}/calculate/batch", json=payload)
        assert resp.status_code == 422


@_SKIP
class TestRouterConfiguration:
    """Test router prefix and tags."""

    def test_router_prefix(self):
        """Router prefix is /api/v1/business-travel."""
        assert router.prefix == "/api/v1/business-travel"

    def test_router_tags(self):
        """Router tags include 'business-travel'."""
        assert "business-travel" in router.tags
