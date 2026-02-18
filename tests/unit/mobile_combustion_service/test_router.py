# -*- coding: utf-8 -*-
"""
Unit tests for Mobile Combustion REST API Router - AGENT-MRV-003

Tests the 20 REST API endpoints mounted at /api/v1/mobile-combustion
using the FastAPI TestClient pattern. Each endpoint is tested for
both success and error conditions.

Target: 56+ tests across 11 test classes.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available",
)


# ===================================================================
# App and client setup
# ===================================================================


def _create_app_and_client():
    """Create a FastAPI app with the mobile combustion router and service."""
    from greenlang.mobile_combustion.setup import MobileCombustionService
    import greenlang.mobile_combustion.setup as setup_mod

    # Reset singletons
    setup_mod._singleton_instance = None
    setup_mod._service = None

    svc = MobileCombustionService()
    setup_mod._singleton_instance = svc
    setup_mod._service = svc

    from greenlang.mobile_combustion.api.router import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    return app, client, svc


@pytest.fixture
def client_and_service():
    """Create a test client with a fresh service instance."""
    import greenlang.mobile_combustion.setup as setup_mod
    app, client, svc = _create_app_and_client()
    yield client, svc
    # Cleanup
    setup_mod._singleton_instance = None
    setup_mod._service = None


@pytest.fixture
def client(client_and_service):
    """Just the test client."""
    return client_and_service[0]


@pytest.fixture
def svc(client_and_service):
    """Just the service."""
    return client_and_service[1]


# ===================================================================
# TestCalculateEndpoints (10 tests)
# ===================================================================


class TestCalculateEndpoints:
    """Test POST /calculate and POST /calculate/batch."""

    def test_calculate_fuel_based(self, client):
        """POST /calculate with fuel-based input returns 200."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total_co2e_kg", 0) > 0

    def test_calculate_distance_based(self, client):
        """POST /calculate with distance-based input returns 200."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "DISTANCE_BASED",
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "distance": 1000.0,
                "distance_unit": "KM",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total_co2e_kg", 0) > 0

    def test_calculate_spend_based(self, client):
        """POST /calculate with spend-based input returns 200."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "SPEND_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "spend_amount": 250.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total_co2e_kg", 0) > 0

    def test_calculate_fuel_based_missing_quantity(self, client):
        """POST /calculate with FUEL_BASED but no fuel_quantity returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        assert resp.status_code == 400
        assert "fuel_quantity" in resp.json().get("detail", "")

    def test_calculate_distance_based_missing_distance(self, client):
        """POST /calculate with DISTANCE_BASED but no distance returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "DISTANCE_BASED",
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
            },
        )
        assert resp.status_code == 400
        assert "distance" in resp.json().get("detail", "")

    def test_calculate_spend_based_missing_amount(self, client):
        """POST /calculate with SPEND_BASED but no spend_amount returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "SPEND_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        assert resp.status_code == 400
        assert "spend_amount" in resp.json().get("detail", "")

    def test_calculate_default_method_is_fuel_based(self, client):
        """POST /calculate without method defaults to FUEL_BASED."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 10.0,
                "fuel_unit": "GALLONS",
            },
        )
        assert resp.status_code == 200

    def test_calculate_batch_success(self, client):
        """POST /calculate/batch with valid inputs returns 200."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate/batch",
            json={
                "inputs": [
                    {
                        "calculation_method": "FUEL_BASED",
                        "vehicle_type": "PASSENGER_CAR_GASOLINE",
                        "fuel_type": "GASOLINE",
                        "fuel_quantity": 50.0,
                        "fuel_unit": "GALLONS",
                    },
                    {
                        "calculation_method": "DISTANCE_BASED",
                        "vehicle_type": "HEAVY_TRUCK_DIESEL",
                        "fuel_type": "DIESEL",
                        "distance": 200.0,
                        "distance_unit": "MILES",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_calculate_batch_empty_inputs(self, client):
        """POST /calculate/batch with empty inputs returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate/batch",
            json={"inputs": []},
        )
        assert resp.status_code == 400

    def test_calculate_batch_missing_inputs(self, client):
        """POST /calculate/batch without inputs key returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/calculate/batch",
            json={"not_inputs": []},
        )
        assert resp.status_code == 400


# ===================================================================
# TestCalculationListEndpoints (4 tests)
# ===================================================================


class TestCalculationListEndpoints:
    """Test GET /calculations and GET /calculations/{calc_id}."""

    def test_list_calculations_empty(self, client):
        """GET /calculations on empty service returns empty list."""
        resp = client.get("/api/v1/mobile-combustion/calculations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_calculations_after_calc(self, client):
        """GET /calculations after a calculation returns results."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 10.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.get("/api/v1/mobile-combustion/calculations")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_calculation_by_id(self, client):
        """GET /calculations/{calc_id} returns calculation with audit trail."""
        calc_resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 25.0,
                "fuel_unit": "GALLONS",
            },
        )
        calc_id = calc_resp.json().get("calculation_id", "")
        resp = client.get(
            f"/api/v1/mobile-combustion/calculations/{calc_id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "audit_trail" in data

    def test_get_calculation_not_found(self, client):
        """GET /calculations/{calc_id} with invalid ID returns 404."""
        resp = client.get(
            "/api/v1/mobile-combustion/calculations/nonexistent-id",
        )
        assert resp.status_code == 404


# ===================================================================
# TestVehicleEndpoints (8 tests)
# ===================================================================


class TestVehicleEndpoints:
    """Test vehicle registration and retrieval endpoints."""

    def test_register_vehicle(self, client):
        """POST /vehicles registers a vehicle and returns 201."""
        resp = client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "name": "Test Car",
                "make": "Honda",
                "model": "Civic",
                "year": 2023,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "vehicle_id" in data

    def test_list_vehicles_empty(self, client):
        """GET /vehicles returns empty list when no vehicles registered."""
        resp = client.get("/api/v1/mobile-combustion/vehicles")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_vehicles_after_registration(self, client):
        """GET /vehicles returns registered vehicles."""
        client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "name": "Fleet Car",
            },
        )
        resp = client.get("/api/v1/mobile-combustion/vehicles")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_vehicle_by_id(self, client):
        """GET /vehicles/{vehicle_id} returns vehicle details."""
        reg_resp = client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_id": "v-test-001",
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
                "name": "Delivery Truck",
            },
        )
        vid = reg_resp.json().get("vehicle_id", "v-test-001")
        resp = client.get(
            f"/api/v1/mobile-combustion/vehicles/{vid}",
        )
        assert resp.status_code == 200
        assert resp.json()["vehicle_type"] == "HEAVY_TRUCK_DIESEL"

    def test_get_vehicle_not_found(self, client):
        """GET /vehicles/{vehicle_id} with invalid ID returns 404."""
        resp = client.get(
            "/api/v1/mobile-combustion/vehicles/no-such-vehicle",
        )
        assert resp.status_code == 404

    def test_list_vehicles_filter_by_type(self, client):
        """GET /vehicles?vehicle_type=... filters correctly."""
        client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
            },
        )
        resp = client.get(
            "/api/v1/mobile-combustion/vehicles",
            params={"vehicle_type": "HEAVY_TRUCK_DIESEL"},
        )
        assert resp.status_code == 200
        vehicles = resp.json()
        for v in vehicles:
            assert v["vehicle_type"] == "HEAVY_TRUCK_DIESEL"

    def test_list_vehicles_filter_by_fuel(self, client):
        """GET /vehicles?fuel_type=... filters correctly."""
        client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        resp = client.get(
            "/api/v1/mobile-combustion/vehicles",
            params={"fuel_type": "GASOLINE"},
        )
        assert resp.status_code == 200
        for v in resp.json():
            assert v["fuel_type"] == "GASOLINE"

    def test_register_vehicle_with_custom_id(self, client):
        """POST /vehicles with explicit vehicle_id uses provided ID."""
        resp = client.post(
            "/api/v1/mobile-combustion/vehicles",
            json={
                "vehicle_id": "custom-v-123",
                "vehicle_type": "MOTORCYCLE",
                "fuel_type": "GASOLINE",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["vehicle_id"] == "custom-v-123"


# ===================================================================
# TestTripEndpoints (6 tests)
# ===================================================================


class TestTripEndpoints:
    """Test trip logging and retrieval endpoints."""

    def test_log_trip(self, client):
        """POST /trips logs a trip and returns 201."""
        resp = client.post(
            "/api/v1/mobile-combustion/trips",
            json={
                "vehicle_id": "v-001",
                "distance_km": 150.0,
                "origin": "London",
                "destination": "Manchester",
                "purpose": "delivery",
            },
        )
        assert resp.status_code == 201
        assert "trip_id" in resp.json()

    def test_log_trip_missing_vehicle_id(self, client):
        """POST /trips without vehicle_id returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/trips",
            json={"distance_km": 100.0},
        )
        assert resp.status_code == 400
        assert "vehicle_id" in resp.json().get("detail", "")

    def test_list_trips_empty(self, client):
        """GET /trips returns empty list when no trips logged."""
        resp = client.get("/api/v1/mobile-combustion/trips")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_trips_after_logging(self, client):
        """GET /trips returns trips after logging."""
        client.post(
            "/api/v1/mobile-combustion/trips",
            json={"vehicle_id": "v-001", "distance_km": 100.0},
        )
        resp = client.get("/api/v1/mobile-combustion/trips")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_trip_by_id(self, client):
        """GET /trips/{trip_id} returns trip details."""
        log_resp = client.post(
            "/api/v1/mobile-combustion/trips",
            json={
                "vehicle_id": "v-001",
                "distance_km": 250.0,
                "purpose": "business",
            },
        )
        tid = log_resp.json().get("trip_id", "")
        resp = client.get(f"/api/v1/mobile-combustion/trips/{tid}")
        assert resp.status_code == 200
        assert resp.json()["vehicle_id"] == "v-001"

    def test_get_trip_not_found(self, client):
        """GET /trips/{trip_id} with invalid ID returns 404."""
        resp = client.get(
            "/api/v1/mobile-combustion/trips/no-such-trip",
        )
        assert resp.status_code == 404


# ===================================================================
# TestFuelEndpoints (4 tests)
# ===================================================================


class TestFuelEndpoints:
    """Test fuel type endpoints."""

    def test_register_fuel(self, client):
        """POST /fuels registers a custom fuel and returns 201."""
        resp = client.post(
            "/api/v1/mobile-combustion/fuels",
            json={
                "fuel_type": "HYDROGEN",
                "display_name": "Hydrogen",
                "category": "GASEOUS",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["fuel_type"] == "HYDROGEN"
        assert "provenance_hash" in data

    def test_register_fuel_missing_type(self, client):
        """POST /fuels without fuel_type returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/fuels",
            json={"display_name": "Unknown"},
        )
        assert resp.status_code == 400

    def test_list_fuels(self, client):
        """GET /fuels returns a list of fuel types."""
        resp = client.get("/api/v1/mobile-combustion/fuels")
        assert resp.status_code == 200
        fuels = resp.json()
        assert isinstance(fuels, list)
        assert len(fuels) > 0

    def test_list_fuels_include_custom(self, client):
        """GET /fuels includes previously registered custom fuels."""
        client.post(
            "/api/v1/mobile-combustion/fuels",
            json={"fuel_type": "SYNFUEL", "category": "SYNTHETIC"},
        )
        resp = client.get("/api/v1/mobile-combustion/fuels")
        fuel_types = [f.get("fuel_type") for f in resp.json()]
        assert "SYNFUEL" in fuel_types


# ===================================================================
# TestFactorEndpoints (4 tests)
# ===================================================================


class TestFactorEndpoints:
    """Test emission factor endpoints."""

    def test_register_factor(self, client):
        """POST /factors registers a custom factor and returns 201."""
        resp = client.post(
            "/api/v1/mobile-combustion/factors",
            json={
                "fuel_type": "HYDROGEN",
                "gas": "CO2",
                "value": 0.0,
                "unit": "kg/gallon",
                "source": "CUSTOM",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["fuel_type"] == "HYDROGEN"
        assert data["gas"] == "CO2"

    def test_register_factor_missing_fields(self, client):
        """POST /factors without fuel_type or gas returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/factors",
            json={"value": 5.0},
        )
        assert resp.status_code == 400

    def test_register_factor_missing_value(self, client):
        """POST /factors without value returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/factors",
            json={"fuel_type": "DIESEL", "gas": "CO2"},
        )
        assert resp.status_code == 400

    def test_list_factors(self, client):
        """GET /factors returns a list of emission factors."""
        resp = client.get("/api/v1/mobile-combustion/factors")
        assert resp.status_code == 200
        factors = resp.json()
        assert isinstance(factors, list)
        assert len(factors) > 0


# ===================================================================
# TestAggregationEndpoints (4 tests)
# ===================================================================


class TestAggregationEndpoints:
    """Test fleet aggregation endpoints."""

    def test_aggregate_fleet(self, client):
        """POST /aggregate returns aggregation result."""
        # First perform a calculation
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 100.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.post(
            "/api/v1/mobile-combustion/aggregate",
            json={"period": "2025-Q1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_co2e_tonnes" in data

    def test_aggregate_empty(self, client):
        """POST /aggregate on empty service returns zero totals."""
        resp = client.post(
            "/api/v1/mobile-combustion/aggregate",
            json={"period": "2025-Q1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_co2e_tonnes"] == 0.0

    def test_list_aggregations_empty(self, client):
        """GET /aggregations returns empty list initially."""
        resp = client.get("/api/v1/mobile-combustion/aggregations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_aggregations_after_aggregate(self, client):
        """GET /aggregations returns stored aggregation results."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
        )
        client.post(
            "/api/v1/mobile-combustion/aggregate",
            json={"period": "2025-Q2"},
        )
        resp = client.get("/api/v1/mobile-combustion/aggregations")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1


# ===================================================================
# TestUncertaintyEndpoint (4 tests)
# ===================================================================


class TestUncertaintyEndpoint:
    """Test POST /uncertainty endpoint."""

    def test_uncertainty_analysis(self, client):
        """POST /uncertainty with valid calc_id returns analysis."""
        calc_resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 100.0,
                "fuel_unit": "GALLONS",
            },
        )
        calc_id = calc_resp.json().get("calculation_id", "")
        resp = client.post(
            "/api/v1/mobile-combustion/uncertainty",
            json={"calculation_id": calc_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_co2e_kg" in data or "calculation_id" in data

    def test_uncertainty_missing_calc_id(self, client):
        """POST /uncertainty without calculation_id returns 400."""
        resp = client.post(
            "/api/v1/mobile-combustion/uncertainty",
            json={},
        )
        assert resp.status_code == 400

    def test_uncertainty_nonexistent_calc(self, client):
        """POST /uncertainty with invalid calc_id returns 404."""
        resp = client.post(
            "/api/v1/mobile-combustion/uncertainty",
            json={"calculation_id": "nonexistent-calc"},
        )
        assert resp.status_code == 404

    def test_uncertainty_has_confidence_interval(self, client):
        """Uncertainty result includes confidence_interval_pct and mean."""
        calc_resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
        )
        calc_id = calc_resp.json().get("calculation_id", "")
        resp = client.post(
            "/api/v1/mobile-combustion/uncertainty",
            json={"calculation_id": calc_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_co2e_kg" in data
        assert "confidence_interval_pct" in data
        assert data["mean_co2e_kg"] > 0


# ===================================================================
# TestComplianceEndpoint (4 tests)
# ===================================================================


class TestComplianceEndpoint:
    """Test POST /compliance/check endpoint."""

    def test_compliance_check(self, client):
        """POST /compliance/check returns compliance result."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 100.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.post(
            "/api/v1/mobile-combustion/compliance/check",
            json={"framework": "GHG_PROTOCOL"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "compliant" in data or "framework" in data

    def test_compliance_check_iso_14064(self, client):
        """POST /compliance/check with ISO_14064 framework."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 50.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.post(
            "/api/v1/mobile-combustion/compliance/check",
            json={"framework": "ISO_14064"},
        )
        assert resp.status_code == 200

    def test_compliance_check_default_framework(self, client):
        """POST /compliance/check without framework defaults to GHG_PROTOCOL."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 25.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.post(
            "/api/v1/mobile-combustion/compliance/check",
            json={},
        )
        assert resp.status_code == 200

    def test_compliance_with_calculation_ids(self, client):
        """POST /compliance/check with explicit calculation_ids."""
        calc_resp = client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 75.0,
                "fuel_unit": "GALLONS",
            },
        )
        calc_id = calc_resp.json().get("calculation_id", "")
        resp = client.post(
            "/api/v1/mobile-combustion/compliance/check",
            json={
                "framework": "GHG_PROTOCOL",
                "calculation_ids": [calc_id],
            },
        )
        assert resp.status_code == 200


# ===================================================================
# TestHealthEndpoint (4 tests)
# ===================================================================


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check(self, client):
        """GET /health returns 200 with status."""
        resp = client.get("/api/v1/mobile-combustion/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_has_engines(self, client):
        """GET /health includes engine availability."""
        resp = client.get("/api/v1/mobile-combustion/health")
        data = resp.json()
        assert "engines" in data
        assert isinstance(data["engines"], dict)

    def test_health_has_version(self, client):
        """GET /health includes version."""
        resp = client.get("/api/v1/mobile-combustion/health")
        data = resp.json()
        assert "version" in data

    def test_health_has_timestamp(self, client):
        """GET /health includes timestamp."""
        resp = client.get("/api/v1/mobile-combustion/health")
        data = resp.json()
        assert "timestamp" in data


# ===================================================================
# TestStatsEndpoint (4 tests)
# ===================================================================


class TestStatsEndpoint:
    """Test GET /stats endpoint."""

    def test_stats_endpoint(self, client):
        """GET /stats returns 200 with statistics."""
        resp = client.get("/api/v1/mobile-combustion/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_calculations" in data

    def test_stats_reflect_calculations(self, client):
        """GET /stats reflects performed calculations."""
        client.post(
            "/api/v1/mobile-combustion/calculate",
            json={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": 10.0,
                "fuel_unit": "GALLONS",
            },
        )
        resp = client.get("/api/v1/mobile-combustion/stats")
        data = resp.json()
        assert data["total_calculations"] >= 1

    def test_stats_has_timestamp(self, client):
        """GET /stats includes timestamp."""
        resp = client.get("/api/v1/mobile-combustion/stats")
        data = resp.json()
        assert "timestamp" in data

    def test_stats_has_vehicle_count(self, client):
        """GET /stats includes vehicle count."""
        resp = client.get("/api/v1/mobile-combustion/stats")
        data = resp.json()
        assert "total_vehicles" in data
