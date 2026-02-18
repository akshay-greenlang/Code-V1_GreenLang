# -*- coding: utf-8 -*-
"""
Unit tests for Refrigerants & F-Gas API Router - AGENT-MRV-002

Tests all 20 REST API endpoints via FastAPI TestClient:
- GET /health, GET /stats
- POST /calculate, POST /calculate/batch
- GET /calculations, GET /calculations/{calc_id}
- POST /refrigerants, GET /refrigerants, GET /refrigerants/{ref_id}
- POST /equipment, GET /equipment, GET /equipment/{equip_id}
- POST /service-events, GET /service-events
- POST /leak-rates, GET /leak-rates
- POST /compliance/check, GET /compliance
- POST /uncertainty
- GET /audit/{calc_id}
- Validation error scenarios (422, 400, 404)
- Route enumeration

Target: 55+ tests, ~750 lines
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

# Guard: skip all tests if FastAPI is not available
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.refrigerants_fgas.api.router import router
from greenlang.refrigerants_fgas.setup import RefrigerantsFGasService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> RefrigerantsFGasService:
    """Create a fresh RefrigerantsFGasService for each test."""
    svc = RefrigerantsFGasService()
    svc.startup()
    return svc


@pytest.fixture
def app(service) -> FastAPI:
    """Create a FastAPI app with the router mounted and service wired."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app, service) -> TestClient:
    """Create a TestClient with the service singleton patched."""
    with patch(
        "greenlang.refrigerants_fgas.api.router._get_service",
        return_value=service,
    ):
        with TestClient(app) as c:
            yield c


PREFIX = "/api/v1/refrigerants-fgas"


# ===========================================================================
# Test health and stats endpoints
# ===========================================================================


class TestHealthAndStats:
    """Tests for GET /health and GET /stats."""

    def test_health_endpoint(self, client):
        """GET /health returns 200 with health status."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["version"] == "1.0.0"
        assert "engines" in data
        assert "timestamp" in data

    def test_stats_endpoint(self, client):
        """GET /stats returns 200 with statistics."""
        resp = client.get(f"{PREFIX}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_calculations" in data
        assert "total_batch_runs" in data
        assert "timestamp" in data


# ===========================================================================
# Test calculation endpoints
# ===========================================================================


class TestCalculateEndpoints:
    """Tests for POST /calculate and POST /calculate/batch."""

    def test_calculate_endpoint(self, client):
        """POST /calculate returns calculation result."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "calculation_id" in data
        assert "processing_time_ms" in data

    def test_calculate_endpoint_mass_balance(self, client):
        """POST /calculate works with mass_balance method."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_134A",
                "charge_kg": 100.0,
                "method": "mass_balance",
                "mass_balance_data": {
                    "inventory_start_kg": 500.0,
                    "purchases_kg": 100.0,
                    "recovery_kg": 50.0,
                    "inventory_end_kg": 450.0,
                },
            },
        )
        assert resp.status_code == 200

    def test_calculate_endpoint_missing_fields(self, client):
        """POST /calculate returns 400 for missing required fields."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={"method": "equipment_based"},
        )
        assert resp.status_code == 400
        assert "required" in resp.json()["detail"].lower()

    def test_calculate_batch_endpoint(self, client):
        """POST /calculate/batch processes multiple inputs."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={
                "inputs": [
                    {"refrigerant_type": "R_410A", "charge_kg": 25.0},
                    {"refrigerant_type": "R_134A", "charge_kg": 10.0},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data or "total_count" in data

    def test_calculate_batch_missing_inputs(self, client):
        """POST /calculate/batch returns 400 for missing inputs."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={},
        )
        assert resp.status_code == 400

    def test_calculate_batch_empty_inputs(self, client):
        """POST /calculate/batch returns 400 for empty inputs list."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={"inputs": []},
        )
        assert resp.status_code == 400


# ===========================================================================
# Test calculation list and detail endpoints
# ===========================================================================


class TestCalculationListAndDetail:
    """Tests for GET /calculations and GET /calculations/{calc_id}."""

    def test_list_calculations_endpoint(self, client):
        """GET /calculations returns a list."""
        # First create a calculation
        client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_calculation_endpoint(self, client):
        """GET /calculations/{calc_id} returns calculation details."""
        # Create calculation first
        create_resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        calc_id = create_resp.json()["calculation_id"]

        resp = client.get(f"{PREFIX}/calculations/{calc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["calculation_id"] == calc_id
        assert "audit_trail" in data

    def test_get_calculation_not_found(self, client):
        """GET /calculations/{calc_id} returns 404 for unknown ID."""
        resp = client.get(f"{PREFIX}/calculations/nonexistent_id")
        assert resp.status_code == 404

    def test_list_calculations_with_filter(self, client):
        """GET /calculations supports refrigerant_type filter."""
        client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"refrigerant_type": "R_410A"},
        )
        assert resp.status_code == 200


# ===========================================================================
# Test refrigerant endpoints
# ===========================================================================


class TestRefrigerantEndpoints:
    """Tests for POST /refrigerants, GET /refrigerants, GET /refrigerants/{ref_id}."""

    def test_register_refrigerant_endpoint(self, client):
        """POST /refrigerants registers a custom refrigerant."""
        resp = client.post(
            f"{PREFIX}/refrigerants",
            json={
                "refrigerant_type": "CUSTOM_GAS_1",
                "category": "OTHER",
                "display_name": "Custom Gas 1",
                "gwp_ar6": 100.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["refrigerant_type"] == "CUSTOM_GAS_1"

    def test_register_refrigerant_missing_type(self, client):
        """POST /refrigerants returns 400 when type is missing."""
        resp = client.post(
            f"{PREFIX}/refrigerants",
            json={"category": "OTHER"},
        )
        assert resp.status_code == 400

    def test_list_refrigerants_endpoint(self, client):
        """GET /refrigerants returns a list."""
        # Register one first
        client.post(
            f"{PREFIX}/refrigerants",
            json={
                "refrigerant_type": "TEST_GAS",
                "category": "HFC",
            },
        )
        resp = client.get(f"{PREFIX}/refrigerants")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_refrigerant_endpoint(self, client):
        """GET /refrigerants/{ref_id} returns refrigerant details."""
        # Register first
        client.post(
            f"{PREFIX}/refrigerants",
            json={
                "refrigerant_type": "TEST_R32",
                "category": "HFC",
                "gwp_ar6": 771.0,
            },
        )
        resp = client.get(f"{PREFIX}/refrigerants/TEST_R32")
        assert resp.status_code == 200
        data = resp.json()
        assert data["refrigerant_type"] == "TEST_R32"

    def test_get_refrigerant_not_found(self, client):
        """GET /refrigerants/{ref_id} returns 404 for unknown type."""
        resp = client.get(f"{PREFIX}/refrigerants/NONEXISTENT")
        assert resp.status_code == 404


# ===========================================================================
# Test equipment endpoints
# ===========================================================================


class TestEquipmentEndpoints:
    """Tests for POST /equipment, GET /equipment, GET /equipment/{equip_id}."""

    def test_register_equipment_endpoint(self, client):
        """POST /equipment registers equipment profile."""
        resp = client.post(
            f"{PREFIX}/equipment",
            json={
                "equipment_type": "COMMERCIAL_AC",
                "name": "Chiller A1",
                "refrigerant_type": "R_410A",
                "charge_kg": 50.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "equipment_id" in data
        assert data["equipment_type"] == "COMMERCIAL_AC"

    def test_list_equipment_endpoint(self, client):
        """GET /equipment returns a list of profiles."""
        client.post(
            f"{PREFIX}/equipment",
            json={
                "equipment_type": "COMMERCIAL_AC",
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        resp = client.get(f"{PREFIX}/equipment")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_equipment_endpoint(self, client):
        """GET /equipment/{equip_id} returns equipment details."""
        create_resp = client.post(
            f"{PREFIX}/equipment",
            json={
                "equipment_id": "eq_test_api_001",
                "equipment_type": "CHILLERS_SCREW",
                "refrigerant_type": "R_134A",
                "charge_kg": 100.0,
            },
        )
        resp = client.get(f"{PREFIX}/equipment/eq_test_api_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["equipment_id"] == "eq_test_api_001"

    def test_get_equipment_not_found(self, client):
        """GET /equipment/{equip_id} returns 404 for unknown ID."""
        resp = client.get(f"{PREFIX}/equipment/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Test service event endpoints
# ===========================================================================


class TestServiceEventEndpoints:
    """Tests for POST /service-events and GET /service-events."""

    def test_log_service_event_endpoint(self, client):
        """POST /service-events logs a service event."""
        resp = client.post(
            f"{PREFIX}/service-events",
            json={
                "equipment_id": "eq_001",
                "event_type": "recharge",
                "refrigerant_type": "R_410A",
                "quantity_kg": 5.0,
                "technician": "Jane Doe",
                "notes": "Annual top-up",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["equipment_id"] == "eq_001"
        assert data["event_type"] == "recharge"

    def test_log_service_event_missing_fields(self, client):
        """POST /service-events returns 400 for missing required fields."""
        resp = client.post(
            f"{PREFIX}/service-events",
            json={"notes": "missing equipment_id and event_type"},
        )
        assert resp.status_code == 400

    def test_list_service_events_endpoint(self, client):
        """GET /service-events returns a list of events."""
        client.post(
            f"{PREFIX}/service-events",
            json={
                "equipment_id": "eq_002",
                "event_type": "installation",
                "quantity_kg": 50.0,
            },
        )
        resp = client.get(f"{PREFIX}/service-events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_service_events_with_filter(self, client):
        """GET /service-events supports equipment_id filter."""
        client.post(
            f"{PREFIX}/service-events",
            json={
                "equipment_id": "eq_filter_test",
                "event_type": "repair",
            },
        )
        resp = client.get(
            f"{PREFIX}/service-events",
            params={"equipment_id": "eq_filter_test"},
        )
        assert resp.status_code == 200


# ===========================================================================
# Test leak rate endpoints
# ===========================================================================


class TestLeakRateEndpoints:
    """Tests for POST /leak-rates and GET /leak-rates."""

    def test_register_leak_rate_endpoint(self, client):
        """POST /leak-rates registers a custom leak rate."""
        resp = client.post(
            f"{PREFIX}/leak-rates",
            json={
                "equipment_type": "COMMERCIAL_AC",
                "base_rate_pct": 8.0,
                "age_factor": 1.1,
                "climate_factor": 1.0,
                "ldar_adjustment": 0.9,
                "source": "custom",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["equipment_type"] == "COMMERCIAL_AC"
        assert data["base_rate_pct"] == 8.0
        assert "effective_rate_pct" in data
        assert "provenance_hash" in data

    def test_register_leak_rate_missing_type(self, client):
        """POST /leak-rates returns 400 when equipment_type missing."""
        resp = client.post(
            f"{PREFIX}/leak-rates",
            json={"base_rate_pct": 5.0},
        )
        assert resp.status_code == 400

    def test_list_leak_rates_endpoint(self, client):
        """GET /leak-rates returns a list of leak rates."""
        client.post(
            f"{PREFIX}/leak-rates",
            json={
                "equipment_type": "INDUSTRIAL_REFRIGERATION",
                "base_rate_pct": 10.0,
            },
        )
        resp = client.get(f"{PREFIX}/leak-rates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ===========================================================================
# Test compliance endpoints
# ===========================================================================


class TestComplianceEndpoints:
    """Tests for POST /compliance/check and GET /compliance."""

    def test_check_compliance_endpoint(self, client):
        """POST /compliance/check returns compliance results."""
        resp = client.post(
            f"{PREFIX}/compliance/check",
            json={
                "frameworks": ["GHG_PROTOCOL", "EU_FGAS_2024"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert data["total_count"] >= 2

    def test_check_compliance_all_defaults(self, client):
        """POST /compliance/check with empty body uses default frameworks."""
        resp = client.post(
            f"{PREFIX}/compliance/check",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= 5

    def test_list_compliance_endpoint(self, client):
        """GET /compliance returns stored compliance records."""
        # First create a compliance check with a calculation_id
        calc_resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        calc_id = calc_resp.json()["calculation_id"]

        client.post(
            f"{PREFIX}/compliance/check",
            json={"calculation_id": calc_id},
        )

        resp = client.get(f"{PREFIX}/compliance")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ===========================================================================
# Test uncertainty endpoint
# ===========================================================================


class TestUncertaintyEndpoint:
    """Tests for POST /uncertainty."""

    def test_uncertainty_endpoint(self, client):
        """POST /uncertainty runs uncertainty analysis."""
        # Create calculation first
        calc_resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        calc_id = calc_resp.json()["calculation_id"]

        resp = client.post(
            f"{PREFIX}/uncertainty",
            json={"calculation_id": calc_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_co2e_kg" in data or "calculation_id" in data

    def test_uncertainty_missing_calc_id(self, client):
        """POST /uncertainty returns 400 when calc_id missing."""
        resp = client.post(
            f"{PREFIX}/uncertainty",
            json={},
        )
        assert resp.status_code == 400

    def test_uncertainty_calc_not_found(self, client):
        """POST /uncertainty returns 404 for unknown calculation."""
        resp = client.post(
            f"{PREFIX}/uncertainty",
            json={"calculation_id": "nonexistent_calc"},
        )
        assert resp.status_code == 404


# ===========================================================================
# Test audit endpoint
# ===========================================================================


class TestAuditEndpoint:
    """Tests for GET /audit/{calc_id}."""

    def test_get_audit_endpoint(self, client):
        """GET /audit/{calc_id} returns audit trail."""
        # Create calculation first
        calc_resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        calc_id = calc_resp.json()["calculation_id"]

        resp = client.get(f"{PREFIX}/audit/{calc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["calculation_id"] == calc_id
        assert "entries" in data
        assert "total_entries" in data

    def test_get_audit_not_found(self, client):
        """GET /audit/{calc_id} returns 404 for unknown calculation."""
        resp = client.get(f"{PREFIX}/audit/nonexistent_calc")
        assert resp.status_code == 404


# ===========================================================================
# Test route enumeration
# ===========================================================================


class TestRouteEnumeration:
    """Tests for verifying all 20 routes are registered."""

    def test_all_routes_exist(self, app):
        """Verify 20 routes are registered on the router."""
        routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path.startswith(PREFIX)
        ]
        # We expect 20 routes (some share paths with different methods)
        route_paths = [(r.path, list(r.methods)) for r in routes]
        assert len(route_paths) >= 20

    def test_route_paths_expected(self, app):
        """Verify specific expected route paths exist."""
        route_paths = set()
        for r in app.routes:
            if hasattr(r, "path") and r.path.startswith(PREFIX):
                route_paths.add(r.path)

        expected_paths = [
            f"{PREFIX}/calculate",
            f"{PREFIX}/calculate/batch",
            f"{PREFIX}/calculations",
            f"{PREFIX}/refrigerants",
            f"{PREFIX}/equipment",
            f"{PREFIX}/service-events",
            f"{PREFIX}/leak-rates",
            f"{PREFIX}/compliance/check",
            f"{PREFIX}/compliance",
            f"{PREFIX}/uncertainty",
            f"{PREFIX}/health",
            f"{PREFIX}/stats",
        ]
        for path in expected_paths:
            assert path in route_paths, f"Expected route {path} not found"

    def test_parameterized_routes_exist(self, app):
        """Verify parameterized routes exist."""
        route_paths = set()
        for r in app.routes:
            if hasattr(r, "path") and r.path.startswith(PREFIX):
                route_paths.add(r.path)

        parameterized = [
            f"{PREFIX}/calculations/{{calc_id}}",
            f"{PREFIX}/refrigerants/{{ref_id}}",
            f"{PREFIX}/equipment/{{equip_id}}",
            f"{PREFIX}/audit/{{calc_id}}",
        ]
        for path in parameterized:
            assert path in route_paths, f"Expected route {path} not found"


# ===========================================================================
# Additional calculation endpoint tests
# ===========================================================================


class TestCalculateEndpointsAdditional:
    """Additional tests for calculation endpoints covering edge cases."""

    def test_calculate_endpoint_screening(self, client):
        """POST /calculate works with screening method."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_407C",
                "charge_kg": 10.0,
                "method": "screening",
                "activity_data": 500.0,
                "screening_factor": 0.01,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "calculation_id" in data

    def test_calculate_endpoint_direct_method(self, client):
        """POST /calculate works with direct measurement method."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_134A",
                "charge_kg": 10.0,
                "method": "direct",
                "measured_emissions_kg": 2.5,
            },
        )
        assert resp.status_code == 200

    def test_calculate_endpoint_with_custom_leak_rate(self, client):
        """POST /calculate passes custom_leak_rate_pct through."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
                "custom_leak_rate_pct": 12.0,
            },
        )
        assert resp.status_code == 200

    def test_calculate_endpoint_with_facility(self, client):
        """POST /calculate with facility_id and equipment_id."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
                "equipment_type": "COMMERCIAL_AC",
                "equipment_id": "eq_router_001",
                "facility_id": "fac_router_001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "provenance_hash" in data

    def test_calculate_endpoint_null_charge(self, client):
        """POST /calculate returns 400 when charge_kg is null."""
        resp = client.post(
            f"{PREFIX}/calculate",
            json={"refrigerant_type": "R_410A", "charge_kg": None},
        )
        assert resp.status_code == 400

    def test_calculate_batch_single_item(self, client):
        """POST /calculate/batch with single item list."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={
                "inputs": [
                    {"refrigerant_type": "R_410A", "charge_kg": 25.0},
                ],
            },
        )
        assert resp.status_code == 200

    def test_calculate_batch_non_list_inputs(self, client):
        """POST /calculate/batch returns 400 when inputs is not a list."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={"inputs": "not_a_list"},
        )
        assert resp.status_code == 400


# ===========================================================================
# Additional pagination and filter tests
# ===========================================================================


class TestPaginationAndFilters:
    """Tests for pagination and filtering across list endpoints."""

    def test_list_calculations_pagination(self, client):
        """GET /calculations with skip and limit returns paginated results."""
        # Create 3 calculations
        for i in range(3):
            client.post(
                f"{PREFIX}/calculate",
                json={
                    "refrigerant_type": "R_410A",
                    "charge_kg": float(i + 1),
                },
            )
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"skip": 1, "limit": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_list_calculations_facility_filter(self, client):
        """GET /calculations filters by facility_id."""
        client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "facility_id": "fac_filter_001",
            },
        )
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"facility_id": "fac_filter_001"},
        )
        assert resp.status_code == 200

    def test_list_equipment_filter_by_type(self, client):
        """GET /equipment filters by equipment_type."""
        client.post(
            f"{PREFIX}/equipment",
            json={
                "equipment_type": "SWITCHGEAR",
                "refrigerant_type": "SF6",
                "charge_kg": 15.0,
            },
        )
        client.post(
            f"{PREFIX}/equipment",
            json={
                "equipment_type": "COMMERCIAL_AC",
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        resp = client.get(
            f"{PREFIX}/equipment",
            params={"equipment_type": "SWITCHGEAR"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for item in data:
            assert item["equipment_type"] == "SWITCHGEAR"

    def test_list_equipment_pagination(self, client):
        """GET /equipment with offset and limit."""
        for i in range(3):
            client.post(
                f"{PREFIX}/equipment",
                json={
                    "equipment_id": f"eq_page_{i}",
                    "equipment_type": "COMMERCIAL_AC",
                    "charge_kg": 10.0,
                },
            )
        resp = client.get(
            f"{PREFIX}/equipment",
            params={"offset": 0, "limit": 2},
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 2

    def test_list_leak_rates_filter_by_type(self, client):
        """GET /leak-rates filters by equipment_type."""
        client.post(
            f"{PREFIX}/leak-rates",
            json={
                "equipment_type": "CHILLERS_SCREW",
                "base_rate_pct": 3.0,
            },
        )
        resp = client.get(
            f"{PREFIX}/leak-rates",
            params={"equipment_type": "CHILLERS_SCREW"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for item in data:
            assert item["equipment_type"] == "CHILLERS_SCREW"

    def test_list_service_events_filter_by_type(self, client):
        """GET /service-events filters by event_type."""
        client.post(
            f"{PREFIX}/service-events",
            json={
                "equipment_id": "eq_filter_type",
                "event_type": "installation",
                "quantity_kg": 25.0,
            },
        )
        resp = client.get(
            f"{PREFIX}/service-events",
            params={"event_type": "installation"},
        )
        assert resp.status_code == 200

    def test_list_refrigerants_with_category_filter(self, client):
        """GET /refrigerants filters by category."""
        client.post(
            f"{PREFIX}/refrigerants",
            json={
                "refrigerant_type": "MY_HFC",
                "category": "HFC",
            },
        )
        resp = client.get(
            f"{PREFIX}/refrigerants",
            params={"category": "HFC"},
        )
        assert resp.status_code == 200

    def test_list_compliance_endpoint_with_framework_filter(self, client):
        """GET /compliance filters by framework."""
        # Create a calculation and compliance check
        calc_resp = client.post(
            f"{PREFIX}/calculate",
            json={
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
        )
        calc_id = calc_resp.json()["calculation_id"]
        client.post(
            f"{PREFIX}/compliance/check",
            json={
                "calculation_id": calc_id,
                "frameworks": ["GHG_PROTOCOL", "EU_FGAS_2024"],
            },
        )
        resp = client.get(
            f"{PREFIX}/compliance",
            params={"framework": "GHG_PROTOCOL"},
        )
        assert resp.status_code == 200
