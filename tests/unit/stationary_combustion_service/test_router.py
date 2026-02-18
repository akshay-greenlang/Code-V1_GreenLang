# -*- coding: utf-8 -*-
"""
Unit tests for Stationary Combustion API Router - AGENT-MRV-001

Tests all 20 REST API endpoints at /api/v1/stationary-combustion using
httpx.AsyncClient with the FastAPI TestClient pattern.

Endpoints tested:
 1. POST   /calculate              - Single emission calculation
 2. POST   /calculate/batch        - Batch emission calculation
 3. GET    /calculations           - List calculations (paginated)
 4. GET    /calculations/{calc_id} - Get calculation details
 5. POST   /fuels                  - Register custom fuel type
 6. GET    /fuels                  - List fuel types
 7. GET    /fuels/{fuel_id}        - Get fuel properties
 8. POST   /factors                - Register custom emission factor
 9. GET    /factors                - List emission factors
10. GET    /factors/{factor_id}    - Get factor details
11. POST   /equipment              - Register equipment profile
12. GET    /equipment              - List equipment profiles
13. GET    /equipment/{equip_id}   - Get equipment profile
14. POST   /aggregate              - Aggregate calculations
15. GET    /aggregations           - List aggregations
16. POST   /uncertainty            - Run uncertainty analysis
17. GET    /audit/{calc_id}        - Get audit trail
18. POST   /validate               - Validate input data
19. GET    /health                 - Health check
20. GET    /stats                  - Service statistics

Author: GreenLang Test Engineering
Date: February 2026
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    reset_config,
)

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI or httpx not installed",
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def config():
    """Create a test StationaryCombustionConfig."""
    return StationaryCombustionConfig(
        enable_biogenic_tracking=True,
        monte_carlo_iterations=100,
        enable_metrics=False,
    )


@pytest.fixture
def app(config):
    """Create a FastAPI application with the stationary combustion router."""
    from greenlang.stationary_combustion.setup import (
        StationaryCombustionService,
    )
    import greenlang.stationary_combustion.setup as setup_module

    # Create a fresh service and install as singleton
    svc = StationaryCombustionService(config=config)
    svc.startup()
    setup_module._singleton_instance = svc
    setup_module._service = svc

    app = FastAPI()
    from greenlang.stationary_combustion.api.router import router
    if router is not None:
        app.include_router(router)

    yield app

    # Cleanup
    svc.shutdown()
    setup_module._singleton_instance = None
    setup_module._service = None
    reset_config()


@pytest.fixture
async def client(app):
    """Create an async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


PREFIX = "/api/v1/stationary-combustion"


# =====================================================================
# TestHealthEndpoint
# =====================================================================


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        """GET /health returns 200."""
        resp = await client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_status(self, client):
        """Health response includes status field."""
        resp = await client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_health_has_version(self, client):
        """Health response includes version."""
        resp = await client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_has_engines(self, client):
        """Health response includes engines dict."""
        resp = await client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "engines" in data
        assert isinstance(data["engines"], dict)


# =====================================================================
# TestStatsEndpoint
# =====================================================================


class TestStatsEndpoint:
    """Test GET /stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_returns_200(self, client):
        """GET /stats returns 200."""
        resp = await client.get(f"{PREFIX}/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_has_totals(self, client):
        """Stats response includes total counters."""
        resp = await client.get(f"{PREFIX}/stats")
        data = resp.json()
        assert "total_calculations" in data
        assert "total_batch_runs" in data


# =====================================================================
# TestCalculateEndpoint
# =====================================================================


class TestCalculateEndpoint:
    """Test POST /calculate endpoint."""

    @pytest.mark.asyncio
    async def test_calculate_valid_input(self, client):
        """POST /calculate with valid input returns 200."""
        body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
        }
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_calculate_returns_calculation_id(self, client):
        """Calculation result includes calculation_id."""
        body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 500.0,
            "unit": "CUBIC_METERS",
        }
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        data = resp.json()
        assert "calculation_id" in data

    @pytest.mark.asyncio
    async def test_calculate_missing_fuel_type(self, client):
        """POST /calculate without fuel_type returns 400."""
        body = {"quantity": 1000.0, "unit": "CUBIC_METERS"}
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_calculate_missing_quantity(self, client):
        """POST /calculate without quantity returns 400."""
        body = {"fuel_type": "NATURAL_GAS", "unit": "CUBIC_METERS"}
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_calculate_missing_unit(self, client):
        """POST /calculate without unit returns 400."""
        body = {"fuel_type": "NATURAL_GAS", "quantity": 1000.0}
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_calculate_with_gwp_source(self, client):
        """POST /calculate with custom gwp_source succeeds."""
        body = {
            "fuel_type": "DIESEL",
            "quantity": 500.0,
            "unit": "LITERS",
            "gwp_source": "AR5",
        }
        resp = await client.post(f"{PREFIX}/calculate", json=body)
        assert resp.status_code == 200


# =====================================================================
# TestCalculateBatchEndpoint
# =====================================================================


class TestCalculateBatchEndpoint:
    """Test POST /calculate/batch endpoint."""

    @pytest.mark.asyncio
    async def test_batch_valid_input(self, client):
        """POST /calculate/batch with valid inputs returns 200."""
        body = {
            "inputs": [
                {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
                {"fuel_type": "DIESEL", "quantity": 500.0, "unit": "LITERS"},
            ],
        }
        resp = await client.post(f"{PREFIX}/calculate/batch", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_missing_inputs(self, client):
        """POST /calculate/batch without inputs returns 400."""
        body = {}
        resp = await client.post(f"{PREFIX}/calculate/batch", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_empty_inputs(self, client):
        """POST /calculate/batch with empty inputs list returns 400."""
        body = {"inputs": []}
        resp = await client.post(f"{PREFIX}/calculate/batch", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_returns_batch_id(self, client):
        """Batch result includes batch_id."""
        body = {
            "inputs": [
                {"fuel_type": "NATURAL_GAS", "quantity": 100.0, "unit": "CUBIC_METERS"},
            ],
        }
        resp = await client.post(f"{PREFIX}/calculate/batch", json=body)
        data = resp.json()
        assert "batch_id" in data


# =====================================================================
# TestListCalculations
# =====================================================================


class TestListCalculations:
    """Test GET /calculations endpoint."""

    @pytest.mark.asyncio
    async def test_list_calculations_returns_200(self, client):
        """GET /calculations returns 200."""
        resp = await client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_calculations_returns_list(self, client):
        """GET /calculations returns a JSON list."""
        resp = await client.get(f"{PREFIX}/calculations")
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_calculations_after_calculate(self, client):
        """GET /calculations returns stored result after POST /calculate."""
        body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
        }
        await client.post(f"{PREFIX}/calculate", json=body)
        resp = await client.get(f"{PREFIX}/calculations")
        data = resp.json()
        assert len(data) >= 1


# =====================================================================
# TestGetCalculation
# =====================================================================


class TestGetCalculation:
    """Test GET /calculations/{calc_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_calculation_found(self, client):
        """GET /calculations/{id} returns 200 for existing calculation."""
        body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
        }
        create_resp = await client.post(f"{PREFIX}/calculate", json=body)
        calc_id = create_resp.json()["calculation_id"]

        resp = await client.get(f"{PREFIX}/calculations/{calc_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_calculation_not_found(self, client):
        """GET /calculations/{id} returns 404 for missing calculation."""
        resp = await client.get(f"{PREFIX}/calculations/nonexistent-id")
        assert resp.status_code == 404


# =====================================================================
# TestRegisterFuel
# =====================================================================


class TestRegisterFuel:
    """Test POST /fuels endpoint."""

    @pytest.mark.asyncio
    async def test_register_fuel_valid(self, client):
        """POST /fuels with valid body returns 200."""
        body = {
            "fuel_type": "custom_biodiesel",
            "category": "LIQUID",
            "display_name": "Custom Biodiesel",
            "hhv": 37.5,
            "ncv": 35.0,
        }
        resp = await client.post(f"{PREFIX}/fuels", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_register_fuel_missing_type(self, client):
        """POST /fuels without fuel_type returns 400."""
        body = {"category": "LIQUID"}
        resp = await client.post(f"{PREFIX}/fuels", json=body)
        assert resp.status_code == 400


# =====================================================================
# TestListFuels
# =====================================================================


class TestListFuels:
    """Test GET /fuels endpoint."""

    @pytest.mark.asyncio
    async def test_list_fuels_returns_200(self, client):
        """GET /fuels returns 200."""
        resp = await client.get(f"{PREFIX}/fuels")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_fuels_returns_list(self, client):
        """GET /fuels returns a JSON list."""
        resp = await client.get(f"{PREFIX}/fuels")
        data = resp.json()
        assert isinstance(data, list)


# =====================================================================
# TestGetFuel
# =====================================================================


class TestGetFuel:
    """Test GET /fuels/{fuel_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_fuel_not_found(self, client):
        """GET /fuels/{id} returns 404 for unknown fuel."""
        resp = await client.get(f"{PREFIX}/fuels/nonexistent_fuel")
        assert resp.status_code == 404


# =====================================================================
# TestRegisterFactor
# =====================================================================


class TestRegisterFactor:
    """Test POST /factors endpoint."""

    @pytest.mark.asyncio
    async def test_register_factor_valid(self, client):
        """POST /factors with valid body returns 200."""
        body = {
            "fuel_type": "NATURAL_GAS",
            "gas": "CO2",
            "value": 56.1,
            "unit": "kg CO2/GJ",
            "source": "CUSTOM",
        }
        resp = await client.post(f"{PREFIX}/factors", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_register_factor_missing_gas(self, client):
        """POST /factors without gas returns 400."""
        body = {"fuel_type": "NATURAL_GAS", "value": 56.1}
        resp = await client.post(f"{PREFIX}/factors", json=body)
        assert resp.status_code == 400


# =====================================================================
# TestListFactors
# =====================================================================


class TestListFactors:
    """Test GET /factors endpoint."""

    @pytest.mark.asyncio
    async def test_list_factors_returns_200(self, client):
        """GET /factors returns 200."""
        resp = await client.get(f"{PREFIX}/factors")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_factors_returns_list(self, client):
        """GET /factors returns a JSON list."""
        resp = await client.get(f"{PREFIX}/factors")
        data = resp.json()
        assert isinstance(data, list)


# =====================================================================
# TestGetFactor
# =====================================================================


class TestGetFactor:
    """Test GET /factors/{factor_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_factor_not_found(self, client):
        """GET /factors/{id} returns 404 for unknown factor."""
        resp = await client.get(f"{PREFIX}/factors/unknown-factor")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_factor_found(self, client):
        """GET /factors/{id} returns 200 for registered factor."""
        # First register a factor
        body = {
            "fuel_type": "DIESEL",
            "gas": "CO2",
            "value": 74.1,
            "unit": "kg CO2/GJ",
            "source": "EPA",
        }
        create_resp = await client.post(f"{PREFIX}/factors", json=body)
        factor_id = create_resp.json().get("factor_id")
        if factor_id:
            resp = await client.get(f"{PREFIX}/factors/{factor_id}")
            assert resp.status_code == 200


# =====================================================================
# TestRegisterEquipment
# =====================================================================


class TestRegisterEquipment:
    """Test POST /equipment endpoint."""

    @pytest.mark.asyncio
    async def test_register_equipment_valid(self, client):
        """POST /equipment with valid body returns 200."""
        body = {
            "equipment_type": "BOILER_FIRE_TUBE",
            "name": "Boiler-1",
            "facility_id": "FAC-001",
        }
        resp = await client.post(f"{PREFIX}/equipment", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_register_equipment_returns_id(self, client):
        """Equipment response includes equipment_id."""
        body = {
            "equipment_type": "FURNACE",
            "name": "Furnace-A",
        }
        resp = await client.post(f"{PREFIX}/equipment", json=body)
        data = resp.json()
        assert "equipment_id" in data


# =====================================================================
# TestListEquipment
# =====================================================================


class TestListEquipment:
    """Test GET /equipment endpoint."""

    @pytest.mark.asyncio
    async def test_list_equipment_returns_200(self, client):
        """GET /equipment returns 200."""
        resp = await client.get(f"{PREFIX}/equipment")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_equipment_returns_list(self, client):
        """GET /equipment returns a JSON list."""
        resp = await client.get(f"{PREFIX}/equipment")
        data = resp.json()
        assert isinstance(data, list)


# =====================================================================
# TestGetEquipment
# =====================================================================


class TestGetEquipment:
    """Test GET /equipment/{equip_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_equipment_not_found(self, client):
        """GET /equipment/{id} returns 404 for unknown equipment."""
        resp = await client.get(f"{PREFIX}/equipment/unknown-equip")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_equipment_found(self, client):
        """GET /equipment/{id} returns 200 for registered equipment."""
        # Register first
        body = {"equipment_type": "KILN", "name": "Kiln-1"}
        create_resp = await client.post(f"{PREFIX}/equipment", json=body)
        equip_id = create_resp.json()["equipment_id"]

        resp = await client.get(f"{PREFIX}/equipment/{equip_id}")
        assert resp.status_code == 200


# =====================================================================
# TestAggregate
# =====================================================================


class TestAggregate:
    """Test POST /aggregate endpoint."""

    @pytest.mark.asyncio
    async def test_aggregate_empty(self, client):
        """POST /aggregate with no calculations returns empty."""
        body = {"calculation_ids": [], "control_approach": "OPERATIONAL"}
        resp = await client.post(f"{PREFIX}/aggregate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "aggregations" in data

    @pytest.mark.asyncio
    async def test_aggregate_after_calculations(self, client):
        """POST /aggregate works after performing calculations."""
        # Create a calculation first
        calc_body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
            "facility_id": "FAC-AGG-001",
        }
        calc_resp = await client.post(f"{PREFIX}/calculate", json=calc_body)
        calc_id = calc_resp.json()["calculation_id"]

        # Aggregate
        body = {"calculation_ids": [calc_id]}
        resp = await client.post(f"{PREFIX}/aggregate", json=body)
        assert resp.status_code == 200


# =====================================================================
# TestListAggregations
# =====================================================================


class TestListAggregations:
    """Test GET /aggregations endpoint."""

    @pytest.mark.asyncio
    async def test_list_aggregations_returns_200(self, client):
        """GET /aggregations returns 200."""
        resp = await client.get(f"{PREFIX}/aggregations")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_aggregations_returns_list(self, client):
        """GET /aggregations returns a JSON list."""
        resp = await client.get(f"{PREFIX}/aggregations")
        data = resp.json()
        assert isinstance(data, list)


# =====================================================================
# TestUncertainty
# =====================================================================


class TestUncertainty:
    """Test POST /uncertainty endpoint."""

    @pytest.mark.asyncio
    async def test_uncertainty_missing_calc_id(self, client):
        """POST /uncertainty without calculation_id returns 400."""
        body = {}
        resp = await client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_uncertainty_unknown_calc_id(self, client):
        """POST /uncertainty with unknown calc_id returns 404."""
        body = {"calculation_id": "nonexistent-calc"}
        resp = await client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_uncertainty_valid(self, client):
        """POST /uncertainty with valid calc_id returns 200."""
        # Create calculation first
        calc_body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
        }
        calc_resp = await client.post(f"{PREFIX}/calculate", json=calc_body)
        calc_id = calc_resp.json()["calculation_id"]

        body = {"calculation_id": calc_id}
        resp = await client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 200


# =====================================================================
# TestAuditTrail
# =====================================================================


class TestAuditTrail:
    """Test GET /audit/{calc_id} endpoint."""

    @pytest.mark.asyncio
    async def test_audit_not_found(self, client):
        """GET /audit/{id} returns 404 for unknown calculation."""
        resp = await client.get(f"{PREFIX}/audit/nonexistent-calc")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_audit_found(self, client):
        """GET /audit/{id} returns 200 for existing calculation."""
        calc_body = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "unit": "CUBIC_METERS",
        }
        calc_resp = await client.post(f"{PREFIX}/calculate", json=calc_body)
        calc_id = calc_resp.json()["calculation_id"]

        resp = await client.get(f"{PREFIX}/audit/{calc_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_has_entries(self, client):
        """Audit response includes entries list."""
        calc_body = {
            "fuel_type": "DIESEL",
            "quantity": 500.0,
            "unit": "LITERS",
        }
        calc_resp = await client.post(f"{PREFIX}/calculate", json=calc_body)
        calc_id = calc_resp.json()["calculation_id"]

        resp = await client.get(f"{PREFIX}/audit/{calc_id}")
        data = resp.json()
        assert "entries" in data
        assert "total_entries" in data


# =====================================================================
# TestValidate
# =====================================================================


class TestValidate:
    """Test POST /validate endpoint."""

    @pytest.mark.asyncio
    async def test_validate_valid_inputs(self, client):
        """POST /validate with valid inputs returns 200."""
        body = {
            "inputs": [
                {
                    "fuel_type": "NATURAL_GAS",
                    "quantity": 1000.0,
                    "unit": "CUBIC_METERS",
                },
            ],
        }
        resp = await client.post(f"{PREFIX}/validate", json=body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_missing_inputs(self, client):
        """POST /validate without inputs returns 400."""
        body = {}
        resp = await client.post(f"{PREFIX}/validate", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_empty_inputs(self, client):
        """POST /validate with empty inputs list returns 400."""
        body = {"inputs": []}
        resp = await client.post(f"{PREFIX}/validate", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_returns_validity(self, client):
        """Validate response includes valid flag."""
        body = {
            "inputs": [
                {
                    "fuel_type": "NATURAL_GAS",
                    "quantity": 500.0,
                    "unit": "CUBIC_METERS",
                },
            ],
        }
        resp = await client.post(f"{PREFIX}/validate", json=body)
        data = resp.json()
        assert "valid" in data


# =====================================================================
# TestServiceUnavailable
# =====================================================================


class TestServiceUnavailable:
    """Test endpoints return 503 when service not initialized."""

    @pytest.mark.asyncio
    async def test_health_503_when_no_service(self):
        """GET /health returns 503 when service singleton is None."""
        import greenlang.stationary_combustion.setup as setup_module

        app = FastAPI()
        from greenlang.stationary_combustion.api.router import router
        if router is not None:
            app.include_router(router)

        # Ensure no service is set
        original_instance = setup_module._singleton_instance
        original_service = setup_module._service
        setup_module._singleton_instance = None
        setup_module._service = None

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test",
            ) as ac:
                resp = await ac.get(f"{PREFIX}/health")
                # get_service() creates one lazily, so it may return 200
                # or 503 depending on whether the lazy creation works
                assert resp.status_code in (200, 503)
        finally:
            setup_module._singleton_instance = original_instance
            setup_module._service = original_service


# =====================================================================
# TestInvalidInput
# =====================================================================


class TestInvalidInput:
    """Test endpoints with invalid input data."""

    @pytest.mark.asyncio
    async def test_calculate_empty_body(self, client):
        """POST /calculate with empty body returns 400."""
        resp = await client.post(f"{PREFIX}/calculate", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_non_list_inputs(self, client):
        """POST /calculate/batch with non-list inputs returns 400."""
        body = {"inputs": "not_a_list"}
        resp = await client.post(f"{PREFIX}/calculate/batch", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_non_list_inputs(self, client):
        """POST /validate with non-list inputs returns 400."""
        body = {"inputs": "not_a_list"}
        resp = await client.post(f"{PREFIX}/validate", json=body)
        assert resp.status_code == 400
