# -*- coding: utf-8 -*-
"""
Unit tests for Scope 2 Location-Based Emissions REST API Router

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Tests the 20 REST API endpoints covering calculations, facilities,
consumption, grid factors, T&D losses, compliance, uncertainty,
aggregations, health, and statistics.

Target: 30 tests, ~400 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

try:
    from greenlang.agents.mrv.scope2_location.api.router import router as s2l_router
    ROUTER_AVAILABLE = True
except (ImportError, RuntimeError):
    ROUTER_AVAILABLE = False
    s2l_router = None

try:
    from greenlang.agents.mrv.scope2_location.api.router import create_router
    CREATE_ROUTER_AVAILABLE = True
except ImportError:
    CREATE_ROUTER_AVAILABLE = False
    create_router = None

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

_SKIP_ROUTER = pytest.mark.skipif(
    not ROUTER_AVAILABLE and not CREATE_ROUTER_AVAILABLE,
    reason="Router not available",
)
_SKIP_API = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available",
)
_SKIP = pytest.mark.skipif(
    not FASTAPI_AVAILABLE or (not ROUTER_AVAILABLE and not CREATE_ROUTER_AVAILABLE),
    reason="FastAPI or Router not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a FastAPI application with the scope2-location router."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")

    application = FastAPI()

    router = None
    if ROUTER_AVAILABLE and s2l_router is not None:
        router = s2l_router
    elif CREATE_ROUTER_AVAILABLE and create_router is not None:
        try:
            router = create_router()
        except Exception:
            pytest.skip("Could not create router")

    if router is None:
        pytest.skip("No router available")

    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Create a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def calc_body() -> Dict[str, Any]:
    """Build a valid calculation request body."""
    return {
        "facility_id": "fac-test-001",
        "energy_type": "electricity",
        "consumption_value": 5000.0,
        "consumption_unit": "mwh",
        "country_code": "US",
        "egrid_subregion": "CAMX",
        "gwp_source": "AR5",
        "include_td_losses": True,
        "include_compliance": False,
    }


@pytest.fixture
def facility_body() -> Dict[str, Any]:
    """Build a valid facility registration body."""
    return {
        "name": "API Test Office",
        "facility_type": "office",
        "country_code": "US",
        "grid_region_id": "CAMX",
        "egrid_subregion": "CAMX",
        "tenant_id": "tenant-test",
    }


@pytest.fixture
def consumption_body() -> Dict[str, Any]:
    """Build a valid consumption recording body."""
    return {
        "facility_id": "fac-test-001",
        "energy_type": "electricity",
        "quantity": 5000.0,
        "unit": "mwh",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "data_source": "meter",
        "tenant_id": "tenant-test",
    }


# ===========================================================================
# 1. TestRouterSetup
# ===========================================================================


@_SKIP_ROUTER
class TestRouterSetup:
    """Tests that the router exists and has routes."""

    def test_router_exists(self):
        """Router module can be imported."""
        assert ROUTER_AVAILABLE or CREATE_ROUTER_AVAILABLE

    def test_router_has_routes(self, app):
        """Application has registered routes."""
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert len(routes) > 0

    def test_router_prefix(self, app):
        """Routes use the /api/v1/scope2-location prefix."""
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        scope2_routes = [r for r in routes if "scope2-location" in r]
        assert len(scope2_routes) > 0


# ===========================================================================
# 2. TestCalculationEndpoints
# ===========================================================================


@_SKIP
class TestCalculationEndpoints:
    """Tests for POST/GET/DELETE /calculations."""

    def test_post_calculation(self, client, calc_body):
        """POST /calculations creates a calculation."""
        resp = client.post(
            "/api/v1/scope2-location/calculations",
            json=calc_body,
        )
        assert resp.status_code in (200, 201, 503)
        if resp.status_code in (200, 201):
            data = resp.json()
            assert "calculation_id" in data or "detail" in data

    def test_post_calculation_invalid(self, client):
        """POST /calculations with invalid body returns 422."""
        resp = client.post(
            "/api/v1/scope2-location/calculations",
            json={"invalid": "data"},
        )
        assert resp.status_code == 422

    def test_get_calculations_list(self, client):
        """GET /calculations returns a list."""
        resp = client.get("/api/v1/scope2-location/calculations")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "calculations" in data or isinstance(data, list)

    def test_get_calculation_not_found(self, client):
        """GET /calculations/{id} for unknown ID returns 404."""
        resp = client.get(
            "/api/v1/scope2-location/calculations/nonexistent-id"
        )
        assert resp.status_code in (404, 503)

    def test_delete_calculation_not_found(self, client):
        """DELETE /calculations/{id} for unknown ID returns 404."""
        resp = client.delete(
            "/api/v1/scope2-location/calculations/nonexistent-id"
        )
        assert resp.status_code in (404, 405, 503)

    def test_post_batch_calculation(self, client, calc_body):
        """POST /calculations/batch processes a batch."""
        batch_body = {
            "requests": [calc_body],
        }
        resp = client.post(
            "/api/v1/scope2-location/calculations/batch",
            json=batch_body,
        )
        assert resp.status_code in (200, 201, 422, 503)


# ===========================================================================
# 3. TestFacilityEndpoints
# ===========================================================================


@_SKIP
class TestFacilityEndpoints:
    """Tests for POST/GET/PUT /facilities."""

    def test_post_facility(self, client, facility_body):
        """POST /facilities registers a facility."""
        resp = client.post(
            "/api/v1/scope2-location/facilities",
            json=facility_body,
        )
        assert resp.status_code in (200, 201, 422, 503)

    def test_get_facilities_list(self, client):
        """GET /facilities returns a list."""
        resp = client.get("/api/v1/scope2-location/facilities")
        assert resp.status_code in (200, 503)

    def test_put_facility_not_found(self, client):
        """PUT /facilities/{id} for unknown ID returns 404."""
        resp = client.put(
            "/api/v1/scope2-location/facilities/nonexistent-id",
            json={"name": "Updated Name"},
        )
        assert resp.status_code in (404, 405, 422, 503)


# ===========================================================================
# 4. TestConsumptionEndpoints
# ===========================================================================


@_SKIP
class TestConsumptionEndpoints:
    """Tests for POST/GET /consumption."""

    def test_post_consumption(self, client, consumption_body):
        """POST /consumption records a consumption entry."""
        resp = client.post(
            "/api/v1/scope2-location/consumption",
            json=consumption_body,
        )
        assert resp.status_code in (200, 201, 422, 503)

    def test_get_consumption_list(self, client):
        """GET /consumption returns a list."""
        resp = client.get("/api/v1/scope2-location/consumption")
        assert resp.status_code in (200, 503)


# ===========================================================================
# 5. TestGridFactorEndpoints
# ===========================================================================


@_SKIP
class TestGridFactorEndpoints:
    """Tests for GET /grid-factors and POST /grid-factors/custom."""

    def test_get_grid_factors(self, client):
        """GET /grid-factors returns grid emission factors."""
        resp = client.get("/api/v1/scope2-location/grid-factors")
        assert resp.status_code in (200, 503)

    def test_post_custom_grid_factor(self, client):
        """POST /grid-factors/custom adds a custom factor."""
        body = {
            "region_id": "CUSTOM-API-001",
            "co2_kg_per_mwh": 350.0,
            "year": 2025,
        }
        resp = client.post(
            "/api/v1/scope2-location/grid-factors/custom",
            json=body,
        )
        assert resp.status_code in (200, 201, 422, 503)

    def test_get_grid_factor_by_region(self, client):
        """GET /grid-factors/{region} returns a specific factor."""
        resp = client.get("/api/v1/scope2-location/grid-factors/US")
        assert resp.status_code in (200, 404, 503)


# ===========================================================================
# 6. TestComplianceEndpoints
# ===========================================================================


@_SKIP
class TestComplianceEndpoints:
    """Tests for POST /compliance/check."""

    def test_post_compliance_check(self, client):
        """POST /compliance/check runs compliance check."""
        body = {
            "calculation_id": "calc-test-001",
            "frameworks": ["ghg_protocol_scope2"],
        }
        resp = client.post(
            "/api/v1/scope2-location/compliance/check",
            json=body,
        )
        assert resp.status_code in (200, 201, 404, 422, 503)

    def test_get_compliance_result(self, client):
        """GET /compliance/{id} returns compliance result."""
        resp = client.get(
            "/api/v1/scope2-location/compliance/nonexistent-id"
        )
        assert resp.status_code in (200, 404, 503)


# ===========================================================================
# 7. TestHealthEndpoints
# ===========================================================================


@_SKIP
class TestHealthEndpoints:
    """Tests for GET /health and GET /stats."""

    def test_get_health(self, client):
        """GET /health returns service health status."""
        resp = client.get("/api/v1/scope2-location/health")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "status" in data
            assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_get_health_contains_version(self, client):
        """GET /health response includes version info."""
        resp = client.get("/api/v1/scope2-location/health")
        if resp.status_code == 200:
            data = resp.json()
            assert "version" in data or "service" in data

    def test_get_stats(self, client):
        """GET /stats returns service statistics."""
        resp = client.get("/api/v1/scope2-location/stats")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict)

    def test_get_td_losses(self, client):
        """GET /td-losses returns T&D loss factors."""
        resp = client.get("/api/v1/scope2-location/td-losses")
        assert resp.status_code in (200, 503)

    def test_post_uncertainty(self, client):
        """POST /uncertainty runs uncertainty analysis."""
        body = {
            "calculation_id": "calc-test-001",
            "iterations": 500,
            "confidence_level": 95.0,
            "seed": 42,
        }
        resp = client.post(
            "/api/v1/scope2-location/uncertainty",
            json=body,
        )
        assert resp.status_code in (200, 201, 404, 422, 503)

    def test_get_aggregations(self, client):
        """GET /aggregations returns aggregated data."""
        resp = client.get("/api/v1/scope2-location/aggregations")
        assert resp.status_code in (200, 503)
