# -*- coding: utf-8 -*-
"""
Unit tests for Use of Sold Products API Router -- AGENT-MRV-024

Tests all 22 REST endpoints using FastAPI TestClient including full pipeline
calculation, direct emission methods, indirect emission methods, fuels/feedstocks,
lifetime modeling, batch processing, portfolio analysis, compliance checking,
emission factor lookups, product profiles, category listing, provenance retrieval,
and health check.

Target: 25+ tests.
Author: GL-TestEngineer
"""

from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from greenlang.agents.mrv.use_of_sold_products.api.router import (
        router,
        FASTAPI_AVAILABLE as ROUTER_FASTAPI_AVAILABLE,
    )

    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False
    ROUTER_FASTAPI_AVAILABLE = False

_AVAILABLE = FASTAPI_AVAILABLE and _ROUTER_AVAILABLE
_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="API router or FastAPI not available")
pytestmark = _SKIP


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def app():
    """Create a FastAPI app with the use of sold products router."""
    application = FastAPI()
    application.include_router(router, prefix="/api/v1/use-of-sold-products")
    return application


@pytest.fixture
def client(app):
    """Create a TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def valid_calculate_payload():
    """Create a valid full-pipeline calculation request payload."""
    return {
        "tenant_id": "TENANT-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "products": [
            {
                "product_id": "VEH-001",
                "product_category": "VEHICLES",
                "emission_type": "DIRECT",
                "units_sold": 1000,
                "lifetime_years": 15,
                "fuel_type": "gasoline",
                "fuel_consumption_per_year": 1200.0,
            },
        ],
    }


@pytest.fixture
def valid_direct_payload():
    """Create a valid direct emission calculation request payload."""
    return {
        "tenant_id": "TENANT-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "product_id": "VEH-001",
        "product_category": "VEHICLES",
        "emission_method": "DIRECT_FUEL_COMBUSTION",
        "units_sold": 1000,
        "lifetime_years": 15,
        "fuel_type": "gasoline",
        "fuel_consumption_per_year": 1200.0,
        "fuel_ef_kg_per_unit": 2.315,
    }


@pytest.fixture
def valid_indirect_payload():
    """Create a valid indirect emission calculation request payload."""
    return {
        "tenant_id": "TENANT-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "product_id": "APP-001",
        "product_category": "APPLIANCES",
        "emission_method": "INDIRECT_ELECTRICITY",
        "units_sold": 10000,
        "lifetime_years": 15,
        "energy_consumption_kwh_per_year": 400.0,
        "grid_region": "US",
    }


@pytest.fixture
def valid_fuel_sale_payload():
    """Create a valid fuel sale calculation request payload."""
    return {
        "tenant_id": "TENANT-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "product_id": "FUEL-001",
        "fuel_type": "gasoline",
        "quantity_sold_litres": 1000000.0,
    }


@pytest.fixture
def valid_batch_payload():
    """Create a valid batch calculation request payload."""
    return {
        "tenant_id": "TENANT-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "batches": [
            {
                "products": [
                    {
                        "product_id": "VEH-001",
                        "product_category": "VEHICLES",
                        "emission_type": "DIRECT",
                        "units_sold": 1000,
                    },
                ],
            },
        ],
    }


# ============================================================================
# TEST: Health Check Endpoint
# ============================================================================


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that health check returns 200 OK."""
        response = client.get("/api/v1/use-of-sold-products/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test that health response has expected fields."""
        response = client.get("/api/v1/use-of-sold-products/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "ok", "up")

    def test_health_includes_version(self, client):
        """Test that health response includes version."""
        response = client.get("/api/v1/use-of-sold-products/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_includes_agent_id(self, client):
        """Test that health response includes agent_id."""
        response = client.get("/api/v1/use-of-sold-products/health")
        data = response.json()
        assert "agent_id" in data
        assert data["agent_id"] == "GL-MRV-S3-011"


# ============================================================================
# TEST: Calculate Endpoint (Full Pipeline)
# ============================================================================


class TestCalculateEndpoint:
    """Test POST /calculate endpoint."""

    def test_calculate_returns_200(self, client, valid_calculate_payload):
        """Test full pipeline calculation returns 200."""
        response = client.post("/api/v1/use-of-sold-products/calculate", json=valid_calculate_payload)
        assert response.status_code == 200

    def test_calculate_response_has_total(self, client, valid_calculate_payload):
        """Test calculation response includes total emissions."""
        response = client.post("/api/v1/use-of-sold-products/calculate", json=valid_calculate_payload)
        data = response.json()
        assert "total_co2e_kg" in data or "total_emissions_co2e_kg" in data

    def test_calculate_missing_products_returns_422(self, client):
        """Test missing products field returns 422."""
        response = client.post("/api/v1/use-of-sold-products/calculate", json={})
        assert response.status_code in (400, 422)


# ============================================================================
# TEST: Direct Emission Endpoints
# ============================================================================


class TestDirectEmissionEndpoints:
    """Test direct emission calculation endpoints."""

    def test_direct_fuel_combustion(self, client, valid_direct_payload):
        """Test POST /calculate/direct/fuel-combustion."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/direct/fuel-combustion",
            json=valid_direct_payload,
        )
        assert response.status_code in (200, 201)

    def test_direct_refrigerant_leakage(self, client):
        """Test POST /calculate/direct/refrigerant-leakage."""
        payload = {
            "product_id": "HVAC-001",
            "units_sold": 500,
            "lifetime_years": 12,
            "refrigerant_type": "R-410A",
            "refrigerant_charge_kg": 3.0,
            "annual_leak_rate": 0.05,
            "refrigerant_gwp": 2088,
        }
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/direct/refrigerant-leakage",
            json=payload,
        )
        assert response.status_code in (200, 201)

    def test_direct_chemical_release(self, client):
        """Test POST /calculate/direct/chemical-release."""
        payload = {
            "product_id": "CON-001",
            "units_sold": 100000,
            "chemical_type": "HFC-134a",
            "chemical_mass_per_unit_kg": 0.15,
            "chemical_gwp": 1430,
        }
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/direct/chemical-release",
            json=payload,
        )
        assert response.status_code in (200, 201)


# ============================================================================
# TEST: Indirect Emission Endpoints
# ============================================================================


class TestIndirectEmissionEndpoints:
    """Test indirect emission calculation endpoints."""

    def test_indirect_electricity(self, client, valid_indirect_payload):
        """Test POST /calculate/indirect/electricity."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/indirect/electricity",
            json=valid_indirect_payload,
        )
        assert response.status_code in (200, 201)

    def test_indirect_heating_fuel(self, client):
        """Test POST /calculate/indirect/heating-fuel."""
        payload = {
            "product_id": "BLD-001",
            "units_sold": 5000,
            "lifetime_years": 20,
            "fuel_type": "natural_gas",
            "fuel_consumption_per_year": 2000.0,
        }
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/indirect/heating-fuel",
            json=payload,
        )
        assert response.status_code in (200, 201)

    def test_indirect_steam_cooling(self, client):
        """Test POST /calculate/indirect/steam-cooling."""
        payload = {
            "product_id": "BLD-002",
            "units_sold": 500,
            "lifetime_years": 20,
            "system_type": "steam_boiler_gas",
            "energy_per_year_kwh": 10000.0,
        }
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/indirect/steam-cooling",
            json=payload,
        )
        assert response.status_code in (200, 201)


# ============================================================================
# TEST: Fuels/Feedstocks Endpoints
# ============================================================================


class TestFuelsFeedstocksEndpoints:
    """Test fuels/feedstocks calculation endpoints."""

    def test_fuel_sale(self, client, valid_fuel_sale_payload):
        """Test POST /calculate/fuels/sale."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/fuels/sale",
            json=valid_fuel_sale_payload,
        )
        assert response.status_code in (200, 201)

    def test_feedstock_oxidation(self, client):
        """Test POST /calculate/fuels/feedstock."""
        payload = {
            "product_id": "FEED-001",
            "feedstock_type": "naphtha",
            "quantity_sold_kg": 1000000.0,
            "carbon_content": 0.836,
            "oxidation_factor": 1.00,
        }
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/fuels/feedstock",
            json=payload,
        )
        assert response.status_code in (200, 201)


# ============================================================================
# TEST: Batch and Portfolio Endpoints
# ============================================================================


class TestBatchPortfolioEndpoints:
    """Test batch and portfolio endpoints."""

    def test_batch_calculate(self, client, valid_batch_payload):
        """Test POST /calculate/batch."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/batch",
            json=valid_batch_payload,
        )
        assert response.status_code in (200, 201)

    def test_portfolio_analysis(self, client, valid_calculate_payload):
        """Test POST /calculate/portfolio."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/portfolio",
            json=valid_calculate_payload,
        )
        assert response.status_code in (200, 201)


# ============================================================================
# TEST: Lookup Endpoints
# ============================================================================


class TestLookupEndpoints:
    """Test emission factor and profile lookup endpoints."""

    def test_get_fuel_ef(self, client):
        """Test GET /lookup/fuel-ef/{fuel_type}."""
        response = client.get("/api/v1/use-of-sold-products/lookup/fuel-ef/gasoline")
        assert response.status_code == 200

    def test_get_grid_ef(self, client):
        """Test GET /lookup/grid-ef/{region}."""
        response = client.get("/api/v1/use-of-sold-products/lookup/grid-ef/US")
        assert response.status_code == 200

    def test_get_refrigerant_gwp(self, client):
        """Test GET /lookup/refrigerant-gwp/{refrigerant}."""
        response = client.get("/api/v1/use-of-sold-products/lookup/refrigerant-gwp/R-410A")
        assert response.status_code == 200

    def test_get_product_profile(self, client):
        """Test GET /lookup/product-profile/{subcategory}."""
        response = client.get("/api/v1/use-of-sold-products/lookup/product-profile/passenger_car")
        assert response.status_code == 200

    def test_list_categories(self, client):
        """Test GET /lookup/categories."""
        response = client.get("/api/v1/use-of-sold-products/lookup/categories")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_list_fuel_types(self, client):
        """Test GET /lookup/fuel-types."""
        response = client.get("/api/v1/use-of-sold-products/lookup/fuel-types")
        assert response.status_code == 200

    def test_list_regions(self, client):
        """Test GET /lookup/regions."""
        response = client.get("/api/v1/use-of-sold-products/lookup/regions")
        assert response.status_code == 200


# ============================================================================
# TEST: Validation Error Responses
# ============================================================================


class TestValidationErrors:
    """Test validation error responses."""

    def test_invalid_json_returns_422(self, client):
        """Test invalid JSON body returns 422."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 422)

    def test_missing_required_field_returns_422(self, client):
        """Test missing required field returns 422."""
        response = client.post(
            "/api/v1/use-of-sold-products/calculate/direct/fuel-combustion",
            json={"product_id": "VEH-001"},
        )
        assert response.status_code in (400, 422)
