# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-028 Investments Agent - REST API Router.

Tests all REST API endpoints using FastAPI TestClient with a mock service
dependency override. Validates status codes, response structure, and
error handling for the 24 endpoints defined in the router.

Coverage:
- POST /calculate/equity (listed equity)
- POST /calculate/private-equity
- POST /calculate/corporate-bond
- POST /calculate/project-finance
- POST /calculate/cre (commercial real estate)
- POST /calculate/mortgage
- POST /calculate/motor-vehicle
- POST /calculate/sovereign-bond
- POST /calculate/portfolio (mixed portfolio)
- POST /calculate/batch
- POST /compliance/check
- GET /calculations/{id}
- GET /calculations
- GET /sector-factors
- GET /country-factors
- GET /pcaf-quality
- GET /asset-classes
- GET /building-benchmarks
- GET /vehicle-factors
- GET /currency-rates
- GET /sovereign-data/{country}
- GET /health
- GET /info
- GET /metrics
- Error responses (400, 404, 422, 500)
- Parametrized tests for asset classes in endpoints

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
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
    from greenlang.investments.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or investments router not available",
)

PREFIX = "/api/v1/investments"


# ===========================================================================
# Mock Service
# ===========================================================================


class MockInvestmentsService:
    """
    Mock service that returns deterministic responses for all API endpoints.

    Every async method returns a dict matching what the router expects,
    so we can test router logic without exercising the real calculation engines.
    """

    # -- calculations -------------------------------------------------------

    async def calculate_equity(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-eq-001",
            "asset_class": "listed_equity",
            "investee_name": data.get("investee_name", "Test Corp"),
            "attribution_factor": 0.0000333,
            "financed_emissions": 1050.0,
            "pcaf_quality_score": 1,
            "provenance_hash": "a" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_private_equity(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-pe-001",
            "asset_class": "private_equity",
            "investee_name": data.get("investee_name", "PE Corp"),
            "attribution_factor": 0.25,
            "financed_emissions": 5750.0,
            "pcaf_quality_score": 2,
            "provenance_hash": "b" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_corporate_bond(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-cb-001",
            "asset_class": "corporate_bond",
            "investee_name": data.get("investee_name", "Bond Corp"),
            "attribution_factor": 0.00015,
            "financed_emissions": 6300.0,
            "pcaf_quality_score": 1,
            "provenance_hash": "c" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_project_finance(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-pf-001",
            "asset_class": "project_finance",
            "project_name": data.get("project_name", "Solar Farm"),
            "attribution_factor": 0.30,
            "financed_emissions": 150.0,
            "pcaf_quality_score": 2,
            "provenance_hash": "d" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_cre(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-cre-001",
            "asset_class": "commercial_real_estate",
            "property_name": data.get("property_name", "Office Tower"),
            "attribution_factor": 0.50,
            "financed_emissions": 250.0,
            "pcaf_quality_score": 2,
            "provenance_hash": "e" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_mortgage(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-mtg-001",
            "asset_class": "mortgage",
            "property_name": data.get("property_name", "123 Oak St"),
            "attribution_factor": 0.75,
            "financed_emissions": 3.5,
            "pcaf_quality_score": 3,
            "provenance_hash": "f" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_motor_vehicle(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-mv-001",
            "asset_class": "motor_vehicle_loan",
            "vehicle_description": data.get("vehicle_description", "Camry"),
            "attribution_factor": 0.714,
            "financed_emissions": 1.7,
            "pcaf_quality_score": 3,
            "provenance_hash": "0" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_sovereign_bond(self, data: dict) -> dict:
        return {
            "calculation_id": "inv-sov-001",
            "asset_class": "sovereign_bond",
            "country": data.get("country", "US"),
            "attribution_factor": 0.00001964,
            "financed_emissions": 102560.0,
            "pcaf_quality_score": 4,
            "provenance_hash": "1" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_portfolio(self, data: dict) -> dict:
        return {
            "portfolio_name": data.get("portfolio_name", "Test Portfolio"),
            "total_financed_emissions": 116015.2,
            "weighted_pcaf_score": 2.5,
            "asset_class_breakdown": {
                "listed_equity": 1050.0,
                "corporate_bond": 6300.0,
                "commercial_real_estate": 250.0,
                "sovereign_bond": 102560.0,
            },
            "provenance_hash": "2" * 64,
            "calculated_at": "2026-01-01T00:00:00Z",
        }

    async def calculate_batch(self, data: dict) -> dict:
        return {
            "batch_id": "batch-001",
            "results": [
                {"calculation_id": "inv-001", "financed_emissions": 1050.0},
            ],
            "total_financed_emissions": 1050.0,
            "count": 1,
            "errors": [],
        }

    async def check_compliance(self, data: dict) -> dict:
        return {
            "framework": data.get("frameworks", ["ghg_protocol"])[0],
            "status": "pass",
            "score": 95.0,
            "findings": [],
            "recommendations": [],
        }

    async def get_calculation(self, calc_id: str) -> dict:
        if calc_id == "not-found":
            return None
        return {
            "calculation_id": calc_id,
            "asset_class": "listed_equity",
            "financed_emissions": 1050.0,
        }

    async def list_calculations(self, **kwargs) -> dict:
        return {
            "calculations": [
                {"calculation_id": "inv-001", "financed_emissions": 1050.0},
            ],
            "total": 1,
        }

    def get_sector_factors(self) -> dict:
        return {
            "energy": 0.850,
            "materials": 0.720,
            "information_technology": 0.080,
        }

    def get_country_factors(self) -> dict:
        return {
            "US": 5222000000,
            "CN": 11500000000,
            "DE": 674000000,
        }

    def get_pcaf_quality_info(self) -> dict:
        return {
            "listed_equity": {1: "Reported, verified", 5: "Estimated"},
        }

    def get_asset_classes(self) -> list:
        return [
            "listed_equity", "corporate_bond", "private_equity",
            "project_finance", "commercial_real_estate", "mortgage",
            "motor_vehicle_loan", "sovereign_bond",
        ]

    def get_building_benchmarks(self) -> dict:
        return {"office": {"temperate": 200.0}}

    def get_vehicle_factors(self) -> dict:
        return {"passenger_car": 0.120, "electric_vehicle": 0.040}

    def get_currency_rates(self) -> dict:
        return {"USD": 1.0, "EUR": 1.085}

    def get_sovereign_data(self, country: str) -> dict:
        return {
            "country": country,
            "gdp_ppp": 25460000000000,
            "total_emissions": 5222000000,
        }

    def health_check(self) -> dict:
        return {
            "status": "healthy",
            "agent_id": "GL-MRV-S3-015",
            "version": "1.0.0",
        }

    def get_agent_info(self) -> dict:
        return {
            "agent_id": "GL-MRV-S3-015",
            "component": "AGENT-MRV-028",
            "version": "1.0.0",
        }

    def get_metrics(self) -> dict:
        return {
            "total_calculations": 42,
            "total_financed_emissions": 500000.0,
        }


# ===========================================================================
# TestClient Setup
# ===========================================================================


@pytest.fixture
def client():
    """Create a TestClient with mocked service."""
    if not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE:
        pytest.skip("FastAPI or router not available")

    app = FastAPI()
    app.include_router(router, prefix=PREFIX)

    mock_svc = MockInvestmentsService()
    app.dependency_overrides[get_service] = lambda: mock_svc

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ===========================================================================
# POST Endpoint Tests
# ===========================================================================


@_SKIP
class TestPostEndpoints:
    """Test POST calculation endpoints."""

    def test_calculate_equity(self, client):
        """Test POST /calculate/equity returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/equity",
            json={
                "asset_class": "listed_equity",
                "investee_name": "Apple Inc.",
                "outstanding_amount": 100000000,
                "evic": 3000000000000,
                "investee_scope1": 22400,
                "investee_scope2": 9100,
                "sector": "information_technology",
                "country": "US",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["financed_emissions"] > 0
        assert len(data["provenance_hash"]) == 64

    def test_calculate_private_equity(self, client):
        """Test POST /calculate/private-equity returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/private-equity",
            json={
                "asset_class": "private_equity",
                "investee_name": "GreenTech",
                "outstanding_amount": 50000000,
                "total_equity_plus_debt": 200000000,
                "investee_scope1": 15000,
                "investee_scope2": 8000,
                "sector": "industrials",
                "country": "US",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["financed_emissions"] > 0

    def test_calculate_corporate_bond(self, client):
        """Test POST /calculate/corporate-bond returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/corporate-bond",
            json={
                "asset_class": "corporate_bond",
                "investee_name": "Tesla",
                "outstanding_amount": 75000000,
                "evic": 500000000000,
                "investee_scope1": 30000,
                "investee_scope2": 12000,
                "sector": "consumer_discretionary",
                "country": "US",
            },
        )
        assert resp.status_code == 200

    def test_calculate_project_finance(self, client):
        """Test POST /calculate/project-finance returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/project-finance",
            json={
                "asset_class": "project_finance",
                "project_name": "Solar Farm",
                "outstanding_amount": 30000000,
                "total_project_cost": 100000000,
                "project_lifetime_years": 25,
                "annual_project_emissions": 500,
                "sector": "utilities",
                "country": "US",
            },
        )
        assert resp.status_code == 200

    def test_calculate_cre(self, client):
        """Test POST /calculate/cre returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/cre",
            json={
                "asset_class": "commercial_real_estate",
                "property_name": "Office Tower",
                "outstanding_amount": 25000000,
                "property_value": 50000000,
                "floor_area_m2": 10000,
                "property_type": "office",
                "country": "US",
            },
        )
        assert resp.status_code == 200

    def test_calculate_mortgage(self, client):
        """Test POST /calculate/mortgage returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/mortgage",
            json={
                "asset_class": "mortgage",
                "property_name": "123 Oak St",
                "outstanding_amount": 300000,
                "property_value": 400000,
                "floor_area_m2": 150,
                "property_type": "residential",
                "country": "US",
            },
        )
        assert resp.status_code == 200

    def test_calculate_motor_vehicle(self, client):
        """Test POST /calculate/motor-vehicle returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/motor-vehicle",
            json={
                "asset_class": "motor_vehicle_loan",
                "vehicle_description": "Toyota Camry",
                "outstanding_amount": 25000,
                "vehicle_value": 35000,
                "vehicle_category": "passenger_car",
                "country": "US",
            },
        )
        assert resp.status_code == 200

    def test_calculate_sovereign_bond(self, client):
        """Test POST /calculate/sovereign-bond returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/sovereign-bond",
            json={
                "asset_class": "sovereign_bond",
                "country": "US",
                "outstanding_amount": 500000000,
                "gdp_ppp": 25460000000000,
                "country_emissions": 5222000000,
            },
        )
        assert resp.status_code == 200

    def test_calculate_portfolio(self, client):
        """Test POST /calculate/portfolio returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/portfolio",
            json={
                "portfolio_name": "Test Portfolio",
                "reporting_year": 2024,
                "investments": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_financed_emissions"] > 0

    def test_calculate_batch(self, client):
        """Test POST /calculate/batch returns 200."""
        resp = client.post(
            f"{PREFIX}/calculate/batch",
            json={
                "investments": [
                    {
                        "asset_class": "listed_equity",
                        "investee_name": "Test",
                        "outstanding_amount": 100,
                    },
                ],
            },
        )
        assert resp.status_code == 200

    def test_compliance_check(self, client):
        """Test POST /compliance/check returns 200."""
        resp = client.post(
            f"{PREFIX}/compliance/check",
            json={
                "frameworks": ["ghg_protocol", "pcaf"],
                "calculation_results": [{"total_co2e": 1000}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pass"


# ===========================================================================
# GET Endpoint Tests
# ===========================================================================


@_SKIP
class TestGetEndpoints:
    """Test GET endpoints."""

    def test_get_calculation_found(self, client):
        """Test GET /calculations/{id} returns 200."""
        resp = client.get(f"{PREFIX}/calculations/inv-eq-001")
        assert resp.status_code == 200

    def test_get_calculation_not_found(self, client):
        """Test GET /calculations/not-found returns 404."""
        resp = client.get(f"{PREFIX}/calculations/not-found")
        assert resp.status_code == 404

    def test_list_calculations(self, client):
        """Test GET /calculations returns 200."""
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200

    def test_get_sector_factors(self, client):
        """Test GET /sector-factors returns 200."""
        resp = client.get(f"{PREFIX}/sector-factors")
        assert resp.status_code == 200
        data = resp.json()
        assert "energy" in data

    def test_get_country_factors(self, client):
        """Test GET /country-factors returns 200."""
        resp = client.get(f"{PREFIX}/country-factors")
        assert resp.status_code == 200

    def test_get_pcaf_quality(self, client):
        """Test GET /pcaf-quality returns 200."""
        resp = client.get(f"{PREFIX}/pcaf-quality")
        assert resp.status_code == 200

    def test_get_asset_classes(self, client):
        """Test GET /asset-classes returns 200."""
        resp = client.get(f"{PREFIX}/asset-classes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8

    def test_get_building_benchmarks(self, client):
        """Test GET /building-benchmarks returns 200."""
        resp = client.get(f"{PREFIX}/building-benchmarks")
        assert resp.status_code == 200

    def test_get_vehicle_factors(self, client):
        """Test GET /vehicle-factors returns 200."""
        resp = client.get(f"{PREFIX}/vehicle-factors")
        assert resp.status_code == 200

    def test_get_currency_rates(self, client):
        """Test GET /currency-rates returns 200."""
        resp = client.get(f"{PREFIX}/currency-rates")
        assert resp.status_code == 200

    def test_get_sovereign_data(self, client):
        """Test GET /sovereign-data/US returns 200."""
        resp = client.get(f"{PREFIX}/sovereign-data/US")
        assert resp.status_code == 200

    def test_health_check(self, client):
        """Test GET /health returns 200."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_agent_info(self, client):
        """Test GET /info returns 200."""
        resp = client.get(f"{PREFIX}/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "GL-MRV-S3-015"

    def test_metrics(self, client):
        """Test GET /metrics returns 200."""
        resp = client.get(f"{PREFIX}/metrics")
        assert resp.status_code == 200


# ===========================================================================
# Error Response Tests
# ===========================================================================


@_SKIP
class TestErrorResponses:
    """Test error response handling."""

    def test_400_invalid_json(self, client):
        """Test 400 for invalid JSON body."""
        resp = client.post(
            f"{PREFIX}/calculate/equity",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in [400, 422]

    def test_422_missing_required_field(self, client):
        """Test 422 for missing required field."""
        resp = client.post(
            f"{PREFIX}/calculate/equity",
            json={"investee_name": "Test"},
        )
        assert resp.status_code == 422

    def test_404_unknown_endpoint(self, client):
        """Test 404 for unknown endpoint."""
        resp = client.get(f"{PREFIX}/nonexistent")
        assert resp.status_code == 404

    def test_422_invalid_asset_class(self, client):
        """Test 422 for invalid asset class value."""
        resp = client.post(
            f"{PREFIX}/calculate/equity",
            json={
                "asset_class": "invalid_class",
                "investee_name": "Test",
                "outstanding_amount": -100,
            },
        )
        assert resp.status_code == 422


# ===========================================================================
# Parametrized Asset Class Tests
# ===========================================================================


@_SKIP
class TestParametrizedAssetClasses:
    """Parametrized tests for asset classes in endpoints."""

    @pytest.mark.parametrize("endpoint,payload", [
        ("equity", {
            "asset_class": "listed_equity",
            "investee_name": "Apple",
            "outstanding_amount": 100000000,
            "evic": 3000000000000,
            "investee_scope1": 22400,
            "investee_scope2": 9100,
            "sector": "information_technology",
            "country": "US",
        }),
        ("corporate-bond", {
            "asset_class": "corporate_bond",
            "investee_name": "Tesla",
            "outstanding_amount": 75000000,
            "evic": 500000000000,
            "investee_scope1": 30000,
            "investee_scope2": 12000,
            "sector": "consumer_discretionary",
            "country": "US",
        }),
        ("cre", {
            "asset_class": "commercial_real_estate",
            "property_name": "Office",
            "outstanding_amount": 25000000,
            "property_value": 50000000,
            "floor_area_m2": 10000,
            "property_type": "office",
            "country": "US",
        }),
        ("sovereign-bond", {
            "asset_class": "sovereign_bond",
            "country": "US",
            "outstanding_amount": 500000000,
            "gdp_ppp": 25460000000000,
            "country_emissions": 5222000000,
        }),
    ])
    def test_all_asset_class_endpoints(self, client, endpoint, payload):
        """Test all asset class calculation endpoints return 200."""
        resp = client.post(
            f"{PREFIX}/calculate/{endpoint}",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["financed_emissions"] > 0

    @pytest.mark.parametrize("endpoint", [
        "sector-factors",
        "country-factors",
        "pcaf-quality",
        "asset-classes",
        "building-benchmarks",
        "vehicle-factors",
        "currency-rates",
        "health",
        "info",
        "metrics",
    ])
    def test_all_get_endpoints(self, client, endpoint):
        """Test all GET reference data endpoints return 200."""
        resp = client.get(f"{PREFIX}/{endpoint}")
        assert resp.status_code == 200
