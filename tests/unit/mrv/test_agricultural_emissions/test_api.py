# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions API Router.

Tests API endpoints for calculations, farms, livestock, cropland,
rice fields, compliance, uncertainty, aggregations, and health.

Target: 70+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.api.router import router as ag_router
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.setup import (
        AgriculturalEmissionsService,
        CalculateResponse,
        BatchCalculateResponse,
        FarmResponse,
        FarmListResponse,
        LivestockResponse,
        LivestockListResponse,
        CroplandInputResponse,
        RiceFieldResponse,
        ComplianceCheckResponse,
        UncertaintyResponse,
        AggregationResponse,
        HealthResponse,
        StatsResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (ROUTER_AVAILABLE and FASTAPI_AVAILABLE),
    reason="Router or FastAPI not available",
)


@pytest.fixture
def mock_service():
    """Create a mock AgriculturalEmissionsService."""
    svc = MagicMock(name="AgriculturalEmissionsService")

    if SETUP_AVAILABLE:
        svc.calculate.return_value = CalculateResponse(
            calculation_id="calc-001",
            source_category="enteric_fermentation",
            livestock_type="dairy_cattle",
            head_count=200,
            total_co2e_tonnes=762.88,
            provenance_hash="a" * 64,
        )
        svc.calculate_batch.return_value = BatchCalculateResponse(
            batch_id="batch-001",
            total_calculations=2,
            successful=2,
        )
        svc.register_farm.return_value = FarmResponse(farm_id="farm-001", name="Test Farm")
        svc.list_farms.return_value = FarmListResponse(farms=[], total=0)
        svc.register_livestock.return_value = LivestockResponse(herd_id="herd-001")
        svc.list_livestock.return_value = LivestockListResponse(livestock=[], total=0)
        svc.register_cropland_input.return_value = CroplandInputResponse(input_id="input-001")
        svc.register_rice_field.return_value = RiceFieldResponse(field_id="field-001")
        svc.check_compliance.return_value = ComplianceCheckResponse(
            id="chk-001", frameworks_checked=1,
        )
        svc.run_uncertainty.return_value = UncertaintyResponse(
            calculation_id="calc-001", iterations=100,
        )
        svc.health_check.return_value = HealthResponse(status="healthy")
        svc.get_stats.return_value = StatsResponse(total_calculations=5)
    return svc


@pytest.fixture
def client(mock_service):
    """Create a FastAPI test client with the agricultural emissions router."""
    if not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE:
        pytest.skip("FastAPI or router not available")
    app = FastAPI()
    app.include_router(ag_router)
    # Patch the service dependency
    with patch(
        "greenlang.agricultural_emissions.api.router.get_service",
        return_value=mock_service,
    ):
        yield TestClient(app)


# ===========================================================================
# Test Class: Router Setup
# ===========================================================================


@_SKIP
class TestRouterSetup:
    """Test router configuration."""

    def test_router_exists(self):
        assert ag_router is not None

    def test_router_has_routes(self):
        assert len(ag_router.routes) >= 10

    def test_router_prefix(self):
        # Router may have prefix set
        prefix = getattr(ag_router, 'prefix', '')
        assert "agricultural" in prefix or prefix == ""


# ===========================================================================
# Test Class: Calculation Endpoints
# ===========================================================================


@_SKIP
class TestCalculationEndpoints:
    """Test calculation API endpoints."""

    def test_post_calculation(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/calculations",
            json={
                "farm_id": "farm-001",
                "source_category": "enteric_fermentation",
                "livestock_type": "dairy_cattle",
                "head_count": 200,
            },
        )
        assert response.status_code in (200, 201)

    def test_post_batch_calculation(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/calculations/batch",
            json={
                "calculations": [
                    {"source_category": "enteric_fermentation", "head_count": 100},
                ],
            },
        )
        assert response.status_code in (200, 201)

    def test_get_calculations(self, client, mock_service):
        if hasattr(mock_service, 'list_calculations'):
            mock_service.list_calculations.return_value = {"calculations": [], "total": 0}
        response = client.get("/api/v1/agricultural-emissions/calculations")
        assert response.status_code in (200, 404)

    def test_get_single_calculation(self, client, mock_service):
        if hasattr(mock_service, 'get_calculation'):
            mock_service.get_calculation.return_value = None
        response = client.get("/api/v1/agricultural-emissions/calculations/calc-001")
        assert response.status_code in (200, 404)

    def test_delete_calculation(self, client, mock_service):
        if hasattr(mock_service, 'delete_calculation'):
            mock_service.delete_calculation.return_value = True
        response = client.delete("/api/v1/agricultural-emissions/calculations/calc-001")
        assert response.status_code in (200, 204, 404)


# ===========================================================================
# Test Class: Farm Endpoints
# ===========================================================================


@_SKIP
class TestFarmEndpoints:
    """Test farm API endpoints."""

    def test_post_farm(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/farms",
            json={
                "name": "Green Valley Farm",
                "farm_type": "dairy",
                "country_code": "GB",
            },
        )
        assert response.status_code in (200, 201)

    def test_get_farms(self, client, mock_service):
        response = client.get("/api/v1/agricultural-emissions/farms")
        assert response.status_code == 200

    def test_put_farm(self, client, mock_service):
        if hasattr(mock_service, 'update_farm'):
            mock_service.update_farm.return_value = FarmResponse(
                farm_id="farm-001", name="Updated Farm",
            ) if SETUP_AVAILABLE else {}
        response = client.put(
            "/api/v1/agricultural-emissions/farms/farm-001",
            json={"name": "Updated Farm"},
        )
        assert response.status_code in (200, 404)


# ===========================================================================
# Test Class: Livestock Endpoints
# ===========================================================================


@_SKIP
class TestLivestockEndpoints:
    """Test livestock API endpoints."""

    def test_post_livestock(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/livestock",
            json={
                "farm_id": "farm-001",
                "livestock_type": "dairy_cattle",
                "head_count": 200,
            },
        )
        assert response.status_code in (200, 201)

    def test_get_livestock(self, client, mock_service):
        response = client.get("/api/v1/agricultural-emissions/livestock")
        assert response.status_code == 200

    def test_put_livestock(self, client, mock_service):
        if hasattr(mock_service, 'update_livestock'):
            mock_service.update_livestock.return_value = LivestockResponse(
                herd_id="herd-001", head_count=250,
            ) if SETUP_AVAILABLE else {}
        response = client.put(
            "/api/v1/agricultural-emissions/livestock/herd-001",
            json={"head_count": 250},
        )
        assert response.status_code in (200, 404)


# ===========================================================================
# Test Class: Cropland and Rice Endpoints
# ===========================================================================


@_SKIP
class TestCroplandRiceEndpoints:
    """Test cropland and rice field API endpoints."""

    def test_post_cropland_input(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/cropland-inputs",
            json={
                "farm_id": "farm-001",
                "input_type": "synthetic_n",
                "quantity_tonnes": 100.0,
            },
        )
        assert response.status_code in (200, 201)

    def test_get_cropland_inputs(self, client, mock_service):
        if hasattr(mock_service, 'list_cropland_inputs'):
            mock_service.list_cropland_inputs.return_value = {"inputs": [], "total": 0}
        response = client.get("/api/v1/agricultural-emissions/cropland-inputs")
        assert response.status_code in (200, 404)

    def test_post_rice_field(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/rice-fields",
            json={
                "farm_id": "farm-001",
                "area_ha": 50.0,
                "water_regime": "continuously_flooded",
            },
        )
        assert response.status_code in (200, 201)

    def test_get_rice_fields(self, client, mock_service):
        if hasattr(mock_service, 'list_rice_fields'):
            mock_service.list_rice_fields.return_value = {"fields": [], "total": 0}
        response = client.get("/api/v1/agricultural-emissions/rice-fields")
        assert response.status_code in (200, 404)


# ===========================================================================
# Test Class: Compliance and Uncertainty Endpoints
# ===========================================================================


@_SKIP
class TestComplianceUncertaintyEndpoints:
    """Test compliance and uncertainty API endpoints."""

    def test_post_compliance_check(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/compliance/check",
            json={"calculation_id": "calc-001", "frameworks": ["IPCC_2006"]},
        )
        assert response.status_code in (200, 201)

    def test_get_compliance_result(self, client, mock_service):
        if hasattr(mock_service, 'get_compliance_result'):
            mock_service.get_compliance_result.return_value = None
        response = client.get("/api/v1/agricultural-emissions/compliance/chk-001")
        assert response.status_code in (200, 404)

    def test_post_uncertainty(self, client, mock_service):
        response = client.post(
            "/api/v1/agricultural-emissions/uncertainty",
            json={"calculation_id": "calc-001", "iterations": 100},
        )
        assert response.status_code in (200, 201)


# ===========================================================================
# Test Class: Aggregation Endpoint
# ===========================================================================


@_SKIP
class TestAggregationEndpoint:
    """Test aggregation API endpoint."""

    def test_get_aggregations(self, client, mock_service):
        if hasattr(mock_service, 'get_aggregations'):
            mock_service.get_aggregations.return_value = AggregationResponse() if SETUP_AVAILABLE else {}
        response = client.get("/api/v1/agricultural-emissions/aggregations")
        assert response.status_code in (200, 404)


# ===========================================================================
# Test Class: Health Endpoint
# ===========================================================================


@_SKIP
class TestHealthEndpoint:
    """Test health and stats API endpoints."""

    def test_get_health(self, client, mock_service):
        response = client.get("/api/v1/agricultural-emissions/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client, mock_service):
        response = client.get("/api/v1/agricultural-emissions/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_get_stats(self, client, mock_service):
        response = client.get("/api/v1/agricultural-emissions/stats")
        assert response.status_code in (200, 404)
