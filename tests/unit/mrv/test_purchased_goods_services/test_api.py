"""
Unit tests for Purchased Goods & Services API router (AGENT-MRV-014).

Tests cover:
- Router creation
- Endpoint definitions
- Request model validation
- Health endpoint
- Error handling (404, 422)
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

try:
    from greenlang.purchased_goods_services.api import (
        router,
        get_router,
    )
    from greenlang.purchased_goods_services.setup import get_service
except ImportError:
    pytest.skip("Purchased Goods Services API not available", allow_module_level=True)


@pytest.fixture
def app():
    """Create test FastAPI app."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Create mock service."""
    with patch('greenlang.purchased_goods_services.api.get_service') as mock:
        service = Mock()
        mock.return_value = service
        yield service


class TestRouterCreation:
    """Test router creation and configuration."""

    def test_router_exists(self):
        """Test router object exists."""
        assert router is not None

    def test_get_router_returns_router(self):
        """Test get_router function returns router."""
        router_obj = get_router()
        assert router_obj is not None
        assert hasattr(router_obj, 'routes')

    def test_router_has_prefix(self):
        """Test router has correct prefix."""
        assert router.prefix == "/purchased-goods-services"

    def test_router_has_tags(self):
        """Test router has correct tags."""
        assert "Purchased Goods & Services" in router.tags


class TestEndpointDefinitions:
    """Test all endpoint definitions exist."""

    def test_health_endpoint_exists(self, client):
        """Test health endpoint exists."""
        response = client.get("/purchased-goods-services/health")
        assert response.status_code in [200, 500]  # Exists, may fail if not mocked

    def test_calculate_spend_based_endpoint_exists(self, client):
        """Test calculate-spend-based endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-spend-based", json={})
        assert response.status_code in [200, 422, 500]  # Exists

    def test_calculate_average_data_endpoint_exists(self, client):
        """Test calculate-average-data endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-average-data", json={})
        assert response.status_code in [200, 422, 500]

    def test_calculate_supplier_specific_endpoint_exists(self, client):
        """Test calculate-supplier-specific endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-supplier-specific", json={})
        assert response.status_code in [200, 422, 500]

    def test_calculate_hybrid_endpoint_exists(self, client):
        """Test calculate-hybrid endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-hybrid", json={})
        assert response.status_code in [200, 422, 500]

    def test_run_pipeline_endpoint_exists(self, client):
        """Test run-pipeline endpoint exists."""
        response = client.post("/purchased-goods-services/run-pipeline", json={})
        assert response.status_code in [200, 422, 500]

    def test_lookup_eeio_factor_endpoint_exists(self, client):
        """Test lookup-eeio-factor endpoint exists."""
        response = client.get("/purchased-goods-services/lookup-eeio-factor")
        assert response.status_code in [200, 422, 500]

    def test_lookup_avgdata_factor_endpoint_exists(self, client):
        """Test lookup-avgdata-factor endpoint exists."""
        response = client.get("/purchased-goods-services/lookup-avgdata-factor")
        assert response.status_code in [200, 422, 500]

    def test_check_compliance_endpoint_exists(self, client):
        """Test check-compliance endpoint exists."""
        response = client.post("/purchased-goods-services/check-compliance", json={})
        assert response.status_code in [200, 422, 500]

    def test_analyze_hotspots_endpoint_exists(self, client):
        """Test analyze-hotspots endpoint exists."""
        response = client.post("/purchased-goods-services/analyze-hotspots", json={})
        assert response.status_code in [200, 422, 500]

    def test_calculate_coverage_endpoint_exists(self, client):
        """Test calculate-coverage endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-coverage", json={})
        assert response.status_code in [200, 422, 500]

    def test_batch_process_endpoint_exists(self, client):
        """Test batch-process endpoint exists."""
        response = client.post("/purchased-goods-services/batch-process", json={})
        assert response.status_code in [200, 422, 500]

    def test_export_results_endpoint_exists(self, client):
        """Test export-results endpoint exists."""
        response = client.post("/purchased-goods-services/export-results", json={})
        assert response.status_code in [200, 422, 500]

    def test_get_statistics_endpoint_exists(self, client):
        """Test get-statistics endpoint exists."""
        response = client.get("/purchased-goods-services/statistics")
        assert response.status_code in [200, 500]

    def test_list_eeio_taxonomies_endpoint_exists(self, client):
        """Test list-eeio-taxonomies endpoint exists."""
        response = client.get("/purchased-goods-services/eeio-taxonomies")
        assert response.status_code in [200, 500]

    def test_list_product_categories_endpoint_exists(self, client):
        """Test list-product-categories endpoint exists."""
        response = client.get("/purchased-goods-services/product-categories")
        assert response.status_code in [200, 500]

    def test_get_compliance_frameworks_endpoint_exists(self, client):
        """Test get-compliance-frameworks endpoint exists."""
        response = client.get("/purchased-goods-services/compliance-frameworks")
        assert response.status_code in [200, 500]

    def test_validate_input_endpoint_exists(self, client):
        """Test validate-input endpoint exists."""
        response = client.post("/purchased-goods-services/validate-input", json={})
        assert response.status_code in [200, 422, 500]

    def test_get_method_recommendations_endpoint_exists(self, client):
        """Test get-method-recommendations endpoint exists."""
        response = client.post("/purchased-goods-services/method-recommendations", json={})
        assert response.status_code in [200, 422, 500]

    def test_calculate_uncertainty_endpoint_exists(self, client):
        """Test calculate-uncertainty endpoint exists."""
        response = client.post("/purchased-goods-services/calculate-uncertainty", json={})
        assert response.status_code in [200, 422, 500]


class TestRequestModelValidation:
    """Test request model validation."""

    def test_spend_based_request_validation(self, client, mock_service):
        """Test spend-based request validation."""
        mock_service.calculate_spend_based.return_value = {
            "total_emissions": "1000.0",
            "method": "spend",
        }

        valid_request = {
            "spend_amount": "100000",
            "category": "raw_materials",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=valid_request
        )

        assert response.status_code == 200

    def test_invalid_spend_amount_rejected(self, client):
        """Test invalid spend amount is rejected."""
        invalid_request = {
            "spend_amount": "not_a_number",
            "category": "raw_materials",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=invalid_request
        )

        assert response.status_code == 422

    def test_missing_required_field_rejected(self, client):
        """Test missing required field is rejected."""
        invalid_request = {
            "category": "raw_materials",
            # Missing spend_amount
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=invalid_request
        )

        assert response.status_code == 422

    def test_negative_spend_rejected(self, client):
        """Test negative spend amount is rejected."""
        invalid_request = {
            "spend_amount": "-1000",
            "category": "raw_materials",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=invalid_request
        )

        assert response.status_code == 422


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_success(self, client, mock_service):
        """Test health endpoint returns success."""
        mock_service.health_check.return_value = {
            "status": "healthy",
            "service": "PurchasedGoodsServicesService",
        }

        response = client.get("/purchased-goods-services/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_includes_timestamp(self, client, mock_service):
        """Test health endpoint includes timestamp."""
        mock_service.health_check.return_value = {
            "status": "healthy",
            "service": "PurchasedGoodsServicesService",
            "timestamp": datetime.now().isoformat(),
        }

        response = client.get("/purchased-goods-services/health")

        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data

    def test_health_endpoint_unhealthy_status(self, client, mock_service):
        """Test health endpoint with unhealthy status."""
        mock_service.health_check.return_value = {
            "status": "unhealthy",
            "service": "PurchasedGoodsServicesService",
            "error": "Database connection failed",
        }

        response = client.get("/purchased-goods-services/health")

        # Should still return 200 but with unhealthy status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestErrorHandling:
    """Test error handling."""

    def test_404_on_invalid_endpoint(self, client):
        """Test 404 error on invalid endpoint."""
        response = client.get("/purchased-goods-services/invalid-endpoint")

        assert response.status_code == 404

    def test_422_on_invalid_json(self, client):
        """Test 422 error on invalid JSON."""
        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_500_on_service_error(self, client, mock_service):
        """Test 500 error when service raises exception."""
        mock_service.calculate_spend_based.side_effect = Exception("Database error")

        request_data = {
            "spend_amount": "100000",
            "category": "raw_materials",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=request_data
        )

        assert response.status_code == 500

    def test_error_response_format(self, client, mock_service):
        """Test error response includes proper format."""
        mock_service.calculate_spend_based.side_effect = ValueError("Invalid category")

        request_data = {
            "spend_amount": "100000",
            "category": "invalid_category",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=request_data
        )

        assert response.status_code in [400, 422, 500]
        data = response.json()
        assert "detail" in data or "error" in data


class TestSpendBasedEndpoint:
    """Test spend-based calculation endpoint."""

    def test_spend_based_calculation_success(self, client, mock_service):
        """Test successful spend-based calculation."""
        mock_service.calculate_spend_based.return_value = {
            "total_emissions": "5000.0",
            "emission_factor": "0.05",
            "method": "spend",
            "dqi": "3.0",
        }

        request_data = {
            "spend_amount": "100000",
            "category": "raw_materials",
            "reporting_year": 2024,
            "taxonomy": "naics",
        }

        response = client.post(
            "/purchased-goods-services/calculate-spend-based",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_emissions" in data
        assert float(data["total_emissions"]) == 5000.0


class TestAverageDataEndpoint:
    """Test average-data calculation endpoint."""

    def test_average_data_calculation_success(self, client, mock_service):
        """Test successful average-data calculation."""
        mock_service.calculate_average_data.return_value = {
            "total_emissions": "2500.0",
            "emission_factor": "2.5",
            "method": "avgdata",
            "dqi": "2.0",
        }

        request_data = {
            "product_id": "P001",
            "quantity": "1000",
            "unit": "kg",
            "product_category": "steel",
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/calculate-average-data",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_emissions" in data


class TestSupplierSpecificEndpoint:
    """Test supplier-specific calculation endpoint."""

    def test_supplier_specific_calculation_success(self, client, mock_service):
        """Test successful supplier-specific calculation."""
        mock_service.calculate_supplier_specific.return_value = {
            "total_emissions": "2500.0",
            "calculation_method": "product_ef",
            "dqi": "1.5",
        }

        request_data = {
            "product_id": "P001",
            "quantity": "1000",
            "unit": "kg",
            "emission_factor": "2.5",
            "ef_unit": "kg_co2e_per_kg",
        }

        response = client.post(
            "/purchased-goods-services/calculate-supplier-specific",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_emissions" in data


class TestHybridEndpoint:
    """Test hybrid calculation endpoint."""

    def test_hybrid_calculation_success(self, client, mock_service):
        """Test successful hybrid calculation."""
        mock_service.calculate_hybrid.return_value = {
            "total_emissions": "10000.0",
            "coverage": "85.0",
            "weighted_dqi": "2.0",
            "method_breakdown": {
                "supplier": "5000.0",
                "avgdata": "3000.0",
                "spend": "2000.0",
            },
        }

        request_data = {
            "items": [
                {"id": "P001", "spend": "10000", "category": "raw_materials"},
                {"id": "P002", "spend": "8000", "category": "packaging"},
            ]
        }

        response = client.post(
            "/purchased-goods-services/calculate-hybrid",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_emissions" in data
        assert "coverage" in data


class TestPipelineEndpoint:
    """Test full pipeline endpoint."""

    def test_pipeline_execution_success(self, client, mock_service):
        """Test successful pipeline execution."""
        mock_service.run_pipeline.return_value = {
            "total_emissions": "15000.0",
            "coverage": "95.0",
            "dqi": "1.8",
            "stages": [{"stage": i, "status": "success"} for i in range(1, 11)],
        }

        request_data = {
            "items": [
                {"id": "P001", "spend": "10000", "category": "raw_materials"},
            ],
            "reporting_year": 2024,
        }

        response = client.post(
            "/purchased-goods-services/run-pipeline",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_emissions" in data
        assert "stages" in data
        assert len(data["stages"]) == 10


class TestComplianceEndpoint:
    """Test compliance checking endpoint."""

    def test_compliance_check_success(self, client, mock_service):
        """Test successful compliance check."""
        mock_service.check_compliance.return_value = {
            "ghg_protocol": {"compliant": True, "missing": []},
            "csrd_esrs": {"compliant": True, "missing": []},
            "cdp": {"compliant": False, "missing": [{"field": "verification"}]},
        }

        request_data = {
            "coverage": "95.0",
            "dqi": "1.8",
            "boundary_complete": True,
            "frameworks": ["ghg_protocol", "csrd_esrs", "cdp"],
        }

        response = client.post(
            "/purchased-goods-services/check-compliance",
            json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "ghg_protocol" in data
        assert "csrd_esrs" in data
        assert "cdp" in data


class TestStatisticsEndpoint:
    """Test statistics endpoint."""

    def test_statistics_retrieval_success(self, client, mock_service):
        """Test successful statistics retrieval."""
        mock_service.get_statistics.return_value = {
            "spend_based_calculations": 150,
            "avgdata_calculations": 75,
            "supplier_specific_calculations": 50,
            "pipeline_executions": 25,
        }

        response = client.get("/purchased-goods-services/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "spend_based_calculations" in data
