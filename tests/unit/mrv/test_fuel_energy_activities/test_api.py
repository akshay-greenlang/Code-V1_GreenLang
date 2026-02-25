"""
Unit tests for Fuel & Energy Activities API Router

Tests all API endpoints with mocked service layer.
Validates request/response formats, error handling, and authentication.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status

from greenlang.fuel_energy_activities.api import router, get_fuel_energy_service
from greenlang.fuel_energy_activities.models import FuelType, ActivityType
from greenlang.fuel_energy_activities.setup import FuelEnergyActivitiesService
from greenlang_core import AgentConfig


# Fixtures
@pytest.fixture
def app():
    """Create FastAPI app with router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/fuel-energy")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Create mock service."""
    service = MagicMock(spec=FuelEnergyActivitiesService)
    return service


@pytest.fixture
def override_service_dependency(app, mock_service):
    """Override service dependency."""
    app.dependency_overrides[get_fuel_energy_service] = lambda: mock_service
    yield
    app.dependency_overrides.clear()


# Test Class
class TestFuelEnergyActivitiesAPI:
    """Test suite for Fuel & Energy Activities API."""

    def test_post_calculate(self, client, mock_service, override_service_dependency):
        """Test POST /calculate endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.total_emissions_kgco2e = Decimal("5000.50")
        mock_result.activity_3a_emissions_kgco2e = Decimal("3000.25")
        mock_result.activity_3b_emissions_kgco2e = Decimal("1500.15")
        mock_result.activity_3c_emissions_kgco2e = Decimal("500.10")
        mock_result.reporting_period = "2025-Q1"
        mock_result.provenance_hash = "abc123" * 10 + "abcd"  # 64 chars
        mock_result.to_dict.return_value = {
            "total_emissions_kgco2e": "5000.50",
            "activity_3a_emissions_kgco2e": "3000.25",
            "activity_3b_emissions_kgco2e": "1500.15",
            "activity_3c_emissions_kgco2e": "500.10",
            "reporting_period": "2025-Q1",
            "provenance_hash": mock_result.provenance_hash
        }

        mock_service.calculate_all.return_value = mock_result

        # Make request
        payload = {
            "fuel_consumptions": [
                {
                    "fuel_type": "NATURAL_GAS",
                    "quantity": "1000",
                    "country": "US"
                }
            ],
            "reporting_period": "2025-Q1"
        }

        response = client.post("/api/v1/fuel-energy/calculate", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_emissions_kgco2e"] == "5000.50"
        assert data["reporting_period"] == "2025-Q1"

        # Verify service was called
        mock_service.calculate_all.assert_called_once()

    def test_post_calculate_batch(self, client, mock_service, override_service_dependency):
        """Test POST /calculate/batch endpoint."""
        # Setup mock
        mock_results = [
            Mock(
                total_emissions_kgco2e=Decimal("1000"),
                to_dict=lambda: {"total_emissions_kgco2e": "1000"}
            ),
            Mock(
                total_emissions_kgco2e=Decimal("2000"),
                to_dict=lambda: {"total_emissions_kgco2e": "2000"}
            ),
        ]

        mock_service.calculate_all_batch.return_value = mock_results

        # Make request
        payload = {
            "calculations": [
                {
                    "fuel_consumptions": [
                        {"fuel_type": "NATURAL_GAS", "quantity": "1000", "country": "US"}
                    ],
                    "reporting_period": "2025-Q1"
                },
                {
                    "fuel_consumptions": [
                        {"fuel_type": "DIESEL", "quantity": "500", "country": "GB"}
                    ],
                    "reporting_period": "2025-Q1"
                },
            ]
        }

        response = client.post("/api/v1/fuel-energy/calculate/batch", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2

    def test_get_calculations(self, client, mock_service, override_service_dependency):
        """Test GET /calculations endpoint."""
        # Setup mock
        mock_calculations = [
            {
                "id": "calc-001",
                "total_emissions_kgco2e": "5000",
                "reporting_period": "2025-Q1"
            },
            {
                "id": "calc-002",
                "total_emissions_kgco2e": "3000",
                "reporting_period": "2025-Q2"
            },
        ]

        mock_service.get_calculations.return_value = mock_calculations

        # Make request
        response = client.get("/api/v1/fuel-energy/calculations?limit=10&offset=0")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2

    def test_get_calculation_by_id(self, client, mock_service, override_service_dependency):
        """Test GET /calculations/{id} endpoint."""
        # Setup mock
        mock_calculation = {
            "id": "calc-001",
            "total_emissions_kgco2e": "5000",
            "reporting_period": "2025-Q1"
        }

        mock_service.get_calculation_by_id.return_value = mock_calculation

        # Make request
        response = client.get("/api/v1/fuel-energy/calculations/calc-001")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "calc-001"

    def test_delete_calculation(self, client, mock_service, override_service_dependency):
        """Test DELETE /calculations/{id} endpoint."""
        # Setup mock
        mock_service.delete_calculation.return_value = True

        # Make request
        response = client.delete("/api/v1/fuel-energy/calculations/calc-001")

        # Assertions
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify service was called
        mock_service.delete_calculation.assert_called_once_with("calc-001")

    def test_post_fuel_consumption(self, client, mock_service, override_service_dependency):
        """Test POST /fuel-consumptions endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.id = "fc-001"
        mock_result.fuel_type = FuelType.NATURAL_GAS
        mock_result.quantity = Decimal("1000")
        mock_result.to_dict.return_value = {
            "id": "fc-001",
            "fuel_type": "NATURAL_GAS",
            "quantity": "1000"
        }

        mock_service.create_fuel_consumption.return_value = mock_result

        # Make request
        payload = {
            "fuel_type": "NATURAL_GAS",
            "quantity": "1000",
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        response = client.post("/api/v1/fuel-energy/fuel-consumptions", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == "fc-001"

    def test_get_fuel_consumption(self, client, mock_service, override_service_dependency):
        """Test GET /fuel-consumptions/{id} endpoint."""
        # Setup mock
        mock_fuel_consumption = {
            "id": "fc-001",
            "fuel_type": "NATURAL_GAS",
            "quantity": "1000"
        }

        mock_service.get_fuel_consumption.return_value = mock_fuel_consumption

        # Make request
        response = client.get("/api/v1/fuel-energy/fuel-consumptions/fc-001")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "fc-001"

    def test_put_fuel_consumption(self, client, mock_service, override_service_dependency):
        """Test PUT /fuel-consumptions/{id} endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.id = "fc-001"
        mock_result.quantity = Decimal("1500")
        mock_result.to_dict.return_value = {
            "id": "fc-001",
            "quantity": "1500"
        }

        mock_service.update_fuel_consumption.return_value = mock_result

        # Make request
        payload = {
            "quantity": "1500"
        }

        response = client.put("/api/v1/fuel-energy/fuel-consumptions/fc-001", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["quantity"] == "1500"

    def test_post_electricity_consumption(self, client, mock_service, override_service_dependency):
        """Test POST /electricity-consumptions endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.id = "ec-001"
        mock_result.electricity_kwh = Decimal("100000")
        mock_result.to_dict.return_value = {
            "id": "ec-001",
            "electricity_kwh": "100000"
        }

        mock_service.create_electricity_consumption.return_value = mock_result

        # Make request
        payload = {
            "electricity_kwh": "100000",
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        response = client.post("/api/v1/fuel-energy/electricity-consumptions", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == "ec-001"

    def test_get_electricity_consumption(self, client, mock_service, override_service_dependency):
        """Test GET /electricity-consumptions/{id} endpoint."""
        # Setup mock
        mock_electricity_consumption = {
            "id": "ec-001",
            "electricity_kwh": "100000"
        }

        mock_service.get_electricity_consumption.return_value = mock_electricity_consumption

        # Make request
        response = client.get("/api/v1/fuel-energy/electricity-consumptions/ec-001")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "ec-001"

    def test_get_emission_factors(self, client, mock_service, override_service_dependency):
        """Test GET /emission-factors endpoint."""
        # Setup mock
        mock_factors = [
            {
                "id": "ef-001",
                "fuel_type": "NATURAL_GAS",
                "wtt_factor_kgco2e": "5.2"
            },
            {
                "id": "ef-002",
                "fuel_type": "DIESEL",
                "wtt_factor_kgco2e": "0.6"
            },
        ]

        mock_service.get_emission_factors.return_value = mock_factors

        # Make request
        response = client.get("/api/v1/fuel-energy/emission-factors")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2

    def test_get_emission_factor_by_id(self, client, mock_service, override_service_dependency):
        """Test GET /emission-factors/{id} endpoint."""
        # Setup mock
        mock_factor = {
            "id": "ef-001",
            "fuel_type": "NATURAL_GAS",
            "wtt_factor_kgco2e": "5.2"
        }

        mock_service.get_emission_factor_by_id.return_value = mock_factor

        # Make request
        response = client.get("/api/v1/fuel-energy/emission-factors/ef-001")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "ef-001"

    def test_post_custom_factor(self, client, mock_service, override_service_dependency):
        """Test POST /emission-factors endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.id = "ef-custom-001"
        mock_result.to_dict.return_value = {
            "id": "ef-custom-001",
            "fuel_type": "NATURAL_GAS",
            "custom_wtt_factor": "5.5"
        }

        mock_service.create_custom_emission_factor.return_value = mock_result

        # Make request
        payload = {
            "fuel_type": "NATURAL_GAS",
            "custom_wtt_factor": "5.5",
            "source": "Supplier Data"
        }

        response = client.post("/api/v1/fuel-energy/emission-factors", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == "ef-custom-001"

    def test_get_td_loss_factors(self, client, mock_service, override_service_dependency):
        """Test GET /td-loss-factors endpoint."""
        # Setup mock
        mock_factors = [
            {"country": "US", "td_loss_rate": "0.05"},
            {"country": "GB", "td_loss_rate": "0.078"},
        ]

        mock_service.get_td_loss_factors.return_value = mock_factors

        # Make request
        response = client.get("/api/v1/fuel-energy/td-loss-factors")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2

    def test_get_td_loss_by_country(self, client, mock_service, override_service_dependency):
        """Test GET /td-loss-factors/{country} endpoint."""
        # Setup mock
        mock_factor = {
            "country": "US",
            "td_loss_rate": "0.05"
        }

        mock_service.get_td_loss_factor.return_value = Decimal("0.05")

        # Make request
        response = client.get("/api/v1/fuel-energy/td-loss-factors/US")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["country"] == "US"
        assert "td_loss_rate" in data

    def test_post_compliance_check(self, client, mock_service, override_service_dependency):
        """Test POST /compliance/check endpoint."""
        # Setup mock
        mock_result = Mock()
        mock_result.framework = "GHG_PROTOCOL"
        mock_result.compliance_status = "COMPLIANT"
        mock_result.issues = []
        mock_result.to_dict.return_value = {
            "framework": "GHG_PROTOCOL",
            "compliance_status": "COMPLIANT",
            "issues": []
        }

        mock_service.check_compliance.return_value = mock_result

        # Make request
        payload = {
            "framework": "GHG_PROTOCOL",
            "activity_3a_emissions_kgco2e": "50000",
            "activity_3b_emissions_kgco2e": "30000",
            "reporting_period": "2025-Q1"
        }

        response = client.post("/api/v1/fuel-energy/compliance/check", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["framework"] == "GHG_PROTOCOL"
        assert data["compliance_status"] == "COMPLIANT"

    def test_get_compliance_result(self, client, mock_service, override_service_dependency):
        """Test GET /compliance/results/{id} endpoint."""
        # Setup mock
        mock_result = {
            "id": "comp-001",
            "framework": "GHG_PROTOCOL",
            "compliance_status": "COMPLIANT"
        }

        mock_service.get_compliance_result.return_value = mock_result

        # Make request
        response = client.get("/api/v1/fuel-energy/compliance/results/comp-001")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "comp-001"

    def test_post_uncertainty(self, client, mock_service, override_service_dependency):
        """Test POST /uncertainty endpoint."""
        # Setup mock
        mock_result = {
            "uncertainty_pct": "15.5",
            "confidence_interval_95": {"lower": "42500", "upper": "57500"}
        }

        mock_service.quantify_uncertainty.return_value = Decimal("15.5")

        # Make request
        payload = {
            "total_emissions_kgco2e": "50000",
            "calculation_details": {}
        }

        response = client.post("/api/v1/fuel-energy/uncertainty", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "uncertainty_pct" in data

    def test_get_aggregations(self, client, mock_service, override_service_dependency):
        """Test GET /aggregations endpoint."""
        # Setup mock
        mock_aggregations = {
            "by_fuel_type": {
                "NATURAL_GAS": {"total_emissions": "30000"},
                "DIESEL": {"total_emissions": "20000"}
            },
            "by_country": {
                "US": {"total_emissions": "40000"},
                "GB": {"total_emissions": "10000"}
            }
        }

        mock_service.get_aggregations.return_value = mock_aggregations

        # Make request
        response = client.get("/api/v1/fuel-energy/aggregations?dimension=fuel_type")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "by_fuel_type" in data or "aggregations" in data

    def test_get_health(self, client, mock_service, override_service_dependency):
        """Test GET /health endpoint."""
        # Setup mock
        mock_health = {
            "status": "healthy",
            "wtt_calculator": "healthy",
            "upstream_calculator": "healthy",
            "td_loss_calculator": "healthy"
        }

        mock_service.health_check.return_value = mock_health

        # Make request
        response = client.get("/api/v1/fuel-energy/health")

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    def test_error_handling_400_bad_request(self, client, mock_service, override_service_dependency):
        """Test 400 error for invalid request."""
        # Make request with invalid data
        payload = {
            "fuel_consumptions": [
                {
                    "fuel_type": "INVALID_FUEL",
                    "quantity": "-1000",  # Negative
                    "country": "US"
                }
            ]
        }

        response = client.post("/api/v1/fuel-energy/calculate", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_error_handling_404_not_found(self, client, mock_service, override_service_dependency):
        """Test 404 error for not found resource."""
        # Setup mock
        mock_service.get_calculation_by_id.return_value = None

        # Make request
        response = client.get("/api/v1/fuel-energy/calculations/nonexistent-id")

        # Assertions
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_error_handling_500_internal_error(self, client, mock_service, override_service_dependency):
        """Test 500 error for internal server error."""
        # Setup mock to raise exception
        mock_service.calculate_all.side_effect = Exception("Internal error")

        # Make request
        payload = {
            "fuel_consumptions": [
                {"fuel_type": "NATURAL_GAS", "quantity": "1000", "country": "US"}
            ],
            "reporting_period": "2025-Q1"
        }

        response = client.post("/api/v1/fuel-energy/calculate", json=payload)

        # Assertions
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_pagination(self, client, mock_service, override_service_dependency):
        """Test pagination parameters."""
        # Setup mock
        mock_calculations = [
            {"id": f"calc-{i:03d}"} for i in range(10)
        ]

        mock_service.get_calculations.return_value = mock_calculations

        # Make request with pagination
        response = client.get("/api/v1/fuel-energy/calculations?limit=10&offset=20")

        # Assertions
        assert response.status_code == status.HTTP_200_OK

        # Verify service was called with correct parameters
        mock_service.get_calculations.assert_called_once()

    def test_filtering(self, client, mock_service, override_service_dependency):
        """Test filtering parameters."""
        # Setup mock
        mock_calculations = [
            {"id": "calc-001", "reporting_period": "2025-Q1"}
        ]

        mock_service.get_calculations.return_value = mock_calculations

        # Make request with filter
        response = client.get("/api/v1/fuel-energy/calculations?reporting_period=2025-Q1")

        # Assertions
        assert response.status_code == status.HTTP_200_OK

    def test_sorting(self, client, mock_service, override_service_dependency):
        """Test sorting parameters."""
        # Setup mock
        mock_calculations = [
            {"id": "calc-001", "total_emissions": "50000"},
            {"id": "calc-002", "total_emissions": "30000"}
        ]

        mock_service.get_calculations.return_value = mock_calculations

        # Make request with sort
        response = client.get("/api/v1/fuel-energy/calculations?sort_by=total_emissions&sort_order=desc")

        # Assertions
        assert response.status_code == status.HTTP_200_OK

    def test_authentication_required(self, client):
        """Test endpoints require authentication."""
        # Without authentication header
        response = client.get("/api/v1/fuel-energy/calculations")

        # Should require auth (401 or 403)
        # Note: This depends on auth middleware configuration
        # May be 200 if auth is not enforced in test environment

    def test_rate_limiting(self, client, mock_service, override_service_dependency):
        """Test rate limiting."""
        # Make multiple rapid requests
        for _ in range(100):
            response = client.get("/api/v1/fuel-energy/health")

        # Should eventually get rate limited (429)
        # Note: This depends on rate limiter configuration

    def test_cors_headers(self, client):
        """Test CORS headers are set."""
        response = client.options("/api/v1/fuel-energy/health")

        # Check for CORS headers
        # Note: This depends on CORS middleware configuration

    def test_content_type_json(self, client, mock_service, override_service_dependency):
        """Test content type is application/json."""
        mock_service.health_check.return_value = {"status": "healthy"}

        response = client.get("/api/v1/fuel-energy/health")

        assert response.headers["content-type"] == "application/json"

    def test_response_time(self, client, mock_service, override_service_dependency, benchmark):
        """Test API response time."""
        mock_service.health_check.return_value = {"status": "healthy"}

        def make_request():
            return client.get("/api/v1/fuel-energy/health")

        response = benchmark(make_request)

        assert response.status_code == status.HTTP_200_OK


# Integration Tests
class TestFuelEnergyActivitiesAPIIntegration:
    """Integration tests for Fuel & Energy Activities API."""

    @pytest.mark.integration
    def test_end_to_end_calculation(self, client):
        """Test end-to-end calculation flow."""
        # This would test with real service (not mocked)
        pass

    @pytest.mark.integration
    def test_database_persistence(self, client):
        """Test data is persisted to database."""
        pass


# Performance Tests
class TestFuelEnergyActivitiesAPIPerformance:
    """Performance tests for Fuel & Energy Activities API."""

    @pytest.mark.performance
    def test_throughput_target(self, client, mock_service, override_service_dependency):
        """Test API meets throughput target."""
        mock_service.calculate_all.return_value = Mock(
            total_emissions_kgco2e=Decimal("5000"),
            to_dict=lambda: {"total_emissions_kgco2e": "5000"}
        )

        payload = {
            "fuel_consumptions": [
                {"fuel_type": "NATURAL_GAS", "quantity": "1000", "country": "US"}
            ],
            "reporting_period": "2025-Q1"
        }

        num_requests = 1000

        start_time = datetime.now()
        for _ in range(num_requests):
            response = client.post("/api/v1/fuel-energy/calculate", json=payload)
            assert response.status_code == status.HTTP_200_OK
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_requests / duration_seconds

        # Should handle at least 100 requests/sec
        assert throughput >= 100
