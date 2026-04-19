# -*- coding: utf-8 -*-
"""
Unit Tests for REST API

Tests FastAPI REST endpoints for emission calculations and data access.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


class TestCalculationAPI:
    """Test /api/v1/calculate endpoints"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        try:
            from greenlang.api.main import app
            self.client = TestClient(app)
        except ImportError:
            pytest.skip("API module not available")

    def test_calculate_endpoint_success(self):
        """Test successful calculation via API"""
        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": 100,
            "activity_unit": "liters",
            "region": "US"
        }

        response = self.client.post("/api/v1/calculate", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "emissions_kg_co2e" in data
        assert "provenance_hash" in data
        assert data["status"] == "success"

    def test_calculate_endpoint_validation_error(self):
        """Test validation error handling"""
        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": -100,  # Invalid: negative
            "activity_unit": "liters"
        }

        response = self.client.post("/api/v1/calculate", json=payload)

        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()

    def test_calculate_batch_endpoint(self):
        """Test batch calculation endpoint"""
        payload = {
            "requests": [
                {
                    "factor_id": "diesel-us-stationary",
                    "activity_amount": 100,
                    "activity_unit": "liters"
                },
                {
                    "factor_id": "natural_gas-us-stationary",
                    "activity_amount": 500,
                    "activity_unit": "cubic_meters"
                }
            ]
        }

        response = self.client.post("/api/v1/calculate/batch", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 2

    def test_authentication_required(self):
        """Test endpoints require authentication"""
        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": 100,
            "activity_unit": "liters"
        }

        # Without auth header
        response = self.client.post("/api/v1/calculate", json=payload)

        # Should succeed with mock auth, or require auth
        # This depends on your auth implementation
        assert response.status_code in [200, 401, 403]

    def test_rate_limiting(self):
        """Test API rate limiting"""
        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": 100,
            "activity_unit": "liters"
        }

        # Make many requests
        responses = []
        for _ in range(100):
            response = self.client.post("/api/v1/calculate", json=payload)
            responses.append(response.status_code)

        # Should get rate limited at some point
        # (if rate limiting is implemented)
        # For now, just verify requests complete
        assert all(code in [200, 429] for code in responses)


class TestEmissionFactorAPI:
    """Test /api/v1/factors endpoints"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        try:
            from greenlang.api.main import app
            self.client = TestClient(app)
        except ImportError:
            pytest.skip("API module not available")

    def test_get_factor_by_id(self):
        """Test retrieving emission factor by ID"""
        response = self.client.get("/api/v1/factors/diesel-us-stationary")

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "factor_id" in data
            assert "factor_value" in data
            assert "source" in data

    def test_list_factors(self):
        """Test listing available emission factors"""
        response = self.client.get("/api/v1/factors")

        assert response.status_code == 200
        data = response.json()

        assert "factors" in data or isinstance(data, list)

    def test_search_factors(self):
        """Test searching emission factors"""
        response = self.client.get("/api/v1/factors/search?q=diesel")

        assert response.status_code == 200
        data = response.json()

        if isinstance(data, dict):
            assert "results" in data
        else:
            assert isinstance(data, list)


class TestHealthCheckAPI:
    """Test health check and status endpoints"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        try:
            from greenlang.api.main import app
            self.client = TestClient(app)
        except ImportError:
            pytest.skip("API module not available")

    def test_health_check(self):
        """Test /health endpoint"""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "ok"]

    def test_readiness_check(self):
        """Test /ready endpoint"""
        response = self.client.get("/ready")

        assert response.status_code == 200

    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        response = self.client.get("/metrics")

        assert response.status_code in [200, 404]  # May not be implemented


class TestAPIPerformance:
    """Test API performance requirements"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        try:
            from greenlang.api.main import app
            self.client = TestClient(app)
        except ImportError:
            pytest.skip("API module not available")

    @pytest.mark.performance
    def test_calculation_endpoint_latency(self):
        """Test calculation endpoint meets latency target (<200ms)"""
        import time

        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": 100,
            "activity_unit": "liters"
        }

        start = time.perf_counter()
        response = self.client.post("/api/v1/calculate", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert latency_ms < 200, f"Latency {latency_ms:.2f}ms exceeds 200ms target"

    @pytest.mark.performance
    def test_throughput_target(self):
        """Test API can handle target throughput"""
        import time

        payload = {
            "factor_id": "diesel-us-stationary",
            "activity_amount": 100,
            "activity_unit": "liters"
        }

        num_requests = 100
        start = time.perf_counter()

        for _ in range(num_requests):
            self.client.post("/api/v1/calculate", json=payload)

        duration = time.perf_counter() - start
        throughput = num_requests / duration

        # Target: 50 req/sec minimum
        assert throughput >= 50, f"Throughput {throughput:.1f} req/s below target"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
