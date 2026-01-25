"""
FastAPI Endpoint Integration Tests

Tests for FastAPI REST API endpoints:
- Health check endpoints
- Agent CRUD operations
- Agent execution endpoints
- Authentication flow

Run with: pytest tests/integration/test_api_integration.py -v
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check and readiness endpoints."""

    def test_health_check_returns_200(self, client: TestClient):
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data

    def test_readiness_check_returns_200(self, client: TestClient):
        """Test /ready endpoint returns ready status."""
        response = client.get("/ready")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data

    def test_health_check_response_time(self, client: TestClient):
        """Test health check responds within acceptable time (<100ms)."""
        import time
        start = time.time()
        response = client.get("/health")
        duration_ms = (time.time() - start) * 1000
        assert response.status_code == status.HTTP_200_OK
        assert duration_ms < 100, f"Health check too slow: {duration_ms}ms"


class TestAgentCRUDEndpoints:
    """Test agent registry CRUD endpoints."""

    @pytest.mark.skip(reason="Requires full API implementation")
    def test_create_agent_success(
        self,
        client: TestClient,
        sample_agent_create_request,
        auth_headers,
    ):
        """Test POST /v1/agents creates agent successfully."""
        response = client.post(
            "/v1/agents",
            json=sample_agent_create_request,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.skip(reason="Requires full API implementation")
    def test_list_agents_returns_paginated_results(
        self,
        client: TestClient,
        auth_headers,
    ):
        """Test GET /v1/agents returns paginated list."""
        response = client.get(
            "/v1/agents",
            params={"limit": 10, "offset": 0},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK


class TestAgentExecutionEndpoints:
    """Test agent execution endpoints."""

    @pytest.mark.skip(reason="Requires full API implementation")
    def test_execute_agent_success(
        self,
        client: TestClient,
        auth_headers,
    ):
        """Test POST /v1/agents/{agent_id}/execute starts execution."""
        execution_request = {
            "input_data": {
                "fuel_type": "natural_gas",
                "quantity": 1000,
                "unit": "m3",
                "region": "US",
            },
            "async_mode": False,
            "timeout_seconds": 60,
        }
        response = client.post(
            "/v1/agents/emissions/carbon_calculator_v1/execute",
            json=execution_request,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_202_ACCEPTED


class TestErrorHandling:
    """Test API error handling and responses."""

    def test_not_found_endpoint(self, client: TestClient):
        """Test 404 for unknown endpoint."""
        response = client.get("/v1/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, client: TestClient):
        """Test unsupported HTTP method returns 405."""
        response = client.put("/health")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
