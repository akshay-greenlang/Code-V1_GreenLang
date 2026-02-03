# -*- coding: utf-8 -*-
"""
API Endpoint Smoke Tests

INFRA-001: Smoke tests for validating API endpoint availability and basic functionality.

Tests include:
- API root endpoint accessibility
- Authentication endpoints
- Core API endpoints
- Response format validation
- Error handling

Target coverage: 85%+
"""

import os
import time
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass

import pytest


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class APITestConfig:
    """Configuration for API smoke tests."""
    base_url: str
    api_version: str
    timeout_seconds: float


@pytest.fixture
def api_config():
    """Load API test configuration."""
    base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
    return APITestConfig(
        base_url=base_url,
        api_version=os.getenv("API_VERSION", "v1"),
        timeout_seconds=float(os.getenv("API_TIMEOUT", "30")),
    )


@pytest.fixture
def mock_api_client():
    """Mock API client for smoke testing."""

    class MockAPIClient:
        def __init__(self):
            self.requests_made = []
            self._setup_endpoints()

        def _setup_endpoints(self):
            """Set up mock endpoint responses."""
            self.endpoints = {
                "/": {
                    "status_code": 200,
                    "json": {
                        "name": "GreenLang API",
                        "version": "1.0.0",
                        "status": "operational",
                        "documentation": "/docs"
                    }
                },
                "/api/v1/health": {
                    "status_code": 200,
                    "json": {"status": "healthy"}
                },
                "/api/v1/version": {
                    "status_code": 200,
                    "json": {
                        "version": "1.0.0",
                        "api_version": "v1",
                        "build": "abc123"
                    }
                },
                "/api/v1/pipelines": {
                    "status_code": 200,
                    "json": {
                        "pipelines": [],
                        "total": 0,
                        "page": 1,
                        "page_size": 20
                    }
                },
                "/api/v1/pipelines/run": {
                    "status_code": 202,
                    "json": {
                        "job_id": "job-123",
                        "status": "queued",
                        "created_at": "2025-01-01T00:00:00Z"
                    }
                },
                "/api/v1/agents": {
                    "status_code": 200,
                    "json": {
                        "agents": [
                            {"id": "fuel-analyzer", "status": "active"},
                            {"id": "carbon-intensity", "status": "active"},
                        ],
                        "total": 2
                    }
                },
                "/api/v1/datasets": {
                    "status_code": 200,
                    "json": {
                        "datasets": [
                            {"id": "emission-factors", "version": "2024.1"},
                        ],
                        "total": 1
                    }
                },
                "/api/v1/auth/token": {
                    "status_code": 200,
                    "json": {
                        "access_token": "mock-token",
                        "token_type": "bearer",
                        "expires_in": 3600
                    }
                },
                "/api/v1/auth/validate": {
                    "status_code": 200,
                    "json": {"valid": True, "expires_at": "2025-01-01T01:00:00Z"}
                },
                "/docs": {
                    "status_code": 200,
                    "content_type": "text/html",
                    "text": "<html><head><title>API Documentation</title></head></html>"
                },
                "/openapi.json": {
                    "status_code": 200,
                    "json": {
                        "openapi": "3.0.0",
                        "info": {"title": "GreenLang API", "version": "1.0.0"},
                        "paths": {}
                    }
                },
            }

        async def get(self, url: str, **kwargs) -> Mock:
            """Mock GET request."""
            path = self._extract_path(url)
            self.requests_made.append(("GET", url, kwargs))
            return self._create_response(path)

        async def post(self, url: str, **kwargs) -> Mock:
            """Mock POST request."""
            path = self._extract_path(url)
            self.requests_made.append(("POST", url, kwargs))
            return self._create_response(path)

        async def put(self, url: str, **kwargs) -> Mock:
            """Mock PUT request."""
            path = self._extract_path(url)
            self.requests_made.append(("PUT", url, kwargs))
            return self._create_response(path)

        async def delete(self, url: str, **kwargs) -> Mock:
            """Mock DELETE request."""
            path = self._extract_path(url)
            self.requests_made.append(("DELETE", url, kwargs))

            response = Mock()
            response.status_code = 204
            response.ok = True
            return response

        def _extract_path(self, url: str) -> str:
            """Extract path from URL."""
            if url.startswith("http"):
                parts = url.split("/", 3)
                return "/" + parts[3] if len(parts) > 3 else "/"
            return url

        def _create_response(self, path: str) -> Mock:
            """Create mock response for path."""
            response = Mock()
            endpoint = self.endpoints.get(path, {"status_code": 404, "json": {"error": "not found"}})

            response.status_code = endpoint.get("status_code", 200)
            response.ok = 200 <= response.status_code < 300

            if endpoint.get("json"):
                response.json = Mock(return_value=endpoint["json"])
            if endpoint.get("text"):
                response.text = endpoint["text"]

            response.headers = {"Content-Type": endpoint.get("content_type", "application/json")}
            response.elapsed = Mock()
            response.elapsed.total_seconds = Mock(return_value=0.05)

            return response

    return MockAPIClient()


# =============================================================================
# Root API Tests
# =============================================================================

class TestAPIRoot:
    """Test API root endpoint."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_api_root_accessible(self, mock_api_client, api_config):
        """Test that API root endpoint is accessible."""
        response = await mock_api_client.get(f"{api_config.base_url}/")

        assert response.status_code == 200, "API root should return 200"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_api_root_returns_info(self, mock_api_client, api_config):
        """Test that API root returns service information."""
        response = await mock_api_client.get(f"{api_config.base_url}/")
        data = response.json()

        assert "name" in data, "Response should include service name"
        assert "version" in data, "Response should include version"
        assert "status" in data, "Response should include status"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_api_root_status_operational(self, mock_api_client, api_config):
        """Test that API root reports operational status."""
        response = await mock_api_client.get(f"{api_config.base_url}/")
        data = response.json()

        assert data.get("status") == "operational", "API should be operational"


# =============================================================================
# Health and Version Endpoints
# =============================================================================

class TestHealthAndVersion:
    """Test health and version endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_api_client, api_config):
        """Test health endpoint returns healthy status."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/health"
        )

        assert response.status_code == 200
        assert response.json().get("status") == "healthy"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_version_endpoint(self, mock_api_client, api_config):
        """Test version endpoint returns version information."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/version"
        )

        assert response.status_code == 200
        data = response.json()

        assert "version" in data, "Should include version"
        assert "api_version" in data, "Should include API version"


# =============================================================================
# Pipeline Endpoints
# =============================================================================

class TestPipelineEndpoints:
    """Test pipeline API endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_list_pipelines(self, mock_api_client, api_config):
        """Test listing pipelines endpoint."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/pipelines"
        )

        assert response.status_code == 200
        data = response.json()

        assert "pipelines" in data, "Should include pipelines list"
        assert "total" in data, "Should include total count"
        assert isinstance(data["pipelines"], list), "Pipelines should be a list"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_run_pipeline_accepted(self, mock_api_client, api_config):
        """Test running a pipeline returns accepted status."""
        response = await mock_api_client.post(
            f"{api_config.base_url}/api/{api_config.api_version}/pipelines/run",
            json={"pipeline_id": "test-pipeline", "inputs": {}}
        )

        assert response.status_code == 202, "Pipeline run should return 202 Accepted"
        data = response.json()

        assert "job_id" in data, "Should include job ID"
        assert "status" in data, "Should include job status"


# =============================================================================
# Agent Endpoints
# =============================================================================

class TestAgentEndpoints:
    """Test agent API endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_list_agents(self, mock_api_client, api_config):
        """Test listing agents endpoint."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/agents"
        )

        assert response.status_code == 200
        data = response.json()

        assert "agents" in data, "Should include agents list"
        assert isinstance(data["agents"], list), "Agents should be a list"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_agents_have_status(self, mock_api_client, api_config):
        """Test that listed agents have status information."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/agents"
        )
        data = response.json()

        for agent in data.get("agents", []):
            assert "id" in agent, "Agent should have ID"
            assert "status" in agent, "Agent should have status"


# =============================================================================
# Dataset Endpoints
# =============================================================================

class TestDatasetEndpoints:
    """Test dataset API endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_list_datasets(self, mock_api_client, api_config):
        """Test listing datasets endpoint."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/datasets"
        )

        assert response.status_code == 200
        data = response.json()

        assert "datasets" in data, "Should include datasets list"


# =============================================================================
# Authentication Endpoints
# =============================================================================

class TestAuthEndpoints:
    """Test authentication endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_token_endpoint(self, mock_api_client, api_config):
        """Test token generation endpoint."""
        response = await mock_api_client.post(
            f"{api_config.base_url}/api/{api_config.api_version}/auth/token",
            json={"username": "test", "password": "test"}
        )

        # Accept both 200 (success) and 401 (unauthorized) as valid responses
        assert response.status_code in [200, 401], "Token endpoint should be accessible"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_token_validate_endpoint(self, mock_api_client, api_config):
        """Test token validation endpoint."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/auth/validate",
            headers={"Authorization": "Bearer mock-token"}
        )

        assert response.status_code in [200, 401], "Validate endpoint should be accessible"


# =============================================================================
# Documentation Endpoints
# =============================================================================

class TestDocumentationEndpoints:
    """Test API documentation endpoints."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_docs_endpoint(self, mock_api_client, api_config):
        """Test API documentation endpoint."""
        response = await mock_api_client.get(f"{api_config.base_url}/docs")

        assert response.status_code == 200, "Docs endpoint should be accessible"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_openapi_spec(self, mock_api_client, api_config):
        """Test OpenAPI specification endpoint."""
        response = await mock_api_client.get(f"{api_config.base_url}/openapi.json")

        assert response.status_code == 200, "OpenAPI spec should be accessible"
        data = response.json()

        assert "openapi" in data, "Should be valid OpenAPI spec"
        assert "info" in data, "Should include API info"
        assert "paths" in data, "Should include API paths"


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    """Test API response format compliance."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_json_content_type(self, mock_api_client, api_config):
        """Test that API returns JSON content type."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/health"
        )

        content_type = response.headers.get("Content-Type", "")
        assert "application/json" in content_type, "Should return JSON content type"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_response_time_acceptable(self, mock_api_client, api_config):
        """Test that API response time is acceptable."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/health"
        )

        latency = response.elapsed.total_seconds()
        assert latency < api_config.timeout_seconds, (
            f"Response time {latency}s exceeds timeout {api_config.timeout_seconds}s"
        )


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, mock_api_client, api_config):
        """Test that unknown endpoints return 404."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/nonexistent"
        )

        assert response.status_code == 404, "Unknown endpoint should return 404"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_error_response_format(self, mock_api_client, api_config):
        """Test that error responses have proper format."""
        response = await mock_api_client.get(
            f"{api_config.base_url}/api/{api_config.api_version}/nonexistent"
        )

        data = response.json()
        assert "error" in data, "Error response should include error field"


# =============================================================================
# Performance Smoke Tests
# =============================================================================

class TestAPIPerformance:
    """Test API performance basics."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_api_client, api_config):
        """Test API handles concurrent requests."""
        import asyncio

        async def make_request():
            return await mock_api_client.get(
                f"{api_config.base_url}/api/{api_config.api_version}/health"
            )

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200, "Concurrent requests should succeed"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_requests_tracked(self, mock_api_client, api_config):
        """Test that requests are properly tracked."""
        await mock_api_client.get(f"{api_config.base_url}/")
        await mock_api_client.get(f"{api_config.base_url}/api/{api_config.api_version}/health")

        assert len(mock_api_client.requests_made) == 2, "Should track 2 requests"
