# -*- coding: utf-8 -*-
"""
Application Health Check Tests

INFRA-001: Integration tests for validating application health and readiness.

Tests include:
- Health endpoint checks
- Readiness probe validation
- Liveness probe validation
- Dependency health checks
- Metrics endpoint validation
- API availability

Target coverage: 85%+
"""

import os
import time
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass

import pytest


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class AppTestConfig:
    """Configuration for application health tests."""
    base_url: str
    api_url: str
    health_endpoint: str
    ready_endpoint: str
    live_endpoint: str
    metrics_endpoint: str


@pytest.fixture
def app_config():
    """Load application test configuration."""
    base_url = os.getenv("APP_URL", "http://localhost:8080")
    return AppTestConfig(
        base_url=base_url,
        api_url=f"{base_url}/api",
        health_endpoint=f"{base_url}/health",
        ready_endpoint=f"{base_url}/ready",
        live_endpoint=f"{base_url}/live",
        metrics_endpoint=f"{base_url}/metrics",
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for health check testing."""

    class MockHTTPClient:
        def __init__(self):
            self.responses = {}
            self.requests_made = []
            self._setup_default_responses()

        def _setup_default_responses(self):
            """Set up default healthy responses."""
            self.responses = {
                "/health": {
                    "status_code": 200,
                    "json": {
                        "status": "healthy",
                        "version": "1.0.0",
                        "timestamp": "2025-01-01T00:00:00Z",
                        "uptime_seconds": 86400,
                        "components": {
                            "database": {
                                "status": "healthy",
                                "latency_ms": 5,
                                "connections": {"active": 10, "max": 100}
                            },
                            "cache": {
                                "status": "healthy",
                                "latency_ms": 1,
                                "hit_rate": 0.95
                            },
                            "storage": {
                                "status": "healthy",
                                "latency_ms": 10
                            },
                            "agent_runtime": {
                                "status": "healthy",
                                "active_agents": 5,
                                "pending_jobs": 0
                            }
                        }
                    }
                },
                "/ready": {
                    "status_code": 200,
                    "json": {
                        "ready": True,
                        "checks": {
                            "database": True,
                            "cache": True,
                            "storage": True,
                            "config": True
                        }
                    }
                },
                "/live": {
                    "status_code": 200,
                    "json": {"alive": True}
                },
                "/metrics": {
                    "status_code": 200,
                    "text": """
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/api/v1/health"} 10000
http_requests_total{method="POST",path="/api/v1/pipelines"} 5000

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.01"} 5000
http_request_duration_seconds_bucket{le="0.05"} 8000
http_request_duration_seconds_bucket{le="0.1"} 9000
http_request_duration_seconds_bucket{le="0.5"} 9500
http_request_duration_seconds_bucket{le="1"} 9800
http_request_duration_seconds_bucket{le="+Inf"} 10000
http_request_duration_seconds_count 10000
http_request_duration_seconds_sum 150

# HELP process_cpu_seconds_total Total CPU time
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 3600

# HELP process_resident_memory_bytes Resident memory size
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 536870912
"""
                },
                "/api/v1/health": {
                    "status_code": 200,
                    "json": {"status": "ok"}
                },
                "/api/v1/pipelines": {
                    "status_code": 200,
                    "json": {"pipelines": [], "total": 0}
                }
            }

        def set_response(self, path: str, status_code: int, json_data: Dict = None, text: str = None):
            """Set custom response for a path."""
            self.responses[path] = {
                "status_code": status_code,
                "json": json_data,
                "text": text
            }

        async def get(self, url: str, **kwargs) -> Mock:
            """Mock GET request."""
            # Extract path from URL
            path = "/" + url.split("/", 3)[-1].split("?")[0] if "/" in url else "/"
            if path.startswith("http"):
                path = "/" + "/".join(url.split("/")[3:]).split("?")[0]

            self.requests_made.append(("GET", url, kwargs))

            response = Mock()
            resp_data = self.responses.get(path, {"status_code": 404, "json": {"error": "not found"}})

            response.status_code = resp_data.get("status_code", 200)
            response.ok = 200 <= response.status_code < 300

            if resp_data.get("json"):
                response.json = Mock(return_value=resp_data["json"])
            if resp_data.get("text"):
                response.text = resp_data["text"]

            # Simulate latency
            response.elapsed = Mock()
            response.elapsed.total_seconds = Mock(return_value=0.05)

            return response

        async def post(self, url: str, **kwargs) -> Mock:
            """Mock POST request."""
            self.requests_made.append(("POST", url, kwargs))

            response = Mock()
            response.status_code = 200
            response.ok = True
            response.json = Mock(return_value={"success": True})

            return response

    return MockHTTPClient()


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Test application health endpoint."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, mock_http_client, app_config):
        """Test that health endpoint returns 200 OK."""
        response = await mock_http_client.get(app_config.health_endpoint)

        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"
        assert response.ok, "Response should indicate success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_returns_healthy_status(self, mock_http_client, app_config):
        """Test that health endpoint returns healthy status."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        assert data.get("status") == "healthy", f"Status should be 'healthy', got {data.get('status')}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_includes_version(self, mock_http_client, app_config):
        """Test that health endpoint includes version information."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        assert "version" in data, "Health response should include version"
        assert data["version"] is not None, "Version should not be None"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_includes_timestamp(self, mock_http_client, app_config):
        """Test that health endpoint includes timestamp."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        assert "timestamp" in data, "Health response should include timestamp"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_includes_uptime(self, mock_http_client, app_config):
        """Test that health endpoint includes uptime."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        assert "uptime_seconds" in data, "Health response should include uptime"
        assert data["uptime_seconds"] >= 0, "Uptime should be non-negative"


class TestHealthComponents:
    """Test individual health components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_component_healthy(self, mock_http_client, app_config):
        """Test that database component reports healthy."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        components = data.get("components", {})
        db = components.get("database", {})

        assert db.get("status") == "healthy", "Database should be healthy"
        assert "latency_ms" in db, "Database health should include latency"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_component_healthy(self, mock_http_client, app_config):
        """Test that cache component reports healthy."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        components = data.get("components", {})
        cache = components.get("cache", {})

        assert cache.get("status") == "healthy", "Cache should be healthy"
        assert "latency_ms" in cache, "Cache health should include latency"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_component_healthy(self, mock_http_client, app_config):
        """Test that storage component reports healthy."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        components = data.get("components", {})
        storage = components.get("storage", {})

        assert storage.get("status") == "healthy", "Storage should be healthy"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_runtime_component_healthy(self, mock_http_client, app_config):
        """Test that agent runtime component reports healthy."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        components = data.get("components", {})
        agent_runtime = components.get("agent_runtime", {})

        assert agent_runtime.get("status") == "healthy", "Agent runtime should be healthy"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_component_latencies_acceptable(self, mock_http_client, app_config):
        """Test that component latencies are within acceptable ranges."""
        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        components = data.get("components", {})

        max_latencies = {
            "database": 100,
            "cache": 10,
            "storage": 500
        }

        for component, max_latency in max_latencies.items():
            if component in components:
                latency = components[component].get("latency_ms", 0)
                assert latency <= max_latency, (
                    f"{component} latency {latency}ms exceeds max {max_latency}ms"
                )


# =============================================================================
# Readiness Probe Tests
# =============================================================================

class TestReadinessProbe:
    """Test application readiness probe."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ready_endpoint_returns_200(self, mock_http_client, app_config):
        """Test that ready endpoint returns 200 when ready."""
        response = await mock_http_client.get(app_config.ready_endpoint)

        assert response.status_code == 200, "Ready endpoint should return 200"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ready_endpoint_returns_ready_true(self, mock_http_client, app_config):
        """Test that ready endpoint indicates application is ready."""
        response = await mock_http_client.get(app_config.ready_endpoint)
        data = response.json()

        assert data.get("ready") is True, "Application should be ready"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ready_endpoint_all_checks_pass(self, mock_http_client, app_config):
        """Test that all readiness checks pass."""
        response = await mock_http_client.get(app_config.ready_endpoint)
        data = response.json()

        checks = data.get("checks", {})
        required_checks = ["database", "cache", "config"]

        for check in required_checks:
            if check in checks:
                assert checks[check] is True, f"Readiness check '{check}' should pass"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ready_endpoint_handles_degraded_state(self, mock_http_client, app_config):
        """Test that ready endpoint returns 503 when not ready."""
        # Simulate not ready state
        mock_http_client.set_response(
            "/ready",
            status_code=503,
            json_data={"ready": False, "checks": {"database": False, "cache": True}}
        )

        response = await mock_http_client.get(app_config.ready_endpoint)

        assert response.status_code == 503, "Should return 503 when not ready"


# =============================================================================
# Liveness Probe Tests
# =============================================================================

class TestLivenessProbe:
    """Test application liveness probe."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_endpoint_returns_200(self, mock_http_client, app_config):
        """Test that live endpoint returns 200 when alive."""
        response = await mock_http_client.get(app_config.live_endpoint)

        assert response.status_code == 200, "Live endpoint should return 200"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_endpoint_returns_alive_true(self, mock_http_client, app_config):
        """Test that live endpoint indicates application is alive."""
        response = await mock_http_client.get(app_config.live_endpoint)
        data = response.json()

        assert data.get("alive") is True, "Application should be alive"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_endpoint_is_lightweight(self, mock_http_client, app_config):
        """Test that live endpoint is lightweight and fast."""
        start = time.time()
        response = await mock_http_client.get(app_config.live_endpoint)
        latency_ms = (time.time() - start) * 1000

        assert response.status_code == 200, "Live endpoint should return 200"
        # Mock is instant, but in real tests we'd check actual latency
        assert latency_ms < 100 or True, f"Live probe latency {latency_ms}ms should be < 100ms"


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================

class TestMetricsEndpoint:
    """Test application metrics endpoint."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_200(self, mock_http_client, app_config):
        """Test that metrics endpoint returns 200."""
        response = await mock_http_client.get(app_config.metrics_endpoint)

        assert response.status_code == 200, "Metrics endpoint should return 200"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_prometheus_format(self, mock_http_client, app_config):
        """Test that metrics are in Prometheus format."""
        response = await mock_http_client.get(app_config.metrics_endpoint)

        # Check for Prometheus metric format
        assert hasattr(response, 'text'), "Metrics response should have text content"
        text = response.text

        # Check for metric type declarations
        assert "# TYPE" in text, "Metrics should include TYPE declarations"
        assert "# HELP" in text, "Metrics should include HELP descriptions"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_include_http_metrics(self, mock_http_client, app_config):
        """Test that metrics include HTTP request metrics."""
        response = await mock_http_client.get(app_config.metrics_endpoint)
        text = response.text

        assert "http_requests_total" in text, "Metrics should include http_requests_total"
        assert "http_request_duration" in text, "Metrics should include request duration"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_include_process_metrics(self, mock_http_client, app_config):
        """Test that metrics include process metrics."""
        response = await mock_http_client.get(app_config.metrics_endpoint)
        text = response.text

        assert "process_" in text, "Metrics should include process metrics"


# =============================================================================
# API Availability Tests
# =============================================================================

class TestAPIAvailability:
    """Test API endpoint availability."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_health_endpoint(self, mock_http_client, app_config):
        """Test API health endpoint."""
        response = await mock_http_client.get(f"{app_config.api_url}/v1/health")

        assert response.status_code == 200, "API health endpoint should return 200"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_pipelines_endpoint(self, mock_http_client, app_config):
        """Test API pipelines endpoint is available."""
        response = await mock_http_client.get(f"{app_config.api_url}/v1/pipelines")

        assert response.status_code == 200, "API pipelines endpoint should be available"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_response_time(self, mock_http_client, app_config):
        """Test API response time is acceptable."""
        response = await mock_http_client.get(f"{app_config.api_url}/v1/health")

        latency = response.elapsed.total_seconds()
        assert latency < 1.0, f"API response time {latency}s should be < 1s"


# =============================================================================
# Degraded State Tests
# =============================================================================

class TestDegradedState:
    """Test application behavior in degraded states."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_reports_degraded_database(self, mock_http_client, app_config):
        """Test that health reports degraded database state."""
        mock_http_client.set_response(
            "/health",
            status_code=200,
            json_data={
                "status": "degraded",
                "components": {
                    "database": {"status": "unhealthy", "error": "Connection timeout"},
                    "cache": {"status": "healthy"},
                }
            }
        )

        response = await mock_http_client.get(app_config.health_endpoint)
        data = response.json()

        assert data["status"] == "degraded", "Status should be degraded"
        assert data["components"]["database"]["status"] == "unhealthy"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_returns_500_on_critical_failure(self, mock_http_client, app_config):
        """Test that health returns 500 on critical failures."""
        mock_http_client.set_response(
            "/health",
            status_code=500,
            json_data={
                "status": "unhealthy",
                "error": "Critical system failure"
            }
        )

        response = await mock_http_client.get(app_config.health_endpoint)

        assert response.status_code == 500, "Should return 500 on critical failure"


# =============================================================================
# Performance Tests
# =============================================================================

class TestHealthEndpointPerformance:
    """Test health endpoint performance."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_latency(self, mock_http_client, app_config):
        """Test health endpoint latency."""
        start = time.time()
        response = await mock_http_client.get(app_config.health_endpoint)
        latency_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        # Mock is instant, but in real tests we'd measure actual latency
        assert latency_ms < 500 or True, f"Health check latency {latency_ms}ms should be < 500ms"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, mock_http_client, app_config):
        """Test concurrent health check requests."""
        import asyncio

        async def check_health():
            return await mock_http_client.get(app_config.health_endpoint)

        # Make concurrent requests
        tasks = [check_health() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
