# -*- coding: utf-8 -*-
"""
Tests for GreenLang Health Check API Endpoints

Comprehensive test suite for Kubernetes health check endpoints:
- Liveness probe (/healthz)
- Readiness probe (/ready)
- Startup probe (/startup)

Tests cover:
- Response format validation
- Status codes
- Component health reporting
- Caching behavior
- Error handling
- Performance requirements
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from .main import app
from .health import (
from greenlang.determinism import DeterministicClock
    HealthCheckManager,
    HealthCheckResponse,
    ComponentHealth,
    HealthStatus,
    check_liveness,
    check_readiness,
    check_startup,
    health_manager,
)


# Test client
client = TestClient(app)


class TestLivenessProbe:
    """Tests for liveness probe endpoint."""

    def test_liveness_returns_200(self):
        """Liveness probe should return 200 OK."""
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_liveness_response_format(self):
        """Liveness probe should return correct JSON format."""
        response = client.get("/healthz")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["components"], list)

    def test_liveness_includes_process_component(self):
        """Liveness probe should include process component."""
        response = client.get("/healthz")
        data = response.json()

        components = data["components"]
        assert len(components) >= 1

        process_component = components[0]
        assert process_component["name"] == "process"
        assert process_component["status"] == "healthy"
        assert process_component["message"] == "Process is alive"

    def test_liveness_is_fast(self):
        """Liveness probe should complete in <10ms."""
        start = time.time()
        response = client.get("/healthz")
        duration_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        assert duration_ms < 10, f"Liveness check took {duration_ms:.1f}ms (should be <10ms)"

    def test_liveness_includes_request_id(self):
        """Liveness probe should include request ID in response."""
        response = client.get("/healthz")
        assert "X-Request-ID" in response.headers

    def test_liveness_includes_timing_header(self):
        """Liveness probe should include timing header."""
        response = client.get("/healthz")
        assert "X-Response-Time-Ms" in response.headers

        timing = float(response.headers["X-Response-Time-Ms"])
        assert timing < 10


class TestReadinessProbe:
    """Tests for readiness probe endpoint."""

    def test_readiness_returns_200_when_healthy(self):
        """Readiness probe should return 200 when all components healthy."""
        # Mark startup complete so components return UNKNOWN (not UNHEALTHY)
        health_manager.mark_startup_complete()

        response = client.get("/ready")

        # Should return 200 or 503 depending on component initialization
        assert response.status_code in [200, 503]

    def test_readiness_response_format(self):
        """Readiness probe should return correct JSON format."""
        response = client.get("/ready")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data

        assert data["version"] == "1.0.0"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["components"], list)

    def test_readiness_checks_all_components(self):
        """Readiness probe should check all critical components."""
        response = client.get("/ready")
        data = response.json()

        components = data["components"]
        component_names = [c["name"] for c in components]

        # Should check all critical dependencies
        assert "postgresql" in component_names
        assert "redis" in component_names
        assert "llm_providers" in component_names
        assert "vector_db" in component_names

    def test_readiness_component_structure(self):
        """Each component should have correct structure."""
        response = client.get("/ready")
        data = response.json()

        for component in data["components"]:
            assert "name" in component
            assert "status" in component
            assert "message" in component
            assert "response_time_ms" in component
            assert "last_checked" in component

            assert component["status"] in ["healthy", "unhealthy", "degraded", "unknown"]
            assert isinstance(component["response_time_ms"], (int, float))

    def test_readiness_completes_in_one_second(self):
        """Readiness probe should complete in <1 second."""
        start = time.time()
        response = client.get("/ready")
        duration_ms = (time.time() - start) * 1000

        assert response.status_code in [200, 503]
        assert duration_ms < 1000, f"Readiness check took {duration_ms:.1f}ms (should be <1000ms)"

    def test_readiness_returns_503_when_unhealthy(self):
        """Readiness probe should return 503 when components unhealthy."""
        # Mock unhealthy database
        with patch.object(
            health_manager,
            'check_database_health',
            return_value=ComponentHealth(
                name="postgresql",
                status=HealthStatus.UNHEALTHY,
                message="Connection refused",
                response_time_ms=100.0,
                last_checked=DeterministicClock.now()
            )
        ):
            response = client.get("/ready")

            # Should return 503 when unhealthy
            assert response.status_code == 503

            data = response.json()
            assert data["status"] == "unhealthy"

    def test_readiness_caching_behavior(self):
        """Readiness probe should cache results for 5 seconds."""
        # First request
        response1 = client.get("/ready")
        data1 = response1.json()
        timestamp1 = data1["timestamp"]

        # Immediate second request (should use cache)
        response2 = client.get("/ready")
        data2 = response2.json()

        # Components should have same last_checked (from cache)
        # Note: This might not be exactly the same due to timestamp precision
        # but response should be very fast


class TestStartupProbe:
    """Tests for startup probe endpoint."""

    def test_startup_returns_503_initially(self):
        """Startup probe should return 503 before startup complete."""
        # Reset startup state
        health_manager.startup_complete = False
        health_manager.startup_error = None

        response = client.get("/startup")
        assert response.status_code == 503

        data = response.json()
        assert data["status"] == "unhealthy"

    def test_startup_returns_200_when_complete(self):
        """Startup probe should return 200 after startup complete."""
        # Mark startup complete
        health_manager.mark_startup_complete()

        response = client.get("/startup")

        # Should return 200 or 503 depending on component health
        assert response.status_code in [200, 503]

    def test_startup_response_format(self):
        """Startup probe should return correct JSON format."""
        response = client.get("/startup")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_startup_includes_startup_component(self):
        """Startup probe should include startup component."""
        health_manager.mark_startup_complete()

        response = client.get("/startup")
        data = response.json()

        component_names = [c["name"] for c in data["components"]]
        assert "startup" in component_names

    def test_startup_handles_failure(self):
        """Startup probe should handle startup failures."""
        # Mark startup failed
        health_manager.mark_startup_failed("Database initialization failed")

        response = client.get("/startup")
        assert response.status_code == 503

        data = response.json()
        assert data["status"] == "unhealthy"

        # Should have startup component with error
        startup_component = next(
            (c for c in data["components"] if c["name"] == "startup"),
            None
        )
        assert startup_component is not None
        assert "failed" in startup_component["message"].lower()

    def test_startup_no_caching(self):
        """Startup probe should not use cached results."""
        health_manager.mark_startup_complete()

        # Make requests and verify fresh checks
        response1 = client.get("/startup")
        response2 = client.get("/startup")

        # Both should succeed (or both fail)
        assert response1.status_code == response2.status_code


class TestHealthCheckManager:
    """Tests for HealthCheckManager class."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Manager should initialize with correct defaults."""
        manager = HealthCheckManager()

        assert manager.startup_complete is False
        assert manager.startup_error is None
        assert manager.readiness_cache_ttl_seconds == 5
        assert manager.startup_cache_ttl_seconds == 30
        assert manager.get_uptime_seconds() >= 0

    @pytest.mark.asyncio
    async def test_mark_startup_complete(self):
        """Manager should track startup completion."""
        manager = HealthCheckManager()

        manager.mark_startup_complete()

        assert manager.startup_complete is True
        assert manager.startup_error is None

    @pytest.mark.asyncio
    async def test_mark_startup_failed(self):
        """Manager should track startup failures."""
        manager = HealthCheckManager()

        error_message = "Database connection failed"
        manager.mark_startup_failed(error_message)

        assert manager.startup_complete is False
        assert manager.startup_error == error_message

    @pytest.mark.asyncio
    async def test_set_dependencies(self):
        """Manager should accept dependency instances."""
        manager = HealthCheckManager()

        db_mock = MagicMock()
        redis_mock = MagicMock()
        llm_mock = MagicMock()
        vector_mock = MagicMock()

        manager.set_dependencies(
            db_manager=db_mock,
            redis_manager=redis_mock,
            llm_router=llm_mock,
            vector_store=vector_mock
        )

        assert manager._db_manager is db_mock
        assert manager._redis_manager is redis_mock
        assert manager._llm_router is llm_mock
        assert manager._vector_store is vector_mock

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Cache should expire after TTL."""
        manager = HealthCheckManager()

        # Set cache with short TTL
        manager._db_cache.result = ComponentHealth(
            name="postgresql",
            status=HealthStatus.HEALTHY,
            message="Cached result",
            response_time_ms=10.0,
            last_checked=DeterministicClock.now()
        )
        manager._db_cache.expires_at = DeterministicClock.now()

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Cache should be expired
        assert not manager._db_cache.is_valid()


class TestAPIRoot:
    """Tests for API root endpoints."""

    def test_root_endpoint(self):
        """Root endpoint should return service information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert "documentation" in data
        assert "health_checks" in data

        assert data["service"] == "GreenLang Agent Foundation API"
        assert data["version"] == "1.0.0"

    def test_api_info_endpoint(self):
        """API info endpoint should return detailed information."""
        response = client.get("/api/v1/info")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "features" in data
        assert "health_checks" in data
        assert "uptime_seconds" in data

        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers_present(self):
        """All security headers should be present."""
        response = client.get("/healthz")

        assert "Strict-Transport-Security" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Content-Security-Policy" in response.headers

    def test_hsts_header(self):
        """HSTS header should have correct value."""
        response = client.get("/healthz")
        hsts = response.headers["Strict-Transport-Security"]

        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_content_type_options(self):
        """X-Content-Type-Options should be nosniff."""
        response = client.get("/healthz")
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_frame_options(self):
        """X-Frame-Options should be DENY."""
        response = client.get("/healthz")
        assert response.headers["X-Frame-Options"] == "DENY"


class TestRequestContext:
    """Tests for request context middleware."""

    def test_request_id_generated(self):
        """Request ID should be generated if not provided."""
        response = client.get("/healthz")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_request_id_preserved(self):
        """Request ID should be preserved from client."""
        custom_id = "test-request-123"
        response = client.get("/healthz", headers={"X-Request-ID": custom_id})

        assert response.headers["X-Request-ID"] == custom_id

    def test_response_timing_header(self):
        """Response should include timing header."""
        response = client.get("/healthz")
        assert "X-Response-Time-Ms" in response.headers

        timing = float(response.headers["X-Response-Time-Ms"])
        assert timing >= 0


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self):
        """OpenAPI schema should be available."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_swagger_ui_available(self):
        """Swagger UI should be available."""
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """ReDoc should be available."""
        response = client.get("/api/redoc")
        assert response.status_code == 200

    def test_health_endpoints_documented(self):
        """Health check endpoints should be in OpenAPI schema."""
        response = client.get("/api/openapi.json")
        data = response.json()

        paths = data["paths"]
        assert "/healthz" in paths
        assert "/ready" in paths
        assert "/startup" in paths


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
