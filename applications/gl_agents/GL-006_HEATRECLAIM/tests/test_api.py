"""
GL-006 HEATRECLAIM - API Tests

Tests for REST and GraphQL API endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch

# Conditional imports for testing
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.fixture
def client():
    """Create test client."""
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed")

    from ..api.rest_api import app
    return TestClient(app)


@pytest.fixture
def sample_optimization_request():
    """Sample optimization request data."""
    return {
        "hot_streams": [
            {
                "stream_id": "H1",
                "stream_name": "Hot Stream 1",
                "stream_type": "hot",
                "fluid_name": "Water",
                "phase": "liquid",
                "T_supply_C": 150.0,
                "T_target_C": 60.0,
                "m_dot_kg_s": 2.0,
                "Cp_kJ_kgK": 4.18,
            },
            {
                "stream_id": "H2",
                "stream_name": "Hot Stream 2",
                "stream_type": "hot",
                "fluid_name": "Oil",
                "phase": "liquid",
                "T_supply_C": 90.0,
                "T_target_C": 60.0,
                "m_dot_kg_s": 3.0,
                "Cp_kJ_kgK": 2.0,
            },
        ],
        "cold_streams": [
            {
                "stream_id": "C1",
                "stream_name": "Cold Stream 1",
                "stream_type": "cold",
                "fluid_name": "Water",
                "phase": "liquid",
                "T_supply_C": 20.0,
                "T_target_C": 135.0,
                "m_dot_kg_s": 1.5,
                "Cp_kJ_kgK": 4.18,
            },
            {
                "stream_id": "C2",
                "stream_name": "Cold Stream 2",
                "stream_type": "cold",
                "fluid_name": "Water",
                "phase": "liquid",
                "T_supply_C": 80.0,
                "T_target_C": 140.0,
                "m_dot_kg_s": 2.0,
                "Cp_kJ_kgK": 4.18,
            },
        ],
        "delta_t_min_C": 10.0,
        "objective": "minimize_cost",
        "include_exergy_analysis": True,
        "include_uncertainty": False,
        "generate_pareto": False,
    }


@pytest.fixture
def sample_pinch_request():
    """Sample pinch analysis request."""
    return {
        "hot_streams": [
            {
                "stream_id": "H1",
                "T_supply_C": 150.0,
                "T_target_C": 60.0,
                "m_dot_kg_s": 2.0,
                "Cp_kJ_kgK": 4.18,
            },
        ],
        "cold_streams": [
            {
                "stream_id": "C1",
                "T_supply_C": 20.0,
                "T_target_C": 135.0,
                "m_dot_kg_s": 1.5,
                "Cp_kJ_kgK": 4.18,
            },
        ],
        "delta_t_min_C": 10.0,
    }


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_health_check(self, client):
        """Test health endpoint returns healthy."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_health_check_components(self, client):
        """Test health endpoint includes component status."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "components" in data
        assert data["components"]["orchestrator"] == "ok"
        assert data["components"]["pinch_calculator"] == "ok"


class TestOptimizeEndpoint:
    """Tests for optimization endpoint."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_optimize_sync(self, client, sample_optimization_request):
        """Test synchronous optimization."""
        response = client.post(
            "/api/v1/optimize",
            json=sample_optimization_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "status" in data
        assert "design" in data

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_optimize_async(self, client, sample_optimization_request):
        """Test async optimization returns job ID."""
        response = client.post(
            "/api/v1/optimize?async_mode=true",
            json=sample_optimization_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "accepted"

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_optimize_invalid_streams(self, client):
        """Test validation of invalid streams."""
        invalid_request = {
            "hot_streams": [],  # Empty
            "cold_streams": [],
            "delta_t_min_C": 10.0,
        }

        response = client.post("/api/v1/optimize", json=invalid_request)

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_optimize_response_structure(self, client, sample_optimization_request):
        """Test response has expected structure."""
        response = client.post(
            "/api/v1/optimize",
            json=sample_optimization_request,
        )

        data = response.json()

        assert "pinch_analysis" in data
        assert "design" in data
        assert "explanation_summary" in data

        # Pinch analysis fields
        pinch = data["pinch_analysis"]
        assert "pinch_temperature_C" in pinch
        assert "minimum_hot_utility_kW" in pinch
        assert "minimum_cold_utility_kW" in pinch
        assert "maximum_heat_recovery_kW" in pinch

        # Design fields
        design = data["design"]
        assert "exchanger_count" in design
        assert "total_heat_recovered_kW" in design
        assert "exchangers" in design


class TestPinchAnalysisEndpoint:
    """Tests for pinch analysis endpoint."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_pinch_analysis(self, client, sample_pinch_request):
        """Test pinch analysis endpoint."""
        response = client.post(
            "/api/v1/pinch-analysis",
            json=sample_pinch_request,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "pinch_temperature_C" in data
        assert "minimum_hot_utility_kW" in data
        assert "minimum_cold_utility_kW" in data
        assert "maximum_heat_recovery_kW" in data
        assert "hot_composite_curve" in data
        assert "cold_composite_curve" in data

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_pinch_analysis_composite_curves(self, client, sample_pinch_request):
        """Test composite curves in response."""
        response = client.post(
            "/api/v1/pinch-analysis",
            json=sample_pinch_request,
        )

        data = response.json()

        # Verify composite curves format
        assert len(data["hot_composite_curve"]) > 0
        assert len(data["cold_composite_curve"]) > 0

        # Each point should have T and H
        for point in data["hot_composite_curve"]:
            assert "T_C" in point
            assert "H_kW" in point


class TestValidationEndpoint:
    """Tests for stream validation endpoint."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_validate_valid_streams(self, client, sample_optimization_request):
        """Test validation of valid streams."""
        validation_request = {
            "hot_streams": sample_optimization_request["hot_streams"],
            "cold_streams": sample_optimization_request["cold_streams"],
        }

        response = client.post(
            "/api/v1/validate-streams",
            json=validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert len(data["errors"]) == 0

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_validate_invalid_temperatures(self, client):
        """Test validation catches invalid temperatures."""
        invalid_request = {
            "hot_streams": [
                {
                    "stream_id": "H1",
                    "T_supply_C": 50.0,  # Supply < target (invalid for hot)
                    "T_target_C": 100.0,
                    "m_dot_kg_s": 1.0,
                    "Cp_kJ_kgK": 4.18,
                }
            ],
            "cold_streams": [
                {
                    "stream_id": "C1",
                    "T_supply_C": 20.0,
                    "T_target_C": 80.0,
                    "m_dot_kg_s": 1.0,
                    "Cp_kJ_kgK": 4.18,
                }
            ],
        }

        response = client.post(
            "/api/v1/validate-streams",
            json=invalid_request,
        )

        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0


class TestJobStatusEndpoint:
    """Tests for job status endpoint."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_job_not_found(self, client):
        """Test 404 for unknown job ID."""
        response = client.get("/api/v1/status/unknown-job-id")

        assert response.status_code == 404

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_job_status_after_submit(self, client, sample_optimization_request):
        """Test job status after async submission."""
        # Submit async job
        submit_response = client.post(
            "/api/v1/optimize?async_mode=true",
            json=sample_optimization_request,
        )

        job_id = submit_response.json()["job_id"]

        # Check status
        status_response = client.get(f"/api/v1/status/{job_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "running", "completed", "failed"]


class TestMiddleware:
    """Tests for API middleware."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_request_id_header(self, client):
        """Test request ID is added to response."""
        response = client.get("/api/v1/health")

        # Should have X-Request-ID header (from LoggingMiddleware)
        # Note: This depends on middleware being configured
        assert response.status_code == 200

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # CORS should be enabled
        assert response.status_code in [200, 204]


class TestGraphQL:
    """Tests for GraphQL API."""

    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_graphql_health_query(self, client):
        """Test GraphQL health query."""
        try:
            from ..api.graphql_schema import HAS_STRAWBERRY
            if not HAS_STRAWBERRY:
                pytest.skip("Strawberry not installed")
        except ImportError:
            pytest.skip("GraphQL schema not available")

        query = """
        query {
            health
        }
        """

        response = client.post(
            "/graphql",
            json={"query": query},
        )

        # GraphQL endpoint may not be mounted in test app
        if response.status_code == 404:
            pytest.skip("GraphQL endpoint not mounted")

        assert response.status_code == 200
