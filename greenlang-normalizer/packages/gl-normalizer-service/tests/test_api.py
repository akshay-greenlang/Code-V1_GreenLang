"""
Tests for GL Normalizer Service API endpoints.

This module provides comprehensive tests for all API endpoints including:
- Single value normalization
- Batch normalization
- Async job processing
- Vocabulary listing
- Health checks

Run tests:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from gl_normalizer_service.main import create_app
from gl_normalizer_service.config import Settings


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        env="development",
        debug=True,
        secret_key="test-secret-key-must-be-32-characters-long",
        rate_limit_enabled=False,  # Disable for tests
    )


@pytest.fixture
def app(settings: Settings):
    """Create test application."""
    return create_app(settings=settings)


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    """Create authentication headers for tests."""
    return {"X-API-Key": "dev_test_api_key_12345"}


# ==============================================================================
# Root Endpoint Tests
# ==============================================================================


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client: TestClient):
        """Root endpoint should return service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "GL Normalizer Service"
        assert "version" in data
        assert "api_revision" in data
        assert data["documentation"] == "/docs"


# ==============================================================================
# Health Endpoint Tests
# ==============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check_returns_healthy(self, client: TestClient):
        """Health endpoint should return healthy status."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
        assert "api_revision" in data

    def test_readiness_check(self, client: TestClient):
        """Readiness endpoint should return ready status."""
        response = client.get("/v1/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_liveness_check(self, client: TestClient):
        """Liveness endpoint should return alive status."""
        response = client.get("/v1/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


# ==============================================================================
# Authentication Tests
# ==============================================================================


class TestAuthentication:
    """Tests for authentication."""

    def test_normalize_requires_auth(self, client: TestClient):
        """Normalize endpoint should require authentication."""
        response = client.post(
            "/v1/normalize",
            json={"value": "100", "unit": "kg CO2"},
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "GLNORM-007"

    def test_api_key_authentication(self, client: TestClient, auth_headers: dict):
        """API key authentication should work."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={"value": "100", "unit": "kg CO2"},
        )

        assert response.status_code == 200


# ==============================================================================
# Normalize Endpoint Tests
# ==============================================================================


class TestNormalizeEndpoint:
    """Tests for single value normalization endpoint."""

    def test_normalize_basic_value(self, client: TestClient, auth_headers: dict):
        """Should normalize a basic value."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={
                "value": "1500",
                "unit": "kg CO2",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "canonical_value" in data
        assert "canonical_unit" in data
        assert "confidence" in data
        assert "needs_review" in data
        assert "audit_id" in data
        assert "api_revision" in data

    def test_normalize_with_target_unit(self, client: TestClient, auth_headers: dict):
        """Should normalize with specified target unit."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={
                "value": "1500",
                "unit": "kg CO2",
                "target_unit": "metric_ton_co2e",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["canonical_unit"] == "metric_ton_co2e"

    def test_normalize_with_context(self, client: TestClient, auth_headers: dict):
        """Should accept context for normalization."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={
                "value": "1500",
                "unit": "kg CO2",
                "entity": "facility_001",
                "context": {"reporting_year": 2025, "scope": 1},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_value"] == "1500"
        assert data["source_unit"] == "kg CO2"

    def test_normalize_numeric_value(self, client: TestClient, auth_headers: dict):
        """Should accept numeric values."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={
                "value": 1500.5,
                "unit": "kg CO2",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_value"] == 1500.5

    def test_normalize_missing_value(self, client: TestClient, auth_headers: dict):
        """Should reject request without value."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={"unit": "kg CO2"},
        )

        assert response.status_code == 422  # Validation error

    def test_normalize_missing_unit(self, client: TestClient, auth_headers: dict):
        """Should reject request without unit."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={"value": "100"},
        )

        assert response.status_code == 422  # Validation error


# ==============================================================================
# Batch Normalize Endpoint Tests
# ==============================================================================


class TestBatchNormalizeEndpoint:
    """Tests for batch normalization endpoint."""

    def test_batch_normalize_basic(self, client: TestClient, auth_headers: dict):
        """Should batch normalize multiple values."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "item_001", "value": "100", "unit": "kg CO2"},
                    {"id": "item_002", "value": "50", "unit": "MWh"},
                    {"id": "item_003", "value": "1000", "unit": "gallons"},
                ],
                "batch_mode": "PARTIAL",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 3
        assert data["summary"]["total"] == 3
        assert "api_revision" in data

    def test_batch_normalize_partial_mode(self, client: TestClient, auth_headers: dict):
        """Should continue on failures in PARTIAL mode."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                    {"id": "2", "value": "200", "unit": "kg CO2"},
                ],
                "batch_mode": "PARTIAL",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total"] == 2

    def test_batch_normalize_fail_fast_mode(self, client: TestClient, auth_headers: dict):
        """Should support FAIL_FAST mode."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                ],
                "batch_mode": "FAIL_FAST",
            },
        )

        assert response.status_code == 200

    def test_batch_normalize_threshold_mode(self, client: TestClient, auth_headers: dict):
        """Should support THRESHOLD mode."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                ],
                "batch_mode": "THRESHOLD",
                "threshold": 0.1,
            },
        )

        assert response.status_code == 200

    def test_batch_normalize_empty_items(self, client: TestClient, auth_headers: dict):
        """Should reject empty items list."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [],
                "batch_mode": "PARTIAL",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_batch_normalize_summary(self, client: TestClient, auth_headers: dict):
        """Should include summary statistics."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                    {"id": "2", "value": "200", "unit": "kg CO2"},
                ],
                "batch_mode": "PARTIAL",
            },
        )

        assert response.status_code == 200
        data = response.json()
        summary = data["summary"]
        assert "total" in summary
        assert "success" in summary
        assert "failed" in summary
        assert "needs_review" in summary
        assert "processing_time_ms" in summary


# ==============================================================================
# Jobs Endpoint Tests
# ==============================================================================


class TestJobsEndpoint:
    """Tests for async job endpoints."""

    def test_create_job(self, client: TestClient, auth_headers: dict):
        """Should create async job."""
        response = client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                    {"id": "2", "value": "200", "unit": "kg CO2"},
                ],
                "batch_mode": "PARTIAL",
                "priority": 5,
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "status_url" in data
        assert data["status"] == "PENDING"
        assert "api_revision" in data

    def test_create_job_with_callback(self, client: TestClient, auth_headers: dict):
        """Should accept callback URL."""
        response = client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                ],
                "batch_mode": "PARTIAL",
                "callback_url": "https://example.com/webhook",
            },
        )

        assert response.status_code == 202

    def test_get_job_status(self, client: TestClient, auth_headers: dict):
        """Should get job status."""
        # Create job first
        create_response = client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "items": [
                    {"id": "1", "value": "100", "unit": "kg CO2"},
                ],
                "batch_mode": "PARTIAL",
            },
        )
        job_id = create_response.json()["job_id"]

        # Get status
        response = client.get(
            f"/v1/jobs/{job_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
        assert "api_revision" in data

    def test_get_job_not_found(self, client: TestClient, auth_headers: dict):
        """Should return 404 for unknown job."""
        response = client.get(
            "/v1/jobs/job_nonexistent",
            headers=auth_headers,
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "GLNORM-005"


# ==============================================================================
# Vocabularies Endpoint Tests
# ==============================================================================


class TestVocabulariesEndpoint:
    """Tests for vocabulary endpoint."""

    def test_list_vocabularies(self, client: TestClient, auth_headers: dict):
        """Should list available vocabularies."""
        response = client.get(
            "/v1/vocabularies",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "vocabularies" in data
        assert "total" in data
        assert len(data["vocabularies"]) > 0
        assert "api_revision" in data

    def test_vocabulary_fields(self, client: TestClient, auth_headers: dict):
        """Vocabulary should have required fields."""
        response = client.get(
            "/v1/vocabularies",
            headers=auth_headers,
        )

        assert response.status_code == 200
        vocab = response.json()["vocabularies"][0]
        assert "id" in vocab
        assert "name" in vocab
        assert "version" in vocab
        assert "entry_count" in vocab
        assert "last_updated" in vocab

    def test_filter_vocabularies_by_category(self, client: TestClient, auth_headers: dict):
        """Should filter vocabularies by category."""
        response = client.get(
            "/v1/vocabularies",
            headers=auth_headers,
            params={"category": "emissions"},
        )

        assert response.status_code == 200
        data = response.json()
        # All returned vocabularies should have emissions category
        for vocab in data["vocabularies"]:
            assert "emissions" in vocab.get("categories", [])


# ==============================================================================
# API Revision Tests
# ==============================================================================


class TestAPIRevision:
    """Tests for API revision in responses."""

    def test_normalize_includes_revision(self, client: TestClient, auth_headers: dict):
        """Normalize response should include api_revision."""
        response = client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={"value": "100", "unit": "kg CO2"},
        )

        assert response.status_code == 200
        assert "api_revision" in response.json()

    def test_batch_includes_revision(self, client: TestClient, auth_headers: dict):
        """Batch response should include api_revision."""
        response = client.post(
            "/v1/normalize/batch",
            headers=auth_headers,
            json={
                "items": [{"id": "1", "value": "100", "unit": "kg CO2"}],
                "batch_mode": "PARTIAL",
            },
        )

        assert response.status_code == 200
        assert "api_revision" in response.json()

    def test_error_includes_revision(self, client: TestClient):
        """Error response should include api_revision."""
        response = client.post(
            "/v1/normalize",
            json={"value": "100", "unit": "kg CO2"},
        )

        assert response.status_code == 401
        assert "api_revision" in response.json()


# ==============================================================================
# Rate Limit Headers Tests
# ==============================================================================


class TestRateLimitHeaders:
    """Tests for rate limit headers."""

    @pytest.fixture
    def rate_limited_client(self) -> TestClient:
        """Create client with rate limiting enabled."""
        settings = Settings(
            env="development",
            secret_key="test-secret-key-must-be-32-characters-long",
            rate_limit_enabled=True,
            rate_limit_requests=100,
            rate_limit_window=60,
        )
        app = create_app(settings=settings)
        return TestClient(app)

    def test_rate_limit_headers_present(
        self, rate_limited_client: TestClient, auth_headers: dict
    ):
        """Response should include rate limit headers."""
        response = rate_limited_client.post(
            "/v1/normalize",
            headers=auth_headers,
            json={"value": "100", "unit": "kg CO2"},
        )

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
