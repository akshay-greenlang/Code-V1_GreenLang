"""
Integration tests for the Normalizer API.

These tests validate the API endpoints with actual service instances.
"""

import pytest
from typing import Dict, Any

# Skip if httpx not available
pytest.importorskip("httpx")


class TestAPIIntegration:
    """Integration tests for the Normalizer API."""

    @pytest.fixture
    def api_base_url(self) -> str:
        """Get API base URL from environment or use default."""
        import os
        return os.getenv("NORMALIZER_API_URL", "http://localhost:8000")

    @pytest.fixture
    def api_client(self, api_base_url: str):
        """Create API client."""
        import httpx
        return httpx.Client(base_url=api_base_url, timeout=30.0)

    @pytest.mark.integration
    def test_health_endpoint(self, api_client):
        """Test health endpoint returns healthy status."""
        response = api_client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    @pytest.mark.integration
    def test_normalize_single_value(self, api_client):
        """Test single value normalization."""
        response = api_client.post(
            "/v1/normalize",
            json={
                "value": "1000",
                "unit": "kg",
                "target_unit": "metric_ton",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("canonical_value") == 1.0
        assert data.get("canonical_unit") == "metric_ton"

    @pytest.mark.integration
    def test_normalize_batch(self, api_client):
        """Test batch normalization."""
        response = api_client.post(
            "/v1/normalize/batch",
            json={
                "items": [
                    {"id": "item_001", "value": "1000", "unit": "kg"},
                    {"id": "item_002", "value": "500", "unit": "kWh"},
                ],
                "batch_mode": "PARTIAL",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    @pytest.mark.integration
    def test_vocabularies_endpoint(self, api_client):
        """Test vocabularies listing endpoint."""
        response = api_client.get("/v1/vocabularies")
        assert response.status_code == 200
        data = response.json()
        assert "vocabularies" in data

    @pytest.mark.integration
    def test_error_handling(self, api_client):
        """Test error handling for invalid input."""
        response = api_client.post(
            "/v1/normalize",
            json={
                "value": "invalid",
                "unit": "unknown_unit",
            },
        )
        # Should return error response
        assert response.status_code in [400, 422]
        data = response.json()
        assert "error" in data


class TestAuditIntegration:
    """Integration tests for audit functionality."""

    @pytest.mark.integration
    def test_audit_event_creation(self):
        """Test that normalization creates audit events."""
        # Stub - would test actual audit event creation
        pass

    @pytest.mark.integration
    def test_audit_event_retrieval(self):
        """Test audit event retrieval."""
        # Stub - would test actual audit retrieval
        pass


class TestCacheIntegration:
    """Integration tests for caching functionality."""

    @pytest.mark.integration
    def test_cache_hit(self):
        """Test that repeated requests hit cache."""
        # Stub - would test actual cache behavior
        pass

    @pytest.mark.integration
    def test_cache_invalidation(self):
        """Test cache invalidation on vocabulary update."""
        # Stub - would test actual cache invalidation
        pass
