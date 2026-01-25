"""
Test Suite for Agent Registry API Endpoints

This module contains integration tests for the FastAPI endpoints:
- Agent CRUD endpoints
- Version management endpoints
- Search and filter endpoints
- Publishing workflow endpoints

Run with: pytest backend/registry/tests/test_api.py -v
"""

import pytest
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.registry.api import router
from backend.registry.service import AgentRegistryService


# Create test app
def create_test_app() -> FastAPI:
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def app():
    """Create test app fixture."""
    return create_test_app()


@pytest.fixture
def client(app):
    """Create test client fixture."""
    return TestClient(app)


class TestAgentEndpoints:
    """Tests for agent CRUD endpoints."""

    def test_create_agent(self, client):
        """Test POST /agents endpoint."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "test-agent",
                "version": "1.0.0",
                "description": "Test agent",
                "category": "test",
                "author": "test-user",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-agent"
        assert data["version"] == "1.0.0"
        assert data["status"] == "draft"

    def test_create_agent_with_full_metadata(self, client):
        """Test creating agent with all fields."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "full-metadata-agent",
                "version": "1.0.0",
                "description": "Agent with all metadata",
                "category": "emissions",
                "author": "greenlang",
                "tags": ["carbon", "ghg"],
                "regulatory_frameworks": ["CBAM", "CSRD"],
                "documentation_url": "https://docs.example.com",
                "repository_url": "https://github.com/example/repo",
                "license": "MIT",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["tags"] == ["carbon", "ghg"]
        assert data["regulatory_frameworks"] == ["CBAM", "CSRD"]
        assert data["license"] == "MIT"

    def test_create_agent_invalid_name(self, client):
        """Test creating agent with invalid name."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "123invalid",
                "version": "1.0.0",
                "category": "test",
                "author": "user",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_list_agents(self, client):
        """Test GET /agents endpoint."""
        response = client.get("/registry/agents")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "meta" in data
        assert isinstance(data["data"], list)

    def test_list_agents_with_filters(self, client):
        """Test listing agents with filters."""
        response = client.get(
            "/registry/agents",
            params={
                "category": "emissions",
                "status": "published",
                "limit": 10,
            },
        )

        assert response.status_code == 200

    def test_list_agents_pagination(self, client):
        """Test pagination parameters."""
        response = client.get(
            "/registry/agents",
            params={
                "limit": 5,
                "offset": 10,
                "sort_by": "downloads",
                "sort_order": "desc",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["limit"] == 5
        assert data["meta"]["offset"] == 10

    def test_get_agent_not_found(self, client):
        """Test getting non-existent agent."""
        response = client.get(f"/registry/agents/{uuid4()}")

        assert response.status_code == 404

    def test_update_agent_not_found(self, client):
        """Test updating non-existent agent."""
        response = client.put(
            f"/registry/agents/{uuid4()}",
            json={"description": "Updated"},
        )

        assert response.status_code == 404

    def test_update_agent_no_changes(self, client):
        """Test update with empty body."""
        # First create an agent
        create_response = client.post(
            "/registry/agents",
            json={
                "name": "update-test",
                "version": "1.0.0",
                "category": "test",
                "author": "user",
            },
        )

        if create_response.status_code == 201:
            agent_id = create_response.json()["id"]

            response = client.put(
                f"/registry/agents/{agent_id}",
                json={},
            )

            assert response.status_code == 400

    def test_delete_agent_not_found(self, client):
        """Test deleting non-existent agent."""
        response = client.delete(f"/registry/agents/{uuid4()}")

        # Should still return 204 (idempotent)
        assert response.status_code in (204, 404)


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_agents(self, client):
        """Test GET /agents/search endpoint."""
        response = client.get(
            "/registry/agents/search",
            params={"q": "carbon"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "meta" in data

    def test_search_with_filters(self, client):
        """Test search with additional filters."""
        response = client.get(
            "/registry/agents/search",
            params={
                "q": "emissions",
                "category": "regulatory",
                "status": "published",
            },
        )

        assert response.status_code == 200

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.get(
            "/registry/agents/search",
            params={"q": ""},
        )

        # Should return validation error
        assert response.status_code == 422

    def test_search_pagination(self, client):
        """Test search pagination."""
        response = client.get(
            "/registry/agents/search",
            params={
                "q": "test",
                "limit": 5,
                "offset": 0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["limit"] == 5


class TestVersionEndpoints:
    """Tests for version management endpoints."""

    def test_list_versions_not_found(self, client):
        """Test listing versions for non-existent agent."""
        response = client.get(f"/registry/agents/{uuid4()}/versions")

        # Returns empty list, not 404
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_version_not_found(self, client):
        """Test getting non-existent version."""
        response = client.get(f"/registry/agents/{uuid4()}/versions/1.0.0")

        assert response.status_code == 404


class TestPublishEndpoints:
    """Tests for publishing workflow endpoints."""

    def test_publish_not_found(self, client):
        """Test publishing non-existent agent."""
        response = client.post(
            f"/registry/agents/{uuid4()}/publish",
            json={"version": "1.0.0"},
        )

        assert response.status_code in (400, 404)

    def test_deprecate_not_found(self, client):
        """Test deprecating non-existent agent."""
        response = client.post(f"/registry/agents/{uuid4()}/deprecate")

        assert response.status_code == 404


class TestDownloadEndpoint:
    """Tests for download endpoint."""

    def test_download_not_found(self, client):
        """Test downloading non-existent agent."""
        response = client.get(f"/registry/agents/{uuid4()}/download")

        assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for statistics endpoint."""

    def test_get_stats(self, client):
        """Test GET /stats endpoint."""
        response = client.get("/registry/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_agents" in data
        assert "total_versions" in data
        assert "total_downloads" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_uuid(self, client):
        """Test with invalid UUID format."""
        response = client.get("/registry/agents/not-a-uuid")

        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test with invalid JSON body."""
        response = client.post(
            "/registry/agents",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test create with missing required fields."""
        response = client.post(
            "/registry/agents",
            json={"name": "test"},  # Missing category and author
        )

        assert response.status_code == 422


class TestRequestValidation:
    """Tests for request validation."""

    def test_name_too_short(self, client):
        """Test name validation - too short."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "ab",
                "version": "1.0.0",
                "category": "test",
                "author": "user",
            },
        )

        assert response.status_code == 422

    def test_name_too_long(self, client):
        """Test name validation - too long."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "a" * 101,
                "version": "1.0.0",
                "category": "test",
                "author": "user",
            },
        )

        assert response.status_code == 422

    def test_invalid_version_format(self, client):
        """Test version format validation."""
        response = client.post(
            "/registry/agents",
            json={
                "name": "test-agent",
                "version": "invalid",
                "category": "test",
                "author": "user",
            },
        )

        assert response.status_code == 422

    def test_limit_out_of_range(self, client):
        """Test limit parameter validation."""
        response = client.get(
            "/registry/agents",
            params={"limit": 1000},
        )

        assert response.status_code == 422


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
