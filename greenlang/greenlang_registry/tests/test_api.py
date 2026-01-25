"""
API Endpoint Tests for GreenLang Agent Registry.

This module tests the 4 core API endpoints:
1. POST /api/v1/registry/agents - Publish agent
2. GET /api/v1/registry/agents - List agents
3. GET /api/v1/registry/agents/{id} - Get agent details
4. POST /api/v1/registry/agents/{id}/promote - Promote agent state
"""

import pytest
from httpx import AsyncClient

from greenlang_registry.db.models import Agent, AgentVersion


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test basic health check returns 200."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_liveness_check(self, client: AsyncClient):
        """Test liveness probe returns alive status."""
        response = await client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"


# =============================================================================
# Publish Agent Tests (POST /api/v1/registry/agents)
# =============================================================================

class TestPublishAgent:
    """Tests for agent publishing endpoint."""

    @pytest.mark.asyncio
    async def test_publish_new_agent_success(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test publishing a new agent succeeds."""
        response = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )

        assert response.status_code == 201

        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == publish_request_data["agent_id"]
        assert data["version"] == publish_request_data["version"]
        assert data["lifecycle_state"] == "draft"
        assert "version_id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_publish_duplicate_version_fails(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test publishing duplicate version returns 409."""
        # First publish
        response1 = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )
        assert response1.status_code == 201

        # Second publish with same version
        response2 = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )
        assert response2.status_code == 409

        data = response2.json()
        assert "VersionExists" in str(data)

    @pytest.mark.asyncio
    async def test_publish_new_version_of_existing_agent(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test publishing new version of existing agent succeeds."""
        # First version
        response1 = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )
        assert response1.status_code == 201

        # Second version
        publish_request_data["version"] = "1.1.0"
        response2 = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )
        assert response2.status_code == 201

        data = response2.json()
        assert data["version"] == "1.1.0"

    @pytest.mark.asyncio
    async def test_publish_invalid_agent_id_fails(self, client: AsyncClient):
        """Test publishing with invalid agent_id fails validation."""
        invalid_request = {
            "agent_id": "Invalid Agent ID!",  # Invalid characters
            "name": "Test",
            "version": "1.0.0",
        }
        response = await client.post(
            "/api/v1/registry/agents",
            json=invalid_request
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_publish_invalid_version_fails(self, client: AsyncClient):
        """Test publishing with invalid version fails validation."""
        invalid_request = {
            "agent_id": "gl-test-agent",
            "name": "Test",
            "version": "not-a-version",  # Invalid semver
        }
        response = await client.post(
            "/api/v1/registry/agents",
            json=invalid_request
        )
        assert response.status_code == 422


# =============================================================================
# List Agents Tests (GET /api/v1/registry/agents)
# =============================================================================

class TestListAgents:
    """Tests for listing agents endpoint."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, client: AsyncClient):
        """Test listing agents when none exist."""
        response = await client.get("/api/v1/registry/agents")
        assert response.status_code == 200

        data = response.json()
        assert data["agents"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    @pytest.mark.asyncio
    async def test_list_agents_with_data(
        self,
        client: AsyncClient,
        sample_agent: Agent
    ):
        """Test listing agents returns data."""
        response = await client.get("/api/v1/registry/agents")
        assert response.status_code == 200

        data = response.json()
        assert len(data["agents"]) >= 1
        assert data["total"] >= 1

        # Verify agent data
        agent = data["agents"][0]
        assert agent["agent_id"] == sample_agent.agent_id
        assert agent["name"] == sample_agent.name

    @pytest.mark.asyncio
    async def test_list_agents_pagination(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test pagination works correctly."""
        # Create multiple agents
        for i in range(5):
            publish_request_data["agent_id"] = f"gl-test-agent-{i}"
            await client.post("/api/v1/registry/agents", json=publish_request_data)

        # Get first page
        response = await client.get(
            "/api/v1/registry/agents",
            params={"page": 1, "page_size": 2}
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["agents"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["has_next"] is True

    @pytest.mark.asyncio
    async def test_list_agents_filter_by_domain(
        self,
        client: AsyncClient,
        sample_agent: Agent
    ):
        """Test filtering agents by domain."""
        response = await client.get(
            "/api/v1/registry/agents",
            params={"domain": sample_agent.domain}
        )
        assert response.status_code == 200

        data = response.json()
        assert all(a["domain"] == sample_agent.domain for a in data["agents"])

    @pytest.mark.asyncio
    async def test_list_agents_search(
        self,
        client: AsyncClient,
        sample_agent: Agent
    ):
        """Test searching agents by name/description."""
        response = await client.get(
            "/api/v1/registry/agents",
            params={"search": "Test"}
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["agents"]) >= 1


# =============================================================================
# Get Agent Tests (GET /api/v1/registry/agents/{id})
# =============================================================================

class TestGetAgent:
    """Tests for getting agent details endpoint."""

    @pytest.mark.asyncio
    async def test_get_agent_success(
        self,
        client: AsyncClient,
        sample_agent_with_versions: tuple
    ):
        """Test getting agent details succeeds."""
        agent, versions = sample_agent_with_versions

        response = await client.get(f"/api/v1/registry/agents/{agent.agent_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["agent"]["agent_id"] == agent.agent_id
        assert data["agent"]["name"] == agent.name
        assert data["version_count"] == len(versions)
        assert len(data["versions"]) == len(versions)

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, client: AsyncClient):
        """Test getting non-existent agent returns 404."""
        response = await client.get("/api/v1/registry/agents/nonexistent-agent")
        assert response.status_code == 404

        data = response.json()
        assert "NotFound" in str(data)

    @pytest.mark.asyncio
    async def test_get_agent_includes_latest_version(
        self,
        client: AsyncClient,
        sample_agent_with_versions: tuple
    ):
        """Test response includes latest non-deprecated version."""
        agent, versions = sample_agent_with_versions

        response = await client.get(f"/api/v1/registry/agents/{agent.agent_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["latest_version"] is not None
        # Latest non-deprecated version should be 2.1.0-alpha (draft)
        assert data["latest_version"]["lifecycle_state"] != "deprecated"


# =============================================================================
# Promote Agent Tests (POST /api/v1/registry/agents/{id}/promote)
# =============================================================================

class TestPromoteAgent:
    """Tests for agent promotion endpoint."""

    @pytest.mark.asyncio
    async def test_promote_draft_to_experimental(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test promoting from draft to experimental succeeds."""
        # Create a draft agent
        response = await client.post(
            "/api/v1/registry/agents",
            json=publish_request_data
        )
        assert response.status_code == 201

        # Promote to experimental
        promote_response = await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={
                "target_state": "experimental",
                "reason": "Ready for testing",
                "promoted_by": "test-user",
            }
        )
        assert promote_response.status_code == 200

        data = promote_response.json()
        assert data["success"] is True
        assert data["from_state"] == "draft"
        assert data["to_state"] == "experimental"

    @pytest.mark.asyncio
    async def test_promote_experimental_to_certified(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test promoting from experimental to certified succeeds."""
        # Create agent
        await client.post("/api/v1/registry/agents", json=publish_request_data)

        # Promote to experimental
        await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={"target_state": "experimental"}
        )

        # Promote to certified
        response = await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={"target_state": "certified", "reason": "All tests passed"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["to_state"] == "certified"

    @pytest.mark.asyncio
    async def test_promote_invalid_transition_fails(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test invalid state transition returns 400."""
        # Create a draft agent
        await client.post("/api/v1/registry/agents", json=publish_request_data)

        # Try to promote directly to certified (invalid)
        response = await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={"target_state": "certified"}
        )
        assert response.status_code == 400

        data = response.json()
        assert "InvalidTransition" in str(data)

    @pytest.mark.asyncio
    async def test_promote_nonexistent_agent_fails(self, client: AsyncClient):
        """Test promoting non-existent agent returns 404."""
        response = await client.post(
            "/api/v1/registry/agents/nonexistent-agent/promote",
            json={"target_state": "experimental"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_promote_to_deprecated(
        self,
        client: AsyncClient,
        publish_request_data: dict
    ):
        """Test deprecating an experimental agent succeeds."""
        # Create agent
        await client.post("/api/v1/registry/agents", json=publish_request_data)

        # Promote to experimental
        await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={"target_state": "experimental"}
        )

        # Deprecate
        response = await client.post(
            f"/api/v1/registry/agents/{publish_request_data['agent_id']}/promote",
            json={"target_state": "deprecated", "reason": "Replaced by v2"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["to_state"] == "deprecated"


# =============================================================================
# Version Endpoint Tests
# =============================================================================

class TestGetVersion:
    """Tests for getting specific version endpoint."""

    @pytest.mark.asyncio
    async def test_get_specific_version(
        self,
        client: AsyncClient,
        sample_agent_with_versions: tuple
    ):
        """Test getting a specific version succeeds."""
        agent, versions = sample_agent_with_versions

        response = await client.get(
            f"/api/v1/registry/agents/{agent.agent_id}/versions/1.0.0"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "1.0.0"
        assert data["agent_id"] == agent.agent_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_version(
        self,
        client: AsyncClient,
        sample_agent: Agent
    ):
        """Test getting non-existent version returns 404."""
        response = await client.get(
            f"/api/v1/registry/agents/{sample_agent.agent_id}/versions/99.99.99"
        )
        assert response.status_code == 404
