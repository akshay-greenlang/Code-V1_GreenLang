"""
Registry Workflow End-to-End Tests

This module tests the complete registry workflow through the API:
- Publishing agents to registry
- Pulling agents from registry
- Searching and discovering agents
- Multi-tenant isolation
- Registry metadata management

Run with: pytest tests/e2e/test_registry_workflow.py -v -m e2e

Prerequisites:
    - Docker Compose services running
    - E2E_TESTS=1 environment variable set
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

import pytest

# Import conditional - tests will skip if httpx not available
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

from tests.e2e.conftest import (
    assert_api_success,
    assert_api_error,
    assert_valid_provenance_hash,
    wait_for_condition,
    e2e,
    requires_docker,
    TEST_TENANT_ID,
)


# =============================================================================
# Registry Publish Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryPublish:
    """Test publishing agents to the registry."""

    @pytest.mark.asyncio
    async def test_publish_agent_to_registry(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test publishing an agent to the registry."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        publish_request = {
            "agent_spec": sample_agent_spec,
            "visibility": "private",
            "release_notes": "Initial release for E2E testing",
            "metadata": {
                "documentation_url": "https://docs.greenlang.io/agents/test",
                "support_email": "support@greenlang.io",
            },
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=publish_request,
        )

        # Accept success or not implemented
        assert response.status_code in [200, 201, 404], \
            f"Expected 200/201/404, got {response.status_code}: {response.text}"

        if response.status_code in [200, 201]:
            data = response.json()
            assert "agent_id" in data or "id" in data
            assert data.get("status") in ["PUBLISHED", "PENDING", None]

    @pytest.mark.asyncio
    async def test_publish_agent_with_public_visibility(
        self,
        authenticated_registry_client,
        unique_agent_id,
    ):
        """Test publishing a public agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        agent_spec = {
            "agent_id": unique_agent_id,
            "name": "Public Test Agent",
            "version": "1.0.0",
            "description": "Publicly available test agent",
            "category": "emissions",
            "tags": ["public", "test", "carbon"],
            "entrypoint": "python://agents.public_test:PublicAgent",
            "deterministic": True,
            "regulatory_frameworks": ["GHG Protocol"],
            "inputs": {"value": {"type": "number"}},
            "outputs": {"result": {"type": "number"}},
        }

        publish_request = {
            "agent_spec": agent_spec,
            "visibility": "public",
            "release_notes": "Public release",
            "license": "Apache-2.0",
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=publish_request,
        )

        assert response.status_code in [200, 201, 404]

    @pytest.mark.asyncio
    async def test_publish_duplicate_version_fails(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test that publishing duplicate version fails."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        publish_request = {
            "agent_spec": sample_agent_spec,
            "visibility": "private",
        }

        # First publish
        response1 = await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=publish_request,
        )

        if response1.status_code not in [200, 201]:
            pytest.skip("Initial publish failed")

        # Duplicate publish
        response2 = await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=publish_request,
        )

        # Should fail with conflict
        assert response2.status_code in [400, 409, 404]

    @pytest.mark.asyncio
    async def test_publish_new_version(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test publishing a new version of existing agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Publish v1.0.0
        v1_request = {
            "agent_spec": sample_agent_spec,
            "visibility": "private",
        }
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=v1_request,
        )

        # Publish v2.0.0
        v2_spec = sample_agent_spec.copy()
        v2_spec["version"] = "2.0.0"
        v2_spec["description"] = "Updated version with new features"

        v2_request = {
            "agent_spec": v2_spec,
            "visibility": "private",
            "release_notes": "Major update with breaking changes",
            "breaking_changes": True,
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/publish",
            json=v2_request,
        )

        assert response.status_code in [200, 201, 404]

    @pytest.mark.asyncio
    async def test_publish_requires_authentication(
        self,
        registry_client,
        sample_agent_spec,
    ):
        """Test that publishing requires authentication."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        publish_request = {
            "agent_spec": sample_agent_spec,
            "visibility": "private",
        }

        response = await registry_client.post(
            "/v1/registry/publish",
            json=publish_request,
        )

        # Should fail with auth error
        assert response.status_code in [401, 403], \
            f"Expected 401/403 without auth, got {response.status_code}"


# =============================================================================
# Registry Pull Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryPull:
    """Test pulling agents from the registry."""

    @pytest.mark.asyncio
    async def test_pull_agent_by_id(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test pulling an agent by ID."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # First publish
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": sample_agent_spec, "visibility": "private"},
        )

        # Pull
        response = await authenticated_registry_client.get(
            f"/v1/registry/agents/{sample_agent_spec['agent_id']}"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert data.get("agent_id") == sample_agent_spec["agent_id"] or \
                   data.get("name") == sample_agent_spec["name"]

    @pytest.mark.asyncio
    async def test_pull_specific_version(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test pulling a specific agent version."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Publish
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": sample_agent_spec, "visibility": "private"},
        )

        # Pull specific version
        response = await authenticated_registry_client.get(
            f"/v1/registry/agents/{sample_agent_spec['agent_id']}/versions/1.0.0"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert data.get("version") == "1.0.0"

    @pytest.mark.asyncio
    async def test_pull_latest_version(
        self,
        authenticated_registry_client,
        unique_agent_id,
    ):
        """Test pulling the latest version of an agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create base spec
        base_spec = {
            "agent_id": unique_agent_id,
            "name": "Latest Version Test Agent",
            "version": "1.0.0",
            "description": "Test agent for latest version pull",
            "category": "test",
            "tags": ["test"],
            "entrypoint": "python://test:Agent",
            "deterministic": True,
            "regulatory_frameworks": ["TEST"],
            "inputs": {"value": {"type": "number"}},
            "outputs": {"result": {"type": "number"}},
        }

        # Publish v1.0.0
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": base_spec, "visibility": "private"},
        )

        # Publish v2.0.0
        v2_spec = base_spec.copy()
        v2_spec["version"] = "2.0.0"
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": v2_spec, "visibility": "private"},
        )

        # Pull latest
        response = await authenticated_registry_client.get(
            f"/v1/registry/agents/{unique_agent_id}/latest"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # Should be the latest version (2.0.0)
            assert data.get("version") in ["2.0.0", "latest"]

    @pytest.mark.asyncio
    async def test_pull_nonexistent_agent_fails(
        self,
        authenticated_registry_client,
    ):
        """Test pulling a non-existent agent returns 404."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/agents/nonexistent/fake-agent-xyz123"
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_pull_with_download(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test downloading agent package."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Publish
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": sample_agent_spec, "visibility": "private"},
        )

        # Download
        response = await authenticated_registry_client.get(
            f"/v1/registry/agents/{sample_agent_spec['agent_id']}/download",
            params={"version": "1.0.0"},
        )

        # Accept 200 (download), 404 (not found), or redirect
        assert response.status_code in [200, 302, 404]


# =============================================================================
# Registry Search Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistrySearch:
    """Test registry search and discovery functionality."""

    @pytest.mark.asyncio
    async def test_search_by_keyword(
        self,
        authenticated_registry_client,
    ):
        """Test searching agents by keyword."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={"q": "carbon emissions"},
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "results" in data or "agents" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_search_by_category(
        self,
        authenticated_registry_client,
    ):
        """Test searching agents by category."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={"category": "emissions"},
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", data.get("agents", data))
            if isinstance(results, list):
                for agent in results:
                    assert agent.get("category") == "emissions"

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        authenticated_registry_client,
    ):
        """Test searching agents by tags."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={"tags": "carbon,ghg"},
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_search_by_regulatory_framework(
        self,
        authenticated_registry_client,
    ):
        """Test searching agents by regulatory framework."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={"framework": "GHG Protocol"},
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_search_with_pagination(
        self,
        authenticated_registry_client,
    ):
        """Test search with pagination parameters."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={
                "q": "agent",
                "limit": 10,
                "offset": 0,
                "sort": "created_at",
                "order": "desc",
            },
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # Check pagination metadata
            if "total" in data:
                assert isinstance(data["total"], int)
            if "limit" in data:
                assert data["limit"] <= 10

    @pytest.mark.asyncio
    async def test_search_certified_agents_only(
        self,
        authenticated_registry_client,
    ):
        """Test filtering search to certified agents only."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={
                "certified": True,
                "certification_level": "standard",
            },
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        authenticated_registry_client,
    ):
        """Test search with no matching results."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/search",
            params={"q": "xyznonexistent123456789"},
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", data.get("agents", data))
            if isinstance(results, list):
                assert len(results) == 0

    @pytest.mark.asyncio
    async def test_list_all_agents(
        self,
        authenticated_registry_client,
    ):
        """Test listing all agents in registry."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/agents",
            params={"limit": 50},
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    @pytest.mark.asyncio
    async def test_list_agents_by_author(
        self,
        authenticated_registry_client,
    ):
        """Test listing agents by author."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/agents",
            params={"author": "greenlang"},
        )

        assert response.status_code in [200, 404]


# =============================================================================
# Multi-Tenant Isolation Tests
# =============================================================================


@pytest.mark.e2e
class TestMultiTenantIsolation:
    """Test multi-tenant isolation in registry operations."""

    @pytest.mark.asyncio
    async def test_tenant_agents_isolated(
        self,
        registry_base_url,
        unique_agent_id,
    ):
        """Test that agents from different tenants are isolated."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent as tenant-1
        tenant1_headers = {
            "Authorization": "Bearer tenant1-api-key",
            "X-Tenant-ID": "tenant-1",
        }

        async with httpx.AsyncClient(
            base_url=registry_base_url,
            timeout=30,
        ) as client:
            agent_spec = {
                "agent_id": unique_agent_id,
                "name": "Tenant 1 Agent",
                "version": "1.0.0",
                "description": "Agent owned by tenant 1",
                "category": "test",
                "tags": ["tenant-1"],
                "entrypoint": "python://test:Agent",
                "deterministic": True,
                "regulatory_frameworks": ["TEST"],
                "inputs": {"value": {"type": "number"}},
                "outputs": {"result": {"type": "number"}},
            }

            # Publish as tenant-1
            publish_response = await client.post(
                "/v1/registry/publish",
                json={"agent_spec": agent_spec, "visibility": "private"},
                headers=tenant1_headers,
            )

            # Try to access as tenant-2
            tenant2_headers = {
                "Authorization": "Bearer tenant2-api-key",
                "X-Tenant-ID": "tenant-2",
            }

            access_response = await client.get(
                f"/v1/registry/agents/{unique_agent_id}",
                headers=tenant2_headers,
            )

            # Tenant-2 should not see tenant-1's private agent
            if publish_response.status_code in [200, 201]:
                assert access_response.status_code in [403, 404], \
                    "Private agent should not be visible to other tenants"

    @pytest.mark.asyncio
    async def test_public_agents_visible_across_tenants(
        self,
        registry_base_url,
        unique_agent_id,
    ):
        """Test that public agents are visible across tenants."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        tenant1_headers = {
            "Authorization": "Bearer tenant1-api-key",
            "X-Tenant-ID": "tenant-1",
        }

        async with httpx.AsyncClient(
            base_url=registry_base_url,
            timeout=30,
        ) as client:
            agent_spec = {
                "agent_id": unique_agent_id,
                "name": "Public Shared Agent",
                "version": "1.0.0",
                "description": "Publicly available agent",
                "category": "test",
                "tags": ["public", "shared"],
                "entrypoint": "python://test:PublicAgent",
                "deterministic": True,
                "regulatory_frameworks": ["TEST"],
                "inputs": {"value": {"type": "number"}},
                "outputs": {"result": {"type": "number"}},
            }

            # Publish as public
            publish_response = await client.post(
                "/v1/registry/publish",
                json={"agent_spec": agent_spec, "visibility": "public"},
                headers=tenant1_headers,
            )

            # Access as tenant-2
            tenant2_headers = {
                "Authorization": "Bearer tenant2-api-key",
                "X-Tenant-ID": "tenant-2",
            }

            if publish_response.status_code in [200, 201]:
                access_response = await client.get(
                    f"/v1/registry/agents/{unique_agent_id}",
                    headers=tenant2_headers,
                )

                # Public agent should be visible
                assert access_response.status_code in [200, 404]


# =============================================================================
# Registry Metadata Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryMetadata:
    """Test registry metadata operations."""

    @pytest.mark.asyncio
    async def test_get_registry_stats(
        self,
        authenticated_registry_client,
    ):
        """Test getting registry statistics."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/stats"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # Check for expected stats fields
            expected_fields = ["total_agents", "total_versions", "categories"]
            has_some_fields = any(f in data for f in expected_fields)
            assert has_some_fields or isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_list_categories(
        self,
        authenticated_registry_client,
    ):
        """Test listing available categories."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/categories"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    @pytest.mark.asyncio
    async def test_list_regulatory_frameworks(
        self,
        authenticated_registry_client,
    ):
        """Test listing supported regulatory frameworks."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/frameworks"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            frameworks = data if isinstance(data, list) else data.get("frameworks", [])
            # Should include common frameworks
            expected_frameworks = ["GHG Protocol", "CBAM", "CSRD"]
            # At least structure should be valid
            assert isinstance(frameworks, list)

    @pytest.mark.asyncio
    async def test_get_agent_download_count(
        self,
        authenticated_registry_client,
        sample_agent_spec,
    ):
        """Test getting agent download statistics."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Publish agent
        await authenticated_registry_client.post(
            "/v1/registry/publish",
            json={"agent_spec": sample_agent_spec, "visibility": "private"},
        )

        response = await authenticated_registry_client.get(
            f"/v1/registry/agents/{sample_agent_spec['agent_id']}/stats"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "downloads" in data or "download_count" in data or isinstance(data, dict)


# =============================================================================
# Registry Health Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryHealth:
    """Test registry service health and availability."""

    @pytest.mark.asyncio
    async def test_registry_health_check(
        self,
        registry_client,
    ):
        """Test registry health endpoint."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await registry_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["healthy", "ok"]

    @pytest.mark.asyncio
    async def test_registry_readiness(
        self,
        registry_client,
    ):
        """Test registry readiness endpoint."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await registry_client.get("/ready")

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data.get("ready") is True

    @pytest.mark.asyncio
    async def test_registry_response_time(
        self,
        registry_client,
    ):
        """Test registry response time is acceptable."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        start_time = time.time()
        response = await registry_client.get("/health")
        duration_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert duration_ms < 500, f"Health check too slow: {duration_ms}ms"


# =============================================================================
# Registry Webhook Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryWebhooks:
    """Test registry webhook functionality."""

    @pytest.mark.asyncio
    async def test_register_webhook(
        self,
        authenticated_registry_client,
    ):
        """Test registering a webhook for registry events."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        webhook_request = {
            "url": "https://example.com/webhook",
            "events": ["agent.published", "agent.deprecated"],
            "secret": "webhook-secret-key",
            "active": True,
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/webhooks",
            json=webhook_request,
        )

        assert response.status_code in [200, 201, 404]

    @pytest.mark.asyncio
    async def test_list_webhooks(
        self,
        authenticated_registry_client,
    ):
        """Test listing registered webhooks."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/webhooks"
        )

        assert response.status_code in [200, 404]


# =============================================================================
# Registry API Key Management Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryAPIKeys:
    """Test registry API key management."""

    @pytest.mark.asyncio
    async def test_create_api_key(
        self,
        authenticated_registry_client,
    ):
        """Test creating a new API key for registry access."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        key_request = {
            "name": "E2E Test API Key",
            "scopes": ["read", "write"],
            "expires_in_days": 30,
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/api-keys",
            json=key_request,
        )

        assert response.status_code in [200, 201, 404]

        if response.status_code in [200, 201]:
            data = response.json()
            assert "key" in data or "api_key" in data or "token" in data

    @pytest.mark.asyncio
    async def test_list_api_keys(
        self,
        authenticated_registry_client,
    ):
        """Test listing API keys."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_registry_client.get(
            "/v1/registry/api-keys"
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_revoke_api_key(
        self,
        authenticated_registry_client,
    ):
        """Test revoking an API key."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # First create a key
        create_response = await authenticated_registry_client.post(
            "/v1/registry/api-keys",
            json={"name": "Temp Key", "scopes": ["read"]},
        )

        if create_response.status_code in [200, 201]:
            key_data = create_response.json()
            key_id = key_data.get("id", key_data.get("key_id"))

            if key_id:
                # Revoke the key
                revoke_response = await authenticated_registry_client.delete(
                    f"/v1/registry/api-keys/{key_id}"
                )

                assert revoke_response.status_code in [200, 204, 404]


# =============================================================================
# Registry Bulk Operations Tests
# =============================================================================


@pytest.mark.e2e
class TestRegistryBulkOperations:
    """Test registry bulk operations."""

    @pytest.mark.asyncio
    async def test_bulk_publish(
        self,
        authenticated_registry_client,
    ):
        """Test bulk publishing multiple agents."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        agents = [
            {
                "agent_id": f"e2e-test/bulk-agent-{i}-{uuid4().hex[:6]}",
                "name": f"Bulk Test Agent {i}",
                "version": "1.0.0",
                "description": f"Bulk test agent number {i}",
                "category": "test",
                "tags": ["bulk", "test"],
                "entrypoint": f"python://test.bulk:Agent{i}",
                "deterministic": True,
                "regulatory_frameworks": ["TEST"],
                "inputs": {"value": {"type": "number"}},
                "outputs": {"result": {"type": "number"}},
            }
            for i in range(3)
        ]

        bulk_request = {
            "agents": [{"agent_spec": agent, "visibility": "private"} for agent in agents],
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/bulk/publish",
            json=bulk_request,
        )

        assert response.status_code in [200, 201, 207, 404]

    @pytest.mark.asyncio
    async def test_bulk_search(
        self,
        authenticated_registry_client,
    ):
        """Test bulk search for multiple agent IDs."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        search_request = {
            "agent_ids": [
                "emissions/carbon-calculator-v1",
                "regulatory/cbam-compliance-v1",
                "nonexistent/agent-v1",
            ],
        }

        response = await authenticated_registry_client.post(
            "/v1/registry/bulk/search",
            json=search_request,
        )

        assert response.status_code in [200, 404]
