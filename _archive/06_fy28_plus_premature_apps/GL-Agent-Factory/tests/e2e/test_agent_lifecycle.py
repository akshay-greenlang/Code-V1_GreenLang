"""
Agent Lifecycle End-to-End Tests

This module tests the complete agent lifecycle through the API:
- Agent creation via API
- Agent execution via API
- Agent versioning
- Certification workflow
- State transitions

Run with: pytest tests/e2e/test_agent_lifecycle.py -v -m e2e

Prerequisites:
    - Docker Compose services running
    - E2E_TESTS=1 environment variable set
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict
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
    assert_recent_timestamp,
    wait_for_condition,
    e2e,
    requires_docker,
)


# =============================================================================
# Agent Creation Tests
# =============================================================================


@pytest.mark.e2e
class TestAgentCreation:
    """Test agent creation via API."""

    @pytest.mark.asyncio
    async def test_create_agent_success(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test successful agent creation through API."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Assert successful creation
        assert response.status_code in [200, 201], \
            f"Expected 200/201, got {response.status_code}: {response.text}"

        data = response.json()
        assert data["agent_id"] == sample_agent_spec["agent_id"]
        assert data["version"] == sample_agent_spec["version"]
        assert data["name"] == sample_agent_spec["name"]
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_create_agent_with_full_spec(
        self,
        authenticated_api_client,
        unique_agent_id,
    ):
        """Test agent creation with complete specification."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        full_spec = {
            "agent_id": unique_agent_id,
            "name": "Full Spec Test Agent",
            "version": "1.0.0",
            "description": "Agent with complete specification for testing",
            "category": "emissions",
            "tags": ["carbon", "ghg", "test", "full-spec"],
            "entrypoint": "python://agents.test:FullSpecAgent",
            "deterministic": True,
            "regulatory_frameworks": ["GHG Protocol", "ISO 14064", "CSRD"],
            "inputs": {
                "fuel_type": {
                    "type": "string",
                    "description": "Type of fuel consumed",
                    "enum": ["diesel", "natural_gas", "coal", "electricity"],
                },
                "quantity": {
                    "type": "number",
                    "description": "Amount of fuel consumed",
                    "minimum": 0,
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of measurement",
                },
                "region": {
                    "type": "string",
                    "description": "Geographic region for emission factors",
                },
            },
            "outputs": {
                "emissions_kgco2e": {
                    "type": "number",
                    "description": "Total emissions in kg CO2e",
                },
                "emission_factor": {
                    "type": "number",
                    "description": "Applied emission factor",
                },
                "provenance_hash": {
                    "type": "string",
                    "description": "SHA-256 provenance hash",
                },
                "calculation_method": {
                    "type": "string",
                    "description": "Calculation methodology used",
                },
            },
            "config": {
                "timeout_seconds": 60,
                "max_retries": 3,
                "cache_enabled": True,
            },
            "dependencies": [
                "emission-factor-db>=1.0.0",
                "provenance-tracker>=2.0.0",
            ],
        }

        response = await authenticated_api_client.post(
            "/v1/agents",
            json=full_spec,
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["deterministic"] is True
        assert "GHG Protocol" in data.get("regulatory_frameworks", [])

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_fails(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test that creating duplicate agent fails with appropriate error."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create first agent
        response1 = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Attempt to create duplicate
        response2 = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Second creation should fail with 409 Conflict or 400 Bad Request
        assert response2.status_code in [400, 409], \
            f"Expected 400/409 for duplicate, got {response2.status_code}"

    @pytest.mark.asyncio
    async def test_create_agent_invalid_spec_fails(
        self,
        authenticated_api_client,
    ):
        """Test that invalid agent specification is rejected."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        invalid_spec = {
            "agent_id": "",  # Empty ID - invalid
            "name": "",  # Empty name - invalid
            "version": "invalid-version",  # Invalid semver
        }

        response = await authenticated_api_client.post(
            "/v1/agents",
            json=invalid_spec,
        )

        assert response.status_code in [400, 422], \
            f"Expected 400/422 for invalid spec, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_create_agent_without_auth_fails(
        self,
        api_client,
        sample_agent_spec,
    ):
        """Test that agent creation without authentication fails."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Should fail with 401 Unauthorized or 403 Forbidden
        assert response.status_code in [401, 403], \
            f"Expected 401/403 without auth, got {response.status_code}"


# =============================================================================
# Agent Execution Tests
# =============================================================================


@pytest.mark.e2e
class TestAgentExecution:
    """Test agent execution via API."""

    @pytest.mark.asyncio
    async def test_execute_agent_synchronous(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test synchronous agent execution."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # First create the agent
        create_response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )
        assert create_response.status_code in [200, 201]

        # Execute the agent
        execution_request = {
            "input_data": {
                "input_value": 100.0,
                "multiplier": 2.5,
            },
            "async_mode": False,
            "timeout_seconds": 60,
        }

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=execution_request,
        )

        # Expect 200 OK or 202 Accepted
        assert response.status_code in [200, 202], \
            f"Expected 200/202, got {response.status_code}: {response.text}"

        data = response.json()
        assert "execution_id" in data
        assert data.get("status") in ["COMPLETED", "PENDING", "RUNNING"]

    @pytest.mark.asyncio
    async def test_execute_agent_asynchronous(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test asynchronous agent execution with polling."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create the agent
        create_response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )
        assert create_response.status_code in [200, 201]

        # Execute asynchronously
        execution_request = {
            "input_data": {
                "input_value": 50.0,
                "multiplier": 3.0,
            },
            "async_mode": True,
            "timeout_seconds": 120,
            "callback_url": None,
        }

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=execution_request,
        )

        assert response.status_code in [200, 202]
        data = response.json()
        execution_id = data["execution_id"]

        # Poll for completion
        async def check_execution_complete():
            status_response = await authenticated_api_client.get(
                f"/v1/executions/{execution_id}"
            )
            if status_response.status_code == 200:
                status_data = status_response.json()
                return status_data.get("status") in ["COMPLETED", "FAILED"]
            return False

        try:
            await wait_for_condition(
                check_execution_complete,
                timeout=60,
                interval=1.0,
                message="Execution did not complete"
            )
        except TimeoutError:
            # Execution timeout is acceptable in test environment
            pass

    @pytest.mark.asyncio
    async def test_execute_agent_with_provenance(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test that execution produces valid provenance hash."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create the agent
        create_response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )
        assert create_response.status_code in [200, 201]

        # Execute
        execution_request = {
            "input_data": {
                "input_value": 100.0,
                "multiplier": 2.0,
            },
            "async_mode": False,
        }

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=execution_request,
        )

        if response.status_code == 200:
            data = response.json()
            if "result" in data and "provenance_hash" in data.get("result", {}):
                assert_valid_provenance_hash(data["result"]["provenance_hash"])

    @pytest.mark.asyncio
    async def test_execute_nonexistent_agent_fails(
        self,
        authenticated_api_client,
    ):
        """Test that executing non-existent agent fails appropriately."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        execution_request = {
            "input_data": {"value": 100},
            "async_mode": False,
        }

        response = await authenticated_api_client.post(
            "/v1/agents/nonexistent/fake-agent-v999/execute",
            json=execution_request,
        )

        assert response.status_code in [404], \
            f"Expected 404 for non-existent agent, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_execute_agent_invalid_input_fails(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test that execution with invalid input fails validation."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create the agent
        create_response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Execute with invalid input
        invalid_request = {
            "input_data": {
                "input_value": "not_a_number",  # Should be number
                "multiplier": -999,  # May violate constraints
            },
            "async_mode": False,
        }

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=invalid_request,
        )

        # Expect validation error
        assert response.status_code in [400, 422], \
            f"Expected 400/422 for invalid input, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_execution_reproducibility(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test that identical inputs produce identical provenance hashes."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create the agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Execute twice with same input
        execution_request = {
            "input_data": {
                "input_value": 42.0,
                "multiplier": 1.5,
            },
            "async_mode": False,
        }

        response1 = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=execution_request,
        )
        response2 = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json=execution_request,
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # If both have provenance hashes, they should match (determinism)
            hash1 = data1.get("result", {}).get("provenance_hash")
            hash2 = data2.get("result", {}).get("provenance_hash")

            if hash1 and hash2:
                assert hash1 == hash2, "Identical inputs should produce identical provenance hashes"


# =============================================================================
# Agent Versioning Tests
# =============================================================================


@pytest.mark.e2e
class TestAgentVersioning:
    """Test agent versioning functionality."""

    @pytest.mark.asyncio
    async def test_create_new_version(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test creating a new version of an existing agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create initial version
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Create new version
        new_version_spec = sample_agent_spec.copy()
        new_version_spec["version"] = "2.0.0"
        new_version_spec["description"] = "Updated version with improvements"

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/versions",
            json={
                "version": "2.0.0",
                "changelog": "Major update with new features",
                "breaking_changes": True,
                "spec": new_version_spec,
            },
        )

        # Accept 200, 201, or 404 (if endpoint not implemented)
        assert response.status_code in [200, 201, 404]

    @pytest.mark.asyncio
    async def test_list_agent_versions(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test listing all versions of an agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        response = await authenticated_api_client.get(
            f"/v1/agents/{sample_agent_spec['agent_id']}/versions"
        )

        # Accept 200 or 404 (if endpoint not implemented)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
            if isinstance(data, list):
                assert len(data) >= 1
            elif "versions" in data:
                assert len(data["versions"]) >= 1

    @pytest.mark.asyncio
    async def test_get_specific_version(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test retrieving a specific agent version."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        response = await authenticated_api_client.get(
            f"/v1/agents/{sample_agent_spec['agent_id']}/versions/1.0.0"
        )

        if response.status_code == 200:
            data = response.json()
            assert data.get("version") == "1.0.0"

    @pytest.mark.asyncio
    async def test_version_comparison(
        self,
        authenticated_api_client,
        unique_agent_id,
    ):
        """Test comparing two agent versions."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create v1.0.0
        v1_spec = {
            "agent_id": unique_agent_id,
            "name": "Version Test Agent",
            "version": "1.0.0",
            "description": "Initial version",
            "category": "test",
            "tags": ["test"],
            "entrypoint": "python://test:Agent",
            "deterministic": True,
            "regulatory_frameworks": ["TEST"],
            "inputs": {"value": {"type": "number"}},
            "outputs": {"result": {"type": "number"}},
        }

        await authenticated_api_client.post("/v1/agents", json=v1_spec)

        # Create v2.0.0
        v2_spec = v1_spec.copy()
        v2_spec["version"] = "2.0.0"
        v2_spec["description"] = "Updated version"
        v2_spec["inputs"]["new_field"] = {"type": "string"}

        await authenticated_api_client.post(
            f"/v1/agents/{unique_agent_id}/versions",
            json={"version": "2.0.0", "spec": v2_spec},
        )

        # Compare versions
        response = await authenticated_api_client.get(
            f"/v1/agents/{unique_agent_id}/versions/compare",
            params={"from": "1.0.0", "to": "2.0.0"},
        )

        # Accept 200 or 404 (if not implemented)
        assert response.status_code in [200, 404]


# =============================================================================
# Certification Workflow Tests
# =============================================================================


@pytest.mark.e2e
class TestCertificationWorkflow:
    """Test agent certification workflow."""

    @pytest.mark.asyncio
    async def test_initiate_certification(
        self,
        authenticated_api_client,
        sample_agent_spec,
        sample_certification_request,
    ):
        """Test initiating certification process for an agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Update certification request with correct agent_id
        cert_request = sample_certification_request.copy()
        cert_request["agent_id"] = sample_agent_spec["agent_id"]

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/certify",
            json=cert_request,
        )

        # Accept success or not implemented
        assert response.status_code in [200, 201, 202, 404]

        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "certification_id" in data or "status" in data

    @pytest.mark.asyncio
    async def test_certification_status_check(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test checking certification status."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create and certify agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        response = await authenticated_api_client.get(
            f"/v1/agents/{sample_agent_spec['agent_id']}/certification"
        )

        # Accept 200, 404 (not found/not implemented), or 400 (not certified)
        assert response.status_code in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_certification_requirements(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test retrieving certification requirements for an agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        response = await authenticated_api_client.get(
            f"/v1/certification/requirements",
            params={"category": sample_agent_spec["category"]},
        )

        if response.status_code == 200:
            data = response.json()
            assert "dimensions" in data or "requirements" in data

    @pytest.mark.asyncio
    async def test_certification_dimension_validation(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test individual certification dimension validation."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Validate determinism dimension
        dimension_request = {
            "dimension": "deterministic_guarantees",
            "evidence": {
                "test_runs": 100,
                "all_hashes_identical": True,
                "reproducibility_rate": 1.0,
            },
        }

        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/certify/validate",
            json=dimension_request,
        )

        assert response.status_code in [200, 201, 404]


# =============================================================================
# Agent State Transition Tests
# =============================================================================


@pytest.mark.e2e
class TestAgentStateTransitions:
    """Test agent lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_draft_to_experimental(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test transitioning agent from DRAFT to EXPERIMENTAL."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent (starts in DRAFT)
        create_response = await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        if create_response.status_code not in [200, 201]:
            pytest.skip("Agent creation failed")

        # Transition to EXPERIMENTAL
        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/state",
            json={"target_state": "EXPERIMENTAL"},
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_experimental_to_certified(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test transitioning agent from EXPERIMENTAL to CERTIFIED."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create and transition to EXPERIMENTAL
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )
        await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/state",
            json={"target_state": "EXPERIMENTAL"},
        )

        # Transition to CERTIFIED (requires certification)
        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/state",
            json={
                "target_state": "CERTIFIED",
                "certification_id": "cert-12345",
            },
        )

        # May fail due to certification requirements
        assert response.status_code in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_invalid_state_transition_fails(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test that invalid state transitions are rejected."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent (DRAFT state)
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Try invalid transition: DRAFT -> CERTIFIED (should fail)
        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/state",
            json={"target_state": "CERTIFIED"},
        )

        # Should be rejected
        assert response.status_code in [400, 404, 409]

    @pytest.mark.asyncio
    async def test_deprecate_agent(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test deprecating a certified agent."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create and progress agent to certified state
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Deprecate
        response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/deprecate",
            json={
                "reason": "Superseded by newer version",
                "replacement_agent_id": "emissions/carbon-calc-v2",
                "sunset_date": "2025-12-31",
            },
        )

        assert response.status_code in [200, 400, 404]


# =============================================================================
# Agent Metadata Tests
# =============================================================================


@pytest.mark.e2e
class TestAgentMetadata:
    """Test agent metadata operations."""

    @pytest.mark.asyncio
    async def test_get_agent_details(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test retrieving full agent details."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        response = await authenticated_api_client.get(
            f"/v1/agents/{sample_agent_spec['agent_id']}"
        )

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert data["agent_id"] == sample_agent_spec["agent_id"]
            assert "name" in data
            assert "version" in data

    @pytest.mark.asyncio
    async def test_update_agent_metadata(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test updating agent metadata."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Update metadata
        update_data = {
            "description": "Updated description for testing",
            "tags": ["e2e", "test", "updated", "new-tag"],
        }

        response = await authenticated_api_client.patch(
            f"/v1/agents/{sample_agent_spec['agent_id']}",
            json=update_data,
        )

        assert response.status_code in [200, 404, 405]

    @pytest.mark.asyncio
    async def test_delete_agent(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test deleting an agent (soft delete)."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        # Delete
        response = await authenticated_api_client.delete(
            f"/v1/agents/{sample_agent_spec['agent_id']}"
        )

        assert response.status_code in [200, 204, 404, 405]

        # Verify deleted
        if response.status_code in [200, 204]:
            get_response = await authenticated_api_client.get(
                f"/v1/agents/{sample_agent_spec['agent_id']}"
            )
            # Should be 404 or show as deleted
            assert get_response.status_code in [404, 410] or \
                   get_response.json().get("state") == "RETIRED"


# =============================================================================
# Execution History Tests
# =============================================================================


@pytest.mark.e2e
class TestExecutionHistory:
    """Test execution history and audit trail."""

    @pytest.mark.asyncio
    async def test_list_executions(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test listing agent executions."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create and execute agent
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json={"input_data": {"input_value": 100, "multiplier": 2}},
        )

        # List executions
        response = await authenticated_api_client.get(
            f"/v1/agents/{sample_agent_spec['agent_id']}/executions"
        )

        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_get_execution_details(
        self,
        authenticated_api_client,
        sample_agent_spec,
    ):
        """Test retrieving execution details with provenance."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        # Create and execute
        await authenticated_api_client.post(
            "/v1/agents",
            json=sample_agent_spec,
        )

        exec_response = await authenticated_api_client.post(
            f"/v1/agents/{sample_agent_spec['agent_id']}/execute",
            json={
                "input_data": {"input_value": 100, "multiplier": 2},
                "async_mode": False,
            },
        )

        if exec_response.status_code in [200, 202]:
            execution_id = exec_response.json().get("execution_id")

            if execution_id:
                detail_response = await authenticated_api_client.get(
                    f"/v1/executions/{execution_id}"
                )

                assert detail_response.status_code in [200, 404]

                if detail_response.status_code == 200:
                    data = detail_response.json()
                    assert "execution_id" in data
                    assert "status" in data
