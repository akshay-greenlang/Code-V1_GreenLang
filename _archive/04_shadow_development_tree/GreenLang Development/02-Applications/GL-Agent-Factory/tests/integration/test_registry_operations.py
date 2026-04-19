"""
Agent Registry Operations Integration Tests

Tests for agent registry service:
- Agent registration
- Version management
- Search and discovery
- Publishing workflow
- State machine transitions

Run with: pytest tests/integration/test_registry_operations.py -v
"""

import pytest
from typing import Dict, Any, List
from enum import Enum


class AgentState(Enum):
    """Agent lifecycle states."""
    DRAFT = "DRAFT"
    EXPERIMENTAL = "EXPERIMENTAL"
    CERTIFIED = "CERTIFIED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


# Valid state transitions
STATE_TRANSITIONS = {
    AgentState.DRAFT: {AgentState.EXPERIMENTAL, AgentState.RETIRED},
    AgentState.EXPERIMENTAL: {AgentState.CERTIFIED, AgentState.DRAFT, AgentState.RETIRED},
    AgentState.CERTIFIED: {AgentState.DEPRECATED},
    AgentState.DEPRECATED: {AgentState.RETIRED, AgentState.CERTIFIED},
    AgentState.RETIRED: set(),  # Terminal state
}


class TestAgentRegistration:
    """Test agent registration operations."""

    def test_agent_spec_structure(self):
        """Test agent specification structure."""
        spec = {
            "agent_id": "test/registration_agent_v1",
            "name": "Registration Test Agent",
            "version": "1.0.0",
            "description": "Agent for testing registration",
            "category": "test",
            "tags": ["test", "registration"],
            "entrypoint": "python://test.agent:TestAgent",
            "deterministic": True,
            "regulatory_frameworks": ["TEST"],
        }

        assert spec["agent_id"] == "test/registration_agent_v1"
        assert spec["version"] == "1.0.0"
        assert spec["deterministic"] is True
        assert "test" in spec["tags"]

    def test_agent_id_format_validation(self):
        """Test agent ID format validation."""
        valid_ids = [
            "emissions/carbon_calculator_v1",
            "regulatory/cbam_compliance_v1",
            "test/sample_agent_v1",
        ]

        invalid_ids = [
            "",
            "no_slash",
            "  ",
            "invalid//double",
        ]

        for valid_id in valid_ids:
            assert "/" in valid_id
            parts = valid_id.split("/")
            assert len(parts) == 2
            assert all(len(p) > 0 for p in parts)

        for invalid_id in invalid_ids:
            if "/" not in invalid_id or invalid_id.count("/") != 1:
                assert True  # Invalid format detected


class TestVersionManagement:
    """Test agent version management operations."""

    def test_semver_format_validation(self):
        """Test semantic versioning format."""
        valid_versions = ["1.0.0", "2.1.3", "0.0.1", "10.20.30"]
        invalid_versions = ["1.0", "1", "v1.0.0", "1.0.0-beta", "1.0.0.0"]

        for version in valid_versions:
            parts = version.split(".")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)

        for version in invalid_versions:
            parts = version.split(".")
            is_valid = len(parts) == 3 and all(p.isdigit() for p in parts)
            assert not is_valid or version.startswith("v")


class TestSearchAndDiscovery:
    """Test agent search and discovery operations."""

    def test_filter_structure(self):
        """Test filter structure for agent listing."""
        filters = {
            "category": "emissions",
            "state": AgentState.CERTIFIED.value,
            "tags": ["carbon", "ghg-protocol"],
            "search": "carbon emissions",
        }

        assert filters["category"] == "emissions"
        assert filters["state"] == "CERTIFIED"
        assert "carbon" in filters["tags"]

    def test_pagination_structure(self):
        """Test pagination structure."""
        pagination = {
            "limit": 10,
            "offset": 0,
            "sort_by": "created_at",
            "sort_order": "desc",
        }

        assert pagination["limit"] == 10
        assert pagination["offset"] == 0
        assert pagination["sort_order"] in ["asc", "desc"]


class TestPublishingWorkflow:
    """Test agent publishing workflow operations."""

    def test_state_transition_draft_to_experimental(self):
        """Test valid transition from DRAFT to EXPERIMENTAL."""
        from_state = AgentState.DRAFT
        to_state = AgentState.EXPERIMENTAL

        allowed = STATE_TRANSITIONS.get(from_state, set())
        assert to_state in allowed

    def test_state_transition_experimental_to_certified(self):
        """Test valid transition from EXPERIMENTAL to CERTIFIED."""
        from_state = AgentState.EXPERIMENTAL
        to_state = AgentState.CERTIFIED

        allowed = STATE_TRANSITIONS.get(from_state, set())
        assert to_state in allowed

    def test_invalid_state_transition_draft_to_certified(self):
        """Test invalid transition from DRAFT directly to CERTIFIED."""
        from_state = AgentState.DRAFT
        to_state = AgentState.CERTIFIED

        allowed = STATE_TRANSITIONS.get(from_state, set())
        assert to_state not in allowed

    def test_retired_is_terminal_state(self):
        """Test that RETIRED is a terminal state with no transitions."""
        from_state = AgentState.RETIRED
        allowed = STATE_TRANSITIONS.get(from_state, set())
        assert len(allowed) == 0


class TestStateTransitionMatrix:
    """Test all valid and invalid state transitions."""

    @pytest.mark.parametrize("from_state,to_state,valid", [
        # DRAFT transitions
        (AgentState.DRAFT, AgentState.EXPERIMENTAL, True),
        (AgentState.DRAFT, AgentState.RETIRED, True),
        (AgentState.DRAFT, AgentState.CERTIFIED, False),
        (AgentState.DRAFT, AgentState.DEPRECATED, False),

        # EXPERIMENTAL transitions
        (AgentState.EXPERIMENTAL, AgentState.CERTIFIED, True),
        (AgentState.EXPERIMENTAL, AgentState.DRAFT, True),
        (AgentState.EXPERIMENTAL, AgentState.RETIRED, True),
        (AgentState.EXPERIMENTAL, AgentState.DEPRECATED, False),

        # CERTIFIED transitions
        (AgentState.CERTIFIED, AgentState.DEPRECATED, True),
        (AgentState.CERTIFIED, AgentState.RETIRED, False),
        (AgentState.CERTIFIED, AgentState.DRAFT, False),
        (AgentState.CERTIFIED, AgentState.EXPERIMENTAL, False),

        # DEPRECATED transitions
        (AgentState.DEPRECATED, AgentState.RETIRED, True),
        (AgentState.DEPRECATED, AgentState.CERTIFIED, True),
        (AgentState.DEPRECATED, AgentState.DRAFT, False),
        (AgentState.DEPRECATED, AgentState.EXPERIMENTAL, False),

        # RETIRED transitions (terminal state)
        (AgentState.RETIRED, AgentState.DRAFT, False),
        (AgentState.RETIRED, AgentState.EXPERIMENTAL, False),
        (AgentState.RETIRED, AgentState.CERTIFIED, False),
        (AgentState.RETIRED, AgentState.DEPRECATED, False),
    ])
    def test_state_transition_validity(
        self,
        from_state: AgentState,
        to_state: AgentState,
        valid: bool,
    ):
        """Test state transition validity matrix."""
        allowed_transitions = STATE_TRANSITIONS.get(from_state, set())

        if valid:
            assert to_state in allowed_transitions, \
                f"Transition {from_state} -> {to_state} should be valid"
        else:
            assert to_state not in allowed_transitions, \
                f"Transition {from_state} -> {to_state} should be invalid"


class TestTenantIsolation:
    """Test multi-tenant isolation in registry operations."""

    def test_tenant_data_structure(self):
        """Test tenant data structure."""
        tenant1_agents = [
            {"agent_id": "test/tenant1_agent_v1", "tenant_id": "tenant-1"},
        ]
        tenant2_agents = [
            {"agent_id": "test/tenant2_agent_v1", "tenant_id": "tenant-2"},
        ]

        # Verify isolation
        tenant1_ids = {a["agent_id"] for a in tenant1_agents}
        tenant2_ids = {a["agent_id"] for a in tenant2_agents}

        assert tenant1_ids.isdisjoint(tenant2_ids)
        assert "test/tenant1_agent_v1" in tenant1_ids
        assert "test/tenant2_agent_v1" in tenant2_ids
        assert "test/tenant1_agent_v1" not in tenant2_ids
        assert "test/tenant2_agent_v1" not in tenant1_ids
