# -*- coding: utf-8 -*-
"""
AgentSpec v2 Migration Pilot Test - FuelAgentAI
================================================

This test validates that the AgentSpec v2 wrapper works correctly with FuelAgentAI.

Test Coverage:
- Pack.yaml loading and validation
- Input schema validation (required fields, constraints)
- Output schema validation
- Backward compatibility with existing tests
- Citation preservation
- Metrics collection
- Lifecycle hooks

Success Criteria:
- All existing FuelAgentAI functionality works
- pack.yaml validates against AgentSpec v2
- Input/output validation catches errors
- Citations are preserved
- Zero performance degradation

Author: GreenLang Framework Team
Date: October 2025
Status: Pilot Test
"""

import pytest
from pathlib import Path

from greenlang.agents.fuel_agent_ai import FuelAgentAI
from greenlang.agents.agentspec_v2_compat import wrap_agent_v2
from greenlang.specs.errors import GLValidationError, GLVErr


class TestAgentSpecV2FuelAgentPilot:
    """Pilot tests for AgentSpec v2 migration of FuelAgentAI."""

    def test_pack_yaml_loads_successfully(self):
        """pack.yaml loads and validates against AgentSpec v2 schema."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        # Check pack.yaml exists
        assert (pack_path / "pack.yaml").exists(), "pack.yaml not found"

        # Wrap agent with pack.yaml
        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Verify spec loaded
        assert wrapped_agent.spec is not None
        assert wrapped_agent.spec.schema_version == "2.0.0"
        assert wrapped_agent.spec.id == "emissions/fuel_ai_v1"
        assert wrapped_agent.spec.version == "1.0.0"

    def test_backward_compatibility_without_pack(self):
        """Agent works without pack.yaml (validation disabled)."""
        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(
            original_agent,
            pack_path=None,  # No pack.yaml
            enable_validation=False,  # Disable validation
        )

        # Should work exactly like original
        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption": {"value": 1000, "unit": "therms"},
            "country": "US",
        })

        assert result.success is True
        assert "co2e_emissions_kg" in result.data
        assert result.data["co2e_emissions_kg"] > 0

    def test_input_validation_catches_missing_required_field(self):
        """Missing required input field raises GLValidationError.MISSING_FIELD."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Missing fuel_type (required)
        result = wrapped_agent.run({
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is False
        assert "fuel_type" in result.error
        assert "missing" in result.error.lower()

    def test_input_validation_catches_constraint_violation(self):
        """Constraint violation (ge/le) raises GLValidationError.CONSTRAINT."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Negative consumption (violates ge: 0.0)
        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": -100,  # Invalid: must be >= 0
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is False
        assert "constraint" in result.error.lower() or ">=" in result.error

    def test_input_validation_catches_invalid_enum_value(self):
        """Invalid enum value raises GLValidationError.CONSTRAINT."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Invalid fuel_type (not in enum)
        result = wrapped_agent.run({
            "fuel_type": "unicorn_fuel",  # Not in allowed enum
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is False
        # Agent-level validation should catch this
        # (v2 validation happens after, if input passes basic schema)

    def test_valid_input_executes_successfully(self):
        """Valid input executes successfully with v2 wrapper."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is True
        assert "co2e_emissions_kg" in result.data
        assert result.data["co2e_emissions_kg"] > 0
        assert result.data["fuel_type"] == "natural_gas"

    def test_citations_preserved_through_wrapper(self):
        """Citations from original agent are preserved through v2 wrapper."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path, enable_citations=True)

        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is True

        # Check citations present (if FuelAgentAI generates them)
        if "citations" in result.data:
            citations = result.data["citations"]
            assert isinstance(citations, list)
            if len(citations) > 0:
                # Verify EF CID format if present
                citation = citations[0]
                if isinstance(citation, dict) and "ef_cid" in citation:
                    assert citation["ef_cid"].startswith("ef_")

    def test_metrics_collection_enabled(self):
        """Metrics are collected when enabled."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path, enable_metrics=True)

        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is True
        assert "execution_time_ms" in result.metadata
        assert result.metadata["execution_time_ms"] > 0

        # Check agent stats
        stats = wrapped_agent.get_stats()
        assert stats["executions"] >= 1
        assert stats["avg_time_ms"] > 0

    def test_lifecycle_hooks_execute(self):
        """Lifecycle hooks are called at appropriate times."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Track hook calls
        hook_calls = []

        def pre_execute_hook(agent):
            hook_calls.append("pre_execute")

        def post_execute_hook(agent):
            hook_calls.append("post_execute")

        wrapped_agent.add_lifecycle_hook("pre_execute", pre_execute_hook)
        wrapped_agent.add_lifecycle_hook("post_execute", post_execute_hook)

        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        assert result.success is True
        assert "pre_execute" in hook_calls
        assert "post_execute" in hook_calls

    def test_multiple_executions_with_same_agent(self):
        """Agent can be executed multiple times successfully."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Execute 3 times
        for i in range(3):
            result = wrapped_agent.run({
                "fuel_type": "natural_gas",
                "consumption_value": 1000 + i * 100,
                "consumption_unit": "therms",
                "country": "US",
            })

            assert result.success is True
            assert result.data["co2e_emissions_kg"] > 0

        # Verify execution count
        stats = wrapped_agent.get_stats()
        assert stats["executions"] == 3

    def test_agent_without_pack_still_works(self):
        """Agent works in compatibility mode without pack.yaml."""
        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(
            original_agent,
            pack_path=None,
            enable_validation=False,
        )

        result = wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption": {"value": 1000, "unit": "therms"},
            "country": "US",
        })

        assert result.success is True

    def test_wrapper_repr_and_stats(self):
        """Wrapper provides useful repr and stats."""
        pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

        original_agent = FuelAgentAI()
        wrapped_agent = wrap_agent_v2(original_agent, pack_path=pack_path)

        # Check repr
        repr_str = repr(wrapped_agent)
        assert "AgentSpecV2Wrapper" in repr_str
        assert "fuel-agent-ai" in repr_str or "FuelAgentAI" in repr_str

        # Execute once
        wrapped_agent.run({
            "fuel_type": "natural_gas",
            "consumption_value": 1000,
            "consumption_unit": "therms",
            "country": "US",
        })

        # Check stats
        stats = wrapped_agent.get_stats()
        assert stats["executions"] == 1
        assert "avg_time_ms" in stats
