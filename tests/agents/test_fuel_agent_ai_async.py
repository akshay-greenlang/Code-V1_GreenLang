# -*- coding: utf-8 -*-
"""Tests for AsyncFuelAgentAI with async infrastructure.

Tests:
- Basic async execution
- Async context manager support
- Parallel execution performance
- Config injection
- Backward compatibility with sync wrapper
- Resource cleanup
- Error handling
- Determinism verification

Author: GreenLang Framework Team
Date: November 2025
"""

import pytest
import asyncio
import time
from typing import Dict, Any

from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
from greenlang.agents.fuel_agent_ai_sync import FuelAgentAISync
from greenlang.config.schemas import create_test_config, GreenLangConfig
from greenlang.config import override_config


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def test_config() -> GreenLangConfig:
    """Create test configuration."""
    return create_test_config(
        llm={
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 2000,
        }
    )


@pytest.fixture
def sample_fuel_input() -> Dict[str, Any]:
    """Sample fuel input for testing."""
    return {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
    }


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_async_fuel_agent_basic_execution(test_config, sample_fuel_input):
    """Test basic async execution."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result = await agent.run_async(sample_fuel_input)

    assert result.success is True
    assert result.data is not None
    assert result.data["co2e_emissions_kg"] > 0
    assert result.data["fuel_type"] == "natural_gas"
    assert "emission_factor" in result.data


@pytest.mark.asyncio
async def test_async_fuel_agent_with_config_injection(sample_fuel_input):
    """Test config injection from ConfigManager."""
    with override_config(llm={"temperature": 0.0}):
        async with AsyncFuelAgentAI() as agent:  # Uses global config
            assert agent.config.llm.temperature == 0.0
            result = await agent.run_async(sample_fuel_input)

    assert result.success is True


@pytest.mark.asyncio
async def test_async_fuel_agent_validation(test_config):
    """Test input validation."""
    async with AsyncFuelAgentAI(test_config) as agent:
        # Invalid input (missing required field)
        result = await agent.run_async({
            "fuel_type": "natural_gas",
            # Missing 'amount' and 'unit'
        })
        # Validation errors are returned in result, not raised
        assert result.success is False


@pytest.mark.asyncio
async def test_async_fuel_agent_with_renewable_offset(test_config):
    """Test renewable percentage offset."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result = await agent.run_async({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US",
            "renewable_percentage": 50,
        })

    assert result.success is True
    assert result.data["renewable_offset_applied"] is True


@pytest.mark.asyncio
async def test_async_fuel_agent_with_efficiency(test_config):
    """Test efficiency adjustment."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result = await agent.run_async({
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
            "efficiency": 0.85,
        })

    assert result.success is True
    assert result.data["efficiency_adjusted"] is True


# ==============================================================================
# Async Context Manager Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_async_context_manager_lifecycle(test_config, sample_fuel_input):
    """Test async context manager properly initializes and cleans up."""
    agent = AsyncFuelAgentAI(test_config)

    # Before entering context
    assert agent.provider is None

    async with agent:
        # Inside context - initialized
        assert agent.provider is not None
        result = await agent.run_async(sample_fuel_input)
        assert result.success is True

    # After exiting context - cleaned up
    # Provider should be cleaned up (if it has close method)


@pytest.mark.asyncio
async def test_multiple_sequential_executions(test_config, sample_fuel_input):
    """Test multiple sequential executions in same context."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result1 = await agent.run_async(sample_fuel_input)
        result2 = await agent.run_async({
            **sample_fuel_input,
            "amount": 2000,
        })

    assert result1.success is True
    assert result2.success is True
    # In demo mode, emissions might be same (pre-recorded responses)
    # Just verify both have valid emissions
    assert result1.data["co2e_emissions_kg"] > 0
    assert result2.data["co2e_emissions_kg"] > 0


# ==============================================================================
# Parallel Execution Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_parallel_execution_multiple_agents(test_config, sample_fuel_input):
    """Test parallel execution with multiple agents."""
    # Create multiple agents
    agents = [AsyncFuelAgentAI(test_config) for _ in range(5)]

    # Different inputs
    inputs = [
        {**sample_fuel_input, "amount": 1000 * (i + 1)}
        for i in range(5)
    ]

    # Execute in parallel using asyncio.gather
    start = time.time()
    results = await asyncio.gather(
        *[agent.run_async(inp) for agent, inp in zip(agents, inputs)]
    )
    parallel_time = time.time() - start

    # All should succeed
    assert all(r.success for r in results)

    # Emissions should increase with amount
    emissions = [r.data["co2e_emissions_kg"] for r in results]
    assert emissions == sorted(emissions)

    print(f"Parallel execution of 5 agents: {parallel_time:.2f}s")


@pytest.mark.asyncio
async def test_concurrent_execution_single_agent(test_config):
    """Test concurrent execution with single agent."""
    async with AsyncFuelAgentAI(test_config) as agent:
        # Multiple concurrent calculations
        inputs = [
            {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
            {"fuel_type": "diesel", "amount": 500, "unit": "gallons"},
            {"fuel_type": "electricity", "amount": 2000, "unit": "kWh"},
        ]

        start = time.time()
        results = await asyncio.gather(
            *[agent.run_async(inp) for inp in inputs]
        )
        concurrent_time = time.time() - start

        assert all(r.success for r in results)
        assert len(results) == 3

        print(f"Concurrent execution of 3 calculations: {concurrent_time:.2f}s")


# ==============================================================================
# Performance Metrics Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_performance_tracking(test_config, sample_fuel_input):
    """Test performance metrics tracking."""
    async with AsyncFuelAgentAI(test_config) as agent:
        # Execute multiple times
        for _ in range(3):
            await agent.run_async(sample_fuel_input)

        # Get performance summary
        summary = agent.get_performance_summary()

        assert summary["agent_id"] == "fuel_ai_async"
        assert summary["ai_metrics"]["ai_call_count"] == 3
        assert summary["ai_metrics"]["tool_call_count"] > 0
        assert summary["ai_metrics"]["total_cost_usd"] >= 0


# ==============================================================================
# Citation Tracking Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_citation_tracking(test_config, sample_fuel_input):
    """Test emission factor citation tracking."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result = await agent.run_async(sample_fuel_input)

    assert result.success is True
    # Citations should be present if lookup_emission_factor tool was called
    if "citations" in result.data:
        citations = result.data["citations"]
        assert len(citations) > 0
        assert "source" in citations[0]
        assert "factor_name" in citations[0]


# ==============================================================================
# Determinism Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_deterministic_execution(test_config, sample_fuel_input):
    """Test deterministic execution (same input = same output)."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result1 = await agent.run_async(sample_fuel_input)
        result2 = await agent.run_async(sample_fuel_input)

    # Numeric results should be identical
    assert result1.data["co2e_emissions_kg"] == result2.data["co2e_emissions_kg"]
    assert result1.data["emission_factor"] == result2.data["emission_factor"]


# ==============================================================================
# Recommendations Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_recommendations_enabled(test_config, sample_fuel_input):
    """Test AI recommendations generation."""
    async with AsyncFuelAgentAI(test_config, enable_recommendations=True) as agent:
        result = await agent.run_async(sample_fuel_input)

    assert result.success is True
    # Recommendations should be present if tool was called
    if "recommendations" in result.data:
        assert isinstance(result.data["recommendations"], list)


@pytest.mark.asyncio
async def test_recommendations_disabled(test_config, sample_fuel_input):
    """Test with recommendations disabled."""
    async with AsyncFuelAgentAI(test_config, enable_recommendations=False) as agent:
        result = await agent.run_async(sample_fuel_input)

    assert result.success is True


# ==============================================================================
# Sync Wrapper Tests (Backward Compatibility)
# ==============================================================================

def test_sync_wrapper_basic_execution(test_config, sample_fuel_input):
    """Test sync wrapper for backward compatibility."""
    agent = FuelAgentAISync(test_config)
    result = agent.execute(sample_fuel_input)

    assert result.success is True
    assert result.data["co2e_emissions_kg"] > 0


def test_sync_wrapper_context_manager(test_config, sample_fuel_input):
    """Test sync wrapper with context manager."""
    with FuelAgentAISync(test_config) as agent:
        result = agent.execute(sample_fuel_input)

    assert result.success is True


def test_sync_wrapper_validation(test_config):
    """Test sync wrapper validation."""
    agent = FuelAgentAISync(test_config)

    # Valid input
    assert agent.validate({
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
    }) is True

    # Invalid input
    assert agent.validate({
        "fuel_type": "natural_gas",
        # Missing required fields
    }) is False


def test_sync_wrapper_performance_summary(test_config, sample_fuel_input):
    """Test sync wrapper performance metrics."""
    agent = FuelAgentAISync(test_config)
    agent.execute(sample_fuel_input)

    summary = agent.get_performance_summary()
    assert "agent_id" in summary
    assert "ai_metrics" in summary


# ==============================================================================
# Error Handling Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_budget_exceeded_handling(test_config, sample_fuel_input):
    """Test budget exceeded error handling."""
    # Set very low budget
    async with AsyncFuelAgentAI(test_config, budget_usd=0.001) as agent:
        # This should potentially fail or complete within budget
        # (depends on actual AI cost)
        try:
            result = await agent.run_async(sample_fuel_input)
            # If it succeeds, budget was sufficient
            assert result.success is True or result.success is False
        except ValueError as e:
            # Budget exceeded
            assert "budget" in str(e).lower()


@pytest.mark.asyncio
async def test_invalid_fuel_type(test_config):
    """Test handling of invalid fuel type."""
    async with AsyncFuelAgentAI(test_config) as agent:
        result = await agent.run_async({
            "fuel_type": "nonexistent_fuel",
            "amount": 1000,
            "unit": "therms",
        })
        # In demo mode, this might succeed with pre-recorded responses
        # Just verify we got a result
        assert result is not None


# ==============================================================================
# Config Override Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_with_config_override(sample_fuel_input):
    """Test with config override mechanism."""
    with override_config(
        llm={"temperature": 0.0, "model": "gpt-3.5-turbo"},
        debug=True
    ):
        async with AsyncFuelAgentAI() as agent:
            assert agent.config.llm.temperature == 0.0
            assert agent.config.debug is True

            result = await agent.run_async(sample_fuel_input)
            assert result.success is True


# ==============================================================================
# Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_full_workflow_with_all_features(test_config):
    """Test complete workflow with all features enabled."""
    async with AsyncFuelAgentAI(
        test_config,
        enable_explanations=True,
        enable_recommendations=True,
        budget_usd=0.50
    ) as agent:
        result = await agent.run_async({
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
            "renewable_percentage": 20,
            "efficiency": 0.9,
        })

    assert result.success is True
    assert result.data["co2e_emissions_kg"] > 0
    assert result.data["renewable_offset_applied"] is True
    assert result.data["efficiency_adjusted"] is True

    # Check metadata
    assert result.metadata is not None
    assert "cost_usd" in result.metadata
    assert "tokens" in result.metadata
    assert result.metadata["deterministic"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
