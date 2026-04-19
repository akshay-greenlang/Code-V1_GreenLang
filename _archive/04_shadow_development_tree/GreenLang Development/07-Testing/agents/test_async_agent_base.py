# -*- coding: utf-8 -*-
"""
Tests for AsyncAgentBase
=========================

Test coverage:
- Async lifecycle methods (initialize, validate, execute, finalize)
- Async context manager support
- Timeout handling
- Resource cleanup
- Parallel execution with asyncio.gather()
- Performance characteristics
"""

import asyncio
import pytest
import time
from typing import Dict, Any

from greenlang.agents.async_agent_base import (
    AsyncAgentBase,
    AsyncAgentExecutionContext,
    AsyncAgentLifecycleState,
    gather_agent_results,
)
from greenlang.exceptions import TimeoutError as GLTimeoutError
from greenlang.agents.base import AgentResult


# ==============================================================================
# Test Agents
# ==============================================================================

class SimpleAsyncAgent(AsyncAgentBase[Dict[str, Any], Dict[str, Any]]):
    """Simple async agent for testing."""

    async def execute_impl_async(
        self, validated_input: Dict[str, Any], context: AsyncAgentExecutionContext
    ) -> Dict[str, Any]:
        """Simple execution that just returns input."""
        return {"result": validated_input.get("value", 0) * 2}


class SlowAsyncAgent(AsyncAgentBase[Dict[str, Any], Dict[str, Any]]):
    """Async agent that simulates slow I/O."""

    async def execute_impl_async(
        self, validated_input: Dict[str, Any], context: AsyncAgentExecutionContext
    ) -> Dict[str, Any]:
        """Simulate slow LLM call."""
        await asyncio.sleep(validated_input.get("sleep", 0.1))
        return {"result": "completed"}


class FailingAsyncAgent(AsyncAgentBase[Dict[str, Any], Dict[str, Any]]):
    """Async agent that fails."""

    async def execute_impl_async(
        self, validated_input: Dict[str, Any], context: AsyncAgentExecutionContext
    ) -> Dict[str, Any]:
        """Raise an error."""
        raise ValueError("Intentional failure")


class ResourceAsyncAgent(AsyncAgentBase[Dict[str, Any], Dict[str, Any]]):
    """Async agent with resource management."""

    def __init__(self):
        super().__init__()
        self.resource_opened = False
        self.resource_closed = False

    async def initialize_impl_async(self) -> None:
        """Open a resource."""
        self.resource_opened = True

    async def execute_impl_async(
        self, validated_input: Dict[str, Any], context: AsyncAgentExecutionContext
    ) -> Dict[str, Any]:
        """Use the resource."""
        if not self.resource_opened:
            raise RuntimeError("Resource not opened")
        return {"status": "used_resource"}

    async def cleanup_async(self) -> None:
        """Close the resource."""
        await super().cleanup_async()
        self.resource_closed = True


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def simple_agent():
    """Create simple async agent."""
    return SimpleAsyncAgent()


@pytest.fixture
def slow_agent():
    """Create slow async agent."""
    return SlowAsyncAgent()


@pytest.fixture
def failing_agent():
    """Create failing async agent."""
    return FailingAsyncAgent()


@pytest.fixture
def resource_agent():
    """Create resource async agent."""
    return ResourceAsyncAgent()


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_simple_execution(simple_agent):
    """Test basic async execution."""
    result = await simple_agent.run_async({"value": 5})

    assert result.success is True
    assert result.data["result"] == 10
    assert "execution_time_ms" in result.metadata
    assert result.metadata["async_mode"] is True


@pytest.mark.asyncio
async def test_lifecycle_states(simple_agent):
    """Test lifecycle state transitions."""
    # Initial state
    assert simple_agent._state == AsyncAgentLifecycleState.UNINITIALIZED

    # Initialize
    await simple_agent.initialize_async()
    assert simple_agent._state == AsyncAgentLifecycleState.INITIALIZED

    # Execute
    result = await simple_agent.run_async({"value": 3})
    assert result.success is True


@pytest.mark.asyncio
async def test_validation(simple_agent):
    """Test input validation."""
    context = AsyncAgentExecutionContext()

    # Valid input
    validated = await simple_agent.validate_async({"value": 5}, context)
    assert validated["value"] == 5
    assert len(context.errors) == 0


@pytest.mark.asyncio
async def test_error_handling(failing_agent):
    """Test error handling in async execution."""
    result = await failing_agent.run_async({"test": "input"})

    assert result.success is False
    assert "Intentional failure" in result.error
    assert "errors" in result.metadata


# ==============================================================================
# Timeout Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_timeout_enforcement(slow_agent):
    """Test that timeout is enforced."""
    # This should timeout (sleep 2s with 0.5s timeout)
    result = await slow_agent.run_async({"sleep": 2.0}, timeout=0.5)

    assert result.success is False
    assert "timed out" in result.error.lower()
    assert "timeout_error" in result.metadata


@pytest.mark.asyncio
async def test_no_timeout_with_fast_execution(slow_agent):
    """Test that fast execution doesn't timeout."""
    # This should complete (sleep 0.1s with 1s timeout)
    result = await slow_agent.run_async({"sleep": 0.1}, timeout=1.0)

    assert result.success is True
    assert result.data["result"] == "completed"


# ==============================================================================
# Context Manager Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_async_context_manager(resource_agent):
    """Test async context manager support."""
    # Context manager should initialize and cleanup
    async with resource_agent as agent:
        assert agent.resource_opened is True
        result = await agent.run_async({"test": "data"})
        assert result.success is True
        assert result.data["status"] == "used_resource"

    # After context, resource should be closed
    assert resource_agent.resource_closed is True


@pytest.mark.asyncio
async def test_cleanup_on_error():
    """Test that cleanup happens even when errors occur."""
    agent = ResourceAsyncAgent()

    try:
        async with agent:
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Cleanup should still have happened
    assert agent.resource_closed is True


# ==============================================================================
# Parallel Execution Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_parallel_execution():
    """Test that multiple agents can run in parallel."""
    # Create multiple agents
    agents = [SimpleAsyncAgent() for _ in range(5)]

    # Run them in parallel
    start = time.time()
    results = await asyncio.gather(*[
        agent.run_async({"value": i}) for i, agent in enumerate(agents)
    ])
    duration = time.time() - start

    # All should succeed
    assert all(r.success for r in results)

    # Check results
    for i, result in enumerate(results):
        assert result.data["result"] == i * 2

    # Should be fast (parallel execution)
    assert duration < 0.5  # Should be nearly instant


@pytest.mark.asyncio
async def test_gather_agent_results():
    """Test utility function for gathering results."""
    agents = [SimpleAsyncAgent() for _ in range(3)]
    inputs = [{"value": 1}, {"value": 2}, {"value": 3}]

    results = await gather_agent_results(*zip(agents, inputs))

    assert len(results) == 3
    assert results[0].data["result"] == 2
    assert results[1].data["result"] == 4
    assert results[2].data["result"] == 6


# ==============================================================================
# Performance Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_parallel_performance_benefit():
    """Test that parallel execution is faster than sequential."""
    # Create 10 slow agents (each sleeps 0.1s)
    agents = [SlowAsyncAgent() for _ in range(10)]
    inputs = [{"sleep": 0.1} for _ in range(10)]

    # Parallel execution
    start_parallel = time.time()
    await asyncio.gather(*[agent.run_async(inp) for agent, inp in zip(agents, inputs)])
    parallel_time = time.time() - start_parallel

    # Sequential execution
    start_sequential = time.time()
    for agent, inp in zip(agents, inputs):
        await agent.run_async(inp)
    sequential_time = time.time() - start_sequential

    # Parallel should be much faster (at least 5x for 10 agents)
    speedup = sequential_time / parallel_time
    assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.2f}x"


# ==============================================================================
# Metrics Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_metrics_collection(simple_agent):
    """Test that metrics are collected."""
    # Execute multiple times
    for i in range(5):
        await simple_agent.run_async({"value": i})

    # Check stats
    stats = simple_agent.get_stats()
    assert stats["executions"] == 5
    assert stats["total_time_ms"] > 0
    assert stats["avg_time_ms"] > 0
    assert stats["async_mode"] is True


@pytest.mark.asyncio
async def test_execution_time_tracking(simple_agent):
    """Test that execution time is tracked."""
    result = await simple_agent.run_async({"value": 1})

    assert result.success is True
    assert "execution_time_ms" in result.metadata
    assert result.metadata["execution_time_ms"] > 0


# ==============================================================================
# Lifecycle Hook Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_lifecycle_hooks():
    """Test that lifecycle hooks are called."""
    agent = SimpleAsyncAgent()
    hook_calls = []

    # Add hooks
    agent.add_lifecycle_hook("pre_execute", lambda a: hook_calls.append("pre_execute"))
    agent.add_lifecycle_hook("post_execute", lambda a: hook_calls.append("post_execute"))

    # Execute
    await agent.run_async({"value": 1})

    # Hooks should have been called
    assert "pre_execute" in hook_calls
    assert "post_execute" in hook_calls


@pytest.mark.asyncio
async def test_async_hooks():
    """Test that async hooks work."""
    agent = SimpleAsyncAgent()
    hook_calls = []

    async def async_hook(a):
        await asyncio.sleep(0.01)
        hook_calls.append("async_hook")

    agent.add_lifecycle_hook("pre_execute", async_hook)

    # Execute
    await agent.run_async({"value": 1})

    # Async hook should have been called
    assert "async_hook" in hook_calls


# ==============================================================================
# Edge Cases
# ==============================================================================

@pytest.mark.asyncio
async def test_multiple_executions():
    """Test that agent can be executed multiple times."""
    agent = SimpleAsyncAgent()

    # Execute multiple times
    results = []
    for i in range(10):
        result = await agent.run_async({"value": i})
        results.append(result)

    # All should succeed
    assert all(r.success for r in results)
    assert len(results) == 10


@pytest.mark.asyncio
async def test_reinitialization():
    """Test that agent can be reinitialized."""
    agent = SimpleAsyncAgent()

    # Initialize
    await agent.initialize_async()
    assert agent._state == AsyncAgentLifecycleState.INITIALIZED

    # Reinitialize
    await agent.initialize_async()
    assert agent._state == AsyncAgentLifecycleState.INITIALIZED


@pytest.mark.asyncio
async def test_cancellation():
    """Test that long-running execution can be cancelled."""
    agent = SlowAsyncAgent()

    # Start execution
    task = asyncio.create_task(agent.run_async({"sleep": 10.0}))

    # Wait a bit then cancel
    await asyncio.sleep(0.1)
    task.cancel()

    # Should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task


# ==============================================================================
# Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_realistic_workflow():
    """Test realistic workflow with multiple agents."""
    # Create a workflow: agent1 -> agent2 -> agent3
    agent1 = SimpleAsyncAgent()
    agent2 = SimpleAsyncAgent()
    agent3 = SimpleAsyncAgent()

    # Step 1
    result1 = await agent1.run_async({"value": 5})
    assert result1.success

    # Step 2 (uses result1)
    result2 = await agent2.run_async({"value": result1.data["result"]})
    assert result2.success

    # Step 3 (uses result2)
    result3 = await agent3.run_async({"value": result2.data["result"]})
    assert result3.success

    # Final result should be 5 * 2 * 2 * 2 = 40
    assert result3.data["result"] == 40


@pytest.mark.asyncio
async def test_mixed_success_and_failure():
    """Test handling mix of successful and failed agents."""
    agents = [
        SimpleAsyncAgent(),  # Success
        FailingAsyncAgent(),  # Failure
        SimpleAsyncAgent(),  # Success
    ]
    inputs = [{"value": 1}, {"test": "input"}, {"value": 2}]

    # Use gather with return_exceptions=True
    results = await asyncio.gather(
        *[agent.run_async(inp) for agent, inp in zip(agents, inputs)],
        return_exceptions=False  # Will not raise, returns AgentResult with success=False
    )

    # Check results
    assert results[0].success is True  # First agent succeeded
    assert results[1].success is False  # Second agent failed
    assert results[2].success is True  # Third agent succeeded


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
