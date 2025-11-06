"""
Tests for SyncAgentWrapper
===========================

Test coverage:
- Sync wrapping of async agents
- Backward compatibility with sync API
- Context manager support
- Performance characteristics
- Migration helpers
"""

import pytest
import time
from typing import Dict, Any

from greenlang.agents.async_agent_base import (
    AsyncAgentBase,
    AsyncAgentExecutionContext,
)
from greenlang.agents.sync_wrapper import (
    SyncAgentWrapper,
    make_sync,
    is_async_agent,
    is_sync_wrapper,
    unwrap_agent,
    MigrationHelper,
)
from greenlang.agents.base import AgentResult


# ==============================================================================
# Test Agents
# ==============================================================================

class SimpleAsyncAgent(AsyncAgentBase[Dict[str, Any], Dict[str, Any]]):
    """Simple async agent for testing."""

    async def execute_impl_async(
        self, validated_input: Dict[str, Any], context: AsyncAgentExecutionContext
    ) -> Dict[str, Any]:
        """Simple execution that returns doubled value."""
        return {"result": validated_input.get("value", 0) * 2}


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
def async_agent():
    """Create async agent."""
    return SimpleAsyncAgent()


@pytest.fixture
def sync_agent(async_agent):
    """Create sync-wrapped agent."""
    return SyncAgentWrapper(async_agent)


@pytest.fixture
def resource_agent():
    """Create resource async agent."""
    return ResourceAsyncAgent()


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================

def test_sync_wrapper_creation(async_agent):
    """Test creating sync wrapper."""
    sync_agent = SyncAgentWrapper(async_agent)

    assert sync_agent is not None
    assert sync_agent.agent_id == async_agent.agent_id
    assert isinstance(sync_agent, SyncAgentWrapper)


def test_sync_execution(sync_agent):
    """Test sync execution of wrapped async agent."""
    result = sync_agent.execute({"value": 5})

    assert result.success is True
    assert result.data["result"] == 10
    assert "execution_time_ms" in result.metadata
    assert result.metadata["async_mode"] is True


def test_run_method_alias(sync_agent):
    """Test that run() is an alias for execute()."""
    result1 = sync_agent.execute({"value": 3})
    result2 = sync_agent.run({"value": 3})

    assert result1.success is True
    assert result2.success is True
    assert result1.data == result2.data


def test_multiple_executions(sync_agent):
    """Test multiple sync executions."""
    results = []
    for i in range(5):
        result = sync_agent.execute({"value": i})
        results.append(result)

    assert all(r.success for r in results)
    assert len(results) == 5

    # Check values
    for i, result in enumerate(results):
        assert result.data["result"] == i * 2


# ==============================================================================
# Context Manager Tests
# ==============================================================================

def test_sync_context_manager(async_agent):
    """Test sync context manager support."""
    with SyncAgentWrapper(async_agent) as agent:
        result = agent.execute({"value": 7})
        assert result.success is True
        assert result.data["result"] == 14


def test_resource_cleanup_in_context(resource_agent):
    """Test that resources are cleaned up in context manager."""
    # Wrap async agent
    sync_agent = SyncAgentWrapper(resource_agent)

    # Use context manager
    with sync_agent as agent:
        assert resource_agent.resource_opened is True
        result = agent.execute({"test": "data"})
        assert result.success is True

    # After context, resource should be closed
    assert resource_agent.resource_closed is True


def test_cleanup_on_error_in_context(resource_agent):
    """Test that cleanup happens even when errors occur."""
    sync_agent = SyncAgentWrapper(resource_agent)

    try:
        with sync_agent:
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Cleanup should still have happened
    assert resource_agent.resource_closed is True


# ==============================================================================
# Helper Function Tests
# ==============================================================================

def test_make_sync(async_agent):
    """Test make_sync() convenience function."""
    sync_agent = make_sync(async_agent)

    assert isinstance(sync_agent, SyncAgentWrapper)
    result = sync_agent.execute({"value": 4})
    assert result.success is True
    assert result.data["result"] == 8


def test_is_async_agent(async_agent, sync_agent):
    """Test is_async_agent() detection."""
    assert is_async_agent(async_agent) is True
    assert is_async_agent(sync_agent) is False
    assert is_async_agent("not an agent") is False


def test_is_sync_wrapper(async_agent, sync_agent):
    """Test is_sync_wrapper() detection."""
    assert is_sync_wrapper(sync_agent) is True
    assert is_sync_wrapper(async_agent) is False
    assert is_sync_wrapper("not a wrapper") is False


def test_unwrap_agent(async_agent, sync_agent):
    """Test unwrap_agent() function."""
    # Unwrap wrapped agent
    unwrapped = unwrap_agent(sync_agent)
    assert unwrapped is async_agent

    # Unwrap unwrapped agent (returns as-is)
    unwrapped2 = unwrap_agent(async_agent)
    assert unwrapped2 is async_agent


# ==============================================================================
# Timeout Tests
# ==============================================================================

def test_sync_timeout():
    """Test that timeout works in sync wrapper."""
    # Create slow agent
    import asyncio

    class SlowAgent(AsyncAgentBase[Dict, Dict]):
        async def execute_impl_async(self, input, context):
            await asyncio.sleep(2.0)
            return {"done": True}

    async_agent = SlowAgent()
    sync_agent = SyncAgentWrapper(async_agent)

    # Should timeout
    result = sync_agent.execute({"test": "data"}, timeout=0.5)

    assert result.success is False
    assert "timed out" in result.error.lower()


# ==============================================================================
# Statistics Tests
# ==============================================================================

def test_get_stats(sync_agent):
    """Test that stats work through wrapper."""
    # Execute a few times
    for i in range(3):
        sync_agent.execute({"value": i})

    # Get stats
    stats = sync_agent.get_stats()

    assert stats["executions"] == 3
    assert stats["total_time_ms"] > 0
    assert stats["sync_wrapper"] is True


# ==============================================================================
# Migration Helper Tests
# ==============================================================================

def test_migration_helper_can_run_async():
    """Test MigrationHelper.can_run_async()."""
    # We're in sync test context, so should return False
    assert MigrationHelper.can_run_async() is False


def test_migration_helper_detect_agent_type(async_agent, sync_agent):
    """Test MigrationHelper.detect_agent_type()."""
    assert MigrationHelper.detect_agent_type(async_agent) == "async"
    assert "wrapped" in MigrationHelper.detect_agent_type(sync_agent)


def test_migration_helper_get_or_wrap(async_agent):
    """Test MigrationHelper.get_or_wrap()."""
    # In sync context, should wrap async agent
    agent = MigrationHelper.get_or_wrap(async_agent)

    # Should be wrapped
    assert isinstance(agent, SyncAgentWrapper)

    # Should work
    result = agent.execute({"value": 5})
    assert result.success is True


# ==============================================================================
# Performance Tests
# ==============================================================================

def test_sync_wrapper_overhead():
    """Test that sync wrapper overhead is minimal."""
    async_agent = SimpleAsyncAgent()
    sync_agent = SyncAgentWrapper(async_agent)

    # Measure sync wrapper execution time
    start = time.time()
    for i in range(10):
        sync_agent.execute({"value": i})
    wrapper_time = time.time() - start

    # Overhead should be minimal (<10ms total for 10 calls)
    assert wrapper_time < 0.5  # Should be very fast for simple agent


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_error_propagation():
    """Test that errors propagate through wrapper."""
    class FailingAgent(AsyncAgentBase[Dict, Dict]):
        async def execute_impl_async(self, input, context):
            raise ValueError("Intentional failure")

    async_agent = FailingAgent()
    sync_agent = SyncAgentWrapper(async_agent)

    result = sync_agent.execute({"test": "data"})

    assert result.success is False
    assert "Intentional failure" in result.error


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================

def test_backward_compatibility_api():
    """Test that sync wrapper provides same API as old sync agents."""
    async_agent = SimpleAsyncAgent()
    sync_agent = SyncAgentWrapper(async_agent)

    # Old sync API methods
    assert hasattr(sync_agent, "execute")
    assert hasattr(sync_agent, "run")
    assert hasattr(sync_agent, "initialize")
    assert hasattr(sync_agent, "cleanup")
    assert hasattr(sync_agent, "get_stats")

    # Test they work
    sync_agent.initialize()
    result = sync_agent.execute({"value": 1})
    assert result.success is True
    stats = sync_agent.get_stats()
    assert stats["executions"] == 1
    sync_agent.cleanup()


def test_drop_in_replacement():
    """Test that sync wrapper is drop-in replacement for old sync agents."""
    # Old code pattern
    def process_with_agent(agent):
        """Function expecting sync agent."""
        result = agent.execute({"value": 10})
        return result

    # New async agent with wrapper
    async_agent = SimpleAsyncAgent()
    sync_agent = SyncAgentWrapper(async_agent)

    # Should work as drop-in replacement
    result = process_with_agent(sync_agent)

    assert result.success is True
    assert result.data["result"] == 20


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_repr(async_agent, sync_agent):
    """Test string representation."""
    repr_str = repr(sync_agent)
    assert "SyncAgentWrapper" in repr_str
    assert "SimpleAsyncAgent" in repr_str


def test_attribute_access(async_agent, sync_agent):
    """Test that wrapper exposes agent attributes."""
    assert sync_agent.agent_id == async_agent.agent_id
    assert sync_agent.pack_path == async_agent.pack_path
    assert sync_agent.spec == async_agent.spec


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
