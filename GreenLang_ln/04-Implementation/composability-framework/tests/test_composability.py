# -*- coding: utf-8 -*-
"""
Unit tests for GreenLang Composability Framework

This module contains comprehensive tests for the GLEL (GreenLang Expression Language)
framework, covering all major components and edge cases.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import hashlib
from typing import Dict, Any, List

from greenlang.core.composability import (
from greenlang.determinism import DeterministicClock
    BaseRunnable,
    AgentRunnable,
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RetryRunnable,
    FallbackRunnable,
    ZeroHallucinationWrapper,
    RunnableConfig,
    ExecutionContext,
    ProvenanceRecord,
    ExecutionMode,
    create_sequential_chain,
    create_parallel_chain,
    create_map_reduce_chain
)


# ============================================================================
# Mock Agents for Testing
# ============================================================================

class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "MockAgent", output: Dict[str, Any] = None):
        self.name = name
        self.output = output or {"result": "processed"}
        self.call_count = 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process method for testing."""
        self.call_count += 1
        return {**input_data, **self.output, "agent": self.name}


class AsyncMockAgent:
    """Async mock agent for testing."""

    def __init__(self, name: str = "AsyncMockAgent", output: Dict[str, Any] = None):
        self.name = name
        self.output = output or {"result": "async_processed"}
        self.call_count = 0

    async def aprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async process method for testing."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate async work
        return {**input_data, **self.output, "agent": self.name}


class FailingAgent:
    """Agent that always fails for testing error handling."""

    def __init__(self, error_message: str = "Agent failed"):
        self.error_message = error_message
        self.call_count = 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Always raises an exception."""
        self.call_count += 1
        raise RuntimeError(self.error_message)


class ConditionalFailingAgent:
    """Agent that fails conditionally for testing retry logic."""

    def __init__(self, fail_count: int = 2):
        self.fail_count = fail_count
        self.call_count = 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fails for first N attempts."""
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise RuntimeError(f"Failed on attempt {self.call_count}")
        return {**input_data, "attempts": self.call_count}


# ============================================================================
# Test ProvenanceRecord
# ============================================================================

class TestProvenanceRecord:
    """Tests for ProvenanceRecord class."""

    def test_provenance_creation(self):
        """Test creating a provenance record."""
        record = ProvenanceRecord(
            agent_id="test_agent",
            input_hash="input_hash_123",
            output_hash="output_hash_456",
            processing_time_ms=100.5
        )

        assert record.agent_id == "test_agent"
        assert record.input_hash == "input_hash_123"
        assert record.output_hash == "output_hash_456"
        assert record.processing_time_ms == 100.5
        assert record.parent_hash is None

    def test_chain_hash_calculation(self):
        """Test chain hash calculation."""
        record = ProvenanceRecord(
            agent_id="test_agent",
            input_hash="input_123",
            output_hash="output_456",
            processing_time_ms=100
        )

        chain_hash = record.calculate_chain_hash()
        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64  # SHA-256 produces 64 hex characters

        # Hash should be deterministic
        chain_hash2 = record.calculate_chain_hash()
        assert chain_hash == chain_hash2

    def test_provenance_with_parent(self):
        """Test provenance with parent hash."""
        parent = ProvenanceRecord(
            agent_id="parent_agent",
            input_hash="parent_input",
            output_hash="parent_output",
            processing_time_ms=50
        )

        child = ProvenanceRecord(
            agent_id="child_agent",
            input_hash="child_input",
            output_hash="child_output",
            processing_time_ms=75,
            parent_hash=parent.calculate_chain_hash()
        )

        assert child.parent_hash == parent.calculate_chain_hash()


# ============================================================================
# Test RunnableConfig
# ============================================================================

class TestRunnableConfig:
    """Tests for RunnableConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RunnableConfig()

        assert config.max_retries == 3
        assert config.retry_delay_ms == 1000
        assert config.timeout_seconds is None
        assert config.batch_size == 100
        assert config.enable_streaming is False
        assert config.enable_provenance is True
        assert config.parallel_workers == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RunnableConfig(
            max_retries=5,
            retry_delay_ms=2000,
            timeout_seconds=30.0,
            batch_size=50,
            enable_streaming=True,
            enable_provenance=False,
            parallel_workers=8
        )

        assert config.max_retries == 5
        assert config.retry_delay_ms == 2000
        assert config.timeout_seconds == 30.0
        assert config.batch_size == 50
        assert config.enable_streaming is True
        assert config.enable_provenance is False
        assert config.parallel_workers == 8


# ============================================================================
# Test ExecutionContext
# ============================================================================

class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_context_creation(self):
        """Test execution context creation."""
        config = RunnableConfig()
        context = ExecutionContext(config)

        assert context.config == config
        assert len(context.provenance_chain) == 0
        assert len(context.execution_id) == 16  # Truncated hash
        assert context.metrics == {}
        assert context.errors == []

    def test_add_provenance(self):
        """Test adding provenance records."""
        config = RunnableConfig(enable_provenance=True)
        context = ExecutionContext(config)

        record1 = ProvenanceRecord(
            agent_id="agent1",
            input_hash="input1",
            output_hash="output1",
            processing_time_ms=100
        )
        context.add_provenance(record1)

        assert len(context.provenance_chain) == 1
        assert context.provenance_chain[0] == record1

        record2 = ProvenanceRecord(
            agent_id="agent2",
            input_hash="input2",
            output_hash="output2",
            processing_time_ms=200
        )
        context.add_provenance(record2)

        assert len(context.provenance_chain) == 2
        assert context.provenance_chain[1].parent_hash == record1.calculate_chain_hash()

    def test_add_provenance_disabled(self):
        """Test that provenance is not added when disabled."""
        config = RunnableConfig(enable_provenance=False)
        context = ExecutionContext(config)

        record = ProvenanceRecord(
            agent_id="agent1",
            input_hash="input1",
            output_hash="output1",
            processing_time_ms=100
        )
        context.add_provenance(record)

        assert len(context.provenance_chain) == 0

    def test_get_total_time(self):
        """Test total time calculation."""
        config = RunnableConfig()
        context = ExecutionContext(config)

        # Wait a bit
        import time
        time.sleep(0.1)

        total_time = context.get_total_time_ms()
        assert total_time >= 100  # At least 100ms
        assert total_time < 200  # But less than 200ms


# ============================================================================
# Test AgentRunnable
# ============================================================================

class TestAgentRunnable:
    """Tests for AgentRunnable class."""

    def test_agent_runnable_creation(self):
        """Test creating an agent runnable."""
        agent = MockAgent("TestAgent")
        runnable = AgentRunnable(agent)

        assert runnable.name == "MockAgent"
        assert runnable.agent == agent

    def test_agent_runnable_invoke(self):
        """Test invoking an agent runnable."""
        agent = MockAgent("TestAgent", {"value": 42})
        runnable = AgentRunnable(agent)

        input_data = {"input": "test"}
        result = runnable.invoke(input_data)

        assert "input" in result
        assert result["value"] == 42
        assert result["agent"] == "TestAgent"
        assert agent.call_count == 1

    def test_agent_runnable_with_provenance(self):
        """Test agent runnable with provenance tracking."""
        agent = MockAgent("TestAgent", {"value": 42})
        runnable = AgentRunnable(agent)
        config = RunnableConfig(enable_provenance=True)

        input_data = {"input": "test"}
        result = runnable.invoke(input_data, config)

        assert "_provenance" in result
        provenance = result["_provenance"]
        assert provenance["agent_id"] == "MockAgent"
        assert "input_hash" in provenance
        assert "output_hash" in provenance
        assert "processing_time_ms" in provenance

    @pytest.mark.asyncio
    async def test_agent_runnable_async(self):
        """Test async invocation of agent runnable."""
        agent = AsyncMockAgent("AsyncTest", {"async_value": 99})
        runnable = AgentRunnable(agent)

        input_data = {"input": "async_test"}
        result = await runnable.ainvoke(input_data)

        assert "input" in result
        assert result["async_value"] == 99
        assert result["agent"] == "AsyncTest"
        assert agent.call_count == 1

    def test_agent_runnable_error_handling(self):
        """Test error handling in agent runnable."""
        agent = FailingAgent("Test error")
        runnable = AgentRunnable(agent)

        input_data = {"input": "test"}
        with pytest.raises(RuntimeError) as exc_info:
            runnable.invoke(input_data)

        assert "Test error" in str(exc_info.value)
        assert agent.call_count == 1


# ============================================================================
# Test RunnableSequence
# ============================================================================

class TestRunnableSequence:
    """Tests for RunnableSequence class."""

    def test_sequence_creation(self):
        """Test creating a sequence of runnables."""
        agent1 = AgentRunnable(MockAgent("Agent1"))
        agent2 = AgentRunnable(MockAgent("Agent2"))
        sequence = RunnableSequence([agent1, agent2])

        assert len(sequence.runnables) == 2
        assert "Agent1" in sequence.name
        assert "Agent2" in sequence.name

    def test_sequence_invoke(self):
        """Test invoking a sequence."""
        agent1 = MockAgent("Agent1", {"step1": "complete"})
        agent2 = MockAgent("Agent2", {"step2": "complete"})

        sequence = RunnableSequence([
            AgentRunnable(agent1),
            AgentRunnable(agent2)
        ])

        input_data = {"input": "test"}
        result = sequence.invoke(input_data)

        assert result["input"] == "test"
        assert result["step1"] == "complete"
        assert result["step2"] == "complete"
        assert result["agent"] == "Agent2"  # Last agent
        assert agent1.call_count == 1
        assert agent2.call_count == 1

    def test_sequence_pipe_operator(self):
        """Test creating sequence with pipe operator."""
        agent1 = AgentRunnable(MockAgent("Agent1"))
        agent2 = AgentRunnable(MockAgent("Agent2"))
        agent3 = AgentRunnable(MockAgent("Agent3"))

        # Test | operator
        sequence = agent1 | agent2 | agent3

        assert isinstance(sequence, RunnableSequence)
        assert len(sequence.runnables) == 3

        result = sequence.invoke({"input": "test"})
        assert result["agent"] == "Agent3"

    def test_sequence_with_provenance(self):
        """Test sequence with provenance tracking."""
        agent1 = AgentRunnable(MockAgent("Agent1"))
        agent2 = AgentRunnable(MockAgent("Agent2"))
        sequence = RunnableSequence([agent1, agent2])

        config = RunnableConfig(enable_provenance=True)
        result = sequence.invoke({"input": "test"}, config)

        assert "_chain_provenance" in result
        assert "_chain_hash" in result
        assert len(result["_chain_provenance"]) == 2

    @pytest.mark.asyncio
    async def test_sequence_async(self):
        """Test async sequence execution."""
        agent1 = AsyncMockAgent("AsyncAgent1", {"async1": True})
        agent2 = AsyncMockAgent("AsyncAgent2", {"async2": True})

        sequence = RunnableSequence([
            AgentRunnable(agent1),
            AgentRunnable(agent2)
        ])

        result = await sequence.ainvoke({"input": "async_test"})

        assert result["async1"] is True
        assert result["async2"] is True
        assert result["agent"] == "AsyncAgent2"

    @pytest.mark.asyncio
    async def test_sequence_streaming(self):
        """Test streaming from sequence."""
        agent1 = AgentRunnable(MockAgent("Agent1", {"step1": "done"}))
        agent2 = AgentRunnable(MockAgent("Agent2", {"step2": "done"}))

        sequence = RunnableSequence([agent1, agent2])

        chunks = []
        async for chunk in sequence.astream({"input": "stream"}):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["step"] == "MockAgent"
        assert chunks[1]["step"] == "MockAgent"


# ============================================================================
# Test RunnableParallel
# ============================================================================

class TestRunnableParallel:
    """Tests for RunnableParallel class."""

    def test_parallel_creation(self):
        """Test creating parallel runnable."""
        agent1 = AgentRunnable(MockAgent("Agent1"))
        agent2 = AgentRunnable(MockAgent("Agent2"))

        parallel = RunnableParallel({
            "branch1": agent1,
            "branch2": agent2
        })

        assert len(parallel.runnables) == 2
        assert "branch1" in parallel.name
        assert "branch2" in parallel.name

    def test_parallel_invoke(self):
        """Test parallel invocation."""
        agent1 = MockAgent("Agent1", {"value1": 10})
        agent2 = MockAgent("Agent2", {"value2": 20})

        parallel = RunnableParallel({
            "branch1": AgentRunnable(agent1),
            "branch2": AgentRunnable(agent2)
        })

        input_data = {"input": "test"}
        result = parallel.invoke(input_data)

        assert "branch1" in result
        assert "branch2" in result
        assert result["branch1"]["value1"] == 10
        assert result["branch2"]["value2"] == 20

    @pytest.mark.asyncio
    async def test_parallel_async(self):
        """Test async parallel execution."""
        agent1 = AsyncMockAgent("AsyncAgent1", {"async_value1": 100})
        agent2 = AsyncMockAgent("AsyncAgent2", {"async_value2": 200})

        parallel = RunnableParallel({
            "async1": AgentRunnable(agent1),
            "async2": AgentRunnable(agent2)
        })

        result = await parallel.ainvoke({"input": "async_parallel"})

        assert result["async1"]["async_value1"] == 100
        assert result["async2"]["async_value2"] == 200

    def test_parallel_with_errors(self):
        """Test parallel execution with branch failures."""
        agent1 = MockAgent("Agent1", {"success": True})
        agent2 = FailingAgent("Branch failed")

        parallel = RunnableParallel({
            "success_branch": AgentRunnable(agent1),
            "failing_branch": AgentRunnable(agent2)
        })

        result = parallel.invoke({"input": "test"})

        assert result["success_branch"]["success"] is True
        assert "error" in result["failing_branch"]
        assert "Branch failed" in result["failing_branch"]["error"]

    def test_parallel_with_provenance(self):
        """Test parallel execution with provenance."""
        parallel = RunnableParallel({
            "branch1": AgentRunnable(MockAgent("Agent1")),
            "branch2": AgentRunnable(MockAgent("Agent2"))
        })

        config = RunnableConfig(enable_provenance=True)
        result = parallel.invoke({"input": "test"}, config)

        assert "_parallel_provenance" in result
        assert "execution_id" in result["_parallel_provenance"]
        assert "total_time_ms" in result["_parallel_provenance"]
        assert result["_parallel_provenance"]["branches"] == ["branch1", "branch2"]


# ============================================================================
# Test RetryRunnable
# ============================================================================

class TestRetryRunnable:
    """Tests for RetryRunnable class."""

    def test_retry_success_first_attempt(self):
        """Test retry when first attempt succeeds."""
        agent = MockAgent("TestAgent", {"success": True})
        runnable = AgentRunnable(agent)
        retry_runnable = RetryRunnable(runnable, max_retries=3)

        result = retry_runnable.invoke({"input": "test"})

        assert result["success"] is True
        assert agent.call_count == 1

    def test_retry_after_failures(self):
        """Test retry after initial failures."""
        agent = ConditionalFailingAgent(fail_count=2)
        runnable = AgentRunnable(agent)
        retry_runnable = RetryRunnable(runnable, max_retries=3, delay_ms=10)

        result = retry_runnable.invoke({"input": "test"})

        assert result["attempts"] == 3
        assert agent.call_count == 3

    def test_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        agent = FailingAgent("Always fails")
        runnable = AgentRunnable(agent)
        retry_runnable = RetryRunnable(runnable, max_retries=2, delay_ms=10)

        with pytest.raises(RuntimeError) as exc_info:
            retry_runnable.invoke({"input": "test"})

        assert "Always fails" in str(exc_info.value)
        assert agent.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_async(self):
        """Test async retry logic."""
        agent = ConditionalFailingAgent(fail_count=1)
        runnable = AgentRunnable(agent)
        retry_runnable = RetryRunnable(runnable, max_retries=2, delay_ms=10)

        result = await retry_runnable.ainvoke({"input": "async_test"})

        assert result["attempts"] == 2
        assert agent.call_count == 2


# ============================================================================
# Test FallbackRunnable
# ============================================================================

class TestFallbackRunnable:
    """Tests for FallbackRunnable class."""

    def test_fallback_primary_succeeds(self):
        """Test fallback when primary succeeds."""
        primary = MockAgent("Primary", {"primary": True})
        fallback = MockAgent("Fallback", {"fallback": True})

        fallback_runnable = FallbackRunnable(
            AgentRunnable(primary),
            AgentRunnable(fallback)
        )

        result = fallback_runnable.invoke({"input": "test"})

        assert result["primary"] is True
        assert primary.call_count == 1
        assert fallback.call_count == 0

    def test_fallback_primary_fails(self):
        """Test fallback when primary fails."""
        primary = FailingAgent("Primary failed")
        fallback = MockAgent("Fallback", {"fallback": True})

        fallback_runnable = FallbackRunnable(
            AgentRunnable(primary),
            AgentRunnable(fallback)
        )

        result = fallback_runnable.invoke({"input": "test"})

        assert result["fallback"] is True
        assert primary.call_count == 1
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_async(self):
        """Test async fallback logic."""
        primary = FailingAgent("Primary async failed")
        fallback = AsyncMockAgent("AsyncFallback", {"async_fallback": True})

        fallback_runnable = FallbackRunnable(
            AgentRunnable(primary),
            AgentRunnable(fallback)
        )

        result = await fallback_runnable.ainvoke({"input": "async_test"})

        assert result["async_fallback"] is True


# ============================================================================
# Test RunnableLambda
# ============================================================================

class TestRunnableLambda:
    """Tests for RunnableLambda class."""

    def test_lambda_runnable(self):
        """Test lambda runnable with simple function."""

        def transform(data: Dict[str, Any]) -> Dict[str, Any]:
            return {**data, "transformed": True, "value_doubled": data.get("value", 0) * 2}

        lambda_runnable = RunnableLambda(transform)

        result = lambda_runnable.invoke({"value": 5})

        assert result["transformed"] is True
        assert result["value_doubled"] == 10

    def test_lambda_with_name(self):
        """Test lambda runnable with custom name."""

        def process(data):
            return {**data, "processed": True}

        lambda_runnable = RunnableLambda(process, name="CustomProcessor")

        assert lambda_runnable.name == "CustomProcessor"

    @pytest.mark.asyncio
    async def test_lambda_async(self):
        """Test async lambda runnable."""

        async def async_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.01)
            return {**data, "async_transformed": True}

        lambda_runnable = RunnableLambda(
            lambda x: x,  # Sync version (not used)
            afunc=async_transform
        )

        result = await lambda_runnable.ainvoke({"input": "async"})

        assert result["async_transformed"] is True

    def test_lambda_with_provenance(self):
        """Test lambda with provenance tracking."""

        def add_timestamp(data: Dict[str, Any]) -> Dict[str, Any]:
            return {**data, "timestamp": DeterministicClock.now().isoformat()}

        lambda_runnable = RunnableLambda(add_timestamp)
        config = RunnableConfig(enable_provenance=True)

        result = lambda_runnable.invoke({"input": "test"}, config)

        assert "_lambda_provenance" in result
        assert "timestamp" in result


# ============================================================================
# Test RunnableBranch
# ============================================================================

class TestRunnableBranch:
    """Tests for RunnableBranch class."""

    def test_branch_condition_matching(self):
        """Test branch with condition matching."""

        def is_large(data):
            return data.get("value", 0) > 100

        def is_medium(data):
            return 50 < data.get("value", 0) <= 100

        large_agent = MockAgent("LargeHandler", {"size": "large"})
        medium_agent = MockAgent("MediumHandler", {"size": "medium"})
        small_agent = MockAgent("SmallHandler", {"size": "small"})

        branch = RunnableBranch(
            branches=[
                (is_large, AgentRunnable(large_agent)),
                (is_medium, AgentRunnable(medium_agent))
            ],
            default=AgentRunnable(small_agent)
        )

        # Test large value
        result = branch.invoke({"value": 150})
        assert result["size"] == "large"

        # Test medium value
        result = branch.invoke({"value": 75})
        assert result["size"] == "medium"

        # Test small value (default)
        result = branch.invoke({"value": 25})
        assert result["size"] == "small"

    def test_branch_no_default_no_match(self):
        """Test branch without default when no conditions match."""

        def always_false(data):
            return False

        agent = MockAgent("Never", {"never": True})
        branch = RunnableBranch(
            branches=[(always_false, AgentRunnable(agent))],
            default=None
        )

        with pytest.raises(ValueError) as exc_info:
            branch.invoke({"input": "test"})

        assert "No branch matched" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_branch_async(self):
        """Test async branch execution."""

        def is_async(data):
            return data.get("mode") == "async"

        async_agent = AsyncMockAgent("AsyncBranch", {"async_result": True})
        sync_agent = MockAgent("SyncBranch", {"sync_result": True})

        branch = RunnableBranch(
            branches=[(is_async, AgentRunnable(async_agent))],
            default=AgentRunnable(sync_agent)
        )

        result = await branch.ainvoke({"mode": "async"})
        assert result["async_result"] is True


# ============================================================================
# Test ZeroHallucinationWrapper
# ============================================================================

class TestZeroHallucinationWrapper:
    """Tests for ZeroHallucinationWrapper class."""

    def test_zero_hallucination_valid_input(self):
        """Test zero-hallucination wrapper with valid input."""

        def validate_positive(data):
            return data.get("value", 0) > 0

        agent = MockAgent("SafeAgent", {"calculated": 100})
        runnable = AgentRunnable(agent)
        safe_runnable = ZeroHallucinationWrapper(
            runnable,
            validation_rules=[validate_positive]
        )

        result = safe_runnable.invoke({"value": 10})

        assert result["calculated"] == 100
        assert agent.call_count == 1

    def test_zero_hallucination_invalid_input(self):
        """Test zero-hallucination wrapper with invalid input."""

        def validate_positive(data):
            return data.get("value", 0) > 0

        agent = MockAgent("SafeAgent")
        runnable = AgentRunnable(agent)
        safe_runnable = ZeroHallucinationWrapper(
            runnable,
            validation_rules=[validate_positive]
        )

        with pytest.raises(ValueError) as exc_info:
            safe_runnable.invoke({"value": -5})

        assert "validation failed" in str(exc_info.value)
        assert agent.call_count == 0

    def test_zero_hallucination_output_validation(self):
        """Test output validation for zero-hallucination."""
        agent = MockAgent("CalcAgent", {
            "emissions": 1000.5,
            "_calculation_method": "IPCC_2021"
        })
        runnable = AgentRunnable(agent)
        safe_runnable = ZeroHallucinationWrapper(runnable)

        # Should not raise any warnings for properly documented calculation
        result = safe_runnable.invoke({"input": "test"})
        assert result["emissions"] == 1000.5

    @pytest.mark.asyncio
    async def test_zero_hallucination_async(self):
        """Test async zero-hallucination wrapper."""

        def validate_required_field(data):
            return "required_field" in data

        agent = AsyncMockAgent("AsyncSafe", {"safe": True})
        runnable = AgentRunnable(agent)
        safe_runnable = ZeroHallucinationWrapper(
            runnable,
            validation_rules=[validate_required_field]
        )

        result = await safe_runnable.ainvoke({"required_field": "present"})
        assert result["safe"] is True


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_sequential_chain(self):
        """Test creating sequential chain with utility function."""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        chain = create_sequential_chain(agent1, agent2)

        assert isinstance(chain, RunnableSequence)
        assert len(chain.runnables) == 2

        result = chain.invoke({"input": "test"})
        assert result["agent"] == "Agent2"

    def test_create_parallel_chain(self):
        """Test creating parallel chain with utility function."""
        agent1 = MockAgent("Agent1", {"val1": 1})
        agent2 = MockAgent("Agent2", {"val2": 2})

        chain = create_parallel_chain(
            first=agent1,
            second=agent2
        )

        assert isinstance(chain, RunnableParallel)
        assert "first" in chain.runnables
        assert "second" in chain.runnables

        result = chain.invoke({"input": "test"})
        assert result["first"]["val1"] == 1
        assert result["second"]["val2"] == 2

    @pytest.mark.asyncio
    async def test_create_map_reduce_chain(self):
        """Test creating map-reduce chain."""

        class MapAgent:
            def batch(self, inputs, config=None):
                return [{"mapped": i * 2} for i in inputs]

            async def abatch(self, inputs, config=None):
                return [{"mapped": i * 2} for i in inputs]

        class ReduceAgent:
            def invoke(self, inputs, config=None):
                total = sum(item["mapped"] for item in inputs)
                return {"reduced": total}

            async def ainvoke(self, inputs, config=None):
                total = sum(item["mapped"] for item in inputs)
                return {"reduced": total}

        mapper = MapAgent()
        reducer = ReduceAgent()

        chain = create_map_reduce_chain(mapper, reducer)

        result = await chain.ainvoke([1, 2, 3, 4, 5])
        assert result["reduced"] == 30  # (1+2+3+4+5) * 2 = 30


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_complex_pipeline(self):
        """Test a complex pipeline with multiple features."""
        # Create agents
        intake = MockAgent("Intake", {"status": "received"})
        validation = MockAgent("Validation", {"validated": True})
        calculation = MockAgent("Calculation", {"emissions": 1000})
        reporting = MockAgent("Reporting", {"report": "generated"})

        # Build complex pipeline
        pipeline = (
            AgentRunnable(intake).with_retry(max_retries=2) |
            AgentRunnable(validation) |
            AgentRunnable(calculation) |
            AgentRunnable(reporting)
        )

        config = RunnableConfig(enable_provenance=True)
        result = pipeline.invoke({"input": "complex"}, config)

        assert result["status"] == "received"
        assert result["validated"] is True
        assert result["emissions"] == 1000
        assert result["report"] == "generated"
        assert "_chain_provenance" in result

    @pytest.mark.asyncio
    async def test_mixed_sync_async_chain(self):
        """Test chain with mixed sync and async agents."""
        sync_agent = MockAgent("SyncAgent", {"sync": True})
        async_agent = AsyncMockAgent("AsyncAgent", {"async": True})

        chain = (
            AgentRunnable(sync_agent) |
            AgentRunnable(async_agent)
        )

        result = await chain.ainvoke({"input": "mixed"})

        assert result["sync"] is True
        assert result["async"] is True

    def test_nested_sequences(self):
        """Test nested sequence composition."""
        agent1 = MockAgent("Agent1", {"step1": 1})
        agent2 = MockAgent("Agent2", {"step2": 2})
        agent3 = MockAgent("Agent3", {"step3": 3})
        agent4 = MockAgent("Agent4", {"step4": 4})

        # Create sub-sequences
        seq1 = AgentRunnable(agent1) | AgentRunnable(agent2)
        seq2 = AgentRunnable(agent3) | AgentRunnable(agent4)

        # Combine sequences
        full_chain = seq1 | seq2

        result = full_chain.invoke({"input": "nested"})

        assert result["step1"] == 1
        assert result["step2"] == 2
        assert result["step3"] == 3
        assert result["step4"] == 4

    def test_error_propagation_in_chain(self):
        """Test that errors properly propagate through chains."""
        agent1 = MockAgent("Agent1")
        failing_agent = FailingAgent("Chain failure")
        agent3 = MockAgent("Agent3")  # Should not be reached

        chain = (
            AgentRunnable(agent1) |
            AgentRunnable(failing_agent) |
            AgentRunnable(agent3)
        )

        with pytest.raises(RuntimeError) as exc_info:
            chain.invoke({"input": "test"})

        assert "Chain failure" in str(exc_info.value)
        assert agent1.call_count == 1
        assert failing_agent.call_count == 1
        assert agent3.call_count == 0

    @pytest.mark.asyncio
    async def test_streaming_with_parallel(self):
        """Test streaming from parallel branches."""
        agent1 = MockAgent("Agent1", {"stream1": 1})
        agent2 = MockAgent("Agent2", {"stream2": 2})

        parallel = RunnableParallel({
            "branch1": AgentRunnable(agent1),
            "branch2": AgentRunnable(agent2)
        })

        chunks = []
        async for chunk in parallel.astream({"input": "stream"}):
            chunks.append(chunk)

        # Should receive chunks from both branches
        assert any("branch1" in chunk for chunk in chunks)
        assert any("branch2" in chunk for chunk in chunks)


# ============================================================================
# Test Performance and Edge Cases
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Tests for performance and edge cases."""

    def test_batch_processing(self):
        """Test batch processing functionality."""
        agent = MockAgent("BatchAgent", {"batched": True})
        runnable = AgentRunnable(agent)

        inputs = [{"id": i} for i in range(10)]
        config = RunnableConfig(batch_size=3)

        results = runnable.batch(inputs, config)

        assert len(results) == 10
        assert all(r["batched"] is True for r in results)
        assert agent.call_count == 10

    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing."""
        agent = AsyncMockAgent("AsyncBatchAgent", {"async_batched": True})
        runnable = AgentRunnable(agent)

        inputs = [{"id": i} for i in range(5)]
        config = RunnableConfig(batch_size=2)

        results = await runnable.abatch(inputs, config)

        assert len(results) == 5
        assert all(r["async_batched"] is True for r in results)

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        agent = MockAgent("EmptyHandler")
        runnable = AgentRunnable(agent)

        result = runnable.invoke({})

        assert result["result"] == "processed"
        assert result["agent"] == "EmptyHandler"

    def test_large_chain_composition(self):
        """Test composing a large chain."""
        agents = [MockAgent(f"Agent{i}", {f"step{i}": i}) for i in range(20)]
        runnables = [AgentRunnable(agent) for agent in agents]

        chain = runnables[0]
        for runnable in runnables[1:]:
            chain = chain | runnable

        result = chain.invoke({"input": "large"})

        # Check that all steps were executed
        for i in range(20):
            assert f"step{i}" in result
            assert result[f"step{i}"] == i

    def test_config_inheritance(self):
        """Test that config is properly inherited through chains."""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        chain = (
            AgentRunnable(agent1) |
            AgentRunnable(agent2)
        )

        config = RunnableConfig(
            enable_provenance=True,
            max_retries=5,
            metadata={"test": "value"}
        )

        result = chain.invoke({"input": "config_test"}, config)

        assert "_chain_provenance" in result
        # Config should be passed through the chain

    def test_with_config_method(self):
        """Test the with_config method."""
        agent = MockAgent("ConfigAgent")
        runnable = AgentRunnable(agent)

        new_runnable = runnable.with_config(
            max_retries=10,
            batch_size=50
        )

        assert new_runnable._config.max_retries == 10
        assert new_runnable._config.batch_size == 50
        # Original should be unchanged
        assert runnable._config.max_retries == 3

    def test_pipe_method(self):
        """Test the explicit pipe method."""
        agent1 = MockAgent("Agent1", {"v1": 1})
        agent2 = MockAgent("Agent2", {"v2": 2})
        agent3 = MockAgent("Agent3", {"v3": 3})

        r1 = AgentRunnable(agent1)
        r2 = AgentRunnable(agent2)
        r3 = AgentRunnable(agent3)

        chain = r1.pipe(r2, r3)

        result = chain.invoke({"input": "pipe"})

        assert result["v1"] == 1
        assert result["v2"] == 2
        assert result["v3"] == 3


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])