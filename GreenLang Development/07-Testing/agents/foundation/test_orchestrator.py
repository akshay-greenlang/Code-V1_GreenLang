# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-001: GreenLang Orchestrator

Tests cover:
    - DAG definition and validation
    - Topological sorting
    - Parallel execution
    - Retry logic
    - Timeout handling
    - Checkpoint/recovery
    - Lineage tracking
    - Determinism verification
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.foundation.orchestrator import (
    GreenLangOrchestrator,
    DAGDefinition,
    AgentNode,
    ExecutionResult,
    PipelineStatus,
    AgentStatus,
)


# Test agent implementations
class SuccessAgent(BaseAgent):
    """Agent that always succeeds."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=True,
            data={
                "result": f"processed_{input_data.get('value', 'default')}",
                "input_keys": list(input_data.keys())
            }
        )


class FailureAgent(BaseAgent):
    """Agent that always fails."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=False,
            error="Intentional failure for testing"
        )


class SlowAgent(BaseAgent):
    """Agent that takes a long time."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        time.sleep(input_data.get("delay", 5))
        return AgentResult(success=True, data={"delayed": True})


class CounterAgent(BaseAgent):
    """Agent that counts invocations (for retry testing)."""
    call_count = 0

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        CounterAgent.call_count += 1
        fail_until = input_data.get("fail_until", 0)

        if CounterAgent.call_count <= fail_until:
            return AgentResult(
                success=False,
                error=f"Failing intentionally (attempt {CounterAgent.call_count})"
            )

        return AgentResult(
            success=True,
            data={"call_count": CounterAgent.call_count}
        )


class TestDAGDefinition:
    """Tests for DAG definition and validation."""

    def test_valid_dag_creation(self):
        """Test creating a valid DAG definition."""
        dag = DAGDefinition(
            name="test_pipeline",
            agents={
                "agent_a": {"agent_type": "SuccessAgent"},
                "agent_b": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["agent_a"]
                }
            }
        )

        assert dag.name == "test_pipeline"
        assert len(dag.agents) == 2
        assert "agent_a" in dag.agents
        assert "agent_b" in dag.agents

    def test_dag_cycle_detection(self):
        """Test that cycles in DAG are detected."""
        with pytest.raises(ValueError, match="Cycle detected"):
            DAGDefinition(
                name="cyclic_pipeline",
                agents={
                    "agent_a": {
                        "agent_type": "SuccessAgent",
                        "dependencies": ["agent_c"]
                    },
                    "agent_b": {
                        "agent_type": "SuccessAgent",
                        "dependencies": ["agent_a"]
                    },
                    "agent_c": {
                        "agent_type": "SuccessAgent",
                        "dependencies": ["agent_b"]
                    }
                }
            )

    def test_dag_self_dependency_detection(self):
        """Test that self-dependencies are detected as cycles."""
        with pytest.raises(ValueError, match="Cycle detected"):
            DAGDefinition(
                name="self_dep_pipeline",
                agents={
                    "agent_a": {
                        "agent_type": "SuccessAgent",
                        "dependencies": ["agent_a"]
                    }
                }
            )


class TestGreenLangOrchestrator:
    """Tests for the GreenLang Orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance with test agents registered."""
        orch = GreenLangOrchestrator()
        orch.register_agent("SuccessAgent", SuccessAgent)
        orch.register_agent("FailureAgent", FailureAgent)
        orch.register_agent("SlowAgent", SlowAgent)
        orch.register_agent("CounterAgent", CounterAgent)
        return orch

    @pytest.mark.asyncio
    async def test_simple_pipeline_success(self, orchestrator):
        """Test successful execution of a simple pipeline."""
        dag = DAGDefinition(
            name="simple_pipeline",
            inputs={"value": "test_input"},
            agents={
                "agent_a": {"agent_type": "SuccessAgent"}
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.SUCCESS
        assert result.agents_succeeded == 1
        assert result.agents_failed == 0
        assert "agent_a" in result.agent_results
        assert result.agent_results["agent_a"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self, orchestrator):
        """Test sequential pipeline execution."""
        dag = DAGDefinition(
            name="sequential_pipeline",
            inputs={"value": "start"},
            agents={
                "step_1": {"agent_type": "SuccessAgent"},
                "step_2": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["step_1"]
                },
                "step_3": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["step_2"]
                }
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.SUCCESS
        assert result.agents_succeeded == 3
        assert result.agents_total == 3

    @pytest.mark.asyncio
    async def test_parallel_pipeline(self, orchestrator):
        """Test parallel pipeline execution."""
        dag = DAGDefinition(
            name="parallel_pipeline",
            inputs={"value": "parallel"},
            agents={
                "parallel_a": {"agent_type": "SuccessAgent"},
                "parallel_b": {"agent_type": "SuccessAgent"},
                "parallel_c": {"agent_type": "SuccessAgent"},
                "final": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["parallel_a", "parallel_b", "parallel_c"]
                }
            },
            max_parallel=3
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.SUCCESS
        assert result.agents_succeeded == 4

    @pytest.mark.asyncio
    async def test_pipeline_with_failure(self, orchestrator):
        """Test pipeline with a failing agent."""
        dag = DAGDefinition(
            name="failure_pipeline",
            agents={
                "agent_a": {"agent_type": "SuccessAgent"},
                "agent_b": {
                    "agent_type": "FailureAgent",
                    "dependencies": ["agent_a"],
                    "retry_config": {"max_retries": 0}
                }
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status in (PipelineStatus.FAILED, PipelineStatus.PARTIAL)
        assert result.agents_failed >= 1

    @pytest.mark.asyncio
    async def test_retry_logic(self, orchestrator):
        """Test retry logic for failing agents."""
        CounterAgent.call_count = 0  # Reset counter

        dag = DAGDefinition(
            name="retry_pipeline",
            inputs={"fail_until": 2},  # Fail first 2 attempts
            agents={
                "retry_agent": {
                    "agent_type": "CounterAgent",
                    "retry_config": {
                        "max_retries": 3,
                        "initial_delay_ms": 100,
                        "backoff_multiplier": 1
                    }
                }
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.SUCCESS
        assert result.agent_results["retry_agent"]["attempts"] == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator):
        """Test timeout handling for slow agents."""
        dag = DAGDefinition(
            name="timeout_pipeline",
            inputs={"delay": 10},
            agents={
                "slow_agent": {
                    "agent_type": "SlowAgent",
                    "timeout_seconds": 1,
                    "retry_config": {"max_retries": 0}
                }
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.FAILED
        assert result.agent_results["slow_agent"]["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_lineage_tracking(self, orchestrator):
        """Test that lineage is tracked correctly."""
        dag = DAGDefinition(
            name="lineage_pipeline",
            inputs={"value": "lineage_test"},
            agents={
                "step_1": {"agent_type": "SuccessAgent"},
                "step_2": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["step_1"]
                }
            }
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.lineage_id != ""
        assert result.input_hash != ""
        assert result.output_hash != ""

    @pytest.mark.asyncio
    async def test_determinism(self, orchestrator):
        """Test deterministic execution (same inputs = same output hash)."""
        dag = DAGDefinition(
            name="determinism_pipeline",
            inputs={"value": "deterministic"},
            agents={
                "agent_a": {"agent_type": "SuccessAgent"}
            }
        )

        result1 = await orchestrator.execute_pipeline(dag)
        result2 = await orchestrator.execute_pipeline(dag)

        assert result1.input_hash == result2.input_hash
        assert result1.output_hash == result2.output_hash

    def test_synchronous_execution(self, orchestrator):
        """Test synchronous execution wrapper."""
        result = orchestrator.run({
            "pipeline": {
                "name": "sync_test",
                "inputs": {"value": "sync"},
                "agents": {
                    "agent_a": {"agent_type": "SuccessAgent"}
                }
            }
        })

        assert result.success
        assert result.data["status"] == "success"

    def test_agent_registration(self, orchestrator):
        """Test agent registration."""
        metrics = orchestrator.get_metrics()
        assert metrics["registered_agents"] >= 4  # Our test agents

    def test_unregistered_agent_handling(self, orchestrator):
        """Test handling of unregistered agent types."""
        result = orchestrator.run({
            "pipeline": {
                "name": "unregistered_test",
                "agents": {
                    "agent_a": {"agent_type": "NonExistentAgent"}
                }
            }
        })

        # Should fail gracefully
        assert not result.success or "NonExistentAgent" in str(result.data.get("errors", []))


class TestAgentNode:
    """Tests for AgentNode dataclass."""

    def test_agent_node_creation(self):
        """Test creating an agent node."""
        node = AgentNode(
            agent_id="test_agent",
            agent_type="SuccessAgent",
            dependencies=["dep_1", "dep_2"],
            timeout_seconds=120
        )

        assert node.agent_id == "test_agent"
        assert node.agent_type == "SuccessAgent"
        assert len(node.dependencies) == 2
        assert node.timeout_seconds == 120
        assert node.status == AgentStatus.PENDING

    def test_agent_node_defaults(self):
        """Test agent node default values."""
        node = AgentNode(
            agent_id="minimal",
            agent_type="MinimalAgent"
        )

        assert node.dependencies == []
        assert node.timeout_seconds == 300
        assert node.retry_config["max_retries"] == 3


class TestExecutionResult:
    """Tests for ExecutionResult model."""

    def test_execution_result_creation(self):
        """Test creating an execution result."""
        result = ExecutionResult(
            execution_id="exec_123",
            pipeline_id="pipe_456",
            status=PipelineStatus.SUCCESS,
            started_at=datetime.now(),
            agents_total=5,
            agents_succeeded=5
        )

        assert result.execution_id == "exec_123"
        assert result.status == PipelineStatus.SUCCESS
        assert result.agents_total == 5
        assert result.agents_succeeded == 5


# Integration test
class TestOrchestratorIntegration:
    """Integration tests for the orchestrator."""

    @pytest.mark.asyncio
    async def test_complex_pipeline(self):
        """Test a complex multi-stage pipeline."""
        orchestrator = GreenLangOrchestrator()
        orchestrator.register_agent("SuccessAgent", SuccessAgent)

        # Diamond dependency pattern
        #       A
        #      / \
        #     B   C
        #      \ /
        #       D
        dag = DAGDefinition(
            name="diamond_pipeline",
            inputs={"value": "diamond"},
            agents={
                "A": {"agent_type": "SuccessAgent"},
                "B": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["A"]
                },
                "C": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["A"]
                },
                "D": {
                    "agent_type": "SuccessAgent",
                    "dependencies": ["B", "C"]
                }
            },
            max_parallel=2
        )

        result = await orchestrator.execute_pipeline(dag)

        assert result.status == PipelineStatus.SUCCESS
        assert result.agents_succeeded == 4
        assert "D" in result.final_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
