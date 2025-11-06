"""Tests for AsyncOrchestrator.

This module tests async workflow execution with parallel step support.

Test Coverage:
- Basic async workflow execution
- Parallel execution of independent steps
- Sequential execution of dependent steps
- Mix of sync and async agents
- Error handling and retries
- Backward compatible sync wrapper
- Dependency analysis

Author: GreenLang Framework Team
Date: November 2025
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime

from greenlang.core.async_orchestrator import AsyncOrchestrator
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig


# Mock Agents for Testing

class MockSyncAgent(BaseAgent):
    """Mock synchronous agent for testing."""

    def __init__(self, agent_id: str, delay: float = 0.1):
        config = AgentConfig(
            agent_id=agent_id,
            name=f"Mock{agent_id}",
            version="1.0.0",
            description="Mock sync agent",
        )
        super().__init__(config)
        self.delay = delay
        self.call_count = 0

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute mock agent."""
        import time
        time.sleep(self.delay)  # Simulate work
        self.call_count += 1

        return AgentResult(
            success=True,
            data={
                "agent_id": self.config.agent_id,
                "input": input_data,
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count,
            },
            metadata={"agent": self.config.agent_id},
        )


try:
    from greenlang.agents.async_agent_base import AsyncAgentBase

    class MockAsyncAgent(AsyncAgentBase):
        """Mock asynchronous agent for testing."""

        def __init__(self, agent_id: str, delay: float = 0.1):
            from greenlang.config import get_config
            config = get_config()
            super().__init__(config, agent_id=agent_id, version="1.0.0")
            self.delay = delay
            self.call_count = 0

        async def execute_impl_async(
            self, input_data: Dict[str, Any], context
        ) -> AgentResult:
            """Execute mock async agent."""
            await asyncio.sleep(self.delay)  # Simulate async work
            self.call_count += 1

            return AgentResult(
                success=True,
                data={
                    "agent_id": self.agent_id,
                    "input": input_data,
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count,
                },
                metadata={"agent": self.agent_id},
            )

    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False
    MockAsyncAgent = None


# Tests

class TestAsyncOrchestrator:
    """Test suite for AsyncOrchestrator."""

    def test_basic_sync_workflow(self):
        """Test basic workflow with sync agents."""
        orchestrator = AsyncOrchestrator()

        # Register agents
        agent1 = MockSyncAgent("agent1", delay=0.05)
        agent2 = MockSyncAgent("agent2", delay=0.05)
        orchestrator.register_agent("agent1", agent1)
        orchestrator.register_agent("agent2", agent2)

        # Create simple workflow
        workflow = Workflow(
            name="test_workflow",
            description="Test workflow",
            steps=[
                WorkflowStep(
                    name="step1",
                    agent_id="agent1",
                    description="First step",
                ),
                WorkflowStep(
                    name="step2",
                    agent_id="agent2",
                    description="Second step",
                ),
            ],
        )
        orchestrator.register_workflow("test_workflow", workflow)

        # Execute
        result = orchestrator.execute_workflow("test_workflow", {"value": 42})

        # Verify
        assert result["success"] is True
        assert len(result["results"]) == 2
        assert "step1" in result["results"]
        assert "step2" in result["results"]

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_basic_async_workflow(self):
        """Test basic workflow with async agents."""
        orchestrator = AsyncOrchestrator()

        # Register async agents
        agent1 = MockAsyncAgent("agent1", delay=0.05)
        agent2 = MockAsyncAgent("agent2", delay=0.05)
        orchestrator.register_agent("agent1", agent1)
        orchestrator.register_agent("agent2", agent2)

        # Create workflow
        workflow = Workflow(
            name="test_async_workflow",
            description="Test async workflow",
            steps=[
                WorkflowStep(
                    name="step1",
                    agent_id="agent1",
                    description="First async step",
                ),
                WorkflowStep(
                    name="step2",
                    agent_id="agent2",
                    description="Second async step",
                ),
            ],
        )
        orchestrator.register_workflow("test_async_workflow", workflow)

        # Execute async
        result = await orchestrator.execute_workflow_async("test_async_workflow", {"value": 42})

        # Verify
        assert result["success"] is True
        assert len(result["results"]) == 2

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution of independent steps."""
        orchestrator = AsyncOrchestrator()

        # Register multiple async agents
        for i in range(5):
            agent = MockAsyncAgent(f"agent{i}", delay=0.1)
            orchestrator.register_agent(f"agent{i}", agent)

        # Create workflow with independent steps (no dependencies)
        workflow = Workflow(
            name="parallel_workflow",
            description="Parallel test workflow",
            steps=[
                WorkflowStep(
                    name=f"step{i}",
                    agent_id=f"agent{i}",
                    description=f"Step {i}",
                )
                for i in range(5)
            ],
        )
        orchestrator.register_workflow("parallel_workflow", workflow)

        # Execute and measure time
        start = asyncio.get_event_loop().time()
        result = await orchestrator.execute_workflow_async("parallel_workflow", {"value": 42})
        duration = asyncio.get_event_loop().time() - start

        # Verify
        assert result["success"] is True
        assert len(result["results"]) == 5

        # Parallel execution should be faster than sequential
        # 5 steps * 0.1s = 0.5s sequential
        # Should complete in ~0.15s parallel (allowing overhead)
        assert duration < 0.3, f"Expected parallel execution, took {duration:.2f}s"

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_dependent_steps(self):
        """Test sequential execution of dependent steps."""
        orchestrator = AsyncOrchestrator()

        # Register agents
        agent1 = MockAsyncAgent("agent1", delay=0.05)
        agent2 = MockAsyncAgent("agent2", delay=0.05)
        orchestrator.register_agent("agent1", agent1)
        orchestrator.register_agent("agent2", agent2)

        # Create workflow with dependencies (step2 depends on step1)
        workflow = Workflow(
            name="dependent_workflow",
            description="Dependent steps workflow",
            steps=[
                WorkflowStep(
                    name="step1",
                    agent_id="agent1",
                    description="First step",
                ),
                WorkflowStep(
                    name="step2",
                    agent_id="agent2",
                    description="Second step (depends on step1)",
                    input_mapping={
                        "previous_result": "results.step1.data"
                    },
                ),
            ],
        )
        orchestrator.register_workflow("dependent_workflow", workflow)

        # Execute
        result = await orchestrator.execute_workflow_async("dependent_workflow", {"value": 42})

        # Verify
        assert result["success"] is True
        assert len(result["results"]) == 2

        # Verify dependency was passed
        step2_result = result["results"]["step2"]
        assert "previous_result" in step2_result["data"]["input"]

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_mixed_sync_async_agents(self):
        """Test workflow with both sync and async agents."""
        orchestrator = AsyncOrchestrator()

        # Register mix of agents
        sync_agent = MockSyncAgent("sync_agent", delay=0.05)
        async_agent = MockAsyncAgent("async_agent", delay=0.05)
        orchestrator.register_agent("sync_agent", sync_agent)
        orchestrator.register_agent("async_agent", async_agent)

        # Create workflow
        workflow = Workflow(
            name="mixed_workflow",
            description="Mixed sync/async workflow",
            steps=[
                WorkflowStep(
                    name="sync_step",
                    agent_id="sync_agent",
                    description="Sync step",
                ),
                WorkflowStep(
                    name="async_step",
                    agent_id="async_agent",
                    description="Async step",
                ),
            ],
        )
        orchestrator.register_workflow("mixed_workflow", workflow)

        # Execute
        result = await orchestrator.execute_workflow_async("mixed_workflow", {"value": 42})

        # Verify both executed successfully
        assert result["success"] is True
        assert len(result["results"]) == 2

    def test_backward_compatible_sync_wrapper(self):
        """Test backward compatible sync execute_workflow()."""
        orchestrator = AsyncOrchestrator()

        # Register sync agent
        agent = MockSyncAgent("agent1", delay=0.05)
        orchestrator.register_agent("agent1", agent)

        # Create workflow
        workflow = Workflow(
            name="compat_workflow",
            description="Compatibility test",
            steps=[
                WorkflowStep(
                    name="step1",
                    agent_id="agent1",
                    description="Step 1",
                ),
            ],
        )
        orchestrator.register_workflow("compat_workflow", workflow)

        # Execute using sync wrapper (should work without asyncio.run)
        result = orchestrator.execute_workflow("compat_workflow", {"value": 42})

        # Verify
        assert result["success"] is True
        assert "step1" in result["results"]

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_dependency_analysis(self):
        """Test dependency analysis groups steps correctly."""
        orchestrator = AsyncOrchestrator()

        # Register agents
        for i in range(4):
            agent = MockAsyncAgent(f"agent{i}", delay=0.05)
            orchestrator.register_agent(f"agent{i}", agent)

        # Create complex workflow:
        # step0 (independent)
        # step1 (independent)
        # step2 (depends on step0)
        # step3 (depends on step1)
        workflow = Workflow(
            name="complex_workflow",
            description="Complex dependency workflow",
            steps=[
                WorkflowStep(name="step0", agent_id="agent0", description="Independent 1"),
                WorkflowStep(name="step1", agent_id="agent1", description="Independent 2"),
                WorkflowStep(
                    name="step2",
                    agent_id="agent2",
                    description="Depends on step0",
                    input_mapping={"prev": "results.step0.data"},
                ),
                WorkflowStep(
                    name="step3",
                    agent_id="agent3",
                    description="Depends on step1",
                    input_mapping={"prev": "results.step1.data"},
                ),
            ],
        )

        # Test dependency analysis
        groups = orchestrator._analyze_dependencies(workflow.steps)

        # Should have 2 groups:
        # Group 1: step0, step1 (independent, run in parallel)
        # Group 2: step2, step3 (depend on group 1, run in parallel)
        assert len(groups) == 2
        assert len(groups[0]) == 2  # step0 and step1
        assert len(groups[1]) == 2  # step2 and step3

    @pytest.mark.skipif(not ASYNC_AGENTS_AVAILABLE, reason="Async agents not available")
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in async workflows."""
        orchestrator = AsyncOrchestrator()

        # Create failing agent
        class FailingAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(
                    agent_id="failing",
                    name="FailingAgent",
                    version="1.0.0",
                )
                super().__init__(config)

            def execute(self, input_data: Dict[str, Any]) -> AgentResult:
                return AgentResult(
                    success=False,
                    error="Simulated failure",
                )

        failing_agent = FailingAgent()
        orchestrator.register_agent("failing", failing_agent)

        # Create workflow with skip on failure
        workflow = Workflow(
            name="error_workflow",
            description="Error handling test",
            steps=[
                WorkflowStep(
                    name="failing_step",
                    agent_id="failing",
                    description="This will fail",
                    on_failure="skip",  # Skip and continue
                ),
            ],
        )
        orchestrator.register_workflow("error_workflow", workflow)

        # Execute
        result = await orchestrator.execute_workflow_async("error_workflow", {"value": 42})

        # Should complete but with errors
        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert result["errors"][0]["step"] == "failing_step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
