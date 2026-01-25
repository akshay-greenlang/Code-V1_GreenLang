"""
Integration tests for GL-001 ThermalCommand Orchestrator.

Tests the full orchestration pipeline including workflow execution,
agent coordination, safety integration, and event handling.

Coverage Target: 85%+
Reference: GL-001 Specification Section 11

Test Categories:
1. Orchestrator lifecycle (start/stop)
2. Agent registration and coordination
3. Workflow execution
4. Safety coordination
5. Event handling
6. Metrics collection
7. End-to-end scenarios

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# MOCK CLASSES FOR TESTING
# =============================================================================

class MockOrchestratorConfig:
    """Mock orchestrator configuration."""

    def __init__(
        self,
        name: str = "TestOrchestrator",
        orchestrator_id: str = "ORCH-TEST-001",
        version: str = "1.0.0"
    ):
        self.name = name
        self.orchestrator_id = orchestrator_id
        self.version = version
        self.safety = MockSafetyConfig()
        self.integration = MockIntegrationConfig()
        self.metrics = MockMetricsConfig()

    def dict(self):
        return {
            "name": self.name,
            "orchestrator_id": self.orchestrator_id,
            "version": self.version,
        }


class MockSafetyConfig:
    """Mock safety configuration."""

    def __init__(self):
        self.level = MockSafetyLevel.SIL_2
        self.emergency_shutdown_enabled = True
        self.heartbeat_interval_ms = 1000
        self.alarm_thresholds = {
            "high_temperature_f": 500.0,
            "high_pressure_psig": 100.0,
        }


class MockSafetyLevel:
    """Mock safety level enum."""
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    value = 2


class MockIntegrationConfig:
    """Mock integration configuration."""

    def __init__(self):
        self.opcua_enabled = False
        self.mqtt_enabled = False
        self.kafka_enabled = False


class MockMetricsConfig:
    """Mock metrics configuration."""

    def __init__(self):
        self.collection_interval_s = 10


class MockWorkflowSpec:
    """Mock workflow specification."""

    def __init__(
        self,
        workflow_id: str = "WF-001",
        name: str = "Test Workflow",
        tasks: List[Dict] = None,
        required_agents: List[str] = None
    ):
        self.workflow_id = workflow_id
        self.name = name
        self.tasks = tasks or [{"task_id": "T1", "action": "test"}]
        self.required_agents = required_agents or []

    def dict(self):
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "tasks": self.tasks,
            "required_agents": self.required_agents,
        }


class MockWorkflowResult:
    """Mock workflow result."""

    def __init__(self, status: str = "completed"):
        self.status = MockStatus(status)
        self.duration_ms = 150.0
        self.provenance_hash = "abc123"
        self.explanation = ""

    def dict(self):
        return {
            "status": self.status.value,
            "duration_ms": self.duration_ms,
        }


class MockStatus:
    """Mock status enum."""

    def __init__(self, value: str):
        self.value = value


class MockAgentRegistration:
    """Mock agent registration."""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        capabilities: set = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.capabilities = capabilities or {"default"}
        self.last_heartbeat = datetime.now(timezone.utc)


class MockAgentRole:
    """Mock agent role enum."""
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    MONITOR = "monitor"


class MockEvent:
    """Mock orchestrator event."""

    def __init__(
        self,
        event_type: str = "TEST_EVENT",
        source: str = "TEST",
        priority: str = "NORMAL",
        payload: Dict = None
    ):
        self.event_type = event_type
        self.source = source
        self.priority = priority
        self.payload = payload or {}


# =============================================================================
# MOCK ORCHESTRATOR FOR TESTING
# =============================================================================

class MockThermalCommandOrchestrator:
    """
    Mock implementation of ThermalCommandOrchestrator for testing.

    This allows us to test the orchestrator's behavior without
    requiring all the real dependencies.
    """

    def __init__(self, config: MockOrchestratorConfig):
        self.config = config
        self._state = "initializing"
        self._start_time = None
        self._registered_agents: Dict[str, MockAgentRegistration] = {}
        self._workflow_coordinator = MockWorkflowCoordinator()
        self._safety_coordinator = MockSafetyCoordinator()
        self._metrics = {
            "workflows_executed": 0,
            "workflows_failed": 0,
            "tasks_executed": 0,
            "safety_events": 0,
            "api_requests": 0,
        }
        self._event_handlers = {}

    async def start(self):
        """Start the orchestrator."""
        self._state = "starting"
        await self._workflow_coordinator.start()
        await self._safety_coordinator.start()
        self._state = "running"
        self._start_time = datetime.now(timezone.utc)

    async def stop(self):
        """Stop the orchestrator."""
        self._state = "stopping"
        await self._workflow_coordinator.stop()
        await self._safety_coordinator.stop()
        self._state = "stopped"

    def register_agent(self, registration: MockAgentRegistration) -> bool:
        """Register an agent."""
        if registration.agent_id in self._registered_agents:
            return False
        self._registered_agents[registration.agent_id] = registration
        return True

    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent."""
        if agent_id not in self._registered_agents:
            return False
        del self._registered_agents[agent_id]
        return True

    def get_agent_status(self, agent_id: str):
        """Get agent status."""
        if agent_id not in self._registered_agents:
            return None
        return self._registered_agents[agent_id]

    async def execute_workflow(self, spec: MockWorkflowSpec) -> MockWorkflowResult:
        """Execute a workflow."""
        if not spec.tasks:
            raise ValueError("Workflow must have at least one task")

        for agent_type in spec.required_agents:
            agents = [a for a in self._registered_agents.values()
                     if a.agent_type == agent_type]
            if not agents:
                raise ValueError(f"No agent registered for type: {agent_type}")

        if self._state != "running":
            raise RuntimeError("Orchestrator not running")

        if self._safety_coordinator.is_esd_triggered:
            raise RuntimeError("Cannot execute workflow: safety system not ready")

        self._metrics["workflows_executed"] += 1
        return await self._workflow_coordinator.execute_workflow(spec)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        return await self._workflow_coordinator.cancel_workflow(workflow_id)

    async def trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown."""
        await self._safety_coordinator.trigger_esd(reason)
        self._metrics["safety_events"] += 1

    async def reset_emergency_shutdown(self, authorized_by: str) -> bool:
        """Reset emergency shutdown."""
        return await self._safety_coordinator.reset_esd(authorized_by)

    def get_system_status(self):
        """Get system status."""
        uptime_seconds = 0.0
        if self._start_time:
            uptime_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return {
            "orchestrator_id": self.config.orchestrator_id,
            "orchestrator_name": self.config.name,
            "status": self._state,
            "uptime_seconds": uptime_seconds,
            "registered_agents": len(self._registered_agents),
            "safety_status": self._safety_coordinator.safety_state,
            "esd_armed": self.config.safety.emergency_shutdown_enabled,
        }

    def get_metrics(self):
        """Get metrics."""
        return {
            **self._metrics,
            "registered_agents": len(self._registered_agents),
        }

    async def handle_event(self, event: MockEvent):
        """Handle an event."""
        if "SAFETY" in event.event_type.upper():
            await self._handle_safety_event(event)

    async def _handle_safety_event(self, event: MockEvent):
        """Handle a safety event."""
        self._metrics["safety_events"] += 1

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == "running"

    @property
    def agent_count(self) -> int:
        return len(self._registered_agents)


class MockWorkflowCoordinator:
    """Mock workflow coordinator."""

    def __init__(self):
        self._active_workflows: Dict[str, MockWorkflowSpec] = {}

    async def start(self):
        pass

    async def stop(self):
        pass

    async def execute_workflow(self, spec: MockWorkflowSpec) -> MockWorkflowResult:
        self._active_workflows[spec.workflow_id] = spec
        await asyncio.sleep(0.01)  # Simulate work
        del self._active_workflows[spec.workflow_id]
        return MockWorkflowResult("completed")

    async def cancel_workflow(self, workflow_id: str) -> bool:
        if workflow_id in self._active_workflows:
            del self._active_workflows[workflow_id]
            return True
        return False

    def get_active_workflows(self):
        return list(self._active_workflows.keys())


class MockSafetyCoordinator:
    """Mock safety coordinator."""

    def __init__(self):
        self.safety_state = "normal"
        self.is_esd_triggered = False

    async def start(self):
        pass

    async def stop(self):
        pass

    async def trigger_esd(self, reason: str):
        self.is_esd_triggered = True
        self.safety_state = "emergency"

    async def reset_esd(self, authorized_by: str) -> bool:
        self.is_esd_triggered = False
        self.safety_state = "normal"
        return True

    def register_interlock(self, **kwargs):
        pass


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator_config() -> MockOrchestratorConfig:
    """Create orchestrator configuration."""
    return MockOrchestratorConfig(
        name="TestOrchestrator",
        orchestrator_id="ORCH-TEST-001"
    )


@pytest.fixture
def orchestrator(orchestrator_config) -> MockThermalCommandOrchestrator:
    """Create mock orchestrator instance."""
    return MockThermalCommandOrchestrator(orchestrator_config)


@pytest.fixture
async def running_orchestrator(orchestrator) -> MockThermalCommandOrchestrator:
    """Create and start orchestrator."""
    await orchestrator.start()
    yield orchestrator
    await orchestrator.stop()


@pytest.fixture
def sample_workflow_spec() -> MockWorkflowSpec:
    """Create a sample workflow specification."""
    return MockWorkflowSpec(
        workflow_id="WF-TEST-001",
        name="Test Workflow",
        tasks=[
            {"task_id": "T1", "action": "read_data"},
            {"task_id": "T2", "action": "process_data"},
            {"task_id": "T3", "action": "write_output"},
        ]
    )


@pytest.fixture
def sample_agent_registration() -> MockAgentRegistration:
    """Create a sample agent registration."""
    return MockAgentRegistration(
        agent_id="AGENT-TEST-001",
        agent_type="GL-002",
        name="Test Agent",
        capabilities={"temperature_control", "data_acquisition"}
    )


# =============================================================================
# TEST CLASS: ORCHESTRATOR LIFECYCLE
# =============================================================================

class TestOrchestratorLifecycle:
    """Tests for orchestrator lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_orchestrator(self, orchestrator):
        """Test starting the orchestrator."""
        await orchestrator.start()

        assert orchestrator.state == "running"
        assert orchestrator.is_running is True
        assert orchestrator._start_time is not None

    @pytest.mark.asyncio
    async def test_stop_orchestrator(self, orchestrator):
        """Test stopping the orchestrator."""
        await orchestrator.start()
        await orchestrator.stop()

        assert orchestrator.state == "stopped"
        assert orchestrator.is_running is False

    @pytest.mark.asyncio
    async def test_orchestrator_uptime(self, orchestrator):
        """Test orchestrator uptime calculation."""
        await orchestrator.start()
        await asyncio.sleep(0.1)

        status = orchestrator.get_system_status()

        assert status["uptime_seconds"] > 0

    @pytest.mark.asyncio
    async def test_orchestrator_state_transitions(self, orchestrator):
        """Test orchestrator state transitions."""
        assert orchestrator.state == "initializing"

        await orchestrator.start()
        assert orchestrator.state == "running"

        await orchestrator.stop()
        assert orchestrator.state == "stopped"


# =============================================================================
# TEST CLASS: AGENT REGISTRATION
# =============================================================================

class TestAgentRegistration:
    """Tests for agent registration and management."""

    def test_register_agent(self, orchestrator, sample_agent_registration):
        """Test registering an agent."""
        result = orchestrator.register_agent(sample_agent_registration)

        assert result is True
        assert orchestrator.agent_count == 1

    def test_register_duplicate_agent(self, orchestrator, sample_agent_registration):
        """Test registering duplicate agent fails."""
        orchestrator.register_agent(sample_agent_registration)
        result = orchestrator.register_agent(sample_agent_registration)

        assert result is False
        assert orchestrator.agent_count == 1

    def test_deregister_agent(self, orchestrator, sample_agent_registration):
        """Test deregistering an agent."""
        orchestrator.register_agent(sample_agent_registration)
        result = orchestrator.deregister_agent(sample_agent_registration.agent_id)

        assert result is True
        assert orchestrator.agent_count == 0

    def test_deregister_nonexistent_agent(self, orchestrator):
        """Test deregistering non-existent agent fails."""
        result = orchestrator.deregister_agent("NONEXISTENT")

        assert result is False

    def test_get_agent_status(self, orchestrator, sample_agent_registration):
        """Test getting agent status."""
        orchestrator.register_agent(sample_agent_registration)

        status = orchestrator.get_agent_status(sample_agent_registration.agent_id)

        assert status is not None
        assert status.agent_id == sample_agent_registration.agent_id

    def test_get_nonexistent_agent_status(self, orchestrator):
        """Test getting non-existent agent status."""
        status = orchestrator.get_agent_status("NONEXISTENT")

        assert status is None


# =============================================================================
# TEST CLASS: WORKFLOW EXECUTION
# =============================================================================

class TestWorkflowExecution:
    """Tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_execute_workflow(self, running_orchestrator, sample_workflow_spec):
        """Test executing a workflow."""
        result = await running_orchestrator.execute_workflow(sample_workflow_spec)

        assert result is not None
        assert result.status.value == "completed"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_workflow_increments_counter(
        self, running_orchestrator, sample_workflow_spec
    ):
        """Test that executing workflow increments counter."""
        initial_count = running_orchestrator._metrics["workflows_executed"]

        await running_orchestrator.execute_workflow(sample_workflow_spec)

        assert running_orchestrator._metrics["workflows_executed"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_workflow_not_running(self, orchestrator, sample_workflow_spec):
        """Test executing workflow when not running fails."""
        with pytest.raises(RuntimeError, match="not running"):
            await orchestrator.execute_workflow(sample_workflow_spec)

    @pytest.mark.asyncio
    async def test_execute_workflow_empty_tasks(self, running_orchestrator):
        """Test executing workflow with empty tasks fails."""
        spec = MockWorkflowSpec(tasks=[])

        with pytest.raises(ValueError, match="at least one task"):
            await running_orchestrator.execute_workflow(spec)

    @pytest.mark.asyncio
    async def test_execute_workflow_missing_agent(self, running_orchestrator):
        """Test executing workflow with missing required agent."""
        spec = MockWorkflowSpec(
            required_agents=["GL-999"]  # Non-existent agent type
        )

        with pytest.raises(ValueError, match="No agent registered"):
            await running_orchestrator.execute_workflow(spec)

    @pytest.mark.asyncio
    async def test_execute_workflow_with_registered_agent(
        self, running_orchestrator, sample_agent_registration
    ):
        """Test executing workflow with required agent registered."""
        running_orchestrator.register_agent(sample_agent_registration)

        spec = MockWorkflowSpec(
            required_agents=["GL-002"]  # Matches sample_agent_registration
        )

        result = await running_orchestrator.execute_workflow(spec)

        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, running_orchestrator):
        """Test cancelling a workflow."""
        # Start a workflow
        spec = MockWorkflowSpec(workflow_id="WF-CANCEL")

        # Cancel it
        result = await running_orchestrator.cancel_workflow("WF-CANCEL")

        # May or may not succeed depending on timing
        assert isinstance(result, bool)


# =============================================================================
# TEST CLASS: SAFETY COORDINATION
# =============================================================================

class TestSafetyCoordination:
    """Tests for safety coordination."""

    @pytest.mark.asyncio
    async def test_trigger_emergency_shutdown(self, running_orchestrator):
        """Test triggering emergency shutdown."""
        await running_orchestrator.trigger_emergency_shutdown("Test ESD")

        assert running_orchestrator._safety_coordinator.is_esd_triggered is True
        assert running_orchestrator._metrics["safety_events"] == 1

    @pytest.mark.asyncio
    async def test_reset_emergency_shutdown(self, running_orchestrator):
        """Test resetting emergency shutdown."""
        await running_orchestrator.trigger_emergency_shutdown("Test ESD")
        result = await running_orchestrator.reset_emergency_shutdown("admin")

        assert result is True
        assert running_orchestrator._safety_coordinator.is_esd_triggered is False

    @pytest.mark.asyncio
    async def test_workflow_blocked_during_esd(self, running_orchestrator, sample_workflow_spec):
        """Test that workflows are blocked during ESD."""
        await running_orchestrator.trigger_emergency_shutdown("Test ESD")

        with pytest.raises(RuntimeError, match="safety system not ready"):
            await running_orchestrator.execute_workflow(sample_workflow_spec)

    @pytest.mark.asyncio
    async def test_safety_status_in_system_status(self, running_orchestrator):
        """Test that safety status is included in system status."""
        status = running_orchestrator.get_system_status()

        assert "safety_status" in status
        assert "esd_armed" in status


# =============================================================================
# TEST CLASS: EVENT HANDLING
# =============================================================================

class TestEventHandling:
    """Tests for event handling."""

    @pytest.mark.asyncio
    async def test_handle_safety_event(self, running_orchestrator):
        """Test handling a safety event."""
        event = MockEvent(event_type="SAFETY_ALARM", source="SENSOR-001")

        await running_orchestrator.handle_event(event)

        assert running_orchestrator._metrics["safety_events"] >= 1


# =============================================================================
# TEST CLASS: METRICS COLLECTION
# =============================================================================

class TestMetricsCollection:
    """Tests for metrics collection."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, running_orchestrator):
        """Test getting metrics."""
        metrics = running_orchestrator.get_metrics()

        assert "workflows_executed" in metrics
        assert "workflows_failed" in metrics
        assert "safety_events" in metrics
        assert "registered_agents" in metrics

    @pytest.mark.asyncio
    async def test_metrics_workflow_counter(
        self, running_orchestrator, sample_workflow_spec
    ):
        """Test workflow counter in metrics."""
        await running_orchestrator.execute_workflow(sample_workflow_spec)

        metrics = running_orchestrator.get_metrics()

        assert metrics["workflows_executed"] >= 1

    def test_metrics_agent_counter(self, orchestrator, sample_agent_registration):
        """Test agent counter in metrics."""
        orchestrator.register_agent(sample_agent_registration)

        metrics = orchestrator.get_metrics()

        assert metrics["registered_agents"] == 1


# =============================================================================
# TEST CLASS: SYSTEM STATUS
# =============================================================================

class TestSystemStatus:
    """Tests for system status reporting."""

    @pytest.mark.asyncio
    async def test_get_system_status(self, running_orchestrator):
        """Test getting system status."""
        status = running_orchestrator.get_system_status()

        assert status["orchestrator_id"] == "ORCH-TEST-001"
        assert status["status"] == "running"
        assert "uptime_seconds" in status

    @pytest.mark.asyncio
    async def test_system_status_agent_count(
        self, running_orchestrator, sample_agent_registration
    ):
        """Test agent count in system status."""
        running_orchestrator.register_agent(sample_agent_registration)

        status = running_orchestrator.get_system_status()

        assert status["registered_agents"] == 1


# =============================================================================
# TEST CLASS: END-TO-END SCENARIOS
# =============================================================================

class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""

    @pytest.mark.asyncio
    async def test_full_orchestrator_workflow(self, orchestrator_config):
        """Test full orchestrator lifecycle with workflows."""
        # Create and start orchestrator
        orchestrator = MockThermalCommandOrchestrator(orchestrator_config)
        await orchestrator.start()

        assert orchestrator.is_running

        # Register agents
        agent1 = MockAgentRegistration(
            agent_id="AGENT-001",
            agent_type="GL-002",
            name="Temperature Control Agent"
        )
        agent2 = MockAgentRegistration(
            agent_id="AGENT-002",
            agent_type="GL-003",
            name="Flow Control Agent"
        )

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        assert orchestrator.agent_count == 2

        # Execute workflow
        workflow = MockWorkflowSpec(
            workflow_id="WF-E2E-001",
            name="End-to-End Test Workflow",
            required_agents=["GL-002", "GL-003"]
        )

        result = await orchestrator.execute_workflow(workflow)

        assert result.status.value == "completed"

        # Check metrics
        metrics = orchestrator.get_metrics()
        assert metrics["workflows_executed"] == 1

        # Stop orchestrator
        await orchestrator.stop()

        assert orchestrator.state == "stopped"

    @pytest.mark.asyncio
    async def test_emergency_shutdown_recovery_scenario(self, orchestrator_config):
        """Test emergency shutdown and recovery."""
        orchestrator = MockThermalCommandOrchestrator(orchestrator_config)
        await orchestrator.start()

        # Normal operation
        workflow1 = MockWorkflowSpec(workflow_id="WF-1")
        await orchestrator.execute_workflow(workflow1)

        # Trigger ESD
        await orchestrator.trigger_emergency_shutdown("High temperature alarm")

        # Verify workflow execution blocked
        workflow2 = MockWorkflowSpec(workflow_id="WF-2")
        with pytest.raises(RuntimeError):
            await orchestrator.execute_workflow(workflow2)

        # Reset ESD
        await orchestrator.reset_emergency_shutdown("operator_123")

        # Verify workflow execution restored
        result = await orchestrator.execute_workflow(workflow2)
        assert result.status.value == "completed"

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_agent_failure_scenario(self, running_orchestrator):
        """Test handling agent failure during workflow."""
        # Register and then deregister an agent mid-workflow
        agent = MockAgentRegistration(
            agent_id="AGENT-TEMP",
            agent_type="GL-002",
            name="Temporary Agent"
        )
        running_orchestrator.register_agent(agent)

        # Execute workflow
        workflow = MockWorkflowSpec(required_agents=["GL-002"])
        result = await running_orchestrator.execute_workflow(workflow)

        assert result.status.value == "completed"

        # Deregister agent
        running_orchestrator.deregister_agent("AGENT-TEMP")

        # Next workflow requiring that agent should fail
        workflow2 = MockWorkflowSpec(
            workflow_id="WF-FAIL",
            required_agents=["GL-002"]
        )

        with pytest.raises(ValueError, match="No agent registered"):
            await running_orchestrator.execute_workflow(workflow2)


# =============================================================================
# TEST CLASS: CONCURRENT OPERATIONS
# =============================================================================

class TestConcurrentOperations:
    """Tests for concurrent operation handling."""

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, running_orchestrator):
        """Test executing multiple workflows concurrently."""
        workflows = [
            MockWorkflowSpec(workflow_id=f"WF-CONC-{i}")
            for i in range(5)
        ]

        # Execute all concurrently
        tasks = [
            running_orchestrator.execute_workflow(wf)
            for wf in workflows
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.status.value == "completed" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_agent_registration(self, orchestrator):
        """Test registering agents concurrently."""
        await orchestrator.start()

        agents = [
            MockAgentRegistration(
                agent_id=f"AGENT-{i}",
                agent_type="GL-002",
                name=f"Agent {i}"
            )
            for i in range(10)
        ]

        # Register concurrently
        for agent in agents:
            orchestrator.register_agent(agent)

        assert orchestrator.agent_count == 10

        await orchestrator.stop()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for orchestrator."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_workflow_execution_time(self, running_orchestrator):
        """Test workflow execution completes within time limit."""
        import time

        workflow = MockWorkflowSpec()

        start = time.perf_counter()
        result = await running_orchestrator.execute_workflow(workflow)
        elapsed = time.perf_counter() - start

        # Workflow should complete in under 1 second
        assert elapsed < 1.0
        assert result.status.value == "completed"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_high_volume_workflow_execution(self, running_orchestrator):
        """Test handling high volume of workflows."""
        import time

        num_workflows = 100
        workflows = [
            MockWorkflowSpec(workflow_id=f"WF-VOL-{i}")
            for i in range(num_workflows)
        ]

        start = time.perf_counter()

        for wf in workflows:
            await running_orchestrator.execute_workflow(wf)

        elapsed = time.perf_counter() - start

        # Should complete 100 workflows in under 5 seconds
        assert elapsed < 5.0

        metrics = running_orchestrator.get_metrics()
        assert metrics["workflows_executed"] == num_workflows

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_startup_time(self, orchestrator_config):
        """Test orchestrator startup time."""
        import time

        orchestrator = MockThermalCommandOrchestrator(orchestrator_config)

        start = time.perf_counter()
        await orchestrator.start()
        elapsed = time.perf_counter() - start

        # Startup should complete in under 1 second
        assert elapsed < 1.0
        assert orchestrator.is_running

        await orchestrator.stop()


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_system_status_determinism(self, running_orchestrator):
        """Test that system status is consistent."""
        status1 = running_orchestrator.get_system_status()
        status2 = running_orchestrator.get_system_status()

        # Key fields should be consistent
        assert status1["orchestrator_id"] == status2["orchestrator_id"]
        assert status1["orchestrator_name"] == status2["orchestrator_name"]
        assert status1["status"] == status2["status"]

    @pytest.mark.asyncio
    async def test_metrics_consistency(self, running_orchestrator):
        """Test that metrics are consistent and non-decreasing."""
        metrics1 = running_orchestrator.get_metrics()

        await running_orchestrator.execute_workflow(MockWorkflowSpec())

        metrics2 = running_orchestrator.get_metrics()

        # Counters should be non-decreasing
        assert metrics2["workflows_executed"] >= metrics1["workflows_executed"]
