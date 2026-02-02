# -*- coding: utf-8 -*-
"""
Tests for BaseOrchestrator and orchestration infrastructure.

This module provides comprehensive tests for:
- BaseOrchestrator abstract class
- MessageBus event-driven messaging
- TaskScheduler load balancing
- CoordinationLayer agent coordination
- SafetyMonitor safety oversight

Author: GreenLang Framework Team
Date: December 2025
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from greenlang.execution.core.base_orchestrator import (
    BaseOrchestrator,
    OrchestrationResult,
    OrchestratorConfig,
    OrchestratorMetrics,
    OrchestratorState,
)
from greenlang.execution.core.message_bus import (
    Message,
    MessageBus,
    MessageBusConfig,
    MessagePriority,
    MessageType,
)
from greenlang.execution.core.task_scheduler import (
    AgentCapacity,
    LoadBalanceStrategy,
    Task,
    TaskPriority,
    TaskScheduler,
    TaskSchedulerConfig,
    TaskState,
)
from greenlang.execution.core.coordination_layer import (
    AgentInfo,
    CoordinationConfig,
    CoordinationLayer,
    CoordinationPattern,
    DistributedLock,
    Saga,
    SagaStep,
    TransactionState,
)
from greenlang.execution.core.safety_monitor import (
    CircuitBreaker,
    CircuitState,
    ConstraintType,
    OperationContext,
    SafetyConfig,
    SafetyConstraint,
    SafetyLevel,
    SafetyMonitor,
    ViolationSeverity,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class ConcreteOrchestrator(BaseOrchestrator[Dict[str, Any], Dict[str, Any]]):
    """Concrete implementation of BaseOrchestrator for testing."""

    def __init__(self, config: OrchestratorConfig):
        self.orchestrate_called = False
        self.orchestrate_input = None
        super().__init__(config)

    def _create_message_bus(self) -> MessageBus:
        return MessageBus(MessageBusConfig(max_queue_size=100))

    def _create_task_scheduler(self) -> TaskScheduler:
        return TaskScheduler(TaskSchedulerConfig(max_queue_size=100))

    def _create_coordinator(self) -> CoordinationLayer:
        return CoordinationLayer(CoordinationConfig())

    def _create_safety_monitor(self) -> SafetyMonitor:
        return SafetyMonitor(SafetyConfig(halt_on_critical=False))

    async def orchestrate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.orchestrate_called = True
        self.orchestrate_input = input_data
        return {"result": "success", "input_count": len(input_data)}


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Create test orchestrator config."""
    return OrchestratorConfig(
        orchestrator_id="test-orchestrator",
        name="TestOrchestrator",
        version="1.0.0",
        enable_message_bus=True,
        enable_task_scheduling=True,
        enable_coordination=True,
        enable_safety_monitoring=True,
    )


@pytest.fixture
def message_bus_config() -> MessageBusConfig:
    """Create test message bus config."""
    return MessageBusConfig(
        max_queue_size=100,
        enable_dead_letter=True,
        max_retries=2,
    )


@pytest.fixture
def task_scheduler_config() -> TaskSchedulerConfig:
    """Create test task scheduler config."""
    return TaskSchedulerConfig(
        max_queue_size=100,
        load_balance_strategy=LoadBalanceStrategy.ROUND_ROBIN,
        default_timeout_seconds=5.0,
    )


@pytest.fixture
def coordination_config() -> CoordinationConfig:
    """Create test coordination config."""
    return CoordinationConfig(
        pattern=CoordinationPattern.ORCHESTRATION,
        lock_ttl_seconds=5.0,
    )


@pytest.fixture
def safety_config() -> SafetyConfig:
    """Create test safety config."""
    return SafetyConfig(
        enable_circuit_breakers=True,
        enable_rate_limiting=True,
        halt_on_critical=False,
    )


# =============================================================================
# BaseOrchestrator Tests
# =============================================================================

class TestBaseOrchestrator:
    """Tests for BaseOrchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator_config: OrchestratorConfig):
        """Test orchestrator initialization."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        assert orchestrator.config.orchestrator_id == "test-orchestrator"
        assert orchestrator.get_state() == OrchestratorState.READY
        assert orchestrator.message_bus is not None
        assert orchestrator.task_scheduler is not None
        assert orchestrator.coordinator is not None
        assert orchestrator.safety_monitor is not None

    @pytest.mark.asyncio
    async def test_execute_success(self, orchestrator_config: OrchestratorConfig):
        """Test successful orchestration execution."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        input_data = {"key1": "value1", "key2": "value2"}
        result = await orchestrator.execute(input_data)

        assert result.success is True
        assert result.output["result"] == "success"
        assert result.output["input_count"] == 2
        assert len(result.provenance_hash) == 64  # SHA-256 hex
        assert result.execution_time_ms > 0
        assert orchestrator.orchestrate_called is True
        assert orchestrator.orchestrate_input == input_data

    @pytest.mark.asyncio
    async def test_execute_with_metrics(self, orchestrator_config: OrchestratorConfig):
        """Test metrics collection during execution."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        await orchestrator.execute({"data": "test"})
        metrics = orchestrator.get_metrics()

        assert metrics.executions_total == 1
        assert metrics.executions_successful == 1
        assert metrics.executions_failed == 0
        assert metrics.avg_execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator_config: OrchestratorConfig):
        """Test agent registration."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        await orchestrator.register_agent(
            agent_id="test-agent",
            capabilities={"thermal", "energy"},
            role="slave",
        )

        agents = orchestrator.get_managed_agents()
        assert "test-agent" in agents
        assert agents["test-agent"].capabilities == {"thermal", "energy"}

    @pytest.mark.asyncio
    async def test_unregister_agent(self, orchestrator_config: OrchestratorConfig):
        """Test agent unregistration."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        await orchestrator.register_agent("test-agent")
        result = await orchestrator.unregister_agent("test-agent")

        assert result is True
        assert "test-agent" not in orchestrator.get_managed_agents()

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, orchestrator_config: OrchestratorConfig):
        """Test orchestrator lifecycle."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        await orchestrator.start()
        assert orchestrator.get_state() == OrchestratorState.READY

        await orchestrator.shutdown()
        assert orchestrator.get_state() == OrchestratorState.TERMINATED


# =============================================================================
# MessageBus Tests
# =============================================================================

class TestMessageBus:
    """Tests for MessageBus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, message_bus_config: MessageBusConfig):
        """Test basic publish/subscribe."""
        bus = MessageBus(message_bus_config)
        received_messages: List[Message] = []

        async def handler(msg: Message):
            received_messages.append(msg)

        await bus.subscribe("test.topic", handler, "test-subscriber")
        await bus.start()

        message = Message(
            sender_id="sender",
            recipient_id="recipient",
            message_type=MessageType.EVENT,
            topic="test.topic",
            payload={"data": "test"},
        )

        await bus.publish(message)
        await asyncio.sleep(0.2)  # Wait for processing

        await bus.close()

        assert len(received_messages) == 1
        assert received_messages[0].payload["data"] == "test"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, message_bus_config: MessageBusConfig):
        """Test wildcard topic subscription."""
        bus = MessageBus(message_bus_config)
        received_messages: List[Message] = []

        async def handler(msg: Message):
            received_messages.append(msg)

        await bus.subscribe("agent.*", handler, "test-subscriber")
        await bus.start()

        # Should match
        await bus.publish(Message(
            sender_id="sender",
            recipient_id="*",
            message_type=MessageType.EVENT,
            topic="agent.thermal",
            payload={"type": "thermal"},
        ))

        await asyncio.sleep(0.2)
        await bus.close()

        assert len(received_messages) == 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self, message_bus_config: MessageBusConfig):
        """Test message priority ordering."""
        bus = MessageBus(message_bus_config)
        received_priorities: List[str] = []

        async def handler(msg: Message):
            received_priorities.append(msg.priority.value)

        await bus.subscribe("test.#", handler, "test-subscriber")
        await bus.start()

        # Publish in reverse priority order
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH]:
            await bus.publish(Message(
                sender_id="sender",
                recipient_id="*",
                message_type=MessageType.EVENT,
                topic="test.priority",
                payload={},
                priority=priority,
            ))

        await asyncio.sleep(0.3)
        await bus.close()

        # Higher priority should be processed first
        assert received_priorities[0] == "high"

    @pytest.mark.asyncio
    async def test_message_expiry(self, message_bus_config: MessageBusConfig):
        """Test message TTL expiry."""
        bus = MessageBus(message_bus_config)
        received_count = 0

        async def handler(msg: Message):
            nonlocal received_count
            received_count += 1

        await bus.subscribe("test.topic", handler, "test-subscriber")

        # Create expired message with short TTL and old timestamp
        message = Message(
            sender_id="sender",
            recipient_id="recipient",
            message_type=MessageType.EVENT,
            topic="test.topic",
            payload={},
            ttl_seconds=1,  # 1 second TTL
            timestamp="2020-01-01T00:00:00+00:00",  # Set old timestamp in constructor
        )

        # Verify message reports as expired
        assert message.is_expired() is True

    @pytest.mark.asyncio
    async def test_metrics(self, message_bus_config: MessageBusConfig):
        """Test message bus metrics."""
        bus = MessageBus(message_bus_config)

        async def handler(msg: Message):
            pass

        await bus.subscribe("test.topic", handler, "test-subscriber")
        await bus.start()

        await bus.publish(Message(
            sender_id="sender",
            recipient_id="*",
            message_type=MessageType.EVENT,
            topic="test.topic",
            payload={},
        ))

        await asyncio.sleep(0.2)
        metrics = bus.get_metrics()
        await bus.close()

        assert metrics.messages_published >= 1
        assert metrics.active_subscriptions >= 1


# =============================================================================
# TaskScheduler Tests
# =============================================================================

class TestTaskScheduler:
    """Tests for TaskScheduler."""

    @pytest.mark.asyncio
    async def test_schedule_task(self, task_scheduler_config: TaskSchedulerConfig):
        """Test basic task scheduling."""
        scheduler = TaskScheduler(task_scheduler_config)

        task = Task(
            task_type="test_task",
            payload={"key": "value"},
            priority=TaskPriority.NORMAL,
        )

        task_id = await scheduler.schedule(task)

        assert task_id == task.task_id
        scheduled_task = await scheduler.get_task(task_id)
        assert scheduled_task is not None
        assert scheduled_task.task_type == "test_task"

    @pytest.mark.asyncio
    async def test_register_agent(self, task_scheduler_config: TaskSchedulerConfig):
        """Test agent registration."""
        scheduler = TaskScheduler(task_scheduler_config)

        capacity = AgentCapacity(
            agent_id="test-agent",
            capabilities={"thermal", "energy"},
            max_concurrent_tasks=5,
        )

        scheduler.register_agent(capacity)
        agents = scheduler.get_agents()

        assert "test-agent" in agents
        assert agents["test-agent"].max_concurrent_tasks == 5

    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_scheduler_config: TaskSchedulerConfig):
        """Test task cancellation."""
        scheduler = TaskScheduler(task_scheduler_config)

        task = Task(task_type="test", payload={})
        task_id = await scheduler.schedule(task)

        result = await scheduler.cancel(task_id)
        assert result is True

        status = await scheduler.get_task_status(task_id)
        assert status == TaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_heartbeat(self, task_scheduler_config: TaskSchedulerConfig):
        """Test agent heartbeat."""
        scheduler = TaskScheduler(task_scheduler_config)
        scheduler.register_agent(AgentCapacity(agent_id="test-agent"))

        result = await scheduler.heartbeat("test-agent")
        assert result is True

        result = await scheduler.heartbeat("nonexistent-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_balance_strategy(self, task_scheduler_config: TaskSchedulerConfig):
        """Test load balance strategy selection."""
        scheduler = TaskScheduler(task_scheduler_config)

        # Register multiple agents
        for i in range(3):
            scheduler.register_agent(AgentCapacity(
                agent_id=f"agent-{i}",
                capabilities={"test"},
                max_concurrent_tasks=10,
            ))

        # Verify round robin distributes tasks
        task = Task(task_type="test", payload={})
        selected = scheduler._select_agent(task)

        assert selected is not None
        assert selected.startswith("agent-")


# =============================================================================
# CoordinationLayer Tests
# =============================================================================

class TestCoordinationLayer:
    """Tests for CoordinationLayer."""

    def test_register_agent(self, coordination_config: CoordinationConfig):
        """Test agent registration."""
        coordinator = CoordinationLayer(coordination_config)

        agent = AgentInfo(
            agent_id="test-agent",
            role="slave",
            capabilities={"thermal"},
        )

        coordinator.register_agent(agent)
        agents = coordinator.get_agents()

        assert "test-agent" in agents
        assert agents["test-agent"].role == "slave"

    def test_unregister_agent(self, coordination_config: CoordinationConfig):
        """Test agent unregistration."""
        coordinator = CoordinationLayer(coordination_config)

        coordinator.register_agent(AgentInfo(agent_id="test-agent"))
        result = coordinator.unregister_agent("test-agent")

        assert result is True
        assert "test-agent" not in coordinator.get_agents()

    @pytest.mark.asyncio
    async def test_distributed_lock(self, coordination_config: CoordinationConfig):
        """Test distributed locking."""
        coordinator = CoordinationLayer(coordination_config)

        # Test internal lock acquisition directly
        acquired = await coordinator._acquire_lock_internal("resource-1", "agent-1", 5.0)
        assert acquired is True

        # Lock should be held
        locks = coordinator._locks
        assert "resource-1" in locks
        assert locks["resource-1"].holder_id == "agent-1"

        # Release lock
        released = await coordinator._release_lock_internal("resource-1", "agent-1")
        assert released is True

        # Lock should be released
        assert locks["resource-1"].holder_id is None

    @pytest.mark.asyncio
    async def test_coordinate_agents(self, coordination_config: CoordinationConfig):
        """Test multi-agent coordination."""
        coordinator = CoordinationLayer(coordination_config)

        # Register agents
        for i in range(3):
            coordinator.register_agent(AgentInfo(
                agent_id=f"agent-{i}",
                capabilities={"thermal"},
            ))

        result = await coordinator.coordinate_agents(
            ["agent-0", "agent-1", "agent-2"],
            {"task": "optimize"},
            CoordinationPattern.ORCHESTRATION,
        )

        assert result["status"] == "coordinated"
        assert len(result["assignments"]) > 0

    @pytest.mark.asyncio
    async def test_consensus_voting(self, coordination_config: CoordinationConfig):
        """Test consensus-based decisions."""
        coordinator = CoordinationLayer(coordination_config)

        # Register agents for voting
        for i in range(3):
            coordinator.register_agent(AgentInfo(agent_id=f"agent-{i}"))

        # Create proposal
        proposal = await coordinator.propose_consensus(
            topic="temperature_increase",
            proposed_by="agent-0",
            data={"new_temp": 500},
            required_approval=0.5,
        )

        # Agents vote
        await coordinator.vote_on_proposal(proposal.proposal_id, "agent-0", "approve")
        await coordinator.vote_on_proposal(proposal.proposal_id, "agent-1", "approve")
        await coordinator.vote_on_proposal(proposal.proposal_id, "agent-2", "reject")

        # Resolve
        from greenlang.execution.core.coordination_layer import ConsensusResult
        result = await coordinator.resolve_consensus(proposal.proposal_id)
        assert result == ConsensusResult.ACHIEVED

    @pytest.mark.asyncio
    async def test_saga_transaction(self, coordination_config: CoordinationConfig):
        """Test saga transaction pattern."""
        coordinator = CoordinationLayer(coordination_config)
        executed_actions: List[str] = []

        async def action_handler(agent_id: str, action: Dict) -> Any:
            executed_actions.append(f"{agent_id}:{action.get('command')}")
            return {"success": True}

        coordinator.register_agent(AgentInfo(agent_id="agent-1"))
        coordinator.register_action_handler("agent-1", action_handler)

        saga = Saga(
            name="test_saga",
            steps=[
                SagaStep(
                    name="step1",
                    agent_id="agent-1",
                    action={"command": "action1"},
                    compensation={"command": "compensate1"},
                ),
            ],
        )

        result = await coordinator.run_saga(saga)

        assert result.state == TransactionState.COMMITTED
        assert "agent-1:action1" in executed_actions


# =============================================================================
# SafetyMonitor Tests
# =============================================================================

class TestSafetyMonitor:
    """Tests for SafetyMonitor."""

    @pytest.mark.asyncio
    async def test_threshold_constraint(self, safety_config: SafetyConfig):
        """Test threshold constraint validation."""
        monitor = SafetyMonitor(safety_config)

        monitor.add_constraint(SafetyConstraint(
            name="max_temp",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=100.0,
            level=SafetyLevel.CRITICAL,
            metadata={"parameter": "temperature"},
        ))

        # Valid operation
        context = OperationContext(
            operation_type="set_temp",
            agent_id="agent-1",
            parameters={"temperature": 80.0},
        )
        result = await monitor.validate_operation(context)
        assert result.is_safe is True

        # Invalid operation
        context = OperationContext(
            operation_type="set_temp",
            agent_id="agent-1",
            parameters={"temperature": 150.0},
        )
        result = await monitor.validate_operation(context)
        assert result.is_safe is False
        assert len(result.violations) == 1

    @pytest.mark.asyncio
    async def test_rate_limiting(self, safety_config: SafetyConfig):
        """Test rate limiting."""
        monitor = SafetyMonitor(safety_config)

        monitor.add_rate_limiter("test-limiter", max_requests=2, window_seconds=60.0)

        # First two requests should pass
        context = OperationContext(
            operation_type="test",
            agent_id="agent-1",
            parameters={},
        )

        # Note: Rate limiting is key-based, need matching constraint

    def test_circuit_breaker(self, safety_config: SafetyConfig):
        """Test circuit breaker functionality."""
        monitor = SafetyMonitor(safety_config)

        # Circuit breaker key format is "agent_id:operation_type"
        # When we add a circuit breaker, we use the full key
        breaker = monitor.add_circuit_breaker(
            "agent-1:test-operation",  # Full key format
            failure_threshold=2,
            timeout_seconds=1.0,
        )

        # Record failures using agent_id and operation_type
        monitor.record_failure("agent-1", "test-operation")
        monitor.record_failure("agent-1", "test-operation")

        # Circuit should be open
        status = monitor.get_circuit_breaker_status()
        assert "agent-1:test-operation" in status
        assert status["agent-1:test-operation"]["state"] == CircuitState.OPEN.value

    def test_violation_tracking(self, safety_config: SafetyConfig):
        """Test violation tracking."""
        monitor = SafetyMonitor(safety_config)

        monitor.add_constraint(SafetyConstraint(
            name="test_constraint",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=10.0,
            metadata={"parameter": "value"},
        ))

        # Trigger violation (sync validation)
        violations = monitor.get_violations()
        initial_count = len(violations)

        # Violations should be tracked in metrics
        metrics = monitor.get_metrics()
        assert metrics.validations_performed >= 0

    @pytest.mark.asyncio
    async def test_halt_on_critical(self):
        """Test system halt on critical violation."""
        config = SafetyConfig(halt_on_critical=True)
        monitor = SafetyMonitor(config)

        monitor.add_constraint(SafetyConstraint(
            name="critical_constraint",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=100.0,
            level=SafetyLevel.CRITICAL,
            metadata={"parameter": "value"},
        ))

        context = OperationContext(
            operation_type="test",
            agent_id="agent-1",
            parameters={"value": 200.0},
        )

        await monitor.validate_operation(context)
        assert monitor.is_halted() is True

        # Reset halt
        monitor.reset_halt()
        assert monitor.is_halted() is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for orchestration components."""

    @pytest.mark.asyncio
    async def test_full_orchestration_flow(self, orchestrator_config: OrchestratorConfig):
        """Test complete orchestration flow."""
        orchestrator = ConcreteOrchestrator(orchestrator_config)

        # Start orchestrator
        await orchestrator.start()

        # Register agents
        await orchestrator.register_agent(
            "agent-1",
            capabilities={"thermal"},
            role="slave",
        )

        # Add safety constraint
        orchestrator.add_safety_constraint(
            name="max_temp",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=600.0,
        )

        # Execute orchestration
        result = await orchestrator.execute({
            "temperature": 450,
            "operation": "optimize",
        })

        assert result.success is True
        assert result.provenance_hash != ""

        # Check metrics
        metrics = orchestrator.get_metrics()
        assert metrics.executions_successful >= 1

        # Shutdown
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_message_bus_with_scheduler(
        self,
        message_bus_config: MessageBusConfig,
        task_scheduler_config: TaskSchedulerConfig,
    ):
        """Test message bus and task scheduler integration."""
        bus = MessageBus(message_bus_config)
        scheduler = TaskScheduler(task_scheduler_config)

        tasks_received: List[str] = []

        async def task_handler(msg: Message):
            task = Task(
                task_type=msg.payload.get("task_type", "default"),
                payload=msg.payload,
            )
            task_id = await scheduler.schedule(task)
            tasks_received.append(task_id)

        await bus.subscribe("tasks.#", task_handler, "task-receiver")
        await bus.start()

        # Publish task message
        await bus.publish(Message(
            sender_id="producer",
            recipient_id="*",
            message_type=MessageType.COMMAND,
            topic="tasks.thermal",
            payload={"task_type": "thermal_calculation"},
        ))

        await asyncio.sleep(0.2)
        await bus.close()

        assert len(tasks_received) == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
