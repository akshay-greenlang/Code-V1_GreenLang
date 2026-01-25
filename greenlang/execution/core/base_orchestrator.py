# -*- coding: utf-8 -*-
"""
BaseOrchestrator - Abstract base class for agent orchestration.

This module provides the BaseOrchestrator abstract class that defines standard
orchestration patterns for multi-agent systems. It integrates message bus,
task scheduling, coordination layer, and safety monitoring into a cohesive
orchestration framework.

Example:
    >>> class ProcessHeatOrchestrator(BaseOrchestrator[ProcessInput, ProcessOutput]):
    ...     async def orchestrate(self, input_data: ProcessInput) -> ProcessOutput:
    ...         # Orchestration logic
    ...         pass

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

from greenlang.core.message_bus import (
    Message,
    MessageBus,
    MessageBusConfig,
    MessagePriority,
    MessageType,
)
from greenlang.core.task_scheduler import (
    AgentCapacity,
    LoadBalanceStrategy,
    Task,
    TaskPriority,
    TaskScheduler,
    TaskSchedulerConfig,
    TaskState,
)
from greenlang.core.coordination_layer import (
    AgentInfo,
    CoordinationConfig,
    CoordinationLayer,
    CoordinationPattern,
    Saga,
    SagaStep,
)
from greenlang.core.safety_monitor import (
    OperationContext,
    SafetyConfig,
    SafetyConstraint,
    SafetyMonitor,
    ConstraintType,
    SafetyLevel,
)

logger = logging.getLogger(__name__)

# Type variables for generic orchestration
InT = TypeVar("InT")
OutT = TypeVar("OutT")


class OrchestratorState(str, Enum):
    """Orchestrator execution states."""

    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    RECOVERING = "recovering"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class OrchestratorConfig:
    """
    Configuration for BaseOrchestrator.

    Attributes:
        orchestrator_id: Unique orchestrator identifier
        name: Orchestrator display name
        version: Version string
        max_concurrent_tasks: Maximum concurrent tasks
        default_timeout_seconds: Default task timeout
        enable_safety_monitoring: Enable safety oversight
        enable_message_bus: Enable async messaging
        enable_task_scheduling: Enable task scheduler
        enable_coordination: Enable coordination layer
        coordination_pattern: Default coordination pattern
        load_balance_strategy: Task distribution strategy
        checkpoint_enabled: Enable state checkpointing
        checkpoint_interval_seconds: Checkpoint interval
        metrics_enabled: Enable metrics collection
        recovery_enabled: Enable error recovery
        max_retries: Maximum retry attempts
    """

    orchestrator_id: str
    name: str = "BaseOrchestrator"
    version: str = "1.0.0"
    max_concurrent_tasks: int = 100
    default_timeout_seconds: float = 120.0
    enable_safety_monitoring: bool = True
    enable_message_bus: bool = True
    enable_task_scheduling: bool = True
    enable_coordination: bool = True
    coordination_pattern: CoordinationPattern = CoordinationPattern.ORCHESTRATION
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED
    checkpoint_enabled: bool = True
    checkpoint_interval_seconds: float = 300.0
    metrics_enabled: bool = True
    recovery_enabled: bool = True
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "orchestrator_id": self.orchestrator_id,
            "name": self.name,
            "version": self.version,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "default_timeout_seconds": self.default_timeout_seconds,
            "enable_safety_monitoring": self.enable_safety_monitoring,
            "enable_message_bus": self.enable_message_bus,
            "enable_task_scheduling": self.enable_task_scheduling,
            "enable_coordination": self.enable_coordination,
            "coordination_pattern": self.coordination_pattern.value,
            "load_balance_strategy": self.load_balance_strategy.value,
            "checkpoint_enabled": self.checkpoint_enabled,
            "metrics_enabled": self.metrics_enabled,
            "recovery_enabled": self.recovery_enabled,
            "max_retries": self.max_retries,
        }


@dataclass
class OrchestratorMetrics:
    """Metrics for orchestrator monitoring."""

    executions_total: int = 0
    executions_successful: int = 0
    executions_failed: int = 0
    avg_execution_time_ms: float = 0.0
    current_state: str = "initializing"
    agents_managed: int = 0
    tasks_pending: int = 0
    tasks_running: int = 0
    messages_processed: int = 0
    safety_validations: int = 0
    violations_detected: int = 0
    last_execution_time: Optional[str] = None
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "executions_total": self.executions_total,
            "executions_successful": self.executions_successful,
            "executions_failed": self.executions_failed,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "current_state": self.current_state,
            "agents_managed": self.agents_managed,
            "tasks_pending": self.tasks_pending,
            "tasks_running": self.tasks_running,
            "messages_processed": self.messages_processed,
            "safety_validations": self.safety_validations,
            "violations_detected": self.violations_detected,
            "last_execution_time": self.last_execution_time,
            "last_updated": self.last_updated,
        }


@dataclass
class OrchestrationResult(Generic[OutT]):
    """
    Result of an orchestration operation.

    Attributes:
        success: Whether orchestration succeeded
        output: Output data (on success)
        execution_time_ms: Execution duration
        provenance_hash: SHA-256 hash for audit trail
        tasks_executed: Number of tasks executed
        agents_coordinated: Number of agents coordinated
        error: Error message (on failure)
        warnings: List of warnings
        metadata: Additional result metadata
    """

    success: bool
    output: Optional[OutT] = None
    execution_time_ms: float = 0.0
    provenance_hash: str = ""
    tasks_executed: int = 0
    agents_coordinated: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "provenance_hash": self.provenance_hash,
            "tasks_executed": self.tasks_executed,
            "agents_coordinated": self.agents_coordinated,
            "error": self.error,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class BaseOrchestrator(ABC, Generic[InT, OutT]):
    """
    Abstract base class for agent orchestrators.

    Provides standard orchestration patterns:
    - Hierarchical master-slave coordination
    - Event-driven async message passing
    - Task scheduling and load balancing
    - Safety monitoring and constraint validation
    - Provenance tracking with SHA-256 hashing

    Subclasses must implement:
    - _create_message_bus(): Create message bus instance
    - _create_task_scheduler(): Create task scheduler instance
    - _create_coordinator(): Create coordination layer instance
    - _create_safety_monitor(): Create safety monitor instance
    - orchestrate(): Main orchestration logic

    Example:
        >>> class ProcessHeatOrchestrator(BaseOrchestrator[Dict, Dict]):
        ...     def _create_message_bus(self) -> MessageBus:
        ...         return MessageBus(MessageBusConfig())
        ...
        ...     def _create_task_scheduler(self) -> TaskScheduler:
        ...         return TaskScheduler(TaskSchedulerConfig())
        ...
        ...     def _create_coordinator(self) -> CoordinationLayer:
        ...         return CoordinationLayer(CoordinationConfig())
        ...
        ...     def _create_safety_monitor(self) -> SafetyMonitor:
        ...         return SafetyMonitor(SafetyConfig())
        ...
        ...     async def orchestrate(self, input_data: Dict) -> Dict:
        ...         # Main orchestration logic
        ...         return result
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        """
        Initialize BaseOrchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self._state = OrchestratorState.INITIALIZING
        self._metrics = OrchestratorMetrics()
        self._execution_times: List[float] = []
        self._managed_agents: Dict[str, AgentInfo] = {}
        self._execution_history: List[Dict[str, Any]] = []

        # Initialize components based on config
        self.message_bus: Optional[MessageBus] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.coordinator: Optional[CoordinationLayer] = None
        self.safety_monitor: Optional[SafetyMonitor] = None

        if config.enable_message_bus:
            self.message_bus = self._create_message_bus()

        if config.enable_task_scheduling:
            self.task_scheduler = self._create_task_scheduler()

        if config.enable_coordination:
            self.coordinator = self._create_coordinator()

        if config.enable_safety_monitoring:
            self.safety_monitor = self._create_safety_monitor()

        self._state = OrchestratorState.READY
        self._metrics.current_state = self._state.value

        logger.info(
            f"BaseOrchestrator {config.orchestrator_id} initialized "
            f"(version: {config.version})"
        )

    @abstractmethod
    def _create_message_bus(self) -> MessageBus:
        """
        Create message bus for agent communication.

        Subclasses should implement this to configure the message bus
        with application-specific settings.

        Returns:
            Configured MessageBus instance
        """
        pass

    @abstractmethod
    def _create_task_scheduler(self) -> TaskScheduler:
        """
        Create task scheduler for load balancing.

        Subclasses should implement this to configure the scheduler
        with application-specific settings.

        Returns:
            Configured TaskScheduler instance
        """
        pass

    @abstractmethod
    def _create_coordinator(self) -> CoordinationLayer:
        """
        Create coordination layer for agent management.

        Subclasses should implement this to configure coordination
        with application-specific patterns.

        Returns:
            Configured CoordinationLayer instance
        """
        pass

    @abstractmethod
    def _create_safety_monitor(self) -> SafetyMonitor:
        """
        Create safety monitor for oversight.

        Subclasses should implement this to configure safety constraints
        specific to the application domain.

        Returns:
            Configured SafetyMonitor instance
        """
        pass

    @abstractmethod
    async def orchestrate(self, input_data: InT) -> OutT:
        """
        Main orchestration logic.

        Subclasses must implement this method to define how agents
        are coordinated, tasks are executed, and results are aggregated.

        Args:
            input_data: Input data for orchestration

        Returns:
            Orchestration output
        """
        pass

    async def execute(self, input_data: InT) -> OrchestrationResult[OutT]:
        """
        Execute orchestration with full lifecycle management.

        This method wraps the orchestrate() method with:
        - State management
        - Safety validation
        - Error handling and recovery
        - Metrics collection
        - Provenance tracking

        Args:
            input_data: Input data for orchestration

        Returns:
            OrchestrationResult with output and metadata
        """
        start_time = time.perf_counter()
        execution_id = f"{self.config.orchestrator_id}-{int(time.time() * 1000)}"

        self._state = OrchestratorState.EXECUTING
        self._metrics.current_state = self._state.value

        logger.info(f"Starting orchestration: {execution_id}")

        try:
            # Safety validation
            if self.safety_monitor:
                context = OperationContext(
                    operation_type="orchestrate",
                    agent_id=self.config.orchestrator_id,
                    parameters={"input": str(type(input_data).__name__)},
                    operation_id=execution_id,
                )
                validation = await self.safety_monitor.validate_operation(context)
                self._metrics.safety_validations += 1

                if not validation.is_safe:
                    self._metrics.violations_detected += len(validation.violations)
                    self._metrics.executions_failed += 1
                    return OrchestrationResult(
                        success=False,
                        error=f"Safety validation failed: {validation.violations[0].message}",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

            # Execute orchestration
            output = await self.orchestrate(input_data)

            # Calculate execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_execution_metrics(execution_time_ms, success=True)

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data, output, execution_id
            )

            # Record success
            if self.safety_monitor:
                self.safety_monitor.record_success(
                    self.config.orchestrator_id, "orchestrate"
                )

            result = OrchestrationResult(
                success=True,
                output=output,
                execution_time_ms=execution_time_ms,
                provenance_hash=provenance_hash,
                tasks_executed=self._get_tasks_executed(),
                agents_coordinated=len(self._managed_agents),
                metadata={
                    "execution_id": execution_id,
                    "orchestrator_id": self.config.orchestrator_id,
                    "version": self.config.version,
                },
            )

            self._state = OrchestratorState.READY
            self._metrics.current_state = self._state.value

            logger.info(
                f"Orchestration completed: {execution_id} "
                f"({execution_time_ms:.2f}ms)"
            )

            return result

        except Exception as e:
            self._state = OrchestratorState.ERROR
            self._metrics.current_state = self._state.value
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            logger.error(f"Orchestration failed: {execution_id} - {e}", exc_info=True)

            # Record failure
            if self.safety_monitor:
                self.safety_monitor.record_failure(
                    self.config.orchestrator_id, "orchestrate"
                )

            # Attempt recovery
            if self.config.recovery_enabled:
                recovery_result = await self._handle_error_recovery(
                    e, input_data, execution_id
                )
                if recovery_result:
                    return recovery_result

            self._update_execution_metrics(execution_time_ms, success=False)

            return OrchestrationResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={
                    "execution_id": execution_id,
                    "error_type": type(e).__name__,
                },
            )

    async def start(self) -> None:
        """
        Start all orchestrator components.

        Call this method to start the message bus processor,
        task scheduler, and other async components.
        """
        logger.info(f"Starting orchestrator: {self.config.orchestrator_id}")

        if self.message_bus:
            await self.message_bus.start()

        if self.task_scheduler:
            await self.task_scheduler.start()

        self._state = OrchestratorState.READY
        self._metrics.current_state = self._state.value

        logger.info(f"Orchestrator started: {self.config.orchestrator_id}")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the orchestrator.

        Stops all components and releases resources.
        """
        logger.info(f"Shutting down orchestrator: {self.config.orchestrator_id}")

        if self.message_bus:
            await self.message_bus.close()

        if self.task_scheduler:
            await self.task_scheduler.stop()

        self._state = OrchestratorState.TERMINATED
        self._metrics.current_state = self._state.value

        logger.info(f"Orchestrator shutdown complete: {self.config.orchestrator_id}")

    async def register_agent(
        self,
        agent_id: str,
        capabilities: Optional[Set[str]] = None,
        role: str = "slave",
        max_concurrent_tasks: int = 10,
    ) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent_id: Unique agent identifier
            capabilities: Set of agent capabilities
            role: Agent role (master, slave, peer)
            max_concurrent_tasks: Max concurrent tasks for agent
        """
        # Register with coordinator
        if self.coordinator:
            agent_info = AgentInfo(
                agent_id=agent_id,
                role=role,
                capabilities=capabilities or set(),
            )
            self.coordinator.register_agent(agent_info)
            self._managed_agents[agent_id] = agent_info

        # Register with task scheduler
        if self.task_scheduler:
            capacity = AgentCapacity(
                agent_id=agent_id,
                capabilities=capabilities or set(),
                max_concurrent_tasks=max_concurrent_tasks,
            )
            self.task_scheduler.register_agent(capacity)

        self._metrics.agents_managed = len(self._managed_agents)
        logger.info(f"Registered agent: {agent_id} (role: {role})")

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if agent was removed
        """
        removed = False

        if self.coordinator:
            removed = self.coordinator.unregister_agent(agent_id) or removed

        if self.task_scheduler:
            removed = self.task_scheduler.unregister_agent(agent_id) or removed

        if agent_id in self._managed_agents:
            del self._managed_agents[agent_id]
            removed = True

        self._metrics.agents_managed = len(self._managed_agents)
        return removed

    async def coordinate_agents(
        self,
        agent_ids: List[str],
        task: Dict[str, Any],
        pattern: Optional[CoordinationPattern] = None,
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents on a task.

        Args:
            agent_ids: Agents to coordinate
            task: Task definition
            pattern: Coordination pattern (uses default if not specified)

        Returns:
            Coordination result
        """
        if not self.coordinator:
            raise RuntimeError("Coordination layer not enabled")

        return await self.coordinator.coordinate_agents(
            agent_ids,
            task,
            pattern or self.config.coordination_pattern,
        )

    async def broadcast_event(self, event: Message) -> bool:
        """
        Broadcast event to all managed agents.

        Args:
            event: Event message to broadcast

        Returns:
            True if message was published
        """
        if not self.message_bus:
            raise RuntimeError("Message bus not enabled")

        result = await self.message_bus.publish(event)
        if result:
            self._metrics.messages_processed += 1
        return result

    async def schedule_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Schedule a task for execution.

        Args:
            task_type: Type of task
            payload: Task parameters
            priority: Task priority
            timeout_seconds: Task timeout

        Returns:
            Task ID
        """
        if not self.task_scheduler:
            raise RuntimeError("Task scheduler not enabled")

        task = Task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
        )

        task_id = await self.task_scheduler.schedule(task)
        self._metrics.tasks_pending = len(self.task_scheduler.get_pending_tasks())
        self._metrics.tasks_running = len(self.task_scheduler.get_running_tasks())

        return task_id

    async def check_safety(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
    ) -> bool:
        """
        Check safety constraints before operation.

        Args:
            operation_type: Type of operation
            parameters: Operation parameters

        Returns:
            True if operation is safe
        """
        if not self.safety_monitor:
            return True  # No safety monitoring = always safe

        context = OperationContext(
            operation_type=operation_type,
            agent_id=self.config.orchestrator_id,
            parameters=parameters,
        )

        validation = await self.safety_monitor.validate_operation(context)
        self._metrics.safety_validations += 1

        if not validation.is_safe:
            self._metrics.violations_detected += len(validation.violations)

        return validation.is_safe

    def add_safety_constraint(
        self,
        name: str,
        constraint_type: ConstraintType,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        level: SafetyLevel = SafetyLevel.MEDIUM,
    ) -> None:
        """
        Add a safety constraint.

        Args:
            name: Constraint name
            constraint_type: Type of constraint
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            level: Safety level
        """
        if not self.safety_monitor:
            raise RuntimeError("Safety monitoring not enabled")

        constraint = SafetyConstraint(
            name=name,
            constraint_type=constraint_type,
            min_value=min_value,
            max_value=max_value,
            level=level,
        )

        self.safety_monitor.add_constraint(constraint)

    async def run_saga(
        self,
        saga: Saga,
    ) -> Saga:
        """
        Execute a saga transaction.

        Args:
            saga: Saga to execute

        Returns:
            Completed saga with results
        """
        if not self.coordinator:
            raise RuntimeError("Coordination layer not enabled")

        return await self.coordinator.run_saga(saga)

    def _calculate_provenance_hash(
        self,
        input_data: InT,
        output_data: OutT,
        execution_id: str,
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            input_data: Input data
            output_data: Output data
            execution_id: Execution ID

        Returns:
            SHA-256 hash string
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        provenance_str = (
            f"{self.config.orchestrator_id}"
            f"{execution_id}"
            f"{str(input_data)}"
            f"{str(output_data)}"
            f"{timestamp}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _update_execution_metrics(
        self,
        execution_time_ms: float,
        success: bool,
    ) -> None:
        """Update execution metrics."""
        self._metrics.executions_total += 1

        if success:
            self._metrics.executions_successful += 1
        else:
            self._metrics.executions_failed += 1

        self._execution_times.append(execution_time_ms)
        if len(self._execution_times) > 1000:
            self._execution_times = self._execution_times[-1000:]

        self._metrics.avg_execution_time_ms = (
            sum(self._execution_times) / len(self._execution_times)
        )
        self._metrics.last_execution_time = datetime.now(timezone.utc).isoformat()
        self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

    def _get_tasks_executed(self) -> int:
        """Get number of tasks executed in last orchestration."""
        if self.task_scheduler:
            metrics = self.task_scheduler.get_metrics()
            return metrics.tasks_completed
        return 0

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: InT,
        execution_id: str,
    ) -> Optional[OrchestrationResult[OutT]]:
        """
        Handle error recovery.

        Subclasses can override this to implement custom recovery logic.

        Args:
            error: Exception that occurred
            input_data: Original input data
            execution_id: Execution ID

        Returns:
            Recovery result or None if recovery failed
        """
        self._state = OrchestratorState.RECOVERING
        self._metrics.current_state = self._state.value

        logger.warning(f"Attempting error recovery for {execution_id}")

        # Default recovery: just log and return failure
        # Subclasses can override for custom recovery
        self._state = OrchestratorState.READY
        self._metrics.current_state = self._state.value

        return None

    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self._state

    def get_metrics(self) -> OrchestratorMetrics:
        """Get current metrics."""
        # Update dynamic metrics
        if self.task_scheduler:
            self._metrics.tasks_pending = len(self.task_scheduler.get_pending_tasks())
            self._metrics.tasks_running = len(self.task_scheduler.get_running_tasks())

        self._metrics.agents_managed = len(self._managed_agents)
        return self._metrics

    def get_managed_agents(self) -> Dict[str, AgentInfo]:
        """Get all managed agents."""
        return self._managed_agents.copy()

    def get_config(self) -> OrchestratorConfig:
        """Get orchestrator configuration."""
        return self.config


# Factory function for creating simple orchestrators
def create_base_orchestrator_config(
    orchestrator_id: str,
    name: str = "DefaultOrchestrator",
    enable_all: bool = True,
) -> OrchestratorConfig:
    """
    Create a base orchestrator configuration.

    Args:
        orchestrator_id: Unique orchestrator ID
        name: Orchestrator name
        enable_all: Enable all features

    Returns:
        Configured OrchestratorConfig
    """
    return OrchestratorConfig(
        orchestrator_id=orchestrator_id,
        name=name,
        enable_message_bus=enable_all,
        enable_task_scheduling=enable_all,
        enable_coordination=enable_all,
        enable_safety_monitoring=enable_all,
    )


__all__ = [
    "BaseOrchestrator",
    "OrchestrationResult",
    "OrchestratorConfig",
    "OrchestratorMetrics",
    "OrchestratorState",
    "create_base_orchestrator_config",
]
