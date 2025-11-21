# -*- coding: utf-8 -*-
"""
Saga - Long-running distributed transaction coordination.

This module implements the Saga pattern for managing distributed transactions
across multiple agents with compensation support for rollback scenarios.

Example:
    >>> saga = SagaOrchestrator(message_bus)
    >>> await saga.initialize()
    >>>
    >>> # Define saga transaction
    >>> transaction = SagaTransaction(
    ...     name="order_processing",
    ...     steps=[
    ...         SagaStep("validate_order", "agent-validator", compensation="cancel_validation"),
    ...         SagaStep("reserve_inventory", "agent-inventory", compensation="release_inventory"),
    ...         SagaStep("process_payment", "agent-payment", compensation="refund_payment"),
    ...         SagaStep("ship_order", "agent-shipping", compensation="cancel_shipment")
    ...     ]
    ... )
    >>> result = await saga.execute(transaction, initial_data)
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone
import uuid
import json
import hashlib
from dataclasses import dataclass

from prometheus_client import Counter, Histogram, Gauge

from .message_bus import MessageBus, Message, MessageType, Priority
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)

# Metrics
saga_execution_counter = Counter('saga_executions_total', 'Total saga executions', ['status'])
saga_step_counter = Counter('saga_steps_total', 'Saga steps executed', ['step_name', 'status'])
saga_compensation_counter = Counter('saga_compensations_total', 'Saga compensations executed')
saga_duration_histogram = Histogram('saga_duration_ms', 'Saga execution duration')


class SagaState(str, Enum):
    """Saga execution states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPENSATING = "COMPENSATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"
    ABORTED = "ABORTED"


class CompensationStrategy(str, Enum):
    """Compensation execution strategies."""
    BACKWARD = "BACKWARD"         # Compensate in reverse order
    FORWARD = "FORWARD"           # Compensate in forward order
    PARALLEL = "PARALLEL"         # Compensate all in parallel
    SELECTIVE = "SELECTIVE"       # Compensate only failed steps
    CASCADE = "CASCADE"           # Compensate with dependencies


class SagaStep(BaseModel):
    """Individual saga step definition."""

    step_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Step name")
    agent_id: str = Field(..., description="Agent to execute step")
    action: str = Field(..., description="Action to perform")
    compensation: Optional[str] = Field(None, description="Compensation action name")
    timeout_ms: int = Field(default=30000, ge=0)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3})
    depends_on: List[str] = Field(default_factory=list, description="Step dependencies")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_pivot: bool = Field(default=False, description="Pivot point - no compensation after this")


class CompensationAction(BaseModel):
    """Compensation action for rollback."""

    action_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Compensation action name")
    agent_id: str = Field(..., description="Agent to execute compensation")
    step_id: str = Field(..., description="Original step being compensated")
    payload: Dict[str, Any] = Field(..., description="Compensation payload")
    timeout_ms: int = Field(default=30000, ge=0)
    executed: bool = Field(default=False)
    result: Optional[Any] = Field(None)


class SagaTransaction(BaseModel):
    """Saga transaction definition."""

    transaction_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Transaction name")
    steps: List[SagaStep] = Field(..., description="Saga steps")
    compensation_strategy: CompensationStrategy = Field(default=CompensationStrategy.BACKWARD)
    isolation_level: str = Field(default="READ_COMMITTED")
    timeout_ms: int = Field(default=300000, ge=0, description="Overall transaction timeout")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('steps')
    def validate_steps(cls, v):
        """Validate saga steps."""
        if not v:
            raise ValueError("Saga must have at least one step")

        # Check for circular dependencies
        step_ids = {step.step_id for step in v}
        for step in v:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Unknown dependency {dep} in step {step.name}")

        return v


class SagaExecution(BaseModel):
    """Saga execution tracking."""

    execution_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    transaction: SagaTransaction = Field(...)
    state: SagaState = Field(default=SagaState.PENDING)
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(None)
    current_step: Optional[str] = Field(None)
    completed_steps: List[str] = Field(default_factory=list)
    failed_step: Optional[str] = Field(None)
    compensated_steps: List[str] = Field(default_factory=list)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    compensation_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    provenance_chain: List[str] = Field(default_factory=list)


class SagaLog(BaseModel):
    """Saga execution log entry."""

    log_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    execution_id: str = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = Field(..., description="Event type")
    step_name: Optional[str] = Field(None)
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(...)


@dataclass
class SagaConfig:
    """Saga orchestrator configuration."""
    enable_persistence: bool = True
    enable_distributed_locking: bool = True
    enable_event_sourcing: bool = True
    max_concurrent_sagas: int = 1000
    default_timeout_ms: int = 300000
    compensation_timeout_ms: int = 60000
    enable_metrics: bool = True


class SagaOrchestrator:
    """
    Saga orchestrator for distributed transactions.

    Manages long-running transactions across multiple agents with
    automatic compensation on failure.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        config: Optional[SagaConfig] = None
    ):
        """Initialize saga orchestrator."""
        self.message_bus = message_bus
        self.config = config or SagaConfig()

        # Execution tracking
        self.executions: Dict[str, SagaExecution] = {}
        self.logs: List[SagaLog] = []

        # Compensation tracking
        self.compensation_queue: List[CompensationAction] = []

        # Distributed locks
        self.locks: Dict[str, asyncio.Lock] = {}

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize saga orchestrator."""
        logger.info("Initializing SagaOrchestrator")

        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._compensation_processor())
        )
        self._tasks.append(
            asyncio.create_task(self._timeout_monitor())
        )

        logger.info("SagaOrchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown saga orchestrator."""
        logger.info("Shutting down SagaOrchestrator")
        self._running = False

        # Process remaining compensations
        await self._process_pending_compensations()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("SagaOrchestrator shutdown complete")

    async def execute(
        self,
        transaction: SagaTransaction,
        initial_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute saga transaction.

        Args:
            transaction: Saga transaction definition
            initial_data: Initial data for the transaction
            execution_id: Optional execution ID

        Returns:
            Transaction result

        Raises:
            SagaException: If transaction fails and cannot be compensated
        """
        execution = SagaExecution(
            execution_id=execution_id or str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            transaction=transaction,
            state=SagaState.RUNNING
        )

        self.executions[execution.execution_id] = execution
        start_time = datetime.now(timezone.utc)

        try:
            logger.info(f"Starting saga {transaction.name} (execution: {execution.execution_id})")

            # Log start event
            await self._log_event(execution, "SAGA_STARTED", details={"initial_data": initial_data})

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_transaction(execution, initial_data),
                timeout=transaction.timeout_ms / 1000
            )

            execution.state = SagaState.COMPLETED
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            # Log completion
            await self._log_event(execution, "SAGA_COMPLETED", details={"result": result})

            saga_execution_counter.labels(status="success").inc()
            return result

        except asyncio.TimeoutError:
            execution.state = SagaState.FAILED
            execution.errors.append(f"Transaction timeout after {transaction.timeout_ms}ms")
            saga_execution_counter.labels(status="timeout").inc()

            # Start compensation
            await self._compensate(execution)
            raise

        except Exception as e:
            execution.state = SagaState.FAILED
            execution.errors.append(str(e))
            saga_execution_counter.labels(status="failure").inc()

            logger.error(f"Saga {transaction.name} failed: {e}")

            # Start compensation
            await self._compensate(execution)
            raise

        finally:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            saga_duration_histogram.observe(duration_ms)

    async def _execute_transaction(
        self,
        execution: SagaExecution,
        initial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute transaction steps."""
        current_data = initial_data
        transaction = execution.transaction

        # Build execution order considering dependencies
        execution_order = self._build_execution_order(transaction.steps)

        for step_id in execution_order:
            step = next(s for s in transaction.steps if s.step_id == step_id)
            execution.current_step = step.name

            try:
                # Check if dependencies are satisfied
                for dep in step.depends_on:
                    if dep not in execution.completed_steps:
                        raise ValueError(f"Dependency {dep} not satisfied for step {step.name}")

                # Execute step
                logger.debug(f"Executing saga step {step.name}")
                result = await self._execute_step(step, current_data, execution)

                # Record result
                execution.step_results[step.step_id] = result
                execution.completed_steps.append(step.step_id)
                current_data = result.get("output", current_data)

                # Log step completion
                await self._log_event(
                    execution,
                    "STEP_COMPLETED",
                    step_name=step.name,
                    details={"result": result}
                )

                saga_step_counter.labels(step_name=step.name, status="success").inc()

                # Check if this is a pivot point
                if step.is_pivot:
                    logger.info(f"Reached pivot point at step {step.name} - compensation disabled")
                    execution.transaction.compensation_strategy = CompensationStrategy.SELECTIVE

            except Exception as e:
                execution.failed_step = step.step_id
                execution.errors.append(f"Step {step.name} failed: {str(e)}")

                # Log step failure
                await self._log_event(
                    execution,
                    "STEP_FAILED",
                    step_name=step.name,
                    details={"error": str(e)}
                )

                saga_step_counter.labels(step_name=step.name, status="failure").inc()
                raise

        return current_data

    async def _execute_step(
        self,
        step: SagaStep,
        input_data: Dict[str, Any],
        execution: SagaExecution
    ) -> Dict[str, Any]:
        """Execute individual saga step."""
        # Send message to agent
        message = Message(
            sender_id=f"saga-{execution.execution_id}",
            recipient_id=step.agent_id,
            message_type=MessageType.REQUEST,
            priority=Priority.HIGH,
            payload={
                "action": step.action,
                "data": input_data,
                "saga_id": execution.execution_id,
                "step_id": step.step_id,
                "metadata": step.metadata
            }
        )

        # Add provenance
        message.calculate_provenance(
            execution.provenance_chain[-1] if execution.provenance_chain else None
        )
        execution.provenance_chain.append(message.provenance.current_hash)

        # Execute with retry
        for attempt in range(step.retry_policy.get("max_attempts", 3)):
            try:
                response = await self.message_bus.request_response(
                    message,
                    timeout_ms=step.timeout_ms
                )

                if response:
                    return response.payload
                else:
                    raise TimeoutError(f"Step {step.name} timeout")

            except Exception as e:
                if attempt < step.retry_policy.get("max_attempts", 3) - 1:
                    backoff = step.retry_policy.get("backoff_ms", 1000) * (2 ** attempt)
                    logger.warning(f"Step {step.name} attempt {attempt + 1} failed, retrying in {backoff}ms")
                    await asyncio.sleep(backoff / 1000)
                else:
                    raise

    async def _compensate(self, execution: SagaExecution) -> None:
        """Execute compensation for failed transaction."""
        execution.state = SagaState.COMPENSATING
        transaction = execution.transaction

        try:
            logger.info(f"Starting compensation for saga {transaction.name}")

            # Log compensation start
            await self._log_event(execution, "COMPENSATION_STARTED")

            # Determine which steps to compensate
            steps_to_compensate = self._get_steps_to_compensate(execution)

            # Execute compensation based on strategy
            if transaction.compensation_strategy == CompensationStrategy.BACKWARD:
                await self._compensate_backward(execution, steps_to_compensate)
            elif transaction.compensation_strategy == CompensationStrategy.FORWARD:
                await self._compensate_forward(execution, steps_to_compensate)
            elif transaction.compensation_strategy == CompensationStrategy.PARALLEL:
                await self._compensate_parallel(execution, steps_to_compensate)
            elif transaction.compensation_strategy == CompensationStrategy.CASCADE:
                await self._compensate_cascade(execution, steps_to_compensate)
            else:
                await self._compensate_selective(execution, steps_to_compensate)

            execution.state = SagaState.COMPENSATED
            logger.info(f"Compensation completed for saga {transaction.name}")

            # Log compensation completion
            await self._log_event(execution, "COMPENSATION_COMPLETED")

            saga_compensation_counter.inc()

        except Exception as e:
            execution.state = SagaState.ABORTED
            execution.errors.append(f"Compensation failed: {str(e)}")
            logger.error(f"Compensation failed for saga {transaction.name}: {e}")

            # Log compensation failure
            await self._log_event(
                execution,
                "COMPENSATION_FAILED",
                details={"error": str(e)}
            )

    async def _compensate_backward(
        self,
        execution: SagaExecution,
        steps_to_compensate: List[SagaStep]
    ) -> None:
        """Compensate steps in reverse order."""
        for step in reversed(steps_to_compensate):
            if step.compensation:
                await self._execute_compensation(execution, step)

    async def _compensate_forward(
        self,
        execution: SagaExecution,
        steps_to_compensate: List[SagaStep]
    ) -> None:
        """Compensate steps in forward order."""
        for step in steps_to_compensate:
            if step.compensation:
                await self._execute_compensation(execution, step)

    async def _compensate_parallel(
        self,
        execution: SagaExecution,
        steps_to_compensate: List[SagaStep]
    ) -> None:
        """Compensate all steps in parallel."""
        tasks = []
        for step in steps_to_compensate:
            if step.compensation:
                tasks.append(self._execute_compensation(execution, step))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _compensate_cascade(
        self,
        execution: SagaExecution,
        steps_to_compensate: List[SagaStep]
    ) -> None:
        """Compensate with dependency cascade."""
        # Build dependency graph
        compensated = set()

        async def compensate_with_deps(step: SagaStep):
            if step.step_id in compensated:
                return

            # Compensate dependencies first
            for dep_id in step.depends_on:
                dep_step = next((s for s in steps_to_compensate if s.step_id == dep_id), None)
                if dep_step:
                    await compensate_with_deps(dep_step)

            # Compensate this step
            if step.compensation:
                await self._execute_compensation(execution, step)
                compensated.add(step.step_id)

        for step in reversed(steps_to_compensate):
            await compensate_with_deps(step)

    async def _compensate_selective(
        self,
        execution: SagaExecution,
        steps_to_compensate: List[SagaStep]
    ) -> None:
        """Compensate only specific steps."""
        # Only compensate steps that were completed before failure
        for step in reversed(steps_to_compensate):
            if step.step_id in execution.completed_steps and step.compensation:
                await self._execute_compensation(execution, step)

    async def _execute_compensation(
        self,
        execution: SagaExecution,
        step: SagaStep
    ) -> None:
        """Execute compensation for a step."""
        try:
            logger.debug(f"Compensating step {step.name}")

            # Get original step result for compensation context
            original_result = execution.step_results.get(step.step_id, {})

            # Create compensation action
            action = CompensationAction(
                name=step.compensation,
                agent_id=step.agent_id,
                step_id=step.step_id,
                payload={
                    "action": step.compensation,
                    "original_result": original_result,
                    "saga_id": execution.execution_id,
                    "step_id": step.step_id
                }
            )

            # Send compensation message
            message = Message(
                sender_id=f"saga-{execution.execution_id}",
                recipient_id=action.agent_id,
                message_type=MessageType.COMMAND,
                priority=Priority.HIGH,
                payload=action.payload
            )

            response = await self.message_bus.request_response(
                message,
                timeout_ms=self.config.compensation_timeout_ms
            )

            if response:
                action.executed = True
                action.result = response.payload
                execution.compensation_results[step.step_id] = response.payload
                execution.compensated_steps.append(step.step_id)

                # Log compensation
                await self._log_event(
                    execution,
                    "STEP_COMPENSATED",
                    step_name=step.name,
                    details={"result": response.payload}
                )
            else:
                raise TimeoutError(f"Compensation timeout for step {step.name}")

        except Exception as e:
            logger.error(f"Compensation failed for step {step.name}: {e}")
            raise

    def _get_steps_to_compensate(self, execution: SagaExecution) -> List[SagaStep]:
        """Get steps that need compensation."""
        transaction = execution.transaction
        steps_to_compensate = []

        # Find pivot point
        pivot_reached = False
        for step in transaction.steps:
            if step.is_pivot and step.step_id in execution.completed_steps:
                pivot_reached = True
                break

        # If pivot reached, no compensation needed
        if pivot_reached:
            return []

        # Get completed steps that need compensation
        for step in transaction.steps:
            if step.step_id in execution.completed_steps:
                steps_to_compensate.append(step)

        return steps_to_compensate

    def _build_execution_order(self, steps: List[SagaStep]) -> List[str]:
        """Build execution order considering dependencies."""
        # Topological sort for dependency resolution
        import networkx as nx

        graph = nx.DiGraph()
        for step in steps:
            graph.add_node(step.step_id)
            for dep in step.depends_on:
                graph.add_edge(dep, step.step_id)

        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Circular dependency detected
            raise ValueError("Circular dependency detected in saga steps")

    async def _log_event(
        self,
        execution: SagaExecution,
        event_type: str,
        step_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log saga event."""
        # Calculate provenance hash
        log_data = {
            "execution_id": execution.execution_id,
            "event_type": event_type,
            "step_name": step_name,
            "details": details or {}
        }
        provenance_hash = hashlib.sha256(
            json.dumps(log_data, sort_keys=True).encode()
        ).hexdigest()

        log_entry = SagaLog(
            execution_id=execution.execution_id,
            event_type=event_type,
            step_name=step_name,
            details=details or {},
            provenance_hash=provenance_hash
        )

        self.logs.append(log_entry)

        # Persist if enabled
        if self.config.enable_event_sourcing:
            await self._persist_log(log_entry)

    async def _persist_log(self, log_entry: SagaLog) -> None:
        """Persist log entry for event sourcing."""
        # Implementation would persist to database/event store
        pass

    async def _compensation_processor(self) -> None:
        """Process pending compensations."""
        while self._running:
            try:
                await self._process_pending_compensations()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Compensation processor error: {e}")

    async def _process_pending_compensations(self) -> None:
        """Process any pending compensations."""
        while self.compensation_queue:
            action = self.compensation_queue.pop(0)
            try:
                # Execute compensation
                message = Message(
                    sender_id="saga-compensator",
                    recipient_id=action.agent_id,
                    message_type=MessageType.COMMAND,
                    priority=Priority.CRITICAL,
                    payload=action.payload
                )

                await self.message_bus.publish(message)
                action.executed = True

            except Exception as e:
                logger.error(f"Failed to process compensation {action.action_id}: {e}")
                # Re-queue for retry
                self.compensation_queue.append(action)

    async def _timeout_monitor(self) -> None:
        """Monitor saga timeouts."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for execution in list(self.executions.values()):
                    if execution.state == SagaState.RUNNING:
                        started = datetime.fromisoformat(execution.started_at)
                        elapsed_ms = (now - started).total_seconds() * 1000

                        if elapsed_ms > execution.transaction.timeout_ms:
                            logger.warning(f"Saga {execution.execution_id} timeout")
                            execution.state = SagaState.FAILED
                            execution.errors.append("Transaction timeout")
                            await self._compensate(execution)

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Timeout monitor error: {e}")

    async def get_execution_status(
        self,
        execution_id: str
    ) -> Optional[SagaExecution]:
        """Get saga execution status."""
        return self.executions.get(execution_id)

    async def get_execution_logs(
        self,
        execution_id: str
    ) -> List[SagaLog]:
        """Get saga execution logs."""
        return [log for log in self.logs if log.execution_id == execution_id]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get saga orchestrator metrics."""
        completed = len([e for e in self.executions.values() if e.state == SagaState.COMPLETED])
        failed = len([e for e in self.executions.values() if e.state == SagaState.FAILED])
        compensated = len([e for e in self.executions.values() if e.state == SagaState.COMPENSATED])

        return {
            "total_executions": len(self.executions),
            "completed": completed,
            "failed": failed,
            "compensated": compensated,
            "success_rate": completed / len(self.executions) if self.executions else 0,
            "compensation_rate": compensated / failed if failed > 0 else 0,
            "pending_compensations": len(self.compensation_queue)
        }