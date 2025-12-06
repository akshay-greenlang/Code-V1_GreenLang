"""
Saga Orchestrator for GreenLang

This module implements the Saga pattern for distributed transactions
across GreenLang agents and microservices.

Features:
- Orchestration-based sagas
- Compensation handling
- State persistence
- Timeout management
- Retry policies
- Audit logging

Example:
    >>> orchestrator = SagaOrchestrator(config)
    >>> saga = await orchestrator.create_saga("emissions-processing")
    >>> saga.add_step("validate", validate_handler, compensate=rollback)
    >>> await orchestrator.execute_saga(saga)
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.events.event_schema import (
    BaseEvent,
    EventMetadata,
    EventType,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SagaStatus(str, Enum):
    """Saga execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class StepStatus(str, Enum):
    """Saga step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class SagaOrchestratorConfig:
    """Configuration for saga orchestrator."""
    storage_backend: str = "memory"
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://localhost/greenlang"
    default_timeout_seconds: int = 300
    default_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_persistence: bool = True
    enable_audit_events: bool = True


class SagaStepResult(BaseModel):
    """Result of a saga step execution."""
    success: bool = Field(..., description="Step succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Step result data")
    error: Optional[str] = Field(default=None, description="Error message")
    duration_ms: float = Field(default=0, description="Execution duration")


class SagaStep(BaseModel):
    """Definition of a saga step."""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Step name")
    order: int = Field(..., description="Execution order")
    status: StepStatus = Field(default=StepStatus.PENDING)
    handler: Optional[Callable] = Field(default=None, exclude=True)
    compensator: Optional[Callable] = Field(default=None, exclude=True)
    timeout_seconds: int = Field(default=60)
    retry_attempts: int = Field(default=3)
    retry_count: int = Field(default=0)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result: Optional[SagaStepResult] = Field(default=None)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Saga(BaseModel):
    """
    Saga definition and state.

    Represents a distributed transaction as a series of steps
    with compensation handlers for rollback.
    """
    saga_id: str = Field(default_factory=lambda: str(uuid4()))
    saga_type: str = Field(..., description="Type of saga")
    correlation_id: Optional[str] = Field(default=None)
    status: SagaStatus = Field(default=SagaStatus.PENDING)
    steps: List[SagaStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    timeout_seconds: int = Field(default=300)
    context: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")

    def add_step(
        self,
        name: str,
        handler: Callable,
        compensator: Optional[Callable] = None,
        timeout_seconds: int = 60,
        retry_attempts: int = 3
    ) -> "Saga":
        """
        Add a step to the saga.

        Args:
            name: Step name
            handler: Step handler function
            compensator: Compensation handler
            timeout_seconds: Step timeout
            retry_attempts: Max retries

        Returns:
            Self for chaining
        """
        step = SagaStep(
            name=name,
            order=len(self.steps),
            handler=handler,
            compensator=compensator,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
        )
        self.steps.append(step)
        return self

    def get_current_step(self) -> Optional[SagaStep]:
        """Get current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def calculate_provenance(self) -> str:
        """Calculate provenance hash for the saga."""
        data = {
            "saga_id": self.saga_id,
            "saga_type": self.saga_type,
            "steps": [s.name for s in self.steps],
            "created_at": self.created_at.isoformat(),
        }
        hash_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()


class SagaStorageBackend:
    """Base class for saga storage backends."""

    async def save(self, saga: Saga) -> None:
        """Save a saga."""
        raise NotImplementedError

    async def get(self, saga_id: str) -> Optional[Saga]:
        """Get a saga by ID."""
        raise NotImplementedError

    async def update(self, saga: Saga) -> None:
        """Update a saga."""
        raise NotImplementedError

    async def list_active(self) -> List[Saga]:
        """List active sagas."""
        raise NotImplementedError


class MemorySagaStorage(SagaStorageBackend):
    """In-memory saga storage for testing."""

    def __init__(self):
        """Initialize memory storage."""
        self._sagas: Dict[str, Saga] = {}

    async def save(self, saga: Saga) -> None:
        """Save a saga."""
        self._sagas[saga.saga_id] = saga

    async def get(self, saga_id: str) -> Optional[Saga]:
        """Get a saga by ID."""
        return self._sagas.get(saga_id)

    async def update(self, saga: Saga) -> None:
        """Update a saga."""
        self._sagas[saga.saga_id] = saga

    async def list_active(self) -> List[Saga]:
        """List active sagas."""
        return [
            s for s in self._sagas.values()
            if s.status in [SagaStatus.PENDING, SagaStatus.RUNNING, SagaStatus.COMPENSATING]
        ]


class SagaOrchestrator:
    """
    Saga orchestrator for distributed transactions.

    Manages saga lifecycle, step execution, and compensation
    for distributed transactions across GreenLang services.

    Attributes:
        config: Orchestrator configuration
        storage: Saga storage backend

    Example:
        >>> config = SagaOrchestratorConfig()
        >>> orchestrator = SagaOrchestrator(config)
        >>> saga = orchestrator.create_saga("order-processing")
        >>> saga.add_step("reserve", reserve_inventory)
        >>> saga.add_step("charge", charge_payment, compensator=refund)
        >>> saga.add_step("ship", create_shipment)
        >>> result = await orchestrator.execute(saga)
    """

    def __init__(self, config: SagaOrchestratorConfig):
        """
        Initialize saga orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self._storage: Optional[SagaStorageBackend] = None
        self._started = False
        self._event_producer = None
        self._recovery_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._metrics: Dict[str, int] = {
            "sagas_started": 0,
            "sagas_completed": 0,
            "sagas_failed": 0,
            "sagas_compensated": 0,
            "steps_executed": 0,
            "steps_compensated": 0,
        }

        logger.info("SagaOrchestrator initialized")

    async def start(self) -> None:
        """
        Start the saga orchestrator.

        Initializes storage and starts recovery task.
        """
        if self._started:
            logger.warning("Orchestrator already started")
            return

        try:
            # Initialize storage
            self._storage = self._create_storage()

            # Start recovery task
            self._recovery_task = asyncio.create_task(
                self._recovery_loop()
            )

            self._started = True
            self._shutdown = False

            logger.info("Saga orchestrator started")

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the saga orchestrator gracefully.
        """
        self._shutdown = True

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        self._started = False
        logger.info("Saga orchestrator stopped")

    def _create_storage(self) -> SagaStorageBackend:
        """Create storage backend."""
        if self.config.storage_backend == "memory":
            return MemorySagaStorage()
        return MemorySagaStorage()

    def create_saga(
        self,
        saga_type: str,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> Saga:
        """
        Create a new saga.

        Args:
            saga_type: Type of saga
            correlation_id: Correlation ID for tracing
            context: Initial context data
            timeout_seconds: Saga timeout

        Returns:
            New Saga instance
        """
        saga = Saga(
            saga_type=saga_type,
            correlation_id=correlation_id or str(uuid4()),
            context=context or {},
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
        )
        saga.provenance_hash = saga.calculate_provenance()
        return saga

    async def execute(
        self,
        saga: Saga,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Saga:
        """
        Execute a saga.

        Args:
            saga: Saga to execute
            initial_data: Initial data for first step

        Returns:
            Completed saga
        """
        self._ensure_started()

        if not saga.steps:
            raise ValueError("Saga has no steps")

        # Initialize saga
        saga.status = SagaStatus.RUNNING
        saga.started_at = datetime.utcnow()

        if initial_data:
            saga.context.update(initial_data)

        await self._storage.save(saga)
        self._metrics["sagas_started"] += 1

        await self._emit_saga_event(saga, EventType.SAGA_STARTED)

        try:
            # Execute steps forward
            saga = await self._execute_steps(saga)

            if saga.status == SagaStatus.RUNNING:
                saga.status = SagaStatus.COMPLETED
                saga.completed_at = datetime.utcnow()
                self._metrics["sagas_completed"] += 1
                await self._emit_saga_event(saga, EventType.SAGA_COMPLETED)

        except Exception as e:
            logger.error(f"Saga execution failed: {e}")
            saga.error = str(e)
            saga.status = SagaStatus.COMPENSATING
            saga = await self._compensate(saga)

        await self._storage.update(saga)
        return saga

    async def _execute_steps(self, saga: Saga) -> Saga:
        """Execute saga steps forward."""
        for i, step in enumerate(saga.steps):
            if step.status != StepStatus.PENDING:
                continue

            saga.current_step_index = i

            # Check timeout
            elapsed = (datetime.utcnow() - saga.started_at).total_seconds()
            if elapsed >= saga.timeout_seconds:
                saga.status = SagaStatus.TIMED_OUT
                saga.error = "Saga timeout exceeded"
                await self._storage.update(saga)
                raise TimeoutError("Saga timeout")

            # Execute step with retry
            step = await self._execute_step_with_retry(saga, step)
            saga.steps[i] = step

            if step.status == StepStatus.FAILED:
                saga.status = SagaStatus.COMPENSATING
                saga.error = step.result.error if step.result else "Step failed"
                await self._storage.update(saga)
                raise Exception(f"Step '{step.name}' failed")

            # Propagate output to context
            if step.output_data:
                saga.context.update(step.output_data)

            await self._storage.update(saga)

        return saga

    async def _execute_step_with_retry(
        self,
        saga: Saga,
        step: SagaStep
    ) -> SagaStep:
        """Execute a step with retry logic."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        step.input_data = dict(saga.context)

        while step.retry_count < step.retry_attempts:
            try:
                result = await self._execute_step(saga, step)
                step.result = result
                step.output_data = result.data or {}

                if result.success:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.utcnow()
                    self._metrics["steps_executed"] += 1
                    await self._emit_step_event(
                        saga, step, EventType.SAGA_STEP_COMPLETED
                    )
                    return step
                else:
                    raise Exception(result.error or "Step failed")

            except asyncio.TimeoutError:
                step.retry_count += 1
                logger.warning(
                    f"Step '{step.name}' timeout (retry {step.retry_count})"
                )
                if step.retry_count < step.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            except Exception as e:
                step.retry_count += 1
                logger.warning(
                    f"Step '{step.name}' failed: {e} (retry {step.retry_count})"
                )
                if step.retry_count < step.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        # Max retries exceeded
        step.status = StepStatus.FAILED
        step.completed_at = datetime.utcnow()
        await self._emit_step_event(saga, step, EventType.SAGA_STEP_FAILED)

        return step

    async def _execute_step(
        self,
        saga: Saga,
        step: SagaStep
    ) -> SagaStepResult:
        """Execute a single step."""
        if not step.handler:
            return SagaStepResult(success=True, data={})

        start_time = datetime.utcnow()

        try:
            if asyncio.iscoroutinefunction(step.handler):
                result = await asyncio.wait_for(
                    step.handler(saga.context),
                    timeout=step.timeout_seconds
                )
            else:
                result = step.handler(saga.context)

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            if isinstance(result, dict):
                return SagaStepResult(
                    success=True,
                    data=result,
                    duration_ms=duration
                )
            else:
                return SagaStepResult(
                    success=bool(result),
                    duration_ms=duration
                )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return SagaStepResult(
                success=False,
                error=str(e),
                duration_ms=duration
            )

    async def _compensate(self, saga: Saga) -> Saga:
        """Execute compensation for failed saga."""
        logger.info(f"Starting compensation for saga {saga.saga_id}")

        # Compensate in reverse order
        for i in range(saga.current_step_index, -1, -1):
            step = saga.steps[i]

            if step.status != StepStatus.COMPLETED:
                continue

            if not step.compensator:
                logger.warning(f"No compensator for step '{step.name}'")
                continue

            step.status = StepStatus.COMPENSATING

            try:
                if asyncio.iscoroutinefunction(step.compensator):
                    await asyncio.wait_for(
                        step.compensator(saga.context, step.output_data),
                        timeout=step.timeout_seconds
                    )
                else:
                    step.compensator(saga.context, step.output_data)

                step.status = StepStatus.COMPENSATED
                self._metrics["steps_compensated"] += 1
                logger.info(f"Compensated step '{step.name}'")

            except Exception as e:
                logger.error(f"Compensation failed for step '{step.name}': {e}")
                step.status = StepStatus.FAILED
                saga.status = SagaStatus.FAILED
                self._metrics["sagas_failed"] += 1
                await self._storage.update(saga)
                return saga

            saga.steps[i] = step
            await self._storage.update(saga)

        saga.status = SagaStatus.COMPENSATED
        saga.completed_at = datetime.utcnow()
        self._metrics["sagas_compensated"] += 1
        await self._emit_saga_event(saga, EventType.SAGA_COMPENSATED)

        return saga

    async def _recovery_loop(self) -> None:
        """Background loop for saga recovery."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute

                active_sagas = await self._storage.list_active()
                for saga in active_sagas:
                    # Check for timed out sagas
                    if saga.started_at:
                        elapsed = (datetime.utcnow() - saga.started_at).total_seconds()
                        if elapsed >= saga.timeout_seconds:
                            logger.warning(f"Recovering timed out saga: {saga.saga_id}")
                            saga.status = SagaStatus.COMPENSATING
                            saga.error = "Saga timeout - recovered"
                            await self._compensate(saga)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")

    async def _emit_saga_event(self, saga: Saga, event_type: EventType) -> None:
        """Emit saga lifecycle event."""
        if not self.config.enable_audit_events:
            return

        event = BaseEvent(
            event_type=event_type.value,
            metadata=EventMetadata(
                correlation_id=saga.correlation_id,
            ),
            data={
                "saga_id": saga.saga_id,
                "saga_type": saga.saga_type,
                "status": saga.status.value,
                "error": saga.error,
                "steps_count": len(saga.steps),
            },
        )

        logger.debug(f"Saga event: {event_type.value} for {saga.saga_id}")

    async def _emit_step_event(
        self,
        saga: Saga,
        step: SagaStep,
        event_type: EventType
    ) -> None:
        """Emit step lifecycle event."""
        if not self.config.enable_audit_events:
            return

        event = BaseEvent(
            event_type=event_type.value,
            metadata=EventMetadata(
                correlation_id=saga.correlation_id,
            ),
            data={
                "saga_id": saga.saga_id,
                "step_id": step.step_id,
                "step_name": step.name,
                "status": step.status.value,
                "retry_count": step.retry_count,
            },
        )

        logger.debug(f"Step event: {event_type.value} for {step.name}")

    async def get_saga(self, saga_id: str) -> Optional[Saga]:
        """
        Get a saga by ID.

        Args:
            saga_id: Saga identifier

        Returns:
            Saga or None
        """
        self._ensure_started()
        return await self._storage.get(saga_id)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "started": self._started,
            **self._metrics,
        }

    def _ensure_started(self) -> None:
        """Ensure orchestrator is started."""
        if not self._started:
            raise RuntimeError("Orchestrator not started")

    async def __aenter__(self) -> "SagaOrchestrator":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
