# -*- coding: utf-8 -*-
"""
Saga orchestration for distributed agent workflows.

Implements the Saga pattern for managing long-running, distributed
transactions across multiple agents with compensation handling.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Saga execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class StepStatus(Enum):
    """Individual step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class SagaStep:
    """
    Individual step in a saga workflow.

    Attributes:
        step_id: Unique step identifier
        name: Human-readable step name
        action: Async function to execute
        compensation: Optional async function to rollback
        timeout_seconds: Step timeout
        retries: Number of retry attempts
        depends_on: List of step IDs this step depends on
    """
    step_id: str
    name: str
    action: Callable[..., Any]
    compensation: Optional[Callable[..., Any]] = None
    timeout_seconds: int = 30
    retries: int = 3
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            'step_id': self.step_id,
            'name': self.name,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'depends_on': self.depends_on
        }


@dataclass
class SagaContext:
    """
    Execution context for a saga.

    Stores intermediate results and state across steps.
    """
    saga_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        self.data[key] = value

    def get_step_result(self, step_id: str) -> Optional[Any]:
        """Get result from a completed step."""
        return self.step_results.get(step_id)


class SagaOrchestrator:
    """
    Orchestrator for managing saga workflows.

    Coordinates execution of multi-step workflows with automatic
    compensation (rollback) on failure.

    Example:
        >>> orchestrator = SagaOrchestrator()
        >>> orchestrator.add_step(SagaStep(
        ...     step_id="step1",
        ...     name="First Step",
        ...     action=async_action,
        ...     compensation=async_rollback
        ... ))
        >>> result = await orchestrator.execute(context)
    """

    def __init__(self, saga_id: Optional[str] = None):
        """
        Initialize saga orchestrator.

        Args:
            saga_id: Optional saga identifier
        """
        self.saga_id = saga_id or str(uuid.uuid4())
        self._steps: Dict[str, SagaStep] = {}
        self._step_order: List[str] = []
        self._status = SagaStatus.PENDING
        self._context: Optional[SagaContext] = None
        self._completed_steps: List[str] = []

        logger.info(f"SagaOrchestrator initialized: {self.saga_id}")

    def add_step(self, step: SagaStep) -> 'SagaOrchestrator':
        """
        Add a step to the saga.

        Args:
            step: Step to add

        Returns:
            Self for method chaining
        """
        self._steps[step.step_id] = step
        self._step_order.append(step.step_id)
        return self

    def add_steps(self, steps: List[SagaStep]) -> 'SagaOrchestrator':
        """
        Add multiple steps to the saga.

        Args:
            steps: List of steps to add

        Returns:
            Self for method chaining
        """
        for step in steps:
            self.add_step(step)
        return self

    async def execute(
        self,
        context: Optional[SagaContext] = None,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the saga workflow.

        Args:
            context: Optional execution context
            initial_data: Optional initial data

        Returns:
            Saga execution result
        """
        self._context = context or SagaContext(
            saga_id=self.saga_id,
            data=initial_data or {}
        )
        self._status = SagaStatus.RUNNING
        self._completed_steps = []

        logger.info(f"Starting saga execution: {self.saga_id}")

        try:
            # Execute steps in order
            for step_id in self._step_order:
                step = self._steps[step_id]

                # Check dependencies
                for dep_id in step.depends_on:
                    dep_step = self._steps.get(dep_id)
                    if dep_step and dep_step.status != StepStatus.COMPLETED:
                        step.status = StepStatus.SKIPPED
                        logger.warning(f"Skipping step {step_id}: dependency {dep_id} not completed")
                        continue

                # Execute step
                success = await self._execute_step(step)
                if not success:
                    # Initiate compensation
                    await self._compensate()
                    return self._create_result(SagaStatus.ROLLED_BACK)

                self._completed_steps.append(step_id)

            self._status = SagaStatus.COMPLETED
            logger.info(f"Saga completed successfully: {self.saga_id}")
            return self._create_result(SagaStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Saga execution failed: {e}")
            await self._compensate()
            return self._create_result(SagaStatus.FAILED, error=str(e))

    async def _execute_step(self, step: SagaStep) -> bool:
        """Execute a single step with retries."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)

        for attempt in range(step.retries):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    step.action(self._context),
                    timeout=step.timeout_seconds
                )

                step.result = result
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now(timezone.utc)

                # Store result in context
                self._context.step_results[step.step_id] = result

                logger.info(f"Step {step.step_id} completed successfully")
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Step {step.step_id} timed out (attempt {attempt + 1}/{step.retries})")
            except Exception as e:
                logger.warning(f"Step {step.step_id} failed (attempt {attempt + 1}/{step.retries}): {e}")
                step.error = str(e)

        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc)
        return False

    async def _compensate(self) -> None:
        """Run compensation for completed steps in reverse order."""
        self._status = SagaStatus.COMPENSATING
        logger.info(f"Starting saga compensation: {self.saga_id}")

        # Compensate in reverse order
        for step_id in reversed(self._completed_steps):
            step = self._steps[step_id]
            if step.compensation:
                try:
                    await asyncio.wait_for(
                        step.compensation(self._context),
                        timeout=step.timeout_seconds
                    )
                    step.status = StepStatus.COMPENSATED
                    logger.info(f"Step {step_id} compensated successfully")
                except Exception as e:
                    logger.error(f"Compensation failed for step {step_id}: {e}")

    def _create_result(
        self,
        status: SagaStatus,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create saga execution result."""
        return {
            'saga_id': self.saga_id,
            'status': status.value,
            'steps': [step.to_dict() for step in self._steps.values()],
            'context_data': self._context.data if self._context else {},
            'step_results': self._context.step_results if self._context else {},
            'completed_steps': self._completed_steps,
            'error': error
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current saga status."""
        return {
            'saga_id': self.saga_id,
            'status': self._status.value,
            'steps': {
                step_id: step.status.value
                for step_id, step in self._steps.items()
            },
            'completed_steps': self._completed_steps
        }
