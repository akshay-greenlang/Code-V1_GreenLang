# -*- coding: utf-8 -*-
"""
GreenLang Saga Framework for GL-006 HeatRecoveryMaximizer.

This module provides saga pattern implementation for coordinating distributed
transactions and ensuring data consistency across heat recovery operations.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SagaStatus(str, Enum):
    """Saga execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Saga step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SagaStep:
    """
    Single step in a saga.

    Attributes:
        name: Step name
        execute_fn: Async function to execute the step
        compensate_fn: Async function to compensate/rollback
        status: Current step status
        result: Step execution result
        error: Error if step failed
        started_at: When step started
        completed_at: When step completed
    """
    name: str
    execute_fn: Callable[..., Any]
    compensate_fn: Optional[Callable[..., Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the step."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()

        try:
            if asyncio.iscoroutinefunction(self.execute_fn):
                self.result = await self.execute_fn(context)
            else:
                self.result = self.execute_fn(context)

            self.status = StepStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            return self.result

        except Exception as e:
            self.status = StepStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.utcnow()
            raise

    async def compensate(self, context: Dict[str, Any]) -> Any:
        """Execute compensation for this step."""
        if self.compensate_fn is None:
            self.status = StepStatus.SKIPPED
            return None

        self.status = StepStatus.COMPENSATING

        try:
            if asyncio.iscoroutinefunction(self.compensate_fn):
                result = await self.compensate_fn(context, self.result)
            else:
                result = self.compensate_fn(context, self.result)

            self.status = StepStatus.COMPENSATED
            return result

        except Exception as e:
            self.status = StepStatus.FAILED
            self.error = f"Compensation failed: {str(e)}"
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class SagaContext:
    """
    Context for saga execution.

    Carries data between saga steps and stores intermediate results.
    """
    saga_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set a value in context."""
        self.data[key] = value

    def get_step_result(self, step_name: str) -> Any:
        """Get result from a previous step."""
        return self.step_results.get(step_name)


class Saga:
    """
    Saga orchestrator for coordinating distributed operations.

    Implements the saga pattern with automatic compensation on failure.

    Example:
        >>> saga = Saga("heat_recovery_analysis")
        >>> saga.add_step("analyze_streams", analyze_fn, compensate_analyze)
        >>> saga.add_step("calculate_pinch", calculate_fn, compensate_calculate)
        >>> result = await saga.execute(initial_data)
    """

    def __init__(self, name: str, max_retries: int = 3):
        """
        Initialize the saga.

        Args:
            name: Saga name
            max_retries: Maximum retry attempts per step
        """
        self.name = name
        self.saga_id = str(uuid.uuid4())
        self.max_retries = max_retries
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.PENDING
        self.context: Optional[SagaContext] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self._logger = logging.getLogger(f"saga.{name}")

    def add_step(
        self,
        name: str,
        execute_fn: Callable[..., Any],
        compensate_fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Saga":
        """
        Add a step to the saga.

        Args:
            name: Step name
            execute_fn: Function to execute
            compensate_fn: Function to compensate
            metadata: Step metadata

        Returns:
            Self for chaining
        """
        step = SagaStep(
            name=name,
            execute_fn=execute_fn,
            compensate_fn=compensate_fn,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return self

    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> SagaContext:
        """
        Execute the saga.

        Args:
            initial_data: Initial data for the saga context

        Returns:
            SagaContext with results

        Raises:
            Exception: If saga fails and compensation is incomplete
        """
        self.context = SagaContext(
            saga_id=self.saga_id,
            data=initial_data or {},
        )
        self.status = SagaStatus.RUNNING
        self.started_at = datetime.utcnow()
        completed_steps: List[SagaStep] = []

        self._logger.info(f"Starting saga {self.name} ({self.saga_id})")

        try:
            for step in self.steps:
                self._logger.info(f"Executing step: {step.name}")

                # Execute with retry
                for attempt in range(self.max_retries):
                    try:
                        result = await step.execute(self.context.data)
                        self.context.step_results[step.name] = result
                        self.context.data[f"{step.name}_result"] = result
                        completed_steps.append(step)
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            raise
                        self._logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

            self.status = SagaStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self._logger.info(f"Saga {self.name} completed successfully")
            return self.context

        except Exception as e:
            self.error = str(e)
            self._logger.error(f"Saga {self.name} failed: {e}")

            # Execute compensation
            await self._compensate(completed_steps)

            self.completed_at = datetime.utcnow()
            raise

    async def _compensate(self, completed_steps: List[SagaStep]):
        """Execute compensation for completed steps in reverse order."""
        self.status = SagaStatus.COMPENSATING
        self._logger.info(f"Starting compensation for {len(completed_steps)} steps")

        # Compensate in reverse order
        for step in reversed(completed_steps):
            try:
                self._logger.info(f"Compensating step: {step.name}")
                await step.compensate(self.context.data)
            except Exception as e:
                self._logger.error(f"Compensation for {step.name} failed: {e}")
                # Continue with other compensations

        self.status = SagaStatus.COMPENSATED

    def get_status(self) -> Dict[str, Any]:
        """Get saga status."""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "steps": [step.to_dict() for step in self.steps],
        }


class HeatRecoverySaga(Saga):
    """
    Specialized saga for heat recovery operations.

    Provides pre-built steps for common heat recovery workflows.
    """

    def __init__(self, name: str = "heat_recovery"):
        """Initialize heat recovery saga."""
        super().__init__(name)

    def with_stream_analysis(
        self,
        analyze_fn: Callable,
        compensate_fn: Optional[Callable] = None,
    ) -> "HeatRecoverySaga":
        """Add stream analysis step."""
        return self.add_step("stream_analysis", analyze_fn, compensate_fn)

    def with_pinch_analysis(
        self,
        pinch_fn: Callable,
        compensate_fn: Optional[Callable] = None,
    ) -> "HeatRecoverySaga":
        """Add pinch analysis step."""
        return self.add_step("pinch_analysis", pinch_fn, compensate_fn)

    def with_network_synthesis(
        self,
        synthesis_fn: Callable,
        compensate_fn: Optional[Callable] = None,
    ) -> "HeatRecoverySaga":
        """Add network synthesis step."""
        return self.add_step("network_synthesis", synthesis_fn, compensate_fn)

    def with_roi_calculation(
        self,
        roi_fn: Callable,
        compensate_fn: Optional[Callable] = None,
    ) -> "HeatRecoverySaga":
        """Add ROI calculation step."""
        return self.add_step("roi_calculation", roi_fn, compensate_fn)

    def with_validation(
        self,
        validate_fn: Callable,
        compensate_fn: Optional[Callable] = None,
    ) -> "HeatRecoverySaga":
        """Add validation step."""
        return self.add_step("validation", validate_fn, compensate_fn)


__all__ = [
    'SagaStatus',
    'StepStatus',
    'SagaStep',
    'SagaContext',
    'Saga',
    'HeatRecoverySaga',
]
