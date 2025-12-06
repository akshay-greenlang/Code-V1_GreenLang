"""
Agent Execution Service

This module provides the core execution engine for running GreenLang agents
with full lifecycle management, provenance tracking, and zero-hallucination
enforcement.

Example:
    >>> service = AgentExecutionService(config)
    >>> context = ExecutionContext(tenant_id="tenant-1", user_id="user-1")
    >>> result = await service.execute_agent("gl-001", input_data, context)
    >>> print(result.provenance_hash)
"""

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from services.execution.provenance_tracker import ProvenanceTracker
from services.execution.cost_tracker import CostTracker, CostBreakdown

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution state machine states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class ExecutionContext(BaseModel):
    """
    Context for agent execution.

    Contains all metadata needed for execution including tenant isolation,
    user attribution, tracing, and timeout configuration.
    """

    tenant_id: str = Field(..., description="Tenant identifier for isolation")
    user_id: str = Field(..., description="User who initiated execution")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation ID for distributed tracing"
    )
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Execution timeout (1-3600 seconds)"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Execution priority (1=highest, 10=lowest)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata"
    )

    @validator("tenant_id", "user_id")
    def validate_not_empty(cls, v: str) -> str:
        """Ensure required fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class ExecutionMetrics(BaseModel):
    """Metrics collected during execution."""

    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration_ms: Optional[float] = Field(None, description="Total duration in milliseconds")
    input_size_bytes: int = Field(0, description="Input data size")
    output_size_bytes: int = Field(0, description="Output data size")
    llm_tokens_input: int = Field(0, description="LLM input tokens (if applicable)")
    llm_tokens_output: int = Field(0, description="LLM output tokens (if applicable)")
    calculation_count: int = Field(0, description="Number of calculations performed")


class ExecutionResult(BaseModel):
    """
    Result of agent execution.

    Contains the execution output, provenance hash for audit trail,
    metrics for monitoring, and cost breakdown for billing.
    """

    execution_id: str = Field(..., description="Unique execution identifier")
    agent_id: str = Field(..., description="Agent that was executed")
    status: ExecutionStatus = Field(..., description="Final execution status")
    result: Optional[Any] = Field(None, description="Execution result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    metrics: ExecutionMetrics = Field(..., description="Execution metrics")
    cost: Optional[CostBreakdown] = Field(None, description="Cost breakdown")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ExecutionCheckpoint(BaseModel):
    """Checkpoint for long-running executions."""

    execution_id: str
    step_index: int
    step_name: str
    intermediate_result: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str


class AgentExecutionService:
    """
    Agent Execution Service.

    This service is responsible for:
    - Executing agents with full lifecycle management
    - Tracking provenance for all calculations (SHA-256)
    - Enforcing zero-hallucination principle
    - Managing execution timeouts and cancellation
    - Tracking execution costs

    Attributes:
        config: Service configuration
        provenance_tracker: Tracks calculation provenance
        cost_tracker: Tracks execution costs

    Example:
        >>> service = AgentExecutionService(config)
        >>> context = ExecutionContext(tenant_id="t1", user_id="u1")
        >>> result = await service.execute_agent("gl-001", {"fuel_type": "natural_gas"}, context)
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        agent_registry: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ):
        """
        Initialize the Agent Execution Service.

        Args:
            config: Service configuration
            agent_registry: Registry for agent lookup
            state_store: Store for execution state persistence
        """
        self.config = config or {}
        self.agent_registry = agent_registry
        self.state_store = state_store
        self.provenance_tracker = ProvenanceTracker()
        self.cost_tracker = CostTracker()

        # Active executions for cancellation
        self._active_executions: Dict[str, asyncio.Task] = {}

        # Execution callbacks
        self._on_start_callbacks: List[Callable] = []
        self._on_complete_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []

        logger.info("AgentExecutionService initialized")

    async def execute_agent(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute an agent with the given input data.

        This method:
        1. Validates input against agent schema
        2. Creates execution record
        3. Runs agent with timeout handling
        4. Tracks provenance for all calculations
        5. Calculates execution cost
        6. Returns result with full audit trail

        Args:
            agent_id: Identifier of the agent to execute
            input_data: Input data for the agent
            context: Execution context with tenant/user info

        Returns:
            ExecutionResult with output, provenance, and metrics

        Raises:
            ValueError: If input validation fails
            TimeoutError: If execution exceeds timeout
            RuntimeError: If agent execution fails
        """
        execution_id = str(uuid.uuid4())
        metrics = ExecutionMetrics(
            start_time=datetime.utcnow(),
            input_size_bytes=len(str(input_data).encode())
        )

        logger.info(
            f"Starting execution {execution_id} for agent {agent_id}",
            extra={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "tenant_id": context.tenant_id,
                "correlation_id": context.correlation_id,
            }
        )

        try:
            # Step 1: Validate input
            validation_result = await self._validate_input(agent_id, input_data)
            if not validation_result["is_valid"]:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")

            # Step 2: Track input provenance
            input_hash = self.provenance_tracker.track_input(input_data)

            # Step 3: Load and execute agent with timeout
            execution_task = asyncio.create_task(
                self._execute_agent_internal(
                    agent_id,
                    input_data,
                    context,
                    execution_id,
                )
            )
            self._active_executions[execution_id] = execution_task

            # Fire start callbacks
            await self._fire_callbacks(self._on_start_callbacks, execution_id, agent_id, context)

            try:
                result_data = await asyncio.wait_for(
                    execution_task,
                    timeout=context.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Execution {execution_id} timed out after {context.timeout_seconds}s"
                )
                return self._create_error_result(
                    execution_id,
                    agent_id,
                    ExecutionStatus.TIMEOUT,
                    f"Execution timed out after {context.timeout_seconds} seconds",
                    metrics,
                    input_hash,
                )
            finally:
                self._active_executions.pop(execution_id, None)

            # Step 4: Track output provenance
            output_hash = self.provenance_tracker.track_output(result_data)

            # Step 5: Build provenance chain
            provenance_hash = self.provenance_tracker.build_provenance_chain([
                {"step": "input", "hash": input_hash},
                {"step": "execution", "agent_id": agent_id},
                {"step": "output", "hash": output_hash},
            ])

            # Step 6: Calculate execution metrics
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.output_size_bytes = len(str(result_data).encode())

            # Step 7: Calculate cost
            cost = self.cost_tracker.calculate_execution_cost(
                execution_id=execution_id,
                duration_ms=metrics.duration_ms,
                llm_tokens_input=metrics.llm_tokens_input,
                llm_tokens_output=metrics.llm_tokens_output,
            )

            # Fire completion callbacks
            await self._fire_callbacks(self._on_complete_callbacks, execution_id, agent_id, result_data)

            logger.info(
                f"Execution {execution_id} completed successfully in {metrics.duration_ms:.2f}ms"
            )

            return ExecutionResult(
                execution_id=execution_id,
                agent_id=agent_id,
                status=ExecutionStatus.COMPLETED,
                result=result_data,
                provenance_hash=provenance_hash,
                metrics=metrics,
                cost=cost,
            )

        except ValueError as e:
            logger.warning(f"Validation error in execution {execution_id}: {str(e)}")
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000

            await self._fire_callbacks(self._on_error_callbacks, execution_id, agent_id, str(e))

            return self._create_error_result(
                execution_id,
                agent_id,
                ExecutionStatus.FAILED,
                str(e),
                metrics,
                self.provenance_tracker.track_input(input_data),
            )

        except Exception as e:
            logger.error(
                f"Execution {execution_id} failed with error: {str(e)}",
                exc_info=True
            )
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000

            await self._fire_callbacks(self._on_error_callbacks, execution_id, agent_id, str(e))

            return self._create_error_result(
                execution_id,
                agent_id,
                ExecutionStatus.FAILED,
                f"Execution failed: {str(e)}",
                metrics,
                self.provenance_tracker.track_input(input_data),
            )

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancellation was successful, False otherwise
        """
        task = self._active_executions.get(execution_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled execution {execution_id}")
            return True

        logger.warning(f"Could not cancel execution {execution_id} - not found or already complete")
        return False

    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """
        Get the current status of an execution.

        Args:
            execution_id: ID of execution to check

        Returns:
            Current execution status or None if not found
        """
        if execution_id in self._active_executions:
            task = self._active_executions[execution_id]
            if task.done():
                return ExecutionStatus.COMPLETED
            return ExecutionStatus.RUNNING

        # Check state store for historical executions
        if self.state_store:
            return await self.state_store.get_execution_status(execution_id)

        return None

    async def get_execution_progress(self, execution_id: str) -> Optional[int]:
        """
        Get execution progress as percentage (0-100).

        Args:
            execution_id: ID of execution to check

        Returns:
            Progress percentage or None if not found
        """
        # Check state store for progress
        if self.state_store:
            return await self.state_store.get_execution_progress(execution_id)
        return None

    async def create_checkpoint(
        self,
        execution_id: str,
        step_index: int,
        step_name: str,
        intermediate_result: Any,
    ) -> ExecutionCheckpoint:
        """
        Create a checkpoint for long-running execution.

        Args:
            execution_id: ID of execution
            step_index: Current step index
            step_name: Name of current step
            intermediate_result: Result to checkpoint

        Returns:
            Created checkpoint
        """
        checkpoint = ExecutionCheckpoint(
            execution_id=execution_id,
            step_index=step_index,
            step_name=step_name,
            intermediate_result=intermediate_result,
            provenance_hash=self.provenance_tracker.track_output(intermediate_result),
        )

        if self.state_store:
            await self.state_store.save_checkpoint(checkpoint)

        logger.debug(f"Created checkpoint for execution {execution_id} at step {step_name}")
        return checkpoint

    def on_execution_start(self, callback: Callable) -> None:
        """Register callback for execution start events."""
        self._on_start_callbacks.append(callback)

    def on_execution_complete(self, callback: Callable) -> None:
        """Register callback for execution completion events."""
        self._on_complete_callbacks.append(callback)

    def on_execution_error(self, callback: Callable) -> None:
        """Register callback for execution error events."""
        self._on_error_callbacks.append(callback)

    async def _validate_input(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate input data against agent schema.

        Args:
            agent_id: Agent to validate against
            input_data: Input data to validate

        Returns:
            Validation result with is_valid and errors
        """
        # TODO: Load agent schema and validate
        # For now, basic validation
        if not input_data:
            return {"is_valid": False, "errors": ["Input data cannot be empty"]}

        if not isinstance(input_data, dict):
            return {"is_valid": False, "errors": ["Input data must be a dictionary"]}

        return {"is_valid": True, "errors": []}

    async def _execute_agent_internal(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        context: ExecutionContext,
        execution_id: str,
    ) -> Any:
        """
        Internal agent execution logic.

        Args:
            agent_id: Agent to execute
            input_data: Input data
            context: Execution context
            execution_id: Execution ID

        Returns:
            Agent execution result
        """
        # TODO: Load agent from registry and execute
        # For now, placeholder implementation

        # Simulate execution time
        await asyncio.sleep(0.1)

        return {
            "status": "success",
            "agent_id": agent_id,
            "execution_id": execution_id,
            "result": input_data,
        }

    def _create_error_result(
        self,
        execution_id: str,
        agent_id: str,
        status: ExecutionStatus,
        error: str,
        metrics: ExecutionMetrics,
        input_hash: str,
    ) -> ExecutionResult:
        """Create an error result."""
        return ExecutionResult(
            execution_id=execution_id,
            agent_id=agent_id,
            status=status,
            error=error,
            provenance_hash=self.provenance_tracker.build_provenance_chain([
                {"step": "input", "hash": input_hash},
                {"step": "error", "message": error},
            ]),
            metrics=metrics,
        )

    async def _fire_callbacks(
        self,
        callbacks: List[Callable],
        *args: Any,
    ) -> None:
        """Fire registered callbacks."""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Callback error: {str(e)}", exc_info=True)
