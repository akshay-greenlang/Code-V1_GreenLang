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
import importlib
import inspect
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError, validator

from services.execution.provenance_tracker import ProvenanceTracker
from services.execution.cost_tracker import CostTracker, CostBreakdown

logger = logging.getLogger(__name__)


# Metrics counters for monitoring
class ExecutionMetricsCounter:
    """Thread-safe metrics counter for execution statistics."""

    def __init__(self):
        self._counters: Dict[str, int] = {
            "executions_total": 0,
            "executions_success": 0,
            "executions_failed": 0,
            "executions_timeout": 0,
            "executions_cancelled": 0,
        }
        self._gauges: Dict[str, float] = {
            "active_executions": 0,
            "avg_duration_ms": 0,
        }
        self._durations: List[float] = []

    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a counter."""
        if counter in self._counters:
            self._counters[counter] += value

    def set_gauge(self, gauge: str, value: float) -> None:
        """Set a gauge value."""
        if gauge in self._gauges:
            self._gauges[gauge] = value

    def record_duration(self, duration_ms: float) -> None:
        """Record an execution duration."""
        self._durations.append(duration_ms)
        # Keep last 1000 for rolling average
        if len(self._durations) > 1000:
            self._durations = self._durations[-1000:]
        self._gauges["avg_duration_ms"] = sum(self._durations) / len(self._durations)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
        }


# Global metrics instance
_metrics = ExecutionMetricsCounter()


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


class ExecutionRecord(BaseModel):
    """Persistent execution record for storage."""

    execution_id: str
    agent_id: str
    tenant_id: str
    user_id: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    input_hash: str
    output_hash: Optional[str] = None
    provenance_hash: str
    duration_ms: Optional[float] = None
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    compute_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    version_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0


class AgentLoader:
    """
    Agent loader for dynamically loading agent classes.

    Supports loading agents from:
    - Module paths (python://path.to.module:ClassName)
    - Registry lookup by agent_id
    """

    # Cache loaded agent classes
    _cache: Dict[str, Type] = {}

    # Known agent mappings (agent_id -> module path)
    AGENT_REGISTRY: Dict[str, str] = {
        "emissions/carbon_calculator_v1": "agents.gl_001_carbon_emissions.agent:CarbonEmissionsAgent",
        "gl-001": "agents.gl_001_carbon_emissions.agent:CarbonEmissionsAgent",
        "gl_001_carbon_emissions": "agents.gl_001_carbon_emissions.agent:CarbonEmissionsAgent",
        "csrd/reporting_agent_v1": "agents.gl_003_csrd_reporting.agent:CSRDReportingAgent",
        "gl-003": "agents.gl_003_csrd_reporting.agent:CSRDReportingAgent",
        "gl_003_csrd_reporting": "agents.gl_003_csrd_reporting.agent:CSRDReportingAgent",
        "eudr/compliance_agent_v1": "agents.gl_004_eudr_compliance.agent:EUDRComplianceAgent",
        "gl-004": "agents.gl_004_eudr_compliance.agent:EUDRComplianceAgent",
        "gl_004_eudr_compliance": "agents.gl_004_eudr_compliance.agent:EUDRComplianceAgent",
        "sbti/validation_agent_v1": "agents.gl_010_sbti_validation.agent:SBTiValidationAgent",
        "gl-010": "agents.gl_010_sbti_validation.agent:SBTiValidationAgent",
        "gl_010_sbti_validation": "agents.gl_010_sbti_validation.agent:SBTiValidationAgent",
        "carbon/offset_agent_v1": "agents.gl_012_carbon_offset.agent:CarbonOffsetAgent",
        "gl-012": "agents.gl_012_carbon_offset.agent:CarbonOffsetAgent",
        "gl_012_carbon_offset": "agents.gl_012_carbon_offset.agent:CarbonOffsetAgent",
        "sb253/disclosure_agent_v1": "agents.gl_013_sb253_disclosure.agent:SB253DisclosureAgent",
        "gl-013": "agents.gl_013_sb253_disclosure.agent:SB253DisclosureAgent",
        "gl_013_sb253_disclosure": "agents.gl_013_sb253_disclosure.agent:SB253DisclosureAgent",
    }

    @classmethod
    def load_agent_class(cls, agent_id: str) -> Type:
        """
        Load an agent class by ID.

        Args:
            agent_id: Agent identifier or module path

        Returns:
            Agent class

        Raises:
            ValueError: If agent not found
            ImportError: If module cannot be loaded
        """
        # Check cache first
        if agent_id in cls._cache:
            logger.debug(f"Loading agent {agent_id} from cache")
            return cls._cache[agent_id]

        # Resolve agent_id to module path
        if agent_id.startswith("python://"):
            module_path = agent_id[9:]  # Strip python:// prefix
        elif agent_id in cls.AGENT_REGISTRY:
            module_path = cls.AGENT_REGISTRY[agent_id]
        else:
            raise ValueError(
                f"Unknown agent: {agent_id}. "
                f"Available agents: {list(cls.AGENT_REGISTRY.keys())}"
            )

        # Parse module:class format
        if ":" not in module_path:
            raise ValueError(
                f"Invalid agent path: {module_path}. Expected format: module.path:ClassName"
            )

        module_name, class_name = module_path.rsplit(":", 1)

        try:
            # Import the module
            logger.debug(f"Importing module: {module_name}")
            module = importlib.import_module(module_name)

            # Get the class
            if not hasattr(module, class_name):
                raise ValueError(f"Class {class_name} not found in module {module_name}")

            agent_class = getattr(module, class_name)

            # Validate agent class has required interface
            if not cls._validate_agent_interface(agent_class):
                raise ValueError(
                    f"Agent class {class_name} does not implement required interface (run method)"
                )

            # Cache and return
            cls._cache[agent_id] = agent_class
            logger.info(f"Loaded agent class: {class_name} from {module_name}")
            return agent_class

        except ImportError as e:
            logger.error(f"Failed to import agent module {module_name}: {e}")
            raise ImportError(f"Cannot load agent module {module_name}: {e}") from e

    @classmethod
    def _validate_agent_interface(cls, agent_class: Type) -> bool:
        """
        Validate that agent class implements required interface.

        Required:
        - run() method that accepts input data
        """
        if not hasattr(agent_class, "run"):
            return False

        # Check run method signature
        run_method = getattr(agent_class, "run")
        if not callable(run_method):
            return False

        return True

    @classmethod
    def get_agent_input_model(cls, agent_class: Type) -> Optional[Type[BaseModel]]:
        """
        Get the input model for an agent.

        Args:
            agent_class: Agent class

        Returns:
            Input model class or None
        """
        # Try to get from run method signature
        if hasattr(agent_class, "run"):
            run_method = agent_class.run
            sig = inspect.signature(run_method)

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.annotation != inspect.Parameter.empty:
                    if isinstance(param.annotation, type) and issubclass(param.annotation, BaseModel):
                        return param.annotation

        # Try common naming conventions
        module = inspect.getmodule(agent_class)
        if module:
            for attr_name in dir(module):
                if "Input" in attr_name and attr_name.endswith("Input"):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseModel):
                        return attr

        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the agent class cache."""
        cls._cache.clear()
        logger.debug("Agent loader cache cleared")


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

        # Execution progress tracking (in-memory)
        self._execution_progress: Dict[str, Dict[str, Any]] = {}

        # Execution records (in-memory cache, persisted to state_store if available)
        self._execution_records: Dict[str, ExecutionRecord] = {}

        # Execution callbacks
        self._on_start_callbacks: List[Callable] = []
        self._on_complete_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []

        # Initialize metrics
        _metrics.set_gauge("active_executions", 0)

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
            input_size_bytes=len(json.dumps(input_data, default=str).encode())
        )

        # Update metrics - increment total and active
        _metrics.increment("executions_total")
        _metrics.set_gauge("active_executions", len(self._active_executions) + 1)

        logger.info(
            f"Starting execution {execution_id} for agent {agent_id}",
            extra={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "tenant_id": context.tenant_id,
                "correlation_id": context.correlation_id,
            }
        )

        # Initialize progress tracking
        self._execution_progress[execution_id] = {
            "progress": 0,
            "status": "Initializing",
            "updated_at": datetime.utcnow(),
        }

        # Initialize input hash for error handling
        input_hash = ""

        try:
            # Step 1: Validate input
            validation_result = await self._validate_input(agent_id, input_data)
            if not validation_result["is_valid"]:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")

            # Step 2: Track input provenance
            input_hash = self.provenance_tracker.track_input(input_data)

            # Step 3: Create initial execution record
            initial_record = ExecutionRecord(
                execution_id=execution_id,
                agent_id=agent_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                status=ExecutionStatus.RUNNING,
                input_data=input_data,
                input_hash=input_hash,
                provenance_hash=input_hash,  # Initial provenance is just input
                created_at=metrics.start_time,
                started_at=datetime.utcnow(),
                progress=0,
            )
            await self._store_execution_record(initial_record)

            # Step 4: Load and execute agent with timeout
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
                _metrics.increment("executions_timeout")
                _metrics.set_gauge("active_executions", len(self._active_executions) - 1)

                # Update execution record with timeout
                timeout_record = ExecutionRecord(
                    execution_id=execution_id,
                    agent_id=agent_id,
                    tenant_id=context.tenant_id,
                    user_id=context.user_id,
                    status=ExecutionStatus.TIMEOUT,
                    input_data=input_data,
                    input_hash=input_hash,
                    provenance_hash=input_hash,
                    error_message=f"Execution timed out after {context.timeout_seconds} seconds",
                    created_at=metrics.start_time,
                    started_at=initial_record.started_at,
                    completed_at=datetime.utcnow(),
                    duration_ms=(datetime.utcnow() - metrics.start_time).total_seconds() * 1000,
                )
                await self._store_execution_record(timeout_record)

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
                self._execution_progress.pop(execution_id, None)
                _metrics.set_gauge("active_executions", len(self._active_executions))

            # Step 5: Track output provenance
            output_hash = self.provenance_tracker.track_output(result_data)

            # Step 6: Build provenance chain
            provenance_hash = self.provenance_tracker.build_provenance_chain([
                {"step": "input", "hash": input_hash},
                {"step": "execution", "agent_id": agent_id},
                {"step": "output", "hash": output_hash},
            ])

            # Step 7: Calculate execution metrics
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.output_size_bytes = len(json.dumps(result_data, default=str).encode())

            # Step 8: Calculate cost
            cost = self.cost_tracker.calculate_execution_cost(
                execution_id=execution_id,
                duration_ms=metrics.duration_ms,
                llm_tokens_input=metrics.llm_tokens_input,
                llm_tokens_output=metrics.llm_tokens_output,
            )

            # Track cost per tenant and agent
            self.cost_tracker.track_tenant_cost(context.tenant_id, cost)
            self.cost_tracker.track_agent_cost(agent_id, cost)

            # Step 9: Update execution record with success
            success_record = ExecutionRecord(
                execution_id=execution_id,
                agent_id=agent_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                status=ExecutionStatus.COMPLETED,
                input_data=input_data,
                output_data=result_data,
                input_hash=input_hash,
                output_hash=output_hash,
                provenance_hash=provenance_hash,
                duration_ms=metrics.duration_ms,
                llm_tokens_input=metrics.llm_tokens_input,
                llm_tokens_output=metrics.llm_tokens_output,
                compute_cost_usd=cost.compute_cost_usd,
                llm_cost_usd=cost.llm_cost_usd,
                total_cost_usd=cost.total_usd,
                created_at=metrics.start_time,
                started_at=initial_record.started_at,
                completed_at=metrics.end_time,
                progress=100,
            )
            await self._store_execution_record(success_record)

            # Update metrics
            _metrics.increment("executions_success")
            _metrics.record_duration(metrics.duration_ms)

            # Fire completion callbacks
            await self._fire_callbacks(self._on_complete_callbacks, execution_id, agent_id, result_data)

            logger.info(
                f"Execution {execution_id} completed successfully in {metrics.duration_ms:.2f}ms",
                extra={
                    "execution_id": execution_id,
                    "agent_id": agent_id,
                    "duration_ms": metrics.duration_ms,
                    "cost_usd": cost.total_usd,
                }
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

            _metrics.increment("executions_failed")

            # Store failed execution record
            if not input_hash:
                input_hash = self.provenance_tracker.track_input(input_data)

            failed_record = ExecutionRecord(
                execution_id=execution_id,
                agent_id=agent_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                status=ExecutionStatus.FAILED,
                input_data=input_data,
                input_hash=input_hash,
                provenance_hash=input_hash,
                error_message=str(e),
                duration_ms=metrics.duration_ms,
                created_at=metrics.start_time,
                completed_at=metrics.end_time,
            )
            await self._store_execution_record(failed_record)

            await self._fire_callbacks(self._on_error_callbacks, execution_id, agent_id, str(e))

            return self._create_error_result(
                execution_id,
                agent_id,
                ExecutionStatus.FAILED,
                str(e),
                metrics,
                input_hash,
            )

        except Exception as e:
            logger.error(
                f"Execution {execution_id} failed with error: {str(e)}",
                exc_info=True
            )
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000

            _metrics.increment("executions_failed")

            # Store failed execution record
            if not input_hash:
                input_hash = self.provenance_tracker.track_input(input_data) if input_data else ""

            failed_record = ExecutionRecord(
                execution_id=execution_id,
                agent_id=agent_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                status=ExecutionStatus.FAILED,
                input_data=input_data,
                input_hash=input_hash,
                provenance_hash=input_hash,
                error_message=f"Execution failed: {str(e)}",
                duration_ms=metrics.duration_ms,
                created_at=metrics.start_time,
                completed_at=metrics.end_time,
            )
            await self._store_execution_record(failed_record)

            await self._fire_callbacks(self._on_error_callbacks, execution_id, agent_id, str(e))

            return self._create_error_result(
                execution_id,
                agent_id,
                ExecutionStatus.FAILED,
                f"Execution failed: {str(e)}",
                metrics,
                input_hash,
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

        This method loads the agent's input model (Pydantic) and validates
        the input data against it, providing detailed error messages.

        Args:
            agent_id: Agent to validate against
            input_data: Input data to validate

        Returns:
            Validation result with is_valid, errors, and validated_data
        """
        errors: List[str] = []

        # Basic validation
        if input_data is None:
            return {"is_valid": False, "errors": ["Input data cannot be None"], "validated_data": None}

        if not isinstance(input_data, dict):
            return {"is_valid": False, "errors": ["Input data must be a dictionary"], "validated_data": None}

        if not input_data:
            return {"is_valid": False, "errors": ["Input data cannot be empty"], "validated_data": None}

        try:
            # Load agent class
            agent_class = AgentLoader.load_agent_class(agent_id)

            # Get input model for schema validation
            input_model = AgentLoader.get_agent_input_model(agent_class)

            if input_model:
                try:
                    # Validate against Pydantic model
                    validated = input_model(**input_data)
                    logger.debug(f"Input validation passed for agent {agent_id}")
                    return {
                        "is_valid": True,
                        "errors": [],
                        "validated_data": validated,
                        "input_model": input_model,
                    }
                except ValidationError as e:
                    # Extract detailed errors from Pydantic
                    for error in e.errors():
                        field_path = ".".join(str(loc) for loc in error["loc"])
                        errors.append(f"{field_path}: {error['msg']}")

                    logger.warning(
                        f"Input validation failed for agent {agent_id}: {errors}"
                    )
                    return {"is_valid": False, "errors": errors, "validated_data": None}

            else:
                # No input model found - perform basic type checking
                logger.warning(
                    f"No input model found for agent {agent_id}, using basic validation"
                )
                return {
                    "is_valid": True,
                    "errors": [],
                    "validated_data": input_data,
                    "input_model": None,
                }

        except (ValueError, ImportError) as e:
            errors.append(f"Failed to load agent for validation: {str(e)}")
            return {"is_valid": False, "errors": errors, "validated_data": None}

    async def _execute_agent_internal(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        context: ExecutionContext,
        execution_id: str,
    ) -> Any:
        """
        Internal agent execution logic.

        This method:
        1. Loads the agent class from the registry
        2. Instantiates the agent with configuration
        3. Prepares input data (converts dict to Pydantic model if needed)
        4. Executes the agent's run() method
        5. Extracts result data (converts Pydantic model to dict)
        6. Updates execution progress

        Args:
            agent_id: Agent to execute
            input_data: Input data
            context: Execution context
            execution_id: Execution ID

        Returns:
            Agent execution result as dictionary

        Raises:
            ValueError: If agent not found or invalid
            RuntimeError: If execution fails
        """
        logger.info(
            f"Executing agent {agent_id} for execution {execution_id}",
            extra={
                "execution_id": execution_id,
                "agent_id": agent_id,
                "tenant_id": context.tenant_id,
            }
        )

        # Update progress: Loading agent
        await self._update_progress(execution_id, 10, "Loading agent")

        try:
            # Step 1: Load agent class
            agent_class = AgentLoader.load_agent_class(agent_id)
            logger.debug(f"Loaded agent class: {agent_class.__name__}")

            # Step 2: Get agent configuration
            agent_config = self._build_agent_config(agent_id, context)

            # Step 3: Instantiate agent
            try:
                agent_instance = agent_class(config=agent_config)
            except TypeError:
                # Agent may not accept config parameter
                agent_instance = agent_class()

            logger.debug(f"Instantiated agent: {agent_instance}")

            # Update progress: Preparing input
            await self._update_progress(execution_id, 20, "Preparing input")

            # Step 4: Prepare input data
            input_model = AgentLoader.get_agent_input_model(agent_class)
            if input_model:
                try:
                    prepared_input = input_model(**input_data)
                except ValidationError as e:
                    raise ValueError(f"Input validation failed: {e}")
            else:
                prepared_input = input_data

            # Update progress: Executing agent
            await self._update_progress(execution_id, 30, "Executing agent")

            # Step 5: Execute agent
            # Handle both sync and async run methods
            run_method = agent_instance.run
            if asyncio.iscoroutinefunction(run_method):
                result = await run_method(prepared_input)
            else:
                # Run sync method in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: run_method(prepared_input)
                )

            # Update progress: Processing result
            await self._update_progress(execution_id, 80, "Processing result")

            # Step 6: Extract result data
            if hasattr(result, "dict"):
                # Pydantic model
                result_data = result.dict()
            elif hasattr(result, "__dict__"):
                # Regular object
                result_data = {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
            elif isinstance(result, dict):
                result_data = result
            else:
                # Primitive value
                result_data = {"result": result}

            # Add execution metadata to result
            result_data["_execution_metadata"] = {
                "execution_id": execution_id,
                "agent_id": agent_id,
                "agent_version": getattr(agent_instance, "VERSION", "unknown"),
                "executed_at": datetime.utcnow().isoformat(),
            }

            # Update progress: Complete
            await self._update_progress(execution_id, 100, "Complete")

            logger.info(
                f"Agent {agent_id} execution completed successfully",
                extra={"execution_id": execution_id}
            )

            return result_data

        except ValueError as e:
            logger.error(f"Agent execution failed (ValueError): {e}", exc_info=True)
            raise

        except ImportError as e:
            logger.error(f"Agent import failed: {e}", exc_info=True)
            raise ValueError(f"Failed to load agent {agent_id}: {e}")

        except Exception as e:
            logger.error(
                f"Agent execution failed with unexpected error: {e}",
                exc_info=True,
                extra={"execution_id": execution_id, "agent_id": agent_id}
            )
            raise RuntimeError(f"Agent execution failed: {e}") from e

    def _build_agent_config(
        self,
        agent_id: str,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Build configuration for agent instantiation.

        Args:
            agent_id: Agent identifier
            context: Execution context

        Returns:
            Configuration dictionary
        """
        return {
            "agent_id": agent_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "correlation_id": context.correlation_id,
            "execution_timeout": context.timeout_seconds,
            **context.metadata,
            **self.config.get("agent_defaults", {}),
        }

    async def _update_progress(
        self,
        execution_id: str,
        progress: int,
        status_message: str,
    ) -> None:
        """
        Update execution progress.

        Args:
            execution_id: Execution identifier
            progress: Progress percentage (0-100)
            status_message: Human-readable status
        """
        # Update in state store if available
        if self.state_store:
            try:
                await self.state_store.update_progress(
                    execution_id,
                    progress,
                    status_message
                )
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")

        # Update in-memory progress tracking
        if execution_id in self._execution_progress:
            self._execution_progress[execution_id] = {
                "progress": progress,
                "status": status_message,
                "updated_at": datetime.utcnow(),
            }

        logger.debug(f"Execution {execution_id}: {progress}% - {status_message}")

    async def _store_execution_record(
        self,
        record: ExecutionRecord,
    ) -> None:
        """
        Store execution record for persistence and retrieval.

        Args:
            record: Execution record to store
        """
        # Store in memory cache
        self._execution_records[record.execution_id] = record

        # Persist to state store if available
        if self.state_store:
            try:
                await self.state_store.save_execution(record.dict())
            except Exception as e:
                logger.error(f"Failed to persist execution record: {e}")

        logger.debug(f"Stored execution record: {record.execution_id}")

    async def get_execution(
        self,
        execution_id: str,
    ) -> Optional[ExecutionRecord]:
        """
        Get execution record by ID.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution record or None if not found
        """
        # Check in-memory cache first
        if execution_id in self._execution_records:
            return self._execution_records[execution_id]

        # Check state store
        if self.state_store:
            try:
                data = await self.state_store.get_execution(execution_id)
                if data:
                    record = ExecutionRecord(**data)
                    self._execution_records[execution_id] = record
                    return record
            except Exception as e:
                logger.error(f"Failed to retrieve execution record: {e}")

        return None

    async def list_executions(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ExecutionRecord]:
        """
        List execution records with filtering.

        Args:
            agent_id: Filter by agent ID
            tenant_id: Filter by tenant ID
            status: Filter by status
            limit: Maximum records to return
            offset: Records to skip

        Returns:
            List of matching execution records
        """
        # Get from in-memory cache
        records = list(self._execution_records.values())

        # Apply filters
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]

        if tenant_id:
            records = [r for r in records if r.tenant_id == tenant_id]

        if status:
            records = [r for r in records if r.status == status]

        # Sort by created_at descending
        records.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        return records[offset:offset + limit]

    async def get_execution_result(
        self,
        execution_id: str,
    ) -> Optional[ExecutionResult]:
        """
        Get the full execution result.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution result or None
        """
        record = await self.get_execution(execution_id)
        if not record:
            return None

        # Build metrics from record
        metrics = ExecutionMetrics(
            start_time=record.created_at,
            end_time=record.completed_at,
            duration_ms=record.duration_ms,
            input_size_bytes=len(json.dumps(record.input_data).encode()) if record.input_data else 0,
            output_size_bytes=len(json.dumps(record.output_data).encode()) if record.output_data else 0,
            llm_tokens_input=record.llm_tokens_input,
            llm_tokens_output=record.llm_tokens_output,
        )

        # Build cost breakdown
        cost = CostBreakdown(
            execution_id=execution_id,
            compute_cost_usd=record.compute_cost_usd,
            llm_cost_usd=record.llm_cost_usd,
            total_usd=record.total_cost_usd,
        ) if record.total_cost_usd > 0 else None

        return ExecutionResult(
            execution_id=record.execution_id,
            agent_id=record.agent_id,
            status=record.status,
            result=record.output_data,
            error=record.error_message,
            provenance_hash=record.provenance_hash,
            metrics=metrics,
            cost=cost,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution service metrics.

        Returns:
            Dictionary of metrics
        """
        return _metrics.get_metrics()

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
