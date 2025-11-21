# -*- coding: utf-8 -*-
"""
BaseAgent - Core abstract base class for all GreenLang agents.

This module provides the foundational BaseAgent class that all GreenLang agents
must inherit from. It implements lifecycle management, state tracking, error
handling, and configuration management following zero-hallucination principles.

Example:
    >>> from base_agent import BaseAgent, AgentConfig
    >>> config = AgentConfig(name="ESGAgent", version="1.0.0")
    >>> class MyAgent(BaseAgent):
    ...     def execute(self, input_data):
    ...         return self._process(input_data)
"""

import asyncio
import hashlib
import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from pathlib import Path

from pydantic import BaseModel, Field, validator, ValidationError
import json
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Configure module logger
logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"
    RECOVERING = "recovering"


class AgentPriority(int, Enum):
    """Agent execution priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class AgentConfig(BaseModel):
    """Configuration model for BaseAgent."""

    # Core configuration
    name: str = Field(..., description="Agent name identifier")
    version: str = Field("1.0.0", description="Agent version")
    agent_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))), description="Unique agent ID")

    # Behavior configuration
    priority: AgentPriority = Field(AgentPriority.NORMAL, description="Execution priority")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Execution timeout")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(5, ge=1, le=60, description="Delay between retries")

    # Resource configuration
    max_memory_mb: Optional[int] = Field(None, ge=128, description="Memory limit in MB")
    max_cpu_percent: Optional[float] = Field(None, ge=1.0, le=100.0, description="CPU usage limit")

    # Logging configuration
    log_level: str = Field("INFO", description="Logging level")
    enable_tracing: bool = Field(True, description="Enable execution tracing")
    enable_metrics: bool = Field(True, description="Enable metrics collection")

    # Persistence configuration
    checkpoint_enabled: bool = Field(False, description="Enable state checkpointing")
    checkpoint_interval_seconds: int = Field(60, ge=10, description="Checkpoint interval")
    state_directory: Optional[Path] = Field(None, description="State persistence directory")

    @validator('name')
    def validate_name(cls, v):
        """Validate agent name format."""
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Agent name must be alphanumeric with underscores/hyphens only")
        return v

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must be in semantic format (x.y.z)")
        return v


class AgentMetadata(BaseModel):
    """Metadata tracking for agent execution."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_executed_at: Optional[datetime] = None
    total_executions: int = Field(0, ge=0)
    total_errors: int = Field(0, ge=0)
    average_execution_time_ms: float = Field(0.0, ge=0.0)
    provenance_chain: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)


class ExecutionContext(BaseModel):
    """Context for agent execution."""

    execution_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    parent_execution_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = Field("production", description="Execution environment")
    debug_mode: bool = Field(False, description="Enable debug mode")
    dry_run: bool = Field(False, description="Dry run without side effects")
    custom_params: Dict[str, Any] = Field(default_factory=dict)


T = TypeVar('T')


class ExecutionResult(BaseModel, Generic[T]):
    """Result of agent execution."""

    success: bool = Field(..., description="Execution success status")
    result: Optional[T] = Field(None, description="Execution result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    execution_time_ms: float = Field(..., ge=0.0, description="Execution duration")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class BaseAgent(ABC):
    """
    Abstract base class for all GreenLang agents.

    This class provides core functionality for agent lifecycle management,
    state tracking, error handling, and execution orchestration. All agents
    must inherit from this class and implement the required abstract methods.

    Attributes:
        config: Agent configuration
        state: Current agent state
        metadata: Agent metadata and metrics
        _state_lock: Asyncio lock for state management
        _error_handlers: Registered error handlers
        _hooks: Lifecycle hooks

    Example:
        >>> class ESGCalculatorAgent(BaseAgent):
        ...     def _execute_core(self, input_data, context):
        ...         # Implementation
        ...         return processed_data
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize BaseAgent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.state = AgentState.UNINITIALIZED
        self.metadata = AgentMetadata()
        self._state_lock = asyncio.Lock()
        self._error_handlers: Dict[type, Callable] = {}
        self._hooks: Dict[str, List[Callable]] = {
            'pre_init': [],
            'post_init': [],
            'pre_execute': [],
            'post_execute': [],
            'on_error': [],
            'on_state_change': []
        }
        self._execution_history: List[ExecutionResult] = []
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up agent-specific logger."""
        agent_logger = logging.getLogger(f"{__name__}.{self.config.name}")
        agent_logger.setLevel(self.config.log_level)

        # Add agent context to log records
        for handler in agent_logger.handlers:
            handler.addFilter(lambda record: self._add_log_context(record))

        return agent_logger

    def _add_log_context(self, record: logging.LogRecord) -> bool:
        """Add agent context to log records."""
        record.agent_id = self.config.agent_id
        record.agent_name = self.config.name
        record.agent_state = self.state
        return True

    async def initialize(self) -> None:
        """
        Initialize the agent.

        Transitions agent from UNINITIALIZED to READY state.
        Executes pre_init and post_init hooks.

        Raises:
            RuntimeError: If agent is not in UNINITIALIZED state
            Exception: If initialization fails
        """
        if self.state != AgentState.UNINITIALIZED:
            raise RuntimeError(f"Cannot initialize agent in {self.state} state")

        async with self._state_lock:
            try:
                self._logger.info(f"Initializing agent {self.config.name}")
                await self._transition_state(AgentState.INITIALIZING)

                # Execute pre-init hooks
                await self._execute_hooks('pre_init')

                # Perform agent-specific initialization
                await self._initialize_core()

                # Load checkpoint if enabled
                if self.config.checkpoint_enabled and self.config.state_directory:
                    await self._load_checkpoint()

                # Execute post-init hooks
                await self._execute_hooks('post_init')

                await self._transition_state(AgentState.READY)
                self._logger.info(f"Agent {self.config.name} initialized successfully")

            except Exception as e:
                self._logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
                await self._transition_state(AgentState.ERROR)
                raise

    async def execute(self, input_data: Any, context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        Execute the agent with given input.

        This method orchestrates the complete execution lifecycle including
        pre/post hooks, error handling, retry logic, and metrics collection.

        Args:
            input_data: Input data for processing
            context: Optional execution context

        Returns:
            ExecutionResult containing output or error information

        Raises:
            RuntimeError: If agent is not in READY state
        """
        if self.state not in [AgentState.READY, AgentState.PAUSED]:
            raise RuntimeError(f"Cannot execute agent in {self.state} state")

        context = context or ExecutionContext()
        start_time = datetime.now(timezone.utc)

        async with self._state_lock:
            await self._transition_state(AgentState.EXECUTING)

        try:
            self._logger.info(f"Executing agent {self.config.name} with execution_id={context.execution_id}")

            # Execute pre-execute hooks
            await self._execute_hooks('pre_execute', input_data, context)

            # Execute with retry logic
            result = await self._execute_with_retry(input_data, context)

            # Calculate execution metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update metadata
            self.metadata.last_executed_at = datetime.now(timezone.utc)
            self.metadata.total_executions += 1
            self._update_average_execution_time(execution_time)

            # Create execution result
            execution_result = ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                provenance_hash=self._calculate_provenance(input_data, result, context),
                metrics=await self._collect_metrics()
            )

            # Execute post-execute hooks
            await self._execute_hooks('post_execute', execution_result)

            # Store in history
            self._execution_history.append(execution_result)
            if len(self._execution_history) > 100:  # Keep last 100 executions
                self._execution_history.pop(0)

            # Checkpoint if enabled
            if self.config.checkpoint_enabled:
                await self._save_checkpoint()

            self._logger.info(f"Agent execution completed successfully in {execution_time:.2f}ms")

            return execution_result

        except Exception as e:
            self._logger.error(f"Agent execution failed: {str(e)}", exc_info=True)

            # Update error metrics
            self.metadata.total_errors += 1
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Execute error hooks
            await self._execute_hooks('on_error', e, context)

            # Create error result
            error_result = ExecutionResult(
                success=False,
                error=str(e),
                error_details={
                    'type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'context': context.dict()
                },
                execution_time_ms=execution_time,
                provenance_hash=self._calculate_provenance(input_data, None, context),
                metrics=await self._collect_metrics()
            )

            self._execution_history.append(error_result)

            # Attempt recovery if configured
            if await self._should_recover(e):
                await self._recover_from_error(e)

            return error_result

        finally:
            async with self._state_lock:
                if self.state == AgentState.EXECUTING:
                    await self._transition_state(AgentState.READY)

    async def _execute_with_retry(self, input_data: Any, context: ExecutionContext) -> Any:
        """Execute with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self._logger.info(f"Retry attempt {attempt}/{self.config.max_retries}")
                    await asyncio.sleep(self.config.retry_delay_seconds)

                # Set timeout for execution
                return await asyncio.wait_for(
                    self._execute_core(input_data, context),
                    timeout=self.config.timeout_seconds
                )

            except asyncio.TimeoutError as e:
                last_error = e
                self._logger.warning(f"Execution timeout after {self.config.timeout_seconds}s")

            except Exception as e:
                last_error = e
                if not self._is_retryable(e):
                    raise

                self._logger.warning(f"Retryable error on attempt {attempt + 1}: {str(e)}")

        # All retries exhausted
        raise last_error or RuntimeError("Execution failed after all retries")

    def _is_retryable(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        # Override in subclasses for custom logic
        non_retryable = (ValidationError, ValueError, TypeError, NotImplementedError)
        return not isinstance(error, non_retryable)

    async def terminate(self) -> None:
        """
        Terminate the agent gracefully.

        Performs cleanup and transitions to TERMINATED state.
        """
        if self.state == AgentState.TERMINATED:
            return

        async with self._state_lock:
            self._logger.info(f"Terminating agent {self.config.name}")

            try:
                # Save final checkpoint if enabled
                if self.config.checkpoint_enabled:
                    await self._save_checkpoint()

                # Perform agent-specific cleanup
                await self._terminate_core()

                await self._transition_state(AgentState.TERMINATED)
                self._logger.info(f"Agent {self.config.name} terminated successfully")

            except Exception as e:
                self._logger.error(f"Error during agent termination: {str(e)}", exc_info=True)
                raise

    async def pause(self) -> None:
        """Pause agent execution."""
        if self.state != AgentState.EXECUTING:
            raise RuntimeError(f"Cannot pause agent in {self.state} state")

        async with self._state_lock:
            await self._transition_state(AgentState.PAUSED)
            self._logger.info(f"Agent {self.config.name} paused")

    async def resume(self) -> None:
        """Resume agent execution."""
        if self.state != AgentState.PAUSED:
            raise RuntimeError(f"Cannot resume agent in {self.state} state")

        async with self._state_lock:
            await self._transition_state(AgentState.READY)
            self._logger.info(f"Agent {self.config.name} resumed")

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a lifecycle hook.

        Args:
            hook_name: Name of the hook (pre_init, post_init, etc.)
            callback: Callback function to register
        """
        if hook_name not in self._hooks:
            raise ValueError(f"Invalid hook name: {hook_name}")

        self._hooks[hook_name].append(callback)
        self._logger.debug(f"Registered hook {hook_name}: {callback.__name__}")

    def register_error_handler(self, error_type: type, handler: Callable) -> None:
        """
        Register an error handler for specific exception type.

        Args:
            error_type: Exception type to handle
            handler: Handler function
        """
        self._error_handlers[error_type] = handler
        self._logger.debug(f"Registered error handler for {error_type.__name__}")

    async def _transition_state(self, new_state: AgentState) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
        """
        old_state = self.state
        self.state = new_state

        self._logger.debug(f"State transition: {old_state} -> {new_state}")

        # Execute state change hooks
        await self._execute_hooks('on_state_change', old_state, new_state)

    async def _execute_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Execute registered hooks."""
        for hook in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self, *args, **kwargs)
                else:
                    hook(self, *args, **kwargs)
            except Exception as e:
                self._logger.error(f"Error in hook {hook_name}: {str(e)}", exc_info=True)

    def _calculate_provenance(self, input_data: Any, output_data: Any, context: ExecutionContext) -> str:
        """
        Calculate SHA-256 hash for audit trail.

        Args:
            input_data: Input data
            output_data: Output data
            context: Execution context

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            'agent_id': self.config.agent_id,
            'agent_name': self.config.name,
            'agent_version': self.config.version,
            'execution_id': context.execution_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_hash': hashlib.sha256(str(input_data).encode()).hexdigest(),
            'output_hash': hashlib.sha256(str(output_data).encode()).hexdigest() if output_data else None
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _update_average_execution_time(self, new_time: float) -> None:
        """Update rolling average execution time."""
        n = self.metadata.total_executions
        current_avg = self.metadata.average_execution_time_ms

        # Calculate new average
        self.metadata.average_execution_time_ms = (current_avg * (n - 1) + new_time) / n

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect execution metrics."""
        metrics = {
            'total_executions': self.metadata.total_executions,
            'total_errors': self.metadata.total_errors,
            'error_rate': self.metadata.total_errors / max(1, self.metadata.total_executions),
            'average_execution_time_ms': self.metadata.average_execution_time_ms,
            'state': self.state,
            'uptime_seconds': (datetime.now(timezone.utc) - self.metadata.created_at).total_seconds()
        }

        # Add custom metrics from subclass
        custom_metrics = await self._collect_custom_metrics()
        metrics.update(custom_metrics)

        return metrics

    async def _should_recover(self, error: Exception) -> bool:
        """Determine if agent should attempt recovery."""
        # Check if we have a specific error handler
        for error_type, handler in self._error_handlers.items():
            if isinstance(error, error_type):
                return True

        # Default: don't recover from critical errors
        critical_errors = (SystemExit, KeyboardInterrupt, MemoryError)
        return not isinstance(error, critical_errors)

    async def _recover_from_error(self, error: Exception) -> None:
        """Attempt to recover from error."""
        async with self._state_lock:
            await self._transition_state(AgentState.RECOVERING)

        try:
            # Check for specific error handler
            for error_type, handler in self._error_handlers.items():
                if isinstance(error, error_type):
                    await handler(self, error)
                    break

            # Perform core recovery
            await self._recover_core(error)

            async with self._state_lock:
                await self._transition_state(AgentState.READY)

            self._logger.info(f"Successfully recovered from {type(error).__name__}")

        except Exception as recovery_error:
            self._logger.error(f"Recovery failed: {str(recovery_error)}", exc_info=True)
            async with self._state_lock:
                await self._transition_state(AgentState.ERROR)

    async def _save_checkpoint(self) -> None:
        """Save agent state checkpoint."""
        if not self.config.state_directory:
            return

        checkpoint_file = self.config.state_directory / f"{self.config.agent_id}_checkpoint.json"

        try:
            checkpoint_data = {
                'config': self.config.dict(),
                'metadata': self.metadata.dict(),
                'state': self.state,
                'custom_state': await self._get_custom_state()
            }

            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            self._logger.debug(f"Checkpoint saved to {checkpoint_file}")

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)

    async def _load_checkpoint(self) -> None:
        """Load agent state from checkpoint."""
        if not self.config.state_directory:
            return

        checkpoint_file = self.config.state_directory / f"{self.config.agent_id}_checkpoint.json"

        if not checkpoint_file.exists():
            self._logger.debug("No checkpoint file found")
            return

        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Restore metadata
            self.metadata = AgentMetadata(**checkpoint_data['metadata'])

            # Restore custom state
            await self._restore_custom_state(checkpoint_data.get('custom_state', {}))

            self._logger.info(f"Checkpoint loaded from {checkpoint_file}")

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    async def _initialize_core(self) -> None:
        """
        Perform agent-specific initialization.

        This method should initialize any resources, connections, or state
        required by the specific agent implementation.
        """
        pass

    @abstractmethod
    async def _execute_core(self, input_data: Any, context: ExecutionContext) -> Any:
        """
        Core execution logic for the agent.

        This method implements the main processing logic of the agent.
        Must follow zero-hallucination principles for any calculations.

        Args:
            input_data: Input data to process
            context: Execution context

        Returns:
            Processed output data
        """
        pass

    @abstractmethod
    async def _terminate_core(self) -> None:
        """
        Perform agent-specific cleanup.

        This method should clean up any resources, close connections, etc.
        """
        pass

    # Optional methods for subclasses to override

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect agent-specific metrics."""
        return {}

    async def _get_custom_state(self) -> Dict[str, Any]:
        """Get custom state for checkpointing."""
        return {}

    async def _restore_custom_state(self, state: Dict[str, Any]) -> None:
        """Restore custom state from checkpoint."""
        pass

    async def _recover_core(self, error: Exception) -> None:
        """Perform agent-specific recovery."""
        pass

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"<{self.__class__.__name__}(name={self.config.name}, state={self.state})>"


# Example implementation for testing
class ExampleAgent(BaseAgent):
    """Example agent implementation for testing."""

    async def _initialize_core(self) -> None:
        """Initialize example agent."""
        self._logger.info("Initializing ExampleAgent resources")
        # Initialize any specific resources

    async def _execute_core(self, input_data: Any, context: ExecutionContext) -> Any:
        """Execute example processing."""
        self._logger.info(f"Processing input: {input_data}")

        # Example: Simple transformation (deterministic, no hallucination)
        if isinstance(input_data, dict):
            result = {k: v * 2 if isinstance(v, (int, float)) else v for k, v in input_data.items()}
        else:
            result = input_data

        # Simulate some processing time
        await asyncio.sleep(0.1)

        return result

    async def _terminate_core(self) -> None:
        """Cleanup example agent."""
        self._logger.info("Cleaning up ExampleAgent resources")
        # Cleanup any specific resources


if __name__ == "__main__":
    # Basic usage example
    async def main():
        """Test the BaseAgent implementation."""
        config = AgentConfig(
            name="test_agent",
            version="1.0.0",
            timeout_seconds=10,
            max_retries=2
        )

        agent = ExampleAgent(config)

        # Initialize agent
        await agent.initialize()
        print(f"Agent initialized: {agent}")

        # Execute agent
        test_input = {"value": 10, "text": "hello"}
        result = await agent.execute(test_input)
        print(f"Execution result: {result.dict()}")

        # Terminate agent
        await agent.terminate()
        print(f"Agent terminated: {agent}")

    # Run the example
    asyncio.run(main())