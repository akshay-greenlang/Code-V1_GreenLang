# -*- coding: utf-8 -*-
"""
Base Agent Framework for GreenLang Agents.

This module provides the foundational classes for building GreenLang agents with
standardized configuration, state management, lifecycle hooks, and execution patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class AgentCapability(str, Enum):
    """Agent capability types."""
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    PREDICTION = "prediction"
    CONTROL = "control"
    REPORTING = "reporting"


@dataclass
class AgentConfig:
    """
    Agent configuration container.

    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable agent name
        version: Agent version string
        description: Agent description
        capabilities: List of agent capabilities
        parameters: Custom configuration parameters
        timeout_seconds: Maximum execution timeout
        retry_count: Number of retries on failure
        deterministic: Whether agent should be deterministic
        seed: Random seed for reproducibility
    """
    agent_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    capabilities: List[AgentCapability] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    deterministic: bool = True
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "deterministic": self.deterministic,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
        capabilities = [
            AgentCapability(c) if isinstance(c, str) else c
            for c in data.get("capabilities", [])
        ]
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            capabilities=capabilities,
            parameters=data.get("parameters", {}),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 3),
            deterministic=data.get("deterministic", True),
            seed=data.get("seed", 42),
        )


@dataclass
class AgentState:
    """
    Agent execution state container.

    Attributes:
        status: Current agent status
        execution_id: Unique execution identifier
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
        inputs: Input data for current execution
        outputs: Output data from execution
        errors: List of errors encountered
        metrics: Execution metrics
        metadata: Additional state metadata
    """
    status: AgentStatus = AgentStatus.INITIALIZING
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the state."""
        self.errors.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "context": context or {},
        })

    def set_metric(self, key: str, value: Any):
        """Set a metric value."""
        self.metrics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "status": self.status.value,
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "errors": self.errors,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


# Type variable for input/output typing
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all GreenLang agents.

    This class provides the foundational infrastructure for building agents with
    standardized lifecycle management, error handling, and execution patterns.

    Example:
        >>> class MyAgent(BaseAgent[InputModel, OutputModel]):
        ...     async def execute(self, input_data: InputModel) -> OutputModel:
        ...         # Implementation
        ...         return OutputModel(result="success")
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with configuration.

        Args:
            config: Agent configuration object
        """
        self.config = config
        self.state = AgentState()
        self._hooks: Dict[str, List[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
            "on_error": [],
            "on_complete": [],
        }
        self._initialized = False
        self._logger = logging.getLogger(f"agent.{config.agent_id}")

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self.config.agent_id

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name

    @property
    def version(self) -> str:
        """Get the agent version."""
        return self.config.version

    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self.state.status

    async def initialize(self):
        """
        Initialize the agent.

        Override this method to perform custom initialization logic.
        """
        self._logger.info(f"Initializing agent {self.agent_id}")
        self.state.status = AgentStatus.INITIALIZING
        await self._on_initialize()
        self.state.status = AgentStatus.READY
        self._initialized = True
        self._logger.info(f"Agent {self.agent_id} initialized successfully")

    async def _on_initialize(self):
        """Hook for custom initialization logic."""
        pass

    @abstractmethod
    async def execute(self, input_data: TInput) -> TOutput:
        """
        Execute the agent's main logic.

        Args:
            input_data: Input data for the agent

        Returns:
            Output data from the agent
        """
        pass

    async def run(self, input_data: TInput) -> TOutput:
        """
        Run the agent with full lifecycle management.

        Args:
            input_data: Input data for the agent

        Returns:
            Output data from the agent
        """
        if not self._initialized:
            await self.initialize()

        self.state.execution_id = str(uuid.uuid4())
        self.state.started_at = datetime.utcnow()
        self.state.inputs = input_data if isinstance(input_data, dict) else {"data": input_data}
        self.state.status = AgentStatus.RUNNING
        self.state.errors = []

        try:
            # Run pre-execute hooks
            await self._run_hooks("pre_execute", input_data)

            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(input_data),
                timeout=self.config.timeout_seconds
            )

            # Store output
            self.state.outputs = result if isinstance(result, dict) else {"result": result}
            self.state.status = AgentStatus.COMPLETED

            # Run post-execute hooks
            await self._run_hooks("post_execute", result)
            await self._run_hooks("on_complete", result)

            return result

        except asyncio.TimeoutError as e:
            self.state.add_error(e, {"reason": "execution_timeout"})
            self.state.status = AgentStatus.FAILED
            await self._run_hooks("on_error", e)
            raise

        except Exception as e:
            self.state.add_error(e)
            self.state.status = AgentStatus.FAILED
            await self._run_hooks("on_error", e)
            raise

        finally:
            self.state.completed_at = datetime.utcnow()

    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register a lifecycle hook.

        Args:
            hook_name: Name of the hook (pre_execute, post_execute, on_error, on_complete)
            callback: Callback function to execute
        """
        if hook_name in self._hooks:
            self._hooks[hook_name].append(callback)

    async def _run_hooks(self, hook_name: str, data: Any):
        """Run all registered hooks for a given hook name."""
        for callback in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self._logger.warning(f"Hook {hook_name} failed: {e}")

    async def shutdown(self):
        """
        Shutdown the agent gracefully.

        Override this method to perform custom cleanup logic.
        """
        self._logger.info(f"Shutting down agent {self.agent_id}")
        self.state.status = AgentStatus.TERMINATED
        await self._on_shutdown()

    async def _on_shutdown(self):
        """Hook for custom shutdown logic."""
        pass

    def get_health(self) -> Dict[str, Any]:
        """
        Get agent health status.

        Returns:
            Health status dictionary
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "initialized": self._initialized,
            "execution_id": self.state.execution_id,
            "error_count": len(self.state.errors),
        }

    @asynccontextmanager
    async def execution_context(self, input_data: TInput):
        """
        Context manager for agent execution.

        Args:
            input_data: Input data for the agent

        Yields:
            Agent state during execution
        """
        await self.initialize()
        self.state.status = AgentStatus.RUNNING
        self.state.started_at = datetime.utcnow()
        try:
            yield self.state
        finally:
            self.state.completed_at = datetime.utcnow()
            if self.state.status == AgentStatus.RUNNING:
                self.state.status = AgentStatus.COMPLETED


__all__ = [
    'AgentStatus',
    'AgentCapability',
    'AgentConfig',
    'AgentState',
    'BaseAgent',
]
