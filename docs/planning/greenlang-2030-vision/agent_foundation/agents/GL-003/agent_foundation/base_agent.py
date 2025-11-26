# -*- coding: utf-8 -*-
"""
Base agent classes for GreenLang agents.

This module provides the foundational classes for implementing GreenLang agents
with support for deterministic execution, state management, and lifecycle hooks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    RECOVERING = "recovering"
    ERROR = "error"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class AgentConfig:
    """
    Configuration for GreenLang agents.

    Attributes:
        agent_id: Unique identifier for the agent (e.g., GL-003)
        name: Human-readable agent name
        version: Semantic version string
        deterministic: Whether agent uses deterministic execution
        temperature: LLM temperature (0.0 for deterministic)
        seed: Random seed for reproducibility
        timeout_seconds: Maximum execution time
        enable_metrics: Enable performance metrics collection
        checkpoint_enabled: Enable state checkpointing
        checkpoint_interval_seconds: Interval between checkpoints
        max_retries: Maximum retry attempts on failure
        state_directory: Directory for persisting state
    """
    agent_id: str
    name: str
    version: str
    deterministic: bool = True
    temperature: float = 0.0
    seed: int = 42
    timeout_seconds: int = 30
    enable_metrics: bool = True
    checkpoint_enabled: bool = True
    checkpoint_interval_seconds: int = 300
    max_retries: int = 3
    state_directory: Optional[Path] = None


class BaseAgent(ABC):
    """
    Abstract base class for all GreenLang agents.

    Provides common functionality including:
    - State management
    - Lifecycle hooks (initialize, execute, terminate)
    - Error handling and recovery
    - Performance metrics
    - Checkpointing

    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def execute(self, input_data):
        ...         return {"result": "processed"}
        ...
        >>> config = AgentConfig(agent_id="GL-XXX", name="MyAgent", version="1.0.0")
        >>> agent = MyAgent(config)
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.state = AgentState.IDLE
        self._metrics: Dict[str, Any] = {}
        self._initialized = False

        logger.info(f"BaseAgent initialized: {config.agent_id} v{config.version}")

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main processing logic.

        Args:
            input_data: Input data for processing

        Returns:
            Processing result dictionary
        """
        pass

    async def initialize(self) -> None:
        """Initialize agent resources."""
        self.state = AgentState.INITIALIZING
        try:
            await self._initialize_core()
            self._initialized = True
            self.state = AgentState.READY
            logger.info(f"Agent {self.config.agent_id} initialized successfully")
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent initialization failed: {e}")
            raise

    async def _initialize_core(self) -> None:
        """Override in subclass to perform agent-specific initialization."""
        pass

    async def terminate(self) -> None:
        """Terminate agent and release resources."""
        try:
            await self._terminate_core()
            self.state = AgentState.TERMINATED
            logger.info(f"Agent {self.config.agent_id} terminated successfully")
        except Exception as e:
            logger.error(f"Agent termination error: {e}")
            raise

    async def _terminate_core(self) -> None:
        """Override in subclass to perform agent-specific cleanup."""
        pass

    async def _execute_core(self, input_data: Any, context: Any) -> Any:
        """
        Core execution logic - override in subclass.

        Args:
            input_data: Input data for processing
            context: Execution context

        Returns:
            Processing result
        """
        return await self.execute(input_data)

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            'agent_id': self.config.agent_id,
            'state': self.state.value,
            'version': self.config.version,
            'initialized': self._initialized,
            'metrics': self._metrics.copy()
        }

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update agent metrics."""
        self._metrics.update(metrics)
