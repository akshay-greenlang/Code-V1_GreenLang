# -*- coding: utf-8 -*-
"""
GreenLang Base Agent Framework
Enhanced base classes for agent development with lifecycle management,
metrics tracking, provenance integration, and resource loading.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Enhanced configuration for GreenLang agents."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of agent's purpose")
    version: str = Field(default="0.0.1", description="Agent version")
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific parameters"
    )
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    resource_paths: List[str] = Field(
        default_factory=list, description="Paths to resource files"
    )
    log_level: str = Field(default="INFO", description="Logging level")


class AgentMetrics(BaseModel):
    """Metrics collected during agent execution."""
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    input_size: int = Field(default=0, description="Size of input data")
    output_size: int = Field(default=0, description="Size of output data")
    records_processed: int = Field(default=0, description="Number of records processed")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Custom agent-specific metrics"
    )


class AgentResult(BaseModel):
    """Enhanced result from agent execution with metrics and provenance."""
    success: bool = Field(..., description="Whether the agent execution was successful")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    metrics: Optional[AgentMetrics] = Field(default=None, description="Execution metrics")
    provenance_id: Optional[str] = Field(default=None, description="Provenance record ID")
    timestamp: Optional[datetime] = Field(default=None, description="Execution timestamp")


class StatsTracker:
    """Tracks execution statistics for agents."""

    def __init__(self):
        self.executions = 0
        self.successes = 0
        self.failures = 0
        self.total_time_ms = 0.0
        self.custom_counters = defaultdict(int)
        self.custom_timers = defaultdict(float)

    def record_execution(self, success: bool, duration_ms: float):
        """Record an execution result."""
        self.executions += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_time_ms += duration_ms

    def increment(self, counter_name: str, value: int = 1):
        """Increment a custom counter."""
        self.custom_counters[counter_name] += value

    def add_time(self, timer_name: str, duration_ms: float):
        """Add time to a custom timer."""
        self.custom_timers[timer_name] += duration_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics as a dictionary."""
        success_rate = (self.successes / self.executions * 100) if self.executions > 0 else 0
        avg_time = (self.total_time_ms / self.executions) if self.executions > 0 else 0

        return {
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(success_rate, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
            "custom_counters": dict(self.custom_counters),
            "custom_timers": {k: round(v, 2) for k, v in self.custom_timers.items()}
        }


class BaseAgent(ABC):
    """
    Enhanced base class for all GreenLang agents.

    Provides:
    - Lifecycle management (init, validate, execute, cleanup)
    - Automatic metrics collection
    - Provenance tracking hooks
    - Resource loading
    - Structured logging
    - Error handling
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(
            name=self.__class__.__name__,
            description=self.__class__.__doc__ or "Base agent",
        )

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        # Initialize metrics tracker
        self.stats = StatsTracker()

        # Resource cache
        self._resources = {}

        # Lifecycle hooks
        self._pre_execute_hooks: List[Callable] = []
        self._post_execute_hooks: List[Callable] = []

        # Call initialization hook
        self.initialize()

    def initialize(self):
        """Override to add custom initialization logic."""
        pass

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Core execution logic - must be implemented by subclasses.

        Args:
            input_data: Input data dictionary

        Returns:
            AgentResult with execution results
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before execution.
        Override to add custom validation logic.
        """
        return True

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data before execution.
        Override to add custom preprocessing logic.
        """
        return input_data

    def postprocess(self, result: AgentResult) -> AgentResult:
        """
        Postprocess result after execution.
        Override to add custom postprocessing logic.
        """
        return result

    def cleanup(self):
        """
        Cleanup resources after execution.
        Override to add custom cleanup logic.
        """
        pass

    def load_resource(self, resource_path: str) -> Any:
        """
        Load a resource file (cached).

        Args:
            resource_path: Path to resource file

        Returns:
            Loaded resource data
        """
        if resource_path in self._resources:
            return self._resources[resource_path]

        path = Path(resource_path)
        if not path.exists():
            raise FileNotFoundError(f"Resource not found: {resource_path}")

        # Basic file loading - can be extended for different formats
        with open(path, 'r') as f:
            data = f.read()

        self._resources[resource_path] = data
        return data

    def add_pre_hook(self, hook: Callable):
        """Add a pre-execution hook."""
        self._pre_execute_hooks.append(hook)

    def add_post_hook(self, hook: Callable):
        """Add a post-execution hook."""
        self._post_execute_hooks.append(hook)

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent with full lifecycle management.

        Args:
            input_data: Input data dictionary

        Returns:
            AgentResult with execution results and metrics
        """
        start_time = time.time()
        metrics = AgentMetrics() if self.config.enable_metrics else None

        try:
            # Check if enabled
            if not self.config.enabled:
                return AgentResult(
                    success=False,
                    error=f"Agent {self.config.name} is disabled",
                    timestamp=DeterministicClock.now()
                )

            # Validate input
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Input validation failed",
                    timestamp=DeterministicClock.now()
                )

            # Record input size
            if metrics:
                metrics.input_size = len(str(input_data))

            # Run pre-execution hooks
            for hook in self._pre_execute_hooks:
                hook(self, input_data)

            # Preprocess
            processed_input = self.preprocess(input_data)

            # Execute
            self.logger.info(f"Executing {self.config.name}")
            result = self.execute(processed_input)

            # Postprocess
            result = self.postprocess(result)

            # Record output size
            if metrics:
                metrics.output_size = len(str(result.data))
                metrics.execution_time_ms = (time.time() - start_time) * 1000
                result.metrics = metrics

            # Set timestamp
            result.timestamp = DeterministicClock.now()

            # Run post-execution hooks
            for hook in self._post_execute_hooks:
                hook(self, result)

            # Record stats
            self.stats.record_execution(result.success, metrics.execution_time_ms if metrics else 0)

            return result

        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            duration_ms = (time.time() - start_time) * 1000
            self.stats.record_execution(False, duration_ms)

            return AgentResult(
                success=False,
                error=str(e),
                timestamp=DeterministicClock.now(),
                metrics=AgentMetrics(execution_time_ms=duration_ms) if self.config.enable_metrics else None
            )

        finally:
            # Always cleanup
            try:
                self.cleanup()
            except Exception as e:
                self.logger.warning(f"Cleanup failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.stats.get_stats()

    def reset_stats(self):
        """Reset execution statistics."""
        self.stats = StatsTracker()

    def __repr__(self):
        return f"{self.config.name}(version={self.config.version}, executions={self.stats.executions})"
