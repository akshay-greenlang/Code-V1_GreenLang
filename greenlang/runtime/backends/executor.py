# -*- coding: utf-8 -*-
"""
Pipeline Executor for GreenLang Runtime
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import yaml

from greenlang.determinism import DeterministicClock
from .base import (
    Pipeline,
    PipelineStep,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
)
from .factory import get_backend_manager

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """High-level pipeline executor with monitoring and management"""

    def __init__(
        self,
        backend_type: str = "local",
        backend_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize pipeline executor

        Args:
            backend_type: Type of backend to use
            backend_config: Backend configuration
        """
        self.backend_type = backend_type
        self.backend_config = backend_config or {}
        self.manager = get_backend_manager()

        # Execution history
        self.history: List[ExecutionResult] = []
        self.active_runs: Dict[str, ExecutionResult] = {}

        # Callbacks
        self.callbacks = {
            "on_start": [],
            "on_complete": [],
            "on_error": [],
            "on_step_complete": [],
        }

        logger.info(f"PipelineExecutor initialized with backend: {backend_type}")

    def execute(
        self,
        pipeline: Union[Pipeline, Dict, Path, str],
        context: Optional[ExecutionContext] = None,
        async_execution: bool = False,
    ) -> ExecutionResult:
        """
        Execute a pipeline

        Args:
            pipeline: Pipeline object, dict, or path to pipeline file
            context: Execution context (created if not provided)
            async_execution: Whether to execute asynchronously

        Returns:
            ExecutionResult
        """
        # Load pipeline if needed
        if isinstance(pipeline, (Path, str)):
            pipeline = self.load_pipeline(pipeline)
        elif isinstance(pipeline, dict):
            pipeline = Pipeline.from_dict(pipeline)

        # Create context if not provided
        if context is None:
            context = ExecutionContext()

        # Add executor metadata
        context.labels["executor"] = "greenlang"
        context.labels["backend"] = self.backend_type

        # Trigger callbacks
        self._trigger_callbacks("on_start", pipeline, context)

        # Track active run
        self.active_runs[context.run_id] = None

        try:
            if async_execution:
                # Execute asynchronously
                return asyncio.create_task(self._execute_async(pipeline, context))
            else:
                # Execute synchronously
                result = self._execute_sync(pipeline, context)
                return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result = ExecutionResult(
                run_id=context.run_id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.FAILED,
                start_time=DeterministicClock.utcnow(),
                errors=[str(e)],
            )

            self._trigger_callbacks("on_error", pipeline, context, result)
            return result

    def _execute_sync(
        self, pipeline: Pipeline, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute pipeline synchronously"""

        backend = self.manager.get_backend(self.backend_type, self.backend_config)
        result = backend.execute(pipeline, context)

        # Update tracking
        self.history.append(result)
        self.active_runs[context.run_id] = result

        # Trigger callbacks
        if result.status == ExecutionStatus.SUCCEEDED:
            self._trigger_callbacks("on_complete", pipeline, context, result)
        else:
            self._trigger_callbacks("on_error", pipeline, context, result)

        # Remove from active runs
        del self.active_runs[context.run_id]

        return result

    async def _execute_async(
        self, pipeline: Pipeline, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute pipeline asynchronously"""

        loop = asyncio.get_event_loop()

        # Run in executor to avoid blocking
        result = await loop.run_in_executor(None, self._execute_sync, pipeline, context)

        return result

    def load_pipeline(self, path: Union[Path, str]) -> Pipeline:
        """
        Load pipeline from file

        Args:
            path: Path to pipeline file (JSON or YAML)

        Returns:
            Pipeline object
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return Pipeline.from_dict(data)

    def save_pipeline(self, pipeline: Pipeline, path: Union[Path, str]):
        """
        Save pipeline to file

        Args:
            pipeline: Pipeline object
            path: Path to save pipeline
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(pipeline.to_dict(), f, default_flow_style=False)
            else:
                json.dump(pipeline.to_dict(), f, indent=2)

        logger.info(f"Pipeline saved to: {path}")

    def get_status(self, run_id: str) -> ExecutionStatus:
        """
        Get status of a pipeline run

        Args:
            run_id: Run ID

        Returns:
            ExecutionStatus
        """
        # Check active runs
        if run_id in self.active_runs:
            if self.active_runs[run_id]:
                return self.active_runs[run_id].status
            else:
                # Still starting
                return ExecutionStatus.PENDING

        # Check history
        for result in self.history:
            if result.run_id == run_id:
                return result.status

        # Check backend
        backend = self.manager.get_backend(self.backend_type, self.backend_config)
        return backend.get_status(run_id)

    def get_logs(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """
        Get logs for a pipeline run

        Args:
            run_id: Run ID
            step_name: Optional step name

        Returns:
            List of log lines
        """
        backend = self.manager.get_backend(self.backend_type, self.backend_config)
        return backend.get_logs(run_id, step_name)

    def cancel(self, run_id: str) -> bool:
        """
        Cancel a pipeline run

        Args:
            run_id: Run ID

        Returns:
            True if cancelled successfully
        """
        backend = self.manager.get_backend(self.backend_type, self.backend_config)
        success = backend.cancel(run_id)

        if success and run_id in self.active_runs:
            del self.active_runs[run_id]

        return success

    def cleanup(self, run_id: str) -> bool:
        """
        Cleanup resources for a pipeline run

        Args:
            run_id: Run ID

        Returns:
            True if cleaned up successfully
        """
        backend = self.manager.get_backend(self.backend_type, self.backend_config)
        return backend.cleanup(run_id)

    def register_callback(self, event: str, callback):
        """
        Register a callback for pipeline events

        Args:
            event: Event name (on_start, on_complete, on_error, on_step_complete)
            callback: Callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.info(f"Registered callback for event: {event}")
        else:
            logger.warning(f"Unknown event: {event}")

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger callbacks for an event"""

        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def get_history(self, limit: int = 10) -> List[ExecutionResult]:
        """
        Get execution history

        Args:
            limit: Maximum number of results

        Returns:
            List of ExecutionResults
        """
        return self.history[-limit:]

    def get_active_runs(self) -> Dict[str, ExecutionResult]:
        """Get currently active runs"""
        return self.active_runs.copy()


class PipelineBuilder:
    """Builder for creating pipelines programmatically"""

    def __init__(self, name: str):
        """
        Initialize pipeline builder

        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.description = ""
        self.labels = {}
        self.annotations = {}
        self.namespace = "default"
        self.service_account = None

    def add_step(
        self,
        name: str,
        command: List[str],
        args: List[str] = None,
        env: Dict[str, str] = None,
        image: str = "greenlang/executor:latest",
        depends_on: List[str] = None,
    ) -> "PipelineBuilder":
        """
        Add a step to the pipeline

        Args:
            name: Step name
            command: Command to execute
            args: Command arguments
            env: Environment variables
            image: Container image
            depends_on: Step dependencies

        Returns:
            Self for chaining
        """
        step = PipelineStep(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            image=image,
            depends_on=depends_on or [],
        )

        self.steps.append(step)
        return self

    def with_description(self, description: str) -> "PipelineBuilder":
        """Set pipeline description"""
        self.description = description
        return self

    def with_labels(self, labels: Dict[str, str]) -> "PipelineBuilder":
        """Set pipeline labels"""
        self.labels.update(labels)
        return self

    def with_annotations(self, annotations: Dict[str, str]) -> "PipelineBuilder":
        """Set pipeline annotations"""
        self.annotations.update(annotations)
        return self

    def with_namespace(self, namespace: str) -> "PipelineBuilder":
        """Set Kubernetes namespace"""
        self.namespace = namespace
        return self

    def with_service_account(self, service_account: str) -> "PipelineBuilder":
        """Set Kubernetes service account"""
        self.service_account = service_account
        return self

    def build(self) -> Pipeline:
        """
        Build the pipeline

        Returns:
            Pipeline object
        """
        return Pipeline(
            name=self.name,
            steps=self.steps,
            description=self.description,
            labels=self.labels,
            annotations=self.annotations,
            namespace=self.namespace,
            service_account=self.service_account,
        )


# Convenience functions
def create_pipeline(name: str) -> PipelineBuilder:
    """Create a new pipeline builder"""
    return PipelineBuilder(name)


def execute_pipeline(
    pipeline: Union[Pipeline, Dict, Path, str],
    backend: str = "local",
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """
    Execute a pipeline with default settings

    Args:
        pipeline: Pipeline to execute
        backend: Backend type
        context: Execution context

    Returns:
        ExecutionResult
    """
    executor = PipelineExecutor(backend_type=backend)
    return executor.execute(pipeline, context)
