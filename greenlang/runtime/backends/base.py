"""
Base Classes for Runtime Backends
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import uuid

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Pipeline execution status"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ResourceRequirements:
    """Resource requirements for execution"""

    def __init__(
        self,
        cpu: str = "100m",
        memory: str = "128Mi",
        gpu: Optional[str] = None,
        ephemeral_storage: str = "1Gi",
    ):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.ephemeral_storage = ephemeral_storage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        resources = {
            "cpu": self.cpu,
            "memory": self.memory,
            "ephemeral-storage": self.ephemeral_storage,
        }
        if self.gpu:
            resources["nvidia.com/gpu"] = self.gpu
        return resources


@dataclass
class PipelineStep:
    """Single step in a pipeline"""

    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    image: str = "greenlang/executor:latest"
    resources: Optional[ResourceRequirements] = None
    timeout: int = 300  # seconds
    retry_count: int = 3
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "image": self.image,
            "resources": self.resources.to_dict() if self.resources else None,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "depends_on": self.depends_on,
        }


@dataclass
class Pipeline:
    """Pipeline definition"""

    name: str
    steps: List[PipelineStep]
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    namespace: str = "default"
    service_account: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "labels": self.labels,
            "annotations": self.annotations,
            "namespace": self.namespace,
            "service_account": self.service_account,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Create from dictionary"""
        steps = []
        for step_data in data.get("steps", []):
            resources = None
            if step_data.get("resources"):
                resources = ResourceRequirements(**step_data["resources"])

            step = PipelineStep(
                name=step_data["name"],
                command=step_data["command"],
                args=step_data.get("args", []),
                env=step_data.get("env", {}),
                image=step_data.get("image", "greenlang/executor:latest"),
                resources=resources,
                timeout=step_data.get("timeout", 300),
                retry_count=step_data.get("retry_count", 3),
                depends_on=step_data.get("depends_on", []),
            )
            steps.append(step)

        return cls(
            name=data["name"],
            steps=steps,
            description=data.get("description", ""),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
            namespace=data.get("namespace", "default"),
            service_account=data.get("service_account"),
        )


@dataclass
class ExecutionContext:
    """Execution context for pipelines"""

    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user: Optional[str] = None
    project: Optional[str] = None
    environment: str = "production"
    parameters: Dict[str, Any] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    config_maps: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "user": self.user,
            "project": self.project,
            "environment": self.environment,
            "parameters": self.parameters,
            "secrets": list(self.secrets.keys()),  # Don't expose secret values
            "config_maps": self.config_maps,
            "volumes": self.volumes,
            "labels": self.labels,
            "annotations": self.annotations,
            "created_at": self.created_at.isoformat(),
        }

    def to_env(self) -> Dict[str, str]:
        """Convert context to environment variables"""
        env = {
            "GL_PIPELINE_ID": self.pipeline_id,
            "GL_RUN_ID": self.run_id,
            "GL_ENVIRONMENT": self.environment,
        }

        if self.user:
            env["GL_USER"] = self.user
        if self.project:
            env["GL_PROJECT"] = self.project

        # Add parameters as environment variables
        for key, value in self.parameters.items():
            env_key = f"GL_PARAM_{key.upper()}"
            env[env_key] = (
                str(value) if not isinstance(value, (dict, list)) else json.dumps(value)
            )

        return env


@dataclass
class ExecutionResult:
    """Result of pipeline execution"""

    run_id: str
    pipeline_name: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "outputs": self.outputs,
            "logs": self.logs,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class Backend(ABC):
    """Abstract base class for execution backends"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backend

        Args:
            config: Backend configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")

    @abstractmethod
    def execute(self, pipeline: Pipeline, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a pipeline

        Args:
            pipeline: Pipeline to execute
            context: Execution context

        Returns:
            ExecutionResult
        """

    @abstractmethod
    def get_status(self, run_id: str) -> ExecutionStatus:
        """
        Get execution status

        Args:
            run_id: Run ID

        Returns:
            ExecutionStatus
        """

    @abstractmethod
    def get_logs(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """
        Get execution logs

        Args:
            run_id: Run ID
            step_name: Optional step name

        Returns:
            List of log lines
        """

    @abstractmethod
    def cancel(self, run_id: str) -> bool:
        """
        Cancel execution

        Args:
            run_id: Run ID

        Returns:
            True if cancelled successfully
        """

    @abstractmethod
    def cleanup(self, run_id: str) -> bool:
        """
        Cleanup execution resources

        Args:
            run_id: Run ID

        Returns:
            True if cleaned up successfully
        """

    def validate_pipeline(self, pipeline: Pipeline) -> List[str]:
        """
        Validate pipeline

        Args:
            pipeline: Pipeline to validate

        Returns:
            List of validation errors
        """
        errors = []

        if not pipeline.name:
            errors.append("Pipeline name is required")

        if not pipeline.steps:
            errors.append("Pipeline must have at least one step")

        step_names = set()
        for step in pipeline.steps:
            if not step.name:
                errors.append("Step name is required")
            elif step.name in step_names:
                errors.append(f"Duplicate step name: {step.name}")
            else:
                step_names.add(step.name)

            if not step.command:
                errors.append(f"Step {step.name} must have a command")

            # Check dependencies
            for dep in step.depends_on:
                if dep not in step_names:
                    errors.append(f"Step {step.name} depends on unknown step: {dep}")

        return errors

    def prepare_environment(self, context: ExecutionContext) -> Dict[str, str]:
        """
        Prepare environment variables

        Args:
            context: Execution context

        Returns:
            Environment variables
        """
        env = context.to_env()

        # Add backend-specific environment variables
        env["GL_BACKEND"] = self.name
        env["GL_BACKEND_CONFIG"] = json.dumps(self.config)

        return env
