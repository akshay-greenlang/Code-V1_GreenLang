# -*- coding: utf-8 -*-
"""
Executor Backend Base Interface
================================

Defines the protocol for GLIP v1 execution backends.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of a step execution."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELED = "canceled"


class ResourceProfile(BaseModel):
    """Resource requirements for a GLIP v1 agent."""
    cpu_request: str = Field(default="100m", description="CPU request (K8s format)")
    cpu_limit: str = Field(default="1000m", description="CPU limit (K8s format)")
    memory_request: str = Field(default="256Mi", description="Memory request (K8s format)")
    memory_limit: str = Field(default="2Gi", description="Memory limit (K8s format)")
    gpu_count: int = Field(default=0, description="Number of GPUs required")
    gpu_resource_key: str = Field(default="nvidia.com/gpu", description="K8s GPU resource key")
    ephemeral_storage_request: Optional[str] = Field(None, description="Ephemeral storage request")
    ephemeral_storage_limit: Optional[str] = Field(None, description="Ephemeral storage limit")

    def to_k8s_resources(self) -> Dict[str, Any]:
        """Convert to K8s resource specification."""
        resources = {
            "requests": {
                "cpu": self.cpu_request,
                "memory": self.memory_request,
            },
            "limits": {
                "cpu": self.cpu_limit,
                "memory": self.memory_limit,
            }
        }

        if self.gpu_count > 0:
            resources["limits"][self.gpu_resource_key] = str(self.gpu_count)
            resources["requests"][self.gpu_resource_key] = str(self.gpu_count)

        if self.ephemeral_storage_request:
            resources["requests"]["ephemeral-storage"] = self.ephemeral_storage_request
        if self.ephemeral_storage_limit:
            resources["limits"]["ephemeral-storage"] = self.ephemeral_storage_limit

        return resources


class RunContext(BaseModel):
    """
    GLIP v1 Input Envelope - The standardized input for all agents.

    This is written to GL_INPUT_URI and read by the agent container.
    """
    # Identity
    run_id: str = Field(..., description="Unique run identifier")
    step_id: str = Field(..., description="Unique step identifier within run")
    pipeline_id: str = Field(..., description="Pipeline definition ID")
    tenant_id: str = Field(..., description="Tenant/namespace identifier")

    # Agent binding
    agent_id: str = Field(..., description="Agent identifier (e.g., GL-MRV-X-001)")
    agent_version: str = Field(..., description="Agent version")
    schema_version: str = Field(default="1.0", description="GLIP schema version")

    # Parameters
    params: Dict[str, Any] = Field(default_factory=dict, description="Agent parameters from pipeline YAML")

    # Upstream artifacts
    inputs: Dict[str, "ArtifactReference"] = Field(
        default_factory=dict,
        description="URIs + checksums for upstream artifacts"
    )

    # Security context
    permissions_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="What this step is allowed to access"
    )

    # Execution control
    deadline_ts: Optional[datetime] = Field(None, description="Absolute deadline timestamp")
    timeout_seconds: int = Field(default=900, description="Timeout in seconds")
    retry_attempt: int = Field(default=0, description="Current retry attempt (0-indexed)")
    idempotency_key: str = Field(..., description="Idempotency key for this attempt")

    # Observability
    trace_id: str = Field(..., description="Distributed trace ID")
    span_id: str = Field(..., description="Span ID for this step")
    log_correlation_id: str = Field(..., description="Log correlation ID")

    def compute_hash(self) -> str:
        """Compute deterministic hash of this context (excluding timing)."""
        # Exclude volatile fields
        hashable = {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "pipeline_id": self.pipeline_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "params": self.params,
            "inputs": {k: v.model_dump() for k, v in self.inputs.items()},
        }
        json_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ArtifactReference(BaseModel):
    """Reference to an artifact with integrity verification."""
    uri: str = Field(..., description="S3 URI to artifact")
    checksum: str = Field(..., description="SHA-256 checksum")
    media_type: Optional[str] = Field(None, description="MIME type")
    size_bytes: Optional[int] = Field(None, description="Size in bytes")


class StepResult(BaseModel):
    """
    GLIP v1 Output - The standardized output from agents.

    Agent writes result.json to GL_OUTPUT_URI.
    """
    success: bool = Field(..., description="Whether step succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    artifacts: Dict[str, ArtifactReference] = Field(
        default_factory=dict,
        description="Named artifacts produced"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code (GL-E-*)")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class StepMetadata(BaseModel):
    """
    GLIP v1 Metadata - Execution metadata from agents.

    Agent writes step_metadata.json to GL_OUTPUT_URI.
    """
    agent_id: str = Field(..., description="Agent ID")
    agent_version: str = Field(..., description="Agent version")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: datetime = Field(..., description="Execution end time")
    duration_ms: float = Field(..., description="Duration in milliseconds")

    # Resource usage
    peak_memory_bytes: Optional[int] = Field(None, description="Peak memory usage")
    cpu_time_ms: Optional[float] = Field(None, description="CPU time consumed")

    # Artifact checksums
    result_checksum: str = Field(..., description="SHA-256 of result.json")
    artifacts_checksums: Dict[str, str] = Field(
        default_factory=dict,
        description="Checksums of each artifact file"
    )

    # Provenance
    input_context_hash: str = Field(..., description="Hash of input RunContext")
    idempotency_key: str = Field(..., description="Idempotency key used")

    # Status
    exit_code: int = Field(..., description="Container exit code")
    status: str = Field(..., description="Final status")


class ExecutionResult(BaseModel):
    """Result from executor backend."""
    step_id: str = Field(..., description="Step ID")
    status: ExecutionStatus = Field(..., description="Execution status")

    # GLIP v1 outputs
    result: Optional[StepResult] = Field(None, description="Step result from result.json")
    metadata: Optional[StepMetadata] = Field(None, description="Step metadata from step_metadata.json")

    # Execution details
    started_at: Optional[datetime] = Field(None, description="When execution started")
    completed_at: Optional[datetime] = Field(None, description="When execution completed")
    duration_ms: Optional[float] = Field(None, description="Total duration")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code (GL-E-*)")
    exit_code: Optional[int] = Field(None, description="Container exit code")

    # Lineage
    input_uri: str = Field(..., description="GL_INPUT_URI used")
    output_uri: str = Field(..., description="GL_OUTPUT_URI used")


class ExecutorBackend(ABC):
    """
    Abstract base class for GLIP v1 execution backends.

    Implementations:
        - K8sExecutor: Kubernetes Jobs
        - LocalExecutor: Local subprocess (dev)
        - LegacyHttpExecutor: HTTP wrapper for legacy agents
    """

    @abstractmethod
    async def execute(
        self,
        context: RunContext,
        container_image: str,
        resources: ResourceProfile,
        namespace: str,
        input_uri: str,
        output_uri: str,
    ) -> ExecutionResult:
        """
        Execute a GLIP v1 agent.

        Args:
            context: RunContext to write to input_uri
            container_image: Docker image for the agent
            resources: Resource profile for the container
            namespace: K8s namespace for execution
            input_uri: S3 URI for input (GL_INPUT_URI)
            output_uri: S3 URI for output (GL_OUTPUT_URI)

        Returns:
            ExecutionResult with status and outputs
        """
        pass

    @abstractmethod
    async def cancel(self, step_id: str, namespace: str) -> bool:
        """
        Cancel a running execution.

        Args:
            step_id: Step ID to cancel
            namespace: K8s namespace

        Returns:
            True if cancellation was initiated
        """
        pass

    @abstractmethod
    async def get_status(self, step_id: str, namespace: str) -> ExecutionStatus:
        """
        Get status of an execution.

        Args:
            step_id: Step ID to check
            namespace: K8s namespace

        Returns:
            Current execution status
        """
        pass

    @abstractmethod
    async def get_logs(
        self,
        step_id: str,
        namespace: str,
        tail_lines: Optional[int] = None
    ) -> str:
        """
        Get logs from an execution.

        Args:
            step_id: Step ID
            namespace: K8s namespace
            tail_lines: Number of lines to tail (None = all)

        Returns:
            Log output as string
        """
        pass

    def generate_idempotency_key(
        self,
        plan_hash: str,
        step_id: str,
        attempt: int
    ) -> str:
        """
        Generate idempotency key for a step attempt.

        Args:
            plan_hash: Hash of the execution plan
            step_id: Step ID
            attempt: Attempt number (0-indexed)

        Returns:
            Idempotency key string
        """
        content = f"{plan_hash}:{step_id}:{attempt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
