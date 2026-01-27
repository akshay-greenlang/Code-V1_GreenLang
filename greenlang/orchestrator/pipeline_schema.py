# -*- coding: utf-8 -*-
"""
GreenLang Pipeline YAML Schema v1
=================================

GL-FOUND-X-001: Declarative Pipeline YAML specification for GreenLang Orchestrator.

This module provides Pydantic models for validating and normalizing Pipeline YAML
definitions. The schema supports:

- Declarative pipeline definition with DAG-based step dependencies
- Parameter templating with Jinja2-style syntax
- Policy attachments for governance enforcement
- Output extraction via JSONPath expressions
- YAML anchor expansion and normalization
- Stable key ordering for deterministic hashing

Pipeline YAML Structure:
    apiVersion: greenlang/v1
    kind: Pipeline
    metadata:
      name: example-pipeline
      namespace: demo
    spec:
      parameters:
        input_uri:
          type: string
          required: true
      steps:
        - id: ingest
          agent: GL-DATA-X-001
          with:
            uri: "{{ params.input_uri }}"

Validation Features:
    - Unique step ID enforcement
    - DAG acyclicity verification
    - Agent reference format validation (GL-*-*-*)
    - Parameter template syntax validation
    - Dependency reference verification

Author: GreenLang Framework Team
Date: January 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants and Patterns
# ==============================================================================

# Supported API version
SUPPORTED_API_VERSION = "greenlang/v1"

# Agent ID pattern: GL-{CATEGORY}-{TYPE}-{NUMBER}
# Examples: GL-DATA-X-001, GL-CALC-A-002, GL-REPORT-X-003
AGENT_ID_PATTERN = re.compile(r"^GL-[A-Z]{2,8}-[A-Z]-\d{3}$")

# Parameter template pattern: {{ params.name }} or {{ steps.id.outputs.key }}
TEMPLATE_PATTERN = re.compile(
    r"\{\{\s*(params\.[a-zA-Z_][a-zA-Z0-9_]*|"
    r"steps\.[a-zA-Z_][a-zA-Z0-9_\-]*\.outputs\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
)

# JSONPath pattern for output extraction: $.field.subfield
JSONPATH_PATTERN = re.compile(r"^\$\.[\w\.\[\]]+$")


# ==============================================================================
# Enums
# ==============================================================================

class ParameterType(str, Enum):
    """Supported parameter types for pipeline parameters."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class PolicySeverity(str, Enum):
    """Policy violation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StepType(str, Enum):
    """Types of pipeline steps (for legacy compatibility)."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    PARALLEL = "parallel"
    CONDITION = "condition"
    LOOP = "loop"


class DataClassification(str, Enum):
    """Data classification levels for governance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# ==============================================================================
# Pipeline Metadata
# ==============================================================================

class PipelineMetadata(BaseModel):
    """
    Pipeline metadata containing identification and organizational information.

    Attributes:
        name: Unique pipeline name within namespace (required)
        namespace: Logical grouping for the pipeline (default: "default")
        labels: Key-value pairs for categorization and filtering
        annotations: Additional metadata for tooling and documentation
        description: Human-readable description of the pipeline
        version: Semantic version of the pipeline definition
        owner: Pipeline owner identifier
        team: Owning team identifier
        tags: List of tags for categorization
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Example:
        >>> metadata = PipelineMetadata(
        ...     name="carbon-footprint-pipeline",
        ...     namespace="production",
        ...     labels={"team": "sustainability", "domain": "scope3"}
        ... )
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=253,
        description="Pipeline name (unique within namespace)"
    )
    namespace: str = Field(
        default="default",
        min_length=1,
        max_length=63,
        description="Namespace for logical grouping"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Labels for categorization and filtering"
    )
    annotations: Dict[str, str] = Field(
        default_factory=dict,
        description="Annotations for tooling and documentation"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Human-readable pipeline description"
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the pipeline"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Pipeline owner identifier"
    )
    team: Optional[str] = Field(
        default=None,
        description="Owning team identifier"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate pipeline name is DNS-compatible."""
        if not re.match(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$", v):
            raise ValueError(
                f"Invalid pipeline name '{v}': must be lowercase alphanumeric "
                "with hyphens, starting and ending with alphanumeric"
            )
        return v

    @field_validator("namespace")
    @classmethod
    def validate_namespace_format(cls, v: str) -> str:
        """Validate namespace is DNS-compatible."""
        if not re.match(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$", v):
            raise ValueError(
                f"Invalid namespace '{v}': must be lowercase alphanumeric "
                "with hyphens, starting and ending with alphanumeric"
            )
        return v

    @field_validator("labels", "annotations")
    @classmethod
    def validate_label_keys(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate label and annotation key/value formats."""
        for key, value in v.items():
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\-\.]*$", key):
                raise ValueError(
                    f"Invalid label/annotation key '{key}': must start with letter or "
                    "underscore, followed by alphanumeric, underscore, hyphen, or dot"
                )
            if len(key) > 63:
                raise ValueError(f"Label/annotation key '{key}' exceeds 63 characters")
            if len(value) > 253:
                raise ValueError(f"Label/annotation value for '{key}' exceeds 253 characters")
        return v


# ==============================================================================
# Parameter Definition
# ==============================================================================

class ParameterDefinition(BaseModel):
    """
    Definition for a pipeline parameter.

    Parameters define inputs that can be provided at pipeline execution time
    and referenced in step configurations using template syntax.

    Attributes:
        type: Data type of the parameter
        required: Whether the parameter must be provided
        default: Default value if not provided
        description: Human-readable description
        enum: List of allowed values (for string/integer types)
        minimum: Minimum value (for integer/number types)
        maximum: Maximum value (for integer/number types)
        pattern: Regex pattern for validation (for string type)
        secret: Whether this parameter contains sensitive data
        validation_pattern: Regex pattern for validation (alias for pattern)

    Example:
        >>> param = ParameterDefinition(
        ...     type=ParameterType.STRING,
        ...     required=True,
        ...     description="S3 URI for input data"
        ... )
    """

    type: Union[ParameterType, str] = Field(
        default=ParameterType.STRING,
        description="Data type of the parameter"
    )
    required: bool = Field(
        default=False,
        description="Whether the parameter is required"
    )
    default: Optional[Any] = Field(
        default=None,
        description="Default value if parameter not provided"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable description"
    )
    enum: Optional[List[Union[str, int, float]]] = Field(
        default=None,
        description="Allowed values for the parameter"
    )
    minimum: Optional[Union[int, float]] = Field(
        default=None,
        description="Minimum value (for numeric types)"
    )
    maximum: Optional[Union[int, float]] = Field(
        default=None,
        description="Maximum value (for numeric types)"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for validation (for string type)"
    )
    secret: bool = Field(
        default=False,
        description="Whether this parameter contains sensitive data"
    )
    validation_pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for validation (alias for pattern)"
    )
    # Legacy field for backward compatibility
    name: Optional[str] = Field(
        default=None,
        description="Parameter name (used in list-based parameter definitions)"
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> Union[ParameterType, str]:
        """Normalize type to ParameterType enum if possible."""
        if isinstance(v, str):
            try:
                return ParameterType(v.lower())
            except ValueError:
                return v
        return v

    @model_validator(mode="after")
    def validate_parameter_constraints(self) -> "ParameterDefinition":
        """Validate parameter constraints are consistent with type."""
        # Use pattern from validation_pattern if pattern not set
        if self.pattern is None and self.validation_pattern is not None:
            object.__setattr__(self, "pattern", self.validation_pattern)

        # Validate minimum <= maximum
        if (self.minimum is not None and self.maximum is not None and
            self.minimum > self.maximum):
            raise ValueError(
                f"minimum ({self.minimum}) cannot exceed maximum ({self.maximum})"
            )

        # Validate pattern is valid regex
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        return self


# ==============================================================================
# Policy Attachment
# ==============================================================================

class PolicyAttachment(BaseModel):
    """
    Policy attachment for governance enforcement on a step.

    Policies can enforce constraints such as data validation rules,
    compliance requirements, or security checks.

    Attributes:
        name: Policy name (must match registered policy)
        severity: Severity level if policy is violated
        params: Parameters to pass to the policy
        enabled: Whether the policy is active

    Example:
        >>> policy = PolicyAttachment(
        ...     name="no_pii_export",
        ...     severity=PolicySeverity.ERROR,
        ...     params={"allow_hashed": True}
        ... )
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Policy name"
    )
    severity: PolicySeverity = Field(
        default=PolicySeverity.ERROR,
        description="Severity level for policy violations"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the policy"
    )
    enabled: bool = Field(
        default=True,
        description="Whether the policy is active"
    )


# ==============================================================================
# Resource Requirements (Legacy)
# ==============================================================================

class ResourceRequirements(BaseModel):
    """Resource requirements for a step."""

    cpu: Optional[str] = Field(None, description="CPU request (e.g., '100m', '1')")
    memory: Optional[str] = Field(None, description="Memory request (e.g., '256Mi')")
    gpu: Optional[int] = Field(None, ge=0, description="GPU count")
    timeout_seconds: int = Field(default=300, ge=1, le=86400, description="Timeout")


# ==============================================================================
# Artifact Definition (Legacy)
# ==============================================================================

class ArtifactDefinition(BaseModel):
    """Definition of an artifact produced or consumed by a step."""

    name: str = Field(..., description="Artifact name")
    path: str = Field(..., description="Path in artifact store")
    type: str = Field(default="file", description="Artifact type")
    classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification"
    )
    retention_days: Optional[int] = Field(None, ge=1, description="Retention period")


# ==============================================================================
# Step Definition
# ==============================================================================

class StepDefinition(BaseModel):
    """
    Definition for a pipeline step.

    Steps are the fundamental units of work in a pipeline. Each step executes
    an agent with specified inputs and produces outputs that can be consumed
    by downstream steps.

    Supports both the new v1 schema format and legacy format for backward
    compatibility.

    Attributes:
        id: Unique step identifier within the pipeline (v1 schema)
        name: Step name (legacy, alias for id)
        agent: Agent ID to execute (format: GL-*-*-*)
        agent_id: Agent ID (legacy alias for agent)
        dependsOn: List of step IDs this step depends on (v1 schema)
        depends_on: List of dependencies (legacy alias for dependsOn)
        with_: Input parameters for the agent (aliased from 'with')
        inputs: Input parameters (legacy alias for with_)
        outputs: Output extraction mapping (key -> JSONPath) or list of output names
        policy: List of policy attachments
        timeoutSeconds: Maximum execution time in seconds
        retries: Number of retry attempts on failure
        retryDelaySeconds: Delay between retries in seconds
        condition: Condition expression for conditional execution
        continueOnError: Whether to continue pipeline on step failure

    Example:
        >>> step = StepDefinition(
        ...     id="ingest",
        ...     agent="GL-DATA-X-001",
        ...     with_={"uri": "{{ params.input_uri }}"},
        ...     outputs={"dataset": "$.artifact.dataset_uri"}
        ... )
    """

    # Primary fields (v1 schema)
    id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=63,
        description="Unique step identifier"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Agent ID to execute (format: GL-CATEGORY-TYPE-NUMBER)"
    )
    dependsOn: List[str] = Field(
        default_factory=list,
        alias="dependsOn",
        description="List of step IDs this step depends on"
    )
    with_: Dict[str, Any] = Field(
        default_factory=dict,
        alias="with",
        description="Input parameters for the agent"
    )
    outputs: Union[Dict[str, str], List[str]] = Field(
        default_factory=dict,
        description="Output extraction mapping (key -> JSONPath) or list of output names"
    )
    policy: List[PolicyAttachment] = Field(
        default_factory=list,
        description="Policy attachments for governance"
    )
    timeoutSeconds: int = Field(
        default=900,
        ge=1,
        le=86400,
        description="Maximum execution time in seconds"
    )
    retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of retry attempts on failure"
    )
    retryDelaySeconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Delay between retries in seconds"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Condition expression for conditional execution"
    )
    continueOnError: bool = Field(
        default=False,
        description="Whether to continue pipeline execution on step failure"
    )

    # Legacy fields for backward compatibility
    name: Optional[str] = Field(
        default=None,
        description="Step name (legacy, use 'id' instead)"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID (legacy, use 'agent' instead)"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Dependencies (legacy, use 'dependsOn' instead)"
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step inputs (legacy, use 'with' instead)"
    )
    step_type: StepType = Field(
        default=StepType.AGENT,
        description="Step type (legacy)"
    )
    image: Optional[str] = Field(
        default=None,
        description="Container image for K8s (legacy)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Step description"
    )
    artifacts_in: List[str] = Field(
        default_factory=list,
        description="Input artifacts (legacy)"
    )
    artifacts_out: List[ArtifactDefinition] = Field(
        default_factory=list,
        description="Output artifacts (legacy)"
    )
    resources: Optional[ResourceRequirements] = Field(
        default=None,
        description="Resources (legacy)"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Retry attempts (legacy, use 'retries' instead)"
    )
    on_failure: str = Field(
        default="fail",
        description="Failure action (legacy)"
    )
    publishes_data: bool = Field(
        default=False,
        description="Publishes external data (legacy)"
    )
    accesses_pii: bool = Field(
        default=False,
        description="Accesses PII (legacy)"
    )
    requires_approval: bool = Field(
        default=False,
        description="Requires approval (legacy)"
    )
    data_regions: List[str] = Field(
        default_factory=list,
        description="Data regions (legacy)"
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def normalize_fields(self) -> "StepDefinition":
        """Normalize legacy fields to v1 schema fields."""
        # Normalize id from name
        if self.id is None and self.name is not None:
            object.__setattr__(self, "id", self.name)

        # Normalize agent from agent_id
        if self.agent is None and self.agent_id is not None:
            object.__setattr__(self, "agent", self.agent_id)

        # Normalize dependsOn from depends_on
        if not self.dependsOn and self.depends_on:
            object.__setattr__(self, "dependsOn", self.depends_on)

        # Normalize with_ from inputs
        if not self.with_ and self.inputs:
            object.__setattr__(self, "with_", self.inputs)

        # Normalize retries from retry_count
        if self.retries == 0 and self.retry_count > 0:
            object.__setattr__(self, "retries", self.retry_count)

        # Validate id is set
        if self.id is None:
            raise ValueError("Step must have 'id' (or 'name' for legacy format)")

        # Validate agent is set
        if self.agent is None:
            raise ValueError("Step must have 'agent' (or 'agent_id' for legacy format)")

        return self

    @field_validator("agent", mode="after")
    @classmethod
    def validate_agent_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate agent ID matches required pattern."""
        if v is not None and not AGENT_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid agent ID '{v}': must match pattern GL-CATEGORY-TYPE-NUMBER "
                "(e.g., GL-DATA-X-001, GL-CALC-A-002)"
            )
        return v

    @field_validator("outputs")
    @classmethod
    def validate_output_definitions(
        cls, v: Union[Dict[str, str], List[str]]
    ) -> Union[Dict[str, str], List[str]]:
        """Validate output definitions."""
        if isinstance(v, dict):
            for key, path in v.items():
                if not JSONPATH_PATTERN.match(path):
                    raise ValueError(
                        f"Invalid JSONPath for output '{key}': '{path}' "
                        "must start with '$.' followed by valid path"
                    )
        return v

    @field_validator("on_failure")
    @classmethod
    def validate_on_failure(cls, v: str) -> str:
        """Validate on_failure action."""
        valid = {"fail", "skip", "continue", "retry"}
        if v not in valid:
            raise ValueError(f"on_failure must be one of {valid}")
        return v

    def get_effective_id(self) -> str:
        """Get the effective step ID."""
        return self.id or self.name or ""


# ==============================================================================
# Pipeline Defaults
# ==============================================================================

class PipelineDefaults(BaseModel):
    """
    Default values applied to all steps in the pipeline.

    These defaults can be overridden by individual step configurations.

    Attributes:
        retries: Default retry count for steps
        timeoutSeconds: Default timeout for steps
        retryDelaySeconds: Default retry delay for steps

    Example:
        >>> defaults = PipelineDefaults(
        ...     retries=2,
        ...     timeoutSeconds=600
        ... )
    """

    retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Default retry count for steps"
    )
    timeoutSeconds: int = Field(
        default=900,
        ge=1,
        le=86400,
        description="Default timeout for steps in seconds"
    )
    retryDelaySeconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Default retry delay for steps in seconds"
    )


# ==============================================================================
# Pipeline Spec
# ==============================================================================

class PipelineSpec(BaseModel):
    """
    Pipeline specification containing parameters, defaults, and steps.

    The spec defines the complete execution plan for the pipeline including
    all steps, their dependencies, and default configurations.

    Supports both dict-based and list-based parameter definitions for
    backward compatibility.

    Attributes:
        parameters: Parameter definitions for pipeline inputs
        defaults: Default values applied to all steps
        steps: List of step definitions forming the DAG
        artifacts: Pipeline-level artifact definitions (legacy)
        concurrency: Maximum concurrent step execution (legacy)
        timeout_seconds: Pipeline-level timeout (legacy)

    Example:
        >>> spec = PipelineSpec(
        ...     parameters={"input_uri": ParameterDefinition(type=ParameterType.STRING, required=True)},
        ...     defaults=PipelineDefaults(retries=2),
        ...     steps=[
        ...         StepDefinition(id="ingest", agent="GL-DATA-X-001"),
        ...         StepDefinition(id="validate", agent="GL-DATA-X-002", dependsOn=["ingest"])
        ...     ]
        ... )
    """

    parameters: Union[Dict[str, ParameterDefinition], List[ParameterDefinition]] = Field(
        default_factory=dict,
        description="Parameter definitions for pipeline inputs"
    )
    defaults: PipelineDefaults = Field(
        default_factory=PipelineDefaults,
        description="Default values applied to all steps"
    )
    steps: List[StepDefinition] = Field(
        ...,
        min_length=1,
        description="List of step definitions (at least one required)"
    )
    # Legacy fields
    artifacts: List[ArtifactDefinition] = Field(
        default_factory=list,
        description="Pipeline artifacts (legacy)"
    )
    concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max concurrency (legacy)"
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Pipeline timeout (legacy)"
    )

    @field_validator("parameters", mode="before")
    @classmethod
    def normalize_parameters(
        cls, v: Union[Dict[str, Any], List[Any], None]
    ) -> Union[Dict[str, ParameterDefinition], List[ParameterDefinition]]:
        """Normalize parameters to consistent format."""
        if v is None:
            return {}
        if isinstance(v, list):
            # Convert list format to dict format
            result = {}
            for param in v:
                if isinstance(param, dict):
                    name = param.get("name")
                    if name:
                        result[name] = ParameterDefinition(**param)
                elif isinstance(param, ParameterDefinition):
                    if param.name:
                        result[param.name] = param
            return result
        return v

    @model_validator(mode="after")
    def validate_pipeline_spec(self) -> "PipelineSpec":
        """Validate pipeline spec including DAG structure and references."""
        # Collect step IDs
        step_ids: Set[str] = set()

        for step in self.steps:
            step_id = step.get_effective_id()
            if step_id in step_ids:
                raise ValueError(f"Duplicate step ID: '{step_id}'")
            step_ids.add(step_id)

        # Validate dependency references
        for step in self.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            for dep in deps:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step_id}' depends on unknown step '{dep}'"
                    )
                if dep == step_id:
                    raise ValueError(
                        f"Step '{step_id}' cannot depend on itself"
                    )

        # Validate DAG is acyclic
        cycle = self._detect_cycle()
        if cycle:
            raise ValueError(
                f"Pipeline contains cycle: {' -> '.join(cycle)}"
            )

        # Validate parameter template references
        self._validate_template_references(step_ids)

        return self

    def _detect_cycle(self) -> Optional[List[str]]:
        """
        Detect cycles in the step dependency graph using Kahn's algorithm.

        Returns:
            List of step IDs forming a cycle if found, None otherwise.
        """
        # Build adjacency list and in-degree map
        graph: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}

        for step in self.steps:
            step_id = step.get_effective_id()
            graph[step_id] = []
            in_degree[step_id] = 0

        for step in self.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            for dep in deps:
                if dep in graph:
                    graph[dep].append(step_id)
                    in_degree[step_id] += 1

        # Kahn's algorithm
        queue: deque[str] = deque()
        for step_id, degree in in_degree.items():
            if degree == 0:
                queue.append(step_id)

        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If not all nodes processed, there's a cycle
        if processed != len(self.steps):
            return self._find_cycle_path()

        return None

    def _find_cycle_path(self) -> List[str]:
        """Find and return the actual cycle path for error reporting."""
        # Build adjacency list (step -> its dependencies)
        graph: Dict[str, List[str]] = {}
        for step in self.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            graph[step_id] = deps.copy()

        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    path.append(neighbor)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for step in self.steps:
            step_id = step.get_effective_id()
            if step_id not in visited:
                if dfs(step_id):
                    for i, node in enumerate(path):
                        if path.count(node) > 1:
                            return path[i:]

        return ["cycle-detected"]

    def _validate_template_references(self, step_ids: Set[str]) -> None:
        """Validate template references in step inputs."""
        # Get parameter names
        param_names: Set[str] = set()
        if isinstance(self.parameters, dict):
            param_names = set(self.parameters.keys())
        elif isinstance(self.parameters, list):
            for param in self.parameters:
                if isinstance(param, ParameterDefinition) and param.name:
                    param_names.add(param.name)

        # Build a map of step_id -> output keys
        step_outputs: Dict[str, Set[str]] = {}
        for step in self.steps:
            step_id = step.get_effective_id()
            if isinstance(step.outputs, dict):
                step_outputs[step_id] = set(step.outputs.keys())
            elif isinstance(step.outputs, list):
                step_outputs[step_id] = set(step.outputs)
            else:
                step_outputs[step_id] = set()

        for step in self.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            available_steps = set(deps)

            # Get inputs to check
            inputs_to_check = step.with_ or step.inputs
            self._check_template_refs(
                inputs_to_check,
                param_names,
                available_steps,
                step_outputs,
                f"step '{step_id}'"
            )

    def _check_template_refs(
        self,
        value: Any,
        param_names: Set[str],
        available_steps: Set[str],
        step_outputs: Dict[str, Set[str]],
        context: str
    ) -> None:
        """Recursively check template references in a value."""
        if isinstance(value, str):
            for match in TEMPLATE_PATTERN.finditer(value):
                ref = match.group(1)

                if ref.startswith("params."):
                    param_name = ref[7:]
                    if param_name not in param_names:
                        raise ValueError(
                            f"In {context}: reference to undefined parameter '{param_name}'"
                        )

                elif ref.startswith("steps."):
                    parts = ref.split(".")
                    if len(parts) != 4:
                        raise ValueError(
                            f"In {context}: invalid step reference '{ref}'"
                        )
                    step_id = parts[1]
                    output_key = parts[3]

                    if step_id not in available_steps:
                        raise ValueError(
                            f"In {context}: reference to step '{step_id}' which is not "
                            "a declared dependency"
                        )

                    if output_key not in step_outputs.get(step_id, set()):
                        raise ValueError(
                            f"In {context}: reference to undefined output '{output_key}' "
                            f"from step '{step_id}'"
                        )

        elif isinstance(value, dict):
            for v in value.values():
                self._check_template_refs(v, param_names, available_steps, step_outputs, context)

        elif isinstance(value, list):
            for item in value:
                self._check_template_refs(item, param_names, available_steps, step_outputs, context)


# ==============================================================================
# Pipeline Definition (Root)
# ==============================================================================

class PipelineDefinition(BaseModel):
    """
    Root pipeline definition model.

    This is the top-level model that represents a complete Pipeline YAML document.
    It validates the entire structure including API version, kind, metadata, and spec.

    Supports both v1 schema (apiVersion: greenlang/v1) and legacy schema
    (api_version: greenlang.io/v1) for backward compatibility.

    Attributes:
        apiVersion: API version (v1: "greenlang/v1")
        api_version: API version (legacy: "greenlang.io/v1")
        kind: Resource kind (must be "Pipeline")
        metadata: Pipeline metadata
        spec: Pipeline specification
        namespace: Namespace (legacy, moved to metadata in v1)
        labels: Labels (legacy, moved to metadata in v1)
        annotations: Annotations (legacy, moved to metadata in v1)

    Example:
        >>> pipeline = PipelineDefinition(
        ...     apiVersion="greenlang/v1",
        ...     kind="Pipeline",
        ...     metadata=PipelineMetadata(name="my-pipeline"),
        ...     spec=PipelineSpec(steps=[
        ...         StepDefinition(id="step1", agent="GL-DATA-X-001")
        ...     ])
        ... )

    Loading from YAML:
        >>> import yaml
        >>> with open("pipeline.yaml") as f:
        ...     data = yaml.safe_load(f)
        >>> pipeline = PipelineDefinition(**data)
    """

    # v1 schema fields
    apiVersion: Optional[str] = Field(
        default=None,
        description="API version (v1: 'greenlang/v1')"
    )
    kind: str = Field(
        default="Pipeline",
        description="Resource kind (must be 'Pipeline')"
    )
    metadata: PipelineMetadata = Field(
        ...,
        description="Pipeline metadata"
    )
    spec: PipelineSpec = Field(
        ...,
        description="Pipeline specification"
    )

    # Legacy schema fields
    api_version: Optional[str] = Field(
        default=None,
        description="API version (legacy: 'greenlang.io/v1')"
    )
    namespace: str = Field(
        default="default",
        description="Namespace (legacy, use metadata.namespace)"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Labels (legacy, use metadata.labels)"
    )
    annotations: Dict[str, str] = Field(
        default_factory=dict,
        description="Annotations (legacy, use metadata.annotations)"
    )

    @model_validator(mode="after")
    def normalize_version(self) -> "PipelineDefinition":
        """Normalize API version field."""
        # Use apiVersion if set, otherwise fall back to api_version
        if self.apiVersion is None and self.api_version is not None:
            object.__setattr__(self, "apiVersion", self.api_version)
        elif self.apiVersion is None:
            object.__setattr__(self, "apiVersion", SUPPORTED_API_VERSION)

        # Validate kind
        if self.kind != "Pipeline":
            raise ValueError(f"Invalid kind '{self.kind}': must be 'Pipeline'")

        return self

    def normalize(self) -> "PipelineDefinition":
        """
        Normalize the pipeline definition for deterministic hashing.

        Normalization includes:
        - Sorting dictionary keys alphabetically
        - Normalizing whitespace in string values
        - Expanding YAML anchors (handled by YAML parser)

        Returns:
            Normalized copy of the pipeline definition.
        """
        return PipelineDefinition.model_validate(
            self._normalize_dict(self.model_dump(by_alias=True, exclude_none=True))
        )

    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize a dictionary for deterministic output."""
        result = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                result[key] = self._normalize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._normalize_dict(item) if isinstance(item, dict)
                    else self._normalize_str(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = self._normalize_str(value)
            else:
                result[key] = value
        return result

    def _normalize_str(self, s: str) -> str:
        """Normalize a string value (trim whitespace, normalize line endings)."""
        return " ".join(s.split())

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the normalized pipeline definition.

        This hash can be used for:
        - Caching compiled pipelines
        - Detecting pipeline changes
        - Audit trail provenance

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        normalized = self.normalize()
        json_str = json.dumps(
            normalized.model_dump(by_alias=True, exclude_none=True),
            sort_keys=True,
            separators=(",", ":")
        )
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def get_execution_order(self) -> List[str]:
        """
        Get step IDs in topologically sorted execution order.

        Returns:
            List of step IDs in valid execution order.
        """
        graph: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}

        for step in self.spec.steps:
            step_id = step.get_effective_id()
            graph[step_id] = []
            in_degree[step_id] = 0

        for step in self.spec.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            for dep in deps:
                if dep in graph:
                    graph[dep].append(step_id)
                    in_degree[step_id] += 1

        queue: deque[str] = deque()
        for step_id, degree in in_degree.items():
            if degree == 0:
                queue.append(step_id)

        result: List[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """
        Get a step by ID.

        Args:
            step_id: Step identifier to look up.

        Returns:
            StepDefinition if found, None otherwise.
        """
        for step in self.spec.steps:
            if step.get_effective_id() == step_id:
                return step
        return None

    def get_step_dependencies(self) -> Dict[str, Set[str]]:
        """Get dependency graph for all steps."""
        result = {}
        for step in self.spec.steps:
            step_id = step.get_effective_id()
            deps = step.dependsOn or step.depends_on
            result[step_id] = set(deps)
        return result

    def apply_defaults(self) -> "PipelineDefinition":
        """
        Apply pipeline defaults to all steps.

        Creates a new PipelineDefinition with default values applied
        to steps that don't have explicit values set.

        Returns:
            New PipelineDefinition with defaults applied.
        """
        defaults = self.spec.defaults
        updated_steps = []

        for step in self.spec.steps:
            step_dict = step.model_dump(by_alias=True)

            # Apply defaults only if step uses default value
            if step.retries == 0 and defaults.retries > 0:
                step_dict["retries"] = defaults.retries
            if step.timeoutSeconds == 900 and defaults.timeoutSeconds != 900:
                step_dict["timeoutSeconds"] = defaults.timeoutSeconds
            if step.retryDelaySeconds == 30 and defaults.retryDelaySeconds != 30:
                step_dict["retryDelaySeconds"] = defaults.retryDelaySeconds

            updated_steps.append(StepDefinition.model_validate(step_dict))

        return PipelineDefinition(
            apiVersion=self.apiVersion,
            kind=self.kind,
            metadata=self.metadata,
            spec=PipelineSpec(
                parameters=self.spec.parameters,
                defaults=self.spec.defaults,
                steps=updated_steps,
                artifacts=self.spec.artifacts,
                concurrency=self.spec.concurrency,
                timeout_seconds=self.spec.timeout_seconds,
            )
        )

    def to_policy_doc(self) -> Dict[str, Any]:
        """Convert to policy-safe document for OPA evaluation."""
        return {
            "api_version": self.apiVersion,
            "kind": self.kind,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "namespace": self.metadata.namespace,
            "owner": self.metadata.owner,
            "team": self.metadata.team,
            "step_count": len(self.spec.steps),
            "steps": [
                {
                    "id": s.get_effective_id(),
                    "agent": s.agent,
                    "publishes_data": s.publishes_data,
                    "accesses_pii": s.accesses_pii,
                    "requires_approval": s.requires_approval,
                    "data_regions": s.data_regions,
                }
                for s in self.spec.steps
            ],
            "labels": self.metadata.labels,
            "annotations": self.metadata.annotations,
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Export the Pydantic model as JSON Schema.

        Returns:
            JSON Schema dictionary.
        """
        return self.model_json_schema()


# ==============================================================================
# Run Configuration (Legacy)
# ==============================================================================

class RunConfig(BaseModel):
    """Configuration for a pipeline run."""

    run_id: str = Field(..., description="Unique run identifier")
    pipeline_name: str = Field(..., description="Pipeline name")
    pipeline_version: str = Field(..., description="Pipeline version")
    namespace: str = Field(default="default", description="Namespace")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")

    # Execution context
    user_id: str = Field(..., description="User initiating run")
    service_account: Optional[str] = Field(None, description="Service account")
    environment: str = Field(default="development", description="Environment")

    # Budget and limits
    estimated_cost_usd: Optional[float] = Field(None, ge=0, description="Estimated cost")
    max_cost_usd: Optional[float] = Field(None, ge=0, description="Maximum cost")
    priority: int = Field(default=5, ge=1, le=10, description="Run priority")

    # Governance
    data_regions: List[str] = Field(default_factory=list, description="Allowed regions")
    classification_level: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification"
    )

    def to_policy_doc(self) -> Dict[str, Any]:
        """Convert to policy-safe document."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "namespace": self.namespace,
            "user_id": self.user_id,
            "service_account": self.service_account,
            "environment": self.environment,
            "estimated_cost_usd": self.estimated_cost_usd,
            "max_cost_usd": self.max_cost_usd,
            "priority": self.priority,
            "data_regions": self.data_regions,
            "classification_level": self.classification_level.value,
            "labels": self.labels,
        }


# ==============================================================================
# Step Result (Legacy)
# ==============================================================================

class StepResult(BaseModel):
    """Result from executing a pipeline step."""

    step_name: str = Field(..., description="Step name")
    success: bool = Field(..., description="Whether step succeeded")
    status: str = Field(default="completed", description="Step status")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Step outputs")
    artifacts: List[str] = Field(default_factory=list, description="Produced artifacts")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration_ms: float = Field(default=0.0, ge=0, description="Execution duration")
    provenance_hash: str = Field(default="", description="Provenance hash")

    output_classification: Optional[DataClassification] = Field(
        None, description="Output data classification"
    )
    export_destinations: List[str] = Field(
        default_factory=list, description="Where data was exported"
    )


# ==============================================================================
# Execution Context (Legacy)
# ==============================================================================

class ExecutionContext(BaseModel):
    """Context passed to policy evaluation during execution."""

    run_id: str = Field(..., description="Run identifier")
    pipeline: PipelineDefinition = Field(..., description="Pipeline definition")
    run_config: RunConfig = Field(..., description="Run configuration")
    current_step: Optional[str] = Field(None, description="Current step name")

    completed_steps: List[str] = Field(
        default_factory=list, description="Completed steps"
    )
    step_results: Dict[str, StepResult] = Field(
        default_factory=dict, description="Step results"
    )
    total_cost_usd: float = Field(default=0.0, ge=0, description="Total cost so far")

    user_roles: List[str] = Field(default_factory=list, description="User roles")
    user_groups: List[str] = Field(default_factory=list, description="User groups")
    permissions: Set[str] = Field(default_factory=set, description="User permissions")

    def to_policy_doc(self) -> Dict[str, Any]:
        """Convert to policy-safe document."""
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline.to_policy_doc(),
            "run_config": self.run_config.to_policy_doc(),
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "total_cost_usd": self.total_cost_usd,
            "user_roles": self.user_roles,
            "user_groups": self.user_groups,
            "permissions": list(self.permissions),
        }


# ==============================================================================
# Utility Functions
# ==============================================================================

def load_pipeline_yaml(yaml_content: str) -> PipelineDefinition:
    """
    Load and validate a pipeline from YAML content.

    This function handles YAML anchor expansion automatically.

    Args:
        yaml_content: YAML string content.

    Returns:
        Validated PipelineDefinition.

    Raises:
        ValueError: If YAML parsing or validation fails.

    Example:
        >>> yaml_str = '''
        ... apiVersion: greenlang/v1
        ... kind: Pipeline
        ... metadata:
        ...   name: test-pipeline
        ... spec:
        ...   steps:
        ...     - id: step1
        ...       agent: GL-DATA-X-001
        ... '''
        >>> pipeline = load_pipeline_yaml(yaml_str)
    """
    import yaml

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if data is None:
        raise ValueError("Empty YAML content")

    return PipelineDefinition.model_validate(data)


def load_pipeline_file(path: str) -> PipelineDefinition:
    """
    Load and validate a pipeline from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated PipelineDefinition.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If YAML parsing or validation fails.
    """
    from pathlib import Path as PathLib

    file_path = PathLib(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return load_pipeline_yaml(f.read())


def validate_agent_id(agent_id: str) -> bool:
    """
    Validate an agent ID matches the required format.

    Args:
        agent_id: Agent ID to validate.

    Returns:
        True if valid, False otherwise.
    """
    return bool(AGENT_ID_PATTERN.match(agent_id))


def extract_template_references(value: Any) -> List[str]:
    """
    Extract all template references from a value.

    Args:
        value: Value to search for template references.

    Returns:
        List of template reference strings.

    Example:
        >>> refs = extract_template_references({"uri": "{{ params.input }}"})
        >>> refs
        ['params.input']
    """
    refs: List[str] = []

    def _extract(v: Any) -> None:
        if isinstance(v, str):
            for match in TEMPLATE_PATTERN.finditer(v):
                refs.append(match.group(1))
        elif isinstance(v, dict):
            for item in v.values():
                _extract(item)
        elif isinstance(v, list):
            for item in v:
                _extract(item)

    _extract(value)
    return refs


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Models
    "PipelineDefinition",
    "PipelineMetadata",
    "PipelineSpec",
    "PipelineDefaults",
    "StepDefinition",
    "ParameterDefinition",
    "PolicyAttachment",
    # Enums
    "ParameterType",
    "PolicySeverity",
    "StepType",
    "DataClassification",
    # Legacy models
    "ResourceRequirements",
    "ArtifactDefinition",
    "RunConfig",
    "StepResult",
    "ExecutionContext",
    # Constants
    "SUPPORTED_API_VERSION",
    "AGENT_ID_PATTERN",
    # Functions
    "load_pipeline_yaml",
    "load_pipeline_file",
    "validate_agent_id",
    "extract_template_references",
]
