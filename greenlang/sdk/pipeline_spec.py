"""Pydantic models for pipeline specification.

This module provides type-safe Pydantic models for defining pipeline configurations,
including steps, error handling, retry policies, and overall pipeline structure.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# =============================================================================
# Error Handling and Retry Models
# =============================================================================


class RetrySpec(BaseModel):
    """Retry configuration for step execution."""

    max: int = Field(..., description="Maximum number of retry attempts", ge=0, le=10)
    backoff_seconds: float = Field(
        ..., description="Backoff delay between retries in seconds", ge=0.0
    )


OnErrorPolicy = Literal["stop", "continue", "skip", "fail"]


class OnErrorObj(BaseModel):
    """Detailed error handling configuration."""

    policy: OnErrorPolicy = Field(..., description="Error handling policy")
    retry: Optional[RetrySpec] = Field(
        None, description="Retry configuration for this error policy"
    )


# Union type for error handling - can be simple string or detailed object
OnErrorSpec = Union[OnErrorPolicy, OnErrorObj]


# =============================================================================
# Step Specification Model
# =============================================================================


class StepSpec(BaseModel):
    """Specification for a single pipeline step."""

    # Core step fields
    name: str = Field(..., description="Step identifier name")
    agent: str = Field(..., description="Agent class or module to execute")
    action: str = Field(default="run", description="Action method to call on agent")

    # Input specifications with reserved keyword handling
    inputs: Optional[Dict[str, Any]] = Field(
        None, description="Input data for the step"
    )
    in_: Optional[Dict[str, Any]] = Field(
        None, description="Input data for the step (alias for inputs field)", alias="in"
    )

    # Alternative input reference (mutually exclusive with inputs)
    inputsRef: Optional[str] = Field(
        None, description="Reference to inputs from another step or context"
    )

    # Step configuration with reserved keyword handling
    with_: Optional[Dict[str, Any]] = Field(
        None, description="Additional configuration parameters", alias="with"
    )

    # Execution control
    condition: Optional[str] = Field(
        None, description="Conditional expression to determine if step should execute"
    )
    parallel: bool = Field(
        default=False, description="Whether this step can run in parallel with others"
    )

    # Error handling and retries
    on_error: OnErrorSpec = Field(
        default="stop",
        description="Error handling policy - can be simple string or detailed configuration",
    )

    # Output configuration
    outputs: Optional[Dict[str, Any]] = Field(
        None, description="Expected output schema or configuration"
    )

    # Optional metadata
    description: Optional[str] = Field(
        None, description="Human-readable description of this step"
    )
    id: Optional[str] = Field(
        None, description="Optional unique identifier for the step"
    )
    timeout: Optional[float] = Field(
        None, description="Step timeout in seconds", ge=0.0
    )

    @model_validator(mode="after")
    def validate_inputs_exclusive(self) -> StepSpec:
        """Ensure inputs/in_ and inputsRef are mutually exclusive."""
        input_fields = [self.inputs, self.in_, self.inputsRef]
        non_none_count = sum(1 for field in input_fields if field is not None)

        if non_none_count > 1:
            raise ValueError(
                "inputs, in_, and inputsRef are mutually exclusive - use only one"
            )

        # Normalize inputs field - if in_ is provided, copy to inputs
        if self.in_ is not None and self.inputs is None:
            self.inputs = self.in_

        return self

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate step name follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("Step name cannot be empty")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Step name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v.strip()


# =============================================================================
# Pipeline Specification Model
# =============================================================================


class PipelineSpec(BaseModel):
    """Complete pipeline specification with metadata and configuration."""

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields

    # Required pipeline identity
    name: str = Field(..., description="Pipeline name identifier")
    version: str = Field(default="1", description="Pipeline version")

    # Pipeline metadata
    description: Optional[str] = Field(
        None, description="Human-readable description of the pipeline"
    )
    author: Optional[str] = Field(None, description="Pipeline author or maintainer")
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Tags for categorizing and discovering pipelines",
    )

    # Core pipeline definition
    steps: List[StepSpec] = Field(
        ..., description="Ordered list of pipeline steps", min_length=1
    )

    # Input/Output specifications
    inputs: Optional[Dict[str, Any]] = Field(
        None, description="Pipeline input schema or default values"
    )
    outputs: Optional[Dict[str, Any]] = Field(
        None, description="Pipeline output mapping or schema"
    )

    # Runtime configuration
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Runtime parameters and their defaults"
    )
    artifacts_dir: str = Field(
        default="out", description="Directory for storing pipeline artifacts"
    )

    # Execution control
    stop_on_error: bool = Field(
        default=True, description="Whether to stop pipeline execution on any step error"
    )
    max_parallel_steps: Optional[int] = Field(
        None, description="Maximum number of steps that can run in parallel", ge=1
    )

    # Global error handling
    on_error: Optional[OnErrorSpec] = Field(
        None, description="Global error handling policy for the pipeline"
    )

    # Lifecycle hooks
    hooks: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None, description="Pipeline lifecycle hooks (on_start, on_complete, on_error)"
    )

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional pipeline metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate pipeline name follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("Pipeline name cannot be empty")
        return v.strip()

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v or not v.strip():
            raise ValueError("Pipeline version cannot be empty")
        return v.strip()

    @field_validator("steps")
    @classmethod
    def validate_steps_unique_names(cls, v: List[StepSpec]) -> List[StepSpec]:
        """Ensure all step names are unique within the pipeline."""
        names = [step.name for step in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in set(names) if names.count(name) > 1]
            raise ValueError(f"Duplicate step names found: {', '.join(duplicates)}")
        return v

    @model_validator(mode="after")
    def validate_pipeline_consistency(self) -> PipelineSpec:
        """Validate overall pipeline consistency and references."""
        # Collect all step names for reference validation
        step_names = {step.name for step in self.steps}

        # Validate step references in conditions and inputsRef
        for step in self.steps:
            # Check condition references
            if step.condition:
                # Simple check for step references in conditions
                # In a production system, you'd want more sophisticated parsing
                for step_name in step_names:
                    if f"steps.{step_name}" in step.condition:
                        # Ensure referenced step comes before current step
                        ref_step_idx = next(
                            i for i, s in enumerate(self.steps) if s.name == step_name
                        )
                        current_step_idx = next(
                            i for i, s in enumerate(self.steps) if s.name == step.name
                        )
                        if ref_step_idx >= current_step_idx:
                            raise ValueError(
                                f"Step '{step.name}' condition references step '{step_name}' "
                                "that appears later in the pipeline"
                            )

            # Check inputsRef references
            if step.inputsRef:
                if step.inputsRef.startswith("$steps."):
                    # Extract referenced step name
                    ref_parts = step.inputsRef.split(".")
                    if len(ref_parts) >= 2:
                        ref_step_name = ref_parts[1]
                        if ref_step_name not in step_names:
                            raise ValueError(
                                f"Step '{step.name}' inputsRef references unknown step '{ref_step_name}'"
                            )

        return self

    def get_step(self, name: str) -> Optional[StepSpec]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_parallel_steps(self) -> List[StepSpec]:
        """Get all steps marked for parallel execution."""
        return [step for step in self.steps if step.parallel]
