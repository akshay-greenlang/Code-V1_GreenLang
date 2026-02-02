# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
import json

from greenlang.exceptions import ValidationError


class WorkflowStep(BaseModel):
    """
    Represents a single step in a workflow execution.

    Each step maps to an agent execution with optional input/output mapping,
    conditional execution, and failure handling configuration.

    Attributes:
        name: Unique step identifier within the workflow
        agent_id: ID of the agent to execute for this step
        description: Human-readable step description
        input_mapping: Maps step inputs from workflow context
        output_key: Key to store step output in workflow context
        condition: Expression to evaluate before executing step
        on_failure: Action on failure (stop, skip, or continue)
        retry_count: Number of retry attempts on failure
    """

    name: str = Field(..., description="Step name")
    agent_id: str = Field(..., description="ID of the agent to execute")
    description: Optional[str] = Field(None, description="Step description")
    input_mapping: Optional[Dict[str, str]] = Field(
        None, description="Maps step input from context"
    )
    output_key: Optional[str] = Field(
        None, description="Key to store step output in context"
    )
    condition: Optional[str] = Field(
        None, description="Condition to evaluate before executing step"
    )
    on_failure: str = Field(
        default="stop", description="Action on failure: stop, skip, or continue"
    )
    retry_count: int = Field(default=0, description="Number of retries on failure")

    def to_policy_doc(self) -> Dict[str, Any]:
        """
        Convert step to a policy-safe document for OPA evaluation.

        This method creates a sanitized dictionary suitable for policy
        enforcement. It exposes only the structural information needed
        for policy decisions while excluding runtime state.

        Returns:
            Dict[str, Any]: Policy-safe document containing:
                - name: Step name
                - agent_id: Agent identifier
                - has_condition: Whether step has conditional execution
                - on_failure: Failure handling strategy
                - input_keys: List of input mapping keys (not values)
                - has_output: Whether step produces output

        Example:
            >>> step = WorkflowStep(name="validate", agent_id="validator")
            >>> doc = step.to_policy_doc()
            >>> # Use in policy evaluation
        """
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "has_condition": self.condition is not None,
            "on_failure": self.on_failure,
            "input_keys": list(self.input_mapping.keys()) if self.input_mapping else [],
            "has_output": self.output_key is not None,
        }


class Workflow(BaseModel):
    """
    Represents a complete workflow definition with steps and configuration.

    A workflow is an ordered sequence of steps, where each step executes
    an agent with specific inputs and outputs. Workflows support conditional
    execution, failure handling, and output mapping.

    Attributes:
        name: Unique workflow identifier
        description: Human-readable workflow description
        version: Semantic version string
        steps: Ordered list of workflow steps
        output_mapping: Maps final workflow output from execution context
        metadata: Additional workflow metadata for extensions

    Example:
        >>> workflow = Workflow(
        ...     name="carbon-calculation",
        ...     description="Calculate carbon emissions",
        ...     steps=[
        ...         WorkflowStep(name="intake", agent_id="intake_agent"),
        ...         WorkflowStep(name="calculate", agent_id="calc_agent"),
        ...     ]
        ... )
    """

    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="0.0.1", description="Workflow version")
    steps: List[WorkflowStep] = Field(..., description="List of workflow steps")
    output_mapping: Optional[Dict[str, str]] = Field(
        None, description="Maps workflow output from context"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_policy_doc(self) -> Dict[str, Any]:
        """
        Convert workflow to a policy-safe document for OPA evaluation.

        This method creates a sanitized dictionary suitable for policy
        enforcement. It includes structural information about the workflow
        and its steps while excluding sensitive runtime state and actual
        input/output values.

        The policy document follows the format expected by GreenLang's
        OPA policy bundles (bundles/run.rego).

        Returns:
            Dict[str, Any]: Policy-safe document containing:
                - name: Workflow name
                - version: Workflow version
                - description: Workflow description
                - step_count: Number of steps in workflow
                - steps: List of step policy documents
                - output_keys: List of output mapping keys
                - metadata: Workflow metadata (filtered for policy use)
                - agents: List of unique agent IDs used in workflow

        Example:
            >>> workflow = Workflow(name="test", description="Test workflow", steps=[...])
            >>> policy_doc = workflow.to_policy_doc()
            >>> # Use with OPA evaluation
            >>> from greenlang.policy import check_run
            >>> check_run(workflow, context)  # Uses to_policy_doc() internally
        """
        # Extract unique agent IDs for capability checking
        agent_ids = list(set(step.agent_id for step in self.steps))

        # Filter metadata for policy-relevant keys only
        policy_metadata = {
            k: v for k, v in self.metadata.items()
            if k in ("region", "environment", "classification", "owner", "tags")
        }

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "step_count": len(self.steps),
            "steps": [step.to_policy_doc() for step in self.steps],
            "output_keys": list(self.output_mapping.keys()) if self.output_mapping else [],
            "metadata": policy_metadata,
            "agents": agent_ids,
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Workflow":
        """Load workflow from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Workflow instance

        Raises:
            ValidationError: If YAML file is invalid or malformed
        """
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except FileNotFoundError as e:
            raise ValidationError(
                message=f"Workflow YAML file not found: {yaml_path}",
                context={"yaml_path": yaml_path, "error": str(e)}
            ) from e
        except yaml.YAMLError as e:
            raise ValidationError(
                message=f"Invalid YAML syntax in workflow file: {yaml_path}",
                context={"yaml_path": yaml_path, "error": str(e)}
            ) from e

    @classmethod
    def from_json(cls, json_path: str) -> "Workflow":
        """Load workflow from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Workflow instance

        Raises:
            ValidationError: If JSON file is invalid or malformed
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError as e:
            raise ValidationError(
                message=f"Workflow JSON file not found: {json_path}",
                context={"json_path": json_path, "error": str(e)}
            ) from e
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON syntax in workflow file: {json_path}",
                context={"json_path": json_path, "error": str(e)}
            ) from e

    def to_yaml(self, path: str) -> None:
        """Export workflow to YAML file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Export workflow to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow.

        Args:
            step: WorkflowStep to add
        """
        self.steps.append(step)

    def remove_step(self, step_name: str) -> None:
        """Remove a step from the workflow by name.

        Args:
            step_name: Name of step to remove
        """
        self.steps = [s for s in self.steps if s.name != step_name]

    def get_step(self, step_name: str) -> Optional[WorkflowStep]:
        """Get a step by name.

        Args:
            step_name: Name of step to retrieve

        Returns:
            WorkflowStep if found, None otherwise
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None

    def validate_workflow(self) -> List[str]:
        """Validate the workflow structure.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        if not self.steps:
            errors.append("Workflow has no steps")

        step_names: set[str] = set()
        for step in self.steps:
            if step.name in step_names:
                errors.append(f"Duplicate step name: {step.name}")
            step_names.add(step.name)

        return errors


class WorkflowBuilder:
    """Builder pattern for constructing workflows programmatically."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize workflow builder.

        Args:
            name: Workflow name
            description: Workflow description
        """
        self.workflow = Workflow(name=name, description=description, steps=[])

    def add_step(self, name: str, agent_id: str, **kwargs: Any) -> "WorkflowBuilder":
        """Add a step to the workflow.

        Args:
            name: Step name
            agent_id: Agent ID to execute
            **kwargs: Additional step parameters

        Returns:
            Self for method chaining
        """
        step = WorkflowStep(name=name, agent_id=agent_id, **kwargs)
        self.workflow.add_step(step)
        return self

    def with_output_mapping(self, mapping: Dict[str, str]) -> "WorkflowBuilder":
        """Set output mapping for the workflow.

        Args:
            mapping: Output field mappings

        Returns:
            Self for method chaining
        """
        self.workflow.output_mapping = mapping
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> "WorkflowBuilder":
        """Set metadata for the workflow.

        Args:
            metadata: Workflow metadata

        Returns:
            Self for method chaining
        """
        self.workflow.metadata = metadata
        return self

    def build(self) -> Workflow:
        """Build and validate the workflow.

        Returns:
            Validated Workflow instance

        Raises:
            ValidationError: If workflow validation fails
        """
        errors = self.workflow.validate_workflow()
        if errors:
            raise ValidationError(
                message="Workflow validation failed",
                context={
                    "workflow_name": self.workflow.name,
                    "validation_errors": errors
                },
                invalid_fields={error.split(":")[0]: error for error in errors}
            )
        return self.workflow
