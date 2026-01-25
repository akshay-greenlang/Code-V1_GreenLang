# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator
======================

Orchestrates the execution of agent workflows with policy enforcement,
retry logic, and comprehensive execution tracking.

Author: GreenLang Framework Team
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from greenlang.agents.base import BaseAgent
from greenlang.execution.core.workflow import Workflow
import logging
import ast

from greenlang.exceptions import ValidationError, ExecutionError, MissingData


class ExecutionState(str, Enum):
    """Enumeration of possible execution states for policy context."""

    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PolicyExecutionContext:
    """
    Execution context for policy enforcement checks.

    This class provides all necessary information for the policy enforcer
    to validate that a workflow execution complies with defined policies
    (e.g., data egress rules, regional compliance, etc.).

    Attributes:
        egress_targets: List of external targets data may be sent to
        region: Geographic region for compliance checks (e.g., 'US', 'EU')
        metadata: Additional metadata from the input data
        execution_state: Current state of the execution
        current_agent: ID of the agent currently being executed (if any)
        run_id: Unique identifier for this execution run
        started_at: Timestamp when execution started

    Example:
        >>> context = PolicyExecutionContext.from_input_data(
        ...     input_data={"metadata": {"location": {"country": "EU"}}},
        ...     run_id="workflow_123_0"
        ... )
        >>> context.execution_state
        <ExecutionState.STARTED: 'started'>
        >>> context.region
        'EU'
    """

    # Policy-relevant fields
    egress_targets: List[str] = field(default_factory=list)
    region: str = "US"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution state tracking
    execution_state: ExecutionState = ExecutionState.PENDING
    current_agent: Optional[str] = None

    # Run identification
    run_id: str = field(default_factory=lambda: f"policy-run-{uuid.uuid4().hex[:12]}")
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_input_data(
        cls,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
        execution_state: ExecutionState = ExecutionState.STARTED
    ) -> "PolicyExecutionContext":
        """
        Create a PolicyExecutionContext from workflow input data.

        Args:
            input_data: Input data dictionary from workflow execution
            run_id: Optional run ID (generated if not provided)
            execution_state: Initial execution state

        Returns:
            Configured PolicyExecutionContext instance
        """
        metadata = input_data.get("metadata", {})
        location = metadata.get("location", {})
        region = location.get("country", "US")

        # Extract egress targets from metadata if present
        egress_targets = metadata.get("egress_targets", [])

        context = cls(
            egress_targets=egress_targets,
            region=region,
            metadata=metadata,
            execution_state=execution_state,
            started_at=datetime.now()
        )

        if run_id:
            context.run_id = run_id

        return context

    def set_current_agent(self, agent_id: Optional[str]) -> None:
        """Update the current agent being executed."""
        self.current_agent = agent_id
        if agent_id:
            self.execution_state = ExecutionState.RUNNING

    def mark_completed(self) -> None:
        """Mark the execution as completed."""
        self.execution_state = ExecutionState.COMPLETED
        self.current_agent = None

    def mark_failed(self) -> None:
        """Mark the execution as failed."""
        self.execution_state = ExecutionState.FAILED
        self.current_agent = None

    def add_egress_target(self, target: str) -> None:
        """Add an egress target for policy validation."""
        if target not in self.egress_targets:
            self.egress_targets.append(target)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/serialization."""
        return {
            "egress_targets": self.egress_targets,
            "region": self.region,
            "metadata": self.metadata,
            "execution_state": self.execution_state.value,
            "current_agent": self.current_agent,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat()
        }

# Import policy enforcement if available
try:
    import sys
    import os

    # Add the core module to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    core_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "core")
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

    from greenlang.policy.enforcer import check_run

    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the execution of agent workflows"""

    def __init__(self) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logger

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register an agent for use in workflows.

        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to register
        """
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {agent_id}")

    def register_workflow(self, workflow_id: str, workflow: Workflow) -> None:
        """Register a workflow for execution.

        Args:
            workflow_id: Unique identifier for the workflow
            workflow: Workflow instance to register
        """
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow_id}")

    def execute_workflow(
        self, workflow_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if workflow_id not in self.workflows:
            raise MissingData(
                message=f"Workflow '{workflow_id}' not found",
                context={
                    "workflow_id": workflow_id,
                    "available_workflows": list(self.workflows.keys())
                },
                data_type="workflow",
                missing_fields=["workflow"]
            )

        workflow = self.workflows[workflow_id]
        execution_id = f"{workflow_id}_{len(self.execution_history)}"

        self.logger.info(f"Starting workflow execution: {execution_id}")

        # Policy enforcement check before execution
        if POLICY_AVAILABLE:
            try:
                # Create execution context for policy check using proper class
                policy_context = PolicyExecutionContext.from_input_data(
                    input_data=input_data,
                    run_id=execution_id,
                    execution_state=ExecutionState.STARTED
                )
                check_run(workflow, policy_context)
                self.logger.info(f"Runtime policy check passed for run_id={policy_context.run_id}")
            except RuntimeError as e:
                error_msg = f"Runtime policy check failed: {e}"
                self.logger.error(error_msg)
                return {
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "success": False,
                    "errors": [{"step": "policy_check", "error": error_msg}],
                    "results": {},
                }
            except Exception as e:
                self.logger.warning(f"Policy check error: {e}")

        context = {
            "input": input_data,
            "results": {},
            "errors": [],
            "workflow_id": workflow_id,
            "execution_id": execution_id,
        }

        for step in workflow.steps:
            if not self._should_execute_step(step, context):
                self.logger.info(f"Skipping step: {step.name}")
                continue

            self.logger.info(f"Executing step: {step.name}")

            # Implement retry logic
            max_retries = step.retry_count if step.retry_count > 0 else 0
            attempt = 0
            step_succeeded = False
            last_error = None

            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        self.logger.info(
                            f"Retrying step {step.name} (attempt {attempt}/{max_retries})"
                        )

                    step_input = self._prepare_step_input(step, context)
                    agent = self.agents.get(step.agent_id)

                    if not agent:
                        raise MissingData(
                            message=f"Agent '{step.agent_id}' not found",
                            context={
                                "agent_id": step.agent_id,
                                "step_name": step.name,
                                "available_agents": list(self.agents.keys())
                            },
                            data_type="agent",
                            missing_fields=["agent"]
                        )

                    result = agent.run(step_input)

                    # Handle both dict and AgentResult returns
                    if isinstance(result, dict):
                        # Convert dict to AgentResult-like structure
                        success = result.get("success", False)
                        context["results"][step.name] = result
                    else:
                        # Assume it's an AgentResult or has success attribute
                        success = getattr(result, "success", False)
                        # Store the data from the AgentResult, not the object itself
                        if hasattr(result, "data"):
                            context["results"][step.name] = {
                                "success": success,
                                "data": result.data,
                            }
                        else:
                            context["results"][step.name] = result

                    if success:
                        step_succeeded = True
                        break  # Success, exit retry loop
                    else:
                        # Step failed but returned normally
                        last_error = (
                            result.get("error", "Unknown error")
                            if isinstance(result, dict)
                            else getattr(result, "error", "Unknown error")
                        )

                        if attempt < max_retries:
                            self.logger.warning(
                                f"Step {step.name} failed, will retry. Error: {last_error}"
                            )
                            attempt += 1
                            continue
                        else:
                            # No more retries
                            break

                except Exception as e:
                    last_error = str(e)
                    self.logger.error(f"Error in step {step.name}: {last_error}")

                    if attempt < max_retries:
                        self.logger.warning(f"Will retry step {step.name}")
                        attempt += 1
                        continue
                    else:
                        # No more retries, handle as final failure
                        break

            # Handle final failure after all retries
            if not step_succeeded:
                if last_error:
                    context["errors"].append(
                        {
                            "step": step.name,
                            "error": last_error,
                            "attempts": attempt + 1,
                        }
                    )

                if step.on_failure == "stop":
                    self.logger.error(
                        f"Step failed after {attempt + 1} attempts, stopping workflow: {step.name}"
                    )
                    break
                elif step.on_failure == "skip":
                    self.logger.warning(
                        f"Step failed after {attempt + 1} attempts, continuing: {step.name}"
                    )
                    continue

        execution_record = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "input": input_data,
            "results": context["results"],
            "errors": context["errors"],
            "success": len(context["errors"]) == 0,
        }

        self.execution_history.append(execution_record)

        return self._format_workflow_output(workflow, context)

    def _should_execute_step(self, step, context: Dict) -> bool:
        if not step.condition:
            return True

        try:
            # Safe expression evaluation using AST
            return self._evaluate_condition(step.condition, context)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False

    def _evaluate_condition(self, expression: str, context: Dict) -> bool:
        """Safely evaluate a boolean expression against the given context."""
        allowed_names = {
            "context": context,
            "input": context.get("input", {}),
            "results": context.get("results", {}),
        }

        def eval_node(node):
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(eval_node(v) for v in node.values)
                if isinstance(node.op, ast.Or):
                    return any(eval_node(v) for v in node.values)
                raise ValidationError(
                    message="Unsupported boolean operator in workflow condition",
                    context={
                        "operator": type(node.op).__name__,
                        "condition": ast.dump(node)
                    },
                    invalid_fields={"operator": f"Boolean operator '{type(node.op).__name__}' not supported"}
                )
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not eval_node(node.operand)
            if isinstance(node, ast.Compare):
                left = eval_node(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    right = eval_node(comp)
                    if isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Gt):
                        ok = left > right
                    elif isinstance(op, ast.GtE):
                        ok = left >= right
                    elif isinstance(op, ast.Lt):
                        ok = left < right
                    elif isinstance(op, ast.LtE):
                        ok = left <= right
                    elif isinstance(op, ast.In):
                        ok = left in right
                    elif isinstance(op, ast.NotIn):
                        ok = left not in right
                    else:
                        raise ValidationError(
                            message="Unsupported comparison operator in workflow condition",
                            context={
                                "operator": type(op).__name__,
                                "condition": ast.dump(node)
                            },
                            invalid_fields={"operator": f"Comparison operator '{type(op).__name__}' not supported"}
                        )
                    if not ok:
                        return False
                    left = right
                return True
            if isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValidationError(
                    message=f"Name '{node.id}' is not allowed in workflow condition",
                    context={
                        "name": node.id,
                        "allowed_names": list(allowed_names.keys()),
                        "condition": ast.dump(node)
                    },
                    invalid_fields={"name": f"'{node.id}' is not a valid name"}
                )
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Subscript):
                value = eval_node(node.value)
                index = eval_node(node.slice)
                return value[index]
            if isinstance(node, ast.Attribute):
                value = eval_node(node.value)
                if isinstance(value, dict):
                    return value.get(node.attr)
                return getattr(value, node.attr)
            raise ValidationError(
                message=f"Unsupported expression in workflow condition",
                context={
                    "expression_type": type(node).__name__,
                    "expression": ast.dump(node)
                },
                invalid_fields={"expression": f"Expression type '{type(node).__name__}' not supported"}
            )

        tree = ast.parse(expression, mode="eval")
        return bool(eval_node(tree.body))

    def _prepare_step_input(self, step, context: Dict) -> Dict[str, Any]:
        if step.input_mapping:
            mapped_input = {}
            for key, path in step.input_mapping.items():
                value = self._get_value_from_path(context, path)
                if value is not None:
                    mapped_input[key] = value
            return mapped_input
        else:
            # Pass the entire context input for now
            # Agents should be able to extract what they need
            return context.get("input", {})

    def _get_value_from_path(self, data: Dict, path: str) -> Any:
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _format_workflow_output(
        self, workflow: Workflow, context: Dict
    ) -> Dict[str, Any]:
        output = {
            "workflow_id": context["workflow_id"],
            "execution_id": context["execution_id"],
            "success": len(context["errors"]) == 0,
            "errors": context["errors"],
        }

        if workflow.output_mapping:
            output["data"] = {}
            for key, path in workflow.output_mapping.items():
                value = self._get_value_from_path(context, path)
                if value is not None:
                    output["data"][key] = value
        else:
            output["results"] = context["results"]

        return output

    def execute_single_agent(
        self, agent_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        agent = self.agents.get(agent_id)
        if not agent:
            raise MissingData(
                message=f"Agent '{agent_id}' not found",
                context={
                    "agent_id": agent_id,
                    "available_agents": list(self.agents.keys())
                },
                data_type="agent",
                missing_fields=["agent"]
            )

        result = agent.run(input_data)

        # Handle both dict and AgentResult types
        if isinstance(result, dict):
            return result
        elif hasattr(result, "model_dump"):
            # Pydantic model
            return result.model_dump()
        elif hasattr(result, "__dict__"):
            # Object with attributes
            return {
                "success": getattr(result, "success", False),
                "data": getattr(result, "data", {}),
                "error": getattr(result, "error", None),
                "metadata": getattr(result, "metadata", {}),
            }
        else:
            return result

    def get_execution_history(self) -> List[Dict]:
        return self.execution_history

    def clear_history(self):
        self.execution_history = []

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def list_workflows(self) -> List[str]:
        return list(self.workflows.keys())

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        return {
            "id": agent_id,
            "name": agent.config.name,
            "description": agent.config.description,
            "version": agent.config.version,
            "enabled": agent.config.enabled,
        }

    def get_workflow_info(self, workflow_id: str) -> Dict[str, Any]:
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "name": step.name,
                    "agent_id": step.agent_id,
                    "description": step.description,
                }
                for step in workflow.steps
            ],
        }


# Alias for backward compatibility and alternative naming convention
WorkflowOrchestrator = Orchestrator


__all__ = [
    "Orchestrator",
    "WorkflowOrchestrator",
    "ExecutionState",
    "PolicyExecutionContext",
]
