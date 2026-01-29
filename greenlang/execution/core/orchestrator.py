# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator
======================

Orchestrates the execution of agent workflows with policy enforcement,
retry logic, checkpoint-based recovery (FR-074), and comprehensive execution tracking.

Author: GreenLang Framework Team
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
import hashlib
import json

from greenlang.agents.base import BaseAgent
from greenlang.execution.core.workflow import Workflow
import logging
import ast

from greenlang.exceptions import ValidationError, ExecutionError, MissingData

# FR-074: Import checkpoint management
try:
    from greenlang.execution.core.checkpoint_manager import (
        CheckpointManager,
        CheckpointState,
        CheckpointStatus,
        CheckpointExecutionContract,
        InMemoryCheckpointStore,
    )
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

# FR-063: Import alert management
try:
    from greenlang.orchestrator.alerting import AlertManager, AlertType, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    AlertManager = None


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
    """
    Orchestrates the execution of agent workflows.

    Features:
    - Policy enforcement before execution
    - Retry logic with configurable attempts
    - FR-074: Checkpoint-based recovery for long-running pipelines
    - Comprehensive execution tracking and history

    Attributes:
        agents: Registered agents available for workflow steps
        workflows: Registered workflow definitions
        execution_history: History of executed workflows
        checkpoint_manager: Optional checkpoint manager for FR-074
        checkpoint_enabled: Whether checkpointing is enabled
    """

    def __init__(
        self,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        checkpoint_enabled: bool = True,
        alert_manager: Optional["AlertManager"] = None,
        alerting_enabled: bool = True,
        default_namespace: str = "default",
    ) -> None:
        """
        Initialize the Orchestrator.

        Args:
            checkpoint_manager: Optional CheckpointManager for FR-074 checkpoint support.
                               If None and checkpoint_enabled=True, creates InMemoryCheckpointStore.
            checkpoint_enabled: Whether to enable checkpoint-based recovery (FR-074)
            alert_manager: Optional AlertManager for FR-063 alert webhooks.
                          If None and alerting_enabled=True, creates default AlertManager.
            alerting_enabled: Whether to enable alert webhooks (FR-063)
            default_namespace: Default namespace for alerts
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logger
        self.default_namespace = default_namespace

        # FR-074: Checkpoint management
        self.checkpoint_enabled = checkpoint_enabled and CHECKPOINT_AVAILABLE
        self._checkpoint_manager = checkpoint_manager

        if self.checkpoint_enabled and self._checkpoint_manager is None:
            try:
                store = InMemoryCheckpointStore()
                self._checkpoint_manager = CheckpointManager(store=store)
                self.logger.info("Initialized Orchestrator with InMemoryCheckpointStore")
            except Exception as e:
                self.logger.warning(f"Failed to initialize checkpoint manager: {e}")
                self.checkpoint_enabled = False

        # FR-063: Alert management
        self.alerting_enabled = alerting_enabled and ALERTING_AVAILABLE
        self._alert_manager = alert_manager

        if self.alerting_enabled and self._alert_manager is None:
            try:
                self._alert_manager = AlertManager()
                self.logger.info("Initialized Orchestrator with AlertManager")
            except Exception as e:
                self.logger.warning(f"Failed to initialize alert manager: {e}")
                self.alerting_enabled = False

    @property
    def checkpoint_manager(self) -> Optional["CheckpointManager"]:
        """Get the checkpoint manager if available."""
        return self._checkpoint_manager

    def set_checkpoint_manager(self, manager: "CheckpointManager") -> None:
        """Set a custom checkpoint manager."""
        self._checkpoint_manager = manager
        self.checkpoint_enabled = manager is not None

    @property
    def alert_manager(self) -> Optional["AlertManager"]:
        """Get the alert manager if available (FR-063)."""
        return self._alert_manager

    def set_alert_manager(self, manager: "AlertManager") -> None:
        """Set a custom alert manager (FR-063)."""
        self._alert_manager = manager
        self.alerting_enabled = manager is not None

    async def _emit_alert_async(
        self,
        alert_type: "AlertType",
        severity: "AlertSeverity",
        run_id: str,
        message: str,
        step_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Emit an alert asynchronously (FR-063).

        This is a helper method for emitting alerts from async contexts.
        """
        if not self.alerting_enabled or self._alert_manager is None:
            return

        try:
            await self._alert_manager.emit_alert(
                namespace=namespace or self.default_namespace,
                alert_type=alert_type,
                severity=severity,
                run_id=run_id,
                message=message,
                step_id=step_id,
                pipeline_id=pipeline_id,
                details=details,
            )
        except Exception as e:
            self.logger.warning(f"Failed to emit alert: {e}")

    def _emit_alert_sync(
        self,
        alert_type: "AlertType",
        severity: "AlertSeverity",
        run_id: str,
        message: str,
        step_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Emit an alert synchronously (FR-063).

        This is a helper method for emitting alerts from sync contexts.
        It schedules the async emission in a fire-and-forget manner.
        """
        if not self.alerting_enabled or self._alert_manager is None:
            return

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as a task if loop is running
                asyncio.create_task(
                    self._emit_alert_async(
                        alert_type=alert_type,
                        severity=severity,
                        run_id=run_id,
                        message=message,
                        step_id=step_id,
                        pipeline_id=pipeline_id,
                        details=details,
                        namespace=namespace,
                    )
                )
            else:
                # Run synchronously if no loop
                loop.run_until_complete(
                    self._emit_alert_async(
                        alert_type=alert_type,
                        severity=severity,
                        run_id=run_id,
                        message=message,
                        step_id=step_id,
                        pipeline_id=pipeline_id,
                        details=details,
                        namespace=namespace,
                    )
                )
        except RuntimeError:
            # No event loop, create one
            asyncio.run(
                self._emit_alert_async(
                    alert_type=alert_type,
                    severity=severity,
                    run_id=run_id,
                    message=message,
                    step_id=step_id,
                    pipeline_id=pipeline_id,
                    details=details,
                    namespace=namespace,
                )
            )
        except Exception as e:
            self.logger.warning(f"Failed to emit alert: {e}")

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
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        resume_from_checkpoint: bool = False,
        run_checkpoint: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow with optional checkpoint-based recovery.

        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            resume_from_checkpoint: Whether to resume from checkpoint (FR-074)
            run_checkpoint: Optional RunCheckpoint to resume from (FR-074)

        Returns:
            Workflow execution result with results, errors, and checkpoint info
        """
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

        # FR-074: Compute plan hash for checkpoint integrity
        plan_hash = self._compute_plan_hash(workflow)

        # FR-074: Initialize or load checkpoint
        checkpoint_run = None
        skipped_steps = set()

        if self.checkpoint_enabled and self._checkpoint_manager:
            import asyncio
            try:
                if resume_from_checkpoint and run_checkpoint:
                    # Use provided checkpoint for retry
                    checkpoint_run = run_checkpoint
                    skipped_steps = set(checkpoint_run.get_completed_steps())
                    self.logger.info(
                        f"Resuming from checkpoint: {len(skipped_steps)} steps will be skipped"
                    )
                else:
                    # Create new checkpoint for this run
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, schedule the coroutine
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self._checkpoint_manager.create_run_checkpoint(
                                    run_id=execution_id,
                                    plan_id=workflow_id,
                                    plan_hash=plan_hash,
                                    pipeline_id=workflow_id,
                                )
                            )
                            checkpoint_run = future.result()
                    else:
                        checkpoint_run = loop.run_until_complete(
                            self._checkpoint_manager.create_run_checkpoint(
                                run_id=execution_id,
                                plan_id=workflow_id,
                                plan_hash=plan_hash,
                                pipeline_id=workflow_id,
                            )
                        )
                    self.logger.info(f"Created checkpoint for run: {execution_id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize checkpoint: {e}")

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
                # FR-063: Emit policy denial alert
                if self.alerting_enabled and ALERTING_AVAILABLE:
                    self._emit_alert_sync(
                        alert_type=AlertType.POLICY_DENIAL,
                        severity=AlertSeverity.MEDIUM,
                        run_id=execution_id,
                        message=f"Policy check failed for workflow '{workflow_id}': {e}",
                        pipeline_id=workflow_id,
                        details={"policy_error": str(e), "workflow_id": workflow_id},
                    )
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

        # FR-074: Load results from skipped steps into context
        if skipped_steps and checkpoint_run:
            for step_id in skipped_steps:
                step_checkpoint = checkpoint_run.get_step_checkpoint(step_id)
                if step_checkpoint and step_checkpoint.outputs:
                    context["results"][step_id] = step_checkpoint.outputs
                    self.logger.info(f"Loaded checkpoint outputs for step: {step_id}")

        for step in workflow.steps:
            # FR-074: Skip steps that completed successfully in previous run
            if step.name in skipped_steps:
                self.logger.info(f"Skipping step (from checkpoint): {step.name}")
                continue

            if not self._should_execute_step(step, context):
                self.logger.info(f"Skipping step (condition): {step.name}")
                continue

            self.logger.info(f"Executing step: {step.name}")

            # FR-074: Generate idempotency key for this step
            idempotency_key = ""
            if self.checkpoint_enabled and CHECKPOINT_AVAILABLE:
                idempotency_key = CheckpointState.generate_idempotency_key(
                    plan_hash=plan_hash,
                    step_id=step.name,
                    attempt=1,
                )

            # Implement retry logic
            max_retries = step.retry_count if step.retry_count > 0 else 0
            attempt = 0
            step_succeeded = False
            last_error = None
            step_outputs = {}

            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        self.logger.info(
                            f"Retrying step {step.name} (attempt {attempt}/{max_retries})"
                        )
                        # FR-074: Update idempotency key for retry attempt
                        if self.checkpoint_enabled and CHECKPOINT_AVAILABLE:
                            idempotency_key = CheckpointState.generate_idempotency_key(
                                plan_hash=plan_hash,
                                step_id=step.name,
                                attempt=attempt + 1,
                            )

                    step_input = self._prepare_step_input(step, context)

                    # FR-074: Add execution contract to step input
                    if self.checkpoint_enabled:
                        step_input["_checkpoint_contract"] = {
                            "run_id": execution_id,
                            "step_id": step.name,
                            "idempotency_key": idempotency_key,
                            "attempt": attempt + 1,
                            "is_retry": attempt > 0,
                            "checkpoint_enabled": True,
                        }

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
                        step_outputs = result
                        context["results"][step.name] = result
                    else:
                        # Assume it's an AgentResult or has success attribute
                        success = getattr(result, "success", False)
                        # Store the data from the AgentResult, not the object itself
                        if hasattr(result, "data"):
                            step_outputs = {
                                "success": success,
                                "data": result.data,
                            }
                            context["results"][step.name] = step_outputs
                        else:
                            step_outputs = {"success": success}
                            context["results"][step.name] = result

                    if success:
                        step_succeeded = True

                        # FR-074: Save checkpoint after successful step
                        if self.checkpoint_enabled and self._checkpoint_manager and checkpoint_run:
                            self._save_step_checkpoint(
                                run_id=execution_id,
                                step_id=step.name,
                                status=CheckpointStatus.COMPLETED,
                                outputs=step_outputs,
                                idempotency_key=idempotency_key,
                                attempt=attempt + 1,
                            )

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

                # FR-074: Save failed checkpoint
                if self.checkpoint_enabled and self._checkpoint_manager and checkpoint_run:
                    self._save_step_checkpoint(
                        run_id=execution_id,
                        step_id=step.name,
                        status=CheckpointStatus.FAILED,
                        outputs={},
                        idempotency_key=idempotency_key,
                        attempt=attempt + 1,
                        error_message=last_error,
                    )

                if step.on_failure == "stop":
                    self.logger.error(
                        f"Step failed after {attempt + 1} attempts, stopping workflow: {step.name}"
                    )
                    # FR-063: Emit step failure alert that stops workflow
                    if self.alerting_enabled and ALERTING_AVAILABLE:
                        self._emit_alert_sync(
                            alert_type=AlertType.RUN_FAILED,
                            severity=AlertSeverity.HIGH,
                            run_id=execution_id,
                            message=f"Step '{step.name}' failed after {attempt + 1} attempts: {last_error}",
                            step_id=step.name,
                            pipeline_id=workflow_id,
                            details={
                                "step_name": step.name,
                                "error": last_error,
                                "attempts": attempt + 1,
                                "on_failure": step.on_failure,
                            },
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

        # FR-063: Emit run completion/failure alert
        if self.alerting_enabled and ALERTING_AVAILABLE:
            if execution_record["success"]:
                self._emit_alert_sync(
                    alert_type=AlertType.RUN_SUCCEEDED,
                    severity=AlertSeverity.INFO,
                    run_id=execution_id,
                    message=f"Workflow '{workflow_id}' completed successfully",
                    pipeline_id=workflow_id,
                    details={"steps_executed": len(context["results"])},
                )
            else:
                self._emit_alert_sync(
                    alert_type=AlertType.RUN_FAILED,
                    severity=AlertSeverity.HIGH,
                    run_id=execution_id,
                    message=f"Workflow '{workflow_id}' failed with {len(context['errors'])} error(s)",
                    pipeline_id=workflow_id,
                    details={
                        "error_count": len(context["errors"]),
                        "errors": context["errors"][:5],
                    },
                )

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

    # =========================================================================
    # FR-074: Checkpoint Helper Methods
    # =========================================================================

    def _compute_plan_hash(self, workflow: Workflow) -> str:
        """
        Compute SHA-256 hash of the workflow plan for checkpoint integrity.

        Args:
            workflow: Workflow to hash

        Returns:
            SHA-256 hex digest of the workflow structure
        """
        plan_data = {
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "name": step.name,
                    "agent_id": step.agent_id,
                    "condition": step.condition,
                    "retry_count": step.retry_count,
                    "on_failure": step.on_failure,
                }
                for step in workflow.steps
            ],
        }
        plan_str = json.dumps(plan_data, sort_keys=True)
        return hashlib.sha256(plan_str.encode()).hexdigest()

    def _save_step_checkpoint(
        self,
        run_id: str,
        step_id: str,
        status: "CheckpointStatus",
        outputs: Dict[str, Any],
        idempotency_key: str,
        attempt: int,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Save checkpoint state for a step.

        Args:
            run_id: Run identifier
            step_id: Step identifier
            status: Checkpoint status (COMPLETED, FAILED, etc.)
            outputs: Step outputs to checkpoint
            idempotency_key: Idempotency key for the step
            attempt: Attempt number
            error_message: Error message if step failed
        """
        if not self.checkpoint_enabled or not self._checkpoint_manager:
            return

        try:
            import asyncio

            state = CheckpointState(
                run_id=run_id,
                step_id=step_id,
                status=status,
                outputs=outputs,
                idempotency_key=idempotency_key,
                attempt=attempt,
                error_message=error_message,
            )

            if status == CheckpointStatus.COMPLETED:
                state.mark_completed(outputs)
            elif status == CheckpointStatus.FAILED:
                state.mark_failed(error_message or "Unknown error")

            # Handle async in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context - schedule as task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._checkpoint_manager.save_checkpoint(run_id, step_id, state)
                        )
                        future.result(timeout=5.0)
                else:
                    loop.run_until_complete(
                        self._checkpoint_manager.save_checkpoint(run_id, step_id, state)
                    )
            except RuntimeError:
                # No event loop, create one
                asyncio.run(
                    self._checkpoint_manager.save_checkpoint(run_id, step_id, state)
                )

            self.logger.debug(f"Saved checkpoint for step {step_id}: {status.value}")

        except Exception as e:
            self.logger.warning(f"Failed to save step checkpoint: {e}")

    async def get_run_checkpoint(self, run_id: str) -> Optional[Any]:
        """
        Get checkpoint for a run (async).

        Args:
            run_id: Run identifier

        Returns:
            RunCheckpoint if exists, None otherwise
        """
        if not self.checkpoint_enabled or not self._checkpoint_manager:
            return None

        return await self._checkpoint_manager.get_run_checkpoint(run_id)

    async def prepare_retry_checkpoint(
        self,
        original_run_id: str,
        new_run_id: str,
        skip_succeeded: bool = True,
        force_rerun_steps: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """
        Prepare a checkpoint for a retry run.

        Args:
            original_run_id: Original failed run ID
            new_run_id: New retry run ID
            skip_succeeded: Whether to skip succeeded steps
            force_rerun_steps: Steps to force re-run

        Returns:
            New RunCheckpoint for retry, or None if preparation failed
        """
        if not self.checkpoint_enabled or not self._checkpoint_manager:
            return None

        return await self._checkpoint_manager.prepare_retry(
            original_run_id=original_run_id,
            new_run_id=new_run_id,
            skip_succeeded=skip_succeeded,
            force_rerun_steps=force_rerun_steps,
        )


# Alias for backward compatibility and alternative naming convention
WorkflowOrchestrator = Orchestrator


__all__ = [
    "Orchestrator",
    "WorkflowOrchestrator",
    "ExecutionState",
    "PolicyExecutionContext",
    # FR-074: Checkpoint support
    "CHECKPOINT_AVAILABLE",
]
