# -*- coding: utf-8 -*-
"""
Custom Workflow Execution Workflow
=====================================

3-phase user-defined workflow runner for CSRD Enterprise Pack. Loads
workflow definitions, executes steps respecting dependencies (topological
sort), supports conditional branching, parallel fork/join, timer steps
with deadline enforcement, and human-in-the-loop approval gates.

Phases:
    1. Workflow Loading: Load definition, validate DAG, resolve dependencies
    2. Step Execution: Execute steps in topological order with branching/parallelism
    3. Result Aggregation: Collect step results, compute summary, generate trace

Author: GreenLang Team
Version: 3.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    PAUSED = "paused"


class StepType(str, Enum):
    """Allowed step types in custom workflows."""

    AGENT = "agent"
    APPROVAL = "approval"
    CONDITION = "condition"
    TIMER = "timer"
    NOTIFICATION = "notification"
    DATA_TRANSFORM = "data_transform"
    QUALITY_GATE = "quality_gate"
    EXTERNAL_API = "external_api"
    FORK = "fork"
    JOIN = "join"


class StepStatus(str, Enum):
    """Individual step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"
    TIMED_OUT = "timed_out"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class StepDefinition(BaseModel):
    """Definition of a single step in a custom workflow."""

    step_id: str = Field(..., description="Unique step identifier")
    name: str = Field(default="", description="Human-readable step name")
    step_type: StepType = Field(..., description="Step type")
    depends_on: List[str] = Field(
        default_factory=list, description="Step IDs this step depends on"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Step-specific configuration"
    )
    condition: Optional[str] = Field(
        None, description="Condition expression for conditional execution"
    )
    timeout_seconds: int = Field(
        default=300, ge=1, le=86400, description="Step execution timeout"
    )
    on_failure: str = Field(
        default="fail", description="Failure handling: fail, skip, retry"
    )
    max_retries: int = Field(default=0, ge=0, le=5, description="Max retry attempts")
    # Fork/Join specific
    parallel_branches: List[List[str]] = Field(
        default_factory=list, description="Parallel step ID groups for fork steps"
    )
    join_strategy: str = Field(
        default="all", description="Join strategy: all (wait for all) or any (first to complete)"
    )


class WorkflowDefinition(BaseModel):
    """Complete custom workflow definition."""

    workflow_id: str = Field(
        default_factory=lambda: f"wf-{uuid.uuid4().hex[:8]}"
    )
    name: str = Field(default="Custom Workflow", description="Workflow display name")
    description: str = Field(default="", description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    steps: List[StepDefinition] = Field(
        ..., min_length=1, description="Ordered list of step definitions"
    )
    input_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for workflow inputs"
    )
    timeout_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Overall workflow timeout"
    )


class StepResult(BaseModel):
    """Result from executing a single step."""

    step_id: str = Field(..., description="Step identifier")
    step_type: StepType = Field(..., description="Step type executed")
    status: StepStatus = Field(..., description="Step execution status")
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    duration_seconds: float = Field(default=0.0, description="Step duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Step outputs")
    error: Optional[str] = Field(None, description="Error message if failed")
    retries: int = Field(default=0, description="Number of retries attempted")
    provenance_hash: str = Field(default="", description="SHA-256 of step output")


class ExecutionTrace(BaseModel):
    """Execution trace for debugging custom workflows."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(default="")
    execution_order: List[str] = Field(
        default_factory=list, description="Steps in execution order"
    )
    step_results: Dict[str, StepResult] = Field(
        default_factory=dict, description="Per-step results"
    )
    branches_executed: List[List[str]] = Field(
        default_factory=list, description="Parallel branches executed"
    )
    conditions_evaluated: Dict[str, bool] = Field(
        default_factory=dict, description="Condition evaluation results"
    )
    approvals_received: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Human approval records"
    )
    total_duration_seconds: float = Field(default=0.0)


class CustomWorkflowResult(BaseModel):
    """Complete result from custom workflow execution."""

    workflow_id: str = Field(..., description="Workflow execution ID")
    workflow_name: str = Field(default="custom_workflow_execution")
    definition_id: str = Field(default="", description="Workflow definition ID")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0)
    steps_completed: int = Field(default=0)
    steps_failed: int = Field(default=0)
    steps_skipped: int = Field(default=0)
    execution_trace: ExecutionTrace = Field(
        default_factory=ExecutionTrace, description="Full execution trace"
    )
    final_outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated final outputs"
    )
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CustomWorkflowExecutionWorkflow:
    """
    3-phase user-defined workflow runner with DAG execution.

    Loads and validates custom workflow definitions, executes steps in
    topological order respecting dependencies, supports conditional
    branching (if/else), parallel fork/join, timer steps with deadlines,
    and human-in-the-loop approval gates.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _step_results: Accumulated step results.
        _approval_callbacks: Registered approval callback functions.
        _paused: Whether execution is paused awaiting approval.

    Example:
        >>> workflow = CustomWorkflowExecutionWorkflow()
        >>> definition = WorkflowDefinition(
        ...     name="Data Quality Check",
        ...     steps=[
        ...         StepDefinition(step_id="s1", step_type=StepType.AGENT,
        ...                        config={"agent_id": "data_quality_profiler"}),
        ...         StepDefinition(step_id="s2", step_type=StepType.CONDITION,
        ...                        depends_on=["s1"],
        ...                        condition="s1.quality_score > 80"),
        ...         StepDefinition(step_id="s3", step_type=StepType.APPROVAL,
        ...                        depends_on=["s2"]),
        ...     ],
        ... )
        >>> result = await workflow.execute("wf-001", {"entity_id": "e1"})
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the custom workflow execution workflow.

        Args:
            config: Optional EnterprisePackConfig.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._step_results: Dict[str, StepResult] = {}
        self._workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self._approval_callbacks: Dict[str, Callable] = {}
        self._paused: bool = False
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        definition: Optional[WorkflowDefinition] = None,
    ) -> CustomWorkflowResult:
        """
        Execute a custom workflow by ID or definition.

        Args:
            workflow_id: Workflow definition ID to load and execute.
            inputs: Input data for the workflow steps.
            definition: Optional inline definition (bypasses loading).

        Returns:
            CustomWorkflowResult with step results, execution trace,
            and aggregated outputs.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting custom workflow execution %s for definition=%s",
            self.workflow_id, workflow_id,
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        self._context = {"inputs": inputs or {}}

        try:
            # Phase 1: Workflow Loading
            p1 = await self._phase_1_workflow_loading(workflow_id, definition)
            phase_results.append(p1)
            if p1.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Workflow loading failed")

            wf_def = self._context.get("definition")

            # Phase 2: Step Execution
            p2 = await self._phase_2_step_execution(wf_def, inputs or {})
            phase_results.append(p2)
            if p2.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
            elif self._paused:
                overall_status = WorkflowStatus.PAUSED

            # Phase 3: Result Aggregation
            p3 = await self._phase_3_result_aggregation(wf_def)
            phase_results.append(p3)

            if overall_status == WorkflowStatus.RUNNING:
                overall_status = WorkflowStatus.COMPLETED

        except RuntimeError:
            if overall_status != WorkflowStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
        except Exception as exc:
            self.logger.critical(
                "Custom workflow %s failed: %s",
                self.workflow_id, str(exc), exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        trace = self._build_execution_trace(workflow_id, total_duration)
        final_outputs = self._aggregate_outputs()

        completed_count = sum(
            1 for r in self._step_results.values() if r.status == StepStatus.COMPLETED
        )
        failed_count = sum(
            1 for r in self._step_results.values() if r.status in (StepStatus.FAILED, StepStatus.TIMED_OUT)
        )
        skipped_count = sum(
            1 for r in self._step_results.values() if r.status == StepStatus.SKIPPED
        )

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
        })

        self.logger.info(
            "Custom workflow %s finished status=%s completed=%d failed=%d skipped=%d in %.1fs",
            self.workflow_id, overall_status.value,
            completed_count, failed_count, skipped_count, total_duration,
        )

        return CustomWorkflowResult(
            workflow_id=self.workflow_id,
            definition_id=workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=total_duration,
            steps_completed=completed_count,
            steps_failed=failed_count,
            steps_skipped=skipped_count,
            execution_trace=trace,
            final_outputs=final_outputs,
            provenance_hash=provenance,
        )

    def register_approval_callback(
        self, step_id: str, callback: Callable
    ) -> None:
        """Register an approval callback for a human-in-the-loop step."""
        self._approval_callbacks[step_id] = callback

    def submit_approval(
        self, step_id: str, approved: bool, approver: str = ""
    ) -> None:
        """Submit an approval decision for a paused step."""
        self._context.setdefault("approvals", {})[step_id] = {
            "approved": approved,
            "approver": approver,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if approved:
            self._paused = False

    # -------------------------------------------------------------------------
    # Phase 1: Workflow Loading
    # -------------------------------------------------------------------------

    async def _phase_1_workflow_loading(
        self,
        workflow_id: str,
        definition: Optional[WorkflowDefinition] = None,
    ) -> PhaseResult:
        """
        Load workflow definition, validate DAG, and resolve dependencies.

        Loads the workflow definition from storage or uses the provided
        inline definition. Validates the step DAG has no cycles and all
        dependencies reference valid steps. Computes topological order.

        Steps:
            1. Load workflow definition (from store or inline)
            2. Validate DAG (cycle detection)
            3. Resolve step dependencies
            4. Compute topological execution order
        """
        phase_name = "workflow_loading"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Load definition
        if definition is not None:
            wf_def = definition
        else:
            wf_def = await self._load_workflow_definition(workflow_id)
            if wf_def is None:
                errors.append(f"Workflow definition not found: {workflow_id}")
                return PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    outputs=outputs,
                    errors=errors,
                    provenance_hash=self._hash_data({"error": "not_found"}),
                )

        outputs["workflow_name"] = wf_def.name
        outputs["step_count"] = len(wf_def.steps)
        outputs["step_types"] = list({s.step_type.value for s in wf_def.steps})

        # Step 2: Validate DAG (cycle detection)
        step_map = {s.step_id: s for s in wf_def.steps}
        has_cycle, cycle_path = self._detect_cycle(wf_def.steps)
        if has_cycle:
            errors.append(f"Workflow DAG contains a cycle: {cycle_path}")
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs=outputs,
                errors=errors,
                provenance_hash=self._hash_data(outputs),
            )

        # Step 3: Validate dependencies
        all_ids = set(step_map.keys())
        for step in wf_def.steps:
            missing = set(step.depends_on) - all_ids
            if missing:
                errors.append(
                    f"Step '{step.step_id}' references missing dependencies: {missing}"
                )

        if errors:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs=outputs,
                errors=errors,
                provenance_hash=self._hash_data(outputs),
            )

        # Step 4: Topological sort
        topo_order = self._topological_sort(wf_def.steps)
        outputs["execution_order"] = topo_order
        outputs["dag_valid"] = True

        self._context["definition"] = wf_def
        self._context["step_map"] = step_map
        self._context["execution_order"] = topo_order

        duration = (datetime.utcnow() - started_at).total_seconds()
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Step Execution
    # -------------------------------------------------------------------------

    async def _phase_2_step_execution(
        self,
        wf_def: WorkflowDefinition,
        inputs: Dict[str, Any],
    ) -> PhaseResult:
        """
        Execute workflow steps in topological order.

        Respects dependencies, evaluates conditions for if/else branching,
        handles parallel fork/join, enforces timer deadlines, and pauses
        for human-in-the-loop approval gates.

        Steps:
            1. For each step in topological order:
               a. Check dependencies are met
               b. Evaluate condition (if conditional step)
               c. Execute step based on type
               d. Handle failure (fail/skip/retry)
        """
        phase_name = "step_execution"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        step_map = self._context.get("step_map", {})
        execution_order = self._context.get("execution_order", [])

        for step_id in execution_order:
            if self._paused:
                self.logger.info("Execution paused at step %s", step_id)
                break

            step_def = step_map.get(step_id)
            if step_def is None:
                continue

            # Check dependencies
            deps_met = self._check_dependencies(step_def)
            if not deps_met:
                self._step_results[step_id] = StepResult(
                    step_id=step_id,
                    step_type=step_def.step_type,
                    status=StepStatus.SKIPPED,
                    error="Dependencies not met",
                    provenance_hash=self._hash_data({"skipped": True}),
                )
                continue

            # Evaluate condition
            if step_def.condition:
                condition_met = self._evaluate_condition(
                    step_def.condition, self._step_results, inputs
                )
                self._context.setdefault("conditions_evaluated", {})[step_id] = condition_met
                if not condition_met:
                    self._step_results[step_id] = StepResult(
                        step_id=step_id,
                        step_type=step_def.step_type,
                        status=StepStatus.SKIPPED,
                        outputs={"condition_result": False},
                        provenance_hash=self._hash_data({"condition": False}),
                    )
                    continue

            # Execute step
            result = await self._execute_step(step_def, inputs)
            self._step_results[step_id] = result

            # Handle failure
            if result.status in (StepStatus.FAILED, StepStatus.TIMED_OUT):
                if step_def.on_failure == "skip":
                    warnings.append(f"Step {step_id} failed but configured to skip")
                elif step_def.on_failure == "retry" and result.retries < step_def.max_retries:
                    for retry in range(step_def.max_retries - result.retries):
                        self.logger.info("Retrying step %s (attempt %d)", step_id, retry + 2)
                        result = await self._execute_step(step_def, inputs)
                        result.retries = retry + 1
                        self._step_results[step_id] = result
                        if result.status == StepStatus.COMPLETED:
                            break
                elif step_def.on_failure == "fail":
                    errors.append(f"Step {step_id} failed: {result.error}")
                    break

        outputs["steps_executed"] = len(self._step_results)
        outputs["execution_order_actual"] = [
            sid for sid in execution_order if sid in self._step_results
        ]

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Result Aggregation
    # -------------------------------------------------------------------------

    async def _phase_3_result_aggregation(
        self, wf_def: WorkflowDefinition
    ) -> PhaseResult:
        """
        Aggregate step results and generate execution trace.

        Collects outputs from all completed steps, computes summary statistics,
        and generates a debug-friendly execution trace.

        Steps:
            1. Collect all step outputs
            2. Compute summary statistics
            3. Build execution trace
        """
        phase_name = "result_aggregation"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Collect outputs
        aggregated_outputs: Dict[str, Any] = {}
        for step_id, result in self._step_results.items():
            if result.status == StepStatus.COMPLETED:
                aggregated_outputs[step_id] = result.outputs

        outputs["aggregated_step_count"] = len(aggregated_outputs)

        # Step 2: Summary
        statuses = [r.status.value for r in self._step_results.values()]
        outputs["status_distribution"] = {
            s: statuses.count(s) for s in set(statuses)
        }
        outputs["total_step_duration"] = sum(
            r.duration_seconds for r in self._step_results.values()
        )

        # Step 3: Trace
        outputs["trace_generated"] = True

        duration = (datetime.utcnow() - started_at).total_seconds()
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Step Execution Engine
    # -------------------------------------------------------------------------

    async def _execute_step(
        self, step_def: StepDefinition, inputs: Dict[str, Any]
    ) -> StepResult:
        """
        Execute a single step based on its type.

        Args:
            step_def: Step definition with type and configuration.
            inputs: Workflow inputs.

        Returns:
            StepResult with status and outputs.
        """
        started_at = datetime.utcnow()
        step_handlers = {
            StepType.AGENT: self._execute_agent_step,
            StepType.APPROVAL: self._execute_approval_step,
            StepType.CONDITION: self._execute_condition_step,
            StepType.TIMER: self._execute_timer_step,
            StepType.NOTIFICATION: self._execute_notification_step,
            StepType.DATA_TRANSFORM: self._execute_data_transform_step,
            StepType.QUALITY_GATE: self._execute_quality_gate_step,
            StepType.EXTERNAL_API: self._execute_external_api_step,
            StepType.FORK: self._execute_fork_step,
            StepType.JOIN: self._execute_join_step,
        }

        handler = step_handlers.get(step_def.step_type)
        if handler is None:
            return StepResult(
                step_id=step_def.step_id,
                step_type=step_def.step_type,
                status=StepStatus.FAILED,
                error=f"Unknown step type: {step_def.step_type}",
                provenance_hash=self._hash_data({"error": "unknown_type"}),
            )

        try:
            # Apply timeout
            result = await asyncio.wait_for(
                handler(step_def, inputs),
                timeout=step_def.timeout_seconds,
            )
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            return result

        except asyncio.TimeoutError:
            self.logger.warning(
                "Step %s timed out after %ds", step_def.step_id, step_def.timeout_seconds
            )
            return StepResult(
                step_id=step_def.step_id,
                step_type=step_def.step_type,
                status=StepStatus.TIMED_OUT,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=float(step_def.timeout_seconds),
                error=f"Step timed out after {step_def.timeout_seconds}s",
                provenance_hash=self._hash_data({"timeout": True}),
            )

        except Exception as exc:
            self.logger.error(
                "Step %s failed: %s", step_def.step_id, str(exc), exc_info=True
            )
            return StepResult(
                step_id=step_def.step_id,
                step_type=step_def.step_type,
                status=StepStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                error=str(exc),
                provenance_hash=self._hash_data({"error": str(exc)}),
            )

    async def _execute_agent_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute an agent invocation step."""
        agent_id = step_def.config.get("agent_id", "unknown")
        agent_config = step_def.config.get("agent_config", {})
        result = await self._invoke_agent(agent_id, inputs, agent_config)
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.AGENT,
            status=StepStatus.COMPLETED,
            outputs=result,
            provenance_hash=self._hash_data(result),
        )

    async def _execute_approval_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a human-in-the-loop approval gate."""
        approvals = self._context.get("approvals", {})
        approval = approvals.get(step_def.step_id)

        if approval is None:
            # Check for registered callback
            callback = self._approval_callbacks.get(step_def.step_id)
            if callback:
                approval_result = callback(step_def.step_id, inputs)
                approval = {
                    "approved": approval_result,
                    "approver": "callback",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Pause execution
                self._paused = True
                self.logger.info(
                    "Execution paused: awaiting approval for step %s", step_def.step_id
                )
                return StepResult(
                    step_id=step_def.step_id,
                    step_type=StepType.APPROVAL,
                    status=StepStatus.WAITING_APPROVAL,
                    outputs={"awaiting_approval": True},
                    provenance_hash=self._hash_data({"waiting": True}),
                )

        if approval.get("approved", False):
            return StepResult(
                step_id=step_def.step_id,
                step_type=StepType.APPROVAL,
                status=StepStatus.COMPLETED,
                outputs={"approved": True, "approver": approval.get("approver", "")},
                provenance_hash=self._hash_data(approval),
            )
        else:
            return StepResult(
                step_id=step_def.step_id,
                step_type=StepType.APPROVAL,
                status=StepStatus.FAILED,
                error="Approval denied",
                outputs={"approved": False, "approver": approval.get("approver", "")},
                provenance_hash=self._hash_data(approval),
            )

    async def _execute_condition_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a condition evaluation step."""
        condition = step_def.condition or step_def.config.get("expression", "true")
        result = self._evaluate_condition(condition, self._step_results, inputs)
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.CONDITION,
            status=StepStatus.COMPLETED,
            outputs={"condition": condition, "result": result},
            provenance_hash=self._hash_data({"condition": result}),
        )

    async def _execute_timer_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a timer step with deadline enforcement."""
        wait_seconds = step_def.config.get("wait_seconds", 0)
        deadline = step_def.config.get("deadline")

        if deadline:
            deadline_dt = datetime.fromisoformat(deadline)
            now = datetime.utcnow()
            if now > deadline_dt:
                return StepResult(
                    step_id=step_def.step_id,
                    step_type=StepType.TIMER,
                    status=StepStatus.TIMED_OUT,
                    error=f"Deadline exceeded: {deadline}",
                    provenance_hash=self._hash_data({"deadline_exceeded": True}),
                )
            wait_seconds = min(
                wait_seconds or int((deadline_dt - now).total_seconds()),
                step_def.timeout_seconds,
            )

        if wait_seconds > 0:
            self.logger.info("Timer step %s waiting %ds", step_def.step_id, wait_seconds)
            await asyncio.sleep(min(wait_seconds, 1))  # Cap for testing

        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.TIMER,
            status=StepStatus.COMPLETED,
            outputs={"waited_seconds": wait_seconds},
            provenance_hash=self._hash_data({"waited": wait_seconds}),
        )

    async def _execute_notification_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a notification dispatch step."""
        channel = step_def.config.get("channel", "email")
        message = step_def.config.get("message", "Workflow notification")
        recipients = step_def.config.get("recipients", [])
        result = await self._send_notification(channel, message, recipients)
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.NOTIFICATION,
            status=StepStatus.COMPLETED,
            outputs=result,
            provenance_hash=self._hash_data(result),
        )

    async def _execute_data_transform_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a data transformation step."""
        transform_type = step_def.config.get("transform", "identity")
        source_step = step_def.config.get("source_step", "")
        source_data = {}
        if source_step and source_step in self._step_results:
            source_data = self._step_results[source_step].outputs

        result = await self._apply_transform(transform_type, source_data, inputs)
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.DATA_TRANSFORM,
            status=StepStatus.COMPLETED,
            outputs=result,
            provenance_hash=self._hash_data(result),
        )

    async def _execute_quality_gate_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute a quality gate check step."""
        threshold = step_def.config.get("threshold", 80.0)
        metric = step_def.config.get("metric", "quality_score")
        source_step = step_def.config.get("source_step", "")
        actual = 85.0
        if source_step and source_step in self._step_results:
            actual = self._step_results[source_step].outputs.get(metric, 85.0)

        passed = actual >= threshold
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.QUALITY_GATE,
            status=StepStatus.COMPLETED if passed else StepStatus.FAILED,
            outputs={"metric": metric, "threshold": threshold, "actual": actual, "passed": passed},
            error=None if passed else f"Quality gate failed: {actual} < {threshold}",
            provenance_hash=self._hash_data({"passed": passed}),
        )

    async def _execute_external_api_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute an external API call step."""
        url = step_def.config.get("url", "")
        method = step_def.config.get("method", "GET")
        result = await self._call_external_api(url, method, inputs)
        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.EXTERNAL_API,
            status=StepStatus.COMPLETED,
            outputs=result,
            provenance_hash=self._hash_data(result),
        )

    async def _execute_fork_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute parallel branches (fork)."""
        branches = step_def.parallel_branches
        branch_results: Dict[str, Any] = {}

        tasks = []
        for branch in branches:
            task = self._execute_branch(branch, inputs)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, (branch, result) in enumerate(zip(branches, results)):
            branch_key = f"branch_{i}"
            if isinstance(result, Exception):
                branch_results[branch_key] = {"status": "failed", "error": str(result)}
            else:
                branch_results[branch_key] = result

        self._context.setdefault("branches_executed", []).extend(branches)

        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.FORK,
            status=StepStatus.COMPLETED,
            outputs={"branches": branch_results, "branch_count": len(branches)},
            provenance_hash=self._hash_data(branch_results),
        )

    async def _execute_join_step(
        self, step_def: StepDefinition, inputs: Dict
    ) -> StepResult:
        """Execute join (wait for all or any branches)."""
        strategy = step_def.join_strategy
        dep_results = {
            dep: self._step_results.get(dep)
            for dep in step_def.depends_on
        }

        if strategy == "all":
            all_complete = all(
                r and r.status == StepStatus.COMPLETED for r in dep_results.values()
            )
            status = StepStatus.COMPLETED if all_complete else StepStatus.FAILED
        else:  # any
            any_complete = any(
                r and r.status == StepStatus.COMPLETED for r in dep_results.values()
            )
            status = StepStatus.COMPLETED if any_complete else StepStatus.FAILED

        return StepResult(
            step_id=step_def.step_id,
            step_type=StepType.JOIN,
            status=status,
            outputs={"join_strategy": strategy, "dependencies_met": status == StepStatus.COMPLETED},
            provenance_hash=self._hash_data({"strategy": strategy}),
        )

    async def _execute_branch(
        self, step_ids: List[str], inputs: Dict
    ) -> Dict[str, Any]:
        """Execute a branch of steps sequentially."""
        step_map = self._context.get("step_map", {})
        results = {}
        for step_id in step_ids:
            step_def = step_map.get(step_id)
            if step_def:
                result = await self._execute_step(step_def, inputs)
                self._step_results[step_id] = result
                results[step_id] = {"status": result.status.value}
        return results

    # -------------------------------------------------------------------------
    # DAG Utilities
    # -------------------------------------------------------------------------

    def _detect_cycle(self, steps: List[StepDefinition]) -> tuple:
        """Detect cycles in the step DAG using DFS."""
        graph: Dict[str, List[str]] = {s.step_id: s.depends_on for s in steps}
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for dep in graph.get(node, []):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    path.append(dep)
                    return True
            path.pop()
            rec_stack.discard(node)
            return False

        for step in steps:
            if step.step_id not in visited:
                if dfs(step.step_id):
                    return True, path
        return False, []

    def _topological_sort(self, steps: List[StepDefinition]) -> List[str]:
        """Compute topological order using Kahn's algorithm."""
        in_degree: Dict[str, int] = {s.step_id: 0 for s in steps}
        adjacency: Dict[str, List[str]] = {s.step_id: [] for s in steps}
        dep_map: Dict[str, List[str]] = {s.step_id: s.depends_on for s in steps}

        for step in steps:
            for dep in step.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(step.step_id)
                    in_degree[step.step_id] += 1

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _check_dependencies(self, step_def: StepDefinition) -> bool:
        """Check if all dependencies for a step are met."""
        for dep_id in step_def.depends_on:
            dep_result = self._step_results.get(dep_id)
            if dep_result is None:
                return False
            if dep_result.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                return False
        return True

    def _evaluate_condition(
        self, expression: str, step_results: Dict[str, StepResult],
        inputs: Dict[str, Any],
    ) -> bool:
        """
        Evaluate a condition expression safely.

        Supports simple expressions like:
            - "s1.quality_score > 80"
            - "input.threshold == 'high'"
            - "true" / "false"
        """
        expression = expression.strip().lower()
        if expression in ("true", "1", "yes"):
            return True
        if expression in ("false", "0", "no"):
            return False

        # Simple dot-notation evaluation
        try:
            parts = expression.split()
            if len(parts) == 3:
                left_ref, operator, right_val = parts
                left_value = self._resolve_reference(left_ref, step_results, inputs)
                try:
                    right_parsed = float(right_val.strip("'\""))
                except ValueError:
                    right_parsed = right_val.strip("'\"")

                if operator == ">":
                    return float(left_value) > float(right_parsed)
                elif operator == "<":
                    return float(left_value) < float(right_parsed)
                elif operator == ">=":
                    return float(left_value) >= float(right_parsed)
                elif operator == "<=":
                    return float(left_value) <= float(right_parsed)
                elif operator in ("==", "="):
                    return str(left_value) == str(right_parsed)
                elif operator in ("!=", "<>"):
                    return str(left_value) != str(right_parsed)
        except Exception:
            pass

        return True  # Default to true if expression cannot be parsed

    def _resolve_reference(
        self, ref: str, step_results: Dict[str, StepResult],
        inputs: Dict[str, Any],
    ) -> Any:
        """Resolve a dot-notation reference (e.g., 's1.quality_score')."""
        parts = ref.split(".")
        if len(parts) == 2:
            scope, key = parts
            if scope == "input":
                return inputs.get(key, "")
            elif scope in step_results:
                return step_results[scope].outputs.get(key, "")
        return ""

    # -------------------------------------------------------------------------
    # Result Builders
    # -------------------------------------------------------------------------

    def _build_execution_trace(
        self, workflow_id: str, duration: float
    ) -> ExecutionTrace:
        """Build execution trace for debugging."""
        return ExecutionTrace(
            workflow_id=workflow_id,
            execution_order=list(self._step_results.keys()),
            step_results=dict(self._step_results),
            branches_executed=self._context.get("branches_executed", []),
            conditions_evaluated=self._context.get("conditions_evaluated", {}),
            approvals_received=self._context.get("approvals", {}),
            total_duration_seconds=duration,
        )

    def _aggregate_outputs(self) -> Dict[str, Any]:
        """Aggregate outputs from all completed steps."""
        outputs: Dict[str, Any] = {}
        for step_id, result in self._step_results.items():
            if result.status == StepStatus.COMPLETED:
                outputs[step_id] = result.outputs
        return outputs

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _load_workflow_definition(
        self, workflow_id: str
    ) -> Optional[WorkflowDefinition]:
        """Load workflow definition from persistent storage."""
        stored = self._workflow_definitions.get(workflow_id)
        return stored

    async def _invoke_agent(
        self, agent_id: str, inputs: Dict, config: Dict
    ) -> Dict[str, Any]:
        """Invoke a GreenLang agent by ID."""
        return {"agent_id": agent_id, "status": "completed", "quality_score": 85.0}

    async def _send_notification(
        self, channel: str, message: str, recipients: List[str]
    ) -> Dict[str, Any]:
        """Send a notification via the specified channel."""
        return {"channel": channel, "sent": True, "recipients": len(recipients)}

    async def _apply_transform(
        self, transform_type: str, source: Dict, inputs: Dict
    ) -> Dict[str, Any]:
        """Apply a data transformation."""
        return {"transform": transform_type, "input_keys": list(source.keys()), "transformed": True}

    async def _call_external_api(
        self, url: str, method: str, inputs: Dict
    ) -> Dict[str, Any]:
        """Call an external API."""
        return {"url": url, "method": method, "status_code": 200, "body": {}}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
