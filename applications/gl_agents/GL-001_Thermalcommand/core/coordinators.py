"""
GL-001 ThermalCommand Orchestrator - Coordinator Modules

This module provides specialized coordinators for workflow execution,
safety management, and optimization across the process heat agent ecosystem.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import logging
import uuid

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_001_thermal_command.schemas import (
    WorkflowSpec,
    WorkflowResult,
    WorkflowStatus,
    TaskSpec,
    TaskResult,
    TaskStatus,
    Priority,
)
from greenlang.agents.process_heat.shared.coordination import (
    MultiAgentCoordinator,
    TaskSpec as CoordTaskSpec,
    Bid,
    AgentRegistration,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE COORDINATOR
# =============================================================================

class BaseCoordinator(ABC):
    """Base class for all coordinators."""

    def __init__(self, name: str) -> None:
        """Initialize the coordinator."""
        self.name = name
        self._active = False

    @abstractmethod
    async def start(self) -> None:
        """Start the coordinator."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the coordinator."""
        pass

    @property
    def is_active(self) -> bool:
        """Check if coordinator is active."""
        return self._active


# =============================================================================
# WORKFLOW COORDINATOR
# =============================================================================

class WorkflowCoordinator(BaseCoordinator):
    """
    Coordinates workflow execution across agents.

    This coordinator handles:
    - Workflow scheduling and execution
    - Task dependency resolution
    - Agent task assignment
    - Checkpoint management
    - Rollback on failure
    """

    def __init__(
        self,
        name: str = "WorkflowCoordinator",
        agent_coordinator: Optional[MultiAgentCoordinator] = None,
    ) -> None:
        """
        Initialize the workflow coordinator.

        Args:
            name: Coordinator name
            agent_coordinator: Multi-agent coordinator for task assignment
        """
        super().__init__(name)
        self._agent_coordinator = agent_coordinator
        self._active_workflows: Dict[str, WorkflowExecution] = {}
        self._workflow_history: List[WorkflowResult] = []
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the workflow coordinator."""
        self._active = True
        logger.info(f"WorkflowCoordinator '{self.name}' started")

    async def stop(self) -> None:
        """Stop the workflow coordinator."""
        self._active = False

        # Cancel active workflows
        for workflow_id in list(self._active_workflows.keys()):
            await self.cancel_workflow(workflow_id)

        logger.info(f"WorkflowCoordinator '{self.name}' stopped")

    async def execute_workflow(
        self,
        spec: WorkflowSpec,
    ) -> WorkflowResult:
        """
        Execute a workflow specification.

        Args:
            spec: Workflow specification

        Returns:
            WorkflowResult with execution results
        """
        logger.info(f"Starting workflow: {spec.workflow_id} ({spec.name})")

        # Create execution context
        execution = WorkflowExecution(
            workflow_id=spec.workflow_id,
            spec=spec,
        )
        self._active_workflows[spec.workflow_id] = execution

        try:
            # Build execution plan
            execution_order = self._build_execution_order(spec)
            logger.debug(f"Execution order: {execution_order}")

            # Execute tasks
            for task_batch in execution_order:
                # Execute batch in parallel
                batch_results = await asyncio.gather(*[
                    self._execute_task(spec.workflow_id, task)
                    for task in task_batch
                ], return_exceptions=True)

                # Process results
                for task, result in zip(task_batch, batch_results):
                    if isinstance(result, Exception):
                        execution.task_results[task.task_id] = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=str(result),
                        )
                        execution.tasks_failed += 1

                        # Check if should rollback
                        if spec.rollback_enabled:
                            await self._rollback_workflow(execution)
                            raise result
                    else:
                        execution.task_results[task.task_id] = result
                        if result.status == TaskStatus.COMPLETED:
                            execution.tasks_completed += 1
                        else:
                            execution.tasks_failed += 1

                # Checkpoint after batch
                if spec.checkpoint_enabled:
                    await self._create_checkpoint(execution)

            # Complete workflow
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now(timezone.utc)

            logger.info(
                f"Workflow completed: {spec.workflow_id} "
                f"({execution.tasks_completed}/{len(spec.tasks)} tasks)"
            )

        except asyncio.TimeoutError:
            execution.status = WorkflowStatus.TIMEOUT
            execution.end_time = datetime.now(timezone.utc)
            execution.error = "Workflow timeout exceeded"
            logger.error(f"Workflow timeout: {spec.workflow_id}")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.error = str(e)
            logger.error(f"Workflow failed: {spec.workflow_id}: {e}")

        finally:
            # Move to history
            result = execution.to_result()
            self._workflow_history.append(result)
            del self._active_workflows[spec.workflow_id]

        return result

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel an active workflow.

        Args:
            workflow_id: Workflow to cancel

        Returns:
            True if cancelled successfully
        """
        execution = self._active_workflows.get(workflow_id)
        if not execution:
            return False

        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now(timezone.utc)

        logger.info(f"Workflow cancelled: {workflow_id}")
        return True

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause an active workflow."""
        execution = self._active_workflows.get(workflow_id)
        if not execution:
            return False

        execution.status = WorkflowStatus.PAUSED
        logger.info(f"Workflow paused: {workflow_id}")
        return True

    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        execution = self._active_workflows.get(workflow_id)
        if not execution or execution.status != WorkflowStatus.PAUSED:
            return False

        execution.status = WorkflowStatus.RUNNING
        logger.info(f"Workflow resumed: {workflow_id}")
        return True

    def _build_execution_order(
        self,
        spec: WorkflowSpec,
    ) -> List[List[TaskSpec]]:
        """
        Build task execution order respecting dependencies.

        Returns batches of tasks that can run in parallel.
        """
        # Build dependency graph
        task_map = {task.task_id: task for task in spec.tasks}
        dependencies = spec.dependencies.copy()

        # Find tasks with no dependencies
        execution_order = []
        completed: Set[str] = set()

        while len(completed) < len(spec.tasks):
            # Find ready tasks
            ready = []
            for task in spec.tasks:
                if task.task_id in completed:
                    continue

                task_deps = dependencies.get(task.task_id, [])
                if all(dep in completed for dep in task_deps):
                    ready.append(task)

            if not ready:
                # Circular dependency detected
                remaining = [
                    t.task_id for t in spec.tasks
                    if t.task_id not in completed
                ]
                raise ValueError(
                    f"Circular dependency detected in tasks: {remaining}"
                )

            execution_order.append(ready)
            completed.update(t.task_id for t in ready)

        return execution_order

    async def _execute_task(
        self,
        workflow_id: str,
        task: TaskSpec,
    ) -> TaskResult:
        """Execute a single task."""
        logger.debug(f"Executing task: {task.task_id} ({task.name})")

        start_time = datetime.now(timezone.utc)

        try:
            # Use contract net protocol to assign task
            if self._agent_coordinator:
                # Create coordination task spec
                coord_task = CoordTaskSpec(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    required_capabilities={task.target_agent_type},
                    parameters=task.inputs,
                )

                # Announce and wait for bids
                self._agent_coordinator.announce_task(coord_task)

                # Wait for bidding period
                await asyncio.sleep(2.0)  # Simplified bidding

                # Award to best bidder
                winner = self._agent_coordinator.evaluate_and_award(
                    task.task_id,
                    strategy="balanced",
                )

                if not winner:
                    raise RuntimeError(f"No agent available for task {task.task_id}")

                # Simulated task execution
                # In production, this would send the task to the winning agent
                await asyncio.sleep(0.1)

                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    assigned_agent=winner,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    output={"result": "success"},
                )

            else:
                # No coordinator, simulate execution
                await asyncio.sleep(0.1)

                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    output={"result": "simulated"},
                )

            result.duration_ms = (
                result.end_time - start_time
            ).total_seconds() * 1000

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error=str(e),
            )

    async def _create_checkpoint(self, execution: "WorkflowExecution") -> str:
        """Create a workflow checkpoint."""
        checkpoint_id = f"{execution.workflow_id}_{len(execution.checkpoints)}"

        checkpoint_data = {
            "workflow_id": execution.workflow_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_tasks": list(execution.task_results.keys()),
            "task_results": {
                tid: tr.dict() for tid, tr in execution.task_results.items()
            },
        }

        self._checkpoints[checkpoint_id] = checkpoint_data
        execution.checkpoints.append(checkpoint_id)

        logger.debug(f"Checkpoint created: {checkpoint_id}")
        return checkpoint_id

    async def _rollback_workflow(self, execution: "WorkflowExecution") -> None:
        """Rollback a failed workflow."""
        logger.warning(f"Rolling back workflow: {execution.workflow_id}")

        # In production, this would undo completed tasks
        # For now, just log the rollback
        for task_id, result in execution.task_results.items():
            if result.status == TaskStatus.COMPLETED:
                logger.info(f"Rolling back task: {task_id}")

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get current workflow status."""
        execution = self._active_workflows.get(workflow_id)
        if execution:
            return execution.status
        return None

    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs."""
        return list(self._active_workflows.keys())


class WorkflowExecution(BaseModel):
    """Workflow execution context."""

    workflow_id: str = Field(..., description="Workflow ID")
    spec: WorkflowSpec = Field(..., description="Workflow specification")
    status: WorkflowStatus = Field(
        default=WorkflowStatus.RUNNING,
        description="Current status"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Start time"
    )
    end_time: Optional[datetime] = Field(default=None, description="End time")
    task_results: Dict[str, TaskResult] = Field(
        default_factory=dict,
        description="Task results"
    )
    tasks_completed: int = Field(default=0, description="Completed tasks")
    tasks_failed: int = Field(default=0, description="Failed tasks")
    checkpoints: List[str] = Field(
        default_factory=list,
        description="Checkpoint IDs"
    )
    error: Optional[str] = Field(default=None, description="Error message")

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def to_result(self) -> WorkflowResult:
        """Convert to WorkflowResult."""
        duration_ms = 0.0
        if self.end_time and self.start_time:
            duration_ms = (
                self.end_time - self.start_time
            ).total_seconds() * 1000

        return WorkflowResult(
            workflow_id=self.workflow_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_ms=duration_ms,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            tasks_total=len(self.spec.tasks),
            task_results=self.task_results,
            error=self.error,
            checkpoints=self.checkpoints,
        )


# =============================================================================
# SAFETY COORDINATOR
# =============================================================================

class SafetyCoordinator(BaseCoordinator):
    """
    Coordinates safety-related operations across agents.

    This coordinator handles:
    - Safety interlock management
    - Emergency shutdown coordination
    - Alarm aggregation and escalation
    - Safety permit management
    """

    def __init__(
        self,
        name: str = "SafetyCoordinator",
        sil_level: int = 3,
    ) -> None:
        """
        Initialize the safety coordinator.

        Args:
            name: Coordinator name
            sil_level: Safety Integrity Level (1-4)
        """
        super().__init__(name)
        self._sil_level = sil_level
        self._interlocks: Dict[str, SafetyInterlock] = {}
        self._permits: Dict[str, SafetyPermit] = {}
        self._esd_triggered = False
        self._safety_state = "normal"

    async def start(self) -> None:
        """Start the safety coordinator."""
        self._active = True
        self._safety_state = "normal"
        logger.info(f"SafetyCoordinator '{self.name}' started (SIL-{self._sil_level})")

    async def stop(self) -> None:
        """Stop the safety coordinator."""
        self._active = False
        logger.info(f"SafetyCoordinator '{self.name}' stopped")

    def register_interlock(
        self,
        interlock_id: str,
        condition: str,
        action: str,
        threshold: float,
    ) -> None:
        """
        Register a safety interlock.

        Args:
            interlock_id: Interlock identifier
            condition: Condition description
            action: Action to take when triggered
            threshold: Trigger threshold
        """
        self._interlocks[interlock_id] = SafetyInterlock(
            interlock_id=interlock_id,
            condition=condition,
            action=action,
            threshold=threshold,
        )
        logger.info(f"Interlock registered: {interlock_id}")

    async def check_interlock(
        self,
        interlock_id: str,
        current_value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check an interlock condition.

        Args:
            interlock_id: Interlock to check
            current_value: Current process value

        Returns:
            Tuple of (is_safe, action_if_triggered)
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            return True, None

        if current_value > interlock.threshold:
            interlock.triggered = True
            interlock.trigger_time = datetime.now(timezone.utc)
            logger.warning(
                f"Interlock triggered: {interlock_id} "
                f"(value={current_value}, threshold={interlock.threshold})"
            )
            return False, interlock.action

        return True, None

    async def trigger_esd(self, reason: str) -> None:
        """
        Trigger Emergency Shutdown.

        Args:
            reason: Reason for ESD
        """
        if self._esd_triggered:
            logger.warning("ESD already triggered")
            return

        self._esd_triggered = True
        self._safety_state = "emergency_shutdown"

        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")

        # In production, this would:
        # 1. Send ESD signal to DCS/PLC
        # 2. Close all fuel valves
        # 3. Open all vent valves
        # 4. Stop all rotating equipment
        # 5. Notify all agents

    async def reset_esd(self, authorized_by: str) -> bool:
        """
        Reset Emergency Shutdown.

        Args:
            authorized_by: Person authorizing reset

        Returns:
            True if reset successful
        """
        if not self._esd_triggered:
            return True

        logger.info(f"ESD reset authorized by: {authorized_by}")

        # Check all interlocks are clear
        for interlock in self._interlocks.values():
            if interlock.triggered:
                logger.warning(
                    f"Cannot reset ESD: Interlock {interlock.interlock_id} still triggered"
                )
                return False

        self._esd_triggered = False
        self._safety_state = "normal"
        logger.info("ESD reset complete")
        return True

    def request_permit(
        self,
        permit_type: str,
        equipment_id: str,
        requested_by: str,
        duration_hours: float = 8.0,
    ) -> Optional[str]:
        """
        Request a safety permit.

        Args:
            permit_type: Type of permit (hot_work, confined_space, etc.)
            equipment_id: Equipment to work on
            requested_by: Person requesting
            duration_hours: Permit duration

        Returns:
            Permit ID if approved, None if denied
        """
        # Check safety state
        if self._safety_state != "normal":
            logger.warning(f"Permit denied: System in {self._safety_state} state")
            return None

        permit_id = str(uuid.uuid4())[:8]
        self._permits[permit_id] = SafetyPermit(
            permit_id=permit_id,
            permit_type=permit_type,
            equipment_id=equipment_id,
            requested_by=requested_by,
            duration_hours=duration_hours,
        )

        logger.info(f"Permit issued: {permit_id} ({permit_type}) for {equipment_id}")
        return permit_id

    def close_permit(self, permit_id: str) -> bool:
        """Close a safety permit."""
        permit = self._permits.get(permit_id)
        if not permit:
            return False

        permit.active = False
        permit.closed_time = datetime.now(timezone.utc)
        logger.info(f"Permit closed: {permit_id}")
        return True

    @property
    def safety_state(self) -> str:
        """Get current safety state."""
        return self._safety_state

    @property
    def is_esd_triggered(self) -> bool:
        """Check if ESD is triggered."""
        return self._esd_triggered


class SafetyInterlock(BaseModel):
    """Safety interlock definition."""

    interlock_id: str
    condition: str
    action: str
    threshold: float
    triggered: bool = False
    trigger_time: Optional[datetime] = None


class SafetyPermit(BaseModel):
    """Safety permit."""

    permit_id: str
    permit_type: str
    equipment_id: str
    requested_by: str
    duration_hours: float
    issued_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    closed_time: Optional[datetime] = None
    active: bool = True


# =============================================================================
# OPTIMIZATION COORDINATOR
# =============================================================================

class OptimizationCoordinator(BaseCoordinator):
    """
    Coordinates optimization across multiple agents.

    This coordinator handles:
    - Multi-objective optimization
    - Constraint management
    - Real-time optimization adjustments
    - Performance tracking
    """

    def __init__(
        self,
        name: str = "OptimizationCoordinator",
    ) -> None:
        """Initialize the optimization coordinator."""
        super().__init__(name)
        self._optimization_targets: Dict[str, OptimizationTarget] = {}
        self._constraints: List[OptimizationConstraint] = []
        self._current_setpoints: Dict[str, float] = {}
        self._optimization_history: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """Start the optimization coordinator."""
        self._active = True
        logger.info(f"OptimizationCoordinator '{self.name}' started")

    async def stop(self) -> None:
        """Stop the optimization coordinator."""
        self._active = False
        logger.info(f"OptimizationCoordinator '{self.name}' stopped")

    def add_target(
        self,
        target_id: str,
        name: str,
        direction: str,
        weight: float = 1.0,
    ) -> None:
        """
        Add an optimization target.

        Args:
            target_id: Target identifier
            name: Target name
            direction: "minimize" or "maximize"
            weight: Relative weight in multi-objective optimization
        """
        self._optimization_targets[target_id] = OptimizationTarget(
            target_id=target_id,
            name=name,
            direction=direction,
            weight=weight,
        )
        logger.info(f"Optimization target added: {name} ({direction})")

    def add_constraint(
        self,
        constraint_id: str,
        variable: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        """
        Add an optimization constraint.

        Args:
            constraint_id: Constraint identifier
            variable: Variable name
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        self._constraints.append(OptimizationConstraint(
            constraint_id=constraint_id,
            variable=variable,
            min_value=min_value,
            max_value=max_value,
        ))
        logger.info(f"Constraint added: {variable} [{min_value}, {max_value}]")

    async def optimize(
        self,
        current_state: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Run optimization to find optimal setpoints.

        Args:
            current_state: Current process state

        Returns:
            Optimized setpoints
        """
        # Check constraints
        for constraint in self._constraints:
            value = current_state.get(constraint.variable)
            if value is not None:
                if constraint.min_value and value < constraint.min_value:
                    logger.warning(
                        f"Constraint violation: {constraint.variable} "
                        f"({value} < {constraint.min_value})"
                    )
                if constraint.max_value and value > constraint.max_value:
                    logger.warning(
                        f"Constraint violation: {constraint.variable} "
                        f"({value} > {constraint.max_value})"
                    )

        # Simple optimization logic (in production, use scipy.optimize)
        optimized_setpoints = current_state.copy()

        # Record history
        self._optimization_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_state": current_state,
            "output_setpoints": optimized_setpoints,
        })

        self._current_setpoints = optimized_setpoints
        return optimized_setpoints

    def get_current_setpoints(self) -> Dict[str, float]:
        """Get current optimized setpoints."""
        return self._current_setpoints.copy()


class OptimizationTarget(BaseModel):
    """Optimization target definition."""

    target_id: str
    name: str
    direction: str  # "minimize" or "maximize"
    weight: float = 1.0
    current_value: Optional[float] = None


class OptimizationConstraint(BaseModel):
    """Optimization constraint."""

    constraint_id: str
    variable: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
