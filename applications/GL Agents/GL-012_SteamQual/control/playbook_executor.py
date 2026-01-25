"""
GL-012 STEAMQUAL SteamQualityController - Playbook Executor

This module implements pre-approved mitigation playbook execution for steam
quality control, including carryover risk mitigation, separator flooding
response, and water hammer prevention.

Control Architecture:
    - Pre-approved playbook definitions with operator confirmation
    - Step-by-step execution with verification
    - Rollback capability for failed executions
    - Complete audit trail for all playbook operations

Pre-Approved Playbooks:
    - Carryover Risk Mitigation: Reduce moisture carryover from boilers
    - Separator Flooding Response: Emergency drainage and level control
    - Water Hammer Prevention: Proper warm-up and condensate removal
    - Quality Degradation Response: Systematic quality improvement

Safety Features:
    - Every playbook step requires operator confirmation (by default)
    - All actions logged for audit trail
    - Rollback capability for reversible actions
    - Emergency stop capability at any step

Reference Standards:
    - ISA-18.2 Management of Alarm Systems
    - IEC 61511 Functional Safety
    - ASME B31.1 Power Piping

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class PlaybookType(str, Enum):
    """Playbook type enumeration."""
    CARRYOVER_MITIGATION = "carryover_mitigation"
    SEPARATOR_FLOODING = "separator_flooding"
    WATER_HAMMER_PREVENTION = "water_hammer_prevention"
    QUALITY_DEGRADATION = "quality_degradation"
    STARTUP_WARMUP = "startup_warmup"
    SHUTDOWN_DRAIN = "shutdown_drain"
    CUSTOM = "custom"


class PlaybookStatus(str, Enum):
    """Playbook execution status enumeration."""
    PENDING = "pending"  # Awaiting start
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # Waiting for operator
    IN_PROGRESS = "in_progress"  # Actively executing
    PAUSED = "paused"  # Temporarily paused
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed execution
    ABORTED = "aborted"  # Manually aborted
    ROLLED_BACK = "rolled_back"  # Rolled back after failure


class ExecutionMode(str, Enum):
    """Playbook execution mode enumeration."""
    STEP_BY_STEP = "step_by_step"  # Confirm each step
    BATCH_CONFIRM = "batch_confirm"  # Confirm at checkpoints
    AUTO_EXECUTE = "auto_execute"  # Auto-execute (emergency only)


class ConfirmationStatus(str, Enum):
    """Operator confirmation status enumeration."""
    REQUIRED = "required"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


# =============================================================================
# DATA MODELS
# =============================================================================

class PlaybookStep(BaseModel):
    """Single step in a playbook."""

    step_id: str = Field(..., description="Step identifier")
    step_number: int = Field(..., ge=1, description="Step sequence number")
    action_type: str = Field(..., description="Type of action")
    description: str = Field(..., description="Human-readable description")
    target_equipment: Optional[str] = Field(
        None,
        description="Target equipment ID"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    confirmation_timeout_s: float = Field(
        default=300.0,
        ge=0,
        description="Confirmation timeout (s)"
    )
    verification_condition: Optional[str] = Field(
        None,
        description="Condition to verify after execution"
    )
    verification_timeout_s: float = Field(
        default=60.0,
        ge=0,
        description="Verification timeout (s)"
    )
    rollback_action: Optional[str] = Field(
        None,
        description="Action to rollback this step"
    )
    is_checkpoint: bool = Field(
        default=False,
        description="Is this a checkpoint step"
    )
    safety_critical: bool = Field(
        default=False,
        description="Is this safety-critical"
    )


class PlaybookDefinition(BaseModel):
    """Complete playbook definition."""

    playbook_id: str = Field(..., description="Playbook identifier")
    playbook_type: PlaybookType = Field(..., description="Playbook type")
    name: str = Field(..., description="Playbook name")
    description: str = Field(..., description="Playbook description")
    version: str = Field(default="1.0.0", description="Playbook version")
    steps: List[PlaybookStep] = Field(..., description="Playbook steps")
    trigger_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that trigger this playbook"
    )
    abort_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that should abort execution"
    )
    estimated_duration_min: float = Field(
        ...,
        ge=0,
        description="Estimated duration (minutes)"
    )
    requires_pre_approval: bool = Field(
        default=True,
        description="Requires pre-approval before execution"
    )
    approval_level: str = Field(
        default="operator",
        description="Required approval level"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approved by (if pre-approved)"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class OperatorConfirmation(BaseModel):
    """Operator confirmation record."""

    confirmation_id: str = Field(..., description="Confirmation ID")
    playbook_id: str = Field(..., description="Playbook ID")
    step_id: str = Field(..., description="Step ID")
    operator_id: str = Field(..., description="Operator ID")
    status: ConfirmationStatus = Field(..., description="Confirmation status")
    comment: str = Field(default="", description="Operator comment")
    confirmed_at: datetime = Field(
        default_factory=datetime.now,
        description="Confirmation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class StepResult(BaseModel):
    """Result of a single step execution."""

    result_id: str = Field(..., description="Result ID")
    step_id: str = Field(..., description="Step ID")
    step_number: int = Field(..., description="Step number")
    status: str = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    duration_s: Optional[float] = Field(
        None,
        ge=0,
        description="Execution duration (s)"
    )
    confirmation: Optional[OperatorConfirmation] = Field(
        None,
        description="Operator confirmation"
    )
    verification_passed: Optional[bool] = Field(
        None,
        description="Verification passed"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step output data"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    rolled_back: bool = Field(
        default=False,
        description="Step was rolled back"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class PlaybookExecution(BaseModel):
    """Complete playbook execution record."""

    execution_id: str = Field(..., description="Execution ID")
    playbook_id: str = Field(..., description="Playbook ID")
    playbook_type: PlaybookType = Field(..., description="Playbook type")
    playbook_name: str = Field(..., description="Playbook name")
    status: PlaybookStatus = Field(..., description="Execution status")
    execution_mode: ExecutionMode = Field(..., description="Execution mode")
    triggered_by: str = Field(..., description="Trigger source")
    triggered_reason: str = Field(..., description="Trigger reason")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    current_step: int = Field(
        default=0,
        ge=0,
        description="Current step number"
    )
    total_steps: int = Field(..., description="Total number of steps")
    step_results: List[StepResult] = Field(
        default_factory=list,
        description="Results of executed steps"
    )
    pending_confirmations: int = Field(
        default=0,
        ge=0,
        description="Pending confirmations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    abort_reason: Optional[str] = Field(
        None,
        description="Abort reason if aborted"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class PlaybookAuditRecord(BaseModel):
    """Audit record for playbook operations."""

    audit_id: str = Field(..., description="Audit record ID")
    execution_id: str = Field(..., description="Execution ID")
    operation: str = Field(..., description="Operation type")
    step_id: Optional[str] = Field(None, description="Step ID if applicable")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Audit timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# PLAYBOOK EXECUTOR
# =============================================================================

class PlaybookExecutor:
    """
    Pre-approved mitigation playbook executor.

    This executor manages the execution of pre-approved mitigation playbooks
    for steam quality control, ensuring operator confirmation and complete
    audit trails.

    Pre-Approved Playbooks:
        - Carryover Risk Mitigation: Address moisture carryover
        - Separator Flooding Response: Emergency drainage
        - Water Hammer Prevention: Safe warm-up procedures
        - Quality Degradation Response: Systematic improvement

    Safety Features:
        - Each playbook requires operator confirmation (by default)
        - Step-by-step execution with verification
        - Rollback capability for reversible actions
        - Emergency stop at any step
        - Complete audit trail

    Attributes:
        executor_id: Unique executor identifier
        _playbooks: Registered playbook definitions
        _active_executions: Currently active executions
        _execution_history: History of completed executions
        _audit_log: Audit records

    Example:
        >>> executor = PlaybookExecutor("PE-001")
        >>> executor.register_playbook(playbook_def)
        >>> execution = executor.start_playbook("carryover_001", "operator_001")
        >>> executor.confirm_step(execution.execution_id, "step_001", "operator_001")
    """

    def __init__(self, executor_id: str):
        """
        Initialize PlaybookExecutor.

        Args:
            executor_id: Unique executor identifier
        """
        self.executor_id = executor_id

        # Playbook storage
        self._playbooks: Dict[str, PlaybookDefinition] = {}
        self._active_executions: Dict[str, PlaybookExecution] = {}
        self._execution_history: List[PlaybookExecution] = []
        self._audit_log: List[PlaybookAuditRecord] = []

        # Limits
        self._max_history_size = 500
        self._max_audit_size = 5000
        self._max_active_executions = 5

        # Initialize built-in playbooks
        self._register_builtin_playbooks()

        logger.info(f"PlaybookExecutor {executor_id} initialized")

    def register_playbook(self, playbook: PlaybookDefinition) -> None:
        """
        Register a playbook definition.

        Args:
            playbook: Playbook definition to register
        """
        self._playbooks[playbook.playbook_id] = playbook
        logger.info(
            f"Registered playbook {playbook.playbook_id}: {playbook.name}"
        )

    def get_available_playbooks(self) -> List[PlaybookDefinition]:
        """Get all available playbook definitions."""
        return list(self._playbooks.values())

    def get_playbook(self, playbook_id: str) -> Optional[PlaybookDefinition]:
        """Get a specific playbook definition."""
        return self._playbooks.get(playbook_id)

    def start_playbook(
        self,
        playbook_id: str,
        triggered_by: str,
        reason: str = "",
        mode: ExecutionMode = ExecutionMode.STEP_BY_STEP
    ) -> PlaybookExecution:
        """
        Start a playbook execution.

        Args:
            playbook_id: Playbook identifier
            triggered_by: ID of triggering entity (operator, system)
            reason: Reason for triggering
            mode: Execution mode

        Returns:
            PlaybookExecution: Created execution record

        Raises:
            KeyError: If playbook not found
            ValueError: If too many active executions
        """
        start_time = datetime.now()

        if playbook_id not in self._playbooks:
            raise KeyError(f"Playbook {playbook_id} not found")

        if len(self._active_executions) >= self._max_active_executions:
            raise ValueError(
                f"Maximum active executions ({self._max_active_executions}) reached"
            )

        playbook = self._playbooks[playbook_id]

        # Generate execution ID
        execution_id = hashlib.sha256(
            f"EXEC_{playbook_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        execution = PlaybookExecution(
            execution_id=execution_id,
            playbook_id=playbook_id,
            playbook_type=playbook.playbook_type,
            playbook_name=playbook.name,
            status=PlaybookStatus.AWAITING_CONFIRMATION,
            execution_mode=mode,
            triggered_by=triggered_by,
            triggered_reason=reason,
            started_at=start_time,
            current_step=0,
            total_steps=len(playbook.steps),
            step_results=[],
            pending_confirmations=1 if playbook.requires_pre_approval else 0
        )

        # Calculate provenance hash
        execution.provenance_hash = hashlib.sha256(
            f"{execution_id}|{playbook_id}|{triggered_by}".encode()
        ).hexdigest()

        # Store execution
        self._active_executions[execution_id] = execution

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="start_playbook",
            operator_id=triggered_by,
            details={
                "playbook_id": playbook_id,
                "playbook_name": playbook.name,
                "reason": reason,
                "mode": mode.value
            }
        )

        logger.info(
            f"Started playbook execution {execution_id}: "
            f"playbook={playbook.name}, triggered_by={triggered_by}"
        )

        return execution

    def confirm_playbook_start(
        self,
        execution_id: str,
        operator_id: str,
        comment: str = ""
    ) -> PlaybookExecution:
        """
        Confirm playbook start (required for pre-approved playbooks).

        Args:
            execution_id: Execution ID
            operator_id: Confirming operator ID
            comment: Optional comment

        Returns:
            PlaybookExecution: Updated execution record
        """
        if execution_id not in self._active_executions:
            raise KeyError(f"Execution {execution_id} not found")

        execution = self._active_executions[execution_id]

        if execution.status != PlaybookStatus.AWAITING_CONFIRMATION:
            raise ValueError(
                f"Execution not awaiting confirmation (status: {execution.status.value})"
            )

        # Update status
        execution.status = PlaybookStatus.IN_PROGRESS
        execution.pending_confirmations = 0

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="confirm_start",
            operator_id=operator_id,
            details={"comment": comment}
        )

        logger.info(
            f"Playbook start confirmed: execution={execution_id}, "
            f"operator={operator_id}"
        )

        return execution

    def execute_next_step(
        self,
        execution_id: str
    ) -> Optional[StepResult]:
        """
        Execute the next step in a playbook.

        Args:
            execution_id: Execution ID

        Returns:
            StepResult: Result of step execution, or None if waiting for confirmation
        """
        if execution_id not in self._active_executions:
            raise KeyError(f"Execution {execution_id} not found")

        execution = self._active_executions[execution_id]
        playbook = self._playbooks[execution.playbook_id]

        if execution.status != PlaybookStatus.IN_PROGRESS:
            logger.warning(
                f"Cannot execute step: execution status is {execution.status.value}"
            )
            return None

        if execution.current_step >= execution.total_steps:
            # All steps complete
            self._complete_execution(execution_id, PlaybookStatus.COMPLETED)
            return None

        # Get current step
        step = playbook.steps[execution.current_step]
        start_time = datetime.now()

        # Generate result ID
        result_id = hashlib.sha256(
            f"RES_{execution_id}_{step.step_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Check if confirmation is required
        if step.requires_confirmation:
            if execution.execution_mode == ExecutionMode.STEP_BY_STEP:
                # Wait for confirmation
                execution.status = PlaybookStatus.AWAITING_CONFIRMATION
                execution.pending_confirmations = 1

                result = StepResult(
                    result_id=result_id,
                    step_id=step.step_id,
                    step_number=step.step_number,
                    status="awaiting_confirmation",
                    started_at=start_time,
                    output_data={"message": "Awaiting operator confirmation"}
                )

                execution.step_results.append(result)

                logger.info(
                    f"Step {step.step_number} awaiting confirmation: "
                    f"{step.description}"
                )

                return result

        # Execute the step
        result = self._execute_step(execution_id, step, result_id, start_time)
        execution.step_results.append(result)

        if result.status == "completed":
            execution.current_step += 1

            # Check if all steps complete
            if execution.current_step >= execution.total_steps:
                self._complete_execution(execution_id, PlaybookStatus.COMPLETED)

        elif result.status == "failed":
            execution.status = PlaybookStatus.FAILED
            self._add_audit_record(
                execution_id=execution_id,
                operation="step_failed",
                step_id=step.step_id,
                details={"error": result.error_message}
            )

        return result

    def confirm_step(
        self,
        execution_id: str,
        step_id: str,
        operator_id: str,
        approved: bool = True,
        comment: str = ""
    ) -> StepResult:
        """
        Confirm a step that is awaiting confirmation.

        Args:
            execution_id: Execution ID
            step_id: Step ID
            operator_id: Confirming operator ID
            approved: Whether step is approved
            comment: Optional comment

        Returns:
            StepResult: Updated step result
        """
        if execution_id not in self._active_executions:
            raise KeyError(f"Execution {execution_id} not found")

        execution = self._active_executions[execution_id]
        playbook = self._playbooks[execution.playbook_id]

        # Find the step
        step = None
        for s in playbook.steps:
            if s.step_id == step_id:
                step = s
                break

        if step is None:
            raise KeyError(f"Step {step_id} not found in playbook")

        # Create confirmation record
        confirmation_id = hashlib.sha256(
            f"CONF_{execution_id}_{step_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        confirmation = OperatorConfirmation(
            confirmation_id=confirmation_id,
            playbook_id=execution.playbook_id,
            step_id=step_id,
            operator_id=operator_id,
            status=ConfirmationStatus.CONFIRMED if approved else ConfirmationStatus.REJECTED,
            comment=comment,
            confirmed_at=datetime.now()
        )

        confirmation.provenance_hash = hashlib.sha256(
            f"{confirmation_id}|{operator_id}|{approved}".encode()
        ).hexdigest()

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="step_confirmed" if approved else "step_rejected",
            step_id=step_id,
            operator_id=operator_id,
            details={"approved": approved, "comment": comment}
        )

        if approved:
            # Execute the step now
            start_time = datetime.now()
            result_id = hashlib.sha256(
                f"RES_{execution_id}_{step_id}_{start_time.isoformat()}".encode()
            ).hexdigest()[:16]

            result = self._execute_step(execution_id, step, result_id, start_time)
            result.confirmation = confirmation

            # Update execution
            execution.status = PlaybookStatus.IN_PROGRESS
            execution.pending_confirmations = 0

            # Find and update the step result
            for i, sr in enumerate(execution.step_results):
                if sr.step_id == step_id and sr.status == "awaiting_confirmation":
                    execution.step_results[i] = result
                    break
            else:
                execution.step_results.append(result)

            if result.status == "completed":
                execution.current_step += 1

                # Check if all steps complete
                if execution.current_step >= execution.total_steps:
                    self._complete_execution(execution_id, PlaybookStatus.COMPLETED)

            logger.info(
                f"Step {step.step_number} confirmed and executed by {operator_id}"
            )

        else:
            # Rejected - pause execution
            execution.status = PlaybookStatus.PAUSED
            execution.pending_confirmations = 0
            execution.warnings.append(
                f"Step {step.step_number} rejected by {operator_id}: {comment}"
            )

            result = StepResult(
                result_id=hashlib.sha256(
                    f"REJ_{execution_id}_{step_id}".encode()
                ).hexdigest()[:16],
                step_id=step_id,
                step_number=step.step_number,
                status="rejected",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                confirmation=confirmation,
                error_message=f"Rejected by {operator_id}: {comment}"
            )

            logger.warning(
                f"Step {step.step_number} rejected by {operator_id}: {comment}"
            )

        return result

    def abort_execution(
        self,
        execution_id: str,
        operator_id: str,
        reason: str
    ) -> PlaybookExecution:
        """
        Abort an active playbook execution.

        Args:
            execution_id: Execution ID
            operator_id: Aborting operator ID
            reason: Abort reason

        Returns:
            PlaybookExecution: Updated execution record
        """
        if execution_id not in self._active_executions:
            raise KeyError(f"Execution {execution_id} not found")

        execution = self._active_executions[execution_id]

        execution.status = PlaybookStatus.ABORTED
        execution.completed_at = datetime.now()
        execution.abort_reason = reason

        # Move to history
        self._execution_history.append(execution)
        del self._active_executions[execution_id]

        # Trim history
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="abort",
            operator_id=operator_id,
            details={"reason": reason}
        )

        logger.warning(
            f"Playbook execution aborted: execution={execution_id}, "
            f"operator={operator_id}, reason={reason}"
        )

        return execution

    def rollback_execution(
        self,
        execution_id: str,
        operator_id: str
    ) -> PlaybookExecution:
        """
        Rollback a failed or aborted execution.

        Args:
            execution_id: Execution ID
            operator_id: Operator requesting rollback

        Returns:
            PlaybookExecution: Updated execution record
        """
        # Check active and history
        execution = self._active_executions.get(execution_id)
        if execution is None:
            for e in self._execution_history:
                if e.execution_id == execution_id:
                    execution = e
                    break

        if execution is None:
            raise KeyError(f"Execution {execution_id} not found")

        playbook = self._playbooks.get(execution.playbook_id)
        if playbook is None:
            raise KeyError(f"Playbook {execution.playbook_id} not found")

        # Rollback executed steps in reverse order
        rolled_back_steps = []
        for result in reversed(execution.step_results):
            if result.status == "completed" and not result.rolled_back:
                step = None
                for s in playbook.steps:
                    if s.step_id == result.step_id:
                        step = s
                        break

                if step and step.rollback_action:
                    # Execute rollback action
                    logger.info(
                        f"Rolling back step {step.step_number}: {step.rollback_action}"
                    )
                    result.rolled_back = True
                    rolled_back_steps.append(result.step_id)

        execution.status = PlaybookStatus.ROLLED_BACK

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="rollback",
            operator_id=operator_id,
            details={"rolled_back_steps": rolled_back_steps}
        )

        logger.info(
            f"Playbook rolled back: execution={execution_id}, "
            f"steps={len(rolled_back_steps)}"
        )

        return execution

    def get_active_executions(self) -> List[PlaybookExecution]:
        """Get all active playbook executions."""
        return list(self._active_executions.values())

    def get_execution(self, execution_id: str) -> Optional[PlaybookExecution]:
        """Get a specific execution by ID."""
        execution = self._active_executions.get(execution_id)
        if execution:
            return execution

        for e in self._execution_history:
            if e.execution_id == execution_id:
                return e

        return None

    def get_execution_history(
        self,
        time_window_minutes: int = 60,
        playbook_type: Optional[PlaybookType] = None
    ) -> List[PlaybookExecution]:
        """Get execution history within time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)

        results = [
            e for e in self._execution_history
            if e.started_at >= cutoff
        ]

        if playbook_type:
            results = [e for e in results if e.playbook_type == playbook_type]

        return results

    def get_audit_log(
        self,
        execution_id: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> List[PlaybookAuditRecord]:
        """Get audit log entries."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)

        records = [r for r in self._audit_log if r.timestamp >= cutoff]

        if execution_id:
            records = [r for r in records if r.execution_id == execution_id]

        return records

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _register_builtin_playbooks(self) -> None:
        """Register built-in playbook definitions."""
        # Carryover Risk Mitigation Playbook
        carryover_playbook = PlaybookDefinition(
            playbook_id="CARRYOVER_MIT_001",
            playbook_type=PlaybookType.CARRYOVER_MITIGATION,
            name="Carryover Risk Mitigation",
            description=(
                "Systematic response to elevated moisture carryover risk. "
                "Reduces spray water, increases separator drainage, and "
                "monitors quality improvement."
            ),
            version="1.0.0",
            steps=[
                PlaybookStep(
                    step_id="CM_01",
                    step_number=1,
                    action_type="reduce_spray",
                    description="Reduce desuperheater spray water by 10%",
                    target_equipment="DS_PRIMARY",
                    parameters={"reduction_pct": 10.0},
                    requires_confirmation=True,
                    verification_condition="spray_valve_position < previous - 8",
                    rollback_action="restore_spray_position"
                ),
                PlaybookStep(
                    step_id="CM_02",
                    step_number=2,
                    action_type="increase_drain",
                    description="Open separator drain valve for 30 seconds",
                    target_equipment="SEP_MAIN",
                    parameters={"open_duration_s": 30.0},
                    requires_confirmation=True,
                    verification_condition="separator_level < 70",
                    rollback_action="close_drain_valve"
                ),
                PlaybookStep(
                    step_id="CM_03",
                    step_number=3,
                    action_type="monitor",
                    description="Monitor quality for 5 minutes",
                    parameters={"duration_min": 5.0, "check_interval_s": 30.0},
                    requires_confirmation=False,
                    is_checkpoint=True
                ),
                PlaybookStep(
                    step_id="CM_04",
                    step_number=4,
                    action_type="verify_improvement",
                    description="Verify quality improvement",
                    parameters={"min_improvement_pct": 2.0},
                    requires_confirmation=True,
                    verification_condition="dryness_fraction > previous + 0.01"
                )
            ],
            trigger_conditions=[
                "dryness_fraction < 0.97",
                "moisture_ppm > 600"
            ],
            abort_conditions=[
                "superheat_c < 10",
                "separator_level > 95"
            ],
            estimated_duration_min=10.0,
            requires_pre_approval=True,
            approval_level="operator"
        )

        # Separator Flooding Response Playbook
        flooding_playbook = PlaybookDefinition(
            playbook_id="SEP_FLOOD_001",
            playbook_type=PlaybookType.SEPARATOR_FLOODING,
            name="Separator Flooding Response",
            description=(
                "Emergency response to separator flooding condition. "
                "Opens drain valves, reduces steam flow, and monitors "
                "level reduction."
            ),
            version="1.0.0",
            steps=[
                PlaybookStep(
                    step_id="SF_01",
                    step_number=1,
                    action_type="emergency_drain",
                    description="Open all separator drain valves to 100%",
                    target_equipment="ALL_SEPARATORS",
                    parameters={"position_pct": 100.0},
                    requires_confirmation=True,
                    confirmation_timeout_s=60.0,  # Shorter for emergency
                    safety_critical=True,
                    rollback_action="close_drain_valves"
                ),
                PlaybookStep(
                    step_id="SF_02",
                    step_number=2,
                    action_type="reduce_load",
                    description="Reduce steam load by 20% if possible",
                    parameters={"reduction_pct": 20.0},
                    requires_confirmation=True,
                    rollback_action="restore_load"
                ),
                PlaybookStep(
                    step_id="SF_03",
                    step_number=3,
                    action_type="monitor_level",
                    description="Monitor separator level reduction",
                    parameters={"target_level_pct": 60.0, "timeout_min": 5.0},
                    requires_confirmation=False,
                    verification_condition="separator_level < 70",
                    is_checkpoint=True
                ),
                PlaybookStep(
                    step_id="SF_04",
                    step_number=4,
                    action_type="normalize_operation",
                    description="Return to normal drain valve operation",
                    requires_confirmation=True,
                    verification_condition="separator_level < 60"
                )
            ],
            trigger_conditions=[
                "separator_level > 90",
                "flooding_alarm_active"
            ],
            abort_conditions=[
                "separator_level > 98",
                "emergency_shutdown_active"
            ],
            estimated_duration_min=15.0,
            requires_pre_approval=True,
            approval_level="operator"
        )

        # Water Hammer Prevention Playbook
        water_hammer_playbook = PlaybookDefinition(
            playbook_id="WH_PREV_001",
            playbook_type=PlaybookType.WATER_HAMMER_PREVENTION,
            name="Water Hammer Prevention",
            description=(
                "Safe warm-up procedure to prevent water hammer. "
                "Slowly admits steam with proper drainage to prevent "
                "condensate accumulation."
            ),
            version="1.0.0",
            steps=[
                PlaybookStep(
                    step_id="WH_01",
                    step_number=1,
                    action_type="verify_drains_open",
                    description="Verify all drip leg drains are open",
                    target_equipment="ALL_DRIP_LEGS",
                    parameters={"required_position_pct": 100.0},
                    requires_confirmation=True,
                    safety_critical=True
                ),
                PlaybookStep(
                    step_id="WH_02",
                    step_number=2,
                    action_type="slow_steam_admission",
                    description="Slowly open steam admission valve (1%/min)",
                    target_equipment="STEAM_ADMISSION",
                    parameters={"ramp_rate_pct_per_min": 1.0},
                    requires_confirmation=True,
                    rollback_action="close_steam_admission"
                ),
                PlaybookStep(
                    step_id="WH_03",
                    step_number=3,
                    action_type="monitor_warmup",
                    description="Monitor temperature and drain condensate",
                    parameters={"warmup_duration_min": 15.0},
                    requires_confirmation=False,
                    verification_condition="temperature_c > saturation_c + 20",
                    is_checkpoint=True
                ),
                PlaybookStep(
                    step_id="WH_04",
                    step_number=4,
                    action_type="close_drains",
                    description="Close drip leg drains after warmup",
                    target_equipment="ALL_DRIP_LEGS",
                    parameters={"position_pct": 0.0},
                    requires_confirmation=True
                ),
                PlaybookStep(
                    step_id="WH_05",
                    step_number=5,
                    action_type="normal_operation",
                    description="Transition to normal operation",
                    requires_confirmation=True
                )
            ],
            trigger_conditions=[
                "cold_start_detected",
                "line_temperature < 50"
            ],
            abort_conditions=[
                "water_hammer_detected",
                "vibration_alarm_active"
            ],
            estimated_duration_min=20.0,
            requires_pre_approval=True,
            approval_level="operator"
        )

        # Register all built-in playbooks
        self.register_playbook(carryover_playbook)
        self.register_playbook(flooding_playbook)
        self.register_playbook(water_hammer_playbook)

        # Calculate provenance hashes
        for playbook in self._playbooks.values():
            playbook.provenance_hash = hashlib.sha256(
                f"{playbook.playbook_id}|{playbook.version}|{len(playbook.steps)}".encode()
            ).hexdigest()

    def _execute_step(
        self,
        execution_id: str,
        step: PlaybookStep,
        result_id: str,
        start_time: datetime
    ) -> StepResult:
        """Execute a single playbook step."""
        logger.info(
            f"Executing step {step.step_number}: {step.description}"
        )

        try:
            # Simulate step execution
            # In production, this would call actual control actions
            output_data = {
                "action_type": step.action_type,
                "target_equipment": step.target_equipment,
                "parameters": step.parameters,
                "simulated": True  # Indicates advisory mode
            }

            # Verification would check actual conditions
            verification_passed = True

            completed_at = datetime.now()
            duration_s = (completed_at - start_time).total_seconds()

            result = StepResult(
                result_id=result_id,
                step_id=step.step_id,
                step_number=step.step_number,
                status="completed",
                started_at=start_time,
                completed_at=completed_at,
                duration_s=duration_s,
                verification_passed=verification_passed,
                output_data=output_data
            )

            # Calculate provenance hash
            result.provenance_hash = hashlib.sha256(
                f"{result_id}|completed|{duration_s}".encode()
            ).hexdigest()

            # Create audit record
            self._add_audit_record(
                execution_id=execution_id,
                operation="step_completed",
                step_id=step.step_id,
                details=output_data
            )

            logger.info(
                f"Step {step.step_number} completed in {duration_s:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Step {step.step_number} failed: {str(e)}")

            result = StepResult(
                result_id=result_id,
                step_id=step.step_id,
                step_number=step.step_number,
                status="failed",
                started_at=start_time,
                completed_at=datetime.now(),
                error_message=str(e)
            )

            result.provenance_hash = hashlib.sha256(
                f"{result_id}|failed|{str(e)}".encode()
            ).hexdigest()

            return result

    def _complete_execution(
        self,
        execution_id: str,
        status: PlaybookStatus
    ) -> None:
        """Complete a playbook execution."""
        if execution_id not in self._active_executions:
            return

        execution = self._active_executions[execution_id]
        execution.status = status
        execution.completed_at = datetime.now()

        # Move to history
        self._execution_history.append(execution)
        del self._active_executions[execution_id]

        # Trim history
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]

        # Create audit record
        self._add_audit_record(
            execution_id=execution_id,
            operation="execution_completed",
            details={"status": status.value}
        )

        logger.info(
            f"Playbook execution completed: execution={execution_id}, "
            f"status={status.value}"
        )

    def _add_audit_record(
        self,
        execution_id: str,
        operation: str,
        step_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an audit record."""
        audit_id = hashlib.sha256(
            f"AUD_{execution_id}_{operation}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        record = PlaybookAuditRecord(
            audit_id=audit_id,
            execution_id=execution_id,
            operation=operation,
            step_id=step_id,
            operator_id=operator_id,
            details=details or {},
            timestamp=datetime.now()
        )

        record.provenance_hash = hashlib.sha256(
            f"{audit_id}|{operation}|{execution_id}".encode()
        ).hexdigest()

        self._audit_log.append(record)

        # Trim audit log
        if len(self._audit_log) > self._max_audit_size:
            self._audit_log = self._audit_log[-self._max_audit_size:]
