"""
BypassWorkflow - Enhanced Bypass Request Workflow Management

This module implements enhanced bypass request workflow management for
Emergency Shutdown Systems per IEC 61511-1 Clause 11.7. Provides
authorization levels, time-limited bypass management, bypass alarm
generation, and automatic bypass expiration.

Key features:
- Bypass request workflow with approval stages
- Authorization levels based on role/SIL
- Time-limited bypass management
- Bypass alarm generation
- Automatic bypass expiration
- Complete audit trail with provenance

Reference: IEC 61511-1 Clause 11.7, ISA TR84.00.09

Example:
    >>> from greenlang.safety.esd.bypass_workflow import BypassWorkflowManager
    >>> manager = BypassWorkflowManager(system_id="ESD-001")
    >>> request = manager.submit_bypass_request(request_data)
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class AuthorizationLevel(str, Enum):
    """Authorization levels for bypass approval."""

    OPERATOR = "operator"  # Can request, not approve
    SUPERVISOR = "supervisor"  # Can approve SIL 1
    ENGINEER = "engineer"  # Can approve SIL 1-2
    SAFETY_ENGINEER = "safety_engineer"  # Can approve SIL 1-3
    PLANT_MANAGER = "plant_manager"  # Can approve all


class WorkflowState(str, Enum):
    """Bypass workflow states."""

    DRAFT = "draft"  # Request being drafted
    SUBMITTED = "submitted"  # Submitted for review
    PENDING_APPROVAL = "pending_approval"  # Awaiting approval
    APPROVED = "approved"  # Approved, not yet active
    ACTIVE = "active"  # Bypass is active
    EXPIRED = "expired"  # Time limit reached
    CANCELLED = "cancelled"  # Cancelled before activation
    REJECTED = "rejected"  # Request rejected


class AlarmPriority(str, Enum):
    """Bypass alarm priority."""

    LOW = "low"  # SIL 1
    MEDIUM = "medium"  # SIL 2
    HIGH = "high"  # SIL 3
    CRITICAL = "critical"  # SIL 4 or special


class BypassRequestData(BaseModel):
    """Bypass request data."""

    request_id: str = Field(
        default_factory=lambda: f"BYPR-{uuid.uuid4().hex[:8].upper()}",
        description="Request identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF to bypass"
    )
    sif_name: str = Field(
        default="",
        description="SIF name"
    )
    sil_level: int = Field(
        default=1,
        ge=0,
        le=4,
        description="SIF SIL level"
    )
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Affected equipment"
    )
    bypass_reason: str = Field(
        ...,
        min_length=20,
        description="Detailed reason for bypass"
    )
    bypass_justification: str = Field(
        default="",
        description="Technical justification"
    )
    requested_duration_hours: float = Field(
        ...,
        gt=0,
        le=72,
        description="Requested duration (max 72h)"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures"
    )
    risk_assessment_ref: Optional[str] = Field(
        None,
        description="Risk assessment reference"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Work permit reference"
    )
    moc_ref: Optional[str] = Field(
        None,
        description="MOC reference if applicable"
    )


class WorkflowRequest(BaseModel):
    """Complete bypass workflow request."""

    workflow_id: str = Field(
        default_factory=lambda: f"WF-{uuid.uuid4().hex[:8].upper()}",
        description="Workflow identifier"
    )
    request_data: BypassRequestData = Field(
        ...,
        description="Request data"
    )
    state: WorkflowState = Field(
        default=WorkflowState.DRAFT,
        description="Current workflow state"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    created_by: str = Field(
        ...,
        description="Requester"
    )
    created_by_level: AuthorizationLevel = Field(
        default=AuthorizationLevel.OPERATOR,
        description="Requester authorization level"
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="Submission timestamp"
    )
    approved_at: Optional[datetime] = Field(
        None,
        description="Approval timestamp"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approver"
    )
    approved_by_level: Optional[AuthorizationLevel] = Field(
        None,
        description="Approver authorization level"
    )
    activated_at: Optional[datetime] = Field(
        None,
        description="Activation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration timestamp"
    )
    deactivated_at: Optional[datetime] = Field(
        None,
        description="Deactivation timestamp"
    )
    deactivated_by: Optional[str] = Field(
        None,
        description="Deactivator"
    )
    final_duration_hours: Optional[float] = Field(
        None,
        description="Approved duration"
    )
    extension_count: int = Field(
        default=0,
        description="Number of extensions"
    )
    rejection_reason: Optional[str] = Field(
        None,
        description="Rejection reason"
    )
    alarms_generated: List[str] = Field(
        default_factory=list,
        description="Alarm IDs generated"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete audit trail"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BypassAlarm(BaseModel):
    """Bypass alarm record."""

    alarm_id: str = Field(
        default_factory=lambda: f"BYPALM-{uuid.uuid4().hex[:8].upper()}",
        description="Alarm identifier"
    )
    workflow_id: str = Field(
        ...,
        description="Associated workflow"
    )
    sif_id: str = Field(
        ...,
        description="Bypassed SIF"
    )
    alarm_type: str = Field(
        ...,
        description="Type of alarm"
    )
    priority: AlarmPriority = Field(
        ...,
        description="Alarm priority"
    )
    message: str = Field(
        ...,
        description="Alarm message"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )
    acknowledged: bool = Field(
        default=False,
        description="Is alarm acknowledged"
    )
    acknowledged_by: Optional[str] = Field(
        None,
        description="Acknowledger"
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Acknowledgment time"
    )
    cleared: bool = Field(
        default=False,
        description="Is alarm cleared"
    )
    cleared_at: Optional[datetime] = Field(
        None,
        description="Clear timestamp"
    )


class BypassWorkflowManager:
    """
    Bypass Workflow Manager for ESD/SIS Systems.

    Manages the complete bypass request workflow including submission,
    approval, activation, and expiration per IEC 61511-1 Clause 11.7.

    Key features:
    - Multi-stage approval workflow
    - Authorization levels based on SIL
    - Time-limited bypass management
    - Automatic alarm generation
    - Automatic expiration
    - Complete audit trail

    The manager follows IEC 61511 principles:
    - All bypasses require authorization
    - Time limits enforced
    - Alarms generated for active bypasses
    - Complete traceability

    Attributes:
        system_id: ESD system identifier
        workflows: Active workflow requests
        alarms: Generated alarms

    Example:
        >>> manager = BypassWorkflowManager(system_id="ESD-001")
        >>> request = manager.submit_bypass_request(request_data, "operator1")
        >>> manager.approve_bypass(request.workflow_id, "engineer1", ...)
    """

    # Authorization requirements by SIL level
    REQUIRED_AUTHORIZATION: Dict[int, AuthorizationLevel] = {
        0: AuthorizationLevel.SUPERVISOR,
        1: AuthorizationLevel.SUPERVISOR,
        2: AuthorizationLevel.ENGINEER,
        3: AuthorizationLevel.SAFETY_ENGINEER,
        4: AuthorizationLevel.PLANT_MANAGER,
    }

    # Maximum duration by SIL level (hours)
    MAX_DURATION_BY_SIL: Dict[int, float] = {
        0: 72.0,
        1: 24.0,
        2: 12.0,
        3: 8.0,
        4: 4.0,
    }

    # Alarm priority by SIL level
    ALARM_PRIORITY_BY_SIL: Dict[int, AlarmPriority] = {
        0: AlarmPriority.LOW,
        1: AlarmPriority.LOW,
        2: AlarmPriority.MEDIUM,
        3: AlarmPriority.HIGH,
        4: AlarmPriority.CRITICAL,
    }

    def __init__(
        self,
        system_id: str,
        alarm_callback: Optional[Callable[[BypassAlarm], None]] = None,
        max_extensions: int = 2
    ):
        """
        Initialize BypassWorkflowManager.

        Args:
            system_id: ESD system identifier
            alarm_callback: Callback for alarm generation
            max_extensions: Maximum extensions allowed
        """
        self.system_id = system_id
        self.alarm_callback = alarm_callback or self._default_alarm_callback
        self.max_extensions = max_extensions

        self.workflows: Dict[str, WorkflowRequest] = {}
        self.alarms: Dict[str, BypassAlarm] = {}
        self.user_authorizations: Dict[str, AuthorizationLevel] = {}

        logger.info(f"BypassWorkflowManager initialized: {system_id}")

    def register_user(
        self,
        user_id: str,
        authorization_level: AuthorizationLevel
    ) -> None:
        """
        Register a user with authorization level.

        Args:
            user_id: User identifier
            authorization_level: Authorization level
        """
        self.user_authorizations[user_id] = authorization_level
        logger.info(f"User {user_id} registered with level {authorization_level.value}")

    def submit_bypass_request(
        self,
        request_data: BypassRequestData,
        requester: str
    ) -> WorkflowRequest:
        """
        Submit a bypass request.

        Args:
            request_data: Bypass request data
            requester: Person submitting request

        Returns:
            WorkflowRequest in submitted state
        """
        # Validate request data
        max_duration = self.MAX_DURATION_BY_SIL.get(
            request_data.sil_level, 24.0
        )

        if request_data.requested_duration_hours > max_duration:
            logger.warning(
                f"Requested duration {request_data.requested_duration_hours}h "
                f"exceeds max {max_duration}h for SIL {request_data.sil_level}"
            )
            request_data.requested_duration_hours = max_duration

        # Get requester authorization
        requester_level = self.user_authorizations.get(
            requester, AuthorizationLevel.OPERATOR
        )

        # Create workflow
        workflow = WorkflowRequest(
            request_data=request_data,
            state=WorkflowState.SUBMITTED,
            created_by=requester,
            created_by_level=requester_level,
            submitted_at=datetime.utcnow(),
        )

        # Add audit entry
        workflow.audit_trail.append({
            "action": "submitted",
            "timestamp": datetime.utcnow().isoformat(),
            "user": requester,
            "user_level": requester_level.value,
            "details": {
                "sif_id": request_data.sif_id,
                "sil_level": request_data.sil_level,
                "requested_duration": request_data.requested_duration_hours,
                "reason": request_data.bypass_reason,
            }
        })

        workflow.provenance_hash = self._calculate_provenance(workflow)
        self.workflows[workflow.workflow_id] = workflow

        logger.info(
            f"Bypass request submitted: {workflow.workflow_id} "
            f"for {request_data.sif_id} by {requester}"
        )

        return workflow

    def approve_bypass(
        self,
        workflow_id: str,
        approver: str,
        approved_duration_hours: Optional[float] = None,
        conditions: Optional[List[str]] = None
    ) -> WorkflowRequest:
        """
        Approve a bypass request.

        Args:
            workflow_id: Workflow to approve
            approver: Person approving
            approved_duration_hours: Approved duration (optional)
            conditions: Additional conditions

        Returns:
            Updated WorkflowRequest

        Raises:
            ValueError: If approval is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]

        if workflow.state != WorkflowState.SUBMITTED:
            raise ValueError(
                f"Cannot approve workflow in state: {workflow.state.value}"
            )

        # Check approver authorization
        approver_level = self.user_authorizations.get(
            approver, AuthorizationLevel.OPERATOR
        )

        required_level = self.REQUIRED_AUTHORIZATION.get(
            workflow.request_data.sil_level,
            AuthorizationLevel.ENGINEER
        )

        if not self._has_authority(approver_level, required_level):
            raise ValueError(
                f"Insufficient authorization: {approver_level.value} "
                f"cannot approve SIL {workflow.request_data.sil_level}"
            )

        # Validate duration
        final_duration = approved_duration_hours or workflow.request_data.requested_duration_hours
        max_duration = self.MAX_DURATION_BY_SIL.get(
            workflow.request_data.sil_level, 24.0
        )

        if final_duration > max_duration:
            final_duration = max_duration

        # Update workflow
        workflow.state = WorkflowState.APPROVED
        workflow.approved_at = datetime.utcnow()
        workflow.approved_by = approver
        workflow.approved_by_level = approver_level
        workflow.final_duration_hours = final_duration

        # Add conditions to compensating measures
        if conditions:
            workflow.request_data.compensating_measures.extend(conditions)

        workflow.audit_trail.append({
            "action": "approved",
            "timestamp": datetime.utcnow().isoformat(),
            "user": approver,
            "user_level": approver_level.value,
            "details": {
                "approved_duration": final_duration,
                "conditions": conditions,
            }
        })

        workflow.provenance_hash = self._calculate_provenance(workflow)

        logger.info(
            f"Bypass approved: {workflow_id} by {approver} "
            f"for {final_duration}h"
        )

        return workflow

    def reject_bypass(
        self,
        workflow_id: str,
        rejector: str,
        reason: str
    ) -> WorkflowRequest:
        """
        Reject a bypass request.

        Args:
            workflow_id: Workflow to reject
            rejector: Person rejecting
            reason: Rejection reason

        Returns:
            Updated WorkflowRequest
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]

        if workflow.state not in [WorkflowState.SUBMITTED, WorkflowState.PENDING_APPROVAL]:
            raise ValueError(
                f"Cannot reject workflow in state: {workflow.state.value}"
            )

        workflow.state = WorkflowState.REJECTED
        workflow.rejection_reason = reason

        workflow.audit_trail.append({
            "action": "rejected",
            "timestamp": datetime.utcnow().isoformat(),
            "user": rejector,
            "details": {
                "reason": reason,
            }
        })

        workflow.provenance_hash = self._calculate_provenance(workflow)

        logger.info(f"Bypass rejected: {workflow_id} - {reason}")

        return workflow

    def activate_bypass(
        self,
        workflow_id: str,
        activated_by: str
    ) -> WorkflowRequest:
        """
        Activate an approved bypass.

        Args:
            workflow_id: Workflow to activate
            activated_by: Person activating

        Returns:
            Updated WorkflowRequest with active bypass
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]

        if workflow.state != WorkflowState.APPROVED:
            raise ValueError(
                f"Cannot activate workflow in state: {workflow.state.value}"
            )

        now = datetime.utcnow()
        workflow.state = WorkflowState.ACTIVE
        workflow.activated_at = now
        workflow.expires_at = now + timedelta(hours=workflow.final_duration_hours)

        workflow.audit_trail.append({
            "action": "activated",
            "timestamp": now.isoformat(),
            "user": activated_by,
            "details": {
                "expires_at": workflow.expires_at.isoformat(),
            }
        })

        # Generate bypass active alarm
        alarm = self._generate_alarm(
            workflow,
            "BYPASS_ACTIVE",
            f"BYPASS ACTIVE: {workflow.request_data.sif_id} "
            f"(expires: {workflow.expires_at.strftime('%Y-%m-%d %H:%M')})"
        )
        workflow.alarms_generated.append(alarm.alarm_id)

        workflow.provenance_hash = self._calculate_provenance(workflow)

        logger.warning(
            f"BYPASS ACTIVATED: {workflow.request_data.sif_id} "
            f"(workflow: {workflow_id})"
        )

        return workflow

    def deactivate_bypass(
        self,
        workflow_id: str,
        deactivated_by: str,
        reason: str = "Normal deactivation"
    ) -> WorkflowRequest:
        """
        Deactivate an active bypass.

        Args:
            workflow_id: Workflow to deactivate
            deactivated_by: Person deactivating
            reason: Deactivation reason

        Returns:
            Updated WorkflowRequest
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]

        if workflow.state != WorkflowState.ACTIVE:
            raise ValueError(
                f"Cannot deactivate workflow in state: {workflow.state.value}"
            )

        now = datetime.utcnow()
        workflow.state = WorkflowState.CANCELLED
        workflow.deactivated_at = now
        workflow.deactivated_by = deactivated_by

        workflow.audit_trail.append({
            "action": "deactivated",
            "timestamp": now.isoformat(),
            "user": deactivated_by,
            "details": {
                "reason": reason,
                "duration_used_hours": (
                    (now - workflow.activated_at).total_seconds() / 3600
                    if workflow.activated_at else 0
                ),
            }
        })

        # Clear bypass alarms
        for alarm_id in workflow.alarms_generated:
            if alarm_id in self.alarms:
                alarm = self.alarms[alarm_id]
                alarm.cleared = True
                alarm.cleared_at = now

        workflow.provenance_hash = self._calculate_provenance(workflow)

        logger.info(
            f"Bypass deactivated: {workflow.request_data.sif_id} "
            f"by {deactivated_by}"
        )

        return workflow

    def extend_bypass(
        self,
        workflow_id: str,
        extended_by: str,
        extension_hours: float,
        justification: str
    ) -> WorkflowRequest:
        """
        Extend an active bypass.

        Args:
            workflow_id: Workflow to extend
            extended_by: Person extending
            extension_hours: Additional hours
            justification: Extension justification

        Returns:
            Updated WorkflowRequest

        Raises:
            ValueError: If extension is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]

        if workflow.state != WorkflowState.ACTIVE:
            raise ValueError(
                f"Cannot extend workflow in state: {workflow.state.value}"
            )

        if workflow.extension_count >= self.max_extensions:
            raise ValueError(
                f"Maximum extensions ({self.max_extensions}) reached"
            )

        # Check authorization
        extender_level = self.user_authorizations.get(
            extended_by, AuthorizationLevel.OPERATOR
        )

        required_level = self.REQUIRED_AUTHORIZATION.get(
            workflow.request_data.sil_level,
            AuthorizationLevel.ENGINEER
        )

        if not self._has_authority(extender_level, required_level):
            raise ValueError(
                f"Insufficient authorization for extension"
            )

        # Validate extension against limits
        max_duration = self.MAX_DURATION_BY_SIL.get(
            workflow.request_data.sil_level, 24.0
        )

        current_total = workflow.final_duration_hours or 0
        new_total = current_total + extension_hours

        if new_total > max_duration * 2:  # Allow up to 2x for extensions
            raise ValueError(
                f"Extended duration {new_total}h exceeds limit {max_duration * 2}h"
            )

        workflow.final_duration_hours = new_total
        workflow.expires_at = workflow.activated_at + timedelta(hours=new_total)
        workflow.extension_count += 1

        workflow.audit_trail.append({
            "action": "extended",
            "timestamp": datetime.utcnow().isoformat(),
            "user": extended_by,
            "user_level": extender_level.value,
            "details": {
                "extension_hours": extension_hours,
                "new_total_hours": new_total,
                "new_expires_at": workflow.expires_at.isoformat(),
                "extension_count": workflow.extension_count,
                "justification": justification,
            }
        })

        # Generate extension alarm
        alarm = self._generate_alarm(
            workflow,
            "BYPASS_EXTENDED",
            f"BYPASS EXTENDED: {workflow.request_data.sif_id} "
            f"by {extension_hours}h (expires: {workflow.expires_at.strftime('%Y-%m-%d %H:%M')})"
        )
        workflow.alarms_generated.append(alarm.alarm_id)

        workflow.provenance_hash = self._calculate_provenance(workflow)

        logger.warning(
            f"Bypass extended: {workflow_id} by {extension_hours}h "
            f"(total: {new_total}h)"
        )

        return workflow

    def check_expired_bypasses(self) -> List[WorkflowRequest]:
        """
        Check and expire bypasses that have exceeded time limit.

        Returns:
            List of newly expired workflows
        """
        now = datetime.utcnow()
        expired = []

        for workflow in self.workflows.values():
            if workflow.state == WorkflowState.ACTIVE:
                if workflow.expires_at and now > workflow.expires_at:
                    workflow.state = WorkflowState.EXPIRED
                    workflow.deactivated_at = now

                    workflow.audit_trail.append({
                        "action": "expired",
                        "timestamp": now.isoformat(),
                        "details": {
                            "scheduled_expiry": workflow.expires_at.isoformat(),
                        }
                    })

                    # Generate expiration alarm
                    alarm = self._generate_alarm(
                        workflow,
                        "BYPASS_EXPIRED",
                        f"BYPASS EXPIRED: {workflow.request_data.sif_id} "
                        f"- SIF protection restored"
                    )
                    workflow.alarms_generated.append(alarm.alarm_id)

                    workflow.provenance_hash = self._calculate_provenance(workflow)
                    expired.append(workflow)

                    logger.warning(
                        f"BYPASS EXPIRED: {workflow.request_data.sif_id} "
                        f"(workflow: {workflow.workflow_id})"
                    )

        return expired

    def check_expiring_soon(
        self,
        warning_minutes: int = 60
    ) -> List[WorkflowRequest]:
        """
        Check for bypasses expiring soon.

        Args:
            warning_minutes: Warning threshold in minutes

        Returns:
            List of workflows expiring soon
        """
        now = datetime.utcnow()
        warning_threshold = now + timedelta(minutes=warning_minutes)

        expiring = []
        for workflow in self.workflows.values():
            if workflow.state == WorkflowState.ACTIVE:
                if (workflow.expires_at and
                    now < workflow.expires_at <= warning_threshold):
                    expiring.append(workflow)

                    # Check if warning already generated
                    existing_warning = any(
                        a.alarm_type == "BYPASS_EXPIRING_SOON"
                        for a in [self.alarms.get(aid) for aid in workflow.alarms_generated]
                        if a
                    )

                    if not existing_warning:
                        alarm = self._generate_alarm(
                            workflow,
                            "BYPASS_EXPIRING_SOON",
                            f"BYPASS EXPIRING: {workflow.request_data.sif_id} "
                            f"in {int((workflow.expires_at - now).total_seconds() / 60)} minutes"
                        )
                        workflow.alarms_generated.append(alarm.alarm_id)

        return expiring

    def acknowledge_alarm(
        self,
        alarm_id: str,
        acknowledged_by: str
    ) -> BypassAlarm:
        """
        Acknowledge a bypass alarm.

        Args:
            alarm_id: Alarm to acknowledge
            acknowledged_by: Person acknowledging

        Returns:
            Updated BypassAlarm
        """
        if alarm_id not in self.alarms:
            raise ValueError(f"Alarm not found: {alarm_id}")

        alarm = self.alarms[alarm_id]
        alarm.acknowledged = True
        alarm.acknowledged_by = acknowledged_by
        alarm.acknowledged_at = datetime.utcnow()

        logger.info(f"Alarm {alarm_id} acknowledged by {acknowledged_by}")

        return alarm

    def get_active_bypasses(self) -> List[WorkflowRequest]:
        """Get all active bypass workflows."""
        return [
            w for w in self.workflows.values()
            if w.state == WorkflowState.ACTIVE
        ]

    def get_pending_approvals(self) -> List[WorkflowRequest]:
        """Get workflows pending approval."""
        return [
            w for w in self.workflows.values()
            if w.state in [WorkflowState.SUBMITTED, WorkflowState.PENDING_APPROVAL]
        ]

    def get_workflow_status(
        self,
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Status dictionary
        """
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]

        remaining_hours = None
        if workflow.state == WorkflowState.ACTIVE and workflow.expires_at:
            remaining = (workflow.expires_at - datetime.utcnow()).total_seconds() / 3600
            remaining_hours = max(0, remaining)

        return {
            "workflow_id": workflow.workflow_id,
            "sif_id": workflow.request_data.sif_id,
            "state": workflow.state.value,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at.isoformat(),
            "approved_by": workflow.approved_by,
            "approved_at": workflow.approved_at.isoformat() if workflow.approved_at else None,
            "activated_at": workflow.activated_at.isoformat() if workflow.activated_at else None,
            "expires_at": workflow.expires_at.isoformat() if workflow.expires_at else None,
            "remaining_hours": remaining_hours,
            "extension_count": workflow.extension_count,
            "active_alarms": len([
                aid for aid in workflow.alarms_generated
                if aid in self.alarms and not self.alarms[aid].acknowledged
            ]),
        }

    def get_bypass_summary(self) -> Dict[str, Any]:
        """
        Get overall bypass summary.

        Returns:
            Summary dictionary
        """
        now = datetime.utcnow()

        active = [w for w in self.workflows.values() if w.state == WorkflowState.ACTIVE]
        pending = [w for w in self.workflows.values()
                   if w.state in [WorkflowState.SUBMITTED, WorkflowState.PENDING_APPROVAL]]

        expiring_1h = [
            w for w in active
            if w.expires_at and (w.expires_at - now).total_seconds() < 3600
        ]

        unacknowledged_alarms = [
            a for a in self.alarms.values()
            if not a.acknowledged and not a.cleared
        ]

        return {
            "report_timestamp": now.isoformat(),
            "system_id": self.system_id,
            "total_workflows": len(self.workflows),
            "active_bypasses": len(active),
            "pending_approvals": len(pending),
            "expiring_within_1h": len(expiring_1h),
            "unacknowledged_alarms": len(unacknowledged_alarms),
            "active_by_sil": {
                sil: len([w for w in active if w.request_data.sil_level == sil])
                for sil in range(0, 5)
            },
            "provenance_hash": hashlib.sha256(
                f"{now.isoformat()}|{len(active)}|{len(pending)}".encode()
            ).hexdigest()
        }

    def _has_authority(
        self,
        user_level: AuthorizationLevel,
        required_level: AuthorizationLevel
    ) -> bool:
        """Check if user has required authority."""
        level_hierarchy = {
            AuthorizationLevel.OPERATOR: 0,
            AuthorizationLevel.SUPERVISOR: 1,
            AuthorizationLevel.ENGINEER: 2,
            AuthorizationLevel.SAFETY_ENGINEER: 3,
            AuthorizationLevel.PLANT_MANAGER: 4,
        }

        return level_hierarchy[user_level] >= level_hierarchy[required_level]

    def _generate_alarm(
        self,
        workflow: WorkflowRequest,
        alarm_type: str,
        message: str
    ) -> BypassAlarm:
        """Generate a bypass alarm."""
        priority = self.ALARM_PRIORITY_BY_SIL.get(
            workflow.request_data.sil_level,
            AlarmPriority.MEDIUM
        )

        alarm = BypassAlarm(
            workflow_id=workflow.workflow_id,
            sif_id=workflow.request_data.sif_id,
            alarm_type=alarm_type,
            priority=priority,
            message=message,
        )

        self.alarms[alarm.alarm_id] = alarm
        self.alarm_callback(alarm)

        return alarm

    def _default_alarm_callback(self, alarm: BypassAlarm) -> None:
        """Default alarm callback."""
        logger.warning(
            f"ALARM [{alarm.priority.value.upper()}]: {alarm.message}"
        )

    def _calculate_provenance(self, workflow: WorkflowRequest) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{workflow.workflow_id}|"
            f"{workflow.request_data.sif_id}|"
            f"{workflow.state.value}|"
            f"{len(workflow.audit_trail)}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
