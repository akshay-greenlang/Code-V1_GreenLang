r"""
ActionTracking - Action Item Tracking for Risk Management

This module implements comprehensive action item tracking for HAZOP, FMEA,
and risk management workflows per IEC 61511 and IEC 61882 standards.

Key Features:
- ActionItem model with priority, status, due dates, and ownership
- ActionTracker class for full CRUD operations
- Escalation rules for overdue items with notification triggers
- Assignment and notification system integration
- Integration with risk matrix for priority determination
- Complete audit trail with provenance tracking

Reference:
- IEC 61511-1:2016 - Functional Safety
- IEC 61882:2016 - Hazard and Operability Studies (HAZOP)
- IEC 60812:2018 - Failure Mode and Effects Analysis (FMEA)

Example:
    >>> from greenlang.safety.action_tracking import ActionTracker, ActionItem
    >>> tracker = ActionTracker()
    >>> action = ActionItem(
    ...     title="Install high-high level alarm",
    ...     description="Add independent high-high level alarm to tank T-101",
    ...     priority=ActionPriority.HIGH,
    ...     assignee="John Smith",
    ...     due_date=datetime.now() + timedelta(days=30)
    ... )
    >>> tracker.create_action(action)
"""

from typing import Dict, List, Optional, Any, Callable, ClassVar
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ActionPriority(str, Enum):
    """Action item priority levels aligned with risk matrix."""

    CRITICAL = "critical"  # Immediate action required (risk level CRITICAL)
    HIGH = "high"  # Action required within 30 days (risk level HIGH)
    MEDIUM = "medium"  # Action required within 90 days (risk level MEDIUM)
    LOW = "low"  # Action required within 365 days (risk level LOW)
    INFORMATIONAL = "informational"  # For tracking only, no deadline


class ActionStatus(str, Enum):
    """Action item lifecycle status."""

    DRAFT = "draft"  # Action being drafted
    OPEN = "open"  # Action assigned and active
    IN_PROGRESS = "in_progress"  # Work underway
    ON_HOLD = "on_hold"  # Temporarily paused
    PENDING_VERIFICATION = "pending_verification"  # Awaiting verification
    COMPLETED = "completed"  # Action completed
    VERIFIED = "verified"  # Completion verified
    CANCELLED = "cancelled"  # Action cancelled
    OVERDUE = "overdue"  # Past due date


class ActionCategory(str, Enum):
    """Action item categories."""

    ENGINEERING = "engineering"  # Engineering design change
    PROCEDURAL = "procedural"  # Procedure update
    TRAINING = "training"  # Training requirement
    MAINTENANCE = "maintenance"  # Maintenance activity
    DOCUMENTATION = "documentation"  # Documentation update
    INVESTIGATION = "investigation"  # Further investigation needed
    PROCUREMENT = "procurement"  # Equipment procurement
    SIS = "sis"  # Safety Instrumented System related
    COMPLIANCE = "compliance"  # Regulatory compliance


class ActionSource(str, Enum):
    """Source of action item."""

    HAZOP = "hazop"  # From HAZOP study
    FMEA = "fmea"  # From FMEA analysis
    LOPA = "lopa"  # From LOPA analysis
    INCIDENT = "incident"  # From incident investigation
    AUDIT = "audit"  # From safety audit
    INSPECTION = "inspection"  # From inspection
    MOC = "moc"  # From Management of Change
    PHA = "pha"  # From Process Hazard Analysis
    REVIEW = "review"  # From periodic review
    OTHER = "other"  # Other source


class EscalationLevel(str, Enum):
    """Escalation levels for overdue actions."""

    NONE = "none"  # No escalation
    LEVEL_1 = "level_1"  # First escalation (supervisor)
    LEVEL_2 = "level_2"  # Second escalation (manager)
    LEVEL_3 = "level_3"  # Third escalation (director)
    EXECUTIVE = "executive"  # Executive escalation


# =============================================================================
# DATA MODELS
# =============================================================================

class ActionComment(BaseModel):
    """Comment or update on an action item."""

    comment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique comment identifier"
    )
    author: str = Field(
        ...,
        description="Comment author"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Comment content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Comment timestamp"
    )
    is_status_change: bool = Field(
        default=False,
        description="Whether this comment represents a status change"
    )


class ActionAttachment(BaseModel):
    """Attachment linked to an action item."""

    attachment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attachment identifier"
    )
    filename: str = Field(
        ...,
        description="Attachment filename"
    )
    file_type: str = Field(
        default="",
        description="MIME type or file extension"
    )
    file_path: str = Field(
        default="",
        description="Path to file storage"
    )
    uploaded_by: str = Field(
        default="",
        description="Person who uploaded"
    )
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Upload timestamp"
    )
    description: str = Field(
        default="",
        description="Attachment description"
    )


class EscalationRule(BaseModel):
    """Escalation rule configuration."""

    level: EscalationLevel = Field(
        ...,
        description="Escalation level"
    )
    days_overdue: int = Field(
        ...,
        ge=0,
        description="Days overdue to trigger this level"
    )
    notify_roles: List[str] = Field(
        default_factory=list,
        description="Roles to notify at this level"
    )
    notify_specific: List[str] = Field(
        default_factory=list,
        description="Specific people to notify"
    )


class ActionItem(BaseModel):
    """Comprehensive action item model."""

    action_id: str = Field(
        default_factory=lambda: f"ACT-{uuid.uuid4().hex[:8].upper()}",
        description="Unique action identifier"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Action title"
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description"
    )

    # Classification
    priority: ActionPriority = Field(
        default=ActionPriority.MEDIUM,
        description="Action priority"
    )
    status: ActionStatus = Field(
        default=ActionStatus.DRAFT,
        description="Current status"
    )
    category: ActionCategory = Field(
        default=ActionCategory.ENGINEERING,
        description="Action category"
    )
    source: ActionSource = Field(
        default=ActionSource.OTHER,
        description="Source of action item"
    )

    # Source references
    source_reference_id: str = Field(
        default="",
        description="Reference ID from source (e.g., HAZOP deviation ID)"
    )
    source_study_id: str = Field(
        default="",
        description="Study ID from source system"
    )
    linked_risk_id: str = Field(
        default="",
        description="Linked risk register entry ID"
    )
    linked_hazard_ids: List[str] = Field(
        default_factory=list,
        description="Linked hazard IDs"
    )

    # Assignment
    assignee: str = Field(
        default="",
        description="Person assigned to action"
    )
    assignee_email: str = Field(
        default="",
        description="Assignee email address"
    )
    responsible_department: str = Field(
        default="",
        description="Responsible department"
    )
    created_by: str = Field(
        default="",
        description="Person who created action"
    )
    verified_by: Optional[str] = Field(
        None,
        description="Person who verified completion"
    )

    # Dates
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation date"
    )
    due_date: Optional[datetime] = Field(
        None,
        description="Due date"
    )
    original_due_date: Optional[datetime] = Field(
        None,
        description="Original due date (before extensions)"
    )
    started_date: Optional[datetime] = Field(
        None,
        description="Date work started"
    )
    completed_date: Optional[datetime] = Field(
        None,
        description="Date action was completed"
    )
    verified_date: Optional[datetime] = Field(
        None,
        description="Date completion was verified"
    )
    extension_count: int = Field(
        default=0,
        ge=0,
        description="Number of due date extensions"
    )

    # Escalation
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.NONE,
        description="Current escalation level"
    )
    last_escalation_date: Optional[datetime] = Field(
        None,
        description="Date of last escalation"
    )

    # Risk context
    risk_severity: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Associated risk severity (1-5)"
    )
    risk_likelihood: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Associated risk likelihood (1-5)"
    )
    risk_reduction_target: Optional[str] = Field(
        None,
        description="Target risk reduction"
    )

    # Content
    acceptance_criteria: str = Field(
        default="",
        description="Criteria for completion acceptance"
    )
    completion_evidence: str = Field(
        default="",
        description="Evidence of completion"
    )
    verification_notes: str = Field(
        default="",
        description="Verification notes"
    )

    # Related items
    comments: List[ActionComment] = Field(
        default_factory=list,
        description="Comments and updates"
    )
    attachments: List[ActionAttachment] = Field(
        default_factory=list,
        description="Attached files"
    )
    related_actions: List[str] = Field(
        default_factory=list,
        description="Related action IDs"
    )

    # Metadata
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for filtering"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}

    @field_validator('due_date')
    @classmethod
    def validate_due_date(cls, v, info):
        """Validate due date is in the future for new actions."""
        # Only validate if this is a new action (no completed_date)
        if v and info.data.get('status') == ActionStatus.DRAFT:
            if v < datetime.utcnow():
                logger.warning("Due date is in the past")
        return v


class NotificationEvent(BaseModel):
    """Notification event for action tracking."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    action_id: str = Field(
        ...,
        description="Related action ID"
    )
    event_type: str = Field(
        ...,
        description="Type of notification event"
    )
    recipients: List[str] = Field(
        default_factory=list,
        description="List of recipient emails or IDs"
    )
    message: str = Field(
        default="",
        description="Notification message"
    )
    sent_at: Optional[datetime] = Field(
        None,
        description="When notification was sent"
    )
    sent_successfully: bool = Field(
        default=False,
        description="Whether notification was sent successfully"
    )


class ActionTrackerConfig(BaseModel):
    """Configuration for action tracker."""

    # Priority-based due date defaults (days from creation)
    default_due_days: Dict[str, int] = Field(
        default_factory=lambda: {
            ActionPriority.CRITICAL.value: 7,
            ActionPriority.HIGH.value: 30,
            ActionPriority.MEDIUM.value: 90,
            ActionPriority.LOW.value: 365,
            ActionPriority.INFORMATIONAL.value: 0,  # No deadline
        },
        description="Default due days by priority"
    )

    # Escalation configuration
    escalation_rules: List[EscalationRule] = Field(
        default_factory=lambda: [
            EscalationRule(
                level=EscalationLevel.LEVEL_1,
                days_overdue=7,
                notify_roles=["supervisor"]
            ),
            EscalationRule(
                level=EscalationLevel.LEVEL_2,
                days_overdue=14,
                notify_roles=["manager"]
            ),
            EscalationRule(
                level=EscalationLevel.LEVEL_3,
                days_overdue=30,
                notify_roles=["director"]
            ),
            EscalationRule(
                level=EscalationLevel.EXECUTIVE,
                days_overdue=60,
                notify_roles=["executive"]
            ),
        ],
        description="Escalation rules"
    )

    # Notification settings
    reminder_days_before: List[int] = Field(
        default_factory=lambda: [7, 3, 1],
        description="Days before due to send reminders"
    )

    # Extension limits
    max_extensions: int = Field(
        default=3,
        description="Maximum allowed due date extensions"
    )
    extension_days_limit: int = Field(
        default=30,
        description="Maximum days per extension"
    )

    # Auto-status updates
    auto_mark_overdue: bool = Field(
        default=True,
        description="Automatically mark actions as overdue"
    )
    require_verification: bool = Field(
        default=True,
        description="Require verification for completion"
    )


# =============================================================================
# ACTION TRACKER
# =============================================================================

class ActionTracker:
    """
    Action Item Tracker for risk management.

    Implements comprehensive action tracking per IEC 61511/61882:
    - Full CRUD operations for action items
    - Priority-based due date management
    - Escalation rules for overdue items
    - Notification system integration
    - Integration with risk matrix
    - Complete audit trail

    Attributes:
        actions: Dict of action_id to ActionItem
        config: ActionTrackerConfig
        notification_handler: Optional callback for notifications

    Example:
        >>> tracker = ActionTracker()
        >>> action = ActionItem(
        ...     title="Install pressure indicator",
        ...     priority=ActionPriority.HIGH,
        ...     assignee="Jane Doe"
        ... )
        >>> created = tracker.create_action(action)
        >>> tracker.update_status(created.action_id, ActionStatus.IN_PROGRESS)
    """

    # Risk level to priority mapping
    RISK_PRIORITY_MAP: ClassVar[Dict[str, ActionPriority]] = {
        "critical": ActionPriority.CRITICAL,
        "high": ActionPriority.HIGH,
        "medium": ActionPriority.MEDIUM,
        "low": ActionPriority.LOW,
    }

    def __init__(
        self,
        config: Optional[ActionTrackerConfig] = None,
        notification_handler: Optional[Callable[[NotificationEvent], bool]] = None
    ):
        """
        Initialize ActionTracker.

        Args:
            config: Optional tracker configuration
            notification_handler: Optional callback for sending notifications
        """
        self.actions: Dict[str, ActionItem] = {}
        self.config = config or ActionTrackerConfig()
        self.notification_handler = notification_handler
        self.notifications: List[NotificationEvent] = []
        self.audit_trail: List[Dict[str, Any]] = []

        logger.info("ActionTracker initialized")

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def create_action(
        self,
        action: ActionItem,
        auto_assign_due_date: bool = True
    ) -> ActionItem:
        """
        Create a new action item.

        Args:
            action: ActionItem to create
            auto_assign_due_date: Auto-assign due date based on priority

        Returns:
            Created ActionItem with calculated fields

        Raises:
            ValueError: If action already exists
        """
        if action.action_id in self.actions:
            raise ValueError(f"Action already exists: {action.action_id}")

        # Auto-assign due date if not set
        if auto_assign_due_date and not action.due_date:
            due_days = self.config.default_due_days.get(
                action.priority.value, 90
            )
            if due_days > 0:
                action.due_date = datetime.utcnow() + timedelta(days=due_days)
                action.original_due_date = action.due_date

        # Set status to OPEN if assignee provided
        if action.assignee and action.status == ActionStatus.DRAFT:
            action.status = ActionStatus.OPEN

        # Calculate provenance hash
        action.provenance_hash = self._calculate_provenance(action)

        # Store action
        self.actions[action.action_id] = action

        # Log audit trail
        self._log_audit("ACTION_CREATED", action.action_id, {
            "title": action.title,
            "priority": action.priority.value,
            "assignee": action.assignee
        })

        # Send notification if assignee set
        if action.assignee:
            self._send_notification(
                action_id=action.action_id,
                event_type="ASSIGNED",
                recipients=[action.assignee_email] if action.assignee_email else [],
                message=f"You have been assigned action: {action.title}"
            )

        logger.info(
            f"Action created: {action.action_id} - {action.title} "
            f"(Priority: {action.priority.value})"
        )
        return action

    def get_action(self, action_id: str) -> Optional[ActionItem]:
        """Get action by ID."""
        return self.actions.get(action_id)

    def update_action(
        self,
        action_id: str,
        updates: Dict[str, Any]
    ) -> ActionItem:
        """
        Update action item fields.

        Args:
            action_id: Action identifier
            updates: Dictionary of field updates

        Returns:
            Updated ActionItem

        Raises:
            ValueError: If action not found
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]
        old_status = action.status

        # Apply updates
        for key, value in updates.items():
            if hasattr(action, key):
                setattr(action, key, value)

        action.updated_at = datetime.utcnow()
        action.provenance_hash = self._calculate_provenance(action)

        # Log status changes
        if action.status != old_status:
            self._add_status_change_comment(
                action, old_status.value, action.status.value
            )

        self._log_audit("ACTION_UPDATED", action_id, {
            "updates": list(updates.keys())
        })

        logger.info(f"Action updated: {action_id}")
        return action

    def update_status(
        self,
        action_id: str,
        new_status: ActionStatus,
        comment: str = "",
        updated_by: str = ""
    ) -> ActionItem:
        """
        Update action status with optional comment.

        Args:
            action_id: Action identifier
            new_status: New status
            comment: Optional status change comment
            updated_by: Person making the update

        Returns:
            Updated ActionItem

        Raises:
            ValueError: If action not found or invalid transition
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]
        old_status = action.status

        # Validate status transition
        if not self._is_valid_transition(old_status, new_status):
            raise ValueError(
                f"Invalid status transition: {old_status.value} -> {new_status.value}"
            )

        # Update status
        action.status = new_status
        action.updated_at = datetime.utcnow()

        # Set date fields based on status
        if new_status == ActionStatus.IN_PROGRESS and not action.started_date:
            action.started_date = datetime.utcnow()
        elif new_status == ActionStatus.COMPLETED:
            action.completed_date = datetime.utcnow()
        elif new_status == ActionStatus.VERIFIED:
            action.verified_date = datetime.utcnow()

        # Add comment
        if comment:
            action.comments.append(ActionComment(
                author=updated_by or "System",
                content=comment,
                is_status_change=True
            ))
        else:
            self._add_status_change_comment(action, old_status.value, new_status.value)

        action.provenance_hash = self._calculate_provenance(action)

        self._log_audit("STATUS_CHANGED", action_id, {
            "old_status": old_status.value,
            "new_status": new_status.value,
            "updated_by": updated_by
        })

        logger.info(
            f"Action {action_id} status: {old_status.value} -> {new_status.value}"
        )
        return action

    def delete_action(self, action_id: str, deleted_by: str = "") -> bool:
        """
        Delete an action item (soft delete by cancellation).

        Args:
            action_id: Action identifier
            deleted_by: Person performing deletion

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If action not found or already completed
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]

        if action.status in [ActionStatus.COMPLETED, ActionStatus.VERIFIED]:
            raise ValueError("Cannot delete completed/verified actions")

        action.status = ActionStatus.CANCELLED
        action.updated_at = datetime.utcnow()
        action.comments.append(ActionComment(
            author=deleted_by or "System",
            content="Action cancelled",
            is_status_change=True
        ))

        self._log_audit("ACTION_CANCELLED", action_id, {
            "deleted_by": deleted_by
        })

        logger.info(f"Action cancelled: {action_id}")
        return True

    def _is_valid_transition(
        self,
        from_status: ActionStatus,
        to_status: ActionStatus
    ) -> bool:
        """Check if status transition is valid."""
        # Define valid transitions
        valid_transitions = {
            ActionStatus.DRAFT: [
                ActionStatus.OPEN, ActionStatus.CANCELLED
            ],
            ActionStatus.OPEN: [
                ActionStatus.IN_PROGRESS, ActionStatus.ON_HOLD,
                ActionStatus.CANCELLED, ActionStatus.OVERDUE
            ],
            ActionStatus.IN_PROGRESS: [
                ActionStatus.PENDING_VERIFICATION, ActionStatus.ON_HOLD,
                ActionStatus.COMPLETED, ActionStatus.OVERDUE
            ],
            ActionStatus.ON_HOLD: [
                ActionStatus.OPEN, ActionStatus.IN_PROGRESS,
                ActionStatus.CANCELLED
            ],
            ActionStatus.PENDING_VERIFICATION: [
                ActionStatus.COMPLETED, ActionStatus.VERIFIED,
                ActionStatus.IN_PROGRESS
            ],
            ActionStatus.COMPLETED: [
                ActionStatus.VERIFIED, ActionStatus.IN_PROGRESS
            ],
            ActionStatus.VERIFIED: [],  # Terminal state
            ActionStatus.CANCELLED: [],  # Terminal state
            ActionStatus.OVERDUE: [
                ActionStatus.IN_PROGRESS, ActionStatus.COMPLETED,
                ActionStatus.CANCELLED
            ],
        }

        return to_status in valid_transitions.get(from_status, [])

    # =========================================================================
    # ASSIGNMENT AND OWNERSHIP
    # =========================================================================

    def assign_action(
        self,
        action_id: str,
        assignee: str,
        assignee_email: str = "",
        assigned_by: str = ""
    ) -> ActionItem:
        """
        Assign action to a person.

        Args:
            action_id: Action identifier
            assignee: Person to assign
            assignee_email: Assignee's email
            assigned_by: Person making assignment

        Returns:
            Updated ActionItem
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]
        old_assignee = action.assignee

        action.assignee = assignee
        action.assignee_email = assignee_email
        action.updated_at = datetime.utcnow()

        # Move from DRAFT to OPEN
        if action.status == ActionStatus.DRAFT:
            action.status = ActionStatus.OPEN

        # Add comment
        action.comments.append(ActionComment(
            author=assigned_by or "System",
            content=f"Assigned to {assignee}" + (
                f" (was: {old_assignee})" if old_assignee else ""
            ),
            is_status_change=False
        ))

        action.provenance_hash = self._calculate_provenance(action)

        # Send notification
        if assignee_email:
            self._send_notification(
                action_id=action_id,
                event_type="ASSIGNED",
                recipients=[assignee_email],
                message=f"Action assigned: {action.title}"
            )

        self._log_audit("ACTION_ASSIGNED", action_id, {
            "old_assignee": old_assignee,
            "new_assignee": assignee,
            "assigned_by": assigned_by
        })

        logger.info(f"Action {action_id} assigned to {assignee}")
        return action

    def reassign_action(
        self,
        action_id: str,
        new_assignee: str,
        new_email: str = "",
        reason: str = "",
        reassigned_by: str = ""
    ) -> ActionItem:
        """
        Reassign action to a different person.

        Args:
            action_id: Action identifier
            new_assignee: New assignee
            new_email: New assignee email
            reason: Reason for reassignment
            reassigned_by: Person performing reassignment

        Returns:
            Updated ActionItem
        """
        return self.assign_action(
            action_id, new_assignee, new_email, reassigned_by
        )

    # =========================================================================
    # DUE DATE MANAGEMENT
    # =========================================================================

    def extend_due_date(
        self,
        action_id: str,
        new_due_date: datetime,
        reason: str,
        extended_by: str = ""
    ) -> ActionItem:
        """
        Extend action due date.

        Args:
            action_id: Action identifier
            new_due_date: New due date
            reason: Reason for extension
            extended_by: Person approving extension

        Returns:
            Updated ActionItem

        Raises:
            ValueError: If extension limit exceeded
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]

        # Check extension limit
        if action.extension_count >= self.config.max_extensions:
            raise ValueError(
                f"Maximum extensions ({self.config.max_extensions}) reached"
            )

        # Check extension days limit
        if action.due_date:
            extension_days = (new_due_date - action.due_date).days
            if extension_days > self.config.extension_days_limit:
                raise ValueError(
                    f"Extension exceeds limit of {self.config.extension_days_limit} days"
                )

        old_due_date = action.due_date
        action.due_date = new_due_date
        action.extension_count += 1
        action.updated_at = datetime.utcnow()

        # Add comment
        action.comments.append(ActionComment(
            author=extended_by or "System",
            content=(
                f"Due date extended from {old_due_date.date() if old_due_date else 'None'} "
                f"to {new_due_date.date()}. Reason: {reason}"
            )
        ))

        # Clear overdue status if applicable
        if action.status == ActionStatus.OVERDUE:
            action.status = ActionStatus.IN_PROGRESS

        action.provenance_hash = self._calculate_provenance(action)

        self._log_audit("DUE_DATE_EXTENDED", action_id, {
            "old_due_date": old_due_date.isoformat() if old_due_date else None,
            "new_due_date": new_due_date.isoformat(),
            "extension_count": action.extension_count,
            "reason": reason
        })

        logger.info(
            f"Action {action_id} due date extended to {new_due_date.date()}"
        )
        return action

    # =========================================================================
    # ESCALATION
    # =========================================================================

    def check_escalations(self) -> List[ActionItem]:
        """
        Check all actions for required escalations.

        Returns:
            List of actions that were escalated
        """
        now = datetime.utcnow()
        escalated = []

        for action in self.actions.values():
            if action.status in [
                ActionStatus.COMPLETED, ActionStatus.VERIFIED,
                ActionStatus.CANCELLED
            ]:
                continue

            if not action.due_date:
                continue

            # Check if overdue
            if action.due_date < now:
                days_overdue = (now - action.due_date).days

                # Update overdue status
                if self.config.auto_mark_overdue:
                    if action.status not in [ActionStatus.OVERDUE]:
                        action.status = ActionStatus.OVERDUE
                        action.updated_at = now

                # Check escalation rules
                for rule in sorted(
                    self.config.escalation_rules,
                    key=lambda x: x.days_overdue,
                    reverse=True
                ):
                    if days_overdue >= rule.days_overdue:
                        if action.escalation_level != rule.level:
                            self._escalate_action(action, rule, days_overdue)
                            escalated.append(action)
                        break

        if escalated:
            logger.warning(f"{len(escalated)} actions escalated")

        return escalated

    def _escalate_action(
        self,
        action: ActionItem,
        rule: EscalationRule,
        days_overdue: int
    ) -> None:
        """Apply escalation to an action."""
        old_level = action.escalation_level
        action.escalation_level = rule.level
        action.last_escalation_date = datetime.utcnow()
        action.updated_at = datetime.utcnow()

        # Add comment
        action.comments.append(ActionComment(
            author="System",
            content=(
                f"Escalated to {rule.level.value} "
                f"({days_overdue} days overdue)"
            ),
            is_status_change=True
        ))

        action.provenance_hash = self._calculate_provenance(action)

        # Send escalation notification
        self._send_notification(
            action_id=action.action_id,
            event_type="ESCALATED",
            recipients=rule.notify_specific,
            message=(
                f"Action {action.action_id} has been escalated to {rule.level.value}. "
                f"It is {days_overdue} days overdue."
            )
        )

        self._log_audit("ACTION_ESCALATED", action.action_id, {
            "old_level": old_level.value,
            "new_level": rule.level.value,
            "days_overdue": days_overdue
        })

    # =========================================================================
    # QUERIES AND FILTERING
    # =========================================================================

    def get_actions_by_status(
        self,
        status: ActionStatus
    ) -> List[ActionItem]:
        """Get actions filtered by status."""
        return [a for a in self.actions.values() if a.status == status]

    def get_actions_by_priority(
        self,
        priority: ActionPriority
    ) -> List[ActionItem]:
        """Get actions filtered by priority."""
        return [a for a in self.actions.values() if a.priority == priority]

    def get_actions_by_assignee(self, assignee: str) -> List[ActionItem]:
        """Get actions assigned to a specific person."""
        return [
            a for a in self.actions.values()
            if a.assignee.lower() == assignee.lower()
        ]

    def get_overdue_actions(self) -> List[ActionItem]:
        """Get all overdue actions."""
        now = datetime.utcnow()
        overdue = []

        for action in self.actions.values():
            if action.status in [
                ActionStatus.COMPLETED, ActionStatus.VERIFIED,
                ActionStatus.CANCELLED
            ]:
                continue

            if action.due_date and action.due_date < now:
                overdue.append(action)

        return sorted(overdue, key=lambda x: x.due_date)

    def get_upcoming_actions(
        self,
        days_ahead: int = 7
    ) -> List[ActionItem]:
        """Get actions due within specified period."""
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)

        upcoming = [
            a for a in self.actions.values()
            if a.status not in [
                ActionStatus.COMPLETED, ActionStatus.VERIFIED,
                ActionStatus.CANCELLED
            ]
            and a.due_date
            and now <= a.due_date <= cutoff
        ]

        return sorted(upcoming, key=lambda x: x.due_date)

    def get_actions_by_source(
        self,
        source: ActionSource,
        source_id: Optional[str] = None
    ) -> List[ActionItem]:
        """Get actions from a specific source."""
        actions = [a for a in self.actions.values() if a.source == source]

        if source_id:
            actions = [
                a for a in actions
                if a.source_reference_id == source_id or a.source_study_id == source_id
            ]

        return actions

    def get_actions_by_risk(self, risk_id: str) -> List[ActionItem]:
        """Get actions linked to a specific risk."""
        return [
            a for a in self.actions.values()
            if a.linked_risk_id == risk_id
        ]

    # =========================================================================
    # COMPLETION AND VERIFICATION
    # =========================================================================

    def complete_action(
        self,
        action_id: str,
        completion_evidence: str,
        completed_by: str = ""
    ) -> ActionItem:
        """
        Mark action as completed.

        Args:
            action_id: Action identifier
            completion_evidence: Evidence of completion
            completed_by: Person completing the action

        Returns:
            Updated ActionItem
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]
        action.completion_evidence = completion_evidence
        action.completed_date = datetime.utcnow()

        if self.config.require_verification:
            action.status = ActionStatus.PENDING_VERIFICATION
        else:
            action.status = ActionStatus.COMPLETED

        action.updated_at = datetime.utcnow()
        action.provenance_hash = self._calculate_provenance(action)

        self._log_audit("ACTION_COMPLETED", action_id, {
            "completed_by": completed_by,
            "has_evidence": bool(completion_evidence)
        })

        logger.info(f"Action {action_id} completed by {completed_by}")
        return action

    def verify_completion(
        self,
        action_id: str,
        verified_by: str,
        verification_notes: str = "",
        approved: bool = True
    ) -> ActionItem:
        """
        Verify action completion.

        Args:
            action_id: Action identifier
            verified_by: Person verifying completion
            verification_notes: Verification notes
            approved: Whether completion is approved

        Returns:
            Updated ActionItem
        """
        if action_id not in self.actions:
            raise ValueError(f"Action not found: {action_id}")

        action = self.actions[action_id]

        if action.status not in [
            ActionStatus.PENDING_VERIFICATION,
            ActionStatus.COMPLETED
        ]:
            raise ValueError(
                f"Action not ready for verification: {action.status.value}"
            )

        action.verified_by = verified_by
        action.verification_notes = verification_notes
        action.verified_date = datetime.utcnow()
        action.updated_at = datetime.utcnow()

        if approved:
            action.status = ActionStatus.VERIFIED
        else:
            action.status = ActionStatus.IN_PROGRESS
            action.comments.append(ActionComment(
                author=verified_by,
                content=f"Verification rejected: {verification_notes}",
                is_status_change=True
            ))

        action.provenance_hash = self._calculate_provenance(action)

        self._log_audit("ACTION_VERIFIED", action_id, {
            "verified_by": verified_by,
            "approved": approved
        })

        logger.info(
            f"Action {action_id} verification: "
            f"{'approved' if approved else 'rejected'}"
        )
        return action

    # =========================================================================
    # REPORTING AND EXPORT
    # =========================================================================

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report of all actions.

        Returns:
            Summary report dictionary
        """
        now = datetime.utcnow()

        # Status breakdown
        status_counts = {}
        for status in ActionStatus:
            status_counts[status.value] = sum(
                1 for a in self.actions.values() if a.status == status
            )

        # Priority breakdown
        priority_counts = {}
        for priority in ActionPriority:
            priority_counts[priority.value] = sum(
                1 for a in self.actions.values() if a.priority == priority
            )

        # Overdue analysis
        overdue_actions = self.get_overdue_actions()
        overdue_by_priority = {}
        for priority in ActionPriority:
            overdue_by_priority[priority.value] = sum(
                1 for a in overdue_actions if a.priority == priority
            )

        # Escalation summary
        escalation_counts = {}
        for level in EscalationLevel:
            escalation_counts[level.value] = sum(
                1 for a in self.actions.values()
                if a.escalation_level == level
            )

        # Completion metrics
        completed_actions = [
            a for a in self.actions.values()
            if a.status in [ActionStatus.COMPLETED, ActionStatus.VERIFIED]
        ]
        on_time_completions = sum(
            1 for a in completed_actions
            if a.completed_date and a.due_date and a.completed_date <= a.due_date
        )

        return {
            "report_date": now.isoformat(),
            "total_actions": len(self.actions),
            "status_breakdown": status_counts,
            "priority_breakdown": priority_counts,
            "overdue_summary": {
                "total_overdue": len(overdue_actions),
                "by_priority": overdue_by_priority,
                "average_days_overdue": (
                    sum((now - a.due_date).days for a in overdue_actions)
                    / len(overdue_actions)
                    if overdue_actions else 0
                )
            },
            "escalation_summary": escalation_counts,
            "completion_metrics": {
                "total_completed": len(completed_actions),
                "on_time_rate": (
                    on_time_completions / len(completed_actions) * 100
                    if completed_actions else 100
                ),
                "average_completion_time_days": (
                    sum(
                        (a.completed_date - a.created_date).days
                        for a in completed_actions
                        if a.completed_date
                    ) / len(completed_actions)
                    if completed_actions else 0
                )
            },
            "provenance_hash": hashlib.sha256(
                f"{now.isoformat()}|{len(self.actions)}|{len(overdue_actions)}".encode()
            ).hexdigest()
        }

    def export_to_json(self) -> str:
        """Export all actions to JSON format."""
        data = {
            "actions": [a.model_dump() for a in self.actions.values()],
            "export_date": datetime.utcnow().isoformat()
        }
        return json.dumps(data, indent=2, default=str)

    def export_to_csv(self) -> str:
        """Export actions to CSV format."""
        headers = [
            "action_id", "title", "priority", "status", "assignee",
            "due_date", "created_date", "source", "escalation_level"
        ]
        lines = [",".join(headers)]

        for a in self.actions.values():
            row = [
                a.action_id,
                f'"{a.title}"',
                a.priority.value,
                a.status.value,
                f'"{a.assignee}"',
                a.due_date.isoformat() if a.due_date else "",
                a.created_date.isoformat(),
                a.source.value,
                a.escalation_level.value
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    # =========================================================================
    # INTEGRATION WITH RISK MATRIX
    # =========================================================================

    def create_from_risk(
        self,
        risk_id: str,
        risk_level: str,
        risk_title: str,
        risk_description: str,
        severity: int,
        likelihood: int,
        mitigation_strategy: str = ""
    ) -> ActionItem:
        """
        Create action item from risk register entry.

        Args:
            risk_id: Risk identifier
            risk_level: Risk level (critical, high, medium, low)
            risk_title: Risk title
            risk_description: Risk description
            severity: Risk severity (1-5)
            likelihood: Risk likelihood (1-5)
            mitigation_strategy: Proposed mitigation

        Returns:
            Created ActionItem
        """
        priority = self.RISK_PRIORITY_MAP.get(
            risk_level.lower(),
            ActionPriority.MEDIUM
        )

        action = ActionItem(
            title=f"Mitigate Risk: {risk_title}",
            description=(
                f"{risk_description}\n\n"
                f"Mitigation Strategy: {mitigation_strategy}"
            ),
            priority=priority,
            category=ActionCategory.ENGINEERING,
            source=ActionSource.PHA,
            linked_risk_id=risk_id,
            risk_severity=severity,
            risk_likelihood=likelihood,
            risk_reduction_target=f"Reduce {risk_level} risk"
        )

        return self.create_action(action)

    def create_from_hazop(
        self,
        hazop_deviation: Dict[str, Any]
    ) -> ActionItem:
        """
        Create action item from HAZOP deviation.

        Args:
            hazop_deviation: HAZOP deviation dictionary

        Returns:
            Created ActionItem
        """
        recommendations = hazop_deviation.get("recommendations", [])
        title = recommendations[0] if recommendations else (
            f"Address deviation: {hazop_deviation.get('deviation_description', 'Unknown')}"
        )

        # Map risk ranking to priority
        risk_ranking = hazop_deviation.get("risk_ranking", 0)
        if risk_ranking >= 15:
            priority = ActionPriority.CRITICAL
        elif risk_ranking >= 12:
            priority = ActionPriority.HIGH
        elif risk_ranking >= 8:
            priority = ActionPriority.MEDIUM
        else:
            priority = ActionPriority.LOW

        action = ActionItem(
            title=title,
            description=(
                f"Deviation: {hazop_deviation.get('deviation_description', '')}\n"
                f"Causes: {', '.join(hazop_deviation.get('causes', []))}\n"
                f"Consequences: {', '.join(hazop_deviation.get('consequences', []))}\n"
                f"Existing Safeguards: {', '.join(hazop_deviation.get('existing_safeguards', []))}"
            ),
            priority=priority,
            category=ActionCategory.ENGINEERING,
            source=ActionSource.HAZOP,
            source_reference_id=hazop_deviation.get("deviation_id", ""),
            source_study_id=hazop_deviation.get("study_id", ""),
            assignee=hazop_deviation.get("action_party", ""),
            risk_severity=hazop_deviation.get("severity"),
            risk_likelihood=hazop_deviation.get("likelihood")
        )

        return self.create_action(action)

    def create_from_fmea(
        self,
        failure_mode: Dict[str, Any]
    ) -> ActionItem:
        """
        Create action item from FMEA failure mode.

        Args:
            failure_mode: FMEA failure mode dictionary

        Returns:
            Created ActionItem
        """
        # Map RPN to priority
        rpn = failure_mode.get("rpn", 0)
        if rpn >= 200:
            priority = ActionPriority.CRITICAL
        elif rpn >= 100:
            priority = ActionPriority.HIGH
        elif rpn >= 50:
            priority = ActionPriority.MEDIUM
        else:
            priority = ActionPriority.LOW

        action = ActionItem(
            title=failure_mode.get("recommended_action", f"Address FM: {failure_mode.get('failure_mode', '')}"),
            description=(
                f"Component: {failure_mode.get('component_name', '')}\n"
                f"Failure Mode: {failure_mode.get('failure_mode', '')}\n"
                f"End Effect: {failure_mode.get('end_effect', '')}\n"
                f"RPN: {rpn}"
            ),
            priority=priority,
            category=ActionCategory.MAINTENANCE,
            source=ActionSource.FMEA,
            source_reference_id=failure_mode.get("fm_id", ""),
            assignee=failure_mode.get("responsibility", "")
        )

        return self.create_action(action)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_provenance(self, action: ActionItem) -> str:
        """Calculate SHA-256 provenance hash for action."""
        data_str = (
            f"{action.action_id}|"
            f"{action.title}|"
            f"{action.status.value}|"
            f"{action.priority.value}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _add_status_change_comment(
        self,
        action: ActionItem,
        old_status: str,
        new_status: str
    ) -> None:
        """Add status change comment to action."""
        action.comments.append(ActionComment(
            author="System",
            content=f"Status changed: {old_status} -> {new_status}",
            is_status_change=True
        ))

    def _send_notification(
        self,
        action_id: str,
        event_type: str,
        recipients: List[str],
        message: str
    ) -> bool:
        """Send notification event."""
        event = NotificationEvent(
            action_id=action_id,
            event_type=event_type,
            recipients=recipients,
            message=message
        )

        if self.notification_handler and recipients:
            try:
                event.sent_successfully = self.notification_handler(event)
                event.sent_at = datetime.utcnow()
            except Exception as e:
                logger.error(f"Notification failed: {e}")
                event.sent_successfully = False

        self.notifications.append(event)
        return event.sent_successfully

    def _log_audit(
        self,
        event_type: str,
        action_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log event to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "action_id": action_id,
            "details": details
        })


if __name__ == "__main__":
    # Example usage
    print("ActionTracking module loaded successfully")

    # Create tracker
    tracker = ActionTracker()

    # Create an action
    action = ActionItem(
        title="Install high-high level alarm on T-101",
        description="Add independent high-high level alarm with shutdown capability",
        priority=ActionPriority.HIGH,
        category=ActionCategory.ENGINEERING,
        source=ActionSource.HAZOP,
        assignee="John Smith"
    )

    created = tracker.create_action(action)
    print(f"Created: {created.action_id}")

    # Check summary
    summary = tracker.get_summary_report()
    print(f"Total Actions: {summary['total_actions']}")
