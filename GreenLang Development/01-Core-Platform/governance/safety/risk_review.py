r"""
RiskReview - Periodic Risk Review Process Management

This module implements periodic risk review scheduling, documentation, and
workflow management per IEC 61511 and ISO 31000 standards.

Key Features:
- ReviewScheduler class for periodic review scheduling
- Review templates and checklists for consistent assessments
- Review meeting documentation and attendance tracking
- Risk reassessment workflow with approval gates
- Complete audit trail for all review changes
- Integration with risk register and action tracker

Review Types:
- Scheduled: Regular periodic reviews (quarterly, annual)
- Triggered: Event-triggered reviews (incident, MOC, audit finding)
- Ad-hoc: On-demand reviews for specific concerns

Reference:
- IEC 61511-1:2016 Clause 5.2.6 - Functional Safety Management
- ISO 31000:2018 - Risk Management (Monitoring and Review)
- OSHA 1910.119 - Process Safety Management (Management of Change)

Example:
    >>> from greenlang.safety.risk_review import ReviewScheduler, ReviewSession
    >>> scheduler = ReviewScheduler()
    >>> session = scheduler.create_review_session(
    ...     title="Q4 2024 Risk Review",
    ...     review_type=ReviewType.QUARTERLY
    ... )
    >>> scheduler.add_risks_to_review(session.session_id, risk_ids)
"""

from typing import Dict, List, Optional, Any, ClassVar
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

class ReviewType(str, Enum):
    """Type of risk review."""

    QUARTERLY = "quarterly"  # Regular quarterly review
    SEMI_ANNUAL = "semi_annual"  # Every 6 months
    ANNUAL = "annual"  # Annual comprehensive review
    INCIDENT_TRIGGERED = "incident_triggered"  # After incident/near-miss
    MOC_TRIGGERED = "moc_triggered"  # After Management of Change
    AUDIT_TRIGGERED = "audit_triggered"  # After audit finding
    REGULATORY = "regulatory"  # Regulatory requirement
    AD_HOC = "ad_hoc"  # On-demand review


class ReviewStatus(str, Enum):
    """Review session status."""

    SCHEDULED = "scheduled"  # Review scheduled
    PREPARATION = "preparation"  # Pre-review preparation
    IN_PROGRESS = "in_progress"  # Review underway
    PENDING_APPROVAL = "pending_approval"  # Awaiting approval
    APPROVED = "approved"  # Review approved
    REJECTED = "rejected"  # Review rejected for rework
    COMPLETED = "completed"  # Review completed
    CANCELLED = "cancelled"  # Review cancelled


class ChecklistItemStatus(str, Enum):
    """Status of review checklist items."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"
    DEFERRED = "deferred"


class ReassessmentDecision(str, Enum):
    """Decision options for risk reassessment."""

    NO_CHANGE = "no_change"  # Risk assessment unchanged
    SEVERITY_CHANGED = "severity_changed"  # Severity rating changed
    LIKELIHOOD_CHANGED = "likelihood_changed"  # Likelihood changed
    BOTH_CHANGED = "both_changed"  # Both ratings changed
    CONTROLS_UPDATED = "controls_updated"  # Controls modified
    RISK_CLOSED = "risk_closed"  # Risk no longer applicable
    RISK_ACCEPTED = "risk_accepted"  # Risk formally accepted
    ESCALATED = "escalated"  # Escalated for further review


class ReviewOutcome(str, Enum):
    """Overall review session outcome."""

    ALL_RISKS_REVIEWED = "all_risks_reviewed"
    PARTIAL_REVIEW = "partial_review"
    NEW_RISKS_IDENTIFIED = "new_risks_identified"
    ACTIONS_GENERATED = "actions_generated"
    ESCALATION_REQUIRED = "escalation_required"
    NO_CHANGES = "no_changes"


# =============================================================================
# DATA MODELS
# =============================================================================

class ChecklistItem(BaseModel):
    """Single item in a review checklist."""

    item_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique item identifier"
    )
    category: str = Field(
        ...,
        description="Checklist category"
    )
    description: str = Field(
        ...,
        description="Item description/question"
    )
    status: ChecklistItemStatus = Field(
        default=ChecklistItemStatus.NOT_STARTED,
        description="Completion status"
    )
    response: str = Field(
        default="",
        description="Response/answer to item"
    )
    evidence: str = Field(
        default="",
        description="Supporting evidence reference"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )
    completed_by: Optional[str] = Field(
        None,
        description="Person who completed item"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether item is mandatory"
    )


class ReviewChecklist(BaseModel):
    """Complete review checklist template."""

    checklist_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique checklist identifier"
    )
    name: str = Field(
        ...,
        description="Checklist name"
    )
    review_type: ReviewType = Field(
        ...,
        description="Applicable review type"
    )
    version: str = Field(
        default="1.0",
        description="Template version"
    )
    items: List[ChecklistItem] = Field(
        default_factory=list,
        description="Checklist items"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )


class Attendee(BaseModel):
    """Review meeting attendee."""

    attendee_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attendee identifier"
    )
    name: str = Field(
        ...,
        description="Attendee name"
    )
    role: str = Field(
        default="",
        description="Role in review (chair, scribe, SME, etc.)"
    )
    department: str = Field(
        default="",
        description="Department"
    )
    email: str = Field(
        default="",
        description="Email address"
    )
    attended: bool = Field(
        default=False,
        description="Whether attendee was present"
    )
    is_required: bool = Field(
        default=True,
        description="Whether attendance is mandatory"
    )
    signature: Optional[str] = Field(
        None,
        description="Digital signature or confirmation"
    )
    signature_date: Optional[datetime] = Field(
        None,
        description="Date of signature"
    )


class MeetingMinutes(BaseModel):
    """Review meeting documentation."""

    meeting_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique meeting identifier"
    )
    session_id: str = Field(
        ...,
        description="Associated review session"
    )
    meeting_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Meeting date and time"
    )
    location: str = Field(
        default="",
        description="Meeting location"
    )
    attendees: List[Attendee] = Field(
        default_factory=list,
        description="Meeting attendees"
    )
    agenda: List[str] = Field(
        default_factory=list,
        description="Meeting agenda items"
    )
    discussion_points: List[str] = Field(
        default_factory=list,
        description="Key discussion points"
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="Decisions made"
    )
    action_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Action items generated"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Planned next steps"
    )
    duration_minutes: int = Field(
        default=60,
        description="Meeting duration"
    )
    scribe: str = Field(
        default="",
        description="Person recording minutes"
    )
    approved: bool = Field(
        default=False,
        description="Minutes approved"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approver name"
    )
    approved_at: Optional[datetime] = Field(
        None,
        description="Approval timestamp"
    )


class RiskReassessment(BaseModel):
    """Individual risk reassessment record."""

    reassessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reassessment identifier"
    )
    session_id: str = Field(
        ...,
        description="Associated review session"
    )
    risk_id: str = Field(
        ...,
        description="Risk being reassessed"
    )
    risk_title: str = Field(
        default="",
        description="Risk title for reference"
    )

    # Previous assessment
    previous_severity: int = Field(..., ge=1, le=5)
    previous_likelihood: int = Field(..., ge=1, le=5)
    previous_risk_level: str = Field(...)
    previous_status: str = Field(...)

    # New assessment
    new_severity: Optional[int] = Field(None, ge=1, le=5)
    new_likelihood: Optional[int] = Field(None, ge=1, le=5)
    new_risk_level: Optional[str] = Field(None)
    new_status: Optional[str] = Field(None)

    # Decision
    decision: ReassessmentDecision = Field(
        default=ReassessmentDecision.NO_CHANGE,
        description="Reassessment decision"
    )
    justification: str = Field(
        default="",
        description="Justification for decision"
    )

    # Controls review
    controls_effective: bool = Field(
        default=True,
        description="Are existing controls effective"
    )
    controls_added: List[str] = Field(
        default_factory=list,
        description="New controls added"
    )
    controls_removed: List[str] = Field(
        default_factory=list,
        description="Controls removed"
    )

    # Actions
    new_actions: List[str] = Field(
        default_factory=list,
        description="New actions identified"
    )

    # Review details
    reviewed_by: str = Field(
        default="",
        description="Reviewer name"
    )
    reviewed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Review timestamp"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )


class ReviewSession(BaseModel):
    """Complete risk review session."""

    session_id: str = Field(
        default_factory=lambda: f"REV-{uuid.uuid4().hex[:8].upper()}",
        description="Unique session identifier"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Review session title"
    )
    description: str = Field(
        default="",
        description="Session description"
    )
    review_type: ReviewType = Field(
        ...,
        description="Type of review"
    )
    status: ReviewStatus = Field(
        default=ReviewStatus.SCHEDULED,
        description="Current status"
    )

    # Scope
    scope: str = Field(
        default="",
        description="Review scope definition"
    )
    facility: str = Field(
        default="",
        description="Facility/area under review"
    )
    risk_ids: List[str] = Field(
        default_factory=list,
        description="Risk IDs included in review"
    )
    categories_included: List[str] = Field(
        default_factory=list,
        description="Risk categories included"
    )

    # Scheduling
    scheduled_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Scheduled review date"
    )
    due_date: Optional[datetime] = Field(
        None,
        description="Due date for completion"
    )
    actual_start_date: Optional[datetime] = Field(
        None,
        description="Actual start date"
    )
    actual_end_date: Optional[datetime] = Field(
        None,
        description="Actual completion date"
    )

    # Participants
    facilitator: str = Field(
        default="",
        description="Review facilitator"
    )
    scribe: str = Field(
        default="",
        description="Session scribe"
    )
    attendees: List[Attendee] = Field(
        default_factory=list,
        description="Review attendees"
    )

    # Content
    checklist: Optional[ReviewChecklist] = Field(
        None,
        description="Review checklist"
    )
    reassessments: List[RiskReassessment] = Field(
        default_factory=list,
        description="Risk reassessments"
    )
    meetings: List[MeetingMinutes] = Field(
        default_factory=list,
        description="Meeting minutes"
    )

    # Outcomes
    outcome: Optional[ReviewOutcome] = Field(
        None,
        description="Overall outcome"
    )
    summary: str = Field(
        default="",
        description="Review summary"
    )
    new_risks_identified: List[str] = Field(
        default_factory=list,
        description="New risks identified"
    )
    actions_generated: List[str] = Field(
        default_factory=list,
        description="Action IDs generated"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )

    # Approval
    requires_approval: bool = Field(
        default=True,
        description="Whether approval is required"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approver name"
    )
    approved_at: Optional[datetime] = Field(
        None,
        description="Approval timestamp"
    )
    approval_comments: str = Field(
        default="",
        description="Approval comments"
    )

    # Metadata
    trigger_event: str = Field(
        default="",
        description="Event that triggered review (if applicable)"
    )
    trigger_reference: str = Field(
        default="",
        description="Reference ID for trigger"
    )
    created_by: str = Field(
        default="",
        description="Person who created session"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
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


class ReviewSchedule(BaseModel):
    """Recurring review schedule configuration."""

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique schedule identifier"
    )
    name: str = Field(
        ...,
        description="Schedule name"
    )
    review_type: ReviewType = Field(
        ...,
        description="Type of review"
    )
    frequency_days: int = Field(
        ...,
        ge=1,
        description="Frequency in days"
    )
    scope_filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filter for risks to include"
    )
    lead_time_days: int = Field(
        default=14,
        description="Days before due to create session"
    )
    last_run: Optional[datetime] = Field(
        None,
        description="Last execution timestamp"
    )
    next_run: Optional[datetime] = Field(
        None,
        description="Next scheduled execution"
    )
    is_active: bool = Field(
        default=True,
        description="Whether schedule is active"
    )
    default_facilitator: str = Field(
        default="",
        description="Default facilitator"
    )
    required_attendees: List[str] = Field(
        default_factory=list,
        description="Default required attendees"
    )


class ReviewSchedulerConfig(BaseModel):
    """Configuration for review scheduler."""

    # Standard intervals (days)
    quarterly_interval: int = Field(default=90)
    semi_annual_interval: int = Field(default=180)
    annual_interval: int = Field(default=365)

    # Review requirements
    require_attendance_quorum: bool = Field(default=True)
    attendance_quorum_percent: int = Field(default=50)
    require_facilitator: bool = Field(default=True)
    require_checklist: bool = Field(default=True)

    # Notification settings
    reminder_days_before: List[int] = Field(
        default_factory=lambda: [14, 7, 1]
    )

    # Escalation
    escalation_days_overdue: int = Field(default=7)


# =============================================================================
# REVIEW SCHEDULER
# =============================================================================

class ReviewScheduler:
    """
    Periodic Risk Review Scheduler and Manager.

    Manages the complete lifecycle of risk reviews including:
    - Scheduled and triggered review sessions
    - Review templates and checklists
    - Meeting documentation
    - Risk reassessment workflow
    - Approval gates and audit trail

    Attributes:
        sessions: Dict of session_id to ReviewSession
        schedules: Dict of schedule_id to ReviewSchedule
        templates: Dict of template_id to ReviewChecklist

    Example:
        >>> scheduler = ReviewScheduler()
        >>> session = scheduler.create_review_session(
        ...     title="Annual Safety Review",
        ...     review_type=ReviewType.ANNUAL
        ... )
    """

    # Standard checklist templates by review type
    STANDARD_CHECKLISTS: ClassVar[Dict[ReviewType, List[Dict[str, Any]]]] = {
        ReviewType.QUARTERLY: [
            {"category": "Risk Status", "description": "Review current status of all high/critical risks", "mandatory": True},
            {"category": "Risk Status", "description": "Verify mitigation progress for overdue items", "mandatory": True},
            {"category": "Controls", "description": "Confirm control effectiveness for high risks", "mandatory": True},
            {"category": "Actions", "description": "Review open action items and completion status", "mandatory": True},
            {"category": "Incidents", "description": "Review any incidents since last review", "mandatory": False},
            {"category": "Changes", "description": "Review any process/equipment changes", "mandatory": True},
        ],
        ReviewType.ANNUAL: [
            {"category": "Comprehensive", "description": "Review ALL risks in register for currency", "mandatory": True},
            {"category": "Comprehensive", "description": "Validate risk severity ratings", "mandatory": True},
            {"category": "Comprehensive", "description": "Validate risk likelihood ratings", "mandatory": True},
            {"category": "Controls", "description": "Full review of control effectiveness", "mandatory": True},
            {"category": "Controls", "description": "Review safeguard verification status", "mandatory": True},
            {"category": "Compliance", "description": "Review regulatory compliance status", "mandatory": True},
            {"category": "Metrics", "description": "Review annual risk KPIs and trends", "mandatory": True},
            {"category": "Lessons Learned", "description": "Document lessons learned", "mandatory": False},
            {"category": "Improvements", "description": "Identify improvement opportunities", "mandatory": False},
        ],
        ReviewType.INCIDENT_TRIGGERED: [
            {"category": "Incident", "description": "Document incident details and root cause", "mandatory": True},
            {"category": "Incident", "description": "Identify related risks affected", "mandatory": True},
            {"category": "Assessment", "description": "Reassess risk ratings based on incident", "mandatory": True},
            {"category": "Controls", "description": "Evaluate control failures or gaps", "mandatory": True},
            {"category": "Actions", "description": "Define corrective actions", "mandatory": True},
            {"category": "Prevention", "description": "Identify preventive measures", "mandatory": True},
        ],
    }

    def __init__(
        self,
        config: Optional[ReviewSchedulerConfig] = None,
        risk_register: Optional[Any] = None,
        action_tracker: Optional[Any] = None
    ):
        """
        Initialize ReviewScheduler.

        Args:
            config: Optional scheduler configuration
            risk_register: Optional RiskRegister instance
            action_tracker: Optional ActionTracker instance
        """
        self.config = config or ReviewSchedulerConfig()
        self.risk_register = risk_register
        self.action_tracker = action_tracker

        self.sessions: Dict[str, ReviewSession] = {}
        self.schedules: Dict[str, ReviewSchedule] = {}
        self.templates: Dict[str, ReviewChecklist] = {}
        self.audit_trail: List[Dict[str, Any]] = []

        # Initialize standard templates
        self._initialize_templates()

        logger.info("ReviewScheduler initialized")

    def _initialize_templates(self) -> None:
        """Initialize standard checklist templates."""
        for review_type, items in self.STANDARD_CHECKLISTS.items():
            checklist = ReviewChecklist(
                name=f"Standard {review_type.value.replace('_', ' ').title()} Checklist",
                review_type=review_type,
                items=[
                    ChecklistItem(
                        category=item["category"],
                        description=item["description"],
                        is_mandatory=item.get("mandatory", True)
                    )
                    for item in items
                ]
            )
            self.templates[checklist.checklist_id] = checklist

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def create_review_session(
        self,
        title: str,
        review_type: ReviewType,
        scheduled_date: Optional[datetime] = None,
        scope: str = "",
        facility: str = "",
        facilitator: str = "",
        created_by: str = ""
    ) -> ReviewSession:
        """
        Create a new review session.

        Args:
            title: Session title
            review_type: Type of review
            scheduled_date: Scheduled date (defaults to now)
            scope: Review scope
            facility: Facility/area
            facilitator: Facilitator name
            created_by: Creator name

        Returns:
            Created ReviewSession
        """
        if scheduled_date is None:
            scheduled_date = datetime.utcnow()

        # Calculate due date based on review type
        if review_type == ReviewType.INCIDENT_TRIGGERED:
            due_days = 7
        elif review_type == ReviewType.QUARTERLY:
            due_days = 14
        else:
            due_days = 30

        session = ReviewSession(
            title=title,
            review_type=review_type,
            scheduled_date=scheduled_date,
            due_date=scheduled_date + timedelta(days=due_days),
            scope=scope,
            facility=facility,
            facilitator=facilitator,
            created_by=created_by,
            status=ReviewStatus.SCHEDULED
        )

        # Add standard checklist
        session.checklist = self._get_template_for_type(review_type)

        session.provenance_hash = self._calculate_provenance(session)
        self.sessions[session.session_id] = session

        self._log_audit("SESSION_CREATED", session.session_id, {
            "title": title,
            "type": review_type.value,
            "scheduled": scheduled_date.isoformat()
        })

        logger.info(f"Review session created: {session.session_id} - {title}")
        return session

    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_session_status(
        self,
        session_id: str,
        new_status: ReviewStatus,
        updated_by: str = ""
    ) -> ReviewSession:
        """
        Update session status.

        Args:
            session_id: Session identifier
            new_status: New status
            updated_by: Person making update

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        old_status = session.status

        session.status = new_status
        session.updated_at = datetime.utcnow()

        # Set date fields based on status
        if new_status == ReviewStatus.IN_PROGRESS and not session.actual_start_date:
            session.actual_start_date = datetime.utcnow()
        elif new_status == ReviewStatus.COMPLETED:
            session.actual_end_date = datetime.utcnow()

        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("STATUS_CHANGED", session_id, {
            "old_status": old_status.value,
            "new_status": new_status.value,
            "updated_by": updated_by
        })

        logger.info(
            f"Session {session_id} status: {old_status.value} -> {new_status.value}"
        )
        return session

    # =========================================================================
    # RISK INCLUSION
    # =========================================================================

    def add_risks_to_review(
        self,
        session_id: str,
        risk_ids: List[str]
    ) -> ReviewSession:
        """
        Add risks to a review session.

        Args:
            session_id: Session identifier
            risk_ids: List of risk IDs to include

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        for risk_id in risk_ids:
            if risk_id not in session.risk_ids:
                session.risk_ids.append(risk_id)

        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        logger.info(f"Added {len(risk_ids)} risks to session {session_id}")
        return session

    def add_risks_by_filter(
        self,
        session_id: str,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        status: Optional[str] = None
    ) -> ReviewSession:
        """
        Add risks matching filter criteria.

        Args:
            session_id: Session identifier
            category: Filter by category
            risk_level: Filter by risk level
            status: Filter by status

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        if not self.risk_register:
            raise ValueError("Risk register not connected")

        matching_risks = []
        for risk in self.risk_register.risks.values():
            if category and risk.category.value != category:
                continue
            if risk_level and risk.risk_level.value != risk_level:
                continue
            if status and risk.status.value != status:
                continue
            matching_risks.append(risk.risk_id)

        return self.add_risks_to_review(session_id, matching_risks)

    # =========================================================================
    # REASSESSMENT WORKFLOW
    # =========================================================================

    def create_reassessment(
        self,
        session_id: str,
        risk_id: str,
        reviewer: str
    ) -> RiskReassessment:
        """
        Create a risk reassessment record.

        Args:
            session_id: Session identifier
            risk_id: Risk being reassessed
            reviewer: Person performing reassessment

        Returns:
            Created RiskReassessment
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        if not self.risk_register:
            raise ValueError("Risk register not connected")

        risk = self.risk_register.get_risk(risk_id)
        if not risk:
            raise ValueError(f"Risk not found: {risk_id}")

        reassessment = RiskReassessment(
            session_id=session_id,
            risk_id=risk_id,
            risk_title=risk.title,
            previous_severity=risk.severity,
            previous_likelihood=risk.likelihood,
            previous_risk_level=risk.risk_level.value,
            previous_status=risk.status.value,
            reviewed_by=reviewer
        )

        session = self.sessions[session_id]
        session.reassessments.append(reassessment)
        session.updated_at = datetime.utcnow()

        logger.info(f"Reassessment created for risk {risk_id} in session {session_id}")
        return reassessment

    def complete_reassessment(
        self,
        session_id: str,
        reassessment_id: str,
        decision: ReassessmentDecision,
        justification: str,
        new_severity: Optional[int] = None,
        new_likelihood: Optional[int] = None,
        controls_effective: bool = True,
        new_actions: Optional[List[str]] = None,
        notes: str = ""
    ) -> RiskReassessment:
        """
        Complete a risk reassessment with decision.

        Args:
            session_id: Session identifier
            reassessment_id: Reassessment identifier
            decision: Reassessment decision
            justification: Justification for decision
            new_severity: New severity rating (if changed)
            new_likelihood: New likelihood rating (if changed)
            controls_effective: Whether controls are effective
            new_actions: New actions identified
            notes: Additional notes

        Returns:
            Updated RiskReassessment
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        reassessment = None

        for r in session.reassessments:
            if r.reassessment_id == reassessment_id:
                reassessment = r
                break

        if not reassessment:
            raise ValueError(f"Reassessment not found: {reassessment_id}")

        reassessment.decision = decision
        reassessment.justification = justification
        reassessment.new_severity = new_severity
        reassessment.new_likelihood = new_likelihood
        reassessment.controls_effective = controls_effective
        reassessment.new_actions = new_actions or []
        reassessment.notes = notes
        reassessment.reviewed_at = datetime.utcnow()

        # Calculate new risk level if ratings changed
        if new_severity and new_likelihood:
            # Import here to avoid circular dependency
            from greenlang.safety.risk_register import RiskMatrix
            reassessment.new_risk_level = RiskMatrix.calculate_risk_level(
                new_severity, new_likelihood
            ).value

        # Update risk register if connected
        if self.risk_register and decision != ReassessmentDecision.NO_CHANGE:
            self._apply_reassessment_to_register(reassessment)

        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("REASSESSMENT_COMPLETED", reassessment_id, {
            "session_id": session_id,
            "risk_id": reassessment.risk_id,
            "decision": decision.value
        })

        logger.info(
            f"Reassessment {reassessment_id} completed: {decision.value}"
        )
        return reassessment

    def _apply_reassessment_to_register(
        self,
        reassessment: RiskReassessment
    ) -> None:
        """Apply reassessment changes to risk register."""
        if not self.risk_register:
            return

        updates = {}

        if reassessment.new_severity:
            updates["severity"] = reassessment.new_severity
        if reassessment.new_likelihood:
            updates["likelihood"] = reassessment.new_likelihood
        if reassessment.new_status:
            updates["status"] = reassessment.new_status

        if updates:
            try:
                self.risk_register.update_risk(reassessment.risk_id, updates)
                logger.info(
                    f"Risk {reassessment.risk_id} updated from reassessment"
                )
            except Exception as e:
                logger.error(f"Failed to update risk: {e}")

    # =========================================================================
    # MEETING DOCUMENTATION
    # =========================================================================

    def create_meeting(
        self,
        session_id: str,
        meeting_date: Optional[datetime] = None,
        location: str = "",
        agenda: Optional[List[str]] = None
    ) -> MeetingMinutes:
        """
        Create meeting minutes for a session.

        Args:
            session_id: Session identifier
            meeting_date: Meeting date
            location: Meeting location
            agenda: Meeting agenda

        Returns:
            Created MeetingMinutes
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        meeting = MeetingMinutes(
            session_id=session_id,
            meeting_date=meeting_date or datetime.utcnow(),
            location=location,
            agenda=agenda or [],
            scribe=session.scribe
        )

        # Copy attendees from session
        meeting.attendees = [
            Attendee(
                name=a.name,
                role=a.role,
                department=a.department,
                email=a.email,
                is_required=a.is_required
            )
            for a in session.attendees
        ]

        session.meetings.append(meeting)
        session.updated_at = datetime.utcnow()

        logger.info(f"Meeting created for session {session_id}")
        return meeting

    def update_meeting(
        self,
        session_id: str,
        meeting_id: str,
        updates: Dict[str, Any]
    ) -> MeetingMinutes:
        """
        Update meeting minutes.

        Args:
            session_id: Session identifier
            meeting_id: Meeting identifier
            updates: Field updates

        Returns:
            Updated MeetingMinutes
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        meeting = None

        for m in session.meetings:
            if m.meeting_id == meeting_id:
                meeting = m
                break

        if not meeting:
            raise ValueError(f"Meeting not found: {meeting_id}")

        for key, value in updates.items():
            if hasattr(meeting, key):
                setattr(meeting, key, value)

        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        return meeting

    def record_attendance(
        self,
        session_id: str,
        meeting_id: str,
        attendee_name: str,
        attended: bool = True,
        signature: str = ""
    ) -> MeetingMinutes:
        """
        Record attendee attendance and signature.

        Args:
            session_id: Session identifier
            meeting_id: Meeting identifier
            attendee_name: Attendee name
            attended: Whether they attended
            signature: Digital signature

        Returns:
            Updated MeetingMinutes
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        meeting = None

        for m in session.meetings:
            if m.meeting_id == meeting_id:
                meeting = m
                break

        if not meeting:
            raise ValueError(f"Meeting not found: {meeting_id}")

        for attendee in meeting.attendees:
            if attendee.name.lower() == attendee_name.lower():
                attendee.attended = attended
                if signature:
                    attendee.signature = signature
                    attendee.signature_date = datetime.utcnow()
                break

        return meeting

    # =========================================================================
    # CHECKLIST MANAGEMENT
    # =========================================================================

    def complete_checklist_item(
        self,
        session_id: str,
        item_id: str,
        status: ChecklistItemStatus,
        response: str = "",
        completed_by: str = ""
    ) -> ReviewSession:
        """
        Complete a checklist item.

        Args:
            session_id: Session identifier
            item_id: Checklist item ID
            status: Completion status
            response: Response/answer
            completed_by: Person completing

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        if not session.checklist:
            raise ValueError("Session has no checklist")

        for item in session.checklist.items:
            if item.item_id == item_id:
                item.status = status
                item.response = response
                item.completed_by = completed_by
                item.completed_at = datetime.utcnow()
                break

        session.updated_at = datetime.utcnow()
        return session

    def get_checklist_progress(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get checklist completion progress.

        Args:
            session_id: Session identifier

        Returns:
            Progress summary dictionary
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        if not session.checklist:
            return {"error": "No checklist"}

        total = len(session.checklist.items)
        completed = sum(
            1 for item in session.checklist.items
            if item.status == ChecklistItemStatus.COMPLETED
        )
        mandatory_total = sum(
            1 for item in session.checklist.items
            if item.is_mandatory
        )
        mandatory_completed = sum(
            1 for item in session.checklist.items
            if item.is_mandatory and item.status == ChecklistItemStatus.COMPLETED
        )

        return {
            "total_items": total,
            "completed_items": completed,
            "completion_percent": completed / total * 100 if total > 0 else 0,
            "mandatory_total": mandatory_total,
            "mandatory_completed": mandatory_completed,
            "mandatory_complete": mandatory_completed == mandatory_total,
            "by_status": {
                status.value: sum(
                    1 for item in session.checklist.items
                    if item.status == status
                )
                for status in ChecklistItemStatus
            }
        }

    # =========================================================================
    # APPROVAL WORKFLOW
    # =========================================================================

    def submit_for_approval(
        self,
        session_id: str,
        summary: str
    ) -> ReviewSession:
        """
        Submit review session for approval.

        Args:
            session_id: Session identifier
            summary: Review summary

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        # Validate readiness
        if session.checklist:
            progress = self.get_checklist_progress(session_id)
            if not progress.get("mandatory_complete", False):
                raise ValueError("Mandatory checklist items not complete")

        session.status = ReviewStatus.PENDING_APPROVAL
        session.summary = summary
        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("SUBMITTED_FOR_APPROVAL", session_id, {
            "summary_length": len(summary)
        })

        logger.info(f"Session {session_id} submitted for approval")
        return session

    def approve_review(
        self,
        session_id: str,
        approved_by: str,
        comments: str = ""
    ) -> ReviewSession:
        """
        Approve a review session.

        Args:
            session_id: Session identifier
            approved_by: Approver name
            comments: Approval comments

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        if session.status != ReviewStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Cannot approve session in status: {session.status.value}"
            )

        session.status = ReviewStatus.APPROVED
        session.approved_by = approved_by
        session.approved_at = datetime.utcnow()
        session.approval_comments = comments
        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("SESSION_APPROVED", session_id, {
            "approved_by": approved_by
        })

        logger.info(f"Session {session_id} approved by {approved_by}")
        return session

    def reject_review(
        self,
        session_id: str,
        rejected_by: str,
        reason: str
    ) -> ReviewSession:
        """
        Reject a review session for rework.

        Args:
            session_id: Session identifier
            rejected_by: Rejector name
            reason: Rejection reason

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        session.status = ReviewStatus.REJECTED
        session.approval_comments = f"Rejected by {rejected_by}: {reason}"
        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("SESSION_REJECTED", session_id, {
            "rejected_by": rejected_by,
            "reason": reason
        })

        logger.info(f"Session {session_id} rejected: {reason}")
        return session

    def complete_review(
        self,
        session_id: str,
        outcome: ReviewOutcome,
        final_summary: str = ""
    ) -> ReviewSession:
        """
        Mark review as completed.

        Args:
            session_id: Session identifier
            outcome: Overall outcome
            final_summary: Final summary

        Returns:
            Updated ReviewSession
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        session.status = ReviewStatus.COMPLETED
        session.outcome = outcome
        session.actual_end_date = datetime.utcnow()

        if final_summary:
            session.summary = final_summary

        session.updated_at = datetime.utcnow()
        session.provenance_hash = self._calculate_provenance(session)

        self._log_audit("SESSION_COMPLETED", session_id, {
            "outcome": outcome.value
        })

        logger.info(f"Session {session_id} completed: {outcome.value}")
        return session

    # =========================================================================
    # SCHEDULING
    # =========================================================================

    def create_schedule(
        self,
        name: str,
        review_type: ReviewType,
        frequency_days: int,
        scope_filter: Optional[Dict[str, Any]] = None,
        default_facilitator: str = "",
        required_attendees: Optional[List[str]] = None
    ) -> ReviewSchedule:
        """
        Create a recurring review schedule.

        Args:
            name: Schedule name
            review_type: Type of review
            frequency_days: Frequency in days
            scope_filter: Filter for risks to include
            default_facilitator: Default facilitator
            required_attendees: Required attendees

        Returns:
            Created ReviewSchedule
        """
        schedule = ReviewSchedule(
            name=name,
            review_type=review_type,
            frequency_days=frequency_days,
            scope_filter=scope_filter or {},
            default_facilitator=default_facilitator,
            required_attendees=required_attendees or [],
            next_run=datetime.utcnow() + timedelta(days=frequency_days)
        )

        self.schedules[schedule.schedule_id] = schedule

        self._log_audit("SCHEDULE_CREATED", schedule.schedule_id, {
            "name": name,
            "frequency_days": frequency_days
        })

        logger.info(f"Review schedule created: {name}")
        return schedule

    def get_due_schedules(self) -> List[ReviewSchedule]:
        """Get schedules due for execution."""
        now = datetime.utcnow()
        due = [
            s for s in self.schedules.values()
            if s.is_active and s.next_run and s.next_run <= now
        ]
        return due

    def execute_schedule(
        self,
        schedule_id: str,
        created_by: str = ""
    ) -> ReviewSession:
        """
        Execute a scheduled review.

        Args:
            schedule_id: Schedule identifier
            created_by: Person executing

        Returns:
            Created ReviewSession
        """
        if schedule_id not in self.schedules:
            raise ValueError(f"Schedule not found: {schedule_id}")

        schedule = self.schedules[schedule_id]

        # Create session
        session = self.create_review_session(
            title=f"{schedule.name} - {datetime.utcnow().strftime('%Y-%m-%d')}",
            review_type=schedule.review_type,
            facilitator=schedule.default_facilitator,
            created_by=created_by
        )

        # Add attendees
        for attendee in schedule.required_attendees:
            session.attendees.append(Attendee(
                name=attendee,
                is_required=True
            ))

        # Add risks based on filter
        if schedule.scope_filter and self.risk_register:
            self.add_risks_by_filter(
                session.session_id,
                category=schedule.scope_filter.get("category"),
                risk_level=schedule.scope_filter.get("risk_level"),
                status=schedule.scope_filter.get("status")
            )

        # Update schedule
        schedule.last_run = datetime.utcnow()
        schedule.next_run = datetime.utcnow() + timedelta(days=schedule.frequency_days)

        logger.info(f"Schedule {schedule_id} executed, created session {session.session_id}")
        return session

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_pending_sessions(self) -> List[ReviewSession]:
        """Get sessions awaiting action."""
        return [
            s for s in self.sessions.values()
            if s.status in [
                ReviewStatus.SCHEDULED,
                ReviewStatus.PREPARATION,
                ReviewStatus.PENDING_APPROVAL
            ]
        ]

    def get_overdue_sessions(self) -> List[ReviewSession]:
        """Get overdue sessions."""
        now = datetime.utcnow()
        return [
            s for s in self.sessions.values()
            if s.due_date and s.due_date < now
            and s.status not in [ReviewStatus.COMPLETED, ReviewStatus.CANCELLED]
        ]

    def get_sessions_for_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ReviewSession]:
        """Get sessions within date range."""
        return [
            s for s in self.sessions.values()
            if start_date <= s.scheduled_date <= end_date
        ]

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_session_report(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive session report.

        Args:
            session_id: Session identifier

        Returns:
            Report dictionary
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        # Reassessment summary
        reassessment_summary = {
            "total": len(session.reassessments),
            "by_decision": {}
        }
        for decision in ReassessmentDecision:
            count = sum(
                1 for r in session.reassessments
                if r.decision == decision
            )
            if count > 0:
                reassessment_summary["by_decision"][decision.value] = count

        # Checklist progress
        checklist_progress = None
        if session.checklist:
            checklist_progress = self.get_checklist_progress(session_id)

        return {
            "session_id": session.session_id,
            "title": session.title,
            "review_type": session.review_type.value,
            "status": session.status.value,
            "scheduled_date": session.scheduled_date.isoformat(),
            "actual_dates": {
                "start": session.actual_start_date.isoformat()
                if session.actual_start_date else None,
                "end": session.actual_end_date.isoformat()
                if session.actual_end_date else None
            },
            "scope": {
                "description": session.scope,
                "facility": session.facility,
                "risk_count": len(session.risk_ids)
            },
            "participants": {
                "facilitator": session.facilitator,
                "attendee_count": len(session.attendees)
            },
            "reassessment_summary": reassessment_summary,
            "checklist_progress": checklist_progress,
            "meetings_count": len(session.meetings),
            "outcome": session.outcome.value if session.outcome else None,
            "summary": session.summary,
            "new_risks_identified": len(session.new_risks_identified),
            "actions_generated": len(session.actions_generated),
            "approval": {
                "required": session.requires_approval,
                "approved_by": session.approved_by,
                "approved_at": session.approved_at.isoformat()
                if session.approved_at else None
            },
            "provenance_hash": session.provenance_hash,
            "generated_at": datetime.utcnow().isoformat()
        }

    def export_to_json(self, session_id: str) -> str:
        """Export session to JSON format."""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        return json.dumps(session.model_dump(), indent=2, default=str)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_template_for_type(
        self,
        review_type: ReviewType
    ) -> Optional[ReviewChecklist]:
        """Get checklist template for review type."""
        for template in self.templates.values():
            if template.review_type == review_type:
                # Return a copy with fresh item IDs
                return ReviewChecklist(
                    name=template.name,
                    review_type=template.review_type,
                    version=template.version,
                    items=[
                        ChecklistItem(
                            category=item.category,
                            description=item.description,
                            is_mandatory=item.is_mandatory
                        )
                        for item in template.items
                    ]
                )
        return None

    def _calculate_provenance(self, session: ReviewSession) -> str:
        """Calculate SHA-256 provenance hash for session."""
        data_str = (
            f"{session.session_id}|"
            f"{session.title}|"
            f"{session.status.value}|"
            f"{len(session.reassessments)}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _log_audit(
        self,
        event_type: str,
        entity_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log event to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "entity_id": entity_id,
            "details": details
        })


if __name__ == "__main__":
    # Example usage
    print("RiskReview module loaded successfully")

    # Create scheduler
    scheduler = ReviewScheduler()

    # Create a quarterly review session
    session = scheduler.create_review_session(
        title="Q4 2024 Risk Review",
        review_type=ReviewType.QUARTERLY,
        scope="All critical and high risks",
        facility="Plant A",
        facilitator="John Smith"
    )

    print(f"Created session: {session.session_id}")
    print(f"Checklist items: {len(session.checklist.items) if session.checklist else 0}")

    # Get report
    report = scheduler.generate_session_report(session.session_id)
    print(f"Status: {report['status']}")
