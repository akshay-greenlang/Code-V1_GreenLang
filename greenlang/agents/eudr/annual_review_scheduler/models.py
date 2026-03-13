# -*- coding: utf-8 -*-
"""
Annual Review Scheduler Agent Models - AGENT-EUDR-034

Pydantic v2 models for review cycle management, deadline tracking,
checklist generation, entity coordination, year-over-year comparison,
calendar management, and notification dispatch.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 Annual Review Scheduler Agent (GL-EUDR-ARS-034)
Regulation: EU 2023/1115 (EUDR) Articles 8, 10, 11, 12, 14, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (14)
# ---------------------------------------------------------------------------


class ReviewCycleStatus(str, enum.Enum):
    """Lifecycle status of an annual review cycle."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"
    PAUSED = "paused"


class DeadlineStatus(str, enum.Enum):
    """Status of a regulatory or internal deadline."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OVERDUE = "overdue"
    COMPLETED = "completed"
    WAIVED = "waived"


class DeadlineType(str, enum.Enum):
    """Type of deadline being tracked."""
    REGULATORY_SUBMISSION = "regulatory_submission"
    INTERNAL_REVIEW = "internal_review"
    DUE_DILIGENCE_RENEWAL = "due_diligence_renewal"
    RISK_ASSESSMENT_UPDATE = "risk_assessment_update"
    DOCUMENTATION_UPDATE = "documentation_update"
    AUDIT_COMPLETION = "audit_completion"
    STAKEHOLDER_REPORTING = "stakeholder_reporting"


class ChecklistItemStatus(str, enum.Enum):
    """Completion status of a checklist item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ChecklistPriority(str, enum.Enum):
    """Priority level for checklist items."""
    MANDATORY = "mandatory"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class EntityType(str, enum.Enum):
    """Type of entity participating in a review."""
    OPERATOR = "operator"
    SUBSIDIARY = "subsidiary"
    SUPPLIER = "supplier"
    TRADER = "trader"
    MONITORING_ORGANIZATION = "monitoring_organization"
    COMPETENT_AUTHORITY = "competent_authority"


class EntityReviewStatus(str, enum.Enum):
    """Review status for an individual entity."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AWAITING_INPUT = "awaiting_input"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class ChangeSignificance(str, enum.Enum):
    """Significance level of a year-over-year change."""
    MINOR = "minor"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class ChangeDirection(str, enum.Enum):
    """Direction of a year-over-year change."""
    IMPROVED = "improved"
    STABLE = "stable"
    DEGRADED = "degraded"


class CalendarEventType(str, enum.Enum):
    """Type of calendar event."""
    REVIEW_START = "review_start"
    REVIEW_DEADLINE = "review_deadline"
    SUBMISSION_WINDOW = "submission_window"
    REGULATORY_DEADLINE = "regulatory_deadline"
    MILESTONE = "milestone"
    REMINDER = "reminder"
    MEETING = "meeting"


class NotificationChannel(str, enum.Enum):
    """Notification delivery channel."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    IN_APP = "in_app"


class NotificationStatus(str, enum.Enum):
    """Delivery status of a notification."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EscalationLevel(str, enum.Enum):
    """Escalation tier for overdue notifications."""
    REVIEWER = "reviewer"
    MANAGER = "manager"
    DIRECTOR = "director"
    COMPLIANCE_OFFICER = "compliance_officer"


class AuditAction(str, enum.Enum):
    """Audit trail action types for annual review events."""
    CREATE_CYCLE = "create_cycle"
    SCHEDULE_TASK = "schedule_task"
    REGISTER_DEADLINE = "register_deadline"
    GENERATE_CHECKLIST = "generate_checklist"
    COORDINATE_ENTITY = "coordinate_entity"
    COMPARE_YEARS = "compare_years"
    ADD_CALENDAR_EVENT = "add_calendar_event"
    SEND_NOTIFICATION = "send_notification"
    ESCALATE = "escalate"
    COMPLETE_REVIEW = "complete_review"
    SUBMIT_TO_AUTHORITY = "submit_to_authority"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-ARS-034"
AGENT_VERSION = "1.0.0"

EUDR_ARTICLES_APPLICABLE: List[str] = [
    "Article 8", "Article 10", "Article 11",
    "Article 12", "Article 14", "Article 29", "Article 31",
]

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

SUPPORTED_COMMODITIES: List[str] = EUDR_COMMODITIES


# ---------------------------------------------------------------------------
# Engine-specific Enums (used by engine tests)
# ---------------------------------------------------------------------------


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class ReviewType(str, enum.Enum):
    """Type of review cycle."""
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    AD_HOC = "ad_hoc"
    TRIGGERED = "triggered"


class ReviewPhase(str, enum.Enum):
    """Phase within a review cycle."""
    PREPARATION = "preparation"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    REVIEW_MEETING = "review_meeting"
    REMEDIATION = "remediation"
    SIGN_OFF = "sign_off"


REVIEW_PHASES_ORDER: List["ReviewPhase"] = [
    ReviewPhase.PREPARATION,
    ReviewPhase.DATA_COLLECTION,
    ReviewPhase.ANALYSIS,
    ReviewPhase.REVIEW_MEETING,
    ReviewPhase.REMEDIATION,
    ReviewPhase.SIGN_OFF,
]


class DeadlineAlertLevel(str, enum.Enum):
    """Severity level of a deadline alert."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ESCALATION = "escalation"


class EntityRole(str, enum.Enum):
    """Role of an entity in a review cycle."""
    LEAD = "lead"
    REVIEWER = "reviewer"
    ANALYST = "analyst"
    APPROVER = "approver"
    CONTRIBUTOR = "contributor"
    OBSERVER = "observer"
    EXTERNAL_AUDITOR = "external_auditor"


class EntityStatus(str, enum.Enum):
    """Status of an entity within a review."""
    ACTIVE = "active"
    INVITED = "invited"
    DECLINED = "declined"
    INACTIVE = "inactive"
    REMOVED = "removed"


class CalendarEntryType(str, enum.Enum):
    """Type of calendar entry."""
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    DEADLINE = "deadline"
    REVIEW_MEETING = "review_meeting"
    MILESTONE = "milestone"
    REMINDER = "reminder"
    SIGN_OFF = "sign_off"


class NotificationPriority(str, enum.Enum):
    """Priority level for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class YearComparisonStatus(str, enum.Enum):
    """Status of a year-over-year comparison."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonDimension(str, enum.Enum):
    """Dimension for year-over-year comparison metrics."""
    COMPLIANCE_RATE = "compliance_rate"
    RISK_SCORE = "risk_score"
    SUPPLIER_COUNT = "supplier_count"
    DEFORESTATION_RATE = "deforestation_rate"
    AUDIT_FINDINGS = "audit_findings"
    DDS_APPROVAL_RATE = "dds_approval_rate"


# ---------------------------------------------------------------------------
# Engine-specific Models (used by engine tests)
# ---------------------------------------------------------------------------


class CommodityScope(BaseModel):
    """Commodity scope within a review cycle."""
    commodity: EUDRCommodity = Field(..., description="EUDR commodity")
    supplier_count: int = Field(default=0, ge=0)
    shipment_count: int = Field(default=0, ge=0)

    model_config = {"frozen": False, "extra": "ignore"}


class ReviewPhaseConfig(BaseModel):
    """Configuration for a single review phase."""
    phase: ReviewPhase = Field(..., description="Review phase")
    duration_days: int = Field(default=30, ge=1)
    required_checklist_items: int = Field(default=0, ge=0)
    auto_advance: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "ignore"}


class ReviewCycle(BaseModel):
    """Full review cycle model used by the ReviewCycleManager engine."""
    cycle_id: str = Field(..., description="Unique cycle identifier")
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(..., description="Year under review")
    review_type: ReviewType = ReviewType.ANNUAL
    commodity_scope: List[CommodityScope] = Field(default_factory=list)
    status: ReviewCycleStatus = ReviewCycleStatus.DRAFT
    current_phase: ReviewPhase = ReviewPhase.PREPARATION
    phase_configs: List[ReviewPhaseConfig] = Field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    created_by: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DeadlineTrack(BaseModel):
    """Deadline tracking model used by the DeadlineTracker engine."""
    deadline_id: str = Field(..., description="Unique deadline identifier")
    cycle_id: str = Field(..., description="Associated review cycle ID")
    phase: ReviewPhase = Field(..., description="Review phase")
    description: str = Field(default="", description="Deadline description")
    due_date: datetime = Field(..., description="Due date")
    status: DeadlineStatus = DeadlineStatus.ON_TRACK
    assigned_entity_id: Optional[str] = None
    warning_days_before: int = Field(default=7, ge=0)
    critical_days_before: int = Field(default=3, ge=0)
    completed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DeadlineAlert(BaseModel):
    """Alert raised for an approaching or overdue deadline."""
    alert_id: str = Field(..., description="Unique alert identifier")
    deadline_id: str = Field(..., description="Associated deadline ID")
    cycle_id: str = Field(..., description="Associated cycle ID")
    alert_level: DeadlineAlertLevel = DeadlineAlertLevel.INFO
    message: str = Field(default="", description="Alert message")
    days_remaining: int = Field(default=0, description="Days remaining")
    acknowledged: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "ignore"}


class ChecklistTemplate(BaseModel):
    """Template for generating checklist items."""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    phase: ReviewPhase = Field(..., description="Target review phase")
    commodity: Optional[EUDRCommodity] = None
    items: List["ChecklistItem"] = Field(default_factory=list)
    version: str = Field(default="1.0.0")
    regulatory_reference: str = Field(default="")

    model_config = {"frozen": False, "extra": "ignore"}


class EntityCoordination(BaseModel):
    """Entity coordination record used by the EntityCoordinator engine."""
    entity_id: str = Field(..., description="Unique entity identifier")
    cycle_id: str = Field(..., description="Associated review cycle ID")
    name: str = Field(..., description="Entity display name")
    role: EntityRole = EntityRole.CONTRIBUTOR
    email: str = Field(..., description="Contact email")
    status: EntityStatus = EntityStatus.ACTIVE
    assigned_phases: List[ReviewPhase] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "ignore"}


class EntityDependency(BaseModel):
    """Dependency between two entities in a review cycle."""
    dependency_id: str = Field(..., description="Unique dependency identifier")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    dependency_type: str = Field(..., description="Type of dependency")
    phase: ReviewPhase = Field(..., description="Applicable phase")
    description: str = Field(default="")
    resolved: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "ignore"}


class YearMetricSnapshot(BaseModel):
    """Snapshot of key metrics for a single year and commodity."""
    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    operator_id: str = Field(..., description="Operator identifier")
    year: int = Field(..., description="Calendar year")
    commodity: EUDRCommodity = Field(..., description="EUDR commodity")
    total_suppliers: int = Field(default=0, ge=0)
    compliant_suppliers: int = Field(default=0, ge=0)
    compliance_rate: Decimal = Field(default=Decimal("0"))
    average_risk_score: Decimal = Field(default=Decimal("0"))
    total_shipments: int = Field(default=0, ge=0)
    deforestation_free_rate: Decimal = Field(default=Decimal("0"))
    dds_submitted: int = Field(default=0, ge=0)
    dds_approved: int = Field(default=0, ge=0)
    audit_findings: int = Field(default=0, ge=0)
    remediation_actions: int = Field(default=0, ge=0)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ComparisonMetric(BaseModel):
    """A single metric comparison between two years."""
    dimension: ComparisonDimension = Field(..., description="Comparison dimension")
    base_value: Decimal = Field(default=Decimal("0"))
    compare_value: Decimal = Field(default=Decimal("0"))
    change: Decimal = Field(default=Decimal("0"))
    percentage_change: Decimal = Field(default=Decimal("0"))

    model_config = {"frozen": False, "extra": "ignore"}


class ComparisonResult(BaseModel):
    """Result of a year-over-year comparison."""
    comparison_id: str = Field(..., description="Unique comparison identifier")
    overall_trend: str = Field(default="", description="Overall trend direction")
    metrics: List[ComparisonMetric] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "ignore"}


class CalendarEntry(BaseModel):
    """Calendar entry used by the CalendarManager engine."""
    entry_id: str = Field(..., description="Unique entry identifier")
    cycle_id: str = Field(..., description="Associated review cycle ID")
    entry_type: CalendarEntryType = CalendarEntryType.REMINDER
    title: str = Field(..., description="Entry title")
    description: str = Field(default="")
    start_time: datetime = Field(..., description="Start time")
    end_time: Optional[datetime] = None
    phase: Optional[ReviewPhase] = None
    attendees: List[str] = Field(default_factory=list)
    location: str = Field(default="")
    recurring: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "ignore"}


class NotificationTemplate(BaseModel):
    """Template for generating notifications."""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    channel: NotificationChannel = NotificationChannel.EMAIL
    subject_template: str = Field(..., description="Subject template string")
    body_template: str = Field(..., description="Body template string")
    priority: NotificationPriority = NotificationPriority.NORMAL
    trigger_event: str = Field(default="")
    days_before: int = Field(default=0, ge=0)

    model_config = {"frozen": False, "extra": "ignore"}


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ReviewTask(BaseModel):
    """A task within a review cycle."""
    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(default="", description="Task title")
    description: str = Field(default="", description="Task description")
    assignee: Optional[str] = Field(None, description="Assigned user or role")
    status: ChecklistItemStatus = ChecklistItemStatus.PENDING
    priority: ChecklistPriority = ChecklistPriority.MEDIUM
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list, description="Dependent task IDs")

    model_config = {"frozen": False, "extra": "ignore"}


class ChecklistItem(BaseModel):
    """A single item in a review checklist."""
    item_id: str = Field(..., description="Unique item identifier")
    cycle_id: str = Field(default="", description="Associated review cycle ID")
    phase: Optional["ReviewPhase"] = Field(None, description="Review phase")
    section: str = Field(default="general", description="Checklist section")
    article_reference: str = Field(default="", description="EUDR article reference")
    title: str = Field(default="", description="Item title")
    description: str = Field(default="", description="Detailed description")
    status: ChecklistItemStatus = ChecklistItemStatus.PENDING
    assigned_to: Optional[str] = Field(None, description="Assigned user email")
    priority: Any = Field(default=0, description="Priority (enum or int)")
    is_mandatory: bool = Field(default=False, description="Whether item is mandatory")
    required: bool = Field(default=False, description="Whether item is required")
    evidence_required: bool = Field(default=False, description="Whether evidence upload is required")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    completed_by: Optional[str] = Field(None, description="Completed by user")
    notes: str = Field(default="", description="Reviewer notes")

    model_config = {"frozen": False, "extra": "ignore"}


class DeadlineEntry(BaseModel):
    """A tracked deadline within the review system."""
    deadline_id: str = Field(..., description="Unique deadline identifier")
    deadline_type: DeadlineType = DeadlineType.REGULATORY_SUBMISSION
    title: str = Field(default="", description="Deadline title")
    due_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: DeadlineStatus = DeadlineStatus.ON_TRACK
    days_remaining: int = Field(default=0, description="Days until deadline")
    responsible_entity: Optional[str] = None
    article_reference: str = Field(default="", description="EUDR article reference")

    model_config = {"frozen": False, "extra": "ignore"}


class EntityReviewInfo(BaseModel):
    """Review information for a coordinated entity."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: EntityType = EntityType.OPERATOR
    entity_name: str = Field(default="", description="Entity display name")
    review_status: EntityReviewStatus = EntityReviewStatus.NOT_STARTED
    completion_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    assigned_reviewer: Optional[str] = None
    parent_entity_id: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "ignore"}


class YearDataPoint(BaseModel):
    """Data point for a single year in a comparison."""
    year: int = Field(..., description="Calendar year")
    supplier_count: int = Field(default=0, ge=0)
    risk_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    compliance_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    deforestation_alerts: int = Field(default=0, ge=0)
    due_diligence_statements: int = Field(default=0, ge=0)
    total_volume_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    high_risk_suppliers: int = Field(default=0, ge=0)

    model_config = {"frozen": False, "extra": "ignore"}


class YearComparison(BaseModel):
    """Full year-over-year comparison result used by engine tests."""
    comparison_id: str = Field(default="", description="Unique comparison identifier")
    operator_id: str = Field(default="", description="Operator identifier")
    commodity: Optional["EUDRCommodity"] = Field(None, description="EUDR commodity")
    base_year: int = Field(default=0, description="Base year")
    compare_year: int = Field(default=0, description="Comparison year")
    base_snapshot: Optional["YearMetricSnapshot"] = Field(None, description="Base year snapshot")
    compare_snapshot: Optional["YearMetricSnapshot"] = Field(None, description="Compare year snapshot")
    status: "YearComparisonStatus" = Field(default=YearComparisonStatus.PENDING, description="Comparison status")
    metrics: List["ComparisonMetric"] = Field(default_factory=list, description="Comparison metrics")
    overall_trend: str = Field(default="", description="Overall trend direction")
    provenance_hash: str = Field(default="", description="Provenance hash")

    model_config = {"frozen": False, "extra": "ignore"}


class YearDimensionComparison(BaseModel):
    """Comparison between two years for a single dimension (original model)."""
    dimension: str = Field(..., description="Comparison dimension name")
    year_a: int = Field(..., description="Earlier year")
    year_b: int = Field(..., description="Later year")
    value_a: Decimal = Field(default=Decimal("0"), description="Value in year A")
    value_b: Decimal = Field(default=Decimal("0"), description="Value in year B")
    absolute_change: Decimal = Field(default=Decimal("0"), description="Absolute change")
    percent_change: Decimal = Field(default=Decimal("0"), description="Percentage change")
    significance: ChangeSignificance = ChangeSignificance.MINOR
    direction: ChangeDirection = ChangeDirection.STABLE

    model_config = {"frozen": False, "extra": "ignore"}


class CalendarEvent(BaseModel):
    """An event on the compliance calendar."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: CalendarEventType = CalendarEventType.REMINDER
    title: str = Field(default="", description="Event title")
    description: str = Field(default="", description="Event description")
    start_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_date: Optional[datetime] = None
    all_day: bool = Field(default=True, description="Whether this is an all-day event")
    recurrence_rule: Optional[str] = Field(None, description="iCal recurrence rule (RRULE)")
    operator_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "ignore"}


class NotificationRecord(BaseModel):
    """Record of a sent notification."""
    notification_id: str = Field(..., description="Unique notification identifier")
    cycle_id: str = Field(default="", description="Associated review cycle ID")
    channel: NotificationChannel = NotificationChannel.EMAIL
    priority: Optional["NotificationPriority"] = None
    recipient: str = Field(default="", description="Recipient identifier")
    subject: str = Field(default="", description="Notification subject")
    body: str = Field(default="", description="Notification body")
    status: NotificationStatus = NotificationStatus.PENDING
    escalation_level: EscalationLevel = EscalationLevel.REVIEWER
    template_id: Optional[str] = None
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)

    model_config = {"frozen": False, "extra": "ignore"}


class ActionRecommendation(BaseModel):
    """Recommended action from review analysis."""
    action: str = Field(..., description="Action description")
    priority: str = Field(default="medium", description="Priority (low/medium/high/critical)")
    deadline_days: int = Field(default=30, ge=1, description="Suggested deadline in days")
    category: str = Field(default="general", description="Action category")

    model_config = {"frozen": False, "extra": "ignore"}


# ---------------------------------------------------------------------------
# Core Models (15+)
# ---------------------------------------------------------------------------


class ReviewCycleRecord(BaseModel):
    """Annual review cycle record.

    Represents a complete annual review cycle for an operator,
    including tasks, deadlines, and completion tracking.
    """
    cycle_id: str = Field(..., description="Unique cycle identifier")
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(..., description="Year under review")
    cycle_status: ReviewCycleStatus = ReviewCycleStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    grace_deadline: Optional[datetime] = None
    tasks: List[ReviewTask] = Field(default_factory=list)
    tasks_total: int = Field(default=0, ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    completion_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    commodities: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DeadlineTrackingRecord(BaseModel):
    """Regulatory and internal deadline tracking record.

    Tracks approaching deadlines, submission status, and
    escalation history for EUDR compliance deadlines.
    """
    tracking_id: str = Field(..., description="Unique tracking identifier")
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(default=0, description="Applicable review year")
    deadlines: List[DeadlineEntry] = Field(default_factory=list)
    total_deadlines: int = Field(default=0, ge=0)
    approaching_count: int = Field(default=0, ge=0)
    overdue_count: int = Field(default=0, ge=0)
    met_count: int = Field(default=0, ge=0)
    submissions_pending: int = Field(default=0, ge=0)
    next_deadline: Optional[datetime] = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ChecklistRecord(BaseModel):
    """Generated review checklist record.

    Contains commodity-specific checklist items derived from
    EUDR article requirements and organizational templates.
    """
    checklist_id: str = Field(..., description="Unique checklist identifier")
    operator_id: str = Field(..., description="Operator identifier")
    cycle_id: str = Field(default="", description="Associated review cycle ID")
    commodity: str = Field(default="general", description="Commodity type")
    template_version: str = Field(default="", description="Template version used")
    items: List[ChecklistItem] = Field(default_factory=list)
    total_items: int = Field(default=0, ge=0)
    completed_items: int = Field(default=0, ge=0)
    mandatory_items: int = Field(default=0, ge=0)
    mandatory_completed: int = Field(default=0, ge=0)
    completion_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    mandatory_completion_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class EntityCoordinationRecord(BaseModel):
    """Entity coordination record for multi-entity reviews.

    Tracks review progress across organizational entities,
    managing dependencies and cascading review requirements.
    """
    coordination_id: str = Field(..., description="Unique coordination identifier")
    operator_id: str = Field(..., description="Root operator identifier")
    cycle_id: str = Field(default="", description="Associated review cycle ID")
    entities: List[EntityReviewInfo] = Field(default_factory=list)
    total_entities: int = Field(default=0, ge=0)
    completed_entities: int = Field(default=0, ge=0)
    cascade_depth: int = Field(default=0, ge=0)
    overall_completion: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    blocked_entities: int = Field(default=0, ge=0)
    escalated_entities: int = Field(default=0, ge=0)
    coordinated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class YearComparisonRecord(BaseModel):
    """Year-over-year comparison record.

    Compares key EUDR compliance metrics across multiple years
    to identify trends, regressions, and areas of improvement.
    """
    comparison_id: str = Field(..., description="Unique comparison identifier")
    operator_id: str = Field(..., description="Operator identifier")
    years_compared: List[int] = Field(default_factory=list)
    data_points: List[YearDataPoint] = Field(default_factory=list)
    comparisons: List[YearDimensionComparison] = Field(default_factory=list)
    overall_trend: ChangeDirection = ChangeDirection.STABLE
    overall_significance: ChangeSignificance = ChangeSignificance.MINOR
    weighted_change_score: Decimal = Field(default=Decimal("0"))
    critical_changes: int = Field(default=0, ge=0)
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CalendarRecord(BaseModel):
    """Compliance calendar record.

    Manages calendar events for review deadlines, milestones,
    regulatory submissions, and reminders.
    """
    calendar_id: str = Field(..., description="Unique calendar record identifier")
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(default=0, description="Applicable review year")
    events: List[CalendarEvent] = Field(default_factory=list)
    total_events: int = Field(default=0, ge=0)
    upcoming_events: int = Field(default=0, ge=0)
    overdue_events: int = Field(default=0, ge=0)
    ical_data: Optional[str] = Field(None, description="Generated iCal data")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class NotificationBatchRecord(BaseModel):
    """Notification batch dispatch record.

    Tracks batch notification sends, acknowledgments,
    failures, and escalation triggers.
    """
    batch_id: str = Field(..., description="Unique batch identifier")
    operator_id: str = Field(..., description="Operator identifier")
    cycle_id: str = Field(default="", description="Associated review cycle ID")
    notifications: List[NotificationRecord] = Field(default_factory=list)
    total_sent: int = Field(default=0, ge=0)
    total_delivered: int = Field(default=0, ge=0)
    total_acknowledged: int = Field(default=0, ge=0)
    total_failed: int = Field(default=0, ge=0)
    escalations_triggered: int = Field(default=0, ge=0)
    channels_used: List[str] = Field(default_factory=list)
    dispatched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ReviewSummary(BaseModel):
    """Annual review summary across all engines."""
    summary_id: str = Field(..., description="Unique summary identifier")
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(default=0, description="Review year")
    cycle_status: ReviewCycleStatus = ReviewCycleStatus.DRAFT
    overall_completion: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    total_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    deadlines_met: int = Field(default=0, ge=0)
    deadlines_overdue: int = Field(default=0, ge=0)
    checklists_completed: int = Field(default=0, ge=0)
    entities_coordinated: int = Field(default=0, ge=0)
    yoy_trend: ChangeDirection = ChangeDirection.STABLE
    notifications_sent: int = Field(default=0, ge=0)
    calendar_events: int = Field(default=0, ge=0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class AuditEntry(BaseModel):
    """An audit trail entry for annual review events."""
    entry_id: str = Field(..., description="Unique audit entry identifier")
    entity_type: str = Field(..., description="Entity type being audited")
    entity_id: str = Field(..., description="Entity identifier")
    action: AuditAction = AuditAction.CREATE_CYCLE
    actor: str = Field(..., description="Actor performing the action")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the Annual Review Scheduler Agent."""
    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
