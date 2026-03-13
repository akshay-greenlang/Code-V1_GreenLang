# -*- coding: utf-8 -*-
"""
Improvement Plan Creator Models - AGENT-EUDR-035

Pydantic v2 models for improvement plan creation, finding aggregation,
gap analysis, SMART action generation, root cause analysis (5-Whys and
fishbone), Eisenhower + risk-based prioritization, progress tracking,
stakeholder coordination (RACI), and plan reporting.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 Improvement Plan Creator (GL-EUDR-IPC-035)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (12)
# ---------------------------------------------------------------------------


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities (Article 1)."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevel(str, enum.Enum):
    """Risk classification levels per EUDR Article 10(2)."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class GapSeverity(str, enum.Enum):
    """Severity classification for compliance gaps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ActionStatus(str, enum.Enum):
    """Lifecycle status of an improvement action."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ActionType(str, enum.Enum):
    """Types of improvement actions aligned to EUDR Article 11."""

    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"
    DOCUMENTATION_UPDATE = "documentation_update"
    TRAINING = "training"
    PROCESS_CHANGE = "process_change"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    AUDIT_ENHANCEMENT = "audit_enhancement"
    POLICY_UPDATE = "policy_update"


class PlanStatus(str, enum.Enum):
    """Lifecycle status of an improvement plan."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"


class EisenhowerQuadrant(str, enum.Enum):
    """Eisenhower matrix quadrant classification."""

    DO_FIRST = "do_first"
    SCHEDULE = "schedule"
    DELEGATE = "delegate"
    ELIMINATE = "eliminate"


class RACIRole(str, enum.Enum):
    """RACI responsibility assignment roles."""

    RESPONSIBLE = "responsible"
    ACCOUNTABLE = "accountable"
    CONSULTED = "consulted"
    INFORMED = "informed"


class FindingSource(str, enum.Enum):
    """Source agent that generated a finding."""

    RISK_ASSESSMENT = "risk_assessment"
    COUNTRY_RISK = "country_risk"
    SUPPLIER_RISK = "supplier_risk"
    COMMODITY_RISK = "commodity_risk"
    DEFORESTATION_ALERT = "deforestation_alert"
    LEGAL_COMPLIANCE = "legal_compliance"
    DOCUMENT_AUTHENTICATION = "document_authentication"
    SATELLITE_MONITORING = "satellite_monitoring"
    MITIGATION_MEASURE = "mitigation_measure"
    AUDIT_MANAGER = "audit_manager"
    MANUAL = "manual"


class FishboneCategory(str, enum.Enum):
    """Ishikawa (fishbone) diagram root cause categories."""

    PEOPLE = "people"
    PROCESS = "process"
    TECHNOLOGY = "technology"
    DATA = "data"
    POLICY = "policy"
    ENVIRONMENT = "environment"
    SUPPLIERS = "suppliers"
    MANAGEMENT = "management"


class NotificationChannel(str, enum.Enum):
    """Notification delivery channel types."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SMS = "sms"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-IPC-035"
AGENT_VERSION = "1.0.0"

DEFAULT_PRIORITY_WEIGHTS: Dict[str, Decimal] = {
    "risk_score": Decimal("0.30"),
    "compliance_impact": Decimal("0.25"),
    "resource_efficiency": Decimal("0.20"),
    "stakeholder_impact": Decimal("0.15"),
    "time_sensitivity": Decimal("0.10"),
}

GAP_SEVERITY_THRESHOLDS: Dict[GapSeverity, Decimal] = {
    GapSeverity.CRITICAL: Decimal("0.80"),
    GapSeverity.HIGH: Decimal("0.60"),
    GapSeverity.MEDIUM: Decimal("0.40"),
    GapSeverity.LOW: Decimal("0.20"),
    GapSeverity.INFORMATIONAL: Decimal("0.00"),
}

SUPPORTED_COMMODITIES: List[str] = [c.value for c in EUDRCommodity]


# ---------------------------------------------------------------------------
# Pydantic Models (15+)
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A finding from an upstream EUDR agent.

    Represents a discrete compliance observation, risk signal, or
    non-conformity detected by one of the EUDR monitoring agents
    (EUDR-016 through EUDR-034).
    """

    finding_id: str = Field(..., description="Unique finding identifier")
    source: FindingSource = Field(
        ..., description="Agent that generated the finding"
    )
    source_agent_id: str = Field(
        default="", description="Source agent identifier (e.g. GL-EUDR-RAE-028)"
    )
    title: str = Field(..., description="Finding title")
    description: str = Field(default="", description="Detailed finding text")
    severity: GapSeverity = Field(
        default=GapSeverity.MEDIUM, description="Finding severity"
    )
    risk_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Associated risk score 0-100",
    )
    commodity: Optional[EUDRCommodity] = Field(
        default=None, description="Affected EUDR commodity"
    )
    operator_id: str = Field(default="", description="Operator under review")
    country_code: str = Field(
        default="", description="ISO 3166-1 alpha-2 country code"
    )
    supplier_id: str = Field(default="", description="Supplier identifier")
    eudr_article_ref: str = Field(
        default="", description="EUDR article reference"
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    evidence_ids: List[str] = Field(
        default_factory=list, description="Supporting evidence references"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional source-specific metadata"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class AggregatedFindings(BaseModel):
    """Consolidated findings from multiple agents.

    Groups, deduplicates, and categorizes findings to provide a unified
    view of compliance status across all EUDR monitoring dimensions.
    """

    aggregation_id: str = Field(
        ..., description="Unique aggregation identifier"
    )
    operator_id: str = Field(..., description="Operator under review")
    findings: List[Finding] = Field(
        default_factory=list, description="Deduplicated findings"
    )
    total_findings: int = Field(default=0, description="Total finding count")
    critical_count: int = Field(default=0, description="Critical findings")
    high_count: int = Field(default=0, description="High-severity findings")
    medium_count: int = Field(default=0, description="Medium-severity findings")
    low_count: int = Field(default=0, description="Low-severity findings")
    source_agents: List[str] = Field(
        default_factory=list, description="Contributing source agents"
    )
    duplicates_removed: int = Field(
        default=0, description="Duplicate findings removed"
    )
    aggregated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ComplianceGap(BaseModel):
    """A compliance gap identified through gap analysis.

    Represents the delta between current compliance state and EUDR
    requirements, with mapped regulatory references and recommended
    remediation approach.
    """

    gap_id: str = Field(..., description="Unique gap identifier")
    plan_id: str = Field(default="", description="Parent improvement plan ID")
    title: str = Field(..., description="Gap title")
    description: str = Field(default="", description="Detailed gap description")
    severity: GapSeverity = Field(
        default=GapSeverity.MEDIUM, description="Gap severity"
    )
    severity_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=1,
        description="Normalized severity score 0-1",
    )
    eudr_article_ref: str = Field(
        default="", description="EUDR article reference"
    )
    eudr_requirement: str = Field(
        default="", description="Specific EUDR requirement text"
    )
    current_state: str = Field(
        default="", description="Current compliance state"
    )
    required_state: str = Field(
        default="", description="Required compliance state"
    )
    finding_ids: List[str] = Field(
        default_factory=list, description="Related finding IDs"
    )
    commodity: Optional[EUDRCommodity] = None
    risk_dimension: str = Field(
        default="", description="Affected risk dimension"
    )
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RootCause(BaseModel):
    """Root cause identified via 5-Whys or fishbone analysis.

    Captures the underlying systemic cause of a compliance gap with
    analysis chain and confidence scoring.
    """

    root_cause_id: str = Field(
        ..., description="Unique root cause identifier"
    )
    gap_id: str = Field(..., description="Associated gap identifier")
    category: FishboneCategory = Field(
        default=FishboneCategory.PROCESS,
        description="Fishbone/Ishikawa category",
    )
    description: str = Field(
        ..., description="Root cause description"
    )
    analysis_chain: List[str] = Field(
        default_factory=list,
        description="5-Whys chain from symptom to root cause",
    )
    depth: int = Field(
        default=1, ge=1, le=10,
        description="Depth in 5-Whys analysis (1=surface, 5=root)",
    )
    confidence: Decimal = Field(
        default=Decimal("0.50"), ge=0, le=1,
        description="Confidence in root cause identification",
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Contributing factors to this root cause",
    )
    systemic: bool = Field(
        default=False,
        description="Whether this is a systemic (cross-cutting) root cause",
    )
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class FishboneAnalysis(BaseModel):
    """Complete fishbone (Ishikawa) diagram analysis for a gap.

    Organizes root causes by category to identify systemic patterns
    across people, process, technology, data, policy, environment,
    suppliers, and management dimensions.
    """

    analysis_id: str = Field(..., description="Unique analysis identifier")
    gap_id: str = Field(..., description="Analyzed gap identifier")
    categories: Dict[str, List[RootCause]] = Field(
        default_factory=dict,
        description="Root causes organized by fishbone category",
    )
    primary_root_cause_id: Optional[str] = Field(
        default=None, description="Primary root cause identifier"
    )
    systemic_causes: List[str] = Field(
        default_factory=list,
        description="Cross-cutting systemic root cause IDs",
    )
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ImprovementAction(BaseModel):
    """A SMART improvement action within an improvement plan.

    Specific, Measurable, Achievable, Relevant, Time-bound action
    designed to close a compliance gap with tracked ownership,
    milestones, and effectiveness metrics.
    """

    action_id: str = Field(..., description="Unique action identifier")
    plan_id: str = Field(..., description="Parent improvement plan ID")
    gap_id: str = Field(default="", description="Associated gap identifier")
    root_cause_id: str = Field(
        default="", description="Addressed root cause ID"
    )
    title: str = Field(..., description="Action title")
    description: str = Field(default="", description="Detailed description")
    action_type: ActionType = Field(
        default=ActionType.CORRECTIVE, description="Action type classification"
    )
    status: ActionStatus = Field(
        default=ActionStatus.DRAFT, description="Current action status"
    )

    # SMART criteria
    specific_outcome: str = Field(
        default="", description="S - Specific expected outcome"
    )
    measurable_kpi: str = Field(
        default="", description="M - Measurable KPI or metric"
    )
    achievable_resources: str = Field(
        default="", description="A - Required resources and feasibility"
    )
    relevant_justification: str = Field(
        default="", description="R - Relevance to EUDR compliance"
    )
    time_bound_deadline: Optional[datetime] = Field(
        default=None, description="T - Target completion deadline"
    )

    # Prioritization
    priority_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Computed priority score 0-100",
    )
    eisenhower_quadrant: EisenhowerQuadrant = Field(
        default=EisenhowerQuadrant.SCHEDULE,
        description="Eisenhower matrix classification",
    )
    urgency_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Urgency score 0-100",
    )
    importance_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Importance score 0-100",
    )

    # Tracking
    assigned_to: Optional[str] = Field(
        default=None, description="Assigned responsible person"
    )
    estimated_effort_hours: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated effort in hours",
    )
    estimated_cost: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated cost in base currency",
    )
    actual_effort_hours: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Actual effort recorded",
    )
    actual_cost: Optional[Decimal] = Field(
        default=None, ge=0, description="Actual cost recorded"
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    extensions_used: int = Field(default=0, description="Extensions granted")

    # Evidence and results
    evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence references"
    )
    effectiveness_score: Optional[Decimal] = Field(
        default=None, ge=0, le=100,
        description="Post-completion effectiveness score",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class StakeholderAssignment(BaseModel):
    """RACI assignment for a stakeholder on an improvement action.

    Maps stakeholders to their responsibility roles for each action
    in the improvement plan per the RACI framework.
    """

    assignment_id: str = Field(
        ..., description="Unique assignment identifier"
    )
    action_id: str = Field(..., description="Action identifier")
    stakeholder_id: str = Field(..., description="Stakeholder identifier")
    stakeholder_name: str = Field(
        default="", description="Stakeholder display name"
    )
    stakeholder_email: str = Field(
        default="", description="Stakeholder email address"
    )
    role: RACIRole = Field(
        ..., description="RACI role for this action"
    )
    department: str = Field(
        default="", description="Department or team"
    )
    notification_channel: NotificationChannel = Field(
        default=NotificationChannel.EMAIL,
        description="Preferred notification channel",
    )
    notified_at: Optional[datetime] = Field(
        default=None, description="Last notification timestamp"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="Acknowledgment timestamp"
    )
    assigned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ProgressMilestone(BaseModel):
    """Progress milestone for tracking action completion."""

    milestone_id: str = Field(
        ..., description="Unique milestone identifier"
    )
    action_id: str = Field(..., description="Parent action identifier")
    title: str = Field(..., description="Milestone title")
    description: str = Field(default="", description="Milestone details")
    due_date: Optional[datetime] = Field(
        default=None, description="Target date"
    )
    completed_at: Optional[datetime] = None
    status: ActionStatus = Field(default=ActionStatus.PROPOSED)
    weight: Decimal = Field(
        default=Decimal("1.00"), ge=0,
        description="Relative weight for progress calculation",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ProgressSnapshot(BaseModel):
    """Point-in-time progress snapshot for an improvement plan.

    Captures completion percentages, on-track metrics, and
    effectiveness indicators at a given point in time.
    """

    snapshot_id: str = Field(
        ..., description="Unique snapshot identifier"
    )
    plan_id: str = Field(..., description="Improvement plan identifier")
    overall_progress: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Overall completion percentage",
    )
    actions_total: int = Field(default=0)
    actions_completed: int = Field(default=0)
    actions_in_progress: int = Field(default=0)
    actions_overdue: int = Field(default=0)
    actions_on_hold: int = Field(default=0)
    gaps_closed: int = Field(default=0)
    gaps_total: int = Field(default=0)
    avg_effectiveness_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
    )
    on_track: bool = Field(
        default=True, description="Whether the plan is on track"
    )
    risk_trend: str = Field(
        default="stable",
        description="Risk trend: improving/stable/worsening",
    )
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class NotificationRecord(BaseModel):
    """Record of a notification sent to a stakeholder."""

    notification_id: str = Field(
        ..., description="Unique notification identifier"
    )
    action_id: str = Field(..., description="Related action identifier")
    stakeholder_id: str = Field(
        ..., description="Recipient stakeholder ID"
    )
    channel: NotificationChannel
    subject: str = Field(default="", description="Notification subject")
    body: str = Field(default="", description="Notification body")
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    delivered: bool = Field(default=False)
    read_at: Optional[datetime] = None

    model_config = {"frozen": False, "extra": "ignore"}


class ImprovementPlan(BaseModel):
    """Top-level improvement plan aggregating all components.

    The central model that ties together findings, gaps, root causes,
    actions, stakeholders, and progress tracking into a cohesive
    improvement plan for EUDR compliance.
    """

    plan_id: str = Field(..., description="Unique plan identifier")
    operator_id: str = Field(..., description="Operator under improvement")
    title: str = Field(default="", description="Plan title")
    description: str = Field(default="", description="Plan description")
    commodity: Optional[EUDRCommodity] = None
    status: PlanStatus = Field(
        default=PlanStatus.DRAFT, description="Plan lifecycle status"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD, description="Current risk level"
    )

    # Components
    aggregation_id: str = Field(
        default="", description="Finding aggregation ID"
    )
    gaps: List[ComplianceGap] = Field(
        default_factory=list, description="Identified compliance gaps"
    )
    root_causes: List[RootCause] = Field(
        default_factory=list, description="Identified root causes"
    )
    actions: List[ImprovementAction] = Field(
        default_factory=list, description="Improvement actions"
    )
    stakeholder_assignments: List[StakeholderAssignment] = Field(
        default_factory=list, description="RACI assignments"
    )

    # Summary metrics
    total_gaps: int = Field(default=0)
    total_actions: int = Field(default=0)
    estimated_total_cost: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total estimated cost",
    )
    estimated_completion_days: int = Field(
        default=0, description="Estimated days to complete all actions"
    )

    # Timeline
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    approved_at: Optional[datetime] = None
    target_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = Field(default=AGENT_ID)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class PlanSummary(BaseModel):
    """Summary view of an improvement plan for listings."""

    plan_id: str
    operator_id: str
    title: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    commodity: Optional[EUDRCommodity] = None
    risk_level: RiskLevel = RiskLevel.STANDARD
    total_gaps: int = 0
    total_actions: int = 0
    actions_completed: int = 0
    overall_progress: Decimal = Decimal("0")
    on_track: bool = True
    created_at: Optional[datetime] = None

    model_config = {"frozen": False, "extra": "ignore"}


class PlanReport(BaseModel):
    """Comprehensive improvement plan report for audit and DDS.

    Combines plan details, gap analysis results, action summaries,
    stakeholder assignments, progress tracking, and provenance
    hashes for inclusion in EUDR Due Diligence Statements.
    """

    report_id: str = Field(..., description="Unique report identifier")
    plan_id: str = Field(..., description="Improvement plan identifier")
    operator_id: str = Field(..., description="Operator identifier")
    commodity: Optional[EUDRCommodity] = None
    plan_status: PlanStatus = PlanStatus.DRAFT
    risk_level: RiskLevel = RiskLevel.STANDARD
    gaps_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Gap count by severity",
    )
    actions_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Action count by status",
    )
    overall_progress: Decimal = Decimal("0")
    on_track: bool = True
    stakeholder_count: int = 0
    estimated_total_cost: Decimal = Decimal("0")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    regulation_reference: str = "EU 2023/1115 Articles 10, 11, 12"
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the Improvement Plan Creator."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
