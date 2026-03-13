# -*- coding: utf-8 -*-
"""
Grievance Mechanism Manager Models - AGENT-EUDR-032

Pydantic v2 models for grievance analytics, root cause analysis, mediation
workflows, remediation tracking, risk scoring, collective grievance handling,
regulatory reporting, and audit trail operations.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 Grievance Mechanism Manager (GL-EUDR-GMM-032)
Regulation: EU 2023/1115 (EUDR); CSDDD Article 8; UNGP Principle 31
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


class PatternType(str, enum.Enum):
    """Detected grievance pattern classification."""
    RECURRING = "recurring"
    CLUSTERED = "clustered"
    SYSTEMIC = "systemic"
    ISOLATED = "isolated"
    ESCALATING = "escalating"


class TrendDirection(str, enum.Enum):
    """Trend direction for grievance metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class AnalysisMethod(str, enum.Enum):
    """Root cause analysis methodology."""
    FIVE_WHYS = "five_whys"
    FISHBONE = "fishbone"
    FAULT_TREE = "fault_tree"
    CORRELATION = "correlation"


class MediationStage(str, enum.Enum):
    """Multi-party mediation workflow stages."""
    INITIATED = "initiated"
    PREPARATION = "preparation"
    DIALOGUE = "dialogue"
    NEGOTIATION = "negotiation"
    SETTLEMENT = "settlement"
    IMPLEMENTATION = "implementation"
    CLOSED = "closed"


class MediatorType(str, enum.Enum):
    """Types of mediators that can be assigned."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    COMMUNITY_ELDER = "community_elder"
    LEGAL = "legal"


class SettlementStatus(str, enum.Enum):
    """Status of a mediation settlement."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class RemediationType(str, enum.Enum):
    """Types of remediation actions."""
    COMPENSATION = "compensation"
    PROCESS_CHANGE = "process_change"
    RELATIONSHIP_REPAIR = "relationship_repair"
    POLICY_REFORM = "policy_reform"
    INFRASTRUCTURE = "infrastructure"


class ImplementationStatus(str, enum.Enum):
    """Remediation implementation lifecycle status."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


class RiskScope(str, enum.Enum):
    """Scope of grievance risk scoring."""
    OPERATOR = "operator"
    SUPPLIER = "supplier"
    COMMODITY = "commodity"
    REGION = "region"


class RiskLevel(str, enum.Enum):
    """Risk level classification."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CollectiveStatus(str, enum.Enum):
    """Lifecycle status of a collective grievance."""
    FORMING = "forming"
    SUBMITTED = "submitted"
    INVESTIGATING = "investigating"
    MEDIATING = "mediating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class NegotiationStatus(str, enum.Enum):
    """Negotiation status for collective grievances."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    STALLED = "stalled"
    AGREEMENT_REACHED = "agreement_reached"
    FAILED = "failed"


class RegulatoryReportType(str, enum.Enum):
    """Types of regulatory compliance reports."""
    EUDR_ARTICLE16 = "eudr_article16"
    CSDDD_ARTICLE8 = "csddd_article8"
    UNGP_EFFECTIVENESS = "ungp_effectiveness"
    ANNUAL_SUMMARY = "annual_summary"


class AuditAction(str, enum.Enum):
    """Audit trail action types for mechanism manager events."""
    CREATE = "create"
    UPDATE = "update"
    ANALYZE = "analyze"
    ADVANCE_STAGE = "advance_stage"
    CLOSE = "close"
    VERIFY = "verify"
    SCORE = "score"
    GENERATE = "generate"
    SUBMIT = "submit"
    MERGE = "merge"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-GMM-032"
AGENT_VERSION = "1.0.0"

MEDIATION_STAGES_ORDERED: List[MediationStage] = [
    MediationStage.INITIATED,
    MediationStage.PREPARATION,
    MediationStage.DIALOGUE,
    MediationStage.NEGOTIATION,
    MediationStage.SETTLEMENT,
    MediationStage.IMPLEMENTATION,
    MediationStage.CLOSED,
]

SEVERITY_SCORES: Dict[str, int] = {
    "critical": 100,
    "high": 75,
    "medium": 50,
    "low": 25,
}


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CausalChainStep(BaseModel):
    """A step in a root cause causal chain."""
    step: int = Field(..., ge=1, description="Step number in causal chain")
    description: str = Field(..., description="Description of causal step")
    step_type: str = Field(default="contributing", description="Step type (root, contributing, proximate)")

    model_config = {"frozen": False, "extra": "ignore"}


class MediationSession(BaseModel):
    """A single mediation session record."""
    session_number: int = Field(..., ge=1, description="Session number")
    date: Optional[datetime] = None
    duration_minutes: int = Field(default=0, ge=0, description="Session duration")
    summary: str = Field(default="", description="Session summary")
    attendees: List[str] = Field(default_factory=list, description="Attendee identifiers")
    outcomes: List[str] = Field(default_factory=list, description="Session outcomes")

    model_config = {"frozen": False, "extra": "ignore"}


class RemediationAction(BaseModel):
    """An individual remediation action with deadline."""
    action: str = Field(..., description="Action description")
    deadline: Optional[datetime] = None
    status: str = Field(default="pending", description="Action status")
    responsible_party: str = Field(default="", description="Responsible party")

    model_config = {"frozen": False, "extra": "ignore"}


class CollectiveDemand(BaseModel):
    """A demand in a collective grievance."""
    demand: str = Field(..., description="Demand description")
    priority: str = Field(default="medium", description="Priority (low/medium/high/critical)")
    negotiable: bool = Field(default=True, description="Whether this demand is negotiable")

    model_config = {"frozen": False, "extra": "ignore"}


class ScoreFactor(BaseModel):
    """A factor contributing to a risk score."""
    factor_name: str = Field(..., description="Factor name")
    weight: Decimal = Field(default=Decimal("0"), ge=0, le=1, description="Factor weight")
    raw_value: Decimal = Field(default=Decimal("0"), description="Raw value")
    weighted_value: Decimal = Field(default=Decimal("0"), description="Weighted contribution")

    model_config = {"frozen": False, "extra": "ignore"}


class ReportSection(BaseModel):
    """A section of a regulatory compliance report."""
    title: str = Field(..., description="Section title")
    content: Dict[str, Any] = Field(default_factory=dict, description="Section content")
    regulatory_reference: str = Field(default="", description="Regulatory article reference")

    model_config = {"frozen": False, "extra": "ignore"}


# ---------------------------------------------------------------------------
# Core Models (15+)
# ---------------------------------------------------------------------------


class GrievanceAnalyticsRecord(BaseModel):
    """Grievance pattern analysis record.

    Represents a detected pattern across multiple grievances including
    recurring issues, geographic clusters, systemic problems, and
    escalating trends.
    """
    analytics_id: str = Field(..., description="Unique analytics record identifier")
    operator_id: str = Field(..., description="Operator whose grievances are analyzed")
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    grievance_ids: List[str] = Field(default_factory=list, description="Analyzed grievance IDs")
    pattern_type: PatternType = PatternType.ISOLATED
    pattern_description: str = Field(default="", description="Pattern description")
    affected_stakeholder_count: int = Field(default=0, ge=0, description="Affected stakeholders")
    root_causes: List[Dict[str, Any]] = Field(default_factory=list, description="Identified root causes")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendations")
    severity_distribution: Dict[str, int] = Field(default_factory=dict, description="Severity counts")
    category_distribution: Dict[str, int] = Field(default_factory=dict, description="Category counts")
    trend_direction: TrendDirection = TrendDirection.STABLE
    trend_confidence: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RootCauseRecord(BaseModel):
    """Root cause analysis record for a specific grievance.

    Supports five-whys, fishbone, fault-tree, and correlation
    analysis methods with confidence scoring and evidence tracking.
    """
    root_cause_id: str = Field(..., description="Unique root cause record identifier")
    grievance_id: str = Field(..., description="Analyzed grievance ID (from EUDR-031)")
    operator_id: str = Field(..., description="Operator identifier")
    analysis_method: AnalysisMethod = AnalysisMethod.FIVE_WHYS
    primary_cause: str = Field(default="", description="Primary root cause")
    contributing_factors: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_depth: int = Field(default=1, ge=1, le=10, description="Analysis depth")
    confidence_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    causal_chain: List[CausalChainStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class MediationRecord(BaseModel):
    """Multi-party mediation workflow record.

    Tracks the full mediation lifecycle from initiation through
    settlement and implementation with session records, agreements,
    and settlement terms.
    """
    mediation_id: str = Field(..., description="Unique mediation identifier")
    grievance_id: str = Field(..., description="Associated grievance ID (from EUDR-031)")
    operator_id: str = Field(..., description="Operator identifier")
    mediation_stage: MediationStage = MediationStage.INITIATED
    parties: List[Dict[str, Any]] = Field(default_factory=list, description="Involved parties")
    mediator_id: Optional[str] = None
    mediator_type: MediatorType = MediatorType.INTERNAL
    sessions: List[MediationSession] = Field(default_factory=list)
    agreements: List[Dict[str, Any]] = Field(default_factory=list)
    settlement_terms: Dict[str, Any] = Field(default_factory=dict)
    settlement_status: SettlementStatus = SettlementStatus.PENDING
    session_count: int = Field(default=0, ge=0)
    total_duration_minutes: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RemediationRecord(BaseModel):
    """Remediation effectiveness tracking record.

    Tracks the full lifecycle of remediation actions from planning
    through verification with stakeholder satisfaction, cost tracking,
    and effectiveness measurement.
    """
    remediation_id: str = Field(..., description="Unique remediation identifier")
    grievance_id: str = Field(..., description="Associated grievance ID (from EUDR-031)")
    operator_id: str = Field(..., description="Operator identifier")
    remediation_type: RemediationType = RemediationType.PROCESS_CHANGE
    remediation_actions: List[RemediationAction] = Field(default_factory=list)
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED
    completion_percentage: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    effectiveness_indicators: Dict[str, Any] = Field(default_factory=dict)
    stakeholder_satisfaction: Optional[Decimal] = Field(default=None, ge=1, le=5)
    cost_incurred: Decimal = Field(default=Decimal("0"), ge=0)
    timeline_adherence: Decimal = Field(default=Decimal("100"), ge=0, le=100)
    verification_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    lessons_learned: str = Field(default="", description="Lessons learned")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RiskScoreRecord(BaseModel):
    """Grievance risk scoring record.

    Predictive risk analytics across operator, supplier, commodity,
    and region scopes with multi-factor weighted scoring.
    """
    risk_score_id: str = Field(..., description="Unique risk score identifier")
    operator_id: str = Field(..., description="Operator identifier")
    scope: RiskScope = RiskScope.OPERATOR
    scope_identifier: str = Field(..., description="Entity within scope")
    risk_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    risk_level: RiskLevel = RiskLevel.LOW
    grievance_frequency: int = Field(default=0, ge=0)
    average_severity: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    resolution_time_trend: TrendDirection = TrendDirection.STABLE
    unresolved_count: int = Field(default=0, ge=0)
    escalation_rate: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    prediction_confidence: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    score_factors: List[ScoreFactor] = Field(default_factory=list)
    historical_scores: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CollectiveGrievanceRecord(BaseModel):
    """Collective/class-action grievance record.

    Manages group complaints from multiple stakeholders with demand
    tracking, negotiation workflow, and representative coordination.
    """
    collective_id: str = Field(..., description="Unique collective grievance identifier")
    operator_id: str = Field(..., description="Operator identifier")
    title: str = Field(..., description="Collective grievance title")
    description: str = Field(default="", description="Description")
    grievance_category: str = Field(default="process", description="Category")
    lead_complainant_id: Optional[str] = None
    affected_stakeholder_count: int = Field(default=1, ge=1)
    individual_grievance_ids: List[str] = Field(default_factory=list)
    collective_status: CollectiveStatus = CollectiveStatus.FORMING
    spokesperson: Optional[str] = None
    representative_body: Optional[str] = None
    collective_demands: List[CollectiveDemand] = Field(default_factory=list)
    negotiation_status: NegotiationStatus = NegotiationStatus.NOT_STARTED
    supply_chain_nodes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RegulatoryReport(BaseModel):
    """Generated regulatory compliance report.

    Audit-ready documentation for EUDR Article 16, CSDDD Article 8,
    UNGP effectiveness assessments, and annual summaries.
    """
    report_id: str = Field(..., description="Unique report identifier")
    operator_id: str = Field(..., description="Operator the report covers")
    report_type: RegulatoryReportType = RegulatoryReportType.ANNUAL_SUMMARY
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    total_grievances: int = Field(default=0, ge=0)
    resolved_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    average_resolution_days: Decimal = Field(default=Decimal("0"), ge=0)
    satisfaction_rating: Optional[Decimal] = Field(default=None, ge=1, le=5)
    top_categories: List[Dict[str, Any]] = Field(default_factory=list)
    top_root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_effectiveness: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    accessibility_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    sections: List[ReportSection] = Field(default_factory=list)
    report_file_reference: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class AuditEntry(BaseModel):
    """An audit trail entry for grievance mechanism manager events."""
    entry_id: str = Field(..., description="Unique audit entry identifier")
    entity_type: str = Field(..., description="Entity type being audited")
    entity_id: str = Field(..., description="Entity identifier")
    action: AuditAction = AuditAction.CREATE
    actor: str = Field(..., description="Actor performing the action")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the Grievance Mechanism Manager."""
    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
