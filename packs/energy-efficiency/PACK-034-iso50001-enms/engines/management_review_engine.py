# -*- coding: utf-8 -*-
"""
ManagementReviewEngine - PACK-034 ISO 50001 EnMS Engine 10
============================================================

ISO 50001:2018 Clause 9.3 management review engine.  Compiles all
required review inputs (energy policy, objectives, EnPIs, audit
findings, nonconformities, compliance obligations, risks, resources),
generates review decisions, produces management review minutes, and
validates completeness against every mandatory Clause 9.3 requirement.

Calculation Methodology:
    EnPI Improvement:
        improvement_pct = (baseline - current) / baseline * 100

    KPI Status Assignment:
        ahead_of_target   : improvement_pct > target_improvement * 1.05
        on_target          : target_improvement * 0.95 <= improvement_pct
        behind_target      : target_improvement * 0.80 <= improvement_pct
        at_risk            : improvement_pct < target_improvement * 0.80
        no_data            : current_value == 0 and baseline_value == 0

    Resource Utilisation:
        utilization_pct = spent / allocated * 100

    Objectives Completion Rate:
        completion_rate = completed / objectives_count * 100

    NC Closure Rate:
        nc_closure_rate = closed_ncs / findings_count * 100

    Decision Auto-Suggestion:
        - EnPI behind_target  => target_revision or process_improvement
        - open_ncs > 0        => process_improvement
        - adequacy needs_increase => resource_allocation
        - policy needs_update  => policy_change
        - overdue_ncs > 0     => training_need (systemic)

    Next Review Date:
        quarterly:   current + 3 months
        semi_annual: current + 6 months
        annual:      current + 12 months
        as_needed:   current + 6 months (default fallback)

Regulatory References:
    - ISO 50001:2018 Clause 9.3 - Management review
    - ISO 50001:2018 Clause 4.1 - Understanding the organisation
    - ISO 50001:2018 Clause 6.2 - Objectives, energy targets, planning
    - ISO 50001:2018 Clause 6.6 - Energy performance indicators
    - ISO 50001:2018 Clause 9.1 - Monitoring, measurement, analysis
    - ISO 50001:2018 Clause 9.2 - Internal audit
    - ISO 50001:2018 Clause 10.1 - Nonconformity and corrective action
    - ISO 50001:2018 Clause 10.2 - Continual improvement

Zero-Hallucination:
    - All KPI thresholds from ISO 50001 guidance documents
    - Deterministic Decimal arithmetic throughout
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "calculation_time_ms",
                         "provenance_hash", "generated_at")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReviewFrequency(str, Enum):
    """Frequency of management reviews.

    QUARTERLY: Every three months.
    SEMI_ANNUAL: Every six months.
    ANNUAL: Once per year.
    AS_NEEDED: Triggered by events rather than calendar.
    """
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    AS_NEEDED = "as_needed"


class ReviewStatus(str, Enum):
    """Management review lifecycle status.

    SCHEDULED: Review date set, not yet started.
    IN_PREPARATION: Inputs being compiled.
    IN_PROGRESS: Review meeting underway.
    COMPLETED: Review finished, minutes pending approval.
    MINUTES_APPROVED: Minutes approved by top management.
    """
    SCHEDULED = "scheduled"
    IN_PREPARATION = "in_preparation"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MINUTES_APPROVED = "minutes_approved"


class DecisionType(str, Enum):
    """Types of management review decisions.

    POLICY_CHANGE: Change to the energy policy.
    RESOURCE_ALLOCATION: Allocation of additional resources.
    TARGET_REVISION: Revision of energy targets or objectives.
    PROCESS_IMPROVEMENT: Improvement to EnMS processes.
    TRAINING_NEED: Identified training or competence need.
    INVESTMENT_APPROVAL: Approval for capital investment.
    NO_ACTION: No action required; acknowledge only.
    """
    POLICY_CHANGE = "policy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    TARGET_REVISION = "target_revision"
    PROCESS_IMPROVEMENT = "process_improvement"
    TRAINING_NEED = "training_need"
    INVESTMENT_APPROVAL = "investment_approval"
    NO_ACTION = "no_action"


class KPIStatus(str, Enum):
    """Energy performance indicator status.

    ON_TARGET: Performance is within acceptable range of the target.
    AHEAD_OF_TARGET: Performance exceeds the target.
    BEHIND_TARGET: Performance is below target but within recovery range.
    AT_RISK: Performance significantly below target; corrective action needed.
    NO_DATA: Insufficient data to determine status.
    """
    ON_TARGET = "on_target"
    AHEAD_OF_TARGET = "ahead_of_target"
    BEHIND_TARGET = "behind_target"
    AT_RISK = "at_risk"
    NO_DATA = "no_data"


class ActionItemPriority(str, Enum):
    """Priority level for management review action items.

    URGENT: Must be addressed within one week.
    HIGH: Must be addressed within one month.
    MEDIUM: Must be addressed within one quarter.
    LOW: Address when convenient; no regulatory risk.
    """
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResourceAdequacy(str, Enum):
    """Assessment of resource adequacy for the EnMS.

    ADEQUATE: Resources are sufficient for current needs.
    NEEDS_INCREASE: Additional resources required.
    NEEDS_REALLOCATION: Existing resources should be redistributed.
    UNDER_REVIEW: Resource adequacy is being evaluated.
    """
    ADEQUATE = "adequate"
    NEEDS_INCREASE = "needs_increase"
    NEEDS_REALLOCATION = "needs_reallocation"
    UNDER_REVIEW = "under_review"


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

# ISO 50001:2018 Clause 9.3.2 - Required management review inputs.
ISO_50001_REVIEW_INPUTS: List[Dict[str, str]] = [
    {
        "ref": "9.3.2 a)",
        "description": "Status of actions from previous management reviews",
        "category": "previous_actions",
    },
    {
        "ref": "9.3.2 b)",
        "description": "Changes in external and internal issues relevant to the EnMS",
        "category": "context_changes",
    },
    {
        "ref": "9.3.2 c) 1)",
        "description": "Energy performance and improvement in energy performance "
                       "indicators (EnPIs)",
        "category": "energy_performance",
    },
    {
        "ref": "9.3.2 c) 2)",
        "description": "Status of objectives and energy targets",
        "category": "objectives_targets",
    },
    {
        "ref": "9.3.2 c) 3)",
        "description": "Energy performance as established by monitoring, measurement, "
                       "and evaluation of compliance obligations",
        "category": "compliance_status",
    },
    {
        "ref": "9.3.2 d)",
        "description": "Status of nonconformities and corrective actions",
        "category": "nonconformities",
    },
    {
        "ref": "9.3.2 e)",
        "description": "Internal audit results",
        "category": "audit_results",
    },
    {
        "ref": "9.3.2 f)",
        "description": "Status of compliance with applicable legal and other "
                       "requirements",
        "category": "legal_compliance",
    },
    {
        "ref": "9.3.2 g)",
        "description": "Risks and opportunities as identified in planning",
        "category": "risks_opportunities",
    },
    {
        "ref": "9.3.2 h)",
        "description": "Degree to which objectives and energy targets have been met",
        "category": "target_achievement",
    },
    {
        "ref": "9.3.2 i)",
        "description": "Energy performance improvement achieved relative to energy "
                       "baseline(s)",
        "category": "baseline_improvement",
    },
    {
        "ref": "9.3.2 j)",
        "description": "Adequacy of resources",
        "category": "resource_adequacy",
    },
    {
        "ref": "9.3.2 k)",
        "description": "Review of adequacy of competence",
        "category": "competence_adequacy",
    },
    {
        "ref": "9.3.2 l)",
        "description": "Opportunities for continual improvement including "
                       "competence",
        "category": "continual_improvement",
    },
    {
        "ref": "9.3.2 m)",
        "description": "Review of the energy policy",
        "category": "energy_policy",
    },
]

# ISO 50001:2018 Clause 9.3.3 - Required management review outputs.
ISO_50001_REVIEW_OUTPUTS: List[Dict[str, str]] = [
    {
        "ref": "9.3.3 a)",
        "description": "Conclusions on the continuing suitability, adequacy, and "
                       "effectiveness of the EnMS",
        "category": "enms_effectiveness",
    },
    {
        "ref": "9.3.3 b)",
        "description": "Decisions related to continual improvement of energy "
                       "performance and of the EnMS",
        "category": "improvement_decisions",
    },
    {
        "ref": "9.3.3 c)",
        "description": "Decisions related to opportunities to improve integration "
                       "with business processes",
        "category": "business_integration",
    },
    {
        "ref": "9.3.3 d)",
        "description": "Decisions related to whether the EnMS achieves its "
                       "intended outcomes",
        "category": "intended_outcomes",
    },
    {
        "ref": "9.3.3 e)",
        "description": "Decisions related to resource allocation",
        "category": "resource_allocation",
    },
    {
        "ref": "9.3.3 f)",
        "description": "Decisions related to improving competence, awareness, "
                       "and communication",
        "category": "competence_awareness",
    },
    {
        "ref": "9.3.3 g)",
        "description": "Decisions related to the energy policy, EnPIs, EnBs, "
                       "objectives, or energy targets and action plans where the "
                       "expected energy performance improvement was not achieved",
        "category": "performance_shortfall",
    },
]

# KPI status thresholds.  Values are multiplied against the target improvement
# percentage to determine the boundary between statuses.
KPI_STATUS_THRESHOLDS: Dict[str, Decimal] = {
    "ahead_of_target": Decimal("1.05"),
    "on_target_lower": Decimal("0.95"),
    "behind_target_lower": Decimal("0.80"),
    "at_risk_upper": Decimal("0.80"),
}

# Priority weights used when auto-suggesting decision priority.
_PRIORITY_WEIGHTS: Dict[str, int] = {
    "at_risk_enpi": 4,
    "behind_target_enpi": 3,
    "overdue_nc": 4,
    "open_nc": 2,
    "major_nc": 3,
    "resource_gap": 2,
    "policy_update": 1,
}

# Decision priority score thresholds.
_PRIORITY_THRESHOLDS: Dict[str, int] = {
    "urgent": 10,
    "high": 6,
    "medium": 3,
    "low": 0,
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ReviewInput(BaseModel):
    """A single management review input item per ISO 50001 Clause 9.3.2.

    Attributes:
        input_id: Unique identifier for this input item.
        input_category: Category matching ISO 50001 9.3.2 sub-clause.
        description: Human-readable description of the input.
        data_summary: Structured data associated with this input.
        status: Current status text (e.g. 'complete', 'pending', 'overdue').
        source_document: Optional reference to the source document.
        clause_ref: ISO 50001 clause reference (e.g. '9.3.2 a)').
    """
    input_id: str = Field(default_factory=_new_uuid, description="Unique input ID")
    input_category: str = Field(
        default="", max_length=100,
        description="ISO 50001 9.3.2 input category",
    )
    description: str = Field(
        default="", max_length=2000,
        description="Input description",
    )
    data_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data summary for this input",
    )
    status: str = Field(
        default="pending", max_length=50,
        description="Input status (complete, pending, overdue)",
    )
    source_document: Optional[str] = Field(
        default=None, max_length=500,
        description="Source document reference",
    )
    clause_ref: str = Field(
        default="", max_length=20,
        description="ISO 50001 clause reference",
    )

    @field_validator("input_category", mode="before")
    @classmethod
    def validate_input_category(cls, v: Any) -> Any:
        """Normalise category to lowercase with underscores."""
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_")
        return v


class PolicyReview(BaseModel):
    """Energy policy review assessment per ISO 50001 Clause 5.2.

    Attributes:
        current_policy_text: Full text of the current energy policy.
        policy_date: Date the current policy was last approved.
        is_aligned_with_strategy: Whether policy aligns with organisational strategy.
        needs_update: Whether the policy requires revision.
        proposed_changes: List of proposed changes to the policy.
        alignment_notes: Notes on strategic alignment assessment.
    """
    current_policy_text: str = Field(
        default="", max_length=10000,
        description="Current energy policy text",
    )
    policy_date: date = Field(
        default_factory=lambda: date.today(),
        description="Date of current policy",
    )
    is_aligned_with_strategy: bool = Field(
        default=True,
        description="Policy aligned with organisational strategy",
    )
    needs_update: bool = Field(
        default=False,
        description="Whether policy needs update",
    )
    proposed_changes: List[str] = Field(
        default_factory=list,
        description="Proposed policy changes",
    )
    alignment_notes: str = Field(
        default="", max_length=2000,
        description="Notes on alignment assessment",
    )


class ObjectivesReview(BaseModel):
    """Review of energy objectives and targets per ISO 50001 Clause 6.2.

    Attributes:
        objectives_count: Total number of active objectives.
        on_target: Number of objectives on target.
        behind_target: Number of objectives behind target.
        completed: Number of objectives completed.
        cancelled: Number of objectives cancelled.
        new_proposed: List of proposed new objectives.
        completion_rate_pct: Percentage of objectives completed.
    """
    objectives_count: int = Field(default=0, ge=0, description="Total objectives")
    on_target: int = Field(default=0, ge=0, description="On target count")
    behind_target: int = Field(default=0, ge=0, description="Behind target count")
    completed: int = Field(default=0, ge=0, description="Completed count")
    cancelled: int = Field(default=0, ge=0, description="Cancelled count")
    new_proposed: List[str] = Field(
        default_factory=list,
        description="Proposed new objectives",
    )
    completion_rate_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Completion rate percentage",
    )

    @field_validator("completion_rate_pct", mode="before")
    @classmethod
    def coerce_completion_rate(cls, v: Any) -> Any:
        """Coerce to Decimal."""
        return _decimal(v)


class EnPISummary(BaseModel):
    """Energy performance indicator summary per ISO 50001 Clause 6.6.

    Attributes:
        enpi_id: Unique EnPI identifier.
        enpi_name: Human-readable EnPI name.
        enpi_type: Type of EnPI (e.g. 'ratio', 'regression', 'absolute').
        baseline_value: Baseline (EnB) value.
        current_value: Current period value.
        target_value: Target value for the review period.
        improvement_pct: Improvement percentage vs baseline.
        status: KPI status classification.
        trend: Trend description (e.g. 'improving', 'stable', 'declining').
        unit: Unit of measurement.
        data_quality_score: Data quality score 0-100.
    """
    enpi_id: str = Field(default_factory=_new_uuid, description="EnPI ID")
    enpi_name: str = Field(default="", max_length=200, description="EnPI name")
    enpi_type: str = Field(
        default="ratio", max_length=50,
        description="EnPI type (ratio, regression, absolute)",
    )
    baseline_value: Decimal = Field(
        default=Decimal("0"), description="Baseline (EnB) value",
    )
    current_value: Decimal = Field(
        default=Decimal("0"), description="Current value",
    )
    target_value: Decimal = Field(
        default=Decimal("0"), description="Target value",
    )
    improvement_pct: Decimal = Field(
        default=Decimal("0"), description="Improvement percentage",
    )
    status: KPIStatus = Field(
        default=KPIStatus.NO_DATA, description="KPI status",
    )
    trend: str = Field(
        default="stable", max_length=50, description="Trend direction",
    )
    unit: str = Field(default="kWh", max_length=30, description="Unit")
    data_quality_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Data quality 0-100",
    )

    @field_validator("baseline_value", "current_value", "target_value",
                     "improvement_pct", "data_quality_score", mode="before")
    @classmethod
    def coerce_decimals(cls, v: Any) -> Any:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> Any:
        """Accept string values for KPIStatus."""
        if isinstance(v, str):
            valid = {s.value for s in KPIStatus}
            if v in valid:
                return KPIStatus(v)
        return v


class ResourceReview(BaseModel):
    """Resource adequacy assessment per ISO 50001 Clause 7.1.

    Attributes:
        resource_id: Unique resource review identifier.
        category: Resource category (e.g. 'financial', 'personnel', 'equipment').
        allocated: Allocated budget or resource quantity.
        spent: Actual expenditure or resource usage.
        utilization_pct: Utilisation percentage.
        adequacy: Adequacy assessment.
        recommendation: Recommendation text.
        notes: Additional notes.
    """
    resource_id: str = Field(default_factory=_new_uuid, description="Resource ID")
    category: str = Field(
        default="", max_length=100,
        description="Resource category",
    )
    allocated: Decimal = Field(
        default=Decimal("0"), ge=0, description="Allocated amount",
    )
    spent: Decimal = Field(
        default=Decimal("0"), ge=0, description="Spent amount",
    )
    utilization_pct: Decimal = Field(
        default=Decimal("0"), ge=0, description="Utilisation percentage",
    )
    adequacy: ResourceAdequacy = Field(
        default=ResourceAdequacy.UNDER_REVIEW,
        description="Adequacy assessment",
    )
    recommendation: str = Field(
        default="", max_length=2000,
        description="Resource recommendation",
    )
    notes: str = Field(default="", max_length=2000, description="Notes")

    @field_validator("allocated", "spent", "utilization_pct", mode="before")
    @classmethod
    def coerce_decimals(cls, v: Any) -> Any:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

    @field_validator("adequacy", mode="before")
    @classmethod
    def validate_adequacy(cls, v: Any) -> Any:
        """Accept string values for ResourceAdequacy."""
        if isinstance(v, str):
            valid = {s.value for s in ResourceAdequacy}
            if v in valid:
                return ResourceAdequacy(v)
        return v


class AuditSummary(BaseModel):
    """Internal audit results summary per ISO 50001 Clause 9.2.

    Attributes:
        total_audits: Number of audits conducted in the review period.
        findings_count: Total number of findings (NC + observations).
        major_ncs: Number of major nonconformities.
        minor_ncs: Number of minor nonconformities.
        closed_ncs: Number of closed nonconformities.
        open_ncs: Number of open (unresolved) nonconformities.
        overdue_ncs: Number of overdue nonconformities.
        observations: Number of observations (not NCs).
        nc_closure_rate_pct: NC closure rate as percentage.
    """
    total_audits: int = Field(default=0, ge=0, description="Total audits")
    findings_count: int = Field(default=0, ge=0, description="Total findings")
    major_ncs: int = Field(default=0, ge=0, description="Major NCs")
    minor_ncs: int = Field(default=0, ge=0, description="Minor NCs")
    closed_ncs: int = Field(default=0, ge=0, description="Closed NCs")
    open_ncs: int = Field(default=0, ge=0, description="Open NCs")
    overdue_ncs: int = Field(default=0, ge=0, description="Overdue NCs")
    observations: int = Field(default=0, ge=0, description="Observations")
    nc_closure_rate_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="NC closure rate %",
    )

    @field_validator("nc_closure_rate_pct", mode="before")
    @classmethod
    def coerce_closure_rate(cls, v: Any) -> Any:
        """Coerce to Decimal."""
        return _decimal(v)


class ContinualImprovement(BaseModel):
    """Continual improvement item per ISO 50001 Clause 10.2.

    Attributes:
        improvement_id: Unique improvement identifier.
        description: Description of the improvement action.
        category: Category (e.g. 'operational', 'technological', 'behavioural').
        energy_savings_kwh: Estimated or verified energy savings (kWh).
        cost_savings: Estimated or verified cost savings.
        implementation_date: Date of implementation.
        status: Current status (proposed, in_progress, completed, verified).
        verified: Whether savings have been verified via M&V.
        responsible_person: Person responsible for implementation.
    """
    improvement_id: str = Field(
        default_factory=_new_uuid, description="Improvement ID",
    )
    description: str = Field(
        default="", max_length=2000, description="Improvement description",
    )
    category: str = Field(
        default="operational", max_length=100,
        description="Improvement category",
    )
    energy_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy savings (kWh)",
    )
    cost_savings: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cost savings",
    )
    implementation_date: date = Field(
        default_factory=lambda: date.today(),
        description="Implementation date",
    )
    status: str = Field(
        default="proposed", max_length=50,
        description="Status (proposed, in_progress, completed, verified)",
    )
    verified: bool = Field(default=False, description="Savings verified")
    responsible_person: str = Field(
        default="", max_length=200, description="Responsible person",
    )

    @field_validator("energy_savings_kwh", "cost_savings", mode="before")
    @classmethod
    def coerce_decimals(cls, v: Any) -> Any:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class ReviewDecision(BaseModel):
    """Management review decision and action item per ISO 50001 Clause 9.3.3.

    Attributes:
        decision_id: Unique decision identifier.
        decision_type: Type of decision.
        description: Description of the decision.
        rationale: Rationale and evidence supporting the decision.
        assigned_to: Person or team assigned.
        due_date: Deadline for completion.
        priority: Priority level.
        resources_required: Description of resources needed.
        clause_ref: ISO 50001 clause driving this decision.
        status: Action status (open, in_progress, completed).
    """
    decision_id: str = Field(
        default_factory=_new_uuid, description="Decision ID",
    )
    decision_type: DecisionType = Field(
        default=DecisionType.NO_ACTION, description="Decision type",
    )
    description: str = Field(
        default="", max_length=2000, description="Decision description",
    )
    rationale: str = Field(
        default="", max_length=2000, description="Decision rationale",
    )
    assigned_to: str = Field(
        default="", max_length=200, description="Assignee",
    )
    due_date: date = Field(
        default_factory=lambda: date.today(),
        description="Due date",
    )
    priority: ActionItemPriority = Field(
        default=ActionItemPriority.MEDIUM, description="Priority",
    )
    resources_required: Optional[str] = Field(
        default=None, max_length=2000,
        description="Resources required",
    )
    clause_ref: str = Field(
        default="", max_length=20,
        description="ISO 50001 clause reference",
    )
    status: str = Field(
        default="open", max_length=50,
        description="Action status",
    )

    @field_validator("decision_type", mode="before")
    @classmethod
    def validate_decision_type(cls, v: Any) -> Any:
        """Accept string values for DecisionType."""
        if isinstance(v, str):
            valid = {s.value for s in DecisionType}
            if v in valid:
                return DecisionType(v)
        return v

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v: Any) -> Any:
        """Accept string values for ActionItemPriority."""
        if isinstance(v, str):
            valid = {s.value for s in ActionItemPriority}
            if v in valid:
                return ActionItemPriority(v)
        return v


class ManagementReviewMinutes(BaseModel):
    """Management review meeting minutes per ISO 50001 Clause 9.3.

    Attributes:
        minutes_id: Unique minutes identifier.
        meeting_date: Date of the review meeting.
        attendees: List of attendee names / roles.
        chairperson: Name of the meeting chairperson (top management).
        review_period_start: Start of the review period.
        review_period_end: End of the review period.
        agenda_items: Ordered list of agenda items discussed.
        summary_notes: Free-text meeting summary notes.
        approved_by: Name of person approving the minutes.
        approval_date: Date minutes were approved.
    """
    minutes_id: str = Field(
        default_factory=_new_uuid, description="Minutes ID",
    )
    meeting_date: date = Field(
        default_factory=lambda: date.today(),
        description="Meeting date",
    )
    attendees: List[str] = Field(
        default_factory=list, description="Attendee names",
    )
    chairperson: str = Field(
        default="", max_length=200, description="Chairperson name",
    )
    review_period_start: date = Field(
        default_factory=lambda: date.today(),
        description="Review period start",
    )
    review_period_end: date = Field(
        default_factory=lambda: date.today(),
        description="Review period end",
    )
    agenda_items: List[str] = Field(
        default_factory=list, description="Agenda items",
    )
    summary_notes: str = Field(
        default="", max_length=10000,
        description="Meeting summary notes",
    )
    approved_by: str = Field(
        default="", max_length=200,
        description="Approved by",
    )
    approval_date: Optional[date] = Field(
        default=None, description="Approval date",
    )


class ManagementReviewResult(BaseModel):
    """Complete management review result per ISO 50001 Clause 9.3.

    Attributes:
        review_id: Unique review identifier.
        enms_id: EnMS identifier.
        review_date: Date/time the review was generated.
        review_period_start: Start of the review period.
        review_period_end: End of the review period.
        status: Current review status.
        inputs: Complete list of review inputs (Clause 9.3.2).
        policy_review: Energy policy review assessment.
        objectives_review: Objectives and targets review.
        enpi_summaries: EnPI performance summaries.
        resource_review: Resource adequacy assessment.
        audit_summary: Internal audit results summary.
        improvements: Continual improvement items.
        decisions: Management review decisions (Clause 9.3.3).
        minutes: Management review meeting minutes.
        next_review_date: Scheduled date for the next review.
        completeness_check: Completeness validation results.
        provenance_hash: SHA-256 provenance hash.
        calculation_time_ms: Processing time in milliseconds.
    """
    review_id: str = Field(
        default_factory=_new_uuid, description="Review ID",
    )
    enms_id: str = Field(default="", max_length=100, description="EnMS ID")
    review_date: datetime = Field(
        default_factory=_utcnow, description="Review date/time",
    )
    review_period_start: date = Field(
        default_factory=lambda: date.today(),
        description="Review period start",
    )
    review_period_end: date = Field(
        default_factory=lambda: date.today(),
        description="Review period end",
    )
    status: ReviewStatus = Field(
        default=ReviewStatus.SCHEDULED, description="Review status",
    )
    inputs: List[ReviewInput] = Field(
        default_factory=list,
        description="Review inputs per Clause 9.3.2",
    )
    policy_review: Optional[PolicyReview] = Field(
        default=None, description="Energy policy review",
    )
    objectives_review: Optional[ObjectivesReview] = Field(
        default=None, description="Objectives review",
    )
    enpi_summaries: List[EnPISummary] = Field(
        default_factory=list, description="EnPI summaries",
    )
    resource_review: Optional[ResourceReview] = Field(
        default=None, description="Resource adequacy review",
    )
    audit_summary: Optional[AuditSummary] = Field(
        default=None, description="Internal audit summary",
    )
    improvements: List[ContinualImprovement] = Field(
        default_factory=list, description="Continual improvement items",
    )
    decisions: List[ReviewDecision] = Field(
        default_factory=list, description="Review decisions",
    )
    minutes: Optional[ManagementReviewMinutes] = Field(
        default=None, description="Meeting minutes",
    )
    next_review_date: Optional[date] = Field(
        default=None, description="Next review date",
    )
    completeness_check: Dict[str, Any] = Field(
        default_factory=dict,
        description="Completeness validation results",
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)",
    )

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> Any:
        """Accept string values for ReviewStatus."""
        if isinstance(v, str):
            valid = {s.value for s in ReviewStatus}
            if v in valid:
                return ReviewStatus(v)
        return v


# ---------------------------------------------------------------------------
# Model Rebuild (required for Pydantic v2 + __future__.annotations)
# ---------------------------------------------------------------------------

ReviewInput.model_rebuild()
PolicyReview.model_rebuild()
ObjectivesReview.model_rebuild()
EnPISummary.model_rebuild()
ResourceReview.model_rebuild()
AuditSummary.model_rebuild()
ContinualImprovement.model_rebuild()
ReviewDecision.model_rebuild()
ManagementReviewMinutes.model_rebuild()
ManagementReviewResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ManagementReviewEngine:
    """ISO 50001 Clause 9.3 management review engine.

    Compiles all required review inputs, evaluates EnPI performance against
    baselines and targets, summarises audit findings, assesses resource
    adequacy, auto-generates review decisions, produces management review
    minutes, and validates completeness against every mandatory Clause 9.3
    requirement.

    Usage::

        engine = ManagementReviewEngine()
        result = engine.prepare_review(
            enms_id="ENMS-001",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            data={
                "policy": {...},
                "objectives": {...},
                "enpi_data": [...],
                "resources": {...},
                "audits": {...},
                "improvements": [...],
                "previous_actions": [...],
                "context_changes": [...],
                "compliance": {...},
                "risks_opportunities": [...],
                "attendees": [...],
                "chairperson": "CEO Name",
            },
        )
        assert result.provenance_hash != ""
        assert result.completeness_check["is_complete"] is True
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ManagementReviewEngine.

        Args:
            config: Optional overrides. Supported keys:
                - review_frequency (str): default review frequency
                - default_currency (str): currency symbol
                - energy_price_per_kwh (Decimal): for cost calculations
                - auto_suggest_decisions (bool): enable auto-decision logic
        """
        self.config = config or {}
        self._review_frequency = ReviewFrequency(
            self.config.get("review_frequency", ReviewFrequency.ANNUAL.value)
        )
        self._currency = str(self.config.get("default_currency", "EUR"))
        self._energy_price = _decimal(
            self.config.get("energy_price_per_kwh", "0.15")
        )
        self._auto_suggest = bool(
            self.config.get("auto_suggest_decisions", True)
        )
        logger.info(
            "ManagementReviewEngine v%s initialised "
            "(frequency=%s, currency=%s, auto_suggest=%s)",
            self.engine_version,
            self._review_frequency.value,
            self._currency,
            self._auto_suggest,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def prepare_review(
        self,
        enms_id: str,
        period_start: date,
        period_end: date,
        data: Dict[str, Any],
    ) -> ManagementReviewResult:
        """Prepare a complete management review per ISO 50001 Clause 9.3.

        Compiles all required inputs, evaluates performance, generates
        decisions, validates completeness, and returns a fully-populated
        ManagementReviewResult with SHA-256 provenance.

        Args:
            enms_id: Energy management system identifier.
            period_start: Start of the review period.
            period_end: End of the review period.
            data: Dictionary containing all review source data.

        Returns:
            ManagementReviewResult with complete review data.

        Raises:
            ValueError: If enms_id is empty or period dates are invalid.
        """
        t0 = time.perf_counter()
        logger.info(
            "Preparing management review: enms_id=%s, period=%s to %s",
            enms_id, period_start, period_end,
        )

        # Validate basic inputs
        if not enms_id or not enms_id.strip():
            raise ValueError("enms_id must not be empty")
        if period_end < period_start:
            raise ValueError(
                f"period_end ({period_end}) must be >= period_start ({period_start})"
            )

        # Step 1: Compile review inputs
        inputs = self.compile_review_inputs(data)
        logger.info("Compiled %d review inputs", len(inputs))

        # Step 2: Review energy policy
        policy_review = self.review_energy_policy(
            data.get("policy", {})
        )

        # Step 3: Review objectives status
        objectives_review = self.review_objectives_status(
            data.get("objectives", {})
        )

        # Step 4: Summarise EnPI performance
        enpi_summaries = self.summarize_enpi_performance(
            data.get("enpi_data", [])
        )
        logger.info("Summarised %d EnPIs", len(enpi_summaries))

        # Step 5: Assess resource adequacy
        resource_review = self.assess_resource_adequacy(
            data.get("resources", {})
        )

        # Step 6: Compile audit summary
        audit_summary = self.compile_audit_summary(
            data.get("audits", {})
        )

        # Step 7: Compile continual improvements
        improvements = self.compile_improvements(
            data.get("improvements", [])
        )

        # Step 8: Generate review decisions
        decisions = self.generate_review_decisions(
            inputs, enpi_summaries, audit_summary,
            policy_review, resource_review,
        )
        logger.info("Generated %d review decisions", len(decisions))

        # Step 9: Calculate next review date
        next_review = self.calculate_next_review_date(
            period_end, self._review_frequency,
        )

        # Step 10: Build result
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = ManagementReviewResult(
            enms_id=enms_id,
            review_date=_utcnow(),
            review_period_start=period_start,
            review_period_end=period_end,
            status=ReviewStatus.COMPLETED,
            inputs=inputs,
            policy_review=policy_review,
            objectives_review=objectives_review,
            enpi_summaries=enpi_summaries,
            resource_review=resource_review,
            audit_summary=audit_summary,
            improvements=improvements,
            decisions=decisions,
            next_review_date=next_review,
            calculation_time_ms=elapsed_ms,
        )

        # Step 11: Generate minutes if attendees provided
        attendees = data.get("attendees", [])
        chairperson = data.get("chairperson", "")
        if attendees:
            result.minutes = self.generate_minutes(
                result, attendees, chairperson,
            )

        # Step 12: Validate completeness
        result.completeness_check = self.validate_review_completeness(result)

        # Step 13: Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        # Update timing after all processing
        result.calculation_time_ms = int(
            (time.perf_counter() - t0) * 1000
        )

        logger.info(
            "Management review prepared: review_id=%s, "
            "inputs=%d, decisions=%d, complete=%s, hash=%s, time=%dms",
            result.review_id,
            len(result.inputs),
            len(result.decisions),
            result.completeness_check.get("is_complete", False),
            result.provenance_hash[:16],
            result.calculation_time_ms,
        )
        return result

    def compile_review_inputs(
        self,
        data: Dict[str, Any],
    ) -> List[ReviewInput]:
        """Compile all ISO 50001 Clause 9.3.2 required review inputs.

        Maps source data to each mandatory review input category.

        Args:
            data: Dictionary containing review source data.

        Returns:
            List of ReviewInput items covering all Clause 9.3.2 items.
        """
        t0 = time.perf_counter()
        inputs: List[ReviewInput] = []

        for req in ISO_50001_REVIEW_INPUTS:
            category = req["category"]
            clause_ref = req["ref"]
            description = req["description"]

            # Extract matching data from the source dictionary
            source_data = data.get(category, {})
            if isinstance(source_data, list):
                data_summary = {"items": source_data, "count": len(source_data)}
            elif isinstance(source_data, dict):
                data_summary = source_data
            else:
                data_summary = {"value": str(source_data)} if source_data else {}

            # Determine status based on data availability
            status = self._assess_input_status(category, data_summary)

            inputs.append(ReviewInput(
                input_category=category,
                description=description,
                data_summary=data_summary,
                status=status,
                clause_ref=clause_ref,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Compiled %d review inputs in %.1f ms", len(inputs), elapsed_ms,
        )
        return inputs

    def review_energy_policy(
        self,
        policy_data: Dict[str, Any],
    ) -> PolicyReview:
        """Review the energy policy per ISO 50001 Clause 5.2.

        Evaluates whether the current energy policy remains aligned with
        the organisational strategy and identifies necessary updates.

        Args:
            policy_data: Dictionary with policy details.

        Returns:
            PolicyReview assessment.
        """
        t0 = time.perf_counter()

        policy_text = str(policy_data.get("current_policy_text", ""))
        raw_date = policy_data.get("policy_date")
        policy_date = self._parse_date(raw_date) if raw_date else date.today()

        is_aligned = bool(policy_data.get("is_aligned_with_strategy", True))
        needs_update = bool(policy_data.get("needs_update", False))
        proposed_changes = list(policy_data.get("proposed_changes", []))
        alignment_notes = str(policy_data.get("alignment_notes", ""))

        # Auto-detect staleness: policy older than 12 months
        days_since_update = (date.today() - policy_date).days
        if days_since_update > 365 and not needs_update:
            needs_update = True
            proposed_changes.append(
                "Policy is older than 12 months; annual review recommended "
                "per ISO 50001 Clause 9.3"
            )
            logger.info(
                "Policy auto-flagged for update: %d days since last revision",
                days_since_update,
            )

        # Check for missing required policy elements
        required_elements = [
            "commitment to continual improvement",
            "energy performance",
            "legal requirements",
            "information and resources",
            "objectives and targets",
        ]
        missing_elements: List[str] = []
        policy_lower = policy_text.lower()
        for element in required_elements:
            if element not in policy_lower and policy_text:
                missing_elements.append(element)

        if missing_elements:
            needs_update = True
            for elem in missing_elements:
                change = f"Add reference to '{elem}' per ISO 50001 Clause 5.2"
                if change not in proposed_changes:
                    proposed_changes.append(change)

        result = PolicyReview(
            current_policy_text=policy_text,
            policy_date=policy_date,
            is_aligned_with_strategy=is_aligned,
            needs_update=needs_update,
            proposed_changes=proposed_changes,
            alignment_notes=alignment_notes,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Policy review: aligned=%s, needs_update=%s, "
            "proposed_changes=%d (%.1f ms)",
            is_aligned, needs_update, len(proposed_changes), elapsed_ms,
        )
        return result

    def review_objectives_status(
        self,
        objectives_data: Dict[str, Any],
    ) -> ObjectivesReview:
        """Review objectives and energy targets per ISO 50001 Clause 6.2.

        Evaluates each objective's status and computes aggregate metrics.

        Args:
            objectives_data: Dictionary with objectives details.

        Returns:
            ObjectivesReview assessment.
        """
        t0 = time.perf_counter()

        total = int(objectives_data.get("objectives_count", 0))
        on_target = int(objectives_data.get("on_target", 0))
        behind = int(objectives_data.get("behind_target", 0))
        completed = int(objectives_data.get("completed", 0))
        cancelled = int(objectives_data.get("cancelled", 0))
        new_proposed = list(objectives_data.get("new_proposed", []))

        # If individual objectives provided, compute counts
        individual = objectives_data.get("objectives", [])
        if individual and isinstance(individual, list) and total == 0:
            total = len(individual)
            on_target = sum(
                1 for o in individual
                if str(o.get("status", "")).lower() == "on_target"
            )
            behind = sum(
                1 for o in individual
                if str(o.get("status", "")).lower() in (
                    "behind_target", "behind", "at_risk",
                )
            )
            completed = sum(
                1 for o in individual
                if str(o.get("status", "")).lower() == "completed"
            )
            cancelled = sum(
                1 for o in individual
                if str(o.get("status", "")).lower() == "cancelled"
            )

        # Calculate completion rate
        completion_rate = _safe_pct(
            _decimal(completed), _decimal(total),
        )
        completion_rate = _round_val(completion_rate, 2)

        result = ObjectivesReview(
            objectives_count=total,
            on_target=on_target,
            behind_target=behind,
            completed=completed,
            cancelled=cancelled,
            new_proposed=new_proposed,
            completion_rate_pct=completion_rate,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Objectives review: total=%d, on_target=%d, behind=%d, "
            "completed=%d, rate=%.1f%% (%.1f ms)",
            total, on_target, behind, completed,
            float(completion_rate), elapsed_ms,
        )
        return result

    def summarize_enpi_performance(
        self,
        enpi_data: List[Any],
    ) -> List[EnPISummary]:
        """Summarise EnPI performance per ISO 50001 Clause 6.6.

        For each EnPI, calculates improvement percentage vs baseline,
        assigns a KPI status using threshold-based classification, and
        determines the trend direction.

        Args:
            enpi_data: List of EnPI data dictionaries.

        Returns:
            List of EnPISummary objects with computed statuses.
        """
        t0 = time.perf_counter()
        summaries: List[EnPISummary] = []

        for item in enpi_data:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict EnPI item: %s", type(item))
                continue

            enpi_name = str(item.get("enpi_name", "Unknown EnPI"))
            enpi_type = str(item.get("enpi_type", "ratio"))
            baseline = _decimal(item.get("baseline_value", 0))
            current = _decimal(item.get("current_value", 0))
            target = _decimal(item.get("target_value", 0))
            unit = str(item.get("unit", "kWh"))
            trend_raw = str(item.get("trend", ""))
            dq_score = _decimal(item.get("data_quality_score", 0))

            # Calculate improvement percentage
            improvement_pct = self._calculate_improvement(
                baseline, current, enpi_type,
            )

            # Determine target improvement percentage
            target_improvement = self._calculate_improvement(
                baseline, target, enpi_type,
            )

            # Assign KPI status
            status = self._assign_kpi_status(
                improvement_pct, target_improvement, baseline, current,
            )

            # Determine trend if not provided
            trend = trend_raw if trend_raw else self._determine_trend(
                improvement_pct, target_improvement,
            )

            summary = EnPISummary(
                enpi_name=enpi_name,
                enpi_type=enpi_type,
                baseline_value=_round_val(baseline, 4),
                current_value=_round_val(current, 4),
                target_value=_round_val(target, 4),
                improvement_pct=_round_val(improvement_pct, 4),
                status=status,
                trend=trend,
                unit=unit,
                data_quality_score=_round_val(dq_score, 2),
            )
            summaries.append(summary)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Summarised %d EnPIs in %.1f ms", len(summaries), elapsed_ms,
        )
        return summaries

    def assess_resource_adequacy(
        self,
        resource_data: Dict[str, Any],
    ) -> ResourceReview:
        """Assess resource adequacy per ISO 50001 Clause 7.1.

        Evaluates whether allocated resources are sufficient for the
        energy management system, based on utilisation rates and budget
        adherence.

        Args:
            resource_data: Dictionary with resource data.

        Returns:
            ResourceReview assessment.
        """
        t0 = time.perf_counter()

        category = str(resource_data.get("category", "overall"))
        allocated = _decimal(resource_data.get("allocated", 0))
        spent = _decimal(resource_data.get("spent", 0))
        recommendation = str(resource_data.get("recommendation", ""))
        notes = str(resource_data.get("notes", ""))

        # Calculate utilisation percentage
        utilization_pct = _safe_pct(spent, allocated)
        utilization_pct = _round_val(utilization_pct, 2)

        # Determine adequacy based on utilisation thresholds
        adequacy = self._determine_adequacy(utilization_pct, resource_data)

        # Auto-generate recommendation if not provided
        if not recommendation:
            recommendation = self._generate_resource_recommendation(
                adequacy, utilization_pct, allocated, spent,
            )

        result = ResourceReview(
            category=category,
            allocated=_round_val(allocated, 2),
            spent=_round_val(spent, 2),
            utilization_pct=utilization_pct,
            adequacy=adequacy,
            recommendation=recommendation,
            notes=notes,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Resource review: category=%s, utilisation=%.1f%%, "
            "adequacy=%s (%.1f ms)",
            category, float(utilization_pct), adequacy.value, elapsed_ms,
        )
        return result

    def compile_audit_summary(
        self,
        audit_data: Dict[str, Any],
    ) -> AuditSummary:
        """Compile internal audit results per ISO 50001 Clause 9.2.

        Aggregates audit finding counts and calculates the NC closure
        rate.

        Args:
            audit_data: Dictionary with audit details.

        Returns:
            AuditSummary with aggregated findings.
        """
        t0 = time.perf_counter()

        total_audits = int(audit_data.get("total_audits", 0))
        findings_count = int(audit_data.get("findings_count", 0))
        major_ncs = int(audit_data.get("major_ncs", 0))
        minor_ncs = int(audit_data.get("minor_ncs", 0))
        closed_ncs = int(audit_data.get("closed_ncs", 0))
        open_ncs = int(audit_data.get("open_ncs", 0))
        overdue_ncs = int(audit_data.get("overdue_ncs", 0))
        observations = int(audit_data.get("observations", 0))

        # If individual audits are provided, aggregate
        audit_list = audit_data.get("audits", [])
        if audit_list and isinstance(audit_list, list) and total_audits == 0:
            total_audits = len(audit_list)
            for audit in audit_list:
                if isinstance(audit, dict):
                    findings_count += int(audit.get("findings_count", 0))
                    major_ncs += int(audit.get("major_ncs", 0))
                    minor_ncs += int(audit.get("minor_ncs", 0))
                    closed_ncs += int(audit.get("closed_ncs", 0))
                    open_ncs += int(audit.get("open_ncs", 0))
                    overdue_ncs += int(audit.get("overdue_ncs", 0))
                    observations += int(audit.get("observations", 0))

        # Calculate NC closure rate
        total_ncs = major_ncs + minor_ncs
        nc_closure_rate = _safe_pct(_decimal(closed_ncs), _decimal(total_ncs))
        nc_closure_rate = _round_val(nc_closure_rate, 2)

        # Auto-calculate findings if not set
        if findings_count == 0 and (total_ncs + observations) > 0:
            findings_count = total_ncs + observations

        result = AuditSummary(
            total_audits=total_audits,
            findings_count=findings_count,
            major_ncs=major_ncs,
            minor_ncs=minor_ncs,
            closed_ncs=closed_ncs,
            open_ncs=open_ncs,
            overdue_ncs=overdue_ncs,
            observations=observations,
            nc_closure_rate_pct=nc_closure_rate,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Audit summary: audits=%d, findings=%d, major=%d, minor=%d, "
            "closed=%d, open=%d, overdue=%d, closure_rate=%.1f%% (%.1f ms)",
            total_audits, findings_count, major_ncs, minor_ncs,
            closed_ncs, open_ncs, overdue_ncs,
            float(nc_closure_rate), elapsed_ms,
        )
        return result

    def compile_improvements(
        self,
        improvement_data: List[Any],
    ) -> List[ContinualImprovement]:
        """Compile continual improvement items per ISO 50001 Clause 10.2.

        Converts raw improvement data into structured ContinualImprovement
        objects with validated energy and cost savings.

        Args:
            improvement_data: List of improvement data dictionaries.

        Returns:
            List of ContinualImprovement objects.
        """
        t0 = time.perf_counter()
        improvements: List[ContinualImprovement] = []

        for item in improvement_data:
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict improvement item: %s", type(item),
                )
                continue

            description = str(item.get("description", ""))
            category = str(item.get("category", "operational"))
            energy_savings = _decimal(item.get("energy_savings_kwh", 0))
            cost_savings = _decimal(item.get("cost_savings", 0))
            impl_date_raw = item.get("implementation_date")
            impl_date = (
                self._parse_date(impl_date_raw)
                if impl_date_raw else date.today()
            )
            status = str(item.get("status", "proposed"))
            verified = bool(item.get("verified", False))
            responsible = str(item.get("responsible_person", ""))

            # Auto-calculate cost savings if energy savings provided
            if energy_savings > Decimal("0") and cost_savings == Decimal("0"):
                cost_savings = _round_val(
                    energy_savings * self._energy_price, 2,
                )

            improvement = ContinualImprovement(
                description=description,
                category=category,
                energy_savings_kwh=_round_val(energy_savings, 2),
                cost_savings=_round_val(cost_savings, 2),
                implementation_date=impl_date,
                status=status,
                verified=verified,
                responsible_person=responsible,
            )
            improvements.append(improvement)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Compiled %d improvement items in %.1f ms",
            len(improvements), elapsed_ms,
        )
        return improvements

    def generate_review_decisions(
        self,
        inputs: List[ReviewInput],
        enpi_summaries: List[EnPISummary],
        audit_summary: AuditSummary,
        policy_review: Optional[PolicyReview] = None,
        resource_review: Optional[ResourceReview] = None,
    ) -> List[ReviewDecision]:
        """Generate management review decisions per ISO 50001 Clause 9.3.3.

        Auto-suggests decisions based on EnPI performance, audit findings,
        policy status, and resource adequacy.  Each decision is mapped to
        the relevant ISO 50001 clause.

        Args:
            inputs: Compiled review inputs.
            enpi_summaries: EnPI performance summaries.
            audit_summary: Internal audit summary.
            policy_review: Optional energy policy review.
            resource_review: Optional resource adequacy review.

        Returns:
            List of ReviewDecision objects.
        """
        t0 = time.perf_counter()
        decisions: List[ReviewDecision] = []

        if not self._auto_suggest:
            logger.info("Auto-suggest disabled; returning empty decisions")
            return decisions

        # Decision 1: EnPI performance issues
        decisions.extend(
            self._suggest_enpi_decisions(enpi_summaries)
        )

        # Decision 2: Audit finding actions
        decisions.extend(
            self._suggest_audit_decisions(audit_summary)
        )

        # Decision 3: Policy update
        if policy_review and policy_review.needs_update:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.POLICY_CHANGE,
                description=(
                    "Update energy policy to address identified gaps: "
                    + "; ".join(policy_review.proposed_changes[:3])
                ),
                rationale=(
                    "Energy policy review identified areas requiring update "
                    "per ISO 50001 Clause 5.2"
                ),
                due_date=date.today() + timedelta(days=90),
                priority=ActionItemPriority.MEDIUM,
                clause_ref="9.3.3 g)",
            ))

        # Decision 4: Resource adequacy
        if resource_review:
            decisions.extend(
                self._suggest_resource_decisions(resource_review)
            )

        # Decision 5: Previous actions overdue
        decisions.extend(
            self._suggest_previous_action_decisions(inputs)
        )

        # Ensure at least one output decision covers each 9.3.3 output
        decisions = self._ensure_output_coverage(decisions)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Generated %d decisions in %.1f ms", len(decisions), elapsed_ms,
        )
        return decisions

    def generate_minutes(
        self,
        result: ManagementReviewResult,
        attendees: List[str],
        chairperson: str,
    ) -> ManagementReviewMinutes:
        """Generate management review meeting minutes.

        Produces a structured minutes record from the review result,
        including standard agenda items derived from ISO 50001 Clause 9.3.

        Args:
            result: The management review result to generate minutes for.
            attendees: List of meeting attendees.
            chairperson: Name of the meeting chairperson.

        Returns:
            ManagementReviewMinutes object.
        """
        t0 = time.perf_counter()

        # Build standard agenda items from ISO 50001 Clause 9.3
        agenda_items = [
            "1. Opening and approval of previous minutes",
            "2. Status of actions from previous management review",
            "3. Changes in external and internal issues",
            "4. Energy performance review and EnPI trends",
            "5. Status of objectives and energy targets",
            "6. Energy performance vs compliance obligations",
            "7. Nonconformity and corrective action status",
            "8. Internal audit results",
            "9. Legal and other compliance status",
            "10. Risks and opportunities",
            "11. Energy performance improvement vs baselines",
            "12. Resource adequacy and competence review",
            "13. Opportunities for continual improvement",
            "14. Energy policy review",
            "15. Review decisions and action items",
            "16. Next review date and closing",
        ]

        # Generate summary notes
        summary_parts: List[str] = []
        summary_parts.append(
            f"Management Review for EnMS {result.enms_id} "
            f"covering period {result.review_period_start} to "
            f"{result.review_period_end}."
        )

        if result.enpi_summaries:
            on_target_count = sum(
                1 for s in result.enpi_summaries
                if s.status in (KPIStatus.ON_TARGET, KPIStatus.AHEAD_OF_TARGET)
            )
            summary_parts.append(
                f"EnPI Performance: {on_target_count} of "
                f"{len(result.enpi_summaries)} EnPIs on or ahead of target."
            )

        if result.audit_summary:
            summary_parts.append(
                f"Audit Findings: {result.audit_summary.findings_count} total, "
                f"{result.audit_summary.open_ncs} open NCs, "
                f"{result.audit_summary.overdue_ncs} overdue."
            )

        if result.objectives_review:
            summary_parts.append(
                f"Objectives: {result.objectives_review.completed} of "
                f"{result.objectives_review.objectives_count} completed "
                f"({float(result.objectives_review.completion_rate_pct):.1f}%)."
            )

        summary_parts.append(
            f"Decisions: {len(result.decisions)} action items generated."
        )
        summary_parts.append(
            f"Next Review: {result.next_review_date}."
        )

        summary_notes = "\n".join(summary_parts)

        minutes = ManagementReviewMinutes(
            meeting_date=date.today(),
            attendees=attendees,
            chairperson=chairperson,
            review_period_start=result.review_period_start,
            review_period_end=result.review_period_end,
            agenda_items=agenda_items,
            summary_notes=summary_notes,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Generated minutes: attendees=%d, agenda_items=%d (%.1f ms)",
            len(attendees), len(agenda_items), elapsed_ms,
        )
        return minutes

    def generate_kpi_dashboard(
        self,
        enpi_summaries: List[EnPISummary],
    ) -> Dict[str, Any]:
        """Generate KPI dashboard data for UI rendering.

        Produces KPI cards, traffic light indicators, trend sparkline
        data, and aggregate performance metrics suitable for front-end
        dashboard rendering.

        Args:
            enpi_summaries: List of EnPI summaries.

        Returns:
            Dictionary with dashboard data structure.
        """
        t0 = time.perf_counter()

        # KPI cards
        kpi_cards: List[Dict[str, Any]] = []
        for summary in enpi_summaries:
            card: Dict[str, Any] = {
                "enpi_id": summary.enpi_id,
                "name": summary.enpi_name,
                "current_value": str(_round_val(summary.current_value, 2)),
                "baseline_value": str(_round_val(summary.baseline_value, 2)),
                "target_value": str(_round_val(summary.target_value, 2)),
                "improvement_pct": str(_round_val(summary.improvement_pct, 2)),
                "unit": summary.unit,
                "status": summary.status.value,
                "trend": summary.trend,
                "data_quality_score": str(
                    _round_val(summary.data_quality_score, 0)
                ),
            }
            kpi_cards.append(card)

        # Traffic light indicators
        traffic_lights: Dict[str, int] = {
            "green": 0,
            "amber": 0,
            "red": 0,
            "grey": 0,
        }
        for summary in enpi_summaries:
            colour = self._status_to_colour(summary.status)
            traffic_lights[colour] = traffic_lights.get(colour, 0) + 1

        # Trend sparkline data (improvement_pct per EnPI)
        sparklines: List[Dict[str, Any]] = []
        for summary in enpi_summaries:
            sparkline: Dict[str, Any] = {
                "enpi_name": summary.enpi_name,
                "data_points": [
                    str(_round_val(summary.baseline_value, 2)),
                    str(_round_val(summary.current_value, 2)),
                    str(_round_val(summary.target_value, 2)),
                ],
                "labels": ["Baseline", "Current", "Target"],
                "status": summary.status.value,
            }
            sparklines.append(sparkline)

        # Aggregate metrics
        total_enpis = len(enpi_summaries)
        if total_enpis > 0:
            avg_improvement = _safe_divide(
                sum(s.improvement_pct for s in enpi_summaries),
                _decimal(total_enpis),
            )
            avg_dq = _safe_divide(
                sum(s.data_quality_score for s in enpi_summaries),
                _decimal(total_enpis),
            )
        else:
            avg_improvement = Decimal("0")
            avg_dq = Decimal("0")

        aggregate: Dict[str, Any] = {
            "total_enpis": total_enpis,
            "average_improvement_pct": str(_round_val(avg_improvement, 2)),
            "average_data_quality": str(_round_val(avg_dq, 2)),
            "on_target_count": traffic_lights["green"],
            "at_risk_count": traffic_lights["red"],
            "behind_target_count": traffic_lights["amber"],
            "no_data_count": traffic_lights["grey"],
        }

        dashboard: Dict[str, Any] = {
            "dashboard_id": _new_uuid(),
            "generated_at": _utcnow().isoformat(),
            "kpi_cards": kpi_cards,
            "traffic_lights": traffic_lights,
            "sparklines": sparklines,
            "aggregate": aggregate,
            "engine_version": self.engine_version,
        }

        # Provenance
        dashboard["provenance_hash"] = _compute_hash(dashboard)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "KPI dashboard generated: %d cards, "
            "green=%d, amber=%d, red=%d (%.1f ms)",
            len(kpi_cards),
            traffic_lights["green"],
            traffic_lights["amber"],
            traffic_lights["red"],
            elapsed_ms,
        )
        return dashboard

    def calculate_next_review_date(
        self,
        current_date: date,
        frequency: ReviewFrequency,
    ) -> date:
        """Calculate the next management review date.

        Args:
            current_date: Reference date (typically period end).
            frequency: Review frequency setting.

        Returns:
            Calculated next review date.
        """
        frequency_days: Dict[str, int] = {
            ReviewFrequency.QUARTERLY.value: 91,
            ReviewFrequency.SEMI_ANNUAL.value: 182,
            ReviewFrequency.ANNUAL.value: 365,
            ReviewFrequency.AS_NEEDED.value: 182,
        }
        days = frequency_days.get(frequency.value, 365)
        next_date = current_date + timedelta(days=days)
        logger.debug(
            "Next review date: %s + %d days = %s (frequency=%s)",
            current_date, days, next_date, frequency.value,
        )
        return next_date

    def validate_review_completeness(
        self,
        result: ManagementReviewResult,
    ) -> Dict[str, Any]:
        """Validate that the review covers all ISO 50001 Clause 9.3 requirements.

        Checks that all required inputs (Clause 9.3.2) are present and all
        required outputs (Clause 9.3.3) are addressed by decisions.

        Args:
            result: The management review result to validate.

        Returns:
            Dictionary with completeness validation results.
        """
        t0 = time.perf_counter()

        # Validate inputs (Clause 9.3.2)
        input_categories_present = {
            inp.input_category for inp in result.inputs
        }
        required_input_categories = {
            req["category"] for req in ISO_50001_REVIEW_INPUTS
        }
        missing_inputs = required_input_categories - input_categories_present
        input_coverage_pct = _safe_pct(
            _decimal(len(input_categories_present & required_input_categories)),
            _decimal(len(required_input_categories)),
        )

        # Check input data completeness (not just presence)
        inputs_with_data = sum(
            1 for inp in result.inputs
            if inp.data_summary and inp.status != "no_data"
        )
        input_data_completeness = _safe_pct(
            _decimal(inputs_with_data), _decimal(len(result.inputs)),
        )

        # Validate outputs (Clause 9.3.3)
        output_coverage: Dict[str, bool] = {}
        for output_req in ISO_50001_REVIEW_OUTPUTS:
            cat = output_req["category"]
            is_covered = self._is_output_covered(cat, result)
            output_coverage[cat] = is_covered

        covered_outputs = sum(1 for v in output_coverage.values() if v)
        output_coverage_pct = _safe_pct(
            _decimal(covered_outputs),
            _decimal(len(ISO_50001_REVIEW_OUTPUTS)),
        )

        # Check structural completeness
        structural_checks: Dict[str, bool] = {
            "has_policy_review": result.policy_review is not None,
            "has_objectives_review": result.objectives_review is not None,
            "has_enpi_summaries": len(result.enpi_summaries) > 0,
            "has_resource_review": result.resource_review is not None,
            "has_audit_summary": result.audit_summary is not None,
            "has_decisions": len(result.decisions) > 0,
            "has_next_review_date": result.next_review_date is not None,
        }

        structural_score = _safe_pct(
            _decimal(sum(1 for v in structural_checks.values() if v)),
            _decimal(len(structural_checks)),
        )

        # Overall completeness
        is_complete = (
            len(missing_inputs) == 0
            and all(output_coverage.values())
            and all(structural_checks.values())
        )

        validation: Dict[str, Any] = {
            "is_complete": is_complete,
            "input_coverage_pct": str(_round_val(input_coverage_pct, 2)),
            "input_data_completeness_pct": str(
                _round_val(input_data_completeness, 2)
            ),
            "missing_inputs": sorted(missing_inputs),
            "output_coverage": output_coverage,
            "output_coverage_pct": str(_round_val(output_coverage_pct, 2)),
            "structural_checks": structural_checks,
            "structural_score_pct": str(_round_val(structural_score, 2)),
            "total_inputs": len(result.inputs),
            "total_decisions": len(result.decisions),
            "total_improvements": len(result.improvements),
            "validated_at": _utcnow().isoformat(),
            "engine_version": self.engine_version,
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Completeness validation: complete=%s, "
            "input_coverage=%.1f%%, output_coverage=%.1f%%, "
            "structural=%.1f%% (%.1f ms)",
            is_complete,
            float(input_coverage_pct),
            float(output_coverage_pct),
            float(structural_score),
            elapsed_ms,
        )
        return validation

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _assess_input_status(
        self,
        category: str,
        data_summary: Dict[str, Any],
    ) -> str:
        """Assess the status of a review input based on data availability.

        Args:
            category: Input category.
            data_summary: Data summary for the input.

        Returns:
            Status string: 'complete', 'partial', 'pending', or 'no_data'.
        """
        if not data_summary:
            return "no_data"

        # Count non-empty fields
        non_empty = sum(
            1 for v in data_summary.values()
            if v is not None and v != "" and v != 0 and v != []
        )
        total = len(data_summary)

        if total == 0:
            return "no_data"

        completeness = _safe_divide(
            _decimal(non_empty), _decimal(total),
        )

        if completeness >= Decimal("0.8"):
            return "complete"
        elif completeness >= Decimal("0.4"):
            return "partial"
        elif completeness > Decimal("0"):
            return "pending"
        return "no_data"

    def _calculate_improvement(
        self,
        baseline: Decimal,
        current: Decimal,
        enpi_type: str,
    ) -> Decimal:
        """Calculate improvement percentage between baseline and current.

        For ratio/absolute EnPIs, improvement is measured as reduction
        (baseline - current) / baseline * 100.  For regression-based
        EnPIs, the same formula applies to the predicted vs actual.

        Args:
            baseline: Baseline value.
            current: Current value.
            enpi_type: Type of EnPI.

        Returns:
            Improvement percentage (positive = improvement).
        """
        if baseline == Decimal("0") and current == Decimal("0"):
            return Decimal("0")
        if baseline == Decimal("0"):
            return Decimal("0")

        # For energy intensity / consumption metrics, lower is better
        improvement = _safe_divide(
            (baseline - current) * Decimal("100"),
            baseline,
        )
        return improvement

    def _assign_kpi_status(
        self,
        improvement_pct: Decimal,
        target_improvement: Decimal,
        baseline: Decimal,
        current: Decimal,
    ) -> KPIStatus:
        """Assign KPI status using threshold-based classification.

        Args:
            improvement_pct: Actual improvement percentage.
            target_improvement: Target improvement percentage.
            baseline: Baseline value.
            current: Current value.

        Returns:
            KPIStatus classification.
        """
        # No data case
        if baseline == Decimal("0") and current == Decimal("0"):
            return KPIStatus.NO_DATA

        # If no target set, use simple positive/negative check
        if target_improvement == Decimal("0"):
            if improvement_pct > Decimal("0"):
                return KPIStatus.ON_TARGET
            elif improvement_pct < Decimal("0"):
                return KPIStatus.BEHIND_TARGET
            return KPIStatus.ON_TARGET

        # Threshold-based classification
        ahead_threshold = target_improvement * KPI_STATUS_THRESHOLDS[
            "ahead_of_target"
        ]
        on_target_threshold = target_improvement * KPI_STATUS_THRESHOLDS[
            "on_target_lower"
        ]
        behind_threshold = target_improvement * KPI_STATUS_THRESHOLDS[
            "behind_target_lower"
        ]

        if improvement_pct >= ahead_threshold:
            return KPIStatus.AHEAD_OF_TARGET
        elif improvement_pct >= on_target_threshold:
            return KPIStatus.ON_TARGET
        elif improvement_pct >= behind_threshold:
            return KPIStatus.BEHIND_TARGET
        return KPIStatus.AT_RISK

    def _determine_trend(
        self,
        improvement_pct: Decimal,
        target_improvement: Decimal,
    ) -> str:
        """Determine trend direction based on improvement vs target.

        Args:
            improvement_pct: Actual improvement percentage.
            target_improvement: Target improvement percentage.

        Returns:
            Trend string: 'improving', 'stable', or 'declining'.
        """
        if target_improvement == Decimal("0"):
            if improvement_pct > Decimal("5"):
                return "improving"
            elif improvement_pct < Decimal("-5"):
                return "declining"
            return "stable"

        ratio = _safe_divide(improvement_pct, target_improvement)
        if ratio > Decimal("1.05"):
            return "improving"
        elif ratio < Decimal("0.80"):
            return "declining"
        return "stable"

    def _determine_adequacy(
        self,
        utilization_pct: Decimal,
        resource_data: Dict[str, Any],
    ) -> ResourceAdequacy:
        """Determine resource adequacy based on utilisation and context.

        Args:
            utilization_pct: Current utilisation percentage.
            resource_data: Raw resource data for additional context.

        Returns:
            ResourceAdequacy classification.
        """
        # If explicitly provided, use that
        explicit = resource_data.get("adequacy")
        if explicit and isinstance(explicit, str):
            valid = {s.value for s in ResourceAdequacy}
            if explicit in valid:
                return ResourceAdequacy(explicit)

        # Threshold-based determination
        if utilization_pct > Decimal("120"):
            return ResourceAdequacy.NEEDS_INCREASE
        elif utilization_pct > Decimal("100"):
            return ResourceAdequacy.NEEDS_REALLOCATION
        elif utilization_pct >= Decimal("50"):
            return ResourceAdequacy.ADEQUATE
        elif utilization_pct > Decimal("0"):
            return ResourceAdequacy.NEEDS_REALLOCATION
        return ResourceAdequacy.UNDER_REVIEW

    def _generate_resource_recommendation(
        self,
        adequacy: ResourceAdequacy,
        utilization_pct: Decimal,
        allocated: Decimal,
        spent: Decimal,
    ) -> str:
        """Generate a resource recommendation based on adequacy assessment.

        Args:
            adequacy: Determined adequacy level.
            utilization_pct: Current utilisation percentage.
            allocated: Allocated amount.
            spent: Spent amount.

        Returns:
            Recommendation text.
        """
        if adequacy == ResourceAdequacy.ADEQUATE:
            return (
                f"Resources are adequate at {float(utilization_pct):.1f}% "
                f"utilisation. Continue current allocation level."
            )
        elif adequacy == ResourceAdequacy.NEEDS_INCREASE:
            shortfall = spent - allocated
            return (
                f"Resources overspent by {float(shortfall):.2f} "
                f"({float(utilization_pct):.1f}% utilisation). "
                f"Recommend increasing allocation by at least "
                f"{float(shortfall * Decimal('1.15')):.2f} to provide buffer."
            )
        elif adequacy == ResourceAdequacy.NEEDS_REALLOCATION:
            return (
                f"Resource utilisation at {float(utilization_pct):.1f}%. "
                f"Recommend reviewing allocation across categories "
                f"to optimise deployment."
            )
        return (
            "Resource adequacy is under review. Recommend completing "
            "resource assessment before next management review."
        )

    def _suggest_enpi_decisions(
        self,
        enpi_summaries: List[EnPISummary],
    ) -> List[ReviewDecision]:
        """Suggest decisions for EnPI performance issues.

        Args:
            enpi_summaries: EnPI performance summaries.

        Returns:
            List of suggested ReviewDecision objects.
        """
        decisions: List[ReviewDecision] = []

        at_risk = [
            s for s in enpi_summaries
            if s.status == KPIStatus.AT_RISK
        ]
        behind = [
            s for s in enpi_summaries
            if s.status == KPIStatus.BEHIND_TARGET
        ]

        if at_risk:
            names = ", ".join(s.enpi_name for s in at_risk[:5])
            decisions.append(ReviewDecision(
                decision_type=DecisionType.PROCESS_IMPROVEMENT,
                description=(
                    f"Investigate and address EnPI performance shortfall "
                    f"for at-risk indicators: {names}"
                ),
                rationale=(
                    f"{len(at_risk)} EnPI(s) classified as AT_RISK. "
                    f"Corrective action required per ISO 50001 Clause 10.1 "
                    f"to restore energy performance."
                ),
                due_date=date.today() + timedelta(days=30),
                priority=ActionItemPriority.HIGH,
                clause_ref="9.3.3 g)",
            ))

        if behind:
            names = ", ".join(s.enpi_name for s in behind[:5])
            decisions.append(ReviewDecision(
                decision_type=DecisionType.TARGET_REVISION,
                description=(
                    f"Evaluate target revision or additional improvement "
                    f"actions for behind-target indicators: {names}"
                ),
                rationale=(
                    f"{len(behind)} EnPI(s) classified as BEHIND_TARGET. "
                    f"Review targets and action plans per "
                    f"ISO 50001 Clause 6.2."
                ),
                due_date=date.today() + timedelta(days=60),
                priority=ActionItemPriority.MEDIUM,
                clause_ref="9.3.3 g)",
            ))

        return decisions

    def _suggest_audit_decisions(
        self,
        audit_summary: AuditSummary,
    ) -> List[ReviewDecision]:
        """Suggest decisions based on audit findings.

        Args:
            audit_summary: Internal audit summary.

        Returns:
            List of suggested ReviewDecision objects.
        """
        decisions: List[ReviewDecision] = []

        if audit_summary.overdue_ncs > 0:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.PROCESS_IMPROVEMENT,
                description=(
                    f"Address {audit_summary.overdue_ncs} overdue "
                    f"nonconformit{'y' if audit_summary.overdue_ncs == 1 else 'ies'} "
                    f"from internal audits"
                ),
                rationale=(
                    f"Overdue NCs indicate systemic corrective action delays. "
                    f"ISO 50001 Clause 10.1 requires timely resolution."
                ),
                due_date=date.today() + timedelta(days=14),
                priority=ActionItemPriority.URGENT,
                clause_ref="9.3.3 b)",
            ))

        if audit_summary.major_ncs > 0:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.PROCESS_IMPROVEMENT,
                description=(
                    f"Resolve {audit_summary.major_ncs} major "
                    f"nonconformit{'y' if audit_summary.major_ncs == 1 else 'ies'} "
                    f"and conduct root cause analysis"
                ),
                rationale=(
                    f"Major NCs represent significant risk to EnMS "
                    f"effectiveness per ISO 50001 Clause 10.1."
                ),
                due_date=date.today() + timedelta(days=30),
                priority=ActionItemPriority.HIGH,
                clause_ref="9.3.3 b)",
            ))

        if audit_summary.open_ncs > 0 and audit_summary.overdue_ncs == 0:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.TRAINING_NEED,
                description=(
                    f"Review corrective action process for "
                    f"{audit_summary.open_ncs} open "
                    f"nonconformit{'y' if audit_summary.open_ncs == 1 else 'ies'}; "
                    f"assess training needs"
                ),
                rationale=(
                    f"Open NCs may indicate competence gaps; review "
                    f"training needs per ISO 50001 Clause 7.2."
                ),
                due_date=date.today() + timedelta(days=60),
                priority=ActionItemPriority.MEDIUM,
                clause_ref="9.3.3 f)",
            ))

        return decisions

    def _suggest_resource_decisions(
        self,
        resource_review: ResourceReview,
    ) -> List[ReviewDecision]:
        """Suggest decisions based on resource adequacy.

        Args:
            resource_review: Resource adequacy review.

        Returns:
            List of suggested ReviewDecision objects.
        """
        decisions: List[ReviewDecision] = []

        if resource_review.adequacy == ResourceAdequacy.NEEDS_INCREASE:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.RESOURCE_ALLOCATION,
                description=(
                    f"Increase {resource_review.category} resource allocation; "
                    f"current utilisation at "
                    f"{float(resource_review.utilization_pct):.1f}%"
                ),
                rationale=(
                    f"Resource utilisation exceeds allocation. "
                    f"Adequate resources required per ISO 50001 Clause 7.1."
                ),
                due_date=date.today() + timedelta(days=30),
                priority=ActionItemPriority.HIGH,
                clause_ref="9.3.3 e)",
            ))

        elif resource_review.adequacy == ResourceAdequacy.NEEDS_REALLOCATION:
            decisions.append(ReviewDecision(
                decision_type=DecisionType.RESOURCE_ALLOCATION,
                description=(
                    f"Reallocate {resource_review.category} resources; "
                    f"current utilisation at "
                    f"{float(resource_review.utilization_pct):.1f}%"
                ),
                rationale=(
                    f"Resource distribution is suboptimal. "
                    f"Reallocation recommended per ISO 50001 Clause 7.1."
                ),
                due_date=date.today() + timedelta(days=60),
                priority=ActionItemPriority.MEDIUM,
                clause_ref="9.3.3 e)",
            ))

        return decisions

    def _suggest_previous_action_decisions(
        self,
        inputs: List[ReviewInput],
    ) -> List[ReviewDecision]:
        """Suggest decisions for overdue previous actions.

        Args:
            inputs: Compiled review inputs.

        Returns:
            List of suggested ReviewDecision objects.
        """
        decisions: List[ReviewDecision] = []

        for inp in inputs:
            if inp.input_category != "previous_actions":
                continue

            items = inp.data_summary.get("items", [])
            if not isinstance(items, list):
                continue

            overdue = [
                i for i in items
                if isinstance(i, dict)
                and str(i.get("status", "")).lower() in ("overdue", "open")
            ]

            if overdue:
                decisions.append(ReviewDecision(
                    decision_type=DecisionType.PROCESS_IMPROVEMENT,
                    description=(
                        f"Expedite {len(overdue)} overdue action(s) from "
                        f"previous management review"
                    ),
                    rationale=(
                        f"Previous review actions remain incomplete. "
                        f"ISO 50001 Clause 9.3 requires status reporting."
                    ),
                    due_date=date.today() + timedelta(days=30),
                    priority=ActionItemPriority.HIGH,
                    clause_ref="9.3.3 b)",
                ))

        return decisions

    def _ensure_output_coverage(
        self,
        decisions: List[ReviewDecision],
    ) -> List[ReviewDecision]:
        """Ensure all ISO 50001 Clause 9.3.3 outputs are covered.

        Adds a NO_ACTION acknowledgement decision for any required
        output category not already addressed by an existing decision.

        Args:
            decisions: Current list of decisions.

        Returns:
            Updated list of decisions with full output coverage.
        """
        # Map clause refs to output categories
        covered_categories: set = set()
        for d in decisions:
            # Extract output categories from clause_ref
            for output_req in ISO_50001_REVIEW_OUTPUTS:
                ref = output_req["ref"]
                if d.clause_ref == ref:
                    covered_categories.add(output_req["category"])

        # Map decision types to output categories
        type_to_category: Dict[str, str] = {
            DecisionType.POLICY_CHANGE.value: "performance_shortfall",
            DecisionType.RESOURCE_ALLOCATION.value: "resource_allocation",
            DecisionType.TARGET_REVISION.value: "performance_shortfall",
            DecisionType.PROCESS_IMPROVEMENT.value: "improvement_decisions",
            DecisionType.TRAINING_NEED.value: "competence_awareness",
            DecisionType.INVESTMENT_APPROVAL.value: "resource_allocation",
        }
        for d in decisions:
            cat = type_to_category.get(d.decision_type.value)
            if cat:
                covered_categories.add(cat)

        # Also mark enms_effectiveness as covered if we have any decisions
        if decisions:
            covered_categories.add("enms_effectiveness")
            covered_categories.add("intended_outcomes")

        # Add acknowledgement decisions for uncovered outputs
        for output_req in ISO_50001_REVIEW_OUTPUTS:
            cat = output_req["category"]
            if cat not in covered_categories:
                decisions.append(ReviewDecision(
                    decision_type=DecisionType.NO_ACTION,
                    description=(
                        f"Acknowledged: {output_req['description']}. "
                        f"No action required at this time."
                    ),
                    rationale=(
                        f"Reviewed per ISO 50001 {output_req['ref']}; "
                        f"current status satisfactory."
                    ),
                    due_date=date.today() + timedelta(days=180),
                    priority=ActionItemPriority.LOW,
                    clause_ref=output_req["ref"],
                ))

        return decisions

    def _is_output_covered(
        self,
        category: str,
        result: ManagementReviewResult,
    ) -> bool:
        """Check whether an ISO 50001 9.3.3 output category is covered.

        Args:
            category: Output category to check.
            result: Management review result.

        Returns:
            True if the output category is covered.
        """
        # Map categories to what constitutes coverage
        if category == "enms_effectiveness":
            return (
                result.policy_review is not None
                and result.objectives_review is not None
                and len(result.enpi_summaries) > 0
            )
        elif category == "improvement_decisions":
            return any(
                d.decision_type in (
                    DecisionType.PROCESS_IMPROVEMENT,
                    DecisionType.TARGET_REVISION,
                )
                for d in result.decisions
            )
        elif category == "business_integration":
            return any(
                d.decision_type == DecisionType.PROCESS_IMPROVEMENT
                for d in result.decisions
            ) or any(
                d.clause_ref == "9.3.3 c)"
                for d in result.decisions
            )
        elif category == "intended_outcomes":
            return (
                result.objectives_review is not None
                and len(result.enpi_summaries) > 0
            )
        elif category == "resource_allocation":
            return (
                result.resource_review is not None
                or any(
                    d.decision_type == DecisionType.RESOURCE_ALLOCATION
                    for d in result.decisions
                )
            )
        elif category == "competence_awareness":
            return any(
                d.decision_type == DecisionType.TRAINING_NEED
                for d in result.decisions
            ) or any(
                d.clause_ref == "9.3.3 f)"
                for d in result.decisions
            )
        elif category == "performance_shortfall":
            return any(
                d.decision_type in (
                    DecisionType.POLICY_CHANGE,
                    DecisionType.TARGET_REVISION,
                )
                for d in result.decisions
            ) or any(
                d.clause_ref == "9.3.3 g)"
                for d in result.decisions
            )
        return False

    def _status_to_colour(self, status: KPIStatus) -> str:
        """Map KPI status to traffic light colour.

        Args:
            status: KPI status.

        Returns:
            Colour string: 'green', 'amber', 'red', or 'grey'.
        """
        colour_map: Dict[KPIStatus, str] = {
            KPIStatus.AHEAD_OF_TARGET: "green",
            KPIStatus.ON_TARGET: "green",
            KPIStatus.BEHIND_TARGET: "amber",
            KPIStatus.AT_RISK: "red",
            KPIStatus.NO_DATA: "grey",
        }
        return colour_map.get(status, "grey")

    def _parse_date(self, value: Any) -> date:
        """Parse a date from various formats.

        Args:
            value: Date value (str, date, or datetime).

        Returns:
            Parsed date object.
        """
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        logger.warning("Could not parse date value: %s; using today", value)
        return date.today()
