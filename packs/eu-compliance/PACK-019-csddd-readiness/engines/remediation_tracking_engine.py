# -*- coding: utf-8 -*-
"""
RemediationTrackingEngine - PACK-019 CSDDD Remediation Tracking Engine
=======================================================================

Tracks and assesses remediation actions taken by companies in response
to actual adverse human rights and environmental impacts, in accordance
with CSDDD Article 10.

The EU Corporate Sustainability Due Diligence Directive (CSDDD /
Directive 2024/1760) requires companies that have caused or
contributed to an actual adverse impact to provide remediation.
Where the company has not caused or contributed to the impact but
it is directly linked to the company's operations, products, or
services through a business relationship, the company shall use
its leverage to seek remediation from the entity causing the impact.

CSDDD Art 10 - Remediation:
    - Para 1: Companies shall provide remediation where they have
      caused or jointly caused an actual adverse impact.
    - Para 2: Where the company has not caused or jointly caused
      the impact but it has contributed to it, the company shall
      provide remediation proportionate to its contribution.
    - Para 3: Where the company has not caused or contributed to
      the adverse impact, but it is directly linked via business
      relationships, the company shall use its leverage.
    - Para 4: Remediation shall be proportionate to the company's
      contribution to the adverse impact.
    - Para 5: Companies shall offer to engage with affected persons
      or their representatives and shall enable remediation through
      their complaints procedure (Art 12).

Types of Remediation (per UNGPs and OECD Guidance):
    - Financial compensation
    - Restitution (restoring the situation)
    - Rehabilitation (medical, psychological, legal support)
    - Guarantees of non-repetition
    - Public or private apology
    - Operational changes to prevent recurrence

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D), Article 10
    - UN Guiding Principles on Business and Human Rights (Principle 22)
    - OECD Guidelines for Multinational Enterprises (Chapter IV)
    - OECD Due Diligence Guidance for Responsible Business Conduct
    - UN Basic Principles on Right to Remedy (GA Res. 60/147)

Zero-Hallucination:
    - Completion rates use count-based arithmetic
    - Financial aggregations use Decimal precision
    - Timeline analysis uses date comparisons
    - Engagement scoring uses boolean counts
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-019 CSDDD Readiness
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )


def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RemediationStatus(str, Enum):
    """Status of a remediation action under CSDDD Art 10.

    Tracks the lifecycle of a remediation action from initiation
    through verification of completion.

    - NOT_STARTED: Action defined but not yet initiated
    - IN_PROGRESS: Action is actively being implemented
    - COMPLETED: Action has been completed
    - VERIFIED: Completion verified by independent party or victim
    - FAILED: Action failed to achieve intended outcome
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


class RemediationType(str, Enum):
    """Type of remediation per UNGPs and OECD Guidance.

    Aligns with the UN Basic Principles on the Right to a Remedy
    and Reparation (GA Res. 60/147) and UNGP Principle 25.

    - FINANCIAL_COMPENSATION: Monetary payment for damages
    - RESTITUTION: Restoring the situation to its original state
    - REHABILITATION: Medical, psychological, or legal support
    - GUARANTEE_NON_REPETITION: Structural changes to prevent recurrence
    - APOLOGY: Public or private acknowledgment of harm
    - OPERATIONAL_CHANGE: Changes to operations, processes, or policies
    """
    FINANCIAL_COMPENSATION = "financial_compensation"
    RESTITUTION = "restitution"
    REHABILITATION = "rehabilitation"
    GUARANTEE_NON_REPETITION = "guarantee_non_repetition"
    APOLOGY = "apology"
    OPERATIONAL_CHANGE = "operational_change"


class CompanyContribution(str, Enum):
    """Level of company contribution to the adverse impact.

    Per CSDDD Art 10 Para 1-3, the nature and extent of remediation
    depends on whether the company caused, contributed to, or is
    directly linked to the adverse impact.
    """
    CAUSED = "caused"
    JOINTLY_CAUSED = "jointly_caused"
    CONTRIBUTED = "contributed"
    DIRECTLY_LINKED = "directly_linked"


class VictimEngagementLevel(str, Enum):
    """Level of engagement with affected persons / victims.

    Per CSDDD Art 10 Para 5, companies shall offer to engage with
    affected persons or their representatives during remediation.
    """
    NONE = "none"
    NOTIFIED = "notified"
    CONSULTED = "consulted"
    ACTIVELY_ENGAGED = "actively_engaged"
    CO_DESIGNED = "co_designed"


class ImpactDomain(str, Enum):
    """Domain of the adverse impact being remediated."""
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENTAL = "environmental"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Victim engagement score mapping
ENGAGEMENT_SCORES: Dict[str, Decimal] = {
    VictimEngagementLevel.NONE.value: Decimal("0"),
    VictimEngagementLevel.NOTIFIED.value: Decimal("25"),
    VictimEngagementLevel.CONSULTED.value: Decimal("50"),
    VictimEngagementLevel.ACTIVELY_ENGAGED.value: Decimal("75"),
    VictimEngagementLevel.CO_DESIGNED.value: Decimal("100"),
}

# Company contribution weightings for proportionality assessment
CONTRIBUTION_WEIGHTS: Dict[str, Decimal] = {
    CompanyContribution.CAUSED.value: Decimal("1.0"),
    CompanyContribution.JOINTLY_CAUSED.value: Decimal("0.75"),
    CompanyContribution.CONTRIBUTED.value: Decimal("0.5"),
    CompanyContribution.DIRECTLY_LINKED.value: Decimal("0.25"),
}

# Completeness criteria for remediation actions
REMEDIATION_COMPLETENESS_CRITERIA: List[str] = [
    "has_description",
    "has_type",
    "has_financial_provision",
    "has_victim_engagement",
    "has_start_date",
    "has_target_completion",
    "has_responsible_person",
    "has_evidence",
    "has_grievance_link",
    "has_contribution_level",
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RemediationAction(BaseModel):
    """A single remediation action per CSDDD Art 10.

    Represents one remediation measure taken or planned by the
    company in response to an actual adverse impact.  Includes
    financial provisions, victim engagement, timelines, and
    evidence of implementation.

    Attributes:
        action_id: Unique identifier for this remediation action.
        adverse_impact_id: Reference to the adverse impact being remediated.
        remediation_type: Type of remediation (compensation, restitution, etc.).
        company_contribution: Level of company contribution to the impact.
        description: Narrative description of the remediation action.
        impact_domain: Whether the impact is human rights, environmental, or both.
        financial_provision_eur: Financial provision in EUR.
        financial_disbursed_eur: Amount actually disbursed in EUR.
        victim_engagement: Level of engagement with affected persons.
        victim_count: Number of affected persons / victims.
        victims_reached: Number of victims actually reached by remediation.
        completion_status: Current status of the remediation action.
        start_date: Actual or planned start date.
        target_completion: Target completion date.
        actual_completion: Actual completion date (if completed/verified).
        responsible_person: Person or role responsible.
        evidence: Supporting evidence or documentation references.
        grievance_mechanism_used: Whether the grievance mechanism was used.
        third_party_involved: Whether a third party is involved.
        third_party_name: Name of the third party (if applicable).
        country: ISO 3166-1 alpha-2 country code.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this remediation action",
    )
    adverse_impact_id: str = Field(
        ...,
        description="Reference to the adverse impact being remediated",
    )
    remediation_type: RemediationType = Field(
        ...,
        description="Type of remediation action",
    )
    company_contribution: CompanyContribution = Field(
        default=CompanyContribution.CONTRIBUTED,
        description="Level of company contribution to the adverse impact",
    )
    description: str = Field(
        default="",
        description="Narrative description of the remediation action",
        max_length=5000,
    )
    impact_domain: ImpactDomain = Field(
        default=ImpactDomain.BOTH,
        description="Domain of the adverse impact being remediated",
    )
    financial_provision_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total financial provision in EUR",
        ge=Decimal("0"),
    )
    financial_disbursed_eur: Decimal = Field(
        default=Decimal("0"),
        description="Amount actually disbursed in EUR",
        ge=Decimal("0"),
    )
    victim_engagement: VictimEngagementLevel = Field(
        default=VictimEngagementLevel.NONE,
        description="Level of engagement with affected persons",
    )
    victim_count: int = Field(
        default=0,
        description="Estimated number of affected persons / victims",
        ge=0,
    )
    victims_reached: int = Field(
        default=0,
        description="Number of victims actually reached by remediation",
        ge=0,
    )
    completion_status: RemediationStatus = Field(
        default=RemediationStatus.NOT_STARTED,
        description="Current status of the remediation action",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Actual or planned start date",
    )
    target_completion: Optional[datetime] = Field(
        default=None,
        description="Target completion date",
    )
    actual_completion: Optional[datetime] = Field(
        default=None,
        description="Actual completion date",
    )
    responsible_person: str = Field(
        default="",
        description="Person or role responsible for implementation",
        max_length=500,
    )
    responsible_department: str = Field(
        default="",
        description="Department responsible",
        max_length=200,
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence or documentation references",
    )
    grievance_mechanism_used: bool = Field(
        default=False,
        description="Whether the grievance mechanism (Art 12) was used",
    )
    third_party_involved: bool = Field(
        default=False,
        description="Whether a third party is involved in remediation",
    )
    third_party_name: str = Field(
        default="",
        description="Name of the third party (if applicable)",
        max_length=500,
    )
    country: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )
    is_proportionate: Optional[bool] = Field(
        default=None,
        description="Whether the remediation is proportionate to contribution",
    )
    outcome_description: str = Field(
        default="",
        description="Description of the remediation outcome",
        max_length=5000,
    )
    lessons_learned: str = Field(
        default="",
        description="Lessons learned from this remediation process",
        max_length=2000,
    )

    @field_validator("country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alphabetic."""
        if v and not v.isalpha():
            raise ValueError("Country code must be alphabetic")
        return v.upper()


class TimelineAnalysis(BaseModel):
    """Timeline analysis for remediation actions.

    Assesses timeliness of remediation delivery including
    on-time completion, average duration, and overdue actions.
    """
    total_actions: int = Field(
        default=0,
        description="Total remediation actions",
        ge=0,
    )
    actions_with_timeline: int = Field(
        default=0,
        description="Actions with both start and target dates",
        ge=0,
    )
    completed_on_time: int = Field(
        default=0,
        description="Actions completed on or before target date",
        ge=0,
    )
    completed_late: int = Field(
        default=0,
        description="Actions completed after target date",
        ge=0,
    )
    on_time_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of completed actions that were on time",
    )
    overdue_count: int = Field(
        default=0,
        description="Actions past target date and not completed",
        ge=0,
    )
    average_duration_days: Decimal = Field(
        default=Decimal("0"),
        description="Average duration in days for completed actions",
    )
    average_delay_days: Decimal = Field(
        default=Decimal("0"),
        description="Average delay in days for late completions",
    )
    earliest_start: Optional[datetime] = Field(
        default=None,
        description="Earliest start date across all actions",
    )
    latest_target: Optional[datetime] = Field(
        default=None,
        description="Latest target completion date",
    )


class FinancialAnalysis(BaseModel):
    """Financial analysis of remediation provisions and disbursements.

    Tracks total provisions, disbursements, utilisation rates,
    and breakdowns by remediation type and company contribution.
    """
    total_financial_provision_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total financial provision across all actions",
    )
    total_disbursed_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total amount disbursed",
    )
    disbursement_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of provisions disbursed",
    )
    by_remediation_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financial provisions by remediation type",
    )
    by_contribution_level: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financial provisions by company contribution level",
    )
    by_impact_domain: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financial provisions by impact domain",
    )
    actions_with_financial_provision: int = Field(
        default=0,
        description="Count of actions with financial provisions",
        ge=0,
    )
    average_provision_per_action_eur: Decimal = Field(
        default=Decimal("0"),
        description="Average financial provision per action",
    )
    average_provision_per_victim_eur: Decimal = Field(
        default=Decimal("0"),
        description="Average financial provision per affected person",
    )


class VictimEngagementAnalysis(BaseModel):
    """Analysis of victim / affected person engagement.

    Assesses the depth and breadth of engagement with affected
    persons throughout the remediation process per Art 10 Para 5.
    """
    total_actions: int = Field(
        default=0,
        description="Total remediation actions",
        ge=0,
    )
    actions_with_engagement: int = Field(
        default=0,
        description="Actions with any victim engagement",
        ge=0,
    )
    engagement_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of actions with victim engagement",
    )
    by_engagement_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions by engagement level",
    )
    average_engagement_score: Decimal = Field(
        default=Decimal("0"),
        description="Average engagement score (0-100)",
    )
    total_victims: int = Field(
        default=0,
        description="Total estimated victims across all actions",
        ge=0,
    )
    total_victims_reached: int = Field(
        default=0,
        description="Total victims reached by remediation",
        ge=0,
    )
    victim_reach_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of victims reached",
    )
    grievance_mechanism_used_count: int = Field(
        default=0,
        description="Actions using the grievance mechanism",
        ge=0,
    )
    grievance_mechanism_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of actions using grievance mechanism",
    )


class CompletenessAssessment(BaseModel):
    """Completeness assessment for remediation action documentation.

    Evaluates whether each action has all required elements
    for CSDDD Art 10 compliance.
    """
    action_id: str = Field(
        ...,
        description="Remediation action ID",
    )
    criteria_met: int = Field(
        default=0,
        description="Number of completeness criteria met",
        ge=0,
    )
    criteria_total: int = Field(
        default=0,
        description="Total completeness criteria",
        ge=0,
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Completeness percentage",
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of missing elements",
    )


class RemediationResult(BaseModel):
    """Complete remediation tracking assessment result.

    Aggregates all remediation actions, financial analysis, victim
    engagement, timeline analysis, effectiveness scoring, and
    recommendations into a single auditable result.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    total_actions: int = Field(
        default=0,
        description="Total remediation actions",
        ge=0,
    )
    actions_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions by remediation type",
    )
    actions_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions by status",
    )
    actions_by_contribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of actions by company contribution level",
    )
    total_financial_provision_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total financial provision across all actions",
    )
    completion_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of actions completed or verified",
    )
    victim_engagement_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of actions with victim engagement",
    )
    effectiveness_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall effectiveness score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    financial_analysis: FinancialAnalysis = Field(
        default_factory=FinancialAnalysis,
        description="Financial analysis of provisions and disbursements",
    )
    victim_engagement_analysis: VictimEngagementAnalysis = Field(
        default_factory=VictimEngagementAnalysis,
        description="Victim engagement analysis",
    )
    timeline_analysis: TimelineAnalysis = Field(
        default_factory=TimelineAnalysis,
        description="Timeline analysis of remediation actions",
    )
    completeness_assessments: List[CompletenessAssessment] = Field(
        default_factory=list,
        description="Per-action completeness assessments",
    )
    average_completeness_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Average completeness across all actions",
    )
    impacts_with_remediation: int = Field(
        default=0,
        description="Number of unique impacts with remediation",
        ge=0,
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritised recommendations",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp (UTC)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail provenance",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RemediationTrackingEngine:
    """CSDDD Remediation Tracking assessment engine.

    Provides deterministic, zero-hallucination assessment of
    remediation actions per CSDDD Article 10.

    The engine evaluates:
    1. **Completion rate** of remediation actions across all statuses.
    2. **Victim engagement** depth and breadth per Art 10 Para 5.
    3. **Financial provisions** and disbursement analysis.
    4. **Timeline compliance** and overdue action tracking.
    5. **Completeness** of each action's documentation.
    6. **Overall effectiveness** score combining multiple factors.

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = RemediationTrackingEngine()
        actions = [RemediationAction(...), ...]
        impact_ids = ["impact-001", "impact-002", ...]
        result = engine.assess_remediation(actions, impact_ids)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Completion Rate                                                      #
    # ------------------------------------------------------------------ #

    def calculate_completion_rate(
        self, actions: List[RemediationAction]
    ) -> Decimal:
        """Calculate the percentage of actions that are completed or verified.

        Actions with status COMPLETED or VERIFIED count as complete.

        Args:
            actions: List of RemediationAction instances.

        Returns:
            Completion rate as percentage (0-100).
        """
        if not actions:
            return Decimal("0.0")

        complete_count = sum(
            1 for a in actions
            if a.completion_status in (
                RemediationStatus.COMPLETED,
                RemediationStatus.VERIFIED,
            )
        )
        return _pct(complete_count, len(actions))

    # ------------------------------------------------------------------ #
    # Victim Engagement                                                    #
    # ------------------------------------------------------------------ #

    def assess_victim_engagement(
        self, actions: List[RemediationAction]
    ) -> VictimEngagementAnalysis:
        """Assess the depth and breadth of victim engagement.

        Evaluates engagement levels across all remediation actions,
        computing engagement rate, average engagement score, victim
        reach rate, and grievance mechanism usage.

        Args:
            actions: List of RemediationAction instances.

        Returns:
            VictimEngagementAnalysis with all metrics.
        """
        if not actions:
            return VictimEngagementAnalysis()

        n = len(actions)

        # Actions with any engagement (above NONE)
        engaged = sum(
            1 for a in actions
            if a.victim_engagement != VictimEngagementLevel.NONE
        )
        engagement_rate = _pct(engaged, n)

        # Distribution by engagement level
        by_level: Dict[str, int] = {}
        for vel in VictimEngagementLevel:
            by_level[vel.value] = sum(
                1 for a in actions if a.victim_engagement == vel
            )

        # Average engagement score
        total_score = Decimal("0")
        for a in actions:
            score = ENGAGEMENT_SCORES.get(
                a.victim_engagement.value, Decimal("0")
            )
            total_score += score
        avg_score = _round_val(total_score / _decimal(n), 1)

        # Victim reach
        total_victims = sum(a.victim_count for a in actions)
        total_reached = sum(a.victims_reached for a in actions)
        reach_rate = _pct(total_reached, total_victims) if total_victims > 0 else Decimal("0.0")

        # Grievance mechanism usage
        gm_count = sum(
            1 for a in actions if a.grievance_mechanism_used
        )
        gm_pct = _pct(gm_count, n)

        return VictimEngagementAnalysis(
            total_actions=n,
            actions_with_engagement=engaged,
            engagement_rate_pct=engagement_rate,
            by_engagement_level=by_level,
            average_engagement_score=avg_score,
            total_victims=total_victims,
            total_victims_reached=total_reached,
            victim_reach_rate_pct=reach_rate,
            grievance_mechanism_used_count=gm_count,
            grievance_mechanism_pct=gm_pct,
        )

    # ------------------------------------------------------------------ #
    # Financial Analysis                                                   #
    # ------------------------------------------------------------------ #

    def analyze_financial_provisions(
        self, actions: List[RemediationAction]
    ) -> FinancialAnalysis:
        """Analyse financial provisions and disbursements.

        Aggregates financial data across all remediation actions,
        computing totals, rates, and breakdowns by type, contribution,
        and impact domain.

        Args:
            actions: List of RemediationAction instances.

        Returns:
            FinancialAnalysis with all financial metrics.
        """
        if not actions:
            return FinancialAnalysis()

        # Totals
        total_provision = sum(a.financial_provision_eur for a in actions)
        total_disbursed = sum(a.financial_disbursed_eur for a in actions)
        disbursement_rate = _pct_dec(total_disbursed, total_provision)

        # By remediation type
        by_type: Dict[str, Decimal] = {}
        for rt in RemediationType:
            type_provision = sum(
                a.financial_provision_eur for a in actions
                if a.remediation_type == rt
            )
            if type_provision > Decimal("0"):
                by_type[rt.value] = type_provision

        # By contribution level
        by_contribution: Dict[str, Decimal] = {}
        for cc in CompanyContribution:
            cc_provision = sum(
                a.financial_provision_eur for a in actions
                if a.company_contribution == cc
            )
            if cc_provision > Decimal("0"):
                by_contribution[cc.value] = cc_provision

        # By impact domain
        by_domain: Dict[str, Decimal] = {}
        for dom in ImpactDomain:
            dom_provision = sum(
                a.financial_provision_eur for a in actions
                if a.impact_domain == dom
            )
            if dom_provision > Decimal("0"):
                by_domain[dom.value] = dom_provision

        # Actions with financial provision
        with_provision = sum(
            1 for a in actions
            if a.financial_provision_eur > Decimal("0")
        )

        # Average per action
        avg_per_action = _safe_divide(
            total_provision, _decimal(len(actions))
        )
        avg_per_action = _round_val(avg_per_action, 2)

        # Average per victim
        total_victims = sum(a.victim_count for a in actions)
        avg_per_victim = _safe_divide(
            total_provision, _decimal(total_victims)
        ) if total_victims > 0 else Decimal("0")
        avg_per_victim = _round_val(avg_per_victim, 2)

        return FinancialAnalysis(
            total_financial_provision_eur=total_provision,
            total_disbursed_eur=total_disbursed,
            disbursement_rate_pct=disbursement_rate,
            by_remediation_type=by_type,
            by_contribution_level=by_contribution,
            by_impact_domain=by_domain,
            actions_with_financial_provision=with_provision,
            average_provision_per_action_eur=avg_per_action,
            average_provision_per_victim_eur=avg_per_victim,
        )

    # ------------------------------------------------------------------ #
    # Timeline Analysis                                                    #
    # ------------------------------------------------------------------ #

    def assess_timeline_compliance(
        self, actions: List[RemediationAction]
    ) -> TimelineAnalysis:
        """Analyse timeline compliance of remediation actions.

        Assesses timeliness by comparing actual completion dates
        against target dates, computing on-time rates, average
        durations, and identifying overdue actions.

        Args:
            actions: List of RemediationAction instances.

        Returns:
            TimelineAnalysis with all timeline metrics.
        """
        if not actions:
            return TimelineAnalysis()

        n = len(actions)
        now = _utcnow()

        # Actions with timeline (have both start and target)
        with_timeline = [
            a for a in actions
            if a.start_date is not None and a.target_completion is not None
        ]

        # Completed actions for on-time analysis
        completed = [
            a for a in actions
            if a.completion_status in (
                RemediationStatus.COMPLETED,
                RemediationStatus.VERIFIED,
            )
        ]

        on_time = 0
        late = 0
        durations: List[int] = []
        delays: List[int] = []

        for a in completed:
            completion_date = a.actual_completion or now

            # Duration calculation
            if a.start_date is not None:
                # Ensure both are timezone-aware
                start = a.start_date
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                comp = completion_date
                if comp.tzinfo is None:
                    comp = comp.replace(tzinfo=timezone.utc)
                duration_days = (comp - start).days
                if duration_days >= 0:
                    durations.append(duration_days)

            # On-time analysis
            if a.target_completion is not None:
                target = a.target_completion
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                comp = completion_date
                if comp.tzinfo is None:
                    comp = comp.replace(tzinfo=timezone.utc)
                if comp <= target:
                    on_time += 1
                else:
                    late += 1
                    delay_days = (comp - target).days
                    if delay_days > 0:
                        delays.append(delay_days)

        # Overdue: past target, not completed
        overdue = 0
        for a in actions:
            if a.target_completion is not None and a.completion_status not in (
                RemediationStatus.COMPLETED,
                RemediationStatus.VERIFIED,
                RemediationStatus.FAILED,
            ):
                target = a.target_completion
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                if now > target:
                    overdue += 1

        # On-time rate (among completed with targets)
        completed_with_target = on_time + late
        on_time_pct = _pct(on_time, completed_with_target)

        # Average duration
        avg_duration = Decimal("0")
        if durations:
            avg_duration = _round_val(
                _decimal(sum(durations)) / _decimal(len(durations)), 1
            )

        # Average delay
        avg_delay = Decimal("0")
        if delays:
            avg_delay = _round_val(
                _decimal(sum(delays)) / _decimal(len(delays)), 1
            )

        # Earliest start and latest target
        starts = [
            a.start_date for a in actions if a.start_date is not None
        ]
        targets = [
            a.target_completion for a in actions
            if a.target_completion is not None
        ]
        earliest_start = min(starts) if starts else None
        latest_target = max(targets) if targets else None

        return TimelineAnalysis(
            total_actions=n,
            actions_with_timeline=len(with_timeline),
            completed_on_time=on_time,
            completed_late=late,
            on_time_rate_pct=on_time_pct,
            overdue_count=overdue,
            average_duration_days=avg_duration,
            average_delay_days=avg_delay,
            earliest_start=earliest_start,
            latest_target=latest_target,
        )

    # ------------------------------------------------------------------ #
    # Completeness Assessment                                              #
    # ------------------------------------------------------------------ #

    def _assess_action_completeness(
        self, action: RemediationAction
    ) -> CompletenessAssessment:
        """Assess documentation completeness of a single action.

        Checks the action against the completeness criteria to
        determine how fully documented it is.

        Args:
            action: RemediationAction to assess.

        Returns:
            CompletenessAssessment with criteria counts and gaps.
        """
        criteria_checks: Dict[str, bool] = {
            "has_description": bool(action.description),
            "has_type": True,  # Always has a type (required field)
            "has_financial_provision": action.financial_provision_eur > Decimal("0"),
            "has_victim_engagement": action.victim_engagement != VictimEngagementLevel.NONE,
            "has_start_date": action.start_date is not None,
            "has_target_completion": action.target_completion is not None,
            "has_responsible_person": bool(action.responsible_person),
            "has_evidence": bool(action.evidence),
            "has_grievance_link": action.grievance_mechanism_used,
            "has_contribution_level": True,  # Always present (has default)
        }

        met = sum(1 for v in criteria_checks.values() if v)
        total = len(criteria_checks)
        pct = _pct(met, total)

        missing = [k for k, v in criteria_checks.items() if not v]

        return CompletenessAssessment(
            action_id=action.action_id,
            criteria_met=met,
            criteria_total=total,
            completeness_pct=pct,
            missing_elements=missing,
        )

    # ------------------------------------------------------------------ #
    # Effectiveness Score                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_effectiveness_score(
        self,
        actions: List[RemediationAction],
        completion_rate: Decimal,
        engagement_analysis: VictimEngagementAnalysis,
        financial_analysis: FinancialAnalysis,
        timeline_analysis: TimelineAnalysis,
        avg_completeness: Decimal,
    ) -> Decimal:
        """Calculate overall effectiveness score for remediation programme.

        Weighted composite of five factors:
        - Completion rate: 25%
        - Victim engagement score: 25%
        - Financial disbursement rate: 20%
        - Timeline compliance: 15%
        - Documentation completeness: 15%

        Args:
            actions: All remediation actions.
            completion_rate: Completion rate percentage.
            engagement_analysis: Victim engagement analysis.
            financial_analysis: Financial analysis.
            timeline_analysis: Timeline analysis.
            avg_completeness: Average completeness percentage.

        Returns:
            Overall effectiveness score (0-100).
        """
        if not actions:
            return Decimal("0")

        # Factor 1: Completion rate (25%)
        f1 = completion_rate * Decimal("0.25")

        # Factor 2: Victim engagement score (25%)
        f2 = engagement_analysis.average_engagement_score * Decimal("0.25")

        # Factor 3: Financial disbursement rate (20%)
        f3 = financial_analysis.disbursement_rate_pct * Decimal("0.20")

        # Factor 4: Timeline compliance (15%)
        f4 = timeline_analysis.on_time_rate_pct * Decimal("0.15")

        # Factor 5: Documentation completeness (15%)
        f5 = avg_completeness * Decimal("0.15")

        total = f1 + f2 + f3 + f4 + f5
        return _round_val(total, 1)

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        actions: List[RemediationAction],
        completion_rate: Decimal,
        engagement_analysis: VictimEngagementAnalysis,
        financial_analysis: FinancialAnalysis,
        timeline_analysis: TimelineAnalysis,
        completeness_assessments: List[CompletenessAssessment],
        impact_ids: List[str],
    ) -> List[str]:
        """Generate prioritised recommendations for remediation improvement.

        Args:
            actions: All remediation actions.
            completion_rate: Current completion rate.
            engagement_analysis: Victim engagement analysis.
            financial_analysis: Financial analysis.
            timeline_analysis: Timeline analysis.
            completeness_assessments: Per-action completeness.
            impact_ids: All identified adverse impact IDs.

        Returns:
            List of prioritised recommendation strings.
        """
        recommendations: List[str] = []

        # Check impact coverage
        covered_impact_ids = set()
        for a in actions:
            covered_impact_ids.add(a.adverse_impact_id)
        uncovered = [
            iid for iid in impact_ids if iid not in covered_impact_ids
        ]
        if uncovered:
            recommendations.append(
                f"Develop remediation actions for {len(uncovered)} adverse "
                f"impact(s) currently without remediation. Art 10 requires "
                f"remediation for all actual adverse impacts caused or "
                f"contributed to."
            )

        # Engagement
        if engagement_analysis.engagement_rate_pct < Decimal("50"):
            recommendations.append(
                "Increase victim engagement: currently only "
                f"{engagement_analysis.engagement_rate_pct}% of actions "
                f"involve engagement with affected persons. Art 10 Para 5 "
                f"requires offering to engage with affected persons."
            )

        if (
            engagement_analysis.total_victims > 0
            and engagement_analysis.victim_reach_rate_pct < Decimal("50")
        ):
            recommendations.append(
                f"Improve victim reach: only "
                f"{engagement_analysis.victim_reach_rate_pct}% of affected "
                f"persons have been reached by remediation. Consider "
                f"additional outreach mechanisms."
            )

        # Timeline
        if timeline_analysis.overdue_count > 0:
            recommendations.append(
                f"Address {timeline_analysis.overdue_count} overdue "
                f"remediation action(s). Review timelines and escalate "
                f"to governance body if necessary."
            )

        if (
            timeline_analysis.on_time_rate_pct < Decimal("70")
            and timeline_analysis.completed_on_time + timeline_analysis.completed_late > 0
        ):
            recommendations.append(
                f"Improve timeliness: only "
                f"{timeline_analysis.on_time_rate_pct}% of completed "
                f"actions were on time. Review resource allocation and "
                f"project management practices."
            )

        # Financial
        if financial_analysis.disbursement_rate_pct < Decimal("50"):
            recommendations.append(
                f"Accelerate financial disbursement: only "
                f"{financial_analysis.disbursement_rate_pct}% of provisions "
                f"have been disbursed. Ensure funds reach affected persons."
            )

        # Grievance mechanism
        if engagement_analysis.grievance_mechanism_pct < Decimal("30"):
            recommendations.append(
                f"Increase use of the grievance mechanism (Art 12) in "
                f"remediation processes. Currently only "
                f"{engagement_analysis.grievance_mechanism_pct}% of actions "
                f"utilise the complaints procedure."
            )

        # Completeness
        low_completeness = [
            ca for ca in completeness_assessments
            if ca.completeness_pct < Decimal("60")
        ]
        if low_completeness:
            recommendations.append(
                f"Improve documentation for {len(low_completeness)} "
                f"remediation action(s) with completeness below 60%. "
                f"Ensure all required elements are documented."
            )

        # Verification
        completed_not_verified = sum(
            1 for a in actions
            if a.completion_status == RemediationStatus.COMPLETED
        )
        verified = sum(
            1 for a in actions
            if a.completion_status == RemediationStatus.VERIFIED
        )
        if completed_not_verified > 0 and verified == 0:
            recommendations.append(
                f"Arrange independent verification for "
                f"{completed_not_verified} completed remediation action(s) "
                f"to confirm effectiveness and stakeholder satisfaction."
            )

        # Failed actions
        failed = sum(
            1 for a in actions
            if a.completion_status == RemediationStatus.FAILED
        )
        if failed > 0:
            recommendations.append(
                f"Review {failed} failed remediation action(s). "
                f"Identify root causes and develop alternative approaches."
            )

        return recommendations

    # ------------------------------------------------------------------ #
    # Main Assessment Entry Point                                          #
    # ------------------------------------------------------------------ #

    def assess_remediation(
        self,
        actions: List[RemediationAction],
        impact_ids: Optional[List[str]] = None,
    ) -> RemediationResult:
        """Run a complete remediation tracking assessment.

        Evaluates all remediation actions for completion, victim
        engagement, financial provisions, timeline compliance,
        documentation completeness, and overall effectiveness.

        This is the primary entry point for the engine.

        Args:
            actions: List of RemediationAction instances.
            impact_ids: Optional list of all identified adverse impact
                IDs for coverage analysis.

        Returns:
            RemediationResult with complete assessment data and
            provenance hash.
        """
        start_time = time.time()
        if impact_ids is None:
            impact_ids = []

        logger.info(
            "Starting remediation assessment: %d actions, %d impact IDs",
            len(actions), len(impact_ids),
        )

        if not actions:
            logger.warning("No remediation actions provided")
            empty_result = RemediationResult(
                total_actions=0,
                recommendations=[
                    "No remediation actions have been defined. Art 10 "
                    "requires companies to provide remediation where they "
                    "have caused or contributed to actual adverse impacts."
                ],
                assessed_at=_utcnow(),
            )
            empty_result.provenance_hash = _compute_hash(empty_result)
            return empty_result

        n = len(actions)

        # Step 1: Completion rate
        completion_rate = self.calculate_completion_rate(actions)

        # Step 2: Actions by type
        by_type: Dict[str, int] = {}
        for rt in RemediationType:
            by_type[rt.value] = sum(
                1 for a in actions if a.remediation_type == rt
            )

        # Step 3: Actions by status
        by_status: Dict[str, int] = {}
        for rs in RemediationStatus:
            by_status[rs.value] = sum(
                1 for a in actions if a.completion_status == rs
            )

        # Step 4: Actions by contribution level
        by_contribution: Dict[str, int] = {}
        for cc in CompanyContribution:
            by_contribution[cc.value] = sum(
                1 for a in actions if a.company_contribution == cc
            )

        # Step 5: Victim engagement analysis
        engagement_analysis = self.assess_victim_engagement(actions)

        # Step 6: Financial analysis
        financial_analysis = self.analyze_financial_provisions(actions)

        # Step 7: Timeline analysis
        timeline_analysis = self.assess_timeline_compliance(actions)

        # Step 8: Completeness assessments
        completeness_assessments: List[CompletenessAssessment] = []
        for a in actions:
            ca = self._assess_action_completeness(a)
            completeness_assessments.append(ca)

        # Average completeness
        if completeness_assessments:
            total_completeness = sum(
                ca.completeness_pct for ca in completeness_assessments
            )
            avg_completeness = _round_val(
                total_completeness / _decimal(len(completeness_assessments)), 1
            )
        else:
            avg_completeness = Decimal("0.0")

        # Step 9: Effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(
            actions=actions,
            completion_rate=completion_rate,
            engagement_analysis=engagement_analysis,
            financial_analysis=financial_analysis,
            timeline_analysis=timeline_analysis,
            avg_completeness=avg_completeness,
        )

        # Step 10: Unique impacts with remediation
        unique_impacts = set()
        for a in actions:
            unique_impacts.add(a.adverse_impact_id)
        impacts_with_remediation = len(unique_impacts)

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            actions=actions,
            completion_rate=completion_rate,
            engagement_analysis=engagement_analysis,
            financial_analysis=financial_analysis,
            timeline_analysis=timeline_analysis,
            completeness_assessments=completeness_assessments,
            impact_ids=impact_ids,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Step 12: Build result
        result = RemediationResult(
            total_actions=n,
            actions_by_type=by_type,
            actions_by_status=by_status,
            actions_by_contribution=by_contribution,
            total_financial_provision_eur=financial_analysis.total_financial_provision_eur,
            completion_rate_pct=completion_rate,
            victim_engagement_rate_pct=engagement_analysis.engagement_rate_pct,
            effectiveness_score=effectiveness_score,
            financial_analysis=financial_analysis,
            victim_engagement_analysis=engagement_analysis,
            timeline_analysis=timeline_analysis,
            completeness_assessments=completeness_assessments,
            average_completeness_pct=avg_completeness,
            impacts_with_remediation=impacts_with_remediation,
            recommendations=recommendations,
            processing_time_ms=_round2(processing_time_ms),
            assessed_at=_utcnow(),
        )

        # Step 13: Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Remediation assessment complete: %d actions, "
            "completion=%.1f%%, engagement=%.1f%%, "
            "effectiveness=%.1f, time=%.2fms",
            n,
            float(completion_rate),
            float(engagement_analysis.engagement_rate_pct),
            float(effectiveness_score),
            processing_time_ms,
        )

        return result
