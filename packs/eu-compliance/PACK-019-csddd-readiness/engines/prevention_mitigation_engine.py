# -*- coding: utf-8 -*-
"""
PreventionMitigationEngine - PACK-019 CSDDD Prevention & Mitigation Engine
===========================================================================

Tracks and assesses prevention measures (Art 8) and corrective actions
(Art 9) taken by companies to address identified actual and potential
adverse human rights and environmental impacts under the EU Corporate
Sustainability Due Diligence Directive (CSDDD / Directive 2024/1760).

CSDDD Art 8 - Preventing Potential Adverse Impacts:
    - Para 1: Companies shall take appropriate measures to prevent, or
      where prevention is not possible, adequately mitigate, potential
      adverse human rights and environmental impacts.
    - Para 2: Appropriate measures include:
      (a) a prevention action plan with reasonable and clearly defined
          timelines for action and qualitative and quantitative
          indicators for measuring improvement
      (b) seeking contractual assurances from direct business partners
          to ensure compliance with the company's code of conduct and
          prevention action plan
      (c) making necessary financial or other investments,
          adjustments or upgrades
      (d) providing targeted and proportionate support for SMEs
      (e) collaborating with other entities including industry
          initiatives
    - Para 3: For contractual assurances, the company shall verify
      compliance through appropriate measures.

CSDDD Art 9 - Bringing Actual Adverse Impacts to an End:
    - Para 1: Companies shall take appropriate measures to bring
      actual adverse human rights and environmental impacts to an end.
    - Para 2: Where an adverse impact cannot immediately be brought
      to an end, the company shall minimise its extent.
    - Para 3: Appropriate measures include:
      (a) neutralising the adverse impact through payment of damages
      (b) a corrective action plan with reasonable and clearly defined
          timelines
      (c) seeking contractual assurances from direct business partners
      (d) making necessary investments, adjustments or upgrades
      (e) providing targeted support for SMEs
      (f) collaborating with other entities

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D), Articles 8-9
    - OECD Due Diligence Guidance for Responsible Business Conduct
    - UN Guiding Principles on Business and Human Rights (UNGPs 19-21)
    - ILO MNE Declaration
    - EU Taxonomy Regulation alignment for investment measures

Zero-Hallucination:
    - Coverage analysis uses set intersection for impact matching
    - Effectiveness scoring uses Decimal arithmetic
    - Budget calculations use Decimal precision
    - Timeline compliance uses date comparison
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
from typing import Any, Dict, List, Optional, Set, Tuple

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


class MeasureType(str, Enum):
    """Type of prevention or mitigation measure under CSDDD Art 8-9.

    - PREVENTION: Measures to prevent potential impacts from occurring
    - MITIGATION: Measures to reduce the severity of potential impacts
    - CESSATION: Measures to bring actual impacts to an end (Art 9)
    - REMEDIATION: Measures to restore or compensate (linked to Art 10)
    """
    PREVENTION = "prevention"
    MITIGATION = "mitigation"
    CESSATION = "cessation"
    REMEDIATION = "remediation"


class MeasureStatus(str, Enum):
    """Implementation status of a prevention or mitigation measure.

    Tracks the lifecycle of each measure from planning through
    completion or cancellation.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class EffectivenessRating(str, Enum):
    """Effectiveness rating for a measure's outcomes.

    Assesses how well a measure has achieved its intended purpose
    of preventing, mitigating, or ending adverse impacts.

    - HIGHLY_EFFECTIVE: Impact fully prevented/ended (score 90-100)
    - EFFECTIVE: Impact substantially reduced (score 70-89)
    - PARTIALLY_EFFECTIVE: Some improvement observed (score 40-69)
    - INEFFECTIVE: No measurable improvement (score 0-39)
    - NOT_ASSESSED: Effectiveness not yet evaluated
    """
    HIGHLY_EFFECTIVE = "highly_effective"
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_ASSESSED = "not_assessed"


class MeasureCategory(str, Enum):
    """Category of measure per CSDDD Art 8 Para 2 / Art 9 Para 3.

    Classifies the nature of the measure following the types
    outlined in the Directive.
    """
    ACTION_PLAN = "action_plan"
    CONTRACTUAL_ASSURANCE = "contractual_assurance"
    INVESTMENT = "investment"
    SME_SUPPORT = "sme_support"
    INDUSTRY_COLLABORATION = "industry_collaboration"
    OPERATIONAL_ADJUSTMENT = "operational_adjustment"
    TRAINING = "training"
    POLICY_CHANGE = "policy_change"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    OTHER = "other"


class ImpactDomain(str, Enum):
    """Domain of the adverse impact being addressed.

    Maps to the CSDDD's two main impact categories.
    """
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENTAL = "environmental"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Effectiveness score ranges for rating determination
EFFECTIVENESS_THRESHOLDS: List[Tuple[Decimal, EffectivenessRating]] = [
    (Decimal("90"), EffectivenessRating.HIGHLY_EFFECTIVE),
    (Decimal("70"), EffectivenessRating.EFFECTIVE),
    (Decimal("40"), EffectivenessRating.PARTIALLY_EFFECTIVE),
    (Decimal("0"), EffectivenessRating.INEFFECTIVE),
]

# Status weights for overall progress calculation
STATUS_PROGRESS_WEIGHTS: Dict[str, Decimal] = {
    MeasureStatus.COMPLETED.value: Decimal("1.0"),
    MeasureStatus.IN_PROGRESS.value: Decimal("0.5"),
    MeasureStatus.PLANNED.value: Decimal("0.1"),
    MeasureStatus.OVERDUE.value: Decimal("0.3"),
    MeasureStatus.CANCELLED.value: Decimal("0.0"),
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PreventionMeasure(BaseModel):
    """A single prevention or mitigation measure per CSDDD Art 8-9.

    Represents one action taken or planned by the company to prevent
    potential adverse impacts (Art 8) or bring actual adverse impacts
    to an end (Art 9).

    Attributes:
        measure_id: Unique identifier for this measure.
        measure_type: Whether prevention, mitigation, cessation, or remediation.
        measure_category: Category of measure (action plan, contractual, etc.).
        description: Narrative description of the measure.
        target_impact_ids: List of adverse impact IDs this measure addresses.
        impact_domain: Human rights, environmental, or both.
        responsible_person: Person or role responsible for implementation.
        responsible_department: Department responsible.
        deadline: Target completion date.
        start_date: Actual or planned start date.
        actual_completion_date: Actual completion date (if completed).
        budget_eur: Budgeted cost in EUR.
        actual_cost_eur: Actual cost incurred in EUR.
        effectiveness_score: Effectiveness score (0-100).
        status: Current implementation status.
        evidence: Supporting evidence or documentation references.
        kpis: Key performance indicators for measuring progress.
        is_verified: Whether the measure has been independently verified.
        verification_body: Name of the verification body (if verified).
    """
    measure_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this measure",
    )
    measure_type: MeasureType = Field(
        ...,
        description="Type of measure (prevention, mitigation, cessation, remediation)",
    )
    measure_category: MeasureCategory = Field(
        default=MeasureCategory.OTHER,
        description="Category of measure per Art 8-9",
    )
    description: str = Field(
        default="",
        description="Narrative description of the measure",
        max_length=5000,
    )
    target_impact_ids: List[str] = Field(
        default_factory=list,
        description="List of adverse impact IDs this measure addresses",
    )
    impact_domain: ImpactDomain = Field(
        default=ImpactDomain.BOTH,
        description="Domain of impact being addressed",
    )
    responsible_person: str = Field(
        default="",
        description="Person or role responsible for implementation",
        max_length=500,
    )
    responsible_department: str = Field(
        default="",
        description="Department responsible for implementation",
        max_length=200,
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Target completion date",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Actual or planned start date",
    )
    actual_completion_date: Optional[datetime] = Field(
        default=None,
        description="Actual completion date (if completed)",
    )
    budget_eur: Decimal = Field(
        default=Decimal("0"),
        description="Budgeted cost in EUR",
        ge=Decimal("0"),
    )
    actual_cost_eur: Decimal = Field(
        default=Decimal("0"),
        description="Actual cost incurred in EUR",
        ge=Decimal("0"),
    )
    effectiveness_score: Decimal = Field(
        default=Decimal("0"),
        description="Effectiveness score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    status: MeasureStatus = Field(
        default=MeasureStatus.PLANNED,
        description="Current implementation status",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence or documentation references",
    )
    kpis: List[str] = Field(
        default_factory=list,
        description="Key performance indicators for this measure",
    )
    is_verified: bool = Field(
        default=False,
        description="Whether the measure has been independently verified",
    )
    verification_body: str = Field(
        default="",
        description="Name of the verification body (if verified)",
        max_length=500,
    )
    involves_sme_support: bool = Field(
        default=False,
        description="Whether the measure involves targeted SME support",
    )
    involves_industry_collaboration: bool = Field(
        default=False,
        description="Whether the measure involves industry collaboration",
    )
    contractual_assurances_obtained: bool = Field(
        default=False,
        description="Whether contractual assurances have been obtained",
    )
    compliance_verified: bool = Field(
        default=False,
        description="Whether compliance with assurances has been verified",
    )


class MeasureEffectiveness(BaseModel):
    """Effectiveness assessment for a single measure.

    Captures the effectiveness rating, numeric score, supporting
    evidence, and assessment date for audit trail purposes.
    """
    measure_id: str = Field(
        ...,
        description="Reference to the measure being assessed",
    )
    rating: EffectivenessRating = Field(
        ...,
        description="Qualitative effectiveness rating",
    )
    score: Decimal = Field(
        ...,
        description="Numeric effectiveness score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence for the assessment",
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow,
        description="Date of effectiveness assessment",
    )
    assessor: str = Field(
        default="",
        description="Person or body that conducted the assessment",
        max_length=500,
    )


class BudgetSummary(BaseModel):
    """Financial summary of prevention and mitigation measures.

    Aggregates budget allocations and actual costs across all
    measures for financial oversight and reporting.
    """
    total_budget_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total budgeted amount across all measures",
    )
    total_actual_cost_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total actual cost incurred",
    )
    budget_utilisation_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of budget utilised",
    )
    by_measure_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Budget breakdown by measure type",
    )
    by_measure_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Budget breakdown by measure category",
    )
    by_impact_domain: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Budget breakdown by impact domain",
    )
    measures_with_budget: int = Field(
        default=0,
        description="Count of measures that have budget allocated",
        ge=0,
    )
    measures_over_budget: int = Field(
        default=0,
        description="Count of measures where actual cost exceeds budget",
        ge=0,
    )
    over_budget_amount_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total amount over budget across all measures",
    )


class CoverageAnalysis(BaseModel):
    """Analysis of how well measures cover identified impacts.

    Determines which adverse impacts have associated prevention
    or mitigation measures and which remain uncovered.
    """
    total_impacts: int = Field(
        default=0,
        description="Total number of identified adverse impacts",
        ge=0,
    )
    covered_impacts: int = Field(
        default=0,
        description="Number of impacts with at least one measure",
        ge=0,
    )
    uncovered_impacts: int = Field(
        default=0,
        description="Number of impacts with no measures",
        ge=0,
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of impacts covered by measures",
    )
    covered_impact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of impacts that are covered",
    )
    uncovered_impact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of impacts that are not covered",
    )
    impacts_with_multiple_measures: int = Field(
        default=0,
        description="Number of impacts addressed by 2+ measures",
        ge=0,
    )
    average_measures_per_impact: Decimal = Field(
        default=Decimal("0"),
        description="Average number of measures per covered impact",
    )


class GapAnalysis(BaseModel):
    """Gap analysis identifying weaknesses in the measure portfolio.

    Identifies impacts without measures, measure types that are
    underrepresented, and areas requiring additional attention.
    """
    uncovered_impact_ids: List[str] = Field(
        default_factory=list,
        description="Impact IDs with no associated measures",
    )
    gaps_by_measure_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of gaps by expected measure type",
    )
    missing_contractual_assurances: int = Field(
        default=0,
        description="Measures lacking contractual assurances",
        ge=0,
    )
    missing_kpis: int = Field(
        default=0,
        description="Measures without defined KPIs",
        ge=0,
    )
    missing_deadlines: int = Field(
        default=0,
        description="Measures without defined deadlines",
        ge=0,
    )
    missing_responsible_person: int = Field(
        default=0,
        description="Measures without a designated responsible person",
        ge=0,
    )
    overdue_measures: int = Field(
        default=0,
        description="Count of overdue measures",
        ge=0,
    )
    unverified_completed_measures: int = Field(
        default=0,
        description="Completed measures without independent verification",
        ge=0,
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations to close gaps",
    )


class EffectivenessSummary(BaseModel):
    """Summary of effectiveness across all measures.

    Aggregates effectiveness ratings and scores for portfolio-level
    assessment of the company's prevention and mitigation programme.
    """
    total_assessed: int = Field(
        default=0,
        description="Total measures with effectiveness assessed",
        ge=0,
    )
    total_not_assessed: int = Field(
        default=0,
        description="Total measures without effectiveness assessment",
        ge=0,
    )
    average_effectiveness_score: Decimal = Field(
        default=Decimal("0"),
        description="Average effectiveness score across assessed measures",
    )
    by_rating: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of measures by effectiveness rating",
    )
    by_measure_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Average effectiveness by measure type",
    )
    highly_effective_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of measures rated highly effective",
    )
    ineffective_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of measures rated ineffective",
    )


class PreventionResult(BaseModel):
    """Complete prevention and mitigation assessment result.

    Aggregates all measure assessments, effectiveness analysis,
    budget summary, coverage analysis, gap analysis, and
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
    total_measures: int = Field(
        default=0,
        description="Total number of measures assessed",
        ge=0,
    )
    measures_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of measures by type",
    )
    measures_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of measures by status",
    )
    measures_by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of measures by category",
    )
    effectiveness_summary: EffectivenessSummary = Field(
        default_factory=EffectivenessSummary,
        description="Effectiveness assessment summary",
    )
    budget_summary: BudgetSummary = Field(
        default_factory=BudgetSummary,
        description="Financial summary of measures",
    )
    coverage_analysis: CoverageAnalysis = Field(
        default_factory=CoverageAnalysis,
        description="Coverage analysis against identified impacts",
    )
    gap_analysis: GapAnalysis = Field(
        default_factory=GapAnalysis,
        description="Gap analysis with recommendations",
    )
    overall_progress_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall progress score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    timeline_compliance_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of measures on track or completed on time",
    )
    verification_rate_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of completed measures with verification",
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


class PreventionMitigationEngine:
    """CSDDD Prevention and Mitigation assessment engine.

    Provides deterministic, zero-hallucination assessment of prevention
    measures (Art 8) and corrective actions (Art 9) against the
    requirements of Directive 2024/1760.

    The engine evaluates:
    1. **Effectiveness** of each measure and the portfolio as a whole.
    2. **Budget** allocation and utilisation across all measures.
    3. **Coverage** of identified adverse impacts by measures.
    4. **Gaps** in the measure portfolio requiring attention.
    5. **Timeline compliance** and measure status distribution.

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = PreventionMitigationEngine()
        measures = [PreventionMeasure(...), ...]
        impact_ids = ["impact-001", "impact-002", ...]
        result = engine.assess_prevention_measures(measures, impact_ids)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Effectiveness Calculation                                            #
    # ------------------------------------------------------------------ #

    def calculate_effectiveness(
        self, measure: PreventionMeasure
    ) -> MeasureEffectiveness:
        """Calculate effectiveness rating for a single measure.

        Maps the measure's numeric effectiveness_score to a
        qualitative EffectivenessRating using fixed thresholds.

        Args:
            measure: PreventionMeasure to assess.

        Returns:
            MeasureEffectiveness with rating and score.
        """
        score = measure.effectiveness_score

        # Determine rating from score thresholds
        if measure.status == MeasureStatus.CANCELLED:
            rating = EffectivenessRating.NOT_ASSESSED
        elif score == Decimal("0") and measure.status in (
            MeasureStatus.PLANNED, MeasureStatus.IN_PROGRESS
        ):
            rating = EffectivenessRating.NOT_ASSESSED
        else:
            rating = self._score_to_effectiveness_rating(score)

        return MeasureEffectiveness(
            measure_id=measure.measure_id,
            rating=rating,
            score=score,
            evidence=measure.evidence,
            assessment_date=_utcnow(),
        )

    @staticmethod
    def _score_to_effectiveness_rating(
        score: Decimal,
    ) -> EffectivenessRating:
        """Map a numeric score to an EffectivenessRating.

        Thresholds:
        - >= 90: HIGHLY_EFFECTIVE
        - >= 70: EFFECTIVE
        - >= 40: PARTIALLY_EFFECTIVE
        - < 40:  INEFFECTIVE

        Args:
            score: Effectiveness score (0-100).

        Returns:
            EffectivenessRating enum value.
        """
        for threshold, rating in EFFECTIVENESS_THRESHOLDS:
            if score >= threshold:
                return rating
        return EffectivenessRating.INEFFECTIVE

    # ------------------------------------------------------------------ #
    # Effectiveness Summary                                                #
    # ------------------------------------------------------------------ #

    def _calculate_effectiveness_summary(
        self,
        measures: List[PreventionMeasure],
        effectiveness_results: List[MeasureEffectiveness],
    ) -> EffectivenessSummary:
        """Compute aggregate effectiveness statistics.

        Calculates average effectiveness, distribution by rating,
        and effectiveness breakdown by measure type.

        Args:
            measures: All PreventionMeasure instances.
            effectiveness_results: Effectiveness for each measure.

        Returns:
            EffectivenessSummary with all aggregations.
        """
        if not effectiveness_results:
            return EffectivenessSummary()

        # Count by rating
        by_rating: Dict[str, int] = {r.value: 0 for r in EffectivenessRating}
        for eff in effectiveness_results:
            by_rating[eff.rating.value] = by_rating.get(eff.rating.value, 0) + 1

        # Assessed vs not assessed
        assessed = [
            e for e in effectiveness_results
            if e.rating != EffectivenessRating.NOT_ASSESSED
        ]
        not_assessed_count = len(effectiveness_results) - len(assessed)

        # Average effectiveness score (assessed only)
        if assessed:
            total_score = sum(e.score for e in assessed)
            avg_score = _round_val(
                total_score / _decimal(len(assessed)), 1
            )
        else:
            avg_score = Decimal("0")

        # By measure type
        measure_map: Dict[str, PreventionMeasure] = {
            m.measure_id: m for m in measures
        }
        type_scores: Dict[str, List[Decimal]] = {}
        for eff in assessed:
            m = measure_map.get(eff.measure_id)
            if m is None:
                continue
            mt = m.measure_type.value
            if mt not in type_scores:
                type_scores[mt] = []
            type_scores[mt].append(eff.score)

        by_type: Dict[str, Decimal] = {}
        for mt, scores in type_scores.items():
            by_type[mt] = _round_val(
                sum(scores) / _decimal(len(scores)), 1
            )

        # Percentages
        total = len(effectiveness_results)
        he_count = by_rating.get(EffectivenessRating.HIGHLY_EFFECTIVE.value, 0)
        ineff_count = by_rating.get(EffectivenessRating.INEFFECTIVE.value, 0)
        he_pct = _pct(he_count, total)
        ineff_pct = _pct(ineff_count, total)

        return EffectivenessSummary(
            total_assessed=len(assessed),
            total_not_assessed=not_assessed_count,
            average_effectiveness_score=avg_score,
            by_rating=by_rating,
            by_measure_type=by_type,
            highly_effective_pct=he_pct,
            ineffective_pct=ineff_pct,
        )

    # ------------------------------------------------------------------ #
    # Budget Calculation                                                   #
    # ------------------------------------------------------------------ #

    def calculate_budget_summary(
        self, measures: List[PreventionMeasure]
    ) -> BudgetSummary:
        """Calculate financial summary across all measures.

        Aggregates budgets and actual costs by measure type, category,
        and impact domain.  Identifies over-budget measures.

        Args:
            measures: List of PreventionMeasure instances.

        Returns:
            BudgetSummary with all financial aggregations.
        """
        if not measures:
            return BudgetSummary()

        total_budget = sum(m.budget_eur for m in measures)
        total_actual = sum(m.actual_cost_eur for m in measures)
        utilisation = _pct_dec(total_actual, total_budget)

        # By measure type
        by_type: Dict[str, Decimal] = {}
        for mt in MeasureType:
            type_budget = sum(
                m.budget_eur for m in measures
                if m.measure_type == mt
            )
            if type_budget > Decimal("0"):
                by_type[mt.value] = type_budget

        # By measure category
        by_category: Dict[str, Decimal] = {}
        for mc in MeasureCategory:
            cat_budget = sum(
                m.budget_eur for m in measures
                if m.measure_category == mc
            )
            if cat_budget > Decimal("0"):
                by_category[mc.value] = cat_budget

        # By impact domain
        by_domain: Dict[str, Decimal] = {}
        for dom in ImpactDomain:
            dom_budget = sum(
                m.budget_eur for m in measures
                if m.impact_domain == dom
            )
            if dom_budget > Decimal("0"):
                by_domain[dom.value] = dom_budget

        # Measures with budget
        with_budget = sum(
            1 for m in measures if m.budget_eur > Decimal("0")
        )

        # Over-budget analysis
        over_budget_count = 0
        over_budget_amount = Decimal("0")
        for m in measures:
            if m.budget_eur > Decimal("0") and m.actual_cost_eur > m.budget_eur:
                over_budget_count += 1
                over_budget_amount += m.actual_cost_eur - m.budget_eur

        return BudgetSummary(
            total_budget_eur=total_budget,
            total_actual_cost_eur=total_actual,
            budget_utilisation_pct=utilisation,
            by_measure_type=by_type,
            by_measure_category=by_category,
            by_impact_domain=by_domain,
            measures_with_budget=with_budget,
            measures_over_budget=over_budget_count,
            over_budget_amount_eur=over_budget_amount,
        )

    # ------------------------------------------------------------------ #
    # Coverage Analysis                                                    #
    # ------------------------------------------------------------------ #

    def analyze_coverage(
        self,
        measures: List[PreventionMeasure],
        impact_ids: List[str],
    ) -> CoverageAnalysis:
        """Analyse how well measures cover identified adverse impacts.

        Determines which impacts have associated measures and which
        remain uncovered.  Computes coverage percentage and average
        measures per covered impact.

        Args:
            measures: List of PreventionMeasure instances.
            impact_ids: List of all identified adverse impact IDs.

        Returns:
            CoverageAnalysis with coverage metrics.
        """
        if not impact_ids:
            return CoverageAnalysis()

        all_impacts: Set[str] = set(impact_ids)
        total_impacts = len(all_impacts)

        # Collect which impacts are covered by measures
        impact_measure_count: Dict[str, int] = {
            iid: 0 for iid in all_impacts
        }
        for m in measures:
            if m.status == MeasureStatus.CANCELLED:
                continue
            for iid in m.target_impact_ids:
                if iid in impact_measure_count:
                    impact_measure_count[iid] += 1

        covered_ids = [
            iid for iid, count in impact_measure_count.items()
            if count > 0
        ]
        uncovered_ids = [
            iid for iid, count in impact_measure_count.items()
            if count == 0
        ]
        multi_measure = sum(
            1 for count in impact_measure_count.values() if count > 1
        )

        covered_count = len(covered_ids)
        coverage_pct = _pct(covered_count, total_impacts)

        # Average measures per covered impact
        if covered_count > 0:
            total_measure_links = sum(
                count for count in impact_measure_count.values()
                if count > 0
            )
            avg_measures = _round_val(
                _decimal(total_measure_links) / _decimal(covered_count), 1
            )
        else:
            avg_measures = Decimal("0")

        return CoverageAnalysis(
            total_impacts=total_impacts,
            covered_impacts=covered_count,
            uncovered_impacts=len(uncovered_ids),
            coverage_pct=coverage_pct,
            covered_impact_ids=sorted(covered_ids),
            uncovered_impact_ids=sorted(uncovered_ids),
            impacts_with_multiple_measures=multi_measure,
            average_measures_per_impact=avg_measures,
        )

    # ------------------------------------------------------------------ #
    # Gap Analysis                                                         #
    # ------------------------------------------------------------------ #

    def identify_gaps(
        self,
        measures: List[PreventionMeasure],
        impact_ids: List[str],
    ) -> GapAnalysis:
        """Identify gaps in the prevention and mitigation portfolio.

        Checks for uncovered impacts, missing process elements
        (KPIs, deadlines, responsible persons), overdue measures,
        and unverified completed measures.

        Args:
            measures: List of PreventionMeasure instances.
            impact_ids: List of all identified adverse impact IDs.

        Returns:
            GapAnalysis with identified gaps and recommendations.
        """
        # Uncovered impacts
        coverage = self.analyze_coverage(measures, impact_ids)
        uncovered_ids = coverage.uncovered_impact_ids

        # Active (non-cancelled) measures for gap checks
        active_measures = [
            m for m in measures if m.status != MeasureStatus.CANCELLED
        ]

        # Gaps by measure type coverage
        gaps_by_type: Dict[str, int] = {}
        for mt in MeasureType:
            type_count = sum(
                1 for m in active_measures if m.measure_type == mt
            )
            if type_count == 0:
                gaps_by_type[mt.value] = 0

        # Missing contractual assurances
        missing_ca = sum(
            1 for m in active_measures
            if not m.contractual_assurances_obtained
            and m.measure_type in (MeasureType.PREVENTION, MeasureType.MITIGATION)
        )

        # Missing KPIs
        missing_kpis = sum(
            1 for m in active_measures if not m.kpis
        )

        # Missing deadlines
        missing_deadlines = sum(
            1 for m in active_measures if m.deadline is None
        )

        # Missing responsible person
        missing_responsible = sum(
            1 for m in active_measures if not m.responsible_person
        )

        # Overdue measures
        now = _utcnow()
        overdue_count = sum(
            1 for m in active_measures
            if m.status == MeasureStatus.OVERDUE
            or (
                m.deadline is not None
                and m.deadline < now
                and m.status not in (
                    MeasureStatus.COMPLETED,
                    MeasureStatus.CANCELLED,
                )
            )
        )

        # Unverified completed measures
        completed = [
            m for m in measures if m.status == MeasureStatus.COMPLETED
        ]
        unverified_completed = sum(
            1 for m in completed if not m.is_verified
        )

        # Generate recommendations
        recommendations = self._generate_gap_recommendations(
            uncovered_ids=uncovered_ids,
            missing_ca=missing_ca,
            missing_kpis=missing_kpis,
            missing_deadlines=missing_deadlines,
            missing_responsible=missing_responsible,
            overdue_count=overdue_count,
            unverified_completed=unverified_completed,
            gaps_by_type=gaps_by_type,
        )

        return GapAnalysis(
            uncovered_impact_ids=uncovered_ids,
            gaps_by_measure_type=gaps_by_type,
            missing_contractual_assurances=missing_ca,
            missing_kpis=missing_kpis,
            missing_deadlines=missing_deadlines,
            missing_responsible_person=missing_responsible,
            overdue_measures=overdue_count,
            unverified_completed_measures=unverified_completed,
            recommendations=recommendations,
        )

    @staticmethod
    def _generate_gap_recommendations(
        uncovered_ids: List[str],
        missing_ca: int,
        missing_kpis: int,
        missing_deadlines: int,
        missing_responsible: int,
        overdue_count: int,
        unverified_completed: int,
        gaps_by_type: Dict[str, int],
    ) -> List[str]:
        """Generate actionable recommendations from identified gaps.

        Args:
            uncovered_ids: Impact IDs without measures.
            missing_ca: Count of measures without contractual assurances.
            missing_kpis: Count of measures without KPIs.
            missing_deadlines: Count of measures without deadlines.
            missing_responsible: Count of measures without responsible person.
            overdue_count: Count of overdue measures.
            unverified_completed: Count of unverified completed measures.
            gaps_by_type: Measure types with no active measures.

        Returns:
            List of recommendation strings, prioritised.
        """
        recommendations: List[str] = []

        if uncovered_ids:
            recommendations.append(
                f"Develop measures for {len(uncovered_ids)} uncovered adverse "
                f"impacts. Art 8 requires appropriate measures for all "
                f"identified potential impacts."
            )

        if overdue_count > 0:
            recommendations.append(
                f"Address {overdue_count} overdue measure(s). Review and "
                f"update timelines or escalate to governance body."
            )

        if missing_kpis > 0:
            recommendations.append(
                f"Define KPIs for {missing_kpis} measure(s). Art 8 Para 2(a) "
                f"requires qualitative and quantitative indicators for "
                f"measuring improvement."
            )

        if missing_deadlines > 0:
            recommendations.append(
                f"Set deadlines for {missing_deadlines} measure(s). "
                f"Art 8 requires clearly defined timelines for action."
            )

        if missing_responsible > 0:
            recommendations.append(
                f"Assign responsible persons for {missing_responsible} "
                f"measure(s) to ensure accountability."
            )

        if missing_ca > 0:
            recommendations.append(
                f"Obtain contractual assurances for {missing_ca} prevention/"
                f"mitigation measure(s) per Art 8 Para 2(b)."
            )

        if unverified_completed > 0:
            recommendations.append(
                f"Arrange independent verification for {unverified_completed} "
                f"completed measure(s) to confirm effectiveness."
            )

        for mt_val in gaps_by_type:
            recommendations.append(
                f"No active {mt_val} measures found. Consider developing "
                f"measures of this type as part of a comprehensive approach."
            )

        return recommendations

    # ------------------------------------------------------------------ #
    # Timeline Compliance                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_timeline_compliance(
        self, measures: List[PreventionMeasure]
    ) -> Decimal:
        """Calculate the percentage of measures on track or completed on time.

        A measure is on-track if:
        - Status is COMPLETED (regardless of timeline)
        - Status is IN_PROGRESS or PLANNED and deadline is in the future

        A measure is off-track if:
        - Status is OVERDUE
        - Deadline has passed and status is not COMPLETED/CANCELLED

        Args:
            measures: List of PreventionMeasure instances.

        Returns:
            Percentage (0-100) of measures on track.
        """
        active = [
            m for m in measures if m.status != MeasureStatus.CANCELLED
        ]
        if not active:
            return Decimal("0.0")

        now = _utcnow()
        on_track = 0
        for m in active:
            if m.status == MeasureStatus.COMPLETED:
                on_track += 1
            elif m.status == MeasureStatus.OVERDUE:
                pass  # Not on track
            elif m.deadline is not None and m.deadline < now:
                pass  # Past deadline, not on track
            else:
                on_track += 1  # Planned or in progress, deadline in future

        return _pct(on_track, len(active))

    # ------------------------------------------------------------------ #
    # Overall Progress Score                                               #
    # ------------------------------------------------------------------ #

    def _calculate_overall_progress(
        self, measures: List[PreventionMeasure]
    ) -> Decimal:
        """Calculate an overall progress score for the measure portfolio.

        Uses status-based weights:
        - COMPLETED: 1.0
        - IN_PROGRESS: 0.5
        - OVERDUE: 0.3
        - PLANNED: 0.1
        - CANCELLED: 0.0

        The overall score is the weighted average * 100.

        Args:
            measures: List of PreventionMeasure instances.

        Returns:
            Overall progress score (0-100).
        """
        if not measures:
            return Decimal("0")

        total_weight = Decimal("0")
        for m in measures:
            w = STATUS_PROGRESS_WEIGHTS.get(
                m.status.value, Decimal("0")
            )
            total_weight += w

        score = _safe_divide(
            total_weight,
            _decimal(len(measures)),
        ) * Decimal("100")

        return _round_val(score, 1)

    # ------------------------------------------------------------------ #
    # Verification Rate                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calculate_verification_rate(
        measures: List[PreventionMeasure],
    ) -> Decimal:
        """Calculate percentage of completed measures with verification.

        Args:
            measures: List of PreventionMeasure instances.

        Returns:
            Verification rate as percentage (0-100).
        """
        completed = [
            m for m in measures if m.status == MeasureStatus.COMPLETED
        ]
        if not completed:
            return Decimal("0.0")

        verified = sum(1 for m in completed if m.is_verified)
        return _pct(verified, len(completed))

    # ------------------------------------------------------------------ #
    # Main Assessment Entry Point                                          #
    # ------------------------------------------------------------------ #

    def assess_prevention_measures(
        self,
        measures: List[PreventionMeasure],
        impact_ids: List[str],
    ) -> PreventionResult:
        """Run a complete prevention and mitigation assessment.

        Evaluates all measures for effectiveness, budget utilisation,
        coverage of identified impacts, gaps, and overall progress.

        This is the primary entry point for the engine.

        Args:
            measures: List of PreventionMeasure instances.
            impact_ids: List of all identified adverse impact IDs.

        Returns:
            PreventionResult with complete assessment data and
            provenance hash.
        """
        start_time = time.time()
        logger.info(
            "Starting prevention/mitigation assessment: %d measures, "
            "%d impact IDs",
            len(measures), len(impact_ids),
        )

        if not measures:
            logger.warning("No prevention measures provided")
            empty_result = PreventionResult(
                total_measures=0,
                coverage_analysis=CoverageAnalysis(
                    total_impacts=len(impact_ids),
                    uncovered_impacts=len(impact_ids),
                    uncovered_impact_ids=sorted(impact_ids),
                    coverage_pct=Decimal("0.0"),
                ),
                gap_analysis=GapAnalysis(
                    uncovered_impact_ids=sorted(impact_ids),
                    recommendations=[
                        "No prevention or mitigation measures have been "
                        "defined. Art 8 requires appropriate measures for "
                        "all identified potential adverse impacts."
                    ],
                ),
                assessed_at=_utcnow(),
            )
            empty_result.provenance_hash = _compute_hash(empty_result)
            return empty_result

        # Step 1: Calculate effectiveness for each measure
        effectiveness_results: List[MeasureEffectiveness] = []
        for m in measures:
            eff = self.calculate_effectiveness(m)
            effectiveness_results.append(eff)

        # Step 2: Effectiveness summary
        effectiveness_summary = self._calculate_effectiveness_summary(
            measures, effectiveness_results
        )

        # Step 3: Budget summary
        budget_summary = self.calculate_budget_summary(measures)

        # Step 4: Coverage analysis
        coverage_analysis = self.analyze_coverage(measures, impact_ids)

        # Step 5: Gap analysis
        gap_analysis = self.identify_gaps(measures, impact_ids)

        # Step 6: Measures distribution
        by_type: Dict[str, int] = {}
        for mt in MeasureType:
            by_type[mt.value] = sum(
                1 for m in measures if m.measure_type == mt
            )

        by_status: Dict[str, int] = {}
        for ms in MeasureStatus:
            by_status[ms.value] = sum(
                1 for m in measures if m.status == ms
            )

        by_category: Dict[str, int] = {}
        for mc in MeasureCategory:
            count = sum(
                1 for m in measures if m.measure_category == mc
            )
            if count > 0:
                by_category[mc.value] = count

        # Step 7: Overall progress
        overall_progress = self._calculate_overall_progress(measures)

        # Step 8: Timeline compliance
        timeline_compliance = self._calculate_timeline_compliance(measures)

        # Step 9: Verification rate
        verification_rate = self._calculate_verification_rate(measures)

        # Step 10: Aggregate recommendations
        recommendations = list(gap_analysis.recommendations)

        processing_time_ms = (time.time() - start_time) * 1000

        # Step 11: Build result
        result = PreventionResult(
            total_measures=len(measures),
            measures_by_type=by_type,
            measures_by_status=by_status,
            measures_by_category=by_category,
            effectiveness_summary=effectiveness_summary,
            budget_summary=budget_summary,
            coverage_analysis=coverage_analysis,
            gap_analysis=gap_analysis,
            overall_progress_score=overall_progress,
            timeline_compliance_pct=timeline_compliance,
            verification_rate_pct=verification_rate,
            recommendations=recommendations,
            processing_time_ms=_round2(processing_time_ms),
            assessed_at=_utcnow(),
        )

        # Step 12: Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Prevention/mitigation assessment complete: %d measures, "
            "coverage=%.1f%%, progress=%.1f, time=%.2fms",
            len(measures),
            float(coverage_analysis.coverage_pct),
            float(overall_progress),
            processing_time_ms,
        )

        return result
