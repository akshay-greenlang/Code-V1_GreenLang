# -*- coding: utf-8 -*-
"""
ClimateTransitionEngine - PACK-019 CSDDD Climate Transition Plan Engine
========================================================================

Assesses corporate climate transition plans per Article 22 of the
EU Corporate Sustainability Due Diligence Directive (CSDDD / CS3D)
and their alignment with the Paris Agreement objectives.

Article 22 requires companies falling within the scope of the CSDDD
to adopt and put into effect a transition plan for climate change
mitigation which aims to ensure, through best efforts, that the
business model and strategy of the company are compatible with the
transition to a sustainable economy and with the limiting of global
warming to 1.5 degrees Celsius in line with the Paris Agreement.

The engine evaluates:
    - Target ambition against 1.5C and well-below-2C pathways
    - Pathway trajectory (linear vs exponential reduction)
    - Implementation readiness (governance, resources, KPIs)
    - Alignment with Science Based Targets initiative (SBTi)
    - Completeness of transition plan elements per Art 22

CSDDD Article 22 Requirements:
    - Art 22(1): Companies shall adopt a transition plan for
      climate change mitigation
    - Art 22(2): The plan shall contain time-bound targets for
      2030 and in five-year steps up to 2050
    - Art 22(3): The plan shall be updated annually
    - Art 22(4): The plan shall be based on the latest available
      scientific evidence

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Article 22: Combating climate change
    - Paris Agreement (2015) Article 2
    - IPCC AR6 (2021-2023) - pathways for 1.5C
    - Science Based Targets initiative (SBTi) Corporate Net-Zero
      Standard v1.1
    - EU Taxonomy Climate Delegated Act 2021/2139
    - ESRS E1-1: Transition plan for climate change mitigation
    - TPT Disclosure Framework (2023)

Zero-Hallucination:
    - Target reduction rates computed from base/target year arithmetic
    - SBTi alignment uses deterministic threshold comparisons
    - Pathway assessment uses linear interpolation vs actual
    - Element scoring uses boolean counting
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


class TransitionPlanStatus(str, Enum):
    """Status of the corporate climate transition plan.

    Tracks the lifecycle stage of the transition plan from
    initial drafting through to target achievement, per
    Art 22 CSDDD requirements.
    """
    DRAFTED = "drafted"
    APPROVED = "approved"
    IMPLEMENTING = "implementing"
    ON_TRACK = "on_track"
    BEHIND_SCHEDULE = "behind_schedule"
    ACHIEVED = "achieved"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes for target setting.

    Per the GHG Protocol Corporate Standard, emissions are
    categorised into three scopes for target coverage assessment.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class AlignmentLevel(str, Enum):
    """Climate scenario alignment classification.

    Based on IPCC AR6 and SBTi frameworks, the alignment level
    indicates how the company's targets compare to global
    temperature pathways.
    """
    PARIS_ALIGNED_15C = "paris_aligned_1_5c"
    WELL_BELOW_2C = "well_below_2c"
    BELOW_2C = "below_2c"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


class TransitionElement(str, Enum):
    """Required elements of a climate transition plan per Art 22 CSDDD.

    Art 22 and the TPT Disclosure Framework define key elements
    that a credible transition plan must contain.
    """
    TARGETS = "targets"
    DECARBONIZATION_LEVERS = "decarbonization_levers"
    INVESTMENT_PLAN = "investment_plan"
    GOVERNANCE = "governance"
    ENGAGEMENT = "engagement"
    MONITORING = "monitoring"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SBTi benchmark annual linear reduction rates (% per year from
# base year).  Based on SBTi Corporate Net-Zero Standard v1.1.
SBTI_15C_ANNUAL_REDUCTION_PCT: Decimal = Decimal("4.2")
SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT: Decimal = Decimal("2.5")
SBTI_2C_ANNUAL_REDUCTION_PCT: Decimal = Decimal("1.8")

# Art 22(2) milestone years
ART_22_MILESTONE_YEARS: List[int] = [2030, 2035, 2040, 2045, 2050]

# Required transition plan elements per Art 22 and TPT
REQUIRED_ELEMENTS: List[str] = [
    TransitionElement.TARGETS.value,
    TransitionElement.DECARBONIZATION_LEVERS.value,
    TransitionElement.INVESTMENT_PLAN.value,
    TransitionElement.GOVERNANCE.value,
    TransitionElement.ENGAGEMENT.value,
    TransitionElement.MONITORING.value,
]

# Scope 3 relevance threshold (% of total emissions)
SCOPE_3_MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("40")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class InterimMilestone(BaseModel):
    """An interim milestone within a climate target trajectory.

    Represents a specific year-level checkpoint on the pathway
    between the base year and target year.
    """
    year: int = Field(
        ..., description="Milestone year", ge=2020, le=2100
    )
    reduction_pct: Decimal = Field(
        ...,
        description="Cumulative reduction % from base year at this milestone",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    emissions_tco2e: Optional[Decimal] = Field(
        default=None,
        description="Absolute emissions at this milestone (tCO2e)",
        ge=Decimal("0"),
    )
    status: str = Field(
        default="planned",
        description="Status of this milestone (planned, on_track, behind, achieved)",
    )


class ClimateTarget(BaseModel):
    """A climate emission reduction target for a specific scope.

    Represents one emission reduction target with base year,
    target year, reduction percentage, and interim milestones,
    per Art 22(2) CSDDD requirements.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this target",
    )
    scope: EmissionScope = Field(
        ...,
        description="Emission scope this target covers",
    )
    base_year: int = Field(
        ...,
        description="Base year for the target",
        ge=2015,
        le=2030,
    )
    base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Base year emissions in tCO2e",
        ge=Decimal("0"),
    )
    target_year: int = Field(
        ...,
        description="Year by which the target must be achieved",
        ge=2025,
        le=2100,
    )
    reduction_pct: Decimal = Field(
        ...,
        description="Target reduction percentage from base year",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    current_year: int = Field(
        default=2025,
        description="Current reporting year",
        ge=2020,
    )
    current_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Current year emissions in tCO2e",
        ge=Decimal("0"),
    )
    aligned_with_15c: bool = Field(
        default=False,
        description="Whether this target is claimed to be 1.5C aligned",
    )
    is_sbti_validated: bool = Field(
        default=False,
        description="Whether this target has been validated by SBTi",
    )
    is_net_zero_target: bool = Field(
        default=False,
        description="Whether this is a net-zero target",
    )
    interim_milestones: List[InterimMilestone] = Field(
        default_factory=list,
        description="Interim milestones between base and target year",
    )
    description: str = Field(
        default="",
        description="Description of the target",
        max_length=2000,
    )

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info: Any) -> int:
        """Validate target year is after base year."""
        base = info.data.get("base_year", 2020)
        if v <= base:
            raise ValueError(
                f"target_year ({v}) must be after base_year ({base})"
            )
        return v


class TransitionPlanDetails(BaseModel):
    """Structural details of the climate transition plan per Art 22.

    Describes the plan's governance, investment, engagement,
    and monitoring arrangements.
    """
    plan_id: str = Field(
        default_factory=_new_uuid,
        description="Unique plan identifier",
    )
    status: TransitionPlanStatus = Field(
        default=TransitionPlanStatus.DRAFTED,
        description="Current status of the transition plan",
    )
    has_targets: bool = Field(
        default=False,
        description="Whether GHG reduction targets are set",
    )
    has_decarbonization_levers: bool = Field(
        default=False,
        description="Whether decarbonization levers are identified",
    )
    decarbonization_levers: List[str] = Field(
        default_factory=list,
        description="List of identified decarbonization levers",
    )
    has_investment_plan: bool = Field(
        default=False,
        description="Whether CapEx/OpEx allocation is defined",
    )
    total_investment_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total planned investment for transition (EUR)",
        ge=Decimal("0"),
    )
    has_governance: bool = Field(
        default=False,
        description="Whether governance oversight is established",
    )
    board_oversight: bool = Field(
        default=False,
        description="Whether the board oversees the transition plan",
    )
    dedicated_team: bool = Field(
        default=False,
        description="Whether a dedicated sustainability/transition team exists",
    )
    kpis_defined: bool = Field(
        default=False,
        description="Whether KPIs for transition progress are defined",
    )
    has_engagement: bool = Field(
        default=False,
        description="Whether stakeholder engagement on transition is conducted",
    )
    has_monitoring: bool = Field(
        default=False,
        description="Whether monitoring and review processes are established",
    )
    annual_review: bool = Field(
        default=False,
        description="Whether the plan is reviewed annually per Art 22(3)",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Date of last plan update",
    )
    scope_3_included: bool = Field(
        default=False,
        description="Whether Scope 3 emissions are included in the plan",
    )
    just_transition_considered: bool = Field(
        default=False,
        description="Whether just transition aspects are addressed",
    )
    locked_in_emissions_assessed: bool = Field(
        default=False,
        description="Whether locked-in emissions have been assessed",
    )
    scenario_analysis_conducted: bool = Field(
        default=False,
        description="Whether climate scenario analysis has been conducted",
    )


class TransitionPlanAssessment(BaseModel):
    """Assessment of a single transition plan element."""
    element: str = Field(
        ..., description="Transition plan element assessed"
    )
    status: str = Field(
        default="not_addressed", description="Element status"
    )
    score: Decimal = Field(
        default=Decimal("0"),
        description="Score for this element (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps for this element",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for this element",
    )


class TargetAnalysis(BaseModel):
    """Analysis of a single climate target against benchmarks."""
    target_id: str = Field(
        default="", description="Target identifier"
    )
    scope: str = Field(
        default="", description="Emission scope"
    )
    annual_reduction_rate_pct: Decimal = Field(
        default=Decimal("0"),
        description="Implied annual linear reduction rate (%)",
    )
    sbti_15c_threshold_pct: Decimal = Field(
        default=SBTI_15C_ANNUAL_REDUCTION_PCT,
        description="SBTi 1.5C annual threshold (%)",
    )
    sbti_wb2c_threshold_pct: Decimal = Field(
        default=SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT,
        description="SBTi well-below-2C annual threshold (%)",
    )
    alignment_level: str = Field(
        default=AlignmentLevel.INSUFFICIENT_DATA.value,
        description="Alignment level determination",
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"),
        description="Progress towards target (%)",
    )
    is_on_track: bool = Field(
        default=False,
        description="Whether progress is on track for linear pathway",
    )
    expected_reduction_by_now_pct: Decimal = Field(
        default=Decimal("0"),
        description="Expected reduction by current year on linear path (%)",
    )
    actual_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Actual reduction achieved to date (%)",
    )
    gap_pp: Decimal = Field(
        default=Decimal("0"),
        description="Gap in percentage points (positive = behind)",
    )


class ClimateTransitionResult(BaseModel):
    """Complete climate transition plan assessment result per Art 22 CSDDD.

    Aggregates target analysis, plan element assessments, alignment
    determination, and implementation readiness into a single result
    with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version used"
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of assessment (UTC)"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    targets_count: int = Field(
        default=0, description="Number of targets assessed"
    )
    target_analysis: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-target analysis results",
    )
    plan_assessment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-element plan assessment results",
    )
    alignment_level: str = Field(
        default=AlignmentLevel.INSUFFICIENT_DATA.value,
        description="Overall alignment level",
    )
    overall_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall transition plan score (0-100)",
    )
    pathway_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pathway trajectory assessment",
    )
    implementation_readiness: Dict[str, Any] = Field(
        default_factory=dict,
        description="Implementation readiness assessment",
    )
    scope_coverage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scope coverage analysis",
    )
    milestones_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Art 22(2) milestone assessment",
    )
    compliance_gaps: List[str] = Field(
        default_factory=list,
        description="Identified compliance gaps under Art 22",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and assessment steps",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ClimateTransitionEngine:
    """CSDDD Article 22 climate transition plan assessment engine.

    Provides deterministic, zero-hallucination assessments for
    corporate climate transition plans against Art 22 CSDDD
    requirements and Paris Agreement alignment:

    - Target ambition against SBTi 1.5C / well-below-2C benchmarks
    - Linear pathway trajectory tracking
    - Implementation readiness (governance, resources, KPIs)
    - Transition plan element completeness
    - Scope coverage analysis (S1, S2, S3)
    - Art 22(2) milestone year coverage

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = ClimateTransitionEngine()
        result = engine.assess_transition_plan(
            targets=[ClimateTarget(...)],
            plan_details=TransitionPlanDetails(...),
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Main Assessment Method                                               #
    # ------------------------------------------------------------------ #

    def assess_transition_plan(
        self,
        targets: List[ClimateTarget],
        plan_details: TransitionPlanDetails,
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> ClimateTransitionResult:
        """Perform a complete climate transition plan assessment.

        Orchestrates evaluation of targets, plan elements, alignment,
        pathway, and implementation readiness per Art 22 CSDDD.

        Args:
            targets: List of ClimateTarget instances.
            plan_details: TransitionPlanDetails describing the plan.
            entity_name: Name of the reporting entity.
            reporting_year: Current reporting year.

        Returns:
            ClimateTransitionResult with complete assessment and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing transition plan: %d targets, entity=%s, year=%d",
            len(targets), entity_name, reporting_year,
        )

        # Step 1: Assess each target
        target_analyses = [
            self.assess_target_ambition(target)
            for target in targets
        ]

        # Step 2: Assess plan elements
        plan_assessments = self._assess_plan_elements(plan_details)

        # Step 3: Calculate overall alignment
        alignment_level = self.calculate_alignment(targets)

        # Step 4: Assess pathway
        pathway_assessment = self.assess_pathway(targets)

        # Step 5: Assess implementation readiness
        implementation_readiness = self.assess_implementation_readiness(
            plan_details
        )

        # Step 6: Scope coverage
        scope_coverage = self._assess_scope_coverage(targets)

        # Step 7: Milestone assessment per Art 22(2)
        milestones_assessment = self._assess_milestones(targets)

        # Step 8: Calculate overall score
        overall_score = self._calculate_overall_score(
            target_analyses, plan_assessments,
            implementation_readiness, alignment_level,
        )

        # Step 9: Compliance gaps
        compliance_gaps = self._identify_compliance_gaps(
            targets, plan_details, target_analyses,
            plan_assessments, scope_coverage, milestones_assessment,
        )

        # Step 10: Recommendations
        recommendations = self._generate_recommendations(
            targets, plan_details, target_analyses,
            plan_assessments, alignment_level,
            implementation_readiness, scope_coverage,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ClimateTransitionResult(
            entity_name=entity_name,
            reporting_year=reporting_year,
            targets_count=len(targets),
            target_analysis=target_analyses,
            plan_assessment=plan_assessments,
            alignment_level=alignment_level,
            overall_score=overall_score,
            pathway_assessment=pathway_assessment,
            implementation_readiness=implementation_readiness,
            scope_coverage=scope_coverage,
            milestones_assessment=milestones_assessment,
            compliance_gaps=compliance_gaps,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Transition plan assessed: alignment=%s, score=%.1f%%, "
            "targets=%d, hash=%s",
            alignment_level, float(overall_score), len(targets),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Target Ambition Assessment                                           #
    # ------------------------------------------------------------------ #

    def assess_target_ambition(
        self, target: ClimateTarget
    ) -> Dict[str, Any]:
        """Assess a single climate target's ambition against SBTi benchmarks.

        Calculates the implied annual linear reduction rate and
        compares it to the SBTi 1.5C and well-below-2C thresholds.
        Also evaluates progress against the linear pathway.

        Args:
            target: ClimateTarget to assess.

        Returns:
            Dict with target analysis metrics and alignment level.
        """
        years = target.target_year - target.base_year
        if years <= 0:
            logger.warning(
                "Target %s has invalid year range: base=%d, target=%d",
                target.target_id, target.base_year, target.target_year,
            )
            return {
                "target_id": target.target_id,
                "scope": target.scope.value,
                "error": "Invalid year range",
                "alignment_level": AlignmentLevel.INSUFFICIENT_DATA.value,
            }

        # Implied annual linear reduction rate
        annual_rate = _round_val(
            target.reduction_pct / _decimal(years), 2
        )

        # Determine alignment level
        if annual_rate >= SBTI_15C_ANNUAL_REDUCTION_PCT:
            alignment = AlignmentLevel.PARIS_ALIGNED_15C.value
        elif annual_rate >= SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT:
            alignment = AlignmentLevel.WELL_BELOW_2C.value
        elif annual_rate >= SBTI_2C_ANNUAL_REDUCTION_PCT:
            alignment = AlignmentLevel.BELOW_2C.value
        else:
            alignment = AlignmentLevel.NOT_ALIGNED.value

        # Progress assessment
        elapsed_years = max(
            0, target.current_year - target.base_year
        )
        expected_reduction_by_now = _round_val(
            annual_rate * _decimal(elapsed_years), 2
        )

        # Actual reduction achieved
        actual_reduction = Decimal("0")
        if target.base_year_emissions_tco2e > Decimal("0"):
            actual_reduction = _round_val(
                (target.base_year_emissions_tco2e
                 - target.current_emissions_tco2e)
                / target.base_year_emissions_tco2e
                * Decimal("100"),
                2,
            )

        # Gap analysis
        gap_pp = _round_val(
            expected_reduction_by_now - actual_reduction, 2
        )
        is_on_track = actual_reduction >= expected_reduction_by_now

        # Progress percentage towards final target
        progress_pct = Decimal("0")
        if target.reduction_pct > Decimal("0"):
            progress_pct = _round_val(
                _safe_divide(
                    actual_reduction, target.reduction_pct
                ) * Decimal("100"),
                1,
            )
            progress_pct = min(progress_pct, Decimal("100"))

        result = {
            "target_id": target.target_id,
            "scope": target.scope.value,
            "base_year": target.base_year,
            "target_year": target.target_year,
            "reduction_pct": str(target.reduction_pct),
            "annual_reduction_rate_pct": str(annual_rate),
            "sbti_15c_threshold_pct": str(SBTI_15C_ANNUAL_REDUCTION_PCT),
            "sbti_wb2c_threshold_pct": str(
                SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT
            ),
            "alignment_level": alignment,
            "is_sbti_validated": target.is_sbti_validated,
            "progress_pct": str(progress_pct),
            "is_on_track": is_on_track,
            "expected_reduction_by_now_pct": str(expected_reduction_by_now),
            "actual_reduction_pct": str(actual_reduction),
            "gap_pp": str(gap_pp),
            "elapsed_years": elapsed_years,
            "remaining_years": max(
                0, target.target_year - target.current_year
            ),
        }

        logger.info(
            "Target %s (%s): annual rate=%.2f%%, alignment=%s, "
            "on_track=%s, progress=%.1f%%",
            target.target_id, target.scope.value,
            float(annual_rate), alignment, is_on_track,
            float(progress_pct),
        )

        return result

    # ------------------------------------------------------------------ #
    # Pathway Assessment                                                   #
    # ------------------------------------------------------------------ #

    def assess_pathway(
        self, targets: List[ClimateTarget]
    ) -> Dict[str, Any]:
        """Assess the emission reduction pathway across all targets.

        Evaluates whether the combined targets define a credible
        trajectory from current emissions to long-term goals,
        including assessment of interim milestones.

        Args:
            targets: List of ClimateTarget instances.

        Returns:
            Dict with pathway assessment metrics.
        """
        if not targets:
            return {
                "has_pathway": False,
                "pathway_type": "none",
                "scopes_covered": [],
                "milestones_defined": 0,
                "provenance_hash": _compute_hash({"empty": True}),
            }

        # Check scope coverage in targets
        scopes_covered = sorted(set(
            t.scope.value for t in targets
        ))

        # Count milestones across all targets
        total_milestones = sum(
            len(t.interim_milestones) for t in targets
        )

        # Assess pathway consistency
        # For each scope, check that targets are sequential and
        # reduction increases over time
        pathway_consistent = True
        pathway_issues: List[str] = []

        scope_targets: Dict[str, List[ClimateTarget]] = {}
        for t in targets:
            sv = t.scope.value
            if sv not in scope_targets:
                scope_targets[sv] = []
            scope_targets[sv].append(t)

        for scope, scope_ts in scope_targets.items():
            sorted_ts = sorted(scope_ts, key=lambda x: x.target_year)
            for i in range(1, len(sorted_ts)):
                prev = sorted_ts[i - 1]
                curr = sorted_ts[i]
                if curr.reduction_pct < prev.reduction_pct:
                    pathway_consistent = False
                    pathway_issues.append(
                        f"{scope}: Target for {curr.target_year} has lower "
                        f"reduction ({curr.reduction_pct}%) than target "
                        f"for {prev.target_year} ({prev.reduction_pct}%)"
                    )

        # Determine pathway type
        if not pathway_consistent:
            pathway_type = "inconsistent"
        elif total_milestones > 0:
            pathway_type = "milestone_based"
        elif len(targets) > 1:
            pathway_type = "multi_target"
        else:
            pathway_type = "single_target"

        # Check if long-term (2050) target exists
        has_2050_target = any(
            t.target_year >= 2050 for t in targets
        )
        has_2030_target = any(
            t.target_year <= 2030 for t in targets
        )

        result = {
            "has_pathway": True,
            "pathway_type": pathway_type,
            "pathway_consistent": pathway_consistent,
            "pathway_issues": pathway_issues,
            "scopes_covered": scopes_covered,
            "scope_count": len(scopes_covered),
            "targets_count": len(targets),
            "milestones_defined": total_milestones,
            "has_2030_target": has_2030_target,
            "has_2050_target": has_2050_target,
            "has_net_zero_target": any(t.is_net_zero_target for t in targets),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Pathway: type=%s, consistent=%s, scopes=%s, "
            "milestones=%d, 2030=%s, 2050=%s",
            pathway_type, pathway_consistent,
            scopes_covered, total_milestones,
            has_2030_target, has_2050_target,
        )

        return result

    # ------------------------------------------------------------------ #
    # Implementation Readiness                                             #
    # ------------------------------------------------------------------ #

    def assess_implementation_readiness(
        self, plan: TransitionPlanDetails
    ) -> Dict[str, Any]:
        """Assess implementation readiness of the transition plan.

        Evaluates governance structures, resources, KPIs, and
        operational arrangements for plan execution.

        Args:
            plan: TransitionPlanDetails describing the plan.

        Returns:
            Dict with implementation readiness assessment.
        """
        readiness_checks = {
            "board_oversight": plan.board_oversight,
            "dedicated_team": plan.dedicated_team,
            "kpis_defined": plan.kpis_defined,
            "investment_allocated": plan.has_investment_plan,
            "annual_review_process": plan.annual_review,
            "monitoring_established": plan.has_monitoring,
            "scope_3_included": plan.scope_3_included,
            "scenario_analysis": plan.scenario_analysis_conducted,
            "just_transition": plan.just_transition_considered,
            "locked_in_assessed": plan.locked_in_emissions_assessed,
        }

        met_count = sum(1 for v in readiness_checks.values() if v)
        total = len(readiness_checks)
        readiness_score = _pct(met_count, total)

        # Readiness level
        if readiness_score >= Decimal("80"):
            readiness_level = "high"
        elif readiness_score >= Decimal("50"):
            readiness_level = "moderate"
        elif readiness_score >= Decimal("25"):
            readiness_level = "low"
        else:
            readiness_level = "insufficient"

        # Gaps
        gaps = [k for k, v in readiness_checks.items() if not v]

        # Recommendations
        recs: List[str] = []
        if not plan.board_oversight:
            recs.append(
                "Establish board-level oversight of the climate "
                "transition plan as required by Art 22 governance."
            )
        if not plan.kpis_defined:
            recs.append(
                "Define measurable KPIs for monitoring transition "
                "plan progress."
            )
        if not plan.annual_review:
            recs.append(
                "Implement annual review process per Art 22(3) CSDDD."
            )
        if not plan.scope_3_included:
            recs.append(
                "Include Scope 3 emissions in the transition plan "
                "for comprehensive value chain coverage."
            )
        if not plan.scenario_analysis_conducted:
            recs.append(
                "Conduct climate scenario analysis to stress-test "
                "the transition plan against different pathways."
            )
        if not plan.locked_in_emissions_assessed:
            recs.append(
                "Assess locked-in emissions from existing assets "
                "and contracts to identify stranded asset risk."
            )
        if not plan.has_investment_plan:
            recs.append(
                "Develop a CapEx/OpEx allocation plan for "
                "transition activities."
            )

        result = {
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "checks": readiness_checks,
            "met_count": met_count,
            "total_checks": total,
            "gaps": gaps,
            "recommendations": recs,
            "plan_status": plan.status.value,
            "total_investment_eur": str(plan.total_investment_eur),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Implementation readiness: %.1f%% (%s), %d/%d checks met",
            float(readiness_score), readiness_level, met_count, total,
        )

        return result

    # ------------------------------------------------------------------ #
    # Alignment Calculation                                                #
    # ------------------------------------------------------------------ #

    def calculate_alignment(
        self, targets: List[ClimateTarget]
    ) -> str:
        """Calculate overall alignment level from all targets.

        Determines the aggregate alignment by taking the lowest
        alignment level across all targets (conservative approach).

        Args:
            targets: List of ClimateTarget instances.

        Returns:
            AlignmentLevel value string.
        """
        if not targets:
            return AlignmentLevel.INSUFFICIENT_DATA.value

        alignment_hierarchy = [
            AlignmentLevel.PARIS_ALIGNED_15C.value,
            AlignmentLevel.WELL_BELOW_2C.value,
            AlignmentLevel.BELOW_2C.value,
            AlignmentLevel.NOT_ALIGNED.value,
        ]

        # Calculate alignment for each target
        target_alignments: List[str] = []
        for target in targets:
            years = target.target_year - target.base_year
            if years <= 0:
                target_alignments.append(
                    AlignmentLevel.INSUFFICIENT_DATA.value
                )
                continue

            annual_rate = target.reduction_pct / _decimal(years)

            if annual_rate >= SBTI_15C_ANNUAL_REDUCTION_PCT:
                target_alignments.append(
                    AlignmentLevel.PARIS_ALIGNED_15C.value
                )
            elif annual_rate >= SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT:
                target_alignments.append(
                    AlignmentLevel.WELL_BELOW_2C.value
                )
            elif annual_rate >= SBTI_2C_ANNUAL_REDUCTION_PCT:
                target_alignments.append(
                    AlignmentLevel.BELOW_2C.value
                )
            else:
                target_alignments.append(
                    AlignmentLevel.NOT_ALIGNED.value
                )

        # Take the least ambitious (highest index in hierarchy)
        worst_index = 0
        for ta in target_alignments:
            if ta == AlignmentLevel.INSUFFICIENT_DATA.value:
                continue
            if ta in alignment_hierarchy:
                idx = alignment_hierarchy.index(ta)
                worst_index = max(worst_index, idx)

        overall = alignment_hierarchy[worst_index]

        logger.info(
            "Overall alignment: %s (from %d targets)",
            overall, len(targets),
        )

        return overall

    # ------------------------------------------------------------------ #
    # Plan Element Assessment                                              #
    # ------------------------------------------------------------------ #

    def _assess_plan_elements(
        self, plan: TransitionPlanDetails
    ) -> List[Dict[str, Any]]:
        """Assess each required transition plan element.

        Args:
            plan: TransitionPlanDetails describing the plan.

        Returns:
            List of element assessment dicts.
        """
        assessments: List[Dict[str, Any]] = []

        # Targets
        targets_score = Decimal("100") if plan.has_targets else Decimal("0")
        assessments.append({
            "element": TransitionElement.TARGETS.value,
            "status": "addressed" if plan.has_targets else "not_addressed",
            "score": targets_score,
            "gaps": [] if plan.has_targets else [
                "No GHG reduction targets defined"
            ],
            "recommendations": [] if plan.has_targets else [
                "Set science-based GHG reduction targets covering "
                "Scope 1, 2, and material Scope 3 categories."
            ],
        })

        # Decarbonization levers
        levers_present = plan.has_decarbonization_levers
        lever_count = len(plan.decarbonization_levers)
        levers_score = Decimal("0")
        if levers_present and lever_count >= 3:
            levers_score = Decimal("100")
        elif levers_present:
            levers_score = _round_val(
                _decimal(lever_count) / Decimal("3") * Decimal("100"), 1
            )
            levers_score = min(levers_score, Decimal("100"))

        levers_gaps: List[str] = []
        levers_recs: List[str] = []
        if not levers_present:
            levers_gaps.append("No decarbonization levers identified")
            levers_recs.append(
                "Identify key decarbonization levers (energy efficiency, "
                "fuel switching, electrification, etc.)."
            )
        elif lever_count < 3:
            levers_gaps.append(
                f"Only {lever_count} lever(s) identified; consider more"
            )
            levers_recs.append(
                "Expand the range of decarbonization levers to "
                "build a more resilient transition pathway."
            )

        assessments.append({
            "element": TransitionElement.DECARBONIZATION_LEVERS.value,
            "status": "addressed" if levers_present else "not_addressed",
            "score": levers_score,
            "lever_count": lever_count,
            "gaps": levers_gaps,
            "recommendations": levers_recs,
        })

        # Investment plan
        inv_score = Decimal("0")
        if plan.has_investment_plan and plan.total_investment_eur > Decimal("0"):
            inv_score = Decimal("100")
        elif plan.has_investment_plan:
            inv_score = Decimal("50")

        assessments.append({
            "element": TransitionElement.INVESTMENT_PLAN.value,
            "status": "addressed" if plan.has_investment_plan else "not_addressed",
            "score": inv_score,
            "total_investment_eur": str(plan.total_investment_eur),
            "gaps": [] if plan.has_investment_plan else [
                "No investment plan defined"
            ],
            "recommendations": [] if plan.has_investment_plan else [
                "Develop a CapEx/OpEx allocation plan for transition "
                "activities with annual budgets."
            ],
        })

        # Governance
        gov_checks = [
            plan.board_oversight,
            plan.dedicated_team,
            plan.kpis_defined,
        ]
        gov_met = sum(1 for c in gov_checks if c)
        gov_score = _pct(gov_met, len(gov_checks))

        gov_gaps: List[str] = []
        gov_recs: List[str] = []
        if not plan.board_oversight:
            gov_gaps.append("No board oversight")
            gov_recs.append("Establish board oversight of the plan.")
        if not plan.dedicated_team:
            gov_gaps.append("No dedicated team")
            gov_recs.append("Assign a dedicated sustainability team.")
        if not plan.kpis_defined:
            gov_gaps.append("No KPIs defined")
            gov_recs.append("Define measurable KPIs for progress tracking.")

        assessments.append({
            "element": TransitionElement.GOVERNANCE.value,
            "status": "addressed" if gov_met > 0 else "not_addressed",
            "score": gov_score,
            "gaps": gov_gaps,
            "recommendations": gov_recs,
        })

        # Engagement
        eng_score = (
            Decimal("100") if plan.has_engagement else Decimal("0")
        )
        assessments.append({
            "element": TransitionElement.ENGAGEMENT.value,
            "status": "addressed" if plan.has_engagement else "not_addressed",
            "score": eng_score,
            "gaps": [] if plan.has_engagement else [
                "No stakeholder engagement on transition"
            ],
            "recommendations": [] if plan.has_engagement else [
                "Conduct stakeholder engagement on the transition "
                "plan including workers, communities, and investors."
            ],
        })

        # Monitoring
        mon_checks = [
            plan.has_monitoring,
            plan.annual_review,
        ]
        mon_met = sum(1 for c in mon_checks if c)
        mon_score = _pct(mon_met, len(mon_checks))

        mon_gaps: List[str] = []
        mon_recs: List[str] = []
        if not plan.has_monitoring:
            mon_gaps.append("No monitoring process")
            mon_recs.append("Establish monitoring process for plan progress.")
        if not plan.annual_review:
            mon_gaps.append("No annual review")
            mon_recs.append(
                "Implement annual review per Art 22(3) CSDDD."
            )

        assessments.append({
            "element": TransitionElement.MONITORING.value,
            "status": "addressed" if mon_met > 0 else "not_addressed",
            "score": mon_score,
            "gaps": mon_gaps,
            "recommendations": mon_recs,
        })

        return assessments

    # ------------------------------------------------------------------ #
    # Scope Coverage Assessment                                            #
    # ------------------------------------------------------------------ #

    def _assess_scope_coverage(
        self, targets: List[ClimateTarget]
    ) -> Dict[str, Any]:
        """Assess which emission scopes are covered by targets.

        Args:
            targets: List of ClimateTarget instances.

        Returns:
            Dict with scope coverage analysis.
        """
        all_scopes = set(s.value for s in EmissionScope)
        covered_scopes: Dict[str, int] = {}

        for target in targets:
            sv = target.scope.value
            covered_scopes[sv] = covered_scopes.get(sv, 0) + 1

        covered = set(covered_scopes.keys())
        missing = sorted(all_scopes - covered)
        coverage_pct = _pct(len(covered), len(all_scopes))

        has_scope_3 = EmissionScope.SCOPE_3.value in covered

        result = {
            "total_scopes": len(all_scopes),
            "scopes_covered": len(covered),
            "scopes_missing": missing,
            "coverage_pct": coverage_pct,
            "targets_by_scope": covered_scopes,
            "has_scope_3": has_scope_3,
            "scope_3_note": (
                "Scope 3 is included in targets."
                if has_scope_3
                else "Scope 3 is not covered. Art 22 and SBTi "
                     "require Scope 3 inclusion when material."
            ),
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    # ------------------------------------------------------------------ #
    # Milestone Assessment per Art 22(2)                                   #
    # ------------------------------------------------------------------ #

    def _assess_milestones(
        self, targets: List[ClimateTarget]
    ) -> Dict[str, Any]:
        """Assess coverage of Art 22(2) milestone years.

        Art 22(2) requires time-bound targets for 2030 and in
        five-year steps up to 2050.

        Args:
            targets: List of ClimateTarget instances.

        Returns:
            Dict with milestone coverage analysis.
        """
        # Collect all target years and milestone years
        covered_years: set = set()
        for target in targets:
            covered_years.add(target.target_year)
            for ms in target.interim_milestones:
                covered_years.add(ms.year)

        required = set(ART_22_MILESTONE_YEARS)
        covered = required & covered_years
        missing = sorted(required - covered_years)

        coverage_pct = _pct(len(covered), len(required))

        result = {
            "required_milestone_years": ART_22_MILESTONE_YEARS,
            "covered_milestone_years": sorted(covered),
            "missing_milestone_years": missing,
            "coverage_pct": coverage_pct,
            "all_target_years": sorted(covered_years),
            "is_complete": len(missing) == 0,
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    # ------------------------------------------------------------------ #
    # Overall Score Calculation                                            #
    # ------------------------------------------------------------------ #

    def _calculate_overall_score(
        self,
        target_analyses: List[Dict[str, Any]],
        plan_assessments: List[Dict[str, Any]],
        implementation_readiness: Dict[str, Any],
        alignment_level: str,
    ) -> Decimal:
        """Calculate overall transition plan score.

        Weighted average:
        - Target ambition: 30%
        - Plan elements: 30%
        - Implementation readiness: 25%
        - Alignment level: 15%

        Args:
            target_analyses: Per-target analysis results.
            plan_assessments: Per-element plan assessment results.
            implementation_readiness: Readiness assessment.
            alignment_level: Overall alignment level.

        Returns:
            Overall score as Decimal (0-100).
        """
        # Target ambition score (average of on_track status)
        target_score = Decimal("0")
        if target_analyses:
            on_track_count = sum(
                1 for ta in target_analyses
                if ta.get("is_on_track", False)
            )
            target_score = _pct(on_track_count, len(target_analyses))

        # Plan element score (average of element scores)
        element_score = Decimal("0")
        if plan_assessments:
            element_scores = [
                _decimal(a.get("score", 0)) for a in plan_assessments
            ]
            element_score = _round_val(
                sum(element_scores) / _decimal(len(element_scores)), 1
            )

        # Readiness score
        readiness_score = _decimal(
            implementation_readiness.get("readiness_score", 0)
        )

        # Alignment score
        alignment_scores = {
            AlignmentLevel.PARIS_ALIGNED_15C.value: Decimal("100"),
            AlignmentLevel.WELL_BELOW_2C.value: Decimal("75"),
            AlignmentLevel.BELOW_2C.value: Decimal("50"),
            AlignmentLevel.NOT_ALIGNED.value: Decimal("20"),
            AlignmentLevel.INSUFFICIENT_DATA.value: Decimal("0"),
        }
        align_score = alignment_scores.get(
            alignment_level, Decimal("0")
        )

        # Weighted average
        overall = _round_val(
            target_score * Decimal("0.30")
            + element_score * Decimal("0.30")
            + readiness_score * Decimal("0.25")
            + align_score * Decimal("0.15"),
            1,
        )

        return overall

    # ------------------------------------------------------------------ #
    # Compliance Gap Identification                                        #
    # ------------------------------------------------------------------ #

    def _identify_compliance_gaps(
        self,
        targets: List[ClimateTarget],
        plan: TransitionPlanDetails,
        target_analyses: List[Dict[str, Any]],
        plan_assessments: List[Dict[str, Any]],
        scope_coverage: Dict[str, Any],
        milestones_assessment: Dict[str, Any],
    ) -> List[str]:
        """Identify compliance gaps under CSDDD Art 22.

        Args:
            targets: Climate targets.
            plan: Plan details.
            target_analyses: Per-target analyses.
            plan_assessments: Per-element assessments.
            scope_coverage: Scope coverage analysis.
            milestones_assessment: Milestone coverage.

        Returns:
            List of compliance gap descriptions.
        """
        gaps: List[str] = []

        # No targets set
        if not targets:
            gaps.append(
                "Art 22(1): No climate targets have been set. "
                "A transition plan requires time-bound GHG "
                "reduction targets."
            )

        # Targets not aligned
        not_aligned = [
            ta for ta in target_analyses
            if ta.get("alignment_level")
            == AlignmentLevel.NOT_ALIGNED.value
        ]
        if not_aligned:
            gaps.append(
                f"Art 22(1): {len(not_aligned)} target(s) are not "
                f"aligned with the Paris Agreement 1.5C objective."
            )

        # Missing plan elements
        for pa in plan_assessments:
            if pa.get("status") == "not_addressed":
                gaps.append(
                    f"Art 22: Transition plan element "
                    f"'{pa['element']}' is not addressed."
                )

        # Missing scopes
        missing_scopes = scope_coverage.get("scopes_missing", [])
        if missing_scopes:
            gaps.append(
                f"Art 22: Emission scopes not covered by targets: "
                f"{', '.join(missing_scopes)}"
            )

        # Missing milestones per Art 22(2)
        missing_ms = milestones_assessment.get(
            "missing_milestone_years", []
        )
        if missing_ms:
            gaps.append(
                f"Art 22(2): Milestone years not covered: "
                f"{', '.join(str(y) for y in missing_ms)}"
            )

        # Annual review not conducted per Art 22(3)
        if not plan.annual_review:
            gaps.append(
                "Art 22(3): Annual review of the transition plan "
                "has not been established."
            )

        # Plan not based on latest science per Art 22(4)
        if not plan.scenario_analysis_conducted:
            gaps.append(
                "Art 22(4): Climate scenario analysis has not been "
                "conducted. The plan should be based on latest "
                "available scientific evidence."
            )

        return gaps

    # ------------------------------------------------------------------ #
    # Recommendations Generation                                           #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        targets: List[ClimateTarget],
        plan: TransitionPlanDetails,
        target_analyses: List[Dict[str, Any]],
        plan_assessments: List[Dict[str, Any]],
        alignment_level: str,
        implementation_readiness: Dict[str, Any],
        scope_coverage: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for transition plan improvement.

        Args:
            targets: Climate targets.
            plan: Plan details.
            target_analyses: Per-target analyses.
            plan_assessments: Per-element assessments.
            alignment_level: Overall alignment.
            implementation_readiness: Readiness assessment.
            scope_coverage: Scope coverage.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: List[str] = []

        # No targets
        if not targets:
            recommendations.append(
                "Set science-based GHG reduction targets covering "
                "Scope 1, 2, and material Scope 3 categories, "
                "aligned with a 1.5C pathway."
            )
            return recommendations

        # Alignment improvement
        if alignment_level == AlignmentLevel.NOT_ALIGNED.value:
            recommendations.append(
                "Current targets are not aligned with the Paris "
                "Agreement. Increase target ambition to achieve at "
                "least 4.2% annual reduction (SBTi 1.5C benchmark)."
            )
        elif alignment_level == AlignmentLevel.BELOW_2C.value:
            recommendations.append(
                "Targets are aligned with below-2C but not 1.5C. "
                "Consider increasing ambition to meet the SBTi "
                "1.5C benchmark of 4.2% annual reduction."
            )

        # Targets behind schedule
        behind = [
            ta for ta in target_analyses
            if not ta.get("is_on_track", True)
        ]
        if behind:
            recommendations.append(
                f"{len(behind)} target(s) are behind schedule. "
                f"Review decarbonization levers and accelerate "
                f"implementation of abatement actions."
            )

        # SBTi validation
        unvalidated = [
            t for t in targets if not t.is_sbti_validated
        ]
        if unvalidated:
            recommendations.append(
                "Submit targets to SBTi for third-party validation "
                "to enhance credibility of the transition plan."
            )

        # Collect element recommendations
        for pa in plan_assessments:
            for rec in pa.get("recommendations", []):
                if rec not in recommendations:
                    recommendations.append(rec)

        # Readiness recommendations
        for rec in implementation_readiness.get("recommendations", []):
            if rec not in recommendations:
                recommendations.append(rec)

        # Scope 3
        if not scope_coverage.get("has_scope_3", False):
            recommendations.append(
                "Include Scope 3 emissions in targets. For most "
                "companies, Scope 3 represents the majority of "
                "total emissions."
            )

        # Net zero
        has_net_zero = any(t.is_net_zero_target for t in targets)
        if not has_net_zero:
            recommendations.append(
                "Consider setting a long-term net-zero target "
                "aligned with the SBTi Corporate Net-Zero Standard."
            )

        # Cap at 15
        if len(recommendations) > 15:
            recommendations = recommendations[:15]

        return recommendations

    # ------------------------------------------------------------------ #
    # Period Comparison                                                     #
    # ------------------------------------------------------------------ #

    def compare_periods(
        self,
        current: ClimateTransitionResult,
        previous: ClimateTransitionResult,
    ) -> Dict[str, Any]:
        """Compare transition plan performance across two periods.

        Args:
            current: Current period result.
            previous: Previous period result.

        Returns:
            Dict with period-over-period changes and provenance.
        """
        comparison = {
            "current_period": current.reporting_year,
            "previous_period": previous.reporting_year,
            "overall_score_change_pp": _round_val(
                current.overall_score - previous.overall_score, 1
            ),
            "alignment_change": {
                "from": previous.alignment_level,
                "to": current.alignment_level,
            },
            "targets_change": (
                current.targets_count - previous.targets_count
            ),
            "direction": (
                "improving"
                if current.overall_score > previous.overall_score
                else (
                    "stable"
                    if current.overall_score == previous.overall_score
                    else "declining"
                )
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison
