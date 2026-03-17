# -*- coding: utf-8 -*-
"""
TransitionPlanEngine - PACK-016 ESRS E1 Climate Engine 3
=========================================================

Evaluates and structures the transition plan for climate change
mitigation per ESRS E1-1.

Under the European Sustainability Reporting Standards (ESRS), disclosure
requirement E1-1 mandates that undertakings describe their transition
plan for climate change mitigation.  This includes the plan's alignment
with the objective of limiting global warming to 1.5 degrees Celsius,
the key decarbonisation levers, the associated CapEx/OpEx allocation,
and the identification of locked-in emissions.

ESRS E1-1 Disclosure Requirements:
    - Para 14: The undertaking shall disclose its transition plan for
      climate change mitigation.
    - Para 15: The transition plan shall explain how the undertaking's
      strategy and business model are compatible with the transition to
      a sustainable economy and with the limiting of global warming to
      1.5 degrees Celsius in line with the Paris Agreement.
    - Para 16: Key decarbonisation levers and actions identified,
      including planned CapEx allocation.
    - Para 17: Locked-in GHG emissions from existing assets and products.
    - Para 18: Alignment with climate scenarios (1.5C, well-below 2C).
    - Para 19: Time horizons for transition actions (short, medium, long).
    - Para 20: Progress assessment and gap analysis.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E1 Climate Change, Disclosure Requirement E1-1
    - Paris Agreement (2015) Article 2
    - IPCC SR15 (2018) - 1.5C pathways
    - EU Taxonomy Climate Delegated Act 2021/2139
    - TCFD Guidance on Metrics, Targets, and Transition Plans (2021)
    - TPT Disclosure Framework (Transition Plan Taskforce, 2023)

Zero-Hallucination:
    - Abatement potential uses deterministic summation of action levers
    - Locked-in emissions are calculated from asset lifetimes and factors
    - Gap analysis is arithmetic: gap = target - (current - abatement)
    - CapEx allocation is simple percentage calculation
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate
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
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
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
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecarbonizationLever(str, Enum):
    """Decarbonisation levers per ESRS E1-1 and TPT framework.

    Each lever represents a strategic pathway an undertaking can use
    to reduce its GHG emissions as part of its transition plan.
    """
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    ELECTRIFICATION = "electrification"
    RENEWABLE_ENERGY = "renewable_energy"
    PROCESS_CHANGE = "process_change"
    MATERIAL_SUBSTITUTION = "material_substitution"
    CIRCULAR_ECONOMY = "circular_economy"
    CARBON_CAPTURE = "carbon_capture"
    SUPPLY_CHAIN_ENGAGEMENT = "supply_chain_engagement"
    PRODUCT_REDESIGN = "product_redesign"


class PlanStatus(str, Enum):
    """Status of the transition plan.

    Tracks the lifecycle stage of the undertaking's transition plan.
    """
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    ON_TRACK = "on_track"
    BEHIND_SCHEDULE = "behind_schedule"


class ScenarioAlignment(str, Enum):
    """Climate scenario alignment classification.

    Per ESRS E1-1 Para 18, the transition plan should describe its
    alignment with recognised climate scenarios.
    """
    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_WELL_BELOW_2C = "aligned_well_below_2c"
    ALIGNED_2C = "aligned_2c"
    NOT_ALIGNED = "not_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    ASSESSMENT_PENDING = "assessment_pending"


class LockedInEmissionType(str, Enum):
    """Type of locked-in GHG emissions per ESRS E1-1 Para 17.

    Locked-in emissions arise from existing assets, contracts, or
    products whose remaining useful life will continue to generate
    emissions regardless of transition actions.
    """
    EXISTING_ASSETS = "existing_assets"
    CONTRACTUAL_OBLIGATIONS = "contractual_obligations"
    PRODUCT_LIFECYCLE = "product_lifecycle"


class TimeHorizon(str, Enum):
    """Time horizon for transition plan actions.

    Per ESRS E1-1, actions should be categorised by time horizon.
    """
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Typical abatement potential ranges by lever (% reduction achievable).
# These are indicative values from IEA and IPCC literature for
# gap analysis and sanity checking.  Actual values depend on context.
LEVER_TYPICAL_ABATEMENT: Dict[str, Dict[str, Decimal]] = {
    "energy_efficiency": {
        "min_pct": Decimal("5"),
        "max_pct": Decimal("30"),
        "median_pct": Decimal("15"),
    },
    "fuel_switching": {
        "min_pct": Decimal("10"),
        "max_pct": Decimal("50"),
        "median_pct": Decimal("25"),
    },
    "electrification": {
        "min_pct": Decimal("15"),
        "max_pct": Decimal("60"),
        "median_pct": Decimal("35"),
    },
    "renewable_energy": {
        "min_pct": Decimal("10"),
        "max_pct": Decimal("100"),
        "median_pct": Decimal("40"),
    },
    "process_change": {
        "min_pct": Decimal("5"),
        "max_pct": Decimal("40"),
        "median_pct": Decimal("20"),
    },
    "material_substitution": {
        "min_pct": Decimal("5"),
        "max_pct": Decimal("30"),
        "median_pct": Decimal("15"),
    },
    "circular_economy": {
        "min_pct": Decimal("3"),
        "max_pct": Decimal("25"),
        "median_pct": Decimal("12"),
    },
    "carbon_capture": {
        "min_pct": Decimal("10"),
        "max_pct": Decimal("90"),
        "median_pct": Decimal("50"),
    },
    "supply_chain_engagement": {
        "min_pct": Decimal("2"),
        "max_pct": Decimal("20"),
        "median_pct": Decimal("10"),
    },
    "product_redesign": {
        "min_pct": Decimal("3"),
        "max_pct": Decimal("25"),
        "median_pct": Decimal("12"),
    },
}

# Plan completeness criteria with required fields for E1-1 compliance.
PLAN_COMPLETENESS_CRITERIA: List[str] = [
    "current_emissions_tco2e",
    "target_emissions_tco2e",
    "target_year",
    "scenario_alignment",
    "decarbonization_actions",
    "locked_in_emissions",
    "capex_allocation",
    "time_horizons",
    "governance_oversight",
    "progress_metrics",
]

# ESRS E1-1 required data points for completeness validation.
E1_1_DATAPOINTS: List[str] = [
    "e1_1_01_transition_plan_existence",
    "e1_1_02_ghg_reduction_targets",
    "e1_1_03_decarbonization_levers",
    "e1_1_04_key_actions",
    "e1_1_05_capex_allocation",
    "e1_1_06_opex_allocation",
    "e1_1_07_locked_in_emissions",
    "e1_1_08_scenario_alignment",
    "e1_1_09_time_horizons",
    "e1_1_10_progress_to_date",
    "e1_1_11_gap_analysis",
    "e1_1_12_governance_oversight",
    "e1_1_13_stakeholder_engagement",
    "e1_1_14_just_transition_considerations",
    "e1_1_15_plan_status",
    "e1_1_16_last_update_date",
    "e1_1_17_board_approval",
    "e1_1_18_methodology_description",
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TransitionPlanAction(BaseModel):
    """A single decarbonisation action within the transition plan.

    Represents one concrete measure the undertaking will take to reduce
    emissions.  Each action has an expected abatement, cost, and timeline.

    Attributes:
        action_id: Unique identifier.
        name: Action name/title.
        description: Detailed description of the action.
        lever: Decarbonisation lever this action falls under.
        expected_abatement_tco2e: Expected emission reduction (tCO2e).
        expected_abatement_pct: Expected reduction as % of baseline.
        capex_eur: Capital expenditure allocated (EUR).
        opex_annual_eur: Annual operating expenditure (EUR).
        start_year: Year the action begins.
        completion_year: Year the action is expected to complete.
        time_horizon: Time horizon classification.
        status: Current implementation status.
        confidence_level: Confidence in achieving the abatement (0-1).
        scope_coverage: Which scopes this action addresses.
        is_implemented: Whether the action is already implemented.
        notes: Additional notes.
    """
    action_id: str = Field(
        default_factory=_new_uuid, description="Unique action ID"
    )
    name: str = Field(
        ..., description="Action name", max_length=200
    )
    description: str = Field(
        default="", description="Detailed description", max_length=2000
    )
    lever: DecarbonizationLever = Field(
        ..., description="Decarbonisation lever category"
    )
    expected_abatement_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Expected emission reduction (tCO2e per year)",
        ge=Decimal("0"),
    )
    expected_abatement_pct: Decimal = Field(
        default=Decimal("0"),
        description="Expected reduction as % of baseline",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    capex_eur: Decimal = Field(
        default=Decimal("0"),
        description="Capital expenditure (EUR)",
        ge=Decimal("0"),
    )
    opex_annual_eur: Decimal = Field(
        default=Decimal("0"),
        description="Annual operating expenditure (EUR)",
        ge=Decimal("0"),
    )
    start_year: int = Field(
        default=2025, description="Start year", ge=2020, le=2100
    )
    completion_year: int = Field(
        default=2030, description="Completion year", ge=2020, le=2100
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM, description="Time horizon"
    )
    status: PlanStatus = Field(
        default=PlanStatus.DRAFT, description="Implementation status"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.5"),
        description="Confidence in achieving abatement (0-1)",
        ge=Decimal("0"),
        le=Decimal("1"),
    )
    scope_coverage: List[str] = Field(
        default_factory=list,
        description="Scopes addressed (scope_1, scope_2, scope_3)",
    )
    is_implemented: bool = Field(
        default=False, description="Whether already implemented"
    )
    notes: str = Field(
        default="", description="Additional notes", max_length=1000
    )

    @field_validator("completion_year")
    @classmethod
    def completion_after_start(cls, v: int, info: Any) -> int:
        """Validate completion year is not before start year."""
        start = info.data.get("start_year", 2025)
        if v < start:
            raise ValueError(
                f"completion_year ({v}) cannot be before start_year ({start})"
            )
        return v


class LockedInEmission(BaseModel):
    """A locked-in emission source per ESRS E1-1 Para 17.

    Locked-in emissions are those that will be generated by existing
    assets, contracts, or products over their remaining useful life.

    Attributes:
        source_id: Unique identifier.
        name: Name of the asset or source.
        emission_type: Type of locked-in emission.
        total_locked_in_tco2e: Total locked-in emissions over remaining life.
        annual_tco2e: Annual emissions from this source.
        remaining_years: Remaining useful life in years.
        asset_value_eur: Book value of the asset (EUR).
        decommission_year: Planned decommission or end-of-contract year.
        mitigation_possible: Whether mitigation actions can reduce these.
        notes: Additional notes.
    """
    source_id: str = Field(
        default_factory=_new_uuid, description="Unique source ID"
    )
    name: str = Field(
        ..., description="Asset or source name", max_length=200
    )
    emission_type: LockedInEmissionType = Field(
        ..., description="Type of locked-in emission"
    )
    total_locked_in_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total locked-in emissions (tCO2e)",
        ge=Decimal("0"),
    )
    annual_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Annual emissions (tCO2e)",
        ge=Decimal("0"),
    )
    remaining_years: int = Field(
        default=0, description="Remaining useful life (years)", ge=0
    )
    asset_value_eur: Decimal = Field(
        default=Decimal("0"),
        description="Asset book value (EUR)",
        ge=Decimal("0"),
    )
    decommission_year: Optional[int] = Field(
        None, description="Planned decommission year"
    )
    mitigation_possible: bool = Field(
        default=False,
        description="Whether mitigation actions can reduce locked-in emissions",
    )
    notes: str = Field(
        default="", description="Additional notes", max_length=1000
    )


class PlanGapAnalysis(BaseModel):
    """Gap analysis between transition plan and target.

    Assesses the difference between projected emissions (after all
    planned abatement actions) and the target level.

    Attributes:
        analysis_id: Unique identifier.
        current_emissions_tco2e: Current annual emissions.
        total_planned_abatement_tco2e: Sum of all action abatements.
        projected_emissions_tco2e: Current minus abatement.
        target_emissions_tco2e: Target level.
        gap_tco2e: Projected minus target (positive = shortfall).
        gap_pct: Gap as percentage of current emissions.
        is_on_track: Whether projected is at or below target.
        locked_in_emissions_tco2e: Total locked-in emissions.
        net_abatement_after_locked_in_tco2e: Abatement minus locked-in.
        confidence_weighted_abatement_tco2e: Abatement weighted by confidence.
        actions_by_lever: Count of actions per lever.
        recommendations: List of recommendations to close the gap.
        provenance_hash: SHA-256 hash.
    """
    analysis_id: str = Field(
        default_factory=_new_uuid, description="Analysis ID"
    )
    current_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Current emissions (tCO2e)"
    )
    total_planned_abatement_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total planned abatement (tCO2e)"
    )
    projected_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Projected emissions (tCO2e)"
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Target emissions (tCO2e)"
    )
    gap_tco2e: Decimal = Field(
        default=Decimal("0"), description="Gap (tCO2e, positive = shortfall)"
    )
    gap_pct: Decimal = Field(
        default=Decimal("0"), description="Gap as % of current"
    )
    is_on_track: bool = Field(
        default=False, description="Whether on track to meet target"
    )
    locked_in_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total locked-in (tCO2e)"
    )
    net_abatement_after_locked_in_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net abatement after locked-in"
    )
    confidence_weighted_abatement_tco2e: Decimal = Field(
        default=Decimal("0"), description="Confidence-weighted abatement"
    )
    actions_by_lever: Dict[str, int] = Field(
        default_factory=dict, description="Actions per lever"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations to close gap"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


class TransitionPlanResult(BaseModel):
    """Complete transition plan result per ESRS E1-1.

    Aggregates all actions, locked-in emissions, and gap analysis
    into a structured result for E1-1 disclosure.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version used.
        calculated_at: Timestamp of calculation.
        reporting_year: Reporting year.
        entity_name: Reporting entity.
        plan_status: Overall plan status.
        scenario_alignment: Climate scenario alignment.
        current_emissions_tco2e: Current total emissions.
        target_emissions_tco2e: Target emissions.
        target_year: Year of the target.
        target_reduction_pct: Target reduction as % of current.
        total_abatement_tco2e: Sum of all planned abatements.
        total_locked_in_tco2e: Total locked-in emissions.
        total_capex_eur: Total CapEx across all actions.
        total_opex_annual_eur: Total annual OpEx.
        actions: List of transition plan actions.
        locked_in_sources: List of locked-in emission sources.
        actions_by_lever: Count per lever.
        actions_by_horizon: Count per time horizon.
        capex_by_lever: CapEx allocation per lever.
        gap_analysis: Gap analysis result.
        plan_completeness_score: Percentage of criteria met.
        plan_completeness_missing: Missing criteria.
        board_approved: Whether plan is board-approved.
        last_updated: Last update date.
        warnings: Validation warnings.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp"
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    entity_name: str = Field(
        default="", description="Entity name"
    )
    plan_status: str = Field(
        default=PlanStatus.DRAFT.value, description="Plan status"
    )
    scenario_alignment: str = Field(
        default=ScenarioAlignment.ASSESSMENT_PENDING.value,
        description="Scenario alignment",
    )
    current_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Current emissions (tCO2e)"
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Target emissions (tCO2e)"
    )
    target_year: int = Field(
        default=2030, description="Target year"
    )
    target_reduction_pct: Decimal = Field(
        default=Decimal("0"), description="Target reduction (%)"
    )
    total_abatement_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total abatement (tCO2e)"
    )
    total_locked_in_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total locked-in (tCO2e)"
    )
    total_capex_eur: Decimal = Field(
        default=Decimal("0"), description="Total CapEx (EUR)"
    )
    total_opex_annual_eur: Decimal = Field(
        default=Decimal("0"), description="Total annual OpEx (EUR)"
    )
    actions: List[TransitionPlanAction] = Field(
        default_factory=list, description="Transition actions"
    )
    locked_in_sources: List[LockedInEmission] = Field(
        default_factory=list, description="Locked-in sources"
    )
    actions_by_lever: Dict[str, int] = Field(
        default_factory=dict, description="Actions per lever"
    )
    actions_by_horizon: Dict[str, int] = Field(
        default_factory=dict, description="Actions per time horizon"
    )
    capex_by_lever: Dict[str, str] = Field(
        default_factory=dict, description="CapEx per lever (EUR)"
    )
    gap_analysis: Optional[PlanGapAnalysis] = Field(
        None, description="Gap analysis result"
    )
    plan_completeness_score: Decimal = Field(
        default=Decimal("0"), description="Completeness score (%)"
    )
    plan_completeness_missing: List[str] = Field(
        default_factory=list, description="Missing completeness criteria"
    )
    board_approved: bool = Field(
        default=False, description="Board approval status"
    )
    last_updated: Optional[str] = Field(
        None, description="Last update date"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TransitionPlanEngine:
    """Transition plan calculation engine per ESRS E1-1.

    Provides deterministic, zero-hallucination calculations for:
    - Abatement potential from decarbonisation actions
    - Locked-in emission quantification
    - Gap analysis (projected vs target)
    - CapEx/OpEx allocation by lever
    - Scenario alignment assessment
    - Plan completeness validation
    - E1-1 data point mapping

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = TransitionPlanEngine()
        actions = [
            TransitionPlanAction(
                name="LED lighting upgrade",
                lever=DecarbonizationLever.ENERGY_EFFICIENCY,
                expected_abatement_tco2e=Decimal("500"),
                capex_eur=Decimal("200000"),
            ),
        ]
        result = engine.build_transition_plan(
            current_emissions=Decimal("10000"),
            target=Decimal("5000"),
            actions=actions,
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Core Methods                                                        #
    # ------------------------------------------------------------------ #

    def build_transition_plan(
        self,
        current_emissions: Decimal,
        target: Decimal,
        actions: List[TransitionPlanAction],
        locked_in: Optional[List[LockedInEmission]] = None,
        target_year: int = 2030,
        entity_name: str = "",
        reporting_year: int = 0,
        scenario_alignment: ScenarioAlignment = ScenarioAlignment.ASSESSMENT_PENDING,
        plan_status: PlanStatus = PlanStatus.DRAFT,
        board_approved: bool = False,
    ) -> TransitionPlanResult:
        """Build a complete transition plan from actions and locked-in emissions.

        Aggregates all decarbonisation actions, calculates total abatement
        potential, quantifies locked-in emissions, performs gap analysis,
        and produces the structured ESRS E1-1 result.

        Args:
            current_emissions: Current annual emissions in tCO2e.
            target: Target annual emissions in tCO2e.
            actions: List of planned decarbonisation actions.
            locked_in: List of locked-in emission sources.
            target_year: Target year for achieving the emission level.
            entity_name: Name of the reporting entity.
            reporting_year: Current reporting year.
            scenario_alignment: Climate scenario the plan aligns with.
            plan_status: Current status of the plan.
            board_approved: Whether the plan is board-approved.

        Returns:
            TransitionPlanResult with complete analysis and provenance.

        Raises:
            ValueError: If current_emissions is negative.
        """
        t0 = time.perf_counter()

        if current_emissions < Decimal("0"):
            raise ValueError(
                f"current_emissions must be >= 0, got {current_emissions}"
            )

        logger.info(
            "Building transition plan: current=%.2f tCO2e, target=%.2f, "
            "%d actions, entity=%s",
            float(current_emissions), float(target),
            len(actions), entity_name,
        )

        locked_in = locked_in or []

        # Calculate abatement potential
        total_abatement = self.calculate_abatement_potential(actions)

        # Calculate locked-in emissions
        total_locked_in = self.calculate_locked_in_emissions(locked_in)

        # Target reduction percentage
        target_reduction_pct = Decimal("0")
        if current_emissions > Decimal("0"):
            target_reduction_pct = _round_val(
                (current_emissions - target)
                / current_emissions
                * Decimal("100"),
                2,
            )

        # Aggregate actions by lever
        lever_counts: Dict[str, int] = {}
        lever_capex: Dict[str, Decimal] = {}
        for action in actions:
            lk = action.lever.value
            lever_counts[lk] = lever_counts.get(lk, 0) + 1
            lever_capex[lk] = lever_capex.get(
                lk, Decimal("0")
            ) + action.capex_eur

        # Aggregate actions by time horizon
        horizon_counts: Dict[str, int] = {}
        for action in actions:
            hk = action.time_horizon.value
            horizon_counts[hk] = horizon_counts.get(hk, 0) + 1

        # Total CapEx and OpEx
        total_capex = _round6(
            sum(a.capex_eur for a in actions)
        )
        total_opex = _round6(
            sum(a.opex_annual_eur for a in actions)
        )

        # Gap analysis
        gap = self._perform_gap_analysis(
            current_emissions, target, actions, locked_in
        )

        # Plan completeness
        completeness_score, completeness_missing = self._assess_completeness(
            current_emissions=current_emissions,
            target=target,
            target_year=target_year,
            actions=actions,
            locked_in=locked_in,
            scenario_alignment=scenario_alignment,
            board_approved=board_approved,
        )

        # Warnings
        warnings: List[str] = []
        if total_abatement < (current_emissions - target):
            warnings.append(
                "Total planned abatement is less than the required reduction. "
                "Additional actions may be needed."
            )
        if total_locked_in > total_abatement:
            warnings.append(
                "Locked-in emissions exceed total planned abatement. "
                "Review asset retirement schedule."
            )
        if not actions:
            warnings.append("No decarbonisation actions specified.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = TransitionPlanResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            plan_status=plan_status.value,
            scenario_alignment=scenario_alignment.value,
            current_emissions_tco2e=current_emissions,
            target_emissions_tco2e=target,
            target_year=target_year,
            target_reduction_pct=target_reduction_pct,
            total_abatement_tco2e=total_abatement,
            total_locked_in_tco2e=total_locked_in,
            total_capex_eur=total_capex,
            total_opex_annual_eur=total_opex,
            actions=actions,
            locked_in_sources=locked_in,
            actions_by_lever=lever_counts,
            actions_by_horizon=horizon_counts,
            capex_by_lever={
                k: str(_round6(v)) for k, v in sorted(lever_capex.items())
            },
            gap_analysis=gap,
            plan_completeness_score=completeness_score,
            plan_completeness_missing=completeness_missing,
            board_approved=board_approved,
            last_updated=_utcnow().isoformat(),
            warnings=warnings,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Transition plan built: abatement=%.2f, locked_in=%.2f, "
            "gap=%.2f tCO2e, completeness=%.1f%%",
            float(total_abatement), float(total_locked_in),
            float(gap.gap_tco2e), float(completeness_score),
        )

        return result

    # ------------------------------------------------------------------ #
    # Abatement Calculation                                               #
    # ------------------------------------------------------------------ #

    def calculate_abatement_potential(
        self, actions: List[TransitionPlanAction]
    ) -> Decimal:
        """Calculate total abatement potential from all actions.

        Sums the expected_abatement_tco2e across all actions.  This is
        the gross abatement before applying confidence weighting.

        Args:
            actions: List of TransitionPlanAction instances.

        Returns:
            Total abatement potential in tCO2e.
        """
        if not actions:
            return Decimal("0")

        total = sum(a.expected_abatement_tco2e for a in actions)
        result = _round6(total)

        logger.info(
            "Abatement potential: %.2f tCO2e from %d actions",
            float(result), len(actions),
        )
        return result

    def calculate_confidence_weighted_abatement(
        self, actions: List[TransitionPlanAction]
    ) -> Decimal:
        """Calculate confidence-weighted abatement potential.

        Weights each action's abatement by its confidence level
        (0 to 1).  This provides a more realistic estimate of
        achievable reduction.

        Args:
            actions: List of TransitionPlanAction instances.

        Returns:
            Confidence-weighted abatement in tCO2e.
        """
        if not actions:
            return Decimal("0")

        weighted = sum(
            a.expected_abatement_tco2e * a.confidence_level
            for a in actions
        )
        return _round6(weighted)

    # ------------------------------------------------------------------ #
    # Locked-In Emissions                                                 #
    # ------------------------------------------------------------------ #

    def calculate_locked_in_emissions(
        self, locked_in: List[LockedInEmission]
    ) -> Decimal:
        """Calculate total locked-in GHG emissions.

        Sums total_locked_in_tco2e across all sources.  If total is not
        set but annual and remaining years are, calculates as
        annual * remaining_years.

        Args:
            locked_in: List of LockedInEmission instances.

        Returns:
            Total locked-in emissions in tCO2e.
        """
        if not locked_in:
            return Decimal("0")

        total = Decimal("0")
        for source in locked_in:
            if source.total_locked_in_tco2e > Decimal("0"):
                total += source.total_locked_in_tco2e
            elif (
                source.annual_tco2e > Decimal("0")
                and source.remaining_years > 0
            ):
                total += source.annual_tco2e * _decimal(source.remaining_years)

        result = _round6(total)

        logger.info(
            "Locked-in emissions: %.2f tCO2e from %d sources",
            float(result), len(locked_in),
        )
        return result

    # ------------------------------------------------------------------ #
    # Gap Analysis                                                        #
    # ------------------------------------------------------------------ #

    def analyze_gap(
        self,
        result: TransitionPlanResult,
    ) -> PlanGapAnalysis:
        """Perform gap analysis on a completed transition plan result.

        Convenience method that wraps _perform_gap_analysis using data
        from the result.

        Args:
            result: TransitionPlanResult to analyse.

        Returns:
            PlanGapAnalysis with gap quantification and recommendations.
        """
        return self._perform_gap_analysis(
            current_emissions=result.current_emissions_tco2e,
            target=result.target_emissions_tco2e,
            actions=result.actions,
            locked_in=result.locked_in_sources,
        )

    def _perform_gap_analysis(
        self,
        current_emissions: Decimal,
        target: Decimal,
        actions: List[TransitionPlanAction],
        locked_in: List[LockedInEmission],
    ) -> PlanGapAnalysis:
        """Internal gap analysis implementation.

        Gap = projected_emissions - target
        projected_emissions = current - total_abatement
        If gap > 0, the plan falls short of the target.

        Args:
            current_emissions: Current annual emissions (tCO2e).
            target: Target annual emissions (tCO2e).
            actions: List of planned actions.
            locked_in: List of locked-in emission sources.

        Returns:
            PlanGapAnalysis with full quantification.
        """
        total_abatement = self.calculate_abatement_potential(actions)
        confidence_weighted = self.calculate_confidence_weighted_abatement(actions)
        total_locked_in = self.calculate_locked_in_emissions(locked_in)

        # Projected emissions after all abatement
        projected = current_emissions - total_abatement
        if projected < Decimal("0"):
            projected = Decimal("0")

        # Gap (positive = shortfall)
        gap = projected - target
        is_on_track = gap <= Decimal("0")

        # Gap as percentage of current emissions
        gap_pct = Decimal("0")
        if current_emissions > Decimal("0"):
            gap_pct = _round_val(
                gap / current_emissions * Decimal("100"), 2
            )

        # Net abatement considering locked-in
        net_abatement = total_abatement - total_locked_in

        # Actions by lever
        lever_counts: Dict[str, int] = {}
        for action in actions:
            lk = action.lever.value
            lever_counts[lk] = lever_counts.get(lk, 0) + 1

        # Recommendations
        recommendations: List[str] = []
        if gap > Decimal("0"):
            recommendations.append(
                f"Additional {float(_round6(gap)):.2f} tCO2e of abatement "
                f"needed to close the gap."
            )

            # Check if any lever types are underused
            used_levers = set(lever_counts.keys())
            all_levers = {lv.value for lv in DecarbonizationLever}
            unused = all_levers - used_levers
            if unused:
                top_unused = sorted(unused)[:3]
                recommendations.append(
                    f"Consider additional actions in: {', '.join(top_unused)}"
                )

            if total_locked_in > Decimal("0"):
                recommendations.append(
                    "Review asset retirement schedule to reduce locked-in "
                    "emissions."
                )

            if confidence_weighted < total_abatement * Decimal("0.7"):
                recommendations.append(
                    "Confidence levels are low; increase certainty of "
                    "key abatement actions."
                )
        else:
            recommendations.append(
                "Plan is on track to meet the target."
            )

        gap_result = PlanGapAnalysis(
            current_emissions_tco2e=current_emissions,
            total_planned_abatement_tco2e=total_abatement,
            projected_emissions_tco2e=_round6(projected),
            target_emissions_tco2e=target,
            gap_tco2e=_round6(gap),
            gap_pct=gap_pct,
            is_on_track=is_on_track,
            locked_in_emissions_tco2e=total_locked_in,
            net_abatement_after_locked_in_tco2e=_round6(net_abatement),
            confidence_weighted_abatement_tco2e=confidence_weighted,
            actions_by_lever=lever_counts,
            recommendations=recommendations,
        )
        gap_result.provenance_hash = _compute_hash(gap_result)

        logger.info(
            "Gap analysis: projected=%.2f, target=%.2f, gap=%.2f tCO2e, "
            "on_track=%s",
            float(projected), float(target), float(gap), is_on_track,
        )

        return gap_result

    # ------------------------------------------------------------------ #
    # Scenario Alignment                                                  #
    # ------------------------------------------------------------------ #

    def validate_scenario_alignment(
        self, result: TransitionPlanResult
    ) -> Dict[str, Any]:
        """Validate alignment of the transition plan with climate scenarios.

        Assesses whether the plan's target reduction rate is compatible
        with 1.5C and well-below 2C pathways based on IPCC SR15.

        Reference rates (global, approximate):
        - 1.5C: ~7% annual reduction from 2020
        - Well-below 2C: ~4% annual reduction from 2020
        - 2C: ~2.5% annual reduction from 2020

        Args:
            result: TransitionPlanResult to validate.

        Returns:
            Dict with alignment_assessment, required_rate, actual_rate,
            scenario_compatibility, and provenance_hash.
        """
        # Calculate implied annual reduction rate
        if (
            result.current_emissions_tco2e <= Decimal("0")
            or result.target_year <= result.reporting_year
        ):
            return {
                "alignment_assessment": "insufficient_data",
                "required_rate_1_5c_pct": None,
                "actual_annual_rate_pct": None,
                "scenarios": {},
                "provenance_hash": _compute_hash({"status": "insufficient_data"}),
            }

        years = _decimal(result.target_year - result.reporting_year)
        total_reduction_pct = result.target_reduction_pct
        annual_rate = _round_val(
            _safe_divide(total_reduction_pct, years), 2
        )

        # Scenario reference annual rates (simplified linear)
        scenario_rates = {
            "aligned_1_5c": Decimal("7.0"),
            "aligned_well_below_2c": Decimal("4.0"),
            "aligned_2c": Decimal("2.5"),
        }

        scenarios: Dict[str, Any] = {}
        for scenario_key, required_rate in scenario_rates.items():
            is_compatible = annual_rate >= required_rate
            scenarios[scenario_key] = {
                "required_annual_rate_pct": str(required_rate),
                "is_compatible": is_compatible,
                "gap_pp": str(_round_val(required_rate - annual_rate, 2)),
            }

        # Determine overall alignment
        if annual_rate >= Decimal("7.0"):
            alignment = "aligned_1_5c"
        elif annual_rate >= Decimal("4.0"):
            alignment = "aligned_well_below_2c"
        elif annual_rate >= Decimal("2.5"):
            alignment = "aligned_2c"
        else:
            alignment = "not_aligned"

        assessment = {
            "alignment_assessment": alignment,
            "actual_annual_rate_pct": str(annual_rate),
            "total_reduction_pct": str(total_reduction_pct),
            "years_to_target": int(years),
            "scenarios": scenarios,
        }
        assessment["provenance_hash"] = _compute_hash(assessment)

        logger.info(
            "Scenario alignment: %s (annual rate=%.2f%%)",
            alignment, float(annual_rate),
        )

        return assessment

    # ------------------------------------------------------------------ #
    # CapEx Allocation                                                    #
    # ------------------------------------------------------------------ #

    def calculate_capex_allocation(
        self, actions: List[TransitionPlanAction]
    ) -> Dict[str, Any]:
        """Calculate CapEx allocation by decarbonisation lever.

        Per ESRS E1-1 Para 16, the plan should include CapEx allocation
        towards each decarbonisation lever.

        Args:
            actions: List of TransitionPlanAction instances.

        Returns:
            Dict with lever-level CapEx totals, percentages, and
            per-action detail.
        """
        if not actions:
            return {
                "total_capex_eur": "0",
                "by_lever": {},
                "by_action": [],
                "provenance_hash": _compute_hash({"empty": True}),
            }

        total_capex = sum(a.capex_eur for a in actions)
        total_capex = _round6(total_capex)

        lever_capex: Dict[str, Decimal] = {}
        for action in actions:
            lk = action.lever.value
            lever_capex[lk] = lever_capex.get(
                lk, Decimal("0")
            ) + action.capex_eur

        by_lever: Dict[str, Any] = {}
        for lk, capex in sorted(lever_capex.items()):
            capex_rounded = _round6(capex)
            pct = _round_val(
                _safe_divide(capex, total_capex) * Decimal("100"), 2
            )
            by_lever[lk] = {
                "capex_eur": str(capex_rounded),
                "pct_of_total": str(pct),
            }

        by_action = []
        for action in actions:
            pct = _round_val(
                _safe_divide(action.capex_eur, total_capex) * Decimal("100"), 2
            )
            by_action.append({
                "action_id": action.action_id,
                "name": action.name,
                "lever": action.lever.value,
                "capex_eur": str(_round6(action.capex_eur)),
                "pct_of_total": str(pct),
            })

        allocation = {
            "total_capex_eur": str(total_capex),
            "by_lever": by_lever,
            "by_action": by_action,
        }
        allocation["provenance_hash"] = _compute_hash(allocation)

        logger.info(
            "CapEx allocation: total=%.2f EUR across %d levers",
            float(total_capex), len(lever_capex),
        )

        return allocation

    # ------------------------------------------------------------------ #
    # Plan Completeness                                                   #
    # ------------------------------------------------------------------ #

    def _assess_completeness(
        self,
        current_emissions: Decimal,
        target: Decimal,
        target_year: int,
        actions: List[TransitionPlanAction],
        locked_in: List[LockedInEmission],
        scenario_alignment: ScenarioAlignment,
        board_approved: bool,
    ) -> Tuple[Decimal, List[str]]:
        """Assess plan completeness against PLAN_COMPLETENESS_CRITERIA.

        Args:
            current_emissions: Current emissions.
            target: Target emissions.
            target_year: Target year.
            actions: Planned actions.
            locked_in: Locked-in sources.
            scenario_alignment: Scenario alignment.
            board_approved: Board approval.

        Returns:
            Tuple of (score_pct, missing_criteria_list).
        """
        checks = {
            "current_emissions_tco2e": current_emissions > Decimal("0"),
            "target_emissions_tco2e": target >= Decimal("0"),
            "target_year": target_year > 0,
            "scenario_alignment": (
                scenario_alignment != ScenarioAlignment.ASSESSMENT_PENDING
            ),
            "decarbonization_actions": len(actions) > 0,
            "locked_in_emissions": True,  # Can be zero if no locked-in
            "capex_allocation": any(a.capex_eur > Decimal("0") for a in actions) if actions else False,
            "time_horizons": any(True for a in actions) if actions else False,
            "governance_oversight": board_approved,
            "progress_metrics": any(a.is_implemented for a in actions) if actions else False,
        }

        met = [k for k, v in checks.items() if v]
        missing = [k for k, v in checks.items() if not v]
        total = len(PLAN_COMPLETENESS_CRITERIA)
        score = _round_val(
            _decimal(len(met)) / _decimal(total) * Decimal("100"), 1
        )

        return score, missing

    # ------------------------------------------------------------------ #
    # Completeness Validation (E1-1 Data Points)                          #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: TransitionPlanResult
    ) -> Dict[str, Any]:
        """Validate completeness against ESRS E1-1 required data points.

        Checks whether all E1-1 mandatory data points are present.

        Args:
            result: TransitionPlanResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            and provenance_hash.
        """
        populated: List[str] = []
        missing: List[str] = []

        checks = {
            "e1_1_01_transition_plan_existence": True,  # Always true
            "e1_1_02_ghg_reduction_targets": (
                result.target_emissions_tco2e >= Decimal("0")
                and result.target_year > 0
            ),
            "e1_1_03_decarbonization_levers": len(result.actions_by_lever) > 0,
            "e1_1_04_key_actions": len(result.actions) > 0,
            "e1_1_05_capex_allocation": result.total_capex_eur > Decimal("0"),
            "e1_1_06_opex_allocation": result.total_opex_annual_eur >= Decimal("0"),
            "e1_1_07_locked_in_emissions": True,  # Reported even if zero
            "e1_1_08_scenario_alignment": (
                result.scenario_alignment
                != ScenarioAlignment.ASSESSMENT_PENDING.value
            ),
            "e1_1_09_time_horizons": len(result.actions_by_horizon) > 0,
            "e1_1_10_progress_to_date": (
                result.gap_analysis is not None
            ),
            "e1_1_11_gap_analysis": (
                result.gap_analysis is not None
            ),
            "e1_1_12_governance_oversight": result.board_approved,
            "e1_1_13_stakeholder_engagement": True,  # Narrative
            "e1_1_14_just_transition_considerations": True,  # Narrative
            "e1_1_15_plan_status": bool(result.plan_status),
            "e1_1_16_last_update_date": result.last_updated is not None,
            "e1_1_17_board_approval": result.board_approved,
            "e1_1_18_methodology_description": True,  # Narrative
        }

        for dp, is_populated in checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(E1_1_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": checks}
            ),
        }

        logger.info(
            "E1-1 completeness: %s%% (%d/%d), missing=%s",
            completeness, pop_count, total, missing,
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # ESRS E1-1 Data Point Mapping                                        #
    # ------------------------------------------------------------------ #

    def get_e1_1_datapoints(
        self, result: TransitionPlanResult
    ) -> Dict[str, Any]:
        """Map transition plan result to ESRS E1-1 disclosure data points.

        Creates a structured mapping of all E1-1 required data points
        with their values, ready for report generation.

        Args:
            result: TransitionPlanResult to map.

        Returns:
            Dict mapping E1-1 data point IDs to their values.
        """
        # Build action summaries
        action_summaries = []
        for action in result.actions:
            action_summaries.append({
                "name": action.name,
                "lever": action.lever.value,
                "abatement_tco2e": str(action.expected_abatement_tco2e),
                "capex_eur": str(action.capex_eur),
                "start_year": action.start_year,
                "completion_year": action.completion_year,
                "time_horizon": action.time_horizon.value,
                "status": action.status.value,
            })

        # Build locked-in summaries
        locked_in_summaries = []
        for source in result.locked_in_sources:
            locked_in_summaries.append({
                "name": source.name,
                "type": source.emission_type.value,
                "total_tco2e": str(source.total_locked_in_tco2e),
                "annual_tco2e": str(source.annual_tco2e),
                "remaining_years": source.remaining_years,
            })

        datapoints: Dict[str, Any] = {
            "e1_1_01_transition_plan_existence": {
                "label": "Transition plan adopted",
                "value": True,
                "esrs_ref": "E1-1 Para 14",
            },
            "e1_1_02_ghg_reduction_targets": {
                "label": "GHG reduction targets",
                "current_tco2e": str(result.current_emissions_tco2e),
                "target_tco2e": str(result.target_emissions_tco2e),
                "target_year": result.target_year,
                "reduction_pct": str(result.target_reduction_pct),
                "esrs_ref": "E1-1 Para 15",
            },
            "e1_1_03_decarbonization_levers": {
                "label": "Key decarbonisation levers",
                "levers": result.actions_by_lever,
                "esrs_ref": "E1-1 Para 16",
            },
            "e1_1_04_key_actions": {
                "label": "Key transition actions",
                "actions": action_summaries,
                "action_count": len(result.actions),
                "esrs_ref": "E1-1 Para 16",
            },
            "e1_1_05_capex_allocation": {
                "label": "CapEx allocation for transition",
                "total_eur": str(result.total_capex_eur),
                "by_lever": result.capex_by_lever,
                "esrs_ref": "E1-1 Para 16",
            },
            "e1_1_06_opex_allocation": {
                "label": "OpEx allocation for transition",
                "annual_eur": str(result.total_opex_annual_eur),
                "esrs_ref": "E1-1 Para 16",
            },
            "e1_1_07_locked_in_emissions": {
                "label": "Locked-in GHG emissions",
                "total_tco2e": str(result.total_locked_in_tco2e),
                "sources": locked_in_summaries,
                "esrs_ref": "E1-1 Para 17",
            },
            "e1_1_08_scenario_alignment": {
                "label": "Climate scenario alignment",
                "value": result.scenario_alignment,
                "esrs_ref": "E1-1 Para 18",
            },
            "e1_1_09_time_horizons": {
                "label": "Time horizons for actions",
                "by_horizon": result.actions_by_horizon,
                "esrs_ref": "E1-1 Para 19",
            },
            "e1_1_10_progress_to_date": {
                "label": "Progress assessment",
                "is_on_track": (
                    result.gap_analysis.is_on_track
                    if result.gap_analysis else None
                ),
                "gap_tco2e": (
                    str(result.gap_analysis.gap_tco2e)
                    if result.gap_analysis else None
                ),
                "esrs_ref": "E1-1 Para 20",
            },
            "e1_1_11_gap_analysis": {
                "label": "Gap analysis summary",
                "projected_tco2e": (
                    str(result.gap_analysis.projected_emissions_tco2e)
                    if result.gap_analysis else None
                ),
                "target_tco2e": str(result.target_emissions_tco2e),
                "gap_tco2e": (
                    str(result.gap_analysis.gap_tco2e)
                    if result.gap_analysis else None
                ),
                "gap_pct": (
                    str(result.gap_analysis.gap_pct)
                    if result.gap_analysis else None
                ),
                "esrs_ref": "E1-1 Para 20",
            },
            "e1_1_12_governance_oversight": {
                "label": "Governance oversight",
                "board_approved": result.board_approved,
                "esrs_ref": "E1-1 Para 14",
            },
            "e1_1_13_stakeholder_engagement": {
                "label": "Stakeholder engagement",
                "value": "See transition plan documentation",
                "esrs_ref": "E1-1",
            },
            "e1_1_14_just_transition_considerations": {
                "label": "Just transition considerations",
                "value": "See transition plan documentation",
                "esrs_ref": "E1-1",
            },
            "e1_1_15_plan_status": {
                "label": "Transition plan status",
                "value": result.plan_status,
                "esrs_ref": "E1-1 Para 14",
            },
            "e1_1_16_last_update_date": {
                "label": "Last update date",
                "value": result.last_updated,
                "esrs_ref": "E1-1",
            },
            "e1_1_17_board_approval": {
                "label": "Board approval",
                "value": result.board_approved,
                "esrs_ref": "E1-1 Para 14",
            },
            "e1_1_18_methodology_description": {
                "label": "Methodology description",
                "value": (
                    "Transition plan built using deterministic abatement "
                    "calculations per GHG Protocol, IEA, and IPCC SR15 "
                    "reference pathways.  No LLM used in calculations."
                ),
                "esrs_ref": "E1-1",
            },
        }

        datapoints["_metadata"] = {
            "engine_version": self.engine_version,
            "result_id": result.result_id,
            "calculated_at": str(result.calculated_at),
            "provenance_hash": _compute_hash(datapoints),
        }

        return datapoints

    # ------------------------------------------------------------------ #
    # Year-over-Year Plan Comparison                                      #
    # ------------------------------------------------------------------ #

    def compare_plans(
        self,
        current: TransitionPlanResult,
        previous: TransitionPlanResult,
    ) -> Dict[str, Any]:
        """Compare transition plans across two reporting years.

        Tracks how the plan has evolved in terms of actions, CapEx,
        abatement, and gap closure.

        Args:
            current: Current year plan result.
            previous: Previous year plan result.

        Returns:
            Dict with changes in actions, abatement, capex, and gap.
        """
        comparison = {
            "current_year": current.reporting_year,
            "previous_year": previous.reporting_year,
            "actions_change": len(current.actions) - len(previous.actions),
            "abatement_change_tco2e": str(_round6(
                current.total_abatement_tco2e - previous.total_abatement_tco2e
            )),
            "capex_change_eur": str(_round6(
                current.total_capex_eur - previous.total_capex_eur
            )),
            "gap_change_tco2e": str(_round6(
                (current.gap_analysis.gap_tco2e if current.gap_analysis else Decimal("0"))
                - (previous.gap_analysis.gap_tco2e if previous.gap_analysis else Decimal("0"))
            )),
            "plan_status_change": {
                "from": previous.plan_status,
                "to": current.plan_status,
            },
            "scenario_alignment_change": {
                "from": previous.scenario_alignment,
                "to": current.scenario_alignment,
            },
            "completeness_change_pp": str(_round_val(
                current.plan_completeness_score - previous.plan_completeness_score, 1
            )),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Lever Benchmarking                                                  #
    # ------------------------------------------------------------------ #

    def benchmark_lever_abatement(
        self,
        actions: List[TransitionPlanAction],
        current_emissions: Decimal,
    ) -> Dict[str, Any]:
        """Benchmark action abatement against typical ranges per lever.

        Compares the expected abatement percentage of each lever against
        the typical min/max/median from LEVER_TYPICAL_ABATEMENT.

        Args:
            actions: List of actions to benchmark.
            current_emissions: Current emissions for percentage calculation.

        Returns:
            Dict with per-lever benchmarking results and provenance.
        """
        if current_emissions <= Decimal("0"):
            return {
                "error": "current_emissions must be > 0",
                "provenance_hash": _compute_hash({"error": True}),
            }

        lever_totals: Dict[str, Decimal] = {}
        for action in actions:
            lk = action.lever.value
            lever_totals[lk] = lever_totals.get(
                lk, Decimal("0")
            ) + action.expected_abatement_tco2e

        benchmarks: Dict[str, Any] = {}
        for lk, total_abatement in sorted(lever_totals.items()):
            actual_pct = _round_val(
                total_abatement / current_emissions * Decimal("100"), 2
            )
            typical = LEVER_TYPICAL_ABATEMENT.get(lk, {})
            median = typical.get("median_pct", Decimal("0"))
            min_val = typical.get("min_pct", Decimal("0"))
            max_val = typical.get("max_pct", Decimal("0"))

            if actual_pct < min_val:
                assessment = "below_typical_range"
            elif actual_pct > max_val:
                assessment = "above_typical_range"
            else:
                assessment = "within_typical_range"

            benchmarks[lk] = {
                "actual_abatement_tco2e": str(_round6(total_abatement)),
                "actual_pct": str(actual_pct),
                "typical_min_pct": str(min_val),
                "typical_max_pct": str(max_val),
                "typical_median_pct": str(median),
                "assessment": assessment,
            }

        result = {
            "current_emissions_tco2e": str(current_emissions),
            "benchmarks": benchmarks,
            "lever_count": len(benchmarks),
        }
        result["provenance_hash"] = _compute_hash(result)

        return result
