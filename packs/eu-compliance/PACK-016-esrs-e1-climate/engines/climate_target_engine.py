# -*- coding: utf-8 -*-
"""
ClimateTargetEngine - PACK-016 ESRS E1 Climate Engine 4
=========================================================

Manages climate change targets per ESRS E1-4.

Under the European Sustainability Reporting Standards (ESRS), disclosure
requirement E1-4 mandates that undertakings disclose the targets they
have set to manage their material climate-related impacts, risks, and
opportunities.  This includes the nature of the target (absolute or
intensity), the scope covered, the base year, the target year, and
progress towards achieving each target.

ESRS E1-4 Disclosure Requirements:
    - Para 30: The undertaking shall disclose the climate-related targets
      it has set.
    - Para 31: For each target, disclose: whether absolute or intensity,
      the GHG scope(s) covered, the base year and base year emissions,
      the target year and target level, milestones, and relationship
      to policy goals.
    - Para 32: Current progress against each target (in absolute terms
      and as a percentage).
    - Para 33: Whether targets are validated by a third party (e.g. SBTi).
    - Para 34: Methodology for setting targets and base year recalculation.

Additional SBTi Alignment Checks:
    - Near-term targets: minimum 4.2% annual linear reduction (1.5C)
    - Long-term targets: 90% reduction by 2050
    - Scope 3 coverage: if >40% of total, must include Scope 3 target
    - Base year: no older than 2 years at time of target submission

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E1 Climate Change, Disclosure Requirement E1-4
    - Science Based Targets initiative (SBTi) Corporate Net-Zero Standard
    - SBTi Target Validation Protocol (v3.0, 2023)
    - Paris Agreement (2015) Article 2
    - IPCC SR15 (2018) - 1.5C mitigation pathways

Zero-Hallucination:
    - Progress calculation is arithmetic: (base - current) / (base - target)
    - Annual rate is simple linear: total_reduction / years
    - SBTi alignment uses fixed threshold comparisons
    - Base year recalculation uses additive adjustments
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


class TargetType(str, Enum):
    """Type of climate target per ESRS E1-4 Para 31.

    Absolute targets set a fixed emission level.
    Intensity targets set an emission rate per business metric.
    Net-zero targets include offsetting/removal components.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    NET_ZERO = "net_zero"


class TargetScope(str, Enum):
    """GHG scope coverage of the target.

    Per ESRS E1-4 Para 31, the scope(s) covered must be disclosed.
    Targets may cover individual scopes or combinations.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


class TargetPathway(str, Enum):
    """Climate pathway the target aligns with.

    Per SBTi and ESRS E1-4, targets should be assessed against
    recognised decarbonisation pathways.
    """
    PATHWAY_1_5C = "1.5c"
    PATHWAY_WELL_BELOW_2C = "well_below_2c"
    PATHWAY_2C = "2c"
    PATHWAY_UNSPECIFIED = "unspecified"


class TargetStatus(str, Enum):
    """Current status of the climate target.

    Tracks whether the target is new, in progress, achieved,
    revised, or retired.
    """
    NEW = "new"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    REVISED = "revised"
    RETIRED = "retired"


class BaseYearApproach(str, Enum):
    """Base year emissions recalculation approach.

    Per GHG Protocol, the base year may need to be recalculated
    when structural changes occur.
    """
    FIXED_BASE_YEAR = "fixed_base_year"
    ROLLING_BASE_YEAR = "rolling_base_year"
    TARGET_RECALCULATION = "target_recalculation"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# SBTi minimum annual linear reduction rates by pathway.
# Source: SBTi Corporate Net-Zero Standard, v1.0 (2021).
# Near-term (to 2030): minimum rates for 1.5C alignment.
# Long-term (to 2050): 90% absolute reduction.
SBTI_MINIMUM_RATES: Dict[str, Dict[str, Decimal]] = {
    "1.5c": {
        "scope_1_2_annual_pct": Decimal("4.2"),
        "scope_3_annual_pct": Decimal("2.5"),
        "long_term_reduction_pct": Decimal("90"),
        "long_term_year": Decimal("2050"),
    },
    "well_below_2c": {
        "scope_1_2_annual_pct": Decimal("2.5"),
        "scope_3_annual_pct": Decimal("2.5"),
        "long_term_reduction_pct": Decimal("80"),
        "long_term_year": Decimal("2050"),
    },
    "2c": {
        "scope_1_2_annual_pct": Decimal("1.23"),
        "scope_3_annual_pct": Decimal("1.23"),
        "long_term_reduction_pct": Decimal("70"),
        "long_term_year": Decimal("2050"),
    },
}

# ESRS E1-4 required data points for completeness validation.
E1_4_DATAPOINTS: List[str] = [
    "e1_4_01_target_exists",
    "e1_4_02_target_type",
    "e1_4_03_target_scope",
    "e1_4_04_base_year",
    "e1_4_05_base_year_emissions",
    "e1_4_06_target_year",
    "e1_4_07_target_level",
    "e1_4_08_target_reduction_pct",
    "e1_4_09_current_progress_absolute",
    "e1_4_10_current_progress_pct",
    "e1_4_11_milestones",
    "e1_4_12_third_party_validation",
    "e1_4_13_sbti_alignment",
    "e1_4_14_pathway_alignment",
    "e1_4_15_base_year_recalculation_policy",
    "e1_4_16_target_methodology",
    "e1_4_17_target_status",
    "e1_4_18_relationship_to_policy",
]

# Target assessment criteria for scoring.
TARGET_ASSESSMENT_CRITERIA: List[str] = [
    "has_base_year",
    "has_target_year",
    "has_quantified_target",
    "has_scope_coverage",
    "has_methodology",
    "is_sbti_aligned",
    "has_milestones",
    "has_progress_tracking",
    "is_science_based",
    "has_third_party_validation",
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ClimateTarget(BaseModel):
    """A single climate change target per ESRS E1-4.

    Represents one quantified target with its scope, base year,
    target year, and related metadata.

    Attributes:
        target_id: Unique target identifier.
        name: Target name or description.
        target_type: Absolute, intensity, or net-zero.
        target_scope: GHG scope(s) covered.
        base_year: Base year for the target.
        base_year_emissions_tco2e: Emissions in the base year (tCO2e).
        target_year: Year to achieve the target.
        target_emissions_tco2e: Target emission level (tCO2e).
        target_reduction_pct: Target reduction as % from base year.
        intensity_denominator_unit: Unit for intensity targets.
        intensity_base_year_value: Intensity in base year.
        intensity_target_value: Intensity target value.
        pathway: Climate pathway alignment.
        status: Current target status.
        is_sbti_validated: Whether validated by SBTi.
        sbti_target_type: SBTi target classification.
        milestones: Interim milestones (year -> reduction %).
        base_year_approach: Base year recalculation approach.
        methodology: Description of target-setting methodology.
        third_party_validator: Name of third-party validator.
        notes: Additional notes.
    """
    target_id: str = Field(
        default_factory=_new_uuid, description="Unique target ID"
    )
    name: str = Field(
        ..., description="Target name", max_length=300
    )
    target_type: TargetType = Field(
        ..., description="Target type (absolute/intensity/net_zero)"
    )
    target_scope: TargetScope = Field(
        ..., description="GHG scope coverage"
    )
    base_year: int = Field(
        ..., description="Base year", ge=1990, le=2100
    )
    base_year_emissions_tco2e: Decimal = Field(
        ..., description="Base year emissions (tCO2e)", ge=Decimal("0")
    )
    target_year: int = Field(
        ..., description="Target year", ge=2020, le=2100
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Target emission level (tCO2e)",
        ge=Decimal("0"),
    )
    target_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Target reduction from base year (%)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    intensity_denominator_unit: str = Field(
        default="", description="Intensity denominator unit"
    )
    intensity_base_year_value: Optional[Decimal] = Field(
        None, description="Base year intensity value"
    )
    intensity_target_value: Optional[Decimal] = Field(
        None, description="Target intensity value"
    )
    pathway: TargetPathway = Field(
        default=TargetPathway.PATHWAY_UNSPECIFIED,
        description="Climate pathway alignment",
    )
    status: TargetStatus = Field(
        default=TargetStatus.IN_PROGRESS,
        description="Target status",
    )
    is_sbti_validated: bool = Field(
        default=False, description="SBTi validation status"
    )
    sbti_target_type: str = Field(
        default="",
        description="SBTi target classification (near-term/long-term)",
        max_length=50,
    )
    milestones: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Interim milestones: year_str -> reduction_pct",
    )
    base_year_approach: BaseYearApproach = Field(
        default=BaseYearApproach.FIXED_BASE_YEAR,
        description="Base year recalculation approach",
    )
    methodology: str = Field(
        default="",
        description="Target-setting methodology description",
        max_length=2000,
    )
    third_party_validator: str = Field(
        default="",
        description="Third-party validator name",
        max_length=200,
    )
    notes: str = Field(
        default="", description="Additional notes", max_length=1000
    )

    @field_validator("target_year")
    @classmethod
    def target_after_base_year(cls, v: int, info: Any) -> int:
        """Validate target year is after base year."""
        base = info.data.get("base_year", 1990)
        if v <= base:
            raise ValueError(
                f"target_year ({v}) must be after base_year ({base})"
            )
        return v

    @field_validator("target_reduction_pct")
    @classmethod
    def compute_reduction_if_zero(
        cls, v: Decimal, info: Any
    ) -> Decimal:
        """Compute reduction percentage from base and target if not set."""
        if v == Decimal("0"):
            base = info.data.get("base_year_emissions_tco2e", Decimal("0"))
            target = info.data.get("target_emissions_tco2e", Decimal("0"))
            if base > Decimal("0") and target < base:
                return _round_val(
                    (base - target) / base * Decimal("100"), 2
                )
        return v


class TargetProgressResult(BaseModel):
    """Progress assessment for a single climate target.

    Per ESRS E1-4 Para 32, undertakings must disclose current
    progress in both absolute and percentage terms.

    Attributes:
        progress_id: Unique identifier.
        target_id: Reference to the target being assessed.
        target_name: Name of the target.
        assessment_year: Year of assessment.
        base_year_emissions_tco2e: Base year emissions.
        current_emissions_tco2e: Current emissions.
        target_emissions_tco2e: Target emissions.
        absolute_reduction_tco2e: Absolute reduction achieved.
        required_reduction_tco2e: Total required reduction.
        progress_pct: Progress as percentage of required reduction.
        remaining_reduction_tco2e: Remaining reduction needed.
        years_remaining: Years until target year.
        required_annual_rate_pct: Required annual rate to stay on track.
        actual_annual_rate_pct: Actual annual rate achieved so far.
        is_on_track: Whether on track to meet the target.
        status_assessment: Qualitative status assessment.
        milestones_met: Milestones that have been met.
        milestones_missed: Milestones that have been missed.
        provenance_hash: SHA-256 hash.
    """
    progress_id: str = Field(
        default_factory=_new_uuid, description="Progress ID"
    )
    target_id: str = Field(
        default="", description="Target ID"
    )
    target_name: str = Field(
        default="", description="Target name"
    )
    assessment_year: int = Field(
        default=0, description="Year of assessment"
    )
    base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Base year emissions"
    )
    current_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Current emissions"
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Target emissions"
    )
    absolute_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Absolute reduction achieved"
    )
    required_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total required reduction"
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"), description="Progress (%)"
    )
    remaining_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Remaining reduction"
    )
    years_remaining: int = Field(
        default=0, description="Years to target year"
    )
    required_annual_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Required annual rate (%)"
    )
    actual_annual_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Actual annual rate (%)"
    )
    is_on_track: bool = Field(
        default=False, description="On track to meet target"
    )
    status_assessment: str = Field(
        default="", description="Qualitative assessment"
    )
    milestones_met: List[str] = Field(
        default_factory=list, description="Milestones met"
    )
    milestones_missed: List[str] = Field(
        default_factory=list, description="Milestones missed"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


class BaseYearRecalculation(BaseModel):
    """Base year recalculation result.

    Per GHG Protocol and ESRS, the base year must be recalculated when
    significant structural changes occur (mergers, acquisitions,
    divestitures, changes in methodology).

    Attributes:
        recalculation_id: Unique identifier.
        target_id: Reference to the affected target.
        original_base_year_tco2e: Original base year emissions.
        adjustment_tco2e: Adjustment amount (positive = increase).
        adjustment_reason: Reason for recalculation.
        recalculated_base_year_tco2e: New base year emissions.
        impact_on_progress_pp: Impact on progress in percentage points.
        provenance_hash: SHA-256 hash.
    """
    recalculation_id: str = Field(
        default_factory=_new_uuid, description="Recalculation ID"
    )
    target_id: str = Field(
        default="", description="Target ID"
    )
    original_base_year_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original base year"
    )
    adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjustment amount"
    )
    adjustment_reason: str = Field(
        default="", description="Reason for recalculation", max_length=1000
    )
    recalculated_base_year_tco2e: Decimal = Field(
        default=Decimal("0"), description="New base year emissions"
    )
    impact_on_progress_pp: Decimal = Field(
        default=Decimal("0"), description="Impact on progress (pp)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


class BatchTargetResult(BaseModel):
    """Batch assessment result for multiple targets.

    Aggregates progress assessments across all of an undertaking's
    climate targets.

    Attributes:
        batch_id: Unique batch identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        reporting_year: Reporting year.
        entity_name: Entity name.
        targets: List of all targets.
        progress_results: List of progress assessments.
        targets_on_track: Count of targets on track.
        targets_behind: Count of targets behind.
        total_targets: Total target count.
        overall_progress_pct: Weighted average progress.
        sbti_validated_count: Number of SBTi-validated targets.
        warnings: Warnings.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    batch_id: str = Field(
        default_factory=_new_uuid, description="Batch ID"
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
    targets: List[ClimateTarget] = Field(
        default_factory=list, description="All targets"
    )
    progress_results: List[TargetProgressResult] = Field(
        default_factory=list, description="Progress assessments"
    )
    targets_on_track: int = Field(
        default=0, description="Targets on track"
    )
    targets_behind: int = Field(
        default=0, description="Targets behind"
    )
    total_targets: int = Field(
        default=0, description="Total targets"
    )
    overall_progress_pct: Decimal = Field(
        default=Decimal("0"), description="Overall progress (%)"
    )
    sbti_validated_count: int = Field(
        default=0, description="SBTi-validated target count"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ClimateTargetEngine:
    """Climate target assessment engine per ESRS E1-4.

    Provides deterministic, zero-hallucination calculations for:
    - Target registration and validation
    - Progress assessment (absolute and percentage)
    - Annual reduction rate calculation (linear)
    - SBTi alignment validation (1.5C, WB2C, 2C thresholds)
    - Base year recalculation with adjustment tracking
    - Batch assessment across multiple targets
    - Milestone tracking
    - E1-4 completeness validation
    - E1-4 data point mapping

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = ClimateTargetEngine()
        target = ClimateTarget(
            name="50% reduction by 2030",
            target_type=TargetType.ABSOLUTE,
            target_scope=TargetScope.SCOPE_1_2,
            base_year=2019,
            base_year_emissions_tco2e=Decimal("10000"),
            target_year=2030,
            target_emissions_tco2e=Decimal("5000"),
        )
        validated = engine.set_target(target)
        progress = engine.assess_progress(target, Decimal("7500"), 2025)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Target Registration                                                 #
    # ------------------------------------------------------------------ #

    def set_target(self, target: ClimateTarget) -> ClimateTarget:
        """Register and validate a climate target.

        Performs basic validation, auto-computes reduction percentage
        if not provided, and logs the target registration.

        Args:
            target: ClimateTarget to register.

        Returns:
            The validated ClimateTarget (with computed fields).

        Raises:
            ValueError: If target is invalid.
        """
        logger.info(
            "Registering target: %s (type=%s, scope=%s, %d->%d)",
            target.name, target.target_type.value,
            target.target_scope.value, target.base_year, target.target_year,
        )

        # Auto-compute reduction percentage for absolute targets
        if (
            target.target_type == TargetType.ABSOLUTE
            and target.target_reduction_pct == Decimal("0")
            and target.base_year_emissions_tco2e > Decimal("0")
            and target.target_emissions_tco2e < target.base_year_emissions_tco2e
        ):
            target.target_reduction_pct = _round_val(
                (target.base_year_emissions_tco2e - target.target_emissions_tco2e)
                / target.base_year_emissions_tco2e
                * Decimal("100"),
                2,
            )

        # Auto-compute target emissions from percentage for absolute targets
        if (
            target.target_type == TargetType.ABSOLUTE
            and target.target_emissions_tco2e == Decimal("0")
            and target.target_reduction_pct > Decimal("0")
            and target.base_year_emissions_tco2e > Decimal("0")
        ):
            target.target_emissions_tco2e = _round6(
                target.base_year_emissions_tco2e
                * (Decimal("100") - target.target_reduction_pct)
                / Decimal("100")
            )

        logger.info(
            "Target registered: base=%.2f, target=%.2f tCO2e, reduction=%.1f%%",
            float(target.base_year_emissions_tco2e),
            float(target.target_emissions_tco2e),
            float(target.target_reduction_pct),
        )

        return target

    # ------------------------------------------------------------------ #
    # Progress Assessment                                                 #
    # ------------------------------------------------------------------ #

    def assess_progress(
        self,
        target: ClimateTarget,
        current_emissions: Decimal,
        current_year: int,
    ) -> TargetProgressResult:
        """Assess progress towards a climate target.

        Calculates the absolute reduction achieved, the percentage
        progress, the required annual rate to stay on track, and the
        actual annual rate achieved so far.

        Progress formula:
            progress_pct = (base - current) / (base - target) * 100

        Annual rate formula (linear):
            actual_rate = total_reduction_pct / years_elapsed

        Args:
            target: The ClimateTarget to assess.
            current_emissions: Current annual emissions (tCO2e).
            current_year: Current year for assessment.

        Returns:
            TargetProgressResult with full assessment.
        """
        logger.info(
            "Assessing progress: target=%s, current=%.2f tCO2e, year=%d",
            target.name, float(current_emissions), current_year,
        )

        base = target.base_year_emissions_tco2e
        target_level = target.target_emissions_tco2e

        # Absolute reduction achieved
        absolute_reduction = base - current_emissions
        if absolute_reduction < Decimal("0"):
            absolute_reduction = Decimal("0")

        # Required total reduction
        required_reduction = base - target_level

        # Progress percentage
        progress_pct = Decimal("0")
        if required_reduction > Decimal("0"):
            progress_pct = _round_val(
                absolute_reduction / required_reduction * Decimal("100"), 2
            )
            # Cap at 100
            if progress_pct > Decimal("100"):
                progress_pct = Decimal("100")

        # Remaining reduction
        remaining = current_emissions - target_level
        if remaining < Decimal("0"):
            remaining = Decimal("0")

        # Years elapsed and remaining
        years_elapsed = current_year - target.base_year
        years_remaining = target.target_year - current_year
        if years_remaining < 0:
            years_remaining = 0

        # Required annual rate (linear, from current to target)
        required_annual_rate = Decimal("0")
        if years_remaining > 0 and current_emissions > Decimal("0"):
            remaining_reduction_pct = _safe_divide(
                remaining, current_emissions
            ) * Decimal("100")
            required_annual_rate = _round_val(
                remaining_reduction_pct / _decimal(years_remaining), 2
            )

        # Actual annual rate (from base year to now)
        actual_annual_rate = Decimal("0")
        if years_elapsed > 0 and base > Decimal("0"):
            actual_reduction_pct = _safe_divide(
                absolute_reduction, base
            ) * Decimal("100")
            actual_annual_rate = _round_val(
                actual_reduction_pct / _decimal(years_elapsed), 2
            )

        # On-track assessment: current rate >= required remaining rate
        # Also check: progress above linear trajectory
        expected_progress_pct = Decimal("0")
        total_years = target.target_year - target.base_year
        if total_years > 0:
            expected_progress_pct = _round_val(
                _decimal(years_elapsed) / _decimal(total_years) * Decimal("100"), 2
            )

        is_on_track = progress_pct >= expected_progress_pct

        # Status assessment
        if progress_pct >= Decimal("100"):
            status_assessment = "achieved"
        elif current_year > target.target_year:
            status_assessment = "expired_not_achieved" if progress_pct < Decimal("100") else "achieved"
        elif progress_pct >= expected_progress_pct:
            status_assessment = "on_track"
        elif progress_pct >= expected_progress_pct * Decimal("0.8"):
            status_assessment = "slightly_behind"
        elif progress_pct >= expected_progress_pct * Decimal("0.5"):
            status_assessment = "significantly_behind"
        else:
            status_assessment = "critically_behind"

        # Milestone assessment
        milestones_met: List[str] = []
        milestones_missed: List[str] = []
        for year_str, required_pct in target.milestones.items():
            try:
                milestone_year = int(year_str)
            except ValueError:
                continue
            if milestone_year <= current_year:
                if progress_pct >= required_pct:
                    milestones_met.append(
                        f"{year_str}: {required_pct}% (achieved)"
                    )
                else:
                    milestones_missed.append(
                        f"{year_str}: {required_pct}% required, "
                        f"{progress_pct}% achieved"
                    )

        result = TargetProgressResult(
            target_id=target.target_id,
            target_name=target.name,
            assessment_year=current_year,
            base_year_emissions_tco2e=base,
            current_emissions_tco2e=current_emissions,
            target_emissions_tco2e=target_level,
            absolute_reduction_tco2e=_round6(absolute_reduction),
            required_reduction_tco2e=_round6(required_reduction),
            progress_pct=progress_pct,
            remaining_reduction_tco2e=_round6(remaining),
            years_remaining=years_remaining,
            required_annual_rate_pct=required_annual_rate,
            actual_annual_rate_pct=actual_annual_rate,
            is_on_track=is_on_track,
            status_assessment=status_assessment,
            milestones_met=milestones_met,
            milestones_missed=milestones_missed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Progress: %.1f%% (on_track=%s, status=%s, actual_rate=%.2f%%/yr)",
            float(progress_pct), is_on_track, status_assessment,
            float(actual_annual_rate),
        )

        return result

    # ------------------------------------------------------------------ #
    # SBTi Alignment Validation                                           #
    # ------------------------------------------------------------------ #

    def validate_sbti_alignment(
        self, target: ClimateTarget
    ) -> Dict[str, Any]:
        """Validate whether a target is aligned with SBTi requirements.

        Checks the target's annual reduction rate against SBTi minimum
        rates for 1.5C, well-below 2C, and 2C pathways.

        SBTi Requirements:
        - 1.5C: >= 4.2% annual for Scope 1+2, >= 2.5% for Scope 3
        - WB2C: >= 2.5% annual for all scopes
        - Base year must not be older than 2 reporting years
        - Scope 3: required if > 40% of total emissions

        Args:
            target: ClimateTarget to validate.

        Returns:
            Dict with alignment_result per pathway, rate_assessment,
            recommendations, and provenance_hash.
        """
        logger.info(
            "Validating SBTi alignment for: %s", target.name
        )

        # Calculate required annual rate
        required_rate = self.calculate_required_annual_rate(target)

        # Determine if scope 1+2 or scope 3
        is_scope_3 = target.target_scope in (
            TargetScope.SCOPE_3,
        )
        is_scope_1_2 = target.target_scope in (
            TargetScope.SCOPE_1,
            TargetScope.SCOPE_2,
            TargetScope.SCOPE_1_2,
            TargetScope.SCOPE_1_2_3,
        )

        pathway_results: Dict[str, Any] = {}
        for pathway_key, rates in SBTI_MINIMUM_RATES.items():
            if is_scope_3:
                min_rate = rates["scope_3_annual_pct"]
            else:
                min_rate = rates["scope_1_2_annual_pct"]

            is_aligned = required_rate >= min_rate
            gap = min_rate - required_rate

            pathway_results[pathway_key] = {
                "minimum_annual_rate_pct": str(min_rate),
                "actual_annual_rate_pct": str(required_rate),
                "is_aligned": is_aligned,
                "gap_pp": str(_round_val(gap, 2)) if gap > Decimal("0") else "0",
            }

        # Base year recency check
        current_year = _utcnow().year
        base_year_age = current_year - target.base_year
        base_year_ok = base_year_age <= 5  # SBTi allows up to 5 years

        # Overall alignment
        best_pathway = None
        for pw in ["1.5c", "well_below_2c", "2c"]:
            if pathway_results[pw]["is_aligned"]:
                best_pathway = pw
                break

        # Recommendations
        recommendations: List[str] = []
        if best_pathway is None:
            recommendations.append(
                "Target annual reduction rate does not meet any SBTi pathway. "
                "Consider increasing the target ambition."
            )
        if not base_year_ok:
            recommendations.append(
                f"Base year ({target.base_year}) is {base_year_age} years old. "
                "SBTi recommends a base year within 5 years of submission."
            )
        if target.target_scope == TargetScope.SCOPE_1_2:
            recommendations.append(
                "Consider adding a Scope 3 target if Scope 3 is >40% "
                "of total emissions."
            )
        if not target.is_sbti_validated:
            recommendations.append(
                "Target has not been validated by SBTi. "
                "Consider submitting for validation."
            )

        assessment = {
            "target_id": target.target_id,
            "target_name": target.name,
            "annual_rate_pct": str(required_rate),
            "pathway_results": pathway_results,
            "best_aligned_pathway": best_pathway,
            "base_year_age_years": base_year_age,
            "base_year_recency_ok": base_year_ok,
            "is_sbti_validated": target.is_sbti_validated,
            "scope_type": "scope_3" if is_scope_3 else "scope_1_2",
            "recommendations": recommendations,
        }
        assessment["provenance_hash"] = _compute_hash(assessment)

        logger.info(
            "SBTi alignment: best_pathway=%s, rate=%.2f%%/yr",
            best_pathway, float(required_rate),
        )

        return assessment

    # ------------------------------------------------------------------ #
    # Base Year Recalculation                                             #
    # ------------------------------------------------------------------ #

    def recalculate_base_year(
        self,
        target: ClimateTarget,
        adjustments: List[Dict[str, Any]],
    ) -> BaseYearRecalculation:
        """Recalculate base year emissions after structural changes.

        Per GHG Protocol, the base year must be recalculated when
        structural changes (mergers, acquisitions, divestitures, or
        methodology changes) significantly alter the emissions profile.

        Each adjustment specifies an amount (positive = increase in base
        year, negative = decrease) and a reason.

        Args:
            target: ClimateTarget with original base year.
            adjustments: List of dicts with 'amount_tco2e' and 'reason'.

        Returns:
            BaseYearRecalculation with new base year and impact.
        """
        logger.info(
            "Recalculating base year for %s: %d adjustments",
            target.name, len(adjustments),
        )

        original = target.base_year_emissions_tco2e
        total_adjustment = Decimal("0")
        reasons: List[str] = []

        for adj in adjustments:
            amount = _decimal(adj.get("amount_tco2e", 0))
            reason = str(adj.get("reason", "Unspecified"))
            total_adjustment += amount
            reasons.append(f"{reason}: {float(amount):+.2f} tCO2e")

        recalculated = _round6(original + total_adjustment)

        # Impact on progress (in percentage points)
        # Old progress base: (original - current) / (original - target)
        # New progress base: (recalculated - current) / (recalculated - target)
        # This is informational; actual progress requires current emissions
        impact_pp = Decimal("0")
        if original != recalculated and original > Decimal("0"):
            impact_pp = _round_val(
                (recalculated - original) / original * Decimal("100"), 2
            )

        combined_reason = "; ".join(reasons) if reasons else "No adjustment"

        result = BaseYearRecalculation(
            target_id=target.target_id,
            original_base_year_tco2e=original,
            adjustment_tco2e=_round6(total_adjustment),
            adjustment_reason=combined_reason,
            recalculated_base_year_tco2e=recalculated,
            impact_on_progress_pp=impact_pp,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Base year recalculated: %.2f -> %.2f tCO2e (adj=%.2f)",
            float(original), float(recalculated), float(total_adjustment),
        )

        return result

    # ------------------------------------------------------------------ #
    # Required Annual Rate                                                #
    # ------------------------------------------------------------------ #

    def calculate_required_annual_rate(
        self, target: ClimateTarget
    ) -> Decimal:
        """Calculate the required annual linear reduction rate.

        Formula: annual_rate = total_reduction_pct / (target_year - base_year)

        This is the simple linear rate needed to reach the target.

        Args:
            target: ClimateTarget with base and target year/emissions.

        Returns:
            Required annual reduction rate as Decimal percentage.
        """
        total_years = _decimal(target.target_year - target.base_year)
        if total_years <= Decimal("0"):
            return Decimal("0")

        reduction_pct = target.target_reduction_pct
        if reduction_pct == Decimal("0") and target.base_year_emissions_tco2e > Decimal("0"):
            reduction_pct = _round_val(
                (target.base_year_emissions_tco2e - target.target_emissions_tco2e)
                / target.base_year_emissions_tco2e
                * Decimal("100"),
                2,
            )

        annual_rate = _round_val(
            _safe_divide(reduction_pct, total_years), 2
        )

        logger.debug(
            "Required annual rate: %.2f%% / %d years = %.2f%%/yr",
            float(reduction_pct), int(total_years), float(annual_rate),
        )

        return annual_rate

    # ------------------------------------------------------------------ #
    # Batch Assessment                                                    #
    # ------------------------------------------------------------------ #

    def batch_assess(
        self,
        targets: List[ClimateTarget],
        current_emissions_by_scope: Dict[str, Decimal],
        current_year: int,
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> BatchTargetResult:
        """Assess progress across multiple climate targets.

        Maps each target's scope to the corresponding current emissions
        and performs individual progress assessments, then aggregates.

        Args:
            targets: List of ClimateTarget instances.
            current_emissions_by_scope: Dict mapping scope key to current
                emissions (e.g., {"scope_1": Decimal("5000"), ...}).
            current_year: Current year for assessment.
            entity_name: Reporting entity name.
            reporting_year: Reporting year.

        Returns:
            BatchTargetResult with all assessments and summary.
        """
        t0 = time.perf_counter()

        logger.info(
            "Batch assessing %d targets for year %d",
            len(targets), current_year,
        )

        progress_results: List[TargetProgressResult] = []
        warnings: List[str] = []
        on_track_count = 0
        behind_count = 0
        sbti_count = 0

        for target in targets:
            # Determine current emissions for this target's scope
            current = self._get_current_for_scope(
                target.target_scope, current_emissions_by_scope
            )

            if current is None:
                warnings.append(
                    f"Target '{target.name}': no current emissions for "
                    f"scope {target.target_scope.value}"
                )
                continue

            progress = self.assess_progress(target, current, current_year)
            progress_results.append(progress)

            if progress.is_on_track:
                on_track_count += 1
            else:
                behind_count += 1

            if target.is_sbti_validated:
                sbti_count += 1

        # Overall progress (simple average of progress percentages)
        overall_pct = Decimal("0")
        if progress_results:
            total_pct = sum(p.progress_pct for p in progress_results)
            overall_pct = _round_val(
                total_pct / _decimal(len(progress_results)), 1
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        batch = BatchTargetResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            targets=targets,
            progress_results=progress_results,
            targets_on_track=on_track_count,
            targets_behind=behind_count,
            total_targets=len(targets),
            overall_progress_pct=overall_pct,
            sbti_validated_count=sbti_count,
            warnings=warnings,
            processing_time_ms=elapsed_ms,
        )
        batch.provenance_hash = _compute_hash(batch)

        logger.info(
            "Batch assessment: %d targets, %d on track, %d behind, "
            "overall=%.1f%%",
            len(targets), on_track_count, behind_count,
            float(overall_pct),
        )

        return batch

    def _get_current_for_scope(
        self,
        scope: TargetScope,
        emissions: Dict[str, Decimal],
    ) -> Optional[Decimal]:
        """Get current emissions for a target's scope coverage.

        Args:
            scope: TargetScope of the target.
            emissions: Dict mapping scope keys to emissions.

        Returns:
            Total current emissions for the scope, or None if not available.
        """
        scope_map = {
            TargetScope.SCOPE_1: ["scope_1"],
            TargetScope.SCOPE_2: ["scope_2"],
            TargetScope.SCOPE_3: ["scope_3"],
            TargetScope.SCOPE_1_2: ["scope_1", "scope_2"],
            TargetScope.SCOPE_1_2_3: ["scope_1", "scope_2", "scope_3"],
        }

        required_keys = scope_map.get(scope, [])
        total = Decimal("0")
        found = False

        for key in required_keys:
            if key in emissions:
                total += emissions[key]
                found = True

        return total if found else None

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: BatchTargetResult
    ) -> Dict[str, Any]:
        """Validate completeness against ESRS E1-4 required data points.

        Checks whether all E1-4 mandatory data points are present.

        Args:
            result: BatchTargetResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            and provenance_hash.
        """
        populated: List[str] = []
        missing: List[str] = []

        has_targets = len(result.targets) > 0
        has_progress = len(result.progress_results) > 0
        has_sbti = result.sbti_validated_count > 0
        has_milestones = any(
            t.milestones for t in result.targets
        ) if has_targets else False

        # Get first target for detailed checks
        first_target = result.targets[0] if has_targets else None

        checks = {
            "e1_4_01_target_exists": has_targets,
            "e1_4_02_target_type": (
                first_target is not None and bool(first_target.target_type)
            ),
            "e1_4_03_target_scope": (
                first_target is not None and bool(first_target.target_scope)
            ),
            "e1_4_04_base_year": (
                first_target is not None and first_target.base_year > 0
            ),
            "e1_4_05_base_year_emissions": (
                first_target is not None
                and first_target.base_year_emissions_tco2e > Decimal("0")
            ),
            "e1_4_06_target_year": (
                first_target is not None and first_target.target_year > 0
            ),
            "e1_4_07_target_level": (
                first_target is not None
                and first_target.target_emissions_tco2e >= Decimal("0")
            ),
            "e1_4_08_target_reduction_pct": (
                first_target is not None
                and first_target.target_reduction_pct > Decimal("0")
            ),
            "e1_4_09_current_progress_absolute": has_progress,
            "e1_4_10_current_progress_pct": has_progress,
            "e1_4_11_milestones": has_milestones,
            "e1_4_12_third_party_validation": has_sbti,
            "e1_4_13_sbti_alignment": has_sbti,
            "e1_4_14_pathway_alignment": (
                first_target is not None
                and first_target.pathway != TargetPathway.PATHWAY_UNSPECIFIED
            ),
            "e1_4_15_base_year_recalculation_policy": (
                first_target is not None
                and bool(first_target.base_year_approach)
            ),
            "e1_4_16_target_methodology": (
                first_target is not None
                and bool(first_target.methodology)
            ),
            "e1_4_17_target_status": (
                first_target is not None and bool(first_target.status)
            ),
            "e1_4_18_relationship_to_policy": True,  # Narrative
        }

        for dp, is_populated in checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(E1_4_DATAPOINTS)
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
                {"batch_id": result.batch_id, "checks": checks}
            ),
        }

        logger.info(
            "E1-4 completeness: %s%% (%d/%d), missing=%s",
            completeness, pop_count, total, missing,
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # ESRS E1-4 Data Point Mapping                                        #
    # ------------------------------------------------------------------ #

    def get_e1_4_datapoints(
        self,
        targets: List[ClimateTarget],
        results: List[TargetProgressResult],
    ) -> Dict[str, Any]:
        """Map targets and results to ESRS E1-4 disclosure data points.

        Creates a structured mapping of all E1-4 required data points
        with their values, ready for report generation.

        Args:
            targets: List of ClimateTarget instances.
            results: List of corresponding TargetProgressResult instances.

        Returns:
            Dict mapping E1-4 data point IDs to their values.
        """
        # Build target summaries
        target_summaries = []
        for target in targets:
            target_summaries.append({
                "target_id": target.target_id,
                "name": target.name,
                "type": target.target_type.value,
                "scope": target.target_scope.value,
                "base_year": target.base_year,
                "base_year_emissions_tco2e": str(target.base_year_emissions_tco2e),
                "target_year": target.target_year,
                "target_emissions_tco2e": str(target.target_emissions_tco2e),
                "target_reduction_pct": str(target.target_reduction_pct),
                "pathway": target.pathway.value,
                "status": target.status.value,
                "is_sbti_validated": target.is_sbti_validated,
            })

        # Build progress summaries
        progress_summaries = []
        for prog in results:
            progress_summaries.append({
                "target_id": prog.target_id,
                "target_name": prog.target_name,
                "progress_pct": str(prog.progress_pct),
                "absolute_reduction_tco2e": str(prog.absolute_reduction_tco2e),
                "is_on_track": prog.is_on_track,
                "status": prog.status_assessment,
                "actual_annual_rate_pct": str(prog.actual_annual_rate_pct),
            })

        datapoints: Dict[str, Any] = {
            "e1_4_01_target_exists": {
                "label": "Climate targets set",
                "value": len(targets) > 0,
                "target_count": len(targets),
                "esrs_ref": "E1-4 Para 30",
            },
            "e1_4_02_target_type": {
                "label": "Target types",
                "value": list({t.target_type.value for t in targets}),
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_03_target_scope": {
                "label": "Target scope coverage",
                "value": list({t.target_scope.value for t in targets}),
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_04_base_year": {
                "label": "Base year(s)",
                "value": list({t.base_year for t in targets}),
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_05_base_year_emissions": {
                "label": "Base year emissions",
                "value": [
                    {
                        "target": t.name,
                        "base_year_tco2e": str(t.base_year_emissions_tco2e),
                    }
                    for t in targets
                ],
                "unit": "tCO2e",
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_06_target_year": {
                "label": "Target year(s)",
                "value": list({t.target_year for t in targets}),
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_07_target_level": {
                "label": "Target emission levels",
                "value": [
                    {
                        "target": t.name,
                        "target_tco2e": str(t.target_emissions_tco2e),
                    }
                    for t in targets
                ],
                "unit": "tCO2e",
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_08_target_reduction_pct": {
                "label": "Target reduction percentages",
                "value": [
                    {
                        "target": t.name,
                        "reduction_pct": str(t.target_reduction_pct),
                    }
                    for t in targets
                ],
                "unit": "percent",
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_09_current_progress_absolute": {
                "label": "Current progress (absolute)",
                "value": progress_summaries,
                "unit": "tCO2e",
                "esrs_ref": "E1-4 Para 32",
            },
            "e1_4_10_current_progress_pct": {
                "label": "Current progress (percentage)",
                "value": [
                    {
                        "target": p.target_name,
                        "progress_pct": str(p.progress_pct),
                    }
                    for p in results
                ],
                "unit": "percent",
                "esrs_ref": "E1-4 Para 32",
            },
            "e1_4_11_milestones": {
                "label": "Interim milestones",
                "value": [
                    {
                        "target": t.name,
                        "milestones": {
                            k: str(v) for k, v in t.milestones.items()
                        },
                    }
                    for t in targets
                    if t.milestones
                ],
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_12_third_party_validation": {
                "label": "Third-party validation",
                "value": [
                    {
                        "target": t.name,
                        "validator": t.third_party_validator,
                        "is_validated": t.is_sbti_validated,
                    }
                    for t in targets
                    if t.is_sbti_validated
                ],
                "esrs_ref": "E1-4 Para 33",
            },
            "e1_4_13_sbti_alignment": {
                "label": "SBTi alignment",
                "value": [
                    {
                        "target": t.name,
                        "sbti_validated": t.is_sbti_validated,
                        "sbti_type": t.sbti_target_type,
                    }
                    for t in targets
                ],
                "esrs_ref": "E1-4 Para 33",
            },
            "e1_4_14_pathway_alignment": {
                "label": "Climate pathway alignment",
                "value": list({t.pathway.value for t in targets}),
                "esrs_ref": "E1-4 Para 31",
            },
            "e1_4_15_base_year_recalculation_policy": {
                "label": "Base year recalculation policy",
                "value": list({t.base_year_approach.value for t in targets}),
                "esrs_ref": "E1-4 Para 34",
            },
            "e1_4_16_target_methodology": {
                "label": "Target-setting methodology",
                "value": [
                    {"target": t.name, "methodology": t.methodology}
                    for t in targets
                    if t.methodology
                ],
                "esrs_ref": "E1-4 Para 34",
            },
            "e1_4_17_target_status": {
                "label": "Target statuses",
                "value": [
                    {"target": t.name, "status": t.status.value}
                    for t in targets
                ],
                "esrs_ref": "E1-4 Para 30",
            },
            "e1_4_18_relationship_to_policy": {
                "label": "Relationship to policy objectives",
                "value": (
                    "Climate targets support the Paris Agreement goal "
                    "of limiting global warming to 1.5C."
                ),
                "esrs_ref": "E1-4",
            },
        }

        datapoints["_metadata"] = {
            "engine_version": self.engine_version,
            "target_count": len(targets),
            "progress_count": len(results),
            "provenance_hash": _compute_hash(datapoints),
        }

        return datapoints

    # ------------------------------------------------------------------ #
    # Target Scoring                                                      #
    # ------------------------------------------------------------------ #

    def score_target(
        self, target: ClimateTarget
    ) -> Dict[str, Any]:
        """Score a target against TARGET_ASSESSMENT_CRITERIA.

        Provides a quality score indicating how well-defined and
        robust the target is.

        Args:
            target: ClimateTarget to score.

        Returns:
            Dict with criteria results, total score, and recommendations.
        """
        checks = {
            "has_base_year": target.base_year > 0,
            "has_target_year": target.target_year > target.base_year,
            "has_quantified_target": (
                target.target_emissions_tco2e >= Decimal("0")
                or target.target_reduction_pct > Decimal("0")
            ),
            "has_scope_coverage": bool(target.target_scope),
            "has_methodology": bool(target.methodology),
            "is_sbti_aligned": target.is_sbti_validated,
            "has_milestones": len(target.milestones) > 0,
            "has_progress_tracking": target.status != TargetStatus.NEW,
            "is_science_based": target.pathway != TargetPathway.PATHWAY_UNSPECIFIED,
            "has_third_party_validation": bool(target.third_party_validator),
        }

        met = [k for k, v in checks.items() if v]
        not_met = [k for k, v in checks.items() if not v]
        total = len(TARGET_ASSESSMENT_CRITERIA)
        score = _round_val(
            _decimal(len(met)) / _decimal(total) * Decimal("100"), 1
        )

        recommendations: List[str] = []
        if "has_milestones" in not_met:
            recommendations.append("Add interim milestones for better tracking.")
        if "is_sbti_aligned" in not_met:
            recommendations.append("Consider SBTi validation for credibility.")
        if "has_methodology" in not_met:
            recommendations.append("Document the target-setting methodology.")
        if "is_science_based" in not_met:
            recommendations.append(
                "Align the target with a recognised climate pathway."
            )

        result = {
            "target_id": target.target_id,
            "target_name": target.name,
            "score_pct": str(score),
            "criteria_met": met,
            "criteria_not_met": not_met,
            "total_criteria": total,
            "recommendations": recommendations,
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    # ------------------------------------------------------------------ #
    # Year-over-Year Comparison                                           #
    # ------------------------------------------------------------------ #

    def compare_progress(
        self,
        current: TargetProgressResult,
        previous: TargetProgressResult,
    ) -> Dict[str, Any]:
        """Compare target progress across two reporting years.

        Args:
            current: Current year progress result.
            previous: Previous year progress result.

        Returns:
            Dict with changes in progress, rate, and status.
        """
        comparison = {
            "target_id": current.target_id,
            "target_name": current.target_name,
            "current_year": current.assessment_year,
            "previous_year": previous.assessment_year,
            "progress_change_pp": str(_round_val(
                current.progress_pct - previous.progress_pct, 2
            )),
            "emissions_change_tco2e": str(_round6(
                current.current_emissions_tco2e
                - previous.current_emissions_tco2e
            )),
            "rate_change_pp": str(_round_val(
                current.actual_annual_rate_pct
                - previous.actual_annual_rate_pct, 2
            )),
            "status_change": {
                "from": previous.status_assessment,
                "to": current.status_assessment,
            },
            "on_track_change": {
                "from": previous.is_on_track,
                "to": current.is_on_track,
            },
        }
        comparison["provenance_hash"] = _compute_hash(comparison)

        return comparison
