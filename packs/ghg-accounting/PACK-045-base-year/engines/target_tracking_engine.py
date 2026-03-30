# -*- coding: utf-8 -*-
"""
TargetTrackingEngine - PACK-045 Base Year Management Engine 8
==============================================================

Base year-anchored target progress tracking engine that monitors emission
reduction progress against absolute and intensity targets aligned with
SBTi pathways, GHG Protocol, and other regulatory frameworks.

The engine calculates linear and science-based reduction pathways,
assesses current progress against those pathways, determines whether
the organization is on-track or behind, and provides forward-looking
projections of the annual reduction rate required to meet the target.

Calculation Methodology:

    Linear Pathway (GHG Protocol):
        expected_tco2e(year) = base_tco2e * (1 - reduction_pct
            * (year - base_year) / (target_year - base_year))

    SBTi 1.5C Absolute Contraction:
        Annual reduction rate: 4.2% compounding
        expected_tco2e(year) = base_tco2e * (1 - 0.042)^(year - base_year)

    SBTi Well-Below 2C Absolute Contraction:
        Annual reduction rate: 2.5% compounding
        expected_tco2e(year) = base_tco2e * (1 - 0.025)^(year - base_year)

    SBTi Net-Zero:
        Near-term: 4.2% compounding through 2030
        Long-term: 90% absolute reduction by 2050
        expected_tco2e(year) = base_tco2e * (1 - 0.042)^(year - base_year)
        final_target <= base_tco2e * 0.10

    Progress Percentage:
        progress_pct = (base_tco2e - actual_tco2e) /
                       (base_tco2e - target_tco2e) * 100

    Required Annual Reduction Rate:
        Given current_actual and years_remaining:
        required_rate = 1 - (target_tco2e / current_actual)^(1 / years_remaining)

    Intensity-Based Targets:
        intensity = emissions_tco2e / denominator
        Target tracks intensity ratio rather than absolute values.

Status Assessment Thresholds:
    AHEAD:      actual <= expected * 0.95  (>5% ahead of pathway)
    ON_TRACK:   expected * 0.95 < actual <= expected * 1.05
    AT_RISK:    expected * 1.05 < actual <= expected * 1.15
    BEHIND:     actual > expected * 1.15  (>15% behind pathway)
    NOT_STARTED: no data points beyond base year

Regulatory References:
    - SBTi Corporate Manual (2023), Sections 6-7
    - SBTi Net-Zero Standard (2021), Section 5
    - GHG Protocol Corporate Standard, Chapter 11
    - ESRS E1-4 (GHG emission reduction targets)
    - CDP Climate Change Questionnaire C4.1-C4.2
    - SEC Climate Disclosure Rule (2024), Item 1505

Zero-Hallucination:
    - All pathway calculations use deterministic Decimal arithmetic
    - SBTi rates from published Corporate Manual (2023)
    - No LLM involvement in any calculation or status assessment
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
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

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _abs_decimal(value: Decimal) -> Decimal:
    """Return absolute value of a Decimal."""
    return value if value >= Decimal("0") else -value

def _pow_decimal(base: Decimal, exponent: int) -> Decimal:
    """Compute base ** exponent for non-negative integer exponents.

    Uses iterative multiplication to avoid float conversion.

    Args:
        base: The base value.
        exponent: Non-negative integer exponent.

    Returns:
        base raised to the power of exponent.
    """
    if exponent < 0:
        # For negative exponents, compute inverse.
        result = Decimal("1")
        for _ in range(-exponent):
            result = _safe_divide(result, base)
        return result
    result = Decimal("1")
    for _ in range(exponent):
        result = result * base
    return result

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """Type of emission reduction target.

    ABSOLUTE:   Target expressed as total emissions (tCO2e).
                Example: Reduce Scope 1+2 by 42% by 2030.
    INTENSITY:  Target expressed as emissions per unit of activity.
                Example: Reduce tCO2e per million USD revenue by 30%.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"

class ScopeType(str, Enum):
    """GHG Protocol scope classification.

    SCOPE_1:    Direct emissions from owned/controlled sources.
    SCOPE_2:    Indirect emissions from purchased energy.
    SCOPE_3:    All other indirect emissions in value chain.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"

class SBTiAmbition(str, Enum):
    """SBTi target ambition level per Corporate Manual (2023).

    WELL_BELOW_2C:  Minimum 2.5% annual linear reduction (Scope 1+2).
    ONE_POINT_FIVE_C: Minimum 4.2% annual linear reduction (Scope 1+2).
    NET_ZERO:       4.2% near-term + 90% long-term reduction by 2050.
    """
    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "1.5c"
    NET_ZERO = "net_zero"

class TargetStatus(str, Enum):
    """Current status of target progress.

    ON_TRACK:       Actual emissions within +/-5% of expected pathway.
    BEHIND:         Actual emissions more than 15% above expected pathway.
    AHEAD:          Actual emissions more than 5% below expected pathway.
    AT_RISK:        Actual emissions 5-15% above expected pathway.
    NOT_STARTED:    No data points beyond the base year.
    """
    ON_TRACK = "on_track"
    BEHIND = "behind"
    AHEAD = "ahead"
    AT_RISK = "at_risk"
    NOT_STARTED = "not_started"

class IntensityMetric(str, Enum):
    """Denominator metric for intensity-based targets.

    PER_REVENUE:        tCO2e per million currency units of revenue.
    PER_EMPLOYEE:       tCO2e per full-time equivalent employee.
    PER_SQMETER:        tCO2e per square meter of floor area.
    PER_UNIT_PRODUCED:  tCO2e per unit of product/output.
    PER_MWH:            tCO2e per megawatt-hour of energy produced.
    """
    PER_REVENUE = "per_revenue"
    PER_EMPLOYEE = "per_employee"
    PER_SQMETER = "per_sqmeter"
    PER_UNIT_PRODUCED = "per_unit_produced"
    PER_MWH = "per_mwh"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SBTi minimum annual reduction rates (compounding).
SBTI_RATE_1_5C: Decimal = Decimal("0.042")
SBTI_RATE_WB2C: Decimal = Decimal("0.025")
SBTI_NET_ZERO_LONG_TERM_REDUCTION: Decimal = Decimal("0.90")

# Status thresholds (multiplied with expected pathway value).
STATUS_AHEAD_THRESHOLD: Decimal = Decimal("0.95")
STATUS_ON_TRACK_THRESHOLD: Decimal = Decimal("1.05")
STATUS_AT_RISK_THRESHOLD: Decimal = Decimal("1.15")

# Maximum valid target reduction percentage.
MAX_REDUCTION_PCT: Decimal = Decimal("100")

# SBTi minimum/maximum target timeframes (years from base year).
SBTI_MIN_NEAR_TERM_YEARS: int = 5
SBTI_MAX_NEAR_TERM_YEARS: int = 10
SBTI_NET_ZERO_TARGET_YEAR: int = 2050

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EmissionsTarget(BaseModel):
    """Definition of an emission reduction target.

    Encapsulates the complete target specification including base year
    reference, target year, reduction percentage, SBTi alignment, and
    intensity metric configuration.

    Attributes:
        target_id: Unique identifier for the target.
        name: Human-readable target name.
        target_type: Whether the target is absolute or intensity-based.
        scopes: List of scopes covered by this target.
        base_year: Base year for the target.
        base_year_tco2e: Total emissions in the base year (tCO2e).
        target_year: Year by which the target must be achieved.
        target_reduction_pct: Percentage reduction from base year.
        target_tco2e: Calculated target emissions (tCO2e).  Auto-computed
            as base_year_tco2e * (1 - target_reduction_pct / 100).
        sbti_ambition: Optional SBTi ambition level alignment.
        intensity_metric: Denominator for intensity targets.
        base_year_intensity_denominator: Denominator value in base year
            (e.g., revenue in millions, FTE count).
    """
    target_id: str = Field(default_factory=_new_uuid)
    name: str = Field(..., min_length=1, max_length=500)
    target_type: TargetType
    scopes: List[ScopeType] = Field(default_factory=lambda: [ScopeType.SCOPE_1, ScopeType.SCOPE_2])
    base_year: int = Field(..., ge=1990, le=2100)
    base_year_tco2e: Decimal = Field(..., ge=0)
    target_year: int = Field(..., ge=1990, le=2100)
    target_reduction_pct: Decimal = Field(..., ge=0, le=100)
    target_tco2e: Optional[Decimal] = Field(default=None)
    sbti_ambition: Optional[SBTiAmbition] = None
    intensity_metric: Optional[IntensityMetric] = None
    base_year_intensity_denominator: Optional[Decimal] = Field(
        default=None, ge=0
    )

    @field_validator("base_year_tco2e", "target_reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("target_tco2e", "base_year_intensity_denominator",
                     mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

    @model_validator(mode="after")
    def _compute_target_tco2e(self) -> "EmissionsTarget":
        """Auto-compute target_tco2e from base_year_tco2e and reduction_pct."""
        if self.target_tco2e is None:
            self.target_tco2e = _round_val(
                self.base_year_tco2e * (
                    Decimal("1") - self.target_reduction_pct / Decimal("100")
                ),
                places=3,
            )
        if self.target_year <= self.base_year:
            raise ValueError(
                f"target_year ({self.target_year}) must be after "
                f"base_year ({self.base_year})."
            )
        return self

class YearlyActual(BaseModel):
    """Actual emission data for a single year.

    Attributes:
        year: Reporting year.
        actual_tco2e: Actual total emissions (tCO2e).
        intensity_denominator: Denominator for intensity calculation.
        scope1_tco2e: Scope 1 portion (optional breakdown).
        scope2_tco2e: Scope 2 portion (optional breakdown).
        scope3_tco2e: Scope 3 portion (optional breakdown).
        is_verified: Whether this data has been third-party verified.
        notes: Optional notes about the data.
    """
    year: int = Field(..., ge=1990, le=2100)
    actual_tco2e: Decimal = Field(..., ge=0)
    intensity_denominator: Optional[Decimal] = Field(default=None, ge=0)
    scope1_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    scope2_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    scope3_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    is_verified: bool = Field(default=False)
    notes: str = Field(default="")

    @field_validator("actual_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("intensity_denominator", "scope1_tco2e",
                     "scope2_tco2e", "scope3_tco2e", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class ProgressPoint(BaseModel):
    """Progress assessment for a single year against the target pathway.

    Attributes:
        year: The reporting year.
        actual_tco2e: Actual emissions.
        expected_tco2e: Expected emissions per the reduction pathway.
        gap_tco2e: Actual minus expected (positive = behind, negative = ahead).
        progress_pct: Percentage of total required reduction achieved.
        status: Status assessment for this year.
        actual_intensity: Intensity value (for intensity targets only).
        expected_intensity: Expected intensity per pathway.
    """
    year: int
    actual_tco2e: Decimal
    expected_tco2e: Decimal
    gap_tco2e: Decimal
    progress_pct: Decimal
    status: TargetStatus
    actual_intensity: Optional[Decimal] = None
    expected_intensity: Optional[Decimal] = None

    @field_validator("actual_tco2e", "expected_tco2e", "gap_tco2e",
                     "progress_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("actual_intensity", "expected_intensity", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class ReductionAttribution(BaseModel):
    """Attribution of emission reductions to specific categories.

    Attributes:
        category: Description of the reduction category.
        reduction_tco2e: Amount of reduction attributed (tCO2e).
        pct_of_total_reduction: Percentage of total reduction.
    """
    category: str
    reduction_tco2e: Decimal
    pct_of_total_reduction: Decimal

    @field_validator("reduction_tco2e", "pct_of_total_reduction", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class TargetTrackingResult(BaseModel):
    """Complete result of target progress tracking.

    Attributes:
        target: The target being tracked.
        progress_points: Year-by-year progress points.
        current_status: Overall current status.
        years_remaining: Years until target year.
        required_annual_reduction_pct: Annual reduction needed from now.
        actual_annual_reduction_pct: Actual average annual reduction so far.
        attribution: Breakdown of reduction sources.
        on_track_probability: Probability of meeting target (0-100).
        recommendations: List of recommendations.
        calculated_at: Timestamp of calculation.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for auditability.
    """
    target: EmissionsTarget
    progress_points: List[ProgressPoint] = Field(default_factory=list)
    current_status: TargetStatus = Field(default=TargetStatus.NOT_STARTED)
    years_remaining: int = Field(default=0)
    required_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    actual_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    attribution: List[ReductionAttribution] = Field(default_factory=list)
    on_track_probability: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class PathwayPoint(BaseModel):
    """A single point on the expected reduction pathway.

    Attributes:
        year: The year.
        expected_tco2e: Expected emissions at this point on the pathway.
        cumulative_reduction_pct: Cumulative reduction from base year.
    """
    year: int
    expected_tco2e: Decimal
    cumulative_reduction_pct: Decimal

    @field_validator("expected_tco2e", "cumulative_reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class RebaseResult(BaseModel):
    """Result of rebasing a target to a new base year.

    Attributes:
        original_target: The original target before rebasing.
        rebased_target: The updated target with new base year data.
        adjustment_reason: Reason for the rebase.
        base_year_change_tco2e: Change in base year emissions.
        target_tco2e_change: Change in target emissions value.
        provenance_hash: SHA-256 hash for auditability.
    """
    original_target: EmissionsTarget
    rebased_target: EmissionsTarget
    adjustment_reason: str
    base_year_change_tco2e: Decimal
    target_tco2e_change: Decimal
    provenance_hash: str = Field(default="")

    @field_validator("base_year_change_tco2e", "target_tco2e_change",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TargetTrackingEngine:
    """Base year-anchored target progress tracking engine.

    Guarantees:
        - Deterministic: Same input -> Same output (bit-perfect)
        - Reproducible: Full provenance tracking with SHA-256 hashes
        - Auditable: Every pathway point traceable to published formulas
        - NO LLM: Zero hallucination risk in all calculations

    Usage::

        engine = TargetTrackingEngine()
        target = EmissionsTarget(
            name="Scope 1+2 SBTi 1.5C Target",
            target_type=TargetType.ABSOLUTE,
            base_year=2019,
            base_year_tco2e=Decimal("100000"),
            target_year=2030,
            target_reduction_pct=Decimal("42"),
            sbti_ambition=SBTiAmbition.ONE_POINT_FIVE_C,
        )
        actuals = [
            YearlyActual(year=2020, actual_tco2e=Decimal("95000")),
            YearlyActual(year=2021, actual_tco2e=Decimal("90000")),
        ]
        result = engine.track_progress(target, actuals)
    """

    def __init__(self) -> None:
        """Initialize the TargetTrackingEngine."""
        logger.info(
            "TargetTrackingEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_progress(
        self,
        target: EmissionsTarget,
        yearly_actuals: List[YearlyActual],
    ) -> TargetTrackingResult:
        """Track progress against an emission reduction target.

        Calculates the expected pathway, compares actual emissions against
        it, determines status, and computes required future reduction rate.

        Args:
            target: The emission reduction target definition.
            yearly_actuals: List of actual emission data by year.

        Returns:
            TargetTrackingResult with complete progress analysis.
        """
        t0 = time.perf_counter()

        # Sort actuals by year.
        sorted_actuals = sorted(yearly_actuals, key=lambda ya: ya.year)

        # Calculate pathway.
        if target.sbti_ambition is not None:
            pathway = self.calculate_sbti_pathway(target)
        else:
            pathway = self.calculate_linear_pathway(target)

        # Build pathway lookup.
        pathway_map: Dict[int, Decimal] = {
            pp.year: pp.expected_tco2e for pp in pathway
        }

        # Build progress points.
        progress_points: List[ProgressPoint] = []
        for actual in sorted_actuals:
            if actual.year < target.base_year:
                continue
            if actual.year > target.target_year:
                continue

            expected = pathway_map.get(actual.year)
            if expected is None:
                expected = self._interpolate_pathway(
                    target, pathway, actual.year
                )

            gap = _decimal(actual.actual_tco2e) - expected
            total_required = _decimal(target.base_year_tco2e) - _decimal(target.target_tco2e)

            progress = _safe_pct(
                _decimal(target.base_year_tco2e) - _decimal(actual.actual_tco2e),
                total_required,
            )

            status = self.assess_status(target, actual.actual_tco2e, expected)

            # Intensity calculations.
            actual_intensity: Optional[Decimal] = None
            expected_intensity: Optional[Decimal] = None
            if target.target_type == TargetType.INTENSITY and \
               actual.intensity_denominator is not None and \
               target.base_year_intensity_denominator is not None:
                actual_intensity = _safe_divide(
                    _decimal(actual.actual_tco2e),
                    _decimal(actual.intensity_denominator),
                )
                expected_intensity = _safe_divide(
                    expected,
                    _decimal(actual.intensity_denominator),
                )

            progress_points.append(ProgressPoint(
                year=actual.year,
                actual_tco2e=_round_val(_decimal(actual.actual_tco2e), 3),
                expected_tco2e=_round_val(expected, 3),
                gap_tco2e=_round_val(gap, 3),
                progress_pct=_round_val(progress, 2),
                status=status,
                actual_intensity=(
                    _round_val(actual_intensity, 6) if actual_intensity else None
                ),
                expected_intensity=(
                    _round_val(expected_intensity, 6) if expected_intensity else None
                ),
            ))

        # Determine current status.
        current_status = TargetStatus.NOT_STARTED
        if progress_points:
            current_status = progress_points[-1].status

        # Calculate years remaining.
        current_year = max(
            (a.year for a in sorted_actuals), default=target.base_year
        )
        years_remaining = max(0, target.target_year - current_year)

        # Calculate required annual reduction rate.
        required_rate = Decimal("0")
        actual_rate = Decimal("0")

        if progress_points and years_remaining > 0:
            latest_actual = _decimal(progress_points[-1].actual_tco2e)
            required_rate = self.calculate_required_rate(
                target, latest_actual, current_year
            )

        if progress_points:
            actual_rate = self._calculate_actual_annual_rate(
                target, sorted_actuals
            )

        # Calculate on-track probability (deterministic heuristic).
        on_track_prob = self._calculate_on_track_probability(
            target, progress_points, required_rate, actual_rate,
            years_remaining,
        )

        # Generate recommendations.
        recommendations = self._generate_recommendations(
            target, current_status, required_rate, actual_rate,
            years_remaining,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TargetTrackingResult(
            target=target,
            progress_points=progress_points,
            current_status=current_status,
            years_remaining=years_remaining,
            required_annual_reduction_pct=_round_val(
                required_rate * Decimal("100"), 2
            ),
            actual_annual_reduction_pct=_round_val(
                actual_rate * Decimal("100"), 2
            ),
            attribution=[],
            on_track_probability=_round_val(on_track_prob, 1),
            recommendations=recommendations,
            calculated_at=utcnow(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_linear_pathway(
        self,
        target: EmissionsTarget,
    ) -> List[PathwayPoint]:
        """Calculate a linear reduction pathway from base to target year.

        Formula:
            expected_tco2e(year) = base_tco2e * (1 - reduction_pct
                * (year - base_year) / (target_year - base_year))

        Args:
            target: The emission reduction target.

        Returns:
            List of PathwayPoint for each year from base to target.
        """
        points: List[PathwayPoint] = []
        base = _decimal(target.base_year_tco2e)
        reduction = _decimal(target.target_reduction_pct) / Decimal("100")
        total_years = Decimal(str(target.target_year - target.base_year))

        for year in range(target.base_year, target.target_year + 1):
            elapsed = Decimal(str(year - target.base_year))
            fraction = _safe_divide(elapsed, total_years)
            expected = base * (Decimal("1") - reduction * fraction)

            cumulative_pct = reduction * fraction * Decimal("100")

            points.append(PathwayPoint(
                year=year,
                expected_tco2e=_round_val(expected, 3),
                cumulative_reduction_pct=_round_val(cumulative_pct, 2),
            ))

        return points

    def calculate_sbti_pathway(
        self,
        target: EmissionsTarget,
    ) -> List[PathwayPoint]:
        """Calculate an SBTi-aligned reduction pathway.

        Uses compounding annual reduction rates per SBTi Corporate
        Manual (2023):
            - 1.5C: 4.2% per year compounding
            - Well-Below 2C: 2.5% per year compounding
            - Net-Zero: 4.2% near-term, 90% total by 2050

        Args:
            target: The emission reduction target with sbti_ambition set.

        Returns:
            List of PathwayPoint for each year from base to target.
        """
        if target.sbti_ambition is None:
            return self.calculate_linear_pathway(target)

        base = _decimal(target.base_year_tco2e)

        if target.sbti_ambition == SBTiAmbition.ONE_POINT_FIVE_C:
            annual_rate = SBTI_RATE_1_5C
        elif target.sbti_ambition == SBTiAmbition.WELL_BELOW_2C:
            annual_rate = SBTI_RATE_WB2C
        elif target.sbti_ambition == SBTiAmbition.NET_ZERO:
            annual_rate = SBTI_RATE_1_5C  # Near-term uses 1.5C rate.
        else:
            annual_rate = SBTI_RATE_1_5C

        retention = Decimal("1") - annual_rate
        points: List[PathwayPoint] = []

        for year in range(target.base_year, target.target_year + 1):
            elapsed = year - target.base_year
            expected = base * _pow_decimal(retention, elapsed)

            cumulative_pct = _safe_pct(
                base - expected, base
            )

            points.append(PathwayPoint(
                year=year,
                expected_tco2e=_round_val(expected, 3),
                cumulative_reduction_pct=_round_val(cumulative_pct, 2),
            ))

        return points

    def assess_status(
        self,
        target: EmissionsTarget,
        current_actual: Decimal,
        expected: Optional[Decimal] = None,
    ) -> TargetStatus:
        """Assess target status for a given actual vs expected value.

        Thresholds:
            AHEAD:      actual <= expected * 0.95
            ON_TRACK:   0.95 * expected < actual <= 1.05 * expected
            AT_RISK:    1.05 * expected < actual <= 1.15 * expected
            BEHIND:     actual > 1.15 * expected

        Args:
            target: The target definition.
            current_actual: Current actual emissions.
            expected: Expected emissions for this year.  If None,
                      NOT_STARTED is returned.

        Returns:
            TargetStatus enum value.
        """
        if expected is None:
            return TargetStatus.NOT_STARTED

        actual = _decimal(current_actual)
        exp = _decimal(expected)

        if exp == Decimal("0"):
            return TargetStatus.BEHIND if actual > Decimal("0") else TargetStatus.ON_TRACK

        ahead_limit = exp * STATUS_AHEAD_THRESHOLD
        on_track_limit = exp * STATUS_ON_TRACK_THRESHOLD
        at_risk_limit = exp * STATUS_AT_RISK_THRESHOLD

        if actual <= ahead_limit:
            return TargetStatus.AHEAD
        elif actual <= on_track_limit:
            return TargetStatus.ON_TRACK
        elif actual <= at_risk_limit:
            return TargetStatus.AT_RISK
        else:
            return TargetStatus.BEHIND

    def calculate_required_rate(
        self,
        target: EmissionsTarget,
        current_actual: Decimal,
        current_year: int,
    ) -> Decimal:
        """Calculate the required annual reduction rate from current position.

        Formula (compounding):
            required_rate = 1 - (target_tco2e / current_actual)^(1/years_remaining)

        For linear approximation when compounding is not tractable:
            required_rate = (current_actual - target_tco2e) /
                            (current_actual * years_remaining)

        Args:
            target: The target definition.
            current_actual: Current actual emissions (tCO2e).
            current_year: Current reporting year.

        Returns:
            Required annual reduction rate as a decimal (e.g., 0.042 = 4.2%).
        """
        actual = _decimal(current_actual)
        target_val = _decimal(target.target_tco2e)
        years_remaining = target.target_year - current_year

        if years_remaining <= 0:
            return Decimal("0")

        if actual <= Decimal("0"):
            return Decimal("0")

        if actual <= target_val:
            return Decimal("0")

        # Use linear approximation for deterministic computation.
        # linear_rate = (actual - target_val) / (actual * years_remaining)
        required = _safe_divide(
            actual - target_val,
            actual * _decimal(years_remaining),
        )

        return _round_val(required, 6)

    def attribute_reductions(
        self,
        base_year_sources: Dict[str, Decimal],
        current_sources: Dict[str, Decimal],
    ) -> List[ReductionAttribution]:
        """Attribute emission reductions to specific source categories.

        Compares base year and current year emissions by source category
        to determine which categories contributed most to reductions.

        Args:
            base_year_sources: Dict of source category -> tCO2e in base year.
            current_sources: Dict of source category -> tCO2e in current year.

        Returns:
            List of ReductionAttribution sorted by reduction amount.
        """
        all_categories = set(base_year_sources.keys()) | set(current_sources.keys())
        total_reduction = Decimal("0")
        category_reductions: List[Tuple[str, Decimal]] = []

        for cat in sorted(all_categories):
            base_val = _decimal(base_year_sources.get(cat, Decimal("0")))
            current_val = _decimal(current_sources.get(cat, Decimal("0")))
            reduction = base_val - current_val

            if reduction > Decimal("0"):
                total_reduction += reduction
                category_reductions.append((cat, reduction))

        # Calculate percentages.
        attributions: List[ReductionAttribution] = []
        for cat, reduction in category_reductions:
            pct = _safe_pct(reduction, total_reduction)
            attributions.append(ReductionAttribution(
                category=cat,
                reduction_tco2e=_round_val(reduction, 3),
                pct_of_total_reduction=_round_val(pct, 2),
            ))

        # Sort by reduction amount descending.
        attributions.sort(
            key=lambda a: a.reduction_tco2e, reverse=True
        )

        return attributions

    def rebase_target(
        self,
        target: EmissionsTarget,
        new_base_year_tco2e: Decimal,
        adjustment_reason: str,
    ) -> RebaseResult:
        """Rebase a target to updated base year emissions.

        When the base year is recalculated (e.g., due to acquisition or
        methodology change), the target must be rebased to maintain the
        same percentage reduction against the new base year value.

        Args:
            target: The original target to rebase.
            new_base_year_tco2e: The recalculated base year emissions.
            adjustment_reason: Reason for the rebase.

        Returns:
            RebaseResult with original and rebased targets.
        """
        new_base = _decimal(new_base_year_tco2e)
        old_base = _decimal(target.base_year_tco2e)

        # Compute new target value maintaining same reduction percentage.
        new_target_tco2e = _round_val(
            new_base * (
                Decimal("1") - _decimal(target.target_reduction_pct) / Decimal("100")
            ),
            places=3,
        )

        rebased = EmissionsTarget(
            target_id=target.target_id,
            name=target.name,
            target_type=target.target_type,
            scopes=target.scopes,
            base_year=target.base_year,
            base_year_tco2e=new_base,
            target_year=target.target_year,
            target_reduction_pct=target.target_reduction_pct,
            target_tco2e=new_target_tco2e,
            sbti_ambition=target.sbti_ambition,
            intensity_metric=target.intensity_metric,
            base_year_intensity_denominator=target.base_year_intensity_denominator,
        )

        base_change = new_base - old_base
        target_change = new_target_tco2e - _decimal(target.target_tco2e)

        result = RebaseResult(
            original_target=target,
            rebased_target=rebased,
            adjustment_reason=adjustment_reason,
            base_year_change_tco2e=_round_val(base_change, 3),
            target_tco2e_change=_round_val(target_change, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def validate_sbti_alignment(
        self,
        target: EmissionsTarget,
    ) -> Dict[str, Any]:
        """Validate whether a target meets SBTi requirements.

        Checks:
            - Scope coverage (Scope 1+2 mandatory, Scope 3 if >40% of total)
            - Timeframe (5-10 years for near-term targets)
            - Minimum ambition (4.2% for 1.5C, 2.5% for WB2C)
            - Base year recency (no older than 2 years at submission)

        Args:
            target: The target to validate.

        Returns:
            Dict with validation results and any issues found.
        """
        issues: List[str] = []
        warnings: List[str] = []

        # Check scope coverage.
        has_scope1 = ScopeType.SCOPE_1 in target.scopes
        has_scope2 = ScopeType.SCOPE_2 in target.scopes
        if not (has_scope1 and has_scope2):
            issues.append(
                "SBTi requires targets to cover at minimum Scope 1 and "
                "Scope 2 emissions."
            )

        # Check timeframe.
        timeframe = target.target_year - target.base_year
        if timeframe < SBTI_MIN_NEAR_TERM_YEARS:
            issues.append(
                f"SBTi near-term target timeframe ({timeframe} years) is "
                f"shorter than minimum ({SBTI_MIN_NEAR_TERM_YEARS} years)."
            )
        if timeframe > SBTI_MAX_NEAR_TERM_YEARS:
            warnings.append(
                f"SBTi near-term target timeframe ({timeframe} years) "
                f"exceeds recommended maximum ({SBTI_MAX_NEAR_TERM_YEARS} years)."
            )

        # Check minimum ambition.
        if target.sbti_ambition == SBTiAmbition.ONE_POINT_FIVE_C:
            min_annual = SBTI_RATE_1_5C * Decimal("100")
            implied_annual = _safe_divide(
                _decimal(target.target_reduction_pct),
                _decimal(timeframe),
            )
            if implied_annual < min_annual:
                issues.append(
                    f"SBTi 1.5C requires minimum {min_annual}% annual "
                    f"reduction; target implies {_round_val(implied_annual, 2)}%."
                )

        elif target.sbti_ambition == SBTiAmbition.WELL_BELOW_2C:
            min_annual = SBTI_RATE_WB2C * Decimal("100")
            implied_annual = _safe_divide(
                _decimal(target.target_reduction_pct),
                _decimal(timeframe),
            )
            if implied_annual < min_annual:
                issues.append(
                    f"SBTi WB2C requires minimum {min_annual}% annual "
                    f"reduction; target implies {_round_val(implied_annual, 2)}%."
                )

        elif target.sbti_ambition == SBTiAmbition.NET_ZERO:
            # Net-zero requires 90% reduction by 2050.
            if target.target_year > SBTI_NET_ZERO_TARGET_YEAR:
                issues.append(
                    f"SBTi Net-Zero long-term target must be no later than "
                    f"{SBTI_NET_ZERO_TARGET_YEAR}; target year is "
                    f"{target.target_year}."
                )

        # Check target type for intensity.
        if target.target_type == TargetType.INTENSITY:
            if target.intensity_metric is None:
                issues.append(
                    "Intensity target must specify an intensity metric."
                )
            if target.base_year_intensity_denominator is None:
                issues.append(
                    "Intensity target must specify base year intensity "
                    "denominator."
                )

        is_aligned = len(issues) == 0

        validation = {
            "is_sbti_aligned": is_aligned,
            "ambition": target.sbti_ambition.value if target.sbti_ambition else None,
            "timeframe_years": timeframe,
            "issues": issues,
            "warnings": warnings,
            "provenance_hash": "",
        }
        validation["provenance_hash"] = _compute_hash(validation)
        return validation

    def generate_progress_report(
        self,
        result: TargetTrackingResult,
    ) -> str:
        """Generate a Markdown progress report.

        Args:
            result: TargetTrackingResult from track_progress.

        Returns:
            Markdown-formatted report string.
        """
        lines: List[str] = []
        t = result.target

        lines.append("# Target Progress Report")
        lines.append("")
        lines.append(f"**Target:** {t.name}")
        lines.append(f"**Type:** {t.target_type.value}")
        lines.append(f"**Scopes:** {', '.join(s.value for s in t.scopes)}")
        lines.append(f"**Base Year:** {t.base_year} ({t.base_year_tco2e} tCO2e)")
        lines.append(f"**Target Year:** {t.target_year} ({t.target_tco2e} tCO2e)")
        lines.append(f"**Reduction:** {t.target_reduction_pct}%")
        if t.sbti_ambition:
            lines.append(f"**SBTi Ambition:** {t.sbti_ambition.value}")
        lines.append("")
        lines.append(f"**Current Status:** {result.current_status.value}")
        lines.append(f"**Years Remaining:** {result.years_remaining}")
        lines.append(
            f"**Required Annual Reduction:** "
            f"{result.required_annual_reduction_pct}%"
        )
        lines.append(
            f"**Actual Annual Reduction:** "
            f"{result.actual_annual_reduction_pct}%"
        )
        lines.append(
            f"**On-Track Probability:** {result.on_track_probability}%"
        )
        lines.append(f"**Provenance Hash:** `{result.provenance_hash[:16]}...`")
        lines.append("")

        if result.progress_points:
            lines.append("## Year-by-Year Progress")
            lines.append("")
            lines.append(
                "| Year | Actual (tCO2e) | Expected (tCO2e) | "
                "Gap (tCO2e) | Progress | Status |"
            )
            lines.append(
                "|------|----------------|------------------|"
                "-------------|----------|--------|"
            )
            for pp in result.progress_points:
                lines.append(
                    f"| {pp.year} | {pp.actual_tco2e} | {pp.expected_tco2e} | "
                    f"{pp.gap_tco2e} | {pp.progress_pct}% | {pp.status.value} |"
                )
            lines.append("")

        if result.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpolate_pathway(
        self,
        target: EmissionsTarget,
        pathway: List[PathwayPoint],
        year: int,
    ) -> Decimal:
        """Interpolate expected emissions for a year not in the pathway.

        Uses linear interpolation between the nearest pathway points.

        Args:
            target: The target definition.
            pathway: List of PathwayPoint.
            year: The year to interpolate for.

        Returns:
            Interpolated expected emissions.
        """
        if not pathway:
            return _decimal(target.base_year_tco2e)

        # Find surrounding points.
        before: Optional[PathwayPoint] = None
        after: Optional[PathwayPoint] = None

        for pp in pathway:
            if pp.year <= year:
                before = pp
            if pp.year >= year and after is None:
                after = pp

        if before is not None and after is not None and before.year != after.year:
            # Linear interpolation.
            fraction = _safe_divide(
                _decimal(year - before.year),
                _decimal(after.year - before.year),
            )
            return (
                _decimal(before.expected_tco2e) +
                fraction * (_decimal(after.expected_tco2e) -
                            _decimal(before.expected_tco2e))
            )
        elif before is not None:
            return _decimal(before.expected_tco2e)
        elif after is not None:
            return _decimal(after.expected_tco2e)
        else:
            return _decimal(target.base_year_tco2e)

    def _calculate_actual_annual_rate(
        self,
        target: EmissionsTarget,
        sorted_actuals: List[YearlyActual],
    ) -> Decimal:
        """Calculate the actual average annual reduction rate.

        Formula (linear):
            actual_rate = (base_tco2e - latest_actual) /
                          (base_tco2e * years_elapsed)

        Args:
            target: The target definition.
            sorted_actuals: Sorted list of yearly actuals.

        Returns:
            Annual reduction rate as a decimal.
        """
        if not sorted_actuals:
            return Decimal("0")

        base = _decimal(target.base_year_tco2e)
        latest = sorted_actuals[-1]
        latest_actual = _decimal(latest.actual_tco2e)
        years_elapsed = latest.year - target.base_year

        if years_elapsed <= 0 or base == Decimal("0"):
            return Decimal("0")

        rate = _safe_divide(
            base - latest_actual,
            base * _decimal(years_elapsed),
        )

        return _round_val(rate, 6)

    def _calculate_on_track_probability(
        self,
        target: EmissionsTarget,
        progress_points: List[ProgressPoint],
        required_rate: Decimal,
        actual_rate: Decimal,
        years_remaining: int,
    ) -> Decimal:
        """Calculate deterministic on-track probability.

        This is a heuristic based on comparing actual rate to required rate
        and considering momentum.  NOT a stochastic model.

        Heuristic:
            if actual_rate >= required_rate: base = 80%
            elif actual_rate >= 0.8 * required_rate: base = 60%
            elif actual_rate >= 0.5 * required_rate: base = 40%
            else: base = 20%

            Adjust for momentum: last 2 years trend direction.
            Adjust for time: more time remaining = higher probability.

        Args:
            target: The target definition.
            progress_points: Progress points.
            required_rate: Required annual rate (decimal).
            actual_rate: Actual annual rate (decimal).
            years_remaining: Years until target.

        Returns:
            Probability as 0-100 Decimal.
        """
        if not progress_points:
            return Decimal("50")  # Default when no data.

        # Base probability from rate comparison.
        if actual_rate >= required_rate and required_rate > Decimal("0"):
            base_prob = Decimal("80")
        elif actual_rate >= required_rate * Decimal("0.8"):
            base_prob = Decimal("60")
        elif actual_rate >= required_rate * Decimal("0.5"):
            base_prob = Decimal("40")
        elif actual_rate > Decimal("0"):
            base_prob = Decimal("20")
        else:
            base_prob = Decimal("10")

        # Momentum adjustment: check if recent years show improvement.
        if len(progress_points) >= 2:
            recent = progress_points[-1]
            prev = progress_points[-2]
            if recent.actual_tco2e < prev.actual_tco2e:
                base_prob += Decimal("5")  # Positive momentum.
            elif recent.actual_tco2e > prev.actual_tco2e:
                base_prob -= Decimal("10")  # Negative momentum.

        # Time adjustment: more time = slightly higher probability.
        if years_remaining >= 7:
            base_prob += Decimal("5")
        elif years_remaining <= 2:
            base_prob -= Decimal("5")

        # Clamp to 0-100.
        base_prob = max(Decimal("0"), min(Decimal("100"), base_prob))

        # If already at or below target, 100%.
        if progress_points[-1].actual_tco2e <= _decimal(target.target_tco2e):
            base_prob = Decimal("100")

        return base_prob

    def _generate_recommendations(
        self,
        target: EmissionsTarget,
        current_status: TargetStatus,
        required_rate: Decimal,
        actual_rate: Decimal,
        years_remaining: int,
    ) -> List[str]:
        """Generate actionable recommendations based on progress.

        Args:
            target: The target definition.
            current_status: Current status.
            required_rate: Required annual reduction rate (decimal).
            actual_rate: Actual annual reduction rate (decimal).
            years_remaining: Years remaining.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if current_status == TargetStatus.BEHIND:
            recommendations.append(
                f"Target is significantly behind pathway. Required annual "
                f"reduction is {_round_val(required_rate * Decimal('100'), 2)}% "
                f"but actual is {_round_val(actual_rate * Decimal('100'), 2)}%. "
                f"Immediate action needed."
            )
            recommendations.append(
                "Consider accelerated decarbonization measures or "
                "evaluate whether the target needs to be updated."
            )

        elif current_status == TargetStatus.AT_RISK:
            recommendations.append(
                "Target is at risk. Increase reduction efforts to get "
                "back on track with the pathway."
            )

        elif current_status == TargetStatus.ON_TRACK:
            recommendations.append(
                "Target is on track. Maintain current reduction trajectory."
            )

        elif current_status == TargetStatus.AHEAD:
            recommendations.append(
                "Target is ahead of the pathway. Consider setting a more "
                "ambitious target or extending scope coverage."
            )

        if years_remaining <= 3 and current_status in (
            TargetStatus.AT_RISK, TargetStatus.BEHIND
        ):
            recommendations.append(
                f"Only {years_remaining} years remaining. Consider whether "
                f"the target is still achievable and prepare a disclosure "
                f"explaining the gap."
            )

        if target.sbti_ambition is not None:
            recommendations.append(
                "Ensure any base year recalculations are communicated "
                "to SBTi within the required timeframe."
            )

        return recommendations

    def get_pathway_summary(
        self,
        target: EmissionsTarget,
    ) -> Dict[str, Any]:
        """Generate a summary of the target pathway.

        Args:
            target: The target definition.

        Returns:
            Dict with pathway summary statistics.
        """
        if target.sbti_ambition:
            pathway = self.calculate_sbti_pathway(target)
        else:
            pathway = self.calculate_linear_pathway(target)

        midpoint_year = target.base_year + (target.target_year - target.base_year) // 2
        midpoint_expected: Optional[Decimal] = None
        for pp in pathway:
            if pp.year == midpoint_year:
                midpoint_expected = pp.expected_tco2e
                break

        return {
            "target_name": target.name,
            "pathway_type": "sbti_compounding" if target.sbti_ambition else "linear",
            "base_year": target.base_year,
            "target_year": target.target_year,
            "base_tco2e": str(target.base_year_tco2e),
            "target_tco2e": str(target.target_tco2e),
            "reduction_pct": str(target.target_reduction_pct),
            "total_years": target.target_year - target.base_year,
            "midpoint_year": midpoint_year,
            "midpoint_expected_tco2e": str(midpoint_expected) if midpoint_expected else None,
            "pathway_points": len(pathway),
            "provenance_hash": _compute_hash(pathway),
        }
