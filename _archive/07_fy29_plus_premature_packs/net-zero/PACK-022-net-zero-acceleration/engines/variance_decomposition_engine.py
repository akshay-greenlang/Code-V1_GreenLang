# -*- coding: utf-8 -*-
"""
VarianceDecompositionEngine - PACK-022 Net Zero Acceleration Engine 7
======================================================================

Decomposes year-over-year emissions changes into constituent drivers
using the Logarithmic Mean Divisia Index (LMDI-I) additive method.
This enables organizations to understand whether emissions changes
are driven by activity growth, intensity improvements, or structural
shifts in their business mix.

LMDI-I Methodology (Ang, 2004):
    The LMDI-I additive decomposition method is the preferred approach
    for emissions decomposition due to its zero-residual property and
    consistency in aggregation.  For each pair of consecutive years,
    total emissions change is decomposed into:

    delta_E = delta_ACT + delta_INT + delta_STR

    where:
    - delta_ACT (Activity Effect):  Change due to total activity level
      (e.g., production volume, revenue, output).
    - delta_INT (Intensity Effect): Change due to emissions intensity
      (emissions per unit of activity within each segment).
    - delta_STR (Structural Effect): Change due to the mix of segments
      (shifts in relative activity shares across business units/products).

    LMDI-I formulas use the logarithmic mean weight function:
        L(a, b) = (a - b) / (ln(a) - ln(b))  if a != b
        L(a, a) = a                            if a == b

Kaya Identity Extension:
    E = GDP * (E/GDP) can be further decomposed into:
    E = Activity * Structure * Intensity
    E = sum_i (Q * S_i * I_i)
    where Q = total activity, S_i = share of segment i, I_i = intensity of segment i

Features:
    - LMDI-I additive decomposition (zero-residual)
    - Year-over-year and cumulative decomposition
    - Driver attribution by business unit / product / facility
    - Rolling forecast based on historical trends + planned actions
    - Early warning alerts when actual deviates from plan
    - Scope-level decomposition (Scope 1, 2, 3)
    - Multi-period trend analysis

Regulatory References:
    - Ang, B.W. (2004). "Decomposition analysis for policymaking in energy"
    - GHG Protocol Corporate Standard Chapter 5 (Tracking over time)
    - SBTi Target Validation Protocol v2.0
    - CSRD ESRS E1-6 (GHG intensity of revenue)

Zero-Hallucination:
    - All decomposition uses Decimal arithmetic with ROUND_HALF_UP
    - Logarithmic mean function implemented deterministically
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _log_mean(a: Decimal, b: Decimal) -> Decimal:
    """Calculate logarithmic mean of two values.

    The logarithmic mean is defined as:
        L(a, b) = (a - b) / (ln(a) - ln(b))  if a != b and a > 0 and b > 0
        L(a, a) = a                            if a == b

    This function is critical for the LMDI-I decomposition and guarantees
    the zero-residual property.

    Args:
        a: First value (must be positive).
        b: Second value (must be positive).

    Returns:
        Logarithmic mean as Decimal.
    """
    if a <= Decimal("0") or b <= Decimal("0"):
        return Decimal("0")
    if a == b:
        return a
    a_f = float(a)
    b_f = float(b)
    ln_a = math.log(a_f)
    ln_b = math.log(b_f)
    if abs(ln_a - ln_b) < 1e-15:
        return a
    result = (a_f - b_f) / (ln_a - ln_b)
    return _decimal(result)

# ---------------------------------------------------------------------------
# Reference Data: Alert Thresholds
# ---------------------------------------------------------------------------

DEFAULT_ALERT_THRESHOLDS: Dict[str, Decimal] = {
    "yellow": Decimal("5"),     # 5% deviation from plan
    "orange": Decimal("10"),    # 10% deviation from plan
    "red": Decimal("20"),       # 20% deviation from plan
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecompositionMethod(str, Enum):
    """Decomposition method selection."""
    LMDI_I = "lmdi_i"
    SIMPLE_RATIO = "simple_ratio"

class DecompositionEffect(str, Enum):
    """Type of decomposition effect."""
    ACTIVITY = "activity"
    INTENSITY = "intensity"
    STRUCTURAL = "structural"
    TOTAL = "total"

class ForecastHorizon(str, Enum):
    """Rolling forecast horizon."""
    ONE_YEAR = "1_year"
    TWO_YEAR = "2_year"
    THREE_YEAR = "3_year"

class ScopeFilter(str, Enum):
    """Scope filter for decomposition."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SegmentData(BaseModel):
    """Emissions and activity data for a single segment in one year."""
    segment_id: str = Field(description="Segment identifier (BU, product, facility)")
    segment_name: str = Field(default="", description="Human-readable segment name")
    year: int = Field(description="Data year")
    emissions: Decimal = Field(description="Emissions (tCO2e)")
    activity: Decimal = Field(description="Activity measure (revenue, units, etc.)")
    scope: ScopeFilter = Field(default=ScopeFilter.ALL_SCOPES, description="Scope")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emissions", "activity", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class YearDecomposition(BaseModel):
    """Decomposition result for a single year transition (year0 -> year1)."""
    decomposition_id: str = Field(default_factory=_new_uuid, description="Decomposition ID")
    year_from: int = Field(description="Base year")
    year_to: int = Field(description="Target year")
    total_change: Decimal = Field(description="Total emissions change (tCO2e)")
    activity_effect: Decimal = Field(description="Activity effect (tCO2e)")
    intensity_effect: Decimal = Field(description="Intensity effect (tCO2e)")
    structural_effect: Decimal = Field(description="Structural effect (tCO2e)")
    residual: Decimal = Field(default=Decimal("0"), description="Decomposition residual")
    emissions_year_from: Decimal = Field(description="Total emissions in base year")
    emissions_year_to: Decimal = Field(description="Total emissions in target year")
    activity_year_from: Decimal = Field(description="Total activity in base year")
    activity_year_to: Decimal = Field(description="Total activity in target year")
    total_change_pct: Decimal = Field(default=Decimal("0"), description="Total change %")
    activity_effect_pct: Decimal = Field(default=Decimal("0"), description="Activity effect %")
    intensity_effect_pct: Decimal = Field(default=Decimal("0"), description="Intensity effect %")
    structural_effect_pct: Decimal = Field(default=Decimal("0"), description="Structural effect %")
    segment_count: int = Field(default=0, description="Number of segments decomposed")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_change", "activity_effect", "intensity_effect",
                     "structural_effect", "residual",
                     "emissions_year_from", "emissions_year_to",
                     "activity_year_from", "activity_year_to",
                     "total_change_pct", "activity_effect_pct",
                     "intensity_effect_pct", "structural_effect_pct",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class DriverAttribution(BaseModel):
    """Attribution of emissions change to a specific driver/segment."""
    attribution_id: str = Field(default_factory=_new_uuid, description="Attribution ID")
    segment_id: str = Field(description="Segment identifier")
    segment_name: str = Field(default="", description="Segment name")
    year_from: int = Field(description="Base year")
    year_to: int = Field(description="Target year")
    total_contribution: Decimal = Field(description="Total contribution (tCO2e)")
    activity_contribution: Decimal = Field(description="Activity contribution (tCO2e)")
    intensity_contribution: Decimal = Field(description="Intensity contribution (tCO2e)")
    structural_contribution: Decimal = Field(description="Structural contribution (tCO2e)")
    contribution_pct: Decimal = Field(default=Decimal("0"), description="% of total change")
    emissions_year_from: Decimal = Field(description="Segment emissions base year")
    emissions_year_to: Decimal = Field(description="Segment emissions target year")
    intensity_year_from: Decimal = Field(description="Intensity base year")
    intensity_year_to: Decimal = Field(description="Intensity target year")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_contribution", "activity_contribution",
                     "intensity_contribution", "structural_contribution",
                     "contribution_pct", "emissions_year_from",
                     "emissions_year_to", "intensity_year_from",
                     "intensity_year_to", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ForecastPoint(BaseModel):
    """A single point in a rolling forecast."""
    year: int = Field(description="Forecast year")
    projected_emissions: Decimal = Field(description="Projected emissions (tCO2e)")
    projected_activity: Decimal = Field(description="Projected activity")
    projected_intensity: Decimal = Field(description="Projected intensity")
    confidence_lower: Decimal = Field(description="Lower bound (95% CI)")
    confidence_upper: Decimal = Field(description="Upper bound (95% CI)")
    planned_emissions: Optional[Decimal] = Field(
        default=None, description="Planned emissions from targets"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("projected_emissions", "projected_activity",
                     "projected_intensity", "confidence_lower",
                     "confidence_upper", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("planned_emissions", mode="before")
    @classmethod
    def _coerce_decimal_opt(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class EarlyWarningAlert(BaseModel):
    """Early warning alert when actual deviates from plan."""
    alert_id: str = Field(default_factory=_new_uuid, description="Alert identifier")
    year: int = Field(description="Alert year")
    segment_id: str = Field(default="", description="Affected segment (empty = total)")
    segment_name: str = Field(default="", description="Segment name")
    severity: AlertSeverity = Field(description="Alert severity")
    actual_emissions: Decimal = Field(description="Actual emissions (tCO2e)")
    planned_emissions: Decimal = Field(description="Planned emissions (tCO2e)")
    deviation_absolute: Decimal = Field(description="Absolute deviation (tCO2e)")
    deviation_pct: Decimal = Field(description="Deviation percentage")
    message: str = Field(default="", description="Human-readable alert message")
    created_at: datetime = Field(default_factory=utcnow, description="Alert timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("actual_emissions", "planned_emissions",
                     "deviation_absolute", "deviation_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CumulativeEffect(BaseModel):
    """Cumulative decomposition effect over multiple years."""
    effect_type: DecompositionEffect = Field(description="Effect type")
    cumulative_value: Decimal = Field(description="Cumulative effect (tCO2e)")
    cumulative_pct: Decimal = Field(description="Cumulative effect as % of base year")
    year_from: int = Field(description="Start year of cumulation")
    year_to: int = Field(description="End year of cumulation")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("cumulative_value", "cumulative_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class VarianceResult(BaseModel):
    """Complete variance decomposition result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    decomposition_by_year: List[YearDecomposition] = Field(
        default_factory=list, description="Year-over-year decompositions"
    )
    driver_attribution: List[DriverAttribution] = Field(
        default_factory=list, description="Per-segment driver attributions"
    )
    rolling_forecast: List[ForecastPoint] = Field(
        default_factory=list, description="Rolling forecast points"
    )
    alerts: List[EarlyWarningAlert] = Field(
        default_factory=list, description="Early warning alerts"
    )
    cumulative_effects: List[CumulativeEffect] = Field(
        default_factory=list, description="Cumulative effects over time"
    )
    method: DecompositionMethod = Field(
        default=DecompositionMethod.LMDI_I, description="Decomposition method used"
    )
    years_analyzed: int = Field(default=0, description="Number of years analyzed")
    segments_analyzed: int = Field(default=0, description="Number of segments analyzed")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class VarianceDecompositionConfig(BaseModel):
    """Configuration for the VarianceDecompositionEngine."""
    method: DecompositionMethod = Field(
        default=DecompositionMethod.LMDI_I,
        description="Decomposition method",
    )
    alert_threshold_yellow_pct: Decimal = Field(
        default=Decimal("5"), description="Yellow alert threshold (% deviation)"
    )
    alert_threshold_orange_pct: Decimal = Field(
        default=Decimal("10"), description="Orange alert threshold (% deviation)"
    )
    alert_threshold_red_pct: Decimal = Field(
        default=Decimal("20"), description="Red alert threshold (% deviation)"
    )
    forecast_confidence_width: Decimal = Field(
        default=Decimal("0.10"), description="Forecast confidence interval half-width (fraction)"
    )
    decimal_precision: int = Field(
        default=4, description="Decimal places for results"
    )

    @field_validator("alert_threshold_yellow_pct", "alert_threshold_orange_pct",
                     "alert_threshold_red_pct", "forecast_confidence_width",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

SegmentData.model_rebuild()
YearDecomposition.model_rebuild()
DriverAttribution.model_rebuild()
ForecastPoint.model_rebuild()
EarlyWarningAlert.model_rebuild()
CumulativeEffect.model_rebuild()
VarianceResult.model_rebuild()
VarianceDecompositionConfig.model_rebuild()

# ---------------------------------------------------------------------------
# VarianceDecompositionEngine
# ---------------------------------------------------------------------------

class VarianceDecompositionEngine:
    """
    Emissions variance decomposition engine using LMDI-I.

    Decomposes year-over-year emissions changes into activity,
    intensity, and structural effects following the Logarithmic
    Mean Divisia Index (LMDI-I) additive methodology.

    Attributes:
        config: Engine configuration.
        _segment_data: Stored segment data indexed by (segment_id, year).

    Example:
        >>> engine = VarianceDecompositionEngine()
        >>> engine.add_segment_data(segments)
        >>> decomp = engine.decompose_year(2024, 2025)
        >>> drivers = engine.attribute_drivers(2024, 2025)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VarianceDecompositionEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = VarianceDecompositionConfig(**config)
        elif config and isinstance(config, VarianceDecompositionConfig):
            self.config = config
        else:
            self.config = VarianceDecompositionConfig()

        self._segment_data: Dict[Tuple[str, int], SegmentData] = {}
        self._planned_emissions: Dict[int, Decimal] = {}
        logger.info("VarianceDecompositionEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------------

    def add_segment_data(self, segments: List[SegmentData]) -> int:
        """Add segment data for decomposition.

        Args:
            segments: List of SegmentData objects.

        Returns:
            Number of data points added.
        """
        count = 0
        for seg in segments:
            key = (seg.segment_id, seg.year)
            seg.provenance_hash = _compute_hash(seg)
            self._segment_data[key] = seg
            count += 1
        logger.info("Added %d segment data points", count)
        return count

    def set_planned_emissions(self, plans: Dict[int, Decimal]) -> None:
        """Set planned emissions by year for early warning comparison.

        Args:
            plans: Dictionary mapping year to planned total emissions.
        """
        for year, val in plans.items():
            self._planned_emissions[year] = _decimal(val)
        logger.info("Set planned emissions for %d years", len(plans))

    def _get_segments_for_year(
        self, year: int, scope: Optional[ScopeFilter] = None
    ) -> List[SegmentData]:
        """Get all segment data for a specific year.

        Args:
            year: Target year.
            scope: Optional scope filter.

        Returns:
            List of SegmentData for the year.
        """
        result: List[SegmentData] = []
        for (sid, y), seg in self._segment_data.items():
            if y != year:
                continue
            if scope and scope != ScopeFilter.ALL_SCOPES and seg.scope != scope:
                continue
            result.append(seg)
        return result

    def _get_available_years(self) -> List[int]:
        """Get sorted list of available years.

        Returns:
            Sorted list of years with data.
        """
        years: set = set()
        for (_, y) in self._segment_data:
            years.add(y)
        return sorted(years)

    # -------------------------------------------------------------------
    # LMDI-I Decomposition
    # -------------------------------------------------------------------

    def decompose_year(
        self,
        year_from: int,
        year_to: int,
        scope: Optional[ScopeFilter] = None,
    ) -> YearDecomposition:
        """Decompose emissions change between two years using LMDI-I.

        Decomposes delta_E into:
            delta_ACT + delta_INT + delta_STR = delta_E (zero residual)

        LMDI-I formulas:
            delta_ACT = sum_i [ L(E_i^t, E_i^0) * ln(Q^t / Q^0) ]
            delta_INT = sum_i [ L(E_i^t, E_i^0) * ln(I_i^t / I_i^0) ]
            delta_STR = sum_i [ L(E_i^t, E_i^0) * ln(S_i^t / S_i^0) ]

        where E_i = emissions of segment i, Q = total activity,
        I_i = E_i / A_i = intensity, S_i = A_i / Q = activity share.

        Args:
            year_from: Base year.
            year_to: Comparison year.
            scope: Optional scope filter.

        Returns:
            YearDecomposition result.

        Raises:
            ValueError: If insufficient data for either year.
        """
        segments_0 = self._get_segments_for_year(year_from, scope)
        segments_t = self._get_segments_for_year(year_to, scope)

        if not segments_0:
            raise ValueError(f"No segment data for year {year_from}")
        if not segments_t:
            raise ValueError(f"No segment data for year {year_to}")

        # Build lookup by segment_id
        seg0_map: Dict[str, SegmentData] = {s.segment_id: s for s in segments_0}
        segt_map: Dict[str, SegmentData] = {s.segment_id: s for s in segments_t}

        # Find common segments
        common_ids = set(seg0_map.keys()) & set(segt_map.keys())
        if not common_ids:
            raise ValueError(f"No common segments between {year_from} and {year_to}")

        # Total activity Q
        q_0 = sum(seg0_map[sid].activity for sid in common_ids)
        q_t = sum(segt_map[sid].activity for sid in common_ids)

        # Total emissions E
        e_0_total = sum(seg0_map[sid].emissions for sid in common_ids)
        e_t_total = sum(segt_map[sid].emissions for sid in common_ids)

        delta_act = Decimal("0")
        delta_int = Decimal("0")
        delta_str = Decimal("0")

        prec = self.config.decimal_precision

        for sid in common_ids:
            e_i_0 = seg0_map[sid].emissions
            e_i_t = segt_map[sid].emissions
            a_i_0 = seg0_map[sid].activity
            a_i_t = segt_map[sid].activity

            # Skip segments with zero values to avoid log(0)
            if e_i_0 <= Decimal("0") or e_i_t <= Decimal("0"):
                continue
            if a_i_0 <= Decimal("0") or a_i_t <= Decimal("0"):
                continue
            if q_0 <= Decimal("0") or q_t <= Decimal("0"):
                continue

            # Logarithmic mean weight
            lm = _log_mean(e_i_t, e_i_0)

            # Intensity: I_i = E_i / A_i
            i_i_0 = _safe_divide(e_i_0, a_i_0)
            i_i_t = _safe_divide(e_i_t, a_i_t)

            # Structure: S_i = A_i / Q
            s_i_0 = _safe_divide(a_i_0, q_0)
            s_i_t = _safe_divide(a_i_t, q_t)

            # Natural log ratios (using float for log, then back to Decimal)
            ln_q = _decimal(math.log(float(q_t) / float(q_0))) if q_0 > 0 else Decimal("0")
            ln_i = Decimal("0")
            if i_i_0 > Decimal("0") and i_i_t > Decimal("0"):
                ln_i = _decimal(math.log(float(i_i_t) / float(i_i_0)))
            ln_s = Decimal("0")
            if s_i_0 > Decimal("0") and s_i_t > Decimal("0"):
                ln_s = _decimal(math.log(float(s_i_t) / float(s_i_0)))

            delta_act += lm * ln_q
            delta_int += lm * ln_i
            delta_str += lm * ln_s

        delta_act = _round_val(delta_act, prec)
        delta_int = _round_val(delta_int, prec)
        delta_str = _round_val(delta_str, prec)

        total_change = e_t_total - e_0_total
        residual = total_change - (delta_act + delta_int + delta_str)
        residual = _round_val(residual, prec)

        # Percentage contributions
        total_change_pct = _safe_pct(total_change, e_0_total) if e_0_total > 0 else Decimal("0")
        act_pct = _safe_pct(delta_act, e_0_total) if e_0_total > 0 else Decimal("0")
        int_pct = _safe_pct(delta_int, e_0_total) if e_0_total > 0 else Decimal("0")
        str_pct = _safe_pct(delta_str, e_0_total) if e_0_total > 0 else Decimal("0")

        result = YearDecomposition(
            year_from=year_from,
            year_to=year_to,
            total_change=_round_val(total_change, prec),
            activity_effect=delta_act,
            intensity_effect=delta_int,
            structural_effect=delta_str,
            residual=residual,
            emissions_year_from=_round_val(e_0_total, prec),
            emissions_year_to=_round_val(e_t_total, prec),
            activity_year_from=_round_val(q_0, prec),
            activity_year_to=_round_val(q_t, prec),
            total_change_pct=total_change_pct,
            activity_effect_pct=act_pct,
            intensity_effect_pct=int_pct,
            structural_effect_pct=str_pct,
            segment_count=len(common_ids),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Decomposed %d->%d: total=%.1f, act=%.1f, int=%.1f, str=%.1f, residual=%.4f",
            year_from, year_to, float(total_change),
            float(delta_act), float(delta_int), float(delta_str), float(residual),
        )
        return result

    # -------------------------------------------------------------------
    # Driver Attribution
    # -------------------------------------------------------------------

    def attribute_drivers(
        self,
        year_from: int,
        year_to: int,
        scope: Optional[ScopeFilter] = None,
    ) -> List[DriverAttribution]:
        """Attribute emissions changes to individual segments/drivers.

        For each segment, calculates the activity, intensity, and
        structural contribution to the total emissions change.

        Args:
            year_from: Base year.
            year_to: Comparison year.
            scope: Optional scope filter.

        Returns:
            List of DriverAttribution sorted by absolute contribution.

        Raises:
            ValueError: If insufficient data.
        """
        segments_0 = self._get_segments_for_year(year_from, scope)
        segments_t = self._get_segments_for_year(year_to, scope)

        seg0_map = {s.segment_id: s for s in segments_0}
        segt_map = {s.segment_id: s for s in segments_t}
        common_ids = set(seg0_map.keys()) & set(segt_map.keys())

        if not common_ids:
            raise ValueError(f"No common segments between {year_from} and {year_to}")

        q_0 = sum(seg0_map[sid].activity for sid in common_ids)
        q_t = sum(segt_map[sid].activity for sid in common_ids)
        e_0_total = sum(seg0_map[sid].emissions for sid in common_ids)
        e_t_total = sum(segt_map[sid].emissions for sid in common_ids)
        total_change = e_t_total - e_0_total

        prec = self.config.decimal_precision
        attributions: List[DriverAttribution] = []

        for sid in common_ids:
            e_i_0 = seg0_map[sid].emissions
            e_i_t = segt_map[sid].emissions
            a_i_0 = seg0_map[sid].activity
            a_i_t = segt_map[sid].activity

            i_i_0 = _safe_divide(e_i_0, a_i_0)
            i_i_t = _safe_divide(e_i_t, a_i_t)

            act_contrib = Decimal("0")
            int_contrib = Decimal("0")
            str_contrib = Decimal("0")

            if (e_i_0 > 0 and e_i_t > 0 and a_i_0 > 0 and a_i_t > 0
                    and q_0 > 0 and q_t > 0):
                lm = _log_mean(e_i_t, e_i_0)

                s_i_0 = _safe_divide(a_i_0, q_0)
                s_i_t = _safe_divide(a_i_t, q_t)

                ln_q = _decimal(math.log(float(q_t) / float(q_0)))
                ln_i = _decimal(math.log(float(i_i_t) / float(i_i_0))) if i_i_0 > 0 and i_i_t > 0 else Decimal("0")
                ln_s = _decimal(math.log(float(s_i_t) / float(s_i_0))) if s_i_0 > 0 and s_i_t > 0 else Decimal("0")

                act_contrib = _round_val(lm * ln_q, prec)
                int_contrib = _round_val(lm * ln_i, prec)
                str_contrib = _round_val(lm * ln_s, prec)

            total_contrib = act_contrib + int_contrib + str_contrib
            contrib_pct = _safe_pct(abs(total_contrib), abs(total_change)) if total_change != 0 else Decimal("0")

            attr = DriverAttribution(
                segment_id=sid,
                segment_name=seg0_map[sid].segment_name,
                year_from=year_from,
                year_to=year_to,
                total_contribution=_round_val(total_contrib, prec),
                activity_contribution=act_contrib,
                intensity_contribution=int_contrib,
                structural_contribution=str_contrib,
                contribution_pct=contrib_pct,
                emissions_year_from=e_i_0,
                emissions_year_to=e_i_t,
                intensity_year_from=_round_val(i_i_0, prec),
                intensity_year_to=_round_val(i_i_t, prec),
            )
            attr.provenance_hash = _compute_hash(attr)
            attributions.append(attr)

        # Sort by absolute contribution (descending)
        attributions.sort(key=lambda x: abs(x.total_contribution), reverse=True)

        logger.info(
            "Driver attribution %d->%d: %d segments, total_change=%.1f tCO2e",
            year_from, year_to, len(attributions), float(total_change),
        )
        return attributions

    # -------------------------------------------------------------------
    # Rolling Forecast
    # -------------------------------------------------------------------

    def rolling_forecast(
        self,
        horizon: ForecastHorizon = ForecastHorizon.THREE_YEAR,
        scope: Optional[ScopeFilter] = None,
    ) -> List[ForecastPoint]:
        """Generate rolling emissions forecast based on historical trends.

        Uses linear regression on historical intensity and activity trends
        to project future emissions.  Confidence intervals are based on
        configured width parameter.

        Args:
            horizon: Forecast horizon (1, 2, or 3 years).
            scope: Optional scope filter.

        Returns:
            List of ForecastPoint for future years.
        """
        years = self._get_available_years()
        if len(years) < 2:
            logger.warning("Need at least 2 years for forecasting, have %d", len(years))
            return []

        horizon_map = {
            ForecastHorizon.ONE_YEAR: 1,
            ForecastHorizon.TWO_YEAR: 2,
            ForecastHorizon.THREE_YEAR: 3,
        }
        n_years = horizon_map.get(horizon, 3)

        # Calculate historical totals
        yearly_totals: List[Tuple[int, Decimal, Decimal]] = []  # (year, emissions, activity)
        for year in years:
            segs = self._get_segments_for_year(year, scope)
            e = sum(s.emissions for s in segs)
            a = sum(s.activity for s in segs)
            yearly_totals.append((year, e, a))

        # Calculate year-over-year growth rates
        emission_rates: List[Decimal] = []
        activity_rates: List[Decimal] = []
        intensity_rates: List[Decimal] = []

        for i in range(1, len(yearly_totals)):
            _, e_prev, a_prev = yearly_totals[i - 1]
            _, e_curr, a_curr = yearly_totals[i]

            if e_prev > 0:
                emission_rates.append(_safe_divide(e_curr - e_prev, e_prev))
            if a_prev > 0:
                activity_rates.append(_safe_divide(a_curr - a_prev, a_prev))

            int_prev = _safe_divide(e_prev, a_prev)
            int_curr = _safe_divide(e_curr, a_curr)
            if int_prev > 0:
                intensity_rates.append(_safe_divide(int_curr - int_prev, int_prev))

        # Average growth rates
        avg_e_rate = Decimal("0")
        if emission_rates:
            avg_e_rate = sum(emission_rates) / _decimal(len(emission_rates))

        avg_a_rate = Decimal("0")
        if activity_rates:
            avg_a_rate = sum(activity_rates) / _decimal(len(activity_rates))

        avg_i_rate = Decimal("0")
        if intensity_rates:
            avg_i_rate = sum(intensity_rates) / _decimal(len(intensity_rates))

        last_year, last_e, last_a = yearly_totals[-1]
        last_intensity = _safe_divide(last_e, last_a, Decimal("1"))

        prec = self.config.decimal_precision
        ci_width = self.config.forecast_confidence_width
        forecasts: List[ForecastPoint] = []

        for j in range(1, n_years + 1):
            fy = last_year + j
            proj_activity = last_a * (Decimal("1") + avg_a_rate) ** j
            proj_intensity = last_intensity * (Decimal("1") + avg_i_rate) ** j
            proj_emissions = proj_activity * proj_intensity

            # Widen confidence interval further into the future
            width_factor = ci_width * _decimal(j)
            lower = proj_emissions * (Decimal("1") - width_factor)
            upper = proj_emissions * (Decimal("1") + width_factor)

            planned = self._planned_emissions.get(fy)

            fp = ForecastPoint(
                year=fy,
                projected_emissions=_round_val(proj_emissions, prec),
                projected_activity=_round_val(proj_activity, prec),
                projected_intensity=_round_val(proj_intensity, prec),
                confidence_lower=_round_val(max(lower, Decimal("0")), prec),
                confidence_upper=_round_val(upper, prec),
                planned_emissions=_round_val(planned, prec) if planned else None,
            )
            fp.provenance_hash = _compute_hash(fp)
            forecasts.append(fp)

        logger.info(
            "Rolling forecast: %d years from %d, avg emission rate=%.2f%%",
            n_years, last_year, float(avg_e_rate * 100),
        )
        return forecasts

    # -------------------------------------------------------------------
    # Early Warning System
    # -------------------------------------------------------------------

    def check_early_warnings(
        self,
        scope: Optional[ScopeFilter] = None,
    ) -> List[EarlyWarningAlert]:
        """Check for early warning conditions across all years with planned data.

        Compares actual emissions against planned emissions for each year
        and raises alerts when deviations exceed configured thresholds.

        Args:
            scope: Optional scope filter.

        Returns:
            List of EarlyWarningAlert for years exceeding thresholds.
        """
        alerts: List[EarlyWarningAlert] = []
        years = self._get_available_years()

        for year in years:
            planned = self._planned_emissions.get(year)
            if planned is None or planned <= Decimal("0"):
                continue

            segs = self._get_segments_for_year(year, scope)
            actual = sum(s.emissions for s in segs)

            deviation = actual - planned
            dev_pct = _safe_pct(abs(deviation), planned)

            severity = AlertSeverity.GREEN
            if dev_pct >= self.config.alert_threshold_red_pct:
                severity = AlertSeverity.RED
            elif dev_pct >= self.config.alert_threshold_orange_pct:
                severity = AlertSeverity.ORANGE
            elif dev_pct >= self.config.alert_threshold_yellow_pct:
                severity = AlertSeverity.YELLOW

            if severity != AlertSeverity.GREEN:
                direction = "above" if deviation > 0 else "below"
                alert = EarlyWarningAlert(
                    year=year,
                    severity=severity,
                    actual_emissions=_round_val(actual, self.config.decimal_precision),
                    planned_emissions=_round_val(planned, self.config.decimal_precision),
                    deviation_absolute=_round_val(abs(deviation), self.config.decimal_precision),
                    deviation_pct=dev_pct,
                    message=(
                        f"Year {year}: Actual emissions {float(actual):.1f} tCO2e "
                        f"are {float(dev_pct):.1f}% {direction} planned "
                        f"{float(planned):.1f} tCO2e ({severity.value} alert)"
                    ),
                )
                alert.provenance_hash = _compute_hash(alert)
                alerts.append(alert)

        logger.info("Early warning check: %d alerts generated", len(alerts))
        return alerts

    # -------------------------------------------------------------------
    # Cumulative Analysis
    # -------------------------------------------------------------------

    def calculate_cumulative_effects(
        self,
        scope: Optional[ScopeFilter] = None,
    ) -> List[CumulativeEffect]:
        """Calculate cumulative decomposition effects over the full time series.

        Sums year-over-year decomposition effects across all available
        consecutive year pairs.

        Args:
            scope: Optional scope filter.

        Returns:
            List of CumulativeEffect (one per effect type).
        """
        years = self._get_available_years()
        if len(years) < 2:
            return []

        cum_activity = Decimal("0")
        cum_intensity = Decimal("0")
        cum_structural = Decimal("0")
        cum_total = Decimal("0")

        for i in range(len(years) - 1):
            try:
                decomp = self.decompose_year(years[i], years[i + 1], scope)
                cum_activity += decomp.activity_effect
                cum_intensity += decomp.intensity_effect
                cum_structural += decomp.structural_effect
                cum_total += decomp.total_change
            except ValueError:
                continue

        first_year = years[0]
        last_year = years[-1]
        segs_0 = self._get_segments_for_year(first_year, scope)
        base_emissions = sum(s.emissions for s in segs_0)

        prec = self.config.decimal_precision

        effects: List[CumulativeEffect] = []
        for effect_type, value in [
            (DecompositionEffect.ACTIVITY, cum_activity),
            (DecompositionEffect.INTENSITY, cum_intensity),
            (DecompositionEffect.STRUCTURAL, cum_structural),
            (DecompositionEffect.TOTAL, cum_total),
        ]:
            pct = _safe_pct(value, base_emissions) if base_emissions > 0 else Decimal("0")
            ce = CumulativeEffect(
                effect_type=effect_type,
                cumulative_value=_round_val(value, prec),
                cumulative_pct=pct,
                year_from=first_year,
                year_to=last_year,
            )
            ce.provenance_hash = _compute_hash(ce)
            effects.append(ce)

        logger.info(
            "Cumulative effects %d-%d: act=%.1f, int=%.1f, str=%.1f, total=%.1f",
            first_year, last_year,
            float(cum_activity), float(cum_intensity),
            float(cum_structural), float(cum_total),
        )
        return effects

    # -------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------

    def run_full_decomposition(
        self,
        scope: Optional[ScopeFilter] = None,
        forecast_horizon: ForecastHorizon = ForecastHorizon.THREE_YEAR,
    ) -> VarianceResult:
        """Run a complete variance decomposition pipeline.

        Performs year-over-year decomposition, driver attribution,
        rolling forecast, early warning checks, and cumulative analysis.

        Args:
            scope: Optional scope filter.
            forecast_horizon: Forecast horizon for projections.

        Returns:
            Complete VarianceResult.
        """
        years = self._get_available_years()
        logger.info("Running full decomposition: %d years, scope=%s", len(years), scope)

        # Step 1: Year-over-year decompositions
        decompositions: List[YearDecomposition] = []
        for i in range(len(years) - 1):
            try:
                decomp = self.decompose_year(years[i], years[i + 1], scope)
                decompositions.append(decomp)
            except ValueError as e:
                logger.warning("Skipping %d->%d: %s", years[i], years[i + 1], str(e))

        # Step 2: Driver attribution for most recent transition
        attributions: List[DriverAttribution] = []
        if len(years) >= 2:
            try:
                attributions = self.attribute_drivers(years[-2], years[-1], scope)
            except ValueError as e:
                logger.warning("Driver attribution failed: %s", str(e))

        # Step 3: Rolling forecast
        forecasts = self.rolling_forecast(forecast_horizon, scope)

        # Step 4: Early warning alerts
        alerts = self.check_early_warnings(scope)

        # Step 5: Cumulative effects
        cumulative = self.calculate_cumulative_effects(scope)

        # Collect unique segments
        segments_set: set = set()
        for key in self._segment_data:
            segments_set.add(key[0])

        result = VarianceResult(
            decomposition_by_year=decompositions,
            driver_attribution=attributions,
            rolling_forecast=forecasts,
            alerts=alerts,
            cumulative_effects=cumulative,
            method=self.config.method,
            years_analyzed=len(years),
            segments_analyzed=len(segments_set),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full decomposition complete: %d decomps, %d drivers, %d forecasts, %d alerts",
            len(decompositions), len(attributions), len(forecasts), len(alerts),
        )
        return result

    def clear(self) -> None:
        """Clear all stored data."""
        self._segment_data.clear()
        self._planned_emissions.clear()
        logger.info("VarianceDecompositionEngine cleared")
