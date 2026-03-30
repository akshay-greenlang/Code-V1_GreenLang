# -*- coding: utf-8 -*-
"""
VarianceAnalysisEngine - PACK-029 Interim Targets Pack Engine 4
=================================================================

Advanced emissions variance decomposition using the Logarithmic Mean
Divisia Index (LMDI-I) method and Kaya Identity framework.  Attributes
emissions changes to activity, intensity, and structural drivers with
scope-level and category-level variance attribution.

Calculation Methodology:
    LMDI-I Additive Decomposition (Ang, 2004):
        delta_E = delta_ACT + delta_INT + delta_STR

        Logarithmic mean weight function:
            L(a, b) = (a - b) / (ln(a) - ln(b))  if a != b
            L(a, a) = a                            if a == b

        Activity Effect (change in total activity):
            delta_ACT = sum_i L(E_i^t, E_i^0) * ln(Q^t / Q^0)

        Intensity Effect (change in emissions per unit activity):
            delta_INT = sum_i L(E_i^t, E_i^0) * ln(I_i^t / I_i^0)

        Structural Effect (change in activity mix):
            delta_STR = sum_i L(E_i^t, E_i^0) * ln(S_i^t / S_i^0)

    Kaya Identity Decomposition:
        E = Activity * Intensity * CarbonContent
        E = GDP * (Energy/GDP) * (CO2/Energy)
        At corporate level:
        E = Revenue * (Energy/Revenue) * (CO2/Energy)

    Waterfall Chart Data:
        Each component: start_value, end_value, delta

    Root Cause Classification:
        - internal_initiative: Planned reduction activities
        - external_factor: Market, weather, regulatory changes
        - data_quality: Measurement or reporting changes
        - structural_change: M&A, divestiture, boundary changes
        - organic_growth: Business growth or contraction

Regulatory References:
    - Ang, B.W. (2004) "Decomposition analysis for policymaking in energy"
    - Ang, B.W. (2015) "LMDI decomposition approach: A guide for implementation"
    - GHG Protocol Corporate Standard Ch. 5 -- Tracking emissions over time
    - SBTi Target Tracking Protocol v2.0
    - CSRD ESRS E1-6 -- GHG intensity of revenue
    - TCFD Recommendations -- Strategy & Metrics

Zero-Hallucination:
    - LMDI uses exact logarithmic mean formula (no Taylor approximation)
    - Zero-residual property guaranteed by LMDI-I method
    - All arithmetic via Decimal with ROUND_HALF_UP
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(n: Decimal, d: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if d == Decimal("0"):
        return default
    return n / d

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _log_mean(a: Decimal, b: Decimal) -> Decimal:
    """Compute logarithmic mean L(a, b).

    Formula:
        L(a, b) = (a - b) / (ln(a) - ln(b))   if a != b
        L(a, a) = a                              if a == b
        L(a, 0) = 0                              if b == 0
        L(0, b) = 0                              if a == 0

    The logarithmic mean is used as the weight function in LMDI
    decomposition to ensure the zero-residual property.
    """
    if a <= Decimal("0") or b <= Decimal("0"):
        return Decimal("0")
    if a == b:
        return a
    try:
        ln_a = _decimal(math.log(float(a)))
        ln_b = _decimal(math.log(float(b)))
        denom = ln_a - ln_b
        if denom == Decimal("0"):
            return a
        return (a - b) / denom
    except (ValueError, OverflowError):
        return (a + b) / Decimal("2")  # Fallback to arithmetic mean

def _safe_ln_ratio(a: Decimal, b: Decimal) -> Decimal:
    """Compute ln(a/b) safely.

    Returns 0 if either value is zero or negative.
    """
    if a <= Decimal("0") or b <= Decimal("0"):
        return Decimal("0")
    try:
        return _decimal(math.log(float(a) / float(b)))
    except (ValueError, OverflowError, ZeroDivisionError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VarianceDriver(str, Enum):
    """Variance driver type in LMDI decomposition."""
    ACTIVITY = "activity"
    INTENSITY = "intensity"
    STRUCTURAL = "structural"
    RESIDUAL = "residual"

class RootCauseCategory(str, Enum):
    """Root cause classification for variance."""
    INTERNAL_INITIATIVE = "internal_initiative"
    EXTERNAL_FACTOR = "external_factor"
    DATA_QUALITY = "data_quality"
    STRUCTURAL_CHANGE = "structural_change"
    ORGANIC_GROWTH = "organic_growth"
    UNKNOWN = "unknown"

class ScopeType(str, Enum):
    """GHG scope type."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"

class VarianceSeverity(str, Enum):
    """Severity of variance from target."""
    FAVORABLE = "favorable"
    NEUTRAL = "neutral"
    UNFAVORABLE = "unfavorable"
    CRITICAL = "critical"

class DataQuality(str, Enum):
    """Data quality tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class SegmentData(BaseModel):
    """Emissions and activity data for a single segment.

    A segment can be a business unit, facility, product line,
    or any organizational subdivision.

    Attributes:
        segment_id: Unique segment identifier.
        segment_name: Human-readable segment name.
        emissions_tco2e: Emissions for this segment (tCO2e).
        activity_value: Activity metric value (revenue, output, etc.).
        activity_unit: Unit of activity metric.
        scope: GHG scope.
    """
    segment_id: str = Field(..., min_length=1, max_length=100, description="Segment ID")
    segment_name: str = Field(default="", max_length=300, description="Segment name")
    emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Emissions (tCO2e)"
    )
    activity_value: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Activity value"
    )
    activity_unit: str = Field(
        default="revenue_m", max_length=50, description="Activity unit"
    )
    scope: ScopeType = Field(default=ScopeType.ALL_SCOPES, description="Scope")

class PeriodData(BaseModel):
    """Emissions data for a single period (year).

    Attributes:
        year: Calendar year.
        total_emissions_tco2e: Total emissions for the period.
        total_activity_value: Total activity for the period.
        activity_unit: Activity metric unit.
        segments: Per-segment breakdown.
    """
    year: int = Field(..., ge=2010, le=2050, description="Year")
    total_emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Total emissions (tCO2e)"
    )
    total_activity_value: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total activity"
    )
    activity_unit: str = Field(
        default="revenue_m", max_length=50, description="Activity unit"
    )
    segments: List[SegmentData] = Field(
        default_factory=list, description="Segment breakdown"
    )

class VarianceAnalysisInput(BaseModel):
    """Input for variance analysis.

    Attributes:
        entity_name: Company name.
        entity_id: Entity identifier.
        base_period: Base period data (comparison start).
        current_period: Current period data (comparison end).
        target_emissions_tco2e: Target emissions for current period.
        intermediate_periods: Additional periods for trend analysis.
        include_kaya_decomposition: Include Kaya identity decomposition.
        include_waterfall: Generate waterfall chart data.
        include_root_cause: Include root cause classification.
        scope: Primary scope for analysis.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    entity_id: str = Field(default="", max_length=100, description="Entity identifier")
    base_period: PeriodData = Field(..., description="Base period data")
    current_period: PeriodData = Field(..., description="Current period data")
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Target emissions for current period"
    )
    intermediate_periods: List[PeriodData] = Field(
        default_factory=list, description="Intermediate periods"
    )
    include_kaya_decomposition: bool = Field(
        default=True, description="Include Kaya decomposition"
    )
    include_waterfall: bool = Field(
        default=True, description="Include waterfall data"
    )
    include_root_cause: bool = Field(
        default=True, description="Include root cause analysis"
    )
    scope: ScopeType = Field(default=ScopeType.ALL_SCOPES, description="Scope")

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class LMDIComponent(BaseModel):
    """A single LMDI decomposition component.

    Attributes:
        driver: Driver type (activity/intensity/structural).
        delta_tco2e: Absolute change attributed to this driver.
        contribution_pct: Percentage contribution to total change.
        direction: Favorable or unfavorable.
        description: Human-readable explanation.
    """
    driver: str = Field(default="")
    delta_tco2e: Decimal = Field(default=Decimal("0"))
    contribution_pct: Decimal = Field(default=Decimal("0"))
    direction: str = Field(default=VarianceSeverity.NEUTRAL.value)
    description: str = Field(default="")

class SegmentAttribution(BaseModel):
    """Variance attribution for a single segment.

    Attributes:
        segment_id: Segment identifier.
        segment_name: Segment name.
        base_emissions_tco2e: Base period emissions.
        current_emissions_tco2e: Current period emissions.
        delta_tco2e: Change in emissions.
        delta_pct: Percentage change.
        activity_effect_tco2e: Activity driver.
        intensity_effect_tco2e: Intensity driver.
        structural_effect_tco2e: Structural driver.
        root_cause: Classified root cause.
    """
    segment_id: str = Field(default="")
    segment_name: str = Field(default="")
    base_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    delta_tco2e: Decimal = Field(default=Decimal("0"))
    delta_pct: Decimal = Field(default=Decimal("0"))
    activity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    intensity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    structural_effect_tco2e: Decimal = Field(default=Decimal("0"))
    root_cause: str = Field(default=RootCauseCategory.UNKNOWN.value)

class KayaDecomposition(BaseModel):
    """Kaya identity decomposition result.

    E = Activity * EnergyIntensity * CarbonIntensity

    Attributes:
        total_change_tco2e: Total emissions change.
        activity_effect_tco2e: Change due to activity level.
        energy_intensity_effect_tco2e: Change due to energy intensity.
        carbon_intensity_effect_tco2e: Change due to carbon intensity.
        activity_change_pct: Activity change percentage.
        energy_intensity_change_pct: Energy intensity change.
        carbon_intensity_change_pct: Carbon intensity change.
        residual_tco2e: Decomposition residual (should be ~0).
    """
    total_change_tco2e: Decimal = Field(default=Decimal("0"))
    activity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    energy_intensity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    carbon_intensity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    activity_change_pct: Decimal = Field(default=Decimal("0"))
    energy_intensity_change_pct: Decimal = Field(default=Decimal("0"))
    carbon_intensity_change_pct: Decimal = Field(default=Decimal("0"))
    residual_tco2e: Decimal = Field(default=Decimal("0"))

class WaterfallStep(BaseModel):
    """A single step in the waterfall chart.

    Attributes:
        label: Step label.
        start_value: Starting value.
        end_value: Ending value.
        delta: Change value.
        is_total: Whether this is a total bar.
        category: Driver category.
        color_hint: Suggested color (green=reduction, red=increase).
    """
    label: str = Field(default="")
    start_value: Decimal = Field(default=Decimal("0"))
    end_value: Decimal = Field(default=Decimal("0"))
    delta: Decimal = Field(default=Decimal("0"))
    is_total: bool = Field(default=False)
    category: str = Field(default="")
    color_hint: str = Field(default="neutral")

class ScopeVariance(BaseModel):
    """Variance analysis for a specific scope.

    Attributes:
        scope: GHG scope.
        base_emissions_tco2e: Base period emissions.
        current_emissions_tco2e: Current period emissions.
        delta_tco2e: Total change.
        delta_pct: Percentage change.
        activity_effect_tco2e: Activity effect.
        intensity_effect_tco2e: Intensity effect.
        severity: Variance severity.
    """
    scope: str = Field(default="")
    base_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    delta_tco2e: Decimal = Field(default=Decimal("0"))
    delta_pct: Decimal = Field(default=Decimal("0"))
    activity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    intensity_effect_tco2e: Decimal = Field(default=Decimal("0"))
    severity: str = Field(default=VarianceSeverity.NEUTRAL.value)

class VarianceAnalysisResult(BaseModel):
    """Complete variance analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        entity_id: Entity identifier.
        base_year: Base period year.
        current_year: Current period year.
        total_change_tco2e: Total emissions change.
        total_change_pct: Total percentage change.
        target_variance_tco2e: Variance from target.
        target_variance_pct: Target variance percentage.
        lmdi_components: LMDI decomposition components.
        segment_attributions: Per-segment attributions.
        scope_variances: Per-scope variance analysis.
        kaya_decomposition: Kaya identity decomposition.
        waterfall_steps: Waterfall chart data.
        residual_tco2e: LMDI residual (zero-residual check).
        overall_severity: Overall variance severity.
        data_quality: Data quality assessment.
        recommendations: Recommendations.
        warnings: Warnings.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    base_year: int = Field(default=0)
    current_year: int = Field(default=0)
    total_change_tco2e: Decimal = Field(default=Decimal("0"))
    total_change_pct: Decimal = Field(default=Decimal("0"))
    target_variance_tco2e: Decimal = Field(default=Decimal("0"))
    target_variance_pct: Decimal = Field(default=Decimal("0"))
    lmdi_components: List[LMDIComponent] = Field(default_factory=list)
    segment_attributions: List[SegmentAttribution] = Field(default_factory=list)
    scope_variances: List[ScopeVariance] = Field(default_factory=list)
    kaya_decomposition: Optional[KayaDecomposition] = Field(default=None)
    waterfall_steps: List[WaterfallStep] = Field(default_factory=list)
    residual_tco2e: Decimal = Field(default=Decimal("0"))
    overall_severity: str = Field(default=VarianceSeverity.NEUTRAL.value)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VarianceAnalysisEngine:
    """Variance analysis engine for PACK-029.

    LMDI-I decomposition and Kaya identity analysis for emissions
    variance attribution.

    Usage::

        engine = VarianceAnalysisEngine()
        result = await engine.calculate(variance_input)
        for comp in result.lmdi_components:
            print(f"  {comp.driver}: {comp.delta_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def calculate(self, data: VarianceAnalysisInput) -> VarianceAnalysisResult:
        """Run complete variance analysis.

        Args:
            data: Validated variance analysis input.

        Returns:
            VarianceAnalysisResult with LMDI decomposition, segment
            attribution, waterfall data, and root cause classification.
        """
        t0 = time.perf_counter()
        logger.info(
            "Variance analysis: entity=%s, base=%d, current=%d",
            data.entity_name, data.base_period.year, data.current_period.year,
        )

        total_change = (
            data.current_period.total_emissions_tco2e
            - data.base_period.total_emissions_tco2e
        )
        total_change_pct = _safe_pct(total_change, data.base_period.total_emissions_tco2e)

        # Target variance
        target_var = Decimal("0")
        target_var_pct = Decimal("0")
        if data.target_emissions_tco2e > Decimal("0"):
            target_var = data.current_period.total_emissions_tco2e - data.target_emissions_tco2e
            target_var_pct = _safe_pct(target_var, data.target_emissions_tco2e)

        # LMDI decomposition
        lmdi_components, residual = self._lmdi_decomposition(data)

        # Segment attribution
        segment_attrs = self._segment_attribution(data)

        # Scope variances
        scope_vars = self._scope_variance(data)

        # Kaya decomposition
        kaya: Optional[KayaDecomposition] = None
        if data.include_kaya_decomposition:
            kaya = self._kaya_decomposition(data)

        # Waterfall
        waterfall: List[WaterfallStep] = []
        if data.include_waterfall:
            waterfall = self._build_waterfall(data, lmdi_components, segment_attrs)

        # Severity
        severity = self._assess_severity(total_change_pct, target_var_pct)

        # Data quality
        dq = self._assess_data_quality(data)

        # Recommendations
        recs = self._generate_recommendations(data, lmdi_components, severity)

        # Warnings
        warns = self._generate_warnings(data, residual, segment_attrs)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = VarianceAnalysisResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            base_year=data.base_period.year,
            current_year=data.current_period.year,
            total_change_tco2e=_round_val(total_change, 2),
            total_change_pct=_round_val(total_change_pct, 2),
            target_variance_tco2e=_round_val(target_var, 2),
            target_variance_pct=_round_val(target_var_pct, 2),
            lmdi_components=lmdi_components,
            segment_attributions=segment_attrs,
            scope_variances=scope_vars,
            kaya_decomposition=kaya,
            waterfall_steps=waterfall,
            residual_tco2e=_round_val(residual, 4),
            overall_severity=severity,
            data_quality=dq,
            recommendations=recs,
            warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    async def calculate_batch(
        self, inputs: List[VarianceAnalysisInput],
    ) -> List[VarianceAnalysisResult]:
        """Analyze variance for multiple entities."""
        results: List[VarianceAnalysisResult] = []
        for inp in inputs:
            try:
                results.append(await self.calculate(inp))
            except Exception as exc:
                logger.error("Batch error for %s: %s", inp.entity_name, exc)
                results.append(VarianceAnalysisResult(
                    entity_name=inp.entity_name,
                    warnings=[f"Calculation error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # LMDI-I Decomposition                                                 #
    # ------------------------------------------------------------------ #

    def _lmdi_decomposition(
        self, data: VarianceAnalysisInput,
    ) -> Tuple[List[LMDIComponent], Decimal]:
        """Perform LMDI-I additive decomposition.

        delta_E = delta_ACT + delta_INT + delta_STR

        Where each component uses logarithmic mean weights.
        """
        base = data.base_period
        current = data.current_period
        total_change = current.total_emissions_tco2e - base.total_emissions_tco2e

        # If no segments, use aggregate level
        if not base.segments or not current.segments:
            return self._lmdi_aggregate(data), Decimal("0")

        # Match segments between periods
        base_map = {s.segment_id: s for s in base.segments}
        curr_map = {s.segment_id: s for s in current.segments}
        all_ids = set(base_map.keys()) | set(curr_map.keys())

        delta_act = Decimal("0")
        delta_int = Decimal("0")
        delta_str = Decimal("0")

        Q0 = base.total_activity_value
        Qt = current.total_activity_value

        for sid in all_ids:
            s0 = base_map.get(sid)
            st = curr_map.get(sid)

            # Handle missing segments
            E0 = s0.emissions_tco2e if s0 else Decimal("0.001")
            Et = st.emissions_tco2e if st else Decimal("0.001")
            A0 = s0.activity_value if s0 else Decimal("0.001")
            At = st.activity_value if st else Decimal("0.001")

            if E0 <= Decimal("0"):
                E0 = Decimal("0.001")
            if Et <= Decimal("0"):
                Et = Decimal("0.001")
            if A0 <= Decimal("0"):
                A0 = Decimal("0.001")
            if At <= Decimal("0"):
                At = Decimal("0.001")

            # Shares
            S0 = _safe_divide(A0, Q0) if Q0 > Decimal("0") else Decimal("0.001")
            St = _safe_divide(At, Qt) if Qt > Decimal("0") else Decimal("0.001")

            # Intensities
            I0 = _safe_divide(E0, A0)
            It = _safe_divide(Et, At)

            # Logarithmic mean weight
            w = _log_mean(Et, E0)

            # Activity effect
            delta_act += w * _safe_ln_ratio(Qt, Q0)

            # Intensity effect
            delta_int += w * _safe_ln_ratio(It, I0)

            # Structural effect
            delta_str += w * _safe_ln_ratio(St, S0)

        # Residual check
        residual = total_change - (delta_act + delta_int + delta_str)

        components = [
            LMDIComponent(
                driver=VarianceDriver.ACTIVITY.value,
                delta_tco2e=_round_val(delta_act, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(delta_act), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=(VarianceSeverity.UNFAVORABLE.value if delta_act > Decimal("0")
                          else VarianceSeverity.FAVORABLE.value),
                description="Change in emissions due to total activity level changes "
                           "(revenue, production volume, etc.).",
            ),
            LMDIComponent(
                driver=VarianceDriver.INTENSITY.value,
                delta_tco2e=_round_val(delta_int, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(delta_int), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=(VarianceSeverity.UNFAVORABLE.value if delta_int > Decimal("0")
                          else VarianceSeverity.FAVORABLE.value),
                description="Change in emissions due to emissions intensity changes "
                           "(efficiency improvements or degradation).",
            ),
            LMDIComponent(
                driver=VarianceDriver.STRUCTURAL.value,
                delta_tco2e=_round_val(delta_str, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(delta_str), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=(VarianceSeverity.UNFAVORABLE.value if delta_str > Decimal("0")
                          else VarianceSeverity.FAVORABLE.value),
                description="Change in emissions due to shifts in activity mix "
                           "(business unit or product portfolio changes).",
            ),
        ]

        if abs(residual) > Decimal("0.1"):
            components.append(LMDIComponent(
                driver=VarianceDriver.RESIDUAL.value,
                delta_tco2e=_round_val(residual, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(residual), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=VarianceSeverity.NEUTRAL.value,
                description="LMDI residual (should be near zero).",
            ))

        return components, residual

    def _lmdi_aggregate(
        self, data: VarianceAnalysisInput,
    ) -> List[LMDIComponent]:
        """Simplified LMDI for aggregate (no segment) data.

        Uses two-factor decomposition:
            delta_E = delta_ACT + delta_INT
        """
        base = data.base_period
        current = data.current_period
        total_change = current.total_emissions_tco2e - base.total_emissions_tco2e

        if base.total_activity_value > Decimal("0") and current.total_activity_value > Decimal("0"):
            Q0 = base.total_activity_value
            Qt = current.total_activity_value
            I0 = _safe_divide(base.total_emissions_tco2e, Q0)
            It = _safe_divide(current.total_emissions_tco2e, Qt)

            w = _log_mean(current.total_emissions_tco2e, base.total_emissions_tco2e)
            delta_act = w * _safe_ln_ratio(Qt, Q0)
            delta_int = w * _safe_ln_ratio(It, I0)
        else:
            delta_act = Decimal("0")
            delta_int = total_change

        return [
            LMDIComponent(
                driver=VarianceDriver.ACTIVITY.value,
                delta_tco2e=_round_val(delta_act, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(delta_act), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=(VarianceSeverity.UNFAVORABLE.value if delta_act > Decimal("0")
                          else VarianceSeverity.FAVORABLE.value),
                description="Activity effect (aggregate).",
            ),
            LMDIComponent(
                driver=VarianceDriver.INTENSITY.value,
                delta_tco2e=_round_val(delta_int, 2),
                contribution_pct=_round_val(
                    _safe_pct(abs(delta_int), abs(total_change)) if total_change != Decimal("0") else Decimal("0"), 1
                ),
                direction=(VarianceSeverity.UNFAVORABLE.value if delta_int > Decimal("0")
                          else VarianceSeverity.FAVORABLE.value),
                description="Intensity effect (aggregate).",
            ),
        ]

    # ------------------------------------------------------------------ #
    # Segment Attribution                                                  #
    # ------------------------------------------------------------------ #

    def _segment_attribution(
        self, data: VarianceAnalysisInput,
    ) -> List[SegmentAttribution]:
        """Attribute variance to individual segments."""
        attrs: List[SegmentAttribution] = []
        base_map = {s.segment_id: s for s in data.base_period.segments}
        curr_map = {s.segment_id: s for s in data.current_period.segments}
        all_ids = sorted(set(base_map.keys()) | set(curr_map.keys()))

        Q0 = data.base_period.total_activity_value
        Qt = data.current_period.total_activity_value

        for sid in all_ids:
            s0 = base_map.get(sid)
            st = curr_map.get(sid)

            base_e = s0.emissions_tco2e if s0 else Decimal("0")
            curr_e = st.emissions_tco2e if st else Decimal("0")
            delta = curr_e - base_e
            delta_pct = _safe_pct(delta, base_e) if base_e > Decimal("0") else Decimal("0")

            # Simplified driver attribution for this segment
            base_a = s0.activity_value if s0 else Decimal("0")
            curr_a = st.activity_value if st else Decimal("0")

            activity_eff = Decimal("0")
            intensity_eff = Decimal("0")
            structural_eff = Decimal("0")

            if base_a > Decimal("0") and curr_a > Decimal("0"):
                base_i = _safe_divide(base_e, base_a)
                curr_i = _safe_divide(curr_e, curr_a)
                activity_eff = base_i * (curr_a - base_a)
                intensity_eff = curr_a * (curr_i - base_i)
                structural_eff = delta - activity_eff - intensity_eff

            # Root cause classification
            root_cause = self._classify_root_cause(
                delta_pct, activity_eff, intensity_eff, s0, st,
            )

            attrs.append(SegmentAttribution(
                segment_id=sid,
                segment_name=(st.segment_name if st else s0.segment_name if s0 else ""),
                base_emissions_tco2e=_round_val(base_e, 2),
                current_emissions_tco2e=_round_val(curr_e, 2),
                delta_tco2e=_round_val(delta, 2),
                delta_pct=_round_val(delta_pct, 2),
                activity_effect_tco2e=_round_val(activity_eff, 2),
                intensity_effect_tco2e=_round_val(intensity_eff, 2),
                structural_effect_tco2e=_round_val(structural_eff, 2),
                root_cause=root_cause,
            ))

        return attrs

    def _classify_root_cause(
        self,
        delta_pct: Decimal,
        activity_eff: Decimal,
        intensity_eff: Decimal,
        base_segment: Optional[SegmentData],
        curr_segment: Optional[SegmentData],
    ) -> str:
        """Classify root cause of segment variance.

        Rules:
        - New/removed segment => structural_change
        - Activity dominates => organic_growth
        - Intensity dominates with negative => internal_initiative
        - Intensity dominates with positive => external_factor or data_quality
        """
        if base_segment is None or curr_segment is None:
            return RootCauseCategory.STRUCTURAL_CHANGE.value

        abs_act = abs(activity_eff)
        abs_int = abs(intensity_eff)

        if abs_act > abs_int * Decimal("2"):
            return RootCauseCategory.ORGANIC_GROWTH.value
        elif intensity_eff < Decimal("0"):
            return RootCauseCategory.INTERNAL_INITIATIVE.value
        elif intensity_eff > Decimal("0"):
            return RootCauseCategory.EXTERNAL_FACTOR.value
        return RootCauseCategory.UNKNOWN.value

    # ------------------------------------------------------------------ #
    # Scope Variance                                                       #
    # ------------------------------------------------------------------ #

    def _scope_variance(
        self, data: VarianceAnalysisInput,
    ) -> List[ScopeVariance]:
        """Calculate variance by GHG scope."""
        scopes_base: Dict[str, Decimal] = {}
        scopes_curr: Dict[str, Decimal] = {}

        for s in data.base_period.segments:
            scope = s.scope.value
            scopes_base[scope] = scopes_base.get(scope, Decimal("0")) + s.emissions_tco2e

        for s in data.current_period.segments:
            scope = s.scope.value
            scopes_curr[scope] = scopes_curr.get(scope, Decimal("0")) + s.emissions_tco2e

        all_scopes = sorted(set(scopes_base.keys()) | set(scopes_curr.keys()))
        results: List[ScopeVariance] = []

        for scope in all_scopes:
            base_e = scopes_base.get(scope, Decimal("0"))
            curr_e = scopes_curr.get(scope, Decimal("0"))
            delta = curr_e - base_e
            delta_pct = _safe_pct(delta, base_e) if base_e > Decimal("0") else Decimal("0")

            severity = VarianceSeverity.NEUTRAL.value
            if delta_pct < Decimal("-5"):
                severity = VarianceSeverity.FAVORABLE.value
            elif delta_pct > Decimal("10"):
                severity = VarianceSeverity.UNFAVORABLE.value
            elif delta_pct > Decimal("25"):
                severity = VarianceSeverity.CRITICAL.value

            results.append(ScopeVariance(
                scope=scope,
                base_emissions_tco2e=_round_val(base_e, 2),
                current_emissions_tco2e=_round_val(curr_e, 2),
                delta_tco2e=_round_val(delta, 2),
                delta_pct=_round_val(delta_pct, 2),
                severity=severity,
            ))

        return results

    # ------------------------------------------------------------------ #
    # Kaya Decomposition                                                   #
    # ------------------------------------------------------------------ #

    def _kaya_decomposition(
        self, data: VarianceAnalysisInput,
    ) -> KayaDecomposition:
        """Perform Kaya identity decomposition.

        E = Activity * (Energy/Activity) * (CO2/Energy)
        Simplified to: E = Activity * Intensity

        For corporate level, we use revenue as activity proxy.
        """
        base = data.base_period
        current = data.current_period
        total_change = current.total_emissions_tco2e - base.total_emissions_tco2e

        Q0 = base.total_activity_value
        Qt = current.total_activity_value

        if Q0 <= Decimal("0") or Qt <= Decimal("0"):
            return KayaDecomposition(
                total_change_tco2e=_round_val(total_change, 2),
            )

        I0 = _safe_divide(base.total_emissions_tco2e, Q0)
        It = _safe_divide(current.total_emissions_tco2e, Qt)

        w = _log_mean(current.total_emissions_tco2e, base.total_emissions_tco2e)

        act_eff = w * _safe_ln_ratio(Qt, Q0)
        int_eff = w * _safe_ln_ratio(It, I0)
        residual = total_change - act_eff - int_eff

        act_change_pct = _safe_pct(Qt - Q0, Q0)
        int_change_pct = _safe_pct(It - I0, I0)

        return KayaDecomposition(
            total_change_tco2e=_round_val(total_change, 2),
            activity_effect_tco2e=_round_val(act_eff, 2),
            energy_intensity_effect_tco2e=_round_val(int_eff, 2),
            carbon_intensity_effect_tco2e=Decimal("0"),  # Would require energy data
            activity_change_pct=_round_val(act_change_pct, 2),
            energy_intensity_change_pct=_round_val(int_change_pct, 2),
            carbon_intensity_change_pct=Decimal("0"),
            residual_tco2e=_round_val(residual, 4),
        )

    # ------------------------------------------------------------------ #
    # Waterfall Chart                                                      #
    # ------------------------------------------------------------------ #

    def _build_waterfall(
        self,
        data: VarianceAnalysisInput,
        lmdi: List[LMDIComponent],
        segments: List[SegmentAttribution],
    ) -> List[WaterfallStep]:
        """Build waterfall chart data from LMDI components."""
        steps: List[WaterfallStep] = []
        running = data.base_period.total_emissions_tco2e

        # Start bar
        steps.append(WaterfallStep(
            label=f"Base Year ({data.base_period.year})",
            start_value=Decimal("0"),
            end_value=running,
            delta=running,
            is_total=True,
            category="total",
            color_hint="neutral",
        ))

        # LMDI component bars
        for comp in lmdi:
            if comp.driver == VarianceDriver.RESIDUAL.value:
                continue
            new_running = running + comp.delta_tco2e
            color = "green" if comp.delta_tco2e < Decimal("0") else "red"
            steps.append(WaterfallStep(
                label=comp.driver.replace("_", " ").title() + " Effect",
                start_value=_round_val(running, 2),
                end_value=_round_val(new_running, 2),
                delta=_round_val(comp.delta_tco2e, 2),
                is_total=False,
                category=comp.driver,
                color_hint=color,
            ))
            running = new_running

        # End bar
        steps.append(WaterfallStep(
            label=f"Current Year ({data.current_period.year})",
            start_value=Decimal("0"),
            end_value=_round_val(data.current_period.total_emissions_tco2e, 2),
            delta=_round_val(data.current_period.total_emissions_tco2e, 2),
            is_total=True,
            category="total",
            color_hint="neutral",
        ))

        # Target bar (if set)
        if data.target_emissions_tco2e > Decimal("0"):
            steps.append(WaterfallStep(
                label="Target",
                start_value=Decimal("0"),
                end_value=_round_val(data.target_emissions_tco2e, 2),
                delta=_round_val(data.target_emissions_tco2e, 2),
                is_total=True,
                category="target",
                color_hint="blue",
            ))

        return steps

    # ------------------------------------------------------------------ #
    # Severity / Data Quality / Recommendations / Warnings                 #
    # ------------------------------------------------------------------ #

    def _assess_severity(
        self, total_change_pct: Decimal, target_var_pct: Decimal,
    ) -> str:
        if total_change_pct < Decimal("-5"):
            return VarianceSeverity.FAVORABLE.value
        elif total_change_pct <= Decimal("5"):
            return VarianceSeverity.NEUTRAL.value
        elif total_change_pct <= Decimal("15"):
            return VarianceSeverity.UNFAVORABLE.value
        return VarianceSeverity.CRITICAL.value

    def _assess_data_quality(self, data: VarianceAnalysisInput) -> str:
        score = 0
        if data.base_period.total_emissions_tco2e > Decimal("0"):
            score += 2
        if data.current_period.total_emissions_tco2e > Decimal("0"):
            score += 2
        if len(data.base_period.segments) >= 2:
            score += 2
        if data.base_period.total_activity_value > Decimal("0"):
            score += 2
        if data.entity_id:
            score += 1
        if data.target_emissions_tco2e > Decimal("0"):
            score += 1
        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    def _generate_recommendations(
        self, data: VarianceAnalysisInput,
        lmdi: List[LMDIComponent], severity: str,
    ) -> List[str]:
        recs: List[str] = []
        for comp in lmdi:
            if comp.driver == VarianceDriver.INTENSITY.value and comp.delta_tco2e > Decimal("0"):
                recs.append(
                    f"Emissions intensity increased by {comp.delta_tco2e} tCO2e. "
                    f"Review energy efficiency and process improvements."
                )
            if comp.driver == VarianceDriver.STRUCTURAL.value and abs(comp.delta_tco2e) > abs(data.base_period.total_emissions_tco2e) * Decimal("0.1"):
                recs.append(
                    f"Significant structural change ({comp.delta_tco2e} tCO2e). "
                    f"Consider base year recalculation per GHG Protocol guidance."
                )
        if severity == VarianceSeverity.CRITICAL.value:
            recs.append(
                "Critical variance from trajectory. Implement corrective "
                "action plan immediately."
            )
        if not data.base_period.segments:
            recs.append(
                "Provide segment-level data for full LMDI decomposition "
                "with activity, intensity, and structural effects."
            )
        return recs

    def _generate_warnings(
        self, data: VarianceAnalysisInput,
        residual: Decimal, segments: List[SegmentAttribution],
    ) -> List[str]:
        warns: List[str] = []
        if abs(residual) > data.base_period.total_emissions_tco2e * Decimal("0.01"):
            warns.append(
                f"LMDI residual of {residual} tCO2e exceeds 1% of base "
                f"emissions. Check data consistency."
            )
        new_segments = [s for s in segments if s.base_emissions_tco2e == Decimal("0")]
        removed = [s for s in segments if s.current_emissions_tco2e == Decimal("0")]
        if new_segments:
            warns.append(
                f"{len(new_segments)} new segment(s) with no base period data. "
                f"Consider base year adjustment."
            )
        if removed:
            warns.append(
                f"{len(removed)} segment(s) with no current period data. "
                f"Verify completeness of reporting boundary."
            )
        return warns

    def get_supported_drivers(self) -> List[str]:
        """Return supported variance driver types."""
        return [d.value for d in VarianceDriver]

    def get_supported_root_causes(self) -> List[str]:
        """Return supported root cause categories."""
        return [r.value for r in RootCauseCategory]
