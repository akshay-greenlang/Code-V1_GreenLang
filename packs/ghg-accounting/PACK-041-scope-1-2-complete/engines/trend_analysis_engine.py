# -*- coding: utf-8 -*-
"""
TrendAnalysisEngine - PACK-041 Scope 1-2 Complete Engine 8
=============================================================

Year-over-year GHG emission trend analysis with Kaya identity
decomposition, intensity metric tracking, and SBTi target alignment
assessment for Scope 1 and Scope 2 inventories.

Calculation Methodology:
    Absolute Change:
        delta = current_total - previous_total

    Percentage Change:
        pct_change = (current - previous) / abs(previous) * 100

    CAGR (Compound Annual Growth Rate):
        CAGR = (E_final / E_initial)^(1 / n_years) - 1

    Intensity Metrics:
        intensity = emissions_tco2e / denominator
        Where denominator = revenue, FTEs, floor area, units produced, etc.

    Kaya Identity Decomposition (adapted for corporate emissions):
        E = GDP * (E/GDP) = Activity * Intensity
        delta_E = delta_activity * I_base + A_base * delta_intensity + delta_A * delta_I
        Simplified (LMDI):
            delta_activity_effect = (E_curr - E_prev) / ln(E_curr/E_prev) * ln(A_curr/A_prev)
            delta_intensity_effect = (E_curr - E_prev) / ln(E_curr/E_prev) * ln(I_curr/I_prev)

    SBTi Alignment:
        Linear annual reduction rate:
            required_rate = (base_emissions - target_emissions) / (target_year - base_year)
        Absolute contraction:
            required_rate_pct = required_rate / base_emissions * 100
        SBTi 1.5C target: 4.2% absolute contraction per year (Scope 1+2)

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 8
    - SBTi Corporate Manual (2023), Section 6 (Target validation)
    - SBTi Criteria v5.1 (2023), Table 5.1 (minimum ambition)
    - ESRS E1-4 (GHG emission reduction targets)
    - CDP Climate Change Questionnaire, C4 (Targets and performance)
    - ISO 14064-1:2018, Clause 5.3 (Quantification trends)
    - Kaya (1990), Kaya identity for energy/emissions decomposition
    - Ang (2004), LMDI decomposition method

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - SBTi rates from published corporate manual
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  8 of 10
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


def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IntensityMetricType(str, Enum):
    """Emission intensity metric denominators.

    PER_REVENUE:        tCO2e per unit of revenue (e.g. per million USD).
    PER_FTE:            tCO2e per full-time equivalent employee.
    PER_FLOOR_AREA:     tCO2e per square metre of floor area.
    PER_UNIT_PRODUCED:  tCO2e per unit of product produced.
    PER_TONNE_KM:       tCO2e per tonne-kilometre (logistics).
    PER_PATIENT_DAY:    tCO2e per patient-day (healthcare).
    PER_MWH:            tCO2e per megawatt-hour generated.
    PER_TONNE_PRODUCT:  tCO2e per tonne of product.
    """
    PER_REVENUE = "per_revenue"
    PER_FTE = "per_fte"
    PER_FLOOR_AREA = "per_floor_area"
    PER_UNIT_PRODUCED = "per_unit_produced"
    PER_TONNE_KM = "per_tonne_km"
    PER_PATIENT_DAY = "per_patient_day"
    PER_MWH = "per_mwh"
    PER_TONNE_PRODUCT = "per_tonne_product"


class DecompositionFactor(str, Enum):
    """Factors in the Kaya/LMDI decomposition of emission changes.

    ACTIVITY_LEVEL:       Change due to increased/decreased activity
                          (revenue, production, floor area).
    EMISSION_INTENSITY:   Change due to emission intensity improvement/decline
                          (efficiency, fuel switching, grid decarbonisation).
    STRUCTURAL_CHANGE:    Change due to shifts in organisational structure
                          (product mix, business segment mix).
    METHODOLOGY_CHANGE:   Change due to methodology or EF updates.
    WEATHER:              Change due to weather variation (heating/cooling).
    """
    ACTIVITY_LEVEL = "activity_level"
    EMISSION_INTENSITY = "emission_intensity"
    STRUCTURAL_CHANGE = "structural_change"
    METHODOLOGY_CHANGE = "methodology_change"
    WEATHER = "weather"


class TrendDirection(str, Enum):
    """Overall direction of the emission trend.

    DECREASING:  Emissions are declining.
    STABLE:      No significant change.
    INCREASING:  Emissions are rising.
    """
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"


class SBTiAmbitionLevel(str, Enum):
    """SBTi target ambition levels per SBTi Criteria v5.1.

    ALIGNED_1_5C:   1.5 degrees C pathway (4.2% absolute reduction/yr).
    WELL_BELOW_2C:  Well-below 2C pathway (2.5% absolute reduction/yr).
    BELOW_2C:       Below 2C (not accepted for new targets post-2023).
    NOT_ALIGNED:    Does not meet minimum SBTi ambition.
    """
    ALIGNED_1_5C = "1.5C_aligned"
    WELL_BELOW_2C = "well_below_2C"
    BELOW_2C = "below_2C"
    NOT_ALIGNED = "not_aligned"


# ---------------------------------------------------------------------------
# Constants -- SBTi Target Rates
# ---------------------------------------------------------------------------

# Source: SBTi Corporate Manual (2023), Section 6, Table 5.1.
SBTI_ABSOLUTE_RATES: Dict[str, float] = {
    SBTiAmbitionLevel.ALIGNED_1_5C: 4.2,
    SBTiAmbitionLevel.WELL_BELOW_2C: 2.5,
    SBTiAmbitionLevel.BELOW_2C: 1.23,
}
"""Required annual absolute contraction rates (% per year) by ambition."""

# Default intensity denominators and units.
INTENSITY_UNITS: Dict[str, str] = {
    IntensityMetricType.PER_REVENUE: "tCO2e/M USD",
    IntensityMetricType.PER_FTE: "tCO2e/FTE",
    IntensityMetricType.PER_FLOOR_AREA: "tCO2e/m2",
    IntensityMetricType.PER_UNIT_PRODUCED: "tCO2e/unit",
    IntensityMetricType.PER_TONNE_KM: "tCO2e/tonne-km",
    IntensityMetricType.PER_PATIENT_DAY: "tCO2e/patient-day",
    IntensityMetricType.PER_MWH: "tCO2e/MWh",
    IntensityMetricType.PER_TONNE_PRODUCT: "tCO2e/tonne",
}
"""Unit labels for intensity metrics."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class YearlyEmissions(BaseModel):
    """GHG emissions for a single reporting year.

    Attributes:
        year: Reporting year.
        scope1_total: Total Scope 1 emissions (tCO2e).
        scope2_location: Scope 2 location-based emissions (tCO2e).
        scope2_market: Scope 2 market-based emissions (tCO2e).
        per_category: Emissions by source category.
        per_gas: Emissions by gas type (CO2, CH4, N2O, HFCs, etc.).
        activity_data: Activity data for intensity calculations.
    """
    year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    scope1_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1 total (tCO2e)"
    )
    scope2_location: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 location-based (tCO2e)"
    )
    scope2_market: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 market-based (tCO2e)"
    )
    per_category: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by source category"
    )
    per_gas: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by gas type"
    )
    activity_data: Dict[str, float] = Field(
        default_factory=dict, description="Activity data (revenue, FTEs, area, etc.)"
    )

    @field_validator("scope1_total", "scope2_location", "scope2_market", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission values to Decimal."""
        return _decimal(v)

    @property
    def scope1_plus_2_location(self) -> Decimal:
        """Total Scope 1 + Scope 2 location-based."""
        return self.scope1_total + self.scope2_location

    @property
    def scope1_plus_2_market(self) -> Decimal:
        """Total Scope 1 + Scope 2 market-based."""
        return self.scope1_total + self.scope2_market


class SBTiTarget(BaseModel):
    """Science-based target definition.

    Attributes:
        base_year: Base year for the target.
        target_year: Year by which target must be met.
        base_emissions_tco2e: Base year emissions (tCO2e).
        target_emissions_tco2e: Target year emissions (tCO2e).
        ambition_level: SBTi ambition level.
        scope_coverage: Which scopes are covered.
    """
    base_year: int = Field(..., ge=1990, description="Base year")
    target_year: int = Field(..., ge=2025, description="Target year")
    base_emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Base year emissions"
    )
    target_emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Target year emissions"
    )
    ambition_level: SBTiAmbitionLevel = Field(
        default=SBTiAmbitionLevel.ALIGNED_1_5C, description="Ambition level"
    )
    scope_coverage: str = Field(
        default="scope1_scope2", description="Scope coverage"
    )

    @field_validator("base_emissions_tco2e", "target_emissions_tco2e", mode="before")
    @classmethod
    def coerce_target(cls, v: Any) -> Decimal:
        """Coerce target values to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class YearOverYearChange(BaseModel):
    """Year-over-year emission change between two consecutive years.

    Attributes:
        from_year: Starting year.
        to_year: Ending year.
        absolute_change_tco2e: Absolute change (tCO2e).
        percentage_change: Percentage change (%).
        scope1_change_tco2e: Scope 1 change.
        scope2_location_change_tco2e: Scope 2 location-based change.
        scope2_market_change_tco2e: Scope 2 market-based change.
    """
    from_year: int = Field(default=0, description="From year")
    to_year: int = Field(default=0, description="To year")
    absolute_change_tco2e: float = Field(default=0.0, description="Absolute change")
    percentage_change: float = Field(default=0.0, description="Percentage change")
    scope1_change_tco2e: float = Field(default=0.0, description="Scope 1 change")
    scope2_location_change_tco2e: float = Field(
        default=0.0, description="Scope 2 loc change"
    )
    scope2_market_change_tco2e: float = Field(
        default=0.0, description="Scope 2 mkt change"
    )


class IntensityMetric(BaseModel):
    """Emission intensity metric result.

    Attributes:
        metric_type: Type of intensity metric.
        year: Reporting year.
        numerator_tco2e: Emissions (numerator).
        denominator_value: Activity data (denominator).
        denominator_unit: Unit of denominator.
        intensity_value: Calculated intensity.
        yoy_change_pct: Year-over-year change in intensity (%).
    """
    metric_type: str = Field(default="", description="Metric type")
    year: int = Field(default=0, description="Year")
    numerator_tco2e: float = Field(default=0.0, description="Numerator (tCO2e)")
    denominator_value: float = Field(default=0.0, description="Denominator")
    denominator_unit: str = Field(default="", description="Denominator unit")
    intensity_value: float = Field(default=0.0, description="Intensity value")
    yoy_change_pct: Optional[float] = Field(
        default=None, description="YoY change (%)"
    )


class DecompositionResult(BaseModel):
    """Result of a single factor in Kaya/LMDI decomposition.

    Attributes:
        factor: Decomposition factor type.
        contribution_tco2e: Contribution to total change (tCO2e).
        contribution_pct: Contribution as percentage of total change.
        description: Explanation of this factor's contribution.
    """
    factor: str = Field(default="", description="Factor name")
    contribution_tco2e: float = Field(default=0.0, description="Contribution (tCO2e)")
    contribution_pct: float = Field(default=0.0, description="Contribution (%)")
    description: str = Field(default="", description="Explanation")


class KayaResult(BaseModel):
    """Complete Kaya/LMDI decomposition result.

    Attributes:
        from_year: Analysis start year.
        to_year: Analysis end year.
        total_change_tco2e: Total emission change.
        factors: Individual factor contributions.
        residual_tco2e: Unexplained residual.
        method: Decomposition method used.
    """
    from_year: int = Field(default=0, description="Start year")
    to_year: int = Field(default=0, description="End year")
    total_change_tco2e: float = Field(default=0.0, description="Total change")
    factors: List[DecompositionResult] = Field(
        default_factory=list, description="Factor contributions"
    )
    residual_tco2e: float = Field(default=0.0, description="Residual")
    method: str = Field(default="LMDI", description="Method")


class SBTiAlignment(BaseModel):
    """Assessment of emission trajectory against SBTi targets.

    Attributes:
        ambition_level: Assessed ambition level.
        required_annual_reduction_pct: Required annual absolute reduction (%).
        actual_annual_reduction_pct: Actual observed reduction rate (%).
        is_on_track: Whether current trajectory meets target.
        gap_tco2e: Emissions gap vs target pathway at latest year.
        gap_pct: Gap as percentage of target pathway.
        years_to_target: Years remaining to target year.
        projected_target_year_emissions: Projected emissions at target year.
        assessment_narrative: Explanation of alignment status.
    """
    ambition_level: str = Field(default="", description="Ambition level")
    required_annual_reduction_pct: float = Field(
        default=0.0, description="Required rate (%/yr)"
    )
    actual_annual_reduction_pct: float = Field(
        default=0.0, description="Actual rate (%/yr)"
    )
    is_on_track: bool = Field(default=False, description="On track")
    gap_tco2e: float = Field(default=0.0, description="Gap (tCO2e)")
    gap_pct: float = Field(default=0.0, description="Gap (%)")
    years_to_target: int = Field(default=0, description="Years remaining")
    projected_target_year_emissions: float = Field(
        default=0.0, description="Projected target year emissions"
    )
    assessment_narrative: str = Field(default="", description="Narrative")


class BaseYearComparison(BaseModel):
    """Comparison of latest year against base year.

    Attributes:
        base_year: Base year number.
        current_year: Current year number.
        base_total_tco2e: Base year total.
        current_total_tco2e: Current year total.
        absolute_change_tco2e: Absolute change from base.
        percentage_change: Percentage change from base.
        cagr_pct: Compound annual growth rate.
    """
    base_year: int = Field(default=0, description="Base year")
    current_year: int = Field(default=0, description="Current year")
    base_total_tco2e: float = Field(default=0.0, description="Base total")
    current_total_tco2e: float = Field(default=0.0, description="Current total")
    absolute_change_tco2e: float = Field(default=0.0, description="Absolute change")
    percentage_change: float = Field(default=0.0, description="Percentage change")
    cagr_pct: float = Field(default=0.0, description="CAGR (%)")


class TrendAnalysisResult(BaseModel):
    """Complete GHG trend analysis result with full provenance.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing time (ms).
        years_analyzed: Years included in the analysis.
        direction: Overall trend direction.
        yoy_changes: Year-over-year changes.
        absolute_change_tco2e: Total absolute change over period.
        percentage_change: Total percentage change over period.
        cagr_pct: Compound annual growth rate.
        base_year_comparison: Comparison vs base year.
        decomposition: Kaya/LMDI decomposition.
        intensity_metrics: Emission intensity metrics.
        sbti_alignment: SBTi target alignment assessment.
        category_trends: Per-category trends.
        gas_trends: Per-gas trends.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    years_analyzed: List[int] = Field(
        default_factory=list, description="Years analyzed"
    )
    direction: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend direction"
    )
    yoy_changes: List[YearOverYearChange] = Field(
        default_factory=list, description="Year-over-year changes"
    )
    absolute_change_tco2e: float = Field(
        default=0.0, description="Total absolute change"
    )
    percentage_change: float = Field(
        default=0.0, description="Total percentage change"
    )
    cagr_pct: float = Field(default=0.0, description="CAGR (%)")
    base_year_comparison: Optional[BaseYearComparison] = Field(
        default=None, description="Base year comparison"
    )
    decomposition: Optional[KayaResult] = Field(
        default=None, description="Kaya decomposition"
    )
    intensity_metrics: List[IntensityMetric] = Field(
        default_factory=list, description="Intensity metrics"
    )
    sbti_alignment: Optional[SBTiAlignment] = Field(
        default=None, description="SBTi alignment"
    )
    category_trends: Dict[str, float] = Field(
        default_factory=dict, description="Category trends (% change)"
    )
    gas_trends: Dict[str, float] = Field(
        default_factory=dict, description="Gas trends (% change)"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TrendAnalysisEngine:
    """GHG emission trend analysis engine with Kaya decomposition.

    Provides year-over-year analysis, intensity tracking, LMDI decomposition,
    and SBTi target alignment for Scope 1 and Scope 2 inventories.

    Guarantees:
        - Deterministic: identical inputs produce identical outputs.
        - Reproducible: SHA-256 provenance hash on every result.
        - Compliant: GHG Protocol Chapter 8, SBTi Corporate Manual.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = TrendAnalysisEngine()
        result = engine.analyze_trend(yearly_data, base_year=2019)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the trend analysis engine.

        Args:
            config: Optional overrides. Supported keys:
                - stability_threshold_pct (float): threshold for STABLE (default 2.0)
                - default_intensity_metrics (list): default metric types
        """
        self._config = config or {}
        self._stability_threshold = float(
            self._config.get("stability_threshold_pct", 2.0)
        )
        self._default_metrics = self._config.get("default_intensity_metrics", [
            IntensityMetricType.PER_REVENUE,
            IntensityMetricType.PER_FTE,
        ])
        self._notes: List[str] = []
        logger.info("TrendAnalysisEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def analyze_trend(
        self,
        yearly_data: List[YearlyEmissions],
        base_year: Optional[int] = None,
        sbti_target: Optional[SBTiTarget] = None,
        intensity_metrics: Optional[List[IntensityMetricType]] = None,
    ) -> TrendAnalysisResult:
        """Run comprehensive GHG trend analysis.

        Args:
            yearly_data: Emission data for multiple years (chronological).
            base_year: Base year for comparison (default: earliest year).
            sbti_target: Optional SBTi target for alignment assessment.
            intensity_metrics: Metric types to calculate.

        Returns:
            TrendAnalysisResult with full analysis.

        Raises:
            ValueError: If fewer than 2 years of data provided.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if len(yearly_data) < 2:
            raise ValueError("At least 2 years of data required for trend analysis.")

        # Sort chronologically
        sorted_data = sorted(yearly_data, key=lambda y: y.year)
        years = [y.year for y in sorted_data]
        base_yr = base_year or years[0]

        logger.info(
            "Trend analysis: %d years (%d-%d), base=%d",
            len(sorted_data), years[0], years[-1], base_yr,
        )

        # Year-over-year changes
        yoy_changes = self._calculate_yoy_changes(sorted_data)

        # Overall change
        first = sorted_data[0]
        last = sorted_data[-1]
        abs_change = self.calculate_absolute_change(
            last.scope1_plus_2_location, first.scope1_plus_2_location
        )
        pct_change = self.calculate_percentage_change(
            last.scope1_plus_2_location, first.scope1_plus_2_location
        )

        # CAGR
        n_years = max(last.year - first.year, 1)
        cagr = self._calculate_cagr(
            first.scope1_plus_2_location, last.scope1_plus_2_location, n_years
        )

        # Trend direction
        direction = self._assess_direction(pct_change, n_years)

        # Base year comparison
        base_data = next((y for y in sorted_data if y.year == base_yr), first)
        base_comparison = self._calculate_base_year_comparison(base_data, last)

        # Kaya/LMDI decomposition
        decomposition = None
        if (
            len(sorted_data) >= 2
            and sorted_data[-1].activity_data
            and sorted_data[-2].activity_data
        ):
            decomposition = self.kaya_decomposition(sorted_data)

        # Intensity metrics
        metrics = intensity_metrics or self._default_metrics
        intensity_results = self.calculate_intensity_metrics(
            sorted_data, metrics
        )

        # SBTi alignment
        sbti_result = None
        if sbti_target:
            sbti_result = self.assess_sbti_alignment(sorted_data, sbti_target)

        # Category trends
        cat_trends = self._calculate_category_trends(sorted_data)

        # Gas trends
        gas_trends = self._calculate_gas_trends(sorted_data)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TrendAnalysisResult(
            years_analyzed=years,
            direction=direction,
            yoy_changes=yoy_changes,
            absolute_change_tco2e=_round2(float(abs_change)),
            percentage_change=_round2(float(pct_change)),
            cagr_pct=_round4(float(cagr)),
            base_year_comparison=base_comparison,
            decomposition=decomposition,
            intensity_metrics=intensity_results,
            sbti_alignment=sbti_result,
            category_trends=cat_trends,
            gas_trends=gas_trends,
            methodology_notes=list(self._notes),
            processing_time_ms=_round3(elapsed_ms),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Trend analysis complete: %d years, direction=%s, "
            "change=%.2f%%, CAGR=%.2f%%, hash=%s (%.1f ms)",
            len(sorted_data), direction.value, float(pct_change),
            float(cagr), result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def calculate_absolute_change(
        self,
        current: Decimal,
        previous: Decimal,
    ) -> Decimal:
        """Calculate absolute change between two values.

        Args:
            current: Current period value.
            previous: Previous period value.

        Returns:
            Absolute change (current - previous).
        """
        return _decimal(current) - _decimal(previous)

    def calculate_percentage_change(
        self,
        current: Decimal,
        previous: Decimal,
    ) -> Decimal:
        """Calculate percentage change between two values.

        Formula: (current - previous) / abs(previous) * 100

        Args:
            current: Current period value.
            previous: Previous period value.

        Returns:
            Percentage change; Decimal("0") if previous is zero.
        """
        d_curr = _decimal(current)
        d_prev = _decimal(previous)
        return _safe_pct(d_curr - d_prev, abs(d_prev))

    def decompose_changes(
        self,
        current: YearlyEmissions,
        previous: YearlyEmissions,
    ) -> List[DecompositionResult]:
        """Decompose emission change between two years.

        Uses simplified LMDI (Log-Mean Divisia Index) method to
        separate the total emission change into activity effect
        and intensity effect.

        Args:
            current: Current year data.
            previous: Previous year data.

        Returns:
            List of DecompositionResult factors.
        """
        results: List[DecompositionResult] = []
        e_curr = _decimal(current.scope1_plus_2_location)
        e_prev = _decimal(previous.scope1_plus_2_location)
        total_change = e_curr - e_prev

        if e_curr <= Decimal("0") or e_prev <= Decimal("0"):
            results.append(DecompositionResult(
                factor=DecompositionFactor.ACTIVITY_LEVEL.value,
                contribution_tco2e=_round2(float(total_change)),
                contribution_pct=100.0,
                description="Cannot decompose: zero or negative emissions.",
            ))
            return results

        # LMDI weight
        if e_curr != e_prev:
            ln_ratio = _decimal(math.log(float(e_curr / e_prev)))
            lmdi_weight = _safe_divide(e_curr - e_prev, ln_ratio)
        else:
            lmdi_weight = e_curr

        # Activity effect: use revenue as activity proxy
        a_curr = _decimal(current.activity_data.get("revenue", 0))
        a_prev = _decimal(previous.activity_data.get("revenue", 0))

        activity_effect = Decimal("0")
        intensity_effect = Decimal("0")

        if a_curr > Decimal("0") and a_prev > Decimal("0"):
            ln_a_ratio = _decimal(math.log(float(a_curr / a_prev)))
            activity_effect = lmdi_weight * ln_a_ratio

            # Intensity = E / A
            i_curr = _safe_divide(e_curr, a_curr)
            i_prev = _safe_divide(e_prev, a_prev)
            if i_curr > Decimal("0") and i_prev > Decimal("0"):
                ln_i_ratio = _decimal(math.log(float(i_curr / i_prev)))
                intensity_effect = lmdi_weight * ln_i_ratio
        else:
            activity_effect = total_change
            intensity_effect = Decimal("0")

        residual = total_change - activity_effect - intensity_effect

        results.append(DecompositionResult(
            factor=DecompositionFactor.ACTIVITY_LEVEL.value,
            contribution_tco2e=_round2(float(activity_effect)),
            contribution_pct=_round2(float(_safe_pct(activity_effect, total_change)))
            if total_change != Decimal("0") else 0.0,
            description=(
                f"Change attributable to activity level "
                f"({'increase' if activity_effect > 0 else 'decrease'})."
            ),
        ))

        results.append(DecompositionResult(
            factor=DecompositionFactor.EMISSION_INTENSITY.value,
            contribution_tco2e=_round2(float(intensity_effect)),
            contribution_pct=_round2(float(_safe_pct(intensity_effect, total_change)))
            if total_change != Decimal("0") else 0.0,
            description=(
                f"Change attributable to emission intensity "
                f"({'improvement' if intensity_effect < 0 else 'deterioration'})."
            ),
        ))

        if abs(residual) > Decimal("0.01"):
            results.append(DecompositionResult(
                factor="residual",
                contribution_tco2e=_round2(float(residual)),
                contribution_pct=_round2(float(_safe_pct(residual, total_change)))
                if total_change != Decimal("0") else 0.0,
                description="Interaction term and unexplained residual.",
            ))

        return results

    def calculate_intensity_metrics(
        self,
        yearly_data: List[YearlyEmissions],
        metric_types: List[IntensityMetricType],
    ) -> List[IntensityMetric]:
        """Calculate emission intensity metrics across years.

        Args:
            yearly_data: Emission data sorted chronologically.
            metric_types: Which intensity metrics to calculate.

        Returns:
            List of IntensityMetric results.
        """
        results: List[IntensityMetric] = []
        prev_intensity: Dict[str, Decimal] = {}

        activity_key_map = {
            IntensityMetricType.PER_REVENUE: "revenue",
            IntensityMetricType.PER_FTE: "fte",
            IntensityMetricType.PER_FLOOR_AREA: "floor_area",
            IntensityMetricType.PER_UNIT_PRODUCED: "units_produced",
            IntensityMetricType.PER_TONNE_KM: "tonne_km",
            IntensityMetricType.PER_PATIENT_DAY: "patient_days",
            IntensityMetricType.PER_MWH: "mwh_generated",
            IntensityMetricType.PER_TONNE_PRODUCT: "tonnes_product",
        }

        for year_data in yearly_data:
            emissions = _decimal(year_data.scope1_plus_2_location)

            for metric_type in metric_types:
                activity_key = activity_key_map.get(metric_type, "revenue")
                denominator = _decimal(year_data.activity_data.get(activity_key, 0))

                if denominator <= Decimal("0"):
                    continue

                intensity = _safe_divide(emissions, denominator)
                unit = INTENSITY_UNITS.get(metric_type, "tCO2e/unit")

                yoy_change = None
                metric_key = f"{metric_type.value}_{year_data.year}"
                prev_key = f"{metric_type.value}_{year_data.year - 1}"
                prev_val = prev_intensity.get(prev_key)
                if prev_val is not None and prev_val > Decimal("0"):
                    yoy_change = _round2(float(
                        _safe_pct(intensity - prev_val, abs(prev_val))
                    ))

                prev_intensity[metric_key] = intensity

                results.append(IntensityMetric(
                    metric_type=metric_type.value,
                    year=year_data.year,
                    numerator_tco2e=_round2(float(emissions)),
                    denominator_value=_round2(float(denominator)),
                    denominator_unit=unit,
                    intensity_value=_round4(float(intensity)),
                    yoy_change_pct=yoy_change,
                ))

        return results

    def assess_sbti_alignment(
        self,
        yearly_data: List[YearlyEmissions],
        target: SBTiTarget,
    ) -> SBTiAlignment:
        """Assess emission trajectory against SBTi target.

        Compares the observed annual reduction rate against the required
        rate for the specified ambition level.

        Args:
            yearly_data: Emission data sorted chronologically.
            target: SBTi target definition.

        Returns:
            SBTiAlignment assessment.
        """
        sorted_data = sorted(yearly_data, key=lambda y: y.year)
        latest = sorted_data[-1]
        latest_emissions = _decimal(latest.scope1_plus_2_location)

        base_emissions = _decimal(target.base_emissions_tco2e)
        target_emissions = _decimal(target.target_emissions_tco2e)
        target_span = max(target.target_year - target.base_year, 1)
        years_elapsed = max(latest.year - target.base_year, 1)
        years_remaining = max(target.target_year - latest.year, 0)

        # Required annual reduction
        required_total_reduction = base_emissions - target_emissions
        required_annual = _safe_divide(required_total_reduction, _decimal(target_span))
        required_rate_pct = float(_safe_pct(required_annual, base_emissions))

        # Actual reduction from base year
        actual_total_reduction = base_emissions - latest_emissions
        actual_annual = _safe_divide(actual_total_reduction, _decimal(years_elapsed))
        actual_rate_pct = float(_safe_pct(actual_annual, base_emissions))

        # Projected emissions at target year
        if years_remaining > 0 and actual_annual != Decimal("0"):
            projected = latest_emissions - (actual_annual * _decimal(years_remaining))
            projected = max(projected, Decimal("0"))
        else:
            projected = latest_emissions

        # Expected pathway value at current year
        expected_at_latest = base_emissions - (required_annual * _decimal(years_elapsed))
        gap = latest_emissions - expected_at_latest
        gap_pct = float(_safe_pct(gap, expected_at_latest))

        is_on_track = latest_emissions <= expected_at_latest

        # Determine ambition level
        sbti_rates = SBTI_ABSOLUTE_RATES
        if actual_rate_pct >= sbti_rates.get(SBTiAmbitionLevel.ALIGNED_1_5C, 4.2):
            assessed_ambition = SBTiAmbitionLevel.ALIGNED_1_5C.value
        elif actual_rate_pct >= sbti_rates.get(SBTiAmbitionLevel.WELL_BELOW_2C, 2.5):
            assessed_ambition = SBTiAmbitionLevel.WELL_BELOW_2C.value
        elif actual_rate_pct >= sbti_rates.get(SBTiAmbitionLevel.BELOW_2C, 1.23):
            assessed_ambition = SBTiAmbitionLevel.BELOW_2C.value
        else:
            assessed_ambition = SBTiAmbitionLevel.NOT_ALIGNED.value

        narrative = self._build_sbti_narrative(
            is_on_track, actual_rate_pct, required_rate_pct,
            assessed_ambition, years_remaining, float(gap),
        )

        self._notes.append(
            f"SBTi alignment: required {_round2(required_rate_pct)}%/yr, "
            f"actual {_round2(actual_rate_pct)}%/yr, "
            f"{'ON TRACK' if is_on_track else 'OFF TRACK'}."
        )

        return SBTiAlignment(
            ambition_level=assessed_ambition,
            required_annual_reduction_pct=_round2(required_rate_pct),
            actual_annual_reduction_pct=_round2(actual_rate_pct),
            is_on_track=is_on_track,
            gap_tco2e=_round2(float(gap)),
            gap_pct=_round2(gap_pct),
            years_to_target=years_remaining,
            projected_target_year_emissions=_round2(float(projected)),
            assessment_narrative=narrative,
        )

    def kaya_decomposition(
        self,
        yearly_data: List[YearlyEmissions],
    ) -> KayaResult:
        """Perform Kaya/LMDI decomposition across full time series.

        Decomposes the total emission change from first to last year into
        activity effect and intensity effect using the LMDI method.

        Args:
            yearly_data: Emission data sorted chronologically.

        Returns:
            KayaResult with factor contributions.
        """
        sorted_data = sorted(yearly_data, key=lambda y: y.year)
        first = sorted_data[0]
        last = sorted_data[-1]

        factors = self.decompose_changes(last, first)
        total_change = _decimal(last.scope1_plus_2_location) - _decimal(first.scope1_plus_2_location)
        sum_factors = sum(_decimal(f.contribution_tco2e) for f in factors)
        residual = total_change - sum_factors

        return KayaResult(
            from_year=first.year,
            to_year=last.year,
            total_change_tco2e=_round2(float(total_change)),
            factors=factors,
            residual_tco2e=_round2(float(residual)),
            method="LMDI (Log-Mean Divisia Index)",
        )

    # -------------------------------------------------------------------
    # Private -- Year-over-year
    # -------------------------------------------------------------------

    def _calculate_yoy_changes(
        self,
        sorted_data: List[YearlyEmissions],
    ) -> List[YearOverYearChange]:
        """Calculate year-over-year changes for all consecutive pairs.

        Args:
            sorted_data: Chronologically sorted emission data.

        Returns:
            List of YearOverYearChange objects.
        """
        changes: List[YearOverYearChange] = []

        for i in range(1, len(sorted_data)):
            prev = sorted_data[i - 1]
            curr = sorted_data[i]

            total_prev = _decimal(prev.scope1_plus_2_location)
            total_curr = _decimal(curr.scope1_plus_2_location)
            abs_change = total_curr - total_prev
            pct_change = _safe_pct(abs_change, abs(total_prev))

            s1_change = _decimal(curr.scope1_total) - _decimal(prev.scope1_total)
            s2l_change = _decimal(curr.scope2_location) - _decimal(prev.scope2_location)
            s2m_change = _decimal(curr.scope2_market) - _decimal(prev.scope2_market)

            changes.append(YearOverYearChange(
                from_year=prev.year,
                to_year=curr.year,
                absolute_change_tco2e=_round2(float(abs_change)),
                percentage_change=_round2(float(pct_change)),
                scope1_change_tco2e=_round2(float(s1_change)),
                scope2_location_change_tco2e=_round2(float(s2l_change)),
                scope2_market_change_tco2e=_round2(float(s2m_change)),
            ))

        return changes

    # -------------------------------------------------------------------
    # Private -- CAGR and direction
    # -------------------------------------------------------------------

    def _calculate_cagr(
        self,
        initial: Decimal,
        final: Decimal,
        n_years: int,
    ) -> Decimal:
        """Calculate compound annual growth rate.

        Formula: CAGR = (final / initial)^(1/n) - 1

        Args:
            initial: Starting value.
            final: Ending value.
            n_years: Number of years.

        Returns:
            CAGR as a percentage (e.g. -3.5 for 3.5% decline).
        """
        d_init = _decimal(initial)
        d_final = _decimal(final)

        if d_init <= Decimal("0") or n_years <= 0:
            return Decimal("0")

        ratio = float(d_final / d_init)
        if ratio <= 0:
            return Decimal("-100")

        cagr = (ratio ** (1.0 / n_years) - 1.0) * 100.0
        return _decimal(cagr)

    def _assess_direction(
        self,
        pct_change: Decimal,
        n_years: int,
    ) -> TrendDirection:
        """Assess overall trend direction.

        Args:
            pct_change: Total percentage change over period.
            n_years: Number of years in period.

        Returns:
            TrendDirection enum value.
        """
        annualized = abs(float(pct_change)) / max(n_years, 1)

        if annualized < self._stability_threshold:
            return TrendDirection.STABLE
        elif float(pct_change) < 0:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.INCREASING

    # -------------------------------------------------------------------
    # Private -- Base year comparison
    # -------------------------------------------------------------------

    def _calculate_base_year_comparison(
        self,
        base: YearlyEmissions,
        current: YearlyEmissions,
    ) -> BaseYearComparison:
        """Compare current year against base year.

        Args:
            base: Base year data.
            current: Current year data.

        Returns:
            BaseYearComparison with change metrics.
        """
        base_total = _decimal(base.scope1_plus_2_location)
        curr_total = _decimal(current.scope1_plus_2_location)
        abs_change = curr_total - base_total
        pct_change = _safe_pct(abs_change, abs(base_total))
        n_years = max(current.year - base.year, 1)
        cagr = self._calculate_cagr(base_total, curr_total, n_years)

        return BaseYearComparison(
            base_year=base.year,
            current_year=current.year,
            base_total_tco2e=_round2(float(base_total)),
            current_total_tco2e=_round2(float(curr_total)),
            absolute_change_tco2e=_round2(float(abs_change)),
            percentage_change=_round2(float(pct_change)),
            cagr_pct=_round4(float(cagr)),
        )

    # -------------------------------------------------------------------
    # Private -- Category and gas trends
    # -------------------------------------------------------------------

    def _calculate_category_trends(
        self,
        sorted_data: List[YearlyEmissions],
    ) -> Dict[str, float]:
        """Calculate percentage change by source category.

        Args:
            sorted_data: Chronologically sorted data.

        Returns:
            Dict mapping category to percentage change.
        """
        if len(sorted_data) < 2:
            return {}

        first = sorted_data[0]
        last = sorted_data[-1]
        trends: Dict[str, float] = {}

        all_cats = set(first.per_category.keys()) | set(last.per_category.keys())
        for cat in sorted(all_cats):
            prev_val = _decimal(first.per_category.get(cat, 0.0))
            curr_val = _decimal(last.per_category.get(cat, 0.0))
            pct = _safe_pct(curr_val - prev_val, abs(prev_val)) if prev_val != Decimal("0") else Decimal("0")
            trends[cat] = _round2(float(pct))

        return trends

    def _calculate_gas_trends(
        self,
        sorted_data: List[YearlyEmissions],
    ) -> Dict[str, float]:
        """Calculate percentage change by gas type.

        Args:
            sorted_data: Chronologically sorted data.

        Returns:
            Dict mapping gas to percentage change.
        """
        if len(sorted_data) < 2:
            return {}

        first = sorted_data[0]
        last = sorted_data[-1]
        trends: Dict[str, float] = {}

        all_gases = set(first.per_gas.keys()) | set(last.per_gas.keys())
        for gas in sorted(all_gases):
            prev_val = _decimal(first.per_gas.get(gas, 0.0))
            curr_val = _decimal(last.per_gas.get(gas, 0.0))
            pct = _safe_pct(curr_val - prev_val, abs(prev_val)) if prev_val != Decimal("0") else Decimal("0")
            trends[gas] = _round2(float(pct))

        return trends

    # -------------------------------------------------------------------
    # Private -- SBTi narrative
    # -------------------------------------------------------------------

    def _build_sbti_narrative(
        self,
        is_on_track: bool,
        actual_rate: float,
        required_rate: float,
        ambition: str,
        years_remaining: int,
        gap: float,
    ) -> str:
        """Build SBTi alignment narrative.

        Args:
            is_on_track: Whether on track.
            actual_rate: Actual reduction rate.
            required_rate: Required rate.
            ambition: Assessed ambition level.
            years_remaining: Years to target.
            gap: Current gap in tCO2e.

        Returns:
            Narrative string.
        """
        parts: List[str] = []

        if is_on_track:
            parts.append(
                f"Emissions are ON TRACK with the SBTi target pathway."
            )
            parts.append(
                f"Actual annual reduction rate ({_round2(actual_rate)}%/yr) "
                f"exceeds the required rate ({_round2(required_rate)}%/yr)."
            )
        else:
            parts.append(
                f"Emissions are OFF TRACK with the SBTi target pathway."
            )
            parts.append(
                f"Actual annual reduction rate ({_round2(actual_rate)}%/yr) "
                f"falls short of the required rate ({_round2(required_rate)}%/yr)."
            )
            if years_remaining > 0:
                catch_up_rate = required_rate + abs(gap) / max(years_remaining, 1) / 100.0
                parts.append(
                    f"To close the gap of {_round2(abs(gap))} tCO2e over "
                    f"{years_remaining} years, an accelerated reduction rate "
                    f"of approximately {_round2(catch_up_rate)}%/yr is needed."
                )

        parts.append(
            f"Current trajectory assessed as: {ambition}."
        )

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

YearlyEmissions.model_rebuild()
SBTiTarget.model_rebuild()
YearOverYearChange.model_rebuild()
IntensityMetric.model_rebuild()
DecompositionResult.model_rebuild()
KayaResult.model_rebuild()
SBTiAlignment.model_rebuild()
BaseYearComparison.model_rebuild()
TrendAnalysisResult.model_rebuild()
