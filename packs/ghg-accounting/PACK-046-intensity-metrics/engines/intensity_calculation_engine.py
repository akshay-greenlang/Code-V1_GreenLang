# -*- coding: utf-8 -*-
"""
IntensityCalculationEngine - PACK-046 Intensity Metrics Engine 2
====================================================================

Core calculation engine for GHG emissions intensity metrics.  Implements
the fundamental intensity formula (emissions / denominator) with full
scope configuration, multi-entity consolidation, and time-series support.

Calculation Methodology:
    Basic Intensity:
        I = E / D
        Where:
            I = intensity (tCO2e per denominator unit)
            E = total emissions for selected scopes (tCO2e)
            D = denominator value (in its standard unit)

    Scope Inclusion:
        SCOPE_1_ONLY:           E = scope_1
        SCOPE_2_LOCATION:       E = scope_2_location
        SCOPE_2_MARKET:         E = scope_2_market
        SCOPE_1_2_LOCATION:     E = scope_1 + scope_2_location
        SCOPE_1_2_MARKET:       E = scope_1 + scope_2_market
        SCOPE_1_2_3:            E = scope_1 + scope_2_location + scope_3
        SCOPE_3_SPECIFIC:       E = SUM(selected scope_3 categories)
        CUSTOM:                 E = SUM(user-specified emission components)

    Multi-Entity Weighted Average:
        I_consolidated = SUM(entity_emissions) / SUM(entity_denominators)
        This is the CORRECT method (NOT average of per-entity intensities).

    Time Series:
        For each period t:  I_t = E_t / D_t
        YoY change:         delta_pct = (I_t - I_{t-1}) / I_{t-1} * 100
        Cumulative change:  cum_pct = (I_t - I_0) / I_0 * 100

    Edge Cases:
        - Zero denominator: returns None with warning flag
        - Negative denominator: rejected with ValueError
        - Partial scope data: flagged with coverage percentage
        - Missing period: gap in time series

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 6
    - GRI 305-4: GHG emissions intensity
    - ESRS E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions
    - CDP Climate Change C6.10: Emissions intensities
    - SEC Climate Disclosure Rule (2024), Item 1504(c)(1)
    - ISO 14064-1:2018 Clause 5.3.4 (GHG intensity metrics)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Formula: I = E / D (no approximation, no rounding until output)
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    default: Optional[Decimal] = None,
) -> Optional[Decimal]:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _round6(value: Any) -> float:
    """Round to 6 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScopeInclusion(str, Enum):
    """Defines which emission scopes to include in the intensity numerator.

    SCOPE_1_ONLY:       Only direct (Scope 1) emissions.
    SCOPE_2_LOCATION:   Only Scope 2 location-based emissions.
    SCOPE_2_MARKET:     Only Scope 2 market-based emissions.
    SCOPE_1_2_LOCATION: Scope 1 + Scope 2 location-based.
    SCOPE_1_2_MARKET:   Scope 1 + Scope 2 market-based.
    SCOPE_1_2_3:        Scope 1 + Scope 2 location + Scope 3 (all).
    SCOPE_3_SPECIFIC:   Specific Scope 3 categories only.
    CUSTOM:             Custom combination of emission components.
    """
    SCOPE_1_ONLY = "scope_1_only"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_1_2_LOCATION = "scope_1_2_location"
    SCOPE_1_2_MARKET = "scope_1_2_market"
    SCOPE_1_2_3 = "scope_1_2_3"
    SCOPE_3_SPECIFIC = "scope_3_specific"
    CUSTOM = "custom"


class IntensityStatus(str, Enum):
    """Status of an intensity calculation result.

    VALID:              Calculation completed successfully.
    ZERO_DENOMINATOR:   Denominator was zero; intensity is undefined.
    PARTIAL_DATA:       Not all scope data was available.
    MISSING_DATA:       Critical data missing; could not calculate.
    ERROR:              Calculation error occurred.
    """
    VALID = "valid"
    ZERO_DENOMINATOR = "zero_denominator"
    PARTIAL_DATA = "partial_data"
    MISSING_DATA = "missing_data"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default output precision (decimal places for intensity result)
DEFAULT_PRECISION: int = 6

# Maximum number of entities for consolidated calculation
MAX_ENTITIES: int = 10000

# Maximum number of periods in a time series
MAX_PERIODS: int = 100

# Scope 3 category names (1-15 per GHG Protocol)
SCOPE_3_CATEGORIES: Dict[int, str] = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class EmissionsData(BaseModel):
    """Emissions data for a single entity and period.

    Attributes:
        scope_1_tco2e:           Scope 1 direct emissions (tCO2e).
        scope_2_location_tco2e:  Scope 2 location-based emissions (tCO2e).
        scope_2_market_tco2e:    Scope 2 market-based emissions (tCO2e).
        scope_3_tco2e:           Total Scope 3 emissions (tCO2e).
        scope_3_categories:      Scope 3 by category (category number -> tCO2e).
        custom_components:       Custom emission components (name -> tCO2e).
    """
    scope_1_tco2e: Optional[Decimal] = Field(default=None, ge=0, description="Scope 1 (tCO2e)")
    scope_2_location_tco2e: Optional[Decimal] = Field(
        default=None, ge=0, description="Scope 2 location (tCO2e)"
    )
    scope_2_market_tco2e: Optional[Decimal] = Field(
        default=None, ge=0, description="Scope 2 market (tCO2e)"
    )
    scope_3_tco2e: Optional[Decimal] = Field(default=None, ge=0, description="Scope 3 total (tCO2e)")
    scope_3_categories: Dict[int, Decimal] = Field(
        default_factory=dict, description="Scope 3 by category"
    )
    custom_components: Dict[str, Decimal] = Field(
        default_factory=dict, description="Custom emission components"
    )

    @field_validator(
        "scope_1_tco2e", "scope_2_location_tco2e",
        "scope_2_market_tco2e", "scope_3_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        """Coerce optional numeric fields to Decimal."""
        if v is None:
            return None
        return _decimal(v)


class IntensityInput(BaseModel):
    """Input for a single intensity calculation.

    Attributes:
        entity_id:          Entity identifier.
        period:             Reporting period (e.g. '2024').
        emissions:          Emissions data.
        denominator_value:  Denominator value (must be > 0).
        denominator_unit:   Denominator unit.
        denominator_id:     Denominator type identifier.
        scope_inclusion:    Which scopes to include.
        scope_3_categories: Specific Scope 3 categories (for SCOPE_3_SPECIFIC).
        output_precision:   Decimal places for output.
    """
    entity_id: str = Field(default="default", description="Entity identifier")
    period: str = Field(..., min_length=1, max_length=20, description="Reporting period")
    emissions: EmissionsData = Field(..., description="Emissions data")
    denominator_value: Decimal = Field(..., description="Denominator value")
    denominator_unit: str = Field(default="unit", description="Denominator unit")
    denominator_id: str = Field(default="", description="Denominator type ID")
    scope_inclusion: ScopeInclusion = Field(
        default=ScopeInclusion.SCOPE_1_2_LOCATION,
        description="Scope inclusion configuration"
    )
    scope_3_categories: List[int] = Field(
        default_factory=list, description="Scope 3 categories to include"
    )
    output_precision: int = Field(
        default=DEFAULT_PRECISION, ge=0, le=12,
        description="Output decimal places"
    )

    @field_validator("denominator_value", mode="before")
    @classmethod
    def coerce_denominator(cls, v: Any) -> Decimal:
        """Coerce denominator to Decimal."""
        return _decimal(v)

    @model_validator(mode="after")
    def check_denominator(self) -> "IntensityInput":
        """Validate denominator is not negative."""
        if self.denominator_value < Decimal("0"):
            raise ValueError(
                f"Denominator value must be non-negative (got {self.denominator_value})"
            )
        return self


class EntityIntensityInput(BaseModel):
    """Input for a single entity in a multi-entity consolidation.

    Attributes:
        entity_id:         Entity identifier.
        entity_name:       Human-readable entity name.
        emissions_tco2e:   Total emissions for selected scopes (tCO2e).
        denominator_value: Denominator value.
        weight:            Optional weight for consolidation.
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    emissions_tco2e: Decimal = Field(..., ge=0, description="Emissions (tCO2e)")
    denominator_value: Decimal = Field(..., description="Denominator value")
    weight: Decimal = Field(default=Decimal("1"), ge=0, description="Consolidation weight")

    @field_validator("emissions_tco2e", "denominator_value", "weight", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        return _decimal(v)


class TimeSeriesInput(BaseModel):
    """Input for time-series intensity calculation.

    Attributes:
        entity_id:     Entity identifier.
        periods:       List of IntensityInput for each period.
        base_period:   Base period for cumulative change (default: first).
    """
    entity_id: str = Field(default="default", description="Entity ID")
    periods: List[IntensityInput] = Field(..., min_length=1, description="Period inputs")
    base_period: Optional[str] = Field(default=None, description="Base period label")


class ConsolidationInput(BaseModel):
    """Input for multi-entity consolidated intensity.

    Attributes:
        consolidation_id:   Consolidation group identifier.
        period:             Reporting period.
        entities:           Entity-level inputs.
        denominator_unit:   Denominator unit.
        denominator_id:     Denominator type ID.
        output_precision:   Output decimal places.
    """
    consolidation_id: str = Field(default="", description="Consolidation ID")
    period: str = Field(..., description="Reporting period")
    entities: List[EntityIntensityInput] = Field(..., min_length=1, description="Entity inputs")
    denominator_unit: str = Field(default="unit", description="Denominator unit")
    denominator_id: str = Field(default="", description="Denominator type ID")
    output_precision: int = Field(default=DEFAULT_PRECISION, ge=0, le=12, description="Precision")


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class IntensityResult(BaseModel):
    """Result of a single intensity calculation.

    Attributes:
        result_id:          Unique result identifier.
        entity_id:          Entity identifier.
        period:             Reporting period.
        intensity_value:    Calculated intensity (tCO2e per unit), or None.
        intensity_unit:     Intensity unit string.
        numerator_tco2e:    Total emissions used as numerator.
        denominator_value:  Denominator value used.
        denominator_unit:   Denominator unit.
        scope_inclusion:    Scope configuration used.
        scope_coverage_pct: Percentage of requested scope data available.
        status:             Calculation status.
        warnings:           Warnings and advisory notes.
        calculated_at:      Calculation timestamp.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash:    SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entity_id: str = Field(default="", description="Entity ID")
    period: str = Field(default="", description="Reporting period")
    intensity_value: Optional[Decimal] = Field(default=None, description="Intensity (tCO2e/unit)")
    intensity_unit: str = Field(default="tCO2e/unit", description="Intensity unit")
    numerator_tco2e: Decimal = Field(default=Decimal("0"), description="Numerator emissions")
    denominator_value: Decimal = Field(default=Decimal("0"), description="Denominator value")
    denominator_unit: str = Field(default="unit", description="Denominator unit")
    scope_inclusion: ScopeInclusion = Field(
        default=ScopeInclusion.SCOPE_1_2_LOCATION, description="Scope config"
    )
    scope_coverage_pct: Decimal = Field(default=Decimal("100"), description="Scope coverage (%)")
    status: IntensityStatus = Field(default=IntensityStatus.VALID, description="Status")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp (ISO 8601)")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class PeriodIntensity(BaseModel):
    """Intensity for a single period in a time series.

    Attributes:
        period:           Reporting period label.
        intensity_value:  Intensity value or None.
        numerator_tco2e:  Emissions numerator.
        denominator_value: Denominator value.
        yoy_change_pct:   Year-over-year change (%).
        cum_change_pct:   Cumulative change from base period (%).
        status:           Calculation status.
    """
    period: str = Field(..., description="Period label")
    intensity_value: Optional[Decimal] = Field(default=None, description="Intensity")
    numerator_tco2e: Decimal = Field(default=Decimal("0"), description="Numerator")
    denominator_value: Decimal = Field(default=Decimal("0"), description="Denominator")
    yoy_change_pct: Optional[Decimal] = Field(default=None, description="YoY change (%)")
    cum_change_pct: Optional[Decimal] = Field(default=None, description="Cumulative change (%)")
    status: IntensityStatus = Field(default=IntensityStatus.VALID, description="Status")


class IntensityTimeSeries(BaseModel):
    """Time series of intensity values with trend metrics.

    Attributes:
        result_id:          Unique result identifier.
        entity_id:          Entity identifier.
        base_period:        Base period for cumulative changes.
        periods:            Period-level intensity values.
        total_change_pct:   Total change from first to last period (%).
        period_count:       Number of periods.
        valid_period_count: Number of periods with valid intensity.
        warnings:           Warnings.
        calculated_at:      Timestamp.
        processing_time_ms: Processing time (ms).
        provenance_hash:    SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entity_id: str = Field(default="", description="Entity ID")
    base_period: str = Field(default="", description="Base period")
    periods: List[PeriodIntensity] = Field(default_factory=list, description="Period data")
    total_change_pct: Optional[Decimal] = Field(default=None, description="Total change (%)")
    period_count: int = Field(default=0, description="Period count")
    valid_period_count: int = Field(default=0, description="Valid period count")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class EntityContribution(BaseModel):
    """An entity's contribution to a consolidated intensity.

    Attributes:
        entity_id:           Entity identifier.
        entity_name:         Entity name.
        entity_intensity:    Per-entity intensity.
        emissions_share_pct: Share of total emissions (%).
        denominator_share_pct: Share of total denominator (%).
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    entity_intensity: Optional[Decimal] = Field(default=None, description="Entity intensity")
    emissions_share_pct: Decimal = Field(default=Decimal("0"), description="Emissions share (%)")
    denominator_share_pct: Decimal = Field(default=Decimal("0"), description="Denominator share (%)")


class ConsolidatedIntensity(BaseModel):
    """Result of multi-entity consolidated intensity calculation.

    Attributes:
        result_id:              Unique result identifier.
        consolidation_id:       Consolidation group ID.
        period:                 Reporting period.
        consolidated_intensity: Weighted average intensity.
        total_emissions_tco2e:  Sum of all entity emissions.
        total_denominator:      Sum of all entity denominators.
        denominator_unit:       Denominator unit.
        entity_count:           Number of entities.
        entity_contributions:   Per-entity breakdown.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    consolidation_id: str = Field(default="", description="Consolidation ID")
    period: str = Field(default="", description="Period")
    consolidated_intensity: Optional[Decimal] = Field(
        default=None, description="Consolidated intensity"
    )
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Total emissions")
    total_denominator: Decimal = Field(default=Decimal("0"), description="Total denominator")
    denominator_unit: str = Field(default="unit", description="Denominator unit")
    entity_count: int = Field(default=0, description="Entity count")
    entity_contributions: List[EntityContribution] = Field(
        default_factory=list, description="Entity contributions"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IntensityCalculationEngine:
    """Core GHG emissions intensity calculation engine.

    Implements the fundamental intensity formula I = E / D with full
    scope configuration, multi-entity weighted consolidation, and
    time-series support.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Complete calculation trace with numerator/denominator.
        - Zero-Hallucination: No LLM in any calculation path.

    Usage:
        engine = IntensityCalculationEngine()
        inp = IntensityInput(
            period="2024",
            emissions=EmissionsData(scope_1_tco2e=1000, scope_2_location_tco2e=500),
            denominator_value=50,
            denominator_unit="USD_million",
            scope_inclusion=ScopeInclusion.SCOPE_1_2_LOCATION,
        )
        result = engine.calculate(inp)
        print(result.intensity_value)  # 30.000000
    """

    def __init__(self) -> None:
        """Initialise the IntensityCalculationEngine."""
        self._version = _MODULE_VERSION
        logger.info("IntensityCalculationEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: IntensityInput) -> IntensityResult:
        """Calculate emissions intensity for a single entity and period.

        Formula:
            intensity = numerator_tco2e / denominator_value

        Args:
            input_data: Intensity calculation input.

        Returns:
            IntensityResult with intensity value and provenance.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # Resolve numerator
        numerator, coverage_pct, scope_warnings = self._resolve_numerator(
            input_data.emissions,
            input_data.scope_inclusion,
            input_data.scope_3_categories,
        )
        warnings.extend(scope_warnings)

        # Check denominator
        denominator = input_data.denominator_value
        status = IntensityStatus.VALID

        if denominator == Decimal("0"):
            status = IntensityStatus.ZERO_DENOMINATOR
            warnings.append("Denominator is zero; intensity is undefined.")
            intensity_value = None
        elif denominator < Decimal("0"):
            raise ValueError(
                f"Denominator must be non-negative (got {denominator})"
            )
        else:
            intensity_value = numerator / denominator
            precision_str = "0." + "0" * input_data.output_precision
            intensity_value = intensity_value.quantize(
                Decimal(precision_str), rounding=ROUND_HALF_UP
            )

        if coverage_pct < Decimal("100") and status == IntensityStatus.VALID:
            status = IntensityStatus.PARTIAL_DATA
            warnings.append(
                f"Scope data coverage: {_round2(coverage_pct)}%. "
                f"Some requested scope data was not available."
            )

        intensity_unit = f"tCO2e/{input_data.denominator_unit}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = IntensityResult(
            entity_id=input_data.entity_id,
            period=input_data.period,
            intensity_value=intensity_value,
            intensity_unit=intensity_unit,
            numerator_tco2e=numerator,
            denominator_value=denominator,
            denominator_unit=input_data.denominator_unit,
            scope_inclusion=input_data.scope_inclusion,
            scope_coverage_pct=coverage_pct,
            status=status,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_time_series(self, ts_input: TimeSeriesInput) -> IntensityTimeSeries:
        """Calculate intensity across multiple periods.

        Computes per-period intensity, year-over-year change, and
        cumulative change from base period.

        Args:
            ts_input: Time series input with multiple period inputs.

        Returns:
            IntensityTimeSeries with period-level data.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        if len(ts_input.periods) > MAX_PERIODS:
            raise ValueError(
                f"Maximum {MAX_PERIODS} periods allowed (got {len(ts_input.periods)})"
            )

        # Calculate each period
        period_results: List[PeriodIntensity] = []
        base_intensity: Optional[Decimal] = None
        prev_intensity: Optional[Decimal] = None

        # Determine base period
        base_period = ts_input.base_period or ts_input.periods[0].period

        for period_input in ts_input.periods:
            result = self.calculate(period_input)

            yoy_change: Optional[Decimal] = None
            cum_change: Optional[Decimal] = None

            if result.intensity_value is not None:
                # Set base intensity
                if period_input.period == base_period:
                    base_intensity = result.intensity_value

                # YoY change
                if prev_intensity is not None and prev_intensity != Decimal("0"):
                    yoy_change = (
                        (result.intensity_value - prev_intensity)
                        / prev_intensity * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                # Cumulative change from base
                if base_intensity is not None and base_intensity != Decimal("0"):
                    cum_change = (
                        (result.intensity_value - base_intensity)
                        / base_intensity * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                prev_intensity = result.intensity_value

            period_results.append(PeriodIntensity(
                period=period_input.period,
                intensity_value=result.intensity_value,
                numerator_tco2e=result.numerator_tco2e,
                denominator_value=result.denominator_value,
                yoy_change_pct=yoy_change,
                cum_change_pct=cum_change,
                status=result.status,
            ))

        # Total change
        valid_periods = [p for p in period_results if p.intensity_value is not None]
        total_change: Optional[Decimal] = None
        if len(valid_periods) >= 2:
            first = valid_periods[0].intensity_value
            last = valid_periods[-1].intensity_value
            if first is not None and last is not None and first != Decimal("0"):
                total_change = (
                    (last - first) / first * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ts_result = IntensityTimeSeries(
            entity_id=ts_input.entity_id,
            base_period=base_period,
            periods=period_results,
            total_change_pct=total_change,
            period_count=len(period_results),
            valid_period_count=len(valid_periods),
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        ts_result.provenance_hash = _compute_hash(ts_result)
        return ts_result

    def calculate_consolidated(
        self,
        consolidation_input: ConsolidationInput,
    ) -> ConsolidatedIntensity:
        """Calculate multi-entity consolidated intensity.

        Uses the correct weighted-average formula:
            I_consolidated = SUM(entity_emissions) / SUM(entity_denominators)

        This is NOT the average of per-entity intensities.

        Args:
            consolidation_input: Multi-entity input.

        Returns:
            ConsolidatedIntensity with breakdown.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        entities = consolidation_input.entities

        if len(entities) > MAX_ENTITIES:
            raise ValueError(
                f"Maximum {MAX_ENTITIES} entities allowed (got {len(entities)})"
            )

        # Weighted sums
        total_emissions = Decimal("0")
        total_denominator = Decimal("0")
        contributions: List[EntityContribution] = []

        for entity in entities:
            weighted_emissions = entity.emissions_tco2e * entity.weight
            weighted_denom = entity.denominator_value * entity.weight
            total_emissions += weighted_emissions
            total_denominator += weighted_denom

        # Per-entity contributions
        for entity in entities:
            weighted_emissions = entity.emissions_tco2e * entity.weight
            weighted_denom = entity.denominator_value * entity.weight

            entity_intensity: Optional[Decimal] = None
            if entity.denominator_value > Decimal("0"):
                entity_intensity = (
                    entity.emissions_tco2e / entity.denominator_value
                ).quantize(
                    Decimal("0." + "0" * consolidation_input.output_precision),
                    rounding=ROUND_HALF_UP,
                )

            emissions_share = Decimal("0")
            if total_emissions > Decimal("0"):
                emissions_share = (
                    weighted_emissions / total_emissions * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            denom_share = Decimal("0")
            if total_denominator > Decimal("0"):
                denom_share = (
                    weighted_denom / total_denominator * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            contributions.append(EntityContribution(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_intensity=entity_intensity,
                emissions_share_pct=emissions_share,
                denominator_share_pct=denom_share,
            ))

        # Consolidated intensity
        consolidated: Optional[Decimal] = None
        if total_denominator > Decimal("0"):
            precision_str = "0." + "0" * consolidation_input.output_precision
            consolidated = (total_emissions / total_denominator).quantize(
                Decimal(precision_str), rounding=ROUND_HALF_UP
            )
        else:
            warnings.append(
                "Total denominator is zero across all entities; "
                "consolidated intensity is undefined."
            )

        # Check for zero-denominator entities
        zero_denom_entities = [
            e.entity_id for e in entities if e.denominator_value == Decimal("0")
        ]
        if zero_denom_entities:
            warnings.append(
                f"Entities with zero denominator: {zero_denom_entities}"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ConsolidatedIntensity(
            consolidation_id=consolidation_input.consolidation_id,
            period=consolidation_input.period,
            consolidated_intensity=consolidated,
            total_emissions_tco2e=total_emissions,
            total_denominator=total_denominator,
            denominator_unit=consolidation_input.denominator_unit,
            entity_count=len(entities),
            entity_contributions=contributions,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        inputs: List[IntensityInput],
    ) -> List[IntensityResult]:
        """Calculate intensity for a batch of inputs.

        Args:
            inputs: List of IntensityInput objects.

        Returns:
            List of IntensityResult objects in the same order.
        """
        return [self.calculate(inp) for inp in inputs]

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _resolve_numerator(
        self,
        emissions: EmissionsData,
        scope_inclusion: ScopeInclusion,
        scope_3_cats: List[int],
    ) -> Tuple[Decimal, Decimal, List[str]]:
        """Resolve the emissions numerator based on scope configuration.

        Returns:
            Tuple of (numerator_tco2e, coverage_pct, warnings).
        """
        numerator = Decimal("0")
        requested = 0
        available = 0
        warnings: List[str] = []

        if scope_inclusion == ScopeInclusion.SCOPE_1_ONLY:
            requested = 1
            if emissions.scope_1_tco2e is not None:
                numerator = emissions.scope_1_tco2e
                available = 1
            else:
                warnings.append("Scope 1 data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_2_LOCATION:
            requested = 1
            if emissions.scope_2_location_tco2e is not None:
                numerator = emissions.scope_2_location_tco2e
                available = 1
            else:
                warnings.append("Scope 2 location-based data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_2_MARKET:
            requested = 1
            if emissions.scope_2_market_tco2e is not None:
                numerator = emissions.scope_2_market_tco2e
                available = 1
            else:
                warnings.append("Scope 2 market-based data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_1_2_LOCATION:
            requested = 2
            if emissions.scope_1_tco2e is not None:
                numerator += emissions.scope_1_tco2e
                available += 1
            else:
                warnings.append("Scope 1 data not available.")
            if emissions.scope_2_location_tco2e is not None:
                numerator += emissions.scope_2_location_tco2e
                available += 1
            else:
                warnings.append("Scope 2 location-based data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_1_2_MARKET:
            requested = 2
            if emissions.scope_1_tco2e is not None:
                numerator += emissions.scope_1_tco2e
                available += 1
            else:
                warnings.append("Scope 1 data not available.")
            if emissions.scope_2_market_tco2e is not None:
                numerator += emissions.scope_2_market_tco2e
                available += 1
            else:
                warnings.append("Scope 2 market-based data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_1_2_3:
            requested = 3
            if emissions.scope_1_tco2e is not None:
                numerator += emissions.scope_1_tco2e
                available += 1
            else:
                warnings.append("Scope 1 data not available.")
            if emissions.scope_2_location_tco2e is not None:
                numerator += emissions.scope_2_location_tco2e
                available += 1
            else:
                warnings.append("Scope 2 location-based data not available.")
            if emissions.scope_3_tco2e is not None:
                numerator += emissions.scope_3_tco2e
                available += 1
            else:
                warnings.append("Scope 3 total data not available.")

        elif scope_inclusion == ScopeInclusion.SCOPE_3_SPECIFIC:
            if not scope_3_cats:
                warnings.append("SCOPE_3_SPECIFIC selected but no categories specified.")
                requested = 1
            else:
                requested = len(scope_3_cats)
                for cat in scope_3_cats:
                    if cat in emissions.scope_3_categories:
                        numerator += _decimal(emissions.scope_3_categories[cat])
                        available += 1
                    else:
                        cat_name = SCOPE_3_CATEGORIES.get(cat, f"Category {cat}")
                        warnings.append(
                            f"Scope 3 {cat_name} (cat {cat}) data not available."
                        )

        elif scope_inclusion == ScopeInclusion.CUSTOM:
            if not emissions.custom_components:
                warnings.append("CUSTOM scope selected but no components provided.")
                requested = 1
            else:
                requested = len(emissions.custom_components)
                for name, value in emissions.custom_components.items():
                    numerator += _decimal(value)
                    available += 1

        # Coverage percentage
        coverage_pct = Decimal("100")
        if requested > 0:
            coverage_pct = (
                Decimal(str(available)) / Decimal(str(requested)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return numerator, coverage_pct, warnings

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """Return engine version string."""
        return self._version

    def get_scope_inclusions(self) -> List[str]:
        """Return list of available scope inclusion options."""
        return [si.value for si in ScopeInclusion]

    def get_scope_3_categories(self) -> Dict[int, str]:
        """Return Scope 3 category names."""
        return dict(SCOPE_3_CATEGORIES)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def calculate_intensity(
    emissions_tco2e: Decimal,
    denominator_value: Decimal,
    precision: int = DEFAULT_PRECISION,
) -> Optional[Decimal]:
    """Simple intensity calculation without full engine setup.

    Args:
        emissions_tco2e:  Total emissions (tCO2e).
        denominator_value: Denominator value.
        precision:         Output decimal places.

    Returns:
        Intensity as Decimal, or None if denominator is zero.
    """
    emissions = _decimal(emissions_tco2e)
    denom = _decimal(denominator_value)
    if denom == Decimal("0"):
        return None
    if denom < Decimal("0"):
        raise ValueError(f"Denominator must be non-negative (got {denom})")
    precision_str = "0." + "0" * precision
    return (emissions / denom).quantize(Decimal(precision_str), rounding=ROUND_HALF_UP)


def calculate_consolidated_intensity(
    entities: List[Tuple[Decimal, Decimal]],
    precision: int = DEFAULT_PRECISION,
) -> Optional[Decimal]:
    """Simple consolidated intensity: SUM(emissions) / SUM(denominators).

    Args:
        entities:  List of (emissions_tco2e, denominator_value) tuples.
        precision: Output decimal places.

    Returns:
        Consolidated intensity, or None if total denominator is zero.
    """
    total_e = sum(_decimal(e) for e, _ in entities)
    total_d = sum(_decimal(d) for _, d in entities)
    if total_d == Decimal("0"):
        return None
    precision_str = "0." + "0" * precision
    return (total_e / total_d).quantize(Decimal(precision_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ScopeInclusion",
    "IntensityStatus",
    # Input Models
    "EmissionsData",
    "IntensityInput",
    "EntityIntensityInput",
    "TimeSeriesInput",
    "ConsolidationInput",
    # Output Models
    "IntensityResult",
    "PeriodIntensity",
    "IntensityTimeSeries",
    "EntityContribution",
    "ConsolidatedIntensity",
    # Engine
    "IntensityCalculationEngine",
    # Convenience
    "calculate_intensity",
    "calculate_consolidated_intensity",
    # Constants
    "SCOPE_3_CATEGORIES",
    "DEFAULT_PRECISION",
]
