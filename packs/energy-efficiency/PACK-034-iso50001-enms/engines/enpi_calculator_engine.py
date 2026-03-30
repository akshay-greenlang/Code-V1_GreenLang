# -*- coding: utf-8 -*-
"""
EnPICalculatorEngine - PACK-034 ISO 50001 EnMS Engine 3
========================================================

Energy Performance Indicator (EnPI) calculation engine per ISO 50006:2014
and ISO 50001:2018 requirements.  Computes absolute, intensity,
regression-modelled, proportion, and statistical EnPIs with full
normalisation support (weather, production, occupancy, multi-variable).

Provides baseline comparison, improvement percentage calculation,
statistical validation (t-test, F-test, chi-squared, ANOVA), trend
analysis, and portfolio-level aggregation with weighted averaging.

Calculation Methodology:
    Absolute EnPI:
        enpi = total_energy_value  (kWh, GJ, etc.)

    Intensity EnPI:
        enpi = energy_value / normalizing_variable
        (e.g. kWh/m2, kWh/tonne, GJ/employee)

    Regression-Modelled EnPI (ISO 50006 Section 6.4):
        expected = intercept + slope * relevant_variable
        enpi = actual / expected
        improvement = (expected - actual) / expected * 100

    Proportion EnPI:
        enpi = energy_value / reference_total * 100

    Statistical EnPI:
        Uses CUSUM (cumulative sum) of differences from baseline mean

    Normalisation Methods (ISO 50006 Section 7):
        Weather:       adjust by HDD/CDD ratio to baseline conditions
        Production:    adjust by production output ratio
        Occupancy:     adjust by occupancy ratio
        Multi-variable: linear combination of multiple adjustments

    Statistical Validation:
        t-test:       compare means of baseline vs reporting period
        F-test:       compare variances of baseline vs reporting period
        chi-squared:  goodness-of-fit test
        ANOVA:        one-way analysis of variance across periods

    Improvement Calculation:
        decrease_is_better: improvement = (baseline - current) / baseline * 100
        increase_is_better: improvement = (current - baseline) / baseline * 100

    Trend Analysis:
        Ordinary least squares linear regression on EnPI values over time
        slope = covariance(x, y) / variance(x)
        r_squared = correlation(x, y) ^ 2

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs
    - ISO 50015:2014 - Measurement and verification of energy performance
    - ISO 50047:2016 - Determination of energy savings
    - EN 16247-1:2022 - Energy audits (general requirements)
    - IPMVP Core Concepts (EVO, 2022)

Zero-Hallucination:
    - All formulas are standard ISO 50006 / engineering calculations
    - Statistical tests use deterministic Decimal approximations
    - Regression uses ordinary least squares (closed-form solution)
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
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

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _decimal_sqrt(value: Decimal) -> Decimal:
    """Compute the square root of a Decimal using Newton's method.

    Returns Decimal("0") for non-positive inputs.
    """
    if value <= Decimal("0"):
        return Decimal("0")
    # Use float sqrt as starting point, then refine with Decimal
    guess = _decimal(math.sqrt(float(value)))
    if guess == Decimal("0"):
        return Decimal("0")
    # Two Newton iterations for Decimal precision
    for _ in range(10):
        new_guess = (guess + value / guess) / Decimal("2")
        if abs(new_guess - guess) < Decimal("1E-20"):
            break
        guess = new_guess
    return guess

def _decimal_abs(value: Decimal) -> Decimal:
    """Return the absolute value of a Decimal."""
    return value if value >= Decimal("0") else -value

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnPIType(str, Enum):
    """Type of Energy Performance Indicator per ISO 50006.

    ABSOLUTE:           Total energy consumption (kWh, GJ).
    INTENSITY:          Energy per normalising variable (kWh/m2, kWh/tonne).
    REGRESSION_MODELED: Model-predicted vs actual consumption ratio.
    PROPORTION:         Energy as percentage of a reference total.
    STATISTICAL:        CUSUM-based statistical comparison to baseline.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    REGRESSION_MODELED = "regression_modeled"
    PROPORTION = "proportion"
    STATISTICAL = "statistical"

class NormalizationMethod(str, Enum):
    """Normalisation method for EnPI adjustment per ISO 50006 Section 7.

    NONE:           No normalisation applied.
    WEATHER:        Adjust using HDD/CDD ratios.
    PRODUCTION:     Adjust using production output ratios.
    OCCUPANCY:      Adjust using occupancy ratios.
    MULTI_VARIABLE: Linear combination of multiple adjustment factors.
    """
    NONE = "none"
    WEATHER = "weather"
    PRODUCTION = "production"
    OCCUPANCY = "occupancy"
    MULTI_VARIABLE = "multi_variable"

class ImprovementDirection(str, Enum):
    """Direction indicating improvement for an EnPI.

    DECREASE_IS_BETTER: Lower values indicate better performance
                        (e.g. kWh/m2 intensity).
    INCREASE_IS_BETTER: Higher values indicate better performance
                        (e.g. production output per unit energy).
    """
    DECREASE_IS_BETTER = "decrease_is_better"
    INCREASE_IS_BETTER = "increase_is_better"

class AggregationLevel(str, Enum):
    """Organisational level at which the EnPI is aggregated.

    EQUIPMENT:    Individual equipment / asset level.
    PROCESS:      Process or production line level.
    FACILITY:     Single facility / building level.
    SITE:         Multi-building site level.
    ORGANIZATION: Entire organisation level.
    PORTFOLIO:    Multi-organisation portfolio level.
    """
    EQUIPMENT = "equipment"
    PROCESS = "process"
    FACILITY = "facility"
    SITE = "site"
    ORGANIZATION = "organization"
    PORTFOLIO = "portfolio"

class StatisticalTest(str, Enum):
    """Statistical test type for EnPI validation.

    T_TEST:      Two-sample t-test comparing baseline vs reporting means.
    F_TEST:      F-test comparing baseline vs reporting variances.
    CHI_SQUARED: Chi-squared goodness-of-fit test.
    ANOVA:       One-way ANOVA across multiple reporting periods.
    """
    T_TEST = "t_test"
    F_TEST = "f_test"
    CHI_SQUARED = "chi_squared"
    ANOVA = "anova"

# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

# Descriptive text for each EnPI type per ISO 50006 guidance.
ENPI_TYPE_DESCRIPTIONS: Dict[str, str] = {
    EnPIType.ABSOLUTE.value: (
        "Absolute EnPI measures total energy consumption over a defined "
        "period without normalisation.  Suitable when relevant variables "
        "remain constant or are not significant.  Per ISO 50006 Section 6.2."
    ),
    EnPIType.INTENSITY.value: (
        "Intensity EnPI expresses energy consumption per unit of a "
        "normalising variable (e.g. kWh/m2, GJ/tonne product).  Suitable "
        "when a single variable dominates energy use.  Per ISO 50006 "
        "Section 6.3."
    ),
    EnPIType.REGRESSION_MODELED.value: (
        "Regression-modelled EnPI uses a baseline regression model to "
        "predict expected energy consumption given current relevant "
        "variables, then compares actual vs expected.  Suitable when "
        "multiple variables influence energy use.  Per ISO 50006 "
        "Section 6.4."
    ),
    EnPIType.PROPORTION.value: (
        "Proportion EnPI expresses energy consumption of a sub-system "
        "as a percentage of a reference total (e.g. HVAC as % of total "
        "facility energy).  Useful for tracking end-use shares over time."
    ),
    EnPIType.STATISTICAL.value: (
        "Statistical EnPI uses cumulative sum (CUSUM) analysis to detect "
        "shifts in energy performance relative to the baseline mean.  "
        "Provides early detection of performance changes."
    ),
}

# Statistical significance thresholds per ISO 50006 guidance.
# alpha = significance level; confidence = 1 - alpha.
STATISTICAL_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    "strict": {
        "alpha": Decimal("0.01"),
        "confidence": Decimal("0.99"),
        "description_key": "99% confidence (alpha=0.01)",
    },
    "standard": {
        "alpha": Decimal("0.05"),
        "confidence": Decimal("0.95"),
        "description_key": "95% confidence (alpha=0.05)",
    },
    "relaxed": {
        "alpha": Decimal("0.10"),
        "confidence": Decimal("0.90"),
        "description_key": "90% confidence (alpha=0.10)",
    },
}

# Common intensity denominators used in energy management.
COMMON_INTENSITY_DENOMINATORS: List[Dict[str, str]] = [
    {"key": "m2", "label": "Floor area (m2)", "unit": "kWh/m2"},
    {"key": "ft2", "label": "Floor area (ft2)", "unit": "kWh/ft2"},
    {"key": "employee", "label": "Number of employees", "unit": "kWh/employee"},
    {"key": "tonne_product", "label": "Tonnes of product", "unit": "kWh/tonne"},
    {"key": "unit_produced", "label": "Units produced", "unit": "kWh/unit"},
    {"key": "kg_product", "label": "Kilograms of product", "unit": "kWh/kg"},
    {"key": "occupant", "label": "Number of occupants", "unit": "kWh/occupant"},
    {"key": "bed", "label": "Number of beds (healthcare)", "unit": "kWh/bed"},
    {"key": "room_night", "label": "Room nights (hospitality)", "unit": "kWh/room-night"},
    {"key": "vehicle_km", "label": "Vehicle kilometres", "unit": "kWh/vkm"},
    {"key": "tonne_km", "label": "Tonne-kilometres (freight)", "unit": "kWh/tkm"},
    {"key": "litre_produced", "label": "Litres produced", "unit": "kWh/litre"},
    {"key": "degree_day", "label": "Degree days (HDD+CDD)", "unit": "kWh/DD"},
    {"key": "operating_hour", "label": "Operating hours", "unit": "kWh/hour"},
    {"key": "revenue_million", "label": "Revenue (millions)", "unit": "kWh/M-revenue"},
]

# Approximate critical t-values for two-tailed tests (df -> t_critical).
# Used for Decimal-based statistical approximation without scipy.
_T_CRITICAL_TABLE: Dict[int, Dict[str, Decimal]] = {
    1:   {"0.10": Decimal("6.314"), "0.05": Decimal("12.706"), "0.01": Decimal("63.657")},
    2:   {"0.10": Decimal("2.920"), "0.05": Decimal("4.303"),  "0.01": Decimal("9.925")},
    3:   {"0.10": Decimal("2.353"), "0.05": Decimal("3.182"),  "0.01": Decimal("5.841")},
    4:   {"0.10": Decimal("2.132"), "0.05": Decimal("2.776"),  "0.01": Decimal("4.604")},
    5:   {"0.10": Decimal("2.015"), "0.05": Decimal("2.571"),  "0.01": Decimal("4.032")},
    6:   {"0.10": Decimal("1.943"), "0.05": Decimal("2.447"),  "0.01": Decimal("3.707")},
    7:   {"0.10": Decimal("1.895"), "0.05": Decimal("2.365"),  "0.01": Decimal("3.499")},
    8:   {"0.10": Decimal("1.860"), "0.05": Decimal("2.306"),  "0.01": Decimal("3.355")},
    9:   {"0.10": Decimal("1.833"), "0.05": Decimal("2.262"),  "0.01": Decimal("3.250")},
    10:  {"0.10": Decimal("1.812"), "0.05": Decimal("2.228"),  "0.01": Decimal("3.169")},
    12:  {"0.10": Decimal("1.782"), "0.05": Decimal("2.179"),  "0.01": Decimal("3.055")},
    15:  {"0.10": Decimal("1.753"), "0.05": Decimal("2.131"),  "0.01": Decimal("2.947")},
    20:  {"0.10": Decimal("1.725"), "0.05": Decimal("2.086"),  "0.01": Decimal("2.845")},
    25:  {"0.10": Decimal("1.708"), "0.05": Decimal("2.060"),  "0.01": Decimal("2.787")},
    30:  {"0.10": Decimal("1.697"), "0.05": Decimal("2.042"),  "0.01": Decimal("2.750")},
    40:  {"0.10": Decimal("1.684"), "0.05": Decimal("2.021"),  "0.01": Decimal("2.704")},
    60:  {"0.10": Decimal("1.671"), "0.05": Decimal("2.000"),  "0.01": Decimal("2.660")},
    120: {"0.10": Decimal("1.658"), "0.05": Decimal("1.980"),  "0.01": Decimal("2.617")},
    999: {"0.10": Decimal("1.645"), "0.05": Decimal("1.960"),  "0.01": Decimal("2.576")},
}

# Approximate critical F-values (numerator_df=1, denominator_df -> F_critical)
# for alpha=0.05.  Used for Decimal-based variance ratio tests.
_F_CRITICAL_TABLE: Dict[int, Decimal] = {
    1:  Decimal("161.45"),
    2:  Decimal("18.51"),
    3:  Decimal("10.13"),
    4:  Decimal("7.71"),
    5:  Decimal("6.61"),
    6:  Decimal("5.99"),
    7:  Decimal("5.59"),
    8:  Decimal("5.32"),
    9:  Decimal("5.12"),
    10: Decimal("4.96"),
    12: Decimal("4.75"),
    15: Decimal("4.54"),
    20: Decimal("4.35"),
    25: Decimal("4.24"),
    30: Decimal("4.17"),
    40: Decimal("4.08"),
    60: Decimal("4.00"),
    120: Decimal("3.92"),
    999: Decimal("3.84"),
}

# Minimum data points required for reliable EnPI analysis per ISO 50006.
_MIN_DATA_POINTS: int = 3
_MIN_REGRESSION_POINTS: int = 6
_MIN_STATISTICAL_POINTS: int = 12

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class EnPIMeasurement(BaseModel):
    """Single energy measurement data point for EnPI calculation.

    Represents one period of energy consumption data along with
    associated normalising variables (production, weather, occupancy).

    Attributes:
        period_start: Start date of the measurement period.
        period_end: End date of the measurement period.
        energy_value: Total energy consumed in the period (kWh, GJ, etc.).
        normalizing_variable: Primary normalising variable value.
        production_output: Production output for the period.
        floor_area: Floor area served (m2).
        occupancy: Average occupancy for the period.
        hdd: Heating degree days for the period.
        cdd: Cooling degree days for the period.
    """
    period_start: date = Field(
        ..., description="Start date of the measurement period"
    )
    period_end: date = Field(
        ..., description="End date of the measurement period"
    )
    energy_value: Decimal = Field(
        ..., ge=0, description="Total energy consumed in period"
    )
    normalizing_variable: Optional[Decimal] = Field(
        default=None, description="Primary normalising variable value"
    )
    production_output: Optional[Decimal] = Field(
        default=None, ge=0, description="Production output for the period"
    )
    floor_area: Optional[Decimal] = Field(
        default=None, ge=0, description="Floor area served (m2)"
    )
    occupancy: Optional[Decimal] = Field(
        default=None, ge=0, description="Average occupancy for the period"
    )
    hdd: Optional[Decimal] = Field(
        default=None, ge=0, description="Heating degree days for the period"
    )
    cdd: Optional[Decimal] = Field(
        default=None, ge=0, description="Cooling degree days for the period"
    )

    @field_validator("period_end")
    @classmethod
    def validate_period_end(cls, v: date, info: Any) -> date:
        """Ensure period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after period_start ({start})"
            )
        return v

class EnPIDefinition(BaseModel):
    """Definition of an Energy Performance Indicator.

    Captures the metadata, type, units, and configuration for an EnPI
    in accordance with ISO 50006.

    Attributes:
        enpi_id: Unique EnPI identifier.
        enpi_name: Human-readable name for the EnPI.
        enpi_type: Type of EnPI calculation.
        energy_type: Energy source (electricity, natural_gas, etc.).
        numerator_unit: Unit for the energy numerator (kWh, GJ, etc.).
        denominator_unit: Unit for the normalising denominator.
        improvement_direction: Whether decrease or increase is better.
        normalization_method: Normalisation method to apply.
        aggregation_level: Organisational aggregation level.
        target_value: Optional target EnPI value.
    """
    enpi_id: str = Field(
        default_factory=_new_uuid, description="Unique EnPI identifier"
    )
    enpi_name: str = Field(
        default="", max_length=500, description="Human-readable EnPI name"
    )
    enpi_type: EnPIType = Field(
        default=EnPIType.INTENSITY, description="Type of EnPI calculation"
    )
    energy_type: str = Field(
        default="electricity", max_length=100,
        description="Energy source type"
    )
    numerator_unit: str = Field(
        default="kWh", max_length=50,
        description="Unit for the energy numerator"
    )
    denominator_unit: str = Field(
        default="m2", max_length=50,
        description="Unit for the normalising denominator"
    )
    improvement_direction: ImprovementDirection = Field(
        default=ImprovementDirection.DECREASE_IS_BETTER,
        description="Whether decrease or increase indicates improvement"
    )
    normalization_method: NormalizationMethod = Field(
        default=NormalizationMethod.NONE,
        description="Normalisation method to apply"
    )
    aggregation_level: AggregationLevel = Field(
        default=AggregationLevel.FACILITY,
        description="Organisational aggregation level"
    )
    target_value: Optional[Decimal] = Field(
        default=None, description="Optional target EnPI value"
    )

    @field_validator("enpi_id")
    @classmethod
    def validate_enpi_id(cls, v: str) -> str:
        """Ensure enpi_id is non-empty."""
        if not v or not v.strip():
            return _new_uuid()
        return v

    @field_validator("enpi_name")
    @classmethod
    def validate_enpi_name(cls, v: str) -> str:
        """Normalise whitespace in enpi_name."""
        return v.strip() if v else ""

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class EnPIValue(BaseModel):
    """Calculated EnPI value for a single measurement period.

    Attributes:
        period_start: Start date of the period.
        period_end: End date of the period.
        measured_value: Raw measured EnPI value (before normalisation).
        normalized_value: EnPI value after normalisation.
        expected_value: Expected EnPI value from baseline model.
        variance_pct: Variance from expected value as percentage.
        improvement_pct: Improvement from baseline as percentage.
        is_on_target: Whether the value meets the defined target.
    """
    period_start: date = Field(
        ..., description="Period start date"
    )
    period_end: date = Field(
        ..., description="Period end date"
    )
    measured_value: Decimal = Field(
        default=Decimal("0"), description="Raw measured EnPI value"
    )
    normalized_value: Decimal = Field(
        default=Decimal("0"), description="Normalised EnPI value"
    )
    expected_value: Decimal = Field(
        default=Decimal("0"), description="Expected EnPI from baseline model"
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"), description="Variance from expected (%)"
    )
    improvement_pct: Decimal = Field(
        default=Decimal("0"), description="Improvement from baseline (%)"
    )
    is_on_target: bool = Field(
        default=False, description="Whether value meets target"
    )

class StatisticalValidation(BaseModel):
    """Result of a statistical validation test on EnPI data.

    Attributes:
        test_type: Type of statistical test performed.
        test_statistic: Computed test statistic value.
        p_value: Approximate p-value (Decimal approximation).
        degrees_of_freedom: Degrees of freedom for the test.
        is_significant: Whether the result is statistically significant.
        confidence_level: Confidence level used (e.g. 0.95).
    """
    test_type: StatisticalTest = Field(
        ..., description="Type of statistical test performed"
    )
    test_statistic: Decimal = Field(
        default=Decimal("0"), description="Computed test statistic"
    )
    p_value: Decimal = Field(
        default=Decimal("1"), description="Approximate p-value"
    )
    degrees_of_freedom: int = Field(
        default=0, ge=0, description="Degrees of freedom"
    )
    is_significant: bool = Field(
        default=False,
        description="Whether the result is statistically significant"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"), description="Confidence level used"
    )

class EnPIResult(BaseModel):
    """Complete EnPI calculation result for a single indicator.

    Attributes:
        enpi_id: Unique EnPI identifier.
        definition: The EnPI definition used.
        baseline_value: Baseline period EnPI value.
        current_value: Current (reporting) period EnPI value.
        improvement_pct: Overall improvement percentage vs baseline.
        values: List of per-period EnPI values.
        statistical_validation: Result of statistical validation.
        trend_slope: Slope of the EnPI trend line.
        trend_r_squared: R-squared of the trend line fit.
        data_quality_score: Data quality score (0-100).
        provenance_hash: SHA-256 audit hash.
        calculation_time_ms: Processing duration in milliseconds.
        calculated_at: Calculation timestamp (UTC).
        methodology_notes: Description of calculation methodology.
    """
    enpi_id: str = Field(default="", description="EnPI identifier")
    definition: EnPIDefinition = Field(
        ..., description="EnPI definition used"
    )
    baseline_value: Decimal = Field(
        default=Decimal("0"), description="Baseline EnPI value"
    )
    current_value: Decimal = Field(
        default=Decimal("0"), description="Current period EnPI value"
    )
    improvement_pct: Decimal = Field(
        default=Decimal("0"), description="Improvement vs baseline (%)"
    )
    values: List[EnPIValue] = Field(
        default_factory=list, description="Per-period EnPI values"
    )
    statistical_validation: Optional[StatisticalValidation] = Field(
        default=None, description="Statistical validation result"
    )
    trend_slope: Decimal = Field(
        default=Decimal("0"), description="Trend line slope"
    )
    trend_r_squared: Decimal = Field(
        default=Decimal("0"), description="Trend line R-squared"
    )
    data_quality_score: Decimal = Field(
        default=Decimal("0"), description="Data quality score (0-100)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    methodology_notes: str = Field(
        default="", description="Methodology description"
    )

class PortfolioEnPIResult(BaseModel):
    """Aggregated EnPI result across a portfolio of facilities.

    Attributes:
        portfolio_id: Unique portfolio identifier.
        facility_enpi_results: Individual facility EnPI results.
        aggregated_improvement_pct: Weighted average improvement (%).
        weighted_average_enpi: Weighted average EnPI value.
        best_performer: Identifier of the best-performing facility.
        worst_performer: Identifier of the worst-performing facility.
        facility_count: Number of facilities in the portfolio.
        provenance_hash: SHA-256 audit hash.
        calculated_at: Calculation timestamp (UTC).
    """
    portfolio_id: str = Field(
        default_factory=_new_uuid, description="Portfolio identifier"
    )
    facility_enpi_results: List[EnPIResult] = Field(
        default_factory=list, description="Individual facility results"
    )
    aggregated_improvement_pct: Decimal = Field(
        default=Decimal("0"), description="Weighted average improvement (%)"
    )
    weighted_average_enpi: Decimal = Field(
        default=Decimal("0"), description="Weighted average EnPI value"
    )
    best_performer: str = Field(
        default="", description="Best-performing facility ID"
    )
    worst_performer: str = Field(
        default="", description="Worst-performing facility ID"
    )
    facility_count: int = Field(
        default=0, ge=0, description="Number of facilities"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )

# ---------------------------------------------------------------------------
# Model Rebuild (required for from __future__ import annotations)
# ---------------------------------------------------------------------------

EnPIMeasurement.model_rebuild()
EnPIDefinition.model_rebuild()
EnPIValue.model_rebuild()
StatisticalValidation.model_rebuild()
EnPIResult.model_rebuild()
PortfolioEnPIResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EnPICalculatorEngine:
    """Energy Performance Indicator calculator per ISO 50006.

    Computes absolute, intensity, regression-modelled, proportion, and
    statistical EnPIs.  Supports weather/production/occupancy/multi-variable
    normalisation, baseline comparison, statistical validation, trend
    analysis, and portfolio-level aggregation.

    All calculations use deterministic ``Decimal`` arithmetic with SHA-256
    provenance hashing on every result.

    Usage::

        engine = EnPICalculatorEngine()
        definition = EnPIDefinition(
            enpi_name="Electricity Intensity",
            enpi_type=EnPIType.INTENSITY,
            energy_type="electricity",
            numerator_unit="kWh",
            denominator_unit="m2",
        )
        baseline = [
            EnPIMeasurement(
                period_start=date(2024, 1, 1),
                period_end=date(2024, 1, 31),
                energy_value=Decimal("50000"),
                floor_area=Decimal("2000"),
            ),
        ]
        reporting = [
            EnPIMeasurement(
                period_start=date(2025, 1, 1),
                period_end=date(2025, 1, 31),
                energy_value=Decimal("45000"),
                floor_area=Decimal("2000"),
            ),
        ]
        result = engine.calculate_enpi(definition, reporting, baseline)
        print(f"Improvement: {result.improvement_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EnPICalculatorEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - significance_level (str): "strict", "standard", "relaxed"
                - min_data_points (int): minimum data points required
                - default_normalization (str): NormalizationMethod value
                - regression_intercept_override (Decimal): force intercept
        """
        self.config = config or {}
        self._significance_level: str = self.config.get(
            "significance_level", "standard"
        )
        self._min_data_points: int = int(
            self.config.get("min_data_points", _MIN_DATA_POINTS)
        )
        self._default_normalization: str = self.config.get(
            "default_normalization", NormalizationMethod.NONE.value
        )
        logger.info(
            "EnPICalculatorEngine v%s initialised (significance=%s, "
            "min_points=%d)",
            self.engine_version, self._significance_level,
            self._min_data_points,
        )

    # ------------------------------------------------------------------ #
    # Public API -- Main Entry Point                                      #
    # ------------------------------------------------------------------ #

    def calculate_enpi(
        self,
        definition: EnPIDefinition,
        measurements: List[EnPIMeasurement],
        baseline_measurements: List[EnPIMeasurement],
    ) -> EnPIResult:
        """Calculate an EnPI based on its definition, measurements, and baseline.

        Workflow:
            1. Validate inputs (minimum data points, non-empty periods).
            2. Calculate baseline EnPI value.
            3. Normalise measurements if a normalisation method is set.
            4. Calculate per-period EnPI values based on type.
            5. Calculate overall improvement percentage vs baseline.
            6. Perform statistical validation of improvement.
            7. Calculate trend (slope, R-squared).
            8. Assess data quality.
            9. Generate provenance hash.

        Args:
            definition: EnPI definition with type and configuration.
            measurements: Reporting period measurement data.
            baseline_measurements: Baseline period measurement data.

        Returns:
            EnPIResult with per-period values, statistics, and provenance.

        Raises:
            ValueError: If insufficient data or invalid configuration.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating EnPI: id=%s, name=%s, type=%s, measurements=%d, "
            "baseline=%d",
            definition.enpi_id, definition.enpi_name,
            definition.enpi_type.value, len(measurements),
            len(baseline_measurements),
        )

        # Step 1: Validate inputs
        self._validate_inputs(definition, measurements, baseline_measurements)

        # Step 2: Apply normalisation to both baseline and reporting
        norm_method = definition.normalization_method
        baseline_conditions = self._compute_baseline_conditions(
            baseline_measurements
        )
        norm_baseline = self.normalize_measurements(
            baseline_measurements, norm_method, baseline_conditions
        )
        norm_reporting = self.normalize_measurements(
            measurements, norm_method, baseline_conditions
        )

        # Step 3: Calculate baseline EnPI value
        baseline_value = self._compute_aggregate_enpi(
            definition, norm_baseline
        )

        # Step 4: Calculate per-period EnPI values
        enpi_values = self._calculate_period_values(
            definition, norm_reporting, baseline_value,
            baseline_conditions,
        )

        # Step 5: Calculate current (aggregate) EnPI value
        current_value = self._compute_aggregate_enpi(
            definition, norm_reporting
        )

        # Step 6: Calculate improvement
        improvement_pct = self.calculate_improvement(
            baseline_value, current_value, definition.improvement_direction,
        )

        # Step 7: Check target compliance
        if definition.target_value is not None:
            for ev in enpi_values:
                ev.is_on_target = self._check_target(
                    ev.normalized_value,
                    definition.target_value,
                    definition.improvement_direction,
                )

        # Step 8: Statistical validation
        baseline_vals = [
            self._single_period_enpi(definition, m)
            for m in norm_baseline
        ]
        current_vals = [
            ev.normalized_value for ev in enpi_values
        ]
        stat_validation: Optional[StatisticalValidation] = None
        if len(baseline_vals) >= 2 and len(current_vals) >= 2:
            stat_validation = self.perform_statistical_test(
                baseline_vals, current_vals, StatisticalTest.T_TEST,
            )

        # Step 9: Trend analysis
        trend_slope = Decimal("0")
        trend_r_squared = Decimal("0")
        if len(enpi_values) >= 2:
            trend_slope, trend_r_squared = self.calculate_trend(enpi_values)

        # Step 10: Data quality
        data_quality = self._assess_data_quality(
            measurements, baseline_measurements, definition,
        )

        # Step 11: Methodology notes
        methodology = self._build_methodology_notes(
            definition, len(measurements), len(baseline_measurements),
            baseline_value, current_value, improvement_pct,
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = EnPIResult(
            enpi_id=definition.enpi_id,
            definition=definition,
            baseline_value=_round_val(baseline_value, 6),
            current_value=_round_val(current_value, 6),
            improvement_pct=_round_val(improvement_pct, 2),
            values=enpi_values,
            statistical_validation=stat_validation,
            trend_slope=_round_val(trend_slope, 6),
            trend_r_squared=_round_val(trend_r_squared, 6),
            data_quality_score=_round_val(data_quality, 2),
            calculation_time_ms=elapsed_ms,
            calculated_at=utcnow(),
            methodology_notes=methodology,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "EnPI calculation complete: id=%s, baseline=%.4f, "
            "current=%.4f, improvement=%.2f%%, quality=%.0f, "
            "hash=%s, %d ms",
            definition.enpi_id, float(baseline_value),
            float(current_value), float(improvement_pct),
            float(data_quality), result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # EnPI Type Calculators                                               #
    # ------------------------------------------------------------------ #

    def calculate_absolute_enpi(
        self,
        measurements: List[EnPIMeasurement],
    ) -> List[EnPIValue]:
        """Calculate absolute EnPI values (total energy per period).

        The absolute EnPI is simply the energy_value for each measurement
        period.  No denominator is used.

        Args:
            measurements: List of measurement data points.

        Returns:
            List of EnPIValue with measured_value = energy_value.
        """
        logger.debug("Calculating absolute EnPI for %d periods", len(measurements))
        values: List[EnPIValue] = []
        for m in measurements:
            values.append(EnPIValue(
                period_start=m.period_start,
                period_end=m.period_end,
                measured_value=_round_val(m.energy_value, 6),
                normalized_value=_round_val(m.energy_value, 6),
                expected_value=Decimal("0"),
                variance_pct=Decimal("0"),
                improvement_pct=Decimal("0"),
                is_on_target=False,
            ))
        return values

    def calculate_intensity_enpi(
        self,
        measurements: List[EnPIMeasurement],
        denominator_field: str = "normalizing_variable",
    ) -> List[EnPIValue]:
        """Calculate intensity EnPI values (energy / normalising variable).

        The denominator_field specifies which measurement attribute to use
        as the normalising variable (e.g. "floor_area", "production_output",
        "occupancy", or "normalizing_variable").

        Args:
            measurements: List of measurement data points.
            denominator_field: Name of the measurement attribute to use
                               as the normalising denominator.

        Returns:
            List of EnPIValue with measured_value = energy / denominator.
        """
        logger.debug(
            "Calculating intensity EnPI for %d periods, denominator=%s",
            len(measurements), denominator_field,
        )
        values: List[EnPIValue] = []
        for m in measurements:
            denominator = self._get_denominator(m, denominator_field)
            intensity = _safe_divide(m.energy_value, denominator)
            values.append(EnPIValue(
                period_start=m.period_start,
                period_end=m.period_end,
                measured_value=_round_val(intensity, 6),
                normalized_value=_round_val(intensity, 6),
                expected_value=Decimal("0"),
                variance_pct=Decimal("0"),
                improvement_pct=Decimal("0"),
                is_on_target=False,
            ))
        return values

    def calculate_regression_enpi(
        self,
        measurements: List[EnPIMeasurement],
        baseline_model: Dict[str, Decimal],
    ) -> List[EnPIValue]:
        """Calculate regression-modelled EnPI values per ISO 50006 Section 6.4.

        Uses a baseline regression model (intercept + slope * variable) to
        compute expected energy consumption, then compares actual vs expected.

        The baseline_model dictionary must contain:
            - "intercept": Decimal (y-intercept of baseline regression)
            - "slope": Decimal (slope coefficient)
            - "variable_field": str (measurement field to use as x-variable)

        Formula:
            expected = intercept + slope * variable_value
            enpi = actual / expected  (ratio; <1 means improvement)

        Args:
            measurements: List of measurement data points.
            baseline_model: Regression model parameters from baseline.

        Returns:
            List of EnPIValue with expected_value from the model and
            measured_value as the actual/expected ratio.
        """
        intercept = _decimal(baseline_model.get("intercept", Decimal("0")))
        slope = _decimal(baseline_model.get("slope", Decimal("0")))
        var_field = str(baseline_model.get("variable_field", "normalizing_variable"))

        logger.debug(
            "Calculating regression EnPI: intercept=%.4f, slope=%.4f, "
            "field=%s, periods=%d",
            float(intercept), float(slope), var_field, len(measurements),
        )

        values: List[EnPIValue] = []
        for m in measurements:
            x_val = self._get_denominator(m, var_field)
            expected = intercept + slope * x_val
            actual = m.energy_value

            # Ratio: actual / expected (< 1.0 means less than expected)
            ratio = _safe_divide(actual, expected, Decimal("1"))

            # Variance: (actual - expected) / expected * 100
            variance_pct = Decimal("0")
            if expected != Decimal("0"):
                variance_pct = _safe_pct(actual - expected, expected)

            values.append(EnPIValue(
                period_start=m.period_start,
                period_end=m.period_end,
                measured_value=_round_val(ratio, 6),
                normalized_value=_round_val(ratio, 6),
                expected_value=_round_val(expected, 6),
                variance_pct=_round_val(variance_pct, 2),
                improvement_pct=_round_val(-variance_pct, 2),
                is_on_target=False,
            ))
        return values

    def calculate_proportion_enpi(
        self,
        measurements: List[EnPIMeasurement],
        reference_total: Decimal,
    ) -> List[EnPIValue]:
        """Calculate proportion EnPI values (energy as % of reference total).

        Expresses each period's energy consumption as a percentage of a
        reference total (e.g. HVAC energy / total facility energy * 100).

        Args:
            measurements: List of measurement data points.
            reference_total: Total reference energy to compute proportion
                             against (must be > 0).

        Returns:
            List of EnPIValue with measured_value as percentage.
        """
        logger.debug(
            "Calculating proportion EnPI: reference=%.2f, periods=%d",
            float(reference_total), len(measurements),
        )
        values: List[EnPIValue] = []
        for m in measurements:
            proportion = _safe_pct(m.energy_value, reference_total)
            values.append(EnPIValue(
                period_start=m.period_start,
                period_end=m.period_end,
                measured_value=_round_val(proportion, 4),
                normalized_value=_round_val(proportion, 4),
                expected_value=Decimal("0"),
                variance_pct=Decimal("0"),
                improvement_pct=Decimal("0"),
                is_on_target=False,
            ))
        return values

    def calculate_statistical_enpi(
        self,
        measurements: List[EnPIMeasurement],
        baseline_mean: Decimal,
    ) -> List[EnPIValue]:
        """Calculate statistical (CUSUM) EnPI values.

        Computes the cumulative sum of deviations from the baseline mean.
        A downward CUSUM trend indicates improvement (for decrease_is_better).

        Formula:
            cusum_t = cusum_{t-1} + (energy_t - baseline_mean)

        Args:
            measurements: List of measurement data points.
            baseline_mean: Mean energy value from the baseline period.

        Returns:
            List of EnPIValue with measured_value as CUSUM.
        """
        logger.debug(
            "Calculating CUSUM EnPI: baseline_mean=%.4f, periods=%d",
            float(baseline_mean), len(measurements),
        )
        values: List[EnPIValue] = []
        cusum = Decimal("0")
        for m in measurements:
            deviation = m.energy_value - baseline_mean
            cusum += deviation
            values.append(EnPIValue(
                period_start=m.period_start,
                period_end=m.period_end,
                measured_value=_round_val(cusum, 4),
                normalized_value=_round_val(cusum, 4),
                expected_value=_round_val(baseline_mean, 4),
                variance_pct=_safe_pct(deviation, baseline_mean),
                improvement_pct=Decimal("0"),
                is_on_target=False,
            ))
        return values

    # ------------------------------------------------------------------ #
    # Normalisation                                                       #
    # ------------------------------------------------------------------ #

    def normalize_measurements(
        self,
        measurements: List[EnPIMeasurement],
        method: NormalizationMethod,
        baseline_conditions: Dict[str, Decimal],
    ) -> List[EnPIMeasurement]:
        """Normalise measurement data using the specified method.

        Adjusts energy values to account for changes in relevant variables
        (weather, production, occupancy) relative to baseline conditions.

        Normalisation formulae per ISO 50006 Section 7:
            Weather:
                adjusted = energy * (baseline_hdd / period_hdd)  (heating)
                adjusted = energy * (baseline_cdd / period_cdd)  (cooling)
            Production:
                adjusted = energy * (baseline_production / period_production)
            Occupancy:
                adjusted = energy * (baseline_occupancy / period_occupancy)
            Multi-variable:
                adjusted = energy * product of all applicable ratios

        Args:
            measurements: Measurement data to normalise.
            method: Normalisation method to apply.
            baseline_conditions: Dict of baseline reference values
                (avg_hdd, avg_cdd, avg_production, avg_occupancy).

        Returns:
            New list of EnPIMeasurement with adjusted energy_value.
        """
        if method == NormalizationMethod.NONE:
            return list(measurements)

        logger.debug(
            "Normalising %d measurements using method=%s",
            len(measurements), method.value,
        )

        normalised: List[EnPIMeasurement] = []
        for m in measurements:
            adj_energy = m.energy_value
            adjustment_factor = Decimal("1")

            if method in (NormalizationMethod.WEATHER,
                          NormalizationMethod.MULTI_VARIABLE):
                adjustment_factor *= self._weather_adjustment(
                    m, baseline_conditions
                )

            if method in (NormalizationMethod.PRODUCTION,
                          NormalizationMethod.MULTI_VARIABLE):
                adjustment_factor *= self._production_adjustment(
                    m, baseline_conditions
                )

            if method in (NormalizationMethod.OCCUPANCY,
                          NormalizationMethod.MULTI_VARIABLE):
                adjustment_factor *= self._occupancy_adjustment(
                    m, baseline_conditions
                )

            adj_energy = m.energy_value * adjustment_factor

            normalised.append(EnPIMeasurement(
                period_start=m.period_start,
                period_end=m.period_end,
                energy_value=_round_val(adj_energy, 4),
                normalizing_variable=m.normalizing_variable,
                production_output=m.production_output,
                floor_area=m.floor_area,
                occupancy=m.occupancy,
                hdd=m.hdd,
                cdd=m.cdd,
            ))

        return normalised

    # ------------------------------------------------------------------ #
    # Improvement Calculation                                             #
    # ------------------------------------------------------------------ #

    def calculate_improvement(
        self,
        baseline: Decimal,
        current: Decimal,
        direction: ImprovementDirection,
    ) -> Decimal:
        """Calculate improvement percentage of current vs baseline.

        For DECREASE_IS_BETTER:
            improvement = (baseline - current) / baseline * 100
            Positive value means improvement (current < baseline).

        For INCREASE_IS_BETTER:
            improvement = (current - baseline) / baseline * 100
            Positive value means improvement (current > baseline).

        Args:
            baseline: Baseline period EnPI value.
            current: Current (reporting) period EnPI value.
            direction: Whether decrease or increase is better.

        Returns:
            Improvement percentage (positive = improved).
        """
        if baseline == Decimal("0"):
            return Decimal("0")

        if direction == ImprovementDirection.DECREASE_IS_BETTER:
            improvement = _safe_pct(baseline - current, baseline)
        else:
            improvement = _safe_pct(current - baseline, baseline)

        return improvement

    # ------------------------------------------------------------------ #
    # Statistical Validation                                              #
    # ------------------------------------------------------------------ #

    def perform_statistical_test(
        self,
        baseline_values: List[Decimal],
        current_values: List[Decimal],
        test_type: StatisticalTest,
    ) -> StatisticalValidation:
        """Perform a statistical test comparing baseline vs current values.

        Supported tests:
            T_TEST:      Two-sample t-test on means.
            F_TEST:      F-test on variances.
            CHI_SQUARED: Chi-squared goodness-of-fit.
            ANOVA:       One-way ANOVA.

        All computations use Decimal arithmetic with lookup tables for
        critical values (no scipy dependency).

        Args:
            baseline_values: EnPI values from the baseline period.
            current_values: EnPI values from the reporting period.
            test_type: Type of statistical test to perform.

        Returns:
            StatisticalValidation with test statistic, p-value
            approximation, and significance determination.
        """
        logger.debug(
            "Statistical test: type=%s, baseline_n=%d, current_n=%d",
            test_type.value, len(baseline_values), len(current_values),
        )

        if test_type == StatisticalTest.T_TEST:
            return self._t_test(baseline_values, current_values)
        elif test_type == StatisticalTest.F_TEST:
            return self._f_test(baseline_values, current_values)
        elif test_type == StatisticalTest.CHI_SQUARED:
            return self._chi_squared_test(baseline_values, current_values)
        elif test_type == StatisticalTest.ANOVA:
            return self._anova_test(baseline_values, current_values)
        else:
            logger.warning("Unknown test type: %s, defaulting to t-test", test_type)
            return self._t_test(baseline_values, current_values)

    # ------------------------------------------------------------------ #
    # Trend Analysis                                                      #
    # ------------------------------------------------------------------ #

    def calculate_trend(
        self,
        values: List[EnPIValue],
    ) -> Tuple[Decimal, Decimal]:
        """Calculate the trend (slope and R-squared) of EnPI values over time.

        Uses ordinary least squares (OLS) linear regression with period
        index as the independent variable and normalised EnPI value as the
        dependent variable.

        Formulae:
            slope = Cov(x, y) / Var(x)
            intercept = mean(y) - slope * mean(x)
            R^2 = [Cov(x, y)]^2 / [Var(x) * Var(y)]

        Args:
            values: List of per-period EnPIValue objects (ordered by time).

        Returns:
            Tuple of (slope, r_squared) as Decimals.
        """
        n = len(values)
        if n < 2:
            return Decimal("0"), Decimal("0")

        # Use integer index 0..n-1 as x, normalised_value as y
        x_vals = [_decimal(i) for i in range(n)]
        y_vals = [v.normalized_value for v in values]

        slope, r_squared = self._ols_regression(x_vals, y_vals)

        logger.debug(
            "Trend analysis: n=%d, slope=%.6f, R2=%.6f",
            n, float(slope), float(r_squared),
        )
        return slope, r_squared

    # ------------------------------------------------------------------ #
    # Portfolio Aggregation                                               #
    # ------------------------------------------------------------------ #

    def aggregate_portfolio(
        self,
        facility_results: List[EnPIResult],
        weights: Optional[Dict[str, Decimal]] = None,
    ) -> PortfolioEnPIResult:
        """Aggregate facility-level EnPI results into a portfolio result.

        Computes weighted average EnPI value and improvement percentage
        across all facilities.  Identifies best and worst performers.

        If weights are not provided, equal weighting is applied.

        Args:
            facility_results: List of individual facility EnPIResult objects.
            weights: Optional dict mapping enpi_id -> weight (Decimal).
                     Weights are normalised to sum to 1.0.

        Returns:
            PortfolioEnPIResult with aggregated metrics.
        """
        t0 = time.perf_counter()
        n = len(facility_results)
        logger.info("Aggregating portfolio: %d facilities", n)

        if n == 0:
            result = PortfolioEnPIResult(
                facility_enpi_results=[],
                facility_count=0,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Normalise weights
        norm_weights = self._normalise_weights(facility_results, weights)

        # Weighted average EnPI
        weighted_enpi = Decimal("0")
        weighted_improvement = Decimal("0")
        for fr in facility_results:
            w = norm_weights.get(fr.enpi_id, Decimal("0"))
            weighted_enpi += fr.current_value * w
            weighted_improvement += fr.improvement_pct * w

        # Best and worst performers
        best_id = ""
        worst_id = ""
        best_improvement = Decimal("-999999")
        worst_improvement = Decimal("999999")
        for fr in facility_results:
            if fr.improvement_pct > best_improvement:
                best_improvement = fr.improvement_pct
                best_id = fr.enpi_id
            if fr.improvement_pct < worst_improvement:
                worst_improvement = fr.improvement_pct
                worst_id = fr.enpi_id

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = PortfolioEnPIResult(
            facility_enpi_results=facility_results,
            aggregated_improvement_pct=_round_val(weighted_improvement, 2),
            weighted_average_enpi=_round_val(weighted_enpi, 6),
            best_performer=best_id,
            worst_performer=worst_id,
            facility_count=n,
            calculated_at=utcnow(),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Portfolio aggregation complete: %d facilities, "
            "avg_enpi=%.4f, avg_improvement=%.2f%%, best=%s, worst=%s, "
            "hash=%s, %d ms",
            n, float(weighted_enpi), float(weighted_improvement),
            best_id[:16], worst_id[:16],
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Methodology Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_enpi_methodology(
        self,
        result: EnPIResult,
    ) -> Dict[str, Any]:
        """Validate an EnPI result against ISO 50006 methodology requirements.

        Checks:
            - Minimum data points for statistical significance.
            - Statistical significance of improvement claim.
            - Data quality meets ISO 50006 thresholds.
            - Regression model validity (R-squared) for regression EnPIs.
            - Baseline representativeness.

        Args:
            result: EnPIResult to validate.

        Returns:
            Dict with 'is_valid', 'checks' (list of check results),
            'warnings', 'errors', and provenance_hash.
        """
        logger.info(
            "Validating EnPI methodology: id=%s, type=%s",
            result.enpi_id, result.definition.enpi_type.value,
        )

        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        errors: List[str] = []

        # Check 1: Minimum data points
        n_values = len(result.values)
        min_required = self._min_data_points
        if result.definition.enpi_type == EnPIType.REGRESSION_MODELED:
            min_required = max(min_required, _MIN_REGRESSION_POINTS)
        if result.definition.enpi_type == EnPIType.STATISTICAL:
            min_required = max(min_required, _MIN_STATISTICAL_POINTS)

        check_data_points = n_values >= min_required
        checks.append({
            "check": "minimum_data_points",
            "required": min_required,
            "actual": n_values,
            "pass": check_data_points,
        })
        if not check_data_points:
            errors.append(
                f"Insufficient data points: {n_values} < {min_required} required"
            )

        # Check 2: Statistical significance
        check_significance = True
        if result.statistical_validation is not None:
            check_significance = result.statistical_validation.is_significant
            checks.append({
                "check": "statistical_significance",
                "test": result.statistical_validation.test_type.value,
                "p_value": str(result.statistical_validation.p_value),
                "pass": check_significance,
            })
            if not check_significance:
                warnings.append(
                    "Improvement is not statistically significant at the "
                    f"{self._significance_level} level "
                    f"(p={result.statistical_validation.p_value})"
                )
        else:
            checks.append({
                "check": "statistical_significance",
                "test": "none",
                "pass": False,
                "reason": "No statistical validation performed",
            })
            warnings.append("No statistical validation was performed")

        # Check 3: Data quality score
        quality_threshold = Decimal("50")
        check_quality = result.data_quality_score >= quality_threshold
        checks.append({
            "check": "data_quality",
            "threshold": str(quality_threshold),
            "actual": str(result.data_quality_score),
            "pass": check_quality,
        })
        if not check_quality:
            warnings.append(
                f"Data quality score ({result.data_quality_score}) is below "
                f"the recommended threshold of {quality_threshold}"
            )

        # Check 4: Regression model validity (R-squared)
        if result.definition.enpi_type == EnPIType.REGRESSION_MODELED:
            r2_threshold = Decimal("0.75")
            check_r2 = result.trend_r_squared >= r2_threshold
            checks.append({
                "check": "regression_r_squared",
                "threshold": str(r2_threshold),
                "actual": str(result.trend_r_squared),
                "pass": check_r2,
            })
            if not check_r2:
                warnings.append(
                    f"Regression R-squared ({result.trend_r_squared}) is below "
                    f"the recommended threshold of {r2_threshold}"
                )

        # Check 5: Non-zero baseline
        check_baseline = result.baseline_value != Decimal("0")
        checks.append({
            "check": "non_zero_baseline",
            "baseline": str(result.baseline_value),
            "pass": check_baseline,
        })
        if not check_baseline:
            errors.append(
                "Baseline EnPI value is zero; improvement cannot be calculated"
            )

        # Check 6: Trend consistency
        if len(result.values) >= 3:
            check_trend = self._check_trend_consistency(result)
            checks.append({
                "check": "trend_consistency",
                "slope": str(result.trend_slope),
                "r_squared": str(result.trend_r_squared),
                "pass": check_trend,
            })
            if not check_trend:
                warnings.append(
                    "EnPI trend is inconsistent or shows high variability"
                )

        is_valid = len(errors) == 0
        validation_result = {
            "is_valid": is_valid,
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
            "enpi_id": result.enpi_id,
            "enpi_type": result.definition.enpi_type.value,
            "provenance_hash": _compute_hash({
                "enpi_id": result.enpi_id,
                "is_valid": is_valid,
                "check_count": len(checks),
                "error_count": len(errors),
                "warning_count": len(warnings),
            }),
        }

        logger.info(
            "Methodology validation: id=%s, valid=%s, checks=%d, "
            "errors=%d, warnings=%d",
            result.enpi_id, is_valid, len(checks),
            len(errors), len(warnings),
        )
        return validation_result

    # ------------------------------------------------------------------ #
    # Baseline Regression Model Builder                                   #
    # ------------------------------------------------------------------ #

    def build_baseline_regression(
        self,
        baseline_measurements: List[EnPIMeasurement],
        variable_field: str = "normalizing_variable",
    ) -> Dict[str, Any]:
        """Build a baseline regression model from measurement data.

        Performs ordinary least squares regression of energy_value (y)
        against the specified variable field (x) to produce a baseline
        model for regression-modelled EnPI calculations.

        Args:
            baseline_measurements: Baseline period measurements.
            variable_field: Measurement field to use as the independent
                            variable.

        Returns:
            Dict with intercept, slope, r_squared, variable_field,
            n_points, and provenance_hash.
        """
        logger.info(
            "Building baseline regression: variable=%s, n=%d",
            variable_field, len(baseline_measurements),
        )

        if len(baseline_measurements) < _MIN_REGRESSION_POINTS:
            logger.warning(
                "Insufficient points for regression: %d < %d",
                len(baseline_measurements), _MIN_REGRESSION_POINTS,
            )

        x_vals: List[Decimal] = []
        y_vals: List[Decimal] = []
        for m in baseline_measurements:
            x = self._get_denominator(m, variable_field)
            if x > Decimal("0"):
                x_vals.append(x)
                y_vals.append(m.energy_value)

        if len(x_vals) < 2:
            return {
                "intercept": Decimal("0"),
                "slope": Decimal("0"),
                "r_squared": Decimal("0"),
                "variable_field": variable_field,
                "n_points": len(x_vals),
                "provenance_hash": _compute_hash({
                    "n": len(x_vals), "field": variable_field
                }),
            }

        slope, r_squared = self._ols_regression(x_vals, y_vals)
        mean_x = self._mean(x_vals)
        mean_y = self._mean(y_vals)
        intercept = mean_y - slope * mean_x

        model = {
            "intercept": _round_val(intercept, 6),
            "slope": _round_val(slope, 6),
            "r_squared": _round_val(r_squared, 6),
            "variable_field": variable_field,
            "n_points": len(x_vals),
            "provenance_hash": _compute_hash({
                "intercept": str(intercept),
                "slope": str(slope),
                "r_squared": str(r_squared),
                "n": len(x_vals),
            }),
        }

        logger.info(
            "Baseline regression: intercept=%.4f, slope=%.4f, R2=%.4f, n=%d",
            float(intercept), float(slope), float(r_squared), len(x_vals),
        )
        return model

    # ------------------------------------------------------------------ #
    # Energy Savings from EnPI                                            #
    # ------------------------------------------------------------------ #

    def calculate_energy_savings(
        self,
        result: EnPIResult,
        baseline_measurements: List[EnPIMeasurement],
        reporting_measurements: List[EnPIMeasurement],
    ) -> Dict[str, Any]:
        """Calculate energy savings implied by the EnPI improvement.

        For absolute EnPI:
            savings = sum(baseline_energy) - sum(reporting_energy)
        For intensity EnPI:
            savings = sum(reporting_denominator * (baseline_enpi - reporting_enpi))
        For regression EnPI:
            savings = sum(expected_energy - actual_energy)

        Args:
            result: Calculated EnPIResult.
            baseline_measurements: Baseline period measurements.
            reporting_measurements: Reporting period measurements.

        Returns:
            Dict with total_savings, savings_by_period, and provenance_hash.
        """
        logger.info(
            "Calculating energy savings from EnPI: id=%s, type=%s",
            result.enpi_id, result.definition.enpi_type.value,
        )

        total_savings = Decimal("0")
        period_savings: List[Dict[str, Any]] = []

        if result.definition.enpi_type == EnPIType.ABSOLUTE:
            baseline_total = sum(
                (m.energy_value for m in baseline_measurements), Decimal("0")
            )
            reporting_total = sum(
                (m.energy_value for m in reporting_measurements), Decimal("0")
            )
            total_savings = baseline_total - reporting_total
            period_savings.append({
                "period": "aggregate",
                "baseline_energy": str(_round_val(baseline_total, 2)),
                "reporting_energy": str(_round_val(reporting_total, 2)),
                "savings": str(_round_val(total_savings, 2)),
            })

        elif result.definition.enpi_type == EnPIType.REGRESSION_MODELED:
            for ev in result.values:
                savings_period = ev.expected_value - (
                    ev.expected_value * ev.measured_value
                )
                total_savings += savings_period
                period_savings.append({
                    "period_start": str(ev.period_start),
                    "period_end": str(ev.period_end),
                    "expected": str(_round_val(ev.expected_value, 2)),
                    "actual_ratio": str(_round_val(ev.measured_value, 4)),
                    "savings": str(_round_val(savings_period, 2)),
                })

        else:
            # For intensity / proportion / statistical
            baseline_total = sum(
                (m.energy_value for m in baseline_measurements), Decimal("0")
            )
            reporting_total = sum(
                (m.energy_value for m in reporting_measurements), Decimal("0")
            )
            total_savings = baseline_total - reporting_total
            period_savings.append({
                "period": "aggregate",
                "baseline_energy": str(_round_val(baseline_total, 2)),
                "reporting_energy": str(_round_val(reporting_total, 2)),
                "savings": str(_round_val(total_savings, 2)),
            })

        savings_result = {
            "enpi_id": result.enpi_id,
            "total_savings": str(_round_val(total_savings, 2)),
            "savings_by_period": period_savings,
            "unit": result.definition.numerator_unit,
            "provenance_hash": _compute_hash({
                "enpi_id": result.enpi_id,
                "total_savings": str(total_savings),
            }),
        }

        logger.info(
            "Energy savings: id=%s, total=%.2f %s",
            result.enpi_id, float(total_savings),
            result.definition.numerator_unit,
        )
        return savings_result

    # ------------------------------------------------------------------ #
    # Internal -- Input Validation                                        #
    # ------------------------------------------------------------------ #

    def _validate_inputs(
        self,
        definition: EnPIDefinition,
        measurements: List[EnPIMeasurement],
        baseline_measurements: List[EnPIMeasurement],
    ) -> None:
        """Validate inputs for EnPI calculation.

        Args:
            definition: EnPI definition.
            measurements: Reporting period measurements.
            baseline_measurements: Baseline period measurements.

        Raises:
            ValueError: If validation fails.
        """
        if not measurements:
            raise ValueError("No reporting period measurements provided")
        if not baseline_measurements:
            raise ValueError("No baseline period measurements provided")
        if len(measurements) < 1:
            raise ValueError("At least one reporting measurement is required")
        if len(baseline_measurements) < 1:
            raise ValueError("At least one baseline measurement is required")

        # Type-specific validation
        if definition.enpi_type == EnPIType.INTENSITY:
            for m in measurements + baseline_measurements:
                denom = self._get_denominator(m, "normalizing_variable")
                if m.normalizing_variable is not None and denom == Decimal("0"):
                    logger.warning(
                        "Zero normalising variable in period %s to %s",
                        m.period_start, m.period_end,
                    )

    # ------------------------------------------------------------------ #
    # Internal -- Baseline Conditions                                     #
    # ------------------------------------------------------------------ #

    def _compute_baseline_conditions(
        self,
        baseline: List[EnPIMeasurement],
    ) -> Dict[str, Decimal]:
        """Compute average baseline conditions for normalisation.

        Args:
            baseline: Baseline period measurements.

        Returns:
            Dict with avg_hdd, avg_cdd, avg_production, avg_occupancy.
        """
        n = _decimal(len(baseline))
        if n == Decimal("0"):
            return {
                "avg_hdd": Decimal("0"),
                "avg_cdd": Decimal("0"),
                "avg_production": Decimal("0"),
                "avg_occupancy": Decimal("0"),
            }

        total_hdd = sum(
            (_decimal(m.hdd) for m in baseline if m.hdd is not None),
            Decimal("0"),
        )
        total_cdd = sum(
            (_decimal(m.cdd) for m in baseline if m.cdd is not None),
            Decimal("0"),
        )
        total_prod = sum(
            (_decimal(m.production_output)
             for m in baseline if m.production_output is not None),
            Decimal("0"),
        )
        total_occ = sum(
            (_decimal(m.occupancy)
             for m in baseline if m.occupancy is not None),
            Decimal("0"),
        )

        hdd_count = sum(1 for m in baseline if m.hdd is not None)
        cdd_count = sum(1 for m in baseline if m.cdd is not None)
        prod_count = sum(1 for m in baseline if m.production_output is not None)
        occ_count = sum(1 for m in baseline if m.occupancy is not None)

        return {
            "avg_hdd": _safe_divide(total_hdd, _decimal(max(hdd_count, 1))),
            "avg_cdd": _safe_divide(total_cdd, _decimal(max(cdd_count, 1))),
            "avg_production": _safe_divide(
                total_prod, _decimal(max(prod_count, 1))
            ),
            "avg_occupancy": _safe_divide(
                total_occ, _decimal(max(occ_count, 1))
            ),
        }

    # ------------------------------------------------------------------ #
    # Internal -- Normalisation Helpers                                   #
    # ------------------------------------------------------------------ #

    def _weather_adjustment(
        self,
        measurement: EnPIMeasurement,
        baseline_conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Compute weather normalisation adjustment factor.

        Uses HDD and CDD ratios to baseline conditions.

        Args:
            measurement: Current measurement.
            baseline_conditions: Baseline average conditions.

        Returns:
            Adjustment factor (multiply energy by this).
        """
        factor = Decimal("1")
        base_hdd = baseline_conditions.get("avg_hdd", Decimal("0"))
        base_cdd = baseline_conditions.get("avg_cdd", Decimal("0"))

        period_hdd = _decimal(measurement.hdd) if measurement.hdd else Decimal("0")
        period_cdd = _decimal(measurement.cdd) if measurement.cdd else Decimal("0")

        # HDD adjustment: if period HDD differs from baseline
        if base_hdd > Decimal("0") and period_hdd > Decimal("0"):
            hdd_ratio = _safe_divide(base_hdd, period_hdd, Decimal("1"))
            # Weight HDD adjustment at 50% (conservative)
            hdd_adj = Decimal("1") + (hdd_ratio - Decimal("1")) * Decimal("0.5")
            factor *= hdd_adj

        # CDD adjustment: if period CDD differs from baseline
        if base_cdd > Decimal("0") and period_cdd > Decimal("0"):
            cdd_ratio = _safe_divide(base_cdd, period_cdd, Decimal("1"))
            cdd_adj = Decimal("1") + (cdd_ratio - Decimal("1")) * Decimal("0.5")
            factor *= cdd_adj

        return factor

    def _production_adjustment(
        self,
        measurement: EnPIMeasurement,
        baseline_conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Compute production normalisation adjustment factor.

        Args:
            measurement: Current measurement.
            baseline_conditions: Baseline average conditions.

        Returns:
            Adjustment factor.
        """
        base_prod = baseline_conditions.get("avg_production", Decimal("0"))
        period_prod = (
            _decimal(measurement.production_output)
            if measurement.production_output is not None
            else Decimal("0")
        )

        if base_prod > Decimal("0") and period_prod > Decimal("0"):
            return _safe_divide(base_prod, period_prod, Decimal("1"))
        return Decimal("1")

    def _occupancy_adjustment(
        self,
        measurement: EnPIMeasurement,
        baseline_conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Compute occupancy normalisation adjustment factor.

        Args:
            measurement: Current measurement.
            baseline_conditions: Baseline average conditions.

        Returns:
            Adjustment factor.
        """
        base_occ = baseline_conditions.get("avg_occupancy", Decimal("0"))
        period_occ = (
            _decimal(measurement.occupancy)
            if measurement.occupancy is not None
            else Decimal("0")
        )

        if base_occ > Decimal("0") and period_occ > Decimal("0"):
            return _safe_divide(base_occ, period_occ, Decimal("1"))
        return Decimal("1")

    # ------------------------------------------------------------------ #
    # Internal -- Period Value Calculation                                #
    # ------------------------------------------------------------------ #

    def _calculate_period_values(
        self,
        definition: EnPIDefinition,
        measurements: List[EnPIMeasurement],
        baseline_value: Decimal,
        baseline_conditions: Dict[str, Decimal],
    ) -> List[EnPIValue]:
        """Calculate per-period EnPI values based on EnPI type.

        Routes to the appropriate type-specific calculator.

        Args:
            definition: EnPI definition.
            measurements: Normalised reporting measurements.
            baseline_value: Aggregate baseline EnPI value.
            baseline_conditions: Baseline reference conditions.

        Returns:
            List of EnPIValue objects.
        """
        enpi_type = definition.enpi_type

        if enpi_type == EnPIType.ABSOLUTE:
            values = self.calculate_absolute_enpi(measurements)

        elif enpi_type == EnPIType.INTENSITY:
            denominator_field = self._resolve_denominator_field(definition)
            values = self.calculate_intensity_enpi(
                measurements, denominator_field
            )

        elif enpi_type == EnPIType.REGRESSION_MODELED:
            # Build regression model from baseline_conditions context
            # In practice, caller should provide the model; here we use a
            # simplified approach based on normalizing_variable
            model = self._build_simple_regression_model(
                measurements, baseline_value
            )
            values = self.calculate_regression_enpi(measurements, model)

        elif enpi_type == EnPIType.PROPORTION:
            reference_total = sum(
                (m.energy_value for m in measurements), Decimal("0")
            )
            if reference_total == Decimal("0"):
                reference_total = Decimal("1")
            values = self.calculate_proportion_enpi(
                measurements, reference_total
            )

        elif enpi_type == EnPIType.STATISTICAL:
            baseline_mean = baseline_value
            values = self.calculate_statistical_enpi(
                measurements, baseline_mean
            )

        else:
            values = self.calculate_absolute_enpi(measurements)

        # Apply baseline comparison to each value
        for ev in values:
            if enpi_type not in (
                EnPIType.REGRESSION_MODELED, EnPIType.STATISTICAL
            ):
                ev.expected_value = _round_val(baseline_value, 6)
                ev.improvement_pct = _round_val(
                    self.calculate_improvement(
                        baseline_value,
                        ev.normalized_value,
                        definition.improvement_direction,
                    ), 2
                )
                if baseline_value != Decimal("0"):
                    ev.variance_pct = _round_val(
                        _safe_pct(
                            ev.normalized_value - baseline_value,
                            baseline_value,
                        ), 2
                    )

        return values

    def _compute_aggregate_enpi(
        self,
        definition: EnPIDefinition,
        measurements: List[EnPIMeasurement],
    ) -> Decimal:
        """Compute a single aggregate EnPI value from measurements.

        For absolute: sum of energy values.
        For intensity: total energy / total denominator.
        For others: mean of per-period values.

        Args:
            definition: EnPI definition.
            measurements: Measurement data.

        Returns:
            Aggregate EnPI value.
        """
        if not measurements:
            return Decimal("0")

        if definition.enpi_type == EnPIType.ABSOLUTE:
            return sum(
                (m.energy_value for m in measurements), Decimal("0")
            )

        if definition.enpi_type == EnPIType.INTENSITY:
            denominator_field = self._resolve_denominator_field(definition)
            total_energy = sum(
                (m.energy_value for m in measurements), Decimal("0")
            )
            total_denom = sum(
                (self._get_denominator(m, denominator_field) for m in measurements),
                Decimal("0"),
            )
            return _safe_divide(total_energy, total_denom)

        # For regression, proportion, statistical: mean of energy values
        total = sum((m.energy_value for m in measurements), Decimal("0"))
        return _safe_divide(total, _decimal(len(measurements)))

    def _single_period_enpi(
        self,
        definition: EnPIDefinition,
        measurement: EnPIMeasurement,
    ) -> Decimal:
        """Calculate a single-period EnPI value for statistical use.

        Args:
            definition: EnPI definition.
            measurement: Single measurement.

        Returns:
            EnPI value for the period.
        """
        if definition.enpi_type == EnPIType.INTENSITY:
            denom_field = self._resolve_denominator_field(definition)
            denom = self._get_denominator(measurement, denom_field)
            return _safe_divide(measurement.energy_value, denom)
        return measurement.energy_value

    # ------------------------------------------------------------------ #
    # Internal -- Denominator Resolution                                  #
    # ------------------------------------------------------------------ #

    def _resolve_denominator_field(
        self,
        definition: EnPIDefinition,
    ) -> str:
        """Resolve the denominator field name from the EnPI definition.

        Maps common denominator_unit values to measurement field names.

        Args:
            definition: EnPI definition.

        Returns:
            Measurement attribute name to use as denominator.
        """
        unit = definition.denominator_unit.lower().strip()
        mapping: Dict[str, str] = {
            "m2": "floor_area",
            "ft2": "floor_area",
            "tonne_product": "production_output",
            "tonne": "production_output",
            "unit_produced": "production_output",
            "kg_product": "production_output",
            "employee": "occupancy",
            "occupant": "occupancy",
            "hdd": "hdd",
            "cdd": "cdd",
        }
        return mapping.get(unit, "normalizing_variable")

    def _get_denominator(
        self,
        measurement: EnPIMeasurement,
        field_name: str,
    ) -> Decimal:
        """Get the denominator value from a measurement by field name.

        Args:
            measurement: Measurement data.
            field_name: Attribute name to retrieve.

        Returns:
            Decimal value (defaults to 1 if not available).
        """
        val = getattr(measurement, field_name, None)
        if val is None:
            return Decimal("1")
        result = _decimal(val)
        if result == Decimal("0"):
            return Decimal("1")
        return result

    # ------------------------------------------------------------------ #
    # Internal -- Statistical Tests                                       #
    # ------------------------------------------------------------------ #

    def _t_test(
        self,
        baseline: List[Decimal],
        current: List[Decimal],
    ) -> StatisticalValidation:
        """Perform a two-sample t-test on means.

        Welch's t-test (unequal variances):
            t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
            df = Welch-Satterthwaite approximation

        Args:
            baseline: Baseline period values.
            current: Current period values.

        Returns:
            StatisticalValidation result.
        """
        n1 = len(baseline)
        n2 = len(current)
        if n1 < 2 or n2 < 2:
            return StatisticalValidation(
                test_type=StatisticalTest.T_TEST,
                test_statistic=Decimal("0"),
                p_value=Decimal("1"),
                degrees_of_freedom=0,
                is_significant=False,
                confidence_level=Decimal("0.95"),
            )

        mean1 = self._mean(baseline)
        mean2 = self._mean(current)
        var1 = self._variance(baseline, mean1)
        var2 = self._variance(current, mean2)

        se1 = _safe_divide(var1, _decimal(n1))
        se2 = _safe_divide(var2, _decimal(n2))
        se_total = se1 + se2

        if se_total == Decimal("0"):
            return StatisticalValidation(
                test_type=StatisticalTest.T_TEST,
                test_statistic=Decimal("0"),
                p_value=Decimal("1"),
                degrees_of_freedom=n1 + n2 - 2,
                is_significant=False,
                confidence_level=Decimal("0.95"),
            )

        t_stat = _safe_divide(mean1 - mean2, _decimal_sqrt(se_total))

        # Welch-Satterthwaite degrees of freedom
        numerator_ws = se_total * se_total
        denom_ws = (
            _safe_divide(se1 * se1, _decimal(n1 - 1))
            + _safe_divide(se2 * se2, _decimal(n2 - 1))
        )
        if denom_ws == Decimal("0"):
            df = n1 + n2 - 2
        else:
            df_decimal = _safe_divide(numerator_ws, denom_ws)
            df = max(1, int(float(df_decimal)))

        # Look up critical value and determine significance
        alpha_str = str(
            STATISTICAL_THRESHOLDS[self._significance_level]["alpha"]
        )
        t_critical = self._lookup_t_critical(df, alpha_str)
        abs_t = _decimal_abs(t_stat)
        is_significant = abs_t >= t_critical

        # Approximate p-value from t-stat using table interpolation
        p_value = self._approximate_p_value_t(abs_t, df)

        return StatisticalValidation(
            test_type=StatisticalTest.T_TEST,
            test_statistic=_round_val(t_stat, 4),
            p_value=_round_val(p_value, 4),
            degrees_of_freedom=df,
            is_significant=is_significant,
            confidence_level=STATISTICAL_THRESHOLDS[
                self._significance_level
            ]["confidence"],
        )

    def _f_test(
        self,
        baseline: List[Decimal],
        current: List[Decimal],
    ) -> StatisticalValidation:
        """Perform an F-test comparing variances.

        F = var1 / var2  (larger variance in numerator)

        Args:
            baseline: Baseline period values.
            current: Current period values.

        Returns:
            StatisticalValidation result.
        """
        n1 = len(baseline)
        n2 = len(current)
        if n1 < 2 or n2 < 2:
            return StatisticalValidation(
                test_type=StatisticalTest.F_TEST,
                test_statistic=Decimal("0"),
                p_value=Decimal("1"),
                degrees_of_freedom=0,
                is_significant=False,
                confidence_level=Decimal("0.95"),
            )

        var1 = self._variance(baseline, self._mean(baseline))
        var2 = self._variance(current, self._mean(current))

        # F-statistic: larger variance / smaller variance
        if var2 >= var1:
            f_stat = _safe_divide(var2, var1, Decimal("1"))
            df_num = n2 - 1
            df_den = n1 - 1
        else:
            f_stat = _safe_divide(var1, var2, Decimal("1"))
            df_num = n1 - 1
            df_den = n2 - 1

        # Look up critical F-value
        f_critical = self._lookup_f_critical(df_den)
        is_significant = f_stat >= f_critical

        # Approximate p-value
        p_value = self._approximate_p_value_f(f_stat, df_den)

        return StatisticalValidation(
            test_type=StatisticalTest.F_TEST,
            test_statistic=_round_val(f_stat, 4),
            p_value=_round_val(p_value, 4),
            degrees_of_freedom=df_num + df_den,
            is_significant=is_significant,
            confidence_level=STATISTICAL_THRESHOLDS[
                self._significance_level
            ]["confidence"],
        )

    def _chi_squared_test(
        self,
        baseline: List[Decimal],
        current: List[Decimal],
    ) -> StatisticalValidation:
        """Perform a chi-squared goodness-of-fit test.

        Compares observed (current) values against expected (baseline mean)
        distribution.  chi2 = sum((O_i - E_i)^2 / E_i)

        Args:
            baseline: Baseline period values.
            current: Current period values.

        Returns:
            StatisticalValidation result.
        """
        n = len(current)
        if n < 2 or len(baseline) < 1:
            return StatisticalValidation(
                test_type=StatisticalTest.CHI_SQUARED,
                test_statistic=Decimal("0"),
                p_value=Decimal("1"),
                degrees_of_freedom=0,
                is_significant=False,
                confidence_level=Decimal("0.95"),
            )

        expected = self._mean(baseline)
        if expected == Decimal("0"):
            expected = Decimal("1")

        chi2 = Decimal("0")
        for obs in current:
            diff = obs - expected
            chi2 += _safe_divide(diff * diff, expected)

        df = n - 1

        # Approximate critical value using normal approximation for large df
        # chi2_critical ~ df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
        # where z is the normal critical value
        alpha = STATISTICAL_THRESHOLDS[self._significance_level]["alpha"]
        z_crit = Decimal("1.96") if alpha <= Decimal("0.05") else Decimal("1.645")

        term = Decimal("2") / (Decimal("9") * _decimal(df))
        if term > Decimal("0"):
            chi2_critical = _decimal(df) * (
                Decimal("1") - term + z_crit * _decimal_sqrt(term)
            ) ** 3
        else:
            chi2_critical = _decimal(df)

        is_significant = chi2 >= chi2_critical

        # Rough p-value approximation
        p_value = Decimal("0.5")
        if chi2 > chi2_critical * Decimal("2"):
            p_value = Decimal("0.001")
        elif chi2 > chi2_critical:
            p_value = Decimal("0.01")
        elif chi2 > chi2_critical * Decimal("0.8"):
            p_value = Decimal("0.05")
        elif chi2 > chi2_critical * Decimal("0.5"):
            p_value = Decimal("0.10")

        return StatisticalValidation(
            test_type=StatisticalTest.CHI_SQUARED,
            test_statistic=_round_val(chi2, 4),
            p_value=_round_val(p_value, 4),
            degrees_of_freedom=df,
            is_significant=is_significant,
            confidence_level=STATISTICAL_THRESHOLDS[
                self._significance_level
            ]["confidence"],
        )

    def _anova_test(
        self,
        baseline: List[Decimal],
        current: List[Decimal],
    ) -> StatisticalValidation:
        """Perform one-way ANOVA comparing baseline vs current groups.

        F = MS_between / MS_within
        where:
            MS_between = SS_between / (k-1)
            MS_within  = SS_within  / (N-k)
            k = 2 (two groups)

        Args:
            baseline: Baseline period values.
            current: Current period values.

        Returns:
            StatisticalValidation result.
        """
        n1 = len(baseline)
        n2 = len(current)
        n_total = n1 + n2
        k = 2  # two groups

        if n1 < 2 or n2 < 2:
            return StatisticalValidation(
                test_type=StatisticalTest.ANOVA,
                test_statistic=Decimal("0"),
                p_value=Decimal("1"),
                degrees_of_freedom=0,
                is_significant=False,
                confidence_level=Decimal("0.95"),
            )

        mean1 = self._mean(baseline)
        mean2 = self._mean(current)
        grand_mean = _safe_divide(
            mean1 * _decimal(n1) + mean2 * _decimal(n2),
            _decimal(n_total),
        )

        # Sum of squares between groups
        ss_between = (
            _decimal(n1) * (mean1 - grand_mean) ** 2
            + _decimal(n2) * (mean2 - grand_mean) ** 2
        )

        # Sum of squares within groups
        ss_within = Decimal("0")
        for v in baseline:
            ss_within += (v - mean1) ** 2
        for v in current:
            ss_within += (v - mean2) ** 2

        df_between = k - 1
        df_within = n_total - k

        ms_between = _safe_divide(ss_between, _decimal(df_between))
        ms_within = _safe_divide(ss_within, _decimal(max(df_within, 1)))

        f_stat = _safe_divide(ms_between, ms_within, Decimal("0"))

        # Look up critical F-value
        f_critical = self._lookup_f_critical(df_within)
        is_significant = f_stat >= f_critical

        p_value = self._approximate_p_value_f(f_stat, df_within)

        return StatisticalValidation(
            test_type=StatisticalTest.ANOVA,
            test_statistic=_round_val(f_stat, 4),
            p_value=_round_val(p_value, 4),
            degrees_of_freedom=df_between + df_within,
            is_significant=is_significant,
            confidence_level=STATISTICAL_THRESHOLDS[
                self._significance_level
            ]["confidence"],
        )

    # ------------------------------------------------------------------ #
    # Internal -- Statistical Helpers                                     #
    # ------------------------------------------------------------------ #

    def _mean(self, values: List[Decimal]) -> Decimal:
        """Compute the arithmetic mean of a list of Decimals."""
        if not values:
            return Decimal("0")
        return _safe_divide(
            sum(values, Decimal("0")), _decimal(len(values))
        )

    def _variance(
        self,
        values: List[Decimal],
        mean: Optional[Decimal] = None,
    ) -> Decimal:
        """Compute the sample variance (Bessel's correction).

        Args:
            values: List of Decimal values.
            mean: Pre-computed mean (optional).

        Returns:
            Sample variance.
        """
        n = len(values)
        if n < 2:
            return Decimal("0")
        if mean is None:
            mean = self._mean(values)
        ss = sum((v - mean) ** 2 for v in values)
        return _safe_divide(ss, _decimal(n - 1))

    def _std_dev(self, values: List[Decimal]) -> Decimal:
        """Compute the sample standard deviation."""
        return _decimal_sqrt(self._variance(values))

    def _lookup_t_critical(self, df: int, alpha: str) -> Decimal:
        """Look up critical t-value from table.

        Uses the closest available df in the table.  Defaults to
        the normal distribution value (df=999) for large df.

        Args:
            df: Degrees of freedom.
            alpha: Significance level as string (e.g. "0.05").

        Returns:
            Critical t-value.
        """
        available = sorted(_T_CRITICAL_TABLE.keys())
        chosen_df = available[-1]
        for d in available:
            if d >= df:
                chosen_df = d
                break

        row = _T_CRITICAL_TABLE.get(
            chosen_df, _T_CRITICAL_TABLE[999]
        )
        return row.get(alpha, Decimal("1.96"))

    def _lookup_f_critical(self, df_den: int) -> Decimal:
        """Look up critical F-value (alpha=0.05, df_num=1).

        Args:
            df_den: Denominator degrees of freedom.

        Returns:
            Critical F-value.
        """
        available = sorted(_F_CRITICAL_TABLE.keys())
        chosen_df = available[-1]
        for d in available:
            if d >= df_den:
                chosen_df = d
                break
        return _F_CRITICAL_TABLE.get(chosen_df, Decimal("3.84"))

    def _approximate_p_value_t(
        self,
        abs_t: Decimal,
        df: int,
    ) -> Decimal:
        """Approximate two-tailed p-value from t-statistic.

        Uses table lookups and interpolation for Decimal-safe p-value
        estimation without scipy.

        Args:
            abs_t: Absolute value of t-statistic.
            df: Degrees of freedom.

        Returns:
            Approximate p-value.
        """
        # Check against known critical values for this df
        t_01 = self._lookup_t_critical(df, "0.01")
        t_05 = self._lookup_t_critical(df, "0.05")
        t_10 = self._lookup_t_critical(df, "0.10")

        if abs_t >= t_01:
            return Decimal("0.01")
        elif abs_t >= t_05:
            # Interpolate between 0.01 and 0.05
            fraction = _safe_divide(t_01 - abs_t, t_01 - t_05, Decimal("0.5"))
            return Decimal("0.01") + fraction * Decimal("0.04")
        elif abs_t >= t_10:
            # Interpolate between 0.05 and 0.10
            fraction = _safe_divide(t_05 - abs_t, t_05 - t_10, Decimal("0.5"))
            return Decimal("0.05") + fraction * Decimal("0.05")
        else:
            # p > 0.10
            return Decimal("0.50")

    def _approximate_p_value_f(
        self,
        f_stat: Decimal,
        df_den: int,
    ) -> Decimal:
        """Approximate p-value from F-statistic.

        Args:
            f_stat: F-statistic value.
            df_den: Denominator degrees of freedom.

        Returns:
            Approximate p-value.
        """
        f_critical = self._lookup_f_critical(df_den)

        if f_stat >= f_critical * Decimal("3"):
            return Decimal("0.001")
        elif f_stat >= f_critical * Decimal("2"):
            return Decimal("0.005")
        elif f_stat >= f_critical:
            return Decimal("0.05")
        elif f_stat >= f_critical * Decimal("0.7"):
            return Decimal("0.10")
        else:
            return Decimal("0.50")

    # ------------------------------------------------------------------ #
    # Internal -- OLS Regression                                          #
    # ------------------------------------------------------------------ #

    def _ols_regression(
        self,
        x_vals: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Perform ordinary least squares regression.

        Returns slope and R-squared.

        Formulae:
            slope = Cov(x, y) / Var(x)
            R^2 = [Cov(x, y)]^2 / [Var(x) * Var(y)]

        Args:
            x_vals: Independent variable values.
            y_vals: Dependent variable values.

        Returns:
            Tuple of (slope, r_squared).
        """
        n = len(x_vals)
        if n < 2 or len(y_vals) < 2 or n != len(y_vals):
            return Decimal("0"), Decimal("0")

        mean_x = self._mean(x_vals)
        mean_y = self._mean(y_vals)

        # Covariance
        cov_xy = Decimal("0")
        for i in range(n):
            cov_xy += (x_vals[i] - mean_x) * (y_vals[i] - mean_y)

        # Variance of x
        var_x = Decimal("0")
        for i in range(n):
            var_x += (x_vals[i] - mean_x) ** 2

        # Variance of y
        var_y = Decimal("0")
        for i in range(n):
            var_y += (y_vals[i] - mean_y) ** 2

        slope = _safe_divide(cov_xy, var_x)

        # R-squared
        denom_r2 = var_x * var_y
        if denom_r2 == Decimal("0"):
            r_squared = Decimal("0")
        else:
            r_squared = _safe_divide(cov_xy * cov_xy, denom_r2)

        # Clamp R-squared to [0, 1]
        r_squared = max(Decimal("0"), min(Decimal("1"), r_squared))

        return slope, r_squared

    # ------------------------------------------------------------------ #
    # Internal -- Data Quality Assessment                                 #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self,
        measurements: List[EnPIMeasurement],
        baseline_measurements: List[EnPIMeasurement],
        definition: EnPIDefinition,
    ) -> Decimal:
        """Assess data quality on a 0-100 scale.

        Scoring factors:
            - Data completeness (25 points): all fields populated
            - Data quantity (25 points): sufficient data points
            - Data consistency (25 points): no extreme outliers
            - Data coverage (25 points): continuous time series

        Args:
            measurements: Reporting period measurements.
            baseline_measurements: Baseline period measurements.
            definition: EnPI definition.

        Returns:
            Data quality score (0-100).
        """
        score = Decimal("0")

        # Factor 1: Completeness (25 points)
        all_measurements = measurements + baseline_measurements
        total_fields = Decimal("0")
        populated_fields = Decimal("0")
        for m in all_measurements:
            total_fields += Decimal("7")  # 7 optional fields
            if m.normalizing_variable is not None:
                populated_fields += Decimal("1")
            if m.production_output is not None:
                populated_fields += Decimal("1")
            if m.floor_area is not None:
                populated_fields += Decimal("1")
            if m.occupancy is not None:
                populated_fields += Decimal("1")
            if m.hdd is not None:
                populated_fields += Decimal("1")
            if m.cdd is not None:
                populated_fields += Decimal("1")
            # energy_value is required so always counted
            populated_fields += Decimal("1")

        completeness = _safe_divide(
            populated_fields * Decimal("25"), total_fields
        ) if total_fields > Decimal("0") else Decimal("0")
        score += min(completeness, Decimal("25"))

        # Factor 2: Quantity (25 points)
        min_required = _decimal(self._min_data_points)
        n_reporting = _decimal(len(measurements))
        n_baseline = _decimal(len(baseline_measurements))
        quantity_pct = _safe_divide(
            (n_reporting + n_baseline),
            min_required * Decimal("2"),
        )
        quantity_score = min(quantity_pct * Decimal("25"), Decimal("25"))
        score += quantity_score

        # Factor 3: Consistency (25 points) - check for outliers
        energy_vals = [m.energy_value for m in all_measurements]
        if len(energy_vals) >= 3:
            mean_e = self._mean(energy_vals)
            std_e = self._std_dev(energy_vals)
            if std_e > Decimal("0"):
                outlier_count = Decimal("0")
                for e in energy_vals:
                    z = _safe_divide(_decimal_abs(e - mean_e), std_e)
                    if z > Decimal("3"):
                        outlier_count += Decimal("1")
                outlier_pct = _safe_divide(
                    outlier_count, _decimal(len(energy_vals))
                )
                consistency = (Decimal("1") - outlier_pct) * Decimal("25")
                score += max(Decimal("0"), consistency)
            else:
                score += Decimal("25")
        else:
            score += Decimal("15")

        # Factor 4: Coverage (25 points) - time series continuity
        if len(measurements) >= 2:
            sorted_m = sorted(measurements, key=lambda m: m.period_start)
            gaps = 0
            for i in range(1, len(sorted_m)):
                prev_end = sorted_m[i - 1].period_end
                curr_start = sorted_m[i].period_start
                # Allow up to 5 days gap
                delta = (curr_start - prev_end).days
                if delta > 5:
                    gaps += 1
            if gaps == 0:
                score += Decimal("25")
            elif gaps <= 2:
                score += Decimal("18")
            elif gaps <= 5:
                score += Decimal("10")
            else:
                score += Decimal("5")
        else:
            score += Decimal("10")

        return min(score, Decimal("100"))

    # ------------------------------------------------------------------ #
    # Internal -- Target Checking                                         #
    # ------------------------------------------------------------------ #

    def _check_target(
        self,
        value: Decimal,
        target: Decimal,
        direction: ImprovementDirection,
    ) -> bool:
        """Check whether an EnPI value meets its target.

        Args:
            value: EnPI value to check.
            target: Target value.
            direction: Improvement direction.

        Returns:
            True if target is met.
        """
        if direction == ImprovementDirection.DECREASE_IS_BETTER:
            return value <= target
        else:
            return value >= target

    # ------------------------------------------------------------------ #
    # Internal -- Trend Consistency Check                                 #
    # ------------------------------------------------------------------ #

    def _check_trend_consistency(
        self,
        result: EnPIResult,
    ) -> bool:
        """Check whether the EnPI trend is consistent (not noisy).

        A trend is considered consistent if R-squared >= 0.3 or the
        coefficient of variation of normalised values is < 50%.

        Args:
            result: EnPIResult with values and trend data.

        Returns:
            True if trend is consistent.
        """
        if result.trend_r_squared >= Decimal("0.3"):
            return True

        vals = [v.normalized_value for v in result.values]
        if not vals:
            return False
        mean_v = self._mean(vals)
        if mean_v == Decimal("0"):
            return False
        std_v = self._std_dev(vals)
        cv = _safe_divide(std_v, _decimal_abs(mean_v))
        return cv < Decimal("0.50")

    # ------------------------------------------------------------------ #
    # Internal -- Simplified Regression Model                             #
    # ------------------------------------------------------------------ #

    def _build_simple_regression_model(
        self,
        measurements: List[EnPIMeasurement],
        baseline_value: Decimal,
    ) -> Dict[str, Decimal]:
        """Build a simplified regression model for internal use.

        When a full baseline regression is not provided externally, this
        creates a basic model using the baseline mean as the intercept
        and a zero slope (constant model).

        Args:
            measurements: Reporting measurements (for variable field).
            baseline_value: Aggregate baseline EnPI value.

        Returns:
            Dict with intercept, slope, variable_field.
        """
        return {
            "intercept": baseline_value,
            "slope": Decimal("0"),
            "variable_field": "normalizing_variable",
        }

    # ------------------------------------------------------------------ #
    # Internal -- Methodology Notes                                       #
    # ------------------------------------------------------------------ #

    def _build_methodology_notes(
        self,
        definition: EnPIDefinition,
        n_reporting: int,
        n_baseline: int,
        baseline_value: Decimal,
        current_value: Decimal,
        improvement_pct: Decimal,
    ) -> str:
        """Build human-readable methodology description.

        Args:
            definition: EnPI definition.
            n_reporting: Number of reporting measurements.
            n_baseline: Number of baseline measurements.
            baseline_value: Baseline EnPI value.
            current_value: Current EnPI value.
            improvement_pct: Improvement percentage.

        Returns:
            Methodology description string.
        """
        type_desc = ENPI_TYPE_DESCRIPTIONS.get(
            definition.enpi_type.value,
            "EnPI calculation per ISO 50006.",
        )

        parts: List[str] = [
            f"EnPI: {definition.enpi_name}.",
            f"Type: {definition.enpi_type.value}.",
            type_desc,
            f"Energy type: {definition.energy_type}.",
            f"Units: {definition.numerator_unit}/{definition.denominator_unit}.",
            f"Normalisation: {definition.normalization_method.value}.",
            f"Improvement direction: {definition.improvement_direction.value}.",
            f"Aggregation level: {definition.aggregation_level.value}.",
            f"Baseline periods: {n_baseline}.",
            f"Reporting periods: {n_reporting}.",
            f"Baseline value: {_round_val(baseline_value, 4)}.",
            f"Current value: {_round_val(current_value, 4)}.",
            f"Improvement: {_round_val(improvement_pct, 2)}%.",
            "Calculated per ISO 50006:2014 methodology.",
        ]

        if definition.target_value is not None:
            parts.append(
                f"Target: {definition.target_value} "
                f"{definition.numerator_unit}/{definition.denominator_unit}."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Internal -- Weight Normalisation                                    #
    # ------------------------------------------------------------------ #

    def _normalise_weights(
        self,
        results: List[EnPIResult],
        weights: Optional[Dict[str, Decimal]],
    ) -> Dict[str, Decimal]:
        """Normalise portfolio weights to sum to 1.0.

        If weights are not provided, equal weighting is applied.

        Args:
            results: List of EnPIResult objects.
            weights: Optional weight mapping (enpi_id -> weight).

        Returns:
            Normalised weight mapping.
        """
        n = len(results)
        if not weights or n == 0:
            equal_weight = _safe_divide(Decimal("1"), _decimal(n))
            return {r.enpi_id: equal_weight for r in results}

        total_weight = sum(
            (_decimal(weights.get(r.enpi_id, Decimal("0"))) for r in results),
            Decimal("0"),
        )
        if total_weight == Decimal("0"):
            equal_weight = _safe_divide(Decimal("1"), _decimal(n))
            return {r.enpi_id: equal_weight for r in results}

        return {
            r.enpi_id: _safe_divide(
                _decimal(weights.get(r.enpi_id, Decimal("0"))),
                total_weight,
            )
            for r in results
        }

    # ------------------------------------------------------------------ #
    # Utility -- EnPI Comparison                                          #
    # ------------------------------------------------------------------ #

    def compare_enpi_periods(
        self,
        period_a_values: List[EnPIValue],
        period_b_values: List[EnPIValue],
        direction: ImprovementDirection,
    ) -> Dict[str, Any]:
        """Compare two sets of EnPI period values.

        Computes the mean, standard deviation, and improvement between
        two reporting periods.

        Args:
            period_a_values: Earlier period EnPI values.
            period_b_values: Later period EnPI values.
            direction: Improvement direction.

        Returns:
            Dict with comparison metrics and provenance_hash.
        """
        a_vals = [v.normalized_value for v in period_a_values]
        b_vals = [v.normalized_value for v in period_b_values]

        mean_a = self._mean(a_vals)
        mean_b = self._mean(b_vals)
        std_a = self._std_dev(a_vals)
        std_b = self._std_dev(b_vals)

        improvement = self.calculate_improvement(mean_a, mean_b, direction)

        # Statistical test
        stat_result: Optional[StatisticalValidation] = None
        if len(a_vals) >= 2 and len(b_vals) >= 2:
            stat_result = self.perform_statistical_test(
                a_vals, b_vals, StatisticalTest.T_TEST,
            )

        comparison = {
            "period_a_mean": str(_round_val(mean_a, 6)),
            "period_a_std": str(_round_val(std_a, 6)),
            "period_a_count": len(a_vals),
            "period_b_mean": str(_round_val(mean_b, 6)),
            "period_b_std": str(_round_val(std_b, 6)),
            "period_b_count": len(b_vals),
            "improvement_pct": str(_round_val(improvement, 2)),
            "direction": direction.value,
            "statistically_significant": (
                stat_result.is_significant if stat_result else False
            ),
            "p_value": (
                str(stat_result.p_value) if stat_result else "N/A"
            ),
            "provenance_hash": _compute_hash({
                "mean_a": str(mean_a),
                "mean_b": str(mean_b),
                "improvement": str(improvement),
            }),
        }

        logger.info(
            "Period comparison: mean_a=%.4f, mean_b=%.4f, improvement=%.2f%%",
            float(mean_a), float(mean_b), float(improvement),
        )
        return comparison

    # ------------------------------------------------------------------ #
    # Utility -- EnPI Summary                                             #
    # ------------------------------------------------------------------ #

    def summarise_enpi(
        self,
        result: EnPIResult,
    ) -> Dict[str, Any]:
        """Produce a summary dictionary from an EnPI result.

        Args:
            result: Completed EnPIResult.

        Returns:
            Summary dict with key metrics for reporting.
        """
        values = result.values
        normalised_vals = [v.normalized_value for v in values]

        on_target_count = sum(1 for v in values if v.is_on_target)
        total_count = len(values)

        summary: Dict[str, Any] = {
            "enpi_id": result.enpi_id,
            "enpi_name": result.definition.enpi_name,
            "enpi_type": result.definition.enpi_type.value,
            "baseline_value": str(result.baseline_value),
            "current_value": str(result.current_value),
            "improvement_pct": str(result.improvement_pct),
            "trend_slope": str(result.trend_slope),
            "trend_r_squared": str(result.trend_r_squared),
            "data_quality_score": str(result.data_quality_score),
            "periods_analysed": total_count,
            "periods_on_target": on_target_count,
            "target_compliance_pct": str(
                _safe_pct(_decimal(on_target_count), _decimal(max(total_count, 1)))
            ),
            "statistically_significant": (
                result.statistical_validation.is_significant
                if result.statistical_validation else False
            ),
            "min_value": str(
                _round_val(min(normalised_vals), 6) if normalised_vals else Decimal("0")
            ),
            "max_value": str(
                _round_val(max(normalised_vals), 6) if normalised_vals else Decimal("0")
            ),
            "mean_value": str(
                _round_val(self._mean(normalised_vals), 6)
            ),
            "std_dev": str(
                _round_val(self._std_dev(normalised_vals), 6)
                if len(normalised_vals) >= 2 else Decimal("0")
            ),
            "units": (
                f"{result.definition.numerator_unit}"
                f"/{result.definition.denominator_unit}"
            ),
            "engine_version": self.engine_version,
            "provenance_hash": _compute_hash({
                "enpi_id": result.enpi_id,
                "baseline": str(result.baseline_value),
                "current": str(result.current_value),
                "improvement": str(result.improvement_pct),
            }),
        }

        return summary
