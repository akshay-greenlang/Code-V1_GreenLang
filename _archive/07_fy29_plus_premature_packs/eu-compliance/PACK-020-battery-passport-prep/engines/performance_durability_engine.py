# -*- coding: utf-8 -*-
"""
PerformanceDurabilityEngine - PACK-020 Battery Passport Prep Engine 4
======================================================================

Calculates and validates battery performance and durability metrics per
EU Battery Regulation Art 10 and Annex IV.

Under Regulation (EU) 2023/1542 (the EU Battery Regulation), Article 10
mandates that rechargeable industrial batteries with a capacity above
2 kWh, EV batteries, and LMT batteries shall meet minimum performance
and durability requirements set out in Annex IV.  The requirements
address electrochemical performance parameters, expected lifetime,
and State of Health (SoH) monitoring.

Regulation (EU) 2023/1542 Framework:
    - Art 10(1): From 18 August 2028, rechargeable industrial batteries
      with a capacity above 2 kWh, EV batteries, and LMT batteries
      placed on the market or put into service shall meet the
      performance and durability requirements set out in Annex IV.
    - Art 10(2): The Commission shall adopt delegated acts specifying
      the minimum values for the electrochemical performance and
      durability parameters in Annex IV.
    - Art 10(3): Batteries shall be accompanied by technical
      documentation demonstrating compliance with Annex IV.
    - Annex IV: Electrochemical performance and durability requirements
      covering rated capacity, internal resistance, cycle life,
      calendar life, energy efficiency, and State of Health.

Annex IV Performance Parameters:
    - Rated capacity (Ah and kWh)
    - Minimum remaining capacity after specified cycles
    - Voltage characteristics (nominal, min, max)
    - Power capability (W and W/kg)
    - Expected cycle life (number of charge/discharge cycles)
    - Expected calendar life (years)
    - Round-trip energy efficiency (%)
    - Internal resistance (mOhm) and its evolution
    - State of Health (SoH) at key milestones
    - State of Charge (SoC) operating range
    - C-rate capabilities
    - Operating temperature range

Durability Rating Thresholds (based on SoH):
    - EXCELLENT: >= 95% SoH
    - GOOD: >= 85% SoH
    - ACCEPTABLE: >= 75% SoH
    - POOR: >= 65% SoH
    - CRITICAL: < 65% SoH

Regulatory References:
    - Regulation (EU) 2023/1542 of the European Parliament and of the
      Council of 12 July 2023 concerning batteries and waste batteries
    - Art 10 - Performance and durability requirements
    - Annex IV - Electrochemical performance and durability parameters
    - IEC 62660-1:2018 - Secondary lithium-ion cells for EV
    - IEC 62620:2014 - Secondary cells for industrial applications
    - ISO 12405 series - Electrically propelled road vehicles

Zero-Hallucination:
    - SoH calculation uses deterministic capacity ratio
    - Durability rating uses rule-based threshold comparison
    - Cycle life assessment uses deterministic percentage calculation
    - Efficiency assessment uses deterministic comparison
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep
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
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PerformanceMetric(str, Enum):
    """Performance metric identifier per Annex IV.

    Enumerates the electrochemical performance and durability parameters
    required for battery passport and compliance assessment under
    Article 10 of the EU Battery Regulation.
    """
    RATED_CAPACITY = "rated_capacity"
    MIN_CAPACITY = "min_capacity"
    REMAINING_CAPACITY = "remaining_capacity"
    VOLTAGE_NOMINAL = "voltage_nominal"
    VOLTAGE_MIN = "voltage_min"
    VOLTAGE_MAX = "voltage_max"
    POWER_CAPABILITY = "power_capability"
    CYCLE_LIFE = "cycle_life"
    CALENDAR_LIFE = "calendar_life"
    EFFICIENCY = "efficiency"
    INTERNAL_RESISTANCE = "internal_resistance"
    SOH = "soh"
    SOC = "soc"
    C_RATE = "c_rate"
    TEMPERATURE_RANGE = "temperature_range"

class DurabilityRating(str, Enum):
    """Durability rating based on State of Health assessment.

    Categorises the battery's overall health and remaining useful
    life based on the current State of Health (SoH) percentage.
    Higher ratings indicate better remaining performance.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class MetricStatus(str, Enum):
    """Validation status for an individual performance metric.

    Indicates whether a metric value falls within acceptable ranges
    as defined by Annex IV and manufacturer specifications.
    """
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"

class BatteryLifeStage(str, Enum):
    """Stage in the battery's service life.

    Identifies where the battery is in its lifecycle, which affects
    the applicable performance thresholds and assessment criteria.
    """
    NEW = "new"
    IN_SERVICE = "in_service"
    MID_LIFE = "mid_life"
    END_OF_FIRST_LIFE = "end_of_first_life"
    SECOND_LIFE = "second_life"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# State of Health thresholds for durability rating.
SOH_THRESHOLDS: Dict[str, Decimal] = {
    DurabilityRating.EXCELLENT.value: Decimal("95"),
    DurabilityRating.GOOD.value: Decimal("85"),
    DurabilityRating.ACCEPTABLE.value: Decimal("75"),
    DurabilityRating.POOR.value: Decimal("65"),
    # CRITICAL is below POOR threshold
}

# Minimum efficiency thresholds by category.
# Values are round-trip energy efficiency percentages.
MIN_EFFICIENCY_THRESHOLDS: Dict[str, Decimal] = {
    "ev": Decimal("85"),
    "industrial": Decimal("80"),
    "lmt": Decimal("82"),
    "portable": Decimal("75"),
    "sli": Decimal("70"),
}

# Expected cycle life ranges by chemistry for plausibility checks.
CHEMISTRY_CYCLE_LIFE: Dict[str, Dict[str, int]] = {
    "nmc": {"low": 500, "typical": 1500, "high": 3000},
    "nca": {"low": 500, "typical": 1500, "high": 2500},
    "lfp": {"low": 2000, "typical": 4000, "high": 8000},
    "nmc811": {"low": 500, "typical": 1500, "high": 3000},
    "nmc622": {"low": 600, "typical": 1500, "high": 3000},
    "nmc532": {"low": 700, "typical": 2000, "high": 3500},
    "lmo": {"low": 300, "typical": 700, "high": 1500},
    "lto": {"low": 5000, "typical": 15000, "high": 30000},
    "lead_acid": {"low": 200, "typical": 500, "high": 1200},
    "nimh": {"low": 300, "typical": 800, "high": 2000},
    "sodium_ion": {"low": 1000, "typical": 3000, "high": 6000},
    "solid_state": {"low": 1000, "typical": 3000, "high": 10000},
}

# Operating temperature range limits for plausibility checks (degC).
TEMPERATURE_PLAUSIBILITY: Dict[str, Dict[str, int]] = {
    "min_absolute": {"value": -40, "description": "Absolute minimum operating temperature"},
    "max_absolute": {"value": 60, "description": "Absolute maximum operating temperature"},
    "typical_min": {"value": -20, "description": "Typical minimum operating temperature"},
    "typical_max": {"value": 45, "description": "Typical maximum operating temperature"},
}

# Performance metric labels and units.
METRIC_LABELS: Dict[str, Dict[str, str]] = {
    PerformanceMetric.RATED_CAPACITY.value: {
        "label": "Rated Capacity",
        "unit": "Ah",
        "annex_ref": "Annex IV, Point 1",
    },
    PerformanceMetric.MIN_CAPACITY.value: {
        "label": "Minimum Remaining Capacity",
        "unit": "Ah",
        "annex_ref": "Annex IV, Point 2",
    },
    PerformanceMetric.REMAINING_CAPACITY.value: {
        "label": "Current Remaining Capacity",
        "unit": "Ah",
        "annex_ref": "Annex IV, Point 2",
    },
    PerformanceMetric.VOLTAGE_NOMINAL.value: {
        "label": "Nominal Voltage",
        "unit": "V",
        "annex_ref": "Annex IV, Point 3",
    },
    PerformanceMetric.VOLTAGE_MIN.value: {
        "label": "Minimum Voltage",
        "unit": "V",
        "annex_ref": "Annex IV, Point 3",
    },
    PerformanceMetric.VOLTAGE_MAX.value: {
        "label": "Maximum Voltage",
        "unit": "V",
        "annex_ref": "Annex IV, Point 3",
    },
    PerformanceMetric.POWER_CAPABILITY.value: {
        "label": "Power Capability",
        "unit": "W",
        "annex_ref": "Annex IV, Point 4",
    },
    PerformanceMetric.CYCLE_LIFE.value: {
        "label": "Cycle Life",
        "unit": "cycles",
        "annex_ref": "Annex IV, Point 5",
    },
    PerformanceMetric.CALENDAR_LIFE.value: {
        "label": "Calendar Life",
        "unit": "years",
        "annex_ref": "Annex IV, Point 6",
    },
    PerformanceMetric.EFFICIENCY.value: {
        "label": "Round-trip Energy Efficiency",
        "unit": "%",
        "annex_ref": "Annex IV, Point 7",
    },
    PerformanceMetric.INTERNAL_RESISTANCE.value: {
        "label": "Internal Resistance",
        "unit": "mOhm",
        "annex_ref": "Annex IV, Point 8",
    },
    PerformanceMetric.SOH.value: {
        "label": "State of Health",
        "unit": "%",
        "annex_ref": "Annex IV, Point 9",
    },
    PerformanceMetric.SOC.value: {
        "label": "State of Charge",
        "unit": "%",
        "annex_ref": "Annex IV, Point 10",
    },
    PerformanceMetric.C_RATE.value: {
        "label": "C-Rate",
        "unit": "C",
        "annex_ref": "Annex IV, Point 11",
    },
    PerformanceMetric.TEMPERATURE_RANGE.value: {
        "label": "Operating Temperature Range",
        "unit": "degC",
        "annex_ref": "Annex IV, Point 12",
    },
}

# Durability rating descriptions.
RATING_DESCRIPTIONS: Dict[str, str] = {
    DurabilityRating.EXCELLENT.value: (
        "Battery health is excellent (SoH >= 95%). No degradation "
        "concerns; battery performs at or near rated specifications."
    ),
    DurabilityRating.GOOD.value: (
        "Battery health is good (SoH >= 85%). Normal degradation "
        "expected at this stage of service life."
    ),
    DurabilityRating.ACCEPTABLE.value: (
        "Battery health is acceptable (SoH >= 75%). Degradation is "
        "within limits but approaching first-life end threshold."
    ),
    DurabilityRating.POOR.value: (
        "Battery health is poor (SoH >= 65%). Significant degradation; "
        "consider planning for replacement or second-life application."
    ),
    DurabilityRating.CRITICAL.value: (
        "Battery health is critical (SoH < 65%). Battery has degraded "
        "beyond the first-life end threshold. Immediate assessment for "
        "replacement, second-life, or recycling is recommended."
    ),
}

# Second-life SoH threshold: below this, the battery should be
# evaluated for second-life applications or recycling.
SECOND_LIFE_SOH_THRESHOLD: Decimal = Decimal("80")

# End-of-first-life SoH threshold (below this is typically
# considered end of first life for EV batteries).
END_OF_FIRST_LIFE_SOH: Decimal = Decimal("70")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PerformanceInput(BaseModel):
    """Input data for battery performance assessment per Art 10.

    Contains all measured and rated performance parameters for a
    battery, used to assess compliance with Annex IV requirements.
    """
    battery_id: str = Field(
        ...,
        description="Unique battery identifier",
        min_length=1,
        max_length=200,
    )
    category: str = Field(
        default="ev",
        description="Battery category (ev, industrial, lmt, portable, sli)",
        max_length=50,
    )
    chemistry: str = Field(
        default="nmc",
        description="Battery chemistry type",
        max_length=50,
    )
    rated_capacity_ah: Optional[Decimal] = Field(
        default=None,
        description="Rated capacity in Ah",
        ge=0,
    )
    current_capacity_ah: Optional[Decimal] = Field(
        default=None,
        description="Current measured capacity in Ah",
        ge=0,
    )
    min_capacity_ah: Optional[Decimal] = Field(
        default=None,
        description="Minimum acceptable capacity in Ah (per spec)",
        ge=0,
    )
    voltage_nominal: Optional[Decimal] = Field(
        default=None,
        description="Nominal voltage in volts",
        ge=0,
    )
    voltage_min: Optional[Decimal] = Field(
        default=None,
        description="Minimum operating voltage in volts",
        ge=0,
    )
    voltage_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum operating voltage in volts",
        ge=0,
    )
    power_capability_w: Optional[Decimal] = Field(
        default=None,
        description="Power capability in watts",
        ge=0,
    )
    cycle_life_expected: Optional[int] = Field(
        default=None,
        description="Expected cycle life (number of cycles)",
        ge=0,
    )
    cycles_completed: Optional[int] = Field(
        default=None,
        description="Number of cycles completed to date",
        ge=0,
    )
    calendar_life_years: Optional[Decimal] = Field(
        default=None,
        description="Expected calendar life in years",
        ge=0,
    )
    age_years: Optional[Decimal] = Field(
        default=None,
        description="Current age of the battery in years",
        ge=0,
    )
    efficiency_pct: Optional[Decimal] = Field(
        default=None,
        description="Round-trip energy efficiency (%)",
        ge=0,
        le=Decimal("100"),
    )
    internal_resistance_mohm: Optional[Decimal] = Field(
        default=None,
        description="Internal resistance in milliohms",
        ge=0,
    )
    initial_resistance_mohm: Optional[Decimal] = Field(
        default=None,
        description="Initial internal resistance at beginning of life (mOhm)",
        ge=0,
    )
    soh_pct: Optional[Decimal] = Field(
        default=None,
        description="State of Health (%), if directly measured",
        ge=0,
        le=Decimal("100"),
    )
    soc_pct: Optional[Decimal] = Field(
        default=None,
        description="Current State of Charge (%)",
        ge=0,
        le=Decimal("100"),
    )
    c_rate_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum C-rate (charge/discharge)",
        ge=0,
    )
    temperature_min: Optional[Decimal] = Field(
        default=None,
        description="Minimum operating temperature (degC)",
    )
    temperature_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum operating temperature (degC)",
    )
    energy_capacity_kwh: Optional[Decimal] = Field(
        default=None,
        description="Rated energy capacity in kWh",
        ge=0,
    )
    weight_kg: Optional[Decimal] = Field(
        default=None,
        description="Battery weight in kg",
        ge=0,
    )

class MetricValidation(BaseModel):
    """Validation result for a single performance metric."""
    metric: PerformanceMetric = Field(
        ...,
        description="Performance metric identifier",
    )
    metric_label: str = Field(
        default="",
        description="Human-readable metric label",
    )
    unit: str = Field(
        default="",
        description="Measurement unit",
    )
    value: Optional[str] = Field(
        default=None,
        description="Measured/reported value",
    )
    status: MetricStatus = Field(
        default=MetricStatus.NOT_ASSESSED,
        description="Validation status",
    )
    note: str = Field(
        default="",
        description="Validation note",
        max_length=1000,
    )
    annex_ref: str = Field(
        default="",
        description="Annex IV reference",
    )

class SoHAssessment(BaseModel):
    """State of Health assessment result.

    Contains the calculated or reported SoH value with durability
    rating and lifecycle stage assessment.
    """
    soh_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="State of Health percentage",
    )
    calculation_method: str = Field(
        default="",
        description="Method used to determine SoH",
    )
    durability_rating: DurabilityRating = Field(
        default=DurabilityRating.CRITICAL,
        description="Durability rating based on SoH",
    )
    rating_description: str = Field(
        default="",
        description="Description of the durability rating",
    )
    life_stage: BatteryLifeStage = Field(
        default=BatteryLifeStage.NEW,
        description="Current stage in the battery's service life",
    )
    second_life_eligible: bool = Field(
        default=False,
        description="Whether the battery is eligible for second-life use",
    )
    remaining_useful_life_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Estimated remaining useful life as percentage",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class CycleLifeAssessment(BaseModel):
    """Cycle life assessment result.

    Evaluates the battery's cycle life performance against the
    expected specification.
    """
    cycles_expected: Optional[int] = Field(
        default=None,
        description="Expected total cycle life",
    )
    cycles_completed: Optional[int] = Field(
        default=None,
        description="Cycles completed to date",
    )
    cycles_remaining: Optional[int] = Field(
        default=None,
        description="Estimated remaining cycles",
    )
    completion_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of expected cycle life completed",
    )
    status: MetricStatus = Field(
        default=MetricStatus.NOT_ASSESSED,
        description="Assessment status",
    )
    note: str = Field(
        default="",
        description="Assessment note",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class EfficiencyAssessment(BaseModel):
    """Energy efficiency assessment result.

    Evaluates the battery's round-trip energy efficiency against
    minimum thresholds for its category.
    """
    efficiency_pct: Optional[Decimal] = Field(
        default=None,
        description="Measured round-trip energy efficiency (%)",
    )
    min_threshold_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Minimum efficiency threshold for the category",
    )
    above_threshold: bool = Field(
        default=False,
        description="Whether efficiency meets the minimum threshold",
    )
    margin_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Margin above/below threshold (percentage points)",
    )
    status: MetricStatus = Field(
        default=MetricStatus.NOT_ASSESSED,
        description="Assessment status",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class PerformanceResult(BaseModel):
    """Result of battery performance and durability assessment.

    Contains the complete assessment of all Annex IV performance
    parameters with durability rating, SoH assessment, cycle life
    evaluation, and compliance status.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of assessment (UTC)",
    )
    battery_id: str = Field(
        ...,
        description="Battery identifier",
    )
    category: str = Field(
        default="",
        description="Battery category",
    )
    chemistry: str = Field(
        default="",
        description="Battery chemistry",
    )
    metrics_validated: List[MetricValidation] = Field(
        default_factory=list,
        description="Per-metric validation results",
    )
    metrics_pass_count: int = Field(
        default=0,
        description="Number of metrics that passed validation",
    )
    metrics_warning_count: int = Field(
        default=0,
        description="Number of metrics with warnings",
    )
    metrics_fail_count: int = Field(
        default=0,
        description="Number of metrics that failed validation",
    )
    metrics_not_assessed: int = Field(
        default=0,
        description="Number of metrics not assessed (missing data)",
    )
    durability_rating: DurabilityRating = Field(
        default=DurabilityRating.CRITICAL,
        description="Overall durability rating",
    )
    durability_description: str = Field(
        default="",
        description="Description of the durability rating",
    )
    soh_assessment: Optional[SoHAssessment] = Field(
        default=None,
        description="State of Health assessment",
    )
    cycle_life_assessment: Optional[CycleLifeAssessment] = Field(
        default=None,
        description="Cycle life assessment",
    )
    efficiency_assessment: Optional[EfficiencyAssessment] = Field(
        default=None,
        description="Energy efficiency assessment",
    )
    compliance_status: str = Field(
        default="not_assessed",
        description="Overall Annex IV compliance status",
    )
    resistance_increase_pct: Optional[Decimal] = Field(
        default=None,
        description="Internal resistance increase from initial (%)",
    )
    specific_power_w_per_kg: Optional[Decimal] = Field(
        default=None,
        description="Specific power (W/kg)",
    )
    specific_energy_wh_per_kg: Optional[Decimal] = Field(
        default=None,
        description="Specific energy (Wh/kg)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for performance improvement",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PerformanceDurabilityEngine:
    """Battery performance and durability engine per Art 10 and Annex IV.

    Provides deterministic, zero-hallucination assessment of:
    - State of Health (SoH) calculation and rating
    - Cycle life tracking and assessment
    - Energy efficiency evaluation
    - Per-metric validation against Annex IV requirements
    - Internal resistance degradation tracking
    - Specific power and energy calculations
    - Durability rating (Excellent to Critical)
    - Second-life eligibility assessment

    All calculations use Decimal arithmetic and are bit-perfect
    reproducible.  No LLM is used in any calculation path.

    Usage::

        engine = PerformanceDurabilityEngine()
        inp = PerformanceInput(
            battery_id="BAT-EV-2025-001",
            category="ev",
            chemistry="nmc811",
            rated_capacity_ah=Decimal("100"),
            current_capacity_ah=Decimal("92"),
            cycle_life_expected=1500,
            cycles_completed=300,
            efficiency_pct=Decimal("93.5"),
            soh_pct=Decimal("92"),
        )
        result = engine.assess_performance(inp)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise PerformanceDurabilityEngine."""
        self._results: List[PerformanceResult] = []
        logger.info(
            "PerformanceDurabilityEngine v%s initialised",
            self.engine_version,
        )

    # ------------------------------------------------------------------ #
    # Main Assessment                                                      #
    # ------------------------------------------------------------------ #

    def assess_performance(
        self, input_data: PerformanceInput
    ) -> PerformanceResult:
        """Perform a complete performance and durability assessment.

        Evaluates all provided performance metrics against Annex IV
        requirements, calculates State of Health if not provided,
        assigns a durability rating, and assesses cycle life and
        efficiency compliance.

        Args:
            input_data: Validated PerformanceInput with measured metrics.

        Returns:
            PerformanceResult with complete assessment.

        Raises:
            ValueError: If critical input validation fails.
        """
        t0 = time.perf_counter()

        # Step 1: Validate input
        validation_errors = self._validate_input_data(input_data)
        if validation_errors:
            raise ValueError(
                f"Input validation failed: {'; '.join(validation_errors)}"
            )

        # Step 2: Calculate or use provided SoH
        soh_assessment = self._build_soh_assessment(input_data)

        # Step 3: Assess cycle life
        cycle_life_assessment = self._build_cycle_life_assessment(input_data)

        # Step 4: Assess efficiency
        efficiency_assessment = self._build_efficiency_assessment(input_data)

        # Step 5: Validate individual metrics
        metric_validations = self.validate_metrics(input_data)

        # Step 6: Determine durability rating from SoH
        soh_value = soh_assessment.soh_pct if soh_assessment else Decimal("0")
        durability_rating = self.assess_durability_from_soh(soh_value)
        durability_desc = RATING_DESCRIPTIONS.get(
            durability_rating.value, ""
        )

        # Step 7: Calculate additional derived metrics
        resistance_increase = self._calculate_resistance_increase(input_data)
        specific_power = self._calculate_specific_power(input_data)
        specific_energy = self._calculate_specific_energy(input_data)

        # Step 8: Count metric statuses
        pass_count = sum(
            1 for mv in metric_validations
            if mv.status == MetricStatus.PASS
        )
        warning_count = sum(
            1 for mv in metric_validations
            if mv.status == MetricStatus.WARNING
        )
        fail_count = sum(
            1 for mv in metric_validations
            if mv.status == MetricStatus.FAIL
        )
        not_assessed_count = sum(
            1 for mv in metric_validations
            if mv.status == MetricStatus.NOT_ASSESSED
        )

        # Step 9: Overall compliance
        compliance_status = self._determine_compliance(
            fail_count, warning_count, pass_count, not_assessed_count
        )

        # Step 10: Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, soh_assessment, cycle_life_assessment,
            efficiency_assessment, metric_validations, durability_rating,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PerformanceResult(
            battery_id=input_data.battery_id,
            category=input_data.category,
            chemistry=input_data.chemistry,
            metrics_validated=metric_validations,
            metrics_pass_count=pass_count,
            metrics_warning_count=warning_count,
            metrics_fail_count=fail_count,
            metrics_not_assessed=not_assessed_count,
            durability_rating=durability_rating,
            durability_description=durability_desc,
            soh_assessment=soh_assessment,
            cycle_life_assessment=cycle_life_assessment,
            efficiency_assessment=efficiency_assessment,
            compliance_status=compliance_status,
            resistance_increase_pct=resistance_increase,
            specific_power_w_per_kg=specific_power,
            specific_energy_wh_per_kg=specific_energy,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        self._results.append(result)

        logger.info(
            "Assessed performance for %s: SoH=%s%%, rating=%s, "
            "pass=%d, warn=%d, fail=%d, compliance=%s in %.3f ms",
            input_data.battery_id,
            soh_value if soh_assessment else "N/A",
            durability_rating.value,
            pass_count, warning_count, fail_count,
            compliance_status,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # State of Health                                                      #
    # ------------------------------------------------------------------ #

    def calculate_soh(
        self,
        capacity_initial: Decimal,
        capacity_current: Decimal,
    ) -> Decimal:
        """Calculate State of Health from capacity measurements.

        Formula (deterministic):
            SoH = (capacity_current / capacity_initial) * 100

        Args:
            capacity_initial: Rated (initial) capacity in Ah.
            capacity_current: Current measured capacity in Ah.

        Returns:
            State of Health as a percentage (0-100).
        """
        if capacity_initial <= 0:
            return Decimal("0.00")

        soh = (capacity_current / capacity_initial) * Decimal("100")

        # Cap at 100% (new batteries may slightly exceed rated)
        if soh > Decimal("100"):
            soh = Decimal("100.00")

        return _round_val(soh, 2)

    # ------------------------------------------------------------------ #
    # Durability Assessment                                                #
    # ------------------------------------------------------------------ #

    def assess_durability(
        self, metrics: Dict[str, Any]
    ) -> DurabilityRating:
        """Assess durability rating from a metrics dictionary.

        Uses SoH as the primary indicator. If SoH is not available,
        attempts to derive it from capacity data.

        Args:
            metrics: Dict with battery metrics (soh_pct, or
                rated_capacity_ah and current_capacity_ah).

        Returns:
            DurabilityRating enum value.
        """
        soh = metrics.get("soh_pct")

        if soh is None:
            rated = metrics.get("rated_capacity_ah")
            current = metrics.get("current_capacity_ah")
            if rated and current:
                soh = self.calculate_soh(
                    _decimal(rated), _decimal(current)
                )

        if soh is None:
            return DurabilityRating.CRITICAL

        return self.assess_durability_from_soh(_decimal(soh))

    def assess_durability_from_soh(
        self, soh_pct: Decimal
    ) -> DurabilityRating:
        """Assign a durability rating based on SoH percentage.

        Thresholds:
            - EXCELLENT: >= 95%
            - GOOD: >= 85%
            - ACCEPTABLE: >= 75%
            - POOR: >= 65%
            - CRITICAL: < 65%

        Args:
            soh_pct: State of Health percentage.

        Returns:
            DurabilityRating enum value.
        """
        val = _decimal(soh_pct)

        if val >= SOH_THRESHOLDS[DurabilityRating.EXCELLENT.value]:
            return DurabilityRating.EXCELLENT
        if val >= SOH_THRESHOLDS[DurabilityRating.GOOD.value]:
            return DurabilityRating.GOOD
        if val >= SOH_THRESHOLDS[DurabilityRating.ACCEPTABLE.value]:
            return DurabilityRating.ACCEPTABLE
        if val >= SOH_THRESHOLDS[DurabilityRating.POOR.value]:
            return DurabilityRating.POOR
        return DurabilityRating.CRITICAL

    # ------------------------------------------------------------------ #
    # Metric Validation                                                    #
    # ------------------------------------------------------------------ #

    def validate_metrics(
        self, input_data: PerformanceInput
    ) -> List[MetricValidation]:
        """Validate all provided performance metrics.

        Checks each metric value against acceptable ranges and
        Annex IV requirements. Assigns PASS, WARNING, FAIL, or
        NOT_ASSESSED status to each metric.

        Args:
            input_data: PerformanceInput with metric values.

        Returns:
            List of MetricValidation results.
        """
        validations: List[MetricValidation] = []

        # Rated capacity
        validations.append(self._validate_capacity(input_data))

        # Remaining capacity
        validations.append(self._validate_remaining_capacity(input_data))

        # Voltage nominal
        validations.append(self._validate_voltage_nominal(input_data))

        # Voltage range
        validations.append(self._validate_voltage_range(input_data))

        # Power capability
        validations.append(self._validate_power(input_data))

        # Cycle life
        validations.append(self._validate_cycle_life_metric(input_data))

        # Calendar life
        validations.append(self._validate_calendar_life(input_data))

        # Efficiency
        validations.append(self._validate_efficiency_metric(input_data))

        # Internal resistance
        validations.append(self._validate_resistance(input_data))

        # State of Health
        validations.append(self._validate_soh_metric(input_data))

        # State of Charge
        validations.append(self._validate_soc(input_data))

        # C-Rate
        validations.append(self._validate_c_rate(input_data))

        # Temperature range
        validations.append(self._validate_temperature(input_data))

        return validations

    # ------------------------------------------------------------------ #
    # Cycle Life Assessment                                                #
    # ------------------------------------------------------------------ #

    def assess_cycle_life(
        self, cycles_completed: int, cycles_expected: int
    ) -> CycleLifeAssessment:
        """Assess cycle life completion and remaining cycles.

        Args:
            cycles_completed: Number of cycles completed to date.
            cycles_expected: Expected total cycle life.

        Returns:
            CycleLifeAssessment with completion and remaining data.
        """
        if cycles_expected <= 0:
            assessment = CycleLifeAssessment(
                cycles_expected=cycles_expected,
                cycles_completed=cycles_completed,
                completion_pct=Decimal("0.00"),
                status=MetricStatus.NOT_ASSESSED,
                note="Expected cycle life not specified",
            )
            assessment.provenance_hash = _compute_hash(assessment)
            return assessment

        completion_pct = _round_val(
            _decimal(cycles_completed) / _decimal(cycles_expected) * Decimal("100"),
            2,
        )

        remaining = max(0, cycles_expected - cycles_completed)

        # Status determination
        if completion_pct < Decimal("50"):
            status = MetricStatus.PASS
            note = "Battery has significant cycle life remaining"
        elif completion_pct < Decimal("80"):
            status = MetricStatus.PASS
            note = "Battery is in mid-life phase"
        elif completion_pct < Decimal("100"):
            status = MetricStatus.WARNING
            note = "Battery is approaching end of expected cycle life"
        else:
            status = MetricStatus.WARNING
            note = "Battery has exceeded expected cycle life"

        assessment = CycleLifeAssessment(
            cycles_expected=cycles_expected,
            cycles_completed=cycles_completed,
            cycles_remaining=remaining,
            completion_pct=completion_pct,
            status=status,
            note=note,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    # ------------------------------------------------------------------ #
    # Batch Processing                                                     #
    # ------------------------------------------------------------------ #

    def assess_batch(
        self, inputs: List[PerformanceInput]
    ) -> List[PerformanceResult]:
        """Assess performance for a batch of batteries.

        Args:
            inputs: List of PerformanceInput objects.

        Returns:
            List of PerformanceResult objects.
        """
        t0 = time.perf_counter()
        results: List[PerformanceResult] = []

        for inp in inputs:
            try:
                result = self.assess_performance(inp)
                results.append(result)
            except ValueError as e:
                logger.warning(
                    "Skipping battery %s due to validation error: %s",
                    inp.battery_id, str(e),
                )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Batch performance assessment: %d/%d assessed in %.3f ms",
            len(results), len(inputs), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Comparison Utilities                                                 #
    # ------------------------------------------------------------------ #

    def compare_results(
        self, results: List[PerformanceResult]
    ) -> Dict[str, Any]:
        """Compare performance results across multiple batteries.

        Args:
            results: List of PerformanceResult objects.

        Returns:
            Dict with comparative analysis.
        """
        t0 = time.perf_counter()

        if not results:
            return {
                "count": 0,
                "comparison": [],
                "provenance_hash": _compute_hash({}),
            }

        entries = []
        for r in results:
            soh_val = "N/A"
            if r.soh_assessment:
                soh_val = str(r.soh_assessment.soh_pct)

            entries.append({
                "battery_id": r.battery_id,
                "category": r.category,
                "chemistry": r.chemistry,
                "durability_rating": r.durability_rating.value,
                "soh_pct": soh_val,
                "compliance_status": r.compliance_status,
                "metrics_pass": r.metrics_pass_count,
                "metrics_fail": r.metrics_fail_count,
            })

        # Rating distribution
        rating_dist: Dict[str, int] = {}
        for r in results:
            rk = r.durability_rating.value
            rating_dist[rk] = rating_dist.get(rk, 0) + 1

        # Compliance distribution
        compliance_dist: Dict[str, int] = {}
        for r in results:
            ck = r.compliance_status
            compliance_dist[ck] = compliance_dist.get(ck, 0) + 1

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        comparison = {
            "count": len(results),
            "entries": entries,
            "rating_distribution": rating_dist,
            "compliance_distribution": compliance_dist,
            "processing_time_ms": elapsed_ms,
        }
        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Documentation Builder                                                #
    # ------------------------------------------------------------------ #

    def build_documentation(
        self, result: PerformanceResult
    ) -> Dict[str, Any]:
        """Build performance documentation for the battery passport.

        Produces a structured document suitable for inclusion in
        Annex XIII Section E of the battery passport.

        Args:
            result: PerformanceResult to document.

        Returns:
            Dict with complete performance documentation.
        """
        t0 = time.perf_counter()

        doc: Dict[str, Any] = {
            "document_id": _new_uuid(),
            "regulation_reference": "Regulation (EU) 2023/1542, Art 10 and Annex IV",
            "document_type": "Performance and Durability Documentation",
            "battery_id": result.battery_id,
            "category": result.category,
            "chemistry": result.chemistry,
            "durability_rating": result.durability_rating.value,
            "durability_description": result.durability_description,
            "compliance_status": result.compliance_status,
            "metrics_summary": {
                "pass": result.metrics_pass_count,
                "warning": result.metrics_warning_count,
                "fail": result.metrics_fail_count,
                "not_assessed": result.metrics_not_assessed,
            },
            "metrics": [
                {
                    "metric": mv.metric.value,
                    "label": mv.metric_label,
                    "unit": mv.unit,
                    "value": mv.value,
                    "status": mv.status.value,
                    "note": mv.note,
                    "annex_ref": mv.annex_ref,
                }
                for mv in result.metrics_validated
            ],
            "soh_assessment": None,
            "cycle_life_assessment": None,
            "efficiency_assessment": None,
            "recommendations": result.recommendations,
            "assessed_at": str(result.assessed_at),
            "engine_version": result.engine_version,
        }

        if result.soh_assessment:
            doc["soh_assessment"] = {
                "soh_pct": str(result.soh_assessment.soh_pct),
                "calculation_method": result.soh_assessment.calculation_method,
                "durability_rating": result.soh_assessment.durability_rating.value,
                "life_stage": result.soh_assessment.life_stage.value,
                "second_life_eligible": result.soh_assessment.second_life_eligible,
                "remaining_useful_life_pct": str(
                    result.soh_assessment.remaining_useful_life_pct
                ),
            }

        if result.cycle_life_assessment:
            doc["cycle_life_assessment"] = {
                "cycles_expected": result.cycle_life_assessment.cycles_expected,
                "cycles_completed": result.cycle_life_assessment.cycles_completed,
                "cycles_remaining": result.cycle_life_assessment.cycles_remaining,
                "completion_pct": str(
                    result.cycle_life_assessment.completion_pct
                ),
                "status": result.cycle_life_assessment.status.value,
            }

        if result.efficiency_assessment:
            doc["efficiency_assessment"] = {
                "efficiency_pct": str(result.efficiency_assessment.efficiency_pct)
                if result.efficiency_assessment.efficiency_pct else None,
                "min_threshold_pct": str(
                    result.efficiency_assessment.min_threshold_pct
                ),
                "above_threshold": result.efficiency_assessment.above_threshold,
                "margin_pct": str(result.efficiency_assessment.margin_pct),
                "status": result.efficiency_assessment.status.value,
            }

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        doc["processing_time_ms"] = elapsed_ms
        doc["provenance_hash"] = _compute_hash(doc)

        logger.info(
            "Built performance documentation for %s in %.3f ms",
            result.battery_id, elapsed_ms,
        )
        return doc

    # ------------------------------------------------------------------ #
    # Reference Data                                                       #
    # ------------------------------------------------------------------ #

    def get_soh_thresholds(self) -> Dict[str, str]:
        """Return SoH threshold values for each durability rating.

        Returns:
            Dict mapping rating to threshold percentage.
        """
        return {k: str(v) for k, v in SOH_THRESHOLDS.items()}

    def get_metric_reference(self) -> Dict[str, Dict[str, str]]:
        """Return reference data for all performance metrics.

        Returns:
            Dict mapping metric to label, unit, and Annex reference.
        """
        return dict(METRIC_LABELS)

    def get_chemistry_cycle_life(
        self, chemistry: str
    ) -> Dict[str, Any]:
        """Return expected cycle life range for a battery chemistry.

        Args:
            chemistry: Battery chemistry identifier.

        Returns:
            Dict with low/typical/high cycle life values.
        """
        data = CHEMISTRY_CYCLE_LIFE.get(chemistry.lower())
        if data is None:
            return {
                "chemistry": chemistry,
                "available": False,
                "note": f"No cycle life data available for {chemistry}",
            }
        return {
            "chemistry": chemistry,
            "available": True,
            "low": data["low"],
            "typical": data["typical"],
            "high": data["high"],
        }

    # ------------------------------------------------------------------ #
    # Registry Management                                                  #
    # ------------------------------------------------------------------ #

    def get_results(self) -> List[PerformanceResult]:
        """Return all assessment results.

        Returns:
            List of PerformanceResult objects.
        """
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        logger.info("PerformanceDurabilityEngine results cleared")

    # ------------------------------------------------------------------ #
    # Private: SoH Assessment Builder                                      #
    # ------------------------------------------------------------------ #

    def _build_soh_assessment(
        self, input_data: PerformanceInput
    ) -> Optional[SoHAssessment]:
        """Build a complete SoH assessment.

        Uses directly reported SoH if available; otherwise calculates
        from rated and current capacity.

from greenlang.schemas import utcnow

        Args:
            input_data: PerformanceInput with capacity/SoH data.

        Returns:
            SoHAssessment or None if insufficient data.
        """
        soh_pct: Optional[Decimal] = None
        calc_method = ""

        if input_data.soh_pct is not None:
            soh_pct = input_data.soh_pct
            calc_method = "directly_reported"
        elif (
            input_data.rated_capacity_ah is not None
            and input_data.current_capacity_ah is not None
            and input_data.rated_capacity_ah > 0
        ):
            soh_pct = self.calculate_soh(
                input_data.rated_capacity_ah,
                input_data.current_capacity_ah,
            )
            calc_method = "calculated_from_capacity"

        if soh_pct is None:
            return None

        soh_val = _decimal(soh_pct)
        rating = self.assess_durability_from_soh(soh_val)
        rating_desc = RATING_DESCRIPTIONS.get(rating.value, "")

        # Determine life stage
        life_stage = self._determine_life_stage(soh_val, input_data)

        # Second-life eligibility
        second_life = (
            soh_val < SECOND_LIFE_SOH_THRESHOLD
            and soh_val >= END_OF_FIRST_LIFE_SOH - Decimal("10")
        )

        # Remaining useful life estimate
        # Assumes end-of-first-life at 70% SoH
        remaining_pct = Decimal("0.00")
        if soh_val > END_OF_FIRST_LIFE_SOH:
            usable_range = Decimal("100") - END_OF_FIRST_LIFE_SOH
            remaining = soh_val - END_OF_FIRST_LIFE_SOH
            remaining_pct = _round_val(
                (remaining / usable_range) * Decimal("100"), 2
            )

        assessment = SoHAssessment(
            soh_pct=_round_val(soh_val, 2),
            calculation_method=calc_method,
            durability_rating=rating,
            rating_description=rating_desc,
            life_stage=life_stage,
            second_life_eligible=second_life,
            remaining_useful_life_pct=remaining_pct,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    # ------------------------------------------------------------------ #
    # Private: Cycle Life Assessment Builder                               #
    # ------------------------------------------------------------------ #

    def _build_cycle_life_assessment(
        self, input_data: PerformanceInput
    ) -> Optional[CycleLifeAssessment]:
        """Build cycle life assessment from input data.

        Args:
            input_data: PerformanceInput with cycle data.

        Returns:
            CycleLifeAssessment or None if insufficient data.
        """
        if (
            input_data.cycle_life_expected is None
            or input_data.cycles_completed is None
        ):
            return None

        return self.assess_cycle_life(
            input_data.cycles_completed,
            input_data.cycle_life_expected,
        )

    # ------------------------------------------------------------------ #
    # Private: Efficiency Assessment Builder                               #
    # ------------------------------------------------------------------ #

    def _build_efficiency_assessment(
        self, input_data: PerformanceInput
    ) -> Optional[EfficiencyAssessment]:
        """Build efficiency assessment from input data.

        Args:
            input_data: PerformanceInput with efficiency data.

        Returns:
            EfficiencyAssessment or None if efficiency not provided.
        """
        if input_data.efficiency_pct is None:
            return None

        eff = input_data.efficiency_pct
        threshold = MIN_EFFICIENCY_THRESHOLDS.get(
            input_data.category.lower(), Decimal("80")
        )

        above = eff >= threshold
        margin = _round_val(eff - threshold, 2)

        if above:
            status = MetricStatus.PASS
        elif margin >= Decimal("-5"):
            status = MetricStatus.WARNING
        else:
            status = MetricStatus.FAIL

        assessment = EfficiencyAssessment(
            efficiency_pct=_round_val(eff, 2),
            min_threshold_pct=threshold,
            above_threshold=above,
            margin_pct=margin,
            status=status,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    # ------------------------------------------------------------------ #
    # Private: Individual Metric Validators                                #
    # ------------------------------------------------------------------ #

    def _validate_capacity(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate rated capacity metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.RATED_CAPACITY.value, {})
        if input_data.rated_capacity_ah is None:
            return MetricValidation(
                metric=PerformanceMetric.RATED_CAPACITY,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Rated capacity not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.RATED_CAPACITY,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.rated_capacity_ah),
            status=MetricStatus.PASS,
            note="Rated capacity provided",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_remaining_capacity(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate remaining capacity metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.REMAINING_CAPACITY.value, {})
        if input_data.current_capacity_ah is None:
            return MetricValidation(
                metric=PerformanceMetric.REMAINING_CAPACITY,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Current capacity not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        status = MetricStatus.PASS
        note = "Current capacity provided"

        if (
            input_data.rated_capacity_ah is not None
            and input_data.rated_capacity_ah > 0
        ):
            retention = (
                input_data.current_capacity_ah
                / input_data.rated_capacity_ah
                * Decimal("100")
            )
            if retention < Decimal("70"):
                status = MetricStatus.FAIL
                note = f"Capacity retention {_round_val(retention, 1)}% is below 70% threshold"
            elif retention < Decimal("80"):
                status = MetricStatus.WARNING
                note = f"Capacity retention {_round_val(retention, 1)}% is approaching end-of-first-life"
            else:
                note = f"Capacity retention {_round_val(retention, 1)}% is within acceptable range"

        return MetricValidation(
            metric=PerformanceMetric.REMAINING_CAPACITY,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.current_capacity_ah),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_voltage_nominal(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate nominal voltage metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.VOLTAGE_NOMINAL.value, {})
        if input_data.voltage_nominal is None:
            return MetricValidation(
                metric=PerformanceMetric.VOLTAGE_NOMINAL,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Nominal voltage not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.VOLTAGE_NOMINAL,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.voltage_nominal),
            status=MetricStatus.PASS,
            note="Nominal voltage provided",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_voltage_range(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate voltage range (min and max)."""
        meta = METRIC_LABELS.get(PerformanceMetric.VOLTAGE_MIN.value, {})
        if input_data.voltage_min is None and input_data.voltage_max is None:
            return MetricValidation(
                metric=PerformanceMetric.VOLTAGE_MIN,
                metric_label="Voltage Range",
                unit=meta.get("unit", "V"),
                status=MetricStatus.NOT_ASSESSED,
                note="Voltage range not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        status = MetricStatus.PASS
        note = "Voltage range provided"

        if (
            input_data.voltage_min is not None
            and input_data.voltage_max is not None
        ):
            if input_data.voltage_min >= input_data.voltage_max:
                status = MetricStatus.FAIL
                note = "Minimum voltage must be less than maximum voltage"
            else:
                value_str = f"{input_data.voltage_min} - {input_data.voltage_max}"
                return MetricValidation(
                    metric=PerformanceMetric.VOLTAGE_MIN,
                    metric_label="Voltage Range",
                    unit="V",
                    value=value_str,
                    status=status,
                    note=note,
                    annex_ref=meta.get("annex_ref", ""),
                )

        value_str = ""
        if input_data.voltage_min is not None:
            value_str = f"min={input_data.voltage_min}"
        if input_data.voltage_max is not None:
            value_str += f" max={input_data.voltage_max}"

        return MetricValidation(
            metric=PerformanceMetric.VOLTAGE_MIN,
            metric_label="Voltage Range",
            unit="V",
            value=value_str.strip(),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_power(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate power capability metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.POWER_CAPABILITY.value, {})
        if input_data.power_capability_w is None:
            return MetricValidation(
                metric=PerformanceMetric.POWER_CAPABILITY,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Power capability not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.POWER_CAPABILITY,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.power_capability_w),
            status=MetricStatus.PASS,
            note="Power capability provided",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_cycle_life_metric(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate cycle life metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.CYCLE_LIFE.value, {})
        if input_data.cycle_life_expected is None:
            return MetricValidation(
                metric=PerformanceMetric.CYCLE_LIFE,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Expected cycle life not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        status = MetricStatus.PASS
        note = f"Expected cycle life: {input_data.cycle_life_expected} cycles"

        # Plausibility check against chemistry benchmarks
        chem_data = CHEMISTRY_CYCLE_LIFE.get(input_data.chemistry.lower())
        if chem_data:
            if input_data.cycle_life_expected < chem_data["low"]:
                status = MetricStatus.WARNING
                note = (
                    f"Expected cycle life of {input_data.cycle_life_expected} "
                    f"is below typical range for {input_data.chemistry} "
                    f"({chem_data['low']}-{chem_data['high']})"
                )
            elif input_data.cycle_life_expected > chem_data["high"]:
                status = MetricStatus.WARNING
                note = (
                    f"Expected cycle life of {input_data.cycle_life_expected} "
                    f"exceeds typical range for {input_data.chemistry} "
                    f"({chem_data['low']}-{chem_data['high']})"
                )

        return MetricValidation(
            metric=PerformanceMetric.CYCLE_LIFE,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.cycle_life_expected),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_calendar_life(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate calendar life metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.CALENDAR_LIFE.value, {})
        if input_data.calendar_life_years is None:
            return MetricValidation(
                metric=PerformanceMetric.CALENDAR_LIFE,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Calendar life not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.CALENDAR_LIFE,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.calendar_life_years),
            status=MetricStatus.PASS,
            note=f"Expected calendar life: {input_data.calendar_life_years} years",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_efficiency_metric(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate energy efficiency metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.EFFICIENCY.value, {})
        if input_data.efficiency_pct is None:
            return MetricValidation(
                metric=PerformanceMetric.EFFICIENCY,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Energy efficiency not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        threshold = MIN_EFFICIENCY_THRESHOLDS.get(
            input_data.category.lower(), Decimal("80")
        )

        if input_data.efficiency_pct >= threshold:
            status = MetricStatus.PASS
            note = f"Efficiency {input_data.efficiency_pct}% meets threshold of {threshold}%"
        elif input_data.efficiency_pct >= threshold - Decimal("5"):
            status = MetricStatus.WARNING
            note = f"Efficiency {input_data.efficiency_pct}% is close to threshold of {threshold}%"
        else:
            status = MetricStatus.FAIL
            note = f"Efficiency {input_data.efficiency_pct}% is below threshold of {threshold}%"

        return MetricValidation(
            metric=PerformanceMetric.EFFICIENCY,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.efficiency_pct),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_resistance(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate internal resistance metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.INTERNAL_RESISTANCE.value, {})
        if input_data.internal_resistance_mohm is None:
            return MetricValidation(
                metric=PerformanceMetric.INTERNAL_RESISTANCE,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Internal resistance not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        status = MetricStatus.PASS
        note = f"Internal resistance: {input_data.internal_resistance_mohm} mOhm"

        # Check resistance increase if initial value is available
        if (
            input_data.initial_resistance_mohm is not None
            and input_data.initial_resistance_mohm > 0
        ):
            increase_pct = _round_val(
                (
                    (input_data.internal_resistance_mohm - input_data.initial_resistance_mohm)
                    / input_data.initial_resistance_mohm
                ) * Decimal("100"),
                2,
            )
            if increase_pct > Decimal("50"):
                status = MetricStatus.FAIL
                note = f"Resistance increased by {increase_pct}% from initial (>50% threshold)"
            elif increase_pct > Decimal("30"):
                status = MetricStatus.WARNING
                note = f"Resistance increased by {increase_pct}% from initial"
            else:
                note = f"Resistance increase of {increase_pct}% is within normal range"

        return MetricValidation(
            metric=PerformanceMetric.INTERNAL_RESISTANCE,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.internal_resistance_mohm),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_soh_metric(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate State of Health metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.SOH.value, {})

        soh = input_data.soh_pct
        if soh is None and (
            input_data.rated_capacity_ah is not None
            and input_data.current_capacity_ah is not None
            and input_data.rated_capacity_ah > 0
        ):
            soh = self.calculate_soh(
                input_data.rated_capacity_ah,
                input_data.current_capacity_ah,
            )

        if soh is None:
            return MetricValidation(
                metric=PerformanceMetric.SOH,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="State of Health not available (no SoH or capacity data)",
                annex_ref=meta.get("annex_ref", ""),
            )

        rating = self.assess_durability_from_soh(soh)

        if rating in (DurabilityRating.EXCELLENT, DurabilityRating.GOOD):
            status = MetricStatus.PASS
        elif rating == DurabilityRating.ACCEPTABLE:
            status = MetricStatus.WARNING
        else:
            status = MetricStatus.FAIL

        return MetricValidation(
            metric=PerformanceMetric.SOH,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(soh),
            status=status,
            note=f"SoH {soh}% rated as {rating.value}",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_soc(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate State of Charge metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.SOC.value, {})
        if input_data.soc_pct is None:
            return MetricValidation(
                metric=PerformanceMetric.SOC,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="State of Charge not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.SOC,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.soc_pct),
            status=MetricStatus.PASS,
            note=f"Current SoC: {input_data.soc_pct}%",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_c_rate(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate C-rate metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.C_RATE.value, {})
        if input_data.c_rate_max is None:
            return MetricValidation(
                metric=PerformanceMetric.C_RATE,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Maximum C-rate not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        return MetricValidation(
            metric=PerformanceMetric.C_RATE,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=str(input_data.c_rate_max),
            status=MetricStatus.PASS,
            note=f"Maximum C-rate: {input_data.c_rate_max}C",
            annex_ref=meta.get("annex_ref", ""),
        )

    def _validate_temperature(
        self, input_data: PerformanceInput
    ) -> MetricValidation:
        """Validate operating temperature range metric."""
        meta = METRIC_LABELS.get(PerformanceMetric.TEMPERATURE_RANGE.value, {})
        if input_data.temperature_min is None and input_data.temperature_max is None:
            return MetricValidation(
                metric=PerformanceMetric.TEMPERATURE_RANGE,
                metric_label=meta.get("label", ""),
                unit=meta.get("unit", ""),
                status=MetricStatus.NOT_ASSESSED,
                note="Operating temperature range not provided",
                annex_ref=meta.get("annex_ref", ""),
            )

        status = MetricStatus.PASS
        note = "Temperature range provided"

        if (
            input_data.temperature_min is not None
            and input_data.temperature_max is not None
        ):
            if input_data.temperature_min >= input_data.temperature_max:
                status = MetricStatus.FAIL
                note = "Minimum temperature must be less than maximum"
            else:
                note = (
                    f"Operating range: {input_data.temperature_min} to "
                    f"{input_data.temperature_max} degC"
                )

        value_str = ""
        if input_data.temperature_min is not None:
            value_str = f"{input_data.temperature_min}"
        if input_data.temperature_max is not None:
            value_str += f" to {input_data.temperature_max}"

        return MetricValidation(
            metric=PerformanceMetric.TEMPERATURE_RANGE,
            metric_label=meta.get("label", ""),
            unit=meta.get("unit", ""),
            value=value_str.strip(),
            status=status,
            note=note,
            annex_ref=meta.get("annex_ref", ""),
        )

    # ------------------------------------------------------------------ #
    # Private: Derived Metrics                                             #
    # ------------------------------------------------------------------ #

    def _calculate_resistance_increase(
        self, input_data: PerformanceInput
    ) -> Optional[Decimal]:
        """Calculate internal resistance increase from initial value.

        Args:
            input_data: PerformanceInput with resistance data.

        Returns:
            Percentage increase or None if data not available.
        """
        if (
            input_data.internal_resistance_mohm is None
            or input_data.initial_resistance_mohm is None
            or input_data.initial_resistance_mohm <= 0
        ):
            return None

        increase = (
            (input_data.internal_resistance_mohm - input_data.initial_resistance_mohm)
            / input_data.initial_resistance_mohm
        ) * Decimal("100")

        return _round_val(increase, 2)

    def _calculate_specific_power(
        self, input_data: PerformanceInput
    ) -> Optional[Decimal]:
        """Calculate specific power (W/kg).

        Args:
            input_data: PerformanceInput with power and weight data.

        Returns:
            Specific power in W/kg or None.
        """
        if (
            input_data.power_capability_w is None
            or input_data.weight_kg is None
            or input_data.weight_kg <= 0
        ):
            return None

        return _round_val(
            input_data.power_capability_w / input_data.weight_kg, 2
        )

    def _calculate_specific_energy(
        self, input_data: PerformanceInput
    ) -> Optional[Decimal]:
        """Calculate specific energy (Wh/kg).

        Args:
            input_data: PerformanceInput with energy and weight data.

        Returns:
            Specific energy in Wh/kg or None.
        """
        if (
            input_data.energy_capacity_kwh is None
            or input_data.weight_kg is None
            or input_data.weight_kg <= 0
        ):
            return None

        wh = input_data.energy_capacity_kwh * Decimal("1000")
        return _round_val(wh / input_data.weight_kg, 2)

    # ------------------------------------------------------------------ #
    # Private: Life Stage and Compliance                                   #
    # ------------------------------------------------------------------ #

    def _determine_life_stage(
        self, soh_pct: Decimal, input_data: PerformanceInput
    ) -> BatteryLifeStage:
        """Determine the battery's current life stage.

        Args:
            soh_pct: State of Health percentage.
            input_data: PerformanceInput for additional context.

        Returns:
            BatteryLifeStage enum value.
        """
        if soh_pct >= Decimal("98"):
            return BatteryLifeStage.NEW
        if soh_pct >= SECOND_LIFE_SOH_THRESHOLD:
            # Check if mid-life based on cycle completion
            if (
                input_data.cycles_completed is not None
                and input_data.cycle_life_expected is not None
                and input_data.cycle_life_expected > 0
            ):
                pct_complete = (
                    _decimal(input_data.cycles_completed)
                    / _decimal(input_data.cycle_life_expected)
                ) * Decimal("100")
                if pct_complete > Decimal("50"):
                    return BatteryLifeStage.MID_LIFE
            return BatteryLifeStage.IN_SERVICE
        if soh_pct >= END_OF_FIRST_LIFE_SOH:
            return BatteryLifeStage.END_OF_FIRST_LIFE
        return BatteryLifeStage.SECOND_LIFE

    def _determine_compliance(
        self,
        fail_count: int,
        warning_count: int,
        pass_count: int,
        not_assessed_count: int,
    ) -> str:
        """Determine overall Annex IV compliance status.

        Args:
            fail_count: Number of failed metrics.
            warning_count: Number of metrics with warnings.
            pass_count: Number of passed metrics.
            not_assessed_count: Number of unassessed metrics.

        Returns:
            Compliance status string.
        """
        if fail_count > 0:
            return "non_compliant"
        if warning_count > 0:
            return "compliant_with_warnings"
        if pass_count > 0:
            return "compliant"
        return "not_assessed"

    def _validate_input_data(
        self, input_data: PerformanceInput
    ) -> List[str]:
        """Validate input data for performance assessment.

        Args:
            input_data: PerformanceInput to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if not input_data.battery_id:
            errors.append("battery_id is required")

        if (
            input_data.current_capacity_ah is not None
            and input_data.rated_capacity_ah is not None
            and input_data.current_capacity_ah > input_data.rated_capacity_ah * Decimal("1.05")
        ):
            errors.append(
                "current_capacity_ah significantly exceeds rated_capacity_ah"
            )

        if (
            input_data.voltage_min is not None
            and input_data.voltage_max is not None
            and input_data.voltage_min > input_data.voltage_max
        ):
            errors.append("voltage_min cannot exceed voltage_max")

        if (
            input_data.temperature_min is not None
            and input_data.temperature_max is not None
            and input_data.temperature_min > input_data.temperature_max
        ):
            errors.append("temperature_min cannot exceed temperature_max")

        return errors

    def _generate_recommendations(
        self,
        input_data: PerformanceInput,
        soh: Optional[SoHAssessment],
        cycle_life: Optional[CycleLifeAssessment],
        efficiency: Optional[EfficiencyAssessment],
        metric_validations: List[MetricValidation],
        durability_rating: DurabilityRating,
    ) -> List[str]:
        """Generate recommendations based on performance assessment.

        Args:
            input_data: Input data.
            soh: SoH assessment.
            cycle_life: Cycle life assessment.
            efficiency: Efficiency assessment.
            metric_validations: Per-metric validation results.
            durability_rating: Overall durability rating.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # SoH-based recommendations
        if soh:
            if durability_rating == DurabilityRating.CRITICAL:
                recommendations.append(
                    f"Battery SoH is critical at {soh.soh_pct}%. "
                    f"Assess for replacement, second-life application, "
                    f"or recycling."
                )
            elif durability_rating == DurabilityRating.POOR:
                recommendations.append(
                    f"Battery SoH is poor at {soh.soh_pct}%. "
                    f"Plan for replacement within the next "
                    f"maintenance cycle."
                )
            elif soh.second_life_eligible:
                recommendations.append(
                    f"Battery is eligible for second-life application "
                    f"(SoH: {soh.soh_pct}%). Consider repurposing for "
                    f"stationary energy storage."
                )

        # Cycle life recommendations
        if cycle_life:
            if (
                cycle_life.completion_pct >= Decimal("80")
                and cycle_life.status == MetricStatus.WARNING
            ):
                recommendations.append(
                    f"Battery has completed {cycle_life.completion_pct}% "
                    f"of expected cycle life "
                    f"({cycle_life.cycles_completed}/{cycle_life.cycles_expected}). "
                    f"Plan for replacement or second-life evaluation."
                )

        # Efficiency recommendations
        if efficiency:
            if efficiency.status == MetricStatus.FAIL:
                recommendations.append(
                    f"Energy efficiency of {efficiency.efficiency_pct}% is "
                    f"below the {efficiency.min_threshold_pct}% threshold. "
                    f"Investigate thermal management and cell balancing."
                )
            elif efficiency.status == MetricStatus.WARNING:
                recommendations.append(
                    f"Energy efficiency of {efficiency.efficiency_pct}% is "
                    f"close to the {efficiency.min_threshold_pct}% threshold. "
                    f"Monitor for further degradation."
                )

        # Resistance recommendations
        if (
            input_data.internal_resistance_mohm is not None
            and input_data.initial_resistance_mohm is not None
            and input_data.initial_resistance_mohm > 0
        ):
            increase = (
                (input_data.internal_resistance_mohm - input_data.initial_resistance_mohm)
                / input_data.initial_resistance_mohm
            ) * Decimal("100")
            if increase > Decimal("50"):
                recommendations.append(
                    f"Internal resistance has increased by "
                    f"{_round_val(increase, 1)}% from initial value. "
                    f"This indicates significant degradation."
                )

        # Missing data recommendations
        not_assessed = [
            mv for mv in metric_validations
            if mv.status == MetricStatus.NOT_ASSESSED
        ]
        if len(not_assessed) > 3:
            recommendations.append(
                f"{len(not_assessed)} performance metrics could not be "
                f"assessed due to missing data. Provide complete Annex IV "
                f"data for a comprehensive assessment."
            )

        # Failed metrics
        failed = [
            mv for mv in metric_validations
            if mv.status == MetricStatus.FAIL
        ]
        if failed:
            names = [mv.metric_label or mv.metric.value for mv in failed[:3]]
            recommendations.append(
                f"{len(failed)} metric(s) failed validation: "
                f"{', '.join(names)}. Address these failures for "
                f"Annex IV compliance."
            )

        return recommendations
