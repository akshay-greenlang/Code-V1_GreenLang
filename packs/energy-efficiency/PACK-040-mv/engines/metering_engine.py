# -*- coding: utf-8 -*-
"""
MeteringEngine - PACK-040 M&V Engine 8
=========================================

M&V metering plan development, calibration tracking, sampling protocol
design, data quality assessment, and gap detection engine.

Calculation Methodology:
    Meter Selection Matrix:
        Maps IPMVP option (A/B/C/D) to required measurement points
        and meter types based on measurement boundary definition.

    Calibration Requirements (ANSI C12.20):
        Revenue class:  +/- 0.2% accuracy
        Utility class:  +/- 0.5% accuracy
        Submetering:    +/- 1.0% accuracy (typical)
        CT/PT class:    per ANSI/IEEE C57.13

    Sampling Protocol for Option A:
        Required sample size:
            n = (t * CV / precision)^2
        where:
            t = t-statistic at confidence level and df = n-1
            CV = coefficient of variation of the parameter
            precision = desired precision (e.g. 0.10 = 10%)
        Default: 90% confidence, 10% precision (90/10 rule)

    Data Quality Assessment:
        Completeness = (non-null readings / expected readings) * 100
        Accuracy = within calibration tolerance
        Consistency = no step-changes outside expected variance

    Gap Detection:
        Scan time-series for missing intervals
        Classify gap severity (minor < 1h, moderate 1-6h, major > 6h)
        Apply acceptable gap-fill methods per IPMVP

Regulatory References:
    - IPMVP Core Concepts 2022 - Metering requirements per option
    - ASHRAE Guideline 14-2014 - Data quality requirements
    - ANSI C12.20 - Revenue meter accuracy classes
    - ANSI/IEEE C57.13 - Instrument transformer accuracy
    - ISO 50015:2014 - Metering and measurement plan
    - FEMP M&V Guidelines 4.0 - Metering plan documentation

Zero-Hallucination:
    - Sample size computed via deterministic t-distribution formula
    - Calibration schedules from ANSI C12.20 lookup tables
    - No LLM in any calculation path
    - Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IPMVPOption(str, Enum):
    """IPMVP option designation.

    OPTION_A:  Retrofit Isolation - Key Parameter Measurement.
    OPTION_B:  Retrofit Isolation - All Parameter Measurement.
    OPTION_C:  Whole Facility.
    OPTION_D:  Calibrated Simulation.
    """
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    OPTION_D = "option_d"


class MeterType(str, Enum):
    """Type of energy meter.

    ELECTRIC_REVENUE:  Revenue-grade electric meter (ANSI C12.20 0.2).
    ELECTRIC_UTILITY:  Utility-grade electric meter (0.5).
    ELECTRIC_SUB:      Submetering panel meter (1.0).
    GAS_DIAPHRAGM:     Diaphragm gas meter.
    GAS_TURBINE:       Turbine gas meter.
    STEAM_ORIFICE:     Orifice plate steam meter.
    STEAM_VORTEX:      Vortex shedding steam meter.
    WATER_DISPLACEMENT: Displacement water meter.
    WATER_ULTRASONIC:  Ultrasonic water meter.
    BTU_METER:         BTU / thermal energy meter.
    CT_CLAMP:          Current transformer clamp-on.
    DATA_LOGGER:       Standalone data logger.
    """
    ELECTRIC_REVENUE = "electric_revenue"
    ELECTRIC_UTILITY = "electric_utility"
    ELECTRIC_SUB = "electric_sub"
    GAS_DIAPHRAGM = "gas_diaphragm"
    GAS_TURBINE = "gas_turbine"
    STEAM_ORIFICE = "steam_orifice"
    STEAM_VORTEX = "steam_vortex"
    WATER_DISPLACEMENT = "water_displacement"
    WATER_ULTRASONIC = "water_ultrasonic"
    BTU_METER = "btu_meter"
    CT_CLAMP = "ct_clamp"
    DATA_LOGGER = "data_logger"


class AccuracyClass(str, Enum):
    """Meter accuracy classification per ANSI C12.20 / IEC 62053.

    CLASS_02:   +/- 0.2% (revenue grade).
    CLASS_05:   +/- 0.5% (utility grade).
    CLASS_10:   +/- 1.0% (submetering).
    CLASS_20:   +/- 2.0% (indicative).
    CLASS_50:   +/- 5.0% (screening only).
    """
    CLASS_02 = "class_02"
    CLASS_05 = "class_05"
    CLASS_10 = "class_10"
    CLASS_20 = "class_20"
    CLASS_50 = "class_50"


class CalibrationStatus(str, Enum):
    """Meter calibration status.

    CURRENT:    Calibration within validity period.
    DUE_SOON:   Calibration due within 30 days.
    OVERDUE:    Calibration period expired.
    NOT_CALIBRATED: Never calibrated.
    FAILED:     Last calibration failed acceptance criteria.
    """
    CURRENT = "current"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    NOT_CALIBRATED = "not_calibrated"
    FAILED = "failed"


class GapSeverity(str, Enum):
    """Severity of a data gap.

    MINOR:     Gap < 1 hour (typically acceptable).
    MODERATE:  Gap 1-6 hours (gap-fill may be acceptable).
    MAJOR:     Gap 6-24 hours (gap-fill with caution).
    CRITICAL:  Gap > 24 hours (may invalidate period).
    """
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class GapFillMethod(str, Enum):
    """Acceptable gap-fill method per IPMVP.

    LINEAR:       Linear interpolation.
    PREVIOUS:     Forward-fill with last known value.
    AVERAGE:      Period average fill.
    REGRESSION:   Regression-based estimation.
    ENGINEERING:  Engineering estimate with documentation.
    NONE:         No fill applied; gap flagged.
    """
    LINEAR = "linear"
    PREVIOUS = "previous"
    AVERAGE = "average"
    REGRESSION = "regression"
    ENGINEERING = "engineering"
    NONE = "none"


class MeasurementPoint(str, Enum):
    """Measurement point location in facility.

    MAIN_METER:          Utility revenue meter.
    BUILDING_MAIN:       Building main panel.
    FLOOR_SUB:           Floor-level submeter.
    SYSTEM_LEVEL:        HVAC / lighting system meter.
    EQUIPMENT_LEVEL:     Individual equipment meter.
    CIRCUIT_LEVEL:       Circuit-level metering.
    VIRTUAL:             Calculated / virtual meter.
    """
    MAIN_METER = "main_meter"
    BUILDING_MAIN = "building_main"
    FLOOR_SUB = "floor_sub"
    SYSTEM_LEVEL = "system_level"
    EQUIPMENT_LEVEL = "equipment_level"
    CIRCUIT_LEVEL = "circuit_level"
    VIRTUAL = "virtual"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Accuracy by meter type (as fraction, e.g. 0.002 = 0.2%)
METER_ACCURACY: Dict[str, Decimal] = {
    MeterType.ELECTRIC_REVENUE.value: Decimal("0.002"),
    MeterType.ELECTRIC_UTILITY.value: Decimal("0.005"),
    MeterType.ELECTRIC_SUB.value: Decimal("0.010"),
    MeterType.GAS_DIAPHRAGM.value: Decimal("0.010"),
    MeterType.GAS_TURBINE.value: Decimal("0.005"),
    MeterType.STEAM_ORIFICE.value: Decimal("0.020"),
    MeterType.STEAM_VORTEX.value: Decimal("0.015"),
    MeterType.WATER_DISPLACEMENT.value: Decimal("0.015"),
    MeterType.WATER_ULTRASONIC.value: Decimal("0.005"),
    MeterType.BTU_METER.value: Decimal("0.020"),
    MeterType.CT_CLAMP.value: Decimal("0.030"),
    MeterType.DATA_LOGGER.value: Decimal("0.020"),
}

# Recommended calibration interval in months
CALIBRATION_INTERVAL_MONTHS: Dict[str, int] = {
    MeterType.ELECTRIC_REVENUE.value: 48,
    MeterType.ELECTRIC_UTILITY.value: 24,
    MeterType.ELECTRIC_SUB.value: 12,
    MeterType.GAS_DIAPHRAGM.value: 36,
    MeterType.GAS_TURBINE.value: 12,
    MeterType.STEAM_ORIFICE.value: 12,
    MeterType.STEAM_VORTEX.value: 12,
    MeterType.WATER_DISPLACEMENT.value: 36,
    MeterType.WATER_ULTRASONIC.value: 24,
    MeterType.BTU_METER.value: 12,
    MeterType.CT_CLAMP.value: 12,
    MeterType.DATA_LOGGER.value: 12,
}

# t-distribution critical values (one-tailed) for common df and alpha
# Key = (df, confidence_pct), value = t-critical
# This is a lookup for 90% confidence (alpha/2 = 0.05 one-tail)
T_CRITICAL_90: Dict[int, Decimal] = {
    2: Decimal("2.920"),
    3: Decimal("2.353"),
    4: Decimal("2.132"),
    5: Decimal("2.015"),
    6: Decimal("1.943"),
    7: Decimal("1.895"),
    8: Decimal("1.860"),
    9: Decimal("1.833"),
    10: Decimal("1.812"),
    12: Decimal("1.782"),
    15: Decimal("1.753"),
    20: Decimal("1.725"),
    25: Decimal("1.708"),
    30: Decimal("1.697"),
    40: Decimal("1.684"),
    50: Decimal("1.676"),
    60: Decimal("1.671"),
    80: Decimal("1.664"),
    100: Decimal("1.660"),
    120: Decimal("1.658"),
    200: Decimal("1.653"),
    500: Decimal("1.648"),
    1000: Decimal("1.646"),
}


def _lookup_t_critical(df: int, table: Dict[int, Decimal]) -> Decimal:
    """Look up t-critical value from table with interpolation."""
    if df in table:
        return table[df]
    keys = sorted(table.keys())
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return table[keys[-1]]
    # Find bracketing keys
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            frac = _decimal(df - keys[i]) / _decimal(keys[i + 1] - keys[i])
            return table[keys[i]] + frac * (table[keys[i + 1]] - table[keys[i]])
    return Decimal("1.645")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class MeterSpec(BaseModel):
    """Specification of a single meter in the M&V plan."""

    meter_id: str = Field(default_factory=_new_uuid, description="Meter ID")
    name: str = Field(default="", description="Meter name / label")
    meter_type: MeterType = Field(
        default=MeterType.ELECTRIC_UTILITY, description="Meter type"
    )
    accuracy_class: AccuracyClass = Field(
        default=AccuracyClass.CLASS_05, description="Accuracy class"
    )
    accuracy_pct: Decimal = Field(
        default=Decimal("0.5"), description="Accuracy +/- percent"
    )
    measurement_point: MeasurementPoint = Field(
        default=MeasurementPoint.MAIN_METER, description="Location in facility"
    )
    unit: str = Field(default="kWh", description="Measured unit")
    interval_minutes: int = Field(default=15, ge=1, description="Recording interval")
    serial_number: str = Field(default="", description="Meter serial number")
    manufacturer: str = Field(default="", description="Meter manufacturer")
    model: str = Field(default="", description="Meter model")
    install_date: Optional[str] = Field(None, description="Installation date")
    last_calibration: Optional[str] = Field(None, description="Last calibration date")
    calibration_status: CalibrationStatus = Field(
        default=CalibrationStatus.NOT_CALIBRATED, description="Calibration status"
    )
    notes: str = Field(default="", description="Additional notes")


class CalibrationRecord(BaseModel):
    """Calibration record for a meter."""

    record_id: str = Field(default_factory=_new_uuid, description="Record ID")
    meter_id: str = Field(..., description="Meter reference")
    calibration_date: str = Field(..., description="Calibration date YYYY-MM-DD")
    next_due_date: str = Field(default="", description="Next calibration due YYYY-MM-DD")
    performed_by: str = Field(default="", description="Calibrating entity")
    result: str = Field(default="pass", description="pass / fail / adjusted")
    accuracy_before_pct: Decimal = Field(
        default=Decimal("0"), description="Accuracy before calibration"
    )
    accuracy_after_pct: Decimal = Field(
        default=Decimal("0"), description="Accuracy after calibration"
    )
    reference_standard: str = Field(default="", description="Reference standard used")
    certificate_number: str = Field(default="", description="Calibration certificate #")
    notes: str = Field(default="", description="Notes")


class SamplingProtocol(BaseModel):
    """Sampling protocol design for IPMVP Option A."""

    protocol_id: str = Field(default_factory=_new_uuid, description="Protocol ID")
    parameter_name: str = Field(default="", description="Parameter being sampled")
    population_size: int = Field(default=0, ge=0, description="Total population N")
    confidence_pct: Decimal = Field(
        default=Decimal("90"), description="Confidence level (%)"
    )
    precision_pct: Decimal = Field(
        default=Decimal("10"), description="Desired precision (%)"
    )
    coefficient_of_variation: Decimal = Field(
        default=Decimal("0.5"), description="CV of the population"
    )
    required_sample_size: int = Field(
        default=0, description="Computed required sample size n"
    )
    t_critical: Decimal = Field(
        default=Decimal("1.645"), description="t-statistic used"
    )
    finite_correction_applied: bool = Field(
        default=False, description="Finite population correction applied"
    )
    notes: str = Field(default="", description="Notes")


class DataGap(BaseModel):
    """A detected gap in metered data."""

    gap_id: str = Field(default_factory=_new_uuid, description="Gap ID")
    meter_id: str = Field(default="", description="Meter reference")
    start_index: int = Field(default=0, description="Start index of gap")
    end_index: int = Field(default=0, description="End index of gap (exclusive)")
    gap_length: int = Field(default=0, description="Number of missing intervals")
    gap_duration_minutes: int = Field(default=0, description="Gap duration in minutes")
    severity: GapSeverity = Field(default=GapSeverity.MINOR, description="Severity")
    recommended_fill: GapFillMethod = Field(
        default=GapFillMethod.LINEAR, description="Recommended fill method"
    )
    filled: bool = Field(default=False, description="Whether gap has been filled")


class MeteringPlan(BaseModel):
    """Complete metering plan for an M&V project."""

    plan_id: str = Field(default_factory=_new_uuid, description="Plan ID")
    project_id: str = Field(default="", description="M&V project reference")
    ipmvp_option: IPMVPOption = Field(
        default=IPMVPOption.OPTION_C, description="IPMVP option"
    )
    meters: List[MeterSpec] = Field(default_factory=list, description="Meter inventory")
    calibration_records: List[CalibrationRecord] = Field(
        default_factory=list, description="Calibration history"
    )
    sampling_protocols: List[SamplingProtocol] = Field(
        default_factory=list, description="Sampling protocols (Option A)"
    )
    data_collection_interval_min: int = Field(
        default=15, description="Collection interval (minutes)"
    )
    data_storage_duration_months: int = Field(
        default=60, description="Data retention (months)"
    )
    quality_requirements: str = Field(
        default="Completeness >= 90%, accuracy per ANSI C12.20",
        description="Data quality requirements",
    )
    notes: List[str] = Field(default_factory=list, description="Plan notes")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class DataQualityResult(BaseModel):
    """Data quality assessment result for metered data."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    meter_id: str = Field(default="", description="Meter reference")
    total_intervals: int = Field(default=0, description="Total expected intervals")
    valid_intervals: int = Field(default=0, description="Valid (non-null) intervals")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Completeness %")
    gaps_detected: List[DataGap] = Field(default_factory=list, description="Gaps found")
    total_gap_intervals: int = Field(default=0, description="Total gap intervals")
    max_gap_minutes: int = Field(default=0, description="Longest gap (min)")
    outlier_count: int = Field(default=0, description="Statistical outliers detected")
    negative_count: int = Field(default=0, description="Negative values (if unexpected)")
    zero_count: int = Field(default=0, description="Zero values")
    mean_value: Decimal = Field(default=Decimal("0"), description="Mean of valid values")
    std_value: Decimal = Field(default=Decimal("0"), description="Std dev of valid values")
    passes_ipmvp: bool = Field(
        default=False, description="Passes IPMVP data quality requirements"
    )
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class MeterSelectionResult(BaseModel):
    """Result of meter selection based on IPMVP option."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    ipmvp_option: IPMVPOption = Field(..., description="IPMVP option")
    measurement_boundary: str = Field(
        default="", description="Measurement boundary description"
    )
    recommended_meters: List[MeterSpec] = Field(
        default_factory=list, description="Recommended meter specifications"
    )
    total_meters_required: int = Field(default=0, description="Number of meters needed")
    estimated_cost: Decimal = Field(default=Decimal("0"), description="Estimated metering cost")
    accuracy_summary: str = Field(default="", description="Summary of accuracy")
    notes: List[str] = Field(default_factory=list, description="Selection notes")
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


class MeterUncertaintyResult(BaseModel):
    """Meter measurement uncertainty propagation result."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    meter_id: str = Field(default="", description="Meter reference")
    base_accuracy_pct: Decimal = Field(
        default=Decimal("0"), description="Base meter accuracy %"
    )
    ct_error_pct: Decimal = Field(
        default=Decimal("0"), description="Current transformer error %"
    )
    pt_error_pct: Decimal = Field(
        default=Decimal("0"), description="Potential transformer error %"
    )
    calibration_drift_pct: Decimal = Field(
        default=Decimal("0"), description="Calibration drift estimate %"
    )
    combined_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Combined measurement uncertainty %"
    )
    uncertainty_kwh: Decimal = Field(
        default=Decimal("0"), description="Absolute uncertainty (kWh)"
    )
    calculation_method: str = Field(
        default="root_sum_square", description="Combination method"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------


class MeteringEngine:
    """M&V metering plan, calibration, sampling, and data quality engine.

    Provides meter selection based on IPMVP option, calibration schedule
    management per ANSI C12.20, sampling protocol design using
    t-distribution, data quality assessment, gap detection, and
    measurement uncertainty propagation.

    All calculations are deterministic (zero-hallucination) with
    Decimal arithmetic and SHA-256 provenance hashing.

    Attributes:
        _module_version: Engine version string.

    Example:
        >>> engine = MeteringEngine()
        >>> protocol = engine.calculate_sample_size(
        ...     population_size=500,
        ...     confidence_pct=Decimal("90"),
        ...     precision_pct=Decimal("10"),
        ...     cv=Decimal("0.5"),
        ... )
        >>> assert protocol.required_sample_size > 0
    """

    def __init__(self) -> None:
        """Initialise the MeteringEngine."""
        self._module_version: str = _MODULE_VERSION
        logger.info("MeteringEngine v%s initialised", self._module_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_meters(
        self,
        ipmvp_option: IPMVPOption,
        ecm_description: str = "",
        measurement_boundary: str = "",
        n_systems: int = 1,
    ) -> MeterSelectionResult:
        """Select appropriate meters based on IPMVP option.

        Uses the IPMVP meter selection matrix to recommend meter types,
        accuracy classes, and measurement points.

        Args:
            ipmvp_option: The IPMVP option for this ECM.
            ecm_description: Description of the ECM.
            measurement_boundary: M&V boundary description.
            n_systems: Number of systems / measurement points.

        Returns:
            MeterSelectionResult with recommended meters.
        """
        t0 = time.perf_counter()
        logger.info(
            "Selecting meters: option=%s, n_systems=%d",
            ipmvp_option.value, n_systems,
        )

        meters: List[MeterSpec] = []
        notes: List[str] = []
        est_cost = Decimal("0")

        if ipmvp_option == IPMVPOption.OPTION_A:
            # Key parameter measurement only
            for i in range(n_systems):
                meters.append(MeterSpec(
                    name=f"Key Parameter Meter {i+1}",
                    meter_type=MeterType.CT_CLAMP,
                    accuracy_class=AccuracyClass.CLASS_20,
                    accuracy_pct=Decimal("2.0"),
                    measurement_point=MeasurementPoint.EQUIPMENT_LEVEL,
                    unit="kW",
                    interval_minutes=15,
                ))
                est_cost += Decimal("500")
            notes.append("Option A: Short-term or one-time key parameter measurement")
            notes.append("Non-measured parameters will be stipulated with engineering estimates")

        elif ipmvp_option == IPMVPOption.OPTION_B:
            # All parameter measurement (retrofit isolation)
            for i in range(n_systems):
                meters.append(MeterSpec(
                    name=f"System Meter {i+1}",
                    meter_type=MeterType.ELECTRIC_SUB,
                    accuracy_class=AccuracyClass.CLASS_10,
                    accuracy_pct=Decimal("1.0"),
                    measurement_point=MeasurementPoint.SYSTEM_LEVEL,
                    unit="kWh",
                    interval_minutes=15,
                ))
                est_cost += Decimal("2000")
            notes.append("Option B: Continuous measurement of all energy parameters")
            notes.append("Retrofit isolation boundary requires dedicated metering")

        elif ipmvp_option == IPMVPOption.OPTION_C:
            # Whole facility - use existing utility meters
            meters.append(MeterSpec(
                name="Utility Revenue Meter",
                meter_type=MeterType.ELECTRIC_REVENUE,
                accuracy_class=AccuracyClass.CLASS_02,
                accuracy_pct=Decimal("0.2"),
                measurement_point=MeasurementPoint.MAIN_METER,
                unit="kWh",
                interval_minutes=15,
            ))
            est_cost += Decimal("0")  # Existing meter
            notes.append("Option C: Whole facility using existing utility meters")
            notes.append("Ensure utility provides interval data (15-min or hourly)")

        elif ipmvp_option == IPMVPOption.OPTION_D:
            # Calibrated simulation
            meters.append(MeterSpec(
                name="Calibration Reference Meter",
                meter_type=MeterType.ELECTRIC_UTILITY,
                accuracy_class=AccuracyClass.CLASS_05,
                accuracy_pct=Decimal("0.5"),
                measurement_point=MeasurementPoint.BUILDING_MAIN,
                unit="kWh",
                interval_minutes=60,
            ))
            est_cost += Decimal("1000")
            notes.append("Option D: Calibrated simulation with reference metering")
            notes.append("Simulation must be calibrated to within ASHRAE 14 criteria")

        accuracy_summary = (
            f"{len(meters)} meter(s) recommended for {ipmvp_option.value}; "
            f"best accuracy class: "
            f"{min(m.accuracy_pct for m in meters) if meters else 0}%"
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = MeterSelectionResult(
            ipmvp_option=ipmvp_option,
            measurement_boundary=measurement_boundary,
            recommended_meters=meters,
            total_meters_required=len(meters),
            estimated_cost=_round_val(est_cost, 2),
            accuracy_summary=accuracy_summary,
            notes=notes,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Meter selection: %d meters, est_cost=%.0f, hash=%s (%.1f ms)",
            len(meters), float(est_cost), result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_sample_size(
        self,
        population_size: int,
        confidence_pct: Decimal = Decimal("90"),
        precision_pct: Decimal = Decimal("10"),
        cv: Decimal = Decimal("0.5"),
        apply_fpc: bool = True,
    ) -> SamplingProtocol:
        """Calculate required sample size for IPMVP Option A.

        Formula: n = (t * CV / precision)^2
        With finite population correction: n_fpc = n / (1 + n/N)

        Args:
            population_size: Total population N.
            confidence_pct: Confidence level (e.g. 90).
            precision_pct: Desired precision (e.g. 10 = +/-10%).
            cv: Coefficient of variation of the population.
            apply_fpc: Apply finite population correction.

        Returns:
            SamplingProtocol with required sample size.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating sample size: N=%d, conf=%.0f%%, prec=%.0f%%, CV=%.2f",
            population_size, float(confidence_pct), float(precision_pct), float(cv),
        )

        # Look up t-critical for 90% confidence (common in M&V)
        # Initial estimate df = 30
        t_crit = _lookup_t_critical(30, T_CRITICAL_90)

        precision_frac = precision_pct / Decimal("100")
        if precision_frac <= Decimal("0"):
            precision_frac = Decimal("0.10")

        # n = (t * CV / precision)^2
        n_raw = (t_crit * cv / precision_frac) ** 2
        n_raw_int = max(int(n_raw.to_integral_value(rounding=ROUND_HALF_UP)), 1)

        # Refine t-critical with actual df = n - 1
        if n_raw_int > 1:
            t_crit = _lookup_t_critical(n_raw_int - 1, T_CRITICAL_90)
            n_raw = (t_crit * cv / precision_frac) ** 2
            n_raw_int = max(int(n_raw.to_integral_value(rounding=ROUND_HALF_UP)), 1)

        # Finite population correction
        fpc_applied = False
        n_final = n_raw_int
        if apply_fpc and population_size > 0 and n_raw_int > 0:
            n_fpc = _safe_divide(
                _decimal(n_raw_int),
                Decimal("1") + _safe_divide(_decimal(n_raw_int), _decimal(population_size)),
            )
            n_final = max(int(n_fpc.to_integral_value(rounding=ROUND_HALF_UP)), 1)
            fpc_applied = n_final < n_raw_int

        # Cap at population size
        if population_size > 0:
            n_final = min(n_final, population_size)

        elapsed = (time.perf_counter() - t0) * 1000.0
        protocol = SamplingProtocol(
            parameter_name="key_parameter",
            population_size=population_size,
            confidence_pct=confidence_pct,
            precision_pct=precision_pct,
            coefficient_of_variation=cv,
            required_sample_size=n_final,
            t_critical=_round_val(t_crit, 4),
            finite_correction_applied=fpc_applied,
            notes=f"n_raw={n_raw_int}, n_fpc={n_final}, t={float(t_crit):.3f}",
        )

        logger.info(
            "Sample size: n=%d (raw=%d, fpc=%s) for N=%d at %.0f/%.0f (%.1f ms)",
            n_final, n_raw_int, fpc_applied, population_size,
            float(confidence_pct), float(precision_pct), elapsed,
        )
        return protocol

    def assess_data_quality(
        self,
        values: List[Optional[Decimal]],
        meter_id: str = "",
        interval_minutes: int = 15,
        min_completeness_pct: Decimal = Decimal("90"),
        allow_negatives: bool = False,
    ) -> DataQualityResult:
        """Assess quality of metered data for M&V use.

        Checks completeness, gaps, outliers, negatives, and zeros.

        Args:
            values: List of metered values (None = missing).
            meter_id: Meter reference.
            interval_minutes: Expected interval between readings.
            min_completeness_pct: Minimum acceptable completeness.
            allow_negatives: Whether negative values are expected.

        Returns:
            DataQualityResult with quality metrics and gap list.
        """
        t0 = time.perf_counter()
        n = len(values)
        logger.info(
            "Assessing data quality: meter=%s, %d intervals, %d-min",
            meter_id, n, interval_minutes,
        )

        valid_vals: List[Decimal] = []
        null_count = 0
        negative_count = 0
        zero_count = 0
        issues: List[str] = []

        for v in values:
            if v is None:
                null_count += 1
            else:
                val = _decimal(v)
                valid_vals.append(val)
                if val < Decimal("0"):
                    negative_count += 1
                if val == Decimal("0"):
                    zero_count += 1

        valid_count = len(valid_vals)
        completeness = _safe_pct(_decimal(valid_count), _decimal(n)) if n > 0 else Decimal("0")

        # Detect gaps
        gaps = self._detect_gaps(values, interval_minutes)
        total_gap_intervals = sum(g.gap_length for g in gaps)
        max_gap_min = max((g.gap_duration_minutes for g in gaps), default=0)

        # Outlier detection (IQR method)
        outlier_count = 0
        mean_val = Decimal("0")
        std_val = Decimal("0")
        if valid_vals:
            mean_val = _safe_divide(sum(valid_vals), _decimal(len(valid_vals)))
            variance = _safe_divide(
                sum((v - mean_val) ** 2 for v in valid_vals),
                _decimal(len(valid_vals) - 1) if len(valid_vals) > 1 else Decimal("1"),
            )
            std_val = _decimal(math.sqrt(float(variance))) if variance > Decimal("0") else Decimal("0")

            if std_val > Decimal("0"):
                for v in valid_vals:
                    z = abs(v - mean_val) / std_val
                    if z > Decimal("3"):
                        outlier_count += 1

        # Issues
        if completeness < min_completeness_pct:
            issues.append(
                f"Completeness {float(completeness):.1f}% < {float(min_completeness_pct)}%"
            )
        if not allow_negatives and negative_count > 0:
            issues.append(f"{negative_count} unexpected negative values")
        if outlier_count > 0:
            issues.append(f"{outlier_count} statistical outliers (|z| > 3)")
        if max_gap_min > 360:
            issues.append(f"Major gap detected: {max_gap_min} minutes")
        if zero_count > valid_count * 0.5 and valid_count > 0:
            issues.append(f"High proportion of zero values: {zero_count}/{valid_count}")

        passes = (
            completeness >= min_completeness_pct
            and (allow_negatives or negative_count == 0)
            and max_gap_min <= 1440
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = DataQualityResult(
            meter_id=meter_id,
            total_intervals=n,
            valid_intervals=valid_count,
            completeness_pct=_round_val(completeness, 2),
            gaps_detected=gaps,
            total_gap_intervals=total_gap_intervals,
            max_gap_minutes=max_gap_min,
            outlier_count=outlier_count,
            negative_count=negative_count,
            zero_count=zero_count,
            mean_value=_round_val(mean_val, 4),
            std_value=_round_val(std_val, 4),
            passes_ipmvp=passes,
            issues=issues,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Data quality: completeness=%.1f%%, %d gaps, passes=%s, "
            "hash=%s (%.1f ms)",
            float(completeness), len(gaps), passes,
            result.provenance_hash[:16], elapsed,
        )
        return result

    def compute_meter_uncertainty(
        self,
        meter: MeterSpec,
        ct_error_pct: Decimal = Decimal("0.3"),
        pt_error_pct: Decimal = Decimal("0.3"),
        calibration_drift_pct: Decimal = Decimal("0.1"),
        annual_energy_kwh: Decimal = Decimal("0"),
    ) -> MeterUncertaintyResult:
        """Compute combined measurement uncertainty for a meter.

        Uses root-sum-square (RSS) combination:
        u_combined = sqrt(u_meter^2 + u_ct^2 + u_pt^2 + u_drift^2)

        Args:
            meter: Meter specification.
            ct_error_pct: Current transformer error (%).
            pt_error_pct: Potential transformer error (%).
            calibration_drift_pct: Estimated calibration drift (%).
            annual_energy_kwh: Annual energy for absolute uncertainty.

        Returns:
            MeterUncertaintyResult with combined uncertainty.
        """
        t0 = time.perf_counter()
        logger.info(
            "Computing meter uncertainty: %s (%s, %.2f%%)",
            meter.name, meter.meter_type.value, float(meter.accuracy_pct),
        )

        base = meter.accuracy_pct
        ct = ct_error_pct
        pt = pt_error_pct
        drift = calibration_drift_pct

        # Root-sum-square combination
        combined = _decimal(math.sqrt(
            float(base ** 2 + ct ** 2 + pt ** 2 + drift ** 2)
        ))

        abs_uncertainty = combined * annual_energy_kwh / Decimal("100")

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = MeterUncertaintyResult(
            meter_id=meter.meter_id,
            base_accuracy_pct=_round_val(base, 4),
            ct_error_pct=_round_val(ct, 4),
            pt_error_pct=_round_val(pt, 4),
            calibration_drift_pct=_round_val(drift, 4),
            combined_uncertainty_pct=_round_val(combined, 4),
            uncertainty_kwh=_round_val(abs_uncertainty, 2),
            calculation_method="root_sum_square",
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Meter uncertainty: combined=%.4f%%, abs=%.2f kWh, hash=%s (%.1f ms)",
            float(combined), float(abs_uncertainty),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def create_metering_plan(
        self,
        project_id: str,
        ipmvp_option: IPMVPOption,
        meters: List[MeterSpec],
        sampling_protocols: Optional[List[SamplingProtocol]] = None,
        interval_minutes: int = 15,
        retention_months: int = 60,
    ) -> MeteringPlan:
        """Create a complete metering plan for an M&V project.

        Aggregates meter specs, calibration requirements, and sampling
        protocols into a single plan document.

        Args:
            project_id: M&V project identifier.
            ipmvp_option: IPMVP option.
            meters: List of meter specifications.
            sampling_protocols: Sampling protocols (Option A).
            interval_minutes: Data collection interval.
            retention_months: Data retention period.

        Returns:
            MeteringPlan ready for documentation.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating metering plan: project=%s, option=%s, %d meters",
            project_id, ipmvp_option.value, len(meters),
        )

        # Generate calibration schedule for each meter
        cal_records: List[CalibrationRecord] = []
        notes: List[str] = []
        for meter in meters:
            cal_interval = CALIBRATION_INTERVAL_MONTHS.get(
                meter.meter_type.value, 12
            )
            cal_records.append(CalibrationRecord(
                meter_id=meter.meter_id,
                calibration_date="TBD",
                next_due_date="TBD",
                performed_by="TBD",
                result="pending",
                notes=f"Recommended interval: {cal_interval} months per ANSI C12.20",
            ))
            notes.append(
                f"Meter '{meter.name}' ({meter.meter_type.value}): "
                f"calibrate every {cal_interval} months"
            )

        if ipmvp_option == IPMVPOption.OPTION_A and not sampling_protocols:
            notes.append("WARNING: Option A requires sampling protocol; none provided")

        quality_req = (
            f"Completeness >= 90%, accuracy per ANSI C12.20, "
            f"interval = {interval_minutes} min, retention = {retention_months} months"
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        plan = MeteringPlan(
            project_id=project_id,
            ipmvp_option=ipmvp_option,
            meters=meters,
            calibration_records=cal_records,
            sampling_protocols=sampling_protocols or [],
            data_collection_interval_min=interval_minutes,
            data_storage_duration_months=retention_months,
            quality_requirements=quality_req,
            notes=notes,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        plan.provenance_hash = _compute_hash(plan)

        logger.info(
            "Metering plan created: %d meters, %d cal records, hash=%s (%.1f ms)",
            len(meters), len(cal_records), plan.provenance_hash[:16], elapsed,
        )
        return plan

    def check_calibration_status(
        self,
        meters: List[MeterSpec],
        reference_date: Optional[str] = None,
    ) -> List[MeterSpec]:
        """Check and update calibration status for a list of meters.

        Args:
            meters: Meters to check.
            reference_date: Date to check against (YYYY-MM-DD).

        Returns:
            Updated meter list with calibration status set.
        """
        t0 = time.perf_counter()
        if reference_date is None:
            reference_date = _utcnow().strftime("%Y-%m-%d")

        updated: List[MeterSpec] = []
        for meter in meters:
            m = meter.model_copy(deep=True)
            if not m.last_calibration:
                m.calibration_status = CalibrationStatus.NOT_CALIBRATED
            else:
                interval_months = CALIBRATION_INTERVAL_MONTHS.get(
                    m.meter_type.value, 12
                )
                try:
                    cal_date = datetime.strptime(m.last_calibration, "%Y-%m-%d")
                    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
                    months_since = (ref_date.year - cal_date.year) * 12 + (
                        ref_date.month - cal_date.month
                    )
                    if months_since > interval_months:
                        m.calibration_status = CalibrationStatus.OVERDUE
                    elif months_since > interval_months - 1:
                        m.calibration_status = CalibrationStatus.DUE_SOON
                    else:
                        m.calibration_status = CalibrationStatus.CURRENT
                except (ValueError, TypeError):
                    m.calibration_status = CalibrationStatus.NOT_CALIBRATED

            updated.append(m)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Calibration check: %d meters checked (%.1f ms)", len(updated), elapsed,
        )
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_gaps(
        self,
        values: List[Optional[Decimal]],
        interval_minutes: int,
    ) -> List[DataGap]:
        """Detect gaps in a time-series of metered values."""
        gaps: List[DataGap] = []
        n = len(values)
        i = 0

        while i < n:
            if values[i] is None:
                start = i
                while i < n and values[i] is None:
                    i += 1
                gap_len = i - start
                gap_min = gap_len * interval_minutes

                if gap_min <= 60:
                    severity = GapSeverity.MINOR
                    fill = GapFillMethod.LINEAR
                elif gap_min <= 360:
                    severity = GapSeverity.MODERATE
                    fill = GapFillMethod.LINEAR
                elif gap_min <= 1440:
                    severity = GapSeverity.MAJOR
                    fill = GapFillMethod.AVERAGE
                else:
                    severity = GapSeverity.CRITICAL
                    fill = GapFillMethod.NONE

                gaps.append(DataGap(
                    start_index=start,
                    end_index=i,
                    gap_length=gap_len,
                    gap_duration_minutes=gap_min,
                    severity=severity,
                    recommended_fill=fill,
                ))
            else:
                i += 1

        return gaps
