# -*- coding: utf-8 -*-
"""
DemandAnalysisEngine - PACK-036 Utility Analysis Engine 3
==========================================================

Electrical demand profile analysis and management engine.  Analyses
interval meter data to characterise facility load profiles, compute
load factors, build load duration curves, identify peak events, and
evaluate demand-side management strategies including peak shaving,
load shifting, demand response, and power factor correction.

Key Calculations:
    Load Factor:
        LF = (Average_Demand / Peak_Demand) * 100
        Higher load factor means more uniform load = lower demand charges.

    Power Factor:
        PF = kW / sqrt(kW^2 + kVAR^2)
        Low PF incurs utility penalties; correction via capacitor banks.

    Capacitor Sizing (kVAR):
        kVAR_needed = kW * (tan(acos(PF_current)) - tan(acos(PF_target)))
        Per IEEE Std 18-2012 and IEEE Std 1036-2020.

    Peak Shaving Storage:
        Storage_kWh = (Peak_kW - Target_kW) * Duration_hours
        Battery round-trip efficiency ~90% (lithium-ion, IEC 62933-2:2017).

    Demand Cost:
        Cost = max(Actual_kW, Ratchet_kW) * Rate_per_kW
        Ratchet_kW = max(monthly_peaks[last N]) * ratchet_percentage

    Coincident Peak:
        Facility demand at the time of the utility system peak.
        Used for transmission/distribution cost allocation (FERC Order 888).

    Load Duration Curve:
        Sort interval demands descending; plot against cumulative hours.
        Percentiles P10-P99 characterise load shape.

    Demand Forecast:
        Simple linear trend on historical monthly peaks.
        slope = covariance(x, y) / variance(x)
        Confidence interval: prediction +/- 1.96 * RMSE (95% CI).

Regulatory References:
    - FERC Order 888 (transmission cost allocation via coincident peak)
    - IEEE Std 18-2012 (shunt capacitors for power factor correction)
    - IEEE Std 1036-2020 (application guide for shunt capacitors)
    - IEC 62933-2:2017 (battery energy storage systems)
    - EN 50160:2010 (voltage characteristics of electricity supply)
    - ASHRAE Guideline 14-2014 (measurement of energy demand)
    - NAESB WEQ Business Practices (demand response programs)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Power factor formulas from IEEE Std 18-2012
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
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

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoadCategory(str, Enum):
    """Electrical load category classification.

    BASE:         Continuous load present 24/7 (lighting, controls, servers).
    INTERMEDIATE: Load present during normal operating hours (HVAC, production).
    PEAK:         Short-duration high load (motor starts, process surges).
    CRITICAL:     Life-safety or mission-critical loads (hospital ICU, data centers).
    """
    BASE = "base"
    INTERMEDIATE = "intermediate"
    PEAK = "peak"
    CRITICAL = "critical"

class DemandPeriod(str, Enum):
    """Time-of-use demand period classification.

    ON_PEAK:       Highest rate period (typically weekday 12:00-18:00).
    OFF_PEAK:      Lowest rate period (nights, weekends, holidays).
    MID_PEAK:      Transitional period between on-peak and off-peak.
    SHOULDER:      Seasonal transitional rate period (spring/fall).
    CRITICAL_PEAK: Utility-declared emergency peak events (CPP tariffs).
    """
    ON_PEAK = "on_peak"
    OFF_PEAK = "off_peak"
    MID_PEAK = "mid_peak"
    SHOULDER = "shoulder"
    CRITICAL_PEAK = "critical_peak"

class DemandResponseType(str, Enum):
    """Demand response strategy classification per NAESB WEQ.

    CURTAILMENT:  Reduce load by shutting down non-essential equipment.
    SHIFTING:     Move flexible loads to off-peak hours.
    GENERATION:   Deploy on-site generation (diesel, gas, CHP).
    STORAGE:      Discharge battery or thermal storage.
    BEHAVIORAL:   Occupant-driven reduction (thermostat setback, lighting).
    """
    CURTAILMENT = "curtailment"
    SHIFTING = "shifting"
    GENERATION = "generation"
    STORAGE = "storage"
    BEHAVIORAL = "behavioral"

class PeakType(str, Enum):
    """Peak demand classification for utility billing.

    FACILITY:        Maximum demand at the facility meter.
    COINCIDENT:      Facility demand at time of utility system peak.
    NON_COINCIDENT:  Maximum across non-simultaneous sub-meter peaks.
    TRANSMISSION:    Peak used for transmission cost allocation (FERC).
    DISTRIBUTION:    Peak used for distribution cost allocation.
    """
    FACILITY = "facility"
    COINCIDENT = "coincident"
    NON_COINCIDENT = "non_coincident"
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"

class IntervalResolution(str, Enum):
    """Interval meter data resolution.

    MIN_1:   1-minute intervals (high-resolution submetering).
    MIN_5:   5-minute intervals (advanced metering infrastructure).
    MIN_15:  15-minute intervals (standard utility metering, EU default).
    MIN_30:  30-minute intervals (UK half-hourly settlement).
    HOURLY:  60-minute intervals (legacy or aggregated data).
    """
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOURLY = "60min"

class LoadShiftStrategy(str, Enum):
    """Load shifting strategy classification.

    THERMAL_STORAGE:     Ice or chilled water storage for HVAC.
    BATTERY:             Battery energy storage system (BESS).
    PROCESS_SCHEDULING:  Reschedule production processes off-peak.
    EV_CHARGING:         Defer electric vehicle charging to off-peak.
    PRECOOLING:          Pre-cool building mass during off-peak.
    """
    THERMAL_STORAGE = "thermal_storage"
    BATTERY = "battery"
    PROCESS_SCHEDULING = "process_scheduling"
    EV_CHARGING = "ev_charging"
    PRECOOLING = "precooling"

# ---------------------------------------------------------------------------
# Constants -- Interval Resolution Mapping
# ---------------------------------------------------------------------------

INTERVAL_MINUTES_MAP: Dict[str, int] = {
    IntervalResolution.MIN_1.value: 1,
    IntervalResolution.MIN_5.value: 5,
    IntervalResolution.MIN_15.value: 15,
    IntervalResolution.MIN_30.value: 30,
    IntervalResolution.HOURLY.value: 60,
}
"""Maps IntervalResolution enum values to integer minutes."""

# Default demand response parameters by strategy type.
# Sources: LBNL Demand Response Research Center, NAESB WEQ.
DEMAND_RESPONSE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    DemandResponseType.CURTAILMENT.value: {
        "typical_reduction_pct": Decimal("15"),
        "max_duration_hours": Decimal("4"),
        "implementation_cost_per_kw": Decimal("50"),
        "description": "Shed non-essential loads during peak events",
    },
    DemandResponseType.SHIFTING.value: {
        "typical_reduction_pct": Decimal("20"),
        "max_duration_hours": Decimal("6"),
        "implementation_cost_per_kw": Decimal("120"),
        "description": "Shift flexible loads to off-peak periods",
    },
    DemandResponseType.GENERATION.value: {
        "typical_reduction_pct": Decimal("30"),
        "max_duration_hours": Decimal("8"),
        "implementation_cost_per_kw": Decimal("500"),
        "description": "On-site generation for peak demand reduction",
    },
    DemandResponseType.STORAGE.value: {
        "typical_reduction_pct": Decimal("25"),
        "max_duration_hours": Decimal("4"),
        "implementation_cost_per_kw": Decimal("800"),
        "description": "Battery or thermal storage discharge",
    },
    DemandResponseType.BEHAVIORAL.value: {
        "typical_reduction_pct": Decimal("5"),
        "max_duration_hours": Decimal("8"),
        "implementation_cost_per_kw": Decimal("10"),
        "description": "Occupant-driven demand reduction",
    },
}
"""Default parameters for demand response strategies."""

# Battery storage reference costs.
# Source: NREL Annual Technology Baseline 2024, utility-scale Li-ion.
BATTERY_COST_PER_KWH: Decimal = Decimal("350")
"""Installed cost per kWh for lithium-ion battery storage (EUR/kWh, 2024)."""

BATTERY_ROUND_TRIP_EFFICIENCY: Decimal = Decimal("0.90")
"""Round-trip efficiency for lithium-ion BESS per IEC 62933-2:2017."""

BATTERY_USEFUL_LIFE_YEARS: int = 15
"""Expected useful life of lithium-ion BESS (calendar years)."""

# Capacitor bank reference costs.
# Source: IEEE Std 1036-2020 application guide.
CAPACITOR_COST_PER_KVAR: Decimal = Decimal("25")
"""Installed cost per kVAR for LV switched capacitor bank (EUR/kVAR)."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class IntervalData(BaseModel):
    """Single interval meter reading.

    Represents one time-stamped demand measurement from an interval
    meter (AMI, sub-meter, or SCADA system).

    Attributes:
        timestamp: Interval start time (UTC).
        demand_kw: Average demand during the interval (kW).
        energy_kwh: Energy consumed during the interval (kWh).
        power_factor: Power factor during the interval (0.0 to 1.0).
        reactive_kvar: Reactive power during the interval (kVAR).
        interval_minutes: Duration of the interval in minutes.
    """
    timestamp: datetime = Field(..., description="Interval start time (UTC)")
    demand_kw: float = Field(..., ge=0, description="Average demand (kW)")
    energy_kwh: float = Field(default=0.0, ge=0, description="Energy consumed (kWh)")
    power_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Power factor (0-1)"
    )
    reactive_kvar: float = Field(
        default=0.0, ge=0, description="Reactive power (kVAR)"
    )
    interval_minutes: int = Field(
        default=15, ge=1, le=60, description="Interval duration (min)"
    )

    @field_validator("demand_kw")
    @classmethod
    def validate_demand(cls, v: float) -> float:
        """Ensure demand is within plausible bounds."""
        if v > 1_000_000:
            raise ValueError("Demand exceeds 1 GW sanity check")
        return v

class TOURateSchedule(BaseModel):
    """Time-of-use rate schedule for demand cost analysis.

    Attributes:
        on_peak_rate_per_kw: Demand charge for on-peak period (EUR/kW/month).
        off_peak_rate_per_kw: Demand charge for off-peak period (EUR/kW/month).
        mid_peak_rate_per_kw: Demand charge for mid-peak period (EUR/kW/month).
        on_peak_energy_rate: Energy rate during on-peak (EUR/kWh).
        off_peak_energy_rate: Energy rate during off-peak (EUR/kWh).
        mid_peak_energy_rate: Energy rate during mid-peak (EUR/kWh).
        pf_penalty_threshold: Power factor below which penalties apply.
        pf_penalty_rate_pct: Percentage surcharge for low power factor.
        ratchet_pct: Ratchet clause percentage (0-100).
        ratchet_months: Number of months for ratchet lookback.
    """
    on_peak_rate_per_kw: float = Field(
        default=15.0, ge=0, description="On-peak demand charge (EUR/kW/month)"
    )
    off_peak_rate_per_kw: float = Field(
        default=5.0, ge=0, description="Off-peak demand charge (EUR/kW/month)"
    )
    mid_peak_rate_per_kw: float = Field(
        default=10.0, ge=0, description="Mid-peak demand charge (EUR/kW/month)"
    )
    on_peak_energy_rate: float = Field(
        default=0.15, ge=0, description="On-peak energy rate (EUR/kWh)"
    )
    off_peak_energy_rate: float = Field(
        default=0.08, ge=0, description="Off-peak energy rate (EUR/kWh)"
    )
    mid_peak_energy_rate: float = Field(
        default=0.11, ge=0, description="Mid-peak energy rate (EUR/kWh)"
    )
    pf_penalty_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="Power factor penalty threshold",
    )
    pf_penalty_rate_pct: float = Field(
        default=1.0, ge=0, description="PF penalty per 0.01 below threshold (%)"
    )
    ratchet_pct: float = Field(
        default=80.0, ge=0, le=100.0, description="Ratchet clause percentage"
    )
    ratchet_months: int = Field(
        default=11, ge=1, le=36, description="Ratchet lookback months"
    )

class TOUScheduleEntry(BaseModel):
    """Single entry in a time-of-use schedule definition.

    Attributes:
        period: Demand period classification.
        hour_start: Start hour (0-23 inclusive).
        hour_end: End hour (0-23 inclusive, exclusive boundary).
        days: Days of week (0=Mon, 6=Sun).
    """
    period: DemandPeriod = Field(..., description="Demand period")
    hour_start: int = Field(..., ge=0, le=23, description="Start hour")
    hour_end: int = Field(..., ge=0, le=24, description="End hour (exclusive)")
    days: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Days of week (0=Mon, 6=Sun)",
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class DemandProfile(BaseModel):
    """Demand profile analysis result.

    Summarises interval meter data into a complete demand profile
    including peak, average, minimum demand and load factor.

    Attributes:
        facility_id: Facility identifier.
        period_start: Analysis period start timestamp.
        period_end: Analysis period end timestamp.
        intervals: Number of intervals analysed.
        peak_demand_kw: Maximum demand recorded (kW).
        average_demand_kw: Average demand over period (kW).
        minimum_demand_kw: Minimum demand recorded (kW).
        load_factor_pct: Load factor percentage (avg/peak * 100).
        peak_timestamp: Timestamp of peak demand occurrence.
        total_energy_kwh: Total energy consumed in period (kWh).
        interval_count: Number of intervals.
        interval_resolution: Resolution of source data.
        hours_analysed: Total hours of data analysed.
    """
    facility_id: str = Field(default="", description="Facility identifier")
    period_start: Optional[datetime] = Field(None, description="Period start")
    period_end: Optional[datetime] = Field(None, description="Period end")
    intervals: List[IntervalData] = Field(
        default_factory=list, description="Source interval data"
    )
    peak_demand_kw: float = Field(default=0.0, ge=0, description="Peak demand (kW)")
    average_demand_kw: float = Field(
        default=0.0, ge=0, description="Average demand (kW)"
    )
    minimum_demand_kw: float = Field(
        default=0.0, ge=0, description="Minimum demand (kW)"
    )
    load_factor_pct: float = Field(
        default=0.0, ge=0, le=100.0, description="Load factor (%)"
    )
    peak_timestamp: Optional[datetime] = Field(
        None, description="Peak demand timestamp"
    )
    total_energy_kwh: float = Field(
        default=0.0, ge=0, description="Total energy (kWh)"
    )
    interval_count: int = Field(default=0, ge=0, description="Number of intervals")
    interval_resolution: str = Field(
        default="15min", description="Interval resolution"
    )
    hours_analysed: float = Field(
        default=0.0, ge=0, description="Total hours of data"
    )

class LoadFactor(BaseModel):
    """Load factor calculation result.

    Load factor measures how uniformly electricity is used.
    LF = (Average_Demand / Peak_Demand) * 100.

    Attributes:
        period: Descriptive period label.
        average_demand_kw: Average demand over the period (kW).
        peak_demand_kw: Peak demand in the period (kW).
        load_factor_pct: Load factor percentage.
        utilization_hours: Equivalent full-load utilization hours.
    """
    period: str = Field(default="", description="Period label")
    average_demand_kw: float = Field(
        default=0.0, ge=0, description="Average demand (kW)"
    )
    peak_demand_kw: float = Field(
        default=0.0, ge=0, description="Peak demand (kW)"
    )
    load_factor_pct: float = Field(
        default=0.0, ge=0, le=100.0, description="Load factor (%)"
    )
    utilization_hours: float = Field(
        default=0.0, ge=0, description="Equivalent full-load hours"
    )

class LoadDurationCurve(BaseModel):
    """Load duration curve analysis result.

    Sorted demands (highest to lowest) against cumulative hours.
    Characterises load shape for capacity planning and tariff analysis.

    Attributes:
        sorted_demands: Demand values sorted descending (kW).
        hours_at_or_above: Cumulative hours at or above each demand level.
        percentiles: Key percentile demand values.
        base_load_kw: Estimated base load (P99 demand).
        peak_load_kw: Maximum demand (P1 / highest).
        load_shape_factor: Ratio of area under curve to rectangle (0-1).
    """
    sorted_demands: List[float] = Field(
        default_factory=list, description="Demands sorted descending (kW)"
    )
    hours_at_or_above: List[float] = Field(
        default_factory=list, description="Cumulative hours at each level"
    )
    percentiles: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentile demand values (P10, P25, P50, P75, P90, P99)",
    )
    base_load_kw: float = Field(default=0.0, ge=0, description="Base load (kW)")
    peak_load_kw: float = Field(default=0.0, ge=0, description="Peak load (kW)")
    load_shape_factor: float = Field(
        default=0.0, ge=0, le=1.0, description="Load shape factor"
    )

class PeakEvent(BaseModel):
    """Identified peak demand event.

    Represents a demand reading that exceeded the defined threshold,
    with associated cost impact and avoidability assessment.

    Attributes:
        event_id: Unique event identifier.
        timestamp: When the peak event occurred.
        demand_kw: Demand during the event (kW).
        duration_minutes: Duration of the event (minutes).
        peak_type: Type of peak (facility, coincident, etc.).
        period: TOU period when the event occurred.
        cost_impact_eur: Estimated cost impact of this peak (EUR).
        avoidable: Whether this peak is considered avoidable.
    """
    event_id: str = Field(default_factory=_new_uuid, description="Event ID")
    timestamp: datetime = Field(..., description="Peak event timestamp")
    demand_kw: float = Field(default=0.0, ge=0, description="Demand (kW)")
    duration_minutes: int = Field(
        default=15, ge=1, description="Event duration (min)"
    )
    peak_type: str = Field(
        default=PeakType.FACILITY.value, description="Peak type"
    )
    period: str = Field(
        default=DemandPeriod.ON_PEAK.value, description="TOU period"
    )
    cost_impact_eur: float = Field(
        default=0.0, ge=0, description="Cost impact (EUR)"
    )
    avoidable: bool = Field(default=False, description="Is peak avoidable")

class DemandResponseOpportunity(BaseModel):
    """Demand response opportunity assessment.

    Evaluates a specific demand response strategy for the facility
    including savings potential and implementation economics.

    Attributes:
        strategy: Demand response strategy type.
        shiftable_kw: Amount of load that can be shifted/curtailed (kW).
        duration_hours: Maximum duration of response (hours).
        annual_savings_eur: Estimated annual demand charge savings (EUR).
        implementation_cost: One-time implementation cost (EUR).
        payback_months: Simple payback period (months).
        description: Strategy description.
    """
    strategy: str = Field(..., description="DR strategy type")
    shiftable_kw: float = Field(
        default=0.0, ge=0, description="Shiftable load (kW)"
    )
    duration_hours: float = Field(
        default=0.0, ge=0, description="Response duration (h)"
    )
    annual_savings_eur: float = Field(
        default=0.0, ge=0, description="Annual savings (EUR)"
    )
    implementation_cost: float = Field(
        default=0.0, ge=0, description="Implementation cost (EUR)"
    )
    payback_months: float = Field(
        default=0.0, ge=0, description="Payback period (months)"
    )
    description: str = Field(default="", description="Strategy description")

class PeakShavingAnalysis(BaseModel):
    """Peak shaving / battery storage analysis result.

    Evaluates the economics of using battery energy storage to reduce
    peak demand charges.

    Attributes:
        current_peak_kw: Current facility peak demand (kW).
        target_peak_kw: Target peak demand after shaving (kW).
        reduction_kw: Peak demand reduction achieved (kW).
        storage_kwh_needed: Battery storage capacity required (kWh).
        annual_demand_savings_eur: Annual demand charge savings (EUR).
        battery_cost_eur: Total battery system cost (EUR).
        payback_years: Simple payback period (years).
        reduction_pct: Percentage of peak reduction.
        daily_cycles: Estimated daily discharge cycles.
    """
    current_peak_kw: float = Field(
        default=0.0, ge=0, description="Current peak (kW)"
    )
    target_peak_kw: float = Field(
        default=0.0, ge=0, description="Target peak (kW)"
    )
    reduction_kw: float = Field(
        default=0.0, ge=0, description="Reduction (kW)"
    )
    storage_kwh_needed: float = Field(
        default=0.0, ge=0, description="Storage needed (kWh)"
    )
    annual_demand_savings_eur: float = Field(
        default=0.0, ge=0, description="Annual savings (EUR)"
    )
    battery_cost_eur: float = Field(
        default=0.0, ge=0, description="Battery cost (EUR)"
    )
    payback_years: float = Field(
        default=0.0, ge=0, description="Payback (years)"
    )
    reduction_pct: float = Field(
        default=0.0, ge=0, le=100.0, description="Reduction (%)"
    )
    daily_cycles: float = Field(
        default=0.0, ge=0, description="Estimated daily cycles"
    )

class PowerFactorAnalysis(BaseModel):
    """Power factor correction analysis result.

    Evaluates the economics of installing capacitor banks to improve
    power factor and reduce utility penalty charges.

    Attributes:
        current_pf: Current average power factor.
        target_pf: Target power factor after correction.
        kvar_needed: Capacitor bank size required (kVAR).
        capacitor_cost_eur: Capacitor bank installed cost (EUR).
        annual_penalty_savings_eur: Annual PF penalty savings (EUR).
        payback_months: Simple payback period (months).
        kw_released: Real power capacity released by PF correction (kW).
        current_kva: Current apparent power (kVA).
        corrected_kva: Apparent power after correction (kVA).
    """
    current_pf: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current PF"
    )
    target_pf: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Target PF"
    )
    kvar_needed: float = Field(
        default=0.0, ge=0, description="Capacitor size (kVAR)"
    )
    capacitor_cost_eur: float = Field(
        default=0.0, ge=0, description="Capacitor cost (EUR)"
    )
    annual_penalty_savings_eur: float = Field(
        default=0.0, ge=0, description="Annual penalty savings (EUR)"
    )
    payback_months: float = Field(
        default=0.0, ge=0, description="Payback (months)"
    )
    kw_released: float = Field(
        default=0.0, ge=0, description="kW capacity released"
    )
    current_kva: float = Field(
        default=0.0, ge=0, description="Current apparent power (kVA)"
    )
    corrected_kva: float = Field(
        default=0.0, ge=0, description="Corrected apparent power (kVA)"
    )

class DemandForecast(BaseModel):
    """Monthly demand forecast data point.

    Attributes:
        forecast_month: Month label (e.g. '2026-04').
        predicted_peak_kw: Predicted peak demand (kW).
        confidence_lower: Lower bound of 95% confidence interval (kW).
        confidence_upper: Upper bound of 95% confidence interval (kW).
        method: Forecasting method used.
    """
    forecast_month: str = Field(..., description="Forecast month label")
    predicted_peak_kw: float = Field(
        default=0.0, ge=0, description="Predicted peak (kW)"
    )
    confidence_lower: float = Field(
        default=0.0, ge=0, description="95% CI lower bound (kW)"
    )
    confidence_upper: float = Field(
        default=0.0, ge=0, description="95% CI upper bound (kW)"
    )
    method: str = Field(default="linear_trend", description="Forecast method")

class RatchetImpact(BaseModel):
    """Ratchet clause impact analysis result.

    Evaluates how the utility ratchet clause (minimum billing demand
    based on historical peak) affects demand charges.

    Attributes:
        monthly_peaks: Original monthly peak demands (kW).
        ratchet_kw: Ratchet-adjusted billing demands (kW).
        ratchet_pct: Ratchet percentage applied.
        excess_charges_eur: Total excess demand charges due to ratchet (EUR).
        annual_ratchet_cost_eur: Annualised ratchet cost impact (EUR).
        months_affected: Number of months where ratchet exceeded actual peak.
        highest_ratchet_kw: Highest ratchet demand in the period (kW).
    """
    monthly_peaks: List[float] = Field(
        default_factory=list, description="Monthly peak demands (kW)"
    )
    ratchet_kw: List[float] = Field(
        default_factory=list, description="Ratchet billing demands (kW)"
    )
    ratchet_pct: float = Field(
        default=0.0, ge=0, le=100.0, description="Ratchet percentage"
    )
    excess_charges_eur: float = Field(
        default=0.0, ge=0, description="Total excess charges (EUR)"
    )
    annual_ratchet_cost_eur: float = Field(
        default=0.0, ge=0, description="Annualised ratchet cost (EUR)"
    )
    months_affected: int = Field(
        default=0, ge=0, description="Months affected by ratchet"
    )
    highest_ratchet_kw: float = Field(
        default=0.0, ge=0, description="Highest ratchet demand (kW)"
    )

class LoadShiftOpportunity(BaseModel):
    """Load shifting opportunity assessment.

    Attributes:
        strategy: Load shifting strategy type.
        shiftable_kw: Load that can be shifted (kW).
        from_period: Period to shift load from.
        to_period: Period to shift load to.
        annual_savings_eur: Annual energy cost savings (EUR).
        implementation_cost_eur: Implementation cost (EUR).
        payback_months: Simple payback period (months).
        energy_shifted_kwh_per_day: Daily energy shifted (kWh).
    """
    strategy: str = Field(..., description="Load shift strategy")
    shiftable_kw: float = Field(
        default=0.0, ge=0, description="Shiftable load (kW)"
    )
    from_period: str = Field(
        default=DemandPeriod.ON_PEAK.value, description="From period"
    )
    to_period: str = Field(
        default=DemandPeriod.OFF_PEAK.value, description="To period"
    )
    annual_savings_eur: float = Field(
        default=0.0, ge=0, description="Annual savings (EUR)"
    )
    implementation_cost_eur: float = Field(
        default=0.0, ge=0, description="Implementation cost (EUR)"
    )
    payback_months: float = Field(
        default=0.0, ge=0, description="Payback (months)"
    )
    energy_shifted_kwh_per_day: float = Field(
        default=0.0, ge=0, description="Daily energy shifted (kWh)"
    )

class DemandAnalysisResult(BaseModel):
    """Complete demand analysis result with full provenance.

    Contains demand profile, load factor, load duration curve, peak
    events, demand response opportunities, and economic analyses.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )

    profile: Optional[DemandProfile] = Field(
        None, description="Demand profile"
    )
    load_factor: Optional[LoadFactor] = Field(
        None, description="Load factor result"
    )
    load_duration_curve: Optional[LoadDurationCurve] = Field(
        None, description="Load duration curve"
    )
    peak_events: List[PeakEvent] = Field(
        default_factory=list, description="Peak events"
    )
    demand_response: List[DemandResponseOpportunity] = Field(
        default_factory=list, description="DR opportunities"
    )
    peak_shaving: Optional[PeakShavingAnalysis] = Field(
        None, description="Peak shaving analysis"
    )
    power_factor: Optional[PowerFactorAnalysis] = Field(
        None, description="Power factor analysis"
    )
    forecasts: List[DemandForecast] = Field(
        default_factory=list, description="Demand forecasts"
    )
    ratchet_impact: Optional[RatchetImpact] = Field(
        None, description="Ratchet clause impact"
    )
    load_shift_opportunities: List[LoadShiftOpportunity] = Field(
        default_factory=list, description="Load shift opportunities"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class DemandAnalysisEngine:
    """Electrical demand profile analysis and management engine.

    Provides deterministic, zero-hallucination analysis of interval
    meter data for:
    - Demand profile characterisation (peak, average, minimum, load factor)
    - Load duration curve construction with percentile analysis
    - Peak event identification and cost impact assessment
    - Demand response opportunity evaluation
    - Peak shaving / battery storage economics
    - Power factor correction sizing and payback
    - Monthly demand forecasting (linear trend)
    - Ratchet clause impact analysis
    - Load shifting opportunity identification

    All calculations use Decimal arithmetic for regulatory-grade precision.
    No LLM is used in any calculation path.

    Usage::

        engine = DemandAnalysisEngine()
        profile = engine.analyze_profile(interval_data)
        lf = engine.calculate_load_factor(profile)
        ldc = engine.build_load_duration_curve(profile)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the demand analysis engine."""
        self._dr_defaults = DEMAND_RESPONSE_DEFAULTS
        self._battery_cost_kwh = BATTERY_COST_PER_KWH
        self._battery_efficiency = BATTERY_ROUND_TRIP_EFFICIENCY
        self._capacitor_cost_kvar = CAPACITOR_COST_PER_KVAR

    # -------------------------------------------------------------------
    # Public API: Profile Analysis
    # -------------------------------------------------------------------

    def analyze_profile(
        self,
        intervals: List[IntervalData],
        facility_id: str = "",
    ) -> DemandProfile:
        """Analyse interval meter data to build a demand profile.

        Computes peak, average, minimum demand, load factor, total
        energy, and period timestamps from raw interval data.

        Args:
            intervals: List of interval meter readings.
            facility_id: Optional facility identifier.

        Returns:
            DemandProfile with complete demand characterisation.

        Raises:
            ValueError: If intervals list is empty.
        """
        t0 = time.perf_counter()

        if not intervals:
            raise ValueError("At least one interval reading is required")

        logger.info(
            "Demand profile analysis: facility=%s, intervals=%d",
            facility_id, len(intervals),
        )

        demands = [_decimal(iv.demand_kw) for iv in intervals]
        energies = [_decimal(iv.energy_kwh) for iv in intervals]

        peak_demand = max(demands)
        min_demand = min(demands)
        total_demand = sum(demands)
        avg_demand = _safe_divide(total_demand, _decimal(len(demands)))
        total_energy = sum(energies)

        # Load factor: (avg / peak) * 100
        load_factor = _safe_pct(avg_demand, peak_demand)

        # Find peak timestamp
        peak_idx = demands.index(peak_demand)
        peak_ts = intervals[peak_idx].timestamp

        # Period boundaries
        timestamps = [iv.timestamp for iv in intervals]
        period_start = min(timestamps)
        period_end = max(timestamps)

        # Calculate total hours
        interval_mins = intervals[0].interval_minutes if intervals else 15
        total_hours = _decimal(len(intervals)) * _decimal(interval_mins) / Decimal("60")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        profile = DemandProfile(
            facility_id=facility_id,
            period_start=period_start,
            period_end=period_end,
            intervals=intervals,
            peak_demand_kw=_round2(float(peak_demand)),
            average_demand_kw=_round2(float(avg_demand)),
            minimum_demand_kw=_round2(float(min_demand)),
            load_factor_pct=_round2(float(load_factor)),
            peak_timestamp=peak_ts,
            total_energy_kwh=_round2(float(total_energy)),
            interval_count=len(intervals),
            interval_resolution=f"{interval_mins}min",
            hours_analysed=_round2(float(total_hours)),
        )

        logger.info(
            "Profile complete: peak=%.1f kW, avg=%.1f kW, LF=%.1f%%, "
            "energy=%.1f kWh (%.1f ms)",
            profile.peak_demand_kw, profile.average_demand_kw,
            profile.load_factor_pct, profile.total_energy_kwh, elapsed_ms,
        )
        return profile

    # -------------------------------------------------------------------
    # Public API: Load Factor
    # -------------------------------------------------------------------

    def calculate_load_factor(
        self,
        profile: DemandProfile,
        period_label: str = "",
    ) -> LoadFactor:
        """Calculate load factor from a demand profile.

        Load Factor = (Average_Demand / Peak_Demand) * 100

        A load factor of 100% indicates perfectly uniform load.
        Lower load factors indicate peaky loads with higher demand charges.

        Utilization hours = Load Factor / 100 * Total Hours in period.

        Args:
            profile: Demand profile from analyze_profile().
            period_label: Optional descriptive period label.

        Returns:
            LoadFactor with computed values.
        """
        avg_d = _decimal(profile.average_demand_kw)
        peak_d = _decimal(profile.peak_demand_kw)

        lf = _safe_pct(avg_d, peak_d)

        # Utilization hours: LF/100 * total hours
        util_hours = _safe_divide(lf, Decimal("100")) * _decimal(profile.hours_analysed)

        logger.info(
            "Load factor: avg=%.1f kW, peak=%.1f kW, LF=%.1f%%, util=%.1f h",
            profile.average_demand_kw, profile.peak_demand_kw,
            float(lf), float(util_hours),
        )

        return LoadFactor(
            period=period_label or f"{profile.period_start} to {profile.period_end}",
            average_demand_kw=_round2(float(avg_d)),
            peak_demand_kw=_round2(float(peak_d)),
            load_factor_pct=_round2(float(lf)),
            utilization_hours=_round2(float(util_hours)),
        )

    # -------------------------------------------------------------------
    # Public API: Load Duration Curve
    # -------------------------------------------------------------------

    def build_load_duration_curve(
        self,
        profile: DemandProfile,
    ) -> LoadDurationCurve:
        """Build a load duration curve from the demand profile.

        Sorts all interval demands from highest to lowest and maps
        each to a cumulative number of hours at or above that level.
        Calculates key percentiles (P10, P25, P50, P75, P90, P99).

        Args:
            profile: Demand profile with interval data.

        Returns:
            LoadDurationCurve with sorted demands and percentiles.

        Raises:
            ValueError: If profile has no interval data.
        """
        if not profile.intervals:
            raise ValueError("Profile must contain interval data for LDC")

        t0 = time.perf_counter()

        demands_dec = [_decimal(iv.demand_kw) for iv in profile.intervals]
        demands_dec.sort(reverse=True)

        interval_mins = profile.intervals[0].interval_minutes
        hours_per_interval = _decimal(interval_mins) / Decimal("60")

        # Cumulative hours
        hours = []
        for i in range(len(demands_dec)):
            hours.append(_round2(float(hours_per_interval * _decimal(i + 1))))

        # Percentiles
        n = len(demands_dec)
        percentiles = self._compute_percentiles(demands_dec, n)

        # Base load = P99, Peak = max
        base_load = percentiles.get("P99", Decimal("0"))
        peak_load = demands_dec[0] if demands_dec else Decimal("0")

        # Load shape factor: area under LDC / (peak * total_hours)
        area_under = sum(demands_dec) * hours_per_interval
        total_hours_dec = _decimal(n) * hours_per_interval
        rectangle_area = peak_load * total_hours_dec
        shape_factor = _safe_divide(area_under, rectangle_area)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        sorted_demands_float = [_round2(float(d)) for d in demands_dec]

        percentiles_float = {k: _round2(float(v)) for k, v in percentiles.items()}

        logger.info(
            "LDC built: %d points, base=%.1f kW, peak=%.1f kW, "
            "shape_factor=%.3f (%.1f ms)",
            n, float(base_load), float(peak_load),
            float(shape_factor), elapsed_ms,
        )

        return LoadDurationCurve(
            sorted_demands=sorted_demands_float,
            hours_at_or_above=hours,
            percentiles=percentiles_float,
            base_load_kw=_round2(float(base_load)),
            peak_load_kw=_round2(float(peak_load)),
            load_shape_factor=_round4(float(shape_factor)),
        )

    # -------------------------------------------------------------------
    # Public API: Peak Event Identification
    # -------------------------------------------------------------------

    def identify_peak_events(
        self,
        profile: DemandProfile,
        threshold_kw: Decimal,
        rate_per_kw: Decimal = Decimal("15"),
    ) -> List[PeakEvent]:
        """Identify peak demand events exceeding a threshold.

        Scans interval data for readings that exceed the given
        threshold and calculates the cost impact of each event
        based on the incremental demand above the threshold.

        Args:
            profile: Demand profile with interval data.
            threshold_kw: Demand threshold (kW) for peak event detection.
            rate_per_kw: Demand charge rate (EUR/kW/month).

        Returns:
            List of PeakEvent objects sorted by demand descending.
        """
        if not profile.intervals:
            return []

        t0 = time.perf_counter()
        events: List[PeakEvent] = []

        for iv in profile.intervals:
            demand = _decimal(iv.demand_kw)
            if demand > threshold_kw:
                excess = demand - threshold_kw
                # Cost impact: excess kW * rate per kW (monthly)
                cost_impact = excess * rate_per_kw

                # Determine if avoidable: demand within top 5% is less avoidable
                peak_d = _decimal(profile.peak_demand_kw)
                avoidable = demand < (peak_d * Decimal("0.95"))

                events.append(PeakEvent(
                    timestamp=iv.timestamp,
                    demand_kw=_round2(float(demand)),
                    duration_minutes=iv.interval_minutes,
                    peak_type=PeakType.FACILITY.value,
                    period=self._classify_period(iv.timestamp),
                    cost_impact_eur=_round2(float(cost_impact)),
                    avoidable=avoidable,
                ))

        # Sort by demand descending
        events.sort(key=lambda e: e.demand_kw, reverse=True)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Peak events identified: %d events above %.1f kW (%.1f ms)",
            len(events), float(threshold_kw), elapsed_ms,
        )
        return events

    # -------------------------------------------------------------------
    # Public API: Demand Response Analysis
    # -------------------------------------------------------------------

    def analyze_demand_response(
        self,
        profile: DemandProfile,
        rate: TOURateSchedule,
    ) -> List[DemandResponseOpportunity]:
        """Analyse demand response opportunities for the facility.

        Evaluates each demand response strategy type against the
        facility demand profile and rate schedule to estimate
        potential savings and payback.

        For each strategy:
            shiftable_kw = peak_demand * typical_reduction_pct / 100
            annual_savings = shiftable_kw * demand_rate * 12
            impl_cost = shiftable_kw * cost_per_kw
            payback_months = impl_cost / (annual_savings / 12)

        Args:
            profile: Demand profile with peak demand.
            rate: Time-of-use rate schedule.

        Returns:
            List of DemandResponseOpportunity objects sorted by payback.
        """
        t0 = time.perf_counter()
        opportunities: List[DemandResponseOpportunity] = []

        peak_d = _decimal(profile.peak_demand_kw)
        demand_rate = _decimal(rate.on_peak_rate_per_kw)

        for strategy_key, defaults in self._dr_defaults.items():
            reduction_pct = defaults["typical_reduction_pct"]
            max_duration = defaults["max_duration_hours"]
            cost_per_kw = defaults["implementation_cost_per_kw"]
            description = defaults["description"]

            # Shiftable kW
            shiftable = peak_d * reduction_pct / Decimal("100")

            # Annual savings: shiftable_kw * demand_rate * 12 months
            annual_savings = shiftable * demand_rate * Decimal("12")

            # Implementation cost
            impl_cost = shiftable * cost_per_kw

            # Payback months
            monthly_savings = _safe_divide(annual_savings, Decimal("12"))
            payback = _safe_divide(impl_cost, monthly_savings)

            opportunities.append(DemandResponseOpportunity(
                strategy=strategy_key,
                shiftable_kw=_round2(float(shiftable)),
                duration_hours=_round2(float(max_duration)),
                annual_savings_eur=_round2(float(annual_savings)),
                implementation_cost=_round2(float(impl_cost)),
                payback_months=_round2(float(payback)),
                description=description,
            ))

        # Sort by payback ascending (shortest payback first)
        opportunities.sort(key=lambda o: o.payback_months)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "DR analysis complete: %d strategies evaluated (%.1f ms)",
            len(opportunities), elapsed_ms,
        )
        return opportunities

    # -------------------------------------------------------------------
    # Public API: Peak Shaving Analysis
    # -------------------------------------------------------------------

    def analyze_peak_shaving(
        self,
        profile: DemandProfile,
        target_reduction_pct: Decimal,
        demand_rate_per_kw: Decimal = Decimal("15"),
        peak_duration_hours: Decimal = Decimal("4"),
    ) -> PeakShavingAnalysis:
        """Analyse peak shaving using battery energy storage.

        Evaluates the economics of installing a BESS to reduce
        facility peak demand by a target percentage.

        Storage sizing:
            reduction_kw = peak * target_reduction_pct / 100
            target_peak = peak - reduction_kw
            storage_kWh = reduction_kw * duration / round_trip_efficiency

        Economics:
            annual_savings = reduction_kw * rate * 12
            battery_cost = storage_kWh * cost_per_kWh
            payback = battery_cost / annual_savings

        Args:
            profile: Demand profile with peak demand.
            target_reduction_pct: Target peak reduction percentage (0-100).
            demand_rate_per_kw: Monthly demand charge rate (EUR/kW).
            peak_duration_hours: Expected peak event duration (hours).

        Returns:
            PeakShavingAnalysis with complete economics.
        """
        t0 = time.perf_counter()

        peak_d = _decimal(profile.peak_demand_kw)
        reduction = peak_d * target_reduction_pct / Decimal("100")
        target_peak = peak_d - reduction

        # Storage sizing with round-trip efficiency
        storage_kwh = _safe_divide(
            reduction * peak_duration_hours,
            self._battery_efficiency,
        )

        # Economics
        annual_savings = reduction * demand_rate_per_kw * Decimal("12")
        battery_cost = storage_kwh * self._battery_cost_kwh
        payback_years = _safe_divide(battery_cost, annual_savings)

        # Daily cycles estimate: assume peak shaving on ~250 working days
        daily_energy = reduction * peak_duration_hours
        daily_cycles = _safe_divide(daily_energy, storage_kwh)

        reduction_pct_actual = _safe_pct(reduction, peak_d)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        logger.info(
            "Peak shaving analysis: %.1f kW -> %.1f kW (%.1f%%), "
            "battery=%.0f kWh, payback=%.1f years (%.1f ms)",
            float(peak_d), float(target_peak), float(reduction_pct_actual),
            float(storage_kwh), float(payback_years), elapsed_ms,
        )

        return PeakShavingAnalysis(
            current_peak_kw=_round2(float(peak_d)),
            target_peak_kw=_round2(float(target_peak)),
            reduction_kw=_round2(float(reduction)),
            storage_kwh_needed=_round2(float(storage_kwh)),
            annual_demand_savings_eur=_round2(float(annual_savings)),
            battery_cost_eur=_round2(float(battery_cost)),
            payback_years=_round2(float(payback_years)),
            reduction_pct=_round2(float(reduction_pct_actual)),
            daily_cycles=_round4(float(daily_cycles)),
        )

    # -------------------------------------------------------------------
    # Public API: Power Factor Analysis
    # -------------------------------------------------------------------

    def analyze_power_factor(
        self,
        profile: DemandProfile,
        current_pf: float,
        target_pf: float,
        rate: TOURateSchedule,
    ) -> PowerFactorAnalysis:
        """Analyse power factor correction economics.

        Calculates capacitor bank size needed to improve power factor
        and evaluates the economics of eliminating PF penalty charges.

        Capacitor sizing (IEEE Std 18-2012):
            kVAR_needed = kW * (tan(acos(PF_current)) - tan(acos(PF_target)))

        Apparent power:
            kVA = kW / PF

        Penalty savings:
            penalty_pct = (PF_threshold - PF_current) / 0.01 * penalty_rate
            annual_savings = monthly_bill * penalty_pct * 12

        Args:
            profile: Demand profile with average demand.
            current_pf: Current average power factor (0.0 to 1.0).
            target_pf: Target power factor (0.0 to 1.0).
            rate: Rate schedule with PF penalty parameters.

        Returns:
            PowerFactorAnalysis with sizing and economics.

        Raises:
            ValueError: If current_pf or target_pf is out of range.
        """
        t0 = time.perf_counter()

        if current_pf <= 0 or current_pf > 1.0:
            raise ValueError(f"Current PF must be in (0, 1.0], got {current_pf}")
        if target_pf <= 0 or target_pf > 1.0:
            raise ValueError(f"Target PF must be in (0, 1.0], got {target_pf}")
        if target_pf <= current_pf:
            raise ValueError(
                f"Target PF ({target_pf}) must exceed current PF ({current_pf})"
            )

        avg_kw = _decimal(profile.average_demand_kw)
        pf_cur = _decimal(current_pf)
        pf_tgt = _decimal(target_pf)

        # Capacitor sizing: kVAR = kW * (tan(acos(PF_cur)) - tan(acos(PF_tgt)))
        # Use math for trig, then convert result to Decimal
        theta_cur = math.acos(float(pf_cur))
        theta_tgt = math.acos(float(pf_tgt))
        tan_diff = _decimal(math.tan(theta_cur) - math.tan(theta_tgt))
        kvar_needed = avg_kw * tan_diff

        # Capacitor cost
        cap_cost = kvar_needed * self._capacitor_cost_kvar

        # Apparent power before and after
        current_kva = _safe_divide(avg_kw, pf_cur)
        corrected_kva = _safe_divide(avg_kw, pf_tgt)
        kw_released = current_kva - corrected_kva

        # PF penalty savings
        annual_penalty_savings = self._calculate_pf_penalty_savings(
            avg_kw, pf_cur, pf_tgt, rate,
        )

        # Payback
        monthly_savings = _safe_divide(annual_penalty_savings, Decimal("12"))
        payback_months = _safe_divide(cap_cost, monthly_savings)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        logger.info(
            "PF analysis: %.3f -> %.3f, kVAR=%.1f, cost=%.0f EUR, "
            "savings=%.0f EUR/yr, payback=%.1f months (%.1f ms)",
            current_pf, target_pf, float(kvar_needed), float(cap_cost),
            float(annual_penalty_savings), float(payback_months), elapsed_ms,
        )

        return PowerFactorAnalysis(
            current_pf=_round4(current_pf),
            target_pf=_round4(target_pf),
            kvar_needed=_round2(float(kvar_needed)),
            capacitor_cost_eur=_round2(float(cap_cost)),
            annual_penalty_savings_eur=_round2(float(annual_penalty_savings)),
            payback_months=_round2(float(payback_months)),
            kw_released=_round2(float(kw_released)),
            current_kva=_round2(float(current_kva)),
            corrected_kva=_round2(float(corrected_kva)),
        )

    # -------------------------------------------------------------------
    # Public API: Demand Forecasting
    # -------------------------------------------------------------------

    def forecast_demand(
        self,
        historical_profiles: List[DemandProfile],
        months_ahead: int = 12,
    ) -> List[DemandForecast]:
        """Forecast future monthly peak demands using linear trend.

        Fits a simple linear regression to historical monthly peak
        demands and projects forward.

        Regression:
            slope = covariance(x, y) / variance(x)
            intercept = mean(y) - slope * mean(x)
            predicted = intercept + slope * x_future

        Confidence interval (95%):
            CI = predicted +/- 1.96 * RMSE

        Args:
            historical_profiles: List of DemandProfile, one per month.
            months_ahead: Number of months to forecast.

        Returns:
            List of DemandForecast objects.

        Raises:
            ValueError: If fewer than 2 historical profiles provided.
        """
        t0 = time.perf_counter()

        if len(historical_profiles) < 2:
            raise ValueError("At least 2 historical profiles required for forecasting")

        logger.info(
            "Demand forecast: %d historical profiles, %d months ahead",
            len(historical_profiles), months_ahead,
        )

        # Extract peak demands indexed by position
        peaks = [_decimal(p.peak_demand_kw) for p in historical_profiles]
        n = len(peaks)
        x_vals = [_decimal(i) for i in range(n)]

        # Linear regression
        slope, intercept = self._linear_regression(x_vals, peaks)

        # RMSE for confidence interval
        rmse = self._calculate_rmse(x_vals, peaks, slope, intercept)

        # Confidence multiplier: 1.96 for 95% CI
        ci_mult = Decimal("1.96")

        # Generate forecasts
        forecasts: List[DemandForecast] = []
        for m in range(1, months_ahead + 1):
            x_future = _decimal(n - 1 + m)
            predicted = intercept + slope * x_future

            # Clamp predicted to non-negative
            if predicted < Decimal("0"):
                predicted = Decimal("0")

            ci_width = ci_mult * rmse
            lower = max(Decimal("0"), predicted - ci_width)
            upper = predicted + ci_width

            # Generate month label from last profile
            month_label = self._generate_month_label(historical_profiles[-1], m)

            forecasts.append(DemandForecast(
                forecast_month=month_label,
                predicted_peak_kw=_round2(float(predicted)),
                confidence_lower=_round2(float(lower)),
                confidence_upper=_round2(float(upper)),
                method="linear_trend",
            ))

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Forecast complete: slope=%.2f kW/month, RMSE=%.1f kW, "
            "%d points (%.1f ms)",
            float(slope), float(rmse), len(forecasts), elapsed_ms,
        )
        return forecasts

    # -------------------------------------------------------------------
    # Public API: Ratchet Impact
    # -------------------------------------------------------------------

    def calculate_ratchet_impact(
        self,
        monthly_peaks: List[Decimal],
        ratchet_pct: Decimal = Decimal("80"),
        demand_rate_per_kw: Decimal = Decimal("15"),
    ) -> RatchetImpact:
        """Calculate the impact of a utility ratchet clause.

        A ratchet clause sets the minimum billing demand at a
        percentage of the highest peak demand over a lookback period.

        Billing_Demand = max(Actual_kW, Ratchet_kW)
        Ratchet_kW = max(historical_peaks) * ratchet_pct / 100

        Args:
            monthly_peaks: Monthly peak demand values (kW).
            ratchet_pct: Ratchet clause percentage (e.g. 80 = 80%).
            demand_rate_per_kw: Monthly demand charge rate (EUR/kW).

        Returns:
            RatchetImpact with cost analysis.
        """
        t0 = time.perf_counter()

        if not monthly_peaks:
            return RatchetImpact(ratchet_pct=_round2(float(ratchet_pct)))

        ratchet_demands: List[float] = []
        total_excess = Decimal("0")
        months_affected = 0
        running_max = Decimal("0")

        for i, peak in enumerate(monthly_peaks):
            peak_d = _decimal(peak)
            # Update running maximum
            if peak_d > running_max:
                running_max = peak_d

            # Ratchet demand: percentage of running max
            ratchet_kw = running_max * ratchet_pct / Decimal("100")

            # Billing demand: max of actual and ratchet
            billing_demand = max(peak_d, ratchet_kw)
            ratchet_demands.append(_round2(float(billing_demand)))

            # Excess charge: if ratchet exceeds actual
            if ratchet_kw > peak_d:
                excess = (ratchet_kw - peak_d) * demand_rate_per_kw
                total_excess += excess
                months_affected += 1

        # Annualise: scale to 12 months if fewer
        n = len(monthly_peaks)
        annual_cost = total_excess
        if n > 0 and n < 12:
            annual_cost = total_excess * _safe_divide(
                Decimal("12"), _decimal(n), Decimal("1"),
            )

        highest_ratchet = max(
            [_decimal(r) for r in ratchet_demands]
        ) if ratchet_demands else Decimal("0")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        logger.info(
            "Ratchet impact: %d months, %d affected, excess=%.0f EUR/yr (%.1f ms)",
            n, months_affected, float(annual_cost), elapsed_ms,
        )

        return RatchetImpact(
            monthly_peaks=[_round2(float(p)) for p in monthly_peaks],
            ratchet_kw=ratchet_demands,
            ratchet_pct=_round2(float(ratchet_pct)),
            excess_charges_eur=_round2(float(total_excess)),
            annual_ratchet_cost_eur=_round2(float(annual_cost)),
            months_affected=months_affected,
            highest_ratchet_kw=_round2(float(highest_ratchet)),
        )

    # -------------------------------------------------------------------
    # Public API: Load Shifting
    # -------------------------------------------------------------------

    def identify_load_shifting(
        self,
        profile: DemandProfile,
        tou_schedule: List[TOUScheduleEntry],
        rate: TOURateSchedule,
    ) -> List[LoadShiftOpportunity]:
        """Identify load shifting opportunities based on TOU schedule.

        Analyses interval data against time-of-use periods to find
        loads that could be shifted from on-peak to off-peak, and
        estimates the energy cost savings.

        Savings per kW shifted:
            savings = (on_peak_energy_rate - off_peak_energy_rate)
                      * shiftable_kW * shift_hours * working_days

        Args:
            profile: Demand profile with interval data.
            tou_schedule: List of TOU schedule entries.
            rate: Time-of-use rate schedule.

        Returns:
            List of LoadShiftOpportunity sorted by annual savings.
        """
        t0 = time.perf_counter()

        if not profile.intervals or not tou_schedule:
            return []

        # Classify intervals by TOU period
        period_demands = self._classify_intervals_by_period(
            profile.intervals, tou_schedule,
        )

        on_peak_avg = self._average_demand_for_period(
            period_demands, DemandPeriod.ON_PEAK.value,
        )
        off_peak_avg = self._average_demand_for_period(
            period_demands, DemandPeriod.OFF_PEAK.value,
        )

        # Shiftable load: difference between on-peak avg and off-peak avg
        # Capped at a reasonable fraction of the on-peak demand
        shiftable_base = on_peak_avg - off_peak_avg
        if shiftable_base < Decimal("0"):
            shiftable_base = Decimal("0")

        rate_diff = _decimal(rate.on_peak_energy_rate) - _decimal(rate.off_peak_energy_rate)
        working_days = Decimal("250")  # Working days per year

        opportunities: List[LoadShiftOpportunity] = []

        # Evaluate each strategy
        strategy_params: List[Tuple[str, Decimal, Decimal, Decimal]] = [
            (
                LoadShiftStrategy.THERMAL_STORAGE.value,
                Decimal("0.30"),   # 30% of shiftable base
                Decimal("6"),      # 6 hours duration
                Decimal("200"),    # EUR/kW implementation cost
            ),
            (
                LoadShiftStrategy.BATTERY.value,
                Decimal("0.25"),   # 25% of shiftable base
                Decimal("4"),      # 4 hours duration
                Decimal("800"),    # EUR/kW implementation cost
            ),
            (
                LoadShiftStrategy.PROCESS_SCHEDULING.value,
                Decimal("0.40"),   # 40% of shiftable base
                Decimal("8"),      # 8 hours duration
                Decimal("50"),     # EUR/kW implementation cost
            ),
            (
                LoadShiftStrategy.EV_CHARGING.value,
                Decimal("0.10"),   # 10% of shiftable base
                Decimal("8"),      # 8 hours charging
                Decimal("30"),     # EUR/kW (smart charger cost)
            ),
            (
                LoadShiftStrategy.PRECOOLING.value,
                Decimal("0.15"),   # 15% of shiftable base
                Decimal("3"),      # 3 hours precooling
                Decimal("25"),     # EUR/kW (controls cost)
            ),
        ]

        for strategy_name, fraction, duration, cost_per_kw in strategy_params:
            shiftable_kw = shiftable_base * fraction
            if shiftable_kw <= Decimal("0"):
                continue

            # Daily energy shifted
            daily_energy = shiftable_kw * duration

            # Annual savings: rate_diff * daily_energy * working_days
            annual_savings = rate_diff * daily_energy * working_days

            # Also add demand charge savings (reduced on-peak demand)
            demand_savings = shiftable_kw * _decimal(rate.on_peak_rate_per_kw) * Decimal("12")
            annual_savings += demand_savings

            # Implementation cost
            impl_cost = shiftable_kw * cost_per_kw

            # Payback
            monthly_savings = _safe_divide(annual_savings, Decimal("12"))
            payback = _safe_divide(impl_cost, monthly_savings)

            opportunities.append(LoadShiftOpportunity(
                strategy=strategy_name,
                shiftable_kw=_round2(float(shiftable_kw)),
                from_period=DemandPeriod.ON_PEAK.value,
                to_period=DemandPeriod.OFF_PEAK.value,
                annual_savings_eur=_round2(float(annual_savings)),
                implementation_cost_eur=_round2(float(impl_cost)),
                payback_months=_round2(float(payback)),
                energy_shifted_kwh_per_day=_round2(float(daily_energy)),
            ))

        # Sort by annual savings descending
        opportunities.sort(key=lambda o: o.annual_savings_eur, reverse=True)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Load shift analysis: %d opportunities found (%.1f ms)",
            len(opportunities), elapsed_ms,
        )
        return opportunities

    # -------------------------------------------------------------------
    # Public API: Full Analysis
    # -------------------------------------------------------------------

    def run_full_analysis(
        self,
        intervals: List[IntervalData],
        facility_id: str = "",
        rate: Optional[TOURateSchedule] = None,
        tou_schedule: Optional[List[TOUScheduleEntry]] = None,
        target_reduction_pct: Decimal = Decimal("10"),
        current_pf: float = 0.85,
        target_pf: float = 0.95,
    ) -> DemandAnalysisResult:
        """Run complete demand analysis pipeline.

        Executes all analysis methods in sequence and returns a
        comprehensive result with provenance hash.

        Args:
            intervals: Interval meter data.
            facility_id: Facility identifier.
            rate: Time-of-use rate schedule (uses defaults if None).
            tou_schedule: TOU schedule entries (optional).
            target_reduction_pct: Target peak reduction for peak shaving.
            current_pf: Current power factor for PF analysis.
            target_pf: Target power factor.

        Returns:
            DemandAnalysisResult with complete analysis and provenance.
        """
        t0 = time.perf_counter()

        if rate is None:
            rate = TOURateSchedule()

        logger.info(
            "Full demand analysis: facility=%s, intervals=%d",
            facility_id, len(intervals),
        )

        # 1. Build demand profile
        profile = self.analyze_profile(intervals, facility_id)

        # 2. Load factor
        load_factor = self.calculate_load_factor(profile)

        # 3. Load duration curve
        ldc = self.build_load_duration_curve(profile)

        # 4. Peak events (threshold = 90% of peak)
        threshold = _decimal(profile.peak_demand_kw) * Decimal("0.90")
        peak_events = self.identify_peak_events(
            profile, threshold, _decimal(rate.on_peak_rate_per_kw),
        )

        # 5. Demand response
        dr_opportunities = self.analyze_demand_response(profile, rate)

        # 6. Peak shaving
        peak_shaving = self.analyze_peak_shaving(
            profile, target_reduction_pct,
            _decimal(rate.on_peak_rate_per_kw),
        )

        # 7. Power factor analysis (only if PF data is available)
        pf_analysis = None
        if current_pf < target_pf:
            try:
                pf_analysis = self.analyze_power_factor(
                    profile, current_pf, target_pf, rate,
                )
            except ValueError as e:
                logger.warning("PF analysis skipped: %s", str(e))

        # 8. Load shifting
        load_shifts: List[LoadShiftOpportunity] = []
        if tou_schedule:
            load_shifts = self.identify_load_shifting(profile, tou_schedule, rate)

        # 9. Recommendations
        recommendations = self._generate_recommendations(
            profile, load_factor, ldc, peak_events, pf_analysis,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = DemandAnalysisResult(
            profile=profile,
            load_factor=load_factor,
            load_duration_curve=ldc,
            peak_events=peak_events,
            demand_response=dr_opportunities,
            peak_shaving=peak_shaving,
            power_factor=pf_analysis,
            load_shift_opportunities=load_shifts,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full analysis complete: facility=%s, hash=%s (%.1f ms)",
            facility_id, result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Internal: Percentile Computation
    # -------------------------------------------------------------------

    def _compute_percentiles(
        self,
        sorted_desc: List[Decimal],
        n: int,
    ) -> Dict[str, Decimal]:
        """Compute key percentiles from a descending-sorted demand list.

        Uses nearest-rank method for percentile calculation.

        Percentiles computed: P10, P25, P50, P75, P90, P99.

        Args:
            sorted_desc: Demands sorted descending.
            n: Number of data points.

        Returns:
            Dict mapping percentile label to demand value.
        """
        if n == 0:
            return {}

        percentile_keys = [
            ("P10", 10), ("P25", 25), ("P50", 50),
            ("P75", 75), ("P90", 90), ("P99", 99),
        ]

        result: Dict[str, Decimal] = {}
        for label, pct in percentile_keys:
            # Nearest-rank: index = ceil(pct/100 * n) - 1
            # For descending sort, P10 means 10% of time demand is at/above
            rank = math.ceil(pct / 100.0 * n)
            idx = max(0, min(rank - 1, n - 1))
            result[label] = sorted_desc[idx]

        return result

    # -------------------------------------------------------------------
    # Internal: Period Classification
    # -------------------------------------------------------------------

    def _classify_period(self, timestamp: datetime) -> str:
        """Classify a timestamp into a TOU demand period.

        Uses a simplified default schedule:
        - On-peak:  weekdays 12:00-18:00
        - Mid-peak: weekdays 08:00-12:00 and 18:00-21:00
        - Off-peak: all other times

        Args:
            timestamp: The timestamp to classify.

        Returns:
            DemandPeriod value string.
        """
        weekday = timestamp.weekday()  # 0=Mon, 6=Sun
        hour = timestamp.hour

        if weekday >= 5:  # Weekend
            return DemandPeriod.OFF_PEAK.value

        if 12 <= hour < 18:
            return DemandPeriod.ON_PEAK.value
        elif (8 <= hour < 12) or (18 <= hour < 21):
            return DemandPeriod.MID_PEAK.value
        else:
            return DemandPeriod.OFF_PEAK.value

    def _classify_interval_period(
        self,
        timestamp: datetime,
        tou_schedule: List[TOUScheduleEntry],
    ) -> str:
        """Classify a timestamp using a custom TOU schedule.

        Args:
            timestamp: The timestamp to classify.
            tou_schedule: List of TOU schedule entries.

        Returns:
            DemandPeriod value string.
        """
        weekday = timestamp.weekday()
        hour = timestamp.hour

        for entry in tou_schedule:
            if weekday in entry.days:
                if entry.hour_start <= hour < entry.hour_end:
                    return entry.period.value

        return DemandPeriod.OFF_PEAK.value

    # -------------------------------------------------------------------
    # Internal: Interval Classification
    # -------------------------------------------------------------------

    def _classify_intervals_by_period(
        self,
        intervals: List[IntervalData],
        tou_schedule: List[TOUScheduleEntry],
    ) -> Dict[str, List[Decimal]]:
        """Classify interval demands by TOU period.

        Args:
            intervals: Interval meter data.
            tou_schedule: TOU schedule entries.

        Returns:
            Dict mapping period name to list of demand values.
        """
        period_demands: Dict[str, List[Decimal]] = {}

        for iv in intervals:
            period = self._classify_interval_period(iv.timestamp, tou_schedule)
            if period not in period_demands:
                period_demands[period] = []
            period_demands[period].append(_decimal(iv.demand_kw))

        return period_demands

    def _average_demand_for_period(
        self,
        period_demands: Dict[str, List[Decimal]],
        period: str,
    ) -> Decimal:
        """Calculate average demand for a specific TOU period.

        Args:
            period_demands: Classified demand values by period.
            period: Period name to compute average for.

        Returns:
            Average demand as Decimal; Decimal("0") if no data.
        """
        demands = period_demands.get(period, [])
        if not demands:
            return Decimal("0")
        return _safe_divide(sum(demands), _decimal(len(demands)))

    # -------------------------------------------------------------------
    # Internal: Linear Regression
    # -------------------------------------------------------------------

    def _linear_regression(
        self,
        x_vals: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Perform ordinary least squares linear regression.

        slope = covariance(x, y) / variance(x)
        intercept = mean(y) - slope * mean(x)

        Args:
            x_vals: Independent variable values.
            y_vals: Dependent variable values.

        Returns:
            Tuple of (slope, intercept) as Decimals.
        """
        n = _decimal(len(x_vals))
        if n <= Decimal("1"):
            return Decimal("0"), y_vals[0] if y_vals else Decimal("0")

        mean_x = _safe_divide(sum(x_vals), n)
        mean_y = _safe_divide(sum(y_vals), n)

        # Covariance and variance
        cov_xy = Decimal("0")
        var_x = Decimal("0")

        for x, y in zip(x_vals, y_vals):
            dx = x - mean_x
            dy = y - mean_y
            cov_xy += dx * dy
            var_x += dx * dx

        slope = _safe_divide(cov_xy, var_x)
        intercept = mean_y - slope * mean_x

        return slope, intercept

    def _calculate_rmse(
        self,
        x_vals: List[Decimal],
        y_vals: List[Decimal],
        slope: Decimal,
        intercept: Decimal,
    ) -> Decimal:
        """Calculate root mean squared error of regression.

        RMSE = sqrt( sum( (y_actual - y_predicted)^2 ) / n )

        Args:
            x_vals: Independent variable values.
            y_vals: Actual dependent variable values.
            slope: Regression slope.
            intercept: Regression intercept.

        Returns:
            RMSE as Decimal.
        """
        n = len(x_vals)
        if n == 0:
            return Decimal("0")

        sse = Decimal("0")
        for x, y in zip(x_vals, y_vals):
            predicted = intercept + slope * x
            residual = y - predicted
            sse += residual * residual

        mse = _safe_divide(sse, _decimal(n))
        # Decimal sqrt via float conversion (acceptable for RMSE)
        rmse = _decimal(math.sqrt(float(mse)))
        return rmse

    # -------------------------------------------------------------------
    # Internal: Month Label Generation
    # -------------------------------------------------------------------

    def _generate_month_label(
        self,
        last_profile: DemandProfile,
        months_offset: int,
    ) -> str:
        """Generate a month label offset from the last profile.

        Args:
            last_profile: Last historical profile.
            months_offset: Number of months to offset forward.

        Returns:
            Month label string (YYYY-MM format).
        """
        if last_profile.period_end is not None:
            base_dt = last_profile.period_end
        elif last_profile.period_start is not None:
            base_dt = last_profile.period_start
        else:
            base_dt = utcnow()

        # Calculate target month
        year = base_dt.year
        month = base_dt.month + months_offset

        # Handle month overflow
        while month > 12:
            month -= 12
            year += 1

        return f"{year:04d}-{month:02d}"

    # -------------------------------------------------------------------
    # Internal: PF Penalty Savings
    # -------------------------------------------------------------------

    def _calculate_pf_penalty_savings(
        self,
        avg_kw: Decimal,
        current_pf: Decimal,
        target_pf: Decimal,
        rate: TOURateSchedule,
    ) -> Decimal:
        """Calculate annual power factor penalty savings.

        Penalty structure:
            If PF < threshold: penalty = (threshold - PF) / 0.01 * rate_pct
            Applied as a percentage surcharge on demand charges.

        Args:
            avg_kw: Average real power demand (kW).
            current_pf: Current power factor.
            target_pf: Target power factor (after correction).
            rate: Rate schedule with PF penalty parameters.

        Returns:
            Annual penalty savings as Decimal (EUR).
        """
        threshold = _decimal(rate.pf_penalty_threshold)
        penalty_rate = _decimal(rate.pf_penalty_rate_pct)
        demand_rate = _decimal(rate.on_peak_rate_per_kw)

        # Current penalty
        current_penalty_pct = Decimal("0")
        if current_pf < threshold:
            steps = _safe_divide(
                threshold - current_pf, Decimal("0.01"),
            )
            current_penalty_pct = steps * penalty_rate

        # Target penalty (should be zero if target >= threshold)
        target_penalty_pct = Decimal("0")
        if target_pf < threshold:
            steps = _safe_divide(
                threshold - target_pf, Decimal("0.01"),
            )
            target_penalty_pct = steps * penalty_rate

        # Monthly demand charge
        monthly_demand_charge = avg_kw * demand_rate

        # Monthly savings
        penalty_diff = current_penalty_pct - target_penalty_pct
        monthly_savings = monthly_demand_charge * penalty_diff / Decimal("100")

        # Annual savings
        return monthly_savings * Decimal("12")

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        profile: DemandProfile,
        load_factor: LoadFactor,
        ldc: LoadDurationCurve,
        peak_events: List[PeakEvent],
        pf_analysis: Optional[PowerFactorAnalysis],
    ) -> List[str]:
        """Generate deterministic recommendations from analysis results.

        All recommendations are threshold-based comparisons. No LLM
        involvement.

        Args:
            profile: Demand profile.
            load_factor: Load factor result.
            ldc: Load duration curve.
            peak_events: Identified peak events.
            pf_analysis: Power factor analysis (may be None).

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Low load factor
        if load_factor.load_factor_pct < 40.0:
            recs.append(
                f"Load factor is {load_factor.load_factor_pct}%, which is below "
                f"the 40% threshold for efficient demand management. Consider "
                f"load shifting or process scheduling to flatten the demand "
                f"profile and reduce peak demand charges."
            )
        elif load_factor.load_factor_pct < 60.0:
            recs.append(
                f"Load factor is {load_factor.load_factor_pct}%, indicating moderate "
                f"load variability. Investigate on-peak loads that could be shifted "
                f"to off-peak periods to improve the load factor above 60%."
            )

        # R2: High base-to-peak ratio suggests improvement potential
        if ldc.peak_load_kw > 0 and ldc.base_load_kw > 0:
            base_to_peak = ldc.base_load_kw / ldc.peak_load_kw
            if base_to_peak < 0.3:
                recs.append(
                    f"Base load is only {_round2(base_to_peak * 100)}% of peak load, "
                    f"indicating highly variable demand. Peak shaving with battery "
                    f"storage could significantly reduce demand charges."
                )

        # R3: Avoidable peak events
        avoidable_count = sum(1 for e in peak_events if e.avoidable)
        if avoidable_count > 0:
            total_avoidable_cost = sum(
                e.cost_impact_eur for e in peak_events if e.avoidable
            )
            recs.append(
                f"{avoidable_count} avoidable peak events identified with a "
                f"combined cost impact of {_round2(total_avoidable_cost)} EUR/month. "
                f"Implement demand limiting controls or load shedding automation."
            )

        # R4: Power factor correction
        if pf_analysis and pf_analysis.current_pf < 0.90:
            recs.append(
                f"Power factor is {pf_analysis.current_pf}, below the 0.90 utility "
                f"threshold. Installing {_round2(pf_analysis.kvar_needed)} kVAR of "
                f"capacitor banks would save {_round2(pf_analysis.annual_penalty_savings_eur)} "
                f"EUR/year with a payback of {_round2(pf_analysis.payback_months)} months."
            )

        # R5: Data quality -- interval count
        if profile.interval_count < 96:
            recs.append(
                "Fewer than 96 intervals (one day of 15-min data) were analysed. "
                "A minimum of 30 days of data is recommended for reliable "
                "demand profile characterisation (ASHRAE Guideline 14-2014)."
            )

        # R6: High peak-to-average ratio
        if profile.average_demand_kw > 0:
            par = profile.peak_demand_kw / profile.average_demand_kw
            if par > 3.0:
                recs.append(
                    f"Peak-to-average ratio is {_round2(par)}, indicating sharp "
                    f"demand spikes. Consider demand limiting relays or soft "
                    f"starters on large motors to reduce inrush current peaks."
                )

        return recs

# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def create_engine() -> DemandAnalysisEngine:
    """Create and return a new DemandAnalysisEngine instance.

    Convenience factory function for use by the GreenLang agent framework.

    Returns:
        Configured DemandAnalysisEngine ready for use.
    """
    return DemandAnalysisEngine()
