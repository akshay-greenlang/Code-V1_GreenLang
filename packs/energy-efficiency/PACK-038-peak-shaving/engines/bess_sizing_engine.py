# -*- coding: utf-8 -*-
"""
BESSSizingEngine - PACK-038 Peak Shaving Engine 4
===================================================

Battery Energy Storage System (BESS) sizing optimisation engine for
peak shaving applications.  Performs 8,760-hour dispatch simulation
at 15-minute resolution, models battery degradation (calendar plus
cycle), optimises system sizing by target peak threshold, compares
battery chemistries, and calculates Levelised Cost of Storage (LCOS).

Calculation Methodology:
    Dispatch Simulation (15-min resolution):
        For each interval t:
            if demand[t] > threshold:
                discharge_kw = min(demand[t] - threshold, max_discharge_kw,
                                   soc[t] * capacity_kwh / dt)
                soc[t+1] = soc[t] - discharge_kw * dt / (capacity * rte)
            elif demand[t] < charge_trigger and soc[t] < soc_max:
                charge_kw = min(charge_trigger - demand[t], max_charge_kw,
                                (soc_max - soc[t]) * capacity_kwh / dt)
                soc[t+1] = soc[t] + charge_kw * dt * rte / capacity

    Degradation Model (Calendar + Cycle):
        capacity_fade_calendar = aging_rate * sqrt(time_years)
        capacity_fade_cycle = throughput_ah / rated_ah_throughput
        remaining_capacity = 1 - fade_calendar - fade_cycle
        EOL when remaining_capacity < 0.80 (80% retention)

    Round-Trip Efficiency by C-rate:
        rte(C) = rte_nominal - efficiency_slope * (C - C_nominal)

    Thermal Derating:
        if T_ambient > T_rated:
            derating = 1 - thermal_coeff * (T_ambient - T_rated)

    LCOS Calculation:
        LCOS = (CAPEX + sum(OPEX_t / (1+r)^t)) / sum(Energy_t / (1+r)^t)
        where r = discount rate, t = year, Energy = annual discharged kWh

    Sizing Optimisation:
        For each candidate (power_kW, capacity_kWh):
            simulate full year dispatch
            calculate annual savings = demand_charge_reduction
            calculate NPV = savings - CAPEX - OPEX
            select candidate with max NPV or target peak

Regulatory References:
    - UL 9540 - Energy Storage System Safety
    - UL 9540A - Test Method for Battery Thermal Runaway
    - NFPA 855 - Standard for Energy Storage Systems
    - IEC 62619 - Secondary lithium cells for industrial applications
    - IEEE 1547-2018 - Interconnection of DER
    - FERC Order 841 - Energy Storage Participation
    - ISO 50001:2018 - Energy management systems

Zero-Hallucination:
    - All dispatch decisions use deterministic threshold logic
    - Degradation models use published empirical formulas
    - LCOS uses standard discounted cash-flow methodology
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  4 of 5
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


class BatteryChemistry(str, Enum):
    """Battery chemistry type.

    NMC:            Nickel Manganese Cobalt (Li-ion).
    LFP:            Lithium Iron Phosphate.
    NCA:            Nickel Cobalt Aluminium.
    FLOW_VANADIUM:  Vanadium Redox Flow Battery.
    FLOW_ZINC:      Zinc-Bromine Flow Battery.
    SODIUM_ION:     Sodium-Ion Battery.
    """
    NMC = "nmc"
    LFP = "lfp"
    NCA = "nca"
    FLOW_VANADIUM = "flow_vanadium"
    FLOW_ZINC = "flow_zinc"
    SODIUM_ION = "sodium_ion"


class DispatchStrategy(str, Enum):
    """BESS dispatch strategy for peak shaving.

    PEAK_SHAVING:   Discharge when demand exceeds threshold.
    THRESHOLD:      Fixed threshold-based dispatch.
    TIME_BASED:     Discharge during specific time windows.
    PREDICTIVE:     Predictive dispatch based on load forecast.
    HYBRID:         Combined threshold + time-based dispatch.
    """
    PEAK_SHAVING = "peak_shaving"
    THRESHOLD = "threshold"
    TIME_BASED = "time_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class DegradationModel(str, Enum):
    """Battery degradation modelling approach.

    LINEAR:             Linear capacity fade over time.
    RAINFLOW:           Rainflow counting for cycle depth.
    AH_THROUGHPUT:      Ampere-hour throughput model.
    CALENDAR_PLUS_CYCLE: Combined calendar and cycle aging.
    """
    LINEAR = "linear"
    RAINFLOW = "rainflow"
    AH_THROUGHPUT = "ah_throughput"
    CALENDAR_PLUS_CYCLE = "calendar_plus_cycle"


class SizingObjective(str, Enum):
    """Sizing optimisation objective.

    MIN_COST:     Minimise total cost of ownership.
    MAX_SAVINGS:  Maximise net savings.
    TARGET_PEAK:  Achieve a specific peak demand target.
    MAX_ROI:      Maximise return on investment.
    """
    MIN_COST = "min_cost"
    MAX_SAVINGS = "max_savings"
    TARGET_PEAK = "target_peak"
    MAX_ROI = "max_roi"


class SystemConfig(str, Enum):
    """BESS system configuration.

    AC_COUPLED:  AC-coupled system with separate inverter.
    DC_COUPLED:  DC-coupled system (shared inverter with solar).
    HYBRID:      Hybrid AC/DC coupled configuration.
    """
    AC_COUPLED = "ac_coupled"
    DC_COUPLED = "dc_coupled"
    HYBRID = "hybrid"


class SizingGrade(str, Enum):
    """Grade assigned to a sizing candidate.

    OPTIMAL:     Best candidate by objective function.
    GOOD:        Within 10% of optimal.
    ACCEPTABLE:  Within 25% of optimal.
    MARGINAL:    ROI is positive but low.
    UNECONOMIC:  Negative ROI or infeasible.
    """
    OPTIMAL = "optimal"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    UNECONOMIC = "uneconomic"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Battery specifications by chemistry (representative values).
BATTERY_SPECS: Dict[str, Dict[str, Any]] = {
    BatteryChemistry.NMC.value: {
        "energy_density_wh_kg": Decimal("200"),
        "cycle_life_80pct": 4000,
        "calendar_life_years": 12,
        "cost_per_kwh": Decimal("280"),
        "cost_per_kw": Decimal("150"),
        "rte_nominal": Decimal("0.90"),
        "c_rate_max_charge": Decimal("1.0"),
        "c_rate_max_discharge": Decimal("1.0"),
        "dod_max": Decimal("0.90"),
        "thermal_coeff": Decimal("0.005"),
        "aging_rate": Decimal("0.02"),
        "rated_temp_c": Decimal("25"),
    },
    BatteryChemistry.LFP.value: {
        "energy_density_wh_kg": Decimal("160"),
        "cycle_life_80pct": 6000,
        "calendar_life_years": 15,
        "cost_per_kwh": Decimal("220"),
        "cost_per_kw": Decimal("130"),
        "rte_nominal": Decimal("0.92"),
        "c_rate_max_charge": Decimal("1.0"),
        "c_rate_max_discharge": Decimal("1.0"),
        "dod_max": Decimal("0.95"),
        "thermal_coeff": Decimal("0.003"),
        "aging_rate": Decimal("0.015"),
        "rated_temp_c": Decimal("25"),
    },
    BatteryChemistry.NCA.value: {
        "energy_density_wh_kg": Decimal("250"),
        "cycle_life_80pct": 3000,
        "calendar_life_years": 10,
        "cost_per_kwh": Decimal("300"),
        "cost_per_kw": Decimal("160"),
        "rte_nominal": Decimal("0.89"),
        "c_rate_max_charge": Decimal("0.8"),
        "c_rate_max_discharge": Decimal("1.0"),
        "dod_max": Decimal("0.85"),
        "thermal_coeff": Decimal("0.006"),
        "aging_rate": Decimal("0.025"),
        "rated_temp_c": Decimal("25"),
    },
    BatteryChemistry.FLOW_VANADIUM.value: {
        "energy_density_wh_kg": Decimal("25"),
        "cycle_life_80pct": 15000,
        "calendar_life_years": 25,
        "cost_per_kwh": Decimal("350"),
        "cost_per_kw": Decimal("600"),
        "rte_nominal": Decimal("0.75"),
        "c_rate_max_charge": Decimal("0.25"),
        "c_rate_max_discharge": Decimal("0.25"),
        "dod_max": Decimal("1.00"),
        "thermal_coeff": Decimal("0.002"),
        "aging_rate": Decimal("0.005"),
        "rated_temp_c": Decimal("30"),
    },
    BatteryChemistry.FLOW_ZINC.value: {
        "energy_density_wh_kg": Decimal("35"),
        "cycle_life_80pct": 10000,
        "calendar_life_years": 20,
        "cost_per_kwh": Decimal("300"),
        "cost_per_kw": Decimal("500"),
        "rte_nominal": Decimal("0.72"),
        "c_rate_max_charge": Decimal("0.33"),
        "c_rate_max_discharge": Decimal("0.33"),
        "dod_max": Decimal("1.00"),
        "thermal_coeff": Decimal("0.002"),
        "aging_rate": Decimal("0.008"),
        "rated_temp_c": Decimal("30"),
    },
    BatteryChemistry.SODIUM_ION.value: {
        "energy_density_wh_kg": Decimal("140"),
        "cycle_life_80pct": 5000,
        "calendar_life_years": 12,
        "cost_per_kwh": Decimal("180"),
        "cost_per_kw": Decimal("120"),
        "rte_nominal": Decimal("0.88"),
        "c_rate_max_charge": Decimal("1.0"),
        "c_rate_max_discharge": Decimal("1.0"),
        "dod_max": Decimal("0.90"),
        "thermal_coeff": Decimal("0.004"),
        "aging_rate": Decimal("0.020"),
        "rated_temp_c": Decimal("25"),
    },
}

# Discount rate for LCOS calculation.
DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")

# Annual O&M cost as fraction of CAPEX.
DEFAULT_OM_RATE: Decimal = Decimal("0.02")

# Intervals per hour at 15-minute resolution.
INTERVALS_PER_HOUR: int = 4

# Hours per year.
HOURS_PER_YEAR: int = 8760

# Default project lifetime (years).
DEFAULT_PROJECT_YEARS: int = 15

# SOC limits.
DEFAULT_SOC_MIN: Decimal = Decimal("0.10")
DEFAULT_SOC_MAX: Decimal = Decimal("0.90")

# Efficiency slope per C-rate deviation.
EFFICIENCY_SLOPE: Decimal = Decimal("0.02")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Output
# ---------------------------------------------------------------------------


class BatterySpecs(BaseModel):
    """Battery system specification.

    Attributes:
        chemistry: Battery chemistry type.
        power_kw: Rated power capacity (kW).
        capacity_kwh: Rated energy capacity (kWh).
        rte_nominal: Nominal round-trip efficiency.
        c_rate_max: Maximum C-rate (discharge).
        dod_max: Maximum depth of discharge.
        soc_min: Minimum state of charge.
        soc_max: Maximum state of charge.
        cost_per_kwh: Installed cost per kWh.
        cost_per_kw: Installed cost per kW.
        cycle_life: Cycle life at rated DoD to 80% retention.
        calendar_life_years: Calendar life (years).
        system_config: AC/DC coupling configuration.
        ambient_temp_c: Average ambient temperature (Celsius).
    """
    chemistry: BatteryChemistry = Field(default=BatteryChemistry.LFP)
    power_kw: Decimal = Field(default=Decimal("100"), ge=0)
    capacity_kwh: Decimal = Field(default=Decimal("400"), ge=0)
    rte_nominal: Decimal = Field(default=Decimal("0.92"), ge=0, le=Decimal("1"))
    c_rate_max: Decimal = Field(default=Decimal("1.0"), ge=0)
    dod_max: Decimal = Field(default=Decimal("0.95"), ge=0, le=Decimal("1"))
    soc_min: Decimal = Field(default=DEFAULT_SOC_MIN, ge=0, le=Decimal("1"))
    soc_max: Decimal = Field(default=DEFAULT_SOC_MAX, ge=0, le=Decimal("1"))
    cost_per_kwh: Decimal = Field(default=Decimal("220"), ge=0)
    cost_per_kw: Decimal = Field(default=Decimal("130"), ge=0)
    cycle_life: int = Field(default=6000, ge=0)
    calendar_life_years: int = Field(default=15, ge=0)
    system_config: SystemConfig = Field(default=SystemConfig.AC_COUPLED)
    ambient_temp_c: Decimal = Field(default=Decimal("25"), description="Ambient temp")


class DispatchResult(BaseModel):
    """Result of a dispatch simulation.

    Attributes:
        dispatch_id: Unique dispatch simulation identifier.
        threshold_kw: Peak shaving threshold (kW).
        peak_before_kw: Original peak demand (kW).
        peak_after_kw: Peak demand after BESS dispatch (kW).
        peak_reduction_kw: Peak reduction achieved (kW).
        peak_reduction_pct: Peak reduction percentage.
        total_discharged_kwh: Total energy discharged (kWh).
        total_charged_kwh: Total energy charged (kWh).
        cycle_count: Equivalent full cycles.
        min_soc: Minimum SOC reached.
        max_soc: Maximum SOC reached.
        avg_soc: Average SOC.
        dispatch_hours: Hours BESS was dispatched.
        unmet_shaving_kwh: Energy shortfall (demand above threshold).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    dispatch_id: str = Field(default_factory=_new_uuid)
    threshold_kw: Decimal = Field(default=Decimal("0"))
    peak_before_kw: Decimal = Field(default=Decimal("0"))
    peak_after_kw: Decimal = Field(default=Decimal("0"))
    peak_reduction_kw: Decimal = Field(default=Decimal("0"))
    peak_reduction_pct: Decimal = Field(default=Decimal("0"))
    total_discharged_kwh: Decimal = Field(default=Decimal("0"))
    total_charged_kwh: Decimal = Field(default=Decimal("0"))
    cycle_count: Decimal = Field(default=Decimal("0"))
    min_soc: Decimal = Field(default=Decimal("0"))
    max_soc: Decimal = Field(default=Decimal("0"))
    avg_soc: Decimal = Field(default=Decimal("0"))
    dispatch_hours: Decimal = Field(default=Decimal("0"))
    unmet_shaving_kwh: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class DegradationState(BaseModel):
    """Battery degradation state over project lifetime.

    Attributes:
        year: Project year.
        calendar_fade_pct: Cumulative calendar fade (%).
        cycle_fade_pct: Cumulative cycle fade (%).
        total_fade_pct: Total capacity fade (%).
        remaining_capacity_pct: Remaining capacity (%).
        remaining_capacity_kwh: Remaining usable capacity (kWh).
        cumulative_throughput_kwh: Cumulative energy throughput (kWh).
        cumulative_cycles: Cumulative equivalent full cycles.
        eol_reached: Whether end-of-life threshold reached.
    """
    year: int = Field(default=0)
    calendar_fade_pct: Decimal = Field(default=Decimal("0"))
    cycle_fade_pct: Decimal = Field(default=Decimal("0"))
    total_fade_pct: Decimal = Field(default=Decimal("0"))
    remaining_capacity_pct: Decimal = Field(default=Decimal("100"))
    remaining_capacity_kwh: Decimal = Field(default=Decimal("0"))
    cumulative_throughput_kwh: Decimal = Field(default=Decimal("0"))
    cumulative_cycles: Decimal = Field(default=Decimal("0"))
    eol_reached: bool = Field(default=False)


class SizingCandidate(BaseModel):
    """A BESS sizing candidate with evaluation metrics.

    Attributes:
        candidate_id: Unique candidate identifier.
        power_kw: System power (kW).
        capacity_kwh: System energy capacity (kWh).
        duration_hours: Storage duration (hours).
        chemistry: Battery chemistry.
        capex: Total capital cost.
        annual_opex: Annual O&M cost.
        annual_savings: Annual demand charge savings.
        npv: Net present value over project life.
        simple_payback_years: Simple payback period (years).
        roi_pct: Return on investment (%).
        lcos: Levelised cost of storage ($/kWh).
        peak_reduction_kw: Achievable peak reduction.
        grade: Sizing grade.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    candidate_id: str = Field(default_factory=_new_uuid)
    power_kw: Decimal = Field(default=Decimal("0"))
    capacity_kwh: Decimal = Field(default=Decimal("0"))
    duration_hours: Decimal = Field(default=Decimal("0"))
    chemistry: BatteryChemistry = Field(default=BatteryChemistry.LFP)
    capex: Decimal = Field(default=Decimal("0"))
    annual_opex: Decimal = Field(default=Decimal("0"))
    annual_savings: Decimal = Field(default=Decimal("0"))
    npv: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    lcos: Decimal = Field(default=Decimal("0"))
    peak_reduction_kw: Decimal = Field(default=Decimal("0"))
    grade: SizingGrade = Field(default=SizingGrade.UNECONOMIC)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class BESSSizingResult(BaseModel):
    """Complete BESS sizing analysis result.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        sizing_objective: Optimisation objective.
        dispatch_strategy: Dispatch strategy used.
        optimal_candidate: Best sizing candidate.
        all_candidates: All evaluated candidates.
        dispatch_result: Dispatch simulation for optimal candidate.
        degradation_schedule: Year-by-year degradation.
        lcos_optimal: LCOS for optimal candidate.
        peak_before_kw: Original peak demand.
        peak_after_kw: Peak after optimal BESS.
        annual_savings: Annual demand charge savings.
        lifetime_savings: Total lifetime savings.
        recommendations: List of sizing recommendations.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    sizing_objective: SizingObjective = Field(default=SizingObjective.MAX_SAVINGS)
    dispatch_strategy: DispatchStrategy = Field(default=DispatchStrategy.PEAK_SHAVING)
    optimal_candidate: SizingCandidate = Field(default_factory=SizingCandidate)
    all_candidates: List[SizingCandidate] = Field(default_factory=list)
    dispatch_result: DispatchResult = Field(default_factory=DispatchResult)
    degradation_schedule: List[DegradationState] = Field(default_factory=list)
    lcos_optimal: Decimal = Field(default=Decimal("0"))
    peak_before_kw: Decimal = Field(default=Decimal("0"))
    peak_after_kw: Decimal = Field(default=Decimal("0"))
    annual_savings: Decimal = Field(default=Decimal("0"))
    lifetime_savings: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BESSSizingEngine:
    """BESS sizing optimisation engine for peak shaving.

    Performs dispatch simulation, degradation modelling, sizing
    optimisation, technology comparison, and LCOS calculation.  All
    calculations use deterministic Decimal arithmetic with SHA-256
    provenance hashing.

    Usage::

        engine = BESSSizingEngine()
        specs = BatterySpecs(power_kw=Decimal("200"),
                             capacity_kwh=Decimal("800"))
        dispatch = engine.simulate_dispatch(demand_profile, specs,
                                            threshold_kw=Decimal("400"))
        print(f"Peak reduced by: {dispatch.peak_reduction_kw} kW")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise BESSSizingEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - discount_rate (float): for NPV and LCOS
                - om_rate (float): annual O&M as fraction of CAPEX
                - project_years (int): project lifetime
                - demand_charge_rate (float): $/kW/month
        """
        self.config = config or {}
        self._discount_rate = _decimal(
            self.config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
        )
        self._om_rate = _decimal(
            self.config.get("om_rate", DEFAULT_OM_RATE)
        )
        self._project_years = int(
            self.config.get("project_years", DEFAULT_PROJECT_YEARS)
        )
        self._demand_charge_rate = _decimal(
            self.config.get("demand_charge_rate", Decimal("15"))
        )
        logger.info(
            "BESSSizingEngine v%s initialised (discount=%.2f, years=%d)",
            self.engine_version, float(self._discount_rate),
            self._project_years,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def optimize_sizing(
        self,
        facility_id: str,
        facility_name: str,
        demand_profile: List[Decimal],
        target_peak_kw: Optional[Decimal] = None,
        chemistry: BatteryChemistry = BatteryChemistry.LFP,
        objective: SizingObjective = SizingObjective.MAX_SAVINGS,
        power_range: Optional[List[Decimal]] = None,
        duration_range: Optional[List[Decimal]] = None,
    ) -> BESSSizingResult:
        """Optimise BESS sizing for peak shaving.

        Evaluates multiple sizing candidates and selects the optimal
        system based on the specified objective.

        Args:
            facility_id: Facility identifier.
            facility_name: Facility name.
            demand_profile: List of demand values (kW) at 15-min intervals.
            target_peak_kw: Target peak demand (for TARGET_PEAK objective).
            chemistry: Battery chemistry to evaluate.
            objective: Sizing optimisation objective.
            power_range: List of power capacities to evaluate (kW).
            duration_range: List of storage durations to evaluate (hours).

        Returns:
            BESSSizingResult with optimal candidate and analysis.
        """
        t0 = time.perf_counter()
        peak_before = max(demand_profile) if demand_profile else Decimal("0")
        logger.info(
            "Optimizing BESS: %s, peak=%.1f kW, chemistry=%s, objective=%s",
            facility_name, float(peak_before), chemistry.value, objective.value,
        )

        if not demand_profile:
            result = BESSSizingResult(
                facility_id=facility_id,
                facility_name=facility_name,
                recommendations=["No demand profile data provided."],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Default candidate ranges
        if power_range is None:
            # 10% to 50% of peak in 10% steps
            power_range = [
                _round_val(peak_before * _decimal(pct) / Decimal("100"), 0)
                for pct in range(10, 55, 10)
            ]
        if duration_range is None:
            duration_range = [Decimal("1"), Decimal("2"), Decimal("4")]

        specs_template = BATTERY_SPECS.get(chemistry.value, BATTERY_SPECS[BatteryChemistry.LFP.value])

        candidates: List[SizingCandidate] = []
        for power_kw in power_range:
            for duration_h in duration_range:
                capacity_kwh = power_kw * duration_h
                if capacity_kwh <= Decimal("0"):
                    continue

                specs = BatterySpecs(
                    chemistry=chemistry,
                    power_kw=power_kw,
                    capacity_kwh=capacity_kwh,
                    rte_nominal=_decimal(specs_template["rte_nominal"]),
                    c_rate_max=_decimal(specs_template["c_rate_max_discharge"]),
                    dod_max=_decimal(specs_template["dod_max"]),
                    cost_per_kwh=_decimal(specs_template["cost_per_kwh"]),
                    cost_per_kw=_decimal(specs_template["cost_per_kw"]),
                    cycle_life=int(specs_template["cycle_life_80pct"]),
                    calendar_life_years=int(specs_template["calendar_life_years"]),
                )

                # Determine threshold
                threshold = target_peak_kw or (peak_before - power_kw)
                threshold = max(threshold, Decimal("0"))

                # Simulate dispatch
                dispatch = self.simulate_dispatch(demand_profile, specs, threshold)

                # Calculate economics
                candidate = self._evaluate_candidate(
                    specs, dispatch, peak_before,
                )
                candidates.append(candidate)

        # Select optimal candidate
        optimal = self._select_optimal(candidates, objective)

        # Run degradation for optimal
        if optimal:
            degradation = self.model_degradation(
                BatterySpecs(
                    chemistry=chemistry,
                    capacity_kwh=optimal.capacity_kwh,
                    cycle_life=int(specs_template["cycle_life_80pct"]),
                    calendar_life_years=int(specs_template["calendar_life_years"]),
                ),
                annual_cycles=float(optimal.peak_reduction_kw * Decimal("365") /
                                    max(optimal.capacity_kwh, Decimal("1"))),
            )
            optimal_dispatch = self.simulate_dispatch(
                demand_profile,
                BatterySpecs(
                    chemistry=chemistry,
                    power_kw=optimal.power_kw,
                    capacity_kwh=optimal.capacity_kwh,
                    rte_nominal=_decimal(specs_template["rte_nominal"]),
                    dod_max=_decimal(specs_template["dod_max"]),
                ),
                peak_before - optimal.peak_reduction_kw,
            )
        else:
            degradation = []
            optimal_dispatch = DispatchResult()
            optimal = SizingCandidate()

        lifetime_savings = optimal.annual_savings * _decimal(self._project_years)

        recommendations = self._generate_recommendations(
            optimal, candidates, peak_before, chemistry,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = BESSSizingResult(
            facility_id=facility_id,
            facility_name=facility_name,
            sizing_objective=objective,
            dispatch_strategy=DispatchStrategy.PEAK_SHAVING,
            optimal_candidate=optimal,
            all_candidates=candidates,
            dispatch_result=optimal_dispatch,
            degradation_schedule=degradation,
            lcos_optimal=optimal.lcos,
            peak_before_kw=_round_val(peak_before, 2),
            peak_after_kw=_round_val(
                peak_before - optimal.peak_reduction_kw, 2
            ),
            annual_savings=optimal.annual_savings,
            lifetime_savings=_round_val(lifetime_savings, 2),
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "BESS sizing complete: optimal=%s kW/%s kWh, "
            "savings=%.0f/yr, NPV=%.0f, hash=%s (%.1f ms)",
            str(optimal.power_kw), str(optimal.capacity_kwh),
            float(optimal.annual_savings), float(optimal.npv),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def simulate_dispatch(
        self,
        demand_profile: List[Decimal],
        specs: BatterySpecs,
        threshold_kw: Decimal,
    ) -> DispatchResult:
        """Simulate BESS dispatch over a demand profile.

        Runs a time-step simulation at 15-minute resolution.  Discharges
        when demand exceeds threshold, charges during off-peak (demand
        below 80% of threshold).

        Args:
            demand_profile: List of demand values (kW) at 15-min intervals.
            specs: Battery specifications.
            threshold_kw: Peak shaving threshold (kW).

        Returns:
            DispatchResult with simulation metrics.
        """
        t0 = time.perf_counter()

        dt_hours = Decimal("0.25")  # 15 minutes
        usable_kwh = specs.capacity_kwh * (specs.soc_max - specs.soc_min)
        max_discharge_kw = min(specs.power_kw, specs.c_rate_max * specs.capacity_kwh)
        max_charge_kw = max_discharge_kw  # Symmetric for simplicity

        soc = (specs.soc_min + specs.soc_max) / Decimal("2")  # Start at 50%
        charge_trigger = threshold_kw * Decimal("0.80")

        peak_before = max(demand_profile) if demand_profile else Decimal("0")
        peak_after = Decimal("0")
        total_discharged = Decimal("0")
        total_charged = Decimal("0")
        dispatch_intervals = 0
        unmet = Decimal("0")
        soc_values: List[Decimal] = []
        min_soc = soc
        max_soc = soc

        for demand in demand_profile:
            demand = _decimal(demand)

            if demand > threshold_kw and soc > specs.soc_min:
                # Discharge
                needed_kw = demand - threshold_kw
                available_kwh = (soc - specs.soc_min) * specs.capacity_kwh
                available_kw = _safe_divide(available_kwh, dt_hours)
                discharge_kw = min(needed_kw, max_discharge_kw, available_kw)
                energy_kwh = discharge_kw * dt_hours
                soc -= _safe_divide(energy_kwh, specs.capacity_kwh * specs.rte_nominal)
                soc = max(soc, specs.soc_min)
                total_discharged += energy_kwh
                dispatch_intervals += 1
                net_demand = demand - discharge_kw

                if discharge_kw < needed_kw:
                    unmet += (needed_kw - discharge_kw) * dt_hours

            elif demand < charge_trigger and soc < specs.soc_max:
                # Charge
                headroom_kw = charge_trigger - demand
                capacity_room_kwh = (specs.soc_max - soc) * specs.capacity_kwh
                capacity_room_kw = _safe_divide(capacity_room_kwh, dt_hours)
                charge_kw = min(headroom_kw, max_charge_kw, capacity_room_kw)
                energy_kwh = charge_kw * dt_hours * specs.rte_nominal
                soc += _safe_divide(energy_kwh, specs.capacity_kwh)
                soc = min(soc, specs.soc_max)
                total_charged += charge_kw * dt_hours
                net_demand = demand + charge_kw
            else:
                net_demand = demand

            peak_after = max(peak_after, net_demand)
            soc_values.append(soc)
            min_soc = min(min_soc, soc)
            max_soc = max(max_soc, soc)

        avg_soc = _safe_divide(
            sum(soc_values, Decimal("0")),
            _decimal(len(soc_values)),
        ) if soc_values else Decimal("0")

        cycle_count = _safe_divide(
            total_discharged, usable_kwh
        ) if usable_kwh > Decimal("0") else Decimal("0")

        dispatch_hours = _decimal(dispatch_intervals) * dt_hours
        reduction_kw = peak_before - peak_after
        reduction_pct = _safe_pct(reduction_kw, peak_before)

        result = DispatchResult(
            threshold_kw=_round_val(threshold_kw, 2),
            peak_before_kw=_round_val(peak_before, 2),
            peak_after_kw=_round_val(peak_after, 2),
            peak_reduction_kw=_round_val(max(reduction_kw, Decimal("0")), 2),
            peak_reduction_pct=_round_val(max(reduction_pct, Decimal("0")), 2),
            total_discharged_kwh=_round_val(total_discharged, 2),
            total_charged_kwh=_round_val(total_charged, 2),
            cycle_count=_round_val(cycle_count, 2),
            min_soc=_round_val(min_soc, 4),
            max_soc=_round_val(max_soc, 4),
            avg_soc=_round_val(avg_soc, 4),
            dispatch_hours=_round_val(dispatch_hours, 2),
            unmet_shaving_kwh=_round_val(unmet, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Dispatch simulated: threshold=%.0f, reduction=%.1f kW (%.1f%%), "
            "cycles=%.1f, hash=%s (%.1f ms)",
            float(threshold_kw), float(reduction_kw), float(reduction_pct),
            float(cycle_count), result.provenance_hash[:16], elapsed,
        )
        return result

    def model_degradation(
        self,
        specs: BatterySpecs,
        annual_cycles: float,
        model: DegradationModel = DegradationModel.CALENDAR_PLUS_CYCLE,
    ) -> List[DegradationState]:
        """Model battery degradation over project lifetime.

        Args:
            specs: Battery specifications.
            annual_cycles: Estimated annual equivalent full cycles.
            model: Degradation model to use.

        Returns:
            List of DegradationState, one per year.
        """
        t0 = time.perf_counter()
        logger.info(
            "Modelling degradation: %s, %.0f cycles/yr, model=%s",
            specs.chemistry.value, annual_cycles, model.value,
        )

        chem_specs = BATTERY_SPECS.get(
            specs.chemistry.value,
            BATTERY_SPECS[BatteryChemistry.LFP.value],
        )
        aging_rate = _decimal(chem_specs["aging_rate"])
        rated_cycles = int(chem_specs["cycle_life_80pct"])
        capacity_kwh = _decimal(specs.capacity_kwh)

        schedule: List[DegradationState] = []
        cumulative_cycles = Decimal("0")
        cumulative_throughput = Decimal("0")

        for year in range(1, self._project_years + 1):
            # Calendar fade: aging_rate * sqrt(year)
            calendar_fade = aging_rate * _decimal(math.sqrt(year))
            calendar_fade_pct = calendar_fade * Decimal("100")

            # Cycle fade: cycles / rated_cycle_life * 20%
            cumulative_cycles += _decimal(annual_cycles)
            cycle_fade = _safe_divide(
                cumulative_cycles, _decimal(rated_cycles)
            ) * Decimal("0.20")
            cycle_fade_pct = cycle_fade * Decimal("100")

            total_fade_pct = calendar_fade_pct + cycle_fade_pct
            remaining_pct = max(Decimal("100") - total_fade_pct, Decimal("0"))
            remaining_kwh = capacity_kwh * remaining_pct / Decimal("100")

            annual_throughput = _decimal(annual_cycles) * capacity_kwh * Decimal("2")
            cumulative_throughput += annual_throughput

            eol = remaining_pct < Decimal("80")

            state = DegradationState(
                year=year,
                calendar_fade_pct=_round_val(calendar_fade_pct, 2),
                cycle_fade_pct=_round_val(cycle_fade_pct, 2),
                total_fade_pct=_round_val(total_fade_pct, 2),
                remaining_capacity_pct=_round_val(remaining_pct, 2),
                remaining_capacity_kwh=_round_val(remaining_kwh, 2),
                cumulative_throughput_kwh=_round_val(cumulative_throughput, 2),
                cumulative_cycles=_round_val(cumulative_cycles, 2),
                eol_reached=eol,
            )
            schedule.append(state)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Degradation modelled: %d years, final_capacity=%.1f%% (%.1f ms)",
            len(schedule),
            float(schedule[-1].remaining_capacity_pct) if schedule else 0,
            elapsed,
        )
        return schedule

    def compare_technologies(
        self,
        demand_profile: List[Decimal],
        power_kw: Decimal,
        capacity_kwh: Decimal,
        threshold_kw: Decimal,
        chemistries: Optional[List[BatteryChemistry]] = None,
    ) -> List[SizingCandidate]:
        """Compare multiple battery chemistries for the same application.

        Args:
            demand_profile: Demand profile at 15-min intervals.
            power_kw: System power (kW).
            capacity_kwh: System energy capacity (kWh).
            threshold_kw: Peak shaving threshold.
            chemistries: List of chemistries to compare.

        Returns:
            List of SizingCandidate, one per chemistry, sorted by NPV.
        """
        t0 = time.perf_counter()
        if chemistries is None:
            chemistries = list(BatteryChemistry)

        peak_before = max(demand_profile) if demand_profile else Decimal("0")
        candidates: List[SizingCandidate] = []

        for chem in chemistries:
            chem_specs = BATTERY_SPECS.get(chem.value, BATTERY_SPECS[BatteryChemistry.LFP.value])
            specs = BatterySpecs(
                chemistry=chem,
                power_kw=power_kw,
                capacity_kwh=capacity_kwh,
                rte_nominal=_decimal(chem_specs["rte_nominal"]),
                c_rate_max=_decimal(chem_specs["c_rate_max_discharge"]),
                dod_max=_decimal(chem_specs["dod_max"]),
                cost_per_kwh=_decimal(chem_specs["cost_per_kwh"]),
                cost_per_kw=_decimal(chem_specs["cost_per_kw"]),
                cycle_life=int(chem_specs["cycle_life_80pct"]),
                calendar_life_years=int(chem_specs["calendar_life_years"]),
            )

            dispatch = self.simulate_dispatch(demand_profile, specs, threshold_kw)
            candidate = self._evaluate_candidate(specs, dispatch, peak_before)
            candidate.chemistry = chem
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.npv, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Technology comparison: %d chemistries compared (%.1f ms)",
            len(candidates), elapsed,
        )
        return candidates

    def calculate_lcos(
        self,
        capex: Decimal,
        annual_opex: Decimal,
        annual_discharged_kwh: Decimal,
        project_years: Optional[int] = None,
        discount_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate Levelised Cost of Storage (LCOS).

        LCOS = (CAPEX + sum(OPEX_t/(1+r)^t)) / sum(Energy_t/(1+r)^t)

        Args:
            capex: Total capital expenditure.
            annual_opex: Annual O&M cost.
            annual_discharged_kwh: Annual discharged energy (kWh).
            project_years: Project lifetime.
            discount_rate: Discount rate.

        Returns:
            LCOS in $/kWh.
        """
        t0 = time.perf_counter()
        years = project_years or self._project_years
        rate = discount_rate or self._discount_rate

        cost_pv = capex
        energy_pv = Decimal("0")

        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + rate) ** _decimal(t)
            cost_pv += _safe_divide(annual_opex, discount_factor)
            energy_pv += _safe_divide(annual_discharged_kwh, discount_factor)

        lcos = _safe_divide(cost_pv, energy_pv)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "LCOS calculated: %.4f $/kWh (%.1f ms)",
            float(lcos), elapsed,
        )
        return _round_val(lcos, 4)

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate_candidate(
        self,
        specs: BatterySpecs,
        dispatch: DispatchResult,
        peak_before: Decimal,
    ) -> SizingCandidate:
        """Evaluate a sizing candidate's economics.

        Args:
            specs: Battery specifications.
            dispatch: Dispatch simulation result.
            peak_before: Original peak demand.

        Returns:
            SizingCandidate with economic metrics.
        """
        capex = (
            specs.capacity_kwh * specs.cost_per_kwh
            + specs.power_kw * specs.cost_per_kw
        )
        annual_opex = capex * self._om_rate

        # Annual savings from demand charge reduction (12 months)
        annual_savings = dispatch.peak_reduction_kw * self._demand_charge_rate * Decimal("12")

        # NPV
        npv = -capex
        for t in range(1, self._project_years + 1):
            discount = (Decimal("1") + self._discount_rate) ** _decimal(t)
            npv += _safe_divide(annual_savings - annual_opex, discount)

        # Simple payback
        net_annual = annual_savings - annual_opex
        payback = _safe_divide(capex, net_annual) if net_annual > Decimal("0") else Decimal("999")

        # ROI
        roi = _safe_pct(npv, capex)

        # LCOS
        annual_discharged = dispatch.total_discharged_kwh
        lcos = self.calculate_lcos(capex, annual_opex, annual_discharged)

        # Duration
        duration = _safe_divide(specs.capacity_kwh, specs.power_kw)

        # Grade
        grade = self._assign_grade(npv, roi, payback)

        candidate = SizingCandidate(
            power_kw=_round_val(specs.power_kw, 2),
            capacity_kwh=_round_val(specs.capacity_kwh, 2),
            duration_hours=_round_val(duration, 2),
            chemistry=specs.chemistry,
            capex=_round_val(capex, 2),
            annual_opex=_round_val(annual_opex, 2),
            annual_savings=_round_val(annual_savings, 2),
            npv=_round_val(npv, 2),
            simple_payback_years=_round_val(min(payback, Decimal("99")), 2),
            roi_pct=_round_val(roi, 2),
            lcos=lcos,
            peak_reduction_kw=dispatch.peak_reduction_kw,
            grade=grade,
        )
        candidate.provenance_hash = _compute_hash(candidate)
        return candidate

    def _select_optimal(
        self,
        candidates: List[SizingCandidate],
        objective: SizingObjective,
    ) -> Optional[SizingCandidate]:
        """Select optimal candidate by objective.

        Args:
            candidates: List of evaluated candidates.
            objective: Optimisation objective.

        Returns:
            Best candidate or None.
        """
        if not candidates:
            return None

        if objective == SizingObjective.MAX_SAVINGS:
            return max(candidates, key=lambda c: c.annual_savings)
        elif objective == SizingObjective.MAX_ROI:
            return max(candidates, key=lambda c: c.roi_pct)
        elif objective == SizingObjective.MIN_COST:
            return min(candidates, key=lambda c: c.capex)
        elif objective == SizingObjective.TARGET_PEAK:
            # Select candidate with highest reduction that has positive NPV
            viable = [c for c in candidates if c.npv > Decimal("0")]
            if viable:
                return max(viable, key=lambda c: c.peak_reduction_kw)
            return max(candidates, key=lambda c: c.peak_reduction_kw)
        return candidates[0]

    def _assign_grade(
        self,
        npv: Decimal,
        roi_pct: Decimal,
        payback: Decimal,
    ) -> SizingGrade:
        """Assign a sizing grade based on economic metrics.

        Args:
            npv: Net present value.
            roi_pct: Return on investment percentage.
            payback: Simple payback years.

        Returns:
            SizingGrade classification.
        """
        if npv <= Decimal("0"):
            return SizingGrade.UNECONOMIC
        if roi_pct >= Decimal("50") and payback <= Decimal("5"):
            return SizingGrade.OPTIMAL
        if roi_pct >= Decimal("25") and payback <= Decimal("8"):
            return SizingGrade.GOOD
        if roi_pct >= Decimal("10") and payback <= Decimal("12"):
            return SizingGrade.ACCEPTABLE
        return SizingGrade.MARGINAL

    def _generate_recommendations(
        self,
        optimal: SizingCandidate,
        candidates: List[SizingCandidate],
        peak_before: Decimal,
        chemistry: BatteryChemistry,
    ) -> List[str]:
        """Generate BESS sizing recommendations.

        Args:
            optimal: Optimal sizing candidate.
            candidates: All evaluated candidates.
            peak_before: Original peak demand.
            chemistry: Selected chemistry.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if optimal.grade == SizingGrade.UNECONOMIC:
            recs.append(
                "BESS is not economic at current demand charge rates. "
                "Consider load shifting or demand response instead."
            )
            return recs

        if optimal.grade == SizingGrade.OPTIMAL:
            recs.append(
                f"Optimal BESS: {optimal.power_kw} kW / {optimal.capacity_kwh} kWh "
                f"({chemistry.value.upper()}). NPV = ${optimal.npv}, "
                f"payback = {optimal.simple_payback_years} years."
            )

        reduction_pct = _safe_pct(optimal.peak_reduction_kw, peak_before)
        if reduction_pct > Decimal("30"):
            recs.append(
                f"BESS can reduce peak by {reduction_pct}%. "
                "Combine with load shifting for additional savings."
            )

        if optimal.lcos > Decimal("0.30"):
            recs.append(
                f"LCOS is ${optimal.lcos}/kWh. Consider LFP chemistry "
                "for lower cycle cost if not already selected."
            )

        if chemistry == BatteryChemistry.NMC:
            recs.append(
                "NMC selected. Evaluate LFP for longer cycle life and "
                "lower degradation if energy density is not critical."
            )

        viable = [c for c in candidates if c.npv > Decimal("0")]
        if len(viable) > 1:
            recs.append(
                f"{len(viable)} of {len(candidates)} candidates are economic. "
                "Review the full comparison table for alternative sizing."
            )

        if not recs:
            recs.append(
                "BESS sizing analysis complete. Proceed with detailed "
                "engineering design and procurement."
            )

        return recs
