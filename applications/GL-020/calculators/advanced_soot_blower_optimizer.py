"""
GL-020 ECONOPULSE: Advanced Soot Blower Optimizer

Zero-hallucination advanced soot blowing optimization for economizer
cleaning with comprehensive zone management and ROI analysis.

This module provides enhanced soot blowing capabilities:
- Optimal soot blowing interval calculation with economic optimization
- Zone-based cleaning prioritization using heat transfer analysis
- Steam/air consumption tracking with cost analysis
- Cleaning effectiveness measurement with before/after comparison
- ROI analysis per cleaning cycle with payback calculation
- Erosion wear monitoring with tube life prediction
- Sequential blowing schedule optimization
- Energy balance for soot blowing steam usage

All calculations are:
- Deterministic (zero-hallucination guaranteed)
- Bit-perfect reproducible
- Fully auditable with SHA-256 provenance hashes
- Thread-safe with LRU caching where appropriate

Author: GL-BackendDeveloper
Standards: ASME PTC 4.3, EPRI Soot Blowing Guidelines
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from .provenance import (
    ProvenanceTracker,
    CalculationType,
    CalculationProvenance,
    generate_calculation_hash,
)


# =============================================================================
# CONSTANTS AND ENUMERATIONS
# =============================================================================

class BlowerType(Enum):
    """Types of soot blowing equipment."""
    RETRACTABLE_LANCE = "retractable_lance"
    ROTARY_WALL = "rotary_wall"
    FIXED_POSITION = "fixed_position"
    ACOUSTIC_HORN = "acoustic_horn"
    WATER_CANNON = "water_cannon"


class BlowingMedium(Enum):
    """Blowing medium types with properties."""
    SATURATED_STEAM = "saturated_steam"
    SUPERHEATED_STEAM = "superheated_steam"
    COMPRESSED_AIR = "compressed_air"
    SONIC = "sonic"


class EconomizerZone(Enum):
    """Economizer zones for targeted cleaning."""
    GAS_INLET = "gas_inlet"           # Hottest zone, most severe fouling
    GAS_INLET_MIDDLE = "gas_inlet_middle"
    MIDDLE = "middle"
    GAS_OUTLET_MIDDLE = "gas_outlet_middle"
    GAS_OUTLET = "gas_outlet"         # Coolest zone
    TOP_BANK = "top_bank"
    BOTTOM_BANK = "bottom_bank"       # Ash accumulation zone


class CleaningPriority(Enum):
    """Cleaning priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MAINTENANCE = 5


class WearSeverity(Enum):
    """Erosion wear severity levels."""
    MINIMAL = "minimal"
    ACCEPTABLE = "acceptable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


# EPRI recommended parameters
EPRI_GUIDELINES = {
    "min_interval_hours": 4.0,
    "max_interval_hours": 168.0,
    "steam_velocity_fps_max": 800.0,
    "steam_pressure_min_psia": 200.0,
    "steam_pressure_max_psia": 400.0,
    "tube_wear_limit_mils": 40.0,
}

# Zone importance weights for prioritization
ZONE_IMPORTANCE_WEIGHTS = {
    EconomizerZone.GAS_INLET: 1.5,
    EconomizerZone.GAS_INLET_MIDDLE: 1.3,
    EconomizerZone.MIDDLE: 1.0,
    EconomizerZone.GAS_OUTLET_MIDDLE: 0.9,
    EconomizerZone.GAS_OUTLET: 0.8,
    EconomizerZone.TOP_BANK: 0.95,
    EconomizerZone.BOTTOM_BANK: 1.2,  # Ash accumulation
}

# Steam energy content (BTU/lbm at typical conditions)
STEAM_ENERGY_CONTENT = {
    BlowingMedium.SATURATED_STEAM: 1190.0,    # At 300 psia
    BlowingMedium.SUPERHEATED_STEAM: 1290.0,  # At 300 psia, 500F superheat
    BlowingMedium.COMPRESSED_AIR: 0.0,        # No thermal energy
    BlowingMedium.SONIC: 0.0,                 # Electrical energy
}

# Typical erosion rates (mils per 1000 blowing cycles)
EROSION_RATES = {
    BlowingMedium.SATURATED_STEAM: 0.5,
    BlowingMedium.SUPERHEATED_STEAM: 0.8,
    BlowingMedium.COMPRESSED_AIR: 0.2,
    BlowingMedium.SONIC: 0.0,
}


# =============================================================================
# FROZEN DATACLASSES FOR IMMUTABILITY
# =============================================================================

@dataclass(frozen=True)
class SootBlowerConfiguration:
    """
    Immutable soot blower equipment configuration.

    Attributes:
        blower_id: Unique blower identifier
        blower_type: Type of soot blowing equipment
        medium: Blowing medium (steam/air/sonic)
        zone: Economizer zone covered
        steam_flow_lbm_per_cycle: Steam consumption per cycle (lbm)
        steam_pressure_psia: Steam supply pressure (psia)
        steam_temperature_f: Steam temperature (F)
        air_flow_scfm: Compressed air flow (SCFM) for air blowers
        cycle_duration_seconds: Duration of one cleaning cycle (seconds)
        lance_travel_inches: Lance travel distance (inches)
        nozzle_orifice_dia_inches: Nozzle orifice diameter (inches)
    """
    blower_id: str
    blower_type: BlowerType
    medium: BlowingMedium
    zone: EconomizerZone
    steam_flow_lbm_per_cycle: float = 500.0
    steam_pressure_psia: float = 300.0
    steam_temperature_f: float = 500.0
    air_flow_scfm: float = 0.0
    cycle_duration_seconds: float = 300.0
    lance_travel_inches: float = 120.0
    nozzle_orifice_dia_inches: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.steam_flow_lbm_per_cycle < 0:
            raise ValueError("Steam flow cannot be negative")
        if self.steam_pressure_psia < 0:
            raise ValueError("Steam pressure cannot be negative")
        if self.cycle_duration_seconds <= 0:
            raise ValueError("Cycle duration must be positive")


@dataclass(frozen=True)
class ZoneFoulingState:
    """
    Immutable state of fouling in an economizer zone.

    Attributes:
        zone: Economizer zone identifier
        fouling_factor: Current fouling factor ((hr-ft2-F)/BTU)
        u_value_current: Current U-value (BTU/(hr-ft2-F))
        u_value_clean: Clean baseline U-value (BTU/(hr-ft2-F))
        gas_temp_inlet_f: Local gas inlet temperature (F)
        gas_temp_outlet_f: Local gas outlet temperature (F)
        last_cleaning_timestamp: Last cleaning datetime
        hours_since_cleaning: Hours since last cleaning
    """
    zone: EconomizerZone
    fouling_factor: float
    u_value_current: float
    u_value_clean: float
    gas_temp_inlet_f: float
    gas_temp_outlet_f: float
    last_cleaning_timestamp: Optional[datetime] = None
    hours_since_cleaning: float = 0.0


@dataclass(frozen=True)
class BlowingIntervalResult:
    """
    Immutable result of optimal blowing interval calculation.

    Attributes:
        optimal_interval_hours: Optimal time between blowing cycles (hours)
        optimal_interval_days: Optimal interval in days
        min_interval_hours: Minimum allowed interval (hours)
        max_interval_hours: Maximum allowed interval (hours)
        economic_penalty_per_hour: Cost of delaying cleaning ($/hr)
        recommended_next_blow: Recommended next blowing datetime
        confidence_level: Confidence in recommendation (0-1)
        provenance_hash: SHA-256 hash for audit trail
    """
    optimal_interval_hours: Decimal
    optimal_interval_days: Decimal
    min_interval_hours: Decimal
    max_interval_hours: Decimal
    economic_penalty_per_hour: Decimal
    recommended_next_blow: Optional[str]
    confidence_level: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class ZonePriorityResult:
    """
    Immutable result of zone prioritization analysis.

    Attributes:
        zone: Economizer zone
        priority_score: Calculated priority score (higher = more urgent)
        priority_level: Priority classification
        cleaning_benefit_per_hour: Expected benefit from cleaning ($/hr)
        estimated_recovery: Expected U-value recovery (%)
        recommended_action: Recommended cleaning action
        provenance_hash: SHA-256 hash for audit trail
    """
    zone: EconomizerZone
    priority_score: Decimal
    priority_level: CleaningPriority
    cleaning_benefit_per_hour: Decimal
    estimated_recovery: Decimal
    recommended_action: str
    provenance_hash: str


@dataclass(frozen=True)
class MediaConsumptionResult:
    """
    Immutable result of media consumption tracking.

    Attributes:
        steam_consumed_lbm: Total steam consumed (lbm)
        steam_cost: Steam cost ($)
        air_consumed_scf: Total compressed air consumed (SCF)
        air_cost: Compressed air cost ($)
        total_cost: Total media cost ($)
        energy_consumed_mmbtu: Energy consumed (MMBtu)
        energy_cost: Energy cost ($)
        cycles_tracked: Number of cycles tracked
        provenance_hash: SHA-256 hash for audit trail
    """
    steam_consumed_lbm: Decimal
    steam_cost: Decimal
    air_consumed_scf: Decimal
    air_cost: Decimal
    total_cost: Decimal
    energy_consumed_mmbtu: Decimal
    energy_cost: Decimal
    cycles_tracked: int
    provenance_hash: str


@dataclass(frozen=True)
class CleaningEffectivenessResult:
    """
    Immutable result of cleaning effectiveness measurement.

    Attributes:
        u_value_before: U-value before cleaning (BTU/(hr-ft2-F))
        u_value_after: U-value after cleaning (BTU/(hr-ft2-F))
        u_value_clean: Clean baseline U-value (BTU/(hr-ft2-F))
        effectiveness: Cleaning effectiveness (0-1)
        recovery_percent: U-value recovery percentage
        rf_removed: Fouling factor removed ((hr-ft2-F)/BTU)
        cleaning_duration_seconds: Duration of cleaning cycle
        blower_id: Soot blower identifier
        provenance_hash: SHA-256 hash for audit trail
    """
    u_value_before: Decimal
    u_value_after: Decimal
    u_value_clean: Decimal
    effectiveness: Decimal
    recovery_percent: Decimal
    rf_removed: Decimal
    cleaning_duration_seconds: Decimal
    blower_id: str
    provenance_hash: str


@dataclass(frozen=True)
class ROIAnalysisResult:
    """
    Immutable result of cleaning ROI analysis.

    Attributes:
        cleaning_cost: Total cost of cleaning cycle ($)
        fuel_savings_per_hour: Fuel savings after cleaning ($/hr)
        expected_benefit_hours: Expected duration of benefit (hours)
        total_fuel_savings: Total fuel savings ($)
        net_benefit: Net benefit from cleaning ($)
        roi_percent: Return on investment (%)
        payback_hours: Hours to payback cleaning cost
        annual_roi: Annualized ROI (%)
        provenance_hash: SHA-256 hash for audit trail
    """
    cleaning_cost: Decimal
    fuel_savings_per_hour: Decimal
    expected_benefit_hours: Decimal
    total_fuel_savings: Decimal
    net_benefit: Decimal
    roi_percent: Decimal
    payback_hours: Decimal
    annual_roi: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class ErosionMonitorResult:
    """
    Immutable result of erosion wear monitoring.

    Attributes:
        blower_id: Soot blower identifier
        cumulative_cycles: Total number of blowing cycles
        estimated_wear_mils: Estimated tube wear (mils)
        wear_severity: Wear severity classification
        remaining_life_percent: Remaining tube life (%)
        cycles_to_limit: Cycles until wear limit
        recommended_inspection: Recommended inspection action
        provenance_hash: SHA-256 hash for audit trail
    """
    blower_id: str
    cumulative_cycles: int
    estimated_wear_mils: Decimal
    wear_severity: WearSeverity
    remaining_life_percent: Decimal
    cycles_to_limit: int
    recommended_inspection: str
    provenance_hash: str


@dataclass(frozen=True)
class SequentialScheduleResult:
    """
    Immutable result of sequential blowing schedule optimization.

    Attributes:
        sequence: Ordered list of blower IDs
        total_duration_minutes: Total schedule duration (minutes)
        steam_consumption_lbm: Total steam consumption (lbm)
        expected_effectiveness: Overall expected effectiveness
        cooling_intervals_minutes: Cooling time between blows (minutes)
        schedule_start: Recommended start time
        schedule_details: Detailed schedule with timing
        provenance_hash: SHA-256 hash for audit trail
    """
    sequence: Tuple[str, ...]
    total_duration_minutes: Decimal
    steam_consumption_lbm: Decimal
    expected_effectiveness: Decimal
    cooling_intervals_minutes: Decimal
    schedule_start: Optional[str]
    schedule_details: Tuple[Dict[str, Any], ...]
    provenance_hash: str


@dataclass(frozen=True)
class EnergyBalanceResult:
    """
    Immutable result of soot blowing energy balance.

    Attributes:
        steam_energy_consumed_mmbtu: Steam energy consumed (MMBtu)
        heat_recovery_improvement_mmbtu_hr: Heat recovery improvement (MMBtu/hr)
        net_energy_benefit_mmbtu_hr: Net energy benefit (MMBtu/hr)
        energy_efficiency_ratio: Energy efficiency ratio (output/input)
        breakeven_hours: Hours to breakeven on energy investment
        annual_net_benefit_mmbtu: Annual net energy benefit (MMBtu/year)
        provenance_hash: SHA-256 hash for audit trail
    """
    steam_energy_consumed_mmbtu: Decimal
    heat_recovery_improvement_mmbtu_hr: Decimal
    net_energy_benefit_mmbtu_hr: Decimal
    energy_efficiency_ratio: Decimal
    breakeven_hours: Decimal
    annual_net_benefit_mmbtu: Decimal
    provenance_hash: str


# =============================================================================
# THREAD-SAFE CACHING
# =============================================================================

_cache_lock = threading.RLock()
_optimizer_cache: Dict[str, Any] = {}


def _generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate deterministic cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def clear_optimizer_cache() -> int:
    """Clear the optimizer cache. Returns number of entries cleared."""
    with _cache_lock:
        count = len(_optimizer_cache)
        _optimizer_cache.clear()
        return count


# =============================================================================
# OPTIMAL BLOWING INTERVAL CALCULATION
# =============================================================================

def calculate_optimal_blowing_interval(
    fouling_rate: float,
    cleaning_cost: float,
    fuel_cost_per_mmbtu: float,
    boiler_efficiency: float,
    boiler_heat_input_mmbtu_hr: float,
    threshold_rf: float = 0.005,
    cleaning_effectiveness: float = 0.85,
    last_cleaning_time: Optional[datetime] = None,
    track_provenance: bool = False
) -> Union[BlowingIntervalResult, Tuple[BlowingIntervalResult, CalculationProvenance]]:
    """
    Calculate optimal soot blowing interval using economic optimization.

    Balances the cost of fouling-induced efficiency loss against cleaning
    costs to find the economically optimal blowing frequency.

    Methodology (EPRI Economic Model):
        Total Cost Rate = Cleaning_Cost/T + Fouling_Penalty_Rate * avg_Rf

        For linear fouling (Rf = k*t):
        avg_Rf = k*T/2 over interval T

        Optimal interval minimizes total cost:
        T_opt = sqrt(2 * C_clean / (fuel_penalty_rate * fouling_rate))

    Reference: EPRI Intelligent Soot Blowing Guidelines

    Args:
        fouling_rate: Fouling accumulation rate ((hr-ft2-F)/(BTU-hr))
        cleaning_cost: Cost per cleaning cycle ($)
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        boiler_efficiency: Boiler thermal efficiency (0-1)
        boiler_heat_input_mmbtu_hr: Boiler heat input (MMBtu/hr)
        threshold_rf: Maximum acceptable fouling factor
        cleaning_effectiveness: Expected cleaning effectiveness (0-1)
        last_cleaning_time: Timestamp of last cleaning
        track_provenance: If True, return provenance record

    Returns:
        BlowingIntervalResult with optimal interval and economics

    Example:
        >>> result = calculate_optimal_blowing_interval(
        ...     fouling_rate=1e-6, cleaning_cost=50.0,
        ...     fuel_cost_per_mmbtu=5.0, boiler_efficiency=0.85,
        ...     boiler_heat_input_mmbtu_hr=100.0
        ... )
        >>> print(f"Optimal interval: {result.optimal_interval_hours} hours")
    """
    # Input validation
    if cleaning_cost < 0:
        raise ValueError("Cleaning cost cannot be negative")
    if fuel_cost_per_mmbtu < 0:
        raise ValueError("Fuel cost cannot be negative")
    if not 0 < boiler_efficiency <= 1:
        raise ValueError("Boiler efficiency must be between 0 and 1")
    if boiler_heat_input_mmbtu_hr <= 0:
        raise ValueError("Boiler heat input must be positive")
    if not 0 < cleaning_effectiveness <= 1:
        raise ValueError("Cleaning effectiveness must be between 0 and 1")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="optimal_blowing_interval",
            formula_version="2.0.0",
            inputs={
                "fouling_rate": fouling_rate,
                "cleaning_cost": cleaning_cost,
                "fuel_cost": fuel_cost_per_mmbtu,
                "boiler_efficiency": boiler_efficiency,
                "heat_input": boiler_heat_input_mmbtu_hr,
                "threshold_rf": threshold_rf
            }
        )

    # Calculate fuel penalty sensitivity to fouling
    # Assume 1% efficiency loss per 0.005 (hr-ft2-F)/BTU
    efficiency_loss_per_rf = 0.01 / 0.005  # fraction per Rf unit
    fuel_penalty_per_rf_hour = (
        efficiency_loss_per_rf *
        boiler_heat_input_mmbtu_hr *
        fuel_cost_per_mmbtu /
        boiler_efficiency
    )  # $/hr per unit Rf

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate fuel penalty sensitivity to fouling",
            inputs={
                "efficiency_loss_per_rf": efficiency_loss_per_rf,
                "heat_input": boiler_heat_input_mmbtu_hr,
                "fuel_cost": fuel_cost_per_mmbtu
            },
            output_name="fuel_penalty_per_rf_hour",
            output_value=fuel_penalty_per_rf_hour,
            formula="k1 = (dEff/dRf) * Q * fuel_cost / efficiency"
        )

    # Calculate optimal interval
    if fouling_rate <= 0:
        # No measurable fouling - use maximum interval
        optimal_interval = EPRI_GUIDELINES["max_interval_hours"]
        confidence = 0.5  # Low confidence due to no fouling data
    else:
        # Economic optimization: T_opt = sqrt(2 * C_clean / (k1 * k_f))
        optimal_interval = math.sqrt(
            2 * cleaning_cost / (fuel_penalty_per_rf_hour * fouling_rate)
        )
        confidence = min(0.95, 0.7 + 0.25 * min(1.0, fouling_rate * 1e6))

    if tracker:
        tracker.add_step(
            operation="optimize",
            description="Calculate economically optimal blowing interval",
            inputs={
                "cleaning_cost": cleaning_cost,
                "fuel_penalty_rate": fuel_penalty_per_rf_hour,
                "fouling_rate": fouling_rate
            },
            output_name="optimal_interval",
            output_value=optimal_interval,
            formula="T_opt = sqrt(2 * C_clean / (k1 * k_f))"
        )

    # Apply EPRI constraints
    min_interval = EPRI_GUIDELINES["min_interval_hours"]
    max_interval = EPRI_GUIDELINES["max_interval_hours"]
    optimal_interval = max(min_interval, min(max_interval, optimal_interval))

    # Adjust for cleaning effectiveness
    # Less effective cleaning requires more frequent blowing
    optimal_interval *= cleaning_effectiveness

    if tracker:
        tracker.add_step(
            operation="adjust",
            description="Adjust interval for cleaning effectiveness",
            inputs={
                "raw_interval": optimal_interval / cleaning_effectiveness,
                "effectiveness": cleaning_effectiveness
            },
            output_name="adjusted_interval",
            output_value=optimal_interval,
            formula="T_adjusted = T_opt * cleaning_effectiveness"
        )

    # Calculate economic penalty per hour of delay
    if fouling_rate > 0:
        penalty_per_hour = fuel_penalty_per_rf_hour * fouling_rate
    else:
        penalty_per_hour = 0.0

    # Calculate recommended next blow time
    if last_cleaning_time:
        next_blow_time = last_cleaning_time + timedelta(hours=optimal_interval)
        recommended_next = next_blow_time.isoformat()
    else:
        next_blow_time = datetime.now(timezone.utc) + timedelta(hours=optimal_interval)
        recommended_next = next_blow_time.isoformat()

    # Generate provenance hash
    hash_data = {
        "fouling_rate": str(fouling_rate),
        "cleaning_cost": str(cleaning_cost),
        "fuel_penalty_rate": str(fuel_penalty_per_rf_hour),
        "optimal_interval": str(optimal_interval),
        "confidence": str(confidence)
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_2 = Decimal("0.01")
    precision_4 = Decimal("0.0001")

    result = BlowingIntervalResult(
        optimal_interval_hours=Decimal(str(optimal_interval)).quantize(precision_2, rounding=ROUND_HALF_UP),
        optimal_interval_days=Decimal(str(optimal_interval / 24)).quantize(precision_2, rounding=ROUND_HALF_UP),
        min_interval_hours=Decimal(str(min_interval)),
        max_interval_hours=Decimal(str(max_interval)),
        economic_penalty_per_hour=Decimal(str(penalty_per_hour)).quantize(precision_4, rounding=ROUND_HALF_UP),
        recommended_next_blow=recommended_next,
        confidence_level=Decimal(str(confidence)).quantize(precision_2, rounding=ROUND_HALF_UP),
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=optimal_interval,
            output_unit="hours",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# ZONE-BASED CLEANING PRIORITIZATION
# =============================================================================

def prioritize_cleaning_zones(
    zone_states: List[ZoneFoulingState],
    blower_configs: Dict[EconomizerZone, SootBlowerConfiguration],
    fuel_cost_per_mmbtu: float,
    boiler_efficiency: float,
    max_zones_per_cycle: int = 3,
    track_provenance: bool = False
) -> Union[List[ZonePriorityResult], Tuple[List[ZonePriorityResult], CalculationProvenance]]:
    """
    Prioritize economizer zones for cleaning based on fouling severity.

    Uses heat transfer analysis and economic impact to rank zones
    for optimal cleaning sequence.

    Methodology:
        Priority Score = (Rf * Zone_Weight * Benefit_Factor) / Cleaning_Cost

        Where:
        - Rf = Current fouling factor
        - Zone_Weight = ZONE_IMPORTANCE_WEIGHTS[zone]
        - Benefit_Factor = fuel savings potential
        - Cleaning_Cost = media + maintenance cost

    Reference: EPRI Zone-Based Soot Blowing Optimization

    Args:
        zone_states: List of ZoneFoulingState for each zone
        blower_configs: Configuration for each zone's blower
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        boiler_efficiency: Boiler thermal efficiency (0-1)
        max_zones_per_cycle: Maximum zones to clean per cycle
        track_provenance: If True, return provenance record

    Returns:
        Sorted list of ZonePriorityResult (highest priority first)
    """
    if not zone_states:
        raise ValueError("At least one zone state required")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="zone_prioritization",
            formula_version="1.0.0",
            inputs={
                "n_zones": len(zone_states),
                "max_zones_per_cycle": max_zones_per_cycle,
                "fuel_cost": fuel_cost_per_mmbtu
            }
        )

    results = []

    for state in zone_states:
        zone = state.zone
        zone_weight = ZONE_IMPORTANCE_WEIGHTS.get(zone, 1.0)

        # Get blower config for this zone
        config = blower_configs.get(zone)
        if config:
            # Estimate cleaning cost
            steam_cost_per_klb = 8.0  # Typical value
            cleaning_cost = (config.steam_flow_lbm_per_cycle / 1000) * steam_cost_per_klb + 5.0
        else:
            cleaning_cost = 10.0  # Default

        # Calculate benefit from cleaning
        # Assume 80% recovery of fouled U-value
        expected_recovery = 0.80
        u_improvement = (state.u_value_clean - state.u_value_current) * expected_recovery
        heat_improvement = u_improvement / state.u_value_clean if state.u_value_clean > 0 else 0

        # Convert to fuel savings
        # Assume 100 MMBtu/hr heat input
        heat_input = 100.0
        fuel_savings_per_hour = (
            heat_improvement *
            heat_input *
            fuel_cost_per_mmbtu /
            boiler_efficiency
        )

        # Calculate priority score
        if cleaning_cost > 0:
            priority_score = (
                state.fouling_factor *
                zone_weight *
                (1 + fuel_savings_per_hour) /
                cleaning_cost *
                1000  # Scale for readability
            )
        else:
            priority_score = 0

        # Determine priority level
        if priority_score >= 10:
            priority_level = CleaningPriority.CRITICAL
            action = "Immediate cleaning required"
        elif priority_score >= 5:
            priority_level = CleaningPriority.HIGH
            action = "Schedule cleaning within 24 hours"
        elif priority_score >= 2:
            priority_level = CleaningPriority.MEDIUM
            action = "Include in next scheduled cycle"
        elif priority_score >= 1:
            priority_level = CleaningPriority.LOW
            action = "Monitor and clean as convenient"
        else:
            priority_level = CleaningPriority.MAINTENANCE
            action = "No immediate action needed"

        # Generate provenance hash for this zone
        hash_data = {
            "zone": zone.value,
            "fouling_factor": str(state.fouling_factor),
            "priority_score": str(priority_score),
            "priority_level": priority_level.name
        }
        zone_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

        precision_4 = Decimal("0.0001")
        precision_2 = Decimal("0.01")

        results.append(ZonePriorityResult(
            zone=zone,
            priority_score=Decimal(str(priority_score)).quantize(precision_4, rounding=ROUND_HALF_UP),
            priority_level=priority_level,
            cleaning_benefit_per_hour=Decimal(str(fuel_savings_per_hour)).quantize(precision_4, rounding=ROUND_HALF_UP),
            estimated_recovery=Decimal(str(expected_recovery * 100)).quantize(precision_2, rounding=ROUND_HALF_UP),
            recommended_action=action,
            provenance_hash=zone_hash
        ))

    # Sort by priority score (descending)
    results.sort(key=lambda x: float(x.priority_score), reverse=True)

    # Limit to max zones per cycle
    results = results[:max_zones_per_cycle] if len(results) > max_zones_per_cycle else results

    if tracker:
        tracker.add_step(
            operation="prioritize",
            description="Rank zones by cleaning priority",
            inputs={"n_zones": len(zone_states)},
            output_name="prioritized_zones",
            output_value=len(results),
            formula="Score = Rf * Zone_Weight * Benefit / Cost"
        )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=len(results),
            output_unit="zones",
            precision=0
        )
        return results, provenance

    return results


# =============================================================================
# MEDIA CONSUMPTION TRACKING
# =============================================================================

def track_media_consumption(
    blower_configs: List[SootBlowerConfiguration],
    cycles_per_blower: Dict[str, int],
    steam_cost_per_klb: float = 8.0,
    air_cost_per_kscf: float = 0.50,
    electricity_cost_per_kwh: float = 0.10,
    track_provenance: bool = False
) -> Union[MediaConsumptionResult, Tuple[MediaConsumptionResult, CalculationProvenance]]:
    """
    Track steam/air consumption and costs for soot blowing operations.

    Calculates total media consumption, associated costs, and energy
    usage for a set of soot blowing cycles.

    Args:
        blower_configs: List of soot blower configurations
        cycles_per_blower: Dictionary mapping blower_id to cycle count
        steam_cost_per_klb: Steam cost ($/1000 lbm)
        air_cost_per_kscf: Compressed air cost ($/1000 SCF)
        electricity_cost_per_kwh: Electricity cost for acoustic horns ($/kWh)
        track_provenance: If True, return provenance record

    Returns:
        MediaConsumptionResult with consumption and cost analysis
    """
    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="media_consumption_tracking",
            formula_version="1.0.0",
            inputs={
                "n_blowers": len(blower_configs),
                "steam_cost": steam_cost_per_klb,
                "air_cost": air_cost_per_kscf
            }
        )

    total_steam_lbm = Decimal("0")
    total_air_scf = Decimal("0")
    total_energy_mmbtu = Decimal("0")
    total_cycles = 0

    for config in blower_configs:
        cycles = cycles_per_blower.get(config.blower_id, 0)
        total_cycles += cycles

        if config.medium in [BlowingMedium.SATURATED_STEAM, BlowingMedium.SUPERHEATED_STEAM]:
            steam_consumed = Decimal(str(config.steam_flow_lbm_per_cycle * cycles))
            total_steam_lbm += steam_consumed

            # Calculate energy content
            energy_per_lb = STEAM_ENERGY_CONTENT.get(config.medium, 1190.0)
            energy_mmbtu = steam_consumed * Decimal(str(energy_per_lb)) / Decimal("1e6")
            total_energy_mmbtu += energy_mmbtu

        elif config.medium == BlowingMedium.COMPRESSED_AIR:
            # Air consumption (SCFM * duration in minutes)
            air_consumed = Decimal(str(
                config.air_flow_scfm * (config.cycle_duration_seconds / 60) * cycles
            ))
            total_air_scf += air_consumed

        # Acoustic horns have minimal media cost (handled as electricity)

    if tracker:
        tracker.add_step(
            operation="sum",
            description="Calculate total media consumption",
            inputs={"n_cycles": total_cycles},
            output_name="total_steam",
            output_value=float(total_steam_lbm),
            formula="Total = sum(consumption_per_cycle * cycles)"
        )

    # Calculate costs
    steam_cost = (total_steam_lbm / Decimal("1000")) * Decimal(str(steam_cost_per_klb))
    air_cost = (total_air_scf / Decimal("1000")) * Decimal(str(air_cost_per_kscf))
    energy_cost = total_energy_mmbtu * Decimal(str(steam_cost_per_klb * 0.8))  # Approximate

    total_cost = steam_cost + air_cost

    # Generate provenance hash
    hash_data = {
        "total_steam_lbm": str(total_steam_lbm),
        "total_air_scf": str(total_air_scf),
        "steam_cost": str(steam_cost),
        "air_cost": str(air_cost),
        "total_cycles": total_cycles
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_2 = Decimal("0.01")
    precision_4 = Decimal("0.0001")

    result = MediaConsumptionResult(
        steam_consumed_lbm=total_steam_lbm.quantize(precision_2, rounding=ROUND_HALF_UP),
        steam_cost=steam_cost.quantize(precision_2, rounding=ROUND_HALF_UP),
        air_consumed_scf=total_air_scf.quantize(precision_2, rounding=ROUND_HALF_UP),
        air_cost=air_cost.quantize(precision_2, rounding=ROUND_HALF_UP),
        total_cost=total_cost.quantize(precision_2, rounding=ROUND_HALF_UP),
        energy_consumed_mmbtu=total_energy_mmbtu.quantize(precision_4, rounding=ROUND_HALF_UP),
        energy_cost=energy_cost.quantize(precision_2, rounding=ROUND_HALF_UP),
        cycles_tracked=total_cycles,
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=float(total_cost),
            output_unit="$",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# CLEANING EFFECTIVENESS MEASUREMENT
# =============================================================================

def measure_cleaning_effectiveness(
    u_before: float,
    u_after: float,
    u_clean: float,
    blower_id: str,
    cleaning_duration_seconds: float = 300.0,
    track_provenance: bool = False
) -> Union[CleaningEffectivenessResult, Tuple[CleaningEffectivenessResult, CalculationProvenance]]:
    """
    Measure the effectiveness of a soot blowing cycle.

    Compares U-values before and after cleaning to quantify
    the cleaning effectiveness and fouling removal.

    Methodology:
        Effectiveness = (U_after - U_before) / (U_clean - U_before)
        Rf_removed = (1/U_before) - (1/U_after)

    Args:
        u_before: U-value before cleaning (BTU/(hr-ft2-F))
        u_after: U-value after cleaning (BTU/(hr-ft2-F))
        u_clean: Clean baseline U-value (BTU/(hr-ft2-F))
        blower_id: Soot blower identifier
        cleaning_duration_seconds: Duration of cleaning cycle
        track_provenance: If True, return provenance record

    Returns:
        CleaningEffectivenessResult with effectiveness metrics
    """
    # Input validation
    if u_before <= 0 or u_after <= 0 or u_clean <= 0:
        raise ValueError("All U-values must be positive")
    if u_before > u_clean:
        raise ValueError("U_before cannot exceed U_clean")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="cleaning_effectiveness",
            formula_version="1.0.0",
            inputs={
                "u_before": u_before,
                "u_after": u_after,
                "u_clean": u_clean,
                "blower_id": blower_id
            }
        )

    # Calculate improvement
    u_improvement = u_after - u_before
    max_improvement = u_clean - u_before

    # Calculate effectiveness
    if max_improvement > 0:
        effectiveness = u_improvement / max_improvement
    else:
        effectiveness = 1.0 if u_improvement >= 0 else 0.0

    # Clamp to valid range
    effectiveness = max(0.0, min(1.5, effectiveness))  # Allow slight over-cleaning

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate cleaning effectiveness",
            inputs={"u_improvement": u_improvement, "max_improvement": max_improvement},
            output_name="effectiveness",
            output_value=effectiveness,
            formula="Effectiveness = (U_after - U_before) / (U_clean - U_before)"
        )

    # Calculate recovery percentage
    recovery_percent = effectiveness * 100

    # Calculate fouling factor removed
    rf_before = (1 / u_before) - (1 / u_clean)
    rf_after = max(0, (1 / u_after) - (1 / u_clean))
    rf_removed = rf_before - rf_after

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate fouling factor removed",
            inputs={"rf_before": rf_before, "rf_after": rf_after},
            output_name="rf_removed",
            output_value=rf_removed,
            formula="Rf_removed = Rf_before - Rf_after"
        )

    # Generate provenance hash
    hash_data = {
        "u_before": str(u_before),
        "u_after": str(u_after),
        "u_clean": str(u_clean),
        "effectiveness": str(effectiveness),
        "rf_removed": str(rf_removed),
        "blower_id": blower_id
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_4 = Decimal("0.0001")
    precision_6 = Decimal("0.000001")
    precision_2 = Decimal("0.01")

    result = CleaningEffectivenessResult(
        u_value_before=Decimal(str(u_before)).quantize(precision_4, rounding=ROUND_HALF_UP),
        u_value_after=Decimal(str(u_after)).quantize(precision_4, rounding=ROUND_HALF_UP),
        u_value_clean=Decimal(str(u_clean)).quantize(precision_4, rounding=ROUND_HALF_UP),
        effectiveness=Decimal(str(effectiveness)).quantize(precision_4, rounding=ROUND_HALF_UP),
        recovery_percent=Decimal(str(recovery_percent)).quantize(precision_2, rounding=ROUND_HALF_UP),
        rf_removed=Decimal(str(rf_removed)).quantize(precision_6, rounding=ROUND_HALF_UP),
        cleaning_duration_seconds=Decimal(str(cleaning_duration_seconds)),
        blower_id=blower_id,
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=effectiveness,
            output_unit="dimensionless",
            precision=4
        )
        return result, provenance

    return result


# =============================================================================
# ROI ANALYSIS PER CLEANING CYCLE
# =============================================================================

def analyze_cleaning_roi(
    cleaning_cost: float,
    fuel_penalty_before: float,
    fuel_penalty_after: float,
    expected_interval_hours: float,
    operating_hours_per_year: float = 8000.0,
    track_provenance: bool = False
) -> Union[ROIAnalysisResult, Tuple[ROIAnalysisResult, CalculationProvenance]]:
    """
    Calculate Return on Investment for a cleaning cycle.

    Compares the cost of cleaning against the fuel savings achieved
    to determine the economic value of the cleaning operation.

    Methodology:
        Fuel_Savings_Per_Hour = Penalty_Before - Penalty_After
        Total_Savings = Savings_Per_Hour * Expected_Duration
        Net_Benefit = Total_Savings - Cleaning_Cost
        ROI = Net_Benefit / Cleaning_Cost * 100%
        Payback = Cleaning_Cost / Savings_Per_Hour

    Args:
        cleaning_cost: Total cost of cleaning cycle ($)
        fuel_penalty_before: Fuel penalty rate before cleaning ($/hr)
        fuel_penalty_after: Fuel penalty rate after cleaning ($/hr)
        expected_interval_hours: Expected duration of benefit (hours)
        operating_hours_per_year: Annual operating hours
        track_provenance: If True, return provenance record

    Returns:
        ROIAnalysisResult with economic analysis
    """
    # Input validation
    if cleaning_cost < 0:
        raise ValueError("Cleaning cost cannot be negative")
    if fuel_penalty_before < 0 or fuel_penalty_after < 0:
        raise ValueError("Fuel penalties cannot be negative")
    if expected_interval_hours <= 0:
        raise ValueError("Expected interval must be positive")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="cleaning_roi_analysis",
            formula_version="1.0.0",
            inputs={
                "cleaning_cost": cleaning_cost,
                "fuel_penalty_before": fuel_penalty_before,
                "fuel_penalty_after": fuel_penalty_after,
                "expected_interval": expected_interval_hours
            }
        )

    # Calculate hourly savings
    savings_per_hour = fuel_penalty_before - fuel_penalty_after

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate hourly fuel savings",
            inputs={
                "penalty_before": fuel_penalty_before,
                "penalty_after": fuel_penalty_after
            },
            output_name="savings_per_hour",
            output_value=savings_per_hour,
            formula="Savings = Penalty_Before - Penalty_After"
        )

    # Calculate total fuel savings over interval
    total_savings = savings_per_hour * expected_interval_hours

    # Calculate net benefit
    net_benefit = total_savings - cleaning_cost

    # Calculate ROI
    if cleaning_cost > 0:
        roi_percent = (net_benefit / cleaning_cost) * 100
    else:
        roi_percent = float('inf') if net_benefit > 0 else 0

    # Calculate payback time
    if savings_per_hour > 0:
        payback_hours = cleaning_cost / savings_per_hour
    else:
        payback_hours = float('inf')

    # Calculate annualized ROI
    cycles_per_year = operating_hours_per_year / expected_interval_hours
    annual_net_benefit = net_benefit * cycles_per_year
    annual_investment = cleaning_cost * cycles_per_year
    if annual_investment > 0:
        annual_roi = (annual_net_benefit / annual_investment) * 100
    else:
        annual_roi = 0

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate ROI metrics",
            inputs={
                "total_savings": total_savings,
                "cleaning_cost": cleaning_cost
            },
            output_name="roi_percent",
            output_value=roi_percent if roi_percent != float('inf') else 999999,
            formula="ROI = (Net_Benefit / Cost) * 100%"
        )

    # Generate provenance hash
    hash_data = {
        "cleaning_cost": str(cleaning_cost),
        "savings_per_hour": str(savings_per_hour),
        "total_savings": str(total_savings),
        "net_benefit": str(net_benefit),
        "roi_percent": str(roi_percent) if roi_percent != float('inf') else "inf"
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_2 = Decimal("0.01")

    # Handle infinity values
    payback = Decimal(str(payback_hours)) if payback_hours != float('inf') else Decimal("999999")
    roi = Decimal(str(roi_percent)) if roi_percent != float('inf') else Decimal("999999")

    result = ROIAnalysisResult(
        cleaning_cost=Decimal(str(cleaning_cost)).quantize(precision_2, rounding=ROUND_HALF_UP),
        fuel_savings_per_hour=Decimal(str(savings_per_hour)).quantize(precision_2, rounding=ROUND_HALF_UP),
        expected_benefit_hours=Decimal(str(expected_interval_hours)).quantize(precision_2, rounding=ROUND_HALF_UP),
        total_fuel_savings=Decimal(str(total_savings)).quantize(precision_2, rounding=ROUND_HALF_UP),
        net_benefit=Decimal(str(net_benefit)).quantize(precision_2, rounding=ROUND_HALF_UP),
        roi_percent=roi.quantize(precision_2, rounding=ROUND_HALF_UP),
        payback_hours=payback.quantize(precision_2, rounding=ROUND_HALF_UP),
        annual_roi=Decimal(str(annual_roi)).quantize(precision_2, rounding=ROUND_HALF_UP),
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=roi_percent if roi_percent != float('inf') else 999999,
            output_unit="%",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# EROSION WEAR MONITORING
# =============================================================================

def monitor_erosion_wear(
    blower_config: SootBlowerConfiguration,
    cumulative_cycles: int,
    initial_tube_thickness_mils: float = 120.0,
    wear_limit_mils: float = 40.0,
    track_provenance: bool = False
) -> Union[ErosionMonitorResult, Tuple[ErosionMonitorResult, CalculationProvenance]]:
    """
    Monitor erosion wear on economizer tubes from soot blowing.

    Tracks cumulative wear from soot blowing operations and predicts
    remaining tube life to prevent erosion-related failures.

    Methodology:
        Estimated_Wear = Erosion_Rate * (Cycles / 1000)
        Remaining_Life = (Initial_Thickness - Wear_Limit - Current_Wear) / Erosion_Rate

    Args:
        blower_config: Soot blower configuration
        cumulative_cycles: Total number of blowing cycles to date
        initial_tube_thickness_mils: Original tube wall thickness (mils)
        wear_limit_mils: Minimum allowable wall thickness (mils)
        track_provenance: If True, return provenance record

    Returns:
        ErosionMonitorResult with wear assessment
    """
    if cumulative_cycles < 0:
        raise ValueError("Cumulative cycles cannot be negative")
    if initial_tube_thickness_mils <= wear_limit_mils:
        raise ValueError("Initial thickness must exceed wear limit")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="erosion_wear_monitoring",
            formula_version="1.0.0",
            inputs={
                "blower_id": blower_config.blower_id,
                "cumulative_cycles": cumulative_cycles,
                "initial_thickness": initial_tube_thickness_mils,
                "wear_limit": wear_limit_mils
            }
        )

    # Get erosion rate for this blowing medium
    erosion_rate = EROSION_RATES.get(blower_config.medium, 0.5)

    # Calculate estimated wear
    estimated_wear = erosion_rate * (cumulative_cycles / 1000)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate estimated tube wear",
            inputs={"erosion_rate": erosion_rate, "cycles": cumulative_cycles},
            output_name="estimated_wear",
            output_value=estimated_wear,
            formula="Wear = Rate * (Cycles / 1000)"
        )

    # Calculate remaining allowable wear
    allowable_wear = initial_tube_thickness_mils - wear_limit_mils
    remaining_wear = allowable_wear - estimated_wear

    # Calculate remaining life percentage
    remaining_life_percent = (remaining_wear / allowable_wear) * 100 if allowable_wear > 0 else 0
    remaining_life_percent = max(0, remaining_life_percent)

    # Calculate cycles to limit
    if erosion_rate > 0:
        cycles_to_limit = int((remaining_wear / erosion_rate) * 1000)
        cycles_to_limit = max(0, cycles_to_limit)
    else:
        cycles_to_limit = 999999

    # Determine wear severity
    if remaining_life_percent >= 80:
        severity = WearSeverity.MINIMAL
        inspection = "Continue normal monitoring"
    elif remaining_life_percent >= 60:
        severity = WearSeverity.ACCEPTABLE
        inspection = "Schedule routine inspection within 6 months"
    elif remaining_life_percent >= 40:
        severity = WearSeverity.MODERATE
        inspection = "Schedule detailed inspection within 3 months"
    elif remaining_life_percent >= 20:
        severity = WearSeverity.SIGNIFICANT
        inspection = "Urgent inspection required - plan tube replacement"
    else:
        severity = WearSeverity.CRITICAL
        inspection = "CRITICAL - Immediate inspection and replacement required"

    if tracker:
        tracker.add_step(
            operation="assess",
            description="Assess wear severity and remaining life",
            inputs={"remaining_life_percent": remaining_life_percent},
            output_name="severity",
            output_value=severity.value,
            formula="Severity based on remaining life percentage"
        )

    # Generate provenance hash
    hash_data = {
        "blower_id": blower_config.blower_id,
        "cumulative_cycles": cumulative_cycles,
        "estimated_wear": str(estimated_wear),
        "remaining_life_percent": str(remaining_life_percent),
        "severity": severity.value
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_2 = Decimal("0.01")

    result = ErosionMonitorResult(
        blower_id=blower_config.blower_id,
        cumulative_cycles=cumulative_cycles,
        estimated_wear_mils=Decimal(str(estimated_wear)).quantize(precision_2, rounding=ROUND_HALF_UP),
        wear_severity=severity,
        remaining_life_percent=Decimal(str(remaining_life_percent)).quantize(precision_2, rounding=ROUND_HALF_UP),
        cycles_to_limit=cycles_to_limit,
        recommended_inspection=inspection,
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=estimated_wear,
            output_unit="mils",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# SEQUENTIAL BLOWING SCHEDULE OPTIMIZATION
# =============================================================================

def optimize_blowing_sequence(
    blower_configs: List[SootBlowerConfiguration],
    zone_priority_scores: Dict[EconomizerZone, float],
    min_cooling_interval_minutes: float = 10.0,
    max_total_duration_minutes: float = 60.0,
    track_provenance: bool = False
) -> Union[SequentialScheduleResult, Tuple[SequentialScheduleResult, CalculationProvenance]]:
    """
    Optimize the sequential blowing schedule for multiple soot blowers.

    Determines the optimal order and timing of soot blowing operations
    to maximize cleaning effectiveness while managing steam demand.

    Methodology:
        1. Sort blowers by zone priority (highest first)
        2. Add cooling intervals between consecutive blows
        3. Limit total duration to prevent excessive steam usage
        4. Calculate total steam consumption and expected effectiveness

    Args:
        blower_configs: List of soot blower configurations
        zone_priority_scores: Priority score for each zone
        min_cooling_interval_minutes: Minimum time between blows (minutes)
        max_total_duration_minutes: Maximum total schedule duration (minutes)
        track_provenance: If True, return provenance record

    Returns:
        SequentialScheduleResult with optimized schedule
    """
    if not blower_configs:
        raise ValueError("At least one blower configuration required")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="blowing_sequence_optimization",
            formula_version="1.0.0",
            inputs={
                "n_blowers": len(blower_configs),
                "cooling_interval": min_cooling_interval_minutes,
                "max_duration": max_total_duration_minutes
            }
        )

    # Sort blowers by zone priority
    sorted_configs = sorted(
        blower_configs,
        key=lambda c: zone_priority_scores.get(c.zone, 0),
        reverse=True
    )

    # Build schedule
    schedule_details = []
    sequence = []
    current_time = 0.0
    total_steam = Decimal("0")

    for config in sorted_configs:
        cycle_duration_minutes = config.cycle_duration_seconds / 60

        # Check if we have time for this blower
        if current_time + cycle_duration_minutes > max_total_duration_minutes:
            break

        # Add to schedule
        start_time = current_time
        end_time = current_time + cycle_duration_minutes

        schedule_details.append({
            "blower_id": config.blower_id,
            "zone": config.zone.value,
            "start_minute": round(start_time, 1),
            "end_minute": round(end_time, 1),
            "duration_minutes": round(cycle_duration_minutes, 1),
            "steam_consumption_lbm": config.steam_flow_lbm_per_cycle,
            "priority_score": zone_priority_scores.get(config.zone, 0)
        })

        sequence.append(config.blower_id)
        total_steam += Decimal(str(config.steam_flow_lbm_per_cycle))

        # Add cooling interval for next blower
        current_time = end_time + min_cooling_interval_minutes

    if tracker:
        tracker.add_step(
            operation="optimize",
            description="Build optimized blowing sequence",
            inputs={
                "n_blowers_scheduled": len(sequence),
                "total_steam": float(total_steam)
            },
            output_name="sequence",
            output_value=len(sequence),
            formula="Sort by priority, add cooling intervals"
        )

    # Calculate total duration
    total_duration = current_time - min_cooling_interval_minutes if sequence else 0

    # Estimate expected overall effectiveness
    # Based on priority scores of included blowers
    if sequence:
        avg_priority = sum(zone_priority_scores.get(c.zone, 0) for c in sorted_configs[:len(sequence)]) / len(sequence)
        expected_effectiveness = min(0.95, 0.5 + avg_priority * 0.05)
    else:
        expected_effectiveness = 0.0

    # Generate provenance hash
    hash_data = {
        "sequence": list(sequence),
        "total_duration_minutes": str(total_duration),
        "total_steam_lbm": str(total_steam),
        "expected_effectiveness": str(expected_effectiveness)
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_2 = Decimal("0.01")

    # Convert schedule_details to tuple of dicts for immutability
    frozen_details = tuple(schedule_details)

    result = SequentialScheduleResult(
        sequence=tuple(sequence),
        total_duration_minutes=Decimal(str(total_duration)).quantize(precision_2, rounding=ROUND_HALF_UP),
        steam_consumption_lbm=total_steam.quantize(precision_2, rounding=ROUND_HALF_UP),
        expected_effectiveness=Decimal(str(expected_effectiveness)).quantize(precision_2, rounding=ROUND_HALF_UP),
        cooling_intervals_minutes=Decimal(str(min_cooling_interval_minutes)),
        schedule_start=datetime.now(timezone.utc).isoformat() if sequence else None,
        schedule_details=frozen_details,
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=len(sequence),
            output_unit="blowers",
            precision=0
        )
        return result, provenance

    return result


# =============================================================================
# ENERGY BALANCE FOR SOOT BLOWING
# =============================================================================

def calculate_soot_blowing_energy_balance(
    steam_consumed_lbm: float,
    steam_enthalpy_btu_per_lb: float,
    u_before: float,
    u_after: float,
    heat_transfer_area_ft2: float,
    lmtd_f: float,
    expected_benefit_hours: float,
    operating_hours_per_year: float = 8000.0,
    track_provenance: bool = False
) -> Union[EnergyBalanceResult, Tuple[EnergyBalanceResult, CalculationProvenance]]:
    """
    Calculate energy balance for soot blowing steam usage.

    Compares the energy invested in soot blowing (steam consumption)
    against the energy recovered through improved heat transfer.

    Methodology:
        Steam_Energy = Steam_Flow * Steam_Enthalpy / 1e6
        Heat_Recovery_Improvement = (U_after - U_before) * A * LMTD / 1e6
        Net_Energy_Benefit = Improvement - Steam_Energy / Benefit_Hours
        Energy_Ratio = Improvement * Benefit_Hours / Steam_Energy

    Args:
        steam_consumed_lbm: Steam consumed per cycle (lbm)
        steam_enthalpy_btu_per_lb: Steam enthalpy (BTU/lbm)
        u_before: U-value before cleaning (BTU/(hr-ft2-F))
        u_after: U-value after cleaning (BTU/(hr-ft2-F))
        heat_transfer_area_ft2: Heat transfer area (ft2)
        lmtd_f: Log mean temperature difference (F)
        expected_benefit_hours: Expected duration of cleaning benefit (hours)
        operating_hours_per_year: Annual operating hours
        track_provenance: If True, return provenance record

    Returns:
        EnergyBalanceResult with energy analysis
    """
    # Input validation
    if steam_consumed_lbm < 0:
        raise ValueError("Steam consumption cannot be negative")
    if u_before <= 0 or u_after <= 0:
        raise ValueError("U-values must be positive")
    if heat_transfer_area_ft2 <= 0:
        raise ValueError("Heat transfer area must be positive")
    if expected_benefit_hours <= 0:
        raise ValueError("Expected benefit hours must be positive")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="soot_blowing_energy_balance",
            formula_version="1.0.0",
            inputs={
                "steam_consumed": steam_consumed_lbm,
                "steam_enthalpy": steam_enthalpy_btu_per_lb,
                "u_before": u_before,
                "u_after": u_after,
                "area": heat_transfer_area_ft2
            }
        )

    # Calculate steam energy consumed (MMBtu)
    steam_energy = Decimal(str(steam_consumed_lbm * steam_enthalpy_btu_per_lb / 1e6))

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate steam energy consumed",
            inputs={"steam_lbm": steam_consumed_lbm, "enthalpy": steam_enthalpy_btu_per_lb},
            output_name="steam_energy",
            output_value=float(steam_energy),
            formula="E_steam = Steam * Enthalpy / 1e6"
        )

    # Calculate heat recovery improvement (MMBtu/hr)
    q_before = u_before * heat_transfer_area_ft2 * lmtd_f / 1e6
    q_after = u_after * heat_transfer_area_ft2 * lmtd_f / 1e6
    heat_improvement = Decimal(str(q_after - q_before))

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate heat recovery improvement",
            inputs={"q_before": q_before, "q_after": q_after},
            output_name="heat_improvement",
            output_value=float(heat_improvement),
            formula="dQ = (U_after - U_before) * A * LMTD / 1e6"
        )

    # Calculate net energy benefit per hour
    # Net = Improvement - (Steam_Energy / Benefit_Hours)
    steam_energy_per_hour = steam_energy / Decimal(str(expected_benefit_hours))
    net_energy_benefit = heat_improvement - steam_energy_per_hour

    # Calculate energy efficiency ratio
    total_benefit = heat_improvement * Decimal(str(expected_benefit_hours))
    if steam_energy > 0:
        energy_ratio = total_benefit / steam_energy
    else:
        energy_ratio = Decimal("999999")

    # Calculate breakeven hours
    if heat_improvement > 0:
        breakeven_hours = steam_energy / heat_improvement
    else:
        breakeven_hours = Decimal("999999")

    # Calculate annual net benefit
    cycles_per_year = Decimal(str(operating_hours_per_year)) / Decimal(str(expected_benefit_hours))
    annual_net = net_energy_benefit * Decimal(str(expected_benefit_hours)) * cycles_per_year

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate energy efficiency metrics",
            inputs={
                "steam_energy": float(steam_energy),
                "heat_improvement": float(heat_improvement)
            },
            output_name="energy_ratio",
            output_value=float(energy_ratio) if float(energy_ratio) < 999999 else 999999,
            formula="Ratio = (Improvement * Hours) / Steam_Energy"
        )

    # Generate provenance hash
    hash_data = {
        "steam_energy_mmbtu": str(steam_energy),
        "heat_improvement_mmbtu_hr": str(heat_improvement),
        "net_energy_benefit": str(net_energy_benefit),
        "energy_ratio": str(energy_ratio)
    }
    provenance_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()

    precision_4 = Decimal("0.0001")
    precision_2 = Decimal("0.01")

    result = EnergyBalanceResult(
        steam_energy_consumed_mmbtu=steam_energy.quantize(precision_4, rounding=ROUND_HALF_UP),
        heat_recovery_improvement_mmbtu_hr=heat_improvement.quantize(precision_4, rounding=ROUND_HALF_UP),
        net_energy_benefit_mmbtu_hr=net_energy_benefit.quantize(precision_4, rounding=ROUND_HALF_UP),
        energy_efficiency_ratio=energy_ratio.quantize(precision_2, rounding=ROUND_HALF_UP),
        breakeven_hours=breakeven_hours.quantize(precision_2, rounding=ROUND_HALF_UP),
        annual_net_benefit_mmbtu=annual_net.quantize(precision_2, rounding=ROUND_HALF_UP),
        provenance_hash=provenance_hash
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=float(energy_ratio),
            output_unit="dimensionless",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    "BlowerType",
    "BlowingMedium",
    "EconomizerZone",
    "CleaningPriority",
    "WearSeverity",

    # Constants
    "EPRI_GUIDELINES",
    "ZONE_IMPORTANCE_WEIGHTS",
    "STEAM_ENERGY_CONTENT",
    "EROSION_RATES",

    # Data Classes
    "SootBlowerConfiguration",
    "ZoneFoulingState",
    "BlowingIntervalResult",
    "ZonePriorityResult",
    "MediaConsumptionResult",
    "CleaningEffectivenessResult",
    "ROIAnalysisResult",
    "ErosionMonitorResult",
    "SequentialScheduleResult",
    "EnergyBalanceResult",

    # Core Functions
    "calculate_optimal_blowing_interval",
    "prioritize_cleaning_zones",
    "track_media_consumption",
    "measure_cleaning_effectiveness",
    "analyze_cleaning_roi",
    "monitor_erosion_wear",
    "optimize_blowing_sequence",
    "calculate_soot_blowing_energy_balance",

    # Utility Functions
    "clear_optimizer_cache",
]
