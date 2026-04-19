"""
GL-020 ECONOPULSE: Soot Blower Optimizer

Zero-hallucination soot blowing optimization calculations for economizer
cleaning management based on EPRI and ASME standards.

This module provides:
- Optimal cleaning interval calculation
- Cleaning effectiveness assessment
- Media cost calculation (steam/air consumption)
- Soot blower sequence optimization
- Cleaning ROI analysis

All calculations are deterministic with complete provenance tracking.

Author: GL-CalculatorEngineer
Standards: ASME PTC 4.3, EPRI Soot Blowing Guidelines
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .provenance import (
    ProvenanceTracker,
    CalculationType,
    CalculationProvenance,
    generate_calculation_hash
)
from .thermal_properties import ValidationError
from .fouling_calculator import (
    calculate_fouling_factor,
    calculate_cleanliness_factor,
    estimate_fuel_penalty,
    FoulingDataPoint
)


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class SootBlowerType(Enum):
    """Types of soot blowing equipment."""
    RETRACTABLE = "retractable"      # Long retractable lance
    ROTARY = "rotary"                # Rotating element
    WALL_BLOWER = "wall_blower"      # Wall-mounted
    AIR_PREHEATER = "air_preheater"  # Specialized for air heater
    ACOUSTIC_HORN = "acoustic_horn"   # Sound wave cleaning


class BlowingMedium(Enum):
    """Blowing medium types."""
    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    SONIC = "sonic"


class EconomizerZone(Enum):
    """Economizer zones for targeted cleaning."""
    INLET = "inlet"          # Gas inlet (hottest, most fouling)
    MIDDLE = "middle"        # Middle section
    OUTLET = "outlet"        # Gas outlet (coldest)
    TOP = "top"              # Top tubes
    BOTTOM = "bottom"        # Bottom tubes


# Default soot blower parameters
DEFAULT_SOOT_BLOWER_PARAMS = {
    "steam_flow_per_cycle_lbm": 500,     # lbm of steam per cleaning cycle
    "steam_pressure_psia": 300,          # Steam supply pressure
    "steam_temperature_f": 500,          # Steam temperature
    "cycle_duration_minutes": 5,         # Duration of cleaning cycle
    "steam_cost_per_klb": 8.0,           # $/1000 lbm of steam
    "labor_cost_per_cycle": 0,           # Automated - no labor cost
    "maintenance_cost_per_cycle": 5.0,   # Average wear/maintenance
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SootBlowerConfig:
    """Configuration for soot blower equipment."""
    blower_id: str
    blower_type: SootBlowerType
    medium: BlowingMedium
    zones_covered: List[EconomizerZone]

    # Media consumption
    steam_flow_per_cycle_lbm: float = 500.0    # lbm per cycle
    air_flow_per_cycle_scfm: float = 0.0       # SCFM for air blowers
    steam_pressure_psia: float = 300.0
    steam_temperature_f: float = 500.0

    # Timing
    cycle_duration_minutes: float = 5.0
    minimum_interval_hours: float = 4.0        # Minimum time between cycles
    maximum_interval_hours: float = 168.0      # Maximum time (1 week)

    # Costs
    steam_cost_per_klb: float = 8.0            # $/1000 lbm
    air_cost_per_kscf: float = 0.50            # $/1000 SCF
    maintenance_cost_per_cycle: float = 5.0


@dataclass
class CleaningEvent:
    """Record of a cleaning event."""
    timestamp: datetime
    blower_id: str
    zone: EconomizerZone
    media_consumed_lbm: float
    duration_minutes: float
    U_before: float                # U-value before cleaning
    U_after: float                 # U-value after cleaning
    effectiveness: float = 0.0     # Calculated effectiveness


@dataclass
class OptimizationResult:
    """Result of cleaning optimization analysis."""
    optimal_interval_hours: float
    recommended_next_cleaning: datetime
    zones_priority: List[EconomizerZone]
    expected_fuel_savings: float           # $/cycle
    estimated_cleaning_cost: float         # $/cycle
    net_benefit_per_cycle: float           # $
    annual_net_benefit: float              # $/year
    provenance_hash: str = ""


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_positive(value: float, param_name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be positive"
        )


def validate_non_negative(value: float, param_name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be non-negative"
        )


# =============================================================================
# OPTIMAL CLEANING INTERVAL
# =============================================================================

def calculate_optimal_cleaning_interval(
    fouling_rate: float,
    fuel_cost_per_hour_at_threshold: float,
    cleaning_cost_per_cycle: float,
    threshold_fouling_factor: float = 0.005,
    current_fouling_factor: float = 0.0,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate the optimal time interval between cleaning cycles.

    This optimization balances the cost of fouling (increased fuel consumption)
    against the cost of cleaning (media, maintenance, production loss).

    Methodology:
        The total cost function over time interval T is:

        Total_Cost = Cleaning_Cost + Integral(Fuel_Penalty * dt) from 0 to T

        For linear fouling:
        Fuel_Penalty(t) = k1 * Rf(t) = k1 * (Rf0 + fouling_rate * t)

        Optimal interval minimizes cost per unit time:
        d/dT[Total_Cost/T] = 0

        Solution:
        T_opt = sqrt(2 * Cleaning_Cost / (k1 * fouling_rate))

        Where k1 = fuel_cost_per_hour_at_threshold / threshold_fouling_factor

    Reference: EPRI Soot Blowing Optimization Guidelines

    Args:
        fouling_rate: Rate of fouling accumulation ((hr-ft2-F)/(BTU-hr))
        fuel_cost_per_hour_at_threshold: Fuel penalty at threshold Rf ($/hr)
        cleaning_cost_per_cycle: Total cost per cleaning cycle ($)
        threshold_fouling_factor: Reference Rf for fuel cost ((hr-ft2-F)/BTU)
        current_fouling_factor: Current Rf ((hr-ft2-F)/BTU)
        track_provenance: If True, return provenance record

    Returns:
        Optimal cleaning interval in hours, optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(fuel_cost_per_hour_at_threshold, "fuel_cost_per_hour_at_threshold")
    validate_positive(cleaning_cost_per_cycle, "cleaning_cost_per_cycle")
    validate_positive(threshold_fouling_factor, "threshold_fouling_factor")
    validate_non_negative(current_fouling_factor, "current_fouling_factor")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="optimal_cleaning_interval",
            formula_version="1.0.0",
            inputs={
                "fouling_rate": fouling_rate,
                "fuel_cost_per_hour_at_threshold": fuel_cost_per_hour_at_threshold,
                "cleaning_cost_per_cycle": cleaning_cost_per_cycle,
                "threshold_fouling_factor": threshold_fouling_factor,
                "current_fouling_factor": current_fouling_factor
            }
        )

    # Handle zero or negative fouling rate
    if fouling_rate <= 0:
        # No fouling - return maximum interval
        T_opt = 168.0  # 1 week default maximum

        if tracker:
            tracker.add_step(
                operation="check",
                description="Fouling rate is zero or negative - use maximum interval",
                inputs={"fouling_rate": fouling_rate},
                output_name="T_opt",
                output_value=T_opt,
                formula="T_opt = T_max (no fouling detected)"
            )
    else:
        # Calculate fuel cost sensitivity
        k1 = fuel_cost_per_hour_at_threshold / threshold_fouling_factor

        if tracker:
            tracker.add_step(
                operation="divide",
                description="Calculate fuel cost sensitivity coefficient",
                inputs={
                    "fuel_cost": fuel_cost_per_hour_at_threshold,
                    "Rf_threshold": threshold_fouling_factor
                },
                output_name="k1",
                output_value=k1,
                formula="k1 = fuel_cost / Rf_threshold"
            )

        # Calculate optimal interval using economic optimization
        # T_opt = sqrt(2 * C_clean / (k1 * dRf/dt))
        T_opt = math.sqrt(2 * cleaning_cost_per_cycle / (k1 * fouling_rate))

        if tracker:
            tracker.add_step(
                operation="optimize",
                description="Calculate optimal cleaning interval",
                inputs={
                    "cleaning_cost": cleaning_cost_per_cycle,
                    "k1": k1,
                    "fouling_rate": fouling_rate
                },
                output_name="T_opt",
                output_value=T_opt,
                formula="T_opt = sqrt(2 * C_clean / (k1 * dRf/dt))"
            )

    # Apply practical constraints
    T_opt = max(4.0, min(168.0, T_opt))  # Between 4 hours and 1 week

    if tracker:
        tracker.add_step(
            operation="constrain",
            description="Apply practical interval constraints",
            inputs={"T_opt_unconstrained": T_opt},
            output_name="T_opt_constrained",
            output_value=T_opt,
            formula="T_opt = max(4, min(168, T_opt))"
        )

    T_opt_rounded = round(T_opt, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=T_opt_rounded,
            output_unit="hours",
            precision=2
        )
        return T_opt_rounded, provenance

    return T_opt_rounded


def calculate_dynamic_cleaning_interval(
    fouling_history: List[FoulingDataPoint],
    cleaning_events: List[CleaningEvent],
    fuel_cost_per_mmbtu: float = 5.0,
    boiler_heat_input_mmbtu_hr: float = 100.0,
    cleaning_cost_per_cycle: float = 50.0,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate dynamic optimal cleaning interval based on actual performance data.

    This uses historical fouling and cleaning data to adapt the cleaning
    schedule to actual operating conditions.

    Methodology:
        1. Calculate average fouling rate from history
        2. Calculate average cleaning effectiveness from events
        3. Estimate fuel penalty function
        4. Optimize interval considering actual recovery

    Reference: EPRI Adaptive Soot Blowing Guidelines

    Args:
        fouling_history: List of fouling measurements
        cleaning_events: List of past cleaning events
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        boiler_heat_input_mmbtu_hr: Boiler heat input (MMBtu/hr)
        cleaning_cost_per_cycle: Cost per cleaning cycle ($)
        track_provenance: If True, return provenance record

    Returns:
        Optimal cleaning interval in hours, optionally with provenance

    Raises:
        ValueError: If insufficient data
    """
    if len(fouling_history) < 2:
        raise ValueError("At least 2 fouling data points required")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="dynamic_cleaning_interval",
            formula_version="1.0.0",
            inputs={
                "n_fouling_points": len(fouling_history),
                "n_cleaning_events": len(cleaning_events),
                "fuel_cost": fuel_cost_per_mmbtu,
                "heat_input": boiler_heat_input_mmbtu_hr,
                "cleaning_cost": cleaning_cost_per_cycle
            }
        )

    # Calculate average fouling rate from history
    sorted_history = sorted(fouling_history, key=lambda x: x.timestamp)
    total_hours = (sorted_history[-1].timestamp - sorted_history[0].timestamp).total_seconds() / 3600
    total_fouling = sorted_history[-1].fouling_factor - sorted_history[0].fouling_factor

    if total_hours > 0:
        avg_fouling_rate = total_fouling / total_hours
    else:
        avg_fouling_rate = 0.0

    if tracker:
        tracker.add_step(
            operation="average",
            description="Calculate average fouling rate from history",
            inputs={"total_fouling": total_fouling, "total_hours": total_hours},
            output_name="avg_fouling_rate",
            output_value=avg_fouling_rate,
            formula="avg_rate = (Rf_end - Rf_start) / (t_end - t_start)"
        )

    # Calculate average cleaning effectiveness
    if cleaning_events:
        effectiveness_values = [e.effectiveness for e in cleaning_events if e.effectiveness > 0]
        if effectiveness_values:
            avg_effectiveness = sum(effectiveness_values) / len(effectiveness_values)
        else:
            avg_effectiveness = 0.8  # Default assumption
    else:
        avg_effectiveness = 0.8

    if tracker:
        tracker.add_step(
            operation="average",
            description="Calculate average cleaning effectiveness",
            inputs={"n_events": len(cleaning_events)},
            output_name="avg_effectiveness",
            output_value=avg_effectiveness,
            formula="avg_effectiveness = mean(effectiveness_i)"
        )

    # Estimate fuel penalty at threshold
    # Assume 1% efficiency loss per 0.005 (hr-ft2-F)/BTU of fouling
    efficiency_loss_per_rf = 1.0 / 0.005  # %/(hr-ft2-F)/BTU
    threshold_rf = 0.005
    efficiency_loss_at_threshold = efficiency_loss_per_rf * threshold_rf

    fuel_penalty_at_threshold = (efficiency_loss_at_threshold / 100) * boiler_heat_input_mmbtu_hr * fuel_cost_per_mmbtu

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate fuel penalty at threshold fouling",
            inputs={
                "efficiency_loss": efficiency_loss_at_threshold,
                "heat_input": boiler_heat_input_mmbtu_hr,
                "fuel_cost": fuel_cost_per_mmbtu
            },
            output_name="fuel_penalty_at_threshold",
            output_value=fuel_penalty_at_threshold,
            formula="penalty = (eta_loss/100) * Q * fuel_cost"
        )

    # Calculate optimal interval
    T_opt = calculate_optimal_cleaning_interval(
        fouling_rate=max(avg_fouling_rate, 1e-9),  # Avoid division by zero
        fuel_cost_per_hour_at_threshold=fuel_penalty_at_threshold,
        cleaning_cost_per_cycle=cleaning_cost_per_cycle,
        threshold_fouling_factor=threshold_rf
    )

    # Adjust for actual cleaning effectiveness
    # If cleaning is less effective, clean more frequently
    T_opt_adjusted = T_opt * avg_effectiveness

    if tracker:
        tracker.add_step(
            operation="adjust",
            description="Adjust interval for cleaning effectiveness",
            inputs={"T_opt": T_opt, "avg_effectiveness": avg_effectiveness},
            output_name="T_opt_adjusted",
            output_value=T_opt_adjusted,
            formula="T_adjusted = T_opt * effectiveness"
        )

    T_opt_adjusted = max(4.0, min(168.0, T_opt_adjusted))
    T_rounded = round(T_opt_adjusted, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=T_rounded,
            output_unit="hours",
            precision=2
        )
        return T_rounded, provenance

    return T_rounded


# =============================================================================
# CLEANING EFFECTIVENESS
# =============================================================================

def calculate_cleaning_effectiveness(
    U_before: float,
    U_after: float,
    U_clean: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate the effectiveness of a cleaning cycle.

    Cleaning effectiveness measures how well the cleaning restored
    heat transfer performance.

    Methodology:
        Effectiveness = (U_after - U_before) / (U_clean - U_before)

        Where:
        - 100% effectiveness means full restoration to clean condition
        - 0% effectiveness means no improvement
        - Negative effectiveness means degradation (unlikely)

    Reference: EPRI Soot Blowing Performance Assessment

    Args:
        U_before: U-value before cleaning (BTU/(hr-ft2-F))
        U_after: U-value after cleaning (BTU/(hr-ft2-F))
        U_clean: Clean (baseline) U-value (BTU/(hr-ft2-F))
        track_provenance: If True, return provenance record

    Returns:
        Cleaning effectiveness (0 to 1+), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U_before, "U_before")
    validate_positive(U_after, "U_after")
    validate_positive(U_clean, "U_clean")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="cleaning_effectiveness",
            formula_version="1.0.0",
            inputs={
                "U_before": U_before,
                "U_after": U_after,
                "U_clean": U_clean
            }
        )

    # Calculate improvement
    U_improvement = U_after - U_before

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate U-value improvement",
            inputs={"U_after": U_after, "U_before": U_before},
            output_name="U_improvement",
            output_value=U_improvement,
            formula="U_improvement = U_after - U_before"
        )

    # Calculate maximum possible improvement
    U_max_improvement = U_clean - U_before

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate maximum possible improvement",
            inputs={"U_clean": U_clean, "U_before": U_before},
            output_name="U_max_improvement",
            output_value=U_max_improvement,
            formula="U_max_improvement = U_clean - U_before"
        )

    # Calculate effectiveness
    if U_max_improvement <= 0:
        # Already at or above clean condition
        effectiveness = 1.0 if U_improvement >= 0 else 0.0
    else:
        effectiveness = U_improvement / U_max_improvement

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate cleaning effectiveness",
            inputs={"U_improvement": U_improvement, "U_max_improvement": U_max_improvement},
            output_name="effectiveness",
            output_value=effectiveness,
            formula="effectiveness = U_improvement / U_max_improvement"
        )

    # Clamp to reasonable range
    effectiveness = max(0.0, min(1.5, effectiveness))  # Allow slight over-cleaning

    effectiveness_rounded = round(effectiveness, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=effectiveness_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return effectiveness_rounded, provenance

    return effectiveness_rounded


def calculate_average_cleaning_effectiveness(
    cleaning_events: List[CleaningEvent],
    U_clean: float,
    zone_filter: Optional[EconomizerZone] = None
) -> Dict[str, float]:
    """
    Calculate average cleaning effectiveness from historical events.

    Args:
        cleaning_events: List of past cleaning events
        U_clean: Clean (baseline) U-value
        zone_filter: Optional zone to filter events

    Returns:
        Dictionary with statistics
    """
    if zone_filter:
        events = [e for e in cleaning_events if e.zone == zone_filter]
    else:
        events = cleaning_events

    if not events:
        return {
            "average_effectiveness": 0.0,
            "min_effectiveness": 0.0,
            "max_effectiveness": 0.0,
            "std_dev": 0.0,
            "n_events": 0
        }

    # Calculate effectiveness for each event
    effectiveness_values = []
    for event in events:
        eff = calculate_cleaning_effectiveness(
            U_before=event.U_before,
            U_after=event.U_after,
            U_clean=U_clean
        )
        effectiveness_values.append(eff)

    n = len(effectiveness_values)
    avg = sum(effectiveness_values) / n
    variance = sum((e - avg)**2 for e in effectiveness_values) / n if n > 1 else 0

    return {
        "average_effectiveness": round(avg, 4),
        "min_effectiveness": round(min(effectiveness_values), 4),
        "max_effectiveness": round(max(effectiveness_values), 4),
        "std_dev": round(math.sqrt(variance), 4),
        "n_events": n
    }


# =============================================================================
# MEDIA COST CALCULATIONS
# =============================================================================

def calculate_media_cost(
    config: SootBlowerConfig,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate the media (steam/air) cost per cleaning cycle.

    Methodology:
        For steam:
            Cost = (steam_flow / 1000) * steam_cost_per_klb

        For compressed air:
            Volume_SCF = air_flow_scfm * duration_minutes
            Cost = (Volume_SCF / 1000) * air_cost_per_kscf

    Reference: Plant operating cost guidelines

    Args:
        config: Soot blower configuration
        track_provenance: If True, return provenance record

    Returns:
        Media cost per cycle ($), optionally with provenance

    Raises:
        ValidationError: If configuration is invalid
    """
    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="media_cost",
            formula_version="1.0.0",
            inputs={
                "blower_type": config.blower_type.value,
                "medium": config.medium.value,
                "steam_flow": config.steam_flow_per_cycle_lbm,
                "air_flow_scfm": config.air_flow_per_cycle_scfm,
                "duration_minutes": config.cycle_duration_minutes
            }
        )

    if config.medium == BlowingMedium.STEAM:
        # Steam cost
        media_cost = (config.steam_flow_per_cycle_lbm / 1000) * config.steam_cost_per_klb

        if tracker:
            tracker.add_step(
                operation="multiply",
                description="Calculate steam media cost",
                inputs={
                    "steam_flow_klb": config.steam_flow_per_cycle_lbm / 1000,
                    "cost_per_klb": config.steam_cost_per_klb
                },
                output_name="media_cost",
                output_value=media_cost,
                formula="cost = (steam_flow / 1000) * steam_cost_per_klb"
            )

    elif config.medium == BlowingMedium.COMPRESSED_AIR:
        # Compressed air cost
        air_volume_scf = config.air_flow_per_cycle_scfm * config.cycle_duration_minutes
        media_cost = (air_volume_scf / 1000) * config.air_cost_per_kscf

        if tracker:
            tracker.add_step(
                operation="multiply",
                description="Calculate air volume",
                inputs={
                    "flow_scfm": config.air_flow_per_cycle_scfm,
                    "duration": config.cycle_duration_minutes
                },
                output_name="air_volume",
                output_value=air_volume_scf,
                formula="volume = flow_scfm * duration_minutes"
            )
            tracker.add_step(
                operation="multiply",
                description="Calculate air media cost",
                inputs={
                    "air_volume_kscf": air_volume_scf / 1000,
                    "cost_per_kscf": config.air_cost_per_kscf
                },
                output_name="media_cost",
                output_value=media_cost,
                formula="cost = (volume / 1000) * air_cost_per_kscf"
            )

    else:  # SONIC
        # Sonic horns have minimal media cost (electricity only)
        media_cost = 1.0  # Nominal cost

        if tracker:
            tracker.add_step(
                operation="constant",
                description="Sonic cleaning nominal cost",
                inputs={},
                output_name="media_cost",
                output_value=media_cost,
                formula="media_cost = nominal_electricity_cost"
            )

    media_cost_rounded = round(media_cost, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=media_cost_rounded,
            output_unit="$",
            precision=2
        )
        return media_cost_rounded, provenance

    return media_cost_rounded


def calculate_total_cleaning_cost(
    config: SootBlowerConfig,
    include_maintenance: bool = True,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate total cost per cleaning cycle including all components.

    Methodology:
        Total_Cost = Media_Cost + Maintenance_Cost + Labor_Cost + Opportunity_Cost

    Args:
        config: Soot blower configuration
        include_maintenance: Include maintenance cost
        track_provenance: If True, return provenance record

    Returns:
        Total cost per cycle ($), optionally with provenance
    """
    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="total_cleaning_cost",
            formula_version="1.0.0",
            inputs={
                "blower_id": config.blower_id,
                "include_maintenance": include_maintenance
            }
        )

    # Media cost
    media_cost = calculate_media_cost(config)

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate media cost",
            inputs={"config": config.blower_id},
            output_name="media_cost",
            output_value=media_cost,
            formula="media_cost = f(steam_flow, cost_rate)"
        )

    # Maintenance cost
    maintenance_cost = config.maintenance_cost_per_cycle if include_maintenance else 0.0

    if tracker:
        tracker.add_step(
            operation="add",
            description="Add maintenance cost",
            inputs={"maintenance_cost": maintenance_cost},
            output_name="maintenance_cost",
            output_value=maintenance_cost,
            formula="maintenance_cost per cycle"
        )

    # Total cost
    total_cost = media_cost + maintenance_cost

    if tracker:
        tracker.add_step(
            operation="add",
            description="Calculate total cleaning cost",
            inputs={"media_cost": media_cost, "maintenance_cost": maintenance_cost},
            output_name="total_cost",
            output_value=total_cost,
            formula="total_cost = media_cost + maintenance_cost"
        )

    total_cost_rounded = round(total_cost, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=total_cost_rounded,
            output_unit="$",
            precision=2
        )
        return total_cost_rounded, provenance

    return total_cost_rounded


# =============================================================================
# SEQUENCE OPTIMIZATION
# =============================================================================

def optimize_soot_blower_sequence(
    zone_fouling_factors: Dict[EconomizerZone, float],
    zone_configs: Dict[EconomizerZone, SootBlowerConfig],
    max_blowers_simultaneously: int = 2,
    track_provenance: bool = False
) -> Union[List[EconomizerZone], Tuple[List[EconomizerZone], CalculationProvenance]]:
    """
    Optimize the sequence of soot blower activation.

    This determines which zones should be cleaned first based on
    fouling severity and expected benefit.

    Methodology:
        Priority is determined by:
        1. Fouling factor (higher = more urgent)
        2. Zone location (inlet zones typically more critical)
        3. Cleaning cost efficiency

        Priority Score = Rf * Zone_Weight / Cleaning_Cost

    Reference: EPRI Intelligent Soot Blowing Guidelines

    Args:
        zone_fouling_factors: Fouling factor for each zone
        zone_configs: Soot blower configuration for each zone
        max_blowers_simultaneously: Maximum concurrent blowers
        track_provenance: If True, return provenance record

    Returns:
        Ordered list of zones to clean, optionally with provenance
    """
    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="optimize_sequence",
            formula_version="1.0.0",
            inputs={
                "zones": list(zone_fouling_factors.keys()),
                "max_simultaneous": max_blowers_simultaneously
            }
        )

    # Zone importance weights
    zone_weights = {
        EconomizerZone.INLET: 1.5,    # Hottest, most fouling impact
        EconomizerZone.MIDDLE: 1.0,
        EconomizerZone.OUTLET: 0.8,
        EconomizerZone.TOP: 0.9,
        EconomizerZone.BOTTOM: 1.1    # Ash accumulation
    }

    # Calculate priority scores
    priority_scores = {}
    for zone, rf in zone_fouling_factors.items():
        weight = zone_weights.get(zone, 1.0)
        config = zone_configs.get(zone)

        if config:
            cost = calculate_total_cleaning_cost(config)
            if cost > 0:
                priority_scores[zone] = (rf * weight) / cost * 1000  # Scale for readability
            else:
                priority_scores[zone] = rf * weight * 1000
        else:
            priority_scores[zone] = rf * weight * 1000

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate priority scores for each zone",
            inputs={"zone_fouling": zone_fouling_factors, "weights": zone_weights},
            output_name="priority_scores",
            output_value=max(priority_scores.values()) if priority_scores else 0,
            formula="priority = (Rf * weight) / cost"
        )

    # Sort zones by priority (highest first)
    sorted_zones = sorted(priority_scores.keys(), key=lambda z: priority_scores[z], reverse=True)

    if tracker:
        tracker.add_step(
            operation="sort",
            description="Sort zones by priority",
            inputs={"scores": {z.value: s for z, s in priority_scores.items()}},
            output_name="sorted_zones",
            output_value=len(sorted_zones),
            formula="Sort descending by priority score"
        )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=len(sorted_zones),
            output_unit="zones",
            precision=0
        )
        return sorted_zones, provenance

    return sorted_zones


# =============================================================================
# ROI CALCULATIONS
# =============================================================================

def calculate_cleaning_roi(
    fuel_penalty_before: float,
    fuel_penalty_after: float,
    cleaning_cost: float,
    time_since_last_cleaning_hours: float,
    track_provenance: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], CalculationProvenance]]:
    """
    Calculate Return on Investment for a cleaning cycle.

    Methodology:
        Fuel Savings = (Penalty_Before - Penalty_After) * Expected_Clean_Duration
        Net Benefit = Fuel_Savings - Cleaning_Cost
        ROI = Net_Benefit / Cleaning_Cost * 100%

    Reference: EPRI Economic Analysis Guidelines

    Args:
        fuel_penalty_before: Fuel penalty rate before cleaning ($/hr)
        fuel_penalty_after: Fuel penalty rate after cleaning ($/hr)
        cleaning_cost: Total cost of cleaning cycle ($)
        time_since_last_cleaning_hours: Hours since previous cleaning
        track_provenance: If True, return provenance record

    Returns:
        Dictionary with ROI metrics, optionally with provenance
    """
    validate_non_negative(fuel_penalty_before, "fuel_penalty_before")
    validate_non_negative(fuel_penalty_after, "fuel_penalty_after")
    validate_positive(cleaning_cost, "cleaning_cost")
    validate_positive(time_since_last_cleaning_hours, "time_since_last_cleaning_hours")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="cleaning_roi",
            formula_version="1.0.0",
            inputs={
                "fuel_penalty_before": fuel_penalty_before,
                "fuel_penalty_after": fuel_penalty_after,
                "cleaning_cost": cleaning_cost,
                "time_since_last_cleaning": time_since_last_cleaning_hours
            }
        )

    # Calculate hourly savings
    hourly_savings = fuel_penalty_before - fuel_penalty_after

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate hourly fuel savings",
            inputs={
                "penalty_before": fuel_penalty_before,
                "penalty_after": fuel_penalty_after
            },
            output_name="hourly_savings",
            output_value=hourly_savings,
            formula="hourly_savings = penalty_before - penalty_after"
        )

    # Estimate duration of benefit (assume same as previous interval)
    expected_benefit_hours = time_since_last_cleaning_hours

    # Total fuel savings
    total_fuel_savings = hourly_savings * expected_benefit_hours

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate total fuel savings",
            inputs={"hourly_savings": hourly_savings, "hours": expected_benefit_hours},
            output_name="total_fuel_savings",
            output_value=total_fuel_savings,
            formula="total_savings = hourly_savings * expected_hours"
        )

    # Net benefit
    net_benefit = total_fuel_savings - cleaning_cost

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate net benefit",
            inputs={"fuel_savings": total_fuel_savings, "cleaning_cost": cleaning_cost},
            output_name="net_benefit",
            output_value=net_benefit,
            formula="net_benefit = fuel_savings - cleaning_cost"
        )

    # ROI
    roi_percent = (net_benefit / cleaning_cost) * 100

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate ROI percentage",
            inputs={"net_benefit": net_benefit, "cleaning_cost": cleaning_cost},
            output_name="roi_percent",
            output_value=roi_percent,
            formula="ROI = (net_benefit / cleaning_cost) * 100%"
        )

    # Payback time (hours until cleaning pays for itself)
    if hourly_savings > 0:
        payback_hours = cleaning_cost / hourly_savings
    else:
        payback_hours = float('inf')

    result = {
        "hourly_savings_dollars": round(hourly_savings, 2),
        "total_fuel_savings_dollars": round(total_fuel_savings, 2),
        "net_benefit_dollars": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 2),
        "payback_hours": round(payback_hours, 2) if payback_hours != float('inf') else float('inf'),
        "cleaning_cost_dollars": round(cleaning_cost, 2)
    }

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=roi_percent,
            output_unit="%",
            precision=2
        )
        return result, provenance

    return result


def calculate_annual_cleaning_economics(
    optimal_interval_hours: float,
    cleaning_cost_per_cycle: float,
    average_fuel_savings_per_cycle: float,
    operating_hours_per_year: float = 8000
) -> Dict[str, float]:
    """
    Calculate annual economics of cleaning program.

    Args:
        optimal_interval_hours: Optimal time between cleanings
        cleaning_cost_per_cycle: Cost per cleaning cycle ($)
        average_fuel_savings_per_cycle: Average fuel saved per cycle ($)
        operating_hours_per_year: Annual operating hours

    Returns:
        Dictionary with annual economic metrics
    """
    # Number of cleaning cycles per year
    cycles_per_year = operating_hours_per_year / optimal_interval_hours

    # Annual costs and benefits
    annual_cleaning_cost = cycles_per_year * cleaning_cost_per_cycle
    annual_fuel_savings = cycles_per_year * average_fuel_savings_per_cycle
    annual_net_benefit = annual_fuel_savings - annual_cleaning_cost

    return {
        "cycles_per_year": round(cycles_per_year, 1),
        "annual_cleaning_cost": round(annual_cleaning_cost, 2),
        "annual_fuel_savings": round(annual_fuel_savings, 2),
        "annual_net_benefit": round(annual_net_benefit, 2),
        "roi_percent": round((annual_net_benefit / annual_cleaning_cost) * 100, 2) if annual_cleaning_cost > 0 else 0
    }


# =============================================================================
# COMPREHENSIVE OPTIMIZATION
# =============================================================================

def optimize_cleaning_program(
    current_fouling_factor: float,
    fouling_rate: float,
    U_clean: float,
    boiler_heat_input_mmbtu_hr: float,
    fuel_cost_per_mmbtu: float,
    soot_blower_config: SootBlowerConfig,
    operating_hours_per_year: float = 8000,
    reference_date: Optional[datetime] = None,
    track_provenance: bool = False
) -> Union[OptimizationResult, Tuple[OptimizationResult, CalculationProvenance]]:
    """
    Comprehensive cleaning program optimization.

    This function combines all optimization calculations to provide
    a complete cleaning strategy recommendation.

    Args:
        current_fouling_factor: Current Rf ((hr-ft2-F)/BTU)
        fouling_rate: Rate of fouling ((hr-ft2-F)/(BTU-hr))
        U_clean: Clean U-value (BTU/(hr-ft2-F))
        boiler_heat_input_mmbtu_hr: Boiler heat input (MMBtu/hr)
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        soot_blower_config: Soot blower configuration
        operating_hours_per_year: Annual operating hours
        reference_date: Reference date for next cleaning (default: now)
        track_provenance: If True, return provenance record

    Returns:
        OptimizationResult with complete recommendations
    """
    if reference_date is None:
        reference_date = datetime.now()

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.SOOT_BLOWER,
            formula_id="optimize_cleaning_program",
            formula_version="1.0.0",
            inputs={
                "current_fouling_factor": current_fouling_factor,
                "fouling_rate": fouling_rate,
                "U_clean": U_clean,
                "heat_input": boiler_heat_input_mmbtu_hr,
                "fuel_cost": fuel_cost_per_mmbtu
            }
        )

    # Calculate cleaning cost
    cleaning_cost = calculate_total_cleaning_cost(soot_blower_config)

    # Calculate fuel penalty at threshold
    threshold_rf = 0.005
    efficiency_loss_at_threshold = 1.0  # Assume 1% loss at threshold
    fuel_penalty_at_threshold = (
        (efficiency_loss_at_threshold / 100) *
        boiler_heat_input_mmbtu_hr *
        fuel_cost_per_mmbtu
    )

    # Calculate optimal interval
    optimal_interval = calculate_optimal_cleaning_interval(
        fouling_rate=max(fouling_rate, 1e-9),
        fuel_cost_per_hour_at_threshold=fuel_penalty_at_threshold,
        cleaning_cost_per_cycle=cleaning_cost,
        threshold_fouling_factor=threshold_rf,
        current_fouling_factor=current_fouling_factor
    )

    # Determine time until next cleaning based on current fouling
    if current_fouling_factor >= threshold_rf:
        # Already past threshold - clean now
        time_to_next = 0.0
    else:
        # Predict time to threshold
        if fouling_rate > 0:
            time_to_threshold = (threshold_rf - current_fouling_factor) / fouling_rate
            time_to_next = min(time_to_threshold, optimal_interval)
        else:
            time_to_next = optimal_interval

    next_cleaning_date = reference_date + timedelta(hours=time_to_next)

    # Calculate expected fuel savings per cycle
    # Assume cleaning restores 80% of lost U-value
    U_fouled = U_clean / (1 + current_fouling_factor * U_clean)
    U_after_cleaning = U_clean * 0.95  # Assume 95% restoration

    cleanliness_before = calculate_cleanliness_factor(U_fouled, U_clean)
    cleanliness_after = calculate_cleanliness_factor(U_after_cleaning, U_clean)

    # Efficiency improvement
    efficiency_improvement = (cleanliness_after - cleanliness_before) / 100

    # Fuel savings over next interval
    expected_fuel_savings = (
        efficiency_improvement *
        boiler_heat_input_mmbtu_hr *
        fuel_cost_per_mmbtu *
        optimal_interval
    )

    # Net benefit
    net_benefit_per_cycle = expected_fuel_savings - cleaning_cost

    # Annual benefit
    cycles_per_year = operating_hours_per_year / optimal_interval
    annual_net_benefit = net_benefit_per_cycle * cycles_per_year

    if tracker:
        tracker.add_step(
            operation="optimize",
            description="Calculate comprehensive optimization result",
            inputs={
                "optimal_interval": optimal_interval,
                "cleaning_cost": cleaning_cost,
                "fuel_savings": expected_fuel_savings
            },
            output_name="optimization_result",
            output_value=annual_net_benefit,
            formula="Comprehensive optimization"
        )

    # Zone priority (use configured zones)
    zones_priority = soot_blower_config.zones_covered

    result = OptimizationResult(
        optimal_interval_hours=round(optimal_interval, 2),
        recommended_next_cleaning=next_cleaning_date,
        zones_priority=zones_priority,
        expected_fuel_savings=round(expected_fuel_savings, 2),
        estimated_cleaning_cost=round(cleaning_cost, 2),
        net_benefit_per_cycle=round(net_benefit_per_cycle, 2),
        annual_net_benefit=round(annual_net_benefit, 2)
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=annual_net_benefit,
            output_unit="$/year",
            precision=2
        )
        result.provenance_hash = provenance.provenance_hash
        return result, provenance

    return result
