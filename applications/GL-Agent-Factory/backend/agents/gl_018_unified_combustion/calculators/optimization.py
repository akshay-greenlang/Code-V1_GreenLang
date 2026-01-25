"""
Combustion Optimization Calculator for GL-018 UNIFIEDCOMBUSTION Agent

This module implements deterministic optimization algorithms for:
- O2 trim control optimization
- CO optimization (minimize emissions while maintaining efficiency)
- Excess air control optimization
- Air-fuel ratio optimization

All optimization calculations are deterministic, physics-based formulas.
NO ML/LLM in the optimization calculation path - ensuring zero hallucination
and 100% reproducibility.

Reference Standards:
- ASME PTC 4 Fired Steam Generators
- API 535 Burners for Fired Heaters
- EPA Combustion Efficiency Guidelines
- Industrial Combustion Best Practices

Example:
    >>> from optimization import optimize_o2_setpoint
    >>> result = optimize_o2_setpoint(
    ...     current_o2=4.5,
    ...     current_co=50,
    ...     fuel_type="natural_gas",
    ...     mode="balanced"
    ... )
    >>> print(f"Optimal O2: {result['optimal_o2_pct']:.1f}%")
"""

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import logging

from .combustion import (
    FUEL_PROPERTIES,
    calculate_excess_air,
    calculate_combustion_efficiency,
    calculate_lambda,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Optimization Constants
# =============================================================================

# O2 optimization parameters by fuel type
O2_OPTIMIZATION_PARAMS: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "optimal_o2_min": 2.0,
        "optimal_o2_max": 3.5,
        "efficiency_optimal_o2": 2.5,  # Peak efficiency O2
        "emissions_optimal_o2": 3.0,   # Low NOx O2
        "safety_min_o2": 1.5,          # Minimum safe O2
        "co_breakthrough_o2": 1.2,     # O2 where CO rises sharply
        "efficiency_per_pct_o2": 0.5,  # Efficiency loss per 1% excess O2
    },
    "propane": {
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 4.0,
        "efficiency_optimal_o2": 3.0,
        "emissions_optimal_o2": 3.5,
        "safety_min_o2": 2.0,
        "co_breakthrough_o2": 1.5,
        "efficiency_per_pct_o2": 0.45,
    },
    "fuel_oil_2": {
        "optimal_o2_min": 3.0,
        "optimal_o2_max": 5.0,
        "efficiency_optimal_o2": 3.5,
        "emissions_optimal_o2": 4.0,
        "safety_min_o2": 2.5,
        "co_breakthrough_o2": 2.0,
        "efficiency_per_pct_o2": 0.55,
    },
    "fuel_oil_6": {
        "optimal_o2_min": 3.5,
        "optimal_o2_max": 6.0,
        "efficiency_optimal_o2": 4.5,
        "emissions_optimal_o2": 5.0,
        "safety_min_o2": 3.0,
        "co_breakthrough_o2": 2.5,
        "efficiency_per_pct_o2": 0.6,
    },
    "hydrogen": {
        "optimal_o2_min": 1.0,
        "optimal_o2_max": 3.0,
        "efficiency_optimal_o2": 1.5,
        "emissions_optimal_o2": 2.0,  # Low NOx for H2
        "safety_min_o2": 0.8,
        "co_breakthrough_o2": 0.5,
        "efficiency_per_pct_o2": 0.4,
    },
    "biogas": {
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 5.0,
        "efficiency_optimal_o2": 3.5,
        "emissions_optimal_o2": 4.0,
        "safety_min_o2": 2.0,
        "co_breakthrough_o2": 1.5,
        "efficiency_per_pct_o2": 0.5,
    },
    "coal": {
        "optimal_o2_min": 4.0,
        "optimal_o2_max": 7.0,
        "efficiency_optimal_o2": 5.0,
        "emissions_optimal_o2": 5.5,
        "safety_min_o2": 3.5,
        "co_breakthrough_o2": 3.0,
        "efficiency_per_pct_o2": 0.65,
    },
    "biomass": {
        "optimal_o2_min": 5.0,
        "optimal_o2_max": 8.0,
        "efficiency_optimal_o2": 6.0,
        "emissions_optimal_o2": 6.5,
        "safety_min_o2": 4.0,
        "co_breakthrough_o2": 3.5,
        "efficiency_per_pct_o2": 0.7,
    },
}

# CO response curve parameters
CO_CURVE_PARAMS = {
    "base_co_ppm": 20,         # Base CO at optimal O2
    "co_sensitivity": 2.0,     # CO increase rate factor
    "co_threshold_warning": 100,
    "co_threshold_high": 200,
    "co_threshold_critical": 400,
}

# NOx optimization parameters
NOX_PARAMS = {
    "thermal_nox_factor": 1.0,     # Relative thermal NOx factor
    "fuel_nox_factor": 0.3,        # Relative fuel NOx factor
    "optimal_lambda_for_nox": 1.15, # Lambda for NOx minimum
}


# =============================================================================
# O2 Trim Optimization
# =============================================================================

def optimize_o2_setpoint(
    current_o2_pct: float,
    current_co_ppm: float,
    fuel_type: str,
    optimization_mode: str = "balanced",
    current_nox_ppm: Optional[float] = None,
    firing_rate_pct: float = 100.0,
    ambient_temp_c: float = 25.0,
    stack_temp_c: float = 350.0,
) -> Dict[str, Any]:
    """
    Calculate optimal O2 setpoint for combustion control.

    This is the primary O2 trim optimization function that balances
    efficiency, emissions, and safety objectives.

    Optimization Modes:
    - "efficiency": Minimize excess air for maximum efficiency
    - "emissions": Optimize for minimum NOx and CO
    - "balanced": Balance efficiency and emissions
    - "safety": Conservative settings prioritizing safety margin

    Formula basis:
    - Efficiency loss = k * (O2 - O2_optimal)  [approximately linear]
    - CO risk increases exponentially as O2 approaches breakthrough point
    - NOx varies with flame temperature (affected by excess air)

    Args:
        current_o2_pct: Current O2 percentage reading
        current_co_ppm: Current CO reading in ppm
        fuel_type: Type of fuel being burned
        optimization_mode: Optimization objective
        current_nox_ppm: Current NOx reading (optional)
        firing_rate_pct: Current firing rate as percentage of full load
        ambient_temp_c: Ambient temperature in Celsius
        stack_temp_c: Stack temperature in Celsius

    Returns:
        Dictionary with:
        - optimal_o2_pct: Recommended O2 setpoint
        - o2_trim_range_low: Lower trim limit
        - o2_trim_range_high: Upper trim limit
        - efficiency_gain_pct: Expected efficiency improvement
        - co_risk_assessment: Risk of CO breakthrough
        - adjustment_direction: "increase", "decrease", or "maintain"
        - reasoning: Explanation of optimization decision

    Example:
        >>> result = optimize_o2_setpoint(4.5, 50, "natural_gas", "balanced")
        >>> print(f"Optimal O2: {result['optimal_o2_pct']:.1f}%")
    """
    # Get fuel-specific parameters
    if fuel_type not in O2_OPTIMIZATION_PARAMS:
        logger.warning(f"Unknown fuel type {fuel_type}, using natural_gas defaults")
        fuel_type = "natural_gas"

    params = O2_OPTIMIZATION_PARAMS[fuel_type]

    # Base optimal O2 based on mode
    if optimization_mode == "efficiency":
        base_optimal = params["efficiency_optimal_o2"]
    elif optimization_mode == "emissions":
        base_optimal = params["emissions_optimal_o2"]
    elif optimization_mode == "safety":
        base_optimal = (params["optimal_o2_min"] + params["optimal_o2_max"]) / 2 + 0.5
    else:  # balanced
        base_optimal = (params["efficiency_optimal_o2"] + params["emissions_optimal_o2"]) / 2

    # Adjust for firing rate (more margin at low fire)
    firing_rate_adjustment = 0.0
    if firing_rate_pct < 50:
        # Add margin at low firing rates for stability
        firing_rate_adjustment = (50 - firing_rate_pct) / 100.0
    elif firing_rate_pct > 90:
        # Can run tighter at high fire
        firing_rate_adjustment = -0.2

    # CO breakthrough risk assessment
    co_risk = assess_co_breakthrough_risk(current_o2_pct, current_co_ppm, params)

    # Adjust optimal O2 based on CO risk
    co_adjustment = 0.0
    if co_risk["risk_level"] == "HIGH":
        co_adjustment = 0.5  # Add safety margin
    elif co_risk["risk_level"] == "CRITICAL":
        co_adjustment = 1.0  # Significant safety margin

    # Calculate final optimal O2
    optimal_o2 = base_optimal + firing_rate_adjustment + co_adjustment

    # Clamp to safe operating range
    optimal_o2 = max(params["safety_min_o2"], min(params["optimal_o2_max"] + 0.5, optimal_o2))

    # Calculate trim range
    trim_range_low = max(params["safety_min_o2"], optimal_o2 - 0.5)
    trim_range_high = min(params["optimal_o2_max"] + 1.0, optimal_o2 + 0.5)

    # Calculate expected efficiency gain
    efficiency_gain = calculate_efficiency_gain(
        current_o2_pct, optimal_o2, fuel_type, stack_temp_c, ambient_temp_c
    )

    # Determine adjustment direction
    if current_o2_pct > optimal_o2 + 0.3:
        adjustment_direction = "decrease"
    elif current_o2_pct < optimal_o2 - 0.3:
        adjustment_direction = "increase"
    else:
        adjustment_direction = "maintain"

    # Generate reasoning
    reasoning = generate_o2_optimization_reasoning(
        current_o2_pct, optimal_o2, optimization_mode,
        efficiency_gain, co_risk, firing_rate_pct
    )

    logger.info(
        f"O2 optimization: current={current_o2_pct}%, optimal={optimal_o2:.1f}%, "
        f"mode={optimization_mode}, efficiency_gain={efficiency_gain:.2f}%"
    )

    return {
        "current_o2_pct": current_o2_pct,
        "optimal_o2_pct": round(optimal_o2, 2),
        "o2_trim_range_low": round(trim_range_low, 2),
        "o2_trim_range_high": round(trim_range_high, 2),
        "efficiency_gain_pct": round(efficiency_gain, 3),
        "co_risk_assessment": co_risk["risk_level"],
        "adjustment_direction": adjustment_direction,
        "adjustment_magnitude": round(abs(optimal_o2 - current_o2_pct), 2),
        "reasoning": reasoning,
        "optimization_mode": optimization_mode,
    }


def assess_co_breakthrough_risk(
    current_o2_pct: float,
    current_co_ppm: float,
    params: Dict[str, float]
) -> Dict[str, Any]:
    """
    Assess risk of CO breakthrough based on current conditions.

    CO breakthrough occurs when O2 drops too low, resulting in
    incomplete combustion and rapidly rising CO levels.

    Args:
        current_o2_pct: Current O2 percentage
        current_co_ppm: Current CO in ppm
        params: Fuel-specific optimization parameters

    Returns:
        Dictionary with risk assessment
    """
    safety_min = params["safety_min_o2"]
    co_breakthrough = params["co_breakthrough_o2"]

    # Calculate margin to CO breakthrough
    margin_to_breakthrough = current_o2_pct - co_breakthrough

    # Assess current CO level
    co_status = "NORMAL"
    if current_co_ppm > CO_CURVE_PARAMS["co_threshold_critical"]:
        co_status = "CRITICAL"
    elif current_co_ppm > CO_CURVE_PARAMS["co_threshold_high"]:
        co_status = "HIGH"
    elif current_co_ppm > CO_CURVE_PARAMS["co_threshold_warning"]:
        co_status = "WARNING"

    # Determine overall risk level
    if margin_to_breakthrough < 0.5 or co_status == "CRITICAL":
        risk_level = "CRITICAL"
    elif margin_to_breakthrough < 1.0 or co_status == "HIGH":
        risk_level = "HIGH"
    elif margin_to_breakthrough < 1.5 or co_status == "WARNING":
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "margin_to_breakthrough": round(margin_to_breakthrough, 2),
        "co_status": co_status,
        "current_co_ppm": current_co_ppm,
        "safe_minimum_o2": safety_min,
        "co_breakthrough_o2": co_breakthrough,
    }


def calculate_efficiency_gain(
    current_o2_pct: float,
    target_o2_pct: float,
    fuel_type: str,
    stack_temp_c: float = 350.0,
    ambient_temp_c: float = 25.0
) -> float:
    """
    Calculate expected efficiency improvement from O2 adjustment.

    Uses combustion efficiency calculations to determine actual
    improvement from changing excess air level.

    Args:
        current_o2_pct: Current O2 percentage
        target_o2_pct: Target O2 percentage
        fuel_type: Type of fuel
        stack_temp_c: Stack temperature
        ambient_temp_c: Ambient temperature

    Returns:
        Expected efficiency improvement in percentage points
    """
    # Calculate current efficiency
    current_eff = calculate_combustion_efficiency(
        current_o2_pct, stack_temp_c, fuel_type, ambient_temp_c
    )

    # Calculate target efficiency
    target_eff = calculate_combustion_efficiency(
        target_o2_pct, stack_temp_c, fuel_type, ambient_temp_c
    )

    efficiency_gain = target_eff["combustion_efficiency"] - current_eff["combustion_efficiency"]

    return efficiency_gain


def generate_o2_optimization_reasoning(
    current_o2: float,
    optimal_o2: float,
    mode: str,
    efficiency_gain: float,
    co_risk: Dict[str, Any],
    firing_rate: float
) -> str:
    """
    Generate human-readable reasoning for O2 optimization recommendation.

    Args:
        current_o2: Current O2 percentage
        optimal_o2: Recommended O2 percentage
        mode: Optimization mode
        efficiency_gain: Expected efficiency improvement
        co_risk: CO risk assessment
        firing_rate: Current firing rate

    Returns:
        Reasoning string
    """
    parts = []

    # Direction of change
    if current_o2 > optimal_o2 + 0.3:
        parts.append(
            f"Current O2 ({current_o2:.1f}%) is higher than optimal ({optimal_o2:.1f}%). "
            f"Reducing excess air will improve efficiency by approximately {abs(efficiency_gain):.2f}%."
        )
    elif current_o2 < optimal_o2 - 0.3:
        parts.append(
            f"Current O2 ({current_o2:.1f}%) is lower than optimal ({optimal_o2:.1f}%). "
            f"Increasing excess air will improve combustion completeness and reduce CO risk."
        )
    else:
        parts.append(
            f"Current O2 ({current_o2:.1f}%) is within optimal range. "
            f"Maintain current setpoint for stable operation."
        )

    # Mode-specific reasoning
    if mode == "efficiency":
        parts.append("Optimization mode is 'efficiency', prioritizing fuel savings.")
    elif mode == "emissions":
        parts.append("Optimization mode is 'emissions', prioritizing NOx and CO reduction.")
    elif mode == "safety":
        parts.append("Optimization mode is 'safety', using conservative setpoints with margin.")

    # CO risk consideration
    if co_risk["risk_level"] in ["HIGH", "CRITICAL"]:
        parts.append(
            f"WARNING: CO breakthrough risk is {co_risk['risk_level']}. "
            f"Added safety margin to O2 setpoint."
        )

    # Firing rate consideration
    if firing_rate < 50:
        parts.append(
            f"Operating at low fire ({firing_rate:.0f}%). "
            f"Additional O2 margin added for flame stability."
        )

    return " ".join(parts)


# =============================================================================
# Excess Air Optimization
# =============================================================================

def optimize_excess_air(
    current_o2_pct: float,
    fuel_type: str,
    optimization_mode: str = "balanced",
    current_damper_position: Optional[float] = None,
    fan_capacity_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Optimize excess air and provide damper/fan recommendations.

    Excess air optimization aims to provide just enough air for
    complete combustion while minimizing heat loss in stack gas.

    Formula:
        Excess Air (%) = O2 / (21 - O2) * 100
        Optimal Excess Air varies by fuel (typically 10-20% for gas)

    Args:
        current_o2_pct: Current O2 percentage
        fuel_type: Type of fuel
        optimization_mode: Optimization objective
        current_damper_position: Current damper position (0-100%)
        fan_capacity_pct: Current fan speed as % of capacity

    Returns:
        Dictionary with excess air optimization results
    """
    if fuel_type not in O2_OPTIMIZATION_PARAMS:
        fuel_type = "natural_gas"

    params = O2_OPTIMIZATION_PARAMS[fuel_type]

    # Calculate current excess air
    current_excess_air = calculate_excess_air(current_o2_pct)

    # Determine optimal excess air based on mode
    if optimization_mode == "efficiency":
        optimal_o2 = params["efficiency_optimal_o2"]
    elif optimization_mode == "emissions":
        optimal_o2 = params["emissions_optimal_o2"]
    else:
        optimal_o2 = (params["efficiency_optimal_o2"] + params["emissions_optimal_o2"]) / 2

    optimal_excess_air = calculate_excess_air(optimal_o2)

    # Calculate required adjustment
    excess_air_adjustment = optimal_excess_air - current_excess_air

    # Estimate damper adjustment (approximate linear relationship)
    damper_adjustment = None
    if current_damper_position is not None:
        # Assume ~2% excess air change per 1% damper change
        damper_adjustment = excess_air_adjustment / 2.0
        # New damper position
        new_damper = max(10, min(100, current_damper_position + damper_adjustment))
        damper_adjustment = new_damper - current_damper_position

    # Estimate fan speed adjustment
    fan_adjustment = None
    if fan_capacity_pct is not None:
        # Assume ~1.5% excess air change per 1% fan speed change
        fan_adjustment = excess_air_adjustment / 1.5
        # New fan speed
        new_fan = max(20, min(100, fan_capacity_pct + fan_adjustment))
        fan_adjustment = new_fan - fan_capacity_pct

    # Calculate fuel savings
    fuel_savings = calculate_fuel_savings(current_excess_air, optimal_excess_air, fuel_type)

    logger.info(
        f"Excess air optimization: current={current_excess_air:.1f}%, "
        f"optimal={optimal_excess_air:.1f}%, savings={fuel_savings:.2f}%"
    )

    return {
        "current_excess_air_pct": round(current_excess_air, 2),
        "optimal_excess_air_pct": round(optimal_excess_air, 2),
        "excess_air_adjustment_pct": round(excess_air_adjustment, 2),
        "damper_adjustment_pct": round(damper_adjustment, 2) if damper_adjustment else None,
        "fan_speed_adjustment_pct": round(fan_adjustment, 2) if fan_adjustment else None,
        "fuel_savings_pct": round(fuel_savings, 3),
        "current_lambda": round(calculate_lambda(current_o2_pct), 3),
        "optimal_lambda": round(calculate_lambda(optimal_o2), 3),
    }


def calculate_fuel_savings(
    current_excess_air: float,
    target_excess_air: float,
    fuel_type: str
) -> float:
    """
    Calculate fuel savings from excess air reduction.

    Approximate formula:
        Fuel Savings (%) = (EA_current - EA_target) * k
        where k is fuel-specific sensitivity factor

    Args:
        current_excess_air: Current excess air percentage
        target_excess_air: Target excess air percentage
        fuel_type: Type of fuel

    Returns:
        Fuel savings percentage
    """
    if fuel_type not in O2_OPTIMIZATION_PARAMS:
        fuel_type = "natural_gas"

    params = O2_OPTIMIZATION_PARAMS[fuel_type]

    # Get efficiency sensitivity to O2
    eff_per_o2 = params["efficiency_per_pct_o2"]

    # Convert excess air change to O2 change
    # EA = O2/(21-O2)*100, approximately linear for small changes
    # dO2/dEA ~ 0.21 at typical operating points
    o2_change = (current_excess_air - target_excess_air) * 0.21 / 100.0

    # Fuel savings
    fuel_savings = o2_change * eff_per_o2

    return max(0, fuel_savings)


# =============================================================================
# CO Optimization
# =============================================================================

def optimize_co_control(
    current_co_ppm: float,
    current_o2_pct: float,
    fuel_type: str,
    max_co_limit: float = 100.0
) -> Dict[str, Any]:
    """
    Optimize combustion for CO control.

    CO optimization balances complete combustion (low CO) with
    efficiency (low excess air). There is a trade-off zone where
    reducing excess air improves efficiency but increases CO risk.

    Args:
        current_co_ppm: Current CO reading in ppm
        current_o2_pct: Current O2 percentage
        fuel_type: Type of fuel
        max_co_limit: Maximum allowable CO in ppm

    Returns:
        Dictionary with CO optimization results
    """
    if fuel_type not in O2_OPTIMIZATION_PARAMS:
        fuel_type = "natural_gas"

    params = O2_OPTIMIZATION_PARAMS[fuel_type]

    # Assess current CO status
    if current_co_ppm <= 50:
        co_status = "EXCELLENT"
    elif current_co_ppm <= 100:
        co_status = "GOOD"
    elif current_co_ppm <= 200:
        co_status = "ACCEPTABLE"
    elif current_co_ppm <= max_co_limit:
        co_status = "WARNING"
    else:
        co_status = "HIGH"

    # Determine if O2 adjustment needed
    adjustment_needed = current_co_ppm > max_co_limit * 0.8

    # Calculate recommended O2 increase if CO is high
    recommended_o2_increase = 0.0
    if adjustment_needed:
        # Estimate O2 increase needed
        # CO typically doubles for each 0.3% O2 reduction below optimal
        co_ratio = current_co_ppm / max_co_limit
        if co_ratio > 1.5:
            recommended_o2_increase = 1.0
        elif co_ratio > 1.2:
            recommended_o2_increase = 0.5
        elif co_ratio > 1.0:
            recommended_o2_increase = 0.3

    recommended_o2 = min(params["optimal_o2_max"], current_o2_pct + recommended_o2_increase)

    # Generate recommendations
    recommendations = []
    if co_status == "HIGH":
        recommendations.append("CRITICAL: Immediately increase combustion air")
        recommendations.append("Check for burner nozzle blockage or damage")
        recommendations.append("Verify fuel quality and composition")
    elif co_status == "WARNING":
        recommendations.append("Increase O2 setpoint by 0.3-0.5%")
        recommendations.append("Schedule burner inspection")
    elif current_o2_pct > params["optimal_o2_max"]:
        recommendations.append("CO is low - consider reducing excess air for efficiency")

    return {
        "current_co_ppm": current_co_ppm,
        "co_status": co_status,
        "max_co_limit": max_co_limit,
        "adjustment_needed": adjustment_needed,
        "current_o2_pct": current_o2_pct,
        "recommended_o2_pct": round(recommended_o2, 2),
        "recommended_o2_increase": round(recommended_o2_increase, 2),
        "recommendations": recommendations,
        "margin_to_limit_ppm": round(max_co_limit - current_co_ppm, 1),
    }


# =============================================================================
# Air-Fuel Ratio Optimization
# =============================================================================

def optimize_air_fuel_ratio(
    current_o2_pct: float,
    fuel_type: str,
    fuel_flow_rate: float,
    air_flow_rate: float,
    optimization_mode: str = "balanced"
) -> Dict[str, Any]:
    """
    Calculate optimal air-fuel ratio and adjustments.

    The air-fuel ratio is the ratio of air mass to fuel mass.
    Stoichiometric ratio provides exactly enough air for complete
    combustion. Actual operation requires some excess air.

    Args:
        current_o2_pct: Current O2 percentage
        fuel_type: Type of fuel
        fuel_flow_rate: Fuel flow rate (any consistent unit)
        air_flow_rate: Air flow rate (same unit basis as fuel)
        optimization_mode: Optimization objective

    Returns:
        Dictionary with air-fuel ratio optimization results
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]
    params = O2_OPTIMIZATION_PARAMS.get(fuel_type, O2_OPTIMIZATION_PARAMS["natural_gas"])

    # Get stoichiometric ratio
    stoich_ratio = fuel["stoich_air_ratio"]

    # Calculate current ratio
    if fuel_flow_rate > 0:
        current_ratio = air_flow_rate / fuel_flow_rate
    else:
        current_ratio = stoich_ratio  # Default to stoich if no fuel flow

    # Calculate current lambda
    current_lambda = calculate_lambda(current_o2_pct)

    # Determine optimal lambda based on mode
    if optimization_mode == "efficiency":
        optimal_lambda = 1.0 + (calculate_excess_air(params["efficiency_optimal_o2"]) / 100)
    elif optimization_mode == "emissions":
        optimal_lambda = 1.0 + (calculate_excess_air(params["emissions_optimal_o2"]) / 100)
    else:
        avg_o2 = (params["efficiency_optimal_o2"] + params["emissions_optimal_o2"]) / 2
        optimal_lambda = 1.0 + (calculate_excess_air(avg_o2) / 100)

    # Calculate optimal air-fuel ratio
    optimal_ratio = stoich_ratio * optimal_lambda

    # Calculate adjustment
    ratio_adjustment = optimal_ratio - current_ratio
    adjustment_percent = (ratio_adjustment / current_ratio) * 100 if current_ratio > 0 else 0

    # Recommend air or fuel adjustment
    if abs(ratio_adjustment) < 0.1 * stoich_ratio:
        recommendation = "Air-fuel ratio is within optimal range. No adjustment needed."
    elif ratio_adjustment > 0:
        recommendation = f"Increase air flow by approximately {abs(adjustment_percent):.1f}% or reduce fuel flow."
    else:
        recommendation = f"Reduce air flow by approximately {abs(adjustment_percent):.1f}% for better efficiency."

    return {
        "current_air_fuel_ratio": round(current_ratio, 3),
        "stoichiometric_ratio": round(stoich_ratio, 3),
        "optimal_ratio": round(optimal_ratio, 3),
        "ratio_adjustment": round(ratio_adjustment, 3),
        "adjustment_percent": round(adjustment_percent, 2),
        "current_lambda": round(current_lambda, 3),
        "optimal_lambda": round(optimal_lambda, 3),
        "recommendation": recommendation,
    }


# =============================================================================
# Comprehensive Optimization
# =============================================================================

def generate_optimization_recommendations(
    current_o2_pct: float,
    current_co_ppm: float,
    fuel_type: str,
    stack_temp_c: float,
    optimization_mode: str = "balanced",
    current_nox_ppm: Optional[float] = None,
    firing_rate_pct: float = 100.0,
    max_co_limit: float = 100.0,
    max_nox_limit: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive optimization recommendations.

    Analyzes all combustion parameters and generates prioritized
    recommendations for improving efficiency, reducing emissions,
    and maintaining safety.

    Args:
        current_o2_pct: Current O2 percentage
        current_co_ppm: Current CO in ppm
        fuel_type: Type of fuel
        stack_temp_c: Stack temperature in Celsius
        optimization_mode: Optimization objective
        current_nox_ppm: Current NOx in ppm (optional)
        firing_rate_pct: Current firing rate percentage
        max_co_limit: Maximum allowable CO
        max_nox_limit: Maximum allowable NOx (optional)

    Returns:
        List of optimization recommendations sorted by priority
    """
    recommendations = []

    # Get O2 optimization
    o2_result = optimize_o2_setpoint(
        current_o2_pct, current_co_ppm, fuel_type,
        optimization_mode, current_nox_ppm, firing_rate_pct
    )

    # Get excess air optimization
    excess_air_result = optimize_excess_air(
        current_o2_pct, fuel_type, optimization_mode
    )

    # Get CO optimization
    co_result = optimize_co_control(
        current_co_ppm, current_o2_pct, fuel_type, max_co_limit
    )

    # Generate recommendations based on results

    # O2 adjustment recommendation
    if o2_result["adjustment_direction"] != "maintain":
        priority = "HIGH" if o2_result["adjustment_magnitude"] > 1.0 else "MEDIUM"
        if o2_result["co_risk_assessment"] in ["HIGH", "CRITICAL"]:
            priority = "CRITICAL"

        recommendations.append({
            "parameter": "O2 Setpoint",
            "current_value": current_o2_pct,
            "recommended_value": o2_result["optimal_o2_pct"],
            "unit": "%",
            "expected_improvement": f"{o2_result['efficiency_gain_pct']:.2f}% efficiency gain",
            "priority": priority,
            "confidence": 0.9,
            "reasoning": o2_result["reasoning"],
        })

    # Excess air recommendation
    if abs(excess_air_result["excess_air_adjustment_pct"]) > 5:
        recommendations.append({
            "parameter": "Excess Air",
            "current_value": excess_air_result["current_excess_air_pct"],
            "recommended_value": excess_air_result["optimal_excess_air_pct"],
            "unit": "%",
            "expected_improvement": f"{excess_air_result['fuel_savings_pct']:.2f}% fuel savings",
            "priority": "MEDIUM",
            "confidence": 0.85,
            "reasoning": f"Adjusting excess air from {excess_air_result['current_excess_air_pct']:.1f}% "
                        f"to {excess_air_result['optimal_excess_air_pct']:.1f}% will improve efficiency.",
        })

    # CO-related recommendation
    if co_result["adjustment_needed"]:
        recommendations.append({
            "parameter": "Combustion Air",
            "current_value": current_o2_pct,
            "recommended_value": co_result["recommended_o2_pct"],
            "unit": "% O2",
            "expected_improvement": "Reduce CO emissions to acceptable level",
            "priority": "HIGH" if co_result["co_status"] in ["WARNING", "HIGH"] else "MEDIUM",
            "confidence": 0.95,
            "reasoning": f"CO at {current_co_ppm} ppm ({co_result['co_status']}). "
                        f"Increase O2 to improve combustion completeness.",
        })

    # Stack temperature recommendation
    if stack_temp_c > 400:
        recommendations.append({
            "parameter": "Stack Temperature",
            "current_value": stack_temp_c,
            "recommended_value": 350.0,
            "unit": "C",
            "expected_improvement": "Heat recovery opportunity",
            "priority": "LOW",
            "confidence": 0.8,
            "reasoning": f"Stack temperature of {stack_temp_c}C indicates heat recovery potential. "
                        f"Consider economizer or air preheater.",
        })

    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 5))

    return recommendations


def calculate_potential_savings(
    current_efficiency: float,
    optimal_efficiency: float,
    heat_input_mw: float,
    fuel_cost_per_mwh: float = 30.0,
    operating_hours_per_year: float = 8000.0
) -> Dict[str, float]:
    """
    Calculate potential annual savings from combustion optimization.

    Args:
        current_efficiency: Current combustion efficiency (%)
        optimal_efficiency: Achievable optimal efficiency (%)
        heat_input_mw: Heat input rate in MW
        fuel_cost_per_mwh: Fuel cost per MWh thermal
        operating_hours_per_year: Annual operating hours

    Returns:
        Dictionary with savings calculations
    """
    # Efficiency improvement
    efficiency_gain = optimal_efficiency - current_efficiency

    # Fuel savings rate (MW)
    if current_efficiency > 0:
        fuel_savings_mw = heat_input_mw * (efficiency_gain / current_efficiency)
    else:
        fuel_savings_mw = 0

    # Annual fuel savings (MWh)
    annual_savings_mwh = fuel_savings_mw * operating_hours_per_year

    # Annual cost savings
    annual_cost_savings = annual_savings_mwh * fuel_cost_per_mwh

    # CO2 reduction (assuming natural gas, ~0.2 kg CO2/kWh)
    co2_reduction_tonnes = annual_savings_mwh * 0.2

    return {
        "efficiency_gain_pct": round(efficiency_gain, 2),
        "fuel_savings_mw": round(fuel_savings_mw, 3),
        "annual_savings_mwh": round(annual_savings_mwh, 0),
        "annual_cost_savings": round(annual_cost_savings, 0),
        "co2_reduction_tonnes": round(co2_reduction_tonnes, 1),
    }
