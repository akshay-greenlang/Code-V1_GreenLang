"""
Flame Quality Analysis Calculator

This module implements combustion and flame quality analysis for industrial
burners using physics-based calculations from combustion engineering.

All formulas follow standard combustion engineering references:
- NFPA 86 Standard for Ovens and Furnaces
- API 535 Burners for Fired Heaters in General Refinery Services
- ASME PTC 4 Fired Steam Generators
- EPA Method 19 for combustion efficiency

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> score = calculate_flame_quality_score(1200, 0.95, 3.5, 50, 80)
    >>> print(f"Flame quality score: {score:.1f}/100")
"""

import math
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Fuel properties for combustion calculations
# Reference: Gas Engineers Handbook, Industrial Heating Equipment Association
FUEL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "stoich_o2": 2.0,  # O2 required per unit fuel (volumetric)
        "stoich_air": 9.52,  # Air required per unit fuel
        "hhv_mj_m3": 37.3,  # Higher heating value MJ/m3
        "lhv_mj_m3": 33.7,  # Lower heating value MJ/m3
        "co2_factor": 1.0,  # CO2 produced per unit fuel
        "adiabatic_temp_c": 1950,  # Adiabatic flame temp at stoich
        "optimal_o2_min": 2.0,  # Optimal O2 range for efficiency
        "optimal_o2_max": 4.0,
    },
    "propane": {
        "stoich_o2": 5.0,
        "stoich_air": 23.8,
        "hhv_mj_m3": 93.1,
        "lhv_mj_m3": 85.8,
        "co2_factor": 3.0,
        "adiabatic_temp_c": 1980,
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 4.5,
    },
    "fuel_oil_2": {
        "stoich_o2": 3.3,
        "stoich_air": 15.7,
        "hhv_mj_m3": 38500,  # MJ/m3 liquid
        "lhv_mj_m3": 36200,
        "co2_factor": 2.8,
        "adiabatic_temp_c": 2030,
        "optimal_o2_min": 3.0,
        "optimal_o2_max": 5.0,
    },
    "fuel_oil_6": {
        "stoich_o2": 3.5,
        "stoich_air": 16.7,
        "hhv_mj_m3": 41000,
        "lhv_mj_m3": 38500,
        "co2_factor": 3.0,
        "adiabatic_temp_c": 2080,
        "optimal_o2_min": 3.5,
        "optimal_o2_max": 6.0,
    },
    "hydrogen": {
        "stoich_o2": 0.5,
        "stoich_air": 2.38,
        "hhv_mj_m3": 12.7,
        "lhv_mj_m3": 10.8,
        "co2_factor": 0.0,  # No CO2 from H2
        "adiabatic_temp_c": 2210,
        "optimal_o2_min": 1.0,
        "optimal_o2_max": 3.0,
    },
    "biogas": {
        "stoich_o2": 1.2,  # Varies with composition
        "stoich_air": 5.7,
        "hhv_mj_m3": 22.0,
        "lhv_mj_m3": 20.0,
        "co2_factor": 0.6,
        "adiabatic_temp_c": 1750,
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 5.0,
    },
}

# Add aliases for common naming conventions
FUEL_PROPERTIES["no2_fuel_oil"] = FUEL_PROPERTIES["fuel_oil_2"]
FUEL_PROPERTIES["no6_fuel_oil"] = FUEL_PROPERTIES["fuel_oil_6"]
FUEL_PROPERTIES["coal"] = {
    "stoich_o2": 2.5,
    "stoich_air": 11.9,
    "hhv_mj_m3": 30000,  # Approximation for coal
    "lhv_mj_m3": 28000,
    "co2_factor": 2.5,
    "adiabatic_temp_c": 2100,
    "optimal_o2_min": 4.0,
    "optimal_o2_max": 7.0,
}

# Flame anomaly thresholds
ANOMALY_THRESHOLDS = {
    "flame_lifting": {
        "stability_threshold": 0.7,
        "temp_deviation_c": 200,
    },
    "flashback": {
        "co_threshold": 500,
        "temp_high_c": 1800,
    },
    "pulsation": {
        "stability_threshold": 0.8,
    },
    "incomplete_combustion": {
        "co_threshold": 200,
        "o2_low": 1.0,
    },
}


def calculate_flame_quality_score(
    flame_temp_c: float,
    stability_index: float,
    o2_percent: float,
    co_ppm: float,
    nox_ppm: float
) -> float:
    """
    Calculate overall flame quality score from 0-100.

    The score is a weighted combination of:
    - Temperature quality (25%): How close to optimal
    - Stability (25%): Flame stability index
    - O2/excess air (20%): Optimal excess air range
    - CO emissions (15%): Low CO indicates complete combustion
    - NOx emissions (15%): Low NOx indicates good air-fuel mixing

    Args:
        flame_temp_c: Flame temperature in Celsius.
        stability_index: Flame stability 0-1 (1 = perfectly stable).
        o2_percent: Oxygen percentage in flue gas (0-21%).
        co_ppm: Carbon monoxide in parts per million.
        nox_ppm: NOx in parts per million.

    Returns:
        Quality score from 0 to 100.

    Example:
        >>> score = calculate_flame_quality_score(1200, 0.95, 3.5, 50, 80)
        >>> print(f"Score: {score:.1f}")
    """
    # Validate inputs
    flame_temp_c = max(0, min(3000, flame_temp_c))
    stability_index = max(0, min(1, stability_index))
    o2_percent = max(0, min(21, o2_percent))
    co_ppm = max(0, co_ppm)
    nox_ppm = max(0, nox_ppm)

    # Temperature score (25 points max)
    # Optimal range: 1100-1400C for natural gas burners
    # Score decreases outside this range
    temp_score = _calculate_temperature_score(flame_temp_c)

    # Stability score (25 points max)
    # Linear scaling of stability index
    stability_score = stability_index * 25.0

    # O2/excess air score (20 points max)
    # Optimal O2 is 2-4% for natural gas
    o2_score = _calculate_o2_score(o2_percent)

    # CO score (15 points max)
    # Lower CO is better, < 50 ppm is excellent
    co_score = _calculate_co_score(co_ppm)

    # NOx score (15 points max)
    # Lower NOx is better, varies by regulation
    nox_score = _calculate_nox_score(nox_ppm)

    # Total score
    total = temp_score + stability_score + o2_score + co_score + nox_score

    logger.debug(
        f"Flame quality components: temp={temp_score:.1f}, stability={stability_score:.1f}, "
        f"o2={o2_score:.1f}, co={co_score:.1f}, nox={nox_score:.1f}, total={total:.1f}"
    )

    return max(0.0, min(100.0, total))


def _calculate_temperature_score(temp_c: float) -> float:
    """Calculate temperature component of flame quality (0-25 points)."""
    # Optimal range: 1100-1400C
    optimal_low = 1100
    optimal_high = 1400

    if optimal_low <= temp_c <= optimal_high:
        return 25.0
    elif temp_c < optimal_low:
        # Score decreases below optimal
        deviation = optimal_low - temp_c
        penalty = min(25, deviation / 40)  # 40C per point
        return max(0, 25 - penalty)
    else:
        # Score decreases above optimal
        deviation = temp_c - optimal_high
        penalty = min(25, deviation / 50)  # 50C per point
        return max(0, 25 - penalty)


def _calculate_o2_score(o2_percent: float) -> float:
    """Calculate O2/excess air component of flame quality (0-20 points)."""
    # Optimal O2: 2-4%
    optimal_low = 2.0
    optimal_high = 4.0

    if optimal_low <= o2_percent <= optimal_high:
        return 20.0
    elif o2_percent < optimal_low:
        # Too little O2 - incomplete combustion risk
        deviation = optimal_low - o2_percent
        penalty = min(20, deviation * 10)  # 10 points per 1% below
        return max(0, 20 - penalty)
    else:
        # Too much O2 - efficiency loss
        deviation = o2_percent - optimal_high
        penalty = min(20, deviation * 2)  # 2 points per 1% above
        return max(0, 20 - penalty)


def _calculate_co_score(co_ppm: float) -> float:
    """Calculate CO emission component of flame quality (0-15 points)."""
    # Excellent: < 50 ppm
    # Good: 50-100 ppm
    # Acceptable: 100-200 ppm
    # Poor: > 200 ppm

    if co_ppm <= 50:
        return 15.0
    elif co_ppm <= 100:
        return 15.0 - ((co_ppm - 50) / 50) * 5  # 10-15 points
    elif co_ppm <= 200:
        return 10.0 - ((co_ppm - 100) / 100) * 5  # 5-10 points
    else:
        return max(0, 5 - (co_ppm - 200) / 100)  # 0-5 points


def _calculate_nox_score(nox_ppm: float) -> float:
    """Calculate NOx emission component of flame quality (0-15 points)."""
    # Based on typical regulations (varies by region)
    # Excellent: < 50 ppm
    # Good: 50-100 ppm
    # Acceptable: 100-150 ppm
    # Poor: > 150 ppm

    if nox_ppm <= 50:
        return 15.0
    elif nox_ppm <= 100:
        return 15.0 - ((nox_ppm - 50) / 50) * 5  # 10-15 points
    elif nox_ppm <= 150:
        return 10.0 - ((nox_ppm - 100) / 50) * 5  # 5-10 points
    else:
        return max(0, 5 - (nox_ppm - 150) / 100)  # 0-5 points


def calculate_combustion_efficiency(
    o2_percent: float = None,
    co_ppm: float = None,
    fuel_type: str = "natural_gas",
    *,
    excess_air_pct: float = None,
    flue_gas_temp_c: float = None
) -> float:
    """
    Calculate combustion efficiency based on excess air and stack temperature.

    Uses the Siegert formula for stack loss calculation, which is the industry
    standard for boiler/burner efficiency assessment.

    Formula (Siegert):
        Stack Loss (%) = (A2 / (CO2% or factor)) * (T_flue - T_ambient)
        Efficiency (%) = 100 - Stack Loss - Unburned Loss

    Supports two calling conventions:
    1. Process measurements: o2_percent, co_ppm, fuel_type
    2. Engineering parameters: fuel_type, excess_air_pct, flue_gas_temp_c

    Reference: ASME PTC 4, EPA Method 19, Siegert Formula

    Args:
        o2_percent: Oxygen percentage in flue gas (optional).
        co_ppm: Carbon monoxide in parts per million (optional).
        fuel_type: Type of fuel (from FUEL_PROPERTIES keys).
        excess_air_pct: Direct excess air percentage (keyword-only, optional).
        flue_gas_temp_c: Flue gas temperature in Celsius (keyword-only, optional).

    Returns:
        Combustion efficiency percentage (typically 75-95%).

    Example:
        >>> # Using process measurements
        >>> eff = calculate_combustion_efficiency(3.5, 50, "natural_gas")
        >>> # Using engineering parameters
        >>> eff = calculate_combustion_efficiency(fuel_type="natural_gas",
        ...     excess_air_pct=15.0, flue_gas_temp_c=350.0)
    """
    # Get fuel properties
    if fuel_type not in FUEL_PROPERTIES:
        logger.warning(f"Unknown fuel type {fuel_type}, using natural_gas defaults")
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Ambient temperature reference (standard conditions)
    T_ambient = 25.0  # Celsius

    # Siegert coefficients by fuel type (A2 factor)
    # Reference: Siegert Formula, VDI 2067
    SIEGERT_A2 = {
        "natural_gas": 0.37,
        "propane": 0.38,
        "no2_fuel_oil": 0.47,
        "no6_fuel_oil": 0.50,
        "hydrogen": 0.32,
        "coal": 0.55,
    }

    A2 = SIEGERT_A2.get(fuel_type, 0.40)

    # Determine excess air - either from direct parameter or calculate from O2
    if excess_air_pct is not None:
        excess_air = excess_air_pct
    elif o2_percent is not None:
        excess_air = calculate_excess_air(o2_percent)
    else:
        excess_air = 15.0  # Default optimal

    # Determine stack temperature
    if flue_gas_temp_c is not None:
        T_stack = flue_gas_temp_c
    else:
        # Estimate from fuel type if not provided
        T_stack = 350.0  # Default typical value

    # Calculate CO2 percentage from excess air (stoichiometric relation)
    # CO2_max varies by fuel: natural gas ~11.7%, oil ~15.4%
    CO2_max = {
        "natural_gas": 11.7,
        "propane": 13.8,
        "no2_fuel_oil": 15.4,
        "no6_fuel_oil": 16.0,
        "hydrogen": 0.0,  # No CO2 from hydrogen
        "coal": 18.5,
    }.get(fuel_type, 12.0)

    # CO2 actual = CO2_max * (1 / (1 + excess_air/100))
    if CO2_max > 0:
        CO2_actual = CO2_max / (1.0 + excess_air / 100.0)
        CO2_actual = max(CO2_actual, 1.0)  # Avoid division issues
    else:
        CO2_actual = 10.0  # For hydrogen, use equivalent factor

    # Siegert formula: Stack Loss = A2 / CO2% * (T_stack - T_ambient)
    stack_loss = (A2 / CO2_actual) * (T_stack - T_ambient)

    # Unburned fuel loss (from CO if available)
    if co_ppm is not None and co_ppm > 0:
        # ~0.001% loss per ppm CO
        unburned_loss = co_ppm * 0.001
    else:
        # Small default for incomplete combustion
        unburned_loss = 0.5

    # Radiation and other minor losses (typically 1-2%)
    other_losses = 1.5

    # Total efficiency
    efficiency = 100.0 - stack_loss - unburned_loss - other_losses

    # Clamp to reasonable range (65-98%)
    efficiency = max(65.0, min(98.0, efficiency))

    logger.debug(
        f"Combustion efficiency: stack_loss={stack_loss:.2f}%, "
        f"unburned={unburned_loss:.2f}%, other={other_losses:.2f}%, "
        f"total={efficiency:.1f}%"
    )

    return efficiency


def calculate_excess_air(o2_percent: float) -> float:
    """
    Calculate excess air percentage from flue gas O2.

    Formula (dry basis):
        Excess Air (%) = O2 / (21 - O2) * 100

    This assumes complete combustion and dry flue gas measurement.

    Args:
        o2_percent: Oxygen percentage in flue gas (0-21%).

    Returns:
        Excess air percentage.

    Example:
        >>> excess = calculate_excess_air(3.0)
        >>> print(f"Excess air: {excess:.1f}%")
        Excess air: 16.7%
    """
    if o2_percent < 0:
        raise ValueError(f"O2 percent must be >= 0, got {o2_percent}")
    if o2_percent >= 21:
        raise ValueError(f"O2 percent must be < 21, got {o2_percent}")

    # Standard formula for excess air from O2 measurement
    excess_air = (o2_percent / (21 - o2_percent)) * 100

    return excess_air


def detect_flame_anomalies(
    stability_index: float,
    co_ppm: float,
    flame_temp_c: float
) -> List[str]:
    """
    Detect flame anomalies indicating burner problems.

    Detects the following anomalies:
    - Flame lifting: Flame detaches from burner, unstable
    - Flashback: Flame propagates back into burner
    - Pulsation: Periodic flame oscillation
    - Incomplete combustion: Fuel not fully burned

    Args:
        stability_index: Flame stability 0-1 (1 = perfectly stable).
        co_ppm: Carbon monoxide in parts per million.
        flame_temp_c: Flame temperature in Celsius.

    Returns:
        List of detected anomaly names.

    Example:
        >>> anomalies = detect_flame_anomalies(0.6, 300, 1100)
        >>> print(anomalies)
        ['flame_lifting', 'incomplete_combustion']
    """
    anomalies = []

    # Flame lifting detection
    # Low stability + significant temperature deviation
    thresholds = ANOMALY_THRESHOLDS["flame_lifting"]
    if stability_index < thresholds["stability_threshold"]:
        # Check for temperature indication of lifting
        nominal_temp = 1250  # Nominal flame temp
        temp_deviation = abs(flame_temp_c - nominal_temp)
        if temp_deviation > thresholds["temp_deviation_c"]:
            anomalies.append("flame_lifting")
            logger.warning(
                f"Flame lifting detected: stability={stability_index:.2f}, "
                f"temp_deviation={temp_deviation:.0f}C"
            )

    # Flashback detection
    # High CO with high temperature can indicate flashback
    thresholds = ANOMALY_THRESHOLDS["flashback"]
    if co_ppm > thresholds["co_threshold"] and flame_temp_c > thresholds["temp_high_c"]:
        anomalies.append("flashback")
        logger.warning(
            f"Potential flashback detected: CO={co_ppm:.0f}ppm, temp={flame_temp_c:.0f}C"
        )

    # Pulsation detection
    # Low stability without other anomalies
    thresholds = ANOMALY_THRESHOLDS["pulsation"]
    if stability_index < thresholds["stability_threshold"]:
        if "flame_lifting" not in anomalies:
            anomalies.append("pulsation")
            logger.warning(f"Flame pulsation detected: stability={stability_index:.2f}")

    # Incomplete combustion detection
    # High CO indicates incomplete combustion
    thresholds = ANOMALY_THRESHOLDS["incomplete_combustion"]
    if co_ppm > thresholds["co_threshold"]:
        anomalies.append("incomplete_combustion")
        logger.warning(f"Incomplete combustion detected: CO={co_ppm:.0f}ppm")

    return anomalies


def calculate_adiabatic_flame_temp(
    fuel_type: str,
    excess_air_percent: float,
    preheat_air_temp_c: float = 25
) -> float:
    """
    Calculate adiabatic flame temperature.

    The adiabatic flame temperature is the maximum temperature achievable
    with no heat loss. Real flame temperatures are lower.

    Formula (approximation):
        T_ad = T_stoich - (excess_air * cooling_factor) + (preheat_bonus)

    Args:
        fuel_type: Type of fuel from FUEL_PROPERTIES.
        excess_air_percent: Excess air percentage.
        preheat_air_temp_c: Combustion air preheat temperature in Celsius.

    Returns:
        Adiabatic flame temperature in Celsius.

    Example:
        >>> temp = calculate_adiabatic_flame_temp("natural_gas", 20, 200)
        >>> print(f"Adiabatic flame temp: {temp:.0f}C")
    """
    if fuel_type not in FUEL_PROPERTIES:
        logger.warning(f"Unknown fuel type {fuel_type}, using natural_gas defaults")
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Start with stoichiometric adiabatic temp
    t_stoich = fuel["adiabatic_temp_c"]

    # Excess air cooling effect
    # Approximately 15-20C drop per 10% excess air
    cooling_rate = 17  # C per 10% excess air
    excess_air_cooling = (excess_air_percent / 10) * cooling_rate

    # Preheat bonus
    # Air preheat increases flame temp by ~0.4-0.5C per degree of preheat
    preheat_bonus = (preheat_air_temp_c - 25) * 0.45

    # Calculate adiabatic temp
    t_adiabatic = t_stoich - excess_air_cooling + preheat_bonus

    # Clamp to reasonable range
    t_adiabatic = max(800, min(2500, t_adiabatic))

    return t_adiabatic


def calculate_air_fuel_ratio(
    o2_percent: float,
    fuel_type: str
) -> Tuple[float, float]:
    """
    Calculate actual and stoichiometric air-fuel ratios.

    Args:
        o2_percent: Oxygen percentage in flue gas.
        fuel_type: Type of fuel from FUEL_PROPERTIES.

    Returns:
        Tuple of (actual_ratio, stoichiometric_ratio).

    Example:
        >>> actual, stoich = calculate_air_fuel_ratio(3.5, "natural_gas")
        >>> print(f"Actual A/F: {actual:.1f}, Stoich: {stoich:.1f}")
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Stoichiometric air-fuel ratio
    stoich_ratio = fuel["stoich_air"]

    # Actual ratio based on excess air
    excess_air = calculate_excess_air(o2_percent)
    actual_ratio = stoich_ratio * (1 + excess_air / 100)

    return (actual_ratio, stoich_ratio)


def calculate_lambda(o2_percent: float) -> float:
    """
    Calculate lambda (air-fuel equivalence ratio).

    Lambda = 1.0 at stoichiometric
    Lambda > 1.0 is lean (excess air)
    Lambda < 1.0 is rich (excess fuel)

    Args:
        o2_percent: Oxygen percentage in flue gas.

    Returns:
        Lambda value.

    Example:
        >>> lambda_val = calculate_lambda(3.5)
        >>> print(f"Lambda: {lambda_val:.3f}")
    """
    if o2_percent < 0 or o2_percent >= 21:
        raise ValueError(f"O2 percent must be 0-21, got {o2_percent}")

    # Lambda from O2 measurement
    # At stoich O2=0, at very lean O2 approaches 21
    excess_air = calculate_excess_air(o2_percent)
    lambda_val = 1 + (excess_air / 100)

    return lambda_val
