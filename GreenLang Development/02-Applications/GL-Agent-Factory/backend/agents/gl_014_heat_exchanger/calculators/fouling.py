"""
Fouling Analysis and Prediction Calculator for Heat Exchangers

This module implements fouling analysis, prediction, and cleaning schedule
optimization for heat exchangers following TEMA standards and industry
best practices.

Fouling reduces heat transfer effectiveness and increases pressure drop,
leading to energy losses and operational costs. This module provides
deterministic models for fouling prediction based on operating conditions.

All formulas follow standard heat transfer engineering references:
- TEMA Standards (10th Edition) - Fouling resistances
- Kern - Process Heat Transfer
- Muller-Steinhagen & Heck correlation for fouling rates
- Ebert-Panchal correlation for crude oil fouling
- HTRI (Heat Transfer Research Inc.) guidelines

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path for numerical values.

Example:
    >>> rf = calculate_fouling_resistance(5000, 0.000088, 1.5)
    >>> print(f"Fouling resistance: {rf:.6f} m2-K/W")
"""

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# TEMA recommended fouling resistances (m2-K/W)
# Reference: TEMA Standards, Table RGP-T2.4
TEMA_FOULING_RESISTANCES: Dict[str, Dict[str, float]] = {
    "water": {
        "sea_water_clean": 0.000088,
        "sea_water_turbid": 0.000176,
        "brackish_water": 0.000352,
        "cooling_tower_treated": 0.000176,
        "cooling_tower_untreated": 0.000528,
        "city_water": 0.000176,
        "river_water_clean": 0.000352,
        "river_water_muddy": 0.000528,
        "boiler_feedwater_treated": 0.000088,
        "boiler_blowdown": 0.000352,
        "condensate": 0.000088,
        "distilled_water": 0.000088,
    },
    "steam": {
        "clean": 0.000088,
        "oil_bearing": 0.000176,
        "exhaust": 0.000176,
    },
    "gases": {
        "air_clean": 0.000176,
        "air_industrial": 0.000352,
        "flue_gas_clean": 0.000176,
        "flue_gas_ash_laden": 0.000881,
        "natural_gas": 0.000088,
        "hydrogen": 0.000088,
        "ammonia": 0.000176,
    },
    "liquids": {
        "light_hydrocarbons": 0.000176,
        "medium_hydrocarbons": 0.000352,
        "heavy_hydrocarbons": 0.000528,
        "crude_oil_dry": 0.000528,
        "crude_oil_wet": 0.000881,
        "gasoline": 0.000176,
        "diesel": 0.000352,
        "fuel_oil": 0.000881,
        "vegetable_oil": 0.000528,
        "organic_solvents": 0.000176,
    },
    "chemicals": {
        "caustic_solutions": 0.000352,
        "acid_solutions": 0.000352,
        "brine": 0.000352,
        "refrigerant": 0.000176,
        "process_fluids": 0.000352,
    },
}


# Fouling rate coefficients for different mechanisms
FOULING_RATE_COEFFICIENTS = {
    "particulate": {
        "deposition_rate": 1e-9,  # m2-K/W per hour base rate
        "velocity_exponent": -1.5,  # Higher velocity reduces fouling
        "temp_coefficient": 0.02,  # Temperature effect per 10C
    },
    "precipitation": {
        "deposition_rate": 2e-9,
        "velocity_exponent": -0.5,
        "activation_energy": 40000,  # J/mol for Arrhenius
    },
    "corrosion": {
        "base_rate": 0.5e-9,
        "ph_factor": 0.1,  # Effect per pH unit from neutral
        "temp_coefficient": 0.03,
    },
    "biological": {
        "growth_rate": 3e-9,
        "max_thickness": 0.003,  # meters
        "temp_optimal": 35,  # Celsius
    },
    "coking": {
        "activation_energy": 70000,  # J/mol
        "film_temp_threshold": 350,  # Celsius
        "base_rate": 5e-9,
    },
}


def get_tema_fouling_resistance(
    fluid_category: str,
    fluid_type: str
) -> float:
    """
    Get TEMA standard fouling resistance for a fluid.

    TEMA provides tabulated fouling resistances for common industrial
    fluids, based on typical operating conditions.

    Args:
        fluid_category: Category of fluid (water, steam, gases, liquids, chemicals).
        fluid_type: Specific fluid type within category.

    Returns:
        Fouling resistance in m2-K/W.

    Raises:
        ValueError: If fluid category or type not found.

    Example:
        >>> rf = get_tema_fouling_resistance("water", "cooling_tower_treated")
        >>> print(f"Fouling resistance: {rf:.6f} m2-K/W")
    """
    category = fluid_category.lower().replace(" ", "_").replace("-", "_")
    fluid = fluid_type.lower().replace(" ", "_").replace("-", "_")

    if category not in TEMA_FOULING_RESISTANCES:
        raise ValueError(
            f"Unknown fluid category: {category}. "
            f"Available: {list(TEMA_FOULING_RESISTANCES.keys())}"
        )

    fluids = TEMA_FOULING_RESISTANCES[category]

    if fluid not in fluids:
        raise ValueError(
            f"Unknown fluid type: {fluid}. "
            f"Available in {category}: {list(fluids.keys())}"
        )

    return fluids[fluid]


def calculate_fouling_resistance(
    operating_hours: float,
    initial_fouling_rate: float,
    asymptotic_factor: float = 1.0,
    asymptotic_rf: Optional[float] = None
) -> float:
    """
    Calculate fouling resistance as a function of time.

    Uses an asymptotic fouling model where fouling rate decreases
    as the surface becomes fouled.

    Formula (asymptotic model):
        Rf(t) = Rf_max * (1 - exp(-k * t))

    where:
        - Rf_max: Asymptotic (maximum) fouling resistance
        - k: Rate constant (1/hours)
        - t: Time (hours)

    Args:
        operating_hours: Operating time since cleaning (hours).
        initial_fouling_rate: Initial fouling rate (m2-K/W per 1000 hours).
        asymptotic_factor: Multiplier for asymptotic behavior. Default 1.0.
        asymptotic_rf: Maximum fouling resistance (m2-K/W). If None,
            estimated from initial rate.

    Returns:
        Current fouling resistance (m2-K/W).

    Example:
        >>> rf = calculate_fouling_resistance(5000, 0.000088, 1.5)
        >>> print(f"Fouling resistance: {rf:.6f} m2-K/W")
    """
    if operating_hours < 0:
        raise ValueError(f"Operating hours must be >= 0, got {operating_hours}")
    if initial_fouling_rate < 0:
        raise ValueError(f"Initial fouling rate must be >= 0")

    if operating_hours == 0:
        return 0.0

    # Estimate asymptotic fouling resistance if not provided
    if asymptotic_rf is None:
        # Assume asymptote is reached at approximately 10000 hours
        asymptotic_rf = initial_fouling_rate * 10 * asymptotic_factor

    # Rate constant (characteristic time ~3000 hours)
    k = initial_fouling_rate / asymptotic_rf if asymptotic_rf > 0 else 0.0003

    # Asymptotic fouling model
    rf = asymptotic_rf * (1 - math.exp(-k * operating_hours))

    logger.debug(
        f"Fouling resistance: t={operating_hours}h, rate={initial_fouling_rate:.2e}, "
        f"Rf_max={asymptotic_rf:.2e}, Rf={rf:.6f} m2-K/W"
    )

    return rf


def calculate_fouling_rate(
    velocity: float,
    wall_temperature: float,
    bulk_temperature: float,
    fouling_mechanism: str = "particulate",
    fluid_properties: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate instantaneous fouling rate based on operating conditions.

    The fouling rate depends on the fouling mechanism and operating
    conditions. This function implements correlations for different
    fouling types.

    Mechanisms:
        - particulate: Particle deposition from fluid
        - precipitation: Chemical precipitation (scaling)
        - corrosion: Corrosion product formation
        - biological: Biofilm growth
        - coking: Hydrocarbon pyrolysis

    Args:
        velocity: Fluid velocity (m/s).
        wall_temperature: Wall/surface temperature (C).
        bulk_temperature: Bulk fluid temperature (C).
        fouling_mechanism: Type of fouling mechanism.
        fluid_properties: Optional fluid properties dict.

    Returns:
        Fouling rate (m2-K/W per 1000 hours).

    Example:
        >>> rate = calculate_fouling_rate(2.0, 85, 70, "particulate")
        >>> print(f"Fouling rate: {rate:.6f} m2-K/W per 1000h")
    """
    mechanism = fouling_mechanism.lower().replace(" ", "_").replace("-", "_")

    if mechanism not in FOULING_RATE_COEFFICIENTS:
        logger.warning(f"Unknown mechanism '{mechanism}', using particulate")
        mechanism = "particulate"

    coeffs = FOULING_RATE_COEFFICIENTS[mechanism]

    # Base rate
    if mechanism == "particulate":
        # Kern correlation modified
        base_rate = coeffs["deposition_rate"]
        velocity_factor = max(0.1, velocity) ** coeffs["velocity_exponent"]
        temp_factor = 1 + coeffs["temp_coefficient"] * (wall_temperature - 60) / 10
        fouling_rate = base_rate * velocity_factor * temp_factor * 1000

    elif mechanism == "precipitation":
        # Muller-Steinhagen type correlation
        base_rate = coeffs["deposition_rate"]
        velocity_factor = max(0.1, velocity) ** coeffs["velocity_exponent"]
        # Arrhenius temperature dependence
        r_gas = 8.314  # J/mol-K
        t_kelvin = wall_temperature + 273.15
        temp_factor = math.exp(-coeffs["activation_energy"] / (r_gas * t_kelvin))
        fouling_rate = base_rate * velocity_factor * temp_factor * 1e6 * 1000

    elif mechanism == "corrosion":
        base_rate = coeffs["base_rate"]
        temp_factor = 1 + coeffs["temp_coefficient"] * (wall_temperature - 25) / 10
        # pH factor (assume neutral if not provided)
        ph = fluid_properties.get("ph", 7.0) if fluid_properties else 7.0
        ph_factor = 1 + coeffs["ph_factor"] * abs(ph - 7)
        fouling_rate = base_rate * temp_factor * ph_factor * 1000

    elif mechanism == "biological":
        # Temperature-dependent growth rate
        growth_rate = coeffs["growth_rate"]
        t_opt = coeffs["temp_optimal"]
        # Optimal growth at t_opt, decreases away from it
        temp_factor = math.exp(-((wall_temperature - t_opt) / 15) ** 2)
        fouling_rate = growth_rate * temp_factor * 1000

    elif mechanism == "coking":
        # Ebert-Panchal type correlation for hydrocarbon fouling
        base_rate = coeffs["base_rate"]
        film_temp = (wall_temperature + bulk_temperature) / 2

        if film_temp < coeffs["film_temp_threshold"]:
            # Below threshold, minimal coking
            fouling_rate = base_rate * 0.1 * 1000
        else:
            # Arrhenius above threshold
            r_gas = 8.314
            t_kelvin = film_temp + 273.15
            temp_factor = math.exp(-coeffs["activation_energy"] / (r_gas * t_kelvin))
            velocity_factor = max(0.1, 1 / velocity)  # Lower velocity = more fouling
            fouling_rate = base_rate * temp_factor * velocity_factor * 1e6 * 1000

    else:
        fouling_rate = 1e-6 * 1000  # Default low rate

    logger.debug(
        f"Fouling rate: mechanism={mechanism}, v={velocity:.2f}m/s, "
        f"T_wall={wall_temperature:.0f}C, rate={fouling_rate:.6f} m2-K/W per 1000h"
    )

    return fouling_rate


def calculate_ua_degradation(
    ua_clean: float,
    fouling_resistance_hot: float,
    fouling_resistance_cold: float,
    area: float = 1.0
) -> Dict[str, float]:
    """
    Calculate UA degradation due to fouling.

    Fouling adds thermal resistance, reducing the overall heat
    transfer coefficient.

    Formula:
        1/U_fouled = 1/U_clean + Rf_hot + Rf_cold
        UA_fouled = U_fouled * A

    Args:
        ua_clean: Clean UA value (W/K).
        fouling_resistance_hot: Hot side fouling resistance (m2-K/W).
        fouling_resistance_cold: Cold side fouling resistance (m2-K/W).
        area: Heat transfer area (m2). Default 1.0 for per-unit calculations.

    Returns:
        Dictionary with:
        - ua_fouled: Fouled UA value (W/K)
        - ua_reduction_percent: Percentage reduction in UA
        - fouling_factor: UA_fouled / UA_clean
        - u_clean: Clean U value (W/m2-K)
        - u_fouled: Fouled U value (W/m2-K)

    Example:
        >>> result = calculate_ua_degradation(50000, 0.0003, 0.0002, 100)
        >>> print(f"UA reduction: {result['ua_reduction_percent']:.1f}%")
    """
    if ua_clean <= 0:
        raise ValueError(f"UA_clean must be > 0, got {ua_clean}")
    if area <= 0:
        raise ValueError(f"Area must be > 0, got {area}")

    # Calculate U values
    u_clean = ua_clean / area

    # Total fouling resistance
    total_rf = fouling_resistance_hot + fouling_resistance_cold

    # Fouled U value
    # 1/U_fouled = 1/U_clean + Rf_total
    u_fouled = 1 / (1 / u_clean + total_rf)

    # Fouled UA
    ua_fouled = u_fouled * area

    # Reduction metrics
    ua_reduction = (ua_clean - ua_fouled) / ua_clean * 100
    fouling_factor = ua_fouled / ua_clean

    logger.debug(
        f"UA degradation: UA_clean={ua_clean:.0f}, UA_fouled={ua_fouled:.0f} W/K, "
        f"reduction={ua_reduction:.1f}%"
    )

    return {
        "ua_fouled": ua_fouled,
        "ua_reduction_percent": ua_reduction,
        "fouling_factor": fouling_factor,
        "u_clean": u_clean,
        "u_fouled": u_fouled,
        "total_fouling_resistance": total_rf,
    }


def predict_fouling_over_time(
    hours_forward: float,
    current_rf: float,
    fouling_rate: float,
    asymptotic_rf: Optional[float] = None,
    time_step_hours: float = 1000
) -> List[Dict[str, float]]:
    """
    Predict fouling resistance over future time period.

    Generates a time series of predicted fouling resistance values.

    Args:
        hours_forward: Hours to predict into future.
        current_rf: Current fouling resistance (m2-K/W).
        fouling_rate: Current fouling rate (m2-K/W per 1000 hours).
        asymptotic_rf: Maximum fouling resistance. If None, estimated.
        time_step_hours: Time step for predictions (hours).

    Returns:
        List of dictionaries with:
        - hours: Time from now
        - rf: Predicted fouling resistance
        - rf_percent_of_max: Percentage of asymptotic value

    Example:
        >>> predictions = predict_fouling_over_time(10000, 0.0001, 0.00005)
        >>> for p in predictions:
        ...     print(f"t={p['hours']}h: Rf={p['rf']:.6f}")
    """
    if hours_forward <= 0:
        return []

    # Estimate asymptotic Rf
    if asymptotic_rf is None:
        asymptotic_rf = max(current_rf * 2, fouling_rate * 15)

    # Calculate rate constant from current state
    if current_rf > 0 and current_rf < asymptotic_rf:
        # Solve for k from current Rf: Rf = Rf_max * (1 - exp(-k*t0))
        # Assuming we're at some reference time t0
        k = fouling_rate / asymptotic_rf
    else:
        k = 0.0001  # Default

    predictions = []
    current_hours = 0

    while current_hours <= hours_forward:
        # Predict Rf at this future time
        delta_rf = (asymptotic_rf - current_rf) * (1 - math.exp(-k * current_hours))
        rf_future = current_rf + delta_rf

        predictions.append({
            "hours": current_hours,
            "rf": rf_future,
            "rf_percent_of_max": rf_future / asymptotic_rf * 100 if asymptotic_rf > 0 else 0,
        })

        current_hours += time_step_hours

    return predictions


def calculate_cleaning_benefit(
    ua_clean: float,
    ua_current: float,
    heat_duty_required: float,
    energy_cost_per_kwh: float = 0.10,
    operating_hours_per_year: float = 8000
) -> Dict[str, float]:
    """
    Calculate the benefit of cleaning a fouled heat exchanger.

    Quantifies energy savings from restoring heat transfer capability.

    Args:
        ua_clean: Clean UA value (W/K).
        ua_current: Current fouled UA value (W/K).
        heat_duty_required: Required heat duty (W).
        energy_cost_per_kwh: Energy cost ($/kWh). Default 0.10.
        operating_hours_per_year: Annual operating hours. Default 8000.

    Returns:
        Dictionary with:
        - ua_improvement: UA increase from cleaning (W/K)
        - ua_improvement_percent: Percentage improvement
        - effectiveness_gain: Estimated effectiveness improvement
        - energy_savings_kwh_year: Annual energy savings (kWh)
        - cost_savings_year: Annual cost savings ($)
        - payback_estimate_hours: Estimated payback time if cleaning costs $5000

    Example:
        >>> result = calculate_cleaning_benefit(50000, 35000, 2000000, 0.12)
        >>> print(f"Annual savings: ${result['cost_savings_year']:.0f}")
    """
    if ua_clean <= 0 or ua_current <= 0:
        raise ValueError("UA values must be > 0")
    if ua_current > ua_clean:
        raise ValueError("Current UA cannot exceed clean UA")

    # UA improvement
    ua_improvement = ua_clean - ua_current
    ua_improvement_pct = ua_improvement / ua_clean * 100

    # Effectiveness improvement (approximation)
    # Assuming NTU proportional to UA
    eff_gain = ua_improvement_pct * 0.5  # Rough estimate

    # Energy loss due to fouling
    # Lower UA means either more energy input or lost production
    # Simplified model: energy loss proportional to UA reduction
    fouling_penalty = ua_improvement / ua_clean

    # Energy savings estimate
    # If fouling reduces capability, supplemental heating/cooling needed
    energy_loss_rate = heat_duty_required * fouling_penalty  # W
    energy_savings_kwh_year = energy_loss_rate * operating_hours_per_year / 1000

    # Cost savings
    cost_savings = energy_savings_kwh_year * energy_cost_per_kwh

    # Payback estimate (assume $5000 cleaning cost)
    typical_cleaning_cost = 5000
    payback_hours = (typical_cleaning_cost / cost_savings * operating_hours_per_year
                     if cost_savings > 0 else float('inf'))

    return {
        "ua_improvement": ua_improvement,
        "ua_improvement_percent": ua_improvement_pct,
        "effectiveness_gain": eff_gain,
        "energy_savings_kwh_year": energy_savings_kwh_year,
        "cost_savings_year": cost_savings,
        "payback_estimate_hours": payback_hours,
    }


def optimize_cleaning_schedule(
    ua_clean: float,
    fouling_rate: float,
    asymptotic_rf: float,
    area: float,
    heat_duty: float,
    cleaning_cost: float,
    energy_cost_per_kwh: float,
    operating_hours_per_year: float = 8000,
    min_ua_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Optimize cleaning interval for minimum total cost.

    Balances cleaning costs against energy losses from fouling.

    The optimal interval minimizes:
        Total Cost = Cleaning Cost / Interval + Energy Loss Cost

    Args:
        ua_clean: Clean UA value (W/K).
        fouling_rate: Fouling rate (m2-K/W per 1000 hours).
        asymptotic_rf: Maximum fouling resistance (m2-K/W).
        area: Heat transfer area (m2).
        heat_duty: Required heat duty (W).
        cleaning_cost: Cost per cleaning ($).
        energy_cost_per_kwh: Energy cost ($/kWh).
        operating_hours_per_year: Annual operating hours.
        min_ua_threshold: Minimum acceptable UA as fraction of clean.

    Returns:
        Dictionary with:
        - optimal_interval_hours: Optimal cleaning interval (hours)
        - optimal_interval_days: Optimal interval in days
        - cleanings_per_year: Number of cleanings per year
        - annual_cleaning_cost: Annual cleaning costs ($)
        - annual_energy_loss: Annual energy loss cost ($)
        - total_annual_cost: Total annual cost ($)
        - ua_at_cleaning: UA when cleaning is performed

    Example:
        >>> result = optimize_cleaning_schedule(
        ...     50000, 0.00005, 0.0006, 100, 2000000, 8000, 0.10
        ... )
        >>> print(f"Optimal interval: {result['optimal_interval_days']:.0f} days")
    """
    if ua_clean <= 0 or fouling_rate < 0 or area <= 0:
        raise ValueError("Invalid parameters")

    # Search for optimal interval
    best_interval = None
    best_total_cost = float('inf')

    # Search range: 500 to 15000 hours
    for interval in range(500, 15001, 100):
        # Calculate UA at end of interval
        rf_at_cleaning = calculate_fouling_resistance(
            interval, fouling_rate, 1.0, asymptotic_rf
        )
        ua_at_cleaning_result = calculate_ua_degradation(
            ua_clean, rf_at_cleaning / 2, rf_at_cleaning / 2, area
        )
        ua_at_cleaning = ua_at_cleaning_result["ua_fouled"]

        # Check threshold
        if ua_at_cleaning / ua_clean < min_ua_threshold:
            continue

        # Average UA over interval (integrate over time)
        # Simplified: use UA at 50% of interval
        rf_mid = calculate_fouling_resistance(
            interval / 2, fouling_rate, 1.0, asymptotic_rf
        )
        ua_mid_result = calculate_ua_degradation(
            ua_clean, rf_mid / 2, rf_mid / 2, area
        )
        ua_avg = (ua_clean + ua_mid_result["ua_fouled"]) / 2

        # Energy loss cost (proportional to UA reduction)
        ua_loss_fraction = (ua_clean - ua_avg) / ua_clean
        energy_loss_rate = heat_duty * ua_loss_fraction  # W
        energy_loss_kwh_year = energy_loss_rate * operating_hours_per_year / 1000
        energy_cost_year = energy_loss_kwh_year * energy_cost_per_kwh

        # Cleaning cost per year
        cleanings_per_year = operating_hours_per_year / interval
        cleaning_cost_year = cleanings_per_year * cleaning_cost

        # Total cost
        total_cost = energy_cost_year + cleaning_cost_year

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_interval = interval
            best_result = {
                "optimal_interval_hours": interval,
                "optimal_interval_days": interval / 24,
                "cleanings_per_year": cleanings_per_year,
                "annual_cleaning_cost": cleaning_cost_year,
                "annual_energy_loss": energy_cost_year,
                "total_annual_cost": total_cost,
                "ua_at_cleaning": ua_at_cleaning,
            }

    if best_interval is None:
        # Default if no valid interval found
        return {
            "optimal_interval_hours": 4000,
            "optimal_interval_days": 167,
            "cleanings_per_year": 2,
            "annual_cleaning_cost": cleaning_cost * 2,
            "annual_energy_loss": 0,
            "total_annual_cost": cleaning_cost * 2,
            "ua_at_cleaning": ua_clean * min_ua_threshold,
        }

    logger.info(
        f"Optimal cleaning: interval={best_interval}h, "
        f"cleanings/year={best_result['cleanings_per_year']:.1f}, "
        f"total_cost=${best_total_cost:.0f}/year"
    )

    return best_result


def calculate_next_cleaning_date(
    current_date: date,
    hours_since_cleaning: float,
    optimal_interval_hours: float,
    operating_hours_per_day: float = 24
) -> Dict[str, any]:
    """
    Calculate recommended next cleaning date.

    Args:
        current_date: Current date.
        hours_since_cleaning: Hours since last cleaning.
        optimal_interval_hours: Optimal cleaning interval (hours).
        operating_hours_per_day: Daily operating hours. Default 24.

    Returns:
        Dictionary with:
        - next_cleaning_date: Recommended cleaning date
        - days_until_cleaning: Days until cleaning
        - hours_until_cleaning: Hours until cleaning
        - utilization_percent: Current interval utilization

    Example:
        >>> result = calculate_next_cleaning_date(date.today(), 3000, 5000, 20)
        >>> print(f"Next cleaning: {result['next_cleaning_date']}")
    """
    if optimal_interval_hours <= 0:
        raise ValueError("Optimal interval must be > 0")
    if operating_hours_per_day <= 0 or operating_hours_per_day > 24:
        raise ValueError("Operating hours per day must be 0-24")

    # Hours remaining until cleaning needed
    hours_remaining = max(0, optimal_interval_hours - hours_since_cleaning)

    # Days remaining
    days_remaining = hours_remaining / operating_hours_per_day

    # Next cleaning date
    next_date = current_date + timedelta(days=int(days_remaining))

    # Utilization
    utilization = hours_since_cleaning / optimal_interval_hours * 100

    return {
        "next_cleaning_date": next_date,
        "days_until_cleaning": days_remaining,
        "hours_until_cleaning": hours_remaining,
        "utilization_percent": utilization,
    }


def analyze_fouling_history(
    ua_measurements: List[Tuple[float, float]],
    ua_clean: float
) -> Dict[str, float]:
    """
    Analyze historical UA measurements to determine fouling characteristics.

    Uses linear regression on ln(UA_clean - UA) vs time to estimate
    fouling rate and asymptotic behavior.

    Args:
        ua_measurements: List of (operating_hours, UA) tuples.
        ua_clean: Clean (design) UA value.

    Returns:
        Dictionary with:
        - estimated_fouling_rate: Fouling rate (m2-K/W per 1000h)
        - asymptotic_rf: Estimated maximum fouling resistance
        - r_squared: Correlation coefficient for fit
        - hours_to_critical: Hours until UA drops to 70% of clean

    Example:
        >>> measurements = [(0, 50000), (2000, 45000), (4000, 42000)]
        >>> result = analyze_fouling_history(measurements, 50000)
        >>> print(f"Fouling rate: {result['estimated_fouling_rate']:.6f}")
    """
    if len(ua_measurements) < 2:
        raise ValueError("Need at least 2 measurements")

    # Sort by time
    measurements = sorted(ua_measurements, key=lambda x: x[0])

    # Calculate fouling resistance at each point
    rf_data = []
    for hours, ua in measurements:
        if ua > 0 and ua < ua_clean:
            # Rf = 1/U_fouled - 1/U_clean (simplified for equal areas)
            rf = (1 / ua - 1 / ua_clean) * 1e6  # Scale for analysis
            rf_data.append((hours, rf))

    if len(rf_data) < 2:
        # Not enough valid data
        return {
            "estimated_fouling_rate": 0.0001,
            "asymptotic_rf": 0.001,
            "r_squared": 0,
            "hours_to_critical": float('inf'),
        }

    # Linear regression on Rf vs time (for early asymptotic region)
    n = len(rf_data)
    sum_t = sum(t for t, _ in rf_data)
    sum_rf = sum(rf for _, rf in rf_data)
    sum_t_rf = sum(t * rf for t, rf in rf_data)
    sum_t2 = sum(t ** 2 for t, _ in rf_data)

    # Slope = fouling rate
    denom = n * sum_t2 - sum_t ** 2
    if abs(denom) < 1e-10:
        slope = 0
    else:
        slope = (n * sum_t_rf - sum_t * sum_rf) / denom

    # Convert back to proper units
    fouling_rate = slope * 1e-6 * 1000  # m2-K/W per 1000 hours

    # Estimate asymptotic Rf (extrapolate)
    max_hours = max(t for t, _ in rf_data)
    asymptotic_rf = max(rf for _, rf in rf_data) * 2 * 1e-6

    # R-squared
    mean_rf = sum_rf / n
    ss_tot = sum((rf - mean_rf) ** 2 for _, rf in rf_data)
    intercept = (sum_rf - slope * sum_t) / n
    ss_res = sum((rf - (slope * t + intercept)) ** 2 for t, rf in rf_data)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Hours to critical (70% of clean UA)
    critical_rf = (1 / (0.7 * ua_clean) - 1 / ua_clean)
    hours_to_critical = critical_rf / fouling_rate * 1000 if fouling_rate > 0 else float('inf')

    return {
        "estimated_fouling_rate": abs(fouling_rate),
        "asymptotic_rf": abs(asymptotic_rf),
        "r_squared": max(0, min(1, r_squared)),
        "hours_to_critical": max(0, hours_to_critical),
    }


def generate_fouling_report(
    ua_clean: float,
    ua_current: float,
    hours_since_cleaning: float,
    area: float,
    hot_side_fluid: str = "cooling_tower_treated",
    cold_side_fluid: str = "process_fluids"
) -> Dict[str, any]:
    """
    Generate comprehensive fouling analysis report.

    Combines multiple analyses into a single report suitable for
    maintenance planning.

    Args:
        ua_clean: Clean UA value (W/K).
        ua_current: Current UA value (W/K).
        hours_since_cleaning: Hours since last cleaning.
        area: Heat transfer area (m2).
        hot_side_fluid: Hot side fluid type.
        cold_side_fluid: Cold side fluid type.

    Returns:
        Comprehensive fouling report dictionary.

    Example:
        >>> report = generate_fouling_report(50000, 42000, 3000, 100)
        >>> print(f"Status: {report['fouling_status']}")
    """
    # Get TEMA fouling resistances
    try:
        rf_hot_tema = get_tema_fouling_resistance("water", hot_side_fluid)
    except ValueError:
        rf_hot_tema = 0.000352  # Default

    try:
        rf_cold_tema = get_tema_fouling_resistance("chemicals", cold_side_fluid)
    except ValueError:
        rf_cold_tema = 0.000352  # Default

    # Calculate current fouling resistance
    u_clean = ua_clean / area
    u_current = ua_current / area

    if u_current > 0:
        total_rf_current = 1 / u_current - 1 / u_clean
    else:
        total_rf_current = 0

    # Performance metrics
    ua_ratio = ua_current / ua_clean
    performance_loss = (1 - ua_ratio) * 100

    # Status determination
    if ua_ratio >= 0.90:
        status = "GOOD"
        urgency = "LOW"
    elif ua_ratio >= 0.80:
        status = "FAIR"
        urgency = "MEDIUM"
    elif ua_ratio >= 0.70:
        status = "POOR"
        urgency = "HIGH"
    else:
        status = "CRITICAL"
        urgency = "IMMEDIATE"

    # Fouling rate estimate
    if hours_since_cleaning > 0:
        observed_rate = total_rf_current / hours_since_cleaning * 1000
    else:
        observed_rate = rf_hot_tema + rf_cold_tema

    # Predicted time to critical
    critical_rf = 1 / (0.7 * u_clean) - 1 / u_clean
    if observed_rate > 0:
        hours_to_critical = (critical_rf - total_rf_current) / observed_rate * 1000
    else:
        hours_to_critical = float('inf')

    return {
        "fouling_status": status,
        "maintenance_urgency": urgency,
        "ua_clean": ua_clean,
        "ua_current": ua_current,
        "ua_ratio": ua_ratio,
        "performance_loss_percent": performance_loss,
        "total_fouling_resistance": total_rf_current,
        "tema_rf_hot": rf_hot_tema,
        "tema_rf_cold": rf_cold_tema,
        "observed_fouling_rate": observed_rate,
        "hours_since_cleaning": hours_since_cleaning,
        "hours_to_critical": hours_to_critical,
        "recommended_action": _get_recommended_action(status),
    }


def _get_recommended_action(status: str) -> str:
    """Get recommended action based on fouling status."""
    actions = {
        "GOOD": "Continue normal monitoring. No immediate action required.",
        "FAIR": "Schedule cleaning within next planned shutdown.",
        "POOR": "Plan cleaning within 30 days. Monitor closely.",
        "CRITICAL": "Immediate cleaning required. Assess process impact.",
    }
    return actions.get(status, "Evaluate fouling condition.")
