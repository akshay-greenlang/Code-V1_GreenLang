"""
Water Chemistry Calculations for Boiler Water Treatment

This module implements ASME/ABMA compliant water chemistry calculations
for boiler water treatment optimization.

All formulas are deterministic physics/chemistry-based calculations
following zero-hallucination principles. No ML/LLM in calculation path.

Standards Reference:
    - ASME Boiler and Pressure Vessel Code Section VII
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - EPRI Water Chemistry Guidelines for Fossil Plants
    - NACE SP0590 Standard Practice

Example:
    >>> limits = get_asme_water_limits(600)  # 600 psig
    >>> status = check_chemistry_compliance(chemistry_data, limits)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ASME/ABMA Boiler Water Chemistry Limits by Pressure Class
# Reference: ABMA Guidelines for Industrial Water Treatment (Table 4)
ASME_WATER_LIMITS: Dict[str, Dict[str, Dict[str, float]]] = {
    "low_pressure": {  # 0-300 psig
        "conductivity_us_cm": {"max": 7000, "target": 3500},
        "tds_ppm": {"max": 3500, "target": 2000},
        "alkalinity_ppm": {"min": 100, "max": 700, "target": 350},
        "silica_ppm": {"max": 150, "target": 75},
        "hardness_ppm": {"max": 0.5, "target": 0},
        "iron_ppm": {"max": 2.0, "target": 0.5},
        "copper_ppm": {"max": 0.5, "target": 0.1},
        "dissolved_oxygen_ppb": {"max": 20, "target": 7},
        "phosphate_ppm": {"min": 30, "max": 60, "target": 45},
        "ph": {"min": 10.5, "max": 11.5, "target": 11.0},
    },
    "medium_pressure": {  # 300-600 psig
        "conductivity_us_cm": {"max": 5500, "target": 2750},
        "tds_ppm": {"max": 2500, "target": 1500},
        "alkalinity_ppm": {"min": 100, "max": 500, "target": 300},
        "silica_ppm": {"max": 90, "target": 45},
        "hardness_ppm": {"max": 0.3, "target": 0},
        "iron_ppm": {"max": 1.0, "target": 0.3},
        "copper_ppm": {"max": 0.25, "target": 0.05},
        "dissolved_oxygen_ppb": {"max": 10, "target": 5},
        "phosphate_ppm": {"min": 20, "max": 40, "target": 30},
        "ph": {"min": 10.0, "max": 11.0, "target": 10.5},
    },
    "high_pressure": {  # 600-900 psig
        "conductivity_us_cm": {"max": 3000, "target": 1500},
        "tds_ppm": {"max": 1500, "target": 750},
        "alkalinity_ppm": {"min": 50, "max": 300, "target": 150},
        "silica_ppm": {"max": 30, "target": 15},
        "hardness_ppm": {"max": 0.2, "target": 0},
        "iron_ppm": {"max": 0.5, "target": 0.1},
        "copper_ppm": {"max": 0.1, "target": 0.02},
        "dissolved_oxygen_ppb": {"max": 7, "target": 3},
        "phosphate_ppm": {"min": 5, "max": 15, "target": 10},
        "ph": {"min": 9.5, "max": 10.5, "target": 10.0},
    },
    "very_high_pressure": {  # 900-1500 psig
        "conductivity_us_cm": {"max": 1000, "target": 500},
        "tds_ppm": {"max": 500, "target": 250},
        "alkalinity_ppm": {"min": 25, "max": 150, "target": 75},
        "silica_ppm": {"max": 10, "target": 5},
        "hardness_ppm": {"max": 0.1, "target": 0},
        "iron_ppm": {"max": 0.2, "target": 0.05},
        "copper_ppm": {"max": 0.05, "target": 0.01},
        "dissolved_oxygen_ppb": {"max": 5, "target": 2},
        "phosphate_ppm": {"min": 2, "max": 8, "target": 5},
        "ph": {"min": 9.0, "max": 10.0, "target": 9.5},
    },
    "supercritical": {  # > 1500 psig
        "conductivity_us_cm": {"max": 200, "target": 100},
        "tds_ppm": {"max": 100, "target": 50},
        "alkalinity_ppm": {"min": 0, "max": 50, "target": 25},
        "silica_ppm": {"max": 2, "target": 1},
        "hardness_ppm": {"max": 0.05, "target": 0},
        "iron_ppm": {"max": 0.1, "target": 0.02},
        "copper_ppm": {"max": 0.02, "target": 0.005},
        "dissolved_oxygen_ppb": {"max": 2, "target": 1},
        "phosphate_ppm": {"min": 0.5, "max": 2, "target": 1},
        "ph": {"min": 8.5, "max": 9.5, "target": 9.0},
    },
}


def determine_pressure_class(operating_pressure_psig: float) -> str:
    """
    Determine ASME pressure class from operating pressure.

    Args:
        operating_pressure_psig: Operating pressure in PSIG.

    Returns:
        Pressure class string for limit lookup.

    Example:
        >>> pressure_class = determine_pressure_class(450)
        >>> print(pressure_class)
        medium_pressure
    """
    if operating_pressure_psig < 0:
        raise ValueError(f"Pressure must be >= 0, got {operating_pressure_psig}")

    if operating_pressure_psig < 300:
        return "low_pressure"
    elif operating_pressure_psig < 600:
        return "medium_pressure"
    elif operating_pressure_psig < 900:
        return "high_pressure"
    elif operating_pressure_psig < 1500:
        return "very_high_pressure"
    else:
        return "supercritical"


def get_asme_water_limits(
    operating_pressure_psig: float,
    pressure_class: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Get ASME/ABMA water chemistry limits for given pressure.

    Args:
        operating_pressure_psig: Operating pressure in PSIG.
        pressure_class: Optional override for pressure class.

    Returns:
        Dictionary of parameter limits with min/max/target values.

    Raises:
        ValueError: If pressure class is unknown.

    Example:
        >>> limits = get_asme_water_limits(600)
        >>> print(limits['silica_ppm']['max'])
        30
    """
    if pressure_class is None:
        pressure_class = determine_pressure_class(operating_pressure_psig)

    if pressure_class not in ASME_WATER_LIMITS:
        raise ValueError(f"Unknown pressure class: {pressure_class}")

    logger.debug(f"Using {pressure_class} limits for {operating_pressure_psig} psig")

    return ASME_WATER_LIMITS[pressure_class]


def check_parameter_compliance(
    parameter_name: str,
    measured_value: float,
    limits: Dict[str, float],
    warning_threshold: float = 0.9,
    critical_threshold: float = 1.1
) -> Dict[str, Any]:
    """
    Check single parameter compliance against limits.

    Args:
        parameter_name: Name of the chemistry parameter.
        measured_value: Measured value.
        limits: Dictionary with min/max/target values.
        warning_threshold: Fraction of limit for warning (default 90%).
        critical_threshold: Fraction of limit for critical (default 110%).

    Returns:
        Dictionary with compliance result including status and deviation.

    Example:
        >>> result = check_parameter_compliance(
        ...     'silica_ppm', 25, {'max': 30, 'target': 15}
        ... )
        >>> print(result['status'])
        WARNING
    """
    max_limit = limits.get('max')
    min_limit = limits.get('min')
    target = limits.get('target')

    # Determine status based on limits
    status = "COMPLIANT"
    deviation_percent = 0.0
    recommendation = None

    # Check against max limit
    if max_limit is not None and max_limit > 0:
        ratio_to_max = measured_value / max_limit
        deviation_percent = (ratio_to_max - 1.0) * 100

        if ratio_to_max >= critical_threshold:
            status = "CRITICAL"
            recommendation = f"Reduce {parameter_name} immediately - exceeds ASME limit"
        elif ratio_to_max >= 1.0:
            status = "NON_COMPLIANT"
            recommendation = f"Reduce {parameter_name} to meet ASME limit"
        elif ratio_to_max >= warning_threshold:
            status = "WARNING"
            recommendation = f"Monitor {parameter_name} - approaching limit"

    # Check against min limit (if status still compliant)
    if min_limit is not None and status == "COMPLIANT":
        if measured_value < min_limit:
            ratio_to_min = measured_value / min_limit if min_limit > 0 else 0
            deviation_percent = (1.0 - ratio_to_min) * -100
            status = "NON_COMPLIANT"
            recommendation = f"Increase {parameter_name} to meet minimum ASME requirement"
        elif measured_value < min_limit * (2 - warning_threshold):
            status = "WARNING"
            recommendation = f"Monitor {parameter_name} - below optimal range"

    # Calculate deviation from target
    if target is not None and target > 0 and deviation_percent == 0:
        deviation_percent = ((measured_value - target) / target) * 100

    return {
        "parameter_name": parameter_name,
        "measured_value": measured_value,
        "min_limit": min_limit,
        "max_limit": max_limit,
        "target_value": target,
        "status": status,
        "deviation_percent": round(deviation_percent, 2),
        "recommendation": recommendation,
    }


def check_chemistry_compliance(
    chemistry_data: Dict[str, float],
    pressure_psig: float,
    warning_threshold: float = 0.9
) -> Tuple[List[Dict[str, Any]], str, float]:
    """
    Check all water chemistry parameters against ASME/ABMA limits.

    Args:
        chemistry_data: Dictionary of chemistry parameter measurements.
        pressure_psig: Operating pressure in PSIG.
        warning_threshold: Threshold for warning status.

    Returns:
        Tuple of (results_list, overall_status, compliance_score).

    Example:
        >>> chemistry = {'conductivity_us_cm': 3000, 'silica_ppm': 25}
        >>> results, status, score = check_chemistry_compliance(chemistry, 600)
    """
    limits = get_asme_water_limits(pressure_psig)
    results = []
    compliance_count = 0
    total_checks = 0

    # Map input parameter names to limits keys
    parameter_mapping = {
        'conductivity_us_cm': 'conductivity_us_cm',
        'tds_ppm': 'tds_ppm',
        'alkalinity_ppm_caco3': 'alkalinity_ppm',
        'alkalinity_ppm': 'alkalinity_ppm',
        'silica_ppm': 'silica_ppm',
        'total_hardness_ppm': 'hardness_ppm',
        'hardness_ppm': 'hardness_ppm',
        'iron_ppm': 'iron_ppm',
        'copper_ppm': 'copper_ppm',
        'dissolved_oxygen_ppb': 'dissolved_oxygen_ppb',
        'phosphate_ppm': 'phosphate_ppm',
        'ph': 'ph',
    }

    # Unit mapping for display
    unit_mapping = {
        'conductivity_us_cm': 'uS/cm',
        'tds_ppm': 'ppm',
        'alkalinity_ppm': 'ppm CaCO3',
        'silica_ppm': 'ppm',
        'hardness_ppm': 'ppm CaCO3',
        'iron_ppm': 'ppm',
        'copper_ppm': 'ppm',
        'dissolved_oxygen_ppb': 'ppb',
        'phosphate_ppm': 'ppm',
        'ph': 'pH units',
    }

    for input_param, limit_key in parameter_mapping.items():
        if input_param in chemistry_data and limit_key in limits:
            measured = chemistry_data[input_param]
            param_limits = limits[limit_key]

            result = check_parameter_compliance(
                input_param,
                measured,
                param_limits,
                warning_threshold
            )

            # Add unit
            result['unit'] = unit_mapping.get(limit_key, '')

            results.append(result)
            total_checks += 1

            if result['status'] in ['COMPLIANT', 'WARNING']:
                compliance_count += 1

    # Calculate overall status
    statuses = [r['status'] for r in results]

    if 'CRITICAL' in statuses:
        overall_status = "CRITICAL"
    elif 'NON_COMPLIANT' in statuses:
        overall_status = "NON_COMPLIANT"
    elif 'WARNING' in statuses:
        overall_status = "WARNING"
    else:
        overall_status = "COMPLIANT"

    # Calculate compliance score (0-100)
    compliance_score = (compliance_count / total_checks * 100) if total_checks > 0 else 100

    logger.info(
        f"Chemistry compliance: {overall_status}, score={compliance_score:.1f}%, "
        f"{compliance_count}/{total_checks} parameters compliant"
    )

    return results, overall_status, compliance_score


def calculate_cycles_of_concentration(
    feedwater_conductivity: float,
    blowdown_conductivity: float
) -> float:
    """
    Calculate cycles of concentration from conductivity.

    Cycles of concentration (COC) indicates how many times the
    dissolved solids have been concentrated in the boiler water.

    Formula:
        COC = Boiler Water Conductivity / Feedwater Conductivity

    Args:
        feedwater_conductivity: Feedwater conductivity in uS/cm.
        blowdown_conductivity: Boiler water/blowdown conductivity in uS/cm.

    Returns:
        Cycles of concentration (typically 1-20).

    Raises:
        ValueError: If feedwater conductivity is zero or negative.

    Example:
        >>> coc = calculate_cycles_of_concentration(200, 3000)
        >>> print(f"COC: {coc:.1f}")
        COC: 15.0
    """
    if feedwater_conductivity <= 0:
        raise ValueError(
            f"Feedwater conductivity must be > 0, got {feedwater_conductivity}"
        )

    if blowdown_conductivity < feedwater_conductivity:
        logger.warning(
            f"Blowdown conductivity ({blowdown_conductivity}) < feedwater ({feedwater_conductivity})"
        )
        return 1.0

    cycles = blowdown_conductivity / feedwater_conductivity

    logger.debug(f"COC calculation: {blowdown_conductivity}/{feedwater_conductivity} = {cycles:.2f}")

    return cycles


def calculate_max_cycles_by_silica(
    feedwater_silica_ppm: float,
    max_boiler_silica_ppm: float
) -> float:
    """
    Calculate maximum allowable cycles limited by silica.

    Silica is a critical limit for boiler water due to carryover
    risk and deposition in turbines.

    Formula:
        Max COC = Max Boiler Silica / Feedwater Silica

    Args:
        feedwater_silica_ppm: Feedwater silica in ppm.
        max_boiler_silica_ppm: Maximum allowable boiler silica per ASME.

    Returns:
        Maximum cycles of concentration limited by silica.

    Example:
        >>> max_coc = calculate_max_cycles_by_silica(5, 30)
        >>> print(f"Max COC by silica: {max_coc:.1f}")
        Max COC by silica: 6.0
    """
    if feedwater_silica_ppm <= 0:
        logger.warning("Feedwater silica is zero - using high COC limit")
        return 50.0  # Practical upper limit

    max_cycles = max_boiler_silica_ppm / feedwater_silica_ppm

    logger.debug(f"Max COC by silica: {max_boiler_silica_ppm}/{feedwater_silica_ppm} = {max_cycles:.2f}")

    return max_cycles


def calculate_max_cycles_by_alkalinity(
    feedwater_alkalinity_ppm: float,
    max_boiler_alkalinity_ppm: float
) -> float:
    """
    Calculate maximum allowable cycles limited by alkalinity.

    High alkalinity can cause caustic embrittlement and foaming.

    Formula:
        Max COC = Max Boiler Alkalinity / Feedwater Alkalinity

    Args:
        feedwater_alkalinity_ppm: Feedwater alkalinity in ppm CaCO3.
        max_boiler_alkalinity_ppm: Maximum allowable per ASME.

    Returns:
        Maximum cycles limited by alkalinity.

    Example:
        >>> max_coc = calculate_max_cycles_by_alkalinity(50, 300)
        >>> print(f"Max COC by alkalinity: {max_coc:.1f}")
        Max COC by alkalinity: 6.0
    """
    if feedwater_alkalinity_ppm <= 0:
        logger.warning("Feedwater alkalinity is zero - using high COC limit")
        return 50.0

    max_cycles = max_boiler_alkalinity_ppm / feedwater_alkalinity_ppm

    logger.debug(f"Max COC by alkalinity: {max_boiler_alkalinity_ppm}/{feedwater_alkalinity_ppm} = {max_cycles:.2f}")

    return max_cycles


def calculate_max_cycles_by_conductivity(
    feedwater_conductivity: float,
    max_boiler_conductivity: float
) -> float:
    """
    Calculate maximum allowable cycles limited by conductivity/TDS.

    Formula:
        Max COC = Max Boiler Conductivity / Feedwater Conductivity

    Args:
        feedwater_conductivity: Feedwater conductivity in uS/cm.
        max_boiler_conductivity: Maximum allowable per ASME.

    Returns:
        Maximum cycles limited by conductivity.

    Example:
        >>> max_coc = calculate_max_cycles_by_conductivity(200, 3000)
        >>> print(f"Max COC by conductivity: {max_coc:.1f}")
        Max COC by conductivity: 15.0
    """
    if feedwater_conductivity <= 0:
        raise ValueError(
            f"Feedwater conductivity must be > 0, got {feedwater_conductivity}"
        )

    max_cycles = max_boiler_conductivity / feedwater_conductivity

    logger.debug(f"Max COC by conductivity: {max_boiler_conductivity}/{feedwater_conductivity} = {max_cycles:.2f}")

    return max_cycles


def calculate_optimal_cycles(
    feedwater_silica: float,
    feedwater_alkalinity: float,
    feedwater_conductivity: float,
    pressure_psig: float
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate optimal cycles of concentration.

    The optimal COC is the minimum of all limiting factors,
    with a practical upper limit of 20.

    Args:
        feedwater_silica: Feedwater silica in ppm.
        feedwater_alkalinity: Feedwater alkalinity in ppm CaCO3.
        feedwater_conductivity: Feedwater conductivity in uS/cm.
        pressure_psig: Operating pressure in PSIG.

    Returns:
        Tuple of (optimal_cycles, limit_breakdown dict).

    Example:
        >>> optimal, limits = calculate_optimal_cycles(5, 50, 200, 600)
        >>> print(f"Optimal COC: {optimal:.1f}")
    """
    limits = get_asme_water_limits(pressure_psig)

    # Calculate max cycles by each limiting factor
    max_by_silica = calculate_max_cycles_by_silica(
        feedwater_silica,
        limits['silica_ppm']['max']
    )

    max_by_alkalinity = calculate_max_cycles_by_alkalinity(
        feedwater_alkalinity,
        limits['alkalinity_ppm']['max']
    )

    max_by_conductivity = calculate_max_cycles_by_conductivity(
        feedwater_conductivity,
        limits['conductivity_us_cm']['max']
    )

    # Optimal is minimum of all limits
    # Apply practical limits: min 3, max 20
    optimal = min(max_by_silica, max_by_alkalinity, max_by_conductivity)
    optimal = max(3.0, min(20.0, optimal))  # Practical bounds

    limit_breakdown = {
        'max_by_silica': round(max_by_silica, 2),
        'max_by_alkalinity': round(max_by_alkalinity, 2),
        'max_by_conductivity': round(max_by_conductivity, 2),
        'optimal': round(optimal, 2),
        'limiting_factor': 'silica' if max_by_silica == min(max_by_silica, max_by_alkalinity, max_by_conductivity)
                          else 'alkalinity' if max_by_alkalinity == min(max_by_silica, max_by_alkalinity, max_by_conductivity)
                          else 'conductivity'
    }

    logger.info(
        f"Optimal COC: {optimal:.1f} (limited by {limit_breakdown['limiting_factor']})"
    )

    return optimal, limit_breakdown


def calculate_tds_from_conductivity(
    conductivity_us_cm: float,
    conversion_factor: float = 0.65
) -> float:
    """
    Estimate TDS from conductivity measurement.

    Formula:
        TDS (ppm) = Conductivity (uS/cm) * Conversion Factor

    The conversion factor typically ranges from 0.5 to 0.7 depending
    on the ionic composition of the water.

    Args:
        conductivity_us_cm: Conductivity in microsiemens/cm.
        conversion_factor: TDS/conductivity ratio (default 0.65).

    Returns:
        Estimated TDS in ppm.

    Example:
        >>> tds = calculate_tds_from_conductivity(3000)
        >>> print(f"TDS: {tds:.0f} ppm")
        TDS: 1950 ppm
    """
    if conductivity_us_cm < 0:
        raise ValueError(f"Conductivity must be >= 0, got {conductivity_us_cm}")

    tds = conductivity_us_cm * conversion_factor

    return tds


def analyze_chemistry_trends(
    chemistry_history: List[Dict[str, float]],
    time_interval_hours: float
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze trends in chemistry parameters.

    Uses linear regression to determine trend direction
    and predict future values.

    Args:
        chemistry_history: List of chemistry reading dictionaries
            (most recent last).
        time_interval_hours: Hours between readings.

    Returns:
        Dictionary mapping parameter name to trend analysis.

    Example:
        >>> history = [
        ...     {'silica_ppm': 20, 'ph': 10.5},
        ...     {'silica_ppm': 22, 'ph': 10.4},
        ...     {'silica_ppm': 25, 'ph': 10.3},
        ... ]
        >>> trends = analyze_chemistry_trends(history, 4)
    """
    if len(chemistry_history) < 2:
        return {}

    trends = {}
    n = len(chemistry_history)

    # Get all parameter keys from first reading
    if not chemistry_history:
        return {}

    parameters = chemistry_history[0].keys()

    for param in parameters:
        # Extract values for this parameter
        values = []
        for reading in chemistry_history:
            if param in reading:
                values.append(reading[param])

        if len(values) < 2:
            continue

        # Calculate linear regression
        times = [i * time_interval_hours for i in range(len(values))]

        mean_t = sum(times) / len(times)
        mean_v = sum(values) / len(values)

        numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
        denominator = sum((t - mean_t) ** 2 for t in times)

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Predict value in 24 hours
        current_value = values[-1]
        predicted_24h = current_value + (slope * 24)

        # Calculate confidence based on R-squared
        ss_res = sum((v - (mean_v + slope * (t - mean_t))) ** 2 for t, v in zip(times, values))
        ss_tot = sum((v - mean_v) ** 2 for v in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))

        trends[param] = {
            'trend_direction': direction,
            'rate_of_change_per_hour': round(slope, 4),
            'predicted_value_24h': round(predicted_24h, 2),
            'confidence_score': round(confidence, 3),
            'current_value': current_value,
        }

    logger.debug(f"Analyzed trends for {len(trends)} parameters")

    return trends


def estimate_silica_solubility(
    temperature_c: float,
    ph: float
) -> float:
    """
    Estimate silica solubility at given temperature and pH.

    Silica solubility increases with temperature and pH.
    This affects carryover potential.

    Reference: EPRI Guidelines

    Args:
        temperature_c: Water temperature in Celsius.
        ph: Water pH value.

    Returns:
        Estimated silica solubility in ppm.

    Example:
        >>> solubility = estimate_silica_solubility(250, 10.0)
        >>> print(f"SiO2 solubility: {solubility:.0f} ppm")
    """
    # Base solubility correlation (empirical)
    # Solubility increases exponentially with temperature
    base_solubility = 100 * math.exp(0.01 * temperature_c)

    # pH effect (higher pH increases solubility)
    ph_factor = 1 + 0.1 * (ph - 7)

    solubility = base_solubility * ph_factor

    # Cap at practical limit
    return min(500, max(10, solubility))


def calculate_caustic_concentration(
    alkalinity_ppm: float,
    ph: float
) -> float:
    """
    Estimate free caustic (NaOH) concentration from alkalinity and pH.

    High free caustic can cause caustic embrittlement.

    Reference: ABMA Guidelines

    Args:
        alkalinity_ppm: Total alkalinity as ppm CaCO3.
        ph: Boiler water pH.

    Returns:
        Estimated free NaOH in ppm.

    Example:
        >>> naoh = calculate_caustic_concentration(300, 11.0)
        >>> print(f"Free NaOH: {naoh:.1f} ppm")
    """
    # At high pH (>11), most alkalinity is in hydroxide form
    # Conversion factor: 1 ppm CaCO3 = 0.8 ppm NaOH (as CaCO3)

    if ph < 10:
        # Below pH 10, most alkalinity is carbonate/bicarbonate
        caustic_fraction = 0.1
    elif ph < 11:
        # pH 10-11, mixed hydroxide/carbonate
        caustic_fraction = 0.3 + (ph - 10) * 0.4
    else:
        # pH > 11, mostly hydroxide
        caustic_fraction = 0.7 + min(0.3, (ph - 11) * 0.15)

    free_naoh = alkalinity_ppm * caustic_fraction * 0.8

    return free_naoh
