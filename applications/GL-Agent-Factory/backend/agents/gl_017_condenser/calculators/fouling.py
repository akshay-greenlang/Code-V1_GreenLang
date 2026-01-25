"""
Fouling Analysis and Prediction Calculator

This module implements fouling analysis, trend prediction, and cleaning
schedule optimization for steam surface condensers.

The fouling model uses deterministic physics-based calculations combined
with statistical trend analysis. ML model predictions are bounded by
physics constraints to ensure zero-hallucination.

All formulas are from standard references:
- HEI Standards for Steam Surface Condensers
- TEMA Standards for Heat Exchangers
- Kern's Process Heat Transfer
- EPRI Condenser Performance Guidelines

Zero-hallucination approach:
- Fouling resistance calculations use deterministic formulas
- Trend projections use statistical regression (not ML inference)
- All predictions are bounded by physical limits

Example:
    >>> fouling_rate = calculate_fouling_rate(
    ...     u_clean=3500, u_current=2800, operating_hours=8760
    ... )
    >>> print(f"Fouling rate: {fouling_rate:.6f} m2K/W per 1000h")
"""

import math
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)


# Standard fouling resistances (m2-K/W)
# Reference: TEMA Standards, Table RGP-T-2.4
STANDARD_FOULING_RESISTANCES = {
    "seawater": {
        "below_50c": 0.000088,
        "above_50c": 0.000176,
    },
    "brackish_water": {
        "below_50c": 0.000176,
        "above_50c": 0.000352,
    },
    "river_water": {
        "clean": 0.000176,
        "typical": 0.000352,
        "dirty": 0.000528,
    },
    "cooling_tower_treated": {
        "below_50c": 0.000088,
        "above_50c": 0.000176,
    },
    "cooling_tower_untreated": {
        "below_50c": 0.000528,
        "above_50c": 0.000704,
    },
    "city_water": 0.000088,
    "distilled_water": 0.000044,
}

# Fouling growth model parameters
# Reference: Kern, EPRI guidelines
FOULING_MODEL_PARAMS = {
    "asymptotic": {
        "k_deposit": 0.0001,  # Deposition rate constant
        "k_removal": 0.01,  # Removal rate constant
    },
    "linear": {
        "rate_clean_water": 0.00001,  # m2K/W per 1000 hours
        "rate_typical": 0.00003,
        "rate_fouled": 0.00006,
    },
}

# Cleaning effectiveness factors
CLEANING_EFFECTIVENESS = {
    "mechanical_ball": 0.95,  # Sponge ball cleaning
    "mechanical_brush": 0.90,  # Brush cleaning
    "chemical_acid": 0.85,  # Acid cleaning
    "chemical_biodispersant": 0.75,  # Biodispersant treatment
    "high_pressure_water": 0.92,  # HP water lance
    "backwash": 0.60,  # Simple backwash
}


def calculate_fouling_resistance(
    u_clean: float,
    u_fouled: float
) -> float:
    """
    Calculate fouling resistance from heat transfer coefficients.

    Formula:
        R_fouling = (1/U_fouled) - (1/U_clean)

    This is the standard TEMA/HEI definition of fouling resistance.

    Reference: TEMA Standards, HEI Standards

    Args:
        u_clean: Clean heat transfer coefficient (W/m2-K).
        u_fouled: Fouled heat transfer coefficient (W/m2-K).

    Returns:
        Fouling resistance in m2-K/W.

    Raises:
        ValueError: If coefficients are invalid.

    Example:
        >>> r_f = calculate_fouling_resistance(3500, 2800)
        >>> print(f"Fouling resistance: {r_f:.6f} m2K/W")
    """
    if u_clean <= 0:
        raise ValueError(f"U_clean must be > 0, got {u_clean}")
    if u_fouled <= 0:
        raise ValueError(f"U_fouled must be > 0, got {u_fouled}")
    if u_fouled > u_clean:
        logger.warning(
            f"U_fouled ({u_fouled}) > U_clean ({u_clean}). "
            f"This indicates measurement error or improved conditions."
        )
        return 0.0

    r_fouling = (1 / u_fouled) - (1 / u_clean)

    return max(0.0, r_fouling)


def calculate_fouling_rate(
    u_clean: float,
    u_current: float,
    operating_hours: float
) -> float:
    """
    Calculate average fouling rate over operating period.

    Formula:
        Fouling_rate = R_fouling / (operating_hours / 1000)

    Where R_fouling = (1/U_current) - (1/U_clean)

    Reference: EPRI guidelines

    Args:
        u_clean: Clean heat transfer coefficient (W/m2-K).
        u_current: Current heat transfer coefficient (W/m2-K).
        operating_hours: Hours since last cleaning.

    Returns:
        Fouling rate in m2-K/W per 1000 operating hours.

    Example:
        >>> rate = calculate_fouling_rate(3500, 2800, 8760)
        >>> print(f"Fouling rate: {rate:.6f} m2K/W per 1000h")
    """
    if operating_hours <= 0:
        raise ValueError(f"Operating hours must be > 0, got {operating_hours}")

    r_fouling = calculate_fouling_resistance(u_clean, u_current)
    fouling_rate = r_fouling / (operating_hours / 1000)

    return fouling_rate


def predict_fouling_linear(
    current_fouling_resistance: float,
    fouling_rate: float,
    hours_ahead: float
) -> float:
    """
    Predict future fouling using linear model.

    Formula:
        R_future = R_current + fouling_rate * (hours_ahead / 1000)

    This is a simple linear extrapolation suitable for short-term
    predictions when fouling is in the linear growth phase.

    Reference: Standard trend analysis

    Args:
        current_fouling_resistance: Current fouling resistance (m2-K/W).
        fouling_rate: Fouling rate (m2-K/W per 1000 hours).
        hours_ahead: Hours to project into future.

    Returns:
        Predicted fouling resistance in m2-K/W.

    Example:
        >>> r_future = predict_fouling_linear(0.0001, 0.00003, 2000)
        >>> print(f"Predicted R_fouling: {r_future:.6f} m2K/W")
    """
    r_future = current_fouling_resistance + fouling_rate * (hours_ahead / 1000)

    # Physical upper bound (can't exceed typical maximum)
    max_fouling = 0.001  # m2-K/W - severe fouling limit
    return min(r_future, max_fouling)


def predict_fouling_asymptotic(
    time_hours: float,
    r_fouling_max: float,
    time_constant_hours: float
) -> float:
    """
    Predict fouling using asymptotic model.

    Formula (Kern asymptotic model):
        R(t) = R_max * (1 - exp(-t/tau))

    Where:
        R_max = asymptotic maximum fouling resistance
        tau = time constant for fouling growth

    This model accounts for the equilibrium between deposition
    and removal that limits maximum fouling.

    Reference: Kern's Process Heat Transfer

    Args:
        time_hours: Time since cleaning in hours.
        r_fouling_max: Asymptotic maximum fouling (m2-K/W).
        time_constant_hours: Time constant for fouling (hours).

    Returns:
        Predicted fouling resistance in m2-K/W.

    Example:
        >>> r = predict_fouling_asymptotic(8760, 0.0003, 5000)
        >>> print(f"Predicted R_fouling: {r:.6f} m2K/W")
    """
    if time_hours < 0:
        raise ValueError(f"Time must be >= 0, got {time_hours}")
    if time_constant_hours <= 0:
        raise ValueError(
            f"Time constant must be > 0, got {time_constant_hours}"
        )

    # Asymptotic model
    r_fouling = r_fouling_max * (1 - math.exp(-time_hours / time_constant_hours))

    return r_fouling


def estimate_fouling_parameters_from_history(
    fouling_history: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Estimate fouling model parameters from historical data.

    Uses linear regression on fouling resistance vs time data
    to determine fouling rate and intercept.

    This is a deterministic statistical calculation, not ML inference.

    Reference: Statistical methods for trend analysis

    Args:
        fouling_history: List of (operating_hours, fouling_resistance) tuples.
            Must have at least 3 data points.

    Returns:
        Dictionary with:
        - fouling_rate: Rate in m2-K/W per 1000 hours
        - intercept: Y-intercept (initial fouling)
        - r_squared: Coefficient of determination
        - model_type: "linear" or "accelerating"

    Raises:
        ValueError: If insufficient data points.

    Example:
        >>> history = [(0, 0), (2000, 0.00005), (4000, 0.00012)]
        >>> params = estimate_fouling_parameters_from_history(history)
    """
    if len(fouling_history) < 3:
        raise ValueError(
            f"Need at least 3 data points, got {len(fouling_history)}"
        )

    # Extract data
    times = [t / 1000 for t, _ in fouling_history]  # Convert to 1000s of hours
    resistances = [r for _, r in fouling_history]

    n = len(times)

    # Calculate means
    mean_t = sum(times) / n
    mean_r = sum(resistances) / n

    # Calculate slope and intercept (linear regression)
    numerator = sum((t - mean_t) * (r - mean_r) for t, r in zip(times, resistances))
    denominator = sum((t - mean_t) ** 2 for t in times)

    if denominator == 0:
        return {
            "fouling_rate": 0.0,
            "intercept": mean_r,
            "r_squared": 0.0,
            "model_type": "constant",
        }

    slope = numerator / denominator
    intercept = mean_r - slope * mean_t

    # Calculate R-squared
    ss_res = sum((r - (intercept + slope * t)) ** 2 for t, r in zip(times, resistances))
    ss_tot = sum((r - mean_r) ** 2 for r in resistances)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Determine model type based on residuals pattern
    # If late points consistently above line, fouling is accelerating
    residuals = [(r - (intercept + slope * t)) for t, r in zip(times, resistances)]
    late_residuals = residuals[len(residuals)//2:]
    avg_late_residual = sum(late_residuals) / len(late_residuals) if late_residuals else 0

    model_type = "accelerating" if avg_late_residual > 0.00001 else "linear"

    logger.info(
        f"Fouling parameter estimation: rate={slope:.6f}/1000h, "
        f"intercept={intercept:.6f}, R2={r_squared:.3f}, type={model_type}"
    )

    return {
        "fouling_rate": slope,
        "intercept": max(0, intercept),
        "r_squared": r_squared,
        "model_type": model_type,
    }


def calculate_hours_to_cleaning_threshold(
    current_fouling: float,
    fouling_rate: float,
    threshold_fouling: float
) -> float:
    """
    Calculate hours until cleaning threshold is reached.

    Formula:
        hours = (R_threshold - R_current) / fouling_rate * 1000

    Reference: Standard trend extrapolation

    Args:
        current_fouling: Current fouling resistance (m2-K/W).
        fouling_rate: Fouling rate (m2-K/W per 1000 hours).
        threshold_fouling: Cleaning threshold resistance (m2-K/W).

    Returns:
        Hours until threshold, or inf if rate is zero/negative.

    Example:
        >>> hours = calculate_hours_to_cleaning_threshold(
        ...     0.0001, 0.00003, 0.0002
        ... )
        >>> print(f"Hours to cleaning: {hours:.0f}")
    """
    if current_fouling >= threshold_fouling:
        return 0.0  # Already at or above threshold

    if fouling_rate <= 0:
        return float('inf')  # Never reaches threshold

    hours = (threshold_fouling - current_fouling) / fouling_rate * 1000

    return hours


def recommend_cleaning_date(
    current_date: date,
    hours_to_threshold: float,
    operating_hours_per_day: float,
    minimum_advance_days: int = 7
) -> date:
    """
    Recommend optimal cleaning date based on fouling projection.

    Accounts for operational scheduling by recommending cleaning
    before the threshold is reached.

    Reference: Maintenance scheduling best practices

    Args:
        current_date: Current date.
        hours_to_threshold: Hours until cleaning threshold.
        operating_hours_per_day: Average daily operating hours.
        minimum_advance_days: Minimum days advance notice.

    Returns:
        Recommended cleaning date.

    Example:
        >>> clean_date = recommend_cleaning_date(
        ...     date.today(), 2000, 20, 7
        ... )
    """
    if hours_to_threshold <= 0:
        return current_date + timedelta(days=minimum_advance_days)

    if operating_hours_per_day <= 0:
        logger.warning("Operating hours per day must be > 0")
        operating_hours_per_day = 24

    # Calculate days to threshold
    days_to_threshold = hours_to_threshold / operating_hours_per_day

    # Apply safety margin (clean at 80% of threshold time)
    safety_factor = 0.80
    safe_days = days_to_threshold * safety_factor

    # Ensure minimum advance notice
    days_to_cleaning = max(minimum_advance_days, int(safe_days))

    return current_date + timedelta(days=days_to_cleaning)


def calculate_cleaning_benefit(
    u_before_cleaning: float,
    u_clean: float,
    cleaning_effectiveness: float,
    heat_duty_kw: float,
    lmtd: float
) -> Dict[str, float]:
    """
    Calculate expected benefit from cleaning.

    Calculates the improvement in U and resulting performance gains.

    Reference: Heat transfer principles, economic analysis

    Args:
        u_before_cleaning: Current fouled U (W/m2-K).
        u_clean: Clean (design) U (W/m2-K).
        cleaning_effectiveness: Cleaning method effectiveness (0-1).
        heat_duty_kw: Current heat duty (kW).
        lmtd: Current LMTD (K).

    Returns:
        Dictionary with:
        - u_after_cleaning: Expected U after cleaning
        - improvement_percent: Improvement in U (%)
        - ttd_reduction_c: Estimated TTD reduction (C)

    Example:
        >>> benefit = calculate_cleaning_benefit(
        ...     2800, 3500, 0.90, 24000, 8.5
        ... )
    """
    # Calculate expected U after cleaning
    u_improvement = (u_clean - u_before_cleaning) * cleaning_effectiveness
    u_after = u_before_cleaning + u_improvement

    # Improvement percentage
    improvement_pct = (u_after - u_before_cleaning) / u_before_cleaning * 100

    # Estimate TTD reduction (simplified)
    # Higher U means lower TTD for same heat duty
    # TTD ~ Q / (U * A), so TTD_new / TTD_old ~ U_old / U_new
    # For LMTD ~ TTD (approximation):
    ttd_ratio = u_before_cleaning / u_after
    ttd_reduction = lmtd * (1 - ttd_ratio)

    return {
        "u_after_cleaning": u_after,
        "improvement_percent": improvement_pct,
        "ttd_reduction_c": ttd_reduction,
    }


def assess_fouling_severity(
    fouling_resistance: float,
    design_fouling: float
) -> Tuple[str, float]:
    """
    Assess fouling severity relative to design allowance.

    Reference: HEI Standards, TEMA Standards

    Args:
        fouling_resistance: Current fouling resistance (m2-K/W).
        design_fouling: Design fouling allowance (m2-K/W).

    Returns:
        Tuple of (severity_level, ratio_to_design).
        severity_level: "CLEAN", "LIGHT", "MODERATE", "HEAVY", "SEVERE"
        ratio_to_design: Fouling as multiple of design allowance

    Example:
        >>> severity, ratio = assess_fouling_severity(0.0003, 0.0002)
        >>> print(f"Severity: {severity}, Ratio: {ratio:.1f}x design")
    """
    if design_fouling <= 0:
        raise ValueError(f"Design fouling must be > 0, got {design_fouling}")

    ratio = fouling_resistance / design_fouling

    if ratio <= 0.25:
        severity = "CLEAN"
    elif ratio <= 0.50:
        severity = "LIGHT"
    elif ratio <= 1.00:
        severity = "MODERATE"
    elif ratio <= 1.50:
        severity = "HEAVY"
    else:
        severity = "SEVERE"

    logger.debug(f"Fouling severity: {severity}, ratio={ratio:.2f}x design")

    return (severity, ratio)


def get_standard_fouling_resistance(
    water_type: str,
    temperature_c: float = 40.0
) -> float:
    """
    Get standard design fouling resistance for water type.

    Reference: TEMA Standards Table RGP-T-2.4

    Args:
        water_type: Type of cooling water.
        temperature_c: Average water temperature.

    Returns:
        Standard fouling resistance in m2-K/W.

    Example:
        >>> r_design = get_standard_fouling_resistance("seawater", 35.0)
        >>> print(f"Design fouling: {r_design:.6f} m2K/W")
    """
    water_type_lower = water_type.lower().replace(" ", "_")

    if water_type_lower not in STANDARD_FOULING_RESISTANCES:
        logger.warning(
            f"Unknown water type '{water_type}', using cooling_tower_treated"
        )
        water_type_lower = "cooling_tower_treated"

    value = STANDARD_FOULING_RESISTANCES[water_type_lower]

    if isinstance(value, dict):
        # Temperature dependent
        if temperature_c < 50:
            return value.get("below_50c", value.get("typical", 0.000176))
        else:
            return value.get("above_50c", value.get("typical", 0.000352))
    else:
        return value


def calculate_fouling_factor_components(
    biological_fouling: float = 0.0,
    scaling_fouling: float = 0.0,
    particulate_fouling: float = 0.0,
    corrosion_fouling: float = 0.0
) -> Dict[str, float]:
    """
    Calculate total fouling from individual components.

    Formula:
        R_total = R_biological + R_scaling + R_particulate + R_corrosion

    Reference: Fouling mechanisms analysis

    Args:
        biological_fouling: Bio-fouling resistance (m2-K/W).
        scaling_fouling: Scale deposit resistance (m2-K/W).
        particulate_fouling: Particulate/silt fouling (m2-K/W).
        corrosion_fouling: Corrosion product fouling (m2-K/W).

    Returns:
        Dictionary with individual and total fouling values.

    Example:
        >>> components = calculate_fouling_factor_components(
        ...     biological_fouling=0.00005,
        ...     scaling_fouling=0.00008
        ... )
    """
    total = biological_fouling + scaling_fouling + particulate_fouling + corrosion_fouling

    # Determine dominant mechanism
    components = {
        "biological": biological_fouling,
        "scaling": scaling_fouling,
        "particulate": particulate_fouling,
        "corrosion": corrosion_fouling,
    }
    dominant = max(components, key=components.get) if total > 0 else "none"

    return {
        "biological_fouling": biological_fouling,
        "scaling_fouling": scaling_fouling,
        "particulate_fouling": particulate_fouling,
        "corrosion_fouling": corrosion_fouling,
        "total_fouling": total,
        "dominant_mechanism": dominant,
    }


def generate_fouling_recommendations(
    fouling_severity: str,
    dominant_mechanism: str,
    fouling_rate: float,
    hours_to_threshold: float
) -> List[Dict[str, str]]:
    """
    Generate fouling management recommendations.

    Reference: EPRI guidelines, best practices

    Args:
        fouling_severity: Current fouling severity level.
        dominant_mechanism: Dominant fouling mechanism.
        fouling_rate: Current fouling rate.
        hours_to_threshold: Hours until cleaning threshold.

    Returns:
        List of recommendation dictionaries.

    Example:
        >>> recs = generate_fouling_recommendations(
        ...     "HEAVY", "biological", 0.00004, 1500
        ... )
    """
    recommendations = []

    # Severity-based recommendations
    if fouling_severity == "SEVERE":
        recommendations.append({
            "action": "Schedule immediate condenser cleaning",
            "priority": "CRITICAL",
            "reason": "Severe fouling detected - significant performance impact",
        })
    elif fouling_severity == "HEAVY":
        recommendations.append({
            "action": "Plan condenser cleaning within 2 weeks",
            "priority": "HIGH",
            "reason": "Heavy fouling - exceeds design allowance",
        })
    elif fouling_severity == "MODERATE":
        recommendations.append({
            "action": "Monitor fouling trend and plan cleaning",
            "priority": "MEDIUM",
            "reason": "Moderate fouling approaching design limit",
        })

    # Mechanism-specific recommendations
    if dominant_mechanism == "biological":
        recommendations.append({
            "action": "Review biocide treatment program effectiveness",
            "priority": "HIGH" if fouling_severity in ["HEAVY", "SEVERE"] else "MEDIUM",
            "reason": "Biological fouling is dominant mechanism",
        })
        recommendations.append({
            "action": "Consider chlorination or oxidizing biocide treatment",
            "priority": "MEDIUM",
            "reason": "Bio-fouling control",
        })
    elif dominant_mechanism == "scaling":
        recommendations.append({
            "action": "Review scale inhibitor dosing",
            "priority": "HIGH" if fouling_severity in ["HEAVY", "SEVERE"] else "MEDIUM",
            "reason": "Scale formation is dominant mechanism",
        })
        recommendations.append({
            "action": "Consider acid cleaning or scale inhibitor upgrade",
            "priority": "MEDIUM",
            "reason": "Scale control",
        })
    elif dominant_mechanism == "particulate":
        recommendations.append({
            "action": "Inspect and clean cooling water strainers",
            "priority": "HIGH",
            "reason": "Particulate fouling indicates filtration issues",
        })
    elif dominant_mechanism == "corrosion":
        recommendations.append({
            "action": "Review corrosion inhibitor program",
            "priority": "HIGH",
            "reason": "Corrosion product fouling detected",
        })
        recommendations.append({
            "action": "Inspect tubes for corrosion damage",
            "priority": "MEDIUM",
            "reason": "Corrosion products indicate tube degradation",
        })

    # Rate-based recommendations
    if fouling_rate > 0.00005:
        recommendations.append({
            "action": "Investigate accelerated fouling cause",
            "priority": "HIGH",
            "reason": f"Fouling rate ({fouling_rate:.6f}/1000h) is abnormally high",
        })

    # Time-based recommendations
    if hours_to_threshold < 1000:
        recommendations.append({
            "action": "Finalize cleaning schedule - threshold imminent",
            "priority": "HIGH",
            "reason": f"Only {hours_to_threshold:.0f} hours until cleaning threshold",
        })

    return recommendations
