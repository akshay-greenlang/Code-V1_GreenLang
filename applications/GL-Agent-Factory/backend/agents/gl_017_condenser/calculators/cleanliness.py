"""
Cleanliness Factor Calculator

This module implements HEI cleanliness factor calculations for steam
surface condensers, including performance tracking, trending, and
optimization recommendations.

The cleanliness factor (CF) is the industry-standard metric for assessing
condenser tube surface condition. It represents the ratio of actual to
clean heat transfer coefficient.

All formulas are from:
- HEI Standards for Steam Surface Condensers, 12th Edition
- ASME PTC 12.2 Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> cf = calculate_cleanliness_factor(u_actual=2800, u_clean=3500)
    >>> print(f"Cleanliness factor: {cf:.2f}")
    Cleanliness factor: 0.80
"""

import math
from typing import Dict, List, Tuple, Optional
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)


# Cleanliness factor thresholds
# Reference: HEI Standards, industry best practices
CF_THRESHOLDS = {
    "excellent": 0.90,
    "good": 0.85,
    "acceptable": 0.80,
    "marginal": 0.75,
    "poor": 0.70,
}

# Design cleanliness factors by application
# Reference: HEI Standards Table 2-3
DESIGN_CLEANLINESS_FACTORS = {
    "seawater_once_through": 0.85,
    "freshwater_once_through": 0.85,
    "cooling_tower_treated": 0.85,
    "cooling_tower_untreated": 0.80,
    "river_water": 0.80,
    "industrial_cooling": 0.85,
}

# Performance degradation per CF reduction
# Reference: EPRI guidelines
PERFORMANCE_IMPACT = {
    "ttd_increase_per_cf_drop": 0.5,  # C per 0.01 CF drop
    "pressure_increase_per_cf_drop": 0.02,  # kPa per 0.01 CF drop
    "power_loss_percent_per_cf_drop": 0.1,  # % per 0.01 CF drop
}


def calculate_cleanliness_factor(
    u_actual: float,
    u_clean: float
) -> float:
    """
    Calculate HEI cleanliness factor.

    Formula:
        CF = U_actual / U_clean

    The cleanliness factor represents the current heat transfer
    capability as a fraction of clean (design) capability.

    Reference: HEI Standards, Section 2.3

    Args:
        u_actual: Actual measured heat transfer coefficient (W/m2-K).
        u_clean: Clean (design) heat transfer coefficient (W/m2-K).

    Returns:
        Cleanliness factor (0.0 to 1.0, typically 0.7-1.0).

    Raises:
        ValueError: If coefficients are invalid.

    Example:
        >>> cf = calculate_cleanliness_factor(2800, 3500)
        >>> print(f"CF: {cf:.3f}")
    """
    if u_clean <= 0:
        raise ValueError(f"U_clean must be > 0, got {u_clean}")
    if u_actual < 0:
        raise ValueError(f"U_actual must be >= 0, got {u_actual}")

    cf = u_actual / u_clean

    if cf > 1.0:
        logger.warning(
            f"CF > 1.0 ({cf:.3f}) indicates U_actual exceeds U_clean. "
            f"This may indicate measurement error or conservative U_clean value."
        )
        cf = min(cf, 1.05)  # Cap at 1.05 for slight measurement variation

    return cf


def calculate_cleanliness_factor_from_temperatures(
    t_sat: float,
    t_cw_in: float,
    t_cw_out: float,
    t_cw_in_design: float,
    t_cw_out_design: float,
    t_sat_design: float
) -> float:
    """
    Calculate cleanliness factor from temperature measurements.

    This method uses the ratio of actual to design LMTD to estimate
    cleanliness factor when direct U measurement is not available.

    Formula:
        CF = LMTD_design / LMTD_actual

    (Adjusted for same heat load conditions)

    Reference: ASME PTC 12.2

    Args:
        t_sat: Actual saturation temperature (C).
        t_cw_in: Actual CW inlet temperature (C).
        t_cw_out: Actual CW outlet temperature (C).
        t_cw_in_design: Design CW inlet temperature (C).
        t_cw_out_design: Design CW outlet temperature (C).
        t_sat_design: Design saturation temperature (C).

    Returns:
        Estimated cleanliness factor.

    Example:
        >>> cf = calculate_cleanliness_factor_from_temperatures(
        ...     40.0, 25.0, 35.0, 38.0, 25.0, 33.0
        ... )
    """
    # Calculate actual LMTD
    dt1_actual = t_sat - t_cw_in
    dt2_actual = t_sat - t_cw_out

    if dt1_actual <= 0 or dt2_actual <= 0:
        raise ValueError("Invalid temperature differences (actual)")

    if dt1_actual == dt2_actual:
        lmtd_actual = dt1_actual
    else:
        lmtd_actual = (dt1_actual - dt2_actual) / math.log(dt1_actual / dt2_actual)

    # Calculate design LMTD
    dt1_design = t_sat_design - t_cw_in_design
    dt2_design = t_sat_design - t_cw_out_design

    if dt1_design <= 0 or dt2_design <= 0:
        raise ValueError("Invalid temperature differences (design)")

    if dt1_design == dt2_design:
        lmtd_design = dt1_design
    else:
        lmtd_design = (dt1_design - dt2_design) / math.log(dt1_design / dt2_design)

    # For same heat duty: Q = U*A*LMTD, so U_actual/U_design = LMTD_design/LMTD_actual
    cf = lmtd_design / lmtd_actual

    # Apply bounds
    cf = max(0.5, min(1.05, cf))

    logger.debug(
        f"CF from temperatures: LMTD_actual={lmtd_actual:.2f}, "
        f"LMTD_design={lmtd_design:.2f}, CF={cf:.3f}"
    )

    return cf


def calculate_effective_u_from_cf(
    u_clean: float,
    cleanliness_factor: float
) -> float:
    """
    Calculate effective U from cleanliness factor.

    Formula:
        U_effective = U_clean * CF

    Reference: HEI definition

    Args:
        u_clean: Clean heat transfer coefficient (W/m2-K).
        cleanliness_factor: Current cleanliness factor (0-1).

    Returns:
        Effective heat transfer coefficient (W/m2-K).

    Example:
        >>> u_eff = calculate_effective_u_from_cf(3500, 0.85)
        >>> print(f"U_effective: {u_eff:.0f} W/m2K")
    """
    if u_clean <= 0:
        raise ValueError(f"U_clean must be > 0, got {u_clean}")
    if cleanliness_factor <= 0 or cleanliness_factor > 1.1:
        raise ValueError(
            f"CF must be 0-1.1, got {cleanliness_factor}"
        )

    return u_clean * cleanliness_factor


def assess_cleanliness_status(
    cleanliness_factor: float
) -> Tuple[str, str]:
    """
    Assess cleanliness factor status and provide guidance.

    Reference: Industry best practices

    Args:
        cleanliness_factor: Current cleanliness factor.

    Returns:
        Tuple of (status, guidance).
        status: "EXCELLENT", "GOOD", "ACCEPTABLE", "MARGINAL", "POOR"
        guidance: Text description of status

    Example:
        >>> status, guidance = assess_cleanliness_status(0.82)
        >>> print(f"Status: {status}")
    """
    thresholds = CF_THRESHOLDS

    if cleanliness_factor >= thresholds["excellent"]:
        status = "EXCELLENT"
        guidance = "Tube surfaces are very clean. Continue current maintenance program."
    elif cleanliness_factor >= thresholds["good"]:
        status = "GOOD"
        guidance = "Tube surfaces are clean. Performance within design expectations."
    elif cleanliness_factor >= thresholds["acceptable"]:
        status = "ACCEPTABLE"
        guidance = "Minor fouling present. Monitor trend for degradation."
    elif cleanliness_factor >= thresholds["marginal"]:
        status = "MARGINAL"
        guidance = "Moderate fouling. Plan cleaning at next convenient outage."
    elif cleanliness_factor >= thresholds["poor"]:
        status = "POOR"
        guidance = "Significant fouling. Schedule cleaning within 4 weeks."
    else:
        status = "CRITICAL"
        guidance = "Severe fouling. Immediate cleaning required to prevent damage."

    return (status, guidance)


def calculate_cf_trend(
    cf_history: List[Tuple[datetime, float]],
    projection_days: int = 30
) -> Dict[str, float]:
    """
    Calculate cleanliness factor trend and project future values.

    Uses linear regression on historical CF data to determine
    degradation rate and project future cleanliness.

    Reference: Statistical trend analysis

    Args:
        cf_history: List of (datetime, cf) tuples, chronologically ordered.
        projection_days: Days to project into future.

    Returns:
        Dictionary with:
        - current_cf: Most recent CF value
        - degradation_rate: CF drop per day
        - projected_cf: CF in projection_days
        - days_to_threshold: Days until CF reaches 0.75 (marginal)
        - r_squared: Trend fit quality

    Raises:
        ValueError: If insufficient data.

    Example:
        >>> history = [
        ...     (datetime(2024, 1, 1), 0.90),
        ...     (datetime(2024, 2, 1), 0.87),
        ...     (datetime(2024, 3, 1), 0.84),
        ... ]
        >>> trend = calculate_cf_trend(history, 60)
    """
    if len(cf_history) < 2:
        raise ValueError("Need at least 2 data points for trend analysis")

    # Extract data
    base_time = cf_history[0][0]
    days = [(t - base_time).days for t, _ in cf_history]
    cfs = [cf for _, cf in cf_history]

    n = len(days)
    current_cf = cfs[-1]

    # Calculate means
    mean_d = sum(days) / n
    mean_cf = sum(cfs) / n

    # Linear regression
    numerator = sum((d - mean_d) * (cf - mean_cf) for d, cf in zip(days, cfs))
    denominator = sum((d - mean_d) ** 2 for d in days)

    if denominator == 0:
        return {
            "current_cf": current_cf,
            "degradation_rate": 0.0,
            "projected_cf": current_cf,
            "days_to_threshold": float('inf'),
            "r_squared": 0.0,
        }

    slope = numerator / denominator  # CF change per day (negative = degrading)
    intercept = mean_cf - slope * mean_d

    # R-squared
    ss_res = sum((cf - (intercept + slope * d)) ** 2 for d, cf in zip(days, cfs))
    ss_tot = sum((cf - mean_cf) ** 2 for cf in cfs)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Degradation rate (positive = degrading)
    degradation_rate = -slope

    # Project future CF
    current_day = days[-1]
    future_day = current_day + projection_days
    projected_cf = intercept + slope * future_day
    projected_cf = max(0.5, min(1.0, projected_cf))  # Bounds

    # Days to threshold (CF = 0.75)
    threshold = CF_THRESHOLDS["marginal"]
    if degradation_rate > 0:
        days_to_threshold = (current_cf - threshold) / degradation_rate
        days_to_threshold = max(0, days_to_threshold)
    else:
        days_to_threshold = float('inf')

    logger.info(
        f"CF trend: current={current_cf:.3f}, rate={degradation_rate:.5f}/day, "
        f"projected={projected_cf:.3f} in {projection_days}d, "
        f"threshold in {days_to_threshold:.0f}d"
    )

    return {
        "current_cf": current_cf,
        "degradation_rate": degradation_rate,
        "projected_cf": projected_cf,
        "days_to_threshold": days_to_threshold,
        "r_squared": r_squared,
    }


def calculate_performance_impact(
    current_cf: float,
    design_cf: float = 0.85
) -> Dict[str, float]:
    """
    Calculate performance impact of reduced cleanliness.

    Reference: EPRI guidelines, heat balance calculations

    Args:
        current_cf: Current cleanliness factor.
        design_cf: Design cleanliness factor.

    Returns:
        Dictionary with performance impacts:
        - cf_deficit: Shortfall from design CF
        - ttd_increase_c: Estimated TTD increase (C)
        - pressure_increase_kpa: Estimated pressure increase (kPa)
        - power_loss_percent: Estimated power loss (%)

    Example:
        >>> impact = calculate_performance_impact(0.75, 0.85)
        >>> print(f"Power loss: {impact['power_loss_percent']:.1f}%")
    """
    cf_deficit = design_cf - current_cf

    if cf_deficit <= 0:
        return {
            "cf_deficit": 0.0,
            "ttd_increase_c": 0.0,
            "pressure_increase_kpa": 0.0,
            "power_loss_percent": 0.0,
        }

    # Calculate impacts (per 0.01 CF drop)
    cf_drops = cf_deficit * 100  # Number of 0.01 drops

    ttd_increase = cf_drops * PERFORMANCE_IMPACT["ttd_increase_per_cf_drop"]
    pressure_increase = cf_drops * PERFORMANCE_IMPACT["pressure_increase_per_cf_drop"]
    power_loss = cf_drops * PERFORMANCE_IMPACT["power_loss_percent_per_cf_drop"]

    return {
        "cf_deficit": cf_deficit,
        "ttd_increase_c": ttd_increase,
        "pressure_increase_kpa": pressure_increase,
        "power_loss_percent": power_loss,
    }


def calculate_cleaning_benefit_cf(
    current_cf: float,
    expected_cf_after_cleaning: float,
    turbine_power_mw: float
) -> Dict[str, float]:
    """
    Calculate expected benefit from cleaning in terms of CF improvement.

    Reference: Economic analysis

    Args:
        current_cf: Current cleanliness factor.
        expected_cf_after_cleaning: Expected CF after cleaning.
        turbine_power_mw: Turbine rated power (MW).

    Returns:
        Dictionary with:
        - cf_improvement: Expected CF improvement
        - power_recovery_mw: Expected power recovery (MW)
        - annual_energy_mwh: Annual energy recovery (MWh)

    Example:
        >>> benefit = calculate_cleaning_benefit_cf(0.75, 0.92, 500)
    """
    cf_improvement = expected_cf_after_cleaning - current_cf

    if cf_improvement <= 0:
        return {
            "cf_improvement": 0.0,
            "power_recovery_mw": 0.0,
            "annual_energy_mwh": 0.0,
        }

    # Power recovery from CF improvement
    cf_drops_recovered = cf_improvement * 100
    power_loss_percent_recovered = (
        cf_drops_recovered * PERFORMANCE_IMPACT["power_loss_percent_per_cf_drop"]
    )
    power_recovery_mw = turbine_power_mw * power_loss_percent_recovered / 100

    # Annual energy (assuming 8000 hours/year operation)
    hours_per_year = 8000
    annual_energy_mwh = power_recovery_mw * hours_per_year

    return {
        "cf_improvement": cf_improvement,
        "power_recovery_mw": power_recovery_mw,
        "annual_energy_mwh": annual_energy_mwh,
    }


def get_design_cleanliness_factor(
    cooling_water_source: str
) -> float:
    """
    Get recommended design cleanliness factor for water source.

    Reference: HEI Standards Table 2-3

    Args:
        cooling_water_source: Type of cooling water system.

    Returns:
        Recommended design cleanliness factor.

    Example:
        >>> cf_design = get_design_cleanliness_factor("cooling_tower_treated")
        >>> print(f"Design CF: {cf_design:.2f}")
    """
    source_lower = cooling_water_source.lower().replace(" ", "_")

    if source_lower in DESIGN_CLEANLINESS_FACTORS:
        return DESIGN_CLEANLINESS_FACTORS[source_lower]
    else:
        logger.warning(
            f"Unknown water source '{cooling_water_source}', using 0.85"
        )
        return 0.85


def calculate_cf_from_fouling_resistance(
    r_fouling: float,
    r_design: float
) -> float:
    """
    Calculate CF from fouling resistance values.

    For small fouling resistances (typical), CF approximates:
        CF ~ (1 + R_design/R_other) / (1 + R_actual/R_other)

    Simplified relation:
        CF ~ 1 - (R_fouling - R_design) * U_clean

    Reference: Heat transfer resistance model

    Args:
        r_fouling: Actual total fouling resistance (m2-K/W).
        r_design: Design fouling resistance (m2-K/W).

    Returns:
        Estimated cleanliness factor.

    Example:
        >>> cf = calculate_cf_from_fouling_resistance(0.0003, 0.0002)
    """
    if r_design <= 0:
        raise ValueError(f"R_design must be > 0, got {r_design}")

    # Ratio-based approximation
    cf = r_design / r_fouling if r_fouling > 0 else 1.0

    # Bound to reasonable range
    return max(0.5, min(1.0, cf))


def generate_cf_recommendations(
    current_cf: float,
    trend_rate: float,
    days_to_threshold: float
) -> List[Dict[str, str]]:
    """
    Generate recommendations based on cleanliness factor analysis.

    Reference: Best practices

    Args:
        current_cf: Current cleanliness factor.
        trend_rate: CF degradation rate (per day).
        days_to_threshold: Days until marginal threshold.

    Returns:
        List of recommendation dictionaries.

    Example:
        >>> recs = generate_cf_recommendations(0.78, 0.0005, 60)
    """
    recommendations = []

    status, _ = assess_cleanliness_status(current_cf)

    # Status-based recommendations
    if status == "CRITICAL":
        recommendations.append({
            "action": "Schedule emergency condenser tube cleaning",
            "priority": "CRITICAL",
            "reason": f"CF of {current_cf:.2f} indicates severe fouling",
        })
    elif status == "POOR":
        recommendations.append({
            "action": "Schedule condenser cleaning within 2 weeks",
            "priority": "HIGH",
            "reason": f"CF of {current_cf:.2f} significantly below acceptable",
        })
    elif status == "MARGINAL":
        recommendations.append({
            "action": "Plan condenser cleaning at next convenient outage",
            "priority": "MEDIUM",
            "reason": f"CF of {current_cf:.2f} approaching poor condition",
        })

    # Trend-based recommendations
    if trend_rate > 0.001:  # Rapid degradation
        recommendations.append({
            "action": "Investigate accelerated fouling cause",
            "priority": "HIGH",
            "reason": f"Rapid CF decline ({trend_rate:.4f}/day)",
        })
        recommendations.append({
            "action": "Review water treatment program effectiveness",
            "priority": "HIGH",
            "reason": "Fast fouling may indicate treatment issues",
        })
    elif trend_rate > 0.0005:  # Moderate degradation
        recommendations.append({
            "action": "Increase monitoring frequency",
            "priority": "MEDIUM",
            "reason": f"Moderate CF decline rate ({trend_rate:.4f}/day)",
        })

    # Time-based recommendations
    if days_to_threshold < 30:
        recommendations.append({
            "action": "Finalize cleaning plan - threshold imminent",
            "priority": "HIGH",
            "reason": f"Only {days_to_threshold:.0f} days until marginal CF",
        })
    elif days_to_threshold < 60:
        recommendations.append({
            "action": "Begin cleaning scheduling discussions",
            "priority": "MEDIUM",
            "reason": f"{days_to_threshold:.0f} days until marginal CF",
        })

    # If CF is good, provide positive feedback
    if status in ["EXCELLENT", "GOOD"] and trend_rate < 0.0003:
        recommendations.append({
            "action": "Continue current maintenance and water treatment program",
            "priority": "LOW",
            "reason": "CF performance is satisfactory",
        })

    return recommendations


def calculate_cf_score(
    current_cf: float,
    trend_rate: float,
    deviation_from_design: float
) -> float:
    """
    Calculate composite cleanliness factor score for reporting.

    Score combines current CF, trend, and design deviation into
    a single 0-100 metric.

    Args:
        current_cf: Current cleanliness factor.
        trend_rate: CF degradation rate (per day).
        deviation_from_design: CF shortfall from design value.

    Returns:
        Composite score 0-100.

    Example:
        >>> score = calculate_cf_score(0.82, 0.0003, 0.03)
        >>> print(f"CF Score: {score:.0f}/100")
    """
    # Component scores
    # CF level (0.70-1.00 maps to 0-50 points)
    cf_normalized = (current_cf - 0.70) / 0.30
    cf_score = max(0, min(50, cf_normalized * 50))

    # Trend (0-0.001/day degradation maps to 0-25 points, lower is better)
    trend_normalized = 1 - min(1, trend_rate / 0.001)
    trend_score = trend_normalized * 25

    # Design deviation (0-0.15 deviation maps to 0-25 points, lower is better)
    dev_normalized = 1 - min(1, deviation_from_design / 0.15)
    dev_score = dev_normalized * 25

    total_score = cf_score + trend_score + dev_score

    return max(0, min(100, total_score))
