"""
Health Score Calculator

This module implements overall health scoring and degradation analysis
for industrial burner components using deterministic weighted formulas.

Health scoring follows industry standards for condition-based maintenance:
- ISO 17359 Condition monitoring and diagnostics of machines
- ISO 13381-1 Condition monitoring - Prognostics

All calculations are deterministic - no ML/LLM in the calculation path.

Example:
    >>> health = calculate_overall_health(15000, 50000, 85, 0.9, 0.8)
    >>> print(f"Health score: {health:.1f}/100")
"""

import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Weight factors for health score components
# Sum to 1.0 for normalized scoring
HEALTH_WEIGHTS = {
    "operating_life_ratio": 0.25,  # % of design life consumed
    "flame_quality": 0.30,  # Current flame quality score
    "cycles_factor": 0.15,  # On/off cycle degradation
    "age_factor": 0.15,  # Calendar age degradation
    "component_health": 0.15,  # Individual component status
}

# Priority thresholds
PRIORITY_THRESHOLDS = {
    "critical": {
        "health_max": 30,
        "rul_max_hours": 500,
        "failure_prob_min": 0.3,
    },
    "high": {
        "health_max": 50,
        "rul_max_hours": 2000,
        "failure_prob_min": 0.15,
    },
    "medium": {
        "health_max": 70,
        "rul_max_hours": 5000,
        "failure_prob_min": 0.05,
    },
    "low": {
        "health_max": 85,
        "rul_max_hours": 10000,
        "failure_prob_min": 0.01,
    },
}

# Component importance weights
COMPONENT_WEIGHTS: Dict[str, float] = {
    "burner_nozzle": 0.25,
    "flame_detector": 0.20,
    "ignition_system": 0.15,
    "fuel_valve": 0.15,
    "air_damper": 0.10,
    "combustion_blower": 0.10,
    "refractory": 0.05,
}


def calculate_overall_health(
    operating_hours: float,
    design_life: float,
    flame_quality: float,
    cycles_factor: float,
    age_factor: float,
    component_health: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate weighted overall health score from 0-100.

    The health score combines multiple factors:
    - Operating life ratio (25%): Hours used vs design life
    - Flame quality (30%): Current combustion quality
    - Cycles factor (15%): On/off cycle wear
    - Age factor (15%): Calendar age degradation
    - Component health (15%): Individual component status

    Higher scores indicate better health.

    Args:
        operating_hours: Total operating hours.
        design_life: Design life in hours.
        flame_quality: Flame quality score 0-100.
        cycles_factor: Cycles wear factor 0-1 (1 = no wear).
        age_factor: Age degradation factor 0-1 (1 = new).
        component_health: Optional dict of component name to health score.

    Returns:
        Overall health score from 0 to 100.

    Example:
        >>> health = calculate_overall_health(15000, 50000, 85, 0.9, 0.8)
        >>> print(f"Health: {health:.1f}")
    """
    # Validate inputs
    operating_hours = max(0, operating_hours)
    design_life = max(1, design_life)  # Avoid division by zero
    flame_quality = max(0, min(100, flame_quality))
    cycles_factor = max(0, min(1, cycles_factor))
    age_factor = max(0, min(1, age_factor))

    # Calculate life ratio score (0-100)
    # Score decreases as operating hours approach design life
    life_ratio = operating_hours / design_life
    life_score = max(0, min(100, (1 - life_ratio) * 100))

    # Flame quality is already 0-100
    flame_score = flame_quality

    # Convert factors to 0-100 scores
    cycles_score = cycles_factor * 100
    age_score = age_factor * 100

    # Component health score
    if component_health:
        component_score = calculate_component_weight(component_health)
    else:
        component_score = 80  # Default assumption if no data

    # Weighted combination
    weights = HEALTH_WEIGHTS
    health = (
        weights["operating_life_ratio"] * life_score +
        weights["flame_quality"] * flame_score +
        weights["cycles_factor"] * cycles_score +
        weights["age_factor"] * age_score +
        weights["component_health"] * component_score
    )

    logger.debug(
        f"Health calculation: life={life_score:.1f}, flame={flame_score:.1f}, "
        f"cycles={cycles_score:.1f}, age={age_score:.1f}, "
        f"components={component_score:.1f}, total={health:.1f}"
    )

    return max(0.0, min(100.0, health))


def calculate_component_weight(
    component_health: Dict[str, float]
) -> float:
    """
    Calculate weighted component health score.

    Components are weighted by their importance to burner operation.
    Unknown components receive default weight of 0.05.

    Args:
        component_health: Dict mapping component name to health score (0-100).

    Returns:
        Weighted component health score (0-100).

    Example:
        >>> health = {"burner_nozzle": 90, "flame_detector": 85}
        >>> score = calculate_component_weight(health)
    """
    if not component_health:
        return 80.0  # Default if no data

    total_weight = 0.0
    weighted_sum = 0.0

    for component, health in component_health.items():
        # Get component weight (default 0.05 for unknown)
        weight = COMPONENT_WEIGHTS.get(component.lower(), 0.05)
        total_weight += weight
        weighted_sum += weight * max(0, min(100, health))

    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 80.0


def calculate_degradation_rate(
    health_history: List[float],
    time_interval_hours: float
) -> float:
    """
    Calculate health degradation rate per 1000 hours.

    Uses linear regression on health history to determine
    the rate of health decline.

    Args:
        health_history: List of health scores (most recent last).
            Must have at least 2 values.
        time_interval_hours: Time between health readings in hours.

    Returns:
        Degradation rate (health points lost per 1000 hours).
        Positive value = degrading, negative = improving.

    Raises:
        ValueError: If insufficient health history data.

    Example:
        >>> history = [95, 92, 88, 85]  # Each 1000 hours apart
        >>> rate = calculate_degradation_rate(history, 1000)
        >>> print(f"Degradation rate: {rate:.2f} points per 1000 hours")
    """
    if len(health_history) < 2:
        raise ValueError("Need at least 2 health readings for degradation calculation")

    if time_interval_hours <= 0:
        raise ValueError(f"Time interval must be > 0, got {time_interval_hours}")

    n = len(health_history)

    # Create time array (in 1000-hour units)
    time_factor = time_interval_hours / 1000.0
    times = [i * time_factor for i in range(n)]

    # Calculate means
    mean_t = sum(times) / n
    mean_h = sum(health_history) / n

    # Calculate slope using linear regression
    # slope = sum((t - mean_t)(h - mean_h)) / sum((t - mean_t)^2)
    numerator = sum((t - mean_t) * (h - mean_h) for t, h in zip(times, health_history))
    denominator = sum((t - mean_t) ** 2 for t in times)

    if denominator == 0:
        # All readings at same time (shouldn't happen)
        return 0.0

    slope = numerator / denominator

    # Degradation rate is negative slope (health decreasing = positive rate)
    degradation_rate = -slope

    logger.debug(
        f"Degradation analysis: n={n}, slope={slope:.3f}, "
        f"degradation_rate={degradation_rate:.2f}/1000h"
    )

    return degradation_rate


def determine_maintenance_priority(
    health_score: float,
    rul_hours: float,
    failure_prob_30d: float
) -> str:
    """
    Determine maintenance priority level.

    Priority is determined by the most critical of:
    - Health score (lower = more critical)
    - Remaining useful life (lower = more critical)
    - 30-day failure probability (higher = more critical)

    Priority Levels:
    - CRITICAL: Immediate maintenance required
    - HIGH: Schedule maintenance within 1 week
    - MEDIUM: Schedule maintenance within 1 month
    - LOW: Monitor and schedule as convenient
    - NONE: No maintenance needed

    Args:
        health_score: Overall health score 0-100.
        rul_hours: Remaining useful life in hours.
        failure_prob_30d: Probability of failure in next 30 days.

    Returns:
        Priority level string: CRITICAL, HIGH, MEDIUM, LOW, or NONE.

    Example:
        >>> priority = determine_maintenance_priority(45, 1500, 0.12)
        >>> print(f"Priority: {priority}")
        Priority: HIGH
    """
    # Check each priority level from most to least critical
    thresholds = PRIORITY_THRESHOLDS

    # CRITICAL check
    critical = thresholds["critical"]
    if (health_score < critical["health_max"] or
        rul_hours < critical["rul_max_hours"] or
        failure_prob_30d > critical["failure_prob_min"]):
        logger.info(
            f"CRITICAL priority: health={health_score:.1f}, "
            f"RUL={rul_hours:.0f}h, failure_prob={failure_prob_30d:.2%}"
        )
        return "CRITICAL"

    # HIGH check
    high = thresholds["high"]
    if (health_score < high["health_max"] or
        rul_hours < high["rul_max_hours"] or
        failure_prob_30d > high["failure_prob_min"]):
        logger.info(
            f"HIGH priority: health={health_score:.1f}, "
            f"RUL={rul_hours:.0f}h, failure_prob={failure_prob_30d:.2%}"
        )
        return "HIGH"

    # MEDIUM check
    medium = thresholds["medium"]
    if (health_score < medium["health_max"] or
        rul_hours < medium["rul_max_hours"] or
        failure_prob_30d > medium["failure_prob_min"]):
        return "MEDIUM"

    # LOW check
    low = thresholds["low"]
    if (health_score < low["health_max"] or
        rul_hours < low["rul_max_hours"] or
        failure_prob_30d > low["failure_prob_min"]):
        return "LOW"

    # All checks passed - no maintenance needed
    return "NONE"


def calculate_health_trend(
    health_history: List[float],
    time_interval_hours: float
) -> Dict[str, float]:
    """
    Calculate health trend statistics.

    Provides trend analysis including:
    - Current rate of change
    - Projected health at various future times
    - Time to reach critical threshold

    Args:
        health_history: List of health scores (most recent last).
        time_interval_hours: Time between readings in hours.

    Returns:
        Dictionary with trend statistics:
        - degradation_rate: Health loss per 1000 hours
        - health_in_1000h: Projected health in 1000 hours
        - health_in_5000h: Projected health in 5000 hours
        - hours_to_critical: Hours until health reaches 30
        - hours_to_zero: Hours until health reaches 0

    Example:
        >>> history = [95, 92, 88, 85]
        >>> trend = calculate_health_trend(history, 1000)
    """
    if len(health_history) < 2:
        return {
            "degradation_rate": 0.0,
            "health_in_1000h": health_history[-1] if health_history else 0,
            "health_in_5000h": health_history[-1] if health_history else 0,
            "hours_to_critical": float('inf'),
            "hours_to_zero": float('inf'),
        }

    # Get degradation rate
    deg_rate = calculate_degradation_rate(health_history, time_interval_hours)
    current_health = health_history[-1]

    # Project future health (assuming linear degradation)
    health_1000h = max(0, current_health - deg_rate)
    health_5000h = max(0, current_health - 5 * deg_rate)

    # Calculate time to thresholds
    critical_threshold = 30
    if deg_rate > 0:
        hours_to_critical = max(0, (current_health - critical_threshold) / deg_rate * 1000)
        hours_to_zero = max(0, current_health / deg_rate * 1000)
    else:
        # Not degrading (or improving)
        hours_to_critical = float('inf')
        hours_to_zero = float('inf')

    return {
        "degradation_rate": deg_rate,
        "health_in_1000h": health_1000h,
        "health_in_5000h": health_5000h,
        "hours_to_critical": hours_to_critical,
        "hours_to_zero": hours_to_zero,
    }


def calculate_condition_index(
    health_score: float,
    flame_quality: float,
    combustion_efficiency: float
) -> float:
    """
    Calculate overall condition index.

    The condition index is a simplified single-number metric
    that combines health, quality, and efficiency.

    Formula:
        CI = 0.4 * health + 0.35 * flame_quality + 0.25 * efficiency

    Args:
        health_score: Overall health score 0-100.
        flame_quality: Flame quality score 0-100.
        combustion_efficiency: Combustion efficiency percentage.

    Returns:
        Condition index from 0 to 100.

    Example:
        >>> ci = calculate_condition_index(85, 90, 95)
        >>> print(f"Condition index: {ci:.1f}")
    """
    # Normalize efficiency to 0-100 scale (assuming 70-100% range)
    efficiency_normalized = (combustion_efficiency - 70) / 30 * 100
    efficiency_normalized = max(0, min(100, efficiency_normalized))

    # Weighted combination
    ci = (
        0.40 * max(0, min(100, health_score)) +
        0.35 * max(0, min(100, flame_quality)) +
        0.25 * efficiency_normalized
    )

    return max(0.0, min(100.0, ci))


def estimate_failure_mode(
    health_score: float,
    flame_quality: float,
    stability_index: float,
    co_ppm: float,
    operating_hours: float,
    cycles_count: int
) -> Tuple[str, float]:
    """
    Estimate most likely failure mode based on current conditions.

    Failure modes:
    - nozzle_wear: Gradual nozzle degradation
    - flame_detector_failure: Flame detector malfunction
    - ignition_failure: Ignition system problems
    - combustion_instability: Flame stability issues
    - fuel_system: Fuel valve or piping issues

    Args:
        health_score: Overall health score.
        flame_quality: Flame quality score.
        stability_index: Flame stability 0-1.
        co_ppm: CO emissions in ppm.
        operating_hours: Total operating hours.
        cycles_count: Total on/off cycles.

    Returns:
        Tuple of (failure_mode, probability).

    Example:
        >>> mode, prob = estimate_failure_mode(75, 65, 0.8, 150, 20000, 5000)
        >>> print(f"Most likely failure: {mode} ({prob:.1%})")
    """
    # Calculate likelihood scores for each mode
    scores = {}

    # Nozzle wear - correlates with operating hours and flame quality
    nozzle_score = 0.0
    if operating_hours > 10000:
        nozzle_score += 20
    if operating_hours > 25000:
        nozzle_score += 30
    if flame_quality < 70:
        nozzle_score += 25
    if co_ppm > 100:
        nozzle_score += 25
    scores["nozzle_wear"] = min(100, nozzle_score)

    # Flame detector - correlates with age and cycles
    detector_score = 0.0
    if cycles_count > 10000:
        detector_score += 30
    if cycles_count > 25000:
        detector_score += 30
    if operating_hours > 30000:
        detector_score += 20
    if health_score < 50:
        detector_score += 20
    scores["flame_detector_failure"] = min(100, detector_score)

    # Ignition failure - correlates with cycles
    ignition_score = 0.0
    if cycles_count > 15000:
        ignition_score += 40
    if cycles_count > 30000:
        ignition_score += 30
    if stability_index < 0.7:
        ignition_score += 30
    scores["ignition_failure"] = min(100, ignition_score)

    # Combustion instability - correlates with stability and CO
    instability_score = 0.0
    if stability_index < 0.9:
        instability_score += 30
    if stability_index < 0.7:
        instability_score += 30
    if co_ppm > 200:
        instability_score += 20
    if flame_quality < 60:
        instability_score += 20
    scores["combustion_instability"] = min(100, instability_score)

    # Fuel system - correlates with age and CO
    fuel_score = 0.0
    if operating_hours > 20000:
        fuel_score += 20
    if co_ppm > 150:
        fuel_score += 30
    if flame_quality < 70:
        fuel_score += 25
    if health_score < 60:
        fuel_score += 25
    scores["fuel_system"] = min(100, fuel_score)

    # Find highest scoring mode
    if not scores:
        return ("unknown", 0.0)

    max_mode = max(scores, key=scores.get)
    max_score = scores[max_mode]

    # Convert score to probability (normalize)
    total_score = sum(scores.values())
    probability = max_score / total_score if total_score > 0 else 0.0

    return (max_mode, probability)
