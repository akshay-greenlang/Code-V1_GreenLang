"""
Maintenance Scheduling Calculator

This module implements maintenance scheduling and replacement decision
logic for industrial burners using deterministic engineering formulas.

All calculations follow industry standards:
- API 535 Burners for Fired Heaters in General Refinery Services
- NFPA 86 Standard for Ovens and Furnaces
- ISO 14224 Petroleum and natural gas industries - Collection of reliability data

Zero-hallucination: All calculations are deterministic formulas.
No ML/LLM in the calculation path.

Example:
    >>> recommendations = generate_maintenance_recommendations(
    ...     component_health, flame_quality, operating_conditions
    ... )
    >>> for rec in recommendations:
    ...     print(f"{rec['priority']}: {rec['action']}")
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Standard maintenance intervals (hours)
MAINTENANCE_INTERVALS = {
    "inspection": 2000,  # Visual inspection
    "cleaning": 4000,  # Cleaning and minor adjustments
    "preventive": 8000,  # Preventive maintenance
    "major": 25000,  # Major overhaul
}

# Component-specific maintenance actions
COMPONENT_MAINTENANCE: Dict[str, List[Dict[str, Any]]] = {
    "burner_nozzle": [
        {
            "action": "Inspect nozzle for wear and deposits",
            "trigger_health": 80,
            "estimated_hours": 0.5,
        },
        {
            "action": "Clean nozzle orifices and passages",
            "trigger_health": 70,
            "estimated_hours": 1.0,
        },
        {
            "action": "Replace burner nozzle",
            "trigger_health": 50,
            "estimated_hours": 2.0,
        },
    ],
    "flame_detector": [
        {
            "action": "Clean flame detector lens",
            "trigger_health": 85,
            "estimated_hours": 0.25,
        },
        {
            "action": "Test and calibrate flame detector",
            "trigger_health": 70,
            "estimated_hours": 0.5,
        },
        {
            "action": "Replace flame detector",
            "trigger_health": 50,
            "estimated_hours": 1.0,
        },
    ],
    "ignition_system": [
        {
            "action": "Inspect ignition electrode gap",
            "trigger_health": 80,
            "estimated_hours": 0.5,
        },
        {
            "action": "Replace ignition electrode",
            "trigger_health": 60,
            "estimated_hours": 1.0,
        },
        {
            "action": "Replace ignition transformer",
            "trigger_health": 40,
            "estimated_hours": 1.5,
        },
    ],
    "fuel_valve": [
        {
            "action": "Check fuel valve operation and seat",
            "trigger_health": 75,
            "estimated_hours": 0.5,
        },
        {
            "action": "Adjust fuel valve linkage",
            "trigger_health": 65,
            "estimated_hours": 1.0,
        },
        {
            "action": "Replace fuel valve assembly",
            "trigger_health": 45,
            "estimated_hours": 3.0,
        },
    ],
    "air_damper": [
        {
            "action": "Lubricate air damper bearings",
            "trigger_health": 80,
            "estimated_hours": 0.5,
        },
        {
            "action": "Adjust air damper linkage and actuator",
            "trigger_health": 65,
            "estimated_hours": 1.0,
        },
        {
            "action": "Replace air damper actuator",
            "trigger_health": 45,
            "estimated_hours": 2.0,
        },
    ],
    "combustion_blower": [
        {
            "action": "Check blower wheel balance and bearings",
            "trigger_health": 75,
            "estimated_hours": 1.0,
        },
        {
            "action": "Replace blower bearings",
            "trigger_health": 55,
            "estimated_hours": 3.0,
        },
        {
            "action": "Replace combustion blower assembly",
            "trigger_health": 35,
            "estimated_hours": 4.0,
        },
    ],
    "refractory": [
        {
            "action": "Inspect refractory for cracks and erosion",
            "trigger_health": 80,
            "estimated_hours": 1.0,
        },
        {
            "action": "Patch refractory damage",
            "trigger_health": 60,
            "estimated_hours": 4.0,
        },
        {
            "action": "Reline burner refractory",
            "trigger_health": 40,
            "estimated_hours": 8.0,
        },
    ],
}

# Flame quality maintenance actions
FLAME_QUALITY_MAINTENANCE: List[Dict[str, Any]] = [
    {
        "trigger": "score_low",
        "threshold": 70,
        "action": "Perform combustion tune-up",
        "reason": "Flame quality below optimal",
        "estimated_hours": 2.0,
    },
    {
        "trigger": "efficiency_low",
        "threshold": 90,
        "action": "Adjust air-fuel ratio for optimal efficiency",
        "reason": "Combustion efficiency below target",
        "estimated_hours": 1.5,
    },
    {
        "trigger": "high_co",
        "threshold": 100,
        "action": "Investigate incomplete combustion",
        "reason": "CO emissions above acceptable limits",
        "estimated_hours": 2.0,
    },
    {
        "trigger": "high_nox",
        "threshold": 100,
        "action": "Adjust burner for NOx reduction",
        "reason": "NOx emissions above regulatory limits",
        "estimated_hours": 2.0,
    },
]

# Anomaly-specific actions
ANOMALY_ACTIONS: Dict[str, Dict[str, Any]] = {
    "flame_lifting": {
        "action": "Adjust fuel pressure and air settings to stabilize flame",
        "reason": "Flame lifting detected - risk of flame-out",
        "priority": "HIGH",
        "estimated_hours": 1.5,
    },
    "flashback": {
        "action": "Reduce firing rate and inspect for fuel system leaks",
        "reason": "Flashback condition detected - safety hazard",
        "priority": "CRITICAL",
        "estimated_hours": 3.0,
    },
    "pulsation": {
        "action": "Adjust combustion air and inspect for resonance causes",
        "reason": "Flame pulsation detected - potential equipment damage",
        "priority": "HIGH",
        "estimated_hours": 2.0,
    },
    "incomplete_combustion": {
        "action": "Increase combustion air and clean fuel nozzles",
        "reason": "Incomplete combustion - efficiency loss and emissions",
        "priority": "MEDIUM",
        "estimated_hours": 1.5,
    },
}

# Replacement decision thresholds
REPLACEMENT_THRESHOLDS = {
    "max_age_years": 20,  # Maximum calendar age
    "min_health_for_repair": 25,  # Below this, replacement preferred
    "min_rul_for_repair": 2000,  # Minimum RUL to justify repair
    "max_repair_cost_ratio": 0.5,  # Max repair cost as % of replacement
}


def generate_maintenance_recommendations(
    component_health: Dict[str, Dict[str, Any]],
    flame_quality: Dict[str, float],
    operating_conditions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate prioritized maintenance recommendations.

    Analyzes component health, flame quality, and operating conditions
    to generate a list of recommended maintenance actions.

    Args:
        component_health: Dict mapping component name to health data.
            Each entry should have 'health_score', 'operating_hours', 'cycles'.
        flame_quality: Dict with 'score' and 'efficiency' keys.
        operating_conditions: Dict with operating data including:
            - operating_hours: Total hours
            - design_life: Design life hours
            - reliability: Current reliability
            - rul_hours: Remaining useful life
            - failure_prob_30d: 30-day failure probability
            - anomalies: List of detected anomalies

    Returns:
        List of recommendation dicts, each containing:
        - action: Description of recommended action
        - priority: CRITICAL, HIGH, MEDIUM, LOW, or NONE
        - reason: Why this action is recommended
        - estimated_hours: Estimated labor hours
        - component: Affected component (if applicable)
        - due_date: Recommended due date (if applicable)

    Example:
        >>> recommendations = generate_maintenance_recommendations(
        ...     {"burner_nozzle": {"health_score": 65}},
        ...     {"score": 75, "efficiency": 92},
        ...     {"operating_hours": 15000, "anomalies": []}
        ... )
    """
    recommendations = []

    # 1. Component-based recommendations
    for comp_name, comp_data in component_health.items():
        comp_recs = _generate_component_recommendations(comp_name, comp_data)
        recommendations.extend(comp_recs)

    # 2. Flame quality recommendations
    flame_recs = _generate_flame_quality_recommendations(flame_quality)
    recommendations.extend(flame_recs)

    # 3. Anomaly-based recommendations
    anomalies = operating_conditions.get("anomalies", [])
    anomaly_recs = _generate_anomaly_recommendations(anomalies)
    recommendations.extend(anomaly_recs)

    # 4. Time-based recommendations
    operating_hours = operating_conditions.get("operating_hours", 0)
    time_recs = _generate_time_based_recommendations(operating_hours)
    recommendations.extend(time_recs)

    # 5. Reliability-based recommendations
    reliability_recs = _generate_reliability_recommendations(operating_conditions)
    recommendations.extend(reliability_recs)

    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "LOW"), 4))

    # Remove duplicates (keep highest priority for same action)
    seen_actions = set()
    unique_recs = []
    for rec in recommendations:
        action_key = rec["action"].lower()
        if action_key not in seen_actions:
            seen_actions.add(action_key)
            unique_recs.append(rec)

    logger.info(f"Generated {len(unique_recs)} maintenance recommendations")

    return unique_recs


def _generate_component_recommendations(
    comp_name: str,
    comp_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate recommendations for a specific component."""
    recommendations = []
    health = comp_data.get("health_score", 100)

    # Get component-specific actions
    comp_key = comp_name.lower().replace(" ", "_")
    actions = COMPONENT_MAINTENANCE.get(comp_key, [])

    for action_def in actions:
        if health < action_def["trigger_health"]:
            priority = _health_to_priority(health)
            recommendations.append({
                "action": action_def["action"],
                "priority": priority,
                "reason": f"{comp_name} health at {health:.0f}%",
                "estimated_hours": action_def["estimated_hours"],
                "component": comp_name,
            })
            break  # Only recommend most urgent action for each component

    return recommendations


def _generate_flame_quality_recommendations(
    flame_quality: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Generate recommendations based on flame quality."""
    recommendations = []
    score = flame_quality.get("score", 100)
    efficiency = flame_quality.get("efficiency", 100)

    for action_def in FLAME_QUALITY_MAINTENANCE:
        trigger = action_def["trigger"]
        threshold = action_def["threshold"]

        should_add = False
        if trigger == "score_low" and score < threshold:
            should_add = True
        elif trigger == "efficiency_low" and efficiency < threshold:
            should_add = True

        if should_add:
            priority = "HIGH" if score < 50 or efficiency < 85 else "MEDIUM"
            recommendations.append({
                "action": action_def["action"],
                "priority": priority,
                "reason": action_def["reason"],
                "estimated_hours": action_def["estimated_hours"],
            })

    return recommendations


def _generate_anomaly_recommendations(
    anomalies: List[str]
) -> List[Dict[str, Any]]:
    """Generate recommendations for detected anomalies."""
    recommendations = []

    for anomaly in anomalies:
        anomaly_key = anomaly.lower().replace(" ", "_")
        if anomaly_key in ANOMALY_ACTIONS:
            action_def = ANOMALY_ACTIONS[anomaly_key]
            recommendations.append({
                "action": action_def["action"],
                "priority": action_def["priority"],
                "reason": action_def["reason"],
                "estimated_hours": action_def["estimated_hours"],
            })

    return recommendations


def _generate_time_based_recommendations(
    operating_hours: float
) -> List[Dict[str, Any]]:
    """Generate time-based preventive maintenance recommendations."""
    recommendations = []

    # Check for overdue maintenance intervals
    for maint_type, interval in MAINTENANCE_INTERVALS.items():
        hours_since = operating_hours % interval
        if hours_since > interval * 0.9:
            # Within 10% of next interval
            recommendations.append({
                "action": f"Schedule {maint_type} maintenance (interval: {interval}h)",
                "priority": "LOW",
                "reason": f"Approaching {interval}-hour {maint_type} interval",
                "estimated_hours": 2.0 if maint_type == "major" else 1.0,
            })

    return recommendations


def _generate_reliability_recommendations(
    operating_conditions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate recommendations based on reliability metrics."""
    recommendations = []

    reliability = operating_conditions.get("reliability", 1.0)
    rul_hours = operating_conditions.get("rul_hours", float('inf'))
    failure_prob = operating_conditions.get("failure_prob_30d", 0.0)

    # Low reliability warning
    if reliability < 0.8:
        recommendations.append({
            "action": "Plan for burner overhaul or replacement",
            "priority": "HIGH" if reliability < 0.6 else "MEDIUM",
            "reason": f"Reliability at {reliability:.1%} - below acceptable threshold",
            "estimated_hours": 8.0,
        })

    # Low RUL warning
    if rul_hours < 1000:
        recommendations.append({
            "action": "Urgent: Schedule comprehensive maintenance",
            "priority": "CRITICAL",
            "reason": f"Only {rul_hours:.0f} hours remaining useful life",
            "estimated_hours": 6.0,
        })
    elif rul_hours < 5000:
        recommendations.append({
            "action": "Plan maintenance to extend burner life",
            "priority": "HIGH",
            "reason": f"RUL of {rul_hours:.0f} hours requires attention",
            "estimated_hours": 4.0,
        })

    # High failure probability warning
    if failure_prob > 0.2:
        recommendations.append({
            "action": "Immediate inspection and preventive maintenance",
            "priority": "CRITICAL",
            "reason": f"{failure_prob:.1%} probability of failure in next 30 days",
            "estimated_hours": 4.0,
        })
    elif failure_prob > 0.1:
        recommendations.append({
            "action": "Schedule preventive maintenance within 2 weeks",
            "priority": "HIGH",
            "reason": f"{failure_prob:.1%} probability of failure in next 30 days",
            "estimated_hours": 3.0,
        })

    return recommendations


def _health_to_priority(health: float) -> str:
    """Convert health score to priority level."""
    if health < 30:
        return "CRITICAL"
    elif health < 50:
        return "HIGH"
    elif health < 70:
        return "MEDIUM"
    elif health < 85:
        return "LOW"
    else:
        return "NONE"


def calculate_next_maintenance_date(
    current_date: date,
    rul_hours: float,
    operating_hours_per_day: float
) -> Optional[date]:
    """
    Calculate optimal next maintenance date.

    The maintenance date is scheduled to occur before the burner
    reaches its remaining useful life limit, with a safety margin.

    Formula:
        maintenance_days = (rul_hours * safety_factor) / operating_hours_per_day

    Args:
        current_date: Current date.
        rul_hours: Remaining useful life in hours.
        operating_hours_per_day: Average daily operating hours.

    Returns:
        Recommended maintenance date, or None if not applicable.

    Example:
        >>> next_date = calculate_next_maintenance_date(
        ...     date.today(), 5000, 16
        ... )
        >>> print(f"Next maintenance: {next_date}")
    """
    if rul_hours <= 0:
        # Already past end of life
        return current_date

    if operating_hours_per_day <= 0:
        logger.warning("Operating hours per day must be > 0")
        return None

    # Apply safety factor (schedule at 70% of RUL)
    safety_factor = 0.7
    safe_rul_hours = rul_hours * safety_factor

    # Calculate days until maintenance
    days_to_maintenance = safe_rul_hours / operating_hours_per_day

    # Round to nearest week (practical scheduling)
    weeks = max(1, round(days_to_maintenance / 7))
    maintenance_date = current_date + timedelta(weeks=weeks)

    logger.debug(
        f"Maintenance scheduling: RUL={rul_hours:.0f}h, "
        f"safe_RUL={safe_rul_hours:.0f}h, days={days_to_maintenance:.0f}, "
        f"date={maintenance_date}"
    )

    return maintenance_date


def should_replace_burner(
    age_years: float,
    health_score: float,
    rul_hours: float,
    repair_cost_ratio: float
) -> Tuple[bool, Optional[str]]:
    """
    Determine if burner should be replaced vs repaired.

    Decision is based on:
    - Calendar age limits
    - Health score thresholds
    - Remaining useful life
    - Repair cost economics

    Args:
        age_years: Burner age in years.
        health_score: Current health score 0-100.
        rul_hours: Remaining useful life in hours.
        repair_cost_ratio: Repair cost as fraction of replacement cost.

    Returns:
        Tuple of (should_replace, reason).
        - should_replace: True if replacement recommended
        - reason: Explanation for the recommendation

    Example:
        >>> replace, reason = should_replace_burner(18, 35, 1500, 0.4)
        >>> print(f"Replace: {replace}, Reason: {reason}")
    """
    thresholds = REPLACEMENT_THRESHOLDS
    reasons = []

    # Check age limit
    if age_years > thresholds["max_age_years"]:
        reasons.append(f"Age ({age_years:.1f} years) exceeds maximum ({thresholds['max_age_years']} years)")

    # Check health threshold
    if health_score < thresholds["min_health_for_repair"]:
        reasons.append(f"Health ({health_score:.0f}%) below repair threshold ({thresholds['min_health_for_repair']}%)")

    # Check RUL threshold
    if rul_hours < thresholds["min_rul_for_repair"]:
        reasons.append(f"RUL ({rul_hours:.0f}h) below minimum for repair ({thresholds['min_rul_for_repair']}h)")

    # Check repair cost economics
    if repair_cost_ratio > thresholds["max_repair_cost_ratio"]:
        reasons.append(f"Repair cost ({repair_cost_ratio:.0%}) exceeds economic threshold ({thresholds['max_repair_cost_ratio']:.0%})")

    # Make decision
    if len(reasons) >= 2:
        # Multiple factors suggest replacement
        return (True, "; ".join(reasons))
    elif len(reasons) == 1:
        # Single factor - replacement recommended but not critical
        return (True, reasons[0])
    else:
        return (False, None)


def calculate_maintenance_cost_benefit(
    repair_cost: float,
    replacement_cost: float,
    current_rul_hours: float,
    post_repair_rul_hours: float,
    new_rul_hours: float,
    operating_cost_per_hour: float
) -> Dict[str, Any]:
    """
    Calculate cost-benefit analysis for repair vs replacement.

    Compares total cost of ownership for repair vs replacement
    over the remaining equipment life.

    Args:
        repair_cost: Cost of repair in currency units.
        replacement_cost: Cost of replacement in currency units.
        current_rul_hours: Current RUL without repair.
        post_repair_rul_hours: Expected RUL after repair.
        new_rul_hours: Expected life of new burner.
        operating_cost_per_hour: Operating cost per hour.

    Returns:
        Dictionary with:
        - repair_cost_per_hour: Repair cost amortized per operating hour
        - replacement_cost_per_hour: Replacement cost amortized per hour
        - recommended_action: "repair" or "replace"
        - cost_savings: Savings from recommended action
        - payback_hours: Hours to recover investment

    Example:
        >>> result = calculate_maintenance_cost_benefit(
        ...     5000, 25000, 2000, 15000, 50000, 10
        ... )
    """
    # Avoid division by zero
    post_repair_rul_hours = max(1, post_repair_rul_hours)
    new_rul_hours = max(1, new_rul_hours)

    # Cost per hour for repair option
    repair_cost_per_hour = repair_cost / post_repair_rul_hours

    # Cost per hour for replacement option
    replacement_cost_per_hour = replacement_cost / new_rul_hours

    # Determine recommended action
    if repair_cost_per_hour < replacement_cost_per_hour:
        recommended = "repair"
        savings_per_hour = replacement_cost_per_hour - repair_cost_per_hour
        total_savings = savings_per_hour * post_repair_rul_hours
        payback = repair_cost / (replacement_cost_per_hour - repair_cost_per_hour) if savings_per_hour > 0 else 0
    else:
        recommended = "replace"
        savings_per_hour = repair_cost_per_hour - replacement_cost_per_hour
        total_savings = savings_per_hour * new_rul_hours
        payback = replacement_cost / (repair_cost_per_hour - replacement_cost_per_hour) if savings_per_hour > 0 else 0

    return {
        "repair_cost_per_hour": repair_cost_per_hour,
        "replacement_cost_per_hour": replacement_cost_per_hour,
        "recommended_action": recommended,
        "cost_savings": total_savings,
        "payback_hours": payback,
    }


def estimate_repair_benefit(
    current_health: float,
    repair_type: str
) -> Tuple[float, float]:
    """
    Estimate health improvement and RUL extension from repair.

    Args:
        current_health: Current health score 0-100.
        repair_type: Type of repair (inspection, cleaning, preventive, major).

    Returns:
        Tuple of (health_improvement, rul_extension_factor).
        - health_improvement: Points added to health score
        - rul_extension_factor: Multiplier for RUL extension

    Example:
        >>> improvement, extension = estimate_repair_benefit(60, "preventive")
        >>> print(f"Health +{improvement}, RUL x{extension:.1f}")
    """
    # Repair benefit estimates based on repair type
    repair_benefits = {
        "inspection": (5, 1.1),  # Small improvement, 10% RUL extension
        "cleaning": (10, 1.3),  # Moderate improvement, 30% extension
        "preventive": (20, 1.5),  # Good improvement, 50% extension
        "major": (40, 2.0),  # Significant improvement, 2x extension
    }

    if repair_type.lower() not in repair_benefits:
        logger.warning(f"Unknown repair type: {repair_type}, using 'preventive' defaults")
        repair_type = "preventive"

    health_improvement, rul_factor = repair_benefits[repair_type.lower()]

    # Cap health at 100
    actual_improvement = min(health_improvement, 100 - current_health)

    return (actual_improvement, rul_factor)
