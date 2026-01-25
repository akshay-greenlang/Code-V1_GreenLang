"""
Chemical Dosing Calculator for Boiler Water Treatment

This module implements chemical dosing calculations for boiler water
treatment programs including oxygen scavengers, phosphate, and pH control.

All formulas are deterministic stoichiometric/empirical calculations
following zero-hallucination principles. No ML/LLM in calculation path.

Standards Reference:
    - ABMA Guidelines for Water Quality
    - NACE SP0590 Standard Practice
    - EPRI Guideline for Feed water, Boiler Water Chemistry

Example:
    >>> dose = calculate_oxygen_scavenger_dose(100, 'sodium_sulfite')
    >>> print(f"Sulfite dose: {dose['dose_ppm']:.1f} ppm")
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# Chemical stoichiometry and dosing constants
# Reference: NACE, ABMA, and chemical manufacturer data

OXYGEN_SCAVENGER_DATA = {
    "sodium_sulfite": {
        "name": "Sodium Sulfite (Na2SO3)",
        "molecular_weight": 126.04,
        "stoichiometry": 8.0,  # 8 ppm Na2SO3 per 1 ppm O2
        "excess_factor": 1.2,  # 20% excess for safety
        "target_residual_ppm": {"min": 20, "max": 60, "typical": 40},
        "max_pressure_psig": 900,  # Not recommended above 900 psig
        "cost_per_lb": 0.35,
    },
    "sodium_bisulfite": {
        "name": "Sodium Bisulfite (NaHSO3)",
        "molecular_weight": 104.06,
        "stoichiometry": 6.5,  # 6.5 ppm per 1 ppm O2
        "excess_factor": 1.2,
        "target_residual_ppm": {"min": 20, "max": 60, "typical": 40},
        "max_pressure_psig": 900,
        "cost_per_lb": 0.40,
    },
    "hydrazine": {
        "name": "Hydrazine (N2H4)",
        "molecular_weight": 32.05,
        "stoichiometry": 1.0,  # 1 ppm per 1 ppm O2
        "excess_factor": 1.3,  # 30% excess typical
        "target_residual_ppm": {"min": 0.02, "max": 0.1, "typical": 0.05},
        "max_pressure_psig": 3000,  # For high pressure systems
        "cost_per_lb": 15.0,  # More expensive, specialty chemical
    },
    "carbohydrazide": {
        "name": "Carbohydrazide (CH6N4O)",
        "molecular_weight": 90.08,
        "stoichiometry": 1.4,  # 1.4 ppm per 1 ppm O2
        "excess_factor": 1.2,
        "target_residual_ppm": {"min": 0.05, "max": 0.2, "typical": 0.1},
        "max_pressure_psig": 2400,
        "cost_per_lb": 8.0,
    },
    "erythorbic_acid": {
        "name": "Erythorbic Acid (C6H8O6)",
        "molecular_weight": 176.12,
        "stoichiometry": 5.5,  # 5.5 ppm per 1 ppm O2
        "excess_factor": 1.2,
        "target_residual_ppm": {"min": 5, "max": 20, "typical": 10},
        "max_pressure_psig": 600,  # Lower pressure systems
        "cost_per_lb": 2.0,
    },
}

PHOSPHATE_DATA = {
    "trisodium_phosphate": {
        "name": "Trisodium Phosphate (Na3PO4)",
        "formula": "Na3PO4",
        "molecular_weight": 163.94,
        "po4_fraction": 0.58,  # PO4 content as fraction
        "alkalinity_contribution": 0.85,  # ppm alkalinity per ppm phosphate
        "target_residual_ppm": {"low_p": 45, "med_p": 30, "high_p": 10},
        "cost_per_lb": 0.50,
    },
    "disodium_phosphate": {
        "name": "Disodium Phosphate (Na2HPO4)",
        "formula": "Na2HPO4",
        "molecular_weight": 141.96,
        "po4_fraction": 0.67,
        "alkalinity_contribution": 0.60,
        "target_residual_ppm": {"low_p": 50, "med_p": 35, "high_p": 12},
        "cost_per_lb": 0.45,
    },
    "monosodium_phosphate": {
        "name": "Monosodium Phosphate (NaH2PO4)",
        "formula": "NaH2PO4",
        "molecular_weight": 119.98,
        "po4_fraction": 0.79,
        "alkalinity_contribution": 0.0,  # Acidic, reduces alkalinity
        "target_residual_ppm": {"low_p": 60, "med_p": 40, "high_p": 15},
        "cost_per_lb": 0.40,
    },
}

PH_CONTROL_DATA = {
    "caustic_soda": {
        "name": "Sodium Hydroxide (NaOH)",
        "molecular_weight": 40.0,
        "ph_change_per_ppm": 0.02,  # Approximate pH change per ppm at typical levels
        "alkalinity_per_ppm": 1.25,  # ppm alkalinity (as CaCO3) per ppm NaOH
        "cost_per_lb": 0.25,
    },
    "soda_ash": {
        "name": "Sodium Carbonate (Na2CO3)",
        "molecular_weight": 105.99,
        "ph_change_per_ppm": 0.01,
        "alkalinity_per_ppm": 0.95,
        "cost_per_lb": 0.15,
    },
    "amine": {
        "name": "Neutralizing Amine",
        "molecular_weight": 45.0,  # Average for morpholine/cyclohexylamine
        "ph_change_per_ppm": 0.015,
        "alkalinity_per_ppm": 0,  # Volatile, doesn't affect boiler alkalinity
        "cost_per_lb": 3.0,
    },
}


def calculate_oxygen_scavenger_dose(
    dissolved_oxygen_ppb: float,
    scavenger_type: str = "sodium_sulfite",
    target_residual: Optional[float] = None,
    current_residual: Optional[float] = None,
    feedwater_flow_gpm: float = 100
) -> Dict[str, Any]:
    """
    Calculate oxygen scavenger dosing requirement.

    Based on stoichiometric reaction plus excess for safety.
    Example: Na2SO3 + 0.5 O2 -> Na2SO4

    Args:
        dissolved_oxygen_ppb: Dissolved oxygen in feedwater (ppb).
        scavenger_type: Type of oxygen scavenger.
        target_residual: Target residual in boiler water (ppm).
        current_residual: Current measured residual (ppm).
        feedwater_flow_gpm: Feedwater flow rate in GPM.

    Returns:
        Dictionary with dosing recommendation.

    Raises:
        ValueError: If unknown scavenger type.

    Example:
        >>> dose = calculate_oxygen_scavenger_dose(100, 'sodium_sulfite')
        >>> print(f"Dose: {dose['dose_ppm']:.2f} ppm")
    """
    if scavenger_type not in OXYGEN_SCAVENGER_DATA:
        raise ValueError(f"Unknown scavenger type: {scavenger_type}")

    scavenger = OXYGEN_SCAVENGER_DATA[scavenger_type]

    # Convert ppb to ppm for calculation
    o2_ppm = dissolved_oxygen_ppb / 1000

    # Stoichiometric dose
    stoich_dose = o2_ppm * scavenger['stoichiometry']

    # Apply excess factor
    recommended_dose = stoich_dose * scavenger['excess_factor']

    # Get target residual
    if target_residual is None:
        target_residual = scavenger['target_residual_ppm']['typical']

    # Adjust for current residual if provided
    adjustment = 0
    if current_residual is not None:
        residual_diff = target_residual - current_residual
        if residual_diff > 0:
            adjustment = residual_diff * 0.5  # Gradual adjustment

    final_dose = recommended_dose + adjustment

    # Calculate feed rate
    # lb/hr = ppm * gpm * 8.34 / 1,000,000 * 60
    feed_rate_lb_hr = final_dose * feedwater_flow_gpm * 8.34 * 60 / 1_000_000

    # Calculate daily cost
    daily_cost = feed_rate_lb_hr * 24 * scavenger['cost_per_lb']

    logger.debug(
        f"O2 scavenger dose: {final_dose:.2f} ppm {scavenger_type}, "
        f"feed rate: {feed_rate_lb_hr:.4f} lb/hr"
    )

    return {
        'chemical_type': scavenger_type,
        'chemical_name': scavenger['name'],
        'dissolved_oxygen_ppb': dissolved_oxygen_ppb,
        'stoichiometric_dose_ppm': round(stoich_dose, 3),
        'recommended_dose_ppm': round(recommended_dose, 3),
        'final_dose_ppm': round(final_dose, 3),
        'target_residual_ppm': target_residual,
        'current_residual_ppm': current_residual,
        'feed_rate_lb_hr': round(feed_rate_lb_hr, 4),
        'feed_rate_gph': round(feed_rate_lb_hr / 8.34, 4),  # Approximate GPH
        'daily_cost': round(daily_cost, 2),
        'max_pressure_psig': scavenger['max_pressure_psig'],
    }


def calculate_phosphate_dose(
    current_phosphate_ppm: float,
    target_phosphate_ppm: float,
    boiler_water_volume_gal: float,
    makeup_rate_gpm: float,
    phosphate_type: str = "trisodium_phosphate",
    blowdown_rate_percent: float = 5
) -> Dict[str, Any]:
    """
    Calculate phosphate treatment dosing requirement.

    Phosphate programs maintain a residual for pH buffering
    and scale prevention.

    Args:
        current_phosphate_ppm: Current phosphate residual in ppm.
        target_phosphate_ppm: Target phosphate residual in ppm.
        boiler_water_volume_gal: Boiler water volume in gallons.
        makeup_rate_gpm: Makeup water rate in GPM.
        phosphate_type: Type of phosphate chemical.
        blowdown_rate_percent: Blowdown rate percentage.

    Returns:
        Dictionary with dosing recommendation.

    Example:
        >>> dose = calculate_phosphate_dose(15, 30, 5000, 50)
        >>> print(f"Dose rate: {dose['feed_rate_lb_hr']:.3f} lb/hr")
    """
    if phosphate_type not in PHOSPHATE_DATA:
        raise ValueError(f"Unknown phosphate type: {phosphate_type}")

    phosphate = PHOSPHATE_DATA[phosphate_type]

    # Calculate phosphate deficit
    deficit_ppm = target_phosphate_ppm - current_phosphate_ppm

    # Phosphate loss through blowdown (continuous consumption)
    # Loss rate = target * blowdown_rate / cycles
    cycles = 100 / blowdown_rate_percent if blowdown_rate_percent > 0 else 10
    loss_rate_ppm_hr = target_phosphate_ppm * (blowdown_rate_percent / 100) * (makeup_rate_gpm * 60 / boiler_water_volume_gal)

    # Initial correction dose (if deficit exists)
    if deficit_ppm > 0:
        correction_lb = (deficit_ppm * boiler_water_volume_gal * 8.34) / 1_000_000
    else:
        correction_lb = 0

    # Maintenance dose rate
    maintenance_lb_hr = (target_phosphate_ppm * makeup_rate_gpm * 8.34 * 60) / 1_000_000 / phosphate['po4_fraction']

    # Convert to chemical weight (accounting for PO4 fraction)
    actual_chemical_lb_hr = maintenance_lb_hr / phosphate['po4_fraction']

    # Calculate alkalinity contribution
    alkalinity_change = target_phosphate_ppm * phosphate['alkalinity_contribution']

    # Calculate daily cost
    daily_cost = actual_chemical_lb_hr * 24 * phosphate['cost_per_lb']

    return {
        'chemical_type': phosphate_type,
        'chemical_name': phosphate['name'],
        'current_residual_ppm': current_phosphate_ppm,
        'target_residual_ppm': target_phosphate_ppm,
        'deficit_ppm': round(max(0, deficit_ppm), 2),
        'correction_dose_lb': round(correction_lb, 4),
        'maintenance_rate_lb_hr': round(actual_chemical_lb_hr, 4),
        'feed_rate_lb_hr': round(actual_chemical_lb_hr, 4),
        'feed_rate_gph': round(actual_chemical_lb_hr / 10, 4),  # Assuming 10 lb/gal solution
        'alkalinity_contribution': round(alkalinity_change, 1),
        'daily_cost': round(daily_cost, 2),
    }


def calculate_ph_adjustment_dose(
    current_ph: float,
    target_ph: float,
    boiler_water_volume_gal: float,
    chemical_type: str = "caustic_soda",
    current_alkalinity_ppm: float = 0
) -> Dict[str, Any]:
    """
    Calculate pH adjustment chemical dosing.

    Uses empirical relationships for pH control in boiler water.
    Actual dose may need adjustment based on buffer capacity.

    Args:
        current_ph: Current boiler water pH.
        target_ph: Target pH value.
        boiler_water_volume_gal: Boiler water volume in gallons.
        chemical_type: Type of pH control chemical.
        current_alkalinity_ppm: Current alkalinity for buffer estimation.

    Returns:
        Dictionary with dosing recommendation.

    Example:
        >>> dose = calculate_ph_adjustment_dose(9.5, 10.5, 5000, 'caustic_soda')
        >>> print(f"Dose: {dose['dose_lb']:.3f} lb")
    """
    if chemical_type not in PH_CONTROL_DATA:
        raise ValueError(f"Unknown pH chemical: {chemical_type}")

    chemical = PH_CONTROL_DATA[chemical_type]

    ph_change_needed = target_ph - current_ph

    if ph_change_needed == 0:
        return {
            'chemical_type': chemical_type,
            'chemical_name': chemical['name'],
            'current_ph': current_ph,
            'target_ph': target_ph,
            'ph_change_needed': 0,
            'dose_ppm': 0,
            'dose_lb': 0,
            'reason': "pH is at target - no adjustment needed"
        }

    # Buffer capacity factor (higher alkalinity = more chemical needed)
    buffer_factor = 1 + (current_alkalinity_ppm / 500)

    # Estimate dose needed
    # This is a simplified empirical relationship
    dose_ppm = abs(ph_change_needed) / chemical['ph_change_per_ppm'] * buffer_factor

    # Limit maximum single dose
    max_dose_ppm = 50  # Safety limit
    dose_ppm = min(dose_ppm, max_dose_ppm)

    # If lowering pH, use acid or reduce caustic feed
    if ph_change_needed < 0:
        action = "reduce" if chemical_type == "caustic_soda" else "add"
        dose_ppm = dose_ppm * 0.5  # More conservative for pH reduction
    else:
        action = "add"

    # Calculate pounds needed
    dose_lb = (dose_ppm * boiler_water_volume_gal * 8.34) / 1_000_000

    # Alkalinity change from this dose
    alkalinity_change = dose_ppm * chemical['alkalinity_per_ppm']

    return {
        'chemical_type': chemical_type,
        'chemical_name': chemical['name'],
        'current_ph': current_ph,
        'target_ph': target_ph,
        'ph_change_needed': round(ph_change_needed, 2),
        'dose_ppm': round(dose_ppm, 2),
        'dose_lb': round(dose_lb, 4),
        'action': action,
        'alkalinity_change_ppm': round(alkalinity_change, 1),
        'new_alkalinity_estimate': round(current_alkalinity_ppm + alkalinity_change, 1),
    }


def select_optimal_scavenger(
    operating_pressure_psig: float,
    dissolved_oxygen_ppb: float,
    environmental_preference: str = "standard"
) -> Dict[str, Any]:
    """
    Select optimal oxygen scavenger based on conditions.

    Args:
        operating_pressure_psig: Boiler operating pressure.
        dissolved_oxygen_ppb: Feedwater dissolved oxygen.
        environmental_preference: "standard", "low_toxicity", or "organic".

    Returns:
        Dictionary with recommended scavenger and reasoning.

    Example:
        >>> result = select_optimal_scavenger(600, 100)
        >>> print(f"Recommended: {result['recommended_scavenger']}")
    """
    candidates = []
    reasoning = []

    for name, data in OXYGEN_SCAVENGER_DATA.items():
        score = 100

        # Pressure compatibility
        if operating_pressure_psig > data['max_pressure_psig']:
            score -= 50
            reasoning.append(f"{name}: pressure exceeds max {data['max_pressure_psig']} psig")
            continue

        # Cost effectiveness (lower cost = higher score)
        cost_score = 20 - min(20, data['cost_per_lb'] * 5)
        score += cost_score

        # Environmental preference
        if environmental_preference == "low_toxicity":
            if name in ["hydrazine"]:
                score -= 30  # Carcinogen concerns
            if name in ["erythorbic_acid", "carbohydrazide"]:
                score += 10
        elif environmental_preference == "organic":
            if name in ["erythorbic_acid"]:
                score += 20
            if name in ["sodium_sulfite", "sodium_bisulfite"]:
                score -= 10

        # High pressure preference
        if operating_pressure_psig > 600:
            if name in ["hydrazine", "carbohydrazide"]:
                score += 15  # Better for high pressure
            if name in ["sodium_sulfite"]:
                score -= 10  # TDS concerns at high pressure

        candidates.append({
            'name': name,
            'score': score,
            'max_pressure': data['max_pressure_psig'],
            'cost_per_lb': data['cost_per_lb'],
        })

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    recommended = candidates[0] if candidates else None

    return {
        'recommended_scavenger': recommended['name'] if recommended else None,
        'score': recommended['score'] if recommended else 0,
        'alternatives': [c['name'] for c in candidates[1:3]],
        'all_candidates': candidates,
        'selection_factors': reasoning,
    }


def generate_dosing_recommendation(
    chemistry_data: Dict[str, float],
    operating_params: Dict[str, float],
    current_residuals: Dict[str, float],
    chemical_inventory: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive dosing recommendations.

    Analyzes current water chemistry and provides dosing
    recommendations for all treatment chemicals.

    Args:
        chemistry_data: Current water chemistry measurements.
        operating_params: Operating parameters (pressure, flow, etc.).
        current_residuals: Current chemical residual levels.
        chemical_inventory: Current chemical inventory.

    Returns:
        List of dosing recommendations sorted by priority.

    Example:
        >>> recommendations = generate_dosing_recommendation(
        ...     {'dissolved_oxygen_ppb': 100, 'ph': 10.0},
        ...     {'pressure_psig': 600, 'feedwater_gpm': 100},
        ...     {'sulfite_ppm': 20},
        ...     []
        ... )
    """
    recommendations = []

    # Oxygen scavenger recommendation
    dissolved_o2 = chemistry_data.get('dissolved_oxygen_ppb', 0)
    pressure = operating_params.get('operating_pressure_psig', 300)
    feedwater_flow = operating_params.get('feedwater_flow_gpm', 100)

    if dissolved_o2 > 7:  # Above ASME limit for most pressures
        # Select appropriate scavenger
        scavenger_selection = select_optimal_scavenger(pressure, dissolved_o2)
        scavenger_type = scavenger_selection['recommended_scavenger'] or 'sodium_sulfite'

        current_residual = current_residuals.get('sulfite_ppm') or current_residuals.get('hydrazine_ppm')

        dose = calculate_oxygen_scavenger_dose(
            dissolved_o2,
            scavenger_type,
            current_residual=current_residual,
            feedwater_flow_gpm=feedwater_flow
        )

        priority = "IMMEDIATE" if dissolved_o2 > 50 else "HIGH" if dissolved_o2 > 20 else "NORMAL"

        recommendations.append({
            'chemical_type': scavenger_type,
            'category': 'oxygen_scavenger',
            'current_dose_rate_gph': dose.get('feed_rate_gph', 0),
            'recommended_dose_rate_gph': dose['feed_rate_gph'],
            'target_residual': dose['target_residual_ppm'],
            'current_residual': current_residual,
            'priority': priority,
            'reason': f"Dissolved O2 at {dissolved_o2} ppb exceeds target of 7 ppb",
            'daily_cost_change': dose['daily_cost'],
        })

    # pH control recommendation
    current_ph = chemistry_data.get('ph', 10.0)
    target_ph = 10.5 if pressure < 600 else 9.8 if pressure < 1500 else 9.0

    if abs(current_ph - target_ph) > 0.3:
        ph_dose = calculate_ph_adjustment_dose(
            current_ph,
            target_ph,
            operating_params.get('boiler_volume_gal', 5000),
            current_alkalinity_ppm=chemistry_data.get('alkalinity_ppm_caco3', 200)
        )

        priority = "HIGH" if abs(current_ph - target_ph) > 1.0 else "NORMAL"

        recommendations.append({
            'chemical_type': 'caustic_soda',
            'category': 'ph_control',
            'current_dose_rate_gph': 0,
            'recommended_dose_rate_gph': ph_dose['dose_lb'] / 10,  # Assuming 10 lb/gal
            'target_residual': target_ph,
            'current_residual': current_ph,
            'priority': priority,
            'reason': f"pH at {current_ph} needs adjustment to target {target_ph}",
            'daily_cost_change': ph_dose['dose_lb'] * 0.25 * 24,  # Rough estimate
        })

    # Phosphate recommendation
    current_po4 = chemistry_data.get('phosphate_ppm', 0)
    target_po4 = 30 if pressure < 600 else 15 if pressure < 900 else 8

    if current_po4 < target_po4 * 0.7 or current_po4 > target_po4 * 1.5:
        po4_dose = calculate_phosphate_dose(
            current_po4,
            target_po4,
            operating_params.get('boiler_volume_gal', 5000),
            feedwater_flow,
            blowdown_rate_percent=operating_params.get('blowdown_rate_percent', 5)
        )

        if current_po4 < target_po4 * 0.5:
            priority = "HIGH"
            reason = f"Phosphate at {current_po4} ppm is critically low (target: {target_po4})"
        elif current_po4 < target_po4 * 0.7:
            priority = "NORMAL"
            reason = f"Phosphate at {current_po4} ppm is below target of {target_po4}"
        else:
            priority = "LOW"
            reason = f"Phosphate at {current_po4} ppm is above target of {target_po4}"

        recommendations.append({
            'chemical_type': 'trisodium_phosphate',
            'category': 'scale_control',
            'current_dose_rate_gph': po4_dose.get('feed_rate_gph', 0),
            'recommended_dose_rate_gph': po4_dose['feed_rate_gph'],
            'target_residual': target_po4,
            'current_residual': current_po4,
            'priority': priority,
            'reason': reason,
            'daily_cost_change': po4_dose['daily_cost'],
        })

    # Sort by priority
    priority_order = {"IMMEDIATE": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3, "NONE": 4}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

    logger.info(f"Generated {len(recommendations)} dosing recommendations")

    return recommendations


def calculate_chemical_consumption(
    dose_rate_gph: float,
    chemical_concentration_percent: float,
    operating_hours_per_day: float = 24
) -> Dict[str, float]:
    """
    Calculate chemical consumption rates.

    Args:
        dose_rate_gph: Dosing rate in gallons per hour.
        chemical_concentration_percent: Chemical concentration (%).
        operating_hours_per_day: Hours of operation per day.

    Returns:
        Dictionary with consumption rates.

    Example:
        >>> consumption = calculate_chemical_consumption(0.5, 30, 24)
        >>> print(f"Daily: {consumption['gallons_per_day']:.1f} gal")
    """
    gal_per_day = dose_rate_gph * operating_hours_per_day
    gal_per_week = gal_per_day * 7
    gal_per_month = gal_per_day * 30

    # Active chemical consumption
    active_lb_per_day = gal_per_day * 8.34 * (chemical_concentration_percent / 100)

    return {
        'gallons_per_hour': round(dose_rate_gph, 3),
        'gallons_per_day': round(gal_per_day, 1),
        'gallons_per_week': round(gal_per_week, 1),
        'gallons_per_month': round(gal_per_month, 0),
        'active_chemical_lb_per_day': round(active_lb_per_day, 2),
    }


def estimate_days_of_supply(
    tank_level_percent: float,
    tank_capacity_gal: float,
    dose_rate_gph: float,
    operating_hours_per_day: float = 24
) -> float:
    """
    Estimate remaining days of chemical supply.

    Args:
        tank_level_percent: Current tank level (%).
        tank_capacity_gal: Tank capacity in gallons.
        dose_rate_gph: Current dosing rate in GPH.
        operating_hours_per_day: Operating hours per day.

    Returns:
        Estimated days of supply remaining.

    Example:
        >>> days = estimate_days_of_supply(50, 500, 0.5, 24)
        >>> print(f"Days remaining: {days:.1f}")
    """
    if dose_rate_gph <= 0:
        return float('inf')

    current_volume = tank_capacity_gal * (tank_level_percent / 100)
    daily_consumption = dose_rate_gph * operating_hours_per_day
    days_remaining = current_volume / daily_consumption

    return round(days_remaining, 1)
