"""
Refractory Degradation Calculator

Physics-based calculations for refractory wear, degradation, and remaining
life estimation based on operating conditions and historical data.

References:
    - ASTM C704: Standard Test Method for Abrasion Resistance of Refractory Materials
    - ASTM C288: Standard Test Method for Disintegration of Refractories
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DegradationMode(str, Enum):
    """Primary refractory degradation mechanisms."""
    EROSION = "EROSION"              # Mechanical wear from particle impact
    CORROSION = "CORROSION"          # Chemical attack (slag, alkali)
    SPALLING = "SPALLING"            # Thermal shock cracking
    OXIDATION = "OXIDATION"          # High-temp oxidation
    THERMAL_CYCLING = "THERMAL_CYCLING"  # Fatigue from temp cycles


# Typical wear rates (mm/year) by application
TYPICAL_WEAR_RATES = {
    "furnace_hearth": 5.0,
    "furnace_sidewall": 8.0,
    "furnace_roof": 12.0,
    "ladle_lining": 25.0,
    "kiln_hot_zone": 15.0,
    "incinerator": 20.0,
    "boiler": 3.0,
}


def calculate_wear_rate(
    thickness_loss_mm: float,
    operating_hours: float,
    operating_factor: float = 1.0
) -> float:
    """
    Calculate refractory wear rate from measured thickness loss.

    Wear rate is typically expressed in mm/year for planning purposes.

    Args:
        thickness_loss_mm: Total thickness lost in mm.
        operating_hours: Hours of operation during measurement period.
        operating_factor: Operating hours per year (default 8760).

    Returns:
        Wear rate in mm/year.

    Example:
        >>> rate = calculate_wear_rate(10, 4000)
        >>> print(f"Wear rate: {rate:.1f} mm/year")
    """
    if operating_hours <= 0:
        logger.warning("Operating hours is zero, cannot calculate wear rate")
        return 0.0

    # Hours per year (full year = 8760)
    hours_per_year = 8760 * operating_factor

    # Wear rate = (loss / hours) * hours_per_year
    wear_rate = (thickness_loss_mm / operating_hours) * hours_per_year

    logger.debug(
        f"Wear rate calculation: loss={thickness_loss_mm:.1f}mm, "
        f"hours={operating_hours:.0f}, rate={wear_rate:.2f} mm/year"
    )

    return wear_rate


def calculate_remaining_life(
    current_thickness_mm: float,
    minimum_thickness_mm: float,
    wear_rate_mm_year: float,
    safety_factor: float = 1.2
) -> float:
    """
    Calculate remaining service life of refractory.

    Remaining life = (current - minimum) / (wear_rate * safety_factor)

    Args:
        current_thickness_mm: Current measured thickness.
        minimum_thickness_mm: Minimum safe operating thickness.
        wear_rate_mm_year: Annual wear rate.
        safety_factor: Factor for conservative estimate (default 1.2).

    Returns:
        Remaining life in years.

    Example:
        >>> life = calculate_remaining_life(150, 50, 10, 1.2)
        >>> print(f"Remaining life: {life:.1f} years")
    """
    if wear_rate_mm_year <= 0:
        logger.info("Zero or negative wear rate - assuming infinite life")
        return float('inf')

    available_thickness = current_thickness_mm - minimum_thickness_mm

    if available_thickness <= 0:
        logger.warning("Current thickness at or below minimum!")
        return 0.0

    # Apply safety factor to wear rate (conservative)
    effective_wear_rate = wear_rate_mm_year * safety_factor

    remaining_life = available_thickness / effective_wear_rate

    logger.debug(
        f"Remaining life: current={current_thickness_mm:.0f}mm, "
        f"min={minimum_thickness_mm:.0f}mm, "
        f"wear={wear_rate_mm_year:.1f}mm/yr, life={remaining_life:.2f} years"
    )

    return remaining_life


def calculate_degradation_factor(
    operating_temperature_c: float,
    max_service_temperature_c: float,
    thermal_cycles: int,
    chemical_exposure_factor: float = 1.0,
    age_years: float = 0.0
) -> float:
    """
    Calculate overall degradation factor for thermal conductivity.

    The degradation factor accounts for:
    1. Temperature stress (operating near limits)
    2. Thermal cycling fatigue
    3. Chemical attack
    4. Age-related deterioration

    A factor > 1.0 indicates increased thermal conductivity
    (worse insulation due to cracks, porosity changes).

    Args:
        operating_temperature_c: Current operating temperature.
        max_service_temperature_c: Maximum rated temperature.
        thermal_cycles: Number of significant thermal cycles.
        chemical_exposure_factor: 1.0 = none, up to 1.5 for severe.
        age_years: Material age in years.

    Returns:
        Degradation factor (1.0 = new, >1.0 = degraded).

    Example:
        >>> factor = calculate_degradation_factor(1200, 1400, 500, 1.1, 5)
        >>> print(f"Degradation factor: {factor:.2f}")
    """
    # Temperature stress factor
    temp_ratio = operating_temperature_c / max_service_temperature_c
    if temp_ratio > 1.0:
        temp_factor = 1.0 + (temp_ratio - 1.0) * 2  # Severe penalty
    elif temp_ratio > 0.9:
        temp_factor = 1.0 + (temp_ratio - 0.9) * 0.5
    else:
        temp_factor = 1.0

    # Thermal cycling factor (fatigue)
    # Typical ceramic fatigue: noticeable after ~100 cycles, significant after ~1000
    if thermal_cycles < 100:
        cycle_factor = 1.0
    elif thermal_cycles < 1000:
        cycle_factor = 1.0 + (thermal_cycles - 100) / 9000  # Up to 0.1
    else:
        cycle_factor = 1.1 + (thermal_cycles - 1000) / 10000  # Continuing

    # Age factor (slow deterioration)
    # Assume ~2% per year degradation
    age_factor = 1.0 + 0.02 * age_years

    # Combine factors (multiplicative)
    total_factor = temp_factor * cycle_factor * chemical_exposure_factor * age_factor

    # Cap at reasonable maximum
    total_factor = min(total_factor, 2.5)

    logger.debug(
        f"Degradation factors: temp={temp_factor:.3f}, cycle={cycle_factor:.3f}, "
        f"chem={chemical_exposure_factor:.3f}, age={age_factor:.3f}, "
        f"total={total_factor:.3f}"
    )

    return total_factor


def estimate_spalling_risk(
    temperature_gradient_c_per_m: float,
    thermal_shock_resistance: float = 1.0,
    current_cycle_rate: float = 0.0,
    existing_crack_density: float = 0.0
) -> Tuple[float, str]:
    """
    Estimate spalling risk based on thermal conditions.

    Spalling occurs when thermal stresses exceed material strength:
    - Temperature gradient induces differential expansion
    - Rapid heating/cooling causes thermal shock
    - Pre-existing cracks concentrate stress

    Args:
        temperature_gradient_c_per_m: Temperature gradient through material.
        thermal_shock_resistance: Material property (1.0 = typical firebrick).
        current_cycle_rate: Thermal cycles per day.
        existing_crack_density: Observed cracks (0 = none, 1 = severe).

    Returns:
        Tuple of (risk_score 0-100, risk_level string).

    Example:
        >>> risk, level = estimate_spalling_risk(5000, 1.0, 2, 0.2)
        >>> print(f"Spalling risk: {risk:.0f}% ({level})")
    """
    # Base risk from gradient
    # Typical limit ~10000 C/m for dense refractory
    gradient_risk = min(100, (temperature_gradient_c_per_m / 10000) * 50)

    # Thermal shock risk from cycling
    # >5 cycles/day is severe
    cycle_risk = min(100, current_cycle_rate * 10)

    # Crack propagation risk
    crack_risk = existing_crack_density * 100

    # Adjust for material resistance
    material_factor = 1.0 / thermal_shock_resistance

    # Combined risk (weighted)
    total_risk = (
        0.4 * gradient_risk +
        0.3 * cycle_risk +
        0.3 * crack_risk
    ) * material_factor

    total_risk = min(100, max(0, total_risk))

    # Determine level
    if total_risk >= 70:
        level = "HIGH"
    elif total_risk >= 40:
        level = "MEDIUM"
    elif total_risk >= 15:
        level = "LOW"
    else:
        level = "MINIMAL"

    logger.debug(
        f"Spalling risk: gradient={gradient_risk:.0f}%, cycle={cycle_risk:.0f}%, "
        f"crack={crack_risk:.0f}%, total={total_risk:.0f}% ({level})"
    )

    return total_risk, level


def calculate_minimum_thickness(
    design_thickness_mm: float,
    material_type: str = "firebrick",
    application: str = "furnace_sidewall"
) -> float:
    """
    Calculate minimum safe operating thickness.

    Minimum thickness is typically 30-50% of design based on:
    - Structural requirements
    - Heat containment needs
    - Safety margins

    Args:
        design_thickness_mm: Original design thickness.
        material_type: Type of refractory.
        application: Furnace location/application.

    Returns:
        Minimum safe thickness in mm.
    """
    # Minimum percentage by application
    min_percentages = {
        "furnace_hearth": 0.40,
        "furnace_sidewall": 0.35,
        "furnace_roof": 0.30,
        "ladle_lining": 0.25,
        "kiln_hot_zone": 0.35,
        "incinerator": 0.30,
        "boiler": 0.40,
    }

    min_pct = min_percentages.get(application.lower(), 0.30)
    minimum_thickness = design_thickness_mm * min_pct

    return minimum_thickness


def predict_failure_date(
    current_thickness_mm: float,
    wear_rate_mm_year: float,
    minimum_thickness_mm: float,
    reference_date: Optional[str] = None
) -> Dict[str, any]:
    """
    Predict when refractory will reach minimum thickness.

    Args:
        current_thickness_mm: Current measured thickness.
        wear_rate_mm_year: Annual wear rate.
        minimum_thickness_mm: Minimum acceptable thickness.
        reference_date: Date of current measurement (ISO format).

    Returns:
        Dictionary with predicted failure date and confidence.
    """
    from datetime import datetime, timedelta

    remaining_years = calculate_remaining_life(
        current_thickness_mm,
        minimum_thickness_mm,
        wear_rate_mm_year,
        safety_factor=1.0  # No safety factor for prediction
    )

    if reference_date:
        try:
            ref = datetime.fromisoformat(reference_date)
        except ValueError:
            ref = datetime.now()
    else:
        ref = datetime.now()

    if remaining_years == float('inf'):
        return {
            "predicted_failure_date": None,
            "remaining_days": float('inf'),
            "confidence": "LOW",
            "note": "Cannot predict - zero or negative wear rate"
        }

    remaining_days = remaining_years * 365.25
    predicted_date = ref + timedelta(days=remaining_days)

    # Confidence based on data quality
    if remaining_years > 5:
        confidence = "LOW"
    elif remaining_years > 2:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    return {
        "predicted_failure_date": predicted_date.isoformat()[:10],
        "remaining_days": int(remaining_days),
        "remaining_years": round(remaining_years, 2),
        "confidence": confidence,
    }
