"""
GL-032 Refractory Monitor - Calculation Formulas

This module implements deterministic calculations for refractory health assessment
including heat loss calculations and remaining life prediction.

ZERO-HALLUCINATION: All calculations use physics-based formulas and
industry-standard correlations.

References:
- API 560: Fired Heaters for General Refinery Service
- ASTM C155: Standard Classification of Insulating Firebrick
- Perry's Chemical Engineers' Handbook (Heat Transfer)
"""

import math
import logging
from typing import Tuple, Dict, List, Optional, NamedTuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HeatLossResult(NamedTuple):
    """Result of heat loss calculation."""
    heat_loss_watts: float
    heat_loss_watts_per_m2: float
    surface_efficiency: float
    insulation_effectiveness: float


class RemainingLifeResult(NamedTuple):
    """Result of remaining life calculation."""
    remaining_life_days: int
    remaining_life_percent: float
    degradation_rate_per_day: float
    failure_date_estimate: datetime
    confidence_level: float


class HotspotResult(NamedTuple):
    """Result of hotspot analysis."""
    location_x: float
    location_y: float
    temperature_celsius: float
    severity: str
    heat_loss_kw: float
    recommended_action: str


def calculate_heat_loss_through_wall(
    hot_face_temp_celsius: float,
    cold_face_temp_celsius: float,
    wall_thickness_meters: float,
    thermal_conductivity_w_per_m_k: float,
    area_m2: float
) -> float:
    """
    Calculate heat loss through refractory wall using Fourier's Law.

    ZERO-HALLUCINATION FORMULA (Fourier's Law):
    Q = (k * A * (T_hot - T_cold)) / L

    Where:
        Q = Heat transfer rate (W)
        k = Thermal conductivity (W/m-K)
        A = Surface area (m2)
        T_hot = Hot face temperature (C or K, difference is same)
        T_cold = Cold face temperature (C or K)
        L = Wall thickness (m)

    Args:
        hot_face_temp_celsius: Temperature on hot side
        cold_face_temp_celsius: Temperature on cold side
        wall_thickness_meters: Total wall thickness
        thermal_conductivity_w_per_m_k: Material thermal conductivity
        area_m2: Wall surface area

    Returns:
        Heat loss in Watts

    Example:
        >>> q = calculate_heat_loss_through_wall(1000, 80, 0.3, 1.5, 10)
        >>> print(f"Heat loss: {q:.0f} W")
    """
    if wall_thickness_meters <= 0:
        raise ValueError(f"Wall thickness must be positive: {wall_thickness_meters}")
    if thermal_conductivity_w_per_m_k <= 0:
        raise ValueError(f"Thermal conductivity must be positive: {thermal_conductivity_w_per_m_k}")
    if area_m2 <= 0:
        raise ValueError(f"Area must be positive: {area_m2}")

    delta_t = hot_face_temp_celsius - cold_face_temp_celsius

    # Fourier's Law
    q_watts = (thermal_conductivity_w_per_m_k * area_m2 * delta_t) / wall_thickness_meters

    logger.debug(
        f"Heat loss calculation: k={thermal_conductivity_w_per_m_k} W/m-K, "
        f"A={area_m2} m2, dT={delta_t} C, L={wall_thickness_meters} m, "
        f"Q={q_watts:.0f} W"
    )

    return q_watts


def calculate_multilayer_heat_loss(
    hot_face_temp_celsius: float,
    ambient_temp_celsius: float,
    layers: List[Dict[str, float]],
    area_m2: float,
    external_h_w_per_m2_k: float = 10.0
) -> Tuple[float, List[float]]:
    """
    Calculate heat loss through multi-layer refractory with convection.

    ZERO-HALLUCINATION FORMULA (Thermal Resistance Network):
    Q = (T_hot - T_ambient) / R_total

    R_total = Sum(L_i / k_i) + 1/h_external

    Args:
        hot_face_temp_celsius: Hot face temperature
        ambient_temp_celsius: Ambient temperature
        layers: List of dicts with 'thickness_m' and 'conductivity_w_per_m_k'
        area_m2: Wall area
        external_h_w_per_m2_k: External convection coefficient

    Returns:
        Tuple of (total_heat_loss_watts, interface_temperatures)
    """
    if not layers:
        raise ValueError("At least one layer is required")

    # Calculate total thermal resistance per unit area
    r_total = 0.0

    for layer in layers:
        thickness = layer.get('thickness_m', 0)
        conductivity = layer.get('conductivity_w_per_m_k', 1.0)
        if thickness <= 0 or conductivity <= 0:
            raise ValueError(f"Invalid layer: {layer}")
        r_total += thickness / conductivity

    # Add external convection resistance
    r_total += 1.0 / external_h_w_per_m2_k

    # Calculate heat flux
    delta_t = hot_face_temp_celsius - ambient_temp_celsius
    q_per_area = delta_t / r_total  # W/m2
    q_total = q_per_area * area_m2  # W

    # Calculate interface temperatures
    interface_temps = [hot_face_temp_celsius]
    current_temp = hot_face_temp_celsius

    for layer in layers:
        thickness = layer.get('thickness_m')
        conductivity = layer.get('conductivity_w_per_m_k')
        r_layer = thickness / conductivity
        temp_drop = q_per_area * r_layer
        current_temp -= temp_drop
        interface_temps.append(current_temp)

    logger.debug(
        f"Multi-layer heat loss: R_total={r_total:.4f} m2-K/W, "
        f"Q={q_total:.0f} W ({q_per_area:.0f} W/m2)"
    )

    return q_total, interface_temps


def calculate_skin_temperature_from_heat_loss(
    heat_loss_w_per_m2: float,
    ambient_temp_celsius: float,
    convection_h: float = 10.0,
    emissivity: float = 0.9
) -> float:
    """
    Calculate skin temperature from known heat loss.

    Iterative solution for combined convection and radiation.

    ZERO-HALLUCINATION FORMULA:
    Q = h_conv * (T_skin - T_amb) + epsilon * sigma * (T_skin^4 - T_amb^4)

    For simplicity, uses linearized form when radiation is small.

    Args:
        heat_loss_w_per_m2: Heat flux at surface
        ambient_temp_celsius: Ambient temperature
        convection_h: Convection coefficient W/m2-K
        emissivity: Surface emissivity (0-1)

    Returns:
        Estimated skin temperature in Celsius
    """
    # Stefan-Boltzmann constant
    SIGMA = 5.67e-8  # W/m2-K4

    # First estimate using convection only
    t_skin_estimate = ambient_temp_celsius + heat_loss_w_per_m2 / convection_h

    # Refine with radiation (simplified)
    t_amb_k = ambient_temp_celsius + 273.15
    t_skin_k = t_skin_estimate + 273.15

    # Radiation heat transfer coefficient (linearized)
    h_rad = emissivity * SIGMA * (t_skin_k + t_amb_k) * (t_skin_k**2 + t_amb_k**2)

    # Combined coefficient
    h_total = convection_h + h_rad

    # Refined skin temperature
    t_skin_refined = ambient_temp_celsius + heat_loss_w_per_m2 / h_total

    return t_skin_refined


def calculate_thermal_gradient(
    hot_face_temp_celsius: float,
    cold_face_temp_celsius: float,
    wall_thickness_meters: float
) -> float:
    """
    Calculate thermal gradient through refractory.

    High gradients indicate potential for thermal shock and spalling.

    ZERO-HALLUCINATION FORMULA:
    Gradient = (T_hot - T_cold) / L [C/m or K/m]

    Args:
        hot_face_temp_celsius: Hot face temperature
        cold_face_temp_celsius: Cold face temperature
        wall_thickness_meters: Wall thickness

    Returns:
        Thermal gradient in C/m
    """
    if wall_thickness_meters <= 0:
        raise ValueError("Wall thickness must be positive")

    gradient = (hot_face_temp_celsius - cold_face_temp_celsius) / wall_thickness_meters

    logger.debug(f"Thermal gradient: {gradient:.0f} C/m")

    return gradient


def calculate_health_index(
    skin_temp_celsius: float,
    design_skin_temp_celsius: float,
    age_days: int,
    design_life_days: int,
    hotspot_count: int = 0,
    hotspot_severity_factor: float = 0.0
) -> float:
    """
    Calculate refractory health index (0-100).

    ZERO-HALLUCINATION: Weighted scoring based on observable parameters.

    Scoring Components (100 points total):
    - Skin temperature vs design: 40 points
    - Age vs design life: 40 points
    - Hotspot penalty: up to -20 points

    Args:
        skin_temp_celsius: Measured skin temperature
        design_skin_temp_celsius: Design/expected skin temperature
        age_days: Current age in days
        design_life_days: Design life in days
        hotspot_count: Number of hotspots detected
        hotspot_severity_factor: Severity factor for hotspots (0-1)

    Returns:
        Health index 0-100
    """
    # Temperature score (40 points max)
    if design_skin_temp_celsius > 0:
        temp_ratio = skin_temp_celsius / design_skin_temp_celsius
        if temp_ratio <= 1.0:
            temp_score = 40.0
        elif temp_ratio <= 1.2:
            temp_score = 40.0 - (temp_ratio - 1.0) * 100  # 20% over = 20 point penalty
        elif temp_ratio <= 1.5:
            temp_score = 20.0 - (temp_ratio - 1.2) * 66.7  # Rapid decline
        else:
            temp_score = 0.0
    else:
        temp_score = 40.0

    # Age score (40 points max)
    if design_life_days > 0:
        age_ratio = age_days / design_life_days
        if age_ratio <= 0.5:
            age_score = 40.0
        elif age_ratio <= 0.8:
            age_score = 40.0 - (age_ratio - 0.5) * 33.3
        elif age_ratio <= 1.0:
            age_score = 30.0 - (age_ratio - 0.8) * 100
        else:
            age_score = max(0, 10.0 - (age_ratio - 1.0) * 50)
    else:
        age_score = 40.0

    # Hotspot penalty (up to 20 points)
    hotspot_penalty = min(20.0, hotspot_count * 5 * (1 + hotspot_severity_factor))

    # Calculate total
    health_index = temp_score + age_score - hotspot_penalty
    health_index = max(0.0, min(100.0, health_index))

    logger.debug(
        f"Health index: {health_index:.1f} "
        f"(temp_score={temp_score:.1f}, age_score={age_score:.1f}, "
        f"hotspot_penalty={hotspot_penalty:.1f})"
    )

    return round(health_index, 1)


def estimate_remaining_life(
    current_health_index: float,
    health_history: List[Tuple[int, float]],  # (days_ago, health_index)
    minimum_acceptable_health: float = 30.0
) -> RemainingLifeResult:
    """
    Estimate remaining useful life based on health index trend.

    Uses linear regression on health history to project failure date.

    Args:
        current_health_index: Current health index
        health_history: List of (days_ago, health_index) tuples
        minimum_acceptable_health: Health index below which replacement needed

    Returns:
        RemainingLifeResult with remaining life estimate
    """
    if current_health_index <= minimum_acceptable_health:
        return RemainingLifeResult(
            remaining_life_days=0,
            remaining_life_percent=0.0,
            degradation_rate_per_day=0.0,
            failure_date_estimate=datetime.utcnow(),
            confidence_level=0.95
        )

    # Add current reading to history
    all_readings = [(0, current_health_index)] + list(health_history)

    if len(all_readings) < 2:
        # Assume default degradation rate of 0.01 per day
        degradation_rate = 0.01
        confidence = 0.5
    else:
        # Simple linear regression
        n = len(all_readings)
        sum_x = sum(r[0] for r in all_readings)
        sum_y = sum(r[1] for r in all_readings)
        sum_xy = sum(r[0] * r[1] for r in all_readings)
        sum_x2 = sum(r[0] ** 2 for r in all_readings)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            degradation_rate = 0.01
            confidence = 0.5
        else:
            # Slope (negative means degradation)
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            degradation_rate = abs(slope) if slope < 0 else 0.01
            confidence = min(0.95, 0.5 + len(all_readings) * 0.1)

    # Calculate remaining life
    health_to_lose = current_health_index - minimum_acceptable_health
    if degradation_rate > 0:
        remaining_days = int(health_to_lose / degradation_rate)
    else:
        remaining_days = 3650  # 10 years if no degradation

    remaining_days = max(0, min(remaining_days, 3650))

    # Calculate failure date
    failure_date = datetime.utcnow() + timedelta(days=remaining_days)

    # Remaining life as percentage
    total_design_life_estimate = (100.0 - minimum_acceptable_health) / max(degradation_rate, 0.001)
    remaining_percent = (remaining_days / total_design_life_estimate) * 100 if total_design_life_estimate > 0 else 0

    return RemainingLifeResult(
        remaining_life_days=remaining_days,
        remaining_life_percent=round(min(100, remaining_percent), 1),
        degradation_rate_per_day=round(degradation_rate, 4),
        failure_date_estimate=failure_date,
        confidence_level=round(confidence, 2)
    )


def analyze_hotspot(
    location_x: float,
    location_y: float,
    temperature_celsius: float,
    surrounding_avg_temp: float,
    design_temp: float,
    area_m2: float = 1.0
) -> HotspotResult:
    """
    Analyze a detected hotspot for severity and recommended action.

    Hotspot Classification:
    - Minor: <20% above surrounding, <10% above design
    - Moderate: 20-50% above surrounding or 10-30% above design
    - Severe: 50-100% above surrounding or 30-50% above design
    - Critical: >100% above surrounding or >50% above design

    Args:
        location_x: X coordinate of hotspot
        location_y: Y coordinate of hotspot
        temperature_celsius: Hotspot temperature
        surrounding_avg_temp: Average surrounding temperature
        design_temp: Design surface temperature
        area_m2: Estimated hotspot area

    Returns:
        HotspotResult with severity and recommendations
    """
    # Calculate ratios
    ratio_to_surrounding = (temperature_celsius / surrounding_avg_temp) if surrounding_avg_temp > 0 else 1.0
    ratio_to_design = (temperature_celsius / design_temp) if design_temp > 0 else 1.0

    excess_over_surrounding = ratio_to_surrounding - 1.0
    excess_over_design = ratio_to_design - 1.0

    # Determine severity
    if excess_over_surrounding > 1.0 or excess_over_design > 0.5:
        severity = "CRITICAL"
        action = "Immediate inspection required; schedule emergency repair"
    elif excess_over_surrounding > 0.5 or excess_over_design > 0.3:
        severity = "SEVERE"
        action = "Schedule urgent inspection; prepare for repair during next outage"
    elif excess_over_surrounding > 0.2 or excess_over_design > 0.1:
        severity = "MODERATE"
        action = "Monitor closely; schedule inspection during next planned outage"
    else:
        severity = "MINOR"
        action = "Continue monitoring; no immediate action required"

    # Estimate heat loss from hotspot
    # Assuming convection coefficient of 10 W/m2-K
    heat_loss_kw = 10 * (temperature_celsius - surrounding_avg_temp) * area_m2 / 1000

    return HotspotResult(
        location_x=location_x,
        location_y=location_y,
        temperature_celsius=temperature_celsius,
        severity=severity,
        heat_loss_kw=round(heat_loss_kw, 2),
        recommended_action=action
    )


def determine_maintenance_priority(
    health_index: float,
    remaining_life_days: int,
    hotspot_count: int,
    critical_hotspots: int
) -> str:
    """
    Determine maintenance priority based on health assessment.

    Args:
        health_index: Current health index (0-100)
        remaining_life_days: Estimated remaining life
        hotspot_count: Total number of hotspots
        critical_hotspots: Number of critical hotspots

    Returns:
        Priority level: CRITICAL, HIGH, MEDIUM, LOW, SCHEDULED
    """
    if critical_hotspots > 0 or health_index < 20:
        return "CRITICAL"
    elif health_index < 40 or remaining_life_days < 30:
        return "HIGH"
    elif health_index < 60 or remaining_life_days < 90 or hotspot_count > 5:
        return "MEDIUM"
    elif health_index < 80 or remaining_life_days < 180:
        return "LOW"
    else:
        return "SCHEDULED"
