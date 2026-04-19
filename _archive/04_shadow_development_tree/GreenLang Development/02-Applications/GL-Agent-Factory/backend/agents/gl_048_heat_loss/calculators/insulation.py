"""
Insulation Analysis Calculator

Physics-based calculations for thermal insulation performance,
economic thickness optimization, and effectiveness assessment.

References:
    - ISO 12241: Thermal Insulation for Building Equipment
    - ASTM C680: Standard Practice for Estimate of Heat Gain or Loss
    - 3E Plus (DOE Insulation Thickness Tool)
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Typical insulation thermal conductivities (W/m-K at mean temp)
INSULATION_CONDUCTIVITY = {
    "mineral_wool": {50: 0.035, 100: 0.040, 150: 0.048, 200: 0.058, 250: 0.070},
    "calcium_silicate": {50: 0.055, 100: 0.060, 150: 0.068, 200: 0.078, 250: 0.090},
    "cellular_glass": {50: 0.045, 100: 0.050, 150: 0.058, 200: 0.068, 250: 0.080},
    "perlite": {50: 0.050, 100: 0.055, 150: 0.062, 200: 0.072, 250: 0.085},
    "fiberglass": {50: 0.033, 100: 0.038, 150: 0.045, 200: 0.055, 250: 0.068},
    "aerogel": {50: 0.015, 100: 0.018, 150: 0.022, 200: 0.028, 250: 0.035},
}


def calculate_insulation_heat_loss(
    process_temp_c: float,
    ambient_temp_c: float,
    insulation_thickness_mm: float,
    insulation_conductivity: float,
    surface_area_m2: float,
    h_outside: float = 10.0
) -> Tuple[float, float, Dict]:
    """
    Calculate heat loss through insulation layer.

    For flat surfaces:
        Q = (T_p - T_a) / (L/k + 1/h_o) * A

    where:
        L = insulation thickness (m)
        k = thermal conductivity (W/m-K)
        h_o = outside surface coefficient (W/m^2-K)

    Args:
        process_temp_c: Process/hot side temperature.
        ambient_temp_c: Ambient temperature.
        insulation_thickness_mm: Insulation thickness in mm.
        insulation_conductivity: Conductivity in W/m-K.
        surface_area_m2: Surface area in m^2.
        h_outside: Outside convection+radiation coefficient.

    Returns:
        Tuple of (heat_loss_watts, surface_temp_c, details).

    Example:
        >>> q, t_surf, details = calculate_insulation_heat_loss(
        ...     200, 20, 50, 0.04, 10, 10
        ... )
    """
    if insulation_thickness_mm <= 0:
        # No insulation - just surface resistance
        q = h_outside * surface_area_m2 * (process_temp_c - ambient_temp_c)
        return q, process_temp_c, {"error": "No insulation"}

    # Convert thickness to meters
    thickness_m = insulation_thickness_mm / 1000

    # Total thermal resistance (per unit area)
    r_insulation = thickness_m / insulation_conductivity
    r_surface = 1 / h_outside
    r_total = r_insulation + r_surface

    # Temperature difference
    delta_t = process_temp_c - ambient_temp_c

    # Heat loss
    q = (delta_t / r_total) * surface_area_m2

    # Surface temperature
    # T_surface = T_ambient + Q * R_surface / A
    t_surface = ambient_temp_c + (q / surface_area_m2) * r_surface

    # Heat flux
    heat_flux = q / surface_area_m2

    details = {
        "r_insulation_m2k_w": round(r_insulation, 4),
        "r_surface_m2k_w": round(r_surface, 4),
        "r_total_m2k_w": round(r_total, 4),
        "heat_flux_w_m2": round(heat_flux, 1),
        "u_value_w_m2k": round(1 / r_total, 3),
    }

    return q, t_surface, details


def calculate_economic_thickness(
    process_temp_c: float,
    ambient_temp_c: float,
    insulation_conductivity: float,
    h_outside: float = 10.0,
    energy_cost_per_kwh: float = 0.08,
    insulation_cost_per_m3: float = 200.0,
    operating_hours_year: int = 8000,
    equipment_life_years: int = 10,
    interest_rate: float = 0.08,
    geometry: str = "flat",
    pipe_diameter_mm: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Calculate economic insulation thickness.

    The economic thickness minimizes total cost (energy + insulation).
    Uses iterative approach to find minimum total annual cost.

    Args:
        process_temp_c: Process temperature.
        ambient_temp_c: Ambient temperature.
        insulation_conductivity: Insulation k value (W/m-K).
        h_outside: Outside surface coefficient.
        energy_cost_per_kwh: Energy cost in $/kWh.
        insulation_cost_per_m3: Insulation material cost in $/m^3.
        operating_hours_year: Annual operating hours.
        equipment_life_years: Expected equipment life.
        interest_rate: Annual interest rate for cost analysis.
        geometry: "flat" or "pipe".
        pipe_diameter_mm: Pipe OD for cylindrical geometry.

    Returns:
        Tuple of (economic_thickness_mm, analysis_dict).

    Example:
        >>> thick, analysis = calculate_economic_thickness(
        ...     300, 20, 0.04, 10, 0.10, 200, 8000, 10, 0.08
        ... )
    """
    delta_t = process_temp_c - ambient_temp_c
    if delta_t <= 0:
        return 0.0, {"note": "No temperature differential"}

    # Capital recovery factor
    crf = (interest_rate * (1 + interest_rate)**equipment_life_years) / (
        (1 + interest_rate)**equipment_life_years - 1
    )

    # Test thicknesses (mm)
    thicknesses = list(range(10, 301, 10))
    min_cost = float('inf')
    economic_thickness = 50  # Default

    results = []

    for thick_mm in thicknesses:
        thick_m = thick_mm / 1000

        # Heat loss per m^2 (flat geometry)
        r_ins = thick_m / insulation_conductivity
        r_surf = 1 / h_outside
        r_total = r_ins + r_surf

        heat_flux = delta_t / r_total  # W/m^2
        heat_loss_kwh_m2_yr = heat_flux * operating_hours_year / 1000

        # Annual energy cost per m^2
        energy_cost_annual = heat_loss_kwh_m2_yr * energy_cost_per_kwh

        # Insulation cost per m^2 (annualized)
        insulation_volume_m3_per_m2 = thick_m
        insulation_capital = insulation_volume_m3_per_m2 * insulation_cost_per_m3
        insulation_cost_annual = insulation_capital * crf

        # Total annual cost
        total_cost = energy_cost_annual + insulation_cost_annual

        results.append({
            "thickness_mm": thick_mm,
            "energy_cost": round(energy_cost_annual, 2),
            "insulation_cost": round(insulation_cost_annual, 2),
            "total_cost": round(total_cost, 2),
        })

        if total_cost < min_cost:
            min_cost = total_cost
            economic_thickness = thick_mm

    # Find optimum in results
    optimum_result = next(
        (r for r in results if r["thickness_mm"] == economic_thickness),
        results[0]
    )

    analysis = {
        "economic_thickness_mm": economic_thickness,
        "annual_energy_cost_per_m2": optimum_result["energy_cost"],
        "annual_insulation_cost_per_m2": optimum_result["insulation_cost"],
        "total_annual_cost_per_m2": optimum_result["total_cost"],
        "capital_recovery_factor": round(crf, 4),
        "all_results": results[:10],  # First 10 for reference
    }

    logger.debug(f"Economic thickness: {economic_thickness}mm")

    return float(economic_thickness), analysis


def calculate_insulation_effectiveness(
    insulation_thickness_mm: float,
    insulation_conductivity: float,
    process_temp_c: float,
    ambient_temp_c: float,
    h_outside: float = 10.0
) -> Tuple[float, Dict]:
    """
    Calculate insulation effectiveness (% heat loss reduction).

    Effectiveness = (Q_bare - Q_insulated) / Q_bare * 100

    Args:
        insulation_thickness_mm: Current insulation thickness.
        insulation_conductivity: Insulation k value.
        process_temp_c: Process temperature.
        ambient_temp_c: Ambient temperature.
        h_outside: Outside surface coefficient.

    Returns:
        Tuple of (effectiveness_pct, details).
    """
    delta_t = process_temp_c - ambient_temp_c
    if delta_t <= 0:
        return 100.0, {"note": "No temperature differential"}

    # Bare surface heat loss (per m^2)
    # Assuming h_bare ~ 15 W/m^2-K for combined convection+radiation
    h_bare = 15.0
    q_bare = h_bare * delta_t

    # Insulated heat loss
    if insulation_thickness_mm > 0:
        thick_m = insulation_thickness_mm / 1000
        r_ins = thick_m / insulation_conductivity
        r_surf = 1 / h_outside
        r_total = r_ins + r_surf
        q_insulated = delta_t / r_total
    else:
        q_insulated = q_bare

    # Effectiveness
    if q_bare > 0:
        effectiveness = (q_bare - q_insulated) / q_bare * 100
    else:
        effectiveness = 0

    # Surface temperature reduction
    t_bare = process_temp_c  # Approximately
    if insulation_thickness_mm > 0:
        t_insulated = ambient_temp_c + q_insulated / h_outside
    else:
        t_insulated = t_bare

    details = {
        "bare_heat_flux_w_m2": round(q_bare, 1),
        "insulated_heat_flux_w_m2": round(q_insulated, 1),
        "heat_flux_reduction_pct": round(effectiveness, 1),
        "bare_surface_temp_c": round(t_bare, 1),
        "insulated_surface_temp_c": round(t_insulated, 1),
        "surface_temp_reduction_c": round(t_bare - t_insulated, 1),
    }

    return round(effectiveness, 1), details


def calculate_surface_temperature_insulated(
    process_temp_c: float,
    ambient_temp_c: float,
    insulation_thickness_mm: float,
    insulation_conductivity: float,
    h_outside: float = 10.0
) -> float:
    """
    Calculate outer surface temperature of insulated surface.

    T_surface = T_ambient + Q * R_surface / A
            = T_ambient + (T_process - T_ambient) * R_surface / R_total

    Args:
        process_temp_c: Process/hot side temperature.
        ambient_temp_c: Ambient temperature.
        insulation_thickness_mm: Insulation thickness in mm.
        insulation_conductivity: Conductivity in W/m-K.
        h_outside: Outside surface coefficient.

    Returns:
        Surface temperature in Celsius.
    """
    if insulation_thickness_mm <= 0:
        return process_temp_c

    thick_m = insulation_thickness_mm / 1000

    r_ins = thick_m / insulation_conductivity
    r_surf = 1 / h_outside
    r_total = r_ins + r_surf

    delta_t = process_temp_c - ambient_temp_c
    t_surface = ambient_temp_c + delta_t * (r_surf / r_total)

    return t_surface


def get_insulation_conductivity(
    material_type: str,
    mean_temperature_c: float
) -> float:
    """
    Get insulation thermal conductivity at given temperature.

    Args:
        material_type: Type of insulation material.
        mean_temperature_c: Mean temperature through insulation.

    Returns:
        Thermal conductivity in W/m-K.
    """
    material = material_type.lower().replace(" ", "_")

    if material not in INSULATION_CONDUCTIVITY:
        logger.warning(f"Unknown material '{material}', using mineral_wool")
        material = "mineral_wool"

    data = INSULATION_CONDUCTIVITY[material]
    temps = sorted(data.keys())

    # Interpolate
    if mean_temperature_c <= temps[0]:
        return data[temps[0]]
    if mean_temperature_c >= temps[-1]:
        return data[temps[-1]]

    for i in range(len(temps) - 1):
        if temps[i] <= mean_temperature_c <= temps[i + 1]:
            t1, t2 = temps[i], temps[i + 1]
            k1, k2 = data[t1], data[t2]
            frac = (mean_temperature_c - t1) / (t2 - t1)
            return k1 + frac * (k2 - k1)

    return data[temps[0]]
