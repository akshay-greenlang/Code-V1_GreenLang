"""
Heat Loss Calculator

This module implements heat loss calculations for insulated surfaces using
deterministic physics-based formulas. Supports both flat and cylindrical
(pipe) surface geometries.

Heat transfer formulas follow engineering standards:
- ASTM C680 Standard Practice for Estimate of Heat Gain/Loss for Insulated Systems
- ISO 12241 Thermal insulation for building equipment and industrial installations
- ASHRAE Handbook Fundamentals

All calculations are deterministic - no ML/LLM in the calculation path.
This ensures zero-hallucination compliance for regulatory calculations.

Example:
    >>> from heat_loss import calculate_flat_surface_heat_loss
    >>> q = calculate_flat_surface_heat_loss(
    ...     t_hot=300, t_ambient=25, thickness=0.05,
    ...     k_insulation=0.04, area=10.0
    ... )
    >>> print(f"Heat loss: {q:.0f} W")
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/m2-K4
ABSOLUTE_ZERO_OFFSET = 273.15  # K


@dataclass
class HeatLossResult:
    """
    Results from heat loss calculation.

    Attributes:
        heat_loss_w: Total heat loss in Watts.
        heat_loss_btu_hr: Total heat loss in BTU/hr.
        surface_temp_c: Calculated external surface temperature.
        heat_flux_w_m2: Heat flux per unit area.
        r_insulation: Insulation thermal resistance.
        r_surface: Surface film resistance.
        r_total: Total thermal resistance.
        calculation_method: Method used (flat or cylindrical).
    """
    heat_loss_w: float
    heat_loss_btu_hr: float
    surface_temp_c: float
    heat_flux_w_m2: float
    r_insulation: float
    r_surface: float
    r_total: float
    calculation_method: str


# Default surface emissivity values
SURFACE_EMISSIVITY = {
    "aluminum_polished": 0.05,
    "aluminum_oxidized": 0.15,
    "aluminum_jacket": 0.10,
    "stainless_steel_polished": 0.15,
    "stainless_steel_weathered": 0.40,
    "galvanized_steel": 0.25,
    "painted_surface": 0.90,
    "black_surface": 0.95,
    "canvas_jacket": 0.85,
    "glass_cloth": 0.75,
    "weathered_jacket": 0.55,
    "default": 0.90,
}


def calculate_convection_coefficient(
    surface_temp_c: float,
    ambient_temp_c: float,
    wind_speed_m_s: float = 0.0,
    surface_orientation: str = "vertical"
) -> float:
    """
    Calculate combined convection heat transfer coefficient.

    Uses empirical correlations for natural and forced convection.

    Natural convection (Churchill-Chu correlation):
        h_nat = Nu * k_air / L

    Forced convection (flat plate):
        h_forced = 0.037 * Re^0.8 * Pr^0.33 * k_air / L

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient temperature in Celsius.
        wind_speed_m_s: Wind speed in m/s (0 for still air).
        surface_orientation: 'vertical', 'horizontal_up', or 'horizontal_down'.

    Returns:
        Combined convection coefficient in W/m2-K.

    Example:
        >>> h = calculate_convection_coefficient(50, 25, wind_speed_m_s=2.0)
        >>> print(f"Convection coefficient: {h:.1f} W/m2-K")
    """
    # Temperature difference
    delta_t = abs(surface_temp_c - ambient_temp_c)
    if delta_t < 1:
        delta_t = 1  # Avoid division issues

    # Film temperature for property evaluation
    t_film = (surface_temp_c + ambient_temp_c) / 2 + ABSOLUTE_ZERO_OFFSET  # Kelvin

    # Air properties at film temperature (simplified correlations)
    k_air = 0.0242 + 0.00007 * (t_film - 300)  # W/m-K
    k_air = max(0.020, min(0.040, k_air))

    # Natural convection coefficient (simplified McAdams correlation)
    # For vertical surfaces: h = 1.42 * (dT/L)^0.25 (laminar)
    # Simplified for typical industrial surfaces:
    if surface_orientation == "vertical":
        h_natural = 1.31 * (delta_t ** 0.33)
    elif surface_orientation == "horizontal_up":
        # Hot surface facing up
        h_natural = 1.52 * (delta_t ** 0.33)
    elif surface_orientation == "horizontal_down":
        # Hot surface facing down
        h_natural = 0.76 * (delta_t ** 0.33)
    else:
        h_natural = 1.31 * (delta_t ** 0.33)

    # Forced convection (if wind present)
    h_forced = 0.0
    if wind_speed_m_s > 0.1:
        # Simplified flat plate correlation
        # h = 5.7 + 3.8 * V (for V < 5 m/s)
        # h = 7.2 * V^0.78 (for V >= 5 m/s)
        if wind_speed_m_s < 5:
            h_forced = 5.7 + 3.8 * wind_speed_m_s
        else:
            h_forced = 7.2 * (wind_speed_m_s ** 0.78)

    # Combined coefficient (root-sum-square for mixed convection)
    h_combined = math.sqrt(h_natural ** 2 + h_forced ** 2)

    # Typical range check
    h_combined = max(2.0, min(100.0, h_combined))

    logger.debug(
        f"Convection: h_nat={h_natural:.2f}, h_forced={h_forced:.2f}, "
        f"h_combined={h_combined:.2f} W/m2-K"
    )

    return h_combined


def calculate_radiation_coefficient(
    surface_temp_c: float,
    ambient_temp_c: float,
    emissivity: float = 0.9
) -> float:
    """
    Calculate linearized radiation heat transfer coefficient.

    For small temperature differences, radiation can be linearized:
    h_rad = emissivity * sigma * (T_s^2 + T_a^2) * (T_s + T_a)

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient temperature in Celsius.
        emissivity: Surface emissivity (0-1).

    Returns:
        Radiation coefficient in W/m2-K.

    Example:
        >>> h_rad = calculate_radiation_coefficient(50, 25, emissivity=0.9)
        >>> print(f"Radiation coefficient: {h_rad:.2f} W/m2-K")
    """
    # Convert to Kelvin
    t_s = surface_temp_c + ABSOLUTE_ZERO_OFFSET
    t_a = ambient_temp_c + ABSOLUTE_ZERO_OFFSET

    # Linearized radiation coefficient
    h_rad = emissivity * STEFAN_BOLTZMANN * (t_s ** 2 + t_a ** 2) * (t_s + t_a)

    # Typical range: 4-8 W/m2-K for near-ambient, higher for hot surfaces
    logger.debug(f"Radiation coefficient: {h_rad:.2f} W/m2-K at Ts={surface_temp_c}C")

    return h_rad


def calculate_surface_coefficient(
    surface_temp_c: float,
    ambient_temp_c: float,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    surface_orientation: str = "vertical"
) -> float:
    """
    Calculate combined surface heat transfer coefficient.

    Combines convection and radiation:
    h_total = h_conv + h_rad

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient temperature in Celsius.
        wind_speed_m_s: Wind speed in m/s.
        emissivity: Surface emissivity (0-1).
        surface_orientation: Surface orientation.

    Returns:
        Combined surface coefficient in W/m2-K.
    """
    h_conv = calculate_convection_coefficient(
        surface_temp_c, ambient_temp_c, wind_speed_m_s, surface_orientation
    )
    h_rad = calculate_radiation_coefficient(
        surface_temp_c, ambient_temp_c, emissivity
    )

    h_total = h_conv + h_rad

    logger.debug(f"Surface coefficient: h_conv={h_conv:.2f} + h_rad={h_rad:.2f} = {h_total:.2f} W/m2-K")

    return h_total


def calculate_flat_surface_heat_loss(
    t_hot_c: float,
    t_ambient_c: float,
    insulation_thickness_m: float,
    k_insulation: float,
    area_m2: float,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    surface_orientation: str = "vertical"
) -> HeatLossResult:
    """
    Calculate heat loss through flat insulated surface.

    Uses 1D steady-state heat conduction with surface convection/radiation.

    Formula:
        Q = A * (T_hot - T_ambient) / R_total
        R_total = (thickness / k) + (1 / h_surface)

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        insulation_thickness_m: Insulation thickness in meters.
        k_insulation: Thermal conductivity of insulation in W/m-K.
        area_m2: Surface area in square meters.
        wind_speed_m_s: Wind speed in m/s.
        emissivity: Surface emissivity (0-1).
        surface_orientation: 'vertical', 'horizontal_up', or 'horizontal_down'.

    Returns:
        HeatLossResult with heat loss and thermal resistances.

    Example:
        >>> result = calculate_flat_surface_heat_loss(
        ...     t_hot_c=300, t_ambient_c=25,
        ...     insulation_thickness_m=0.05, k_insulation=0.04,
        ...     area_m2=10.0
        ... )
        >>> print(f"Heat loss: {result.heat_loss_w:.0f} W")
    """
    # Input validation
    if insulation_thickness_m <= 0:
        raise ValueError("Insulation thickness must be > 0")
    if k_insulation <= 0:
        raise ValueError("Thermal conductivity must be > 0")
    if area_m2 <= 0:
        raise ValueError("Area must be > 0")

    # Temperature difference
    delta_t = t_hot_c - t_ambient_c

    # If no temperature difference, no heat loss
    if abs(delta_t) < 0.01:
        return HeatLossResult(
            heat_loss_w=0.0,
            heat_loss_btu_hr=0.0,
            surface_temp_c=t_ambient_c,
            heat_flux_w_m2=0.0,
            r_insulation=insulation_thickness_m / k_insulation,
            r_surface=0.1,
            r_total=0.1,
            calculation_method="flat_surface"
        )

    # Iterative solution for surface temperature
    # Initial guess for surface temperature
    t_surface = t_ambient_c + delta_t * 0.1

    for iteration in range(20):
        # Calculate surface coefficient at current surface temp
        h_surface = calculate_surface_coefficient(
            t_surface, t_ambient_c, wind_speed_m_s, emissivity, surface_orientation
        )

        # Thermal resistances (per unit area)
        r_insulation = insulation_thickness_m / k_insulation
        r_surface = 1.0 / h_surface
        r_total = r_insulation + r_surface

        # Heat flux
        q_flux = delta_t / r_total  # W/m2

        # New surface temperature estimate
        t_surface_new = t_hot_c - q_flux * r_insulation

        # Check convergence
        if abs(t_surface_new - t_surface) < 0.01:
            t_surface = t_surface_new
            break

        t_surface = t_surface_new

    # Final calculations
    q_total = q_flux * area_m2  # Total heat loss in W
    q_btu_hr = q_total * 3.412  # Convert to BTU/hr

    logger.debug(
        f"Flat surface heat loss: Q={q_total:.1f}W, Ts={t_surface:.1f}C, "
        f"R_ins={r_insulation:.4f}, R_surf={r_surface:.4f}"
    )

    return HeatLossResult(
        heat_loss_w=q_total,
        heat_loss_btu_hr=q_btu_hr,
        surface_temp_c=t_surface,
        heat_flux_w_m2=q_flux,
        r_insulation=r_insulation,
        r_surface=r_surface,
        r_total=r_total,
        calculation_method="flat_surface"
    )


def calculate_cylindrical_heat_loss(
    t_hot_c: float,
    t_ambient_c: float,
    pipe_outer_radius_m: float,
    insulation_thickness_m: float,
    k_insulation: float,
    pipe_length_m: float,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9
) -> HeatLossResult:
    """
    Calculate heat loss through cylindrical (pipe) insulated surface.

    Uses radial heat conduction for cylindrical geometry.

    Formula:
        Q = 2 * pi * L * (T_hot - T_ambient) / R_total
        R_total = ln(r2/r1)/(2*pi*k) + 1/(h * 2*pi*r2)

    Where:
        r1 = pipe outer radius
        r2 = r1 + insulation thickness

    Args:
        t_hot_c: Hot surface (pipe) temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        pipe_outer_radius_m: Pipe outer radius in meters.
        insulation_thickness_m: Insulation thickness in meters.
        k_insulation: Thermal conductivity in W/m-K.
        pipe_length_m: Pipe length in meters.
        wind_speed_m_s: Wind speed in m/s.
        emissivity: Surface emissivity (0-1).

    Returns:
        HeatLossResult with heat loss and thermal resistances.

    Example:
        >>> result = calculate_cylindrical_heat_loss(
        ...     t_hot_c=180, t_ambient_c=25,
        ...     pipe_outer_radius_m=0.05, insulation_thickness_m=0.05,
        ...     k_insulation=0.04, pipe_length_m=100
        ... )
        >>> print(f"Heat loss: {result.heat_loss_w:.0f} W")
    """
    # Input validation
    if pipe_outer_radius_m <= 0:
        raise ValueError("Pipe radius must be > 0")
    if insulation_thickness_m <= 0:
        raise ValueError("Insulation thickness must be > 0")
    if k_insulation <= 0:
        raise ValueError("Thermal conductivity must be > 0")
    if pipe_length_m <= 0:
        raise ValueError("Pipe length must be > 0")

    # Radii
    r1 = pipe_outer_radius_m  # Inner radius of insulation
    r2 = r1 + insulation_thickness_m  # Outer radius of insulation

    # Temperature difference
    delta_t = t_hot_c - t_ambient_c

    # If no temperature difference, no heat loss
    if abs(delta_t) < 0.01:
        return HeatLossResult(
            heat_loss_w=0.0,
            heat_loss_btu_hr=0.0,
            surface_temp_c=t_ambient_c,
            heat_flux_w_m2=0.0,
            r_insulation=math.log(r2 / r1) / (2 * math.pi * k_insulation * pipe_length_m),
            r_surface=0.1,
            r_total=0.1,
            calculation_method="cylindrical"
        )

    # Iterative solution for surface temperature
    t_surface = t_ambient_c + delta_t * 0.1

    for iteration in range(20):
        # Surface coefficient at outer surface
        h_surface = calculate_surface_coefficient(
            t_surface, t_ambient_c, wind_speed_m_s, emissivity, "vertical"
        )

        # Thermal resistances (for length L)
        # R_insulation = ln(r2/r1) / (2 * pi * k * L)
        r_insulation = math.log(r2 / r1) / (2 * math.pi * k_insulation * pipe_length_m)

        # R_surface = 1 / (h * A_outer) = 1 / (h * 2 * pi * r2 * L)
        a_outer = 2 * math.pi * r2 * pipe_length_m
        r_surface = 1.0 / (h_surface * a_outer)

        r_total = r_insulation + r_surface

        # Total heat loss
        q_total = delta_t / r_total  # W

        # New surface temperature
        t_surface_new = t_hot_c - q_total * r_insulation

        # Check convergence
        if abs(t_surface_new - t_surface) < 0.01:
            t_surface = t_surface_new
            break

        t_surface = t_surface_new

    # Heat flux based on outer surface area
    q_flux = q_total / a_outer  # W/m2

    q_btu_hr = q_total * 3.412  # Convert to BTU/hr

    logger.debug(
        f"Cylindrical heat loss: Q={q_total:.1f}W, Ts={t_surface:.1f}C, "
        f"r1={r1:.4f}m, r2={r2:.4f}m"
    )

    return HeatLossResult(
        heat_loss_w=q_total,
        heat_loss_btu_hr=q_btu_hr,
        surface_temp_c=t_surface,
        heat_flux_w_m2=q_flux,
        r_insulation=r_insulation,
        r_surface=r_surface,
        r_total=r_total,
        calculation_method="cylindrical"
    )


def calculate_bare_surface_heat_loss(
    t_hot_c: float,
    t_ambient_c: float,
    area_m2: float,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    surface_orientation: str = "vertical"
) -> HeatLossResult:
    """
    Calculate heat loss from uninsulated (bare) surface.

    Used as baseline to calculate insulation savings.

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        area_m2: Surface area in square meters.
        wind_speed_m_s: Wind speed in m/s.
        emissivity: Surface emissivity (0-1).
        surface_orientation: Surface orientation.

    Returns:
        HeatLossResult for bare surface.
    """
    delta_t = t_hot_c - t_ambient_c

    if abs(delta_t) < 0.01:
        return HeatLossResult(
            heat_loss_w=0.0,
            heat_loss_btu_hr=0.0,
            surface_temp_c=t_hot_c,
            heat_flux_w_m2=0.0,
            r_insulation=0.0,
            r_surface=0.1,
            r_total=0.1,
            calculation_method="bare_surface"
        )

    # Surface coefficient
    h_surface = calculate_surface_coefficient(
        t_hot_c, t_ambient_c, wind_speed_m_s, emissivity, surface_orientation
    )

    # Heat loss directly from surface
    r_surface = 1.0 / h_surface  # m2-K/W
    q_flux = delta_t * h_surface  # W/m2
    q_total = q_flux * area_m2  # W
    q_btu_hr = q_total * 3.412

    return HeatLossResult(
        heat_loss_w=q_total,
        heat_loss_btu_hr=q_btu_hr,
        surface_temp_c=t_hot_c,
        heat_flux_w_m2=q_flux,
        r_insulation=0.0,
        r_surface=r_surface,
        r_total=r_surface,
        calculation_method="bare_surface"
    )


def calculate_heat_loss_savings(
    bare_heat_loss: HeatLossResult,
    insulated_heat_loss: HeatLossResult
) -> Dict[str, float]:
    """
    Calculate heat loss savings from insulation.

    Args:
        bare_heat_loss: Heat loss result for uninsulated surface.
        insulated_heat_loss: Heat loss result for insulated surface.

    Returns:
        Dictionary with savings metrics.
    """
    q_bare = bare_heat_loss.heat_loss_w
    q_insulated = insulated_heat_loss.heat_loss_w

    savings_w = q_bare - q_insulated
    savings_btu_hr = savings_w * 3.412
    savings_percent = (savings_w / q_bare * 100) if q_bare > 0 else 0

    return {
        "savings_w": savings_w,
        "savings_btu_hr": savings_btu_hr,
        "savings_percent": savings_percent,
        "bare_heat_loss_w": q_bare,
        "insulated_heat_loss_w": q_insulated,
    }


def calculate_multilayer_heat_loss(
    t_hot_c: float,
    t_ambient_c: float,
    layers: List[Tuple[float, float]],
    area_m2: float,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    geometry: str = "flat"
) -> HeatLossResult:
    """
    Calculate heat loss through multi-layer insulation.

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        layers: List of (thickness_m, k_value) tuples for each layer.
        area_m2: Surface area in square meters.
        wind_speed_m_s: Wind speed in m/s.
        emissivity: Surface emissivity.
        geometry: 'flat' or 'cylindrical'.

    Returns:
        HeatLossResult for multi-layer system.
    """
    if not layers:
        raise ValueError("At least one insulation layer is required")

    # Total insulation resistance
    r_insulation_total = 0.0
    total_thickness = 0.0

    for thickness, k_value in layers:
        if thickness <= 0 or k_value <= 0:
            raise ValueError("Layer thickness and k-value must be > 0")
        r_insulation_total += thickness / k_value
        total_thickness += thickness

    # Average k-value for reporting
    k_avg = total_thickness / r_insulation_total if r_insulation_total > 0 else 0.04

    # Use single-layer calculation with equivalent resistance
    if geometry == "flat":
        return calculate_flat_surface_heat_loss(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            insulation_thickness_m=total_thickness,
            k_insulation=k_avg,
            area_m2=area_m2,
            wind_speed_m_s=wind_speed_m_s,
            emissivity=emissivity
        )
    else:
        # For cylindrical, would need inner radius - use flat as approximation
        return calculate_flat_surface_heat_loss(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            insulation_thickness_m=total_thickness,
            k_insulation=k_avg,
            area_m2=area_m2,
            wind_speed_m_s=wind_speed_m_s,
            emissivity=emissivity
        )


def convert_watts_to_btu_hr(watts: float) -> float:
    """Convert Watts to BTU/hr."""
    return watts * 3.412


def convert_btu_hr_to_watts(btu_hr: float) -> float:
    """Convert BTU/hr to Watts."""
    return btu_hr / 3.412


def calculate_annual_energy_loss(
    heat_loss_w: float,
    operating_hours_per_year: float = 8760
) -> Dict[str, float]:
    """
    Calculate annual energy loss from heat loss.

    Args:
        heat_loss_w: Heat loss in Watts.
        operating_hours_per_year: Annual operating hours (default 8760 = 24/7).

    Returns:
        Dictionary with annual energy loss in various units.
    """
    # Energy in different units
    kwh_per_year = (heat_loss_w / 1000) * operating_hours_per_year
    mwh_per_year = kwh_per_year / 1000
    gj_per_year = kwh_per_year * 0.0036
    mmbtu_per_year = kwh_per_year * 0.003412

    return {
        "kwh_per_year": kwh_per_year,
        "mwh_per_year": mwh_per_year,
        "gj_per_year": gj_per_year,
        "mmbtu_per_year": mmbtu_per_year,
        "operating_hours": operating_hours_per_year,
    }


def estimate_heat_loss_from_ir_data(
    surface_temp_c: float,
    ambient_temp_c: float,
    emissivity: float,
    area_m2: float,
    wind_speed_m_s: float = 0.0
) -> HeatLossResult:
    """
    Estimate heat loss from IR camera temperature measurements.

    This is the key function for thermal imaging integration.
    Uses measured surface temperature to back-calculate heat loss.

    Args:
        surface_temp_c: Measured surface temperature from IR camera.
        ambient_temp_c: Ambient temperature.
        emissivity: Surface emissivity (critical for accurate IR measurement).
        area_m2: Area of the measured surface.
        wind_speed_m_s: Wind speed at measurement time.

    Returns:
        HeatLossResult based on IR measurements.
    """
    # Calculate surface coefficient at measured temperature
    h_surface = calculate_surface_coefficient(
        surface_temp_c, ambient_temp_c, wind_speed_m_s, emissivity, "vertical"
    )

    # Heat flux = h * (T_surface - T_ambient)
    delta_t = surface_temp_c - ambient_temp_c
    q_flux = h_surface * delta_t

    # Total heat loss
    q_total = q_flux * area_m2
    q_btu_hr = q_total * 3.412

    # Surface resistance
    r_surface = 1.0 / h_surface

    logger.info(
        f"IR-based heat loss estimate: Ts={surface_temp_c}C, "
        f"Q={q_total:.0f}W, q={q_flux:.1f}W/m2"
    )

    return HeatLossResult(
        heat_loss_w=q_total,
        heat_loss_btu_hr=q_btu_hr,
        surface_temp_c=surface_temp_c,
        heat_flux_w_m2=q_flux,
        r_insulation=0.0,  # Unknown from IR measurement alone
        r_surface=r_surface,
        r_total=r_surface,
        calculation_method="ir_measurement"
    )
