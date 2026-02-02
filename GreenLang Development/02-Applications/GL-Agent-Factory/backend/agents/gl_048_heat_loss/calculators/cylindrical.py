"""
Cylindrical Heat Transfer Calculator

Physics-based calculations for heat loss from pipes, tanks, and other
cylindrical equipment with optional insulation.

References:
    - ISO 12241: Thermal Insulation for Building Equipment
    - ASTM C680: Standard Practice for Estimate of Heat Gain or Loss
    - Incropera: Fundamentals of Heat and Mass Transfer
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_pipe_heat_loss(
    pipe_od_mm: float,
    process_temp_c: float,
    ambient_temp_c: float,
    pipe_length_m: float = 1.0,
    insulation_thickness_mm: float = 0.0,
    insulation_conductivity: float = 0.04,
    pipe_conductivity: float = 50.0,
    pipe_wall_mm: float = 5.0,
    h_inside: float = 500.0,
    h_outside: float = 10.0
) -> Tuple[float, float, Dict]:
    """
    Calculate heat loss from insulated pipe using radial conduction.

    For cylindrical geometry (per unit length):
        Q/L = (T_i - T_o) / [1/(2*pi*r_i*h_i) + ln(r_o/r_i)/(2*pi*k_pipe)
              + ln(r_ins/r_o)/(2*pi*k_ins) + 1/(2*pi*r_ins*h_o)]

    Args:
        pipe_od_mm: Pipe outer diameter in mm.
        process_temp_c: Process fluid temperature.
        ambient_temp_c: Ambient temperature.
        pipe_length_m: Pipe length in meters.
        insulation_thickness_mm: Insulation thickness in mm.
        insulation_conductivity: Insulation k value (W/m-K).
        pipe_conductivity: Pipe material k value (W/m-K).
        pipe_wall_mm: Pipe wall thickness in mm.
        h_inside: Inside convection coefficient.
        h_outside: Outside convection+radiation coefficient.

    Returns:
        Tuple of (heat_loss_watts, surface_temp_c, details).

    Example:
        >>> q, t_surf, details = calculate_pipe_heat_loss(
        ...     100, 200, 20, 10, 50, 0.04, 50, 5, 500, 10
        ... )
    """
    # Convert to meters
    r_i = (pipe_od_mm / 2 - pipe_wall_mm) / 1000  # Inside radius
    r_o = pipe_od_mm / 2 / 1000  # Outside radius (pipe)
    r_ins = (pipe_od_mm / 2 + insulation_thickness_mm) / 1000  # Outer radius

    if r_i <= 0:
        raise ValueError("Invalid pipe dimensions")

    # Calculate thermal resistances (per meter length)
    # Inside convection
    r_conv_in = 1 / (2 * math.pi * r_i * h_inside)

    # Pipe wall conduction
    r_cond_pipe = math.log(r_o / r_i) / (2 * math.pi * pipe_conductivity)

    # Insulation conduction (if present)
    if insulation_thickness_mm > 0:
        r_cond_ins = math.log(r_ins / r_o) / (2 * math.pi * insulation_conductivity)
        r_conv_out = 1 / (2 * math.pi * r_ins * h_outside)
    else:
        r_cond_ins = 0
        r_conv_out = 1 / (2 * math.pi * r_o * h_outside)
        r_ins = r_o

    # Total resistance
    r_total = r_conv_in + r_cond_pipe + r_cond_ins + r_conv_out

    # Temperature difference
    delta_t = process_temp_c - ambient_temp_c

    # Heat loss per meter
    q_per_m = delta_t / r_total

    # Total heat loss
    q_total = q_per_m * pipe_length_m

    # Surface temperature
    # T_surface = T_ambient + Q * R_conv_out
    t_surface = ambient_temp_c + q_per_m * r_conv_out

    # Calculate outer surface area
    outer_area_m2 = 2 * math.pi * r_ins * pipe_length_m

    # Heat flux at outer surface
    heat_flux = q_total / outer_area_m2 if outer_area_m2 > 0 else 0

    details = {
        "r_conv_inside_m_k_w": round(r_conv_in, 6),
        "r_cond_pipe_m_k_w": round(r_cond_pipe, 6),
        "r_cond_insulation_m_k_w": round(r_cond_ins, 6),
        "r_conv_outside_m_k_w": round(r_conv_out, 6),
        "r_total_m_k_w": round(r_total, 6),
        "heat_loss_per_meter_w": round(q_per_m, 1),
        "outer_surface_area_m2": round(outer_area_m2, 3),
        "heat_flux_w_m2": round(heat_flux, 1),
    }

    logger.debug(
        f"Pipe heat loss: OD={pipe_od_mm}mm, ins={insulation_thickness_mm}mm, "
        f"Q={q_total:.0f}W, T_surf={t_surface:.1f}C"
    )

    return q_total, t_surface, details


def calculate_critical_radius(
    insulation_conductivity: float,
    h_outside: float
) -> float:
    """
    Calculate critical insulation radius.

    For cylindrical geometry, there exists a critical radius where
    adding insulation actually INCREASES heat loss (for small pipes).

    r_critical = k / h

    Insulation should only be added if r_pipe > r_critical.

    Args:
        insulation_conductivity: Insulation k value (W/m-K).
        h_outside: Outside surface coefficient (W/m^2-K).

    Returns:
        Critical radius in meters.

    Example:
        >>> r_crit = calculate_critical_radius(0.04, 10)
        >>> print(f"Critical radius: {r_crit*1000:.1f} mm")
    """
    r_critical = insulation_conductivity / h_outside
    return r_critical


def calculate_multilayer_pipe_heat_loss(
    pipe_od_mm: float,
    process_temp_c: float,
    ambient_temp_c: float,
    pipe_length_m: float,
    layers: List[Dict],
    pipe_conductivity: float = 50.0,
    pipe_wall_mm: float = 5.0,
    h_inside: float = 500.0,
    h_outside: float = 10.0
) -> Tuple[float, List[float], Dict]:
    """
    Calculate heat loss from pipe with multiple insulation layers.

    Each layer is defined by:
        {"thickness_mm": float, "conductivity": float, "name": str}

    Args:
        pipe_od_mm: Pipe outer diameter.
        process_temp_c: Process temperature.
        ambient_temp_c: Ambient temperature.
        pipe_length_m: Pipe length.
        layers: List of layer dictionaries.
        pipe_conductivity: Pipe material conductivity.
        pipe_wall_mm: Pipe wall thickness.
        h_inside: Inside coefficient.
        h_outside: Outside coefficient.

    Returns:
        Tuple of (heat_loss_watts, interface_temps, details).
    """
    # Convert to meters
    r_i = (pipe_od_mm / 2 - pipe_wall_mm) / 1000
    r_current = pipe_od_mm / 2 / 1000  # Start at pipe OD

    # Inside convection
    r_conv_in = 1 / (2 * math.pi * r_i * h_inside)

    # Pipe wall
    r_cond_pipe = math.log(r_current / r_i) / (2 * math.pi * pipe_conductivity)

    # Build up layer resistances
    layer_resistances = []
    layer_radii = [r_current]

    for layer in layers:
        thick_m = layer["thickness_mm"] / 1000
        k = layer["conductivity"]
        r_outer = r_current + thick_m

        r_layer = math.log(r_outer / r_current) / (2 * math.pi * k)
        layer_resistances.append(r_layer)
        layer_radii.append(r_outer)
        r_current = r_outer

    # Outside convection
    r_conv_out = 1 / (2 * math.pi * r_current * h_outside)

    # Total resistance
    r_total = r_conv_in + r_cond_pipe + sum(layer_resistances) + r_conv_out

    # Heat loss
    delta_t = process_temp_c - ambient_temp_c
    q_per_m = delta_t / r_total
    q_total = q_per_m * pipe_length_m

    # Interface temperatures
    interface_temps = [process_temp_c]
    temp_drop = 0

    # Inner pipe surface
    temp_drop += q_per_m * r_conv_in
    interface_temps.append(process_temp_c - temp_drop)

    # Outer pipe surface
    temp_drop += q_per_m * r_cond_pipe
    interface_temps.append(process_temp_c - temp_drop)

    # Each insulation layer interface
    for r_layer in layer_resistances:
        temp_drop += q_per_m * r_layer
        interface_temps.append(process_temp_c - temp_drop)

    t_surface = interface_temps[-1]

    details = {
        "r_total_m_k_w": round(r_total, 6),
        "heat_loss_per_meter_w": round(q_per_m, 1),
        "total_heat_loss_w": round(q_total, 1),
        "surface_temperature_c": round(t_surface, 1),
        "layer_radii_mm": [round(r * 1000, 1) for r in layer_radii],
    }

    return q_total, interface_temps, details


def calculate_tank_heat_loss(
    diameter_m: float,
    height_m: float,
    process_temp_c: float,
    ambient_temp_c: float,
    insulation_thickness_mm: float = 0.0,
    insulation_conductivity: float = 0.04,
    h_outside: float = 10.0,
    roof_insulated: bool = True,
    floor_insulated: bool = False,
    floor_u_value: float = 0.5
) -> Tuple[float, Dict]:
    """
    Calculate total heat loss from vertical cylindrical tank.

    Tank heat loss includes:
    - Cylindrical shell (sidewall)
    - Roof (flat circular)
    - Floor (may have ground contact)

    Args:
        diameter_m: Tank diameter in meters.
        height_m: Tank height in meters.
        process_temp_c: Liquid/process temperature.
        ambient_temp_c: Ambient air temperature.
        insulation_thickness_mm: Insulation thickness on shell/roof.
        insulation_conductivity: Insulation k value.
        h_outside: Outside surface coefficient.
        roof_insulated: Whether roof has insulation.
        floor_insulated: Whether floor is insulated.
        floor_u_value: Floor U-value (W/m^2-K) if not insulated.

    Returns:
        Tuple of (total_heat_loss_watts, breakdown_dict).

    Example:
        >>> q, breakdown = calculate_tank_heat_loss(
        ...     5.0, 10.0, 90, 20, 75, 0.04
        ... )
    """
    delta_t = process_temp_c - ambient_temp_c

    # Calculate areas
    radius_m = diameter_m / 2
    area_roof = math.pi * radius_m ** 2
    area_floor = area_roof
    area_shell = 2 * math.pi * radius_m * height_m
    area_total = area_roof + area_floor + area_shell

    # Shell heat loss (cylindrical)
    r_i = radius_m
    r_o = radius_m + insulation_thickness_mm / 1000

    if insulation_thickness_mm > 0:
        # Resistance per unit area (approximation for large radius)
        r_ins = insulation_thickness_mm / 1000 / insulation_conductivity
        r_surf = 1 / h_outside
        r_shell = r_ins + r_surf
        q_shell = (delta_t / r_shell) * area_shell
        t_surf_shell = ambient_temp_c + (delta_t * r_surf / r_shell)
    else:
        q_shell = h_outside * area_shell * delta_t
        t_surf_shell = process_temp_c

    # Roof heat loss (flat surface)
    if roof_insulated and insulation_thickness_mm > 0:
        r_ins = insulation_thickness_mm / 1000 / insulation_conductivity
        r_surf = 1 / h_outside
        r_roof = r_ins + r_surf
        q_roof = (delta_t / r_roof) * area_roof
        t_surf_roof = ambient_temp_c + (delta_t * r_surf / r_roof)
    else:
        q_roof = h_outside * area_roof * delta_t
        t_surf_roof = process_temp_c

    # Floor heat loss
    if floor_insulated and insulation_thickness_mm > 0:
        r_ins = insulation_thickness_mm / 1000 / insulation_conductivity
        # Ground resistance approximation
        r_ground = 1 / floor_u_value
        r_floor = r_ins + r_ground
        q_floor = (delta_t / r_floor) * area_floor
    else:
        # Direct ground contact
        q_floor = floor_u_value * area_floor * delta_t

    # Total
    q_total = q_shell + q_roof + q_floor

    breakdown = {
        "shell_area_m2": round(area_shell, 2),
        "roof_area_m2": round(area_roof, 2),
        "floor_area_m2": round(area_floor, 2),
        "total_area_m2": round(area_total, 2),
        "shell_heat_loss_w": round(q_shell, 0),
        "roof_heat_loss_w": round(q_roof, 0),
        "floor_heat_loss_w": round(q_floor, 0),
        "total_heat_loss_w": round(q_total, 0),
        "shell_surface_temp_c": round(t_surf_shell, 1),
        "roof_surface_temp_c": round(t_surf_roof, 1),
        "shell_percentage": round(q_shell / q_total * 100, 1) if q_total > 0 else 0,
        "roof_percentage": round(q_roof / q_total * 100, 1) if q_total > 0 else 0,
        "floor_percentage": round(q_floor / q_total * 100, 1) if q_total > 0 else 0,
    }

    logger.debug(
        f"Tank heat loss: D={diameter_m}m, H={height_m}m, "
        f"Q_total={q_total/1000:.1f}kW"
    )

    return q_total, breakdown
