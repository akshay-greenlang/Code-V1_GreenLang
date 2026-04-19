"""
HEI Heat Transfer Calculator

This module implements heat transfer calculations for surface condensers
following HEI (Heat Exchange Institute) Standards for Steam Surface Condensers.

All formulas are from standard heat transfer references:
- HEI Standards for Steam Surface Condensers, 12th Edition
- ASME PTC 12.2 Steam Surface Condensers
- Perry's Chemical Engineers' Handbook (Heat Transfer Section)

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> u_overall = calculate_overall_heat_transfer_coefficient(
    ...     cooling_water_velocity=2.0,
    ...     tube_material="admiralty_brass",
    ...     tube_od_mm=25.4,
    ...     tube_wall_mm=1.245,
    ...     cleanliness_factor=0.85
    ... )
    >>> print(f"Overall U: {u_overall:.1f} W/m2K")
"""

import math
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Tube material thermal conductivity (W/m-K)
# Reference: HEI Standards, Table 2-1
TUBE_THERMAL_CONDUCTIVITY: Dict[str, float] = {
    "admiralty_brass": 111.0,
    "aluminum_brass": 101.0,
    "aluminum_bronze": 73.0,
    "arsenical_copper": 340.0,
    "copper_nickel_90_10": 45.0,
    "copper_nickel_70_30": 29.0,
    "stainless_steel_304": 16.3,
    "stainless_steel_316": 16.3,
    "titanium": 21.0,
    "duplex_stainless": 19.0,
    "carbon_steel": 51.0,
}

# HEI tube material correction factors (Fm)
# Reference: HEI Standards, Table 2-2
HEI_MATERIAL_FACTORS: Dict[str, float] = {
    "admiralty_brass": 1.00,
    "aluminum_brass": 1.00,
    "aluminum_bronze": 0.98,
    "arsenical_copper": 1.04,
    "copper_nickel_90_10": 0.94,
    "copper_nickel_70_30": 0.86,
    "stainless_steel_304": 0.83,
    "stainless_steel_316": 0.83,
    "titanium": 0.85,
    "duplex_stainless": 0.84,
    "carbon_steel": 0.92,
}

# HEI temperature correction factor coefficients
# Reference: HEI Standards, Equation 2-3
HEI_TEMP_CORRECTION = {
    "a": 0.5602,
    "b": 0.0118,
    "c": -0.0000503,
}

# Water properties (for heat transfer correlations)
# Simplified properties at typical condenser temperatures
WATER_PROPERTIES = {
    "thermal_conductivity": 0.628,  # W/m-K at 30C
    "specific_heat": 4180,  # J/kg-K
    "density": 995,  # kg/m3 at 30C
    "prandtl_number": 5.4,  # at 30C
}


def calculate_overall_heat_transfer_coefficient(
    cooling_water_velocity: float,
    tube_material: str,
    tube_od_mm: float,
    tube_wall_mm: float,
    cleanliness_factor: float = 0.85,
    inlet_water_temp_c: float = 25.0
) -> float:
    """
    Calculate overall heat transfer coefficient per HEI Standards.

    The HEI method calculates U using material-specific correlations
    accounting for tube material, geometry, water velocity, and fouling.

    Formula (HEI):
        U = U_base * Fm * Ft * CF

    Where:
        U_base = f(velocity, tube diameter)
        Fm = material correction factor
        Ft = temperature correction factor
        CF = cleanliness factor

    Reference: HEI Standards for Steam Surface Condensers, Section 2

    Args:
        cooling_water_velocity: Tube-side water velocity in m/s.
        tube_material: Tube material (from TUBE_THERMAL_CONDUCTIVITY keys).
        tube_od_mm: Tube outer diameter in millimeters.
        tube_wall_mm: Tube wall thickness in millimeters.
        cleanliness_factor: HEI cleanliness factor (0.0-1.0, typically 0.85).
        inlet_water_temp_c: Inlet cooling water temperature in Celsius.

    Returns:
        Overall heat transfer coefficient in W/m2-K.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> u = calculate_overall_heat_transfer_coefficient(
        ...     2.0, "admiralty_brass", 25.4, 1.245, 0.85
        ... )
        >>> print(f"U = {u:.1f} W/m2K")
    """
    # Validate inputs
    if cooling_water_velocity <= 0 or cooling_water_velocity > 4.0:
        raise ValueError(
            f"Water velocity must be 0-4 m/s, got {cooling_water_velocity}"
        )
    if tube_od_mm <= 0:
        raise ValueError(f"Tube OD must be > 0, got {tube_od_mm}")
    if tube_wall_mm <= 0 or tube_wall_mm >= tube_od_mm / 2:
        raise ValueError(f"Invalid tube wall thickness: {tube_wall_mm}")
    if cleanliness_factor <= 0 or cleanliness_factor > 1.0:
        raise ValueError(
            f"Cleanliness factor must be 0-1, got {cleanliness_factor}"
        )

    # Get material properties
    tube_material_lower = tube_material.lower().replace(" ", "_")
    if tube_material_lower not in HEI_MATERIAL_FACTORS:
        logger.warning(
            f"Unknown tube material {tube_material}, using admiralty_brass"
        )
        tube_material_lower = "admiralty_brass"

    # Calculate tube inner diameter (mm -> m)
    tube_id_m = (tube_od_mm - 2 * tube_wall_mm) / 1000

    # Calculate base heat transfer coefficient (HEI correlation)
    # U_base = 2840 * V^0.5 for standard conditions
    # Reference: HEI Standards, Equation 2-1
    u_base = calculate_hei_base_coefficient(
        cooling_water_velocity,
        tube_od_mm / 1000  # Convert to meters
    )

    # Apply material correction factor (Fm)
    fm = HEI_MATERIAL_FACTORS[tube_material_lower]

    # Calculate temperature correction factor (Ft)
    ft = calculate_temperature_correction_factor(inlet_water_temp_c)

    # Apply cleanliness factor
    u_overall = u_base * fm * ft * cleanliness_factor

    logger.debug(
        f"U calculation: U_base={u_base:.1f}, Fm={fm:.2f}, Ft={ft:.3f}, "
        f"CF={cleanliness_factor:.2f}, U_overall={u_overall:.1f} W/m2K"
    )

    return u_overall


def calculate_hei_base_coefficient(
    velocity: float,
    tube_od: float
) -> float:
    """
    Calculate HEI base heat transfer coefficient.

    Formula (HEI Standards):
        U_base = C1 * V^0.5 * (do/0.0254)^(-0.2)

    Where:
        C1 = 2840 W/m2K (base coefficient)
        V = water velocity (m/s)
        do = tube outer diameter (m)

    Reference: HEI Standards, Equation 2-1

    Args:
        velocity: Cooling water velocity in m/s.
        tube_od: Tube outer diameter in meters.

    Returns:
        Base heat transfer coefficient in W/m2-K.
    """
    # Base coefficient at standard conditions
    c1 = 2840  # W/m2K

    # Reference diameter (1 inch = 0.0254 m)
    d_ref = 0.0254

    # HEI correlation
    u_base = c1 * (velocity ** 0.5) * ((tube_od / d_ref) ** (-0.2))

    return u_base


def calculate_temperature_correction_factor(
    inlet_temp_c: float
) -> float:
    """
    Calculate HEI temperature correction factor.

    Formula (HEI):
        Ft = a + b*T + c*T^2

    Where T is inlet water temperature in Celsius.

    Reference: HEI Standards, Equation 2-3

    Args:
        inlet_temp_c: Inlet cooling water temperature in Celsius.

    Returns:
        Temperature correction factor (typically 0.8-1.2).
    """
    a = HEI_TEMP_CORRECTION["a"]
    b = HEI_TEMP_CORRECTION["b"]
    c = HEI_TEMP_CORRECTION["c"]

    ft = a + b * inlet_temp_c + c * (inlet_temp_c ** 2)

    # Clamp to reasonable range
    return max(0.7, min(1.3, ft))


def calculate_lmtd(
    t_sat: float,
    t_cw_in: float,
    t_cw_out: float
) -> float:
    """
    Calculate Log Mean Temperature Difference (LMTD) for condenser.

    For a condenser, the steam side is isothermal (saturation temperature),
    so the LMTD simplifies to:

    Formula:
        LMTD = (dT1 - dT2) / ln(dT1/dT2)

    Where:
        dT1 = T_sat - T_cw_in (hot end)
        dT2 = T_sat - T_cw_out (cold end) = TTD

    Reference: ASME PTC 12.2, Section 5

    Args:
        t_sat: Steam saturation temperature in Celsius.
        t_cw_in: Cooling water inlet temperature in Celsius.
        t_cw_out: Cooling water outlet temperature in Celsius.

    Returns:
        Log mean temperature difference in Kelvin (or Celsius difference).

    Raises:
        ValueError: If temperature differences are invalid.

    Example:
        >>> lmtd = calculate_lmtd(38.0, 25.0, 35.0)
        >>> print(f"LMTD: {lmtd:.2f} K")
    """
    # Temperature differences
    dt1 = t_sat - t_cw_in  # Hot end
    dt2 = t_sat - t_cw_out  # Cold end (TTD)

    if dt1 <= 0 or dt2 <= 0:
        raise ValueError(
            f"Invalid temperature differences: dT1={dt1}, dT2={dt2}. "
            f"Cooling water must be cooler than steam."
        )

    if dt1 == dt2:
        # Equal temperature differences - LMTD equals the difference
        return dt1

    # Standard LMTD formula
    lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

    return lmtd


def calculate_heat_duty(
    steam_flow_kg_s: float,
    latent_heat_kj_kg: float,
    subcooling_c: float = 0.0,
    specific_heat_kj_kg_k: float = 4.18
) -> float:
    """
    Calculate condenser heat duty.

    Formula:
        Q = m_steam * (h_fg + Cp * subcooling)

    Reference: ASME PTC 12.2

    Args:
        steam_flow_kg_s: Steam/condensate flow rate in kg/s.
        latent_heat_kj_kg: Latent heat of condensation in kJ/kg.
        subcooling_c: Degrees of condensate subcooling in Celsius.
        specific_heat_kj_kg_k: Condensate specific heat in kJ/kg-K.

    Returns:
        Heat duty in kW.

    Example:
        >>> q = calculate_heat_duty(10.0, 2400, 2.0)
        >>> print(f"Heat duty: {q:.0f} kW")
    """
    if steam_flow_kg_s < 0:
        raise ValueError(f"Steam flow must be >= 0, got {steam_flow_kg_s}")
    if latent_heat_kj_kg <= 0:
        raise ValueError(
            f"Latent heat must be > 0, got {latent_heat_kj_kg}"
        )

    # Total heat = latent + sensible (subcooling)
    q_latent = steam_flow_kg_s * latent_heat_kj_kg
    q_subcooling = steam_flow_kg_s * specific_heat_kj_kg_k * subcooling_c

    return q_latent + q_subcooling


def calculate_required_surface_area(
    heat_duty_kw: float,
    u_overall: float,
    lmtd: float
) -> float:
    """
    Calculate required heat transfer surface area.

    Formula:
        A = Q / (U * LMTD)

    Reference: Standard heat transfer equation

    Args:
        heat_duty_kw: Heat duty in kW.
        u_overall: Overall heat transfer coefficient in W/m2-K.
        lmtd: Log mean temperature difference in K.

    Returns:
        Required surface area in m2.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> area = calculate_required_surface_area(24000, 3200, 8.5)
        >>> print(f"Required area: {area:.0f} m2")
    """
    if heat_duty_kw <= 0:
        raise ValueError(f"Heat duty must be > 0, got {heat_duty_kw}")
    if u_overall <= 0:
        raise ValueError(f"U must be > 0, got {u_overall}")
    if lmtd <= 0:
        raise ValueError(f"LMTD must be > 0, got {lmtd}")

    # Convert kW to W
    q_watts = heat_duty_kw * 1000

    # A = Q / (U * LMTD)
    area = q_watts / (u_overall * lmtd)

    return area


def calculate_tube_side_heat_transfer(
    velocity: float,
    tube_id: float,
    water_temp_c: float = 30.0
) -> float:
    """
    Calculate tube-side (water) heat transfer coefficient.

    Uses Dittus-Boelter correlation for turbulent flow in tubes:
        Nu = 0.023 * Re^0.8 * Pr^0.4

    Reference: Perry's Chemical Engineers' Handbook

    Args:
        velocity: Water velocity in m/s.
        tube_id: Tube inner diameter in meters.
        water_temp_c: Average water temperature in Celsius.

    Returns:
        Tube-side heat transfer coefficient in W/m2-K.

    Example:
        >>> h_tube = calculate_tube_side_heat_transfer(2.0, 0.022)
        >>> print(f"h_tube: {h_tube:.0f} W/m2K")
    """
    # Get water properties (simplified - constant properties)
    k = WATER_PROPERTIES["thermal_conductivity"]
    rho = WATER_PROPERTIES["density"]
    cp = WATER_PROPERTIES["specific_heat"]
    pr = WATER_PROPERTIES["prandtl_number"]

    # Dynamic viscosity at 30C (approximate)
    mu = 0.0008  # Pa-s

    # Reynolds number
    re = rho * velocity * tube_id / mu

    if re < 2300:
        logger.warning(
            f"Laminar flow (Re={re:.0f}). Dittus-Boelter may not apply."
        )
        # Use Sieder-Tate for laminar
        nu = 3.66
    else:
        # Dittus-Boelter for turbulent flow (heating case)
        nu = 0.023 * (re ** 0.8) * (pr ** 0.4)

    # h = Nu * k / D
    h_tube = nu * k / tube_id

    logger.debug(
        f"Tube-side: Re={re:.0f}, Nu={nu:.1f}, h={h_tube:.0f} W/m2K"
    )

    return h_tube


def calculate_shell_side_heat_transfer(
    saturation_pressure_kpa: float,
    tube_od: float,
    air_fraction: float = 0.0
) -> float:
    """
    Calculate shell-side (condensing steam) heat transfer coefficient.

    Uses Nusselt film condensation theory for horizontal tubes:
        h = 0.725 * [rho_l * g * h_fg * k_l^3 / (mu_l * dT * D)]^0.25

    Simplified correlation for steam condensation:
        h = C * (1 - air_fraction)

    Where C is based on pressure and geometry.

    Reference: Kern's Process Heat Transfer, Nusselt theory

    Args:
        saturation_pressure_kpa: Steam saturation pressure in kPa.
        tube_od: Tube outer diameter in meters.
        air_fraction: Mass fraction of air in steam (0.0-1.0).

    Returns:
        Shell-side heat transfer coefficient in W/m2-K.

    Example:
        >>> h_shell = calculate_shell_side_heat_transfer(7.0, 0.0254, 0.001)
        >>> print(f"h_shell: {h_shell:.0f} W/m2K")
    """
    # Base coefficient for film condensation on horizontal tubes
    # Typical range: 8000-15000 W/m2K for steam at vacuum pressures

    # Pressure effect (lower pressure = lower h due to vapor properties)
    if saturation_pressure_kpa < 5:
        h_base = 8000
    elif saturation_pressure_kpa < 10:
        h_base = 10000
    elif saturation_pressure_kpa < 20:
        h_base = 12000
    else:
        h_base = 14000

    # Tube diameter effect (smaller tubes = higher h)
    # Reference: h ~ D^(-0.25) from Nusselt theory
    d_ref = 0.0254  # 1 inch reference
    d_factor = (d_ref / tube_od) ** 0.25

    # Air/non-condensable effect (dramatic reduction in h)
    # Reference: HEI Standards, air blanketing effects
    if air_fraction > 0:
        # Empirical: h reduces by ~(1 - 0.9*air_fraction) for small fractions
        air_factor = max(0.1, 1 - 9 * air_fraction)
    else:
        air_factor = 1.0

    h_shell = h_base * d_factor * air_factor

    logger.debug(
        f"Shell-side: h_base={h_base}, d_factor={d_factor:.3f}, "
        f"air_factor={air_factor:.3f}, h_shell={h_shell:.0f} W/m2K"
    )

    return h_shell


def calculate_tube_wall_resistance(
    tube_od: float,
    tube_id: float,
    thermal_conductivity: float
) -> float:
    """
    Calculate thermal resistance of tube wall.

    Formula (cylindrical wall):
        R_wall = ln(Do/Di) / (2 * pi * k * L)

    Per unit area (based on OD):
        R_wall = Do * ln(Do/Di) / (2 * k)

    Reference: Standard heat conduction

    Args:
        tube_od: Tube outer diameter in meters.
        tube_id: Tube inner diameter in meters.
        thermal_conductivity: Tube material conductivity in W/m-K.

    Returns:
        Wall thermal resistance in m2-K/W.

    Example:
        >>> r_wall = calculate_tube_wall_resistance(0.0254, 0.0229, 111)
        >>> print(f"R_wall: {r_wall:.6f} m2K/W")
    """
    if tube_od <= tube_id:
        raise ValueError(f"OD ({tube_od}) must be > ID ({tube_id})")
    if thermal_conductivity <= 0:
        raise ValueError(f"Conductivity must be > 0, got {thermal_conductivity}")

    # R = Do * ln(Do/Di) / (2*k)
    r_wall = tube_od * math.log(tube_od / tube_id) / (2 * thermal_conductivity)

    return r_wall


def calculate_overall_u_from_resistances(
    h_tube: float,
    h_shell: float,
    r_wall: float,
    r_fouling_tube: float = 0.0,
    r_fouling_shell: float = 0.0,
    area_ratio: float = 1.0
) -> float:
    """
    Calculate overall U from individual resistances.

    Formula:
        1/U = 1/h_shell + R_fouling_shell + R_wall +
              (Ao/Ai) * (R_fouling_tube + 1/h_tube)

    Reference: Standard heat transfer resistance model

    Args:
        h_tube: Tube-side heat transfer coefficient (W/m2-K).
        h_shell: Shell-side heat transfer coefficient (W/m2-K).
        r_wall: Wall thermal resistance (m2-K/W).
        r_fouling_tube: Tube-side fouling resistance (m2-K/W).
        r_fouling_shell: Shell-side fouling resistance (m2-K/W).
        area_ratio: Ratio of outer to inner surface area (Ao/Ai).

    Returns:
        Overall heat transfer coefficient in W/m2-K.

    Example:
        >>> u = calculate_overall_u_from_resistances(
        ...     h_tube=8000, h_shell=10000, r_wall=0.00001
        ... )
    """
    if h_tube <= 0 or h_shell <= 0:
        raise ValueError("Heat transfer coefficients must be > 0")

    # Sum of resistances (based on outside area)
    r_total = (
        1 / h_shell +
        r_fouling_shell +
        r_wall +
        area_ratio * (r_fouling_tube + 1 / h_tube)
    )

    # U = 1 / R_total
    u_overall = 1 / r_total

    return u_overall


def calculate_ttd(
    t_sat: float,
    t_cw_out: float
) -> float:
    """
    Calculate Terminal Temperature Difference (TTD).

    Formula:
        TTD = T_sat - T_cw_out

    The TTD is a key performance indicator for condensers.
    Lower TTD indicates better performance.

    Typical values:
        - Excellent: < 3C
        - Good: 3-5C
        - Acceptable: 5-8C
        - Poor: > 8C

    Reference: HEI Standards, ASME PTC 12.2

    Args:
        t_sat: Steam saturation temperature in Celsius.
        t_cw_out: Cooling water outlet temperature in Celsius.

    Returns:
        Terminal temperature difference in Celsius (or Kelvin).

    Raises:
        ValueError: If TTD would be negative.

    Example:
        >>> ttd = calculate_ttd(38.5, 35.0)
        >>> print(f"TTD: {ttd:.1f} C")
    """
    ttd = t_sat - t_cw_out

    if ttd < 0:
        raise ValueError(
            f"Negative TTD ({ttd:.2f}C) indicates cooling water hotter than steam. "
            f"Check input temperatures."
        )

    return ttd


def calculate_cooling_water_rise(
    t_cw_in: float,
    t_cw_out: float
) -> float:
    """
    Calculate cooling water temperature rise.

    Formula:
        Rise = T_cw_out - T_cw_in

    Reference: Standard temperature measurement

    Args:
        t_cw_in: Cooling water inlet temperature in Celsius.
        t_cw_out: Cooling water outlet temperature in Celsius.

    Returns:
        Temperature rise in Celsius.

    Example:
        >>> rise = calculate_cooling_water_rise(25.0, 35.0)
        >>> print(f"Rise: {rise:.1f} C")
    """
    return t_cw_out - t_cw_in


def calculate_cooling_water_flow(
    heat_duty_kw: float,
    t_cw_in: float,
    t_cw_out: float,
    specific_heat: float = 4.18  # kJ/kg-K
) -> float:
    """
    Calculate required cooling water flow rate.

    Formula:
        m_cw = Q / (Cp * dT)

    Reference: Energy balance

    Args:
        heat_duty_kw: Condenser heat duty in kW.
        t_cw_in: Cooling water inlet temperature in Celsius.
        t_cw_out: Cooling water outlet temperature in Celsius.
        specific_heat: Water specific heat in kJ/kg-K.

    Returns:
        Cooling water flow rate in kg/s.

    Example:
        >>> flow = calculate_cooling_water_flow(24000, 25.0, 35.0)
        >>> print(f"Flow: {flow:.0f} kg/s")
    """
    dt = t_cw_out - t_cw_in
    if dt <= 0:
        raise ValueError(f"Temperature rise must be > 0, got {dt}")

    m_cw = heat_duty_kw / (specific_heat * dt)

    return m_cw


def estimate_latent_heat(
    saturation_temp_c: float
) -> float:
    """
    Estimate latent heat of vaporization at given saturation temperature.

    Uses simplified correlation for water:
        h_fg = 2500.9 - 2.36*T - 0.0016*T^2

    Valid for T from 0 to 100C.

    Reference: Steam tables correlation

    Args:
        saturation_temp_c: Saturation temperature in Celsius.

    Returns:
        Latent heat of vaporization in kJ/kg.

    Example:
        >>> h_fg = estimate_latent_heat(40.0)
        >>> print(f"Latent heat: {h_fg:.1f} kJ/kg")
    """
    t = saturation_temp_c

    # Correlation for latent heat
    h_fg = 2500.9 - 2.36 * t - 0.0016 * (t ** 2)

    # Clamp to reasonable range
    return max(2200, min(2500, h_fg))
