"""
Surface Heat Loss Calculator

Physics-based calculations for convection and radiation heat loss
from hot surfaces to ambient environment.

References:
    - ISO 12241: Thermal Insulation for Building Equipment
    - ASTM C680: Standard Practice for Estimate of Heat Gain or Loss
    - VDI 2055: Thermal Insulation for Heated and Cooled Equipment
"""

import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/(m^2*K^4)

# Air properties at various temperatures
AIR_PROPERTIES = {
    # temp_c: (density kg/m3, viscosity Pa*s, k W/m-K, Pr)
    0: (1.293, 1.71e-5, 0.0243, 0.715),
    20: (1.205, 1.81e-5, 0.0257, 0.713),
    40: (1.127, 1.91e-5, 0.0271, 0.711),
    60: (1.060, 2.00e-5, 0.0285, 0.709),
    80: (1.000, 2.09e-5, 0.0299, 0.708),
    100: (0.946, 2.18e-5, 0.0314, 0.707),
}


def calculate_convection_coefficient(
    surface_temp_c: float,
    ambient_temp_c: float,
    wind_speed_m_s: float = 0.0,
    surface_orientation: str = "vertical",
    characteristic_length_m: float = 1.0
) -> float:
    """
    Calculate convection heat transfer coefficient.

    Combines natural and forced convection using appropriate correlations.

    For natural convection (vertical surfaces):
        Nu = 0.59 * (Gr * Pr)^0.25  for 10^4 < Gr*Pr < 10^9
        Nu = 0.10 * (Gr * Pr)^0.33  for Gr*Pr > 10^9

    For forced convection (external flow):
        h = 5.0 + 3.8 * v  (simplified correlation for air)

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient air temperature in Celsius.
        wind_speed_m_s: Wind or air velocity in m/s.
        surface_orientation: "vertical", "horizontal_up", "horizontal_down".
        characteristic_length_m: Surface height or diameter.

    Returns:
        Convection coefficient h in W/(m^2*K).

    Example:
        >>> h = calculate_convection_coefficient(100, 20, 0, "vertical", 1.0)
        >>> print(f"h = {h:.1f} W/(m^2*K)")
    """
    delta_t = abs(surface_temp_c - ambient_temp_c)
    if delta_t < 0.1:
        return 5.0  # Minimum for still air

    # Film temperature for properties
    film_temp = (surface_temp_c + ambient_temp_c) / 2

    # Get air properties (interpolate)
    rho, mu, k_air, pr = _interpolate_air_properties(film_temp)

    # Natural convection coefficient
    h_natural = _calculate_natural_convection(
        delta_t, surface_temp_c > ambient_temp_c,
        rho, mu, k_air, pr,
        characteristic_length_m, surface_orientation
    )

    # Forced convection coefficient
    h_forced = 0.0
    if wind_speed_m_s > 0.1:
        # Simplified correlation: h = 5.0 + 3.8 * v
        h_forced = 5.0 + 3.8 * wind_speed_m_s

    # Combined convection (take larger or use correlation)
    # For mixed convection: h = (h_nat^3 + h_forced^3)^(1/3)
    if h_forced > 0 and h_natural > 0:
        h_combined = (h_natural**3 + h_forced**3)**(1/3)
    else:
        h_combined = max(h_natural, h_forced, 5.0)

    logger.debug(
        f"Convection: h_nat={h_natural:.1f}, h_forced={h_forced:.1f}, "
        f"h_combined={h_combined:.1f} W/(m^2*K)"
    )

    return h_combined


def _interpolate_air_properties(temp_c: float) -> Tuple[float, float, float, float]:
    """Interpolate air properties at given temperature."""
    temps = sorted(AIR_PROPERTIES.keys())

    if temp_c <= temps[0]:
        return AIR_PROPERTIES[temps[0]]
    if temp_c >= temps[-1]:
        return AIR_PROPERTIES[temps[-1]]

    # Find bracketing temperatures
    for i in range(len(temps) - 1):
        if temps[i] <= temp_c <= temps[i + 1]:
            t1, t2 = temps[i], temps[i + 1]
            frac = (temp_c - t1) / (t2 - t1)
            props1 = AIR_PROPERTIES[t1]
            props2 = AIR_PROPERTIES[t2]
            return tuple(
                p1 + frac * (p2 - p1) for p1, p2 in zip(props1, props2)
            )

    return AIR_PROPERTIES[20]  # Default


def _calculate_natural_convection(
    delta_t: float,
    surface_hotter: bool,
    rho: float,
    mu: float,
    k: float,
    pr: float,
    length: float,
    orientation: str
) -> float:
    """Calculate natural convection coefficient using Nusselt correlations."""
    g = 9.81  # m/s^2
    beta = 1 / (273.15 + 20)  # Thermal expansion coefficient for air

    # Grashof number
    gr = g * beta * abs(delta_t) * length**3 * rho**2 / mu**2

    # Rayleigh number
    ra = gr * pr

    # Nusselt number correlation depends on orientation
    if orientation == "vertical":
        if ra < 1e4:
            nu = 1.0
        elif ra < 1e9:
            nu = 0.59 * ra**0.25
        else:
            nu = 0.10 * ra**(1/3)

    elif orientation == "horizontal_up":
        # Heated surface facing up or cooled surface facing down
        if surface_hotter:
            if ra < 1e7:
                nu = 0.54 * ra**0.25
            else:
                nu = 0.15 * ra**(1/3)
        else:
            nu = 0.27 * ra**0.25

    elif orientation == "horizontal_down":
        # Heated surface facing down or cooled surface facing up
        if surface_hotter:
            nu = 0.27 * ra**0.25
        else:
            if ra < 1e7:
                nu = 0.54 * ra**0.25
            else:
                nu = 0.15 * ra**(1/3)
    else:
        nu = 0.59 * ra**0.25  # Default to vertical

    h = nu * k / length
    return max(5.0, h)  # Minimum 5 W/(m^2*K)


def calculate_convection_heat_loss(
    surface_temp_c: float,
    ambient_temp_c: float,
    surface_area_m2: float,
    h_convection: Optional[float] = None,
    wind_speed_m_s: float = 0.0
) -> float:
    """
    Calculate convection heat loss from surface.

    Q_conv = h * A * (T_s - T_a)

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient temperature in Celsius.
        surface_area_m2: Heat transfer area in m^2.
        h_convection: Convection coefficient (auto-calculated if None).
        wind_speed_m_s: Wind speed for auto-calculation.

    Returns:
        Convection heat loss in Watts.

    Example:
        >>> q = calculate_convection_heat_loss(100, 20, 10)
        >>> print(f"Q_conv = {q/1000:.1f} kW")
    """
    if h_convection is None:
        h_convection = calculate_convection_coefficient(
            surface_temp_c, ambient_temp_c, wind_speed_m_s
        )

    delta_t = surface_temp_c - ambient_temp_c
    q_conv = h_convection * surface_area_m2 * delta_t

    return q_conv


def calculate_radiation_heat_loss(
    surface_temp_c: float,
    ambient_temp_c: float,
    surface_area_m2: float,
    emissivity: float = 0.9,
    view_factor: float = 1.0
) -> float:
    """
    Calculate radiation heat loss from surface using Stefan-Boltzmann law.

    Q_rad = epsilon * sigma * F * A * (T_s^4 - T_a^4)

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient/surroundings temperature in Celsius.
        surface_area_m2: Radiating surface area in m^2.
        emissivity: Surface emissivity (0-1, default 0.9).
        view_factor: Geometric view factor to surroundings (default 1.0).

    Returns:
        Radiation heat loss in Watts.

    Example:
        >>> q = calculate_radiation_heat_loss(100, 20, 10, 0.9)
        >>> print(f"Q_rad = {q/1000:.1f} kW")
    """
    # Convert to Kelvin
    t_s_k = surface_temp_c + 273.15
    t_a_k = ambient_temp_c + 273.15

    # Stefan-Boltzmann radiation
    q_rad = emissivity * STEFAN_BOLTZMANN * view_factor * surface_area_m2 * (
        t_s_k**4 - t_a_k**4
    )

    return q_rad


def calculate_combined_surface_heat_loss(
    surface_temp_c: float,
    ambient_temp_c: float,
    surface_area_m2: float,
    emissivity: float = 0.9,
    wind_speed_m_s: float = 0.0,
    surface_orientation: str = "vertical"
) -> Tuple[float, float, float, dict]:
    """
    Calculate combined convection and radiation heat loss.

    Returns breakdown of heat loss components.

    Args:
        surface_temp_c: Surface temperature in Celsius.
        ambient_temp_c: Ambient temperature in Celsius.
        surface_area_m2: Surface area in m^2.
        emissivity: Surface emissivity.
        wind_speed_m_s: Wind speed.
        surface_orientation: Surface orientation.

    Returns:
        Tuple of (q_convection, q_radiation, q_total, details_dict).

    Example:
        >>> q_conv, q_rad, q_total, details = calculate_combined_surface_heat_loss(
        ...     100, 20, 10, 0.9, 0
        ... )
    """
    # Calculate convection coefficient
    h_conv = calculate_convection_coefficient(
        surface_temp_c, ambient_temp_c, wind_speed_m_s, surface_orientation
    )

    # Convection heat loss
    q_conv = calculate_convection_heat_loss(
        surface_temp_c, ambient_temp_c, surface_area_m2, h_conv
    )

    # Radiation heat loss
    q_rad = calculate_radiation_heat_loss(
        surface_temp_c, ambient_temp_c, surface_area_m2, emissivity
    )

    # Total
    q_total = q_conv + q_rad

    # Calculate equivalent combined coefficient
    delta_t = surface_temp_c - ambient_temp_c
    if abs(delta_t) > 0.1:
        h_combined = q_total / (surface_area_m2 * delta_t)
        h_rad = q_rad / (surface_area_m2 * delta_t)
    else:
        h_combined = h_conv
        h_rad = 0

    details = {
        "h_convection": round(h_conv, 2),
        "h_radiation_equiv": round(h_rad, 2),
        "h_combined": round(h_combined, 2),
        "heat_flux_w_m2": round(q_total / surface_area_m2, 1) if surface_area_m2 > 0 else 0,
        "convection_fraction": round(q_conv / q_total * 100, 1) if q_total > 0 else 0,
        "radiation_fraction": round(q_rad / q_total * 100, 1) if q_total > 0 else 0,
    }

    return q_conv, q_rad, q_total, details
