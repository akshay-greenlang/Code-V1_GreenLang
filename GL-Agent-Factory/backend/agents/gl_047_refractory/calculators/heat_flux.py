"""
Heat Flux Calculation Engine

Physics-based calculations for heat transfer through refractory materials.
Implements Fourier's law of heat conduction and related thermal analysis.

References:
    - ASTM C155: Standard Classification of Insulating Firebrick
    - ISO 836: Terminology for Refractories
    - ASTM C201: Standard Test Method for Thermal Conductivity
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thermal conductivity values (W/m-K) at various temperatures
# Format: {material: {temp_c: k_value}}
THERMAL_CONDUCTIVITY_DATA = {
    "firebrick": {
        20: 1.0, 200: 1.1, 400: 1.2, 600: 1.3, 800: 1.4, 1000: 1.5, 1200: 1.6
    },
    "castable": {
        20: 1.2, 200: 1.3, 400: 1.4, 600: 1.5, 800: 1.6, 1000: 1.7, 1200: 1.8
    },
    "ceramic_fiber": {
        20: 0.06, 200: 0.08, 400: 0.12, 600: 0.18, 800: 0.25, 1000: 0.35, 1200: 0.45
    },
    "insulating_firebrick": {
        20: 0.2, 200: 0.22, 400: 0.25, 600: 0.28, 800: 0.32, 1000: 0.36, 1200: 0.40
    },
    "dense_alumina": {
        20: 2.5, 200: 2.8, 400: 3.0, 600: 3.2, 800: 3.3, 1000: 3.4, 1200: 3.5
    },
}


def calculate_heat_flux(
    hot_face_temp_c: float,
    cold_face_temp_c: float,
    thickness_m: float,
    thermal_conductivity: float
) -> float:
    """
    Calculate heat flux through refractory using Fourier's law.

    Fourier's Law (1D steady-state):
        q = k * (T_hot - T_cold) / L

    where:
        q = heat flux (W/m^2)
        k = thermal conductivity (W/m-K)
        T_hot, T_cold = temperatures (K or C, difference is same)
        L = thickness (m)

    Args:
        hot_face_temp_c: Hot face temperature in Celsius.
        cold_face_temp_c: Cold face temperature in Celsius.
        thickness_m: Refractory thickness in meters.
        thermal_conductivity: Material conductivity in W/m-K.

    Returns:
        Heat flux in W/m^2 (positive = heat flowing from hot to cold).

    Raises:
        ValueError: If thickness is zero or negative.

    Example:
        >>> q = calculate_heat_flux(1200, 200, 0.23, 1.5)
        >>> print(f"Heat flux: {q:.0f} W/m^2")
        Heat flux: 6522 W/m^2
    """
    if thickness_m <= 0:
        raise ValueError(f"Thickness must be > 0, got {thickness_m}")

    if thermal_conductivity <= 0:
        raise ValueError(f"Thermal conductivity must be > 0, got {thermal_conductivity}")

    delta_t = hot_face_temp_c - cold_face_temp_c
    heat_flux = thermal_conductivity * delta_t / thickness_m

    logger.debug(
        f"Heat flux: k={thermal_conductivity:.2f}, dT={delta_t:.0f}C, "
        f"L={thickness_m*1000:.0f}mm, q={heat_flux:.0f} W/m^2"
    )

    return heat_flux


def calculate_conduction_heat_flux(
    temperatures: List[float],
    thicknesses: List[float],
    conductivities: List[float]
) -> Tuple[float, List[float]]:
    """
    Calculate heat flux through multi-layer refractory system.

    For multi-layer systems in series:
        q = (T_1 - T_n) / sum(L_i / k_i)

    Args:
        temperatures: List of interface temperatures (n+1 values for n layers).
        thicknesses: List of layer thicknesses in meters (n values).
        conductivities: List of thermal conductivities in W/m-K (n values).

    Returns:
        Tuple of (total_heat_flux, interface_heat_fluxes).

    Example:
        >>> temps = [1200, 800, 200]  # Hot, interface, cold
        >>> thick = [0.15, 0.10]  # Two layers
        >>> k = [1.5, 0.3]  # Dense then insulating
        >>> q, q_layers = calculate_conduction_heat_flux(temps, thick, k)
    """
    if len(temperatures) != len(thicknesses) + 1:
        raise ValueError(
            f"Need {len(thicknesses) + 1} temperatures for {len(thicknesses)} layers"
        )

    if len(thicknesses) != len(conductivities):
        raise ValueError("Thicknesses and conductivities must have same length")

    # Calculate total thermal resistance
    total_resistance = sum(
        L / k for L, k in zip(thicknesses, conductivities)
    )

    if total_resistance <= 0:
        raise ValueError("Total thermal resistance must be > 0")

    # Total temperature difference
    delta_t_total = temperatures[0] - temperatures[-1]

    # Total heat flux
    total_heat_flux = delta_t_total / total_resistance

    # Calculate heat flux through each layer (should be equal in steady state)
    layer_fluxes = []
    for i, (L, k) in enumerate(zip(thicknesses, conductivities)):
        delta_t_layer = temperatures[i] - temperatures[i + 1]
        q_layer = k * delta_t_layer / L
        layer_fluxes.append(q_layer)

    return total_heat_flux, layer_fluxes


def calculate_thermal_conductivity(
    material_type: str,
    mean_temperature_c: float,
    degradation_factor: float = 1.0
) -> float:
    """
    Calculate thermal conductivity with temperature dependence.

    Thermal conductivity varies with temperature. For refractories,
    it generally increases with temperature (unlike metals).

    The degradation factor accounts for material deterioration:
    - 1.0 = new material
    - > 1.0 = degraded (higher conductivity due to cracks/porosity)

    Args:
        material_type: Type of refractory material.
        mean_temperature_c: Mean temperature through the material.
        degradation_factor: Factor for material degradation (1.0 = new).

    Returns:
        Thermal conductivity in W/m-K.

    Example:
        >>> k = calculate_thermal_conductivity("firebrick", 600, 1.1)
        >>> print(f"Conductivity: {k:.2f} W/m-K")
    """
    material = material_type.lower().replace(" ", "_")

    if material not in THERMAL_CONDUCTIVITY_DATA:
        logger.warning(f"Unknown material '{material}', using firebrick defaults")
        material = "firebrick"

    data = THERMAL_CONDUCTIVITY_DATA[material]

    # Get available temperature points
    temps = sorted(data.keys())

    # Interpolate/extrapolate
    if mean_temperature_c <= temps[0]:
        base_k = data[temps[0]]
    elif mean_temperature_c >= temps[-1]:
        base_k = data[temps[-1]]
    else:
        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= mean_temperature_c <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                k1, k2 = data[t1], data[t2]
                frac = (mean_temperature_c - t1) / (t2 - t1)
                base_k = k1 + frac * (k2 - k1)
                break
        else:
            base_k = data[temps[0]]

    # Apply degradation factor
    effective_k = base_k * degradation_factor

    logger.debug(
        f"Conductivity for {material} at {mean_temperature_c:.0f}C: "
        f"base={base_k:.3f}, degraded={effective_k:.3f} W/m-K"
    )

    return effective_k


def calculate_temperature_gradient(
    hot_face_temp_c: float,
    cold_face_temp_c: float,
    thickness_m: float
) -> float:
    """
    Calculate temperature gradient through refractory.

    The temperature gradient indicates thermal stress potential:
    - High gradient = high thermal stress, spalling risk
    - Typical limit: < 100 C/cm for dense refractories

    Args:
        hot_face_temp_c: Hot face temperature in Celsius.
        cold_face_temp_c: Cold face temperature in Celsius.
        thickness_m: Refractory thickness in meters.

    Returns:
        Temperature gradient in C/m (or K/m).
    """
    if thickness_m <= 0:
        raise ValueError(f"Thickness must be > 0, got {thickness_m}")

    gradient = (hot_face_temp_c - cold_face_temp_c) / thickness_m

    # Convert to C/cm for comparison
    gradient_per_cm = gradient / 100

    if gradient_per_cm > 100:
        logger.warning(
            f"High temperature gradient: {gradient_per_cm:.1f} C/cm - "
            "thermal shock/spalling risk"
        )

    return gradient


def calculate_heat_loss_rate(
    heat_flux_w_m2: float,
    surface_area_m2: float
) -> float:
    """
    Calculate total heat loss rate.

    Args:
        heat_flux_w_m2: Heat flux in W/m^2.
        surface_area_m2: Total surface area in m^2.

    Returns:
        Heat loss rate in kW.
    """
    return heat_flux_w_m2 * surface_area_m2 / 1000


def calculate_interface_temperature(
    hot_face_temp_c: float,
    cold_face_temp_c: float,
    distance_from_hot_m: float,
    total_thickness_m: float
) -> float:
    """
    Calculate temperature at a given depth (assuming linear profile).

    For steady-state conduction with constant k, temperature
    varies linearly through the material.

    Args:
        hot_face_temp_c: Hot face temperature.
        cold_face_temp_c: Cold face temperature.
        distance_from_hot_m: Distance from hot face.
        total_thickness_m: Total thickness.

    Returns:
        Temperature at specified depth in Celsius.
    """
    if total_thickness_m <= 0:
        raise ValueError("Thickness must be > 0")

    if distance_from_hot_m < 0 or distance_from_hot_m > total_thickness_m:
        raise ValueError("Distance must be within material thickness")

    fraction = distance_from_hot_m / total_thickness_m
    temp = hot_face_temp_c - fraction * (hot_face_temp_c - cold_face_temp_c)

    return temp


def calculate_thermal_resistance(
    thickness_m: float,
    thermal_conductivity: float,
    area_m2: float = 1.0
) -> float:
    """
    Calculate thermal resistance of refractory layer.

    R = L / (k * A)

    For unit area (A=1), R_unit = L / k (m^2-K/W)

    Args:
        thickness_m: Layer thickness in meters.
        thermal_conductivity: Conductivity in W/m-K.
        area_m2: Surface area (default 1.0 for unit resistance).

    Returns:
        Thermal resistance in K/W (or m^2-K/W if area=1).
    """
    if thickness_m <= 0 or thermal_conductivity <= 0 or area_m2 <= 0:
        raise ValueError("All parameters must be > 0")

    resistance = thickness_m / (thermal_conductivity * area_m2)

    return resistance
