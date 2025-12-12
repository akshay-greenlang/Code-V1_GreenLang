"""
Draft Calculation Engine

Physics-based calculations for furnace draft and stack effect.
Implements equations from API 560 and NFPA 86 standards.

The stack effect (natural draft) is driven by the density difference
between hot flue gas and ambient air, creating buoyancy that draws
combustion products up through the stack.

References:
    - API 560 Section 8: Fired Heaters for General Refinery Service
    - NFPA 86: Standard for Ovens and Furnaces
    - ASHRAE Fundamentals: Stack Effect Calculations
"""

import math
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Physical constants
GAS_CONSTANT_AIR = 287.058  # J/(kg*K) - Specific gas constant for air
GRAVITY = 9.81  # m/s^2
STANDARD_PRESSURE_PA = 101325  # Pa
STANDARD_DENSITY_KG_M3 = 1.293  # kg/m^3 at 0C, 1 atm
INCHES_WC_TO_PA = 249.089  # 1 inWC = 249.089 Pa


def calculate_stack_effect(
    stack_temp_c: float,
    ambient_temp_c: float,
    height_m: float,
    atmospheric_pressure_pa: float = STANDARD_PRESSURE_PA,
    flue_gas_molecular_weight: float = 28.97
) -> float:
    """
    Calculate stack effect (natural draft) in Pascals.

    The stack effect is caused by the buoyancy of hot gases relative to
    cooler ambient air. This creates a pressure differential that drives
    natural draft in furnaces and chimneys.

    Formula (derived from ideal gas law and hydrostatics):
        Delta_P = g * H * P_atm * M / R * (1/T_ambient - 1/T_stack)

    Simplified form used in practice:
        Delta_P = C * H * (1/T_a - 1/T_s)  [Pa]

    where:
        C = constant based on air properties (3462.5 for air)
        H = stack height (m)
        T_a, T_s = absolute temperatures (K)

    Args:
        stack_temp_c: Stack gas temperature in Celsius. Must be > ambient.
        ambient_temp_c: Ambient air temperature in Celsius.
        height_m: Effective stack height in meters. Must be > 0.
        atmospheric_pressure_pa: Local atmospheric pressure in Pa.
        flue_gas_molecular_weight: Molecular weight of flue gas (default 28.97 for air).

    Returns:
        Stack effect in Pascals (positive = upward draft).

    Raises:
        ValueError: If parameters are physically invalid.

    Example:
        >>> stack_effect = calculate_stack_effect(400, 20, 30)
        >>> print(f"Stack effect: {stack_effect:.1f} Pa")
        Stack effect: 89.4 Pa
    """
    if height_m <= 0:
        raise ValueError(f"Stack height must be > 0, got {height_m}")
    if stack_temp_c <= ambient_temp_c:
        logger.warning(
            f"Stack temp ({stack_temp_c}C) <= ambient ({ambient_temp_c}C) - "
            "negative or zero draft expected"
        )

    # Convert to Kelvin
    stack_k = stack_temp_c + 273.15
    ambient_k = ambient_temp_c + 273.15

    # Validate temperatures
    if stack_k <= 0 or ambient_k <= 0:
        raise ValueError("Temperatures must be > -273.15C (absolute zero)")

    # Calculate stack effect using the standard formula
    # C = P * M * g / R = 101325 * 0.02897 * 9.81 / 8.314 = 3462.5 for air
    # For flue gases, adjust by molecular weight ratio
    mw_ratio = flue_gas_molecular_weight / 28.97
    c_factor = 3462.5 * mw_ratio * (atmospheric_pressure_pa / STANDARD_PRESSURE_PA)

    # Stack effect: positive means upward draft (negative pressure at furnace)
    stack_effect_pa = c_factor * height_m * (1.0 / ambient_k - 1.0 / stack_k)

    logger.debug(
        f"Stack effect calculation: H={height_m}m, Ts={stack_temp_c}C, "
        f"Ta={ambient_temp_c}C, SE={stack_effect_pa:.2f} Pa"
    )

    return stack_effect_pa


def calculate_stack_effect_inwc(
    stack_temp_c: float,
    ambient_temp_c: float,
    height_m: float
) -> float:
    """
    Calculate stack effect in inches of water column (inWC).

    This is the common unit used in North American furnace controls.
    Negative values indicate vacuum (draft) relative to atmosphere.

    Formula (per API 560):
        SE = 7.64 * H * (1/T_a - 1/T_s)  [inWC]

    where temperatures are in Kelvin and H is in meters.

    Args:
        stack_temp_c: Stack gas temperature in Celsius.
        ambient_temp_c: Ambient air temperature in Celsius.
        height_m: Effective stack height in meters.

    Returns:
        Stack effect in inches water column (positive = upward draft).

    Example:
        >>> se_inwc = calculate_stack_effect_inwc(400, 20, 30)
        >>> print(f"Stack effect: {se_inwc:.3f} inWC")
    """
    stack_k = stack_temp_c + 273.15
    ambient_k = ambient_temp_c + 273.15

    if height_m <= 0:
        raise ValueError(f"Stack height must be > 0, got {height_m}")

    # API 560 formula constant: 7.64 when H is in meters, T in Kelvin
    stack_effect_inwc = 7.64 * height_m * (1.0 / ambient_k - 1.0 / stack_k)

    return round(stack_effect_inwc, 4)


def calculate_draft_loss(
    flue_gas_flow_kg_s: float,
    duct_area_m2: float,
    duct_length_m: float,
    friction_factor: float = 0.02,
    flue_gas_density_kg_m3: float = 0.5,
    local_loss_coeff: float = 1.5
) -> float:
    """
    Calculate draft loss through ductwork and stack.

    Uses the Darcy-Weisbach equation for friction losses plus
    minor losses from bends, dampers, and expansions.

    Formula:
        Delta_P_friction = f * (L/D) * (rho * v^2 / 2)
        Delta_P_local = K * (rho * v^2 / 2)

    Args:
        flue_gas_flow_kg_s: Mass flow rate of flue gas in kg/s.
        duct_area_m2: Cross-sectional area of duct in m^2.
        duct_length_m: Total duct length in meters.
        friction_factor: Darcy friction factor (0.015-0.025 typical).
        flue_gas_density_kg_m3: Flue gas density in kg/m^3.
        local_loss_coeff: Sum of local loss coefficients (K-factors).

    Returns:
        Total draft loss in Pascals.

    Example:
        >>> loss = calculate_draft_loss(10, 2.0, 50, 0.02, 0.5, 2.0)
        >>> print(f"Draft loss: {loss:.1f} Pa")
    """
    if duct_area_m2 <= 0:
        raise ValueError(f"Duct area must be > 0, got {duct_area_m2}")
    if flue_gas_density_kg_m3 <= 0:
        raise ValueError(f"Density must be > 0, got {flue_gas_density_kg_m3}")

    # Calculate velocity
    velocity = flue_gas_flow_kg_s / (flue_gas_density_kg_m3 * duct_area_m2)

    # Hydraulic diameter (assuming circular duct)
    diameter = math.sqrt(4 * duct_area_m2 / math.pi)

    # Dynamic pressure
    dynamic_pressure = 0.5 * flue_gas_density_kg_m3 * velocity ** 2

    # Friction loss (Darcy-Weisbach)
    friction_loss = friction_factor * (duct_length_m / diameter) * dynamic_pressure

    # Local losses (bends, dampers, etc.)
    local_loss = local_loss_coeff * dynamic_pressure

    total_loss = friction_loss + local_loss

    logger.debug(
        f"Draft loss: v={velocity:.1f} m/s, friction={friction_loss:.1f} Pa, "
        f"local={local_loss:.1f} Pa, total={total_loss:.1f} Pa"
    )

    return total_loss


def calculate_theoretical_draft(
    stack_effect_pa: float,
    draft_loss_pa: float
) -> float:
    """
    Calculate net theoretical draft available at furnace.

    The net draft is the stack effect minus all flow losses.

    Args:
        stack_effect_pa: Total stack effect in Pa (positive = upward).
        draft_loss_pa: Total draft losses in Pa (always positive).

    Returns:
        Net draft in Pascals (positive = vacuum at furnace).
    """
    net_draft = stack_effect_pa - draft_loss_pa
    return net_draft


def calculate_draft_velocity(
    draft_pressure_pa: float,
    gas_density_kg_m3: float = 0.5
) -> float:
    """
    Calculate theoretical gas velocity from draft pressure.

    Uses Bernoulli's equation:
        v = sqrt(2 * Delta_P / rho)

    Args:
        draft_pressure_pa: Draft pressure differential in Pa.
        gas_density_kg_m3: Flue gas density in kg/m^3.

    Returns:
        Gas velocity in m/s.
    """
    if draft_pressure_pa < 0:
        logger.warning(f"Negative draft pressure {draft_pressure_pa} Pa")
        return 0.0

    if gas_density_kg_m3 <= 0:
        raise ValueError(f"Density must be > 0, got {gas_density_kg_m3}")

    velocity = math.sqrt(2 * abs(draft_pressure_pa) / gas_density_kg_m3)
    return velocity


def calculate_flue_gas_density(
    temperature_c: float,
    molecular_weight: float = 28.97,
    pressure_pa: float = STANDARD_PRESSURE_PA
) -> float:
    """
    Calculate flue gas density using ideal gas law.

    Formula:
        rho = P * M / (R * T)

    Args:
        temperature_c: Gas temperature in Celsius.
        molecular_weight: Molecular weight of gas (28.97 for air, ~28 for flue gas).
        pressure_pa: Absolute pressure in Pascals.

    Returns:
        Gas density in kg/m^3.
    """
    temp_k = temperature_c + 273.15
    if temp_k <= 0:
        raise ValueError("Temperature must be > -273.15C")

    # Universal gas constant R = 8.314 J/(mol*K)
    # rho = P * M / (R * T) where M is in kg/mol
    density = pressure_pa * (molecular_weight / 1000) / (8.314 * temp_k)

    return density
