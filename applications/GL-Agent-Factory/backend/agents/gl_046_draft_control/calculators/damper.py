"""
Damper Optimization Calculator

Physics-based calculations for furnace damper positioning and control.
Implements control valve equations adapted for damper applications.

References:
    - ISA-75.01: Control Valve Sizing
    - ASHRAE Handbook: Damper Characteristics
    - API 560: Fired Heater Draft Control
"""

import math
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_optimal_damper_position(
    required_draft_pa: float,
    available_draft_pa: float,
    current_position_pct: float,
    damper_characteristic: str = "equal_percentage"
) -> float:
    """
    Calculate optimal damper position for required draft.

    Dampers follow control valve characteristics:
    - Linear: Flow proportional to position
    - Equal percentage: Flow proportional to exponential of position
    - Quick opening: High gain at low positions

    For draft control, equal percentage is preferred for stability.

    Formula (equal percentage):
        Position = sqrt(required / available) * 100

    Args:
        required_draft_pa: Required draft pressure in Pa (positive value).
        available_draft_pa: Available draft from stack effect in Pa.
        current_position_pct: Current damper position (0-100%).
        damper_characteristic: Type of damper - "linear", "equal_percentage", "quick_opening".

    Returns:
        Optimal damper position in percent (0-100).

    Example:
        >>> optimal = calculate_optimal_damper_position(25, 100, 50, "equal_percentage")
        >>> print(f"Optimal position: {optimal:.1f}%")
        Optimal position: 50.0%
    """
    if available_draft_pa <= 0:
        logger.warning(f"No available draft ({available_draft_pa} Pa)")
        return current_position_pct

    if required_draft_pa < 0:
        logger.warning(f"Negative required draft ({required_draft_pa} Pa)")
        return max(0, current_position_pct - 10)  # Close damper

    # Calculate ratio
    ratio = abs(required_draft_pa) / abs(available_draft_pa)

    # Apply characteristic curve
    if damper_characteristic == "linear":
        # Linear: position = ratio
        optimal = ratio * 100

    elif damper_characteristic == "equal_percentage":
        # Equal percentage: position = sqrt(ratio) for flow relationship
        # Draft ~ flow^2, so position = sqrt(ratio)
        optimal = math.sqrt(ratio) * 100

    elif damper_characteristic == "quick_opening":
        # Quick opening: position = ratio^2
        optimal = (ratio ** 0.5) * 100

    else:
        # Default to equal percentage
        optimal = math.sqrt(ratio) * 100

    # Clamp to valid range
    optimal = max(0.0, min(100.0, optimal))

    logger.debug(
        f"Damper optimization: required={required_draft_pa:.1f} Pa, "
        f"available={available_draft_pa:.1f} Pa, optimal={optimal:.1f}%"
    )

    return round(optimal, 1)


def calculate_damper_cv(
    damper_area_m2: float,
    position_pct: float,
    characteristic: str = "equal_percentage",
    rangeability: float = 50.0
) -> float:
    """
    Calculate damper Cv (flow coefficient) at given position.

    The Cv represents the flow capacity of the damper.
    Equal percentage characteristic provides better control at low flows.

    Formula (equal percentage):
        Cv = Cv_max * R^(x-1)

    where:
        R = rangeability (typically 50:1)
        x = position/100

    Args:
        damper_area_m2: Damper free area when fully open.
        position_pct: Damper position (0-100%).
        characteristic: Damper type ("linear" or "equal_percentage").
        rangeability: Rangeability ratio (Cv_max/Cv_min).

    Returns:
        Effective Cv in m^2.

    Example:
        >>> cv = calculate_damper_cv(1.0, 50, "equal_percentage", 50)
        >>> print(f"Cv at 50%: {cv:.3f} m^2")
    """
    if position_pct <= 0:
        return 0.0

    # Normalize position
    x = min(1.0, position_pct / 100.0)

    # Maximum Cv (fully open) - assume Cv ~= 0.6 * Area for typical damper
    cv_max = 0.6 * damper_area_m2

    if characteristic == "linear":
        cv = cv_max * x
    else:  # equal_percentage
        # Cv = Cv_max * R^(x-1)
        cv = cv_max * (rangeability ** (x - 1))

    return cv


def calculate_pressure_drop_damper(
    flow_rate_m3_s: float,
    damper_cv: float,
    gas_density_kg_m3: float = 0.5
) -> float:
    """
    Calculate pressure drop across damper at given flow.

    Uses modified control valve equation:
        Q = Cv * sqrt(Delta_P / rho)
        Delta_P = rho * (Q / Cv)^2

    Args:
        flow_rate_m3_s: Volumetric flow rate in m^3/s.
        damper_cv: Damper Cv at current position.
        gas_density_kg_m3: Gas density in kg/m^3.

    Returns:
        Pressure drop in Pascals.
    """
    if damper_cv <= 0:
        logger.warning("Damper Cv is zero - infinite pressure drop")
        return float('inf')

    if flow_rate_m3_s <= 0:
        return 0.0

    delta_p = gas_density_kg_m3 * (flow_rate_m3_s / damper_cv) ** 2
    return delta_p


def calculate_damper_authority(
    damper_dp_design: float,
    system_dp_design: float
) -> float:
    """
    Calculate damper authority (N).

    Authority indicates how much control the damper has over system flow.
    Higher authority (>0.5) provides more linear control response.

    Formula:
        N = Delta_P_damper / (Delta_P_damper + Delta_P_system)

    Args:
        damper_dp_design: Design pressure drop across damper in Pa.
        system_dp_design: Design pressure drop across rest of system in Pa.

    Returns:
        Authority value between 0 and 1.

    Example:
        >>> authority = calculate_damper_authority(100, 100)
        >>> print(f"Authority: {authority:.2f}")
        Authority: 0.50
    """
    total_dp = damper_dp_design + system_dp_design

    if total_dp <= 0:
        return 0.0

    authority = damper_dp_design / total_dp

    # Warn if authority is too low
    if authority < 0.3:
        logger.warning(
            f"Damper authority {authority:.2f} is low - control may be poor"
        )

    return authority


def calculate_damper_position_for_flow(
    target_flow_pct: float,
    damper_characteristic: str = "equal_percentage",
    rangeability: float = 50.0
) -> float:
    """
    Calculate damper position needed for target flow percentage.

    Inverse of the flow characteristic curve.

    Args:
        target_flow_pct: Desired flow as percentage of maximum (0-100).
        damper_characteristic: "linear" or "equal_percentage".
        rangeability: Rangeability ratio.

    Returns:
        Required damper position in percent.
    """
    if target_flow_pct <= 0:
        return 0.0
    if target_flow_pct >= 100:
        return 100.0

    flow_ratio = target_flow_pct / 100.0

    if damper_characteristic == "linear":
        position = flow_ratio * 100

    else:  # equal_percentage
        # Invert: x = 1 + log_R(flow_ratio)
        # x = 1 + ln(flow_ratio) / ln(R)
        if flow_ratio < 1 / rangeability:
            position = 0.0
        else:
            position = (1 + math.log(flow_ratio) / math.log(rangeability)) * 100

    return max(0.0, min(100.0, position))


def calculate_leakage_rate(
    damper_area_m2: float,
    pressure_diff_pa: float,
    leakage_class: str = "class_1"
) -> float:
    """
    Calculate damper leakage rate when closed.

    Per ASHRAE standards, dampers are classified by leakage:
    - Class 1: < 4 cfm/ft^2 at 1" WC
    - Class 2: < 10 cfm/ft^2 at 1" WC
    - Class 3: < 40 cfm/ft^2 at 1" WC

    Args:
        damper_area_m2: Damper blade area in m^2.
        pressure_diff_pa: Pressure differential in Pa.
        leakage_class: Damper leakage class.

    Returns:
        Leakage flow rate in m^3/s.
    """
    # Convert area to ft^2
    area_ft2 = damper_area_m2 * 10.764

    # Leakage coefficients (cfm/ft^2 at 1" WC)
    leakage_rates = {
        "class_1": 4.0,
        "class_2": 10.0,
        "class_3": 40.0,
    }

    base_rate = leakage_rates.get(leakage_class, 10.0)

    # Scale with pressure (leakage ~ sqrt(dP))
    # 1" WC = 249 Pa
    pressure_ratio = math.sqrt(abs(pressure_diff_pa) / 249.0) if pressure_diff_pa else 0

    # Calculate leakage in cfm then convert to m^3/s
    leakage_cfm = base_rate * area_ft2 * pressure_ratio
    leakage_m3_s = leakage_cfm * 0.000472  # cfm to m^3/s

    return leakage_m3_s
