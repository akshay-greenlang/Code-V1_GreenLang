"""
Draft Control Safety Calculations

Safety interlock calculations and checks for furnace draft control.
Implements NFPA 86 safety requirements for combustion air systems.

References:
    - NFPA 86: Standard for Ovens and Furnaces
    - NFPA 87: Standard for Fluid Heaters
    - FM Global 6.0: Hot Work and Welding
"""

import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyStatus(str, Enum):
    """Safety interlock status levels."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    ALARM = "ALARM"
    TRIP = "TRIP"


# NFPA 86 Draft Limits (typical values)
DRAFT_LIMITS = {
    "max_positive_inwc": 0.05,  # Maximum allowed positive pressure
    "min_draft_inwc": -0.5,    # Maximum vacuum (too much air infiltration)
    "target_range_min": -0.15,  # Optimal range minimum
    "target_range_max": -0.05,  # Optimal range maximum
    "critical_positive_inwc": 0.1,  # Emergency trip threshold
}


def check_draft_safety(
    furnace_pressure_inwc: float,
    stack_velocity_m_s: float = 0,
    combustion_air_flow_pct: float = 100,
    flame_detected: bool = True,
    purge_complete: bool = True
) -> Tuple[SafetyStatus, List[str]]:
    """
    Check furnace draft against safety limits per NFPA 86.

    NFPA 86 requires:
    - Negative pressure in furnace to prevent flue gas escape
    - Adequate combustion air supply
    - Proper purge completion before ignition
    - Stack flow monitoring

    Args:
        furnace_pressure_inwc: Furnace pressure in inches WC.
        stack_velocity_m_s: Flue gas velocity in stack (m/s).
        combustion_air_flow_pct: Combustion air flow as % of design.
        flame_detected: Flame detector status.
        purge_complete: Pre-ignition purge status.

    Returns:
        Tuple of (SafetyStatus, list of safety messages).

    Example:
        >>> status, messages = check_draft_safety(0.02, 5.0, 95, True, True)
        >>> print(f"Status: {status}, Messages: {messages}")
    """
    status = SafetyStatus.SAFE
    messages = []

    # Check positive pressure - CRITICAL SAFETY ISSUE
    if furnace_pressure_inwc >= DRAFT_LIMITS["critical_positive_inwc"]:
        status = SafetyStatus.TRIP
        messages.append(
            f"CRITICAL: Positive pressure {furnace_pressure_inwc:.3f} inWC exceeds trip limit - "
            "EMERGENCY SHUTDOWN required"
        )

    elif furnace_pressure_inwc >= DRAFT_LIMITS["max_positive_inwc"]:
        status = max(status, SafetyStatus.ALARM)
        messages.append(
            f"ALARM: Positive pressure {furnace_pressure_inwc:.3f} inWC - "
            "flue gas escape risk, operator action required"
        )

    elif furnace_pressure_inwc > DRAFT_LIMITS["target_range_max"]:
        status = max(status, SafetyStatus.WARNING)
        messages.append(
            f"WARNING: Pressure {furnace_pressure_inwc:.3f} inWC above target range"
        )

    # Check excessive negative pressure (air infiltration)
    if furnace_pressure_inwc < DRAFT_LIMITS["min_draft_inwc"]:
        status = max(status, SafetyStatus.WARNING)
        messages.append(
            f"WARNING: Excessive draft {furnace_pressure_inwc:.3f} inWC - "
            "air infiltration may affect combustion"
        )

    # Check combustion air flow
    if combustion_air_flow_pct < 90:
        status = max(status, SafetyStatus.WARNING)
        messages.append(
            f"WARNING: Low combustion air flow {combustion_air_flow_pct:.1f}% - "
            "check fan/damper"
        )
    if combustion_air_flow_pct < 70:
        status = max(status, SafetyStatus.ALARM)
        messages.append(
            f"ALARM: Critically low combustion air {combustion_air_flow_pct:.1f}%"
        )

    # Check stack velocity (indicates flow)
    if stack_velocity_m_s < 2.0 and flame_detected:
        status = max(status, SafetyStatus.WARNING)
        messages.append(
            f"WARNING: Low stack velocity {stack_velocity_m_s:.1f} m/s - "
            "check for blockage"
        )

    # Check purge status
    if not purge_complete and flame_detected:
        status = max(status, SafetyStatus.ALARM)
        messages.append(
            "ALARM: Flame detected without purge completion - "
            "verify safety interlock"
        )

    # No issues - optimal
    if status == SafetyStatus.SAFE:
        if (DRAFT_LIMITS["target_range_min"] <= furnace_pressure_inwc
                <= DRAFT_LIMITS["target_range_max"]):
            messages.append("Draft pressure within optimal range")
        else:
            messages.append("Draft pressure acceptable")

    logger.info(
        f"Draft safety check: status={status.value}, "
        f"pressure={furnace_pressure_inwc:.3f} inWC"
    )

    return status, messages


def calculate_flue_gas_velocity(
    flue_gas_flow_kg_s: float,
    stack_area_m2: float,
    gas_temperature_c: float,
    molecular_weight: float = 28.97
) -> float:
    """
    Calculate flue gas velocity in stack.

    Uses ideal gas law for density and continuity for velocity.

    Args:
        flue_gas_flow_kg_s: Mass flow rate in kg/s.
        stack_area_m2: Stack cross-sectional area in m^2.
        gas_temperature_c: Gas temperature in Celsius.
        molecular_weight: Gas molecular weight (28.97 for air).

    Returns:
        Gas velocity in m/s.
    """
    if stack_area_m2 <= 0:
        raise ValueError(f"Stack area must be > 0, got {stack_area_m2}")

    # Calculate density from ideal gas law
    temp_k = gas_temperature_c + 273.15
    pressure_pa = 101325  # Assume atmospheric
    density = pressure_pa * (molecular_weight / 1000) / (8.314 * temp_k)

    # Velocity = mass_flow / (density * area)
    velocity = flue_gas_flow_kg_s / (density * stack_area_m2)

    return velocity


def check_positive_pressure_risk(
    stack_effect_inwc: float,
    draft_loss_inwc: float,
    wind_effect_inwc: float = 0,
    safety_margin: float = 0.05
) -> Tuple[bool, float]:
    """
    Check if positive pressure condition is possible.

    Positive pressure can occur when:
    - Stack effect is too low (cold stack)
    - Excessive draft losses
    - Adverse wind conditions

    Args:
        stack_effect_inwc: Available stack effect in inWC.
        draft_loss_inwc: System draft losses in inWC.
        wind_effect_inwc: Wind-induced pressure (+ = helps, - = hurts).
        safety_margin: Minimum required draft margin in inWC.

    Returns:
        Tuple of (risk_exists, net_draft_available).
    """
    # Net available draft
    net_draft = stack_effect_inwc - draft_loss_inwc + wind_effect_inwc

    # Check if sufficient margin
    risk_exists = net_draft < safety_margin

    if risk_exists:
        logger.warning(
            f"Positive pressure risk: net_draft={net_draft:.3f} inWC, "
            f"margin={safety_margin:.3f} inWC"
        )

    return risk_exists, net_draft


def get_safety_interlock_status(
    pressure_switch_ok: bool,
    damper_position_confirmed: bool,
    combustion_air_proven: bool,
    stack_temp_ok: bool,
    flame_safety_ok: bool
) -> Dict[str, any]:
    """
    Get status of all draft-related safety interlocks.

    Per NFPA 86, the following interlocks must be satisfied:
    - Combustion air flow proven
    - Stack damper open confirmed
    - Pressure switch within limits
    - Stack temperature within limits

    Args:
        pressure_switch_ok: Furnace pressure switch status.
        damper_position_confirmed: Stack damper position confirmed.
        combustion_air_proven: Air flow switch proven.
        stack_temp_ok: Stack temperature within limits.
        flame_safety_ok: Flame safety system status.

    Returns:
        Dictionary with interlock status and permit-to-fire.
    """
    interlocks = {
        "pressure_switch": pressure_switch_ok,
        "damper_position": damper_position_confirmed,
        "combustion_air": combustion_air_proven,
        "stack_temperature": stack_temp_ok,
        "flame_safety": flame_safety_ok,
    }

    # All interlocks must be true for permit
    permit_to_fire = all(interlocks.values())

    # List failed interlocks
    failed = [name for name, status in interlocks.items() if not status]

    result = {
        "interlocks": interlocks,
        "permit_to_fire": permit_to_fire,
        "failed_interlocks": failed,
        "status": SafetyStatus.SAFE if permit_to_fire else SafetyStatus.ALARM,
    }

    if not permit_to_fire:
        logger.warning(f"Safety interlocks failed: {failed}")

    return result


def calculate_minimum_stack_height(
    design_draft_inwc: float,
    stack_temp_c: float,
    ambient_temp_c: float,
    safety_factor: float = 1.25
) -> float:
    """
    Calculate minimum stack height for required draft.

    Rearranging the stack effect equation:
        H = SE / (7.64 * (1/Ta - 1/Ts))

    Args:
        design_draft_inwc: Required draft in inWC.
        stack_temp_c: Expected stack temperature in C.
        ambient_temp_c: Design ambient temperature in C.
        safety_factor: Safety factor for height (typically 1.25).

    Returns:
        Minimum stack height in meters.
    """
    stack_k = stack_temp_c + 273.15
    ambient_k = ambient_temp_c + 273.15

    if stack_k <= ambient_k:
        raise ValueError("Stack temp must be > ambient for natural draft")

    temp_factor = (1 / ambient_k) - (1 / stack_k)

    if temp_factor <= 0:
        raise ValueError("Invalid temperature differential")

    # Minimum height
    min_height = abs(design_draft_inwc) / (7.64 * temp_factor)

    # Apply safety factor
    design_height = min_height * safety_factor

    logger.info(
        f"Stack height calculation: min={min_height:.1f}m, "
        f"design={design_height:.1f}m with SF={safety_factor}"
    )

    return design_height
