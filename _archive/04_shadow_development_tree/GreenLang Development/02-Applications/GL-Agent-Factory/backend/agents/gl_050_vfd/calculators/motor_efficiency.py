"""
Motor and VFD Efficiency Calculator

Physics-based calculations for motor efficiency, VFD efficiency,
and combined system efficiency at various operating points.

References:
    - IEC 60034-30: Efficiency Classes of AC Motors
    - NEMA MG1: Motors and Generators
    - IEC 61800-9: Energy Efficiency of VFD Systems
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Motor efficiency data by size and load (IE3 class typical)
# Format: {power_kw: {load_pct: efficiency_pct}}
MOTOR_EFFICIENCY_DATA = {
    7.5: {25: 82.5, 50: 89.0, 75: 91.0, 100: 91.0},
    15: {25: 85.0, 50: 90.5, 75: 92.0, 100: 92.5},
    30: {25: 87.0, 50: 91.5, 75: 93.0, 100: 93.5},
    55: {25: 88.5, 50: 92.5, 75: 94.0, 100: 94.5},
    75: {25: 89.0, 50: 93.0, 75: 94.5, 100: 95.0},
    110: {25: 89.5, 50: 93.5, 75: 94.8, 100: 95.2},
    160: {25: 90.0, 50: 94.0, 75: 95.0, 100: 95.5},
    200: {25: 90.5, 50: 94.2, 75: 95.2, 100: 95.6},
}


def calculate_motor_efficiency(
    rated_power_kw: float,
    load_pct: float,
    motor_class: str = "IE3",
    speed_pct: float = 100.0
) -> float:
    """
    Calculate motor efficiency at given load point.

    Motor efficiency varies with load:
    - Peak efficiency typically at 75-100% load
    - Efficiency drops significantly below 50% load
    - VFD operation may reduce efficiency 1-3% at full speed

    Args:
        rated_power_kw: Motor rated power in kW.
        load_pct: Current load as percentage of rated (0-100+).
        motor_class: Efficiency class (IE1, IE2, IE3, IE4).
        speed_pct: Operating speed as percentage of rated.

    Returns:
        Motor efficiency as percentage (0-100).

    Example:
        >>> eff = calculate_motor_efficiency(55, 75, "IE3")
        >>> print(f"Motor efficiency: {eff:.1f}%")
    """
    # Find closest motor size in data
    motor_sizes = sorted(MOTOR_EFFICIENCY_DATA.keys())
    closest_size = min(motor_sizes, key=lambda x: abs(x - rated_power_kw))

    eff_data = MOTOR_EFFICIENCY_DATA[closest_size]

    # Interpolate for load
    load_points = sorted(eff_data.keys())

    if load_pct <= load_points[0]:
        base_efficiency = eff_data[load_points[0]] * (load_pct / load_points[0])
    elif load_pct >= load_points[-1]:
        base_efficiency = eff_data[load_points[-1]]
    else:
        # Linear interpolation
        for i in range(len(load_points) - 1):
            if load_points[i] <= load_pct <= load_points[i + 1]:
                l1, l2 = load_points[i], load_points[i + 1]
                e1, e2 = eff_data[l1], eff_data[l2]
                frac = (load_pct - l1) / (l2 - l1)
                base_efficiency = e1 + frac * (e2 - e1)
                break
        else:
            base_efficiency = 90.0

    # Adjust for motor class
    class_adjustments = {
        "IE1": -3.0,
        "IE2": -1.5,
        "IE3": 0.0,
        "IE4": 1.5,
    }
    class_adj = class_adjustments.get(motor_class.upper(), 0)
    base_efficiency += class_adj

    # Adjust for speed (VFD operation reduces efficiency slightly at reduced speed)
    if speed_pct < 100:
        # Approximate: 0.5% reduction per 10% speed reduction below 50% speed
        speed_penalty = max(0, (50 - speed_pct) * 0.05)
        base_efficiency -= speed_penalty

    # Ensure valid range
    efficiency = max(50, min(99, base_efficiency))

    logger.debug(
        f"Motor efficiency: {rated_power_kw}kW, {load_pct}% load, "
        f"{motor_class}, {speed_pct}% speed = {efficiency:.1f}%"
    )

    return efficiency


def calculate_vfd_efficiency(
    rated_power_kw: float,
    load_pct: float,
    speed_pct: float = 100.0,
    vfd_type: str = "standard"
) -> float:
    """
    Calculate VFD (Variable Frequency Drive) efficiency.

    VFD efficiency depends on:
    - Load level
    - Output frequency (speed)
    - Drive size and topology

    Modern VFDs typically 95-98% efficient at full load/speed.

    Args:
        rated_power_kw: VFD rated power in kW.
        load_pct: Output load as percentage.
        speed_pct: Output speed/frequency as percentage.
        vfd_type: "standard", "regenerative", "multilevel".

    Returns:
        VFD efficiency as percentage.

    Example:
        >>> eff = calculate_vfd_efficiency(55, 75, 80)
        >>> print(f"VFD efficiency: {eff:.1f}%")
    """
    # Base efficiency at full load/speed
    if rated_power_kw < 10:
        base_efficiency = 95.0
    elif rated_power_kw < 50:
        base_efficiency = 96.5
    elif rated_power_kw < 200:
        base_efficiency = 97.5
    else:
        base_efficiency = 98.0

    # Type adjustment
    type_adj = {
        "standard": 0,
        "regenerative": -0.5,  # Slightly lower due to additional components
        "multilevel": 0.5,    # Better efficiency at partial load
    }
    base_efficiency += type_adj.get(vfd_type, 0)

    # Load derating
    # VFD efficiency drops at light loads due to fixed losses
    if load_pct < 25:
        load_penalty = (25 - load_pct) * 0.2
    elif load_pct < 50:
        load_penalty = (50 - load_pct) * 0.05
    else:
        load_penalty = 0

    # Speed/frequency penalty
    # Below ~30% speed, efficiency drops due to low output power
    if speed_pct < 30:
        speed_penalty = (30 - speed_pct) * 0.1
    else:
        speed_penalty = 0

    efficiency = base_efficiency - load_penalty - speed_penalty
    efficiency = max(80, min(99, efficiency))

    logger.debug(
        f"VFD efficiency: {rated_power_kw}kW, {load_pct}% load, "
        f"{speed_pct}% speed = {efficiency:.1f}%"
    )

    return efficiency


def calculate_system_efficiency(
    motor_efficiency_pct: float,
    vfd_efficiency_pct: float,
    mechanical_efficiency_pct: float = 95.0
) -> float:
    """
    Calculate overall system efficiency (VFD + motor + mechanical).

    System efficiency = Motor * VFD * Mechanical efficiency

    Args:
        motor_efficiency_pct: Motor efficiency percentage.
        vfd_efficiency_pct: VFD efficiency percentage.
        mechanical_efficiency_pct: Coupling/gearbox efficiency.

    Returns:
        Overall system efficiency percentage.
    """
    # Convert percentages to decimals, multiply, convert back
    system_eff = (
        (motor_efficiency_pct / 100) *
        (vfd_efficiency_pct / 100) *
        (mechanical_efficiency_pct / 100)
    ) * 100

    return round(system_eff, 2)


def calculate_power_factor(
    load_pct: float,
    has_vfd: bool = True,
    pf_correction: bool = False
) -> float:
    """
    Calculate system power factor.

    VFDs typically have poor input power factor at light loads
    unless equipped with active front end or power factor correction.

    Args:
        load_pct: System load percentage.
        has_vfd: Whether system has VFD.
        pf_correction: Whether VFD has power factor correction.

    Returns:
        Power factor (0-1).
    """
    if not has_vfd:
        # Direct-on-line motor power factor
        if load_pct >= 75:
            pf = 0.85
        elif load_pct >= 50:
            pf = 0.80
        elif load_pct >= 25:
            pf = 0.70
        else:
            pf = 0.50
    else:
        if pf_correction:
            # Active front end or LCL filter
            pf = 0.95 + (load_pct / 100) * 0.04  # 0.95-0.99
        else:
            # Standard 6-pulse drive
            if load_pct >= 75:
                pf = 0.95
            elif load_pct >= 50:
                pf = 0.92
            elif load_pct >= 25:
                pf = 0.85
            else:
                pf = 0.70

    return round(min(0.99, pf), 3)


def calculate_motor_losses(
    input_power_kw: float,
    efficiency_pct: float
) -> Dict[str, float]:
    """
    Calculate motor loss breakdown.

    Motor losses include:
    - Stator copper losses (I^2R)
    - Rotor copper losses
    - Core losses (iron losses)
    - Mechanical losses (friction, windage)
    - Stray load losses

    Args:
        input_power_kw: Motor input power.
        efficiency_pct: Motor efficiency.

    Returns:
        Dictionary with loss breakdown.
    """
    output_power = input_power_kw * efficiency_pct / 100
    total_losses = input_power_kw - output_power

    # Approximate loss distribution (typical for induction motor)
    losses = {
        "stator_copper_kw": total_losses * 0.35,
        "rotor_copper_kw": total_losses * 0.25,
        "core_losses_kw": total_losses * 0.20,
        "mechanical_losses_kw": total_losses * 0.10,
        "stray_load_losses_kw": total_losses * 0.10,
        "total_losses_kw": total_losses,
        "output_power_kw": output_power,
    }

    return {k: round(v, 3) for k, v in losses.items()}


def calculate_derating_factor(
    ambient_temp_c: float,
    altitude_m: float,
    base_temp_c: float = 40.0,
    base_altitude_m: float = 1000.0
) -> float:
    """
    Calculate motor/VFD derating factor for ambient conditions.

    Motors and VFDs are rated for standard conditions (typically 40C, 1000m).
    Higher temperature or altitude requires derating.

    Args:
        ambient_temp_c: Actual ambient temperature.
        altitude_m: Installation altitude in meters.
        base_temp_c: Standard rating temperature.
        base_altitude_m: Standard rating altitude.

    Returns:
        Derating factor (0-1, 1.0 = no derating needed).
    """
    # Temperature derating (1% per degree above base)
    if ambient_temp_c > base_temp_c:
        temp_derating = 1 - (ambient_temp_c - base_temp_c) * 0.01
    else:
        temp_derating = 1.0

    # Altitude derating (1% per 100m above base)
    if altitude_m > base_altitude_m:
        alt_derating = 1 - (altitude_m - base_altitude_m) / 100 * 0.01
    else:
        alt_derating = 1.0

    # Combined derating (use minimum)
    derating = min(temp_derating, alt_derating)
    derating = max(0.5, min(1.0, derating))

    return round(derating, 3)
