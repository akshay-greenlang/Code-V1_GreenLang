"""
Fan/Pump Affinity Laws Calculator

Physics-based calculations using the fundamental affinity laws
for centrifugal fans and pumps.

The affinity laws relate flow, head, and power to rotational speed:
    - Flow is proportional to speed (Q ~ N)
    - Head is proportional to speed squared (H ~ N^2)
    - Power is proportional to speed cubed (P ~ N^3)

References:
    - ASHRAE Handbook: Fundamentals
    - Hydraulic Institute Standards
    - IEC 61800-9: Energy Efficiency of VFD Systems
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_flow_at_speed(
    rated_flow: float,
    rated_speed_rpm: float,
    actual_speed_rpm: float
) -> float:
    """
    Calculate flow rate at a different speed using affinity law.

    First Affinity Law:
        Q1/Q2 = N1/N2
        Q2 = Q1 * (N2/N1)

    Args:
        rated_flow: Flow at rated speed (m^3/s, m^3/h, GPM, etc.).
        rated_speed_rpm: Rated motor speed in RPM.
        actual_speed_rpm: Actual operating speed in RPM.

    Returns:
        Flow at actual speed (same units as rated_flow).

    Example:
        >>> flow = calculate_flow_at_speed(100, 1800, 1200)
        >>> print(f"Flow at 1200 RPM: {flow:.1f}")
        Flow at 1200 RPM: 66.7
    """
    if rated_speed_rpm <= 0:
        raise ValueError("Rated speed must be > 0")
    if actual_speed_rpm < 0:
        raise ValueError("Actual speed must be >= 0")

    speed_ratio = actual_speed_rpm / rated_speed_rpm
    flow = rated_flow * speed_ratio

    logger.debug(
        f"Flow calculation: Q_rated={rated_flow}, N_ratio={speed_ratio:.3f}, "
        f"Q_actual={flow:.2f}"
    )

    return flow


def calculate_head_at_speed(
    rated_head: float,
    rated_speed_rpm: float,
    actual_speed_rpm: float
) -> float:
    """
    Calculate head/pressure at a different speed using affinity law.

    Second Affinity Law:
        H1/H2 = (N1/N2)^2
        H2 = H1 * (N2/N1)^2

    Args:
        rated_head: Head at rated speed (m, ft, kPa, etc.).
        rated_speed_rpm: Rated motor speed in RPM.
        actual_speed_rpm: Actual operating speed in RPM.

    Returns:
        Head at actual speed (same units as rated_head).

    Example:
        >>> head = calculate_head_at_speed(50, 1800, 1200)
        >>> print(f"Head at 1200 RPM: {head:.1f}")
        Head at 1200 RPM: 22.2
    """
    if rated_speed_rpm <= 0:
        raise ValueError("Rated speed must be > 0")
    if actual_speed_rpm < 0:
        raise ValueError("Actual speed must be >= 0")

    speed_ratio = actual_speed_rpm / rated_speed_rpm
    head = rated_head * (speed_ratio ** 2)

    logger.debug(
        f"Head calculation: H_rated={rated_head}, N_ratio={speed_ratio:.3f}, "
        f"H_actual={head:.2f}"
    )

    return head


def calculate_power_at_speed(
    rated_power_kw: float,
    rated_speed_rpm: float,
    actual_speed_rpm: float,
    load_type: str = "variable_torque"
) -> float:
    """
    Calculate power at a different speed using affinity law.

    Third Affinity Law (for centrifugal loads):
        P1/P2 = (N1/N2)^3
        P2 = P1 * (N2/N1)^3

    For constant torque loads:
        P2 = P1 * (N2/N1)

    Args:
        rated_power_kw: Power at rated speed in kW.
        rated_speed_rpm: Rated motor speed in RPM.
        actual_speed_rpm: Actual operating speed in RPM.
        load_type: "variable_torque" (fans/pumps) or "constant_torque" (conveyors).

    Returns:
        Power at actual speed in kW.

    Example:
        >>> power = calculate_power_at_speed(75, 1800, 1200, "variable_torque")
        >>> print(f"Power at 1200 RPM: {power:.1f} kW")
        Power at 1200 RPM: 22.2 kW
    """
    if rated_speed_rpm <= 0:
        raise ValueError("Rated speed must be > 0")
    if actual_speed_rpm < 0:
        raise ValueError("Actual speed must be >= 0")

    speed_ratio = actual_speed_rpm / rated_speed_rpm

    if load_type == "constant_torque":
        # P ~ N (constant torque)
        exponent = 1
    elif load_type == "constant_power":
        # P constant (rare)
        exponent = 0
    else:
        # Variable torque (fans, centrifugal pumps): P ~ N^3
        exponent = 3

    power = rated_power_kw * (speed_ratio ** exponent)

    logger.debug(
        f"Power calculation: P_rated={rated_power_kw}kW, N_ratio={speed_ratio:.3f}, "
        f"exponent={exponent}, P_actual={power:.2f}kW"
    )

    return power


def calculate_speed_for_flow(
    target_flow: float,
    rated_flow: float,
    rated_speed_rpm: float,
    min_speed_pct: float = 20.0,
    max_speed_pct: float = 100.0
) -> Tuple[float, bool]:
    """
    Calculate required speed to achieve target flow.

    Inverse of first affinity law:
        N2 = N1 * (Q2/Q1)

    Args:
        target_flow: Desired flow rate.
        rated_flow: Flow at rated speed.
        rated_speed_rpm: Rated motor speed.
        min_speed_pct: Minimum allowed speed as % of rated.
        max_speed_pct: Maximum allowed speed as % of rated.

    Returns:
        Tuple of (required_speed_rpm, within_limits).

    Example:
        >>> speed, ok = calculate_speed_for_flow(70, 100, 1800, 30, 100)
        >>> print(f"Required speed: {speed:.0f} RPM, within limits: {ok}")
    """
    if rated_flow <= 0:
        raise ValueError("Rated flow must be > 0")

    flow_ratio = target_flow / rated_flow
    required_speed = rated_speed_rpm * flow_ratio

    min_speed = rated_speed_rpm * min_speed_pct / 100
    max_speed = rated_speed_rpm * max_speed_pct / 100

    within_limits = min_speed <= required_speed <= max_speed

    if required_speed < min_speed:
        logger.warning(
            f"Required speed {required_speed:.0f} RPM below minimum {min_speed:.0f}"
        )
    elif required_speed > max_speed:
        logger.warning(
            f"Required speed {required_speed:.0f} RPM above maximum {max_speed:.0f}"
        )

    return required_speed, within_limits


def calculate_speed_for_head(
    target_head: float,
    rated_head: float,
    rated_speed_rpm: float,
    min_speed_pct: float = 20.0,
    max_speed_pct: float = 100.0
) -> Tuple[float, bool]:
    """
    Calculate required speed to achieve target head/pressure.

    Inverse of second affinity law:
        N2 = N1 * sqrt(H2/H1)

    Args:
        target_head: Desired head/pressure.
        rated_head: Head at rated speed.
        rated_speed_rpm: Rated motor speed.
        min_speed_pct: Minimum allowed speed as % of rated.
        max_speed_pct: Maximum allowed speed as % of rated.

    Returns:
        Tuple of (required_speed_rpm, within_limits).
    """
    if rated_head <= 0:
        raise ValueError("Rated head must be > 0")
    if target_head < 0:
        raise ValueError("Target head must be >= 0")

    head_ratio = target_head / rated_head
    required_speed = rated_speed_rpm * math.sqrt(head_ratio)

    min_speed = rated_speed_rpm * min_speed_pct / 100
    max_speed = rated_speed_rpm * max_speed_pct / 100

    within_limits = min_speed <= required_speed <= max_speed

    return required_speed, within_limits


def calculate_system_curve_intersection(
    pump_curve: Dict[float, float],
    system_curve_k: float,
    static_head: float = 0
) -> Tuple[float, float]:
    """
    Find operating point where pump curve intersects system curve.

    Pump curve: H = f(Q) from manufacturer
    System curve: H = H_static + K * Q^2

    Args:
        pump_curve: Dictionary {flow: head} points from pump curve.
        system_curve_k: System resistance coefficient.
        static_head: Static head in system.

    Returns:
        Tuple of (flow_at_intersection, head_at_intersection).
    """
    # Simple iteration to find intersection
    best_flow = 0
    best_diff = float('inf')

    flows = sorted(pump_curve.keys())

    for q in flows:
        pump_head = pump_curve[q]
        system_head = static_head + system_curve_k * q**2

        diff = abs(pump_head - system_head)
        if diff < best_diff:
            best_diff = diff
            best_flow = q
            best_head = (pump_head + system_head) / 2

    # Refine with interpolation if needed
    return best_flow, best_head


def calculate_specific_speed(
    flow_m3_s: float,
    head_m: float,
    speed_rpm: float
) -> float:
    """
    Calculate pump/fan specific speed (dimensionless).

    Specific speed characterizes the pump/fan type:
        Ns = N * sqrt(Q) / H^0.75

    Common ranges:
        - Radial flow: 10-80
        - Mixed flow: 80-160
        - Axial flow: 160-500

    Args:
        flow_m3_s: Flow rate in m^3/s.
        head_m: Head in meters.
        speed_rpm: Rotational speed in RPM.

    Returns:
        Dimensionless specific speed.
    """
    if head_m <= 0:
        raise ValueError("Head must be > 0")
    if flow_m3_s <= 0:
        return 0

    ns = speed_rpm * math.sqrt(flow_m3_s) / (head_m ** 0.75)

    logger.debug(f"Specific speed: Q={flow_m3_s}, H={head_m}, N={speed_rpm}, Ns={ns:.1f}")

    return ns
