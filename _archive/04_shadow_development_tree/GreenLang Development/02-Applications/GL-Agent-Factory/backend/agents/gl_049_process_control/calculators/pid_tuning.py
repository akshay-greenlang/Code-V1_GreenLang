"""
PID Tuning Calculator

Physics-based calculations for PID controller tuning using
classical and modern tuning methods.

References:
    - ISA-5.1: Instrumentation Symbols and Identification
    - ISA-95: Enterprise-Control System Integration
    - Ziegler-Nichols (1942) Classic tuning rules
    - Cohen-Coon (1953) Process reaction curve method
"""

import math
import logging
from typing import Dict, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class TuningMethod(str, Enum):
    """PID tuning methods."""
    ZIEGLER_NICHOLS_OPEN = "ZIEGLER_NICHOLS_OPEN"
    ZIEGLER_NICHOLS_CLOSED = "ZIEGLER_NICHOLS_CLOSED"
    COHEN_COON = "COHEN_COON"
    IMC = "IMC"  # Internal Model Control
    LAMBDA = "LAMBDA"


class ControllerType(str, Enum):
    """Controller types."""
    P = "P"
    PI = "PI"
    PID = "PID"


def calculate_pid_parameters(
    process_gain: float,
    time_constant: float,
    dead_time: float,
    controller_type: ControllerType = ControllerType.PID,
    tuning_method: TuningMethod = TuningMethod.ZIEGLER_NICHOLS_OPEN,
    aggressiveness: float = 1.0
) -> Dict[str, float]:
    """
    Calculate PID parameters based on process characteristics.

    Uses first-order plus dead-time (FOPDT) model:
        G(s) = K * exp(-L*s) / (tau*s + 1)

    Args:
        process_gain: Process gain K (output/input change).
        time_constant: Process time constant tau (seconds).
        dead_time: Process dead time/delay L (seconds).
        controller_type: P, PI, or PID.
        tuning_method: Tuning method to use.
        aggressiveness: Tuning aggressiveness factor (0.5-2.0).

    Returns:
        Dictionary with Kp, Ki, Kd parameters.

    Example:
        >>> params = calculate_pid_parameters(1.5, 60, 10, ControllerType.PID)
        >>> print(f"Kp={params['kp']:.2f}, Ki={params['ki']:.4f}")
    """
    if process_gain == 0:
        raise ValueError("Process gain cannot be zero")
    if time_constant <= 0:
        raise ValueError("Time constant must be positive")
    if dead_time < 0:
        raise ValueError("Dead time cannot be negative")

    # Select tuning method
    if tuning_method == TuningMethod.ZIEGLER_NICHOLS_OPEN:
        params = tune_ziegler_nichols(
            process_gain, time_constant, dead_time, controller_type, closed_loop=False
        )
    elif tuning_method == TuningMethod.ZIEGLER_NICHOLS_CLOSED:
        params = tune_ziegler_nichols(
            process_gain, time_constant, dead_time, controller_type, closed_loop=True
        )
    elif tuning_method == TuningMethod.COHEN_COON:
        params = tune_cohen_coon(
            process_gain, time_constant, dead_time, controller_type
        )
    elif tuning_method == TuningMethod.IMC:
        params = tune_imc(
            process_gain, time_constant, dead_time, controller_type
        )
    elif tuning_method == TuningMethod.LAMBDA:
        params = tune_lambda(
            process_gain, time_constant, dead_time, controller_type
        )
    else:
        params = tune_ziegler_nichols(
            process_gain, time_constant, dead_time, controller_type, closed_loop=False
        )

    # Apply aggressiveness factor
    params["kp"] *= aggressiveness
    params["ki"] *= aggressiveness
    params["kd"] *= aggressiveness

    # Round values
    params["kp"] = round(params["kp"], 4)
    params["ki"] = round(params["ki"], 6)
    params["kd"] = round(params["kd"], 4)

    logger.debug(
        f"PID tuning ({tuning_method.value}): Kp={params['kp']}, "
        f"Ki={params['ki']}, Kd={params['kd']}"
    )

    return params


def tune_ziegler_nichols(
    k: float,
    tau: float,
    l: float,
    controller_type: ControllerType,
    closed_loop: bool = False
) -> Dict[str, float]:
    """
    Ziegler-Nichols tuning method.

    Open-loop method uses process reaction curve.
    Closed-loop method uses ultimate gain and period.

    Open-loop tuning rules:
        P:   Kp = tau / (K * L)
        PI:  Kp = 0.9 * tau / (K * L), Ti = 3.33 * L
        PID: Kp = 1.2 * tau / (K * L), Ti = 2 * L, Td = 0.5 * L

    Args:
        k: Process gain.
        tau: Time constant.
        l: Dead time.
        controller_type: P, PI, or PID.
        closed_loop: Use closed-loop method.

    Returns:
        Dictionary with kp, ki, kd values.
    """
    if l <= 0:
        l = tau * 0.1  # Estimate small dead time

    if closed_loop:
        # Estimate ultimate gain and period from FOPDT model
        # Ku ~ 4 * tau / (pi * K * L)
        # Pu ~ 4 * L
        ku = 4 * tau / (math.pi * k * l) if k != 0 and l != 0 else 1
        pu = 4 * l

        if controller_type == ControllerType.P:
            kp = 0.5 * ku
            ti = float('inf')
            td = 0
        elif controller_type == ControllerType.PI:
            kp = 0.45 * ku
            ti = pu / 1.2
            td = 0
        else:  # PID
            kp = 0.6 * ku
            ti = pu / 2
            td = pu / 8
    else:
        # Open-loop method
        if controller_type == ControllerType.P:
            kp = tau / (k * l) if k != 0 and l != 0 else 1
            ti = float('inf')
            td = 0
        elif controller_type == ControllerType.PI:
            kp = 0.9 * tau / (k * l) if k != 0 and l != 0 else 1
            ti = 3.33 * l
            td = 0
        else:  # PID
            kp = 1.2 * tau / (k * l) if k != 0 and l != 0 else 1
            ti = 2 * l
            td = 0.5 * l

    # Convert Ti, Td to Ki, Kd
    # Ki = Kp / Ti (integral gain)
    # Kd = Kp * Td (derivative gain)
    ki = kp / ti if ti > 0 and ti != float('inf') else 0
    kd = kp * td

    return {"kp": kp, "ki": ki, "kd": kd, "ti": ti, "td": td}


def tune_cohen_coon(
    k: float,
    tau: float,
    l: float,
    controller_type: ControllerType
) -> Dict[str, float]:
    """
    Cohen-Coon tuning method.

    More conservative than Z-N for processes with significant dead time.

    Args:
        k: Process gain.
        tau: Time constant.
        l: Dead time.
        controller_type: P, PI, or PID.

    Returns:
        Dictionary with kp, ki, kd values.
    """
    if l <= 0:
        l = tau * 0.1

    # Normalized dead time
    r = l / tau if tau > 0 else 0.5

    if controller_type == ControllerType.P:
        kp = (1/k) * (tau/l) * (1 + r/3)
        ti = float('inf')
        td = 0
    elif controller_type == ControllerType.PI:
        kp = (1/k) * (tau/l) * (0.9 + r/12)
        ti = l * (30 + 3*r) / (9 + 20*r)
        td = 0
    else:  # PID
        kp = (1/k) * (tau/l) * (4/3 + r/4)
        ti = l * (32 + 6*r) / (13 + 8*r)
        td = l * 4 / (11 + 2*r)

    ki = kp / ti if ti > 0 and ti != float('inf') else 0
    kd = kp * td

    return {"kp": kp, "ki": ki, "kd": kd, "ti": ti, "td": td}


def tune_imc(
    k: float,
    tau: float,
    l: float,
    controller_type: ControllerType,
    lambda_factor: float = 1.0
) -> Dict[str, float]:
    """
    Internal Model Control (IMC) tuning method.

    Provides good robustness and allows trade-off between
    performance and robustness via lambda parameter.

    Args:
        k: Process gain.
        tau: Time constant.
        l: Dead time.
        controller_type: PI or PID (P not typical for IMC).
        lambda_factor: Closed-loop time constant multiplier.

    Returns:
        Dictionary with kp, ki, kd values.
    """
    # Desired closed-loop time constant
    # Lambda = filter time constant (typically lambda = max(0.25*tau, 0.8*L))
    lambda_cl = max(lambda_factor * tau, 0.8 * l)

    if controller_type == ControllerType.P:
        kp = tau / (k * (lambda_cl + l))
        ti = float('inf')
        td = 0
    elif controller_type == ControllerType.PI:
        kp = tau / (k * (lambda_cl + l))
        ti = tau
        td = 0
    else:  # PID
        kp = (2*tau + l) / (k * 2 * (lambda_cl + l))
        ti = tau + l/2
        td = tau * l / (2*tau + l)

    ki = kp / ti if ti > 0 and ti != float('inf') else 0
    kd = kp * td

    return {"kp": kp, "ki": ki, "kd": kd, "ti": ti, "td": td}


def tune_lambda(
    k: float,
    tau: float,
    l: float,
    controller_type: ControllerType,
    lambda_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Lambda tuning method.

    User specifies desired closed-loop response time (lambda).
    Good for processes requiring specific response characteristics.

    Args:
        k: Process gain.
        tau: Time constant.
        l: Dead time.
        controller_type: Controller type.
        lambda_value: Desired closed-loop time constant.

    Returns:
        Dictionary with kp, ki, kd values.
    """
    # Default lambda = max(3*L, tau)
    if lambda_value is None:
        lambda_value = max(3 * l, tau)

    if controller_type == ControllerType.PI:
        kp = tau / (k * (lambda_value + l))
        ti = tau
        td = 0
    else:  # PID
        kp = tau / (k * (lambda_value + l))
        ti = tau
        td = l / 2

    ki = kp / ti if ti > 0 else 0
    kd = kp * td

    return {"kp": kp, "ki": ki, "kd": kd, "ti": ti, "td": td, "lambda": lambda_value}


def calculate_process_response(
    process_gain: float,
    time_constant: float,
    dead_time: float,
    step_size: float = 1.0,
    time_span: float = 300.0,
    dt: float = 1.0
) -> Tuple[List[float], List[float]]:
    """
    Calculate process step response for FOPDT model.

    G(s) = K * exp(-L*s) / (tau*s + 1)

    Args:
        process_gain: Process gain K.
        time_constant: Time constant tau (seconds).
        dead_time: Dead time L (seconds).
        step_size: Input step magnitude.
        time_span: Total simulation time.
        dt: Time step.

    Returns:
        Tuple of (time_array, response_array).
    """
    times = []
    response = []

    t = 0
    while t <= time_span:
        times.append(t)

        if t < dead_time:
            # Before dead time, output is zero
            y = 0
        else:
            # First-order response after dead time
            t_effective = t - dead_time
            y = process_gain * step_size * (1 - math.exp(-t_effective / time_constant))

        response.append(y)
        t += dt

    return times, response


def estimate_fopdt_parameters(
    times: List[float],
    responses: List[float],
    step_size: float
) -> Dict[str, float]:
    """
    Estimate FOPDT model parameters from step response data.

    Uses two-point method for parameter estimation.

    Args:
        times: Time values.
        responses: Response values.
        step_size: Input step size.

    Returns:
        Dictionary with K, tau, L estimates.
    """
    if len(times) != len(responses) or len(times) < 10:
        raise ValueError("Insufficient data for parameter estimation")

    # Find steady-state value (process gain)
    y_ss = responses[-1]
    k = y_ss / step_size if step_size != 0 else 1

    # Normalize response
    y_norm = [y / y_ss if y_ss != 0 else 0 for y in responses]

    # Find t_28 (28.3% of final value) and t_63 (63.2%)
    t_28 = None
    t_63 = None

    for t, y in zip(times, y_norm):
        if t_28 is None and y >= 0.283:
            t_28 = t
        if t_63 is None and y >= 0.632:
            t_63 = t
            break

    if t_28 is None or t_63 is None:
        # Fallback to simple estimation
        tau = (times[-1] - times[0]) / 5
        l = tau / 5
    else:
        # Two-point estimation
        tau = 1.5 * (t_63 - t_28)
        l = t_63 - tau

    return {
        "process_gain": round(k, 4),
        "time_constant": round(max(1, tau), 2),
        "dead_time": round(max(0, l), 2),
    }
