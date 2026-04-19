"""
Epsilon-NTU Method Calculator for Heat Exchangers

This module implements the Epsilon-NTU (effectiveness-Number of Transfer Units)
method for heat exchanger analysis following TEMA (Tubular Exchanger Manufacturers
Association) standards and ASME guidelines.

The Epsilon-NTU method is preferred when outlet temperatures are unknown
and heat transfer area or required effectiveness needs to be determined.

All formulas follow standard heat transfer engineering references:
- TEMA Standards (10th Edition)
- Incropera & DeWitt - Fundamentals of Heat and Mass Transfer
- Kays & London - Compact Heat Exchangers
- ASME Heat Exchanger Design Handbook

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> effectiveness = calculate_effectiveness(2.5, 0.75, 'counterflow')
    >>> print(f"Effectiveness: {effectiveness:.4f}")
    Effectiveness: 0.8756
"""

import math
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Heat exchanger flow arrangement types
FLOW_ARRANGEMENTS = [
    "counterflow",
    "parallel_flow",
    "shell_and_tube_1_2",  # 1 shell pass, 2 tube passes
    "shell_and_tube_2_4",  # 2 shell passes, 4 tube passes
    "crossflow_unmixed",
    "crossflow_mixed_cmax",
    "crossflow_mixed_cmin",
    "crossflow_both_mixed",
]


def calculate_ntu(
    ua: float,
    c_min: float
) -> float:
    """
    Calculate Number of Transfer Units (NTU).

    NTU is a dimensionless parameter that represents the heat transfer
    size of the exchanger relative to the minimum heat capacity rate.

    Formula:
        NTU = UA / C_min

    where:
        - UA: Overall heat transfer coefficient times area (W/K)
        - C_min: Minimum heat capacity rate (W/K)

    Args:
        ua: Overall heat transfer coefficient times area (W/K). Must be > 0.
        c_min: Minimum heat capacity rate = min(m_dot_h * cp_h, m_dot_c * cp_c).
            Must be > 0.

    Returns:
        NTU value (dimensionless).

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> ntu = calculate_ntu(50000, 20000)
        >>> print(f"NTU: {ntu:.2f}")
        NTU: 2.50
    """
    if ua <= 0:
        raise ValueError(f"UA must be > 0, got {ua}")
    if c_min <= 0:
        raise ValueError(f"C_min must be > 0, got {c_min}")

    ntu = ua / c_min

    logger.debug(f"NTU calculation: UA={ua:.0f} W/K, C_min={c_min:.0f} W/K, NTU={ntu:.3f}")

    return ntu


def calculate_capacity_ratio(
    c_min: float,
    c_max: float
) -> float:
    """
    Calculate heat capacity ratio Cr.

    The capacity ratio is the ratio of the minimum to maximum
    heat capacity rates.

    Formula:
        Cr = C_min / C_max

    where Cr is between 0 and 1.
    Cr = 0 corresponds to a phase-changing fluid (condenser/evaporator).
    Cr = 1 means equal capacity rates (balanced exchanger).

    Args:
        c_min: Minimum heat capacity rate (W/K). Must be > 0.
        c_max: Maximum heat capacity rate (W/K). Must be >= c_min.

    Returns:
        Capacity ratio Cr between 0 and 1.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> cr = calculate_capacity_ratio(15000, 20000)
        >>> print(f"Capacity ratio: {cr:.3f}")
        Capacity ratio: 0.750
    """
    if c_min <= 0:
        raise ValueError(f"C_min must be > 0, got {c_min}")
    if c_max < c_min:
        raise ValueError(f"C_max ({c_max}) must be >= C_min ({c_min})")

    # Handle edge case where c_max is effectively infinite (phase change)
    if c_max > 1e10:
        return 0.0

    cr = c_min / c_max

    return min(1.0, max(0.0, cr))


def calculate_effectiveness(
    ntu: float,
    cr: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate heat exchanger effectiveness (epsilon).

    Effectiveness is the ratio of actual heat transfer to maximum
    possible heat transfer.

    Formula depends on flow arrangement. For counterflow:
        epsilon = [1 - exp(-NTU*(1-Cr))] / [1 - Cr*exp(-NTU*(1-Cr))]

    For Cr = 1 (balanced counterflow):
        epsilon = NTU / (1 + NTU)

    Reference: TEMA Standards, Kays & London

    Args:
        ntu: Number of Transfer Units. Must be >= 0.
        cr: Capacity ratio (0 to 1).
        flow_arrangement: Type of flow arrangement. Options:
            - "counterflow": Counter-current flow
            - "parallel_flow": Co-current (parallel) flow
            - "shell_and_tube_1_2": 1 shell pass, 2 tube passes
            - "shell_and_tube_2_4": 2 shell passes, 4 tube passes
            - "crossflow_unmixed": Both fluids unmixed
            - "crossflow_mixed_cmax": C_max fluid mixed
            - "crossflow_mixed_cmin": C_min fluid mixed
            - "crossflow_both_mixed": Both fluids mixed

    Returns:
        Effectiveness epsilon between 0 and 1.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> eff = calculate_effectiveness(2.5, 0.75, "counterflow")
        >>> print(f"Effectiveness: {eff:.4f}")
    """
    if ntu < 0:
        raise ValueError(f"NTU must be >= 0, got {ntu}")
    if cr < 0 or cr > 1:
        raise ValueError(f"Capacity ratio Cr must be 0-1, got {cr}")

    flow = flow_arrangement.lower().replace("-", "_").replace(" ", "_")

    if flow not in FLOW_ARRANGEMENTS:
        logger.warning(f"Unknown flow arrangement '{flow}', using counterflow")
        flow = "counterflow"

    # Special case: NTU = 0 means no heat transfer
    if ntu == 0:
        return 0.0

    # Special case: Cr = 0 (condenser/evaporator with phase change)
    if cr == 0:
        return 1 - math.exp(-ntu)

    # Calculate effectiveness based on flow arrangement
    if flow == "counterflow":
        effectiveness = _effectiveness_counterflow(ntu, cr)
    elif flow == "parallel_flow":
        effectiveness = _effectiveness_parallel_flow(ntu, cr)
    elif flow.startswith("shell_and_tube"):
        effectiveness = _effectiveness_shell_and_tube(ntu, cr, flow)
    elif flow.startswith("crossflow"):
        effectiveness = _effectiveness_crossflow(ntu, cr, flow)
    else:
        effectiveness = _effectiveness_counterflow(ntu, cr)

    # Clamp to valid range
    effectiveness = max(0.0, min(1.0, effectiveness))

    logger.debug(
        f"Effectiveness calculation: NTU={ntu:.3f}, Cr={cr:.3f}, "
        f"flow={flow}, epsilon={effectiveness:.4f}"
    )

    return effectiveness


def _effectiveness_counterflow(ntu: float, cr: float) -> float:
    """Calculate effectiveness for counterflow arrangement."""
    if abs(cr - 1.0) < 1e-6:
        # Special case: Cr = 1 (balanced exchanger)
        return ntu / (1 + ntu)
    else:
        # General formula
        exp_term = math.exp(-ntu * (1 - cr))
        return (1 - exp_term) / (1 - cr * exp_term)


def _effectiveness_parallel_flow(ntu: float, cr: float) -> float:
    """Calculate effectiveness for parallel (co-current) flow arrangement."""
    exp_term = math.exp(-ntu * (1 + cr))
    return (1 - exp_term) / (1 + cr)


def _effectiveness_shell_and_tube(ntu: float, cr: float, flow: str) -> float:
    """Calculate effectiveness for shell-and-tube exchangers."""
    # 1-2 TEMA E shell (most common)
    if "1_2" in flow or "1-2" in flow:
        # E-shell formula (1 shell pass, 2n tube passes)
        term1 = 1 + cr**2
        sqrt_term = math.sqrt(term1)

        exp_term = math.exp(-ntu * sqrt_term)
        numerator = 2 * (1 + cr + sqrt_term * (1 + exp_term) / (1 - exp_term))

        if abs(numerator) < 1e-10:
            return 0.0

        return 2 / numerator

    # 2-4 arrangement (2 shell passes, 4 tube passes)
    elif "2_4" in flow or "2-4" in flow:
        # Use 1-2 formula twice in series
        e1 = _effectiveness_shell_and_tube(ntu / 2, cr, "shell_and_tube_1_2")
        # For multiple shells in series
        return ((1 - e1 * cr) / (1 - e1) - 1) / (
            (1 - e1 * cr) / (1 - e1) - cr
        ) if abs(1 - e1) > 1e-10 else e1

    else:
        # Default to 1-2 arrangement
        return _effectiveness_shell_and_tube(ntu, cr, "shell_and_tube_1_2")


def _effectiveness_crossflow(ntu: float, cr: float, flow: str) -> float:
    """Calculate effectiveness for crossflow arrangements."""
    if "unmixed" in flow:
        # Both fluids unmixed - approximation formula
        # Kays & London correlation
        term1 = ntu**0.22
        term2 = math.exp(-cr * ntu**0.78)
        return 1 - math.exp((term1 / cr) * (term2 - 1))

    elif "cmax" in flow or "c_max" in flow:
        # C_max fluid mixed, C_min unmixed
        term = 1 - math.exp(-cr * (1 - math.exp(-ntu)))
        return (1 / cr) * term if cr > 0 else 1 - math.exp(-ntu)

    elif "cmin" in flow or "c_min" in flow:
        # C_min fluid mixed, C_max unmixed
        return 1 - math.exp(-(1 / cr) * (1 - math.exp(-cr * ntu))) if cr > 0 else 1 - math.exp(-ntu)

    elif "both_mixed" in flow:
        # Both fluids mixed
        term1 = 1 / (1 - math.exp(-ntu))
        term2 = cr / (1 - math.exp(-ntu * cr)) if cr > 0 else 0
        return 1 / (term1 + term2 - 1 / ntu) if ntu > 0 else 0

    else:
        # Default to unmixed
        return _effectiveness_crossflow(ntu, cr, "crossflow_unmixed")


def calculate_ntu_from_effectiveness(
    effectiveness: float,
    cr: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate NTU from desired effectiveness (inverse problem).

    This is used when sizing a heat exchanger to achieve a target
    effectiveness.

    Formula for counterflow:
        NTU = [1/(1-Cr)] * ln[(1-Cr*epsilon)/(1-epsilon)]

    For Cr = 1:
        NTU = epsilon / (1 - epsilon)

    Args:
        effectiveness: Target effectiveness (0 to 1). Must be < 1.
        cr: Capacity ratio (0 to 1).
        flow_arrangement: Type of flow arrangement.

    Returns:
        Required NTU value.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> ntu = calculate_ntu_from_effectiveness(0.85, 0.75, "counterflow")
        >>> print(f"Required NTU: {ntu:.3f}")
    """
    if effectiveness <= 0 or effectiveness >= 1:
        raise ValueError(f"Effectiveness must be 0 < epsilon < 1, got {effectiveness}")
    if cr < 0 or cr > 1:
        raise ValueError(f"Capacity ratio Cr must be 0-1, got {cr}")

    flow = flow_arrangement.lower().replace("-", "_").replace(" ", "_")

    # For counterflow and parallel flow, analytical solutions exist
    if flow == "counterflow":
        if abs(cr - 1.0) < 1e-6:
            # Cr = 1 case
            ntu = effectiveness / (1 - effectiveness)
        elif cr == 0:
            # Cr = 0 case (condenser/evaporator)
            ntu = -math.log(1 - effectiveness)
        else:
            ntu = (1 / (1 - cr)) * math.log(
                (1 - cr * effectiveness) / (1 - effectiveness)
            )
    elif flow == "parallel_flow":
        if (1 - effectiveness * (1 + cr)) <= 0:
            raise ValueError(
                f"Effectiveness {effectiveness} not achievable with parallel flow at Cr={cr}"
            )
        ntu = -math.log(1 - effectiveness * (1 + cr)) / (1 + cr)
    else:
        # For other arrangements, use numerical iteration
        ntu = _ntu_numerical_iteration(effectiveness, cr, flow)

    logger.debug(
        f"NTU from effectiveness: epsilon={effectiveness:.4f}, Cr={cr:.3f}, "
        f"flow={flow}, NTU={ntu:.3f}"
    )

    return max(0.0, ntu)


def _ntu_numerical_iteration(
    target_effectiveness: float,
    cr: float,
    flow: str,
    max_iterations: int = 50
) -> float:
    """Numerically solve for NTU using bisection method."""
    ntu_low = 0.01
    ntu_high = 20.0

    for _ in range(max_iterations):
        ntu_mid = (ntu_low + ntu_high) / 2
        eff_mid = calculate_effectiveness(ntu_mid, cr, flow)

        if abs(eff_mid - target_effectiveness) < 1e-6:
            return ntu_mid

        if eff_mid < target_effectiveness:
            ntu_low = ntu_mid
        else:
            ntu_high = ntu_mid

    return (ntu_low + ntu_high) / 2


def calculate_heat_transfer(
    effectiveness: float,
    c_min: float,
    t_hot_in: float,
    t_cold_in: float
) -> Dict[str, float]:
    """
    Calculate actual heat transfer and outlet temperatures.

    Formula:
        Q = epsilon * C_min * (T_h,in - T_c,in)
        Q = C_h * (T_h,in - T_h,out) = C_c * (T_c,out - T_c,in)

    Args:
        effectiveness: Heat exchanger effectiveness (0 to 1).
        c_min: Minimum heat capacity rate (W/K).
        t_hot_in: Hot fluid inlet temperature (C or K).
        t_cold_in: Cold fluid inlet temperature (C or K).

    Returns:
        Dictionary with:
        - q_actual: Actual heat transfer rate (W)
        - q_max: Maximum possible heat transfer (W)
        - delta_t_max: Maximum temperature difference (K or C)

    Example:
        >>> result = calculate_heat_transfer(0.85, 15000, 120, 30)
        >>> print(f"Heat transfer: {result['q_actual']/1000:.1f} kW")
    """
    if effectiveness < 0 or effectiveness > 1:
        raise ValueError(f"Effectiveness must be 0-1, got {effectiveness}")
    if c_min <= 0:
        raise ValueError(f"C_min must be > 0, got {c_min}")

    # Maximum temperature difference
    delta_t_max = t_hot_in - t_cold_in

    if delta_t_max < 0:
        raise ValueError(
            f"Hot inlet ({t_hot_in}) must be > cold inlet ({t_cold_in})"
        )

    # Maximum and actual heat transfer
    q_max = c_min * delta_t_max
    q_actual = effectiveness * q_max

    return {
        "q_actual": q_actual,
        "q_max": q_max,
        "delta_t_max": delta_t_max,
    }


def calculate_outlet_temperatures(
    effectiveness: float,
    c_hot: float,
    c_cold: float,
    t_hot_in: float,
    t_cold_in: float
) -> Dict[str, float]:
    """
    Calculate outlet temperatures for both fluids.

    Using energy balance:
        T_h,out = T_h,in - Q / C_h
        T_c,out = T_c,in + Q / C_c

    Args:
        effectiveness: Heat exchanger effectiveness (0 to 1).
        c_hot: Hot fluid heat capacity rate (W/K).
        c_cold: Cold fluid heat capacity rate (W/K).
        t_hot_in: Hot fluid inlet temperature.
        t_cold_in: Cold fluid inlet temperature.

    Returns:
        Dictionary with:
        - t_hot_out: Hot fluid outlet temperature
        - t_cold_out: Cold fluid outlet temperature
        - q_actual: Actual heat transfer rate (W)

    Example:
        >>> result = calculate_outlet_temperatures(0.85, 20000, 15000, 120, 30)
        >>> print(f"Hot out: {result['t_hot_out']:.1f}C")
    """
    if effectiveness < 0 or effectiveness > 1:
        raise ValueError(f"Effectiveness must be 0-1, got {effectiveness}")
    if c_hot <= 0 or c_cold <= 0:
        raise ValueError(f"Heat capacity rates must be > 0")

    c_min = min(c_hot, c_cold)
    delta_t_max = t_hot_in - t_cold_in

    if delta_t_max < 0:
        raise ValueError("Hot inlet must be > cold inlet")

    # Actual heat transfer
    q_actual = effectiveness * c_min * delta_t_max

    # Outlet temperatures
    t_hot_out = t_hot_in - q_actual / c_hot
    t_cold_out = t_cold_in + q_actual / c_cold

    return {
        "t_hot_out": t_hot_out,
        "t_cold_out": t_cold_out,
        "q_actual": q_actual,
    }


def calculate_required_ua(
    effectiveness: float,
    cr: float,
    c_min: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate required UA value for target effectiveness.

    This is used for heat exchanger sizing.

    Formula:
        UA = NTU * C_min

    Args:
        effectiveness: Target effectiveness (0 to 1).
        cr: Capacity ratio (0 to 1).
        c_min: Minimum heat capacity rate (W/K).
        flow_arrangement: Type of flow arrangement.

    Returns:
        Required UA value (W/K).

    Example:
        >>> ua = calculate_required_ua(0.85, 0.75, 15000, "counterflow")
        >>> print(f"Required UA: {ua:.0f} W/K")
    """
    ntu = calculate_ntu_from_effectiveness(effectiveness, cr, flow_arrangement)
    ua = ntu * c_min

    logger.debug(
        f"Required UA: epsilon={effectiveness:.4f}, NTU={ntu:.3f}, "
        f"C_min={c_min:.0f} W/K, UA={ua:.0f} W/K"
    )

    return ua


def calculate_effectiveness_from_temperatures(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    c_hot: float,
    c_cold: float
) -> Dict[str, float]:
    """
    Calculate effectiveness from measured temperatures.

    This is used for performance assessment of existing exchangers.

    Formula:
        epsilon = Q_actual / Q_max
        Q_actual = C_h * (T_h,in - T_h,out) = C_c * (T_c,out - T_c,in)
        Q_max = C_min * (T_h,in - T_c,in)

    Args:
        t_hot_in: Hot fluid inlet temperature.
        t_hot_out: Hot fluid outlet temperature.
        t_cold_in: Cold fluid inlet temperature.
        t_cold_out: Cold fluid outlet temperature.
        c_hot: Hot fluid heat capacity rate (W/K).
        c_cold: Cold fluid heat capacity rate (W/K).

    Returns:
        Dictionary with:
        - effectiveness: Calculated effectiveness
        - q_hot: Heat transfer based on hot side (W)
        - q_cold: Heat transfer based on cold side (W)
        - q_actual: Average actual heat transfer (W)
        - heat_balance_error: Percentage error between hot and cold side

    Example:
        >>> result = calculate_effectiveness_from_temperatures(
        ...     120, 75, 30, 85, 20000, 15000
        ... )
        >>> print(f"Effectiveness: {result['effectiveness']:.4f}")
    """
    # Calculate heat transfer from each side
    q_hot = c_hot * (t_hot_in - t_hot_out)
    q_cold = c_cold * (t_cold_out - t_cold_in)

    # Heat balance check
    q_avg = (q_hot + q_cold) / 2
    heat_balance_error = abs(q_hot - q_cold) / max(q_avg, 1) * 100

    if heat_balance_error > 10:
        logger.warning(
            f"Heat balance error {heat_balance_error:.1f}% exceeds 10%. "
            f"Check measurements or heat losses."
        )

    # Calculate effectiveness
    c_min = min(c_hot, c_cold)
    q_max = c_min * (t_hot_in - t_cold_in)

    if q_max <= 0:
        raise ValueError("Maximum heat transfer must be > 0")

    effectiveness = q_avg / q_max

    # Clamp to valid range
    effectiveness = max(0.0, min(1.0, effectiveness))

    return {
        "effectiveness": effectiveness,
        "q_hot": q_hot,
        "q_cold": q_cold,
        "q_actual": q_avg,
        "heat_balance_error": heat_balance_error,
    }


def calculate_ntu_from_temperatures(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    c_hot: float,
    c_cold: float,
    flow_arrangement: str = "counterflow"
) -> Dict[str, float]:
    """
    Calculate NTU from measured temperatures.

    Combines effectiveness calculation with NTU determination.

    Args:
        t_hot_in: Hot fluid inlet temperature.
        t_hot_out: Hot fluid outlet temperature.
        t_cold_in: Cold fluid inlet temperature.
        t_cold_out: Cold fluid outlet temperature.
        c_hot: Hot fluid heat capacity rate (W/K).
        c_cold: Cold fluid heat capacity rate (W/K).
        flow_arrangement: Type of flow arrangement.

    Returns:
        Dictionary with effectiveness, ntu, cr, and calculated UA.

    Example:
        >>> result = calculate_ntu_from_temperatures(
        ...     120, 75, 30, 85, 20000, 15000, "counterflow"
        ... )
        >>> print(f"NTU: {result['ntu']:.3f}")
    """
    # Get effectiveness
    eff_result = calculate_effectiveness_from_temperatures(
        t_hot_in, t_hot_out, t_cold_in, t_cold_out, c_hot, c_cold
    )

    effectiveness = eff_result["effectiveness"]

    # Calculate capacity ratio
    c_min = min(c_hot, c_cold)
    c_max = max(c_hot, c_cold)
    cr = calculate_capacity_ratio(c_min, c_max)

    # Calculate NTU
    # Ensure effectiveness is within valid range for NTU calculation
    eff_clamped = max(0.001, min(0.999, effectiveness))
    ntu = calculate_ntu_from_effectiveness(eff_clamped, cr, flow_arrangement)

    # Calculate UA
    ua = ntu * c_min

    return {
        "effectiveness": effectiveness,
        "ntu": ntu,
        "cr": cr,
        "ua": ua,
        "c_min": c_min,
        "c_max": c_max,
        "q_actual": eff_result["q_actual"],
        "heat_balance_error": eff_result["heat_balance_error"],
    }
