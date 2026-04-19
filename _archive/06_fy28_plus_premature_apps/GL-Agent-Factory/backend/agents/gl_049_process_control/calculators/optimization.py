"""
Process Control Optimization Calculator

Physics-based calculations for setpoint optimization, cascade control,
and loop interaction analysis.

References:
    - ISA-95: Enterprise-Control System Integration
    - ISA-88: Batch Control
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_setpoint_optimization(
    current_setpoint: float,
    process_values: List[float],
    energy_input: List[float],
    target_efficiency: float = 95.0,
    constraints: Optional[Dict[str, float]] = None
) -> Dict[str, any]:
    """
    Calculate optimized setpoint for improved efficiency.

    Analyzes the relationship between setpoint and energy consumption
    to find optimal operating point.

    Args:
        current_setpoint: Current setpoint value.
        process_values: Historical process values.
        energy_input: Corresponding energy input values.
        target_efficiency: Target efficiency percentage.
        constraints: Min/max constraints {"min": float, "max": float}.

    Returns:
        Dictionary with optimized setpoint and analysis.

    Example:
        >>> opt = calculate_setpoint_optimization(
        ...     100, [98, 99, 100, 101, 102], [10, 9.5, 9.2, 9.5, 10]
        ... )
    """
    if len(process_values) != len(energy_input) or len(process_values) < 3:
        return {
            "optimized_setpoint": current_setpoint,
            "error": "Insufficient data"
        }

    # Find minimum energy point
    min_energy_idx = energy_input.index(min(energy_input))
    optimal_pv = process_values[min_energy_idx]

    # Calculate energy-efficiency curve slope
    # dE/dPV at current operating point
    n = len(process_values)
    mean_pv = sum(process_values) / n
    mean_e = sum(energy_input) / n

    # Linear regression for sensitivity
    numerator = sum((pv - mean_pv) * (e - mean_e) for pv, e in zip(process_values, energy_input))
    denominator = sum((pv - mean_pv)**2 for pv in process_values)

    if denominator > 0:
        slope = numerator / denominator  # dE/dPV
    else:
        slope = 0

    # Calculate potential savings
    if slope != 0:
        # Moving setpoint toward optimal
        sp_adjustment = optimal_pv - current_setpoint
        energy_change = slope * sp_adjustment
        energy_savings_pct = abs(energy_change) / mean_e * 100 if mean_e > 0 else 0
    else:
        sp_adjustment = 0
        energy_savings_pct = 0

    # Apply constraints
    optimized_sp = optimal_pv
    if constraints:
        if "min" in constraints:
            optimized_sp = max(constraints["min"], optimized_sp)
        if "max" in constraints:
            optimized_sp = min(constraints["max"], optimized_sp)

    return {
        "current_setpoint": current_setpoint,
        "optimized_setpoint": round(optimized_sp, 2),
        "setpoint_change": round(optimized_sp - current_setpoint, 2),
        "energy_sensitivity": round(slope, 4),
        "potential_energy_savings_pct": round(energy_savings_pct, 2),
        "optimal_process_value": round(optimal_pv, 2),
        "mean_energy_input": round(mean_e, 2),
    }


def calculate_cascade_benefit(
    primary_pv: List[float],
    secondary_pv: List[float],
    primary_sp: float,
    disturbance_times: Optional[List[int]] = None
) -> Dict[str, any]:
    """
    Calculate potential benefit of cascade control configuration.

    Cascade control can improve disturbance rejection by using
    a secondary (slave) loop to handle inner dynamics.

    Args:
        primary_pv: Primary process variable readings.
        secondary_pv: Secondary (inner) process variable readings.
        primary_sp: Primary setpoint.
        disturbance_times: Sample indices where disturbances occurred.

    Returns:
        Dictionary with cascade benefit analysis.
    """
    if len(primary_pv) != len(secondary_pv) or len(primary_pv) < 10:
        return {"cascade_recommended": False, "error": "Insufficient data"}

    # Calculate correlation between secondary and primary
    n = len(primary_pv)
    mean_p = sum(primary_pv) / n
    mean_s = sum(secondary_pv) / n

    # Covariance
    cov = sum((p - mean_p) * (s - mean_s) for p, s in zip(primary_pv, secondary_pv)) / n

    # Standard deviations
    std_p = math.sqrt(sum((p - mean_p)**2 for p in primary_pv) / n)
    std_s = math.sqrt(sum((s - mean_s)**2 for s in secondary_pv) / n)

    # Correlation coefficient
    if std_p > 0 and std_s > 0:
        correlation = cov / (std_p * std_s)
    else:
        correlation = 0

    # Lag analysis (simplified cross-correlation at different lags)
    max_lag = min(20, n // 4)
    best_lag = 0
    best_corr = abs(correlation)

    for lag in range(1, max_lag):
        lagged_cov = sum(
            (primary_pv[i + lag] - mean_p) * (secondary_pv[i] - mean_s)
            for i in range(n - lag)
        ) / (n - lag)
        lagged_corr = abs(lagged_cov / (std_p * std_s)) if std_p > 0 and std_s > 0 else 0

        if lagged_corr > best_corr:
            best_corr = lagged_corr
            best_lag = lag

    # Variability ratio
    # Higher secondary variability relative to primary suggests cascade benefit
    variability_ratio = std_s / std_p if std_p > 0 else 1

    # Cascade is beneficial when:
    # 1. High correlation (disturbances propagate)
    # 2. Significant lag (secondary responds first)
    # 3. Higher secondary variability

    cascade_score = (
        0.4 * abs(correlation) +
        0.3 * min(1, best_lag / 10) +
        0.3 * min(1, variability_ratio)
    ) * 100

    cascade_recommended = cascade_score > 50 and best_lag > 0

    # Estimate improvement
    if cascade_recommended:
        estimated_improvement = min(50, cascade_score * 0.5)
    else:
        estimated_improvement = 0

    return {
        "cascade_recommended": cascade_recommended,
        "cascade_score": round(cascade_score, 1),
        "correlation": round(correlation, 3),
        "optimal_lag_samples": best_lag,
        "variability_ratio": round(variability_ratio, 3),
        "primary_std": round(std_p, 4),
        "secondary_std": round(std_s, 4),
        "estimated_improvement_pct": round(estimated_improvement, 1),
    }


def calculate_loop_interaction(
    loop1_output: List[float],
    loop1_pv: List[float],
    loop2_output: List[float],
    loop2_pv: List[float]
) -> Dict[str, any]:
    """
    Calculate interaction between two control loops.

    Loop interaction occurs when the output of one loop affects
    the process variable of another. High interaction may require
    decoupling control.

    Args:
        loop1_output: Output from loop 1.
        loop1_pv: Process variable for loop 1.
        loop2_output: Output from loop 2.
        loop2_pv: Process variable for loop 2.

    Returns:
        Dictionary with interaction analysis.
    """
    if not (len(loop1_output) == len(loop1_pv) == len(loop2_output) == len(loop2_pv)):
        return {"error": "Data length mismatch"}

    n = len(loop1_output)
    if n < 10:
        return {"error": "Insufficient data"}

    # Calculate gain matrix elements (relative gain)
    # K11 = correlation(out1, pv1)
    # K12 = correlation(out1, pv2)  <- interaction
    # K21 = correlation(out2, pv1)  <- interaction
    # K22 = correlation(out2, pv2)

    def calc_correlation(x, y):
        mx, my = sum(x)/n, sum(y)/n
        cov = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y)) / n
        sx = math.sqrt(sum((xi-mx)**2 for xi in x) / n)
        sy = math.sqrt(sum((yi-my)**2 for yi in y) / n)
        return cov / (sx * sy) if sx > 0 and sy > 0 else 0

    k11 = calc_correlation(loop1_output, loop1_pv)
    k12 = calc_correlation(loop1_output, loop2_pv)
    k21 = calc_correlation(loop2_output, loop1_pv)
    k22 = calc_correlation(loop2_output, loop2_pv)

    # Relative Gain Array (RGA) for 2x2 system
    # lambda_11 = K11*K22 / (K11*K22 - K12*K21)
    det = k11*k22 - k12*k21
    if abs(det) > 0.01:
        lambda_11 = k11 * k22 / det
        lambda_12 = -k12 * k21 / det
    else:
        lambda_11 = 1
        lambda_12 = 0

    # Interaction assessment
    # lambda near 1 = low interaction
    # lambda < 0.5 or > 2 = high interaction
    interaction_level = abs(1 - lambda_11)

    if interaction_level < 0.2:
        interaction = "LOW"
        decoupling_needed = False
    elif interaction_level < 0.5:
        interaction = "MODERATE"
        decoupling_needed = False
    else:
        interaction = "HIGH"
        decoupling_needed = True

    return {
        "gain_matrix": {
            "k11": round(k11, 3),
            "k12": round(k12, 3),
            "k21": round(k21, 3),
            "k22": round(k22, 3),
        },
        "rga_lambda_11": round(lambda_11, 3),
        "rga_lambda_12": round(lambda_12, 3),
        "interaction_level": interaction,
        "interaction_index": round(interaction_level, 3),
        "decoupling_recommended": decoupling_needed,
        "pairing_recommendation": "1-1, 2-2" if lambda_11 > 0.5 else "1-2, 2-1",
    }


def calculate_valve_performance(
    output_values: List[float],
    position_feedback: List[float],
    sample_time: float = 1.0
) -> Dict[str, any]:
    """
    Calculate control valve performance metrics.

    Analyzes valve response, stiction, and hysteresis.

    Args:
        output_values: Controller output commands (%).
        position_feedback: Actual valve position feedback (%).
        sample_time: Time between samples.

    Returns:
        Dictionary with valve performance metrics.
    """
    if len(output_values) != len(position_feedback) or len(output_values) < 10:
        return {"error": "Insufficient data"}

    n = len(output_values)

    # Position error
    errors = [out - pos for out, pos in zip(output_values, position_feedback)]
    mean_error = sum(errors) / n
    std_error = math.sqrt(sum((e - mean_error)**2 for e in errors) / n)

    # Total valve travel
    travel = sum(abs(output_values[i+1] - output_values[i]) for i in range(n-1))

    # Reversals (direction changes)
    reversals = 0
    for i in range(1, n-1):
        d1 = output_values[i] - output_values[i-1]
        d2 = output_values[i+1] - output_values[i]
        if d1 * d2 < 0:  # Sign change
            reversals += 1

    # Stiction detection (valve doesn't move with small commands)
    small_moves = 0
    stuck_moves = 0
    for i in range(n-1):
        cmd_change = abs(output_values[i+1] - output_values[i])
        pos_change = abs(position_feedback[i+1] - position_feedback[i])
        if 0.5 < cmd_change < 5:  # Small command change
            small_moves += 1
            if pos_change < 0.1:  # Valve didn't move
                stuck_moves += 1

    stiction_index = stuck_moves / small_moves * 100 if small_moves > 0 else 0

    # Performance classification
    if std_error > 5 or stiction_index > 30:
        performance = "POOR"
    elif std_error > 2 or stiction_index > 15:
        performance = "FAIR"
    elif std_error > 1 or stiction_index > 5:
        performance = "GOOD"
    else:
        performance = "EXCELLENT"

    return {
        "mean_position_error_pct": round(mean_error, 2),
        "std_position_error_pct": round(std_error, 2),
        "total_travel_pct": round(travel, 1),
        "travel_per_hour": round(travel / (n * sample_time / 3600), 1),
        "reversal_count": reversals,
        "reversal_rate_per_min": round(reversals / (n * sample_time / 60), 2),
        "stiction_index_pct": round(stiction_index, 1),
        "valve_performance": performance,
    }
