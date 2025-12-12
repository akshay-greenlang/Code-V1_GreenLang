"""
Control Loop Analysis Calculator

Physics-based calculations for control loop performance analysis
and diagnostics.

References:
    - ISA-95: Enterprise-Control System Integration
    - ISA TR75.25.02: Control Loop Performance Monitoring
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class LoopPerformance(str, Enum):
    """Control loop performance levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


def analyze_control_loop(
    setpoint: float,
    process_values: List[float],
    output_values: List[float],
    sample_time: float = 1.0
) -> Dict[str, any]:
    """
    Comprehensive control loop analysis.

    Calculates key performance metrics:
    - Error statistics (mean, std, min, max)
    - Integral errors (IAE, ISE, ITAE)
    - Settling characteristics
    - Output activity (valve travel)
    - Performance classification

    Args:
        setpoint: Control setpoint value.
        process_values: List of process variable readings.
        output_values: List of controller output values.
        sample_time: Time between samples in seconds.

    Returns:
        Dictionary with comprehensive analysis results.

    Example:
        >>> pv = [100, 99, 99.5, 100.2, 100.1, 100.0]
        >>> out = [50, 52, 51, 49, 50, 50]
        >>> analysis = analyze_control_loop(100, pv, out, 1.0)
    """
    if len(process_values) < 2:
        return {"error": "Insufficient data"}

    n = len(process_values)

    # Calculate errors
    errors = [pv - setpoint for pv in process_values]
    abs_errors = [abs(e) for e in errors]

    # Basic statistics
    mean_error = sum(errors) / n
    mean_abs_error = sum(abs_errors) / n
    std_error = math.sqrt(sum((e - mean_error)**2 for e in errors) / n)
    max_error = max(abs_errors)
    min_pv = min(process_values)
    max_pv = max(process_values)

    # Integral errors
    iae = calculate_integral_absolute_error(errors, sample_time)
    ise = sum(e**2 * sample_time for e in errors)
    itae = sum(abs(e) * (i * sample_time) for i, e in enumerate(errors))

    # Settling analysis
    settling_info = calculate_settling_time(
        process_values, setpoint, sample_time
    )

    # Overshoot
    overshoot_pct = calculate_overshoot(process_values, setpoint)

    # Output (valve) activity
    if output_values:
        output_changes = [
            abs(output_values[i+1] - output_values[i])
            for i in range(len(output_values) - 1)
        ]
        total_travel = sum(output_changes)
        avg_travel_rate = sum(output_changes) / len(output_changes) if output_changes else 0
        output_std = math.sqrt(
            sum((v - sum(output_values)/len(output_values))**2 for v in output_values) / len(output_values)
        )
    else:
        total_travel = 0
        avg_travel_rate = 0
        output_std = 0

    # Performance classification
    performance = classify_performance(
        mean_abs_error / abs(setpoint) * 100 if setpoint != 0 else mean_abs_error,
        std_error / abs(setpoint) * 100 if setpoint != 0 else std_error,
        overshoot_pct
    )

    # Control index (0-100, higher is better)
    control_index = calculate_control_index(
        mean_abs_error, std_error, setpoint, overshoot_pct, settling_info["settling_time"]
    )

    return {
        "setpoint": setpoint,
        "n_samples": n,
        "sample_time_s": sample_time,
        "duration_s": n * sample_time,
        # Error statistics
        "mean_error": round(mean_error, 4),
        "mean_absolute_error": round(mean_abs_error, 4),
        "std_error": round(std_error, 4),
        "max_error": round(max_error, 4),
        "error_band_pct": round((max_pv - min_pv) / abs(setpoint) * 100 if setpoint else 0, 2),
        # Integral errors
        "iae": round(iae, 2),
        "ise": round(ise, 2),
        "itae": round(itae, 2),
        # Settling
        "settling_time_s": settling_info["settling_time"],
        "settled": settling_info["settled"],
        "settling_band_pct": settling_info["band_pct"],
        # Overshoot
        "overshoot_pct": round(overshoot_pct, 2),
        # Output activity
        "total_output_travel_pct": round(total_travel, 2),
        "avg_output_change_rate": round(avg_travel_rate, 4),
        "output_std": round(output_std, 2),
        # Overall
        "performance": performance.value,
        "control_index": round(control_index, 1),
    }


def calculate_integral_absolute_error(
    errors: List[float],
    sample_time: float
) -> float:
    """
    Calculate Integral of Absolute Error (IAE).

    IAE = integral(|error| dt)

    Lower IAE indicates better control.

    Args:
        errors: List of error values.
        sample_time: Time between samples.

    Returns:
        IAE value.
    """
    return sum(abs(e) * sample_time for e in errors)


def calculate_settling_time(
    process_values: List[float],
    setpoint: float,
    sample_time: float,
    band_pct: float = 2.0
) -> Dict[str, any]:
    """
    Calculate settling time (time to stay within band).

    Settling time is when the process value enters and stays
    within the specified band around setpoint.

    Args:
        process_values: List of process values.
        setpoint: Target setpoint.
        sample_time: Time between samples.
        band_pct: Settling band as percentage of setpoint.

    Returns:
        Dictionary with settling time and status.
    """
    if not process_values or setpoint == 0:
        return {"settling_time": 0, "settled": False, "band_pct": band_pct}

    band = abs(setpoint) * band_pct / 100

    # Find last time value left the band, then settling time is after that
    settling_index = 0
    last_outside = -1

    for i, pv in enumerate(process_values):
        if abs(pv - setpoint) > band:
            last_outside = i

    if last_outside < len(process_values) - 1:
        settling_index = last_outside + 1
        settling_time = settling_index * sample_time
        settled = True
    else:
        settling_time = len(process_values) * sample_time
        settled = False

    return {
        "settling_time": round(settling_time, 2),
        "settled": settled,
        "band_pct": band_pct,
    }


def calculate_overshoot(
    process_values: List[float],
    setpoint: float,
    initial_value: Optional[float] = None
) -> float:
    """
    Calculate percentage overshoot.

    Overshoot = (peak - setpoint) / (setpoint - initial) * 100

    Args:
        process_values: List of process values.
        setpoint: Target setpoint.
        initial_value: Initial process value (uses first value if None).

    Returns:
        Overshoot as percentage.
    """
    if not process_values:
        return 0.0

    if initial_value is None:
        initial_value = process_values[0]

    step_change = setpoint - initial_value

    if step_change == 0:
        return 0.0

    # Determine if step up or step down
    if step_change > 0:
        peak = max(process_values)
        overshoot = (peak - setpoint) / step_change * 100
    else:
        peak = min(process_values)
        overshoot = (setpoint - peak) / abs(step_change) * 100

    return max(0, overshoot)


def classify_performance(
    error_pct: float,
    std_pct: float,
    overshoot_pct: float
) -> LoopPerformance:
    """
    Classify control loop performance.

    Args:
        error_pct: Mean absolute error as % of setpoint.
        std_pct: Standard deviation as % of setpoint.
        overshoot_pct: Overshoot percentage.

    Returns:
        Performance classification.
    """
    # Scoring based on multiple criteria
    score = 100

    # Error penalty
    if error_pct > 5:
        score -= 30
    elif error_pct > 2:
        score -= 15
    elif error_pct > 1:
        score -= 5

    # Variability penalty
    if std_pct > 5:
        score -= 25
    elif std_pct > 2:
        score -= 10
    elif std_pct > 1:
        score -= 5

    # Overshoot penalty
    if overshoot_pct > 20:
        score -= 25
    elif overshoot_pct > 10:
        score -= 15
    elif overshoot_pct > 5:
        score -= 5

    if score >= 85:
        return LoopPerformance.EXCELLENT
    elif score >= 70:
        return LoopPerformance.GOOD
    elif score >= 50:
        return LoopPerformance.ACCEPTABLE
    elif score >= 30:
        return LoopPerformance.POOR
    else:
        return LoopPerformance.CRITICAL


def calculate_control_index(
    mae: float,
    std: float,
    setpoint: float,
    overshoot: float,
    settling_time: float,
    target_settling: float = 60.0
) -> float:
    """
    Calculate overall control index (0-100).

    Higher values indicate better control.

    Args:
        mae: Mean absolute error.
        std: Standard deviation of error.
        setpoint: Setpoint value.
        overshoot: Overshoot percentage.
        settling_time: Settling time in seconds.
        target_settling: Target settling time.

    Returns:
        Control index 0-100.
    """
    if setpoint == 0:
        return 50.0

    # Normalize metrics
    error_score = max(0, 100 - (mae / abs(setpoint) * 100 * 10))
    std_score = max(0, 100 - (std / abs(setpoint) * 100 * 10))
    overshoot_score = max(0, 100 - overshoot * 2)
    settling_score = max(0, 100 - (settling_time / target_settling - 1) * 50) if settling_time > 0 else 50

    # Weighted average
    index = (
        0.30 * error_score +
        0.25 * std_score +
        0.25 * overshoot_score +
        0.20 * settling_score
    )

    return max(0, min(100, index))


def detect_oscillation(
    process_values: List[float],
    sample_time: float
) -> Dict[str, any]:
    """
    Detect oscillation in process variable.

    Uses zero-crossing analysis to identify periodic behavior.

    Args:
        process_values: List of process values.
        sample_time: Time between samples.

    Returns:
        Dictionary with oscillation analysis.
    """
    if len(process_values) < 10:
        return {"oscillating": False, "period": 0}

    # Remove mean
    mean_pv = sum(process_values) / len(process_values)
    detrended = [pv - mean_pv for pv in process_values]

    # Count zero crossings
    crossings = 0
    for i in range(len(detrended) - 1):
        if detrended[i] * detrended[i+1] < 0:
            crossings += 1

    # Estimate period from crossings
    total_time = len(process_values) * sample_time
    if crossings > 2:
        # Period = 2 * time / crossings (half-periods per crossing)
        estimated_period = 2 * total_time / crossings
        oscillating = True
    else:
        estimated_period = 0
        oscillating = False

    # Amplitude (peak-to-peak / 2)
    amplitude = (max(process_values) - min(process_values)) / 2

    return {
        "oscillating": oscillating,
        "zero_crossings": crossings,
        "estimated_period_s": round(estimated_period, 2),
        "amplitude": round(amplitude, 4),
        "relative_amplitude_pct": round(amplitude / mean_pv * 100, 2) if mean_pv != 0 else 0,
    }
