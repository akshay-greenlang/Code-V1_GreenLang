"""
Derived Variables Module for GL-004 BURNMASTER

This module implements feature engineering calculations for combustion
optimization machine learning models. All calculations are deterministic
and auditable with complete provenance tracking.

Key Calculations:
- Normalized load
- Turndown ratio
- Constraint margins
- Stability features from time series

Reference Standards:
- Process control engineering best practices
- Statistical signal processing

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import hashlib
import json


@dataclass
class StabilityFeatures:
    """Combustion stability features extracted from signals."""
    flame_signal_mean: float
    flame_signal_std: float
    flame_signal_cv: float  # Coefficient of variation
    flame_signal_min: float
    flame_signal_max: float
    flame_signal_range: float

    o2_mean: float
    o2_std: float
    o2_cv: float
    o2_rate_of_change: float  # Average absolute rate of change

    flame_o2_correlation: float  # Cross-correlation
    stability_index: float  # Overall stability score 0-1

    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "flame_mean": str(self.flame_signal_mean),
            "o2_mean": str(self.o2_mean),
            "stability": str(self.stability_index)
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ConstraintMargins:
    """Operating constraint margins."""
    margins: Dict[str, float]  # Parameter -> margin (positive = within limits)
    violations: List[str]  # List of violated constraints
    closest_constraint: str  # Name of closest constraint
    closest_margin_pct: float  # Margin to closest constraint (%)
    overall_safety: float  # 0-1, 1 = all constraints satisfied with margin


# ============================================================================
# Load and Turndown Calculations
# ============================================================================

def compute_normalized_load(duty: float, design_duty: float) -> float:
    """
    Compute normalized load as fraction of design capacity.

    Args:
        duty: Current thermal duty (MW or any consistent unit)
        design_duty: Design/maximum thermal duty (same units)

    Returns:
        Normalized load (0-1+, can exceed 1 if over design)

    Raises:
        ValueError: If design_duty <= 0

    Physics:
        Normalized load = Current duty / Design duty
        Used for equipment performance curves and efficiency maps.

    Deterministic: YES
    """
    if design_duty <= 0:
        raise ValueError(f"Design duty must be positive, got {design_duty}")

    if duty < 0:
        raise ValueError(f"Duty must be non-negative, got {duty}")

    normalized = duty / design_duty

    return round(normalized, 4)


def compute_turndown_ratio(current_load: float, min_stable_load: float) -> float:
    """
    Compute turndown ratio relative to minimum stable load.

    Args:
        current_load: Current load (MW or fraction)
        min_stable_load: Minimum stable load (same units)

    Returns:
        Turndown ratio (>= 1 when above minimum)

    Raises:
        ValueError: If min_stable_load <= 0

    Physics:
        Turndown ratio = Current load / Minimum stable load
        Values < 1 indicate operation below minimum stable load (unstable).
        Typical fired heater turndown: 3:1 to 10:1 (max:min).

    Deterministic: YES
    """
    if min_stable_load <= 0:
        raise ValueError(f"Min stable load must be positive, got {min_stable_load}")

    if current_load < 0:
        raise ValueError(f"Current load must be non-negative, got {current_load}")

    turndown = current_load / min_stable_load

    return round(turndown, 4)


def compute_load_factor(
    actual_production: float,
    rated_capacity: float,
    time_period_hours: float
) -> float:
    """
    Compute load factor (capacity utilization).

    Args:
        actual_production: Actual energy/production in period
        rated_capacity: Rated capacity (power or rate)
        time_period_hours: Duration of period (hours)

    Returns:
        Load factor (0-1)

    Physics:
        Load factor = Actual production / (Capacity * Time)
        Measures how efficiently capacity is utilized over time.

    Deterministic: YES
    """
    if rated_capacity <= 0:
        raise ValueError(f"Rated capacity must be positive, got {rated_capacity}")

    if time_period_hours <= 0:
        raise ValueError(f"Time period must be positive, got {time_period_hours}")

    max_production = rated_capacity * time_period_hours
    load_factor = actual_production / max_production

    return round(min(1.0, max(0.0, load_factor)), 4)


# ============================================================================
# Constraint Margin Calculations
# ============================================================================

def compute_constraint_margins(
    operating_point: Dict[str, float],
    limits: Dict[str, Dict[str, float]]
) -> ConstraintMargins:
    """
    Compute margins to operating constraints.

    Args:
        operating_point: Dict of parameter -> current value
            Example: {"stack_o2": 3.0, "flue_temp": 250, "co_ppm": 50}
        limits: Dict of parameter -> {"min": value, "max": value}
            Example: {"stack_o2": {"min": 1.5, "max": 8.0}}

    Returns:
        ConstraintMargins with margin analysis

    Physics:
        Margin = (Value - Limit) / |Limit| * 100 for min constraints
        Margin = (Limit - Value) / |Limit| * 100 for max constraints
        Positive margin = within limits
        Negative margin = constraint violated

    Deterministic: YES
    """
    margins = {}
    violations = []
    closest_constraint = None
    closest_margin = float('inf')

    for param, value in operating_point.items():
        if param not in limits:
            continue

        param_limits = limits[param]

        # Check minimum constraint
        if "min" in param_limits:
            min_limit = param_limits["min"]
            if min_limit != 0:
                min_margin = (value - min_limit) / abs(min_limit) * 100
            else:
                min_margin = value * 100 if value >= 0 else -100

            margins[f"{param}_min"] = round(min_margin, 2)

            if min_margin < 0:
                violations.append(f"{param} below minimum ({value} < {min_limit})")

            if abs(min_margin) < abs(closest_margin):
                closest_margin = min_margin
                closest_constraint = f"{param}_min"

        # Check maximum constraint
        if "max" in param_limits:
            max_limit = param_limits["max"]
            if max_limit != 0:
                max_margin = (max_limit - value) / abs(max_limit) * 100
            else:
                max_margin = -value * 100 if value > 0 else 100

            margins[f"{param}_max"] = round(max_margin, 2)

            if max_margin < 0:
                violations.append(f"{param} above maximum ({value} > {max_limit})")

            if abs(max_margin) < abs(closest_margin):
                closest_margin = max_margin
                closest_constraint = f"{param}_max"

    # Overall safety score
    if not margins:
        overall_safety = 1.0
    else:
        min_margin = min(margins.values())
        if min_margin >= 10:
            overall_safety = 1.0
        elif min_margin >= 0:
            overall_safety = 0.5 + 0.5 * (min_margin / 10)
        else:
            overall_safety = max(0, 0.5 + min_margin / 20)

    return ConstraintMargins(
        margins=margins,
        violations=violations,
        closest_constraint=closest_constraint or "none",
        closest_margin_pct=round(closest_margin, 2) if closest_margin != float('inf') else 100.0,
        overall_safety=round(overall_safety, 3)
    )


# ============================================================================
# Stability Feature Calculations
# ============================================================================

def compute_stability_features(
    flame_signal: np.ndarray,
    o2_signal: np.ndarray,
    sample_rate_hz: float = 1.0
) -> StabilityFeatures:
    """
    Compute combustion stability features from time series data.

    Args:
        flame_signal: Array of flame detector readings (e.g., UV intensity)
        o2_signal: Array of O2 analyzer readings (%)
        sample_rate_hz: Sampling rate (Hz)

    Returns:
        StabilityFeatures with statistical measures

    Physics:
        Stable combustion:
        - Steady flame signal with low variance
        - Steady O2 with low rate of change
        - Negative correlation between flame and O2 (more flame -> less O2)

        Unstable combustion:
        - High flame variance (flickering)
        - Oscillating O2
        - O2 excursions outside normal range

    Deterministic: YES
    """
    if len(flame_signal) == 0 or len(o2_signal) == 0:
        raise ValueError("Signal arrays cannot be empty")

    if len(flame_signal) != len(o2_signal):
        raise ValueError("Signal arrays must have same length")

    warnings = []

    # Flame signal statistics
    flame_mean = float(np.mean(flame_signal))
    flame_std = float(np.std(flame_signal))
    flame_min = float(np.min(flame_signal))
    flame_max = float(np.max(flame_signal))
    flame_range = flame_max - flame_min
    flame_cv = flame_std / flame_mean if flame_mean != 0 else 0

    # O2 signal statistics
    o2_mean = float(np.mean(o2_signal))
    o2_std = float(np.std(o2_signal))
    o2_cv = o2_std / o2_mean if o2_mean != 0 else 0

    # O2 rate of change
    if len(o2_signal) > 1:
        o2_diff = np.diff(o2_signal)
        o2_rate = float(np.mean(np.abs(o2_diff))) * sample_rate_hz  # %/s
    else:
        o2_rate = 0.0

    # Cross-correlation between flame and O2
    if flame_std > 0 and o2_std > 0:
        correlation = float(np.corrcoef(flame_signal, o2_signal)[0, 1])
    else:
        correlation = 0.0

    # Stability index (0-1, higher is more stable)
    # Based on:
    # - Low flame CV (stable flame)
    # - Low O2 rate of change
    # - O2 within normal range (2-6%)
    # - Negative flame-O2 correlation (expected physics)

    flame_stability = max(0, 1 - flame_cv * 5)  # CV < 0.2 is good
    o2_stability = max(0, 1 - o2_rate * 2)  # Rate < 0.5 %/s is good
    o2_range_ok = 1.0 if 1.5 < o2_mean < 8.0 else 0.5
    correlation_score = max(0, 0.5 - correlation)  # Expect negative correlation

    stability_index = (
        0.3 * flame_stability +
        0.3 * o2_stability +
        0.2 * o2_range_ok +
        0.2 * correlation_score
    )

    # Generate warnings
    if flame_cv > 0.3:
        warnings.append("High flame variability - possible instability")
    if flame_min < 0.1 * flame_mean:
        warnings.append("Flame signal dropped significantly - check flame")
    if o2_mean < 1.5:
        warnings.append("O2 very low - risk of incomplete combustion")
    if o2_mean > 8.0:
        warnings.append("O2 very high - excessive excess air")
    if o2_rate > 1.0:
        warnings.append("High O2 rate of change - possible control instability")
    if correlation > 0.5:
        warnings.append("Unusual flame-O2 correlation - check sensors")

    return StabilityFeatures(
        flame_signal_mean=round(flame_mean, 4),
        flame_signal_std=round(flame_std, 4),
        flame_signal_cv=round(flame_cv, 4),
        flame_signal_min=round(flame_min, 4),
        flame_signal_max=round(flame_max, 4),
        flame_signal_range=round(flame_range, 4),
        o2_mean=round(o2_mean, 4),
        o2_std=round(o2_std, 4),
        o2_cv=round(o2_cv, 4),
        o2_rate_of_change=round(o2_rate, 4),
        flame_o2_correlation=round(correlation, 4),
        stability_index=round(stability_index, 4),
        warnings=warnings
    )


# ============================================================================
# Additional Derived Features
# ============================================================================

def compute_efficiency_deviation(
    current_efficiency: float,
    baseline_efficiency: float
) -> float:
    """
    Compute efficiency deviation from baseline.

    Args:
        current_efficiency: Current efficiency (%)
        baseline_efficiency: Baseline/design efficiency (%)

    Returns:
        Deviation in percentage points

    Deterministic: YES
    """
    if baseline_efficiency <= 0:
        raise ValueError(f"Baseline efficiency must be positive, got {baseline_efficiency}")

    deviation = current_efficiency - baseline_efficiency
    return round(deviation, 2)


def compute_emission_intensity(
    emission_rate: float,
    production_rate: float
) -> float:
    """
    Compute emission intensity (emissions per unit production).

    Args:
        emission_rate: Emission rate (kg/h CO2 or other)
        production_rate: Production rate (units/h)

    Returns:
        Emission intensity (kg/unit)

    Deterministic: YES
    """
    if production_rate <= 0:
        raise ValueError(f"Production rate must be positive, got {production_rate}")

    intensity = emission_rate / production_rate
    return round(intensity, 4)


def compute_heat_rate_deviation(
    current_heat_rate: float,
    design_heat_rate: float
) -> float:
    """
    Compute heat rate deviation from design.

    Args:
        current_heat_rate: Current heat rate (kJ/kWh or MJ/tonne)
        design_heat_rate: Design heat rate (same units)

    Returns:
        Deviation as percentage of design

    Deterministic: YES
    """
    if design_heat_rate <= 0:
        raise ValueError(f"Design heat rate must be positive, got {design_heat_rate}")

    deviation_pct = (current_heat_rate - design_heat_rate) / design_heat_rate * 100
    return round(deviation_pct, 2)


def compute_operating_envelope_distance(
    operating_point: Dict[str, float],
    envelope_center: Dict[str, float],
    envelope_scale: Dict[str, float]
) -> float:
    """
    Compute normalized distance from optimal operating envelope center.

    Args:
        operating_point: Current operating parameters
        envelope_center: Center of optimal envelope (same keys)
        envelope_scale: Scale factors for each parameter

    Returns:
        Normalized Euclidean distance (0 = at center)

    Physics:
        Distance = sqrt(sum((x - center) / scale)^2)
        Used for proximity to optimal operating conditions.

    Deterministic: YES
    """
    sum_sq = 0.0
    count = 0

    for param, value in operating_point.items():
        if param in envelope_center and param in envelope_scale:
            center = envelope_center[param]
            scale = envelope_scale[param]

            if scale > 0:
                normalized_diff = (value - center) / scale
                sum_sq += normalized_diff ** 2
                count += 1

    if count == 0:
        return 0.0

    distance = np.sqrt(sum_sq / count)
    return round(float(distance), 4)


def compute_trend_features(
    signal: np.ndarray,
    window_size: int = 10
) -> Dict[str, float]:
    """
    Compute trend features from a time series.

    Args:
        signal: Time series data
        window_size: Window for moving average

    Returns:
        Dict with trend features:
        - trend_slope: Linear regression slope
        - trend_r2: R-squared of linear fit
        - momentum: Recent change vs historical
        - volatility: Rolling standard deviation

    Deterministic: YES
    """
    if len(signal) < 3:
        return {
            "trend_slope": 0.0,
            "trend_r2": 0.0,
            "momentum": 0.0,
            "volatility": 0.0
        }

    # Linear regression for trend
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, 1)
    slope = coeffs[0]

    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((signal - y_pred) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Momentum (recent average vs overall average)
    recent = signal[-min(window_size, len(signal)):]
    momentum = np.mean(recent) - np.mean(signal)

    # Volatility (rolling std)
    if len(signal) >= window_size:
        volatility = float(np.std(signal[-window_size:]))
    else:
        volatility = float(np.std(signal))

    return {
        "trend_slope": round(float(slope), 6),
        "trend_r2": round(float(r2), 4),
        "momentum": round(float(momentum), 4),
        "volatility": round(float(volatility), 4)
    }


def compute_cross_correlation_lag(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: int = 10
) -> Tuple[int, float]:
    """
    Find lag at which two signals have maximum cross-correlation.

    Args:
        signal1: First time series
        signal2: Second time series
        max_lag: Maximum lag to check

    Returns:
        (optimal_lag, max_correlation)

    Physics:
        Useful for finding time delays in control loops
        (e.g., fuel change -> O2 response delay)

    Deterministic: YES
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have same length")

    if len(signal1) < max_lag + 1:
        max_lag = len(signal1) - 1

    best_lag = 0
    best_corr = -1

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s1 = signal1[-lag:]
            s2 = signal2[:lag]
        elif lag > 0:
            s1 = signal1[:-lag]
            s2 = signal2[lag:]
        else:
            s1 = signal1
            s2 = signal2

        if len(s1) > 0 and np.std(s1) > 0 and np.std(s2) > 0:
            corr = float(np.corrcoef(s1, s2)[0, 1])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

    return best_lag, round(best_corr, 4)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_operating_point(
    operating_point: Dict[str, float],
    expected_params: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate operating point has all expected parameters.

    Args:
        operating_point: Dict of parameter values
        expected_params: List of required parameters

    Returns:
        (is_valid, missing_params)

    Deterministic: YES
    """
    missing = [p for p in expected_params if p not in operating_point]
    return len(missing) == 0, missing


def validate_signal_quality(
    signal: np.ndarray,
    min_length: int = 10,
    max_nan_fraction: float = 0.1
) -> Tuple[bool, List[str]]:
    """
    Validate signal quality for feature extraction.

    Args:
        signal: Time series data
        min_length: Minimum required length
        max_nan_fraction: Maximum allowed NaN fraction

    Returns:
        (is_valid, issues)

    Deterministic: YES
    """
    issues = []

    if len(signal) < min_length:
        issues.append(f"Signal too short: {len(signal)} < {min_length}")

    nan_fraction = np.sum(np.isnan(signal)) / len(signal) if len(signal) > 0 else 0
    if nan_fraction > max_nan_fraction:
        issues.append(f"Too many NaNs: {nan_fraction:.1%} > {max_nan_fraction:.1%}")

    if np.std(signal[~np.isnan(signal)]) == 0:
        issues.append("Signal has zero variance")

    return len(issues) == 0, issues
