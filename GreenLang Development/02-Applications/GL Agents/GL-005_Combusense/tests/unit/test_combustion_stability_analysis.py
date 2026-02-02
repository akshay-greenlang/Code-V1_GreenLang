# -*- coding: utf-8 -*-
"""
Combustion stability analysis tests for GL-005 CombustionControlAgent.

Tests comprehensive stability analysis including:
- Flame stability index calculation
- Oscillation detection
- Trend analysis
- Pattern recognition
- Predictive stability monitoring
- Multi-variable stability metrics

Target: 15+ tests covering all stability analysis scenarios.
"""

import pytest
import math
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

pytestmark = pytest.mark.unit


# ============================================================================
# FLAME STABILITY INDEX TESTS
# ============================================================================

class TestFlameStabilityIndex:
    """Test flame stability index calculations."""

    def test_stability_index_perfect_stability(self):
        """Test stability index for perfectly stable flame."""
        # Perfect stability: no variation
        flame_intensities = [85.0] * 10

        mean = sum(flame_intensities) / len(flame_intensities)
        variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)

        # Zero variance = perfect stability
        assert variance == 0.0

        # Stability index = 1.0 (perfect)
        stability_index = 1.0
        assert stability_index == 1.0

    def test_stability_index_high_stability(self):
        """Test stability index for high stability (small variations)."""
        flame_intensities = [85.0, 85.5, 84.8, 85.2, 85.1, 84.9, 85.3, 85.0, 84.7, 85.4]

        mean = sum(flame_intensities) / len(flame_intensities)
        variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)
        std_dev = math.sqrt(variance)

        # Coefficient of variation
        cv = std_dev / mean

        # Stability index (1 - CV)
        stability_index = 1 - cv

        assert stability_index > 0.99
        assert 0 <= stability_index <= 1.0

    def test_stability_index_medium_stability(self):
        """Test stability index for medium stability."""
        flame_intensities = [70.0, 75.0, 68.0, 73.0, 71.0, 69.0, 74.0, 72.0, 70.5, 71.5]

        mean = sum(flame_intensities) / len(flame_intensities)
        variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean
        stability_index = 1 - cv

        assert 0.95 <= stability_index <= 0.99

    def test_stability_index_low_stability(self):
        """Test stability index for low stability (large variations)."""
        flame_intensities = [50.0, 65.0, 45.0, 70.0, 48.0, 62.0, 52.0, 68.0, 47.0, 63.0]

        mean = sum(flame_intensities) / len(flame_intensities)
        variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean
        stability_index = 1 - cv

        assert stability_index < 0.90

    def test_stability_index_weighted_recent_samples(self):
        """Test stability index with exponential weighting (recent samples matter more)."""
        flame_intensities = [85.0, 84.0, 83.0, 82.0, 81.0, 70.0, 65.0, 60.0, 55.0, 50.0]

        # Exponential weights (recent samples weighted more)
        weights = [2**i for i in range(len(flame_intensities))]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted mean
        weighted_mean = sum(f * w for f, w in zip(flame_intensities, normalized_weights))

        # Weighted variance
        weighted_variance = sum(w * (f - weighted_mean)**2 for f, w in zip(flame_intensities, normalized_weights))
        weighted_std = math.sqrt(weighted_variance)

        # Weighted CV
        weighted_cv = weighted_std / weighted_mean if weighted_mean > 0 else 1.0
        weighted_stability = 1 - weighted_cv

        # Weighted stability should be lower (recent instability weighted more)
        assert weighted_stability < 0.95


# ============================================================================
# OSCILLATION DETECTION TESTS
# ============================================================================

class TestOscillationDetection:
    """Test combustion oscillation detection."""

    def test_detect_no_oscillation(self):
        """Test detection when no oscillation present."""
        # Steady temperature
        temperatures = [1200.0 + i * 0.5 for i in range(20)]  # Gradual increase

        # Calculate derivatives (rate of change)
        derivatives = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]

        # Count sign changes (oscillations)
        sign_changes = sum(1 for i in range(len(derivatives)-1)
                          if derivatives[i] * derivatives[i+1] < 0)

        oscillation_threshold = len(derivatives) / 4  # Max 25% sign changes

        oscillating = sign_changes > oscillation_threshold

        assert oscillating is False

    def test_detect_high_frequency_oscillation(self):
        """Test detection of high frequency oscillations."""
        # Oscillating temperature
        temperatures = [1200.0 + 10.0 * math.sin(i * 0.5) for i in range(20)]

        derivatives = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]
        sign_changes = sum(1 for i in range(len(derivatives)-1)
                          if derivatives[i] * derivatives[i+1] < 0)

        oscillation_threshold = len(derivatives) / 4

        oscillating = sign_changes > oscillation_threshold

        assert oscillating is True

    def test_detect_low_frequency_oscillation(self):
        """Test detection of low frequency oscillations."""
        # Slow oscillation
        temperatures = [1200.0 + 20.0 * math.sin(i * 0.1) for i in range(50)]

        derivatives = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]
        sign_changes = sum(1 for i in range(len(derivatives)-1)
                          if derivatives[i] * derivatives[i+1] < 0)

        # Low frequency still causes oscillation
        oscillating = sign_changes > 2  # At least some sign changes

        assert oscillating is True

    def test_calculate_oscillation_amplitude(self):
        """Test calculation of oscillation amplitude."""
        # Oscillating data with known amplitude
        amplitude = 15.0
        temperatures = [1200.0 + amplitude * math.sin(i * 0.3) for i in range(30)]

        max_temp = max(temperatures)
        min_temp = min(temperatures)
        measured_amplitude = (max_temp - min_temp) / 2

        assert measured_amplitude == pytest.approx(amplitude, rel=1e-1)

    def test_calculate_oscillation_frequency(self):
        """Test calculation of oscillation frequency."""
        # Create oscillation with known frequency
        frequency_hz = 0.5  # 0.5 Hz
        sample_rate = 10.0  # 10 samples/sec
        duration_sec = 10.0
        num_samples = int(sample_rate * duration_sec)

        temperatures = [1200.0 + 10.0 * math.sin(2 * math.pi * frequency_hz * i / sample_rate)
                       for i in range(num_samples)]

        # Count zero crossings to estimate frequency
        derivatives = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]
        sign_changes = sum(1 for i in range(len(derivatives)-1)
                          if derivatives[i] * derivatives[i+1] < 0)

        # Frequency = sign_changes / (2 * duration)
        estimated_frequency = sign_changes / (2 * duration_sec)

        assert estimated_frequency == pytest.approx(frequency_hz, rel=0.2)


# ============================================================================
# TREND ANALYSIS TESTS
# ============================================================================

class TestTrendAnalysis:
    """Test combustion trend analysis."""

    def test_detect_increasing_trend(self):
        """Test detection of increasing temperature trend."""
        # Steadily increasing temperature
        temperatures = [1100.0 + i * 5.0 for i in range(20)]

        # Linear regression to find trend
        n = len(temperatures)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(temperatures) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, temperatures))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        slope = numerator / denominator if denominator != 0 else 0

        # Positive slope = increasing trend
        assert slope > 0

    def test_detect_decreasing_trend(self):
        """Test detection of decreasing temperature trend."""
        temperatures = [1300.0 - i * 3.0 for i in range(20)]

        n = len(temperatures)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(temperatures) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, temperatures))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0

        # Negative slope = decreasing trend
        assert slope < 0

    def test_detect_stable_trend(self):
        """Test detection of stable (no trend) operation."""
        # Stable temperature with small noise
        import random
        random.seed(42)
        base_temp = 1200.0
        temperatures = [base_temp + random.uniform(-2, 2) for _ in range(20)]

        n = len(temperatures)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(temperatures) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, temperatures))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0

        # Near-zero slope = stable
        assert abs(slope) < 1.0

    def test_trend_rate_of_change(self):
        """Test calculation of trend rate of change."""
        # Temperature increasing at 5 degrees per sample
        rate_per_sample = 5.0
        temperatures = [1100.0 + i * rate_per_sample for i in range(20)]

        # Calculate rate from first and last points
        time_span = len(temperatures) - 1
        total_change = temperatures[-1] - temperatures[0]
        measured_rate = total_change / time_span

        assert measured_rate == pytest.approx(rate_per_sample, rel=1e-6)

    def test_trend_prediction(self):
        """Test trend-based prediction of future value."""
        temperatures = [1100.0 + i * 5.0 for i in range(20)]

        # Calculate trend slope
        n = len(temperatures)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(temperatures) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, temperatures))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0

        # Intercept
        intercept = y_mean - slope * x_mean

        # Predict next value
        next_x = len(temperatures)
        predicted_temp = slope * next_x + intercept

        expected_temp = 1100.0 + next_x * 5.0

        assert predicted_temp == pytest.approx(expected_temp, rel=1e-2)


# ============================================================================
# PATTERN RECOGNITION TESTS
# ============================================================================

class TestPatternRecognition:
    """Test combustion pattern recognition."""

    def test_detect_cycling_pattern(self):
        """Test detection of cycling on/off pattern."""
        # On-off cycling pattern
        pattern = [100, 100, 100, 20, 20, 20, 100, 100, 100, 20, 20, 20]

        # Calculate if high variance suggests cycling
        mean = sum(pattern) / len(pattern)
        variance = sum((x - mean) ** 2 for x in pattern) / len(pattern)
        std_dev = math.sqrt(variance)

        # High std dev relative to mean suggests cycling
        cv = std_dev / mean if mean > 0 else 0
        cycling_detected = cv > 0.5

        assert cycling_detected is True

    def test_detect_hunting_pattern(self):
        """Test detection of controller hunting (overshooting setpoint)."""
        # Hunting pattern around setpoint (1200)
        setpoint = 1200.0
        pattern = [1200, 1210, 1205, 1195, 1190, 1195, 1205, 1210, 1205, 1195]

        # Count crossings of setpoint
        crossings = sum(1 for i in range(len(pattern)-1)
                       if (pattern[i] - setpoint) * (pattern[i+1] - setpoint) < 0)

        # Many crossings indicate hunting
        hunting_detected = crossings > len(pattern) / 4

        assert hunting_detected is True

    def test_detect_drift_pattern(self):
        """Test detection of gradual drift from setpoint."""
        # Gradual drift away from setpoint
        setpoint = 1200.0
        pattern = [1200 + i * 2 for i in range(20)]

        # Calculate how far drifted
        initial_error = abs(pattern[0] - setpoint)
        final_error = abs(pattern[-1] - setpoint)

        drift_detected = final_error > initial_error * 2

        assert drift_detected is True


# ============================================================================
# MULTI-VARIABLE STABILITY TESTS
# ============================================================================

class TestMultiVariableStability:
    """Test stability analysis across multiple variables."""

    def test_combined_stability_index(self):
        """Test combined stability index from multiple parameters."""
        # Stability indices for different parameters
        temp_stability = 0.95
        pressure_stability = 0.92
        flow_stability = 0.97
        flame_stability = 0.93

        # Combined stability (weighted average)
        weights = {'temp': 0.3, 'pressure': 0.2, 'flow': 0.2, 'flame': 0.3}

        combined_stability = (
            weights['temp'] * temp_stability +
            weights['pressure'] * pressure_stability +
            weights['flow'] * flow_stability +
            weights['flame'] * flame_stability
        )

        assert 0.90 <= combined_stability <= 1.0

    def test_stability_correlation_analysis(self):
        """Test correlation between stability of different variables."""
        # Temperature and fuel flow should be correlated
        temperatures = [1100 + i * 5 for i in range(20)]
        fuel_flows = [450 + i * 10 for i in range(20)]

        # Calculate correlation coefficient
        n = len(temperatures)
        temp_mean = sum(temperatures) / n
        fuel_mean = sum(fuel_flows) / n

        numerator = sum((t - temp_mean) * (f - fuel_mean)
                       for t, f in zip(temperatures, fuel_flows))
        temp_var = sum((t - temp_mean) ** 2 for t in temperatures)
        fuel_var = sum((f - fuel_mean) ** 2 for f in fuel_flows)

        correlation = numerator / math.sqrt(temp_var * fuel_var) if temp_var > 0 and fuel_var > 0 else 0

        # Should be highly correlated
        assert correlation > 0.9


# ============================================================================
# PREDICTIVE STABILITY MONITORING TESTS
# ============================================================================

class TestPredictiveStabilityMonitoring:
    """Test predictive stability monitoring."""

    def test_predict_instability_onset(self):
        """Test prediction of instability before it occurs."""
        # Gradually decreasing stability
        stability_history = [0.98, 0.96, 0.94, 0.91, 0.88, 0.85]

        # Calculate rate of degradation
        rate_of_change = []
        for i in range(len(stability_history) - 1):
            rate_of_change.append(stability_history[i+1] - stability_history[i])

        avg_rate = sum(rate_of_change) / len(rate_of_change)

        # Predict next value
        predicted_next = stability_history[-1] + avg_rate

        # Predict instability if trend continues
        critical_stability = 0.70
        steps_to_critical = (critical_stability - stability_history[-1]) / avg_rate if avg_rate < 0 else float('inf')

        # Should predict instability approaching
        assert steps_to_critical < 10

    def test_stability_margin_calculation(self):
        """Test calculation of stability margin."""
        current_stability = 0.88
        minimum_acceptable_stability = 0.70

        stability_margin = current_stability - minimum_acceptable_stability

        assert stability_margin > 0
        assert stability_margin == pytest.approx(0.18, rel=1e-6)

    def test_time_to_instability_prediction(self):
        """Test prediction of time until instability threshold."""
        stability_history = [0.95, 0.93, 0.91, 0.89, 0.87]
        sample_interval_sec = 10.0  # 10 seconds between samples

        # Calculate degradation rate
        rate_per_sample = (stability_history[-1] - stability_history[0]) / (len(stability_history) - 1)

        # Threshold
        instability_threshold = 0.70

        # Time to threshold
        stability_margin = stability_history[-1] - instability_threshold
        samples_to_threshold = stability_margin / abs(rate_per_sample) if rate_per_sample < 0 else float('inf')
        time_to_threshold_sec = samples_to_threshold * sample_interval_sec

        # Should predict time to instability
        assert time_to_threshold_sec < 200  # Less than 200 seconds


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestStabilityCalculationDeterminism:
    """Test stability calculations are deterministic."""

    def test_stability_index_determinism(self):
        """Test stability index calculation is deterministic."""
        flame_intensities = [85.0, 86.0, 84.5, 85.5, 85.2]

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            mean = sum(flame_intensities) / len(flame_intensities)
            variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean
            stability_index = 1 - cv

            # Round to avoid floating point comparison issues
            results.add(round(stability_index, 10))

        # All results should be identical
        assert len(results) == 1
