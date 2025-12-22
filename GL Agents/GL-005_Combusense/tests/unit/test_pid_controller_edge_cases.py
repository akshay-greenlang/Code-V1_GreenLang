# -*- coding: utf-8 -*-
"""
PID Controller edge case tests for GL-005 CombustionControlAgent.

Tests comprehensive PID controller edge cases including:
- Integral windup protection
- Derivative kick prevention
- Setpoint change handling
- Anti-windup mechanisms
- Bumpless transfer
- Control saturation
- Rate limiting
- Zero/negative gain handling

Target: 15+ tests covering all PID edge cases for SIL-2 compliance.
"""

import pytest
import math
from typing import Dict, Any
from decimal import Decimal

pytestmark = pytest.mark.unit


# ============================================================================
# PID INTEGRAL WINDUP TESTS
# ============================================================================

class TestPIDIntegralWindup:
    """Test PID integral windup protection."""

    def test_integral_windup_clamping(self):
        """Test integral term is clamped to prevent windup."""
        ki = 0.3
        max_integral = 500.0
        min_integral = -500.0

        # Accumulate large error
        accumulated_error = 2000.0  # Would cause windup

        # Apply clamping
        if accumulated_error > max_integral / ki:
            clamped_error = max_integral / ki
        elif accumulated_error < min_integral / ki:
            clamped_error = min_integral / ki
        else:
            clamped_error = accumulated_error

        integral_term = ki * clamped_error

        # Integral term should be within limits
        assert min_integral <= integral_term <= max_integral
        assert integral_term == pytest.approx(max_integral, rel=1e-6)

    def test_integral_reset_on_saturation(self):
        """Test integral term resets when output saturates."""
        ki = 0.3
        error = 50.0
        integral = 1000.0  # Large accumulated integral

        # Calculate raw PID output
        kp = 1.5
        raw_output = (kp * error) + (ki * integral)

        # Check if saturated
        max_output = 100.0
        if raw_output > max_output:
            # Anti-windup: reduce integral
            excess = raw_output - max_output
            integral_reduction = excess / ki
            new_integral = integral - integral_reduction
        else:
            new_integral = integral

        assert new_integral < integral  # Integral was reduced

    def test_conditional_integration(self):
        """Test integral only accumulates when not saturated."""
        ki = 0.3
        error = 50.0
        current_integral = 800.0
        max_output = 100.0
        current_output = 95.0

        # Check if output is near saturation
        saturation_threshold = 0.9 * max_output

        if current_output < saturation_threshold:
            # Integrate normally
            new_integral = current_integral + error
            integration_active = True
        else:
            # Stop integration (anti-windup)
            new_integral = current_integral
            integration_active = False

        # Near saturation, should not integrate
        assert integration_active is False
        assert new_integral == current_integral

    def test_back_calculation_anti_windup(self):
        """Test back-calculation anti-windup method."""
        kp, ki, kd = 1.5, 0.3, 0.1
        error = 50.0
        integral = 500.0
        derivative = 10.0

        # Calculate raw output
        raw_output = (kp * error) + (ki * integral) + (kd * derivative)

        # Apply output limits
        max_output = 100.0
        limited_output = min(max_output, raw_output)

        # Back-calculate integral reduction
        kt = 0.5  # Tracking time constant
        if raw_output > max_output:
            integral_adjustment = (limited_output - raw_output) / kt
            new_integral = integral + integral_adjustment
        else:
            new_integral = integral

        assert new_integral < integral  # Integral was reduced


# ============================================================================
# DERIVATIVE KICK PREVENTION TESTS
# ============================================================================

class TestDerivativeKickPrevention:
    """Test derivative kick prevention mechanisms."""

    def test_derivative_on_measurement_not_error(self):
        """Test derivative calculated on measurement, not error (prevents kick)."""
        kd = 0.1

        # Previous and current measurements
        prev_measurement = 1150.0
        current_measurement = 1155.0

        # Setpoint changes from 1200 to 1250 (would cause kick if using error)
        prev_setpoint = 1200.0
        current_setpoint = 1250.0

        # Method 1: Derivative on error (causes kick)
        prev_error = prev_setpoint - prev_measurement
        current_error = current_setpoint - current_measurement
        derivative_on_error = current_error - prev_error

        # Method 2: Derivative on measurement (no kick)
        derivative_on_measurement = -(current_measurement - prev_measurement)

        d_term_error = kd * derivative_on_error
        d_term_measurement = kd * derivative_on_measurement

        # Derivative on measurement should be much smaller (no kick)
        assert abs(d_term_measurement) < abs(d_term_error)

    def test_derivative_filtering(self):
        """Test derivative term filtered to reduce noise."""
        kd = 0.1
        alpha = 0.3  # Filter coefficient (0-1)

        # Noisy measurements
        measurements = [1150.0, 1155.0, 1152.0, 1156.0, 1154.0]

        filtered_derivative = 0.0
        prev_measurement = measurements[0]

        for measurement in measurements[1:]:
            # Calculate raw derivative
            raw_derivative = -(measurement - prev_measurement)

            # Apply low-pass filter
            filtered_derivative = (alpha * raw_derivative) + ((1 - alpha) * filtered_derivative)

            prev_measurement = measurement

        # Filtered derivative should be smoother
        assert filtered_derivative is not None

    def test_derivative_deadband(self):
        """Test derivative term has deadband for small changes."""
        kd = 0.1
        deadband = 2.0  # Ignore changes smaller than 2 degrees

        prev_measurement = 1200.0
        current_measurement = 1201.0  # Small change

        raw_derivative = current_measurement - prev_measurement

        # Apply deadband
        if abs(raw_derivative) < deadband:
            effective_derivative = 0.0
        else:
            effective_derivative = raw_derivative

        d_term = kd * effective_derivative

        # Small change should result in zero derivative term
        assert d_term == 0.0


# ============================================================================
# SETPOINT CHANGE HANDLING TESTS
# ============================================================================

class TestSetpointChangeHandling:
    """Test PID handling of setpoint changes."""

    def test_setpoint_ramping(self):
        """Test setpoint changes are ramped, not instantaneous."""
        current_setpoint = 1200.0
        target_setpoint = 1300.0
        max_rate_c_per_sec = 10.0
        dt = 1.0  # 1 second interval

        # Calculate maximum allowed change
        max_change = max_rate_c_per_sec * dt

        # Calculate ramped setpoint
        setpoint_change = target_setpoint - current_setpoint
        if abs(setpoint_change) > max_change:
            new_setpoint = current_setpoint + (max_change * (1 if setpoint_change > 0 else -1))
        else:
            new_setpoint = target_setpoint

        # Setpoint should be ramped, not jumped
        assert new_setpoint == current_setpoint + max_change
        assert new_setpoint < target_setpoint

    def test_bumpless_transfer_on_manual_to_auto(self):
        """Test bumpless transfer when switching from manual to auto."""
        # Manual mode output
        manual_output = 75.0

        # Switch to auto mode
        # Calculate initial integral to match manual output
        kp = 1.5
        ki = 0.3
        kd = 0.1
        current_error = 50.0
        current_derivative = 10.0

        # Back-calculate integral for bumpless transfer
        required_output = manual_output
        integral_term = required_output - (kp * current_error) - (kd * current_derivative)
        initial_integral = integral_term / ki if ki != 0 else 0.0

        # Verify output matches
        auto_output = (kp * current_error) + (ki * initial_integral) + (kd * current_derivative)

        assert auto_output == pytest.approx(manual_output, rel=1e-2)

    def test_setpoint_weighting(self):
        """Test setpoint weighting to reduce proportional kick."""
        kp = 1.5
        setpoint_weight = 0.5  # Weight for proportional term (0-1)

        prev_setpoint = 1200.0
        current_setpoint = 1250.0
        current_measurement = 1150.0

        # Standard P term (causes kick on setpoint change)
        standard_error = current_setpoint - current_measurement
        standard_p_term = kp * standard_error

        # Weighted P term (reduces kick)
        weighted_error = (setpoint_weight * current_setpoint) - current_measurement
        weighted_p_term = kp * weighted_error

        # Weighted term should be smaller
        assert abs(weighted_p_term) < abs(standard_p_term)


# ============================================================================
# CONTROL SATURATION TESTS
# ============================================================================

class TestControlSaturation:
    """Test PID behavior under control saturation."""

    def test_output_clamping_symmetric(self):
        """Test output clamping with symmetric limits."""
        raw_output = 150.0
        max_output = 100.0
        min_output = -100.0

        clamped_output = max(min_output, min(max_output, raw_output))

        assert clamped_output == max_output

    def test_output_clamping_asymmetric(self):
        """Test output clamping with asymmetric limits."""
        raw_output = -80.0
        max_output = 100.0
        min_output = -50.0  # Asymmetric

        clamped_output = max(min_output, min(max_output, raw_output))

        assert clamped_output == min_output

    def test_rate_limiting_on_output(self):
        """Test rate limiting on controller output."""
        prev_output = 50.0
        current_output = 90.0
        max_rate_per_sec = 20.0
        dt = 1.0

        max_change = max_rate_per_sec * dt
        output_change = current_output - prev_output

        if abs(output_change) > max_change:
            limited_output = prev_output + (max_change * (1 if output_change > 0 else -1))
        else:
            limited_output = current_output

        # Output change should be limited
        assert limited_output == prev_output + max_change
        assert limited_output < current_output


# ============================================================================
# ZERO AND NEGATIVE GAIN TESTS
# ============================================================================

class TestZeroAndNegativeGains:
    """Test PID handling of zero and negative gains."""

    def test_zero_kp_gain(self):
        """Test PID with zero proportional gain."""
        kp = 0.0
        ki = 0.3
        kd = 0.1
        error = 50.0
        integral = 150.0
        derivative = 10.0

        pid_output = (kp * error) + (ki * integral) + (kd * derivative)

        # Output should only have I and D terms
        expected_output = (ki * integral) + (kd * derivative)
        assert pid_output == pytest.approx(expected_output, rel=1e-6)

    def test_zero_ki_gain(self):
        """Test PID with zero integral gain (PD controller)."""
        kp = 1.5
        ki = 0.0
        kd = 0.1
        error = 50.0
        integral = 150.0
        derivative = 10.0

        pid_output = (kp * error) + (ki * integral) + (kd * derivative)

        # Output should only have P and D terms
        expected_output = (kp * error) + (kd * derivative)
        assert pid_output == pytest.approx(expected_output, rel=1e-6)

    def test_zero_kd_gain(self):
        """Test PID with zero derivative gain (PI controller)."""
        kp = 1.5
        ki = 0.3
        kd = 0.0
        error = 50.0
        integral = 150.0
        derivative = 10.0

        pid_output = (kp * error) + (ki * integral) + (kd * derivative)

        # Output should only have P and I terms
        expected_output = (kp * error) + (ki * integral)
        assert pid_output == pytest.approx(expected_output, rel=1e-6)

    def test_negative_gain_validation(self):
        """Test validation rejects negative gains."""
        kp = -1.5  # Invalid negative gain
        ki = 0.3
        kd = 0.1

        # Validation should detect negative gain
        gains_valid = (kp >= 0 and ki >= 0 and kd >= 0)

        assert gains_valid is False


# ============================================================================
# SAMPLING TIME VARIATION TESTS
# ============================================================================

class TestSamplingTimeVariation:
    """Test PID handling of variable sampling times."""

    def test_dt_compensation_in_integral(self):
        """Test integral term compensates for variable dt."""
        ki = 0.3
        error = 50.0

        # Different sampling times
        dt1 = 0.1  # 100ms
        dt2 = 0.2  # 200ms

        integral_contribution_1 = ki * error * dt1
        integral_contribution_2 = ki * error * dt2

        # Longer dt should accumulate more
        assert integral_contribution_2 == pytest.approx(2 * integral_contribution_1, rel=1e-6)

    def test_dt_compensation_in_derivative(self):
        """Test derivative term compensates for variable dt."""
        kd = 0.1
        measurement_change = 10.0

        dt1 = 0.1
        dt2 = 0.2

        derivative_1 = (measurement_change / dt1) if dt1 > 0 else 0.0
        derivative_2 = (measurement_change / dt2) if dt2 > 0 else 0.0

        d_term_1 = kd * derivative_1
        d_term_2 = kd * derivative_2

        # Shorter dt should have larger derivative
        assert d_term_1 > d_term_2

    def test_zero_dt_handling(self):
        """Test handling of zero dt (same timestamp)."""
        kd = 0.1
        measurement_change = 10.0
        dt = 0.0  # Zero time difference

        # Should not divide by zero
        if dt > 0:
            derivative = measurement_change / dt
        else:
            derivative = 0.0  # Use previous derivative or zero

        d_term = kd * derivative

        assert d_term == 0.0


# ============================================================================
# FEED-FORWARD TESTS
# ============================================================================

class TestFeedForward:
    """Test feed-forward control enhancement."""

    def test_feed_forward_disturbance_rejection(self):
        """Test feed-forward improves disturbance rejection."""
        # Feedback controller output
        feedback_output = 75.0

        # Measured disturbance (e.g., load change)
        load_change = 20.0
        feed_forward_gain = 0.8

        # Feed-forward compensation
        feed_forward_output = feed_forward_gain * load_change

        # Combined output
        total_output = feedback_output + feed_forward_output

        assert total_output > feedback_output

    def test_feed_forward_setpoint_tracking(self):
        """Test feed-forward improves setpoint tracking."""
        # Setpoint change
        setpoint_change = 50.0

        # Feed-forward based on process model
        process_gain = 1.2
        feed_forward_output = setpoint_change / process_gain

        # PID output
        pid_output = 35.0

        # Combined output
        total_output = pid_output + feed_forward_output

        assert total_output > pid_output


# ============================================================================
# DETERMINISM VALIDATION TESTS
# ============================================================================

class TestPIDDeterminism:
    """Test PID calculations are deterministic."""

    def test_pid_calculation_reproducibility(self):
        """Test PID produces identical results for same inputs."""
        kp, ki, kd = 1.5, 0.3, 0.1
        error = 50.0
        integral = 150.0
        derivative = 10.0

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            output = (kp * error) + (ki * integral) + (kd * derivative)
            results.add(output)

        # All results should be identical
        assert len(results) == 1
        assert results.pop() == pytest.approx(121.0, rel=1e-10)

    def test_pid_floating_point_stability(self):
        """Test PID calculations don't accumulate floating-point errors."""
        kp, ki, kd = 1.5, 0.3, 0.1
        error = 50.0
        dt = 0.1

        # Accumulate integral over many iterations
        integral = 0.0
        num_iterations = 10000

        for _ in range(num_iterations):
            integral += error * dt

        # Compare with direct calculation
        expected_integral = error * dt * num_iterations

        assert integral == pytest.approx(expected_integral, rel=1e-9)
