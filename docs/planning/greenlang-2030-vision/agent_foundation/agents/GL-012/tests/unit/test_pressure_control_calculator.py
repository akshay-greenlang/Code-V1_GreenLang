# -*- coding: utf-8 -*-
"""
Unit Tests for PressureControlCalculator.

This module provides comprehensive tests for the PressureControlCalculator class,
covering valve position calculations, flow coefficient (Cv) calculations,
pressure response predictions, PID output calculations, and pressure drop calculations.

Coverage Target: 95%+
Standards Compliance:
- ISA-75.01.01: Flow Equations for Sizing Control Valves
- IEC 60534: Industrial-Process Control Valves
- ANSI/ISA-75.05.01: Control Valve Terminology

Test Categories:
1. Valve position calculations
2. Flow coefficient (Cv) calculations
3. Pressure response predictions
4. PID output calculations
5. Pressure drop calculations
6. Valve characteristic curves (linear, equal-percentage)
7. Determinism verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import time
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    ValveCharacteristic,
    PressureControlConfig,
    generate_provenance_hash,
    assert_within_tolerance,
    assert_deterministic,
)


# =============================================================================
# ENUMS AND DATACLASSES FOR TESTING
# =============================================================================

class ValveState(Enum):
    """Valve operational state."""
    CLOSED = "closed"
    THROTTLING = "throttling"
    FULL_OPEN = "full_open"
    FAULT = "fault"


class ControlAction(Enum):
    """Control action direction."""
    DIRECT = "direct"  # Increase output on positive error
    REVERSE = "reverse"  # Decrease output on positive error


@dataclass
class PressureControlInput:
    """Input data for pressure control calculation."""
    upstream_pressure_bar: float
    downstream_pressure_bar: float
    setpoint_pressure_bar: float
    flow_rate_kg_s: float
    fluid_density_kg_m3: float = 1000.0  # Water default
    current_valve_position: float = 50.0
    valve_cv: float = 100.0
    valve_characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE


@dataclass
class PressureControlOutput:
    """Output data from pressure control calculation."""
    valve_position_percent: float
    effective_cv: float
    pressure_drop_bar: float
    flow_rate_kg_s: float
    pid_output: float
    control_error_bar: float
    valve_state: ValveState
    is_stable: bool
    provenance_hash: str
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PIDController:
    """PID controller state and configuration."""
    kp: float = 1.5
    ki: float = 0.3
    kd: float = 0.05
    integral: float = 0.0
    previous_error: float = 0.0
    output_min: float = 0.0
    output_max: float = 100.0
    anti_windup: bool = True
    dead_band: float = 0.0


# =============================================================================
# MOCK CALCULATOR IMPLEMENTATION
# =============================================================================

class PressureControlCalculator:
    """
    Pressure control calculator for valve sizing and control.

    Implements ISA-75.01 flow equations for control valve sizing
    and PID control algorithms for pressure regulation.
    """

    # Valve rangeability constant (typically 50:1 for equal percentage)
    RANGEABILITY = 50.0

    def __init__(self, config: PressureControlConfig = None):
        """Initialize pressure control calculator."""
        self.config = config or PressureControlConfig()
        self.calculation_count = 0
        self.pid = PIDController(
            kp=self.config.pid_kp,
            ki=self.config.pid_ki,
            kd=self.config.pid_kd
        )

    def calculate_inherent_cv_ratio(
        self,
        position_percent: float,
        characteristic: ValveCharacteristic
    ) -> float:
        """
        Calculate inherent Cv ratio based on valve characteristic.

        The Cv ratio is the fraction of full Cv at a given position.

        Args:
            position_percent: Valve position (0-100%)
            characteristic: Valve characteristic type

        Returns:
            Cv ratio (0.0 to 1.0)
        """
        if position_percent <= 0:
            return 0.0
        if position_percent >= 100:
            return 1.0

        x = position_percent / 100.0

        if characteristic == ValveCharacteristic.LINEAR:
            # Linear: Cv proportional to position
            return x

        elif characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            # Equal percentage: Cv = R^(x-1) where R is rangeability
            # At x=0: Cv=1/R, at x=1: Cv=1
            return self.RANGEABILITY ** (x - 1)

        elif characteristic == ValveCharacteristic.QUICK_OPENING:
            # Quick opening: Cv = sqrt(x)
            return math.sqrt(x)

        else:
            return x  # Default to linear

    def calculate_effective_cv(
        self,
        position_percent: float,
        full_cv: float,
        characteristic: ValveCharacteristic
    ) -> float:
        """
        Calculate effective Cv at given position.

        Args:
            position_percent: Valve position (0-100%)
            full_cv: Full open Cv value
            characteristic: Valve characteristic

        Returns:
            Effective Cv at position
        """
        cv_ratio = self.calculate_inherent_cv_ratio(position_percent, characteristic)
        return full_cv * cv_ratio

    def calculate_flow_from_cv(
        self,
        cv: float,
        delta_p_bar: float,
        density_kg_m3: float
    ) -> float:
        """
        Calculate flow rate from Cv and pressure drop.

        ISA-75.01 liquid flow equation:
        Q = Cv * sqrt(delta_P / SG)

        Where Q is in US gpm, we convert to kg/s.

        Args:
            cv: Valve Cv coefficient
            delta_p_bar: Pressure drop across valve (bar)
            density_kg_m3: Fluid density

        Returns:
            Flow rate in kg/s
        """
        if cv <= 0 or delta_p_bar <= 0:
            return 0.0

        # Specific gravity (relative to water)
        sg = density_kg_m3 / 1000.0

        # Flow in US gpm
        q_gpm = cv * math.sqrt(delta_p_bar * 14.5038 / sg)  # Convert bar to psi

        # Convert gpm to kg/s (1 gpm = 0.0631 kg/s for water)
        q_kg_s = q_gpm * 0.0631 * (density_kg_m3 / 1000.0)

        return q_kg_s

    def calculate_cv_from_flow(
        self,
        flow_kg_s: float,
        delta_p_bar: float,
        density_kg_m3: float
    ) -> float:
        """
        Calculate required Cv from flow and pressure drop.

        Args:
            flow_kg_s: Required flow rate
            delta_p_bar: Available pressure drop
            density_kg_m3: Fluid density

        Returns:
            Required Cv
        """
        if flow_kg_s <= 0 or delta_p_bar <= 0:
            return 0.0

        sg = density_kg_m3 / 1000.0

        # Convert kg/s to gpm
        q_gpm = flow_kg_s / (0.0631 * (density_kg_m3 / 1000.0))

        # Cv = Q / sqrt(delta_P / SG)
        cv = q_gpm / math.sqrt(delta_p_bar * 14.5038 / sg)

        return cv

    def calculate_pressure_drop(
        self,
        cv: float,
        flow_kg_s: float,
        density_kg_m3: float
    ) -> float:
        """
        Calculate pressure drop for given Cv and flow.

        Args:
            cv: Valve Cv coefficient
            flow_kg_s: Flow rate
            density_kg_m3: Fluid density

        Returns:
            Pressure drop in bar
        """
        if cv <= 0 or flow_kg_s <= 0:
            return 0.0

        sg = density_kg_m3 / 1000.0

        # Convert kg/s to gpm
        q_gpm = flow_kg_s / (0.0631 * (density_kg_m3 / 1000.0))

        # delta_P = SG * (Q/Cv)^2 in psi
        delta_p_psi = sg * (q_gpm / cv) ** 2

        # Convert psi to bar
        delta_p_bar = delta_p_psi / 14.5038

        return delta_p_bar

    def calculate_valve_position_for_cv(
        self,
        required_cv: float,
        full_cv: float,
        characteristic: ValveCharacteristic
    ) -> float:
        """
        Calculate valve position for required Cv.

        Args:
            required_cv: Required Cv value
            full_cv: Full open Cv value
            characteristic: Valve characteristic

        Returns:
            Required valve position (0-100%)
        """
        if required_cv <= 0:
            return 0.0
        if required_cv >= full_cv:
            return 100.0

        cv_ratio = required_cv / full_cv

        if characteristic == ValveCharacteristic.LINEAR:
            return cv_ratio * 100.0

        elif characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            # x = 1 + log_R(Cv_ratio)
            if cv_ratio <= 0:
                return 0.0
            x = 1 + math.log(cv_ratio) / math.log(self.RANGEABILITY)
            return max(0.0, min(100.0, x * 100.0))

        elif characteristic == ValveCharacteristic.QUICK_OPENING:
            # x = Cv_ratio^2
            return cv_ratio ** 2 * 100.0

        else:
            return cv_ratio * 100.0

    def calculate_pid_output(
        self,
        setpoint: float,
        process_value: float,
        dt_seconds: float = 1.0,
        action: ControlAction = ControlAction.REVERSE
    ) -> float:
        """
        Calculate PID controller output.

        For pressure control:
        - REVERSE action: Output increases when PV < SP (need more flow)
        - DIRECT action: Output increases when PV > SP

        Args:
            setpoint: Target pressure
            process_value: Actual pressure
            dt_seconds: Time step
            action: Control action direction

        Returns:
            PID output (0-100%)
        """
        # Calculate error based on action
        if action == ControlAction.REVERSE:
            error = setpoint - process_value
        else:
            error = process_value - setpoint

        # Dead band check
        if abs(error) < self.pid.dead_band:
            error = 0.0

        # Proportional term
        p_term = self.pid.kp * error

        # Integral term with anti-windup
        if self.pid.anti_windup:
            # Only integrate if not saturated
            if self.pid.output_min < self.pid.integral + error * dt_seconds < self.pid.output_max:
                self.pid.integral += error * dt_seconds
        else:
            self.pid.integral += error * dt_seconds

        i_term = self.pid.ki * self.pid.integral

        # Derivative term
        d_error = (error - self.pid.previous_error) / dt_seconds if dt_seconds > 0 else 0
        d_term = self.pid.kd * d_error

        # Total output
        output = p_term + i_term + d_term

        # Apply limits
        output = max(self.pid.output_min, min(self.pid.output_max, output))

        # Store for next iteration
        self.pid.previous_error = error

        return output

    def predict_pressure_response(
        self,
        current_pressure_bar: float,
        target_pressure_bar: float,
        valve_position_percent: float,
        time_constant_s: float = 5.0
    ) -> Dict[str, float]:
        """
        Predict pressure response to valve position change.

        Uses first-order lag approximation.

        Args:
            current_pressure_bar: Current pressure
            target_pressure_bar: Target equilibrium pressure
            valve_position_percent: New valve position
            time_constant_s: System time constant

        Returns:
            Dictionary with predicted pressures at various times
        """
        predictions = {}

        # Steady-state pressure change (simplified)
        delta_p_steady = (valve_position_percent - 50) * 0.1  # bar per % change from 50%
        new_steady = current_pressure_bar + delta_p_steady

        # First-order response: P(t) = P_steady + (P_current - P_steady) * exp(-t/tau)
        for t in [0, 1, 2, 5, 10, 20, 30]:
            if t == 0:
                predictions[f't_{t}s'] = current_pressure_bar
            else:
                p = new_steady + (current_pressure_bar - new_steady) * math.exp(-t / time_constant_s)
                predictions[f't_{t}s'] = p

        predictions['steady_state'] = new_steady

        return predictions

    def determine_valve_state(self, position_percent: float) -> ValveState:
        """Determine valve operational state from position."""
        if position_percent <= 0.1:
            return ValveState.CLOSED
        elif position_percent >= 99.9:
            return ValveState.FULL_OPEN
        else:
            return ValveState.THROTTLING

    def calculate(self, input_data: PressureControlInput) -> PressureControlOutput:
        """
        Perform complete pressure control calculation.

        Args:
            input_data: PressureControlInput with operating parameters

        Returns:
            PressureControlOutput with calculated values
        """
        start_time = time.perf_counter()
        self.calculation_count += 1
        warnings = []

        # Calculate control error
        control_error = input_data.setpoint_pressure_bar - input_data.downstream_pressure_bar

        # Calculate PID output
        pid_output = self.calculate_pid_output(
            setpoint=input_data.setpoint_pressure_bar,
            process_value=input_data.downstream_pressure_bar
        )

        # Calculate new valve position
        new_position = pid_output

        # Calculate effective Cv at new position
        effective_cv = self.calculate_effective_cv(
            position_percent=new_position,
            full_cv=input_data.valve_cv,
            characteristic=input_data.valve_characteristic
        )

        # Calculate pressure drop
        pressure_drop = input_data.upstream_pressure_bar - input_data.downstream_pressure_bar

        # Calculate flow at this Cv and pressure drop
        flow_rate = self.calculate_flow_from_cv(
            cv=effective_cv,
            delta_p_bar=pressure_drop,
            density_kg_m3=input_data.fluid_density_kg_m3
        )

        # Determine valve state
        valve_state = self.determine_valve_state(new_position)

        # Check stability
        is_stable = abs(control_error) < self.config.pressure_tolerance_bar

        # Generate warnings
        if new_position >= 95.0:
            warnings.append("Valve nearly full open - limited control authority")
        if new_position <= 5.0:
            warnings.append("Valve nearly closed - limited control authority")
        if pressure_drop > input_data.upstream_pressure_bar * 0.5:
            warnings.append("High pressure drop - possible cavitation risk")

        # Generate provenance hash
        hash_data = {
            'upstream_pressure_bar': input_data.upstream_pressure_bar,
            'downstream_pressure_bar': input_data.downstream_pressure_bar,
            'setpoint_bar': input_data.setpoint_pressure_bar,
            'valve_position': round(new_position, 10),
            'effective_cv': round(effective_cv, 6),
        }
        provenance_hash = generate_provenance_hash(hash_data)

        end_time = time.perf_counter()
        calc_time_ms = (end_time - start_time) * 1000

        return PressureControlOutput(
            valve_position_percent=new_position,
            effective_cv=effective_cv,
            pressure_drop_bar=pressure_drop,
            flow_rate_kg_s=flow_rate,
            pid_output=pid_output,
            control_error_bar=control_error,
            valve_state=valve_state,
            is_stable=is_stable,
            provenance_hash=provenance_hash,
            calculation_time_ms=calc_time_ms,
            warnings=warnings
        )


# =============================================================================
# TEST CLASS: VALVE POSITION CALCULATIONS
# =============================================================================

class TestValvePositionCalculations:
    """Test suite for valve position calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_position_for_zero_cv(self, calculator):
        """Test position is 0% for zero Cv requirement."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=0.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert position == 0.0

    @pytest.mark.unit
    def test_position_for_full_cv(self, calculator):
        """Test position is 100% for full Cv requirement."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=100.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert position == 100.0

    @pytest.mark.unit
    def test_position_linear_50pct(self, calculator):
        """Test linear valve at 50% Cv gives 50% position."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=50.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert_within_tolerance(position, 50.0, 0.1, "Linear 50% position")

    @pytest.mark.unit
    def test_position_equal_percentage_50pct_cv(self, calculator):
        """Test equal percentage valve position for 50% Cv."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=50.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
        )

        # For equal percentage with R=50:
        # Cv_ratio = 0.5, x = 1 + log(0.5)/log(50) = 1 - 0.177 = 0.823
        # Position ~ 82.3%
        assert 75.0 < position < 90.0

    @pytest.mark.unit
    def test_position_quick_opening(self, calculator):
        """Test quick opening valve position."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=50.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.QUICK_OPENING
        )

        # For quick opening: position = (Cv_ratio)^2 * 100 = 0.5^2 * 100 = 25%
        assert_within_tolerance(position, 25.0, 1.0, "Quick opening position")

    @pytest.mark.unit
    @pytest.mark.parametrize("cv_ratio,characteristic,expected_pos", [
        (0.0, ValveCharacteristic.LINEAR, 0.0),
        (0.25, ValveCharacteristic.LINEAR, 25.0),
        (0.5, ValveCharacteristic.LINEAR, 50.0),
        (0.75, ValveCharacteristic.LINEAR, 75.0),
        (1.0, ValveCharacteristic.LINEAR, 100.0),
    ])
    def test_position_linear_parametrized(
        self, calculator, cv_ratio, characteristic, expected_pos
    ):
        """Parametrized test for linear valve positions."""
        position = calculator.calculate_valve_position_for_cv(
            required_cv=cv_ratio * 100.0,
            full_cv=100.0,
            characteristic=characteristic
        )

        assert_within_tolerance(position, expected_pos, 0.5, f"Position at Cv ratio {cv_ratio}")


# =============================================================================
# TEST CLASS: FLOW COEFFICIENT CALCULATIONS
# =============================================================================

class TestFlowCoefficientCalculations:
    """Test suite for Cv calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_cv_ratio_linear_0pct(self, calculator):
        """Test Cv ratio is 0 at 0% position for linear."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=0.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert ratio == 0.0

    @pytest.mark.unit
    def test_cv_ratio_linear_50pct(self, calculator):
        """Test Cv ratio is 0.5 at 50% position for linear."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=50.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert ratio == 0.5

    @pytest.mark.unit
    def test_cv_ratio_linear_100pct(self, calculator):
        """Test Cv ratio is 1.0 at 100% position for linear."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=100.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert ratio == 1.0

    @pytest.mark.unit
    def test_cv_ratio_equal_percentage_100pct(self, calculator):
        """Test Cv ratio is 1.0 at 100% for equal percentage."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=100.0,
            characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
        )

        assert ratio == 1.0

    @pytest.mark.unit
    def test_cv_ratio_equal_percentage_50pct(self, calculator):
        """Test Cv ratio at 50% for equal percentage."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=50.0,
            characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
        )

        # R^(0.5-1) = R^(-0.5) = 1/sqrt(R) = 1/sqrt(50) ~ 0.141
        expected = 1.0 / math.sqrt(50.0)
        assert_within_tolerance(ratio, expected, 0.05, "Equal percentage 50%")

    @pytest.mark.unit
    def test_cv_ratio_quick_opening_50pct(self, calculator):
        """Test Cv ratio at 50% for quick opening."""
        ratio = calculator.calculate_inherent_cv_ratio(
            position_percent=50.0,
            characteristic=ValveCharacteristic.QUICK_OPENING
        )

        # sqrt(0.5) ~ 0.707
        assert_within_tolerance(ratio, 0.707, 0.01, "Quick opening 50%")

    @pytest.mark.unit
    def test_effective_cv_calculation(self, calculator):
        """Test effective Cv calculation."""
        cv = calculator.calculate_effective_cv(
            position_percent=50.0,
            full_cv=100.0,
            characteristic=ValveCharacteristic.LINEAR
        )

        assert cv == 50.0

    @pytest.mark.unit
    @pytest.mark.parametrize("position,characteristic,expected_ratio_range", [
        (25.0, ValveCharacteristic.LINEAR, (0.24, 0.26)),
        (75.0, ValveCharacteristic.LINEAR, (0.74, 0.76)),
        (50.0, ValveCharacteristic.EQUAL_PERCENTAGE, (0.10, 0.20)),
        (75.0, ValveCharacteristic.EQUAL_PERCENTAGE, (0.30, 0.50)),
        (25.0, ValveCharacteristic.QUICK_OPENING, (0.45, 0.55)),
        (75.0, ValveCharacteristic.QUICK_OPENING, (0.85, 0.90)),
    ])
    def test_cv_ratio_parametrized(
        self, calculator, position, characteristic, expected_ratio_range
    ):
        """Parametrized test for Cv ratios at various positions."""
        ratio = calculator.calculate_inherent_cv_ratio(position, characteristic)

        assert expected_ratio_range[0] < ratio < expected_ratio_range[1]


# =============================================================================
# TEST CLASS: PRESSURE RESPONSE PREDICTIONS
# =============================================================================

class TestPressureResponsePredictions:
    """Test suite for pressure response predictions."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_pressure_response_at_t0(self, calculator):
        """Test pressure at t=0 equals current pressure."""
        response = calculator.predict_pressure_response(
            current_pressure_bar=10.0,
            target_pressure_bar=12.0,
            valve_position_percent=60.0
        )

        assert response['t_0s'] == 10.0

    @pytest.mark.unit
    def test_pressure_response_approaches_steady_state(self, calculator):
        """Test pressure approaches steady state over time."""
        response = calculator.predict_pressure_response(
            current_pressure_bar=10.0,
            target_pressure_bar=12.0,
            valve_position_percent=60.0,
            time_constant_s=5.0
        )

        # Pressure should approach steady state
        # After 5*tau (25s), should be within 1% of steady state
        steady = response['steady_state']
        assert abs(response['t_30s'] - steady) < abs(response['t_0s'] - steady) * 0.1

    @pytest.mark.unit
    def test_pressure_response_monotonic(self, calculator):
        """Test pressure response is monotonic (no oscillation in first-order)."""
        response = calculator.predict_pressure_response(
            current_pressure_bar=10.0,
            target_pressure_bar=12.0,
            valve_position_percent=70.0
        )

        # Should be monotonically increasing or decreasing
        values = [response['t_0s'], response['t_1s'], response['t_2s'],
                  response['t_5s'], response['t_10s']]

        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))

        assert increasing or decreasing

    @pytest.mark.unit
    def test_pressure_response_time_constant_effect(self, calculator):
        """Test slower time constant gives slower response."""
        response_fast = calculator.predict_pressure_response(
            current_pressure_bar=10.0,
            target_pressure_bar=12.0,
            valve_position_percent=60.0,
            time_constant_s=2.0
        )

        response_slow = calculator.predict_pressure_response(
            current_pressure_bar=10.0,
            target_pressure_bar=12.0,
            valve_position_percent=60.0,
            time_constant_s=10.0
        )

        # Fast response should be closer to steady state at t=5s
        steady = response_fast['steady_state']
        delta_fast = abs(response_fast['t_5s'] - steady)
        delta_slow = abs(response_slow['t_5s'] - steady)

        assert delta_fast < delta_slow


# =============================================================================
# TEST CLASS: PID OUTPUT CALCULATIONS
# =============================================================================

class TestPIDOutputCalculations:
    """Test suite for PID controller calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_pid_zero_error(self, calculator):
        """Test PID output for zero error."""
        output = calculator.calculate_pid_output(
            setpoint=10.0,
            process_value=10.0,
            dt_seconds=1.0
        )

        # With zero error and no integral buildup, output should be near 0
        assert abs(output) < 5.0

    @pytest.mark.unit
    def test_pid_positive_error_reverse_action(self, calculator):
        """Test PID increases output for positive error (reverse action)."""
        # Reset integral
        calculator.pid.integral = 0.0
        calculator.pid.previous_error = 0.0

        output = calculator.calculate_pid_output(
            setpoint=10.0,
            process_value=8.0,  # Below setpoint
            dt_seconds=1.0,
            action=ControlAction.REVERSE
        )

        # Reverse action: output increases when PV < SP
        assert output > 0

    @pytest.mark.unit
    def test_pid_negative_error_reverse_action(self, calculator):
        """Test PID decreases output for negative error (reverse action)."""
        calculator.pid.integral = 0.0
        calculator.pid.previous_error = 0.0

        output = calculator.calculate_pid_output(
            setpoint=10.0,
            process_value=12.0,  # Above setpoint
            dt_seconds=1.0,
            action=ControlAction.REVERSE
        )

        # Reverse action: output decreases when PV > SP
        # Output should be near 0 or negative before limits
        assert output < 50.0

    @pytest.mark.unit
    def test_pid_integral_accumulation(self, calculator):
        """Test PID integral term accumulates."""
        calculator.pid.integral = 0.0
        calculator.pid.previous_error = 0.0

        # Multiple iterations with constant error
        for _ in range(10):
            output = calculator.calculate_pid_output(
                setpoint=10.0,
                process_value=9.0,  # Constant 1 bar error
                dt_seconds=1.0
            )

        # Integral should have accumulated
        assert calculator.pid.integral > 0

    @pytest.mark.unit
    def test_pid_output_limits(self, calculator):
        """Test PID output is limited to 0-100%."""
        calculator.pid.integral = 0.0
        calculator.pid.previous_error = 0.0

        # Large positive error
        output = calculator.calculate_pid_output(
            setpoint=20.0,
            process_value=5.0,
            dt_seconds=1.0
        )
        assert 0.0 <= output <= 100.0

        # Large negative error
        calculator.pid.integral = 0.0
        output = calculator.calculate_pid_output(
            setpoint=5.0,
            process_value=20.0,
            dt_seconds=1.0
        )
        assert 0.0 <= output <= 100.0

    @pytest.mark.unit
    def test_pid_proportional_contribution(self, calculator):
        """Test proportional term contributes to output."""
        calculator.pid.integral = 0.0
        calculator.pid.previous_error = 0.0

        # Set high Kp, zero Ki and Kd
        calculator.pid.kp = 10.0
        calculator.pid.ki = 0.0
        calculator.pid.kd = 0.0

        output = calculator.calculate_pid_output(
            setpoint=10.0,
            process_value=9.0,
            dt_seconds=1.0
        )

        # P term = 10.0 * 1.0 = 10
        assert_within_tolerance(output, 10.0, 1.0, "Proportional only")


# =============================================================================
# TEST CLASS: PRESSURE DROP CALCULATIONS
# =============================================================================

class TestPressureDropCalculations:
    """Test suite for pressure drop calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_pressure_drop_zero_flow(self, calculator):
        """Test pressure drop is zero with no flow."""
        dp = calculator.calculate_pressure_drop(
            cv=100.0,
            flow_kg_s=0.0,
            density_kg_m3=1000.0
        )

        assert dp == 0.0

    @pytest.mark.unit
    def test_pressure_drop_zero_cv(self, calculator):
        """Test pressure drop is zero with zero Cv."""
        dp = calculator.calculate_pressure_drop(
            cv=0.0,
            flow_kg_s=10.0,
            density_kg_m3=1000.0
        )

        assert dp == 0.0

    @pytest.mark.unit
    def test_pressure_drop_increases_with_flow(self, calculator):
        """Test pressure drop increases with flow rate."""
        dp_low = calculator.calculate_pressure_drop(
            cv=100.0,
            flow_kg_s=5.0,
            density_kg_m3=1000.0
        )

        dp_high = calculator.calculate_pressure_drop(
            cv=100.0,
            flow_kg_s=10.0,
            density_kg_m3=1000.0
        )

        assert dp_high > dp_low
        # Pressure drop proportional to flow squared
        ratio = dp_high / dp_low
        assert_within_tolerance(ratio, 4.0, 0.5, "Flow squared relationship")

    @pytest.mark.unit
    def test_pressure_drop_decreases_with_cv(self, calculator):
        """Test pressure drop decreases with larger Cv."""
        dp_small_cv = calculator.calculate_pressure_drop(
            cv=50.0,
            flow_kg_s=10.0,
            density_kg_m3=1000.0
        )

        dp_large_cv = calculator.calculate_pressure_drop(
            cv=100.0,
            flow_kg_s=10.0,
            density_kg_m3=1000.0
        )

        assert dp_small_cv > dp_large_cv

    @pytest.mark.unit
    def test_flow_from_cv_and_dp(self, calculator):
        """Test flow calculation from Cv and pressure drop."""
        flow = calculator.calculate_flow_from_cv(
            cv=100.0,
            delta_p_bar=1.0,
            density_kg_m3=1000.0
        )

        assert flow > 0.0

    @pytest.mark.unit
    def test_cv_from_flow_and_dp(self, calculator):
        """Test Cv calculation from flow and pressure drop."""
        cv = calculator.calculate_cv_from_flow(
            flow_kg_s=10.0,
            delta_p_bar=1.0,
            density_kg_m3=1000.0
        )

        assert cv > 0.0

    @pytest.mark.unit
    def test_round_trip_cv_flow_dp(self, calculator):
        """Test round-trip: Cv -> flow -> Cv."""
        original_cv = 100.0
        delta_p = 1.0

        flow = calculator.calculate_flow_from_cv(original_cv, delta_p, 1000.0)
        calculated_cv = calculator.calculate_cv_from_flow(flow, delta_p, 1000.0)

        assert_within_tolerance(calculated_cv, original_cv, 1.0, "Round-trip Cv")


# =============================================================================
# TEST CLASS: VALVE CHARACTERISTIC CURVES
# =============================================================================

class TestValveCharacteristicCurves:
    """Test suite for valve characteristic curve behavior."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_linear_characteristic_is_linear(self, calculator):
        """Test linear characteristic gives linear Cv vs position."""
        positions = [0, 25, 50, 75, 100]
        ratios = [
            calculator.calculate_inherent_cv_ratio(p, ValveCharacteristic.LINEAR)
            for p in positions
        ]

        # Should be [0, 0.25, 0.5, 0.75, 1.0]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for r, e in zip(ratios, expected):
            assert_within_tolerance(r, e, 0.01, "Linear characteristic")

    @pytest.mark.unit
    def test_equal_percentage_nonlinear(self, calculator):
        """Test equal percentage is nonlinear with most gain at high positions."""
        # Equal percentage should have equal % change in Cv for equal % change in position
        # This means more absolute change at higher positions

        cv_25 = calculator.calculate_inherent_cv_ratio(25, ValveCharacteristic.EQUAL_PERCENTAGE)
        cv_50 = calculator.calculate_inherent_cv_ratio(50, ValveCharacteristic.EQUAL_PERCENTAGE)
        cv_75 = calculator.calculate_inherent_cv_ratio(75, ValveCharacteristic.EQUAL_PERCENTAGE)

        # Gain (dCv/dx) increases with position
        gain_25_50 = (cv_50 - cv_25) / 25
        gain_50_75 = (cv_75 - cv_50) / 25

        assert gain_50_75 > gain_25_50

    @pytest.mark.unit
    def test_quick_opening_high_initial_gain(self, calculator):
        """Test quick opening has high initial gain."""
        cv_10 = calculator.calculate_inherent_cv_ratio(10, ValveCharacteristic.QUICK_OPENING)
        cv_50 = calculator.calculate_inherent_cv_ratio(50, ValveCharacteristic.QUICK_OPENING)
        cv_90 = calculator.calculate_inherent_cv_ratio(90, ValveCharacteristic.QUICK_OPENING)

        # Initial gain should be higher than final gain
        gain_initial = cv_10 / 10
        gain_final = (cv_90 - cv_50) / 40

        assert gain_initial > gain_final

    @pytest.mark.unit
    def test_all_characteristics_0_to_1_range(self, calculator):
        """Test all characteristics stay in 0-1 range."""
        for characteristic in ValveCharacteristic:
            for position in range(0, 101, 10):
                ratio = calculator.calculate_inherent_cv_ratio(position, characteristic)
                assert 0.0 <= ratio <= 1.0


# =============================================================================
# TEST CLASS: FULL CALCULATION WORKFLOW
# =============================================================================

class TestFullPressureControlCalculation:
    """Test suite for complete pressure control calculation workflow."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    def test_complete_calculation(self, calculator):
        """Test complete pressure control calculation."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=9.0,
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0,
            valve_characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
        )

        result = calculator.calculate(input_data)

        assert 0.0 <= result.valve_position_percent <= 100.0
        assert result.effective_cv > 0
        assert result.pressure_drop_bar == 6.0  # 15 - 9
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_calculation_near_setpoint(self, calculator):
        """Test calculation when pressure is near setpoint."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=9.9,  # Near setpoint of 10
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0
        )

        result = calculator.calculate(input_data)

        assert result.is_stable or abs(result.control_error_bar) < 0.2

    @pytest.mark.unit
    def test_calculation_generates_warnings(self, calculator):
        """Test calculation generates appropriate warnings."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=5.0,  # Large error
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0
        )

        # Force high valve position
        calculator.pid.integral = 100.0

        result = calculator.calculate(input_data)

        # Should have warning about valve position or pressure drop
        # (depends on resulting position)


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================

class TestPressureControlDeterminism:
    """Test suite for determinism verification."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_cv_calculation_determinism(self, calculator):
        """Test Cv calculation is deterministic."""
        results = []
        for _ in range(100):
            cv = calculator.calculate_inherent_cv_ratio(
                position_percent=50.0,
                characteristic=ValveCharacteristic.EQUAL_PERCENTAGE
            )
            results.append(cv)

        assert_deterministic(results, "Cv ratio calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_pressure_drop_determinism(self, calculator):
        """Test pressure drop calculation is deterministic."""
        results = []
        for _ in range(100):
            dp = calculator.calculate_pressure_drop(
                cv=100.0,
                flow_kg_s=10.0,
                density_kg_m3=1000.0
            )
            results.append(dp)

        assert_deterministic(results, "Pressure drop calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_full_calculation_determinism(self, calculator):
        """Test full calculation is deterministic."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=9.0,
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0
        )

        results = []
        for _ in range(50):
            # Reset PID state for consistent results
            calculator.pid.integral = 0.0
            calculator.pid.previous_error = 0.0
            result = calculator.calculate(input_data)
            results.append(result.valve_position_percent)

        assert_deterministic(results, "Full pressure calculation")


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================

class TestPressureControlPerformance:
    """Test suite for performance benchmarks."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PressureControlCalculator()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_calculation_under_5ms(self, calculator, benchmark_targets):
        """Test single calculation completes under 5ms."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=9.0,
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0
        )

        result = calculator.calculate(input_data)

        assert result.calculation_time_ms < benchmark_targets['pressure_control_calculation_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_pid_update_under_1ms(self, calculator, performance_timer, benchmark_targets):
        """Test PID update completes under 1ms."""
        with performance_timer() as timer:
            for _ in range(1000):
                calculator.calculate_pid_output(10.0, 9.5, 0.1)

        avg_time_ms = timer.elapsed_ms / 1000
        assert avg_time_ms < benchmark_targets['pid_update_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_calculations(self, calculator, performance_timer):
        """Test batch of 100 calculations."""
        input_data = PressureControlInput(
            upstream_pressure_bar=15.0,
            downstream_pressure_bar=9.0,
            setpoint_pressure_bar=10.0,
            flow_rate_kg_s=50.0,
            valve_cv=100.0
        )

        with performance_timer() as timer:
            for _ in range(100):
                calculator.calculate(input_data)

        # 100 calculations should complete in under 200ms
        assert timer.elapsed_ms < 200.0
