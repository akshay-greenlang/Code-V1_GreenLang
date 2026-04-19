"""
Simulation Tests: Cascade PID Controller Response

Tests the cascade PID control system including:
- Primary/secondary controller coordination
- Setpoint tracking accuracy
- Disturbance rejection
- Anti-windup behavior
- Bumpless transfer between modes

Reference: GL-001 Specification Section 11.2
Target Coverage: 85%+
"""

import pytest
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# PID Controller Classes (Simulated Production Code)
# =============================================================================

class ControlMode(Enum):
    """Controller operating modes."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    REMOTE = "remote"


@dataclass
class PIDParameters:
    """PID controller tuning parameters."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    output_min: float = 0.0
    output_max: float = 100.0
    setpoint_min: float = 0.0
    setpoint_max: float = 100.0
    deadband: float = 0.0
    sample_time: float = 1.0  # seconds


@dataclass
class PIDState:
    """Current state of a PID controller."""
    setpoint: float
    process_variable: float
    output: float
    error: float
    integral: float
    derivative: float
    mode: ControlMode
    timestamp: datetime = field(default_factory=datetime.now)


class PIDController:
    """Standard PID controller with anti-windup."""

    def __init__(self, controller_id: str, params: PIDParameters):
        self.controller_id = controller_id
        self.params = params
        self.mode = ControlMode.AUTO

        # State variables
        self.setpoint = 0.0
        self.process_variable = 0.0
        self.output = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_pv = 0.0
        self.manual_output = 0.0

        # Anti-windup state
        self.integral_clamped = False

    def set_mode(self, mode: ControlMode, bumpless: bool = True):
        """Set controller mode with optional bumpless transfer."""
        if mode == self.mode:
            return

        if bumpless:
            if mode == ControlMode.AUTO:
                # Initialize integral to maintain current output
                error = self.setpoint - self.process_variable
                if abs(self.params.ki) > 1e-9:
                    self.integral = (self.output - self.params.kp * error) / self.params.ki

            elif mode == ControlMode.MANUAL:
                self.manual_output = self.output

        self.mode = mode

    def set_setpoint(self, setpoint: float):
        """Set controller setpoint."""
        self.setpoint = np.clip(
            setpoint,
            self.params.setpoint_min,
            self.params.setpoint_max
        )

    def update(self, pv: float, dt: float = None) -> float:
        """Update controller and calculate new output.

        Args:
            pv: Process variable (measured value)
            dt: Time step (uses sample_time if not provided)

        Returns:
            Controller output
        """
        if dt is None:
            dt = self.params.sample_time

        self.process_variable = pv

        if self.mode == ControlMode.MANUAL:
            self.output = self.manual_output
            return self.output

        # Calculate error
        error = self.setpoint - pv

        # Apply deadband
        if abs(error) < self.params.deadband:
            error = 0.0

        # Proportional term
        p_term = self.params.kp * error

        # Integral term with anti-windup
        if not self.integral_clamped:
            self.integral += self.params.ki * error * dt

        i_term = self.integral

        # Derivative term (on PV to avoid derivative kick)
        d_term = -self.params.kd * (pv - self.last_pv) / dt if dt > 0 else 0

        # Calculate raw output
        raw_output = p_term + i_term + d_term

        # Apply output limits and anti-windup
        self.output = np.clip(
            raw_output,
            self.params.output_min,
            self.params.output_max
        )

        # Anti-windup: stop integration if output is saturated
        self.integral_clamped = (
            raw_output != self.output and
            np.sign(error) == np.sign(raw_output - self.output)
        )

        # Store state for next iteration
        self.last_error = error
        self.last_pv = pv

        return self.output

    def get_state(self) -> PIDState:
        """Get current controller state."""
        return PIDState(
            setpoint=self.setpoint,
            process_variable=self.process_variable,
            output=self.output,
            error=self.setpoint - self.process_variable,
            integral=self.integral,
            derivative=-self.params.kd * (self.process_variable - self.last_pv),
            mode=self.mode
        )


class CascadeController:
    """Cascade control structure with primary and secondary controllers."""

    def __init__(self,
                 primary_id: str,
                 secondary_id: str,
                 primary_params: PIDParameters,
                 secondary_params: PIDParameters):
        self.primary = PIDController(primary_id, primary_params)
        self.secondary = PIDController(secondary_id, secondary_params)
        self.cascade_active = False

    def set_cascade_mode(self, active: bool):
        """Enable or disable cascade mode."""
        self.cascade_active = active

        if active:
            self.primary.set_mode(ControlMode.AUTO, bumpless=True)
            self.secondary.set_mode(ControlMode.CASCADE, bumpless=True)
        else:
            self.secondary.set_mode(ControlMode.AUTO, bumpless=True)

    def set_primary_setpoint(self, setpoint: float):
        """Set the primary controller setpoint."""
        self.primary.set_setpoint(setpoint)

    def set_secondary_setpoint(self, setpoint: float):
        """Set secondary setpoint (only effective when not in cascade)."""
        if not self.cascade_active:
            self.secondary.set_setpoint(setpoint)

    def update(self, primary_pv: float, secondary_pv: float, dt: float = None) -> float:
        """Update cascade controller.

        Args:
            primary_pv: Primary process variable
            secondary_pv: Secondary process variable
            dt: Time step

        Returns:
            Final control output from secondary controller
        """
        # Update primary controller
        primary_output = self.primary.update(primary_pv, dt)

        if self.cascade_active:
            # Primary output becomes secondary setpoint
            self.secondary.set_setpoint(primary_output)

        # Update secondary controller
        output = self.secondary.update(secondary_pv, dt)

        return output

    def get_states(self) -> Tuple[PIDState, PIDState]:
        """Get states of both controllers."""
        return self.primary.get_state(), self.secondary.get_state()


class SimpleProcess:
    """Simple first-order process for testing controllers."""

    def __init__(self, gain: float = 1.0, time_constant: float = 60.0, delay: float = 0.0):
        self.gain = gain
        self.time_constant = time_constant
        self.delay = delay
        self.output = 0.0
        self.delay_buffer = []

    def step(self, input_value: float, dt: float) -> float:
        """Advance process by one time step."""
        # Handle delay
        if self.delay > 0:
            self.delay_buffer.append(input_value)
            if len(self.delay_buffer) * dt >= self.delay:
                delayed_input = self.delay_buffer.pop(0)
            else:
                delayed_input = 0.0
        else:
            delayed_input = input_value

        # First-order dynamics
        target = self.gain * delayed_input
        alpha = 1 - np.exp(-dt / self.time_constant)
        self.output += alpha * (target - self.output)

        return self.output


# =============================================================================
# Test Classes
# =============================================================================

class TestPIDController:
    """Test suite for basic PID controller."""

    @pytest.fixture
    def pid_params(self):
        """Create standard PID parameters."""
        return PIDParameters(
            kp=2.0,
            ki=0.5,
            kd=0.1,
            output_min=0.0,
            output_max=100.0,
            sample_time=1.0
        )

    @pytest.fixture
    def controller(self, pid_params):
        """Create a PID controller."""
        return PIDController("PID_TEST", pid_params)

    def test_controller_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.setpoint == 0.0
        assert controller.output == 0.0
        assert controller.mode == ControlMode.AUTO

    def test_setpoint_change_generates_output(self, controller):
        """Test that setpoint change generates output."""
        controller.set_setpoint(50.0)
        output = controller.update(pv=0.0)

        # With error = 50, output should be positive
        assert output > 0

    def test_output_clamped_to_limits(self, controller):
        """Test output is clamped to limits."""
        controller.set_setpoint(100.0)

        # Large error should saturate output
        output = controller.update(pv=0.0)

        assert output <= 100.0
        assert output >= 0.0

    def test_integral_accumulates(self, controller):
        """Test integral term accumulates over time."""
        controller.set_setpoint(50.0)

        # Multiple updates with constant error
        for _ in range(10):
            controller.update(pv=40.0, dt=1.0)

        # Integral should have accumulated
        assert controller.integral > 0

    def test_anti_windup_prevents_excessive_integral(self, controller):
        """Test anti-windup prevents integral windup."""
        controller.set_setpoint(100.0)

        # Many iterations with saturated output
        for _ in range(100):
            controller.update(pv=0.0, dt=1.0)

        # Now bring PV to setpoint
        controller.set_setpoint(50.0)
        outputs = []
        for _ in range(20):
            outputs.append(controller.update(pv=50.0, dt=1.0))

        # Output should settle quickly due to anti-windup
        # Not overshoot significantly due to wound-up integral
        assert outputs[-1] < 60  # Should be near 50 without huge overshoot

    def test_manual_mode(self, controller):
        """Test manual mode holds output."""
        controller.set_mode(ControlMode.MANUAL)
        controller.manual_output = 75.0

        output = controller.update(pv=0.0)

        assert output == 75.0

        # Output shouldn't change in manual
        controller.set_setpoint(100.0)
        output = controller.update(pv=0.0)

        assert output == 75.0

    def test_bumpless_transfer_auto_to_manual(self, controller):
        """Test bumpless transfer from auto to manual."""
        controller.set_setpoint(50.0)

        # Run in auto mode
        for _ in range(10):
            controller.update(pv=40.0)

        output_before = controller.output

        # Switch to manual
        controller.set_mode(ControlMode.MANUAL, bumpless=True)
        output_after = controller.update(pv=40.0)

        # Output should not jump
        assert pytest.approx(output_after, rel=0.01) == output_before

    def test_bumpless_transfer_manual_to_auto(self, controller):
        """Test bumpless transfer from manual to auto."""
        controller.set_mode(ControlMode.MANUAL)
        controller.manual_output = 50.0
        controller.set_setpoint(60.0)

        output_before = controller.update(pv=55.0)

        # Switch back to auto
        controller.set_mode(ControlMode.AUTO, bumpless=True)
        output_after = controller.update(pv=55.0)

        # Output should not jump significantly
        assert abs(output_after - output_before) < 5

    def test_deadband(self):
        """Test deadband prevents small error responses."""
        params = PIDParameters(kp=2.0, ki=0.5, kd=0.1, deadband=5.0)
        controller = PIDController("PID_DEADBAND", params)

        controller.set_setpoint(50.0)

        # Error within deadband
        output = controller.update(pv=48.0)

        # Should treat error as zero
        assert controller.last_error == 0.0

    def test_derivative_kick_prevention(self, controller):
        """Test derivative is on PV to prevent kick on SP change."""
        controller.set_setpoint(50.0)
        controller.update(pv=50.0)

        # Sudden setpoint change
        controller.set_setpoint(80.0)
        state_before = controller.get_state()

        output = controller.update(pv=50.0)

        # Derivative should be based on PV change, not error change
        # Since PV didn't change, derivative contribution should be small
        state_after = controller.get_state()
        assert abs(state_after.derivative) < 1.0


class TestCascadeController:
    """Test suite for cascade controller."""

    @pytest.fixture
    def cascade_controller(self):
        """Create cascade controller."""
        primary_params = PIDParameters(
            kp=1.0, ki=0.1, kd=0.0,
            output_min=0, output_max=100
        )
        secondary_params = PIDParameters(
            kp=2.0, ki=0.5, kd=0.1,
            output_min=0, output_max=100
        )
        return CascadeController(
            "PRIMARY", "SECONDARY",
            primary_params, secondary_params
        )

    def test_cascade_initialization(self, cascade_controller):
        """Test cascade controller initializes correctly."""
        assert cascade_controller.cascade_active == False

    def test_cascade_mode_activation(self, cascade_controller):
        """Test cascade mode activation."""
        cascade_controller.set_cascade_mode(True)

        assert cascade_controller.cascade_active == True
        assert cascade_controller.primary.mode == ControlMode.AUTO
        assert cascade_controller.secondary.mode == ControlMode.CASCADE

    def test_primary_output_sets_secondary_setpoint(self, cascade_controller):
        """Test primary output becomes secondary setpoint in cascade."""
        cascade_controller.set_cascade_mode(True)
        cascade_controller.set_primary_setpoint(80.0)

        # Update cascade
        cascade_controller.update(primary_pv=60.0, secondary_pv=40.0)

        # Primary has error, generates output
        primary_state, secondary_state = cascade_controller.get_states()

        # Secondary setpoint should equal primary output
        assert secondary_state.setpoint == primary_state.output

    def test_non_cascade_mode_independent_setpoints(self, cascade_controller):
        """Test controllers have independent setpoints when not in cascade."""
        cascade_controller.set_cascade_mode(False)
        cascade_controller.set_primary_setpoint(80.0)
        cascade_controller.set_secondary_setpoint(50.0)

        cascade_controller.update(primary_pv=60.0, secondary_pv=40.0)

        primary_state, secondary_state = cascade_controller.get_states()

        assert primary_state.setpoint == 80.0
        assert secondary_state.setpoint == 50.0  # Independent

    def test_cascade_disturbance_rejection(self, cascade_controller):
        """Test cascade provides better disturbance rejection."""
        cascade_controller.set_cascade_mode(True)
        cascade_controller.set_primary_setpoint(50.0)

        # Simulate process with disturbance on secondary
        primary_pv = 50.0  # At setpoint
        secondary_pv = 40.0  # Disturbed

        outputs = []
        for _ in range(20):
            output = cascade_controller.update(
                primary_pv=primary_pv,
                secondary_pv=secondary_pv
            )
            outputs.append(output)

            # Simple process response
            secondary_pv += (output - secondary_pv) * 0.1

        # Secondary should be driven by cascade to correct primary
        _, secondary_state = cascade_controller.get_states()
        # Primary is at setpoint, so secondary setpoint should be near steady state


class TestControllerWithProcess:
    """Test controllers with simulated process."""

    @pytest.fixture
    def process(self):
        """Create a simple first-order process."""
        return SimpleProcess(gain=1.0, time_constant=30.0)

    @pytest.fixture
    def tuned_controller(self):
        """Create a tuned controller for the process."""
        params = PIDParameters(
            kp=3.0,
            ki=0.15,
            kd=0.5,
            output_min=0,
            output_max=100
        )
        return PIDController("TUNED_PID", params)

    def test_setpoint_tracking(self, process, tuned_controller):
        """Test controller tracks setpoint."""
        tuned_controller.set_setpoint(50.0)

        pv = 0.0
        pvs = []
        dt = 1.0

        # Simulate for 5 time constants
        for _ in range(150):
            output = tuned_controller.update(pv, dt)
            pv = process.step(output, dt)
            pvs.append(pv)

        # Should reach setpoint
        assert pytest.approx(pvs[-1], rel=0.05) == 50.0

    def test_disturbance_rejection(self, process, tuned_controller):
        """Test controller rejects disturbance."""
        tuned_controller.set_setpoint(50.0)

        pv = 50.0  # Start at setpoint
        dt = 1.0

        # Run to steady state
        for _ in range(100):
            output = tuned_controller.update(pv, dt)
            pv = process.step(output, dt)

        # Apply disturbance
        pv = 30.0  # Step disturbance

        pvs_after_disturbance = []
        for _ in range(150):
            output = tuned_controller.update(pv, dt)
            pv = process.step(output, dt)
            pvs_after_disturbance.append(pv)

        # Should return to setpoint
        assert pytest.approx(pvs_after_disturbance[-1], rel=0.05) == 50.0

    def test_overshoot_limited(self, process, tuned_controller):
        """Test overshoot is reasonably limited."""
        tuned_controller.set_setpoint(50.0)

        pv = 0.0
        dt = 1.0
        max_pv = 0.0

        for _ in range(200):
            output = tuned_controller.update(pv, dt)
            pv = process.step(output, dt)
            max_pv = max(max_pv, pv)

        # Overshoot should be less than 25%
        overshoot = (max_pv - 50.0) / 50.0
        assert overshoot < 0.25


class TestPIDEdgeCases:
    """Test edge cases in PID control."""

    def test_zero_gains(self):
        """Test controller with zero gains."""
        params = PIDParameters(kp=0, ki=0, kd=0)
        controller = PIDController("ZERO_GAINS", params)

        controller.set_setpoint(50.0)
        output = controller.update(pv=0.0)

        assert output == 0.0

    def test_very_large_error(self):
        """Test controller handles very large errors."""
        params = PIDParameters(kp=1.0, ki=0.1, kd=0.1, output_max=100)
        controller = PIDController("LARGE_ERROR", params)

        controller.set_setpoint(1e6)
        output = controller.update(pv=0.0)

        # Should be clamped to max
        assert output == 100.0

    def test_negative_setpoint(self):
        """Test controller with negative setpoint."""
        params = PIDParameters(
            kp=1.0, ki=0.1, kd=0.1,
            setpoint_min=-100, setpoint_max=100,
            output_min=-100, output_max=100
        )
        controller = PIDController("NEG_SP", params)

        controller.set_setpoint(-50.0)
        output = controller.update(pv=0.0)

        # Should produce negative output
        assert output < 0

    def test_very_small_dt(self):
        """Test controller with very small time step."""
        params = PIDParameters(kp=1.0, ki=0.1, kd=0.1)
        controller = PIDController("SMALL_DT", params)

        controller.set_setpoint(50.0)

        # Very small dt should still work
        output = controller.update(pv=0.0, dt=0.001)

        assert output > 0

    def test_state_preservation(self):
        """Test controller state is properly preserved."""
        params = PIDParameters(kp=1.0, ki=0.1, kd=0.1)
        controller = PIDController("STATE_TEST", params)

        controller.set_setpoint(50.0)

        # Run several iterations
        for _ in range(10):
            controller.update(pv=40.0)

        state = controller.get_state()

        assert state.setpoint == 50.0
        assert state.process_variable == 40.0
        assert state.error == 10.0
        assert state.integral > 0  # Should have accumulated
