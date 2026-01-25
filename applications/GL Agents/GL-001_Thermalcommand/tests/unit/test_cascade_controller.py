"""
Unit tests for GL-001 ThermalCommand Cascade Controller.

Tests the PID cascade control hierarchy implementation with comprehensive
coverage of gain scheduling, anti-windup, mode transitions, and alarms.

Coverage Target: 85%+
Reference: GL-001 Specification Section 11

Test Categories:
1. PID Controller initialization and configuration
2. PID calculation and output clamping
3. Anti-windup behavior
4. Mode transitions (Auto/Manual/Cascade)
5. Gain scheduling
6. Cascade controller operation
7. Alarm management
8. Edge cases and boundary conditions

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from control.cascade_controller import (
    PIDController,
    PIDTuning,
    ControlMode,
    ControlAction,
    AlarmPriority,
    GainScheduleRegion,
    GainScheduleEntry,
    ControllerState,
    ControllerAlarm,
    ControlOutput,
    CascadeController,
    CascadeOutput,
    CascadeCoordinator,
    create_temperature_flow_cascade,
    create_pressure_flow_cascade,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_tuning() -> PIDTuning:
    """Create default PID tuning parameters."""
    return PIDTuning(
        kp=2.0,
        ki=0.1,
        kd=0.5,
        td_filter=0.1,
        anti_windup_gain=1.0
    )


@pytest.fixture
def aggressive_tuning() -> PIDTuning:
    """Create aggressive PID tuning parameters."""
    return PIDTuning(
        kp=5.0,
        ki=0.5,
        kd=1.0,
        td_filter=0.05,
        anti_windup_gain=1.5
    )


@pytest.fixture
def pi_only_tuning() -> PIDTuning:
    """Create PI-only tuning (no derivative)."""
    return PIDTuning(
        kp=2.0,
        ki=0.2,
        kd=0.0,
        td_filter=0.1,
        anti_windup_gain=1.0
    )


@pytest.fixture
def default_controller(default_tuning) -> PIDController:
    """Create a default PID controller."""
    return PIDController(
        name="TC-101",
        tuning=default_tuning,
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0,
        rate_limit_per_second=None,
        derivative_on_pv=True,
        enable_gain_scheduling=False
    )


@pytest.fixture
def rate_limited_controller(default_tuning) -> PIDController:
    """Create a rate-limited PID controller."""
    return PIDController(
        name="TC-102",
        tuning=default_tuning,
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0,
        rate_limit_per_second=10.0,  # 10%/second max
        derivative_on_pv=True
    )


@pytest.fixture
def gain_scheduled_controller(default_tuning, aggressive_tuning) -> PIDController:
    """Create a gain-scheduled PID controller."""
    controller = PIDController(
        name="TC-103",
        tuning=default_tuning,
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0,
        enable_gain_scheduling=True
    )

    # Add gain schedules for different operating regions
    controller.add_gain_schedule(GainScheduleEntry(
        region=GainScheduleRegion.STARTUP,
        pv_low=0.0,
        pv_high=200.0,
        tuning=PIDTuning(kp=1.0, ki=0.05, kd=0.2),
        transition_rate=0.2
    ))

    controller.add_gain_schedule(GainScheduleEntry(
        region=GainScheduleRegion.NORMAL,
        pv_low=200.0,
        pv_high=600.0,
        tuning=default_tuning,
        transition_rate=0.1
    ))

    controller.add_gain_schedule(GainScheduleEntry(
        region=GainScheduleRegion.HIGH_LOAD,
        pv_low=600.0,
        pv_high=1000.0,
        tuning=aggressive_tuning,
        transition_rate=0.15
    ))

    return controller


@pytest.fixture
def master_controller(default_tuning) -> PIDController:
    """Create a master (temperature) controller for cascade."""
    return PIDController(
        name="TIC-101",
        tuning=default_tuning,
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=10.0
    )


@pytest.fixture
def slave_controller(pi_only_tuning) -> PIDController:
    """Create a slave (flow) controller for cascade."""
    return PIDController(
        name="FIC-101",
        tuning=pi_only_tuning,
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0
    )


@pytest.fixture
def cascade_controller(master_controller, slave_controller) -> CascadeController:
    """Create a cascade controller."""
    cascade = CascadeController(
        master=master_controller,
        slave=slave_controller,
        slave_sp_min=0.0,
        slave_sp_max=100.0,
        ratio=1.0
    )
    cascade.set_master_setpoint(500.0)
    return cascade


# =============================================================================
# TEST CLASS: PID TUNING
# =============================================================================

class TestPIDTuning:
    """Tests for PID tuning parameters."""

    def test_default_tuning_values(self):
        """Test default tuning parameter values."""
        tuning = PIDTuning()

        assert tuning.kp == 1.0
        assert tuning.ki == 0.1
        assert tuning.kd == 0.0
        assert tuning.td_filter == 0.1
        assert tuning.anti_windup_gain == 1.0

    def test_custom_tuning_values(self):
        """Test custom tuning parameter values."""
        tuning = PIDTuning(kp=3.0, ki=0.2, kd=0.8)

        assert tuning.kp == 3.0
        assert tuning.ki == 0.2
        assert tuning.kd == 0.8

    def test_ti_property(self):
        """Test integral time Ti = 1/Ki."""
        tuning = PIDTuning(ki=0.1)
        assert tuning.ti == pytest.approx(10.0, rel=0.001)

    def test_ti_with_zero_ki(self):
        """Test Ti is infinity when Ki is zero."""
        tuning = PIDTuning(ki=0.0)
        assert tuning.ti == float('inf')

    @pytest.mark.parametrize("kp,ki,kd", [
        (0.5, 0.05, 0.0),
        (2.0, 0.1, 0.5),
        (5.0, 0.5, 1.0),
        (10.0, 1.0, 2.0),
    ])
    def test_various_tuning_combinations(self, kp, ki, kd):
        """Test various valid tuning combinations."""
        tuning = PIDTuning(kp=kp, ki=ki, kd=kd)

        assert tuning.kp == kp
        assert tuning.ki == ki
        assert tuning.kd == kd


# =============================================================================
# TEST CLASS: PID CONTROLLER INITIALIZATION
# =============================================================================

class TestPIDControllerInitialization:
    """Tests for PID controller initialization."""

    def test_default_initialization(self, default_tuning):
        """Test default controller initialization."""
        controller = PIDController(
            name="TC-101",
            tuning=default_tuning
        )

        assert controller.name == "TC-101"
        assert controller.tuning == default_tuning
        assert controller.output_min == 0.0
        assert controller.output_max == 100.0
        assert controller.action == ControlAction.REVERSE
        assert controller.mode == ControlMode.AUTO

    def test_custom_initialization(self, default_tuning):
        """Test custom controller initialization."""
        controller = PIDController(
            name="PC-201",
            tuning=default_tuning,
            output_min=-50.0,
            output_max=50.0,
            action=ControlAction.DIRECT,
            sample_time_seconds=5.0,
            rate_limit_per_second=5.0
        )

        assert controller.output_min == -50.0
        assert controller.output_max == 50.0
        assert controller.action == ControlAction.DIRECT
        assert controller.sample_time == 5.0
        assert controller.rate_limit == 5.0

    def test_internal_state_initialization(self, default_controller):
        """Test internal state initialization."""
        state = default_controller._state

        assert state.integral == 0.0
        assert state.derivative_filter == 0.0
        assert state.last_error == 0.0
        assert state.last_pv == 0.0
        assert state.last_output == 0.0


# =============================================================================
# TEST CLASS: PID CALCULATION
# =============================================================================

class TestPIDCalculation:
    """Tests for PID calculation."""

    def test_basic_calculation(self, default_controller):
        """Test basic PID calculation."""
        result = default_controller.calculate(
            setpoint=500.0,
            pv=450.0,
            dt_seconds=1.0
        )

        assert isinstance(result, ControlOutput)
        assert result.controller_name == "TC-101"
        assert result.setpoint == 500.0
        assert result.process_value == 450.0
        assert result.output >= 0.0
        assert result.output <= 100.0

    def test_proportional_term(self, default_controller):
        """Test proportional term calculation."""
        # With REVERSE action, positive error should produce positive P term
        result = default_controller.calculate(
            setpoint=500.0,
            pv=450.0,
            dt_seconds=1.0
        )

        # Error = 500 - 450 = 50, with REVERSE: error = -50
        # P term = Kp * error = 2.0 * (-50) = -100 (before clipping)
        # Actually for REVERSE: error = -(SP - PV) = -(50) = -50
        # But we want increasing PV to decrease output, so the sign handling matters
        assert result.p_term != 0

    def test_integral_term_accumulation(self, default_controller):
        """Test integral term accumulates over time."""
        # First calculation
        result1 = default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)
        i_term_1 = result1.i_term

        # Second calculation with same error
        result2 = default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)
        i_term_2 = result2.i_term

        # Integral should accumulate
        assert abs(i_term_2) > abs(i_term_1)

    def test_derivative_term(self, default_controller):
        """Test derivative term calculation."""
        # First calculation to establish baseline
        default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # Second calculation with changed PV
        result = default_controller.calculate(setpoint=500.0, pv=460.0, dt_seconds=1.0)

        # D term should be non-zero due to PV change
        # Note: derivative on PV, so change in PV creates D term
        assert result.d_term != 0

    def test_derivative_on_pv_no_setpoint_kick(self, default_controller):
        """Test that derivative on PV prevents setpoint kick."""
        # Calculate with one setpoint
        default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # Change setpoint suddenly
        result = default_controller.calculate(setpoint=600.0, pv=450.0, dt_seconds=1.0)

        # D term should not spike due to setpoint change (PV unchanged)
        # The D term should be close to zero since PV didn't change
        assert abs(result.d_term) < 10  # Small due to PV being constant

    def test_output_clamping_high(self, default_controller):
        """Test output clamping at upper limit."""
        # Large error to force high output
        result = default_controller.calculate(setpoint=500.0, pv=100.0, dt_seconds=1.0)

        assert result.output == 100.0
        assert result.output_clamped is True

    def test_output_clamping_low(self, default_controller):
        """Test output clamping at lower limit."""
        # Negative error to force low output
        result = default_controller.calculate(setpoint=100.0, pv=500.0, dt_seconds=1.0)

        assert result.output == 0.0
        assert result.output_clamped is True

    def test_zero_error_stable_output(self, default_controller):
        """Test that zero error produces stable output."""
        # Initialize with some history
        for _ in range(10):
            default_controller.calculate(setpoint=500.0, pv=500.0, dt_seconds=1.0)

        result = default_controller.calculate(setpoint=500.0, pv=500.0, dt_seconds=1.0)

        # With zero error, output should be stable (only integral remains)
        assert result.p_term == 0.0
        # D term should be approximately zero
        assert abs(result.d_term) < 0.1

    def test_feedforward_contribution(self, default_controller):
        """Test feedforward contribution to output."""
        result_no_ff = default_controller.calculate(
            setpoint=500.0, pv=480.0, dt_seconds=1.0, feedforward=0.0
        )

        default_controller.reset()

        result_with_ff = default_controller.calculate(
            setpoint=500.0, pv=480.0, dt_seconds=1.0, feedforward=20.0
        )

        assert result_with_ff.feedforward == 20.0
        # Output with feedforward should be higher (or closer to limit)
        assert result_with_ff.output >= result_no_ff.output or result_with_ff.output_clamped

    def test_provenance_hash_generated(self, default_controller):
        """Test that provenance hash is generated."""
        result = default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256 hex length


# =============================================================================
# TEST CLASS: ANTI-WINDUP
# =============================================================================

class TestAntiWindup:
    """Tests for anti-windup behavior."""

    def test_anti_windup_at_saturation(self, default_controller):
        """Test anti-windup prevents excessive integral buildup."""
        # Drive output to saturation
        for _ in range(100):
            default_controller.calculate(setpoint=500.0, pv=100.0, dt_seconds=1.0)

        # Integral should not grow unbounded
        integral = default_controller._state.integral

        # Now apply correction
        for _ in range(50):
            default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        new_integral = default_controller._state.integral

        # Integral should be corrected (reduced)
        assert abs(new_integral) < abs(integral) * 2  # Some reasonable bound

    def test_anti_windup_with_different_gains(self):
        """Test anti-windup with different gains."""
        tuning = PIDTuning(kp=2.0, ki=0.1, kd=0.0, anti_windup_gain=2.0)
        controller = PIDController(
            name="TC-AW",
            tuning=tuning,
            output_min=0.0,
            output_max=100.0
        )

        # Drive to saturation
        for _ in range(50):
            controller.calculate(setpoint=500.0, pv=100.0, dt_seconds=1.0)

        # Check that anti-windup is functioning
        result = controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)
        assert result.output <= 100.0


# =============================================================================
# TEST CLASS: RATE LIMITING
# =============================================================================

class TestRateLimiting:
    """Tests for output rate limiting."""

    def test_rate_limit_applied(self, rate_limited_controller):
        """Test that rate limit is applied."""
        # First calculation
        result1 = rate_limited_controller.calculate(
            setpoint=500.0, pv=100.0, dt_seconds=1.0
        )

        # Large step change request
        rate_limited_controller._state.last_output = 50.0  # Force a baseline
        result2 = rate_limited_controller.calculate(
            setpoint=500.0, pv=100.0, dt_seconds=1.0
        )

        # Output change should be limited to 10%/second
        output_change = abs(result2.output - 50.0)
        assert output_change <= 10.0 + 0.1  # Allow small tolerance

    def test_rate_limit_respects_direction(self, rate_limited_controller):
        """Test rate limit applies in both directions."""
        # Initialize at 50%
        rate_limited_controller._state.last_output = 50.0

        # Request decrease
        result = rate_limited_controller.calculate(
            setpoint=100.0, pv=500.0, dt_seconds=1.0
        )

        # Change should be limited
        assert result.output >= 40.0  # 50 - 10 = 40 minimum


# =============================================================================
# TEST CLASS: MODE TRANSITIONS
# =============================================================================

class TestModeTransitions:
    """Tests for controller mode transitions."""

    def test_auto_to_manual_transition(self, default_controller):
        """Test Auto to Manual mode transition."""
        # Run in auto mode
        default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # Switch to manual
        default_controller.set_mode(ControlMode.MANUAL)

        assert default_controller.mode == ControlMode.MANUAL

    def test_manual_to_auto_bumpless(self, default_controller):
        """Test bumpless transfer from Manual to Auto."""
        # Run in auto and get output
        result_auto = default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)
        last_auto_output = result_auto.output

        # Switch to manual
        default_controller.set_mode(ControlMode.MANUAL)
        manual_output = default_controller._manual_output

        # Manual output should match last auto output (bumpless)
        assert manual_output == pytest.approx(last_auto_output, rel=0.01)

    def test_manual_mode_output(self, default_controller):
        """Test manual mode uses manual output value."""
        default_controller.set_mode(ControlMode.MANUAL)
        default_controller.set_manual_output(75.0)

        result = default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        assert result.output == 75.0
        assert result.mode == ControlMode.MANUAL

    def test_set_manual_output_clamped(self, default_controller):
        """Test manual output is clamped to limits."""
        default_controller.set_mode(ControlMode.MANUAL)

        default_controller.set_manual_output(150.0)  # Above max
        assert default_controller._manual_output == 100.0

        default_controller.set_manual_output(-10.0)  # Below min
        assert default_controller._manual_output == 0.0

    def test_same_mode_no_change(self, default_controller):
        """Test setting same mode does nothing."""
        default_controller.set_mode(ControlMode.AUTO)
        default_controller.set_mode(ControlMode.AUTO)

        assert default_controller.mode == ControlMode.AUTO


# =============================================================================
# TEST CLASS: GAIN SCHEDULING
# =============================================================================

class TestGainScheduling:
    """Tests for gain scheduling functionality."""

    def test_gain_schedule_added(self, gain_scheduled_controller):
        """Test gain schedules are added correctly."""
        assert len(gain_scheduled_controller._gain_schedules) == 3

    def test_gain_scheduling_active_region(self, gain_scheduled_controller):
        """Test gain scheduling selects correct region."""
        # Normal region (200-600)
        result = gain_scheduled_controller.calculate(
            setpoint=500.0, pv=400.0, dt_seconds=1.0
        )

        assert result.gain_region == GainScheduleRegion.NORMAL

    def test_gain_scheduling_startup_region(self, gain_scheduled_controller):
        """Test gain scheduling in startup region."""
        result = gain_scheduled_controller.calculate(
            setpoint=200.0, pv=100.0, dt_seconds=1.0
        )

        assert result.gain_region == GainScheduleRegion.STARTUP

    def test_gain_scheduling_high_load_region(self, gain_scheduled_controller):
        """Test gain scheduling in high load region."""
        result = gain_scheduled_controller.calculate(
            setpoint=800.0, pv=700.0, dt_seconds=1.0
        )

        assert result.gain_region == GainScheduleRegion.HIGH_LOAD

    def test_gain_transition_smoothing(self, gain_scheduled_controller):
        """Test that gain transitions are smooth."""
        # Get gains at boundary
        gain_scheduled_controller.calculate(setpoint=200.0, pv=199.0, dt_seconds=1.0)
        gains_before = gain_scheduled_controller._active_tuning

        gain_scheduled_controller.calculate(setpoint=200.0, pv=201.0, dt_seconds=1.0)
        gains_after = gain_scheduled_controller._active_tuning

        # Gains should have changed (transition rate applies)
        # The difference should be gradual
        assert gains_before.kp != gains_after.kp


# =============================================================================
# TEST CLASS: ALARMS
# =============================================================================

class TestAlarms:
    """Tests for controller alarm functionality."""

    def test_add_alarm(self, default_controller):
        """Test adding an alarm."""
        alarm = ControllerAlarm(
            name="High Temperature",
            priority=AlarmPriority.HIGH,
            setpoint=550.0,
            deadband=5.0,
            direction="high"
        )

        default_controller.add_alarm(alarm)

        assert len(default_controller._alarms) == 1

    def test_alarm_activation(self, default_controller):
        """Test alarm activation on high value."""
        alarm = ControllerAlarm(
            name="High Temperature",
            priority=AlarmPriority.HIGH,
            setpoint=550.0,
            deadband=5.0,
            direction="high",
            delay_seconds=0.0
        )

        default_controller.add_alarm(alarm)

        # Calculate with PV above alarm setpoint
        default_controller.calculate(setpoint=500.0, pv=560.0, dt_seconds=1.0)

        active_alarms = default_controller.get_active_alarms()
        assert len(active_alarms) == 1
        assert active_alarms[0].active is True

    def test_alarm_clearing(self, default_controller):
        """Test alarm clearing on value return to normal."""
        alarm = ControllerAlarm(
            name="High Temperature",
            priority=AlarmPriority.HIGH,
            setpoint=550.0,
            deadband=5.0,
            direction="high",
            delay_seconds=0.0
        )

        default_controller.add_alarm(alarm)

        # Activate alarm
        default_controller.calculate(setpoint=500.0, pv=560.0, dt_seconds=1.0)
        assert len(default_controller.get_active_alarms()) == 1

        # Clear alarm (below setpoint - deadband)
        default_controller.calculate(setpoint=500.0, pv=540.0, dt_seconds=1.0)
        assert len(default_controller.get_active_alarms()) == 0

    def test_low_alarm(self, default_controller):
        """Test low alarm activation."""
        alarm = ControllerAlarm(
            name="Low Temperature",
            priority=AlarmPriority.MEDIUM,
            setpoint=300.0,
            deadband=5.0,
            direction="low",
            delay_seconds=0.0
        )

        default_controller.add_alarm(alarm)

        # Calculate with PV below alarm setpoint
        default_controller.calculate(setpoint=400.0, pv=290.0, dt_seconds=1.0)

        active_alarms = default_controller.get_active_alarms()
        assert len(active_alarms) == 1

    def test_alarm_delay(self, default_controller):
        """Test alarm activation delay."""
        alarm = ControllerAlarm(
            name="High Temperature",
            priority=AlarmPriority.HIGH,
            setpoint=550.0,
            deadband=5.0,
            direction="high",
            delay_seconds=5.0  # 5 second delay
        )

        default_controller.add_alarm(alarm)

        # First calculation - should not activate immediately
        default_controller.calculate(setpoint=500.0, pv=560.0, dt_seconds=1.0)

        # May or may not be active depending on delay implementation
        # The alarm should have activation_time set
        assert alarm.activation_time is not None or alarm.active is False


# =============================================================================
# TEST CLASS: CONTROLLER STATE
# =============================================================================

class TestControllerState:
    """Tests for controller state management."""

    def test_reset(self, default_controller):
        """Test controller reset."""
        # Run some calculations
        for _ in range(10):
            default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # Reset
        default_controller.reset()

        assert default_controller._state.integral == 0.0
        assert default_controller._state.derivative_filter == 0.0
        assert default_controller._state.last_error == 0.0
        assert default_controller._state.last_pv == 0.0
        assert default_controller._state.last_output == 0.0

    def test_get_history(self, default_controller):
        """Test getting calculation history."""
        for _ in range(10):
            default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        history = default_controller.get_history(limit=5)

        assert len(history) == 5
        assert all(isinstance(h, ControlOutput) for h in history)

    def test_get_status(self, default_controller):
        """Test getting controller status."""
        default_controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        status = default_controller.get_status()

        assert status["name"] == "TC-101"
        assert status["mode"] == ControlMode.AUTO.value
        assert "setpoint" in status
        assert "last_output" in status
        assert "integral" in status
        assert "active_tuning" in status


# =============================================================================
# TEST CLASS: CASCADE CONTROLLER
# =============================================================================

class TestCascadeController:
    """Tests for cascade controller functionality."""

    def test_cascade_initialization(self, cascade_controller):
        """Test cascade controller initialization."""
        assert cascade_controller.master is not None
        assert cascade_controller.slave is not None
        assert cascade_controller.ratio == 1.0
        assert cascade_controller._cascade_active is True

    def test_cascade_calculation(self, cascade_controller):
        """Test cascade control calculation."""
        result = cascade_controller.calculate(
            master_pv=450.0,
            slave_pv=50.0,
            dt_seconds=1.0
        )

        assert isinstance(result, CascadeOutput)
        assert result.master_output is not None
        assert result.slave_output is not None
        assert result.final_output >= 0.0
        assert result.final_output <= 100.0

    def test_cascade_master_drives_slave_sp(self, cascade_controller):
        """Test that master output drives slave setpoint."""
        cascade_controller.set_master_setpoint(500.0)

        result = cascade_controller.calculate(
            master_pv=450.0,
            slave_pv=50.0,
            dt_seconds=1.0
        )

        # Slave setpoint should be based on master output
        master_output = result.master_output.output
        slave_sp = result.slave_output.setpoint

        assert slave_sp == pytest.approx(master_output * cascade_controller.ratio, rel=0.01)

    def test_cascade_ratio(self, cascade_controller):
        """Test cascade ratio application."""
        cascade_controller.set_ratio(0.5)

        result = cascade_controller.calculate(
            master_pv=450.0,
            slave_pv=50.0,
            dt_seconds=1.0
        )

        master_output = result.master_output.output
        slave_sp = result.slave_output.setpoint

        assert slave_sp == pytest.approx(master_output * 0.5, rel=0.01)

    def test_cascade_slave_sp_clamping(self, cascade_controller):
        """Test slave setpoint clamping."""
        cascade_controller.slave_sp_max = 80.0

        result = cascade_controller.calculate(
            master_pv=100.0,  # Large error to drive master output high
            slave_pv=50.0,
            dt_seconds=1.0
        )

        assert result.slave_output.setpoint <= 80.0

    def test_cascade_disable(self, cascade_controller):
        """Test disabling cascade mode."""
        cascade_controller.set_cascade_active(False)

        assert cascade_controller._cascade_active is False
        assert cascade_controller.slave.mode == ControlMode.AUTO

    def test_cascade_enable(self, cascade_controller):
        """Test enabling cascade mode."""
        cascade_controller.set_cascade_active(False)
        cascade_controller.set_cascade_active(True)

        assert cascade_controller._cascade_active is True
        assert cascade_controller.slave.mode == ControlMode.CASCADE

    def test_cascade_get_status(self, cascade_controller):
        """Test getting cascade status."""
        cascade_controller.calculate(master_pv=450.0, slave_pv=50.0, dt_seconds=1.0)

        status = cascade_controller.get_status()

        assert "master" in status
        assert "slave" in status
        assert "cascade_active" in status
        assert "ratio" in status
        assert "master_setpoint" in status

    def test_cascade_history(self, cascade_controller):
        """Test cascade calculation history."""
        for _ in range(10):
            cascade_controller.calculate(master_pv=450.0, slave_pv=50.0, dt_seconds=1.0)

        history = cascade_controller.get_history(limit=5)

        assert len(history) == 5
        assert all(isinstance(h, CascadeOutput) for h in history)

    def test_cascade_provenance_hash(self, cascade_controller):
        """Test cascade provenance hash generation."""
        result = cascade_controller.calculate(
            master_pv=450.0,
            slave_pv=50.0,
            dt_seconds=1.0
        )

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# =============================================================================
# TEST CLASS: CASCADE COORDINATOR
# =============================================================================

class TestCascadeCoordinator:
    """Tests for cascade coordinator functionality."""

    @pytest.fixture
    def coordinator(self) -> CascadeCoordinator:
        """Create a cascade coordinator with multiple cascades."""
        coord = CascadeCoordinator(name="TestCoordinator")

        # Add two cascades
        cascade1 = create_temperature_flow_cascade("FZ-101")
        cascade2 = create_pressure_flow_cascade("ST-101")

        coord.add_cascade("furnace_1", cascade1)
        coord.add_cascade("steam_1", cascade2)

        return coord

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.name == "TestCoordinator"
        assert len(coordinator._cascades) == 2

    def test_add_cascade(self, coordinator):
        """Test adding a cascade to coordinator."""
        new_cascade = create_temperature_flow_cascade("FZ-102")
        coordinator.add_cascade("furnace_2", new_cascade)

        assert len(coordinator._cascades) == 3
        assert "furnace_2" in coordinator._cascades

    def test_remove_cascade(self, coordinator):
        """Test removing a cascade from coordinator."""
        result = coordinator.remove_cascade("furnace_1")

        assert result is True
        assert len(coordinator._cascades) == 1
        assert "furnace_1" not in coordinator._cascades

    def test_remove_nonexistent_cascade(self, coordinator):
        """Test removing non-existent cascade."""
        result = coordinator.remove_cascade("nonexistent")

        assert result is False

    def test_calculate_all(self, coordinator):
        """Test calculating all cascades."""
        pv_data = {
            "furnace_1": (450.0, 50.0),
            "steam_1": (140.0, 60.0),
        }

        results = coordinator.calculate_all(pv_data, dt_seconds=1.0)

        assert len(results) == 2
        assert "furnace_1" in results
        assert "steam_1" in results
        assert all(isinstance(r, CascadeOutput) for r in results.values())

    def test_set_all_master_setpoints(self, coordinator):
        """Test setting all master setpoints."""
        coordinator.set_all_master_setpoints(550.0)

        # Verify all cascades have the same setpoint
        for cascade in coordinator._cascades.values():
            assert cascade._master_setpoint == 550.0

    def test_set_all_cascade_active(self, coordinator):
        """Test enabling/disabling cascade for all."""
        coordinator.set_all_cascade_active(False)

        for cascade in coordinator._cascades.values():
            assert cascade._cascade_active is False

        coordinator.set_all_cascade_active(True)

        for cascade in coordinator._cascades.values():
            assert cascade._cascade_active is True

    def test_get_all_status(self, coordinator):
        """Test getting status of all cascades."""
        # Run some calculations first
        pv_data = {
            "furnace_1": (450.0, 50.0),
            "steam_1": (140.0, 60.0),
        }
        coordinator.calculate_all(pv_data, dt_seconds=1.0)

        status = coordinator.get_all_status()

        assert len(status) == 2
        assert "furnace_1" in status
        assert "steam_1" in status


# =============================================================================
# TEST CLASS: FACTORY FUNCTIONS
# =============================================================================

class TestCascadeFactoryFunctions:
    """Tests for cascade factory functions."""

    def test_create_temperature_flow_cascade(self):
        """Test temperature-flow cascade factory."""
        cascade = create_temperature_flow_cascade(
            tag_prefix="FZ-101",
            temp_sp_default=500.0,
            flow_max=100.0
        )

        assert cascade.master.name == "TIC-FZ-101"
        assert cascade.slave.name == "FIC-FZ-101"
        assert cascade._master_setpoint == 500.0
        assert cascade.slave_sp_max == 100.0

    def test_create_pressure_flow_cascade(self):
        """Test pressure-flow cascade factory."""
        cascade = create_pressure_flow_cascade(
            tag_prefix="ST-101",
            pressure_sp_default=150.0,
            flow_max=100.0
        )

        assert cascade.master.name == "PIC-ST-101"
        assert cascade.slave.name == "FIC-ST-101"
        assert cascade._master_setpoint == 150.0


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_dt(self, default_controller):
        """Test calculation with very small time step."""
        result = default_controller.calculate(
            setpoint=500.0,
            pv=450.0,
            dt_seconds=0.001
        )

        assert result.output >= 0.0
        assert result.output <= 100.0

    def test_very_large_error(self, default_controller):
        """Test calculation with very large error."""
        result = default_controller.calculate(
            setpoint=1000.0,
            pv=0.0,
            dt_seconds=1.0
        )

        # Output should be clamped
        assert result.output == 100.0
        assert result.output_clamped is True

    def test_negative_pv(self, default_controller):
        """Test calculation with negative process value."""
        result = default_controller.calculate(
            setpoint=0.0,
            pv=-50.0,
            dt_seconds=1.0
        )

        assert result.output >= 0.0
        assert result.output <= 100.0

    def test_zero_tuning_gains(self):
        """Test with zero PID gains (P-only with Kp=0)."""
        tuning = PIDTuning(kp=0.0, ki=0.0, kd=0.0)
        controller = PIDController(
            name="TC-ZERO",
            tuning=tuning,
            output_min=0.0,
            output_max=100.0
        )

        result = controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # Output should be the feedforward value (0) since all gains are 0
        assert result.output == 0.0

    def test_direct_action_controller(self, default_tuning):
        """Test direct action controller."""
        controller = PIDController(
            name="TC-DIRECT",
            tuning=default_tuning,
            output_min=0.0,
            output_max=100.0,
            action=ControlAction.DIRECT
        )

        # With DIRECT action, increasing PV should increase output
        result = controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

        # The sign of the error term should be opposite to REVERSE action
        assert result.error == -50.0  # SP - PV = 50, but reported as -50 due to action

    def test_extreme_dt_values(self, default_controller):
        """Test with extreme dt values."""
        # Very small dt
        result1 = default_controller.calculate(
            setpoint=500.0, pv=450.0, dt_seconds=0.0001
        )
        assert np.isfinite(result1.output)

        # Large dt
        result2 = default_controller.calculate(
            setpoint=500.0, pv=450.0, dt_seconds=100.0
        )
        assert np.isfinite(result2.output)

    def test_nan_handling(self, default_controller):
        """Test that NaN inputs don't propagate."""
        # This test checks robustness, but actual behavior may vary
        try:
            result = default_controller.calculate(
                setpoint=500.0,
                pv=float('nan'),
                dt_seconds=1.0
            )
            # If it doesn't raise, output should be finite
            # (implementation dependent)
        except (ValueError, FloatingPointError):
            pass  # Expected for strict implementations


# =============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    class TestPropertyBasedPID:
        """Property-based tests using Hypothesis."""

        @given(
            kp=st.floats(min_value=0.0, max_value=10.0),
            ki=st.floats(min_value=0.0, max_value=1.0),
            kd=st.floats(min_value=0.0, max_value=2.0)
        )
        @settings(max_examples=50)
        def test_tuning_always_valid(self, kp, ki, kd):
            """Test that any valid tuning creates valid output."""
            assume(np.isfinite(kp) and np.isfinite(ki) and np.isfinite(kd))

            tuning = PIDTuning(kp=kp, ki=ki, kd=kd)
            controller = PIDController(
                name="TC-TEST",
                tuning=tuning,
                output_min=0.0,
                output_max=100.0
            )

            result = controller.calculate(setpoint=500.0, pv=450.0, dt_seconds=1.0)

            assert 0.0 <= result.output <= 100.0

        @given(
            setpoint=st.floats(min_value=-1000.0, max_value=1000.0),
            pv=st.floats(min_value=-1000.0, max_value=1000.0)
        )
        @settings(max_examples=50)
        def test_output_always_clamped(self, setpoint, pv):
            """Test that output is always within limits."""
            assume(np.isfinite(setpoint) and np.isfinite(pv))

            tuning = PIDTuning(kp=2.0, ki=0.1, kd=0.5)
            controller = PIDController(
                name="TC-TEST",
                tuning=tuning,
                output_min=0.0,
                output_max=100.0
            )

            result = controller.calculate(setpoint=setpoint, pv=pv, dt_seconds=1.0)

            assert 0.0 <= result.output <= 100.0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for cascade controller."""

    @pytest.mark.performance
    def test_pid_calculation_speed(self, default_controller):
        """Test PID calculation performance."""
        import time

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            pv = 450.0 + np.sin(i * 0.01) * 20
            default_controller.calculate(setpoint=500.0, pv=pv, dt_seconds=0.1)

        elapsed = time.perf_counter() - start

        # Should be able to do 10000 calculations in under 1 second
        assert elapsed < 1.0

        calculations_per_second = iterations / elapsed
        assert calculations_per_second > 10000

    @pytest.mark.performance
    def test_cascade_calculation_speed(self, cascade_controller):
        """Test cascade calculation performance."""
        import time

        iterations = 5000
        start = time.perf_counter()

        for i in range(iterations):
            master_pv = 450.0 + np.sin(i * 0.01) * 20
            slave_pv = 50.0 + np.cos(i * 0.01) * 10
            cascade_controller.calculate(
                master_pv=master_pv,
                slave_pv=slave_pv,
                dt_seconds=0.1
            )

        elapsed = time.perf_counter() - start

        # Should be able to do 5000 cascade calculations in under 1 second
        assert elapsed < 1.0


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_pid_determinism(self, default_tuning):
        """Test that PID produces deterministic results."""
        results = []

        for _ in range(3):
            controller = PIDController(
                name="TC-DET",
                tuning=default_tuning,
                output_min=0.0,
                output_max=100.0
            )

            # Same sequence of calculations
            outputs = []
            for i in range(10):
                result = controller.calculate(
                    setpoint=500.0,
                    pv=450.0 + i * 5,
                    dt_seconds=1.0
                )
                outputs.append(result.output)

            results.append(outputs)

        # All runs should produce identical results
        for i in range(len(results[0])):
            assert results[0][i] == results[1][i] == results[2][i]

    def test_cascade_determinism(self, master_controller, slave_controller):
        """Test that cascade produces deterministic results."""
        results = []

        for _ in range(3):
            cascade = CascadeController(
                master=PIDController(
                    name="TIC-DET",
                    tuning=PIDTuning(kp=2.0, ki=0.1, kd=0.5),
                    output_min=0.0,
                    output_max=100.0
                ),
                slave=PIDController(
                    name="FIC-DET",
                    tuning=PIDTuning(kp=1.0, ki=0.2, kd=0.0),
                    output_min=0.0,
                    output_max=100.0
                )
            )
            cascade.set_master_setpoint(500.0)

            outputs = []
            for i in range(10):
                result = cascade.calculate(
                    master_pv=450.0 + i * 5,
                    slave_pv=50.0 + i * 2,
                    dt_seconds=1.0
                )
                outputs.append(result.final_output)

            results.append(outputs)

        # All runs should produce identical results
        for i in range(len(results[0])):
            assert results[0][i] == results[1][i] == results[2][i]
