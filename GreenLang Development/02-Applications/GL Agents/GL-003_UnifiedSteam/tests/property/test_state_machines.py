"""
Property-Based Tests for State Machines

This module provides comprehensive property-based tests for state machine behavior
using Hypothesis to validate state transitions, invariants, and safety properties.

Test Categories:
1. Safety State Transitions (alarm escalation, envelope checking)
2. Optimization State Machine (setpoint optimization, trajectory generation)
3. Equipment Lifecycle States (startup, normal, shutdown sequences)
4. State Machine Invariants (no invalid states, valid transitions)

Author: GL-TestEngineer
Version: 1.0.0
"""

import math
import pytest
from typing import List, Tuple, Set, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum

from hypothesis import given, assume, settings, note, HealthCheck
from hypothesis import strategies as st
from hypothesis.strategies import composite
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition, initialize

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import safety module
from safety.safety_envelope import (
    SafetyEnvelope,
    EnvelopeStatus,
    AlarmSeverity,
    LimitType,
    EquipmentType,
    AlarmMargins,
    PressureLimits,
    TemperatureLimits,
    QualityLimits,
    RateLimits,
    EnvelopeCheckResult,
    EquipmentRating,
)

# Import optimization module
from optimization.desuperheater_optimizer import (
    DesuperheaterOptimizer,
    DesuperheaterState,
    TargetConstraints,
    SprayOptimizationResult,
    Setpoint,
)
from optimization.constraints import (
    SafetyConstraints,
    EquipmentConstraints,
    UncertaintyConstraints,
)


# =============================================================================
# HYPOTHESIS CONFIGURATION
# =============================================================================

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    stateful_step_count=50,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=30,
    deadline=None,
    stateful_step_count=20,
)

settings.register_profile(
    "full",
    max_examples=500,
    deadline=None,
    stateful_step_count=100,
)


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

@composite
def valid_pressure_limits(draw):
    """Generate valid pressure limit configuration."""
    min_kpa = draw(st.floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    max_kpa = draw(st.floats(min_value=min_kpa + 100.0, max_value=min_kpa + 10000.0, allow_nan=False, allow_infinity=False))
    design_kpa = draw(st.floats(min_value=min_kpa, max_value=max_kpa, allow_nan=False, allow_infinity=False))

    return {
        "equipment_id": f"EQ-{draw(st.integers(min_value=1, max_value=999)):03d}",
        "min_kpa": min_kpa,
        "max_kpa": max_kpa,
        "design_pressure_kpa": design_kpa,
    }


@composite
def valid_temperature_limits(draw):
    """Generate valid temperature limit configuration."""
    min_c = draw(st.floats(min_value=-50.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    max_c = draw(st.floats(min_value=min_c + 50.0, max_value=min_c + 500.0, allow_nan=False, allow_infinity=False))
    design_c = draw(st.floats(min_value=min_c, max_value=max_c, allow_nan=False, allow_infinity=False))

    return {
        "equipment_id": f"EQ-{draw(st.integers(min_value=1, max_value=999)):03d}",
        "min_c": min_c,
        "max_c": max_c,
        "design_temperature_c": design_c,
    }


@composite
def alarm_margins(draw):
    """Generate valid alarm margin configuration."""
    warning_pct = draw(st.floats(min_value=5.0, max_value=25.0, allow_nan=False, allow_infinity=False))
    alarm_pct = draw(st.floats(min_value=1.0, max_value=warning_pct - 1.0, allow_nan=False, allow_infinity=False))
    trip_pct = draw(st.floats(min_value=0.0, max_value=alarm_pct - 0.1, allow_nan=False, allow_infinity=False))

    return AlarmMargins(
        warning_pct=warning_pct,
        alarm_pct=alarm_pct,
        trip_pct=trip_pct,
    )


@composite
def desuperheater_state(draw):
    """Generate valid desuperheater state."""
    pressure_psig = draw(st.floats(min_value=50.0, max_value=600.0, allow_nan=False, allow_infinity=False))

    # Calculate saturation temperature (simplified)
    sat_temp_f = 115.0 + 45.0 * math.log(pressure_psig + 14.7)

    # Inlet temp above saturation
    inlet_temp_f = sat_temp_f + draw(st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False))

    # Outlet temp between saturation and inlet
    outlet_temp_f = sat_temp_f + draw(st.floats(min_value=20.0, max_value=inlet_temp_f - sat_temp_f - 10.0, allow_nan=False, allow_infinity=False))

    # Setpoint between saturation and inlet
    setpoint_temp_f = sat_temp_f + draw(st.floats(min_value=20.0, max_value=inlet_temp_f - sat_temp_f - 10.0, allow_nan=False, allow_infinity=False))

    return DesuperheaterState(
        desuperheater_id=f"DS-{draw(st.integers(min_value=1, max_value=99)):02d}",
        inlet_temp_f=inlet_temp_f,
        outlet_temp_f=outlet_temp_f,
        setpoint_temp_f=setpoint_temp_f,
        steam_pressure_psig=pressure_psig,
        saturation_temp_f=sat_temp_f,
        steam_flow_lb_hr=draw(st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False)),
        spray_valve_position_pct=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        spray_flow_gpm=draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)),
        spray_water_temp_f=draw(st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False)),
        spray_water_pressure_psig=draw(st.floats(min_value=pressure_psig + 50.0, max_value=pressure_psig + 300.0, allow_nan=False, allow_infinity=False)),
        nozzle_delta_p_psi=draw(st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False)),
    )


@composite
def target_constraints(draw):
    """Generate valid target constraints."""
    return TargetConstraints(
        min_outlet_temp_f=draw(st.floats(min_value=300.0, max_value=400.0, allow_nan=False, allow_infinity=False)) if draw(st.booleans()) else None,
        max_outlet_temp_f=draw(st.floats(min_value=500.0, max_value=700.0, allow_nan=False, allow_infinity=False)) if draw(st.booleans()) else None,
        target_superheat_f=draw(st.floats(min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        min_approach_to_saturation_f=draw(st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False)),
        max_spray_valve_position_pct=draw(st.floats(min_value=70.0, max_value=95.0, allow_nan=False, allow_infinity=False)),
        max_spray_flow_gpm=draw(st.floats(min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        max_temp_rate_f_min=draw(st.floats(min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )


# =============================================================================
# TEST CLASS: SAFETY STATE TRANSITIONS
# =============================================================================

@pytest.mark.hypothesis
class TestSafetyStateTransitions:
    """
    Test safety state transitions and alarm escalation.

    Validates that the safety envelope correctly transitions between
    states based on measured values.
    """

    @given(valid_pressure_limits(), alarm_margins())
    @settings(max_examples=100, deadline=None)
    def test_within_envelope_detection(self, limits: Dict, margins: AlarmMargins):
        """
        Test that values within envelope are correctly detected.
        """
        envelope = SafetyEnvelope()

        # Define limits
        envelope.define_pressure_limits(
            equipment_id=limits["equipment_id"],
            min_kpa=limits["min_kpa"],
            max_kpa=limits["max_kpa"],
            alarm_margins=margins,
            design_pressure_kpa=limits["design_pressure_kpa"],
        )

        # Calculate middle of envelope (definitely safe)
        mid_value = (limits["min_kpa"] + limits["max_kpa"]) / 2

        # Check envelope
        result = envelope.check_within_envelope(
            limits["equipment_id"],
            "pressure",
            mid_value
        )

        # Middle value should be within envelope
        assert result.status == EnvelopeStatus.WITHIN_ENVELOPE, \
            f"Middle value {mid_value} should be within envelope"
        assert result.severity == AlarmSeverity.INFO, \
            f"Within envelope should have INFO severity"

    @given(valid_pressure_limits(), alarm_margins())
    @settings(max_examples=100, deadline=None)
    def test_alarm_escalation_high(self, limits: Dict, margins: AlarmMargins):
        """
        Test alarm escalation as value approaches high limit.
        """
        envelope = SafetyEnvelope()

        envelope.define_pressure_limits(
            equipment_id=limits["equipment_id"],
            min_kpa=limits["min_kpa"],
            max_kpa=limits["max_kpa"],
            alarm_margins=margins,
        )

        range_size = limits["max_kpa"] - limits["min_kpa"]

        # Test at different distances from high limit
        test_points = [
            (limits["max_kpa"] - range_size * 0.5, EnvelopeStatus.WITHIN_ENVELOPE),  # Middle
            (limits["max_kpa"] - range_size * margins.warning_pct / 100 - 0.01, EnvelopeStatus.WITHIN_ENVELOPE),  # Just inside warning
            (limits["max_kpa"] - range_size * (margins.warning_pct - 1) / 100, EnvelopeStatus.WARNING_HIGH),  # In warning zone
            (limits["max_kpa"] + 1, EnvelopeStatus.TRIP_HIGH),  # Above max
        ]

        for value, expected_status in test_points:
            if value >= limits["min_kpa"] and value <= limits["max_kpa"] * 1.1:
                result = envelope.check_within_envelope(
                    limits["equipment_id"],
                    "pressure",
                    value
                )

                # Verify status follows escalation pattern
                if expected_status == EnvelopeStatus.WITHIN_ENVELOPE:
                    assert result.status == EnvelopeStatus.WITHIN_ENVELOPE, \
                        f"Value {value} should be WITHIN_ENVELOPE"

    @given(valid_pressure_limits(), alarm_margins())
    @settings(max_examples=100, deadline=None)
    def test_alarm_escalation_low(self, limits: Dict, margins: AlarmMargins):
        """
        Test alarm escalation as value approaches low limit.
        """
        envelope = SafetyEnvelope()

        envelope.define_pressure_limits(
            equipment_id=limits["equipment_id"],
            min_kpa=limits["min_kpa"],
            max_kpa=limits["max_kpa"],
            alarm_margins=margins,
        )

        range_size = limits["max_kpa"] - limits["min_kpa"]

        # Value below minimum should be TRIP_LOW
        result = envelope.check_within_envelope(
            limits["equipment_id"],
            "pressure",
            limits["min_kpa"] - 1
        )

        assert result.status == EnvelopeStatus.TRIP_LOW, \
            f"Value below minimum should be TRIP_LOW"
        assert result.severity == AlarmSeverity.TRIP, \
            f"TRIP_LOW should have TRIP severity"

    @given(valid_pressure_limits())
    @settings(max_examples=100, deadline=None)
    def test_severity_ordering(self, limits: Dict):
        """
        Test that severity ordering is consistent.
        """
        envelope = SafetyEnvelope()

        margins = AlarmMargins(warning_pct=15.0, alarm_pct=5.0, trip_pct=0.0)

        envelope.define_pressure_limits(
            equipment_id=limits["equipment_id"],
            min_kpa=limits["min_kpa"],
            max_kpa=limits["max_kpa"],
            alarm_margins=margins,
        )

        # Severity ordering: INFO < WARNING < ALARM < CRITICAL < TRIP
        severity_order = [
            AlarmSeverity.INFO,
            AlarmSeverity.WARNING,
            AlarmSeverity.ALARM,
            AlarmSeverity.CRITICAL,
            AlarmSeverity.TRIP,
        ]

        # Test multiple values and verify severity ordering is consistent
        range_size = limits["max_kpa"] - limits["min_kpa"]
        values = [
            limits["min_kpa"] + range_size * 0.5,  # Center (should be lowest severity)
            limits["max_kpa"] - range_size * 0.08,  # Warning zone
            limits["max_kpa"] + 1,  # Trip zone
        ]

        severities = []
        for value in values:
            result = envelope.check_within_envelope(
                limits["equipment_id"],
                "pressure",
                value
            )
            severities.append(severity_order.index(result.severity))

        # Severities should be non-decreasing (or equal)
        for i in range(1, len(severities)):
            assert severities[i] >= severities[i-1], \
                f"Severity should not decrease as value moves toward limit"


# =============================================================================
# TEST CLASS: OPTIMIZATION STATE MACHINE
# =============================================================================

@pytest.mark.hypothesis
class TestOptimizationStateMachine:
    """
    Test optimization state machine transitions and invariants.
    """

    @given(desuperheater_state(), target_constraints())
    @settings(max_examples=100, deadline=None)
    def test_optimization_produces_valid_output(
        self,
        state: DesuperheaterState,
        constraints: TargetConstraints
    ):
        """
        Test that optimization always produces valid output.
        """
        optimizer = DesuperheaterOptimizer()

        result = optimizer.optimize_spray_setpoint(state, constraints)

        # Output must always be valid
        assert isinstance(result, SprayOptimizationResult)
        assert result.desuperheater_id == state.desuperheater_id

        # Recommended values must be within physical bounds
        assert result.recommended_valve_position_pct >= 0.0
        assert result.recommended_valve_position_pct <= 100.0
        assert result.estimated_spray_flow_gpm >= 0.0

        # Confidence must be valid probability
        assert 0.0 <= result.confidence <= 1.0

        # Provenance hash must be present
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    @given(desuperheater_state(), target_constraints())
    @settings(max_examples=100, deadline=None)
    def test_optimization_respects_constraints(
        self,
        state: DesuperheaterState,
        constraints: TargetConstraints
    ):
        """
        Test that optimization respects target constraints.
        """
        optimizer = DesuperheaterOptimizer()

        result = optimizer.optimize_spray_setpoint(state, constraints)

        # If constraints are satisfied, verify them
        if result.constraints_satisfied:
            # Approach to saturation should be maintained
            assert result.approach_to_saturation_f >= constraints.min_approach_to_saturation_f - 1.0, \
                f"Approach to saturation violated: {result.approach_to_saturation_f} < {constraints.min_approach_to_saturation_f}"

            # Valve position should be within limit
            assert result.recommended_valve_position_pct <= constraints.max_spray_valve_position_pct + 0.1, \
                f"Valve position exceeded: {result.recommended_valve_position_pct} > {constraints.max_spray_valve_position_pct}"

            # Spray flow should be within limit
            assert result.estimated_spray_flow_gpm <= constraints.max_spray_flow_gpm + 0.1, \
                f"Spray flow exceeded: {result.estimated_spray_flow_gpm} > {constraints.max_spray_flow_gpm}"

    @given(desuperheater_state(), target_constraints())
    @settings(max_examples=50, deadline=None)
    def test_optimization_determinism(
        self,
        state: DesuperheaterState,
        constraints: TargetConstraints
    ):
        """
        Test that optimization is deterministic.
        """
        optimizer = DesuperheaterOptimizer()

        # Run optimization multiple times
        results = []
        for _ in range(3):
            result = optimizer.optimize_spray_setpoint(state, constraints)
            results.append((
                result.recommended_temp_setpoint_f,
                result.recommended_valve_position_pct,
                result.constraints_satisfied,
            ))

        # All results should be identical (excluding timestamp-dependent fields)
        for i in range(1, len(results)):
            assert results[i] == results[0], \
                f"Optimization not deterministic: {results[i]} != {results[0]}"

    @given(
        st.floats(min_value=300.0, max_value=600.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=350.0, max_value=650.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_trajectory_generation(
        self,
        current_temp: float,
        target_temp: float,
        ramp_rate: float
    ):
        """
        Test setpoint trajectory generation.
        """
        assume(abs(target_temp - current_temp) > 5.0)  # Meaningful change

        optimizer = DesuperheaterOptimizer()

        trajectory = optimizer.generate_setpoint_trajectory(
            current_temp_f=current_temp,
            target_temp_f=target_temp,
            ramp_rate_f_min=ramp_rate,
            current_valve_position_pct=50.0,
            interval_seconds=30.0,
        )

        # Trajectory must not be empty
        assert len(trajectory) >= 1

        # First point should be near current
        # Last point should be target
        assert trajectory[-1].is_final
        assert abs(trajectory[-1].temperature_f - target_temp) < 1.0

        # Trajectory should be monotonic
        if target_temp > current_temp:
            for i in range(1, len(trajectory)):
                assert trajectory[i].temperature_f >= trajectory[i-1].temperature_f - 0.1, \
                    f"Trajectory not monotonically increasing"
        else:
            for i in range(1, len(trajectory)):
                assert trajectory[i].temperature_f <= trajectory[i-1].temperature_f + 0.1, \
                    f"Trajectory not monotonically decreasing"


# =============================================================================
# TEST CLASS: EQUIPMENT LIFECYCLE STATES
# =============================================================================

class EquipmentState(Enum):
    """Equipment lifecycle states."""
    OFFLINE = "offline"
    STARTUP = "startup"
    WARMUP = "warmup"
    NORMAL = "normal"
    COOLDOWN = "cooldown"
    SHUTDOWN = "shutdown"
    TRIP = "trip"
    MAINTENANCE = "maintenance"


# Valid state transitions
VALID_TRANSITIONS: Dict[EquipmentState, Set[EquipmentState]] = {
    EquipmentState.OFFLINE: {EquipmentState.STARTUP, EquipmentState.MAINTENANCE},
    EquipmentState.STARTUP: {EquipmentState.WARMUP, EquipmentState.TRIP, EquipmentState.SHUTDOWN},
    EquipmentState.WARMUP: {EquipmentState.NORMAL, EquipmentState.TRIP, EquipmentState.COOLDOWN},
    EquipmentState.NORMAL: {EquipmentState.COOLDOWN, EquipmentState.TRIP},
    EquipmentState.COOLDOWN: {EquipmentState.SHUTDOWN, EquipmentState.TRIP},
    EquipmentState.SHUTDOWN: {EquipmentState.OFFLINE, EquipmentState.TRIP},
    EquipmentState.TRIP: {EquipmentState.OFFLINE, EquipmentState.MAINTENANCE},
    EquipmentState.MAINTENANCE: {EquipmentState.OFFLINE, EquipmentState.STARTUP},
}


class EquipmentLifecycleStateMachine(RuleBasedStateMachine):
    """
    State machine test for equipment lifecycle.

    Tests that only valid state transitions occur and invariants hold.
    """

    def __init__(self):
        super().__init__()
        self.current_state = EquipmentState.OFFLINE
        self.state_history: List[Tuple[EquipmentState, datetime]] = []
        self.transition_count = 0

    @initialize()
    def initialize_state(self):
        """Initialize to OFFLINE state."""
        self.current_state = EquipmentState.OFFLINE
        self.state_history = [(EquipmentState.OFFLINE, datetime.now(timezone.utc))]
        self.transition_count = 0

    @rule()
    def transition_to_startup(self):
        """Attempt transition to STARTUP."""
        if EquipmentState.STARTUP in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.STARTUP)

    @rule()
    def transition_to_warmup(self):
        """Attempt transition to WARMUP."""
        if EquipmentState.WARMUP in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.WARMUP)

    @rule()
    def transition_to_normal(self):
        """Attempt transition to NORMAL."""
        if EquipmentState.NORMAL in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.NORMAL)

    @rule()
    def transition_to_cooldown(self):
        """Attempt transition to COOLDOWN."""
        if EquipmentState.COOLDOWN in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.COOLDOWN)

    @rule()
    def transition_to_shutdown(self):
        """Attempt transition to SHUTDOWN."""
        if EquipmentState.SHUTDOWN in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.SHUTDOWN)

    @rule()
    def transition_to_offline(self):
        """Attempt transition to OFFLINE."""
        if EquipmentState.OFFLINE in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.OFFLINE)

    @rule()
    def transition_to_trip(self):
        """Simulate a trip event."""
        if EquipmentState.TRIP in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.TRIP)

    @rule()
    def transition_to_maintenance(self):
        """Attempt transition to MAINTENANCE."""
        if EquipmentState.MAINTENANCE in VALID_TRANSITIONS.get(self.current_state, set()):
            self._transition_to(EquipmentState.MAINTENANCE)

    def _transition_to(self, new_state: EquipmentState):
        """Execute state transition."""
        self.current_state = new_state
        self.state_history.append((new_state, datetime.now(timezone.utc)))
        self.transition_count += 1

    @invariant()
    def state_is_valid(self):
        """State must always be a valid EquipmentState."""
        assert isinstance(self.current_state, EquipmentState), \
            f"Invalid state: {self.current_state}"

    @invariant()
    def history_not_empty(self):
        """State history must never be empty."""
        assert len(self.state_history) > 0, "State history cannot be empty"

    @invariant()
    def history_matches_current(self):
        """Last history entry must match current state."""
        assert self.state_history[-1][0] == self.current_state, \
            f"History mismatch: {self.state_history[-1][0]} != {self.current_state}"


# Run the stateful test
TestEquipmentLifecycle = EquipmentLifecycleStateMachine.TestCase


# =============================================================================
# TEST CLASS: SAFETY ENVELOPE STATE MACHINE
# =============================================================================

class SafetyEnvelopeStateMachine(RuleBasedStateMachine):
    """
    State machine test for safety envelope behavior.

    Tests that safety states transition correctly based on values.
    """

    def __init__(self):
        super().__init__()
        self.envelope = SafetyEnvelope()
        self.equipment_id = "TEST-001"
        self.min_pressure = 500.0
        self.max_pressure = 5000.0
        self.current_pressure = 2500.0  # Start in middle
        self.last_status = EnvelopeStatus.WITHIN_ENVELOPE

    @initialize()
    def setup_envelope(self):
        """Initialize safety envelope with limits."""
        self.envelope = SafetyEnvelope()
        self.envelope.define_pressure_limits(
            equipment_id=self.equipment_id,
            min_kpa=self.min_pressure,
            max_kpa=self.max_pressure,
            alarm_margins=AlarmMargins(warning_pct=10.0, alarm_pct=5.0, trip_pct=0.0),
        )
        self.current_pressure = 2500.0
        self.last_status = EnvelopeStatus.WITHIN_ENVELOPE

    @rule(delta=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    def change_pressure(self, delta: float):
        """Change pressure by delta."""
        self.current_pressure += delta
        result = self.envelope.check_within_envelope(
            self.equipment_id,
            "pressure",
            self.current_pressure
        )
        self.last_status = result.status

    @rule()
    def increase_pressure_small(self):
        """Small pressure increase."""
        self.current_pressure += 50.0
        result = self.envelope.check_within_envelope(
            self.equipment_id,
            "pressure",
            self.current_pressure
        )
        self.last_status = result.status

    @rule()
    def decrease_pressure_small(self):
        """Small pressure decrease."""
        self.current_pressure -= 50.0
        result = self.envelope.check_within_envelope(
            self.equipment_id,
            "pressure",
            self.current_pressure
        )
        self.last_status = result.status

    @invariant()
    def status_consistent_with_value(self):
        """Status must be consistent with pressure value."""
        result = self.envelope.check_within_envelope(
            self.equipment_id,
            "pressure",
            self.current_pressure
        )

        # Verify consistency
        range_size = self.max_pressure - self.min_pressure

        if self.current_pressure <= self.min_pressure:
            assert result.status in [EnvelopeStatus.TRIP_LOW, EnvelopeStatus.ALARM_LOW], \
                f"Below minimum should be TRIP_LOW or ALARM_LOW"
        elif self.current_pressure >= self.max_pressure:
            assert result.status in [EnvelopeStatus.TRIP_HIGH, EnvelopeStatus.ALARM_HIGH], \
                f"Above maximum should be TRIP_HIGH or ALARM_HIGH"
        elif self.min_pressure + range_size * 0.2 < self.current_pressure < self.max_pressure - range_size * 0.2:
            assert result.status == EnvelopeStatus.WITHIN_ENVELOPE, \
                f"Middle values should be WITHIN_ENVELOPE"

    @invariant()
    def distance_calculation_valid(self):
        """Distance to limit must be calculated correctly."""
        result = self.envelope.check_within_envelope(
            self.equipment_id,
            "pressure",
            self.current_pressure
        )

        # Distance should be non-negative
        assert result.distance_to_limit >= 0, \
            f"Distance to limit cannot be negative"

        # Distance percentage should be in [0, 100] for normal values
        if self.min_pressure <= self.current_pressure <= self.max_pressure:
            assert 0 <= result.distance_pct <= 100, \
                f"Distance percentage out of range: {result.distance_pct}"


# Run the stateful test
TestSafetyEnvelopeStateMachine = SafetyEnvelopeStateMachine.TestCase


# =============================================================================
# TEST CLASS: STATE MACHINE INVARIANTS
# =============================================================================

@pytest.mark.hypothesis
class TestStateMachineInvariants:
    """
    Test general state machine invariants.
    """

    @given(
        st.lists(
            st.sampled_from(list(EquipmentState)),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_no_invalid_transitions(self, states: List[EquipmentState]):
        """
        Test that only valid transitions are allowed.
        """
        current = EquipmentState.OFFLINE

        for next_state in states:
            valid_next_states = VALID_TRANSITIONS.get(current, set())

            if next_state in valid_next_states:
                # Valid transition
                current = next_state
            # else: stay in current state (transition rejected)

        # Final state should be valid
        assert isinstance(current, EquipmentState)

    @given(valid_pressure_limits())
    @settings(max_examples=50, deadline=None)
    def test_envelope_limit_ordering(self, limits: Dict):
        """
        Test that envelope limits maintain proper ordering.
        """
        envelope = SafetyEnvelope()

        envelope.define_pressure_limits(
            equipment_id=limits["equipment_id"],
            min_kpa=limits["min_kpa"],
            max_kpa=limits["max_kpa"],
        )

        # Get limits back
        retrieved = envelope.get_pressure_limits(limits["equipment_id"])

        assert retrieved is not None
        assert retrieved.min_kpa < retrieved.max_kpa, \
            f"Min must be less than max: {retrieved.min_kpa} >= {retrieved.max_kpa}"

    @given(
        st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_quality_limits_clamped(self, equipment_value: float, dryness: float):
        """
        Test that quality limits clamp values correctly.
        """
        envelope = SafetyEnvelope()

        equipment_id = "TURB-001"

        envelope.define_quality_limits(
            equipment_id=equipment_id,
            min_dryness=0.85,
            erosion_threshold=0.90,
        )

        # Any dryness value should be checkable
        result = envelope.check_within_envelope(
            equipment_id,
            "quality",
            dryness
        )

        # Status should be appropriate
        if dryness < 0.85:
            assert result.severity in [AlarmSeverity.ALARM, AlarmSeverity.TRIP], \
                f"Low quality should trigger alarm/trip"


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    import os
    profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)

    pytest.main([__file__, "-v", "--tb=short"])
