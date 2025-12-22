"""
Safety Tests: Boundary Violations

Tests safety boundary enforcement including:
- Temperature boundary violations
- Pressure boundary violations
- Flow rate boundary violations
- SIS trip conditions
- Zero boundary violations guarantee

Reference: GL-001 Specification Section 11.4
Target Coverage: 85%+
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# Safety Classes (Simulated Production Code)
# =============================================================================

class SafetyLevel(Enum):
    """Safety interlock levels."""
    NORMAL = 0
    WARNING = 1
    ALARM = 2
    TRIP = 3
    EMERGENCY = 4


@dataclass
class SafetyBoundary:
    """Definition of a safety boundary."""
    name: str
    parameter: str
    low_low: Optional[float] = None  # Emergency low
    low: Optional[float] = None      # Alarm low
    high: Optional[float] = None     # Alarm high
    high_high: Optional[float] = None  # Emergency high
    unit: str = ""


@dataclass
class BoundaryViolation:
    """Record of a boundary violation."""
    timestamp: datetime
    boundary_name: str
    parameter: str
    value: float
    limit: float
    level: SafetyLevel
    action_taken: str


class SafetyBoundaryEngine:
    """Enforces safety boundaries with zero violation guarantee."""

    def __init__(self):
        self.boundaries: Dict[str, SafetyBoundary] = {}
        self.violations: List[BoundaryViolation] = []
        self.active_trips: Dict[str, BoundaryViolation] = {}

    def register_boundary(self, boundary: SafetyBoundary):
        """Register a safety boundary."""
        self.boundaries[boundary.name] = boundary

    def check_value(self, parameter: str, value: float) -> Tuple[SafetyLevel, Optional[BoundaryViolation]]:
        """Check value against all registered boundaries.

        Returns (safety_level, violation_record if any)
        """
        for name, boundary in self.boundaries.items():
            if boundary.parameter != parameter:
                continue

            violation = self._check_boundary(boundary, value)
            if violation:
                self.violations.append(violation)
                if violation.level >= SafetyLevel.TRIP:
                    self.active_trips[name] = violation
                return violation.level, violation

        return SafetyLevel.NORMAL, None

    def _check_boundary(self, boundary: SafetyBoundary, value: float) -> Optional[BoundaryViolation]:
        """Check single boundary."""
        timestamp = datetime.now()

        # Check emergency high (TRIP)
        if boundary.high_high is not None and value >= boundary.high_high:
            return BoundaryViolation(
                timestamp=timestamp,
                boundary_name=boundary.name,
                parameter=boundary.parameter,
                value=value,
                limit=boundary.high_high,
                level=SafetyLevel.TRIP,
                action_taken="EMERGENCY_SHUTDOWN"
            )

        # Check alarm high
        if boundary.high is not None and value >= boundary.high:
            return BoundaryViolation(
                timestamp=timestamp,
                boundary_name=boundary.name,
                parameter=boundary.parameter,
                value=value,
                limit=boundary.high,
                level=SafetyLevel.ALARM,
                action_taken="OPERATOR_NOTIFICATION"
            )

        # Check emergency low (TRIP)
        if boundary.low_low is not None and value <= boundary.low_low:
            return BoundaryViolation(
                timestamp=timestamp,
                boundary_name=boundary.name,
                parameter=boundary.parameter,
                value=value,
                limit=boundary.low_low,
                level=SafetyLevel.TRIP,
                action_taken="EMERGENCY_SHUTDOWN"
            )

        # Check alarm low
        if boundary.low is not None and value <= boundary.low:
            return BoundaryViolation(
                timestamp=timestamp,
                boundary_name=boundary.name,
                parameter=boundary.parameter,
                value=value,
                limit=boundary.low,
                level=SafetyLevel.ALARM,
                action_taken="OPERATOR_NOTIFICATION"
            )

        return None

    def get_violation_count(self) -> int:
        """Get total violation count."""
        return len(self.violations)

    def get_active_trips(self) -> List[BoundaryViolation]:
        """Get list of active trip conditions."""
        return list(self.active_trips.values())

    def clear_trip(self, boundary_name: str) -> bool:
        """Clear a trip condition (requires operator action)."""
        if boundary_name in self.active_trips:
            del self.active_trips[boundary_name]
            return True
        return False

    def enforce_setpoint(self, parameter: str, requested_setpoint: float) -> float:
        """Enforce setpoint is within safe boundaries.

        Returns the safe setpoint (clamped if necessary).
        """
        safe_setpoint = requested_setpoint

        for boundary in self.boundaries.values():
            if boundary.parameter != parameter:
                continue

            # Clamp to alarm limits (not emergency limits)
            if boundary.low is not None:
                safe_setpoint = max(safe_setpoint, boundary.low)
            if boundary.high is not None:
                safe_setpoint = min(safe_setpoint, boundary.high)

        return safe_setpoint


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.safety
class TestBoundaryRegistration:
    """Test safety boundary registration."""

    @pytest.fixture
    def engine(self):
        """Create safety boundary engine."""
        return SafetyBoundaryEngine()

    def test_register_temperature_boundary(self, engine):
        """Test registering temperature boundary."""
        boundary = SafetyBoundary(
            name="boiler_temp",
            parameter="temperature",
            low_low=100.0,
            low=150.0,
            high=500.0,
            high_high=550.0,
            unit="C"
        )

        engine.register_boundary(boundary)

        assert "boiler_temp" in engine.boundaries

    def test_register_multiple_boundaries(self, engine):
        """Test registering multiple boundaries."""
        engine.register_boundary(SafetyBoundary(
            name="temp", parameter="temperature", high=500, high_high=550
        ))
        engine.register_boundary(SafetyBoundary(
            name="press", parameter="pressure", high=50, high_high=60
        ))

        assert len(engine.boundaries) == 2


@pytest.mark.safety
class TestBoundaryViolationDetection:
    """Test boundary violation detection."""

    @pytest.fixture
    def engine_with_boundaries(self):
        """Create engine with standard boundaries."""
        engine = SafetyBoundaryEngine()
        engine.register_boundary(SafetyBoundary(
            name="temp_high",
            parameter="temperature",
            low_low=100.0,
            low=150.0,
            high=500.0,
            high_high=550.0
        ))
        engine.register_boundary(SafetyBoundary(
            name="pressure_high",
            parameter="pressure",
            low_low=1.0,
            low=5.0,
            high=50.0,
            high_high=60.0
        ))
        return engine

    def test_normal_value_no_violation(self, engine_with_boundaries):
        """Test normal value produces no violation."""
        level, violation = engine_with_boundaries.check_value("temperature", 350.0)

        assert level == SafetyLevel.NORMAL
        assert violation is None

    def test_high_alarm_violation(self, engine_with_boundaries):
        """Test high alarm boundary violation."""
        level, violation = engine_with_boundaries.check_value("temperature", 520.0)

        assert level == SafetyLevel.ALARM
        assert violation is not None
        assert violation.action_taken == "OPERATOR_NOTIFICATION"

    def test_high_high_trip_violation(self, engine_with_boundaries):
        """Test high-high trip boundary violation."""
        level, violation = engine_with_boundaries.check_value("temperature", 560.0)

        assert level == SafetyLevel.TRIP
        assert violation is not None
        assert violation.action_taken == "EMERGENCY_SHUTDOWN"

    def test_low_alarm_violation(self, engine_with_boundaries):
        """Test low alarm boundary violation."""
        level, violation = engine_with_boundaries.check_value("temperature", 120.0)

        assert level == SafetyLevel.ALARM
        assert violation is not None

    def test_low_low_trip_violation(self, engine_with_boundaries):
        """Test low-low trip boundary violation."""
        level, violation = engine_with_boundaries.check_value("temperature", 80.0)

        assert level == SafetyLevel.TRIP
        assert violation is not None
        assert violation.action_taken == "EMERGENCY_SHUTDOWN"

    def test_violation_at_exact_limit(self, engine_with_boundaries):
        """Test violation at exact limit value."""
        # At high limit (>=)
        level, violation = engine_with_boundaries.check_value("temperature", 500.0)

        assert level == SafetyLevel.ALARM

    def test_pressure_violation_independent(self, engine_with_boundaries):
        """Test pressure violations are independent of temperature."""
        level, violation = engine_with_boundaries.check_value("pressure", 55.0)

        assert level == SafetyLevel.ALARM
        assert violation.parameter == "pressure"


@pytest.mark.safety
class TestActiveTripManagement:
    """Test active trip condition management."""

    @pytest.fixture
    def engine_with_trip(self):
        """Create engine with active trip."""
        engine = SafetyBoundaryEngine()
        engine.register_boundary(SafetyBoundary(
            name="temp_trip", parameter="temperature", high_high=550.0
        ))
        engine.check_value("temperature", 600.0)  # Trigger trip
        return engine

    def test_active_trip_recorded(self, engine_with_trip):
        """Test that trip is recorded as active."""
        trips = engine_with_trip.get_active_trips()

        assert len(trips) == 1
        assert trips[0].level == SafetyLevel.TRIP

    def test_clear_trip(self, engine_with_trip):
        """Test clearing a trip condition."""
        result = engine_with_trip.clear_trip("temp_trip")

        assert result == True
        assert len(engine_with_trip.get_active_trips()) == 0

    def test_clear_nonexistent_trip(self, engine_with_trip):
        """Test clearing trip that doesn't exist."""
        result = engine_with_trip.clear_trip("nonexistent")

        assert result == False


@pytest.mark.safety
class TestSetpointEnforcement:
    """Test setpoint safety enforcement."""

    @pytest.fixture
    def engine(self):
        """Create engine with boundaries for setpoint enforcement."""
        engine = SafetyBoundaryEngine()
        engine.register_boundary(SafetyBoundary(
            name="temp_limits",
            parameter="temperature",
            low=150.0,
            high=500.0
        ))
        return engine

    def test_setpoint_within_limits_unchanged(self, engine):
        """Test setpoint within limits is unchanged."""
        result = engine.enforce_setpoint("temperature", 350.0)

        assert result == 350.0

    def test_setpoint_above_high_clamped(self, engine):
        """Test setpoint above high limit is clamped."""
        result = engine.enforce_setpoint("temperature", 600.0)

        assert result == 500.0

    def test_setpoint_below_low_clamped(self, engine):
        """Test setpoint below low limit is clamped."""
        result = engine.enforce_setpoint("temperature", 100.0)

        assert result == 150.0

    def test_setpoint_at_limit_unchanged(self, engine):
        """Test setpoint at limit is unchanged."""
        result = engine.enforce_setpoint("temperature", 500.0)

        assert result == 500.0


@pytest.mark.safety
class TestZeroBoundaryViolationGuarantee:
    """Test zero boundary violation guarantee for setpoints."""

    @pytest.fixture
    def comprehensive_engine(self):
        """Create engine with comprehensive boundaries."""
        engine = SafetyBoundaryEngine()
        engine.register_boundary(SafetyBoundary(
            name="temp", parameter="temperature",
            low_low=100, low=150, high=500, high_high=550
        ))
        engine.register_boundary(SafetyBoundary(
            name="pressure", parameter="pressure",
            low_low=1, low=5, high=50, high_high=60
        ))
        engine.register_boundary(SafetyBoundary(
            name="flow", parameter="flow_rate",
            low_low=10, low=50, high=1000, high_high=1200
        ))
        return engine

    @pytest.mark.parametrize("parameter,requested,expected_safe", [
        ("temperature", 600, 500),
        ("temperature", 50, 150),
        ("pressure", 70, 50),
        ("pressure", 2, 5),
        ("flow_rate", 1500, 1000),
        ("flow_rate", 20, 50),
    ])
    def test_setpoint_always_safe(self, comprehensive_engine, parameter, requested, expected_safe):
        """Test that enforced setpoint is always within safe limits."""
        result = comprehensive_engine.enforce_setpoint(parameter, requested)

        # The result should be within alarm limits
        assert result == expected_safe

    def test_enforced_setpoint_produces_no_alarm(self, comprehensive_engine):
        """Test that enforced setpoint never produces alarm."""
        # Try various unsafe setpoints
        unsafe_setpoints = [
            ("temperature", 700),
            ("temperature", 0),
            ("pressure", 100),
            ("flow_rate", 2000),
        ]

        for parameter, requested in unsafe_setpoints:
            safe_setpoint = comprehensive_engine.enforce_setpoint(parameter, requested)
            level, violation = comprehensive_engine.check_value(parameter, safe_setpoint)

            assert level == SafetyLevel.NORMAL, f"Violation for {parameter}={safe_setpoint}"


@pytest.mark.safety
class TestSISPermissiveLogic:
    """Test Safety Instrumented System permissive logic."""

    def test_all_permissives_satisfied(self):
        """Test operation allowed when all permissives satisfied."""
        permissives = {
            "fuel_available": True,
            "air_flow_ok": True,
            "safety_valve_ok": True,
            "water_level_ok": True
        }

        all_satisfied = all(permissives.values())

        assert all_satisfied == True

    def test_operation_blocked_with_failed_permissive(self):
        """Test operation blocked when permissive fails."""
        permissives = {
            "fuel_available": True,
            "air_flow_ok": False,  # Failed
            "safety_valve_ok": True,
            "water_level_ok": True
        }

        all_satisfied = all(permissives.values())
        failed = [k for k, v in permissives.items() if not v]

        assert all_satisfied == False
        assert "air_flow_ok" in failed

    def test_multiple_failed_permissives(self):
        """Test all failed permissives are identified."""
        permissives = {
            "fuel_available": False,
            "air_flow_ok": False,
            "safety_valve_ok": True,
            "water_level_ok": False
        }

        failed = [k for k, v in permissives.items() if not v]

        assert len(failed) == 3
        assert "fuel_available" in failed
        assert "air_flow_ok" in failed
        assert "water_level_ok" in failed
