"""
Unit Tests: Safety Constraint Validator

Tests the safety constraint validation for steam quality control:
1. Minimum dryness fraction constraints
2. Maximum carryover limits
3. Temperature and pressure bounds
4. Rate-of-change limits
5. Alarm and interlock conditions

Reference: Plant safety specifications and regulatory requirements
Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone


# =============================================================================
# Enumerations
# =============================================================================

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    CRITICAL = "critical"


class ConstraintStatus(Enum):
    """Status of constraint evaluation."""
    OK = "ok"
    VIOLATED = "violated"
    APPROACHING = "approaching"
    UNKNOWN = "unknown"


class ConstraintType(Enum):
    """Types of constraints."""
    MIN_DRYNESS = "min_dryness"
    MAX_CARRYOVER = "max_carryover"
    MIN_PRESSURE = "min_pressure"
    MAX_PRESSURE = "max_pressure"
    MIN_TEMPERATURE = "min_temperature"
    MAX_TEMPERATURE = "max_temperature"
    MAX_RATE_OF_CHANGE = "max_rate_of_change"
    INTERLOCK = "interlock"


# =============================================================================
# Constants
# =============================================================================

# Default safety thresholds
DEFAULT_MIN_DRYNESS = 0.85
DEFAULT_MAX_CARRYOVER_TDS_PPM = 100.0
DEFAULT_MIN_PRESSURE_MPA = 0.1
DEFAULT_MAX_PRESSURE_MPA = 15.0
DEFAULT_MIN_TEMPERATURE_K = 373.15  # 100C
DEFAULT_MAX_TEMPERATURE_K = 600.0

# Rate of change limits
DEFAULT_MAX_PRESSURE_RATE_MPA_MIN = 0.5
DEFAULT_MAX_TEMPERATURE_RATE_K_MIN = 10.0

# Approaching threshold (percentage of limit)
APPROACHING_THRESHOLD = 0.9


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Constraint:
    """Definition of a safety constraint."""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    limit_value: float
    severity: ConstraintSeverity
    unit: str = ""
    enabled: bool = True
    deadband: float = 0.0  # Hysteresis for alarm clearing


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    constraint_id: str
    constraint_type: ConstraintType
    current_value: float
    limit_value: float
    severity: ConstraintSeverity
    status: ConstraintStatus
    deviation_percent: float
    message: str
    timestamp: datetime = None
    provenance_hash: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class SteamQualityMeasurement:
    """Steam quality measurement for constraint checking."""
    dryness_fraction: float
    pressure_mpa: float
    temperature_k: float
    tds_ppm: float = 0.0
    silica_ppb: float = 0.0
    conductivity_us_cm: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ConstraintCheckResult:
    """Result of constraint validation check."""
    all_passed: bool
    violations: List[ConstraintViolation]
    warnings: List[ConstraintViolation]
    info_messages: List[str]
    worst_severity: Optional[ConstraintSeverity]
    provenance_hash: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class RateOfChangeData:
    """Data for rate of change calculation."""
    current_value: float
    previous_value: float
    time_delta_seconds: float


# =============================================================================
# Constraint Validator Implementation
# =============================================================================

class ConstraintValidationError(Exception):
    """Error in constraint validation."""
    pass


def create_default_constraints() -> List[Constraint]:
    """
    Create default set of safety constraints.

    Returns:
        List of Constraint objects
    """
    return [
        Constraint(
            constraint_id="MIN_DRYNESS_ALARM",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Minimum steam dryness fraction",
            limit_value=DEFAULT_MIN_DRYNESS,
            severity=ConstraintSeverity.ALARM,
            unit="fraction",
            deadband=0.02,
        ),
        Constraint(
            constraint_id="MIN_DRYNESS_TRIP",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Critical minimum steam dryness",
            limit_value=0.75,  # Trip at very low quality
            severity=ConstraintSeverity.TRIP,
            unit="fraction",
        ),
        Constraint(
            constraint_id="MAX_CARRYOVER_TDS",
            constraint_type=ConstraintType.MAX_CARRYOVER,
            description="Maximum TDS carryover",
            limit_value=DEFAULT_MAX_CARRYOVER_TDS_PPM,
            severity=ConstraintSeverity.ALARM,
            unit="ppm",
            deadband=5.0,
        ),
        Constraint(
            constraint_id="MAX_PRESSURE",
            constraint_type=ConstraintType.MAX_PRESSURE,
            description="Maximum operating pressure",
            limit_value=DEFAULT_MAX_PRESSURE_MPA,
            severity=ConstraintSeverity.TRIP,
            unit="MPa",
        ),
        Constraint(
            constraint_id="MIN_PRESSURE",
            constraint_type=ConstraintType.MIN_PRESSURE,
            description="Minimum operating pressure",
            limit_value=DEFAULT_MIN_PRESSURE_MPA,
            severity=ConstraintSeverity.WARNING,
            unit="MPa",
        ),
        Constraint(
            constraint_id="MAX_TEMPERATURE",
            constraint_type=ConstraintType.MAX_TEMPERATURE,
            description="Maximum operating temperature",
            limit_value=DEFAULT_MAX_TEMPERATURE_K,
            severity=ConstraintSeverity.TRIP,
            unit="K",
        ),
    ]


def evaluate_constraint(
    constraint: Constraint,
    current_value: float,
    is_minimum: bool = True,
) -> ConstraintViolation:
    """
    Evaluate a single constraint against current value.

    Args:
        constraint: Constraint definition
        current_value: Current measured value
        is_minimum: True if this is a minimum constraint

    Returns:
        ConstraintViolation with evaluation result
    """
    limit = constraint.limit_value

    if is_minimum:
        # For minimum constraints, violation is when value < limit
        is_violated = current_value < limit
        is_approaching = not is_violated and current_value < limit / APPROACHING_THRESHOLD
        deviation = (limit - current_value) / limit * 100 if limit != 0 else 0
    else:
        # For maximum constraints, violation is when value > limit
        is_violated = current_value > limit
        is_approaching = not is_violated and current_value > limit * APPROACHING_THRESHOLD
        deviation = (current_value - limit) / limit * 100 if limit != 0 else 0

    if is_violated:
        status = ConstraintStatus.VIOLATED
        message = f"{constraint.description}: {current_value:.3f} {constraint.unit} violates limit {limit:.3f}"
    elif is_approaching:
        status = ConstraintStatus.APPROACHING
        message = f"{constraint.description}: {current_value:.3f} {constraint.unit} approaching limit {limit:.3f}"
    else:
        status = ConstraintStatus.OK
        message = f"{constraint.description}: OK"
        deviation = 0

    # Calculate provenance hash
    inputs = {
        "constraint_id": constraint.constraint_id,
        "current_value": round(current_value, 10),
        "limit_value": round(limit, 10),
    }
    provenance_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    return ConstraintViolation(
        constraint_id=constraint.constraint_id,
        constraint_type=constraint.constraint_type,
        current_value=current_value,
        limit_value=limit,
        severity=constraint.severity if is_violated or is_approaching else ConstraintSeverity.INFO,
        status=status,
        deviation_percent=deviation,
        message=message,
        provenance_hash=provenance_hash,
    )


def calculate_rate_of_change(data: RateOfChangeData) -> float:
    """
    Calculate rate of change per minute.

    Args:
        data: RateOfChangeData with current and previous values

    Returns:
        Rate of change per minute
    """
    if data.time_delta_seconds <= 0:
        return 0.0

    rate_per_second = (data.current_value - data.previous_value) / data.time_delta_seconds
    rate_per_minute = rate_per_second * 60

    return rate_per_minute


def validate_steam_quality(
    measurement: SteamQualityMeasurement,
    constraints: List[Constraint] = None,
    previous_measurement: Optional[SteamQualityMeasurement] = None,
) -> ConstraintCheckResult:
    """
    Validate steam quality measurement against all constraints.

    Args:
        measurement: Current steam quality measurement
        constraints: List of constraints (uses defaults if None)
        previous_measurement: Previous measurement for rate of change checks

    Returns:
        ConstraintCheckResult with all violations and status
    """
    if constraints is None:
        constraints = create_default_constraints()

    violations = []
    warnings = []
    info_messages = []

    for constraint in constraints:
        if not constraint.enabled:
            continue

        # Get the value to check based on constraint type
        if constraint.constraint_type == ConstraintType.MIN_DRYNESS:
            value = measurement.dryness_fraction
            is_min = True
        elif constraint.constraint_type == ConstraintType.MAX_CARRYOVER:
            value = measurement.tds_ppm
            is_min = False
        elif constraint.constraint_type == ConstraintType.MIN_PRESSURE:
            value = measurement.pressure_mpa
            is_min = True
        elif constraint.constraint_type == ConstraintType.MAX_PRESSURE:
            value = measurement.pressure_mpa
            is_min = False
        elif constraint.constraint_type == ConstraintType.MIN_TEMPERATURE:
            value = measurement.temperature_k
            is_min = True
        elif constraint.constraint_type == ConstraintType.MAX_TEMPERATURE:
            value = measurement.temperature_k
            is_min = False
        else:
            continue  # Skip unknown constraint types

        result = evaluate_constraint(constraint, value, is_min)

        if result.status == ConstraintStatus.VIOLATED:
            violations.append(result)
        elif result.status == ConstraintStatus.APPROACHING:
            warnings.append(result)
        else:
            info_messages.append(result.message)

    # Check rate of change constraints
    if previous_measurement is not None:
        time_delta = (measurement.timestamp - previous_measurement.timestamp).total_seconds()

        if time_delta > 0:
            # Pressure rate of change
            pressure_rate = calculate_rate_of_change(RateOfChangeData(
                current_value=measurement.pressure_mpa,
                previous_value=previous_measurement.pressure_mpa,
                time_delta_seconds=time_delta,
            ))

            if abs(pressure_rate) > DEFAULT_MAX_PRESSURE_RATE_MPA_MIN:
                violations.append(ConstraintViolation(
                    constraint_id="PRESSURE_RATE",
                    constraint_type=ConstraintType.MAX_RATE_OF_CHANGE,
                    current_value=pressure_rate,
                    limit_value=DEFAULT_MAX_PRESSURE_RATE_MPA_MIN,
                    severity=ConstraintSeverity.WARNING,
                    status=ConstraintStatus.VIOLATED,
                    deviation_percent=(abs(pressure_rate) - DEFAULT_MAX_PRESSURE_RATE_MPA_MIN) / DEFAULT_MAX_PRESSURE_RATE_MPA_MIN * 100,
                    message=f"Pressure rate of change {pressure_rate:.3f} MPa/min exceeds limit",
                ))

            # Temperature rate of change
            temp_rate = calculate_rate_of_change(RateOfChangeData(
                current_value=measurement.temperature_k,
                previous_value=previous_measurement.temperature_k,
                time_delta_seconds=time_delta,
            ))

            if abs(temp_rate) > DEFAULT_MAX_TEMPERATURE_RATE_K_MIN:
                violations.append(ConstraintViolation(
                    constraint_id="TEMPERATURE_RATE",
                    constraint_type=ConstraintType.MAX_RATE_OF_CHANGE,
                    current_value=temp_rate,
                    limit_value=DEFAULT_MAX_TEMPERATURE_RATE_K_MIN,
                    severity=ConstraintSeverity.WARNING,
                    status=ConstraintStatus.VIOLATED,
                    deviation_percent=(abs(temp_rate) - DEFAULT_MAX_TEMPERATURE_RATE_K_MIN) / DEFAULT_MAX_TEMPERATURE_RATE_K_MIN * 100,
                    message=f"Temperature rate of change {temp_rate:.3f} K/min exceeds limit",
                ))

    # Determine worst severity
    all_issues = violations + warnings
    if all_issues:
        severity_order = [ConstraintSeverity.INFO, ConstraintSeverity.WARNING,
                         ConstraintSeverity.ALARM, ConstraintSeverity.TRIP, ConstraintSeverity.CRITICAL]
        worst = max(all_issues, key=lambda x: severity_order.index(x.severity))
        worst_severity = worst.severity
    else:
        worst_severity = None

    # Calculate overall provenance hash
    hash_inputs = {
        "measurement": {
            "dryness_fraction": round(measurement.dryness_fraction, 10),
            "pressure_mpa": round(measurement.pressure_mpa, 10),
            "temperature_k": round(measurement.temperature_k, 10),
            "tds_ppm": round(measurement.tds_ppm, 10),
        },
        "n_violations": len(violations),
        "n_warnings": len(warnings),
    }
    provenance_hash = hashlib.sha256(json.dumps(hash_inputs, sort_keys=True).encode()).hexdigest()

    return ConstraintCheckResult(
        all_passed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        info_messages=info_messages,
        worst_severity=worst_severity,
        provenance_hash=provenance_hash,
    )


def check_interlock_conditions(
    measurements: Dict[str, float],
    interlock_rules: Dict[str, Tuple[str, float]],
) -> List[ConstraintViolation]:
    """
    Check interlock conditions based on multiple measurements.

    Args:
        measurements: Dictionary of measurement tag -> value
        interlock_rules: Dictionary of interlock_id -> (condition_expression, limit)

    Returns:
        List of ConstraintViolation for any violated interlocks
    """
    violations = []

    for interlock_id, (condition, limit) in interlock_rules.items():
        try:
            # Simple interlock check (in production, use a proper expression parser)
            # Format: "tag_name operator"
            parts = condition.split()
            if len(parts) != 2:
                continue

            tag, operator = parts

            if tag not in measurements:
                continue

            value = measurements[tag]

            is_violated = False
            if operator == "<" and value < limit:
                is_violated = True
            elif operator == ">" and value > limit:
                is_violated = True
            elif operator == "<=" and value <= limit:
                is_violated = True
            elif operator == ">=" and value >= limit:
                is_violated = True

            if is_violated:
                violations.append(ConstraintViolation(
                    constraint_id=interlock_id,
                    constraint_type=ConstraintType.INTERLOCK,
                    current_value=value,
                    limit_value=limit,
                    severity=ConstraintSeverity.TRIP,
                    status=ConstraintStatus.VIOLATED,
                    deviation_percent=0,
                    message=f"Interlock {interlock_id}: {tag} {operator} {limit}",
                ))

        except Exception:
            continue  # Skip malformed rules

    return violations


# =============================================================================
# Test Classes
# =============================================================================

class TestDefaultConstraints:
    """Tests for default constraint creation."""

    def test_default_constraints_created(self):
        """Test that default constraints are created."""
        constraints = create_default_constraints()

        assert len(constraints) > 0
        assert any(c.constraint_type == ConstraintType.MIN_DRYNESS for c in constraints)
        assert any(c.constraint_type == ConstraintType.MAX_CARRYOVER for c in constraints)
        assert any(c.constraint_type == ConstraintType.MAX_PRESSURE for c in constraints)

    def test_all_constraints_enabled(self):
        """Test that all default constraints are enabled."""
        constraints = create_default_constraints()

        assert all(c.enabled for c in constraints)

    def test_constraints_have_valid_limits(self):
        """Test that all constraints have valid limits."""
        constraints = create_default_constraints()

        for c in constraints:
            assert c.limit_value > 0 or c.constraint_type == ConstraintType.MIN_PRESSURE

    def test_constraints_have_descriptions(self):
        """Test that all constraints have descriptions."""
        constraints = create_default_constraints()

        for c in constraints:
            assert c.description
            assert len(c.description) > 5


class TestConstraintEvaluation:
    """Tests for individual constraint evaluation."""

    @pytest.fixture
    def min_dryness_constraint(self) -> Constraint:
        """Create minimum dryness constraint."""
        return Constraint(
            constraint_id="MIN_DRYNESS_TEST",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Test minimum dryness",
            limit_value=0.85,
            severity=ConstraintSeverity.ALARM,
            unit="fraction",
        )

    @pytest.fixture
    def max_pressure_constraint(self) -> Constraint:
        """Create maximum pressure constraint."""
        return Constraint(
            constraint_id="MAX_PRESSURE_TEST",
            constraint_type=ConstraintType.MAX_PRESSURE,
            description="Test maximum pressure",
            limit_value=10.0,
            severity=ConstraintSeverity.TRIP,
            unit="MPa",
        )

    def test_min_constraint_ok(self, min_dryness_constraint):
        """Test minimum constraint passes when value > limit."""
        result = evaluate_constraint(min_dryness_constraint, 0.95, is_minimum=True)

        assert result.status == ConstraintStatus.OK

    def test_min_constraint_violated(self, min_dryness_constraint):
        """Test minimum constraint violated when value < limit."""
        result = evaluate_constraint(min_dryness_constraint, 0.75, is_minimum=True)

        assert result.status == ConstraintStatus.VIOLATED
        assert result.severity == ConstraintSeverity.ALARM

    def test_min_constraint_approaching(self, min_dryness_constraint):
        """Test minimum constraint approaching when close to limit."""
        # Approaching threshold is 90%, so 0.85 * 0.9 = 0.765
        # Value between 0.765 and 0.85 should be "approaching"
        # Actually, for minimum: approaching when value < limit / 0.9
        # 0.85 / 0.9 = 0.944, so value between 0.85 and 0.944 is approaching
        result = evaluate_constraint(min_dryness_constraint, 0.87, is_minimum=True)

        assert result.status == ConstraintStatus.APPROACHING

    def test_max_constraint_ok(self, max_pressure_constraint):
        """Test maximum constraint passes when value < limit."""
        result = evaluate_constraint(max_pressure_constraint, 8.0, is_minimum=False)

        assert result.status == ConstraintStatus.OK

    def test_max_constraint_violated(self, max_pressure_constraint):
        """Test maximum constraint violated when value > limit."""
        result = evaluate_constraint(max_pressure_constraint, 12.0, is_minimum=False)

        assert result.status == ConstraintStatus.VIOLATED
        assert result.severity == ConstraintSeverity.TRIP

    def test_max_constraint_approaching(self, max_pressure_constraint):
        """Test maximum constraint approaching when close to limit."""
        # Approaching when value > limit * 0.9 = 10.0 * 0.9 = 9.0
        result = evaluate_constraint(max_pressure_constraint, 9.5, is_minimum=False)

        assert result.status == ConstraintStatus.APPROACHING

    def test_deviation_calculated(self, min_dryness_constraint):
        """Test that deviation percentage is calculated."""
        result = evaluate_constraint(min_dryness_constraint, 0.75, is_minimum=True)

        assert result.deviation_percent > 0
        # (0.85 - 0.75) / 0.85 * 100 = 11.76%
        assert result.deviation_percent == pytest.approx(11.76, abs=0.1)

    def test_provenance_hash_generated(self, min_dryness_constraint):
        """Test that provenance hash is generated."""
        result = evaluate_constraint(min_dryness_constraint, 0.9, is_minimum=True)

        assert result.provenance_hash
        assert len(result.provenance_hash) == 64


class TestSteamQualityValidation:
    """Tests for steam quality validation."""

    @pytest.fixture
    def valid_measurement(self) -> SteamQualityMeasurement:
        """Create valid steam quality measurement."""
        return SteamQualityMeasurement(
            dryness_fraction=0.95,
            pressure_mpa=5.0,
            temperature_k=500.0,
            tds_ppm=20.0,
        )

    @pytest.fixture
    def low_quality_measurement(self) -> SteamQualityMeasurement:
        """Create low quality measurement (should violate)."""
        return SteamQualityMeasurement(
            dryness_fraction=0.70,  # Below minimum
            pressure_mpa=5.0,
            temperature_k=500.0,
            tds_ppm=20.0,
        )

    @pytest.fixture
    def high_carryover_measurement(self) -> SteamQualityMeasurement:
        """Create high carryover measurement (should violate)."""
        return SteamQualityMeasurement(
            dryness_fraction=0.95,
            pressure_mpa=5.0,
            temperature_k=500.0,
            tds_ppm=150.0,  # Above maximum
        )

    def test_valid_measurement_passes(self, valid_measurement):
        """Test that valid measurement passes all constraints."""
        result = validate_steam_quality(valid_measurement)

        assert result.all_passed
        assert len(result.violations) == 0

    def test_low_quality_violates(self, low_quality_measurement):
        """Test that low quality measurement violates constraints."""
        result = validate_steam_quality(low_quality_measurement)

        assert not result.all_passed
        assert len(result.violations) > 0
        assert any(v.constraint_type == ConstraintType.MIN_DRYNESS for v in result.violations)

    def test_high_carryover_violates(self, high_carryover_measurement):
        """Test that high carryover measurement violates constraints."""
        result = validate_steam_quality(high_carryover_measurement)

        assert not result.all_passed
        assert any(v.constraint_type == ConstraintType.MAX_CARRYOVER for v in result.violations)

    def test_worst_severity_determined(self, low_quality_measurement):
        """Test that worst severity is correctly determined."""
        result = validate_steam_quality(low_quality_measurement)

        assert result.worst_severity is not None
        # Should include TRIP severity from MIN_DRYNESS_TRIP constraint
        assert result.worst_severity in [ConstraintSeverity.ALARM, ConstraintSeverity.TRIP]

    def test_provenance_hash_generated(self, valid_measurement):
        """Test that provenance hash is generated."""
        result = validate_steam_quality(valid_measurement)

        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, valid_measurement):
        """Test that provenance hash is deterministic."""
        result1 = validate_steam_quality(valid_measurement)
        result2 = validate_steam_quality(valid_measurement)

        assert result1.provenance_hash == result2.provenance_hash


class TestRateOfChange:
    """Tests for rate of change constraint checking."""

    def test_rate_of_change_calculation(self):
        """Test rate of change calculation."""
        data = RateOfChangeData(
            current_value=10.0,
            previous_value=5.0,
            time_delta_seconds=60.0,  # 1 minute
        )

        rate = calculate_rate_of_change(data)

        assert rate == pytest.approx(5.0)  # 5 units/minute

    def test_zero_time_delta_returns_zero(self):
        """Test that zero time delta returns zero rate."""
        data = RateOfChangeData(
            current_value=10.0,
            previous_value=5.0,
            time_delta_seconds=0.0,
        )

        rate = calculate_rate_of_change(data)

        assert rate == 0.0

    def test_negative_time_delta_returns_zero(self):
        """Test that negative time delta returns zero rate."""
        data = RateOfChangeData(
            current_value=10.0,
            previous_value=5.0,
            time_delta_seconds=-60.0,
        )

        rate = calculate_rate_of_change(data)

        assert rate == 0.0

    def test_rapid_pressure_change_detected(self):
        """Test that rapid pressure change is detected."""
        base_time = datetime.now(timezone.utc)

        previous = SteamQualityMeasurement(
            dryness_fraction=0.95,
            pressure_mpa=5.0,
            temperature_k=500.0,
            timestamp=base_time - timedelta(seconds=60),
        )

        current = SteamQualityMeasurement(
            dryness_fraction=0.95,
            pressure_mpa=6.0,  # 1 MPa change in 1 minute
            temperature_k=500.0,
            timestamp=base_time,
        )

        result = validate_steam_quality(current, previous_measurement=previous)

        # Should have a rate of change violation
        rate_violations = [v for v in result.violations
                          if v.constraint_type == ConstraintType.MAX_RATE_OF_CHANGE]
        assert len(rate_violations) > 0


class TestInterlockConditions:
    """Tests for interlock condition checking."""

    def test_interlock_not_violated(self):
        """Test interlock passes when condition not met."""
        measurements = {"pressure": 5.0, "temperature": 500.0}
        rules = {"INTERLOCK_1": ("pressure >", 10.0)}  # pressure > 10

        violations = check_interlock_conditions(measurements, rules)

        assert len(violations) == 0

    def test_interlock_violated(self):
        """Test interlock fails when condition met."""
        measurements = {"pressure": 15.0, "temperature": 500.0}
        rules = {"INTERLOCK_1": ("pressure >", 10.0)}  # pressure > 10

        violations = check_interlock_conditions(measurements, rules)

        assert len(violations) > 0
        assert violations[0].constraint_type == ConstraintType.INTERLOCK

    def test_missing_tag_ignored(self):
        """Test that missing tag in measurements is ignored."""
        measurements = {"pressure": 5.0}
        rules = {"INTERLOCK_1": ("temperature >", 600.0)}  # temperature not in measurements

        violations = check_interlock_conditions(measurements, rules)

        assert len(violations) == 0


class TestConstraintStatus:
    """Tests for constraint status handling."""

    def test_ok_status_for_valid_values(self):
        """Test OK status for valid values."""
        constraint = Constraint(
            constraint_id="TEST",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Test",
            limit_value=0.85,
            severity=ConstraintSeverity.ALARM,
        )

        result = evaluate_constraint(constraint, 0.95, is_minimum=True)

        assert result.status == ConstraintStatus.OK

    def test_violated_status_for_invalid_values(self):
        """Test VIOLATED status for invalid values."""
        constraint = Constraint(
            constraint_id="TEST",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Test",
            limit_value=0.85,
            severity=ConstraintSeverity.ALARM,
        )

        result = evaluate_constraint(constraint, 0.70, is_minimum=True)

        assert result.status == ConstraintStatus.VIOLATED


class TestSeverityLevels:
    """Tests for severity level handling."""

    def test_info_severity_for_ok_status(self):
        """Test INFO severity for OK status."""
        constraint = Constraint(
            constraint_id="TEST",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Test",
            limit_value=0.85,
            severity=ConstraintSeverity.ALARM,
        )

        result = evaluate_constraint(constraint, 0.95, is_minimum=True)

        assert result.severity == ConstraintSeverity.INFO

    def test_constraint_severity_for_violation(self):
        """Test constraint's severity is used for violations."""
        constraint = Constraint(
            constraint_id="TEST",
            constraint_type=ConstraintType.MIN_DRYNESS,
            description="Test",
            limit_value=0.85,
            severity=ConstraintSeverity.TRIP,  # Set to TRIP
        )

        result = evaluate_constraint(constraint, 0.70, is_minimum=True)

        assert result.severity == ConstraintSeverity.TRIP


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_validation_deterministic(self):
        """Test that validation is deterministic."""
        measurement = SteamQualityMeasurement(
            dryness_fraction=0.90,
            pressure_mpa=5.0,
            temperature_k=500.0,
            tds_ppm=50.0,
        )

        results = [validate_steam_quality(measurement) for _ in range(10)]

        first = results[0]
        for result in results[1:]:
            assert result.all_passed == first.all_passed
            assert result.provenance_hash == first.provenance_hash
            assert len(result.violations) == len(first.violations)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_dryness_fraction(self):
        """Test validation with zero dryness fraction."""
        measurement = SteamQualityMeasurement(
            dryness_fraction=0.0,  # Completely liquid
            pressure_mpa=5.0,
            temperature_k=500.0,
        )

        result = validate_steam_quality(measurement)

        assert not result.all_passed  # Should violate minimum dryness

    def test_one_dryness_fraction(self):
        """Test validation with dryness fraction of 1."""
        measurement = SteamQualityMeasurement(
            dryness_fraction=1.0,  # Completely vapor
            pressure_mpa=5.0,
            temperature_k=500.0,
        )

        result = validate_steam_quality(measurement)

        assert result.all_passed

    def test_empty_constraints_list(self):
        """Test validation with empty constraints list."""
        measurement = SteamQualityMeasurement(
            dryness_fraction=0.5,  # Would normally violate
            pressure_mpa=5.0,
            temperature_k=500.0,
        )

        result = validate_steam_quality(measurement, constraints=[])

        assert result.all_passed  # No constraints to violate

    def test_disabled_constraint(self):
        """Test that disabled constraint is ignored."""
        measurement = SteamQualityMeasurement(
            dryness_fraction=0.5,  # Would violate minimum
            pressure_mpa=5.0,
            temperature_k=500.0,
        )

        constraints = [
            Constraint(
                constraint_id="MIN_DRYNESS",
                constraint_type=ConstraintType.MIN_DRYNESS,
                description="Test",
                limit_value=0.85,
                severity=ConstraintSeverity.ALARM,
                enabled=False,  # Disabled
            )
        ]

        result = validate_steam_quality(measurement, constraints=constraints)

        assert result.all_passed  # Disabled constraint not checked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
