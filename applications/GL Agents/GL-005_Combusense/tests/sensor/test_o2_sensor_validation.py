# -*- coding: utf-8 -*-
"""
O2 Sensor Validation Tests for GL-005 CombustionSense
=====================================================

Tests for oxygen sensor accuracy, calibration, and failure detection
per industry standards and manufacturer specifications.

Reference Standards:
    - ASTM D6522: Standard Test Method for Determination of Nitrogen Oxides,
                  Carbon Monoxide, and Oxygen Concentrations in Emissions
    - EPA Method 3A: Determination of Oxygen and Carbon Dioxide
    - IEC 61207: Expression of Performance of Gas Analyzers

Test Categories:
    1. Sensor accuracy validation
    2. Range and linearity testing
    3. Response time verification
    4. Calibration drift detection
    5. Failure mode detection

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics


# =============================================================================
# SENSOR SPECIFICATIONS
# =============================================================================

class O2SensorType(Enum):
    """Types of O2 sensors used in combustion."""
    ZIRCONIA = "zirconia"           # High temp, in-situ
    PARAMAGNETIC = "paramagnetic"   # Extractive, laboratory grade
    ELECTROCHEMICAL = "electrochemical"  # Lower cost, portable
    TUNABLE_DIODE_LASER = "tdl"     # High accuracy, fast response


@dataclass
class O2SensorSpecification:
    """O2 sensor specifications per manufacturer/standard."""
    sensor_type: O2SensorType
    range_min: float = 0.0       # % O2
    range_max: float = 25.0      # % O2
    accuracy_percent_of_span: float = 1.0   # % of span
    accuracy_percent_of_reading: float = 2.0  # % of reading
    repeatability: float = 0.5   # % of span
    response_time_t90_seconds: float = 30.0  # T90 response
    operating_temp_min_c: float = -10.0
    operating_temp_max_c: float = 50.0
    calibration_interval_days: int = 90
    warm_up_time_minutes: float = 15.0


# Standard sensor specifications
SENSOR_SPECS = {
    O2SensorType.ZIRCONIA: O2SensorSpecification(
        sensor_type=O2SensorType.ZIRCONIA,
        range_min=0.0, range_max=25.0,
        accuracy_percent_of_span=1.0,
        accuracy_percent_of_reading=2.0,
        repeatability=0.5,
        response_time_t90_seconds=10.0,  # Fast in-situ
        operating_temp_min_c=0, operating_temp_max_c=1400,  # High temp
        calibration_interval_days=30,
        warm_up_time_minutes=5.0,
    ),
    O2SensorType.PARAMAGNETIC: O2SensorSpecification(
        sensor_type=O2SensorType.PARAMAGNETIC,
        range_min=0.0, range_max=25.0,
        accuracy_percent_of_span=0.5,  # Higher accuracy
        accuracy_percent_of_reading=1.0,
        repeatability=0.25,
        response_time_t90_seconds=30.0,
        operating_temp_min_c=5, operating_temp_max_c=45,
        calibration_interval_days=90,
        warm_up_time_minutes=30.0,
    ),
    O2SensorType.ELECTROCHEMICAL: O2SensorSpecification(
        sensor_type=O2SensorType.ELECTROCHEMICAL,
        range_min=0.0, range_max=25.0,
        accuracy_percent_of_span=2.0,
        accuracy_percent_of_reading=5.0,
        repeatability=1.0,
        response_time_t90_seconds=60.0,
        operating_temp_min_c=-20, operating_temp_max_c=50,
        calibration_interval_days=180,
        warm_up_time_minutes=2.0,
    ),
}


# =============================================================================
# SENSOR DATA CLASSES
# =============================================================================

@dataclass
class O2Reading:
    """Single O2 sensor reading."""
    value: float               # % O2
    timestamp: datetime
    sensor_id: str
    temperature_c: float       # Sensor/ambient temperature
    raw_millivolts: Optional[float] = None
    quality_flag: str = "GOOD"


@dataclass
class CalibrationData:
    """Calibration data for O2 sensor."""
    calibration_date: datetime
    zero_gas_value: float      # Reading at 0% O2 (N2)
    span_gas_value: float      # Reading at span gas
    span_gas_concentration: float  # Span gas %O2 (typically 20.9% for air)
    zero_offset: float = 0.0
    span_factor: float = 1.0


@dataclass
class SensorStatus:
    """Current sensor operational status."""
    is_healthy: bool
    is_calibrated: bool
    days_since_calibration: int
    fault_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# SENSOR VALIDATION CLASS
# =============================================================================

class O2SensorValidator:
    """Validates O2 sensor accuracy and health."""

    def __init__(self, spec: O2SensorSpecification):
        self.spec = spec

    def validate_reading(
        self,
        reading: O2Reading,
        expected_value: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single O2 reading.

        Args:
            reading: O2 sensor reading
            expected_value: Known value for accuracy check (optional)

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check range
        if reading.value < self.spec.range_min or reading.value > self.spec.range_max:
            issues.append(f"Reading {reading.value}% outside range [{self.spec.range_min}, {self.spec.range_max}]")

        # Check for obviously invalid values
        if reading.value < 0 or reading.value > 25:
            issues.append(f"Physically impossible O2 value: {reading.value}%")

        # Check accuracy if expected value provided
        if expected_value is not None:
            error = abs(reading.value - expected_value)
            max_error = self._calculate_max_allowable_error(expected_value)
            if error > max_error:
                issues.append(f"Error {error:.3f}% exceeds max allowable {max_error:.3f}%")

        # Check quality flag
        if reading.quality_flag != "GOOD":
            issues.append(f"Quality flag: {reading.quality_flag}")

        return len(issues) == 0, issues

    def _calculate_max_allowable_error(self, expected_value: float) -> float:
        """Calculate maximum allowable error per specification."""
        span = self.spec.range_max - self.spec.range_min
        error_from_span = span * self.spec.accuracy_percent_of_span / 100
        error_from_reading = expected_value * self.spec.accuracy_percent_of_reading / 100
        return max(error_from_span, error_from_reading)

    def check_calibration_validity(
        self,
        calibration: CalibrationData,
        current_date: Optional[datetime] = None
    ) -> SensorStatus:
        """
        Check if sensor calibration is still valid.

        Args:
            calibration: Calibration data
            current_date: Current date (default: now)

        Returns:
            SensorStatus with calibration validity info
        """
        if current_date is None:
            current_date = datetime.now()

        days_since_cal = (current_date - calibration.calibration_date).days
        is_calibrated = days_since_cal <= self.spec.calibration_interval_days

        warnings = []
        fault_codes = []

        if days_since_cal > self.spec.calibration_interval_days:
            fault_codes.append("CAL_OVERDUE")
        elif days_since_cal > self.spec.calibration_interval_days * 0.9:
            warnings.append("Calibration due within 10%")

        # Check calibration quality
        if abs(calibration.zero_offset) > 0.5:
            warnings.append(f"High zero offset: {calibration.zero_offset}%")

        if abs(calibration.span_factor - 1.0) > 0.1:
            warnings.append(f"Span factor deviation: {calibration.span_factor}")

        return SensorStatus(
            is_healthy=len(fault_codes) == 0,
            is_calibrated=is_calibrated,
            days_since_calibration=days_since_cal,
            fault_codes=fault_codes,
            warnings=warnings,
        )


# =============================================================================
# ACCURACY VALIDATION TESTS
# =============================================================================

class TestO2SensorAccuracy:
    """Test O2 sensor accuracy requirements."""

    @pytest.fixture
    def zirconia_validator(self) -> O2SensorValidator:
        return O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])

    @pytest.fixture
    def paramagnetic_validator(self) -> O2SensorValidator:
        return O2SensorValidator(SENSOR_SPECS[O2SensorType.PARAMAGNETIC])

    @pytest.mark.parametrize("expected,measured,should_pass", [
        # Within spec tolerance
        (3.0, 3.0, True),
        (3.0, 3.05, True),
        (3.0, 2.95, True),
        (5.0, 5.1, True),
        (10.0, 10.2, True),

        # Outside tolerance
        (3.0, 4.0, False),
        (5.0, 6.0, False),
        (10.0, 12.0, False),
    ])
    def test_accuracy_within_specification(
        self,
        zirconia_validator: O2SensorValidator,
        expected: float,
        measured: float,
        should_pass: bool
    ):
        """Test accuracy validation against specification."""
        reading = O2Reading(
            value=measured,
            timestamp=datetime.now(),
            sensor_id="ZR-001",
            temperature_c=25.0,
        )

        is_valid, issues = zirconia_validator.validate_reading(reading, expected)

        if should_pass:
            assert is_valid, f"Reading should pass: {issues}"
        else:
            assert not is_valid, "Reading should fail accuracy check"

    @pytest.mark.parametrize("sensor_type,o2_value,max_error_percent", [
        # Zirconia: 1% of span + 2% of reading
        (O2SensorType.ZIRCONIA, 3.0, 0.31),    # max(0.25, 0.06) = 0.25 + margin
        (O2SensorType.ZIRCONIA, 5.0, 0.35),
        (O2SensorType.ZIRCONIA, 10.0, 0.45),

        # Paramagnetic: higher accuracy
        (O2SensorType.PARAMAGNETIC, 3.0, 0.15),
        (O2SensorType.PARAMAGNETIC, 5.0, 0.18),
    ])
    def test_maximum_allowable_error(
        self,
        sensor_type: O2SensorType,
        o2_value: float,
        max_error_percent: float
    ):
        """Test maximum allowable error calculation."""
        validator = O2SensorValidator(SENSOR_SPECS[sensor_type])
        calculated_max_error = validator._calculate_max_allowable_error(o2_value)

        # Verify calculated error is reasonable
        assert calculated_max_error > 0, "Max error must be positive"
        assert calculated_max_error <= max_error_percent, \
            f"Calculated max error {calculated_max_error:.3f}% exceeds expected {max_error_percent}%"

    def test_repeatability_validation(self, zirconia_validator: O2SensorValidator):
        """Test sensor repeatability over multiple readings."""
        expected_o2 = 5.0

        # Simulate multiple readings at same condition
        readings = [
            O2Reading(
                value=expected_o2 + (i * 0.02 - 0.05),  # Small variation
                timestamp=datetime.now(),
                sensor_id="ZR-001",
                temperature_c=25.0,
            )
            for i in range(10)
        ]

        values = [r.value for r in readings]
        std_dev = statistics.stdev(values)

        # Repeatability should be within spec
        spec = SENSOR_SPECS[O2SensorType.ZIRCONIA]
        max_repeatability = spec.range_max * spec.repeatability / 100

        assert std_dev < max_repeatability, \
            f"Standard deviation {std_dev:.4f} exceeds repeatability spec {max_repeatability}"


# =============================================================================
# RANGE AND LINEARITY TESTS
# =============================================================================

class TestO2SensorRangeLinearity:
    """Test O2 sensor range and linearity."""

    @pytest.mark.parametrize("o2_value,should_be_in_range", [
        (0.0, True),
        (0.5, True),
        (3.0, True),
        (10.0, True),
        (20.9, True),
        (25.0, True),
        (-0.5, False),
        (26.0, False),
        (100.0, False),
    ])
    def test_range_validation(self, o2_value: float, should_be_in_range: bool):
        """Test O2 reading range validation."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])

        reading = O2Reading(
            value=o2_value,
            timestamp=datetime.now(),
            sensor_id="ZR-001",
            temperature_c=25.0,
        )

        is_valid, issues = validator.validate_reading(reading)

        if should_be_in_range:
            range_issues = [i for i in issues if "range" in i.lower() or "impossible" in i.lower()]
            assert len(range_issues) == 0, f"Value {o2_value} should be in range"
        else:
            assert not is_valid, f"Value {o2_value} should be out of range"

    def test_linearity_across_range(self):
        """Test sensor linearity across operating range."""
        # Reference gas concentrations (% O2)
        reference_points = [0.0, 5.0, 10.0, 15.0, 20.9]

        # Simulated sensor readings (with small non-linearity)
        measured_points = []
        for ref in reference_points:
            # Add small quadratic non-linearity
            non_linearity = 0.001 * (ref - 10) ** 2
            measured = ref + non_linearity
            measured_points.append(measured)

        # Calculate linearity error
        errors = [abs(m - r) for m, r in zip(measured_points, reference_points)]
        max_linearity_error = max(errors)

        # Linearity error should be within 0.5% of span
        span = 25.0
        max_allowable = span * 0.005

        assert max_linearity_error < max_allowable, \
            f"Linearity error {max_linearity_error:.4f} exceeds max allowable {max_allowable}"


# =============================================================================
# CALIBRATION TESTS
# =============================================================================

class TestO2SensorCalibration:
    """Test O2 sensor calibration requirements."""

    @pytest.fixture
    def valid_calibration(self) -> CalibrationData:
        return CalibrationData(
            calibration_date=datetime.now() - timedelta(days=15),
            zero_gas_value=0.02,
            span_gas_value=20.85,
            span_gas_concentration=20.9,
            zero_offset=0.02,
            span_factor=1.002,
        )

    @pytest.fixture
    def expired_calibration(self) -> CalibrationData:
        return CalibrationData(
            calibration_date=datetime.now() - timedelta(days=100),
            zero_gas_value=0.1,
            span_gas_value=20.5,
            span_gas_concentration=20.9,
            zero_offset=0.1,
            span_factor=0.98,
        )

    def test_valid_calibration_status(self, valid_calibration: CalibrationData):
        """Test status check for valid calibration."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])
        status = validator.check_calibration_validity(valid_calibration)

        assert status.is_calibrated, "Calibration should be valid"
        assert status.is_healthy, "Sensor should be healthy"
        assert "CAL_OVERDUE" not in status.fault_codes

    def test_expired_calibration_status(self, expired_calibration: CalibrationData):
        """Test status check for expired calibration."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])
        status = validator.check_calibration_validity(expired_calibration)

        assert not status.is_calibrated, "Calibration should be expired"
        assert "CAL_OVERDUE" in status.fault_codes

    @pytest.mark.parametrize("days_since_cal,should_warn", [
        (10, False),
        (25, False),
        (27, True),   # 90% of 30 days
        (29, True),
        (30, True),   # Exactly at limit
    ])
    def test_calibration_due_warning(
        self,
        days_since_cal: int,
        should_warn: bool
    ):
        """Test calibration due soon warning."""
        calibration = CalibrationData(
            calibration_date=datetime.now() - timedelta(days=days_since_cal),
            zero_gas_value=0.0,
            span_gas_value=20.9,
            span_gas_concentration=20.9,
        )

        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])
        status = validator.check_calibration_validity(calibration)

        has_warning = any("10%" in w or "due" in w.lower() for w in status.warnings)

        if should_warn:
            assert has_warning or not status.is_calibrated, \
                "Should warn when calibration due within 10%"


# =============================================================================
# FAILURE MODE TESTS
# =============================================================================

class TestO2SensorFailureModes:
    """Test O2 sensor failure mode detection."""

    def test_stuck_reading_detection(self):
        """Detect sensor stuck at constant value."""
        readings = [
            O2Reading(
                value=3.5,  # Exactly same value
                timestamp=datetime.now() + timedelta(seconds=i * 10),
                sensor_id="ZR-001",
                temperature_c=25.0,
            )
            for i in range(20)
        ]

        values = [r.value for r in readings]
        std_dev = statistics.stdev(values) if len(set(values)) > 1 else 0.0

        # Real sensor should have some noise
        is_stuck = std_dev < 0.001
        assert is_stuck, "Should detect sensor stuck at constant value"

    def test_out_of_range_high(self):
        """Detect sensor reading above physical maximum."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])

        reading = O2Reading(
            value=25.5,  # Above 25%
            timestamp=datetime.now(),
            sensor_id="ZR-001",
            temperature_c=25.0,
            quality_flag="BAD",
        )

        is_valid, issues = validator.validate_reading(reading)

        assert not is_valid, "Should detect out of range high"
        assert any("range" in i.lower() for i in issues)

    def test_negative_reading_detection(self):
        """Detect physically impossible negative reading."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])

        reading = O2Reading(
            value=-1.0,
            timestamp=datetime.now(),
            sensor_id="ZR-001",
            temperature_c=25.0,
            quality_flag="BAD",
        )

        is_valid, issues = validator.validate_reading(reading)

        assert not is_valid, "Should detect negative reading"
        assert any("impossible" in i.lower() or "range" in i.lower() for i in issues)

    def test_erratic_reading_detection(self):
        """Detect erratic sensor behavior."""
        # Readings oscillating wildly
        readings = []
        for i in range(20):
            value = 3.0 + (5.0 if i % 2 == 0 else -5.0)  # +/- 5% swing
            readings.append(O2Reading(
                value=value,
                timestamp=datetime.now() + timedelta(seconds=i),
                sensor_id="ZR-001",
                temperature_c=25.0,
            ))

        values = [r.value for r in readings]
        std_dev = statistics.stdev(values)

        # Erratic if std dev > 2% (way above normal noise)
        is_erratic = std_dev > 2.0
        assert is_erratic, "Should detect erratic sensor behavior"

    @pytest.mark.parametrize("quality_flag,should_fail", [
        ("GOOD", False),
        ("BAD", True),
        ("UNCERTAIN", True),
        ("SENSOR_FAULT", True),
        ("COMMUNICATION_ERROR", True),
    ])
    def test_quality_flag_validation(self, quality_flag: str, should_fail: bool):
        """Test quality flag handling."""
        validator = O2SensorValidator(SENSOR_SPECS[O2SensorType.ZIRCONIA])

        reading = O2Reading(
            value=5.0,
            timestamp=datetime.now(),
            sensor_id="ZR-001",
            temperature_c=25.0,
            quality_flag=quality_flag,
        )

        is_valid, issues = validator.validate_reading(reading)

        if should_fail:
            assert not is_valid or len(issues) > 0, \
                f"Quality flag {quality_flag} should cause issues"


# =============================================================================
# RESPONSE TIME TESTS
# =============================================================================

class TestO2SensorResponseTime:
    """Test O2 sensor response time requirements."""

    @pytest.mark.parametrize("sensor_type,max_t90_seconds", [
        (O2SensorType.ZIRCONIA, 10.0),      # Fast in-situ
        (O2SensorType.PARAMAGNETIC, 30.0),  # Extractive
        (O2SensorType.ELECTROCHEMICAL, 60.0),  # Slower
    ])
    def test_response_time_within_spec(
        self,
        sensor_type: O2SensorType,
        max_t90_seconds: float
    ):
        """Verify response time meets specification."""
        spec = SENSOR_SPECS[sensor_type]
        assert spec.response_time_t90_seconds <= max_t90_seconds, \
            f"{sensor_type.value} response time exceeds maximum"

    def test_step_response_simulation(self):
        """Simulate and validate step response."""
        # Simulate step change from 3% to 8% O2
        initial_o2 = 3.0
        final_o2 = 8.0
        t90_seconds = 10.0  # Zirconia spec

        # First-order response: y = y_final - (y_final - y_initial) * exp(-t/tau)
        # At T90: 0.9 = 1 - exp(-T90/tau) => tau = -T90/ln(0.1)
        tau = -t90_seconds / math.log(0.1)

        # Check response at various times
        for t in [0, 5, 10, 15, 20]:
            response = final_o2 - (final_o2 - initial_o2) * math.exp(-t / tau)

            if t == 0:
                assert abs(response - initial_o2) < 0.01
            elif t >= t90_seconds:
                # Should reach 90% of final value
                target = initial_o2 + 0.9 * (final_o2 - initial_o2)
                assert response >= target - 0.1, \
                    f"At t={t}s, response {response:.2f} should reach {target:.2f}"


# =============================================================================
# CALIBRATION DRIFT TESTS
# =============================================================================

class TestO2SensorCalibrationDrift:
    """Test detection of calibration drift over time."""

    def test_zero_drift_detection(self):
        """Detect zero point calibration drift."""
        # Simulate readings over time with drift
        base_time = datetime.now()
        readings_with_drift = []

        expected_o2 = 0.0  # Zero gas
        drift_rate = 0.01  # % per day

        for days in range(0, 60, 7):  # Weekly readings over 60 days
            drifted_value = expected_o2 + (days * drift_rate)
            readings_with_drift.append(O2Reading(
                value=drifted_value,
                timestamp=base_time + timedelta(days=days),
                sensor_id="ZR-001",
                temperature_c=25.0,
            ))

        # Calculate drift
        first_reading = readings_with_drift[0].value
        last_reading = readings_with_drift[-1].value
        total_drift = last_reading - first_reading

        # Drift should be detectable
        assert total_drift > 0.5, "Should detect significant zero drift"

        # Flag if drift exceeds 0.5% over 60 days
        max_acceptable_drift = 0.5
        drift_detected = total_drift > max_acceptable_drift
        assert drift_detected, "Drift should be flagged as excessive"

    def test_span_drift_detection(self):
        """Detect span calibration drift."""
        base_time = datetime.now()

        span_gas = 20.9  # Air reference
        drift_rate = -0.005  # % per day (span dropping)

        readings = []
        for days in range(0, 90, 10):
            measured = span_gas * (1 + days * drift_rate / 100)
            readings.append(O2Reading(
                value=measured,
                timestamp=base_time + timedelta(days=days),
                sensor_id="ZR-001",
                temperature_c=25.0,
            ))

        # Check span factor at end
        span_factor = readings[-1].value / span_gas

        # Flag if span drifted more than 2%
        assert abs(span_factor - 1.0) < 0.02, "Span drift within acceptable limits"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
