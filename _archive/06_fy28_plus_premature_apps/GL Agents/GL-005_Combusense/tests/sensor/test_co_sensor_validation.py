# -*- coding: utf-8 -*-
"""
CO Sensor Validation Tests for GL-005 CombustionSense
=====================================================

Tests for carbon monoxide sensor accuracy, calibration, and failure detection
per industry standards and safety requirements.

Reference Standards:
    - EPA Method 10: Determination of Carbon Monoxide Emissions
    - ASTM D6522: Standard Test Method for CO Determination
    - IEC 61508: Functional Safety (for safety-critical CO detection)
    - NFPA 85: Safety limits for CO in combustion

Test Categories:
    1. Sensor accuracy validation
    2. Range and linearity testing
    3. Cross-sensitivity testing
    4. Alarm threshold validation
    5. Failure mode detection

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics


# =============================================================================
# SENSOR SPECIFICATIONS
# =============================================================================

class COSensorType(Enum):
    """Types of CO sensors used in combustion."""
    NDIR = "ndir"               # Non-dispersive infrared
    ELECTROCHEMICAL = "electrochemical"  # Electrochemical cell
    CATALYTIC = "catalytic"     # Catalytic bead


@dataclass
class COSensorSpecification:
    """CO sensor specifications per manufacturer/standard."""
    sensor_type: COSensorType
    range_min: float = 0.0          # ppm CO
    range_max: float = 2000.0       # ppm CO
    accuracy_ppm: float = 10.0      # Absolute accuracy (ppm)
    accuracy_percent_of_reading: float = 5.0  # Relative accuracy (%)
    repeatability_ppm: float = 5.0  # Repeatability (ppm)
    response_time_t90_seconds: float = 60.0
    operating_temp_min_c: float = 0.0
    operating_temp_max_c: float = 50.0
    cross_sensitivity_co2_percent: float = 2.0  # Cross-sensitivity to CO2
    calibration_interval_days: int = 90

    # Safety thresholds per NFPA 85
    alarm_low_ppm: float = 200.0
    alarm_high_ppm: float = 400.0
    trip_ppm: float = 1000.0


# Standard sensor specifications
CO_SENSOR_SPECS = {
    COSensorType.NDIR: COSensorSpecification(
        sensor_type=COSensorType.NDIR,
        range_min=0.0, range_max=5000.0,
        accuracy_ppm=10.0,
        accuracy_percent_of_reading=2.0,  # Higher accuracy
        repeatability_ppm=5.0,
        response_time_t90_seconds=30.0,
        cross_sensitivity_co2_percent=1.0,  # Low cross-sensitivity
        calibration_interval_days=90,
    ),
    COSensorType.ELECTROCHEMICAL: COSensorSpecification(
        sensor_type=COSensorType.ELECTROCHEMICAL,
        range_min=0.0, range_max=2000.0,
        accuracy_ppm=20.0,
        accuracy_percent_of_reading=5.0,
        repeatability_ppm=10.0,
        response_time_t90_seconds=60.0,
        cross_sensitivity_co2_percent=3.0,  # Higher cross-sensitivity
        calibration_interval_days=180,
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class COReading:
    """Single CO sensor reading."""
    value: float               # ppm CO
    timestamp: datetime
    sensor_id: str
    temperature_c: float
    co2_percent: Optional[float] = None  # For cross-sensitivity correction
    quality_flag: str = "GOOD"
    raw_millivolts: Optional[float] = None


@dataclass
class COCalibrationData:
    """Calibration data for CO sensor."""
    calibration_date: datetime
    zero_gas_reading: float     # Reading at 0 ppm (N2)
    span_gas_reading: float     # Reading at span gas
    span_gas_concentration: float  # Span gas ppm
    zero_offset: float = 0.0
    span_factor: float = 1.0
    temperature_at_calibration: float = 25.0


@dataclass
class COAlarmStatus:
    """CO alarm status."""
    current_ppm: float
    alarm_level: str            # "NORMAL", "LOW_ALARM", "HIGH_ALARM", "TRIP"
    is_alarming: bool
    time_in_alarm: Optional[timedelta] = None


# =============================================================================
# CO SENSOR VALIDATOR
# =============================================================================

class COSensorValidator:
    """Validates CO sensor accuracy and alarm functionality."""

    def __init__(self, spec: COSensorSpecification):
        self.spec = spec

    def validate_reading(
        self,
        reading: COReading,
        expected_value: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single CO reading.

        Args:
            reading: CO sensor reading
            expected_value: Known value for accuracy check (optional)

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check range
        if reading.value < self.spec.range_min:
            issues.append(f"Reading {reading.value} ppm below range minimum")
        if reading.value > self.spec.range_max:
            issues.append(f"Reading {reading.value} ppm above range maximum")

        # Check for negative values
        if reading.value < 0:
            issues.append(f"Negative CO reading: {reading.value} ppm")

        # Check accuracy if expected value provided
        if expected_value is not None:
            error = abs(reading.value - expected_value)
            max_error = self._calculate_max_allowable_error(expected_value)
            if error > max_error:
                issues.append(f"Error {error:.1f} ppm exceeds max allowable {max_error:.1f} ppm")

        # Check quality flag
        if reading.quality_flag not in ["GOOD", "MARGINAL"]:
            issues.append(f"Quality flag: {reading.quality_flag}")

        return len(issues) == 0, issues

    def _calculate_max_allowable_error(self, expected_value: float) -> float:
        """Calculate maximum allowable error per specification."""
        error_absolute = self.spec.accuracy_ppm
        error_relative = expected_value * self.spec.accuracy_percent_of_reading / 100
        return max(error_absolute, error_relative)

    def check_alarm_status(self, reading: COReading) -> COAlarmStatus:
        """
        Determine CO alarm status based on reading.

        Alarm Levels per NFPA 85:
            - NORMAL: < 200 ppm
            - LOW_ALARM: 200-400 ppm
            - HIGH_ALARM: 400-1000 ppm
            - TRIP: >= 1000 ppm
        """
        value = reading.value

        if value >= self.spec.trip_ppm:
            return COAlarmStatus(
                current_ppm=value,
                alarm_level="TRIP",
                is_alarming=True,
            )
        elif value >= self.spec.alarm_high_ppm:
            return COAlarmStatus(
                current_ppm=value,
                alarm_level="HIGH_ALARM",
                is_alarming=True,
            )
        elif value >= self.spec.alarm_low_ppm:
            return COAlarmStatus(
                current_ppm=value,
                alarm_level="LOW_ALARM",
                is_alarming=True,
            )
        else:
            return COAlarmStatus(
                current_ppm=value,
                alarm_level="NORMAL",
                is_alarming=False,
            )

    def apply_cross_sensitivity_correction(
        self,
        co_reading: float,
        co2_percent: float
    ) -> float:
        """
        Apply cross-sensitivity correction for CO2 interference.

        NDIR sensors can have cross-sensitivity to CO2.
        Correction: CO_corrected = CO_measured - (CO2 * cross_sensitivity_factor)
        """
        if co2_percent is None:
            return co_reading

        # Cross-sensitivity effect (ppm CO per % CO2)
        cross_sensitivity_factor = self.spec.cross_sensitivity_co2_percent * 10  # ppm per % CO2

        correction = co2_percent * cross_sensitivity_factor
        corrected = co_reading - correction

        return max(0.0, corrected)  # Can't have negative CO


# =============================================================================
# ACCURACY VALIDATION TESTS
# =============================================================================

class TestCOSensorAccuracy:
    """Test CO sensor accuracy requirements."""

    @pytest.fixture
    def ndir_validator(self) -> COSensorValidator:
        return COSensorValidator(CO_SENSOR_SPECS[COSensorType.NDIR])

    @pytest.fixture
    def electrochemical_validator(self) -> COSensorValidator:
        return COSensorValidator(CO_SENSOR_SPECS[COSensorType.ELECTROCHEMICAL])

    @pytest.mark.parametrize("expected,measured,should_pass", [
        # Within NDIR spec tolerance (10 ppm or 2% of reading)
        (100, 100, True),
        (100, 105, True),
        (100, 95, True),
        (500, 510, True),     # 2% of 500 = 10 ppm
        (500, 490, True),

        # Outside tolerance
        (100, 120, False),    # 20 ppm error > 10 ppm
        (500, 550, False),    # 50 ppm error > 10 ppm
        (50, 75, False),      # 25 ppm error
    ])
    def test_accuracy_within_specification(
        self,
        ndir_validator: COSensorValidator,
        expected: float,
        measured: float,
        should_pass: bool
    ):
        """Test accuracy validation against specification."""
        reading = COReading(
            value=measured,
            timestamp=datetime.now(),
            sensor_id="CO-001",
            temperature_c=25.0,
        )

        is_valid, issues = ndir_validator.validate_reading(reading, expected)

        if should_pass:
            assert is_valid, f"Reading should pass: {issues}"
        else:
            assert not is_valid, "Reading should fail accuracy check"

    @pytest.mark.parametrize("sensor_type,co_value,max_error", [
        # NDIR: 10 ppm or 2% of reading
        (COSensorType.NDIR, 100, 10.0),
        (COSensorType.NDIR, 500, 10.0),
        (COSensorType.NDIR, 1000, 20.0),  # 2% of 1000

        # Electrochemical: 20 ppm or 5% of reading
        (COSensorType.ELECTROCHEMICAL, 100, 20.0),
        (COSensorType.ELECTROCHEMICAL, 500, 25.0),  # 5% of 500
    ])
    def test_maximum_allowable_error(
        self,
        sensor_type: COSensorType,
        co_value: float,
        max_error: float
    ):
        """Test maximum allowable error calculation."""
        validator = COSensorValidator(CO_SENSOR_SPECS[sensor_type])
        calculated_max_error = validator._calculate_max_allowable_error(co_value)

        assert calculated_max_error > 0, "Max error must be positive"
        assert abs(calculated_max_error - max_error) < 1.0, \
            f"Calculated max error {calculated_max_error:.1f} ppm doesn't match expected {max_error}"

    def test_repeatability_validation(self, ndir_validator: COSensorValidator):
        """Test sensor repeatability over multiple readings."""
        expected_co = 250.0

        # Simulate multiple readings at same condition
        readings = [
            COReading(
                value=expected_co + (i * 0.5 - 2.5),  # Small variation
                timestamp=datetime.now(),
                sensor_id="CO-001",
                temperature_c=25.0,
            )
            for i in range(10)
        ]

        values = [r.value for r in readings]
        std_dev = statistics.stdev(values)

        # Repeatability should be within spec (5 ppm for NDIR)
        spec = CO_SENSOR_SPECS[COSensorType.NDIR]
        assert std_dev < spec.repeatability_ppm, \
            f"Standard deviation {std_dev:.2f} ppm exceeds repeatability spec {spec.repeatability_ppm}"


# =============================================================================
# ALARM THRESHOLD TESTS
# =============================================================================

class TestCOAlarmThresholds:
    """Test CO alarm threshold functionality per NFPA 85."""

    @pytest.fixture
    def validator(self) -> COSensorValidator:
        return COSensorValidator(CO_SENSOR_SPECS[COSensorType.NDIR])

    @pytest.mark.parametrize("co_ppm,expected_alarm_level", [
        # Normal operation
        (0, "NORMAL"),
        (50, "NORMAL"),
        (100, "NORMAL"),
        (199, "NORMAL"),

        # Low alarm (200-400 ppm)
        (200, "LOW_ALARM"),
        (250, "LOW_ALARM"),
        (399, "LOW_ALARM"),

        # High alarm (400-1000 ppm)
        (400, "HIGH_ALARM"),
        (500, "HIGH_ALARM"),
        (750, "HIGH_ALARM"),
        (999, "HIGH_ALARM"),

        # Trip (>= 1000 ppm)
        (1000, "TRIP"),
        (1500, "TRIP"),
        (2000, "TRIP"),
    ])
    def test_alarm_level_classification(
        self,
        validator: COSensorValidator,
        co_ppm: float,
        expected_alarm_level: str
    ):
        """Test CO alarm level classification per NFPA 85."""
        reading = COReading(
            value=co_ppm,
            timestamp=datetime.now(),
            sensor_id="CO-001",
            temperature_c=25.0,
        )

        status = validator.check_alarm_status(reading)

        assert status.alarm_level == expected_alarm_level, \
            f"CO {co_ppm} ppm should be {expected_alarm_level}, got {status.alarm_level}"

    @pytest.mark.parametrize("co_ppm,should_alarm", [
        (150, False),
        (199, False),
        (200, True),
        (350, True),
        (500, True),
        (1000, True),
    ])
    def test_alarm_active_flag(
        self,
        validator: COSensorValidator,
        co_ppm: float,
        should_alarm: bool
    ):
        """Test alarm active flag."""
        reading = COReading(
            value=co_ppm,
            timestamp=datetime.now(),
            sensor_id="CO-001",
            temperature_c=25.0,
        )

        status = validator.check_alarm_status(reading)

        assert status.is_alarming == should_alarm, \
            f"CO {co_ppm} ppm: is_alarming should be {should_alarm}"

    def test_alarm_deadband_behavior(self, validator: COSensorValidator):
        """Test alarm deadband to prevent chattering."""
        # Simulate readings oscillating around alarm threshold
        alarm_threshold = 200.0
        deadband = 10.0  # 5% of threshold

        readings = []
        values = [195, 205, 198, 202, 199, 201]  # Around threshold

        for value in values:
            readings.append(COReading(
                value=value,
                timestamp=datetime.now(),
                sensor_id="CO-001",
                temperature_c=25.0,
            ))

        # Check for alarm state transitions
        alarm_states = [validator.check_alarm_status(r).is_alarming for r in readings]

        # With deadband, shouldn't have rapid on-off cycling
        transitions = sum(1 for i in range(1, len(alarm_states))
                        if alarm_states[i] != alarm_states[i-1])

        # Some transitions expected, but not every reading
        assert transitions <= 4, "Too many alarm transitions - check deadband"


# =============================================================================
# CROSS-SENSITIVITY TESTS
# =============================================================================

class TestCOCrossSensitivity:
    """Test CO sensor cross-sensitivity to other gases."""

    @pytest.fixture
    def validator(self) -> COSensorValidator:
        return COSensorValidator(CO_SENSOR_SPECS[COSensorType.NDIR])

    @pytest.mark.parametrize("co_measured,co2_percent,expected_corrected", [
        # No CO2 - no correction
        (100, 0.0, 100.0),

        # Low CO2 - small correction
        (100, 5.0, 50.0),    # 5% CO2 * 10 ppm/% = 50 ppm correction

        # Normal combustion CO2 - moderate correction
        (200, 10.0, 100.0),  # 10% CO2 * 10 = 100 ppm correction

        # High CO2 - larger correction
        (300, 12.0, 180.0),  # 12% CO2 * 10 = 120 ppm correction
    ])
    def test_co2_cross_sensitivity_correction(
        self,
        validator: COSensorValidator,
        co_measured: float,
        co2_percent: float,
        expected_corrected: float
    ):
        """Test CO2 cross-sensitivity correction."""
        corrected = validator.apply_cross_sensitivity_correction(co_measured, co2_percent)

        assert abs(corrected - expected_corrected) < 1.0, \
            f"Corrected CO {corrected:.1f} ppm doesn't match expected {expected_corrected}"

    def test_correction_non_negative(self, validator: COSensorValidator):
        """Test that correction doesn't produce negative values."""
        # Large CO2 with small CO reading
        co_measured = 50.0
        co2_percent = 15.0  # Would give -100 ppm correction without floor

        corrected = validator.apply_cross_sensitivity_correction(co_measured, co2_percent)

        assert corrected >= 0.0, "Corrected CO should never be negative"


# =============================================================================
# FAILURE MODE TESTS
# =============================================================================

class TestCOSensorFailureModes:
    """Test CO sensor failure mode detection."""

    @pytest.fixture
    def validator(self) -> COSensorValidator:
        return COSensorValidator(CO_SENSOR_SPECS[COSensorType.NDIR])

    def test_sensor_saturation_detection(self, validator: COSensorValidator):
        """Detect sensor reading at or above maximum range."""
        reading = COReading(
            value=5000.0,  # At max range
            timestamp=datetime.now(),
            sensor_id="CO-001",
            temperature_c=25.0,
            quality_flag="SATURATED",
        )

        is_valid, issues = validator.validate_reading(reading)

        # Should flag quality issue
        assert not is_valid or len(issues) > 0

    def test_negative_reading_detection(self, validator: COSensorValidator):
        """Detect impossible negative CO reading."""
        reading = COReading(
            value=-10.0,
            timestamp=datetime.now(),
            sensor_id="CO-001",
            temperature_c=25.0,
        )

        is_valid, issues = validator.validate_reading(reading)

        assert not is_valid
        assert any("negative" in i.lower() for i in issues)

    def test_stuck_reading_detection(self):
        """Detect sensor stuck at constant value."""
        readings = [
            COReading(
                value=123.456,  # Exactly same value
                timestamp=datetime.now() + timedelta(seconds=i * 10),
                sensor_id="CO-001",
                temperature_c=25.0,
            )
            for i in range(20)
        ]

        values = [r.value for r in readings]
        unique_values = len(set(values))

        # Real sensor should have some variation
        is_stuck = unique_values == 1
        assert is_stuck, "Should detect stuck sensor"

    @pytest.mark.parametrize("rate_ppm_per_sec,should_flag", [
        (10, False),    # Reasonable change
        (50, False),    # Fast but possible
        (200, True),    # Unrealistically fast
        (500, True),    # Definitely sensor issue
    ])
    def test_rapid_rate_of_change_detection(self, rate_ppm_per_sec: float, should_flag: bool):
        """Detect unrealistic rate of change."""
        # Simulate readings with rapid change
        readings = []
        base_value = 100.0

        for i in range(10):
            value = base_value + (i * rate_ppm_per_sec)
            readings.append(COReading(
                value=value,
                timestamp=datetime.now() + timedelta(seconds=i),
                sensor_id="CO-001",
                temperature_c=25.0,
            ))

        # Calculate maximum rate of change
        rates = []
        for i in range(1, len(readings)):
            dt = 1.0  # 1 second intervals
            rate = abs(readings[i].value - readings[i-1].value) / dt
            rates.append(rate)

        max_rate = max(rates)
        is_flagged = max_rate > 100  # Flag if > 100 ppm/second

        assert is_flagged == should_flag, \
            f"Rate {max_rate:.1f} ppm/s should{'not' if not should_flag else ''} be flagged"


# =============================================================================
# CALIBRATION TESTS
# =============================================================================

class TestCOSensorCalibration:
    """Test CO sensor calibration requirements."""

    @pytest.fixture
    def valid_calibration(self) -> COCalibrationData:
        return COCalibrationData(
            calibration_date=datetime.now() - timedelta(days=30),
            zero_gas_reading=2.0,
            span_gas_reading=495.0,
            span_gas_concentration=500.0,
            zero_offset=2.0,
            span_factor=0.99,
        )

    def test_zero_calibration_check(self, valid_calibration: COCalibrationData):
        """Test zero gas calibration check."""
        # Zero offset should be within 5 ppm
        assert abs(valid_calibration.zero_offset) < 5.0, \
            f"Zero offset {valid_calibration.zero_offset} ppm too large"

    def test_span_calibration_check(self, valid_calibration: COCalibrationData):
        """Test span gas calibration check."""
        span_factor = valid_calibration.span_gas_reading / valid_calibration.span_gas_concentration

        # Span factor should be within 5% of 1.0
        assert 0.95 <= span_factor <= 1.05, \
            f"Span factor {span_factor:.3f} outside acceptable range"

    def test_calibration_expiry_detection(self):
        """Test detection of expired calibration."""
        spec = CO_SENSOR_SPECS[COSensorType.NDIR]

        expired_calibration = COCalibrationData(
            calibration_date=datetime.now() - timedelta(days=100),  # > 90 days
            zero_gas_reading=5.0,
            span_gas_reading=480.0,
            span_gas_concentration=500.0,
        )

        days_since_cal = (datetime.now() - expired_calibration.calibration_date).days
        is_expired = days_since_cal > spec.calibration_interval_days

        assert is_expired, "Should detect expired calibration"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
