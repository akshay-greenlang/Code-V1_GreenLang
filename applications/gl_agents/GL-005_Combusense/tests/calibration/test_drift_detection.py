# -*- coding: utf-8 -*-
"""
Sensor Calibration Drift Detection Tests for GL-005 CombustionSense
===================================================================

Tests for detecting sensor calibration drift over time including:
    - Zero drift detection
    - Span drift detection
    - Long-term stability analysis
    - Calibration due warnings
    - Automatic drift compensation

Reference Standards:
    - EPA Method 3A: Continuous monitoring requirements
    - IEC 61207: Performance expression for gas analyzers

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# DRIFT DETECTION CLASSES
# =============================================================================

class DriftType(Enum):
    """Types of calibration drift."""
    NONE = "none"
    ZERO_DRIFT = "zero_drift"
    SPAN_DRIFT = "span_drift"
    BOTH = "both"


class DriftSeverity(Enum):
    """Severity of detected drift."""
    NONE = "none"
    MINOR = "minor"          # < 1% of span
    MODERATE = "moderate"    # 1-2% of span
    SIGNIFICANT = "significant"  # 2-5% of span
    CRITICAL = "critical"    # > 5% of span


@dataclass
class CalibrationCheckData:
    """Data from calibration check."""
    check_date: datetime
    reference_value: float   # Known reference gas value
    sensor_reading: float    # Sensor reading
    ambient_temp_c: float = 25.0
    ambient_pressure_kpa: float = 101.325


@dataclass
class DriftAnalysisResult:
    """Result of drift analysis."""
    drift_type: DriftType
    severity: DriftSeverity
    zero_drift_value: float
    span_drift_percent: float
    days_since_last_cal: int
    calibration_recommended: bool
    details: List[str]


@dataclass
class SensorCalibrationHistory:
    """Calibration history for a sensor."""
    sensor_id: str
    parameter: str
    span: float              # Full scale range
    initial_cal_date: datetime
    last_cal_date: datetime
    calibration_checks: List[CalibrationCheckData] = field(default_factory=list)


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class CalibrationDriftDetector:
    """
    Detects calibration drift in sensors over time.

    Detection Methods:
        - Periodic zero gas checks
        - Periodic span gas checks
        - Trend analysis of deviations
        - Environmental compensation
    """

    def __init__(self):
        self.sensors: Dict[str, SensorCalibrationHistory] = {}
        # Drift thresholds (% of span)
        self.minor_threshold = 0.5
        self.moderate_threshold = 1.0
        self.significant_threshold = 2.0
        self.critical_threshold = 5.0

    def register_sensor(self, history: SensorCalibrationHistory) -> None:
        """Register a sensor for drift monitoring."""
        self.sensors[history.sensor_id] = history

    def add_calibration_check(
        self,
        sensor_id: str,
        check_data: CalibrationCheckData
    ) -> None:
        """Add a calibration check result."""
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        self.sensors[sensor_id].calibration_checks.append(check_data)

    def analyze_drift(self, sensor_id: str) -> DriftAnalysisResult:
        """
        Analyze drift for a sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            DriftAnalysisResult with drift analysis
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        history = self.sensors[sensor_id]

        if not history.calibration_checks:
            return DriftAnalysisResult(
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                zero_drift_value=0.0,
                span_drift_percent=0.0,
                days_since_last_cal=(datetime.now() - history.last_cal_date).days,
                calibration_recommended=False,
                details=["No calibration checks recorded"],
            )

        # Separate zero and span checks
        zero_checks = [c for c in history.calibration_checks if c.reference_value == 0.0]
        span_checks = [c for c in history.calibration_checks if c.reference_value > 0.0]

        # Analyze zero drift
        zero_drift = 0.0
        if zero_checks:
            zero_readings = [c.sensor_reading for c in zero_checks]
            zero_drift = zero_readings[-1] if zero_readings else 0.0

        # Analyze span drift
        span_drift_pct = 0.0
        if span_checks:
            # Calculate span factor for most recent check
            latest_span = span_checks[-1]
            span_factor = latest_span.sensor_reading / latest_span.reference_value
            span_drift_pct = abs(1.0 - span_factor) * 100

        # Determine drift type and severity
        drift_type, severity = self._determine_drift(
            zero_drift, span_drift_pct, history.span
        )

        # Calculate days since calibration
        days_since_cal = (datetime.now() - history.last_cal_date).days

        # Determine if calibration recommended
        cal_recommended = severity in [DriftSeverity.SIGNIFICANT, DriftSeverity.CRITICAL]

        details = self._generate_details(
            zero_drift, span_drift_pct, history.span, severity
        )

        return DriftAnalysisResult(
            drift_type=drift_type,
            severity=severity,
            zero_drift_value=zero_drift,
            span_drift_percent=span_drift_pct,
            days_since_last_cal=days_since_cal,
            calibration_recommended=cal_recommended,
            details=details,
        )

    def _determine_drift(
        self,
        zero_drift: float,
        span_drift_pct: float,
        span: float
    ) -> Tuple[DriftType, DriftSeverity]:
        """Determine drift type and severity."""
        zero_drift_pct = abs(zero_drift / span * 100) if span > 0 else 0

        has_zero = zero_drift_pct > self.minor_threshold
        has_span = span_drift_pct > self.minor_threshold

        if has_zero and has_span:
            drift_type = DriftType.BOTH
        elif has_zero:
            drift_type = DriftType.ZERO_DRIFT
        elif has_span:
            drift_type = DriftType.SPAN_DRIFT
        else:
            drift_type = DriftType.NONE

        # Overall severity is the worst of the two
        max_drift = max(zero_drift_pct, span_drift_pct)

        if max_drift > self.critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif max_drift > self.significant_threshold:
            severity = DriftSeverity.SIGNIFICANT
        elif max_drift > self.moderate_threshold:
            severity = DriftSeverity.MODERATE
        elif max_drift > self.minor_threshold:
            severity = DriftSeverity.MINOR
        else:
            severity = DriftSeverity.NONE

        return drift_type, severity

    def _generate_details(
        self,
        zero_drift: float,
        span_drift_pct: float,
        span: float,
        severity: DriftSeverity
    ) -> List[str]:
        """Generate drift analysis details."""
        details = []

        if abs(zero_drift) > 0:
            details.append(f"Zero drift: {zero_drift:.3f} ({abs(zero_drift/span*100):.2f}% of span)")

        if span_drift_pct > 0:
            details.append(f"Span drift: {span_drift_pct:.2f}%")

        if severity == DriftSeverity.CRITICAL:
            details.append("CRITICAL: Immediate recalibration required")
        elif severity == DriftSeverity.SIGNIFICANT:
            details.append("Recalibration recommended within 24 hours")
        elif severity == DriftSeverity.MODERATE:
            details.append("Schedule recalibration within 1 week")

        return details

    def calculate_correction_factor(
        self,
        sensor_id: str
    ) -> Tuple[float, float]:
        """
        Calculate zero and span correction factors.

        Returns:
            Tuple of (zero_offset, span_factor)
        """
        if sensor_id not in self.sensors:
            return 0.0, 1.0

        history = self.sensors[sensor_id]

        # Zero offset from latest zero check
        zero_offset = 0.0
        zero_checks = [c for c in history.calibration_checks if c.reference_value == 0.0]
        if zero_checks:
            zero_offset = -zero_checks[-1].sensor_reading

        # Span factor from latest span check
        span_factor = 1.0
        span_checks = [c for c in history.calibration_checks if c.reference_value > 0.0]
        if span_checks:
            latest = span_checks[-1]
            if latest.sensor_reading != 0:
                span_factor = latest.reference_value / latest.sensor_reading

        return zero_offset, span_factor


# =============================================================================
# ZERO DRIFT TESTS
# =============================================================================

class TestZeroDriftDetection:
    """Test zero drift detection."""

    @pytest.fixture
    def detector(self) -> CalibrationDriftDetector:
        """Create detector with registered sensor."""
        d = CalibrationDriftDetector()
        d.register_sensor(SensorCalibrationHistory(
            sensor_id="O2-001",
            parameter="O2",
            span=25.0,  # 0-25% O2
            initial_cal_date=datetime.now() - timedelta(days=30),
            last_cal_date=datetime.now() - timedelta(days=30),
        ))
        return d

    def test_no_zero_drift(self, detector: CalibrationDriftDetector):
        """Test no zero drift detected."""
        # Zero check with perfect reading
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=0.0,
        ))

        result = detector.analyze_drift("O2-001")

        assert result.drift_type in [DriftType.NONE, DriftType.SPAN_DRIFT]
        assert result.severity == DriftSeverity.NONE

    @pytest.mark.parametrize("zero_reading,expected_severity", [
        (0.1, DriftSeverity.MINOR),      # 0.4% of 25% span
        (0.3, DriftSeverity.MODERATE),   # 1.2% of span
        (0.8, DriftSeverity.SIGNIFICANT), # 3.2% of span
        (1.5, DriftSeverity.CRITICAL),   # 6% of span
    ])
    def test_zero_drift_severity_levels(
        self,
        detector: CalibrationDriftDetector,
        zero_reading: float,
        expected_severity: DriftSeverity
    ):
        """Test zero drift severity classification."""
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=zero_reading,
        ))

        result = detector.analyze_drift("O2-001")

        assert result.severity == expected_severity
        assert DriftType.ZERO_DRIFT in [result.drift_type, DriftType.BOTH]


# =============================================================================
# SPAN DRIFT TESTS
# =============================================================================

class TestSpanDriftDetection:
    """Test span drift detection."""

    @pytest.fixture
    def detector(self) -> CalibrationDriftDetector:
        """Create detector with registered sensor."""
        d = CalibrationDriftDetector()
        d.register_sensor(SensorCalibrationHistory(
            sensor_id="O2-001",
            parameter="O2",
            span=25.0,
            initial_cal_date=datetime.now() - timedelta(days=30),
            last_cal_date=datetime.now() - timedelta(days=30),
        ))
        return d

    def test_no_span_drift(self, detector: CalibrationDriftDetector):
        """Test no span drift detected."""
        # Span check with perfect reading
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=20.9,  # Air reference
            sensor_reading=20.9,
        ))

        result = detector.analyze_drift("O2-001")

        assert result.severity == DriftSeverity.NONE
        assert result.span_drift_percent < 0.5

    @pytest.mark.parametrize("reading,reference,expected_severity", [
        # 1% high - minor
        (21.1, 20.9, DriftSeverity.MINOR),
        # 2% low - moderate
        (20.5, 20.9, DriftSeverity.MODERATE),
        # 4% drift - significant
        (20.0, 20.9, DriftSeverity.SIGNIFICANT),
        # 8% drift - critical
        (19.2, 20.9, DriftSeverity.CRITICAL),
    ])
    def test_span_drift_severity_levels(
        self,
        detector: CalibrationDriftDetector,
        reading: float,
        reference: float,
        expected_severity: DriftSeverity
    ):
        """Test span drift severity classification."""
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=reference,
            sensor_reading=reading,
        ))

        result = detector.analyze_drift("O2-001")

        assert result.severity == expected_severity


# =============================================================================
# COMBINED DRIFT TESTS
# =============================================================================

class TestCombinedDrift:
    """Test combined zero and span drift."""

    @pytest.fixture
    def detector(self) -> CalibrationDriftDetector:
        d = CalibrationDriftDetector()
        d.register_sensor(SensorCalibrationHistory(
            sensor_id="CO-001",
            parameter="CO",
            span=2000.0,  # 0-2000 ppm
            initial_cal_date=datetime.now() - timedelta(days=30),
            last_cal_date=datetime.now() - timedelta(days=30),
        ))
        return d

    def test_both_drift_types(self, detector: CalibrationDriftDetector):
        """Test detection of both zero and span drift."""
        # Zero drift
        detector.add_calibration_check("CO-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=20.0,  # 1% of span
        ))

        # Span drift
        detector.add_calibration_check("CO-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=500.0,
            sensor_reading=510.0,  # 2% high
        ))

        result = detector.analyze_drift("CO-001")

        assert result.drift_type == DriftType.BOTH
        assert abs(result.zero_drift_value - 20.0) < 0.1


# =============================================================================
# CORRECTION FACTOR TESTS
# =============================================================================

class TestCorrectionFactors:
    """Test drift correction factor calculation."""

    @pytest.fixture
    def detector(self) -> CalibrationDriftDetector:
        d = CalibrationDriftDetector()
        d.register_sensor(SensorCalibrationHistory(
            sensor_id="O2-001",
            parameter="O2",
            span=25.0,
            initial_cal_date=datetime.now() - timedelta(days=30),
            last_cal_date=datetime.now() - timedelta(days=30),
        ))
        return d

    def test_zero_offset_calculation(self, detector: CalibrationDriftDetector):
        """Test zero offset correction calculation."""
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=0.2,  # Reading 0.2% high at zero
        ))

        zero_offset, span_factor = detector.calculate_correction_factor("O2-001")

        assert abs(zero_offset - (-0.2)) < 0.01
        assert abs(span_factor - 1.0) < 0.01

    def test_span_factor_calculation(self, detector: CalibrationDriftDetector):
        """Test span factor correction calculation."""
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=20.9,
            sensor_reading=20.5,  # Reading 2% low
        ))

        zero_offset, span_factor = detector.calculate_correction_factor("O2-001")

        # Correction factor should increase readings
        expected_factor = 20.9 / 20.5
        assert abs(span_factor - expected_factor) < 0.01

    def test_apply_correction(self, detector: CalibrationDriftDetector):
        """Test applying correction factors."""
        # Add zero check
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=0.1,
        ))

        # Add span check
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=20.9,
            sensor_reading=20.5,
        ))

        zero_offset, span_factor = detector.calculate_correction_factor("O2-001")

        # Apply correction to a reading
        raw_reading = 5.0
        corrected = (raw_reading + zero_offset) * span_factor

        # Correction should be in right direction
        assert corrected != raw_reading


# =============================================================================
# LONG-TERM TRENDING TESTS
# =============================================================================

class TestLongTermTrending:
    """Test long-term drift trending."""

    @pytest.fixture
    def detector(self) -> CalibrationDriftDetector:
        d = CalibrationDriftDetector()
        d.register_sensor(SensorCalibrationHistory(
            sensor_id="O2-001",
            parameter="O2",
            span=25.0,
            initial_cal_date=datetime.now() - timedelta(days=90),
            last_cal_date=datetime.now() - timedelta(days=90),
        ))
        return d

    def test_progressive_drift_detection(self, detector: CalibrationDriftDetector):
        """Test detection of progressively increasing drift."""
        base_time = datetime.now() - timedelta(days=60)

        # Add zero checks over time showing progressive drift
        for day in range(0, 60, 10):
            detector.add_calibration_check("O2-001", CalibrationCheckData(
                check_date=base_time + timedelta(days=day),
                reference_value=0.0,
                sensor_reading=0.05 * (day / 10),  # Increasing drift
            ))

        result = detector.analyze_drift("O2-001")

        # Latest drift should be detected
        assert result.zero_drift_value > 0.2
        assert result.drift_type in [DriftType.ZERO_DRIFT, DriftType.BOTH]

    def test_calibration_recommended_flag(self, detector: CalibrationDriftDetector):
        """Test calibration recommended flag."""
        # Add significant drift
        detector.add_calibration_check("O2-001", CalibrationCheckData(
            check_date=datetime.now(),
            reference_value=0.0,
            sensor_reading=0.8,  # 3.2% of span
        ))

        result = detector.analyze_drift("O2-001")

        assert result.calibration_recommended
        assert "recalibration" in " ".join(result.details).lower()


# =============================================================================
# INIT FILE FOR CALIBRATION TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
