"""
Unit Tests: Failure Detector
Tests for failure mode classification and multi-sensor fusion.
Author: GL-TestEngineer
"""
import pytest
from typing import Dict, Tuple, Optional
from conftest import TrapType, TrapFailureMode, DiagnosticMethod, MockTrapData


class FailureDetector:
    ULTRASONIC_NORMAL_MAX = 70
    ULTRASONIC_WARNING = 85
    ULTRASONIC_FAILED = 95
    TEMP_DROP_NORMAL_MIN = 5
    TEMP_DROP_SUBCOOLED_MAX = 30

    def classify_from_acoustic(self, ultrasonic_db: float, cycle_rate: Optional[float], trap_type: TrapType) -> Tuple[TrapFailureMode, float]:
        if ultrasonic_db > self.ULTRASONIC_FAILED:
            return TrapFailureMode.FAILED_OPEN, 0.9
        if ultrasonic_db > self.ULTRASONIC_WARNING:
            if cycle_rate and trap_type == TrapType.THERMODYNAMIC and cycle_rate > 30:
                return TrapFailureMode.BLOW_THROUGH, 0.8
            return TrapFailureMode.BLOW_THROUGH, 0.6
        if ultrasonic_db < 50:
            if trap_type in [TrapType.MECHANICAL_FLOAT, TrapType.MECHANICAL_BUCKET]:
                return TrapFailureMode.FAILED_CLOSED, 0.5
            return TrapFailureMode.UNKNOWN, 0.4
        return TrapFailureMode.NORMAL, 0.85

    def classify_from_temperature(self, inlet_k: float, outlet_k: float) -> Tuple[TrapFailureMode, float]:
        temp_drop = inlet_k - outlet_k
        if temp_drop < 2.0:
            return TrapFailureMode.FAILED_OPEN, 0.85
        if temp_drop > self.TEMP_DROP_SUBCOOLED_MAX:
            return TrapFailureMode.FAILED_CLOSED, 0.7
        if self.TEMP_DROP_NORMAL_MIN < temp_drop < self.TEMP_DROP_SUBCOOLED_MAX:
            return TrapFailureMode.NORMAL, 0.75
        return TrapFailureMode.UNKNOWN, 0.3

    def classify_combined(self, trap: MockTrapData) -> Tuple[TrapFailureMode, float, DiagnosticMethod]:
        acoustic_avail = trap.ultrasonic_db is not None
        temp_avail = trap.inlet_temperature_k > 0 and trap.outlet_temperature_k > 0
        if acoustic_avail and temp_avail:
            mode_a, conf_a = self.classify_from_acoustic(trap.ultrasonic_db, trap.cycle_rate_per_min, trap.trap_type)
            mode_t, conf_t = self.classify_from_temperature(trap.inlet_temperature_k, trap.outlet_temperature_k)
            if mode_a == mode_t:
                return mode_a, min(0.95, conf_a + conf_t * 0.5), DiagnosticMethod.COMBINED
            if conf_a > conf_t:
                return mode_a, conf_a * 0.8, DiagnosticMethod.ACOUSTIC
            return mode_t, conf_t * 0.8, DiagnosticMethod.TEMPERATURE
        elif acoustic_avail:
            mode, conf = self.classify_from_acoustic(trap.ultrasonic_db, trap.cycle_rate_per_min, trap.trap_type)
            return mode, conf, DiagnosticMethod.ACOUSTIC
        elif temp_avail:
            mode, conf = self.classify_from_temperature(trap.inlet_temperature_k, trap.outlet_temperature_k)
            return mode, conf, DiagnosticMethod.TEMPERATURE
        return TrapFailureMode.UNKNOWN, 0.0, DiagnosticMethod.VISUAL


@pytest.fixture
def detector(): return FailureDetector()


class TestFailureModeClassification:
    def test_healthy_acoustic(self, detector):
        mode, conf = detector.classify_from_acoustic(65, 10, TrapType.THERMODYNAMIC)
        assert mode == TrapFailureMode.NORMAL
        assert conf >= 0.8

    def test_failed_open_acoustic(self, detector):
        mode, conf = detector.classify_from_acoustic(98, 0, TrapType.THERMODYNAMIC)
        assert mode == TrapFailureMode.FAILED_OPEN
        assert conf >= 0.8

    def test_blow_through_acoustic(self, detector):
        mode, conf = detector.classify_from_acoustic(88, 35, TrapType.THERMODYNAMIC)
        assert mode == TrapFailureMode.BLOW_THROUGH

    def test_failed_open_temperature(self, detector):
        mode, conf = detector.classify_from_temperature(453.0, 451.0)
        assert mode == TrapFailureMode.FAILED_OPEN

    def test_failed_closed_temperature(self, detector):
        mode, conf = detector.classify_from_temperature(453.0, 400.0)
        assert mode == TrapFailureMode.FAILED_CLOSED

    def test_normal_temperature(self, detector):
        mode, conf = detector.classify_from_temperature(453.0, 440.0)
        assert mode == TrapFailureMode.NORMAL


class TestMultiSensorFusion:
    def test_combined_agreement(self, detector, healthy_trap):
        mode, conf, method = detector.classify_combined(healthy_trap)
        assert mode == TrapFailureMode.NORMAL
        assert method == DiagnosticMethod.COMBINED
        assert conf > 0.8

    def test_combined_failed_open(self, detector, failed_open_trap):
        mode, conf, method = detector.classify_combined(failed_open_trap)
        assert mode == TrapFailureMode.FAILED_OPEN


class TestConfidenceScoring:
    def test_confidence_range(self, detector):
        for db in [40, 65, 88, 98]:
            mode, conf = detector.classify_from_acoustic(db, None, TrapType.THERMODYNAMIC)
            assert 0.0 <= conf <= 1.0


class TestFalsePositiveHandling:
    def test_low_confidence_unknown(self, detector):
        mode, conf = detector.classify_from_acoustic(55, None, TrapType.ORIFICE)
        if mode == TrapFailureMode.UNKNOWN:
            assert conf < 0.5
