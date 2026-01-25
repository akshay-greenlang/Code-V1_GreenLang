"""
Unit Tests: Acoustic Calculator
Tests for dB level analysis, frequency pattern detection, and acoustic diagnostics.
Author: GL-TestEngineer
"""
import pytest
import math
import hashlib
import json
from typing import Dict, List, Any
from conftest import TrapType, TrapFailureMode


class AcousticCalculator:
    ULTRASONIC_NORMAL_MAX_DB = 70
    ULTRASONIC_WARNING_DB = 85
    ULTRASONIC_FAILED_DB = 95
    FREQUENCY_THRESHOLDS = {
        TrapType.THERMODYNAMIC: (2000, 4000),
        TrapType.MECHANICAL_FLOAT: (1500, 3500),
        TrapType.MECHANICAL_BUCKET: (800, 2500),
        TrapType.THERMOSTATIC: (1000, 3000),
        TrapType.ORIFICE: (500, 2000),
    }

    def __init__(self):
        self.measurement_uncertainty = 0.02

    def analyze_db_level(self, ultrasonic_db):
        if ultrasonic_db < 0:
            raise ValueError("Ultrasonic dB cannot be negative")
        if ultrasonic_db <= self.ULTRASONIC_NORMAL_MAX_DB:
            status, severity = "NORMAL", 0.0
        elif ultrasonic_db <= self.ULTRASONIC_WARNING_DB:
            status = "WARNING"
            severity = (ultrasonic_db - self.ULTRASONIC_NORMAL_MAX_DB) / (self.ULTRASONIC_WARNING_DB - self.ULTRASONIC_NORMAL_MAX_DB)
        elif ultrasonic_db <= self.ULTRASONIC_FAILED_DB:
            status = "FAILED"
            severity = 0.5 + 0.5 * (ultrasonic_db - self.ULTRASONIC_WARNING_DB) / (self.ULTRASONIC_FAILED_DB - self.ULTRASONIC_WARNING_DB)
        else:
            status, severity = "CRITICAL", 1.0
        return {"ultrasonic_db": ultrasonic_db, "status": status, "severity": min(1.0, severity), "confidence": 1.0 - self.measurement_uncertainty}

    def analyze_frequency_pattern(self, frequency_hz, trap_type):
        if frequency_hz < 0:
            raise ValueError("Frequency cannot be negative")
        thresholds = self.FREQUENCY_THRESHOLDS.get(trap_type, (1000, 3000))
        low_freq, high_freq = thresholds
        if frequency_hz < low_freq * 0.5:
            pattern, failure = "LOW_ACTIVITY", TrapFailureMode.FAILED_CLOSED
        elif frequency_hz < low_freq:
            pattern, failure = "REDUCED_ACTIVITY", TrapFailureMode.PARTIAL_BLOCKAGE
        elif frequency_hz <= high_freq:
            pattern, failure = "NORMAL", TrapFailureMode.NORMAL
        elif frequency_hz <= high_freq * 1.5:
            pattern, failure = "ELEVATED", TrapFailureMode.BLOW_THROUGH
        else:
            pattern, failure = "CONTINUOUS_FLOW", TrapFailureMode.FAILED_OPEN
        return {"frequency_hz": frequency_hz, "trap_type": trap_type.name, "pattern": pattern, "failure_indication": failure.name, "normal_range": thresholds}

    def calculate_acoustic_energy(self, db_level, duration_seconds):
        if db_level < 0 or duration_seconds <= 0:
            return 0.0
        return 1e-12 * (10 ** (db_level / 10)) * duration_seconds

    def detect_cycling_pattern(self, measurements, sample_rate_hz):
        if len(measurements) < 10:
            return {"detected": False, "reason": "insufficient_data"}
        crossings, mean_val = 0, sum(measurements) / len(measurements)
        for i in range(1, len(measurements)):
            if (measurements[i - 1] < mean_val and measurements[i] >= mean_val) or (measurements[i - 1] >= mean_val and measurements[i] < mean_val):
                crossings += 1
        duration_seconds = len(measurements) / sample_rate_hz
        cpm = (crossings / 2) / duration_seconds * 60
        if cpm < 1:
            status, fm = "NO_CYCLING", TrapFailureMode.FAILED_CLOSED
        elif cpm < 5:
            status, fm = "LOW_CYCLING", TrapFailureMode.PARTIAL_BLOCKAGE
        elif cpm <= 30:
            status, fm = "NORMAL_CYCLING", TrapFailureMode.NORMAL
        else:
            status, fm = "RAPID_CYCLING", TrapFailureMode.BLOW_THROUGH
        return {"detected": True, "cycles_per_minute": cpm, "status": status, "failure_mode": fm.name, "duration_seconds": duration_seconds}

    def combine_acoustic_signals(self, ultrasonic_db, frequency_hz, trap_type):
        db_analysis = self.analyze_db_level(ultrasonic_db)
        freq_analysis = self.analyze_frequency_pattern(frequency_hz, trap_type)
        status_scores = {"NORMAL": 0, "WARNING": 0.3, "FAILED": 0.7, "CRITICAL": 1.0}
        pattern_scores = {"NORMAL": 0, "REDUCED_ACTIVITY": 0.3, "LOW_ACTIVITY": 0.5, "ELEVATED": 0.5, "CONTINUOUS_FLOW": 1.0}
        combined_score = 0.6 * status_scores.get(db_analysis["status"], 0.5) + 0.4 * pattern_scores.get(freq_analysis["pattern"], 0.5)
        if combined_score < 0.2:
            overall_status, failure_mode = "HEALTHY", TrapFailureMode.NORMAL
        elif combined_score < 0.5:
            overall_status, failure_mode = "WARNING", TrapFailureMode.UNKNOWN
        else:
            overall_status = "FAILED"
            failure_mode = TrapFailureMode[freq_analysis["failure_indication"]]
        data_for_hash = {"ultrasonic_db": ultrasonic_db, "frequency_hz": frequency_hz, "trap_type": trap_type.name, "combined_score": combined_score}
        provenance_hash = hashlib.sha256(json.dumps(data_for_hash, sort_keys=True).encode()).hexdigest()
        return {"overall_status": overall_status, "failure_mode": failure_mode.name, "combined_score": combined_score, "db_analysis": db_analysis, "frequency_analysis": freq_analysis, "provenance_hash": provenance_hash}


@pytest.fixture
def calculator():
    return AcousticCalculator()


class TestDBLevelAnalysis:
    def test_normal_db_level(self, calculator):
        result = calculator.analyze_db_level(65.0)
        assert result["status"] == "NORMAL"

    def test_warning_db_level(self, calculator):
        result = calculator.analyze_db_level(78.0)
        assert result["status"] == "WARNING"

    def test_failed_db_level(self, calculator):
        result = calculator.analyze_db_level(90.0)
        assert result["status"] == "FAILED"

    def test_critical_db_level(self, calculator):
        result = calculator.analyze_db_level(100.0)
        assert result["status"] == "CRITICAL"

    def test_negative_db_raises_error(self, calculator):
        with pytest.raises(ValueError):
            calculator.analyze_db_level(-10.0)


class TestFrequencyPatternAnalysis:
    def test_normal_frequency(self, calculator):
        result = calculator.analyze_frequency_pattern(3000.0, TrapType.THERMODYNAMIC)
        assert result["pattern"] == "NORMAL"

    def test_low_frequency(self, calculator):
        result = calculator.analyze_frequency_pattern(500.0, TrapType.THERMODYNAMIC)
        assert result["pattern"] == "LOW_ACTIVITY"


class TestCombinedAcousticSignals:
    def test_healthy_combined(self, calculator):
        result = calculator.combine_acoustic_signals(65.0, 3000.0, TrapType.THERMODYNAMIC)
        assert result["overall_status"] == "HEALTHY"

    def test_failed_combined(self, calculator):
        result = calculator.combine_acoustic_signals(98.0, 8000.0, TrapType.THERMODYNAMIC)
        assert result["overall_status"] == "FAILED"

    def test_provenance_hash(self, calculator):
        result = calculator.combine_acoustic_signals(65.0, 3000.0, TrapType.THERMODYNAMIC)
        assert len(result["provenance_hash"]) == 64
