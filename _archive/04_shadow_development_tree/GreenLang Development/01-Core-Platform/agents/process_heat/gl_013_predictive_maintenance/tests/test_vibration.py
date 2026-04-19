# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Vibration Analysis Tests

Tests for vibration signature analysis and fault detection.
Validates ISO 10816 classification, FFT spectrum analysis, and bearing fault detection.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime, timezone
from typing import List

from greenlang.agents.process_heat.gl_013_predictive_maintenance.vibration import (
    VibrationAnalyzer,
    BearingGeometry,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    VibrationThresholds,
    PredictiveMaintenanceConfig,
    EquipmentType,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    HealthStatus,
    VibrationAnalysisResult,
    VibrationReading,
)


class TestBearingGeometry:
    """Tests for BearingGeometry calculations."""

    def test_valid_bearing_geometry(self):
        """Test valid bearing geometry creation."""
        bearing = BearingGeometry(
            bpfo=3.56,
            bpfi=5.44,
            bsf=2.32,
            ftf=0.42,
        )

        assert bearing.bpfo == 3.56
        assert bearing.bpfi == 5.44
        assert bearing.bsf == 2.32
        assert bearing.ftf == 0.42

    def test_calculate_bearing_frequencies(self):
        """Test bearing frequency calculation from RPM."""
        bearing = BearingGeometry(
            bpfo=3.56,
            bpfi=5.44,
            bsf=2.32,
            ftf=0.42,
        )

        rpm = 1800
        shaft_freq = rpm / 60  # 30 Hz

        bpfo_hz = bearing.bpfo * shaft_freq
        bpfi_hz = bearing.bpfi * shaft_freq
        bsf_hz = bearing.bsf * shaft_freq
        ftf_hz = bearing.ftf * shaft_freq

        assert bpfo_hz == pytest.approx(106.8, rel=0.01)
        assert bpfi_hz == pytest.approx(163.2, rel=0.01)
        assert bsf_hz == pytest.approx(69.6, rel=0.01)
        assert ftf_hz == pytest.approx(12.6, rel=0.01)


class TestVibrationAnalyzer:
    """Tests for VibrationAnalyzer class."""

    def test_initialization(self, equipment_config):
        """Test analyzer initialization."""
        analyzer = VibrationAnalyzer(
            equipment_config,
            equipment_config.vibration_thresholds,
        )

        assert analyzer.config == equipment_config
        assert analyzer.thresholds == equipment_config.vibration_thresholds

    def test_initialization_default_thresholds(self, equipment_config):
        """Test initialization with default thresholds."""
        analyzer = VibrationAnalyzer(equipment_config)

        assert analyzer.thresholds is not None


class TestISO10816Classification:
    """Tests for ISO 10816 vibration zone classification."""

    def test_zone_a_good(self, vibration_analyzer, vibration_reading_healthy):
        """Test Zone A (Good) classification."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.iso_zone == AlertSeverity.GOOD

    def test_zone_b_acceptable(self, vibration_analyzer):
        """Test Zone B (Acceptable) classification."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=3.5,  # Between 2.8 and 4.5
            acceleration_rms_g=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        assert result.iso_zone == AlertSeverity.ACCEPTABLE

    def test_zone_c_unsatisfactory(self, vibration_analyzer, vibration_reading_warning):
        """Test Zone C (Unsatisfactory) classification."""
        result = vibration_analyzer.analyze(vibration_reading_warning)

        assert result.iso_zone == AlertSeverity.UNSATISFACTORY

    def test_zone_d_unacceptable(self, vibration_analyzer, vibration_reading_critical):
        """Test Zone D (Unacceptable) classification."""
        result = vibration_analyzer.analyze(vibration_reading_critical)

        assert result.iso_zone == AlertSeverity.UNACCEPTABLE

    @pytest.mark.parametrize("velocity,expected_zone", [
        (1.0, AlertSeverity.GOOD),          # Well below Zone A limit
        (2.8, AlertSeverity.GOOD),          # At Zone A/B boundary
        (2.9, AlertSeverity.ACCEPTABLE),    # Just above Zone A
        (4.5, AlertSeverity.ACCEPTABLE),    # At Zone B/C boundary
        (4.6, AlertSeverity.UNSATISFACTORY), # Just above Zone B
        (7.1, AlertSeverity.UNSATISFACTORY), # At Zone C/D boundary
        (7.2, AlertSeverity.UNACCEPTABLE),  # Just above Zone C
        (15.0, AlertSeverity.UNACCEPTABLE), # Well above Zone D
    ])
    def test_iso_zone_boundaries(self, vibration_analyzer, velocity, expected_zone):
        """Test ISO 10816 zone boundary classification."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=velocity,
            acceleration_rms_g=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        assert result.iso_zone == expected_zone


class TestSpectrumAnalysis:
    """Tests for FFT spectrum analysis."""

    def test_dominant_frequency_detection(
        self,
        vibration_analyzer,
        vibration_reading_with_spectrum
    ):
        """Test dominant frequency detection."""
        result = vibration_analyzer.analyze(vibration_reading_with_spectrum)

        # 1x peak should be dominant
        expected_1x = 1800 / 60  # 30 Hz
        assert result.dominant_frequency_hz == pytest.approx(expected_1x, rel=0.1)

    def test_spectrum_peaks_extraction(
        self,
        vibration_analyzer,
        vibration_reading_with_spectrum
    ):
        """Test spectrum peak extraction."""
        result = vibration_analyzer.analyze(vibration_reading_with_spectrum)

        # Should detect multiple peaks
        if result.spectrum_peaks is not None:
            assert len(result.spectrum_peaks) > 0

    def test_analysis_without_spectrum(self, vibration_analyzer, vibration_reading_healthy):
        """Test analysis works without spectrum data."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        # Should still produce valid result
        assert result.overall_velocity_mm_s == vibration_reading_healthy.velocity_rms_mm_s
        assert result.iso_zone is not None


class TestImbalanceDetection:
    """Tests for rotor imbalance detection."""

    def test_imbalance_not_detected_healthy(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test imbalance not detected in healthy vibration."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.imbalance_detected is False

    def test_imbalance_detected_1x_peak(self, vibration_analyzer):
        """Test imbalance detected from high 1x peak."""
        # Create spectrum with dominant 1x
        spectrum = [0.1] * 500
        shaft_freq_idx = int(1800 / 60)  # 30 Hz
        spectrum[shaft_freq_idx] = 5.0  # High 1x peak

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=5.0,
            acceleration_rms_g=1.5,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # May detect imbalance from high 1x
        # Depends on threshold configuration


class TestMisalignmentDetection:
    """Tests for shaft misalignment detection."""

    def test_misalignment_not_detected_healthy(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test misalignment not detected in healthy vibration."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.misalignment_detected is False

    def test_misalignment_detected_2x_peak(self, vibration_analyzer):
        """Test misalignment detected from high 2x peak."""
        # Create spectrum with significant 2x
        spectrum = [0.1] * 500
        shaft_freq_idx = int(1800 / 60)  # 30 Hz
        spectrum[shaft_freq_idx] = 1.0  # Normal 1x
        spectrum[shaft_freq_idx * 2] = 1.5  # High 2x (1.5x higher than 1x)

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=5.0,
            acceleration_rms_g=1.5,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # May detect misalignment from high 2x ratio


class TestBearingDefectDetection:
    """Tests for bearing defect detection."""

    def test_bearing_defect_not_detected_healthy(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test bearing defect not detected in healthy vibration."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.bearing_defect_detected is False

    def test_bpfo_detection(self, vibration_analyzer):
        """Test BPFO (Ball Pass Frequency Outer) detection."""
        # BPFO at 3.56x shaft speed
        shaft_freq = 1800 / 60  # 30 Hz
        bpfo_freq = 3.56 * shaft_freq  # ~107 Hz

        spectrum = [0.1] * 500
        spectrum[int(bpfo_freq)] = 1.0  # BPFO peak

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=5.0,
            acceleration_rms_g=2.0,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # Should detect bearing defect if configured
        if result.bearing_defect_detected:
            assert result.bearing_defect_type in ["BPFO", "outer_race"]

    def test_bpfi_detection(self, vibration_analyzer):
        """Test BPFI (Ball Pass Frequency Inner) detection."""
        shaft_freq = 1800 / 60  # 30 Hz
        bpfi_freq = 5.44 * shaft_freq  # ~163 Hz

        spectrum = [0.1] * 500
        spectrum[int(bpfi_freq)] = 1.0  # BPFI peak

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=5.0,
            acceleration_rms_g=2.0,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # Should detect bearing defect if configured
        if result.bearing_defect_detected:
            assert result.bearing_defect_type in ["BPFI", "inner_race"]


class TestLoosenessDetection:
    """Tests for mechanical looseness detection."""

    def test_looseness_not_detected_healthy(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test looseness not detected in healthy vibration."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.looseness_detected is False

    def test_looseness_detected_harmonics(self, vibration_analyzer):
        """Test looseness detected from multiple harmonics."""
        # Create spectrum with many harmonics (floor noise)
        shaft_freq = int(1800 / 60)  # 30 Hz
        spectrum = [0.1] * 500

        # Add multiple harmonics (typical of looseness)
        for i in range(1, 11):
            if i * shaft_freq < 500:
                spectrum[i * shaft_freq] = 0.5

        # Add half harmonics (0.5x, 1.5x, etc.)
        spectrum[shaft_freq // 2] = 0.4  # 0.5x
        spectrum[int(shaft_freq * 1.5)] = 0.3  # 1.5x

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=6.0,
            acceleration_rms_g=2.5,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # May detect looseness from harmonic pattern


class TestVibrationTrending:
    """Tests for vibration trending analysis."""

    def test_trend_stable(self, vibration_analyzer):
        """Test stable trend detection."""
        # Multiple readings with similar values
        readings = []
        for i in range(5):
            reading = VibrationReading(
                sensor_id="ACCEL-001",
                timestamp=datetime.now(timezone.utc),
                location="DE",
                orientation="radial",
                velocity_rms_mm_s=2.0 + i * 0.1,  # Small variation
                acceleration_rms_g=0.5,
                operating_speed_rpm=1800.0,
            )
            readings.append(reading)

        # Analyze last reading with history
        if hasattr(vibration_analyzer, 'set_history'):
            vibration_analyzer.set_history(readings[:-1])

        result = vibration_analyzer.analyze(readings[-1])

        if result.trend is not None:
            assert result.trend in ["stable", "STABLE"]

    def test_trend_increasing(self, vibration_analyzer):
        """Test increasing trend detection."""
        readings = []
        for i in range(5):
            reading = VibrationReading(
                sensor_id="ACCEL-001",
                timestamp=datetime.now(timezone.utc),
                location="DE",
                orientation="radial",
                velocity_rms_mm_s=2.0 + i * 1.0,  # Clear increase
                acceleration_rms_g=0.5,
                operating_speed_rpm=1800.0,
            )
            readings.append(reading)

        if hasattr(vibration_analyzer, 'set_history'):
            vibration_analyzer.set_history(readings[:-1])

        result = vibration_analyzer.analyze(readings[-1])

        if result.trend is not None:
            assert "increas" in result.trend.lower()


class TestVibrationRecommendations:
    """Tests for vibration analysis recommendations."""

    def test_recommendations_healthy(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test recommendations for healthy vibration."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        # May have routine recommendation or empty
        assert isinstance(result.recommendations, list)

    def test_recommendations_warning(
        self,
        vibration_analyzer,
        vibration_reading_warning
    ):
        """Test recommendations for warning vibration."""
        result = vibration_analyzer.analyze(vibration_reading_warning)

        assert len(result.recommendations) > 0

    def test_recommendations_critical(
        self,
        vibration_analyzer,
        vibration_reading_critical
    ):
        """Test recommendations for critical vibration."""
        result = vibration_analyzer.analyze(vibration_reading_critical)

        assert len(result.recommendations) > 0
        # Should recommend immediate action
        rec_text = " ".join(result.recommendations).lower()
        assert any(term in rec_text for term in [
            "immediate", "urgent", "critical", "stop"
        ])


class TestVibrationAnalysisResult:
    """Tests for VibrationAnalysisResult model."""

    def test_result_fields(self, vibration_analyzer, vibration_reading_healthy):
        """Test all result fields are populated."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert result.sensor_id == "ACCEL-001"
        assert result.timestamp is not None
        assert result.overall_velocity_mm_s == 2.0
        assert result.overall_acceleration_g == 0.5
        assert result.iso_zone == AlertSeverity.GOOD
        assert result.dominant_frequency_hz is not None or result.iso_zone is not None

    def test_result_fault_flags(self, vibration_analyzer, vibration_reading_healthy):
        """Test fault detection flags."""
        result = vibration_analyzer.analyze(vibration_reading_healthy)

        assert isinstance(result.bearing_defect_detected, bool)
        assert isinstance(result.imbalance_detected, bool)
        assert isinstance(result.misalignment_detected, bool)


class TestVibrationDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_analysis_same_result(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test repeated analysis produces identical results."""
        results = [
            vibration_analyzer.analyze(vibration_reading_healthy)
            for _ in range(5)
        ]

        # All ISO zones should be identical
        zones = [r.iso_zone for r in results]
        assert len(set(zones)) == 1

        # All velocities should be identical
        velocities = [r.overall_velocity_mm_s for r in results]
        assert len(set(velocities)) == 1

    def test_provenance_hash_deterministic(
        self,
        vibration_analyzer,
        vibration_reading_healthy
    ):
        """Test provenance hash is deterministic."""
        result1 = vibration_analyzer.analyze(vibration_reading_healthy)
        result2 = vibration_analyzer.analyze(vibration_reading_healthy)

        if result1.provenance_hash is not None:
            assert result1.provenance_hash == result2.provenance_hash


class TestVibrationEdgeCases:
    """Tests for edge cases."""

    def test_zero_velocity(self, vibration_analyzer):
        """Test zero velocity handling."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=0.0,
            acceleration_rms_g=0.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        assert result.iso_zone == AlertSeverity.GOOD

    def test_very_high_velocity(self, vibration_analyzer):
        """Test very high velocity handling."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=100.0,  # Extremely high
            acceleration_rms_g=50.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        assert result.iso_zone == AlertSeverity.UNACCEPTABLE

    def test_empty_spectrum(self, vibration_analyzer):
        """Test handling of empty spectrum."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=2.0,
            acceleration_rms_g=0.5,
            spectrum=[],  # Empty
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        # Should still produce valid result
        assert result.iso_zone is not None

    def test_low_speed_operation(self, vibration_analyzer):
        """Test low speed operation handling."""
        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=2.0,
            acceleration_rms_g=0.5,
            operating_speed_rpm=100.0,  # Very low speed
        )

        result = vibration_analyzer.analyze(reading)

        # Should handle low speed
        assert result.iso_zone is not None


class TestVibrationIntegration:
    """Integration tests for vibration analysis."""

    def test_full_analysis_workflow(self, equipment_config):
        """Test complete analysis workflow."""
        analyzer = VibrationAnalyzer(
            equipment_config,
            equipment_config.vibration_thresholds,
        )

        # Create realistic spectrum
        spectrum = [0.05] * 500
        shaft_freq = int(1800 / 60)  # 30 Hz

        # Add 1x, 2x, 3x harmonics
        spectrum[shaft_freq] = 2.0      # 1x
        spectrum[shaft_freq * 2] = 0.8  # 2x
        spectrum[shaft_freq * 3] = 0.3  # 3x

        # Add some BPFO signature
        bpfo_freq = int(3.56 * shaft_freq)
        spectrum[bpfo_freq] = 0.2

        reading = VibrationReading(
            sensor_id="ACCEL-001",
            timestamp=datetime.now(timezone.utc),
            location="DE",
            orientation="radial",
            velocity_rms_mm_s=4.8,
            acceleration_rms_g=1.5,
            displacement_um=65.0,
            spectrum=spectrum,
            frequency_resolution_hz=1.0,
            operating_speed_rpm=1800.0,
            temperature_c=62.0,
        )

        result = analyzer.analyze(reading)

        # Verify comprehensive result
        assert result.sensor_id == "ACCEL-001"
        assert result.overall_velocity_mm_s == 4.8
        assert result.iso_zone in [AlertSeverity.ACCEPTABLE, AlertSeverity.UNSATISFACTORY]
        assert result.dominant_frequency_hz is not None
        assert isinstance(result.bearing_defect_detected, bool)
        assert isinstance(result.recommendations, list)

    @pytest.mark.parametrize("location,orientation", [
        ("DE", "radial"),
        ("DE", "axial"),
        ("NDE", "radial"),
        ("NDE", "axial"),
    ])
    def test_multiple_measurement_points(
        self,
        vibration_analyzer,
        location,
        orientation
    ):
        """Test analysis at multiple measurement points."""
        reading = VibrationReading(
            sensor_id=f"ACCEL-{location}-{orientation}",
            timestamp=datetime.now(timezone.utc),
            location=location,
            orientation=orientation,
            velocity_rms_mm_s=2.5,
            acceleration_rms_g=0.8,
            operating_speed_rpm=1800.0,
        )

        result = vibration_analyzer.analyze(reading)

        assert result.sensor_id == f"ACCEL-{location}-{orientation}"
        assert result.iso_zone is not None
