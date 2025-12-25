"""
GL-013 Signal Processing Tests - Author: GL-TestEngineer

Unit tests for vibration analysis, FFT processing, bearing frequency calculations,
thermal feature extraction, and ISO 10816 vibration severity classification.
"""

import pytest
import numpy as np
import math


class TestFFTAccuracy:
    """Test Fast Fourier Transform accuracy and peak detection."""

    def test_fft_peak_detection(self, mock_vibration_analyzer):
        """Test FFT correctly identifies peak frequency."""
        result = mock_vibration_analyzer.compute_fft()
        assert result["peak_frequency"] > 0
        assert result["peak_amplitude"] > 0

    def test_fft_shaft_frequency(self, sample_vibration_fft_data):
        """Test shaft frequency extraction from FFT."""
        shaft_hz = sample_vibration_fft_data["shaft_frequency_hz"]
        assert shaft_hz > 0
        assert shaft_hz < 100

    def test_fft_harmonic_detection(self, sample_vibration_fft_data):
        """Test detection of shaft harmonics (1X, 2X, 3X)."""
        peak_1x = sample_vibration_fft_data["peak_1x"]
        peak_2x = sample_vibration_fft_data["peak_2x"]
        peak_3x = sample_vibration_fft_data["peak_3x"]
        assert peak_1x > peak_2x
        assert peak_2x > peak_3x

    def test_fft_overall_rms(self, sample_vibration_fft_data):
        """Test overall RMS calculation from FFT."""
        overall_rms = sample_vibration_fft_data["overall_rms"]
        assert overall_rms > 0
        assert overall_rms < 10

    def test_fft_crest_factor(self, sample_vibration_fft_data):
        """Test crest factor calculation (peak/RMS)."""
        crest_factor = sample_vibration_fft_data["crest_factor"]
        assert 1.0 < crest_factor < 20.0


class TestBearingFrequencyCalculations:
    """Test bearing fault frequency calculations per ISO 15243."""

    def test_bpfo_calculation(self, sample_bearing_6205, sample_bearing_frequencies_6205):
        """Test Ball Pass Frequency Outer race calculation."""
        shaft_freq = sample_bearing_frequencies_6205["shaft_frequency_hz"]
        bpfo = sample_bearing_frequencies_6205["bpfo_hz"]
        n = sample_bearing_6205["num_rolling_elements"]
        bd = sample_bearing_6205["ball_diameter_mm"]
        pd = sample_bearing_6205["pitch_diameter_mm"]
        theta = math.radians(sample_bearing_6205["contact_angle_deg"])
        expected = (n / 2) * shaft_freq * (1 - (bd / pd) * math.cos(theta))
        assert abs(bpfo - expected) < 0.01

    def test_bpfi_calculation(self, sample_bearing_6205, sample_bearing_frequencies_6205):
        """Test Ball Pass Frequency Inner race calculation."""
        shaft_freq = sample_bearing_frequencies_6205["shaft_frequency_hz"]
        bpfi = sample_bearing_frequencies_6205["bpfi_hz"]
        n = sample_bearing_6205["num_rolling_elements"]
        bd = sample_bearing_6205["ball_diameter_mm"]
        pd = sample_bearing_6205["pitch_diameter_mm"]
        theta = math.radians(sample_bearing_6205["contact_angle_deg"])
        expected = (n / 2) * shaft_freq * (1 + (bd / pd) * math.cos(theta))
        assert abs(bpfi - expected) < 0.01

    def test_bsf_calculation(self, sample_bearing_6205, sample_bearing_frequencies_6205):
        """Test Ball Spin Frequency calculation."""
        bsf = sample_bearing_frequencies_6205["bsf_hz"]
        assert bsf > 0

    def test_ftf_calculation(self, sample_bearing_6205, sample_bearing_frequencies_6205):
        """Test Fundamental Train Frequency (cage) calculation."""
        ftf = sample_bearing_frequencies_6205["ftf_hz"]
        shaft_freq = sample_bearing_frequencies_6205["shaft_frequency_hz"]
        assert 0.3 * shaft_freq < ftf < 0.5 * shaft_freq

    def test_frequency_ordering(self, sample_bearing_frequencies_6205):
        """Test expected ordering of bearing frequencies."""
        ftf = sample_bearing_frequencies_6205["ftf_hz"]
        bpfo = sample_bearing_frequencies_6205["bpfo_hz"]
        bpfi = sample_bearing_frequencies_6205["bpfi_hz"]
        assert ftf < bpfo < bpfi

    def test_bearing_fault_detection(self, mock_vibration_analyzer):
        """Test bearing fault detection logic."""
        result = mock_vibration_analyzer.detect_bearing_faults()
        assert "bpfo_detected" in result
        assert "bpfi_detected" in result
        assert "bsf_detected" in result
        assert "ftf_detected" in result
        assert result["overall_severity"] in ["normal", "warning", "alarm", "danger"]



class TestThermalFeatureExtraction:
    """Test thermal sensor feature extraction."""

    def test_temperature_mean(self, sample_temperature_readings):
        """Test mean temperature calculation."""
        temps = [r.value for r in sample_temperature_readings]
        mean_temp = np.mean(temps)
        assert 50 < mean_temp < 80

    def test_temperature_std(self, sample_temperature_readings):
        """Test temperature standard deviation calculation."""
        temps = [r.value for r in sample_temperature_readings]
        std_temp = np.std(temps)
        assert std_temp >= 0
        assert std_temp < 10

    def test_temperature_trend(self, sample_temperature_readings):
        """Test temperature trend detection."""
        temps = [r.value for r in sample_temperature_readings]
        x = np.arange(len(temps))
        if len(temps) > 1:
            slope = np.polyfit(x, temps, 1)[0]
            assert abs(slope) < 1.0

    def test_temperature_max(self, sample_temperature_readings):
        """Test maximum temperature detection."""
        temps = [r.value for r in sample_temperature_readings]
        max_temp = max(temps)
        assert max_temp < 100

    def test_thermal_rate_of_change(self, sample_temperature_readings):
        """Test rate of temperature change."""
        temps = [r.value for r in sample_temperature_readings]
        if len(temps) > 1:
            rate_of_change = np.diff(temps)
            max_rate = np.max(np.abs(rate_of_change))
            assert max_rate < 5.0


class TestVibrationISO10816:
    """Test ISO 10816-3 vibration severity classification."""

    def test_zone_a_classification(self, sample_vibration_limits_class_ii):
        """Test Zone A (newly commissioned) limits."""
        limits = sample_vibration_limits_class_ii
        assert "zone_A" in limits
        assert limits["zone_A"]["max_velocity_mm_s"] == 2.8

    def test_zone_b_classification(self, sample_vibration_limits_class_ii):
        """Test Zone B (unrestricted operation) limits."""
        limits = sample_vibration_limits_class_ii
        assert "zone_B" in limits
        assert limits["zone_B"]["max_velocity_mm_s"] == 7.1

    def test_zone_c_classification(self, sample_vibration_limits_class_ii):
        """Test Zone C (restricted operation) limits."""
        limits = sample_vibration_limits_class_ii
        assert "zone_C" in limits
        assert limits["zone_C"]["max_velocity_mm_s"] == 11.2

    def test_zone_d_classification(self, sample_vibration_limits_class_ii):
        """Test Zone D (damage may occur) limits."""
        limits = sample_vibration_limits_class_ii
        assert "zone_D" in limits
        assert limits["zone_D"]["min_velocity_mm_s"] == 11.2

    def test_zone_ordering(self, sample_vibration_limits_class_ii):
        """Test zones are ordered by severity."""
        limits = sample_vibration_limits_class_ii
        assert limits["zone_A"]["max_velocity_mm_s"] < limits["zone_B"]["max_velocity_mm_s"]
        assert limits["zone_B"]["max_velocity_mm_s"] < limits["zone_C"]["max_velocity_mm_s"]

    def test_velocity_classification(self, sample_vibration_limits_class_ii):
        """Test velocity value classification into zones."""
        limits = sample_vibration_limits_class_ii

        def classify_zone(velocity_mm_s):
            if velocity_mm_s <= limits["zone_A"]["max_velocity_mm_s"]:
                return "A"
            elif velocity_mm_s <= limits["zone_B"]["max_velocity_mm_s"]:
                return "B"
            elif velocity_mm_s <= limits["zone_C"]["max_velocity_mm_s"]:
                return "C"
            else:
                return "D"

        assert classify_zone(2.0) == "A"
        assert classify_zone(5.0) == "B"
        assert classify_zone(9.0) == "C"
        assert classify_zone(15.0) == "D"



class TestVibrationTrending:
    """Test vibration trend analysis."""

    def test_overall_rms_level(self, sample_vibration_readings):
        """Test overall RMS vibration level."""
        values = [r.value for r in sample_vibration_readings]
        rms = np.sqrt(np.mean(np.array(values)**2))
        assert rms > 0
        assert rms < 10

    def test_peak_value(self, sample_vibration_readings):
        """Test peak vibration value."""
        values = [r.value for r in sample_vibration_readings]
        peak = np.max(np.abs(values))
        assert peak > 0

    def test_vibration_stability(self, sample_vibration_readings):
        """Test vibration reading stability."""
        values = [r.value for r in sample_vibration_readings]
        std = np.std(values)
        mean = np.mean(values)
        cv = std / abs(mean) if mean != 0 else 0
        assert cv < 1.0

    def test_kurtosis_calculation(self, sample_vibration_fft_data):
        """Test kurtosis as indicator of impulsiveness."""
        kurtosis = sample_vibration_fft_data["kurtosis"]
        assert kurtosis >= 0

    def test_trend_direction(self, sample_vibration_readings):
        """Test trend direction detection."""
        values = [r.value for r in sample_vibration_readings]
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        assert isinstance(slope, float)


class TestCurrentSignatureAnalysis:
    """Test Motor Current Signature Analysis (MCSA)."""

    def test_current_rms(self, sample_current_readings):
        """Test RMS current calculation."""
        values = [r.value for r in sample_current_readings]
        rms = np.sqrt(np.mean(np.array(values)**2))
        assert rms > 0

    def test_current_imbalance_detection(self, sample_current_readings):
        """Test current imbalance detection logic."""
        values = [r.value for r in sample_current_readings]
        cv = np.std(values) / np.mean(values)
        assert cv < 0.1

    def test_current_range(self, sample_current_readings):
        """Test current readings are within expected range."""
        for reading in sample_current_readings:
            assert 0 < reading.value < 100


class TestFeatureExtraction:
    """Test signal feature extraction."""

    def test_feature_extraction_complete(self, mock_vibration_analyzer):
        """Test all features are extracted."""
        result = mock_vibration_analyzer.extract_features()
        required_features = ["rms", "peak", "crest_factor", "kurtosis", "skewness"]
        for feature in required_features:
            assert feature in result
            assert result[feature] is not None

    def test_crest_factor_bounds(self, mock_vibration_analyzer):
        """Test crest factor is within reasonable bounds."""
        result = mock_vibration_analyzer.extract_features()
        cf = result["crest_factor"]
        assert 1.0 <= cf <= 20.0

    def test_skewness_bounds(self, mock_vibration_analyzer):
        """Test skewness is within reasonable bounds."""
        result = mock_vibration_analyzer.extract_features()
        skewness = result["skewness"]
        assert -5.0 <= skewness <= 5.0
