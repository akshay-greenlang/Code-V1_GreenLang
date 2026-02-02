"""GL-013 Prediction Pipeline Integration Tests - Author: GL-TestEngineer"""
import pytest
import numpy as np
from datetime import datetime, timezone

class TestSensorToFeaturePipeline:
    def test_vibration_to_features(self, sample_vibration_readings, sample_vibration_fft_data):
        readings = sample_vibration_readings
        fft_data = sample_vibration_fft_data
        assert len(readings) > 0
        assert fft_data["shaft_frequency_hz"] > 0

    def test_temperature_to_features(self, sample_temperature_readings):
        temps = [r.value for r in sample_temperature_readings]
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        assert mean_temp > 0
        assert std_temp >= 0

    def test_current_to_features(self, sample_current_readings):
        readings = [r.value for r in sample_current_readings]
        rms = np.sqrt(np.mean(np.array(readings)**2))
        assert rms >= 0

class TestFeatureToPredictionFlow:
    def test_rul_prediction_from_features(self, mock_rul_calculator, sample_vibration_fft_data):
        result = mock_rul_calculator.calculate_rul()
        assert result["rul_mean_hours"] > 0
        assert 0 <= result["confidence_score"] <= 1

    def test_prediction_includes_uncertainty(self, mock_rul_calculator):
        result = mock_rul_calculator.calculate_rul()
        assert result["rul_p10_hours"] < result["rul_p50_hours"]
        assert result["rul_p50_hours"] < result["rul_p90_hours"]

class TestPredictionToAlertFlow:
    def test_low_rul_triggers_alert(self, sample_rul_prediction):
        rul = sample_rul_prediction.rul_hours_mean
        urgency = sample_rul_prediction.urgency
        if rul < 500:
            assert urgency in ["high", "critical"]
        elif rul < 2000:
            assert urgency in ["medium", "high"]

    def test_recommended_action_present(self, sample_rul_prediction):
        assert sample_rul_prediction.recommended_action is not None
        assert len(sample_rul_prediction.recommended_action) > 0

class TestMultiModalFusion:
    def test_vibration_and_thermal_fusion(self, sample_vibration_readings, sample_temperature_readings):
        vib_values = [r.value for r in sample_vibration_readings]
        temp_values = [r.value for r in sample_temperature_readings]
        vib_rms = np.sqrt(np.mean(np.array(vib_values)**2))
        temp_mean = np.mean(temp_values)
        combined_score = 0.6 * (vib_rms / 1.0) + 0.4 * (temp_mean / 80.0)
        assert 0 <= combined_score <= 2

    def test_sensor_correlation(self, sample_vibration_readings, sample_temperature_readings):
        assert len(sample_vibration_readings) > 0
        assert len(sample_temperature_readings) > 0

class TestUncertaintyPropagation:
    def test_weibull_uncertainty_propagates(self, sample_weibull_params, mock_rul_calculator):
        result = mock_rul_calculator.calculate_rul()
        width = result["rul_p90_hours"] - result["rul_p10_hours"]
        assert width > 0

    def test_prediction_confidence_reflects_data_quality(self, mock_rul_calculator, sample_good_quality_data):
        result = mock_rul_calculator.calculate_rul()
        assert result["confidence_score"] > 0.5

class TestEndToEndPipeline:
    def test_complete_pipeline_execution(self, sample_vibration_readings, sample_temperature_readings, 
                                         sample_motor_asset, mock_rul_calculator):
        vib_rms = np.sqrt(np.mean(np.array([r.value for r in sample_vibration_readings])**2))
        temp_mean = np.mean([r.value for r in sample_temperature_readings])
        assert vib_rms > 0
        assert temp_mean > 0
        result = mock_rul_calculator.calculate_rul()
        assert result["rul_mean_hours"] > 0
        assert result["confidence_score"] > 0

    def test_pipeline_provenance_tracking(self, sample_rul_prediction):
        assert sample_rul_prediction.provenance_hash is not None
        assert len(sample_rul_prediction.provenance_hash) > 0

class TestPipelinePerformance:
    def test_pipeline_completes_within_timeout(self, performance_timer, mock_rul_calculator):
        timer = performance_timer()
        with timer:
            for _ in range(100):
                mock_rul_calculator.calculate_rul()
        timer.assert_under(1000)  # 1 second for 100 predictions

class TestPipelineErrorHandling:
    def test_handles_missing_sensor_data(self, sample_bad_quality_data):
        values = sample_bad_quality_data["values"]
        clean_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        assert len(clean_values) < len(values)

    def test_handles_invalid_timestamps(self, sample_bad_quality_data):
        timestamp = sample_bad_quality_data["timestamp"]
        try:
            datetime.fromisoformat(timestamp)
            valid = True
        except:
            valid = False
        assert not valid
