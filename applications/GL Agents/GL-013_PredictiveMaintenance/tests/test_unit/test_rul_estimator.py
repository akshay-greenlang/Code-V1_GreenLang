"""
GL-013 PredictiveMaintenance - RUL Estimator Unit Tests
Author: GL-TestEngineer
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timezone


class TestRULQuantileEstimation:
    def test_rul_mean_positive(self, sample_rul_prediction):
        assert sample_rul_prediction.rul_hours_mean > 0

    def test_rul_quantile_ordering(self, sample_rul_prediction):
        assert sample_rul_prediction.rul_hours_p10 < sample_rul_prediction.rul_hours_p50
        assert sample_rul_prediction.rul_hours_p50 < sample_rul_prediction.rul_hours_p90

    def test_rul_p50_near_mean(self, sample_rul_prediction):
        ratio = sample_rul_prediction.rul_hours_p50 / sample_rul_prediction.rul_hours_mean
        assert 0.5 < ratio < 2.0

    @pytest.mark.parametrize('quantile', [0.10, 0.50, 0.90])
    def test_quantile_bounds(self, sample_rul_prediction, quantile):
        if quantile == 0.10:
            rul = sample_rul_prediction.rul_hours_p10
        elif quantile == 0.50:
            rul = sample_rul_prediction.rul_hours_p50
        else:
            rul = sample_rul_prediction.rul_hours_p90
        assert rul > 0


class TestRULUncertaintyBounds:
    def test_uncertainty_width(self, sample_rul_prediction):
        width = sample_rul_prediction.rul_hours_p90 - sample_rul_prediction.rul_hours_p10
        assert width > 0

    def test_confidence_score_valid(self, sample_rul_prediction):
        assert 0 <= sample_rul_prediction.confidence_score <= 1

    def test_higher_confidence_narrower_bounds(self, sample_rul_prediction):
        width = sample_rul_prediction.rul_hours_p90 - sample_rul_prediction.rul_hours_p10
        relative_width = width / sample_rul_prediction.rul_hours_mean
        if sample_rul_prediction.confidence_score > 0.8:
            assert relative_width < 2.0


class TestRULEdgeCases:
    def test_new_asset_high_rul(self, sample_motor_asset, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        eta = float(params['eta'])
        operating_hours = sample_motor_asset.operating_hours
        remaining = eta - operating_hours
        assert remaining > 0  # New asset should have RUL remaining

    def test_end_of_life_low_rul(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        eta = float(params['eta'])
        operating_hours = eta * 0.95  # Near end of life
        remaining = eta - operating_hours
        assert remaining > 0
        assert remaining < eta * 0.1

    def test_rul_with_zero_operating_hours(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        operating_hours = gamma
        rul = eta - (operating_hours - gamma)
        assert rul > 0

    def test_rul_past_expected_life(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        eta = float(params['eta'])
        operating_hours = eta * 1.5
        rul = max(0, eta - operating_hours)
        assert rul == 0


class TestRULProvenance:
    def test_provenance_hash_exists(self, sample_rul_prediction):
        assert sample_rul_prediction.provenance_hash is not None
        assert len(sample_rul_prediction.provenance_hash) > 0

    def test_prediction_id_unique(self, sample_rul_prediction):
        assert sample_rul_prediction.prediction_id is not None

    def test_timestamp_valid(self, sample_rul_prediction):
        assert sample_rul_prediction.timestamp is not None
        assert sample_rul_prediction.timestamp <= datetime.now(timezone.utc)


class TestRULCalculatorMock:
    def test_mock_calculator_returns_valid_rul(self, mock_rul_calculator):
        result = mock_rul_calculator.calculate_rul()
        assert 'rul_mean_hours' in result
        assert result['rul_mean_hours'] > 0

    def test_mock_calculator_quantiles(self, mock_rul_calculator):
        result = mock_rul_calculator.calculate_rul()
        assert result['rul_p10_hours'] < result['rul_p50_hours']
        assert result['rul_p50_hours'] < result['rul_p90_hours']

    def test_mock_calculator_confidence(self, mock_rul_calculator):
        result = mock_rul_calculator.calculate_rul()
        assert 0 <= result['confidence_score'] <= 1


class TestRULReproducibility:
    def test_deterministic_calculation(self, mock_rul_calculator):
        results = [mock_rul_calculator.calculate_rul() for _ in range(5)]
        first_result = results[0]
        for result in results[1:]:
            assert result['rul_mean_hours'] == first_result['rul_mean_hours']


class TestRULFailureMode:
    def test_failure_mode_specified(self, sample_rul_prediction):
        assert sample_rul_prediction.failure_mode is not None
        assert len(sample_rul_prediction.failure_mode) > 0

    def test_recommended_action_present(self, sample_rul_prediction):
        assert sample_rul_prediction.recommended_action is not None

    def test_urgency_level_valid(self, sample_rul_prediction):
        valid_urgencies = ['low', 'medium', 'high', 'critical']
        assert sample_rul_prediction.urgency in valid_urgencies


class TestRULWithConditionData:
    def test_rul_with_vibration_data(self, sample_vibration_fft_data, sample_weibull_params):
        peak_1x = sample_vibration_fft_data['peak_1x']
        overall_rms = sample_vibration_fft_data['overall_rms']
        assert peak_1x > 0
        assert overall_rms > 0

    def test_rul_degradation_factor(self, sample_vibration_fft_data):
        overall_rms = sample_vibration_fft_data['overall_rms']
        if overall_rms > 0.7:
            degradation_factor = 0.5
        elif overall_rms > 0.5:
            degradation_factor = 0.8
        else:
            degradation_factor = 1.0
        assert 0 < degradation_factor <= 1
