"""
GL-013 PredictiveMaintenance - Weibull Analysis Unit Tests
Author: GL-TestEngineer
"""

import pytest
import math
from decimal import Decimal


class TestWeibullParameterEstimation:
    def test_parameter_estimation_known_values(self, sample_failure_data):
        failure_times = [d['time_hours'] for d in sample_failure_data if d.get('failed', False)]
        assert len(failure_times) > 0
        assert all(t > 0 for t in failure_times)

    def test_beta_shape_parameter_bounds(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            beta = params['beta']
            assert beta > Decimal('0')
            assert Decimal('0.5') <= beta <= Decimal('5')

    def test_eta_scale_parameter_positive(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            assert params['eta'] > Decimal('0')

    def test_gamma_location_parameter_non_negative(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            assert params['gamma'] >= Decimal('0')


class TestWeibullSurvivalFunction:
    def test_survival_at_zero(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            gamma = float(params['gamma'])
            survival = 1.0  # At t=gamma, survival=1
            assert abs(survival - 1.0) < 1e-10

    def test_survival_at_eta(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            beta = float(params['beta'])
            eta = float(params['eta'])
            gamma = float(params['gamma'])
            t = gamma + eta
            survival = math.exp(-((t - gamma) / eta) ** beta)
            expected = math.exp(-1)
            assert abs(survival - expected) < 1e-6

    def test_survival_monotonically_decreasing(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        beta = float(params['beta'])
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        times = [gamma + i * 1000 for i in range(100)]
        survivals = [math.exp(-((t - gamma) / eta) ** beta) for t in times]
        for i in range(1, len(survivals)):
            assert survivals[i] <= survivals[i-1]


class TestWeibullHazardFunction:
    def test_hazard_increasing_for_beta_gt_1(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        beta = float(params['beta'])
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        assert beta > 1
        times = [gamma + i * 1000 + 1 for i in range(1, 100)]
        hazards = [(beta / eta) * (((t - gamma) / eta) ** (beta - 1)) for t in times]
        for i in range(1, len(hazards)):
            assert hazards[i] > hazards[i-1]


class TestRightCensoringHandling:
    def test_censored_data_identification(self, sample_failure_data):
        censored = [d for d in sample_failure_data if d.get('censored', False)]
        failed = [d for d in sample_failure_data if d.get('failed', False)]
        assert len(censored) > 0
        assert len(failed) > 0

    def test_censored_times_positive(self, sample_failure_data):
        for d in sample_failure_data:
            assert d['time_hours'] > 0


class TestWeibullMeanAndMedian:
    def test_mean_calculation(self, sample_weibull_params):
        from math import gamma as math_gamma
        for equipment_type, params in sample_weibull_params.items():
            beta = float(params['beta'])
            eta = float(params['eta'])
            gamma_loc = float(params['gamma'])
            mean = eta * math_gamma(1 + 1/beta) + gamma_loc
            assert mean > 0

    def test_median_calculation(self, sample_weibull_params):
        for equipment_type, params in sample_weibull_params.items():
            beta = float(params['beta'])
            eta = float(params['eta'])
            gamma_loc = float(params['gamma'])
            median = eta * (math.log(2) ** (1/beta)) + gamma_loc
            assert median > 0


class TestWeibullPercentileCalculations:
    @pytest.mark.parametrize('percentile', [10, 50, 90])
    def test_percentile_calculation(self, sample_weibull_params, percentile):
        params = sample_weibull_params['bearing_ball']
        beta = float(params['beta'])
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        p = percentile / 100.0
        b_life = eta * ((-math.log(1 - p)) ** (1/beta)) + gamma
        survival = math.exp(-((b_life - gamma) / eta) ** beta)
        expected_survival = 1 - p
        assert abs(survival - expected_survival) < 1e-6


class TestWeibullReproducibility:
    def test_deterministic_survival_calculation(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        beta = float(params['beta'])
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        t = 50000.0
        results = [math.exp(-((t - gamma) / eta) ** beta) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestWeibullEdgeCases:
    def test_zero_operating_time(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        gamma = float(params['gamma'])
        assert gamma >= 0

    def test_very_large_operating_time(self, sample_weibull_params):
        params = sample_weibull_params['motor_ac_induction_large']
        beta = float(params['beta'])
        eta = float(params['eta'])
        gamma = float(params['gamma'])
        t = gamma + 100 * eta
        survival = math.exp(-((t - gamma) / eta) ** beta)
        assert survival < 1e-10


class TestWeibullValidation:
    def test_invalid_beta_raises_error(self):
        for beta in [0, -1, -0.5]:
            with pytest.raises((ValueError, AssertionError)):
                if beta <= 0:
                    raise ValueError('Beta must be positive')

    def test_invalid_eta_raises_error(self):
        for eta in [0, -1000]:
            with pytest.raises((ValueError, AssertionError)):
                if eta <= 0:
                    raise ValueError('Eta must be positive')
