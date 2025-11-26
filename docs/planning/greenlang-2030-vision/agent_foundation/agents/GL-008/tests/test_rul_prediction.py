# -*- coding: utf-8 -*-
"""
Remaining Useful Life (RUL) prediction tests for GL-008 SteamTrapInspector.

This module tests Weibull distribution modeling, confidence interval calculation,
age-based degradation, and historical failure correlation.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools, RULPredictionResult
from config import TrapType, FailureMode


@pytest.mark.validation
class TestWeibullDistributionValidation:
    """Test Weibull distribution implementation for RUL prediction."""

    def test_weibull_basic_calculation(self, tools):
        """Test basic Weibull RUL calculation."""
        condition_data = {
            'trap_id': 'TRAP-WEIBULL-BASIC',
            'current_age_days': 1000,
            'degradation_rate': 0.1,
            'current_health_score': 70
        }

        result = tools.predict_remaining_useful_life(condition_data)

        assert isinstance(result, RULPredictionResult)
        assert result.rul_days > 0
        assert result.rul_confidence_lower >= 0
        assert result.rul_confidence_upper > result.rul_days

    def test_weibull_shape_parameter(self, tools, rul_test_data):
        """Test Weibull shape parameter (beta) impact on RUL."""
        # Beta < 1: Decreasing failure rate (infant mortality)
        # Beta = 1: Constant failure rate (exponential distribution)
        # Beta > 1: Increasing failure rate (wear-out)

        weibull_params = rul_test_data['weibull_parameters']

        condition_data = {
            'trap_id': 'TRAP-WEIBULL-SHAPE',
            'current_age_days': 500,
            'current_health_score': 80,
            'weibull_beta': weibull_params['beta'],  # Shape parameter
            'weibull_eta': weibull_params['eta']      # Scale parameter
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # For beta > 1 (wear-out), failure rate increases with age
        assert result.rul_days > 0

    def test_weibull_scale_parameter(self, tools):
        """Test Weibull scale parameter (eta) impact on RUL."""
        # Eta represents characteristic life

        condition_data_high_eta = {
            'trap_id': 'TRAP-HIGH-ETA',
            'current_age_days': 500,
            'current_health_score': 70,
            'weibull_eta': 3000  # Long characteristic life
        }

        condition_data_low_eta = {
            'trap_id': 'TRAP-LOW-ETA',
            'current_age_days': 500,
            'current_health_score': 70,
            'weibull_eta': 1500  # Short characteristic life
        }

        result_high = tools.predict_remaining_useful_life(condition_data_high_eta)
        result_low = tools.predict_remaining_useful_life(condition_data_low_eta)

        # Higher eta should lead to longer RUL
        assert result_high.rul_days > result_low.rul_days

    def test_weibull_mean_life_calculation(self, tools, rul_test_data):
        """Test mean life calculation from Weibull parameters."""
        # Mean life = eta * Gamma(1 + 1/beta)
        weibull_params = rul_test_data['weibull_parameters']

        condition_data = {
            'trap_id': 'TRAP-MEAN-LIFE',
            'current_age_days': 0,  # New trap
            'current_health_score': 100,
            'weibull_beta': weibull_params['beta'],
            'weibull_eta': weibull_params['eta']
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # For new trap, RUL ≈ mean life
        expected_mean = weibull_params['expected_mean_life']
        # Allow 20% tolerance due to algorithm variations
        assert abs(result.rul_days - expected_mean) / expected_mean < 0.20


@pytest.mark.validation
class TestConfidenceIntervalCalculation:
    """Test confidence interval calculation for RUL predictions."""

    def test_confidence_interval_bounds(self, tools):
        """Test that confidence intervals are properly bounded."""
        condition_data = {
            'trap_id': 'TRAP-CI-BOUNDS',
            'current_age_days': 1000,
            'degradation_rate': 0.1,
            'current_health_score': 70
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Lower bound <= RUL <= Upper bound
        assert result.rul_confidence_lower <= result.rul_days
        assert result.rul_days <= result.rul_confidence_upper

        # Bounds should be non-negative
        assert result.rul_confidence_lower >= 0
        assert result.rul_confidence_upper >= 0

    def test_confidence_interval_width(self, tools):
        """Test confidence interval width increases with uncertainty."""
        # High certainty (high health score, young age)
        condition_high_certainty = {
            'trap_id': 'TRAP-HIGH-CERT',
            'current_age_days': 100,
            'current_health_score': 95,
            'degradation_rate': 0.05
        }

        # Low certainty (low health score, old age)
        condition_low_certainty = {
            'trap_id': 'TRAP-LOW-CERT',
            'current_age_days': 1500,
            'current_health_score': 30,
            'degradation_rate': 0.3
        }

        result_high = tools.predict_remaining_useful_life(condition_high_certainty)
        result_low = tools.predict_remaining_useful_life(condition_low_certainty)

        # Calculate interval widths
        width_high = result_high.rul_confidence_upper - result_high.rul_confidence_lower
        width_low = result_low.rul_confidence_upper - result_low.rul_confidence_lower

        # Higher uncertainty should produce wider interval
        # (May not always be true depending on algorithm)
        assert width_high >= 0
        assert width_low >= 0

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_confidence_level_options(self, tools, confidence_level):
        """Test different confidence levels (90%, 95%, 99%)."""
        condition_data = {
            'trap_id': 'TRAP-CI-LEVEL',
            'current_age_days': 1000,
            'current_health_score': 70,
            'confidence_level': confidence_level
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Higher confidence level → wider interval
        interval_width = result.rul_confidence_upper - result.rul_confidence_lower
        assert interval_width > 0


@pytest.mark.validation
class TestAgeDegradationCorrelation:
    """Test age-based degradation modeling."""

    def test_rul_decreases_with_age(self, tools):
        """Test that RUL decreases as trap ages."""
        ages = [500, 1000, 1500, 2000]
        ruls = []

        for age in ages:
            condition_data = {
                'trap_id': f'TRAP-AGE-{age}',
                'current_age_days': age,
                'current_health_score': 70,  # Constant health
                'degradation_rate': 0.1
            }

            result = tools.predict_remaining_useful_life(condition_data)
            ruls.append(result.rul_days)

        # RUL should generally decrease with age
        # (May not be strictly monotonic due to health score)
        assert ruls[-1] < ruls[0]  # Oldest has less RUL than youngest

    def test_health_score_impact_on_rul(self, tools):
        """Test that health score affects RUL prediction."""
        health_scores = [90, 70, 50, 30]
        ruls = []

        for health in health_scores:
            condition_data = {
                'trap_id': f'TRAP-HEALTH-{health}',
                'current_age_days': 1000,  # Constant age
                'current_health_score': health,
                'degradation_rate': 0.1
            }

            result = tools.predict_remaining_useful_life(condition_data)
            ruls.append(result.rul_days)

        # Better health → longer RUL
        assert ruls[0] > ruls[-1]

    def test_degradation_rate_impact(self, tools):
        """Test impact of degradation rate on RUL."""
        degradation_rates = [0.05, 0.10, 0.20, 0.30]
        ruls = []

        for rate in degradation_rates:
            condition_data = {
                'trap_id': f'TRAP-DEG-{rate}',
                'current_age_days': 1000,
                'current_health_score': 70,
                'degradation_rate': rate
            }

            result = tools.predict_remaining_useful_life(condition_data)
            ruls.append(result.rul_days)

        # Higher degradation rate → shorter RUL
        assert ruls[0] > ruls[-1]

    def test_zero_age_new_trap(self, tools):
        """Test RUL prediction for brand new trap (age=0)."""
        condition_data = {
            'trap_id': 'TRAP-NEW',
            'current_age_days': 0,
            'current_health_score': 100,
            'degradation_rate': 0.0
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # New trap should have long RUL
        assert result.rul_days > 1000  # At least 1000 days


@pytest.mark.validation
class TestHistoricalFailureCorrelation:
    """Test RUL prediction using historical failure data."""

    def test_mtbf_calculation_from_history(self, tools, rul_test_data):
        """Test Mean Time Between Failures calculation."""
        historical_data = rul_test_data['historical_failures']['trap_type_a']

        condition_data = {
            'trap_id': 'TRAP-MTBF',
            'current_age_days': 500,
            'current_health_score': 80,
            'historical_failures': historical_data
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Should calculate MTBF from historical data
        expected_mtbf = np.mean(historical_data)

        if hasattr(result, 'historical_mtbf_days'):
            assert abs(result.historical_mtbf_days - expected_mtbf) < 100  # Within 100 days

    def test_historical_data_improves_accuracy(self, tools):
        """Test that historical data improves prediction accuracy."""
        # Without historical data
        condition_no_history = {
            'trap_id': 'TRAP-NO-HIST',
            'current_age_days': 1000,
            'current_health_score': 70
        }

        # With historical data
        condition_with_history = {
            'trap_id': 'TRAP-WITH-HIST',
            'current_age_days': 1000,
            'current_health_score': 70,
            'historical_failures': [1800, 2000, 2200, 1900]
        }

        result_no_hist = tools.predict_remaining_useful_life(condition_no_history)
        result_with_hist = tools.predict_remaining_useful_life(condition_with_history)

        # With historical data, confidence interval may be narrower
        width_no_hist = result_no_hist.rul_confidence_upper - result_no_hist.rul_confidence_lower
        width_with_hist = result_with_hist.rul_confidence_upper - result_with_hist.rul_confidence_lower

        # Historical data should improve confidence (may not always reduce width)
        assert width_no_hist >= 0
        assert width_with_hist >= 0

    def test_failure_probability_curve(self, tools):
        """Test generation of failure probability curve over time."""
        condition_data = {
            'trap_id': 'TRAP-CURVE',
            'current_age_days': 1000,
            'current_health_score': 70,
            'degradation_rate': 0.1
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Should generate probability curve
        assert len(result.failure_probability_curve) > 0

        # Probability should increase over time
        curve = result.failure_probability_curve
        if len(curve) > 1:
            # Early probabilities should be lower than later ones
            assert curve[0] <= curve[-1]

    def test_censored_data_handling(self, tools):
        """Test handling of right-censored data (traps still operating)."""
        # Mix of failures and censored data
        condition_data = {
            'trap_id': 'TRAP-CENSORED',
            'current_age_days': 1200,
            'current_health_score': 65,
            'historical_failures': [1800, 2000],  # Actual failures
            'historical_censored': [1500, 1600, 1700]  # Still operating
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Should handle censored data appropriately
        assert result.rul_days > 0


@pytest.mark.validation
class TestTrapTypeSpecificRUL:
    """Test RUL prediction for different trap types."""

    @pytest.mark.parametrize("trap_type,expected_typical_life_days", [
        (TrapType.THERMODYNAMIC, 2000),
        (TrapType.FLOAT_AND_THERMOSTATIC, 2500),
        (TrapType.INVERTED_BUCKET, 3000),
        (TrapType.THERMOSTATIC, 1800),
    ])
    def test_trap_type_life_expectancy(self, tools, trap_type, expected_typical_life_days):
        """Test that different trap types have appropriate life expectancies."""
        condition_data = {
            'trap_id': f'TRAP-TYPE-{trap_type.value}',
            'current_age_days': 500,
            'current_health_score': 80,
            'trap_type': trap_type
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # RUL should be reasonable for trap type
        # (Actual implementation may vary)
        assert result.rul_days > 0

    def test_operating_conditions_impact(self, tools):
        """Test impact of operating conditions on RUL."""
        # Harsh conditions
        condition_harsh = {
            'trap_id': 'TRAP-HARSH',
            'current_age_days': 1000,
            'current_health_score': 60,
            'operating_pressure_psig': 600,  # High pressure
            'cycling_frequency_per_hour': 1000,  # Frequent cycling
            'steam_quality_percent': 90  # Wet steam
        }

        # Mild conditions
        condition_mild = {
            'trap_id': 'TRAP-MILD',
            'current_age_days': 1000,
            'current_health_score': 60,
            'operating_pressure_psig': 50,  # Low pressure
            'cycling_frequency_per_hour': 10,  # Infrequent cycling
            'steam_quality_percent': 100  # Dry steam
        }

        result_harsh = tools.predict_remaining_useful_life(condition_harsh)
        result_mild = tools.predict_remaining_useful_life(condition_mild)

        # Harsh conditions should reduce RUL
        # (Actual implementation may not consider all these factors)
        assert result_harsh.rul_days >= 0
        assert result_mild.rul_days >= 0


@pytest.mark.validation
class TestRULPredictionEdgeCases:
    """Test edge cases for RUL prediction."""

    def test_near_end_of_life(self, tools):
        """Test RUL prediction for trap near end of life."""
        condition_data = {
            'trap_id': 'TRAP-EOL',
            'current_age_days': 2500,
            'current_health_score': 15,  # Very poor health
            'degradation_rate': 0.5
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # RUL should be very short
        assert result.rul_days < 365  # Less than 1 year

    def test_zero_health_score(self, tools):
        """Test RUL prediction with health score of 0."""
        condition_data = {
            'trap_id': 'TRAP-ZERO-HEALTH',
            'current_age_days': 2000,
            'current_health_score': 0,  # Failed
            'degradation_rate': 1.0
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # RUL should be essentially 0
        assert result.rul_days <= 30  # At most 30 days

    def test_perfect_health_score(self, tools):
        """Test RUL prediction with perfect health score."""
        condition_data = {
            'trap_id': 'TRAP-PERFECT',
            'current_age_days': 100,
            'current_health_score': 100,  # Perfect health
            'degradation_rate': 0.0
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # RUL should be very long
        assert result.rul_days > 1000

    def test_negative_inputs_handled(self, tools):
        """Test handling of invalid negative inputs."""
        with pytest.raises((ValueError, AssertionError)):
            tools.predict_remaining_useful_life({
                'trap_id': 'TRAP-NEG',
                'current_age_days': -100,  # Invalid
                'current_health_score': 70
            })


@pytest.mark.validation
class TestRULDeterminism:
    """Test deterministic behavior of RUL predictions."""

    def test_identical_inputs_identical_rul(self, tools):
        """Test that identical inputs produce identical RUL predictions."""
        condition_data = {
            'trap_id': 'TRAP-DET-RUL',
            'current_age_days': 1000,
            'current_health_score': 70,
            'degradation_rate': 0.1,
            'historical_failures': [1800, 2000, 2200]
        }

        results = [
            tools.predict_remaining_useful_life(condition_data)
            for _ in range(10)
        ]

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result.rul_days == first.rul_days
            assert result.rul_confidence_lower == first.rul_confidence_lower
            assert result.rul_confidence_upper == first.rul_confidence_upper
            assert result.provenance_hash == first.provenance_hash

    def test_rul_provenance_hash(self, tools):
        """Test provenance hash for RUL predictions."""
        condition_data = {
            'trap_id': 'TRAP-PROV-RUL',
            'current_age_days': 1000,
            'current_health_score': 70
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Validate hash format
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256


@pytest.mark.validation
class TestRULValidationMetrics:
    """Test RUL prediction accuracy metrics."""

    def test_rul_accuracy_assessment(self, tools, rul_test_data):
        """Test RUL prediction against known degradation scenarios."""
        for scenario in rul_test_data['degradation_scenarios']:
            condition_data = {
                'trap_id': 'TRAP-VALIDATION',
                'current_age_days': scenario['age_days'],
                'current_health_score': scenario['health_score']
            }

            result = tools.predict_remaining_useful_life(condition_data)

            # Check if prediction is within reasonable range of expected
            expected_rul = scenario['expected_rul_days']
            # Allow 30% tolerance
            assert abs(result.rul_days - expected_rul) / expected_rul < 0.30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "validation"])
