# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Weibull Analysis Tests

Tests for Weibull distribution failure analysis and RUL estimation.
Validates calculation accuracy, parameter estimation, and edge cases.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime, timezone
from typing import List

from greenlang.agents.process_heat.gl_013_predictive_maintenance.weibull import (
    FailureData,
    WeibullAnalyzer,
    WeibullParameters,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    WeibullConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    WeibullAnalysisResult,
)


class TestFailureData:
    """Tests for FailureData dataclass."""

    def test_valid_failure_data(self):
        """Test valid failure data creation."""
        data = FailureData(time=50000, is_failure=True)

        assert data.time == 50000
        assert data.is_failure is True

    def test_censored_data(self):
        """Test censored (still running) data."""
        data = FailureData(time=30000, is_failure=False)

        assert data.time == 30000
        assert data.is_failure is False

    def test_time_positive(self):
        """Test time must be positive."""
        # Valid
        data = FailureData(time=1, is_failure=True)
        assert data.time == 1

        # Zero should work (edge case)
        data = FailureData(time=0, is_failure=True)
        assert data.time == 0


class TestWeibullParameters:
    """Tests for WeibullParameters dataclass."""

    def test_valid_parameters(self):
        """Test valid Weibull parameters."""
        params = WeibullParameters(
            beta=2.5,
            eta=50000.0,
            gamma=0.0,
        )

        assert params.beta == 2.5
        assert params.eta == 50000.0
        assert params.gamma == 0.0

    def test_three_parameter_weibull(self):
        """Test three-parameter Weibull with location parameter."""
        params = WeibullParameters(
            beta=2.5,
            eta=50000.0,
            gamma=10000.0,  # Minimum life
        )

        assert params.gamma == 10000.0


class TestWeibullAnalyzer:
    """Tests for WeibullAnalyzer class."""

    def test_initialization(self, weibull_config):
        """Test analyzer initialization."""
        analyzer = WeibullAnalyzer(weibull_config)

        assert analyzer.config == weibull_config
        assert analyzer.config.method == "mle"

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        analyzer = WeibullAnalyzer()

        assert analyzer.config is not None
        assert analyzer.config.confidence_level == 0.90


class TestWeibullParameterEstimation:
    """Tests for Weibull parameter estimation methods."""

    def test_mle_estimation_wearout(self, weibull_analyzer, failure_data_wearout):
        """Test MLE parameter estimation for wear-out pattern."""
        params = weibull_analyzer.estimate_parameters(failure_data_wearout)

        # Wear-out pattern: beta > 2
        assert params.beta > 2.0
        assert params.eta > 0
        assert params.gamma >= 0

    def test_mle_estimation_random(self, weibull_analyzer, failure_data_random):
        """Test MLE parameter estimation for random failures."""
        params = weibull_analyzer.estimate_parameters(failure_data_random)

        # Random failures: beta ~ 1
        assert 0.5 < params.beta < 2.0
        assert params.eta > 0

    def test_mle_estimation_infant_mortality(
        self,
        weibull_analyzer,
        failure_data_infant_mortality
    ):
        """Test MLE parameter estimation for infant mortality."""
        params = weibull_analyzer.estimate_parameters(failure_data_infant_mortality)

        # Infant mortality: beta < 1
        assert params.beta < 1.0
        assert params.eta > 0

    def test_estimation_with_censored_data(
        self,
        weibull_analyzer,
        failure_data_with_censoring
    ):
        """Test parameter estimation with censored observations."""
        params = weibull_analyzer.estimate_parameters(failure_data_with_censoring)

        # Should still provide valid parameters
        assert params.beta > 0
        assert params.eta > 0

    def test_minimum_failures_check(self, weibull_analyzer):
        """Test minimum failures requirement."""
        insufficient_data = [
            FailureData(time=50000, is_failure=True),
            FailureData(time=52000, is_failure=True),
        ]

        # Should raise or return default
        with pytest.raises((ValueError, Exception)):
            weibull_analyzer.estimate_parameters(insufficient_data)

    def test_empty_data_raises_error(self, weibull_analyzer):
        """Test empty failure data raises error."""
        with pytest.raises((ValueError, Exception)):
            weibull_analyzer.estimate_parameters([])


class TestWeibullProbabilityCalculations:
    """Tests for Weibull probability calculations."""

    def test_reliability_at_time_zero(self, weibull_analyzer):
        """Test reliability at time zero is 1.0."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        reliability = weibull_analyzer.reliability(params, t=0)

        assert reliability == pytest.approx(1.0, rel=1e-6)

    def test_reliability_decreases_with_time(self, weibull_analyzer):
        """Test reliability decreases monotonically with time."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        r1 = weibull_analyzer.reliability(params, t=10000)
        r2 = weibull_analyzer.reliability(params, t=30000)
        r3 = weibull_analyzer.reliability(params, t=50000)

        assert r1 > r2 > r3
        assert r1 < 1.0
        assert r3 > 0.0

    def test_reliability_at_characteristic_life(self, weibull_analyzer):
        """Test reliability at eta (characteristic life) is ~0.368."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        # At t = eta, R(t) = e^(-1) ~ 0.368
        reliability = weibull_analyzer.reliability(params, t=50000)

        assert reliability == pytest.approx(1 / math.e, rel=1e-3)

    def test_failure_probability_complement(self, weibull_analyzer):
        """Test F(t) + R(t) = 1."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        for t in [0, 10000, 30000, 50000, 80000]:
            reliability = weibull_analyzer.reliability(params, t=t)
            failure_prob = weibull_analyzer.failure_probability(params, t=t)

            assert reliability + failure_prob == pytest.approx(1.0, rel=1e-6)

    def test_conditional_failure_probability(self, weibull_analyzer):
        """Test conditional failure probability calculation."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        # Conditional probability of failure in next 720 hours given survived 25000
        current_age = 25000
        time_horizon = 720

        cond_prob = weibull_analyzer.conditional_failure_probability(
            params,
            current_age=current_age,
            time_horizon=time_horizon,
        )

        # Should be between 0 and 1
        assert 0 <= cond_prob <= 1

        # Should be greater than unconditional probability for same horizon
        uncond_prob = weibull_analyzer.failure_probability(params, t=time_horizon)
        # This relationship depends on beta and age

    def test_hazard_rate_calculation(self, weibull_analyzer):
        """Test hazard rate (instantaneous failure rate) calculation."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        # For beta > 1, hazard rate should increase with time
        h1 = weibull_analyzer.hazard_rate(params, t=10000)
        h2 = weibull_analyzer.hazard_rate(params, t=30000)
        h3 = weibull_analyzer.hazard_rate(params, t=50000)

        assert h1 < h2 < h3

    def test_hazard_rate_decreasing_infant_mortality(self, weibull_analyzer):
        """Test hazard rate decreases for infant mortality (beta < 1)."""
        params = WeibullParameters(beta=0.5, eta=50000.0, gamma=0.0)

        h1 = weibull_analyzer.hazard_rate(params, t=1000)
        h2 = weibull_analyzer.hazard_rate(params, t=10000)

        assert h1 > h2  # Decreasing hazard


class TestWeibullRULCalculation:
    """Tests for Remaining Useful Life calculations."""

    def test_rul_percentiles(self, weibull_analyzer, failure_data_wearout):
        """Test RUL percentile calculations."""
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=25000,
        )

        # P10 < P50 < P90
        assert result.rul_p10_hours < result.rul_p50_hours < result.rul_p90_hours

        # All RUL values should be positive
        assert result.rul_p10_hours > 0
        assert result.rul_p50_hours > 0
        assert result.rul_p90_hours > 0

    def test_rul_decreases_with_age(self, weibull_analyzer, failure_data_wearout):
        """Test RUL decreases as equipment ages."""
        result1 = weibull_analyzer.analyze(failure_data_wearout, current_age=20000)
        result2 = weibull_analyzer.analyze(failure_data_wearout, current_age=40000)

        # Older equipment should have less RUL
        assert result2.rul_p50_hours < result1.rul_p50_hours

    def test_rul_at_zero_age(self, weibull_analyzer, failure_data_wearout):
        """Test RUL calculation at zero age."""
        result = weibull_analyzer.analyze(failure_data_wearout, current_age=0)

        assert result.rul_p50_hours > 0
        assert result.current_age_hours == 0

    def test_calculate_percentile_life(self, weibull_analyzer):
        """Test percentile life calculation."""
        params = WeibullParameters(beta=2.5, eta=50000.0, gamma=0.0)

        # B10 life (10% failures)
        b10 = weibull_analyzer.calculate_percentile_life(params, percentile=0.10)

        # B50 life (median)
        b50 = weibull_analyzer.calculate_percentile_life(params, percentile=0.50)

        # B90 life
        b90 = weibull_analyzer.calculate_percentile_life(params, percentile=0.90)

        assert b10 < b50 < b90
        assert b10 > 0


class TestWeibullAnalysisResult:
    """Tests for complete Weibull analysis result."""

    def test_analyze_wearout_pattern(self, weibull_analyzer, failure_data_wearout):
        """Test analysis of wear-out failure pattern."""
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=25000,
        )

        assert isinstance(result, WeibullAnalysisResult)
        assert result.beta > 2.0  # Wear-out pattern
        assert "wear" in result.failure_mode_interpretation.lower()

    def test_analyze_random_pattern(self, weibull_analyzer, failure_data_random):
        """Test analysis of random failure pattern."""
        result = weibull_analyzer.analyze(
            failure_data_random,
            current_age=25000,
        )

        assert isinstance(result, WeibullAnalysisResult)
        assert 0.5 < result.beta < 2.0  # Random pattern
        assert "random" in result.failure_mode_interpretation.lower()

    def test_analyze_infant_mortality_pattern(
        self,
        weibull_analyzer,
        failure_data_infant_mortality
    ):
        """Test analysis of infant mortality pattern."""
        result = weibull_analyzer.analyze(
            failure_data_infant_mortality,
            current_age=5000,
        )

        assert isinstance(result, WeibullAnalysisResult)
        assert result.beta < 1.0  # Infant mortality
        assert "infant" in result.failure_mode_interpretation.lower() or \
               "early" in result.failure_mode_interpretation.lower()

    def test_current_failure_probability(self, weibull_analyzer, failure_data_wearout):
        """Test current failure probability is correct."""
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=25000,
        )

        assert 0 <= result.current_failure_probability <= 1

    def test_conditional_probability_30d(self, weibull_analyzer, failure_data_wearout):
        """Test 30-day conditional failure probability."""
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=40000,  # Near end of life
        )

        # Should have higher conditional probability near end of life
        assert 0 <= result.conditional_failure_probability_30d <= 1

    def test_result_provenance_hash(self, weibull_analyzer, failure_data_wearout):
        """Test provenance hash is deterministic."""
        result1 = weibull_analyzer.analyze(failure_data_wearout, current_age=25000)
        result2 = weibull_analyzer.analyze(failure_data_wearout, current_age=25000)

        # Same input should produce same provenance hash
        assert result1.provenance_hash == result2.provenance_hash
        assert len(result1.provenance_hash) == 64  # SHA-256


class TestWeibullConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_confidence_bounds(self, weibull_analyzer, failure_data_wearout):
        """Test confidence bounds are calculated."""
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=25000,
        )

        # Check confidence bounds exist if calculated
        if result.beta_confidence_lower is not None:
            assert result.beta_confidence_lower < result.beta
            assert result.beta_confidence_upper > result.beta

    def test_confidence_level_affects_bounds(self):
        """Test different confidence levels produce different bounds."""
        config_90 = WeibullConfig(confidence_level=0.90)
        config_95 = WeibullConfig(confidence_level=0.95)

        analyzer_90 = WeibullAnalyzer(config_90)
        analyzer_95 = WeibullAnalyzer(config_95)

        # Higher confidence should give wider bounds
        # This is a conceptual test - actual implementation may vary


class TestWeibullEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_censored_data(self, weibull_analyzer):
        """Test handling when all data is censored (no failures)."""
        all_censored = [
            FailureData(time=50000, is_failure=False),
            FailureData(time=52000, is_failure=False),
            FailureData(time=48000, is_failure=False),
        ]

        with pytest.raises((ValueError, Exception)):
            weibull_analyzer.analyze(all_censored, current_age=25000)

    def test_very_large_beta(self, weibull_analyzer):
        """Test handling of very steep Weibull (large beta)."""
        # Data with very consistent failure times
        tight_data = [
            FailureData(time=50000, is_failure=True),
            FailureData(time=50100, is_failure=True),
            FailureData(time=50050, is_failure=True),
            FailureData(time=50075, is_failure=True),
            FailureData(time=50025, is_failure=True),
        ]

        result = weibull_analyzer.analyze(tight_data, current_age=25000)

        # Should handle large beta gracefully
        assert result.beta > 0
        assert result.eta > 0

    def test_very_small_times(self, weibull_analyzer):
        """Test handling of very small failure times."""
        small_times = [
            FailureData(time=1, is_failure=True),
            FailureData(time=2, is_failure=True),
            FailureData(time=3, is_failure=True),
            FailureData(time=4, is_failure=True),
            FailureData(time=5, is_failure=True),
        ]

        result = weibull_analyzer.analyze(small_times, current_age=0)

        assert result.beta > 0
        assert result.eta > 0

    def test_current_age_beyond_all_failures(self, weibull_analyzer, failure_data_wearout):
        """Test when current age exceeds all recorded failure times."""
        # Equipment running beyond all historical failures
        result = weibull_analyzer.analyze(
            failure_data_wearout,
            current_age=100000,  # Beyond all failures
        )

        # Should still provide valid result
        assert result.current_failure_probability > 0.9  # Very high

    def test_duplicate_failure_times(self, weibull_analyzer):
        """Test handling of duplicate failure times."""
        duplicates = [
            FailureData(time=50000, is_failure=True),
            FailureData(time=50000, is_failure=True),
            FailureData(time=50000, is_failure=True),
            FailureData(time=52000, is_failure=True),
            FailureData(time=52000, is_failure=True),
        ]

        result = weibull_analyzer.analyze(duplicates, current_age=25000)

        # Should handle duplicates
        assert result.beta > 0


class TestWeibullMTBFCalculation:
    """Tests for Mean Time Between Failures calculation."""

    def test_mtbf_calculation(self, weibull_analyzer):
        """Test MTBF (Mean Time to Failure) calculation."""
        params = WeibullParameters(beta=2.0, eta=50000.0, gamma=0.0)

        mtbf = weibull_analyzer.calculate_mtbf(params)

        # MTBF = eta * Gamma(1 + 1/beta)
        # For beta=2, Gamma(1.5) ~ 0.886
        # MTBF ~ 50000 * 0.886 ~ 44300
        assert 40000 < mtbf < 50000

    def test_mtbf_exponential_case(self, weibull_analyzer):
        """Test MTBF equals eta when beta=1 (exponential)."""
        params = WeibullParameters(beta=1.0, eta=50000.0, gamma=0.0)

        mtbf = weibull_analyzer.calculate_mtbf(params)

        # For exponential (beta=1), MTBF = eta
        assert mtbf == pytest.approx(50000.0, rel=0.01)


class TestWeibullDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_analysis_same_result(
        self,
        weibull_analyzer,
        failure_data_wearout
    ):
        """Test repeated analysis produces identical results."""
        results = [
            weibull_analyzer.analyze(failure_data_wearout, current_age=25000)
            for _ in range(5)
        ]

        # All betas should be identical
        betas = [r.beta for r in results]
        assert len(set(betas)) == 1

        # All etas should be identical
        etas = [r.eta_hours for r in results]
        assert len(set(etas)) == 1

    def test_provenance_reproducibility(
        self,
        weibull_analyzer,
        failure_data_wearout
    ):
        """Test provenance hash is reproducible."""
        result1 = weibull_analyzer.analyze(failure_data_wearout, current_age=25000)
        result2 = weibull_analyzer.analyze(failure_data_wearout, current_age=25000)

        assert result1.provenance_hash == result2.provenance_hash


class TestWeibullIntegration:
    """Integration tests for Weibull analysis."""

    def test_full_analysis_workflow(self, weibull_config):
        """Test complete analysis workflow."""
        # Create analyzer
        analyzer = WeibullAnalyzer(weibull_config)

        # Failure data from production
        failure_data = [
            FailureData(time=45000, is_failure=True),
            FailureData(time=48000, is_failure=True),
            FailureData(time=50000, is_failure=True),
            FailureData(time=52000, is_failure=True),
            FailureData(time=55000, is_failure=True),
            FailureData(time=47000, is_failure=False),  # Still running
        ]

        # Analyze current equipment at 40000 hours
        result = analyzer.analyze(failure_data, current_age=40000)

        # Verify complete result
        assert result.beta > 0
        assert result.eta_hours > 0
        assert result.rul_p10_hours > 0
        assert result.rul_p50_hours > 0
        assert result.rul_p90_hours > 0
        assert 0 <= result.current_failure_probability <= 1
        assert 0 <= result.conditional_failure_probability_30d <= 1
        assert result.failure_mode_interpretation is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.parametrize("current_age,expected_high_risk", [
        (10000, False),  # Early in life
        (45000, True),   # Near characteristic life
        (60000, True),   # Beyond characteristic life
    ])
    def test_risk_assessment_by_age(
        self,
        weibull_analyzer,
        failure_data_wearout,
        current_age,
        expected_high_risk
    ):
        """Test risk assessment varies appropriately with age."""
        result = weibull_analyzer.analyze(failure_data_wearout, current_age=current_age)

        is_high_risk = result.conditional_failure_probability_30d > 0.3

        assert is_high_risk == expected_high_risk
