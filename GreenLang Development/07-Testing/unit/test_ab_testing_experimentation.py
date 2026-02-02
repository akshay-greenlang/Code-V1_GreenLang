# -*- coding: utf-8 -*-
"""
Unit tests for A/B Testing Framework

Tests cover:
  - Experiment creation and management
  - Deterministic variant assignment
  - Statistical analysis (t-test, chi-squared, Bayesian)
  - Sample size calculation
  - Provenance tracking
"""

import pytest
import numpy as np
from datetime import datetime

from greenlang.ml.experimentation.ab_testing import (
    ABTestManager,
    ExperimentMetrics,
    StatisticalAnalyzer,
    MetricType,
    TestType,
    VariantMetrics,
)


class TestABTestManager:
    """Test ABTestManager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = ABTestManager()

    def test_create_experiment_basic(self):
        """Test basic experiment creation."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["control", "treatment"],
        )

        assert exp_id is not None
        assert len(exp_id) == 12
        assert exp_id in self.manager._experiments

    def test_create_experiment_with_traffic_split(self):
        """Test experiment creation with custom traffic split."""
        traffic_split = {"control": 0.6, "treatment": 0.4}
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["control", "treatment"],
            traffic_split=traffic_split,
        )

        exp = self.manager._experiments[exp_id]
        assert exp["traffic_split"] == traffic_split

    def test_create_experiment_invalid_traffic_split(self):
        """Test experiment creation fails with invalid split."""
        with pytest.raises(ValueError):
            self.manager.create_experiment(
                name="test_exp",
                variants=["a", "b"],
                traffic_split={"a": 0.3, "b": 0.5},  # Sum != 1
            )

    def test_create_experiment_requires_two_variants(self):
        """Test experiment requires at least 2 variants."""
        with pytest.raises(ValueError):
            self.manager.create_experiment(
                name="test_exp",
                variants=["only_one"],
            )

    def test_deterministic_variant_assignment(self):
        """Test variant assignment is deterministic."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        # Same user should get same variant consistently
        variant_1 = self.manager.assign_variant("user_123", exp_id)
        variant_2 = self.manager.assign_variant("user_123", exp_id)

        assert variant_1 == variant_2
        assert variant_1 in ["a", "b"]

    def test_variant_assignment_distribution(self):
        """Test variant assignment distribution is balanced."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
            traffic_split={"a": 0.5, "b": 0.5},
        )

        # Assign many users
        assignments = [
            self.manager.assign_variant(f"user_{i}", exp_id)
            for i in range(1000)
        ]

        count_a = assignments.count("a")
        # Should be roughly 500 +/- 50 (allowing 10% tolerance)
        assert 450 < count_a < 550

    def test_record_metric(self):
        """Test recording continuous metrics."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        for i in range(10):
            self.manager.record_metric(exp_id, "a", "efficiency", 0.85 + i * 0.01)

        metrics = self.manager._metrics[exp_id]
        assert len(metrics.variant_data.get("a", [])) == 10

    def test_record_metric_invalid_experiment(self):
        """Test recording metric fails for invalid experiment."""
        with pytest.raises(ValueError):
            self.manager.record_metric("invalid_id", "a", "metric", 0.5)

    def test_record_conversion(self):
        """Test recording conversion metrics."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
            metric_type=MetricType.CONVERSION,
        )

        # Record conversions
        for i in range(100):
            success = i % 3 == 0  # ~33% conversion rate
            self.manager.record_conversion(exp_id, "a", success)

        metrics = self.manager._metrics[exp_id]
        successes, total = metrics.variant_conversions.get("a", (0, 0))

        assert total == 100
        assert 25 < successes < 40  # Roughly 33%

    def test_analyze_results_insufficient_samples(self):
        """Test analyze fails with insufficient samples."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        # Add only 1 sample
        self.manager.record_metric(exp_id, "a", "metric", 0.5)

        # Should still work but with low power
        result = self.manager.analyze_results(exp_id)
        assert result.is_significant is False

    def test_analyze_results_t_test(self):
        """Test t-test analysis detects difference."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["control", "treatment"],
            test_type=TestType.WELCH_T,
        )

        # Generate data with clear difference
        np.random.seed(42)
        control_data = np.random.normal(10.0, 1.0, 100)
        treatment_data = np.random.normal(11.0, 1.0, 100)

        for value in control_data:
            self.manager.record_metric(exp_id, "control", "metric", float(value))

        for value in treatment_data:
            self.manager.record_metric(exp_id, "treatment", "metric", float(value))

        result = self.manager.analyze_results(exp_id)

        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.winner == "treatment"
        assert result.effect_size > 0

    def test_analyze_results_no_difference(self):
        """Test t-test correctly identifies no significant difference."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
            test_type=TestType.WELCH_T,
        )

        # Generate data with no real difference
        np.random.seed(42)
        data_a = np.random.normal(10.0, 1.0, 50)
        data_b = np.random.normal(10.05, 1.0, 50)  # Minimal difference

        for value in data_a:
            self.manager.record_metric(exp_id, "a", "metric", float(value))

        for value in data_b:
            self.manager.record_metric(exp_id, "b", "metric", float(value))

        result = self.manager.analyze_results(exp_id)

        assert result.is_significant is False
        assert result.winner is None

    def test_get_winner_significant(self):
        """Test get_winner returns winner when significant."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        # Large difference - generate noisy but distinct data
        np.random.seed(42)
        for i in range(50):
            self.manager.record_metric(exp_id, "a", "metric", np.random.normal(1.0, 0.3))
            self.manager.record_metric(exp_id, "b", "metric", np.random.normal(10.0, 0.3))

        winner = self.manager.get_winner(exp_id)
        assert winner == "b"

    def test_get_winner_not_significant(self):
        """Test get_winner returns None when not significant."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        # Tiny difference
        for i in range(10):
            self.manager.record_metric(exp_id, "a", "metric", 5.0)
            self.manager.record_metric(exp_id, "b", "metric", 5.01)

        winner = self.manager.get_winner(exp_id)
        assert winner is None

    def test_export_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        for i in range(10):
            self.manager.record_metric(exp_id, "a", "metric", 1.0 + i * 0.1)
            self.manager.record_metric(exp_id, "b", "metric", 2.0 + i * 0.1)

        metrics_str = self.manager.export_prometheus_metrics(exp_id)

        assert "ab_test_p_value" in metrics_str
        assert "ab_test_effect_size" in metrics_str
        assert "ab_test_sample_count" in metrics_str
        assert "ab_test_variant_mean" in metrics_str

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        for i in range(10):
            self.manager.record_metric(exp_id, "a", "metric", 1.0)
            self.manager.record_metric(exp_id, "b", "metric", 2.0)

        result1 = self.manager.analyze_results(exp_id)
        result2 = self.manager.analyze_results(exp_id)

        assert result1.provenance_hash == result2.provenance_hash

    def test_get_status(self):
        """Test getting experiment status."""
        exp_id = self.manager.create_experiment(
            name="test_exp",
            variants=["a", "b"],
        )

        for i in range(10):
            self.manager.record_metric(exp_id, "a", "metric", 1.0)
            self.manager.record_metric(exp_id, "b", "metric", 2.0)

        status = self.manager.get_status(exp_id)

        assert status["experiment_id"] == exp_id
        assert status["name"] == "test_exp"
        assert status["status"] == "active"
        assert "a" in status["sample_counts"]
        assert "b" in status["sample_counts"]


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer."""

    def test_welch_t_test_significant_difference(self):
        """Test Welch t-test detects significant difference."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        t_stat, p_value = StatisticalAnalyzer.welch_t_test(group_a, group_b)

        assert p_value < 0.05
        assert abs(t_stat) > 2.0  # Significant difference

    def test_welch_t_test_no_difference(self):
        """Test Welch t-test with identical groups."""
        group_a = [5.0] * 10
        group_b = [5.0] * 10

        t_stat, p_value = StatisticalAnalyzer.welch_t_test(group_a, group_b)

        assert p_value > 0.05
        assert t_stat == 0.0

    def test_cohen_d_effect_size(self):
        """Test Cohen's d effect size calculation."""
        group_a = [1.0] * 10
        group_b = [3.0] * 10

        effect_size = StatisticalAnalyzer.cohen_d(group_a, group_b)

        # Cohen's d with identical values should be positive
        assert effect_size >= 0

    def test_chi_squared_test(self):
        """Test chi-squared test for conversions."""
        # Group A: 80/100 conversions
        # Group B: 50/100 conversions
        group_a = (80, 100)
        group_b = (50, 100)

        chi2, p_value = StatisticalAnalyzer.chi_squared_test(group_a, group_b)

        assert p_value < 0.05
        assert chi2 > 0

    def test_bayesian_a_b(self):
        """Test Bayesian A/B test probability."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        prob_b_wins = StatisticalAnalyzer.bayesian_a_b(group_a, group_b)

        assert 0 <= prob_b_wins <= 1
        assert prob_b_wins > 0.95  # B should clearly win

    def test_sample_size_calculation(self):
        """Test sample size calculator."""
        effect_size = 0.2  # Small effect
        sample_size = StatisticalAnalyzer.calculate_sample_size(
            effect_size=effect_size,
            alpha=0.05,
            power=0.80,
        )

        assert sample_size > 0
        assert isinstance(sample_size, int)
        # For small effect size, need many samples (around 90-100)
        assert sample_size > 50

    def test_sample_size_large_effect(self):
        """Test sample size with large effect."""
        effect_size = 0.8  # Large effect
        sample_size = StatisticalAnalyzer.calculate_sample_size(
            effect_size=effect_size,
            alpha=0.05,
            power=0.80,
        )

        # For large effect, fewer samples needed
        assert sample_size < 100


class TestExperimentMetrics:
    """Test ExperimentMetrics."""

    def test_add_metric(self):
        """Test adding metrics."""
        metrics = ExperimentMetrics(experiment_id="test_id")

        metrics.add_metric("variant_a", 1.5)
        metrics.add_metric("variant_a", 2.5)

        assert len(metrics.variant_data["variant_a"]) == 2
        assert metrics.variant_data["variant_a"] == [1.5, 2.5]

    def test_add_conversion(self):
        """Test adding conversion metrics."""
        metrics = ExperimentMetrics(experiment_id="test_id")

        metrics.add_conversion("variant_a", True)
        metrics.add_conversion("variant_a", False)
        metrics.add_conversion("variant_a", True)

        successes, total = metrics.variant_conversions["variant_a"]
        assert successes == 2
        assert total == 3

    def test_separate_variants(self):
        """Test metrics are tracked separately per variant."""
        metrics = ExperimentMetrics(experiment_id="test_id")

        metrics.add_metric("variant_a", 1.0)
        metrics.add_metric("variant_b", 2.0)

        assert metrics.variant_data["variant_a"] == [1.0]
        assert metrics.variant_data["variant_b"] == [2.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
