# -*- coding: utf-8 -*-
"""
Tests for LIME Explainer module.

Validates:
- LimeTabularExplainer functionality
- Local surrogate model generation
- Feature contribution extraction
- Confidence intervals
- Determinism and reproducibility
"""

import pytest
import numpy as np

from explainability.lime_explainer import (
    LIMEExplainer,
    TabularLIMEExplainer,
    LIMEConfig,
    aggregate_lime_explanations,
    compare_lime_explanations,
    validate_lime_explanation,
)
from explainability.explanation_schemas import (
    PredictionType,
    LIMEExplanation,
    ConfidenceBounds,
)


class TestLIMEConfig:
    """Tests for LIME configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LIMEConfig()
        assert config.random_seed == 42
        assert config.num_samples == 5000
        assert config.num_features == 10
        assert config.mode == "regression"
        assert config.discretize_continuous is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LIMEConfig(
            random_seed=123,
            num_samples=1000,
            num_features=5,
            mode="classification"
        )
        assert config.random_seed == 123
        assert config.num_samples == 1000
        assert config.num_features == 5
        assert config.mode == "classification"


@pytest.mark.lime
class TestLIMEExplainer:
    """Tests for base LIME explainer."""

    def test_initialization(self, training_data, feature_names):
        """Test explainer initialization."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        assert explainer.feature_names == feature_names
        assert explainer._lime_explainer is not None

    def test_initialization_with_config(self, training_data, feature_names):
        """Test explainer initialization with custom config."""
        config = LIMEConfig(random_seed=123, num_samples=2000)
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )

        assert explainer.config.random_seed == 123
        assert explainer.config.num_samples == 2000

    def test_explain_instance(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test explaining single instance."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        assert isinstance(explanation, LIMEExplanation)
        assert explanation.prediction_type == PredictionType.DEMAND_FORECAST
        assert len(explanation.feature_contributions) > 0
        assert explanation.local_model_r2 >= 0
        assert explanation.neighborhood_size > 0

    def test_explain_batch(
        self,
        training_data,
        sample_batch,
        feature_names,
        mock_prediction_function
    ):
        """Test explaining batch of instances."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explanations = explainer.explain_batch(
            sample_batch,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        assert len(explanations) == len(sample_batch)
        for exp in explanations:
            assert isinstance(exp, LIMEExplanation)

    def test_compute_confidence_intervals(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test computing confidence intervals."""
        config = LIMEConfig(num_samples=1000)  # Smaller for speed
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )

        intervals = explainer.compute_confidence_intervals(
            sample_instance,
            mock_prediction_function,
            num_bootstrap=20,  # Smaller for speed
            confidence_level=0.95
        )

        assert isinstance(intervals, dict)
        for name, bounds in intervals.items():
            assert isinstance(bounds, ConfidenceBounds)
            assert bounds.lower_bound <= bounds.upper_bound
            assert bounds.confidence_level == 0.95

    def test_get_local_model(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test getting local linear model coefficients."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        coefficients, intercept, r2 = explainer.get_local_model(
            sample_instance,
            mock_prediction_function
        )

        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0
        assert isinstance(intercept, float)
        assert 0 <= r2 <= 1

    def test_generate_surrogate_model_data(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test generating surrogate model data."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        surrogate_data = explainer.generate_surrogate_model_data(
            sample_instance,
            mock_prediction_function
        )

        assert "coefficients" in surrogate_data
        assert "intercept" in surrogate_data
        assert "r_squared" in surrogate_data
        assert "original_prediction" in surrogate_data
        assert "surrogate_prediction" in surrogate_data
        assert "prediction_error" in surrogate_data

    def test_caching(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that caching works correctly."""
        config = LIMEConfig(cache_enabled=True)
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )

        # First call
        exp1 = explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        # Second call (should be cached)
        exp2 = explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        assert exp1.explanation_id == exp2.explanation_id

    def test_clear_cache(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test clearing explanation cache."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        assert len(explainer._explanation_cache) > 0

        explainer.clear_cache()

        assert len(explainer._explanation_cache) == 0


@pytest.mark.determinism
class TestLIMEDeterminism:
    """Tests for LIME determinism and reproducibility."""

    def test_reproducible_explanations(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that explanations are reproducible with same seed."""
        config = LIMEConfig(random_seed=42, cache_enabled=False)

        # First run
        explainer1 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )
        exp1 = explainer1.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        # Second run with same seed
        explainer2 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )
        exp2 = explainer2.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        # Should have identical results
        assert exp1.prediction_value == exp2.prediction_value
        assert exp1.local_model_r2 == exp2.local_model_r2
        assert len(exp1.feature_contributions) == len(exp2.feature_contributions)

        for c1, c2 in zip(exp1.feature_contributions, exp2.feature_contributions):
            assert c1.feature_name == c2.feature_name
            assert abs(c1.contribution - c2.contribution) < 1e-6

    def test_different_seeds_different_results(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that different seeds can produce different results."""
        config1 = LIMEConfig(random_seed=42, cache_enabled=False)
        config2 = LIMEConfig(random_seed=123, cache_enabled=False)

        explainer1 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config1
        )

        explainer2 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config2
        )

        exp1 = explainer1.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        exp2 = explainer2.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        # Results may differ slightly due to different sampling
        # Both should have valid explanations
        assert exp1.local_model_r2 > 0
        assert exp2.local_model_r2 > 0


@pytest.mark.lime
class TestTabularLIMEExplainer:
    """Tests for specialized tabular LIME explainer."""

    def test_initialization(self, training_data, feature_names):
        """Test TabularLIMEExplainer initialization."""
        explainer = TabularLIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        assert explainer._lime_explainer is not None

    def test_explain(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test TabularLIMEExplainer explanation."""
        explainer = TabularLIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.HEALTH_SCORE
        )

        assert isinstance(explanation, LIMEExplanation)


@pytest.mark.lime
class TestLIMEUtilities:
    """Tests for LIME utility functions."""

    def test_aggregate_lime_explanations(
        self,
        training_data,
        sample_batch,
        feature_names,
        mock_prediction_function
    ):
        """Test aggregating multiple LIME explanations."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explanations = explainer.explain_batch(
            sample_batch[:5],  # Use smaller batch
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        aggregated = aggregate_lime_explanations(explanations)

        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        # Should sum to 1 (normalized)
        assert abs(sum(aggregated.values()) - 1.0) < 0.01

    def test_compare_lime_explanations(
        self,
        training_data,
        sample_batch,
        feature_names,
        mock_prediction_function
    ):
        """Test comparing two LIME explanations."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        exp1 = explainer.explain_instance(
            sample_batch[0],
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        exp2 = explainer.explain_instance(
            sample_batch[1],
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        comparison = compare_lime_explanations(exp1, exp2)

        assert "prediction_change" in comparison
        assert "r2_change" in comparison
        assert "feature_differences" in comparison

    def test_validate_lime_explanation_pass(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test LIME explanation validation passes."""
        explainer = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = explainer.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        is_valid, issues = validate_lime_explanation(explanation)

        # Should pass for well-trained model
        if explanation.local_model_r2 >= 0.7:
            assert is_valid is True
            assert len(issues) == 0

    def test_validate_lime_explanation_low_r2(self):
        """Test LIME explanation validation with low R2."""
        from explainability.explanation_schemas import LIMEExplanation, FeatureContribution
        from datetime import datetime

        # Create explanation with low R2
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.DEMAND_FORECAST,
            prediction_value=100.0,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temp",
                    feature_value=250.0,
                    contribution=10.0,
                    contribution_percentage=100.0,
                    direction="positive"
                )
            ],
            local_model_r2=0.5,  # Low R2
            local_model_intercept=50.0,
            neighborhood_size=1000,
            kernel_width=0.75,
            timestamp=datetime.utcnow(),
            computation_time_ms=100.0,
            random_seed=42
        )

        is_valid, issues = validate_lime_explanation(explanation, min_r2=0.7)

        assert is_valid is False
        assert len(issues) > 0
        assert any("fidelity" in issue.lower() for issue in issues)


@pytest.mark.lime
class TestLIMEExplanationSchema:
    """Tests for LIME explanation schema."""

    def test_lime_explanation_creation(self):
        """Test creating LIMEExplanation."""
        from datetime import datetime
        from explainability.explanation_schemas import FeatureContribution

        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.DEMAND_FORECAST,
            prediction_value=150.0,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temperature",
                    feature_value=300.0,
                    contribution=25.0,
                    contribution_percentage=50.0,
                    direction="positive"
                ),
                FeatureContribution(
                    feature_name="pressure",
                    feature_value=15.0,
                    contribution=-25.0,
                    contribution_percentage=50.0,
                    direction="negative"
                )
            ],
            local_model_r2=0.85,
            local_model_intercept=100.0,
            neighborhood_size=5000,
            kernel_width=0.75,
            timestamp=datetime.utcnow(),
            computation_time_ms=500.0,
            random_seed=42
        )

        assert explanation.prediction_value == 150.0
        assert explanation.local_model_r2 == 0.85
        assert len(explanation.feature_contributions) == 2
        assert explanation.is_reliable is True

    def test_lime_explanation_is_reliable_property(self):
        """Test is_reliable property."""
        from datetime import datetime
        from explainability.explanation_schemas import FeatureContribution

        # High R2 - reliable
        high_r2 = LIMEExplanation(
            explanation_id="test1",
            prediction_type=PredictionType.DEMAND_FORECAST,
            prediction_value=100.0,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temp",
                    feature_value=250.0,
                    contribution=10.0,
                    contribution_percentage=100.0,
                    direction="positive"
                )
            ],
            local_model_r2=0.9,
            local_model_intercept=90.0,
            neighborhood_size=5000,
            kernel_width=0.75,
            timestamp=datetime.utcnow(),
            computation_time_ms=100.0,
            random_seed=42
        )
        assert high_r2.is_reliable is True

        # Low R2 - not reliable
        low_r2 = LIMEExplanation(
            explanation_id="test2",
            prediction_type=PredictionType.DEMAND_FORECAST,
            prediction_value=100.0,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temp",
                    feature_value=250.0,
                    contribution=10.0,
                    contribution_percentage=100.0,
                    direction="positive"
                )
            ],
            local_model_r2=0.5,
            local_model_intercept=90.0,
            neighborhood_size=5000,
            kernel_width=0.75,
            timestamp=datetime.utcnow(),
            computation_time_ms=100.0,
            random_seed=42
        )
        assert low_r2.is_reliable is False
