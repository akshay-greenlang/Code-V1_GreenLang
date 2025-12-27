"""
Tests for LIME Explainer Module

Comprehensive tests for LIMEExplainerService including:
- TabularExplainer initialization and explanations
- Local model quality metrics (R-squared)
- Confidence interval computation
- Batch explanations
- Caching and provenance tracking

Author: GreenLang AI Team
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from explainability.explanation_schemas import (
    PredictionType,
    LIMEExplanation,
    FeatureContribution,
    ConfidenceBounds,
    UncertaintyRange,
)
from explainability.lime_explainer import (
    LIMEConfig,
    LIMEExplainerService,
    TabularLIMEExplainer,
    ClassificationLIMEExplainer,
    aggregate_lime_explanations,
    compare_lime_explanations,
    compute_explanation_stability,
    LIME_AVAILABLE,
)


# Test fixtures

@pytest.fixture
def sample_feature_names():
    """Sample feature names for testing."""
    return ["temperature", "pressure", "flow_rate", "efficiency"]


@pytest.fixture
def sample_training_data():
    """Sample training data for LIME initialization."""
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture
def sample_instance():
    """Sample instance for explanation."""
    return np.array([350.0, 101325.0, 10.5, 0.85])


@pytest.fixture
def sample_config():
    """Sample LIME configuration."""
    return LIMEConfig(
        random_seed=42,
        num_samples=1000,
        num_features=4,
        mode="regression",
        cache_enabled=True
    )


@pytest.fixture
def mock_predict_fn():
    """Mock prediction function."""
    def predict(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.sum(x, axis=1) * 0.01
    return predict


class TestLIMEConfig:
    """Tests for LIMEConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LIMEConfig()

        assert config.random_seed == 42
        assert config.num_samples == 5000
        assert config.num_features == 10
        assert config.mode == "regression"
        assert config.discretize_continuous is True
        assert config.cache_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LIMEConfig(
            random_seed=123,
            num_samples=2000,
            mode="classification",
            cache_enabled=False
        )

        assert config.random_seed == 123
        assert config.num_samples == 2000
        assert config.mode == "classification"
        assert config.cache_enabled is False


class TestLIMEExplainerService:
    """Tests for LIMEExplainerService."""

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_initialization(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test service initialization."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config,
            agent_id="GL-TEST"
        )

        assert service.feature_names == sample_feature_names
        assert service.agent_id == "GL-TEST"
        assert service.config.kernel_width is not None  # Auto-computed

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_initialization_kernel_width_auto(
        self,
        sample_feature_names,
        sample_training_data
    ):
        """Test kernel width auto-computation."""
        config = LIMEConfig(kernel_width=None)

        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=config
        )

        # Should be sqrt(num_features) * 0.75
        expected = np.sqrt(4) * 0.75
        assert abs(service.config.kernel_width - expected) < 0.01

    def test_initialization_wrong_dimensions(self, sample_feature_names):
        """Test initialization fails with 1D training data."""
        if not LIME_AVAILABLE:
            pytest.skip("LIME not installed")

        training_data = np.array([1, 2, 3, 4])  # 1D

        with pytest.raises(ValueError, match="2D array"):
            LIMEExplainerService(
                training_data=training_data,
                feature_names=sample_feature_names
            )

    def test_initialization_feature_mismatch(self, sample_training_data):
        """Test initialization fails with feature count mismatch."""
        if not LIME_AVAILABLE:
            pytest.skip("LIME not installed")

        feature_names = ["a", "b"]  # Only 2 names for 4 columns

        with pytest.raises(ValueError, match="must match"):
            LIMEExplainerService(
                training_data=sample_training_data,
                feature_names=feature_names
            )

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_validate_instance_1d(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test instance validation for 1D array."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        validated = service._validate_instance(instance)

        assert validated.ndim == 1
        assert len(validated) == 4

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_validate_instance_2d(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test instance validation for 2D array."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        instance = np.array([[1.0, 2.0, 3.0, 4.0]])
        validated = service._validate_instance(instance)

        assert validated.ndim == 1
        assert len(validated) == 4

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_cache_key_computation(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data,
        sample_instance
    ):
        """Test cache key is deterministic."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        key1 = service._compute_cache_key(sample_instance, 4, 1000)
        key2 = service._compute_cache_key(sample_instance, 4, 1000)
        key3 = service._compute_cache_key(sample_instance, 5, 1000)

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_get_prediction(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data,
        sample_instance,
        mock_predict_fn
    ):
        """Test getting prediction value."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        prediction = service._get_prediction(sample_instance, mock_predict_fn)

        assert isinstance(prediction, float)

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_validate_explanation(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test explanation validation."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        # Valid explanation
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("temp", 350.0, 0.2, 100.0, "positive", 0.5),
            ],
            local_model_r2=0.85,
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        is_valid, issues = service.validate_explanation(explanation)

        assert is_valid
        assert len(issues) == 0

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_validate_explanation_low_r2(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test explanation validation fails with low R2."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[],
            local_model_r2=0.3,  # Low R2
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        is_valid, issues = service.validate_explanation(explanation)

        assert not is_valid
        assert len(issues) > 0
        assert any("fidelity" in issue.lower() for issue in issues)

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_cache_stats(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test cache statistics."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        stats = service.get_cache_stats()

        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_clear_cache(
        self,
        sample_config,
        sample_feature_names,
        sample_training_data
    ):
        """Test cache clearing."""
        service = LIMEExplainerService(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            config=sample_config
        )

        service._cache["test_key"] = Mock()
        service.clear_cache()

        assert len(service._cache) == 0


class TestTabularLIMEExplainer:
    """Tests for TabularLIMEExplainer convenience class."""

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_initialization(self, sample_feature_names, sample_training_data):
        """Test TabularLIMEExplainer initialization."""
        explainer = TabularLIMEExplainer(
            training_data=sample_training_data,
            feature_names=sample_feature_names
        )

        assert explainer.config.mode == "regression"
        assert explainer.config.discretize_continuous is True


class TestClassificationLIMEExplainer:
    """Tests for ClassificationLIMEExplainer convenience class."""

    @pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME not installed")
    def test_initialization(self, sample_feature_names, sample_training_data):
        """Test ClassificationLIMEExplainer initialization."""
        explainer = ClassificationLIMEExplainer(
            training_data=sample_training_data,
            feature_names=sample_feature_names,
            class_names=["low", "high"]
        )

        assert explainer.config.mode == "classification"
        assert explainer.class_names == ["low", "high"]


class TestLIMEUtilityFunctions:
    """Tests for LIME utility functions."""

    def test_aggregate_lime_explanations(self):
        """Test aggregation of multiple explanations."""
        explanations = [
            LIMEExplanation(
                explanation_id="exp1",
                prediction_type=PredictionType.REGRESSION,
                prediction_value=0.85,
                feature_contributions=[
                    FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
                    FeatureContribution("pressure", 101325.0, 0.1, 30.0, "positive", 0.5),
                ],
                local_model_r2=0.85,
                local_model_intercept=0.5,
                neighborhood_size=1000,
                kernel_width=1.5
            ),
            LIMEExplanation(
                explanation_id="exp2",
                prediction_type=PredictionType.REGRESSION,
                prediction_value=0.82,
                feature_contributions=[
                    FeatureContribution("temp", 360.0, 0.15, 45.0, "positive", 0.5),
                    FeatureContribution("pressure", 110000.0, 0.12, 35.0, "positive", 0.5),
                ],
                local_model_r2=0.82,
                local_model_intercept=0.5,
                neighborhood_size=1000,
                kernel_width=1.5
            ),
        ]

        aggregated = aggregate_lime_explanations(explanations)

        assert "temp" in aggregated
        assert "pressure" in aggregated
        total = sum(aggregated.values())
        assert abs(total - 1.0) < 0.01

    def test_compare_lime_explanations(self):
        """Test comparison of two explanations."""
        exp1 = LIMEExplanation(
            explanation_id="exp1",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
            ],
            local_model_r2=0.85,
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        exp2 = LIMEExplanation(
            explanation_id="exp2",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.90,
            feature_contributions=[
                FeatureContribution("temp", 360.0, 0.25, 55.0, "positive", 0.5),
            ],
            local_model_r2=0.88,
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        comparison = compare_lime_explanations(exp1, exp2)

        assert "prediction_change" in comparison
        assert comparison["prediction_change"] == 0.05
        assert "r2_change" in comparison
        assert "feature_differences" in comparison


class TestLIMEExplanationSchema:
    """Tests for LIMEExplanation data schema."""

    def test_explanation_creation(self):
        """Test LIMEExplanation creation."""
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
            ],
            local_model_r2=0.85,
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        assert explanation.explanation_id == "test123"
        assert explanation.local_model_r2 == 0.85
        assert explanation.provenance_hash is not None

    def test_explanation_to_dict(self):
        """Test LIMEExplanation serialization."""
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[],
            local_model_r2=0.85,
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        result = explanation.to_dict()

        assert "explanation_id" in result
        assert "local_model_r2" in result
        assert "provenance_hash" in result

    def test_is_reliable_pass(self):
        """Test reliability check passes."""
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[],
            local_model_r2=0.85,  # Above threshold
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        assert explanation.is_reliable(min_r2=0.7)

    def test_is_reliable_fail(self):
        """Test reliability check fails."""
        explanation = LIMEExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            prediction_value=0.85,
            feature_contributions=[],
            local_model_r2=0.5,  # Below threshold
            local_model_intercept=0.5,
            neighborhood_size=1000,
            kernel_width=1.5
        )

        assert not explanation.is_reliable(min_r2=0.7)


class TestConfidenceBounds:
    """Tests for ConfidenceBounds schema."""

    def test_confidence_bounds_creation(self):
        """Test ConfidenceBounds creation."""
        bounds = ConfidenceBounds(
            lower_bound=0.1,
            upper_bound=0.3,
            confidence_level=0.95,
            method="bootstrap"
        )

        assert bounds.lower_bound == 0.1
        assert bounds.upper_bound == 0.3
        assert bounds.confidence_level == 0.95

    def test_contains(self):
        """Test value containment check."""
        bounds = ConfidenceBounds(0.1, 0.3)

        assert bounds.contains(0.2)
        assert bounds.contains(0.1)
        assert bounds.contains(0.3)
        assert not bounds.contains(0.05)
        assert not bounds.contains(0.35)

    def test_width(self):
        """Test interval width calculation."""
        bounds = ConfidenceBounds(0.1, 0.3)

        assert bounds.width() == 0.2


class TestUncertaintyRange:
    """Tests for UncertaintyRange schema."""

    def test_uncertainty_range_creation(self):
        """Test UncertaintyRange creation."""
        uncertainty = UncertaintyRange(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            percentile_5=0.3,
            percentile_95=0.7,
            num_samples=100
        )

        assert uncertainty.mean == 0.5
        assert uncertainty.std == 0.1
        assert uncertainty.num_samples == 100

    def test_coefficient_of_variation(self):
        """Test CV calculation."""
        uncertainty = UncertaintyRange(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            percentile_5=0.3,
            percentile_95=0.7,
            num_samples=100
        )

        assert uncertainty.coefficient_of_variation == 0.2

    def test_coefficient_of_variation_zero_mean(self):
        """Test CV with zero mean."""
        uncertainty = UncertaintyRange(
            mean=0.0,
            std=0.1,
            min_value=-0.2,
            max_value=0.2,
            percentile_5=-0.1,
            percentile_95=0.1,
            num_samples=100
        )

        assert uncertainty.coefficient_of_variation == float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
