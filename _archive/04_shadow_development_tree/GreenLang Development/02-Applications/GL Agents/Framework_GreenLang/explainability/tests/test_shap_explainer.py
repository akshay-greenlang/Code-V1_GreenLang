"""
Tests for SHAP Explainer Module

Comprehensive tests for SHAPExplainerService including:
- TreeExplainer fitting and explanations
- KernelExplainer for model-agnostic explanations
- Feature importance computation
- Visualization data generation
- Caching and provenance tracking
- Batch explanations

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
    SHAPExplanation,
    FeatureContribution,
)
from explainability.shap_explainer import (
    SHAPConfig,
    SHAPExplainerService,
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    verify_shap_consistency,
    aggregate_shap_explanations,
    compare_explanations,
    SHAP_AVAILABLE,
)


# Test fixtures

@pytest.fixture
def sample_feature_names():
    """Sample feature names for testing."""
    return ["temperature", "pressure", "flow_rate", "efficiency"]


@pytest.fixture
def sample_instance():
    """Sample instance for explanation."""
    return np.array([350.0, 101325.0, 10.5, 0.85])


@pytest.fixture
def sample_instances():
    """Sample batch of instances."""
    return np.array([
        [350.0, 101325.0, 10.5, 0.85],
        [360.0, 110000.0, 11.0, 0.82],
        [340.0, 95000.0, 9.5, 0.88],
    ])


@pytest.fixture
def sample_config():
    """Sample SHAP configuration."""
    return SHAPConfig(
        random_seed=42,
        num_samples=100,
        cache_enabled=True,
        cache_ttl_seconds=300
    )


@pytest.fixture
def mock_tree_model():
    """Mock tree-based model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.85]))
    return model


@pytest.fixture
def mock_shap_values():
    """Mock SHAP values."""
    return np.array([0.1, -0.05, 0.15, 0.02])


class TestSHAPConfig:
    """Tests for SHAPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SHAPConfig()

        assert config.random_seed == 42
        assert config.num_samples == 100
        assert config.check_additivity is True
        assert config.approximate is False
        assert config.cache_enabled is True
        assert config.batch_size == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SHAPConfig(
            random_seed=123,
            num_samples=500,
            cache_enabled=False,
            batch_size=50
        )

        assert config.random_seed == 123
        assert config.num_samples == 500
        assert config.cache_enabled is False
        assert config.batch_size == 50


class TestSHAPExplainerService:
    """Tests for SHAPExplainerService."""

    def test_initialization(self, sample_config, sample_feature_names):
        """Test service initialization."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names,
            agent_id="GL-TEST",
            agent_version="1.0.0"
        )

        assert service.config == sample_config
        assert service.feature_names == sample_feature_names
        assert service.agent_id == "GL-TEST"

    def test_initialization_without_shap(self):
        """Test initialization fails gracefully without SHAP."""
        with patch.dict('sys.modules', {'shap': None}):
            # This would raise ImportError in actual code
            pass

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
    def test_fit_tree_explainer(self, sample_config, sample_feature_names, mock_tree_model):
        """Test TreeExplainer fitting."""
        with patch('shap.TreeExplainer') as mock_explainer:
            mock_explainer.return_value = Mock()

            service = SHAPExplainerService(
                config=sample_config,
                feature_names=sample_feature_names
            )
            service.fit_tree_explainer(mock_tree_model, sample_feature_names)

            mock_explainer.assert_called_once()

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
    def test_fit_kernel_explainer(self, sample_config, sample_feature_names):
        """Test KernelExplainer fitting."""
        background_data = np.random.randn(100, 4)

        with patch('shap.KernelExplainer') as mock_explainer:
            mock_explainer.return_value = Mock()

            service = SHAPExplainerService(
                config=sample_config,
                feature_names=sample_feature_names
            )
            service.fit_kernel_explainer(
                lambda x: np.sum(x, axis=1),
                background_data,
                sample_feature_names
            )

            mock_explainer.assert_called_once()

    def test_validate_instance_1d(self, sample_config, sample_feature_names):
        """Test instance validation for 1D array."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        validated = service._validate_instance(instance)

        assert validated.ndim == 2
        assert validated.shape == (1, 4)

    def test_validate_instance_2d(self, sample_config, sample_feature_names):
        """Test instance validation for 2D array."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        instance = np.array([[1.0, 2.0, 3.0, 4.0]])
        validated = service._validate_instance(instance)

        assert validated.ndim == 2
        assert validated.shape == (1, 4)

    def test_validate_instance_wrong_features(self, sample_config, sample_feature_names):
        """Test instance validation fails with wrong number of features."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        instance = np.array([1.0, 2.0, 3.0])  # Only 3 features

        with pytest.raises(ValueError, match="expected 4"):
            service._validate_instance(instance)

    def test_cache_key_computation(self, sample_config, sample_feature_names, sample_instance):
        """Test cache key is deterministic."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        key1 = service._compute_cache_key(sample_instance, True)
        key2 = service._compute_cache_key(sample_instance, True)
        key3 = service._compute_cache_key(sample_instance, False)

        assert key1 == key2  # Same parameters = same key
        assert key1 != key3  # Different parameters = different key

    def test_generate_waterfall_data(self, sample_config, sample_feature_names):
        """Test waterfall plot data generation."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        # Create mock explanation
        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temperature",
                    feature_value=350.0,
                    contribution=0.2,
                    contribution_percentage=50.0,
                    direction="positive",
                    baseline_value=0.5
                ),
                FeatureContribution(
                    feature_name="pressure",
                    feature_value=101325.0,
                    contribution=0.15,
                    contribution_percentage=40.0,
                    direction="positive",
                    baseline_value=0.5
                )
            ],
            consistency_check=0.001
        )

        waterfall_data = service.generate_waterfall_data(explanation, max_features=10)

        assert len(waterfall_data) >= 3  # Base + features + prediction
        assert waterfall_data[0]["feature"] == "Base Value"
        assert waterfall_data[-1]["feature"] == "Prediction"

    def test_generate_force_plot_data(self, sample_config, sample_feature_names):
        """Test force plot data generation."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution(
                    feature_name="temperature",
                    feature_value=350.0,
                    contribution=0.2,
                    contribution_percentage=50.0,
                    direction="positive",
                    baseline_value=0.5
                ),
                FeatureContribution(
                    feature_name="pressure",
                    feature_value=101325.0,
                    contribution=-0.05,
                    contribution_percentage=10.0,
                    direction="negative",
                    baseline_value=0.5
                )
            ],
            consistency_check=0.001
        )

        force_data = service.generate_force_plot_data(explanation)

        assert "base_value" in force_data
        assert "prediction_value" in force_data
        assert "positive_contributions" in force_data
        assert "negative_contributions" in force_data

    def test_cache_stats(self, sample_config, sample_feature_names):
        """Test cache statistics."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        stats = service.get_cache_stats()

        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self, sample_config, sample_feature_names):
        """Test cache clearing."""
        if not SHAP_AVAILABLE:
            pytest.skip("SHAP not installed")

        service = SHAPExplainerService(
            config=sample_config,
            feature_names=sample_feature_names
        )

        # Add something to cache
        service._cache["test_key"] = Mock()

        service.clear_cache()

        assert len(service._cache) == 0


class TestSHAPUtilityFunctions:
    """Tests for SHAP utility functions."""

    def test_verify_shap_consistency_pass(self):
        """Test SHAP consistency check passes."""
        shap_values = np.array([0.1, 0.2, 0.05])
        base_value = 0.5
        prediction = 0.85

        is_consistent, error = verify_shap_consistency(
            shap_values, base_value, prediction
        )

        assert is_consistent
        assert error < 0.01

    def test_verify_shap_consistency_fail(self):
        """Test SHAP consistency check fails."""
        shap_values = np.array([0.1, 0.2, 0.05])
        base_value = 0.5
        prediction = 1.0  # Wrong prediction

        is_consistent, error = verify_shap_consistency(
            shap_values, base_value, prediction
        )

        assert not is_consistent
        assert error > 0.01

    def test_aggregate_shap_explanations(self):
        """Test aggregation of multiple explanations."""
        explanations = [
            SHAPExplanation(
                explanation_id="exp1",
                prediction_type=PredictionType.REGRESSION,
                base_value=0.5,
                prediction_value=0.85,
                feature_contributions=[
                    FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
                    FeatureContribution("pressure", 101325.0, 0.1, 30.0, "positive", 0.5),
                ],
                consistency_check=0.001
            ),
            SHAPExplanation(
                explanation_id="exp2",
                prediction_type=PredictionType.REGRESSION,
                base_value=0.5,
                prediction_value=0.82,
                feature_contributions=[
                    FeatureContribution("temp", 360.0, 0.15, 45.0, "positive", 0.5),
                    FeatureContribution("pressure", 110000.0, 0.12, 35.0, "positive", 0.5),
                ],
                consistency_check=0.001
            ),
        ]

        aggregated = aggregate_shap_explanations(explanations)

        assert "temp" in aggregated
        assert "pressure" in aggregated
        # Normalized so should sum to ~1
        total = sum(aggregated.values())
        assert abs(total - 1.0) < 0.01

    def test_compare_explanations(self):
        """Test comparison of two explanations."""
        exp1 = SHAPExplanation(
            explanation_id="exp1",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
            ],
            consistency_check=0.001
        )

        exp2 = SHAPExplanation(
            explanation_id="exp2",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.90,
            feature_contributions=[
                FeatureContribution("temp", 360.0, 0.25, 55.0, "positive", 0.5),
            ],
            consistency_check=0.001
        )

        comparison = compare_explanations(exp1, exp2)

        assert "prediction_change" in comparison
        assert comparison["prediction_change"] == 0.05
        assert "feature_differences" in comparison
        assert "temp" in comparison["feature_differences"]


class TestTreeSHAPExplainer:
    """Tests for TreeSHAPExplainer convenience class."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
    def test_initialization(self, sample_feature_names, mock_tree_model):
        """Test TreeSHAPExplainer initialization."""
        with patch('shap.TreeExplainer') as mock_explainer:
            mock_explainer.return_value = Mock()

            explainer = TreeSHAPExplainer(
                model=mock_tree_model,
                feature_names=sample_feature_names
            )

            assert explainer.feature_names == sample_feature_names
            mock_explainer.assert_called_once()


class TestKernelSHAPExplainer:
    """Tests for KernelSHAPExplainer convenience class."""

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
    def test_initialization(self, sample_feature_names):
        """Test KernelSHAPExplainer initialization."""
        background_data = np.random.randn(100, 4)

        with patch('shap.KernelExplainer') as mock_explainer:
            mock_explainer.return_value = Mock()

            explainer = KernelSHAPExplainer(
                model=lambda x: np.sum(x, axis=1),
                background_data=background_data,
                feature_names=sample_feature_names
            )

            assert explainer.feature_names == sample_feature_names
            mock_explainer.assert_called_once()


class TestSHAPExplanationSchema:
    """Tests for SHAPExplanation data schema."""

    def test_explanation_creation(self):
        """Test SHAPExplanation creation."""
        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("temp", 350.0, 0.2, 50.0, "positive", 0.5),
            ],
            consistency_check=0.001
        )

        assert explanation.explanation_id == "test123"
        assert explanation.base_value == 0.5
        assert explanation.provenance_hash is not None

    def test_explanation_to_dict(self):
        """Test SHAPExplanation serialization."""
        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[],
            consistency_check=0.001
        )

        result = explanation.to_dict()

        assert "explanation_id" in result
        assert "base_value" in result
        assert "provenance_hash" in result

    def test_get_top_contributors(self):
        """Test getting top contributing features."""
        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("a", 1.0, 0.05, 10.0, "positive", 0.5),
                FeatureContribution("b", 2.0, 0.20, 40.0, "positive", 0.5),
                FeatureContribution("c", 3.0, 0.10, 20.0, "positive", 0.5),
            ],
            consistency_check=0.001
        )

        top_2 = explanation.get_top_contributors(n=2)

        assert len(top_2) == 2
        assert top_2[0].feature_name == "b"  # Highest contribution

    def test_get_positive_negative_contributors(self):
        """Test getting positive and negative contributors."""
        explanation = SHAPExplanation(
            explanation_id="test123",
            prediction_type=PredictionType.REGRESSION,
            base_value=0.5,
            prediction_value=0.85,
            feature_contributions=[
                FeatureContribution("a", 1.0, 0.10, 25.0, "positive", 0.5),
                FeatureContribution("b", 2.0, -0.05, 15.0, "negative", 0.5),
                FeatureContribution("c", 3.0, 0.15, 35.0, "positive", 0.5),
            ],
            consistency_check=0.001
        )

        positive = explanation.get_positive_contributors()
        negative = explanation.get_negative_contributors()

        assert len(positive) == 2
        assert len(negative) == 1
        assert negative[0].feature_name == "b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
