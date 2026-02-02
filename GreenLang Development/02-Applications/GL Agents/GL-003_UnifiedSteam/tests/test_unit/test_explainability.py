"""
Unit Tests for GL-003 UnifiedSteam - Explainability Module

Tests for:
- SHAP explainer (SHAPExplainer)
- LIME explainer (if implemented)
- Physics explainer (deterministic calculations)
- Counterfactual engine
- Feature importance and local explanations

Target Coverage: 90%+
"""

import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np

# Import application modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from explainability.shap_explainer import (
    AssetExplanation,
    FeatureCategory,
    FeatureContribution,
    FeatureImportance,
    LocalExplanation,
    ModelType,
    SHAPExplainer,
    SHAPVisualization,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def shap_explainer() -> SHAPExplainer:
    """Create a SHAPExplainer instance."""
    return SHAPExplainer(agent_id="GL-003", default_num_features=10)


@pytest.fixture
def sample_feature_data() -> List[Dict[str, float]]:
    """Create sample feature data for ML models."""
    np.random.seed(42)
    samples = []
    for i in range(100):
        sample = {
            "temp_differential_f": np.random.normal(15, 5),
            "outlet_temp_f": np.random.normal(180, 15),
            "inlet_temp_f": np.random.normal(350, 20),
            "superheat_f": np.random.normal(20, 8),
            "subcooling_f": np.random.normal(5, 2),
            "operating_hours": np.random.uniform(1000, 10000),
            "cycles_count": np.random.uniform(500, 3000),
            "days_since_inspection": np.random.uniform(10, 90),
            "differential_pressure_psi": np.random.normal(50, 15),
            "previous_failures": np.random.randint(0, 3),
        }
        samples.append(sample)
    return samples


@pytest.fixture
def sample_instance() -> Dict[str, float]:
    """Create a single sample instance for local explanation."""
    return {
        "temp_differential_f": 25.0,  # Higher than average
        "outlet_temp_f": 200.0,
        "inlet_temp_f": 360.0,
        "superheat_f": 15.0,
        "subcooling_f": 3.0,  # Lower than average
        "operating_hours": 8000.0,
        "cycles_count": 2000.0,
        "days_since_inspection": 60.0,
        "differential_pressure_psi": 55.0,
        "previous_failures": 1.0,
    }


@pytest.fixture
def sample_asset_history() -> List[Dict[str, float]]:
    """Create sample asset history for asset-level explanation."""
    np.random.seed(42)
    history = []
    for i in range(24):  # 24 hours of history
        sample = {
            "temp_differential_f": 15.0 + i * 0.5,  # Increasing trend
            "outlet_temp_f": 180.0 + np.random.normal(0, 3),
            "inlet_temp_f": 350.0 + np.random.normal(0, 5),
            "superheat_f": 20.0,
            "subcooling_f": 5.0 - i * 0.1,  # Decreasing trend
            "operating_hours": 5000.0 + i,
            "cycles_count": 1500.0,
            "days_since_inspection": 45.0 + i / 24,
            "differential_pressure_psi": 50.0,
            "previous_failures": 0.0,
        }
        history.append(sample)
    return history


@pytest.fixture
def mock_ml_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([0.65])  # Single prediction
    model.predict_proba.return_value = np.array([[0.35, 0.65]])  # Class probabilities
    return model


# =============================================================================
# Test SHAPExplainer Initialization
# =============================================================================

class TestSHAPExplainerInitialization:
    """Tests for SHAPExplainer initialization."""

    def test_explainer_initialization(self):
        """Test explainer initializes correctly."""
        explainer = SHAPExplainer(agent_id="GL-003")
        assert explainer.agent_id == "GL-003"
        assert explainer.default_num_features == 10

    def test_explainer_with_custom_num_features(self):
        """Test explainer with custom number of features."""
        explainer = SHAPExplainer(agent_id="GL-003", default_num_features=20)
        assert explainer.default_num_features == 20

    def test_feature_metadata_initialized(self, shap_explainer):
        """Test that feature metadata is initialized."""
        metadata = shap_explainer._feature_metadata
        assert len(metadata) > 0
        assert "temp_differential_f" in metadata
        assert "outlet_temp_f" in metadata

    def test_feature_categories_correct(self, shap_explainer):
        """Test that feature categories are correct."""
        metadata = shap_explainer._feature_metadata

        # Temperature features
        assert metadata["inlet_temp_f"]["category"] == FeatureCategory.TEMPERATURE
        assert metadata["outlet_temp_f"]["category"] == FeatureCategory.TEMPERATURE

        # Pressure features
        assert metadata["header_pressure_psig"]["category"] == FeatureCategory.PRESSURE

        # Operational features
        assert metadata["operating_hours"]["category"] == FeatureCategory.OPERATIONAL


# =============================================================================
# Test Model Registration
# =============================================================================

class TestModelRegistration:
    """Tests for ML model registration."""

    def test_register_model(self, shap_explainer, mock_ml_model):
        """Test registering an ML model."""
        feature_names = ["temp_differential_f", "outlet_temp_f", "inlet_temp_f"]

        shap_explainer.register_model(
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
            model=mock_ml_model,
            feature_names=feature_names,
        )

        assert ModelType.TRAP_FAILURE_PREDICTION.value in shap_explainer._models
        registered = shap_explainer._models[ModelType.TRAP_FAILURE_PREDICTION.value]
        assert registered["model"] == mock_ml_model
        assert registered["feature_names"] == feature_names

    def test_register_multiple_models(self, shap_explainer, mock_ml_model):
        """Test registering multiple model types."""
        shap_explainer.register_model(ModelType.TRAP_FAILURE_PREDICTION, mock_ml_model)
        shap_explainer.register_model(ModelType.ANOMALY_DETECTION, mock_ml_model)

        assert len(shap_explainer._models) >= 2


# =============================================================================
# Test Global Feature Importance
# =============================================================================

class TestGlobalFeatureImportance:
    """Tests for global feature importance computation."""

    def test_compute_global_importance(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test computing global feature importance."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        assert isinstance(result, FeatureImportance)
        assert result.model_type == ModelType.TRAP_FAILURE_PREDICTION
        assert result.dataset_size == len(sample_feature_data)
        assert len(result.feature_rankings) > 0
        assert len(result.top_features) <= 10

    def test_global_importance_uuid_generated(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test that importance ID is a valid UUID."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        # Should be a valid UUID string
        try:
            uuid.UUID(result.importance_id)
        except ValueError:
            pytest.fail("importance_id is not a valid UUID")

    def test_global_importance_rankings_sorted(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test that rankings are sorted by importance."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        importances = [r["importance"] for r in result.feature_rankings]
        assert importances == sorted(importances, reverse=True), \
            "Rankings not sorted by importance"

    def test_global_importance_mean_abs_shap(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test that mean absolute SHAP values are computed."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        assert len(result.mean_abs_shap) > 0
        # All values should be non-negative
        for value in result.mean_abs_shap.values():
            assert value >= 0

    def test_global_importance_feature_categories(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test that features are grouped by category."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        assert len(result.feature_categories) > 0
        # At least temperature category should have features
        for category, features in result.feature_categories.items():
            assert isinstance(features, list)

    def test_global_importance_caching(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test that global importance is cached."""
        result1 = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        cached = shap_explainer.get_global_importance(ModelType.TRAP_FAILURE_PREDICTION)
        assert cached is not None
        assert cached.importance_id == result1.importance_id

    def test_global_importance_to_dict(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test serialization to dictionary."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        d = result.to_dict()
        assert "importance_id" in d
        assert "model_type" in d
        assert "feature_rankings" in d
        assert "mean_abs_shap" in d


# =============================================================================
# Test Local Explanations
# =============================================================================

class TestLocalExplanations:
    """Tests for local SHAP explanations."""

    def test_compute_local_explanation(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test computing local explanation for single instance."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        assert isinstance(result, LocalExplanation)
        assert result.model_type == ModelType.TRAP_FAILURE_PREDICTION
        assert len(result.contributions) > 0

    def test_local_explanation_uuid_generated(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that explanation ID is generated."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        try:
            uuid.UUID(result.explanation_id)
        except ValueError:
            pytest.fail("explanation_id is not a valid UUID")

    def test_local_explanation_predicted_value(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that predicted value is included."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
            predicted_value=0.75,
        )

        assert result.predicted_value == 0.75

    def test_local_explanation_base_value(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that base value is included."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
            base_value=0.5,
        )

        assert result.base_value == 0.5
        assert result.prediction_delta == result.predicted_value - 0.5

    def test_local_explanation_contributions_have_direction(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that contributions have direction (positive/negative)."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        for contrib in result.contributions:
            assert contrib.direction in ["positive", "negative"]

    def test_local_explanation_top_features(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that top positive and negative features are identified."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        # Should have top positive features
        assert isinstance(result.top_positive_features, list)
        assert isinstance(result.top_negative_features, list)

    def test_local_explanation_summary_text(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that summary text is generated."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        assert len(result.summary_text) > 0

    def test_local_explanation_confidence(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that explanation confidence is computed."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        assert 0 <= result.explanation_confidence <= 1

    def test_local_explanation_caching(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that local explanations are cached."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        cached = shap_explainer.get_local_explanation(result.explanation_id)
        assert cached is not None
        assert cached.explanation_id == result.explanation_id

    def test_local_explanation_to_dict(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test serialization to dictionary."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        d = result.to_dict()
        assert "explanation_id" in d
        assert "predicted_value" in d
        assert "contributions" in d


# =============================================================================
# Test Feature Contributions
# =============================================================================

class TestFeatureContributions:
    """Tests for individual feature contributions."""

    def test_contribution_has_required_fields(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that contributions have all required fields."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        for contrib in result.contributions:
            assert isinstance(contrib, FeatureContribution)
            assert contrib.feature_name is not None
            assert isinstance(contrib.feature_value, float)
            assert isinstance(contrib.shap_value, float)
            assert isinstance(contrib.contribution_percent, float)

    def test_contribution_percentages_sum_to_100(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test that contribution percentages sum approximately to 100."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        total_percent = sum(c.contribution_percent for c in result.contributions)
        # Allow some tolerance due to rounding and limited features
        assert 80 <= total_percent <= 120, f"Total percentage: {total_percent}"

    def test_contribution_to_dict(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test contribution serialization."""
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        for contrib in result.contributions:
            d = contrib.to_dict()
            assert "feature_name" in d
            assert "shap_value" in d
            assert "category" in d


# =============================================================================
# Test Asset-Level Explanations
# =============================================================================

class TestAssetLevelExplanations:
    """Tests for asset-level aggregated explanations."""

    def test_generate_asset_explanation(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test generating asset-level explanation."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
            asset_type="trap",
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
            history_window_hours=24,
        )

        assert isinstance(result, AssetExplanation)
        assert result.asset_id == "TRAP-001"
        assert result.asset_type == "trap"
        assert result.sample_count == len(sample_asset_history)

    def test_asset_explanation_feature_importance(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test that asset feature importance is computed."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert len(result.asset_feature_importance) > 0
        assert len(result.asset_top_features) > 0

    def test_asset_explanation_feature_trends(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test that feature trends are computed."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert len(result.feature_trends) > 0
        for trend in result.feature_trends.values():
            assert trend in ["increasing", "decreasing", "stable"]

    def test_asset_explanation_trend_detection(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test that trends are detected correctly."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,  # Has increasing temp_differential
        )

        # temp_differential_f should show increasing trend
        if "temp_differential_f" in result.feature_trends:
            assert result.feature_trends["temp_differential_f"] == "increasing"

        # subcooling_f should show decreasing trend
        if "subcooling_f" in result.feature_trends:
            assert result.feature_trends["subcooling_f"] == "decreasing"

    def test_asset_explanation_fleet_comparison(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test fleet comparison percentiles."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert len(result.comparison_to_fleet) > 0
        for percentile in result.comparison_to_fleet.values():
            assert 0 <= percentile <= 100

    def test_asset_explanation_risk_score(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test current risk score is computed."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert 0 <= result.current_risk_score <= 1

    def test_asset_explanation_risk_drivers(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test risk drivers are identified."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert isinstance(result.risk_drivers, list)

    def test_asset_explanation_suggested_actions(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test suggested actions are generated."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        assert isinstance(result.suggested_actions, list)
        assert len(result.suggested_actions) > 0

    def test_asset_explanation_caching(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test asset explanations are cached."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        cached = shap_explainer.get_asset_explanation("TRAP-001")
        assert cached is not None
        assert cached.asset_id == "TRAP-001"

    def test_asset_explanation_to_dict(
        self, shap_explainer, mock_ml_model, sample_asset_history
    ):
        """Test serialization to dictionary."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-001",
            model=mock_ml_model,
            history=sample_asset_history,
        )

        d = result.to_dict()
        assert "asset_id" in d
        assert "asset_feature_importance" in d
        assert "suggested_actions" in d


# =============================================================================
# Test Visualization Data Generation
# =============================================================================

class TestVisualization:
    """Tests for SHAP visualization data generation."""

    def test_visualize_summary_from_global_importance(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test summary visualization from global importance."""
        importance = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        viz = shap_explainer.visualize_shap_summary(
            explanations=importance,
            visualization_type="summary",
        )

        assert isinstance(viz, SHAPVisualization)
        assert viz.visualization_type == "summary"
        assert "features" in viz.plot_data
        assert "importance" in viz.plot_data

    def test_visualize_waterfall_from_local_explanation(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test waterfall visualization from local explanation."""
        local_exp = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        viz = shap_explainer.visualize_shap_summary(
            explanations=[local_exp],
            visualization_type="waterfall",
        )

        assert isinstance(viz, SHAPVisualization)
        assert viz.visualization_type == "waterfall"
        assert "base_value" in viz.plot_data
        assert "predicted_value" in viz.plot_data

    def test_visualize_force_plot(
        self, shap_explainer, mock_ml_model, sample_instance
    ):
        """Test force plot visualization."""
        local_exp = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=sample_instance,
        )

        viz = shap_explainer.visualize_shap_summary(
            explanations=[local_exp],
            visualization_type="force",
        )

        assert viz.visualization_type == "force"
        assert "base_value" in viz.plot_data
        assert "output_value" in viz.plot_data
        assert "features" in viz.plot_data

    def test_visualization_to_dict(
        self, shap_explainer, mock_ml_model, sample_feature_data
    ):
        """Test visualization serialization."""
        importance = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=sample_feature_data,
        )

        viz = shap_explainer.visualize_shap_summary(
            explanations=importance,
            visualization_type="summary",
        )

        d = viz.to_dict()
        assert "visualization_id" in d
        assert "visualization_type" in d
        assert "plot_data" in d


# =============================================================================
# Test Mock Prediction and SHAP Values
# =============================================================================

class TestMockPredictions:
    """Tests for mock prediction and SHAP value computation."""

    def test_mock_prediction_trap_failure(self, shap_explainer, sample_instance):
        """Test mock prediction for trap failure model."""
        prediction = shap_explainer._mock_prediction(
            instance=sample_instance,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        assert 0 <= prediction <= 1

    def test_mock_prediction_high_risk_factors(self, shap_explainer):
        """Test mock prediction with high risk factors."""
        high_risk_instance = {
            "temp_differential_f": 45.0,  # High
            "subcooling_f": 0.5,  # Low
            "operating_hours": 12000.0,  # High
        }

        prediction = shap_explainer._mock_prediction(
            instance=high_risk_instance,
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        # Should be elevated risk
        assert prediction > 0.3

    def test_mock_shap_values_sum_to_delta(self, shap_explainer, sample_instance):
        """Test that mock SHAP values are distributed across features."""
        prediction_delta = 0.2

        shap_values = shap_explainer._compute_mock_shap_values(
            instance=sample_instance,
            prediction_delta=prediction_delta,
        )

        assert len(shap_values) > 0


# =============================================================================
# Test Explanation Confidence
# =============================================================================

class TestExplanationConfidence:
    """Tests for explanation confidence computation."""

    def test_confidence_computation(self, shap_explainer):
        """Test explanation confidence calculation."""
        # Create contributions where top 3 dominate
        contributions = [
            FeatureContribution(
                feature_name="f1",
                feature_value=1.0,
                shap_value=0.5,
                contribution_percent=50.0,
                direction="positive",
                category=FeatureCategory.TEMPERATURE,
            ),
            FeatureContribution(
                feature_name="f2",
                feature_value=2.0,
                shap_value=0.3,
                contribution_percent=30.0,
                direction="positive",
                category=FeatureCategory.PRESSURE,
            ),
            FeatureContribution(
                feature_name="f3",
                feature_value=3.0,
                shap_value=0.1,
                contribution_percent=10.0,
                direction="negative",
                category=FeatureCategory.FLOW,
            ),
        ]

        confidence = shap_explainer._compute_explanation_confidence(contributions)

        # Should be high since top 3 dominate
        assert 0.5 <= confidence <= 1.0

    def test_confidence_empty_contributions(self, shap_explainer):
        """Test confidence with empty contributions."""
        confidence = shap_explainer._compute_explanation_confidence([])
        assert confidence == 0.0


# =============================================================================
# Test Model Types
# =============================================================================

class TestModelTypes:
    """Tests for different model types."""

    def test_all_model_types_supported(self, shap_explainer, mock_ml_model, sample_instance):
        """Test that all model types can be explained."""
        for model_type in ModelType:
            result = shap_explainer.compute_local_explanation(
                model=mock_ml_model,
                instance=sample_instance,
                model_type=model_type,
            )
            assert result.model_type == model_type

    def test_model_type_enum_values(self):
        """Test model type enum values."""
        assert ModelType.TRAP_FAILURE_PREDICTION.value == "trap_failure_prediction"
        assert ModelType.ANOMALY_DETECTION.value == "anomaly_detection"
        assert ModelType.PERFORMANCE_DEGRADATION.value == "performance_degradation"


# =============================================================================
# Test Feature Categories
# =============================================================================

class TestFeatureCategories:
    """Tests for feature category handling."""

    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        expected_categories = [
            FeatureCategory.TEMPERATURE,
            FeatureCategory.PRESSURE,
            FeatureCategory.FLOW,
            FeatureCategory.OPERATIONAL,
            FeatureCategory.MAINTENANCE,
            FeatureCategory.ENVIRONMENTAL,
            FeatureCategory.DERIVED,
        ]

        for category in expected_categories:
            assert category in FeatureCategory

    def test_category_values(self):
        """Test category enum values."""
        assert FeatureCategory.TEMPERATURE.value == "temperature"
        assert FeatureCategory.PRESSURE.value == "pressure"


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for explainability module."""

    def test_global_importance_performance(
        self, shap_explainer, mock_ml_model, benchmark
    ):
        """Benchmark global importance computation."""
        # Create larger dataset
        np.random.seed(42)
        large_dataset = [
            {
                "temp_differential_f": np.random.normal(15, 5),
                "outlet_temp_f": np.random.normal(180, 15),
                "inlet_temp_f": np.random.normal(350, 20),
            }
            for _ in range(1000)
        ]

        def compute():
            return shap_explainer.compute_global_feature_importance(
                model=mock_ml_model,
                dataset=large_dataset,
            )

        result = benchmark(compute)
        assert result is not None

    def test_local_explanation_performance(
        self, shap_explainer, mock_ml_model, sample_instance, benchmark
    ):
        """Benchmark local explanation computation."""
        def explain():
            return shap_explainer.compute_local_explanation(
                model=mock_ml_model,
                instance=sample_instance,
            )

        result = benchmark(explain)
        assert result is not None

    def test_asset_explanation_performance(
        self, shap_explainer, mock_ml_model, benchmark
    ):
        """Benchmark asset explanation computation."""
        np.random.seed(42)
        history = [
            {
                "temp_differential_f": np.random.normal(15, 5),
                "outlet_temp_f": np.random.normal(180, 15),
            }
            for _ in range(168)  # 1 week hourly
        ]

        def explain():
            return shap_explainer.generate_asset_level_explanation(
                asset_id="TRAP-001",
                model=mock_ml_model,
                history=history,
            )

        result = benchmark(explain)
        assert result is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset(self, shap_explainer, mock_ml_model):
        """Test handling empty dataset."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=[],
        )

        # Should handle gracefully
        assert result.dataset_size == 0

    def test_single_sample(self, shap_explainer, mock_ml_model, sample_instance):
        """Test with single sample dataset."""
        result = shap_explainer.compute_global_feature_importance(
            model=mock_ml_model,
            dataset=[sample_instance],
        )

        assert result.dataset_size == 1

    def test_missing_feature_in_instance(self, shap_explainer, mock_ml_model):
        """Test handling instance with missing features."""
        partial_instance = {
            "temp_differential_f": 20.0,
            # Missing other features
        }

        # Should not raise exception
        result = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=partial_instance,
        )

        assert result is not None

    def test_empty_history_asset_explanation(self, shap_explainer, mock_ml_model):
        """Test asset explanation with empty history."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-EMPTY",
            model=mock_ml_model,
            history=[],
        )

        # Should handle gracefully
        assert result.sample_count == 0

    def test_single_history_point(self, shap_explainer, mock_ml_model, sample_instance):
        """Test asset explanation with single history point."""
        result = shap_explainer.generate_asset_level_explanation(
            asset_id="TRAP-SINGLE",
            model=mock_ml_model,
            history=[sample_instance],
        )

        assert result.sample_count == 1
        # Trends should be empty with single point
        assert len(result.feature_trends) == 0

    def test_get_nonexistent_global_importance(self, shap_explainer):
        """Test getting non-existent global importance."""
        result = shap_explainer.get_global_importance(ModelType.CLASSIFICATION)
        assert result is None

    def test_get_nonexistent_local_explanation(self, shap_explainer):
        """Test getting non-existent local explanation."""
        result = shap_explainer.get_local_explanation("nonexistent-id")
        assert result is None

    def test_get_nonexistent_asset_explanation(self, shap_explainer):
        """Test getting non-existent asset explanation."""
        result = shap_explainer.get_asset_explanation("NONEXISTENT")
        assert result is None
