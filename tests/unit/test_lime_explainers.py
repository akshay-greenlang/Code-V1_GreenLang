# -*- coding: utf-8 -*-
"""
Unit tests for LIME Explainers (TASK-022).

Tests comprehensive functionality for:
- ProcessHeatLIMEExplainer with caching
- GL001LIMEExplainer for orchestrator decisions
- GL010LIMEExplainer for emissions predictions
- GL013LIMEExplainer for failure predictions
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from greenlang.ml.explainability.lime_explainer import (
    LIMEExplainer,
    ProcessHeatLIMEExplainer,
    GL001LIMEExplainer,
    GL010LIMEExplainer,
    GL013LIMEExplainer,
    LIMEExplainerConfig,
    LIMEResult,
    LIMEBatchResult,
    LIMEMode,
)


class MockModel:
    """Mock ML model for testing."""

    def predict_proba(self, X):
        """Return mock probabilities."""
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.column_stack([
            0.3 * np.ones(n_samples),
            0.5 * np.ones(n_samples),
            0.2 * np.ones(n_samples)
        ])

    def predict(self, X):
        """Return mock predictions."""
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.random.rand(n_samples)


class TestLIMEExplainerConfig:
    """Test LIMEExplainerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LIMEExplainerConfig()
        assert config.mode == LIMEMode.TABULAR
        assert config.kernel_width == 0.75
        assert config.num_samples == 5000
        assert config.num_features == 10
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LIMEExplainerConfig(
            kernel_width=0.5,
            num_samples=1000,
            feature_names=["a", "b", "c"]
        )
        assert config.kernel_width == 0.5
        assert config.num_samples == 1000
        assert config.feature_names == ["a", "b", "c"]

    def test_invalid_discretizer(self):
        """Test invalid discretizer raises error."""
        with pytest.raises(ValueError):
            LIMEExplainerConfig(discretizer="invalid")


class TestLIMEExplainer:
    """Test LIMEExplainer base class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = MockModel()
        explainer = LIMEExplainer(model)
        assert explainer.config.mode == LIMEMode.TABULAR
        assert explainer.model == model

    def test_init_with_training_data(self):
        """Test initialization with training data."""
        model = MockModel()
        training_data = np.random.randn(100, 5)
        explainer = LIMEExplainer(model, training_data=training_data)

        assert "mean" in explainer._training_data_stats
        assert explainer._training_data_stats["n_features"] == 5
        assert explainer._training_data_stats["n_samples"] == 100

    def test_compute_training_stats(self):
        """Test training data statistics computation."""
        model = MockModel()
        training_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        explainer = LIMEExplainer(model, training_data=training_data)

        stats = explainer._training_data_stats
        assert stats["n_features"] == 3
        assert stats["n_samples"] == 3
        assert len(stats["mean"]) == 3
        assert len(stats["std"]) == 3

    def test_get_prediction_function_proba(self):
        """Test getting prediction function from predict_proba."""
        model = MockModel()
        explainer = LIMEExplainer(model)
        pred_fn = explainer._get_prediction_function()

        X = np.array([[1, 2, 3]])
        result = pred_fn(X)
        assert result.shape == (1, 3)

    def test_get_prediction_function_predict(self):
        """Test getting prediction function from predict."""
        model = Mock()
        model.predict_proba = None
        model.predict = Mock(return_value=np.array([0.5]))
        explainer = LIMEExplainer(model)

        # Should use predict method
        pred_fn = explainer._get_prediction_function()
        assert pred_fn is not None

    def test_get_prediction_function_no_method(self):
        """Test error when model has no prediction method."""
        model = Mock(spec=[])
        explainer = LIMEExplainer(model)

        with pytest.raises(ValueError):
            explainer._get_prediction_function()

    def test_calculate_provenance_deterministic(self):
        """Test that provenance hash is deterministic."""
        model = MockModel()
        explainer = LIMEExplainer(model)

        instance = np.array([1.0, 2.0, 3.0])
        explanation = {"feature_0": 0.3, "feature_1": -0.2}

        hash1 = explainer._calculate_provenance(instance, explanation)
        hash2 = explainer._calculate_provenance(instance, explanation)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


class TestProcessHeatLIMEExplainer:
    """Test ProcessHeatLIMEExplainer with caching."""

    def test_init_with_cache(self):
        """Test initialization with cache."""
        model = MockModel()
        explainer = ProcessHeatLIMEExplainer(model, cache_size=100)

        assert explainer.cache_size == 100
        assert explainer._cache_hits == 0
        assert explainer._cache_misses == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        model = MockModel()
        explainer = ProcessHeatLIMEExplainer(model)

        instance = np.array([1.0, 2.0, 3.0])
        key = explainer._get_cache_key(instance)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex length

    def test_cache_key_consistency(self):
        """Test that cache keys are consistent."""
        model = MockModel()
        explainer = ProcessHeatLIMEExplainer(model)

        instance = np.array([1.0, 2.0, 3.0])
        key1 = explainer._get_cache_key(instance)
        key2 = explainer._get_cache_key(instance)

        assert key1 == key2

    def test_get_cache_stats(self):
        """Test cache statistics."""
        model = MockModel()
        explainer = ProcessHeatLIMEExplainer(model, cache_size=10)

        stats = explainer.get_cache_stats()

        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["cached_items"] == 0
        assert stats["cache_size"] == 10

    def test_clear_cache(self):
        """Test cache clearing."""
        model = MockModel()
        training_data = np.random.randn(50, 5)
        explainer = ProcessHeatLIMEExplainer(
            model,
            training_data=training_data,
            cache_size=100
        )

        # Note: Actual caching requires LIME, which we're mocking
        explainer._explanation_cache["test_key"] = Mock()

        assert len(explainer._explanation_cache) == 1

        explainer.clear_cache()

        assert len(explainer._explanation_cache) == 0


class TestGL001LIMEExplainer:
    """Test GL001LIMEExplainer."""

    def test_init_default_config(self):
        """Test GL001 initialization with default config."""
        model = MockModel()
        explainer = GL001LIMEExplainer(model)

        assert explainer.config.class_names == ["low_power", "medium_power", "high_power"]
        assert len(explainer.config.feature_names) == 10

    def test_feature_names(self):
        """Test GL001 feature names."""
        model = MockModel()
        explainer = GL001LIMEExplainer(model)

        expected_features = [
            "setpoint_temp", "current_temp", "boiler_power",
            "demand_forecast", "weather_temp", "system_efficiency",
            "fuel_cost", "grid_price", "thermal_load", "ambient_humidity"
        ]

        assert explainer.config.feature_names == expected_features

    def test_explain_decision_returns_dict(self):
        """Test that explain_decision returns proper structure."""
        model = MockModel()
        training_data = np.random.randn(50, 10)
        explainer = GL001LIMEExplainer(model, training_data=training_data)

        instance = np.random.randn(10)

        # Mock the parent explain_instance to avoid LIME dependency
        with patch.object(
            ProcessHeatLIMEExplainer,
            'explain_instance',
            return_value=LIMEResult(
                local_explanation={"feature": 0.5},
                intercept=0.1,
                local_prediction=0.5,
                model_prediction=0.5,
                r_squared=0.8,
                feature_weights=[("feature", 0.5)],
                provenance_hash="abc123",
                processing_time_ms=10.0,
                num_samples_used=5000,
                timestamp=datetime.utcnow()
            )
        ):
            result = explainer.explain_decision(instance)

            assert "decision_type" in result
            assert "model_prediction" in result
            assert "explanation" in result
            assert "top_factors" in result
            assert "confidence" in result
            assert "provenance_hash" in result


class TestGL010LIMEExplainer:
    """Test GL010LIMEExplainer."""

    def test_init_default_config(self):
        """Test GL010 initialization with default config."""
        model = MockModel()
        explainer = GL010LIMEExplainer(model)

        assert explainer.config.class_names == ["low_emissions", "medium_emissions", "high_emissions"]
        assert len(explainer.config.feature_names) == 10

    def test_feature_names(self):
        """Test GL010 feature names."""
        model = MockModel()
        explainer = GL010LIMEExplainer(model)

        expected_features = [
            "fuel_type", "fuel_quantity", "combustion_efficiency",
            "emission_factor", "co2_content", "ch4_content",
            "n2o_content", "operating_hours", "temperature", "oxygen_level"
        ]

        assert explainer.config.feature_names == expected_features

    def test_explain_emission_prediction(self):
        """Test emission prediction explanation."""
        model = MockModel()
        training_data = np.random.randn(50, 10)
        explainer = GL010LIMEExplainer(model, training_data=training_data)

        instance = np.random.randn(10)

        with patch.object(
            ProcessHeatLIMEExplainer,
            'explain_instance',
            return_value=LIMEResult(
                local_explanation={"fuel_type": 0.4},
                intercept=0.2,
                local_prediction=100.0,
                model_prediction=105.0,
                r_squared=0.85,
                feature_weights=[("fuel_type", 0.4)],
                provenance_hash="def456",
                processing_time_ms=12.0,
                num_samples_used=5000,
                timestamp=datetime.utcnow()
            )
        ):
            result = explainer.explain_emission_prediction(instance, emission_scope="scope1")

            assert result["emission_scope"] == "scope1"
            assert "predicted_emissions" in result
            assert "contributing_factors" in result
            assert "model_reliability" in result


class TestGL013LIMEExplainer:
    """Test GL013LIMEExplainer."""

    def test_init_default_config(self):
        """Test GL013 initialization with default config."""
        model = MockModel()
        explainer = GL013LIMEExplainer(model)

        assert explainer.config.class_names == ["healthy", "warning", "failure_imminent"]
        assert len(explainer.config.feature_names) == 10

    def test_classify_risk_low(self):
        """Test risk classification for low probability."""
        risk = GL013LIMEExplainer._classify_risk(0.2)
        assert risk == "low"

    def test_classify_risk_medium(self):
        """Test risk classification for medium probability."""
        risk = GL013LIMEExplainer._classify_risk(0.5)
        assert risk == "medium"

    def test_classify_risk_high(self):
        """Test risk classification for high probability."""
        risk = GL013LIMEExplainer._classify_risk(0.8)
        assert risk == "high"

    def test_explain_failure_prediction(self):
        """Test failure prediction explanation."""
        model = MockModel()
        training_data = np.random.randn(50, 10)
        explainer = GL013LIMEExplainer(model, training_data=training_data)

        instance = np.random.randn(10)

        with patch.object(
            ProcessHeatLIMEExplainer,
            'explain_instance',
            return_value=LIMEResult(
                local_explanation={"vibration_level": 0.6},
                intercept=0.3,
                local_prediction=0.75,
                model_prediction=0.75,
                r_squared=0.9,
                feature_weights=[("vibration_level", 0.6)],
                provenance_hash="ghi789",
                processing_time_ms=11.0,
                num_samples_used=5000,
                timestamp=datetime.utcnow()
            )
        ):
            result = explainer.explain_failure_prediction(instance, equipment_id="pump_001")

            assert result["equipment_id"] == "pump_001"
            assert result["failure_risk_level"] == "high"
            assert "failure_probability" in result
            assert "top_risk_factors" in result


class TestReportGeneration:
    """Test report generation functionality."""

    def test_generate_html_report_single(self):
        """Test HTML report generation for single explanation."""
        model = MockModel()
        explainer = LIMEExplainer(model)

        result = LIMEResult(
            local_explanation={"feature_0": 0.5},
            intercept=0.1,
            local_prediction=0.5,
            model_prediction=0.5,
            r_squared=0.8,
            feature_weights=[("feature_0", 0.5)],
            provenance_hash="abc123",
            processing_time_ms=10.0,
            num_samples_used=5000,
            timestamp=datetime.utcnow()
        )

        report = explainer.generate_report(result, output_format="html")

        assert "<html>" in report
        assert "LIME Explanation Report" in report
        assert "feature_0" in report
        assert "+0.5000" in report

    def test_generate_json_report_single(self):
        """Test JSON report generation for single explanation."""
        model = MockModel()
        explainer = LIMEExplainer(model)

        result = LIMEResult(
            local_explanation={"feature_0": 0.5},
            intercept=0.1,
            local_prediction=0.5,
            model_prediction=0.5,
            r_squared=0.8,
            feature_weights=[("feature_0", 0.5)],
            provenance_hash="abc123",
            processing_time_ms=10.0,
            num_samples_used=5000,
            timestamp=datetime.utcnow()
        )

        report = explainer.generate_report(result, output_format="json")

        assert '"type": "single"' in report
        assert '"model_prediction": 0.5' in report

    def test_generate_markdown_report_single(self):
        """Test Markdown report generation for single explanation."""
        model = MockModel()
        explainer = LIMEExplainer(model)

        result = LIMEResult(
            local_explanation={"feature_0": 0.5},
            intercept=0.1,
            local_prediction=0.5,
            model_prediction=0.5,
            r_squared=0.8,
            feature_weights=[("feature_0", 0.5)],
            provenance_hash="abc123",
            processing_time_ms=10.0,
            num_samples_used=5000,
            timestamp=datetime.utcnow()
        )

        report = explainer.generate_report(result, output_format="markdown")

        assert "# LIME Explanation Report" in report
        assert "feature_0" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
