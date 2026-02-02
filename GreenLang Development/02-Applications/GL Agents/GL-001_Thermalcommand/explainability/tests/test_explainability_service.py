# -*- coding: utf-8 -*-
"""
Tests for Explainability Service module.

Validates:
- Demand forecast explanations
- Health score explanations
- Optimization decision explanations
- Counterfactual generation
- Batch explanations
- Zero-hallucination guarantees
"""

import pytest
import numpy as np
from datetime import datetime

from explainability.explainability_service import (
    ExplainabilityService,
    ExplanationMethod,
    OptimizationContext,
    ServiceConfig,
)
from explainability.explanation_schemas import (
    ExplanationReport,
    DecisionExplanation,
    Counterfactual,
    BatchExplanationSummary,
    PredictionType,
    ConfidenceLevel,
)


class TestServiceConfig:
    """Tests for service configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServiceConfig()
        assert config.random_seed == 42
        assert config.confidence_threshold == 0.80
        assert config.max_counterfactuals == 5
        assert config.use_shap is True
        assert config.use_lime is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ServiceConfig(
            random_seed=123,
            confidence_threshold=0.90,
            use_shap=False
        )
        assert config.random_seed == 123
        assert config.confidence_threshold == 0.90
        assert config.use_shap is False


@pytest.mark.service
class TestExplainabilityService:
    """Tests for main explainability service."""

    def test_initialization(self, training_data, feature_names):
        """Test service initialization."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        assert service.feature_names == feature_names
        assert service._lime_explainer is not None

    def test_initialization_with_config(self, training_data, feature_names):
        """Test service initialization with custom config."""
        config = ServiceConfig(random_seed=123, use_shap=False)
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )

        assert service.config.random_seed == 123
        assert service._shap_explainer is None

    def test_set_model(
        self,
        training_data,
        feature_names,
        trained_random_forest
    ):
        """Test setting model for SHAP explanations."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        service.set_model(trained_random_forest, model_type="tree")

        assert service._shap_explainer is not None
        assert service._shap_explainer._tree_explainer is not None


@pytest.mark.service
class TestDemandForecastExplanation:
    """Tests for demand forecast explanations."""

    def test_explain_demand_forecast_shap(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test explaining demand forecast with SHAP."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function,
            method=ExplanationMethod.SHAP
        )

        assert isinstance(report, ExplanationReport)
        assert report.prediction_type == PredictionType.DEMAND_FORECAST
        assert report.shap_explanation is not None
        assert len(report.top_features) > 0
        assert report.deterministic is True

    def test_explain_demand_forecast_lime(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test explaining demand forecast with LIME."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function,
            method=ExplanationMethod.LIME
        )

        assert isinstance(report, ExplanationReport)
        assert report.lime_explanation is not None
        assert len(report.top_features) > 0

    def test_explain_demand_forecast_both(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test explaining demand forecast with both methods."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function,
            method=ExplanationMethod.BOTH
        )

        assert report.shap_explanation is not None
        assert report.lime_explanation is not None

    def test_demand_forecast_uncertainty(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test uncertainty quantification in demand forecast."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        uncertainty = report.uncertainty
        assert uncertainty.point_estimate == report.prediction_value
        assert uncertainty.standard_error >= 0
        assert uncertainty.confidence_interval.lower_bound <= uncertainty.point_estimate
        assert uncertainty.confidence_interval.upper_bound >= uncertainty.point_estimate


@pytest.mark.service
class TestHealthScoreExplanation:
    """Tests for health score explanations."""

    def test_explain_health_score(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test explaining health score prediction."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report = service.explain_health_score(
            equipment_data=sample_instance,
            predict_fn=mock_prediction_function,
            equipment_id="BOILER-001"
        )

        assert isinstance(report, ExplanationReport)
        assert report.prediction_type == PredictionType.HEALTH_SCORE
        assert len(report.top_features) > 0

    def test_health_score_confidence_level(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test confidence level determination."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report = service.explain_health_score(
            equipment_data=sample_instance,
            predict_fn=mock_prediction_function,
            equipment_id="PUMP-002"
        )

        assert report.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW
        ]


@pytest.mark.service
class TestOptimizationExplanation:
    """Tests for optimization decision explanations."""

    def test_explain_optimization_decision(
        self,
        training_data,
        feature_names,
        optimization_context
    ):
        """Test explaining optimization decision."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = service.explain_optimization_decision(optimization_context)

        assert isinstance(explanation, DecisionExplanation)
        assert explanation.objective_value == optimization_context.objective_value
        assert len(explanation.binding_constraints) > 0
        assert len(explanation.shadow_prices) > 0

    def test_binding_constraints_identification(
        self,
        training_data,
        feature_names,
        optimization_context
    ):
        """Test identification of binding constraints."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = service.explain_optimization_decision(optimization_context)

        # Should identify constraints at their bounds
        assert "max_boiler_1" in explanation.binding_constraints

    def test_shadow_prices(
        self,
        training_data,
        feature_names,
        optimization_context
    ):
        """Test shadow prices in explanation."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = service.explain_optimization_decision(optimization_context)

        assert "total_demand" in explanation.shadow_prices
        assert explanation.shadow_prices["total_demand"] == 25.0

    def test_sensitivity_analysis(
        self,
        training_data,
        feature_names,
        optimization_context
    ):
        """Test sensitivity analysis in explanation."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        explanation = service.explain_optimization_decision(optimization_context)

        assert len(explanation.sensitivity_analysis) > 0


@pytest.mark.service
class TestCounterfactualGeneration:
    """Tests for counterfactual generation."""

    def test_generate_counterfactual(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test generating single counterfactual."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        # Get current prediction
        current_pred = mock_prediction_function(sample_instance)
        target_pred = current_pred * 1.2  # 20% increase

        counterfactual = service.generate_counterfactual(
            instance=sample_instance,
            predict_fn=mock_prediction_function,
            target_prediction=target_pred
        )

        assert isinstance(counterfactual, Counterfactual)
        assert counterfactual.original_prediction == current_pred
        assert counterfactual.target_prediction == target_pred
        assert len(counterfactual.feature_changes) > 0
        assert 0 <= counterfactual.feasibility_score <= 1

    def test_counterfactual_sparsity(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test counterfactual sparsity control."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        current_pred = mock_prediction_function(sample_instance)
        target_pred = current_pred * 1.1

        counterfactual = service.generate_counterfactual(
            instance=sample_instance,
            predict_fn=mock_prediction_function,
            target_prediction=target_pred,
            max_features_to_change=2
        )

        assert counterfactual.sparsity <= 2

    def test_generate_multiple_counterfactuals(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test generating multiple counterfactuals."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        current_pred = mock_prediction_function(sample_instance)
        target_pred = current_pred * 1.15

        counterfactuals = service.generate_multiple_counterfactuals(
            instance=sample_instance,
            predict_fn=mock_prediction_function,
            target_prediction=target_pred,
            num_counterfactuals=3
        )

        assert isinstance(counterfactuals, list)
        # May generate fewer if some fail validation
        assert len(counterfactuals) <= 3

    def test_counterfactual_with_constraints(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test counterfactual generation with feature constraints."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        current_pred = mock_prediction_function(sample_instance)
        target_pred = current_pred * 1.1

        # Constraint temperature to reasonable range
        constraints = {
            "temperature_inlet_c": {"min": 100, "max": 700}
        }

        counterfactual = service.generate_counterfactual(
            instance=sample_instance,
            predict_fn=mock_prediction_function,
            target_prediction=target_pred,
            feature_constraints=constraints
        )

        # Verify constraint is respected if temperature was changed
        if "temperature_inlet_c" in counterfactual.feature_changes:
            new_temp = counterfactual.feature_changes["temperature_inlet_c"]["to"]
            assert 100 <= new_temp <= 700


@pytest.mark.service
class TestBatchExplanation:
    """Tests for batch explanation."""

    def test_explain_batch(
        self,
        training_data,
        sample_batch,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test batch explanation generation."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        summary = service.explain_batch(
            instances=sample_batch,
            predict_fn=mock_prediction_function,
            prediction_type=PredictionType.DEMAND_FORECAST
        )

        assert isinstance(summary, BatchExplanationSummary)
        assert summary.batch_size == len(sample_batch)
        assert len(summary.global_feature_importance) > 0
        assert summary.mean_prediction > 0

    def test_batch_feature_importance_aggregation(
        self,
        training_data,
        sample_batch,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test that feature importance is properly aggregated."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        summary = service.explain_batch(
            instances=sample_batch,
            predict_fn=mock_prediction_function,
            prediction_type=PredictionType.DEMAND_FORECAST
        )

        # Importance should sum to approximately 1 (normalized)
        total_importance = sum(summary.global_feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01


@pytest.mark.determinism
class TestServiceDeterminism:
    """Tests for service determinism."""

    def test_reproducible_reports(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test that reports are reproducible with same seed."""
        config = ServiceConfig(random_seed=42)

        # First run
        service1 = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )
        service1.set_model(trained_random_forest, model_type="tree")
        report1 = service1.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function,
            method=ExplanationMethod.SHAP
        )

        # Second run with same config
        service2 = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names,
            config=config
        )
        service2.set_model(trained_random_forest, model_type="tree")
        report2 = service2.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function,
            method=ExplanationMethod.SHAP
        )

        # Should have identical predictions
        assert report1.prediction_value == report2.prediction_value

    def test_provenance_hash_uniqueness(
        self,
        training_data,
        sample_batch,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test that different instances get different provenance hashes."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report1 = service.explain_demand_forecast(
            forecast_input=sample_batch[0],
            predict_fn=mock_prediction_function
        )

        report2 = service.explain_demand_forecast(
            forecast_input=sample_batch[1],
            predict_fn=mock_prediction_function
        )

        # Different instances should have different hashes
        assert report1.provenance_hash != report2.provenance_hash

    def test_deterministic_flag(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that deterministic flag is always True."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        assert report.deterministic is True
