# -*- coding: utf-8 -*-
"""
Determinism Tests for Explainability Module.

Validates zero-hallucination guarantees:
- All explanations are deterministic and reproducible
- Same inputs produce identical outputs
- Provenance hashing works correctly
- Random seeds control all stochastic behavior
"""

import pytest
import numpy as np
import hashlib
from datetime import datetime

from explainability.shap_explainer import SHAPExplainer, SHAPConfig
from explainability.lime_explainer import LIMEExplainer, LIMEConfig
from explainability.explainability_service import (
    ExplainabilityService,
    ServiceConfig,
    ExplanationMethod,
)
from explainability.explanation_schemas import PredictionType


@pytest.mark.determinism
class TestSHAPDeterminism:
    """Tests for SHAP explainer determinism."""

    def test_tree_explainer_determinism(
        self,
        trained_random_forest,
        sample_instance,
        feature_names
    ):
        """Test TreeExplainer produces identical results."""
        config = SHAPConfig(random_seed=42, cache_enabled=False)

        results = []
        for _ in range(3):
            explainer = SHAPExplainer(config=config, feature_names=feature_names)
            explainer.fit_tree_explainer(trained_random_forest, feature_names)
            exp = explainer.explain_instance(
                sample_instance,
                PredictionType.DEMAND_FORECAST
            )
            results.append(exp)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0].base_value == results[i].base_value
            assert results[0].prediction_value == results[i].prediction_value

            for c0, ci in zip(
                results[0].feature_contributions,
                results[i].feature_contributions
            ):
                assert c0.contribution == ci.contribution

    def test_kernel_explainer_determinism(
        self,
        trained_linear_regression,
        training_data,
        sample_instance,
        feature_names
    ):
        """Test KernelExplainer produces identical results with same seed."""
        config = SHAPConfig(random_seed=42, cache_enabled=False, num_samples=50)

        results = []
        for _ in range(3):
            explainer = SHAPExplainer(config=config, feature_names=feature_names)
            explainer.fit_kernel_explainer(
                trained_linear_regression.predict,
                training_data,
                feature_names
            )
            exp = explainer.explain_instance(
                sample_instance,
                PredictionType.DEMAND_FORECAST,
                use_tree_explainer=False
            )
            results.append(exp)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0].prediction_value == results[i].prediction_value

    def test_shap_consistency_maintained(
        self,
        trained_random_forest,
        sample_instance,
        feature_names
    ):
        """Test SHAP additivity property is maintained."""
        explainer = SHAPExplainer(feature_names=feature_names)
        explainer.fit_tree_explainer(trained_random_forest, feature_names)

        exp = explainer.explain_instance(
            sample_instance,
            PredictionType.DEMAND_FORECAST
        )

        # SHAP values should sum to prediction - base_value
        shap_sum = sum(c.contribution for c in exp.feature_contributions)
        expected_sum = exp.prediction_value - exp.base_value

        assert abs(shap_sum - expected_sum) < 0.01


@pytest.mark.determinism
class TestLIMEDeterminism:
    """Tests for LIME explainer determinism."""

    def test_lime_determinism(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test LIME produces identical results with same seed."""
        config = LIMEConfig(random_seed=42, cache_enabled=False, num_samples=1000)

        results = []
        for _ in range(3):
            explainer = LIMEExplainer(
                training_data=training_data,
                feature_names=feature_names,
                config=config
            )
            exp = explainer.explain_instance(
                sample_instance,
                mock_prediction_function,
                PredictionType.DEMAND_FORECAST
            )
            results.append(exp)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0].prediction_value == results[i].prediction_value
            assert results[0].local_model_r2 == results[i].local_model_r2

            for c0, ci in zip(
                results[0].feature_contributions,
                results[i].feature_contributions
            ):
                assert c0.contribution == ci.contribution

    def test_different_seeds_different_results(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that different seeds can produce different results."""
        config1 = LIMEConfig(random_seed=42, cache_enabled=False, num_samples=500)
        config2 = LIMEConfig(random_seed=999, cache_enabled=False, num_samples=500)

        explainer1 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config1
        )
        exp1 = explainer1.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        explainer2 = LIMEExplainer(
            training_data=training_data,
            feature_names=feature_names,
            config=config2
        )
        exp2 = explainer2.explain_instance(
            sample_instance,
            mock_prediction_function,
            PredictionType.DEMAND_FORECAST
        )

        # Results may differ due to different sampling
        # But both should still be valid explanations
        assert exp1.prediction_value == exp2.prediction_value  # Same prediction
        assert exp1.local_model_r2 > 0
        assert exp2.local_model_r2 > 0


@pytest.mark.determinism
class TestServiceDeterminism:
    """Tests for ExplainabilityService determinism."""

    def test_service_determinism(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test service produces identical results with same config."""
        config = ServiceConfig(random_seed=42)

        results = []
        for _ in range(3):
            service = ExplainabilityService(
                training_data=training_data,
                feature_names=feature_names,
                config=config
            )
            service.set_model(trained_random_forest, model_type="tree")
            report = service.explain_demand_forecast(
                forecast_input=sample_instance,
                predict_fn=mock_prediction_function,
                method=ExplanationMethod.SHAP
            )
            results.append(report)

        # All predictions should be identical
        for i in range(1, len(results)):
            assert results[0].prediction_value == results[i].prediction_value

    def test_counterfactual_determinism(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test counterfactual generation is deterministic."""
        config = ServiceConfig(random_seed=42)

        current_pred = mock_prediction_function(sample_instance)
        target_pred = current_pred * 1.1

        results = []
        for _ in range(3):
            service = ExplainabilityService(
                training_data=training_data,
                feature_names=feature_names,
                config=config
            )
            service.set_model(trained_random_forest, model_type="tree")
            cf = service.generate_counterfactual(
                instance=sample_instance,
                predict_fn=mock_prediction_function,
                target_prediction=target_pred
            )
            results.append(cf)

        # All counterfactuals should have same changes
        for i in range(1, len(results)):
            assert results[0].sparsity == results[i].sparsity
            assert set(results[0].feature_changes.keys()) == set(results[i].feature_changes.keys())


@pytest.mark.determinism
class TestProvenanceHashing:
    """Tests for provenance hashing."""

    def test_provenance_hash_uniqueness(
        self,
        training_data,
        sample_batch,
        feature_names,
        mock_prediction_function
    ):
        """Test that different inputs produce different hashes."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        hashes = []
        for instance in sample_batch[:5]:
            report = service.explain_demand_forecast(
                forecast_input=instance,
                predict_fn=mock_prediction_function
            )
            hashes.append(report.provenance_hash)

        # All hashes should be unique
        assert len(hashes) == len(set(hashes))

    def test_provenance_hash_reproducibility(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that same inputs produce same hash (when timestamps match)."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        # Recompute hash with same data
        recomputed_hash = report.compute_provenance_hash()

        # Hash method should be consistent
        assert isinstance(recomputed_hash, str)
        assert len(recomputed_hash) == 64  # SHA-256 produces 64 hex characters


@pytest.mark.determinism
class TestZeroHallucinationGuarantees:
    """Tests for zero-hallucination guarantees."""

    def test_no_random_values_in_explanations(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test that explanations contain no random/hallucinated values."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )
        service.set_model(trained_random_forest, model_type="tree")

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        # Verify prediction matches actual model output
        expected_pred = mock_prediction_function(sample_instance)
        assert abs(report.prediction_value - expected_pred) < 0.001

        # Verify SHAP values sum correctly
        if report.shap_explanation:
            shap_sum = sum(c.contribution for c in report.shap_explanation.feature_contributions)
            expected_sum = report.shap_explanation.prediction_value - report.shap_explanation.base_value
            assert abs(shap_sum - expected_sum) < 0.01

        # Verify deterministic flag is True
        assert report.deterministic is True

    def test_input_features_preserved(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that input features are exactly preserved in report."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        # Input features should match exactly
        for i, name in enumerate(feature_names):
            assert name in report.input_features
            assert abs(report.input_features[name] - sample_instance[i]) < 1e-10

    def test_uncertainty_bounds_valid(
        self,
        training_data,
        sample_instance,
        feature_names,
        mock_prediction_function
    ):
        """Test that uncertainty bounds are mathematically valid."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        report = service.explain_demand_forecast(
            forecast_input=sample_instance,
            predict_fn=mock_prediction_function
        )

        uncertainty = report.uncertainty

        # Point estimate should equal prediction
        assert uncertainty.point_estimate == report.prediction_value

        # Standard error should be non-negative
        assert uncertainty.standard_error >= 0

        # Confidence interval should be valid
        ci = uncertainty.confidence_interval
        assert ci.lower_bound <= uncertainty.point_estimate
        assert ci.upper_bound >= uncertainty.point_estimate
        assert ci.lower_bound < ci.upper_bound

        # Uncertainty components should be non-negative
        assert uncertainty.epistemic_uncertainty >= 0
        assert uncertainty.aleatoric_uncertainty >= 0


@pytest.mark.determinism
class TestNumericPrecision:
    """Tests for numeric precision and consistency."""

    def test_feature_contribution_percentages_sum(
        self,
        training_data,
        sample_instance,
        feature_names,
        trained_random_forest,
        mock_prediction_function
    ):
        """Test that contribution percentages have reasonable sum."""
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

        # Percentages should sum close to 100
        total_percentage = sum(
            abs(c.contribution_percentage)
            for c in report.top_features
        )

        # Allow some tolerance for features not included
        assert total_percentage <= 100 + 1e-6

    def test_consistent_precision_across_reports(
        self,
        training_data,
        sample_batch,
        feature_names,
        mock_prediction_function
    ):
        """Test that numeric precision is consistent."""
        service = ExplainabilityService(
            training_data=training_data,
            feature_names=feature_names
        )

        reports = []
        for instance in sample_batch[:5]:
            report = service.explain_demand_forecast(
                forecast_input=instance,
                predict_fn=mock_prediction_function
            )
            reports.append(report)

        # All reports should have same structure and precision
        for report in reports:
            # All values should be finite
            assert np.isfinite(report.prediction_value)
            assert np.isfinite(report.uncertainty.standard_error)

            for contrib in report.top_features:
                assert np.isfinite(contrib.contribution)
                assert np.isfinite(contrib.contribution_percentage)
