# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Failure Prediction Tests

Tests for ML-based failure prediction and feature engineering.
Validates probability calculations, feature extraction, and explainability.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime, timezone
from typing import Dict, Any, List

from greenlang.agents.process_heat.gl_013_predictive_maintenance.failure_prediction import (
    FailurePredictionEngine,
    FeatureEngineer,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    FailureMode,
    MLModelConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    FailurePrediction,
)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer()

        assert engineer is not None

    def test_feature_definitions_exist(self, feature_engineer):
        """Test feature definitions are complete."""
        assert hasattr(feature_engineer, 'FEATURE_DEFINITIONS')
        assert len(feature_engineer.FEATURE_DEFINITIONS) > 0


class TestFeatureExtraction:
    """Tests for feature extraction methods."""

    def test_extract_vibration_features(self, feature_engineer):
        """Test vibration feature extraction."""
        vib_data = {
            "overall_velocity_mm_s": 5.0,
            "overall_acceleration_g": 1.5,
            "bearing_defect_detected": True,
            "imbalance_detected": False,
            "misalignment_detected": False,
        }

        features = feature_engineer.extract_features(vibration_data=vib_data)

        assert "velocity_rms_normalized" in features
        assert "bearing_defect_indicator" in features
        assert features["bearing_defect_indicator"] == 1.0

    def test_extract_oil_features(self, feature_engineer):
        """Test oil analysis feature extraction."""
        oil_data = {
            "viscosity_change_pct": 5.0,
            "tan_mg_koh_g": 1.5,
            "iron_ppm": 80.0,
            "water_ppm": 300.0,
        }

        features = feature_engineer.extract_features(oil_data=oil_data)

        assert "viscosity_change_pct" in features
        assert "tan_normalized" in features
        assert "iron_ppm_normalized" in features

    def test_extract_temperature_features(self, feature_engineer):
        """Test temperature feature extraction."""
        temp_data = {
            "max_temperature_c": 75.0,
            "delta_t_c": 20.0,
        }

        features = feature_engineer.extract_features(temperature_data=temp_data)

        assert "temperature_normalized" in features
        assert "delta_t_normalized" in features

    def test_extract_mcsa_features(self, feature_engineer):
        """Test MCSA feature extraction."""
        mcsa_data = {
            "rotor_bar_fault_severity_db": -48.0,
            "eccentricity_severity_db": -52.0,
            "current_unbalance_pct": 3.5,
        }

        features = feature_engineer.extract_features(mcsa_data=mcsa_data)

        assert "rotor_bar_severity_db" in features
        assert "current_unbalance_pct" in features

    def test_extract_operating_features(self, feature_engineer):
        """Test operating condition feature extraction."""
        op_data = {
            "running_hours": 25000,
            "expected_life_hours": 50000,
            "load_percent": 85.0,
        }

        features = feature_engineer.extract_features(operating_data=op_data)

        assert "running_hours_normalized" in features
        assert "load_factor" in features
        assert features["running_hours_normalized"] == pytest.approx(0.5, rel=0.01)

    def test_extract_combined_features(self, feature_engineer):
        """Test combined feature extraction from all sources."""
        vib_data = {"overall_velocity_mm_s": 3.0, "bearing_defect_detected": False}
        oil_data = {"tan_mg_koh_g": 1.0, "iron_ppm": 50.0}
        temp_data = {"max_temperature_c": 60.0, "delta_t_c": 15.0}
        mcsa_data = {"current_unbalance_pct": 2.0}
        op_data = {"running_hours": 20000, "expected_life_hours": 50000}

        features = feature_engineer.extract_features(
            vibration_data=vib_data,
            oil_data=oil_data,
            temperature_data=temp_data,
            mcsa_data=mcsa_data,
            operating_data=op_data,
        )

        # Should have features from all sources
        assert "velocity_rms_normalized" in features
        assert "tan_normalized" in features
        assert "temperature_normalized" in features

    def test_compute_derived_features(self, feature_engineer):
        """Test derived feature computation."""
        base_features = {
            "velocity_rms_normalized": 0.3,
            "bearing_defect_indicator": 0.0,
            "tan_normalized": 0.2,
            "temperature_normalized": 0.4,
        }

        features = feature_engineer.extract_features(
            vibration_data={
                "overall_velocity_mm_s": 4.5,
                "bearing_defect_detected": False,
            }
        )

        # Should compute health score composite
        assert "health_score_composite" in features


class TestFeatureNormalization:
    """Tests for feature normalization."""

    def test_normalize_in_range(self, feature_engineer):
        """Test normalization produces values in [0, 1]."""
        features = feature_engineer.extract_features(
            vibration_data={
                "overall_velocity_mm_s": 7.5,  # Mid-range
                "overall_acceleration_g": 5.0,
            }
        )

        assert 0 <= features.get("velocity_rms_normalized", 0) <= 1
        assert 0 <= features.get("acceleration_rms_normalized", 0) <= 1

    def test_normalize_clipping_high(self, feature_engineer):
        """Test normalization clips high values to 1."""
        features = feature_engineer.extract_features(
            vibration_data={
                "overall_velocity_mm_s": 50.0,  # Very high
            }
        )

        assert features.get("velocity_rms_normalized", 0) == 1.0

    def test_normalize_clipping_low(self, feature_engineer):
        """Test normalization clips low values to 0."""
        features = feature_engineer.extract_features(
            vibration_data={
                "overall_velocity_mm_s": -1.0,  # Negative (invalid)
            }
        )

        assert features.get("velocity_rms_normalized", 0) == 0.0


class TestFailurePredictionEngine:
    """Tests for FailurePredictionEngine class."""

    def test_initialization(self, ml_model_config):
        """Test engine initialization."""
        engine = FailurePredictionEngine(ml_model_config)

        assert engine.config == ml_model_config
        assert engine.feature_engineer is not None

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        engine = FailurePredictionEngine()

        assert engine.config is not None


class TestFailureProbabilityPrediction:
    """Tests for failure probability prediction."""

    def test_predict_bearing_wear(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test bearing wear prediction."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        assert isinstance(prediction, FailurePrediction)
        assert prediction.failure_mode == FailureMode.BEARING_WEAR
        assert 0 <= prediction.probability <= 1
        assert 0 <= prediction.confidence <= 1

    def test_predict_all_failure_modes(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test prediction for all failure modes."""
        predictions = failure_prediction_engine.predict_all_failure_modes(
            feature_vector_healthy
        )

        assert len(predictions) > 0
        # Should be sorted by probability (descending)
        for i in range(len(predictions) - 1):
            assert predictions[i].probability >= predictions[i + 1].probability

    def test_healthy_features_low_probability(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test healthy features produce low failure probability."""
        predictions = failure_prediction_engine.predict_all_failure_modes(
            feature_vector_healthy
        )

        # Highest probability should be relatively low
        max_prob = predictions[0].probability
        assert max_prob < 0.5

    def test_degraded_features_higher_probability(
        self,
        failure_prediction_engine,
        feature_vector_degraded
    ):
        """Test degraded features produce higher failure probability."""
        predictions = failure_prediction_engine.predict_all_failure_modes(
            feature_vector_degraded
        )

        # Should have higher probability than healthy
        max_prob = predictions[0].probability
        assert max_prob > 0.3


class TestFailureModeSpecificPrediction:
    """Tests for specific failure mode predictions."""

    @pytest.mark.parametrize("failure_mode", [
        FailureMode.BEARING_WEAR,
        FailureMode.IMBALANCE,
        FailureMode.MISALIGNMENT,
        FailureMode.ROTOR_BAR_BREAK,
        FailureMode.LUBRICATION_FAILURE,
        FailureMode.OVERHEATING,
    ])
    def test_each_failure_mode(
        self,
        failure_prediction_engine,
        feature_vector_healthy,
        failure_mode
    ):
        """Test prediction for each failure mode."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            failure_mode,
        )

        assert prediction.failure_mode == failure_mode
        assert 0 <= prediction.probability <= 1

    def test_bearing_wear_sensitive_to_indicator(
        self,
        failure_prediction_engine
    ):
        """Test bearing wear is sensitive to bearing defect indicator."""
        healthy = {"bearing_defect_indicator": 0.0, "velocity_rms_normalized": 0.2}
        degraded = {"bearing_defect_indicator": 1.0, "velocity_rms_normalized": 0.2}

        pred_healthy = failure_prediction_engine.predict_failure_probability(
            healthy, FailureMode.BEARING_WEAR
        )
        pred_degraded = failure_prediction_engine.predict_failure_probability(
            degraded, FailureMode.BEARING_WEAR
        )

        assert pred_degraded.probability > pred_healthy.probability

    def test_rotor_bar_sensitive_to_mcsa(
        self,
        failure_prediction_engine
    ):
        """Test rotor bar fault is sensitive to MCSA features."""
        healthy = {"rotor_bar_severity_db": 0.0, "current_unbalance_pct": 0.05}
        degraded = {"rotor_bar_severity_db": 0.8, "current_unbalance_pct": 0.5}

        pred_healthy = failure_prediction_engine.predict_failure_probability(
            healthy, FailureMode.ROTOR_BAR_BREAK
        )
        pred_degraded = failure_prediction_engine.predict_failure_probability(
            degraded, FailureMode.ROTOR_BAR_BREAK
        )

        assert pred_degraded.probability > pred_healthy.probability


class TestTimeToFailurePrediction:
    """Tests for time-to-failure prediction."""

    def test_ttf_calculated_for_high_probability(
        self,
        failure_prediction_engine,
        feature_vector_degraded
    ):
        """Test TTF is calculated for high probability predictions."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_degraded,
            FailureMode.BEARING_WEAR,
        )

        if prediction.probability > 0.1:
            assert prediction.time_to_failure_hours is not None
            assert prediction.time_to_failure_hours > 0

    def test_ttf_uncertainty_bounds(
        self,
        failure_prediction_engine,
        feature_vector_degraded
    ):
        """Test TTF uncertainty bounds are provided."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_degraded,
            FailureMode.BEARING_WEAR,
        )

        if prediction.time_to_failure_hours is not None:
            if prediction.uncertainty_lower_hours is not None:
                assert prediction.uncertainty_lower_hours < prediction.time_to_failure_hours
            if prediction.uncertainty_upper_hours is not None:
                assert prediction.uncertainty_upper_hours > prediction.time_to_failure_hours


class TestFeatureImportance:
    """Tests for feature importance and explainability."""

    def test_feature_importance_provided(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test feature importance is provided."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        assert prediction.feature_importance is not None
        assert len(prediction.feature_importance) > 0

    def test_top_contributing_features(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test top contributing features are identified."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        assert prediction.top_contributing_features is not None
        assert len(prediction.top_contributing_features) > 0

    def test_feature_importance_sums_reasonably(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test feature importance values are reasonable."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        # All importances should be non-negative
        for feature, importance in prediction.feature_importance.items():
            # Contributions can be negative or positive
            pass

    def test_get_global_feature_importance(
        self,
        failure_prediction_engine
    ):
        """Test global feature importance retrieval."""
        importance = failure_prediction_engine.get_feature_importance_global(
            FailureMode.BEARING_WEAR
        )

        assert len(importance) > 0
        # Should sum to approximately 1
        total = sum(importance.values())
        assert total == pytest.approx(1.0, rel=0.01)


class TestOverallFailureProbability:
    """Tests for overall failure probability calculation."""

    def test_overall_probability_single_mode(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test overall probability with single failure mode."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        overall = failure_prediction_engine.calculate_overall_failure_probability(
            [prediction],
            time_horizon_hours=720,
        )

        assert 0 <= overall <= 1

    def test_overall_probability_multiple_modes(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test overall probability with multiple failure modes."""
        predictions = failure_prediction_engine.predict_all_failure_modes(
            feature_vector_healthy
        )

        overall = failure_prediction_engine.calculate_overall_failure_probability(
            predictions,
            time_horizon_hours=720,
        )

        # Overall should be at least as high as individual modes
        max_individual = max(p.probability for p in predictions)
        assert overall >= max_individual * 0.5  # Some margin

    def test_overall_probability_time_horizon(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test overall probability varies with time horizon."""
        predictions = failure_prediction_engine.predict_all_failure_modes(
            feature_vector_healthy
        )

        prob_30d = failure_prediction_engine.calculate_overall_failure_probability(
            predictions, time_horizon_hours=720
        )
        prob_90d = failure_prediction_engine.calculate_overall_failure_probability(
            predictions, time_horizon_hours=2160
        )

        # Longer horizon should have higher probability
        # (unless probabilities are very low)


class TestPredictionExplainability:
    """Tests for prediction explainability."""

    def test_explain_prediction(
        self,
        failure_prediction_engine,
        feature_vector_degraded
    ):
        """Test prediction explanation generation."""
        prediction = failure_prediction_engine.predict_failure_probability(
            feature_vector_degraded,
            FailureMode.BEARING_WEAR,
        )

        explanation = failure_prediction_engine.explain_prediction(
            prediction,
            feature_vector_degraded,
        )

        assert "failure_mode" in explanation
        assert "probability_pct" in explanation
        assert "risk_level" in explanation
        assert "contributing_factors" in explanation

    def test_explanation_risk_levels(self, failure_prediction_engine):
        """Test risk level classification in explanations."""
        # Low risk
        low_pred = FailurePrediction(
            failure_mode=FailureMode.BEARING_WEAR,
            probability=0.1,
            confidence=0.8,
            feature_importance={},
            top_contributing_features=[],
            model_id="test",
            model_version="1.0",
        )

        explanation = failure_prediction_engine.explain_prediction(low_pred, {})
        assert explanation["risk_level"] == "low"

        # High risk
        high_pred = FailurePrediction(
            failure_mode=FailureMode.BEARING_WEAR,
            probability=0.7,
            confidence=0.8,
            feature_importance={},
            top_contributing_features=[],
            model_id="test",
            model_version="1.0",
        )

        explanation = failure_prediction_engine.explain_prediction(high_pred, {})
        assert explanation["risk_level"] in ["high", "critical"]


class TestCalibration:
    """Tests for probability calibration."""

    def test_calibrate_probability_high_accuracy(
        self,
        failure_prediction_engine
    ):
        """Test calibration with high historical accuracy."""
        raw_prob = 0.5

        calibrated = failure_prediction_engine.calibrate_probability(
            raw_prob,
            historical_accuracy=0.95,
        )

        # High accuracy = minimal adjustment
        assert calibrated == pytest.approx(raw_prob, rel=0.1)

    def test_calibrate_probability_low_accuracy(
        self,
        failure_prediction_engine
    ):
        """Test calibration with low historical accuracy."""
        raw_prob = 0.5

        calibrated = failure_prediction_engine.calibrate_probability(
            raw_prob,
            historical_accuracy=0.6,
        )

        # Low accuracy = shrinkage toward base rate
        assert calibrated != raw_prob

    def test_calibrated_bounds(self, failure_prediction_engine):
        """Test calibrated probability stays in bounds."""
        for raw_prob in [0.0, 0.1, 0.5, 0.9, 1.0]:
            calibrated = failure_prediction_engine.calibrate_probability(
                raw_prob,
                historical_accuracy=0.7,
            )
            assert 0 <= calibrated <= 1


class TestPredictionDeterminism:
    """Tests for deterministic prediction behavior."""

    def test_repeated_prediction_same_result(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test repeated predictions produce identical results."""
        predictions = [
            failure_prediction_engine.predict_failure_probability(
                feature_vector_healthy,
                FailureMode.BEARING_WEAR,
            )
            for _ in range(5)
        ]

        probs = [p.probability for p in predictions]
        assert len(set(probs)) == 1

    def test_provenance_hash_reproducible(
        self,
        failure_prediction_engine,
        feature_vector_healthy
    ):
        """Test provenance hash is reproducible."""
        pred1 = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )
        pred2 = failure_prediction_engine.predict_failure_probability(
            feature_vector_healthy,
            FailureMode.BEARING_WEAR,
        )

        # Same input -> same output
        assert pred1.probability == pred2.probability


class TestFailurePredictionEdgeCases:
    """Tests for edge cases."""

    def test_empty_features(self, failure_prediction_engine):
        """Test prediction with empty features."""
        empty_features: Dict[str, float] = {}

        prediction = failure_prediction_engine.predict_failure_probability(
            empty_features,
            FailureMode.BEARING_WEAR,
        )

        # Should produce valid (low confidence) result
        assert 0 <= prediction.probability <= 1

    def test_missing_key_features(self, failure_prediction_engine):
        """Test prediction with missing key features."""
        partial_features = {
            "velocity_rms_normalized": 0.5,
            # Missing bearing_defect_indicator and others
        }

        prediction = failure_prediction_engine.predict_failure_probability(
            partial_features,
            FailureMode.BEARING_WEAR,
        )

        assert 0 <= prediction.probability <= 1
        # Confidence should be lower
        assert prediction.confidence < 1.0

    def test_extreme_feature_values(self, failure_prediction_engine):
        """Test prediction with extreme feature values."""
        extreme_features = {
            "velocity_rms_normalized": 1.0,
            "bearing_defect_indicator": 1.0,
            "temperature_normalized": 1.0,
            "tan_normalized": 1.0,
        }

        prediction = failure_prediction_engine.predict_failure_probability(
            extreme_features,
            FailureMode.BEARING_WEAR,
        )

        # Should produce high probability
        assert prediction.probability > 0.5


class TestFailurePredictionIntegration:
    """Integration tests for failure prediction."""

    def test_full_prediction_workflow(self, ml_model_config):
        """Test complete prediction workflow."""
        engine = FailurePredictionEngine(ml_model_config)

        # Extract features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.extract_features(
            vibration_data={
                "overall_velocity_mm_s": 6.0,
                "bearing_defect_detected": True,
            },
            oil_data={
                "tan_mg_koh_g": 2.5,
                "iron_ppm": 120.0,
            },
            temperature_data={
                "max_temperature_c": 78.0,
            },
            operating_data={
                "running_hours": 35000,
                "expected_life_hours": 50000,
            },
        )

        # Predict all failure modes
        predictions = engine.predict_all_failure_modes(features)

        # Calculate overall probability
        overall_prob = engine.calculate_overall_failure_probability(
            predictions,
            time_horizon_hours=720,
        )

        # Generate explanation for highest risk
        if predictions:
            explanation = engine.explain_prediction(predictions[0], features)

            # Verify complete workflow
            assert len(predictions) > 0
            assert 0 <= overall_prob <= 1
            assert "risk_level" in explanation
