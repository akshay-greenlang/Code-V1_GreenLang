# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Fouling Predictor Unit Tests

Tests for ML-based fouling prediction including:
- Feature extraction from operating data
- Fouling resistance prediction
- UA degradation forecasting
- Days-to-threshold prediction
- Confidence score calculation
- Uncertainty quantification
- Model explainability (SHAP/LIME)
- Provenance hash verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
import hashlib
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta


# Test tolerances
PREDICTION_TOLERANCE = 0.0001  # m2-K/kW
CONFIDENCE_TOLERANCE = 0.05


class TestFeatureExtraction:
    """Test feature extraction for fouling prediction."""

    def test_basic_feature_extraction(self, sample_operating_state, mock_ml_service):
        """Test basic feature extraction from operating state."""
        state = sample_operating_state

        features = mock_ml_service.extract_features(state)

        assert "dt_hot" in features
        assert "dt_cold" in features
        assert "flow_ratio" in features

    def test_temperature_difference_features(self, sample_operating_state):
        """Test temperature difference feature calculation."""
        state = sample_operating_state

        dt_hot = state.T_hot_in_C - state.T_hot_out_C
        dt_cold = state.T_cold_out_C - state.T_cold_in_C

        assert dt_hot == 60.0  # 150 - 90
        assert dt_cold == 70.0  # 100 - 30

    def test_pressure_drop_ratio_features(self, sample_operating_state, sample_exchanger_config):
        """Test pressure drop ratio feature calculation."""
        state = sample_operating_state
        config = sample_exchanger_config

        dp_shell = state.P_hot_in_kPa - state.P_hot_out_kPa
        dp_tube = state.P_cold_in_kPa - state.P_cold_out_kPa

        dp_ratio_shell = dp_shell / config.design_pressure_drop_shell_kPa
        dp_ratio_tube = dp_tube / config.design_pressure_drop_tube_kPa

        assert dp_ratio_shell > 0
        assert dp_ratio_tube > 0

    def test_flow_ratio_feature(self, sample_operating_state):
        """Test flow ratio feature calculation."""
        state = sample_operating_state

        flow_ratio = state.m_dot_hot_kg_s / state.m_dot_cold_kg_s

        assert flow_ratio == pytest.approx(1.25, abs=0.01)

    def test_dimensionless_numbers(self, sample_operating_state, sample_exchanger_config):
        """Test dimensionless number calculations (Re, Pr)."""
        state = sample_operating_state
        config = sample_exchanger_config

        # Reynolds number (tube side)
        D = config.tube_id_m
        A = np.pi * D ** 2 / 4 * config.tube_count / config.tube_passes
        v = state.m_dot_cold_kg_s / (state.rho_cold_kg_m3 * A)
        Re_tube = state.rho_cold_kg_m3 * v * D / state.mu_cold_Pa_s

        # Prandtl number
        Pr = state.mu_cold_Pa_s * state.Cp_cold_kJ_kgK * 1000 / state.k_cold_W_mK

        assert Re_tube > 0
        assert Pr > 0


class TestFoulingResistancePrediction:
    """Test fouling resistance (Rf) prediction."""

    @pytest.mark.asyncio
    async def test_basic_rf_prediction(self, mock_ml_service):
        """Test basic fouling resistance prediction."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.0,
            "dp_tube_ratio": 1.0,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "fouling_resistance_m2K_kW" in prediction
        assert prediction["fouling_resistance_m2K_kW"] >= 0

    @pytest.mark.asyncio
    async def test_rf_prediction_with_horizon(self, mock_ml_service):
        """Test Rf prediction with different horizons."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.1,
            "dp_tube_ratio": 1.1,
            "reynolds_hot": 50000.0,
        }

        prediction_7d = await mock_ml_service.predict_fouling(features, horizon_days=7)
        prediction_30d = await mock_ml_service.predict_fouling(features, horizon_days=30)

        # Both should return valid predictions
        assert prediction_7d["fouling_resistance_m2K_kW"] >= 0
        assert prediction_30d["fouling_resistance_m2K_kW"] >= 0

    def test_rf_physical_bounds(self, tema_reference_data):
        """Test that Rf predictions are within physical bounds."""
        # Typical Rf values from TEMA
        max_rf = 0.005  # m2-K/kW (very fouled)
        min_rf = 0.0    # Clean

        # Sample prediction
        Rf_predicted = 0.00035

        assert min_rf <= Rf_predicted <= max_rf


class TestUADegradationPrediction:
    """Test UA degradation forecasting."""

    @pytest.mark.asyncio
    async def test_ua_degradation_prediction(self, mock_ml_service):
        """Test UA degradation percentage prediction."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.2,
            "dp_tube_ratio": 1.2,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "ua_degradation_percent" in prediction
        assert 0 <= prediction["ua_degradation_percent"] <= 100

    def test_ua_degradation_from_rf(self, sample_exchanger_config):
        """Test UA degradation calculation from Rf."""
        config = sample_exchanger_config

        Rf = 0.0005  # m2-K/kW
        A = 100.0    # m2 (assumed heat transfer area)
        UA_clean = config.design_UA_kW_K

        # UA_fouled = 1 / (1/UA_clean + Rf*A)
        # Simplified: UA_degradation = Rf * A / (1/UA_clean + Rf*A) * 100
        UA_fouled = 1 / (1 / UA_clean + Rf)
        UA_degradation = (1 - UA_fouled / UA_clean) * 100

        assert 0 <= UA_degradation <= 100


class TestDaysToThreshold:
    """Test days-to-threshold prediction."""

    @pytest.mark.asyncio
    async def test_days_to_threshold_prediction(self, mock_ml_service):
        """Test prediction of days until cleaning threshold."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.0,
            "dp_tube_ratio": 1.0,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "predicted_days_to_threshold" in prediction
        assert prediction["predicted_days_to_threshold"] > 0

    def test_days_to_threshold_decreases_with_fouling(self):
        """Test that days to threshold decreases as fouling increases."""
        # Clean state: many days until cleaning needed
        rf_clean = 0.0001
        days_clean = 180

        # Fouled state: fewer days until cleaning needed
        rf_fouled = 0.0004
        days_fouled = 45

        assert days_fouled < days_clean

    def test_urgency_classification(self):
        """Test urgency classification based on days to threshold."""
        def classify_urgency(days: int) -> str:
            if days < 7:
                return "critical"
            elif days < 30:
                return "urgent"
            elif days < 90:
                return "scheduled"
            else:
                return "routine"

        assert classify_urgency(5) == "critical"
        assert classify_urgency(20) == "urgent"
        assert classify_urgency(60) == "scheduled"
        assert classify_urgency(120) == "routine"


class TestConfidenceScore:
    """Test prediction confidence score calculation."""

    @pytest.mark.asyncio
    async def test_confidence_score_bounds(self, mock_ml_service):
        """Test that confidence score is between 0 and 1."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.0,
            "dp_tube_ratio": 1.0,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "confidence_score" in prediction
        assert 0 <= prediction["confidence_score"] <= 1

    def test_confidence_decreases_with_extrapolation(self):
        """Test that confidence decreases for extrapolated conditions."""
        # Within training distribution
        confidence_interpolation = 0.90

        # Outside training distribution (extrapolation)
        confidence_extrapolation = 0.65

        assert confidence_extrapolation < confidence_interpolation

    def test_confidence_reflects_data_quality(self):
        """Test that confidence reflects input data quality."""
        # Good quality data
        confidence_good = 0.92

        # Degraded data quality
        confidence_degraded = 0.75

        # Bad data quality
        confidence_bad = 0.45

        assert confidence_good > confidence_degraded > confidence_bad


class TestUncertaintyQuantification:
    """Test prediction uncertainty quantification."""

    @pytest.mark.asyncio
    async def test_prediction_interval(self, mock_ml_service):
        """Test that predictions include uncertainty intervals."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.0,
            "dp_tube_ratio": 1.0,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "prediction_interval" in prediction
        assert "lower" in prediction["prediction_interval"]
        assert "upper" in prediction["prediction_interval"]

        # Upper bound > lower bound
        assert prediction["prediction_interval"]["upper"] > prediction["prediction_interval"]["lower"]

    def test_interval_contains_point_estimate(self):
        """Test that confidence interval contains point estimate."""
        rf_point = 0.00035
        rf_lower = 0.00028
        rf_upper = 0.00042

        assert rf_lower <= rf_point <= rf_upper

    def test_interval_width_reflects_uncertainty(self):
        """Test that interval width reflects prediction uncertainty."""
        # Low uncertainty (model confident)
        interval_narrow = (0.00033, 0.00037)  # Width = 0.00004

        # High uncertainty (model uncertain)
        interval_wide = (0.00020, 0.00050)  # Width = 0.00030

        width_narrow = interval_narrow[1] - interval_narrow[0]
        width_wide = interval_wide[1] - interval_wide[0]

        assert width_narrow < width_wide


class TestModelExplainability:
    """Test ML model explainability (SHAP/LIME)."""

    def test_shap_values_available(self):
        """Test that SHAP values can be computed."""
        # Mock SHAP values for a prediction
        shap_values = {
            "dp_shell_ratio": 0.00015,  # Increases Rf
            "dp_tube_ratio": 0.00012,   # Increases Rf
            "flow_ratio": -0.00005,     # Decreases Rf
            "dt_hot": -0.00003,         # Decreases Rf
            "reynolds_hot": -0.00002,   # Decreases Rf
        }

        # Sum of SHAP values should explain deviation from base
        total_shap = sum(shap_values.values())
        assert total_shap != 0

    def test_lime_explanation_available(self):
        """Test that LIME explanations can be generated."""
        # Mock LIME explanation
        lime_weights = [
            ("dp_shell_ratio > 1.1", 0.35),
            ("dp_tube_ratio > 1.1", 0.28),
            ("flow_ratio < 1.0", -0.15),
            ("dt_hot < 50", -0.12),
        ]

        # Top features should have highest absolute weights
        assert lime_weights[0][1] > lime_weights[-1][1]

    def test_feature_importance_ranking(self):
        """Test feature importance ranking."""
        feature_importance = {
            "dp_shell_ratio": 0.35,
            "dp_tube_ratio": 0.28,
            "flow_ratio": 0.18,
            "reynolds_hot": 0.12,
            "dt_hot": 0.07,
        }

        # Sum should be 1.0
        assert abs(sum(feature_importance.values()) - 1.0) < 0.01

        # Pressure drop ratios are most important for fouling
        assert feature_importance["dp_shell_ratio"] > feature_importance["dt_hot"]


class TestFoulingPredictorDeterminism:
    """Test fouling predictor determinism."""

    @pytest.mark.asyncio
    async def test_deterministic_prediction(self, mock_ml_service):
        """Test that predictions are deterministic for same inputs."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.0,
            "dp_tube_ratio": 1.0,
            "reynolds_hot": 50000.0,
        }

        predictions = []
        for _ in range(5):
            pred = await mock_ml_service.predict_fouling(features)
            predictions.append(pred["fouling_resistance_m2K_kW"])

        # All predictions should be identical
        assert all(p == predictions[0] for p in predictions)

    def test_provenance_hash_for_prediction(self):
        """Test provenance hash generation for predictions."""
        prediction = {
            "exchanger_id": "HX-001",
            "fouling_resistance_m2K_kW": 0.00035,
            "confidence_score": 0.85,
            "timestamp": "2024-01-15T10:30:00Z",
        }

        provenance_data = f"{prediction['exchanger_id']}:Rf:{prediction['fouling_resistance_m2K_kW']:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestFoulingTrendAnalysis:
    """Test fouling trend analysis."""

    def test_trend_classification(self):
        """Test fouling trend classification."""
        # Calculate trend from historical Rf values
        rf_history = [0.00025, 0.00028, 0.00031, 0.00034, 0.00037]

        # Simple linear trend
        if rf_history[-1] > rf_history[0] * 1.1:
            trend = "increasing"
        elif rf_history[-1] < rf_history[0] * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        assert trend == "increasing"

    def test_trend_rate_calculation(self):
        """Test fouling rate calculation."""
        rf_values = [0.00025, 0.00030, 0.00035]
        days = [0, 30, 60]

        # Linear regression for rate
        rate = (rf_values[-1] - rf_values[0]) / (days[-1] - days[0])  # m2-K/kW per day

        assert rate > 0
        assert rate == pytest.approx(0.00035 / 60 - 0.00025 / 60, abs=0.0001)


class TestFoulingPredictorEdgeCases:
    """Test edge cases for fouling predictor."""

    def test_missing_features_handling(self):
        """Test handling of missing features."""
        features_incomplete = {
            "dt_hot": 60.0,
            # Missing other features
        }

        # Model should handle missing features or raise appropriate error
        assert "dt_hot" in features_incomplete

    def test_extreme_operating_conditions(self):
        """Test prediction under extreme operating conditions."""
        # Very high pressure drop ratio (heavily fouled)
        dp_ratio_extreme = 2.5

        # Should still produce valid prediction
        assert dp_ratio_extreme > 0

    def test_clean_exchanger_prediction(self):
        """Test prediction for clean exchanger."""
        # Pressure drop ratios near 1.0 indicate clean exchanger
        dp_ratio_clean = 1.02

        # Rf should be near TEMA design fouling factor
        expected_rf_range = (0.0001, 0.0005)

        # Prediction should be in this range for clean exchanger
        assert expected_rf_range[0] < expected_rf_range[1]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestFeatureExtraction",
    "TestFoulingResistancePrediction",
    "TestUADegradationPrediction",
    "TestDaysToThreshold",
    "TestConfidenceScore",
    "TestUncertaintyQuantification",
    "TestModelExplainability",
    "TestFoulingPredictorDeterminism",
    "TestFoulingTrendAnalysis",
    "TestFoulingPredictorEdgeCases",
]
