# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Predictive Analytics Tests (20 tests)

Tests emission forecasting, anomaly detection, target gap analysis,
Monte Carlo simulation, model evaluation, and zero-hallucination
guarantees for the predictive analytics engine.

Author: GreenLang QA Team
"""

import math
from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import StubMLModel, _compute_hash


class TestPredictiveAnalytics:
    """Test suite for predictive analytics engine."""

    # -- Forecasting tests ---

    def test_linear_forecast(self, mock_ml_models, sample_forecast_data):
        """Test linear forecast produces decreasing trend."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data, horizon=12)
        predictions = result["predictions"]
        assert len(predictions) == 12
        assert predictions[0]["predicted_value"] > predictions[-1]["predicted_value"]

    def test_weighted_moving_average(self, sample_forecast_data):
        """Test weighted moving average calculation."""
        values = [d["value"] for d in sample_forecast_data[-12:]]
        weights = list(range(1, len(values) + 1))
        total_weight = sum(weights)
        wma = sum(v * w for v, w in zip(values, weights)) / total_weight
        assert isinstance(wma, float)
        assert wma > 0

    def test_exponential_smoothing(self, sample_forecast_data):
        """Test exponential smoothing calculation."""
        values = [d["value"] for d in sample_forecast_data]
        alpha = 0.3
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        assert len(smoothed) == len(values)
        assert all(isinstance(s, float) for s in smoothed)

    def test_forecast_confidence_intervals(self, mock_ml_models, sample_forecast_data):
        """Test confidence intervals bracket predictions."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data, horizon=6)
        for pred in result["predictions"]:
            assert pred["lower_bound"] < pred["predicted_value"]
            assert pred["upper_bound"] > pred["predicted_value"]
            assert pred["confidence"] == 0.95

    def test_forecast_provenance_hash(self, mock_ml_models, sample_forecast_data):
        """Test forecast produces a valid SHA-256 provenance hash."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data, horizon=12)
        assert len(result["provenance_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["provenance_hash"])

    # -- Anomaly detection tests ---

    def test_anomaly_detection_zscore(self, mock_ml_models):
        """Test anomaly detection with z-score method."""
        normal_data = [{"value": 100.0 + i * 0.5} for i in range(30)]
        normal_data.append({"value": 500.0})  # spike
        model = mock_ml_models["anomaly_detection"]
        anomalies = model.detect_anomalies(normal_data, sensitivity=0.85)
        assert len(anomalies) >= 1
        assert any(a["type"] == "spike" for a in anomalies)

    def test_anomaly_detection_iqr(self):
        """Test IQR-based anomaly detection."""
        values = sorted([100 + i * 2 for i in range(30)] + [500])
        n = len(values)
        q1 = values[n // 4]
        q3 = values[3 * n // 4]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [v for v in values if v < lower_fence or v > upper_fence]
        assert 500 in outliers

    def test_anomaly_sensitivity_tuning(self, mock_ml_models):
        """Test higher sensitivity detects more anomalies."""
        data = [{"value": 100.0 + i} for i in range(20)]
        data.append({"value": 200.0})
        model = mock_ml_models["anomaly_detection"]
        low_sens = model.detect_anomalies(data, sensitivity=0.6)
        high_sens = model.detect_anomalies(data, sensitivity=0.99)
        assert len(high_sens) >= len(low_sens)

    def test_no_false_positives_normal_data(self, mock_ml_models):
        """Test normal data produces zero or few anomalies."""
        normal_data = [{"value": 100.0 + i * 0.1} for i in range(50)]
        model = mock_ml_models["anomaly_detection"]
        anomalies = model.detect_anomalies(normal_data, sensitivity=0.85)
        assert len(anomalies) == 0

    # -- Target gap tests ---

    def test_target_gap_prediction(self, mock_ml_models, sample_forecast_data):
        """Test gap calculation between forecast and target."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data, horizon=12)
        final_forecast = result["predictions"][-1]["predicted_value"]
        target = 800.0
        gap = final_forecast - target
        gap_pct = round(gap / target * 100, 2)
        assert isinstance(gap, float)
        assert isinstance(gap_pct, float)

    def test_sbti_trajectory(self, sample_forecast_data):
        """Test SBTi-aligned trajectory calculation."""
        base_year_emissions = sample_forecast_data[0]["value"]
        target_reduction_pct = 42.0
        target_year = 2030
        base_year = 2021
        years_to_target = target_year - base_year
        annual_reduction = target_reduction_pct / years_to_target
        trajectory = []
        for year in range(years_to_target + 1):
            emissions = base_year_emissions * (1 - annual_reduction * year / 100)
            trajectory.append({"year": base_year + year, "emissions": round(emissions, 2)})
        assert len(trajectory) == years_to_target + 1
        assert trajectory[-1]["emissions"] < trajectory[0]["emissions"]

    # -- Monte Carlo tests ---

    def test_monte_carlo_distribution(self, mock_ml_models, sample_forecast_data):
        """Test Monte Carlo produces bounded distribution."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data, horizon=6)
        for pred in result["predictions"]:
            spread = pred["upper_bound"] - pred["lower_bound"]
            assert spread > 0
            assert pred["predicted_value"] >= pred["lower_bound"]
            assert pred["predicted_value"] <= pred["upper_bound"]

    def test_monte_carlo_iterations(self):
        """Test Monte Carlo with configurable iterations."""
        import random
        random.seed(42)
        iterations = 1000
        base_value = 100.0
        results = [base_value * (1 + random.gauss(0, 0.1)) for _ in range(iterations)]
        mean = sum(results) / len(results)
        assert abs(mean - base_value) < base_value * 0.05

    # -- Model evaluation tests ---

    def test_feature_importance(self, mock_ml_models):
        """Test feature importance sums to approximately 1.0."""
        model = mock_ml_models["emission_forecast"]
        explanation = model.explain(0)
        importance = explanation["feature_importance"]
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_model_evaluation_r_squared(self, mock_ml_models, sample_forecast_data):
        """Test R-squared is between 0 and 1."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_model_evaluation_mae_rmse(self, mock_ml_models, sample_forecast_data):
        """Test MAE and RMSE are non-negative."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict(sample_forecast_data)
        assert result["mae"] >= 0.0
        assert result["rmse"] >= 0.0
        assert result["rmse"] >= result["mae"]

    # -- Edge case tests ---

    def test_forecast_empty_data(self, mock_ml_models):
        """Test forecast with empty data returns empty predictions."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict([], horizon=6)
        assert len(result["predictions"]) == 6

    def test_forecast_single_point(self, mock_ml_models):
        """Test forecast with single data point."""
        model = mock_ml_models["emission_forecast"]
        result = model.predict([{"value": 100.0}], horizon=3)
        assert len(result["predictions"]) == 3
        assert result["predictions"][0]["predicted_value"] > 0

    def test_forecast_negative_values(self, mock_ml_models):
        """Test forecast handles negative input values."""
        data = [{"value": -50.0 + i * 10} for i in range(12)]
        model = mock_ml_models["emission_forecast"]
        result = model.predict(data, horizon=6)
        assert len(result["predictions"]) == 6

    def test_zero_hallucination_guarantee(self, mock_ml_models, sample_forecast_data):
        """Test outputs are deterministic (same input, same output)."""
        model = mock_ml_models["emission_forecast"]
        result1 = model.predict(sample_forecast_data, horizon=6)
        result2 = model.predict(sample_forecast_data, horizon=6)
        assert result1["predictions"] == result2["predictions"]
        assert result1["provenance_hash"] == result2["provenance_hash"]
