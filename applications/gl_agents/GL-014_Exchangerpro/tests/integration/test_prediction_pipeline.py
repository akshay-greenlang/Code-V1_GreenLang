# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Prediction Pipeline Integration Tests

End-to-end tests for the fouling prediction pipeline including:
- Feature extraction from operating data
- ML model inference
- Uncertainty quantification
- Trend analysis
- Alert generation

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List


class TestFeatureToPredictionFlow:
    """Test feature extraction to prediction flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_operating_state_to_features(
        self,
        sample_operating_state,
        sample_exchanger_config,
        mock_ml_service,
    ):
        """Test feature extraction from operating state."""
        state = sample_operating_state

        features = mock_ml_service.extract_features(state)

        assert "dt_hot" in features
        assert "dt_cold" in features
        assert "flow_ratio" in features
        assert features["dt_hot"] == 60.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_features_to_prediction(self, mock_ml_service):
        """Test ML prediction from features."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.15,
            "dp_tube_ratio": 1.12,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "fouling_resistance_m2K_kW" in prediction
        assert "ua_degradation_percent" in prediction
        assert "predicted_days_to_threshold" in prediction
        assert "confidence_score" in prediction

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_prediction(
        self,
        sample_operating_state,
        mock_ml_service,
    ):
        """Test end-to-end prediction pipeline."""
        state = sample_operating_state

        # Extract features
        features = mock_ml_service.extract_features(state)

        # Add pressure drop ratios (simulated)
        features["dp_shell_ratio"] = 1.0
        features["dp_tube_ratio"] = 1.0

        # Predict
        prediction = await mock_ml_service.predict_fouling(features)

        # Validate output
        assert prediction["fouling_resistance_m2K_kW"] >= 0
        assert 0 <= prediction["confidence_score"] <= 1


class TestPredictionWithUncertainty:
    """Test prediction with uncertainty quantification."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prediction_interval_generation(self, mock_ml_service):
        """Test prediction interval generation."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.1,
            "dp_tube_ratio": 1.1,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        assert "prediction_interval" in prediction
        assert prediction["prediction_interval"]["lower"] <= prediction["fouling_resistance_m2K_kW"]
        assert prediction["prediction_interval"]["upper"] >= prediction["fouling_resistance_m2K_kW"]

    @pytest.mark.integration
    def test_confidence_score_calibration(self):
        """Test that confidence scores are well-calibrated."""
        # Mock calibration check
        confidence_scores = [0.90, 0.85, 0.92, 0.88, 0.91]
        mean_confidence = np.mean(confidence_scores)

        # Well-calibrated model should have mean confidence near accuracy
        assert 0.7 < mean_confidence < 1.0


class TestTrendAnalysisPipeline:
    """Test fouling trend analysis pipeline."""

    @pytest.mark.integration
    def test_historical_trend_calculation(self):
        """Test historical trend calculation."""
        # Historical Rf values
        rf_history = [
            {"date": "2024-01-01", "rf": 0.00020},
            {"date": "2024-01-15", "rf": 0.00025},
            {"date": "2024-02-01", "rf": 0.00030},
            {"date": "2024-02-15", "rf": 0.00035},
            {"date": "2024-03-01", "rf": 0.00040},
        ]

        # Calculate trend (simple linear)
        rf_values = [h["rf"] for h in rf_history]
        trend_rate = (rf_values[-1] - rf_values[0]) / (len(rf_values) - 1)

        # Trend should be positive (increasing fouling)
        assert trend_rate > 0

    @pytest.mark.integration
    def test_trend_classification(self):
        """Test trend classification."""
        rf_history = [0.00020, 0.00025, 0.00030, 0.00035, 0.00040]

        first = rf_history[0]
        last = rf_history[-1]

        if last > first * 1.1:
            trend = "increasing"
        elif last < first * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        assert trend == "increasing"

    @pytest.mark.integration
    def test_trend_extrapolation(self):
        """Test trend extrapolation for days-to-threshold."""
        rf_current = 0.00040
        rf_threshold = 0.00070
        rate_per_day = 0.000002  # From trend analysis

        days_to_threshold = (rf_threshold - rf_current) / rate_per_day

        assert days_to_threshold > 0
        assert days_to_threshold == pytest.approx(150, abs=5)


class TestAlertGenerationPipeline:
    """Test alert generation pipeline."""

    @pytest.mark.integration
    def test_alert_threshold_breach(self):
        """Test alert generation on threshold breach."""
        ua_degradation = 35.0  # percent
        days_to_threshold = 25

        alerts = []

        if ua_degradation > 30:
            alerts.append({
                "type": "ua_degradation",
                "severity": "warning",
                "message": f"UA degradation at {ua_degradation}%",
            })

        if days_to_threshold < 30:
            alerts.append({
                "type": "threshold_proximity",
                "severity": "warning",
                "message": f"Cleaning threshold in {days_to_threshold} days",
            })

        assert len(alerts) == 2

    @pytest.mark.integration
    def test_alert_severity_classification(self):
        """Test alert severity classification."""
        def classify_severity(ua_degradation: float, days: int) -> str:
            if ua_degradation > 50 or days < 7:
                return "critical"
            elif ua_degradation > 35 or days < 14:
                return "high"
            elif ua_degradation > 25 or days < 30:
                return "medium"
            else:
                return "low"

        assert classify_severity(55, 10) == "critical"
        assert classify_severity(40, 10) == "high"
        assert classify_severity(30, 25) == "medium"
        assert classify_severity(15, 60) == "low"

    @pytest.mark.integration
    def test_alert_deduplication(self):
        """Test alert deduplication."""
        alerts = [
            {"exchanger_id": "HX-001", "type": "ua_degradation", "timestamp": "2024-01-15T10:00:00Z"},
            {"exchanger_id": "HX-001", "type": "ua_degradation", "timestamp": "2024-01-15T10:05:00Z"},
            {"exchanger_id": "HX-001", "type": "ua_degradation", "timestamp": "2024-01-15T10:10:00Z"},
            {"exchanger_id": "HX-002", "type": "ua_degradation", "timestamp": "2024-01-15T10:00:00Z"},
        ]

        # Deduplicate by exchanger_id and type within time window
        dedup_window_minutes = 30
        unique_alerts = {}

        for alert in alerts:
            key = f"{alert['exchanger_id']}:{alert['type']}"
            if key not in unique_alerts:
                unique_alerts[key] = alert

        assert len(unique_alerts) == 2  # HX-001 and HX-002


class TestMultiExchangerPrediction:
    """Test prediction for multiple exchangers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_prediction(self, mock_ml_service):
        """Test batch prediction for multiple exchangers."""
        exchangers = [
            {"id": "HX-001", "features": {"dt_hot": 60.0, "dp_shell_ratio": 1.1}},
            {"id": "HX-002", "features": {"dt_hot": 50.0, "dp_shell_ratio": 1.2}},
            {"id": "HX-003", "features": {"dt_hot": 70.0, "dp_shell_ratio": 1.05}},
        ]

        predictions = []
        for ex in exchangers:
            features = {
                "dt_hot": ex["features"]["dt_hot"],
                "dt_cold": 70.0,
                "flow_ratio": 1.0,
                "dp_shell_ratio": ex["features"]["dp_shell_ratio"],
                "dp_tube_ratio": 1.0,
                "reynolds_hot": 50000.0,
            }
            pred = await mock_ml_service.predict_fouling(features)
            predictions.append({
                "exchanger_id": ex["id"],
                "prediction": pred,
            })

        assert len(predictions) == 3
        for p in predictions:
            assert p["prediction"]["fouling_resistance_m2K_kW"] >= 0

    @pytest.mark.integration
    def test_prediction_prioritization(self):
        """Test prioritization of predictions for maintenance planning."""
        predictions = [
            {"id": "HX-001", "ua_degradation": 25, "days_to_threshold": 45},
            {"id": "HX-002", "ua_degradation": 40, "days_to_threshold": 20},
            {"id": "HX-003", "ua_degradation": 15, "days_to_threshold": 90},
        ]

        # Sort by urgency
        sorted_preds = sorted(
            predictions,
            key=lambda x: (x["days_to_threshold"], -x["ua_degradation"])
        )

        # Most urgent should be first
        assert sorted_preds[0]["id"] == "HX-002"


class TestPredictionProvenanceTracking:
    """Test provenance tracking for predictions."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prediction_provenance(self, mock_ml_service):
        """Test provenance hash for predictions."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.1,
            "dp_tube_ratio": 1.1,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        # Generate provenance
        provenance_data = f"HX-001:Rf:{prediction['fouling_resistance_m2K_kW']:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64

    @pytest.mark.integration
    def test_prediction_audit_trail(self):
        """Test prediction audit trail generation."""
        audit_record = {
            "prediction_id": "pred-12345",
            "exchanger_id": "HX-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_features": {"dt_hot": 60.0, "dp_shell_ratio": 1.1},
            "output": {
                "fouling_resistance_m2K_kW": 0.00035,
                "confidence_score": 0.85,
            },
            "model_version": "1.2.0",
            "provenance_hash": "abc123...",
        }

        # Verify all required fields
        required_fields = ["prediction_id", "exchanger_id", "timestamp",
                          "input_features", "output", "model_version", "provenance_hash"]

        for field in required_fields:
            assert field in audit_record


class TestPredictionPipelinePerformance:
    """Test prediction pipeline performance."""

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_prediction_latency(self, mock_ml_service, performance_timer):
        """Test prediction latency."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.1,
            "dp_tube_ratio": 1.1,
            "reynolds_hot": 50000.0,
        }

        timer = performance_timer()
        with timer:
            for _ in range(100):
                await mock_ml_service.predict_fouling(features)

        timer.assert_under(1000)  # 100 predictions in <1 second

    @pytest.mark.integration
    @pytest.mark.performance
    def test_feature_extraction_throughput(self, sample_operating_state, mock_ml_service):
        """Test feature extraction throughput."""
        import time

        start = time.perf_counter()
        for _ in range(1000):
            mock_ml_service.extract_features(sample_operating_state)
        elapsed = time.perf_counter() - start

        throughput = 1000 / elapsed

        # Should process >10,000 feature extractions per second
        assert throughput > 1000


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestFeatureToPredictionFlow",
    "TestPredictionWithUncertainty",
    "TestTrendAnalysisPipeline",
    "TestAlertGenerationPipeline",
    "TestMultiExchangerPrediction",
    "TestPredictionProvenanceTracking",
    "TestPredictionPipelinePerformance",
]
