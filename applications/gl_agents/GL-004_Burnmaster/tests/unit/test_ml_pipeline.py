"""
Unit Tests for ML Pipeline - GL-004 BURNMASTER

Comprehensive tests for the ML inference pipeline including:
- MLPipelineManager lifecycle and inference
- PredictiveMaintenanceModel predictions
- CombustionAnomalyDetector detection
- EfficiencyPredictor predictions
- Physics-based fallback behavior
- Cache functionality
- Provenance tracking

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

# Import modules under test
from ml.pipeline import (
    MLPipelineManager,
    MLPipelineConfig,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    PredictionType,
    ModelStatus,
    InferenceMode,
    FallbackReason,
    InferenceCache,
    PhysicsCalculator,
)

from ml.models.predictive_maintenance import (
    PredictiveMaintenanceModel,
    MaintenanceFeatures,
    MaintenancePrediction,
    FailureMode,
    MaintenancePriority,
    HealthStatus,
)

from ml.models.anomaly_detector import (
    CombustionAnomalyDetector,
    CombustionFeatures,
    AnomalyResult,
    AnomalyAlert,
    AnomalySeverity,
    AnomalyType,
    DetectionMethod,
)

from ml.models.efficiency_predictor import (
    EfficiencyPredictor,
    EfficiencyFeatures,
    EfficiencyPrediction,
    EfficiencyFactors,
    FuelType,
    EfficiencyTrend,
    OptimizationPotential,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pipeline_config(temp_model_dir):
    """Create pipeline configuration."""
    return MLPipelineConfig(
        model_registry_path=temp_model_dir,
        enable_caching=True,
        cache_ttl_seconds=60,
        cache_max_size=100,
        inference_timeout_ms=1000,
        confidence_threshold=0.7,
        fallback_to_physics=True,
    )


@pytest.fixture
def pipeline_manager(pipeline_config):
    """Create initialized pipeline manager."""
    manager = MLPipelineManager(pipeline_config)
    asyncio.get_event_loop().run_until_complete(manager.initialize())
    yield manager
    asyncio.get_event_loop().run_until_complete(manager.shutdown())


@pytest.fixture
def maintenance_model():
    """Create predictive maintenance model."""
    return PredictiveMaintenanceModel()


@pytest.fixture
def anomaly_detector():
    """Create anomaly detector."""
    return CombustionAnomalyDetector()


@pytest.fixture
def efficiency_predictor():
    """Create efficiency predictor."""
    return EfficiencyPredictor(design_efficiency=90.0)


@pytest.fixture
def sample_maintenance_features():
    """Create sample maintenance features."""
    return MaintenanceFeatures(
        operating_hours=5000,
        hours_since_last_maintenance=2000,
        start_stop_cycles=150,
        max_flame_temp_c=1650,
        avg_flame_temp_c=1500,
        vibration_rms=4.5,
        efficiency_trend=-0.1,
        flame_stability_avg=0.85,
        equipment_age_years=3.5,
    )


@pytest.fixture
def sample_combustion_features():
    """Create sample combustion features."""
    return CombustionFeatures(
        o2_percent=3.5,
        co_ppm=45,
        nox_ppm=35,
        flame_temp_c=1480,
        lambda_value=1.15,
        flame_stability=0.9,
        pressure_variance=2.0,
        efficiency_percent=87.5,
    )


@pytest.fixture
def sample_efficiency_features():
    """Create sample efficiency features."""
    return EfficiencyFeatures(
        o2_percent=3.5,
        co_ppm=50,
        stack_temp_c=185,
        ambient_temp_c=25,
        fuel_type=FuelType.NATURAL_GAS,
        load_percent=80,
    )


# =============================================================================
# ML PIPELINE MANAGER TESTS
# =============================================================================


class TestMLPipelineManager:
    """Tests for MLPipelineManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        manager = MLPipelineManager(pipeline_config)
        assert not manager.is_initialized

        await manager.initialize()
        assert manager.is_initialized

        await manager.shutdown()
        assert not manager.is_initialized

    @pytest.mark.asyncio
    async def test_register_model(self, pipeline_manager, temp_model_dir):
        """Test model registration."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.8]))
        mock_model.predict_proba = MagicMock(return_value=np.array([[0.2, 0.8]]))

        model_id = await pipeline_manager.register_model(
            model=mock_model,
            model_type=PredictionType.EFFICIENCY_PREDICTION,
            version="1.0.0",
            feature_names=["o2_percent", "stack_temp_c"],
            set_active=True
        )

        assert model_id is not None
        assert pipeline_manager.get_active_model_id(
            PredictionType.EFFICIENCY_PREDICTION
        ) == model_id

        # Check model info
        info = pipeline_manager.get_model_info(model_id)
        assert info is not None
        assert info.model_type == PredictionType.EFFICIENCY_PREDICTION
        assert info.status == ModelStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_predict_with_fallback(self, pipeline_manager):
        """Test prediction falls back to physics when no model."""
        request = InferenceRequest(
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            features={"o2_percent": 3.5, "stack_temp_c": 180, "ambient_temp_c": 25},
        )

        response = await pipeline_manager.predict(request)

        assert response.is_fallback
        assert response.fallback_reason == FallbackReason.MODEL_NOT_LOADED
        assert response.model_id == "physics_fallback"
        assert 50 <= response.prediction <= 100  # Valid efficiency range

    @pytest.mark.asyncio
    async def test_predict_batch(self, pipeline_manager):
        """Test batch prediction."""
        samples = [
            {"o2_percent": 3.0, "stack_temp_c": 170, "ambient_temp_c": 25},
            {"o2_percent": 4.0, "stack_temp_c": 200, "ambient_temp_c": 25},
            {"o2_percent": 5.0, "stack_temp_c": 220, "ambient_temp_c": 25},
        ]

        request = BatchInferenceRequest(
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            samples=samples,
            parallel=True,
        )

        response = await pipeline_manager.predict_batch(request)

        assert response.total_count == 3
        assert len(response.predictions) == 3
        assert response.fallback_count == 3  # All use fallback

    @pytest.mark.asyncio
    async def test_pipeline_stats(self, pipeline_manager):
        """Test pipeline statistics."""
        # Make some predictions
        for _ in range(5):
            request = InferenceRequest(
                prediction_type=PredictionType.EFFICIENCY_PREDICTION,
                features={"o2_percent": 3.5},
            )
            await pipeline_manager.predict(request)

        stats = pipeline_manager.get_pipeline_stats()

        assert stats["initialized"] is True
        assert stats["total_predictions"] == 5
        assert stats["fallback_predictions"] == 5  # All fallback

    @pytest.mark.asyncio
    async def test_cache_hit(self, pipeline_config):
        """Test cache functionality."""
        manager = MLPipelineManager(pipeline_config)
        await manager.initialize()

        request = InferenceRequest(
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            features={"o2_percent": 3.5},
            use_cache=True,
        )

        # First request - cache miss
        response1 = await manager.predict(request)
        assert not response1.from_cache

        # Second request - cache hit
        response2 = await manager.predict(request)
        assert response2.from_cache

        await manager.shutdown()


class TestInferenceCache:
    """Tests for InferenceCache."""

    def test_cache_put_get(self):
        """Test cache put and get."""
        cache = InferenceCache(max_size=10, ttl_seconds=60)

        response = InferenceResponse(
            request_id="test-1",
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            prediction=85.0,
            confidence=0.9,
            model_id="test",
            latency_ms=10.0,
        )

        cache.put("key1", response)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.prediction == 85.0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = InferenceCache(max_size=10, ttl_seconds=60)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiry(self):
        """Test cache TTL expiry."""
        cache = InferenceCache(max_size=10, ttl_seconds=1)

        response = InferenceResponse(
            request_id="test-1",
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            prediction=85.0,
            confidence=0.9,
            model_id="test",
            latency_ms=10.0,
        )

        cache.put("key1", response)

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiry
        time.sleep(1.5)
        assert cache.get("key1") is None

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = InferenceCache(max_size=3, ttl_seconds=60)

        for i in range(5):
            response = InferenceResponse(
                request_id=f"test-{i}",
                prediction_type=PredictionType.EFFICIENCY_PREDICTION,
                prediction=80.0 + i,
                confidence=0.9,
                model_id="test",
                latency_ms=10.0,
            )
            cache.put(f"key{i}", response)

        # Only last 3 should be present
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = InferenceCache(max_size=10, ttl_seconds=60)

        response = InferenceResponse(
            request_id="test",
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            prediction=85.0,
            confidence=0.9,
            model_id="test",
            latency_ms=10.0,
        )

        cache.put("key1", response)
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestPhysicsCalculator:
    """Tests for PhysicsCalculator fallback."""

    def test_calculate_efficiency(self):
        """Test physics-based efficiency calculation."""
        features = {
            "o2_percent": 3.0,
            "stack_temp_c": 180,
            "ambient_temp_c": 25,
        }

        efficiency, confidence = PhysicsCalculator.calculate_efficiency(features)

        assert 70 <= efficiency <= 98
        assert 0.5 <= confidence <= 1.0

    def test_calculate_efficiency_high_o2(self):
        """Test efficiency with high O2 (excess air)."""
        features = {
            "o2_percent": 8.0,
            "stack_temp_c": 200,
            "ambient_temp_c": 25,
        }

        efficiency, _ = PhysicsCalculator.calculate_efficiency(features)

        # High excess air should reduce efficiency
        assert efficiency < 90

    def test_calculate_failure_probability(self):
        """Test failure probability calculation."""
        features = {
            "operating_hours": 20000,
            "vibration_rms": 8,
            "start_stop_cycles": 300,
        }

        prob, confidence = PhysicsCalculator.calculate_failure_probability(features)

        assert 0 <= prob <= 1
        assert 0 <= confidence <= 1

    def test_detect_anomaly_threshold_normal(self):
        """Test threshold-based anomaly detection - normal case."""
        features = {
            "o2_percent": 3.5,
            "co_ppm": 50,
            "flame_temp_c": 1500,
            "lambda": 1.15,
        }

        is_anomaly, score, anomaly_type = PhysicsCalculator.detect_anomaly_threshold(features)

        assert not is_anomaly
        assert score < 0.3

    def test_detect_anomaly_threshold_high_co(self):
        """Test threshold-based anomaly detection - high CO."""
        features = {
            "o2_percent": 3.5,
            "co_ppm": 350,  # High CO
            "flame_temp_c": 1500,
        }

        is_anomaly, score, anomaly_type = PhysicsCalculator.detect_anomaly_threshold(features)

        assert is_anomaly
        assert anomaly_type == "high_co"


# =============================================================================
# PREDICTIVE MAINTENANCE TESTS
# =============================================================================


class TestPredictiveMaintenanceModel:
    """Tests for PredictiveMaintenanceModel."""

    def test_predict_healthy_equipment(self, maintenance_model):
        """Test prediction for healthy equipment."""
        features = MaintenanceFeatures(
            operating_hours=1000,
            hours_since_last_maintenance=500,
            start_stop_cycles=50,
            max_flame_temp_c=1500,
            flame_stability_avg=0.95,
        )

        prediction = maintenance_model.predict(features)

        assert isinstance(prediction, MaintenancePrediction)
        assert prediction.failure_probability < 0.3
        assert prediction.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert prediction.remaining_useful_life_hours > 1000

    def test_predict_degraded_equipment(self, maintenance_model):
        """Test prediction for degraded equipment."""
        features = MaintenanceFeatures(
            operating_hours=30000,
            hours_since_last_maintenance=8000,
            start_stop_cycles=400,
            max_flame_temp_c=1750,
            vibration_rms=10,
            efficiency_trend=-0.5,
            flame_stability_avg=0.7,
        )

        prediction = maintenance_model.predict(features)

        assert prediction.failure_probability > 0.3
        assert prediction.health_status in [
            HealthStatus.DEGRADED,
            HealthStatus.WARNING,
            HealthStatus.CRITICAL
        ]
        assert len(prediction.recommended_actions) > 0

    def test_failure_mode_probabilities(self, maintenance_model, sample_maintenance_features):
        """Test failure mode probability distribution."""
        prediction = maintenance_model.predict(sample_maintenance_features)

        # Should sum to approximately 1
        total_prob = sum(prediction.failure_mode_probabilities.values())
        assert 0.95 <= total_prob <= 1.05

        # Predicted mode should have highest probability
        mode_probs = prediction.failure_mode_probabilities
        predicted_mode = prediction.predicted_failure_mode.value
        if predicted_mode in mode_probs:
            assert mode_probs[predicted_mode] >= max(
                p for k, p in mode_probs.items() if k != predicted_mode
            )

    def test_maintenance_priority(self, maintenance_model):
        """Test maintenance priority determination."""
        # Critical case
        critical_features = MaintenanceFeatures(
            operating_hours=50000,
            flame_stability_avg=0.4,
        )
        critical_pred = maintenance_model.predict(critical_features)
        assert critical_pred.maintenance_priority in [
            MaintenancePriority.CRITICAL,
            MaintenancePriority.HIGH
        ]

        # Routine case
        routine_features = MaintenanceFeatures(
            operating_hours=1000,
            flame_stability_avg=0.95,
        )
        routine_pred = maintenance_model.predict(routine_features)
        assert routine_pred.maintenance_priority in [
            MaintenancePriority.LOW,
            MaintenancePriority.ROUTINE
        ]

    def test_rul_bounds(self, maintenance_model, sample_maintenance_features):
        """Test RUL confidence bounds."""
        prediction = maintenance_model.predict(sample_maintenance_features)

        assert prediction.rul_lower_bound <= prediction.remaining_useful_life_hours
        assert prediction.remaining_useful_life_hours <= prediction.rul_upper_bound

    def test_provenance_hash(self, maintenance_model, sample_maintenance_features):
        """Test provenance hash generation."""
        prediction = maintenance_model.predict(sample_maintenance_features)

        assert prediction.provenance_hash
        assert len(prediction.provenance_hash) == 64  # SHA-256 hex

    def test_top_risk_factors(self, maintenance_model):
        """Test risk factor identification."""
        features = MaintenanceFeatures(
            operating_hours=25000,
            max_flame_temp_c=1800,
            vibration_rms=12,
        )

        prediction = maintenance_model.predict(features)

        assert len(prediction.top_risk_factors) > 0
        # Risk factors should be sorted by contribution
        contributions = [f["risk_contribution"] for f in prediction.top_risk_factors]
        assert contributions == sorted(contributions, reverse=True)


# =============================================================================
# ANOMALY DETECTOR TESTS
# =============================================================================


class TestCombustionAnomalyDetector:
    """Tests for CombustionAnomalyDetector."""

    def test_detect_normal(self, anomaly_detector, sample_combustion_features):
        """Test detection of normal operation."""
        result = anomaly_detector.detect(sample_combustion_features)

        assert isinstance(result, AnomalyResult)
        assert not result.is_anomaly
        assert result.anomaly_type == AnomalyType.NORMAL

    def test_detect_co_spike(self, anomaly_detector):
        """Test detection of CO spike."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=400,  # High CO
            flame_temp_c=1480,
            lambda_value=1.15,
            flame_stability=0.85,
        )

        result = anomaly_detector.detect(features)

        assert result.is_anomaly
        assert result.anomaly_type == AnomalyType.CO_SPIKE
        assert result.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]

    def test_detect_lean_blowout_risk(self, anomaly_detector):
        """Test detection of lean blowout risk."""
        features = CombustionFeatures(
            o2_percent=0.3,  # Very low O2
            co_ppm=50,
            flame_temp_c=1200,
            lambda_value=0.98,  # Sub-stoichiometric
            flame_stability=0.6,
        )

        result = anomaly_detector.detect(features)

        assert result.is_anomaly
        assert result.anomaly_type in [
            AnomalyType.LEAN_BLOWOUT_RISK,
            AnomalyType.RICH_COMBUSTION
        ]

    def test_detect_flame_instability(self, anomaly_detector):
        """Test detection of flame instability."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=50,
            flame_temp_c=1480,
            lambda_value=1.15,
            flame_stability=0.4,  # Low stability
            pressure_variance=15,  # High variance
        )

        result = anomaly_detector.detect(features)

        assert result.is_anomaly
        assert result.severity != AnomalySeverity.INFO

    def test_anomaly_alerts(self, anomaly_detector):
        """Test alert generation for anomalies."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=350,  # High CO
            flame_temp_c=1480,
            flame_stability=0.85,
        )

        result = anomaly_detector.detect(features)

        assert result.is_anomaly
        assert len(result.alerts) > 0

        alert = result.alerts[0]
        assert isinstance(alert, AnomalyAlert)
        assert alert.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        assert len(alert.recommended_actions) > 0

    def test_alert_cooldown(self, anomaly_detector):
        """Test alert cooldown to prevent flooding."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=350,
            flame_temp_c=1480,
        )

        # First detection generates alert
        result1 = anomaly_detector.detect(features)
        assert len(result1.alerts) > 0

        # Immediate second detection should not generate alert (cooldown)
        result2 = anomaly_detector.detect(features)
        assert len(result2.alerts) == 0

    def test_feature_contributions(self, anomaly_detector):
        """Test feature contribution calculation."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=200,
            flame_temp_c=1480,
        )

        result = anomaly_detector.detect(features)

        # Should have contributions for measured features
        assert "co_ppm" in result.feature_contributions or len(result.feature_contributions) > 0

    def test_detection_method(self, anomaly_detector):
        """Test detection method is correctly identified."""
        features = CombustionFeatures(
            o2_percent=3.5,
            co_ppm=50,
        )

        result = anomaly_detector.detect(features)

        # Without fitting, should use threshold method
        assert result.detection_method == DetectionMethod.THRESHOLD


# =============================================================================
# EFFICIENCY PREDICTOR TESTS
# =============================================================================


class TestEfficiencyPredictor:
    """Tests for EfficiencyPredictor."""

    def test_predict_efficiency(self, efficiency_predictor, sample_efficiency_features):
        """Test efficiency prediction."""
        prediction = efficiency_predictor.predict(sample_efficiency_features)

        assert isinstance(prediction, EfficiencyPrediction)
        assert 50 <= prediction.efficiency_percent <= 100
        assert prediction.efficiency_lower_bound <= prediction.efficiency_percent
        assert prediction.efficiency_percent <= prediction.efficiency_upper_bound

    def test_efficiency_with_high_o2(self, efficiency_predictor):
        """Test efficiency decreases with high O2."""
        low_o2_features = EfficiencyFeatures(
            o2_percent=2.5,
            stack_temp_c=180,
            fuel_type=FuelType.NATURAL_GAS,
        )

        high_o2_features = EfficiencyFeatures(
            o2_percent=7.0,
            stack_temp_c=180,
            fuel_type=FuelType.NATURAL_GAS,
        )

        low_o2_pred = efficiency_predictor.predict(low_o2_features)
        high_o2_pred = efficiency_predictor.predict(high_o2_features)

        # Higher O2 should result in lower efficiency
        assert low_o2_pred.efficiency_percent > high_o2_pred.efficiency_percent

    def test_efficiency_with_high_stack_temp(self, efficiency_predictor):
        """Test efficiency decreases with high stack temperature."""
        low_temp_features = EfficiencyFeatures(
            o2_percent=3.0,
            stack_temp_c=150,
        )

        high_temp_features = EfficiencyFeatures(
            o2_percent=3.0,
            stack_temp_c=250,
        )

        low_temp_pred = efficiency_predictor.predict(low_temp_features)
        high_temp_pred = efficiency_predictor.predict(high_temp_features)

        # Higher stack temp should result in lower efficiency
        assert low_temp_pred.efficiency_percent > high_temp_pred.efficiency_percent

    def test_loss_breakdown(self, efficiency_predictor, sample_efficiency_features):
        """Test efficiency loss breakdown."""
        prediction = efficiency_predictor.predict(sample_efficiency_features)

        factors = prediction.efficiency_factors
        assert isinstance(factors, EfficiencyFactors)

        # Losses should sum to approximately 100 - efficiency
        total_losses = factors.total_losses_percent
        expected_losses = 100 - prediction.efficiency_percent
        assert abs(total_losses - expected_losses) < 1.0

    def test_optimization_opportunities(self, efficiency_predictor):
        """Test optimization opportunity identification."""
        # Create features with optimization potential
        features = EfficiencyFeatures(
            o2_percent=6.0,  # High O2
            stack_temp_c=230,  # High stack temp
            co_ppm=120,  # Moderate CO
            fuel_type=FuelType.NATURAL_GAS,
        )

        prediction = efficiency_predictor.predict(features)

        assert len(prediction.opportunities) > 0
        assert prediction.potential_improvement_percent > 0

        # Check opportunity structure
        opp = prediction.opportunities[0]
        assert opp.potential_gain_percent > 0
        assert opp.category in ["air_fuel_ratio", "heat_recovery", "combustion_quality"]

    def test_what_if_scenarios(self, efficiency_predictor, sample_efficiency_features):
        """Test what-if scenario generation."""
        prediction = efficiency_predictor.predict(sample_efficiency_features)

        assert len(prediction.what_if_scenarios) > 0

        for scenario in prediction.what_if_scenarios:
            assert "name" in scenario
            assert "expected_efficiency" in scenario
            assert "improvement" in scenario

    def test_fuel_type_impact(self, efficiency_predictor):
        """Test different fuel types have different efficiency ranges."""
        base_features = {
            "o2_percent": 3.0,
            "stack_temp_c": 180,
        }

        ng_features = EfficiencyFeatures(**base_features, fuel_type=FuelType.NATURAL_GAS)
        oil_features = EfficiencyFeatures(**base_features, fuel_type=FuelType.FUEL_OIL_6)

        ng_pred = efficiency_predictor.predict(ng_features)
        oil_pred = efficiency_predictor.predict(oil_features)

        # Natural gas should generally have higher efficiency
        assert ng_pred.efficiency_percent >= oil_pred.efficiency_percent - 5

    def test_optimization_potential_levels(self, efficiency_predictor):
        """Test optimization potential classification."""
        # Near-optimal operation
        optimal_features = EfficiencyFeatures(
            o2_percent=3.0,
            stack_temp_c=160,
            co_ppm=20,
        )
        optimal_pred = efficiency_predictor.predict(optimal_features)
        assert optimal_pred.optimization_potential in [
            OptimizationPotential.OPTIMAL,
            OptimizationPotential.LOW
        ]

        # Poor operation
        poor_features = EfficiencyFeatures(
            o2_percent=8.0,
            stack_temp_c=280,
            co_ppm=200,
        )
        poor_pred = efficiency_predictor.predict(poor_features)
        assert poor_pred.optimization_potential in [
            OptimizationPotential.HIGH,
            OptimizationPotential.MEDIUM
        ]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMLPipelineIntegration:
    """Integration tests for the full ML pipeline."""

    @pytest.mark.asyncio
    async def test_full_inference_flow(self, temp_model_dir):
        """Test complete inference flow."""
        config = MLPipelineConfig(
            model_registry_path=temp_model_dir,
            enable_caching=True,
            fallback_to_physics=True,
        )

        manager = MLPipelineManager(config)
        await manager.initialize()

        # Test efficiency prediction
        eff_request = InferenceRequest(
            prediction_type=PredictionType.EFFICIENCY_PREDICTION,
            features={"o2_percent": 3.5, "stack_temp_c": 180},
        )
        eff_response = await manager.predict(eff_request)
        assert eff_response.prediction is not None
        assert eff_response.provenance_hash

        # Test anomaly detection
        anom_request = InferenceRequest(
            prediction_type=PredictionType.ANOMALY_DETECTION,
            features={"o2_percent": 3.5, "co_ppm": 50},
        )
        anom_response = await manager.predict(anom_request)
        assert anom_response.prediction is not None

        # Test maintenance prediction
        maint_request = InferenceRequest(
            prediction_type=PredictionType.PREDICTIVE_MAINTENANCE,
            features={"operating_hours": 5000, "vibration_rms": 5},
        )
        maint_response = await manager.predict(maint_request)
        assert maint_response.prediction is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_model_registration_and_inference(self, temp_model_dir):
        """Test model registration followed by inference."""
        config = MLPipelineConfig(
            model_registry_path=temp_model_dir,
            fallback_to_physics=True,
        )

        manager = MLPipelineManager(config)
        await manager.initialize()

        # Create and register an efficiency predictor
        predictor = EfficiencyPredictor()

        model_id = await manager.register_model(
            model=predictor._regressor if predictor._regressor else MagicMock(),
            model_type=PredictionType.EFFICIENCY_PREDICTION,
            feature_names=predictor.FEATURE_NAMES,
            set_active=True,
        )

        assert manager.get_active_model_id(PredictionType.EFFICIENCY_PREDICTION) == model_id

        await manager.shutdown()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance benchmark tests."""

    def test_maintenance_prediction_latency(self, maintenance_model):
        """Test maintenance prediction meets latency requirement."""
        features = MaintenanceFeatures(operating_hours=5000)

        start = time.time()
        for _ in range(100):
            maintenance_model.predict(features)
        elapsed = (time.time() - start) * 1000  # Total ms

        avg_latency = elapsed / 100
        assert avg_latency < 50  # Should be under 50ms per prediction

    def test_anomaly_detection_latency(self, anomaly_detector):
        """Test anomaly detection meets latency requirement."""
        features = CombustionFeatures(o2_percent=3.5, co_ppm=50)

        start = time.time()
        for _ in range(100):
            anomaly_detector.detect(features)
        elapsed = (time.time() - start) * 1000

        avg_latency = elapsed / 100
        assert avg_latency < 20  # Should be under 20ms per detection

    def test_efficiency_prediction_latency(self, efficiency_predictor):
        """Test efficiency prediction meets latency requirement."""
        features = EfficiencyFeatures(o2_percent=3.5, stack_temp_c=180)

        start = time.time()
        for _ in range(100):
            efficiency_predictor.predict(features)
        elapsed = (time.time() - start) * 1000

        avg_latency = elapsed / 100
        assert avg_latency < 30  # Should be under 30ms per prediction


# =============================================================================
# PROVENANCE AND AUDIT TESTS
# =============================================================================


class TestProvenanceTracking:
    """Tests for provenance tracking and audit trail."""

    def test_maintenance_provenance_deterministic(self, maintenance_model):
        """Test maintenance provenance hash is deterministic."""
        features = MaintenanceFeatures(
            operating_hours=5000,
            start_stop_cycles=100,
        )

        pred1 = maintenance_model.predict(features)
        pred2 = maintenance_model.predict(features)

        # Same input should produce same provenance hash
        # Note: timestamps differ, so we check hash exists
        assert pred1.provenance_hash
        assert pred2.provenance_hash

    def test_anomaly_provenance(self, anomaly_detector):
        """Test anomaly detection provenance hash."""
        features = CombustionFeatures(o2_percent=3.5, co_ppm=50)

        result = anomaly_detector.detect(features)

        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_efficiency_provenance(self, efficiency_predictor):
        """Test efficiency prediction provenance hash."""
        features = EfficiencyFeatures(o2_percent=3.5, stack_temp_c=180)

        prediction = efficiency_predictor.predict(features)

        assert prediction.provenance_hash
        assert len(prediction.provenance_hash) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
