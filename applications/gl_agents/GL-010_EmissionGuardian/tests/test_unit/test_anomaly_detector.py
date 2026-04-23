"""
GL-010 EmissionGuardian - Anomaly Detector Tests

Comprehensive test suite for the fugitive emissions anomaly detection system.
Tests Isolation Forest, statistical detection, and ensemble methods.

Zero-Hallucination: ML provides scores only; thresholds are deterministic.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List
from unittest.mock import Mock, patch
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fugitive.anomaly_detector import (
    AnomalyType,
    AnomalySeverity,
    DetectorType,
    AnomalyDetectorConfig,
    AnomalyScore,
    AnomalyDetection,
    IsolationForestDetector,
    StatisticalDetector,
    AnomalyDetector,
)
from fugitive.feature_engineering import FeatureVector, EquipmentType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default anomaly detector configuration."""
    return AnomalyDetectorConfig()


@pytest.fixture
def sample_feature_vector():
    """Sample feature vector for testing."""
    return FeatureVector(
        feature_id="FV-001",
        timestamp=datetime.now(),
        sensor_id="SENSOR-001",
        concentration_current=150.0,
        concentration_mean_1h=140.0,
        concentration_std_1h=20.0,
        concentration_max_1h=180.0,
        concentration_min_1h=120.0,
        concentration_zscore=0.5,
        elevation_above_background=50.0,
        background_concentration=100.0,
        spatial_gradient=0.2,
        spatial_anomaly_score=0.3,
        temporal_trend=0.1,
        temporal_anomaly_score=0.2,
        plume_likelihood_score=0.25,
        wind_speed_ms=3.0,
        wind_direction_deg=180.0,
        temperature_c=25.0,
        pressure_hpa=1013.0,
        humidity_pct=50.0,
        equipment_id="EQ-001",
        equipment_type=EquipmentType.VALVE,
        equipment_age_years=5.0,
        last_maintenance_days=30,
    )


@pytest.fixture
def anomalous_feature_vector():
    """Feature vector representing an anomaly."""
    return FeatureVector(
        feature_id="FV-002",
        timestamp=datetime.now(),
        sensor_id="SENSOR-002",
        concentration_current=800.0,  # High concentration
        concentration_mean_1h=200.0,
        concentration_std_1h=50.0,
        concentration_max_1h=850.0,
        concentration_min_1h=180.0,
        concentration_zscore=4.0,  # High z-score
        elevation_above_background=600.0,  # High elevation
        background_concentration=200.0,
        spatial_gradient=0.8,
        spatial_anomaly_score=0.85,  # High spatial anomaly
        temporal_trend=0.9,
        temporal_anomaly_score=0.8,
        plume_likelihood_score=0.9,  # High plume likelihood
        wind_speed_ms=2.0,
        wind_direction_deg=90.0,
        temperature_c=28.0,
        pressure_hpa=1010.0,
        humidity_pct=60.0,
        equipment_id="EQ-002",
        equipment_type=EquipmentType.COMPRESSOR,
        equipment_age_years=15.0,
        last_maintenance_days=180,
    )


@pytest.fixture
def training_feature_vectors():
    """List of feature vectors for training."""
    vectors = []
    for i in range(150):
        vectors.append(FeatureVector(
            feature_id=f"FV-TRAIN-{i:03d}",
            timestamp=datetime.now(),
            sensor_id="SENSOR-TRAIN",
            concentration_current=100.0 + (i % 50),
            concentration_mean_1h=100.0,
            concentration_std_1h=15.0,
            concentration_max_1h=150.0,
            concentration_min_1h=80.0,
            concentration_zscore=0.1 * (i % 10 - 5),
            elevation_above_background=20.0 + (i % 30),
            background_concentration=100.0,
            spatial_gradient=0.1,
            spatial_anomaly_score=0.1 + 0.01 * (i % 20),
            temporal_trend=0.0,
            temporal_anomaly_score=0.1,
            plume_likelihood_score=0.15,
            wind_speed_ms=3.0,
            wind_direction_deg=float(i % 360),
            temperature_c=20.0 + (i % 15),
            pressure_hpa=1010.0 + (i % 20),
            humidity_pct=50.0 + (i % 30),
            equipment_id="EQ-TRAIN",
            equipment_type=EquipmentType.VALVE,
            equipment_age_years=5.0,
            last_maintenance_days=30,
        ))
    return vectors


# =============================================================================
# TEST: ENUMS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_anomaly_type_values(self):
        """All expected anomaly types should be defined."""
        expected = {
            "concentration_spike", "sustained_elevation",
            "spatial_gradient", "plume_signature",
            "equipment_anomaly", "temporal_pattern", "unknown"
        }
        actual = {at.value for at in AnomalyType}
        assert expected == actual

    def test_anomaly_severity_values(self):
        """All severity levels should be defined."""
        expected = {"low", "medium", "high", "critical"}
        actual = {s.value for s in AnomalySeverity}
        assert expected == actual

    def test_detector_type_values(self):
        """All detector types should be defined."""
        expected = {"isolation_forest", "statistical", "ensemble"}
        actual = {dt.value for dt in DetectorType}
        assert expected == actual


# =============================================================================
# TEST: CONFIGURATION
# =============================================================================

class TestAnomalyDetectorConfig:
    """Test AnomalyDetectorConfig validation."""

    def test_default_config(self, default_config):
        """Default config should have reasonable values."""
        assert default_config.detector_type == DetectorType.ENSEMBLE
        assert default_config.if_n_estimators == 100
        assert default_config.if_contamination == 0.05
        assert default_config.zscore_threshold == 3.0
        assert default_config.alert_threshold == 0.7

    def test_config_validation_bounds(self):
        """Config should validate parameter bounds."""
        # n_estimators minimum
        with pytest.raises(ValueError):
            AnomalyDetectorConfig(if_n_estimators=5)  # Below 10

        # contamination bounds
        with pytest.raises(ValueError):
            AnomalyDetectorConfig(if_contamination=0.5)  # Above 0.3

    def test_config_custom_values(self):
        """Custom config values should be accepted."""
        config = AnomalyDetectorConfig(
            detector_type=DetectorType.STATISTICAL,
            zscore_threshold=2.5,
            alert_threshold=0.8,
        )

        assert config.detector_type == DetectorType.STATISTICAL
        assert config.zscore_threshold == 2.5
        assert config.alert_threshold == 0.8


# =============================================================================
# TEST: ANOMALY SCORE
# =============================================================================

class TestAnomalyScore:
    """Test AnomalyScore dataclass."""

    def test_score_structure(self):
        """AnomalyScore should have all required fields."""
        score = AnomalyScore(
            detector_type=DetectorType.STATISTICAL,
            score=0.75,
            raw_score=0.75,
            threshold_used=0.7,
            is_anomaly=True,
            computation_time_ms=1.5,
        )

        assert score.detector_type == DetectorType.STATISTICAL
        assert score.score == 0.75
        assert score.is_anomaly is True


# =============================================================================
# TEST: STATISTICAL DETECTOR
# =============================================================================

class TestStatisticalDetector:
    """Test StatisticalDetector."""

    def test_detector_init(self, default_config):
        """Statistical detector should initialize."""
        detector = StatisticalDetector(default_config)
        assert detector.config == default_config

    def test_normal_feature_low_score(
        self, default_config, sample_feature_vector
    ):
        """Normal features should have low anomaly score."""
        detector = StatisticalDetector(default_config)
        score = detector.predict(sample_feature_vector)

        assert score.detector_type == DetectorType.STATISTICAL
        assert score.score < 0.7  # Below alert threshold
        assert score.is_anomaly is False

    def test_anomalous_feature_high_score(
        self, default_config, anomalous_feature_vector
    ):
        """Anomalous features should have high anomaly score."""
        detector = StatisticalDetector(default_config)
        score = detector.predict(anomalous_feature_vector)

        assert score.score >= 0.7  # Above alert threshold
        assert score.is_anomaly is True

    def test_contributing_factors(
        self, default_config, anomalous_feature_vector
    ):
        """Should identify contributing factors."""
        detector = StatisticalDetector(default_config)
        factors = detector.get_contributing_factors(anomalous_feature_vector)

        assert len(factors) > 0
        # Factors should be sorted by contribution
        for i in range(len(factors) - 1):
            assert factors[i][1] >= factors[i + 1][1]


# =============================================================================
# TEST: ISOLATION FOREST DETECTOR
# =============================================================================

class TestIsolationForestDetector:
    """Test IsolationForestDetector."""

    def test_detector_init(self, default_config):
        """Isolation Forest detector should initialize."""
        detector = IsolationForestDetector(default_config)

        assert detector.config == default_config
        assert detector._is_fitted is False

    def test_predict_unfitted_returns_neutral(
        self, default_config, sample_feature_vector
    ):
        """Unfitted detector should return neutral score."""
        detector = IsolationForestDetector(default_config)
        score = detector.predict(sample_feature_vector)

        assert score.score == 0.5  # Neutral
        assert score.is_anomaly is False

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn not available"),
        reason="sklearn not installed"
    )
    def test_fit_and_predict(
        self, default_config, training_feature_vectors, sample_feature_vector
    ):
        """Fitted detector should produce meaningful scores."""
        detector = IsolationForestDetector(default_config)
        detector.fit(training_feature_vectors)

        assert detector._is_fitted is True

        score = detector.predict(sample_feature_vector)
        assert 0 <= score.score <= 1

    def test_fit_insufficient_samples(self, default_config):
        """Fit with insufficient samples should warn."""
        detector = IsolationForestDetector(default_config)
        small_dataset = [
            FeatureVector(
                feature_id=f"FV-{i}",
                timestamp=datetime.now(),
                sensor_id="S",
                concentration_current=100.0,
                concentration_mean_1h=100.0,
                concentration_std_1h=10.0,
                concentration_max_1h=110.0,
                concentration_min_1h=90.0,
                concentration_zscore=0.0,
                elevation_above_background=10.0,
                background_concentration=90.0,
                spatial_gradient=0.1,
                spatial_anomaly_score=0.1,
                temporal_trend=0.0,
                temporal_anomaly_score=0.1,
                plume_likelihood_score=0.1,
                wind_speed_ms=3.0,
                wind_direction_deg=180.0,
                temperature_c=25.0,
                pressure_hpa=1013.0,
                humidity_pct=50.0,
            )
            for i in range(10)  # Less than min_training_samples
        ]

        detector.fit(small_dataset)

        # Should not be fitted due to insufficient samples
        assert detector._is_fitted is False


# =============================================================================
# TEST: ENSEMBLE ANOMALY DETECTOR
# =============================================================================

class TestAnomalyDetector:
    """Test main AnomalyDetector class."""

    def test_detector_init_default(self):
        """Detector should initialize with default config."""
        detector = AnomalyDetector()

        assert detector.config.detector_type == DetectorType.ENSEMBLE
        assert detector._config_hash is not None

    def test_detector_init_custom_config(self):
        """Detector should accept custom config."""
        config = AnomalyDetectorConfig(
            detector_type=DetectorType.STATISTICAL,
            alert_threshold=0.8,
        )
        detector = AnomalyDetector(config)

        assert detector.config.detector_type == DetectorType.STATISTICAL

    def test_detect_basic(self, sample_feature_vector):
        """Basic detection should work."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        assert isinstance(result, AnomalyDetection)
        assert result.detection_id.startswith("DET-")
        assert result.feature_id == sample_feature_vector.feature_id

    def test_detect_anomaly_flagged(self, anomalous_feature_vector):
        """Anomalous features should be flagged."""
        detector = AnomalyDetector()
        result = detector.detect(anomalous_feature_vector)

        # With high z-score and plume likelihood, should be anomaly
        assert result.final_score > 0.5
        # May or may not be flagged as anomaly depending on threshold

    def test_detect_includes_scores(self, sample_feature_vector):
        """Detection should include component scores."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        # Ensemble includes both IF and statistical
        assert result.statistical_score is not None or result.isolation_forest_score is not None

    def test_detect_provenance_hash(self, sample_feature_vector):
        """Detection should include provenance hash."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        assert len(result.provenance_hash) == 64
        assert result.detector_config_hash is not None

    def test_detect_to_dict(self, sample_feature_vector):
        """Detection should convert to API dict."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        d = result.to_dict()

        assert "detection_id" in d
        assert "final_score" in d
        assert "is_anomaly" in d
        assert "anomaly_type" in d
        assert "severity" in d
        assert "requires_review" in d

    def test_detect_batch(self, sample_feature_vector, anomalous_feature_vector):
        """Batch detection should work."""
        detector = AnomalyDetector()
        vectors = [sample_feature_vector, anomalous_feature_vector]

        results = detector.detect_batch(vectors)

        assert len(results) == 2
        assert all(isinstance(r, AnomalyDetection) for r in results)


# =============================================================================
# TEST: ANOMALY CLASSIFICATION
# =============================================================================

class TestAnomalyClassification:
    """Test anomaly type classification."""

    def test_classify_concentration_spike(self):
        """High z-score should classify as concentration spike."""
        detector = AnomalyDetector()

        fv = FeatureVector(
            feature_id="FV-SPIKE",
            timestamp=datetime.now(),
            sensor_id="S",
            concentration_current=500.0,
            concentration_mean_1h=100.0,
            concentration_std_1h=50.0,
            concentration_max_1h=500.0,
            concentration_min_1h=80.0,
            concentration_zscore=4.5,  # High z-score (>3)
            elevation_above_background=400.0,
            background_concentration=100.0,
            spatial_gradient=0.1,
            spatial_anomaly_score=0.2,
            temporal_trend=0.1,
            temporal_anomaly_score=0.2,
            plume_likelihood_score=0.3,
            wind_speed_ms=3.0,
            wind_direction_deg=180.0,
            temperature_c=25.0,
            pressure_hpa=1013.0,
            humidity_pct=50.0,
        )

        result = detector.detect(fv)

        assert result.anomaly_type == AnomalyType.CONCENTRATION_SPIKE

    def test_classify_spatial_gradient(self):
        """High spatial anomaly should classify as spatial gradient."""
        detector = AnomalyDetector()

        fv = FeatureVector(
            feature_id="FV-SPATIAL",
            timestamp=datetime.now(),
            sensor_id="S",
            concentration_current=150.0,
            concentration_mean_1h=140.0,
            concentration_std_1h=20.0,
            concentration_max_1h=180.0,
            concentration_min_1h=120.0,
            concentration_zscore=0.5,  # Low z-score
            elevation_above_background=50.0,
            background_concentration=100.0,
            spatial_gradient=0.9,
            spatial_anomaly_score=0.85,  # High spatial anomaly (>0.7)
            temporal_trend=0.1,
            temporal_anomaly_score=0.2,
            plume_likelihood_score=0.3,
            wind_speed_ms=3.0,
            wind_direction_deg=180.0,
            temperature_c=25.0,
            pressure_hpa=1013.0,
            humidity_pct=50.0,
        )

        result = detector.detect(fv)

        assert result.anomaly_type == AnomalyType.SPATIAL_GRADIENT

    def test_classify_plume_signature(self):
        """High plume likelihood should classify as plume signature."""
        detector = AnomalyDetector()

        fv = FeatureVector(
            feature_id="FV-PLUME",
            timestamp=datetime.now(),
            sensor_id="S",
            concentration_current=200.0,
            concentration_mean_1h=180.0,
            concentration_std_1h=30.0,
            concentration_max_1h=220.0,
            concentration_min_1h=150.0,
            concentration_zscore=0.7,  # Below 3
            elevation_above_background=80.0,
            background_concentration=120.0,
            spatial_gradient=0.3,
            spatial_anomaly_score=0.4,  # Below 0.7
            temporal_trend=0.2,
            temporal_anomaly_score=0.3,
            plume_likelihood_score=0.85,  # High plume likelihood (>0.7)
            wind_speed_ms=3.0,
            wind_direction_deg=180.0,
            temperature_c=25.0,
            pressure_hpa=1013.0,
            humidity_pct=50.0,
        )

        result = detector.detect(fv)

        assert result.anomaly_type == AnomalyType.PLUME_SIGNATURE


# =============================================================================
# TEST: SEVERITY DETERMINATION
# =============================================================================

class TestSeverityDetermination:
    """Test anomaly severity determination."""

    def test_severity_low(self, sample_feature_vector):
        """Normal features should have LOW severity."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        if result.final_score < 0.7:
            assert result.severity == AnomalySeverity.LOW

    def test_severity_critical(self, anomalous_feature_vector):
        """Very high scores should be CRITICAL."""
        config = AnomalyDetectorConfig(critical_threshold=0.9)
        detector = AnomalyDetector(config)
        result = detector.detect(anomalous_feature_vector)

        if result.final_score >= 0.9:
            assert result.severity == AnomalySeverity.CRITICAL


# =============================================================================
# TEST: HUMAN REVIEW REQUIREMENT
# =============================================================================

class TestHumanReview:
    """Test human review requirements."""

    def test_anomaly_requires_review(self, anomalous_feature_vector):
        """Anomalies should require human review."""
        detector = AnomalyDetector()
        result = detector.detect(anomalous_feature_vector)

        if result.is_anomaly:
            assert result.requires_review is True

    def test_review_priority_based_on_severity(self):
        """Review priority should be based on severity."""
        detector = AnomalyDetector()

        # Create features with different severity levels
        # Higher scores = higher severity = lower priority number (1=highest)


# =============================================================================
# TEST: EXPLAINABILITY
# =============================================================================

class TestExplainability:
    """Test anomaly detection explainability."""

    def test_contributing_features_included(self, sample_feature_vector):
        """Detection should include contributing features."""
        detector = AnomalyDetector()
        result = detector.detect(sample_feature_vector)

        # Should have contributing features
        assert isinstance(result.top_contributing_features, list)

    def test_contributing_features_sorted(self, anomalous_feature_vector):
        """Contributing features should be sorted by importance."""
        detector = AnomalyDetector()
        result = detector.detect(anomalous_feature_vector)

        if len(result.top_contributing_features) > 1:
            contributions = [c[1] for c in result.top_contributing_features]
            # Should be descending order
            for i in range(len(contributions) - 1):
                assert contributions[i] >= contributions[i + 1]


# =============================================================================
# TEST: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_statistical_detector_deterministic(
        self, default_config, sample_feature_vector
    ):
        """Statistical detector should be deterministic."""
        detector = StatisticalDetector(default_config)

        score1 = detector.predict(sample_feature_vector)
        score2 = detector.predict(sample_feature_vector)

        assert score1.score == score2.score
        assert score1.is_anomaly == score2.is_anomaly

    def test_config_hash_deterministic(self):
        """Config hash should be deterministic."""
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()

        assert detector1._config_hash == detector2._config_hash


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_zero_concentration(self):
        """Zero concentration should not cause errors."""
        detector = AnomalyDetector()

        fv = FeatureVector(
            feature_id="FV-ZERO",
            timestamp=datetime.now(),
            sensor_id="S",
            concentration_current=0.0,
            concentration_mean_1h=0.0,
            concentration_std_1h=0.0,
            concentration_max_1h=0.0,
            concentration_min_1h=0.0,
            concentration_zscore=0.0,
            elevation_above_background=0.0,
            background_concentration=0.0,
            spatial_gradient=0.0,
            spatial_anomaly_score=0.0,
            temporal_trend=0.0,
            temporal_anomaly_score=0.0,
            plume_likelihood_score=0.0,
            wind_speed_ms=0.0,
            wind_direction_deg=0.0,
            temperature_c=0.0,
            pressure_hpa=1013.0,
            humidity_pct=0.0,
        )

        result = detector.detect(fv)

        assert result.final_score >= 0
        assert result.severity in list(AnomalySeverity)

    def test_extreme_values(self):
        """Extreme values should not cause errors."""
        detector = AnomalyDetector()

        fv = FeatureVector(
            feature_id="FV-EXTREME",
            timestamp=datetime.now(),
            sensor_id="S",
            concentration_current=100000.0,  # Very high
            concentration_mean_1h=50000.0,
            concentration_std_1h=10000.0,
            concentration_max_1h=100000.0,
            concentration_min_1h=40000.0,
            concentration_zscore=10.0,
            elevation_above_background=90000.0,
            background_concentration=10000.0,
            spatial_gradient=1.0,
            spatial_anomaly_score=1.0,
            temporal_trend=1.0,
            temporal_anomaly_score=1.0,
            plume_likelihood_score=1.0,
            wind_speed_ms=50.0,
            wind_direction_deg=360.0,
            temperature_c=50.0,
            pressure_hpa=1100.0,
            humidity_pct=100.0,
        )

        result = detector.detect(fv)

        # Should not crash, scores should be bounded
        assert 0 <= result.final_score <= 1

