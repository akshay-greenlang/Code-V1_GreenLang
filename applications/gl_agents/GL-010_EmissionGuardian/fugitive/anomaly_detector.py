# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Fugitive Emissions Anomaly Detector

This module implements ML-based anomaly detection for fugitive emissions
using Isolation Forest and statistical methods with full explainability.

Supported Models:
    - Isolation Forest (unsupervised anomaly detection)
    - Statistical Z-score detection
    - Ensemble combination

Zero-Hallucination Principle:
    - ML used only for anomaly scoring, not compliance decisions
    - All thresholds are deterministic and configurable
    - Human review required for all alerts
    - Complete provenance tracking

Author: GreenLang GL-010 EmissionsGuardian
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import json

import numpy as np
from pydantic import BaseModel, Field

from .feature_engineering import FeatureVector, EquipmentType

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    CONCENTRATION_SPIKE = "concentration_spike"
    SUSTAINED_ELEVATION = "sustained_elevation"
    SPATIAL_GRADIENT = "spatial_gradient"
    PLUME_SIGNATURE = "plume_signature"
    EQUIPMENT_ANOMALY = "equipment_anomaly"
    TEMPORAL_PATTERN = "temporal_pattern"
    UNKNOWN = "unknown"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectorType(str, Enum):
    """Available anomaly detector types."""
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


class AnomalyDetectorConfig(BaseModel):
    """Configuration for anomaly detection."""
    # Detector selection
    detector_type: DetectorType = Field(default=DetectorType.ENSEMBLE)

    # Isolation Forest parameters
    if_n_estimators: int = Field(default=100, ge=10, le=500)
    if_contamination: float = Field(default=0.05, ge=0.01, le=0.3)
    if_max_samples: int = Field(default=256, ge=64, le=1024)
    if_random_state: int = Field(default=42)

    # Statistical thresholds
    zscore_threshold: float = Field(default=3.0, ge=2.0, le=5.0)
    concentration_spike_ppm: float = Field(default=500.0, ge=100.0, le=10000.0)
    elevation_threshold_ppm: float = Field(default=100.0, ge=10.0, le=1000.0)

    # Ensemble weights
    ensemble_if_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    ensemble_stat_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # Alert thresholds
    alert_threshold: float = Field(default=0.7, ge=0.5, le=0.99)
    critical_threshold: float = Field(default=0.9, ge=0.7, le=0.99)

    # Minimum samples for training
    min_training_samples: int = Field(default=100, ge=50, le=10000)


@dataclass
class AnomalyScore:
    """Anomaly score from a single detector."""
    detector_type: DetectorType
    score: float  # 0-1, higher = more anomalous
    raw_score: float  # Original detector output
    threshold_used: float
    is_anomaly: bool
    computation_time_ms: float


@dataclass
class AnomalyDetection:
    """Complete anomaly detection result."""
    detection_id: str
    feature_id: str
    timestamp: datetime

    # Scores
    final_score: float  # Combined score (0-1)
    isolation_forest_score: Optional[float] = None
    statistical_score: Optional[float] = None

    # Classification
    is_anomaly: bool = False
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    severity: AnomalySeverity = AnomalySeverity.LOW

    # Contributing factors
    top_contributing_features: List[Tuple[str, float]] = field(default_factory=list)

    # Context
    concentration_ppm: float = 0.0
    background_ppm: float = 0.0
    elevation_ppm: float = 0.0
    zscore: float = 0.0

    # Equipment context
    equipment_id: Optional[str] = None
    equipment_type: Optional[EquipmentType] = None

    # Human review
    requires_review: bool = True
    review_priority: int = 1  # 1=highest

    # Provenance
    model_version: str = "1.0.0"
    detector_config_hash: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "final_score": round(self.final_score, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "concentration_ppm": round(self.concentration_ppm, 2),
            "elevation_ppm": round(self.elevation_ppm, 2),
            "requires_review": self.requires_review,
            "review_priority": self.review_priority,
            "top_features": [
                {"feature": f, "contribution": round(c, 4)}
                for f, c in self.top_contributing_features[:5]
            ],
            "provenance_hash": self.provenance_hash,
        }


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.

    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are isolated closer to the root.
    """

    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self._model = None
        self._is_fitted = False
        self._training_samples = 0
        self._feature_names: List[str] = []

    def fit(self, feature_vectors: List[FeatureVector]) -> None:
        """Fit the Isolation Forest model."""
        if len(feature_vectors) < self.config.min_training_samples:
            logger.warning(
                f"Insufficient samples for training: {len(feature_vectors)} < "
                f"{self.config.min_training_samples}"
            )
            return

        # Convert to numpy array
        X = np.array([fv.to_array() for fv in feature_vectors])
        self._feature_names = FeatureVector.feature_names()

        # Import sklearn here to avoid hard dependency
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error("sklearn not available - using statistical fallback")
            return

        self._model = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.if_contamination,
            max_samples=min(self.config.if_max_samples, len(X)),
            random_state=self.config.if_random_state,
            n_jobs=-1
        )

        self._model.fit(X)
        self._is_fitted = True
        self._training_samples = len(feature_vectors)
        logger.info(f"Isolation Forest fitted on {len(X)} samples")

    def predict(self, feature_vector: FeatureVector) -> AnomalyScore:
        """Predict anomaly score for a feature vector."""
        import time
        start_time = time.time()

        if not self._is_fitted or self._model is None:
            # Return neutral score if not fitted
            return AnomalyScore(
                detector_type=DetectorType.ISOLATION_FOREST,
                score=0.5,
                raw_score=0.0,
                threshold_used=self.config.alert_threshold,
                is_anomaly=False,
                computation_time_ms=0.0
            )

        X = np.array([feature_vector.to_array()])

        # Get anomaly score (-1 for anomalies, 1 for normal)
        raw_score = self._model.decision_function(X)[0]

        # Convert to 0-1 scale (higher = more anomalous)
        # decision_function returns negative for anomalies
        normalized_score = 1.0 / (1.0 + np.exp(raw_score))

        is_anomaly = normalized_score >= self.config.alert_threshold

        computation_time = (time.time() - start_time) * 1000

        return AnomalyScore(
            detector_type=DetectorType.ISOLATION_FOREST,
            score=float(normalized_score),
            raw_score=float(raw_score),
            threshold_used=self.config.alert_threshold,
            is_anomaly=is_anomaly,
            computation_time_ms=computation_time
        )

    def get_feature_importance(
        self,
        feature_vector: FeatureVector
    ) -> List[Tuple[str, float]]:
        """
        Calculate approximate feature importance using permutation.

        This provides basic explainability for Isolation Forest predictions.
        """
        if not self._is_fitted or self._model is None:
            return []

        X = np.array([feature_vector.to_array()])
        base_score = self._model.decision_function(X)[0]

        importances: List[Tuple[str, float]] = []

        for i, name in enumerate(self._feature_names):
            # Permute feature
            X_permuted = X.copy()
            X_permuted[0, i] = 0.0  # Zero out feature

            permuted_score = self._model.decision_function(X_permuted)[0]
            importance = abs(base_score - permuted_score)
            importances.append((name, float(importance)))

        # Sort by importance
        importances.sort(key=lambda x: x[1], reverse=True)
        return importances


class StatisticalDetector:
    """
    Statistical anomaly detector using Z-scores and thresholds.

    Provides deterministic anomaly detection as a baseline/fallback.
    """

    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config

    def predict(self, feature_vector: FeatureVector) -> AnomalyScore:
        """Calculate statistical anomaly score."""
        import time
        start_time = time.time()

        scores: List[float] = []

        # Z-score based detection
        zscore_component = min(1.0, abs(feature_vector.concentration_zscore) / self.config.zscore_threshold)
        scores.append(zscore_component)

        # Concentration spike detection
        if feature_vector.concentration_current > self.config.concentration_spike_ppm:
            spike_component = min(1.0, feature_vector.concentration_current / (2 * self.config.concentration_spike_ppm))
            scores.append(spike_component)

        # Elevation above background
        if feature_vector.elevation_above_background > self.config.elevation_threshold_ppm:
            elevation_component = min(1.0, feature_vector.elevation_above_background / (2 * self.config.elevation_threshold_ppm))
            scores.append(elevation_component)

        # Spatial anomaly
        scores.append(feature_vector.spatial_anomaly_score)

        # Temporal anomaly
        scores.append(feature_vector.temporal_anomaly_score)

        # Plume likelihood
        scores.append(feature_vector.plume_likelihood_score)

        # Combine scores (weighted average favoring maximum)
        if scores:
            final_score = 0.7 * max(scores) + 0.3 * (sum(scores) / len(scores))
        else:
            final_score = 0.0

        is_anomaly = final_score >= self.config.alert_threshold

        computation_time = (time.time() - start_time) * 1000

        return AnomalyScore(
            detector_type=DetectorType.STATISTICAL,
            score=float(final_score),
            raw_score=float(final_score),
            threshold_used=self.config.alert_threshold,
            is_anomaly=is_anomaly,
            computation_time_ms=computation_time
        )

    def get_contributing_factors(
        self,
        feature_vector: FeatureVector
    ) -> List[Tuple[str, float]]:
        """Get contributing factors for statistical detection."""
        factors: List[Tuple[str, float]] = []

        # Z-score contribution
        zscore_contribution = abs(feature_vector.concentration_zscore) / self.config.zscore_threshold
        factors.append(("concentration_zscore", min(1.0, zscore_contribution)))

        # Elevation contribution
        elevation_contribution = feature_vector.elevation_above_background / self.config.elevation_threshold_ppm
        factors.append(("elevation_above_background", min(1.0, elevation_contribution)))

        # Spatial anomaly
        factors.append(("spatial_anomaly_score", feature_vector.spatial_anomaly_score))

        # Temporal anomaly
        factors.append(("temporal_anomaly_score", feature_vector.temporal_anomaly_score))

        # Plume likelihood
        factors.append(("plume_likelihood_score", feature_vector.plume_likelihood_score))

        # Current concentration
        conc_contribution = feature_vector.concentration_current / self.config.concentration_spike_ppm
        factors.append(("concentration_current", min(1.0, conc_contribution)))

        # Sort by contribution
        factors.sort(key=lambda x: x[1], reverse=True)
        return factors


class AnomalyDetector:
    """
    Ensemble Anomaly Detector for Fugitive Emissions.

    Combines Isolation Forest and statistical detection with
    configurable weights and full explainability.

    Zero-Hallucination: ML provides scores only; thresholds are deterministic.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self._if_detector = IsolationForestDetector(self.config)
        self._stat_detector = StatisticalDetector(self.config)
        self._detection_counter = 0

        # Calculate config hash for provenance
        config_str = json.dumps(self.config.model_dump(), sort_keys=True)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        logger.info(f"AnomalyDetector initialized: {self.config.detector_type.value}")

    def fit(self, feature_vectors: List[FeatureVector]) -> None:
        """Fit the anomaly detector models."""
        if self.config.detector_type in [DetectorType.ISOLATION_FOREST, DetectorType.ENSEMBLE]:
            self._if_detector.fit(feature_vectors)

    def detect(self, feature_vector: FeatureVector) -> AnomalyDetection:
        """
        Detect anomalies in a feature vector.

        Returns complete detection result with explainability.
        """
        self._detection_counter += 1
        detection_id = f"DET-{feature_vector.timestamp.strftime('%Y%m%d%H%M%S')}-{self._detection_counter:06d}"

        # Get scores from detectors
        if_score: Optional[AnomalyScore] = None
        stat_score: Optional[AnomalyScore] = None

        if self.config.detector_type in [DetectorType.ISOLATION_FOREST, DetectorType.ENSEMBLE]:
            if_score = self._if_detector.predict(feature_vector)

        if self.config.detector_type in [DetectorType.STATISTICAL, DetectorType.ENSEMBLE]:
            stat_score = self._stat_detector.predict(feature_vector)

        # Calculate final score
        final_score = self._calculate_final_score(if_score, stat_score)

        # Determine if anomaly
        is_anomaly = final_score >= self.config.alert_threshold

        # Classify anomaly type
        anomaly_type = self._classify_anomaly_type(feature_vector)

        # Determine severity
        severity = self._determine_severity(final_score, feature_vector)

        # Get contributing features
        contributing_features = self._get_contributing_features(feature_vector)

        # Calculate review priority
        review_priority = self._calculate_review_priority(severity, final_score)

        # Calculate provenance hash
        provenance_content = (
            f"{detection_id}|{feature_vector.feature_id}|{final_score}|{is_anomaly}"
        )
        provenance_hash = hashlib.sha256(provenance_content.encode()).hexdigest()

        return AnomalyDetection(
            detection_id=detection_id,
            feature_id=feature_vector.feature_id,
            timestamp=feature_vector.timestamp,
            final_score=final_score,
            isolation_forest_score=if_score.score if if_score else None,
            statistical_score=stat_score.score if stat_score else None,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            top_contributing_features=contributing_features,
            concentration_ppm=feature_vector.concentration_current,
            background_ppm=feature_vector.background_concentration,
            elevation_ppm=feature_vector.elevation_above_background,
            zscore=feature_vector.concentration_zscore,
            requires_review=is_anomaly,
            review_priority=review_priority,
            model_version="1.0.0",
            detector_config_hash=self._config_hash,
            provenance_hash=provenance_hash
        )

    def detect_batch(
        self,
        feature_vectors: List[FeatureVector]
    ) -> List[AnomalyDetection]:
        """Detect anomalies in a batch of feature vectors."""
        return [self.detect(fv) for fv in feature_vectors]

    def _calculate_final_score(
        self,
        if_score: Optional[AnomalyScore],
        stat_score: Optional[AnomalyScore]
    ) -> float:
        """Calculate weighted final score."""
        if self.config.detector_type == DetectorType.ISOLATION_FOREST:
            return if_score.score if if_score else 0.0

        if self.config.detector_type == DetectorType.STATISTICAL:
            return stat_score.score if stat_score else 0.0

        # Ensemble
        scores = []
        weights = []

        if if_score:
            scores.append(if_score.score)
            weights.append(self.config.ensemble_if_weight)

        if stat_score:
            scores.append(stat_score.score)
            weights.append(self.config.ensemble_stat_weight)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _classify_anomaly_type(self, feature_vector: FeatureVector) -> AnomalyType:
        """Classify the type of anomaly based on features."""
        # Check for concentration spike
        if feature_vector.concentration_zscore > 3.0:
            return AnomalyType.CONCENTRATION_SPIKE

        # Check for sustained elevation
        if feature_vector.elevation_above_background > self.config.elevation_threshold_ppm:
            return AnomalyType.SUSTAINED_ELEVATION

        # Check for spatial gradient
        if feature_vector.spatial_anomaly_score > 0.7:
            return AnomalyType.SPATIAL_GRADIENT

        # Check for plume signature
        if feature_vector.plume_likelihood_score > 0.7:
            return AnomalyType.PLUME_SIGNATURE

        # Check for temporal pattern
        if feature_vector.temporal_anomaly_score > 0.7:
            return AnomalyType.TEMPORAL_PATTERN

        return AnomalyType.UNKNOWN

    def _determine_severity(
        self,
        final_score: float,
        feature_vector: FeatureVector
    ) -> AnomalySeverity:
        """Determine anomaly severity."""
        if final_score >= self.config.critical_threshold:
            return AnomalySeverity.CRITICAL

        if final_score >= 0.85:
            return AnomalySeverity.HIGH

        if final_score >= self.config.alert_threshold:
            return AnomalySeverity.MEDIUM

        return AnomalySeverity.LOW

    def _get_contributing_features(
        self,
        feature_vector: FeatureVector
    ) -> List[Tuple[str, float]]:
        """Get top contributing features for explainability."""
        # Combine IF and statistical contributions
        contributions: Dict[str, float] = {}

        # Get IF contributions if available
        if_contributions = self._if_detector.get_feature_importance(feature_vector)
        for name, importance in if_contributions:
            contributions[name] = contributions.get(name, 0) + importance * self.config.ensemble_if_weight

        # Get statistical contributions
        stat_contributions = self._stat_detector.get_contributing_factors(feature_vector)
        for name, importance in stat_contributions:
            contributions[name] = contributions.get(name, 0) + importance * self.config.ensemble_stat_weight

        # Sort and return top contributors
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_contributions[:10]

    def _calculate_review_priority(
        self,
        severity: AnomalySeverity,
        score: float
    ) -> int:
        """Calculate review priority (1=highest)."""
        if severity == AnomalySeverity.CRITICAL:
            return 1
        if severity == AnomalySeverity.HIGH:
            return 2
        if severity == AnomalySeverity.MEDIUM:
            return 3
        return 4


# Export all public classes
__all__ = [
    "AnomalyType",
    "AnomalySeverity",
    "DetectorType",
    "AnomalyDetectorConfig",
    "AnomalyScore",
    "AnomalyDetection",
    "IsolationForestDetector",
    "StatisticalDetector",
    "AnomalyDetector",
]
