# -*- coding: utf-8 -*-
"""
Anomaly Detection for GL-013 PredictiveMaintenance Agent.

Provides anomaly detection using multiple methods:
- Isolation Forest (fast, unsupervised)
- Statistical methods (z-score, IQR)
- Autoencoder reconstruction error (deep learning)

All detections include severity scoring and explainability.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies."""
    POINT = "point"  # Single point anomaly
    CONTEXTUAL = "contextual"  # Anomalous in context
    COLLECTIVE = "collective"  # Group of points anomalous together


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.01  # Expected proportion of outliers
    threshold: float = 0.5  # Anomaly score threshold
    random_seed: int = 42
    n_estimators: int = 100
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    detection_id: str
    asset_id: str
    timestamp: datetime

    # Detection results
    is_anomaly: bool
    anomaly_score: float  # 0-1, higher = more anomalous
    severity: AnomalySeverity
    anomaly_type: AnomalyType

    # Details
    anomalous_features: Dict[str, float] = field(default_factory=dict)
    expected_values: Dict[str, float] = field(default_factory=dict)
    deviation_scores: Dict[str, float] = field(default_factory=dict)

    # Explanation
    explanation: str = ""
    contributing_factors: List[str] = field(default_factory=list)

    # Provenance
    detector_type: str = ""
    model_version: str = "1.0.0"
    provenance_hash: str = ""
    computation_time_ms: float = 0.0


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self._is_fitted = False
        self._feature_names: List[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit the anomaly detector."""
        pass

    @abstractmethod
    def detect(
        self,
        X: np.ndarray,
        asset_id: str = "",
    ) -> AnomalyResult:
        """Detect anomalies in the input."""
        pass

    def _compute_severity(self, score: float) -> AnomalySeverity:
        """Map anomaly score to severity level."""
        if score >= 0.9:
            return AnomalySeverity.CRITICAL
        elif score >= 0.75:
            return AnomalySeverity.HIGH
        elif score >= 0.5:
            return AnomalySeverity.MEDIUM
        elif score >= 0.25:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    def _generate_id(self) -> str:
        """Generate unique detection ID."""
        import uuid
        return f"anom_{uuid.uuid4().hex[:12]}"

    def _compute_provenance_hash(self, *args) -> str:
        """Compute provenance hash."""
        content = "|".join(str(a) for a in args)
        return hashlib.sha256(content.encode()).hexdigest()


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest based anomaly detector.

    Fast, unsupervised method that works well for high-dimensional data.
    Isolates anomalies by random partitioning.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for IsolationForestDetector")

        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._training_scores: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Fit Isolation Forest model.

        Args:
            X: Training data (n_samples, n_features)
            feature_names: Names of features
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit Isolation Forest
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)

        # Store training scores for reference
        self._training_scores = -self._model.score_samples(X_scaled)

        self._is_fitted = True

        logger.info(
            f"IsolationForestDetector fitted with {X.shape[0]} samples, "
            f"{X.shape[1]} features"
        )

    def detect(
        self,
        X: np.ndarray,
        asset_id: str = "",
    ) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest.

        Args:
            X: Input data (1D or 2D array)
            asset_id: Asset identifier

        Returns:
            AnomalyResult with detection details
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale features
        X_scaled = self._scaler.transform(X)

        # Get anomaly score (-1 to 1, where -1 is most anomalous)
        raw_score = -self._model.score_samples(X_scaled)[0]

        # Normalize to 0-1 range using training distribution
        if self._training_scores is not None:
            min_score = np.min(self._training_scores)
            max_score = np.max(self._training_scores)
            anomaly_score = (raw_score - min_score) / (max_score - min_score + 1e-10)
            anomaly_score = np.clip(anomaly_score, 0, 1)
        else:
            anomaly_score = (raw_score + 0.5)  # Rough normalization

        # Determine if anomaly
        is_anomaly = anomaly_score >= self.config.threshold

        # Find contributing features (deviation from mean)
        deviations = np.abs(X_scaled[0])
        top_features_idx = np.argsort(deviations)[::-1][:5]

        anomalous_features = {}
        expected_values = {}
        deviation_scores = {}

        for idx in top_features_idx:
            name = self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}"
            anomalous_features[name] = float(X[0, idx])
            expected_values[name] = float(self._scaler.mean_[idx])
            deviation_scores[name] = float(deviations[idx])

        # Generate explanation
        contributing = [name for name in list(deviation_scores.keys())[:3]]
        explanation = f"Anomaly detected based on unusual values in: {', '.join(contributing)}"

        severity = self._compute_severity(anomaly_score)
        computation_time = (time.time() - start_time) * 1000

        return AnomalyResult(
            detection_id=self._generate_id(),
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            severity=severity,
            anomaly_type=AnomalyType.POINT,
            anomalous_features=anomalous_features,
            expected_values=expected_values,
            deviation_scores=deviation_scores,
            explanation=explanation if is_anomaly else "No anomaly detected",
            contributing_factors=contributing,
            detector_type="isolation_forest",
            provenance_hash=self._compute_provenance_hash(
                asset_id, anomaly_score, datetime.utcnow().isoformat()
            ),
            computation_time_ms=computation_time,
        )


class StatisticalDetector(AnomalyDetector):
    """
    Statistical anomaly detector using z-score and IQR methods.

    Simple but effective for univariate and low-dimensional data.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        super().__init__(config)
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._q1: Optional[np.ndarray] = None
        self._q3: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Compute statistics from training data.

        Args:
            X: Training data
            feature_names: Names of features
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        self._means = np.mean(X, axis=0)
        self._stds = np.std(X, axis=0) + 1e-10
        self._q1 = np.percentile(X, 25, axis=0)
        self._q3 = np.percentile(X, 75, axis=0)

        self._is_fitted = True

        logger.info(f"StatisticalDetector fitted with {X.shape[0]} samples")

    def detect(
        self,
        X: np.ndarray,
        asset_id: str = "",
    ) -> AnomalyResult:
        """
        Detect anomalies using statistical methods.

        Args:
            X: Input data
            asset_id: Asset identifier

        Returns:
            AnomalyResult with detection details
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Z-score method
        z_scores = np.abs((X[0] - self._means) / self._stds)

        # IQR method
        iqr = self._q3 - self._q1
        iqr_lower = self._q1 - self.config.iqr_multiplier * iqr
        iqr_upper = self._q3 + self.config.iqr_multiplier * iqr
        iqr_outliers = (X[0] < iqr_lower) | (X[0] > iqr_upper)

        # Combine methods
        z_outliers = z_scores > self.config.z_score_threshold
        combined_outliers = z_outliers | iqr_outliers

        # Anomaly score based on proportion of anomalous features
        n_anomalous = np.sum(combined_outliers)
        n_features = len(combined_outliers)
        proportion_anomalous = n_anomalous / n_features

        # Weight by severity of deviations
        max_z = np.max(z_scores)
        anomaly_score = min(1.0, proportion_anomalous * 0.5 + (max_z / 10) * 0.5)

        is_anomaly = anomaly_score >= self.config.threshold or n_anomalous >= 2

        # Get anomalous features
        anomalous_features = {}
        expected_values = {}
        deviation_scores = {}

        for i in range(n_features):
            if combined_outliers[i]:
                name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                anomalous_features[name] = float(X[0, i])
                expected_values[name] = float(self._means[i])
                deviation_scores[name] = float(z_scores[i])

        severity = self._compute_severity(anomaly_score)
        computation_time = (time.time() - start_time) * 1000

        explanation = (
            f"Detected {n_anomalous} anomalous features out of {n_features}"
            if is_anomaly else "No statistical anomalies detected"
        )

        return AnomalyResult(
            detection_id=self._generate_id(),
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            severity=severity,
            anomaly_type=AnomalyType.POINT,
            anomalous_features=anomalous_features,
            expected_values=expected_values,
            deviation_scores=deviation_scores,
            explanation=explanation,
            contributing_factors=list(anomalous_features.keys()),
            detector_type="statistical",
            provenance_hash=self._compute_provenance_hash(
                asset_id, anomaly_score, datetime.utcnow().isoformat()
            ),
            computation_time_ms=computation_time,
        )


class AutoencoderDetector(AnomalyDetector):
    """
    Autoencoder-based anomaly detector.

    Uses reconstruction error as anomaly score.
    Placeholder implementation - requires TensorFlow/PyTorch for full version.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        super().__init__(config)
        self._encoder = None
        self._decoder = None
        self._threshold = 0.0
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Fit autoencoder model.

        For this placeholder, we use a simple linear reconstruction.
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        if SKLEARN_AVAILABLE:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            # Simple linear "autoencoder" using mean
            self._means = np.mean(X_scaled, axis=0)
            self._stds = np.std(X_scaled, axis=0) + 1e-10

            # Compute reconstruction error threshold from training data
            reconstruction_errors = np.mean((X_scaled - self._means) ** 2, axis=1)
            self._threshold = np.percentile(reconstruction_errors, 95)

        self._is_fitted = True
        logger.info("AutoencoderDetector fitted (placeholder implementation)")

    def detect(
        self,
        X: np.ndarray,
        asset_id: str = "",
    ) -> AnomalyResult:
        """
        Detect anomalies using reconstruction error.
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)

        # Compute reconstruction error
        reconstruction_error = np.mean((X_scaled - self._means) ** 2)

        # Normalize to 0-1 score
        anomaly_score = min(1.0, reconstruction_error / (2 * self._threshold))

        is_anomaly = reconstruction_error > self._threshold

        # Find features with highest reconstruction error
        feature_errors = (X_scaled[0] - self._means) ** 2
        top_idx = np.argsort(feature_errors)[::-1][:5]

        anomalous_features = {}
        expected_values = {}
        deviation_scores = {}

        for idx in top_idx:
            name = self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}"
            anomalous_features[name] = float(X[0, idx])
            expected_values[name] = float(self._scaler.mean_[idx])
            deviation_scores[name] = float(feature_errors[idx])

        severity = self._compute_severity(anomaly_score)
        computation_time = (time.time() - start_time) * 1000

        return AnomalyResult(
            detection_id=self._generate_id(),
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            severity=severity,
            anomaly_type=AnomalyType.POINT,
            anomalous_features=anomalous_features,
            expected_values=expected_values,
            deviation_scores=deviation_scores,
            explanation=f"Reconstruction error: {reconstruction_error:.4f}",
            contributing_factors=list(anomalous_features.keys())[:3],
            detector_type="autoencoder",
            provenance_hash=self._compute_provenance_hash(
                asset_id, anomaly_score, datetime.utcnow().isoformat()
            ),
            computation_time_ms=computation_time,
        )


class EnsembleAnomalyDetector(AnomalyDetector):
    """
    Ensemble anomaly detector combining multiple methods.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        super().__init__(config)
        self._detectors: List[AnomalyDetector] = []

        if SKLEARN_AVAILABLE:
            self._detectors.append(IsolationForestDetector(config))
        self._detectors.append(StatisticalDetector(config))

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit all detectors in the ensemble."""
        for detector in self._detectors:
            detector.fit(X, feature_names)

        self._is_fitted = True
        logger.info(f"EnsembleAnomalyDetector fitted with {len(self._detectors)} detectors")

    def detect(
        self,
        X: np.ndarray,
        asset_id: str = "",
    ) -> AnomalyResult:
        """Detect anomalies using ensemble voting."""
        start_time = time.time()

        results = [d.detect(X, asset_id) for d in self._detectors]

        # Aggregate scores
        scores = [r.anomaly_score for r in results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)

        # Voting for is_anomaly
        votes = [r.is_anomaly for r in results]
        majority_vote = sum(votes) > len(votes) / 2

        # Combine anomalous features
        combined_features = {}
        for r in results:
            combined_features.update(r.anomalous_features)

        severity = self._compute_severity(avg_score)
        computation_time = (time.time() - start_time) * 1000

        return AnomalyResult(
            detection_id=self._generate_id(),
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            is_anomaly=majority_vote,
            anomaly_score=float(avg_score),
            severity=severity,
            anomaly_type=AnomalyType.POINT,
            anomalous_features=combined_features,
            explanation=f"Ensemble detection: {sum(votes)}/{len(votes)} detectors flagged anomaly",
            detector_type="ensemble",
            provenance_hash=self._compute_provenance_hash(
                asset_id, avg_score, datetime.utcnow().isoformat()
            ),
            computation_time_ms=computation_time,
        )


def create_anomaly_detector(
    detector_type: str = "isolation_forest",
    config: Optional[AnomalyDetectorConfig] = None,
) -> AnomalyDetector:
    """
    Factory function to create anomaly detector.

    Args:
        detector_type: Type of detector (isolation_forest, statistical, autoencoder, ensemble)
        config: Configuration

    Returns:
        AnomalyDetector instance
    """
    config = config or AnomalyDetectorConfig()

    if detector_type == "isolation_forest":
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to statistical")
            return StatisticalDetector(config)
        return IsolationForestDetector(config)
    elif detector_type == "statistical":
        return StatisticalDetector(config)
    elif detector_type == "autoencoder":
        return AutoencoderDetector(config)
    elif detector_type == "ensemble":
        return EnsembleAnomalyDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
