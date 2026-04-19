# -*- coding: utf-8 -*-
"""
TASK-069: Anomaly Detection for Predictions

This module provides anomaly detection capabilities for GreenLang Process Heat
ML model predictions, including Isolation Forest, One-class SVM, DBSCAN
clustering, Z-score based detection, and anomaly alerts with provenance.

Detecting anomalous predictions is critical for identifying potential model
failures, unusual operating conditions, or data quality issues in Process Heat
applications.

Example:
    >>> from greenlang.ml.robustness import PredictionAnomalyDetector
    >>> detector = PredictionAnomalyDetector(config=AnomalyDetectionConfig())
    >>> detector.fit(reference_predictions)
    >>> result = detector.detect(new_predictions)
    >>> if result.n_anomalies > 0:
    ...     for alert in result.alerts:
    ...         print(f"Anomaly at sample {alert.sample_index}: {alert.reason}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    DBSCAN = "dbscan"
    Z_SCORE = "z_score"
    MAD = "mad"  # Median Absolute Deviation
    ENSEMBLE = "ensemble"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of anomaly alerts."""
    OUTLIER = "outlier"
    CLUSTER_OUTLIER = "cluster_outlier"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    EXTREME_VALUE = "extreme_value"
    RATE_OF_CHANGE = "rate_of_change"


# =============================================================================
# Configuration
# =============================================================================

class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection."""

    # Detection methods
    methods: List[AnomalyMethod] = Field(
        default_factory=lambda: [
            AnomalyMethod.ISOLATION_FOREST,
            AnomalyMethod.Z_SCORE
        ],
        description="Anomaly detection methods to use"
    )

    # Isolation Forest parameters
    if_n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of trees in Isolation Forest"
    )
    if_contamination: float = Field(
        default=0.05,
        gt=0,
        lt=0.5,
        description="Expected anomaly fraction"
    )
    if_max_samples: Union[int, float, str] = Field(
        default="auto",
        description="Samples for training each tree"
    )

    # One-Class SVM parameters
    svm_kernel: str = Field(
        default="rbf",
        description="Kernel type (rbf, linear, poly)"
    )
    svm_nu: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Upper bound on training errors"
    )
    svm_gamma: str = Field(
        default="scale",
        description="Kernel coefficient"
    )

    # DBSCAN parameters
    dbscan_eps: float = Field(
        default=0.5,
        gt=0,
        description="Maximum distance for neighbors"
    )
    dbscan_min_samples: int = Field(
        default=5,
        ge=1,
        description="Minimum samples for core point"
    )

    # Z-score parameters
    z_score_threshold: float = Field(
        default=3.0,
        gt=0,
        description="Z-score threshold for anomaly"
    )

    # MAD parameters
    mad_threshold: float = Field(
        default=3.5,
        gt=0,
        description="Modified Z-score threshold"
    )

    # Ensemble settings
    ensemble_voting: str = Field(
        default="majority",
        description="Voting strategy: majority, any, all"
    )

    # Alert settings
    generate_alerts: bool = Field(
        default=True,
        description="Generate detailed alerts"
    )
    alert_threshold_pct: float = Field(
        default=5.0,
        gt=0,
        description="Alert if anomaly rate exceeds this %"
    )

    # General
    random_state: int = Field(
        default=42,
        description="Random seed"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )


# =============================================================================
# Result Models
# =============================================================================

class AnomalyAlert(BaseModel):
    """Alert for a detected anomaly."""

    sample_index: int = Field(..., description="Sample index")
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AnomalySeverity = Field(..., description="Severity")

    prediction_value: float = Field(..., description="Prediction value")
    anomaly_score: float = Field(..., description="Anomaly score")
    detection_methods: List[str] = Field(
        ...,
        description="Methods that detected this anomaly"
    )

    reason: str = Field(..., description="Human-readable reason")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Alert timestamp"
    )


class MethodResult(BaseModel):
    """Result from a single detection method."""

    method: str = Field(..., description="Method name")
    n_anomalies: int = Field(..., description="Number detected")
    anomaly_rate: float = Field(..., description="Anomaly rate")
    anomaly_indices: List[int] = Field(..., description="Anomaly indices")
    scores: Optional[List[float]] = Field(
        default=None,
        description="Anomaly scores (if available)"
    )
    threshold_used: Optional[float] = Field(
        default=None,
        description="Detection threshold"
    )


class AnomalyDetectionResult(BaseModel):
    """Comprehensive anomaly detection result."""

    # Summary
    n_samples: int = Field(..., description="Total samples analyzed")
    n_anomalies: int = Field(..., description="Anomalies detected")
    anomaly_rate: float = Field(..., description="Overall anomaly rate")
    anomaly_indices: List[int] = Field(..., description="Indices of anomalies")

    # Per-method results
    method_results: List[MethodResult] = Field(
        ...,
        description="Results per method"
    )

    # Alerts
    alerts: List[AnomalyAlert] = Field(
        default_factory=list,
        description="Generated alerts"
    )
    alert_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert count by severity"
    )

    # Scores
    ensemble_scores: Optional[List[float]] = Field(
        default=None,
        description="Combined anomaly scores"
    )

    # Statistics
    prediction_mean: float = Field(..., description="Mean of predictions")
    prediction_std: float = Field(..., description="Std of predictions")
    anomaly_mean: Optional[float] = Field(
        default=None,
        description="Mean of anomalous predictions"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )


# =============================================================================
# Anomaly Detection Implementations
# =============================================================================

class IsolationForestDetector:
    """
    Isolation Forest implementation for anomaly detection.

    Isolation Forest isolates anomalies by randomly selecting a feature
    and randomly selecting a split value. Anomalies require fewer splits
    to isolate, resulting in shorter path lengths.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self._rng = np.random.RandomState(random_state)
        self._trees: List[Dict] = []
        self._threshold: float = 0.0
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit Isolation Forest to normal data."""
        n_samples, n_features = X.shape

        # Determine max samples
        if self.max_samples == "auto":
            max_samples = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = min(self.max_samples, n_samples)

        # Build trees
        self._trees = []
        for _ in range(self.n_estimators):
            # Sample data
            sample_idx = self._rng.choice(n_samples, max_samples, replace=False)
            X_sample = X[sample_idx]

            # Build tree
            tree = self._build_tree(X_sample, 0, int(np.ceil(np.log2(max_samples))))
            self._trees.append(tree)

        # Calculate threshold from training data
        scores = self._score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True
        return self

    def _build_tree(
        self,
        X: np.ndarray,
        current_depth: int,
        max_depth: int
    ) -> Dict:
        """Build a single isolation tree recursively."""
        n_samples, n_features = X.shape

        # Terminal conditions
        if current_depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Random feature and split
        feature = self._rng.randint(n_features)
        feature_min = X[:, feature].min()
        feature_max = X[:, feature].max()

        if feature_min == feature_max:
            return {"type": "leaf", "size": n_samples}

        split_value = self._rng.uniform(feature_min, feature_max)

        # Split data
        left_mask = X[:, feature] < split_value
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feature,
            "split_value": split_value,
            "left": self._build_tree(X[left_mask], current_depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], current_depth + 1, max_depth)
        }

    def _path_length(self, x: np.ndarray, tree: Dict, current_depth: int = 0) -> float:
        """Calculate path length for a sample."""
        if tree["type"] == "leaf":
            # Add expected path length for remaining samples
            n = tree["size"]
            if n <= 1:
                return current_depth
            else:
                # Average path length formula
                c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
                return current_depth + c

        if x[tree["feature"]] < tree["split_value"]:
            return self._path_length(x, tree["left"], current_depth + 1)
        else:
            return self._path_length(x, tree["right"], current_depth + 1)

    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores (higher = more anomalous)."""
        n_samples = len(X)
        path_lengths = np.zeros(n_samples)

        for tree in self._trees:
            for i in range(n_samples):
                path_lengths[i] += self._path_length(X[i], tree)

        # Average path length
        avg_path = path_lengths / self.n_estimators

        # Normalize by expected path length
        n = len(X)
        c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 1 else 1

        # Anomaly score
        scores = 2 ** (-avg_path / c)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        scores = self._score_samples(X)
        return np.where(scores > self._threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._score_samples(X)


class OneClassSVMDetector:
    """
    One-Class SVM implementation for anomaly detection.

    Learns a decision boundary around normal data. Points outside
    the boundary are classified as anomalies.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
        random_state: int = 42
    ):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self._rng = np.random.RandomState(random_state)
        self._support_vectors: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._rho: float = 0.0
        self._gamma_value: float = 1.0
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """Fit One-Class SVM to normal data."""
        n_samples, n_features = X.shape

        # Set gamma
        if self.gamma == "scale":
            self._gamma_value = 1.0 / (n_features * X.var())
        elif self.gamma == "auto":
            self._gamma_value = 1.0 / n_features
        else:
            self._gamma_value = float(self.gamma)

        # Simplified: use all points as support vectors
        # In practice, would solve QP optimization
        self._support_vectors = X.copy()

        # Simplified alpha (uniform weights)
        n_sv = min(int(self.nu * n_samples) + 1, n_samples)
        self._alpha = np.zeros(n_samples)
        self._alpha[:n_sv] = 1.0 / n_sv

        # Calculate rho (threshold)
        K = self._kernel_matrix(X, self._support_vectors)
        decision_values = np.dot(K, self._alpha)
        self._rho = np.percentile(decision_values, 100 * self.nu)

        self._is_fitted = True
        return self

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            sq_dist = np.sum((X[:, None] - Y[None, :]) ** 2, axis=2)
            return np.exp(-self._gamma_value * sq_dist)
        elif self.kernel == "linear":
            return np.dot(X, Y.T)
        else:
            # Default to RBF
            sq_dist = np.sum((X[:, None] - Y[None, :]) ** 2, axis=2)
            return np.exp(-self._gamma_value * sq_dist)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        K = self._kernel_matrix(X, self._support_vectors)
        decision_values = np.dot(K, self._alpha) - self._rho

        return np.where(decision_values < 0, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get decision function values (more negative = more anomalous)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        K = self._kernel_matrix(X, self._support_vectors)
        return np.dot(K, self._alpha) - self._rho


class DBSCANDetector:
    """
    DBSCAN implementation for anomaly detection.

    Points not belonging to any cluster (noise points) are anomalies.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5
    ):
        self.eps = eps
        self.min_samples = min_samples
        self._reference_data: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        """Fit DBSCAN to establish reference clusters."""
        self._reference_data = X.copy()
        self._is_fitted = True
        return self

    def _distance_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        return np.sqrt(np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly/noise, 1 for normal/clustered)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Combine reference and new data
        combined = np.vstack([self._reference_data, X])
        n_ref = len(self._reference_data)
        n_total = len(combined)

        # Distance matrix
        distances = self._distance_matrix(combined, combined)

        # Find neighbors within eps
        neighbors = [np.where(distances[i] <= self.eps)[0] for i in range(n_total)]

        # Identify core points
        is_core = np.array([len(n) >= self.min_samples for n in neighbors])

        # Cluster assignment (-1 = noise)
        labels = np.full(n_total, -1)
        cluster_id = 0

        for i in range(n_total):
            if labels[i] != -1 or not is_core[i]:
                continue

            # Start new cluster
            stack = [i]
            while stack:
                point = stack.pop()
                if labels[point] == -1:
                    labels[point] = cluster_id
                    if is_core[point]:
                        for neighbor in neighbors[point]:
                            if labels[neighbor] == -1:
                                stack.append(neighbor)

            cluster_id += 1

        # Get labels for new data only
        new_labels = labels[n_ref:]

        # Convert to prediction format
        return np.where(new_labels == -1, -1, 1)


class ZScoreDetector:
    """Z-score based anomaly detection."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        """Fit to reference data."""
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std = np.where(self._std == 0, 1e-10, self._std)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        z_scores = np.abs((X - self._mean) / self._std)

        # Anomaly if any feature exceeds threshold
        is_anomaly = np.any(z_scores > self.threshold, axis=1)

        return np.where(is_anomaly, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get max z-score per sample."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        z_scores = np.abs((X - self._mean) / self._std)
        return np.max(z_scores, axis=1)


class MADDetector:
    """Median Absolute Deviation based anomaly detection."""

    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
        self._median: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "MADDetector":
        """Fit to reference data."""
        self._median = np.median(X, axis=0)
        self._mad = np.median(np.abs(X - self._median), axis=0)
        self._mad = np.where(self._mad == 0, 1e-10, self._mad)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Modified Z-score
        modified_z = 0.6745 * (X - self._median) / self._mad

        is_anomaly = np.any(np.abs(modified_z) > self.threshold, axis=1)

        return np.where(is_anomaly, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get max modified z-score per sample."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        modified_z = 0.6745 * np.abs(X - self._median) / self._mad
        return np.max(modified_z, axis=1)


# =============================================================================
# Main Anomaly Detector
# =============================================================================

class PredictionAnomalyDetector:
    """
    Comprehensive Prediction Anomaly Detector for Process Heat ML.

    This detector identifies anomalous predictions using multiple methods:
    - Isolation Forest: Efficient for high-dimensional data
    - One-Class SVM: Kernel-based boundary learning
    - DBSCAN: Clustering-based outlier detection
    - Z-score: Statistical outlier detection
    - MAD: Robust to outliers in training data

    All calculations are deterministic for reproducibility.

    Attributes:
        config: Detection configuration
        _detectors: Fitted detector instances

    Example:
        >>> detector = PredictionAnomalyDetector(
        ...     config=AnomalyDetectionConfig(
        ...         methods=[AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.Z_SCORE]
        ...     )
        ... )
        >>> detector.fit(normal_predictions)
        >>> result = detector.detect(new_predictions)
        >>> print(f"Found {result.n_anomalies} anomalies ({result.anomaly_rate:.1%})")
    """

    def __init__(
        self,
        config: Optional[AnomalyDetectionConfig] = None
    ):
        """
        Initialize anomaly detector.

        Args:
            config: Detection configuration
        """
        self.config = config or AnomalyDetectionConfig()
        self._detectors: Dict[str, Any] = {}
        self._is_fitted = False
        self._reference_stats: Dict[str, float] = {}

        logger.info(
            f"PredictionAnomalyDetector initialized: "
            f"methods={[m.value for m in self.config.methods]}"
        )

    def fit(
        self,
        predictions: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "PredictionAnomalyDetector":
        """
        Fit detector to reference (normal) predictions.

        Args:
            predictions: Reference predictions
            feature_names: Optional feature names

        Returns:
            Self for chaining
        """
        predictions = np.atleast_2d(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Store reference statistics
        self._reference_stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions))
        }

        # Fit each detector
        for method in self.config.methods:
            if method == AnomalyMethod.ISOLATION_FOREST:
                detector = IsolationForestDetector(
                    n_estimators=self.config.if_n_estimators,
                    contamination=self.config.if_contamination,
                    max_samples=self.config.if_max_samples,
                    random_state=self.config.random_state
                )
                detector.fit(predictions)
                self._detectors["isolation_forest"] = detector

            elif method == AnomalyMethod.ONE_CLASS_SVM:
                detector = OneClassSVMDetector(
                    kernel=self.config.svm_kernel,
                    nu=self.config.svm_nu,
                    gamma=self.config.svm_gamma,
                    random_state=self.config.random_state
                )
                detector.fit(predictions)
                self._detectors["one_class_svm"] = detector

            elif method == AnomalyMethod.DBSCAN:
                detector = DBSCANDetector(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples
                )
                detector.fit(predictions)
                self._detectors["dbscan"] = detector

            elif method == AnomalyMethod.Z_SCORE:
                detector = ZScoreDetector(
                    threshold=self.config.z_score_threshold
                )
                detector.fit(predictions)
                self._detectors["z_score"] = detector

            elif method == AnomalyMethod.MAD:
                detector = MADDetector(
                    threshold=self.config.mad_threshold
                )
                detector.fit(predictions)
                self._detectors["mad"] = detector

        self._is_fitted = True
        logger.info(f"Fitted {len(self._detectors)} detectors")

        return self

    def detect(
        self,
        predictions: np.ndarray,
        sample_indices: Optional[np.ndarray] = None
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in predictions.

        Args:
            predictions: Predictions to analyze
            sample_indices: Optional sample indices for alerts

        Returns:
            Comprehensive detection result

        Example:
            >>> result = detector.detect(predictions)
            >>> for i in result.anomaly_indices:
            ...     print(f"Anomaly at sample {i}: {predictions[i]}")
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        predictions = np.atleast_2d(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        n_samples = len(predictions)

        if sample_indices is None:
            sample_indices = np.arange(n_samples)

        # Run each detector
        method_results = []
        all_predictions = {}
        all_scores = {}

        for method_name, detector in self._detectors.items():
            preds = detector.predict(predictions)
            anomaly_mask = preds == -1
            anomaly_indices = np.where(anomaly_mask)[0].tolist()

            # Get scores if available
            scores = None
            if hasattr(detector, "score_samples"):
                scores = detector.score_samples(predictions).tolist()

            method_results.append(MethodResult(
                method=method_name,
                n_anomalies=int(np.sum(anomaly_mask)),
                anomaly_rate=float(np.mean(anomaly_mask)),
                anomaly_indices=anomaly_indices,
                scores=scores,
                threshold_used=getattr(detector, "_threshold", None) or getattr(detector, "threshold", None)
            ))

            all_predictions[method_name] = anomaly_mask
            if scores:
                all_scores[method_name] = scores

        # Ensemble voting
        if len(all_predictions) > 1:
            prediction_matrix = np.column_stack(list(all_predictions.values()))

            if self.config.ensemble_voting == "majority":
                ensemble_anomaly = np.sum(prediction_matrix, axis=1) > len(all_predictions) / 2
            elif self.config.ensemble_voting == "any":
                ensemble_anomaly = np.any(prediction_matrix, axis=1)
            else:  # all
                ensemble_anomaly = np.all(prediction_matrix, axis=1)
        else:
            ensemble_anomaly = list(all_predictions.values())[0]

        anomaly_indices = np.where(ensemble_anomaly)[0].tolist()
        n_anomalies = len(anomaly_indices)
        anomaly_rate = n_anomalies / n_samples

        # Calculate ensemble scores
        ensemble_scores = None
        if all_scores:
            score_matrix = np.column_stack(list(all_scores.values()))
            ensemble_scores = np.mean(score_matrix, axis=1).tolist()

        # Generate alerts
        alerts = []
        if self.config.generate_alerts and n_anomalies > 0:
            alerts = self._generate_alerts(
                predictions, anomaly_indices, sample_indices, all_predictions
            )

        alert_summary = {}
        for alert in alerts:
            sev = alert.severity.value
            alert_summary[sev] = alert_summary.get(sev, 0) + 1

        # Statistics
        prediction_mean = float(np.mean(predictions))
        prediction_std = float(np.std(predictions))
        anomaly_mean = (
            float(np.mean(predictions[ensemble_anomaly]))
            if n_anomalies > 0 else None
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            anomaly_rate, method_results, alerts
        )

        # Provenance
        provenance_hash = self._calculate_provenance(n_samples, n_anomalies)

        result = AnomalyDetectionResult(
            n_samples=n_samples,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            anomaly_indices=anomaly_indices,
            method_results=method_results,
            alerts=alerts,
            alert_summary=alert_summary,
            ensemble_scores=ensemble_scores,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            anomaly_mean=anomaly_mean,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

        logger.info(
            f"Anomaly detection complete: {n_anomalies}/{n_samples} anomalies "
            f"({anomaly_rate:.1%})"
        )

        return result

    def _generate_alerts(
        self,
        predictions: np.ndarray,
        anomaly_indices: List[int],
        sample_indices: np.ndarray,
        method_predictions: Dict[str, np.ndarray]
    ) -> List[AnomalyAlert]:
        """Generate detailed alerts for anomalies."""
        alerts = []

        for idx in anomaly_indices:
            pred_value = float(predictions[idx].flatten()[0])

            # Determine which methods detected this
            detection_methods = [
                name for name, preds in method_predictions.items()
                if preds[idx]
            ]

            # Calculate anomaly score
            if hasattr(self._detectors.get("isolation_forest"), "score_samples"):
                score = float(self._detectors["isolation_forest"].score_samples(
                    predictions[idx:idx+1]
                )[0])
            else:
                # Z-score based
                z = abs(pred_value - self._reference_stats["mean"]) / (
                    self._reference_stats["std"] + 1e-10
                )
                score = float(z)

            # Determine severity
            if score > 5 or abs(pred_value) > 10 * self._reference_stats["std"]:
                severity = AnomalySeverity.CRITICAL
            elif score > 3:
                severity = AnomalySeverity.HIGH
            elif score > 2:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW

            # Determine alert type
            if pred_value > self._reference_stats["max"] * 1.5:
                alert_type = AlertType.EXTREME_VALUE
                reason = f"Value {pred_value:.4f} exceeds reference max by >50%"
            elif pred_value < self._reference_stats["min"] * 0.5:
                alert_type = AlertType.EXTREME_VALUE
                reason = f"Value {pred_value:.4f} below reference min by >50%"
            else:
                alert_type = AlertType.OUTLIER
                reason = f"Value {pred_value:.4f} detected as outlier by {len(detection_methods)} method(s)"

            alerts.append(AnomalyAlert(
                sample_index=int(sample_indices[idx]),
                alert_type=alert_type,
                severity=severity,
                prediction_value=pred_value,
                anomaly_score=score,
                detection_methods=detection_methods,
                reason=reason,
                context={
                    "reference_mean": self._reference_stats["mean"],
                    "reference_std": self._reference_stats["std"]
                }
            ))

        return alerts

    def _generate_recommendations(
        self,
        anomaly_rate: float,
        method_results: List[MethodResult],
        alerts: List[AnomalyAlert]
    ) -> List[str]:
        """Generate recommendations based on detection results."""
        recommendations = []

        if anomaly_rate > self.config.alert_threshold_pct / 100:
            recommendations.append(
                f"High anomaly rate ({anomaly_rate:.1%}) detected. "
                "Investigate data quality or model performance."
            )

        critical_alerts = [a for a in alerts if a.severity == AnomalySeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(
                f"{len(critical_alerts)} CRITICAL anomalies detected. "
                "Immediate investigation recommended."
            )

        # Check for method disagreement
        detection_rates = [r.anomaly_rate for r in method_results]
        if max(detection_rates) > 2 * min(detection_rates) if detection_rates else False:
            recommendations.append(
                "Significant disagreement between detection methods. "
                "Consider tuning detection thresholds."
            )

        if not recommendations:
            recommendations.append(
                "Anomaly detection complete. No immediate action required."
            )

        return recommendations

    def _calculate_provenance(
        self,
        n_samples: int,
        n_anomalies: int
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        provenance_data = (
            f"{n_samples}|{n_anomalies}|"
            f"{len(self._detectors)}|"
            f"{self.config.random_state}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()


# =============================================================================
# Unit Tests
# =============================================================================

class TestPredictionAnomalyDetector:
    """Unit tests for PredictionAnomalyDetector."""

    def test_isolation_forest_detector(self):
        """Test Isolation Forest implementation."""
        detector = IsolationForestDetector(n_estimators=50, contamination=0.1)

        X_normal = np.random.randn(200, 2)
        X_anomaly = np.random.randn(20, 2) + 5

        detector.fit(X_normal)

        normal_preds = detector.predict(X_normal)
        anomaly_preds = detector.predict(X_anomaly)

        # Most normal should be labeled 1
        assert np.mean(normal_preds == 1) > 0.8
        # Most anomalies should be labeled -1
        assert np.mean(anomaly_preds == -1) > 0.5

    def test_z_score_detector(self):
        """Test Z-score detector."""
        detector = ZScoreDetector(threshold=3.0)

        X_normal = np.random.randn(100, 1)
        detector.fit(X_normal)

        # Normal values should pass
        X_test_normal = np.random.randn(10, 1)
        preds_normal = detector.predict(X_test_normal)
        assert np.mean(preds_normal == 1) > 0.9

        # Extreme values should be flagged
        X_test_extreme = np.array([[10.0], [-10.0]])
        preds_extreme = detector.predict(X_test_extreme)
        assert np.all(preds_extreme == -1)

    def test_full_detector(self):
        """Test full anomaly detector."""
        config = AnomalyDetectionConfig(
            methods=[AnomalyMethod.Z_SCORE, AnomalyMethod.MAD],
            z_score_threshold=3.0,
            mad_threshold=3.5
        )
        detector = PredictionAnomalyDetector(config)

        # Fit on normal data
        X_normal = np.random.randn(200, 1)
        detector.fit(X_normal)

        # Detect on mixed data
        X_test = np.vstack([
            np.random.randn(50, 1),
            np.random.randn(10, 1) + 10  # Anomalies
        ])

        result = detector.detect(X_test)

        assert result.n_samples == 60
        assert result.n_anomalies > 0
        assert result.anomaly_rate > 0
        assert len(result.method_results) == 2

    def test_alert_generation(self):
        """Test alert generation."""
        config = AnomalyDetectionConfig(
            methods=[AnomalyMethod.Z_SCORE],
            generate_alerts=True
        )
        detector = PredictionAnomalyDetector(config)

        X_normal = np.random.randn(100, 1)
        detector.fit(X_normal)

        X_test = np.array([[100.0]])  # Extreme anomaly
        result = detector.detect(X_test)

        assert len(result.alerts) > 0
        assert result.alerts[0].severity in [
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL
        ]

    def test_provenance_deterministic(self):
        """Test provenance is deterministic."""
        detector = PredictionAnomalyDetector()

        hash1 = detector._calculate_provenance(100, 5)
        hash2 = detector._calculate_provenance(100, 5)

        assert hash1 == hash2
        assert len(hash1) == 64
