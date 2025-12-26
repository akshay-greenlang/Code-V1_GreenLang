# -*- coding: utf-8 -*-
"""
Drift Detection for Fouling Prediction - GL-014 Exchangerpro Agent.

Provides drift monitoring:
- Feature drift detection (data distribution changes)
- Prediction drift detection (model output changes)
- Retraining triggers based on drift severity

Helps maintain model reliability by detecting when models
may need retraining due to changes in data distribution.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import numpy as np

try:
    from scipy import stats
    from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift."""
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONCEPT = "concept"
    LABEL = "label"


class DriftTestMethod(str, Enum):
    """Statistical test methods for drift detection."""
    KS_TEST = "kolmogorov_smirnov"
    WASSERSTEIN = "wasserstein"
    PSI = "population_stability_index"
    CHI_SQUARE = "chi_square"
    JENSEN_SHANNON = "jensen_shannon"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DriftDetectorConfig:
    """Configuration for drift detection."""

    # Window sizes
    reference_window_size: int = 1000  # Samples in reference distribution
    detection_window_size: int = 100  # Samples in detection window
    min_samples_for_detection: int = 50

    # Test methods
    primary_test_method: DriftTestMethod = DriftTestMethod.KS_TEST
    secondary_test_method: DriftTestMethod = DriftTestMethod.PSI

    # Thresholds
    p_value_threshold: float = 0.05  # For statistical tests
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    wasserstein_warning_threshold: float = 0.1
    wasserstein_critical_threshold: float = 0.2

    # Feature drift thresholds
    max_drifted_features_ratio: float = 0.3  # Trigger if >30% features drift

    # Prediction drift thresholds
    prediction_drift_threshold: float = 0.15

    # Retraining triggers
    enable_auto_retrain_trigger: bool = True
    consecutive_drift_threshold: int = 3  # Consecutive windows with drift
    drift_cooldown_hours: float = 24.0  # Min time between retrain triggers

    # Monitoring
    check_frequency_samples: int = 100  # Check every N samples


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class FeatureDrift:
    """Drift result for a single feature."""

    feature_name: str
    is_drifted: bool
    severity: DriftSeverity

    # Test statistics
    test_method: DriftTestMethod
    statistic: float
    p_value: float

    # Distribution metrics
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    mean_shift: float
    std_ratio: float

    # PSI (Population Stability Index)
    psi: Optional[float] = None

    # Wasserstein distance
    wasserstein: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "is_drifted": self.is_drifted,
            "severity": self.severity.value,
            "test_method": self.test_method.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "mean_shift": self.mean_shift,
            "psi": self.psi,
            "wasserstein": self.wasserstein,
        }


@dataclass
class PredictionDrift:
    """Drift result for predictions."""

    is_drifted: bool
    severity: DriftSeverity

    # Statistics
    test_method: DriftTestMethod
    statistic: float
    p_value: float

    # Distribution metrics
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    mean_shift_ratio: float

    # Quantile shifts
    quantile_shifts: Dict[float, float] = field(default_factory=dict)

    # PSI
    psi: Optional[float] = None


@dataclass
class RetrainingTrigger:
    """Retraining trigger information."""

    should_retrain: bool
    trigger_reason: str
    severity: DriftSeverity

    # Details
    drifted_features: List[str] = field(default_factory=list)
    feature_drift_ratio: float = 0.0
    prediction_drift_detected: bool = False
    consecutive_drift_windows: int = 0

    # Timing
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    last_retrain: Optional[datetime] = None
    time_since_last_retrain_hours: Optional[float] = None

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class DriftResult:
    """Complete drift detection result."""

    timestamp: datetime
    window_id: str

    # Overall drift status
    overall_drift_detected: bool
    overall_severity: DriftSeverity

    # Feature drift
    feature_drifts: Dict[str, FeatureDrift] = field(default_factory=dict)
    n_features_drifted: int = 0
    features_drifted_ratio: float = 0.0
    most_drifted_features: List[str] = field(default_factory=list)

    # Prediction drift
    prediction_drift: Optional[PredictionDrift] = None

    # Retraining trigger
    retrain_trigger: Optional[RetrainingTrigger] = None

    # Metadata
    reference_window_size: int = 0
    detection_window_size: int = 0
    computation_time_ms: float = 0.0
    provenance_hash: str = ""

    def get_summary(self) -> Dict[str, Any]:
        """Get drift summary for display."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_drift": self.overall_drift_detected,
            "severity": self.overall_severity.value,
            "features_drifted": self.n_features_drifted,
            "features_drifted_ratio": self.features_drifted_ratio,
            "most_drifted": self.most_drifted_features[:5],
            "should_retrain": self.retrain_trigger.should_retrain if self.retrain_trigger else False,
        }


# =============================================================================
# Statistical Tests
# =============================================================================

class StatisticalTests:
    """Collection of statistical tests for drift detection."""

    @staticmethod
    def kolmogorov_smirnov(
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference.

        Returns:
            Tuple of (statistic, p_value)
        """
        if not SCIPY_AVAILABLE:
            return 0.0, 1.0

        statistic, p_value = ks_2samp(reference, current)
        return float(statistic), float(p_value)

    @staticmethod
    def wasserstein(
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Wasserstein distance (Earth Mover's Distance).

        Returns:
            Wasserstein distance (lower = more similar)
        """
        if not SCIPY_AVAILABLE:
            return 0.0

        # Normalize for comparison
        ref_std = np.std(reference)
        if ref_std > 0:
            reference = reference / ref_std
            current = current / ref_std

        return float(wasserstein_distance(reference, current))

    @staticmethod
    def population_stability_index(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change

        Returns:
            PSI value
        """
        # Create bins from reference distribution
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Avoid division by zero
        ref_props = np.clip(ref_props, 1e-10, None)
        cur_props = np.clip(cur_props, 1e-10, None)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    @staticmethod
    def jensen_shannon_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Jensen-Shannon divergence (symmetric KL divergence).

        Returns:
            JS divergence (0 = identical, 1 = completely different)
        """
        # Create bins
        all_data = np.concatenate([reference, current])
        bins = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Avoid log(0)
        ref_props = np.clip(ref_props, 1e-10, None)
        cur_props = np.clip(cur_props, 1e-10, None)

        # Mixture distribution
        m = (ref_props + cur_props) / 2

        # KL divergences
        kl_ref = np.sum(ref_props * np.log(ref_props / m))
        kl_cur = np.sum(cur_props * np.log(cur_props / m))

        # JS divergence
        js = (kl_ref + kl_cur) / 2

        return float(js)


# =============================================================================
# Drift Detector
# =============================================================================

class DriftDetector:
    """
    Drift detection for fouling prediction models.

    Monitors:
    - Feature distribution drift
    - Prediction distribution drift
    - Concept drift indicators

    Provides retraining triggers based on drift severity.

    Example:
        >>> detector = DriftDetector(config)
        >>> detector.set_reference(X_train, y_pred_train)
        >>>
        >>> # During inference
        >>> for X, y_pred in stream:
        ...     detector.add_samples(X, y_pred)
        ...     result = detector.check_drift()
        ...     if result.retrain_trigger.should_retrain:
        ...         print("Retraining recommended!")
    """

    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        """
        Initialize drift detector.

        Args:
            config: Drift detection configuration
        """
        self.config = config or DriftDetectorConfig()

        # Reference distributions
        self._reference_features: Optional[np.ndarray] = None
        self._reference_predictions: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

        # Detection windows (sliding)
        self._current_features: deque = deque(maxlen=config.detection_window_size if config else 100)
        self._current_predictions: deque = deque(maxlen=config.detection_window_size if config else 100)

        # Statistics cache
        self._reference_stats: Dict[str, Dict[str, float]] = {}
        self._consecutive_drift_count: int = 0
        self._last_retrain_trigger: Optional[datetime] = None

        # History
        self._drift_history: List[DriftResult] = []
        self._window_counter: int = 0

        logger.info(
            f"DriftDetector initialized with reference_window={self.config.reference_window_size}, "
            f"detection_window={self.config.detection_window_size}"
        )

    def set_reference(
        self,
        X: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference distribution from training data.

        Args:
            X: Reference feature matrix
            predictions: Reference predictions (optional)
            feature_names: Names of features
        """
        self._reference_features = X.copy()
        self._reference_predictions = predictions.copy() if predictions is not None else None
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Pre-compute reference statistics
        self._reference_stats = {}
        for i, name in enumerate(self._feature_names):
            values = X[:, i]
            self._reference_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }

        logger.info(
            f"Reference set with {X.shape[0]} samples, {X.shape[1]} features"
        )

    def add_samples(
        self,
        X: np.ndarray,
        predictions: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add samples to detection window.

        Args:
            X: Feature matrix (can be single sample or batch)
            predictions: Predictions (optional)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for i in range(X.shape[0]):
            self._current_features.append(X[i])

            if predictions is not None:
                pred = predictions[i] if hasattr(predictions, '__getitem__') else predictions
                self._current_predictions.append(pred)

    def check_drift(
        self,
        force_check: bool = False,
    ) -> Optional[DriftResult]:
        """
        Check for drift in current window.

        Args:
            force_check: Force check even if window not full

        Returns:
            DriftResult if check performed, None otherwise
        """
        n_samples = len(self._current_features)

        # Check if we have enough samples
        if not force_check and n_samples < self.config.min_samples_for_detection:
            return None

        if self._reference_features is None:
            raise ValueError("Reference not set. Call set_reference() first.")

        start_time = time.time()
        self._window_counter += 1

        # Convert current window to array
        current_features = np.array(list(self._current_features))
        current_predictions = np.array(list(self._current_predictions)) if self._current_predictions else None

        # Detect feature drift
        feature_drifts = self._detect_feature_drift(current_features)

        # Detect prediction drift
        prediction_drift = None
        if current_predictions is not None and self._reference_predictions is not None:
            prediction_drift = self._detect_prediction_drift(current_predictions)

        # Compute overall drift
        drifted_features = [name for name, drift in feature_drifts.items() if drift.is_drifted]
        n_drifted = len(drifted_features)
        drift_ratio = n_drifted / len(self._feature_names) if self._feature_names else 0

        # Determine overall severity
        overall_severity = self._compute_overall_severity(feature_drifts, prediction_drift)
        overall_drift = overall_severity != DriftSeverity.NONE

        # Update consecutive drift count
        if overall_drift:
            self._consecutive_drift_count += 1
        else:
            self._consecutive_drift_count = 0

        # Check for retraining trigger
        retrain_trigger = self._evaluate_retrain_trigger(
            feature_drifts, prediction_drift, overall_severity
        )

        # Get most drifted features by severity
        most_drifted = sorted(
            [(name, drift) for name, drift in feature_drifts.items() if drift.is_drifted],
            key=lambda x: x[1].statistic,
            reverse=True
        )[:10]

        computation_time = (time.time() - start_time) * 1000

        result = DriftResult(
            timestamp=datetime.utcnow(),
            window_id=f"window_{self._window_counter}",
            overall_drift_detected=overall_drift,
            overall_severity=overall_severity,
            feature_drifts=feature_drifts,
            n_features_drifted=n_drifted,
            features_drifted_ratio=drift_ratio,
            most_drifted_features=[name for name, _ in most_drifted],
            prediction_drift=prediction_drift,
            retrain_trigger=retrain_trigger,
            reference_window_size=len(self._reference_features),
            detection_window_size=n_samples,
            computation_time_ms=computation_time,
            provenance_hash=self._compute_provenance_hash(overall_severity, n_drifted),
        )

        self._drift_history.append(result)

        if overall_drift:
            logger.warning(
                f"Drift detected: severity={overall_severity.value}, "
                f"features_drifted={n_drifted}/{len(self._feature_names)}"
            )

        return result

    def _detect_feature_drift(
        self,
        current_features: np.ndarray,
    ) -> Dict[str, FeatureDrift]:
        """Detect drift for each feature."""
        feature_drifts = {}

        for i, name in enumerate(self._feature_names):
            reference = self._reference_features[:, i]
            current = current_features[:, i]

            # Primary test (KS test)
            if self.config.primary_test_method == DriftTestMethod.KS_TEST:
                statistic, p_value = StatisticalTests.kolmogorov_smirnov(reference, current)
            else:
                statistic, p_value = 0.0, 1.0

            # PSI
            psi = StatisticalTests.population_stability_index(reference, current)

            # Wasserstein
            wasserstein = StatisticalTests.wasserstein(reference, current)

            # Compute distribution statistics
            ref_mean = self._reference_stats[name]["mean"]
            ref_std = self._reference_stats[name]["std"]
            cur_mean = float(np.mean(current))
            cur_std = float(np.std(current))

            mean_shift = cur_mean - ref_mean
            std_ratio = cur_std / max(ref_std, 1e-10)

            # Determine if drifted
            is_drifted = (
                p_value < self.config.p_value_threshold or
                psi >= self.config.psi_warning_threshold
            )

            # Determine severity
            if psi >= self.config.psi_critical_threshold:
                severity = DriftSeverity.CRITICAL
            elif psi >= self.config.psi_warning_threshold:
                severity = DriftSeverity.HIGH
            elif p_value < self.config.p_value_threshold:
                severity = DriftSeverity.MODERATE
            elif wasserstein >= self.config.wasserstein_warning_threshold:
                severity = DriftSeverity.LOW
            else:
                severity = DriftSeverity.NONE

            feature_drifts[name] = FeatureDrift(
                feature_name=name,
                is_drifted=is_drifted,
                severity=severity,
                test_method=self.config.primary_test_method,
                statistic=statistic,
                p_value=p_value,
                reference_mean=ref_mean,
                current_mean=cur_mean,
                reference_std=ref_std,
                current_std=cur_std,
                mean_shift=mean_shift,
                std_ratio=std_ratio,
                psi=psi,
                wasserstein=wasserstein,
            )

        return feature_drifts

    def _detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
    ) -> PredictionDrift:
        """Detect drift in predictions."""
        reference = self._reference_predictions

        # KS test
        statistic, p_value = StatisticalTests.kolmogorov_smirnov(reference, current_predictions)

        # PSI
        psi = StatisticalTests.population_stability_index(reference, current_predictions)

        # Distribution statistics
        ref_mean = float(np.mean(reference))
        ref_std = float(np.std(reference))
        cur_mean = float(np.mean(current_predictions))
        cur_std = float(np.std(current_predictions))

        mean_shift_ratio = abs(cur_mean - ref_mean) / max(ref_std, 1e-10)

        # Quantile shifts
        quantile_shifts = {}
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            ref_q = float(np.percentile(reference, q * 100))
            cur_q = float(np.percentile(current_predictions, q * 100))
            quantile_shifts[q] = cur_q - ref_q

        # Determine drift
        is_drifted = (
            p_value < self.config.p_value_threshold or
            mean_shift_ratio > self.config.prediction_drift_threshold
        )

        # Severity
        if psi >= self.config.psi_critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif psi >= self.config.psi_warning_threshold:
            severity = DriftSeverity.HIGH
        elif is_drifted:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.NONE

        return PredictionDrift(
            is_drifted=is_drifted,
            severity=severity,
            test_method=DriftTestMethod.KS_TEST,
            statistic=statistic,
            p_value=p_value,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            reference_std=ref_std,
            current_std=cur_std,
            mean_shift_ratio=mean_shift_ratio,
            quantile_shifts=quantile_shifts,
            psi=psi,
        )

    def _compute_overall_severity(
        self,
        feature_drifts: Dict[str, FeatureDrift],
        prediction_drift: Optional[PredictionDrift],
    ) -> DriftSeverity:
        """Compute overall drift severity."""
        # Get max feature severity
        feature_severities = [d.severity for d in feature_drifts.values()]
        max_feature_severity = max(feature_severities, default=DriftSeverity.NONE)

        # Get prediction severity
        pred_severity = prediction_drift.severity if prediction_drift else DriftSeverity.NONE

        # Check feature drift ratio
        n_drifted = sum(1 for d in feature_drifts.values() if d.is_drifted)
        drift_ratio = n_drifted / len(feature_drifts) if feature_drifts else 0

        # Combine severities
        if drift_ratio >= self.config.max_drifted_features_ratio:
            if max_feature_severity == DriftSeverity.CRITICAL:
                return DriftSeverity.CRITICAL
            elif max_feature_severity in [DriftSeverity.HIGH, DriftSeverity.MODERATE]:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.MODERATE

        if pred_severity == DriftSeverity.CRITICAL:
            return DriftSeverity.CRITICAL

        # Return max of feature and prediction severity
        severity_order = [
            DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MODERATE,
            DriftSeverity.HIGH, DriftSeverity.CRITICAL
        ]

        max_idx = max(
            severity_order.index(max_feature_severity),
            severity_order.index(pred_severity)
        )

        return severity_order[max_idx]

    def _evaluate_retrain_trigger(
        self,
        feature_drifts: Dict[str, FeatureDrift],
        prediction_drift: Optional[PredictionDrift],
        overall_severity: DriftSeverity,
    ) -> RetrainingTrigger:
        """Evaluate if retraining should be triggered."""
        drifted_features = [name for name, d in feature_drifts.items() if d.is_drifted]
        n_drifted = len(drifted_features)
        drift_ratio = n_drifted / len(feature_drifts) if feature_drifts else 0

        pred_drift = prediction_drift.is_drifted if prediction_drift else False

        # Check cooldown
        time_since_last = None
        if self._last_retrain_trigger:
            time_since_last = (datetime.utcnow() - self._last_retrain_trigger).total_seconds() / 3600
            in_cooldown = time_since_last < self.config.drift_cooldown_hours
        else:
            in_cooldown = False

        # Evaluate trigger conditions
        reasons = []
        should_retrain = False

        if overall_severity == DriftSeverity.CRITICAL:
            reasons.append("Critical drift severity detected")
            should_retrain = True

        if self._consecutive_drift_count >= self.config.consecutive_drift_threshold:
            reasons.append(f"Drift persisted for {self._consecutive_drift_count} consecutive windows")
            should_retrain = True

        if drift_ratio >= self.config.max_drifted_features_ratio and overall_severity in [DriftSeverity.HIGH, DriftSeverity.MODERATE]:
            reasons.append(f"{drift_ratio:.0%} of features drifted")
            should_retrain = True

        if pred_drift and prediction_drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            reasons.append("Significant prediction drift detected")
            should_retrain = True

        # Check cooldown
        if should_retrain and in_cooldown and overall_severity != DriftSeverity.CRITICAL:
            reasons.append(f"In cooldown period ({time_since_last:.1f}h since last trigger)")
            should_retrain = False

        # Build recommendations
        recommendations = []
        if should_retrain:
            recommendations.append("Collect new training data from recent period")
            recommendations.append("Validate model on recent holdout set before retraining")
            if n_drifted > 0:
                recommendations.append(f"Investigate drifted features: {', '.join(drifted_features[:5])}")

        trigger = RetrainingTrigger(
            should_retrain=should_retrain and self.config.enable_auto_retrain_trigger,
            trigger_reason="; ".join(reasons) if reasons else "No drift detected",
            severity=overall_severity,
            drifted_features=drifted_features,
            feature_drift_ratio=drift_ratio,
            prediction_drift_detected=pred_drift,
            consecutive_drift_windows=self._consecutive_drift_count,
            triggered_at=datetime.utcnow(),
            last_retrain=self._last_retrain_trigger,
            time_since_last_retrain_hours=time_since_last,
            recommended_actions=recommendations,
        )

        # Update last trigger time
        if trigger.should_retrain:
            self._last_retrain_trigger = datetime.utcnow()

        return trigger

    def _compute_provenance_hash(
        self,
        severity: DriftSeverity,
        n_drifted: int,
    ) -> str:
        """Compute provenance hash."""
        content = f"{self._window_counter}|{severity.value}|{n_drifted}|{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def reset_detection_window(self) -> None:
        """Clear current detection window."""
        self._current_features.clear()
        self._current_predictions.clear()
        self._consecutive_drift_count = 0

    def update_reference(
        self,
        X: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        merge: bool = False,
    ) -> None:
        """
        Update reference distribution.

        Args:
            X: New reference data
            predictions: New reference predictions
            merge: If True, merge with existing reference
        """
        if merge and self._reference_features is not None:
            # Merge and keep most recent
            max_size = self.config.reference_window_size
            combined = np.vstack([self._reference_features, X])
            self._reference_features = combined[-max_size:]

            if predictions is not None and self._reference_predictions is not None:
                combined_pred = np.concatenate([self._reference_predictions, predictions])
                self._reference_predictions = combined_pred[-max_size:]
        else:
            self.set_reference(X, predictions, self._feature_names)

        # Update statistics
        for i, name in enumerate(self._feature_names):
            values = self._reference_features[:, i]
            self._reference_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }

        logger.info(f"Reference updated with {self._reference_features.shape[0]} samples")

    def get_history(
        self,
        n_windows: Optional[int] = None,
    ) -> List[DriftResult]:
        """Get drift detection history."""
        if n_windows is None:
            return self._drift_history.copy()
        return self._drift_history[-n_windows:]

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get reference statistics for all features."""
        return self._reference_stats.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get drift detector status."""
        return {
            "is_initialized": self._reference_features is not None,
            "reference_size": len(self._reference_features) if self._reference_features is not None else 0,
            "current_window_size": len(self._current_features),
            "n_features": len(self._feature_names),
            "consecutive_drift_count": self._consecutive_drift_count,
            "last_retrain_trigger": self._last_retrain_trigger.isoformat() if self._last_retrain_trigger else None,
            "total_windows_checked": self._window_counter,
            "history_size": len(self._drift_history),
        }
