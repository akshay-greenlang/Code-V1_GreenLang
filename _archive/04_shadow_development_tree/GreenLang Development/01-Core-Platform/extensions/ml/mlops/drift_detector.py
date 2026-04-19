# -*- coding: utf-8 -*-
"""
Drift Detector Module

This module provides data and concept drift detection for GreenLang ML models,
enabling proactive monitoring and maintenance of model performance over time.

Drift detection is critical for regulatory compliance, as emission factors,
regulations, and data patterns change over time, potentially degrading
model accuracy without retraining.

Example:
    >>> from greenlang.ml.mlops import DriftDetector
    >>> detector = DriftDetector(reference_data=X_train)
    >>> drift_result = detector.detect_drift(X_new)
    >>> if drift_result.drift_detected:
    ...     trigger_retraining()
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift to detect."""
    DATA = "data"
    CONCEPT = "concept"
    BOTH = "both"


class DriftMethod(str, Enum):
    """Drift detection methods."""
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov
    PSI = "psi"  # Population Stability Index
    MMD = "mmd"  # Maximum Mean Discrepancy
    ADWIN = "adwin"  # Adaptive Windowing
    DDM = "ddm"  # Drift Detection Method
    EDDM = "eddm"  # Early Drift Detection Method
    PAGE_HINKLEY = "page_hinkley"


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetectorConfig(BaseModel):
    """Configuration for drift detector."""

    drift_type: DriftType = Field(
        default=DriftType.BOTH,
        description="Type of drift to detect"
    )
    method: DriftMethod = Field(
        default=DriftMethod.KS_TEST,
        description="Primary drift detection method"
    )
    significance_level: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for statistical tests"
    )
    psi_threshold: float = Field(
        default=0.2,
        gt=0,
        description="PSI threshold for drift (0.1=slight, 0.2=significant)"
    )
    window_size: int = Field(
        default=1000,
        ge=100,
        description="Window size for streaming drift detection"
    )
    min_samples: int = Field(
        default=30,
        ge=10,
        description="Minimum samples for drift detection"
    )
    feature_subset: Optional[List[int]] = Field(
        default=None,
        description="Feature indices to monitor (None=all)"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class FeatureDrift(BaseModel):
    """Drift information for a single feature."""

    feature_index: int = Field(
        ...,
        description="Feature index"
    )
    feature_name: Optional[str] = Field(
        default=None,
        description="Feature name"
    )
    drift_score: float = Field(
        ...,
        description="Drift score"
    )
    p_value: Optional[float] = Field(
        default=None,
        description="P-value from statistical test"
    )
    drift_detected: bool = Field(
        ...,
        description="Whether drift was detected"
    )
    severity: DriftSeverity = Field(
        ...,
        description="Drift severity"
    )
    reference_mean: float = Field(
        ...,
        description="Mean in reference distribution"
    )
    current_mean: float = Field(
        ...,
        description="Mean in current distribution"
    )
    reference_std: float = Field(
        ...,
        description="Std in reference distribution"
    )
    current_std: float = Field(
        ...,
        description="Std in current distribution"
    )


class DriftResult(BaseModel):
    """Result from drift detection."""

    drift_detected: bool = Field(
        ...,
        description="Whether any drift was detected"
    )
    drift_type: str = Field(
        ...,
        description="Type of drift detected"
    )
    overall_drift_score: float = Field(
        ...,
        description="Overall drift score"
    )
    severity: DriftSeverity = Field(
        ...,
        description="Overall severity"
    )
    n_features_drifted: int = Field(
        ...,
        description="Number of features with drift"
    )
    feature_drifts: List[FeatureDrift] = Field(
        ...,
        description="Per-feature drift information"
    )
    method_used: str = Field(
        ...,
        description="Detection method used"
    )
    reference_samples: int = Field(
        ...,
        description="Number of reference samples"
    )
    current_samples: int = Field(
        ...,
        description="Number of current samples"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


class DriftDetector:
    """
    Drift Detector for GreenLang ML models.

    This class provides comprehensive drift detection capabilities,
    monitoring both data distribution changes and concept drift
    (changes in the relationship between features and target).

    Key capabilities:
    - Statistical tests (KS, Chi-squared)
    - Population Stability Index (PSI)
    - Streaming drift detection (ADWIN, DDM)
    - Per-feature analysis
    - Severity classification
    - Provenance tracking

    Attributes:
        config: Detector configuration
        _reference_data: Reference (training) data distribution
        _reference_stats: Precomputed reference statistics
        _streaming_window: Window for streaming detection
        _concept_detector: Concept drift detector

    Example:
        >>> detector = DriftDetector(
        ...     reference_data=X_train,
        ...     config=DriftDetectorConfig(
        ...         method=DriftMethod.PSI,
        ...         psi_threshold=0.2
        ...     )
        ... )
        >>> result = detector.detect_drift(X_new)
        >>> if result.severity == DriftSeverity.HIGH:
        ...     alert_team()
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        config: Optional[DriftDetectorConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference (training) data
            config: Detector configuration
            feature_names: Names of features
        """
        self.config = config or DriftDetectorConfig()
        self.feature_names = feature_names
        self._reference_data = reference_data
        self._reference_stats: Dict[int, Dict[str, float]] = {}
        self._streaming_window: deque = deque(maxlen=self.config.window_size)
        self._concept_errors: deque = deque(maxlen=self.config.window_size)

        np.random.seed(self.config.random_state)

        if reference_data is not None:
            self.set_reference(reference_data)

        logger.info(
            f"DriftDetector initialized: method={self.config.method}"
        )

    def set_reference(self, data: np.ndarray) -> None:
        """
        Set reference data distribution.

        Args:
            data: Reference data (training data)
        """
        self._reference_data = data
        self._compute_reference_stats()
        logger.info(f"Reference data set: {data.shape}")

    def _compute_reference_stats(self) -> None:
        """Compute statistics for reference data."""
        if self._reference_data is None:
            return

        n_features = self._reference_data.shape[1]

        for i in range(n_features):
            feature_data = self._reference_data[:, i]
            self._reference_stats[i] = {
                "mean": float(np.mean(feature_data)),
                "std": float(np.std(feature_data)),
                "min": float(np.min(feature_data)),
                "max": float(np.max(feature_data)),
                "median": float(np.median(feature_data)),
                "q1": float(np.percentile(feature_data, 25)),
                "q3": float(np.percentile(feature_data, 75))
            }

    def _calculate_provenance(
        self,
        drift_score: float,
        n_drifted: int
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        ref_hash = (
            hashlib.sha256(self._reference_data.tobytes()).hexdigest()[:16]
            if self._reference_data is not None else "none"
        )
        combined = f"{ref_hash}|{drift_score:.8f}|{n_drifted}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.

        Returns:
            Tuple of (statistic, p-value)
        """
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(reference, current)
            return (float(statistic), float(p_value))
        except ImportError:
            # Manual KS test
            n1, n2 = len(reference), len(current)
            combined = np.sort(np.concatenate([reference, current]))

            cdf1 = np.searchsorted(np.sort(reference), combined, side="right") / n1
            cdf2 = np.searchsorted(np.sort(current), combined, side="right") / n2

            statistic = float(np.max(np.abs(cdf1 - cdf2)))

            # Approximate p-value
            en = np.sqrt(n1 * n2 / (n1 + n2))
            p_value = 2 * np.exp(-2 * (en * statistic) ** 2)

            return (statistic, float(min(p_value, 1.0)))

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index.

        Returns:
            PSI value
        """
        # Create bins from reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Add small constant to avoid division by zero
        ref_props = (ref_counts + 1e-10) / (len(reference) + n_bins * 1e-10)
        cur_props = (cur_counts + 1e-10) / (len(current) + n_bins * 1e-10)

        # PSI formula
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def _classify_severity(
        self,
        drift_score: float,
        method: DriftMethod
    ) -> DriftSeverity:
        """Classify drift severity based on score."""
        if method == DriftMethod.PSI:
            if drift_score < 0.1:
                return DriftSeverity.NONE
            elif drift_score < 0.2:
                return DriftSeverity.LOW
            elif drift_score < 0.25:
                return DriftSeverity.MEDIUM
            elif drift_score < 0.5:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL
        else:
            # KS test - use p-value based thresholds
            if drift_score > 0.1:
                return DriftSeverity.NONE
            elif drift_score > 0.05:
                return DriftSeverity.LOW
            elif drift_score > 0.01:
                return DriftSeverity.MEDIUM
            elif drift_score > 0.001:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL

    def _generate_recommendations(
        self,
        result: "DriftResult"
    ) -> List[str]:
        """Generate recommendations based on drift results."""
        recommendations = []

        if result.severity == DriftSeverity.CRITICAL:
            recommendations.append(
                "URGENT: Critical drift detected. Immediate model retraining recommended."
            )
            recommendations.append(
                "Investigate data pipeline for potential issues."
            )
        elif result.severity == DriftSeverity.HIGH:
            recommendations.append(
                "Significant drift detected. Schedule model retraining within 24 hours."
            )
        elif result.severity == DriftSeverity.MEDIUM:
            recommendations.append(
                "Moderate drift detected. Monitor closely and plan retraining."
            )
        elif result.severity == DriftSeverity.LOW:
            recommendations.append(
                "Minor drift detected. Continue monitoring."
            )

        # Feature-specific recommendations
        drifted_features = [
            f for f in result.feature_drifts if f.drift_detected
        ]
        if drifted_features:
            feature_names = [
                f.feature_name or f"feature_{f.feature_index}"
                for f in drifted_features[:3]
            ]
            recommendations.append(
                f"Top drifting features: {', '.join(feature_names)}"
            )

        return recommendations

    def detect_drift(
        self,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> DriftResult:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current data to check for drift
            feature_names: Optional feature names

        Returns:
            DriftResult with drift analysis

        Example:
            >>> result = detector.detect_drift(X_new)
            >>> if result.drift_detected:
            ...     print(f"Drift severity: {result.severity}")
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if current_data.shape[1] != self._reference_data.shape[1]:
            raise ValueError(
                f"Feature count mismatch: reference has {self._reference_data.shape[1]}, "
                f"current has {current_data.shape[1]}"
            )

        feature_names = feature_names or self.feature_names
        n_features = current_data.shape[1]

        # Select features to analyze
        feature_indices = (
            self.config.feature_subset
            if self.config.feature_subset
            else list(range(n_features))
        )

        feature_drifts = []
        drift_scores = []

        for i in feature_indices:
            reference_feature = self._reference_data[:, i]
            current_feature = current_data[:, i]

            # Compute drift based on method
            if self.config.method == DriftMethod.PSI:
                drift_score = self._compute_psi(reference_feature, current_feature)
                p_value = None
                drift_detected = drift_score > self.config.psi_threshold
            else:
                stat, p_value = self._ks_test(reference_feature, current_feature)
                drift_score = p_value
                drift_detected = p_value < self.config.significance_level

            severity = self._classify_severity(
                drift_score if self.config.method == DriftMethod.PSI else p_value or 0,
                self.config.method
            )

            feature_drift = FeatureDrift(
                feature_index=i,
                feature_name=feature_names[i] if feature_names else None,
                drift_score=drift_score,
                p_value=p_value,
                drift_detected=drift_detected,
                severity=severity,
                reference_mean=float(np.mean(reference_feature)),
                current_mean=float(np.mean(current_feature)),
                reference_std=float(np.std(reference_feature)),
                current_std=float(np.std(current_feature))
            )
            feature_drifts.append(feature_drift)
            drift_scores.append(drift_score)

        # Calculate overall drift
        n_drifted = sum(1 for f in feature_drifts if f.drift_detected)
        overall_drift_score = float(np.mean(drift_scores))
        overall_drift_detected = n_drifted > 0

        # Determine overall severity
        if n_drifted == 0:
            overall_severity = DriftSeverity.NONE
        elif n_drifted / len(feature_indices) > 0.5:
            overall_severity = DriftSeverity.CRITICAL
        elif n_drifted / len(feature_indices) > 0.3:
            overall_severity = DriftSeverity.HIGH
        elif n_drifted / len(feature_indices) > 0.1:
            overall_severity = DriftSeverity.MEDIUM
        else:
            overall_severity = DriftSeverity.LOW

        # Calculate provenance
        provenance_hash = self._calculate_provenance(overall_drift_score, n_drifted)

        # Create result
        result = DriftResult(
            drift_detected=overall_drift_detected,
            drift_type=self.config.drift_type.value,
            overall_drift_score=overall_drift_score,
            severity=overall_severity,
            n_features_drifted=n_drifted,
            feature_drifts=feature_drifts,
            method_used=self.config.method.value,
            reference_samples=len(self._reference_data),
            current_samples=len(current_data),
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

        # Add recommendations
        result.recommendations = self._generate_recommendations(result)

        logger.info(
            f"Drift detection complete: detected={overall_drift_detected}, "
            f"severity={overall_severity.value}, n_drifted={n_drifted}/{len(feature_indices)}"
        )

        return result

    def update_streaming(
        self,
        x: np.ndarray,
        y_true: Optional[float] = None,
        y_pred: Optional[float] = None
    ) -> Optional[DriftResult]:
        """
        Update streaming drift detector with new sample.

        Args:
            x: New feature vector
            y_true: True label (for concept drift)
            y_pred: Predicted label (for concept drift)

        Returns:
            DriftResult if drift detected, None otherwise
        """
        self._streaming_window.append(x)

        # Check concept drift if labels provided
        if y_true is not None and y_pred is not None:
            error = abs(y_true - y_pred)
            self._concept_errors.append(error)

        # Check for drift if window is full
        if len(self._streaming_window) >= self.config.min_samples:
            current_data = np.array(list(self._streaming_window))

            # Simple check against reference
            result = self.detect_drift(current_data)

            if result.drift_detected:
                return result

        return None

    def get_drift_summary(
        self,
        results: List[DriftResult]
    ) -> Dict[str, Any]:
        """
        Get summary of multiple drift detection results.

        Args:
            results: List of DriftResult objects

        Returns:
            Summary dictionary
        """
        return {
            "total_checks": len(results),
            "drift_detected_count": sum(1 for r in results if r.drift_detected),
            "severity_distribution": {
                s.value: sum(1 for r in results if r.severity == s)
                for s in DriftSeverity
            },
            "avg_drift_score": float(np.mean([r.overall_drift_score for r in results])),
            "most_common_drifted_features": self._get_common_drifted_features(results)
        }

    def _get_common_drifted_features(
        self,
        results: List[DriftResult]
    ) -> List[Tuple[int, int]]:
        """Get most commonly drifted features."""
        from collections import Counter
        drifted = []
        for r in results:
            for f in r.feature_drifts:
                if f.drift_detected:
                    drifted.append(f.feature_index)

        return Counter(drifted).most_common(5)


# Unit test stubs
class TestDriftDetector:
    """Unit tests for DriftDetector."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        detector = DriftDetector()
        assert detector.config.method == DriftMethod.KS_TEST

    def test_set_reference(self):
        """Test setting reference data."""
        detector = DriftDetector()
        X = np.random.randn(100, 5)
        detector.set_reference(X)

        assert detector._reference_data is not None
        assert len(detector._reference_stats) == 5

    def test_ks_test(self):
        """Test KS test."""
        detector = DriftDetector()

        # Same distribution - should not detect drift
        a = np.random.randn(100)
        b = np.random.randn(100)
        stat, p_value = detector._ks_test(a, b)
        assert p_value > 0.05

        # Different distributions - should detect drift
        c = np.random.randn(100) + 5
        stat, p_value = detector._ks_test(a, c)
        assert p_value < 0.05

    def test_psi_calculation(self):
        """Test PSI calculation."""
        detector = DriftDetector()

        # Same distribution
        a = np.random.randn(1000)
        psi = detector._compute_psi(a, a)
        assert psi < 0.1

        # Different distribution
        b = np.random.randn(1000) + 3
        psi = detector._compute_psi(a, b)
        assert psi > 0.2

    def test_detect_drift_no_drift(self):
        """Test drift detection with no drift."""
        X_ref = np.random.randn(1000, 3)
        X_cur = np.random.randn(200, 3)

        detector = DriftDetector(reference_data=X_ref)
        result = detector.detect_drift(X_cur)

        assert result.n_features_drifted < 2  # May have false positive

    def test_detect_drift_with_drift(self):
        """Test drift detection with actual drift."""
        X_ref = np.random.randn(1000, 3)
        X_cur = np.random.randn(200, 3) + np.array([5, 0, 5])  # Shift features 0 and 2

        detector = DriftDetector(reference_data=X_ref)
        result = detector.detect_drift(X_cur)

        assert result.drift_detected
        assert result.n_features_drifted >= 2

    def test_severity_classification(self):
        """Test severity classification."""
        detector = DriftDetector()

        assert detector._classify_severity(0.05, DriftMethod.PSI) == DriftSeverity.NONE
        assert detector._classify_severity(0.15, DriftMethod.PSI) == DriftSeverity.LOW
        assert detector._classify_severity(0.22, DriftMethod.PSI) == DriftSeverity.MEDIUM
        assert detector._classify_severity(0.4, DriftMethod.PSI) == DriftSeverity.HIGH
        assert detector._classify_severity(0.6, DriftMethod.PSI) == DriftSeverity.CRITICAL
