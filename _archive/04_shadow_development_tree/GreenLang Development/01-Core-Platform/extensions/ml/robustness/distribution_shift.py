# -*- coding: utf-8 -*-
"""
Distribution Shift Module

This module provides distribution shift detection and handling for GreenLang
ML models, identifying when input data differs significantly from training
distribution and adapting predictions accordingly.

Distribution shift is a major cause of model degradation in production,
especially in regulatory contexts where data patterns evolve with new
regulations, regions, or industry practices.

Example:
    >>> from greenlang.ml.robustness import DistributionShift
    >>> shift_detector = DistributionShift(reference_data=X_train)
    >>> result = shift_detector.detect_shift(X_new)
    >>> if result.shift_detected:
    ...     apply_correction(X_new, result.correction_weights)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ShiftType(str, Enum):
    """Types of distribution shift."""
    COVARIATE = "covariate"  # P(X) changes
    LABEL = "label"  # P(Y) changes
    CONCEPT = "concept"  # P(Y|X) changes
    PRIOR = "prior"  # P(Y) changes
    FULL = "full"  # All distributions change


class CorrectionMethod(str, Enum):
    """Methods for correcting distribution shift."""
    IMPORTANCE_WEIGHTING = "importance_weighting"
    DENSITY_RATIO = "density_ratio"
    DOMAIN_ADAPTATION = "domain_adaptation"
    REWEIGHTING = "reweighting"
    NONE = "none"


class DistributionShiftConfig(BaseModel):
    """Configuration for distribution shift detection."""

    shift_type: ShiftType = Field(
        default=ShiftType.COVARIATE,
        description="Type of shift to detect"
    )
    correction_method: CorrectionMethod = Field(
        default=CorrectionMethod.IMPORTANCE_WEIGHTING,
        description="Correction method"
    )
    threshold: float = Field(
        default=0.1,
        gt=0,
        description="Threshold for shift detection"
    )
    n_bins: int = Field(
        default=20,
        ge=5,
        description="Bins for density estimation"
    )
    max_weight: float = Field(
        default=10.0,
        gt=1,
        description="Maximum importance weight"
    )
    clip_weights: bool = Field(
        default=True,
        description="Clip extreme weights"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class FeatureShift(BaseModel):
    """Shift information for a single feature."""

    feature_index: int = Field(
        ...,
        description="Feature index"
    )
    feature_name: Optional[str] = Field(
        default=None,
        description="Feature name"
    )
    shift_score: float = Field(
        ...,
        description="Shift score"
    )
    shift_detected: bool = Field(
        ...,
        description="Whether shift detected"
    )
    reference_mean: float = Field(
        ...,
        description="Reference distribution mean"
    )
    current_mean: float = Field(
        ...,
        description="Current distribution mean"
    )
    reference_std: float = Field(
        ...,
        description="Reference distribution std"
    )
    current_std: float = Field(
        ...,
        description="Current distribution std"
    )
    mean_shift: float = Field(
        ...,
        description="Shift in mean"
    )
    std_ratio: float = Field(
        ...,
        description="Ratio of standard deviations"
    )


class ShiftResult(BaseModel):
    """Result from distribution shift detection."""

    shift_detected: bool = Field(
        ...,
        description="Whether any shift detected"
    )
    overall_shift_score: float = Field(
        ...,
        description="Overall shift score"
    )
    shift_type: str = Field(
        ...,
        description="Type of shift detected"
    )
    n_features_shifted: int = Field(
        ...,
        description="Number of features with shift"
    )
    feature_shifts: List[FeatureShift] = Field(
        ...,
        description="Per-feature shift information"
    )
    correction_weights: Optional[List[float]] = Field(
        default=None,
        description="Importance weights for correction"
    )
    domain_distance: float = Field(
        ...,
        description="Distance between domains"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )


class DistributionShift:
    """
    Distribution Shift Detector for GreenLang.

    This class detects and corrects distribution shift between
    training and deployment data, ensuring model reliability
    when data patterns change.

    Key capabilities:
    - Covariate shift detection
    - Importance weighting
    - Domain adaptation
    - Per-feature analysis
    - Correction weights

    Attributes:
        config: Configuration
        _reference_data: Reference distribution
        _reference_stats: Reference statistics
        _density_estimator: Density ratio estimator

    Example:
        >>> shift_detector = DistributionShift(
        ...     reference_data=X_train,
        ...     config=DistributionShiftConfig(
        ...         correction_method=CorrectionMethod.IMPORTANCE_WEIGHTING
        ...     )
        ... )
        >>> result = shift_detector.detect_shift(X_test)
        >>> if result.shift_detected:
        ...     # Apply correction weights
        ...     weighted_pred = model.predict(X_test) * result.correction_weights
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        config: Optional[DistributionShiftConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize distribution shift detector.

        Args:
            reference_data: Reference (training) distribution
            config: Configuration
            feature_names: Feature names
        """
        self.config = config or DistributionShiftConfig()
        self.feature_names = feature_names
        self._reference_data = reference_data
        self._reference_stats: Dict[int, Dict[str, float]] = {}
        self._density_estimator = None

        np.random.seed(self.config.random_state)

        if reference_data is not None:
            self.set_reference(reference_data)

        logger.info(
            f"DistributionShift initialized: type={self.config.shift_type}"
        )

    def set_reference(self, data: np.ndarray) -> None:
        """
        Set reference distribution.

        Args:
            data: Reference data
        """
        self._reference_data = data
        self._compute_reference_stats()
        logger.info(f"Reference set: {data.shape}")

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

    def _estimate_density_ratio(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> np.ndarray:
        """
        Estimate density ratio P_current / P_reference.

        Args:
            reference: Reference distribution samples
            current: Current distribution samples

        Returns:
            Density ratios for current samples
        """
        # Simple histogram-based density estimation
        n_bins = self.config.n_bins

        # For each sample in current, estimate P(x|current) / P(x|reference)
        ratios = np.ones(len(current))

        for i in range(current.shape[1]):
            # Compute histograms
            all_data = np.concatenate([reference[:, i], current[:, i]])
            bin_edges = np.histogram_bin_edges(all_data, bins=n_bins)

            ref_hist, _ = np.histogram(reference[:, i], bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(current[:, i], bins=bin_edges, density=True)

            # Add small constant for smoothing
            ref_hist = ref_hist + 1e-10
            cur_hist = cur_hist + 1e-10

            # Assign ratios to each sample
            bin_indices = np.digitize(current[:, i], bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            feature_ratios = cur_hist[bin_indices] / ref_hist[bin_indices]
            ratios *= feature_ratios

        return ratios

    def _compute_importance_weights(
        self,
        current: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance weights for distribution shift correction.

        Args:
            current: Current distribution samples

        Returns:
            Importance weights
        """
        if self._reference_data is None:
            return np.ones(len(current))

        # Estimate density ratio
        ratios = self._estimate_density_ratio(self._reference_data, current)

        # Invert for importance weighting (we want P_ref / P_current)
        weights = 1.0 / (ratios + 1e-10)

        # Clip extreme weights
        if self.config.clip_weights:
            weights = np.clip(weights, 1.0 / self.config.max_weight, self.config.max_weight)

        # Normalize to sum to n
        weights = weights * len(weights) / np.sum(weights)

        return weights

    def _calculate_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        gamma: Optional[float] = None
    ) -> float:
        """
        Calculate Maximum Mean Discrepancy between two distributions.

        Args:
            X: Samples from first distribution
            Y: Samples from second distribution
            gamma: RBF kernel bandwidth

        Returns:
            MMD value
        """
        if gamma is None:
            # Median heuristic for bandwidth
            combined = np.vstack([X, Y])
            distances = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
            gamma = 1.0 / (2 * np.median(distances[distances > 0]))

        def rbf_kernel(a, b):
            return np.exp(-gamma * np.sum((a[:, None] - b[None, :]) ** 2, axis=2))

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        n, m = len(X), len(Y)

        mmd = (
            np.sum(K_XX) / (n * n) -
            2 * np.sum(K_XY) / (n * m) +
            np.sum(K_YY) / (m * m)
        )

        return float(max(0, mmd))

    def _calculate_provenance(
        self,
        shift_score: float,
        n_shifted: int
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{shift_score:.8f}|{n_shifted}|{self.config.shift_type.value}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def detect_shift(
        self,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ShiftResult:
        """
        Detect distribution shift.

        Args:
            current_data: Current distribution samples
            feature_names: Optional feature names

        Returns:
            ShiftResult with detection and correction info

        Example:
            >>> result = detector.detect_shift(X_new)
            >>> for feat in result.feature_shifts[:5]:
            ...     print(f"Feature {feat.feature_index}: shift={feat.shift_score:.3f}")
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if current_data.shape[1] != self._reference_data.shape[1]:
            raise ValueError("Feature count mismatch")

        feature_names = feature_names or self.feature_names
        n_features = current_data.shape[1]

        # Analyze per-feature shift
        feature_shifts = []
        shift_scores = []

        for i in range(n_features):
            ref_feat = self._reference_data[:, i]
            cur_feat = current_data[:, i]

            # Calculate statistics
            ref_mean = float(np.mean(ref_feat))
            ref_std = float(np.std(ref_feat))
            cur_mean = float(np.mean(cur_feat))
            cur_std = float(np.std(cur_feat))

            # Normalized mean shift
            mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
            std_ratio = cur_std / (ref_std + 1e-10)

            # Combine into shift score
            shift_score = 0.5 * mean_shift + 0.5 * abs(np.log(std_ratio + 1e-10))
            shift_detected = shift_score > self.config.threshold

            feature_shifts.append(FeatureShift(
                feature_index=i,
                feature_name=feature_names[i] if feature_names else None,
                shift_score=shift_score,
                shift_detected=shift_detected,
                reference_mean=ref_mean,
                current_mean=cur_mean,
                reference_std=ref_std,
                current_std=cur_std,
                mean_shift=mean_shift,
                std_ratio=std_ratio
            ))
            shift_scores.append(shift_score)

        # Overall shift metrics
        n_shifted = sum(1 for f in feature_shifts if f.shift_detected)
        overall_shift_score = float(np.mean(shift_scores))
        shift_detected = n_shifted > 0 or overall_shift_score > self.config.threshold

        # Calculate domain distance using MMD
        domain_distance = self._calculate_mmd(
            self._reference_data, current_data
        )

        # Compute correction weights if needed
        correction_weights = None
        if (shift_detected and
            self.config.correction_method != CorrectionMethod.NONE):
            correction_weights = self._compute_importance_weights(current_data).tolist()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            shift_detected, n_shifted, overall_shift_score, feature_shifts
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(overall_shift_score, n_shifted)

        logger.info(
            f"Shift detection complete: detected={shift_detected}, "
            f"score={overall_shift_score:.4f}, n_shifted={n_shifted}/{n_features}"
        )

        return ShiftResult(
            shift_detected=shift_detected,
            overall_shift_score=overall_shift_score,
            shift_type=self.config.shift_type.value,
            n_features_shifted=n_shifted,
            feature_shifts=feature_shifts,
            correction_weights=correction_weights,
            domain_distance=domain_distance,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def _generate_recommendations(
        self,
        shift_detected: bool,
        n_shifted: int,
        score: float,
        feature_shifts: List[FeatureShift]
    ) -> List[str]:
        """Generate recommendations based on shift analysis."""
        recommendations = []

        if not shift_detected:
            recommendations.append(
                "No significant distribution shift detected. "
                "Model predictions should be reliable."
            )
            return recommendations

        if score > 0.5:
            recommendations.append(
                "CRITICAL: Significant distribution shift detected. "
                "Consider retraining the model or applying domain adaptation."
            )
        elif score > 0.2:
            recommendations.append(
                "Moderate distribution shift detected. "
                "Apply importance weighting for predictions."
            )
        else:
            recommendations.append(
                "Minor distribution shift detected. "
                "Monitor predictions closely."
            )

        # Feature-specific recommendations
        high_shift_features = [f for f in feature_shifts if f.shift_score > 0.3]
        if high_shift_features:
            names = [
                f.feature_name or f"feature_{f.feature_index}"
                for f in high_shift_features[:3]
            ]
            recommendations.append(
                f"Features with highest shift: {', '.join(names)}. "
                "Review data collection for these features."
            )

        return recommendations

    def apply_correction(
        self,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply correction weights to predictions.

        Args:
            predictions: Model predictions
            weights: Correction weights

        Returns:
            Corrected predictions
        """
        # For regression, weighted average approach
        return predictions * weights / np.mean(weights)


# Unit test stubs
class TestDistributionShift:
    """Unit tests for DistributionShift."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        detector = DistributionShift()
        assert detector.config.shift_type == ShiftType.COVARIATE

    def test_set_reference(self):
        """Test setting reference data."""
        detector = DistributionShift()
        X = np.random.randn(100, 5)
        detector.set_reference(X)

        assert detector._reference_data is not None
        assert len(detector._reference_stats) == 5

    def test_detect_no_shift(self):
        """Test detection with no shift."""
        X_ref = np.random.randn(500, 3)
        X_cur = np.random.randn(100, 3)

        detector = DistributionShift(reference_data=X_ref)
        result = detector.detect_shift(X_cur)

        # Should detect minimal shift
        assert result.overall_shift_score < 0.5

    def test_detect_with_shift(self):
        """Test detection with actual shift."""
        X_ref = np.random.randn(500, 3)
        X_cur = np.random.randn(100, 3) + np.array([3, 0, 3])

        detector = DistributionShift(reference_data=X_ref)
        result = detector.detect_shift(X_cur)

        assert result.shift_detected
        assert result.n_features_shifted >= 2

    def test_importance_weights(self):
        """Test importance weight computation."""
        X_ref = np.random.randn(500, 3)
        X_cur = np.random.randn(100, 3)

        detector = DistributionShift(reference_data=X_ref)
        weights = detector._compute_importance_weights(X_cur)

        assert len(weights) == 100
        assert np.all(weights > 0)

    def test_mmd_calculation(self):
        """Test MMD calculation."""
        detector = DistributionShift()

        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 3)

        mmd = detector._calculate_mmd(X, Y)
        assert mmd >= 0

        # Same distribution should have lower MMD
        mmd_same = detector._calculate_mmd(X, X)
        assert mmd_same < mmd
