# -*- coding: utf-8 -*-
"""
TASK-062: Distribution Shift Detection

This module provides comprehensive distribution shift detection for
GreenLang Process Heat ML models, including Population Stability Index (PSI),
Kolmogorov-Smirnov test, Maximum Mean Discrepancy (MMD), Chi-squared test,
and shift severity classification.

Distribution shift is a major cause of model degradation in production,
especially in Process Heat applications where operating conditions vary.

Example:
    >>> from greenlang.ml.robustness import DistributionShiftDetector
    >>> detector = DistributionShiftDetector(reference_data=X_train)
    >>> result = detector.detect_shift(X_production)
    >>> if result.shift_severity == ShiftSeverity.CRITICAL:
    ...     trigger_model_retraining()
"""

from typing import Any, Dict, List, Optional, Union, Tuple
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

class ShiftType(str, Enum):
    """Types of distribution shift."""
    COVARIATE = "covariate"  # P(X) changes
    LABEL = "label"  # P(Y) changes
    CONCEPT = "concept"  # P(Y|X) changes
    PRIOR = "prior"  # Class prior changes
    TEMPORAL = "temporal"  # Time-based shift


class ShiftSeverity(str, Enum):
    """Severity levels for distribution shift."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Statistical methods for shift detection."""
    PSI = "psi"  # Population Stability Index
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov test
    MMD = "mmd"  # Maximum Mean Discrepancy
    CHI_SQUARED = "chi_squared"  # Chi-squared test
    WASSERSTEIN = "wasserstein"  # Wasserstein distance
    ENERGY_DISTANCE = "energy_distance"


# =============================================================================
# Configuration
# =============================================================================

class DistributionShiftConfig(BaseModel):
    """Configuration for distribution shift detection."""

    # Detection methods to use
    detection_methods: List[DetectionMethod] = Field(
        default_factory=lambda: [
            DetectionMethod.PSI,
            DetectionMethod.KS_TEST,
            DetectionMethod.MMD
        ],
        description="Methods for shift detection"
    )

    # PSI thresholds
    psi_threshold_minor: float = Field(
        default=0.1,
        gt=0,
        description="PSI threshold for minor shift"
    )
    psi_threshold_significant: float = Field(
        default=0.2,
        gt=0,
        description="PSI threshold for significant shift"
    )
    psi_threshold_critical: float = Field(
        default=0.25,
        gt=0,
        description="PSI threshold for critical shift"
    )
    psi_n_bins: int = Field(
        default=10,
        ge=5,
        le=100,
        description="Number of bins for PSI calculation"
    )

    # KS test thresholds
    ks_significance_level: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for KS test"
    )

    # MMD settings
    mmd_threshold: float = Field(
        default=0.1,
        gt=0,
        description="MMD threshold for shift detection"
    )
    mmd_kernel: str = Field(
        default="rbf",
        description="Kernel type for MMD (rbf, linear, polynomial)"
    )
    mmd_gamma: Optional[float] = Field(
        default=None,
        description="Gamma for RBF kernel (None = median heuristic)"
    )

    # Chi-squared settings
    chi2_significance_level: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for Chi-squared test"
    )
    chi2_n_bins: int = Field(
        default=10,
        ge=5,
        description="Bins for continuous feature discretization"
    )

    # General settings
    min_samples: int = Field(
        default=30,
        ge=10,
        description="Minimum samples for reliable detection"
    )
    feature_subset: Optional[List[int]] = Field(
        default=None,
        description="Feature indices to monitor (None = all)"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashing"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


# =============================================================================
# Result Models
# =============================================================================

class FeatureShiftResult(BaseModel):
    """Shift detection result for a single feature."""

    feature_index: int = Field(..., description="Feature index")
    feature_name: Optional[str] = Field(default=None, description="Feature name")

    # Statistics
    reference_mean: float = Field(..., description="Reference mean")
    current_mean: float = Field(..., description="Current mean")
    reference_std: float = Field(..., description="Reference std")
    current_std: float = Field(..., description="Current std")
    mean_shift: float = Field(..., description="Absolute mean shift")
    mean_shift_normalized: float = Field(..., description="Normalized mean shift")

    # PSI result
    psi_value: Optional[float] = Field(default=None, description="PSI value")
    psi_shift_detected: Optional[bool] = Field(default=None, description="PSI shift detected")

    # KS test result
    ks_statistic: Optional[float] = Field(default=None, description="KS statistic")
    ks_p_value: Optional[float] = Field(default=None, description="KS p-value")
    ks_shift_detected: Optional[bool] = Field(default=None, description="KS shift detected")

    # Chi-squared result
    chi2_statistic: Optional[float] = Field(default=None, description="Chi-squared statistic")
    chi2_p_value: Optional[float] = Field(default=None, description="Chi-squared p-value")
    chi2_shift_detected: Optional[bool] = Field(default=None, description="Chi-squared shift detected")

    # Overall
    shift_detected: bool = Field(..., description="Overall shift detected")
    severity: ShiftSeverity = Field(..., description="Shift severity")


class ShiftDetectionResult(BaseModel):
    """Comprehensive distribution shift detection result."""

    # Summary
    shift_detected: bool = Field(..., description="Whether shift detected")
    shift_severity: ShiftSeverity = Field(..., description="Overall severity")
    shift_type: ShiftType = Field(
        default=ShiftType.COVARIATE,
        description="Type of shift detected"
    )

    # Feature-level results
    feature_results: List[FeatureShiftResult] = Field(
        default_factory=list,
        description="Per-feature results"
    )
    n_features_shifted: int = Field(..., description="Number of shifted features")
    shifted_feature_indices: List[int] = Field(
        default_factory=list,
        description="Indices of shifted features"
    )

    # Aggregate metrics
    overall_psi: Optional[float] = Field(default=None, description="Mean PSI")
    overall_ks_statistic: Optional[float] = Field(default=None, description="Max KS statistic")
    mmd_distance: Optional[float] = Field(default=None, description="MMD distance")
    wasserstein_distance: Optional[float] = Field(default=None, description="Wasserstein")

    # Correction weights (for importance weighting)
    correction_weights: Optional[List[float]] = Field(
        default=None,
        description="Importance weights for shift correction"
    )

    # Sample info
    reference_samples: int = Field(..., description="Reference sample count")
    current_samples: int = Field(..., description="Current sample count")

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
    methods_used: List[str] = Field(
        default_factory=list,
        description="Detection methods applied"
    )


# =============================================================================
# Distribution Shift Detector
# =============================================================================

class DistributionShiftDetector:
    """
    Comprehensive Distribution Shift Detector for Process Heat ML.

    This detector implements multiple statistical methods for detecting
    distribution shift between reference and current data:

    - Population Stability Index (PSI): Industry-standard for monitoring
    - Kolmogorov-Smirnov Test: Non-parametric distribution comparison
    - Maximum Mean Discrepancy (MMD): Kernel-based distance measure
    - Chi-squared Test: For categorical features

    All calculations are deterministic for reproducibility.

    Attributes:
        config: Detection configuration
        _reference_data: Reference distribution
        _reference_stats: Precomputed reference statistics
        feature_names: Optional feature names

    Example:
        >>> detector = DistributionShiftDetector(
        ...     reference_data=X_train,
        ...     config=DistributionShiftConfig(
        ...         detection_methods=[DetectionMethod.PSI, DetectionMethod.KS_TEST]
        ...     )
        ... )
        >>> result = detector.detect_shift(X_production)
        >>> for feat in result.feature_results:
        ...     if feat.shift_detected:
        ...         print(f"Shift in {feat.feature_name}: PSI={feat.psi_value:.3f}")
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
            reference_data: Reference (training) data
            config: Detection configuration
            feature_names: Optional feature names
        """
        self.config = config or DistributionShiftConfig()
        self.feature_names = feature_names
        self._reference_data: Optional[np.ndarray] = None
        self._reference_stats: Dict[int, Dict[str, Any]] = {}
        self._reference_histograms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Initialize RNG
        self._rng = np.random.RandomState(self.config.random_state)

        if reference_data is not None:
            self.set_reference(reference_data)

        logger.info(
            f"DistributionShiftDetector initialized: "
            f"methods={[m.value for m in self.config.detection_methods]}"
        )

    def set_reference(self, data: np.ndarray) -> None:
        """
        Set reference distribution.

        Args:
            data: Reference data (typically training data)
        """
        self._reference_data = data.copy()
        self._compute_reference_stats()
        logger.info(f"Reference set: shape={data.shape}")

    def _compute_reference_stats(self) -> None:
        """Precompute reference statistics (deterministic)."""
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
                "q3": float(np.percentile(feature_data, 75)),
                "n_samples": len(feature_data)
            }

            # Precompute histogram for PSI
            hist, bin_edges = np.histogram(
                feature_data,
                bins=self.config.psi_n_bins,
                density=False
            )
            self._reference_histograms[i] = (hist.astype(float), bin_edges)

    # =========================================================================
    # PSI Calculation
    # =========================================================================

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Calculate Population Stability Index (deterministic).

        PSI measures the shift between two distributions:
        PSI = sum((current% - reference%) * ln(current% / reference%))

        Interpretation:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Minor shift
        - 0.2 <= PSI < 0.25: Significant shift
        - PSI >= 0.25: Critical shift

        Args:
            reference: Reference distribution samples
            current: Current distribution samples
            n_bins: Number of bins (uses config default if None)

        Returns:
            PSI value
        """
        n_bins = n_bins or self.config.psi_n_bins

        # Create bins from combined data for consistency
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        # Add small buffer to ensure all values fall within bins
        bin_edges = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)

        # Calculate histograms
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with Laplace smoothing to avoid division by zero
        ref_props = (ref_counts + 1) / (len(reference) + n_bins)
        cur_props = (cur_counts + 1) / (len(current) + n_bins)

        # PSI formula
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    # =========================================================================
    # Kolmogorov-Smirnov Test
    # =========================================================================

    def kolmogorov_smirnov_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform two-sample Kolmogorov-Smirnov test (deterministic).

        The KS test compares the empirical CDFs of two samples.
        The statistic is the maximum difference between CDFs.

        Args:
            reference: Reference distribution samples
            current: Current distribution samples

        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(reference, current)
            return (float(statistic), float(p_value))
        except ImportError:
            # Manual implementation if scipy not available
            return self._manual_ks_test(reference, current)

    def _manual_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """Manual KS test implementation (deterministic)."""
        n1, n2 = len(reference), len(current)

        # Combine and sort all values
        combined = np.sort(np.concatenate([reference, current]))

        # Calculate empirical CDFs at each point
        cdf1 = np.searchsorted(np.sort(reference), combined, side="right") / n1
        cdf2 = np.searchsorted(np.sort(current), combined, side="right") / n2

        # KS statistic is maximum CDF difference
        statistic = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value using asymptotic distribution
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * (en * statistic) ** 2)
        p_value = float(min(max(p_value, 0.0), 1.0))

        return (statistic, p_value)

    # =========================================================================
    # Maximum Mean Discrepancy
    # =========================================================================

    def calculate_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        gamma: Optional[float] = None
    ) -> float:
        """
        Calculate Maximum Mean Discrepancy (deterministic).

        MMD measures the distance between distributions in a
        reproducing kernel Hilbert space.

        Args:
            X: Samples from first distribution
            Y: Samples from second distribution
            gamma: RBF kernel bandwidth (None = median heuristic)

        Returns:
            MMD value (squared)
        """
        if gamma is None:
            # Median heuristic for bandwidth
            combined = np.vstack([X, Y])
            pairwise_sq = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
            median_dist = np.median(pairwise_sq[pairwise_sq > 0])
            gamma = 1.0 / (2 * median_dist + 1e-10)

        # RBF kernel function
        def rbf_kernel(A, B):
            sq_dist = np.sum((A[:, None] - B[None, :]) ** 2, axis=2)
            return np.exp(-gamma * sq_dist)

        # Compute kernel matrices
        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        n, m = len(X), len(Y)

        # Unbiased MMD^2 estimator
        # Remove diagonal elements for unbiased estimate
        mmd2 = (
            (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) -
            2 * np.sum(K_XY) / (n * m) +
            (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
        )

        return float(max(0.0, mmd2))

    # =========================================================================
    # Chi-Squared Test
    # =========================================================================

    def chi_squared_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: Optional[int] = None,
        is_categorical: bool = False
    ) -> Tuple[float, float]:
        """
        Perform Chi-squared test for independence (deterministic).

        For categorical features, directly compare frequencies.
        For continuous features, discretize into bins first.

        Args:
            reference: Reference distribution samples
            current: Current distribution samples
            n_bins: Number of bins for continuous features
            is_categorical: Whether feature is categorical

        Returns:
            Tuple of (Chi-squared statistic, p-value)
        """
        n_bins = n_bins or self.config.chi2_n_bins

        if is_categorical:
            # Get unique categories
            categories = np.unique(np.concatenate([reference, current]))

            ref_counts = np.array([np.sum(reference == c) for c in categories])
            cur_counts = np.array([np.sum(current == c) for c in categories])
        else:
            # Discretize continuous feature
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)

            ref_counts, _ = np.histogram(reference, bins=bins)
            cur_counts, _ = np.histogram(current, bins=bins)

        # Add smoothing
        ref_counts = ref_counts.astype(float) + 1
        cur_counts = cur_counts.astype(float) + 1

        # Expected counts under null hypothesis
        total_ref = np.sum(ref_counts)
        total_cur = np.sum(cur_counts)
        total = total_ref + total_cur

        expected_ref = ref_counts + cur_counts
        expected_cur = ref_counts + cur_counts

        # Scale by sample proportions
        expected_ref = expected_ref * (total_ref / total)
        expected_cur = expected_cur * (total_cur / total)

        # Chi-squared statistic
        chi2 = (
            np.sum((ref_counts - expected_ref) ** 2 / expected_ref) +
            np.sum((cur_counts - expected_cur) ** 2 / expected_cur)
        )

        # Degrees of freedom
        df = len(ref_counts) - 1

        # Approximate p-value using chi-squared distribution
        try:
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi2, df)
        except ImportError:
            # Rough approximation
            p_value = float(np.exp(-chi2 / (2 * df)))

        return (float(chi2), float(p_value))

    # =========================================================================
    # Severity Classification
    # =========================================================================

    def classify_severity(
        self,
        psi: Optional[float] = None,
        ks_p_value: Optional[float] = None,
        mmd: Optional[float] = None,
        n_features_shifted: int = 0,
        total_features: int = 1
    ) -> ShiftSeverity:
        """
        Classify shift severity (deterministic).

        Args:
            psi: PSI value (if available)
            ks_p_value: KS test p-value (if available)
            mmd: MMD value (if available)
            n_features_shifted: Number of features with detected shift
            total_features: Total number of features

        Returns:
            ShiftSeverity enum
        """
        severity_scores = []

        # PSI-based severity
        if psi is not None:
            if psi >= self.config.psi_threshold_critical:
                severity_scores.append(4)  # CRITICAL
            elif psi >= self.config.psi_threshold_significant:
                severity_scores.append(3)  # SIGNIFICANT
            elif psi >= self.config.psi_threshold_minor:
                severity_scores.append(2)  # MODERATE
            elif psi > 0.05:
                severity_scores.append(1)  # MINOR
            else:
                severity_scores.append(0)  # NONE

        # KS-based severity
        if ks_p_value is not None:
            if ks_p_value < 0.001:
                severity_scores.append(4)
            elif ks_p_value < 0.01:
                severity_scores.append(3)
            elif ks_p_value < self.config.ks_significance_level:
                severity_scores.append(2)
            elif ks_p_value < 0.1:
                severity_scores.append(1)
            else:
                severity_scores.append(0)

        # MMD-based severity
        if mmd is not None:
            if mmd >= self.config.mmd_threshold * 2:
                severity_scores.append(4)
            elif mmd >= self.config.mmd_threshold:
                severity_scores.append(3)
            elif mmd >= self.config.mmd_threshold * 0.5:
                severity_scores.append(2)
            elif mmd > 0.01:
                severity_scores.append(1)
            else:
                severity_scores.append(0)

        # Feature proportion severity
        if total_features > 0:
            shift_ratio = n_features_shifted / total_features
            if shift_ratio > 0.5:
                severity_scores.append(4)
            elif shift_ratio > 0.3:
                severity_scores.append(3)
            elif shift_ratio > 0.1:
                severity_scores.append(2)
            elif shift_ratio > 0:
                severity_scores.append(1)

        if not severity_scores:
            return ShiftSeverity.NONE

        # Take maximum severity
        max_score = max(severity_scores)

        severity_map = {
            0: ShiftSeverity.NONE,
            1: ShiftSeverity.MINOR,
            2: ShiftSeverity.MODERATE,
            3: ShiftSeverity.SIGNIFICANT,
            4: ShiftSeverity.CRITICAL
        }

        return severity_map.get(max_score, ShiftSeverity.NONE)

    # =========================================================================
    # Importance Weighting
    # =========================================================================

    def compute_importance_weights(
        self,
        current_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance weights for distribution shift correction.

        These weights can be used to reweight predictions to account
        for distribution shift.

        Args:
            current_data: Current data samples

        Returns:
            Importance weights (one per sample)
        """
        if self._reference_data is None:
            return np.ones(len(current_data))

        # Estimate density ratio using histogram-based method
        weights = np.ones(len(current_data))

        for i in range(current_data.shape[1]):
            ref_hist, bin_edges = self._reference_histograms.get(
                i,
                np.histogram(self._reference_data[:, i], bins=self.config.psi_n_bins)
            )

            # Normalize to density
            ref_density = ref_hist / (np.sum(ref_hist) + 1e-10)

            # Current density
            cur_hist, _ = np.histogram(current_data[:, i], bins=bin_edges)
            cur_density = cur_hist / (np.sum(cur_hist) + 1e-10)

            # Add smoothing
            ref_density = ref_density + 1e-10
            cur_density = cur_density + 1e-10

            # Assign bin to each sample
            bin_indices = np.digitize(current_data[:, i], bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, len(ref_density) - 1)

            # Weight = P_ref / P_cur
            feature_weights = ref_density[bin_indices] / cur_density[bin_indices]
            weights *= feature_weights

        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10.0)

        # Normalize to sum to n
        weights = weights * len(weights) / np.sum(weights)

        return weights

    # =========================================================================
    # Main Detection Method
    # =========================================================================

    def detect_shift(
        self,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        compute_weights: bool = True
    ) -> ShiftDetectionResult:
        """
        Detect distribution shift between reference and current data.

        Args:
            current_data: Current data to check for shift
            feature_names: Optional feature names
            compute_weights: Whether to compute correction weights

        Returns:
            Comprehensive shift detection result

        Example:
            >>> result = detector.detect_shift(X_production)
            >>> if result.shift_severity >= ShiftSeverity.SIGNIFICANT:
            ...     print(f"Critical shift detected in {result.n_features_shifted} features")
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if current_data.shape[1] != self._reference_data.shape[1]:
            raise ValueError(
                f"Feature count mismatch: reference={self._reference_data.shape[1]}, "
                f"current={current_data.shape[1]}"
            )

        if len(current_data) < self.config.min_samples:
            logger.warning(
                f"Sample size ({len(current_data)}) below minimum ({self.config.min_samples}). "
                "Results may be unreliable."
            )

        feature_names = feature_names or self.feature_names
        n_features = current_data.shape[1]

        # Select features to analyze
        feature_indices = (
            self.config.feature_subset
            if self.config.feature_subset
            else list(range(n_features))
        )

        # Analyze each feature
        feature_results = []
        psi_values = []
        ks_statistics = []
        shifted_indices = []

        for i in feature_indices:
            ref_feature = self._reference_data[:, i]
            cur_feature = current_data[:, i]
            ref_stats = self._reference_stats.get(i, {})

            # Calculate statistics
            cur_mean = float(np.mean(cur_feature))
            cur_std = float(np.std(cur_feature))
            ref_mean = ref_stats.get("mean", 0.0)
            ref_std = ref_stats.get("std", 1.0)

            mean_shift = abs(cur_mean - ref_mean)
            mean_shift_norm = mean_shift / (ref_std + 1e-10)

            # PSI
            psi_value = None
            psi_shift = None
            if DetectionMethod.PSI in self.config.detection_methods:
                psi_value = self.calculate_psi(ref_feature, cur_feature)
                psi_shift = psi_value >= self.config.psi_threshold_minor
                psi_values.append(psi_value)

            # KS test
            ks_stat = None
            ks_pval = None
            ks_shift = None
            if DetectionMethod.KS_TEST in self.config.detection_methods:
                ks_stat, ks_pval = self.kolmogorov_smirnov_test(ref_feature, cur_feature)
                ks_shift = ks_pval < self.config.ks_significance_level
                ks_statistics.append(ks_stat)

            # Chi-squared
            chi2_stat = None
            chi2_pval = None
            chi2_shift = None
            if DetectionMethod.CHI_SQUARED in self.config.detection_methods:
                chi2_stat, chi2_pval = self.chi_squared_test(ref_feature, cur_feature)
                chi2_shift = chi2_pval < self.config.chi2_significance_level

            # Overall shift for this feature
            shift_indicators = [
                x for x in [psi_shift, ks_shift, chi2_shift]
                if x is not None
            ]
            shift_detected = any(shift_indicators) if shift_indicators else False

            if shift_detected:
                shifted_indices.append(i)

            # Severity for this feature
            severity = self.classify_severity(
                psi=psi_value,
                ks_p_value=ks_pval,
                n_features_shifted=1 if shift_detected else 0,
                total_features=1
            )

            feature_results.append(FeatureShiftResult(
                feature_index=i,
                feature_name=feature_names[i] if feature_names else None,
                reference_mean=ref_mean,
                current_mean=cur_mean,
                reference_std=ref_std,
                current_std=cur_std,
                mean_shift=mean_shift,
                mean_shift_normalized=mean_shift_norm,
                psi_value=psi_value,
                psi_shift_detected=psi_shift,
                ks_statistic=ks_stat,
                ks_p_value=ks_pval,
                ks_shift_detected=ks_shift,
                chi2_statistic=chi2_stat,
                chi2_p_value=chi2_pval,
                chi2_shift_detected=chi2_shift,
                shift_detected=shift_detected,
                severity=severity
            ))

        # Calculate MMD on full data if requested
        mmd_distance = None
        if DetectionMethod.MMD in self.config.detection_methods:
            mmd_distance = self.calculate_mmd(self._reference_data, current_data)

        # Aggregate metrics
        overall_psi = float(np.mean(psi_values)) if psi_values else None
        max_ks = float(np.max(ks_statistics)) if ks_statistics else None

        # Overall severity
        overall_severity = self.classify_severity(
            psi=overall_psi,
            mmd=mmd_distance,
            n_features_shifted=len(shifted_indices),
            total_features=len(feature_indices)
        )

        # Compute correction weights
        correction_weights = None
        if compute_weights and len(shifted_indices) > 0:
            correction_weights = self.compute_importance_weights(current_data).tolist()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_severity,
            len(shifted_indices),
            len(feature_indices),
            feature_results
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            overall_severity,
            len(shifted_indices),
            overall_psi
        )

        result = ShiftDetectionResult(
            shift_detected=len(shifted_indices) > 0,
            shift_severity=overall_severity,
            shift_type=ShiftType.COVARIATE,
            feature_results=feature_results,
            n_features_shifted=len(shifted_indices),
            shifted_feature_indices=shifted_indices,
            overall_psi=overall_psi,
            overall_ks_statistic=max_ks,
            mmd_distance=mmd_distance,
            correction_weights=correction_weights,
            reference_samples=len(self._reference_data),
            current_samples=len(current_data),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow(),
            methods_used=[m.value for m in self.config.detection_methods]
        )

        logger.info(
            f"Shift detection complete: detected={result.shift_detected}, "
            f"severity={overall_severity.value}, shifted={len(shifted_indices)}/{len(feature_indices)}"
        )

        return result

    def _generate_recommendations(
        self,
        severity: ShiftSeverity,
        n_shifted: int,
        total_features: int,
        feature_results: List[FeatureShiftResult]
    ) -> List[str]:
        """Generate recommendations based on shift analysis."""
        recommendations = []

        if severity == ShiftSeverity.NONE:
            recommendations.append(
                "No significant distribution shift detected. Model predictions should be reliable."
            )
            return recommendations

        if severity == ShiftSeverity.CRITICAL:
            recommendations.append(
                "CRITICAL: Major distribution shift detected. Immediate model retraining recommended."
            )
            recommendations.append(
                "Consider investigating data pipeline for potential issues or data quality problems."
            )
        elif severity == ShiftSeverity.SIGNIFICANT:
            recommendations.append(
                "Significant shift detected. Schedule model retraining within 24-48 hours."
            )
            recommendations.append(
                "Apply importance weighting to predictions to partially mitigate shift effects."
            )
        elif severity == ShiftSeverity.MODERATE:
            recommendations.append(
                "Moderate shift detected. Monitor predictions closely and plan retraining."
            )
        else:
            recommendations.append(
                "Minor shift detected. Continue monitoring but no immediate action required."
            )

        # Feature-specific recommendations
        high_shift_features = [
            f for f in feature_results
            if f.severity in [ShiftSeverity.SIGNIFICANT, ShiftSeverity.CRITICAL]
        ]
        if high_shift_features:
            names = [
                f.feature_name or f"feature_{f.feature_index}"
                for f in high_shift_features[:3]
            ]
            recommendations.append(
                f"Features with highest shift: {', '.join(names)}. "
                "Review data collection process for these features."
            )

        return recommendations

    def _calculate_provenance(
        self,
        severity: ShiftSeverity,
        n_shifted: int,
        psi: Optional[float]
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        provenance_data = (
            f"{severity.value}|{n_shifted}|"
            f"{psi if psi else 0.0:.8f}|"
            f"{self.config.random_state}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()


# =============================================================================
# Unit Tests
# =============================================================================

class TestDistributionShiftDetector:
    """Unit tests for DistributionShiftDetector."""

    def test_psi_no_shift(self):
        """Test PSI with identical distributions."""
        detector = DistributionShiftDetector()
        data = np.random.randn(1000)

        psi = detector.calculate_psi(data, data)

        assert psi < 0.01  # Should be very small

    def test_psi_with_shift(self):
        """Test PSI with shifted distribution."""
        detector = DistributionShiftDetector()
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 2  # Shifted mean

        psi = detector.calculate_psi(ref, cur)

        assert psi > 0.2  # Should detect significant shift

    def test_ks_test_no_shift(self):
        """Test KS test with same distribution."""
        detector = DistributionShiftDetector()
        ref = np.random.randn(500)
        cur = np.random.randn(500)

        stat, p_value = detector.kolmogorov_smirnov_test(ref, cur)

        assert p_value > 0.05  # Should not reject null hypothesis

    def test_ks_test_with_shift(self):
        """Test KS test with different distributions."""
        detector = DistributionShiftDetector()
        ref = np.random.randn(500)
        cur = np.random.randn(500) + 3  # Shifted

        stat, p_value = detector.kolmogorov_smirnov_test(ref, cur)

        assert p_value < 0.05  # Should reject null hypothesis
        assert stat > 0.3  # Large KS statistic

    def test_mmd_calculation(self):
        """Test MMD calculation."""
        detector = DistributionShiftDetector()

        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5) + 2

        mmd = detector.calculate_mmd(X, Y)

        assert mmd > 0  # Should be positive
        assert mmd > detector.calculate_mmd(X, X + 0.1 * np.random.randn(100, 5))

    def test_chi_squared_test(self):
        """Test Chi-squared test."""
        detector = DistributionShiftDetector()

        ref = np.random.randn(500)
        cur = np.random.randn(500)

        chi2, p_value = detector.chi_squared_test(ref, cur)

        assert chi2 >= 0
        assert 0 <= p_value <= 1

    def test_detect_shift_no_shift(self):
        """Test full detection with no shift."""
        X_ref = np.random.randn(500, 5)
        X_cur = np.random.randn(200, 5)

        detector = DistributionShiftDetector(reference_data=X_ref)
        result = detector.detect_shift(X_cur)

        assert result.shift_severity in [ShiftSeverity.NONE, ShiftSeverity.MINOR]

    def test_detect_shift_with_shift(self):
        """Test full detection with actual shift."""
        X_ref = np.random.randn(500, 5)
        X_cur = np.random.randn(200, 5) + np.array([3, 0, 2, 0, 3])

        detector = DistributionShiftDetector(reference_data=X_ref)
        result = detector.detect_shift(X_cur)

        assert result.shift_detected
        assert result.n_features_shifted >= 2
        assert result.shift_severity in [ShiftSeverity.SIGNIFICANT, ShiftSeverity.CRITICAL]

    def test_importance_weights(self):
        """Test importance weight computation."""
        X_ref = np.random.randn(500, 3)
        X_cur = np.random.randn(100, 3)

        detector = DistributionShiftDetector(reference_data=X_ref)
        weights = detector.compute_importance_weights(X_cur)

        assert len(weights) == 100
        assert np.all(weights > 0)
        assert np.isclose(np.mean(weights), 1.0, atol=0.1)

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        detector = DistributionShiftDetector()

        hash1 = detector._calculate_provenance(ShiftSeverity.MODERATE, 3, 0.15)
        hash2 = detector._calculate_provenance(ShiftSeverity.MODERATE, 3, 0.15)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
