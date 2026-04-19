# -*- coding: utf-8 -*-
"""
Calibration Module

This module provides probability calibration capabilities for GreenLang ML
models, enabling reliable uncertainty estimates through temperature scaling
and other calibration methods.

Calibration ensures that predicted probabilities reflect true likelihoods,
critical for regulatory compliance where confidence estimates must be
accurate and trustworthy.

Example:
    >>> from greenlang.ml.uncertainty import Calibrator
    >>> calibrator = Calibrator(method="temperature_scaling")
    >>> calibrator.fit(y_proba, y_true)
    >>> calibrated_proba = calibrator.calibrate(y_proba_new)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    """Calibration methods."""
    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC = "isotonic"
    HISTOGRAM_BINNING = "histogram_binning"
    BBQ = "bbq"  # Bayesian Binning into Quantiles
    VECTOR_SCALING = "vector_scaling"


class CalibratorConfig(BaseModel):
    """Configuration for calibrator."""

    method: CalibrationMethod = Field(
        default=CalibrationMethod.TEMPERATURE_SCALING,
        description="Calibration method"
    )
    n_bins: int = Field(
        default=15,
        ge=5,
        le=100,
        description="Number of bins for histogram-based methods"
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        description="Maximum iterations for optimization"
    )
    tolerance: float = Field(
        default=1e-8,
        gt=0,
        description="Convergence tolerance"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class CalibrationMetrics(BaseModel):
    """Calibration quality metrics."""

    ece: float = Field(
        ...,
        description="Expected Calibration Error"
    )
    mce: float = Field(
        ...,
        description="Maximum Calibration Error"
    )
    brier_score: float = Field(
        ...,
        description="Brier Score"
    )
    reliability_diagram: Dict[str, List[float]] = Field(
        ...,
        description="Data for reliability diagram"
    )
    is_calibrated: bool = Field(
        ...,
        description="Whether model is well-calibrated"
    )
    calibration_threshold: float = Field(
        default=0.05,
        description="ECE threshold for good calibration"
    )


class CalibrationResult(BaseModel):
    """Result from calibration."""

    calibrated_probabilities: List[float] = Field(
        ...,
        description="Calibrated probability estimates"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Learned temperature (for temp scaling)"
    )
    metrics_before: CalibrationMetrics = Field(
        ...,
        description="Calibration metrics before"
    )
    metrics_after: CalibrationMetrics = Field(
        ...,
        description="Calibration metrics after"
    )
    improvement: float = Field(
        ...,
        description="ECE improvement ratio"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calibration timestamp"
    )


class Calibrator:
    """
    Probability Calibrator for GreenLang ML models.

    This class provides calibration methods to ensure that predicted
    probabilities are well-calibrated, meaning predicted confidence
    levels match actual accuracy.

    Key capabilities:
    - Temperature scaling (simple, effective)
    - Platt scaling (sigmoid fit)
    - Isotonic regression
    - Histogram binning
    - Calibration quality metrics

    Attributes:
        config: Calibrator configuration
        temperature: Learned temperature parameter
        _platt_params: Platt scaling parameters
        _isotonic_model: Isotonic regression model

    Example:
        >>> calibrator = Calibrator(config=CalibratorConfig(
        ...     method=CalibrationMethod.TEMPERATURE_SCALING
        ... ))
        >>> calibrator.fit(logits_val, y_val)
        >>> calibrated = calibrator.calibrate(logits_test)
    """

    def __init__(self, config: Optional[CalibratorConfig] = None):
        """
        Initialize calibrator.

        Args:
            config: Calibrator configuration
        """
        self.config = config or CalibratorConfig()
        self.temperature: float = 1.0
        self._platt_params: Tuple[float, float] = (1.0, 0.0)
        self._isotonic_model: Optional[Any] = None
        self._bin_boundaries: np.ndarray = np.array([])
        self._bin_corrections: np.ndarray = np.array([])
        self._is_fitted = False

        logger.info(
            f"Calibrator initialized: method={self.config.method}"
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _nll_loss(self, temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Negative log-likelihood loss for temperature scaling.

        Args:
            temperature: Temperature parameter
            logits: Model logits
            labels: True labels

        Returns:
            NLL loss value
        """
        scaled_logits = logits / temperature
        probs = self._softmax(scaled_logits)

        # Handle binary case
        if len(probs.shape) == 1 or probs.shape[1] == 1:
            probs = probs.flatten()
            # Binary cross-entropy
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            loss = -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )
        else:
            # Multi-class cross-entropy
            log_probs = np.log(probs + 1e-15)
            loss = -np.mean(log_probs[np.arange(len(labels)), labels.astype(int)])

        return loss

    def _calculate_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Tuple[float, Dict[str, List[float]]]:
        """
        Calculate Expected Calibration Error.

        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_bins: Number of bins

        Returns:
            Tuple of (ECE, reliability diagram data)
        """
        n_bins = n_bins or self.config.n_bins

        # Get confidences and predictions
        if len(probabilities.shape) > 1:
            confidences = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
        else:
            confidences = np.maximum(probabilities, 1 - probabilities)
            predictions = (probabilities > 0.5).astype(int)

        accuracies = (predictions == labels).astype(float)

        # Bin boundaries
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if np.sum(in_bin) > 0:
                bin_acc = np.mean(accuracies[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                bin_count = np.sum(in_bin)

                ece += np.abs(bin_acc - bin_conf) * bin_count / len(probabilities)

                bin_accuracies.append(float(bin_acc))
                bin_confidences.append(float(bin_conf))
                bin_counts.append(int(bin_count))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_counts.append(0)

        reliability_data = {
            "bin_centers": [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)],
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts
        }

        return float(ece), reliability_data

    def _calculate_mce(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """Calculate Maximum Calibration Error."""
        n_bins = n_bins or self.config.n_bins

        if len(probabilities.shape) > 1:
            confidences = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
        else:
            confidences = np.maximum(probabilities, 1 - probabilities)
            predictions = (probabilities > 0.5).astype(int)

        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if np.sum(in_bin) > 0:
                bin_acc = np.mean(accuracies[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                mce = max(mce, np.abs(bin_acc - bin_conf))

        return float(mce)

    def _calculate_brier(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate Brier Score."""
        if len(probabilities.shape) > 1:
            # Multi-class: one-hot encode labels
            n_classes = probabilities.shape[1]
            one_hot = np.zeros((len(labels), n_classes))
            one_hot[np.arange(len(labels)), labels.astype(int)] = 1
            return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        else:
            return float(np.mean((probabilities - labels) ** 2))

    def _get_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> CalibrationMetrics:
        """Calculate all calibration metrics."""
        ece, reliability_diagram = self._calculate_ece(probabilities, labels)
        mce = self._calculate_mce(probabilities, labels)
        brier = self._calculate_brier(probabilities, labels)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_diagram=reliability_diagram,
            is_calibrated=ece < 0.05,
            calibration_threshold=0.05
        )

    def _calculate_provenance(
        self,
        calibrated: np.ndarray,
        temperature: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{self.config.method.value}|{temperature:.8f}|"
            f"{calibrated.sum():.8f}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def fit(
        self,
        logits_or_proba: np.ndarray,
        labels: np.ndarray,
        is_logits: bool = True
    ) -> "Calibrator":
        """
        Fit calibrator on validation data.

        Args:
            logits_or_proba: Model logits or probabilities
            labels: True labels
            is_logits: Whether input is logits (vs probabilities)

        Returns:
            self

        Example:
            >>> calibrator.fit(logits_val, y_val)
        """
        logger.info(f"Fitting calibrator with {len(labels)} samples")

        if is_logits:
            logits = logits_or_proba
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)
        else:
            # Convert probabilities to logits
            proba = np.clip(logits_or_proba, 1e-15, 1 - 1e-15)
            logits = np.log(proba / (1 - proba))
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)

        if self.config.method == CalibrationMethod.TEMPERATURE_SCALING:
            self._fit_temperature_scaling(logits, labels)

        elif self.config.method == CalibrationMethod.PLATT_SCALING:
            self._fit_platt_scaling(logits, labels)

        elif self.config.method == CalibrationMethod.ISOTONIC:
            self._fit_isotonic(logits, labels)

        elif self.config.method == CalibrationMethod.HISTOGRAM_BINNING:
            self._fit_histogram_binning(logits, labels)

        else:
            self._fit_temperature_scaling(logits, labels)

        self._is_fitted = True
        logger.info(f"Calibrator fitted: temperature={self.temperature:.4f}")

        return self

    def _fit_temperature_scaling(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit temperature scaling."""
        result = minimize(
            lambda t: self._nll_loss(t[0], logits, labels),
            x0=[1.0],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0)],
            options={"maxiter": self.config.max_iter, "ftol": self.config.tolerance}
        )

        self.temperature = float(result.x[0])

    def _fit_platt_scaling(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit Platt scaling (sigmoid with A and B)."""
        if len(logits.shape) > 1:
            logits = logits[:, -1]  # Use last class logit for binary

        def platt_loss(params):
            a, b = params
            proba = 1 / (1 + np.exp(-a * logits - b))
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return -np.mean(
                labels * np.log(proba) + (1 - labels) * np.log(1 - proba)
            )

        result = minimize(
            platt_loss,
            x0=[1.0, 0.0],
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iter}
        )

        self._platt_params = tuple(result.x)

    def _fit_isotonic(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit isotonic regression."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            raise ImportError("sklearn required for isotonic calibration")

        if len(logits.shape) > 1:
            proba = self._softmax(logits)[:, -1]
        else:
            proba = 1 / (1 + np.exp(-logits))

        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self._isotonic_model.fit(proba, labels)

    def _fit_histogram_binning(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit histogram binning."""
        if len(logits.shape) > 1:
            proba = self._softmax(logits)[:, -1]
        else:
            proba = 1 / (1 + np.exp(-logits))

        self._bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        self._bin_corrections = np.zeros(self.config.n_bins)

        for i in range(self.config.n_bins):
            in_bin = (proba > self._bin_boundaries[i]) & (proba <= self._bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                self._bin_corrections[i] = np.mean(labels[in_bin])
            else:
                self._bin_corrections[i] = (self._bin_boundaries[i] + self._bin_boundaries[i + 1]) / 2

    def calibrate(
        self,
        logits_or_proba: np.ndarray,
        is_logits: bool = True
    ) -> np.ndarray:
        """
        Calibrate new predictions.

        Args:
            logits_or_proba: Model logits or probabilities
            is_logits: Whether input is logits

        Returns:
            Calibrated probabilities

        Example:
            >>> calibrated = calibrator.calibrate(logits_test)
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        if is_logits:
            logits = logits_or_proba
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)
        else:
            proba = np.clip(logits_or_proba, 1e-15, 1 - 1e-15)
            logits = np.log(proba / (1 - proba))
            if len(logits.shape) == 1:
                logits = logits.reshape(-1, 1)

        if self.config.method == CalibrationMethod.TEMPERATURE_SCALING:
            scaled_logits = logits / self.temperature
            return self._softmax(scaled_logits)

        elif self.config.method == CalibrationMethod.PLATT_SCALING:
            if len(logits.shape) > 1:
                logits = logits[:, -1]
            a, b = self._platt_params
            return 1 / (1 + np.exp(-a * logits - b))

        elif self.config.method == CalibrationMethod.ISOTONIC:
            if len(logits.shape) > 1:
                proba = self._softmax(logits)[:, -1]
            else:
                proba = 1 / (1 + np.exp(-logits))
            return self._isotonic_model.transform(proba)

        elif self.config.method == CalibrationMethod.HISTOGRAM_BINNING:
            if len(logits.shape) > 1:
                proba = self._softmax(logits)[:, -1]
            else:
                proba = 1 / (1 + np.exp(-logits))

            calibrated = np.zeros_like(proba)
            for i in range(self.config.n_bins):
                in_bin = (proba > self._bin_boundaries[i]) & (proba <= self._bin_boundaries[i + 1])
                calibrated[in_bin] = self._bin_corrections[i]

            return calibrated

        return self._softmax(logits / self.temperature)

    def calibrate_with_metrics(
        self,
        logits_or_proba: np.ndarray,
        labels: np.ndarray,
        is_logits: bool = True
    ) -> CalibrationResult:
        """
        Calibrate and return full metrics.

        Args:
            logits_or_proba: Model logits or probabilities
            labels: True labels
            is_logits: Whether input is logits

        Returns:
            CalibrationResult with metrics
        """
        if is_logits:
            before_proba = self._softmax(logits_or_proba)
        else:
            before_proba = logits_or_proba

        calibrated = self.calibrate(logits_or_proba, is_logits)

        metrics_before = self._get_metrics(before_proba, labels)
        metrics_after = self._get_metrics(calibrated, labels)

        improvement = (metrics_before.ece - metrics_after.ece) / (metrics_before.ece + 1e-10)

        provenance_hash = self._calculate_provenance(calibrated, self.temperature)

        return CalibrationResult(
            calibrated_probabilities=calibrated.flatten().tolist(),
            temperature=self.temperature if self.config.method == CalibrationMethod.TEMPERATURE_SCALING else None,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=float(improvement),
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )


class ReliabilityDiagramData(BaseModel):
    """Data for generating reliability diagrams."""

    bin_centers: List[float] = Field(
        ...,
        description="Center of each confidence bin"
    )
    bin_accuracies: List[float] = Field(
        ...,
        description="Accuracy in each bin"
    )
    bin_confidences: List[float] = Field(
        ...,
        description="Mean confidence in each bin"
    )
    bin_counts: List[int] = Field(
        ...,
        description="Number of samples in each bin"
    )
    ece: float = Field(
        ...,
        description="Expected Calibration Error"
    )
    mce: float = Field(
        ...,
        description="Maximum Calibration Error"
    )
    n_bins: int = Field(
        ...,
        description="Number of bins"
    )
    n_samples: int = Field(
        ...,
        description="Total number of samples"
    )
    is_well_calibrated: bool = Field(
        ...,
        description="Whether ECE is below threshold"
    )
    calibration_gap: List[float] = Field(
        ...,
        description="Gap between accuracy and confidence per bin"
    )


class CalibrationDiagnostics:
    """
    Calibration Diagnostics for comprehensive analysis.

    Provides detailed diagnostic information about model calibration,
    including reliability diagrams, calibration gaps, and statistical tests.

    Example:
        >>> diagnostics = CalibrationDiagnostics()
        >>> diagram_data = diagnostics.compute_reliability_diagram(probas, labels)
        >>> print(f"ECE: {diagram_data.ece:.4f}")
    """

    def __init__(self, n_bins: int = 15, threshold: float = 0.05):
        """
        Initialize calibration diagnostics.

        Args:
            n_bins: Number of bins for reliability diagram
            threshold: ECE threshold for good calibration
        """
        self.n_bins = n_bins
        self.threshold = threshold

    def compute_reliability_diagram(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> ReliabilityDiagramData:
        """
        Compute data for reliability diagram.

        Args:
            probabilities: Predicted probabilities
            labels: True labels

        Returns:
            ReliabilityDiagramData for plotting
        """
        # Handle multi-class vs binary
        if len(probabilities.shape) > 1:
            confidences = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
        else:
            confidences = np.maximum(probabilities, 1 - probabilities)
            predictions = (probabilities > 0.5).astype(int)

        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        calibration_gaps = []

        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            bin_centers.append(float(bin_center))

            if np.sum(in_bin) > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                bin_count = int(np.sum(in_bin))
                gap = abs(bin_acc - bin_conf)

                ece += gap * bin_count / len(probabilities)
                mce = max(mce, gap)

                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
                calibration_gaps.append(float(bin_acc - bin_conf))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(bin_center)
                bin_counts.append(0)
                calibration_gaps.append(0.0)

        return ReliabilityDiagramData(
            bin_centers=bin_centers,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            ece=float(ece),
            mce=float(mce),
            n_bins=self.n_bins,
            n_samples=len(probabilities),
            is_well_calibrated=ece < self.threshold,
            calibration_gap=calibration_gaps
        )

    def compute_calibration_curve(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Compute smooth calibration curve using kernel smoothing.

        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_points: Number of points in curve

        Returns:
            Dictionary with x and y values for curve
        """
        if len(probabilities.shape) > 1:
            probs = np.max(probabilities, axis=1)
            preds = np.argmax(probabilities, axis=1)
        else:
            probs = probabilities
            preds = (probabilities > 0.5).astype(int)

        accuracies = (preds == labels).astype(float)

        # Sort by probability
        sorted_indices = np.argsort(probs)
        sorted_probs = probs[sorted_indices]
        sorted_accs = accuracies[sorted_indices]

        # Compute cumulative statistics
        x_points = np.linspace(0, 1, n_points)
        y_points = []

        for x in x_points:
            # Use samples near this probability
            window = 0.1
            mask = (sorted_probs >= x - window) & (sorted_probs <= x + window)
            if np.sum(mask) > 0:
                y_points.append(float(np.mean(sorted_accs[mask])))
            else:
                y_points.append(float(x))  # Default to perfect calibration

        return {
            "predicted_probability": x_points.tolist(),
            "empirical_accuracy": y_points,
            "perfect_calibration": x_points.tolist()
        }

    def compute_class_calibration(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute calibration metrics per class.

        Args:
            probabilities: Predicted probabilities (n_samples x n_classes)
            labels: True labels

        Returns:
            Per-class calibration metrics
        """
        if len(probabilities.shape) == 1:
            # Binary case
            return {
                0: self._compute_class_metrics(1 - probabilities, (labels == 0).astype(int)),
                1: self._compute_class_metrics(probabilities, (labels == 1).astype(int))
            }

        n_classes = probabilities.shape[1]
        class_metrics = {}

        for c in range(n_classes):
            class_probs = probabilities[:, c]
            class_labels = (labels == c).astype(int)
            class_metrics[c] = self._compute_class_metrics(class_probs, class_labels)

        return class_metrics

    def _compute_class_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute calibration metrics for single class."""
        # Binary cross-entropy
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        bce = -np.mean(
            labels * np.log(probs_clipped) +
            (1 - labels) * np.log(1 - probs_clipped)
        )

        # Brier score for this class
        brier = float(np.mean((probs - labels) ** 2))

        # ECE for this class
        diagram = self.compute_reliability_diagram(probs, labels)

        return {
            "ece": diagram.ece,
            "brier_score": brier,
            "binary_cross_entropy": float(bce),
            "mean_predicted_prob": float(np.mean(probs)),
            "actual_frequency": float(np.mean(labels))
        }


class BinaryCalibrator:
    """
    Specialized calibrator for binary classification.

    Optimized for binary classification with Platt scaling
    and isotonic regression.

    Example:
        >>> calibrator = BinaryCalibrator(method="platt")
        >>> calibrator.fit(probas, labels)
        >>> calibrated = calibrator.calibrate(new_probas)
    """

    def __init__(self, method: str = "platt"):
        """
        Initialize binary calibrator.

        Args:
            method: "platt" or "isotonic"
        """
        self.method = method
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._isotonic_model = None
        self._is_fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> "BinaryCalibrator":
        """
        Fit the calibrator.

        Args:
            probabilities: Predicted probabilities for positive class
            labels: True binary labels

        Returns:
            self
        """
        if self.method == "platt":
            self._fit_platt(probabilities, labels)
        else:
            self._fit_isotonic(probabilities, labels)

        self._is_fitted = True
        return self

    def _fit_platt(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit Platt scaling for binary classification."""
        # Convert probabilities to logits
        eps = 1e-15
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        def platt_loss(params):
            a, b = params
            scaled = 1 / (1 + np.exp(-a * logits - b))
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(
                labels * np.log(scaled) +
                (1 - labels) * np.log(1 - scaled)
            )

        result = minimize(
            platt_loss,
            x0=[1.0, 0.0],
            method="L-BFGS-B",
            options={"maxiter": 1000}
        )

        self._platt_a, self._platt_b = result.x

    def _fit_isotonic(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Fit isotonic regression."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            raise ImportError("sklearn required for isotonic calibration")

        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self._isotonic_model.fit(probabilities, labels)

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            probabilities: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted.")

        if self.method == "platt":
            eps = 1e-15
            probs_clipped = np.clip(probabilities, eps, 1 - eps)
            logits = np.log(probs_clipped / (1 - probs_clipped))
            return 1 / (1 + np.exp(-self._platt_a * logits - self._platt_b))
        else:
            return self._isotonic_model.transform(probabilities)


# Unit test stubs
class TestCalibrator:
    """Unit tests for Calibrator."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        calibrator = Calibrator()
        assert calibrator.config.method == CalibrationMethod.TEMPERATURE_SCALING

    def test_ece_calculation(self):
        """Test ECE calculation."""
        calibrator = Calibrator()

        # Perfect calibration
        proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # Approximate expected accuracy for each bin
        labels = np.array([0, 0, 1, 1, 1])

        ece, _ = calibrator._calculate_ece(proba, labels, n_bins=5)
        assert ece >= 0  # ECE is non-negative

    def test_temperature_scaling(self):
        """Test temperature scaling."""
        calibrator = Calibrator(config=CalibratorConfig(
            method=CalibrationMethod.TEMPERATURE_SCALING
        ))

        logits = np.random.randn(100, 2) * 2
        labels = np.random.randint(0, 2, 100)

        calibrator.fit(logits, labels)
        assert calibrator.temperature > 0

    def test_calibrate(self):
        """Test calibration."""
        calibrator = Calibrator()

        logits = np.random.randn(100, 2)
        labels = np.random.randint(0, 2, 100)

        calibrator.fit(logits, labels)

        test_logits = np.random.randn(10, 2)
        calibrated = calibrator.calibrate(test_logits)

        assert calibrated.shape == test_logits.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        calibrator = Calibrator()

        calibrated = np.array([0.5, 0.6, 0.7])
        hash1 = calibrator._calculate_provenance(calibrated, 1.5)
        hash2 = calibrator._calculate_provenance(calibrated, 1.5)

        assert hash1 == hash2
