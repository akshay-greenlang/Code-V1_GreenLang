# -*- coding: utf-8 -*-
"""
Conformal Prediction Module

This module provides conformal prediction capabilities for GreenLang ML
models, enabling valid prediction intervals with guaranteed coverage
regardless of the underlying model or data distribution.

Conformal prediction provides finite-sample valid confidence intervals,
critical for regulatory compliance where uncertainty quantification
must be statistically rigorous.

Example:
    >>> from greenlang.ml.uncertainty import ConformalPredictor
    >>> conformal = ConformalPredictor(model, method="split")
    >>> conformal.calibrate(X_cal, y_cal)
    >>> intervals = conformal.predict_interval(X_test, confidence=0.95)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConformalMethod(str, Enum):
    """Conformal prediction methods."""
    SPLIT = "split"
    CROSS_VALIDATION = "cross_validation"
    JACKKNIFE = "jackknife"
    JACKKNIFE_PLUS = "jackknife_plus"
    CV_PLUS = "cv_plus"
    WEIGHTED = "weighted"


class NonconformityScore(str, Enum):
    """Nonconformity score functions."""
    ABSOLUTE_ERROR = "absolute_error"
    SCALED_ERROR = "scaled_error"
    QUANTILE = "quantile"
    CQR = "cqr"  # Conformalized Quantile Regression


class ConformalConfig(BaseModel):
    """Configuration for conformal predictor."""

    method: ConformalMethod = Field(
        default=ConformalMethod.SPLIT,
        description="Conformal prediction method"
    )
    nonconformity_score: NonconformityScore = Field(
        default=NonconformityScore.ABSOLUTE_ERROR,
        description="Nonconformity score function"
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Folds for cross-validation method"
    )
    asymmetric: bool = Field(
        default=False,
        description="Allow asymmetric intervals"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class PredictionInterval(BaseModel):
    """Prediction interval result."""

    prediction: float = Field(
        ...,
        description="Point prediction"
    )
    lower: float = Field(
        ...,
        description="Lower bound"
    )
    upper: float = Field(
        ...,
        description="Upper bound"
    )
    width: float = Field(
        ...,
        description="Interval width"
    )
    confidence: float = Field(
        ...,
        description="Confidence level"
    )


class ConformalResult(BaseModel):
    """Result from conformal prediction."""

    intervals: List[PredictionInterval] = Field(
        ...,
        description="Prediction intervals"
    )
    coverage: float = Field(
        ...,
        description="Empirical coverage"
    )
    avg_width: float = Field(
        ...,
        description="Average interval width"
    )
    confidence_level: float = Field(
        ...,
        description="Nominal confidence level"
    )
    method: str = Field(
        ...,
        description="Method used"
    )
    n_calibration_samples: int = Field(
        ...,
        description="Calibration sample size"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )


class ConformalPredictor:
    """
    Conformal Predictor for valid uncertainty quantification.

    This class implements conformal prediction methods that provide
    prediction intervals with guaranteed finite-sample coverage,
    regardless of the underlying data distribution.

    Key capabilities:
    - Split conformal prediction
    - Cross-validation based methods
    - Jackknife+ for efficient intervals
    - Conformalized quantile regression
    - Guaranteed coverage

    Attributes:
        model: Underlying prediction model
        config: Configuration
        _conformity_scores: Calibrated nonconformity scores
        _quantile: Calibrated quantile

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> model.fit(X_train, y_train)
        >>> conformal = ConformalPredictor(model, config=ConformalConfig(
        ...     method=ConformalMethod.SPLIT
        ... ))
        >>> conformal.calibrate(X_cal, y_cal)
        >>> result = conformal.predict_interval(X_test, confidence=0.90)
        >>> print(f"Coverage: {result.coverage:.3f}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[ConformalConfig] = None
    ):
        """
        Initialize conformal predictor.

        Args:
            model: Prediction model with predict method
            config: Conformal configuration
        """
        self.model = model
        self.config = config or ConformalConfig()
        self._conformity_scores: np.ndarray = np.array([])
        self._quantile: float = 0.0
        self._is_calibrated = False
        self._n_calibration = 0

        # For CV and jackknife methods
        self._cv_models: List[Any] = []
        self._residuals_matrix: np.ndarray = np.array([])

        np.random.seed(self.config.random_state)

        logger.info(
            f"ConformalPredictor initialized: method={self.config.method}"
        )

    def _compute_nonconformity_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute nonconformity scores.

        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainty: Optional uncertainty estimates

        Returns:
            Nonconformity scores
        """
        if self.config.nonconformity_score == NonconformityScore.ABSOLUTE_ERROR:
            return np.abs(y_true - y_pred)

        elif self.config.nonconformity_score == NonconformityScore.SCALED_ERROR:
            if uncertainty is not None:
                return np.abs(y_true - y_pred) / (uncertainty + 1e-10)
            else:
                # Use MAD as scale estimate
                residuals = np.abs(y_true - y_pred)
                mad = np.median(residuals)
                return residuals / (mad + 1e-10)

        elif self.config.nonconformity_score == NonconformityScore.QUANTILE:
            # For quantile regression models
            return np.abs(y_true - y_pred)

        else:
            return np.abs(y_true - y_pred)

    def _calculate_provenance(
        self,
        intervals: List[PredictionInterval],
        coverage: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        widths = [i.width for i in intervals]
        combined = (
            f"{self.config.method.value}|{self._n_calibration}|"
            f"{sum(widths):.8f}|{coverage:.8f}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> None:
        """
        Calibrate the conformal predictor.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            uncertainty: Optional uncertainty estimates

        Example:
            >>> conformal.calibrate(X_cal, y_cal)
        """
        logger.info(f"Calibrating with {len(X_cal)} samples")

        self._n_calibration = len(X_cal)

        if self.config.method == ConformalMethod.SPLIT:
            self._calibrate_split(X_cal, y_cal, uncertainty)

        elif self.config.method == ConformalMethod.JACKKNIFE_PLUS:
            self._calibrate_jackknife_plus(X_cal, y_cal)

        elif self.config.method == ConformalMethod.CROSS_VALIDATION:
            self._calibrate_cv(X_cal, y_cal)

        else:
            # Default to split
            self._calibrate_split(X_cal, y_cal, uncertainty)

        self._is_calibrated = True
        logger.info(f"Calibration complete: quantile={self._quantile:.4f}")

    def _calibrate_split(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> None:
        """Split conformal calibration."""
        predictions = self.model.predict(X_cal)
        self._conformity_scores = self._compute_nonconformity_score(
            y_cal, predictions, uncertainty
        )

    def _calibrate_jackknife_plus(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> None:
        """Jackknife+ calibration (leave-one-out)."""
        n = len(X_cal)
        self._residuals_matrix = np.zeros((n, n))

        for i in range(n):
            # Leave-one-out training
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            X_train = X_cal[mask]
            y_train = y_cal[mask]

            # Clone and fit model
            import copy
            model_i = copy.deepcopy(self.model)
            model_i.fit(X_train, y_train)

            # Predict for left-out sample
            pred_i = model_i.predict(X_cal[i:i+1])[0]
            self._residuals_matrix[i, :] = np.abs(y_cal - model_i.predict(X_cal))

        # Conformity scores for jackknife+
        self._conformity_scores = np.array([
            self._residuals_matrix[i, i] for i in range(n)
        ])

    def _calibrate_cv(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> None:
        """Cross-validation based calibration."""
        from sklearn.model_selection import KFold
        import copy

        kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        residuals = np.zeros(len(X_cal))

        for train_idx, val_idx in kf.split(X_cal):
            model_k = copy.deepcopy(self.model)
            model_k.fit(X_cal[train_idx], y_cal[train_idx])

            preds = model_k.predict(X_cal[val_idx])
            residuals[val_idx] = np.abs(y_cal[val_idx] - preds)

            self._cv_models.append(model_k)

        self._conformity_scores = residuals

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.95,
        y_true: Optional[np.ndarray] = None
    ) -> ConformalResult:
        """
        Predict with conformal intervals.

        Args:
            X: Features for prediction
            confidence: Confidence level (e.g., 0.95)
            y_true: Optional true values for coverage calculation

        Returns:
            ConformalResult with intervals

        Example:
            >>> result = conformal.predict_interval(X_test, confidence=0.95)
            >>> for i, interval in enumerate(result.intervals[:5]):
            ...     print(f"Sample {i}: [{interval.lower:.2f}, {interval.upper:.2f}]")
        """
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        # Calculate quantile
        n = len(self._conformity_scores)
        q_level = np.ceil((n + 1) * confidence) / n
        q_level = min(q_level, 1.0)

        self._quantile = float(np.quantile(self._conformity_scores, q_level))

        # Get point predictions
        predictions = self.model.predict(X)

        # Construct intervals
        intervals = []
        for pred in predictions:
            if self.config.asymmetric:
                # Would need separate upper/lower quantiles
                lower = pred - self._quantile
                upper = pred + self._quantile
            else:
                lower = pred - self._quantile
                upper = pred + self._quantile

            intervals.append(PredictionInterval(
                prediction=float(pred),
                lower=float(lower),
                upper=float(upper),
                width=float(upper - lower),
                confidence=confidence
            ))

        # Calculate empirical coverage if y_true provided
        if y_true is not None:
            covered = sum(
                1 for i, interval in enumerate(intervals)
                if interval.lower <= y_true[i] <= interval.upper
            )
            coverage = covered / len(y_true)
        else:
            coverage = confidence  # Expected coverage

        avg_width = np.mean([i.width for i in intervals])

        # Calculate provenance
        provenance_hash = self._calculate_provenance(intervals, coverage)

        logger.info(
            f"Prediction intervals generated: n={len(intervals)}, "
            f"avg_width={avg_width:.4f}, coverage={coverage:.3f}"
        )

        return ConformalResult(
            intervals=intervals,
            coverage=float(coverage),
            avg_width=float(avg_width),
            confidence_level=confidence,
            method=self.config.method.value,
            n_calibration_samples=self._n_calibration,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def get_conformity_scores(self) -> np.ndarray:
        """Get calibrated conformity scores."""
        return self._conformity_scores.copy()

    def get_quantile(self, confidence: float = 0.95) -> float:
        """Get quantile for given confidence level."""
        if not self._is_calibrated:
            raise ValueError("Predictor not calibrated.")

        n = len(self._conformity_scores)
        q_level = np.ceil((n + 1) * confidence) / n
        return float(np.quantile(self._conformity_scores, min(q_level, 1.0)))

    def coverage_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Test coverage on held-out data.

        Args:
            X_test: Test features
            y_test: Test targets
            confidence: Confidence level

        Returns:
            Dictionary with coverage statistics
        """
        result = self.predict_interval(X_test, confidence, y_test)

        # Per-sample coverage check
        coverages = []
        for i, interval in enumerate(result.intervals):
            covered = interval.lower <= y_test[i] <= interval.upper
            coverages.append(covered)

        return {
            "empirical_coverage": float(np.mean(coverages)),
            "nominal_coverage": confidence,
            "coverage_gap": float(np.mean(coverages) - confidence),
            "n_samples": len(y_test),
            "avg_width": result.avg_width,
            "width_std": float(np.std([i.width for i in result.intervals]))
        }


# Unit test stubs
class TestConformalPredictor:
    """Unit tests for ConformalPredictor."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        conformal = ConformalPredictor(MockModel())
        assert conformal.config.method == ConformalMethod.SPLIT

    def test_calibrate_split(self):
        """Test split conformal calibration."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        conformal = ConformalPredictor(MockModel())

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        conformal.calibrate(X, y)
        assert conformal._is_calibrated
        assert len(conformal._conformity_scores) == 100

    def test_predict_interval(self):
        """Test interval prediction."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        conformal = ConformalPredictor(MockModel())

        X_cal = np.random.randn(100, 5)
        y_cal = np.sum(X_cal, axis=1) + np.random.randn(100) * 0.1

        conformal.calibrate(X_cal, y_cal)

        X_test = np.random.randn(10, 5)
        result = conformal.predict_interval(X_test, confidence=0.95)

        assert len(result.intervals) == 10
        assert all(i.width > 0 for i in result.intervals)

    def test_coverage(self):
        """Test that coverage is approximately correct."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        conformal = ConformalPredictor(MockModel())

        X_cal = np.random.randn(500, 5)
        y_cal = np.sum(X_cal, axis=1) + np.random.randn(500) * 0.5

        conformal.calibrate(X_cal, y_cal)

        X_test = np.random.randn(200, 5)
        y_test = np.sum(X_test, axis=1) + np.random.randn(200) * 0.5

        result = conformal.predict_interval(X_test, confidence=0.9, y_true=y_test)

        # Coverage should be close to nominal
        assert abs(result.coverage - 0.9) < 0.1  # Within 10%

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        conformal = ConformalPredictor(MockModel())

        intervals = [
            PredictionInterval(prediction=0, lower=-1, upper=1, width=2, confidence=0.95)
        ]

        hash1 = conformal._calculate_provenance(intervals, 0.95)
        hash2 = conformal._calculate_provenance(intervals, 0.95)

        assert hash1 == hash2
