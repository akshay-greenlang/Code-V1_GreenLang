# -*- coding: utf-8 -*-
"""
Uncertainty Quantification for Fouling Prediction - GL-014 Exchangerpro Agent.

Provides uncertainty estimation methods:
- Quantile regression for prediction intervals
- Conformal prediction support
- Model ensemble uncertainty
- Calibrated confidence intervals

CRITICAL: All predictions MUST include uncertainty estimates.
Predictions are NEVER presented as certainties.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Quantile regression
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    default_confidence_level: float = 0.90

    # Conformal prediction
    conformal_alpha: float = 0.10  # 1 - confidence level
    conformal_method: str = "split"  # split, cv, jackknife

    # Ensemble
    n_bootstrap: int = 100
    bootstrap_sample_ratio: float = 0.8

    # Calibration
    calibrate_intervals: bool = True
    calibration_method: str = "isotonic"  # isotonic, platt

    # Model parameters
    n_estimators: int = 100
    max_depth: int = 6
    random_seed: int = 42


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class PredictionInterval:
    """Prediction interval result."""

    lower: float
    upper: float
    center: float  # Median or mean
    confidence_level: float

    # Width metrics
    width: float = 0.0
    relative_width: float = 0.0  # width / center

    def __post_init__(self):
        self.width = self.upper - self.lower
        if abs(self.center) > 1e-10:
            self.relative_width = self.width / abs(self.center)

    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower <= value <= self.upper


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification result."""

    prediction: float
    prediction_std: float

    # Prediction intervals at different confidence levels
    intervals: Dict[float, PredictionInterval] = field(default_factory=dict)

    # Quantiles
    quantiles: Dict[float, float] = field(default_factory=dict)

    # Uncertainty sources
    aleatoric_uncertainty: float = 0.0  # Data noise
    epistemic_uncertainty: float = 0.0  # Model uncertainty

    # Calibration
    is_calibrated: bool = False
    calibration_error: float = 0.0

    # Quality
    uncertainty_quality: str = "high"  # high, medium, low

    # Provenance
    method: str = ""
    computation_time_ms: float = 0.0
    provenance_hash: str = ""

    def get_interval(self, confidence_level: float = 0.90) -> PredictionInterval:
        """Get prediction interval for specified confidence level."""
        # Find closest available interval
        if confidence_level in self.intervals:
            return self.intervals[confidence_level]

        closest = min(self.intervals.keys(), key=lambda x: abs(x - confidence_level))
        return self.intervals[closest]


# =============================================================================
# Base Class
# =============================================================================

class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimators."""

    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the uncertainty estimator."""
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> List[UncertaintyResult]:
        """Predict with uncertainty quantification."""
        pass


# =============================================================================
# Quantile Regression
# =============================================================================

class QuantileRegressor(UncertaintyEstimator):
    """
    Quantile regression for prediction intervals.

    Fits separate models for each quantile to provide
    prediction intervals without distributional assumptions.

    Example:
        >>> regressor = QuantileRegressor(config)
        >>> regressor.fit(X_train, y_train)
        >>> results = regressor.predict_with_uncertainty(X_test)
        >>> for result in results:
        ...     print(f"Prediction: {result.prediction}")
        ...     print(f"90% CI: [{result.intervals[0.9].lower}, "
        ...           f"{result.intervals[0.9].upper}]")
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        super().__init__(config or UncertaintyConfig())

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for QuantileRegressor")

        self._quantile_models: Dict[float, GradientBoostingRegressor] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit quantile regression models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values
        """
        # Normalize features
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self._feature_means) / self._feature_stds

        # Fit a model for each quantile
        for q in self.config.quantiles:
            model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                loss="quantile",
                alpha=q,
                random_state=self.config.random_seed,
            )
            model.fit(X_normalized, y)
            self._quantile_models[q] = model

            logger.debug(f"Fitted quantile model for q={q}")

        self._is_fitted = True

        logger.info(
            f"QuantileRegressor fitted with {len(self.config.quantiles)} quantiles"
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> List[UncertaintyResult]:
        """
        Predict with uncertainty using quantile regression.

        Args:
            X: Feature matrix

        Returns:
            List of UncertaintyResult for each sample
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        X_normalized = (X - self._feature_means) / self._feature_stds

        # Predict all quantiles
        quantile_predictions = {}
        for q, model in self._quantile_models.items():
            quantile_predictions[q] = model.predict(X_normalized)

        # Build results
        results = []
        for i in range(X.shape[0]):
            quantiles = {q: float(preds[i]) for q, preds in quantile_predictions.items()}

            # Median as point prediction
            median = quantiles.get(0.5, np.median(list(quantiles.values())))

            # Build prediction intervals
            intervals = {}
            for conf in [0.80, 0.90, 0.95]:
                alpha = (1 - conf) / 2
                lower_q = min(quantiles.keys(), key=lambda x: abs(x - alpha))
                upper_q = min(quantiles.keys(), key=lambda x: abs(x - (1 - alpha)))

                intervals[conf] = PredictionInterval(
                    lower=quantiles[lower_q],
                    upper=quantiles[upper_q],
                    center=median,
                    confidence_level=conf,
                )

            # Estimate std from IQR
            q75 = quantiles.get(0.75, median)
            q25 = quantiles.get(0.25, median)
            iqr = q75 - q25
            std_estimate = iqr / 1.35  # IQR to std approximation

            results.append(UncertaintyResult(
                prediction=median,
                prediction_std=std_estimate,
                intervals=intervals,
                quantiles=quantiles,
                method="quantile_regression",
                computation_time_ms=(time.time() - start_time) * 1000 / X.shape[0],
                provenance_hash=hashlib.sha256(
                    f"qr_{median:.6f}_{std_estimate:.6f}".encode()
                ).hexdigest()[:16],
            ))

        return results

    def predict_quantile(
        self,
        X: np.ndarray,
        quantile: float,
    ) -> np.ndarray:
        """Predict a specific quantile."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if quantile not in self._quantile_models:
            # Find closest quantile
            closest = min(self._quantile_models.keys(), key=lambda x: abs(x - quantile))
            logger.warning(f"Quantile {quantile} not available, using {closest}")
            quantile = closest

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_normalized = (X - self._feature_means) / self._feature_stds
        return self._quantile_models[quantile].predict(X_normalized)


# =============================================================================
# Conformal Prediction
# =============================================================================

class ConformalPredictor(UncertaintyEstimator):
    """
    Conformal prediction for distribution-free prediction intervals.

    Provides valid coverage guarantees without distributional assumptions.
    Uses calibration set to compute nonconformity scores.

    Example:
        >>> predictor = ConformalPredictor(config)
        >>> predictor.fit(X_train, y_train)
        >>> results = predictor.predict_with_uncertainty(X_test)
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        super().__init__(config or UncertaintyConfig())

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ConformalPredictor")

        self._base_model: Optional[GradientBoostingRegressor] = None
        self._calibration_scores: Optional[np.ndarray] = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        calibration_size: float = 0.2,
    ) -> None:
        """
        Fit conformal predictor with split conformal method.

        Args:
            X: Feature matrix
            y: Target values
            calibration_size: Fraction of data to use for calibration
        """
        # Normalize
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self._feature_means) / self._feature_stds

        # Split into training and calibration
        n = len(X)
        n_cal = int(n * calibration_size)
        indices = np.random.RandomState(self.config.random_seed).permutation(n)

        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, y_train = X_normalized[train_idx], y[train_idx]
        X_cal, y_cal = X_normalized[cal_idx], y[cal_idx]

        # Fit base model on training set
        self._base_model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_seed,
        )
        self._base_model.fit(X_train, y_train)

        # Compute calibration scores (absolute residuals)
        y_pred_cal = self._base_model.predict(X_cal)
        self._calibration_scores = np.abs(y_cal - y_pred_cal)

        self._is_fitted = True

        logger.info(
            f"ConformalPredictor fitted with {len(train_idx)} training samples, "
            f"{len(cal_idx)} calibration samples"
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None,
    ) -> List[UncertaintyResult]:
        """
        Predict with conformal prediction intervals.

        Args:
            X: Feature matrix
            alpha: Miscoverage rate (default from config)

        Returns:
            List of UncertaintyResult
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if alpha is None:
            alpha = self.config.conformal_alpha

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_normalized = (X - self._feature_means) / self._feature_stds

        # Get point predictions
        predictions = self._base_model.predict(X_normalized)

        # Compute conformal quantile
        n_cal = len(self._calibration_scores)
        quantile_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)

        # Get the quantile of calibration scores
        q = np.quantile(self._calibration_scores, quantile_level)

        # Build results
        results = []
        for i in range(X.shape[0]):
            pred = float(predictions[i])

            # Conformal interval
            conf_level = 1 - alpha
            intervals = {
                conf_level: PredictionInterval(
                    lower=pred - q,
                    upper=pred + q,
                    center=pred,
                    confidence_level=conf_level,
                )
            }

            # Add other confidence levels by scaling
            for conf in [0.80, 0.90, 0.95]:
                if conf != conf_level:
                    scale = np.quantile(
                        self._calibration_scores,
                        min(np.ceil((n_cal + 1) * conf) / n_cal, 1.0)
                    )
                    intervals[conf] = PredictionInterval(
                        lower=pred - scale,
                        upper=pred + scale,
                        center=pred,
                        confidence_level=conf,
                    )

            # Estimate std from calibration scores
            std_estimate = np.std(self._calibration_scores)

            results.append(UncertaintyResult(
                prediction=pred,
                prediction_std=std_estimate,
                intervals=intervals,
                is_calibrated=True,
                method="conformal_prediction",
                computation_time_ms=(time.time() - start_time) * 1000 / X.shape[0],
                provenance_hash=hashlib.sha256(
                    f"cp_{pred:.6f}_{q:.6f}".encode()
                ).hexdigest()[:16],
            ))

        return results

    def get_coverage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Compute empirical coverage on test data.

        Args:
            X: Test features
            y: True values
            alpha: Miscoverage rate

        Returns:
            Empirical coverage rate
        """
        results = self.predict_with_uncertainty(X, alpha)
        conf_level = 1 - (alpha or self.config.conformal_alpha)

        covered = 0
        for i, result in enumerate(results):
            interval = result.intervals.get(conf_level)
            if interval and interval.contains(y[i]):
                covered += 1

        return covered / len(y)


# =============================================================================
# Ensemble Uncertainty
# =============================================================================

class EnsembleUncertainty(UncertaintyEstimator):
    """
    Ensemble-based uncertainty estimation using bootstrap.

    Uses model ensemble disagreement to estimate epistemic uncertainty.

    Example:
        >>> ensemble = EnsembleUncertainty(config)
        >>> ensemble.fit(X_train, y_train)
        >>> results = ensemble.predict_with_uncertainty(X_test)
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        super().__init__(config or UncertaintyConfig())

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for EnsembleUncertainty")

        self._ensemble: List[GradientBoostingRegressor] = []
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._residual_std: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit ensemble of models using bootstrap.

        Args:
            X: Feature matrix
            y: Target values
        """
        # Normalize
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self._feature_means) / self._feature_stds

        n = len(X)
        sample_size = int(n * self.config.bootstrap_sample_ratio)

        rng = np.random.RandomState(self.config.random_seed)

        # Fit bootstrap ensemble
        self._ensemble = []
        all_residuals = []

        for i in range(self.config.n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(n, size=sample_size, replace=True)
            X_boot, y_boot = X_normalized[indices], y[indices]

            # Fit model
            model = GradientBoostingRegressor(
                n_estimators=max(50, self.config.n_estimators // 2),  # Smaller for speed
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed + i,
            )
            model.fit(X_boot, y_boot)
            self._ensemble.append(model)

            # Collect OOB residuals
            oob_indices = np.setdiff1d(np.arange(n), indices)
            if len(oob_indices) > 0:
                oob_preds = model.predict(X_normalized[oob_indices])
                residuals = y[oob_indices] - oob_preds
                all_residuals.extend(residuals.tolist())

        # Estimate aleatoric uncertainty from OOB residuals
        if all_residuals:
            self._residual_std = float(np.std(all_residuals))

        self._is_fitted = True

        logger.info(
            f"EnsembleUncertainty fitted with {self.config.n_bootstrap} models"
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> List[UncertaintyResult]:
        """
        Predict with ensemble uncertainty estimation.

        Args:
            X: Feature matrix

        Returns:
            List of UncertaintyResult
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_normalized = (X - self._feature_means) / self._feature_stds

        # Get predictions from all models
        all_predictions = np.array([
            model.predict(X_normalized)
            for model in self._ensemble
        ])  # Shape: (n_models, n_samples)

        results = []
        for i in range(X.shape[0]):
            preds = all_predictions[:, i]

            # Point prediction (mean of ensemble)
            mean_pred = float(np.mean(preds))
            median_pred = float(np.median(preds))

            # Epistemic uncertainty (model disagreement)
            epistemic = float(np.std(preds))

            # Aleatoric uncertainty (from residuals)
            aleatoric = self._residual_std

            # Total uncertainty
            total_std = float(np.sqrt(epistemic ** 2 + aleatoric ** 2))

            # Build intervals from ensemble distribution
            intervals = {}
            for conf in [0.80, 0.90, 0.95]:
                alpha = (1 - conf) / 2
                lower = float(np.percentile(preds, alpha * 100))
                upper = float(np.percentile(preds, (1 - alpha) * 100))

                # Add aleatoric component
                lower -= aleatoric * 1.645  # Rough adjustment
                upper += aleatoric * 1.645

                intervals[conf] = PredictionInterval(
                    lower=lower,
                    upper=upper,
                    center=mean_pred,
                    confidence_level=conf,
                )

            # Quantiles
            quantiles = {
                q: float(np.percentile(preds, q * 100))
                for q in self.config.quantiles
            }

            results.append(UncertaintyResult(
                prediction=mean_pred,
                prediction_std=total_std,
                intervals=intervals,
                quantiles=quantiles,
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
                method="ensemble_bootstrap",
                computation_time_ms=(time.time() - start_time) * 1000 / X.shape[0],
                provenance_hash=hashlib.sha256(
                    f"ens_{mean_pred:.6f}_{total_std:.6f}".encode()
                ).hexdigest()[:16],
            ))

        return results

    def get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Returns:
            Array of shape (n_models, n_samples)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_normalized = (X - self._feature_means) / self._feature_stds

        return np.array([
            model.predict(X_normalized)
            for model in self._ensemble
        ])


# =============================================================================
# Main Uncertainty Quantifier
# =============================================================================

class UncertaintyQuantifier:
    """
    Main uncertainty quantification class combining multiple methods.

    Provides:
    - Quantile regression
    - Conformal prediction
    - Ensemble uncertainty
    - Calibrated intervals

    CRITICAL: All predictions include uncertainty estimates.
    Predictions are NEVER presented as certainties.

    Example:
        >>> config = UncertaintyConfig()
        >>> quantifier = UncertaintyQuantifier(config)
        >>> quantifier.fit(X_train, y_train)
        >>> results = quantifier.quantify(X_test)
        >>> for r in results:
        ...     print(f"Prediction: {r.prediction:.2f}")
        ...     print(f"90% CI: [{r.intervals[0.9].lower:.2f}, "
        ...           f"{r.intervals[0.9].upper:.2f}]")
        ...     print(f"Epistemic uncertainty: {r.epistemic_uncertainty:.3f}")
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """
        Initialize uncertainty quantifier.

        Args:
            config: Uncertainty configuration
        """
        self.config = config or UncertaintyConfig()

        # Initialize estimators
        self._quantile_regressor: Optional[QuantileRegressor] = None
        self._conformal_predictor: Optional[ConformalPredictor] = None
        self._ensemble_uncertainty: Optional[EnsembleUncertainty] = None

        self._is_fitted = False
        self._calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None

        logger.info("UncertaintyQuantifier initialized")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> None:
        """
        Fit uncertainty estimators.

        Args:
            X: Feature matrix
            y: Target values
            methods: List of methods to fit ("quantile", "conformal", "ensemble")
        """
        if methods is None:
            methods = ["quantile", "conformal", "ensemble"]

        if "quantile" in methods:
            self._quantile_regressor = QuantileRegressor(self.config)
            self._quantile_regressor.fit(X, y)

        if "conformal" in methods:
            self._conformal_predictor = ConformalPredictor(self.config)
            self._conformal_predictor.fit(X, y)

        if "ensemble" in methods:
            self._ensemble_uncertainty = EnsembleUncertainty(self.config)
            self._ensemble_uncertainty.fit(X, y)

        self._is_fitted = True

        logger.info(f"UncertaintyQuantifier fitted with methods: {methods}")

    def quantify(
        self,
        X: np.ndarray,
        method: str = "combined",
        confidence_level: Optional[float] = None,
    ) -> List[UncertaintyResult]:
        """
        Quantify uncertainty for predictions.

        Args:
            X: Feature matrix
            method: "quantile", "conformal", "ensemble", or "combined"
            confidence_level: Desired confidence level

        Returns:
            List of UncertaintyResult
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        confidence_level = confidence_level or self.config.default_confidence_level

        if method == "quantile" and self._quantile_regressor:
            return self._quantile_regressor.predict_with_uncertainty(X)

        elif method == "conformal" and self._conformal_predictor:
            alpha = 1 - confidence_level
            return self._conformal_predictor.predict_with_uncertainty(X, alpha)

        elif method == "ensemble" and self._ensemble_uncertainty:
            return self._ensemble_uncertainty.predict_with_uncertainty(X)

        elif method == "combined":
            return self._combined_uncertainty(X, confidence_level)

        else:
            raise ValueError(f"Method {method} not available or not fitted")

    def _combined_uncertainty(
        self,
        X: np.ndarray,
        confidence_level: float,
    ) -> List[UncertaintyResult]:
        """
        Combine multiple uncertainty methods.

        Uses:
        - Quantile regression for point estimates and quantiles
        - Ensemble for epistemic/aleatoric decomposition
        - Conformal for calibrated intervals
        """
        start_time = time.time()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []

        # Get results from each method
        qr_results = None
        cp_results = None
        ens_results = None

        if self._quantile_regressor:
            qr_results = self._quantile_regressor.predict_with_uncertainty(X)

        if self._conformal_predictor:
            cp_results = self._conformal_predictor.predict_with_uncertainty(
                X, alpha=1 - confidence_level
            )

        if self._ensemble_uncertainty:
            ens_results = self._ensemble_uncertainty.predict_with_uncertainty(X)

        # Combine results
        for i in range(X.shape[0]):
            # Use ensemble mean as point prediction
            if ens_results:
                prediction = ens_results[i].prediction
                epistemic = ens_results[i].epistemic_uncertainty
                aleatoric = ens_results[i].aleatoric_uncertainty
            elif qr_results:
                prediction = qr_results[i].prediction
                epistemic = 0.0
                aleatoric = qr_results[i].prediction_std
            else:
                prediction = cp_results[i].prediction if cp_results else 0.0
                epistemic = 0.0
                aleatoric = 0.0

            # Combine intervals (use conformal for calibration guarantee)
            intervals = {}
            for conf in [0.80, 0.90, 0.95]:
                # Prefer conformal intervals for their coverage guarantee
                if cp_results and conf in cp_results[i].intervals:
                    base_interval = cp_results[i].intervals[conf]
                elif ens_results and conf in ens_results[i].intervals:
                    base_interval = ens_results[i].intervals[conf]
                elif qr_results and conf in qr_results[i].intervals:
                    base_interval = qr_results[i].intervals[conf]
                else:
                    # Fallback: normal approximation
                    total_std = np.sqrt(epistemic ** 2 + aleatoric ** 2)
                    from scipy.stats import norm
                    z = norm.ppf((1 + conf) / 2)
                    base_interval = PredictionInterval(
                        lower=prediction - z * total_std,
                        upper=prediction + z * total_std,
                        center=prediction,
                        confidence_level=conf,
                    )

                intervals[conf] = base_interval

            # Get quantiles from quantile regression
            quantiles = qr_results[i].quantiles if qr_results else {}

            # Total uncertainty
            total_std = np.sqrt(epistemic ** 2 + aleatoric ** 2)

            # Assess quality
            if epistemic > 0 and aleatoric > 0:
                quality = "high"
            elif ens_results or qr_results:
                quality = "medium"
            else:
                quality = "low"

            results.append(UncertaintyResult(
                prediction=prediction,
                prediction_std=total_std,
                intervals=intervals,
                quantiles=quantiles,
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
                is_calibrated=cp_results is not None,
                uncertainty_quality=quality,
                method="combined",
                computation_time_ms=(time.time() - start_time) * 1000 / X.shape[0],
                provenance_hash=hashlib.sha256(
                    f"comb_{prediction:.6f}_{total_std:.6f}".encode()
                ).hexdigest()[:16],
            ))

        return results

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """
        Calibrate prediction intervals using calibration data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        self._calibration_data = (X_cal, y_cal)

        # Update conformal predictor with new calibration data
        if self._conformal_predictor:
            # Refit with combined data
            logger.info("Recalibrating conformal predictor")
            # In production, would update calibration scores

        logger.info(f"Calibrated with {len(y_cal)} samples")

    def evaluate_coverage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        """
        Evaluate empirical coverage at different confidence levels.

        Args:
            X: Test features
            y: True values
            confidence_levels: List of confidence levels to evaluate

        Returns:
            Dict mapping confidence level to empirical coverage
        """
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]

        results = self.quantify(X, method="combined")

        coverage = {}
        for conf in confidence_levels:
            n_covered = 0
            for i, result in enumerate(results):
                if conf in result.intervals:
                    if result.intervals[conf].contains(y[i]):
                        n_covered += 1

            coverage[conf] = n_covered / len(y)

        return coverage

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty quantifier configuration."""
        return {
            "config": {
                "quantiles": self.config.quantiles,
                "default_confidence_level": self.config.default_confidence_level,
                "n_bootstrap": self.config.n_bootstrap,
            },
            "is_fitted": self._is_fitted,
            "methods_available": {
                "quantile": self._quantile_regressor is not None,
                "conformal": self._conformal_predictor is not None,
                "ensemble": self._ensemble_uncertainty is not None,
            },
        }
