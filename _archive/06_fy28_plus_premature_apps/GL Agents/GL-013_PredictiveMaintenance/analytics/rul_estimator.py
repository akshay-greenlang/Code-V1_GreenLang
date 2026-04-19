# -*- coding: utf-8 -*-
"""
Remaining Useful Life (RUL) Estimator for GL-013 PredictiveMaintenance Agent.

Provides RUL estimation using multiple methods:
- Weibull-based survival analysis
- Gradient Boosting with uncertainty
- Ensemble methods for robust predictions

All predictions include uncertainty quantification.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .weibull_survival import WeibullSurvivalAnalyzer, WeibullDistribution

logger = logging.getLogger(__name__)


class RULModelType(str, Enum):
    """Types of RUL estimation models."""
    WEIBULL = "weibull"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


@dataclass
class RULEstimatorConfig:
    """Configuration for RUL estimation."""
    model_type: RULModelType = RULModelType.WEIBULL
    confidence_level: float = 0.95
    random_seed: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    n_bootstrap: int = 100
    prediction_horizon_hours: int = 8760  # 1 year


@dataclass
class RULResult:
    """Result of RUL estimation."""
    asset_id: str
    timestamp: datetime
    current_age_hours: float

    # RUL predictions (in hours)
    rul_median: float
    rul_mean: float
    rul_p10: float  # 10th percentile (optimistic)
    rul_p90: float  # 90th percentile (pessimistic)

    # Uncertainty
    rul_std: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float

    # Model information
    model_type: RULModelType
    model_version: str = "1.0.0"

    # Quality indicators
    prediction_quality: str = "high"  # high, medium, low
    data_quality_score: float = 1.0

    # Provenance
    provenance_hash: str = ""
    computation_time_ms: float = 0.0

    @property
    def rul_days(self) -> float:
        """RUL in days."""
        return self.rul_median / 24.0

    @property
    def is_critical(self) -> bool:
        """Check if RUL is critical (< 7 days)."""
        return self.rul_median < 168  # 7 days in hours


class RULEstimator(ABC):
    """Abstract base class for RUL estimators."""

    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        self.config = config or RULEstimatorConfig()
        self._is_fitted = False
        self._model_version = "1.0.0"

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        times_to_failure: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the RUL model."""
        pass

    @abstractmethod
    def predict(
        self,
        features: np.ndarray,
        current_age: float,
        asset_id: str = "",
    ) -> RULResult:
        """Predict RUL for given features."""
        pass

    def _compute_provenance_hash(self, *args) -> str:
        """Compute provenance hash."""
        content = "|".join(str(a) for a in args)
        return hashlib.sha256(content.encode()).hexdigest()


class WeibullRULEstimator(RULEstimator):
    """
    RUL estimator based on Weibull survival analysis.

    Uses historical failure data to fit Weibull distribution
    and predict RUL with uncertainty quantification.
    """

    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        super().__init__(config)
        self._survival_analyzer = WeibullSurvivalAnalyzer(
            confidence_level=self.config.confidence_level,
            random_seed=self.config.random_seed,
        )
        self._distribution: Optional[WeibullDistribution] = None

    def fit(
        self,
        features: np.ndarray,
        times_to_failure: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit Weibull model to failure data.

        Args:
            features: Feature matrix (not used directly in basic Weibull)
            times_to_failure: Time to failure or censoring (hours)
            events: 1 = failure observed, 0 = censored (default: all 1s)
        """
        if events is None:
            events = np.ones(len(times_to_failure), dtype=np.int32)

        self._survival_analyzer.fit(times_to_failure, events)
        self._distribution = self._survival_analyzer.get_distribution()
        self._is_fitted = True

        logger.info(
            f"WeibullRULEstimator fitted: shape={self._distribution.shape:.3f}, "
            f"scale={self._distribution.scale:.1f} hours"
        )

    def predict(
        self,
        features: np.ndarray,
        current_age: float,
        asset_id: str = "",
    ) -> RULResult:
        """
        Predict RUL given current age.

        Args:
            features: Current feature vector (for consistency, may not be used)
            current_age: Current operating hours
            asset_id: Asset identifier

        Returns:
            RULResult with predictions and uncertainty
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get RUL predictions from Weibull
        rul_preds = self._survival_analyzer.predict_rul(
            current_age=current_age,
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        )

        # Extract values
        rul_p10 = rul_preds.get("p10_rul_hours", 0.0)
        rul_p25 = rul_preds.get("p25_rul_hours", 0.0)
        rul_median = rul_preds.get("p50_rul_hours", 0.0)
        rul_p75 = rul_preds.get("p75_rul_hours", 0.0)
        rul_p90 = rul_preds.get("p90_rul_hours", 0.0)
        rul_mean = rul_preds.get("mean_rul_hours", 0.0)

        # Estimate standard deviation from IQR
        iqr = rul_p75 - rul_p25
        rul_std = iqr / 1.35  # IQR to std approximation for normal

        # Confidence interval based on p10 and p90
        ci_lower = rul_p10
        ci_upper = rul_p90

        # Prediction quality based on distribution properties
        if rul_median > 720:  # > 30 days
            quality = "high"
        elif rul_median > 168:  # > 7 days
            quality = "medium"
        else:
            quality = "low"

        computation_time = (time.time() - start_time) * 1000

        provenance_hash = self._compute_provenance_hash(
            asset_id, current_age, rul_median, datetime.utcnow().isoformat()
        )

        return RULResult(
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            current_age_hours=current_age,
            rul_median=rul_median,
            rul_mean=rul_mean,
            rul_p10=rul_p10,
            rul_p90=rul_p90,
            rul_std=rul_std,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            model_type=RULModelType.WEIBULL,
            model_version=self._model_version,
            prediction_quality=quality,
            provenance_hash=provenance_hash,
            computation_time_ms=computation_time,
        )


class GradientBoostingRULEstimator(RULEstimator):
    """
    RUL estimator using Gradient Boosting with quantile regression.

    Provides point estimates and uncertainty via quantile prediction.
    """

    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GradientBoostingRULEstimator")

        self._model_median: Optional[GradientBoostingRegressor] = None
        self._model_lower: Optional[GradientBoostingRegressor] = None
        self._model_upper: Optional[GradientBoostingRegressor] = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        features: np.ndarray,
        times_to_failure: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit Gradient Boosting models for quantile regression.

        Args:
            features: Feature matrix (n_samples, n_features)
            times_to_failure: Target RUL values (hours)
            events: Event indicators (not used, for compatibility)
        """
        # Normalize features
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0) + 1e-8
        X_normalized = (features - self._feature_means) / self._feature_stds

        # Fit median model
        self._model_median = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            loss="squared_error",
            random_state=self.config.random_seed,
        )
        self._model_median.fit(X_normalized, times_to_failure)

        # Fit quantile models for uncertainty
        alpha = (1 - self.config.confidence_level) / 2

        self._model_lower = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            loss="quantile",
            alpha=alpha,
            random_state=self.config.random_seed,
        )
        self._model_lower.fit(X_normalized, times_to_failure)

        self._model_upper = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            loss="quantile",
            alpha=1 - alpha,
            random_state=self.config.random_seed,
        )
        self._model_upper.fit(X_normalized, times_to_failure)

        self._is_fitted = True

        logger.info(
            f"GradientBoostingRULEstimator fitted with {len(times_to_failure)} samples"
        )

    def predict(
        self,
        features: np.ndarray,
        current_age: float,
        asset_id: str = "",
    ) -> RULResult:
        """
        Predict RUL from features.

        Args:
            features: Feature vector or matrix
            current_age: Current operating hours
            asset_id: Asset identifier

        Returns:
            RULResult with predictions and uncertainty
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Normalize
        X_normalized = (features - self._feature_means) / self._feature_stds

        # Predict
        rul_median = float(self._model_median.predict(X_normalized)[0])
        rul_lower = float(self._model_lower.predict(X_normalized)[0])
        rul_upper = float(self._model_upper.predict(X_normalized)[0])

        # Ensure non-negative
        rul_median = max(0, rul_median)
        rul_lower = max(0, rul_lower)
        rul_upper = max(0, rul_upper)

        # Estimate std from quantile range
        rul_std = (rul_upper - rul_lower) / (2 * 1.96)  # Approximate for 95% CI

        # Approximate p10 and p90
        rul_p10 = max(0, rul_median - 1.28 * rul_std)
        rul_p90 = rul_median + 1.28 * rul_std

        # Prediction quality
        if rul_std / max(rul_median, 1) < 0.2:
            quality = "high"
        elif rul_std / max(rul_median, 1) < 0.5:
            quality = "medium"
        else:
            quality = "low"

        computation_time = (time.time() - start_time) * 1000

        provenance_hash = self._compute_provenance_hash(
            asset_id, current_age, rul_median, datetime.utcnow().isoformat()
        )

        return RULResult(
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            current_age_hours=current_age,
            rul_median=rul_median,
            rul_mean=rul_median,  # GB doesn't give mean directly
            rul_p10=rul_p10,
            rul_p90=rul_p90,
            rul_std=rul_std,
            confidence_interval_lower=rul_lower,
            confidence_interval_upper=rul_upper,
            confidence_level=self.config.confidence_level,
            model_type=RULModelType.GRADIENT_BOOSTING,
            model_version=self._model_version,
            prediction_quality=quality,
            provenance_hash=provenance_hash,
            computation_time_ms=computation_time,
        )


class EnsembleRULEstimator(RULEstimator):
    """
    Ensemble RUL estimator combining multiple models.

    Combines predictions from Weibull and ML-based estimators
    for robust predictions with improved uncertainty quantification.
    """

    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        super().__init__(config)
        self._weibull_estimator = WeibullRULEstimator(config)
        self._gb_estimator: Optional[GradientBoostingRULEstimator] = None
        if SKLEARN_AVAILABLE:
            self._gb_estimator = GradientBoostingRULEstimator(config)
        self._weights = {"weibull": 0.5, "gb": 0.5}

    def fit(
        self,
        features: np.ndarray,
        times_to_failure: np.ndarray,
        events: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit ensemble of RUL models.

        Args:
            features: Feature matrix
            times_to_failure: Target RUL values
            events: Event indicators
        """
        # Fit Weibull
        self._weibull_estimator.fit(features, times_to_failure, events)

        # Fit GB if available
        if self._gb_estimator is not None:
            self._gb_estimator.fit(features, times_to_failure, events)

        self._is_fitted = True

        logger.info("EnsembleRULEstimator fitted")

    def predict(
        self,
        features: np.ndarray,
        current_age: float,
        asset_id: str = "",
    ) -> RULResult:
        """
        Predict RUL using ensemble of models.

        Args:
            features: Feature vector
            current_age: Current operating hours
            asset_id: Asset identifier

        Returns:
            RULResult with ensemble predictions
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get Weibull prediction
        weibull_result = self._weibull_estimator.predict(features, current_age, asset_id)

        # Get GB prediction if available
        if self._gb_estimator is not None:
            gb_result = self._gb_estimator.predict(features, current_age, asset_id)

            # Weighted average
            w_weibull = self._weights["weibull"]
            w_gb = self._weights["gb"]

            rul_median = w_weibull * weibull_result.rul_median + w_gb * gb_result.rul_median
            rul_mean = w_weibull * weibull_result.rul_mean + w_gb * gb_result.rul_mean
            rul_p10 = min(weibull_result.rul_p10, gb_result.rul_p10)
            rul_p90 = max(weibull_result.rul_p90, gb_result.rul_p90)

            # Combined uncertainty (conservative)
            rul_std = np.sqrt(
                w_weibull * weibull_result.rul_std ** 2 +
                w_gb * gb_result.rul_std ** 2 +
                w_weibull * w_gb * (weibull_result.rul_median - gb_result.rul_median) ** 2
            )
        else:
            # Weibull only
            rul_median = weibull_result.rul_median
            rul_mean = weibull_result.rul_mean
            rul_p10 = weibull_result.rul_p10
            rul_p90 = weibull_result.rul_p90
            rul_std = weibull_result.rul_std

        ci_lower = rul_p10
        ci_upper = rul_p90

        # Quality based on ensemble agreement
        quality = "high" if rul_std / max(rul_median, 1) < 0.3 else "medium"

        computation_time = (time.time() - start_time) * 1000

        provenance_hash = self._compute_provenance_hash(
            asset_id, current_age, rul_median, datetime.utcnow().isoformat()
        )

        return RULResult(
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            current_age_hours=current_age,
            rul_median=rul_median,
            rul_mean=rul_mean,
            rul_p10=rul_p10,
            rul_p90=rul_p90,
            rul_std=rul_std,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            model_type=RULModelType.ENSEMBLE,
            model_version=self._model_version,
            prediction_quality=quality,
            provenance_hash=provenance_hash,
            computation_time_ms=computation_time,
        )


def create_rul_estimator(
    model_type: RULModelType = RULModelType.WEIBULL,
    config: Optional[RULEstimatorConfig] = None,
) -> RULEstimator:
    """
    Factory function to create RUL estimator.

    Args:
        model_type: Type of RUL model
        config: Configuration

    Returns:
        RULEstimator instance
    """
    if config is None:
        config = RULEstimatorConfig(model_type=model_type)

    if model_type == RULModelType.WEIBULL:
        return WeibullRULEstimator(config)
    elif model_type == RULModelType.GRADIENT_BOOSTING:
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to Weibull")
            return WeibullRULEstimator(config)
        return GradientBoostingRULEstimator(config)
    elif model_type == RULModelType.ENSEMBLE:
        return EnsembleRULEstimator(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
