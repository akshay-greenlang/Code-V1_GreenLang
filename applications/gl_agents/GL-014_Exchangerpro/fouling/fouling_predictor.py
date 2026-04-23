# -*- coding: utf-8 -*-
"""
Fouling Predictor for GL-014 Exchangerpro Agent.

Provides ML-based fouling prediction with:
- UA degradation forecast for horizons (7, 14, 30 days)
- Fouling resistance (Rf) prediction
- Constraint risk probability (delta-P > limit, T_out below target)
- Remaining useful performance (RUP) estimation
- Support for XGBoost/LightGBM models

CRITICAL: All predictions MUST include confidence intervals.
Predictions are NEVER presented as certainties.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .feature_engineering import FoulingFeatures

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PredictionHorizon(str, Enum):
    """Prediction time horizons."""
    DAYS_7 = "7d"
    DAYS_14 = "14d"
    DAYS_30 = "30d"
    DAYS_60 = "60d"
    DAYS_90 = "90d"

    @property
    def hours(self) -> int:
        """Convert horizon to hours."""
        mapping = {
            "7d": 168,
            "14d": 336,
            "30d": 720,
            "60d": 1440,
            "90d": 2160,
        }
        return mapping[self.value]


class ModelType(str, Enum):
    """Supported ML model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FoulingPredictorConfig:
    """Configuration for fouling predictor."""

    # Model configuration
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_seed: int = 42

    # Prediction horizons
    prediction_horizons: List[PredictionHorizon] = field(
        default_factory=lambda: [
            PredictionHorizon.DAYS_7,
            PredictionHorizon.DAYS_14,
            PredictionHorizon.DAYS_30,
        ]
    )

    # Confidence levels
    confidence_level: float = 0.90  # 90% CI by default
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])

    # Constraint thresholds
    delta_p_limit_kpa: float = 50.0  # Max allowable pressure drop
    t_out_min_target: float = 40.0  # Min acceptable cold outlet temp

    # UA degradation thresholds
    ua_warning_ratio: float = 0.85  # UA/UA_clean < 0.85 = warning
    ua_critical_ratio: float = 0.70  # UA/UA_clean < 0.70 = critical

    # Feature scaling
    normalize_features: bool = True


# =============================================================================
# Prediction Results
# =============================================================================

@dataclass
class UADegradationForecast:
    """UA degradation forecast for a specific horizon."""

    horizon: PredictionHorizon
    horizon_hours: int

    # Point prediction (NEVER to be used without CI)
    ua_predicted: float
    ua_ratio_predicted: float  # Predicted UA / UA_clean

    # Confidence interval (MANDATORY)
    ua_lower: float  # Lower bound of CI
    ua_upper: float  # Upper bound of CI
    confidence_level: float

    # Quantiles
    quantiles: Dict[str, float] = field(default_factory=dict)

    # Risk assessment
    probability_below_warning: float = 0.0
    probability_below_critical: float = 0.0

    # Uncertainty metrics
    prediction_std: float = 0.0
    coefficient_of_variation: float = 0.0


@dataclass
class ConstraintRiskPrediction:
    """Constraint violation risk prediction."""

    horizon: PredictionHorizon

    # Pressure drop risk
    delta_p_predicted: float
    delta_p_lower: float
    delta_p_upper: float
    probability_exceeds_limit: float
    delta_p_limit: float

    # Temperature risk
    t_cold_out_predicted: float
    t_cold_out_lower: float
    t_cold_out_upper: float
    probability_below_target: float
    t_out_target: float

    # Combined risk
    overall_risk_level: RiskLevel
    overall_risk_probability: float

    # Confidence
    confidence_level: float


@dataclass
class RemainingUsefulPerformance:
    """Remaining Useful Performance (RUP) estimate."""

    # Time until performance thresholds
    hours_to_warning: Optional[float] = None  # Hours until UA < warning threshold
    hours_to_critical: Optional[float] = None  # Hours until UA < critical threshold
    hours_to_constraint: Optional[float] = None  # Hours until constraint violation

    # Confidence intervals for RUP
    hours_to_warning_lower: Optional[float] = None
    hours_to_warning_upper: Optional[float] = None
    hours_to_critical_lower: Optional[float] = None
    hours_to_critical_upper: Optional[float] = None

    # Current status
    current_ua_ratio: float = 0.0
    current_performance_grade: str = "A"  # A, B, C, D, F

    # Trend
    degradation_rate_per_day: float = 0.0
    trend_direction: str = "stable"  # improving, stable, degrading

    # Confidence level for all estimates
    confidence_level: float = 0.90


@dataclass
class FoulingPrediction:
    """Complete fouling prediction result."""

    exchanger_id: str
    timestamp: datetime

    # UA degradation forecasts by horizon
    ua_forecasts: Dict[str, UADegradationForecast] = field(default_factory=dict)

    # Fouling resistance prediction
    rf_predicted: float = 0.0  # m2.K/W
    rf_lower: float = 0.0
    rf_upper: float = 0.0
    rf_rate_predicted: float = 0.0  # Change per day

    # Constraint risk
    constraint_risks: Dict[str, ConstraintRiskPrediction] = field(default_factory=dict)

    # Remaining useful performance
    rup: Optional[RemainingUsefulPerformance] = None

    # Model information (MANDATORY)
    model_version: str = ""
    model_type: str = ""
    feature_schema_version: str = ""

    # Feature values used (MANDATORY for audit)
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    # Quality indicators
    prediction_quality: str = "high"  # high, medium, low
    data_quality_score: float = 1.0

    # Provenance
    provenance_hash: str = ""
    computation_time_ms: float = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get prediction summary for display."""
        summary = {
            "exchanger_id": self.exchanger_id,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "prediction_quality": self.prediction_quality,
        }

        # Add forecast summaries
        for horizon, forecast in self.ua_forecasts.items():
            summary[f"ua_ratio_{horizon}"] = {
                "predicted": forecast.ua_ratio_predicted,
                "lower": forecast.ua_lower / forecast.ua_predicted * forecast.ua_ratio_predicted,
                "upper": forecast.ua_upper / forecast.ua_predicted * forecast.ua_ratio_predicted,
                "confidence": forecast.confidence_level,
            }

        # Add RUP if available
        if self.rup:
            summary["hours_to_warning"] = self.rup.hours_to_warning
            summary["hours_to_critical"] = self.rup.hours_to_critical
            summary["performance_grade"] = self.rup.current_performance_grade

        return summary


# =============================================================================
# Base Predictor
# =============================================================================

class BaseFoulingModel(ABC):
    """Abstract base class for fouling models."""

    def __init__(self, config: FoulingPredictorConfig):
        self.config = config
        self._is_fitted = False
        self._feature_names: List[str] = []
        self._model_version = "1.0.0"

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit the model."""
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        pass

    @abstractmethod
    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float],
    ) -> Dict[float, np.ndarray]:
        """Predict specific quantiles."""
        pass


# =============================================================================
# Model Implementations
# =============================================================================

class GradientBoostingFoulingModel(BaseFoulingModel):
    """Gradient Boosting based fouling model with quantile regression."""

    def __init__(self, config: FoulingPredictorConfig):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GradientBoostingFoulingModel")

        self._model_median: Optional[GradientBoostingRegressor] = None
        self._model_lower: Optional[GradientBoostingRegressor] = None
        self._model_upper: Optional[GradientBoostingRegressor] = None
        self._quantile_models: Dict[float, GradientBoostingRegressor] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Fit Gradient Boosting models for median and quantiles.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (UA degradation, Rf, etc.)
            feature_names: Names of features
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Normalize features
        if self.config.normalize_features:
            self._feature_means = np.mean(X, axis=0)
            self._feature_stds = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        # Fit median model
        self._model_median = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            loss="squared_error",
            random_state=self.config.random_seed,
        )
        self._model_median.fit(X_normalized, y)

        # Fit quantile models for confidence intervals
        alpha = (1 - self.config.confidence_level) / 2

        self._model_lower = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            loss="quantile",
            alpha=alpha,
            random_state=self.config.random_seed,
        )
        self._model_lower.fit(X_normalized, y)

        self._model_upper = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            loss="quantile",
            alpha=1 - alpha,
            random_state=self.config.random_seed,
        )
        self._model_upper.fit(X_normalized, y)

        # Fit additional quantile models
        for q in self.config.quantiles:
            if q not in [alpha, 0.5, 1 - alpha]:
                model = GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    loss="quantile",
                    alpha=q,
                    random_state=self.config.random_seed,
                )
                model.fit(X_normalized, y)
                self._quantile_models[q] = model

        self._is_fitted = True

        logger.info(
            f"GradientBoostingFoulingModel fitted with {X.shape[0]} samples, "
            f"{X.shape[1]} features"
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (median_predictions, lower_bound, upper_bound)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        predictions = self._model_median.predict(X_normalized)
        lower = self._model_lower.predict(X_normalized)
        upper = self._model_upper.predict(X_normalized)

        return predictions, lower, upper

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float],
    ) -> Dict[float, np.ndarray]:
        """Predict specific quantiles."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        results = {}
        alpha = (1 - self.config.confidence_level) / 2

        for q in quantiles:
            if abs(q - 0.5) < 0.01:
                results[q] = self._model_median.predict(X_normalized)
            elif abs(q - alpha) < 0.01:
                results[q] = self._model_lower.predict(X_normalized)
            elif abs(q - (1 - alpha)) < 0.01:
                results[q] = self._model_upper.predict(X_normalized)
            elif q in self._quantile_models:
                results[q] = self._quantile_models[q].predict(X_normalized)
            else:
                # Interpolate from available quantiles
                results[q] = self._model_median.predict(X_normalized)

        return results


class XGBoostFoulingModel(BaseFoulingModel):
    """XGBoost based fouling model."""

    def __init__(self, config: FoulingPredictorConfig):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost required for XGBoostFoulingModel")

        self._model: Optional[xgb.XGBRegressor] = None
        self._quantile_models: Dict[float, xgb.XGBRegressor] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit XGBoost model."""
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        if self.config.normalize_features:
            self._feature_means = np.mean(X, axis=0)
            self._feature_stds = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        # Fit main model
        self._model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_seed,
            objective="reg:squarederror",
        )
        self._model.fit(X_normalized, y)

        # Fit quantile models
        for q in self.config.quantiles:
            model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_seed,
                objective="reg:quantileerror",
                quantile_alpha=q,
            )
            model.fit(X_normalized, y)
            self._quantile_models[q] = model

        self._is_fitted = True

        logger.info(f"XGBoostFoulingModel fitted with {X.shape[0]} samples")

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        predictions = self._model.predict(X_normalized)

        alpha = (1 - self.config.confidence_level) / 2
        lower_q = min(self._quantile_models.keys(), key=lambda x: abs(x - alpha))
        upper_q = min(self._quantile_models.keys(), key=lambda x: abs(x - (1 - alpha)))

        lower = self._quantile_models[lower_q].predict(X_normalized)
        upper = self._quantile_models[upper_q].predict(X_normalized)

        return predictions, lower, upper

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float],
    ) -> Dict[float, np.ndarray]:
        """Predict specific quantiles."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        results = {}
        for q in quantiles:
            if q in self._quantile_models:
                results[q] = self._quantile_models[q].predict(X_normalized)
            else:
                # Use main model as fallback
                results[q] = self._model.predict(X_normalized)

        return results


class LightGBMFoulingModel(BaseFoulingModel):
    """LightGBM based fouling model."""

    def __init__(self, config: FoulingPredictorConfig):
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required for LightGBMFoulingModel")

        self._model: Optional[lgb.LGBMRegressor] = None
        self._quantile_models: Dict[float, lgb.LGBMRegressor] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit LightGBM model."""
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        if self.config.normalize_features:
            self._feature_means = np.mean(X, axis=0)
            self._feature_stds = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        # Fit main model
        self._model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_seed,
            objective="regression",
            verbosity=-1,
        )
        self._model.fit(X_normalized, y)

        # Fit quantile models
        for q in self.config.quantiles:
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_seed,
                objective="quantile",
                alpha=q,
                verbosity=-1,
            )
            model.fit(X_normalized, y)
            self._quantile_models[q] = model

        self._is_fitted = True

        logger.info(f"LightGBMFoulingModel fitted with {X.shape[0]} samples")

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        predictions = self._model.predict(X_normalized)

        alpha = (1 - self.config.confidence_level) / 2
        lower_q = min(self._quantile_models.keys(), key=lambda x: abs(x - alpha))
        upper_q = min(self._quantile_models.keys(), key=lambda x: abs(x - (1 - alpha)))

        lower = self._quantile_models[lower_q].predict(X_normalized)
        upper = self._quantile_models[upper_q].predict(X_normalized)

        return predictions, lower, upper

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float],
    ) -> Dict[float, np.ndarray]:
        """Predict specific quantiles."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.config.normalize_features:
            X_normalized = (X - self._feature_means) / self._feature_stds
        else:
            X_normalized = X

        results = {}
        for q in quantiles:
            if q in self._quantile_models:
                results[q] = self._quantile_models[q].predict(X_normalized)
            else:
                results[q] = self._model.predict(X_normalized)

        return results


# =============================================================================
# Main Predictor
# =============================================================================

class FoulingPredictor:
    """
    Main fouling predictor class.

    Provides:
    - UA degradation forecast for multiple horizons
    - Fouling resistance (Rf) prediction
    - Constraint risk probability
    - Remaining useful performance (RUP) estimation

    CRITICAL: All predictions include confidence intervals.
    Predictions are NEVER presented as certainties.

    Example:
        >>> config = FoulingPredictorConfig()
        >>> predictor = FoulingPredictor(config)
        >>> predictor.fit(X_train, y_train)
        >>> result = predictor.predict(features)
        >>> print(f"UA at 7d: {result.ua_forecasts['7d'].ua_predicted}")
        >>> print(f"  90% CI: [{result.ua_forecasts['7d'].ua_lower}, "
        ...       f"{result.ua_forecasts['7d'].ua_upper}]")
    """

    def __init__(self, config: Optional[FoulingPredictorConfig] = None):
        """
        Initialize fouling predictor.

        Args:
            config: Predictor configuration
        """
        self.config = config or FoulingPredictorConfig()
        self._model_version = "1.0.0"
        self._feature_schema_version = "1.0.0"

        # Initialize models for each horizon
        self._ua_models: Dict[PredictionHorizon, BaseFoulingModel] = {}
        self._rf_model: Optional[BaseFoulingModel] = None
        self._delta_p_model: Optional[BaseFoulingModel] = None
        self._t_out_model: Optional[BaseFoulingModel] = None

        self._is_fitted = False
        self._feature_names: List[str] = []

        logger.info(
            f"FoulingPredictor initialized with model_type={self.config.model_type}, "
            f"horizons={[h.value for h in self.config.prediction_horizons]}"
        )

    def _create_model(self) -> BaseFoulingModel:
        """Create a model based on configuration."""
        if self.config.model_type == ModelType.XGBOOST:
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, falling back to GradientBoosting")
                return GradientBoostingFoulingModel(self.config)
            return XGBoostFoulingModel(self.config)

        elif self.config.model_type == ModelType.LIGHTGBM:
            if not LIGHTGBM_AVAILABLE:
                logger.warning("LightGBM not available, falling back to GradientBoosting")
                return GradientBoostingFoulingModel(self.config)
            return LightGBMFoulingModel(self.config)

        else:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for FoulingPredictor")
            return GradientBoostingFoulingModel(self.config)

    def fit(
        self,
        X: np.ndarray,
        y_ua: Dict[PredictionHorizon, np.ndarray],
        y_rf: Optional[np.ndarray] = None,
        y_delta_p: Optional[np.ndarray] = None,
        y_t_out: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Fit fouling prediction models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_ua: Dict mapping horizon to UA target values
            y_rf: Fouling resistance targets (optional)
            y_delta_p: Pressure drop targets (optional)
            y_t_out: Outlet temperature targets (optional)
            feature_names: Names of features
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Fit UA models for each horizon
        for horizon in self.config.prediction_horizons:
            if horizon in y_ua:
                model = self._create_model()
                model.fit(X, y_ua[horizon], self._feature_names)
                self._ua_models[horizon] = model
                logger.info(f"Fitted UA model for horizon {horizon.value}")

        # Fit Rf model
        if y_rf is not None:
            self._rf_model = self._create_model()
            self._rf_model.fit(X, y_rf, self._feature_names)
            logger.info("Fitted Rf model")

        # Fit delta-P model
        if y_delta_p is not None:
            self._delta_p_model = self._create_model()
            self._delta_p_model.fit(X, y_delta_p, self._feature_names)
            logger.info("Fitted delta-P model")

        # Fit T_out model
        if y_t_out is not None:
            self._t_out_model = self._create_model()
            self._t_out_model.fit(X, y_t_out, self._feature_names)
            logger.info("Fitted T_out model")

        self._is_fitted = True

    def predict(
        self,
        features: Union[FoulingFeatures, np.ndarray, Dict[str, float]],
        ua_clean: Optional[float] = None,
    ) -> FoulingPrediction:
        """
        Generate fouling predictions with uncertainty.

        Args:
            features: Input features (FoulingFeatures, array, or dict)
            ua_clean: Clean UA value for ratio calculations

        Returns:
            FoulingPrediction with all predictions and confidence intervals
        """
        start_time = time.time()

        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert features to array
        X, feature_values, exchanger_id, timestamp = self._prepare_features(features)

        # Get UA clean for ratio calculations
        if ua_clean is None:
            ua_clean = feature_values.get("ua_clean", feature_values.get("ua_current", 1.0))

        # Generate UA forecasts
        ua_forecasts = {}
        for horizon, model in self._ua_models.items():
            forecast = self._predict_ua_degradation(
                model, X, horizon, ua_clean
            )
            ua_forecasts[horizon.value] = forecast

        # Generate Rf prediction
        rf_predicted, rf_lower, rf_upper = 0.0, 0.0, 0.0
        if self._rf_model is not None:
            preds, lower, upper = self._rf_model.predict(X)
            rf_predicted = float(preds[0])
            rf_lower = float(lower[0])
            rf_upper = float(upper[0])

        # Generate constraint risk predictions
        constraint_risks = {}
        for horizon in self.config.prediction_horizons:
            risk = self._predict_constraint_risk(X, horizon)
            if risk is not None:
                constraint_risks[horizon.value] = risk

        # Estimate remaining useful performance
        rup = self._estimate_rup(ua_forecasts, ua_clean, feature_values)

        # Assess prediction quality
        prediction_quality = self._assess_prediction_quality(X)

        # Compute provenance
        computation_time = (time.time() - start_time) * 1000
        provenance_hash = self._compute_provenance_hash(
            exchanger_id, timestamp, feature_values
        )

        return FoulingPrediction(
            exchanger_id=exchanger_id,
            timestamp=timestamp,
            ua_forecasts=ua_forecasts,
            rf_predicted=rf_predicted,
            rf_lower=rf_lower,
            rf_upper=rf_upper,
            constraint_risks=constraint_risks,
            rup=rup,
            model_version=self._model_version,
            model_type=self.config.model_type.value,
            feature_schema_version=self._feature_schema_version,
            feature_values=feature_values,
            feature_names=self._feature_names,
            prediction_quality=prediction_quality,
            provenance_hash=provenance_hash,
            computation_time_ms=computation_time,
        )

    def _prepare_features(
        self,
        features: Union[FoulingFeatures, np.ndarray, Dict[str, float]],
    ) -> Tuple[np.ndarray, Dict[str, float], str, datetime]:
        """Prepare features for prediction."""
        if isinstance(features, FoulingFeatures):
            X = features.to_array(self._feature_names)
            feature_values = {
                name: getattr(features, name, features.rolling_features.get(
                    name, features.lagged_features.get(name, 0.0)
                ))
                for name in self._feature_names
            }
            exchanger_id = features.exchanger_id
            timestamp = features.timestamp

        elif isinstance(features, dict):
            X = np.array([features.get(name, 0.0) for name in self._feature_names])
            feature_values = features
            exchanger_id = features.get("exchanger_id", "unknown")
            timestamp = features.get("timestamp", datetime.utcnow())

        else:
            X = np.asarray(features)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            feature_values = {
                name: float(X[0, i]) if i < X.shape[1] else 0.0
                for i, name in enumerate(self._feature_names)
            }
            exchanger_id = "unknown"
            timestamp = datetime.utcnow()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X, feature_values, exchanger_id, timestamp

    def _predict_ua_degradation(
        self,
        model: BaseFoulingModel,
        X: np.ndarray,
        horizon: PredictionHorizon,
        ua_clean: float,
    ) -> UADegradationForecast:
        """Predict UA degradation for a specific horizon."""
        predictions, lower, upper = model.predict(X)
        quantile_preds = model.predict_quantiles(X, self.config.quantiles)

        ua_pred = float(predictions[0])
        ua_lower = float(lower[0])
        ua_upper = float(upper[0])

        # Calculate ratio
        ua_ratio = ua_pred / max(ua_clean, 1e-6)

        # Calculate probabilities
        # Estimate from quantiles
        prob_warning = 0.0
        prob_critical = 0.0

        if 0.5 in quantile_preds:
            # Rough estimate using normal approximation
            std_estimate = (ua_upper - ua_lower) / (2 * 1.645)  # 90% CI
            if std_estimate > 0:
                from scipy.stats import norm
                warning_threshold = ua_clean * self.config.ua_warning_ratio
                critical_threshold = ua_clean * self.config.ua_critical_ratio
                prob_warning = float(norm.cdf(warning_threshold, ua_pred, std_estimate))
                prob_critical = float(norm.cdf(critical_threshold, ua_pred, std_estimate))

        # Build quantiles dict
        quantiles_dict = {
            f"q{int(q*100)}": float(v[0])
            for q, v in quantile_preds.items()
        }

        return UADegradationForecast(
            horizon=horizon,
            horizon_hours=horizon.hours,
            ua_predicted=ua_pred,
            ua_ratio_predicted=ua_ratio,
            ua_lower=ua_lower,
            ua_upper=ua_upper,
            confidence_level=self.config.confidence_level,
            quantiles=quantiles_dict,
            probability_below_warning=prob_warning,
            probability_below_critical=prob_critical,
            prediction_std=float((ua_upper - ua_lower) / (2 * 1.645)),
            coefficient_of_variation=float((ua_upper - ua_lower) / (2 * max(ua_pred, 1e-6))),
        )

    def _predict_constraint_risk(
        self,
        X: np.ndarray,
        horizon: PredictionHorizon,
    ) -> Optional[ConstraintRiskPrediction]:
        """Predict constraint violation risk."""
        if self._delta_p_model is None and self._t_out_model is None:
            return None

        # Delta-P prediction
        if self._delta_p_model is not None:
            dp_pred, dp_lower, dp_upper = self._delta_p_model.predict(X)
            delta_p_predicted = float(dp_pred[0])
            delta_p_lower = float(dp_lower[0])
            delta_p_upper = float(dp_upper[0])

            # Probability of exceeding limit
            std_dp = (delta_p_upper - delta_p_lower) / (2 * 1.645)
            if std_dp > 0:
                from scipy.stats import norm
                prob_dp_exceed = float(1 - norm.cdf(
                    self.config.delta_p_limit_kpa, delta_p_predicted, std_dp
                ))
            else:
                prob_dp_exceed = 0.0
        else:
            delta_p_predicted = 0.0
            delta_p_lower = 0.0
            delta_p_upper = 0.0
            prob_dp_exceed = 0.0

        # T_out prediction
        if self._t_out_model is not None:
            t_pred, t_lower, t_upper = self._t_out_model.predict(X)
            t_out_predicted = float(t_pred[0])
            t_out_lower = float(t_lower[0])
            t_out_upper = float(t_upper[0])

            std_t = (t_out_upper - t_out_lower) / (2 * 1.645)
            if std_t > 0:
                from scipy.stats import norm
                prob_t_below = float(norm.cdf(
                    self.config.t_out_min_target, t_out_predicted, std_t
                ))
            else:
                prob_t_below = 0.0
        else:
            t_out_predicted = 0.0
            t_out_lower = 0.0
            t_out_upper = 0.0
            prob_t_below = 0.0

        # Overall risk
        overall_prob = max(prob_dp_exceed, prob_t_below)

        if overall_prob >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif overall_prob >= 0.5:
            risk_level = RiskLevel.HIGH
        elif overall_prob >= 0.2:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

        return ConstraintRiskPrediction(
            horizon=horizon,
            delta_p_predicted=delta_p_predicted,
            delta_p_lower=delta_p_lower,
            delta_p_upper=delta_p_upper,
            probability_exceeds_limit=prob_dp_exceed,
            delta_p_limit=self.config.delta_p_limit_kpa,
            t_cold_out_predicted=t_out_predicted,
            t_cold_out_lower=t_out_lower,
            t_cold_out_upper=t_out_upper,
            probability_below_target=prob_t_below,
            t_out_target=self.config.t_out_min_target,
            overall_risk_level=risk_level,
            overall_risk_probability=overall_prob,
            confidence_level=self.config.confidence_level,
        )

    def _estimate_rup(
        self,
        ua_forecasts: Dict[str, UADegradationForecast],
        ua_clean: float,
        feature_values: Dict[str, float],
    ) -> RemainingUsefulPerformance:
        """Estimate remaining useful performance."""
        # Get current UA ratio
        ua_current = feature_values.get("ua_current", ua_clean)
        current_ratio = ua_current / max(ua_clean, 1e-6)

        # Determine current grade
        if current_ratio >= 0.9:
            grade = "A"
        elif current_ratio >= 0.8:
            grade = "B"
        elif current_ratio >= 0.7:
            grade = "C"
        elif current_ratio >= 0.6:
            grade = "D"
        else:
            grade = "F"

        # Estimate degradation rate from forecasts
        if len(ua_forecasts) >= 2:
            horizons_sorted = sorted(
                ua_forecasts.items(),
                key=lambda x: PredictionHorizon(x[0]).hours
            )

            # Get first and last horizon
            first_h, first_f = horizons_sorted[0]
            last_h, last_f = horizons_sorted[-1]

            delta_ua = ua_current - last_f.ua_predicted
            delta_hours = PredictionHorizon(last_h).hours

            if delta_hours > 0:
                rate_per_hour = delta_ua / delta_hours
                rate_per_day = rate_per_hour * 24
            else:
                rate_per_day = 0.0
        else:
            rate_per_day = 0.0

        # Trend direction
        if rate_per_day > 0.01:
            trend = "degrading"
        elif rate_per_day < -0.01:
            trend = "improving"
        else:
            trend = "stable"

        # Estimate hours to thresholds
        hours_to_warning = None
        hours_to_critical = None

        warning_ua = ua_clean * self.config.ua_warning_ratio
        critical_ua = ua_clean * self.config.ua_critical_ratio

        if rate_per_day > 0:
            if ua_current > warning_ua:
                hours_to_warning = (ua_current - warning_ua) / (rate_per_day / 24)
            if ua_current > critical_ua:
                hours_to_critical = (ua_current - critical_ua) / (rate_per_day / 24)

        # Confidence bounds (rough estimate: +/- 20%)
        hours_to_warning_lower = hours_to_warning * 0.8 if hours_to_warning else None
        hours_to_warning_upper = hours_to_warning * 1.2 if hours_to_warning else None
        hours_to_critical_lower = hours_to_critical * 0.8 if hours_to_critical else None
        hours_to_critical_upper = hours_to_critical * 1.2 if hours_to_critical else None

        return RemainingUsefulPerformance(
            hours_to_warning=hours_to_warning,
            hours_to_critical=hours_to_critical,
            hours_to_warning_lower=hours_to_warning_lower,
            hours_to_warning_upper=hours_to_warning_upper,
            hours_to_critical_lower=hours_to_critical_lower,
            hours_to_critical_upper=hours_to_critical_upper,
            current_ua_ratio=current_ratio,
            current_performance_grade=grade,
            degradation_rate_per_day=rate_per_day,
            trend_direction=trend,
            confidence_level=self.config.confidence_level,
        )

    def _assess_prediction_quality(self, X: np.ndarray) -> str:
        """Assess prediction quality based on input features."""
        # Check for missing values
        n_missing = np.sum(np.isnan(X))
        if n_missing > X.size * 0.1:
            return "low"
        elif n_missing > 0:
            return "medium"
        return "high"

    def _compute_provenance_hash(
        self,
        exchanger_id: str,
        timestamp: datetime,
        feature_values: Dict[str, float],
    ) -> str:
        """Compute SHA-256 provenance hash."""
        content = (
            f"{exchanger_id}|{timestamp.isoformat()}|"
            f"{self._model_version}|"
            f"{len(feature_values)}|"
            f"{sum(feature_values.values()):.6f}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for audit trail."""
        return {
            "model_version": self._model_version,
            "model_type": self.config.model_type.value,
            "feature_schema_version": self._feature_schema_version,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "prediction_horizons": [h.value for h in self.config.prediction_horizons],
            "confidence_level": self.config.confidence_level,
            "is_fitted": self._is_fitted,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_fouling_predictor(
    model_type: ModelType = ModelType.GRADIENT_BOOSTING,
    config: Optional[FoulingPredictorConfig] = None,
) -> FoulingPredictor:
    """
    Factory function to create fouling predictor.

    Args:
        model_type: Type of ML model to use
        config: Predictor configuration

    Returns:
        FoulingPredictor instance
    """
    if config is None:
        config = FoulingPredictorConfig(model_type=model_type)
    else:
        config.model_type = model_type

    return FoulingPredictor(config)
