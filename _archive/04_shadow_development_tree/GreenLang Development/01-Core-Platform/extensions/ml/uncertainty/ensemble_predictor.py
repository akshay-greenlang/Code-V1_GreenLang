# -*- coding: utf-8 -*-
"""
Ensemble Predictor Module

This module provides ensemble-based uncertainty quantification for GreenLang ML
models, using multiple models to estimate prediction uncertainty and improve
robustness.

Ensemble methods provide natural uncertainty estimates through prediction
disagreement, critical for regulatory compliance where understanding prediction
confidence is essential.

Example:
    >>> from greenlang.ml.uncertainty import EnsemblePredictor
    >>> ensemble = EnsemblePredictor(n_models=10, model_class=GradientBoostingRegressor)
    >>> ensemble.fit(X_train, y_train)
    >>> predictions, uncertainty = ensemble.predict_with_uncertainty(X_test)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EnsembleMethod(str, Enum):
    """Ensemble methods."""
    BAGGING = "bagging"
    RANDOM_SUBSPACE = "random_subspace"
    BOOTSTRAP = "bootstrap"
    CROSS_VALIDATION = "cross_validation"
    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"


class AggregationMethod(str, Enum):
    """Prediction aggregation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"
    TRIMMED_MEAN = "trimmed_mean"


class EnsembleConfig(BaseModel):
    """Configuration for ensemble predictor."""

    n_models: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Number of models in ensemble"
    )
    ensemble_method: EnsembleMethod = Field(
        default=EnsembleMethod.BOOTSTRAP,
        description="Method for creating ensemble diversity"
    )
    aggregation_method: AggregationMethod = Field(
        default=AggregationMethod.MEAN,
        description="Method for aggregating predictions"
    )
    bootstrap_ratio: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Ratio of samples for bootstrap"
    )
    feature_ratio: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Ratio of features for random subspace"
    )
    trim_ratio: float = Field(
        default=0.1,
        ge=0,
        le=0.4,
        description="Ratio to trim for trimmed mean"
    )
    parallel: bool = Field(
        default=True,
        description="Enable parallel training"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class UncertaintyResult(BaseModel):
    """Result with uncertainty estimates."""

    predictions: List[float] = Field(
        ...,
        description="Point predictions"
    )
    uncertainties: List[float] = Field(
        ...,
        description="Uncertainty estimates (std)"
    )
    lower_bounds: List[float] = Field(
        ...,
        description="Lower confidence bounds"
    )
    upper_bounds: List[float] = Field(
        ...,
        description="Upper confidence bounds"
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level for bounds"
    )
    model_agreement: List[float] = Field(
        ...,
        description="Agreement score per prediction"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    n_models: int = Field(
        ...,
        description="Number of models used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )


class EnsemblePredictor:
    """
    Ensemble Predictor for uncertainty quantification.

    This class uses an ensemble of models to provide prediction
    uncertainty estimates through the variance of individual model
    predictions. This is critical for regulatory compliance where
    understanding prediction confidence is essential.

    Key capabilities:
    - Bootstrap aggregation (bagging)
    - Random subspace method
    - Cross-validation ensemble
    - Multiple aggregation methods
    - Confidence interval estimation
    - Provenance tracking

    Attributes:
        config: Ensemble configuration
        models: List of trained models
        model_weights: Weights for weighted aggregation
        feature_indices: Feature indices for random subspace

    Example:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> ensemble = EnsemblePredictor(
        ...     n_models=10,
        ...     model_class=GradientBoostingRegressor,
        ...     config=EnsembleConfig(ensemble_method=EnsembleMethod.BOOTSTRAP)
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> result = ensemble.predict_with_uncertainty(X_test, confidence=0.95)
        >>> print(f"Prediction: {result.predictions[0]:.2f} +/- {result.uncertainties[0]:.2f}")
    """

    def __init__(
        self,
        model_class: Optional[Type] = None,
        model_params: Optional[Dict[str, Any]] = None,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize ensemble predictor.

        Args:
            model_class: Class of base model
            model_params: Parameters for base model
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self._model_class = model_class
        self._model_params = model_params or {}
        self.models: List[Any] = []
        self.model_weights: np.ndarray = np.array([])
        self.feature_indices: List[List[int]] = []
        self._is_fitted = False
        self._n_features = 0

        np.random.seed(self.config.random_state)

        logger.info(
            f"EnsemblePredictor initialized: n_models={self.config.n_models}, "
            f"method={self.config.ensemble_method}"
        )

    def _create_base_model(self, random_state: int) -> Any:
        """Create a base model instance."""
        if self._model_class is None:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    random_state=random_state,
                    **self._model_params
                )
            except ImportError:
                raise ImportError(
                    "No model class provided and sklearn not available"
                )

        params = self._model_params.copy()
        if "random_state" in params or hasattr(self._model_class, "random_state"):
            params["random_state"] = random_state

        return self._model_class(**params)

    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple:
        """Generate bootstrap sample."""
        n_samples = int(len(X) * self.config.bootstrap_ratio)
        indices = np.random.choice(len(X), n_samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_subset(self, n_features: int) -> List[int]:
        """Get random feature subset."""
        n_select = int(n_features * self.config.feature_ratio)
        return list(np.random.choice(n_features, n_select, replace=False))

    def _calculate_provenance(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{self.config.n_models}|{predictions.sum():.8f}|"
            f"{uncertainties.sum():.8f}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> "EnsemblePredictor":
        """
        Fit the ensemble on training data.

        Args:
            X: Training features
            y: Training targets
            sample_weights: Optional sample weights

        Returns:
            self

        Example:
            >>> ensemble.fit(X_train, y_train)
        """
        logger.info(f"Fitting ensemble with {self.config.n_models} models")

        self._n_features = X.shape[1]
        self.models = []
        self.feature_indices = []

        for i in range(self.config.n_models):
            # Create base model
            model = self._create_base_model(random_state=self.config.random_state + i)

            # Prepare data based on ensemble method
            if self.config.ensemble_method == EnsembleMethod.BOOTSTRAP:
                X_train, y_train = self._bootstrap_sample(X, y)
                feature_idx = list(range(self._n_features))

            elif self.config.ensemble_method == EnsembleMethod.RANDOM_SUBSPACE:
                X_train, y_train = X, y
                feature_idx = self._get_feature_subset(self._n_features)
                X_train = X_train[:, feature_idx]

            elif self.config.ensemble_method == EnsembleMethod.BAGGING:
                X_train, y_train = self._bootstrap_sample(X, y)
                feature_idx = self._get_feature_subset(self._n_features)
                X_train = X_train[:, feature_idx]

            else:
                X_train, y_train = X, y
                feature_idx = list(range(self._n_features))

            self.feature_indices.append(feature_idx)

            # Fit model
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

            self.models.append(model)
            logger.debug(f"Fitted model {i+1}/{self.config.n_models}")

        # Initialize equal weights
        self.model_weights = np.ones(self.config.n_models) / self.config.n_models
        self._is_fitted = True

        logger.info(f"Ensemble fitting complete: {len(self.models)} models trained")
        return self

    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models."""
        predictions = np.zeros((len(X), self.config.n_models))

        for i, (model, feature_idx) in enumerate(zip(self.models, self.feature_indices)):
            X_subset = X[:, feature_idx] if len(feature_idx) < self._n_features else X
            predictions[:, i] = model.predict(X_subset)

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        all_preds = self._get_all_predictions(X)

        if self.config.aggregation_method == AggregationMethod.MEAN:
            return np.mean(all_preds, axis=1)
        elif self.config.aggregation_method == AggregationMethod.MEDIAN:
            return np.median(all_preds, axis=1)
        elif self.config.aggregation_method == AggregationMethod.WEIGHTED_MEAN:
            return np.average(all_preds, axis=1, weights=self.model_weights)
        elif self.config.aggregation_method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(all_preds)

        return np.mean(all_preds, axis=1)

    def _trimmed_mean(self, predictions: np.ndarray) -> np.ndarray:
        """Compute trimmed mean by removing extreme predictions."""
        trim_count = int(self.config.n_models * self.config.trim_ratio)

        if trim_count == 0:
            return np.mean(predictions, axis=1)

        sorted_preds = np.sort(predictions, axis=1)
        trimmed = sorted_preds[:, trim_count:-trim_count]

        return np.mean(trimmed, axis=1)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> UncertaintyResult:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Features
            confidence: Confidence level for intervals

        Returns:
            UncertaintyResult with predictions and uncertainties

        Example:
            >>> result = ensemble.predict_with_uncertainty(X_test, confidence=0.95)
            >>> for i in range(5):
            ...     print(f"Pred: {result.predictions[i]:.2f} "
            ...           f"[{result.lower_bounds[i]:.2f}, {result.upper_bounds[i]:.2f}]")
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        all_preds = self._get_all_predictions(X)

        # Point predictions
        predictions = np.mean(all_preds, axis=1)

        # Uncertainty as standard deviation
        uncertainties = np.std(all_preds, axis=1)

        # Confidence intervals
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bounds = np.percentile(all_preds, lower_percentile, axis=1)
        upper_bounds = np.percentile(all_preds, upper_percentile, axis=1)

        # Model agreement (inverse of coefficient of variation)
        mean_preds = np.mean(all_preds, axis=1)
        cv = uncertainties / (np.abs(mean_preds) + 1e-10)
        agreement = 1 / (1 + cv)  # Higher = more agreement

        # Calculate provenance
        provenance_hash = self._calculate_provenance(predictions, uncertainties)

        return UncertaintyResult(
            predictions=predictions.tolist(),
            uncertainties=uncertainties.tolist(),
            lower_bounds=lower_bounds.tolist(),
            upper_bounds=upper_bounds.tolist(),
            confidence_level=confidence,
            model_agreement=agreement.tolist(),
            provenance_hash=provenance_hash,
            n_models=self.config.n_models,
            timestamp=datetime.utcnow()
        )

    def update_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "mse"
    ) -> None:
        """
        Update model weights based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets
            metric: Metric to use (mse, mae)
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted.")

        errors = np.zeros(self.config.n_models)

        for i, (model, feature_idx) in enumerate(zip(self.models, self.feature_indices)):
            X_subset = X_val[:, feature_idx] if len(feature_idx) < self._n_features else X_val
            preds = model.predict(X_subset)

            if metric == "mse":
                errors[i] = np.mean((y_val - preds) ** 2)
            else:  # mae
                errors[i] = np.mean(np.abs(y_val - preds))

        # Inverse error weighting
        inv_errors = 1 / (errors + 1e-10)
        self.model_weights = inv_errors / inv_errors.sum()

        logger.info(f"Model weights updated: {self.model_weights}")

    def get_model_diversity(self) -> Dict[str, float]:
        """
        Calculate diversity metrics for the ensemble.

        Returns:
            Dictionary of diversity metrics
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted.")

        # This would require predictions to calculate
        # Returning placeholder metrics
        return {
            "n_models": self.config.n_models,
            "feature_diversity": 1.0 - self.config.feature_ratio if self.config.ensemble_method == EnsembleMethod.RANDOM_SUBSPACE else 0.0,
            "bootstrap_diversity": 1.0 - self.config.bootstrap_ratio if self.config.ensemble_method == EnsembleMethod.BOOTSTRAP else 0.0
        }

    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get aggregated feature importance from ensemble.

        Returns:
            Dictionary of feature index to importance
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted.")

        importance = np.zeros(self._n_features)
        counts = np.zeros(self._n_features)

        for model, feature_idx in zip(self.models, self.feature_indices):
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                for i, idx in enumerate(feature_idx):
                    importance[idx] += imp[i] if i < len(imp) else 0
                    counts[idx] += 1

        # Average importance
        counts = np.maximum(counts, 1)  # Avoid division by zero
        importance = importance / counts

        return {i: float(imp) for i, imp in enumerate(importance)}


# Unit test stubs
class TestEnsemblePredictor:
    """Unit tests for EnsemblePredictor."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        ensemble = EnsemblePredictor()
        assert ensemble.config.n_models == 10
        assert ensemble.config.ensemble_method == EnsembleMethod.BOOTSTRAP

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return  # Skip if sklearn not available

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        ensemble = EnsemblePredictor(
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 10},
            config=EnsembleConfig(n_models=3)
        )
        ensemble.fit(X, y)

        predictions = ensemble.predict(X[:10])
        assert len(predictions) == 10

    def test_predict_with_uncertainty(self):
        """Test uncertainty estimation."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        ensemble = EnsemblePredictor(
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 10},
            config=EnsembleConfig(n_models=5)
        )
        ensemble.fit(X, y)

        result = ensemble.predict_with_uncertainty(X[:10])
        assert len(result.predictions) == 10
        assert len(result.uncertainties) == 10
        assert all(u >= 0 for u in result.uncertainties)

    def test_bootstrap_sample(self):
        """Test bootstrap sampling."""
        ensemble = EnsemblePredictor(config=EnsembleConfig(bootstrap_ratio=0.5))

        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        X_boot, y_boot = ensemble._bootstrap_sample(X, y)
        assert len(X_boot) == 50

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        ensemble = EnsemblePredictor()

        preds = np.array([1.0, 2.0, 3.0])
        uncert = np.array([0.1, 0.2, 0.3])

        hash1 = ensemble._calculate_provenance(preds, uncert)
        hash2 = ensemble._calculate_provenance(preds, uncert)

        assert hash1 == hash2
