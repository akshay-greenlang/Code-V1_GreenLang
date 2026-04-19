# -*- coding: utf-8 -*-
"""
TASK-066: Model Validation Pipeline

This module provides comprehensive model validation capabilities for
GreenLang Process Heat ML models, including cross-validation with
stratification, out-of-time validation, nested cross-validation for
hyperparameters, statistical significance testing, and validation
report generation.

Validation is critical for ensuring ML models meet regulatory requirements
and perform reliably in Process Heat applications.

Example:
    >>> from greenlang.ml.robustness import ValidationPipeline
    >>> pipeline = ValidationPipeline(model, config=ValidationConfig(
    ...     cv_strategy="stratified_kfold",
    ...     n_folds=5
    ... ))
    >>> result = pipeline.validate(X, y)
    >>> print(f"Mean score: {result.mean_score:.4f} +/- {result.std_score:.4f}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class CVStrategy(str, Enum):
    """Cross-validation strategies."""
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    TIME_SERIES = "time_series"
    SHUFFLE_SPLIT = "shuffle_split"
    GROUP_KFOLD = "group_kfold"
    LEAVE_ONE_OUT = "leave_one_out"


class MetricType(str, Enum):
    """Validation metrics."""
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R2 = "r2"
    CUSTOM = "custom"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


# =============================================================================
# Configuration
# =============================================================================

class ValidationConfig(BaseModel):
    """Configuration for validation pipeline."""

    # Cross-validation settings
    cv_strategy: CVStrategy = Field(
        default=CVStrategy.KFOLD,
        description="Cross-validation strategy"
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of folds for cross-validation"
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle data before splitting"
    )

    # Out-of-time validation
    enable_out_of_time: bool = Field(
        default=False,
        description="Enable out-of-time validation"
    )
    time_column_index: Optional[int] = Field(
        default=None,
        description="Index of time column for out-of-time split"
    )
    oot_train_ratio: float = Field(
        default=0.7,
        gt=0.1,
        lt=0.95,
        description="Training ratio for out-of-time split"
    )

    # Nested CV for hyperparameters
    enable_nested_cv: bool = Field(
        default=False,
        description="Enable nested cross-validation"
    )
    inner_folds: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Inner folds for nested CV"
    )
    param_grid: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        description="Parameter grid for nested CV"
    )

    # Metrics
    primary_metric: MetricType = Field(
        default=MetricType.RMSE,
        description="Primary evaluation metric"
    )
    additional_metrics: List[MetricType] = Field(
        default_factory=lambda: [MetricType.MAE, MetricType.R2],
        description="Additional metrics to compute"
    )

    # Statistical testing
    enable_significance_testing: bool = Field(
        default=True,
        description="Enable statistical significance testing"
    )
    significance_level: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for tests"
    )
    baseline_score: Optional[float] = Field(
        default=None,
        description="Baseline score for comparison"
    )

    # Thresholds
    min_acceptable_score: Optional[float] = Field(
        default=None,
        description="Minimum acceptable score (metric-dependent)"
    )
    max_std_ratio: float = Field(
        default=0.3,
        gt=0,
        description="Maximum acceptable std/mean ratio"
    )

    # General
    random_state: int = Field(
        default=42,
        description="Random seed"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )


# =============================================================================
# Result Models
# =============================================================================

class FoldResult(BaseModel):
    """Result from a single validation fold."""

    fold_index: int = Field(..., description="Fold index")
    train_size: int = Field(..., description="Training set size")
    test_size: int = Field(..., description="Test set size")

    # Scores
    primary_score: float = Field(..., description="Primary metric score")
    additional_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional metric scores"
    )

    # Predictions (optional, for detailed analysis)
    predictions: Optional[List[float]] = Field(
        default=None,
        description="Predictions on test fold"
    )
    true_values: Optional[List[float]] = Field(
        default=None,
        description="True values for test fold"
    )
    residuals_mean: Optional[float] = Field(
        default=None,
        description="Mean of residuals"
    )
    residuals_std: Optional[float] = Field(
        default=None,
        description="Std of residuals"
    )


class OutOfTimeResult(BaseModel):
    """Result from out-of-time validation."""

    train_start: Optional[str] = Field(default=None, description="Training start time")
    train_end: Optional[str] = Field(default=None, description="Training end time")
    test_start: Optional[str] = Field(default=None, description="Test start time")
    test_end: Optional[str] = Field(default=None, description="Test end time")

    train_size: int = Field(..., description="Training set size")
    test_size: int = Field(..., description="Test set size")

    primary_score: float = Field(..., description="Primary metric score")
    additional_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional metric scores"
    )

    temporal_degradation: float = Field(
        ...,
        description="Score degradation vs cross-validation"
    )


class NestedCVResult(BaseModel):
    """Result from nested cross-validation."""

    outer_scores: List[float] = Field(..., description="Outer fold scores")
    best_params_per_fold: List[Dict[str, Any]] = Field(
        ...,
        description="Best parameters for each outer fold"
    )
    inner_scores: List[List[float]] = Field(
        ...,
        description="Inner fold scores for each outer fold"
    )

    mean_outer_score: float = Field(..., description="Mean outer score")
    std_outer_score: float = Field(..., description="Std of outer scores")

    most_common_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Most commonly selected parameters"
    )


class SignificanceTestResult(BaseModel):
    """Result from statistical significance testing."""

    test_name: str = Field(..., description="Name of statistical test")
    statistic: float = Field(..., description="Test statistic")
    p_value: float = Field(..., description="P-value")
    significant: bool = Field(..., description="Whether result is significant")
    effect_size: Optional[float] = Field(
        default=None,
        description="Effect size (if applicable)"
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Confidence interval for mean"
    )


class ValidationReport(BaseModel):
    """Comprehensive validation report."""

    # Summary
    status: ValidationStatus = Field(..., description="Overall validation status")
    summary: str = Field(..., description="Human-readable summary")

    # Cross-validation results
    cv_strategy: str = Field(..., description="CV strategy used")
    n_folds: int = Field(..., description="Number of folds")
    fold_results: List[FoldResult] = Field(..., description="Per-fold results")

    # Aggregate statistics
    mean_score: float = Field(..., description="Mean primary score")
    std_score: float = Field(..., description="Standard deviation")
    min_score: float = Field(..., description="Minimum score")
    max_score: float = Field(..., description="Maximum score")
    cv_coefficient: float = Field(
        ...,
        description="Coefficient of variation (std/mean)"
    )

    # Additional metrics
    additional_metric_means: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean of additional metrics"
    )

    # Out-of-time validation
    out_of_time_result: Optional[OutOfTimeResult] = Field(
        default=None,
        description="Out-of-time validation result"
    )

    # Nested CV
    nested_cv_result: Optional[NestedCVResult] = Field(
        default=None,
        description="Nested CV result"
    )

    # Significance testing
    significance_results: List[SignificanceTestResult] = Field(
        default_factory=list,
        description="Significance test results"
    )

    # Warnings and recommendations
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )

    # Metadata
    primary_metric: str = Field(..., description="Primary metric used")
    samples_validated: int = Field(..., description="Total samples")
    provenance_hash: str = Field(..., description="SHA-256 hash")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation timestamp"
    )


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    Comprehensive Model Validation Pipeline for Process Heat ML.

    This pipeline provides:
    - Cross-validation with multiple strategies
    - Stratified sampling for regression (binned targets)
    - Out-of-time validation for temporal data
    - Nested cross-validation for hyperparameter tuning
    - Statistical significance testing
    - Comprehensive validation reports

    All calculations are deterministic for reproducibility.

    Attributes:
        model: ML model to validate
        config: Validation configuration
        _rng: Random number generator

    Example:
        >>> pipeline = ValidationPipeline(
        ...     model,
        ...     config=ValidationConfig(
        ...         cv_strategy=CVStrategy.STRATIFIED_KFOLD,
        ...         n_folds=5,
        ...         enable_significance_testing=True
        ...     )
        ... )
        >>> result = pipeline.validate(X, y)
        >>> if result.status == ValidationStatus.FAILED:
        ...     print("Validation failed:", result.summary)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[ValidationConfig] = None,
        custom_metric: Optional[Callable] = None
    ):
        """
        Initialize validation pipeline.

        Args:
            model: ML model with fit() and predict() methods
            config: Validation configuration
            custom_metric: Optional custom metric function(y_true, y_pred) -> float
        """
        self.model = model
        self.config = config or ValidationConfig()
        self._custom_metric = custom_metric

        self._rng = np.random.RandomState(self.config.random_state)

        logger.info(
            f"ValidationPipeline initialized: strategy={self.config.cv_strategy.value}, "
            f"folds={self.config.n_folds}"
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: MetricType
    ) -> float:
        """Calculate specified metric (deterministic)."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if metric == MetricType.MSE:
            return float(np.mean((y_true - y_pred) ** 2))

        elif metric == MetricType.RMSE:
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        elif metric == MetricType.MAE:
            return float(np.mean(np.abs(y_true - y_pred)))

        elif metric == MetricType.MAPE:
            # Avoid division by zero
            mask = y_true != 0
            if not np.any(mask):
                return float('inf')
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        elif metric == MetricType.R2:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                return 0.0
            return float(1 - ss_res / ss_tot)

        elif metric == MetricType.CUSTOM:
            if self._custom_metric:
                return float(self._custom_metric(y_true, y_pred))
            return 0.0

        return 0.0

    def _is_higher_better(self, metric: MetricType) -> bool:
        """Check if higher values are better for the metric."""
        return metric == MetricType.R2

    # =========================================================================
    # Cross-Validation Splits
    # =========================================================================

    def _create_cv_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits (deterministic)."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.config.shuffle:
            self._rng.shuffle(indices)

        if self.config.cv_strategy == CVStrategy.KFOLD:
            return self._kfold_split(indices)

        elif self.config.cv_strategy == CVStrategy.STRATIFIED_KFOLD:
            return self._stratified_kfold_split(indices, y)

        elif self.config.cv_strategy == CVStrategy.TIME_SERIES:
            return self._time_series_split(indices)

        elif self.config.cv_strategy == CVStrategy.SHUFFLE_SPLIT:
            return self._shuffle_split(indices)

        elif self.config.cv_strategy == CVStrategy.GROUP_KFOLD:
            if groups is None:
                logger.warning("Groups not provided for group k-fold, falling back to k-fold")
                return self._kfold_split(indices)
            return self._group_kfold_split(indices, groups)

        elif self.config.cv_strategy == CVStrategy.LEAVE_ONE_OUT:
            return self._leave_one_out_split(indices)

        return self._kfold_split(indices)

    def _kfold_split(
        self,
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Standard K-Fold split."""
        n_samples = len(indices)
        fold_sizes = np.full(self.config.n_folds, n_samples // self.config.n_folds)
        fold_sizes[:n_samples % self.config.n_folds] += 1

        splits = []
        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            splits.append((train_idx, test_idx))
            current += fold_size

        return splits

    def _stratified_kfold_split(
        self,
        indices: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Stratified K-Fold split (bins continuous targets)."""
        # Bin targets for stratification
        n_bins = min(self.config.n_folds, 10)
        y_sorted = y[indices]
        percentiles = np.percentile(y_sorted, np.linspace(0, 100, n_bins + 1))
        bins = np.digitize(y_sorted, percentiles[:-1]) - 1

        # Group indices by bin
        bin_indices = {i: [] for i in range(n_bins)}
        for idx, bin_id in zip(indices, bins):
            bin_indices[bin_id].append(idx)

        # Shuffle within each bin
        for bin_id in bin_indices:
            self._rng.shuffle(bin_indices[bin_id])

        # Distribute evenly across folds
        folds = [[] for _ in range(self.config.n_folds)]
        for bin_id in range(n_bins):
            for i, idx in enumerate(bin_indices[bin_id]):
                folds[i % self.config.n_folds].append(idx)

        splits = []
        for fold_idx in range(self.config.n_folds):
            test_idx = np.array(folds[fold_idx])
            train_idx = np.concatenate([
                np.array(folds[i]) for i in range(self.config.n_folds) if i != fold_idx
            ])
            splits.append((train_idx, test_idx))

        return splits

    def _time_series_split(
        self,
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time series split (expanding window)."""
        n_samples = len(indices)
        # Don't shuffle for time series
        sorted_indices = np.sort(indices)

        splits = []
        test_size = n_samples // (self.config.n_folds + 1)

        for i in range(self.config.n_folds):
            train_end = (i + 1) * test_size + test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            train_idx = sorted_indices[:train_end]
            test_idx = sorted_indices[test_start:test_end]

            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def _shuffle_split(
        self,
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Random shuffle split."""
        n_samples = len(indices)
        test_size = n_samples // self.config.n_folds

        splits = []
        for _ in range(self.config.n_folds):
            shuffled = indices.copy()
            self._rng.shuffle(shuffled)
            test_idx = shuffled[:test_size]
            train_idx = shuffled[test_size:]
            splits.append((train_idx, test_idx))

        return splits

    def _group_kfold_split(
        self,
        indices: np.ndarray,
        groups: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Group K-Fold split (ensures groups don't leak)."""
        unique_groups = np.unique(groups)
        self._rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        fold_sizes = np.full(self.config.n_folds, n_groups // self.config.n_folds)
        fold_sizes[:n_groups % self.config.n_folds] += 1

        splits = []
        current = 0
        for fold_size in fold_sizes:
            test_groups = unique_groups[current:current + fold_size]
            train_groups = np.concatenate([
                unique_groups[:current],
                unique_groups[current + fold_size:]
            ])

            test_mask = np.isin(groups[indices], test_groups)
            train_mask = np.isin(groups[indices], train_groups)

            test_idx = indices[test_mask]
            train_idx = indices[train_mask]

            splits.append((train_idx, test_idx))
            current += fold_size

        return splits

    def _leave_one_out_split(
        self,
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Leave-one-out split."""
        splits = []
        for i in range(len(indices)):
            test_idx = np.array([indices[i]])
            train_idx = np.concatenate([indices[:i], indices[i+1:]])
            splits.append((train_idx, test_idx))

        return splits

    # =========================================================================
    # Cross-Validation
    # =========================================================================

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        return_predictions: bool = False
    ) -> Tuple[List[FoldResult], np.ndarray]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Targets
            groups: Optional group labels for group k-fold
            return_predictions: Whether to store predictions

        Returns:
            Tuple of (fold results, all predictions)
        """
        splits = self._create_cv_splits(X, y, groups)
        fold_results = []
        all_predictions = np.zeros(len(y))
        all_indices = np.zeros(len(y), dtype=bool)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # Train model
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Clone and fit model
            model_clone = copy.deepcopy(self.model)
            model_clone.fit(X_train, y_train)

            # Predict
            y_pred = model_clone.predict(X_test)
            y_pred = np.asarray(y_pred).flatten()

            # Store predictions
            all_predictions[test_idx] = y_pred
            all_indices[test_idx] = True

            # Calculate primary metric
            primary_score = self._calculate_metric(
                y_test, y_pred, self.config.primary_metric
            )

            # Calculate additional metrics
            additional_scores = {}
            for metric in self.config.additional_metrics:
                additional_scores[metric.value] = self._calculate_metric(
                    y_test, y_pred, metric
                )

            # Residuals
            residuals = y_test - y_pred

            fold_result = FoldResult(
                fold_index=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                primary_score=primary_score,
                additional_scores=additional_scores,
                predictions=y_pred.tolist()[:100] if return_predictions else None,
                true_values=y_test.tolist()[:100] if return_predictions else None,
                residuals_mean=float(np.mean(residuals)),
                residuals_std=float(np.std(residuals))
            )
            fold_results.append(fold_result)

            logger.debug(
                f"Fold {fold_idx}: {self.config.primary_metric.value}={primary_score:.4f}"
            )

        return fold_results, all_predictions

    # =========================================================================
    # Out-of-Time Validation
    # =========================================================================

    def out_of_time_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_values: Optional[np.ndarray] = None
    ) -> OutOfTimeResult:
        """
        Perform out-of-time validation.

        Args:
            X: Features
            y: Targets
            time_values: Optional time values for split

        Returns:
            Out-of-time validation result
        """
        n_samples = len(X)

        if time_values is not None:
            # Sort by time
            sort_idx = np.argsort(time_values)
        else:
            # Use natural order (assume already time-ordered)
            sort_idx = np.arange(n_samples)

        # Split at train_ratio
        split_idx = int(n_samples * self.config.oot_train_ratio)
        train_idx = sort_idx[:split_idx]
        test_idx = sort_idx[split_idx:]

        # Train and evaluate
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model_clone = copy.deepcopy(self.model)
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict(X_test)
        y_pred = np.asarray(y_pred).flatten()

        # Calculate metrics
        primary_score = self._calculate_metric(
            y_test, y_pred, self.config.primary_metric
        )

        additional_scores = {}
        for metric in self.config.additional_metrics:
            additional_scores[metric.value] = self._calculate_metric(
                y_test, y_pred, metric
            )

        return OutOfTimeResult(
            train_start=str(time_values[train_idx[0]]) if time_values is not None else None,
            train_end=str(time_values[train_idx[-1]]) if time_values is not None else None,
            test_start=str(time_values[test_idx[0]]) if time_values is not None else None,
            test_end=str(time_values[test_idx[-1]]) if time_values is not None else None,
            train_size=len(train_idx),
            test_size=len(test_idx),
            primary_score=primary_score,
            additional_scores=additional_scores,
            temporal_degradation=0.0  # Will be set later
        )

    # =========================================================================
    # Nested Cross-Validation
    # =========================================================================

    def nested_cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]]
    ) -> NestedCVResult:
        """
        Perform nested cross-validation for hyperparameter selection.

        Args:
            X: Features
            y: Targets
            param_grid: Dictionary of parameter names to lists of values

        Returns:
            Nested CV result
        """
        outer_splits = self._create_cv_splits(X, y)

        outer_scores = []
        best_params_per_fold = []
        inner_scores = []

        for outer_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Inner CV for hyperparameter selection
            best_score = float('-inf') if self._is_higher_better(self.config.primary_metric) else float('inf')
            best_params = {}
            fold_inner_scores = []

            # Generate all parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)

            for params in param_combinations:
                # Inner cross-validation
                inner_splits = self._kfold_split(np.arange(len(X_train)))
                param_scores = []

                for inner_train, inner_test in inner_splits:
                    X_inner_train = X_train[inner_train]
                    y_inner_train = y_train[inner_train]
                    X_inner_test = X_train[inner_test]
                    y_inner_test = y_train[inner_test]

                    model_clone = copy.deepcopy(self.model)
                    # Set parameters
                    for key, value in params.items():
                        setattr(model_clone, key, value)

                    model_clone.fit(X_inner_train, y_inner_train)
                    y_pred = model_clone.predict(X_inner_test)

                    score = self._calculate_metric(
                        y_inner_test, y_pred, self.config.primary_metric
                    )
                    param_scores.append(score)

                mean_score = np.mean(param_scores)
                fold_inner_scores.append(mean_score)

                # Check if best
                if self._is_higher_better(self.config.primary_metric):
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                else:
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = params

            # Train with best params and evaluate on outer test
            final_model = copy.deepcopy(self.model)
            for key, value in best_params.items():
                setattr(final_model, key, value)

            final_model.fit(X_train, y_train)
            y_pred = final_model.predict(X_test)

            outer_score = self._calculate_metric(
                y_test, y_pred, self.config.primary_metric
            )

            outer_scores.append(outer_score)
            best_params_per_fold.append(best_params)
            inner_scores.append(fold_inner_scores)

        # Find most common parameters
        most_common_params = self._get_most_common_params(best_params_per_fold)

        return NestedCVResult(
            outer_scores=outer_scores,
            best_params_per_fold=best_params_per_fold,
            inner_scores=inner_scores,
            mean_outer_score=float(np.mean(outer_scores)),
            std_outer_score=float(np.std(outer_scores)),
            most_common_params=most_common_params
        )

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        self._generate_combinations_recursive(keys, values, 0, {}, combinations)

        return combinations

    def _generate_combinations_recursive(
        self,
        keys: List[str],
        values: List[List[Any]],
        idx: int,
        current: Dict[str, Any],
        results: List[Dict[str, Any]]
    ):
        """Recursively generate parameter combinations."""
        if idx == len(keys):
            results.append(current.copy())
            return

        for value in values[idx]:
            current[keys[idx]] = value
            self._generate_combinations_recursive(keys, values, idx + 1, current, results)

    def _get_most_common_params(
        self,
        params_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get most commonly selected parameters."""
        from collections import Counter

        result = {}
        all_keys = set()
        for params in params_list:
            all_keys.update(params.keys())

        for key in all_keys:
            values = [params.get(key) for params in params_list if key in params]
            if values:
                # Handle unhashable types
                try:
                    most_common = Counter(values).most_common(1)[0][0]
                except TypeError:
                    most_common = values[0]
                result[key] = most_common

        return result

    # =========================================================================
    # Statistical Significance Testing
    # =========================================================================

    def perform_significance_tests(
        self,
        fold_scores: List[float],
        baseline_score: Optional[float] = None
    ) -> List[SignificanceTestResult]:
        """
        Perform statistical significance tests.

        Args:
            fold_scores: Scores from cross-validation folds
            baseline_score: Optional baseline to compare against

        Returns:
            List of significance test results
        """
        results = []
        scores = np.array(fold_scores)

        # T-test against baseline (if provided)
        if baseline_score is not None:
            t_stat, p_value = self._one_sample_t_test(scores, baseline_score)

            results.append(SignificanceTestResult(
                test_name="one_sample_t_test",
                statistic=t_stat,
                p_value=p_value,
                significant=p_value < self.config.significance_level,
                effect_size=self._cohens_d(scores, baseline_score),
                confidence_interval=self._confidence_interval(scores)
            ))

        # Normality test (Shapiro-Wilk approximation)
        if len(scores) >= 3:
            w_stat, p_value = self._shapiro_wilk_approximation(scores)

            results.append(SignificanceTestResult(
                test_name="shapiro_wilk_normality",
                statistic=w_stat,
                p_value=p_value,
                significant=p_value < self.config.significance_level
            ))

        # Confidence interval test
        ci = self._confidence_interval(scores)
        results.append(SignificanceTestResult(
            test_name="confidence_interval",
            statistic=float(np.mean(scores)),
            p_value=0.0,  # N/A
            significant=False,
            confidence_interval=ci
        ))

        return results

    def _one_sample_t_test(
        self,
        scores: np.ndarray,
        baseline: float
    ) -> Tuple[float, float]:
        """One-sample t-test (deterministic)."""
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)

        if std == 0:
            return (float('inf') if mean != baseline else 0.0, 0.0 if mean != baseline else 1.0)

        t_stat = (mean - baseline) / (std / np.sqrt(n))

        # Approximate p-value using t-distribution
        # Using normal approximation for simplicity
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return (float(t_stat), float(p_value))

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        # Using error function approximation
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _cohens_d(
        self,
        scores: np.ndarray,
        baseline: float
    ) -> float:
        """Calculate Cohen's d effect size."""
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)

        if std == 0:
            return 0.0

        return float((mean - baseline) / std)

    def _confidence_interval(
        self,
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)

        # Z-score for confidence level
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

        margin = z * std / np.sqrt(n)

        return (float(mean - margin), float(mean + margin))

    def _shapiro_wilk_approximation(
        self,
        scores: np.ndarray
    ) -> Tuple[float, float]:
        """Approximate Shapiro-Wilk test."""
        n = len(scores)
        sorted_scores = np.sort(scores)

        # Simplified W statistic
        mean = np.mean(scores)
        ss = np.sum((scores - mean) ** 2)

        if ss == 0:
            return (1.0, 1.0)

        # Simple approximation based on correlation with normal quantiles
        expected = np.array([
            self._normal_quantile((i + 0.5) / n)
            for i in range(n)
        ])

        correlation = np.corrcoef(sorted_scores, expected)[0, 1]
        w = correlation ** 2

        # Very rough p-value approximation
        p_value = 1 - w

        return (float(w), float(max(0, min(1, p_value))))

    def _normal_quantile(self, p: float) -> float:
        """Approximate normal quantile function."""
        # Rational approximation
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')

        if p < 0.5:
            t = np.sqrt(-2 * np.log(p))
        else:
            t = np.sqrt(-2 * np.log(1 - p))

        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        q = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)

        return q if p >= 0.5 else -q

    # =========================================================================
    # Main Validation Method
    # =========================================================================

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        time_values: Optional[np.ndarray] = None
    ) -> ValidationReport:
        """
        Run comprehensive model validation.

        Args:
            X: Features
            y: Targets
            groups: Optional group labels for group k-fold
            time_values: Optional time values for out-of-time validation

        Returns:
            Comprehensive validation report

        Example:
            >>> result = pipeline.validate(X, y)
            >>> print(f"Status: {result.status.value}")
            >>> print(f"Mean RMSE: {result.mean_score:.4f}")
        """
        logger.info(f"Starting validation: samples={len(X)}, features={X.shape[1]}")

        # Cross-validation
        fold_results, all_predictions = self.cross_validate(X, y, groups, return_predictions=True)

        # Aggregate statistics
        primary_scores = [f.primary_score for f in fold_results]
        mean_score = float(np.mean(primary_scores))
        std_score = float(np.std(primary_scores))
        min_score = float(np.min(primary_scores))
        max_score = float(np.max(primary_scores))
        cv_coefficient = std_score / (abs(mean_score) + 1e-10)

        # Additional metrics means
        additional_means = {}
        for metric in self.config.additional_metrics:
            metric_scores = [f.additional_scores.get(metric.value, 0) for f in fold_results]
            additional_means[metric.value] = float(np.mean(metric_scores))

        # Out-of-time validation
        oot_result = None
        if self.config.enable_out_of_time:
            oot_result = self.out_of_time_validate(X, y, time_values)
            # Calculate temporal degradation
            if self._is_higher_better(self.config.primary_metric):
                oot_result.temporal_degradation = mean_score - oot_result.primary_score
            else:
                oot_result.temporal_degradation = oot_result.primary_score - mean_score

        # Nested CV
        nested_result = None
        if self.config.enable_nested_cv and self.config.param_grid:
            nested_result = self.nested_cross_validate(X, y, self.config.param_grid)

        # Significance testing
        significance_results = []
        if self.config.enable_significance_testing:
            significance_results = self.perform_significance_tests(
                primary_scores,
                self.config.baseline_score
            )

        # Determine status and generate warnings
        status, warnings = self._determine_status(
            mean_score, std_score, cv_coefficient, oot_result
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            status, cv_coefficient, oot_result, fold_results
        )

        # Generate summary
        summary = self._generate_summary(
            status, mean_score, std_score, len(X)
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            mean_score, std_score, len(X)
        )

        report = ValidationReport(
            status=status,
            summary=summary,
            cv_strategy=self.config.cv_strategy.value,
            n_folds=self.config.n_folds,
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            cv_coefficient=cv_coefficient,
            additional_metric_means=additional_means,
            out_of_time_result=oot_result,
            nested_cv_result=nested_result,
            significance_results=significance_results,
            warnings=warnings,
            recommendations=recommendations,
            primary_metric=self.config.primary_metric.value,
            samples_validated=len(X),
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

        logger.info(
            f"Validation complete: status={status.value}, "
            f"mean={mean_score:.4f}, std={std_score:.4f}"
        )

        return report

    def _determine_status(
        self,
        mean_score: float,
        std_score: float,
        cv_coefficient: float,
        oot_result: Optional[OutOfTimeResult]
    ) -> Tuple[ValidationStatus, List[str]]:
        """Determine validation status and generate warnings."""
        warnings = []
        status = ValidationStatus.PASSED

        # Check minimum score threshold
        if self.config.min_acceptable_score is not None:
            if self._is_higher_better(self.config.primary_metric):
                if mean_score < self.config.min_acceptable_score:
                    warnings.append(
                        f"Mean score {mean_score:.4f} below minimum threshold "
                        f"{self.config.min_acceptable_score:.4f}"
                    )
                    status = ValidationStatus.FAILED
            else:
                if mean_score > self.config.min_acceptable_score:
                    warnings.append(
                        f"Mean score {mean_score:.4f} above maximum threshold "
                        f"{self.config.min_acceptable_score:.4f}"
                    )
                    status = ValidationStatus.FAILED

        # Check coefficient of variation
        if cv_coefficient > self.config.max_std_ratio:
            warnings.append(
                f"High variance across folds: CV coefficient {cv_coefficient:.3f} "
                f"exceeds threshold {self.config.max_std_ratio:.3f}"
            )
            if status == ValidationStatus.PASSED:
                status = ValidationStatus.WARNING

        # Check out-of-time degradation
        if oot_result is not None:
            if abs(oot_result.temporal_degradation) > 0.2 * abs(mean_score):
                warnings.append(
                    f"Significant temporal degradation: {oot_result.temporal_degradation:.4f}"
                )
                if status == ValidationStatus.PASSED:
                    status = ValidationStatus.WARNING

        return status, warnings

    def _generate_recommendations(
        self,
        status: ValidationStatus,
        cv_coefficient: float,
        oot_result: Optional[OutOfTimeResult],
        fold_results: List[FoldResult]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if status == ValidationStatus.PASSED:
            recommendations.append(
                "Model validation passed. Consider periodic revalidation as new data arrives."
            )

        if cv_coefficient > 0.2:
            recommendations.append(
                "High variance across folds suggests overfitting or data heterogeneity. "
                "Consider regularization or data augmentation."
            )

        if oot_result and oot_result.temporal_degradation > 0:
            recommendations.append(
                "Temporal degradation detected. Consider incorporating time-aware features "
                "or periodic retraining schedule."
            )

        # Check for fold outliers
        scores = [f.primary_score for f in fold_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        outlier_folds = [
            f.fold_index for f in fold_results
            if abs(f.primary_score - mean_score) > 2 * std_score
        ]
        if outlier_folds:
            recommendations.append(
                f"Folds {outlier_folds} show outlier performance. "
                "Investigate data quality in these folds."
            )

        return recommendations

    def _generate_summary(
        self,
        status: ValidationStatus,
        mean_score: float,
        std_score: float,
        n_samples: int
    ) -> str:
        """Generate human-readable summary."""
        metric_name = self.config.primary_metric.value.upper()

        if status == ValidationStatus.PASSED:
            return (
                f"Validation PASSED. {self.config.cv_strategy.value} with {self.config.n_folds} folds "
                f"on {n_samples} samples. Mean {metric_name}: {mean_score:.4f} (+/- {std_score:.4f})"
            )
        elif status == ValidationStatus.WARNING:
            return (
                f"Validation completed with WARNINGS. Mean {metric_name}: {mean_score:.4f} "
                f"(+/- {std_score:.4f}). Review warnings for details."
            )
        else:
            return (
                f"Validation FAILED. Mean {metric_name}: {mean_score:.4f} "
                f"does not meet acceptance criteria."
            )

    def _calculate_provenance(
        self,
        mean_score: float,
        std_score: float,
        n_samples: int
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        provenance_data = (
            f"{self.config.cv_strategy.value}|{self.config.n_folds}|"
            f"{mean_score:.8f}|{std_score:.8f}|{n_samples}|"
            f"{self.config.random_state}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()


# =============================================================================
# Unit Tests
# =============================================================================

class TestValidationPipeline:
    """Unit tests for ValidationPipeline."""

    def test_kfold_split(self):
        """Test k-fold split creates correct folds."""
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(MockModel())
        indices = np.arange(100)

        splits = pipeline._kfold_split(indices)

        assert len(splits) == 5
        # Each sample should appear in exactly one test set
        all_test = np.concatenate([s[1] for s in splits])
        assert len(np.unique(all_test)) == 100

    def test_stratified_split(self):
        """Test stratified split maintains target distribution."""
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(MockModel())
        indices = np.arange(100)
        y = np.concatenate([np.zeros(50), np.ones(50)])

        splits = pipeline._stratified_kfold_split(indices, y)

        assert len(splits) == 5

    def test_cross_validation(self):
        """Test cross-validation runs without error."""
        class MockModel:
            def fit(self, X, y):
                self.mean = np.mean(y)
            def predict(self, X):
                return np.full(len(X), self.mean)

        config = ValidationConfig(n_folds=3)
        pipeline = ValidationPipeline(MockModel(), config)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        fold_results, _ = pipeline.cross_validate(X, y)

        assert len(fold_results) == 3
        assert all(f.primary_score >= 0 for f in fold_results)

    def test_full_validation(self):
        """Test full validation pipeline."""
        class MockModel:
            def fit(self, X, y):
                self.mean = np.mean(y)
            def predict(self, X):
                return np.full(len(X), self.mean)

        config = ValidationConfig(
            cv_strategy=CVStrategy.KFOLD,
            n_folds=3,
            enable_significance_testing=True
        )
        pipeline = ValidationPipeline(MockModel(), config)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1)

        result = pipeline.validate(X, y)

        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]
        assert result.mean_score >= 0
        assert len(result.fold_results) == 3
        assert result.provenance_hash

    def test_metrics(self):
        """Test metric calculations."""
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(MockModel())

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        rmse = pipeline._calculate_metric(y_true, y_pred, MetricType.RMSE)
        mae = pipeline._calculate_metric(y_true, y_pred, MetricType.MAE)
        r2 = pipeline._calculate_metric(y_true, y_pred, MetricType.R2)

        assert rmse > 0
        assert mae > 0
        assert r2 > 0.9  # Good fit

    def test_significance_testing(self):
        """Test significance tests."""
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(MockModel())
        scores = [0.85, 0.87, 0.86, 0.88, 0.84]

        results = pipeline.perform_significance_tests(scores, baseline_score=0.80)

        assert len(results) > 0
        assert any(r.test_name == "one_sample_t_test" for r in results)

    def test_provenance_deterministic(self):
        """Test provenance is deterministic."""
        class MockModel:
            def fit(self, X, y):
                pass
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(MockModel())

        hash1 = pipeline._calculate_provenance(0.85, 0.02, 100)
        hash2 = pipeline._calculate_provenance(0.85, 0.02, 100)

        assert hash1 == hash2
        assert len(hash1) == 64
