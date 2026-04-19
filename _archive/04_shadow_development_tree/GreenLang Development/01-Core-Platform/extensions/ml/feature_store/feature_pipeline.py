# -*- coding: utf-8 -*-
"""
Feature Engineering Pipeline for GreenLang Process Heat Agents

This module provides the feature engineering pipeline for Process Heat
agents, including:
- Rolling aggregations (1h, 24h, 7d windows)
- Lag features for time series analysis
- Statistical features (mean, std, min, max, percentiles)
- Cross-feature interactions
- Feature normalization and scaling

The pipeline follows GreenLang's zero-hallucination principles by
using deterministic transformations with SHA-256 provenance tracking.

Example:
    >>> from greenlang.ml.feature_store.feature_pipeline import FeaturePipeline
    >>>
    >>> pipeline = FeaturePipeline()
    >>> transformed_features = pipeline.transform(raw_features)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class AggregationType(str, Enum):
    """Types of aggregation functions."""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    RANGE = "range"
    PERCENTILE_25 = "percentile_25"
    PERCENTILE_75 = "percentile_75"
    PERCENTILE_90 = "percentile_90"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"


class WindowType(str, Enum):
    """Types of rolling windows."""
    HOURLY_1 = "1h"
    HOURLY_4 = "4h"
    HOURLY_8 = "8h"
    HOURLY_12 = "12h"
    DAILY_1 = "24h"
    DAILY_7 = "7d"
    DAILY_30 = "30d"
    DAILY_90 = "90d"


class ScalingMethod(str, Enum):
    """Methods for feature scaling."""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"  # 0-1 scaling
    ROBUST = "robust"  # Median/IQR scaling
    LOG = "log"  # Log transformation
    NONE = "none"  # No scaling


class RollingAggregation(BaseModel):
    """
    Configuration for a rolling aggregation.

    Attributes:
        feature_name: Source feature to aggregate
        window: Rolling window size
        aggregations: List of aggregation functions to apply
        min_periods: Minimum number of observations required

    Example:
        >>> config = RollingAggregation(
        ...     feature_name="efficiency",
        ...     window="24h",
        ...     aggregations=["mean", "std", "min", "max"]
        ... )
    """

    feature_name: str = Field(
        ...,
        description="Source feature name"
    )
    window: WindowType = Field(
        ...,
        description="Rolling window size"
    )
    aggregations: List[AggregationType] = Field(
        default_factory=lambda: [AggregationType.MEAN, AggregationType.STD],
        description="Aggregation functions to apply"
    )
    min_periods: int = Field(
        default=1,
        ge=1,
        description="Minimum observations in window"
    )
    output_prefix: Optional[str] = Field(
        None,
        description="Prefix for output feature names"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

    def get_output_feature_names(self) -> List[str]:
        """Get list of output feature names."""
        prefix = self.output_prefix or self.feature_name
        return [
            f"{prefix}_{self.window}_{agg}"
            for agg in self.aggregations
        ]


class LagFeatureConfig(BaseModel):
    """
    Configuration for lag features.

    Attributes:
        feature_name: Source feature to create lags for
        lag_periods: List of lag periods
        include_diff: Whether to include difference features

    Example:
        >>> config = LagFeatureConfig(
        ...     feature_name="steam_flow",
        ...     lag_periods=[1, 6, 24],
        ...     include_diff=True
        ... )
    """

    feature_name: str = Field(
        ...,
        description="Source feature name"
    )
    lag_periods: List[int] = Field(
        default_factory=lambda: [1, 6, 12, 24],
        description="Lag periods (in data points)"
    )
    include_diff: bool = Field(
        default=True,
        description="Include difference from lag"
    )
    include_pct_change: bool = Field(
        default=False,
        description="Include percentage change from lag"
    )

    def get_output_feature_names(self) -> List[str]:
        """Get list of output feature names."""
        names = []
        for lag in self.lag_periods:
            names.append(f"{self.feature_name}_lag_{lag}")
            if self.include_diff:
                names.append(f"{self.feature_name}_diff_{lag}")
            if self.include_pct_change:
                names.append(f"{self.feature_name}_pct_change_{lag}")
        return names


class StatisticalFeatures(BaseModel):
    """
    Configuration for statistical feature extraction.

    Attributes:
        feature_names: Features to compute statistics for
        statistics: List of statistics to compute
        window_size: Window size for computation

    Example:
        >>> config = StatisticalFeatures(
        ...     feature_names=["efficiency", "steam_flow"],
        ...     statistics=["mean", "std", "skew", "kurtosis"]
        ... )
    """

    feature_names: List[str] = Field(
        ...,
        description="Features to compute statistics for"
    )
    statistics: List[str] = Field(
        default_factory=lambda: ["mean", "std", "min", "max", "range"],
        description="Statistics to compute"
    )
    window_size: int = Field(
        default=24,
        ge=1,
        description="Window size for computation"
    )
    percentiles: List[int] = Field(
        default_factory=lambda: [25, 50, 75, 90, 95],
        description="Percentiles to compute"
    )


class FeatureInteraction(BaseModel):
    """
    Configuration for cross-feature interactions.

    Attributes:
        feature_a: First feature
        feature_b: Second feature
        operation: Interaction operation

    Example:
        >>> config = FeatureInteraction(
        ...     feature_a="steam_flow",
        ...     feature_b="fuel_rate",
        ...     operation="ratio"
        ... )
    """

    feature_a: str = Field(..., description="First feature name")
    feature_b: str = Field(..., description="Second feature name")
    operation: str = Field(
        default="multiply",
        description="Operation: multiply, divide, ratio, add, subtract"
    )
    output_name: Optional[str] = Field(
        None,
        description="Output feature name"
    )

    def get_output_feature_name(self) -> str:
        """Get output feature name."""
        if self.output_name:
            return self.output_name
        return f"{self.feature_a}_{self.operation}_{self.feature_b}"


class FeaturePipelineConfig(BaseModel):
    """
    Complete configuration for feature pipeline.

    Attributes:
        rolling_aggregations: Rolling aggregation configs
        lag_features: Lag feature configs
        statistical_features: Statistical feature configs
        interactions: Feature interaction configs
        scaling_method: Feature scaling method
        drop_nulls: Whether to drop null values
    """

    rolling_aggregations: List[RollingAggregation] = Field(
        default_factory=list,
        description="Rolling aggregation configurations"
    )
    lag_features: List[LagFeatureConfig] = Field(
        default_factory=list,
        description="Lag feature configurations"
    )
    statistical_features: Optional[StatisticalFeatures] = Field(
        None,
        description="Statistical feature configuration"
    )
    interactions: List[FeatureInteraction] = Field(
        default_factory=list,
        description="Feature interaction configurations"
    )
    scaling_method: ScalingMethod = Field(
        default=ScalingMethod.STANDARD,
        description="Feature scaling method"
    )
    drop_nulls: bool = Field(
        default=False,
        description="Drop rows with null values"
    )
    fill_method: str = Field(
        default="forward",
        description="Method to fill nulls: forward, backward, zero, mean"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


# =============================================================================
# FEATURE TRANSFORMATION RESULT
# =============================================================================

@dataclass
class TransformationResult:
    """Result from feature transformation."""

    features: Dict[str, List[float]]
    feature_names: List[str]
    num_samples: int
    transformation_chain: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": self.features,
            "feature_names": self.feature_names,
            "num_samples": self.num_samples,
            "transformation_chain": self.transformation_chain,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_mean(values: List[float]) -> float:
    """Calculate mean of values, handling empty lists."""
    if not values:
        return float('nan')
    return sum(values) / len(values)


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of values."""
    if len(values) < 2:
        return float('nan')
    mean = _calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of values."""
    if not values:
        return float('nan')
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _calculate_median(values: List[float]) -> float:
    """Calculate median of values."""
    return _calculate_percentile(values, 50)


def _safe_divide(a: float, b: float, default: float = float('nan')) -> float:
    """Safely divide two numbers."""
    if b == 0 or math.isnan(b):
        return default
    return a / b


def _safe_log(value: float, default: float = float('nan')) -> float:
    """Safely compute natural log."""
    if value <= 0 or math.isnan(value):
        return default
    return math.log(value)


# =============================================================================
# FEATURE PIPELINE CLASS
# =============================================================================

class FeaturePipeline:
    """
    Feature Engineering Pipeline for Process Heat Agents.

    This class provides comprehensive feature engineering capabilities
    including rolling aggregations, lag features, statistical features,
    and cross-feature interactions.

    Key Features:
    - Rolling aggregations with configurable windows (1h, 24h, 7d)
    - Lag features for time series analysis
    - Statistical features (mean, std, min, max, percentiles)
    - Cross-feature interactions (ratios, products, differences)
    - Feature normalization and scaling
    - SHA-256 provenance tracking

    Attributes:
        config: Pipeline configuration
        _scaling_params: Learned scaling parameters

    Example:
        >>> pipeline = FeaturePipeline()
        >>>
        >>> # Configure rolling aggregations
        >>> pipeline.add_rolling_aggregation(
        ...     feature_name="efficiency",
        ...     window="24h",
        ...     aggregations=["mean", "std"]
        ... )
        >>>
        >>> # Transform features
        >>> result = pipeline.transform(raw_features)
    """

    def __init__(
        self,
        config: Optional[FeaturePipelineConfig] = None
    ):
        """
        Initialize the FeaturePipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or FeaturePipelineConfig()
        self._scaling_params: Dict[str, Dict[str, float]] = {}
        self._is_fitted = False
        self._provenance_enabled = self.config.enable_provenance

        logger.info("FeaturePipeline initialized")

    def add_rolling_aggregation(
        self,
        feature_name: str,
        window: str,
        aggregations: Optional[List[str]] = None,
        min_periods: int = 1
    ) -> "FeaturePipeline":
        """
        Add a rolling aggregation configuration.

        Args:
            feature_name: Source feature to aggregate
            window: Window size (e.g., "1h", "24h", "7d")
            aggregations: List of aggregation functions
            min_periods: Minimum observations required

        Returns:
            Self for chaining

        Example:
            >>> pipeline.add_rolling_aggregation(
            ...     feature_name="efficiency",
            ...     window="24h",
            ...     aggregations=["mean", "std", "min", "max"]
            ... )
        """
        config = RollingAggregation(
            feature_name=feature_name,
            window=window,
            aggregations=aggregations or ["mean", "std"],
            min_periods=min_periods
        )
        self.config.rolling_aggregations.append(config)
        return self

    def add_lag_features(
        self,
        feature_name: str,
        lag_periods: Optional[List[int]] = None,
        include_diff: bool = True,
        include_pct_change: bool = False
    ) -> "FeaturePipeline":
        """
        Add lag feature configuration.

        Args:
            feature_name: Source feature name
            lag_periods: List of lag periods
            include_diff: Include difference features
            include_pct_change: Include percentage change features

        Returns:
            Self for chaining

        Example:
            >>> pipeline.add_lag_features(
            ...     feature_name="steam_flow",
            ...     lag_periods=[1, 6, 12, 24],
            ...     include_diff=True
            ... )
        """
        config = LagFeatureConfig(
            feature_name=feature_name,
            lag_periods=lag_periods or [1, 6, 12, 24],
            include_diff=include_diff,
            include_pct_change=include_pct_change
        )
        self.config.lag_features.append(config)
        return self

    def add_interaction(
        self,
        feature_a: str,
        feature_b: str,
        operation: str = "ratio",
        output_name: Optional[str] = None
    ) -> "FeaturePipeline":
        """
        Add cross-feature interaction.

        Args:
            feature_a: First feature name
            feature_b: Second feature name
            operation: Operation type (multiply, divide, ratio, add, subtract)
            output_name: Output feature name

        Returns:
            Self for chaining

        Example:
            >>> pipeline.add_interaction(
            ...     feature_a="steam_flow",
            ...     feature_b="fuel_rate",
            ...     operation="ratio",
            ...     output_name="steam_fuel_ratio"
            ... )
        """
        config = FeatureInteraction(
            feature_a=feature_a,
            feature_b=feature_b,
            operation=operation,
            output_name=output_name
        )
        self.config.interactions.append(config)
        return self

    def _get_window_size_hours(self, window: str) -> int:
        """Convert window string to hours."""
        window_mapping = {
            "1h": 1,
            "4h": 4,
            "8h": 8,
            "12h": 12,
            "24h": 24,
            "7d": 168,
            "30d": 720,
            "90d": 2160
        }
        return window_mapping.get(window, 24)

    def _apply_aggregation(
        self,
        values: List[float],
        aggregation: str
    ) -> float:
        """Apply aggregation function to values."""
        if not values:
            return float('nan')

        # Filter out NaN values
        valid_values = [v for v in values if not math.isnan(v)]
        if not valid_values:
            return float('nan')

        aggregation_funcs = {
            "mean": lambda v: _calculate_mean(v),
            "sum": lambda v: sum(v),
            "min": lambda v: min(v),
            "max": lambda v: max(v),
            "std": lambda v: _calculate_std(v),
            "var": lambda v: _calculate_std(v) ** 2 if len(v) >= 2 else float('nan'),
            "median": lambda v: _calculate_median(v),
            "count": lambda v: float(len(v)),
            "first": lambda v: v[0] if v else float('nan'),
            "last": lambda v: v[-1] if v else float('nan'),
            "range": lambda v: max(v) - min(v) if v else float('nan'),
            "percentile_25": lambda v: _calculate_percentile(v, 25),
            "percentile_75": lambda v: _calculate_percentile(v, 75),
            "percentile_90": lambda v: _calculate_percentile(v, 90),
            "percentile_95": lambda v: _calculate_percentile(v, 95),
            "percentile_99": lambda v: _calculate_percentile(v, 99),
        }

        func = aggregation_funcs.get(aggregation)
        if func:
            return func(valid_values)

        logger.warning(f"Unknown aggregation: {aggregation}")
        return float('nan')

    def _compute_rolling_aggregations(
        self,
        features: Dict[str, List[float]],
        num_samples: int
    ) -> Dict[str, List[float]]:
        """
        Compute rolling aggregations for configured features.

        Args:
            features: Input feature dictionary
            num_samples: Number of samples

        Returns:
            Dictionary of computed rolling features
        """
        result: Dict[str, List[float]] = {}

        for config in self.config.rolling_aggregations:
            source_values = features.get(config.feature_name, [])
            if not source_values:
                logger.warning(f"Feature {config.feature_name} not found for rolling aggregation")
                continue

            window_size = self._get_window_size_hours(config.window)

            # For each aggregation type
            for agg in config.aggregations:
                output_name = f"{config.feature_name}_{config.window}_{agg}"
                output_values = []

                for i in range(num_samples):
                    # Get window of values (looking back)
                    start_idx = max(0, i - window_size + 1)
                    window_values = source_values[start_idx:i + 1]

                    if len(window_values) >= config.min_periods:
                        agg_value = self._apply_aggregation(window_values, agg)
                    else:
                        agg_value = float('nan')

                    output_values.append(agg_value)

                result[output_name] = output_values

        return result

    def _compute_lag_features(
        self,
        features: Dict[str, List[float]],
        num_samples: int
    ) -> Dict[str, List[float]]:
        """
        Compute lag features for configured features.

        Args:
            features: Input feature dictionary
            num_samples: Number of samples

        Returns:
            Dictionary of computed lag features
        """
        result: Dict[str, List[float]] = {}

        for config in self.config.lag_features:
            source_values = features.get(config.feature_name, [])
            if not source_values:
                logger.warning(f"Feature {config.feature_name} not found for lag features")
                continue

            for lag in config.lag_periods:
                # Lag feature
                lag_name = f"{config.feature_name}_lag_{lag}"
                lag_values = []

                for i in range(num_samples):
                    if i >= lag:
                        lag_values.append(source_values[i - lag])
                    else:
                        lag_values.append(float('nan'))

                result[lag_name] = lag_values

                # Difference feature
                if config.include_diff:
                    diff_name = f"{config.feature_name}_diff_{lag}"
                    diff_values = []

                    for i in range(num_samples):
                        if i >= lag:
                            current = source_values[i]
                            lagged = source_values[i - lag]
                            if not math.isnan(current) and not math.isnan(lagged):
                                diff_values.append(current - lagged)
                            else:
                                diff_values.append(float('nan'))
                        else:
                            diff_values.append(float('nan'))

                    result[diff_name] = diff_values

                # Percentage change feature
                if config.include_pct_change:
                    pct_name = f"{config.feature_name}_pct_change_{lag}"
                    pct_values = []

                    for i in range(num_samples):
                        if i >= lag:
                            current = source_values[i]
                            lagged = source_values[i - lag]
                            if not math.isnan(current) and not math.isnan(lagged) and lagged != 0:
                                pct_values.append((current - lagged) / lagged * 100)
                            else:
                                pct_values.append(float('nan'))
                        else:
                            pct_values.append(float('nan'))

                    result[pct_name] = pct_values

        return result

    def _compute_statistical_features(
        self,
        features: Dict[str, List[float]],
        num_samples: int
    ) -> Dict[str, List[float]]:
        """
        Compute statistical features for configured features.

        Args:
            features: Input feature dictionary
            num_samples: Number of samples

        Returns:
            Dictionary of computed statistical features
        """
        result: Dict[str, List[float]] = {}

        config = self.config.statistical_features
        if not config:
            return result

        window_size = config.window_size

        for feature_name in config.feature_names:
            source_values = features.get(feature_name, [])
            if not source_values:
                continue

            for stat in config.statistics:
                output_name = f"{feature_name}_stat_{stat}"
                output_values = []

                for i in range(num_samples):
                    start_idx = max(0, i - window_size + 1)
                    window_values = source_values[start_idx:i + 1]

                    stat_value = self._apply_aggregation(window_values, stat)
                    output_values.append(stat_value)

                result[output_name] = output_values

            # Compute percentiles
            for pct in config.percentiles:
                output_name = f"{feature_name}_stat_p{pct}"
                output_values = []

                for i in range(num_samples):
                    start_idx = max(0, i - window_size + 1)
                    window_values = source_values[start_idx:i + 1]

                    if window_values:
                        pct_value = _calculate_percentile(window_values, pct)
                    else:
                        pct_value = float('nan')
                    output_values.append(pct_value)

                result[output_name] = output_values

        return result

    def _compute_interactions(
        self,
        features: Dict[str, List[float]],
        num_samples: int
    ) -> Dict[str, List[float]]:
        """
        Compute cross-feature interactions.

        Args:
            features: Input feature dictionary
            num_samples: Number of samples

        Returns:
            Dictionary of computed interaction features
        """
        result: Dict[str, List[float]] = {}

        for config in self.config.interactions:
            values_a = features.get(config.feature_a, [])
            values_b = features.get(config.feature_b, [])

            if not values_a or not values_b:
                logger.warning(
                    f"Missing features for interaction: {config.feature_a} or {config.feature_b}"
                )
                continue

            output_name = config.get_output_feature_name()
            output_values = []

            for i in range(num_samples):
                a = values_a[i] if i < len(values_a) else float('nan')
                b = values_b[i] if i < len(values_b) else float('nan')

                if math.isnan(a) or math.isnan(b):
                    output_values.append(float('nan'))
                    continue

                if config.operation == "multiply":
                    output_values.append(a * b)
                elif config.operation == "divide" or config.operation == "ratio":
                    output_values.append(_safe_divide(a, b))
                elif config.operation == "add":
                    output_values.append(a + b)
                elif config.operation == "subtract":
                    output_values.append(a - b)
                else:
                    logger.warning(f"Unknown operation: {config.operation}")
                    output_values.append(float('nan'))

            result[output_name] = output_values

        return result

    def _apply_scaling(
        self,
        features: Dict[str, List[float]],
        fit: bool = True
    ) -> Dict[str, List[float]]:
        """
        Apply feature scaling.

        Args:
            features: Features to scale
            fit: Whether to fit scaling parameters

        Returns:
            Scaled features
        """
        if self.config.scaling_method == ScalingMethod.NONE:
            return features

        result: Dict[str, List[float]] = {}

        for name, values in features.items():
            valid_values = [v for v in values if not math.isnan(v)]

            if not valid_values:
                result[name] = values
                continue

            if fit:
                # Compute scaling parameters
                if self.config.scaling_method == ScalingMethod.STANDARD:
                    mean = _calculate_mean(valid_values)
                    std = _calculate_std(valid_values)
                    self._scaling_params[name] = {"mean": mean, "std": std}

                elif self.config.scaling_method == ScalingMethod.MINMAX:
                    min_val = min(valid_values)
                    max_val = max(valid_values)
                    self._scaling_params[name] = {"min": min_val, "max": max_val}

                elif self.config.scaling_method == ScalingMethod.ROBUST:
                    median = _calculate_median(valid_values)
                    q25 = _calculate_percentile(valid_values, 25)
                    q75 = _calculate_percentile(valid_values, 75)
                    iqr = q75 - q25
                    self._scaling_params[name] = {"median": median, "iqr": iqr}

            # Apply scaling
            params = self._scaling_params.get(name, {})
            scaled_values = []

            for v in values:
                if math.isnan(v):
                    scaled_values.append(float('nan'))
                    continue

                if self.config.scaling_method == ScalingMethod.STANDARD:
                    mean = params.get("mean", 0)
                    std = params.get("std", 1)
                    scaled_values.append(_safe_divide(v - mean, std, v))

                elif self.config.scaling_method == ScalingMethod.MINMAX:
                    min_val = params.get("min", 0)
                    max_val = params.get("max", 1)
                    range_val = max_val - min_val
                    scaled_values.append(_safe_divide(v - min_val, range_val, 0))

                elif self.config.scaling_method == ScalingMethod.ROBUST:
                    median = params.get("median", 0)
                    iqr = params.get("iqr", 1)
                    scaled_values.append(_safe_divide(v - median, iqr, v))

                elif self.config.scaling_method == ScalingMethod.LOG:
                    scaled_values.append(_safe_log(v))

                else:
                    scaled_values.append(v)

            result[name] = scaled_values

        return result

    def _fill_nulls(
        self,
        features: Dict[str, List[float]],
        num_samples: int
    ) -> Dict[str, List[float]]:
        """
        Fill null values in features.

        Args:
            features: Features with nulls
            num_samples: Number of samples

        Returns:
            Features with nulls filled
        """
        if self.config.drop_nulls:
            # Drop rows with any nulls (handled separately)
            return features

        result: Dict[str, List[float]] = {}

        for name, values in features.items():
            filled_values = values.copy()

            if self.config.fill_method == "forward":
                last_valid = float('nan')
                for i in range(len(filled_values)):
                    if math.isnan(filled_values[i]):
                        filled_values[i] = last_valid
                    else:
                        last_valid = filled_values[i]

            elif self.config.fill_method == "backward":
                last_valid = float('nan')
                for i in range(len(filled_values) - 1, -1, -1):
                    if math.isnan(filled_values[i]):
                        filled_values[i] = last_valid
                    else:
                        last_valid = filled_values[i]

            elif self.config.fill_method == "zero":
                filled_values = [0.0 if math.isnan(v) else v for v in filled_values]

            elif self.config.fill_method == "mean":
                valid_values = [v for v in values if not math.isnan(v)]
                mean_val = _calculate_mean(valid_values) if valid_values else 0.0
                filled_values = [mean_val if math.isnan(v) else v for v in filled_values]

            result[name] = filled_values

        return result

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash."""
        if not self._provenance_enabled:
            return ""

        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate provenance hash: {e}")
            return ""

    def fit(
        self,
        features: Dict[str, List[float]]
    ) -> "FeaturePipeline":
        """
        Fit scaling parameters on training data.

        Args:
            features: Training features

        Returns:
            Self for chaining
        """
        # Compute scaling parameters
        self._apply_scaling(features, fit=True)
        self._is_fitted = True
        logger.info(f"Pipeline fitted with {len(self._scaling_params)} scaling parameters")
        return self

    def transform(
        self,
        features: Dict[str, List[float]],
        fit: bool = False
    ) -> TransformationResult:
        """
        Transform features using the pipeline.

        Args:
            features: Input features as dictionary of lists
            fit: Whether to fit scaling parameters

        Returns:
            TransformationResult with transformed features

        Example:
            >>> raw_features = {
            ...     "efficiency": [0.85, 0.86, 0.84, 0.87],
            ...     "steam_flow": [1000, 1020, 980, 1050]
            ... }
            >>> result = pipeline.transform(raw_features)
        """
        start_time = datetime.now(timezone.utc)
        transformation_chain = []

        # Determine number of samples
        num_samples = max(len(v) for v in features.values()) if features else 0

        # Start with input features
        result_features = dict(features)

        # Apply rolling aggregations
        if self.config.rolling_aggregations:
            rolling_features = self._compute_rolling_aggregations(features, num_samples)
            result_features.update(rolling_features)
            transformation_chain.append(
                f"rolling_aggregations:{len(self.config.rolling_aggregations)}"
            )

        # Apply lag features
        if self.config.lag_features:
            lag_features = self._compute_lag_features(features, num_samples)
            result_features.update(lag_features)
            transformation_chain.append(
                f"lag_features:{len(self.config.lag_features)}"
            )

        # Apply statistical features
        if self.config.statistical_features:
            stat_features = self._compute_statistical_features(features, num_samples)
            result_features.update(stat_features)
            transformation_chain.append("statistical_features")

        # Apply interactions
        if self.config.interactions:
            interaction_features = self._compute_interactions(result_features, num_samples)
            result_features.update(interaction_features)
            transformation_chain.append(f"interactions:{len(self.config.interactions)}")

        # Fill nulls
        result_features = self._fill_nulls(result_features, num_samples)
        transformation_chain.append(f"fill_nulls:{self.config.fill_method}")

        # Apply scaling
        if self.config.scaling_method != ScalingMethod.NONE:
            result_features = self._apply_scaling(result_features, fit=fit)
            transformation_chain.append(f"scaling:{self.config.scaling_method}")

        # Calculate processing time
        end_time = datetime.now(timezone.utc)
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate provenance hash
        provenance_data = {
            "input_features": list(features.keys()),
            "output_features": list(result_features.keys()),
            "transformation_chain": transformation_chain,
            "timestamp": end_time.isoformat()
        }
        provenance_hash = self._calculate_provenance_hash(provenance_data)

        result = TransformationResult(
            features=result_features,
            feature_names=list(result_features.keys()),
            num_samples=num_samples,
            transformation_chain=transformation_chain,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            metadata={
                "scaling_method": self.config.scaling_method,
                "fill_method": self.config.fill_method,
                "num_input_features": len(features),
                "num_output_features": len(result_features)
            }
        )

        logger.info(
            f"Transformed {len(features)} features to {len(result_features)} features "
            f"in {processing_time_ms:.2f}ms"
        )

        return result

    def fit_transform(
        self,
        features: Dict[str, List[float]]
    ) -> TransformationResult:
        """
        Fit and transform features.

        Args:
            features: Input features

        Returns:
            TransformationResult with transformed features
        """
        return self.transform(features, fit=True)

    def get_feature_names(self) -> List[str]:
        """Get list of all output feature names."""
        names = []

        for config in self.config.rolling_aggregations:
            names.extend(config.get_output_feature_names())

        for config in self.config.lag_features:
            names.extend(config.get_output_feature_names())

        for config in self.config.interactions:
            names.append(config.get_output_feature_name())

        return names

    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration as dictionary."""
        return self.config.dict()

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "FeaturePipeline":
        """
        Create pipeline from configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configured FeaturePipeline
        """
        config = FeaturePipelineConfig(**config_dict)
        return cls(config=config)

    @classmethod
    def create_default_boiler_pipeline(cls) -> "FeaturePipeline":
        """
        Create default pipeline for boiler features.

        Returns:
            Configured FeaturePipeline for boiler analysis
        """
        pipeline = cls()

        # Add rolling aggregations for key metrics
        for feature in ["efficiency", "steam_flow", "fuel_rate"]:
            for window in ["1h", "24h", "7d"]:
                pipeline.add_rolling_aggregation(
                    feature_name=feature,
                    window=window,
                    aggregations=["mean", "std", "min", "max"]
                )

        # Add lag features
        pipeline.add_lag_features(
            feature_name="efficiency",
            lag_periods=[1, 6, 12, 24],
            include_diff=True
        )

        # Add interactions
        pipeline.add_interaction(
            feature_a="steam_flow",
            feature_b="fuel_rate",
            operation="ratio",
            output_name="steam_fuel_ratio"
        )

        return pipeline

    @classmethod
    def create_default_emissions_pipeline(cls) -> "FeaturePipeline":
        """
        Create default pipeline for emissions features.

        Returns:
            Configured FeaturePipeline for emissions analysis
        """
        pipeline = cls()

        # Add rolling aggregations
        for feature in ["co2_rate", "intensity", "total_ghg"]:
            for window in ["24h", "7d", "30d"]:
                pipeline.add_rolling_aggregation(
                    feature_name=feature,
                    window=window,
                    aggregations=["mean", "sum", "max"]
                )

        # Add lag features for trend analysis
        pipeline.add_lag_features(
            feature_name="intensity",
            lag_periods=[24, 168],  # 1 day, 1 week
            include_diff=True,
            include_pct_change=True
        )

        return pipeline

    @classmethod
    def create_default_predictive_pipeline(cls) -> "FeaturePipeline":
        """
        Create default pipeline for predictive maintenance features.

        Returns:
            Configured FeaturePipeline for predictive maintenance
        """
        pipeline = cls()

        # Add rolling aggregations for health indicators
        for feature in ["fouling_index", "failure_probability", "health_index"]:
            pipeline.add_rolling_aggregation(
                feature_name=feature,
                window="24h",
                aggregations=["mean", "std", "max"]
            )
            pipeline.add_rolling_aggregation(
                feature_name=feature,
                window="7d",
                aggregations=["mean", "std", "max", "percentile_90"]
            )

        # Add lag features for trend analysis
        for feature in ["fouling_index", "health_index"]:
            pipeline.add_lag_features(
                feature_name=feature,
                lag_periods=[24, 168],
                include_diff=True
            )

        return pipeline
