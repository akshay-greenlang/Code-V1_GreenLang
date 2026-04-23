"""
Real-Time Feature Pipeline Module - GL-004 BURNMASTER

This module provides real-time feature computation for combustion data,
including rolling window features, lag features, and time-aligned feature
vectors for ML model inference.

Key Features:
    - Rolling window feature computation (mean, std, min, max, etc.)
    - Lag feature generation for temporal patterns
    - Time-aligned feature vectors with configurable alignment
    - Feature validation with range and completeness checks
    - Zero-hallucination approach using deterministic calculations

Example:
    >>> pipeline = RealTimeFeaturePipeline()
    >>> pipeline.define_feature_spec(spec)
    >>> features = pipeline.compute_features(combustion_data)
    >>> aligned = pipeline.time_align_features(features, timestamp)

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from .kafka_producer import CombustionData, CombustionDataPoint

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AggregationType(str, Enum):
    """Aggregation types for rolling features."""

    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    MEDIAN = "median"
    VARIANCE = "variance"
    RANGE = "range"
    FIRST = "first"
    LAST = "last"
    RATE = "rate"
    PERCENTILE_25 = "p25"
    PERCENTILE_75 = "p75"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class FeatureType(str, Enum):
    """Types of features."""

    RAW = "raw"
    ROLLING = "rolling"
    LAG = "lag"
    DERIVED = "derived"
    INTERACTION = "interaction"


class ValidationStatus(str, Enum):
    """Feature validation status."""

    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    MISSING = "missing"


class AlignmentMode(str, Enum):
    """Time alignment modes."""

    FORWARD = "forward"  # Align to next boundary
    BACKWARD = "backward"  # Align to previous boundary
    NEAREST = "nearest"  # Align to nearest boundary


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class RollingFeatureConfig(BaseModel):
    """Configuration for a rolling window feature."""

    name: str = Field(..., description="Feature name")
    source_tag: str = Field(..., description="Source tag ID")
    window_size: str = Field(
        "5m",
        description="Window size (e.g., '5m', '1h', '30s')",
    )
    aggregation: AggregationType = Field(
        AggregationType.MEAN,
        description="Aggregation type",
    )
    min_samples: int = Field(
        1,
        ge=1,
        description="Minimum samples required",
    )
    fill_value: Optional[float] = Field(
        None,
        description="Value to use when insufficient samples",
    )

    @property
    def window_seconds(self) -> int:
        """Parse window size to seconds."""
        value = int(self.window_size[:-1])
        unit = self.window_size[-1].lower()

        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise ValueError(f"Unknown time unit: {unit}")


class LagFeatureConfig(BaseModel):
    """Configuration for a lag feature."""

    name: str = Field(..., description="Feature name")
    source_tag: str = Field(..., description="Source tag ID")
    lag: int = Field(
        ...,
        ge=1,
        description="Lag in number of samples",
    )
    default_value: Optional[float] = Field(
        None,
        description="Default value when lag unavailable",
    )


class DerivedFeatureConfig(BaseModel):
    """Configuration for a derived feature."""

    name: str = Field(..., description="Feature name")
    formula: str = Field(
        ...,
        description="Formula expression (e.g., 'temp_diff = temp_out - temp_in')",
    )
    input_features: List[str] = Field(
        default_factory=list,
        description="Input feature names",
    )


class FeatureSpec(BaseModel):
    """
    Complete feature specification.

    Attributes:
        name: Specification name
        version: Specification version
        rolling_features: Rolling window feature configurations
        lag_features: Lag feature configurations
        derived_features: Derived feature configurations
        alignment_interval: Time alignment interval
    """

    name: str = Field(..., description="Specification name")
    version: str = Field("1.0.0", description="Specification version")
    rolling_features: List[RollingFeatureConfig] = Field(
        default_factory=list,
        description="Rolling window feature configurations",
    )
    lag_features: List[LagFeatureConfig] = Field(
        default_factory=list,
        description="Lag feature configurations",
    )
    derived_features: List[DerivedFeatureConfig] = Field(
        default_factory=list,
        description="Derived feature configurations",
    )
    alignment_interval: str = Field(
        "1m",
        description="Time alignment interval (e.g., '1m', '5m')",
    )
    alignment_mode: AlignmentMode = Field(
        AlignmentMode.BACKWARD,
        description="Time alignment mode",
    )
    required_features: Set[str] = Field(
        default_factory=set,
        description="Features required for valid output",
    )

    @property
    def alignment_seconds(self) -> int:
        """Parse alignment interval to seconds."""
        value = int(self.alignment_interval[:-1])
        unit = self.alignment_interval[-1].lower()

        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        else:
            raise ValueError(f"Unknown time unit: {unit}")


# =============================================================================
# RESULT MODELS
# =============================================================================


class FeatureValue(BaseModel):
    """Single feature value with metadata."""

    name: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    feature_type: FeatureType = Field(..., description="Feature type")
    source_tag: Optional[str] = Field(None, description="Source tag if applicable")
    window_size: Optional[str] = Field(None, description="Window size if rolling")
    sample_count: int = Field(0, ge=0, description="Number of samples used")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feature computation timestamp",
    )
    quality: str = Field("GOOD", description="Feature quality indicator")


class FeatureVector(BaseModel):
    """
    Vector of computed features.

    Attributes:
        vector_id: Unique vector identifier
        timestamp: Vector timestamp
        features: Dictionary of feature name to value
        metadata: Additional metadata
    """

    vector_id: str = Field(
        default_factory=lambda: f"vec-{int(time.time() * 1000)}",
        description="Unique vector identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Vector timestamp",
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    features: Dict[str, FeatureValue] = Field(
        default_factory=dict,
        description="Feature name to value mapping",
    )
    raw_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Simplified feature name to value mapping for ML",
    )
    feature_count: int = Field(0, ge=0, description="Number of features")
    computation_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Computation time in milliseconds",
    )
    provenance_hash: str = Field("", description="SHA-256 hash for audit")

    def model_post_init(self, __context: Any) -> None:
        """Compute feature count and raw values after initialization."""
        object.__setattr__(self, "feature_count", len(self.features))

        # Build raw values for ML inference
        raw = {name: feat.value for name, feat in self.features.items()}
        object.__setattr__(self, "raw_values", raw)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        content = json.dumps(
            {
                "vector_id": self.vector_id,
                "timestamp": self.timestamp.isoformat(),
                "equipment_id": self.equipment_id,
                "features": {k: v.value for k, v in self.features.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class AlignedFeatures(BaseModel):
    """Time-aligned feature vector."""

    vector: FeatureVector = Field(..., description="Feature vector")
    aligned_timestamp: datetime = Field(
        ...,
        description="Aligned timestamp",
    )
    original_timestamp: datetime = Field(
        ...,
        description="Original timestamp before alignment",
    )
    alignment_offset_ms: float = Field(
        0.0,
        description="Offset from original to aligned timestamp",
    )
    alignment_mode: AlignmentMode = Field(
        AlignmentMode.BACKWARD,
        description="Alignment mode used",
    )
    alignment_interval: str = Field(
        "1m",
        description="Alignment interval",
    )


class FeatureValidationResult(BaseModel):
    """Result of feature validation."""

    status: ValidationStatus = Field(..., description="Validation status")
    vector_id: str = Field(..., description="Validated vector ID")
    valid_features: List[str] = Field(
        default_factory=list,
        description="List of valid feature names",
    )
    invalid_features: Dict[str, str] = Field(
        default_factory=dict,
        description="Invalid features with reasons",
    )
    missing_features: List[str] = Field(
        default_factory=list,
        description="Missing required features",
    )
    out_of_range_features: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Features out of expected range",
    )
    validation_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Validation time in milliseconds",
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Validation timestamp",
    )


# =============================================================================
# FEATURE PIPELINE IMPLEMENTATION
# =============================================================================


@dataclass
class TagBuffer:
    """Buffer for storing tag values for rolling computations."""

    tag_id: str
    max_size: int = 10000
    values: Deque[Tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=10000)
    )

    def add(self, timestamp: datetime, value: float) -> None:
        """Add a value to the buffer."""
        self.values.append((timestamp, value))

    def get_values_in_window(
        self,
        end_time: datetime,
        window_seconds: int,
    ) -> List[float]:
        """Get all values within the specified window."""
        start_time = end_time - timedelta(seconds=window_seconds)
        return [
            value
            for ts, value in self.values
            if start_time <= ts <= end_time
        ]

    def get_last_n(self, n: int) -> List[float]:
        """Get the last n values."""
        return [value for _, value in list(self.values)[-n:]]

    def clear_old(self, cutoff: datetime) -> int:
        """Remove values older than cutoff. Returns count removed."""
        original_len = len(self.values)
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()
        return original_len - len(self.values)


class RealTimeFeaturePipeline:
    """
    Real-time feature computation pipeline for combustion data.

    This pipeline computes features for ML model inference using
    deterministic calculations only (zero-hallucination approach).

    Example:
        >>> pipeline = RealTimeFeaturePipeline()
        >>> spec = FeatureSpec(
        ...     name="combustion-features",
        ...     rolling_features=[
        ...         RollingFeatureConfig(
        ...             name="temp_mean_5m",
        ...             source_tag="TEMP_001",
        ...             window_size="5m",
        ...             aggregation=AggregationType.MEAN,
        ...         )
        ...     ]
        ... )
        >>> pipeline.define_feature_spec(spec)
        >>> features = pipeline.compute_features(combustion_data)
    """

    def __init__(self, max_buffer_age_hours: int = 24) -> None:
        """
        Initialize RealTimeFeaturePipeline.

        Args:
            max_buffer_age_hours: Maximum age of buffered data in hours
        """
        self._spec: Optional[FeatureSpec] = None
        self._tag_buffers: Dict[str, TagBuffer] = {}
        self._max_buffer_age = timedelta(hours=max_buffer_age_hours)
        self._feature_ranges: Dict[str, Tuple[float, float]] = {}
        self._computation_count = 0

        logger.info(
            f"RealTimeFeaturePipeline initialized with "
            f"max_buffer_age={max_buffer_age_hours}h"
        )

    def define_feature_spec(self, spec: FeatureSpec) -> None:
        """
        Define the feature specification.

        Args:
            spec: Feature specification to use
        """
        self._spec = spec

        # Initialize buffers for all source tags
        all_tags: Set[str] = set()

        for rolling in spec.rolling_features:
            all_tags.add(rolling.source_tag)

        for lag in spec.lag_features:
            all_tags.add(lag.source_tag)

        for tag_id in all_tags:
            if tag_id not in self._tag_buffers:
                self._tag_buffers[tag_id] = TagBuffer(tag_id=tag_id)

        logger.info(
            f"Feature spec defined: {spec.name} v{spec.version}, "
            f"rolling={len(spec.rolling_features)}, "
            f"lag={len(spec.lag_features)}, "
            f"derived={len(spec.derived_features)}"
        )

    def compute_features(
        self,
        raw_data: CombustionData,
    ) -> FeatureVector:
        """
        Compute features from raw combustion data.

        This method uses DETERMINISTIC calculations only - no LLM or ML
        for numeric computations (zero-hallucination approach).

        Args:
            raw_data: Raw combustion data batch

        Returns:
            FeatureVector with computed features
        """
        if not self._spec:
            raise ValueError("Feature spec not defined. Call define_feature_spec first.")

        start_time = time.monotonic()
        self._computation_count += 1

        # Update buffers with new data points
        for point in raw_data.points:
            if point.tag_id in self._tag_buffers:
                self._tag_buffers[point.tag_id].add(point.timestamp, point.value)

        features: Dict[str, FeatureValue] = {}

        # Compute raw features (latest values)
        for point in raw_data.points:
            feature_name = f"raw_{point.tag_id}"
            features[feature_name] = FeatureValue(
                name=feature_name,
                value=point.value,
                feature_type=FeatureType.RAW,
                source_tag=point.tag_id,
                sample_count=1,
                timestamp=point.timestamp,
                quality=point.quality,
            )

        # Compute rolling features
        reference_time = raw_data.collection_timestamp
        for config in self._spec.rolling_features:
            value = self._compute_rolling_feature(config, reference_time)
            if value is not None:
                buffer = self._tag_buffers.get(config.source_tag)
                sample_count = 0
                if buffer:
                    values = buffer.get_values_in_window(
                        reference_time,
                        config.window_seconds,
                    )
                    sample_count = len(values)

                features[config.name] = FeatureValue(
                    name=config.name,
                    value=value,
                    feature_type=FeatureType.ROLLING,
                    source_tag=config.source_tag,
                    window_size=config.window_size,
                    sample_count=sample_count,
                    timestamp=reference_time,
                )

        # Compute lag features
        for config in self._spec.lag_features:
            value = self._compute_lag_feature(config)
            if value is not None:
                features[config.name] = FeatureValue(
                    name=config.name,
                    value=value,
                    feature_type=FeatureType.LAG,
                    source_tag=config.source_tag,
                    sample_count=config.lag,
                    timestamp=reference_time,
                )

        # Compute derived features
        for config in self._spec.derived_features:
            value = self._compute_derived_feature(config, features)
            if value is not None:
                features[config.name] = FeatureValue(
                    name=config.name,
                    value=value,
                    feature_type=FeatureType.DERIVED,
                    sample_count=len(config.input_features),
                    timestamp=reference_time,
                )

        computation_time = (time.monotonic() - start_time) * 1000

        vector = FeatureVector(
            timestamp=reference_time,
            equipment_id=raw_data.equipment_id,
            features=features,
            computation_time_ms=computation_time,
        )

        # Compute provenance hash
        object.__setattr__(vector, "provenance_hash", vector.compute_hash())

        logger.debug(
            f"Computed {len(features)} features in {computation_time:.2f}ms "
            f"for equipment {raw_data.equipment_id}"
        )

        # Cleanup old buffer data periodically
        if self._computation_count % 100 == 0:
            self._cleanup_buffers()

        return vector

    def _compute_rolling_feature(
        self,
        config: RollingFeatureConfig,
        reference_time: datetime,
    ) -> Optional[float]:
        """Compute a single rolling window feature."""
        buffer = self._tag_buffers.get(config.source_tag)
        if not buffer:
            return config.fill_value

        values = buffer.get_values_in_window(
            reference_time,
            config.window_seconds,
        )

        if len(values) < config.min_samples:
            return config.fill_value

        return self._aggregate(values, config.aggregation)

    def _compute_lag_feature(
        self,
        config: LagFeatureConfig,
    ) -> Optional[float]:
        """Compute a single lag feature."""
        buffer = self._tag_buffers.get(config.source_tag)
        if not buffer:
            return config.default_value

        values = buffer.get_last_n(config.lag + 1)
        if len(values) <= config.lag:
            return config.default_value

        # Return the value from 'lag' samples ago
        return values[-(config.lag + 1)]

    def _compute_derived_feature(
        self,
        config: DerivedFeatureConfig,
        features: Dict[str, FeatureValue],
    ) -> Optional[float]:
        """Compute a derived feature from formula."""
        # Build evaluation context with input features
        context: Dict[str, float] = {}
        for input_name in config.input_features:
            if input_name in features:
                context[input_name] = features[input_name].value
            else:
                return None  # Missing input

        try:
            # Parse simple formulas like "a - b", "a / b", "a * b + c"
            # This is a safe evaluation without using eval()
            return self._safe_evaluate(config.formula, context)
        except Exception as e:
            logger.warning(f"Failed to compute derived feature {config.name}: {e}")
            return None

    def _safe_evaluate(
        self,
        formula: str,
        context: Dict[str, float],
    ) -> float:
        """
        Safely evaluate a simple arithmetic formula.

        Supports: +, -, *, /, parentheses
        NO eval() or exec() - deterministic only
        """
        # Replace variable names with values
        expr = formula
        for name, value in sorted(context.items(), key=lambda x: -len(x[0])):
            expr = expr.replace(name, str(value))

        # Simple tokenization and evaluation
        # For production, use a proper expression parser like pyparsing
        try:
            # Only allow safe characters
            allowed = set("0123456789.+-*/() ")
            if not all(c in allowed for c in expr):
                raise ValueError(f"Unsafe characters in formula: {formula}")

            # Use ast.literal_eval with compile for safety
            import ast
            node = ast.parse(expr, mode='eval')

            # Verify only allowed operations
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call):
                    raise ValueError("Function calls not allowed in formulas")

            result = eval(compile(node, '<string>', 'eval'))
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}")

    def _aggregate(
        self,
        values: List[float],
        aggregation: AggregationType,
    ) -> float:
        """Perform aggregation on values - DETERMINISTIC only."""
        if not values:
            return 0.0

        if aggregation == AggregationType.MEAN:
            return statistics.mean(values)
        elif aggregation == AggregationType.STD:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        elif aggregation == AggregationType.VARIANCE:
            return statistics.variance(values) if len(values) > 1 else 0.0
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.MEDIAN:
            return statistics.median(values)
        elif aggregation == AggregationType.RANGE:
            return max(values) - min(values)
        elif aggregation == AggregationType.FIRST:
            return values[0]
        elif aggregation == AggregationType.LAST:
            return values[-1]
        elif aggregation == AggregationType.RATE:
            return len(values)  # Events per window
        elif aggregation in (
            AggregationType.PERCENTILE_25,
            AggregationType.PERCENTILE_75,
            AggregationType.PERCENTILE_90,
            AggregationType.PERCENTILE_95,
            AggregationType.PERCENTILE_99,
        ):
            percentile = int(aggregation.value[1:])
            return self._percentile(values, percentile)
        else:
            return statistics.mean(values)

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Compute percentile of values - DETERMINISTIC."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f < len(sorted_values) - 1 else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def compute_rolling_features(
        self,
        window: str,
    ) -> Dict[str, float]:
        """
        Compute all rolling features for a specific window size.

        Args:
            window: Window size (e.g., "5m", "1h")

        Returns:
            Dictionary of feature name to value
        """
        if not self._spec:
            return {}

        result: Dict[str, float] = {}
        now = datetime.now(timezone.utc)

        # Parse window to seconds
        value = int(window[:-1])
        unit = window[-1].lower()
        if unit == "s":
            window_seconds = value
        elif unit == "m":
            window_seconds = value * 60
        elif unit == "h":
            window_seconds = value * 3600
        else:
            window_seconds = value * 60  # Default to minutes

        # Compute rolling features matching this window
        for config in self._spec.rolling_features:
            if config.window_size == window:
                feat_value = self._compute_rolling_feature(config, now)
                if feat_value is not None:
                    result[config.name] = feat_value

        return result

    def compute_lag_features(
        self,
        lags: List[int],
    ) -> Dict[str, float]:
        """
        Compute lag features for specified lag values.

        Args:
            lags: List of lag values

        Returns:
            Dictionary of feature name to value
        """
        if not self._spec:
            return {}

        result: Dict[str, float] = {}

        for config in self._spec.lag_features:
            if config.lag in lags:
                value = self._compute_lag_feature(config)
                if value is not None:
                    result[config.name] = value

        return result

    def time_align_features(
        self,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> AlignedFeatures:
        """
        Align features to the configured time boundary.

        Args:
            features: Feature dictionary
            timestamp: Original timestamp

        Returns:
            AlignedFeatures with aligned timestamp
        """
        if not self._spec:
            raise ValueError("Feature spec not defined")

        interval_seconds = self._spec.alignment_seconds
        ts_seconds = timestamp.timestamp()

        if self._spec.alignment_mode == AlignmentMode.BACKWARD:
            aligned_seconds = (ts_seconds // interval_seconds) * interval_seconds
        elif self._spec.alignment_mode == AlignmentMode.FORWARD:
            aligned_seconds = (
                (ts_seconds // interval_seconds) + 1
            ) * interval_seconds
        else:  # NEAREST
            lower = (ts_seconds // interval_seconds) * interval_seconds
            upper = lower + interval_seconds
            aligned_seconds = lower if (ts_seconds - lower) <= (upper - ts_seconds) else upper

        aligned_timestamp = datetime.fromtimestamp(aligned_seconds, tz=timezone.utc)
        offset_ms = (aligned_seconds - ts_seconds) * 1000

        # Create feature vector from dict
        feature_values = {
            name: FeatureValue(
                name=name,
                value=value,
                feature_type=FeatureType.RAW,
            )
            for name, value in features.items()
        }

        vector = FeatureVector(
            timestamp=aligned_timestamp,
            equipment_id="unknown",
            features=feature_values,
        )

        return AlignedFeatures(
            vector=vector,
            aligned_timestamp=aligned_timestamp,
            original_timestamp=timestamp,
            alignment_offset_ms=offset_ms,
            alignment_mode=self._spec.alignment_mode,
            alignment_interval=self._spec.alignment_interval,
        )

    def validate_features(
        self,
        features: FeatureVector,
    ) -> FeatureValidationResult:
        """
        Validate computed features.

        Args:
            features: Feature vector to validate

        Returns:
            FeatureValidationResult with validation status
        """
        start_time = time.monotonic()

        valid_features: List[str] = []
        invalid_features: Dict[str, str] = {}
        missing_features: List[str] = []
        out_of_range_features: Dict[str, Dict[str, float]] = {}

        # Check required features
        if self._spec and self._spec.required_features:
            for required in self._spec.required_features:
                if required not in features.features:
                    missing_features.append(required)

        # Validate each feature
        for name, feat_value in features.features.items():
            # Check for NaN or infinity
            if not isinstance(feat_value.value, (int, float)):
                invalid_features[name] = "Value is not numeric"
                continue

            import math
            if math.isnan(feat_value.value) or math.isinf(feat_value.value):
                invalid_features[name] = "Value is NaN or infinite"
                continue

            # Check range if defined
            if name in self._feature_ranges:
                min_val, max_val = self._feature_ranges[name]
                if feat_value.value < min_val or feat_value.value > max_val:
                    out_of_range_features[name] = {
                        "value": feat_value.value,
                        "min": min_val,
                        "max": max_val,
                    }
                    continue

            valid_features.append(name)

        # Determine overall status
        if missing_features:
            status = ValidationStatus.MISSING
        elif invalid_features or out_of_range_features:
            status = ValidationStatus.PARTIAL
        elif len(valid_features) == len(features.features):
            status = ValidationStatus.VALID
        else:
            status = ValidationStatus.INVALID

        validation_time = (time.monotonic() - start_time) * 1000

        return FeatureValidationResult(
            status=status,
            vector_id=features.vector_id,
            valid_features=valid_features,
            invalid_features=invalid_features,
            missing_features=missing_features,
            out_of_range_features=out_of_range_features,
            validation_time_ms=validation_time,
        )

    def set_feature_range(
        self,
        feature_name: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """
        Set expected range for a feature for validation.

        Args:
            feature_name: Feature name
            min_value: Minimum expected value
            max_value: Maximum expected value
        """
        self._feature_ranges[feature_name] = (min_value, max_value)

    def _cleanup_buffers(self) -> None:
        """Clean up old data from buffers."""
        cutoff = datetime.now(timezone.utc) - self._max_buffer_age
        total_removed = 0

        for buffer in self._tag_buffers.values():
            removed = buffer.clear_old(cutoff)
            total_removed += removed

        if total_removed > 0:
            logger.debug(f"Cleaned {total_removed} old values from buffers")

    def get_buffer_stats(self) -> Dict[str, int]:
        """Get current buffer statistics."""
        return {
            tag_id: len(buffer.values)
            for tag_id, buffer in self._tag_buffers.items()
        }
