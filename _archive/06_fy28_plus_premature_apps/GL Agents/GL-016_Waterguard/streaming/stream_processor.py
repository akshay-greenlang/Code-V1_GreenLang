"""
GL-016 Waterguard Stream Processors

Data cleaning and feature engineering processors for real-time boiler water
chemistry data streams. Implements normalization, interpolation, quality
scoring, and derived feature computation.
"""

from __future__ import annotations

import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from streaming.kafka_schemas import (
    CleanedChemistryMessage,
    CleanedReading,
    DerivedFeature,
    FeatureMessage,
    QualityCode,
    RawChemistryMessage,
    SensorReading,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Flag System
# =============================================================================

class QualityFlag(Enum):
    """Data quality flags for processed readings."""
    GOOD = auto()
    INTERPOLATED = auto()
    EXTRAPOLATED = auto()
    CLAMPED_HIGH = auto()
    CLAMPED_LOW = auto()
    STALE = auto()
    BAD_SOURCE = auto()
    OUT_OF_RANGE = auto()
    SPIKE_REMOVED = auto()
    RATE_LIMITED = auto()


@dataclass
class QualityAssessment:
    """Quality assessment result for a reading."""
    flags: List[QualityFlag] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0
    notes: List[str] = field(default_factory=list)

    def add_flag(self, flag: QualityFlag, note: str = "", penalty: float = 0.1) -> None:
        """Add a quality flag with optional penalty."""
        if flag not in self.flags:
            self.flags.append(flag)
            self.score = max(0.0, self.score - penalty)
            if note:
                self.notes.append(note)


# =============================================================================
# Tag Configuration
# =============================================================================

@dataclass
class TagConfig:
    """Configuration for a sensor tag."""
    tag_id: str
    name: str
    engineering_units: str
    min_value: float
    max_value: float
    normal_min: float
    normal_max: float
    rate_of_change_limit: float  # Max change per second
    deadband: float  # Minimum change to register
    interpolation_max_gap_seconds: int = 300  # Max gap for interpolation
    stale_seconds: int = 60  # When to mark as stale


# Default tag configurations for boiler chemistry
DEFAULT_TAG_CONFIGS: Dict[str, TagConfig] = {
    "phosphate": TagConfig(
        tag_id="AI_PO4_001",
        name="Phosphate",
        engineering_units="ppm",
        min_value=0.0,
        max_value=50.0,
        normal_min=2.0,
        normal_max=10.0,
        rate_of_change_limit=2.0,
        deadband=0.1,
    ),
    "conductivity": TagConfig(
        tag_id="AI_COND_001",
        name="Conductivity",
        engineering_units="uS/cm",
        min_value=0.0,
        max_value=10000.0,
        normal_min=100.0,
        normal_max=3000.0,
        rate_of_change_limit=100.0,
        deadband=5.0,
    ),
    "ph": TagConfig(
        tag_id="AI_PH_001",
        name="pH",
        engineering_units="pH",
        min_value=0.0,
        max_value=14.0,
        normal_min=8.0,
        normal_max=11.0,
        rate_of_change_limit=0.5,
        deadband=0.05,
    ),
    "dissolved_oxygen": TagConfig(
        tag_id="AI_DO_001",
        name="Dissolved Oxygen",
        engineering_units="ppb",
        min_value=0.0,
        max_value=1000.0,
        normal_min=0.0,
        normal_max=10.0,
        rate_of_change_limit=5.0,
        deadband=0.5,
    ),
    "silica": TagConfig(
        tag_id="AI_SIO2_001",
        name="Silica",
        engineering_units="ppm",
        min_value=0.0,
        max_value=100.0,
        normal_min=0.0,
        normal_max=5.0,
        rate_of_change_limit=1.0,
        deadband=0.1,
    ),
    "sodium": TagConfig(
        tag_id="AI_NA_001",
        name="Sodium",
        engineering_units="ppb",
        min_value=0.0,
        max_value=10000.0,
        normal_min=0.0,
        normal_max=100.0,
        rate_of_change_limit=10.0,
        deadband=1.0,
    ),
}


# =============================================================================
# Interpolation Strategies
# =============================================================================

class InterpolationMethod(str, Enum):
    """Interpolation methods for missing data."""
    LINEAR = "linear"
    LAST_VALUE = "last_value"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    NONE = "none"


class Interpolator:
    """Interpolation utilities for missing sensor data."""

    @staticmethod
    def linear(
        timestamp: datetime,
        prev_value: float,
        prev_time: datetime,
        next_value: float,
        next_time: datetime,
    ) -> float:
        """Linear interpolation between two points."""
        total_seconds = (next_time - prev_time).total_seconds()
        if total_seconds == 0:
            return prev_value
        elapsed = (timestamp - prev_time).total_seconds()
        ratio = elapsed / total_seconds
        return prev_value + ratio * (next_value - prev_value)

    @staticmethod
    def weighted_average(
        values: List[Tuple[float, datetime]],
        target_time: datetime,
        decay_seconds: float = 60.0,
    ) -> float:
        """Weighted average with exponential time decay."""
        if not values:
            raise ValueError("No values for weighted average")

        weights = []
        vals = []
        for val, ts in values:
            age_seconds = abs((target_time - ts).total_seconds())
            weight = math.exp(-age_seconds / decay_seconds)
            weights.append(weight)
            vals.append(val)

        total_weight = sum(weights)
        if total_weight == 0:
            return vals[0]

        return sum(v * w for v, w in zip(vals, weights)) / total_weight


# =============================================================================
# Reading History Buffer
# =============================================================================

@dataclass
class HistoricalReading:
    """Historical reading for trend analysis."""
    value: float
    timestamp: datetime
    quality: QualityCode


class ReadingBuffer:
    """Circular buffer for historical readings."""

    def __init__(self, max_size: int = 1000, max_age_seconds: int = 3600):
        self._buffer: Deque[HistoricalReading] = deque(maxlen=max_size)
        self._max_age_seconds = max_age_seconds

    def add(self, value: float, timestamp: datetime, quality: QualityCode) -> None:
        """Add a reading to the buffer."""
        self._buffer.append(HistoricalReading(value, timestamp, quality))
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        """Remove readings older than max_age_seconds."""
        cutoff = datetime.utcnow() - timedelta(seconds=self._max_age_seconds)
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def get_last(self, n: int = 1) -> List[HistoricalReading]:
        """Get last N readings."""
        return list(self._buffer)[-n:]

    def get_since(self, since: datetime) -> List[HistoricalReading]:
        """Get readings since timestamp."""
        return [r for r in self._buffer if r.timestamp >= since]

    def get_values(self) -> List[float]:
        """Get all values."""
        return [r.value for r in self._buffer]

    def get_good_values(self) -> List[float]:
        """Get only good quality values."""
        return [r.value for r in self._buffer if r.quality == QualityCode.GOOD]

    @property
    def latest(self) -> Optional[HistoricalReading]:
        """Get most recent reading."""
        return self._buffer[-1] if self._buffer else None

    def __len__(self) -> int:
        return len(self._buffer)


# =============================================================================
# Data Cleaning Processor
# =============================================================================

class DataCleaningProcessor:
    """
    Processor for cleaning and normalizing raw chemistry data.

    Performs:
    - Unit conversion and normalization
    - Quality code validation
    - Outlier detection and clamping
    - Spike removal
    - Missing value interpolation
    - Rate-of-change limiting
    - Quality scoring
    """

    def __init__(
        self,
        tag_configs: Optional[Dict[str, TagConfig]] = None,
        interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
        enable_spike_removal: bool = True,
        spike_threshold_stddev: float = 3.0,
    ):
        """
        Initialize the data cleaning processor.

        Args:
            tag_configs: Tag configuration dictionary
            interpolation_method: Method for interpolating missing values
            enable_spike_removal: Enable spike detection and removal
            spike_threshold_stddev: Standard deviations for spike detection
        """
        self.tag_configs = tag_configs or DEFAULT_TAG_CONFIGS
        self.interpolation_method = interpolation_method
        self.enable_spike_removal = enable_spike_removal
        self.spike_threshold_stddev = spike_threshold_stddev

        # Reading history buffers per tag
        self._buffers: Dict[str, ReadingBuffer] = {}

        # Metrics
        self._processed_count = 0
        self._interpolated_count = 0
        self._clamped_count = 0
        self._spike_removed_count = 0

    def _get_buffer(self, tag_id: str) -> ReadingBuffer:
        """Get or create buffer for tag."""
        if tag_id not in self._buffers:
            self._buffers[tag_id] = ReadingBuffer()
        return self._buffers[tag_id]

    def _get_tag_config(self, tag_id: str) -> Optional[TagConfig]:
        """Get tag configuration by ID or name."""
        for config in self.tag_configs.values():
            if config.tag_id == tag_id or config.name.lower() == tag_id.lower():
                return config
        return None

    def process(self, raw_message: RawChemistryMessage) -> CleanedChemistryMessage:
        """
        Process a raw chemistry message.

        Args:
            raw_message: Raw sensor data

        Returns:
            Cleaned and validated message
        """
        import time
        start_time = time.time()

        cleaned_readings: List[CleanedReading] = []
        quality_scores: List[float] = []

        for reading in raw_message.readings:
            cleaned = self._clean_reading(reading)
            cleaned_readings.append(cleaned)
            quality_scores.append(cleaned.quality_score)

            # Update history buffer
            buffer = self._get_buffer(reading.tag_id)
            buffer.add(cleaned.cleaned_value, reading.timestamp, cleaned.quality)

        # Calculate overall quality score
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        processing_time_ms = int((time.time() - start_time) * 1000)
        self._processed_count += 1

        return CleanedChemistryMessage(
            source=raw_message.source,
            trace_id=raw_message.trace_id,
            boiler_id=raw_message.boiler_id,
            readings=cleaned_readings,
            raw_message_id=raw_message.message_id,
            processing_time_ms=processing_time_ms,
            overall_quality_score=overall_score,
        )

    def _clean_reading(self, reading: SensorReading) -> CleanedReading:
        """Clean a single sensor reading."""
        config = self._get_tag_config(reading.tag_id)
        buffer = self._get_buffer(reading.tag_id)
        assessment = QualityAssessment()

        raw_value = reading.value
        cleaned_value = raw_value

        # Check source quality
        if reading.quality == QualityCode.BAD:
            assessment.add_flag(
                QualityFlag.BAD_SOURCE,
                "Source quality is BAD",
                penalty=0.5
            )
            # Attempt interpolation
            cleaned_value = self._interpolate(buffer, reading.timestamp)
            if cleaned_value is not None:
                assessment.add_flag(QualityFlag.INTERPOLATED, "Value interpolated")
            else:
                cleaned_value = buffer.latest.value if buffer.latest else raw_value

        # Apply range clamping
        if config:
            if cleaned_value < config.min_value:
                cleaned_value = config.min_value
                assessment.add_flag(
                    QualityFlag.CLAMPED_LOW,
                    f"Clamped to min {config.min_value}",
                    penalty=0.2
                )
                self._clamped_count += 1
            elif cleaned_value > config.max_value:
                cleaned_value = config.max_value
                assessment.add_flag(
                    QualityFlag.CLAMPED_HIGH,
                    f"Clamped to max {config.max_value}",
                    penalty=0.2
                )
                self._clamped_count += 1

        # Spike detection
        if self.enable_spike_removal and len(buffer) >= 10:
            if self._is_spike(cleaned_value, buffer):
                assessment.add_flag(
                    QualityFlag.SPIKE_REMOVED,
                    "Spike detected and removed",
                    penalty=0.3
                )
                cleaned_value = self._interpolate(buffer, reading.timestamp) or cleaned_value
                self._spike_removed_count += 1

        # Rate of change limiting
        if config and buffer.latest:
            cleaned_value = self._apply_rate_limit(
                cleaned_value,
                buffer.latest.value,
                buffer.latest.timestamp,
                reading.timestamp,
                config.rate_of_change_limit,
                assessment,
            )

        # Staleness check
        if buffer.latest:
            age = (reading.timestamp - buffer.latest.timestamp).total_seconds()
            config_stale = config.stale_seconds if config else 60
            if age > config_stale:
                assessment.add_flag(
                    QualityFlag.STALE,
                    f"Data age {age:.0f}s exceeds threshold",
                    penalty=0.15
                )

        # Determine final quality code
        final_quality = self._determine_quality_code(reading.quality, assessment)

        return CleanedReading(
            tag_id=reading.tag_id,
            raw_value=raw_value,
            cleaned_value=cleaned_value,
            quality=final_quality,
            quality_score=assessment.score,
            engineering_units=reading.engineering_units,
            was_interpolated=QualityFlag.INTERPOLATED in assessment.flags,
            was_clamped=(
                QualityFlag.CLAMPED_HIGH in assessment.flags or
                QualityFlag.CLAMPED_LOW in assessment.flags
            ),
            interpolation_method=(
                self.interpolation_method.value
                if QualityFlag.INTERPOLATED in assessment.flags
                else None
            ),
        )

    def _interpolate(
        self,
        buffer: ReadingBuffer,
        target_time: datetime,
    ) -> Optional[float]:
        """Interpolate a missing value."""
        if len(buffer) < 2:
            return None

        readings = buffer.get_last(10)

        if self.interpolation_method == InterpolationMethod.LAST_VALUE:
            return readings[-1].value

        elif self.interpolation_method == InterpolationMethod.AVERAGE:
            values = [r.value for r in readings if r.quality == QualityCode.GOOD]
            return statistics.mean(values) if values else None

        elif self.interpolation_method == InterpolationMethod.LINEAR:
            # Find surrounding good readings
            good_readings = [r for r in readings if r.quality == QualityCode.GOOD]
            if len(good_readings) < 2:
                return good_readings[-1].value if good_readings else None

            prev = good_readings[-2]
            next_r = good_readings[-1]
            return Interpolator.linear(
                target_time,
                prev.value,
                prev.timestamp,
                next_r.value,
                next_r.timestamp,
            )

        elif self.interpolation_method == InterpolationMethod.WEIGHTED:
            values_times = [
                (r.value, r.timestamp)
                for r in readings
                if r.quality == QualityCode.GOOD
            ]
            if values_times:
                return Interpolator.weighted_average(values_times, target_time)

        self._interpolated_count += 1
        return None

    def _is_spike(self, value: float, buffer: ReadingBuffer) -> bool:
        """Detect if value is a spike based on historical data."""
        good_values = buffer.get_good_values()
        if len(good_values) < 5:
            return False

        mean = statistics.mean(good_values)
        stdev = statistics.stdev(good_values)

        if stdev == 0:
            return value != mean

        z_score = abs(value - mean) / stdev
        return z_score > self.spike_threshold_stddev

    def _apply_rate_limit(
        self,
        new_value: float,
        prev_value: float,
        prev_time: datetime,
        new_time: datetime,
        max_rate: float,
        assessment: QualityAssessment,
    ) -> float:
        """Apply rate of change limiting."""
        delta_time = (new_time - prev_time).total_seconds()
        if delta_time <= 0:
            return new_value

        delta_value = new_value - prev_value
        actual_rate = abs(delta_value) / delta_time

        if actual_rate > max_rate:
            # Limit the change
            max_change = max_rate * delta_time
            if delta_value > 0:
                limited_value = prev_value + max_change
            else:
                limited_value = prev_value - max_change

            assessment.add_flag(
                QualityFlag.RATE_LIMITED,
                f"Rate limited from {actual_rate:.2f} to {max_rate:.2f} per second",
                penalty=0.1
            )
            return limited_value

        return new_value

    def _determine_quality_code(
        self,
        source_quality: QualityCode,
        assessment: QualityAssessment,
    ) -> QualityCode:
        """Determine final quality code based on assessment."""
        if source_quality == QualityCode.BAD:
            return QualityCode.BAD

        if QualityFlag.BAD_SOURCE in assessment.flags:
            if QualityFlag.INTERPOLATED in assessment.flags:
                return QualityCode.SUBSTITUTED
            return QualityCode.BAD

        if assessment.score < 0.5:
            return QualityCode.UNCERTAIN

        if assessment.score < 0.8:
            return QualityCode.UNCERTAIN

        if QualityFlag.STALE in assessment.flags:
            return QualityCode.STALE

        return QualityCode.GOOD

    @property
    def stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed_count": self._processed_count,
            "interpolated_count": self._interpolated_count,
            "clamped_count": self._clamped_count,
            "spike_removed_count": self._spike_removed_count,
        }


# =============================================================================
# Feature Engineering Processor
# =============================================================================

@dataclass
class FeatureSpec:
    """Specification for a derived feature."""
    name: str
    input_tags: List[str]
    calculation: str  # Method name
    window_seconds: int = 300
    min_samples: int = 5


class FeatureEngineeringProcessor:
    """
    Processor for computing derived features from cleaned chemistry data.

    Features include:
    - Rolling statistics (mean, std, min, max)
    - Rate of change and trends
    - Cross-parameter correlations
    - Anomaly indicators
    """

    # Feature specifications
    FEATURE_SPECS: List[FeatureSpec] = [
        FeatureSpec("phosphate_mean_5m", ["phosphate"], "rolling_mean", 300, 5),
        FeatureSpec("phosphate_std_5m", ["phosphate"], "rolling_std", 300, 5),
        FeatureSpec("phosphate_rate_of_change", ["phosphate"], "rate_of_change", 60, 2),
        FeatureSpec("conductivity_mean_5m", ["conductivity"], "rolling_mean", 300, 5),
        FeatureSpec("conductivity_std_5m", ["conductivity"], "rolling_std", 300, 5),
        FeatureSpec("ph_mean_5m", ["ph"], "rolling_mean", 300, 5),
        FeatureSpec("ph_trend", ["ph"], "trend", 300, 10),
        FeatureSpec("do_mean_5m", ["dissolved_oxygen"], "rolling_mean", 300, 5),
        FeatureSpec("cond_phosphate_ratio", ["conductivity", "phosphate"], "ratio", 60, 2),
        FeatureSpec("chemistry_stability", ["phosphate", "ph", "conductivity"], "stability_index", 300, 10),
    ]

    def __init__(
        self,
        feature_specs: Optional[List[FeatureSpec]] = None,
        default_window_seconds: int = 300,
    ):
        """
        Initialize feature engineering processor.

        Args:
            feature_specs: Custom feature specifications
            default_window_seconds: Default window size
        """
        self.feature_specs = feature_specs or self.FEATURE_SPECS
        self.default_window_seconds = default_window_seconds

        # History buffers per tag
        self._buffers: Dict[str, ReadingBuffer] = {}

    def _get_buffer(self, tag_id: str) -> ReadingBuffer:
        """Get or create buffer for tag."""
        if tag_id not in self._buffers:
            self._buffers[tag_id] = ReadingBuffer(max_age_seconds=3600)
        return self._buffers[tag_id]

    def update_buffers(self, cleaned_message: CleanedChemistryMessage) -> None:
        """Update history buffers with cleaned readings."""
        for reading in cleaned_message.readings:
            buffer = self._get_buffer(reading.tag_id)
            buffer.add(
                reading.cleaned_value,
                cleaned_message.timestamp,
                reading.quality,
            )

    def compute_features(
        self,
        cleaned_message: CleanedChemistryMessage,
    ) -> FeatureMessage:
        """
        Compute derived features from cleaned data.

        Args:
            cleaned_message: Cleaned chemistry message

        Returns:
            Feature message with computed features
        """
        # Update buffers first
        self.update_buffers(cleaned_message)

        features: List[DerivedFeature] = []
        window_end = cleaned_message.timestamp

        for spec in self.feature_specs:
            try:
                feature = self._compute_feature(spec, window_end)
                if feature:
                    features.append(feature)
            except Exception as e:
                logger.warning(f"Failed to compute feature {spec.name}: {e}")

        window_start = window_end - timedelta(seconds=self.default_window_seconds)

        return FeatureMessage(
            source=cleaned_message.source,
            trace_id=cleaned_message.trace_id,
            boiler_id=cleaned_message.boiler_id,
            features=features,
            cleaned_message_id=cleaned_message.message_id,
            window_start=window_start,
            window_end=window_end,
            window_size_seconds=self.default_window_seconds,
        )

    def _compute_feature(
        self,
        spec: FeatureSpec,
        window_end: datetime,
    ) -> Optional[DerivedFeature]:
        """Compute a single feature."""
        window_start = window_end - timedelta(seconds=spec.window_seconds)

        # Gather values from all input tags
        all_values: Dict[str, List[float]] = {}
        for tag_id in spec.input_tags:
            buffer = self._get_buffer(tag_id)
            readings = buffer.get_since(window_start)
            values = [r.value for r in readings if r.quality == QualityCode.GOOD]

            if len(values) < spec.min_samples:
                return None  # Insufficient data

            all_values[tag_id] = values

        # Compute based on calculation type
        value = None
        confidence = 1.0

        if spec.calculation == "rolling_mean":
            values = all_values[spec.input_tags[0]]
            value = statistics.mean(values)
            confidence = min(1.0, len(values) / (spec.min_samples * 2))

        elif spec.calculation == "rolling_std":
            values = all_values[spec.input_tags[0]]
            value = statistics.stdev(values) if len(values) >= 2 else 0.0
            confidence = min(1.0, len(values) / (spec.min_samples * 2))

        elif spec.calculation == "rate_of_change":
            values = all_values[spec.input_tags[0]]
            if len(values) >= 2:
                value = (values[-1] - values[0]) / spec.window_seconds
                confidence = 0.9

        elif spec.calculation == "trend":
            values = all_values[spec.input_tags[0]]
            # Simple linear regression slope
            if len(values) >= 3:
                x = list(range(len(values)))
                x_mean = statistics.mean(x)
                y_mean = statistics.mean(values)
                numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
                denominator = sum((xi - x_mean) ** 2 for xi in x)
                value = numerator / denominator if denominator != 0 else 0.0
                confidence = 0.85

        elif spec.calculation == "ratio":
            if len(spec.input_tags) >= 2:
                vals1 = all_values[spec.input_tags[0]]
                vals2 = all_values[spec.input_tags[1]]
                if vals1 and vals2:
                    mean1 = statistics.mean(vals1)
                    mean2 = statistics.mean(vals2)
                    value = mean1 / mean2 if mean2 != 0 else 0.0
                    confidence = 0.9

        elif spec.calculation == "stability_index":
            # Combined stability metric (0-1, higher is more stable)
            std_devs = []
            for tag_id in spec.input_tags:
                if tag_id in all_values and len(all_values[tag_id]) >= 2:
                    std = statistics.stdev(all_values[tag_id])
                    mean = statistics.mean(all_values[tag_id])
                    cv = std / mean if mean != 0 else 0.0  # Coefficient of variation
                    std_devs.append(cv)

            if std_devs:
                avg_cv = statistics.mean(std_devs)
                value = max(0.0, 1.0 - avg_cv)  # Invert: low CV = high stability
                confidence = 0.8

        if value is None:
            return None

        return DerivedFeature(
            feature_name=spec.name,
            value=value,
            confidence=confidence,
            input_tags=spec.input_tags,
            calculation_method=spec.calculation,
        )

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [spec.name for spec in self.feature_specs]
