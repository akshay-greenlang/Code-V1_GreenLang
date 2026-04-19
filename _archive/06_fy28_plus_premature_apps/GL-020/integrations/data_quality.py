"""
GL-020 ECONOPULSE - Data Quality Management Module

Enterprise-grade data quality validation and management providing:
- Range checking (physically plausible values)
- Rate-of-change limits (spike detection)
- Sensor redundancy management (2oo3 voting)
- Bad data substitution strategies
- Data quality flags and confidence scores

Thread-safe for concurrent validation operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import threading
import statistics
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class QualityFlag(Enum):
    """Data quality flags following OPC/ISA standards."""
    GOOD = "good"
    GOOD_LOCAL_OVERRIDE = "good_local_override"
    GOOD_CLAMPED = "good_clamped"
    UNCERTAIN = "uncertain"
    UNCERTAIN_LAST_USABLE_VALUE = "uncertain_last_usable_value"
    UNCERTAIN_SENSOR_CAL = "uncertain_sensor_cal"
    UNCERTAIN_ENGINEERING_UNITS_EXCEEDED = "uncertain_eu_exceeded"
    UNCERTAIN_SUB_NORMAL = "uncertain_sub_normal"
    BAD = "bad"
    BAD_CONFIGURATION_ERROR = "bad_config_error"
    BAD_NOT_CONNECTED = "bad_not_connected"
    BAD_DEVICE_FAILURE = "bad_device_failure"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"
    BAD_WAITING_FOR_INITIAL_DATA = "bad_waiting"

    def is_good(self) -> bool:
        """Check if quality is good."""
        return self in (
            QualityFlag.GOOD,
            QualityFlag.GOOD_LOCAL_OVERRIDE,
            QualityFlag.GOOD_CLAMPED,
        )

    def is_usable(self) -> bool:
        """Check if value is usable (good or uncertain)."""
        return self.is_good() or self.name.startswith("UNCERTAIN")


class SubstitutionMethod(Enum):
    """Methods for substituting bad data."""
    NONE = "none"
    LAST_GOOD_VALUE = "last_good_value"
    LINEAR_INTERPOLATION = "linear_interpolation"
    AVERAGE = "average"
    MANUAL_ENTRY = "manual_entry"
    REDUNDANT_SENSOR = "redundant_sensor"
    CALCULATED = "calculated"


class VotingMethod(Enum):
    """Sensor redundancy voting methods."""
    TWO_OUT_OF_THREE = "2oo3"
    AVERAGE_OF_GOOD = "avg_good"
    MEDIAN = "median"
    HIGH_SELECT = "high_select"
    LOW_SELECT = "low_select"
    PRIORITY = "priority"


class ValidationResult(Enum):
    """Result of validation check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfidenceScore:
    """
    Confidence score with component breakdown.

    Total score is 0-100 based on weighted components:
    - Sensor quality: 30%
    - Range validity: 25%
    - Rate of change: 20%
    - Consistency: 15%
    - Calibration status: 10%
    """
    sensor_quality_score: float = 100.0
    range_validity_score: float = 100.0
    rate_of_change_score: float = 100.0
    consistency_score: float = 100.0
    calibration_score: float = 100.0

    # Weights
    WEIGHT_SENSOR: float = 0.30
    WEIGHT_RANGE: float = 0.25
    WEIGHT_ROC: float = 0.20
    WEIGHT_CONSISTENCY: float = 0.15
    WEIGHT_CALIBRATION: float = 0.10

    @property
    def total_score(self) -> float:
        """Calculate weighted total confidence score."""
        return (
            self.sensor_quality_score * self.WEIGHT_SENSOR +
            self.range_validity_score * self.WEIGHT_RANGE +
            self.rate_of_change_score * self.WEIGHT_ROC +
            self.consistency_score * self.WEIGHT_CONSISTENCY +
            self.calibration_score * self.WEIGHT_CALIBRATION
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "sensor_quality": self.sensor_quality_score,
            "range_validity": self.range_validity_score,
            "rate_of_change": self.rate_of_change_score,
            "consistency": self.consistency_score,
            "calibration": self.calibration_score,
            "total": self.total_score,
        }


@dataclass
class RangeCheck:
    """Configuration for range validation."""
    tag_name: str
    low_limit: float
    high_limit: float
    low_low_limit: Optional[float] = None  # Critical low
    high_high_limit: Optional[float] = None  # Critical high
    engineering_unit: str = ""
    description: str = ""
    clamp_to_limits: bool = False  # Clamp values to limits instead of flagging

    def validate(self, value: float) -> Tuple[ValidationResult, QualityFlag, str]:
        """
        Validate a value against range limits.

        Returns:
            Tuple of (result, quality_flag, message)
        """
        # Check critical limits first
        if self.low_low_limit is not None and value < self.low_low_limit:
            return (
                ValidationResult.FAILED,
                QualityFlag.BAD_SENSOR_FAILURE,
                f"Value {value} below critical low limit {self.low_low_limit}",
            )

        if self.high_high_limit is not None and value > self.high_high_limit:
            return (
                ValidationResult.FAILED,
                QualityFlag.BAD_SENSOR_FAILURE,
                f"Value {value} above critical high limit {self.high_high_limit}",
            )

        # Check normal limits
        if value < self.low_limit:
            if self.clamp_to_limits:
                return (
                    ValidationResult.WARNING,
                    QualityFlag.GOOD_CLAMPED,
                    f"Value clamped from {value} to {self.low_limit}",
                )
            return (
                ValidationResult.WARNING,
                QualityFlag.UNCERTAIN_ENGINEERING_UNITS_EXCEEDED,
                f"Value {value} below low limit {self.low_limit}",
            )

        if value > self.high_limit:
            if self.clamp_to_limits:
                return (
                    ValidationResult.WARNING,
                    QualityFlag.GOOD_CLAMPED,
                    f"Value clamped from {value} to {self.high_limit}",
                )
            return (
                ValidationResult.WARNING,
                QualityFlag.UNCERTAIN_ENGINEERING_UNITS_EXCEEDED,
                f"Value {value} above high limit {self.high_limit}",
            )

        return (ValidationResult.PASSED, QualityFlag.GOOD, "")

    def clamp(self, value: float) -> float:
        """Clamp value to range limits."""
        return max(self.low_limit, min(self.high_limit, value))


@dataclass
class RateOfChangeLimit:
    """Configuration for rate-of-change validation."""
    tag_name: str
    max_rate_per_second: float
    max_rate_per_minute: Optional[float] = None
    window_seconds: float = 60.0
    description: str = ""

    def validate(
        self,
        current_value: float,
        previous_value: float,
        time_delta_seconds: float,
    ) -> Tuple[ValidationResult, QualityFlag, str]:
        """
        Validate rate of change between two values.

        Returns:
            Tuple of (result, quality_flag, message)
        """
        if time_delta_seconds <= 0:
            return (ValidationResult.SKIPPED, QualityFlag.UNCERTAIN, "Invalid time delta")

        rate_per_second = abs(current_value - previous_value) / time_delta_seconds

        if rate_per_second > self.max_rate_per_second:
            return (
                ValidationResult.FAILED,
                QualityFlag.UNCERTAIN,
                f"Rate of change {rate_per_second:.4f}/s exceeds limit {self.max_rate_per_second}/s",
            )

        if self.max_rate_per_minute:
            rate_per_minute = rate_per_second * 60
            if rate_per_minute > self.max_rate_per_minute:
                return (
                    ValidationResult.WARNING,
                    QualityFlag.UNCERTAIN,
                    f"Rate of change {rate_per_minute:.2f}/min exceeds limit {self.max_rate_per_minute}/min",
                )

        return (ValidationResult.PASSED, QualityFlag.GOOD, "")


@dataclass
class ValidatedValue:
    """Result of data validation with quality information."""
    tag_name: str
    original_value: float
    validated_value: float
    timestamp: datetime
    quality_flag: QualityFlag
    confidence_score: ConfidenceScore
    validation_results: List[Tuple[str, ValidationResult, str]] = field(default_factory=list)
    substituted: bool = False
    substitution_method: Optional[SubstitutionMethod] = None
    substitution_source: str = ""

    @property
    def is_good(self) -> bool:
        """Check if validated value is good quality."""
        return self.quality_flag.is_good()

    @property
    def is_usable(self) -> bool:
        """Check if validated value is usable."""
        return self.quality_flag.is_usable()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "original_value": self.original_value,
            "validated_value": self.validated_value,
            "timestamp": self.timestamp.isoformat(),
            "quality_flag": self.quality_flag.value,
            "confidence_score": self.confidence_score.to_dict(),
            "validation_results": [
                {"check": r[0], "result": r[1].value, "message": r[2]}
                for r in self.validation_results
            ],
            "substituted": self.substituted,
            "substitution_method": (
                self.substitution_method.value if self.substitution_method else None
            ),
        }


@dataclass
class RedundantSensorGroup:
    """Configuration for redundant sensor group."""
    group_name: str
    sensor_tags: List[str]
    voting_method: VotingMethod = VotingMethod.TWO_OUT_OF_THREE
    max_deviation_percent: float = 5.0  # Maximum allowed deviation between sensors
    output_tag: str = ""
    description: str = ""
    priority_order: List[str] = field(default_factory=list)  # For priority voting


@dataclass
class BadDataSubstitution:
    """Configuration for bad data substitution."""
    tag_name: str
    method: SubstitutionMethod
    fallback_value: Optional[float] = None
    max_substitution_duration_seconds: float = 300.0
    interpolation_max_gap_seconds: float = 60.0
    source_tag: Optional[str] = None  # For calculated or redundant substitution
    calculation_expression: str = ""


# =============================================================================
# Data Quality Validator
# =============================================================================

class DataQualityValidator:
    """
    Enterprise data quality validation engine.

    Provides comprehensive validation including:
    - Range checking with configurable limits
    - Rate-of-change spike detection
    - Statistical outlier detection
    - Historical comparison
    - Sensor redundancy voting
    - Bad data substitution
    """

    def __init__(
        self,
        history_window_size: int = 1000,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize data quality validator.

        Args:
            history_window_size: Size of value history buffer per tag
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self._lock = threading.RLock()

        # Configuration stores
        self._range_checks: Dict[str, RangeCheck] = {}
        self._roc_limits: Dict[str, RateOfChangeLimit] = {}
        self._substitution_configs: Dict[str, BadDataSubstitution] = {}

        # Value history for statistical analysis
        self._value_history: Dict[str, deque] = {}
        self._history_window_size = history_window_size

        # Last good values for substitution
        self._last_good_values: Dict[str, Tuple[float, datetime]] = {}

        # Outlier detection threshold
        self._outlier_std_threshold = outlier_std_threshold

        # Substitution tracking
        self._substitution_start: Dict[str, datetime] = {}

        logger.info("Initialized DataQualityValidator")

    def configure_range_check(self, config: RangeCheck) -> None:
        """Add or update range check configuration."""
        with self._lock:
            self._range_checks[config.tag_name] = config
            logger.debug(f"Configured range check for {config.tag_name}")

    def configure_roc_limit(self, config: RateOfChangeLimit) -> None:
        """Add or update rate-of-change limit configuration."""
        with self._lock:
            self._roc_limits[config.tag_name] = config
            logger.debug(f"Configured ROC limit for {config.tag_name}")

    def configure_substitution(self, config: BadDataSubstitution) -> None:
        """Add or update bad data substitution configuration."""
        with self._lock:
            self._substitution_configs[config.tag_name] = config
            logger.debug(f"Configured substitution for {config.tag_name}")

    def validate(
        self,
        tag_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        source_quality: Optional[QualityFlag] = None,
    ) -> ValidatedValue:
        """
        Validate a single value with all configured checks.

        Args:
            tag_name: Tag name being validated
            value: Value to validate
            timestamp: Value timestamp (default: now)
            source_quality: Quality flag from source (e.g., sensor)

        Returns:
            ValidatedValue with quality assessment
        """
        if timestamp is None:
            timestamp = datetime.now()

        validation_results = []
        quality_flag = source_quality or QualityFlag.GOOD
        confidence = ConfidenceScore()
        validated_value = value
        substituted = False
        sub_method = None
        sub_source = ""

        with self._lock:
            # Initialize history if needed
            if tag_name not in self._value_history:
                self._value_history[tag_name] = deque(maxlen=self._history_window_size)

            # 1. Check source quality
            if source_quality and not source_quality.is_good():
                confidence.sensor_quality_score = 50.0 if source_quality.is_usable() else 0.0
                validation_results.append((
                    "source_quality",
                    ValidationResult.WARNING if source_quality.is_usable() else ValidationResult.FAILED,
                    f"Source quality: {source_quality.value}",
                ))

            # 2. Range check
            if tag_name in self._range_checks:
                range_config = self._range_checks[tag_name]
                result, flag, message = range_config.validate(value)

                validation_results.append(("range_check", result, message))

                if result == ValidationResult.FAILED:
                    quality_flag = flag
                    confidence.range_validity_score = 0.0
                elif result == ValidationResult.WARNING:
                    if range_config.clamp_to_limits:
                        validated_value = range_config.clamp(value)
                    quality_flag = flag
                    confidence.range_validity_score = 50.0
            else:
                validation_results.append((
                    "range_check",
                    ValidationResult.SKIPPED,
                    "No range check configured",
                ))

            # 3. Rate of change check
            if tag_name in self._roc_limits and self._value_history[tag_name]:
                roc_config = self._roc_limits[tag_name]
                prev_value, prev_time = self._value_history[tag_name][-1]
                time_delta = (timestamp - prev_time).total_seconds()

                result, flag, message = roc_config.validate(
                    value, prev_value, time_delta
                )

                validation_results.append(("rate_of_change", result, message))

                if result == ValidationResult.FAILED:
                    if quality_flag.is_good():
                        quality_flag = flag
                    confidence.rate_of_change_score = 0.0
                elif result == ValidationResult.WARNING:
                    confidence.rate_of_change_score = 50.0
            else:
                validation_results.append((
                    "rate_of_change",
                    ValidationResult.SKIPPED,
                    "No ROC limit configured or insufficient history",
                ))

            # 4. Statistical outlier detection
            if len(self._value_history[tag_name]) >= 10:
                outlier_result = self._check_outlier(tag_name, value)
                validation_results.append(outlier_result)

                if outlier_result[1] == ValidationResult.WARNING:
                    confidence.consistency_score = 50.0
            else:
                validation_results.append((
                    "outlier_check",
                    ValidationResult.SKIPPED,
                    "Insufficient history for outlier detection",
                ))

            # 5. Apply substitution if needed
            if not quality_flag.is_usable():
                sub_result = self._apply_substitution(tag_name, value, timestamp)

                if sub_result:
                    validated_value, sub_method, sub_source = sub_result
                    substituted = True
                    quality_flag = QualityFlag.UNCERTAIN_LAST_USABLE_VALUE

                    validation_results.append((
                        "substitution",
                        ValidationResult.WARNING,
                        f"Applied {sub_method.value} substitution from {sub_source}",
                    ))

            # 6. Update history
            self._value_history[tag_name].append((validated_value, timestamp))

            # Update last good value
            if quality_flag.is_good():
                self._last_good_values[tag_name] = (validated_value, timestamp)

        return ValidatedValue(
            tag_name=tag_name,
            original_value=value,
            validated_value=validated_value,
            timestamp=timestamp,
            quality_flag=quality_flag,
            confidence_score=confidence,
            validation_results=validation_results,
            substituted=substituted,
            substitution_method=sub_method,
            substitution_source=sub_source,
        )

    def _check_outlier(
        self, tag_name: str, value: float
    ) -> Tuple[str, ValidationResult, str]:
        """Check if value is a statistical outlier."""
        history = self._value_history[tag_name]
        values = [v for v, _ in history]

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std > 0:
            z_score = abs(value - mean) / std

            if z_score > self._outlier_std_threshold:
                return (
                    "outlier_check",
                    ValidationResult.WARNING,
                    f"Value {value:.4f} is {z_score:.2f} std deviations from mean {mean:.4f}",
                )

        return ("outlier_check", ValidationResult.PASSED, "")

    def _apply_substitution(
        self,
        tag_name: str,
        value: float,
        timestamp: datetime,
    ) -> Optional[Tuple[float, SubstitutionMethod, str]]:
        """Apply bad data substitution if configured."""
        config = self._substitution_configs.get(tag_name)

        if not config:
            return None

        # Check substitution duration limit
        if tag_name in self._substitution_start:
            duration = (timestamp - self._substitution_start[tag_name]).total_seconds()
            if duration > config.max_substitution_duration_seconds:
                logger.warning(
                    f"Substitution duration exceeded for {tag_name} "
                    f"({duration:.0f}s > {config.max_substitution_duration_seconds}s)"
                )
                return None
        else:
            self._substitution_start[tag_name] = timestamp

        # Apply substitution method
        if config.method == SubstitutionMethod.LAST_GOOD_VALUE:
            if tag_name in self._last_good_values:
                last_value, last_time = self._last_good_values[tag_name]
                return (last_value, config.method, f"Last good value at {last_time}")

        elif config.method == SubstitutionMethod.LINEAR_INTERPOLATION:
            # Would need future value - typically handled differently
            if tag_name in self._last_good_values:
                last_value, _ = self._last_good_values[tag_name]
                return (last_value, config.method, "Interpolated (using last good)")

        elif config.method == SubstitutionMethod.AVERAGE:
            history = self._value_history.get(tag_name, [])
            if history:
                avg = statistics.mean([v for v, _ in history])
                return (avg, config.method, f"Average of {len(history)} values")

        elif config.method == SubstitutionMethod.MANUAL_ENTRY:
            if config.fallback_value is not None:
                return (config.fallback_value, config.method, "Manual fallback value")

        return None

    def validate_batch(
        self,
        values: Dict[str, Tuple[float, datetime]],
        source_qualities: Optional[Dict[str, QualityFlag]] = None,
    ) -> Dict[str, ValidatedValue]:
        """
        Validate multiple values in batch.

        Args:
            values: Dictionary mapping tag names to (value, timestamp) tuples
            source_qualities: Optional quality flags from sources

        Returns:
            Dictionary mapping tag names to ValidatedValue objects
        """
        results = {}
        source_qualities = source_qualities or {}

        for tag_name, (value, timestamp) in values.items():
            results[tag_name] = self.validate(
                tag_name=tag_name,
                value=value,
                timestamp=timestamp,
                source_quality=source_qualities.get(tag_name),
            )

        return results

    def get_statistics(self, tag_name: str) -> Dict[str, Any]:
        """Get validation statistics for a tag."""
        with self._lock:
            history = self._value_history.get(tag_name, [])

            if not history:
                return {"tag_name": tag_name, "sample_count": 0}

            values = [v for v, _ in history]

            return {
                "tag_name": tag_name,
                "sample_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "last_value": values[-1],
                "last_timestamp": history[-1][1].isoformat(),
            }

    def clear_history(self, tag_name: Optional[str] = None) -> None:
        """Clear value history for tag(s)."""
        with self._lock:
            if tag_name:
                self._value_history.pop(tag_name, None)
                self._last_good_values.pop(tag_name, None)
                self._substitution_start.pop(tag_name, None)
            else:
                self._value_history.clear()
                self._last_good_values.clear()
                self._substitution_start.clear()


# =============================================================================
# Redundancy Manager
# =============================================================================

class RedundancyManager:
    """
    Manages sensor redundancy voting and selection.

    Supports multiple voting methods:
    - 2-out-of-3 (2oo3): Requires 2 sensors to agree
    - Average of good: Average all good-quality readings
    - Median: Use median value
    - High/Low select: Use highest or lowest value
    - Priority: Use first available by priority
    """

    def __init__(self, validator: Optional[DataQualityValidator] = None):
        """
        Initialize redundancy manager.

        Args:
            validator: Optional DataQualityValidator for quality assessment
        """
        self._lock = threading.RLock()
        self._validator = validator
        self._sensor_groups: Dict[str, RedundantSensorGroup] = {}
        self._sensor_health: Dict[str, Dict[str, bool]] = {}

        logger.info("Initialized RedundancyManager")

    def register_group(self, group: RedundantSensorGroup) -> None:
        """Register a redundant sensor group."""
        with self._lock:
            self._sensor_groups[group.group_name] = group
            self._sensor_health[group.group_name] = {
                tag: True for tag in group.sensor_tags
            }
            logger.info(
                f"Registered redundant sensor group: {group.group_name} "
                f"with {len(group.sensor_tags)} sensors"
            )

    def get_group(self, group_name: str) -> Optional[RedundantSensorGroup]:
        """Get a sensor group by name."""
        with self._lock:
            return self._sensor_groups.get(group_name)

    def vote(
        self,
        group_name: str,
        sensor_values: Dict[str, Tuple[float, QualityFlag]],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """
        Perform voting on redundant sensor values.

        Args:
            group_name: Name of sensor group
            sensor_values: Dict mapping sensor tag to (value, quality) tuples

        Returns:
            Tuple of (selected_value, quality_flag, selection_reason)
        """
        with self._lock:
            group = self._sensor_groups.get(group_name)

            if not group:
                return None, QualityFlag.BAD_CONFIGURATION_ERROR, f"Unknown group: {group_name}"

            # Filter to good/usable values
            good_values = {}
            usable_values = {}

            for tag, (value, quality) in sensor_values.items():
                if quality.is_good():
                    good_values[tag] = value
                    usable_values[tag] = value
                elif quality.is_usable():
                    usable_values[tag] = value

                # Update health tracking
                if tag in self._sensor_health[group_name]:
                    self._sensor_health[group_name][tag] = quality.is_usable()

            # Check minimum sensors
            if not usable_values:
                return None, QualityFlag.BAD_SENSOR_FAILURE, "No usable sensor values"

            # Apply voting method
            if group.voting_method == VotingMethod.TWO_OUT_OF_THREE:
                return self._vote_2oo3(group, good_values, usable_values)

            elif group.voting_method == VotingMethod.AVERAGE_OF_GOOD:
                return self._vote_average(group, good_values, usable_values)

            elif group.voting_method == VotingMethod.MEDIAN:
                return self._vote_median(group, usable_values)

            elif group.voting_method == VotingMethod.HIGH_SELECT:
                return self._vote_high_select(group, usable_values)

            elif group.voting_method == VotingMethod.LOW_SELECT:
                return self._vote_low_select(group, usable_values)

            elif group.voting_method == VotingMethod.PRIORITY:
                return self._vote_priority(group, sensor_values)

            return None, QualityFlag.BAD_CONFIGURATION_ERROR, "Unknown voting method"

    def _vote_2oo3(
        self,
        group: RedundantSensorGroup,
        good_values: Dict[str, float],
        usable_values: Dict[str, float],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """2-out-of-3 voting logic."""
        values_list = list(usable_values.values())

        if len(values_list) < 2:
            # Fall back to single sensor if available
            if values_list:
                return (
                    values_list[0],
                    QualityFlag.UNCERTAIN,
                    "Single sensor fallback - insufficient for 2oo3",
                )
            return None, QualityFlag.BAD, "Insufficient sensors for 2oo3"

        if len(values_list) >= 3:
            # Full 2oo3 voting
            sorted_values = sorted(values_list)
            median = sorted_values[1]  # Middle value

            # Check if 2 sensors agree (within deviation)
            span = max(values_list) - min(values_list)
            avg = sum(values_list) / len(values_list)
            deviation_pct = (span / avg * 100) if avg != 0 else 0

            if deviation_pct <= group.max_deviation_percent:
                # All sensors agree - use median
                return (
                    median,
                    QualityFlag.GOOD,
                    f"2oo3 voting: median of {len(values_list)} sensors (deviation: {deviation_pct:.1f}%)",
                )
            else:
                # Sensors disagree - check for outlier
                for i, v in enumerate(sorted_values):
                    others = [sorted_values[j] for j in range(len(sorted_values)) if j != i]
                    others_span = max(others) - min(others)
                    others_avg = sum(others) / len(others)
                    others_dev_pct = (others_span / others_avg * 100) if others_avg != 0 else 0

                    if others_dev_pct <= group.max_deviation_percent:
                        # Found outlier - use average of others
                        return (
                            others_avg,
                            QualityFlag.GOOD,
                            f"2oo3 voting: outlier rejected, using average of {len(others)} sensors",
                        )

                # No clear agreement
                return (
                    median,
                    QualityFlag.UNCERTAIN,
                    f"2oo3 voting: high deviation ({deviation_pct:.1f}%), using median",
                )

        else:
            # Only 2 sensors - average if they agree
            span = abs(values_list[0] - values_list[1])
            avg = sum(values_list) / 2
            deviation_pct = (span / avg * 100) if avg != 0 else 0

            if deviation_pct <= group.max_deviation_percent:
                return (
                    avg,
                    QualityFlag.GOOD,
                    f"2oo3 voting: average of 2 sensors (deviation: {deviation_pct:.1f}%)",
                )
            else:
                return (
                    avg,
                    QualityFlag.UNCERTAIN,
                    f"2oo3 voting: 2 sensors disagree ({deviation_pct:.1f}%), using average",
                )

    def _vote_average(
        self,
        group: RedundantSensorGroup,
        good_values: Dict[str, float],
        usable_values: Dict[str, float],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """Average of good values voting."""
        if good_values:
            values = list(good_values.values())
            avg = sum(values) / len(values)
            return (
                avg,
                QualityFlag.GOOD,
                f"Average of {len(values)} good sensors",
            )
        elif usable_values:
            values = list(usable_values.values())
            avg = sum(values) / len(values)
            return (
                avg,
                QualityFlag.UNCERTAIN,
                f"Average of {len(values)} usable sensors (no good quality)",
            )

        return None, QualityFlag.BAD, "No usable sensors"

    def _vote_median(
        self,
        group: RedundantSensorGroup,
        usable_values: Dict[str, float],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """Median value voting."""
        values = list(usable_values.values())
        median = statistics.median(values)
        return (
            median,
            QualityFlag.GOOD,
            f"Median of {len(values)} sensors",
        )

    def _vote_high_select(
        self,
        group: RedundantSensorGroup,
        usable_values: Dict[str, float],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """High select voting (use maximum)."""
        tag, value = max(usable_values.items(), key=lambda x: x[1])
        return (
            value,
            QualityFlag.GOOD,
            f"High select: {tag} = {value}",
        )

    def _vote_low_select(
        self,
        group: RedundantSensorGroup,
        usable_values: Dict[str, float],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """Low select voting (use minimum)."""
        tag, value = min(usable_values.items(), key=lambda x: x[1])
        return (
            value,
            QualityFlag.GOOD,
            f"Low select: {tag} = {value}",
        )

    def _vote_priority(
        self,
        group: RedundantSensorGroup,
        sensor_values: Dict[str, Tuple[float, QualityFlag]],
    ) -> Tuple[Optional[float], QualityFlag, str]:
        """Priority-based selection."""
        priority_list = group.priority_order or group.sensor_tags

        for tag in priority_list:
            if tag in sensor_values:
                value, quality = sensor_values[tag]
                if quality.is_usable():
                    return (
                        value,
                        quality,
                        f"Priority select: {tag}",
                    )

        return None, QualityFlag.BAD, "No usable sensors in priority list"

    def get_group_health(self, group_name: str) -> Dict[str, Any]:
        """Get health status of a sensor group."""
        with self._lock:
            group = self._sensor_groups.get(group_name)
            health = self._sensor_health.get(group_name, {})

            if not group:
                return {"error": f"Unknown group: {group_name}"}

            healthy_count = sum(1 for v in health.values() if v)
            total_count = len(group.sensor_tags)

            return {
                "group_name": group_name,
                "total_sensors": total_count,
                "healthy_sensors": healthy_count,
                "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
                "sensor_status": health,
                "voting_method": group.voting_method.value,
            }


# =============================================================================
# Factory Functions
# =============================================================================

def create_economizer_validator() -> DataQualityValidator:
    """
    Create a DataQualityValidator pre-configured for economizer sensors.

    Returns:
        Configured DataQualityValidator instance
    """
    validator = DataQualityValidator()

    # Configure feedwater temperature range checks
    validator.configure_range_check(RangeCheck(
        tag_name="ECON.FW.TEMP.INLET",
        low_limit=40.0,
        high_limit=180.0,
        low_low_limit=20.0,
        high_high_limit=200.0,
        engineering_unit="C",
        description="Feedwater inlet temperature",
    ))

    validator.configure_range_check(RangeCheck(
        tag_name="ECON.FW.TEMP.OUTLET",
        low_limit=80.0,
        high_limit=250.0,
        low_low_limit=50.0,
        high_high_limit=280.0,
        engineering_unit="C",
        description="Feedwater outlet temperature",
    ))

    # Configure flue gas temperature range checks
    validator.configure_range_check(RangeCheck(
        tag_name="ECON.FG.TEMP.INLET",
        low_limit=200.0,
        high_limit=450.0,
        low_low_limit=150.0,
        high_high_limit=500.0,
        engineering_unit="C",
        description="Flue gas inlet temperature",
    ))

    validator.configure_range_check(RangeCheck(
        tag_name="ECON.FG.TEMP.OUTLET",
        low_limit=100.0,
        high_limit=300.0,
        low_low_limit=80.0,
        high_high_limit=350.0,
        engineering_unit="C",
        description="Flue gas outlet temperature",
    ))

    # Configure rate-of-change limits for temperatures
    validator.configure_roc_limit(RateOfChangeLimit(
        tag_name="ECON.FW.TEMP.INLET",
        max_rate_per_second=2.0,
        max_rate_per_minute=50.0,
        description="Feedwater inlet temperature ROC",
    ))

    validator.configure_roc_limit(RateOfChangeLimit(
        tag_name="ECON.FG.TEMP.INLET",
        max_rate_per_second=5.0,
        max_rate_per_minute=100.0,
        description="Flue gas inlet temperature ROC",
    ))

    # Configure differential pressure range checks
    validator.configure_range_check(RangeCheck(
        tag_name="ECON.DP.GASSIDE",
        low_limit=0.0,
        high_limit=5.0,
        high_high_limit=8.0,
        engineering_unit="kPa",
        description="Gas side differential pressure",
    ))

    # Configure substitution for critical sensors
    validator.configure_substitution(BadDataSubstitution(
        tag_name="ECON.FW.TEMP.INLET",
        method=SubstitutionMethod.LAST_GOOD_VALUE,
        max_substitution_duration_seconds=300.0,
    ))

    logger.info("Created economizer-specific DataQualityValidator")
    return validator


def create_economizer_redundancy_manager(
    validator: Optional[DataQualityValidator] = None
) -> RedundancyManager:
    """
    Create a RedundancyManager pre-configured for economizer sensors.

    Args:
        validator: Optional DataQualityValidator instance

    Returns:
        Configured RedundancyManager instance
    """
    manager = RedundancyManager(validator=validator)

    # Register redundant temperature sensor groups
    manager.register_group(RedundantSensorGroup(
        group_name="FW_INLET_TEMP",
        sensor_tags=[
            "ECON.FW.TEMP.INLET.A",
            "ECON.FW.TEMP.INLET.B",
            "ECON.FW.TEMP.INLET.C",
        ],
        voting_method=VotingMethod.TWO_OUT_OF_THREE,
        max_deviation_percent=2.0,
        output_tag="ECON.FW.TEMP.INLET",
        description="Feedwater inlet temperature (redundant)",
    ))

    manager.register_group(RedundantSensorGroup(
        group_name="FG_INLET_TEMP",
        sensor_tags=[
            "ECON.FG.TEMP.INLET.A",
            "ECON.FG.TEMP.INLET.B",
            "ECON.FG.TEMP.INLET.C",
        ],
        voting_method=VotingMethod.TWO_OUT_OF_THREE,
        max_deviation_percent=3.0,
        output_tag="ECON.FG.TEMP.INLET",
        description="Flue gas inlet temperature (redundant)",
    ))

    # Register flow rate redundancy group
    manager.register_group(RedundantSensorGroup(
        group_name="FW_FLOW",
        sensor_tags=[
            "ECON.FW.FLOW.A",
            "ECON.FW.FLOW.B",
        ],
        voting_method=VotingMethod.AVERAGE_OF_GOOD,
        max_deviation_percent=5.0,
        output_tag="ECON.FW.FLOW",
        description="Feedwater flow rate (redundant)",
    ))

    logger.info("Created economizer-specific RedundancyManager")
    return manager
