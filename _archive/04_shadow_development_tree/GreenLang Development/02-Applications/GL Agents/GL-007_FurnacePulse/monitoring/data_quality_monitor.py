"""
DataQualityMonitor - Sensor Signal Quality Monitoring for FurnacePulse

This module implements the DataQualityMonitor for tracking signal quality,
detecting drift and stuck values, measuring completeness, and monitoring
data pipeline latency.

Signal Quality Flags:
    - GOOD: Signal within normal operating parameters
    - BAD: Signal failed quality checks
    - SUSPECT: Signal quality questionable, requires review
    - MISSING: Signal not received or null

Example:
    >>> config = DataQualityMonitorConfig()
    >>> monitor = DataQualityMonitor(config)
    >>> quality = monitor.assess_signal_quality(sensor_tag, readings)
    >>> if quality.flag != SignalQualityFlag.GOOD:
    ...     print(f"Quality issue: {quality.issues}")
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SignalQualityFlag(str, Enum):
    """Signal quality flag values."""

    GOOD = "GOOD"
    BAD = "BAD"
    SUSPECT = "SUSPECT"
    MISSING = "MISSING"


class DriftType(str, Enum):
    """Types of drift detected in signals."""

    NONE = "NONE"
    GRADUAL = "GRADUAL"
    SUDDEN = "SUDDEN"
    OSCILLATING = "OSCILLATING"
    BIAS = "BIAS"


class StuckType(str, Enum):
    """Types of stuck value conditions."""

    NONE = "NONE"
    CONSTANT = "CONSTANT"
    RATE_LIMITED = "RATE_LIMITED"
    QUANTIZED = "QUANTIZED"


@dataclass
class SensorReading:
    """A single sensor reading with timestamp."""

    timestamp: datetime
    value: float
    raw_value: Optional[float] = None
    quality_code: int = 192  # Default OPC good quality
    source: str = ""


@dataclass
class SignalStatistics:
    """Statistical summary of a signal window."""

    mean: float
    std_dev: float
    min_value: float
    max_value: float
    range_value: float
    count: int
    missing_count: int
    completeness: float
    first_timestamp: datetime
    last_timestamp: datetime


class DriftResult(BaseModel):
    """Result of drift detection analysis."""

    drift_type: DriftType = Field(default=DriftType.NONE)
    drift_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Drift severity 0-1")
    drift_direction: Optional[str] = Field(None, description="UP, DOWN, or OSCILLATING")
    drift_rate: Optional[float] = Field(None, description="Rate of drift per hour")
    baseline_mean: Optional[float] = Field(None)
    current_mean: Optional[float] = Field(None)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_time: datetime = Field(default_factory=datetime.utcnow)
    details: str = Field(default="")


class StuckResult(BaseModel):
    """Result of stuck value detection analysis."""

    stuck_type: StuckType = Field(default=StuckType.NONE)
    is_stuck: bool = Field(default=False)
    stuck_value: Optional[float] = Field(None)
    stuck_duration_seconds: float = Field(default=0.0)
    variance: float = Field(default=0.0)
    unique_value_count: int = Field(default=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_time: datetime = Field(default_factory=datetime.utcnow)
    details: str = Field(default="")


class SignalQuality(BaseModel):
    """Complete signal quality assessment."""

    sensor_tag: str = Field(..., description="Sensor tag identifier")
    flag: SignalQualityFlag = Field(..., description="Overall quality flag")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    completeness_percent: float = Field(default=100.0, ge=0.0, le=100.0)
    drift_result: Optional[DriftResult] = Field(None)
    stuck_result: Optional[StuckResult] = Field(None)
    latency_seconds: float = Field(default=0.0)
    statistics: Optional[SignalStatistics] = Field(None)
    issues: List[str] = Field(default_factory=list)
    score: float = Field(default=100.0, ge=0.0, le=100.0, description="Quality score 0-100")
    provenance_hash: str = Field(default="")


class SensorConfig(BaseModel):
    """Configuration for a specific sensor."""

    sensor_tag: str = Field(..., description="Sensor tag identifier")
    description: str = Field(default="")
    unit: str = Field(default="")
    expected_min: float = Field(default=float("-inf"))
    expected_max: float = Field(default=float("inf"))
    expected_rate_of_change_max: float = Field(
        default=float("inf"), description="Max expected change per minute"
    )
    stuck_threshold_seconds: float = Field(
        default=300.0, description="Seconds of no change to flag stuck"
    )
    stuck_variance_threshold: float = Field(
        default=0.001, description="Variance threshold for stuck detection"
    )
    drift_baseline_window_hours: float = Field(
        default=24.0, description="Hours of data for baseline"
    )
    drift_detection_window_hours: float = Field(
        default=2.0, description="Hours of data for drift detection"
    )
    drift_threshold_sigma: float = Field(
        default=3.0, description="Standard deviations for drift alert"
    )
    latency_warning_seconds: float = Field(
        default=60.0, description="Latency threshold for warning"
    )
    latency_critical_seconds: float = Field(
        default=300.0, description="Latency threshold for critical"
    )
    is_critical: bool = Field(default=False, description="Whether sensor is safety-critical")


class DataQualityMonitorConfig(BaseModel):
    """Configuration for DataQualityMonitor."""

    default_window_size: int = Field(
        default=1000, ge=100, description="Default readings to retain per sensor"
    )
    default_stuck_threshold_seconds: float = Field(default=300.0, ge=30.0)
    default_stuck_variance_threshold: float = Field(default=0.001, ge=0.0)
    default_drift_threshold_sigma: float = Field(default=3.0, ge=1.0)
    default_latency_warning_seconds: float = Field(default=60.0, ge=5.0)
    default_latency_critical_seconds: float = Field(default=300.0, ge=30.0)
    completeness_good_threshold: float = Field(
        default=95.0, ge=50.0, le=100.0, description="Threshold for GOOD completeness"
    )
    completeness_suspect_threshold: float = Field(
        default=80.0, ge=0.0, le=100.0, description="Threshold for SUSPECT completeness"
    )
    enable_drift_detection: bool = Field(default=True)
    enable_stuck_detection: bool = Field(default=True)
    baseline_retention_hours: int = Field(default=168, ge=24, description="Hours to retain baseline")
    report_interval_seconds: int = Field(default=60, ge=10)


@dataclass
class SensorState:
    """Internal state tracking for a sensor."""

    sensor_tag: str
    config: SensorConfig
    readings: Deque[SensorReading]
    baseline_readings: Deque[SensorReading]
    last_reading: Optional[SensorReading] = None
    last_quality: Optional[SignalQuality] = None
    last_nonzero_change: Optional[datetime] = None
    stuck_start_time: Optional[datetime] = None
    consecutive_stuck_readings: int = 0
    drift_alert_active: bool = False
    quality_history: Deque[SignalQualityFlag] = field(default_factory=lambda: deque(maxlen=100))


class DataQualityMonitor:
    """
    Data quality monitoring for FurnacePulse sensor signals.

    This class tracks signal quality including completeness, drift detection,
    stuck value detection, and latency monitoring for all configured sensors.

    Attributes:
        config: Monitor configuration
        sensor_states: State tracking for each sensor
        sensor_configs: Configuration for each sensor
        quality_callbacks: Callbacks for quality events

    Example:
        >>> config = DataQualityMonitorConfig()
        >>> monitor = DataQualityMonitor(config)
        >>> monitor.register_sensor(SensorConfig(sensor_tag="TMT-101"))
        >>> monitor.add_reading("TMT-101", SensorReading(timestamp=now, value=850.0))
        >>> quality = monitor.get_quality("TMT-101")
    """

    def __init__(self, config: DataQualityMonitorConfig):
        """
        Initialize DataQualityMonitor.

        Args:
            config: Monitor configuration
        """
        self.config = config
        self.sensor_states: Dict[str, SensorState] = {}
        self.sensor_configs: Dict[str, SensorConfig] = {}
        self._quality_callbacks: List[callable] = []

        logger.info("DataQualityMonitor initialized")

    def register_sensor(self, sensor_config: SensorConfig) -> None:
        """
        Register a sensor for monitoring.

        Args:
            sensor_config: Configuration for the sensor
        """
        sensor_tag = sensor_config.sensor_tag

        self.sensor_configs[sensor_tag] = sensor_config
        self.sensor_states[sensor_tag] = SensorState(
            sensor_tag=sensor_tag,
            config=sensor_config,
            readings=deque(maxlen=self.config.default_window_size),
            baseline_readings=deque(maxlen=int(
                sensor_config.drift_baseline_window_hours * 3600 / 5  # Assume 5s interval
            )),
        )

        logger.debug("Registered sensor for monitoring: %s", sensor_tag)

    def add_reading(self, sensor_tag: str, reading: SensorReading) -> SignalQuality:
        """
        Add a sensor reading and assess quality.

        Args:
            sensor_tag: Sensor tag identifier
            reading: The sensor reading

        Returns:
            SignalQuality assessment for the reading
        """
        # Auto-register if not known
        if sensor_tag not in self.sensor_states:
            self.register_sensor(SensorConfig(sensor_tag=sensor_tag))

        state = self.sensor_states[sensor_tag]

        # Add to readings window
        state.readings.append(reading)
        state.baseline_readings.append(reading)
        state.last_reading = reading

        # Assess quality
        quality = self._assess_quality(state, reading)
        state.last_quality = quality
        state.quality_history.append(quality.flag)

        # Trigger callbacks for non-GOOD quality
        if quality.flag != SignalQualityFlag.GOOD:
            self._trigger_quality_callbacks(quality)

        return quality

    def add_readings_batch(
        self,
        sensor_tag: str,
        readings: List[SensorReading],
    ) -> SignalQuality:
        """
        Add multiple readings in batch.

        Args:
            sensor_tag: Sensor tag identifier
            readings: List of readings to add

        Returns:
            SignalQuality assessment for the batch
        """
        if not readings:
            return SignalQuality(
                sensor_tag=sensor_tag,
                flag=SignalQualityFlag.MISSING,
                issues=["No readings provided"],
                score=0.0,
            )

        # Add all readings
        for reading in readings[:-1]:
            self.add_reading(sensor_tag, reading)

        # Return quality for the last reading
        return self.add_reading(sensor_tag, readings[-1])

    def _assess_quality(
        self,
        state: SensorState,
        reading: SensorReading,
    ) -> SignalQuality:
        """
        Assess quality for a reading.

        Args:
            state: Sensor state
            reading: Current reading

        Returns:
            SignalQuality assessment
        """
        start_time = datetime.utcnow()
        issues = []
        score = 100.0
        flag = SignalQualityFlag.GOOD

        config = state.config

        # Check for missing/null value
        if reading.value is None or math.isnan(reading.value):
            return SignalQuality(
                sensor_tag=state.sensor_tag,
                flag=SignalQualityFlag.MISSING,
                issues=["Value is null or NaN"],
                score=0.0,
                provenance_hash=self._calculate_provenance_hash(state.sensor_tag, reading),
            )

        # Check value range
        if reading.value < config.expected_min:
            issues.append(f"Value {reading.value} below expected minimum {config.expected_min}")
            score -= 30
            flag = SignalQualityFlag.SUSPECT

        if reading.value > config.expected_max:
            issues.append(f"Value {reading.value} above expected maximum {config.expected_max}")
            score -= 30
            flag = SignalQualityFlag.SUSPECT

        # Check OPC quality code
        if reading.quality_code < 192:  # OPC quality < GOOD
            issues.append(f"OPC quality code indicates bad quality: {reading.quality_code}")
            score -= 40
            flag = SignalQualityFlag.BAD

        # Check rate of change
        if state.last_reading and len(state.readings) >= 2:
            prev_reading = list(state.readings)[-2]
            time_diff = (reading.timestamp - prev_reading.timestamp).total_seconds()
            if time_diff > 0:
                rate = abs(reading.value - prev_reading.value) / (time_diff / 60.0)
                if rate > config.expected_rate_of_change_max:
                    issues.append(
                        f"Rate of change {rate:.2f}/min exceeds max {config.expected_rate_of_change_max}"
                    )
                    score -= 20
                    if flag == SignalQualityFlag.GOOD:
                        flag = SignalQualityFlag.SUSPECT

        # Calculate statistics
        stats = self._calculate_statistics(state)

        # Check completeness
        completeness = stats.completeness if stats else 100.0
        if completeness < self.config.completeness_suspect_threshold:
            issues.append(f"Low completeness: {completeness:.1f}%")
            score -= 30
            flag = SignalQualityFlag.BAD
        elif completeness < self.config.completeness_good_threshold:
            issues.append(f"Reduced completeness: {completeness:.1f}%")
            score -= 15
            if flag == SignalQualityFlag.GOOD:
                flag = SignalQualityFlag.SUSPECT

        # Check latency
        latency = (datetime.utcnow() - reading.timestamp).total_seconds()
        if latency > config.latency_critical_seconds:
            issues.append(f"Critical latency: {latency:.1f}s")
            score -= 40
            flag = SignalQualityFlag.BAD
        elif latency > config.latency_warning_seconds:
            issues.append(f"Elevated latency: {latency:.1f}s")
            score -= 15
            if flag == SignalQualityFlag.GOOD:
                flag = SignalQualityFlag.SUSPECT

        # Drift detection
        drift_result = None
        if self.config.enable_drift_detection and len(state.baseline_readings) > 100:
            drift_result = self._detect_drift(state)
            if drift_result.drift_type != DriftType.NONE:
                issues.append(f"Drift detected: {drift_result.drift_type.value} ({drift_result.details})")
                score -= 25
                if flag == SignalQualityFlag.GOOD:
                    flag = SignalQualityFlag.SUSPECT

        # Stuck value detection
        stuck_result = None
        if self.config.enable_stuck_detection and len(state.readings) > 10:
            stuck_result = self._detect_stuck(state)
            if stuck_result.is_stuck:
                issues.append(
                    f"Stuck value detected: {stuck_result.stuck_type.value} "
                    f"(value={stuck_result.stuck_value}, duration={stuck_result.stuck_duration_seconds:.0f}s)"
                )
                score -= 40
                flag = SignalQualityFlag.BAD

        # Ensure score is in valid range
        score = max(0.0, min(100.0, score))

        # Override flag based on score
        if score < 30:
            flag = SignalQualityFlag.BAD
        elif score < 60 and flag == SignalQualityFlag.GOOD:
            flag = SignalQualityFlag.SUSPECT

        quality = SignalQuality(
            sensor_tag=state.sensor_tag,
            flag=flag,
            completeness_percent=completeness,
            drift_result=drift_result,
            stuck_result=stuck_result,
            latency_seconds=latency,
            statistics=stats,
            issues=issues,
            score=score,
            provenance_hash=self._calculate_provenance_hash(state.sensor_tag, reading),
        )

        return quality

    def _calculate_statistics(self, state: SensorState) -> Optional[SignalStatistics]:
        """Calculate statistics for the current reading window."""
        if len(state.readings) < 2:
            return None

        values = [r.value for r in state.readings if r.value is not None and not math.isnan(r.value)]

        if len(values) < 2:
            return None

        timestamps = [r.timestamp for r in state.readings]

        return SignalStatistics(
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            range_value=max(values) - min(values),
            count=len(values),
            missing_count=len(state.readings) - len(values),
            completeness=(len(values) / len(state.readings)) * 100 if state.readings else 100.0,
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps),
        )

    def _detect_drift(self, state: SensorState) -> DriftResult:
        """
        Detect drift in sensor readings.

        Uses statistical comparison between baseline window and recent window.
        """
        config = state.config

        # Get baseline values (older readings)
        baseline_values = [
            r.value for r in list(state.baseline_readings)[:-len(state.readings)]
            if r.value is not None and not math.isnan(r.value)
        ]

        # Get recent values
        recent_values = [
            r.value for r in state.readings
            if r.value is not None and not math.isnan(r.value)
        ]

        if len(baseline_values) < 50 or len(recent_values) < 10:
            return DriftResult(drift_type=DriftType.NONE, details="Insufficient data for drift detection")

        baseline_mean = statistics.mean(baseline_values)
        baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0.001
        recent_mean = statistics.mean(recent_values)

        # Calculate z-score for drift
        z_score = abs(recent_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0

        if z_score < config.drift_threshold_sigma:
            return DriftResult(
                drift_type=DriftType.NONE,
                drift_score=min(1.0, z_score / config.drift_threshold_sigma),
                baseline_mean=baseline_mean,
                current_mean=recent_mean,
                confidence=0.8,
                details=f"Z-score {z_score:.2f} below threshold {config.drift_threshold_sigma}",
            )

        # Determine drift direction
        drift_direction = "UP" if recent_mean > baseline_mean else "DOWN"

        # Calculate drift rate
        time_span_hours = (
            state.readings[-1].timestamp - state.readings[0].timestamp
        ).total_seconds() / 3600 if len(state.readings) > 1 else 1

        drift_rate = abs(recent_mean - baseline_mean) / time_span_hours if time_span_hours > 0 else 0

        # Determine drift type
        recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        if recent_std > baseline_std * 2:
            drift_type = DriftType.OSCILLATING
        elif z_score > config.drift_threshold_sigma * 2:
            drift_type = DriftType.SUDDEN
        else:
            drift_type = DriftType.GRADUAL

        return DriftResult(
            drift_type=drift_type,
            drift_score=min(1.0, z_score / (config.drift_threshold_sigma * 2)),
            drift_direction=drift_direction,
            drift_rate=drift_rate,
            baseline_mean=baseline_mean,
            current_mean=recent_mean,
            confidence=min(0.95, 0.5 + (len(recent_values) / 100)),
            details=f"Mean shifted from {baseline_mean:.2f} to {recent_mean:.2f} (z={z_score:.2f})",
        )

    def _detect_stuck(self, state: SensorState) -> StuckResult:
        """
        Detect stuck value conditions.

        Checks for constant values, rate-limited behavior, and quantization.
        """
        config = state.config

        recent_readings = list(state.readings)[-100:]  # Last 100 readings
        if len(recent_readings) < 10:
            return StuckResult(stuck_type=StuckType.NONE, details="Insufficient data")

        values = [r.value for r in recent_readings if r.value is not None]
        if len(values) < 10:
            return StuckResult(stuck_type=StuckType.NONE, details="Insufficient valid values")

        # Calculate variance
        variance = statistics.variance(values) if len(values) > 1 else 0

        # Count unique values
        unique_values = len(set(round(v, 6) for v in values))

        # Time span
        time_span = (
            recent_readings[-1].timestamp - recent_readings[0].timestamp
        ).total_seconds()

        # Check for constant value (very low variance)
        if variance < config.stuck_variance_threshold:
            stuck_value = statistics.mean(values)

            # Track stuck duration
            if state.stuck_start_time is None:
                state.stuck_start_time = recent_readings[0].timestamp
            stuck_duration = (datetime.utcnow() - state.stuck_start_time).total_seconds()

            if stuck_duration >= config.stuck_threshold_seconds:
                return StuckResult(
                    stuck_type=StuckType.CONSTANT,
                    is_stuck=True,
                    stuck_value=stuck_value,
                    stuck_duration_seconds=stuck_duration,
                    variance=variance,
                    unique_value_count=unique_values,
                    confidence=0.95,
                    details=f"Constant value {stuck_value:.2f} for {stuck_duration:.0f}s",
                )
        else:
            # Reset stuck tracking
            state.stuck_start_time = None

        # Check for quantized/step behavior (few unique values)
        if unique_values <= 3 and time_span > 60:
            return StuckResult(
                stuck_type=StuckType.QUANTIZED,
                is_stuck=True,
                stuck_value=statistics.mode(values) if values else None,
                stuck_duration_seconds=time_span,
                variance=variance,
                unique_value_count=unique_values,
                confidence=0.8,
                details=f"Only {unique_values} unique values in {time_span:.0f}s window",
            )

        # Check for rate-limited behavior (changes happen but slowly)
        if unique_values > 3:
            sorted_values = sorted(set(values))
            min_step = min(
                abs(sorted_values[i + 1] - sorted_values[i])
                for i in range(len(sorted_values) - 1)
            ) if len(sorted_values) > 1 else 0

            if min_step > 0:
                expected_changes = time_span / 5  # Assume 5s update rate
                actual_changes = unique_values - 1
                if actual_changes < expected_changes * 0.1:
                    return StuckResult(
                        stuck_type=StuckType.RATE_LIMITED,
                        is_stuck=True,
                        stuck_value=values[-1],
                        stuck_duration_seconds=time_span,
                        variance=variance,
                        unique_value_count=unique_values,
                        confidence=0.7,
                        details=f"Only {actual_changes} changes in {time_span:.0f}s (expected ~{expected_changes:.0f})",
                    )

        return StuckResult(
            stuck_type=StuckType.NONE,
            is_stuck=False,
            variance=variance,
            unique_value_count=unique_values,
            confidence=0.9,
            details="Signal showing normal variation",
        )

    def _calculate_provenance_hash(
        self,
        sensor_tag: str,
        reading: SensorReading,
    ) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = f"{sensor_tag}|{reading.timestamp.isoformat()}|{reading.value}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _trigger_quality_callbacks(self, quality: SignalQuality) -> None:
        """Trigger registered callbacks for quality events."""
        for callback in self._quality_callbacks:
            try:
                callback(quality)
            except Exception as e:
                logger.error("Quality callback failed: %s", e)

    def register_quality_callback(self, callback: callable) -> None:
        """Register a callback for quality events."""
        self._quality_callbacks.append(callback)

    def get_quality(self, sensor_tag: str) -> Optional[SignalQuality]:
        """
        Get the latest quality assessment for a sensor.

        Args:
            sensor_tag: Sensor tag identifier

        Returns:
            Latest SignalQuality or None if sensor not found
        """
        if sensor_tag not in self.sensor_states:
            return None
        return self.sensor_states[sensor_tag].last_quality

    def get_all_quality_flags(self) -> Dict[str, SignalQualityFlag]:
        """
        Get current quality flags for all sensors.

        Returns:
            Dictionary mapping sensor tags to quality flags
        """
        return {
            tag: state.last_quality.flag if state.last_quality else SignalQualityFlag.MISSING
            for tag, state in self.sensor_states.items()
        }

    def get_sensors_by_quality(
        self,
        flag: SignalQualityFlag,
    ) -> List[str]:
        """
        Get sensors with a specific quality flag.

        Args:
            flag: Quality flag to filter by

        Returns:
            List of sensor tags with the specified flag
        """
        return [
            tag for tag, state in self.sensor_states.items()
            if state.last_quality and state.last_quality.flag == flag
        ]

    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Get a summary of quality across all sensors.

        Returns:
            Dictionary with quality statistics
        """
        total_sensors = len(self.sensor_states)
        if total_sensors == 0:
            return {
                "total_sensors": 0,
                "by_flag": {},
                "average_score": 0.0,
                "average_completeness": 0.0,
                "sensors_with_drift": 0,
                "sensors_stuck": 0,
            }

        by_flag = defaultdict(int)
        scores = []
        completeness_values = []
        drift_count = 0
        stuck_count = 0

        for state in self.sensor_states.values():
            if state.last_quality:
                by_flag[state.last_quality.flag.value] += 1
                scores.append(state.last_quality.score)
                completeness_values.append(state.last_quality.completeness_percent)

                if state.last_quality.drift_result and state.last_quality.drift_result.drift_type != DriftType.NONE:
                    drift_count += 1

                if state.last_quality.stuck_result and state.last_quality.stuck_result.is_stuck:
                    stuck_count += 1
            else:
                by_flag[SignalQualityFlag.MISSING.value] += 1

        return {
            "total_sensors": total_sensors,
            "by_flag": dict(by_flag),
            "average_score": statistics.mean(scores) if scores else 0.0,
            "average_completeness": statistics.mean(completeness_values) if completeness_values else 0.0,
            "sensors_with_drift": drift_count,
            "sensors_stuck": stuck_count,
            "good_percent": (by_flag.get(SignalQualityFlag.GOOD.value, 0) / total_sensors) * 100,
        }

    def get_drift_alerts(self) -> List[Tuple[str, DriftResult]]:
        """
        Get all active drift alerts.

        Returns:
            List of (sensor_tag, DriftResult) tuples for sensors with drift
        """
        alerts = []
        for tag, state in self.sensor_states.items():
            if state.last_quality and state.last_quality.drift_result:
                if state.last_quality.drift_result.drift_type != DriftType.NONE:
                    alerts.append((tag, state.last_quality.drift_result))
        return alerts

    def get_stuck_alerts(self) -> List[Tuple[str, StuckResult]]:
        """
        Get all active stuck value alerts.

        Returns:
            List of (sensor_tag, StuckResult) tuples for stuck sensors
        """
        alerts = []
        for tag, state in self.sensor_states.items():
            if state.last_quality and state.last_quality.stuck_result:
                if state.last_quality.stuck_result.is_stuck:
                    alerts.append((tag, state.last_quality.stuck_result))
        return alerts

    def get_latency_statistics(self) -> Dict[str, float]:
        """
        Get latency statistics across all sensors.

        Returns:
            Dictionary with latency statistics
        """
        latencies = [
            state.last_quality.latency_seconds
            for state in self.sensor_states.values()
            if state.last_quality
        ]

        if not latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
            }

        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)

        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1],
        }

    def cleanup_old_data(self) -> int:
        """
        Clean up old baseline data beyond retention period.

        Returns:
            Number of readings cleaned up
        """
        cutoff = datetime.utcnow() - timedelta(hours=self.config.baseline_retention_hours)
        cleaned = 0

        for state in self.sensor_states.values():
            original_len = len(state.baseline_readings)
            state.baseline_readings = deque(
                [r for r in state.baseline_readings if r.timestamp > cutoff],
                maxlen=state.baseline_readings.maxlen,
            )
            cleaned += original_len - len(state.baseline_readings)

        if cleaned > 0:
            logger.debug("Cleaned up %d old baseline readings", cleaned)

        return cleaned
