"""
Stream Processor Module - GL-004 BURNMASTER

This module provides real-time stream processing capabilities for combustion data,
including transformations, real-time anomaly detection, alerting, and metric
aggregation.

Key Features:
    - Real-time data transformations
    - Anomaly detection for combustion parameters
    - Alert triggering with severity levels
    - Metric aggregation with windowing
    - Zero-hallucination approach for calculations

Example:
    >>> processor = CombustionStreamProcessor(config)
    >>> processed = await processor.process_stream(data_stream)
    >>> anomalies = processor.detect_anomalies_realtime(data)
    >>> alerts = processor.trigger_alerts(anomalies)

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from .kafka_producer import CombustionData, CombustionDataPoint

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AnomalyType(str, Enum):
    """Types of detected anomalies."""

    THRESHOLD_VIOLATION = "threshold_violation"
    RATE_OF_CHANGE = "rate_of_change"
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    SENSOR_DRIFT = "sensor_drift"
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TransformationType(str, Enum):
    """Data transformation types."""

    NORMALIZE = "normalize"
    SCALE = "scale"
    UNIT_CONVERT = "unit_convert"
    FILTER = "filter"
    SMOOTH = "smooth"
    INTERPOLATE = "interpolate"
    AGGREGATE = "aggregate"


class AggregationWindow(str, Enum):
    """Aggregation window types."""

    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection."""

    tag_id: str = Field(..., description="Tag ID to monitor")
    threshold_min: Optional[float] = Field(None, description="Minimum threshold")
    threshold_max: Optional[float] = Field(None, description="Maximum threshold")
    rate_of_change_limit: Optional[float] = Field(
        None,
        description="Maximum rate of change per second",
    )
    zscore_threshold: float = Field(
        3.0,
        ge=1.0,
        description="Z-score threshold for statistical outlier detection",
    )
    window_size: int = Field(
        100,
        ge=10,
        description="Window size for statistical calculations",
    )
    stale_threshold_seconds: int = Field(
        60,
        ge=1,
        description="Seconds before data is considered stale",
    )


class AlertConfig(BaseModel):
    """Configuration for alert triggering."""

    name: str = Field(..., description="Alert name")
    anomaly_types: List[AnomalyType] = Field(
        default_factory=list,
        description="Anomaly types that trigger this alert",
    )
    severity: AlertSeverity = Field(
        AlertSeverity.WARNING,
        description="Alert severity",
    )
    cooldown_seconds: int = Field(
        60,
        ge=0,
        description="Cooldown period between alerts",
    )
    required_count: int = Field(
        1,
        ge=1,
        description="Required anomaly count to trigger",
    )
    notification_channels: List[str] = Field(
        default_factory=list,
        description="Notification channels",
    )


class StreamProcessorConfig(BaseModel):
    """Configuration for stream processor."""

    processor_id: str = Field(
        default_factory=lambda: f"proc-{uuid.uuid4().hex[:8]}",
        description="Unique processor identifier",
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    anomaly_configs: List[AnomalyDetectionConfig] = Field(
        default_factory=list,
        description="Anomaly detection configurations",
    )
    alert_configs: List[AlertConfig] = Field(
        default_factory=list,
        description="Alert configurations",
    )
    aggregation_windows: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h"],
        description="Aggregation window sizes",
    )
    enable_transformations: bool = Field(
        True,
        description="Enable data transformations",
    )
    buffer_size: int = Field(
        1000,
        ge=100,
        description="Buffer size for historical data",
    )


# =============================================================================
# DATA MODELS
# =============================================================================


class DataStream(BaseModel):
    """Input data stream."""

    stream_id: str = Field(
        default_factory=lambda: f"stream-{uuid.uuid4().hex[:8]}",
        description="Stream identifier",
    )
    source: str = Field(..., description="Stream source")
    data_points: List[CombustionDataPoint] = Field(
        default_factory=list,
        description="Data points in stream",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stream metadata",
    )


class TransformedData(BaseModel):
    """Transformed data point."""

    original: CombustionDataPoint = Field(..., description="Original data point")
    transformed_value: float = Field(..., description="Transformed value")
    transformation: TransformationType = Field(
        ...,
        description="Applied transformation",
    )
    transformation_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation parameters",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Transformation timestamp",
    )


class ProcessedStream(BaseModel):
    """Processed data stream output."""

    stream_id: str = Field(..., description="Original stream ID")
    processor_id: str = Field(..., description="Processor ID")
    data_points: List[TransformedData] = Field(
        default_factory=list,
        description="Processed data points",
    )
    anomalies: List["Anomaly"] = Field(
        default_factory=list,
        description="Detected anomalies",
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed metrics",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Processing timestamp",
    )
    provenance_hash: str = Field("", description="SHA-256 hash for audit")


class Anomaly(BaseModel):
    """Detected anomaly."""

    anomaly_id: str = Field(
        default_factory=lambda: f"anom-{uuid.uuid4().hex[:8]}",
        description="Anomaly identifier",
    )
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    tag_id: str = Field(..., description="Affected tag ID")
    equipment_id: str = Field(..., description="Equipment identifier")
    value: float = Field(..., description="Anomalous value")
    expected_range: Optional[Tuple[float, float]] = Field(
        None,
        description="Expected value range",
    )
    deviation: float = Field(
        0.0,
        description="Deviation from expected",
    )
    severity: AlertSeverity = Field(
        AlertSeverity.WARNING,
        description="Anomaly severity",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )


class AlertResult(BaseModel):
    """Result of alert triggering."""

    alert_id: str = Field(
        default_factory=lambda: f"alert-{uuid.uuid4().hex[:8]}",
        description="Alert identifier",
    )
    alert_name: str = Field(..., description="Alert name")
    severity: AlertSeverity = Field(..., description="Alert severity")
    triggered: bool = Field(..., description="Whether alert was triggered")
    anomalies: List[Anomaly] = Field(
        default_factory=list,
        description="Triggering anomalies",
    )
    message: str = Field("", description="Alert message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp",
    )
    notification_sent: bool = Field(
        False,
        description="Whether notification was sent",
    )
    channels_notified: List[str] = Field(
        default_factory=list,
        description="Notified channels",
    )
    suppressed: bool = Field(
        False,
        description="Whether alert was suppressed (cooldown)",
    )


class AggregatedMetrics(BaseModel):
    """Aggregated metrics for a time window."""

    window_start: datetime = Field(..., description="Window start time")
    window_end: datetime = Field(..., description="Window end time")
    window_size: str = Field(..., description="Window size (e.g., '5m')")
    equipment_id: str = Field(..., description="Equipment identifier")
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics per tag: {tag_id: {metric_name: value}}",
    )
    sample_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Sample counts per tag",
    )
    aggregated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Aggregation timestamp",
    )
    provenance_hash: str = Field("", description="SHA-256 hash for audit")


# =============================================================================
# STREAM PROCESSOR IMPLEMENTATION
# =============================================================================


@dataclass
class TagHistory:
    """Historical data for a tag."""

    tag_id: str
    values: Deque[Tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    last_value: Optional[float] = None
    last_timestamp: Optional[datetime] = None

    def add(self, timestamp: datetime, value: float) -> None:
        """Add a value to history."""
        self.values.append((timestamp, value))
        self.last_value = value
        self.last_timestamp = timestamp

    def get_recent(self, n: int) -> List[float]:
        """Get last n values."""
        return [v for _, v in list(self.values)[-n:]]

    def get_rate_of_change(self) -> Optional[float]:
        """Calculate rate of change (per second)."""
        if len(self.values) < 2:
            return None
        values = list(self.values)
        t1, v1 = values[-2]
        t2, v2 = values[-1]
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            return None
        return (v2 - v1) / dt


class CombustionStreamProcessor:
    """
    Real-time stream processor for combustion data.

    This processor handles data transformations, anomaly detection,
    alerting, and metric aggregation using deterministic calculations
    only (zero-hallucination approach).

    Example:
        >>> config = StreamProcessorConfig(equipment_id="BOILER-001")
        >>> processor = CombustionStreamProcessor(config)
        >>> async for result in processor.process_stream(data_stream):
        ...     print(f"Processed: {len(result.data_points)} points")
    """

    def __init__(self, config: StreamProcessorConfig) -> None:
        """
        Initialize CombustionStreamProcessor.

        Args:
            config: Processor configuration
        """
        self.config = config
        self._tag_history: Dict[str, TagHistory] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}
        self._anomaly_counts: Dict[str, int] = defaultdict(int)
        self._aggregation_buffers: Dict[str, Dict[str, Deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=10000))
        )
        self._processing_count = 0

        # Build anomaly config lookup
        self._anomaly_configs: Dict[str, AnomalyDetectionConfig] = {
            cfg.tag_id: cfg for cfg in config.anomaly_configs
        }

        # Build alert config lookup
        self._alert_configs: Dict[str, AlertConfig] = {
            cfg.name: cfg for cfg in config.alert_configs
        }

        logger.info(
            f"CombustionStreamProcessor initialized: "
            f"processor_id={config.processor_id}, "
            f"equipment_id={config.equipment_id}, "
            f"anomaly_configs={len(config.anomaly_configs)}, "
            f"alert_configs={len(config.alert_configs)}"
        )

    async def process_stream(
        self,
        stream: DataStream,
    ) -> ProcessedStream:
        """
        Process a data stream.

        Args:
            stream: Input data stream

        Returns:
            ProcessedStream with transformed data and detected anomalies
        """
        start_time = time.monotonic()
        self._processing_count += 1

        transformed_data: List[TransformedData] = []
        all_anomalies: List[Anomaly] = []

        for point in stream.data_points:
            # Update history
            self._update_history(point)

            # Apply transformations
            if self.config.enable_transformations:
                transformed = self.apply_transformations(
                    CombustionData(
                        equipment_id=self.config.equipment_id,
                        source_system="stream",
                        points=[point],
                    )
                )
                transformed_data.extend(transformed)

            # Detect anomalies
            anomalies = self.detect_anomalies_realtime(
                CombustionData(
                    equipment_id=self.config.equipment_id,
                    source_system="stream",
                    points=[point],
                )
            )
            all_anomalies.extend(anomalies)

        # Compute metrics
        metrics = self._compute_stream_metrics(stream.data_points)

        processing_time = (time.monotonic() - start_time) * 1000

        result = ProcessedStream(
            stream_id=stream.stream_id,
            processor_id=self.config.processor_id,
            data_points=transformed_data,
            anomalies=all_anomalies,
            metrics=metrics,
            processing_time_ms=processing_time,
        )

        # Compute provenance hash
        content = json.dumps(
            {
                "stream_id": result.stream_id,
                "processor_id": result.processor_id,
                "point_count": len(result.data_points),
                "anomaly_count": len(result.anomalies),
            },
            sort_keys=True,
        )
        object.__setattr__(
            result,
            "provenance_hash",
            hashlib.sha256(content.encode()).hexdigest(),
        )

        logger.debug(
            f"Processed stream {stream.stream_id}: "
            f"{len(transformed_data)} points, {len(all_anomalies)} anomalies, "
            f"{processing_time:.2f}ms"
        )

        return result

    def apply_transformations(
        self,
        data: CombustionData,
    ) -> List[TransformedData]:
        """
        Apply transformations to combustion data.

        All transformations are DETERMINISTIC - no ML or LLM involved.

        Args:
            data: Input combustion data

        Returns:
            List of transformed data points
        """
        transformed: List[TransformedData] = []

        for point in data.points:
            # Normalization transformation
            history = self._tag_history.get(point.tag_id)
            if history and len(history.values) >= 10:
                values = history.get_recent(100)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 1.0

                if std > 0:
                    normalized_value = (point.value - mean) / std
                else:
                    normalized_value = 0.0

                transformed.append(
                    TransformedData(
                        original=point,
                        transformed_value=normalized_value,
                        transformation=TransformationType.NORMALIZE,
                        transformation_params={"mean": mean, "std": std},
                    )
                )

            # Smoothing transformation (exponential moving average)
            if history and history.last_value is not None:
                alpha = 0.3  # Smoothing factor
                smoothed_value = alpha * point.value + (1 - alpha) * history.last_value

                transformed.append(
                    TransformedData(
                        original=point,
                        transformed_value=smoothed_value,
                        transformation=TransformationType.SMOOTH,
                        transformation_params={"alpha": alpha},
                    )
                )

        return transformed

    def detect_anomalies_realtime(
        self,
        data: CombustionData,
    ) -> List[Anomaly]:
        """
        Detect anomalies in real-time using DETERMINISTIC methods.

        No ML or LLM involved - uses threshold, statistical, and rate-based
        detection only.

        Args:
            data: Input combustion data

        Returns:
            List of detected anomalies
        """
        anomalies: List[Anomaly] = []

        for point in data.points:
            config = self._anomaly_configs.get(point.tag_id)
            if not config:
                continue

            history = self._tag_history.get(point.tag_id)

            # Threshold violation check
            if config.threshold_min is not None and point.value < config.threshold_min:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                        tag_id=point.tag_id,
                        equipment_id=data.equipment_id,
                        value=point.value,
                        expected_range=(config.threshold_min, config.threshold_max or float("inf")),
                        deviation=config.threshold_min - point.value,
                        severity=self._determine_severity(
                            config.threshold_min - point.value,
                            config.threshold_min,
                        ),
                        context={"violation_type": "below_minimum"},
                    )
                )

            if config.threshold_max is not None and point.value > config.threshold_max:
                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.THRESHOLD_VIOLATION,
                        tag_id=point.tag_id,
                        equipment_id=data.equipment_id,
                        value=point.value,
                        expected_range=(config.threshold_min or float("-inf"), config.threshold_max),
                        deviation=point.value - config.threshold_max,
                        severity=self._determine_severity(
                            point.value - config.threshold_max,
                            config.threshold_max,
                        ),
                        context={"violation_type": "above_maximum"},
                    )
                )

            # Rate of change check
            if config.rate_of_change_limit and history:
                rate = history.get_rate_of_change()
                if rate is not None and abs(rate) > config.rate_of_change_limit:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.RATE_OF_CHANGE,
                            tag_id=point.tag_id,
                            equipment_id=data.equipment_id,
                            value=point.value,
                            deviation=abs(rate) - config.rate_of_change_limit,
                            severity=AlertSeverity.WARNING,
                            context={
                                "rate": rate,
                                "limit": config.rate_of_change_limit,
                            },
                        )
                    )

            # Statistical outlier check (z-score)
            if history and len(history.values) >= config.window_size:
                values = history.get_recent(config.window_size)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0

                if std > 0:
                    zscore = abs((point.value - mean) / std)
                    if zscore > config.zscore_threshold:
                        anomalies.append(
                            Anomaly(
                                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                                tag_id=point.tag_id,
                                equipment_id=data.equipment_id,
                                value=point.value,
                                deviation=zscore,
                                severity=self._severity_from_zscore(zscore),
                                context={
                                    "zscore": zscore,
                                    "mean": mean,
                                    "std": std,
                                },
                            )
                        )

            # Stale data check
            if history and history.last_timestamp:
                staleness = (datetime.now(timezone.utc) - history.last_timestamp).total_seconds()
                if staleness > config.stale_threshold_seconds:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.STALE_DATA,
                            tag_id=point.tag_id,
                            equipment_id=data.equipment_id,
                            value=point.value,
                            deviation=staleness - config.stale_threshold_seconds,
                            severity=AlertSeverity.WARNING,
                            context={
                                "staleness_seconds": staleness,
                                "threshold": config.stale_threshold_seconds,
                            },
                        )
                    )

        return anomalies

    def trigger_alerts(
        self,
        anomalies: List[Anomaly],
    ) -> AlertResult:
        """
        Trigger alerts based on detected anomalies.

        Args:
            anomalies: List of detected anomalies

        Returns:
            AlertResult with triggering status
        """
        if not anomalies:
            return AlertResult(
                alert_name="no_anomalies",
                severity=AlertSeverity.INFO,
                triggered=False,
                message="No anomalies detected",
            )

        # Group anomalies by type
        anomaly_by_type: Dict[AnomalyType, List[Anomaly]] = defaultdict(list)
        for anomaly in anomalies:
            anomaly_by_type[anomaly.anomaly_type].append(anomaly)

        # Check each alert configuration
        triggered_alerts: List[AlertResult] = []

        for alert_config in self.config.alert_configs:
            matching_anomalies: List[Anomaly] = []

            for anomaly_type in alert_config.anomaly_types:
                matching_anomalies.extend(anomaly_by_type.get(anomaly_type, []))

            if len(matching_anomalies) >= alert_config.required_count:
                # Check cooldown
                last_triggered = self._alert_cooldowns.get(alert_config.name)
                now = datetime.now(timezone.utc)

                if last_triggered and (now - last_triggered).total_seconds() < alert_config.cooldown_seconds:
                    triggered_alerts.append(
                        AlertResult(
                            alert_name=alert_config.name,
                            severity=alert_config.severity,
                            triggered=False,
                            anomalies=matching_anomalies,
                            message="Alert suppressed due to cooldown",
                            suppressed=True,
                        )
                    )
                    continue

                # Trigger alert
                self._alert_cooldowns[alert_config.name] = now

                message = (
                    f"Alert triggered: {alert_config.name} - "
                    f"{len(matching_anomalies)} anomalies detected"
                )

                # Simulate notification (in production, actually send)
                channels_notified = alert_config.notification_channels

                triggered_alerts.append(
                    AlertResult(
                        alert_name=alert_config.name,
                        severity=alert_config.severity,
                        triggered=True,
                        anomalies=matching_anomalies,
                        message=message,
                        notification_sent=len(channels_notified) > 0,
                        channels_notified=channels_notified,
                    )
                )

                logger.warning(
                    f"Alert triggered: {alert_config.name}, "
                    f"severity={alert_config.severity.value}, "
                    f"anomalies={len(matching_anomalies)}"
                )

        # Return the highest severity alert or a summary
        if triggered_alerts:
            # Sort by severity
            severity_order = {
                AlertSeverity.EMERGENCY: 0,
                AlertSeverity.CRITICAL: 1,
                AlertSeverity.WARNING: 2,
                AlertSeverity.INFO: 3,
            }
            triggered_alerts.sort(key=lambda a: severity_order.get(a.severity, 4))
            return triggered_alerts[0]

        return AlertResult(
            alert_name="no_matching_alerts",
            severity=AlertSeverity.INFO,
            triggered=False,
            anomalies=anomalies,
            message=f"{len(anomalies)} anomalies detected but no alert criteria met",
        )

    def aggregate_metrics(
        self,
        window: str,
    ) -> AggregatedMetrics:
        """
        Aggregate metrics over a time window.

        Uses DETERMINISTIC calculations only.

        Args:
            window: Window size (e.g., "5m", "1h")

        Returns:
            AggregatedMetrics with computed statistics
        """
        now = datetime.now(timezone.utc)

        # Parse window
        value = int(window[:-1])
        unit = window[-1].lower()

        if unit == "s":
            window_seconds = value
        elif unit == "m":
            window_seconds = value * 60
        elif unit == "h":
            window_seconds = value * 3600
        else:
            window_seconds = value * 60

        window_start = now - timedelta(seconds=window_seconds)

        metrics: Dict[str, Dict[str, float]] = {}
        sample_counts: Dict[str, int] = {}

        for tag_id, history in self._tag_history.items():
            # Get values in window
            values_in_window = [
                v for ts, v in history.values
                if ts >= window_start
            ]

            if not values_in_window:
                continue

            sample_counts[tag_id] = len(values_in_window)

            # Compute metrics using DETERMINISTIC calculations
            metrics[tag_id] = {
                "mean": statistics.mean(values_in_window),
                "min": min(values_in_window),
                "max": max(values_in_window),
                "std": statistics.stdev(values_in_window) if len(values_in_window) > 1 else 0.0,
                "count": float(len(values_in_window)),
                "sum": sum(values_in_window),
                "range": max(values_in_window) - min(values_in_window),
            }

            if len(values_in_window) > 1:
                metrics[tag_id]["median"] = statistics.median(values_in_window)

        result = AggregatedMetrics(
            window_start=window_start,
            window_end=now,
            window_size=window,
            equipment_id=self.config.equipment_id,
            metrics=metrics,
            sample_counts=sample_counts,
        )

        # Compute provenance hash
        content = json.dumps(
            {
                "window_start": result.window_start.isoformat(),
                "window_end": result.window_end.isoformat(),
                "equipment_id": result.equipment_id,
                "tag_count": len(metrics),
            },
            sort_keys=True,
        )
        object.__setattr__(
            result,
            "provenance_hash",
            hashlib.sha256(content.encode()).hexdigest(),
        )

        logger.debug(
            f"Aggregated metrics for window {window}: "
            f"{len(metrics)} tags, {sum(sample_counts.values())} samples"
        )

        return result

    def _update_history(self, point: CombustionDataPoint) -> None:
        """Update tag history with new data point."""
        if point.tag_id not in self._tag_history:
            self._tag_history[point.tag_id] = TagHistory(tag_id=point.tag_id)

        self._tag_history[point.tag_id].add(point.timestamp, point.value)

    def _compute_stream_metrics(
        self,
        points: List[CombustionDataPoint],
    ) -> Dict[str, float]:
        """Compute basic stream metrics."""
        if not points:
            return {}

        values = [p.value for p in points]

        return {
            "point_count": float(len(points)),
            "value_mean": statistics.mean(values),
            "value_min": min(values),
            "value_max": max(values),
            "value_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _determine_severity(
        self,
        deviation: float,
        threshold: float,
    ) -> AlertSeverity:
        """Determine severity based on deviation percentage."""
        if threshold == 0:
            return AlertSeverity.WARNING

        deviation_pct = abs(deviation / threshold) * 100

        if deviation_pct > 50:
            return AlertSeverity.EMERGENCY
        elif deviation_pct > 25:
            return AlertSeverity.CRITICAL
        elif deviation_pct > 10:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _severity_from_zscore(self, zscore: float) -> AlertSeverity:
        """Determine severity from z-score."""
        if zscore > 5:
            return AlertSeverity.EMERGENCY
        elif zscore > 4:
            return AlertSeverity.CRITICAL
        elif zscore > 3:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def get_tag_history(self, tag_id: str) -> Optional[TagHistory]:
        """Get history for a specific tag."""
        return self._tag_history.get(tag_id)

    def clear_history(self) -> None:
        """Clear all tag histories."""
        self._tag_history.clear()
        logger.info("Tag histories cleared")

    @property
    def processing_count(self) -> int:
        """Return total processing count."""
        return self._processing_count
