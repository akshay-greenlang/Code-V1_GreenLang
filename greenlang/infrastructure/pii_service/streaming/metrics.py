# -*- coding: utf-8 -*-
"""
Streaming PII Scanner Metrics - SEC-011

Prometheus metrics for streaming PII scanner observability.
Provides counters, histograms, and gauges for monitoring
Kafka and Kinesis PII scanning performance.

Metrics are lazily initialized on first use so that the module can be
imported even when prometheus_client is not installed (metrics become no-ops).

Registered Metrics:
    - gl_pii_stream_processed_total: Messages processed by topic and action
    - gl_pii_stream_blocked_total: Messages blocked by topic and PII type
    - gl_pii_stream_detections_total: PII detections by topic and type
    - gl_pii_stream_errors_total: Processing errors by topic and error type
    - gl_pii_stream_processing_seconds: Processing time histogram
    - gl_pii_stream_lag_seconds: Consumer lag gauge
    - gl_pii_stream_batch_size: Batch size histogram
    - gl_pii_stream_running: Whether scanner is running (gauge)

Usage:
    >>> metrics = StreamingPIIMetrics()
    >>> metrics.record_processed("events.raw", "redacted")
    >>> metrics.record_blocked("events.raw", "ssn")
    >>> metrics.record_processing_time(5.2)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy Prometheus Handles
# ---------------------------------------------------------------------------


class _PrometheusHandles:
    """Lazy-initialized Prometheus metric objects.

    Metrics are created on first call to ensure_initialized().
    If prometheus_client is not installed, all handles remain None
    and recording methods become safe no-ops.
    """

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    _available: bool = False

    # Counters
    stream_processed_total: Any = None
    stream_blocked_total: Any = None
    stream_detections_total: Any = None
    stream_errors_total: Any = None
    stream_dlq_total: Any = None

    # Histograms
    processing_seconds: Any = None
    batch_size: Any = None
    message_size_bytes: Any = None

    # Gauges
    stream_running: Any = None
    stream_lag_seconds: Any = None
    consumer_offset: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Thread-safe via a class-level lock.

        Returns:
            True if prometheus_client is available and metrics are
            registered, False otherwise.
        """
        if cls._initialized:
            return cls._available

        with cls._lock:
            if cls._initialized:
                return cls._available

            cls._initialized = True

            try:
                from prometheus_client import Counter, Gauge, Histogram
            except ImportError:
                logger.info(
                    "prometheus_client not installed; streaming PII metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_pii_stream"

            # -- Counters --------------------------------------------------

            cls.stream_processed_total = Counter(
                f"{prefix}_processed_total",
                "Total stream messages processed",
                ["topic", "action"],
            )

            cls.stream_blocked_total = Counter(
                f"{prefix}_blocked_total",
                "Total stream messages blocked due to PII",
                ["topic", "pii_type"],
            )

            cls.stream_detections_total = Counter(
                f"{prefix}_detections_total",
                "Total PII detections in stream messages",
                ["topic", "pii_type"],
            )

            cls.stream_errors_total = Counter(
                f"{prefix}_errors_total",
                "Total stream processing errors",
                ["topic", "error_type"],
            )

            cls.stream_dlq_total = Counter(
                f"{prefix}_dlq_total",
                "Total messages sent to dead letter queue",
                ["topic", "reason"],
            )

            # -- Histograms ------------------------------------------------

            cls.processing_seconds = Histogram(
                f"{prefix}_processing_seconds",
                "Message processing time in seconds",
                ["backend"],  # kafka, kinesis
                buckets=(
                    0.0001,  # 0.1ms
                    0.0005,  # 0.5ms
                    0.001,  # 1ms
                    0.005,  # 5ms
                    0.01,  # 10ms
                    0.025,  # 25ms
                    0.05,  # 50ms
                    0.1,  # 100ms
                    0.25,  # 250ms
                    0.5,  # 500ms
                    1.0,  # 1s
                ),
            )

            cls.batch_size = Histogram(
                f"{prefix}_batch_size",
                "Messages per batch",
                ["backend"],
                buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            )

            cls.message_size_bytes = Histogram(
                f"{prefix}_message_size_bytes",
                "Message size in bytes",
                ["topic"],
                buckets=(
                    100,
                    500,
                    1000,
                    5000,
                    10000,
                    50000,
                    100000,
                    500000,
                    1000000,
                ),
            )

            # -- Gauges ----------------------------------------------------

            cls.stream_running = Gauge(
                f"{prefix}_running",
                "Whether the stream scanner is running (1=yes, 0=no)",
                ["backend", "consumer_group"],
            )

            cls.stream_lag_seconds = Gauge(
                f"{prefix}_lag_seconds",
                "Consumer lag in seconds",
                ["topic", "partition"],
            )

            cls.consumer_offset = Gauge(
                f"{prefix}_consumer_offset",
                "Current consumer offset",
                ["topic", "partition"],
            )

            cls._available = True
            logger.info("Streaming PII Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# StreamingPIIMetrics
# ---------------------------------------------------------------------------


class StreamingPIIMetrics:
    """Manages Prometheus metrics for streaming PII scanning.

    All recording methods are safe no-ops when prometheus_client is
    not installed. Thread-safe.

    This class provides a high-level interface for recording metrics
    during stream processing, abstracting away the Prometheus details.

    Example:
        >>> metrics = StreamingPIIMetrics()
        >>> metrics.record_processed("events.raw", "redacted")
        >>> metrics.record_blocked("events.raw", "ssn")
        >>> metrics.record_processing_time(5.2)  # milliseconds
    """

    def __init__(self, backend: str = "kafka") -> None:
        """Initialize streaming metrics.

        Args:
            backend: Backend identifier (kafka/kinesis) for labels.
        """
        self._backend = backend
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Message Processing Metrics
    # ------------------------------------------------------------------

    def record_processed(
        self,
        topic: str,
        action: str,
    ) -> None:
        """Record a processed message.

        Args:
            topic: Source topic/stream name.
            action: Action taken (allowed, redacted, blocked, error).
        """
        if not self._available:
            return

        _PrometheusHandles.stream_processed_total.labels(
            topic=topic, action=action
        ).inc()

    def record_blocked(
        self,
        topic: str,
        pii_type: str,
    ) -> None:
        """Record a blocked message.

        Args:
            topic: Source topic/stream name.
            pii_type: Primary PII type that caused the block.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_blocked_total.labels(
            topic=topic, pii_type=pii_type
        ).inc()

    def record_detection(
        self,
        topic: str,
        pii_type: str,
    ) -> None:
        """Record a PII detection.

        Args:
            topic: Source topic/stream name.
            pii_type: Type of PII detected.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_detections_total.labels(
            topic=topic, pii_type=pii_type
        ).inc()

    def record_error(
        self,
        topic: str,
        error_type: str,
    ) -> None:
        """Record a processing error.

        Args:
            topic: Source topic/stream name.
            error_type: Type of error encountered.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_errors_total.labels(
            topic=topic, error_type=error_type
        ).inc()

    def record_dlq_message(
        self,
        topic: str,
        reason: str = "pii_blocked",
    ) -> None:
        """Record a message sent to dead letter queue.

        Args:
            topic: Source topic/stream name.
            reason: Reason for DLQ routing.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_dlq_total.labels(
            topic=topic, reason=reason
        ).inc()

    # ------------------------------------------------------------------
    # Timing Metrics
    # ------------------------------------------------------------------

    def record_processing_time(
        self,
        time_ms: float,
    ) -> None:
        """Record message processing time.

        Args:
            time_ms: Processing time in milliseconds.
        """
        if not self._available:
            return

        # Convert to seconds for Prometheus convention
        _PrometheusHandles.processing_seconds.labels(
            backend=self._backend
        ).observe(time_ms / 1000.0)

    def record_batch_size(
        self,
        size: int,
    ) -> None:
        """Record batch size.

        Args:
            size: Number of messages in batch.
        """
        if not self._available:
            return

        _PrometheusHandles.batch_size.labels(backend=self._backend).observe(size)

    def record_message_size(
        self,
        topic: str,
        size_bytes: int,
    ) -> None:
        """Record message size.

        Args:
            topic: Source topic/stream name.
            size_bytes: Message size in bytes.
        """
        if not self._available:
            return

        _PrometheusHandles.message_size_bytes.labels(topic=topic).observe(size_bytes)

    # ------------------------------------------------------------------
    # State Metrics
    # ------------------------------------------------------------------

    def set_running(
        self,
        consumer_group: str,
        running: bool,
    ) -> None:
        """Set scanner running state.

        Args:
            consumer_group: Consumer group identifier.
            running: Whether scanner is running.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_running.labels(
            backend=self._backend, consumer_group=consumer_group
        ).set(1 if running else 0)

    def set_consumer_lag(
        self,
        topic: str,
        partition: int,
        lag_seconds: float,
    ) -> None:
        """Set consumer lag.

        Args:
            topic: Topic name.
            partition: Partition number.
            lag_seconds: Lag in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_lag_seconds.labels(
            topic=topic, partition=str(partition)
        ).set(lag_seconds)

    def set_consumer_offset(
        self,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Set current consumer offset.

        Args:
            topic: Topic name.
            partition: Partition number.
            offset: Current offset.
        """
        if not self._available:
            return

        _PrometheusHandles.consumer_offset.labels(
            topic=topic, partition=str(partition)
        ).set(offset)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------


_global_streaming_metrics: Optional[StreamingPIIMetrics] = None


def get_streaming_metrics(backend: str = "kafka") -> StreamingPIIMetrics:
    """Get or create the global streaming metrics instance.

    Args:
        backend: Backend type for metric labels.

    Returns:
        The global StreamingPIIMetrics instance.
    """
    global _global_streaming_metrics

    if _global_streaming_metrics is None:
        _global_streaming_metrics = StreamingPIIMetrics(backend=backend)

    return _global_streaming_metrics


def reset_streaming_metrics(backend: str = "kafka") -> StreamingPIIMetrics:
    """Reset and return a new streaming metrics instance.

    This is primarily useful for testing.

    Args:
        backend: Backend type for metric labels.

    Returns:
        A new StreamingPIIMetrics instance.
    """
    global _global_streaming_metrics
    _global_streaming_metrics = StreamingPIIMetrics(backend=backend)
    return _global_streaming_metrics


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "StreamingPIIMetrics",
    "get_streaming_metrics",
    "reset_streaming_metrics",
]
