# -*- coding: utf-8 -*-
"""
Audit Metrics - Centralized Audit Logging Service (SEC-005)

Prometheus counters, gauges, and histograms for audit service observability.
Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_audit_events_total (Counter): Total audit events by type/severity/result.
    - gl_audit_event_latency_seconds (Histogram): Audit processing latency by stage.
    - gl_audit_events_queued (Gauge): Events currently in the async queue.
    - gl_audit_db_write_failures_total (Counter): Database write failures.
    - gl_audit_stream_connections (Gauge): Active WebSocket connections.
    - gl_audit_export_jobs_total (Counter): Export job completions by status.
    - gl_audit_report_generation_seconds (Histogram): Report generation time.

Classes:
    - AuditMetrics: Singleton-style metrics manager.

Example:
    >>> metrics = AuditMetrics()
    >>> metrics.record_event("auth.login_success", "info", "success")
    >>> metrics.observe_latency("write", 0.005)
    >>> metrics.set_queue_depth(42)

Security Compliance:
    - SOC 2 CC7.2 (System Monitoring)
    - ISO 27001 A.12.4.1 (Event Logging)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: lazy Prometheus metric handles
# ---------------------------------------------------------------------------


class _PrometheusHandles:
    """Lazy-initialized Prometheus metric objects.

    Metrics are created on first call to :meth:`ensure_initialized`.
    If ``prometheus_client`` is not installed, all handles remain ``None``
    and recording methods become safe no-ops.
    """

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    _available: bool = False

    # Counters
    events_total: Any = None
    db_write_failures_total: Any = None
    export_jobs_total: Any = None

    # Gauges
    events_queued: Any = None
    stream_connections: Any = None

    # Histograms
    event_latency_seconds: Any = None
    report_generation_seconds: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Thread-safe via a class-level lock.

        Returns:
            ``True`` if prometheus_client is available and metrics are
            registered, ``False`` otherwise.
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
                    "prometheus_client not installed; audit metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_audit"

            # -- Counters --------------------------------------------------

            cls.events_total = Counter(
                f"{prefix}_events_total",
                "Total audit events logged",
                ["event_type", "severity", "result"],
            )

            cls.db_write_failures_total = Counter(
                f"{prefix}_db_write_failures_total",
                "Database write failures for audit events",
                ["error_type"],
            )

            cls.export_jobs_total = Counter(
                f"{prefix}_export_jobs_total",
                "Total audit export jobs completed",
                ["status"],
            )

            # -- Gauges ----------------------------------------------------

            cls.events_queued = Gauge(
                f"{prefix}_events_queued",
                "Number of audit events currently in the async write queue",
            )

            cls.stream_connections = Gauge(
                f"{prefix}_stream_connections",
                "Number of active WebSocket connections for audit streaming",
            )

            # -- Histograms ------------------------------------------------

            # Sub-millisecond buckets for low-latency audit path
            cls.event_latency_seconds = Histogram(
                f"{prefix}_event_latency_seconds",
                "Audit event processing latency in seconds",
                ["stage"],
                buckets=(
                    0.0001,  # 0.1ms
                    0.0005,  # 0.5ms
                    0.001,   # 1ms
                    0.0025,  # 2.5ms
                    0.005,   # 5ms
                    0.01,    # 10ms
                    0.025,   # 25ms
                    0.05,    # 50ms
                    0.1,     # 100ms
                    0.25,    # 250ms
                    0.5,     # 500ms
                    1.0,     # 1s
                ),
            )

            cls.report_generation_seconds = Histogram(
                f"{prefix}_report_generation_seconds",
                "Time to generate audit reports",
                ["report_type"],
                buckets=(
                    0.1,    # 100ms
                    0.5,    # 500ms
                    1.0,    # 1s
                    2.5,    # 2.5s
                    5.0,    # 5s
                    10.0,   # 10s
                    30.0,   # 30s
                    60.0,   # 1 min
                    120.0,  # 2 min
                    300.0,  # 5 min
                ),
            )

            cls._available = True
            logger.info("Audit Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# AuditMetrics
# ---------------------------------------------------------------------------


class AuditMetrics:
    """Manages Prometheus metrics for the audit service.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed. Thread-safe.

    Example:
        >>> m = AuditMetrics()
        >>> m.record_event("auth.login_success", "info", "success")
        >>> m.observe_latency("write", 0.003)
        >>> m.set_queue_depth(100)
    """

    def __init__(self, prefix: str = "gl_audit") -> None:
        """Initialize audit metrics.

        Args:
            prefix: Metric name prefix (used for documentation only;
                actual prefix is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        severity: str,
        result: str = "success",
    ) -> None:
        """Record an audit event.

        Args:
            event_type: Type of audit event (e.g., ``auth.login_success``,
                ``data.access``, ``config.change``).
            severity: Event severity (``debug``, ``info``, ``warning``,
                ``error``, ``critical``).
            result: Outcome (``success``, ``failure``, ``denied``).
        """
        if not self._available:
            return

        _PrometheusHandles.events_total.labels(
            event_type=event_type,
            severity=severity,
            result=result,
        ).inc()

    def record_events_batch(
        self,
        event_type: str,
        severity: str,
        result: str,
        count: int,
    ) -> None:
        """Record multiple audit events at once.

        Args:
            event_type: Type of audit events.
            severity: Event severity.
            result: Outcome.
            count: Number of events to record.
        """
        if not self._available:
            return

        _PrometheusHandles.events_total.labels(
            event_type=event_type,
            severity=severity,
            result=result,
        ).inc(count)

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def observe_latency(self, stage: str, duration_s: float) -> None:
        """Record processing latency for an audit pipeline stage.

        Stages:
            - ``capture``: Time to capture request/response data
            - ``enrich``: Time to enrich with user context
            - ``hash``: Time to compute integrity hash
            - ``queue``: Time to enqueue for async write
            - ``write``: Time to write to database
            - ``stream``: Time to broadcast to WebSocket clients

        Args:
            stage: Pipeline stage name.
            duration_s: Duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.event_latency_seconds.labels(
            stage=stage
        ).observe(duration_s)

    def observe_report_generation(
        self,
        report_type: str,
        duration_s: float,
    ) -> None:
        """Record time to generate an audit report.

        Args:
            report_type: Type of report (``compliance``, ``security``,
                ``activity``, ``export``).
            duration_s: Generation time in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.report_generation_seconds.labels(
            report_type=report_type
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Queue depth
    # ------------------------------------------------------------------

    def set_queue_depth(self, depth: int) -> None:
        """Set the current audit event queue depth.

        Args:
            depth: Number of events currently queued.
        """
        if not self._available:
            return

        _PrometheusHandles.events_queued.set(depth)

    def inc_queue_depth(self, count: int = 1) -> None:
        """Increment the queue depth gauge.

        Args:
            count: Number to increment by.
        """
        if not self._available:
            return

        _PrometheusHandles.events_queued.inc(count)

    def dec_queue_depth(self, count: int = 1) -> None:
        """Decrement the queue depth gauge.

        Args:
            count: Number to decrement by.
        """
        if not self._available:
            return

        _PrometheusHandles.events_queued.dec(count)

    # ------------------------------------------------------------------
    # DB write failures
    # ------------------------------------------------------------------

    def record_db_write_failure(self, error_type: str) -> None:
        """Record a database write failure.

        Args:
            error_type: Type of error (``connection``, ``timeout``,
                ``constraint_violation``, ``disk_full``, ``unknown``).
        """
        if not self._available:
            return

        _PrometheusHandles.db_write_failures_total.labels(
            error_type=error_type
        ).inc()

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    def set_stream_connections(self, count: int) -> None:
        """Set the number of active WebSocket connections.

        Args:
            count: Current connection count.
        """
        if not self._available:
            return

        _PrometheusHandles.stream_connections.set(count)

    def inc_stream_connections(self) -> None:
        """Increment WebSocket connection count by 1."""
        if not self._available:
            return

        _PrometheusHandles.stream_connections.inc()

    def dec_stream_connections(self) -> None:
        """Decrement WebSocket connection count by 1."""
        if not self._available:
            return

        _PrometheusHandles.stream_connections.dec()

    # ------------------------------------------------------------------
    # Export jobs
    # ------------------------------------------------------------------

    def record_export_job(self, status: str) -> None:
        """Record an export job completion.

        Args:
            status: Job status (``success``, ``failure``, ``cancelled``).
        """
        if not self._available:
            return

        _PrometheusHandles.export_jobs_total.labels(
            status=status
        ).inc()

    # ------------------------------------------------------------------
    # Helper context managers
    # ------------------------------------------------------------------

    def latency_timer(self, stage: str) -> "_LatencyTimer":
        """Get a context manager for timing a pipeline stage.

        Args:
            stage: The pipeline stage being timed.

        Returns:
            Context manager that records duration on exit.

        Example:
            >>> metrics = AuditMetrics()
            >>> with metrics.latency_timer("write"):
            ...     await write_to_db(event)
        """
        return _LatencyTimer(self, stage)


class _LatencyTimer:
    """Context manager for timing audit pipeline stages."""

    def __init__(self, metrics: AuditMetrics, stage: str) -> None:
        self._metrics = metrics
        self._stage = stage
        self._start: Optional[float] = None

    def __enter__(self) -> "_LatencyTimer":
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            import time
            duration = time.perf_counter() - self._start
            self._metrics.observe_latency(self._stage, duration)


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

_default_metrics: Optional[AuditMetrics] = None


def get_audit_metrics() -> AuditMetrics:
    """Get the default singleton AuditMetrics instance.

    Returns:
        Shared AuditMetrics instance.
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = AuditMetrics()
    return _default_metrics


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "AuditMetrics",
    "get_audit_metrics",
]
