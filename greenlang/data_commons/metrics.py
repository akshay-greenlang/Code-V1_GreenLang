# -*- coding: utf-8 -*-
"""
MetricsFactory - Shared Prometheus metric boilerplate for data-layer agents.

Eliminates ~280 lines of duplicate metric definitions per agent by providing
a factory that creates the 6 standard metrics every data agent needs
(operations counter, duration histogram, validation errors counter, batch
jobs counter, active jobs gauge, queue size gauge) plus factory methods for
agent-specific custom metrics.

All 20 ``greenlang/agents/data/*/metrics.py`` modules follow the same
pattern.  This factory centralises that pattern so each agent file shrinks
from ~330 lines to ~50-80 lines (factory instantiation + agent-specific
metrics).

Example -- minimal agent metrics file::

    from greenlang.data_commons.metrics import MetricsFactory, PROMETHEUS_AVAILABLE

    m = MetricsFactory("gl_pdf", "PDF Extractor")

    # Standard metrics available via m.operations_total, m.processing_duration, etc.
    # Standard helpers via m.record_operation(), m.record_batch_job(), etc.

    # Agent-specific extras
    pdf_pages_extracted_total = m.create_custom_counter(
        "pages_extracted_total", "Total pages extracted from PDF documents",
    )

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; MetricsFactory metrics disabled"
    )

# ---------------------------------------------------------------------------
# Default bucket configurations
# ---------------------------------------------------------------------------

#: Standard processing-duration buckets (sub-second to 5-minute extractions).
DURATION_BUCKETS: Tuple[float, ...] = (
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 15.0, 30.0, 60.0, 120.0, 300.0,
)

#: Confidence/quality score buckets (0.1 to 1.0 in 0.1 increments).
CONFIDENCE_BUCKETS: Tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

#: Long-running operation buckets (ERP syncs, batch jobs, etc.).
LONG_DURATION_BUCKETS: Tuple[float, ...] = (
    0.1, 0.5, 1.0, 2.5, 5.0, 10.0,
    30.0, 60.0, 120.0, 300.0, 480.0, 600.0,
)


# ---------------------------------------------------------------------------
# MetricsFactory
# ---------------------------------------------------------------------------


class MetricsFactory:
    """Factory for creating standard Prometheus metrics with consistent naming.

    Every data-layer agent defines the same 6 core metrics with an
    agent-specific prefix.  This class creates those metrics once and
    provides safe helper methods that no-op when ``prometheus_client`` is
    not installed.

    Attributes:
        prefix: Metric name prefix (e.g. ``"gl_pdf"``, ``"gl_erp"``).
        service_name: Human-readable service name for metric descriptions.
        available: Whether ``prometheus_client`` is importable.
        operations_total: Counter for completed operations (labels: ``type``, ``tenant_id``).
        processing_duration: Histogram for operation duration in seconds.
        validation_errors_total: Counter for validation errors (labels: ``severity``, ``type``).
        batch_jobs_total: Counter for batch jobs (labels: ``status``).
        active_jobs: Gauge for currently active jobs.
        queue_size: Gauge for current queue depth.

    Example::

        >>> m = MetricsFactory("gl_pdf", "PDF Extractor")
        >>> m.record_operation("invoice", "tenant-1", 1.23)
        >>> m.record_validation_error("high", "missing_field")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        prefix: str,
        service_name: str,
        *,
        duration_buckets: Tuple[float, ...] = DURATION_BUCKETS,
        operations_label_names: Tuple[str, ...] = ("type", "tenant_id"),
        validation_label_names: Tuple[str, ...] = ("severity", "type"),
    ) -> None:
        """Initialise the factory and register all standard metrics.

        Args:
            prefix: Metric name prefix (e.g. ``"gl_pdf"``).  All metric
                names will start with ``{prefix}_``.
            service_name: Human-readable name used in metric descriptions.
            duration_buckets: Histogram buckets for processing duration.
                Defaults to :data:`DURATION_BUCKETS`.
            operations_label_names: Label names for the operations counter.
                Defaults to ``("type", "tenant_id")``.
            validation_label_names: Label names for the validation errors
                counter.  Defaults to ``("severity", "type")``.
        """
        self.prefix = prefix
        self.service_name = service_name
        self.available = PROMETHEUS_AVAILABLE
        self._operations_label_names = operations_label_names
        self._validation_label_names = validation_label_names

        if PROMETHEUS_AVAILABLE:
            self.operations_total: Optional[Any] = Counter(
                f"{prefix}_operations_total",
                f"Total {service_name} operations completed",
                labelnames=list(operations_label_names),
            )
            self.processing_duration: Optional[Any] = Histogram(
                f"{prefix}_processing_duration_seconds",
                f"{service_name} processing duration in seconds",
                buckets=duration_buckets,
            )
            self.validation_errors_total: Optional[Any] = Counter(
                f"{prefix}_validation_errors_total",
                f"Total validation errors detected during {service_name} processing",
                labelnames=list(validation_label_names),
            )
            self.batch_jobs_total: Optional[Any] = Counter(
                f"{prefix}_batch_jobs_total",
                f"Total {service_name} batch jobs",
                labelnames=["status"],
            )
            self.active_jobs: Optional[Any] = Gauge(
                f"{prefix}_active_jobs",
                f"Number of currently active {service_name} jobs",
            )
            self.queue_size: Optional[Any] = Gauge(
                f"{prefix}_queue_size",
                f"Current number of items waiting in {service_name} queue",
            )
        else:
            self.operations_total = None
            self.processing_duration = None
            self.validation_errors_total = None
            self.batch_jobs_total = None
            self.active_jobs = None
            self.queue_size = None

    # ------------------------------------------------------------------
    # Standard helper methods (safe no-ops when prometheus is absent)
    # ------------------------------------------------------------------

    def record_operation(self, duration_seconds: float, **labels: str) -> None:
        """Record a completed operation with duration.

        Increments :attr:`operations_total` with the given *labels* and
        observes *duration_seconds* on :attr:`processing_duration`.

        Args:
            duration_seconds: Processing duration in seconds.
            **labels: Label key-value pairs matching
                ``operations_label_names`` (default ``type`` and
                ``tenant_id``).
        """
        if not self.available:
            return
        self.operations_total.labels(**labels).inc()  # type: ignore[union-attr]
        self.processing_duration.observe(duration_seconds)  # type: ignore[union-attr]

    def record_validation_error(self, **labels: str) -> None:
        """Record a validation error.

        Args:
            **labels: Label key-value pairs matching
                ``validation_label_names`` (default ``severity`` and
                ``type``).
        """
        if not self.available:
            return
        self.validation_errors_total.labels(**labels).inc()  # type: ignore[union-attr]

    def record_batch_job(self, status: str) -> None:
        """Record a batch job event.

        Args:
            status: Batch job status (submitted, completed, failed, partial).
        """
        if not self.available:
            return
        self.batch_jobs_total.labels(status=status).inc()  # type: ignore[union-attr]

    def update_active_jobs(self, delta: int) -> None:
        """Increment or decrement the active jobs gauge.

        Args:
            delta: Positive to increment, negative to decrement.
        """
        if not self.available:
            return
        if delta > 0:
            self.active_jobs.inc(delta)  # type: ignore[union-attr]
        elif delta < 0:
            self.active_jobs.dec(abs(delta))  # type: ignore[union-attr]

    def update_queue_size(self, size: int) -> None:
        """Set the current queue depth.

        Args:
            size: Current queue depth.
        """
        if not self.available:
            return
        self.queue_size.set(size)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Custom metric factory methods
    # ------------------------------------------------------------------

    def create_custom_counter(
        self,
        name: str,
        description: str,
        labelnames: Optional[Sequence[str]] = None,
    ) -> Optional[Any]:
        """Create an agent-specific Counter metric.

        The metric name is automatically prefixed with ``{self.prefix}_``.

        Args:
            name: Metric name suffix (e.g. ``"pages_extracted_total"``).
            description: Human-readable metric description.
            labelnames: Optional label names for the counter.

        Returns:
            A ``prometheus_client.Counter`` instance, or ``None`` if
            prometheus_client is not installed.
        """
        if not self.available:
            return None
        kwargs: dict[str, Any] = {}
        if labelnames:
            kwargs["labelnames"] = list(labelnames)
        return Counter(f"{self.prefix}_{name}", description, **kwargs)

    def create_custom_histogram(
        self,
        name: str,
        description: str,
        buckets: Tuple[float, ...] = DURATION_BUCKETS,
        labelnames: Optional[Sequence[str]] = None,
    ) -> Optional[Any]:
        """Create an agent-specific Histogram metric.

        The metric name is automatically prefixed with ``{self.prefix}_``.

        Args:
            name: Metric name suffix (e.g. ``"extraction_confidence"``).
            description: Human-readable metric description.
            buckets: Histogram bucket boundaries.
            labelnames: Optional label names for the histogram.

        Returns:
            A ``prometheus_client.Histogram`` instance, or ``None`` if
            prometheus_client is not installed.
        """
        if not self.available:
            return None
        kwargs: dict[str, Any] = {"buckets": buckets}
        if labelnames:
            kwargs["labelnames"] = list(labelnames)
        return Histogram(f"{self.prefix}_{name}", description, **kwargs)

    def create_custom_gauge(
        self,
        name: str,
        description: str,
        labelnames: Optional[Sequence[str]] = None,
    ) -> Optional[Any]:
        """Create an agent-specific Gauge metric.

        The metric name is automatically prefixed with ``{self.prefix}_``.

        Args:
            name: Metric name suffix (e.g. ``"sync_queue_size"``).
            description: Human-readable metric description.
            labelnames: Optional label names for the gauge.

        Returns:
            A ``prometheus_client.Gauge`` instance, or ``None`` if
            prometheus_client is not installed.
        """
        if not self.available:
            return None
        kwargs: dict[str, Any] = {}
        if labelnames:
            kwargs["labelnames"] = list(labelnames)
        return Gauge(f"{self.prefix}_{name}", description, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers for custom metrics (safe no-ops)
    # ------------------------------------------------------------------

    @staticmethod
    def safe_inc(
        metric: Optional[Any],
        amount: Union[int, float] = 1,
        **labels: str,
    ) -> None:
        """Safely increment a Counter or labelled Counter.

        Args:
            metric: A Counter instance or ``None``.
            amount: Increment amount (default 1).
            **labels: Optional label key-value pairs.
        """
        if metric is None:
            return
        if labels:
            metric.labels(**labels).inc(amount)
        else:
            metric.inc(amount)

    @staticmethod
    def safe_observe(
        metric: Optional[Any],
        value: float,
        **labels: str,
    ) -> None:
        """Safely observe a value on a Histogram.

        Args:
            metric: A Histogram instance or ``None``.
            value: Value to observe.
            **labels: Optional label key-value pairs.
        """
        if metric is None:
            return
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)

    @staticmethod
    def safe_set(
        metric: Optional[Any],
        value: Union[int, float],
        **labels: str,
    ) -> None:
        """Safely set a Gauge value.

        Args:
            metric: A Gauge instance or ``None``.
            value: Value to set.
            **labels: Optional label key-value pairs.
        """
        if metric is None:
            return
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "enabled" if self.available else "disabled"
        return f"MetricsFactory(prefix={self.prefix!r}, service={self.service_name!r}, {status})"


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "DURATION_BUCKETS",
    "CONFIDENCE_BUCKETS",
    "LONG_DURATION_BUCKETS",
    "MetricsFactory",
]
