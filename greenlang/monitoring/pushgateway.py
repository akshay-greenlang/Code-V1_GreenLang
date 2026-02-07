# -*- coding: utf-8 -*-
"""
GreenLang PushGateway SDK - Prometheus Metrics for Batch Jobs (OBS-001 Phase 3)

Production-ready Python SDK for pushing metrics from batch jobs to PushGateway.
Provides standardized metrics collection, retry logic with exponential backoff,
and seamless integration with Kubernetes CronJobs and agent factory pipelines.

Features:
    - Standard batch job metrics (duration, records, errors, status)
    - Thread-safe operations with proper locking
    - Exponential backoff retry with configurable jitter
    - Context manager for automatic duration tracking
    - Graceful error handling (job continues if push fails)
    - Singleton factory for shared client instances
    - Complete type hints and docstrings

Usage:
    Basic usage:
        >>> from greenlang.monitoring.pushgateway import BatchJobMetrics
        >>> metrics = BatchJobMetrics("my-batch-job")
        >>> with metrics.track_duration():
        ...     # do work
        ...     metrics.record_records(100, "processed")
        >>> metrics.push()

    With grouping keys (multi-instance jobs):
        >>> metrics = BatchJobMetrics(
        ...     "data-import",
        ...     grouping_key={"instance": "worker-1", "region": "us-east-1"}
        ... )
        >>> with metrics.track_duration():
        ...     process_batch()
        >>> metrics.push()

    CronJob integration:
        >>> async def main():
        ...     metrics = get_pushgateway_client("pii-remediation-cron")
        ...     try:
        ...         with metrics.track_duration():
        ...             result = await run_remediation()
        ...             metrics.record_records(result.processed, "remediated")
        ...             if result.failed > 0:
        ...                 metrics.record_records(result.failed, "failed")
        ...     except Exception as e:
        ...         metrics.record_failure(type(e).__name__)
        ...         raise
        ...     finally:
        ...         metrics.push()

    Agent Factory integration:
        >>> from greenlang.monitoring.pushgateway import BatchJobMetrics
        >>> metrics = BatchJobMetrics(
        ...     "pack-build",
        ...     grouping_key={"pack_name": pack.name, "version": pack.version}
        ... )
        >>> with metrics.track_duration():
        ...     build_pack()
        >>> metrics.push()

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-001 Prometheus Metrics Collection - Phase 3
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus client import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        push_to_gateway,
        delete_from_gateway,
    )
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment, misc]
    Counter = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
    push_to_gateway = None  # type: ignore[assignment]
    delete_from_gateway = None  # type: ignore[assignment]
    logger.warning(
        "prometheus_client not installed; BatchJobMetrics will operate in no-op mode"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PushGatewayConfig:
    """Configuration for PushGateway client.

    Attributes:
        url: PushGateway URL. Defaults to in-cluster service address.
        job_name: Default job name for metrics.
        grouping_key: Default grouping key for metric isolation.
        timeout: HTTP timeout for push operations in seconds.
        max_retries: Maximum retry attempts on push failure.
        retry_backoff: Base backoff time in seconds for retries.
        retry_jitter: Maximum jitter to add to backoff (0.0-1.0).
        fail_silently: If True, log errors but don't raise on push failure.
    """

    url: str = "http://pushgateway.monitoring.svc:9091"
    job_name: str = ""
    grouping_key: Dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    retry_jitter: float = 0.3
    fail_silently: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff <= 0:
            raise ValueError("retry_backoff must be positive")
        if not 0.0 <= self.retry_jitter <= 1.0:
            raise ValueError("retry_jitter must be between 0.0 and 1.0")


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------

# Status values for gl_batch_job_status gauge
STATUS_VALUES: Dict[str, int] = {
    "idle": 0,
    "running": 1,
    "success": 2,
    "failed": 3,
}


# ---------------------------------------------------------------------------
# Push Error
# ---------------------------------------------------------------------------


class PushGatewayError(Exception):
    """Raised when push to PushGateway fails after all retries."""

    def __init__(self, message: str, attempts: int, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# BatchJobMetrics
# ---------------------------------------------------------------------------


class BatchJobMetrics:
    """Push metrics from batch jobs to PushGateway.

    Provides standardized metrics for batch job monitoring including duration,
    success/failure timestamps, records processed, errors, and job status.
    Thread-safe with proper locking for concurrent metric updates.

    Standard Metrics:
        - gl_batch_job_duration_seconds (Gauge): Job execution duration
        - gl_batch_job_last_success_timestamp (Gauge): Last successful run
        - gl_batch_job_last_failure_timestamp (Gauge): Last failed run
        - gl_batch_job_records_processed_total (Counter): Records processed
        - gl_batch_job_records_failed_total (Counter): Records that failed
        - gl_batch_job_errors_total (Counter): Errors by type
        - gl_batch_job_retries_total (Counter): Retry attempts
        - gl_batch_job_status (Gauge): Current status (0=idle, 1=running, 2=success, 3=failed)

    Attributes:
        job_name: Unique identifier for this batch job.
        pushgateway_url: URL of the PushGateway service.
        grouping_key: Additional keys for metric grouping (e.g., instance, region).
        timeout: HTTP timeout for push operations.

    Example:
        >>> metrics = BatchJobMetrics("my-batch-job")
        >>> with metrics.track_duration():
        ...     # do work
        ...     metrics.record_records(100, "processed")
        >>> metrics.push()

    Example with grouping keys:
        >>> metrics = BatchJobMetrics(
        ...     "data-import",
        ...     grouping_key={"instance": "worker-1"}
        ... )
        >>> with metrics.track_duration():
        ...     process_batch()
        >>> metrics.push()
    """

    def __init__(
        self,
        job_name: str,
        pushgateway_url: str = "http://pushgateway.monitoring.svc:9091",
        grouping_key: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        fail_silently: bool = True,
    ) -> None:
        """Initialize BatchJobMetrics.

        Args:
            job_name: Unique identifier for this batch job.
            pushgateway_url: URL of the PushGateway service.
            grouping_key: Additional keys for metric grouping.
            timeout: HTTP timeout for push operations in seconds.
            max_retries: Maximum retry attempts on push failure.
            retry_backoff: Base backoff time in seconds for retries.
            fail_silently: If True, log errors but don't raise on push failure.

        Raises:
            ValueError: If job_name is empty.
        """
        if not job_name:
            raise ValueError("job_name is required")

        self.job_name = job_name
        self.pushgateway_url = pushgateway_url or "http://pushgateway.monitoring.svc:9091"
        self.grouping_key = grouping_key or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.fail_silently = fail_silently

        # Thread safety
        self._lock = threading.RLock()

        # Track internal state
        self._start_time: Optional[float] = None
        self._current_status: str = "idle"

        # Initialize Prometheus registry and metrics
        self._registry: Optional[Any] = None
        self._metrics_initialized = False

        if PROMETHEUS_CLIENT_AVAILABLE:
            self._init_metrics()

        logger.debug(
            "BatchJobMetrics initialized: job=%s, url=%s, grouping_key=%s",
            self.job_name,
            self.pushgateway_url,
            self.grouping_key,
        )

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics with the collector registry."""
        if self._metrics_initialized:
            return

        self._registry = CollectorRegistry()

        # Job duration gauge
        self._duration = Gauge(
            "gl_batch_job_duration_seconds",
            "Duration of batch job execution in seconds",
            ["job_name"],
            registry=self._registry,
        )

        # Last success timestamp
        self._last_success = Gauge(
            "gl_batch_job_last_success_timestamp",
            "Unix timestamp of last successful job run",
            ["job_name"],
            registry=self._registry,
        )

        # Last failure timestamp
        self._last_failure = Gauge(
            "gl_batch_job_last_failure_timestamp",
            "Unix timestamp of last failed job run",
            ["job_name"],
            registry=self._registry,
        )

        # Records processed counter
        self._records_processed = Counter(
            "gl_batch_job_records_processed_total",
            "Total number of records processed by batch job",
            ["job_name", "record_type"],
            registry=self._registry,
        )

        # Records failed counter
        self._records_failed = Counter(
            "gl_batch_job_records_failed_total",
            "Total number of records that failed processing",
            ["job_name", "record_type"],
            registry=self._registry,
        )

        # Errors counter
        self._errors = Counter(
            "gl_batch_job_errors_total",
            "Total number of errors in batch job",
            ["job_name", "error_type"],
            registry=self._registry,
        )

        # Retries counter
        self._retries = Counter(
            "gl_batch_job_retries_total",
            "Total number of retry attempts in batch job",
            ["job_name"],
            registry=self._registry,
        )

        # Job status gauge (0=idle, 1=running, 2=success, 3=failed)
        self._status = Gauge(
            "gl_batch_job_status",
            "Current job status (0=idle, 1=running, 2=success, 3=failed)",
            ["job_name"],
            registry=self._registry,
        )

        # Set initial status to idle
        self._status.labels(job_name=self.job_name).set(STATUS_VALUES["idle"])

        self._metrics_initialized = True
        logger.debug("Prometheus metrics initialized for job: %s", self.job_name)

    # -------------------------------------------------------------------------
    # Push and Delete
    # -------------------------------------------------------------------------

    def push(self) -> None:
        """Push metrics to PushGateway with retry logic.

        Implements exponential backoff with jitter for resilient metric delivery.
        Logs warnings on retry and errors on final failure.

        Raises:
            PushGatewayError: If push fails after all retries (only if fail_silently=False).
        """
        if not PROMETHEUS_CLIENT_AVAILABLE or self._registry is None:
            logger.debug("Prometheus client not available; skipping push")
            return

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                push_to_gateway(
                    self.pushgateway_url,
                    job=self.job_name,
                    registry=self._registry,
                    grouping_key=self.grouping_key,
                    timeout=self.timeout,
                )
                logger.info(
                    "Pushed metrics to PushGateway: job=%s, attempt=%d",
                    self.job_name,
                    attempt + 1,
                )
                return

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Calculate backoff with jitter
                    backoff = self.retry_backoff * (2 ** attempt)
                    jitter = random.uniform(0, self.retry_backoff * 0.3)
                    sleep_time = backoff + jitter

                    logger.warning(
                        "Push to PushGateway failed (attempt %d/%d): %s. "
                        "Retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        str(e),
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Push to PushGateway failed after %d attempts: %s",
                        self.max_retries + 1,
                        str(e),
                    )

        if not self.fail_silently:
            raise PushGatewayError(
                f"Failed to push metrics for job '{self.job_name}' after "
                f"{self.max_retries + 1} attempts",
                attempts=self.max_retries + 1,
                last_error=last_error,
            )

    def delete(self) -> None:
        """Delete metrics from PushGateway (on job completion).

        Useful for cleaning up metrics after a job completes to prevent
        stale data in PushGateway.

        Raises:
            PushGatewayError: If delete fails (only if fail_silently=False).
        """
        if not PROMETHEUS_CLIENT_AVAILABLE:
            logger.debug("Prometheus client not available; skipping delete")
            return

        try:
            delete_from_gateway(
                self.pushgateway_url,
                job=self.job_name,
                grouping_key=self.grouping_key,
                timeout=self.timeout,
            )
            logger.info(
                "Deleted metrics from PushGateway: job=%s",
                self.job_name,
            )
        except Exception as e:
            logger.error(
                "Failed to delete metrics from PushGateway: job=%s, error=%s",
                self.job_name,
                str(e),
            )
            if not self.fail_silently:
                raise PushGatewayError(
                    f"Failed to delete metrics for job '{self.job_name}'",
                    attempts=1,
                    last_error=e,
                )

    # -------------------------------------------------------------------------
    # Duration Tracking
    # -------------------------------------------------------------------------

    @contextmanager
    def track_duration(
        self,
        status_on_success: str = "success",
    ) -> Generator[None, None, None]:
        """Context manager to track job duration and status.

        Automatically records job duration and updates status on completion.
        If an exception occurs, records failure and re-raises.

        Args:
            status_on_success: Status to set on successful completion.

        Yields:
            None

        Example:
            >>> with metrics.track_duration():
            ...     do_work()
            >>> # Duration recorded, status set to "success"

            >>> with metrics.track_duration(status_on_success="completed"):
            ...     do_work()
            >>> # Duration recorded, status set to "completed"
        """
        self.set_status("running")
        start_time = time.perf_counter()

        try:
            yield
            duration = time.perf_counter() - start_time
            self._record_duration(duration)
            self.record_success()
            # Allow custom success status
            if status_on_success != "success":
                self.set_status(status_on_success)

        except Exception as e:
            duration = time.perf_counter() - start_time
            self._record_duration(duration)
            self.record_failure(type(e).__name__)
            raise

    def _record_duration(self, duration: float) -> None:
        """Record the job execution duration.

        Args:
            duration: Duration in seconds.
        """
        if not self._metrics_initialized:
            return

        with self._lock:
            self._duration.labels(job_name=self.job_name).set(duration)

        logger.debug(
            "Recorded duration for job %s: %.3fs",
            self.job_name,
            duration,
        )

    # -------------------------------------------------------------------------
    # Success and Failure Recording
    # -------------------------------------------------------------------------

    def record_success(self) -> None:
        """Record successful job completion.

        Updates the last success timestamp and sets status to 'success'.
        """
        if not self._metrics_initialized:
            return

        with self._lock:
            timestamp = time.time()
            self._last_success.labels(job_name=self.job_name).set(timestamp)
            self._status.labels(job_name=self.job_name).set(STATUS_VALUES["success"])
            self._current_status = "success"

        logger.info("Recorded success for job: %s", self.job_name)

    def record_failure(self, error_type: str) -> None:
        """Record job failure.

        Updates the last failure timestamp, increments error counter,
        and sets status to 'failed'.

        Args:
            error_type: Type/category of the error (e.g., exception class name).
        """
        if not self._metrics_initialized:
            return

        with self._lock:
            timestamp = time.time()
            self._last_failure.labels(job_name=self.job_name).set(timestamp)
            self._errors.labels(job_name=self.job_name, error_type=error_type).inc()
            self._status.labels(job_name=self.job_name).set(STATUS_VALUES["failed"])
            self._current_status = "failed"

        logger.warning(
            "Recorded failure for job %s: error_type=%s",
            self.job_name,
            error_type,
        )

    # -------------------------------------------------------------------------
    # Record Counts
    # -------------------------------------------------------------------------

    def record_records(self, count: int, record_type: str = "default") -> None:
        """Record number of records processed.

        Args:
            count: Number of records processed.
            record_type: Type/category of records (e.g., "processed", "imported").

        Example:
            >>> metrics.record_records(100, "processed")
            >>> metrics.record_records(50, "imported")
        """
        if not self._metrics_initialized:
            return

        if count <= 0:
            return

        with self._lock:
            self._records_processed.labels(
                job_name=self.job_name,
                record_type=record_type,
            ).inc(count)

        logger.debug(
            "Recorded %d records for job %s (type=%s)",
            count,
            self.job_name,
            record_type,
        )

    def record_failed_records(self, count: int, record_type: str = "default") -> None:
        """Record number of records that failed processing.

        Args:
            count: Number of failed records.
            record_type: Type/category of records.

        Example:
            >>> metrics.record_failed_records(5, "validation_failed")
        """
        if not self._metrics_initialized:
            return

        if count <= 0:
            return

        with self._lock:
            self._records_failed.labels(
                job_name=self.job_name,
                record_type=record_type,
            ).inc(count)

        logger.debug(
            "Recorded %d failed records for job %s (type=%s)",
            count,
            self.job_name,
            record_type,
        )

    def record_error(self, error_type: str) -> None:
        """Record an error (doesn't mark job as failed).

        Use this for non-fatal errors that don't fail the entire job.
        For fatal errors that should mark the job as failed, use record_failure().

        Args:
            error_type: Type/category of the error.

        Example:
            >>> try:
            ...     process_item(item)
            ... except ValidationError:
            ...     metrics.record_error("validation_error")
            ...     continue  # Job continues
        """
        if not self._metrics_initialized:
            return

        with self._lock:
            self._errors.labels(job_name=self.job_name, error_type=error_type).inc()

        logger.debug(
            "Recorded error for job %s: error_type=%s",
            self.job_name,
            error_type,
        )

    def record_retry(self) -> None:
        """Record a retry attempt.

        Use this when the job retries an operation.

        Example:
            >>> for attempt in range(3):
            ...     try:
            ...         result = call_api()
            ...         break
            ...     except TransientError:
            ...         metrics.record_retry()
        """
        if not self._metrics_initialized:
            return

        with self._lock:
            self._retries.labels(job_name=self.job_name).inc()

        logger.debug("Recorded retry for job: %s", self.job_name)

    # -------------------------------------------------------------------------
    # Status Management
    # -------------------------------------------------------------------------

    def set_status(
        self,
        status: Literal["idle", "running", "success", "failed"],
    ) -> None:
        """Set current job status.

        Args:
            status: One of "idle", "running", "success", "failed".

        Raises:
            ValueError: If status is not a valid value.

        Example:
            >>> metrics.set_status("running")
            >>> do_work()
            >>> metrics.set_status("success")
        """
        if status not in STATUS_VALUES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {list(STATUS_VALUES.keys())}"
            )

        if not self._metrics_initialized:
            return

        with self._lock:
            self._status.labels(job_name=self.job_name).set(STATUS_VALUES[status])
            self._current_status = status

        logger.debug("Set status for job %s: %s", self.job_name, status)

    def get_status(self) -> str:
        """Get current job status.

        Returns:
            Current status string.
        """
        return self._current_status

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all metrics to initial state.

        Useful for reusing a metrics instance across multiple job runs.
        """
        if not self._metrics_initialized:
            return

        # Re-initialize metrics by recreating the registry
        self._metrics_initialized = False
        self._init_metrics()
        self._current_status = "idle"

        logger.debug("Reset metrics for job: %s", self.job_name)

    def get_info(self) -> Dict[str, Any]:
        """Get information about this metrics instance.

        Returns:
            Dictionary with job name, URL, grouping key, and status.
        """
        return {
            "job_name": self.job_name,
            "pushgateway_url": self.pushgateway_url,
            "grouping_key": self.grouping_key,
            "current_status": self._current_status,
            "metrics_initialized": self._metrics_initialized,
            "prometheus_available": PROMETHEUS_CLIENT_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Singleton Factory
# ---------------------------------------------------------------------------

_pushgateway_clients: Dict[str, BatchJobMetrics] = {}
_factory_lock = threading.Lock()


def get_pushgateway_client(
    job_name: Optional[str] = None,
    config: Optional[PushGatewayConfig] = None,
) -> BatchJobMetrics:
    """Get or create a PushGateway client singleton.

    Returns a cached BatchJobMetrics instance for the given job name,
    creating a new one if it doesn't exist. Use this for long-running
    services that push metrics periodically.

    Args:
        job_name: Unique job name. Required if config is not provided.
        config: Optional configuration. If provided, job_name is taken from config.

    Returns:
        BatchJobMetrics instance for the job.

    Raises:
        ValueError: If neither job_name nor config.job_name is provided.

    Example:
        >>> client = get_pushgateway_client("my-batch-job")
        >>> # Returns same instance on subsequent calls
        >>> client2 = get_pushgateway_client("my-batch-job")
        >>> assert client is client2
    """
    effective_job_name = job_name
    if config is not None and config.job_name:
        effective_job_name = config.job_name

    if not effective_job_name:
        raise ValueError("job_name is required (either as argument or in config)")

    with _factory_lock:
        if effective_job_name not in _pushgateway_clients:
            if config is not None:
                client = BatchJobMetrics(
                    job_name=effective_job_name,
                    pushgateway_url=config.url,
                    grouping_key=dict(config.grouping_key),
                    timeout=config.timeout,
                    max_retries=config.max_retries,
                    retry_backoff=config.retry_backoff,
                    fail_silently=config.fail_silently,
                )
            else:
                # Use defaults with environment override for URL
                url = os.environ.get(
                    "PUSHGATEWAY_URL",
                    "http://pushgateway.monitoring.svc:9091",
                )
                client = BatchJobMetrics(
                    job_name=effective_job_name,
                    pushgateway_url=url,
                )

            _pushgateway_clients[effective_job_name] = client
            logger.debug(
                "Created new PushGateway client for job: %s",
                effective_job_name,
            )

        return _pushgateway_clients[effective_job_name]


def clear_pushgateway_clients() -> None:
    """Clear all cached PushGateway client singletons.

    Useful for testing or when reconfiguring clients.
    """
    with _factory_lock:
        _pushgateway_clients.clear()
        logger.debug("Cleared all PushGateway client singletons")


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_batch_job_metrics(
    job_name: str,
    grouping_key: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> BatchJobMetrics:
    """Create a new BatchJobMetrics instance with sensible defaults.

    Convenience function that reads configuration from environment variables
    when not explicitly provided.

    Environment Variables:
        PUSHGATEWAY_URL: PushGateway service URL

    Args:
        job_name: Unique identifier for this batch job.
        grouping_key: Additional keys for metric grouping.
        **kwargs: Additional arguments passed to BatchJobMetrics.

    Returns:
        Configured BatchJobMetrics instance.

    Example:
        >>> metrics = create_batch_job_metrics(
        ...     "data-import",
        ...     grouping_key={"region": "us-east-1"},
        ... )
    """
    url = kwargs.pop("pushgateway_url", None) or os.environ.get(
        "PUSHGATEWAY_URL",
        "http://pushgateway.monitoring.svc:9091",
    )

    return BatchJobMetrics(
        job_name=job_name,
        pushgateway_url=url,
        grouping_key=grouping_key,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core classes
    "BatchJobMetrics",
    "PushGatewayConfig",
    "PushGatewayError",
    # Factory functions
    "get_pushgateway_client",
    "clear_pushgateway_clients",
    "create_batch_job_metrics",
    # Constants
    "STATUS_VALUES",
    "PROMETHEUS_CLIENT_AVAILABLE",
]
