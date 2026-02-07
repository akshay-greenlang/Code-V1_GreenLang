# -*- coding: utf-8 -*-
"""
Secrets Service Metrics - SEC-006 Deploy Secrets Management

Prometheus counters, gauges, and histograms for secrets management observability.
Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_secrets_operations_total (Counter): Operations by type/secret_type/result.
    - gl_secrets_operation_duration_seconds (Histogram): Operation latency.
    - gl_vault_auth_renewals_total (Counter): Vault auth token renewals by status.
    - gl_vault_lease_ttl_seconds (Gauge): Remaining TTL for leases.
    - gl_secrets_cache_hits_total (Counter): Cache hits.
    - gl_secrets_cache_misses_total (Counter): Cache misses.
    - gl_secrets_rotation_total (Counter): Rotation events by type/status.
    - gl_eso_sync_total (Counter): External Secrets Operator sync events.
    - gl_secrets_rotation_last_success_timestamp (Gauge): Last successful rotation.
    - gl_eso_sync_last_success_timestamp (Gauge): Last successful ESO sync.

Classes:
    - SecretsMetrics: Singleton-style metrics manager with convenience methods.

Example:
    >>> metrics = SecretsMetrics()
    >>> metrics.record_operation("read", "database", "success", duration_s=0.005)
    >>> metrics.record_cache_hit()
    >>> with metrics.latency_timer("write"):
    ...     vault_client.write_secret(path, data)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

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
    operations_total: Any = None
    vault_auth_renewals_total: Any = None
    cache_hits_total: Any = None
    cache_misses_total: Any = None
    rotation_total: Any = None
    eso_sync_total: Any = None

    # Gauges
    vault_lease_ttl_seconds: Any = None
    rotation_last_success_timestamp: Any = None
    eso_sync_last_success_timestamp: Any = None

    # Histograms
    operation_duration_seconds: Any = None

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
                    "prometheus_client not installed; secrets metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_secrets"
            vault_prefix = "gl_vault"

            # -- Counters --------------------------------------------------

            cls.operations_total = Counter(
                f"{prefix}_operations_total",
                "Total secrets operations",
                ["operation", "secret_type", "result"],
            )

            cls.vault_auth_renewals_total = Counter(
                f"{vault_prefix}_auth_renewals_total",
                "Vault authentication token renewals",
                ["status"],
            )

            cls.cache_hits_total = Counter(
                f"{prefix}_cache_hits_total",
                "Total secrets cache hits",
            )

            cls.cache_misses_total = Counter(
                f"{prefix}_cache_misses_total",
                "Total secrets cache misses",
            )

            cls.rotation_total = Counter(
                f"{prefix}_rotation_total",
                "Total secret rotation events",
                ["secret_type", "status"],
            )

            cls.eso_sync_total = Counter(
                f"{prefix[:-1]}_eso_sync_total",  # gl_eso_sync_total
                "External Secrets Operator sync events",
                ["secret", "status"],
            )

            # -- Gauges ----------------------------------------------------

            cls.vault_lease_ttl_seconds = Gauge(
                f"{vault_prefix}_lease_ttl_seconds",
                "Remaining TTL in seconds for Vault leases",
                ["secret_type", "path"],
            )

            cls.rotation_last_success_timestamp = Gauge(
                f"{prefix}_rotation_last_success_timestamp",
                "Unix timestamp of last successful rotation",
                ["secret_type"],
            )

            cls.eso_sync_last_success_timestamp = Gauge(
                f"{prefix[:-1]}_eso_sync_last_success_timestamp",
                "Unix timestamp of last successful ESO sync",
                ["secret"],
            )

            # -- Histograms ------------------------------------------------

            cls.operation_duration_seconds = Histogram(
                f"{prefix}_operation_duration_seconds",
                "Secrets operation duration in seconds",
                ["operation"],
                buckets=(
                    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                ),
            )

            cls._available = True
            logger.info("Secrets Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# SecretsMetrics
# ---------------------------------------------------------------------------


class SecretsMetrics:
    """Manages Prometheus metrics for the secrets service.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed. Thread-safe.

    Example:
        >>> m = SecretsMetrics()
        >>> m.record_operation("read", "database", "success", duration_s=0.003)
        >>> m.record_cache_hit()
        >>> m.record_rotation("api_key", "success")
    """

    def __init__(self, prefix: str = "gl_secrets") -> None:
        """Initialize secrets metrics.

        Args:
            prefix: Metric name prefix (used for documentation only;
                actual prefix is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    @property
    def available(self) -> bool:
        """Return True if Prometheus metrics are available."""
        return self._available

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def record_operation(
        self,
        operation: str,
        secret_type: str,
        result: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """Record a secrets operation.

        Args:
            operation: Operation type (``read``, ``write``, ``delete``,
                ``list``, ``rotate``).
            secret_type: Type of secret (``database``, ``api_key``,
                ``certificate``, ``encryption_key``, ``oauth``).
            result: Outcome (``success``, ``failure``, ``denied``).
            duration_s: Operation duration in seconds (optional).
        """
        if not self._available:
            return

        _PrometheusHandles.operations_total.labels(
            operation=operation,
            secret_type=secret_type,
            result=result,
        ).inc()

        if duration_s is not None:
            _PrometheusHandles.operation_duration_seconds.labels(
                operation=operation
            ).observe(duration_s)

    def observe_operation_duration(
        self,
        operation: str,
        duration_s: float,
    ) -> None:
        """Record operation duration in the histogram.

        Args:
            operation: Operation type.
            duration_s: Duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.operation_duration_seconds.labels(
            operation=operation
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Vault auth renewals
    # ------------------------------------------------------------------

    def record_vault_auth_renewal(self, status: str) -> None:
        """Record a Vault authentication token renewal attempt.

        Args:
            status: Renewal status (``success``, ``failure``).
        """
        if not self._available:
            return

        _PrometheusHandles.vault_auth_renewals_total.labels(
            status=status
        ).inc()

    # ------------------------------------------------------------------
    # Lease TTL tracking
    # ------------------------------------------------------------------

    def set_lease_ttl(
        self,
        secret_type: str,
        path: str,
        ttl_seconds: float,
    ) -> None:
        """Set the remaining TTL for a Vault lease.

        Args:
            secret_type: Type of secret.
            path: Vault path for the lease.
            ttl_seconds: Remaining TTL in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.vault_lease_ttl_seconds.labels(
            secret_type=secret_type,
            path=path,
        ).set(ttl_seconds)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def record_cache_hit(self) -> None:
        """Record a secrets cache hit."""
        if not self._available:
            return

        _PrometheusHandles.cache_hits_total.inc()

    def record_cache_miss(self) -> None:
        """Record a secrets cache miss."""
        if not self._available:
            return

        _PrometheusHandles.cache_misses_total.inc()

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def record_rotation(self, secret_type: str, status: str) -> None:
        """Record a secret rotation event.

        Args:
            secret_type: Type of secret being rotated.
            status: Rotation status (``success``, ``failure``, ``skipped``).
        """
        if not self._available:
            return

        _PrometheusHandles.rotation_total.labels(
            secret_type=secret_type,
            status=status,
        ).inc()

        if status == "success":
            _PrometheusHandles.rotation_last_success_timestamp.labels(
                secret_type=secret_type
            ).set(time.time())

    def set_rotation_last_success(
        self,
        secret_type: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Set the last successful rotation timestamp.

        Args:
            secret_type: Type of secret.
            timestamp: Unix timestamp (defaults to now).
        """
        if not self._available:
            return

        _PrometheusHandles.rotation_last_success_timestamp.labels(
            secret_type=secret_type
        ).set(timestamp or time.time())

    # ------------------------------------------------------------------
    # ESO (External Secrets Operator)
    # ------------------------------------------------------------------

    def record_eso_sync(self, secret: str, status: str) -> None:
        """Record an External Secrets Operator sync event.

        Args:
            secret: Name of the ExternalSecret resource.
            status: Sync status (``success``, ``failure``).
        """
        if not self._available:
            return

        _PrometheusHandles.eso_sync_total.labels(
            secret=secret,
            status=status,
        ).inc()

        if status == "success":
            _PrometheusHandles.eso_sync_last_success_timestamp.labels(
                secret=secret
            ).set(time.time())

    def set_eso_sync_last_success(
        self,
        secret: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Set the last successful ESO sync timestamp.

        Args:
            secret: Name of the ExternalSecret resource.
            timestamp: Unix timestamp (defaults to now).
        """
        if not self._available:
            return

        _PrometheusHandles.eso_sync_last_success_timestamp.labels(
            secret=secret
        ).set(timestamp or time.time())

    # ------------------------------------------------------------------
    # Latency timer context manager
    # ------------------------------------------------------------------

    @contextmanager
    def latency_timer(
        self,
        operation: str,
    ) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Automatically records the duration to the histogram when the
        context exits.

        Args:
            operation: Operation type for labeling the metric.

        Yields:
            None

        Example:
            >>> with metrics.latency_timer("read"):
            ...     result = vault_client.read(path)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_operation_duration(operation, duration)


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_metrics: Optional[SecretsMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> SecretsMetrics:
    """Get the default SecretsMetrics singleton.

    Thread-safe lazy initialization.

    Returns:
        SecretsMetrics instance.

    Example:
        >>> from greenlang.infrastructure.secrets_service.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_cache_hit()
    """
    global _default_metrics
    if _default_metrics is None:
        with _metrics_lock:
            if _default_metrics is None:
                _default_metrics = SecretsMetrics()
    return _default_metrics


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "SecretsMetrics",
    "get_metrics",
]
