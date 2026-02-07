# -*- coding: utf-8 -*-
"""
Encryption Metrics - SEC-003: Encryption at Rest

Prometheus counters, gauges, and histograms for encryption service observability.
Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_encryption_operations_total (Counter): Operations by type/status/data_class.
    - gl_encryption_duration_seconds (Histogram): Operation latency by type.
    - gl_encryption_key_cache_hits_total (Counter): DEK cache hits by key_type.
    - gl_encryption_key_cache_misses_total (Counter): DEK cache misses by key_type.
    - gl_encryption_failures_total (Counter): Failures by error_type.
    - gl_kms_calls_total (Counter): KMS API calls by operation/status.
    - gl_kms_latency_seconds (Histogram): KMS API call latency by operation.
    - gl_encryption_active_keys (Gauge): Active encryption keys by key_type.
    - gl_encryption_cache_size (Gauge): Current DEK cache size.
    - gl_encryption_bytes_processed (Counter): Bytes encrypted/decrypted.

Classes:
    - EncryptionMetrics: Metrics manager for the encryption service.

Example:
    >>> metrics = EncryptionMetrics()
    >>> metrics.record_operation("encrypt", "success", "pii", duration_s=0.001)
    >>> metrics.record_cache_hit("dek")
    >>> metrics.record_kms_call("GenerateDataKey", "success", duration_s=0.05)

Author: GreenLang Framework Team
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
    operations_total: Any = None
    key_cache_hits: Any = None
    key_cache_misses: Any = None
    failures_total: Any = None
    kms_calls_total: Any = None
    bytes_processed: Any = None

    # Gauges
    active_keys: Any = None
    cache_size: Any = None

    # Histograms
    duration_seconds: Any = None
    kms_latency_seconds: Any = None

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
                    "prometheus_client not installed; encryption metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_encryption"

            # -- Counters --------------------------------------------------

            cls.operations_total = Counter(
                f"{prefix}_operations_total",
                "Total encryption operations",
                ["operation", "status", "data_class"],
            )

            cls.key_cache_hits = Counter(
                f"{prefix}_key_cache_hits_total",
                "DEK cache hits",
                ["key_type"],
            )

            cls.key_cache_misses = Counter(
                f"{prefix}_key_cache_misses_total",
                "DEK cache misses",
                ["key_type"],
            )

            cls.failures_total = Counter(
                f"{prefix}_failures_total",
                "Encryption failures by error type",
                ["error_type"],
            )

            cls.kms_calls_total = Counter(
                "gl_kms_calls_total",
                "KMS API calls",
                ["operation", "status"],
            )

            cls.bytes_processed = Counter(
                f"{prefix}_bytes_processed_total",
                "Total bytes encrypted/decrypted",
                ["operation"],
            )

            # -- Gauges ----------------------------------------------------

            cls.active_keys = Gauge(
                f"{prefix}_active_keys",
                "Number of active encryption keys",
                ["key_type"],
            )

            cls.cache_size = Gauge(
                f"{prefix}_cache_size",
                "Current DEK cache size",
                ["cache_type"],
            )

            # -- Histograms ------------------------------------------------

            cls.duration_seconds = Histogram(
                f"{prefix}_duration_seconds",
                "Encryption operation duration in seconds",
                ["operation"],
                buckets=(
                    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                ),
            )

            cls.kms_latency_seconds = Histogram(
                "gl_kms_latency_seconds",
                "KMS API call latency in seconds",
                ["operation"],
                buckets=(
                    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
                ),
            )

            cls._available = True
            logger.info("Encryption Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# EncryptionMetrics
# ---------------------------------------------------------------------------


class EncryptionMetrics:
    """Manages Prometheus metrics for the encryption service.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed. Thread-safe.

    Example:
        >>> m = EncryptionMetrics()
        >>> m.record_operation("encrypt", "success", "pii", duration_s=0.001)
        >>> m.record_cache_hit("dek")
        >>> m.record_kms_call("GenerateDataKey", "success", duration_s=0.05)
    """

    def __init__(self, prefix: str = "gl_encryption") -> None:
        """Initialize encryption metrics.

        Args:
            prefix: Metric name prefix (used for documentation only;
                actual prefix is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def record_operation(
        self,
        operation: str,
        status: str,
        data_class: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """Record an encryption operation.

        Args:
            operation: Operation type (``encrypt``, ``decrypt``).
            status: Outcome (``success``, ``failure``).
            data_class: Data classification (``pii``, ``secret``, ``confidential``).
            duration_s: Operation duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.operations_total.labels(
            operation=operation, status=status, data_class=data_class
        ).inc()

        if duration_s is not None:
            _PrometheusHandles.duration_seconds.labels(
                operation=operation
            ).observe(duration_s)

    def record_encryption(
        self,
        status: str,
        data_class: str,
        duration_s: Optional[float] = None,
        bytes_count: Optional[int] = None,
    ) -> None:
        """Record an encryption operation.

        Args:
            status: Outcome (``success``, ``failure``).
            data_class: Data classification level.
            duration_s: Operation duration in seconds.
            bytes_count: Number of bytes encrypted.
        """
        self.record_operation("encrypt", status, data_class, duration_s)
        if bytes_count is not None and self._available:
            _PrometheusHandles.bytes_processed.labels(
                operation="encrypt"
            ).inc(bytes_count)

    def record_decryption(
        self,
        status: str,
        data_class: str,
        duration_s: Optional[float] = None,
        bytes_count: Optional[int] = None,
    ) -> None:
        """Record a decryption operation.

        Args:
            status: Outcome (``success``, ``failure``).
            data_class: Data classification level.
            duration_s: Operation duration in seconds.
            bytes_count: Number of bytes decrypted.
        """
        self.record_operation("decrypt", status, data_class, duration_s)
        if bytes_count is not None and self._available:
            _PrometheusHandles.bytes_processed.labels(
                operation="decrypt"
            ).inc(bytes_count)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def record_cache_hit(self, key_type: str = "dek") -> None:
        """Record a DEK cache hit.

        Args:
            key_type: Type of key (``dek``).
        """
        if not self._available:
            return

        _PrometheusHandles.key_cache_hits.labels(key_type=key_type).inc()

    def record_cache_miss(self, key_type: str = "dek") -> None:
        """Record a DEK cache miss.

        Args:
            key_type: Type of key (``dek``).
        """
        if not self._available:
            return

        _PrometheusHandles.key_cache_misses.labels(key_type=key_type).inc()

    def set_cache_size(self, size: int, cache_type: str = "dek") -> None:
        """Set the current cache size.

        Args:
            size: Current number of cached keys.
            cache_type: Type of cache (``dek``).
        """
        if not self._available:
            return

        _PrometheusHandles.cache_size.labels(cache_type=cache_type).set(size)

    # ------------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------------

    def record_failure(self, error_type: str) -> None:
        """Record an encryption failure.

        Args:
            error_type: Type of error (``validation``, ``kms``, ``integrity``,
                ``key_expired``, ``context_mismatch``).
        """
        if not self._available:
            return

        _PrometheusHandles.failures_total.labels(error_type=error_type).inc()

    # ------------------------------------------------------------------
    # KMS
    # ------------------------------------------------------------------

    def record_kms_call(
        self,
        operation: str,
        status: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """Record a KMS API call.

        Args:
            operation: KMS operation (``GenerateDataKey``, ``Encrypt``,
                ``Decrypt``).
            status: Outcome (``success``, ``failure``).
            duration_s: Call duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.kms_calls_total.labels(
            operation=operation, status=status
        ).inc()

        if duration_s is not None:
            _PrometheusHandles.kms_latency_seconds.labels(
                operation=operation
            ).observe(duration_s)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def set_active_keys(self, key_type: str, count: int) -> None:
        """Set the number of active encryption keys.

        Args:
            key_type: Type of key (``dek``, ``kek``).
            count: Current active key count.
        """
        if not self._available:
            return

        _PrometheusHandles.active_keys.labels(key_type=key_type).set(count)

    def inc_active_keys(self, key_type: str = "dek") -> None:
        """Increment the active key count by 1.

        Args:
            key_type: Type of key (``dek``, ``kek``).
        """
        if not self._available:
            return

        _PrometheusHandles.active_keys.labels(key_type=key_type).inc()

    def dec_active_keys(self, key_type: str = "dek") -> None:
        """Decrement the active key count by 1.

        Args:
            key_type: Type of key (``dek``, ``kek``).
        """
        if not self._available:
            return

        _PrometheusHandles.active_keys.labels(key_type=key_type).dec()


__all__ = ["EncryptionMetrics"]
