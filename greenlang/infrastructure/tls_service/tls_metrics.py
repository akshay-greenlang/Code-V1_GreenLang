# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - TLS Metrics
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
TLS metrics for Prometheus observability.

Provides Prometheus counters, gauges, and histograms for TLS connection
monitoring, handshake performance, and error tracking.

Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_tls_connections_total (Counter): TLS connections by protocol/cipher/status.
    - gl_tls_handshake_duration_seconds (Histogram): Handshake latency.
    - gl_tls_certificate_expiry_days (Gauge): Days until certificate expiry.
    - gl_tls_errors_total (Counter): TLS errors by error_type.
    - gl_tls_active_connections (Gauge): Current active TLS connections.
    - gl_tls_protocol_version (Gauge): Protocol version distribution.

Example:
    >>> from greenlang.infrastructure.tls_service.tls_metrics import (
    ...     TLSMetrics,
    ...     get_tls_metrics,
    ... )
    >>> metrics = get_tls_metrics()
    >>> metrics.record_connection("TLSv1.3", "TLS_AES_256_GCM_SHA384", "success")
    >>> metrics.record_handshake_duration(0.015, "postgresql")

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
# Internal: Lazy Prometheus Metric Handles
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
    connections_total: Any = None
    errors_total: Any = None
    handshakes_total: Any = None
    certificate_verifications: Any = None

    # Gauges
    active_connections: Any = None
    certificate_expiry_days: Any = None
    protocol_version_info: Any = None

    # Histograms
    handshake_duration: Any = None
    connection_duration: Any = None

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
                    "prometheus_client not installed; TLS metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_tls"

            # -- Counters --------------------------------------------------

            cls.connections_total = Counter(
                f"{prefix}_connections_total",
                "Total TLS connections",
                ["protocol", "cipher", "status", "service"],
            )

            cls.errors_total = Counter(
                f"{prefix}_errors_total",
                "TLS errors by type",
                ["error_type", "service"],
            )

            cls.handshakes_total = Counter(
                f"{prefix}_handshakes_total",
                "Total TLS handshakes",
                ["protocol", "status", "service"],
            )

            cls.certificate_verifications = Counter(
                f"{prefix}_certificate_verifications_total",
                "Certificate verification attempts",
                ["result", "service"],
            )

            # -- Gauges ----------------------------------------------------

            cls.active_connections = Gauge(
                f"{prefix}_active_connections",
                "Current active TLS connections",
                ["service"],
            )

            cls.certificate_expiry_days = Gauge(
                f"{prefix}_certificate_expiry_days",
                "Days until certificate expires",
                ["service", "cert_type"],
            )

            cls.protocol_version_info = Gauge(
                f"{prefix}_protocol_version",
                "TLS protocol version in use (1.2=12, 1.3=13)",
                ["service"],
            )

            # -- Histograms ------------------------------------------------

            cls.handshake_duration = Histogram(
                f"{prefix}_handshake_duration_seconds",
                "TLS handshake duration in seconds",
                ["protocol", "service"],
                buckets=(
                    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                ),
            )

            cls.connection_duration = Histogram(
                f"{prefix}_connection_duration_seconds",
                "TLS connection lifetime in seconds",
                ["service"],
                buckets=(
                    1, 5, 10, 30, 60, 300, 600, 1800, 3600,
                ),
            )

            cls._available = True
            logger.info("TLS Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# TLSMetrics
# ---------------------------------------------------------------------------


class TLSMetrics:
    """Manages Prometheus metrics for TLS connections.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed. Thread-safe.

    Example:
        >>> m = TLSMetrics(service="postgresql")
        >>> m.record_connection("TLSv1.3", "TLS_AES_256_GCM_SHA384", "success")
        >>> m.record_handshake_duration(0.015)
        >>> m.record_error("certificate_expired")
    """

    def __init__(self, service: str = "default") -> None:
        """Initialize TLS metrics.

        Args:
            service: Service name label for metrics.
        """
        self._service = service
        self._available = _PrometheusHandles.ensure_initialized()

    @property
    def service(self) -> str:
        """Get the service name for this metrics instance."""
        return self._service

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def record_connection(
        self,
        protocol: str,
        cipher: str,
        status: str,
    ) -> None:
        """Record a TLS connection attempt.

        Args:
            protocol: TLS protocol version (e.g., "TLSv1.3").
            cipher: Cipher suite used.
            status: Connection status ("success", "failure").
        """
        if not self._available:
            return

        _PrometheusHandles.connections_total.labels(
            protocol=protocol,
            cipher=cipher,
            status=status,
            service=self._service,
        ).inc()

    def record_connection_success(
        self,
        protocol: str,
        cipher: str,
    ) -> None:
        """Record a successful TLS connection.

        Args:
            protocol: TLS protocol version.
            cipher: Cipher suite used.
        """
        self.record_connection(protocol, cipher, "success")

    def record_connection_failure(
        self,
        protocol: str = "unknown",
        cipher: str = "unknown",
    ) -> None:
        """Record a failed TLS connection.

        Args:
            protocol: TLS protocol version (if known).
            cipher: Cipher suite (if known).
        """
        self.record_connection(protocol, cipher, "failure")

    def set_active_connections(self, count: int) -> None:
        """Set the current number of active TLS connections.

        Args:
            count: Number of active connections.
        """
        if not self._available:
            return

        _PrometheusHandles.active_connections.labels(
            service=self._service
        ).set(count)

    def inc_active_connections(self) -> None:
        """Increment active connection count by 1."""
        if not self._available:
            return

        _PrometheusHandles.active_connections.labels(
            service=self._service
        ).inc()

    def dec_active_connections(self) -> None:
        """Decrement active connection count by 1."""
        if not self._available:
            return

        _PrometheusHandles.active_connections.labels(
            service=self._service
        ).dec()

    # ------------------------------------------------------------------
    # Handshakes
    # ------------------------------------------------------------------

    def record_handshake(
        self,
        protocol: str,
        status: str,
        duration_s: Optional[float] = None,
    ) -> None:
        """Record a TLS handshake.

        Args:
            protocol: TLS protocol version.
            status: Handshake status ("success", "failure").
            duration_s: Handshake duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.handshakes_total.labels(
            protocol=protocol,
            status=status,
            service=self._service,
        ).inc()

        if duration_s is not None:
            _PrometheusHandles.handshake_duration.labels(
                protocol=protocol,
                service=self._service,
            ).observe(duration_s)

    def record_handshake_duration(
        self,
        duration_s: float,
        protocol: str = "TLSv1.3",
    ) -> None:
        """Record TLS handshake duration.

        Args:
            duration_s: Handshake duration in seconds.
            protocol: TLS protocol version.
        """
        if not self._available:
            return

        _PrometheusHandles.handshake_duration.labels(
            protocol=protocol,
            service=self._service,
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Connection Duration
    # ------------------------------------------------------------------

    def record_connection_duration(self, duration_s: float) -> None:
        """Record TLS connection lifetime.

        Args:
            duration_s: Connection duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.connection_duration.labels(
            service=self._service
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Certificates
    # ------------------------------------------------------------------

    def set_certificate_expiry(
        self,
        days: int,
        cert_type: str = "server",
    ) -> None:
        """Set days until certificate expires.

        Args:
            days: Days until expiry.
            cert_type: Certificate type ("server", "client", "ca").
        """
        if not self._available:
            return

        _PrometheusHandles.certificate_expiry_days.labels(
            service=self._service,
            cert_type=cert_type,
        ).set(days)

    def record_certificate_verification(
        self,
        result: str,
    ) -> None:
        """Record a certificate verification attempt.

        Args:
            result: Verification result ("success", "failure", "expired").
        """
        if not self._available:
            return

        _PrometheusHandles.certificate_verifications.labels(
            result=result,
            service=self._service,
        ).inc()

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def record_error(self, error_type: str) -> None:
        """Record a TLS error.

        Args:
            error_type: Type of error (e.g., "certificate_expired",
                "hostname_mismatch", "handshake_failed", "protocol_error").
        """
        if not self._available:
            return

        _PrometheusHandles.errors_total.labels(
            error_type=error_type,
            service=self._service,
        ).inc()

    # ------------------------------------------------------------------
    # Protocol Version
    # ------------------------------------------------------------------

    def set_protocol_version(self, version: str) -> None:
        """Set the TLS protocol version in use.

        Args:
            version: Protocol version string (e.g., "TLSv1.3").
        """
        if not self._available:
            return

        # Convert version string to numeric for graphing
        version_map = {
            "TLSv1.0": 10,
            "TLSv1.1": 11,
            "TLSv1.2": 12,
            "TLSv1.3": 13,
        }
        numeric = version_map.get(version, 0)

        _PrometheusHandles.protocol_version_info.labels(
            service=self._service
        ).set(numeric)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_metrics: Optional[TLSMetrics] = None
_global_metrics_lock = threading.Lock()


def get_tls_metrics(service: str = "default") -> TLSMetrics:
    """Get or create TLS metrics instance.

    Args:
        service: Service name for metrics labels.

    Returns:
        TLSMetrics instance.
    """
    global _global_metrics

    if service == "default":
        with _global_metrics_lock:
            if _global_metrics is None:
                _global_metrics = TLSMetrics(service="default")
            return _global_metrics
    else:
        return TLSMetrics(service=service)


__all__ = [
    "TLSMetrics",
    "get_tls_metrics",
]
