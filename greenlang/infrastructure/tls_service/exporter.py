# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - Metrics Exporter
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
TLS metrics exporter for Prometheus monitoring.

Scans certificates and exports expiry, protocol, and error metrics.
Provides real-time visibility into TLS configuration and certificate health
across all GreenLang services.

Follows the GreenLang zero-hallucination principle: all certificate inspection
uses the standard library ``ssl`` module and optional ``cryptography`` library.

Classes:
    - CertificateInfo: Dataclass containing certificate metadata.
    - TLSCertificateScanner: Scanner for monitoring TLS certificates.
    - TLSMetricsExporter: Prometheus metrics exporter for TLS.

Functions:
    - get_certificate_info: Async function to get certificate info from endpoint.
    - scan_certificate_sync: Synchronous wrapper for certificate scanning.

Example:
    >>> from greenlang.infrastructure.tls_service.exporter import (
    ...     TLSCertificateScanner,
    ...     get_certificate_info,
    ... )
    >>> scanner = TLSCertificateScanner([("api.greenlang.io", 443)])
    >>> certs = await scanner.scan_all()
    >>> for cert in certs:
    ...     print(f"{cert.host}: expires in {cert.days_until_expiry} days")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import socket
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CertificateInfo:
    """
    Certificate information extracted from a TLS endpoint.

    Attributes:
        host: Hostname of the endpoint.
        port: Port number.
        subject: Certificate subject common name.
        issuer: Certificate issuer common name.
        not_before: Certificate validity start date.
        not_after: Certificate validity end date.
        days_until_expiry: Days remaining until certificate expires.
        serial_number: Certificate serial number (hex).
        tls_version: Negotiated TLS protocol version.
        cipher_suite: Negotiated cipher suite name.
        valid: Whether the certificate is valid and trusted.
        error: Error message if connection or validation failed.
        san: Subject Alternative Names.
        fingerprint_sha256: SHA-256 fingerprint of the certificate.
        key_size: Public key size in bits.
        signature_algorithm: Signature algorithm used.
    """

    host: str
    port: int
    subject: str
    issuer: str
    not_before: datetime
    not_after: datetime
    days_until_expiry: int
    serial_number: str
    tls_version: str
    cipher_suite: str
    valid: bool
    error: Optional[str] = None
    san: List[str] = field(default_factory=list)
    fingerprint_sha256: Optional[str] = None
    key_size: Optional[int] = None
    signature_algorithm: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if the certificate is expired."""
        return self.days_until_expiry <= 0

    @property
    def is_expiring_soon(self) -> bool:
        """Check if certificate expires within 14 days."""
        return 0 < self.days_until_expiry <= 14

    @property
    def is_critical(self) -> bool:
        """Check if certificate expires within 7 days."""
        return 0 < self.days_until_expiry <= 7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "subject": self.subject,
            "issuer": self.issuer,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "days_until_expiry": self.days_until_expiry,
            "serial_number": self.serial_number,
            "tls_version": self.tls_version,
            "cipher_suite": self.cipher_suite,
            "valid": self.valid,
            "error": self.error,
            "san": self.san,
            "fingerprint_sha256": self.fingerprint_sha256,
            "key_size": self.key_size,
            "signature_algorithm": self.signature_algorithm,
        }


# ---------------------------------------------------------------------------
# Certificate Info Retrieval
# ---------------------------------------------------------------------------


async def get_certificate_info(
    host: str,
    port: int = 443,
    timeout: float = 10.0,
    verify: bool = True,
) -> CertificateInfo:
    """
    Get certificate information from a TLS endpoint.

    Connects to the specified host and port, performs a TLS handshake, and
    extracts certificate metadata including expiry, subject, issuer, and
    cryptographic properties.

    Args:
        host: Hostname to connect to.
        port: Port number (default 443).
        timeout: Connection timeout in seconds.
        verify: Whether to verify the certificate chain.

    Returns:
        CertificateInfo with certificate details.

    Note:
        This function is async but uses synchronous socket operations
        internally. For high-volume scanning, use TLSCertificateScanner
        which manages concurrent connections efficiently.
    """
    try:
        # Create SSL context
        context = ssl.create_default_context()
        if not verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

        # Connect and get certificate
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                cert_bin = ssock.getpeercert(binary_form=True)
                cipher = ssock.cipher()
                version = ssock.version()

                # Parse certificate dates
                not_before = _parse_cert_date(cert.get("notBefore", ""))
                not_after = _parse_cert_date(cert.get("notAfter", ""))

                now = datetime.now(timezone.utc)
                days_until_expiry = (not_after - now).days if not_after else -1

                # Extract subject and issuer
                subject = _extract_cn(cert.get("subject", []))
                issuer = _extract_cn(cert.get("issuer", []))

                # Extract SAN
                san = _extract_san(cert.get("subjectAltName", []))

                # Calculate fingerprint
                fingerprint = _calculate_fingerprint(cert_bin) if cert_bin else None

                return CertificateInfo(
                    host=host,
                    port=port,
                    subject=subject,
                    issuer=issuer,
                    not_before=not_before,
                    not_after=not_after,
                    days_until_expiry=days_until_expiry,
                    serial_number=str(cert.get("serialNumber", "Unknown")),
                    tls_version=version or "Unknown",
                    cipher_suite=cipher[0] if cipher else "Unknown",
                    valid=True,
                    san=san,
                    fingerprint_sha256=fingerprint,
                )

    except ssl.SSLCertVerificationError as e:
        logger.warning(f"Certificate verification failed for {host}:{port}: {e}")
        return _create_error_cert_info(host, port, f"Verification failed: {e}")

    except ssl.SSLError as e:
        logger.warning(f"SSL error for {host}:{port}: {e}")
        return _create_error_cert_info(host, port, f"SSL error: {e}")

    except socket.timeout:
        logger.warning(f"Connection timeout for {host}:{port}")
        return _create_error_cert_info(host, port, "Connection timeout")

    except socket.gaierror as e:
        logger.warning(f"DNS resolution failed for {host}:{port}: {e}")
        return _create_error_cert_info(host, port, f"DNS resolution failed: {e}")

    except ConnectionRefusedError:
        logger.warning(f"Connection refused for {host}:{port}")
        return _create_error_cert_info(host, port, "Connection refused")

    except Exception as e:
        logger.error(f"Failed to get certificate info for {host}:{port}: {e}")
        return _create_error_cert_info(host, port, str(e))


def scan_certificate_sync(
    host: str,
    port: int = 443,
    timeout: float = 10.0,
    verify: bool = True,
) -> CertificateInfo:
    """
    Synchronous wrapper for get_certificate_info.

    Args:
        host: Hostname to connect to.
        port: Port number.
        timeout: Connection timeout in seconds.
        verify: Whether to verify certificate chain.

    Returns:
        CertificateInfo with certificate details.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        get_certificate_info(host, port, timeout, verify)
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _parse_cert_date(date_str: str) -> datetime:
    """Parse certificate date string to datetime."""
    if not date_str:
        return datetime.min.replace(tzinfo=timezone.utc)

    try:
        # Format: "Feb  5 12:34:56 2026 GMT"
        dt = datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            # Alternative format
            dt = datetime.strptime(date_str, "%b  %d %H:%M:%S %Y %Z")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Failed to parse certificate date: {date_str}")
            return datetime.min.replace(tzinfo=timezone.utc)


def _extract_cn(name_tuple: tuple) -> str:
    """Extract Common Name from certificate name tuple."""
    for rdn in name_tuple:
        for name, value in rdn:
            if name == "commonName":
                return value
    return "Unknown"


def _extract_san(san_tuple: tuple) -> List[str]:
    """Extract Subject Alternative Names."""
    san_list = []
    for san_type, san_value in san_tuple:
        if san_type == "DNS":
            san_list.append(san_value)
        elif san_type == "IP Address":
            san_list.append(f"IP:{san_value}")
    return san_list


def _calculate_fingerprint(cert_der: bytes) -> str:
    """Calculate SHA-256 fingerprint of certificate."""
    import hashlib
    digest = hashlib.sha256(cert_der).hexdigest()
    return ":".join(digest[i:i+2].upper() for i in range(0, len(digest), 2))


def _create_error_cert_info(host: str, port: int, error: str) -> CertificateInfo:
    """Create CertificateInfo for error cases."""
    return CertificateInfo(
        host=host,
        port=port,
        subject="Unknown",
        issuer="Unknown",
        not_before=datetime.min.replace(tzinfo=timezone.utc),
        not_after=datetime.min.replace(tzinfo=timezone.utc),
        days_until_expiry=-1,
        serial_number="Unknown",
        tls_version="Unknown",
        cipher_suite="Unknown",
        valid=False,
        error=error,
    )


# ---------------------------------------------------------------------------
# TLS Certificate Scanner
# ---------------------------------------------------------------------------


class TLSCertificateScanner:
    """
    Scanner for monitoring TLS certificates across multiple endpoints.

    Provides efficient concurrent scanning of multiple endpoints and
    maintains a cache of certificate information for quick access.

    Example:
        >>> scanner = TLSCertificateScanner([
        ...     ("api.greenlang.io", 443),
        ...     ("greenlang.io", 443),
        ...     ("postgres.internal", 5432),
        ... ])
        >>> certs = await scanner.scan_all()
        >>> expiring = scanner.get_expiring_soon(days=14)
    """

    def __init__(
        self,
        endpoints: List[Tuple[str, int]],
        timeout: float = 10.0,
        verify: bool = True,
    ):
        """
        Initialize scanner with endpoints to monitor.

        Args:
            endpoints: List of (host, port) tuples.
            timeout: Connection timeout in seconds.
            verify: Whether to verify certificate chains.
        """
        self.endpoints = endpoints
        self.timeout = timeout
        self.verify = verify
        self._cache: Dict[str, CertificateInfo] = {}
        self._last_scan: Optional[datetime] = None
        self._scan_duration: Optional[float] = None

    async def scan_all(self) -> List[CertificateInfo]:
        """
        Scan all configured endpoints concurrently.

        Returns:
            List of CertificateInfo for all endpoints.
        """
        start_time = time.perf_counter()

        tasks = [
            get_certificate_info(host, port, self.timeout, self.verify)
            for host, port in self.endpoints
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        certs = []
        for result in results:
            if isinstance(result, CertificateInfo):
                self._cache[f"{result.host}:{result.port}"] = result
                certs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scan error: {result}")

        self._last_scan = datetime.now(timezone.utc)
        self._scan_duration = time.perf_counter() - start_time

        logger.info(
            f"Scanned {len(certs)} certificates in {self._scan_duration:.2f}s"
        )

        return certs

    def scan_all_sync(self) -> List[CertificateInfo]:
        """Synchronous wrapper for scan_all."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.scan_all())

    def get_cached(self, host: str, port: int) -> Optional[CertificateInfo]:
        """Get cached certificate info for an endpoint."""
        return self._cache.get(f"{host}:{port}")

    def get_all_cached(self) -> List[CertificateInfo]:
        """Get all cached certificate info."""
        return list(self._cache.values())

    def get_expiring_soon(self, days: int = 14) -> List[CertificateInfo]:
        """
        Get certificates expiring within specified days.

        Args:
            days: Number of days threshold.

        Returns:
            List of certificates expiring within the threshold.
        """
        return [
            cert for cert in self._cache.values()
            if cert.valid and 0 < cert.days_until_expiry <= days
        ]

    def get_expired(self) -> List[CertificateInfo]:
        """Get expired certificates."""
        return [
            cert for cert in self._cache.values()
            if cert.valid and cert.days_until_expiry <= 0
        ]

    def get_invalid(self) -> List[CertificateInfo]:
        """Get certificates with validation errors."""
        return [
            cert for cert in self._cache.values()
            if not cert.valid
        ]

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """Get timestamp of last scan."""
        return self._last_scan

    @property
    def last_scan_duration(self) -> Optional[float]:
        """Get duration of last scan in seconds."""
        return self._scan_duration


# ---------------------------------------------------------------------------
# Prometheus Metrics Exporter
# ---------------------------------------------------------------------------


class TLSMetricsExporter:
    """
    Prometheus metrics exporter for TLS certificate monitoring.

    Exports metrics for certificate expiry, TLS protocol versions,
    cipher suites, and connection errors.

    Metrics exported:
        - gl_tls_certificate_expiry_seconds: Time until certificate expires
        - gl_tls_certificate_valid: Certificate validity status
        - gl_tls_connections_total: Total TLS connections by protocol/cipher
        - gl_tls_errors_total: TLS error count by type
        - gl_tls_handshake_duration_seconds: TLS handshake latency histogram
        - gl_tls_protocol_downgrades_total: Blocked protocol downgrade attempts
        - gl_mtls_enforcement_enabled: mTLS enforcement status

    Example:
        >>> exporter = TLSMetricsExporter()
        >>> exporter.record_connection("TLSv1.3", "TLS_AES_256_GCM_SHA384")
        >>> exporter.record_certificate_expiry("api.greenlang.io", 45)
    """

    def __init__(self):
        """Initialize the metrics exporter."""
        self._metrics: Dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics if available."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, Info

            self._metrics["certificate_expiry"] = Gauge(
                "gl_tls_certificate_expiry_seconds",
                "Seconds until TLS certificate expires",
                ["domain", "port"],
            )

            self._metrics["certificate_valid"] = Gauge(
                "gl_tls_certificate_valid",
                "TLS certificate validity status (1=valid, 0=invalid)",
                ["domain", "port"],
            )

            self._metrics["connections_total"] = Counter(
                "gl_tls_connections_total",
                "Total TLS connections",
                ["protocol", "cipher"],
            )

            self._metrics["errors_total"] = Counter(
                "gl_tls_errors_total",
                "Total TLS errors",
                ["error_type"],
            )

            self._metrics["handshake_duration"] = Histogram(
                "gl_tls_handshake_duration_seconds",
                "TLS handshake duration in seconds",
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            )

            self._metrics["protocol_downgrades"] = Counter(
                "gl_tls_protocol_downgrades_total",
                "Blocked TLS protocol downgrade attempts",
                ["attempted_protocol"],
            )

            self._metrics["mtls_enforcement"] = Gauge(
                "gl_mtls_enforcement_enabled",
                "mTLS enforcement status (1=enabled, 0=disabled)",
                ["namespace"],
            )

            self._metrics["scan_duration"] = Gauge(
                "gl_tls_scan_duration_seconds",
                "Duration of last certificate scan",
            )

            logger.info("Prometheus TLS metrics initialized")

        except ImportError:
            logger.warning("prometheus_client not available; metrics disabled")

    def record_certificate_expiry(
        self,
        domain: str,
        days: int,
        port: int = 443,
        valid: bool = True,
    ) -> None:
        """
        Record certificate expiry metric.

        Args:
            domain: Domain name.
            days: Days until expiry.
            port: Port number.
            valid: Whether certificate is valid.
        """
        if "certificate_expiry" in self._metrics:
            self._metrics["certificate_expiry"].labels(
                domain=domain, port=str(port)
            ).set(days * 86400)  # Convert to seconds

        if "certificate_valid" in self._metrics:
            self._metrics["certificate_valid"].labels(
                domain=domain, port=str(port)
            ).set(1 if valid else 0)

    def record_connection(
        self,
        protocol: str,
        cipher: str,
        duration: Optional[float] = None,
    ) -> None:
        """
        Record TLS connection metric.

        Args:
            protocol: TLS protocol version (e.g., "TLSv1.3").
            cipher: Cipher suite name.
            duration: Handshake duration in seconds.
        """
        if "connections_total" in self._metrics:
            self._metrics["connections_total"].labels(
                protocol=protocol, cipher=cipher
            ).inc()

        if duration is not None and "handshake_duration" in self._metrics:
            self._metrics["handshake_duration"].observe(duration)

    def record_error(self, error_type: str) -> None:
        """
        Record TLS error metric.

        Args:
            error_type: Type of error (e.g., "handshake_failure").
        """
        if "errors_total" in self._metrics:
            self._metrics["errors_total"].labels(error_type=error_type).inc()

    def record_protocol_downgrade(self, attempted_protocol: str) -> None:
        """
        Record blocked protocol downgrade attempt.

        Args:
            attempted_protocol: Protocol that was blocked (e.g., "TLSv1.0").
        """
        if "protocol_downgrades" in self._metrics:
            self._metrics["protocol_downgrades"].labels(
                attempted_protocol=attempted_protocol
            ).inc()

    def record_mtls_status(self, namespace: str, enabled: bool) -> None:
        """
        Record mTLS enforcement status.

        Args:
            namespace: Kubernetes namespace.
            enabled: Whether mTLS is enforced.
        """
        if "mtls_enforcement" in self._metrics:
            self._metrics["mtls_enforcement"].labels(
                namespace=namespace
            ).set(1 if enabled else 0)

    def record_scan_duration(self, duration: float) -> None:
        """
        Record certificate scan duration.

        Args:
            duration: Scan duration in seconds.
        """
        if "scan_duration" in self._metrics:
            self._metrics["scan_duration"].set(duration)

    def update_from_scanner(self, scanner: TLSCertificateScanner) -> None:
        """
        Update metrics from a TLSCertificateScanner.

        Args:
            scanner: Scanner with cached certificate info.
        """
        for cert in scanner.get_all_cached():
            self.record_certificate_expiry(
                domain=cert.host,
                days=cert.days_until_expiry,
                port=cert.port,
                valid=cert.valid,
            )

        if scanner.last_scan_duration:
            self.record_scan_duration(scanner.last_scan_duration)


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------


# Default endpoints to monitor
DEFAULT_ENDPOINTS: List[Tuple[str, int]] = [
    ("api.greenlang.io", 443),
    ("greenlang.io", 443),
]


# Singleton exporter instance
_exporter: Optional[TLSMetricsExporter] = None


def get_metrics_exporter() -> TLSMetricsExporter:
    """Get the singleton TLS metrics exporter instance."""
    global _exporter
    if _exporter is None:
        _exporter = TLSMetricsExporter()
    return _exporter


__all__ = [
    # Data classes
    "CertificateInfo",
    # Functions
    "get_certificate_info",
    "scan_certificate_sync",
    # Classes
    "TLSCertificateScanner",
    "TLSMetricsExporter",
    # Helpers
    "get_metrics_exporter",
    "DEFAULT_ENDPOINTS",
]
