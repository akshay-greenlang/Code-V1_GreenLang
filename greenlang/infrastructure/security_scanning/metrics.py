# -*- coding: utf-8 -*-
"""
Security Scanning Metrics - SEC-007 Security Scanning Pipeline

Prometheus counters, gauges, and histograms for security scanning observability.
Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_security_vulnerabilities_total (Counter): Total vulnerabilities by severity/status.
    - gl_security_scan_duration_seconds (Histogram): Scan duration by scanner.
    - gl_security_findings_total (Counter): Total findings by scanner/severity.
    - gl_security_remediation_days (Histogram): Days to remediation by severity.
    - gl_security_images_signed_total (Counter): Total signed container images.
    - gl_security_sbom_components_total (Gauge): SBOM component count.
    - gl_security_sla_breach_total (Counter): SLA breaches by severity.
    - gl_security_pii_findings_total (Counter): PII findings by classification.

Classes:
    - SecurityMetrics: Singleton-style metrics manager.

Example:
    >>> metrics = SecurityMetrics()
    >>> metrics.record_scan_duration("trivy", 45.3)
    >>> metrics.record_vulnerability_found("critical", "open")

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
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
    vulnerabilities_total: Any = None
    findings_total: Any = None
    images_signed_total: Any = None
    sla_breach_total: Any = None
    pii_findings_total: Any = None
    scans_total: Any = None
    scan_errors_total: Any = None

    # Gauges
    vulnerabilities_open: Any = None
    sbom_components_total: Any = None
    scanner_health: Any = None
    coverage_score: Any = None

    # Histograms
    scan_duration_seconds: Any = None
    remediation_days: Any = None
    risk_score_distribution: Any = None

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
                    "prometheus_client not installed; security metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_security"

            # -- Counters --------------------------------------------------

            cls.vulnerabilities_total = Counter(
                f"{prefix}_vulnerabilities_total",
                "Total vulnerabilities discovered",
                ["severity", "status"],
            )

            cls.findings_total = Counter(
                f"{prefix}_findings_total",
                "Total scanner findings",
                ["scanner", "severity", "finding_type"],
            )

            cls.images_signed_total = Counter(
                f"{prefix}_images_signed_total",
                "Total container images signed",
                ["registry"],
            )

            cls.sla_breach_total = Counter(
                f"{prefix}_sla_breach_total",
                "Total SLA breaches",
                ["severity"],
            )

            cls.pii_findings_total = Counter(
                f"{prefix}_pii_findings_total",
                "Total PII findings",
                ["classification", "pii_type"],
            )

            cls.scans_total = Counter(
                f"{prefix}_scans_total",
                "Total security scans executed",
                ["scanner", "target_type", "status"],
            )

            cls.scan_errors_total = Counter(
                f"{prefix}_scan_errors_total",
                "Total scan errors",
                ["scanner", "error_type"],
            )

            # -- Gauges ----------------------------------------------------

            cls.vulnerabilities_open = Gauge(
                f"{prefix}_vulnerabilities_open",
                "Currently open vulnerabilities",
                ["severity"],
            )

            cls.sbom_components_total = Gauge(
                f"{prefix}_sbom_components_total",
                "Total components in SBOM",
                ["image"],
            )

            cls.scanner_health = Gauge(
                f"{prefix}_scanner_health",
                "Scanner health status (1=healthy, 0=unhealthy)",
                ["scanner"],
            )

            cls.coverage_score = Gauge(
                f"{prefix}_coverage_score",
                "Overall security scanning coverage score (0-100)",
            )

            # -- Histograms ------------------------------------------------

            cls.scan_duration_seconds = Histogram(
                f"{prefix}_scan_duration_seconds",
                "Scan duration in seconds",
                ["scanner", "target_type"],
                buckets=(
                    5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600,
                ),
            )

            cls.remediation_days = Histogram(
                f"{prefix}_remediation_days",
                "Days to remediation by severity",
                ["severity"],
                buckets=(
                    0.5, 1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365,
                ),
            )

            cls.risk_score_distribution = Histogram(
                f"{prefix}_risk_score",
                "Risk score distribution",
                ["severity"],
                buckets=(
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                ),
            )

            cls._available = True
            logger.info("Security Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# SecurityMetrics
# ---------------------------------------------------------------------------


class SecurityMetrics:
    """Manages Prometheus metrics for security scanning.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed. Thread-safe.

    Example:
        >>> m = SecurityMetrics()
        >>> m.record_vulnerability_found("critical", "open")
        >>> m.record_scan_completed("trivy", "image", "completed", 45.3)
    """

    def __init__(self, prefix: str = "gl_security") -> None:
        """Initialize security metrics.

        Args:
            prefix: Metric name prefix (documentation only; actual prefix
                is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Vulnerabilities
    # ------------------------------------------------------------------

    def record_vulnerability_found(
        self,
        severity: str,
        status: str,
    ) -> None:
        """Record a vulnerability discovery.

        Args:
            severity: Severity level (critical, high, medium, low, info).
            status: Initial status (open, resolved, etc.).
        """
        if not self._available:
            return

        _PrometheusHandles.vulnerabilities_total.labels(
            severity=severity, status=status
        ).inc()

    def set_open_vulnerabilities(
        self,
        severity: str,
        count: int,
    ) -> None:
        """Set the open vulnerability gauge for a severity.

        Args:
            severity: Severity level.
            count: Current count of open vulnerabilities.
        """
        if not self._available:
            return

        _PrometheusHandles.vulnerabilities_open.labels(
            severity=severity
        ).set(count)

    def record_sla_breach(self, severity: str) -> None:
        """Record an SLA breach.

        Args:
            severity: Severity of the breached vulnerability.
        """
        if not self._available:
            return

        _PrometheusHandles.sla_breach_total.labels(severity=severity).inc()

    def record_remediation_time(
        self,
        severity: str,
        days: float,
    ) -> None:
        """Record remediation time.

        Args:
            severity: Vulnerability severity.
            days: Days to remediation.
        """
        if not self._available:
            return

        _PrometheusHandles.remediation_days.labels(severity=severity).observe(days)

    def record_risk_score(
        self,
        severity: str,
        score: float,
    ) -> None:
        """Record a risk score.

        Args:
            severity: Vulnerability severity.
            score: Risk score (0-10).
        """
        if not self._available:
            return

        _PrometheusHandles.risk_score_distribution.labels(
            severity=severity
        ).observe(score)

    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

    def record_scan_completed(
        self,
        scanner: str,
        target_type: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a completed scan.

        Args:
            scanner: Scanner name (trivy, bandit, etc.).
            target_type: Target type (image, repository, etc.).
            status: Scan status (completed, failed).
            duration_seconds: Scan duration.
        """
        if not self._available:
            return

        _PrometheusHandles.scans_total.labels(
            scanner=scanner, target_type=target_type, status=status
        ).inc()

        _PrometheusHandles.scan_duration_seconds.labels(
            scanner=scanner, target_type=target_type
        ).observe(duration_seconds)

    def record_scan_error(
        self,
        scanner: str,
        error_type: str,
    ) -> None:
        """Record a scan error.

        Args:
            scanner: Scanner name.
            error_type: Error type (timeout, crash, etc.).
        """
        if not self._available:
            return

        _PrometheusHandles.scan_errors_total.labels(
            scanner=scanner, error_type=error_type
        ).inc()

    def set_scanner_health(
        self,
        scanner: str,
        healthy: bool,
    ) -> None:
        """Set scanner health status.

        Args:
            scanner: Scanner name.
            healthy: Whether scanner is healthy.
        """
        if not self._available:
            return

        _PrometheusHandles.scanner_health.labels(
            scanner=scanner
        ).set(1 if healthy else 0)

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def record_finding(
        self,
        scanner: str,
        severity: str,
        finding_type: str,
    ) -> None:
        """Record a scanner finding.

        Args:
            scanner: Scanner name.
            severity: Finding severity.
            finding_type: Finding type (sast, sca, secret, etc.).
        """
        if not self._available:
            return

        _PrometheusHandles.findings_total.labels(
            scanner=scanner, severity=severity, finding_type=finding_type
        ).inc()

    # ------------------------------------------------------------------
    # Container/SBOM
    # ------------------------------------------------------------------

    def record_image_signed(self, registry: str = "ecr") -> None:
        """Record a signed container image.

        Args:
            registry: Container registry name.
        """
        if not self._available:
            return

        _PrometheusHandles.images_signed_total.labels(registry=registry).inc()

    def set_sbom_components(
        self,
        image: str,
        count: int,
    ) -> None:
        """Set SBOM component count for an image.

        Args:
            image: Image identifier.
            count: Component count.
        """
        if not self._available:
            return

        _PrometheusHandles.sbom_components_total.labels(image=image).set(count)

    # ------------------------------------------------------------------
    # PII
    # ------------------------------------------------------------------

    def record_pii_finding(
        self,
        classification: str,
        pii_type: str,
    ) -> None:
        """Record a PII finding.

        Args:
            classification: Data classification (pii, phi, pci).
            pii_type: PII type (ssn, credit_card, etc.).
        """
        if not self._available:
            return

        _PrometheusHandles.pii_findings_total.labels(
            classification=classification, pii_type=pii_type
        ).inc()

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def set_coverage_score(self, score: float) -> None:
        """Set overall security coverage score.

        Args:
            score: Coverage score (0-100).
        """
        if not self._available:
            return

        _PrometheusHandles.coverage_score.set(score)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_security_metrics: Optional[SecurityMetrics] = None


def get_security_metrics() -> SecurityMetrics:
    """Get or create the global security metrics instance.

    Returns:
        The global SecurityMetrics instance.
    """
    global _global_security_metrics

    if _global_security_metrics is None:
        _global_security_metrics = SecurityMetrics()

    return _global_security_metrics


__all__ = [
    "SecurityMetrics",
    "get_security_metrics",
]
