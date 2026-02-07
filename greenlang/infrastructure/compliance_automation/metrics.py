# -*- coding: utf-8 -*-
"""
Compliance Automation Metrics - SEC-010 Phase 5

Prometheus metrics for monitoring the GreenLang multi-compliance automation
system. Tracks compliance scores, DSAR processing, consent management, and
framework-specific metrics.

Metrics Exposed:
- gl_secops_compliance_score: Compliance score by framework (0-100)
- gl_secops_compliance_controls_total: Control counts by framework and status
- gl_secops_dsar_pending: Number of pending DSAR requests
- gl_secops_dsar_completed_total: Completed DSAR requests by type
- gl_secops_dsar_processing_seconds: DSAR processing time
- gl_secops_dsar_sla_compliance: DSAR SLA compliance percentage
- gl_secops_consent_grants_total: Consent grants by purpose
- gl_secops_consent_revocations_total: Consent revocations by purpose
- gl_secops_retention_deleted_total: Records deleted by retention enforcement
- gl_secops_pii_records_discovered: PII records discovered during scans

Classes:
    - ComplianceMetrics: Prometheus metrics collector.
    - get_compliance_metrics: Factory function for metrics singleton.

Example:
    >>> from greenlang.infrastructure.compliance_automation.metrics import (
    ...     get_compliance_metrics,
    ... )
    >>> metrics = get_compliance_metrics()
    >>> metrics.set_compliance_score("iso27001", 95.5)
    >>> metrics.record_dsar_completion("access", 2500.0)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics Labels
# ---------------------------------------------------------------------------

# Framework labels
FRAMEWORK_LABELS = ["framework"]

# DSAR labels
DSAR_LABELS = ["request_type", "regulation"]

# Control labels
CONTROL_LABELS = ["framework", "status"]

# Consent labels
CONSENT_LABELS = ["purpose"]

# Retention labels
RETENTION_LABELS = ["data_category", "action"]

# PII discovery labels
PII_LABELS = ["source_system", "data_category"]


# ---------------------------------------------------------------------------
# Compliance Metrics Class
# ---------------------------------------------------------------------------


class ComplianceMetrics:
    """Prometheus metrics for compliance automation.

    Collects and exposes metrics about compliance status, DSAR processing,
    consent management, and data retention enforcement.

    Attributes:
        compliance_score: Gauge for compliance score by framework.
        controls_total: Gauge for control counts by framework and status.
        dsar_pending: Gauge for pending DSAR count.
        dsar_completed: Counter for completed DSARs.
        dsar_processing_time: Histogram for DSAR processing duration.
        dsar_sla_compliance: Gauge for DSAR SLA compliance percentage.
        consent_grants: Counter for consent grants.
        consent_revocations: Counter for consent revocations.
        retention_deleted: Counter for retention deletions.
        pii_discovered: Gauge for PII records discovered.

    Example:
        >>> metrics = ComplianceMetrics()
        >>> metrics.set_compliance_score("iso27001", 95.5)
        >>> metrics.increment_dsar_completed("access", "gdpr")
    """

    def __init__(self, namespace: str = "gl_secops") -> None:
        """Initialize compliance metrics.

        Args:
            namespace: Metric name prefix (default: gl_secops).
        """
        self.namespace = namespace
        self._initialized = False

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not available. Metrics will be no-ops."
            )
            self._create_noop_metrics()
            return

        self._create_prometheus_metrics()
        self._initialized = True

        logger.info("Initialized ComplianceMetrics with namespace: %s", namespace)

    def _create_prometheus_metrics(self) -> None:
        """Create Prometheus metric objects."""
        # Compliance score by framework (0-100)
        self.compliance_score = Gauge(
            f"{self.namespace}_compliance_score",
            "Compliance score by framework (0-100)",
            FRAMEWORK_LABELS,
        )

        # Control counts by framework and status
        self.controls_total = Gauge(
            f"{self.namespace}_compliance_controls_total",
            "Total controls by framework and status",
            CONTROL_LABELS,
        )

        # Pending DSAR count
        self.dsar_pending = Gauge(
            f"{self.namespace}_dsar_pending",
            "Number of pending DSAR requests",
            DSAR_LABELS,
        )

        # Completed DSAR counter
        self.dsar_completed = Counter(
            f"{self.namespace}_dsar_completed_total",
            "Total completed DSAR requests",
            DSAR_LABELS,
        )

        # DSAR processing time histogram
        self.dsar_processing_time = Histogram(
            f"{self.namespace}_dsar_processing_seconds",
            "DSAR processing time in seconds",
            DSAR_LABELS,
            buckets=[60, 300, 900, 3600, 86400, 604800, 2592000],  # 1m to 30d
        )

        # DSAR SLA compliance percentage
        self.dsar_sla_compliance = Gauge(
            f"{self.namespace}_dsar_sla_compliance",
            "DSAR SLA compliance percentage (0-100)",
        )

        # Consent grants counter
        self.consent_grants = Counter(
            f"{self.namespace}_consent_grants_total",
            "Total consent grants by purpose",
            CONSENT_LABELS,
        )

        # Consent revocations counter
        self.consent_revocations = Counter(
            f"{self.namespace}_consent_revocations_total",
            "Total consent revocations by purpose",
            CONSENT_LABELS,
        )

        # Retention deleted records counter
        self.retention_deleted = Counter(
            f"{self.namespace}_retention_deleted_total",
            "Total records deleted by retention enforcement",
            RETENTION_LABELS,
        )

        # Retention anonymized records counter
        self.retention_anonymized = Counter(
            f"{self.namespace}_retention_anonymized_total",
            "Total records anonymized by retention enforcement",
            RETENTION_LABELS,
        )

        # PII records discovered gauge
        self.pii_discovered = Gauge(
            f"{self.namespace}_pii_records_discovered",
            "PII records discovered during scans",
            PII_LABELS,
        )

        # Assessment duration histogram
        self.assessment_duration = Histogram(
            f"{self.namespace}_assessment_duration_seconds",
            "Compliance assessment duration in seconds",
            FRAMEWORK_LABELS,
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        # Evidence collection counter
        self.evidence_collected = Counter(
            f"{self.namespace}_evidence_collected_total",
            "Total evidence items collected",
            FRAMEWORK_LABELS,
        )

        # Compliance info
        self.compliance_info = Info(
            f"{self.namespace}_compliance",
            "Compliance automation system information",
        )
        self.compliance_info.info({
            "version": "1.0.0",
            "frameworks_supported": "iso27001,gdpr,pci_dss,ccpa,lgpd",
        })

    def _create_noop_metrics(self) -> None:
        """Create no-op metrics when Prometheus is not available."""
        class NoOpMetric:
            def labels(self, *args, **kwargs):
                return self
            def set(self, *args, **kwargs):
                pass
            def inc(self, *args, **kwargs):
                pass
            def observe(self, *args, **kwargs):
                pass
            def info(self, *args, **kwargs):
                pass

        self.compliance_score = NoOpMetric()
        self.controls_total = NoOpMetric()
        self.dsar_pending = NoOpMetric()
        self.dsar_completed = NoOpMetric()
        self.dsar_processing_time = NoOpMetric()
        self.dsar_sla_compliance = NoOpMetric()
        self.consent_grants = NoOpMetric()
        self.consent_revocations = NoOpMetric()
        self.retention_deleted = NoOpMetric()
        self.retention_anonymized = NoOpMetric()
        self.pii_discovered = NoOpMetric()
        self.assessment_duration = NoOpMetric()
        self.evidence_collected = NoOpMetric()
        self.compliance_info = NoOpMetric()

    # -------------------------------------------------------------------------
    # Compliance Score Methods
    # -------------------------------------------------------------------------

    def set_compliance_score(self, framework: str, score: float) -> None:
        """Set the compliance score for a framework.

        Args:
            framework: Framework name (iso27001, gdpr, pci_dss, etc.).
            score: Compliance score (0-100).
        """
        self.compliance_score.labels(framework=framework).set(score)
        logger.debug("Set compliance score: %s = %.2f", framework, score)

    def set_control_count(
        self,
        framework: str,
        status: str,
        count: int,
    ) -> None:
        """Set the control count for a framework and status.

        Args:
            framework: Framework name.
            status: Control status (compliant, non_compliant, not_applicable).
            count: Number of controls.
        """
        self.controls_total.labels(framework=framework, status=status).set(count)

    # -------------------------------------------------------------------------
    # DSAR Methods
    # -------------------------------------------------------------------------

    def set_dsar_pending(
        self,
        request_type: str,
        regulation: str,
        count: int,
    ) -> None:
        """Set the count of pending DSAR requests.

        Args:
            request_type: DSAR type (access, erasure, etc.).
            regulation: Regulation (gdpr, ccpa, lgpd).
            count: Number of pending requests.
        """
        self.dsar_pending.labels(
            request_type=request_type,
            regulation=regulation,
        ).set(count)

    def increment_dsar_completed(
        self,
        request_type: str,
        regulation: str,
    ) -> None:
        """Increment the completed DSAR counter.

        Args:
            request_type: DSAR type.
            regulation: Regulation.
        """
        self.dsar_completed.labels(
            request_type=request_type,
            regulation=regulation,
        ).inc()

    def record_dsar_processing_time(
        self,
        request_type: str,
        regulation: str,
        duration_seconds: float,
    ) -> None:
        """Record DSAR processing duration.

        Args:
            request_type: DSAR type.
            regulation: Regulation.
            duration_seconds: Processing time in seconds.
        """
        self.dsar_processing_time.labels(
            request_type=request_type,
            regulation=regulation,
        ).observe(duration_seconds)

    def set_dsar_sla_compliance(self, percentage: float) -> None:
        """Set the DSAR SLA compliance percentage.

        Args:
            percentage: SLA compliance percentage (0-100).
        """
        self.dsar_sla_compliance.set(percentage)

    # -------------------------------------------------------------------------
    # Consent Methods
    # -------------------------------------------------------------------------

    def increment_consent_grant(self, purpose: str) -> None:
        """Increment the consent grant counter.

        Args:
            purpose: Consent purpose.
        """
        self.consent_grants.labels(purpose=purpose).inc()

    def increment_consent_revocation(self, purpose: str) -> None:
        """Increment the consent revocation counter.

        Args:
            purpose: Consent purpose.
        """
        self.consent_revocations.labels(purpose=purpose).inc()

    # -------------------------------------------------------------------------
    # Retention Methods
    # -------------------------------------------------------------------------

    def increment_retention_deleted(
        self,
        data_category: str,
        count: int = 1,
    ) -> None:
        """Increment the retention deleted records counter.

        Args:
            data_category: Data category deleted.
            count: Number of records deleted.
        """
        self.retention_deleted.labels(
            data_category=data_category,
            action="delete",
        ).inc(count)

    def increment_retention_anonymized(
        self,
        data_category: str,
        count: int = 1,
    ) -> None:
        """Increment the retention anonymized records counter.

        Args:
            data_category: Data category anonymized.
            count: Number of records anonymized.
        """
        self.retention_anonymized.labels(
            data_category=data_category,
            action="anonymize",
        ).inc(count)

    # -------------------------------------------------------------------------
    # PII Discovery Methods
    # -------------------------------------------------------------------------

    def set_pii_discovered(
        self,
        source_system: str,
        data_category: str,
        count: int,
    ) -> None:
        """Set the count of PII records discovered.

        Args:
            source_system: System where PII was found.
            data_category: Category of PII.
            count: Number of records.
        """
        self.pii_discovered.labels(
            source_system=source_system,
            data_category=data_category,
        ).set(count)

    # -------------------------------------------------------------------------
    # Assessment Methods
    # -------------------------------------------------------------------------

    def record_assessment_duration(
        self,
        framework: str,
        duration_seconds: float,
    ) -> None:
        """Record compliance assessment duration.

        Args:
            framework: Framework assessed.
            duration_seconds: Assessment duration in seconds.
        """
        self.assessment_duration.labels(framework=framework).observe(duration_seconds)

    def increment_evidence_collected(
        self,
        framework: str,
        count: int = 1,
    ) -> None:
        """Increment the evidence collected counter.

        Args:
            framework: Framework for which evidence was collected.
            count: Number of evidence items collected.
        """
        self.evidence_collected.labels(framework=framework).inc(count)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

_metrics_instance: Optional[ComplianceMetrics] = None


def get_compliance_metrics() -> ComplianceMetrics:
    """Get the compliance metrics singleton.

    Creates a ComplianceMetrics instance on first call and returns the
    cached instance on subsequent calls.

    Returns:
        The ComplianceMetrics singleton.

    Example:
        >>> metrics = get_compliance_metrics()
        >>> metrics.set_compliance_score("iso27001", 95.5)
    """
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = ComplianceMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """Reset the metrics singleton. Used for testing."""
    global _metrics_instance
    _metrics_instance = None


__all__ = [
    "ComplianceMetrics",
    "get_compliance_metrics",
    "reset_metrics",
]
