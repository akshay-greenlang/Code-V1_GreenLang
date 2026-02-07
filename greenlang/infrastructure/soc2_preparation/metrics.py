# -*- coding: utf-8 -*-
"""
SOC 2 Prometheus Metrics - SEC-009

Defines all Prometheus metrics for the GreenLang SOC 2 Type II preparation
platform and provides helper functions for recording assessment events,
evidence collection, auditor requests, and compliance status.

Metrics are organized by subsystem:
    - Assessment: Assessment runs, scores, completion times
    - Evidence: Evidence uploads, collection, verification
    - Requests: Auditor request SLA tracking
    - Gaps: Gap counts, remediation progress
    - Compliance: Overall readiness metrics

All metrics use the ``soc2_`` prefix to avoid collisions with other
GreenLang Prometheus metrics.

Example:
    >>> from greenlang.infrastructure.soc2_preparation.metrics import (
    ...     record_assessment_completed,
    ...     record_evidence_uploaded,
    ...     update_readiness_score,
    ... )
    >>> record_assessment_completed(
    ...     assessment_id="uuid",
    ...     score=78.5,
    ...     criteria_count=33,
    ...     duration_seconds=45.2
    ... )
    >>> record_evidence_uploaded(
    ...     evidence_type="document",
    ...     criterion_id="CC6.1",
    ...     is_automated=False
    ... )

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]
    Summary = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:

    # -- Assessment Metrics --------------------------------------------------

    soc2_assessments_total = Counter(
        "soc2_assessments_total",
        "Total number of SOC 2 self-assessments run",
        labelnames=["status", "categories"],
    )
    """Counter: Total assessments. Labels: status (completed/failed), categories."""

    soc2_assessment_score = Gauge(
        "soc2_assessment_score",
        "Latest SOC 2 assessment score (0-100)",
        labelnames=["category"],
    )
    """Gauge: Latest assessment score by category."""

    soc2_assessment_duration_seconds = Histogram(
        "soc2_assessment_duration_seconds",
        "Duration of SOC 2 assessments in seconds",
        labelnames=["categories"],
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    )
    """Histogram: Assessment duration. Labels: categories."""

    soc2_criteria_assessed_total = Counter(
        "soc2_criteria_assessed_total",
        "Total number of criteria assessed",
        labelnames=["category", "score_level"],
    )
    """Counter: Criteria assessed. Labels: category, score_level."""

    # -- Evidence Metrics ----------------------------------------------------

    soc2_evidence_uploaded_total = Counter(
        "soc2_evidence_uploaded_total",
        "Total number of evidence items uploaded",
        labelnames=["evidence_type", "collection_method"],
    )
    """Counter: Evidence uploads. Labels: evidence_type, collection_method."""

    soc2_evidence_verified_total = Counter(
        "soc2_evidence_verified_total",
        "Total number of evidence items verified",
        labelnames=["evidence_type"],
    )
    """Counter: Evidence verifications. Labels: evidence_type."""

    soc2_evidence_storage_bytes = Gauge(
        "soc2_evidence_storage_bytes",
        "Total bytes of evidence stored",
        labelnames=["bucket"],
    )
    """Gauge: Evidence storage size. Labels: bucket."""

    # -- Auditor Request Metrics ---------------------------------------------

    soc2_requests_total = Counter(
        "soc2_requests_total",
        "Total number of auditor requests received",
        labelnames=["priority", "status"],
    )
    """Counter: Auditor requests. Labels: priority, status."""

    soc2_request_sla_seconds = Histogram(
        "soc2_request_sla_seconds",
        "Time to fulfill auditor requests in seconds",
        labelnames=["priority"],
        buckets=(
            3600,      # 1 hour
            14400,     # 4 hours
            86400,     # 1 day
            172800,    # 2 days
            259200,    # 3 days
            604800,    # 1 week
        ),
    )
    """Histogram: Request fulfillment time. Labels: priority."""

    soc2_requests_overdue = Gauge(
        "soc2_requests_overdue",
        "Number of auditor requests past SLA",
        labelnames=["priority"],
    )
    """Gauge: Overdue requests by priority."""

    # -- Gap Metrics ---------------------------------------------------------

    soc2_gaps_total = Gauge(
        "soc2_gaps_total",
        "Current number of compliance gaps",
        labelnames=["risk_level", "category"],
    )
    """Gauge: Gap counts. Labels: risk_level, category."""

    soc2_gap_remediation_hours = Gauge(
        "soc2_gap_remediation_hours",
        "Estimated hours to remediate all gaps",
        labelnames=["risk_level"],
    )
    """Gauge: Remediation effort by risk level."""

    soc2_remediations_total = Counter(
        "soc2_remediations_total",
        "Total number of remediations completed",
        labelnames=["status"],
    )
    """Counter: Remediations. Labels: status."""

    # -- Compliance Metrics --------------------------------------------------

    soc2_readiness_percentage = Gauge(
        "soc2_readiness_percentage",
        "SOC 2 audit readiness percentage (0-100)",
        labelnames=["category"],
    )
    """Gauge: Readiness percentage by category."""

    soc2_compliant_criteria_total = Gauge(
        "soc2_compliant_criteria_total",
        "Number of criteria at COMPLIANT level",
        labelnames=["category"],
    )
    """Gauge: Compliant criteria count by category."""

    soc2_control_status = Gauge(
        "soc2_control_status",
        "Control implementation status distribution",
        labelnames=["status"],
    )
    """Gauge: Control status distribution. Labels: status."""

    # -- Finding Metrics -----------------------------------------------------

    soc2_findings_total = Gauge(
        "soc2_findings_total",
        "Total number of audit findings",
        labelnames=["classification", "status"],
    )
    """Gauge: Findings count. Labels: classification, status."""

    # -- Attestation Metrics -------------------------------------------------

    soc2_attestations_pending = Gauge(
        "soc2_attestations_pending",
        "Number of attestations pending signatures",
    )
    """Gauge: Pending attestations."""

    soc2_attestation_signatures = Counter(
        "soc2_attestation_signatures",
        "Total attestation signatures collected",
        labelnames=["attestation_type"],
    )
    """Counter: Attestation signatures. Labels: attestation_type."""

    # -- Audit Project Metrics -----------------------------------------------

    soc2_project_milestone_status = Gauge(
        "soc2_project_milestone_status",
        "Audit project milestone status (1=completed, 0=pending)",
        labelnames=["project_id", "milestone"],
    )
    """Gauge: Milestone status. Labels: project_id, milestone."""

    soc2_project_completion_percentage = Gauge(
        "soc2_project_completion_percentage",
        "Audit project completion percentage",
        labelnames=["project_id"],
    )
    """Gauge: Project completion. Labels: project_id."""

else:
    # Provide no-op stubs when prometheus_client is not installed
    logger.info(
        "prometheus_client not available; SOC 2 metrics will be no-ops"
    )

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is not installed."""

        def labels(self, *args, **kwargs):  # noqa: ANN
            """Return self for chaining."""
            return self

        def inc(self, amount=1):  # noqa: ANN
            """No-op increment."""

        def dec(self, amount=1):  # noqa: ANN
            """No-op decrement."""

        def set(self, value):  # noqa: ANN
            """No-op set."""

        def observe(self, amount):  # noqa: ANN
            """No-op observe."""

    soc2_assessments_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_assessment_score = _NoOpMetric()  # type: ignore[assignment]
    soc2_assessment_duration_seconds = _NoOpMetric()  # type: ignore[assignment]
    soc2_criteria_assessed_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_evidence_uploaded_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_evidence_verified_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_evidence_storage_bytes = _NoOpMetric()  # type: ignore[assignment]
    soc2_requests_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_request_sla_seconds = _NoOpMetric()  # type: ignore[assignment]
    soc2_requests_overdue = _NoOpMetric()  # type: ignore[assignment]
    soc2_gaps_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_gap_remediation_hours = _NoOpMetric()  # type: ignore[assignment]
    soc2_remediations_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_readiness_percentage = _NoOpMetric()  # type: ignore[assignment]
    soc2_compliant_criteria_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_control_status = _NoOpMetric()  # type: ignore[assignment]
    soc2_findings_total = _NoOpMetric()  # type: ignore[assignment]
    soc2_attestations_pending = _NoOpMetric()  # type: ignore[assignment]
    soc2_attestation_signatures = _NoOpMetric()  # type: ignore[assignment]
    soc2_project_milestone_status = _NoOpMetric()  # type: ignore[assignment]
    soc2_project_completion_percentage = _NoOpMetric()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_assessment_completed(
    assessment_id: str,
    score: float,
    criteria_count: int,
    duration_seconds: float,
    categories: str = "security",
    status: str = "completed",
) -> None:
    """Record a completed assessment.

    Args:
        assessment_id: UUID of the assessment.
        score: Overall assessment score (0-100).
        criteria_count: Number of criteria assessed.
        duration_seconds: Time taken to run assessment.
        categories: Comma-separated list of categories assessed.
        status: Assessment status (completed/failed).
    """
    try:
        soc2_assessments_total.labels(
            status=status,
            categories=categories,
        ).inc()

        soc2_assessment_duration_seconds.labels(
            categories=categories,
        ).observe(duration_seconds)

        soc2_assessment_score.labels(category="overall").set(score)

        logger.debug(
            "Recorded assessment: id=%s, score=%.2f, criteria=%d, duration=%.2fs",
            assessment_id,
            score,
            criteria_count,
            duration_seconds,
        )
    except Exception as exc:
        logger.debug("Failed to record assessment metric: %s", exc)


def record_evidence_uploaded(
    evidence_type: str,
    criterion_id: str,
    is_automated: bool = False,
    file_size_bytes: int = 0,
) -> None:
    """Record an evidence upload event.

    Args:
        evidence_type: Type of evidence (document, screenshot, etc.).
        criterion_id: SOC 2 criterion the evidence supports.
        is_automated: Whether evidence was collected automatically.
        file_size_bytes: Size of the evidence file.
    """
    try:
        collection_method = "automated" if is_automated else "manual"
        soc2_evidence_uploaded_total.labels(
            evidence_type=evidence_type,
            collection_method=collection_method,
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record evidence upload metric: %s", exc)


def record_evidence_verified(
    evidence_type: str,
    verified_by: str,
) -> None:
    """Record an evidence verification event.

    Args:
        evidence_type: Type of evidence verified.
        verified_by: UUID of verifier.
    """
    try:
        soc2_evidence_verified_total.labels(
            evidence_type=evidence_type,
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record evidence verification metric: %s", exc)


def record_auditor_request(
    request_id: str,
    priority: str,
    status: str,
    fulfillment_seconds: Optional[float] = None,
) -> None:
    """Record an auditor request event.

    Args:
        request_id: UUID of the request.
        priority: Request priority (critical/high/normal/low).
        status: Request status (open/fulfilled/overdue).
        fulfillment_seconds: Time to fulfill, if fulfilled.
    """
    try:
        soc2_requests_total.labels(
            priority=priority,
            status=status,
        ).inc()

        if fulfillment_seconds is not None:
            soc2_request_sla_seconds.labels(
                priority=priority,
            ).observe(fulfillment_seconds)
    except Exception as exc:
        logger.debug("Failed to record auditor request metric: %s", exc)


def update_overdue_requests(
    critical: int = 0,
    high: int = 0,
    normal: int = 0,
    low: int = 0,
) -> None:
    """Update overdue request gauges.

    Args:
        critical: Count of overdue critical requests.
        high: Count of overdue high requests.
        normal: Count of overdue normal requests.
        low: Count of overdue low requests.
    """
    try:
        soc2_requests_overdue.labels(priority="critical").set(critical)
        soc2_requests_overdue.labels(priority="high").set(high)
        soc2_requests_overdue.labels(priority="normal").set(normal)
        soc2_requests_overdue.labels(priority="low").set(low)
    except Exception as exc:
        logger.debug("Failed to update overdue requests metric: %s", exc)


def update_gaps(
    critical_count: int,
    high_count: int,
    medium_count: int,
    low_count: int,
    category: str = "all",
) -> None:
    """Update gap count gauges.

    Args:
        critical_count: Number of critical gaps.
        high_count: Number of high-risk gaps.
        medium_count: Number of medium-risk gaps.
        low_count: Number of low-risk gaps.
        category: TSC category for the gaps.
    """
    try:
        soc2_gaps_total.labels(risk_level="critical", category=category).set(critical_count)
        soc2_gaps_total.labels(risk_level="high", category=category).set(high_count)
        soc2_gaps_total.labels(risk_level="medium", category=category).set(medium_count)
        soc2_gaps_total.labels(risk_level="low", category=category).set(low_count)
    except Exception as exc:
        logger.debug("Failed to update gaps metric: %s", exc)


def update_remediation_effort(
    critical_hours: int,
    high_hours: int,
    medium_hours: int,
    low_hours: int,
) -> None:
    """Update remediation effort gauges.

    Args:
        critical_hours: Hours to remediate critical gaps.
        high_hours: Hours to remediate high-risk gaps.
        medium_hours: Hours to remediate medium-risk gaps.
        low_hours: Hours to remediate low-risk gaps.
    """
    try:
        soc2_gap_remediation_hours.labels(risk_level="critical").set(critical_hours)
        soc2_gap_remediation_hours.labels(risk_level="high").set(high_hours)
        soc2_gap_remediation_hours.labels(risk_level="medium").set(medium_hours)
        soc2_gap_remediation_hours.labels(risk_level="low").set(low_hours)
    except Exception as exc:
        logger.debug("Failed to update remediation effort metric: %s", exc)


def update_readiness_score(
    overall: float,
    security: Optional[float] = None,
    availability: Optional[float] = None,
    confidentiality: Optional[float] = None,
    processing_integrity: Optional[float] = None,
    privacy: Optional[float] = None,
) -> None:
    """Update readiness percentage gauges.

    Args:
        overall: Overall readiness percentage.
        security: Security category readiness.
        availability: Availability category readiness.
        confidentiality: Confidentiality category readiness.
        processing_integrity: Processing integrity readiness.
        privacy: Privacy category readiness.
    """
    try:
        soc2_readiness_percentage.labels(category="overall").set(overall)

        if security is not None:
            soc2_readiness_percentage.labels(category="security").set(security)
        if availability is not None:
            soc2_readiness_percentage.labels(category="availability").set(availability)
        if confidentiality is not None:
            soc2_readiness_percentage.labels(category="confidentiality").set(confidentiality)
        if processing_integrity is not None:
            soc2_readiness_percentage.labels(category="processing_integrity").set(processing_integrity)
        if privacy is not None:
            soc2_readiness_percentage.labels(category="privacy").set(privacy)
    except Exception as exc:
        logger.debug("Failed to update readiness score metric: %s", exc)


def update_compliant_criteria(
    total: int,
    security: int = 0,
    availability: int = 0,
    confidentiality: int = 0,
    processing_integrity: int = 0,
    privacy: int = 0,
) -> None:
    """Update compliant criteria count gauges.

    Args:
        total: Total compliant criteria.
        security: Compliant security criteria.
        availability: Compliant availability criteria.
        confidentiality: Compliant confidentiality criteria.
        processing_integrity: Compliant processing integrity criteria.
        privacy: Compliant privacy criteria.
    """
    try:
        soc2_compliant_criteria_total.labels(category="total").set(total)
        soc2_compliant_criteria_total.labels(category="security").set(security)
        soc2_compliant_criteria_total.labels(category="availability").set(availability)
        soc2_compliant_criteria_total.labels(category="confidentiality").set(confidentiality)
        soc2_compliant_criteria_total.labels(category="processing_integrity").set(processing_integrity)
        soc2_compliant_criteria_total.labels(category="privacy").set(privacy)
    except Exception as exc:
        logger.debug("Failed to update compliant criteria metric: %s", exc)


def record_finding(
    classification: str,
    status: str,
    count: int = 1,
) -> None:
    """Record audit finding metrics.

    Args:
        classification: Finding classification (observation/exception/deficiency).
        status: Finding status (open/remediated/closed).
        count: Number of findings.
    """
    try:
        soc2_findings_total.labels(
            classification=classification,
            status=status,
        ).set(count)
    except Exception as exc:
        logger.debug("Failed to record finding metric: %s", exc)


def record_attestation_signature(
    attestation_type: str,
) -> None:
    """Record an attestation signature event.

    Args:
        attestation_type: Type of attestation signed.
    """
    try:
        soc2_attestation_signatures.labels(
            attestation_type=attestation_type,
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record attestation signature metric: %s", exc)


def update_project_completion(
    project_id: str,
    completion_percentage: float,
) -> None:
    """Update audit project completion gauge.

    Args:
        project_id: UUID of the audit project.
        completion_percentage: Project completion percentage.
    """
    try:
        soc2_project_completion_percentage.labels(
            project_id=project_id,
        ).set(completion_percentage)
    except Exception as exc:
        logger.debug("Failed to update project completion metric: %s", exc)


# ---------------------------------------------------------------------------
# SOC2Metrics Class (For Convenient Import)
# ---------------------------------------------------------------------------


class SOC2Metrics:
    """Facade class for SOC 2 metrics.

    Provides a convenient interface for recording all SOC 2 metrics
    through a single object. Useful for dependency injection.

    Example:
        >>> metrics = SOC2Metrics()
        >>> metrics.record_assessment_completed(...)
        >>> metrics.update_readiness_score(78.5)
    """

    @staticmethod
    def record_assessment_completed(
        assessment_id: str,
        score: float,
        criteria_count: int,
        duration_seconds: float,
        categories: str = "security",
        status: str = "completed",
    ) -> None:
        """Record a completed assessment."""
        record_assessment_completed(
            assessment_id=assessment_id,
            score=score,
            criteria_count=criteria_count,
            duration_seconds=duration_seconds,
            categories=categories,
            status=status,
        )

    @staticmethod
    def record_evidence_uploaded(
        evidence_type: str,
        criterion_id: str,
        is_automated: bool = False,
        file_size_bytes: int = 0,
    ) -> None:
        """Record an evidence upload event."""
        record_evidence_uploaded(
            evidence_type=evidence_type,
            criterion_id=criterion_id,
            is_automated=is_automated,
            file_size_bytes=file_size_bytes,
        )

    @staticmethod
    def update_readiness_score(
        overall: float,
        **category_scores: float,
    ) -> None:
        """Update readiness percentage gauges."""
        update_readiness_score(overall=overall, **category_scores)

    @staticmethod
    def update_gaps(
        critical_count: int,
        high_count: int,
        medium_count: int,
        low_count: int,
        category: str = "all",
    ) -> None:
        """Update gap count gauges."""
        update_gaps(
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            category=category,
        )


__all__ = [
    # Metrics
    "soc2_assessments_total",
    "soc2_assessment_score",
    "soc2_assessment_duration_seconds",
    "soc2_criteria_assessed_total",
    "soc2_evidence_uploaded_total",
    "soc2_evidence_verified_total",
    "soc2_evidence_storage_bytes",
    "soc2_requests_total",
    "soc2_request_sla_seconds",
    "soc2_requests_overdue",
    "soc2_gaps_total",
    "soc2_gap_remediation_hours",
    "soc2_remediations_total",
    "soc2_readiness_percentage",
    "soc2_compliant_criteria_total",
    "soc2_control_status",
    "soc2_findings_total",
    "soc2_attestations_pending",
    "soc2_attestation_signatures",
    "soc2_project_milestone_status",
    "soc2_project_completion_percentage",
    # Helper Functions
    "record_assessment_completed",
    "record_evidence_uploaded",
    "record_evidence_verified",
    "record_auditor_request",
    "update_overdue_requests",
    "update_gaps",
    "update_remediation_effort",
    "update_readiness_score",
    "update_compliant_criteria",
    "record_finding",
    "record_attestation_signature",
    "update_project_completion",
    # Class
    "SOC2Metrics",
]
