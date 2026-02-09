# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-008: Supplier Questionnaire Processor

12 Prometheus metrics for supplier questionnaire service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_supplier_quest_templates_total (Counter, labels: framework, status)
    2.  gl_supplier_quest_distributions_total (Counter, labels: channel, status)
    3.  gl_supplier_quest_responses_total (Counter, labels: channel, status)
    4.  gl_supplier_quest_validations_total (Counter, labels: level, result)
    5.  gl_supplier_quest_scores_total (Counter, labels: framework, tier)
    6.  gl_supplier_quest_followups_total (Counter, labels: type, status)
    7.  gl_supplier_quest_response_rate (Gauge, labels: campaign_id)
    8.  gl_supplier_quest_processing_duration_seconds (Histogram, labels: operation)
    9.  gl_supplier_quest_active_campaigns (Gauge)
    10. gl_supplier_quest_pending_responses (Gauge)
    11. gl_supplier_quest_processing_errors_total (Counter, labels: engine, error_type)
    12. gl_supplier_quest_data_quality_score (Histogram, labels: framework)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; supplier questionnaire metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Questionnaire templates created/updated by framework and status
    supplier_quest_templates_total = Counter(
        "gl_supplier_quest_templates_total",
        "Total questionnaire templates managed",
        labelnames=["framework", "status"],
    )

    # 2. Questionnaire distributions by channel and status
    supplier_quest_distributions_total = Counter(
        "gl_supplier_quest_distributions_total",
        "Total questionnaire distributions sent",
        labelnames=["channel", "status"],
    )

    # 3. Questionnaire responses received by channel and status
    supplier_quest_responses_total = Counter(
        "gl_supplier_quest_responses_total",
        "Total questionnaire responses received",
        labelnames=["channel", "status"],
    )

    # 4. Validations performed by level and result
    supplier_quest_validations_total = Counter(
        "gl_supplier_quest_validations_total",
        "Total response validations performed",
        labelnames=["level", "result"],
    )

    # 5. Scores calculated by framework and performance tier
    supplier_quest_scores_total = Counter(
        "gl_supplier_quest_scores_total",
        "Total response scores calculated",
        labelnames=["framework", "tier"],
    )

    # 6. Follow-up actions by type and status
    supplier_quest_followups_total = Counter(
        "gl_supplier_quest_followups_total",
        "Total follow-up actions triggered",
        labelnames=["type", "status"],
    )

    # 7. Response rate gauge per campaign
    supplier_quest_response_rate = Gauge(
        "gl_supplier_quest_response_rate",
        "Current response rate percentage per campaign",
        labelnames=["campaign_id"],
    )

    # 8. Processing duration histogram by operation type
    supplier_quest_processing_duration_seconds = Histogram(
        "gl_supplier_quest_processing_duration_seconds",
        "Supplier questionnaire processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 9. Currently active campaigns gauge
    supplier_quest_active_campaigns = Gauge(
        "gl_supplier_quest_active_campaigns",
        "Number of currently active questionnaire campaigns",
    )

    # 10. Pending responses gauge
    supplier_quest_pending_responses = Gauge(
        "gl_supplier_quest_pending_responses",
        "Number of questionnaire responses awaiting submission",
    )

    # 11. Processing errors by engine and error type
    supplier_quest_processing_errors_total = Counter(
        "gl_supplier_quest_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["engine", "error_type"],
    )

    # 12. Data quality score histogram by framework
    supplier_quest_data_quality_score = Histogram(
        "gl_supplier_quest_data_quality_score",
        "Data quality scores for questionnaire responses",
        labelnames=["framework"],
        buckets=(
            10.0, 20.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 95.0, 100.0,
        ),
    )

else:
    # No-op placeholders
    supplier_quest_templates_total = None  # type: ignore[assignment]
    supplier_quest_distributions_total = None  # type: ignore[assignment]
    supplier_quest_responses_total = None  # type: ignore[assignment]
    supplier_quest_validations_total = None  # type: ignore[assignment]
    supplier_quest_scores_total = None  # type: ignore[assignment]
    supplier_quest_followups_total = None  # type: ignore[assignment]
    supplier_quest_response_rate = None  # type: ignore[assignment]
    supplier_quest_processing_duration_seconds = None  # type: ignore[assignment]
    supplier_quest_active_campaigns = None  # type: ignore[assignment]
    supplier_quest_pending_responses = None  # type: ignore[assignment]
    supplier_quest_processing_errors_total = None  # type: ignore[assignment]
    supplier_quest_data_quality_score = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_template(framework: str, status: str) -> None:
    """Record a questionnaire template operation.

    Args:
        framework: Questionnaire framework (cdp, ecovadis, custom, gri, etc.).
        status: Operation status (created, updated, cloned, deleted).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_templates_total.labels(
        framework=framework, status=status,
    ).inc()


def record_distribution(channel: str, status: str) -> None:
    """Record a questionnaire distribution event.

    Args:
        channel: Distribution channel (email, portal, api, bulk).
        status: Distribution status (sent, delivered, bounced, failed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_distributions_total.labels(
        channel=channel, status=status,
    ).inc()


def record_response(channel: str, status: str) -> None:
    """Record a questionnaire response event.

    Args:
        channel: Response channel (portal, api, email, upload).
        status: Response status (submitted, draft, finalized, rejected).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_responses_total.labels(
        channel=channel, status=status,
    ).inc()


def record_validation(level: str, result: str) -> None:
    """Record a validation event.

    Args:
        level: Validation level (completeness, consistency, evidence, cross_field).
        result: Validation result (pass, fail, warning).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_validations_total.labels(
        level=level, result=result,
    ).inc()


def record_score(framework: str, tier: str) -> None:
    """Record a scoring event.

    Args:
        framework: Questionnaire framework used for scoring.
        tier: Performance tier result (leader, advanced, developing, lagging).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_scores_total.labels(
        framework=framework, tier=tier,
    ).inc()


def record_followup(followup_type: str, status: str) -> None:
    """Record a follow-up action event.

    Args:
        followup_type: Follow-up type (reminder, escalation, deadline_extension).
        status: Follow-up status (scheduled, sent, acknowledged, expired).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_followups_total.labels(
        type=followup_type, status=status,
    ).inc()


def update_response_rate(campaign_id: str, rate: float) -> None:
    """Update the response rate for a campaign.

    Args:
        campaign_id: Campaign identifier.
        rate: Response rate percentage (0.0 - 100.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_response_rate.labels(
        campaign_id=campaign_id,
    ).set(rate)


def record_processing_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (create_template, distribute, validate, score, etc.).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def update_active_campaigns(delta: int) -> None:
    """Update the active campaigns gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        supplier_quest_active_campaigns.inc(delta)
    elif delta < 0:
        supplier_quest_active_campaigns.dec(abs(delta))


def update_pending_responses(delta: int) -> None:
    """Update the pending responses gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        supplier_quest_pending_responses.inc(delta)
    elif delta < 0:
        supplier_quest_pending_responses.dec(abs(delta))


def record_processing_error(engine: str, error_type: str) -> None:
    """Record a processing error event.

    Args:
        engine: Engine that produced the error (template, distribution, validation, etc.).
        error_type: Error classification (validation, timeout, data, integration, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_processing_errors_total.labels(
        engine=engine, error_type=error_type,
    ).inc()


def record_data_quality(framework: str, score: float) -> None:
    """Record a data quality score observation.

    Args:
        framework: Questionnaire framework.
        score: Data quality score (0.0 - 100.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    supplier_quest_data_quality_score.labels(
        framework=framework,
    ).observe(score)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "supplier_quest_templates_total",
    "supplier_quest_distributions_total",
    "supplier_quest_responses_total",
    "supplier_quest_validations_total",
    "supplier_quest_scores_total",
    "supplier_quest_followups_total",
    "supplier_quest_response_rate",
    "supplier_quest_processing_duration_seconds",
    "supplier_quest_active_campaigns",
    "supplier_quest_pending_responses",
    "supplier_quest_processing_errors_total",
    "supplier_quest_data_quality_score",
    # Helper functions
    "record_template",
    "record_distribution",
    "record_response",
    "record_validation",
    "record_score",
    "record_followup",
    "update_response_rate",
    "record_processing_duration",
    "update_active_campaigns",
    "update_pending_responses",
    "record_processing_error",
    "record_data_quality",
]
