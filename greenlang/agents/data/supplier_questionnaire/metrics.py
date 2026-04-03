# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-008: Supplier Questionnaire Processor

12 Prometheus metrics for supplier questionnaire service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_supplier_quest_operations_total (Counter, labels: type, tenant_id)
    2.  gl_supplier_quest_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_supplier_quest_validation_errors_total (Counter, labels: severity, type)
    4.  gl_supplier_quest_batch_jobs_total (Counter, labels: status)
    5.  gl_supplier_quest_active_jobs (Gauge)
    6.  gl_supplier_quest_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_supplier_quest_templates_total (Counter, labels: framework, status)
    8.  gl_supplier_quest_distributions_total (Counter, labels: channel, status)
    9.  gl_supplier_quest_responses_total (Counter, labels: channel, status)
    10. gl_supplier_quest_scores_total (Counter, labels: framework, tier)
    11. gl_supplier_quest_followups_total (Counter, labels: type, status)
    12. gl_supplier_quest_data_quality_score (Histogram, labels: framework)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_supplier_quest",
    "Supplier Questionnaire",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

supplier_quest_templates_total = m.create_custom_counter(
    "templates_total",
    "Total questionnaire templates managed",
    labelnames=["framework", "status"],
)

supplier_quest_distributions_total = m.create_custom_counter(
    "distributions_total",
    "Total questionnaire distributions sent",
    labelnames=["channel", "status"],
)

supplier_quest_responses_total = m.create_custom_counter(
    "responses_total",
    "Total questionnaire responses received",
    labelnames=["channel", "status"],
)

supplier_quest_validations_total = m.create_custom_counter(
    "validations_total",
    "Total response validations performed",
    labelnames=["level", "result"],
)

supplier_quest_scores_total = m.create_custom_counter(
    "scores_total",
    "Total response scores calculated",
    labelnames=["framework", "tier"],
)

supplier_quest_followups_total = m.create_custom_counter(
    "followups_total",
    "Total follow-up actions triggered",
    labelnames=["type", "status"],
)

supplier_quest_response_rate = m.create_custom_gauge(
    "response_rate",
    "Current response rate percentage per campaign",
    labelnames=["campaign_id"],
)

supplier_quest_active_campaigns = m.create_custom_gauge(
    "active_campaigns",
    "Number of currently active questionnaire campaigns",
)

supplier_quest_pending_responses = m.create_custom_gauge(
    "pending_responses",
    "Number of questionnaire responses awaiting submission",
)

supplier_quest_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["engine", "error_type"],
)

supplier_quest_data_quality_score = m.create_custom_histogram(
    "data_quality_score",
    "Data quality scores for questionnaire responses",
    buckets=(
        10.0, 20.0, 30.0, 40.0, 50.0,
        60.0, 70.0, 80.0, 90.0, 95.0, 100.0,
    ),
    labelnames=["framework"],
)

# Backward-compat alias for standard metrics expected by __init__.py
supplier_quest_processing_duration_seconds = m.processing_duration


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_template(framework: str, status: str) -> None:
    """Record a questionnaire template operation.

    Args:
        framework: Questionnaire framework (cdp, ecovadis, custom, gri, etc.).
        status: Operation status (created, updated, cloned, deleted).
    """
    m.safe_inc(supplier_quest_templates_total, 1, framework=framework, status=status)


def record_distribution(channel: str, status: str) -> None:
    """Record a questionnaire distribution event.

    Args:
        channel: Distribution channel (email, portal, api, bulk).
        status: Distribution status (sent, delivered, bounced, failed).
    """
    m.safe_inc(supplier_quest_distributions_total, 1, channel=channel, status=status)


def record_response(channel: str, status: str) -> None:
    """Record a questionnaire response event.

    Args:
        channel: Response channel (portal, api, email, upload).
        status: Response status (submitted, draft, finalized, rejected).
    """
    m.safe_inc(supplier_quest_responses_total, 1, channel=channel, status=status)


def record_validation(level: str, result: str) -> None:
    """Record a validation event.

    Args:
        level: Validation level (completeness, consistency, evidence, cross_field).
        result: Validation result (pass, fail, warning).
    """
    m.safe_inc(supplier_quest_validations_total, 1, level=level, result=result)


def record_score(framework: str, tier: str) -> None:
    """Record a scoring event.

    Args:
        framework: Questionnaire framework used for scoring.
        tier: Performance tier result (leader, advanced, developing, lagging).
    """
    m.safe_inc(supplier_quest_scores_total, 1, framework=framework, tier=tier)


def record_followup(followup_type: str, status: str) -> None:
    """Record a follow-up action event.

    Args:
        followup_type: Follow-up type (reminder, escalation, deadline_extension).
        status: Follow-up status (scheduled, sent, acknowledged, expired).
    """
    m.safe_inc(supplier_quest_followups_total, 1, type=followup_type, status=status)


def update_response_rate(campaign_id: str, rate: float) -> None:
    """Update the response rate for a campaign.

    Args:
        campaign_id: Campaign identifier.
        rate: Response rate percentage (0.0 - 100.0).
    """
    m.safe_set(supplier_quest_response_rate, rate, campaign_id=campaign_id)


def record_processing_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (create_template, distribute, validate, score, etc.).
        duration: Duration in seconds.
    """
    m.record_operation(duration, type=operation, tenant_id="default")


def update_active_campaigns(delta: int) -> None:
    """Update the active campaigns gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not m.available:
        return
    if supplier_quest_active_campaigns is not None:
        if delta > 0:
            supplier_quest_active_campaigns.inc(delta)
        elif delta < 0:
            supplier_quest_active_campaigns.dec(abs(delta))


def update_pending_responses(delta: int) -> None:
    """Update the pending responses gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not m.available:
        return
    if supplier_quest_pending_responses is not None:
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
    m.safe_inc(
        supplier_quest_processing_errors_total, 1,
        engine=engine, error_type=error_type,
    )


def record_data_quality(framework: str, score: float) -> None:
    """Record a data quality score observation.

    Args:
        framework: Questionnaire framework.
        score: Data quality score (0.0 - 100.0).
    """
    m.safe_observe(supplier_quest_data_quality_score, score, framework=framework)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "supplier_quest_templates_total",
    "supplier_quest_distributions_total",
    "supplier_quest_responses_total",
    "supplier_quest_validations_total",
    "supplier_quest_scores_total",
    "supplier_quest_followups_total",
    "supplier_quest_response_rate",
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
