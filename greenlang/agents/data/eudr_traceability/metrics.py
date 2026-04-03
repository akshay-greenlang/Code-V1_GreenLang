# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-004: EUDR Traceability Connector

12 Prometheus metrics for EUDR traceability service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_eudr_operations_total (Counter, labels: type, tenant_id)
    2.  gl_eudr_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_eudr_validation_errors_total (Counter, labels: severity, type)
    4.  gl_eudr_batch_jobs_total (Counter, labels: status)
    5.  gl_eudr_active_jobs (Gauge)
    6.  gl_eudr_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_eudr_plots_registered_total (Counter, labels: commodity, country)
    8.  gl_eudr_custody_transfers_total (Counter, labels: commodity, custody_model)
    9.  gl_eudr_dds_generated_total (Counter, labels: commodity, dds_type)
    10. gl_eudr_risk_assessments_total (Counter, labels: risk_level, target_type)
    11. gl_eudr_commodity_classifications_total (Counter, labels: commodity)
    12. gl_eudr_compliance_checks_total (Counter, labels: article, result)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
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
    "gl_eudr",
    "EUDR Traceability",
    duration_buckets=(
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

eudr_plots_registered_total = m.create_custom_counter(
    "plots_registered_total",
    "Total plots registered in EUDR traceability system",
    labelnames=["commodity", "country"],
)

eudr_custody_transfers_total = m.create_custom_counter(
    "custody_transfers_total",
    "Total chain of custody transfers recorded",
    labelnames=["commodity", "custody_model"],
)

eudr_dds_generated_total = m.create_custom_counter(
    "dds_generated_total",
    "Total due diligence statements generated",
    labelnames=["commodity", "dds_type"],
)

eudr_risk_assessments_total = m.create_custom_counter(
    "risk_assessments_total",
    "Total risk assessments performed",
    labelnames=["risk_level", "target_type"],
)

eudr_commodity_classifications_total = m.create_custom_counter(
    "commodity_classifications_total",
    "Total commodity classifications performed",
    labelnames=["commodity"],
)

eudr_compliance_checks_total = m.create_custom_counter(
    "compliance_checks_total",
    "Total compliance checks performed",
    labelnames=["article", "result"],
)

eudr_eu_submissions_total = m.create_custom_counter(
    "eu_submissions_total",
    "Total submissions to EU Information System",
    labelnames=["status"],
)

eudr_supplier_declarations_total = m.create_custom_counter(
    "supplier_declarations_total",
    "Total supplier declarations registered",
    labelnames=["commodity"],
)

eudr_processing_errors_total = m.create_custom_counter(
    "errors_total",
    "Total EUDR traceability processing errors",
    labelnames=["operation"],
)

eudr_active_plots = m.create_custom_gauge(
    "active_plots",
    "Number of currently active registered plots",
)

eudr_pending_submissions = m.create_custom_gauge(
    "pending_verifications",
    "Number of pending verifications or submissions",
)

# Backward-compat aliases for standard metrics expected by __init__.py
eudr_processing_duration_seconds = m.processing_duration


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_plot_registered(commodity: str, country: str) -> None:
    """Record a plot registration event.

    Args:
        commodity: EUDR commodity type.
        country: Country code (ISO 3166-1 alpha-2).
    """
    m.safe_inc(eudr_plots_registered_total, 1, commodity=commodity, country=country)


def record_custody_transfer(commodity: str, custody_model: str) -> None:
    """Record a chain of custody transfer event.

    Args:
        commodity: EUDR commodity type.
        custody_model: Chain of custody model used.
    """
    m.safe_inc(
        eudr_custody_transfers_total, 1,
        commodity=commodity, custody_model=custody_model,
    )


def record_dds_generated(commodity: str, dds_type: str = "operator") -> None:
    """Record a due diligence statement generation event.

    Args:
        commodity: EUDR commodity type.
        dds_type: DDS type (operator, trader, operator_trader).
    """
    m.safe_inc(eudr_dds_generated_total, 1, commodity=commodity, dds_type=dds_type)


def record_risk_assessment(risk_level: str, target_type: str) -> None:
    """Record a risk assessment event.

    Args:
        risk_level: Assessed risk level (low, standard, high).
        target_type: Type of entity assessed (plot, operator, shipment).
    """
    m.safe_inc(
        eudr_risk_assessments_total, 1,
        risk_level=risk_level, target_type=target_type,
    )


def record_commodity_classification(commodity: str) -> None:
    """Record a commodity classification event.

    Args:
        commodity: Classified EUDR commodity or 'unknown'.
    """
    m.safe_inc(eudr_commodity_classifications_total, 1, commodity=commodity)


def record_compliance_check(article: str, result: str) -> None:
    """Record a compliance check event.

    Args:
        article: EUDR article checked (e.g. 'article_3').
        result: Check result (compliant, non_compliant, partial, pending).
    """
    m.safe_inc(eudr_compliance_checks_total, 1, article=article, result=result)


def record_eu_submission(status: str) -> None:
    """Record an EU Information System submission event.

    Args:
        status: Submission status (submitted, accepted, rejected, failed).
    """
    m.safe_inc(eudr_eu_submissions_total, 1, status=status)


def record_supplier_declaration(commodity: str) -> None:
    """Record a supplier declaration event.

    Args:
        commodity: EUDR commodity declared.
    """
    m.safe_inc(eudr_supplier_declarations_total, 1, commodity=commodity)


def record_processing_error(operation: str) -> None:
    """Record a processing error event.

    Args:
        operation: Operation that failed.
    """
    m.safe_inc(eudr_processing_errors_total, 1, operation=operation)


def record_batch_operation(operation: str, duration_seconds: float) -> None:
    """Record a batch operation duration.

    Args:
        operation: Operation name.
        duration_seconds: Duration in seconds.
    """
    m.record_operation(duration_seconds, type=operation, tenant_id="batch")


def update_active_plots(delta: int) -> None:
    """Update the active plots gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not m.available:
        return
    if eudr_active_plots is not None:
        if delta > 0:
            eudr_active_plots.inc(delta)
        elif delta < 0:
            eudr_active_plots.dec(abs(delta))


def update_pending_submissions(count: int) -> None:
    """Set the current pending submissions/verifications count.

    Args:
        count: Current pending count.
    """
    m.safe_set(eudr_pending_submissions, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "eudr_plots_registered_total",
    "eudr_custody_transfers_total",
    "eudr_dds_generated_total",
    "eudr_risk_assessments_total",
    "eudr_commodity_classifications_total",
    "eudr_compliance_checks_total",
    "eudr_eu_submissions_total",
    "eudr_supplier_declarations_total",
    "eudr_processing_errors_total",
    "eudr_active_plots",
    "eudr_pending_submissions",
    # Helper functions
    "record_plot_registered",
    "record_custody_transfer",
    "record_dds_generated",
    "record_risk_assessment",
    "record_commodity_classification",
    "record_compliance_check",
    "record_eu_submission",
    "record_supplier_declaration",
    "record_processing_error",
    "record_batch_operation",
    "update_active_plots",
    "update_pending_submissions",
]
