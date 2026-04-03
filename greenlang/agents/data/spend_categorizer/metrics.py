# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-009: Spend Data Categorizer

12 Prometheus metrics for spend data categorizer service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_spend_cat_operations_total (Counter, labels: type, tenant_id)
    2.  gl_spend_cat_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_spend_cat_validation_errors_total (Counter, labels: severity, type)
    4.  gl_spend_cat_batch_jobs_total (Counter, labels: status)
    5.  gl_spend_cat_active_jobs (Gauge)
    6.  gl_spend_cat_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_spend_cat_records_ingested_total (Counter, labels: source)
    8.  gl_spend_cat_records_classified_total (Counter, labels: taxonomy)
    9.  gl_spend_cat_scope3_mapped_total (Counter, labels: category)
    10. gl_spend_cat_emissions_calculated_total (Counter, labels: source)
    11. gl_spend_cat_classification_confidence (Histogram, buckets: 0.1-1.0)
    12. gl_spend_cat_emission_factor_lookups_total (Counter, labels: source)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    CONFIDENCE_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_spend_cat",
    "Spend Categorizer",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

spend_cat_records_ingested_total = m.create_custom_counter(
    "records_ingested_total",
    "Total spend records ingested",
    labelnames=["source"],
)

spend_cat_records_classified_total = m.create_custom_counter(
    "records_classified_total",
    "Total spend records classified",
    labelnames=["taxonomy"],
)

spend_cat_scope3_mapped_total = m.create_custom_counter(
    "scope3_mapped_total",
    "Total spend records mapped to Scope 3 categories",
    labelnames=["category"],
)

spend_cat_emissions_calculated_total = m.create_custom_counter(
    "emissions_calculated_total",
    "Total emission calculations completed",
    labelnames=["source"],
)

spend_cat_rules_evaluated_total = m.create_custom_counter(
    "rules_evaluated_total",
    "Total classification rules evaluated",
    labelnames=["result"],
)

spend_cat_reports_generated_total = m.create_custom_counter(
    "reports_generated_total",
    "Total reports generated",
    labelnames=["format"],
)

spend_cat_classification_confidence = m.create_custom_histogram(
    "classification_confidence",
    "Classification confidence score distribution",
    buckets=CONFIDENCE_BUCKETS,
)

spend_cat_active_batches = m.create_custom_gauge(
    "active_batches",
    "Number of currently active processing batches",
)

spend_cat_total_spend_usd = m.create_custom_gauge(
    "total_spend_usd",
    "Total cumulative spend amount in USD",
)

spend_cat_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)

spend_cat_emission_factor_lookups_total = m.create_custom_counter(
    "emission_factor_lookups_total",
    "Total emission factor lookups performed",
    labelnames=["source"],
)

# Backward-compat alias for standard metrics expected by __init__.py
spend_cat_processing_duration_seconds = m.processing_duration


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_ingestion(source: str) -> None:
    """Record a spend record ingestion event.

    Args:
        source: Ingestion source (csv, excel, api, erp, manual).
    """
    m.safe_inc(spend_cat_records_ingested_total, 1, source=source)


def record_classification(taxonomy: str) -> None:
    """Record a spend record classification event.

    Args:
        taxonomy: Taxonomy system used (unspsc, naics, nace, custom).
    """
    m.safe_inc(spend_cat_records_classified_total, 1, taxonomy=taxonomy)


def record_scope3_mapping(category: str) -> None:
    """Record a Scope 3 category mapping event.

    Args:
        category: GHG Protocol Scope 3 category (cat1..cat15).
    """
    m.safe_inc(spend_cat_scope3_mapped_total, 1, category=category)


def record_emission_calculation(source: str) -> None:
    """Record an emission calculation completion event.

    Args:
        source: Emission factor source (eeio, exiobase, defra, ecoinvent).
    """
    m.safe_inc(spend_cat_emissions_calculated_total, 1, source=source)


def record_rule_evaluation(result: str) -> None:
    """Record a classification rule evaluation event.

    Args:
        result: Evaluation result (match, no_match).
    """
    m.safe_inc(spend_cat_rules_evaluated_total, 1, result=result)


def record_report_generation(report_format: str) -> None:
    """Record a report generation event.

    Args:
        report_format: Report format (json, csv, excel, pdf).
    """
    m.safe_inc(spend_cat_reports_generated_total, 1, format=report_format)


def record_classification_confidence(confidence: float) -> None:
    """Record a classification confidence score observation.

    Args:
        confidence: Confidence score (0.0 - 1.0).
    """
    m.safe_observe(spend_cat_classification_confidence, confidence)


def record_processing_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (ingest, classify, map_scope3, calculate, etc.).
        duration: Duration in seconds.
    """
    m.record_operation(duration, type=operation, tenant_id="default")


def update_active_batches(delta: int) -> None:
    """Update the active batches gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not m.available:
        return
    if spend_cat_active_batches is not None:
        if delta > 0:
            spend_cat_active_batches.inc(delta)
        elif delta < 0:
            spend_cat_active_batches.dec(abs(delta))


def update_total_spend(amount: float) -> None:
    """Update the total spend USD gauge.

    Args:
        amount: Amount to add to the total spend (may be negative for corrections).
    """
    if not m.available:
        return
    if spend_cat_total_spend_usd is not None:
        spend_cat_total_spend_usd.inc(amount)


def record_processing_error(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data, integration, unknown).
    """
    m.safe_inc(spend_cat_processing_errors_total, 1, error_type=error_type)


def record_factor_lookup(source: str) -> None:
    """Record an emission factor lookup event.

    Args:
        source: Factor database source (eeio, exiobase, defra, ecoinvent).
    """
    m.safe_inc(spend_cat_emission_factor_lookups_total, 1, source=source)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "spend_cat_records_ingested_total",
    "spend_cat_records_classified_total",
    "spend_cat_scope3_mapped_total",
    "spend_cat_emissions_calculated_total",
    "spend_cat_rules_evaluated_total",
    "spend_cat_reports_generated_total",
    "spend_cat_classification_confidence",
    "spend_cat_active_batches",
    "spend_cat_total_spend_usd",
    "spend_cat_processing_errors_total",
    "spend_cat_emission_factor_lookups_total",
    # Helper functions
    "record_ingestion",
    "record_classification",
    "record_scope3_mapping",
    "record_emission_calculation",
    "record_rule_evaluation",
    "record_report_generation",
    "record_classification_confidence",
    "record_processing_duration",
    "update_active_batches",
    "update_total_spend",
    "record_processing_error",
    "record_factor_lookup",
]
