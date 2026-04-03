# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-002: Excel & CSV Normalizer

12 Prometheus metrics for Excel normalizer service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_excel_operations_total (Counter, labels: type, tenant_id)
    2.  gl_excel_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_excel_validation_errors_total (Counter, labels: severity, type)
    4.  gl_excel_batch_jobs_total (Counter, labels: status)
    5.  gl_excel_active_jobs (Gauge)
    6.  gl_excel_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_excel_rows_normalized_total (Counter)
    8.  gl_excel_columns_mapped_total (Counter, labels: strategy, data_type)
    9.  gl_excel_mapping_confidence (Histogram, buckets: 0.1-1.0 by 0.1)
    10. gl_excel_quality_score (Histogram, buckets: 0.1-1.0 by 0.1)
    11. gl_excel_transforms_total (Counter, labels: operation)
    12. gl_excel_type_detections_total (Counter, labels: data_type)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
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

m = MetricsFactory("gl_excel", "Excel Normalizer")

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

excel_rows_normalized_total = m.create_custom_counter(
    "rows_normalized_total",
    "Total rows normalized from Excel/CSV files",
)

excel_columns_mapped_total = m.create_custom_counter(
    "columns_mapped_total",
    "Total columns mapped to canonical fields",
    labelnames=["strategy", "data_type"],
)

excel_mapping_confidence = m.create_custom_histogram(
    "mapping_confidence",
    "Column mapping confidence score distribution",
    buckets=CONFIDENCE_BUCKETS,
)

excel_quality_score = m.create_custom_histogram(
    "quality_score",
    "Data quality score distribution",
    buckets=CONFIDENCE_BUCKETS,
)

excel_transforms_total = m.create_custom_counter(
    "transforms_total",
    "Total data transformation operations",
    labelnames=["operation"],
)

excel_type_detections_total = m.create_custom_counter(
    "type_detections_total",
    "Total data type detections by type",
    labelnames=["data_type"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_file_processed(
    file_format: str,
    tenant_id: str,
    duration_seconds: float,
) -> None:
    """Record a file processing completion with duration.

    Args:
        file_format: Format of file processed (xlsx, xls, csv, tsv).
        tenant_id: Tenant identifier.
        duration_seconds: Total processing duration in seconds.
    """
    m.record_operation(duration_seconds, type=file_format, tenant_id=tenant_id)


def record_rows_normalized(row_count: int) -> None:
    """Record the number of rows normalized from a file.

    Args:
        row_count: Number of rows normalized.
    """
    m.safe_inc(excel_rows_normalized_total, row_count)


def record_columns_mapped(count: int, strategy: str, data_type: str) -> None:
    """Record mapped columns by strategy and detected data type.

    Args:
        count: Number of columns mapped.
        strategy: Mapping strategy used (exact, fuzzy, synonym, pattern, manual).
        data_type: Detected data type of the column.
    """
    m.safe_inc(excel_columns_mapped_total, count, strategy=strategy, data_type=data_type)


def record_mapping_confidence(confidence: float) -> None:
    """Record a column mapping confidence score.

    Args:
        confidence: Confidence score between 0.0 and 1.0.
    """
    m.safe_observe(excel_mapping_confidence, confidence)


def record_quality_score(score: float) -> None:
    """Record a data quality score.

    Args:
        score: Quality score between 0.0 and 1.0.
    """
    m.safe_observe(excel_quality_score, score)


def record_validation_finding(severity: str, rule_name: str) -> None:
    """Record a validation finding during normalization.

    Args:
        severity: Finding severity (error, warning, info).
        rule_name: Name of the validation rule that triggered.
    """
    m.record_validation_error(severity=severity, type=rule_name)


def record_transform(operation: str) -> None:
    """Record a data transformation operation.

    Args:
        operation: Transform operation type (pivot, dedup, merge, etc.).
    """
    m.safe_inc(excel_transforms_total, 1, operation=operation)


def record_type_detection(data_type: str) -> None:
    """Record a data type detection event.

    Args:
        data_type: Detected data type.
    """
    m.safe_inc(excel_type_detections_total, 1, data_type=data_type)


def record_batch_job(status: str) -> None:
    """Record a batch job event.

    Args:
        status: Batch job status (submitted, completed, failed, partial).
    """
    m.record_batch_job(status)


def update_active_jobs(delta: int) -> None:
    """Update the active jobs gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    m.update_active_jobs(delta)


def update_queue_size(size: int) -> None:
    """Set the current normalization queue size.

    Args:
        size: Current queue depth.
    """
    m.update_queue_size(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Agent-specific metric objects
    "excel_rows_normalized_total",
    "excel_columns_mapped_total",
    "excel_mapping_confidence",
    "excel_quality_score",
    "excel_transforms_total",
    "excel_type_detections_total",
    # Helper functions
    "record_file_processed",
    "record_rows_normalized",
    "record_columns_mapped",
    "record_mapping_confidence",
    "record_quality_score",
    "record_validation_finding",
    "record_transform",
    "record_type_detection",
    "record_batch_job",
    "update_active_jobs",
    "update_queue_size",
]
