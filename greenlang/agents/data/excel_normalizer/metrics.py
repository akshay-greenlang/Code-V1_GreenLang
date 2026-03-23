# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-002: Excel & CSV Normalizer

12 Prometheus metrics for Excel normalizer service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_excel_files_processed_total (Counter, labels: file_format, tenant_id)
    2.  gl_excel_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_excel_rows_normalized_total (Counter)
    4.  gl_excel_columns_mapped_total (Counter, labels: strategy, data_type)
    5.  gl_excel_mapping_confidence (Histogram, buckets: 0.1-1.0 by 0.1)
    6.  gl_excel_quality_score (Histogram, buckets: 0.1-1.0 by 0.1)
    7.  gl_excel_validation_findings_total (Counter, labels: severity, rule_name)
    8.  gl_excel_transforms_total (Counter, labels: operation)
    9.  gl_excel_type_detections_total (Counter, labels: data_type)
    10. gl_excel_batch_jobs_total (Counter, labels: status)
    11. gl_excel_active_jobs (Gauge)
    12. gl_excel_queue_size (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
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
        "prometheus_client not installed; Excel normalizer metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Files processed by format and tenant
    excel_files_processed_total = Counter(
        "gl_excel_files_processed_total",
        "Total Excel/CSV files processed",
        labelnames=["file_format", "tenant_id"],
    )

    # 2. Processing duration histogram (12 buckets covering sub-second to
    #    multi-minute large-file normalizations)
    excel_processing_duration_seconds = Histogram(
        "gl_excel_processing_duration_seconds",
        "Excel/CSV file processing duration in seconds",
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 15.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 3. Total rows normalized across all files
    excel_rows_normalized_total = Counter(
        "gl_excel_rows_normalized_total",
        "Total rows normalized from Excel/CSV files",
    )

    # 4. Columns mapped by strategy and detected data type
    excel_columns_mapped_total = Counter(
        "gl_excel_columns_mapped_total",
        "Total columns mapped to canonical fields",
        labelnames=["strategy", "data_type"],
    )

    # 5. Column mapping confidence scores (0.1 to 1.0 in 0.1 increments)
    excel_mapping_confidence = Histogram(
        "gl_excel_mapping_confidence",
        "Column mapping confidence score distribution",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )

    # 6. Data quality scores (0.1 to 1.0 in 0.1 increments)
    excel_quality_score = Histogram(
        "gl_excel_quality_score",
        "Data quality score distribution",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )

    # 7. Validation findings by severity and rule name
    excel_validation_findings_total = Counter(
        "gl_excel_validation_findings_total",
        "Total validation findings detected during normalization",
        labelnames=["severity", "rule_name"],
    )

    # 8. Transform operations by type
    excel_transforms_total = Counter(
        "gl_excel_transforms_total",
        "Total data transformation operations",
        labelnames=["operation"],
    )

    # 9. Data type detections by type
    excel_type_detections_total = Counter(
        "gl_excel_type_detections_total",
        "Total data type detections by type",
        labelnames=["data_type"],
    )

    # 10. Batch jobs by status
    excel_batch_jobs_total = Counter(
        "gl_excel_batch_jobs_total",
        "Total batch normalization jobs",
        labelnames=["status"],
    )

    # 11. Currently active normalization jobs
    excel_active_jobs = Gauge(
        "gl_excel_active_jobs",
        "Number of currently active normalization jobs",
    )

    # 12. Current normalization queue depth
    excel_queue_size = Gauge(
        "gl_excel_queue_size",
        "Current number of files waiting in normalization queue",
    )

else:
    # No-op placeholders
    excel_files_processed_total = None  # type: ignore[assignment]
    excel_processing_duration_seconds = None  # type: ignore[assignment]
    excel_rows_normalized_total = None  # type: ignore[assignment]
    excel_columns_mapped_total = None  # type: ignore[assignment]
    excel_mapping_confidence = None  # type: ignore[assignment]
    excel_quality_score = None  # type: ignore[assignment]
    excel_validation_findings_total = None  # type: ignore[assignment]
    excel_transforms_total = None  # type: ignore[assignment]
    excel_type_detections_total = None  # type: ignore[assignment]
    excel_batch_jobs_total = None  # type: ignore[assignment]
    excel_active_jobs = None  # type: ignore[assignment]
    excel_queue_size = None  # type: ignore[assignment]


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
    if not PROMETHEUS_AVAILABLE:
        return
    excel_files_processed_total.labels(
        file_format=file_format, tenant_id=tenant_id,
    ).inc()
    excel_processing_duration_seconds.observe(duration_seconds)


def record_rows_normalized(row_count: int) -> None:
    """Record the number of rows normalized from a file.

    Args:
        row_count: Number of rows normalized.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_rows_normalized_total.inc(row_count)


def record_columns_mapped(
    count: int,
    strategy: str,
    data_type: str,
) -> None:
    """Record mapped columns by strategy and detected data type.

    Args:
        count: Number of columns mapped.
        strategy: Mapping strategy used (exact, fuzzy, synonym, pattern, manual).
        data_type: Detected data type of the column.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_columns_mapped_total.labels(
        strategy=strategy, data_type=data_type,
    ).inc(count)


def record_mapping_confidence(confidence: float) -> None:
    """Record a column mapping confidence score.

    Args:
        confidence: Confidence score between 0.0 and 1.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_mapping_confidence.observe(confidence)


def record_quality_score(score: float) -> None:
    """Record a data quality score.

    Args:
        score: Quality score between 0.0 and 1.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_quality_score.observe(score)


def record_validation_finding(severity: str, rule_name: str) -> None:
    """Record a validation finding during normalization.

    Args:
        severity: Finding severity (error, warning, info).
        rule_name: Name of the validation rule that triggered.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_validation_findings_total.labels(
        severity=severity, rule_name=rule_name,
    ).inc()


def record_transform(operation: str) -> None:
    """Record a data transformation operation.

    Args:
        operation: Transform operation type (pivot, dedup, merge, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_transforms_total.labels(operation=operation).inc()


def record_type_detection(data_type: str) -> None:
    """Record a data type detection event.

    Args:
        data_type: Detected data type.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_type_detections_total.labels(data_type=data_type).inc()


def record_batch_job(status: str) -> None:
    """Record a batch job event.

    Args:
        status: Batch job status (submitted, completed, failed, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_batch_jobs_total.labels(status=status).inc()


def update_active_jobs(delta: int) -> None:
    """Update the active jobs gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        excel_active_jobs.inc(delta)
    elif delta < 0:
        excel_active_jobs.dec(abs(delta))


def update_queue_size(size: int) -> None:
    """Set the current normalization queue size.

    Args:
        size: Current queue depth.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    excel_queue_size.set(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "excel_files_processed_total",
    "excel_processing_duration_seconds",
    "excel_rows_normalized_total",
    "excel_columns_mapped_total",
    "excel_mapping_confidence",
    "excel_quality_score",
    "excel_validation_findings_total",
    "excel_transforms_total",
    "excel_type_detections_total",
    "excel_batch_jobs_total",
    "excel_active_jobs",
    "excel_queue_size",
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
