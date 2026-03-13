# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-009: Chain of Custody

18 Prometheus metrics for chain of custody agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_coc_`` prefix (GreenLang EUDR Chain
of Custody) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (13):
        1.  gl_eudr_coc_events_recorded_total               - Custody events recorded
        2.  gl_eudr_coc_batches_created_total                - Batches created
        3.  gl_eudr_coc_batch_operations_total               - Batch operations (split/merge/blend)
        4.  gl_eudr_coc_mass_balance_entries_total           - Mass balance ledger entries
        5.  gl_eudr_coc_transformations_total                - Processing transformations
        6.  gl_eudr_coc_documents_linked_total               - Documents linked to events/batches
        7.  gl_eudr_coc_verifications_total                  - Chain verifications performed
        8.  gl_eudr_coc_verification_failures_total          - Chain verification failures
        9.  gl_eudr_coc_reports_generated_total              - Reports generated
        10. gl_eudr_coc_mass_balance_overdrafts_total        - Mass balance overdraft events
        11. gl_eudr_coc_custody_gaps_total                   - Custody gaps detected
        12. gl_eudr_coc_batch_jobs_total                     - Batch processing jobs
        13. gl_eudr_coc_api_errors_total                     - API errors by operation

    Histograms (3):
        14. gl_eudr_coc_event_recording_duration_seconds     - Event recording latency
        15. gl_eudr_coc_verification_duration_seconds        - Verification latency
        16. gl_eudr_coc_mass_balance_duration_seconds        - Mass balance operation latency

    Gauges (2):
        17. gl_eudr_coc_active_batches                       - Currently active batches
        18. gl_eudr_coc_chain_completeness_avg               - Avg chain completeness score

Label Values Reference:
    event_type:
        transfer, receipt, storage_in, storage_out, processing_in,
        processing_out, export, import_, inspection, sampling.
    operation_type:
        split, merge, blend, transform.
    entry_type:
        input, output, adjustment, carry_forward.
    process_type:
        drying, fermentation, roasting, milling, refining, pressing,
        extraction, etc. (25+ process types).
    severity:
        critical, high, medium, low.
    report_format:
        json, pdf, csv, eudr_xml.
    operation:
        record_event, create_batch, split_batch, merge_batch,
        blend_batch, record_transformation, link_document,
        verify_chain, generate_report, mass_balance_entry,
        assign_model, reconcile.

Example:
    >>> from greenlang.agents.eudr.chain_of_custody.metrics import (
    ...     record_event_recorded,
    ...     record_batch_created,
    ...     observe_event_recording_duration,
    ...     set_active_batches,
    ... )
    >>> record_event_recorded("transfer")
    >>> record_batch_created()
    >>> observe_event_recording_duration(0.045)
    >>> set_active_batches(156)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody (GL-EUDR-COC-009)
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
        "prometheus_client not installed; "
        "chain of custody metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
        buckets: tuple = (),
    ):  # type: ignore[return]
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics per PRD Section 7.3)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # -- Counters (13) -------------------------------------------------------

    # 1. Custody events recorded by event type
    coc_events_recorded_total = _safe_counter(
        "gl_eudr_coc_events_recorded_total",
        "Total custody events recorded by event type",
        labelnames=["event_type"],
    )

    # 2. Batches created
    coc_batches_created_total = _safe_counter(
        "gl_eudr_coc_batches_created_total",
        "Total commodity batches created",
    )

    # 3. Batch operations by operation type (split/merge/blend)
    coc_batch_operations_total = _safe_counter(
        "gl_eudr_coc_batch_operations_total",
        "Total batch operations performed by operation type",
        labelnames=["operation_type"],
    )

    # 4. Mass balance ledger entries by entry type
    coc_mass_balance_entries_total = _safe_counter(
        "gl_eudr_coc_mass_balance_entries_total",
        "Total mass balance ledger entries by entry type",
        labelnames=["entry_type"],
    )

    # 5. Processing transformations by process type
    coc_transformations_total = _safe_counter(
        "gl_eudr_coc_transformations_total",
        "Total processing transformations by process type",
        labelnames=["process_type"],
    )

    # 6. Documents linked to events/batches
    coc_documents_linked_total = _safe_counter(
        "gl_eudr_coc_documents_linked_total",
        "Total documents linked to custody events or batches",
    )

    # 7. Chain verifications performed
    coc_verifications_total = _safe_counter(
        "gl_eudr_coc_verifications_total",
        "Total chain of custody verifications performed",
    )

    # 8. Chain verification failures
    coc_verification_failures_total = _safe_counter(
        "gl_eudr_coc_verification_failures_total",
        "Total chain of custody verification failures",
    )

    # 9. Reports generated by format
    coc_reports_generated_total = _safe_counter(
        "gl_eudr_coc_reports_generated_total",
        "Total chain of custody reports generated by format",
        labelnames=["report_format"],
    )

    # 10. Mass balance overdraft events
    coc_mass_balance_overdrafts_total = _safe_counter(
        "gl_eudr_coc_mass_balance_overdrafts_total",
        "Total mass balance overdraft events detected",
    )

    # 11. Custody gaps detected by severity
    coc_custody_gaps_total = _safe_counter(
        "gl_eudr_coc_custody_gaps_total",
        "Total custody chain gaps detected by severity",
        labelnames=["severity"],
    )

    # 12. Batch processing jobs
    coc_batch_jobs_total = _safe_counter(
        "gl_eudr_coc_batch_jobs_total",
        "Total batch processing jobs executed",
    )

    # 13. API errors by operation
    coc_api_errors_total = _safe_counter(
        "gl_eudr_coc_api_errors_total",
        "Total API errors encountered by operation",
        labelnames=["operation"],
    )

    # -- Histograms (3) ------------------------------------------------------

    # 14. Event recording latency
    coc_event_recording_duration_seconds = _safe_histogram(
        "gl_eudr_coc_event_recording_duration_seconds",
        "Duration of custody event recording operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 15. Verification latency
    coc_verification_duration_seconds = _safe_histogram(
        "gl_eudr_coc_verification_duration_seconds",
        "Duration of chain of custody verification operations in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 16. Mass balance operation latency
    coc_mass_balance_duration_seconds = _safe_histogram(
        "gl_eudr_coc_mass_balance_duration_seconds",
        "Duration of mass balance operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0,
        ),
    )

    # -- Gauges (2) ----------------------------------------------------------

    # 17. Currently active batches
    coc_active_batches = _safe_gauge(
        "gl_eudr_coc_active_batches",
        "Number of currently active commodity batches",
    )

    # 18. Average chain completeness score
    coc_chain_completeness_avg = _safe_gauge(
        "gl_eudr_coc_chain_completeness_avg",
        "Average chain of custody completeness score (0.0-1.0)",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    coc_events_recorded_total = None               # type: ignore[assignment]
    coc_batches_created_total = None                # type: ignore[assignment]
    coc_batch_operations_total = None               # type: ignore[assignment]
    coc_mass_balance_entries_total = None            # type: ignore[assignment]
    coc_transformations_total = None                 # type: ignore[assignment]
    coc_documents_linked_total = None                # type: ignore[assignment]
    coc_verifications_total = None                   # type: ignore[assignment]
    coc_verification_failures_total = None           # type: ignore[assignment]
    coc_reports_generated_total = None               # type: ignore[assignment]
    coc_mass_balance_overdrafts_total = None          # type: ignore[assignment]
    coc_custody_gaps_total = None                    # type: ignore[assignment]
    coc_batch_jobs_total = None                      # type: ignore[assignment]
    coc_api_errors_total = None                      # type: ignore[assignment]
    coc_event_recording_duration_seconds = None       # type: ignore[assignment]
    coc_verification_duration_seconds = None          # type: ignore[assignment]
    coc_mass_balance_duration_seconds = None          # type: ignore[assignment]
    coc_active_batches = None                        # type: ignore[assignment]
    coc_chain_completeness_avg = None                # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_event_recorded(event_type: str) -> None:
    """Record a custody event recording event.

    Args:
        event_type: Type of custody event (transfer, receipt, storage_in,
            storage_out, processing_in, processing_out, export, import_,
            inspection, sampling).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_events_recorded_total.labels(event_type=event_type).inc()


def record_batch_created() -> None:
    """Record a batch creation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_batches_created_total.inc()


def record_batch_operation(operation_type: str) -> None:
    """Record a batch operation event.

    Args:
        operation_type: Type of operation (split, merge, blend, transform).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_batch_operations_total.labels(operation_type=operation_type).inc()


def record_mass_balance_entry(entry_type: str) -> None:
    """Record a mass balance ledger entry.

    Args:
        entry_type: Type of entry (input, output, adjustment, carry_forward).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_mass_balance_entries_total.labels(entry_type=entry_type).inc()


def record_transformation(process_type: str) -> None:
    """Record a processing transformation event.

    Args:
        process_type: Type of processing (drying, fermentation, roasting,
            milling, refining, pressing, extraction, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_transformations_total.labels(process_type=process_type).inc()


def record_document_linked() -> None:
    """Record a document linkage event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_documents_linked_total.inc()


def record_verification() -> None:
    """Record a chain verification event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_verifications_total.inc()


def record_verification_failure() -> None:
    """Record a chain verification failure event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_verification_failures_total.inc()


def record_report_generated(report_format: str) -> None:
    """Record a report generation event.

    Args:
        report_format: Format of the report (json, pdf, csv, eudr_xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_reports_generated_total.labels(report_format=report_format).inc()


def record_mass_balance_overdraft() -> None:
    """Record a mass balance overdraft event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_mass_balance_overdrafts_total.inc()


def record_custody_gap(severity: str) -> None:
    """Record a custody gap detection event.

    Args:
        severity: Severity level (critical, high, medium, low).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_custody_gaps_total.labels(severity=severity).inc()


def record_batch_job() -> None:
    """Record a batch processing job execution event."""
    if not PROMETHEUS_AVAILABLE:
        return
    coc_batch_jobs_total.inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (record_event,
            create_batch, split_batch, merge_batch, blend_batch,
            record_transformation, link_document, verify_chain,
            generate_report, mass_balance_entry, assign_model,
            reconcile).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_api_errors_total.labels(operation=operation).inc()


def observe_event_recording_duration(seconds: float) -> None:
    """Record the duration of a custody event recording operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_event_recording_duration_seconds.observe(seconds)


def observe_verification_duration(seconds: float) -> None:
    """Record the duration of a chain verification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_verification_duration_seconds.observe(seconds)


def observe_mass_balance_duration(seconds: float) -> None:
    """Record the duration of a mass balance operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_mass_balance_duration_seconds.observe(seconds)


def set_active_batches(count: int) -> None:
    """Set the gauge for currently active batches.

    Args:
        count: Number of active batches. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_active_batches.set(count)


def set_chain_completeness_avg(score: float) -> None:
    """Set the gauge for average chain completeness score.

    Args:
        score: Average completeness score (0.0-1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    coc_chain_completeness_avg.set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "coc_events_recorded_total",
    "coc_batches_created_total",
    "coc_batch_operations_total",
    "coc_mass_balance_entries_total",
    "coc_transformations_total",
    "coc_documents_linked_total",
    "coc_verifications_total",
    "coc_verification_failures_total",
    "coc_reports_generated_total",
    "coc_mass_balance_overdrafts_total",
    "coc_custody_gaps_total",
    "coc_batch_jobs_total",
    "coc_api_errors_total",
    "coc_event_recording_duration_seconds",
    "coc_verification_duration_seconds",
    "coc_mass_balance_duration_seconds",
    "coc_active_batches",
    "coc_chain_completeness_avg",
    # Helper functions
    "record_event_recorded",
    "record_batch_created",
    "record_batch_operation",
    "record_mass_balance_entry",
    "record_transformation",
    "record_document_linked",
    "record_verification",
    "record_verification_failure",
    "record_report_generated",
    "record_mass_balance_overdraft",
    "record_custody_gap",
    "record_batch_job",
    "record_api_error",
    "observe_event_recording_duration",
    "observe_verification_duration",
    "observe_mass_balance_duration",
    "set_active_batches",
    "set_chain_completeness_avg",
]
