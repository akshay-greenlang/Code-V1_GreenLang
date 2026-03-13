# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-011: Mass Balance Calculator

18 Prometheus metrics for mass balance calculator agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_mbc_`` prefix (GreenLang EUDR Mass
Balance Calculator) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (13):
        1.  gl_eudr_mbc_ledger_entries_total               - Ledger entries recorded
        2.  gl_eudr_mbc_input_entries_total                 - Input entries recorded
        3.  gl_eudr_mbc_output_entries_total                - Output entries recorded
        4.  gl_eudr_mbc_overdrafts_detected_total           - Overdrafts detected
        5.  gl_eudr_mbc_overdrafts_critical_total           - Critical overdrafts
        6.  gl_eudr_mbc_conversion_validations_total        - Conversion factor validations
        7.  gl_eudr_mbc_conversion_rejections_total         - Conversion factor rejections
        8.  gl_eudr_mbc_losses_recorded_total               - Losses recorded
        9.  gl_eudr_mbc_credits_expired_total               - Credits expired
        10. gl_eudr_mbc_reconciliations_total               - Reconciliations performed
        11. gl_eudr_mbc_reports_generated_total             - Reports generated
        12. gl_eudr_mbc_batch_jobs_total                    - Batch processing jobs
        13. gl_eudr_mbc_api_errors_total                    - API errors by operation

    Histograms (3):
        14. gl_eudr_mbc_entry_recording_duration_seconds    - Entry recording latency
        15. gl_eudr_mbc_reconciliation_duration_seconds     - Reconciliation latency
        16. gl_eudr_mbc_overdraft_check_duration_seconds    - Overdraft check latency

    Gauges (2):
        17. gl_eudr_mbc_active_ledgers                      - Currently active ledgers
        18. gl_eudr_mbc_total_balance_kg                    - Total balance in kg

Label Values Reference:
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    entry_type:
        input, output, adjustment, loss, waste, carry_forward_in,
        carry_forward_out, expiry.
    severity:
        warning, violation, critical.
    standard:
        rspo, fsc, iscc, utz_ra, fairtrade, eudr_default.
    facility_id:
        Dynamic per-facility identifier.
    report_type:
        reconciliation, consolidation, overdraft, variance, evidence.
    operation:
        create_ledger, record_entry, bulk_entry, create_period,
        validate_factor, check_overdraft, record_loss,
        run_reconciliation, generate_report, consolidate.

Example:
    >>> from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    ...     record_ledger_entry,
    ...     record_input_entry,
    ...     record_overdraft_detected,
    ...     observe_entry_recording_duration,
    ...     set_active_ledgers,
    ... )
    >>> record_ledger_entry("input", "cocoa")
    >>> record_input_entry("cocoa")
    >>> record_overdraft_detected("warning")
    >>> observe_entry_recording_duration(0.045)
    >>> set_active_ledgers(42)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator (GL-EUDR-MBC-011)
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
        "mass balance calculator metrics disabled"
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
            return Histogram(
                name, doc, labelnames=labelnames or [], **kw,
            )
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

    # 1. Ledger entries recorded by entry type and commodity
    mbc_ledger_entries_total = _safe_counter(
        "gl_eudr_mbc_ledger_entries_total",
        "Total ledger entries recorded by entry type and commodity",
        labelnames=["entry_type", "commodity"],
    )

    # 2. Input entries recorded by commodity
    mbc_input_entries_total = _safe_counter(
        "gl_eudr_mbc_input_entries_total",
        "Total input entries recorded by commodity",
        labelnames=["commodity"],
    )

    # 3. Output entries recorded by commodity
    mbc_output_entries_total = _safe_counter(
        "gl_eudr_mbc_output_entries_total",
        "Total output entries recorded by commodity",
        labelnames=["commodity"],
    )

    # 4. Overdrafts detected by severity
    mbc_overdrafts_detected_total = _safe_counter(
        "gl_eudr_mbc_overdrafts_detected_total",
        "Total overdraft events detected by severity",
        labelnames=["severity"],
    )

    # 5. Critical overdrafts detected
    mbc_overdrafts_critical_total = _safe_counter(
        "gl_eudr_mbc_overdrafts_critical_total",
        "Total critical overdraft events detected",
    )

    # 6. Conversion factor validations by commodity
    mbc_conversion_validations_total = _safe_counter(
        "gl_eudr_mbc_conversion_validations_total",
        "Total conversion factor validations performed by commodity",
        labelnames=["commodity"],
    )

    # 7. Conversion factor rejections by commodity
    mbc_conversion_rejections_total = _safe_counter(
        "gl_eudr_mbc_conversion_rejections_total",
        "Total conversion factor rejections by commodity",
        labelnames=["commodity"],
    )

    # 8. Losses recorded by loss type
    mbc_losses_recorded_total = _safe_counter(
        "gl_eudr_mbc_losses_recorded_total",
        "Total losses recorded by loss type",
        labelnames=["loss_type"],
    )

    # 9. Credits expired by standard
    mbc_credits_expired_total = _safe_counter(
        "gl_eudr_mbc_credits_expired_total",
        "Total credits expired by certification standard",
        labelnames=["standard"],
    )

    # 10. Reconciliations performed by facility
    mbc_reconciliations_total = _safe_counter(
        "gl_eudr_mbc_reconciliations_total",
        "Total reconciliations performed",
        labelnames=["facility_id"],
    )

    # 11. Reports generated by report type
    mbc_reports_generated_total = _safe_counter(
        "gl_eudr_mbc_reports_generated_total",
        "Total reports generated by report type",
        labelnames=["report_type"],
    )

    # 12. Batch processing jobs
    mbc_batch_jobs_total = _safe_counter(
        "gl_eudr_mbc_batch_jobs_total",
        "Total batch processing jobs executed",
    )

    # 13. API errors by operation
    mbc_api_errors_total = _safe_counter(
        "gl_eudr_mbc_api_errors_total",
        "Total API errors encountered by operation",
        labelnames=["operation"],
    )

    # -- Histograms (3) ------------------------------------------------------

    # 14. Entry recording latency
    mbc_entry_recording_duration_seconds = _safe_histogram(
        "gl_eudr_mbc_entry_recording_duration_seconds",
        "Duration of ledger entry recording operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 15. Reconciliation latency
    mbc_reconciliation_duration_seconds = _safe_histogram(
        "gl_eudr_mbc_reconciliation_duration_seconds",
        "Duration of reconciliation operations in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 16. Overdraft check latency
    mbc_overdraft_check_duration_seconds = _safe_histogram(
        "gl_eudr_mbc_overdraft_check_duration_seconds",
        "Duration of overdraft check operations in seconds",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0,
        ),
    )

    # -- Gauges (2) ----------------------------------------------------------

    # 17. Currently active ledgers
    mbc_active_ledgers = _safe_gauge(
        "gl_eudr_mbc_active_ledgers",
        "Number of currently active mass balance ledgers",
    )

    # 18. Total balance in kilograms across all ledgers
    mbc_total_balance_kg = _safe_gauge(
        "gl_eudr_mbc_total_balance_kg",
        "Total mass balance across all active ledgers in kilograms",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    mbc_ledger_entries_total = None               # type: ignore[assignment]
    mbc_input_entries_total = None                 # type: ignore[assignment]
    mbc_output_entries_total = None                # type: ignore[assignment]
    mbc_overdrafts_detected_total = None           # type: ignore[assignment]
    mbc_overdrafts_critical_total = None           # type: ignore[assignment]
    mbc_conversion_validations_total = None        # type: ignore[assignment]
    mbc_conversion_rejections_total = None         # type: ignore[assignment]
    mbc_losses_recorded_total = None               # type: ignore[assignment]
    mbc_credits_expired_total = None               # type: ignore[assignment]
    mbc_reconciliations_total = None               # type: ignore[assignment]
    mbc_reports_generated_total = None             # type: ignore[assignment]
    mbc_batch_jobs_total = None                    # type: ignore[assignment]
    mbc_api_errors_total = None                    # type: ignore[assignment]
    mbc_entry_recording_duration_seconds = None    # type: ignore[assignment]
    mbc_reconciliation_duration_seconds = None     # type: ignore[assignment]
    mbc_overdraft_check_duration_seconds = None    # type: ignore[assignment]
    mbc_active_ledgers = None                      # type: ignore[assignment]
    mbc_total_balance_kg = None                    # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_ledger_entry(entry_type: str, commodity: str) -> None:
    """Record a ledger entry event by entry type and commodity.

    Args:
        entry_type: Type of ledger entry (input, output, adjustment,
            loss, waste, carry_forward_in, carry_forward_out, expiry).
        commodity: EUDR commodity (cattle, cocoa, coffee, oil_palm,
            rubber, soya, wood).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_ledger_entries_total.labels(
        entry_type=entry_type, commodity=commodity,
    ).inc()


def record_input_entry(commodity: str) -> None:
    """Record an input entry event by commodity.

    Args:
        commodity: EUDR commodity identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_input_entries_total.labels(commodity=commodity).inc()


def record_output_entry(commodity: str) -> None:
    """Record an output entry event by commodity.

    Args:
        commodity: EUDR commodity identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_output_entries_total.labels(commodity=commodity).inc()


def record_overdraft_detected(severity: str) -> None:
    """Record an overdraft detection event by severity.

    Args:
        severity: Overdraft severity (warning, violation, critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_overdrafts_detected_total.labels(severity=severity).inc()


def record_overdraft_critical() -> None:
    """Record a critical overdraft event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_overdrafts_critical_total.inc()


def record_conversion_validation(commodity: str) -> None:
    """Record a conversion factor validation event.

    Args:
        commodity: EUDR commodity being validated.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_conversion_validations_total.labels(commodity=commodity).inc()


def record_conversion_rejection(commodity: str) -> None:
    """Record a conversion factor rejection event.

    Args:
        commodity: EUDR commodity that was rejected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_conversion_rejections_total.labels(commodity=commodity).inc()


def record_loss_recorded(loss_type: str) -> None:
    """Record a loss recording event by loss type.

    Args:
        loss_type: Type of loss (processing_loss, transport_loss,
            storage_loss, quality_rejection, spillage,
            contamination_loss).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_losses_recorded_total.labels(loss_type=loss_type).inc()


def record_credit_expired(standard: str) -> None:
    """Record a credit expiry event by standard.

    Args:
        standard: Certification standard (rspo, fsc, iscc, utz_ra,
            fairtrade, eudr_default).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_credits_expired_total.labels(standard=standard).inc()


def record_reconciliation(facility_id: str) -> None:
    """Record a reconciliation event by facility.

    Args:
        facility_id: Identifier of the facility being reconciled.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_reconciliations_total.labels(facility_id=facility_id).inc()


def record_report_generated(report_type: str) -> None:
    """Record a report generation event.

    Args:
        report_type: Type of report (reconciliation, consolidation,
            overdraft, variance, evidence).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_reports_generated_total.labels(report_type=report_type).inc()


def record_batch_job() -> None:
    """Record a batch processing job execution event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_batch_jobs_total.inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (create_ledger,
            record_entry, bulk_entry, create_period, validate_factor,
            check_overdraft, record_loss, run_reconciliation,
            generate_report, consolidate).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_api_errors_total.labels(operation=operation).inc()


def observe_entry_recording_duration(seconds: float) -> None:
    """Record the duration of a ledger entry recording operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_entry_recording_duration_seconds.observe(seconds)


def observe_reconciliation_duration(seconds: float) -> None:
    """Record the duration of a reconciliation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_reconciliation_duration_seconds.observe(seconds)


def observe_overdraft_check_duration(seconds: float) -> None:
    """Record the duration of an overdraft check operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_overdraft_check_duration_seconds.observe(seconds)


def set_active_ledgers(count: int) -> None:
    """Set the gauge for currently active ledgers.

    Args:
        count: Number of active ledgers. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_active_ledgers.set(count)


def set_total_balance_kg(balance: float) -> None:
    """Set the gauge for total mass balance across all ledgers.

    Args:
        balance: Total balance in kilograms.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mbc_total_balance_kg.set(balance)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "mbc_ledger_entries_total",
    "mbc_input_entries_total",
    "mbc_output_entries_total",
    "mbc_overdrafts_detected_total",
    "mbc_overdrafts_critical_total",
    "mbc_conversion_validations_total",
    "mbc_conversion_rejections_total",
    "mbc_losses_recorded_total",
    "mbc_credits_expired_total",
    "mbc_reconciliations_total",
    "mbc_reports_generated_total",
    "mbc_batch_jobs_total",
    "mbc_api_errors_total",
    "mbc_entry_recording_duration_seconds",
    "mbc_reconciliation_duration_seconds",
    "mbc_overdraft_check_duration_seconds",
    "mbc_active_ledgers",
    "mbc_total_balance_kg",
    # Helper functions
    "record_ledger_entry",
    "record_input_entry",
    "record_output_entry",
    "record_overdraft_detected",
    "record_overdraft_critical",
    "record_conversion_validation",
    "record_conversion_rejection",
    "record_loss_recorded",
    "record_credit_expired",
    "record_reconciliation",
    "record_report_generated",
    "record_batch_job",
    "record_api_error",
    "observe_entry_recording_duration",
    "observe_reconciliation_duration",
    "observe_overdraft_check_duration",
    "set_active_ledgers",
    "set_total_balance_kg",
]
