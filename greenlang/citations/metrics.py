# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-005: Citations & Evidence

12 Prometheus metrics for citations and evidence monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_citations_operations_total (Counter)
    2.  gl_citations_operation_duration_seconds (Histogram)
    3.  gl_citations_verifications_total (Counter)
    4.  gl_citations_verification_failures_total (Counter)
    5.  gl_citations_evidence_packages_total (Counter)
    6.  gl_citations_evidence_items_total (Counter)
    7.  gl_citations_exports_total (Counter)
    8.  gl_citations_total (Gauge)
    9.  gl_citations_packages_total (Gauge)
    10. gl_citations_cache_hits_total (Counter)
    11. gl_citations_cache_misses_total (Counter)
    12. gl_citations_provenance_chain_depth (Histogram)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
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
        "prometheus_client not installed; citations metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Operations count
    citations_operations_total = Counter(
        "gl_citations_operations_total",
        "Total citations registry operations performed",
        labelnames=["operation", "result"],
    )

    # 2. Operation duration
    citations_operation_duration_seconds = Histogram(
        "gl_citations_operation_duration_seconds",
        "Citations registry operation duration in seconds",
        labelnames=["operation"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 3. Verifications count
    citations_verifications_total = Counter(
        "gl_citations_verifications_total",
        "Total citation verifications performed",
        labelnames=["result"],
    )

    # 4. Verification failures by reason
    citations_verification_failures_total = Counter(
        "gl_citations_verification_failures_total",
        "Total citation verification failures by reason",
        labelnames=["reason"],
    )

    # 5. Evidence packages created
    citations_evidence_packages_total = Counter(
        "gl_citations_evidence_packages_total",
        "Total evidence packages created",
    )

    # 6. Evidence items added
    citations_evidence_items_total = Counter(
        "gl_citations_evidence_items_total",
        "Total evidence items added to packages",
    )

    # 7. Exports by format
    citations_exports_total = Counter(
        "gl_citations_exports_total",
        "Total citation exports performed",
        labelnames=["format"],
    )

    # 8. Total citations gauge
    citations_total = Gauge(
        "gl_citations_total",
        "Current number of citations in registry",
    )

    # 9. Total packages gauge
    citations_packages_total = Gauge(
        "gl_citations_packages_total",
        "Current number of evidence packages",
    )

    # 10. Cache hits
    citations_cache_hits_total = Counter(
        "gl_citations_cache_hits_total",
        "Total citations cache hits",
    )

    # 11. Cache misses
    citations_cache_misses_total = Counter(
        "gl_citations_cache_misses_total",
        "Total citations cache misses",
    )

    # 12. Provenance chain depth
    citations_provenance_chain_depth = Histogram(
        "gl_citations_provenance_chain_depth",
        "Depth of citation provenance chains",
        buckets=(1, 2, 3, 4, 5, 7, 10, 15, 20),
    )

else:
    # No-op placeholders
    citations_operations_total = None  # type: ignore[assignment]
    citations_operation_duration_seconds = None  # type: ignore[assignment]
    citations_verifications_total = None  # type: ignore[assignment]
    citations_verification_failures_total = None  # type: ignore[assignment]
    citations_evidence_packages_total = None  # type: ignore[assignment]
    citations_evidence_items_total = None  # type: ignore[assignment]
    citations_exports_total = None  # type: ignore[assignment]
    citations_total = None  # type: ignore[assignment]
    citations_packages_total = None  # type: ignore[assignment]
    citations_cache_hits_total = None  # type: ignore[assignment]
    citations_cache_misses_total = None  # type: ignore[assignment]
    citations_provenance_chain_depth = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_operation(operation: str, result: str, duration_seconds: float) -> None:
    """Record a citations registry operation.

    Args:
        operation: Operation name (create, get, update, delete, etc.).
        result: Operation result ("success" or "error").
        duration_seconds: Operation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_operations_total.labels(operation=operation, result=result).inc()
    citations_operation_duration_seconds.labels(operation=operation).observe(
        duration_seconds,
    )


def record_verification(result: str) -> None:
    """Record a verification execution.

    Args:
        result: Verification result ("pass" or "fail").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_verifications_total.labels(result=result).inc()


def record_verification_failure(reason: str) -> None:
    """Record a verification failure for a specific reason.

    Args:
        reason: The reason the verification failed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_verification_failures_total.labels(reason=reason).inc()


def record_evidence_package() -> None:
    """Record a new evidence package creation."""
    if not PROMETHEUS_AVAILABLE:
        return
    citations_evidence_packages_total.inc()


def record_evidence_item() -> None:
    """Record an evidence item addition."""
    if not PROMETHEUS_AVAILABLE:
        return
    citations_evidence_items_total.inc()


def record_export(export_format: str) -> None:
    """Record a citation export operation.

    Args:
        export_format: Format of the export (bibtex, json, csl).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_exports_total.labels(format=export_format).inc()


def update_citations_count(count: int) -> None:
    """Set the total citations gauge.

    Args:
        count: Current number of citations.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_total.set(count)


def update_packages_count(count: int) -> None:
    """Set the total packages gauge.

    Args:
        count: Current number of packages.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_packages_total.set(count)


def record_cache_hit() -> None:
    """Record a citations cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    citations_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record a citations cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    citations_cache_misses_total.inc()


def record_provenance_depth(depth: int) -> None:
    """Record a provenance chain depth measurement.

    Args:
        depth: Depth of the provenance chain.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    citations_provenance_chain_depth.observe(depth)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "citations_operations_total",
    "citations_operation_duration_seconds",
    "citations_verifications_total",
    "citations_verification_failures_total",
    "citations_evidence_packages_total",
    "citations_evidence_items_total",
    "citations_exports_total",
    "citations_total",
    "citations_packages_total",
    "citations_cache_hits_total",
    "citations_cache_misses_total",
    "citations_provenance_chain_depth",
    # Helper functions
    "record_operation",
    "record_verification",
    "record_verification_failure",
    "record_evidence_package",
    "record_evidence_item",
    "record_export",
    "update_citations_count",
    "update_packages_count",
    "record_cache_hit",
    "record_cache_miss",
    "record_provenance_depth",
]
