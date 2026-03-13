# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-027: Information Gathering Agent

18 Prometheus metrics for information gathering agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_iga_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (8):
        1.  gl_eudr_iga_gathering_operations_total       - Operations initiated
        2.  gl_eudr_iga_external_queries_total            - External queries by source
        3.  gl_eudr_iga_certifications_verified_total     - Certs verified by body
        4.  gl_eudr_iga_public_data_harvests_total        - Harvests by source
        5.  gl_eudr_iga_suppliers_aggregated_total        - Suppliers aggregated
        6.  gl_eudr_iga_completeness_validations_total    - Validations by class
        7.  gl_eudr_iga_packages_assembled_total          - Packages assembled
        8.  gl_eudr_iga_api_errors_total                  - API errors by operation

    Histograms (5):
        9.  gl_eudr_iga_external_query_duration_seconds   - Query latency
        10. gl_eudr_iga_certification_verification_duration_seconds - Cert latency
        11. gl_eudr_iga_harvest_duration_seconds           - Harvest latency
        12. gl_eudr_iga_aggregation_duration_seconds       - Aggregation latency
        13. gl_eudr_iga_package_assembly_duration_seconds  - Assembly latency

    Gauges (5):
        14. gl_eudr_iga_active_operations                  - Active operations
        15. gl_eudr_iga_stale_data_sources                 - Stale sources
        16. gl_eudr_iga_expiring_certificates              - Expiring certs
        17. gl_eudr_iga_cache_hit_ratio                    - Cache hit ratio
        18. gl_eudr_iga_normalization_errors                - Norm errors by type

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
Status: Production Ready
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not available; metrics disabled")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    # Counters
    _GATHERING_OPS = Counter(
        "gl_eudr_iga_gathering_operations_total",
        "Information gathering operations initiated",
        ["commodity", "status"],
    )
    _EXTERNAL_QUERIES = Counter(
        "gl_eudr_iga_external_queries_total",
        "External database queries executed",
        ["source", "status"],
    )
    _CERTS_VERIFIED = Counter(
        "gl_eudr_iga_certifications_verified_total",
        "Certificates verified",
        ["body", "status"],
    )
    _HARVESTS = Counter(
        "gl_eudr_iga_public_data_harvests_total",
        "Public data harvests completed",
        ["source"],
    )
    _SUPPLIERS_AGG = Counter(
        "gl_eudr_iga_suppliers_aggregated_total",
        "Supplier profiles aggregated",
        ["commodity"],
    )
    _COMPLETENESS_VALS = Counter(
        "gl_eudr_iga_completeness_validations_total",
        "Completeness validations performed",
        ["classification"],
    )
    _PACKAGES = Counter(
        "gl_eudr_iga_packages_assembled_total",
        "Information packages assembled",
        ["commodity"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_iga_api_errors_total",
        "API errors by operation type",
        ["operation"],
    )

    # Histograms
    _QUERY_DURATION = Histogram(
        "gl_eudr_iga_external_query_duration_seconds",
        "External database query latency",
        ["source"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _CERT_DURATION = Histogram(
        "gl_eudr_iga_certification_verification_duration_seconds",
        "Certificate verification latency",
        ["body"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _HARVEST_DURATION = Histogram(
        "gl_eudr_iga_harvest_duration_seconds",
        "Public data harvest latency",
        ["source"],
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )
    _AGG_DURATION = Histogram(
        "gl_eudr_iga_aggregation_duration_seconds",
        "Supplier aggregation latency",
        buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    )
    _PKG_DURATION = Histogram(
        "gl_eudr_iga_package_assembly_duration_seconds",
        "Package assembly latency",
        ["commodity"],
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    # Gauges
    _ACTIVE_OPS = Gauge(
        "gl_eudr_iga_active_operations",
        "Currently active gathering operations",
    )
    _STALE_SOURCES = Gauge(
        "gl_eudr_iga_stale_data_sources",
        "Data sources exceeding freshness threshold",
    )
    _EXPIRING_CERTS = Gauge(
        "gl_eudr_iga_expiring_certificates",
        "Certificates expiring within 90 days",
    )
    _CACHE_HIT = Gauge(
        "gl_eudr_iga_cache_hit_ratio",
        "External query cache hit ratio",
    )
    _NORM_ERRORS = Counter(
        "gl_eudr_iga_normalization_errors",
        "Data normalization errors",
        ["type"],
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_gathering_operation(commodity: str, status: str) -> None:
    """Record a gathering operation metric."""
    if _PROMETHEUS_AVAILABLE:
        _GATHERING_OPS.labels(commodity=commodity, status=status).inc()


def record_external_query(source: str, status: str) -> None:
    """Record an external database query metric."""
    if _PROMETHEUS_AVAILABLE:
        _EXTERNAL_QUERIES.labels(source=source, status=status).inc()


def record_certification_verified(body: str, status: str) -> None:
    """Record a certificate verification metric."""
    if _PROMETHEUS_AVAILABLE:
        _CERTS_VERIFIED.labels(body=body, status=status).inc()


def record_public_data_harvest(source: str) -> None:
    """Record a public data harvest metric."""
    if _PROMETHEUS_AVAILABLE:
        _HARVESTS.labels(source=source).inc()


def record_supplier_aggregated(commodity: str) -> None:
    """Record a supplier aggregation metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUPPLIERS_AGG.labels(commodity=commodity).inc()


def record_completeness_validation(classification: str) -> None:
    """Record a completeness validation metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLETENESS_VALS.labels(classification=classification).inc()


def record_package_assembled(commodity: str) -> None:
    """Record a package assembly metric."""
    if _PROMETHEUS_AVAILABLE:
        _PACKAGES.labels(commodity=commodity).inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


def observe_external_query_duration(source: str, duration: float) -> None:
    """Observe external query duration."""
    if _PROMETHEUS_AVAILABLE:
        _QUERY_DURATION.labels(source=source).observe(duration)


def observe_certification_duration(body: str, duration: float) -> None:
    """Observe certification verification duration."""
    if _PROMETHEUS_AVAILABLE:
        _CERT_DURATION.labels(body=body).observe(duration)


def observe_harvest_duration(source: str, duration: float) -> None:
    """Observe public data harvest duration."""
    if _PROMETHEUS_AVAILABLE:
        _HARVEST_DURATION.labels(source=source).observe(duration)


def observe_aggregation_duration(duration: float) -> None:
    """Observe supplier aggregation duration."""
    if _PROMETHEUS_AVAILABLE:
        _AGG_DURATION.observe(duration)


def observe_package_assembly_duration(commodity: str, duration: float) -> None:
    """Observe package assembly duration."""
    if _PROMETHEUS_AVAILABLE:
        _PKG_DURATION.labels(commodity=commodity).observe(duration)


def set_active_operations(count: int) -> None:
    """Set gauge of active operations."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_OPS.set(count)


def set_stale_data_sources(count: int) -> None:
    """Set gauge of stale data sources."""
    if _PROMETHEUS_AVAILABLE:
        _STALE_SOURCES.set(count)


def set_expiring_certificates(count: int) -> None:
    """Set gauge of expiring certificates."""
    if _PROMETHEUS_AVAILABLE:
        _EXPIRING_CERTS.set(count)


def set_cache_hit_ratio(ratio: float) -> None:
    """Set cache hit ratio gauge."""
    if _PROMETHEUS_AVAILABLE:
        _CACHE_HIT.set(ratio)


def record_normalization_error(type_name: str) -> None:
    """Record a normalization error."""
    if _PROMETHEUS_AVAILABLE:
        _NORM_ERRORS.labels(type=type_name).inc()
