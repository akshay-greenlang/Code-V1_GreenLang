# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-002: Refrigerants & F-Gas Agent

12 Prometheus metrics for refrigerants and fluorinated gas agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_rf_`` prefix (GreenLang Refrigerants & F-Gas)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_rf_calculations_total              (Counter,   labels: method, refrigerant_type, status)
    2.  gl_rf_emissions_kg_co2e_total         (Counter,   labels: refrigerant_type, category)
    3.  gl_rf_refrigerant_lookups_total       (Counter,   labels: source)
    4.  gl_rf_leak_rate_selections_total      (Counter,   labels: equipment_type, lifecycle_stage)
    5.  gl_rf_equipment_events_total          (Counter,   labels: equipment_type, event_type)
    6.  gl_rf_uncertainty_runs_total          (Counter,   labels: method)
    7.  gl_rf_compliance_checks_total         (Counter,   labels: framework, status)
    8.  gl_rf_batch_jobs_total                (Counter,   labels: status)
    9.  gl_rf_calculation_duration_seconds    (Histogram, labels: operation)
    10. gl_rf_batch_size                      (Histogram, labels: method)
    11. gl_rf_active_calculations             (Gauge)
    12. gl_rf_refrigerants_loaded             (Gauge,     labels: source)

Label Values Reference:
    method:
        EQUIPMENT_BASED, MASS_BALANCE, SCREENING, DIRECT_MEASUREMENT,
        TOP_DOWN, monte_carlo, analytical, tier_default.
    refrigerant_type:
        R_32, R_125, R_134A, R_410A, R_404A, R_407C, SF6_GAS, NF3_GAS,
        R_1234YF, R_22, R_744, R_290, CUSTOM, etc.
    status:
        completed, failed, pending, running, cancelled.
    category:
        hfc, hfc_blend, hfo, pfc, sf6, nf3, hcfc, cfc, natural, other.
    source:
        AR4, AR5, AR6, AR6_20YR, CUSTOM, database, cache.
    equipment_type:
        commercial_refrigeration_centralized, commercial_refrigeration_standalone,
        industrial_refrigeration, residential_ac, commercial_ac,
        chillers_centrifugal, chillers_screw, heat_pumps,
        transport_refrigeration, switchgear, semiconductor,
        fire_suppression, foam_blowing, aerosols, solvents.
    lifecycle_stage:
        installation, operating, end_of_life.
    event_type:
        installation, recharge, repair, recovery, leak_check,
        decommissioning, conversion.
    framework:
        ghg_protocol, iso_14064, csrd_esrs_e1, epa_40cfr98_dd,
        epa_40cfr98_oo, epa_40cfr98_l, eu_fgas_2024_573,
        kigali_amendment, uk_fgas.
    operation:
        single_calculation, batch_calculation, refrigerant_lookup,
        blend_decomposition, leak_rate_estimation, gwp_application,
        uncertainty_analysis, compliance_check, provenance_hash,
        pipeline_run.

Example:
    >>> from greenlang.refrigerants_fgas.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     set_active_calculations,
    ... )
    >>> record_calculation("EQUIPMENT_BASED", "R_410A", "completed")
    >>> record_emissions("R_410A", "hfc_blend", 2500.0)
    >>> set_active_calculations(3)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
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
        "prometheus_client not installed; refrigerants & f-gas metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by method, refrigerant type, and status
    rf_calculations_total = Counter(
        "gl_rf_calculations_total",
        "Total refrigerant and F-gas emission calculations performed",
        labelnames=["method", "refrigerant_type", "status"],
    )

    # 2. Cumulative emissions by refrigerant type and category
    rf_emissions_kg_co2e_total = Counter(
        "gl_rf_emissions_kg_co2e_total",
        "Cumulative emissions in kg CO2e by refrigerant type and category",
        labelnames=["refrigerant_type", "category"],
    )

    # 3. Refrigerant property and GWP lookups by source
    rf_refrigerant_lookups_total = Counter(
        "gl_rf_refrigerant_lookups_total",
        "Total refrigerant property and GWP lookups by source",
        labelnames=["source"],
    )

    # 4. Leak rate selections by equipment type and lifecycle stage
    rf_leak_rate_selections_total = Counter(
        "gl_rf_leak_rate_selections_total",
        "Total leak rate selections by equipment type and lifecycle stage",
        labelnames=["equipment_type", "lifecycle_stage"],
    )

    # 5. Equipment service events by equipment type and event type
    rf_equipment_events_total = Counter(
        "gl_rf_equipment_events_total",
        "Total equipment service events by type and event type",
        labelnames=["equipment_type", "event_type"],
    )

    # 6. Uncertainty analysis runs by method
    rf_uncertainty_runs_total = Counter(
        "gl_rf_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 7. Compliance checks by framework and status
    rf_compliance_checks_total = Counter(
        "gl_rf_compliance_checks_total",
        "Total compliance checks by regulatory framework and status",
        labelnames=["framework", "status"],
    )

    # 8. Batch calculation jobs by status
    rf_batch_jobs_total = Counter(
        "gl_rf_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    rf_calculation_duration_seconds = Histogram(
        "gl_rf_calculation_duration_seconds",
        "Duration of refrigerant and F-gas calculation operations in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 10. Batch size histogram by calculation method
    rf_batch_size = Histogram(
        "gl_rf_batch_size",
        "Number of calculations in batch requests by method",
        labelnames=["method"],
        buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
    )

    # 11. Currently active (in-progress) calculations
    rf_active_calculations = Gauge(
        "gl_rf_active_calculations",
        "Number of currently active refrigerant and F-gas calculations",
    )

    # 12. Number of refrigerants loaded by source
    rf_refrigerants_loaded = Gauge(
        "gl_rf_refrigerants_loaded",
        "Number of refrigerant definitions currently loaded by source",
        labelnames=["source"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    rf_calculations_total = None              # type: ignore[assignment]
    rf_emissions_kg_co2e_total = None         # type: ignore[assignment]
    rf_refrigerant_lookups_total = None       # type: ignore[assignment]
    rf_leak_rate_selections_total = None      # type: ignore[assignment]
    rf_equipment_events_total = None          # type: ignore[assignment]
    rf_uncertainty_runs_total = None          # type: ignore[assignment]
    rf_compliance_checks_total = None         # type: ignore[assignment]
    rf_batch_jobs_total = None                # type: ignore[assignment]
    rf_calculation_duration_seconds = None    # type: ignore[assignment]
    rf_batch_size = None                      # type: ignore[assignment]
    rf_active_calculations = None             # type: ignore[assignment]
    rf_refrigerants_loaded = None             # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(method: str, refrigerant_type: str, status: str) -> None:
    """Record a refrigerant and F-gas calculation event.

    Args:
        method: Calculation method used (EQUIPMENT_BASED, MASS_BALANCE,
            SCREENING, DIRECT_MEASUREMENT, TOP_DOWN).
        refrigerant_type: Type of refrigerant calculated (R_410A, SF6_GAS, etc.).
        status: Completion status (completed, failed, pending, running, cancelled).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_calculations_total.labels(
        method=method,
        refrigerant_type=refrigerant_type,
        status=status,
    ).inc()


def record_emissions(
    refrigerant_type: str, category: str, kg_co2e: float = 1.0
) -> None:
    """Record cumulative emissions by refrigerant type and category.

    Args:
        refrigerant_type: Type of refrigerant (R_410A, SF6_GAS, etc.).
        category: Refrigerant category (hfc, hfc_blend, pfc, sf6, etc.).
        kg_co2e: Emission amount in kg CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_emissions_kg_co2e_total.labels(
        refrigerant_type=refrigerant_type,
        category=category,
    ).inc(kg_co2e)


def record_refrigerant_lookup(source: str) -> None:
    """Record a refrigerant property or GWP lookup event.

    Args:
        source: Source queried (AR4, AR5, AR6, AR6_20YR, CUSTOM,
            database, cache).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_refrigerant_lookups_total.labels(
        source=source,
    ).inc()


def record_leak_rate_selection(
    equipment_type: str, lifecycle_stage: str
) -> None:
    """Record a leak rate selection event.

    Args:
        equipment_type: Type of equipment (commercial_refrigeration_centralized,
            chillers_centrifugal, switchgear, etc.).
        lifecycle_stage: Equipment lifecycle stage (installation, operating,
            end_of_life).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_leak_rate_selections_total.labels(
        equipment_type=equipment_type,
        lifecycle_stage=lifecycle_stage,
    ).inc()


def record_equipment_event(equipment_type: str, event_type: str) -> None:
    """Record an equipment service event.

    Args:
        equipment_type: Type of equipment (commercial_ac, switchgear, etc.).
        event_type: Service event type (installation, recharge, repair,
            recovery, leak_check, decommissioning, conversion).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_equipment_events_total.labels(
        equipment_type=equipment_type,
        event_type=event_type,
    ).inc()


def record_uncertainty(method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        method: Uncertainty method applied (monte_carlo, analytical,
            tier_default).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a regulatory compliance check event.

    Args:
        framework: Regulatory framework checked (ghg_protocol, eu_fgas_2024_573,
            kigali_amendment, etc.).
        status: Compliance status result (compliant, warning, non_compliant,
            exempted, not_applicable).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_compliance_checks_total.labels(
        framework=framework,
        status=status,
    ).inc()


def record_batch(status: str) -> None:
    """Record a batch calculation job event.

    Args:
        status: Completion status of the batch job
            (completed, failed, pending, running, cancelled).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_batch_jobs_total.labels(
        status=status,
    ).inc()


def observe_calculation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a calculation operation.

    Args:
        operation: Type of operation being measured
            (single_calculation, batch_calculation, refrigerant_lookup,
            blend_decomposition, leak_rate_estimation, gwp_application,
            uncertainty_analysis, compliance_check, provenance_hash,
            pipeline_run).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_calculation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_batch_size(method: str, size: int) -> None:
    """Record the size of a batch calculation request.

    Args:
        method: Calculation method used in the batch (EQUIPMENT_BASED,
            MASS_BALANCE, SCREENING, or "mixed" for multi-method batches).
        size: Number of individual calculations in the batch.
            Buckets: 1, 5, 10, 50, 100, 500, 1000, 5000, 10000.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_batch_size.labels(
        method=method,
    ).observe(size)


def set_active_calculations(count: int) -> None:
    """Set the gauge for currently active in-progress calculations.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of calculations currently in progress. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_active_calculations.set(count)


def set_refrigerants_loaded(source: str, count: int) -> None:
    """Set the gauge for number of refrigerant definitions loaded by source.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        source: Source of the loaded definitions (AR4, AR5, AR6,
            AR6_20YR, CUSTOM, database).
        count: Number of refrigerant definitions currently loaded from
            this source. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    rf_refrigerants_loaded.labels(
        source=source,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "rf_calculations_total",
    "rf_emissions_kg_co2e_total",
    "rf_refrigerant_lookups_total",
    "rf_leak_rate_selections_total",
    "rf_equipment_events_total",
    "rf_uncertainty_runs_total",
    "rf_compliance_checks_total",
    "rf_batch_jobs_total",
    "rf_calculation_duration_seconds",
    "rf_batch_size",
    "rf_active_calculations",
    "rf_refrigerants_loaded",
    # Helper functions
    "record_calculation",
    "record_emissions",
    "record_refrigerant_lookup",
    "record_leak_rate_selection",
    "record_equipment_event",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_refrigerants_loaded",
]
