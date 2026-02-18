# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-005: Fugitive Emissions Agent

12 Prometheus metrics for fugitive emissions agent service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_fe_`` prefix (GreenLang Fugitive Emissions)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_fe_calculations_total              (Counter,   labels: source_type, method, status)
    2.  gl_fe_emissions_kg_co2e_total         (Counter,   labels: source_type, gas)
    3.  gl_fe_source_lookups_total            (Counter,   labels: source)
    4.  gl_fe_factor_selections_total         (Counter,   labels: method, source)
    5.  gl_fe_ldar_surveys_total              (Counter,   labels: survey_type, status)
    6.  gl_fe_uncertainty_runs_total          (Counter,   labels: method)
    7.  gl_fe_compliance_checks_total         (Counter,   labels: framework, status)
    8.  gl_fe_batch_jobs_total                (Counter,   labels: status)
    9.  gl_fe_calculation_duration_seconds    (Histogram, labels: operation)
    10. gl_fe_batch_size                      (Histogram, labels: method)
    11. gl_fe_active_calculations             (Gauge)
    12. gl_fe_components_registered           (Gauge,     labels: component_type)

Label Values Reference:
    source_type:
        wellhead, separator, dehydrator, pneumatic_controller_high,
        pneumatic_controller_low, pneumatic_controller_intermittent,
        compressor_centrifugal, compressor_reciprocating, acid_gas_removal,
        glycol_dehydrator, pipeline_main, pipeline_service, meter_regulator,
        tank_fixed_roof, tank_floating_roof, coal_mine_underground,
        coal_mine_surface, coal_handling, abandoned_mine,
        wastewater_lagoon, wastewater_digester, wastewater_aerobic,
        valve_gas, pump_seal, compressor_seal, flange_connector.
    method:
        AVERAGE_EMISSION_FACTOR, SCREENING_RANGES, CORRELATION_EQUATION,
        ENGINEERING_ESTIMATE, DIRECT_MEASUREMENT.
    status:
        completed, failed, pending, running.
    gas:
        CH4, CO2, N2O, VOC.
    source:
        EPA, IPCC, DEFRA, EU_ETS, API, CUSTOM.
    survey_type:
        OGI, METHOD_21, AVO, HI_FLOW.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, EPA_40CFR98, UK_SECR,
        EU_ETS.
    component_type:
        valve, pump, compressor, pressure_relief_device, connector,
        open_ended_line, sampling_connection, flange, other.
    operation:
        single_calculation, batch_calculation, factor_lookup,
        unit_conversion, gwp_application, uncertainty_analysis,
        provenance_hash, compliance_check, ldar_survey_processing,
        component_registration.

Example:
    >>> from greenlang.fugitive_emissions.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     track_active_calculation,
    ... )
    >>> record_calculation("valve_gas", "AVERAGE_EMISSION_FACTOR", "completed")
    >>> record_emissions("valve_gas", "CH4", 120.5)
    >>> track_active_calculation(3)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
        "prometheus_client not installed; fugitive emissions metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by source type, method, and completion status
    fe_calculations_total = Counter(
        "gl_fe_calculations_total",
        "Total fugitive emission calculations performed",
        labelnames=["source_type", "method", "status"],
    )

    # 2. Cumulative emissions by source type and greenhouse gas
    fe_emissions_kg_co2e_total = Counter(
        "gl_fe_emissions_kg_co2e_total",
        "Cumulative fugitive emissions in kg CO2e by source type and gas",
        labelnames=["source_type", "gas"],
    )

    # 3. Source data lookups by emission factor source authority
    fe_source_lookups_total = Counter(
        "gl_fe_source_lookups_total",
        "Total fugitive emission source data and factor lookups by source",
        labelnames=["source"],
    )

    # 4. Emission factor selections by calculation method and source authority
    fe_factor_selections_total = Counter(
        "gl_fe_factor_selections_total",
        "Total emission factor selections by method and source",
        labelnames=["method", "source"],
    )

    # 5. LDAR survey events by survey type and completion status
    fe_ldar_surveys_total = Counter(
        "gl_fe_ldar_surveys_total",
        "Total LDAR survey events by survey type and status",
        labelnames=["survey_type", "status"],
    )

    # 6. Uncertainty analysis runs by method
    fe_uncertainty_runs_total = Counter(
        "gl_fe_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 7. Compliance checks by framework and status
    fe_compliance_checks_total = Counter(
        "gl_fe_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 8. Batch calculation jobs by completion status
    fe_batch_jobs_total = Counter(
        "gl_fe_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    fe_calculation_duration_seconds = Histogram(
        "gl_fe_calculation_duration_seconds",
        "Duration of fugitive emission calculation operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 10. Batch size histogram by calculation method
    fe_batch_size = Histogram(
        "gl_fe_batch_size",
        "Number of calculations in batch requests by method",
        labelnames=["method"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500),
    )

    # 11. Currently active (in-progress) calculations
    fe_active_calculations = Gauge(
        "gl_fe_active_calculations",
        "Number of currently active fugitive emission calculations",
    )

    # 12. Number of registered equipment components by component type
    fe_components_registered = Gauge(
        "gl_fe_components_registered",
        "Number of registered equipment components by component type",
        labelnames=["component_type"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    fe_calculations_total = None              # type: ignore[assignment]
    fe_emissions_kg_co2e_total = None         # type: ignore[assignment]
    fe_source_lookups_total = None            # type: ignore[assignment]
    fe_factor_selections_total = None         # type: ignore[assignment]
    fe_ldar_surveys_total = None              # type: ignore[assignment]
    fe_uncertainty_runs_total = None          # type: ignore[assignment]
    fe_compliance_checks_total = None         # type: ignore[assignment]
    fe_batch_jobs_total = None                # type: ignore[assignment]
    fe_calculation_duration_seconds = None    # type: ignore[assignment]
    fe_batch_size = None                      # type: ignore[assignment]
    fe_active_calculations = None             # type: ignore[assignment]
    fe_components_registered = None           # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    source_type: str,
    method: str,
    status: str,
) -> None:
    """Record a fugitive emission calculation event.

    Args:
        source_type: Type of fugitive emission source (wellhead, separator,
            dehydrator, pneumatic_controller_high, valve_gas, pump_seal,
            coal_mine_underground, wastewater_lagoon, etc.).
        method: Calculation method used (AVERAGE_EMISSION_FACTOR,
            SCREENING_RANGES, CORRELATION_EQUATION, ENGINEERING_ESTIMATE,
            DIRECT_MEASUREMENT).
        status: Completion status (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_calculations_total.labels(
        source_type=source_type,
        method=method,
        status=status,
    ).inc()


def record_emissions(
    source_type: str,
    gas: str,
    kg_co2e: float = 1.0,
) -> None:
    """Record cumulative emissions by source type and greenhouse gas.

    Args:
        source_type: Type of fugitive emission source (wellhead, separator,
            valve_gas, pump_seal, coal_mine_underground, etc.).
        gas: Greenhouse gas species (CH4, CO2, N2O, VOC).
        kg_co2e: Emission amount in kg CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_emissions_kg_co2e_total.labels(
        source_type=source_type,
        gas=gas,
    ).inc(kg_co2e)


def record_source_lookup(source: str) -> None:
    """Record a source data or emission factor lookup event.

    Args:
        source: Source authority queried (EPA, IPCC, DEFRA, EU_ETS,
            API, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_source_lookups_total.labels(
        source=source,
    ).inc()


def record_factor_selection(method: str, source: str) -> None:
    """Record an emission factor selection event.

    Args:
        method: Calculation method for which the factor was selected
            (AVERAGE_EMISSION_FACTOR, SCREENING_RANGES,
            CORRELATION_EQUATION, ENGINEERING_ESTIMATE,
            DIRECT_MEASUREMENT).
        source: Source authority of the selected factor
            (EPA, IPCC, DEFRA, EU_ETS, API, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_factor_selections_total.labels(
        method=method,
        source=source,
    ).inc()


def record_ldar_survey(survey_type: str, status: str) -> None:
    """Record an LDAR survey event.

    Args:
        survey_type: Type of leak detection survey performed (OGI,
            METHOD_21, AVO, HI_FLOW).
        status: Completion status of the survey (completed, failed,
            pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_ldar_surveys_total.labels(
        survey_type=survey_type,
        status=status,
    ).inc()


def record_uncertainty_run(method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        method: Uncertainty method applied
            (monte_carlo, analytical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a compliance check event.

    Args:
        framework: Regulatory framework checked
            (GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, EPA_40CFR98,
            UK_SECR, EU_ETS).
        status: Compliance status result
            (COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_CHECKED).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_compliance_checks_total.labels(
        framework=framework,
        status=status,
    ).inc()


def record_batch_job(status: str) -> None:
    """Record a batch calculation job event.

    Args:
        status: Completion status of the batch job
            (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_batch_jobs_total.labels(
        status=status,
    ).inc()


def observe_calculation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a calculation operation.

    Args:
        operation: Type of operation being measured
            (single_calculation, batch_calculation, factor_lookup,
            unit_conversion, gwp_application, uncertainty_analysis,
            provenance_hash, compliance_check, ldar_survey_processing,
            component_registration).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_calculation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_batch_size(method: str, size: int) -> None:
    """Record the size of a batch calculation request.

    Args:
        method: Calculation method used for the batch (AVERAGE_EMISSION_FACTOR,
            SCREENING_RANGES, CORRELATION_EQUATION, ENGINEERING_ESTIMATE,
            DIRECT_MEASUREMENT, or "mixed" for multi-method batches).
        size: Number of individual calculations in the batch.
            Buckets: 1, 5, 10, 25, 50, 100, 250, 500.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_batch_size.labels(
        method=method,
    ).observe(size)


def track_active_calculation(count: int) -> None:
    """Set the gauge for currently active in-progress calculations.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of calculations currently in progress. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_active_calculations.set(count)


def set_components_registered(component_type: str, count: int) -> None:
    """Set the gauge for number of registered equipment components by type.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        component_type: Equipment component type classification (valve,
            pump, compressor, pressure_relief_device, connector,
            open_ended_line, sampling_connection, flange, other).
        count: Number of components currently registered of this type.
            Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fe_components_registered.labels(
        component_type=component_type,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "fe_calculations_total",
    "fe_emissions_kg_co2e_total",
    "fe_source_lookups_total",
    "fe_factor_selections_total",
    "fe_ldar_surveys_total",
    "fe_uncertainty_runs_total",
    "fe_compliance_checks_total",
    "fe_batch_jobs_total",
    "fe_calculation_duration_seconds",
    "fe_batch_size",
    "fe_active_calculations",
    "fe_components_registered",
    # Helper functions
    "record_calculation",
    "record_emissions",
    "record_source_lookup",
    "record_factor_selection",
    "record_ldar_survey",
    "record_uncertainty_run",
    "record_compliance_check",
    "record_batch_job",
    "observe_calculation_duration",
    "observe_batch_size",
    "track_active_calculation",
    "set_components_registered",
]
