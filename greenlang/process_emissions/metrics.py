# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-004: Process Emissions Agent

12 Prometheus metrics for process emissions agent service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_pe_`` prefix (GreenLang Process Emissions)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_pe_calculations_total              (Counter,   labels: process_type, method, status)
    2.  gl_pe_emissions_kg_co2e_total         (Counter,   labels: process_type, gas)
    3.  gl_pe_process_lookups_total            (Counter,   labels: source)
    4.  gl_pe_factor_selections_total          (Counter,   labels: tier, source)
    5.  gl_pe_material_operations_total        (Counter,   labels: operation_type, material_type)
    6.  gl_pe_uncertainty_runs_total           (Counter,   labels: method)
    7.  gl_pe_compliance_checks_total          (Counter,   labels: framework, status)
    8.  gl_pe_batch_jobs_total                 (Counter,   labels: status)
    9.  gl_pe_calculation_duration_seconds     (Histogram, labels: operation)
    10. gl_pe_batch_size                       (Histogram, labels: method)
    11. gl_pe_active_calculations              (Gauge)
    12. gl_pe_process_units_registered         (Gauge,     labels: process_type)

Label Values Reference:
    process_type:
        cement_production, lime_production, glass_production, ceramics,
        soda_ash, ammonia_production, nitric_acid, adipic_acid,
        carbide_production, petrochemical, hydrogen_production,
        phosphoric_acid, titanium_dioxide, iron_steel, aluminum_smelting,
        ferroalloy, lead_production, zinc_production, copper_smelting,
        semiconductor, flat_panel_display, photovoltaic, pulp_production,
        paper_production, food_beverage.
    method:
        EMISSION_FACTOR, MASS_BALANCE, STOICHIOMETRIC, DIRECT_MEASUREMENT.
    status:
        completed, failed, pending, running.
    gas:
        CO2, CH4, N2O, CF4, C2F6, SF6, NF3, HFC.
    source:
        EPA, IPCC, DEFRA, EU_ETS, CUSTOM.
    tier:
        TIER_1, TIER_2, TIER_3.
    operation_type:
        register, update, delete, consume, transform, query.
    material_type:
        calcium_carbonate, magnesium_carbonate, iron_carbonate,
        calcium_oxide, magnesium_oxide, alumina, iron_ore, bauxite,
        coke, natural_gas_feedstock, naphtha, clinker, calcium_hydroxide,
        calcium_fluoride, sodium_carbonate.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, EPA_40CFR98, UK_SECR, EU_ETS.
    operation:
        single_calculation, batch_calculation, factor_lookup,
        unit_conversion, gwp_application, uncertainty_analysis,
        provenance_hash, compliance_check, material_balance,
        stoichiometric_calc.

Example:
    >>> from greenlang.process_emissions.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     set_active_calculations,
    ... )
    >>> record_calculation("cement_production", "EMISSION_FACTOR", "completed")
    >>> record_emissions("cement_production", "CO2", 45000.0)
    >>> set_active_calculations(3)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
        "prometheus_client not installed; process emissions metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by process type, method, and completion status
    pe_calculations_total = Counter(
        "gl_pe_calculations_total",
        "Total process emission calculations performed",
        labelnames=["process_type", "method", "status"],
    )

    # 2. Cumulative emissions by process type and greenhouse gas
    pe_emissions_kg_co2e_total = Counter(
        "gl_pe_emissions_kg_co2e_total",
        "Cumulative emissions in kg CO2e by process type and gas",
        labelnames=["process_type", "gas"],
    )

    # 3. Process data lookups by emission factor source authority
    pe_process_lookups_total = Counter(
        "gl_pe_process_lookups_total",
        "Total process data and emission factor lookups by source",
        labelnames=["source"],
    )

    # 4. Emission factor selections by tier and source authority
    pe_factor_selections_total = Counter(
        "gl_pe_factor_selections_total",
        "Total emission factor selections by tier and source",
        labelnames=["tier", "source"],
    )

    # 5. Material operations by operation type and material type
    pe_material_operations_total = Counter(
        "gl_pe_material_operations_total",
        "Total material operations by operation type and material type",
        labelnames=["operation_type", "material_type"],
    )

    # 6. Uncertainty analysis runs by method
    pe_uncertainty_runs_total = Counter(
        "gl_pe_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 7. Compliance checks by framework and status
    pe_compliance_checks_total = Counter(
        "gl_pe_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 8. Batch calculation jobs by completion status
    pe_batch_jobs_total = Counter(
        "gl_pe_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    pe_calculation_duration_seconds = Histogram(
        "gl_pe_calculation_duration_seconds",
        "Duration of process emission calculation operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 10. Batch size histogram by calculation method
    pe_batch_size = Histogram(
        "gl_pe_batch_size",
        "Number of calculations in batch requests by method",
        labelnames=["method"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500),
    )

    # 11. Currently active (in-progress) calculations
    pe_active_calculations = Gauge(
        "gl_pe_active_calculations",
        "Number of currently active process emission calculations",
    )

    # 12. Number of registered process units by process type
    pe_process_units_registered = Gauge(
        "gl_pe_process_units_registered",
        "Number of registered process units by process type",
        labelnames=["process_type"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    pe_calculations_total = None              # type: ignore[assignment]
    pe_emissions_kg_co2e_total = None         # type: ignore[assignment]
    pe_process_lookups_total = None           # type: ignore[assignment]
    pe_factor_selections_total = None         # type: ignore[assignment]
    pe_material_operations_total = None       # type: ignore[assignment]
    pe_uncertainty_runs_total = None          # type: ignore[assignment]
    pe_compliance_checks_total = None         # type: ignore[assignment]
    pe_batch_jobs_total = None                # type: ignore[assignment]
    pe_calculation_duration_seconds = None    # type: ignore[assignment]
    pe_batch_size = None                      # type: ignore[assignment]
    pe_active_calculations = None             # type: ignore[assignment]
    pe_process_units_registered = None        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    process_type: str,
    method: str,
    status: str,
) -> None:
    """Record a process emission calculation event.

    Args:
        process_type: Type of industrial process (cement_production,
            iron_steel, aluminum_smelting, nitric_acid, etc.).
        method: Calculation method used (EMISSION_FACTOR, MASS_BALANCE,
            STOICHIOMETRIC, DIRECT_MEASUREMENT).
        status: Completion status (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_calculations_total.labels(
        process_type=process_type,
        method=method,
        status=status,
    ).inc()


def record_emissions(
    process_type: str,
    gas: str,
    kg_co2e: float = 1.0,
) -> None:
    """Record cumulative emissions by process type and greenhouse gas.

    Args:
        process_type: Type of industrial process (cement_production,
            iron_steel, aluminum_smelting, nitric_acid, etc.).
        gas: Greenhouse gas species (CO2, CH4, N2O, CF4, C2F6, SF6,
            NF3, HFC).
        kg_co2e: Emission amount in kg CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_emissions_kg_co2e_total.labels(
        process_type=process_type,
        gas=gas,
    ).inc(kg_co2e)


def record_process_lookup(source: str) -> None:
    """Record a process data or emission factor lookup event.

    Args:
        source: Source authority queried (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_process_lookups_total.labels(
        source=source,
    ).inc()


def record_factor_selection(tier: str, source: str) -> None:
    """Record an emission factor selection event.

    Args:
        tier: Calculation tier for which the factor was selected
            (TIER_1, TIER_2, TIER_3).
        source: Source authority of the selected factor
            (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_factor_selections_total.labels(
        tier=tier,
        source=source,
    ).inc()


def record_material_operation(
    operation_type: str,
    material_type: str,
) -> None:
    """Record a material operation event.

    Args:
        operation_type: Type of material operation performed
            (register, update, delete, consume, transform, query).
        material_type: Type of raw material involved
            (calcium_carbonate, iron_ore, alumina, coke, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_material_operations_total.labels(
        operation_type=operation_type,
        material_type=material_type,
    ).inc()


def record_uncertainty(method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        method: Uncertainty method applied
            (monte_carlo, analytical, tier_default).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_uncertainty_runs_total.labels(
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
    pe_compliance_checks_total.labels(
        framework=framework,
        status=status,
    ).inc()


def record_batch(status: str) -> None:
    """Record a batch calculation job event.

    Args:
        status: Completion status of the batch job
            (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_batch_jobs_total.labels(
        status=status,
    ).inc()


def observe_calculation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a calculation operation.

    Args:
        operation: Type of operation being measured
            (single_calculation, batch_calculation, factor_lookup,
            unit_conversion, gwp_application, uncertainty_analysis,
            provenance_hash, compliance_check, material_balance,
            stoichiometric_calc).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_calculation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_batch_size(method: str, size: int) -> None:
    """Record the size of a batch calculation request.

    Args:
        method: Calculation method used for the batch (EMISSION_FACTOR,
            MASS_BALANCE, STOICHIOMETRIC, DIRECT_MEASUREMENT, or "mixed"
            for multi-method batches).
        size: Number of individual calculations in the batch.
            Buckets: 1, 5, 10, 25, 50, 100, 250, 500.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_batch_size.labels(
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
    pe_active_calculations.set(count)


def set_process_units_registered(process_type: str, count: int) -> None:
    """Set the gauge for number of registered process units by type.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        process_type: Industrial process type classification
            (cement_production, iron_steel, aluminum_smelting, etc.).
        count: Number of process units currently registered of this type.
            Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pe_process_units_registered.labels(
        process_type=process_type,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "pe_calculations_total",
    "pe_emissions_kg_co2e_total",
    "pe_process_lookups_total",
    "pe_factor_selections_total",
    "pe_material_operations_total",
    "pe_uncertainty_runs_total",
    "pe_compliance_checks_total",
    "pe_batch_jobs_total",
    "pe_calculation_duration_seconds",
    "pe_batch_size",
    "pe_active_calculations",
    "pe_process_units_registered",
    # Helper functions
    "record_calculation",
    "record_emissions",
    "record_process_lookup",
    "record_factor_selection",
    "record_material_operation",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_process_units_registered",
]
