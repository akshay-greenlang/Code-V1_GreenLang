# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-001: Stationary Combustion Agent

12 Prometheus metrics for stationary combustion agent service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_sc_`` prefix (GreenLang Stationary Combustion)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_sc_calculations_total              (Counter,   labels: fuel_type, tier, status)
    2.  gl_sc_emissions_kg_co2e_total         (Counter,   labels: fuel_type, gas)
    3.  gl_sc_fuel_lookups_total              (Counter,   labels: source)
    4.  gl_sc_factor_selections_total         (Counter,   labels: tier, source)
    5.  gl_sc_equipment_profiles_total        (Counter,   labels: equipment_type, action)
    6.  gl_sc_uncertainty_runs_total          (Counter,   labels: tier, method)
    7.  gl_sc_audit_entries_total             (Counter,   labels: step_name)
    8.  gl_sc_batch_jobs_total                (Counter,   labels: status)
    9.  gl_sc_calculation_duration_seconds    (Histogram, labels: operation)
    10. gl_sc_batch_size                      (Histogram, labels: fuel_type)
    11. gl_sc_active_calculations             (Gauge)
    12. gl_sc_emission_factors_loaded         (Gauge,     labels: source)

Label Values Reference:
    fuel_type:
        natural_gas, diesel, gasoline, lpg, propane, kerosene, jet_fuel,
        fuel_oil_2, fuel_oil_6, coal_bituminous, coal_anthracite,
        coal_sub_bituminous, coal_lignite, petroleum_coke, wood,
        biomass_solid, biomass_liquid, biogas, landfill_gas,
        coke_oven_gas, blast_furnace_gas, peat, waste_oil, msw.
    tier:
        TIER_1, TIER_2, TIER_3.
    status:
        completed, failed, pending, running.
    gas:
        CO2, CH4, N2O.
    source:
        EPA, IPCC, DEFRA, EU_ETS, CUSTOM.
    equipment_type:
        boiler_fire_tube, boiler_water_tube, furnace, process_heater,
        gas_turbine_simple, gas_turbine_combined, reciprocating_engine,
        kiln, oven, dryer, flare, incinerator, thermal_oxidizer.
    action:
        register, update, delete.
    method:
        monte_carlo, analytical, tier_default.
    operation:
        single_calculation, batch_calculation, factor_lookup,
        unit_conversion, gwp_application, uncertainty_analysis,
        provenance_hash, audit_generation.
    step_name:
        validate_input, lookup_fuel_properties, lookup_emission_factor,
        convert_units, calculate_energy, apply_oxidation_factor,
        calculate_gas_emissions, apply_gwp, aggregate_totals,
        generate_provenance, generate_audit.

Example:
    >>> from greenlang.stationary_combustion.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     set_active_calculations,
    ... )
    >>> record_calculation("natural_gas", "TIER_1", "completed")
    >>> record_emissions("natural_gas", "CO2", 1500.0)
    >>> set_active_calculations(5)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
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
        "prometheus_client not installed; stationary combustion metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by fuel type, tier, and completion status
    sc_calculations_total = Counter(
        "gl_sc_calculations_total",
        "Total stationary combustion emission calculations performed",
        labelnames=["fuel_type", "tier", "status"],
    )

    # 2. Cumulative emissions by fuel type and greenhouse gas
    sc_emissions_kg_co2e_total = Counter(
        "gl_sc_emissions_kg_co2e_total",
        "Cumulative emissions in kg CO2e by fuel type and gas",
        labelnames=["fuel_type", "gas"],
    )

    # 3. Fuel property lookups by emission factor source authority
    sc_fuel_lookups_total = Counter(
        "gl_sc_fuel_lookups_total",
        "Total fuel property and heating value lookups by source",
        labelnames=["source"],
    )

    # 4. Emission factor selections by tier and source authority
    sc_factor_selections_total = Counter(
        "gl_sc_factor_selections_total",
        "Total emission factor selections by tier and source",
        labelnames=["tier", "source"],
    )

    # 5. Equipment profile operations by type and action
    sc_equipment_profiles_total = Counter(
        "gl_sc_equipment_profiles_total",
        "Total equipment profile operations by type and action",
        labelnames=["equipment_type", "action"],
    )

    # 6. Uncertainty analysis runs by tier and method
    sc_uncertainty_runs_total = Counter(
        "gl_sc_uncertainty_runs_total",
        "Total uncertainty quantification runs by tier and method",
        labelnames=["tier", "method"],
    )

    # 7. Audit trail entries created by step name
    sc_audit_entries_total = Counter(
        "gl_sc_audit_entries_total",
        "Total audit trail entries created by step name",
        labelnames=["step_name"],
    )

    # 8. Batch calculation jobs by completion status
    sc_batch_jobs_total = Counter(
        "gl_sc_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    sc_calculation_duration_seconds = Histogram(
        "gl_sc_calculation_duration_seconds",
        "Duration of stationary combustion calculation operations in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 10. Batch size histogram by fuel type
    sc_batch_size = Histogram(
        "gl_sc_batch_size",
        "Number of calculations in batch requests by fuel type",
        labelnames=["fuel_type"],
        buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
    )

    # 11. Currently active (in-progress) calculations
    sc_active_calculations = Gauge(
        "gl_sc_active_calculations",
        "Number of currently active stationary combustion calculations",
    )

    # 12. Number of emission factors loaded by source authority
    sc_emission_factors_loaded = Gauge(
        "gl_sc_emission_factors_loaded",
        "Number of emission factors currently loaded by source",
        labelnames=["source"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    sc_calculations_total = None              # type: ignore[assignment]
    sc_emissions_kg_co2e_total = None         # type: ignore[assignment]
    sc_fuel_lookups_total = None              # type: ignore[assignment]
    sc_factor_selections_total = None         # type: ignore[assignment]
    sc_equipment_profiles_total = None        # type: ignore[assignment]
    sc_uncertainty_runs_total = None          # type: ignore[assignment]
    sc_audit_entries_total = None             # type: ignore[assignment]
    sc_batch_jobs_total = None                # type: ignore[assignment]
    sc_calculation_duration_seconds = None    # type: ignore[assignment]
    sc_batch_size = None                      # type: ignore[assignment]
    sc_active_calculations = None             # type: ignore[assignment]
    sc_emission_factors_loaded = None         # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(fuel_type: str, tier: str, status: str) -> None:
    """Record a stationary combustion calculation event.

    Args:
        fuel_type: Type of fuel combusted (natural_gas, diesel, etc.).
        tier: Calculation tier used (TIER_1, TIER_2, TIER_3).
        status: Completion status (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_calculations_total.labels(
        fuel_type=fuel_type,
        tier=tier,
        status=status,
    ).inc()


def record_emissions(fuel_type: str, gas: str, kg_co2e: float = 1.0) -> None:
    """Record cumulative emissions by fuel type and greenhouse gas.

    Args:
        fuel_type: Type of fuel combusted (natural_gas, diesel, etc.).
        gas: Greenhouse gas species (CO2, CH4, N2O).
        kg_co2e: Emission amount in kg CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_emissions_kg_co2e_total.labels(
        fuel_type=fuel_type,
        gas=gas,
    ).inc(kg_co2e)


def record_fuel_lookup(source: str) -> None:
    """Record a fuel property or heating value lookup event.

    Args:
        source: Source authority queried (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_fuel_lookups_total.labels(
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
    sc_factor_selections_total.labels(
        tier=tier,
        source=source,
    ).inc()


def record_equipment(equipment_type: str, action: str) -> None:
    """Record an equipment profile operation.

    Args:
        equipment_type: Type of combustion equipment
            (boiler_fire_tube, furnace, gas_turbine_simple, etc.).
        action: Operation performed (register, update, delete).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_equipment_profiles_total.labels(
        equipment_type=equipment_type,
        action=action,
    ).inc()


def record_uncertainty(tier: str, method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        tier: Calculation tier used (TIER_1, TIER_2, TIER_3).
        method: Uncertainty method applied
            (monte_carlo, analytical, tier_default).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_uncertainty_runs_total.labels(
        tier=tier,
        method=method,
    ).inc()


def record_audit(step_name: str) -> None:
    """Record an audit trail entry creation.

    Args:
        step_name: Name of the calculation step being audited
            (validate_input, lookup_emission_factor, calculate_energy,
            apply_gwp, aggregate_totals, generate_provenance, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_audit_entries_total.labels(
        step_name=step_name,
    ).inc()


def record_batch(status: str) -> None:
    """Record a batch calculation job event.

    Args:
        status: Completion status of the batch job
            (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_batch_jobs_total.labels(
        status=status,
    ).inc()


def observe_calculation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a calculation operation.

    Args:
        operation: Type of operation being measured
            (single_calculation, batch_calculation, factor_lookup,
            unit_conversion, gwp_application, uncertainty_analysis,
            provenance_hash, audit_generation).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_calculation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_batch_size(fuel_type: str, size: int) -> None:
    """Record the size of a batch calculation request.

    Args:
        fuel_type: Primary fuel type in the batch (or "mixed" for
            multi-fuel batches).
        size: Number of individual calculations in the batch.
            Buckets: 1, 5, 10, 50, 100, 500, 1000, 5000, 10000.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_batch_size.labels(
        fuel_type=fuel_type,
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
    sc_active_calculations.set(count)


def set_factors_loaded(source: str, count: int) -> None:
    """Set the gauge for number of emission factors loaded by source.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        source: Source authority of the loaded factors
            (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
        count: Number of emission factors currently loaded from this
            source. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sc_emission_factors_loaded.labels(
        source=source,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "sc_calculations_total",
    "sc_emissions_kg_co2e_total",
    "sc_fuel_lookups_total",
    "sc_factor_selections_total",
    "sc_equipment_profiles_total",
    "sc_uncertainty_runs_total",
    "sc_audit_entries_total",
    "sc_batch_jobs_total",
    "sc_calculation_duration_seconds",
    "sc_batch_size",
    "sc_active_calculations",
    "sc_emission_factors_loaded",
    # Helper functions
    "record_calculation",
    "record_emissions",
    "record_fuel_lookup",
    "record_factor_selection",
    "record_equipment",
    "record_uncertainty",
    "record_audit",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_factors_loaded",
]
