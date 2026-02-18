# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-003: Mobile Combustion Agent

12 Prometheus metrics for mobile combustion agent service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_mc_`` prefix (GreenLang Mobile Combustion)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_mc_calculations_total              (Counter,   labels: vehicle_type, method, status)
    2.  gl_mc_emissions_kg_co2e_total         (Counter,   labels: vehicle_type, fuel_type, gas)
    3.  gl_mc_vehicle_lookups_total            (Counter,   labels: source)
    4.  gl_mc_factor_selections_total          (Counter,   labels: method, source)
    5.  gl_mc_fleet_operations_total           (Counter,   labels: operation_type, vehicle_type)
    6.  gl_mc_uncertainty_runs_total           (Counter,   labels: method)
    7.  gl_mc_compliance_checks_total          (Counter,   labels: framework, status)
    8.  gl_mc_batch_jobs_total                 (Counter,   labels: status)
    9.  gl_mc_calculation_duration_seconds     (Histogram, labels: operation)
    10. gl_mc_batch_size                       (Histogram, labels: method)
    11. gl_mc_active_calculations              (Gauge)
    12. gl_mc_vehicles_registered              (Gauge,     labels: vehicle_type)

Label Values Reference:
    vehicle_type:
        PASSENGER_CAR_GASOLINE, PASSENGER_CAR_DIESEL, PASSENGER_CAR_HYBRID,
        PASSENGER_CAR_PHEV, LIGHT_DUTY_TRUCK_GASOLINE, LIGHT_DUTY_TRUCK_DIESEL,
        MEDIUM_DUTY_TRUCK_GASOLINE, MEDIUM_DUTY_TRUCK_DIESEL, HEAVY_DUTY_TRUCK,
        BUS_DIESEL, BUS_CNG, MOTORCYCLE, VAN_LCV, CONSTRUCTION_EQUIPMENT,
        AGRICULTURAL_EQUIPMENT, INDUSTRIAL_EQUIPMENT, MINING_EQUIPMENT,
        FORKLIFT, INLAND_VESSEL, COASTAL_VESSEL, OCEAN_VESSEL,
        CORPORATE_JET, HELICOPTER, TURBOPROP, DIESEL_LOCOMOTIVE.
    fuel_type:
        GASOLINE, DIESEL, BIODIESEL_B5, BIODIESEL_B20, BIODIESEL_B100,
        ETHANOL_E10, ETHANOL_E85, CNG, LNG, LPG, PROPANE,
        JET_FUEL_A, AVGAS, MARINE_DIESEL_OIL, HEAVY_FUEL_OIL,
        SUSTAINABLE_AVIATION_FUEL.
    method:
        FUEL_BASED, DISTANCE_BASED, SPEND_BASED.
    status:
        completed, failed, pending, running.
    gas:
        CO2, CH4, N2O.
    source:
        EPA, IPCC, DEFRA, EU_ETS, CUSTOM.
    operation_type:
        register, update, deregister, query.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, EPA_40CFR98, UK_SECR, EU_ETS.
    operation:
        single_calculation, batch_calculation, factor_lookup,
        distance_conversion, fuel_economy_conversion, gwp_application,
        uncertainty_analysis, provenance_hash, fleet_aggregation,
        compliance_check.

Example:
    >>> from greenlang.mobile_combustion.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     set_active_calculations,
    ... )
    >>> record_calculation("PASSENGER_CAR_GASOLINE", "FUEL_BASED", "completed")
    >>> record_emissions("PASSENGER_CAR_GASOLINE", "GASOLINE", "CO2", 1500.0)
    >>> set_active_calculations(5)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
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
        "prometheus_client not installed; mobile combustion metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by vehicle type, method, and completion status
    mc_calculations_total = Counter(
        "gl_mc_calculations_total",
        "Total mobile combustion emission calculations performed",
        labelnames=["vehicle_type", "method", "status"],
    )

    # 2. Cumulative emissions by vehicle type, fuel type, and greenhouse gas
    mc_emissions_kg_co2e_total = Counter(
        "gl_mc_emissions_kg_co2e_total",
        "Cumulative emissions in kg CO2e by vehicle type, fuel type, and gas",
        labelnames=["vehicle_type", "fuel_type", "gas"],
    )

    # 3. Vehicle data lookups by emission factor source authority
    mc_vehicle_lookups_total = Counter(
        "gl_mc_vehicle_lookups_total",
        "Total vehicle data and emission factor lookups by source",
        labelnames=["source"],
    )

    # 4. Emission factor selections by calculation method and source authority
    mc_factor_selections_total = Counter(
        "gl_mc_factor_selections_total",
        "Total emission factor selections by method and source",
        labelnames=["method", "source"],
    )

    # 5. Fleet management operations by operation type and vehicle type
    mc_fleet_operations_total = Counter(
        "gl_mc_fleet_operations_total",
        "Total fleet management operations by type and vehicle type",
        labelnames=["operation_type", "vehicle_type"],
    )

    # 6. Uncertainty analysis runs by method
    mc_uncertainty_runs_total = Counter(
        "gl_mc_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 7. Compliance checks by framework and status
    mc_compliance_checks_total = Counter(
        "gl_mc_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 8. Batch calculation jobs by completion status
    mc_batch_jobs_total = Counter(
        "gl_mc_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    mc_calculation_duration_seconds = Histogram(
        "gl_mc_calculation_duration_seconds",
        "Duration of mobile combustion calculation operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 10. Batch size histogram by calculation method
    mc_batch_size = Histogram(
        "gl_mc_batch_size",
        "Number of calculations in batch requests by method",
        labelnames=["method"],
        buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
    )

    # 11. Currently active (in-progress) calculations
    mc_active_calculations = Gauge(
        "gl_mc_active_calculations",
        "Number of currently active mobile combustion calculations",
    )

    # 12. Number of registered vehicles by vehicle type
    mc_vehicles_registered = Gauge(
        "gl_mc_vehicles_registered",
        "Number of registered fleet vehicles by vehicle type",
        labelnames=["vehicle_type"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    mc_calculations_total = None              # type: ignore[assignment]
    mc_emissions_kg_co2e_total = None         # type: ignore[assignment]
    mc_vehicle_lookups_total = None           # type: ignore[assignment]
    mc_factor_selections_total = None         # type: ignore[assignment]
    mc_fleet_operations_total = None          # type: ignore[assignment]
    mc_uncertainty_runs_total = None          # type: ignore[assignment]
    mc_compliance_checks_total = None         # type: ignore[assignment]
    mc_batch_jobs_total = None                # type: ignore[assignment]
    mc_calculation_duration_seconds = None    # type: ignore[assignment]
    mc_batch_size = None                      # type: ignore[assignment]
    mc_active_calculations = None             # type: ignore[assignment]
    mc_vehicles_registered = None             # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    vehicle_type: str,
    method: str,
    status: str,
) -> None:
    """Record a mobile combustion calculation event.

    Args:
        vehicle_type: Type of vehicle (PASSENGER_CAR_GASOLINE, etc.).
        method: Calculation method used (FUEL_BASED, DISTANCE_BASED,
            SPEND_BASED).
        status: Completion status (completed, failed, pending, running).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_calculations_total.labels(
        vehicle_type=vehicle_type,
        method=method,
        status=status,
    ).inc()


def record_emissions(
    vehicle_type: str,
    fuel_type: str,
    gas: str,
    kg_co2e: float = 1.0,
) -> None:
    """Record cumulative emissions by vehicle type, fuel type, and gas.

    Args:
        vehicle_type: Type of vehicle (PASSENGER_CAR_GASOLINE, etc.).
        fuel_type: Type of fuel consumed (GASOLINE, DIESEL, etc.).
        gas: Greenhouse gas species (CO2, CH4, N2O).
        kg_co2e: Emission amount in kg CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_emissions_kg_co2e_total.labels(
        vehicle_type=vehicle_type,
        fuel_type=fuel_type,
        gas=gas,
    ).inc(kg_co2e)


def record_vehicle_lookup(source: str) -> None:
    """Record a vehicle data or emission factor lookup event.

    Args:
        source: Source authority queried (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_vehicle_lookups_total.labels(
        source=source,
    ).inc()


def record_factor_selection(method: str, source: str) -> None:
    """Record an emission factor selection event.

    Args:
        method: Calculation method for which the factor was selected
            (FUEL_BASED, DISTANCE_BASED, SPEND_BASED).
        source: Source authority of the selected factor
            (EPA, IPCC, DEFRA, EU_ETS, CUSTOM).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_factor_selections_total.labels(
        method=method,
        source=source,
    ).inc()


def record_fleet_operation(
    operation_type: str,
    vehicle_type: str,
) -> None:
    """Record a fleet management operation.

    Args:
        operation_type: Type of fleet operation performed
            (register, update, deregister, query).
        vehicle_type: Type of vehicle involved in the operation.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_fleet_operations_total.labels(
        operation_type=operation_type,
        vehicle_type=vehicle_type,
    ).inc()


def record_uncertainty(method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        method: Uncertainty method applied
            (monte_carlo, analytical, tier_default).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a compliance check event.

    Args:
        framework: Regulatory framework checked
            (GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, etc.).
        status: Compliance status result
            (COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW, EXEMPT).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_compliance_checks_total.labels(
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
    mc_batch_jobs_total.labels(
        status=status,
    ).inc()


def observe_calculation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a calculation operation.

    Args:
        operation: Type of operation being measured
            (single_calculation, batch_calculation, factor_lookup,
            distance_conversion, fuel_economy_conversion, gwp_application,
            uncertainty_analysis, provenance_hash, fleet_aggregation,
            compliance_check).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_calculation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_batch_size(method: str, size: int) -> None:
    """Record the size of a batch calculation request.

    Args:
        method: Calculation method used for the batch (FUEL_BASED,
            DISTANCE_BASED, SPEND_BASED, or "mixed" for multi-method
            batches).
        size: Number of individual calculations in the batch.
            Buckets: 1, 5, 10, 50, 100, 500, 1000, 5000, 10000.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_batch_size.labels(
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
    mc_active_calculations.set(count)


def set_vehicles_registered(vehicle_type: str, count: int) -> None:
    """Set the gauge for number of registered fleet vehicles by type.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        vehicle_type: Vehicle type classification
            (PASSENGER_CAR_GASOLINE, HEAVY_DUTY_TRUCK, etc.).
        count: Number of vehicles currently registered of this type.
            Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mc_vehicles_registered.labels(
        vehicle_type=vehicle_type,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "mc_calculations_total",
    "mc_emissions_kg_co2e_total",
    "mc_vehicle_lookups_total",
    "mc_factor_selections_total",
    "mc_fleet_operations_total",
    "mc_uncertainty_runs_total",
    "mc_compliance_checks_total",
    "mc_batch_jobs_total",
    "mc_calculation_duration_seconds",
    "mc_batch_size",
    "mc_active_calculations",
    "mc_vehicles_registered",
    # Helper functions
    "record_calculation",
    "record_emissions",
    "record_vehicle_lookup",
    "record_factor_selection",
    "record_fleet_operation",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_vehicles_registered",
]
