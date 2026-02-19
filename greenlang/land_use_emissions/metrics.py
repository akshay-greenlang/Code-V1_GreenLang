# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-006: Land Use Emissions Agent

12 Prometheus metrics for land use emissions agent service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_lu_`` prefix (GreenLang Land Use) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_lu_calculations_total              (Counter,   labels: tier, method, land_category)
    2.  gl_lu_calculation_duration_seconds    (Histogram, labels: tier, method)
    3.  gl_lu_calculation_errors_total        (Counter,   labels: error_type)
    4.  gl_lu_emissions_tco2e_total           (Counter,   labels: gas, pool, land_category)
    5.  gl_lu_removals_tco2e_total            (Counter,   labels: pool, land_category)
    6.  gl_lu_transitions_total               (Counter,   labels: from_category, to_category)
    7.  gl_lu_carbon_stock_snapshots_total    (Counter,   labels: pool)
    8.  gl_lu_soc_assessments_total           (Counter,   labels: climate_zone, soil_type)
    9.  gl_lu_compliance_checks_total         (Counter,   labels: framework, status)
    10. gl_lu_uncertainty_runs_total          (Counter,   labels: method)
    11. gl_lu_batch_size                      (Histogram, labels: operation)
    12. gl_lu_active_parcels                  (Gauge,     labels: tenant_id)

Label Values Reference:
    tier:
        tier_1, tier_2, tier_3.
    method:
        stock_difference, gain_loss.
    land_category:
        forest_land, cropland, grassland, wetland, settlement, other_land.
    gas:
        CO2, CH4, N2O, CO.
    pool:
        above_ground_biomass, below_ground_biomass, dead_wood, litter,
        soil_organic_carbon.
    from_category / to_category:
        forest_land, cropland, grassland, wetland, settlement, other_land.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, unknown_error.
    climate_zone:
        tropical_wet, tropical_moist, tropical_dry, tropical_montane,
        warm_temperate_moist, warm_temperate_dry, cool_temperate_moist,
        cool_temperate_dry, boreal_moist, boreal_dry, polar_moist,
        polar_dry.
    soil_type:
        high_activity_clay, low_activity_clay, sandy, spodic, volcanic,
        wetland_organic, other.
    framework:
        GHG_PROTOCOL, IPCC_2006, CSRD_ESRS_E1, EU_LULUCF, UK_SECR,
        UNFCCC.
    status:
        compliant, non_compliant, partial, not_assessed.
    operation:
        single_calculation, batch_calculation, soc_assessment,
        uncertainty_analysis, compliance_check, aggregation,
        parcel_registration, transition_recording, snapshot_recording,
        provenance_hash.

Example:
    >>> from greenlang.land_use_emissions.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     track_active_parcels,
    ... )
    >>> record_calculation("tier_1", "stock_difference", "forest_land")
    >>> record_emissions("CO2", "above_ground_biomass", "forest_land", 150.5)
    >>> track_active_parcels("tenant_001", 42)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
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
        "prometheus_client not installed; land use emissions metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by tier, method, and land category
    lu_calculations_total = Counter(
        "gl_lu_calculations_total",
        "Total land use emission calculations performed",
        labelnames=["tier", "method", "land_category"],
    )

    # 2. Calculation duration histogram by tier and method
    lu_calculation_duration_seconds = Histogram(
        "gl_lu_calculation_duration_seconds",
        "Duration of land use emission calculation operations in seconds",
        labelnames=["tier", "method"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 3. Calculation errors by error type
    lu_calculation_errors_total = Counter(
        "gl_lu_calculation_errors_total",
        "Total land use emission calculation errors by error type",
        labelnames=["error_type"],
    )

    # 4. Cumulative emissions by gas, pool, and land category
    lu_emissions_tco2e_total = Counter(
        "gl_lu_emissions_tco2e_total",
        "Cumulative land use emissions in tCO2e by gas, pool, and category",
        labelnames=["gas", "pool", "land_category"],
    )

    # 5. Cumulative removals by pool and land category
    lu_removals_tco2e_total = Counter(
        "gl_lu_removals_tco2e_total",
        "Cumulative land use carbon removals in tCO2e by pool and category",
        labelnames=["pool", "land_category"],
    )

    # 6. Land-use transitions by from/to category
    lu_transitions_total = Counter(
        "gl_lu_transitions_total",
        "Total land-use transition events by from and to category",
        labelnames=["from_category", "to_category"],
    )

    # 7. Carbon stock snapshots recorded by pool
    lu_carbon_stock_snapshots_total = Counter(
        "gl_lu_carbon_stock_snapshots_total",
        "Total carbon stock snapshot recordings by pool",
        labelnames=["pool"],
    )

    # 8. SOC assessments by climate zone and soil type
    lu_soc_assessments_total = Counter(
        "gl_lu_soc_assessments_total",
        "Total soil organic carbon assessments by climate zone and soil type",
        labelnames=["climate_zone", "soil_type"],
    )

    # 9. Compliance checks by framework and status
    lu_compliance_checks_total = Counter(
        "gl_lu_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 10. Uncertainty analysis runs by method
    lu_uncertainty_runs_total = Counter(
        "gl_lu_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 11. Batch size histogram by operation type
    lu_batch_size = Histogram(
        "gl_lu_batch_size",
        "Number of items in batch operations by operation type",
        labelnames=["operation"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000),
    )

    # 12. Currently active parcels by tenant
    lu_active_parcels = Gauge(
        "gl_lu_active_parcels",
        "Number of currently active land parcels by tenant",
        labelnames=["tenant_id"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    lu_calculations_total = None              # type: ignore[assignment]
    lu_calculation_duration_seconds = None     # type: ignore[assignment]
    lu_calculation_errors_total = None         # type: ignore[assignment]
    lu_emissions_tco2e_total = None            # type: ignore[assignment]
    lu_removals_tco2e_total = None             # type: ignore[assignment]
    lu_transitions_total = None                # type: ignore[assignment]
    lu_carbon_stock_snapshots_total = None     # type: ignore[assignment]
    lu_soc_assessments_total = None            # type: ignore[assignment]
    lu_compliance_checks_total = None          # type: ignore[assignment]
    lu_uncertainty_runs_total = None            # type: ignore[assignment]
    lu_batch_size = None                       # type: ignore[assignment]
    lu_active_parcels = None                   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    tier: str,
    method: str,
    land_category: str,
) -> None:
    """Record a land use emission calculation event.

    Args:
        tier: IPCC calculation tier used (tier_1, tier_2, tier_3).
        method: Calculation method used (stock_difference, gain_loss).
        land_category: IPCC land category (forest_land, cropland,
            grassland, wetland, settlement, other_land).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_calculations_total.labels(
        tier=tier,
        method=method,
        land_category=land_category,
    ).inc()


def observe_calculation_duration(
    tier: str,
    method: str,
    seconds: float,
) -> None:
    """Record the duration of a calculation operation.

    Args:
        tier: IPCC calculation tier used (tier_1, tier_2, tier_3).
        method: Calculation method used (stock_difference, gain_loss).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_calculation_duration_seconds.labels(
        tier=tier,
        method=method,
    ).observe(seconds)


def record_calculation_error(error_type: str) -> None:
    """Record a calculation error event.

    Args:
        error_type: Type of error encountered (validation_error,
            calculation_error, database_error, configuration_error,
            timeout_error, unknown_error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_calculation_errors_total.labels(
        error_type=error_type,
    ).inc()


def record_emissions(
    gas: str,
    pool: str,
    land_category: str,
    tco2e: float = 1.0,
) -> None:
    """Record cumulative emissions by gas, pool, and land category.

    Args:
        gas: Greenhouse gas species (CO2, CH4, N2O, CO).
        pool: Carbon pool (above_ground_biomass, below_ground_biomass,
            dead_wood, litter, soil_organic_carbon).
        land_category: IPCC land category (forest_land, cropland,
            grassland, wetland, settlement, other_land).
        tco2e: Emission amount in tonnes CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_emissions_tco2e_total.labels(
        gas=gas,
        pool=pool,
        land_category=land_category,
    ).inc(tco2e)


def record_removals(
    pool: str,
    land_category: str,
    tco2e: float = 1.0,
) -> None:
    """Record cumulative carbon removals by pool and land category.

    Args:
        pool: Carbon pool (above_ground_biomass, below_ground_biomass,
            dead_wood, litter, soil_organic_carbon).
        land_category: IPCC land category (forest_land, cropland,
            grassland, wetland, settlement, other_land).
        tco2e: Removal amount in tonnes CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_removals_tco2e_total.labels(
        pool=pool,
        land_category=land_category,
    ).inc(tco2e)


def record_transition(
    from_category: str,
    to_category: str,
) -> None:
    """Record a land-use transition event.

    Args:
        from_category: IPCC land category before transition.
        to_category: IPCC land category after transition.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_transitions_total.labels(
        from_category=from_category,
        to_category=to_category,
    ).inc()


def record_carbon_stock_snapshot(pool: str) -> None:
    """Record a carbon stock snapshot event.

    Args:
        pool: Carbon pool of the snapshot (above_ground_biomass,
            below_ground_biomass, dead_wood, litter,
            soil_organic_carbon).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_carbon_stock_snapshots_total.labels(
        pool=pool,
    ).inc()


def record_soc_assessment(
    climate_zone: str,
    soil_type: str,
) -> None:
    """Record a soil organic carbon assessment event.

    Args:
        climate_zone: IPCC climate zone of the assessment.
        soil_type: IPCC soil type of the assessment.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_soc_assessments_total.labels(
        climate_zone=climate_zone,
        soil_type=soil_type,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a compliance check event.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            IPCC_2006, CSRD_ESRS_E1, EU_LULUCF, UK_SECR, UNFCCC).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_compliance_checks_total.labels(
        framework=framework,
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
    lu_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def observe_batch_size(operation: str, size: int) -> None:
    """Record the size of a batch operation.

    Args:
        operation: Type of batch operation (single_calculation,
            batch_calculation, soc_assessment, uncertainty_analysis,
            compliance_check, aggregation, parcel_registration,
            transition_recording, snapshot_recording,
            provenance_hash).
        size: Number of individual items in the batch.
            Buckets: 1, 5, 10, 25, 50, 100, 250, 500, 1000,
            5000, 10000.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_batch_size.labels(
        operation=operation,
    ).observe(size)


def track_active_parcels(tenant_id: str, count: int) -> None:
    """Set the gauge for currently active land parcels by tenant.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        tenant_id: Tenant identifier for the parcel count.
        count: Number of active parcels for the tenant. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    lu_active_parcels.labels(
        tenant_id=tenant_id,
    ).set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "lu_calculations_total",
    "lu_calculation_duration_seconds",
    "lu_calculation_errors_total",
    "lu_emissions_tco2e_total",
    "lu_removals_tco2e_total",
    "lu_transitions_total",
    "lu_carbon_stock_snapshots_total",
    "lu_soc_assessments_total",
    "lu_compliance_checks_total",
    "lu_uncertainty_runs_total",
    "lu_batch_size",
    "lu_active_parcels",
    # Helper functions
    "record_calculation",
    "observe_calculation_duration",
    "record_calculation_error",
    "record_emissions",
    "record_removals",
    "record_transition",
    "record_carbon_stock_snapshot",
    "record_soc_assessment",
    "record_compliance_check",
    "record_uncertainty_run",
    "observe_batch_size",
    "track_active_parcels",
]
