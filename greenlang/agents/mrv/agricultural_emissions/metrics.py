# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-008: Agricultural Emissions Agent

12 Prometheus metrics for agricultural emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_ag_`` prefix (GreenLang Agricultural) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_ag_calculations_total              (Counter,   labels: emission_source, calculation_method, animal_type_or_crop)
    2.  gl_ag_calculation_duration_seconds    (Histogram, labels: emission_source, calculation_method)
    3.  gl_ag_calculation_errors_total        (Counter,   labels: error_type)
    4.  gl_ag_emissions_tco2e_total           (Counter,   labels: gas, emission_source)
    5.  gl_ag_enteric_calculations_total      (Counter,   labels: animal_type, calculation_method)
    6.  gl_ag_manure_calculations_total       (Counter,   labels: animal_type, manure_system)
    7.  gl_ag_cropland_calculations_total     (Counter,   labels: input_type)
    8.  gl_ag_rice_calculations_total         (Counter,   labels: water_regime)
    9.  gl_ag_field_burning_calculations_total (Counter,  labels: crop_type)
    10. gl_ag_compliance_checks_total         (Counter,   labels: framework, status)
    11. gl_ag_uncertainty_runs_total          (Counter,   labels: method)
    12. gl_ag_active_farms                    (Gauge,     labels: tenant_id)

Label Values Reference:
    emission_source:
        enteric_fermentation, manure_management, rice_cultivation,
        agricultural_soils, field_burning, liming, urea_application,
        crop_residue, prescribed_burning, savanna_burning.
    calculation_method:
        ipcc_tier1, ipcc_tier2, ipcc_tier3, country_specific,
        direct_measurement, emission_factor, mass_balance.
    animal_type_or_crop:
        dairy_cattle, non_dairy_cattle, buffalo, sheep, goats,
        swine, poultry, horses, mules_asses, camels, llamas,
        rice, wheat, maize, sugarcane, cotton, other_crop.
    gas:
        CO2, CH4, N2O, biogenic_CO2.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, unknown_error.
    animal_type:
        dairy_cattle, non_dairy_cattle, buffalo, sheep, goats,
        swine, poultry, horses, mules_asses, camels, llamas.
    manure_system:
        pasture_range, daily_spread, solid_storage, dry_lot,
        liquid_slurry, uncovered_anaerobic_lagoon, pit_storage,
        anaerobic_digester, burned_for_fuel, deep_bedding,
        composting, aerobic_treatment.
    input_type:
        fertilizer, liming, urea, residue.
    water_regime:
        continuously_flooded, intermittent_single, intermittent_multiple,
        rainfed_regular, rainfed_drought, deep_water, upland.
    crop_type:
        rice, wheat, maize, sugarcane, cotton, barley, oats,
        sorghum, millet, other.
    framework:
        GHG_PROTOCOL, IPCC_2006, IPCC_2019, CSRD_ESRS_E1,
        SBTi_FLAG, EPA_GHGRP, UNFCCC.
    status:
        compliant, non_compliant, partial, not_assessed.
    method:
        monte_carlo, analytical, error_propagation,
        ipcc_default_uncertainty.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     track_active_farms,
    ... )
    >>> record_calculation("enteric_fermentation", "ipcc_tier1", "dairy_cattle")
    >>> record_emissions("CH4", "enteric_fermentation", 42.5)
    >>> track_active_farms("tenant_001", 15)

    Using the MetricsCollector singleton:

    >>> from greenlang.agents.mrv.agricultural_emissions.metrics import MetricsCollector
    >>> collector = MetricsCollector()
    >>> collector.record_calculation("manure_management", "ipcc_tier2", "swine")
    >>> collector.record_emissions("N2O", "manure_management", 8.3)
    >>> collector.track_active_farms("tenant_002", 9)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading

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
        "prometheus_client not installed; agricultural emissions metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by emission source, calculation method,
    #    and animal type or crop
    ag_calculations_total = Counter(
        "gl_ag_calculations_total",
        "Total agricultural emission calculations performed",
        labelnames=["emission_source", "calculation_method", "animal_type_or_crop"],
    )

    # 2. Calculation duration histogram by emission source and calculation method
    ag_calculation_duration_seconds = Histogram(
        "gl_ag_calculation_duration_seconds",
        "Duration of agricultural emission calculation operations in seconds",
        labelnames=["emission_source", "calculation_method"],
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 3. Calculation errors by error type
    ag_calculation_errors_total = Counter(
        "gl_ag_calculation_errors_total",
        "Total agricultural emission calculation errors by error type",
        labelnames=["error_type"],
    )

    # 4. Cumulative emissions by gas and emission source
    ag_emissions_tco2e_total = Counter(
        "gl_ag_emissions_tco2e_total",
        "Cumulative agricultural emissions in tCO2e by gas and emission source",
        labelnames=["gas", "emission_source"],
    )

    # 5. Enteric fermentation calculations by animal type and calculation method
    ag_enteric_calculations_total = Counter(
        "gl_ag_enteric_calculations_total",
        "Total enteric fermentation emission calculations by animal type "
        "and calculation method",
        labelnames=["animal_type", "calculation_method"],
    )

    # 6. Manure management calculations by animal type and manure system
    ag_manure_calculations_total = Counter(
        "gl_ag_manure_calculations_total",
        "Total manure management emission calculations by animal type "
        "and manure management system",
        labelnames=["animal_type", "manure_system"],
    )

    # 7. Cropland calculations by input type (fertilizer/liming/urea/residue)
    ag_cropland_calculations_total = Counter(
        "gl_ag_cropland_calculations_total",
        "Total cropland emission calculations by agricultural input type",
        labelnames=["input_type"],
    )

    # 8. Rice cultivation calculations by water regime
    ag_rice_calculations_total = Counter(
        "gl_ag_rice_calculations_total",
        "Total rice cultivation emission calculations by water regime",
        labelnames=["water_regime"],
    )

    # 9. Field burning calculations by crop type
    ag_field_burning_calculations_total = Counter(
        "gl_ag_field_burning_calculations_total",
        "Total field burning emission calculations by crop type",
        labelnames=["crop_type"],
    )

    # 10. Compliance checks by framework and status
    ag_compliance_checks_total = Counter(
        "gl_ag_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 11. Uncertainty analysis runs by method
    ag_uncertainty_runs_total = Counter(
        "gl_ag_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 12. Currently active farms by tenant
    ag_active_farms = Gauge(
        "gl_ag_active_farms",
        "Number of currently active farms by tenant",
        labelnames=["tenant_id"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    ag_calculations_total = None              # type: ignore[assignment]
    ag_calculation_duration_seconds = None     # type: ignore[assignment]
    ag_calculation_errors_total = None         # type: ignore[assignment]
    ag_emissions_tco2e_total = None            # type: ignore[assignment]
    ag_enteric_calculations_total = None       # type: ignore[assignment]
    ag_manure_calculations_total = None        # type: ignore[assignment]
    ag_cropland_calculations_total = None      # type: ignore[assignment]
    ag_rice_calculations_total = None          # type: ignore[assignment]
    ag_field_burning_calculations_total = None  # type: ignore[assignment]
    ag_compliance_checks_total = None          # type: ignore[assignment]
    ag_uncertainty_runs_total = None           # type: ignore[assignment]
    ag_active_farms = None                    # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    emission_source: str,
    calculation_method: str,
    animal_type_or_crop: str,
) -> None:
    """Record an agricultural emission calculation event.

    Args:
        emission_source: Agricultural emission source category
            (enteric_fermentation, manure_management, rice_cultivation,
            agricultural_soils, field_burning, liming, urea_application,
            crop_residue, prescribed_burning, savanna_burning).
        calculation_method: Calculation methodology applied (ipcc_tier1,
            ipcc_tier2, ipcc_tier3, country_specific, direct_measurement,
            emission_factor, mass_balance).
        animal_type_or_crop: Animal type or crop species involved
            (dairy_cattle, non_dairy_cattle, buffalo, sheep, goats,
            swine, poultry, horses, mules_asses, camels, llamas,
            rice, wheat, maize, sugarcane, cotton, other_crop).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_calculations_total.labels(
        emission_source=emission_source,
        calculation_method=calculation_method,
        animal_type_or_crop=animal_type_or_crop,
    ).inc()


def observe_calculation_duration(
    emission_source: str,
    calculation_method: str,
    seconds: float,
) -> None:
    """Record the duration of an agricultural emission calculation operation.

    Args:
        emission_source: Agricultural emission source category
            (enteric_fermentation, manure_management, rice_cultivation,
            agricultural_soils, field_burning, liming, urea_application,
            crop_residue, prescribed_burning, savanna_burning).
        calculation_method: Calculation methodology applied (ipcc_tier1,
            ipcc_tier2, ipcc_tier3, country_specific, direct_measurement,
            emission_factor, mass_balance).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_calculation_duration_seconds.labels(
        emission_source=emission_source,
        calculation_method=calculation_method,
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
    ag_calculation_errors_total.labels(
        error_type=error_type,
    ).inc()


def record_emissions(
    gas: str,
    emission_source: str,
    tco2e: float = 1.0,
) -> None:
    """Record cumulative emissions by gas and emission source.

    Args:
        gas: Greenhouse gas species (CO2, CH4, N2O, biogenic_CO2).
        emission_source: Agricultural emission source that produced the
            emissions (enteric_fermentation, manure_management,
            rice_cultivation, agricultural_soils, field_burning, liming,
            urea_application, crop_residue, prescribed_burning,
            savanna_burning).
        tco2e: Emission amount in tonnes CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_emissions_tco2e_total.labels(
        gas=gas,
        emission_source=emission_source,
    ).inc(tco2e)


def record_enteric_calculation(
    animal_type: str,
    calculation_method: str,
) -> None:
    """Record an enteric fermentation calculation event.

    Enteric fermentation is the digestive process in ruminant livestock
    that produces methane (CH4) as a by-product. This metric tracks
    calculations by animal type and methodology tier.

    Args:
        animal_type: Type of animal for enteric fermentation calculation
            (dairy_cattle, non_dairy_cattle, buffalo, sheep, goats,
            swine, poultry, horses, mules_asses, camels, llamas).
        calculation_method: Calculation methodology applied (ipcc_tier1,
            ipcc_tier2, ipcc_tier3, country_specific, direct_measurement,
            emission_factor, mass_balance).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_enteric_calculations_total.labels(
        animal_type=animal_type,
        calculation_method=calculation_method,
    ).inc()


def record_manure_calculation(
    animal_type: str,
    manure_system: str,
) -> None:
    """Record a manure management calculation event.

    Manure management systems produce CH4 from anaerobic decomposition
    and N2O from nitrification/denitrification. This metric tracks
    calculations by animal type and manure management system.

    Args:
        animal_type: Type of animal for manure management calculation
            (dairy_cattle, non_dairy_cattle, buffalo, sheep, goats,
            swine, poultry, horses, mules_asses, camels, llamas).
        manure_system: Manure management system used (pasture_range,
            daily_spread, solid_storage, dry_lot, liquid_slurry,
            uncovered_anaerobic_lagoon, pit_storage, anaerobic_digester,
            burned_for_fuel, deep_bedding, composting,
            aerobic_treatment).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_manure_calculations_total.labels(
        animal_type=animal_type,
        manure_system=manure_system,
    ).inc()


def record_cropland_calculation(input_type: str) -> None:
    """Record a cropland emission calculation event.

    Cropland emissions arise from fertilizer application (synthetic N
    producing N2O), lite limestone application (producing CO2), urea
    hydrolysis (producing CO2), and crop residue decomposition
    (producing N2O). This metric tracks calculations by input type.

    Args:
        input_type: Type of agricultural input for the calculation
            (fertilizer, liming, urea, residue).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_cropland_calculations_total.labels(
        input_type=input_type,
    ).inc()


def record_rice_calculation(water_regime: str) -> None:
    """Record a rice cultivation emission calculation event.

    Rice paddies are a significant source of CH4 due to anaerobic
    decomposition of organic material in flooded conditions. Emissions
    vary substantially by water management regime. This metric tracks
    calculations by water regime.

    Args:
        water_regime: Water management regime for rice cultivation
            (continuously_flooded, intermittent_single,
            intermittent_multiple, rainfed_regular, rainfed_drought,
            deep_water, upland).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_rice_calculations_total.labels(
        water_regime=water_regime,
    ).inc()


def record_field_burning_calculation(crop_type: str) -> None:
    """Record a field burning emission calculation event.

    Prescribed burning of agricultural residues in-field produces CH4,
    N2O, CO, and NOx. This metric tracks calculations by crop type
    whose residues are being burned.

    Args:
        crop_type: Type of crop whose residues are being burned (rice,
            wheat, maize, sugarcane, cotton, barley, oats, sorghum,
            millet, other).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_field_burning_calculations_total.labels(
        crop_type=crop_type,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a compliance check event.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            IPCC_2006, IPCC_2019, CSRD_ESRS_E1, SBTi_FLAG,
            EPA_GHGRP, UNFCCC).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_compliance_checks_total.labels(
        framework=framework,
        status=status,
    ).inc()


def record_uncertainty_run(method: str) -> None:
    """Record an uncertainty quantification run.

    Args:
        method: Uncertainty method applied (monte_carlo, analytical,
            error_propagation, ipcc_default_uncertainty).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def track_active_farms(tenant_id: str, count: int) -> None:
    """Set the gauge for currently active farms by tenant.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        tenant_id: Tenant identifier for the farm count.
        count: Number of active farms for the tenant. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    ag_active_farms.labels(
        tenant_id=tenant_id,
    ).set(count)


# ---------------------------------------------------------------------------
# MetricsCollector - Thread-safe singleton facade
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe singleton that wraps all agricultural Prometheus metrics.

    The :class:`MetricsCollector` provides an object-oriented interface for
    recording agricultural emissions metrics. Only one instance is created
    regardless of how many times the constructor is invoked, making it safe
    for use in multi-threaded agent pipelines, batch processors, and async
    workers.

    All ``record_*`` methods delegate to the module-level helper functions
    which gracefully no-op when ``prometheus_client`` is not installed.

    Thread Safety:
        Uses a reentrant lock (``threading.RLock``) around the singleton
        creation to prevent race conditions during first access in
        concurrent environments.

    Attributes:
        prometheus_available: Whether the prometheus_client library is
            importable and metrics are active.

    Example:
        >>> collector = MetricsCollector()
        >>> assert collector is MetricsCollector()  # same instance
        >>> collector.record_calculation(
        ...     "enteric_fermentation", "ipcc_tier1", "dairy_cattle"
        ... )
        >>> collector.record_emissions(
        ...     "CH4", "enteric_fermentation", 42.5
        ... )
        >>> collector.track_active_farms("tenant_001", 15)
    """

    _instance: MetricsCollector | None = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> MetricsCollector:
        """Return the singleton MetricsCollector instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Returns:
            The singleton :class:`MetricsCollector` instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.prometheus_available = PROMETHEUS_AVAILABLE
                    cls._instance = instance
                    logger.debug("MetricsCollector singleton created")
        return cls._instance

    # -- Calculation tracking ------------------------------------------------

    def record_calculation(
        self,
        emission_source: str,
        calculation_method: str,
        animal_type_or_crop: str,
    ) -> None:
        """Record an agricultural emission calculation event.

        Args:
            emission_source: Agricultural emission source category.
            calculation_method: Calculation methodology applied.
            animal_type_or_crop: Animal type or crop species involved.
        """
        record_calculation(emission_source, calculation_method, animal_type_or_crop)

    def observe_calculation_duration(
        self,
        emission_source: str,
        calculation_method: str,
        seconds: float,
    ) -> None:
        """Record the duration of an agricultural emission calculation.

        Args:
            emission_source: Agricultural emission source category.
            calculation_method: Calculation methodology applied.
            seconds: Operation wall-clock time in seconds.
        """
        observe_calculation_duration(emission_source, calculation_method, seconds)

    def record_calculation_error(self, error_type: str) -> None:
        """Record a calculation error event.

        Args:
            error_type: Type of error encountered.
        """
        record_calculation_error(error_type)

    # -- Emissions tracking --------------------------------------------------

    def record_emissions(
        self,
        gas: str,
        emission_source: str,
        tco2e: float = 1.0,
    ) -> None:
        """Record cumulative emissions by gas and emission source.

        Args:
            gas: Greenhouse gas species.
            emission_source: Agricultural emission source.
            tco2e: Emission amount in tonnes CO2e.
        """
        record_emissions(gas, emission_source, tco2e)

    # -- Enteric fermentation tracking ---------------------------------------

    def record_enteric_calculation(
        self,
        animal_type: str,
        calculation_method: str,
    ) -> None:
        """Record an enteric fermentation calculation event.

        Args:
            animal_type: Type of animal for enteric fermentation.
            calculation_method: Calculation methodology applied.
        """
        record_enteric_calculation(animal_type, calculation_method)

    # -- Manure management tracking ------------------------------------------

    def record_manure_calculation(
        self,
        animal_type: str,
        manure_system: str,
    ) -> None:
        """Record a manure management calculation event.

        Args:
            animal_type: Type of animal for manure management.
            manure_system: Manure management system used.
        """
        record_manure_calculation(animal_type, manure_system)

    # -- Cropland tracking ---------------------------------------------------

    def record_cropland_calculation(self, input_type: str) -> None:
        """Record a cropland emission calculation event.

        Args:
            input_type: Type of agricultural input (fertilizer, liming,
                urea, residue).
        """
        record_cropland_calculation(input_type)

    # -- Rice cultivation tracking -------------------------------------------

    def record_rice_calculation(self, water_regime: str) -> None:
        """Record a rice cultivation emission calculation event.

        Args:
            water_regime: Water management regime for rice cultivation.
        """
        record_rice_calculation(water_regime)

    # -- Field burning tracking ----------------------------------------------

    def record_field_burning_calculation(self, crop_type: str) -> None:
        """Record a field burning emission calculation event.

        Args:
            crop_type: Type of crop whose residues are being burned.
        """
        record_field_burning_calculation(crop_type)

    # -- Compliance and uncertainty ------------------------------------------

    def record_compliance_check(self, framework: str, status: str) -> None:
        """Record a compliance check event.

        Args:
            framework: Regulatory framework checked.
            status: Compliance status result.
        """
        record_compliance_check(framework, status)

    def record_uncertainty_run(self, method: str) -> None:
        """Record an uncertainty quantification run.

        Args:
            method: Uncertainty method applied.
        """
        record_uncertainty_run(method)

    # -- Farm gauge ----------------------------------------------------------

    def track_active_farms(self, tenant_id: str, count: int) -> None:
        """Set the gauge for active farms by tenant.

        Args:
            tenant_id: Tenant identifier.
            count: Number of active farms. Must be >= 0.
        """
        track_active_farms(tenant_id, count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "ag_calculations_total",
    "ag_calculation_duration_seconds",
    "ag_calculation_errors_total",
    "ag_emissions_tco2e_total",
    "ag_enteric_calculations_total",
    "ag_manure_calculations_total",
    "ag_cropland_calculations_total",
    "ag_rice_calculations_total",
    "ag_field_burning_calculations_total",
    "ag_compliance_checks_total",
    "ag_uncertainty_runs_total",
    "ag_active_farms",
    # Helper functions
    "record_calculation",
    "observe_calculation_duration",
    "record_calculation_error",
    "record_emissions",
    "record_enteric_calculation",
    "record_manure_calculation",
    "record_cropland_calculation",
    "record_rice_calculation",
    "record_field_burning_calculation",
    "record_compliance_check",
    "record_uncertainty_run",
    "track_active_farms",
    # Collector class
    "MetricsCollector",
]
