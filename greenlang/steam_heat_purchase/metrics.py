# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-011: Steam/Heat Purchase Agent

12 Prometheus metrics for steam and heat purchase emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_shp_`` prefix (GreenLang Steam/Heat Purchase)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_shp_calculations_total                     (Counter,   labels: energy_type, calculation_method, status, tenant_id)
    2.  gl_shp_calculation_duration_seconds           (Histogram, labels: energy_type, calculation_method)
    3.  gl_shp_total_co2e_kg                          (Counter,   labels: energy_type, fuel_type, tenant_id)
    4.  gl_shp_biogenic_co2_kg                        (Counter,   labels: fuel_type, tenant_id)
    5.  gl_shp_batch_calculations_total               (Counter,   labels: status, tenant_id)
    6.  gl_shp_batch_size                             (Histogram, labels: tenant_id)
    7.  gl_shp_chp_allocations_total                  (Counter,   labels: method, fuel_type, tenant_id)
    8.  gl_shp_uncertainty_analyses_total              (Counter,   labels: method, tenant_id)
    9.  gl_shp_compliance_checks_total                (Counter,   labels: framework, status, tenant_id)
    10. gl_shp_active_facilities                      (Gauge,     labels: facility_type, tenant_id)
    11. gl_shp_database_lookups_total                 (Counter,   labels: lookup_type, status)
    12. gl_shp_errors_total                           (Counter,   labels: engine, error_type, tenant_id)

Label Values Reference:
    energy_type:
        steam, hot_water, chilled_water, district_heat, district_cooling,
        process_heat, low_pressure_steam, medium_pressure_steam,
        high_pressure_steam, condensate_return.
    calculation_method:
        supplier_specific, default_emission_factor, fuel_specific,
        chp_allocation, efficiency_ratio, energy_balance,
        residual_mix, weighted_average, direct_measurement.
    status (calculations_total):
        success, failure, partial, skipped.
    status (batch_calculations_total):
        success, failure, partial.
    status (compliance_checks_total):
        compliant, non_compliant, partial, not_assessed, warning.
    status (database_lookups_total):
        hit, miss, error, timeout.
    fuel_type:
        natural_gas, coal, fuel_oil, biomass, biogas, waste,
        municipal_solid_waste, industrial_waste, wood_chips,
        wood_pellets, bagasse, landfill_gas, mixed, unknown.
    method (chp_allocations_total):
        efficiency, energy, exergy, iea_fixed_heat,
        iea_fixed_power, carnot, finnish, ppa.
    method (uncertainty_analyses_total):
        monte_carlo, analytical, error_propagation,
        ipcc_default_uncertainty, bootstrap.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, UK_SECR,
        EPA_CHP, EU_ETS, EU_EED, EPA_EGRID, IEA.
    facility_type:
        chp_plant, district_heating, district_cooling,
        industrial_boiler, waste_heat_recovery, geothermal,
        solar_thermal, heat_pump, biomass_plant.
    engine:
        database, calculator, allocator, validator, compliance,
        uncertainty, pipeline.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, allocation_error,
        efficiency_error, unit_conversion_error, unknown_error.
    tenant_id:
        Free-form string identifying the tenant for multi-tenant
        isolation and per-tenant metric aggregation.
    lookup_type:
        emission_factor, fuel_property, efficiency_default,
        chp_parameter, grid_factor, conversion_factor.

Example:
    Using the SteamHeatPurchaseMetrics singleton class:

    >>> from greenlang.steam_heat_purchase.metrics import SteamHeatPurchaseMetrics
    >>> metrics = SteamHeatPurchaseMetrics()
    >>> metrics.record_calculation(
    ...     "steam", "supplier_specific", "success", 0.042,
    ...     150.3, 0.0, "natural_gas", "tenant-001"
    ... )
    >>> metrics.record_chp_allocation("efficiency", "natural_gas", "tenant-001")
    >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant", "tenant-001")
    >>> metrics.set_active_facilities("chp_plant", 12, "tenant-001")

    Using the module-level convenience function:

    >>> from greenlang.steam_heat_purchase.metrics import get_metrics
    >>> m = get_metrics()
    >>> m.record_calculation(
    ...     "hot_water", "default_emission_factor", "success",
    ...     0.065, 45.0, 2.1, "biomass", "tenant-002"
    ... )
    >>> m.record_error("calculator", "validation_error", "tenant-002")
    >>> summary = m.get_metrics_summary()

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    logger.info(
        "prometheus_client not installed; "
        "Steam/Heat Purchase emissions metrics disabled"
    )


# ---------------------------------------------------------------------------
# SteamHeatPurchaseMetrics - Thread-safe singleton with graceful fallback
# ---------------------------------------------------------------------------


class SteamHeatPurchaseMetrics:
    """Thread-safe singleton for all Steam/Heat Purchase Prometheus metrics.

    The :class:`SteamHeatPurchaseMetrics` encapsulates the full set of 12
    Prometheus metrics used by the Steam/Heat Purchase Emissions Agent.
    It implements the singleton pattern so that only one instance (and
    therefore one set of Prometheus collectors) is created per process,
    regardless of how many callers instantiate it.

    When ``prometheus_client`` is not installed, every recording method
    becomes a silent no-op -- no exceptions are raised and no state is
    mutated.  This allows agent code to call metrics unconditionally
    without guarding on library availability.

    An optional ``registry`` parameter can be supplied to direct all
    metrics into a custom :class:`CollectorRegistry` (useful for testing
    or multi-registry deployments).  When ``None`` (the default), the
    process-wide default registry is used.

    Thread Safety:
        Uses a reentrant lock (``threading.RLock``) around the singleton
        creation to prevent race conditions during first access in
        concurrent environments.  Individual metric recording calls are
        inherently thread-safe because ``prometheus_client`` collectors
        use atomic operations internally.

    Attributes:
        prometheus_available: Whether the ``prometheus_client`` library
            was successfully imported and metrics are active.

    Example:
        >>> metrics = SteamHeatPurchaseMetrics()
        >>> assert metrics is SteamHeatPurchaseMetrics()  # same instance
        >>> metrics.record_calculation(
        ...     "steam", "supplier_specific", "success",
        ...     0.042, 150.3, 0.0, "natural_gas", "tenant-001"
        ... )
        >>> metrics.record_compliance_check(
        ...     "GHG_PROTOCOL", "compliant", "tenant-001"
        ... )
        >>> metrics.set_active_facilities("chp_plant", 25, "tenant-001")
        >>> summary = metrics.get_metrics_summary()
    """

    _instance: Optional[SteamHeatPurchaseMetrics] = None
    _lock: threading.RLock = threading.RLock()

    # -- Singleton construction ---------------------------------------------

    def __new__(
        cls,
        registry: Any = None,
    ) -> SteamHeatPurchaseMetrics:
        """Return the singleton SteamHeatPurchaseMetrics instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Args:
            registry: Optional :class:`CollectorRegistry` for metric
                registration.  Ignored after the first construction.

        Returns:
            The singleton :class:`SteamHeatPurchaseMetrics` instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(
        self,
        registry: Any = None,
    ) -> None:
        """Initialize all 12 Prometheus metrics.

        This method is idempotent: after the first call, subsequent
        invocations are silently skipped to prevent duplicate collector
        registration errors.

        Args:
            registry: Optional :class:`CollectorRegistry`.  When
                ``None``, the default process-wide registry is used.
                Only honoured on the very first initialization.
        """
        if self._initialized:
            return

        self.prometheus_available: bool = PROMETHEUS_AVAILABLE
        self._registry = registry

        if PROMETHEUS_AVAILABLE:
            self._init_metrics(registry)
        else:
            self._init_noop_metrics()

        self._initialized = True
        logger.debug(
            "SteamHeatPurchaseMetrics singleton initialized "
            "(prometheus_available=%s)",
            self.prometheus_available,
        )

    # -- Private metric initialization --------------------------------------

    def _init_metrics(self, registry: Any) -> None:
        """Create all 12 Prometheus collectors.

        Args:
            registry: Optional :class:`CollectorRegistry` or ``None``
                for the default registry.
        """
        reg_kwargs: Dict[str, Any] = {}
        if registry is not None:
            reg_kwargs["registry"] = registry

        # 1. gl_shp_calculations_total
        #    Total steam/heat purchase emission calculations performed,
        #    segmented by energy type, calculation method, status, and
        #    tenant.  This is the primary throughput counter for the
        #    agent and drives SLI calculations for availability and
        #    error rate.
        self.calculations_total = Counter(
            "gl_shp_calculations_total",
            "Total steam/heat purchase emission calculations performed",
            labelnames=[
                "energy_type",
                "calculation_method",
                "status",
                "tenant_id",
            ],
            **reg_kwargs,
        )

        # 2. gl_shp_calculation_duration_seconds
        #    Duration histogram for steam/heat purchase calculations,
        #    segmented by energy type and calculation method.  Bucket
        #    boundaries are tuned for typical calculation latencies
        #    ranging from 10ms (simple default EF) to 10s (complex
        #    CHP allocation with Monte Carlo uncertainty).
        self.calculation_duration_seconds = Histogram(
            "gl_shp_calculation_duration_seconds",
            "Duration of steam/heat purchase calculation operations "
            "in seconds",
            labelnames=["energy_type", "calculation_method"],
            buckets=(
                0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0,
            ),
            **reg_kwargs,
        )

        # 3. gl_shp_total_co2e_kg
        #    Cumulative CO2-equivalent emissions in kilograms from
        #    steam/heat purchases, segmented by energy type, fuel type,
        #    and tenant.  Uses kilograms as the base unit for precision
        #    in facility-level calculations.
        self.total_co2e_kg = Counter(
            "gl_shp_total_co2e_kg",
            "Cumulative steam/heat purchase emissions in kg CO2e "
            "by energy type and fuel type",
            labelnames=["energy_type", "fuel_type", "tenant_id"],
            **reg_kwargs,
        )

        # 4. gl_shp_biogenic_co2_kg
        #    Cumulative biogenic CO2 emissions in kilograms from
        #    biomass-fueled steam and heat systems.  Biogenic CO2 is
        #    reported separately per GHG Protocol and CSRD requirements
        #    as it originates from short-cycle carbon in biomass fuels
        #    (wood chips, bagasse, biogas, etc.).
        self.biogenic_co2_kg = Counter(
            "gl_shp_biogenic_co2_kg",
            "Cumulative biogenic CO2 emissions in kg from "
            "biomass-fueled steam/heat systems",
            labelnames=["fuel_type", "tenant_id"],
            **reg_kwargs,
        )

        # 5. gl_shp_batch_calculations_total
        #    Total batch calculation jobs processed, segmented by
        #    completion status and tenant.  Batch calculations process
        #    multiple facilities or time periods in a single operation
        #    for efficiency.
        self.batch_calculations_total = Counter(
            "gl_shp_batch_calculations_total",
            "Total batch calculation jobs processed by status",
            labelnames=["status", "tenant_id"],
            **reg_kwargs,
        )

        # 6. gl_shp_batch_size
        #    Histogram of batch sizes (number of records per batch
        #    calculation), segmented by tenant.  Bucket boundaries
        #    range from single-record batches to bulk operations of
        #    10,000+ records, calibrated against typical facility
        #    portfolio sizes.
        self.batch_size = Histogram(
            "gl_shp_batch_size",
            "Distribution of batch calculation sizes "
            "(number of records per batch)",
            labelnames=["tenant_id"],
            buckets=(
                1.0, 5.0, 10.0, 25.0, 50.0,
                100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0,
            ),
            **reg_kwargs,
        )

        # 7. gl_shp_chp_allocations_total
        #    Total Combined Heat and Power (CHP) allocation calculations,
        #    segmented by allocation method, fuel type, and tenant.  CHP
        #    allocation is required to split total CHP plant emissions
        #    between the heat/steam output and the electricity output
        #    per GHG Protocol, EU ETS, and EPA CHP guidance.
        self.chp_allocations_total = Counter(
            "gl_shp_chp_allocations_total",
            "Total CHP allocation calculations by method and fuel type",
            labelnames=["method", "fuel_type", "tenant_id"],
            **reg_kwargs,
        )

        # 8. gl_shp_uncertainty_analyses_total
        #    Total uncertainty analysis runs for steam/heat purchase
        #    emission calculations, segmented by statistical method
        #    and tenant.  Uncertainty quantification follows IPCC
        #    guidelines for Tier 1/2/3 approaches.
        self.uncertainty_analyses_total = Counter(
            "gl_shp_uncertainty_analyses_total",
            "Total uncertainty analysis runs by method",
            labelnames=["method", "tenant_id"],
            **reg_kwargs,
        )

        # 9. gl_shp_compliance_checks_total
        #    Compliance checks against regulatory frameworks, segmented
        #    by framework name, pass/fail status, and tenant.  Supports
        #    GHG Protocol, ISO 14064, CSRD ESRS E1, UK SECR, EPA CHP
        #    output-based standards, EU ETS, and EU EED requirements.
        self.compliance_checks_total = Counter(
            "gl_shp_compliance_checks_total",
            "Total compliance checks by framework and status",
            labelnames=["framework", "status", "tenant_id"],
            **reg_kwargs,
        )

        # 10. gl_shp_active_facilities
        #     Gauge tracking the number of currently active facilities
        #     that purchase steam or heat, segmented by facility type
        #     and tenant.  This metric reflects the real-time portfolio
        #     of monitored facilities for capacity planning and
        #     dashboard visualization.
        self.active_facilities = Gauge(
            "gl_shp_active_facilities",
            "Number of currently active steam/heat purchase facilities "
            "by facility type",
            labelnames=["facility_type", "tenant_id"],
            **reg_kwargs,
        )

        # 11. gl_shp_database_lookups_total
        #     Database lookup operations for emission factors, fuel
        #     properties, CHP parameters, and other reference data,
        #     segmented by lookup type and cache hit/miss status.
        #     Tracks data layer performance and cache efficiency.
        self.database_lookups_total = Counter(
            "gl_shp_database_lookups_total",
            "Total database lookups by lookup type and status",
            labelnames=["lookup_type", "status"],
            **reg_kwargs,
        )

        # 12. gl_shp_errors_total
        #     Error events by engine component, error classification,
        #     and tenant.  Used for operational alerting, error budget
        #     tracking (OBS-005 SLO integration), and root cause
        #     analysis across the agent pipeline.
        self.errors_total = Counter(
            "gl_shp_errors_total",
            "Total steam/heat purchase agent errors by engine "
            "and error type",
            labelnames=["engine", "error_type", "tenant_id"],
            **reg_kwargs,
        )

    def _init_noop_metrics(self) -> None:
        """Set all metric attributes to ``None`` for the no-op fallback.

        When ``prometheus_client`` is not installed, every attribute is
        ``None`` and every ``record_*`` / ``set_*`` method returns
        immediately without side effects.
        """
        self.calculations_total = None  # type: ignore[assignment]
        self.calculation_duration_seconds = None  # type: ignore[assignment]
        self.total_co2e_kg = None  # type: ignore[assignment]
        self.biogenic_co2_kg = None  # type: ignore[assignment]
        self.batch_calculations_total = None  # type: ignore[assignment]
        self.batch_size = None  # type: ignore[assignment]
        self.chp_allocations_total = None  # type: ignore[assignment]
        self.uncertainty_analyses_total = None  # type: ignore[assignment]
        self.compliance_checks_total = None  # type: ignore[assignment]
        self.active_facilities = None  # type: ignore[assignment]
        self.database_lookups_total = None  # type: ignore[assignment]
        self.errors_total = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Recording methods -- Calculations
    # -----------------------------------------------------------------------

    def record_calculation(
        self,
        energy_type: str,
        method: str,
        status: str,
        duration: float,
        co2e_kg: float,
        biogenic_kg: float,
        fuel_type: str,
        tenant_id: str,
    ) -> None:
        """Record a steam/heat purchase emission calculation.

        This is the primary recording method that atomically updates
        four metrics at once: the calculation counter, the duration
        histogram, the cumulative CO2e counter, and (when non-zero)
        the biogenic CO2 counter.

        Args:
            energy_type: Type of purchased thermal energy (steam,
                hot_water, chilled_water, district_heat,
                district_cooling, process_heat, low_pressure_steam,
                medium_pressure_steam, high_pressure_steam,
                condensate_return).
            method: Calculation methodology applied (supplier_specific,
                default_emission_factor, fuel_specific, chp_allocation,
                efficiency_ratio, energy_balance, residual_mix,
                weighted_average, direct_measurement).
            status: Calculation outcome (success, failure, partial,
                skipped).
            duration: Calculation wall-clock time in seconds.
                Histogram buckets: 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0.
            co2e_kg: Total CO2-equivalent emissions in kilograms
                resulting from this calculation.  Must be >= 0.
            biogenic_kg: Biogenic CO2 emissions in kilograms from
                biomass-fueled sources.  Pass 0.0 when not applicable.
                Must be >= 0.
            fuel_type: Fuel type of the steam/heat source (natural_gas,
                coal, fuel_oil, biomass, biogas, waste,
                municipal_solid_waste, industrial_waste, wood_chips,
                wood_pellets, bagasse, landfill_gas, mixed, unknown).
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.calculations_total.labels(
            energy_type=energy_type,
            calculation_method=method,
            status=status,
            tenant_id=tenant_id,
        ).inc()

        self.calculation_duration_seconds.labels(
            energy_type=energy_type,
            calculation_method=method,
        ).observe(duration)

        if co2e_kg > 0:
            self.total_co2e_kg.labels(
                energy_type=energy_type,
                fuel_type=fuel_type,
                tenant_id=tenant_id,
            ).inc(co2e_kg)

        if biogenic_kg > 0:
            self.biogenic_co2_kg.labels(
                fuel_type=fuel_type,
                tenant_id=tenant_id,
            ).inc(biogenic_kg)

    # -----------------------------------------------------------------------
    # Recording methods -- Batch calculations
    # -----------------------------------------------------------------------

    def record_batch(
        self,
        status: str,
        size: int,
        tenant_id: str,
    ) -> None:
        """Record a batch calculation job.

        Atomically updates the batch counter and the batch size
        histogram.  Batch calculations process multiple facilities
        or time periods in a single operation to improve throughput
        and reduce per-request overhead.

        Args:
            status: Batch job outcome (success, failure, partial).
            size: Number of individual records in the batch.
                Must be >= 1.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.batch_calculations_total.labels(
            status=status,
            tenant_id=tenant_id,
        ).inc()

        self.batch_size.labels(
            tenant_id=tenant_id,
        ).observe(size)

    # -----------------------------------------------------------------------
    # Recording methods -- CHP allocation
    # -----------------------------------------------------------------------

    def record_chp_allocation(
        self,
        method: str,
        fuel_type: str,
        tenant_id: str,
    ) -> None:
        """Record a Combined Heat and Power (CHP) allocation calculation.

        CHP plants produce both heat/steam and electricity
        simultaneously.  The allocation method determines how total
        plant emissions are split between the thermal and electrical
        outputs.  This counter tracks the allocation methodology used
        for each CHP calculation to monitor method distribution and
        support regulatory compliance auditing.

        Supported allocation methods include the efficiency method
        (default for GHG Protocol), energy method (simple energy
        ratio), exergy method (thermodynamic quality weighting),
        IEA fixed-heat and fixed-power approaches, the Carnot method
        (based on theoretical efficiency), and the Finnish method
        (used in Nordic markets).

        Args:
            method: CHP allocation method applied (efficiency, energy,
                exergy, iea_fixed_heat, iea_fixed_power, carnot,
                finnish, ppa).
            fuel_type: Fuel type of the CHP plant (natural_gas, coal,
                fuel_oil, biomass, biogas, waste, municipal_solid_waste,
                industrial_waste, wood_chips, wood_pellets, bagasse,
                landfill_gas, mixed, unknown).
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.chp_allocations_total.labels(
            method=method,
            fuel_type=fuel_type,
            tenant_id=tenant_id,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Uncertainty analysis
    # -----------------------------------------------------------------------

    def record_uncertainty(
        self,
        method: str,
        tenant_id: str,
    ) -> None:
        """Record an uncertainty analysis run.

        Uncertainty quantification is performed following IPCC 2006
        Guidelines Volume 1, Chapter 3 and the GHG Protocol guidance
        on uncertainty assessment.  This counter tracks each run to
        monitor analytical workload and method preference across
        tenants.

        Args:
            method: Statistical method applied (monte_carlo, analytical,
                error_propagation, ipcc_default_uncertainty, bootstrap).
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.uncertainty_analyses_total.labels(
            method=method,
            tenant_id=tenant_id,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Compliance checks
    # -----------------------------------------------------------------------

    def record_compliance_check(
        self,
        framework: str,
        status: str,
        tenant_id: str,
    ) -> None:
        """Record a regulatory compliance check event.

        Compliance checks validate that steam/heat purchase emission
        calculations meet the requirements of the specified regulatory
        framework.  Results drive the compliance dashboard and can
        trigger automated remediation workflows when non-compliance
        is detected.

        Args:
            framework: Regulatory framework checked (GHG_PROTOCOL,
                ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_CHP, EU_ETS,
                EU_EED, EPA_EGRID, IEA).
            status: Compliance status result (compliant, non_compliant,
                partial, not_assessed, warning).
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.compliance_checks_total.labels(
            framework=framework,
            status=status,
            tenant_id=tenant_id,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Active facilities gauge
    # -----------------------------------------------------------------------

    def set_active_facilities(
        self,
        facility_type: str,
        count: int,
        tenant_id: str,
    ) -> None:
        """Set the gauge for currently active steam/heat purchase facilities.

        This is an absolute set (not an increment) so the caller is
        responsible for computing the correct current count of active
        facilities for the given type and tenant.  The gauge reflects
        the real-time portfolio of monitored facilities and is used
        for capacity planning dashboards and per-tenant utilization
        tracking.

        Args:
            facility_type: Type of facility (chp_plant,
                district_heating, district_cooling, industrial_boiler,
                waste_heat_recovery, geothermal, solar_thermal,
                heat_pump, biomass_plant).
            count: Number of currently active facilities of this type.
                Must be >= 0.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.active_facilities.labels(
            facility_type=facility_type,
            tenant_id=tenant_id,
        ).set(count)

    # -----------------------------------------------------------------------
    # Recording methods -- Database lookups
    # -----------------------------------------------------------------------

    def record_db_lookup(
        self,
        lookup_type: str,
        status: str,
    ) -> None:
        """Record a database lookup operation.

        Tracks reference data lookups for emission factors, fuel
        properties, CHP parameters, conversion factors, and other
        values required for steam/heat purchase emission calculations.
        The hit/miss/error segmentation enables cache efficiency
        monitoring and data layer performance optimization.

        Args:
            lookup_type: Category of data being looked up
                (emission_factor, fuel_property, efficiency_default,
                chp_parameter, grid_factor, conversion_factor).
            status: Lookup outcome (hit, miss, error, timeout).
        """
        if not self.prometheus_available:
            return

        self.database_lookups_total.labels(
            lookup_type=lookup_type,
            status=status,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Errors
    # -----------------------------------------------------------------------

    def record_error(
        self,
        engine: str,
        error_type: str,
        tenant_id: str,
    ) -> None:
        """Record an error event for operational alerting.

        Error events are segmented by the engine component that raised
        the error, the error classification, and the tenant.  This
        three-dimensional labeling enables targeted alerting rules
        (e.g. alert on database errors exceeding 5/min) and root
        cause analysis by isolating errors to specific pipeline
        stages.

        Args:
            engine: Engine component that produced the error (database,
                calculator, allocator, validator, compliance,
                uncertainty, pipeline).
            error_type: Classification of the error (validation_error,
                calculation_error, database_error, configuration_error,
                timeout_error, allocation_error, efficiency_error,
                unit_conversion_error, unknown_error).
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.errors_total.labels(
            engine=engine,
            error_type=error_type,
            tenant_id=tenant_id,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Emissions by energy type
    # -----------------------------------------------------------------------

    def record_emissions(
        self,
        energy_type: str,
        fuel_type: str,
        co2e_kg: float,
        tenant_id: str,
    ) -> None:
        """Record emissions by energy type and fuel type.

        Use this method to record emissions outside of the primary
        :meth:`record_calculation` flow, for example when adjusting
        emissions due to CHP reallocation, retroactive corrections,
        or supplementary emission sources.

        Args:
            energy_type: Type of purchased thermal energy (steam,
                hot_water, chilled_water, district_heat,
                district_cooling, process_heat, low_pressure_steam,
                medium_pressure_steam, high_pressure_steam,
                condensate_return).
            fuel_type: Fuel type of the steam/heat source (natural_gas,
                coal, fuel_oil, biomass, biogas, waste,
                municipal_solid_waste, industrial_waste, wood_chips,
                wood_pellets, bagasse, landfill_gas, mixed, unknown).
            co2e_kg: Emission amount in kilograms CO2e to add to the
                counter.  Must be > 0.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.total_co2e_kg.labels(
            energy_type=energy_type,
            fuel_type=fuel_type,
            tenant_id=tenant_id,
        ).inc(co2e_kg)

    # -----------------------------------------------------------------------
    # Recording methods -- Biogenic emissions standalone
    # -----------------------------------------------------------------------

    def record_biogenic_emissions(
        self,
        fuel_type: str,
        biogenic_kg: float,
        tenant_id: str,
    ) -> None:
        """Record biogenic CO2 emissions separately.

        Use this method to record biogenic CO2 outside of the primary
        :meth:`record_calculation` flow, for example when processing
        biomass fuel adjustments, retroactive biomass fraction
        corrections, or supplementary biogenic sources.

        Biogenic CO2 is reported separately per GHG Protocol Corporate
        Standard (Chapter 9) and CSRD ESRS E1 requirements.  It
        originates from short-cycle carbon in biomass fuels and is
        excluded from the Scope 2 total but must be disclosed.

        Args:
            fuel_type: Biomass fuel type (biomass, biogas, wood_chips,
                wood_pellets, bagasse, landfill_gas, waste,
                municipal_solid_waste).
            biogenic_kg: Biogenic CO2 amount in kilograms.  Must be > 0.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        self.biogenic_co2_kg.labels(
            fuel_type=fuel_type,
            tenant_id=tenant_id,
        ).inc(biogenic_kg)

    # -----------------------------------------------------------------------
    # Metrics summary
    # -----------------------------------------------------------------------

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a dictionary summarising current metric values.

        The summary provides a snapshot of all 12 metrics in a format
        suitable for JSON serialisation, health-check endpoints, and
        diagnostic logging.

        When ``prometheus_client`` is not available the returned dict
        contains the metric names as keys but all values are ``None``.

        Returns:
            Dictionary with one key per metric.  Counter and Gauge
            values are numeric; the Histogram returns its documented
            bucket boundaries.

        Example:
            >>> metrics = SteamHeatPurchaseMetrics()
            >>> summary = metrics.get_metrics_summary()
            >>> isinstance(summary, dict)
            True
            >>> "gl_shp_calculations_total" in summary
            True
        """
        if not self.prometheus_available:
            return {
                "gl_shp_calculations_total": None,
                "gl_shp_calculation_duration_seconds": None,
                "gl_shp_total_co2e_kg": None,
                "gl_shp_biogenic_co2_kg": None,
                "gl_shp_batch_calculations_total": None,
                "gl_shp_batch_size": None,
                "gl_shp_chp_allocations_total": None,
                "gl_shp_uncertainty_analyses_total": None,
                "gl_shp_compliance_checks_total": None,
                "gl_shp_active_facilities": None,
                "gl_shp_database_lookups_total": None,
                "gl_shp_errors_total": None,
                "prometheus_available": False,
            }

        return {
            "gl_shp_calculations_total": _safe_describe(
                self.calculations_total
            ),
            "gl_shp_calculation_duration_seconds": _safe_describe(
                self.calculation_duration_seconds
            ),
            "gl_shp_total_co2e_kg": _safe_describe(
                self.total_co2e_kg
            ),
            "gl_shp_biogenic_co2_kg": _safe_describe(
                self.biogenic_co2_kg
            ),
            "gl_shp_batch_calculations_total": _safe_describe(
                self.batch_calculations_total
            ),
            "gl_shp_batch_size": _safe_describe(
                self.batch_size
            ),
            "gl_shp_chp_allocations_total": _safe_describe(
                self.chp_allocations_total
            ),
            "gl_shp_uncertainty_analyses_total": _safe_describe(
                self.uncertainty_analyses_total
            ),
            "gl_shp_compliance_checks_total": _safe_describe(
                self.compliance_checks_total
            ),
            "gl_shp_active_facilities": _safe_describe(
                self.active_facilities
            ),
            "gl_shp_database_lookups_total": _safe_describe(
                self.database_lookups_total
            ),
            "gl_shp_errors_total": _safe_describe(
                self.errors_total
            ),
            "prometheus_available": True,
        }

    # -----------------------------------------------------------------------
    # Singleton reset (testing only)
    # -----------------------------------------------------------------------

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton instance.

        **Testing only.**  This method is not part of the public API
        and must never be called in production code.  It is provided
        so that unit tests can create fresh metric instances with a
        custom :class:`CollectorRegistry` to avoid ``Duplicated
        timeseries`` errors.

        After calling ``_reset()``, the next instantiation will
        perform a full initialization.
        """
        with cls._lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_describe(metric: Any) -> Dict[str, Any]:
    """Extract a description dict from a Prometheus metric collector.

    Args:
        metric: A ``prometheus_client`` Counter, Histogram, or Gauge
            instance.

    Returns:
        Dictionary with ``name``, ``documentation``, ``type``, and
        ``labels`` keys.
    """
    try:
        result: Dict[str, Any] = {
            "name": metric._name,
            "documentation": metric._documentation,
            "type": metric._type,
        }
        if hasattr(metric, "_labelnames"):
            result["labels"] = list(metric._labelnames)
        return result
    except Exception:
        return {"name": "unknown", "documentation": "", "type": "unknown"}


# ---------------------------------------------------------------------------
# Module-level convenience: default singleton accessor
# ---------------------------------------------------------------------------

_default_metrics: Optional[SteamHeatPurchaseMetrics] = None


def get_metrics() -> SteamHeatPurchaseMetrics:
    """Return the module-level default :class:`SteamHeatPurchaseMetrics` instance.

    This function provides a convenient module-level accessor that
    lazily creates and caches a :class:`SteamHeatPurchaseMetrics`
    singleton.  It is the recommended entry point for agent code that
    does not need a custom :class:`CollectorRegistry`.

    Returns:
        The shared :class:`SteamHeatPurchaseMetrics` instance.

    Example:
        >>> from greenlang.steam_heat_purchase.metrics import get_metrics
        >>> m = get_metrics()
        >>> m.record_calculation(
        ...     "steam", "supplier_specific", "success",
        ...     0.05, 120.0, 0.0, "natural_gas", "tenant-001"
        ... )
        >>> m.record_error("calculator", "validation_error", "tenant-001")
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = SteamHeatPurchaseMetrics()
    return _default_metrics


def reset() -> None:
    """Reset the module-level metrics singleton.

    **Testing only.**  Resets both the module-level cached reference
    and the class-level singleton instance.  After calling this
    function, the next call to :func:`get_metrics` or
    ``SteamHeatPurchaseMetrics()`` will create a completely fresh
    instance with new Prometheus collectors.

    This is essential for test isolation -- without resetting, tests
    that create metrics with a custom :class:`CollectorRegistry`
    would encounter ``Duplicated timeseries`` errors from the
    process-wide default registry.

    Example:
        >>> from greenlang.steam_heat_purchase.metrics import reset
        >>> reset()  # clears singleton for test isolation
    """
    global _default_metrics
    _default_metrics = None
    SteamHeatPurchaseMetrics._reset()


# ---------------------------------------------------------------------------
# Module-level helper functions (safe to call without prometheus_client)
# ---------------------------------------------------------------------------
# These thin wrappers delegate to the singleton and mirror the flat-function
# API style used by other MRV metrics modules for backward compatibility.


def record_calculation(
    energy_type: str,
    method: str,
    status: str,
    duration: float,
    co2e_kg: float,
    biogenic_kg: float,
    fuel_type: str,
    tenant_id: str,
) -> None:
    """Record a steam/heat purchase emission calculation.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_calculation`.

    Args:
        energy_type: Type of purchased thermal energy (steam,
            hot_water, chilled_water, district_heat, district_cooling,
            process_heat, low_pressure_steam, medium_pressure_steam,
            high_pressure_steam, condensate_return).
        method: Calculation methodology applied (supplier_specific,
            default_emission_factor, fuel_specific, chp_allocation,
            efficiency_ratio, energy_balance, residual_mix,
            weighted_average, direct_measurement).
        status: Calculation outcome (success, failure, partial,
            skipped).
        duration: Calculation wall-clock time in seconds.
        co2e_kg: Total CO2-equivalent emissions in kilograms.
        biogenic_kg: Biogenic CO2 emissions in kilograms.
        fuel_type: Fuel type of the steam/heat source.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_calculation(
        energy_type, method, status, duration,
        co2e_kg, biogenic_kg, fuel_type, tenant_id,
    )


def record_batch(
    status: str,
    size: int,
    tenant_id: str,
) -> None:
    """Record a batch calculation job.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_batch`.

    Args:
        status: Batch job outcome (success, failure, partial).
        size: Number of individual records in the batch.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_batch(status, size, tenant_id)


def record_chp_allocation(
    method: str,
    fuel_type: str,
    tenant_id: str,
) -> None:
    """Record a CHP allocation calculation.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_chp_allocation`.

    Args:
        method: CHP allocation method (efficiency, energy, exergy,
            iea_fixed_heat, iea_fixed_power, carnot, finnish, ppa).
        fuel_type: Fuel type of the CHP plant (natural_gas, coal,
            fuel_oil, biomass, biogas, waste, municipal_solid_waste,
            industrial_waste, wood_chips, wood_pellets, bagasse,
            landfill_gas, mixed, unknown).
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_chp_allocation(method, fuel_type, tenant_id)


def record_uncertainty(
    method: str,
    tenant_id: str,
) -> None:
    """Record an uncertainty analysis run.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_uncertainty`.

    Args:
        method: Statistical method applied (monte_carlo, analytical,
            error_propagation, ipcc_default_uncertainty, bootstrap).
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_uncertainty(method, tenant_id)


def record_compliance_check(
    framework: str,
    status: str,
    tenant_id: str,
) -> None:
    """Record a regulatory compliance check event.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_compliance_check`.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_CHP, EU_ETS,
            EU_EED, EPA_EGRID, IEA).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed, warning).
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_compliance_check(framework, status, tenant_id)


def set_active_facilities(
    facility_type: str,
    count: int,
    tenant_id: str,
) -> None:
    """Set the gauge for currently active facilities.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.set_active_facilities`.

    Args:
        facility_type: Type of facility (chp_plant, district_heating,
            district_cooling, industrial_boiler, waste_heat_recovery,
            geothermal, solar_thermal, heat_pump, biomass_plant).
        count: Number of currently active facilities.  Must be >= 0.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().set_active_facilities(facility_type, count, tenant_id)


def record_db_lookup(
    lookup_type: str,
    status: str,
) -> None:
    """Record a database lookup operation.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_db_lookup`.

    Args:
        lookup_type: Category of data being looked up
            (emission_factor, fuel_property, efficiency_default,
            chp_parameter, grid_factor, conversion_factor).
        status: Lookup outcome (hit, miss, error, timeout).
    """
    get_metrics().record_db_lookup(lookup_type, status)


def record_error(
    engine: str,
    error_type: str,
    tenant_id: str,
) -> None:
    """Record an error event.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_error`.

    Args:
        engine: Engine component that produced the error (database,
            calculator, allocator, validator, compliance,
            uncertainty, pipeline).
        error_type: Classification of the error (validation_error,
            calculation_error, database_error, configuration_error,
            timeout_error, allocation_error, efficiency_error,
            unit_conversion_error, unknown_error).
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_error(engine, error_type, tenant_id)


def record_emissions(
    energy_type: str,
    fuel_type: str,
    co2e_kg: float,
    tenant_id: str,
) -> None:
    """Record emissions by energy type and fuel type.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_emissions`.

    Args:
        energy_type: Type of purchased thermal energy (steam,
            hot_water, chilled_water, district_heat, district_cooling,
            process_heat, low_pressure_steam, medium_pressure_steam,
            high_pressure_steam, condensate_return).
        fuel_type: Fuel type of the steam/heat source (natural_gas,
            coal, fuel_oil, biomass, biogas, waste,
            municipal_solid_waste, industrial_waste, wood_chips,
            wood_pellets, bagasse, landfill_gas, mixed, unknown).
        co2e_kg: Emission amount in kilograms CO2e.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_emissions(energy_type, fuel_type, co2e_kg, tenant_id)


def record_biogenic_emissions(
    fuel_type: str,
    biogenic_kg: float,
    tenant_id: str,
) -> None:
    """Record biogenic CO2 emissions separately.

    Convenience wrapper around
    :meth:`SteamHeatPurchaseMetrics.record_biogenic_emissions`.

    Args:
        fuel_type: Biomass fuel type (biomass, biogas, wood_chips,
            wood_pellets, bagasse, landfill_gas, waste,
            municipal_solid_waste).
        biogenic_kg: Biogenic CO2 amount in kilograms.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_biogenic_emissions(
        fuel_type, biogenic_kg, tenant_id
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Singleton class
    "SteamHeatPurchaseMetrics",
    # Module-level convenience accessor
    "get_metrics",
    # Module-level reset (testing)
    "reset",
    # Module-level helper functions
    "record_calculation",
    "record_batch",
    "record_chp_allocation",
    "record_uncertainty",
    "record_compliance_check",
    "set_active_facilities",
    "record_db_lookup",
    "record_error",
    "record_emissions",
    "record_biogenic_emissions",
]
