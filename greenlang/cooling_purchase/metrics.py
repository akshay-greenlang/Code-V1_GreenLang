# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-012: Cooling Purchase Agent

12 Prometheus metrics for cooling purchase emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_cp_`` prefix (GreenLang Cooling Purchase)
for consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_cp_calculations_total                     (Counter,   labels: technology, calculation_type, tier, tenant_id, status)
    2.  gl_cp_calculation_duration_seconds           (Histogram, labels: technology, calculation_type, tier)
    3.  gl_cp_emissions_kgco2e_total                 (Counter,   labels: technology, calculation_type, tenant_id)
    4.  gl_cp_cooling_output_kwh_th_total            (Counter,   labels: technology, tenant_id)
    5.  gl_cp_cop_used                               (Histogram, labels: technology, condenser_type)
    6.  gl_cp_batch_size                             (Histogram, labels: tenant_id)
    7.  gl_cp_uncertainty_runs_total                  (Counter,   labels: method, tier)
    8.  gl_cp_compliance_checks_total                (Counter,   labels: framework, status)
    9.  gl_cp_database_queries_total                 (Counter,   labels: operation, table)
    10. gl_cp_database_query_duration_seconds        (Histogram, labels: operation, table)
    11. gl_cp_errors_total                           (Counter,   labels: error_type, operation)
    12. gl_cp_refrigerant_leakage_kgco2e_total       (Counter,   labels: refrigerant, tenant_id)

Label Values Reference:
    technology:
        electric_chiller, absorption_chiller, district_cooling,
        free_cooling, thermal_energy_storage, air_cooled_chiller,
        water_cooled_chiller, evaporative_cooling, magnetic_bearing,
        centrifugal_chiller, screw_chiller, scroll_chiller,
        reciprocating_chiller, hybrid_chiller, ice_storage,
        chilled_water_storage, phase_change_material, unknown.
    calculation_type:
        electric, absorption, district, free_cooling, tes, batch.
    tier:
        tier_1, tier_2, tier_3, default.
    status (calculations_total):
        success, failure, partial, skipped.
    status (compliance_checks_total):
        compliant, non_compliant, partial, not_assessed, warning.
    condenser_type:
        air_cooled, water_cooled, evaporative, hybrid,
        ground_source, seawater, river_water, unknown.
    method (uncertainty_runs_total):
        monte_carlo, analytical.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, ASHRAE_90_1,
        EU_EED, EU_ETS, LEED, BREEAM, IEA.
    operation (database_queries_total, database_query_duration_seconds):
        read, write.
    table:
        cooling_factors, technology_defaults, refrigerant_gwp,
        cop_reference, tenant_config, calculation_results,
        compliance_records, emission_factors, grid_factors.
    error_type:
        validation, calculation, database, auth.
    operation (errors_total):
        intake, calculation, compliance, batch, uncertainty,
        database, refrigerant, pipeline.
    refrigerant:
        R-134a, R-410A, R-407C, R-32, R-1234yf, R-1234ze,
        R-290, R-717, R-744, R-404A, R-507A, R-22, R-123,
        R-245fa, R-1233zd, R-513A, R-514A, R-515B,
        CO2, NH3, propane, unknown.
    tenant_id:
        Free-form string identifying the tenant for multi-tenant
        isolation and per-tenant metric aggregation.

Example:
    Using the CoolingPurchaseMetrics singleton class:

    >>> from greenlang.cooling_purchase.metrics import CoolingPurchaseMetrics
    >>> metrics = CoolingPurchaseMetrics()
    >>> metrics.record_calculation(
    ...     "electric_chiller", "electric", "tier_1", "tenant-001",
    ...     "success", 0.042, 150.3, 500.0, 5.2, "water_cooled"
    ... )
    >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
    >>> metrics.record_refrigerant_leakage("R-134a", "tenant-001", 42.5)

    Using the module-level convenience function:

    >>> from greenlang.cooling_purchase.metrics import get_metrics
    >>> m = get_metrics()
    >>> m.record_calculation(
    ...     "absorption_chiller", "absorption", "tier_2", "tenant-002",
    ...     "success", 0.065, 45.0, 200.0, 1.2, "air_cooled"
    ... )
    >>> m.record_error("validation", "intake")
    >>> summary = m.get_metrics_summary()

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
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
        Histogram,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    logger.info(
        "prometheus_client not installed; "
        "Cooling Purchase emissions metrics disabled"
    )


# ---------------------------------------------------------------------------
# CoolingPurchaseMetrics - Thread-safe singleton with graceful fallback
# ---------------------------------------------------------------------------


class CoolingPurchaseMetrics:
    """Thread-safe singleton for all Cooling Purchase Prometheus metrics.

    The :class:`CoolingPurchaseMetrics` encapsulates the full set of 12
    Prometheus metrics used by the Cooling Purchase Emissions Agent
    (AGENT-MRV-012, GL-MRV-X-023).  It implements the singleton pattern
    so that only one instance (and therefore one set of Prometheus
    collectors) is created per process, regardless of how many callers
    instantiate it.

    When ``prometheus_client`` is not installed, every recording method
    becomes a silent no-op -- no exceptions are raised and no state is
    mutated.  This allows agent code to call metrics unconditionally
    without guarding on library availability.

    An optional ``registry`` parameter can be supplied to direct all
    metrics into a custom :class:`CollectorRegistry` (useful for testing
    or multi-registry deployments).  When ``None`` (the default), the
    process-wide default registry is used.

    The Cooling Purchase Agent covers Scope 2 indirect emissions from
    purchased cooling energy, including electric chillers, absorption
    chillers, district cooling systems, free cooling, and thermal energy
    storage (TES).  Metrics track calculation throughput, COP (Coefficient
    of Performance) distributions, cooling output in kWh_th, refrigerant
    leakage emissions, and regulatory compliance status across multiple
    frameworks (GHG Protocol, ISO 14064, CSRD ESRS E1, ASHRAE 90.1,
    EU EED, EU ETS, LEED, BREEAM, IEA).

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
        >>> metrics = CoolingPurchaseMetrics()
        >>> assert metrics is CoolingPurchaseMetrics()  # same instance
        >>> metrics.record_calculation(
        ...     "electric_chiller", "electric", "tier_1", "tenant-001",
        ...     "success", 0.042, 150.3, 500.0, 5.2, "water_cooled"
        ... )
        >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
        >>> metrics.record_refrigerant_leakage("R-134a", "tenant-001", 42.5)
        >>> summary = metrics.get_metrics_summary()
    """

    _instance: Optional[CoolingPurchaseMetrics] = None
    _lock: threading.RLock = threading.RLock()

    # -- Singleton construction ---------------------------------------------

    def __new__(
        cls,
        registry: Any = None,
    ) -> CoolingPurchaseMetrics:
        """Return the singleton CoolingPurchaseMetrics instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Args:
            registry: Optional :class:`CollectorRegistry` for metric
                registration.  Ignored after the first construction.

        Returns:
            The singleton :class:`CoolingPurchaseMetrics` instance.
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
            "CoolingPurchaseMetrics singleton initialized "
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

        # 1. gl_cp_calculations_total
        #    Total cooling purchase emission calculations performed,
        #    segmented by cooling technology, calculation type (electric,
        #    absorption, district, free_cooling, tes, batch), IPCC tier
        #    level, tenant, and completion status.  This is the primary
        #    throughput counter for the agent and drives SLI calculations
        #    for availability and error rate monitoring.
        self.calculations_total = Counter(
            "gl_cp_calculations_total",
            "Total cooling purchase emission calculations performed",
            labelnames=[
                "technology",
                "calculation_type",
                "tier",
                "tenant_id",
                "status",
            ],
            **reg_kwargs,
        )

        # 2. gl_cp_calculation_duration_seconds
        #    Duration histogram for cooling purchase emission calculations,
        #    segmented by cooling technology, calculation type, and IPCC
        #    tier level.  Bucket boundaries are tuned for typical
        #    calculation latencies ranging from 1ms (simple default EF
        #    lookup) to 5s (complex absorption chiller with Monte Carlo
        #    uncertainty quantification and multi-source reconciliation).
        self.calculation_duration_seconds = Histogram(
            "gl_cp_calculation_duration_seconds",
            "Duration of cooling purchase calculation operations "
            "in seconds",
            labelnames=[
                "technology",
                "calculation_type",
                "tier",
            ],
            buckets=(
                0.001, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            ),
            **reg_kwargs,
        )

        # 3. gl_cp_emissions_kgco2e_total
        #    Cumulative CO2-equivalent emissions in kilograms from
        #    cooling purchases, segmented by cooling technology,
        #    calculation type, and tenant.  Uses kilograms as the base
        #    unit for precision in facility-level calculations.
        #    Emissions include both electricity-related indirect
        #    emissions (Scope 2) and any direct refrigerant leakage
        #    contributions allocated to the cooling purchase.
        self.emissions_kgco2e_total = Counter(
            "gl_cp_emissions_kgco2e_total",
            "Cumulative cooling purchase emissions in kg CO2e "
            "by technology and calculation type",
            labelnames=[
                "technology",
                "calculation_type",
                "tenant_id",
            ],
            **reg_kwargs,
        )

        # 4. gl_cp_cooling_output_kwh_th_total
        #    Cumulative cooling energy output in kilowatt-hours thermal
        #    (kWh_th), segmented by cooling technology and tenant.
        #    Tracks the total useful cooling delivered by purchased
        #    cooling systems.  This metric is essential for calculating
        #    emission intensity ratios (kgCO2e per kWh_th of cooling)
        #    and for benchmarking cooling system performance against
        #    ASHRAE 90.1 and LEED baselines.
        self.cooling_output_kwh_th_total = Counter(
            "gl_cp_cooling_output_kwh_th_total",
            "Cumulative cooling energy output in kWh_th "
            "by technology",
            labelnames=[
                "technology",
                "tenant_id",
            ],
            **reg_kwargs,
        )

        # 5. gl_cp_cop_used
        #    Histogram of Coefficient of Performance (COP) values used
        #    in cooling purchase emission calculations, segmented by
        #    cooling technology and condenser type.  COP is the ratio
        #    of cooling output to energy input and is a key efficiency
        #    metric for cooling systems.  Typical COP ranges:
        #    - Air-cooled electric chillers: 2.5 - 4.5
        #    - Water-cooled electric chillers: 4.0 - 7.0
        #    - Absorption chillers (single-effect): 0.6 - 0.8
        #    - Absorption chillers (double-effect): 1.0 - 1.4
        #    - District cooling: 3.0 - 6.0
        #    - Free cooling: 10.0 - 30.0 (effective COP)
        #    Bucket boundaries span from 0.5 (low-efficiency absorption)
        #    to 30.0 (highly efficient free cooling / economizer mode).
        self.cop_used = Histogram(
            "gl_cp_cop_used",
            "Distribution of COP (Coefficient of Performance) "
            "values used in cooling calculations",
            labelnames=[
                "technology",
                "condenser_type",
            ],
            buckets=(
                0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 10.0, 15.0, 20.0, 30.0,
            ),
            **reg_kwargs,
        )

        # 6. gl_cp_batch_size
        #    Histogram of batch sizes (number of records per batch
        #    calculation), segmented by tenant.  Batch calculations
        #    process multiple cooling systems, facilities, or time
        #    periods in a single operation for throughput optimization.
        #    Bucket boundaries range from single-record batches to
        #    bulk operations of 10,000+ records, calibrated against
        #    typical facility portfolio sizes for district cooling
        #    networks and multi-site cooling inventories.
        self.batch_size = Histogram(
            "gl_cp_batch_size",
            "Distribution of batch calculation sizes "
            "(number of records per batch)",
            labelnames=[
                "tenant_id",
            ],
            buckets=(
                1.0, 5.0, 10.0, 50.0, 100.0,
                500.0, 1000.0, 5000.0, 10000.0,
            ),
            **reg_kwargs,
        )

        # 7. gl_cp_uncertainty_runs_total
        #    Total uncertainty analysis runs for cooling purchase
        #    emission calculations, segmented by statistical method
        #    (monte_carlo or analytical) and IPCC tier level.
        #    Uncertainty quantification follows IPCC 2006 Guidelines
        #    Volume 1, Chapter 3 and the GHG Protocol guidance on
        #    uncertainty assessment.  Monte Carlo simulations are
        #    used for Tier 2/3 calculations with correlated inputs,
        #    while analytical (error propagation) methods are used
        #    for simpler Tier 1 calculations.
        self.uncertainty_runs_total = Counter(
            "gl_cp_uncertainty_runs_total",
            "Total uncertainty analysis runs by method and tier",
            labelnames=[
                "method",
                "tier",
            ],
            **reg_kwargs,
        )

        # 8. gl_cp_compliance_checks_total
        #    Compliance checks against regulatory frameworks, segmented
        #    by framework name and pass/fail status.  Supports GHG
        #    Protocol Corporate Standard, ISO 14064-1, CSRD ESRS E1,
        #    ASHRAE Standard 90.1 (energy efficiency), EU Energy
        #    Efficiency Directive (EED), EU Emissions Trading System
        #    (ETS), LEED, BREEAM, and IEA requirements.  Results
        #    drive the compliance dashboard and can trigger automated
        #    remediation workflows when non-compliance is detected.
        self.compliance_checks_total = Counter(
            "gl_cp_compliance_checks_total",
            "Total compliance checks by framework and status",
            labelnames=[
                "framework",
                "status",
            ],
            **reg_kwargs,
        )

        # 9. gl_cp_database_queries_total
        #    Database query operations for cooling emission factors,
        #    technology defaults, refrigerant GWP values, COP reference
        #    data, and other reference tables, segmented by operation
        #    type (read/write) and target table.  Tracks data layer
        #    throughput for capacity planning and performance monitoring.
        self.database_queries_total = Counter(
            "gl_cp_database_queries_total",
            "Total database queries by operation type and table",
            labelnames=[
                "operation",
                "table",
            ],
            **reg_kwargs,
        )

        # 10. gl_cp_database_query_duration_seconds
        #     Duration histogram for database query operations, segmented
        #     by operation type (read/write) and target table.  Bucket
        #     boundaries range from 1ms (cached COP lookup) to 1s
        #     (complex multi-join queries for historical emission factor
        #     retrieval with temporal validity checks).  Monitors data
        #     layer latency for SLO compliance and query optimization.
        self.database_query_duration_seconds = Histogram(
            "gl_cp_database_query_duration_seconds",
            "Duration of database query operations in seconds",
            labelnames=[
                "operation",
                "table",
            ],
            buckets=(
                0.001, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0,
            ),
            **reg_kwargs,
        )

        # 11. gl_cp_errors_total
        #     Error events by error classification and operation context.
        #     Used for operational alerting, error budget tracking
        #     (OBS-005 SLO integration), and root cause analysis across
        #     the cooling purchase agent pipeline.  Error types cover
        #     validation failures (malformed input), calculation errors
        #     (arithmetic/overflow), database errors (connection/query),
        #     and authentication/authorization errors (tenant access).
        self.errors_total = Counter(
            "gl_cp_errors_total",
            "Total cooling purchase agent errors by error type "
            "and operation",
            labelnames=[
                "error_type",
                "operation",
            ],
            **reg_kwargs,
        )

        # 12. gl_cp_refrigerant_leakage_kgco2e_total
        #     Cumulative CO2-equivalent emissions in kilograms from
        #     refrigerant leakage in cooling systems, segmented by
        #     refrigerant type and tenant.  Refrigerant leakage is a
        #     significant source of direct GHG emissions from cooling
        #     systems, particularly for high-GWP HFC refrigerants
        #     (R-134a GWP=1430, R-410A GWP=2088, R-404A GWP=3922).
        #     This metric tracks leakage emissions separately from
        #     electricity-related Scope 2 emissions to support
        #     refrigerant management programmes, EU F-Gas Regulation
        #     compliance, and Kigali Amendment phase-down tracking.
        self.refrigerant_leakage_kgco2e_total = Counter(
            "gl_cp_refrigerant_leakage_kgco2e_total",
            "Cumulative refrigerant leakage emissions in kg CO2e "
            "by refrigerant type",
            labelnames=[
                "refrigerant",
                "tenant_id",
            ],
            **reg_kwargs,
        )

    def _init_noop_metrics(self) -> None:
        """Set all metric attributes to ``None`` for the no-op fallback.

        When ``prometheus_client`` is not installed, every attribute is
        ``None`` and every ``record_*`` method returns immediately
        without side effects.
        """
        self.calculations_total = None  # type: ignore[assignment]
        self.calculation_duration_seconds = None  # type: ignore[assignment]
        self.emissions_kgco2e_total = None  # type: ignore[assignment]
        self.cooling_output_kwh_th_total = None  # type: ignore[assignment]
        self.cop_used = None  # type: ignore[assignment]
        self.batch_size = None  # type: ignore[assignment]
        self.uncertainty_runs_total = None  # type: ignore[assignment]
        self.compliance_checks_total = None  # type: ignore[assignment]
        self.database_queries_total = None  # type: ignore[assignment]
        self.database_query_duration_seconds = None  # type: ignore[assignment]
        self.errors_total = None  # type: ignore[assignment]
        self.refrigerant_leakage_kgco2e_total = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Recording methods -- Calculations
    # -----------------------------------------------------------------------

    def record_calculation(
        self,
        technology: str,
        calculation_type: str,
        tier: str,
        tenant_id: str,
        status: str,
        duration_s: float,
        emissions_kgco2e: float,
        cooling_kwh_th: float,
        cop_used: float,
        condenser_type: str,
    ) -> None:
        """Record a cooling purchase emission calculation.

        This is the primary recording method that atomically updates
        five metrics at once: the calculation counter, the duration
        histogram, the cumulative emissions counter, the cooling output
        counter, and the COP histogram.

        A single call to this method captures the full lifecycle of one
        cooling emission calculation, providing correlated data points
        for throughput monitoring (calculations_total), latency tracking
        (calculation_duration_seconds), emission accounting
        (emissions_kgco2e_total), energy output tracking
        (cooling_output_kwh_th_total), and efficiency benchmarking
        (cop_used).

        Args:
            technology: Cooling technology used (electric_chiller,
                absorption_chiller, district_cooling, free_cooling,
                thermal_energy_storage, air_cooled_chiller,
                water_cooled_chiller, evaporative_cooling,
                magnetic_bearing, centrifugal_chiller, screw_chiller,
                scroll_chiller, reciprocating_chiller, hybrid_chiller,
                ice_storage, chilled_water_storage,
                phase_change_material, unknown).
            calculation_type: Calculation methodology applied (electric,
                absorption, district, free_cooling, tes, batch).
            tier: IPCC tier level used for the calculation (tier_1,
                tier_2, tier_3, default).
            tenant_id: Tenant identifier for multi-tenant isolation.
            status: Calculation outcome (success, failure, partial,
                skipped).
            duration_s: Calculation wall-clock time in seconds.
                Histogram buckets: 0.001, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0.
            emissions_kgco2e: Total CO2-equivalent emissions in
                kilograms resulting from this calculation.  Must be
                >= 0.  Includes both electricity-related and any
                allocated refrigerant leakage emissions.
            cooling_kwh_th: Cooling energy output in kilowatt-hours
                thermal (kWh_th) for the calculation.  Must be >= 0.
            cop_used: Coefficient of Performance value used in the
                calculation.  Typical range 0.5 (single-effect
                absorption) to 30.0 (free cooling / economizer).
                Pass 0.0 when COP is not applicable (e.g. district
                cooling with direct emission factor).
            condenser_type: Type of condenser used (air_cooled,
                water_cooled, evaporative, hybrid, ground_source,
                seawater, river_water, unknown).
        """
        if not self.prometheus_available:
            return

        self.calculations_total.labels(
            technology=technology,
            calculation_type=calculation_type,
            tier=tier,
            tenant_id=tenant_id,
            status=status,
        ).inc()

        self.calculation_duration_seconds.labels(
            technology=technology,
            calculation_type=calculation_type,
            tier=tier,
        ).observe(duration_s)

        if emissions_kgco2e > 0:
            self.emissions_kgco2e_total.labels(
                technology=technology,
                calculation_type=calculation_type,
                tenant_id=tenant_id,
            ).inc(emissions_kgco2e)

        if cooling_kwh_th > 0:
            self.cooling_output_kwh_th_total.labels(
                technology=technology,
                tenant_id=tenant_id,
            ).inc(cooling_kwh_th)

        if cop_used > 0:
            self.cop_used.labels(
                technology=technology,
                condenser_type=condenser_type,
            ).observe(cop_used)

    # -----------------------------------------------------------------------
    # Recording methods -- Batch calculations
    # -----------------------------------------------------------------------

    def record_batch(
        self,
        tenant_id: str,
        batch_size: int,
    ) -> None:
        """Record a batch calculation job.

        Updates the batch size histogram to track the distribution
        of batch sizes across cooling purchase calculations.  Batch
        calculations process multiple cooling systems, facilities,
        or time periods in a single operation to improve throughput
        and reduce per-request overhead.

        Use the :meth:`record_calculation` method with
        ``calculation_type="batch"`` to record individual calculation
        outcomes within the batch.  This method focuses solely on
        the batch size distribution for capacity planning and
        workload characterization.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation.
            batch_size: Number of individual records in the batch.
                Must be >= 1.  Histogram buckets: 1, 5, 10, 50, 100,
                500, 1000, 5000, 10000.
        """
        if not self.prometheus_available:
            return

        self.batch_size.labels(
            tenant_id=tenant_id,
        ).observe(batch_size)

    # -----------------------------------------------------------------------
    # Recording methods -- Uncertainty analysis
    # -----------------------------------------------------------------------

    def record_uncertainty(
        self,
        method: str,
        tier: str,
    ) -> None:
        """Record an uncertainty analysis run.

        Uncertainty quantification is performed following IPCC 2006
        Guidelines Volume 1, Chapter 3 and the GHG Protocol guidance
        on uncertainty assessment.  This counter tracks each run to
        monitor analytical workload and method preference.

        For cooling purchase calculations, uncertainty arises from
        several sources:
        - COP measurement uncertainty (manufacturer specs vs. actual)
        - Grid emission factor uncertainty (location-based vs.
          market-based, temporal variation)
        - Cooling load measurement uncertainty (metering accuracy)
        - Refrigerant charge and leakage rate uncertainty
        - District cooling system allocation uncertainty

        Monte Carlo simulations are typically used for Tier 2/3
        calculations with correlated inputs (e.g. COP varies with
        ambient temperature which also affects cooling load), while
        analytical (error propagation) methods are used for simpler
        Tier 1 calculations with independent uncertainty factors.

        Args:
            method: Statistical method applied (monte_carlo,
                analytical).
            tier: IPCC tier level for which uncertainty is being
                quantified (tier_1, tier_2, tier_3, default).
        """
        if not self.prometheus_available:
            return

        self.uncertainty_runs_total.labels(
            method=method,
            tier=tier,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Compliance checks
    # -----------------------------------------------------------------------

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """Record a regulatory compliance check event.

        Compliance checks validate that cooling purchase emission
        calculations meet the requirements of the specified regulatory
        framework.  Results drive the compliance dashboard and can
        trigger automated remediation workflows when non-compliance
        is detected.

        Supported frameworks and their cooling-specific requirements:
        - GHG Protocol Corporate Standard: Scope 2 purchased cooling
        - ISO 14064-1: Indirect energy emissions category
        - CSRD ESRS E1: Energy consumption and mix disclosure
        - ASHRAE 90.1: Minimum cooling efficiency requirements
        - EU EED: Energy efficiency and district cooling reporting
        - EU ETS: Cooling-related emissions for covered installations
        - LEED: Cooling performance for green building certification
        - BREEAM: Cooling system efficiency assessment
        - IEA: Energy statistics and efficiency indicators

        Args:
            framework: Regulatory framework checked (GHG_PROTOCOL,
                ISO_14064, CSRD_ESRS_E1, ASHRAE_90_1, EU_EED, EU_ETS,
                LEED, BREEAM, IEA).
            status: Compliance status result (compliant, non_compliant,
                partial, not_assessed, warning).
        """
        if not self.prometheus_available:
            return

        self.compliance_checks_total.labels(
            framework=framework,
            status=status,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Database queries
    # -----------------------------------------------------------------------

    def record_db_query(
        self,
        operation: str,
        table: str,
        duration_s: float,
    ) -> None:
        """Record a database query operation.

        Atomically updates both the database query counter and the
        query duration histogram.  Tracks reference data queries for
        cooling emission factors, technology default parameters,
        refrigerant GWP values, COP reference data, grid emission
        factors, and other values required for cooling purchase
        emission calculations.

        The operation/table segmentation enables targeted performance
        monitoring: read-heavy workloads on cooling_factors and
        cop_reference tables are expected during calculation pipelines,
        while write operations to calculation_results and
        compliance_records are expected during result persistence.

        Args:
            operation: Query operation type (read, write).
            table: Target database table (cooling_factors,
                technology_defaults, refrigerant_gwp, cop_reference,
                tenant_config, calculation_results,
                compliance_records, emission_factors, grid_factors).
            duration_s: Query wall-clock time in seconds.  Histogram
                buckets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0.
        """
        if not self.prometheus_available:
            return

        self.database_queries_total.labels(
            operation=operation,
            table=table,
        ).inc()

        self.database_query_duration_seconds.labels(
            operation=operation,
            table=table,
        ).observe(duration_s)

    # -----------------------------------------------------------------------
    # Recording methods -- Errors
    # -----------------------------------------------------------------------

    def record_error(
        self,
        error_type: str,
        operation: str,
    ) -> None:
        """Record an error event for operational alerting.

        Error events are segmented by the error classification and
        the operation context in which the error occurred.  This
        two-dimensional labeling enables targeted alerting rules
        (e.g. alert on database errors exceeding 5/min during
        calculation operations) and root cause analysis by isolating
        errors to specific pipeline stages.

        Error type categories:
        - validation: Input data fails schema or business rule checks
          (e.g. negative COP, missing technology type, invalid tenant)
        - calculation: Arithmetic errors, overflow, division by zero,
          COP out of range, impossible emission factor combinations
        - database: Connection failures, query timeouts, constraint
          violations, missing reference data
        - auth: Authentication failures, authorization denials, tenant
          isolation violations, expired tokens

        Operation contexts:
        - intake: Initial data reception and parsing
        - calculation: Core emission calculation logic
        - compliance: Regulatory compliance checking
        - batch: Batch processing orchestration
        - uncertainty: Uncertainty analysis runs
        - database: Database interaction layer
        - refrigerant: Refrigerant leakage calculation
        - pipeline: End-to-end pipeline orchestration

        Args:
            error_type: Classification of the error (validation,
                calculation, database, auth).
            operation: Operation context where the error occurred
                (intake, calculation, compliance, batch, uncertainty,
                database, refrigerant, pipeline).
        """
        if not self.prometheus_available:
            return

        self.errors_total.labels(
            error_type=error_type,
            operation=operation,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Refrigerant leakage
    # -----------------------------------------------------------------------

    def record_refrigerant_leakage(
        self,
        refrigerant: str,
        tenant_id: str,
        emissions_kgco2e: float,
    ) -> None:
        """Record refrigerant leakage emissions from cooling systems.

        Refrigerant leakage is a significant source of direct GHG
        emissions from cooling systems.  High-GWP hydrofluorocarbon
        (HFC) refrigerants are being phased down under the Kigali
        Amendment to the Montreal Protocol and the EU F-Gas Regulation
        (EU 517/2014).

        This metric tracks leakage emissions separately from
        electricity-related Scope 2 emissions to enable:
        - Refrigerant management programme effectiveness monitoring
        - EU F-Gas Regulation phase-down compliance tracking
        - Kigali Amendment HFC consumption quota monitoring
        - Transition planning from high-GWP to low-GWP refrigerants
        - Leak detection and repair (LDAR) programme ROI assessment

        Common refrigerant GWP values (AR5 100-year):
        - R-134a: 1430 (automotive, centrifugal chillers)
        - R-410A: 2088 (commercial AC, heat pumps)
        - R-407C: 1774 (commercial AC retrofit)
        - R-32: 675 (new commercial AC)
        - R-1234yf: <1 (automotive replacement for R-134a)
        - R-1234ze(E): 7 (centrifugal chiller replacement)
        - R-290 (propane): 3 (small commercial)
        - R-717 (ammonia): 0 (industrial chillers)
        - R-744 (CO2): 1 (transcritical systems)
        - R-404A: 3922 (refrigeration, being phased out)
        - R-22: 1810 (legacy systems, banned for new equipment)

        Args:
            refrigerant: Refrigerant type designation (R-134a, R-410A,
                R-407C, R-32, R-1234yf, R-1234ze, R-290, R-717, R-744,
                R-404A, R-507A, R-22, R-123, R-245fa, R-1233zd, R-513A,
                R-514A, R-515B, CO2, NH3, propane, unknown).
            tenant_id: Tenant identifier for multi-tenant isolation.
            emissions_kgco2e: Refrigerant leakage emissions in
                kilograms CO2e.  Calculated as: leakage_kg * GWP.
                Must be > 0.
        """
        if not self.prometheus_available:
            return

        if emissions_kgco2e > 0:
            self.refrigerant_leakage_kgco2e_total.labels(
                refrigerant=refrigerant,
                tenant_id=tenant_id,
            ).inc(emissions_kgco2e)

    # -----------------------------------------------------------------------
    # Recording methods -- Emissions standalone
    # -----------------------------------------------------------------------

    def record_emissions(
        self,
        technology: str,
        calculation_type: str,
        emissions_kgco2e: float,
        tenant_id: str,
    ) -> None:
        """Record emissions by technology and calculation type.

        Use this method to record emissions outside of the primary
        :meth:`record_calculation` flow, for example when adjusting
        emissions due to COP recalibration, retroactive grid emission
        factor corrections, market-based vs. location-based method
        switching, or supplementary emission sources.

        Args:
            technology: Cooling technology used (electric_chiller,
                absorption_chiller, district_cooling, free_cooling,
                thermal_energy_storage, air_cooled_chiller,
                water_cooled_chiller, evaporative_cooling,
                magnetic_bearing, centrifugal_chiller, screw_chiller,
                scroll_chiller, reciprocating_chiller, hybrid_chiller,
                ice_storage, chilled_water_storage,
                phase_change_material, unknown).
            calculation_type: Calculation methodology applied (electric,
                absorption, district, free_cooling, tes, batch).
            emissions_kgco2e: Emission amount in kilograms CO2e to add
                to the counter.  Must be > 0.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        if emissions_kgco2e > 0:
            self.emissions_kgco2e_total.labels(
                technology=technology,
                calculation_type=calculation_type,
                tenant_id=tenant_id,
            ).inc(emissions_kgco2e)

    # -----------------------------------------------------------------------
    # Recording methods -- Cooling output standalone
    # -----------------------------------------------------------------------

    def record_cooling_output(
        self,
        technology: str,
        cooling_kwh_th: float,
        tenant_id: str,
    ) -> None:
        """Record cooling energy output separately.

        Use this method to record cooling output outside of the primary
        :meth:`record_calculation` flow, for example when processing
        metered cooling data from building management systems (BMS),
        district cooling network telemetry, or supplementary cooling
        load measurements.

        Cooling output is tracked in kilowatt-hours thermal (kWh_th)
        to distinguish thermal energy from electrical energy (kWh_e).
        One ton of refrigeration (TR) equals approximately 3.517 kWh_th.

        Args:
            technology: Cooling technology used (electric_chiller,
                absorption_chiller, district_cooling, free_cooling,
                thermal_energy_storage, air_cooled_chiller,
                water_cooled_chiller, evaporative_cooling,
                magnetic_bearing, centrifugal_chiller, screw_chiller,
                scroll_chiller, reciprocating_chiller, hybrid_chiller,
                ice_storage, chilled_water_storage,
                phase_change_material, unknown).
            cooling_kwh_th: Cooling energy output in kWh_th.
                Must be > 0.
            tenant_id: Tenant identifier for multi-tenant isolation.
        """
        if not self.prometheus_available:
            return

        if cooling_kwh_th > 0:
            self.cooling_output_kwh_th_total.labels(
                technology=technology,
                tenant_id=tenant_id,
            ).inc(cooling_kwh_th)

    # -----------------------------------------------------------------------
    # Recording methods -- COP standalone
    # -----------------------------------------------------------------------

    def record_cop(
        self,
        technology: str,
        condenser_type: str,
        cop_value: float,
    ) -> None:
        """Record a COP observation separately.

        Use this method to record COP values outside of the primary
        :meth:`record_calculation` flow, for example when ingesting
        manufacturer nameplate COP data, field-measured COP values
        from commissioning reports, or seasonal COP (SCOP) values
        from annual performance assessments.

        The COP histogram enables benchmarking of cooling system
        efficiency across the portfolio and identification of
        underperforming equipment that may benefit from maintenance,
        retrofitting, or replacement.

        Args:
            technology: Cooling technology used (electric_chiller,
                absorption_chiller, district_cooling, free_cooling,
                thermal_energy_storage, air_cooled_chiller,
                water_cooled_chiller, evaporative_cooling,
                magnetic_bearing, centrifugal_chiller, screw_chiller,
                scroll_chiller, reciprocating_chiller, hybrid_chiller,
                ice_storage, chilled_water_storage,
                phase_change_material, unknown).
            condenser_type: Type of condenser (air_cooled, water_cooled,
                evaporative, hybrid, ground_source, seawater,
                river_water, unknown).
            cop_value: Coefficient of Performance value.  Histogram
                buckets: 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                10.0, 15.0, 20.0, 30.0.
        """
        if not self.prometheus_available:
            return

        if cop_value > 0:
            self.cop_used.labels(
                technology=technology,
                condenser_type=condenser_type,
            ).observe(cop_value)

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

        The summary includes metadata for each metric (name,
        documentation, type, labels) to support introspection by
        monitoring tools, Grafana dashboard provisioning scripts,
        and automated metric catalog generation.

        Returns:
            Dictionary with one key per metric.  Counter values show
            metric metadata; the Histogram returns its documented
            bucket boundaries and label names.

        Example:
            >>> metrics = CoolingPurchaseMetrics()
            >>> summary = metrics.get_metrics_summary()
            >>> isinstance(summary, dict)
            True
            >>> "gl_cp_calculations_total" in summary
            True
            >>> summary["prometheus_available"] in (True, False)
            True
        """
        if not self.prometheus_available:
            return {
                "gl_cp_calculations_total": None,
                "gl_cp_calculation_duration_seconds": None,
                "gl_cp_emissions_kgco2e_total": None,
                "gl_cp_cooling_output_kwh_th_total": None,
                "gl_cp_cop_used": None,
                "gl_cp_batch_size": None,
                "gl_cp_uncertainty_runs_total": None,
                "gl_cp_compliance_checks_total": None,
                "gl_cp_database_queries_total": None,
                "gl_cp_database_query_duration_seconds": None,
                "gl_cp_errors_total": None,
                "gl_cp_refrigerant_leakage_kgco2e_total": None,
                "prometheus_available": False,
            }

        return {
            "gl_cp_calculations_total": _safe_describe(
                self.calculations_total
            ),
            "gl_cp_calculation_duration_seconds": _safe_describe(
                self.calculation_duration_seconds
            ),
            "gl_cp_emissions_kgco2e_total": _safe_describe(
                self.emissions_kgco2e_total
            ),
            "gl_cp_cooling_output_kwh_th_total": _safe_describe(
                self.cooling_output_kwh_th_total
            ),
            "gl_cp_cop_used": _safe_describe(
                self.cop_used
            ),
            "gl_cp_batch_size": _safe_describe(
                self.batch_size
            ),
            "gl_cp_uncertainty_runs_total": _safe_describe(
                self.uncertainty_runs_total
            ),
            "gl_cp_compliance_checks_total": _safe_describe(
                self.compliance_checks_total
            ),
            "gl_cp_database_queries_total": _safe_describe(
                self.database_queries_total
            ),
            "gl_cp_database_query_duration_seconds": _safe_describe(
                self.database_query_duration_seconds
            ),
            "gl_cp_errors_total": _safe_describe(
                self.errors_total
            ),
            "gl_cp_refrigerant_leakage_kgco2e_total": _safe_describe(
                self.refrigerant_leakage_kgco2e_total
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

    Safely introspects the internal attributes of a ``prometheus_client``
    metric collector to extract metadata for the metrics summary.  Uses
    a try/except guard to handle any unexpected attribute access errors
    gracefully, returning a fallback dict with ``"unknown"`` values.

    Args:
        metric: A ``prometheus_client`` Counter or Histogram instance.

    Returns:
        Dictionary with ``name``, ``documentation``, ``type``, and
        ``labels`` keys.  The ``labels`` key is only present if the
        metric has ``_labelnames`` defined.
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

_default_metrics: Optional[CoolingPurchaseMetrics] = None


def get_metrics() -> CoolingPurchaseMetrics:
    """Return the module-level default :class:`CoolingPurchaseMetrics` instance.

    This function provides a convenient module-level accessor that
    lazily creates and caches a :class:`CoolingPurchaseMetrics`
    singleton.  It is the recommended entry point for agent code that
    does not need a custom :class:`CollectorRegistry`.

    Returns:
        The shared :class:`CoolingPurchaseMetrics` instance.

    Example:
        >>> from greenlang.cooling_purchase.metrics import get_metrics
        >>> m = get_metrics()
        >>> m.record_calculation(
        ...     "electric_chiller", "electric", "tier_1", "tenant-001",
        ...     "success", 0.05, 120.0, 400.0, 5.5, "water_cooled"
        ... )
        >>> m.record_error("validation", "intake")
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = CoolingPurchaseMetrics()
    return _default_metrics


def reset() -> None:
    """Reset the module-level metrics singleton.

    **Testing only.**  Resets both the module-level cached reference
    and the class-level singleton instance.  After calling this
    function, the next call to :func:`get_metrics` or
    ``CoolingPurchaseMetrics()`` will create a completely fresh
    instance with new Prometheus collectors.

    This is essential for test isolation -- without resetting, tests
    that create metrics with a custom :class:`CollectorRegistry`
    would encounter ``Duplicated timeseries`` errors from the
    process-wide default registry.

    Example:
        >>> from greenlang.cooling_purchase.metrics import reset
        >>> reset()  # clears singleton for test isolation
    """
    global _default_metrics
    _default_metrics = None
    CoolingPurchaseMetrics._reset()


# ---------------------------------------------------------------------------
# Module-level helper functions (safe to call without prometheus_client)
# ---------------------------------------------------------------------------
# These thin wrappers delegate to the singleton and mirror the flat-function
# API style used by other MRV metrics modules for backward compatibility.


def record_calculation(
    technology: str,
    calculation_type: str,
    tier: str,
    tenant_id: str,
    status: str,
    duration_s: float,
    emissions_kgco2e: float,
    cooling_kwh_th: float,
    cop_used: float,
    condenser_type: str,
) -> None:
    """Record a cooling purchase emission calculation.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_calculation`.

    Args:
        technology: Cooling technology used (electric_chiller,
            absorption_chiller, district_cooling, free_cooling,
            thermal_energy_storage, air_cooled_chiller,
            water_cooled_chiller, evaporative_cooling,
            magnetic_bearing, centrifugal_chiller, screw_chiller,
            scroll_chiller, reciprocating_chiller, hybrid_chiller,
            ice_storage, chilled_water_storage,
            phase_change_material, unknown).
        calculation_type: Calculation methodology applied (electric,
            absorption, district, free_cooling, tes, batch).
        tier: IPCC tier level (tier_1, tier_2, tier_3, default).
        tenant_id: Tenant identifier for multi-tenant isolation.
        status: Calculation outcome (success, failure, partial,
            skipped).
        duration_s: Calculation wall-clock time in seconds.
        emissions_kgco2e: Total CO2-equivalent emissions in kg.
        cooling_kwh_th: Cooling energy output in kWh_th.
        cop_used: Coefficient of Performance value used.
        condenser_type: Type of condenser used (air_cooled,
            water_cooled, evaporative, hybrid, ground_source,
            seawater, river_water, unknown).
    """
    get_metrics().record_calculation(
        technology, calculation_type, tier, tenant_id,
        status, duration_s, emissions_kgco2e, cooling_kwh_th,
        cop_used, condenser_type,
    )


def record_batch(
    tenant_id: str,
    batch_size: int,
) -> None:
    """Record a batch calculation job.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_batch`.

    Args:
        tenant_id: Tenant identifier for multi-tenant isolation.
        batch_size: Number of individual records in the batch.
    """
    get_metrics().record_batch(tenant_id, batch_size)


def record_uncertainty(
    method: str,
    tier: str,
) -> None:
    """Record an uncertainty analysis run.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_uncertainty`.

    Args:
        method: Statistical method applied (monte_carlo, analytical).
        tier: IPCC tier level (tier_1, tier_2, tier_3, default).
    """
    get_metrics().record_uncertainty(method, tier)


def record_compliance_check(
    framework: str,
    status: str,
) -> None:
    """Record a regulatory compliance check event.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_compliance_check`.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            ISO_14064, CSRD_ESRS_E1, ASHRAE_90_1, EU_EED, EU_ETS,
            LEED, BREEAM, IEA).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed, warning).
    """
    get_metrics().record_compliance_check(framework, status)


def record_db_query(
    operation: str,
    table: str,
    duration_s: float,
) -> None:
    """Record a database query operation.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_db_query`.

    Args:
        operation: Query operation type (read, write).
        table: Target database table (cooling_factors,
            technology_defaults, refrigerant_gwp, cop_reference,
            tenant_config, calculation_results, compliance_records,
            emission_factors, grid_factors).
        duration_s: Query wall-clock time in seconds.
    """
    get_metrics().record_db_query(operation, table, duration_s)


def record_error(
    error_type: str,
    operation: str,
) -> None:
    """Record an error event.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_error`.

    Args:
        error_type: Classification of the error (validation,
            calculation, database, auth).
        operation: Operation context where the error occurred
            (intake, calculation, compliance, batch, uncertainty,
            database, refrigerant, pipeline).
    """
    get_metrics().record_error(error_type, operation)


def record_refrigerant_leakage(
    refrigerant: str,
    tenant_id: str,
    emissions_kgco2e: float,
) -> None:
    """Record refrigerant leakage emissions.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_refrigerant_leakage`.

    Args:
        refrigerant: Refrigerant type designation (R-134a, R-410A,
            R-407C, R-32, R-1234yf, R-1234ze, R-290, R-717, R-744,
            R-404A, R-507A, R-22, R-123, R-245fa, R-1233zd, R-513A,
            R-514A, R-515B, CO2, NH3, propane, unknown).
        tenant_id: Tenant identifier for multi-tenant isolation.
        emissions_kgco2e: Refrigerant leakage emissions in kg CO2e.
    """
    get_metrics().record_refrigerant_leakage(
        refrigerant, tenant_id, emissions_kgco2e
    )


def record_emissions(
    technology: str,
    calculation_type: str,
    emissions_kgco2e: float,
    tenant_id: str,
) -> None:
    """Record emissions by technology and calculation type.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_emissions`.

    Args:
        technology: Cooling technology used (electric_chiller,
            absorption_chiller, district_cooling, free_cooling,
            thermal_energy_storage, air_cooled_chiller,
            water_cooled_chiller, evaporative_cooling,
            magnetic_bearing, centrifugal_chiller, screw_chiller,
            scroll_chiller, reciprocating_chiller, hybrid_chiller,
            ice_storage, chilled_water_storage,
            phase_change_material, unknown).
        calculation_type: Calculation methodology applied (electric,
            absorption, district, free_cooling, tes, batch).
        emissions_kgco2e: Emission amount in kilograms CO2e.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_emissions(
        technology, calculation_type, emissions_kgco2e, tenant_id
    )


def record_cooling_output(
    technology: str,
    cooling_kwh_th: float,
    tenant_id: str,
) -> None:
    """Record cooling energy output separately.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_cooling_output`.

    Args:
        technology: Cooling technology used (electric_chiller,
            absorption_chiller, district_cooling, free_cooling,
            thermal_energy_storage, air_cooled_chiller,
            water_cooled_chiller, evaporative_cooling,
            magnetic_bearing, centrifugal_chiller, screw_chiller,
            scroll_chiller, reciprocating_chiller, hybrid_chiller,
            ice_storage, chilled_water_storage,
            phase_change_material, unknown).
        cooling_kwh_th: Cooling energy output in kWh_th.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """
    get_metrics().record_cooling_output(
        technology, cooling_kwh_th, tenant_id
    )


def record_cop(
    technology: str,
    condenser_type: str,
    cop_value: float,
) -> None:
    """Record a COP observation separately.

    Convenience wrapper around
    :meth:`CoolingPurchaseMetrics.record_cop`.

    Args:
        technology: Cooling technology used (electric_chiller,
            absorption_chiller, district_cooling, free_cooling,
            thermal_energy_storage, air_cooled_chiller,
            water_cooled_chiller, evaporative_cooling,
            magnetic_bearing, centrifugal_chiller, screw_chiller,
            scroll_chiller, reciprocating_chiller, hybrid_chiller,
            ice_storage, chilled_water_storage,
            phase_change_material, unknown).
        condenser_type: Type of condenser (air_cooled, water_cooled,
            evaporative, hybrid, ground_source, seawater,
            river_water, unknown).
        cop_value: Coefficient of Performance value.
    """
    get_metrics().record_cop(technology, condenser_type, cop_value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Singleton class
    "CoolingPurchaseMetrics",
    # Module-level convenience accessor
    "get_metrics",
    # Module-level reset (testing)
    "reset",
    # Module-level helper functions
    "record_calculation",
    "record_batch",
    "record_uncertainty",
    "record_compliance_check",
    "record_db_query",
    "record_error",
    "record_refrigerant_leakage",
    "record_emissions",
    "record_cooling_output",
    "record_cop",
]
