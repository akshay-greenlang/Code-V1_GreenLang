# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-007: Waste Treatment Emissions Agent

12 Prometheus metrics for on-site waste treatment emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_wt_`` prefix (GreenLang Waste Treatment) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_wt_calculations_total              (Counter,   labels: treatment_method, calculation_method, waste_category)
    2.  gl_wt_calculation_duration_seconds    (Histogram, labels: treatment_method, calculation_method)
    3.  gl_wt_calculation_errors_total        (Counter,   labels: error_type)
    4.  gl_wt_emissions_tco2e_total           (Counter,   labels: gas, treatment_method, waste_category)
    5.  gl_wt_waste_processed_tonnes_total    (Counter,   labels: treatment_method, waste_category)
    6.  gl_wt_methane_recovery_tonnes_total   (Counter,   labels: recovery_type)
    7.  gl_wt_energy_recovered_gj_total       (Counter,   labels: energy_type)
    8.  gl_wt_biological_treatments_total     (Counter,   labels: bio_type)
    9.  gl_wt_thermal_treatments_total        (Counter,   labels: thermal_type)
    10. gl_wt_compliance_checks_total         (Counter,   labels: framework, status)
    11. gl_wt_uncertainty_runs_total          (Counter,   labels: method)
    12. gl_wt_active_facilities               (Gauge,     labels: tenant_id)

Label Values Reference:
    treatment_method:
        landfill, incineration, composting, anaerobic_digestion,
        mechanical_biological_treatment, open_burning, pyrolysis,
        gasification, wastewater_treatment, recycling.
    calculation_method:
        ipcc_default, first_order_decay, mass_balance, direct_measurement,
        emission_factor, site_specific.
    waste_category:
        municipal_solid_waste, industrial_waste, clinical_waste,
        construction_demolition, hazardous_waste, agricultural_waste,
        sewage_sludge, electronic_waste, food_waste, garden_waste.
    gas:
        CO2, CH4, N2O, CO, biogenic_CO2.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, unknown_error.
    recovery_type:
        captured, flared, utilized, vented.
    energy_type:
        electricity, heat.
    bio_type:
        composting, ad, mbt.
    thermal_type:
        incineration, pyrolysis, gasification.
    framework:
        GHG_PROTOCOL, IPCC_2006, CSRD_ESRS_E1, EU_WASTE_DIRECTIVE,
        UK_SECR, EPA_GHGRP, UNFCCC.
    status:
        compliant, non_compliant, partial, not_assessed.
    method:
        monte_carlo, analytical, error_propagation,
        ipcc_default_uncertainty.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     track_active_facilities,
    ... )
    >>> record_calculation("landfill", "first_order_decay", "municipal_solid_waste")
    >>> record_emissions("CH4", "landfill", "municipal_solid_waste", 42.5)
    >>> track_active_facilities("tenant_001", 7)

    Using the MetricsCollector singleton:

    >>> from greenlang.agents.mrv.waste_treatment_emissions.metrics import MetricsCollector
    >>> collector = MetricsCollector()
    >>> collector.record_calculation("incineration", "mass_balance", "clinical_waste")
    >>> collector.record_emissions("CO2", "incineration", "clinical_waste", 120.0)
    >>> collector.track_active_facilities("tenant_002", 3)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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
        "prometheus_client not installed; waste treatment emissions metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by treatment method, calculation method, and waste category
    wt_calculations_total = Counter(
        "gl_wt_calculations_total",
        "Total waste treatment emission calculations performed",
        labelnames=["treatment_method", "calculation_method", "waste_category"],
    )

    # 2. Calculation duration histogram by treatment method and calculation method
    wt_calculation_duration_seconds = Histogram(
        "gl_wt_calculation_duration_seconds",
        "Duration of waste treatment emission calculation operations in seconds",
        labelnames=["treatment_method", "calculation_method"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 3. Calculation errors by error type
    wt_calculation_errors_total = Counter(
        "gl_wt_calculation_errors_total",
        "Total waste treatment emission calculation errors by error type",
        labelnames=["error_type"],
    )

    # 4. Cumulative emissions by gas, treatment method, and waste category
    wt_emissions_tco2e_total = Counter(
        "gl_wt_emissions_tco2e_total",
        "Cumulative waste treatment emissions in tCO2e by gas, treatment method, "
        "and waste category",
        labelnames=["gas", "treatment_method", "waste_category"],
    )

    # 5. Cumulative waste processed by treatment method and waste category
    wt_waste_processed_tonnes_total = Counter(
        "gl_wt_waste_processed_tonnes_total",
        "Cumulative waste processed in tonnes by treatment method and waste category",
        labelnames=["treatment_method", "waste_category"],
    )

    # 6. Methane recovery by recovery type (captured/flared/utilized/vented)
    wt_methane_recovery_tonnes_total = Counter(
        "gl_wt_methane_recovery_tonnes_total",
        "Cumulative methane recovery in tonnes by recovery pathway",
        labelnames=["recovery_type"],
    )

    # 7. Energy recovered by energy type (electricity/heat)
    wt_energy_recovered_gj_total = Counter(
        "gl_wt_energy_recovered_gj_total",
        "Cumulative energy recovered in gigajoules by energy type",
        labelnames=["energy_type"],
    )

    # 8. Biological treatment events by biological treatment type
    wt_biological_treatments_total = Counter(
        "gl_wt_biological_treatments_total",
        "Total biological waste treatment events by treatment type",
        labelnames=["bio_type"],
    )

    # 9. Thermal treatment events by thermal treatment type
    wt_thermal_treatments_total = Counter(
        "gl_wt_thermal_treatments_total",
        "Total thermal waste treatment events by treatment type",
        labelnames=["thermal_type"],
    )

    # 10. Compliance checks by framework and status
    wt_compliance_checks_total = Counter(
        "gl_wt_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 11. Uncertainty analysis runs by method
    wt_uncertainty_runs_total = Counter(
        "gl_wt_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 12. Currently active waste treatment facilities by tenant
    wt_active_facilities = Gauge(
        "gl_wt_active_facilities",
        "Number of currently active waste treatment facilities by tenant",
        labelnames=["tenant_id"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    wt_calculations_total = None              # type: ignore[assignment]
    wt_calculation_duration_seconds = None     # type: ignore[assignment]
    wt_calculation_errors_total = None         # type: ignore[assignment]
    wt_emissions_tco2e_total = None            # type: ignore[assignment]
    wt_waste_processed_tonnes_total = None     # type: ignore[assignment]
    wt_methane_recovery_tonnes_total = None    # type: ignore[assignment]
    wt_energy_recovered_gj_total = None        # type: ignore[assignment]
    wt_biological_treatments_total = None      # type: ignore[assignment]
    wt_thermal_treatments_total = None         # type: ignore[assignment]
    wt_compliance_checks_total = None          # type: ignore[assignment]
    wt_uncertainty_runs_total = None           # type: ignore[assignment]
    wt_active_facilities = None               # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_calculation(
    treatment_method: str,
    calculation_method: str,
    waste_category: str,
) -> None:
    """Record a waste treatment emission calculation event.

    Args:
        treatment_method: Waste treatment method used (landfill, incineration,
            composting, anaerobic_digestion, mechanical_biological_treatment,
            open_burning, pyrolysis, gasification, wastewater_treatment,
            recycling).
        calculation_method: Calculation methodology applied (ipcc_default,
            first_order_decay, mass_balance, direct_measurement,
            emission_factor, site_specific).
        waste_category: Category of waste treated (municipal_solid_waste,
            industrial_waste, clinical_waste, construction_demolition,
            hazardous_waste, agricultural_waste, sewage_sludge,
            electronic_waste, food_waste, garden_waste).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_calculations_total.labels(
        treatment_method=treatment_method,
        calculation_method=calculation_method,
        waste_category=waste_category,
    ).inc()


def observe_calculation_duration(
    treatment_method: str,
    calculation_method: str,
    seconds: float,
) -> None:
    """Record the duration of a waste treatment calculation operation.

    Args:
        treatment_method: Waste treatment method used (landfill, incineration,
            composting, anaerobic_digestion, mechanical_biological_treatment,
            open_burning, pyrolysis, gasification, wastewater_treatment,
            recycling).
        calculation_method: Calculation methodology applied (ipcc_default,
            first_order_decay, mass_balance, direct_measurement,
            emission_factor, site_specific).
        seconds: Operation wall-clock time in seconds.
            Buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_calculation_duration_seconds.labels(
        treatment_method=treatment_method,
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
    wt_calculation_errors_total.labels(
        error_type=error_type,
    ).inc()


def record_emissions(
    gas: str,
    treatment_method: str,
    waste_category: str,
    tco2e: float = 1.0,
) -> None:
    """Record cumulative emissions by gas, treatment method, and waste category.

    Args:
        gas: Greenhouse gas species (CO2, CH4, N2O, CO, biogenic_CO2).
        treatment_method: Waste treatment method that produced the emissions
            (landfill, incineration, composting, anaerobic_digestion,
            mechanical_biological_treatment, open_burning, pyrolysis,
            gasification, wastewater_treatment, recycling).
        waste_category: Category of waste treated (municipal_solid_waste,
            industrial_waste, clinical_waste, construction_demolition,
            hazardous_waste, agricultural_waste, sewage_sludge,
            electronic_waste, food_waste, garden_waste).
        tco2e: Emission amount in tonnes CO2e to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_emissions_tco2e_total.labels(
        gas=gas,
        treatment_method=treatment_method,
        waste_category=waste_category,
    ).inc(tco2e)


def record_waste_processed(
    treatment_method: str,
    waste_category: str,
    tonnes: float = 1.0,
) -> None:
    """Record cumulative waste processed by treatment method and waste category.

    Args:
        treatment_method: Waste treatment method applied (landfill,
            incineration, composting, anaerobic_digestion,
            mechanical_biological_treatment, open_burning, pyrolysis,
            gasification, wastewater_treatment, recycling).
        waste_category: Category of waste treated (municipal_solid_waste,
            industrial_waste, clinical_waste, construction_demolition,
            hazardous_waste, agricultural_waste, sewage_sludge,
            electronic_waste, food_waste, garden_waste).
        tonnes: Waste amount in tonnes to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_waste_processed_tonnes_total.labels(
        treatment_method=treatment_method,
        waste_category=waste_category,
    ).inc(tonnes)


def record_methane_recovery(
    recovery_type: str,
    tonnes: float = 1.0,
) -> None:
    """Record cumulative methane recovery by recovery pathway.

    Args:
        recovery_type: Methane recovery pathway (captured, flared,
            utilized, vented).
        tonnes: Methane amount in tonnes to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_methane_recovery_tonnes_total.labels(
        recovery_type=recovery_type,
    ).inc(tonnes)


def record_energy_recovered(
    energy_type: str,
    gigajoules: float = 1.0,
) -> None:
    """Record cumulative energy recovered by energy type.

    Args:
        energy_type: Type of energy recovered (electricity, heat).
        gigajoules: Energy amount in gigajoules to add to the counter.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_energy_recovered_gj_total.labels(
        energy_type=energy_type,
    ).inc(gigajoules)


def record_biological_treatment(bio_type: str) -> None:
    """Record a biological waste treatment event.

    Args:
        bio_type: Type of biological treatment applied (composting,
            ad, mbt). ``ad`` = anaerobic digestion,
            ``mbt`` = mechanical biological treatment.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_biological_treatments_total.labels(
        bio_type=bio_type,
    ).inc()


def record_thermal_treatment(thermal_type: str) -> None:
    """Record a thermal waste treatment event.

    Args:
        thermal_type: Type of thermal treatment applied (incineration,
            pyrolysis, gasification).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_thermal_treatments_total.labels(
        thermal_type=thermal_type,
    ).inc()


def record_compliance_check(framework: str, status: str) -> None:
    """Record a compliance check event.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            IPCC_2006, CSRD_ESRS_E1, EU_WASTE_DIRECTIVE, UK_SECR,
            EPA_GHGRP, UNFCCC).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_compliance_checks_total.labels(
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
    wt_uncertainty_runs_total.labels(
        method=method,
    ).inc()


def track_active_facilities(tenant_id: str, count: int) -> None:
    """Set the gauge for currently active waste treatment facilities by tenant.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        tenant_id: Tenant identifier for the facility count.
        count: Number of active facilities for the tenant. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    wt_active_facilities.labels(
        tenant_id=tenant_id,
    ).set(count)


# ---------------------------------------------------------------------------
# MetricsCollector - Thread-safe singleton facade
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe singleton that wraps all waste treatment Prometheus metrics.

    The :class:`MetricsCollector` provides an object-oriented interface for
    recording waste treatment metrics. Only one instance is created regardless
    of how many times the constructor is invoked, making it safe for use in
    multi-threaded agent pipelines, batch processors, and async workers.

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
        ...     "landfill", "first_order_decay", "municipal_solid_waste"
        ... )
        >>> collector.record_emissions(
        ...     "CH4", "landfill", "municipal_solid_waste", 42.5
        ... )
        >>> collector.track_active_facilities("tenant_001", 7)
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
        treatment_method: str,
        calculation_method: str,
        waste_category: str,
    ) -> None:
        """Record a waste treatment emission calculation event.

        Args:
            treatment_method: Waste treatment method used.
            calculation_method: Calculation methodology applied.
            waste_category: Category of waste treated.
        """
        record_calculation(treatment_method, calculation_method, waste_category)

    def observe_calculation_duration(
        self,
        treatment_method: str,
        calculation_method: str,
        seconds: float,
    ) -> None:
        """Record the duration of a waste treatment calculation.

        Args:
            treatment_method: Waste treatment method used.
            calculation_method: Calculation methodology applied.
            seconds: Operation wall-clock time in seconds.
        """
        observe_calculation_duration(treatment_method, calculation_method, seconds)

    def record_calculation_error(self, error_type: str) -> None:
        """Record a calculation error event.

        Args:
            error_type: Type of error encountered.
        """
        record_calculation_error(error_type)

    # -- Emissions and waste tracking ----------------------------------------

    def record_emissions(
        self,
        gas: str,
        treatment_method: str,
        waste_category: str,
        tco2e: float = 1.0,
    ) -> None:
        """Record cumulative emissions by gas, treatment method, and category.

        Args:
            gas: Greenhouse gas species.
            treatment_method: Waste treatment method.
            waste_category: Waste category.
            tco2e: Emission amount in tonnes CO2e.
        """
        record_emissions(gas, treatment_method, waste_category, tco2e)

    def record_waste_processed(
        self,
        treatment_method: str,
        waste_category: str,
        tonnes: float = 1.0,
    ) -> None:
        """Record cumulative waste processed.

        Args:
            treatment_method: Waste treatment method applied.
            waste_category: Category of waste treated.
            tonnes: Waste amount in tonnes.
        """
        record_waste_processed(treatment_method, waste_category, tonnes)

    # -- Recovery tracking ---------------------------------------------------

    def record_methane_recovery(
        self,
        recovery_type: str,
        tonnes: float = 1.0,
    ) -> None:
        """Record cumulative methane recovery by pathway.

        Args:
            recovery_type: Methane recovery pathway.
            tonnes: Methane amount in tonnes.
        """
        record_methane_recovery(recovery_type, tonnes)

    def record_energy_recovered(
        self,
        energy_type: str,
        gigajoules: float = 1.0,
    ) -> None:
        """Record cumulative energy recovered.

        Args:
            energy_type: Type of energy recovered.
            gigajoules: Energy amount in gigajoules.
        """
        record_energy_recovered(energy_type, gigajoules)

    # -- Treatment type tracking ---------------------------------------------

    def record_biological_treatment(self, bio_type: str) -> None:
        """Record a biological waste treatment event.

        Args:
            bio_type: Type of biological treatment applied.
        """
        record_biological_treatment(bio_type)

    def record_thermal_treatment(self, thermal_type: str) -> None:
        """Record a thermal waste treatment event.

        Args:
            thermal_type: Type of thermal treatment applied.
        """
        record_thermal_treatment(thermal_type)

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

    # -- Facility gauge ------------------------------------------------------

    def track_active_facilities(self, tenant_id: str, count: int) -> None:
        """Set the gauge for active waste treatment facilities by tenant.

        Args:
            tenant_id: Tenant identifier.
            count: Number of active facilities. Must be >= 0.
        """
        track_active_facilities(tenant_id, count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "wt_calculations_total",
    "wt_calculation_duration_seconds",
    "wt_calculation_errors_total",
    "wt_emissions_tco2e_total",
    "wt_waste_processed_tonnes_total",
    "wt_methane_recovery_tonnes_total",
    "wt_energy_recovered_gj_total",
    "wt_biological_treatments_total",
    "wt_thermal_treatments_total",
    "wt_compliance_checks_total",
    "wt_uncertainty_runs_total",
    "wt_active_facilities",
    # Helper functions
    "record_calculation",
    "observe_calculation_duration",
    "record_calculation_error",
    "record_emissions",
    "record_waste_processed",
    "record_methane_recovery",
    "record_energy_recovered",
    "record_biological_treatment",
    "record_thermal_treatment",
    "record_compliance_check",
    "record_uncertainty_run",
    "track_active_facilities",
    # Collector class
    "MetricsCollector",
]
