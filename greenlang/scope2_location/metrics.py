# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

12 Prometheus metrics for Scope 2 location-based emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_s2l_`` prefix (GreenLang Scope 2 Location) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_s2l_calculations_total                   (Counter,   labels: energy_type, calculation_method)
    2.  gl_s2l_calculation_duration_seconds         (Histogram, labels: energy_type)
    3.  gl_s2l_emissions_co2e_tonnes                (Counter,   labels: energy_type, gas)
    4.  gl_s2l_consumption_mwh_total                (Counter,   labels: energy_type, facility_type)
    5.  gl_s2l_electricity_calculations_total       (Counter,   no labels)
    6.  gl_s2l_steam_heat_cooling_calculations_total (Counter,  labels: energy_type)
    7.  gl_s2l_compliance_checks_total              (Counter,   labels: framework, status)
    8.  gl_s2l_uncertainty_runs_total               (Counter,   labels: method)
    9.  gl_s2l_errors_total                         (Counter,   labels: error_type)
    10. gl_s2l_active_facilities                    (Gauge,     no labels)
    11. gl_s2l_grid_factor_lookups_total            (Counter,   labels: source)
    12. gl_s2l_td_loss_adjustments_total            (Counter,   no labels)

Label Values Reference:
    energy_type:
        electricity, steam, heating, cooling, chilled_water, hot_water.
    calculation_method:
        location_based, grid_average, subregion_average, country_average,
        residual_mix.
    gas:
        CO2, CH4, N2O, CO2e.
    facility_type:
        office, manufacturing, warehouse, data_center, retail, hospital,
        campus, mixed_use, cold_storage, laboratory.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_EGRID,
        EU_ETS, RE100.
    status:
        compliant, non_compliant, partial, not_assessed.
    method:
        monte_carlo, analytical, error_propagation,
        ipcc_default_uncertainty, bootstrap.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, grid_factor_error,
        unit_conversion_error, unknown_error.
    source:
        EPA_EGRID, IEA, DEFRA, IPCC, UNFCCC, AIB, RE_DISS,
        CUSTOM, NATIONAL_REGISTRY.

Example:
    Using the Scope2LocationMetrics singleton class:

    >>> from greenlang.scope2_location.metrics import Scope2LocationMetrics
    >>> metrics = Scope2LocationMetrics()
    >>> metrics.record_calculation("electricity", "location_based", 0.042, 150.3)
    >>> metrics.record_electricity_calculation(0.038)
    >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
    >>> metrics.set_active_facilities(12)

    Using the module-level convenience function:

    >>> from greenlang.scope2_location.metrics import get_metrics
    >>> m = get_metrics()
    >>> m.record_calculation("steam", "grid_average", 0.065, 45.0)
    >>> m.record_error("validation_error")
    >>> summary = m.get_metrics_summary()

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
        "Scope 2 location-based emissions metrics disabled"
    )


# ---------------------------------------------------------------------------
# Scope2LocationMetrics - Thread-safe singleton with graceful fallback
# ---------------------------------------------------------------------------


class Scope2LocationMetrics:
    """Thread-safe singleton for all Scope 2 location-based Prometheus metrics.

    The :class:`Scope2LocationMetrics` encapsulates the full set of 12
    Prometheus metrics used by the Scope 2 Location-Based Emissions Agent.
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
        >>> metrics = Scope2LocationMetrics()
        >>> assert metrics is Scope2LocationMetrics()  # same instance
        >>> metrics.record_calculation(
        ...     "electricity", "location_based", 0.042, 150.3
        ... )
        >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
        >>> metrics.set_active_facilities(12)
        >>> summary = metrics.get_metrics_summary()
    """

    _instance: Optional[Scope2LocationMetrics] = None
    _lock: threading.RLock = threading.RLock()

    # -- Singleton construction ---------------------------------------------

    def __new__(
        cls,
        registry: Any = None,
    ) -> Scope2LocationMetrics:
        """Return the singleton Scope2LocationMetrics instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Args:
            registry: Optional :class:`CollectorRegistry` for metric
                registration.  Ignored after the first construction.

        Returns:
            The singleton :class:`Scope2LocationMetrics` instance.
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
            "Scope2LocationMetrics singleton initialized "
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

        # 1. gl_s2l_calculations_total
        #    Total Scope 2 location-based emission calculations performed,
        #    segmented by energy type and calculation method.
        self.calculations_total = Counter(
            "gl_s2l_calculations_total",
            "Total Scope 2 location-based emission calculations performed",
            labelnames=["energy_type", "calculation_method"],
            **reg_kwargs,
        )

        # 2. gl_s2l_calculation_duration_seconds
        #    Duration histogram for Scope 2 location-based calculations,
        #    segmented by energy type.
        self.calculation_duration_seconds = Histogram(
            "gl_s2l_calculation_duration_seconds",
            "Duration of Scope 2 location-based calculation operations "
            "in seconds",
            labelnames=["energy_type"],
            buckets=(
                0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0,
            ),
            **reg_kwargs,
        )

        # 3. gl_s2l_emissions_co2e_tonnes
        #    Cumulative emissions in tonnes CO2e by energy type and
        #    greenhouse gas species.
        self.emissions_co2e_tonnes = Counter(
            "gl_s2l_emissions_co2e_tonnes",
            "Cumulative Scope 2 location-based emissions in tonnes CO2e "
            "by energy type and gas",
            labelnames=["energy_type", "gas"],
            **reg_kwargs,
        )

        # 4. gl_s2l_consumption_mwh_total
        #    Cumulative energy consumption in MWh by energy type and
        #    facility type.
        self.consumption_mwh_total = Counter(
            "gl_s2l_consumption_mwh_total",
            "Cumulative energy consumption in MWh by energy type and "
            "facility type",
            labelnames=["energy_type", "facility_type"],
            **reg_kwargs,
        )

        # 5. gl_s2l_electricity_calculations_total
        #    Dedicated counter for electricity-specific calculations
        #    (no labels for fast cardinality).
        self.electricity_calculations_total = Counter(
            "gl_s2l_electricity_calculations_total",
            "Total electricity-specific Scope 2 location-based calculations",
            **reg_kwargs,
        )

        # 6. gl_s2l_steam_heat_cooling_calculations_total
        #    Counter for steam, district heating, and cooling calculations
        #    segmented by energy type sub-category.
        self.steam_heat_cooling_calculations_total = Counter(
            "gl_s2l_steam_heat_cooling_calculations_total",
            "Total steam, heating, and cooling Scope 2 location-based "
            "calculations by energy type",
            labelnames=["energy_type"],
            **reg_kwargs,
        )

        # 7. gl_s2l_compliance_checks_total
        #    Compliance checks by regulatory framework and pass/fail status.
        self.compliance_checks_total = Counter(
            "gl_s2l_compliance_checks_total",
            "Total compliance checks by framework and status",
            labelnames=["framework", "status"],
            **reg_kwargs,
        )

        # 8. gl_s2l_uncertainty_runs_total
        #    Uncertainty analysis runs by statistical method.
        self.uncertainty_runs_total = Counter(
            "gl_s2l_uncertainty_runs_total",
            "Total uncertainty quantification runs by method",
            labelnames=["method"],
            **reg_kwargs,
        )

        # 9. gl_s2l_errors_total
        #    Error events by error type for operational alerting.
        self.errors_total = Counter(
            "gl_s2l_errors_total",
            "Total Scope 2 location-based calculation errors by error type",
            labelnames=["error_type"],
            **reg_kwargs,
        )

        # 10. gl_s2l_active_facilities
        #     Gauge tracking the number of currently active facilities
        #     with Scope 2 calculations (no labels for fast reads).
        self.active_facilities = Gauge(
            "gl_s2l_active_facilities",
            "Number of currently active facilities with Scope 2 "
            "location-based calculations",
            **reg_kwargs,
        )

        # 11. gl_s2l_grid_factor_lookups_total
        #     Grid emission factor lookups by data source authority.
        self.grid_factor_lookups_total = Counter(
            "gl_s2l_grid_factor_lookups_total",
            "Total grid emission factor lookups by source authority",
            labelnames=["source"],
            **reg_kwargs,
        )

        # 12. gl_s2l_td_loss_adjustments_total
        #     Transmission and distribution loss adjustments applied
        #     (no labels for simplicity).
        self.td_loss_adjustments_total = Counter(
            "gl_s2l_td_loss_adjustments_total",
            "Total transmission and distribution loss adjustments applied",
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
        self.emissions_co2e_tonnes = None  # type: ignore[assignment]
        self.consumption_mwh_total = None  # type: ignore[assignment]
        self.electricity_calculations_total = None  # type: ignore[assignment]
        self.steam_heat_cooling_calculations_total = None  # type: ignore[assignment]
        self.compliance_checks_total = None  # type: ignore[assignment]
        self.uncertainty_runs_total = None  # type: ignore[assignment]
        self.errors_total = None  # type: ignore[assignment]
        self.active_facilities = None  # type: ignore[assignment]
        self.grid_factor_lookups_total = None  # type: ignore[assignment]
        self.td_loss_adjustments_total = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Recording methods -- Calculations
    # -----------------------------------------------------------------------

    def record_calculation(
        self,
        energy_type: str,
        method: str,
        duration: float,
        co2e_tonnes: float,
    ) -> None:
        """Record a Scope 2 location-based emission calculation.

        This is the primary recording method that atomically updates
        three metrics at once: the calculation counter, the duration
        histogram, and the cumulative emissions counter (under the
        aggregate ``CO2e`` gas label).

        Args:
            energy_type: Type of purchased energy (electricity, steam,
                heating, cooling, chilled_water, hot_water).
            method: Calculation methodology applied (location_based,
                grid_average, subregion_average, country_average,
                residual_mix).
            duration: Calculation wall-clock time in seconds.
                Histogram buckets: 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0.
            co2e_tonnes: Total CO2-equivalent emissions in tonnes
                resulting from this calculation.
        """
        if not self.prometheus_available:
            return

        self.calculations_total.labels(
            energy_type=energy_type,
            calculation_method=method,
        ).inc()

        self.calculation_duration_seconds.labels(
            energy_type=energy_type,
        ).observe(duration)

        self.emissions_co2e_tonnes.labels(
            energy_type=energy_type,
            gas="CO2e",
        ).inc(co2e_tonnes)

    def record_electricity_calculation(self, duration: float) -> None:
        """Record an electricity-specific Scope 2 calculation.

        Increments the dedicated electricity calculation counter and
        observes the duration on the histogram under the ``electricity``
        energy type label.

        Args:
            duration: Calculation wall-clock time in seconds.
                Histogram buckets: 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0.
        """
        if not self.prometheus_available:
            return

        self.electricity_calculations_total.inc()

        self.calculation_duration_seconds.labels(
            energy_type="electricity",
        ).observe(duration)

    def record_steam_heat_cooling(
        self,
        energy_type: str,
        duration: float,
    ) -> None:
        """Record a steam, heating, or cooling Scope 2 calculation.

        Increments the steam/heat/cooling counter with the given energy
        type label and observes the duration on the histogram.

        Args:
            energy_type: Sub-category of thermal energy (steam, heating,
                cooling, chilled_water, hot_water).
            duration: Calculation wall-clock time in seconds.
                Histogram buckets: 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0.
        """
        if not self.prometheus_available:
            return

        self.steam_heat_cooling_calculations_total.labels(
            energy_type=energy_type,
        ).inc()

        self.calculation_duration_seconds.labels(
            energy_type=energy_type,
        ).observe(duration)

    # -----------------------------------------------------------------------
    # Recording methods -- Compliance and uncertainty
    # -----------------------------------------------------------------------

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """Record a regulatory compliance check event.

        Args:
            framework: Regulatory framework checked (GHG_PROTOCOL,
                ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_EGRID,
                EU_ETS, RE100).
            status: Compliance status result (compliant, non_compliant,
                partial, not_assessed).
        """
        if not self.prometheus_available:
            return

        self.compliance_checks_total.labels(
            framework=framework,
            status=status,
        ).inc()

    def record_uncertainty_run(self, method: str) -> None:
        """Record an uncertainty quantification run.

        Args:
            method: Statistical method applied (monte_carlo, analytical,
                error_propagation, ipcc_default_uncertainty, bootstrap).
        """
        if not self.prometheus_available:
            return

        self.uncertainty_runs_total.labels(
            method=method,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Errors
    # -----------------------------------------------------------------------

    def record_error(self, error_type: str) -> None:
        """Record an error event for operational alerting.

        Args:
            error_type: Classification of the error (validation_error,
                calculation_error, database_error, configuration_error,
                timeout_error, grid_factor_error, unit_conversion_error,
                unknown_error).
        """
        if not self.prometheus_available:
            return

        self.errors_total.labels(
            error_type=error_type,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Facility gauge
    # -----------------------------------------------------------------------

    def set_active_facilities(self, count: int) -> None:
        """Set the gauge for currently active facilities.

        This is an absolute set (not an increment) so the caller is
        responsible for computing the correct current count.

        Args:
            count: Number of facilities currently performing Scope 2
                location-based calculations.  Must be >= 0.
        """
        if not self.prometheus_available:
            return

        self.active_facilities.set(count)

    # -----------------------------------------------------------------------
    # Recording methods -- Grid factor lookups
    # -----------------------------------------------------------------------

    def record_grid_factor_lookup(self, source: str) -> None:
        """Record a grid emission factor lookup event.

        Each time the agent resolves a grid-average or sub-regional
        emission factor from an authoritative source, this counter is
        incremented.

        Args:
            source: Data source authority used for the lookup
                (EPA_EGRID, IEA, DEFRA, IPCC, UNFCCC, AIB,
                RE_DISS, CUSTOM, NATIONAL_REGISTRY).
        """
        if not self.prometheus_available:
            return

        self.grid_factor_lookups_total.labels(
            source=source,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- T&D loss adjustments
    # -----------------------------------------------------------------------

    def record_td_loss_adjustment(self) -> None:
        """Record a transmission and distribution loss adjustment.

        Incremented each time a T&D loss factor is applied to adjust
        consumption from the point of delivery to the point of
        generation.  This helps track how frequently T&D corrections
        are applied across calculations.
        """
        if not self.prometheus_available:
            return

        self.td_loss_adjustments_total.inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Consumption and emissions by gas
    # -----------------------------------------------------------------------

    def record_consumption(
        self,
        energy_type: str,
        facility_type: str,
        mwh: float,
    ) -> None:
        """Record energy consumption in MWh.

        Args:
            energy_type: Type of purchased energy (electricity, steam,
                heating, cooling, chilled_water, hot_water).
            facility_type: Classification of the consuming facility
                (office, manufacturing, warehouse, data_center, retail,
                hospital, campus, mixed_use, cold_storage, laboratory).
            mwh: Energy consumption in megawatt-hours to add to the
                counter.  Must be > 0.
        """
        if not self.prometheus_available:
            return

        self.consumption_mwh_total.labels(
            energy_type=energy_type,
            facility_type=facility_type,
        ).inc(mwh)

    def record_emissions(
        self,
        energy_type: str,
        gas: str,
        tonnes: float,
    ) -> None:
        """Record emissions by individual greenhouse gas species.

        Use this method to record gas-specific emissions (CO2, CH4,
        N2O) separately from the aggregate CO2e value recorded by
        :meth:`record_calculation`.

        Args:
            energy_type: Type of purchased energy (electricity, steam,
                heating, cooling, chilled_water, hot_water).
            gas: Greenhouse gas species (CO2, CH4, N2O, CO2e).
            tonnes: Emission amount in tonnes to add to the counter.
                Must be > 0.
        """
        if not self.prometheus_available:
            return

        self.emissions_co2e_tonnes.labels(
            energy_type=energy_type,
            gas=gas,
        ).inc(tonnes)

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
            >>> metrics = Scope2LocationMetrics()
            >>> summary = metrics.get_metrics_summary()
            >>> isinstance(summary, dict)
            True
            >>> "gl_s2l_calculations_total" in summary
            True
        """
        if not self.prometheus_available:
            return {
                "gl_s2l_calculations_total": None,
                "gl_s2l_calculation_duration_seconds": None,
                "gl_s2l_emissions_co2e_tonnes": None,
                "gl_s2l_consumption_mwh_total": None,
                "gl_s2l_electricity_calculations_total": None,
                "gl_s2l_steam_heat_cooling_calculations_total": None,
                "gl_s2l_compliance_checks_total": None,
                "gl_s2l_uncertainty_runs_total": None,
                "gl_s2l_errors_total": None,
                "gl_s2l_active_facilities": None,
                "gl_s2l_grid_factor_lookups_total": None,
                "gl_s2l_td_loss_adjustments_total": None,
                "prometheus_available": False,
            }

        return {
            "gl_s2l_calculations_total": _safe_describe(
                self.calculations_total
            ),
            "gl_s2l_calculation_duration_seconds": _safe_describe(
                self.calculation_duration_seconds
            ),
            "gl_s2l_emissions_co2e_tonnes": _safe_describe(
                self.emissions_co2e_tonnes
            ),
            "gl_s2l_consumption_mwh_total": _safe_describe(
                self.consumption_mwh_total
            ),
            "gl_s2l_electricity_calculations_total": _safe_describe(
                self.electricity_calculations_total
            ),
            "gl_s2l_steam_heat_cooling_calculations_total": _safe_describe(
                self.steam_heat_cooling_calculations_total
            ),
            "gl_s2l_compliance_checks_total": _safe_describe(
                self.compliance_checks_total
            ),
            "gl_s2l_uncertainty_runs_total": _safe_describe(
                self.uncertainty_runs_total
            ),
            "gl_s2l_errors_total": _safe_describe(
                self.errors_total
            ),
            "gl_s2l_active_facilities": _safe_describe(
                self.active_facilities
            ),
            "gl_s2l_grid_factor_lookups_total": _safe_describe(
                self.grid_factor_lookups_total
            ),
            "gl_s2l_td_loss_adjustments_total": _safe_describe(
                self.td_loss_adjustments_total
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

_default_metrics: Optional[Scope2LocationMetrics] = None


def get_metrics() -> Scope2LocationMetrics:
    """Return the module-level default :class:`Scope2LocationMetrics` instance.

    This function provides a convenient module-level accessor that
    lazily creates and caches a :class:`Scope2LocationMetrics` singleton.
    It is the recommended entry point for agent code that does not need
    a custom :class:`CollectorRegistry`.

    Returns:
        The shared :class:`Scope2LocationMetrics` instance.

    Example:
        >>> from greenlang.scope2_location.metrics import get_metrics
        >>> m = get_metrics()
        >>> m.record_calculation("electricity", "location_based", 0.05, 120.0)
        >>> m.record_error("validation_error")
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = Scope2LocationMetrics()
    return _default_metrics


# ---------------------------------------------------------------------------
# Module-level helper functions (safe to call without prometheus_client)
# ---------------------------------------------------------------------------
# These thin wrappers delegate to the singleton and mirror the flat-function
# API style used by other MRV metrics modules for backward compatibility.


def record_calculation(
    energy_type: str,
    method: str,
    duration: float,
    co2e_tonnes: float,
) -> None:
    """Record a Scope 2 location-based emission calculation.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_calculation`.

    Args:
        energy_type: Type of purchased energy (electricity, steam,
            heating, cooling, chilled_water, hot_water).
        method: Calculation methodology applied (location_based,
            grid_average, subregion_average, country_average,
            residual_mix).
        duration: Calculation wall-clock time in seconds.
        co2e_tonnes: Total CO2-equivalent emissions in tonnes.
    """
    get_metrics().record_calculation(energy_type, method, duration, co2e_tonnes)


def record_electricity_calculation(duration: float) -> None:
    """Record an electricity-specific Scope 2 calculation.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_electricity_calculation`.

    Args:
        duration: Calculation wall-clock time in seconds.
    """
    get_metrics().record_electricity_calculation(duration)


def record_steam_heat_cooling(
    energy_type: str,
    duration: float,
) -> None:
    """Record a steam, heating, or cooling Scope 2 calculation.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_steam_heat_cooling`.

    Args:
        energy_type: Sub-category of thermal energy (steam, heating,
            cooling, chilled_water, hot_water).
        duration: Calculation wall-clock time in seconds.
    """
    get_metrics().record_steam_heat_cooling(energy_type, duration)


def record_compliance_check(framework: str, status: str) -> None:
    """Record a regulatory compliance check event.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_compliance_check`.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_EGRID,
            EU_ETS, RE100).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed).
    """
    get_metrics().record_compliance_check(framework, status)


def record_uncertainty_run(method: str) -> None:
    """Record an uncertainty quantification run.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_uncertainty_run`.

    Args:
        method: Statistical method applied (monte_carlo, analytical,
            error_propagation, ipcc_default_uncertainty, bootstrap).
    """
    get_metrics().record_uncertainty_run(method)


def record_error(error_type: str) -> None:
    """Record an error event.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_error`.

    Args:
        error_type: Classification of the error (validation_error,
            calculation_error, database_error, configuration_error,
            timeout_error, grid_factor_error, unit_conversion_error,
            unknown_error).
    """
    get_metrics().record_error(error_type)


def set_active_facilities(count: int) -> None:
    """Set the gauge for currently active facilities.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.set_active_facilities`.

    Args:
        count: Number of active facilities.  Must be >= 0.
    """
    get_metrics().set_active_facilities(count)


def record_grid_factor_lookup(source: str) -> None:
    """Record a grid emission factor lookup event.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_grid_factor_lookup`.

    Args:
        source: Data source authority (EPA_EGRID, IEA, DEFRA, IPCC,
            UNFCCC, AIB, RE_DISS, CUSTOM, NATIONAL_REGISTRY).
    """
    get_metrics().record_grid_factor_lookup(source)


def record_td_loss_adjustment() -> None:
    """Record a transmission and distribution loss adjustment.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_td_loss_adjustment`.
    """
    get_metrics().record_td_loss_adjustment()


def record_consumption(
    energy_type: str,
    facility_type: str,
    mwh: float,
) -> None:
    """Record energy consumption in MWh.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_consumption`.

    Args:
        energy_type: Type of purchased energy (electricity, steam,
            heating, cooling, chilled_water, hot_water).
        facility_type: Classification of the consuming facility
            (office, manufacturing, warehouse, data_center, retail,
            hospital, campus, mixed_use, cold_storage, laboratory).
        mwh: Energy consumption in megawatt-hours.
    """
    get_metrics().record_consumption(energy_type, facility_type, mwh)


def record_emissions(
    energy_type: str,
    gas: str,
    tonnes: float,
) -> None:
    """Record emissions by individual greenhouse gas species.

    Convenience wrapper around
    :meth:`Scope2LocationMetrics.record_emissions`.

    Args:
        energy_type: Type of purchased energy (electricity, steam,
            heating, cooling, chilled_water, hot_water).
        gas: Greenhouse gas species (CO2, CH4, N2O, CO2e).
        tonnes: Emission amount in tonnes.
    """
    get_metrics().record_emissions(energy_type, gas, tonnes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Singleton class
    "Scope2LocationMetrics",
    # Module-level convenience accessor
    "get_metrics",
    # Module-level helper functions
    "record_calculation",
    "record_electricity_calculation",
    "record_steam_heat_cooling",
    "record_compliance_check",
    "record_uncertainty_run",
    "record_error",
    "set_active_facilities",
    "record_grid_factor_lookup",
    "record_td_loss_adjustment",
    "record_consumption",
    "record_emissions",
]
