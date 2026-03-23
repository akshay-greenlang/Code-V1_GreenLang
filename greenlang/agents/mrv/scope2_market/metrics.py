# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

12 Prometheus metrics for Scope 2 market-based emissions agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_s2m_`` prefix (GreenLang Scope 2 Market) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_s2m_calculations_total                    (Counter,   labels: instrument_type, calculation_method)
    2.  gl_s2m_calculation_duration_seconds          (Histogram, labels: calculation_method)
    3.  gl_s2m_emissions_co2e_tonnes                 (Counter,   labels: instrument_type, gas)
    4.  gl_s2m_instruments_registered_total           (Counter,   labels: instrument_type, status)
    5.  gl_s2m_instruments_retired_total              (Counter,   labels: instrument_type)
    6.  gl_s2m_coverage_percentage                    (Gauge,     labels: facility_id)
    7.  gl_s2m_compliance_checks_total                (Counter,   labels: framework, status)
    8.  gl_s2m_uncertainty_runs_total                 (Counter,   labels: method)
    9.  gl_s2m_errors_total                           (Counter,   labels: error_type)
    10. gl_s2m_dual_reports_total                     (Counter,   labels: status)
    11. gl_s2m_residual_mix_lookups_total             (Counter,   labels: source)
    12. gl_s2m_active_instruments                     (Gauge,     labels: instrument_type)

Label Values Reference:
    instrument_type:
        eac, rec, go, i_rec, rego, lgc, t_rec, ppa, vppa, green_tariff,
        direct_line, self_generated, bundled, unbundled, residual_mix.
    calculation_method:
        contractual, market_based, supplier_specific, residual_mix,
        energy_attribute_certificate, power_purchase_agreement,
        green_tariff, direct_line, self_generation.
    gas:
        CO2, CH4, N2O, CO2e.
    status (instruments_registered_total):
        active, pending, verified, expired, cancelled.
    status (compliance_checks_total):
        compliant, non_compliant, partial, not_assessed.
    status (dual_reports_total):
        complete, partial, location_only, market_only, reconciled.
    framework:
        GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_GREEN_E,
        EU_ETS, RE100, CDP, TCFD.
    method:
        monte_carlo, analytical, error_propagation,
        ipcc_default_uncertainty, bootstrap.
    error_type:
        validation_error, calculation_error, database_error,
        configuration_error, timeout_error, instrument_error,
        coverage_error, unit_conversion_error, unknown_error.
    source:
        AIB_RESIDUAL_MIX, GREEN_E, RE_DISS, NATIONAL_REGISTRY,
        SUPPLIER_SPECIFIC, IEA, DEFRA, CUSTOM.
    facility_id:
        Free-form string identifying the facility for coverage tracking.

Example:
    Using the Scope2MarketMetrics singleton class:

    >>> from greenlang.agents.mrv.scope2_market.metrics import Scope2MarketMetrics
    >>> metrics = Scope2MarketMetrics()
    >>> metrics.record_calculation("rec", "contractual", 0.042, 150.3)
    >>> metrics.record_instrument_registered("eac", "active")
    >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
    >>> metrics.set_active_instruments("rec", 42)

    Using the module-level convenience function:

    >>> from greenlang.agents.mrv.scope2_market.metrics import get_metrics
    >>> m = get_metrics()
    >>> m.record_calculation("ppa", "power_purchase_agreement", 0.065, 45.0)
    >>> m.record_error("validation_error")
    >>> summary = m.get_metrics_summary()

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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
        "Scope 2 market-based emissions metrics disabled"
    )


# ---------------------------------------------------------------------------
# Scope2MarketMetrics - Thread-safe singleton with graceful fallback
# ---------------------------------------------------------------------------


class Scope2MarketMetrics:
    """Thread-safe singleton for all Scope 2 market-based Prometheus metrics.

    The :class:`Scope2MarketMetrics` encapsulates the full set of 12
    Prometheus metrics used by the Scope 2 Market-Based Emissions Agent.
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
        >>> metrics = Scope2MarketMetrics()
        >>> assert metrics is Scope2MarketMetrics()  # same instance
        >>> metrics.record_calculation(
        ...     "rec", "contractual", 0.042, 150.3
        ... )
        >>> metrics.record_compliance_check("GHG_PROTOCOL", "compliant")
        >>> metrics.set_active_instruments("eac", 25)
        >>> summary = metrics.get_metrics_summary()
    """

    _instance: Optional[Scope2MarketMetrics] = None
    _lock: threading.RLock = threading.RLock()

    # -- Singleton construction ---------------------------------------------

    def __new__(
        cls,
        registry: Any = None,
    ) -> Scope2MarketMetrics:
        """Return the singleton Scope2MarketMetrics instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Args:
            registry: Optional :class:`CollectorRegistry` for metric
                registration.  Ignored after the first construction.

        Returns:
            The singleton :class:`Scope2MarketMetrics` instance.
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
            "Scope2MarketMetrics singleton initialized "
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

        # 1. gl_s2m_calculations_total
        #    Total Scope 2 market-based emission calculations performed,
        #    segmented by contractual instrument type and calculation method.
        self.calculations_total = Counter(
            "gl_s2m_calculations_total",
            "Total Scope 2 market-based emission calculations performed",
            labelnames=["instrument_type", "calculation_method"],
            **reg_kwargs,
        )

        # 2. gl_s2m_calculation_duration_seconds
        #    Duration histogram for Scope 2 market-based calculations,
        #    segmented by calculation method.
        self.calculation_duration_seconds = Histogram(
            "gl_s2m_calculation_duration_seconds",
            "Duration of Scope 2 market-based calculation operations "
            "in seconds",
            labelnames=["calculation_method"],
            buckets=(
                0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0,
            ),
            **reg_kwargs,
        )

        # 3. gl_s2m_emissions_co2e_tonnes
        #    Cumulative emissions in tonnes CO2e by instrument type and
        #    greenhouse gas species.
        self.emissions_co2e_tonnes = Counter(
            "gl_s2m_emissions_co2e_tonnes",
            "Cumulative Scope 2 market-based emissions in tonnes CO2e "
            "by instrument type and gas",
            labelnames=["instrument_type", "gas"],
            **reg_kwargs,
        )

        # 4. gl_s2m_instruments_registered_total
        #    Total contractual instruments registered in the system,
        #    segmented by instrument type and registration status.
        self.instruments_registered_total = Counter(
            "gl_s2m_instruments_registered_total",
            "Total contractual instruments registered by type and status",
            labelnames=["instrument_type", "status"],
            **reg_kwargs,
        )

        # 5. gl_s2m_instruments_retired_total
        #    Total contractual instruments retired (consumed) to claim
        #    emissions reductions, segmented by instrument type.
        self.instruments_retired_total = Counter(
            "gl_s2m_instruments_retired_total",
            "Total contractual instruments retired by type",
            labelnames=["instrument_type"],
            **reg_kwargs,
        )

        # 6. gl_s2m_coverage_percentage
        #    Gauge tracking what percentage of a facility's electricity
        #    consumption is covered by contractual instruments.
        #    Values range from 0.0 to 100.0.
        self.coverage_percentage = Gauge(
            "gl_s2m_coverage_percentage",
            "Percentage of facility electricity consumption covered by "
            "contractual instruments",
            labelnames=["facility_id"],
            **reg_kwargs,
        )

        # 7. gl_s2m_compliance_checks_total
        #    Compliance checks by regulatory framework and pass/fail status.
        self.compliance_checks_total = Counter(
            "gl_s2m_compliance_checks_total",
            "Total compliance checks by framework and status",
            labelnames=["framework", "status"],
            **reg_kwargs,
        )

        # 8. gl_s2m_uncertainty_runs_total
        #    Uncertainty analysis runs by statistical method.
        self.uncertainty_runs_total = Counter(
            "gl_s2m_uncertainty_runs_total",
            "Total uncertainty quantification runs by method",
            labelnames=["method"],
            **reg_kwargs,
        )

        # 9. gl_s2m_errors_total
        #    Error events by error type for operational alerting.
        self.errors_total = Counter(
            "gl_s2m_errors_total",
            "Total Scope 2 market-based calculation errors by error type",
            labelnames=["error_type"],
            **reg_kwargs,
        )

        # 10. gl_s2m_dual_reports_total
        #     Counter tracking dual reporting events (location + market)
        #     as required by GHG Protocol Scope 2 Guidance, segmented
        #     by completion status.
        self.dual_reports_total = Counter(
            "gl_s2m_dual_reports_total",
            "Total dual reporting events (location + market) by status",
            labelnames=["status"],
            **reg_kwargs,
        )

        # 11. gl_s2m_residual_mix_lookups_total
        #     Residual mix emission factor lookups by data source
        #     authority, used when no contractual instrument is available.
        self.residual_mix_lookups_total = Counter(
            "gl_s2m_residual_mix_lookups_total",
            "Total residual mix emission factor lookups by source",
            labelnames=["source"],
            **reg_kwargs,
        )

        # 12. gl_s2m_active_instruments
        #     Gauge tracking the number of currently active contractual
        #     instruments by instrument type, reflecting the real-time
        #     portfolio of energy attribute certificates.
        self.active_instruments = Gauge(
            "gl_s2m_active_instruments",
            "Number of currently active contractual instruments "
            "by instrument type",
            labelnames=["instrument_type"],
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
        self.instruments_registered_total = None  # type: ignore[assignment]
        self.instruments_retired_total = None  # type: ignore[assignment]
        self.coverage_percentage = None  # type: ignore[assignment]
        self.compliance_checks_total = None  # type: ignore[assignment]
        self.uncertainty_runs_total = None  # type: ignore[assignment]
        self.errors_total = None  # type: ignore[assignment]
        self.dual_reports_total = None  # type: ignore[assignment]
        self.residual_mix_lookups_total = None  # type: ignore[assignment]
        self.active_instruments = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Recording methods -- Calculations
    # -----------------------------------------------------------------------

    def record_calculation(
        self,
        instrument_type: str,
        method: str,
        duration: float,
        co2e_tonnes: float,
    ) -> None:
        """Record a Scope 2 market-based emission calculation.

        This is the primary recording method that atomically updates
        three metrics at once: the calculation counter, the duration
        histogram, and the cumulative emissions counter (under the
        aggregate ``CO2e`` gas label).

        Args:
            instrument_type: Type of contractual instrument used (eac,
                rec, go, i_rec, rego, lgc, t_rec, ppa, vppa,
                green_tariff, direct_line, self_generated, bundled,
                unbundled, residual_mix).
            method: Calculation methodology applied (contractual,
                market_based, supplier_specific, residual_mix,
                energy_attribute_certificate, power_purchase_agreement,
                green_tariff, direct_line, self_generation).
            duration: Calculation wall-clock time in seconds.
                Histogram buckets: 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0.
            co2e_tonnes: Total CO2-equivalent emissions in tonnes
                resulting from this calculation.
        """
        if not self.prometheus_available:
            return

        self.calculations_total.labels(
            instrument_type=instrument_type,
            calculation_method=method,
        ).inc()

        self.calculation_duration_seconds.labels(
            calculation_method=method,
        ).observe(duration)

        self.emissions_co2e_tonnes.labels(
            instrument_type=instrument_type,
            gas="CO2e",
        ).inc(co2e_tonnes)

    # -----------------------------------------------------------------------
    # Recording methods -- Instrument lifecycle
    # -----------------------------------------------------------------------

    def record_instrument_registered(
        self,
        instrument_type: str,
        status: str,
    ) -> None:
        """Record a contractual instrument registration event.

        Incremented each time a new energy attribute certificate,
        renewable energy certificate, guarantee of origin, power
        purchase agreement, or other contractual instrument is
        registered in the system.

        Args:
            instrument_type: Type of contractual instrument (eac,
                rec, go, i_rec, rego, lgc, t_rec, ppa, vppa,
                green_tariff, direct_line, self_generated, bundled,
                unbundled, residual_mix).
            status: Registration status of the instrument (active,
                pending, verified, expired, cancelled).
        """
        if not self.prometheus_available:
            return

        self.instruments_registered_total.labels(
            instrument_type=instrument_type,
            status=status,
        ).inc()

    def record_instrument_retired(
        self,
        instrument_type: str,
    ) -> None:
        """Record a contractual instrument retirement event.

        Incremented each time a contractual instrument is retired
        (consumed) to make an emissions claim.  Retirement prevents
        double-counting by removing the instrument from the active
        certificate pool.

        Args:
            instrument_type: Type of contractual instrument retired
                (eac, rec, go, i_rec, rego, lgc, t_rec, ppa, vppa,
                green_tariff, direct_line, self_generated, bundled,
                unbundled, residual_mix).
        """
        if not self.prometheus_available:
            return

        self.instruments_retired_total.labels(
            instrument_type=instrument_type,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Coverage gauge
    # -----------------------------------------------------------------------

    def set_coverage_percentage(
        self,
        facility_id: str,
        pct: float,
    ) -> None:
        """Set the coverage percentage gauge for a facility.

        This is an absolute set (not an increment) reflecting what
        fraction of a facility's electricity consumption is covered
        by contractual instruments.  When coverage is 100%, the
        facility's entire Scope 2 market-based emissions can be
        calculated using supplier-specific or instrument-specific
        emission factors.  Any uncovered portion uses the residual
        mix factor.

        Args:
            facility_id: Unique identifier for the facility being
                tracked.  Free-form string (e.g. "FAC-001",
                "US-CA-SFO-DC1").
            pct: Coverage percentage as a float from 0.0 to 100.0.
                Values outside this range are accepted but may
                indicate a data quality issue.
        """
        if not self.prometheus_available:
            return

        self.coverage_percentage.labels(
            facility_id=facility_id,
        ).set(pct)

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
                ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_GREEN_E,
                EU_ETS, RE100, CDP, TCFD).
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
                timeout_error, instrument_error, coverage_error,
                unit_conversion_error, unknown_error).
        """
        if not self.prometheus_available:
            return

        self.errors_total.labels(
            error_type=error_type,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Dual reporting
    # -----------------------------------------------------------------------

    def record_dual_report(self, status: str) -> None:
        """Record a dual reporting event.

        GHG Protocol Scope 2 Guidance (2015) requires organizations to
        report Scope 2 emissions using both the location-based and
        market-based methods.  This counter tracks each dual reporting
        event and its completion status.

        Args:
            status: Dual report completion status (complete, partial,
                location_only, market_only, reconciled).
        """
        if not self.prometheus_available:
            return

        self.dual_reports_total.labels(
            status=status,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Residual mix lookups
    # -----------------------------------------------------------------------

    def record_residual_mix_lookup(self, source: str) -> None:
        """Record a residual mix emission factor lookup event.

        Each time the agent resolves a residual mix emission factor
        for uncovered consumption (i.e. consumption not matched by
        a contractual instrument), this counter is incremented.
        The residual mix represents the emission intensity of the
        remaining electricity supply after all tracked instruments
        have been subtracted.

        Args:
            source: Data source authority used for the lookup
                (AIB_RESIDUAL_MIX, GREEN_E, RE_DISS,
                NATIONAL_REGISTRY, SUPPLIER_SPECIFIC, IEA, DEFRA,
                CUSTOM).
        """
        if not self.prometheus_available:
            return

        self.residual_mix_lookups_total.labels(
            source=source,
        ).inc()

    # -----------------------------------------------------------------------
    # Recording methods -- Active instruments gauge
    # -----------------------------------------------------------------------

    def set_active_instruments(
        self,
        instrument_type: str,
        count: int,
    ) -> None:
        """Set the gauge for currently active contractual instruments.

        This is an absolute set (not an increment) so the caller is
        responsible for computing the correct current count of active
        instruments for the given type.

        Args:
            instrument_type: Type of contractual instrument (eac,
                rec, go, i_rec, rego, lgc, t_rec, ppa, vppa,
                green_tariff, direct_line, self_generated, bundled,
                unbundled, residual_mix).
            count: Number of currently active instruments of this
                type.  Must be >= 0.
        """
        if not self.prometheus_available:
            return

        self.active_instruments.labels(
            instrument_type=instrument_type,
        ).set(count)

    # -----------------------------------------------------------------------
    # Recording methods -- Emissions by gas
    # -----------------------------------------------------------------------

    def record_emissions(
        self,
        instrument_type: str,
        gas: str,
        tonnes: float,
    ) -> None:
        """Record emissions by individual greenhouse gas species.

        Use this method to record gas-specific emissions (CO2, CH4,
        N2O) separately from the aggregate CO2e value recorded by
        :meth:`record_calculation`.

        Args:
            instrument_type: Type of contractual instrument associated
                with the emissions (eac, rec, go, i_rec, rego, lgc,
                t_rec, ppa, vppa, green_tariff, direct_line,
                self_generated, bundled, unbundled, residual_mix).
            gas: Greenhouse gas species (CO2, CH4, N2O, CO2e).
            tonnes: Emission amount in tonnes to add to the counter.
                Must be > 0.
        """
        if not self.prometheus_available:
            return

        self.emissions_co2e_tonnes.labels(
            instrument_type=instrument_type,
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
            >>> metrics = Scope2MarketMetrics()
            >>> summary = metrics.get_metrics_summary()
            >>> isinstance(summary, dict)
            True
            >>> "gl_s2m_calculations_total" in summary
            True
        """
        if not self.prometheus_available:
            return {
                "gl_s2m_calculations_total": None,
                "gl_s2m_calculation_duration_seconds": None,
                "gl_s2m_emissions_co2e_tonnes": None,
                "gl_s2m_instruments_registered_total": None,
                "gl_s2m_instruments_retired_total": None,
                "gl_s2m_coverage_percentage": None,
                "gl_s2m_compliance_checks_total": None,
                "gl_s2m_uncertainty_runs_total": None,
                "gl_s2m_errors_total": None,
                "gl_s2m_dual_reports_total": None,
                "gl_s2m_residual_mix_lookups_total": None,
                "gl_s2m_active_instruments": None,
                "prometheus_available": False,
            }

        return {
            "gl_s2m_calculations_total": _safe_describe(
                self.calculations_total
            ),
            "gl_s2m_calculation_duration_seconds": _safe_describe(
                self.calculation_duration_seconds
            ),
            "gl_s2m_emissions_co2e_tonnes": _safe_describe(
                self.emissions_co2e_tonnes
            ),
            "gl_s2m_instruments_registered_total": _safe_describe(
                self.instruments_registered_total
            ),
            "gl_s2m_instruments_retired_total": _safe_describe(
                self.instruments_retired_total
            ),
            "gl_s2m_coverage_percentage": _safe_describe(
                self.coverage_percentage
            ),
            "gl_s2m_compliance_checks_total": _safe_describe(
                self.compliance_checks_total
            ),
            "gl_s2m_uncertainty_runs_total": _safe_describe(
                self.uncertainty_runs_total
            ),
            "gl_s2m_errors_total": _safe_describe(
                self.errors_total
            ),
            "gl_s2m_dual_reports_total": _safe_describe(
                self.dual_reports_total
            ),
            "gl_s2m_residual_mix_lookups_total": _safe_describe(
                self.residual_mix_lookups_total
            ),
            "gl_s2m_active_instruments": _safe_describe(
                self.active_instruments
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

_default_metrics: Optional[Scope2MarketMetrics] = None


def get_metrics() -> Scope2MarketMetrics:
    """Return the module-level default :class:`Scope2MarketMetrics` instance.

    This function provides a convenient module-level accessor that
    lazily creates and caches a :class:`Scope2MarketMetrics` singleton.
    It is the recommended entry point for agent code that does not need
    a custom :class:`CollectorRegistry`.

    Returns:
        The shared :class:`Scope2MarketMetrics` instance.

    Example:
        >>> from greenlang.agents.mrv.scope2_market.metrics import get_metrics
        >>> m = get_metrics()
        >>> m.record_calculation("rec", "contractual", 0.05, 120.0)
        >>> m.record_error("validation_error")
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = Scope2MarketMetrics()
    return _default_metrics


# ---------------------------------------------------------------------------
# Module-level helper functions (safe to call without prometheus_client)
# ---------------------------------------------------------------------------
# These thin wrappers delegate to the singleton and mirror the flat-function
# API style used by other MRV metrics modules for backward compatibility.


def record_calculation(
    instrument_type: str,
    method: str,
    duration: float,
    co2e_tonnes: float,
) -> None:
    """Record a Scope 2 market-based emission calculation.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_calculation`.

    Args:
        instrument_type: Type of contractual instrument used (eac,
            rec, go, i_rec, rego, lgc, t_rec, ppa, vppa,
            green_tariff, direct_line, self_generated, bundled,
            unbundled, residual_mix).
        method: Calculation methodology applied (contractual,
            market_based, supplier_specific, residual_mix,
            energy_attribute_certificate, power_purchase_agreement,
            green_tariff, direct_line, self_generation).
        duration: Calculation wall-clock time in seconds.
        co2e_tonnes: Total CO2-equivalent emissions in tonnes.
    """
    get_metrics().record_calculation(
        instrument_type, method, duration, co2e_tonnes
    )


def record_instrument_registered(
    instrument_type: str,
    status: str,
) -> None:
    """Record a contractual instrument registration event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_instrument_registered`.

    Args:
        instrument_type: Type of contractual instrument (eac, rec,
            go, i_rec, rego, lgc, t_rec, ppa, vppa, green_tariff,
            direct_line, self_generated, bundled, unbundled,
            residual_mix).
        status: Registration status (active, pending, verified,
            expired, cancelled).
    """
    get_metrics().record_instrument_registered(instrument_type, status)


def record_instrument_retired(instrument_type: str) -> None:
    """Record a contractual instrument retirement event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_instrument_retired`.

    Args:
        instrument_type: Type of contractual instrument retired (eac,
            rec, go, i_rec, rego, lgc, t_rec, ppa, vppa, green_tariff,
            direct_line, self_generated, bundled, unbundled,
            residual_mix).
    """
    get_metrics().record_instrument_retired(instrument_type)


def set_coverage_percentage(facility_id: str, pct: float) -> None:
    """Set the coverage percentage gauge for a facility.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.set_coverage_percentage`.

    Args:
        facility_id: Unique identifier for the facility.
        pct: Coverage percentage as a float from 0.0 to 100.0.
    """
    get_metrics().set_coverage_percentage(facility_id, pct)


def record_compliance_check(framework: str, status: str) -> None:
    """Record a regulatory compliance check event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_compliance_check`.

    Args:
        framework: Regulatory framework checked (GHG_PROTOCOL,
            ISO_14064, CSRD_ESRS_E1, UK_SECR, EPA_GREEN_E,
            EU_ETS, RE100, CDP, TCFD).
        status: Compliance status result (compliant, non_compliant,
            partial, not_assessed).
    """
    get_metrics().record_compliance_check(framework, status)


def record_uncertainty_run(method: str) -> None:
    """Record an uncertainty quantification run.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_uncertainty_run`.

    Args:
        method: Statistical method applied (monte_carlo, analytical,
            error_propagation, ipcc_default_uncertainty, bootstrap).
    """
    get_metrics().record_uncertainty_run(method)


def record_error(error_type: str) -> None:
    """Record an error event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_error`.

    Args:
        error_type: Classification of the error (validation_error,
            calculation_error, database_error, configuration_error,
            timeout_error, instrument_error, coverage_error,
            unit_conversion_error, unknown_error).
    """
    get_metrics().record_error(error_type)


def record_dual_report(status: str) -> None:
    """Record a dual reporting event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_dual_report`.

    Args:
        status: Dual report completion status (complete, partial,
            location_only, market_only, reconciled).
    """
    get_metrics().record_dual_report(status)


def record_residual_mix_lookup(source: str) -> None:
    """Record a residual mix emission factor lookup event.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_residual_mix_lookup`.

    Args:
        source: Data source authority (AIB_RESIDUAL_MIX, GREEN_E,
            RE_DISS, NATIONAL_REGISTRY, SUPPLIER_SPECIFIC, IEA,
            DEFRA, CUSTOM).
    """
    get_metrics().record_residual_mix_lookup(source)


def set_active_instruments(instrument_type: str, count: int) -> None:
    """Set the gauge for currently active contractual instruments.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.set_active_instruments`.

    Args:
        instrument_type: Type of contractual instrument (eac, rec,
            go, i_rec, rego, lgc, t_rec, ppa, vppa, green_tariff,
            direct_line, self_generated, bundled, unbundled,
            residual_mix).
        count: Number of currently active instruments.  Must be >= 0.
    """
    get_metrics().set_active_instruments(instrument_type, count)


def record_emissions(
    instrument_type: str,
    gas: str,
    tonnes: float,
) -> None:
    """Record emissions by individual greenhouse gas species.

    Convenience wrapper around
    :meth:`Scope2MarketMetrics.record_emissions`.

    Args:
        instrument_type: Type of contractual instrument associated
            with the emissions (eac, rec, go, i_rec, rego, lgc,
            t_rec, ppa, vppa, green_tariff, direct_line,
            self_generated, bundled, unbundled, residual_mix).
        gas: Greenhouse gas species (CO2, CH4, N2O, CO2e).
        tonnes: Emission amount in tonnes.
    """
    get_metrics().record_emissions(instrument_type, gas, tonnes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Singleton class
    "Scope2MarketMetrics",
    # Module-level convenience accessor
    "get_metrics",
    # Module-level helper functions
    "record_calculation",
    "record_instrument_registered",
    "record_instrument_retired",
    "set_coverage_percentage",
    "record_compliance_check",
    "record_uncertainty_run",
    "record_error",
    "record_dual_report",
    "record_residual_mix_lookup",
    "set_active_instruments",
    "record_emissions",
]
