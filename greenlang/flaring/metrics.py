# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-MRV-006: Flaring Agent

12 Prometheus metrics for flaring agent service monitoring with graceful
fallback when prometheus_client is not installed.

All metric names use the ``gl_fl_`` prefix (GreenLang Flaring) for
consistent identification in Prometheus queries, Grafana dashboards,
and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_fl_calculations_total              (Counter,   labels: flare_type, method, status)
    2.  gl_fl_emissions_kg_co2e_total         (Counter,   labels: flare_type, gas)
    3.  gl_fl_flare_lookups_total             (Counter,   labels: source)
    4.  gl_fl_factor_selections_total         (Counter,   labels: method, source)
    5.  gl_fl_flaring_events_total            (Counter,   labels: event_category, flare_type)
    6.  gl_fl_uncertainty_runs_total          (Counter,   labels: method)
    7.  gl_fl_compliance_checks_total         (Counter,   labels: framework, status)
    8.  gl_fl_batch_jobs_total                (Counter,   labels: status)
    9.  gl_fl_calculation_duration_seconds    (Histogram, labels: operation)
    10. gl_fl_batch_size                      (Histogram, labels: method)
    11. gl_fl_active_calculations             (Gauge)
    12. gl_fl_flare_systems_registered        (Gauge,     labels: flare_type)

Label Values Reference:
    flare_type:
        elevated_steam_assisted, elevated_air_assisted, elevated_unassisted,
        enclosed_ground, multi_point_ground, offshore_marine, candlestick,
        low_pressure.
    method:
        gas_composition, default_emission_factor, engineering_estimate,
        direct_measurement.
    status:
        completed, failed, pending, in_progress, validated.
    gas:
        CO2, CH4, N2O, BLACK_CARBON.
    source:
        EPA, IPCC, DEFRA, EU_ETS, API, CUSTOM.
    event_category:
        routine, non_routine, emergency, maintenance, pilot_purge,
        well_completion.
    framework:
        ghg_protocol, iso_14064, csrd_esrs, epa_subpart_w, eu_ets_mrr,
        eu_methane_reg, world_bank_zrf, ogmp_2_0.
    operation:
        single_calculation, batch_calculation, composition_analysis,
        heating_value_calc, ce_determination, factor_lookup,
        uncertainty_analysis, compliance_check, provenance_hash,
        pilot_purge_calc.

Example:
    >>> from greenlang.flaring.metrics import (
    ...     record_calculation,
    ...     record_emissions,
    ...     set_active_calculations,
    ... )
    >>> record_calculation("elevated_steam_assisted", "gas_composition", "completed")
    >>> record_emissions("elevated_steam_assisted", "CO2", 1500.0)
    >>> set_active_calculations(5)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Optional

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
        "prometheus_client not installed; flaring agent metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Calculation events by flare type, method, and completion status
    fl_calculations_total = Counter(
        "gl_fl_calculations_total",
        "Total flaring emission calculations performed",
        labelnames=["flare_type", "method", "status"],
    )

    # 2. Cumulative emissions by flare type and greenhouse gas
    fl_emissions_kg_co2e_total = Counter(
        "gl_fl_emissions_kg_co2e_total",
        "Cumulative flaring emissions in kg CO2e by flare type and gas",
        labelnames=["flare_type", "gas"],
    )

    # 3. Flare system lookups by emission factor source authority
    fl_flare_lookups_total = Counter(
        "gl_fl_flare_lookups_total",
        "Total flare system and gas composition lookups by source",
        labelnames=["source"],
    )

    # 4. Emission factor selections by calculation method and source
    fl_factor_selections_total = Counter(
        "gl_fl_factor_selections_total",
        "Total emission factor selections by method and source",
        labelnames=["method", "source"],
    )

    # 5. Flaring events logged by event category and flare type
    fl_flaring_events_total = Counter(
        "gl_fl_flaring_events_total",
        "Total flaring events logged by category and flare type",
        labelnames=["event_category", "flare_type"],
    )

    # 6. Uncertainty analysis runs by calculation method
    fl_uncertainty_runs_total = Counter(
        "gl_fl_uncertainty_runs_total",
        "Total uncertainty quantification runs by method",
        labelnames=["method"],
    )

    # 7. Compliance checks by framework and result status
    fl_compliance_checks_total = Counter(
        "gl_fl_compliance_checks_total",
        "Total compliance checks by framework and status",
        labelnames=["framework", "status"],
    )

    # 8. Batch calculation jobs by completion status
    fl_batch_jobs_total = Counter(
        "gl_fl_batch_jobs_total",
        "Total batch calculation jobs by status",
        labelnames=["status"],
    )

    # 9. Calculation duration histogram by operation type
    fl_calculation_duration_seconds = Histogram(
        "gl_fl_calculation_duration_seconds",
        "Duration of flaring calculation operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 10. Batch size histogram by calculation method
    fl_batch_size = Histogram(
        "gl_fl_batch_size",
        "Number of calculations in batch requests by method",
        labelnames=["method"],
        buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
    )

    # 11. Currently active (in-progress) calculations
    fl_active_calculations = Gauge(
        "gl_fl_active_calculations",
        "Number of currently active flaring calculations",
    )

    # 12. Registered flare systems by flare type
    fl_flare_systems_registered = Gauge(
        "gl_fl_flare_systems_registered",
        "Number of flare systems currently registered by type",
        labelnames=["flare_type"],
    )

else:
    # No-op placeholders when prometheus_client is not available
    fl_calculations_total = None              # type: ignore[assignment]
    fl_emissions_kg_co2e_total = None         # type: ignore[assignment]
    fl_flare_lookups_total = None             # type: ignore[assignment]
    fl_factor_selections_total = None         # type: ignore[assignment]
    fl_flaring_events_total = None            # type: ignore[assignment]
    fl_uncertainty_runs_total = None          # type: ignore[assignment]
    fl_compliance_checks_total = None         # type: ignore[assignment]
    fl_batch_jobs_total = None                # type: ignore[assignment]
    fl_calculation_duration_seconds = None    # type: ignore[assignment]
    fl_batch_size = None                      # type: ignore[assignment]
    fl_active_calculations = None             # type: ignore[assignment]
    fl_flare_systems_registered = None        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# MetricsCollector class
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Facade for recording flaring agent Prometheus metrics.

    Provides typed methods for each metric operation with built-in
    guards for when prometheus_client is not installed. All methods
    are safe to call regardless of PROMETHEUS_AVAILABLE state.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_calculation("enclosed_ground", "gas_composition", "completed")
        >>> collector.observe_duration("single_calculation", 0.045)
    """

    @staticmethod
    def record_calculation(
        flare_type: str, method: str, status: str
    ) -> None:
        """Record a flaring emission calculation event.

        Args:
            flare_type: Type of flare system used.
            method: Calculation method applied.
            status: Completion status.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_calculations_total.labels(
            flare_type=flare_type,
            method=method,
            status=status,
        ).inc()

    @staticmethod
    def record_emissions(
        flare_type: str, gas: str, kg_co2e: float = 1.0
    ) -> None:
        """Record cumulative emissions by flare type and gas.

        Args:
            flare_type: Type of flare system.
            gas: Greenhouse gas species (CO2, CH4, N2O, BLACK_CARBON).
            kg_co2e: Emission amount in kg CO2e.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_emissions_kg_co2e_total.labels(
            flare_type=flare_type,
            gas=gas,
        ).inc(kg_co2e)

    @staticmethod
    def record_flare_lookup(source: str) -> None:
        """Record a flare system or gas composition lookup event.

        Args:
            source: Source authority queried (EPA, IPCC, etc.).
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_flare_lookups_total.labels(
            source=source,
        ).inc()

    @staticmethod
    def record_factor_selection(method: str, source: str) -> None:
        """Record an emission factor selection event.

        Args:
            method: Calculation method for which the factor was selected.
            source: Source authority of the selected factor.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_factor_selections_total.labels(
            method=method,
            source=source,
        ).inc()

    @staticmethod
    def record_event(event_category: str, flare_type: str) -> None:
        """Record a flaring event being logged.

        Args:
            event_category: Category of the flaring event.
            flare_type: Type of flare system.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_flaring_events_total.labels(
            event_category=event_category,
            flare_type=flare_type,
        ).inc()

    @staticmethod
    def record_uncertainty(method: str) -> None:
        """Record an uncertainty quantification run.

        Args:
            method: Calculation method used for uncertainty analysis.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_uncertainty_runs_total.labels(
            method=method,
        ).inc()

    @staticmethod
    def record_compliance(framework: str, status: str) -> None:
        """Record a compliance check event.

        Args:
            framework: Regulatory framework checked.
            status: Result status (compliant, non_compliant, partial).
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_compliance_checks_total.labels(
            framework=framework,
            status=status,
        ).inc()

    @staticmethod
    def record_batch(status: str) -> None:
        """Record a batch calculation job event.

        Args:
            status: Completion status of the batch job.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_batch_jobs_total.labels(
            status=status,
        ).inc()

    @staticmethod
    def observe_duration(operation: str, seconds: float) -> None:
        """Record the duration of a calculation operation.

        Args:
            operation: Type of operation being measured.
            seconds: Operation wall-clock time in seconds.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_calculation_duration_seconds.labels(
            operation=operation,
        ).observe(seconds)

    @staticmethod
    def observe_batch_size(method: str, size: int) -> None:
        """Record the size of a batch calculation request.

        Args:
            method: Primary calculation method in the batch.
            size: Number of individual calculations in the batch.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_batch_size.labels(
            method=method,
        ).observe(size)

    @staticmethod
    def set_active_calculations(count: int) -> None:
        """Set the gauge for currently active calculations.

        Args:
            count: Number of calculations currently in progress.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_active_calculations.set(count)

    @staticmethod
    def set_flare_systems_registered(
        flare_type: str, count: int
    ) -> None:
        """Set the gauge for registered flare systems by type.

        Args:
            flare_type: Type of flare system.
            count: Number of registered systems of this type.
        """
        if not PROMETHEUS_AVAILABLE:
            return
        fl_flare_systems_registered.labels(
            flare_type=flare_type,
        ).set(count)


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward compatibility with SC pattern)
# ---------------------------------------------------------------------------


def record_calculation(flare_type: str, method: str, status: str) -> None:
    """Record a flaring calculation event. See MetricsCollector."""
    MetricsCollector.record_calculation(flare_type, method, status)


def record_emissions(
    flare_type: str, gas: str, kg_co2e: float = 1.0
) -> None:
    """Record cumulative emissions. See MetricsCollector."""
    MetricsCollector.record_emissions(flare_type, gas, kg_co2e)


def record_flare_lookup(source: str) -> None:
    """Record a flare lookup event. See MetricsCollector."""
    MetricsCollector.record_flare_lookup(source)


def record_factor_selection(method: str, source: str) -> None:
    """Record an emission factor selection. See MetricsCollector."""
    MetricsCollector.record_factor_selection(method, source)


def record_event(event_category: str, flare_type: str) -> None:
    """Record a flaring event. See MetricsCollector."""
    MetricsCollector.record_event(event_category, flare_type)


def record_uncertainty(method: str) -> None:
    """Record an uncertainty run. See MetricsCollector."""
    MetricsCollector.record_uncertainty(method)


def record_compliance(framework: str, status: str) -> None:
    """Record a compliance check. See MetricsCollector."""
    MetricsCollector.record_compliance(framework, status)


def record_batch(status: str) -> None:
    """Record a batch job event. See MetricsCollector."""
    MetricsCollector.record_batch(status)


def observe_duration(operation: str, seconds: float) -> None:
    """Record operation duration. See MetricsCollector."""
    MetricsCollector.observe_duration(operation, seconds)


def observe_batch_size(method: str, size: int) -> None:
    """Record batch size. See MetricsCollector."""
    MetricsCollector.observe_batch_size(method, size)


def set_active_calculations(count: int) -> None:
    """Set active calculations gauge. See MetricsCollector."""
    MetricsCollector.set_active_calculations(count)


def set_flare_systems_registered(flare_type: str, count: int) -> None:
    """Set registered flare systems gauge. See MetricsCollector."""
    MetricsCollector.set_flare_systems_registered(flare_type, count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "fl_calculations_total",
    "fl_emissions_kg_co2e_total",
    "fl_flare_lookups_total",
    "fl_factor_selections_total",
    "fl_flaring_events_total",
    "fl_uncertainty_runs_total",
    "fl_compliance_checks_total",
    "fl_batch_jobs_total",
    "fl_calculation_duration_seconds",
    "fl_batch_size",
    "fl_active_calculations",
    "fl_flare_systems_registered",
    # MetricsCollector class
    "MetricsCollector",
    # Module-level helper functions
    "record_calculation",
    "record_emissions",
    "record_flare_lookup",
    "record_factor_selection",
    "record_event",
    "record_uncertainty",
    "record_compliance",
    "record_batch",
    "observe_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_flare_systems_registered",
]
