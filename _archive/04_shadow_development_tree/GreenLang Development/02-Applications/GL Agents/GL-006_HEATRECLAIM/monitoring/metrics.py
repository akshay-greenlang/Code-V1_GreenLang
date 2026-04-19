"""
GL-006 HEATRECLAIM - Prometheus Metrics Exporter

Comprehensive metrics for heat recovery optimization monitoring.
Implements RED (Rate, Errors, Duration) and USE patterns with
domain-specific optimization KPIs.

Metric Categories:
1. Optimization KPIs - Exergy recovery, cost savings, efficiency
2. Safety Metrics - Constraint violations, thresholds
3. Operational Metrics - API latency, throughput, errors
4. Business Metrics - Energy savings, carbon reduction

Standards:
- OpenMetrics specification
- Prometheus naming conventions
- Kubernetes observability patterns
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZATION KPI METRICS
# =============================================================================

# Exergy recovery metrics
EXERGY_RECOVERED_KW = Gauge(
    "heatreclaim_exergy_recovered_kw",
    "Exergy recovered through heat integration (kW)",
    labelnames=["facility", "design_id"],
)

EXERGY_DESTRUCTION_KW = Gauge(
    "heatreclaim_exergy_destruction_kw",
    "Exergy destruction in heat exchanger network (kW)",
    labelnames=["facility", "design_id"],
)

EXERGY_EFFICIENCY = Gauge(
    "heatreclaim_exergy_efficiency_ratio",
    "Second-law efficiency of heat exchanger network (0-1)",
    labelnames=["facility", "design_id"],
)

# Pinch analysis metrics
PINCH_TEMPERATURE_K = Gauge(
    "heatreclaim_pinch_temperature_kelvin",
    "Pinch point temperature (K)",
    labelnames=["facility", "design_id"],
)

MINIMUM_APPROACH_TEMPERATURE_K = Gauge(
    "heatreclaim_min_approach_temp_kelvin",
    "Minimum approach temperature delta (K)",
    labelnames=["facility", "design_id"],
)

HOT_UTILITY_KW = Gauge(
    "heatreclaim_hot_utility_kw",
    "Minimum hot utility requirement (kW)",
    labelnames=["facility", "design_id"],
)

COLD_UTILITY_KW = Gauge(
    "heatreclaim_cold_utility_kw",
    "Minimum cold utility requirement (kW)",
    labelnames=["facility", "design_id"],
)

# HEN optimization metrics
HEN_CAPITAL_COST_USD = Gauge(
    "heatreclaim_hen_capital_cost_usd",
    "Heat exchanger network capital cost (USD)",
    labelnames=["facility", "design_id"],
)

HEN_ANNUAL_OPERATING_COST_USD = Gauge(
    "heatreclaim_hen_operating_cost_usd",
    "Heat exchanger network annual operating cost (USD)",
    labelnames=["facility", "design_id"],
)

HEN_TOTAL_AREA_M2 = Gauge(
    "heatreclaim_hen_total_area_m2",
    "Total heat exchanger area (m^2)",
    labelnames=["facility", "design_id"],
)

HEN_NUM_EXCHANGERS = Gauge(
    "heatreclaim_hen_num_exchangers",
    "Number of heat exchangers in network",
    labelnames=["facility", "design_id"],
)

# Economic metrics
PAYBACK_PERIOD_YEARS = Gauge(
    "heatreclaim_payback_period_years",
    "Investment payback period (years)",
    labelnames=["facility", "design_id"],
)

NPV_USD = Gauge(
    "heatreclaim_npv_usd",
    "Net present value of optimization project (USD)",
    labelnames=["facility", "design_id"],
)

IRR_PERCENT = Gauge(
    "heatreclaim_irr_percent",
    "Internal rate of return (%)",
    labelnames=["facility", "design_id"],
)

ANNUAL_SAVINGS_USD = Gauge(
    "heatreclaim_annual_savings_usd",
    "Annual cost savings from optimization (USD)",
    labelnames=["facility", "design_id"],
)

# Carbon reduction metrics
CO2_REDUCTION_TONNES_YR = Gauge(
    "heatreclaim_co2_reduction_tonnes_yr",
    "Annual CO2 emissions reduction (tonnes/year)",
    labelnames=["facility", "design_id"],
)

ENERGY_SAVINGS_GJ_YR = Gauge(
    "heatreclaim_energy_savings_gj_yr",
    "Annual energy savings (GJ/year)",
    labelnames=["facility", "design_id"],
)


# =============================================================================
# OPTIMIZATION RUN METRICS
# =============================================================================

OPTIMIZATION_RUNS_TOTAL = Counter(
    "heatreclaim_optimization_runs_total",
    "Total optimization runs",
    labelnames=["algorithm", "status"],
)

OPTIMIZATION_DURATION_SECONDS = Histogram(
    "heatreclaim_optimization_duration_seconds",
    "Optimization run duration",
    labelnames=["algorithm"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

OPTIMIZATION_ITERATIONS = Histogram(
    "heatreclaim_optimization_iterations",
    "Number of optimization iterations",
    labelnames=["algorithm"],
    buckets=(10, 50, 100, 500, 1000, 5000, 10000),
)

MILP_GAP_PERCENT = Gauge(
    "heatreclaim_milp_gap_percent",
    "MILP optimality gap (%)",
    labelnames=["facility", "design_id"],
)

PARETO_SOLUTIONS_COUNT = Gauge(
    "heatreclaim_pareto_solutions_count",
    "Number of Pareto-optimal solutions generated",
    labelnames=["facility", "design_id"],
)


# =============================================================================
# SAFETY METRICS
# =============================================================================

SAFETY_CHECKS_TOTAL = Counter(
    "heatreclaim_safety_checks_total",
    "Total safety constraint checks",
    labelnames=["check_type", "result"],
)

SAFETY_VIOLATIONS_TOTAL = Counter(
    "heatreclaim_safety_violations_total",
    "Total safety constraint violations",
    labelnames=["violation_type", "severity"],
)

APPROACH_TEMP_MARGIN_K = Gauge(
    "heatreclaim_approach_temp_margin_kelvin",
    "Margin above minimum approach temperature (K)",
    labelnames=["facility", "exchanger_id"],
)

FILM_TEMP_MARGIN_K = Gauge(
    "heatreclaim_film_temp_margin_kelvin",
    "Margin below maximum film temperature (K)",
    labelnames=["facility", "exchanger_id"],
)

PRESSURE_DROP_MARGIN_BAR = Gauge(
    "heatreclaim_pressure_drop_margin_bar",
    "Margin below maximum pressure drop (bar)",
    labelnames=["facility", "exchanger_id"],
)


# =============================================================================
# OPERATIONAL METRICS
# =============================================================================

API_REQUESTS_TOTAL = Counter(
    "heatreclaim_api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status"],
)

API_LATENCY_SECONDS = Histogram(
    "heatreclaim_api_latency_seconds",
    "API request latency",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ERRORS_TOTAL = Counter(
    "heatreclaim_errors_total",
    "Total errors",
    labelnames=["error_type", "component"],
)

ACTIVE_OPTIMIZATIONS = Gauge(
    "heatreclaim_active_optimizations",
    "Currently running optimizations",
)

QUEUE_DEPTH = Gauge(
    "heatreclaim_queue_depth",
    "Optimization job queue depth",
    labelnames=["priority"],
)


# =============================================================================
# AGENT INFO METRIC
# =============================================================================

AGENT_INFO = Info(
    "heatreclaim_agent",
    "Agent metadata",
)


# =============================================================================
# METRICS DATA CLASSES
# =============================================================================

@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization run."""

    design_id: str
    facility: str
    algorithm: str
    success: bool
    duration_seconds: float
    iterations: int
    objective_value: float

    # Pinch analysis results
    pinch_temperature_k: Optional[float] = None
    min_approach_temp_k: Optional[float] = None
    hot_utility_kw: Optional[float] = None
    cold_utility_kw: Optional[float] = None

    # Exergy metrics
    exergy_recovered_kw: Optional[float] = None
    exergy_destruction_kw: Optional[float] = None
    exergy_efficiency: Optional[float] = None

    # HEN metrics
    hen_capital_cost_usd: Optional[float] = None
    hen_operating_cost_usd: Optional[float] = None
    hen_total_area_m2: Optional[float] = None
    hen_num_exchangers: Optional[int] = None

    # Economic metrics
    payback_years: Optional[float] = None
    npv_usd: Optional[float] = None
    irr_percent: Optional[float] = None
    annual_savings_usd: Optional[float] = None

    # Environmental metrics
    co2_reduction_tonnes_yr: Optional[float] = None
    energy_savings_gj_yr: Optional[float] = None

    # MILP specific
    milp_gap_percent: Optional[float] = None
    pareto_solutions: Optional[int] = None


@dataclass
class SafetyMetrics:
    """Metrics from a safety check."""

    design_id: str
    facility: str
    check_type: str
    passed: bool
    violations: int
    severity: str

    approach_temp_margin_k: Optional[float] = None
    film_temp_margin_k: Optional[float] = None
    pressure_drop_margin_bar: Optional[float] = None


# =============================================================================
# METRICS EXPORTER CLASS
# =============================================================================

class HeatReclaimMetrics:
    """
    Prometheus metrics exporter for GL-006 HEATRECLAIM.

    Provides comprehensive metrics collection and exposure for
    heat recovery optimization monitoring and alerting.

    Usage:
        metrics = HeatReclaimMetrics()
        metrics.set_agent_info("1.0.0", "production")

        # Record optimization run
        with metrics.measure_optimization("milp") as timer:
            result = optimizer.optimize(design)

        metrics.record_optimization(OptimizationMetrics(...))

        # Get metrics for HTTP response
        content = metrics.get_metrics()
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics exporter.

        Args:
            registry: Optional custom registry for testing
        """
        self.registry = registry or REGISTRY
        self._initialized = False

    def set_agent_info(
        self,
        version: str,
        environment: str,
        instance_id: Optional[str] = None,
    ) -> None:
        """Set agent metadata."""
        AGENT_INFO.info({
            "version": version,
            "environment": environment,
            "instance_id": instance_id or "default",
            "agent_id": "GL-006",
            "agent_name": "HEATRECLAIM",
        })
        self._initialized = True

    @contextmanager
    def measure_optimization(
        self, algorithm: str
    ) -> Generator[None, None, None]:
        """
        Context manager to measure optimization duration.

        Args:
            algorithm: Optimization algorithm used

        Example:
            with metrics.measure_optimization("milp"):
                result = optimizer.optimize(design)
        """
        start_time = time.perf_counter()
        ACTIVE_OPTIMIZATIONS.inc()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            OPTIMIZATION_DURATION_SECONDS.labels(algorithm=algorithm).observe(duration)
            ACTIVE_OPTIMIZATIONS.dec()

    def record_optimization(self, metrics: OptimizationMetrics) -> None:
        """
        Record metrics from a completed optimization.

        Args:
            metrics: Optimization metrics to record
        """
        # Counter metrics
        status = "success" if metrics.success else "failure"
        OPTIMIZATION_RUNS_TOTAL.labels(
            algorithm=metrics.algorithm,
            status=status,
        ).inc()

        OPTIMIZATION_ITERATIONS.labels(
            algorithm=metrics.algorithm
        ).observe(metrics.iterations)

        labels = {"facility": metrics.facility, "design_id": metrics.design_id}

        # Pinch analysis metrics
        if metrics.pinch_temperature_k is not None:
            PINCH_TEMPERATURE_K.labels(**labels).set(metrics.pinch_temperature_k)
        if metrics.min_approach_temp_k is not None:
            MINIMUM_APPROACH_TEMPERATURE_K.labels(**labels).set(metrics.min_approach_temp_k)
        if metrics.hot_utility_kw is not None:
            HOT_UTILITY_KW.labels(**labels).set(metrics.hot_utility_kw)
        if metrics.cold_utility_kw is not None:
            COLD_UTILITY_KW.labels(**labels).set(metrics.cold_utility_kw)

        # Exergy metrics
        if metrics.exergy_recovered_kw is not None:
            EXERGY_RECOVERED_KW.labels(**labels).set(metrics.exergy_recovered_kw)
        if metrics.exergy_destruction_kw is not None:
            EXERGY_DESTRUCTION_KW.labels(**labels).set(metrics.exergy_destruction_kw)
        if metrics.exergy_efficiency is not None:
            EXERGY_EFFICIENCY.labels(**labels).set(metrics.exergy_efficiency)

        # HEN metrics
        if metrics.hen_capital_cost_usd is not None:
            HEN_CAPITAL_COST_USD.labels(**labels).set(metrics.hen_capital_cost_usd)
        if metrics.hen_operating_cost_usd is not None:
            HEN_ANNUAL_OPERATING_COST_USD.labels(**labels).set(metrics.hen_operating_cost_usd)
        if metrics.hen_total_area_m2 is not None:
            HEN_TOTAL_AREA_M2.labels(**labels).set(metrics.hen_total_area_m2)
        if metrics.hen_num_exchangers is not None:
            HEN_NUM_EXCHANGERS.labels(**labels).set(metrics.hen_num_exchangers)

        # Economic metrics
        if metrics.payback_years is not None:
            PAYBACK_PERIOD_YEARS.labels(**labels).set(metrics.payback_years)
        if metrics.npv_usd is not None:
            NPV_USD.labels(**labels).set(metrics.npv_usd)
        if metrics.irr_percent is not None:
            IRR_PERCENT.labels(**labels).set(metrics.irr_percent)
        if metrics.annual_savings_usd is not None:
            ANNUAL_SAVINGS_USD.labels(**labels).set(metrics.annual_savings_usd)

        # Environmental metrics
        if metrics.co2_reduction_tonnes_yr is not None:
            CO2_REDUCTION_TONNES_YR.labels(**labels).set(metrics.co2_reduction_tonnes_yr)
        if metrics.energy_savings_gj_yr is not None:
            ENERGY_SAVINGS_GJ_YR.labels(**labels).set(metrics.energy_savings_gj_yr)

        # MILP specific
        if metrics.milp_gap_percent is not None:
            MILP_GAP_PERCENT.labels(**labels).set(metrics.milp_gap_percent)
        if metrics.pareto_solutions is not None:
            PARETO_SOLUTIONS_COUNT.labels(**labels).set(metrics.pareto_solutions)

        logger.debug(
            f"Recorded optimization metrics for {metrics.design_id}: "
            f"success={metrics.success}, duration={metrics.duration_seconds:.2f}s"
        )

    def record_safety_check(self, metrics: SafetyMetrics) -> None:
        """
        Record metrics from a safety check.

        Args:
            metrics: Safety check metrics to record
        """
        result = "pass" if metrics.passed else "fail"
        SAFETY_CHECKS_TOTAL.labels(
            check_type=metrics.check_type,
            result=result,
        ).inc()

        if not metrics.passed:
            SAFETY_VIOLATIONS_TOTAL.labels(
                violation_type=metrics.check_type,
                severity=metrics.severity,
            ).inc(metrics.violations)

        labels = {"facility": metrics.facility, "exchanger_id": metrics.design_id}

        if metrics.approach_temp_margin_k is not None:
            APPROACH_TEMP_MARGIN_K.labels(**labels).set(metrics.approach_temp_margin_k)
        if metrics.film_temp_margin_k is not None:
            FILM_TEMP_MARGIN_K.labels(**labels).set(metrics.film_temp_margin_k)
        if metrics.pressure_drop_margin_bar is not None:
            PRESSURE_DROP_MARGIN_BAR.labels(**labels).set(metrics.pressure_drop_margin_bar)

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
    ) -> None:
        """Record API request metrics."""
        API_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
        ).inc()

        API_LATENCY_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

    def record_error(
        self,
        error_type: str,
        component: str,
    ) -> None:
        """Record error occurrence."""
        ERRORS_TOTAL.labels(
            error_type=error_type,
            component=component,
        ).inc()

    def set_queue_depth(self, priority: str, depth: int) -> None:
        """Set job queue depth."""
        QUEUE_DEPTH.labels(priority=priority).set(depth)

    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_metrics_instance: Optional[HeatReclaimMetrics] = None


def get_metrics() -> HeatReclaimMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = HeatReclaimMetrics()
    return _metrics_instance


def record_optimization_run(
    design_id: str,
    facility: str,
    algorithm: str,
    success: bool,
    duration_seconds: float,
    iterations: int,
    objective_value: float,
    **kwargs,
) -> None:
    """Convenience function to record optimization run."""
    metrics = OptimizationMetrics(
        design_id=design_id,
        facility=facility,
        algorithm=algorithm,
        success=success,
        duration_seconds=duration_seconds,
        iterations=iterations,
        objective_value=objective_value,
        **kwargs,
    )
    get_metrics().record_optimization(metrics)


def record_safety_check(
    design_id: str,
    facility: str,
    check_type: str,
    passed: bool,
    violations: int = 0,
    severity: str = "warning",
    **kwargs,
) -> None:
    """Convenience function to record safety check."""
    metrics = SafetyMetrics(
        design_id=design_id,
        facility=facility,
        check_type=check_type,
        passed=passed,
        violations=violations,
        severity=severity,
        **kwargs,
    )
    get_metrics().record_safety_check(metrics)


def record_design_analysis(
    design_id: str,
    facility: str,
    exergy_recovered_kw: float,
    exergy_efficiency: float,
    annual_savings_usd: float,
    co2_reduction_tonnes_yr: float,
) -> None:
    """Convenience function to record design analysis KPIs."""
    labels = {"facility": facility, "design_id": design_id}
    EXERGY_RECOVERED_KW.labels(**labels).set(exergy_recovered_kw)
    EXERGY_EFFICIENCY.labels(**labels).set(exergy_efficiency)
    ANNUAL_SAVINGS_USD.labels(**labels).set(annual_savings_usd)
    CO2_REDUCTION_TONNES_YR.labels(**labels).set(co2_reduction_tonnes_yr)
