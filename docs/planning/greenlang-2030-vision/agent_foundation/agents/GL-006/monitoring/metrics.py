# -*- coding: utf-8 -*-
"""
GL-006 HeatRecoveryMaximizer Prometheus Metrics Module.

This module defines 50+ Prometheus metrics for comprehensive monitoring of
heat recovery operations, calculations, system health, and business outcomes.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        start_http_server,
        multiprocess,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")

    # Stub implementations
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

    def generate_latest(*args): return b""
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None


# Metric prefix
METRIC_PREFIX = "gl006"


class HeatRecoveryMetricsCollector:
    """
    Comprehensive Prometheus metrics collector for GL-006 HeatRecoveryMaximizer.

    Provides 50+ metrics across the following categories:
    - Agent Info & Health
    - API Request Metrics
    - Stream Analysis Metrics
    - Pinch Analysis Metrics
    - Heat Exchanger Network Metrics
    - Exergy Analysis Metrics
    - Economic Analysis Metrics
    - Calculation Performance Metrics
    - Validation Metrics
    - Integration Metrics
    - Error & Exception Metrics
    - Business Outcome Metrics
    """

    def __init__(self, registry=None):
        """Initialize the metrics collector."""
        self.registry = registry or REGISTRY

        # =================================================================
        # 1. AGENT INFO & HEALTH METRICS (5 metrics)
        # =================================================================

        # Agent information
        self.agent_info = Info(
            f"{METRIC_PREFIX}_agent_info",
            "GL-006 HeatRecoveryMaximizer agent information",
        )

        # Agent status (1=ready, 0=not ready)
        self.agent_status = Gauge(
            f"{METRIC_PREFIX}_agent_status",
            "Agent operational status (1=ready, 0=not ready)",
        )

        # Agent uptime
        self.agent_uptime_seconds = Gauge(
            f"{METRIC_PREFIX}_agent_uptime_seconds",
            "Agent uptime in seconds",
        )

        # Agent last activity timestamp
        self.agent_last_activity_timestamp = Gauge(
            f"{METRIC_PREFIX}_agent_last_activity_timestamp",
            "Timestamp of last agent activity",
        )

        # Agent health score (0-100)
        self.agent_health_score = Gauge(
            f"{METRIC_PREFIX}_agent_health_score",
            "Agent overall health score (0-100)",
        )

        # =================================================================
        # 2. API REQUEST METRICS (8 metrics)
        # =================================================================

        # Total requests counter
        self.http_requests_total = Counter(
            f"{METRIC_PREFIX}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        # Request duration histogram
        self.http_request_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        # Active requests gauge
        self.http_requests_in_progress = Gauge(
            f"{METRIC_PREFIX}_http_requests_in_progress",
            "Number of HTTP requests in progress",
            ["method", "endpoint"],
        )

        # Request size histogram
        self.http_request_size_bytes = Histogram(
            f"{METRIC_PREFIX}_http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
        )

        # Response size histogram
        self.http_response_size_bytes = Histogram(
            f"{METRIC_PREFIX}_http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
        )

        # Rate limited requests
        self.http_rate_limited_total = Counter(
            f"{METRIC_PREFIX}_http_rate_limited_total",
            "Total rate limited requests",
            ["endpoint"],
        )

        # Authentication failures
        self.http_auth_failures_total = Counter(
            f"{METRIC_PREFIX}_http_auth_failures_total",
            "Total authentication failures",
            ["reason"],
        )

        # Request queue size
        self.request_queue_size = Gauge(
            f"{METRIC_PREFIX}_request_queue_size",
            "Number of requests in queue",
        )

        # =================================================================
        # 3. STREAM ANALYSIS METRICS (8 metrics)
        # =================================================================

        # Streams analyzed counter
        self.streams_analyzed_total = Counter(
            f"{METRIC_PREFIX}_streams_analyzed_total",
            "Total number of streams analyzed",
            ["stream_type"],
        )

        # Current stream count
        self.streams_current_count = Gauge(
            f"{METRIC_PREFIX}_streams_current_count",
            "Current number of streams being tracked",
            ["stream_type"],
        )

        # Stream temperature gauge
        self.stream_temperature_celsius = Gauge(
            f"{METRIC_PREFIX}_stream_temperature_celsius",
            "Stream temperature in Celsius",
            ["stream_id", "position"],  # position: inlet/outlet
        )

        # Stream flow rate gauge
        self.stream_flow_rate_kg_per_second = Gauge(
            f"{METRIC_PREFIX}_stream_flow_rate_kg_per_second",
            "Stream flow rate in kg/s",
            ["stream_id"],
        )

        # Stream heat duty gauge
        self.stream_heat_duty_kw = Gauge(
            f"{METRIC_PREFIX}_stream_heat_duty_kw",
            "Stream heat duty in kW",
            ["stream_id", "type"],  # type: available/required
        )

        # Total recoverable heat gauge
        self.total_recoverable_heat_kw = Gauge(
            f"{METRIC_PREFIX}_total_recoverable_heat_kw",
            "Total recoverable heat in kW",
        )

        # Actually recovered heat gauge
        self.actual_recovered_heat_kw = Gauge(
            f"{METRIC_PREFIX}_actual_recovered_heat_kw",
            "Actually recovered heat in kW",
        )

        # Heat recovery efficiency
        self.heat_recovery_efficiency_percent = Gauge(
            f"{METRIC_PREFIX}_heat_recovery_efficiency_percent",
            "Heat recovery efficiency percentage",
        )

        # =================================================================
        # 4. PINCH ANALYSIS METRICS (6 metrics)
        # =================================================================

        # Pinch analyses performed
        self.pinch_analyses_total = Counter(
            f"{METRIC_PREFIX}_pinch_analyses_total",
            "Total pinch analyses performed",
        )

        # Pinch temperature gauge
        self.pinch_temperature_celsius = Gauge(
            f"{METRIC_PREFIX}_pinch_temperature_celsius",
            "Calculated pinch temperature in Celsius",
        )

        # Minimum hot utility requirement
        self.min_hot_utility_kw = Gauge(
            f"{METRIC_PREFIX}_min_hot_utility_kw",
            "Minimum hot utility requirement in kW",
        )

        # Minimum cold utility requirement
        self.min_cold_utility_kw = Gauge(
            f"{METRIC_PREFIX}_min_cold_utility_kw",
            "Minimum cold utility requirement in kW",
        )

        # Pinch analysis duration
        self.pinch_analysis_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_pinch_analysis_duration_seconds",
            "Pinch analysis calculation duration in seconds",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Cross-pinch heat transfer violations
        self.pinch_violations_total = Counter(
            f"{METRIC_PREFIX}_pinch_violations_total",
            "Total cross-pinch heat transfer violations detected",
        )

        # =================================================================
        # 5. HEAT EXCHANGER NETWORK METRICS (7 metrics)
        # =================================================================

        # Network syntheses performed
        self.network_syntheses_total = Counter(
            f"{METRIC_PREFIX}_network_syntheses_total",
            "Total heat exchanger network syntheses",
        )

        # Number of exchangers in network
        self.network_exchanger_count = Gauge(
            f"{METRIC_PREFIX}_network_exchanger_count",
            "Number of heat exchangers in synthesized network",
        )

        # Total network heat duty
        self.network_total_duty_kw = Gauge(
            f"{METRIC_PREFIX}_network_total_duty_kw",
            "Total heat duty of network in kW",
        )

        # Network area
        self.network_total_area_m2 = Gauge(
            f"{METRIC_PREFIX}_network_total_area_m2",
            "Total heat exchanger area in m2",
        )

        # Network synthesis duration
        self.network_synthesis_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_network_synthesis_duration_seconds",
            "Network synthesis duration in seconds",
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # Network optimization iterations
        self.network_optimization_iterations = Histogram(
            f"{METRIC_PREFIX}_network_optimization_iterations",
            "Number of iterations in network optimization",
            buckets=[10, 50, 100, 200, 500, 1000],
        )

        # Network convergence status
        self.network_convergence_achieved = Counter(
            f"{METRIC_PREFIX}_network_convergence_achieved_total",
            "Network optimization convergence outcomes",
            ["status"],  # converged/not_converged
        )

        # =================================================================
        # 6. EXERGY ANALYSIS METRICS (5 metrics)
        # =================================================================

        # Exergy analyses performed
        self.exergy_analyses_total = Counter(
            f"{METRIC_PREFIX}_exergy_analyses_total",
            "Total exergy analyses performed",
        )

        # Exergy input
        self.exergy_input_kw = Gauge(
            f"{METRIC_PREFIX}_exergy_input_kw",
            "Total exergy input in kW",
        )

        # Exergy destruction
        self.exergy_destruction_kw = Gauge(
            f"{METRIC_PREFIX}_exergy_destruction_kw",
            "Exergy destruction in kW",
        )

        # Second law efficiency
        self.second_law_efficiency_percent = Gauge(
            f"{METRIC_PREFIX}_second_law_efficiency_percent",
            "Second law (exergetic) efficiency percentage",
        )

        # Exergy analysis duration
        self.exergy_analysis_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_exergy_analysis_duration_seconds",
            "Exergy analysis duration in seconds",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
        )

        # =================================================================
        # 7. ECONOMIC ANALYSIS METRICS (8 metrics)
        # =================================================================

        # Economic analyses performed
        self.economic_analyses_total = Counter(
            f"{METRIC_PREFIX}_economic_analyses_total",
            "Total economic analyses performed",
        )

        # Capital cost gauge
        self.capital_cost_usd = Gauge(
            f"{METRIC_PREFIX}_capital_cost_usd",
            "Total capital cost in USD",
            ["project_id"],
        )

        # Annual savings gauge
        self.annual_savings_usd = Gauge(
            f"{METRIC_PREFIX}_annual_savings_usd",
            "Annual energy savings in USD",
            ["project_id"],
        )

        # ROI gauge
        self.roi_percent = Gauge(
            f"{METRIC_PREFIX}_roi_percent",
            "Return on investment percentage",
            ["project_id"],
        )

        # NPV gauge
        self.npv_usd = Gauge(
            f"{METRIC_PREFIX}_npv_usd",
            "Net present value in USD",
            ["project_id"],
        )

        # Payback period gauge
        self.payback_years = Gauge(
            f"{METRIC_PREFIX}_payback_years",
            "Simple payback period in years",
            ["project_id"],
        )

        # IRR gauge
        self.irr_percent = Gauge(
            f"{METRIC_PREFIX}_irr_percent",
            "Internal rate of return percentage",
            ["project_id"],
        )

        # Economic analysis duration
        self.economic_analysis_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_economic_analysis_duration_seconds",
            "Economic analysis duration in seconds",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0],
        )

        # =================================================================
        # 8. CALCULATION PERFORMANCE METRICS (6 metrics)
        # =================================================================

        # Calculations performed
        self.calculations_total = Counter(
            f"{METRIC_PREFIX}_calculations_total",
            "Total calculations performed",
            ["calculation_type"],
        )

        # Calculation duration
        self.calculation_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_calculation_duration_seconds",
            "Calculation duration in seconds",
            ["calculation_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
        )

        # Calculation memory usage
        self.calculation_memory_bytes = Histogram(
            f"{METRIC_PREFIX}_calculation_memory_bytes",
            "Memory usage during calculation in bytes",
            ["calculation_type"],
            buckets=[1e6, 1e7, 1e8, 5e8, 1e9],
        )

        # Active calculations
        self.active_calculations = Gauge(
            f"{METRIC_PREFIX}_active_calculations",
            "Number of calculations in progress",
            ["calculation_type"],
        )

        # Calculation queue size
        self.calculation_queue_size = Gauge(
            f"{METRIC_PREFIX}_calculation_queue_size",
            "Number of calculations in queue",
        )

        # Calculation retries
        self.calculation_retries_total = Counter(
            f"{METRIC_PREFIX}_calculation_retries_total",
            "Total calculation retries",
            ["calculation_type", "reason"],
        )

        # =================================================================
        # 9. VALIDATION METRICS (5 metrics)
        # =================================================================

        # Validations performed
        self.validations_total = Counter(
            f"{METRIC_PREFIX}_validations_total",
            "Total validations performed",
            ["validation_type"],
        )

        # Validation failures
        self.validation_failures_total = Counter(
            f"{METRIC_PREFIX}_validation_failures_total",
            "Total validation failures",
            ["validation_type", "severity"],
        )

        # Thermodynamic validation errors
        self.thermodynamic_validation_errors_total = Counter(
            f"{METRIC_PREFIX}_thermodynamic_validation_errors_total",
            "Total thermodynamic validation errors",
            ["error_code"],
        )

        # Energy balance errors
        self.energy_balance_error_percent = Histogram(
            f"{METRIC_PREFIX}_energy_balance_error_percent",
            "Energy balance error percentage",
            buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # Validation duration
        self.validation_duration_seconds = Histogram(
            f"{METRIC_PREFIX}_validation_duration_seconds",
            "Validation duration in seconds",
            ["validation_type"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
        )

        # =================================================================
        # 10. INTEGRATION METRICS (6 metrics)
        # =================================================================

        # SCADA connection status
        self.scada_connection_status = Gauge(
            f"{METRIC_PREFIX}_scada_connection_status",
            "SCADA connection status (1=connected, 0=disconnected)",
        )

        # Historian connection status
        self.historian_connection_status = Gauge(
            f"{METRIC_PREFIX}_historian_connection_status",
            "Historian connection status (1=connected, 0=disconnected)",
        )

        # Integration requests
        self.integration_requests_total = Counter(
            f"{METRIC_PREFIX}_integration_requests_total",
            "Total integration requests",
            ["system", "operation"],
        )

        # Integration failures
        self.integration_failures_total = Counter(
            f"{METRIC_PREFIX}_integration_failures_total",
            "Total integration failures",
            ["system", "error_type"],
        )

        # Integration latency
        self.integration_latency_seconds = Histogram(
            f"{METRIC_PREFIX}_integration_latency_seconds",
            "Integration request latency in seconds",
            ["system"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
        )

        # Data points collected
        self.data_points_collected_total = Counter(
            f"{METRIC_PREFIX}_data_points_collected_total",
            "Total data points collected from integrations",
            ["source"],
        )

        # =================================================================
        # 11. ERROR & EXCEPTION METRICS (4 metrics)
        # =================================================================

        # Errors total
        self.errors_total = Counter(
            f"{METRIC_PREFIX}_errors_total",
            "Total errors encountered",
            ["error_type", "component"],
        )

        # Exceptions total
        self.exceptions_total = Counter(
            f"{METRIC_PREFIX}_exceptions_total",
            "Total exceptions caught",
            ["exception_type", "component"],
        )

        # Last error timestamp
        self.last_error_timestamp = Gauge(
            f"{METRIC_PREFIX}_last_error_timestamp",
            "Timestamp of last error",
        )

        # Error rate (errors per minute)
        self.error_rate_per_minute = Gauge(
            f"{METRIC_PREFIX}_error_rate_per_minute",
            "Current error rate per minute",
        )

        # =================================================================
        # 12. BUSINESS OUTCOME METRICS (5 metrics)
        # =================================================================

        # Energy saved
        self.energy_saved_kwh_total = Counter(
            f"{METRIC_PREFIX}_energy_saved_kwh_total",
            "Total energy saved in kWh",
        )

        # CO2 avoided
        self.co2_avoided_kg_total = Counter(
            f"{METRIC_PREFIX}_co2_avoided_kg_total",
            "Total CO2 emissions avoided in kg",
        )

        # Cost savings realized
        self.cost_savings_usd_total = Counter(
            f"{METRIC_PREFIX}_cost_savings_usd_total",
            "Total cost savings realized in USD",
        )

        # Projects completed
        self.projects_completed_total = Counter(
            f"{METRIC_PREFIX}_projects_completed_total",
            "Total heat recovery projects completed",
            ["outcome"],  # successful/failed/cancelled
        )

        # Opportunities identified
        self.opportunities_identified_total = Counter(
            f"{METRIC_PREFIX}_opportunities_identified_total",
            "Total heat recovery opportunities identified",
            ["priority"],  # high/medium/low
        )

        # Initialize agent info
        self.set_agent_info()

    def set_agent_info(self):
        """Set agent information metric."""
        self.agent_info.info({
            "agent_id": "GL-006",
            "codename": "HEATRECLAIM",
            "name": "HeatRecoveryMaximizer",
            "version": "1.0.0",
            "domain": "heat_recovery",
        })

    # =================================================================
    # HELPER METHODS
    # =================================================================

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)
        if request_size > 0:
            self.http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(request_size)
        if response_size > 0:
            self.http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)

    def record_stream_analysis(self, stream_type: str):
        """Record stream analysis metrics."""
        self.streams_analyzed_total.labels(stream_type=stream_type).inc()

    def record_pinch_analysis(
        self,
        pinch_temp: float,
        hot_utility: float,
        cold_utility: float,
        duration: float,
    ):
        """Record pinch analysis results."""
        self.pinch_analyses_total.inc()
        self.pinch_temperature_celsius.set(pinch_temp)
        self.min_hot_utility_kw.set(hot_utility)
        self.min_cold_utility_kw.set(cold_utility)
        self.pinch_analysis_duration_seconds.observe(duration)

    def record_network_synthesis(
        self,
        exchanger_count: int,
        total_duty: float,
        total_area: float,
        duration: float,
        iterations: int,
        converged: bool,
    ):
        """Record network synthesis results."""
        self.network_syntheses_total.inc()
        self.network_exchanger_count.set(exchanger_count)
        self.network_total_duty_kw.set(total_duty)
        self.network_total_area_m2.set(total_area)
        self.network_synthesis_duration_seconds.observe(duration)
        self.network_optimization_iterations.observe(iterations)
        self.network_convergence_achieved.labels(
            status="converged" if converged else "not_converged"
        ).inc()

    def record_economic_analysis(
        self,
        project_id: str,
        capital_cost: float,
        annual_savings: float,
        roi: float,
        npv: float,
        payback: float,
        irr: float,
        duration: float,
    ):
        """Record economic analysis results."""
        self.economic_analyses_total.inc()
        self.capital_cost_usd.labels(project_id=project_id).set(capital_cost)
        self.annual_savings_usd.labels(project_id=project_id).set(annual_savings)
        self.roi_percent.labels(project_id=project_id).set(roi)
        self.npv_usd.labels(project_id=project_id).set(npv)
        self.payback_years.labels(project_id=project_id).set(payback)
        self.irr_percent.labels(project_id=project_id).set(irr)
        self.economic_analysis_duration_seconds.observe(duration)

    def record_calculation(
        self,
        calculation_type: str,
        duration: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ):
        """Record calculation metrics."""
        self.calculations_total.labels(calculation_type=calculation_type).inc()
        self.calculation_duration_seconds.labels(
            calculation_type=calculation_type
        ).observe(duration)

        if not success and error_type:
            self.calculation_retries_total.labels(
                calculation_type=calculation_type,
                reason=error_type,
            ).inc()

    def record_validation(
        self,
        validation_type: str,
        success: bool,
        severity: str = "error",
        duration: float = 0,
    ):
        """Record validation metrics."""
        self.validations_total.labels(validation_type=validation_type).inc()

        if not success:
            self.validation_failures_total.labels(
                validation_type=validation_type,
                severity=severity,
            ).inc()

        if duration > 0:
            self.validation_duration_seconds.labels(
                validation_type=validation_type
            ).observe(duration)

    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.errors_total.labels(error_type=error_type, component=component).inc()
        self.last_error_timestamp.set(time.time())

    def record_business_outcome(
        self,
        energy_saved_kwh: float = 0,
        co2_avoided_kg: float = 0,
        cost_savings_usd: float = 0,
    ):
        """Record business outcome metrics."""
        if energy_saved_kwh > 0:
            self.energy_saved_kwh_total.inc(energy_saved_kwh)
        if co2_avoided_kg > 0:
            self.co2_avoided_kg_total.inc(co2_avoided_kg)
        if cost_savings_usd > 0:
            self.cost_savings_usd_total.inc(cost_savings_usd)


# Global metrics collector instance
metrics_collector = HeatRecoveryMetricsCollector()


def setup_metrics(port: int = 9090) -> None:
    """
    Setup Prometheus metrics server.

    Args:
        port: Port to expose metrics on
    """
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


def get_metrics_app():
    """
    Get a FastAPI app for serving metrics.

    Returns:
        FastAPI application for metrics endpoint
    """
    try:
        from fastapi import FastAPI, Response

        app = FastAPI(title="GL-006 Metrics")

        @app.get("/metrics")
        async def metrics():
            if PROMETHEUS_AVAILABLE:
                return Response(
                    content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST,
                )
            return Response(content=b"", media_type="text/plain")

        @app.get("/health")
        async def health():
            return {"status": "healthy", "agent_id": "GL-006"}

        return app

    except ImportError:
        logger.warning("FastAPI not available for metrics app")
        return None


# Decorators for automatic metric collection
def timed_calculation(calculation_type: str):
    """Decorator to time calculations and record metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics_collector.active_calculations.labels(
                calculation_type=calculation_type
            ).inc()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_calculation(calculation_type, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_calculation(
                    calculation_type, duration, False, type(e).__name__
                )
                raise
            finally:
                metrics_collector.active_calculations.labels(
                    calculation_type=calculation_type
                ).dec()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics_collector.active_calculations.labels(
                calculation_type=calculation_type
            ).inc()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_calculation(calculation_type, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_calculation(
                    calculation_type, duration, False, type(e).__name__
                )
                raise
            finally:
                metrics_collector.active_calculations.labels(
                    calculation_type=calculation_type
                ).dec()

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def track_request(method: str, endpoint: str):
    """Context manager for tracking HTTP requests."""
    metrics_collector.http_requests_in_progress.labels(
        method=method, endpoint=endpoint
    ).inc()
    start_time = time.time()
    status_code = 200
    try:
        yield
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start_time
        metrics_collector.http_requests_in_progress.labels(
            method=method, endpoint=endpoint
        ).dec()
        metrics_collector.record_request(method, endpoint, status_code, duration)


__all__ = [
    'PROMETHEUS_AVAILABLE',
    'HeatRecoveryMetricsCollector',
    'metrics_collector',
    'setup_metrics',
    'get_metrics_app',
    'timed_calculation',
    'track_request',
]
