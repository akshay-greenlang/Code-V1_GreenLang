"""
Prometheus metrics for GL-002 BoilerEfficiencyOptimizer.

Provides comprehensive metrics tracking for:
- HTTP request latency and throughput
- Boiler optimization performance
- Emissions compliance status
- System resource utilization
- External API latency
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Optional, Callable
import functools
import time
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HTTP REQUEST METRICS
# ==============================================================================

http_requests_total = Counter(
    'gl_002_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'gl_002_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

http_request_size_bytes = Histogram(
    'gl_002_http_request_size_bytes',
    'HTTP request body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

http_response_size_bytes = Histogram(
    'gl_002_http_response_size_bytes',
    'HTTP response body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

# ==============================================================================
# OPTIMIZATION METRICS
# ==============================================================================

optimization_requests_total = Counter(
    'gl_002_optimization_requests_total',
    'Total optimization requests',
    ['strategy', 'status']  # status: success, failure, timeout
)

optimization_duration_seconds = Histogram(
    'gl_002_optimization_duration_seconds',
    'Optimization execution time',
    ['strategy'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

optimization_efficiency_improvement = Histogram(
    'gl_002_optimization_efficiency_improvement_percent',
    'Fuel efficiency improvement from optimization',
    ['strategy'],
    buckets=(0, 1, 2, 3, 5, 10, 15, 20)
)

optimization_cost_savings_usd = Histogram(
    'gl_002_optimization_cost_savings_usd_per_hour',
    'Cost savings from optimization',
    ['fuel_type'],
    buckets=(0, 10, 25, 50, 100, 250, 500, 1000)
)

optimization_emissions_reduction_kg_hr = Gauge(
    'gl_002_optimization_emissions_reduction_kg_hr',
    'CO2 emissions reduction',
    ['boiler_id']
)

# ==============================================================================
# BOILER OPERATING METRICS
# ==============================================================================

boiler_efficiency_percent = Gauge(
    'gl_002_boiler_efficiency_percent',
    'Current boiler thermal efficiency',
    ['boiler_id', 'fuel_type']
)

boiler_steam_flow_kg_hr = Gauge(
    'gl_002_boiler_steam_flow_kg_hr',
    'Steam flow rate',
    ['boiler_id']
)

boiler_fuel_flow_kg_hr = Gauge(
    'gl_002_boiler_fuel_flow_kg_hr',
    'Fuel flow rate',
    ['boiler_id', 'fuel_type']
)

boiler_combustion_temperature_c = Gauge(
    'gl_002_boiler_combustion_temperature_c',
    'Combustion temperature',
    ['boiler_id']
)

boiler_excess_air_percent = Gauge(
    'gl_002_boiler_excess_air_percent',
    'Excess air percentage',
    ['boiler_id']
)

boiler_pressure_bar = Gauge(
    'gl_002_boiler_pressure_bar',
    'Boiler pressure',
    ['boiler_id']
)

boiler_load_percent = Gauge(
    'gl_002_boiler_load_percent',
    'Boiler load percentage',
    ['boiler_id']
)

# ==============================================================================
# EMISSIONS METRICS
# ==============================================================================

emissions_co2_kg_hr = Gauge(
    'gl_002_emissions_co2_kg_hr',
    'CO2 emissions rate',
    ['boiler_id']
)

emissions_nox_ppm = Gauge(
    'gl_002_emissions_nox_ppm',
    'NOx emissions concentration',
    ['boiler_id']
)

emissions_co_ppm = Gauge(
    'gl_002_emissions_co_ppm',
    'CO emissions concentration',
    ['boiler_id']
)

emissions_so2_ppm = Gauge(
    'gl_002_emissions_so2_ppm',
    'SO2 emissions concentration',
    ['boiler_id']
)

emissions_compliance_violations = Counter(
    'gl_002_emissions_compliance_violations_total',
    'Total emissions compliance violations',
    ['boiler_id', 'pollutant', 'limit_type']
)

emissions_compliance_status = Gauge(
    'gl_002_emissions_compliance_status',
    'Compliance status (1=compliant, 0=violation)',
    ['boiler_id']
)

# ==============================================================================
# DATABASE METRICS
# ==============================================================================

db_connection_pool_size = Gauge(
    'gl_002_db_connection_pool_size',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'gl_002_db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

db_query_errors_total = Counter(
    'gl_002_db_query_errors_total',
    'Total database query errors',
    ['query_type', 'error_type']
)

# ==============================================================================
# CACHE METRICS
# ==============================================================================

cache_hits_total = Counter(
    'gl_002_cache_hits_total',
    'Total cache hits',
    ['cache_key_pattern']
)

cache_misses_total = Counter(
    'gl_002_cache_misses_total',
    'Total cache misses',
    ['cache_key_pattern']
)

cache_evictions_total = Counter(
    'gl_002_cache_evictions_total',
    'Total cache evictions',
    ['cache_key_pattern']
)

# ==============================================================================
# EXTERNAL API METRICS
# ==============================================================================

external_api_requests_total = Counter(
    'gl_002_external_api_requests_total',
    'Total external API requests',
    ['api_name', 'status_code']
)

external_api_duration_seconds = Histogram(
    'gl_002_external_api_duration_seconds',
    'External API latency',
    ['api_name'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

external_api_errors_total = Counter(
    'gl_002_external_api_errors_total',
    'Total external API errors',
    ['api_name', 'error_type']
)

# ==============================================================================
# SYSTEM METRICS
# ==============================================================================

system_uptime_seconds = Gauge(
    'gl_002_system_uptime_seconds',
    'Application uptime'
)

system_memory_usage_bytes = Gauge(
    'gl_002_system_memory_usage_bytes',
    'Memory usage',
    ['type']  # type: rss, vms, heap
)

system_cpu_usage_percent = Gauge(
    'gl_002_system_cpu_usage_percent',
    'CPU usage percentage'
)

system_disk_usage_bytes = Gauge(
    'gl_002_system_disk_usage_bytes',
    'Disk usage',
    ['mount_point']
)

# ==============================================================================
# BUSINESS METRICS
# ==============================================================================

optimization_annual_savings_usd = Gauge(
    'gl_002_optimization_annual_savings_usd',
    'Estimated annual cost savings',
    ['boiler_id']
)

optimization_annual_emissions_reduction_tons = Gauge(
    'gl_002_optimization_annual_emissions_reduction_tons',
    'Estimated annual CO2 reduction',
    ['boiler_id']
)

optimization_payback_period_months = Gauge(
    'gl_002_optimization_payback_period_months',
    'Payback period for recommendations',
    ['recommendation_type']
)

# ==============================================================================
# DECORATOR FOR AUTOMATIC METRICS TRACKING
# ==============================================================================

def track_request_metrics(method: str, endpoint: str):
    """
    Decorator to automatically track HTTP request metrics.

    Usage:
        @track_request_metrics('GET', '/api/v1/health')
        async def health_endpoint():
            return {"status": "healthy"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='error'
                ).inc()
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='error'
                ).inc()
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_optimization_metrics(strategy: str):
    """
    Decorator to track optimization request metrics.

    Usage:
        @track_optimization_metrics('fuel_efficiency')
        async def optimize():
            return {"improvement": 5.2}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='success'
                ).inc()

                # Track efficiency improvement if available
                if isinstance(result, dict) and 'improvement_percent' in result:
                    optimization_efficiency_improvement.labels(
                        strategy=strategy
                    ).observe(result['improvement_percent'])

                return result
            except Exception as e:
                duration = time.time() - start_time
                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='failure'
                ).inc()
                raise

        return async_wrapper

    return decorator


# ==============================================================================
# COLLECTOR FOR CUSTOM METRICS
# ==============================================================================

class MetricsCollector:
    """Collects and updates system metrics."""

    @staticmethod
    def update_boiler_metrics(boiler_id: str, metrics: dict):
        """Update boiler operating metrics."""
        if 'efficiency_percent' in metrics:
            boiler_efficiency_percent.labels(
                boiler_id=boiler_id,
                fuel_type=metrics.get('fuel_type', 'unknown')
            ).set(metrics['efficiency_percent'])

        if 'steam_flow_kg_hr' in metrics:
            boiler_steam_flow_kg_hr.labels(boiler_id=boiler_id).set(
                metrics['steam_flow_kg_hr']
            )

        if 'fuel_flow_kg_hr' in metrics:
            boiler_fuel_flow_kg_hr.labels(
                boiler_id=boiler_id,
                fuel_type=metrics.get('fuel_type', 'unknown')
            ).set(metrics['fuel_flow_kg_hr'])

        if 'combustion_temperature_c' in metrics:
            boiler_combustion_temperature_c.labels(boiler_id=boiler_id).set(
                metrics['combustion_temperature_c']
            )

        if 'excess_air_percent' in metrics:
            boiler_excess_air_percent.labels(boiler_id=boiler_id).set(
                metrics['excess_air_percent']
            )

        if 'pressure_bar' in metrics:
            boiler_pressure_bar.labels(boiler_id=boiler_id).set(
                metrics['pressure_bar']
            )

        if 'load_percent' in metrics:
            boiler_load_percent.labels(boiler_id=boiler_id).set(
                metrics['load_percent']
            )

    @staticmethod
    def update_emissions_metrics(boiler_id: str, emissions: dict):
        """Update emissions metrics."""
        if 'co2_kg_hr' in emissions:
            emissions_co2_kg_hr.labels(boiler_id=boiler_id).set(
                emissions['co2_kg_hr']
            )

        if 'nox_ppm' in emissions:
            emissions_nox_ppm.labels(boiler_id=boiler_id).set(
                emissions['nox_ppm']
            )

        if 'co_ppm' in emissions:
            emissions_co_ppm.labels(boiler_id=boiler_id).set(
                emissions['co_ppm']
            )

        if 'so2_ppm' in emissions:
            emissions_so2_ppm.labels(boiler_id=boiler_id).set(
                emissions['so2_ppm']
            )

        if 'compliance_status' in emissions:
            # 1 = compliant, 0 = violation
            compliance_value = 1 if emissions['compliance_status'] == 'compliant' else 0
            emissions_compliance_status.labels(boiler_id=boiler_id).set(
                compliance_value
            )
