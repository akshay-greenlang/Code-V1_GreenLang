# -*- coding: utf-8 -*-
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
# DETERMINISM METRICS
# ==============================================================================

determinism_verification_failures = Counter(
    'gl_002_determinism_verification_failures_total',
    'Total determinism verification failures',
    ['violation_type']  # ai_config, calculation, provenance_hash, cache_key, seed, random, timestamp
)

determinism_score = Gauge(
    'gl_002_determinism_score_percent',
    'Determinism score (0-100%, target: 100%)',
    ['component']  # orchestrator, tools, calculators, validators
)

determinism_verification_duration_seconds = Histogram(
    'gl_002_determinism_verification_duration_seconds',
    'Time spent verifying determinism',
    ['verification_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0)
)

provenance_hash_verifications = Counter(
    'gl_002_provenance_hash_verifications_total',
    'Total provenance hash verifications',
    ['status']  # success, failure
)

cache_key_determinism_checks = Counter(
    'gl_002_cache_key_determinism_checks_total',
    'Total cache key determinism checks',
    ['status']  # deterministic, non_deterministic
)

ai_config_determinism_checks = Counter(
    'gl_002_ai_config_determinism_checks_total',
    'Total AI configuration determinism checks',
    ['status']  # compliant, violation
)

seed_propagation_checks = Counter(
    'gl_002_seed_propagation_checks_total',
    'Total random seed propagation checks',
    ['status']  # valid, invalid
)

unseeded_random_operations_detected = Counter(
    'gl_002_unseeded_random_operations_detected_total',
    'Total unseeded random operations detected',
    ['operation_type']  # random.random, random.randint, random.choice, etc.
)

timestamp_calculations_detected = Counter(
    'gl_002_timestamp_calculations_detected_total',
    'Total timestamp-based calculations detected',
    ['pattern']  # datetime.now, time.time, etc.
)

golden_test_results = Counter(
    'gl_002_golden_test_results_total',
    'Golden test results',
    ['test_name', 'status']  # pass, fail
)

calculation_determinism_runs = Histogram(
    'gl_002_calculation_determinism_runs',
    'Number of runs required to verify calculation determinism',
    ['function_name'],
    buckets=(1, 2, 3, 5, 10, 20, 50, 100)
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

    @staticmethod
    def record_cache_operation(operation: str, hit: bool):
        """Record cache hit/miss."""
        cache_key_pattern = operation.split('_')[0] if '_' in operation else operation
        if hit:
            cache_hits_total.labels(cache_key_pattern=cache_key_pattern).inc()
        else:
            cache_misses_total.labels(cache_key_pattern=cache_key_pattern).inc()

    @staticmethod
    def record_cache_eviction(cache_key_pattern: str):
        """Record cache eviction."""
        cache_evictions_total.labels(cache_key_pattern=cache_key_pattern).inc()

    @staticmethod
    def update_system_metrics(metrics: dict):
        """Update system resource metrics."""
        if 'uptime_seconds' in metrics:
            system_uptime_seconds.set(metrics['uptime_seconds'])

        if 'memory_rss_bytes' in metrics:
            system_memory_usage_bytes.labels(type='rss').set(metrics['memory_rss_bytes'])

        if 'memory_vms_bytes' in metrics:
            system_memory_usage_bytes.labels(type='vms').set(metrics['memory_vms_bytes'])

        if 'cpu_percent' in metrics:
            system_cpu_usage_percent.set(metrics['cpu_percent'])

        if 'disk_usage' in metrics:
            for mount_point, usage in metrics['disk_usage'].items():
                system_disk_usage_bytes.labels(mount_point=mount_point).set(usage)

    @staticmethod
    def record_optimization_result(boiler_id: str, fuel_type: str, strategy: str, result: dict):
        """Record complete optimization result."""
        # Efficiency improvement
        if 'improvement_percent' in result:
            optimization_efficiency_improvement.labels(strategy=strategy).observe(
                result['improvement_percent']
            )

        # Cost savings
        if 'cost_savings_usd_hr' in result:
            optimization_cost_savings_usd.labels(fuel_type=fuel_type).observe(
                result['cost_savings_usd_hr']
            )

        # Emissions reduction
        if 'emissions_reduction_kg_hr' in result:
            optimization_emissions_reduction_kg_hr.labels(boiler_id=boiler_id).set(
                result['emissions_reduction_kg_hr']
            )

        # Annual projections
        if 'annual_savings_usd' in result:
            optimization_annual_savings_usd.labels(boiler_id=boiler_id).set(
                result['annual_savings_usd']
            )

        if 'annual_emissions_reduction_tons' in result:
            optimization_annual_emissions_reduction_tons.labels(boiler_id=boiler_id).set(
                result['annual_emissions_reduction_tons']
            )

    @staticmethod
    def update_determinism_metrics(component: str, score: float):
        """
        Update determinism score metrics.

        Args:
            component: Component name (orchestrator, tools, calculators, validators)
            score: Determinism score (0-100%)
        """
        determinism_score.labels(component=component).set(score)

    @staticmethod
    def record_determinism_violation(violation_type: str):
        """
        Record determinism verification failure.

        Args:
            violation_type: Type of violation (ai_config, calculation, provenance_hash, etc.)
        """
        determinism_verification_failures.labels(violation_type=violation_type).inc()

    @staticmethod
    def record_provenance_verification(success: bool):
        """
        Record provenance hash verification.

        Args:
            success: Whether verification succeeded
        """
        status = 'success' if success else 'failure'
        provenance_hash_verifications.labels(status=status).inc()

    @staticmethod
    def record_cache_key_check(is_deterministic: bool):
        """
        Record cache key determinism check.

        Args:
            is_deterministic: Whether cache key is deterministic
        """
        status = 'deterministic' if is_deterministic else 'non_deterministic'
        cache_key_determinism_checks.labels(status=status).inc()

    @staticmethod
    def record_ai_config_check(is_compliant: bool):
        """
        Record AI configuration determinism check.

        Args:
            is_compliant: Whether AI config is compliant
        """
        status = 'compliant' if is_compliant else 'violation'
        ai_config_determinism_checks.labels(status=status).inc()

    @staticmethod
    def record_seed_propagation_check(is_valid: bool):
        """
        Record random seed propagation check.

        Args:
            is_valid: Whether seed propagation is valid
        """
        status = 'valid' if is_valid else 'invalid'
        seed_propagation_checks.labels(status=status).inc()

    @staticmethod
    def record_unseeded_random_operation(operation_type: str):
        """
        Record detected unseeded random operation.

        Args:
            operation_type: Type of random operation (random.random, etc.)
        """
        unseeded_random_operations_detected.labels(operation_type=operation_type).inc()

    @staticmethod
    def record_timestamp_calculation(pattern: str):
        """
        Record detected timestamp-based calculation.

        Args:
            pattern: Timestamp pattern detected (datetime.now, time.time, etc.)
        """
        timestamp_calculations_detected.labels(pattern=pattern).inc()

    @staticmethod
    def record_golden_test_result(test_name: str, passed: bool):
        """
        Record golden test result.

        Args:
            test_name: Name of golden test
            passed: Whether test passed
        """
        status = 'pass' if passed else 'fail'
        golden_test_results.labels(test_name=test_name, status=status).inc()
