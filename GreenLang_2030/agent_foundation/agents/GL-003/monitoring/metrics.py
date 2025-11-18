"""
Prometheus metrics for GL-003 SteamSystemAnalyzer.

Provides comprehensive metrics tracking for:
- HTTP request latency and throughput
- Steam system analysis performance
- Steam operating metrics (pressure, temperature, flow, condensate)
- Distribution efficiency tracking
- Heat loss monitoring
- Leak detection and steam trap performance
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
    'gl_003_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'gl_003_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

http_request_size_bytes = Histogram(
    'gl_003_http_request_size_bytes',
    'HTTP request body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

http_response_size_bytes = Histogram(
    'gl_003_http_response_size_bytes',
    'HTTP response body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

# ==============================================================================
# ANALYSIS METRICS
# ==============================================================================

analysis_requests_total = Counter(
    'gl_003_analysis_requests_total',
    'Total steam system analysis requests',
    ['analysis_type', 'status']  # status: success, failure, timeout
)

analysis_duration_seconds = Histogram(
    'gl_003_analysis_duration_seconds',
    'Analysis execution time',
    ['analysis_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

analysis_efficiency_improvement = Histogram(
    'gl_003_analysis_efficiency_improvement_percent',
    'Distribution efficiency improvement from analysis',
    ['analysis_type'],
    buckets=(0, 1, 2, 3, 5, 10, 15, 20)
)

analysis_cost_savings_usd = Histogram(
    'gl_003_analysis_cost_savings_usd_per_hour',
    'Cost savings from steam optimization',
    ['steam_type'],
    buckets=(0, 10, 25, 50, 100, 250, 500, 1000)
)

analysis_energy_savings_mwh = Gauge(
    'gl_003_analysis_energy_savings_mwh',
    'Energy savings in MWh',
    ['system_id']
)

# ==============================================================================
# STEAM SYSTEM OPERATING METRICS
# ==============================================================================

steam_pressure_bar = Gauge(
    'gl_003_steam_pressure_bar',
    'Steam pressure at measurement point',
    ['system_id', 'location', 'steam_type']  # steam_type: high_pressure, medium_pressure, low_pressure
)

steam_temperature_c = Gauge(
    'gl_003_steam_temperature_c',
    'Steam temperature',
    ['system_id', 'location', 'steam_type']
)

steam_flow_rate_kg_hr = Gauge(
    'gl_003_steam_flow_rate_kg_hr',
    'Steam flow rate',
    ['system_id', 'location']
)

condensate_return_rate_kg_hr = Gauge(
    'gl_003_condensate_return_rate_kg_hr',
    'Condensate return rate',
    ['system_id', 'location']
)

condensate_return_percent = Gauge(
    'gl_003_condensate_return_percent',
    'Condensate return percentage',
    ['system_id']
)

steam_quality_percent = Gauge(
    'gl_003_steam_quality_percent',
    'Steam quality (dryness fraction)',
    ['system_id', 'location']
)

steam_superheat_c = Gauge(
    'gl_003_steam_superheat_c',
    'Steam superheat temperature',
    ['system_id', 'location']
)

# ==============================================================================
# DISTRIBUTION EFFICIENCY METRICS
# ==============================================================================

distribution_efficiency_percent = Gauge(
    'gl_003_distribution_efficiency_percent',
    'Overall steam distribution efficiency',
    ['system_id']
)

pipe_heat_loss_kw = Gauge(
    'gl_003_pipe_heat_loss_kw',
    'Heat loss through piping',
    ['system_id', 'pipe_segment']
)

insulation_effectiveness_percent = Gauge(
    'gl_003_insulation_effectiveness_percent',
    'Insulation effectiveness',
    ['system_id', 'pipe_segment']
)

pressure_drop_bar = Gauge(
    'gl_003_pressure_drop_bar',
    'Pressure drop across distribution',
    ['system_id', 'segment']
)

distribution_losses_percent = Gauge(
    'gl_003_distribution_losses_percent',
    'Total distribution losses',
    ['system_id']
)

# ==============================================================================
# LEAK DETECTION METRICS
# ==============================================================================

steam_leaks_detected = Counter(
    'gl_003_steam_leaks_detected_total',
    'Total steam leaks detected',
    ['system_id', 'severity']  # severity: minor, moderate, major, critical
)

active_leaks_count = Gauge(
    'gl_003_active_leaks_count',
    'Current number of active leaks',
    ['system_id', 'severity']
)

leak_loss_rate_kg_hr = Gauge(
    'gl_003_leak_loss_rate_kg_hr',
    'Steam loss rate from leaks',
    ['system_id', 'leak_id']
)

leak_cost_impact_usd_hr = Gauge(
    'gl_003_leak_cost_impact_usd_hr',
    'Cost impact of steam leaks per hour',
    ['system_id']
)

leak_detection_confidence = Gauge(
    'gl_003_leak_detection_confidence_percent',
    'Confidence level of leak detection',
    ['system_id', 'leak_id']
)

# ==============================================================================
# STEAM TRAP PERFORMANCE METRICS
# ==============================================================================

steam_trap_operational_count = Gauge(
    'gl_003_steam_trap_operational_count',
    'Number of operational steam traps',
    ['system_id', 'trap_type']
)

steam_trap_failed_count = Gauge(
    'gl_003_steam_trap_failed_count',
    'Number of failed steam traps',
    ['system_id', 'trap_type', 'failure_mode']  # failure_mode: blowing, plugged, cold
)

steam_trap_performance_score = Gauge(
    'gl_003_steam_trap_performance_score_percent',
    'Steam trap performance score',
    ['system_id', 'trap_id']
)

steam_trap_losses_kg_hr = Gauge(
    'gl_003_steam_trap_losses_kg_hr',
    'Steam losses through failed traps',
    ['system_id', 'trap_id']
)

steam_trap_testing_frequency = Counter(
    'gl_003_steam_trap_testing_total',
    'Steam trap testing operations',
    ['system_id', 'trap_type', 'result']  # result: passed, failed
)

# ==============================================================================
# HEAT EXCHANGER METRICS
# ==============================================================================

heat_exchanger_efficiency_percent = Gauge(
    'gl_003_heat_exchanger_efficiency_percent',
    'Heat exchanger thermal efficiency',
    ['system_id', 'exchanger_id']
)

heat_exchanger_fouling_factor = Gauge(
    'gl_003_heat_exchanger_fouling_factor',
    'Heat exchanger fouling factor',
    ['system_id', 'exchanger_id']
)

heat_exchanger_capacity_mw = Gauge(
    'gl_003_heat_exchanger_capacity_mw',
    'Heat exchanger thermal capacity',
    ['system_id', 'exchanger_id']
)

# ==============================================================================
# DATABASE METRICS
# ==============================================================================

db_connection_pool_size = Gauge(
    'gl_003_db_connection_pool_size',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'gl_003_db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

db_query_errors_total = Counter(
    'gl_003_db_query_errors_total',
    'Total database query errors',
    ['query_type', 'error_type']
)

# ==============================================================================
# CACHE METRICS
# ==============================================================================

cache_hits_total = Counter(
    'gl_003_cache_hits_total',
    'Total cache hits',
    ['cache_key_pattern']
)

cache_misses_total = Counter(
    'gl_003_cache_misses_total',
    'Total cache misses',
    ['cache_key_pattern']
)

cache_evictions_total = Counter(
    'gl_003_cache_evictions_total',
    'Total cache evictions',
    ['cache_key_pattern']
)

# ==============================================================================
# EXTERNAL API METRICS
# ==============================================================================

external_api_requests_total = Counter(
    'gl_003_external_api_requests_total',
    'Total external API requests',
    ['api_name', 'status_code']
)

external_api_duration_seconds = Histogram(
    'gl_003_external_api_duration_seconds',
    'External API latency',
    ['api_name'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

external_api_errors_total = Counter(
    'gl_003_external_api_errors_total',
    'Total external API errors',
    ['api_name', 'error_type']
)

# ==============================================================================
# SYSTEM METRICS
# ==============================================================================

system_uptime_seconds = Gauge(
    'gl_003_system_uptime_seconds',
    'Application uptime'
)

system_memory_usage_bytes = Gauge(
    'gl_003_system_memory_usage_bytes',
    'Memory usage',
    ['type']  # type: rss, vms, heap
)

system_cpu_usage_percent = Gauge(
    'gl_003_system_cpu_usage_percent',
    'CPU usage percentage'
)

system_disk_usage_bytes = Gauge(
    'gl_003_system_disk_usage_bytes',
    'Disk usage',
    ['mount_point']
)

# ==============================================================================
# BUSINESS METRICS
# ==============================================================================

analysis_annual_savings_usd = Gauge(
    'gl_003_analysis_annual_savings_usd',
    'Estimated annual cost savings',
    ['system_id']
)

analysis_annual_energy_savings_mwh = Gauge(
    'gl_003_analysis_annual_energy_savings_mwh',
    'Estimated annual energy savings',
    ['system_id']
)

analysis_payback_period_months = Gauge(
    'gl_003_analysis_payback_period_months',
    'Payback period for recommendations',
    ['recommendation_type']
)

steam_cost_per_ton = Gauge(
    'gl_003_steam_cost_per_ton_usd',
    'Cost per ton of steam produced',
    ['system_id']
)

# ==============================================================================
# DETERMINISM METRICS
# ==============================================================================

determinism_verification_failures = Counter(
    'gl_003_determinism_verification_failures_total',
    'Total determinism verification failures',
    ['violation_type']  # ai_config, calculation, provenance_hash, cache_key, seed, random, timestamp
)

determinism_score = Gauge(
    'gl_003_determinism_score_percent',
    'Determinism score (0-100%, target: 100%)',
    ['component']  # orchestrator, tools, calculators, validators
)

determinism_verification_duration_seconds = Histogram(
    'gl_003_determinism_verification_duration_seconds',
    'Time spent verifying determinism',
    ['verification_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0)
)

provenance_hash_verifications = Counter(
    'gl_003_provenance_hash_verifications_total',
    'Total provenance hash verifications',
    ['status']  # success, failure
)

cache_key_determinism_checks = Counter(
    'gl_003_cache_key_determinism_checks_total',
    'Total cache key determinism checks',
    ['status']  # deterministic, non_deterministic
)

ai_config_determinism_checks = Counter(
    'gl_003_ai_config_determinism_checks_total',
    'Total AI configuration determinism checks',
    ['status']  # compliant, violation
)

seed_propagation_checks = Counter(
    'gl_003_seed_propagation_checks_total',
    'Total random seed propagation checks',
    ['status']  # valid, invalid
)

unseeded_random_operations_detected = Counter(
    'gl_003_unseeded_random_operations_detected_total',
    'Total unseeded random operations detected',
    ['operation_type']  # random.random, random.randint, random.choice, etc.
)

timestamp_calculations_detected = Counter(
    'gl_003_timestamp_calculations_detected_total',
    'Total timestamp-based calculations detected',
    ['pattern']  # datetime.now, time.time, etc.
)

golden_test_results = Counter(
    'gl_003_golden_test_results_total',
    'Golden test results',
    ['test_name', 'status']  # pass, fail
)

calculation_determinism_runs = Histogram(
    'gl_003_calculation_determinism_runs',
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


def track_analysis_metrics(analysis_type: str):
    """
    Decorator to track steam system analysis request metrics.

    Usage:
        @track_analysis_metrics('distribution_efficiency')
        async def analyze():
            return {"improvement": 5.2}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
                    status='success'
                ).inc()

                # Track efficiency improvement if available
                if isinstance(result, dict) and 'improvement_percent' in result:
                    analysis_efficiency_improvement.labels(
                        analysis_type=analysis_type
                    ).observe(result['improvement_percent'])

                return result
            except Exception as e:
                duration = time.time() - start_time
                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
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
    def update_steam_system_metrics(system_id: str, metrics: dict):
        """Update steam system operating metrics."""
        if 'pressure_bar' in metrics:
            steam_pressure_bar.labels(
                system_id=system_id,
                location=metrics.get('location', 'unknown'),
                steam_type=metrics.get('steam_type', 'unknown')
            ).set(metrics['pressure_bar'])

        if 'temperature_c' in metrics:
            steam_temperature_c.labels(
                system_id=system_id,
                location=metrics.get('location', 'unknown'),
                steam_type=metrics.get('steam_type', 'unknown')
            ).set(metrics['temperature_c'])

        if 'flow_rate_kg_hr' in metrics:
            steam_flow_rate_kg_hr.labels(
                system_id=system_id,
                location=metrics.get('location', 'unknown')
            ).set(metrics['flow_rate_kg_hr'])

        if 'condensate_return_rate_kg_hr' in metrics:
            condensate_return_rate_kg_hr.labels(
                system_id=system_id,
                location=metrics.get('location', 'unknown')
            ).set(metrics['condensate_return_rate_kg_hr'])

        if 'condensate_return_percent' in metrics:
            condensate_return_percent.labels(system_id=system_id).set(
                metrics['condensate_return_percent']
            )

        if 'steam_quality_percent' in metrics:
            steam_quality_percent.labels(
                system_id=system_id,
                location=metrics.get('location', 'unknown')
            ).set(metrics['steam_quality_percent'])

        if 'distribution_efficiency_percent' in metrics:
            distribution_efficiency_percent.labels(system_id=system_id).set(
                metrics['distribution_efficiency_percent']
            )

    @staticmethod
    def update_leak_metrics(system_id: str, leaks: dict):
        """Update leak detection metrics."""
        if 'active_leaks' in leaks:
            for severity, count in leaks['active_leaks'].items():
                active_leaks_count.labels(
                    system_id=system_id,
                    severity=severity
                ).set(count)

        if 'total_leak_cost_usd_hr' in leaks:
            leak_cost_impact_usd_hr.labels(system_id=system_id).set(
                leaks['total_leak_cost_usd_hr']
            )

    @staticmethod
    def update_steam_trap_metrics(system_id: str, traps: dict):
        """Update steam trap performance metrics."""
        if 'operational_count' in traps:
            for trap_type, count in traps['operational_count'].items():
                steam_trap_operational_count.labels(
                    system_id=system_id,
                    trap_type=trap_type
                ).set(count)

        if 'failed_count' in traps:
            for trap_type, failures in traps['failed_count'].items():
                for failure_mode, count in failures.items():
                    steam_trap_failed_count.labels(
                        system_id=system_id,
                        trap_type=trap_type,
                        failure_mode=failure_mode
                    ).set(count)

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
    def record_analysis_result(system_id: str, steam_type: str, analysis_type: str, result: dict):
        """Record complete analysis result."""
        # Efficiency improvement
        if 'improvement_percent' in result:
            analysis_efficiency_improvement.labels(analysis_type=analysis_type).observe(
                result['improvement_percent']
            )

        # Cost savings
        if 'cost_savings_usd_hr' in result:
            analysis_cost_savings_usd.labels(steam_type=steam_type).observe(
                result['cost_savings_usd_hr']
            )

        # Energy savings
        if 'energy_savings_mwh' in result:
            analysis_energy_savings_mwh.labels(system_id=system_id).set(
                result['energy_savings_mwh']
            )

        # Annual projections
        if 'annual_savings_usd' in result:
            analysis_annual_savings_usd.labels(system_id=system_id).set(
                result['annual_savings_usd']
            )

        if 'annual_energy_savings_mwh' in result:
            analysis_annual_energy_savings_mwh.labels(system_id=system_id).set(
                result['annual_energy_savings_mwh']
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
