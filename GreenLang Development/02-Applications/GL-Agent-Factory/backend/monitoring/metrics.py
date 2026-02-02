"""
GreenLang Application Metrics

Production-grade Prometheus metrics for comprehensive observability.
Includes counters, histograms, and gauges for tracking all critical operations.
"""

import time
import functools
from typing import Optional, Callable, Any
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    Summary,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# =============================================================================
# Application Info
# =============================================================================

app_info = Info(
    'gl_app',
    'GreenLang application information'
)

# Set application info on import
app_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'service': 'greenlang-agent-factory',
})

# =============================================================================
# Counters - Track cumulative values that only increase
# =============================================================================

# Calculation metrics
calculations_total = Counter(
    'gl_calculations_total',
    'Total number of calculations performed',
    ['agent', 'status', 'calculation_type']
)

# Emission factor lookup metrics
ef_lookups_total = Counter(
    'gl_ef_lookups_total',
    'Total number of emission factor lookups',
    ['source', 'region', 'status', 'cache']
)

# Error metrics
errors_total = Counter(
    'gl_errors_total',
    'Total number of errors encountered',
    ['agent', 'error_type', 'severity']
)

# HTTP request metrics
http_requests_total = Counter(
    'gl_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'handler', 'status']
)

# Agent operation metrics
agent_operations_total = Counter(
    'gl_agent_operations_total',
    'Total agent operations',
    ['agent', 'operation', 'status']
)

# Authentication metrics
auth_attempts_total = Counter(
    'gl_auth_attempts_total',
    'Total authentication attempts',
    ['method', 'status']
)

# Data validation metrics
validation_total = Counter(
    'gl_validation_total',
    'Total data validation operations',
    ['validator', 'status']
)

# =============================================================================
# Histograms - Track distributions of values
# =============================================================================

# Calculation duration
calculation_duration = Histogram(
    'gl_calculation_duration_seconds',
    'Duration of calculations in seconds',
    ['agent', 'calculation_type'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Emission factor lookup duration
ef_lookup_duration = Histogram(
    'gl_ef_lookup_duration_seconds',
    'Duration of emission factor lookups in seconds',
    ['source'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# HTTP request duration
http_request_duration = Histogram(
    'gl_http_request_duration_seconds',
    'Duration of HTTP requests in seconds',
    ['method', 'handler'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

# Database query duration
db_query_duration = Histogram(
    'gl_db_query_duration_seconds',
    'Duration of database queries in seconds',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Cache operation duration
cache_operation_duration = Histogram(
    'gl_cache_operation_duration_seconds',
    'Duration of cache operations in seconds',
    ['operation'],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

# Message queue latency
queue_message_duration = Histogram(
    'gl_queue_message_duration_seconds',
    'Time spent processing queue messages',
    ['queue', 'message_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

# Request payload size
request_size_bytes = Histogram(
    'gl_request_size_bytes',
    'Size of HTTP request payloads',
    ['handler'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# Response payload size
response_size_bytes = Histogram(
    'gl_response_size_bytes',
    'Size of HTTP response payloads',
    ['handler'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# =============================================================================
# Gauges - Track values that can increase or decrease
# =============================================================================

# Active agents gauge
active_agents = Gauge(
    'gl_active_agents',
    'Number of currently active agents',
    ['agent_type']
)

# Registry agents count
registry_agents_count = Gauge(
    'gl_registry_agents_count',
    'Total number of agents in the registry',
    ['status']
)

# Active calculations (in-progress)
active_calculations = Gauge(
    'gl_active_calculations',
    'Number of calculations currently in progress',
    ['agent']
)

# Cache size
cache_size = Gauge(
    'gl_cache_size_bytes',
    'Current size of cache in bytes',
    ['cache_name']
)

# Cache item count
cache_items = Gauge(
    'gl_cache_items',
    'Number of items in cache',
    ['cache_name']
)

# Queue depth
queue_depth = Gauge(
    'gl_queue_depth',
    'Number of messages in queue',
    ['queue']
)

# Database connection pool
db_connections = Gauge(
    'gl_db_connections',
    'Database connection pool statistics',
    ['state']  # active, idle, waiting
)

# Worker status
worker_status = Gauge(
    'gl_worker_status',
    'Worker process status (1=running, 0=stopped)',
    ['worker_id']
)

# Last successful calculation timestamp
last_successful_calculation = Gauge(
    'gl_last_successful_calculation_timestamp',
    'Timestamp of last successful calculation',
    ['agent']
)

# =============================================================================
# Summaries - Track distributions with quantile calculations
# =============================================================================

# Calculation result size
calculation_result_summary = Summary(
    'gl_calculation_result_size_bytes',
    'Size of calculation results',
    ['agent']
)

# =============================================================================
# Helper Functions and Decorators
# =============================================================================

def track_calculation(
    agent: str,
    status: str = "success",
    calculation_type: str = "default",
    duration: Optional[float] = None
) -> None:
    """Track a calculation operation."""
    calculations_total.labels(
        agent=agent,
        status=status,
        calculation_type=calculation_type
    ).inc()

    if duration is not None:
        calculation_duration.labels(
            agent=agent,
            calculation_type=calculation_type
        ).observe(duration)


def track_ef_lookup(
    source: str,
    region: str = "global",
    status: str = "success",
    cache: str = "miss",
    duration: Optional[float] = None
) -> None:
    """Track an emission factor lookup."""
    ef_lookups_total.labels(
        source=source,
        region=region,
        status=status,
        cache=cache
    ).inc()

    if duration is not None:
        ef_lookup_duration.labels(source=source).observe(duration)


def track_error(
    agent: str,
    error_type: str,
    severity: str = "error"
) -> None:
    """Track an error occurrence."""
    errors_total.labels(
        agent=agent,
        error_type=error_type,
        severity=severity
    ).inc()


def track_http_request(
    method: str,
    handler: str,
    status: int,
    duration: Optional[float] = None,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None
) -> None:
    """Track an HTTP request."""
    status_str = str(status)

    http_requests_total.labels(
        method=method,
        handler=handler,
        status=status_str
    ).inc()

    if duration is not None:
        http_request_duration.labels(
            method=method,
            handler=handler
        ).observe(duration)

    if request_size is not None:
        request_size_bytes.labels(handler=handler).observe(request_size)

    if response_size is not None:
        response_size_bytes.labels(handler=handler).observe(response_size)


@contextmanager
def track_calculation_time(agent: str, calculation_type: str = "default"):
    """Context manager to track calculation duration."""
    active_calculations.labels(agent=agent).inc()
    start_time = time.perf_counter()
    status = "success"

    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start_time
        active_calculations.labels(agent=agent).dec()
        track_calculation(
            agent=agent,
            status=status,
            calculation_type=calculation_type,
            duration=duration
        )


@contextmanager
def track_ef_lookup_time(source: str, region: str = "global", cache: str = "miss"):
    """Context manager to track EF lookup duration."""
    start_time = time.perf_counter()
    status = "success"

    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start_time
        track_ef_lookup(
            source=source,
            region=region,
            status=status,
            cache=cache,
            duration=duration
        )


@contextmanager
def track_db_query_time(operation: str, table: str):
    """Context manager to track database query duration."""
    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        db_query_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)


def calculation_metrics(agent: str, calculation_type: str = "default"):
    """Decorator to track calculation metrics."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with track_calculation_time(agent, calculation_type):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with track_calculation_time(agent, calculation_type):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def ef_lookup_metrics(source: str, region: str = "global"):
    """Decorator to track emission factor lookup metrics."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with track_ef_lookup_time(source, region):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with track_ef_lookup_time(source, region):
                return func(*args, **kwargs)

        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is an async coroutine."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# =============================================================================
# Metrics Export Functions
# =============================================================================

def get_metrics() -> bytes:
    """Generate latest metrics in Prometheus format."""
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get the content type for metrics response."""
    return CONTENT_TYPE_LATEST


# =============================================================================
# Metrics Initialization
# =============================================================================

def initialize_metrics():
    """Initialize metrics with default values."""
    # Initialize agent gauges
    active_agents.labels(agent_type="calculation").set(0)
    active_agents.labels(agent_type="lookup").set(0)
    active_agents.labels(agent_type="validation").set(0)

    # Initialize registry gauges
    registry_agents_count.labels(status="active").set(0)
    registry_agents_count.labels(status="inactive").set(0)
    registry_agents_count.labels(status="deprecated").set(0)

    # Initialize cache gauges
    cache_size.labels(cache_name="ef_cache").set(0)
    cache_size.labels(cache_name="result_cache").set(0)
    cache_items.labels(cache_name="ef_cache").set(0)
    cache_items.labels(cache_name="result_cache").set(0)

    # Initialize queue gauges
    queue_depth.labels(queue="calculations").set(0)
    queue_depth.labels(queue="notifications").set(0)

    # Initialize database connection gauges
    db_connections.labels(state="active").set(0)
    db_connections.labels(state="idle").set(0)
    db_connections.labels(state="waiting").set(0)


# Initialize on module load
initialize_metrics()
