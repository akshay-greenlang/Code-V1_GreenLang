"""
Circuit Breaker Metrics for Prometheus

Production-grade metrics collection for circuit breaker monitoring across all
external dependencies (Factor Broker, LLM Providers, ERP SAP, etc.).

Metrics Provided:
- circuit_breaker_state: Current state (closed=0, open=1, half_open=2)
- circuit_breaker_failures_total: Total failures per service
- circuit_breaker_successes_total: Total successes per service
- circuit_breaker_state_changes_total: State transition counter
- circuit_breaker_rejection_total: Requests rejected when open
- circuit_breaker_latency_seconds: Request latency histogram

Integration:
    # In your application startup
    from greenlang.resilience.metrics import get_circuit_breaker_metrics

    metrics = get_circuit_breaker_metrics()

    # Track circuit breaker events
    metrics.record_state_change(service="factor_broker", from_state="closed", to_state="open")
    metrics.record_failure(service="factor_broker")
    metrics.record_success(service="llm_provider", latency_seconds=1.5)

Version: 1.0.0
Author: GreenLang VCCI Team (Health Check & Monitoring)
Date: 2025-11-09
"""

import time
from typing import Optional, Dict, Any
from enum import Enum

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Circuit breaker metrics disabled.")


# ============================================================================
# CIRCUIT BREAKER STATES (mirrors resilience.py)
# ============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # 0 - Normal operation
    OPEN = "open"          # 1 - Failing, fast-fail all requests
    HALF_OPEN = "half_open"  # 2 - Testing if recovered


# Map states to numeric values for Prometheus gauge
STATE_TO_NUMBER = {
    CircuitState.CLOSED: 0,
    CircuitState.OPEN: 1,
    CircuitState.HALF_OPEN: 2,
}


# ============================================================================
# CIRCUIT BREAKER METRICS
# ============================================================================

class CircuitBreakerMetrics:
    """
    Prometheus metrics for circuit breaker monitoring.

    Tracks circuit breaker health across all external dependencies:
    - Factor Broker (emission factor database)
    - LLM Provider (Anthropic Claude, OpenAI)
    - ERP SAP (enterprise resource planning)
    - Database (PostgreSQL)
    - Redis (cache)
    - Weaviate (vector database)

    Metrics are exposed at /metrics endpoint for Prometheus scraping.
    """

    def __init__(self, registry: Optional['CollectorRegistry'] = None, namespace: str = "greenlang"):
        """
        Initialize circuit breaker metrics.

        Args:
            registry: Prometheus registry (None for default)
            namespace: Metric namespace prefix
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required for circuit breaker metrics. "
                "Install it with: pip install prometheus-client"
            )

        self.registry = registry or CollectorRegistry()
        self.namespace = namespace

        # Initialize all metrics
        self._init_circuit_breaker_metrics()
        self._init_dependency_metrics()

    def _init_circuit_breaker_metrics(self):
        """Initialize circuit breaker-specific metrics."""

        # Circuit breaker state (gauge: 0=closed, 1=open, 2=half_open)
        self.circuit_breaker_state = Gauge(
            f'{self.namespace}_circuit_breaker_state',
            'Current circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['service'],  # service: factor_broker, llm_provider, erp_sap, etc.
            registry=self.registry
        )

        # Total failures
        self.circuit_breaker_failures_total = Counter(
            f'{self.namespace}_circuit_breaker_failures_total',
            'Total number of failures tracked by circuit breaker',
            ['service'],
            registry=self.registry
        )

        # Total successes
        self.circuit_breaker_successes_total = Counter(
            f'{self.namespace}_circuit_breaker_successes_total',
            'Total number of successful requests through circuit breaker',
            ['service'],
            registry=self.registry
        )

        # State changes (transitions)
        self.circuit_breaker_state_changes_total = Counter(
            f'{self.namespace}_circuit_breaker_state_changes_total',
            'Total number of circuit breaker state transitions',
            ['service', 'from_state', 'to_state'],
            registry=self.registry
        )

        # Rejections when circuit is open
        self.circuit_breaker_rejection_total = Counter(
            f'{self.namespace}_circuit_breaker_rejection_total',
            'Total requests rejected due to open circuit breaker',
            ['service'],
            registry=self.registry
        )

        # Request latency through circuit breaker
        self.circuit_breaker_latency_seconds = Histogram(
            f'{self.namespace}_circuit_breaker_latency_seconds',
            'Request latency through circuit breaker in seconds',
            ['service', 'status'],  # status: success, failure
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Consecutive failures (gauge - resets to 0 on success)
        self.circuit_breaker_consecutive_failures = Gauge(
            f'{self.namespace}_circuit_breaker_consecutive_failures',
            'Current consecutive failure count',
            ['service'],
            registry=self.registry
        )

        # Time since last state change
        self.circuit_breaker_last_state_change_timestamp = Gauge(
            f'{self.namespace}_circuit_breaker_last_state_change_timestamp',
            'Unix timestamp of last circuit breaker state change',
            ['service'],
            registry=self.registry
        )

    def _init_dependency_metrics(self):
        """Initialize dependency health metrics."""

        # Dependency health status (gauge: 0=unhealthy, 1=degraded, 2=healthy)
        self.dependency_health_status = Gauge(
            f'{self.namespace}_dependency_health_status',
            'Dependency health status (0=unhealthy, 1=degraded, 2=healthy)',
            ['dependency'],  # dependency: database, redis, factor_broker, llm_provider, erp_sap
            registry=self.registry
        )

        # Dependency latency
        self.dependency_latency_ms = Gauge(
            f'{self.namespace}_dependency_latency_ms',
            'Last measured dependency latency in milliseconds',
            ['dependency'],
            registry=self.registry
        )

        # Dependency check total
        self.dependency_check_total = Counter(
            f'{self.namespace}_dependency_check_total',
            'Total dependency health checks performed',
            ['dependency', 'status'],  # status: success, failure
            registry=self.registry
        )

    # ========================================================================
    # RECORDING METHODS
    # ========================================================================

    def record_state_change(
        self,
        service: str,
        from_state: str,
        to_state: str
    ):
        """
        Record a circuit breaker state transition.

        Args:
            service: Service name (factor_broker, llm_provider, etc.)
            from_state: Previous state (closed, open, half_open)
            to_state: New state (closed, open, half_open)
        """
        # Update state gauge
        try:
            state_enum = CircuitState(to_state)
            self.circuit_breaker_state.labels(service=service).set(
                STATE_TO_NUMBER[state_enum]
            )
        except ValueError:
            # Invalid state, default to closed
            self.circuit_breaker_state.labels(service=service).set(0)

        # Increment state change counter
        self.circuit_breaker_state_changes_total.labels(
            service=service,
            from_state=from_state,
            to_state=to_state
        ).inc()

        # Update timestamp
        self.circuit_breaker_last_state_change_timestamp.labels(
            service=service
        ).set(time.time())

    def record_failure(
        self,
        service: str,
        latency_seconds: Optional[float] = None
    ):
        """
        Record a circuit breaker failure.

        Args:
            service: Service name
            latency_seconds: Request latency (optional)
        """
        self.circuit_breaker_failures_total.labels(service=service).inc()

        if latency_seconds is not None:
            self.circuit_breaker_latency_seconds.labels(
                service=service,
                status="failure"
            ).observe(latency_seconds)

        # Increment consecutive failures (caller should reset on success)
        current = self.circuit_breaker_consecutive_failures.labels(service=service)._value.get()
        self.circuit_breaker_consecutive_failures.labels(service=service).set(
            (current or 0) + 1
        )

    def record_success(
        self,
        service: str,
        latency_seconds: Optional[float] = None
    ):
        """
        Record a successful circuit breaker request.

        Args:
            service: Service name
            latency_seconds: Request latency (optional)
        """
        self.circuit_breaker_successes_total.labels(service=service).inc()

        if latency_seconds is not None:
            self.circuit_breaker_latency_seconds.labels(
                service=service,
                status="success"
            ).observe(latency_seconds)

        # Reset consecutive failures on success
        self.circuit_breaker_consecutive_failures.labels(service=service).set(0)

    def record_rejection(self, service: str):
        """
        Record a request rejection due to open circuit.

        Args:
            service: Service name
        """
        self.circuit_breaker_rejection_total.labels(service=service).inc()

    def set_dependency_health(
        self,
        dependency: str,
        status: str,
        latency_ms: Optional[float] = None
    ):
        """
        Update dependency health status.

        Args:
            dependency: Dependency name (database, redis, etc.)
            status: Health status (healthy, degraded, unhealthy)
            latency_ms: Latency in milliseconds (optional)
        """
        # Map status to numeric value
        status_map = {
            "unhealthy": 0,
            "degraded": 1,
            "healthy": 2
        }

        status_value = status_map.get(status, 0)
        self.dependency_health_status.labels(dependency=dependency).set(status_value)

        if latency_ms is not None:
            self.dependency_latency_ms.labels(dependency=dependency).set(latency_ms)

        # Record health check
        check_status = "success" if status in ["healthy", "degraded"] else "failure"
        self.dependency_check_total.labels(
            dependency=dependency,
            status=check_status
        ).inc()

    def update_from_circuit_breaker(self, service: str, stats: Any):
        """
        Update metrics from CircuitBreakerStats object.

        Args:
            service: Service name
            stats: CircuitBreakerStats object from resilience module
        """
        # Update state
        self.circuit_breaker_state.labels(service=service).set(
            STATE_TO_NUMBER[stats.state]
        )

        # Update consecutive failures
        self.circuit_breaker_consecutive_failures.labels(service=service).set(
            stats.failure_count
        )

        # Update timestamp
        self.circuit_breaker_last_state_change_timestamp.labels(service=service).set(
            stats.last_state_change
        )

    def export_text(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry).decode('utf-8')

    def get_content_type(self) -> str:
        """Get content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_global_circuit_breaker_metrics: Optional[CircuitBreakerMetrics] = None


def get_circuit_breaker_metrics() -> CircuitBreakerMetrics:
    """
    Get or create global circuit breaker metrics instance.

    Returns:
        CircuitBreakerMetrics singleton instance
    """
    global _global_circuit_breaker_metrics

    if _global_circuit_breaker_metrics is None:
        if PROMETHEUS_AVAILABLE:
            _global_circuit_breaker_metrics = CircuitBreakerMetrics()
        else:
            raise ImportError("Prometheus client not available")

    return _global_circuit_breaker_metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def track_circuit_breaker_call(service: str, metrics: Optional[CircuitBreakerMetrics] = None):
    """
    Decorator to automatically track circuit breaker calls.

    Usage:
        @track_circuit_breaker_call(service="factor_broker")
        async def fetch_emission_factors():
            # Your API call here
            return result

    Args:
        service: Service name
        metrics: CircuitBreakerMetrics instance (uses global if None)
    """
    def decorator(func):
        import asyncio
        from functools import wraps

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            m = metrics or get_circuit_breaker_metrics()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                m.record_success(service=service, latency_seconds=latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                m.record_failure(service=service, latency_seconds=latency)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            m = metrics or get_circuit_breaker_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                m.record_success(service=service, latency_seconds=latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                m.record_failure(service=service, latency_seconds=latency)
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create metrics
    metrics = CircuitBreakerMetrics()

    print("Circuit Breaker Metrics Example\n")
    print("=" * 80)

    # Simulate circuit breaker events

    # Factor Broker - normal operation
    metrics.record_success(service="factor_broker", latency_seconds=0.15)
    metrics.record_success(service="factor_broker", latency_seconds=0.12)
    metrics.set_dependency_health("factor_broker", "healthy", latency_ms=150)

    # LLM Provider - experiencing failures
    metrics.record_failure(service="llm_provider", latency_seconds=5.0)
    metrics.record_failure(service="llm_provider", latency_seconds=5.0)
    metrics.record_failure(service="llm_provider", latency_seconds=5.0)
    metrics.record_state_change("llm_provider", "closed", "open")
    metrics.set_dependency_health("llm_provider", "degraded", latency_ms=5000)

    # LLM Provider - rejecting requests
    metrics.record_rejection("llm_provider")
    metrics.record_rejection("llm_provider")

    # ERP SAP - healthy
    metrics.record_success(service="erp_sap", latency_seconds=0.8)
    metrics.set_dependency_health("erp_sap", "healthy", latency_ms=800)

    # Database - healthy
    metrics.set_dependency_health("database", "healthy", latency_ms=5)

    # Redis - healthy
    metrics.set_dependency_health("redis", "healthy", latency_ms=2)

    # Export metrics
    print("\nPROMETHEUS METRICS EXPORT (Circuit Breaker)")
    print("=" * 80)
    print(metrics.export_text())
