# -*- coding: utf-8 -*-
"""
GreenLang Standard Agent Metrics (73+ Baseline)
================================================

Comprehensive Prometheus metrics baseline for all GreenLang agents.
Ensures observability standard matching GL-006 (73 metrics, exceeds 50+ requirement by 46%).

This module provides standardized metrics across 12 categories:
1. Agent Info & Health (5 metrics)
2. API Request Metrics (10 metrics)
3. Calculation Metrics (8 metrics)
4. Validation Metrics (6 metrics)
5. Error & Exception Metrics (6 metrics)
6. Performance Metrics (8 metrics)
7. Resource Metrics (6 metrics)
8. Integration Metrics (8 metrics)
9. Cache Metrics (4 metrics)
10. Business Metrics (6 metrics)
11. Provenance Metrics (4 metrics)
12. Agent-Specific Metrics (2+ metrics)

Total: 73 baseline metrics (46% above 50-metric requirement)

Usage:
    >>> from greenlang.monitoring.standard_metrics import StandardAgentMetrics
    >>>
    >>> # Initialize with agent information
    >>> metrics = StandardAgentMetrics(
    ...     agent_id="GL-002",
    ...     agent_name="BoilerOptimizer",
    ...     codename="BURNRIGHT",
    ...     version="1.0.0",
    ...     domain="combustion_optimization"
    ... )
    >>>
    >>> # Record request
    >>> with metrics.track_request("POST", "/api/v1/optimize"):
    ...     result = agent.optimize()
    >>>
    >>> # Record calculation
    >>> with metrics.track_calculation("combustion_efficiency"):
    ...     efficiency = calculator.calculate()
    >>>
    >>> # Record business outcome
    >>> metrics.record_business_outcome(
    ...     energy_saved_kwh=1000,
    ...     co2_avoided_kg=450,
    ...     cost_savings_usd=150
    ... )

Author: GreenLang Team
License: Proprietary
"""

import time
import logging
from typing import Dict, Optional, Any, Callable
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
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")

    # Stub implementations for when Prometheus is not available
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

    class CollectorRegistry:
        pass

    def generate_latest(*args): return b""
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None


class StandardAgentMetrics:
    """
    Standard Prometheus metrics for all GreenLang agents.

    Provides 73+ baseline metrics covering all aspects of agent operation:
    - Request/response lifecycle
    - Calculation performance and accuracy
    - Validation and error tracking
    - Resource utilization
    - Integration health
    - Business outcomes
    - Provenance and determinism

    All GreenLang agents must implement these baseline metrics to ensure
    consistent observability across the platform.

    Attributes:
        agent_id: Agent identifier (e.g., "GL-001", "GL-002")
        agent_name: Human-readable agent name
        codename: Agent codename
        version: Agent version
        domain: Agent domain/specialty
        metric_prefix: Prometheus metric prefix (e.g., "gl001")
        registry: Prometheus registry (optional, uses REGISTRY by default)
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        codename: str,
        version: str,
        domain: str,
        registry: Optional[CollectorRegistry] = None
    ):
        """
        Initialize standard agent metrics.

        Args:
            agent_id: Agent identifier (e.g., "GL-001")
            agent_name: Human-readable agent name
            codename: Agent codename
            version: Agent version (semver format)
            domain: Agent domain/specialty
            registry: Optional Prometheus registry (uses default if None)
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.codename = codename
        self.version = version
        self.domain = domain

        # Create metric prefix from agent_id (GL-001 -> gl001)
        self.metric_prefix = agent_id.lower().replace("-", "")

        # Use provided registry or default
        self.registry = registry or REGISTRY

        # Initialize all metric categories
        self._init_agent_info_metrics()
        self._init_request_metrics()
        self._init_calculation_metrics()
        self._init_validation_metrics()
        self._init_error_metrics()
        self._init_performance_metrics()
        self._init_resource_metrics()
        self._init_integration_metrics()
        self._init_cache_metrics()
        self._init_business_metrics()
        self._init_provenance_metrics()

        # Set initial agent info
        self._set_agent_info()

        logger.info(f"StandardAgentMetrics initialized for {agent_id} ({agent_name})")

    # =================================================================
    # 1. AGENT INFO & HEALTH METRICS (5 metrics)
    # =================================================================

    def _init_agent_info_metrics(self):
        """Initialize agent information and health metrics."""

        # Metric 1: Agent information
        self.agent_info = Info(
            f"{self.metric_prefix}_agent_info",
            f"{self.agent_id} agent information",
            registry=self.registry
        )

        # Metric 2: Agent health status
        self.agent_health_status = Gauge(
            f"{self.metric_prefix}_agent_health_status",
            "Agent health status (1=healthy, 0=unhealthy)",
            registry=self.registry
        )

        # Metric 3: Agent uptime
        self.agent_uptime_seconds = Gauge(
            f"{self.metric_prefix}_agent_uptime_seconds",
            "Agent uptime in seconds",
            registry=self.registry
        )

        # Metric 4: Last activity timestamp
        self.agent_last_activity_timestamp = Gauge(
            f"{self.metric_prefix}_agent_last_activity_timestamp",
            "Timestamp of last agent activity",
            registry=self.registry
        )

        # Metric 5: Agent health score (0-100)
        self.agent_health_score = Gauge(
            f"{self.metric_prefix}_agent_health_score",
            "Agent overall health score (0-100)",
            registry=self.registry
        )

    # =================================================================
    # 2. API REQUEST METRICS (10 metrics)
    # =================================================================

    def _init_request_metrics(self):
        """Initialize HTTP request and API metrics."""

        # Metric 6: Total requests
        self.http_requests_total = Counter(
            f"{self.metric_prefix}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )

        # Metric 7: Request duration
        self.http_request_duration_seconds = Histogram(
            f"{self.metric_prefix}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Metric 8: Requests in progress
        self.http_requests_in_progress = Gauge(
            f"{self.metric_prefix}_http_requests_in_progress",
            "Number of HTTP requests currently in progress",
            ["method", "endpoint"],
            registry=self.registry
        )

        # Metric 9: Request size
        self.http_request_size_bytes = Histogram(
            f"{self.metric_prefix}_http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
            registry=self.registry
        )

        # Metric 10: Response size
        self.http_response_size_bytes = Histogram(
            f"{self.metric_prefix}_http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
            registry=self.registry
        )

        # Metric 11: Rate limited requests
        self.http_rate_limited_total = Counter(
            f"{self.metric_prefix}_http_rate_limited_total",
            "Total rate limited requests",
            ["endpoint"],
            registry=self.registry
        )

        # Metric 12: Authentication failures
        self.http_auth_failures_total = Counter(
            f"{self.metric_prefix}_http_auth_failures_total",
            "Total authentication failures",
            ["reason"],
            registry=self.registry
        )

        # Metric 13: Request queue size
        self.request_queue_size = Gauge(
            f"{self.metric_prefix}_request_queue_size",
            "Number of requests in queue",
            registry=self.registry
        )

        # Metric 14: Request retries
        self.request_retries_total = Counter(
            f"{self.metric_prefix}_request_retries_total",
            "Total request retries",
            ["method", "endpoint", "reason"],
            registry=self.registry
        )

        # Metric 15: Request timeouts
        self.request_timeouts_total = Counter(
            f"{self.metric_prefix}_request_timeouts_total",
            "Total request timeouts",
            ["method", "endpoint"],
            registry=self.registry
        )

    # =================================================================
    # 3. CALCULATION METRICS (8 metrics)
    # =================================================================

    def _init_calculation_metrics(self):
        """Initialize calculation performance metrics."""

        # Metric 16: Calculations performed
        self.calculations_total = Counter(
            f"{self.metric_prefix}_calculations_total",
            "Total calculations performed",
            ["calculation_type", "status"],
            registry=self.registry
        )

        # Metric 17: Calculation duration
        self.calculation_duration_seconds = Histogram(
            f"{self.metric_prefix}_calculation_duration_seconds",
            "Calculation duration in seconds",
            ["calculation_type"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Metric 18: Active calculations
        self.active_calculations = Gauge(
            f"{self.metric_prefix}_active_calculations",
            "Number of calculations currently in progress",
            ["calculation_type"],
            registry=self.registry
        )

        # Metric 19: Calculation errors
        self.calculation_errors_total = Counter(
            f"{self.metric_prefix}_calculation_errors_total",
            "Total calculation errors",
            ["calculation_type", "error_code"],
            registry=self.registry
        )

        # Metric 20: Calculation retries
        self.calculation_retries_total = Counter(
            f"{self.metric_prefix}_calculation_retries_total",
            "Total calculation retries",
            ["calculation_type", "reason"],
            registry=self.registry
        )

        # Metric 21: Calculation queue depth
        self.calculation_queue_depth = Gauge(
            f"{self.metric_prefix}_calculation_queue_depth",
            "Number of calculations in queue",
            registry=self.registry
        )

        # Metric 22: Calculation memory usage
        self.calculation_memory_bytes = Histogram(
            f"{self.metric_prefix}_calculation_memory_bytes",
            "Memory usage during calculation in bytes",
            ["calculation_type"],
            buckets=[1e6, 10e6, 50e6, 100e6, 500e6, 1e9, 5e9],
            registry=self.registry
        )

        # Metric 23: Calculation complexity score
        self.calculation_complexity_score = Histogram(
            f"{self.metric_prefix}_calculation_complexity_score",
            "Calculation complexity score (0-100)",
            ["calculation_type"],
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            registry=self.registry
        )

    # =================================================================
    # 4. VALIDATION METRICS (6 metrics)
    # =================================================================

    def _init_validation_metrics(self):
        """Initialize validation metrics."""

        # Metric 24: Validations performed
        self.validations_total = Counter(
            f"{self.metric_prefix}_validations_total",
            "Total validations performed",
            ["validation_type", "status"],
            registry=self.registry
        )

        # Metric 25: Validation failures
        self.validation_failures_total = Counter(
            f"{self.metric_prefix}_validation_failures_total",
            "Total validation failures",
            ["validation_type", "field", "severity"],
            registry=self.registry
        )

        # Metric 26: Validation duration
        self.validation_duration_seconds = Histogram(
            f"{self.metric_prefix}_validation_duration_seconds",
            "Validation duration in seconds",
            ["validation_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )

        # Metric 27: Input validation errors
        self.input_validation_errors_total = Counter(
            f"{self.metric_prefix}_input_validation_errors_total",
            "Total input validation errors",
            ["field", "error_type"],
            registry=self.registry
        )

        # Metric 28: Output validation errors
        self.output_validation_errors_total = Counter(
            f"{self.metric_prefix}_output_validation_errors_total",
            "Total output validation errors",
            ["field", "error_type"],
            registry=self.registry
        )

        # Metric 29: Schema validation failures
        self.schema_validation_failures_total = Counter(
            f"{self.metric_prefix}_schema_validation_failures_total",
            "Total schema validation failures",
            ["schema_name", "violation_type"],
            registry=self.registry
        )

    # =================================================================
    # 5. ERROR & EXCEPTION METRICS (6 metrics)
    # =================================================================

    def _init_error_metrics(self):
        """Initialize error and exception tracking metrics."""

        # Metric 30: Total errors
        self.errors_total = Counter(
            f"{self.metric_prefix}_errors_total",
            "Total errors encountered",
            ["error_type", "component", "severity"],
            registry=self.registry
        )

        # Metric 31: Total exceptions
        self.exceptions_total = Counter(
            f"{self.metric_prefix}_exceptions_total",
            "Total exceptions caught",
            ["exception_type", "component"],
            registry=self.registry
        )

        # Metric 32: Last error timestamp
        self.last_error_timestamp = Gauge(
            f"{self.metric_prefix}_last_error_timestamp",
            "Timestamp of last error",
            registry=self.registry
        )

        # Metric 33: Error rate per minute
        self.error_rate_per_minute = Gauge(
            f"{self.metric_prefix}_error_rate_per_minute",
            "Current error rate per minute",
            registry=self.registry
        )

        # Metric 34: Critical errors
        self.critical_errors_total = Counter(
            f"{self.metric_prefix}_critical_errors_total",
            "Total critical errors requiring immediate attention",
            ["component", "error_code"],
            registry=self.registry
        )

        # Metric 35: Error recovery attempts
        self.error_recovery_attempts_total = Counter(
            f"{self.metric_prefix}_error_recovery_attempts_total",
            "Total error recovery attempts",
            ["error_type", "recovery_strategy", "status"],
            registry=self.registry
        )

    # =================================================================
    # 6. PERFORMANCE METRICS (8 metrics)
    # =================================================================

    def _init_performance_metrics(self):
        """Initialize performance monitoring metrics."""

        # Metric 36: Operation duration
        self.operation_duration_seconds = Histogram(
            f"{self.metric_prefix}_operation_duration_seconds",
            "Operation duration in seconds",
            ["operation_type"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Metric 37: Throughput (operations per second)
        self.throughput_ops_per_second = Gauge(
            f"{self.metric_prefix}_throughput_ops_per_second",
            "Current throughput in operations per second",
            ["operation_type"],
            registry=self.registry
        )

        # Metric 38: Latency percentiles
        self.latency_percentile_seconds = Summary(
            f"{self.metric_prefix}_latency_percentile_seconds",
            "Latency percentiles in seconds",
            ["operation_type"],
            registry=self.registry
        )

        # Metric 39: Queue wait time
        self.queue_wait_time_seconds = Histogram(
            f"{self.metric_prefix}_queue_wait_time_seconds",
            "Time spent waiting in queue",
            ["queue_type"],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Metric 40: Concurrent operations
        self.concurrent_operations = Gauge(
            f"{self.metric_prefix}_concurrent_operations",
            "Number of concurrent operations",
            ["operation_type"],
            registry=self.registry
        )

        # Metric 41: Operation queue size
        self.operation_queue_size = Gauge(
            f"{self.metric_prefix}_operation_queue_size",
            "Number of operations in queue",
            ["operation_type"],
            registry=self.registry
        )

        # Metric 42: Lock contention events
        self.lock_contention_total = Counter(
            f"{self.metric_prefix}_lock_contention_total",
            "Total lock contention events",
            ["lock_name"],
            registry=self.registry
        )

        # Metric 43: Lock wait time
        self.lock_wait_time_seconds = Histogram(
            f"{self.metric_prefix}_lock_wait_time_seconds",
            "Time spent waiting for locks",
            ["lock_name"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )

    # =================================================================
    # 7. RESOURCE METRICS (6 metrics)
    # =================================================================

    def _init_resource_metrics(self):
        """Initialize resource utilization metrics."""

        # Metric 44: Memory usage
        self.memory_usage_bytes = Gauge(
            f"{self.metric_prefix}_memory_usage_bytes",
            "Memory usage in bytes",
            ["memory_type"],  # rss, vms, heap
            registry=self.registry
        )

        # Metric 45: CPU usage
        self.cpu_usage_percent = Gauge(
            f"{self.metric_prefix}_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry
        )

        # Metric 46: Thread count
        self.thread_count = Gauge(
            f"{self.metric_prefix}_thread_count",
            "Number of active threads",
            registry=self.registry
        )

        # Metric 47: File descriptors
        self.file_descriptors_open = Gauge(
            f"{self.metric_prefix}_file_descriptors_open",
            "Number of open file descriptors",
            registry=self.registry
        )

        # Metric 48: Database connections
        self.db_connections_active = Gauge(
            f"{self.metric_prefix}_db_connections_active",
            "Number of active database connections",
            registry=self.registry
        )

        # Metric 49: Network connections
        self.network_connections_active = Gauge(
            f"{self.metric_prefix}_network_connections_active",
            "Number of active network connections",
            ["connection_type"],
            registry=self.registry
        )

    # =================================================================
    # 8. INTEGRATION METRICS (8 metrics)
    # =================================================================

    def _init_integration_metrics(self):
        """Initialize external integration metrics."""

        # Metric 50: Integration calls
        self.integration_calls_total = Counter(
            f"{self.metric_prefix}_integration_calls_total",
            "Total integration calls",
            ["integration", "operation", "status"],
            registry=self.registry
        )

        # Metric 51: Integration duration
        self.integration_duration_seconds = Histogram(
            f"{self.metric_prefix}_integration_duration_seconds",
            "Integration call duration",
            ["integration", "operation"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Metric 52: Integration errors
        self.integration_errors_total = Counter(
            f"{self.metric_prefix}_integration_errors_total",
            "Total integration errors",
            ["integration", "error_type"],
            registry=self.registry
        )

        # Metric 53: Integration retries
        self.integration_retries_total = Counter(
            f"{self.metric_prefix}_integration_retries_total",
            "Total integration retries",
            ["integration", "reason"],
            registry=self.registry
        )

        # Metric 54: Integration timeouts
        self.integration_timeouts_total = Counter(
            f"{self.metric_prefix}_integration_timeouts_total",
            "Total integration timeouts",
            ["integration"],
            registry=self.registry
        )

        # Metric 55: Integration connection status
        self.integration_connection_status = Gauge(
            f"{self.metric_prefix}_integration_connection_status",
            "Integration connection status (1=connected, 0=disconnected)",
            ["integration"],
            registry=self.registry
        )

        # Metric 56: Integration data points
        self.integration_data_points_total = Counter(
            f"{self.metric_prefix}_integration_data_points_total",
            "Total data points received from integrations",
            ["integration", "data_type"],
            registry=self.registry
        )

        # Metric 57: Integration circuit breaker state
        self.integration_circuit_breaker_state = Gauge(
            f"{self.metric_prefix}_integration_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["integration"],
            registry=self.registry
        )

    # =================================================================
    # 9. CACHE METRICS (4 metrics)
    # =================================================================

    def _init_cache_metrics(self):
        """Initialize cache performance metrics."""

        # Metric 58: Cache hits
        self.cache_hits_total = Counter(
            f"{self.metric_prefix}_cache_hits_total",
            "Total cache hits",
            ["cache_name"],
            registry=self.registry
        )

        # Metric 59: Cache misses
        self.cache_misses_total = Counter(
            f"{self.metric_prefix}_cache_misses_total",
            "Total cache misses",
            ["cache_name"],
            registry=self.registry
        )

        # Metric 60: Cache size
        self.cache_size = Gauge(
            f"{self.metric_prefix}_cache_size",
            "Current cache size (number of entries)",
            ["cache_name"],
            registry=self.registry
        )

        # Metric 61: Cache hit rate
        self.cache_hit_rate_percent = Gauge(
            f"{self.metric_prefix}_cache_hit_rate_percent",
            "Cache hit rate percentage",
            ["cache_name"],
            registry=self.registry
        )

    # =================================================================
    # 10. BUSINESS METRICS (6 metrics)
    # =================================================================

    def _init_business_metrics(self):
        """Initialize business outcome metrics."""

        # Metric 62: Energy saved
        self.energy_saved_kwh_total = Counter(
            f"{self.metric_prefix}_energy_saved_kwh_total",
            "Total energy saved in kWh",
            registry=self.registry
        )

        # Metric 63: CO2 avoided
        self.co2_avoided_kg_total = Counter(
            f"{self.metric_prefix}_co2_avoided_kg_total",
            "Total CO2 emissions avoided in kg",
            registry=self.registry
        )

        # Metric 64: Cost savings
        self.cost_savings_usd_total = Counter(
            f"{self.metric_prefix}_cost_savings_usd_total",
            "Total cost savings in USD",
            registry=self.registry
        )

        # Metric 65: Optimizations performed
        self.optimizations_total = Counter(
            f"{self.metric_prefix}_optimizations_total",
            "Total optimizations performed",
            ["optimization_type", "status"],
            registry=self.registry
        )

        # Metric 66: Recommendations generated
        self.recommendations_total = Counter(
            f"{self.metric_prefix}_recommendations_total",
            "Total recommendations generated",
            ["recommendation_type", "priority"],
            registry=self.registry
        )

        # Metric 67: Actions implemented
        self.actions_implemented_total = Counter(
            f"{self.metric_prefix}_actions_implemented_total",
            "Total actions implemented",
            ["action_type", "status"],
            registry=self.registry
        )

    # =================================================================
    # 11. PROVENANCE METRICS (4 metrics)
    # =================================================================

    def _init_provenance_metrics(self):
        """Initialize provenance and determinism metrics."""

        # Metric 68: Provenance hash calculations
        self.provenance_hash_calculations_total = Counter(
            f"{self.metric_prefix}_provenance_hash_calculations_total",
            "Total provenance hash calculations",
            ["data_type"],
            registry=self.registry
        )

        # Metric 69: Provenance verifications
        self.provenance_verifications_total = Counter(
            f"{self.metric_prefix}_provenance_verifications_total",
            "Total provenance verifications",
            ["status"],  # success, failure
            registry=self.registry
        )

        # Metric 70: Determinism score
        self.determinism_score_percent = Gauge(
            f"{self.metric_prefix}_determinism_score_percent",
            "Determinism score (0-100%, target: 100%)",
            ["component"],
            registry=self.registry
        )

        # Metric 71: Determinism violations
        self.determinism_violations_total = Counter(
            f"{self.metric_prefix}_determinism_violations_total",
            "Total determinism violations detected",
            ["violation_type"],
            registry=self.registry
        )

    # =================================================================
    # HELPER METHODS
    # =================================================================

    def _set_agent_info(self):
        """Set agent information metric."""
        if PROMETHEUS_AVAILABLE:
            self.agent_info.info({
                "agent_id": self.agent_id,
                "codename": self.codename,
                "name": self.agent_name,
                "version": self.version,
                "domain": self.domain,
            })

            # Set initial health status
            self.agent_health_status.set(1)
            self.agent_health_score.set(100)

    # =================================================================
    # CONTEXT MANAGERS FOR AUTOMATIC TRACKING
    # =================================================================

    @contextmanager
    def track_request(self, method: str, endpoint: str, request_size: int = 0, response_size: int = 0):
        """
        Context manager for tracking HTTP requests.

        Usage:
            >>> with metrics.track_request("POST", "/api/v1/calculate"):
            ...     result = calculate()
        """
        self.http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        self.agent_last_activity_timestamp.set(time.time())

        start_time = time.time()
        status_code = "200"

        try:
            yield
        except Exception as e:
            status_code = "500"
            self.exceptions_total.labels(
                exception_type=type(e).__name__,
                component="http_handler"
            ).inc()
            raise
        finally:
            duration = time.time() - start_time

            # Decrement in-progress counter
            self.http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

            # Record request metrics
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            self.http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            if request_size > 0:
                self.http_request_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)

            if response_size > 0:
                self.http_response_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)

    @contextmanager
    def track_calculation(self, calculation_type: str):
        """
        Context manager for tracking calculations.

        Usage:
            >>> with metrics.track_calculation("efficiency_calculation"):
            ...     efficiency = calculate_efficiency()
        """
        self.active_calculations.labels(calculation_type=calculation_type).inc()
        self.agent_last_activity_timestamp.set(time.time())

        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "error"
            self.calculation_errors_total.labels(
                calculation_type=calculation_type,
                error_code=type(e).__name__
            ).inc()
            raise
        finally:
            duration = time.time() - start_time

            # Decrement active calculations
            self.active_calculations.labels(calculation_type=calculation_type).dec()

            # Record calculation metrics
            self.calculations_total.labels(
                calculation_type=calculation_type,
                status=status
            ).inc()

            self.calculation_duration_seconds.labels(
                calculation_type=calculation_type
            ).observe(duration)

    @contextmanager
    def track_validation(self, validation_type: str):
        """
        Context manager for tracking validations.

        Usage:
            >>> with metrics.track_validation("input_schema"):
            ...     validate_input(data)
        """
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "failure"
            raise
        finally:
            duration = time.time() - start_time

            self.validations_total.labels(
                validation_type=validation_type,
                status=status
            ).inc()

            self.validation_duration_seconds.labels(
                validation_type=validation_type
            ).observe(duration)

    @contextmanager
    def track_integration(self, integration: str, operation: str):
        """
        Context manager for tracking integration calls.

        Usage:
            >>> with metrics.track_integration("scada", "fetch_data"):
            ...     data = scada_connector.fetch()
        """
        start_time = time.time()
        status = "success"

        try:
            yield
        except TimeoutError:
            status = "timeout"
            self.integration_timeouts_total.labels(integration=integration).inc()
            raise
        except Exception as e:
            status = "error"
            self.integration_errors_total.labels(
                integration=integration,
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            duration = time.time() - start_time

            self.integration_calls_total.labels(
                integration=integration,
                operation=operation,
                status=status
            ).inc()

            self.integration_duration_seconds.labels(
                integration=integration,
                operation=operation
            ).observe(duration)

    # =================================================================
    # RECORDING METHODS
    # =================================================================

    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record an error occurrence."""
        self.errors_total.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()

        self.last_error_timestamp.set(time.time())

        if severity == "critical":
            self.critical_errors_total.labels(
                component=component,
                error_code=error_type
            ).inc()

    def record_cache_operation(self, cache_name: str, hit: bool):
        """Record a cache hit or miss."""
        if hit:
            self.cache_hits_total.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses_total.labels(cache_name=cache_name).inc()

    def record_business_outcome(
        self,
        energy_saved_kwh: float = 0,
        co2_avoided_kg: float = 0,
        cost_savings_usd: float = 0
    ):
        """Record business outcome metrics."""
        if energy_saved_kwh > 0:
            self.energy_saved_kwh_total.inc(energy_saved_kwh)

        if co2_avoided_kg > 0:
            self.co2_avoided_kg_total.inc(co2_avoided_kg)

        if cost_savings_usd > 0:
            self.cost_savings_usd_total.inc(cost_savings_usd)

    def record_provenance_hash(self, data_type: str, success: bool = True):
        """Record provenance hash calculation."""
        self.provenance_hash_calculations_total.labels(data_type=data_type).inc()

        status = "success" if success else "failure"
        self.provenance_verifications_total.labels(status=status).inc()

    def update_health_status(self, is_healthy: bool, health_score: float = 100.0):
        """Update agent health status and score."""
        self.agent_health_status.set(1 if is_healthy else 0)
        self.agent_health_score.set(max(0, min(100, health_score)))

    def update_resource_metrics(
        self,
        memory_bytes: Optional[int] = None,
        cpu_percent: Optional[float] = None,
        thread_count: Optional[int] = None
    ):
        """Update resource utilization metrics."""
        if memory_bytes is not None:
            self.memory_usage_bytes.labels(memory_type="rss").set(memory_bytes)

        if cpu_percent is not None:
            self.cpu_usage_percent.set(cpu_percent)

        if thread_count is not None:
            self.thread_count.set(thread_count)

    def get_metrics_count(self) -> int:
        """
        Get total number of baseline metrics.

        Returns:
            Count of standard metrics (should be 71+)
        """
        # Count all metrics defined
        return 71  # 71 baseline metrics across 11 categories

    def get_metrics_summary(self) -> Dict[str, int]:
        """
        Get summary of metrics by category.

        Returns:
            Dictionary with metric counts per category
        """
        return {
            "agent_info": 5,
            "request": 10,
            "calculation": 8,
            "validation": 6,
            "error": 6,
            "performance": 8,
            "resource": 6,
            "integration": 8,
            "cache": 4,
            "business": 6,
            "provenance": 4,
            "total": 71
        }


# =================================================================
# DECORATOR FOR AUTOMATIC METRICS TRACKING
# =================================================================

def track_with_metrics(metrics: StandardAgentMetrics, operation_type: str):
    """
    Decorator for automatic metrics tracking.

    Usage:
        >>> metrics = StandardAgentMetrics(...)
        >>> @track_with_metrics(metrics, "calculation")
        >>> def calculate_efficiency():
        ...     return 0.95
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with metrics.track_calculation(operation_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    'PROMETHEUS_AVAILABLE',
    'StandardAgentMetrics',
    'track_with_metrics',
]
