# -*- coding: utf-8 -*-
"""
Metrics collection for GreenLang using Prometheus
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import threading
from collections import defaultdict
from greenlang.determinism import DeterministicClock
from greenlang.intelligence import ChatMessage

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        push_to_gateway,
        REGISTRY,
    )
    from prometheus_client.exposition import start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def observe(self, amount):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            self.value = 0

        def inc(self):
            self.value += 1

        def dec(self):
            self.value -= 1

        def set(self, value):
            self.value = value

    class Summary:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, amount):
            pass

    class Info:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, value):
            pass

    CollectorRegistry = object
    REGISTRY = None

logger = logging.getLogger(__name__)


# Core metrics
pipeline_runs = Counter(
    "gl_pipeline_runs_total", "Total pipeline runs", ["pipeline", "status", "tenant_id"]
)

pipeline_duration = Histogram(
    "gl_pipeline_duration_seconds",
    "Pipeline execution time",
    ["pipeline", "tenant_id"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

active_executions = Gauge(
    "gl_active_executions", "Currently running pipelines", ["tenant_id"]
)

# Resource metrics
resource_usage = Gauge(
    "gl_resource_usage", "Resource usage metrics", ["resource_type", "tenant_id"]
)

cpu_usage = Gauge("gl_cpu_usage_percent", "CPU usage percentage", ["tenant_id"])

memory_usage = Gauge("gl_memory_usage_bytes", "Memory usage in bytes", ["tenant_id"])

disk_usage = Gauge("gl_disk_usage_bytes", "Disk usage in bytes", ["tenant_id", "path"])

# Pack metrics
pack_operations = Counter(
    "gl_pack_operations_total", "Pack operations", ["operation", "status", "tenant_id"]
)

pack_size = Histogram(
    "gl_pack_size_bytes",
    "Pack size distribution",
    ["pack_name", "tenant_id"],
    buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824),
)

# API metrics
api_requests = Counter(
    "gl_api_requests_total",
    "API requests",
    ["method", "endpoint", "status_code", "tenant_id"],
)

api_latency = Histogram(
    "gl_api_latency_seconds",
    "API request latency",
    ["method", "endpoint", "tenant_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Error metrics
errors = Counter(
    "gl_errors_total", "Total errors", ["error_type", "component", "tenant_id"]
)

# Cache metrics
cache_hits = Counter("gl_cache_hits_total", "Cache hits", ["cache_name", "tenant_id"])

cache_misses = Counter(
    "gl_cache_misses_total", "Cache misses", ["cache_name", "tenant_id"]
)

# Database metrics
db_queries = Counter(
    "gl_db_queries_total", "Database queries", ["query_type", "table", "tenant_id"]
)

db_query_duration = Histogram(
    "gl_db_query_duration_seconds",
    "Database query duration",
    ["query_type", "table", "tenant_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

db_connections = Gauge(
    "gl_db_connections", "Database connections", ["state", "tenant_id"]
)

# System info
system_info = Info("gl_system", "System information")


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class CustomMetric:
    """Custom metric definition"""

    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    metric: Optional[Any] = None


class MetricsCollector:
    """Centralized metrics collection"""

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        push_gateway: Optional[str] = None,
        job_name: str = "greenlang",
    ):
        """
        Initialize metrics collector

        Args:
            registry: Prometheus registry
            push_gateway: Push gateway URL
            job_name: Job name for push gateway
        """
        self.registry = registry or REGISTRY
        self.push_gateway = push_gateway
        self.job_name = job_name
        self.custom_metrics: Dict[str, CustomMetric] = {}
        self.collection_interval = 60  # seconds
        self._collection_thread = None
        self._stop_collection = threading.Event()

        # Initialize system info
        if PROMETHEUS_AVAILABLE:
            system_info.info(
                {"version": "1.0.0", "python_version": "3.11", "platform": "linux"}
            )

        logger.info("MetricsCollector initialized")

    def register_custom_metric(self, metric: CustomMetric):
        """
        Register a custom metric

        Args:
            metric: Custom metric definition
        """
        if metric.name in self.custom_metrics:
            logger.warning(f"Metric {metric.name} already registered")
            return

        # Create the actual metric
        if metric.type == MetricType.COUNTER:
            metric.metric = Counter(
                metric.name, metric.description, metric.labels, registry=self.registry
            )
        elif metric.type == MetricType.GAUGE:
            metric.metric = Gauge(
                metric.name, metric.description, metric.labels, registry=self.registry
            )
        elif metric.type == MetricType.HISTOGRAM:
            metric.metric = Histogram(
                metric.name,
                metric.description,
                metric.labels,
                buckets=metric.buckets,
                registry=self.registry,
            )
        elif metric.type == MetricType.SUMMARY:
            metric.metric = Summary(
                metric.name, metric.description, metric.labels, registry=self.registry
            )
        elif metric.type == MetricType.INFO:
            metric.metric = Info(
                metric.name, metric.description, metric.labels, registry=self.registry
            )

        self.custom_metrics[metric.name] = metric
        logger.info(f"Registered custom metric: {metric.name}")

    def record_metric(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a custom metric value

        Args:
            metric_name: Metric name
            value: Metric value
            labels: Label values
        """
        metric = self.custom_metrics.get(metric_name)
        if not metric or not metric.metric:
            logger.warning(f"Metric {metric_name} not found")
            return

        labels = labels or {}

        if metric.type == MetricType.COUNTER:
            metric.metric.labels(**labels).inc(value)
        elif metric.type == MetricType.GAUGE:
            metric.metric.labels(**labels).set(value)
        elif metric.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            metric.metric.labels(**labels).observe(value)
        elif metric.type == MetricType.INFO:
            metric.metric.info(labels)

    def collect_system_metrics(self, tenant_id: str = "system"):
        """
        Collect system-level metrics

        Args:
            tenant_id: Tenant ID for metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.labels(tenant_id=tenant_id).set(cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage.labels(tenant_id=tenant_id).set(memory.used)
            resource_usage.labels(
                resource_type="memory_percent", tenant_id=tenant_id
            ).set(memory.percent)

            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage.labels(
                        tenant_id=tenant_id, path=partition.mountpoint
                    ).set(usage.used)
                except PermissionError:
                    continue

            # Network metrics
            net_io = psutil.net_io_counters()
            resource_usage.labels(
                resource_type="network_bytes_sent", tenant_id=tenant_id
            ).set(net_io.bytes_sent)
            resource_usage.labels(
                resource_type="network_bytes_recv", tenant_id=tenant_id
            ).set(net_io.bytes_recv)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            errors.labels(
                error_type="metric_collection", component="system", tenant_id=tenant_id
            ).inc()

    def start_collection(self, port: int = 8000):
        """
        Start metrics collection server

        Args:
            port: Port for metrics endpoint
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return

        # Start HTTP server
        start_http_server(port, registry=self.registry)
        logger.info(f"Metrics server started on port {port}")

        # Start background collection
        self._collection_thread = threading.Thread(
            target=self._collect_loop, daemon=True
        )
        self._collection_thread.start()

    def stop_collection(self):
        """Stop metrics collection"""
        self._stop_collection.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)

    def _collect_loop(self):
        """Background collection loop"""
        while not self._stop_collection.is_set():
            try:
                self.collect_system_metrics()

                # Push to gateway if configured
                if self.push_gateway and PROMETHEUS_AVAILABLE:
                    push_to_gateway(
                        self.push_gateway, job=self.job_name, registry=self.registry
                    )

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")

            self._stop_collection.wait(self.collection_interval)

    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format

        Returns:
            Metrics data
        """
        if not PROMETHEUS_AVAILABLE:
            return b""

        return generate_latest(self.registry)


def track_execution(pipeline: str, tenant_id: str = "default"):
    """
    Decorator to track pipeline execution metrics

    Args:
        pipeline: Pipeline name
        tenant_id: Tenant ID
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Increment active executions
            active_executions.labels(tenant_id=tenant_id).inc()
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success
                pipeline_runs.labels(
                    pipeline=pipeline, status="success", tenant_id=tenant_id
                ).inc()

                return result

            except Exception as e:
                # Record failure
                pipeline_runs.labels(
                    pipeline=pipeline, status="failure", tenant_id=tenant_id
                ).inc()

                errors.labels(
                    error_type=type(e).__name__,
                    component="pipeline",
                    tenant_id=tenant_id,
                ).inc()

                raise

            finally:
                # Record duration
                duration = time.time() - start_time
                pipeline_duration.labels(
                    pipeline=pipeline, tenant_id=tenant_id
                ).observe(duration)

                # Decrement active executions
                active_executions.labels(tenant_id=tenant_id).dec()

        return wrapper

    return decorator


def track_resource(resource_type: str, tenant_id: str = "default"):
    """
    Decorator to track resource usage

    Args:
        resource_type: Type of resource
        tenant_id: Tenant ID
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Track resource before
            process = psutil.Process()
            mem_before = process.memory_info().rss

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Track resource after
                mem_after = process.memory_info().rss
                mem_used = mem_after - mem_before

                resource_usage.labels(
                    resource_type=f"{resource_type}_memory", tenant_id=tenant_id
                ).set(mem_used)

                duration = time.time() - start_time
                resource_usage.labels(
                    resource_type=f"{resource_type}_duration", tenant_id=tenant_id
                ).set(duration)

        return wrapper

    return decorator


def track_api_request(method: str, endpoint: str, tenant_id: str = "default"):
    """
    Decorator to track API requests

    Args:
        method: HTTP method
        endpoint: API endpoint
        tenant_id: Tenant ID
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Assume result has status_code attribute
                status_code = getattr(result, "status_code", 200)

                api_requests.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code),
                    tenant_id=tenant_id,
                ).inc()

                return result

            except Exception:
                api_requests.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code="500",
                    tenant_id=tenant_id,
                ).inc()
                raise

            finally:
                duration = time.time() - start_time
                api_latency.labels(
                    method=method, endpoint=endpoint, tenant_id=tenant_id
                ).observe(duration)

        return wrapper

    return decorator


def track_cache(cache_name: str, tenant_id: str = "default"):
    """
    Track cache hit/miss

    Args:
        cache_name: Cache name
        tenant_id: Tenant ID
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Check if cache hit (assuming None means miss)
            if result is not None:
                cache_hits.labels(cache_name=cache_name, tenant_id=tenant_id).inc()
            else:
                cache_misses.labels(cache_name=cache_name, tenant_id=tenant_id).inc()

            return result

        return wrapper

    return decorator


def track_database_query(query_type: str, table: str, tenant_id: str = "default"):
    """
    Track database queries

    Args:
        query_type: Type of query (select, insert, update, delete)
        table: Table name
        tenant_id: Tenant ID
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                db_queries.labels(
                    query_type=query_type, table=table, tenant_id=tenant_id
                ).inc()

                return result

            finally:
                duration = time.time() - start_time
                db_query_duration.labels(
                    query_type=query_type, table=table, tenant_id=tenant_id
                ).observe(duration)

        return wrapper

    return decorator


class MetricsAggregator:
    """Aggregate metrics over time windows"""

    def __init__(self):
        """Initialize metrics aggregator"""
        self.metrics_buffer: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.retention_hours = 24

    def add_metric(
        self, metric_name: str, value: float, timestamp: Optional[datetime] = None
    ):
        """
        Add metric to buffer

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp (default: now)
        """
        timestamp = timestamp or DeterministicClock.utcnow()
        self.metrics_buffer[metric_name].append((timestamp, value))

        # Clean old metrics
        self._clean_old_metrics(metric_name)

    def get_aggregates(
        self, metric_name: str, window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get aggregated metrics

        Args:
            metric_name: Metric name
            window_minutes: Time window in minutes

        Returns:
            Aggregated metrics (min, max, avg, p50, p95, p99)
        """
        cutoff = DeterministicClock.utcnow() - timedelta(minutes=window_minutes)
        values = [v for t, v in self.metrics_buffer.get(metric_name, []) if t >= cutoff]

        if not values:
            return {}

        values.sort()

        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
            "count": len(values),
        }

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0

        k = (len(values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f < len(values) - 1 else f

        if f == c:
            return values[f]

        return values[f] + (k - f) * (values[c] - values[f])

    def _clean_old_metrics(self, metric_name: str):
        """Remove old metrics from buffer"""
        cutoff = DeterministicClock.utcnow() - timedelta(hours=self.retention_hours)
        self.metrics_buffer[metric_name] = [
            (t, v) for t, v in self.metrics_buffer[metric_name] if t >= cutoff
        ]


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_metrics_aggregator: Optional[MetricsAggregator] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if not _metrics_collector:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics_aggregator() -> MetricsAggregator:
    """Get global metrics aggregator"""
    global _metrics_aggregator
    if not _metrics_aggregator:
        _metrics_aggregator = MetricsAggregator()
    return _metrics_aggregator


def get_metrics_registry() -> Optional[CollectorRegistry]:
    """Get Prometheus registry"""
    return REGISTRY if PROMETHEUS_AVAILABLE else None
