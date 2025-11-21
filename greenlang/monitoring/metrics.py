# -*- coding: utf-8 -*-
"""
GreenLang Metrics Collection and Prometheus Integration
======================================================

Production-ready metrics collection system for GreenLang v0.2.0 with:
- Prometheus metrics export
- Custom metric definitions
- Resource usage monitoring
- Performance tracking
- Automatic histogram buckets for latency measurements

Usage:
    # Setup metrics collection
    metrics = setup_metrics(enable_prometheus=True, port=9090)

    # Record pipeline execution
    with metrics.pipeline_duration.time():
        result = executor.run(pipeline, inputs)

    # Record custom metrics
    metrics.record_custom("api_requests", 1, labels={"method": "POST"})
"""

import json
import logging
import os
import psutil
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import uuid4
from greenlang.determinism import DeterministicClock

# Try to import Prometheus client
try:
    from prometheus_client import (
        CollectorRegistry, Gauge, Counter, Histogram, Summary,
        start_http_server, generate_latest, CONTENT_TYPE_LATEST
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    # Create dummy classes for when Prometheus is not available
    class CollectorRegistry:
        pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

    class Counter:
        def __init__(self, *args, **kwargs):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

    class Summary:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class CustomMetric:
    """Custom metric definition"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None


@dataclass
class MetricValue:
    """Single metric value with metadata"""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: Optional[str] = None


class MetricsBuffer:
    """Thread-safe buffer for collecting metrics"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, metric: MetricValue):
        """Add metric to buffer"""
        with self.lock:
            self.buffer.append(metric)

    def get_recent(self, seconds: int = 60) -> List[MetricValue]:
        """Get metrics from last N seconds"""
        cutoff = time.time() - seconds
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff]

    def clear_old(self, max_age_seconds: int = 3600):
        """Clear metrics older than max_age_seconds"""
        cutoff = time.time() - max_age_seconds
        with self.lock:
            while self.buffer and self.buffer[0].timestamp < cutoff:
                self.buffer.popleft()


class PrometheusExporter:
    """Prometheus metrics exporter"""

    def __init__(self, port: int = 9090, registry: Optional[CollectorRegistry] = None):
        if not HAS_PROMETHEUS:
            raise ImportError("prometheus_client is required for PrometheusExporter")

        self.port = port
        self.registry = registry or CollectorRegistry()
        self.server_thread = None
        self.metrics = {}
        self._setup_default_metrics()

    def _setup_default_metrics(self):
        """Setup default GreenLang metrics"""
        # Pipeline metrics
        self.metrics['pipeline_executions_total'] = Counter(
            'greenlang_pipeline_executions_total',
            'Total number of pipeline executions',
            ['pipeline_name', 'status'],
            registry=self.registry
        )

        self.metrics['pipeline_duration_seconds'] = Histogram(
            'greenlang_pipeline_duration_seconds',
            'Time spent executing pipelines',
            ['pipeline_name'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf')],
            registry=self.registry
        )

        # Agent metrics
        self.metrics['agent_executions_total'] = Counter(
            'greenlang_agent_executions_total',
            'Total number of agent executions',
            ['agent_name', 'status'],
            registry=self.registry
        )

        self.metrics['agent_duration_seconds'] = Histogram(
            'greenlang_agent_duration_seconds',
            'Time spent executing agents',
            ['agent_name'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float('inf')],
            registry=self.registry
        )

        # System metrics
        self.metrics['memory_usage_bytes'] = Gauge(
            'greenlang_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        self.metrics['cpu_usage_percent'] = Gauge(
            'greenlang_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

        # Error metrics
        self.metrics['errors_total'] = Counter(
            'greenlang_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )

        # Pack metrics
        self.metrics['pack_loads_total'] = Counter(
            'greenlang_pack_loads_total',
            'Total number of pack loads',
            ['pack_name', 'status'],
            registry=self.registry
        )

        # Context metrics
        self.metrics['context_operations_total'] = Counter(
            'greenlang_context_operations_total',
            'Total number of context operations',
            ['operation_type'],
            registry=self.registry
        )

        # Custom metrics registry
        self.metrics['custom_metrics'] = {}

    def start_server(self):
        """Start Prometheus metrics server"""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("Prometheus server already running")
            return

        try:
            self.server_thread = threading.Thread(
                target=lambda: start_http_server(self.port, registry=self.registry),
                daemon=True
            )
            self.server_thread.start()
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def register_custom_metric(self, metric: CustomMetric) -> str:
        """Register a custom metric"""
        metric_name = f"greenlang_{metric.name}"

        if metric.type == MetricType.COUNTER:
            prom_metric = Counter(
                metric_name, metric.description, metric.labels, registry=self.registry
            )
        elif metric.type == MetricType.GAUGE:
            prom_metric = Gauge(
                metric_name, metric.description, metric.labels, registry=self.registry
            )
        elif metric.type == MetricType.HISTOGRAM:
            buckets = metric.buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')]
            prom_metric = Histogram(
                metric_name, metric.description, metric.labels, buckets=buckets, registry=self.registry
            )
        elif metric.type == MetricType.SUMMARY:
            prom_metric = Summary(
                metric_name, metric.description, metric.labels, registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric.type}")

        self.metrics['custom_metrics'][metric.name] = prom_metric
        logger.info(f"Registered custom metric: {metric_name}")
        return metric_name

    def record_metric(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a metric value"""
        labels = labels or {}

        if name in self.metrics:
            metric = self.metrics[name]
            if hasattr(metric, 'inc'):  # Counter
                metric.labels(**labels).inc(value)
            elif hasattr(metric, 'set'):  # Gauge
                metric.set(value)
            elif hasattr(metric, 'observe'):  # Histogram/Summary
                metric.labels(**labels).observe(value)
        elif name in self.metrics['custom_metrics']:
            metric = self.metrics['custom_metrics'][name]
            if hasattr(metric, 'inc'):  # Counter
                metric.labels(**labels).inc(value)
            elif hasattr(metric, 'set'):  # Gauge
                metric.labels(**labels).set(value)
            elif hasattr(metric, 'observe'):  # Histogram/Summary
                metric.labels(**labels).observe(value)

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return generate_latest(self.registry).decode('utf-8')


class MetricsCollector:
    """Main metrics collection system"""

    def __init__(self,
                 enable_prometheus: bool = False,
                 prometheus_port: int = 9090,
                 buffer_size: int = 10000,
                 enable_system_metrics: bool = True):

        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.buffer = MetricsBuffer(max_size=buffer_size)
        self.enable_system_metrics = enable_system_metrics

        # Prometheus exporter
        self.prometheus = None
        if enable_prometheus and HAS_PROMETHEUS:
            try:
                self.prometheus = PrometheusExporter(port=prometheus_port)
                self.prometheus.start_server()
            except Exception as e:
                logger.error(f"Failed to setup Prometheus exporter: {e}")

        # System metrics collection
        self.system_metrics_thread = None
        if enable_system_metrics:
            self._start_system_metrics_collection()

        # Performance tracking
        self.operation_times = defaultdict(list)

        # Custom metrics registry
        self.custom_metrics = {}

        logger.info("MetricsCollector initialized")

    def _start_system_metrics_collection(self):
        """Start background system metrics collection"""
        def collect_system_metrics():
            process = psutil.Process()

            while True:
                try:
                    # Memory metrics
                    memory_info = process.memory_info()
                    self.record_metric("memory_usage_bytes", memory_info.rss)

                    # CPU metrics
                    cpu_percent = process.cpu_percent(interval=None)
                    self.record_metric("cpu_usage_percent", cpu_percent)

                    # System-wide metrics
                    vm = psutil.virtual_memory()
                    self.record_metric("system_memory_used_percent", vm.percent)
                    self.record_metric("system_memory_available_bytes", vm.available)

                    # Disk I/O
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.record_metric("disk_read_bytes_total", disk_io.read_bytes)
                        self.record_metric("disk_write_bytes_total", disk_io.write_bytes)

                    time.sleep(10)  # Collect every 10 seconds

                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)  # Retry after 30 seconds on error

        self.system_metrics_thread = threading.Thread(
            target=collect_system_metrics,
            daemon=True
        )
        self.system_metrics_thread.start()

    def record_metric(self, name: str, value: Union[int, float],
                     labels: Dict[str, str] = None, unit: Optional[str] = None):
        """Record a metric value"""
        labels = labels or {}

        # Store in buffer
        metric_value = MetricValue(
            name=name,
            value=value,
            labels=labels,
            unit=unit
        )
        self.buffer.add(metric_value)

        # Send to Prometheus if enabled
        if self.prometheus:
            self.prometheus.record_metric(name, value, labels)

    def register_custom_metric(self, metric: CustomMetric) -> str:
        """Register a custom metric"""
        self.custom_metrics[metric.name] = metric

        if self.prometheus:
            return self.prometheus.register_custom_metric(metric)

        return metric.name

    def record_custom_metric(self, name: str, value: Union[int, float],
                           labels: Dict[str, str] = None):
        """Record a custom metric value"""
        if name not in self.custom_metrics:
            # Auto-register as gauge if not found
            self.register_custom_metric(CustomMetric(
                name=name,
                type=MetricType.GAUGE,
                description=f"Custom metric: {name}"
            ))

        self.record_metric(f"custom_{name}", value, labels)

    @contextmanager
    def time_operation(self, operation: str, labels: Dict[str, str] = None):
        """Context manager to time operations"""
        start_time = time.perf_counter()
        labels = labels or {}

        try:
            yield
            success = True
        except Exception as e:
            success = False
            self.record_error(operation, type(e).__name__)
            raise
        finally:
            duration = time.perf_counter() - start_time

            # Record timing
            self.record_metric(
                f"{operation}_duration_seconds",
                duration,
                labels={**labels, "status": "success" if success else "error"}
            )

            # Track in operation times for statistics
            self.operation_times[operation].append(duration)

            # Limit stored operation times
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-500:]

    def record_pipeline_execution(self, pipeline_name: str, duration: float, success: bool):
        """Record pipeline execution metrics"""
        status = "success" if success else "error"

        self.record_metric("pipeline_executions_total", 1, {
            "pipeline_name": pipeline_name,
            "status": status
        })

        self.record_metric("pipeline_duration_seconds", duration, {
            "pipeline_name": pipeline_name
        })

    def record_agent_execution(self, agent_name: str, duration: float, success: bool):
        """Record agent execution metrics"""
        status = "success" if success else "error"

        self.record_metric("agent_executions_total", 1, {
            "agent_name": agent_name,
            "status": status
        })

        self.record_metric("agent_duration_seconds", duration, {
            "agent_name": agent_name
        })

    def record_pack_load(self, pack_name: str, success: bool):
        """Record pack loading metrics"""
        status = "success" if success else "error"

        self.record_metric("pack_loads_total", 1, {
            "pack_name": pack_name,
            "status": status
        })

    def record_error(self, component: str, error_type: str):
        """Record error metrics"""
        self.record_metric("errors_total", 1, {
            "component": component,
            "error_type": error_type
        })

    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        times = self.operation_times.get(operation, [])
        if not times:
            return {}

        times_sorted = sorted(times)
        n = len(times_sorted)

        return {
            "count": n,
            "mean": sum(times_sorted) / n,
            "min": times_sorted[0],
            "max": times_sorted[-1],
            "p50": times_sorted[n // 2],
            "p95": times_sorted[int(n * 0.95)] if n > 20 else times_sorted[-1],
            "p99": times_sorted[int(n * 0.99)] if n > 100 else times_sorted[-1]
        }

    def get_recent_metrics(self, seconds: int = 60) -> List[MetricValue]:
        """Get recent metrics from buffer"""
        return self.buffer.get_recent(seconds)

    def export_metrics_json(self, output_file: Optional[Path] = None) -> str:
        """Export metrics to JSON format"""
        recent_metrics = self.get_recent_metrics(3600)  # Last hour

        # Group by metric name
        metrics_data = defaultdict(list)
        for metric in recent_metrics:
            metrics_data[metric.name].append({
                "value": metric.value,
                "labels": metric.labels,
                "timestamp": metric.timestamp,
                "unit": metric.unit
            })

        # Add operation statistics
        operation_stats = {}
        for operation in self.operation_times:
            operation_stats[operation] = self.get_operation_stats(operation)

        export_data = {
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "metrics": dict(metrics_data),
            "operation_stats": operation_stats,
            "system_info": {
                "pid": os.getpid(),
                "memory_rss": psutil.Process().memory_info().rss,
                "cpu_count": psutil.cpu_count(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            }
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to {output_file}")

        return json.dumps(export_data, indent=2)

    def cleanup(self):
        """Cleanup resources"""
        # Clear old metrics from buffer
        self.buffer.clear_old(max_age_seconds=3600)

        # Clear old operation times
        for operation in list(self.operation_times.keys()):
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-50:]

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get metrics for health checking"""
        recent = self.get_recent_metrics(60)
        error_count = len([m for m in recent if "error" in m.name])

        # Get current system metrics
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "metrics_collected_last_minute": len(recent),
            "errors_last_minute": error_count,
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "buffer_size": len(self.buffer.buffer),
            "custom_metrics_registered": len(self.custom_metrics),
            "prometheus_enabled": self.prometheus is not None
        }


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def setup_metrics(enable_prometheus: bool = False,
                 prometheus_port: int = 9090,
                 buffer_size: int = 10000,
                 enable_system_metrics: bool = True) -> MetricsCollector:
    """Setup global metrics collector"""
    global _global_metrics

    if _global_metrics is None:
        _global_metrics = MetricsCollector(
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port,
            buffer_size=buffer_size,
            enable_system_metrics=enable_system_metrics
        )

    return _global_metrics


def get_metrics() -> Optional[MetricsCollector]:
    """Get the global metrics collector"""
    return _global_metrics


# Convenience decorators for common operations
def track_pipeline_execution(func):
    """Decorator to track pipeline execution metrics"""
    def wrapper(*args, **kwargs):
        metrics = get_metrics()
        if not metrics:
            return func(*args, **kwargs)

        pipeline_name = kwargs.get('pipeline_name', 'unknown')

        with metrics.time_operation("pipeline_execution", {"pipeline": pipeline_name}):
            result = func(*args, **kwargs)

        success = getattr(result, 'success', True)
        duration = getattr(result, 'duration', 0)

        metrics.record_pipeline_execution(pipeline_name, duration, success)
        return result

    return wrapper


def track_agent_execution(func):
    """Decorator to track agent execution metrics"""
    def wrapper(*args, **kwargs):
        metrics = get_metrics()
        if not metrics:
            return func(*args, **kwargs)

        # Try to get agent name from self or args
        agent_name = "unknown"
        if args and hasattr(args[0], '__class__'):
            agent_name = args[0].__class__.__name__

        with metrics.time_operation("agent_execution", {"agent": agent_name}):
            result = func(*args, **kwargs)

        success = getattr(result, 'success', True)
        duration = getattr(result, 'duration', 0)

        metrics.record_agent_execution(agent_name, duration, success)
        return result

    return wrapper


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_metrics():
        """Test metrics collection"""
        print("Setting up metrics collection...")

        # Setup with Prometheus if available
        metrics = setup_metrics(
            enable_prometheus=HAS_PROMETHEUS,
            prometheus_port=9091,
            enable_system_metrics=True
        )

        # Register custom metrics
        custom_metric = CustomMetric(
            name="test_operations",
            type=MetricType.COUNTER,
            description="Test operations performed",
            labels=["operation_type"]
        )
        metrics.register_custom_metric(custom_metric)

        print("Recording test metrics...")

        # Simulate some operations
        for i in range(50):
            with metrics.time_operation("test_operation", {"iteration": str(i)}):
                await asyncio.sleep(0.01)  # Simulate work

            metrics.record_custom_metric("test_operations", 1, {"operation_type": "test"})

            if i % 10 == 0:
                print(f"  Completed {i+1} operations")

        # Record some pipeline executions
        metrics.record_pipeline_execution("test_pipeline", 0.5, True)
        metrics.record_pipeline_execution("test_pipeline", 0.7, True)
        metrics.record_pipeline_execution("test_pipeline", 1.2, False)

        # Get statistics
        stats = metrics.get_operation_stats("test_operation")
        print(f"Operation stats: {json.dumps(stats, indent=2)}")

        # Export metrics
        json_export = metrics.export_metrics_json()
        print(f"Exported {len(json_export)} bytes of metrics data")

        if HAS_PROMETHEUS:
            print(f"Prometheus metrics available at http://localhost:9091/metrics")
            prometheus_data = metrics.prometheus.get_metrics_text()
            print(f"Prometheus export: {len(prometheus_data)} bytes")

        print("Metrics test completed!")

    asyncio.run(test_metrics())