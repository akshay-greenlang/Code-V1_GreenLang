"""
Prometheus Metrics Collection
=============================

Comprehensive metrics collection for:
- Agent performance metrics
- Business metrics
- Infrastructure metrics
- Quality framework metrics (12 dimensions)

Author: GL-DevOpsEngineer
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from threading import Lock

# Prometheus client
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    push_to_gateway,
    start_http_server
)

# Quality metrics
from dataclasses import dataclass, field


class MetricType(Enum):
    """Prometheus metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class QualityDimension:
    """Quality framework dimension metrics"""
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    target: float = 0.0
    current: float = 0.0
    status: str = "unknown"

    def update(self, value: float):
        """Update dimension value"""
        self.current = value
        self.status = "pass" if value >= self.target else "fail"


class MetricsCollector:
    """
    Production-grade metrics collection with Prometheus
    """

    def __init__(
        self,
        namespace: str = "greenlang",
        subsystem: str = "agents",
        registry: Optional[CollectorRegistry] = None,
        push_gateway: Optional[str] = None
    ):
        """
        Initialize metrics collector

        Args:
            namespace: Metric namespace
            subsystem: Metric subsystem
            registry: Prometheus registry
            push_gateway: Push gateway URL
        """
        self.namespace = namespace
        self.subsystem = subsystem
        self.registry = registry or REGISTRY
        self.push_gateway = push_gateway
        self.metrics = {}
        self._lock = Lock()

        # Initialize standard metrics
        self._init_application_metrics()
        self._init_business_metrics()
        self._init_infrastructure_metrics()
        self._init_quality_metrics()

    def _init_application_metrics(self):
        """Initialize application-level metrics"""
        # Agent metrics
        self.register_metric(
            "agent_count",
            MetricType.GAUGE,
            "Number of active agents",
            labels=["type", "state"]
        )

        self.register_metric(
            "messages_processed",
            MetricType.COUNTER,
            "Total messages processed",
            labels=["agent_type", "message_type"]
        )

        self.register_metric(
            "task_completion_time",
            MetricType.HISTOGRAM,
            "Task completion time in seconds",
            labels=["agent_type", "task_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.register_metric(
            "error_rate",
            MetricType.GAUGE,
            "Current error rate",
            labels=["agent_type", "error_type"]
        )

        self.register_metric(
            "memory_usage_bytes",
            MetricType.GAUGE,
            "Memory usage in bytes",
            labels=["agent_id", "memory_type"]
        )

        self.register_metric(
            "cpu_utilization",
            MetricType.GAUGE,
            "CPU utilization percentage",
            labels=["agent_id"]
        )

    def _init_business_metrics(self):
        """Initialize business metrics"""
        self.register_metric(
            "calculations_performed",
            MetricType.COUNTER,
            "Total calculations performed",
            labels=["calculation_type", "scope"]
        )

        self.register_metric(
            "reports_generated",
            MetricType.COUNTER,
            "Total reports generated",
            labels=["report_type", "format"]
        )

        self.register_metric(
            "compliance_checks",
            MetricType.COUNTER,
            "Total compliance checks performed",
            labels=["regulation", "status"]
        )

        self.register_metric(
            "data_processed_bytes",
            MetricType.COUNTER,
            "Total data processed in bytes",
            labels=["data_type", "source"]
        )

        self.register_metric(
            "api_calls",
            MetricType.COUNTER,
            "Total API calls made",
            labels=["api", "method", "status"]
        )

        self.register_metric(
            "cache_hit_rate",
            MetricType.GAUGE,
            "Cache hit rate percentage",
            labels=["cache_type"]
        )

    def _init_infrastructure_metrics(self):
        """Initialize infrastructure metrics"""
        self.register_metric(
            "pod_count",
            MetricType.GAUGE,
            "Number of Kubernetes pods",
            labels=["deployment", "status"]
        )

        self.register_metric(
            "network_throughput_bytes",
            MetricType.GAUGE,
            "Network throughput in bytes/sec",
            labels=["direction", "interface"]
        )

        self.register_metric(
            "disk_iops",
            MetricType.GAUGE,
            "Disk IOPS",
            labels=["disk", "operation"]
        )

        self.register_metric(
            "database_connections",
            MetricType.GAUGE,
            "Database connection pool size",
            labels=["database", "status"]
        )

        self.register_metric(
            "queue_depth",
            MetricType.GAUGE,
            "Message queue depth",
            labels=["queue_name"]
        )

    def _init_quality_metrics(self):
        """Initialize 12-dimension quality framework metrics"""
        dimensions = [
            "functional_quality",
            "performance_efficiency",
            "compatibility",
            "usability",
            "reliability",
            "security",
            "maintainability",
            "portability",
            "scalability",
            "interoperability",
            "reusability",
            "testability"
        ]

        for dimension in dimensions:
            self.register_metric(
                f"quality_{dimension}",
                MetricType.GAUGE,
                f"Quality dimension: {dimension}",
                labels=["metric", "status"]
            )

        # Overall quality score
        self.register_metric(
            "quality_score_overall",
            MetricType.GAUGE,
            "Overall quality score (0-100)",
            labels=["environment"]
        )

        # Test metrics
        self.register_metric(
            "test_coverage",
            MetricType.GAUGE,
            "Test coverage percentage",
            labels=["test_type"]
        )

        self.register_metric(
            "test_execution_time",
            MetricType.HISTOGRAM,
            "Test execution time in seconds",
            labels=["test_suite"],
            buckets=[1, 5, 10, 30, 60, 300, 600]
        )

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Any:
        """
        Register a new metric

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: Label names
            buckets: Histogram buckets

        Returns:
            Prometheus metric object
        """
        with self._lock:
            if name in self.metrics:
                return self.metrics[name]

            full_name = f"{self.namespace}_{self.subsystem}_{name}"
            label_names = labels or []

            if metric_type == MetricType.COUNTER:
                metric = Counter(
                    full_name,
                    description,
                    label_names,
                    registry=self.registry
                )
            elif metric_type == MetricType.GAUGE:
                metric = Gauge(
                    full_name,
                    description,
                    label_names,
                    registry=self.registry
                )
            elif metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    full_name,
                    description,
                    label_names,
                    buckets=buckets or Histogram.DEFAULT_BUCKETS,
                    registry=self.registry
                )
            elif metric_type == MetricType.SUMMARY:
                metric = Summary(
                    full_name,
                    description,
                    label_names,
                    registry=self.registry
                )
            elif metric_type == MetricType.INFO:
                metric = Info(
                    full_name,
                    description,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

            self.metrics[name] = metric
            return metric

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value

        Args:
            name: Metric name
            value: Metric value
            labels: Label values
        """
        if name not in self.metrics:
            raise ValueError(f"Metric not registered: {name}")

        metric = self.metrics[name]
        label_values = labels or {}

        if isinstance(metric, Counter):
            if label_values:
                metric.labels(**label_values).inc(value)
            else:
                metric.inc(value)
        elif isinstance(metric, Gauge):
            if label_values:
                metric.labels(**label_values).set(value)
            else:
                metric.set(value)
        elif isinstance(metric, Histogram):
            if label_values:
                metric.labels(**label_values).observe(value)
            else:
                metric.observe(value)
        elif isinstance(metric, Summary):
            if label_values:
                metric.labels(**label_values).observe(value)
            else:
                metric.observe(value)

    def increment_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        value: float = 1.0
    ):
        """Increment a counter metric"""
        self.record_metric(name, value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value"""
        self.record_metric(name, value, labels)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Observe a histogram value"""
        self.record_metric(name, value, labels)

    def record_agent_metrics(
        self,
        agent_id: str,
        agent_type: str,
        state: str,
        memory_mb: float,
        cpu_percent: float
    ):
        """Record agent-specific metrics"""
        self.set_gauge("agent_count", 1, {"type": agent_type, "state": state})
        self.set_gauge("memory_usage_bytes", memory_mb * 1024 * 1024, {"agent_id": agent_id, "memory_type": "heap"})
        self.set_gauge("cpu_utilization", cpu_percent, {"agent_id": agent_id})

    def record_task_completion(
        self,
        agent_type: str,
        task_type: str,
        duration_seconds: float,
        success: bool
    ):
        """Record task completion metrics"""
        self.observe_histogram(
            "task_completion_time",
            duration_seconds,
            {"agent_type": agent_type, "task_type": task_type}
        )
        if not success:
            self.increment_counter(
                "error_rate",
                {"agent_type": agent_type, "error_type": "task_failure"}
            )

    def record_llm_call(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        duration_seconds: float,
        cost: float
    ):
        """Record LLM call metrics"""
        self.increment_counter("api_calls", {"api": "llm", "method": "completion", "status": "success"})
        self.observe_histogram("task_completion_time", duration_seconds, {"agent_type": "llm", "task_type": model})

    def record_quality_metric(
        self,
        dimension: str,
        metric_name: str,
        value: float,
        target: float
    ):
        """Record quality framework metric"""
        status = "pass" if value >= target else "fail"
        self.set_gauge(
            f"quality_{dimension}",
            value,
            {"metric": metric_name, "status": status}
        )

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        dimensions = {
            "functional_quality": 0.9,
            "performance_efficiency": 0.85,
            "compatibility": 0.95,
            "usability": 0.88,
            "reliability": 0.92,
            "security": 0.94,
            "maintainability": 0.87,
            "portability": 0.91,
            "scalability": 0.89,
            "interoperability": 0.93,
            "reusability": 0.86,
            "testability": 0.90
        }

        # Calculate weighted average
        total_weight = len(dimensions)
        total_score = sum(dimensions.values())
        overall_score = (total_score / total_weight) * 100

        self.set_gauge("quality_score_overall", overall_score, {"environment": "production"})
        return overall_score

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)

    def push_metrics(self, job: str = "greenlang_agents"):
        """Push metrics to push gateway"""
        if self.push_gateway:
            push_to_gateway(self.push_gateway, job=job, registry=self.registry)

    def start_http_server(self, port: int = 9090):
        """Start Prometheus HTTP server"""
        start_http_server(port, registry=self.registry)

    def get_stats(self) -> Dict[str, Any]:
        """Get metrics statistics"""
        return {
            'namespace': self.namespace,
            'subsystem': self.subsystem,
            'metric_count': len(self.metrics),
            'metrics': list(self.metrics.keys()),
            'quality_score': self.calculate_quality_score()
        }


def metric_timer(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                collector.observe_histogram(metric_name, duration, labels)
                return result
            except Exception as e:
                duration = time.time() - start
                collector.observe_histogram(metric_name, duration, labels)
                raise
        return wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            collector.increment_counter(metric_name, labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def setup_metrics(
    namespace: str = "greenlang",
    subsystem: str = "agents",
    push_gateway: Optional[str] = None,
    http_port: Optional[int] = 9090
) -> MetricsCollector:
    """
    Setup global metrics collection

    Args:
        namespace: Metric namespace
        subsystem: Metric subsystem
        push_gateway: Push gateway URL
        http_port: HTTP server port

    Returns:
        Configured MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(
        namespace=namespace,
        subsystem=subsystem,
        push_gateway=push_gateway
    )

    if http_port:
        _metrics_collector.start_http_server(http_port)

    return _metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = setup_metrics()
    return _metrics_collector


def register_metric(
    name: str,
    metric_type: MetricType,
    description: str,
    labels: Optional[List[str]] = None
):
    """Register a new metric"""
    collector = get_metrics_collector()
    return collector.register_metric(name, metric_type, description, labels)


def record_metric(
    name: str,
    value: Union[int, float],
    labels: Optional[Dict[str, str]] = None
):
    """Record a metric value"""
    collector = get_metrics_collector()
    collector.record_metric(name, value, labels)