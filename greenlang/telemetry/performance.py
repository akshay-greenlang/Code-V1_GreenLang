"""
Performance monitoring for GreenLang
"""

import time
import functools
import threading
import psutil
import gc
import tracemalloc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import cProfile
import pstats
import io
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data"""

    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class PerformanceProfile:
    """Performance profile results"""

    function: str
    total_time: float
    calls: int
    avg_time: float
    min_time: float
    max_time: float
    memory_used: int
    cpu_percent: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "function": self.function,
            "total_time": self.total_time,
            "calls": self.calls,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "memory_used": self.memory_used,
            "cpu_percent": self.cpu_percent,
        }


class PerformanceMonitor:
    """Monitor application performance"""

    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.profiler = None
        self.memory_tracking = False
        self.cpu_tracking = False
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

        # Performance thresholds
        self.thresholds = {
            "response_time_ms": 1000,
            "memory_mb": 500,
            "cpu_percent": 80,
        }

        logger.info("PerformanceMonitor initialized")

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Record a performance metric

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags
        """
        metric = PerformanceMetric(name=name, value=value, unit=unit, tags=tags or {})

        with self._lock:
            self.metrics[name].append(metric)

    def get_metrics(
        self, name: str, window_seconds: int = 300
    ) -> List[PerformanceMetric]:
        """
        Get metrics within time window

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            List of metrics
        """
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)

        with self._lock:
            return [m for m in self.metrics.get(name, []) if m.timestamp >= cutoff]

    def get_statistics(self, name: str, window_seconds: int = 300) -> Dict[str, float]:
        """
        Get metric statistics

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            Statistics dictionary
        """
        metrics = self.get_metrics(name, window_seconds)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def start_profiling(self):
        """Start CPU profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        logger.info("Started CPU profiling")

    def stop_profiling(self) -> str:
        """
        Stop CPU profiling and return results

        Returns:
            Profiling results as string
        """
        if not self.profiler:
            return "No profiling data"

        self.profiler.disable()

        # Get statistics
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        result = stream.getvalue()
        self.profiler = None

        logger.info("Stopped CPU profiling")
        return result

    def start_memory_tracking(self):
        """Start memory tracking"""
        if not self.memory_tracking:
            tracemalloc.start()
            self.memory_tracking = True
            logger.info("Started memory tracking")

    def stop_memory_tracking(self) -> Dict[str, Any]:
        """
        Stop memory tracking and return results

        Returns:
            Memory usage statistics
        """
        if not self.memory_tracking:
            return {}

        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        self.memory_tracking = False

        # Get top memory consumers
        top_stats = snapshot.statistics("lineno")[:10]

        results = {
            "top_consumers": [
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats
            ],
            "total_mb": sum(stat.size for stat in top_stats) / 1024 / 1024,
        }

        logger.info("Stopped memory tracking")
        return results

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage"""
        process = psutil.Process()

        return {
            "percent": process.cpu_percent(interval=1),
            "system_percent": psutil.cpu_percent(interval=1),
            "threads": process.num_threads(),
            "context_switches": (
                process.num_ctx_switches().voluntary
                if hasattr(process.num_ctx_switches(), "voluntary")
                else 0
            ),
        }

    def start_monitoring(self, interval_seconds: int = 60):
        """
        Start background monitoring

        Args:
            interval_seconds: Monitoring interval
        """
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_seconds,), daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Started performance monitoring (interval={interval_seconds}s)")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")

    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Record system metrics
                memory = self.get_memory_usage()
                self.record_metric("memory_rss", memory["rss_mb"], "MB")
                self.record_metric("memory_percent", memory["percent"], "%")

                cpu = self.get_cpu_usage()
                self.record_metric("cpu_percent", cpu["percent"], "%")
                self.record_metric("cpu_threads", cpu["threads"], "count")

                # Check thresholds
                self._check_thresholds(memory, cpu)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            self._stop_monitoring.wait(interval)

    def _check_thresholds(self, memory: Dict[str, float], cpu: Dict[str, float]):
        """Check performance thresholds"""
        if memory["rss_mb"] > self.thresholds["memory_mb"]:
            logger.warning(f"Memory threshold exceeded: {memory['rss_mb']:.1f}MB")

        if cpu["percent"] > self.thresholds["cpu_percent"]:
            logger.warning(f"CPU threshold exceeded: {cpu['percent']:.1f}%")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function performance

    Args:
        func: Function to profile
    """
    monitor = get_performance_monitor()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record start state
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Execute function
        try:
            result = func(*args, **kwargs)
            status = "success"
        except Exception as e:
            status = "error"
            raise
        finally:
            # Record metrics
            duration = (time.time() - start_time) * 1000  # Convert to ms
            memory_used = psutil.Process().memory_info().rss - start_memory

            monitor.record_metric(
                f"function.{func.__name__}.duration", duration, "ms", {"status": status}
            )

            monitor.record_metric(
                f"function.{func.__name__}.memory",
                memory_used / 1024 / 1024,
                "MB",
                {"status": status},
            )

        return result

    return wrapper


@contextmanager
def measure_latency(operation: str):
    """
    Context manager to measure operation latency

    Args:
        operation: Operation name
    """
    monitor = get_performance_monitor()
    start_time = time.time()

    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000
        monitor.record_metric(f"latency.{operation}", duration, "ms")


def track_memory(operation: str):
    """
    Decorator to track memory usage

    Args:
        operation: Operation name
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()

            # Get memory before
            gc.collect()
            memory_before = psutil.Process().memory_info().rss

            # Execute function
            result = func(*args, **kwargs)

            # Get memory after
            gc.collect()
            memory_after = psutil.Process().memory_info().rss
            memory_used = (memory_after - memory_before) / 1024 / 1024

            monitor.record_metric(f"memory.{operation}", memory_used, "MB")

            return result

        return wrapper

    return decorator


class PerformanceAnalyzer:
    """Analyze performance data"""

    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize analyzer

        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze performance bottlenecks

        Returns:
            Analysis results
        """
        bottlenecks = {"slow_operations": [], "memory_leaks": [], "cpu_intensive": []}

        # Find slow operations
        for metric_name in self.monitor.metrics:
            if "duration" in metric_name or "latency" in metric_name:
                stats = self.monitor.get_statistics(metric_name)
                if stats.get("p95", 0) > 1000:  # > 1 second
                    bottlenecks["slow_operations"].append(
                        {
                            "operation": metric_name,
                            "p95_ms": stats["p95"],
                            "avg_ms": stats["avg"],
                        }
                    )

        # Check for memory leaks
        memory_metrics = self.monitor.get_metrics("memory_rss", 3600)
        if len(memory_metrics) > 10:
            # Calculate memory growth rate
            first_10 = memory_metrics[:10]
            last_10 = memory_metrics[-10:]

            avg_first = sum(m.value for m in first_10) / len(first_10)
            avg_last = sum(m.value for m in last_10) / len(last_10)

            growth_rate = (avg_last - avg_first) / avg_first * 100

            if growth_rate > 20:  # 20% growth
                bottlenecks["memory_leaks"].append(
                    {
                        "growth_rate_percent": growth_rate,
                        "initial_mb": avg_first,
                        "current_mb": avg_last,
                    }
                )

        # Find CPU intensive operations
        cpu_metrics = self.monitor.get_metrics("cpu_percent", 300)
        high_cpu = [m for m in cpu_metrics if m.value > 80]

        if len(high_cpu) > len(cpu_metrics) * 0.1:  # > 10% of time
            bottlenecks["cpu_intensive"].append(
                {
                    "high_cpu_periods": len(high_cpu),
                    "avg_cpu_percent": sum(m.value for m in high_cpu) / len(high_cpu),
                }
            )

        return bottlenecks

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate performance report

        Returns:
            Performance report
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "memory": self.monitor.get_memory_usage(),
                "cpu": self.monitor.get_cpu_usage(),
            },
            "metrics_summary": {},
            "bottlenecks": self.analyze_bottlenecks(),
        }

        # Add metrics summary
        for metric_name in self.monitor.metrics:
            stats = self.monitor.get_statistics(metric_name)
            if stats:
                report["metrics_summary"][metric_name] = stats

        return report


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    monitor = get_performance_monitor()
    analyzer = PerformanceAnalyzer(monitor)
    return analyzer.generate_report()


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    global _performance_monitor
    if not _performance_monitor:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
