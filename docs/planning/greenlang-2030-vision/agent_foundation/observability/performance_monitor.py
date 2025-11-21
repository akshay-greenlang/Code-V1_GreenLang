# -*- coding: utf-8 -*-
"""
Performance Monitoring and SLA Tracking
========================================

Real-time performance monitoring with:
- SLA compliance tracking
- Latency monitoring
- Error rate tracking
- Throughput analysis
- Resource utilization monitoring

Author: GL-DevOpsEngineer
"""

import time
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from threading import Lock, Thread
from enum import Enum
from greenlang.determinism import DeterministicClock


class SLATarget:
    """SLA target definitions"""
    AVAILABILITY = 99.99  # Four nines availability
    LATENCY_P50 = 100     # 100ms median
    LATENCY_P95 = 500     # 500ms 95th percentile
    LATENCY_P99 = 2000    # 2000ms 99th percentile
    ERROR_RATE = 0.1      # 0.1% error rate
    THROUGHPUT_MIN = 1000 # 1000 requests/second minimum


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    availability: float
    cpu_usage: float
    memory_usage: float
    active_connections: int


@dataclass
class SLAStatus:
    """SLA compliance status"""
    metric: str
    target: float
    current: float
    compliant: bool
    breach_duration: Optional[timedelta] = None
    breach_count: int = 0


class PerformanceWindow:
    """Sliding window for performance metrics"""

    def __init__(self, window_size_seconds: int = 300):
        """
        Initialize performance window

        Args:
            window_size_seconds: Window size in seconds (default 5 minutes)
        """
        self.window_size = window_size_seconds
        self.data = deque()
        self._lock = Lock()

    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add value to window"""
        with self._lock:
            ts = timestamp or DeterministicClock.utcnow()
            self.data.append((ts, value))
            self._cleanup()

    def _cleanup(self):
        """Remove old entries outside window"""
        cutoff = DeterministicClock.utcnow() - timedelta(seconds=self.window_size)
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()

    def get_values(self) -> List[float]:
        """Get all values in window"""
        with self._lock:
            self._cleanup()
            return [value for _, value in self.data]

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Calculate percentile value"""
        values = self.get_values()
        if not values:
            return None
        return statistics.quantiles(values, n=100)[int(percentile)]

    def get_average(self) -> Optional[float]:
        """Calculate average value"""
        values = self.get_values()
        return statistics.mean(values) if values else None

    def get_count(self) -> int:
        """Get count of values in window"""
        return len(self.get_values())


class LatencyTracker:
    """Track and analyze latency metrics"""

    def __init__(self, window_size_seconds: int = 300):
        """
        Initialize latency tracker

        Args:
            window_size_seconds: Analysis window size
        """
        self.window = PerformanceWindow(window_size_seconds)
        self.histograms = defaultdict(PerformanceWindow)
        self._lock = Lock()

        # Latency buckets for histogram (in ms)
        self.buckets = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def record(self, latency_ms: float, operation: Optional[str] = None):
        """
        Record latency measurement

        Args:
            latency_ms: Latency in milliseconds
            operation: Operation name for categorization
        """
        self.window.add(latency_ms)

        if operation:
            self.histograms[operation].add(latency_ms)

    def get_percentiles(self) -> Tuple[float, float, float]:
        """Get P50, P95, P99 latencies"""
        values = self.window.get_values()
        if not values:
            return 0.0, 0.0, 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        p50 = sorted_values[int(n * 0.50)]
        p95 = sorted_values[int(n * 0.95)]
        p99 = sorted_values[int(n * 0.99)]

        return p50, p95, p99

    def get_histogram(self) -> Dict[str, int]:
        """Get latency histogram"""
        values = self.window.get_values()
        histogram = {f"<{bucket}ms": 0 for bucket in self.buckets}
        histogram[f">={self.buckets[-1]}ms"] = 0

        for value in values:
            added = False
            for bucket in self.buckets:
                if value < bucket:
                    histogram[f"<{bucket}ms"] += 1
                    added = True
                    break
            if not added:
                histogram[f">={self.buckets[-1]}ms"] += 1

        return histogram

    def get_slow_queries(self, threshold_ms: float = 1000) -> List[Tuple[datetime, float]]:
        """Get queries slower than threshold"""
        with self._lock:
            return [(ts, val) for ts, val in self.window.data if val > threshold_ms]


class ErrorRateMonitor:
    """Monitor and track error rates"""

    def __init__(self, window_size_seconds: int = 300):
        """
        Initialize error rate monitor

        Args:
            window_size_seconds: Analysis window size
        """
        self.window_size = window_size_seconds
        self.success_count = PerformanceWindow(window_size_seconds)
        self.error_count = PerformanceWindow(window_size_seconds)
        self.error_types = defaultdict(int)
        self._lock = Lock()

    def record_success(self):
        """Record successful operation"""
        self.success_count.add(1)

    def record_error(self, error_type: Optional[str] = None):
        """Record failed operation"""
        self.error_count.add(1)
        if error_type:
            with self._lock:
                self.error_types[error_type] += 1

    def get_error_rate(self) -> float:
        """Calculate current error rate"""
        successes = self.success_count.get_count()
        errors = self.error_count.get_count()
        total = successes + errors

        if total == 0:
            return 0.0

        return (errors / total) * 100

    def get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of error types"""
        with self._lock:
            return dict(self.error_types)

    def reset_error_types(self):
        """Reset error type counters"""
        with self._lock:
            self.error_types.clear()


class ThroughputMonitor:
    """Monitor request throughput"""

    def __init__(self, window_size_seconds: int = 60):
        """
        Initialize throughput monitor

        Args:
            window_size_seconds: Analysis window size
        """
        self.window = PerformanceWindow(window_size_seconds)
        self.window_size = window_size_seconds

    def record_request(self):
        """Record a processed request"""
        self.window.add(1)

    def get_throughput(self) -> float:
        """Get current throughput (requests per second)"""
        count = self.window.get_count()
        return count / self.window_size if self.window_size > 0 else 0


class SLATracker:
    """Track SLA compliance"""

    def __init__(
        self,
        availability_target: float = SLATarget.AVAILABILITY,
        latency_targets: Optional[Dict[str, float]] = None,
        error_rate_target: float = SLATarget.ERROR_RATE
    ):
        """
        Initialize SLA tracker

        Args:
            availability_target: Target availability percentage
            latency_targets: Latency targets by percentile
            error_rate_target: Target error rate percentage
        """
        self.availability_target = availability_target
        self.latency_targets = latency_targets or {
            'p50': SLATarget.LATENCY_P50,
            'p95': SLATarget.LATENCY_P95,
            'p99': SLATarget.LATENCY_P99
        }
        self.error_rate_target = error_rate_target

        # Tracking
        self.uptime_start = DeterministicClock.utcnow()
        self.downtime_total = timedelta(0)
        self.current_down_start = None
        self.breach_history = []
        self._lock = Lock()

    def record_downtime_start(self):
        """Record start of downtime"""
        with self._lock:
            if not self.current_down_start:
                self.current_down_start = DeterministicClock.utcnow()

    def record_downtime_end(self):
        """Record end of downtime"""
        with self._lock:
            if self.current_down_start:
                downtime = DeterministicClock.utcnow() - self.current_down_start
                self.downtime_total += downtime
                self.breach_history.append({
                    'type': 'availability',
                    'start': self.current_down_start,
                    'duration': downtime
                })
                self.current_down_start = None

    def get_availability(self) -> float:
        """Calculate current availability percentage"""
        total_time = DeterministicClock.utcnow() - self.uptime_start
        uptime = total_time - self.downtime_total

        if total_time.total_seconds() == 0:
            return 100.0

        return (uptime.total_seconds() / total_time.total_seconds()) * 100

    def check_compliance(
        self,
        current_latencies: Tuple[float, float, float],
        current_error_rate: float
    ) -> List[SLAStatus]:
        """
        Check SLA compliance

        Args:
            current_latencies: Current P50, P95, P99 latencies
            current_error_rate: Current error rate

        Returns:
            List of SLA status objects
        """
        statuses = []

        # Check availability
        availability = self.get_availability()
        statuses.append(SLAStatus(
            metric='availability',
            target=self.availability_target,
            current=availability,
            compliant=availability >= self.availability_target
        ))

        # Check latencies
        p50, p95, p99 = current_latencies
        for percentile, value in [('p50', p50), ('p95', p95), ('p99', p99)]:
            target = self.latency_targets.get(percentile, float('inf'))
            statuses.append(SLAStatus(
                metric=f'latency_{percentile}',
                target=target,
                current=value,
                compliant=value <= target
            ))

        # Check error rate
        statuses.append(SLAStatus(
            metric='error_rate',
            target=self.error_rate_target,
            current=current_error_rate,
            compliant=current_error_rate <= self.error_rate_target
        ))

        return statuses


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """

    def __init__(
        self,
        name: str = "greenlang_agents",
        window_size: int = 300,
        sla_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize performance monitor

        Args:
            name: Monitor name
            window_size: Monitoring window in seconds
            sla_config: SLA configuration
        """
        self.name = name
        self.window_size = window_size

        # Components
        self.latency_tracker = LatencyTracker(window_size)
        self.error_monitor = ErrorRateMonitor(window_size)
        self.throughput_monitor = ThroughputMonitor(window_size)
        self.sla_tracker = SLATracker(**(sla_config or {}))

        # Resource tracking
        self.resource_metrics = {
            'cpu_usage': PerformanceWindow(window_size),
            'memory_usage': PerformanceWindow(window_size),
            'active_connections': PerformanceWindow(window_size)
        }

        # History
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1 min intervals
        self._lock = Lock()

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring"""
        self.monitoring = True
        self.monitor_thread = Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring:
            snapshot = self.get_current_metrics()
            self.metrics_history.append(snapshot)
            time.sleep(interval)

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        operation: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """
        Record request metrics

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            operation: Operation name
            error_type: Error type if failed
        """
        # Record latency
        self.latency_tracker.record(latency_ms, operation)

        # Record success/error
        if success:
            self.error_monitor.record_success()
        else:
            self.error_monitor.record_error(error_type)

        # Record throughput
        self.throughput_monitor.record_request()

    def record_resource_usage(
        self,
        cpu_percent: float,
        memory_mb: float,
        connections: int
    ):
        """Record resource usage metrics"""
        self.resource_metrics['cpu_usage'].add(cpu_percent)
        self.resource_metrics['memory_usage'].add(memory_mb)
        self.resource_metrics['active_connections'].add(connections)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot"""
        p50, p95, p99 = self.latency_tracker.get_percentiles()

        return PerformanceMetrics(
            timestamp=DeterministicClock.utcnow(),
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput=self.throughput_monitor.get_throughput(),
            error_rate=self.error_monitor.get_error_rate(),
            availability=self.sla_tracker.get_availability(),
            cpu_usage=self.resource_metrics['cpu_usage'].get_average() or 0,
            memory_usage=self.resource_metrics['memory_usage'].get_average() or 0,
            active_connections=int(self.resource_metrics['active_connections'].get_average() or 0)
        )

    def check_sla_compliance(self) -> List[SLAStatus]:
        """Check current SLA compliance"""
        latencies = self.latency_tracker.get_percentiles()
        error_rate = self.error_monitor.get_error_rate()
        return self.sla_tracker.check_compliance(latencies, error_rate)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current = self.get_current_metrics()
        sla_status = self.check_sla_compliance()

        return {
            'name': self.name,
            'timestamp': current.timestamp.isoformat(),
            'metrics': {
                'latency': {
                    'p50': current.latency_p50,
                    'p95': current.latency_p95,
                    'p99': current.latency_p99,
                    'histogram': self.latency_tracker.get_histogram()
                },
                'throughput': current.throughput,
                'error_rate': current.error_rate,
                'availability': current.availability,
                'resources': {
                    'cpu_usage': current.cpu_usage,
                    'memory_usage': current.memory_usage,
                    'active_connections': current.active_connections
                }
            },
            'sla_compliance': [
                {
                    'metric': s.metric,
                    'target': s.target,
                    'current': s.current,
                    'compliant': s.compliant
                }
                for s in sla_status
            ],
            'error_breakdown': self.error_monitor.get_error_breakdown(),
            'slow_queries': len(self.latency_tracker.get_slow_queries())
        }

    def get_historical_metrics(
        self,
        duration_hours: int = 24
    ) -> List[PerformanceMetrics]:
        """Get historical metrics"""
        cutoff = DeterministicClock.utcnow() - timedelta(hours=duration_hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]