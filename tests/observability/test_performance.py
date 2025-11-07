"""
Tests for performance monitoring
"""

import pytest
import time
from greenlang.observability import (
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceAnalyzer,
    get_performance_monitor,
    profile_function,
    measure_latency,
    track_memory,
)


class TestPerformanceMetric:
    """Test PerformanceMetric functionality"""

    def test_performance_metric_creation(self):
        """Test creating performance metric"""
        metric = PerformanceMetric(
            name="response_time",
            value=0.123,
            unit="s",
            tags={"endpoint": "/api/test"},
        )
        assert metric.name == "response_time"
        assert metric.value == 0.123
        assert metric.unit == "s"
        assert metric.tags["endpoint"] == "/api/test"

    def test_performance_metric_to_dict(self):
        """Test converting metric to dictionary"""
        metric = PerformanceMetric(name="cpu_usage", value=45.2, unit="%")
        data = metric.to_dict()
        assert data["name"] == "cpu_usage"
        assert data["value"] == 45.2
        assert data["unit"] == "%"


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert len(monitor.metrics) == 0

    def test_record_metric(self):
        """Test recording a metric"""
        monitor = PerformanceMonitor()
        monitor.record_metric("test_metric", 42.0, "ms")
        assert "test_metric" in monitor.metrics

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics"""
        monitor = PerformanceMonitor()
        monitor.record_metric("metric1", 10.0, "ms")
        monitor.record_metric("metric1", 20.0, "ms")
        monitor.record_metric("metric1", 30.0, "ms")
        metrics = monitor.get_metrics("metric1")
        assert len(metrics) == 3

    def test_get_metrics_with_time_window(self):
        """Test getting metrics within time window"""
        monitor = PerformanceMonitor()
        monitor.record_metric("test", 100.0, "ms")
        time.sleep(0.1)
        metrics = monitor.get_metrics("test", window_seconds=1)
        assert len(metrics) == 1

    def test_get_statistics(self):
        """Test getting metric statistics"""
        monitor = PerformanceMonitor()
        monitor.record_metric("latency", 10.0, "ms")
        monitor.record_metric("latency", 20.0, "ms")
        monitor.record_metric("latency", 30.0, "ms")

        stats = monitor.get_statistics("latency")
        assert stats["count"] == 3
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["avg"] == 20.0

    def test_get_statistics_empty(self):
        """Test getting statistics for nonexistent metric"""
        monitor = PerformanceMonitor()
        stats = monitor.get_statistics("nonexistent")
        assert stats == {}

    def test_get_memory_usage(self):
        """Test getting memory usage"""
        monitor = PerformanceMonitor()
        memory = monitor.get_memory_usage()
        assert "rss_mb" in memory
        assert "vms_mb" in memory
        assert "percent" in memory

    def test_get_cpu_usage(self):
        """Test getting CPU usage"""
        monitor = PerformanceMonitor()
        cpu = monitor.get_cpu_usage()
        assert "percent" in cpu
        assert "threads" in cpu

    def test_start_profiling(self):
        """Test starting CPU profiling"""
        monitor = PerformanceMonitor()
        monitor.start_profiling()
        assert monitor.profiler is not None

    def test_stop_profiling(self):
        """Test stopping CPU profiling"""
        monitor = PerformanceMonitor()
        monitor.start_profiling()
        time.sleep(0.1)
        result = monitor.stop_profiling()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_memory_tracking(self):
        """Test memory tracking"""
        monitor = PerformanceMonitor()
        monitor.start_memory_tracking()
        # Allocate some memory
        data = [i for i in range(10000)]
        results = monitor.stop_memory_tracking()
        assert "top_consumers" in results
        assert "total_mb" in results


class TestPerformanceDecorators:
    """Test performance decorator functions"""

    def test_profile_function_decorator(self):
        """Test profile_function decorator"""

        @profile_function
        def test_function():
            time.sleep(0.01)
            return "done"

        result = test_function()
        assert result == "done"

    def test_profile_function_with_exception(self):
        """Test profile_function decorator with exception"""

        @profile_function
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_measure_latency_context(self):
        """Test measure_latency context manager"""
        with measure_latency("test_operation"):
            time.sleep(0.01)
        # Should not raise exception

    def test_track_memory_decorator(self):
        """Test track_memory decorator"""

        @track_memory("memory_test")
        def memory_function():
            data = [i for i in range(1000)]
            return len(data)

        result = memory_function()
        assert result == 1000


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer functionality"""

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        monitor = PerformanceMonitor()
        analyzer = PerformanceAnalyzer(monitor)
        assert analyzer.monitor is monitor

    def test_analyze_bottlenecks(self):
        """Test analyzing performance bottlenecks"""
        monitor = PerformanceMonitor()

        # Add slow operations
        for _ in range(10):
            monitor.record_metric("latency.slow_api", 2000.0, "ms")

        analyzer = PerformanceAnalyzer(monitor)
        bottlenecks = analyzer.analyze_bottlenecks()

        assert "slow_operations" in bottlenecks
        assert "memory_leaks" in bottlenecks
        assert "cpu_intensive" in bottlenecks

    def test_generate_report(self):
        """Test generating performance report"""
        monitor = PerformanceMonitor()
        monitor.record_metric("test_metric", 100.0, "ms")

        analyzer = PerformanceAnalyzer(monitor)
        report = analyzer.generate_report()

        assert "timestamp" in report
        assert "system" in report
        assert "metrics_summary" in report
        assert "bottlenecks" in report


class TestGlobalPerformanceInstances:
    """Test global performance monitor instances"""

    def test_get_performance_monitor_singleton(self):
        """Test getting global performance monitor"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        assert monitor1 is monitor2

    def test_performance_monitor_functional(self):
        """Test global performance monitor is functional"""
        monitor = get_performance_monitor()
        monitor.record_metric("test", 42.0, "ms")
        metrics = monitor.get_metrics("test")
        assert len(metrics) > 0


class TestPerformanceIntegration:
    """Integration tests for performance monitoring"""

    def test_full_performance_monitoring_workflow(self):
        """Test complete performance monitoring workflow"""
        monitor = get_performance_monitor()

        # Record various metrics
        monitor.record_metric("api_latency", 150.0, "ms", {"endpoint": "/api/test"})
        monitor.record_metric("api_latency", 200.0, "ms", {"endpoint": "/api/test"})
        monitor.record_metric("memory_usage", 512.0, "MB")

        # Get statistics
        stats = monitor.get_statistics("api_latency")
        assert stats["count"] == 2
        assert stats["avg"] == 175.0

        # Generate report
        analyzer = PerformanceAnalyzer(monitor)
        report = analyzer.generate_report()
        assert report is not None

    def test_performance_profiling_workflow(self):
        """Test performance profiling workflow"""
        monitor = get_performance_monitor()

        # Profile a function
        @profile_function
        def complex_operation():
            result = 0
            for i in range(1000):
                result += i
            return result

        result = complex_operation()
        assert result == sum(range(1000))

        # Check metrics were recorded
        metrics = list(monitor.metrics.keys())
        function_metrics = [m for m in metrics if "complex_operation" in m]
        assert len(function_metrics) > 0
