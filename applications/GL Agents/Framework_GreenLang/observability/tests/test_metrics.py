"""
GreenLang Observability - Metrics Module Tests
===============================================

Comprehensive unit tests for Prometheus metrics functionality.
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from observability.metrics import (
    MetricsRegistry,
    MetricsConfig,
    Counter,
    Gauge,
    Histogram,
    Summary,
    LabelSet,
    MetricType,
    HistogramTimer,
    get_default_registry,
    calculation_counter,
    calculation_latency,
    queue_depth,
    active_tasks,
    response_size,
)


class TestLabelSet:
    """Tests for LabelSet class."""

    def test_label_set_hash(self) -> None:
        """Test LabelSet hashing for dict keys."""
        labels1 = LabelSet({"a": "1", "b": "2"})
        labels2 = LabelSet({"a": "1", "b": "2"})
        labels3 = LabelSet({"a": "1", "b": "3"})

        assert hash(labels1) == hash(labels2)
        assert hash(labels1) != hash(labels3)

    def test_label_set_equality(self) -> None:
        """Test LabelSet equality."""
        labels1 = LabelSet({"a": "1"})
        labels2 = LabelSet({"a": "1"})
        labels3 = LabelSet({"a": "2"})

        assert labels1 == labels2
        assert labels1 != labels3

    def test_to_prometheus_format(self) -> None:
        """Test Prometheus label format."""
        labels = LabelSet({"agent_id": "GL-006", "type": "thermal"})
        formatted = labels.to_prometheus_format()

        assert 'agent_id="GL-006"' in formatted
        assert 'type="thermal"' in formatted
        assert formatted.startswith("{")
        assert formatted.endswith("}")

    def test_empty_labels(self) -> None:
        """Test empty label set."""
        labels = LabelSet({})
        assert labels.to_prometheus_format() == ""


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_creation(self) -> None:
        """Test counter creation."""
        counter = Counter(
            name="requests_total",
            description="Total requests",
            labels=["method", "status"],
        )

        assert counter.name == "requests_total"
        assert counter.metric_type == MetricType.COUNTER

    def test_counter_increment(self) -> None:
        """Test counter increment."""
        counter = Counter("test_counter", "Test counter")
        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5)
        assert counter.get() == 6.0

    def test_counter_with_labels(self) -> None:
        """Test counter with labels."""
        counter = Counter(
            "requests",
            "Requests",
            labels=["method"],
        )

        counter.inc(labels={"method": "GET"})
        counter.inc(2, labels={"method": "POST"})

        assert counter.get(labels={"method": "GET"}) == 1.0
        assert counter.get(labels={"method": "POST"}) == 2.0

    def test_counter_negative_increment(self) -> None:
        """Test counter rejects negative increment."""
        counter = Counter("test", "Test")

        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_counter_missing_labels(self) -> None:
        """Test counter with missing required labels."""
        counter = Counter("test", "Test", labels=["required"])

        with pytest.raises(ValueError):
            counter.inc()  # Missing required label

    def test_counter_prometheus_format(self) -> None:
        """Test Prometheus exposition format."""
        counter = Counter(
            "test_counter",
            "Test counter",
            namespace="greenlang",
        )
        counter.inc(5)

        output = counter.to_prometheus_format()
        assert "# HELP greenlang_test_counter Test counter" in output
        assert "# TYPE greenlang_test_counter counter" in output
        assert "greenlang_test_counter 5" in output

    def test_counter_clear(self) -> None:
        """Test clearing counter values."""
        counter = Counter("test", "Test")
        counter.inc(10)
        assert counter.get() == 10.0

        counter.clear()
        assert counter.get() == 0.0


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_creation(self) -> None:
        """Test gauge creation."""
        gauge = Gauge(
            name="temperature",
            description="Current temperature",
        )

        assert gauge.name == "temperature"
        assert gauge.metric_type == MetricType.GAUGE

    def test_gauge_set(self) -> None:
        """Test gauge set."""
        gauge = Gauge("temp", "Temperature")
        gauge.set(42.5)
        assert gauge.get() == 42.5

    def test_gauge_inc_dec(self) -> None:
        """Test gauge increment and decrement."""
        gauge = Gauge("count", "Count")
        gauge.set(10)

        gauge.inc()
        assert gauge.get() == 11.0

        gauge.inc(5)
        assert gauge.get() == 16.0

        gauge.dec()
        assert gauge.get() == 15.0

        gauge.dec(3)
        assert gauge.get() == 12.0

    def test_gauge_with_labels(self) -> None:
        """Test gauge with labels."""
        gauge = Gauge("queue", "Queue depth", labels=["name"])

        gauge.set(10, labels={"name": "input"})
        gauge.set(5, labels={"name": "output"})

        assert gauge.get(labels={"name": "input"}) == 10.0
        assert gauge.get(labels={"name": "output"}) == 5.0

    def test_gauge_set_to_current_time(self) -> None:
        """Test setting gauge to current time."""
        gauge = Gauge("last_update", "Last update time")
        before = time.time()
        gauge.set_to_current_time()
        after = time.time()

        value = gauge.get()
        assert before <= value <= after

    def test_gauge_prometheus_format(self) -> None:
        """Test Prometheus exposition format."""
        gauge = Gauge("test_gauge", "Test gauge", namespace="app")
        gauge.set(100)

        output = gauge.to_prometheus_format()
        assert "# HELP app_test_gauge Test gauge" in output
        assert "# TYPE app_test_gauge gauge" in output
        assert "app_test_gauge 100" in output


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_creation(self) -> None:
        """Test histogram creation."""
        histogram = Histogram(
            name="latency",
            description="Request latency",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )

        assert histogram.name == "latency"
        assert histogram.metric_type == MetricType.HISTOGRAM
        assert 0.1 in histogram.buckets
        assert float("inf") in histogram.buckets  # +Inf always added

    def test_histogram_observe(self) -> None:
        """Test histogram observation."""
        histogram = Histogram(
            "latency",
            "Latency",
            buckets=(0.1, 0.5, 1.0),
        )

        histogram.observe(0.05)  # <= 0.1
        histogram.observe(0.3)   # <= 0.5
        histogram.observe(0.8)   # <= 1.0
        histogram.observe(2.0)   # <= +Inf

        assert histogram.get_sample_count() == 4
        assert histogram.get_sample_sum() == pytest.approx(3.15)

    def test_histogram_with_labels(self) -> None:
        """Test histogram with labels."""
        histogram = Histogram(
            "duration",
            "Duration",
            labels=["method"],
            buckets=(0.1, 1.0),
        )

        histogram.observe(0.05, labels={"method": "GET"})
        histogram.observe(0.5, labels={"method": "POST"})

        assert histogram.get_sample_count(labels={"method": "GET"}) == 1
        assert histogram.get_sample_count(labels={"method": "POST"}) == 1

    def test_histogram_timer(self) -> None:
        """Test histogram timer context manager."""
        histogram = Histogram("duration", "Duration", buckets=(0.01, 0.1, 1.0))

        with histogram.time():
            time.sleep(0.05)

        assert histogram.get_sample_count() == 1
        assert histogram.get_sample_sum() >= 0.05

    def test_histogram_prometheus_format(self) -> None:
        """Test Prometheus exposition format."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            namespace="app",
            buckets=(0.1, 1.0),
        )
        histogram.observe(0.5)

        output = histogram.to_prometheus_format()
        assert "# HELP app_test_histogram Test histogram" in output
        assert "# TYPE app_test_histogram histogram" in output
        assert "app_test_histogram_bucket" in output
        assert "app_test_histogram_sum" in output
        assert "app_test_histogram_count" in output
        assert 'le="0.1"' in output
        assert 'le="+Inf"' in output


class TestSummary:
    """Tests for Summary metric."""

    def test_summary_creation(self) -> None:
        """Test summary creation."""
        summary = Summary(
            name="response_size",
            description="Response size bytes",
        )

        assert summary.name == "response_size"
        assert summary.metric_type == MetricType.SUMMARY

    def test_summary_observe(self) -> None:
        """Test summary observation."""
        summary = Summary("size", "Size")

        for i in range(100):
            summary.observe(i)

        output = summary.to_prometheus_format()
        assert "_sum" in output
        assert "_count" in output
        assert 'quantile="0.5"' in output
        assert 'quantile="0.9"' in output

    def test_summary_with_labels(self) -> None:
        """Test summary with labels."""
        summary = Summary("size", "Size", labels=["endpoint"])

        summary.observe(100, labels={"endpoint": "/api/v1"})
        summary.observe(200, labels={"endpoint": "/api/v2"})

        output = summary.to_prometheus_format()
        assert 'endpoint="/api/v1"' in output
        assert 'endpoint="/api/v2"' in output


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = MetricsRegistry(namespace="test")
        assert registry.config.namespace == "test"

    def test_registry_counter(self) -> None:
        """Test creating counter via registry."""
        registry = MetricsRegistry(namespace="app")
        counter = registry.counter("requests", "Total requests")

        assert counter.full_name == "app_requests"
        counter.inc()
        assert counter.get() == 1.0

    def test_registry_gauge(self) -> None:
        """Test creating gauge via registry."""
        registry = MetricsRegistry(namespace="app")
        gauge = registry.gauge("temperature", "Temperature")

        assert gauge.full_name == "app_temperature"
        gauge.set(42)
        assert gauge.get() == 42.0

    def test_registry_histogram(self) -> None:
        """Test creating histogram via registry."""
        registry = MetricsRegistry(namespace="app")
        histogram = registry.histogram(
            "latency",
            "Latency",
            buckets=(0.1, 1.0),
        )

        assert histogram.full_name == "app_latency"
        histogram.observe(0.5)
        assert histogram.get_sample_count() == 1

    def test_registry_summary(self) -> None:
        """Test creating summary via registry."""
        registry = MetricsRegistry(namespace="app")
        summary = registry.summary("size", "Size")

        assert summary.full_name == "app_size"
        summary.observe(100)

    def test_registry_duplicate_metric(self) -> None:
        """Test handling duplicate metric registration."""
        registry = MetricsRegistry(namespace="app")
        counter1 = registry.counter("test", "Test")
        counter2 = registry.counter("test", "Test duplicate")

        # Should return same counter
        assert counter1 is counter2

    def test_registry_get(self) -> None:
        """Test getting metric by name."""
        registry = MetricsRegistry(namespace="app")
        registry.counter("test", "Test")

        metric = registry.get("test")
        assert metric is not None
        assert metric.name == "test"

    def test_registry_expose(self) -> None:
        """Test exposing all metrics."""
        registry = MetricsRegistry(namespace="app")
        registry.counter("requests", "Requests").inc()
        registry.gauge("temp", "Temp").set(42)

        output = registry.expose()
        assert "app_requests" in output
        assert "app_temp" in output

    def test_registry_clear(self) -> None:
        """Test clearing all metric values."""
        registry = MetricsRegistry(namespace="app")
        counter = registry.counter("test", "Test")
        counter.inc(10)

        registry.clear()
        assert counter.get() == 0.0

    def test_registry_unregister(self) -> None:
        """Test unregistering a metric."""
        registry = MetricsRegistry(namespace="app")
        registry.counter("test", "Test")

        assert registry.unregister("test") is True
        assert registry.get("test") is None
        assert registry.unregister("nonexistent") is False


class TestDefaultRegistry:
    """Tests for default registry and pre-defined metrics."""

    def test_get_default_registry(self) -> None:
        """Test getting default registry."""
        registry = get_default_registry()
        assert registry is not None
        assert isinstance(registry, MetricsRegistry)

    def test_calculation_counter(self) -> None:
        """Test pre-defined calculation counter."""
        calculation_counter.inc(labels={
            "agent_id": "GL-006",
            "calculation_type": "pinch",
            "status": "success",
        })

        value = calculation_counter.get(labels={
            "agent_id": "GL-006",
            "calculation_type": "pinch",
            "status": "success",
        })
        assert value >= 1.0

    def test_calculation_latency(self) -> None:
        """Test pre-defined calculation latency histogram."""
        with calculation_latency.time(labels={
            "agent_id": "GL-006",
            "calculation_type": "pinch",
        }):
            time.sleep(0.01)

    def test_queue_depth(self) -> None:
        """Test pre-defined queue depth gauge."""
        queue_depth.set(10, labels={
            "agent_id": "GL-006",
            "queue_name": "input",
        })

        value = queue_depth.get(labels={
            "agent_id": "GL-006",
            "queue_name": "input",
        })
        assert value == 10.0

    def test_active_tasks(self) -> None:
        """Test pre-defined active tasks gauge."""
        active_tasks.set(5, labels={
            "agent_id": "GL-006",
            "task_type": "calculation",
        })

        active_tasks.inc(labels={
            "agent_id": "GL-006",
            "task_type": "calculation",
        })

        value = active_tasks.get(labels={
            "agent_id": "GL-006",
            "task_type": "calculation",
        })
        assert value == 6.0

    def test_response_size(self) -> None:
        """Test pre-defined response size summary."""
        response_size.observe(1024, labels={
            "agent_id": "GL-006",
            "endpoint": "/calculate",
        })


class TestThreadSafety:
    """Tests for thread safety of metrics."""

    def test_counter_thread_safety(self) -> None:
        """Test counter is thread-safe."""
        counter = Counter("concurrent", "Concurrent counter")
        threads = []
        iterations = 1000

        def increment() -> None:
            for _ in range(iterations):
                counter.inc()

        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert counter.get() == iterations * 10

    def test_histogram_thread_safety(self) -> None:
        """Test histogram is thread-safe."""
        histogram = Histogram("concurrent", "Concurrent", buckets=(1, 10))
        threads = []
        iterations = 100

        def observe() -> None:
            for i in range(iterations):
                histogram.observe(i % 10)

        for _ in range(10):
            t = threading.Thread(target=observe)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert histogram.get_sample_count() == iterations * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
