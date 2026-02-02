# -*- coding: utf-8 -*-
"""
Tests for metrics collection and exposition
"""

import pytest
import time
from greenlang.observability import (
    MetricsCollector,
    MetricType,
    CustomMetric,
    MetricsAggregator,
    get_metrics_collector,
    track_execution,
    track_api_request,
    track_cache,
)

try:
    from prometheus_client import CollectorRegistry
except ImportError:
    CollectorRegistry = None


class TestMetricsCollector:
    """Test MetricsCollector functionality"""

    def test_metrics_collector_initialization(self):
        """Test metrics collector can be initialized"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        assert collector is not None
        assert collector.custom_metrics == {}

    def test_register_counter_metric(self):
        """Test registering a counter metric"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_counter_1",
            type=MetricType.COUNTER,
            description="Test counter",
            labels=["tenant_id"],
        )
        collector.register_custom_metric(metric)
        assert "test_counter_1" in collector.custom_metrics

    def test_register_gauge_metric(self):
        """Test registering a gauge metric"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_gauge_1",
            type=MetricType.GAUGE,
            description="Test gauge",
            labels=["resource_type"],
        )
        collector.register_custom_metric(metric)
        assert "test_gauge_1" in collector.custom_metrics

    def test_register_histogram_metric(self):
        """Test registering a histogram metric"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_histogram_1",
            type=MetricType.HISTOGRAM,
            description="Test histogram",
            labels=["operation"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0],
        )
        collector.register_custom_metric(metric)
        assert "test_histogram_1" in collector.custom_metrics

    def test_record_counter_metric(self):
        """Test recording counter metric value"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_counter_2",
            type=MetricType.COUNTER,
            description="Test counter",
            labels=["status"],
        )
        collector.register_custom_metric(metric)
        collector.record_metric("test_counter_2", 1.0, {"status": "success"})
        # No exception means success

    def test_record_gauge_metric(self):
        """Test recording gauge metric value"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_gauge_2",
            type=MetricType.GAUGE,
            description="Test gauge",
            labels=["resource"],
        )
        collector.register_custom_metric(metric)
        collector.record_metric("test_gauge_2", 42.0, {"resource": "cpu"})
        # No exception means success

    def test_record_histogram_metric(self):
        """Test recording histogram metric value"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metric = CustomMetric(
            name="test_histogram_2",
            type=MetricType.HISTOGRAM,
            description="Test histogram",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0],
        )
        collector.register_custom_metric(metric)
        collector.record_metric("test_histogram_2", 0.25, {"endpoint": "/api/test"})
        # No exception means success

    def test_collect_system_metrics(self):
        """Test collecting system-level metrics"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        collector.collect_system_metrics(tenant_id="test")
        # Should not raise exception

    def test_get_metrics_prometheus_format(self):
        """Test getting metrics in Prometheus format"""
        registry = CollectorRegistry() if CollectorRegistry else None
        collector = MetricsCollector(registry=registry)
        metrics_data = collector.get_metrics()
        assert isinstance(metrics_data, bytes)


class TestMetricsDecorators:
    """Test metrics decorator functions"""

    def test_track_execution_decorator_success(self):
        """Test track_execution decorator on successful function"""

        @track_execution(pipeline="test_pipeline", tenant_id="test")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_track_execution_decorator_failure(self):
        """Test track_execution decorator on failing function"""

        @track_execution(pipeline="test_pipeline", tenant_id="test")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_track_api_request_decorator(self):
        """Test track_api_request decorator"""

        class Response:
            status_code = 200

        @track_api_request(method="GET", endpoint="/test", tenant_id="test")
        def api_handler():
            return Response()

        result = api_handler()
        assert result.status_code == 200

    def test_track_cache_decorator_hit(self):
        """Test track_cache decorator on cache hit"""

        @track_cache(cache_name="test_cache", tenant_id="test")
        def cache_lookup():
            return {"data": "cached_value"}

        result = cache_lookup()
        assert result is not None

    def test_track_cache_decorator_miss(self):
        """Test track_cache decorator on cache miss"""

        @track_cache(cache_name="test_cache", tenant_id="test")
        def cache_lookup_miss():
            return None

        result = cache_lookup_miss()
        assert result is None


class TestMetricsAggregator:
    """Test MetricsAggregator functionality"""

    def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        aggregator = MetricsAggregator()
        assert aggregator is not None
        assert aggregator.retention_hours == 24

    def test_add_metric(self):
        """Test adding metric to aggregator"""
        aggregator = MetricsAggregator()
        aggregator.add_metric("test_metric", 42.0)
        assert "test_metric" in aggregator.metrics_buffer

    def test_get_aggregates_empty(self):
        """Test getting aggregates with no data"""
        aggregator = MetricsAggregator()
        result = aggregator.get_aggregates("nonexistent_metric")
        assert result == {}

    def test_get_aggregates_with_data(self):
        """Test getting aggregates with data"""
        aggregator = MetricsAggregator()
        aggregator.add_metric("test_metric", 10.0)
        aggregator.add_metric("test_metric", 20.0)
        aggregator.add_metric("test_metric", 30.0)

        result = aggregator.get_aggregates("test_metric", window_minutes=60)
        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        assert result["min"] == 10.0
        assert result["max"] == 30.0
        assert result["avg"] == 20.0

    def test_percentile_calculation(self):
        """Test percentile calculation"""
        aggregator = MetricsAggregator()
        values = list(range(1, 101))  # 1 to 100
        p50 = aggregator._percentile(values, 50)
        p95 = aggregator._percentile(values, 95)
        p99 = aggregator._percentile(values, 99)

        assert 49 <= p50 <= 51
        assert 94 <= p95 <= 96
        assert 98 <= p99 <= 100


class TestGlobalMetricsInstances:
    """Test global metrics instances"""

    def test_get_metrics_collector_singleton(self):
        """Test getting global metrics collector"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2

    def test_metrics_collector_is_functional(self):
        """Test global metrics collector is functional"""
        collector = get_metrics_collector()
        collector.collect_system_metrics()
        # Should not raise exception
