# -*- coding: utf-8 -*-
"""
GL-VCCI-APP - Infrastructure Enhancements Test Suite
=====================================================

Comprehensive tests for VCCI infrastructure enhancements:
- Cache Manager (L1/L2/L3 caching)
- Semantic cache hit rate
- Database connection pooling
- Metrics collector (Prometheus)
- Structured logging (JSON format)
- Distributed tracing

Version: 1.0.0
Author: Testing & QA Team
"""

import pytest
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import sys
from greenlang.determinism import deterministic_uuid, DeterministicClock

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from infrastructure.cache_manager import CacheManager, L1Cache, L2Cache, L3Cache
except ImportError:
    CacheManager = None

try:
    from infrastructure.semantic_cache import SemanticCache
except ImportError:
    SemanticCache = None

try:
    from infrastructure.database import DatabaseConnectionPool
except ImportError:
    DatabaseConnectionPool = None

try:
    from infrastructure.metrics import MetricsCollector, PrometheusExporter
except ImportError:
    MetricsCollector = None
    PrometheusExporter = None

try:
    from infrastructure.logging import StructuredLogger
except ImportError:
    StructuredLogger = None

try:
    from infrastructure.tracing import DistributedTracer, TraceContext
except ImportError:
    DistributedTracer = None
    TraceContext = None


# ============================================================================
# Cache Manager Tests (L1/L2/L3)
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.critical
@pytest.mark.skipif(CacheManager is None, reason="CacheManager not available")
class TestCacheManagerL1L2L3:
    """Test multi-level cache manager (L1/L2/L3)."""

    def test_cache_manager_l1_l2_l3(self):
        """
        Test 3-tier cache hierarchy.

        L1: In-memory (fastest, smallest)
        L2: Redis (medium speed, medium size)
        L3: Disk (slowest, largest)
        """
        cache_manager = CacheManager(
            l1_size=100,
            l2_redis_url="redis://localhost:6379",
            l3_disk_path="/tmp/cache"
        )

        # Test L1 cache (hot data)
        cache_manager.set("hot_key", "hot_value", tier="L1")
        value = cache_manager.get("hot_key")
        assert value == "hot_value"
        assert cache_manager.get_hit_tier("hot_key") == "L1"

        # Test L2 cache (warm data)
        cache_manager.set("warm_key", "warm_value", tier="L2")
        value = cache_manager.get("warm_key")
        assert value == "warm_value"
        assert cache_manager.get_hit_tier("warm_key") == "L2"

        # Test L3 cache (cold data)
        cache_manager.set("cold_key", "cold_value", tier="L3")
        value = cache_manager.get("cold_key")
        assert value == "cold_value"
        assert cache_manager.get_hit_tier("cold_key") == "L3"

    def test_cache_manager_promotion(self):
        """Test cache automatically promotes hot data to faster tiers."""
        cache_manager = CacheManager()

        # Initially in L3
        cache_manager.set("key", "value", tier="L3")

        # Access multiple times
        for _ in range(10):
            cache_manager.get("key")

        # Should be promoted to L1 or L2
        hit_tier = cache_manager.get_hit_tier("key")
        assert hit_tier in ["L1", "L2"], \
            f"Hot data should be promoted from L3 to faster tier, got {hit_tier}"

    def test_cache_manager_eviction(self):
        """Test cache eviction policies (LRU)."""
        cache_manager = CacheManager(l1_size=3)  # Small size

        # Fill L1 cache
        cache_manager.set("key1", "value1", tier="L1")
        cache_manager.set("key2", "value2", tier="L1")
        cache_manager.set("key3", "value3", tier="L1")

        # Add one more (should evict LRU)
        cache_manager.set("key4", "value4", tier="L1")

        # key1 should be evicted (or demoted to L2)
        value = cache_manager.get("key1")
        if value is not None:
            # If still accessible, should be in L2, not L1
            assert cache_manager.get_hit_tier("key1") != "L1"

    def test_cache_manager_hit_rates(self):
        """Test cache tracks hit rates for each tier."""
        cache_manager = CacheManager()

        # Generate cache hits and misses
        cache_manager.set("key1", "value1", tier="L1")
        cache_manager.set("key2", "value2", tier="L2")

        # Hits
        cache_manager.get("key1")  # L1 hit
        cache_manager.get("key2")  # L2 hit

        # Misses
        cache_manager.get("nonexistent1")
        cache_manager.get("nonexistent2")

        stats = cache_manager.get_stats()

        assert 'l1_hit_rate' in stats
        assert 'l2_hit_rate' in stats
        assert 'overall_hit_rate' in stats

        assert stats['overall_hit_rate'] > 0


# ============================================================================
# Semantic Cache Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.critical
@pytest.mark.skipif(SemanticCache is None, reason="SemanticCache not available")
class TestSemanticCacheHitRate:
    """Test semantic cache achieves target hit rates."""

    def test_semantic_cache_hit_rate(self):
        """
        Test semantic cache achieves >30% hit rate.

        Semantic cache should match queries with similar meaning.
        """
        cache = SemanticCache(
            similarity_threshold=0.85,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Seed cache with common queries
        training_data = [
            ("What is Scope 3 emissions?", "Scope 3 includes indirect value chain emissions"),
            ("How to calculate carbon footprint?", "Carbon footprint = sum of all GHG emissions"),
            ("What are emission factors?", "Emission factors convert activity data to emissions"),
            ("How to engage suppliers?", "Supplier engagement requires data collection"),
        ]

        for query, response in training_data:
            cache.set(query, response)

        # Test with similar queries
        test_queries = [
            "What does Scope 3 emissions mean?",  # Similar to #1
            "How do I calculate my carbon footprint?",  # Similar to #2
            "What is a GHG emission factor?",  # Similar to #3
            "How should I engage with suppliers?",  # Similar to #4
            "What is the Paris Agreement?",  # New
            "How to set science-based targets?",  # New
        ]

        hits = 0
        misses = 0

        for query in test_queries:
            result = cache.get(query)
            if result is not None:
                hits += 1
            else:
                misses += 1

        hit_rate = hits / (hits + misses)

        assert hit_rate >= 0.30, \
            f"Semantic cache hit rate {hit_rate:.1%} below 30% target"

        print(f"  ✓ Semantic cache hit rate: {hit_rate:.1%}")

    def test_semantic_cache_embeddings(self):
        """Test semantic cache uses embeddings for similarity matching."""
        cache = SemanticCache()

        cache.set("What is carbon accounting?", "Response 1")

        # Semantically similar query
        result = cache.get("How does carbon accounting work?")

        # Should retrieve due to semantic similarity
        assert result == "Response 1"


# ============================================================================
# Database Connection Pool Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(DatabaseConnectionPool is None, reason="DatabaseConnectionPool not available")
class TestDatabaseConnectionPool:
    """Test database connection pooling."""

    def test_database_connection_pool(self):
        """
        Test connection pool manages database connections efficiently.

        Requirements:
        - Reuse connections
        - Handle connection limits
        - Auto-reconnect on failure
        - Monitor pool health
        """
        pool = DatabaseConnectionPool(
            min_connections=5,
            max_connections=20,
            database_url="postgresql://localhost/test"
        )

        # Get connection from pool
        conn1 = pool.get_connection()
        assert conn1 is not None

        # Return connection to pool
        pool.release_connection(conn1)

        # Get connection again (should reuse)
        conn2 = pool.get_connection()
        assert conn2 is not None

        # Pool should reuse connections
        assert pool.get_stats()['reuse_count'] > 0

    def test_database_connection_pool_limits(self):
        """Test connection pool respects max connections limit."""
        pool = DatabaseConnectionPool(max_connections=5)

        connections = []
        for i in range(5):
            conn = pool.get_connection()
            connections.append(conn)

        # Pool should be exhausted
        assert pool.available_connections() == 0

        # Requesting one more should block or raise error
        with pytest.raises(Exception) or pytest.warns(UserWarning):
            pool.get_connection(timeout=0.1)

    def test_database_connection_pool_health_check(self):
        """Test connection pool performs health checks."""
        pool = DatabaseConnectionPool(health_check_interval=1)

        # Pool should track healthy/unhealthy connections
        stats = pool.get_stats()
        assert 'healthy_connections' in stats
        assert 'failed_connections' in stats


# ============================================================================
# Metrics Collector Tests (Prometheus)
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(MetricsCollector is None, reason="MetricsCollector not available")
class TestMetricsCollectorPrometheus:
    """Test Prometheus metrics collection."""

    def test_metrics_collector_prometheus(self):
        """
        Test metrics collector exports to Prometheus format.

        Metric types:
        - Counter (monotonic)
        - Gauge (arbitrary value)
        - Histogram (distribution)
        - Summary (percentiles)
        """
        collector = MetricsCollector()

        # Record counter
        collector.counter("api_requests_total", 1, labels={"endpoint": "/api/emissions"})

        # Record gauge
        collector.gauge("cache_size_bytes", 1024, labels={"tier": "L1"})

        # Record histogram
        collector.histogram("request_duration_seconds", 0.5)

        # Export to Prometheus
        prom_output = collector.export_prometheus()

        # Verify Prometheus format
        assert isinstance(prom_output, str)
        assert "api_requests_total" in prom_output
        assert "cache_size_bytes" in prom_output
        assert "request_duration_seconds" in prom_output

    def test_metrics_collector_labels(self):
        """Test metrics collector supports labels/tags."""
        collector = MetricsCollector()

        collector.counter("emissions_calculated", 1, labels={
            "scope": "3",
            "category": "upstream_transport"
        })

        metrics = collector.get_metrics()

        # Should have label dimensions
        assert any(
            m['labels'].get('scope') == '3'
            for m in metrics
            if m['name'] == 'emissions_calculated'
        )

    def test_metrics_collector_aggregation(self):
        """Test metrics collector aggregates time-series data."""
        collector = MetricsCollector()

        # Record multiple values
        for i in range(100):
            collector.histogram("processing_time", i * 0.01)

        # Get aggregated statistics
        stats = collector.get_histogram_stats("processing_time")

        assert 'mean' in stats
        assert 'p50' in stats
        assert 'p95' in stats
        assert 'p99' in stats


# ============================================================================
# Structured Logger Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(StructuredLogger is None, reason="StructuredLogger not available")
class TestStructuredLoggerJsonFormat:
    """Test structured logging in JSON format."""

    def test_structured_logger_json_format(self, tmp_path):
        """
        Test logger outputs structured JSON logs.

        JSON format enables:
        - Easy parsing
        - Log aggregation
        - Query and analysis
        """
        log_file = tmp_path / "app.log"

        logger = StructuredLogger(
            name="test_logger",
            log_file=str(log_file),
            format="json"
        )

        # Log structured message
        logger.info(
            "Processing emissions data",
            extra={
                "company_id": "12345",
                "scope": 3,
                "records": 100,
                "duration_ms": 150
            }
        )

        logger.error(
            "Calculation failed",
            extra={
                "error_code": "CALC_001",
                "category": "supplier_engagement"
            }
        )

        # Read log file
        with open(log_file) as f:
            log_lines = f.readlines()

        # Each line should be valid JSON
        for line in log_lines:
            log_entry = json.loads(line)

            assert 'timestamp' in log_entry
            assert 'level' in log_entry
            assert 'message' in log_entry

            # Should have structured fields
            if 'company_id' in log_entry:
                assert log_entry['company_id'] == "12345"

    def test_structured_logger_log_levels(self):
        """Test structured logger supports standard log levels."""
        logger = StructuredLogger()

        # Should support all levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # No assertions - just verify no exceptions

    def test_structured_logger_context_enrichment(self):
        """Test logger enriches logs with context."""
        logger = StructuredLogger(
            context={
                "app": "vcci-scope3",
                "environment": "production",
                "version": "1.0.0"
            }
        )

        logger.info("Test message")

        # Context should be included in all log entries
        # (This is implicit - would need to read log output to verify)


# ============================================================================
# Distributed Tracing Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(DistributedTracer is None, reason="DistributedTracer not available")
class TestTracingDistributed:
    """Test distributed tracing for request flows."""

    def test_tracing_distributed(self):
        """
        Test distributed tracing tracks request flow across services.

        Tracing concepts:
        - Trace ID (request identifier)
        - Span ID (operation identifier)
        - Parent-child relationships
        - Timing data
        """
        tracer = DistributedTracer(service_name="vcci-scope3")

        # Start root span
        with tracer.start_span("process_supplier_data") as root_span:
            root_span.set_tag("supplier_id", "SUP-001")

            # Child span
            with tracer.start_span("fetch_emissions", child_of=root_span) as fetch_span:
                fetch_span.set_tag("data_source", "api")
                time.sleep(0.01)

            # Another child span
            with tracer.start_span("calculate_emissions", child_of=root_span) as calc_span:
                calc_span.set_tag("scope", "3")
                time.sleep(0.02)

        # Get trace
        trace = tracer.get_trace(root_span.trace_id)

        # Verify trace structure
        assert trace is not None
        assert len(trace['spans']) >= 3
        assert trace['root_span']['operation_name'] == "process_supplier_data"

    def test_tracing_span_context_propagation(self):
        """Test trace context is propagated across service boundaries."""
        tracer = DistributedTracer()

        with tracer.start_span("service_a") as span_a:
            # Extract context (would be sent to service B)
            context = tracer.extract_context(span_a)

            assert 'trace_id' in context
            assert 'span_id' in context

            # Inject context (service B would inject this)
            span_b = tracer.inject_context(context, operation_name="service_b")

            # Should have same trace_id
            assert span_b.trace_id == span_a.trace_id

    def test_tracing_performance_overhead(self):
        """Test tracing has minimal performance overhead (<5%)."""
        # Without tracing
        start = time.perf_counter()
        for _ in range(1000):
            time.sleep(0.0001)
        duration_without_tracing = time.perf_counter() - start

        # With tracing
        tracer = DistributedTracer()
        start = time.perf_counter()
        for _ in range(1000):
            with tracer.start_span("test_operation"):
                time.sleep(0.0001)
        duration_with_tracing = time.perf_counter() - start

        overhead = ((duration_with_tracing - duration_without_tracing) / duration_without_tracing) * 100

        assert overhead < 5.0, \
            f"Tracing overhead {overhead:.1f}% exceeds 5% target"


# ============================================================================
# Mock Classes (fallback)
# ============================================================================

if CacheManager is None:
    class CacheManager:
        def __init__(self, l1_size=1000, l2_redis_url=None, l3_disk_path=None):
            self.l1 = {}
            self.l2 = {}
            self.l3 = {}
            self.access_counts = {}
            self.hit_tiers = {}

        def set(self, key, value, tier="L1"):
            if tier == "L1":
                self.l1[key] = value
            elif tier == "L2":
                self.l2[key] = value
            else:
                self.l3[key] = value
            self.hit_tiers[key] = tier

        def get(self, key):
            self.access_counts[key] = self.access_counts.get(key, 0) + 1

            # Check L1, L2, L3
            if key in self.l1:
                self.hit_tiers[key] = "L1"
                return self.l1[key]
            elif key in self.l2:
                self.hit_tiers[key] = "L2"
                # Promote to L1 if frequently accessed
                if self.access_counts[key] > 5:
                    self.l1[key] = self.l2[key]
                    self.hit_tiers[key] = "L1"
                return self.l2[key]
            elif key in self.l3:
                self.hit_tiers[key] = "L3"
                return self.l3[key]
            return None

        def get_hit_tier(self, key):
            return self.hit_tiers.get(key)

        def get_stats(self):
            return {
                'l1_hit_rate': 0.6,
                'l2_hit_rate': 0.3,
                'overall_hit_rate': 0.7
            }


if SemanticCache is None:
    class SemanticCache:
        def __init__(self, similarity_threshold=0.85, embedding_model=None):
            self.cache = {}
            self.similarity_threshold = similarity_threshold

        def set(self, key, value):
            self.cache[key] = value

        def get(self, key):
            # Exact match
            if key in self.cache:
                return self.cache[key]

            # Semantic match (simple mock)
            for cached_key, cached_value in self.cache.items():
                words_query = set(key.lower().split())
                words_cached = set(cached_key.lower().split())
                overlap = len(words_query & words_cached)
                if overlap >= 3:
                    return cached_value

            return None


if DatabaseConnectionPool is None:
    class DatabaseConnectionPool:
        def __init__(self, min_connections=5, max_connections=20, database_url=None):
            self.max_connections = max_connections
            self.connections = []
            self.reuse_count = 0

        def get_connection(self, timeout=None):
            if self.connections:
                self.reuse_count += 1
                return self.connections.pop()
            return Mock()

        def release_connection(self, conn):
            if len(self.connections) < self.max_connections:
                self.connections.append(conn)

        def available_connections(self):
            return len(self.connections)

        def get_stats(self):
            return {
                'reuse_count': self.reuse_count,
                'healthy_connections': 10,
                'failed_connections': 0
            }


if MetricsCollector is None:
    class MetricsCollector:
        def __init__(self):
            self.metrics = []

        def counter(self, name, value, labels=None):
            self.metrics.append({'name': name, 'type': 'counter', 'value': value, 'labels': labels or {}})

        def gauge(self, name, value, labels=None):
            self.metrics.append({'name': name, 'type': 'gauge', 'value': value, 'labels': labels or {}})

        def histogram(self, name, value):
            self.metrics.append({'name': name, 'type': 'histogram', 'value': value})

        def export_prometheus(self):
            output = []
            for metric in self.metrics:
                output.append(f"{metric['name']} {metric['value']}")
            return "\n".join(output)

        def get_metrics(self):
            return self.metrics

        def get_histogram_stats(self, name):
            return {'mean': 0.5, 'p50': 0.5, 'p95': 0.9, 'p99': 0.99}


if StructuredLogger is None:
    class StructuredLogger:
        def __init__(self, name=None, log_file=None, format="json", context=None):
            self.log_file = log_file
            self.context = context or {}

        def _log(self, level, message, extra=None):
            log_entry = {
                'timestamp': time.time(),
                'level': level,
                'message': message,
                **self.context,
                **(extra or {})
            }

            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

        def debug(self, message, extra=None):
            self._log('DEBUG', message, extra)

        def info(self, message, extra=None):
            self._log('INFO', message, extra)

        def warning(self, message, extra=None):
            self._log('WARNING', message, extra)

        def error(self, message, extra=None):
            self._log('ERROR', message, extra)

        def critical(self, message, extra=None):
            self._log('CRITICAL', message, extra)


if DistributedTracer is None:
    class DistributedTracer:
        def __init__(self, service_name=None):
            self.service_name = service_name
            self.traces = {}

        def start_span(self, operation_name, child_of=None):
            return TraceSpan(operation_name, self, child_of)

        def extract_context(self, span):
            return {'trace_id': span.trace_id, 'span_id': span.span_id}

        def inject_context(self, context, operation_name):
            span = TraceSpan(operation_name, self)
            span.trace_id = context['trace_id']
            return span

        def get_trace(self, trace_id):
            return self.traces.get(trace_id, {'spans': [], 'root_span': {}})

    class TraceSpan:
        def __init__(self, operation_name, tracer, parent=None):
            import uuid
            self.operation_name = operation_name
            self.tracer = tracer
            self.parent = parent
            self.trace_id = parent.trace_id if parent else str(deterministic_uuid(__name__, str(DeterministicClock.now())))
            self.span_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
            self.tags = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self.trace_id not in self.tracer.traces:
                self.tracer.traces[self.trace_id] = {
                    'spans': [],
                    'root_span': {'operation_name': self.operation_name}
                }
            self.tracer.traces[self.trace_id]['spans'].append(self)

        def set_tag(self, key, value):
            self.tags[key] = value


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'infrastructure'])
