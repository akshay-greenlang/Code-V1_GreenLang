# -*- coding: utf-8 -*-
"""
Comprehensive Performance Monitoring System
GL-VCCI Scope 3 Platform - Performance Optimization

This module provides comprehensive performance monitoring and metrics:
- Database query performance tracking
- API endpoint latency monitoring
- Cache hit rate metrics
- Connection pool monitoring
- Slow query logging
- Prometheus metrics export

Metrics Tracked:
- Query duration (P50, P95, P99)
- Request latency (P50, P95, P99)
- Throughput (requests/second)
- Cache hit rates
- Connection pool utilization
- Error rates

Version: 1.0.0
Team: Performance Optimization Team
Date: 2025-11-09
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from collections import deque
import statistics

from prometheus_client import (
from greenlang.determinism import DeterministicClock
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

class PerformanceMetrics:
    """
    Comprehensive performance metrics using Prometheus.

    Metrics:
    - Database performance
    - API performance
    - Cache performance
    - System resources
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize performance metrics.

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()

        # ===================================================================
        # DATABASE METRICS
        # ===================================================================

        # Query duration histogram
        self.db_query_duration = Histogram(
            "greenlang_database_query_duration_seconds",
            "Database query duration in seconds",
            ["query_type"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry
        )

        # Connection pool metrics
        self.db_pool_size = Gauge(
            "greenlang_database_connection_pool_size",
            "Database connection pool size",
            ["state"],  # checked_out, available, overflow
            registry=self.registry
        )

        # Slow queries counter
        self.db_slow_queries = Counter(
            "greenlang_database_slow_queries_total",
            "Total number of slow database queries (>1s)",
            ["query_type"],
            registry=self.registry
        )

        # Query errors
        self.db_query_errors = Counter(
            "greenlang_database_query_errors_total",
            "Total number of database query errors",
            ["error_type"],
            registry=self.registry
        )

        # ===================================================================
        # API METRICS
        # ===================================================================

        # HTTP request duration
        self.http_request_duration = Histogram(
            "greenlang_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint", "status"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        # HTTP requests total
        self.http_requests_total = Counter(
            "greenlang_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry
        )

        # Active requests gauge
        self.http_requests_active = Gauge(
            "greenlang_http_requests_active",
            "Number of active HTTP requests",
            ["method", "endpoint"],
            registry=self.registry
        )

        # ===================================================================
        # CACHE METRICS
        # ===================================================================

        # Cache hit rate
        self.cache_hit_rate = Gauge(
            "greenlang_cache_hit_rate",
            "Cache hit rate percentage",
            ["cache_type"],  # l1, l2, api
            registry=self.registry
        )

        # Cache operations
        self.cache_operations = Counter(
            "greenlang_cache_operations_total",
            "Total cache operations",
            ["cache_type", "operation", "result"],  # operation: get/set, result: hit/miss
            registry=self.registry
        )

        # Cache evictions
        self.cache_evictions = Counter(
            "greenlang_cache_evictions_total",
            "Total cache evictions",
            ["cache_type"],
            registry=self.registry
        )

        # ===================================================================
        # BATCH PROCESSING METRICS
        # ===================================================================

        # Batch processing duration
        self.batch_processing_duration = Histogram(
            "greenlang_batch_processing_duration_seconds",
            "Batch processing duration in seconds",
            ["batch_type"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
            registry=self.registry
        )

        # Batch throughput
        self.batch_throughput = Gauge(
            "greenlang_batch_throughput_records_per_second",
            "Batch processing throughput in records/second",
            ["batch_type"],
            registry=self.registry
        )

        # ===================================================================
        # BUSINESS METRICS
        # ===================================================================

        # Emissions calculations
        self.emissions_calculations_total = Counter(
            "greenlang_emissions_calculations_total",
            "Total emissions calculations performed",
            ["scope3_category", "status"],
            registry=self.registry
        )

        # Supplier data ingestion
        self.supplier_ingestion_total = Counter(
            "greenlang_supplier_ingestion_total",
            "Total supplier records ingested",
            ["status"],
            registry=self.registry
        )

    # =======================================================================
    # CONTEXT MANAGERS FOR AUTOMATIC TRACKING
    # =======================================================================

    @contextmanager
    def track_db_query(self, query_type: str, slow_threshold: float = 1.0):
        """
        Context manager to track database query performance.

        Usage:
            with metrics.track_db_query("select_emissions"):
                result = db.query(Emission).all()
        """
        start_time = time.time()

        try:
            yield
        except Exception as e:
            self.db_query_errors.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time

            # Record duration
            self.db_query_duration.labels(query_type=query_type).observe(duration)

            # Track slow queries
            if duration > slow_threshold:
                self.db_slow_queries.labels(query_type=query_type).inc()
                logger.warning(
                    f"Slow query detected: {query_type} took {duration:.2f}s"
                )

    @asynccontextmanager
    async def track_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: Optional[int] = None
    ):
        """
        Async context manager to track HTTP request performance.

        Usage:
            async with metrics.track_http_request("GET", "/api/v1/emissions"):
                response = await process_request()
        """
        start_time = time.time()

        # Increment active requests
        self.http_requests_active.labels(method=method, endpoint=endpoint).inc()

        try:
            yield
        finally:
            duration = time.time() - start_time

            # Decrement active requests
            self.http_requests_active.labels(method=method, endpoint=endpoint).dec()

            # Use provided status code or default
            status = str(status_code) if status_code else "200"

            # Record metrics
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).observe(duration)

            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()

    def track_cache_operation(
        self,
        cache_type: str,
        operation: str,
        result: str
    ):
        """
        Track cache operation.

        Args:
            cache_type: Type of cache (l1, l2, api)
            operation: Operation type (get, set, delete)
            result: Operation result (hit, miss, success, error)
        """
        self.cache_operations.labels(
            cache_type=cache_type,
            operation=operation,
            result=result
        ).inc()

    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate metric"""
        self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)

    def update_pool_size(
        self,
        checked_out: int,
        available: int,
        overflow: int
    ):
        """Update connection pool size metrics"""
        self.db_pool_size.labels(state="checked_out").set(checked_out)
        self.db_pool_size.labels(state="available").set(available)
        self.db_pool_size.labels(state="overflow").set(overflow)


# ============================================================================
# SLOW QUERY LOGGER
# ============================================================================

@dataclass
class SlowQueryRecord:
    """Record of a slow query"""
    query_text: str
    duration_seconds: float
    timestamp: datetime
    query_type: str
    params: Optional[Dict[str, Any]] = None


class SlowQueryLogger:
    """
    Logger for slow database queries.

    Features:
    - Automatic slow query detection
    - Query history tracking
    - Top N slowest queries
    """

    def __init__(
        self,
        threshold_seconds: float = 1.0,
        max_history: int = 100
    ):
        """
        Initialize slow query logger.

        Args:
            threshold_seconds: Queries slower than this are logged
            max_history: Maximum number of slow queries to keep
        """
        self.threshold_seconds = threshold_seconds
        self.slow_queries: deque = deque(maxlen=max_history)

    def log_query(
        self,
        query_text: str,
        duration_seconds: float,
        query_type: str = "unknown",
        params: Optional[Dict[str, Any]] = None
    ):
        """Log query if it exceeds threshold"""
        if duration_seconds > self.threshold_seconds:
            record = SlowQueryRecord(
                query_text=query_text[:500],  # Truncate long queries
                duration_seconds=duration_seconds,
                timestamp=DeterministicClock.utcnow(),
                query_type=query_type,
                params=params
            )

            self.slow_queries.append(record)

            logger.warning(
                f"Slow query detected: {query_type} took {duration_seconds:.2f}s\n"
                f"Query: {query_text[:200]}..."
            )

    def get_slowest_queries(self, n: int = 10) -> List[SlowQueryRecord]:
        """Get N slowest queries"""
        sorted_queries = sorted(
            self.slow_queries,
            key=lambda q: q.duration_seconds,
            reverse=True
        )
        return sorted_queries[:n]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.slow_queries:
            return {
                "total_slow_queries": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "slowest_query_type": None
            }

        durations = [q.duration_seconds for q in self.slow_queries]

        return {
            "total_slow_queries": len(self.slow_queries),
            "avg_duration": round(statistics.mean(durations), 2),
            "max_duration": round(max(durations), 2),
            "p95_duration": round(statistics.quantiles(durations, n=20)[18], 2),
            "slowest_query_type": max(
                self.slow_queries,
                key=lambda q: q.duration_seconds
            ).query_type
        }


# ============================================================================
# LATENCY TRACKER
# ============================================================================

class LatencyTracker:
    """
    Track and calculate latency percentiles.

    Tracks P50, P95, P99 latencies for performance monitoring.
    """

    def __init__(self, max_samples: int = 10000):
        """
        Initialize latency tracker.

        Args:
            max_samples: Maximum number of samples to keep
        """
        self.samples: deque = deque(maxlen=max_samples)

    def record(self, latency_seconds: float):
        """Record a latency sample"""
        self.samples.append(latency_seconds)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.samples:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "count": 0
            }

        sorted_samples = sorted(self.samples)
        count = len(sorted_samples)

        return {
            "p50": round(sorted_samples[int(count * 0.50)], 3),
            "p95": round(sorted_samples[int(count * 0.95)], 3),
            "p99": round(sorted_samples[int(count * 0.99)], 3),
            "min": round(min(sorted_samples), 3),
            "max": round(max(sorted_samples), 3),
            "avg": round(statistics.mean(sorted_samples), 3),
            "count": count
        }


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Centralized performance monitoring.

    Combines all performance tracking into single interface.
    """

    def __init__(self):
        """Initialize performance monitor"""
        self.metrics = PerformanceMetrics()
        self.slow_query_logger = SlowQueryLogger()
        self.api_latency_tracker = LatencyTracker()
        self.db_latency_tracker = LatencyTracker()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "api_latency": self.api_latency_tracker.get_percentiles(),
            "database_latency": self.db_latency_tracker.get_percentiles(),
            "slow_queries": self.slow_query_logger.get_summary(),
            "prometheus_metrics": "/metrics"  # Link to Prometheus endpoint
        }

    def export_prometheus_metrics(self) -> bytes:
        """Export Prometheus metrics"""
        return generate_latest(self.metrics.registry)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()

    return _performance_monitor


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_metrics_endpoint():
    """
    Create FastAPI endpoint for Prometheus metrics.

    Usage:
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.get("/metrics")
        async def metrics():
            monitor = get_performance_monitor()
            content = monitor.export_prometheus_metrics()
            return Response(content=content, media_type=CONTENT_TYPE_LATEST)
    """
    pass


__all__ = [
    "PerformanceMetrics",
    "SlowQueryLogger",
    "SlowQueryRecord",
    "LatencyTracker",
    "PerformanceMonitor",
    "get_performance_monitor",
]
