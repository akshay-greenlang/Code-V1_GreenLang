"""
Prometheus metrics collector for pgvector operations.

Exposes vector-specific metrics for monitoring search latency,
embedding counts, index sizes, and operational health.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed; metrics will be no-ops")


# ============================================================================
# Metric Definitions
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Vector embedding counts
    PGVECTOR_EMBEDDINGS_TOTAL = Gauge(
        "pgvector_embeddings_total",
        "Total number of vector embeddings stored",
        ["namespace"],
    )

    # Search latency histogram
    PGVECTOR_SEARCH_LATENCY = Histogram(
        "pgvector_search_latency_ms",
        "Vector search query latency in milliseconds",
        ["search_type", "namespace"],
        buckets=[5, 10, 20, 50, 75, 100, 150, 200, 500, 1000],
    )

    # Search throughput counter
    PGVECTOR_SEARCH_TOTAL = Counter(
        "pgvector_search_total",
        "Total number of vector searches performed",
        ["search_type", "namespace"],
    )

    # Insert rate counter
    PGVECTOR_INSERT_TOTAL = Counter(
        "pgvector_insert_total",
        "Total number of embeddings inserted",
        ["namespace", "status"],  # status: success, failed, duplicate
    )

    # Insert latency
    PGVECTOR_INSERT_LATENCY = Histogram(
        "pgvector_insert_latency_ms",
        "Batch insert latency in milliseconds",
        ["namespace"],
        buckets=[10, 50, 100, 500, 1000, 5000, 10000],
    )

    # Index size gauge
    PGVECTOR_INDEX_SIZE_BYTES = Gauge(
        "pgvector_index_size_bytes",
        "Size of pgvector indexes in bytes",
        ["index_name"],
    )

    # Search recall gauge
    PGVECTOR_SEARCH_RECALL = Gauge(
        "pgvector_search_recall",
        "Estimated search recall rate",
        ["namespace"],
    )

    # Connection pool metrics
    PGVECTOR_POOL_SIZE = Gauge(
        "pgvector_pool_size",
        "Current connection pool size",
        ["pool_type"],  # writer, reader
    )

    PGVECTOR_POOL_AVAILABLE = Gauge(
        "pgvector_pool_available",
        "Available connections in pool",
        ["pool_type"],
    )

    # Job metrics
    PGVECTOR_JOBS_TOTAL = Counter(
        "pgvector_jobs_total",
        "Total number of embedding jobs",
        ["status"],
    )

    PGVECTOR_JOBS_DURATION = Histogram(
        "pgvector_jobs_duration_seconds",
        "Embedding job duration in seconds",
        buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
    )

    # Database info
    PGVECTOR_INFO = Info(
        "pgvector",
        "pgvector extension information",
    )


# ============================================================================
# Metric Recording Functions
# ============================================================================


def record_search_latency(
    latency_ms: float, search_type: str, namespace: str
) -> None:
    """Record a search operation's latency."""
    if not PROMETHEUS_AVAILABLE:
        return
    PGVECTOR_SEARCH_LATENCY.labels(
        search_type=search_type, namespace=namespace
    ).observe(latency_ms)
    PGVECTOR_SEARCH_TOTAL.labels(
        search_type=search_type, namespace=namespace
    ).inc()


def record_insert(
    count: int, namespace: str, status: str = "success"
) -> None:
    """Record embedding insert operations."""
    if not PROMETHEUS_AVAILABLE:
        return
    PGVECTOR_INSERT_TOTAL.labels(namespace=namespace, status=status).inc(count)


def record_insert_latency(latency_ms: float, namespace: str) -> None:
    """Record batch insert latency."""
    if not PROMETHEUS_AVAILABLE:
        return
    PGVECTOR_INSERT_LATENCY.labels(namespace=namespace).observe(latency_ms)


def update_embedding_counts(namespace_counts: dict) -> None:
    """Update embedding count gauges per namespace."""
    if not PROMETHEUS_AVAILABLE:
        return
    for namespace, count in namespace_counts.items():
        PGVECTOR_EMBEDDINGS_TOTAL.labels(namespace=namespace).set(count)


def update_index_sizes(indexes: list) -> None:
    """Update index size gauges."""
    if not PROMETHEUS_AVAILABLE:
        return
    for idx in indexes:
        PGVECTOR_INDEX_SIZE_BYTES.labels(
            index_name=idx.get("name", "unknown")
        ).set(idx.get("size_bytes", 0))


def update_pool_metrics(
    writer_size: int,
    writer_available: int,
    reader_size: int = 0,
    reader_available: int = 0,
) -> None:
    """Update connection pool metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    PGVECTOR_POOL_SIZE.labels(pool_type="writer").set(writer_size)
    PGVECTOR_POOL_AVAILABLE.labels(pool_type="writer").set(writer_available)
    PGVECTOR_POOL_SIZE.labels(pool_type="reader").set(reader_size)
    PGVECTOR_POOL_AVAILABLE.labels(pool_type="reader").set(reader_available)


def set_pgvector_info(version: str, dimensions: str = "384") -> None:
    """Set pgvector extension info."""
    if not PROMETHEUS_AVAILABLE:
        return
    PGVECTOR_INFO.info({
        "version": version,
        "default_dimensions": dimensions,
        "index_type": "hnsw",
    })


# ============================================================================
# Periodic Metrics Collector
# ============================================================================


class PgvectorMetricsCollector:
    """
    Periodic metrics collector for pgvector.

    Runs on a configurable interval to update Prometheus gauges
    with current database state (embedding counts, index sizes, etc).
    """

    def __init__(self, db_connection, interval_seconds: int = 60):
        self.db = db_connection
        self.interval = interval_seconds
        self._running = False

    async def collect_once(self) -> dict:
        """Collect all metrics once."""
        metrics = {}

        try:
            # Embedding counts per namespace
            rows = await self.db.execute(
                """
                SELECT namespace, COUNT(*) as count
                FROM vector_embeddings
                GROUP BY namespace
                """,
                use_reader=True,
            )
            namespace_counts = {r["namespace"]: r["count"] for r in rows}
            update_embedding_counts(namespace_counts)
            metrics["namespace_counts"] = namespace_counts

            # Index sizes
            idx_rows = await self.db.execute(
                """
                SELECT
                    indexname as name,
                    pg_relation_size(indexname::regclass) as size_bytes
                FROM pg_indexes
                WHERE tablename = 'vector_embeddings'
                """,
                use_reader=True,
            )
            indexes = [dict(r) for r in idx_rows]
            update_index_sizes(indexes)
            metrics["indexes"] = indexes

            # pgvector version
            version_row = await self.db.execute_one(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
                use_reader=True,
            )
            if version_row:
                set_pgvector_info(version_row["extversion"])
                metrics["pgvector_version"] = version_row["extversion"]

        except Exception as e:
            logger.error("Metrics collection failed: %s", e)
            metrics["error"] = str(e)

        return metrics

    async def start(self) -> None:
        """Start periodic metrics collection."""
        import asyncio

        self._running = True
        logger.info(
            "Starting pgvector metrics collector (interval=%ds)", self.interval
        )
        while self._running:
            await self.collect_once()
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        """Stop periodic metrics collection."""
        self._running = False
        logger.info("Stopped pgvector metrics collector")
