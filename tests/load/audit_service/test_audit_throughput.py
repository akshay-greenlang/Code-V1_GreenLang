# -*- coding: utf-8 -*-
"""
Audit Service Throughput Load Tests - SEC-005

Tests high-throughput event ingestion, query performance, and export
capabilities under load.

Performance targets:
- Ingestion: 10,000+ events/second
- Query latency: P95 < 100ms
- Export: 1M+ events exportable
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Skip if dependencies not available
# ---------------------------------------------------------------------------
try:
    import asyncpg
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

try:
    from greenlang.infrastructure.audit_service.service import AuditService
    from greenlang.infrastructure.audit_service.event_model import EventBuilder
    from greenlang.infrastructure.audit_service.repository import AuditEventRepository
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False


pytestmark = [
    pytest.mark.load,
    pytest.mark.performance,
    pytest.mark.skipif(not _HAS_DEPS, reason="asyncpg not installed"),
    pytest.mark.skipif(not _HAS_MODULE, reason="audit_service module not implemented"),
]


# ============================================================================
# Test Configuration
# ============================================================================

LOAD_CONFIG = {
    "ingestion": {
        "target_events_per_second": 10000,
        "test_duration_seconds": 10,
        "batch_size": 100,
    },
    "query": {
        "target_p95_latency_ms": 100,
        "concurrent_queries": 50,
        "queries_per_client": 100,
    },
    "export": {
        "target_events": 100000,
        "max_duration_seconds": 60,
    },
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "greenlang_test",
        "user": "greenlang",
        "password": "test_password",
    },
}


# ============================================================================
# Helpers
# ============================================================================


def _create_test_event(
    tenant_id: str,
    index: int,
) -> Any:
    """Create a test event for load testing."""
    return EventBuilder(f"load.test_event_{index % 10}") \
        .with_tenant(tenant_id) \
        .with_user(f"u-load-{index % 100}") \
        .with_severity("info") \
        .with_detail("index", index) \
        .with_detail("timestamp", datetime.now(timezone.utc).isoformat()) \
        .build()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def db_pool():
    """Create a PostgreSQL connection pool optimized for load testing."""
    pool = await asyncpg.create_pool(
        host=LOAD_CONFIG["postgres"]["host"],
        port=LOAD_CONFIG["postgres"]["port"],
        database=LOAD_CONFIG["postgres"]["database"],
        user=LOAD_CONFIG["postgres"]["user"],
        password=LOAD_CONFIG["postgres"]["password"],
        min_size=10,
        max_size=50,
    )
    yield pool
    await pool.close()


@pytest.fixture
async def audit_service(db_pool):
    """Create an AuditService configured for load testing."""
    from greenlang.infrastructure.audit_service.service import AuditService, AuditServiceConfig
    from greenlang.infrastructure.audit_service.event_collector import EventCollector, CollectorConfig
    from greenlang.infrastructure.audit_service.event_router import EventRouter, RouterConfig

    collector_config = CollectorConfig(
        max_queue_size=100000,
        batch_size=LOAD_CONFIG["ingestion"]["batch_size"],
    )

    router_config = RouterConfig(
        database_batch_size=LOAD_CONFIG["ingestion"]["batch_size"],
        enable_loki=False,  # Disable for pure DB load test
        enable_redis_pubsub=False,
    )

    service_config = AuditServiceConfig(
        enable_async_processing=True,
        worker_count=4,
        flush_interval_ms=100,
    )

    router = EventRouter(
        config=router_config,
        db_pool=db_pool,
    )

    service = AuditService(
        config=service_config,
        collector=EventCollector(config=collector_config),
        router=router,
    )

    await service.start()
    yield service
    await service.stop()


@pytest.fixture
async def event_repository(db_pool):
    """Create an AuditEventRepository."""
    return AuditEventRepository(db_pool=db_pool)


@pytest.fixture
def load_tenant_id() -> str:
    """Generate a unique tenant ID for load test isolation."""
    return f"t-load-{uuid.uuid4().hex[:8]}"


# ============================================================================
# TestIngestionThroughput
# ============================================================================


class TestIngestionThroughput:
    """Load tests for event ingestion throughput."""

    @pytest.mark.asyncio
    async def test_sustained_ingestion_rate(
        self,
        audit_service,
        load_tenant_id,
    ) -> None:
        """Test sustained event ingestion at target rate."""
        config = LOAD_CONFIG["ingestion"]
        target_eps = config["target_events_per_second"]
        duration = config["test_duration_seconds"]
        total_events = target_eps * duration

        start_time = time.monotonic()
        events_logged = 0

        # Log events at target rate
        for i in range(total_events):
            event = _create_test_event(load_tenant_id, i)
            await audit_service.log_event(event)
            events_logged += 1

            # Rate limiting to avoid overwhelming
            if events_logged % 1000 == 0:
                elapsed = time.monotonic() - start_time
                expected_elapsed = events_logged / target_eps
                if elapsed < expected_elapsed:
                    await asyncio.sleep(expected_elapsed - elapsed)

        end_time = time.monotonic()
        actual_duration = end_time - start_time
        actual_eps = events_logged / actual_duration

        # Flush remaining events
        await audit_service.flush()

        print(f"\n=== Ingestion Results ===")
        print(f"Events logged: {events_logged}")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Actual EPS: {actual_eps:.0f}")
        print(f"Target EPS: {target_eps}")

        # Should achieve at least 80% of target
        assert actual_eps >= target_eps * 0.8, \
            f"Ingestion rate {actual_eps:.0f} EPS below 80% of target {target_eps}"

    @pytest.mark.asyncio
    async def test_batch_ingestion_throughput(
        self,
        audit_service,
        load_tenant_id,
    ) -> None:
        """Test batch event ingestion throughput."""
        batch_size = 1000
        num_batches = 100
        total_events = batch_size * num_batches

        latencies = []
        start_time = time.monotonic()

        for batch_num in range(num_batches):
            batch_start = time.monotonic()

            events = [
                _create_test_event(load_tenant_id, batch_num * batch_size + i)
                for i in range(batch_size)
            ]

            for event in events:
                await audit_service.log_event(event)

            batch_end = time.monotonic()
            latencies.append((batch_end - batch_start) * 1000)  # ms

        end_time = time.monotonic()
        total_duration = end_time - start_time

        await audit_service.flush()

        throughput = total_events / total_duration
        avg_batch_latency = statistics.mean(latencies)
        p95_batch_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n=== Batch Ingestion Results ===")
        print(f"Total events: {total_events}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Throughput: {throughput:.0f} events/s")
        print(f"Avg batch latency: {avg_batch_latency:.2f}ms")
        print(f"P95 batch latency: {p95_batch_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_producers(
        self,
        audit_service,
        load_tenant_id,
    ) -> None:
        """Test concurrent event producers."""
        num_producers = 10
        events_per_producer = 1000

        async def producer(producer_id: int):
            events_logged = 0
            for i in range(events_per_producer):
                event = _create_test_event(
                    load_tenant_id,
                    producer_id * events_per_producer + i,
                )
                await audit_service.log_event(event)
                events_logged += 1
            return events_logged

        start_time = time.monotonic()

        results = await asyncio.gather(
            *[producer(i) for i in range(num_producers)]
        )

        end_time = time.monotonic()
        total_events = sum(results)
        duration = end_time - start_time

        await audit_service.flush()

        print(f"\n=== Concurrent Producers Results ===")
        print(f"Producers: {num_producers}")
        print(f"Total events: {total_events}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {total_events / duration:.0f} events/s")


# ============================================================================
# TestQueryPerformance
# ============================================================================


class TestQueryPerformance:
    """Load tests for query performance."""

    @pytest.mark.asyncio
    async def test_simple_query_latency(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test simple query latency under load."""
        num_queries = 100
        latencies = []

        for _ in range(num_queries):
            start = time.monotonic()
            await event_repository.get_events(
                tenant_id=load_tenant_id,
                page=1,
                page_size=20,
            )
            end = time.monotonic()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\n=== Query Latency Results ===")
        print(f"Queries: {num_queries}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")

        target = LOAD_CONFIG["query"]["target_p95_latency_ms"]
        assert p95_latency <= target, \
            f"P95 latency {p95_latency:.2f}ms exceeds target {target}ms"

    @pytest.mark.asyncio
    async def test_filtered_query_performance(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test query performance with filters."""
        num_queries = 50
        latencies = []

        filters = [
            {"category": "auth"},
            {"severity": "error"},
            {"user_id": "u-load-1"},
            {"start_date": datetime.now(timezone.utc) - timedelta(days=7)},
        ]

        for i in range(num_queries):
            filter_combo = filters[i % len(filters)]

            start = time.monotonic()
            await event_repository.get_events(
                tenant_id=load_tenant_id,
                **filter_combo,
                page=1,
                page_size=50,
            )
            end = time.monotonic()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n=== Filtered Query Results ===")
        print(f"Queries: {num_queries}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_queries(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test concurrent query performance."""
        config = LOAD_CONFIG["query"]
        num_clients = config["concurrent_queries"]
        queries_per_client = config["queries_per_client"]

        async def query_client(client_id: int) -> List[float]:
            latencies = []
            for _ in range(queries_per_client):
                start = time.monotonic()
                await event_repository.get_events(
                    tenant_id=load_tenant_id,
                    page=1,
                    page_size=20,
                )
                end = time.monotonic()
                latencies.append((end - start) * 1000)
            return latencies

        start_time = time.monotonic()

        results = await asyncio.gather(
            *[query_client(i) for i in range(num_clients)]
        )

        end_time = time.monotonic()
        total_duration = end_time - start_time

        all_latencies = [lat for client_lats in results for lat in client_lats]
        total_queries = len(all_latencies)
        qps = total_queries / total_duration

        avg_latency = statistics.mean(all_latencies)
        p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)]

        print(f"\n=== Concurrent Query Results ===")
        print(f"Clients: {num_clients}")
        print(f"Total queries: {total_queries}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"QPS: {qps:.0f}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")


# ============================================================================
# TestExportPerformance
# ============================================================================


class TestExportPerformance:
    """Load tests for export performance."""

    @pytest.mark.asyncio
    async def test_large_export_csv(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test exporting large dataset to CSV."""
        from greenlang.infrastructure.audit_service.export import AuditExportService

        config = LOAD_CONFIG["export"]
        target_events = config["target_events"]

        export_service = AuditExportService(
            event_repository=event_repository,
        )

        start_time = time.monotonic()

        job = await export_service.create_export_job(
            format="csv",
            filters={"tenant_id": load_tenant_id},
        )

        completed = await export_service.execute_job(job.job_id)

        end_time = time.monotonic()
        duration = end_time - start_time

        print(f"\n=== CSV Export Results ===")
        print(f"Rows exported: {completed.row_count}")
        print(f"Duration: {duration:.2f}s")
        if completed.row_count > 0:
            print(f"Rate: {completed.row_count / duration:.0f} rows/s")

        max_duration = config["max_duration_seconds"]
        assert duration <= max_duration, \
            f"Export took {duration:.2f}s, exceeds max {max_duration}s"

    @pytest.mark.asyncio
    async def test_large_export_json(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test exporting large dataset to JSON."""
        from greenlang.infrastructure.audit_service.export import AuditExportService

        export_service = AuditExportService(
            event_repository=event_repository,
        )

        start_time = time.monotonic()

        job = await export_service.create_export_job(
            format="json",
            filters={"tenant_id": load_tenant_id},
        )

        completed = await export_service.execute_job(job.job_id)

        end_time = time.monotonic()
        duration = end_time - start_time

        print(f"\n=== JSON Export Results ===")
        print(f"Rows exported: {completed.row_count}")
        print(f"Duration: {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_streaming_export(
        self,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test streaming export for memory efficiency."""
        from greenlang.infrastructure.audit_service.export import AuditExportService

        export_service = AuditExportService(
            event_repository=event_repository,
        )

        chunks_received = 0
        total_rows = 0

        start_time = time.monotonic()

        async for chunk in export_service.stream_export(
            format="csv",
            filters={"tenant_id": load_tenant_id},
            chunk_size=1000,
        ):
            chunks_received += 1
            total_rows += len(chunk) if isinstance(chunk, list) else 1

        end_time = time.monotonic()
        duration = end_time - start_time

        print(f"\n=== Streaming Export Results ===")
        print(f"Chunks received: {chunks_received}")
        print(f"Total rows: {total_rows}")
        print(f"Duration: {duration:.2f}s")


# ============================================================================
# TestSystemStability
# ============================================================================


class TestSystemStability:
    """Tests for system stability under sustained load."""

    @pytest.mark.asyncio
    async def test_sustained_mixed_workload(
        self,
        audit_service,
        event_repository,
        load_tenant_id,
    ) -> None:
        """Test system stability under mixed read/write workload."""
        duration_seconds = 30
        write_rate = 100  # events per second
        read_rate = 10  # queries per second

        errors = []
        writes_completed = 0
        reads_completed = 0

        async def writer():
            nonlocal writes_completed
            while time.monotonic() - start_time < duration_seconds:
                try:
                    event = _create_test_event(load_tenant_id, writes_completed)
                    await audit_service.log_event(event)
                    writes_completed += 1
                    await asyncio.sleep(1 / write_rate)
                except Exception as e:
                    errors.append(f"Write error: {e}")

        async def reader():
            nonlocal reads_completed
            while time.monotonic() - start_time < duration_seconds:
                try:
                    await event_repository.get_events(
                        tenant_id=load_tenant_id,
                        page=1,
                        page_size=20,
                    )
                    reads_completed += 1
                    await asyncio.sleep(1 / read_rate)
                except Exception as e:
                    errors.append(f"Read error: {e}")

        start_time = time.monotonic()

        # Run writers and readers concurrently
        await asyncio.gather(
            writer(),
            writer(),
            reader(),
            reader(),
        )

        await audit_service.flush()

        print(f"\n=== Stability Test Results ===")
        print(f"Duration: {duration_seconds}s")
        print(f"Writes completed: {writes_completed}")
        print(f"Reads completed: {reads_completed}")
        print(f"Errors: {len(errors)}")

        # Should have minimal errors
        error_rate = len(errors) / (writes_completed + reads_completed)
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1%"
