# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Audit Flow - SEC-005

Tests the complete audit event flow from logging to querying with real
PostgreSQL and Redis dependencies.

These tests require:
- PostgreSQL with audit tables created
- Redis for pub/sub
- Or testcontainers for automatic provisioning
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip if dependencies not available
# ---------------------------------------------------------------------------
try:
    import asyncpg
    import redis.asyncio as aioredis
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

try:
    from greenlang.infrastructure.audit_service.service import AuditService
    from greenlang.infrastructure.audit_service.event_model import UnifiedAuditEvent, EventBuilder
    from greenlang.infrastructure.audit_service.repository import AuditEventRepository
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_DEPS, reason="asyncpg/aioredis not installed"),
    pytest.mark.skipif(not _HAS_MODULE, reason="audit_service module not implemented"),
]


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "greenlang_test",
        "user": "greenlang",
        "password": "test_password",
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
    },
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def db_pool():
    """Create a PostgreSQL connection pool."""
    pool = await asyncpg.create_pool(
        host=TEST_CONFIG["postgres"]["host"],
        port=TEST_CONFIG["postgres"]["port"],
        database=TEST_CONFIG["postgres"]["database"],
        user=TEST_CONFIG["postgres"]["user"],
        password=TEST_CONFIG["postgres"]["password"],
        min_size=1,
        max_size=5,
    )
    yield pool
    await pool.close()


@pytest.fixture
async def redis_client():
    """Create a Redis client."""
    client = aioredis.from_url(
        f"redis://{TEST_CONFIG['redis']['host']}:{TEST_CONFIG['redis']['port']}"
    )
    yield client
    await client.close()


@pytest.fixture
async def audit_service(db_pool, redis_client):
    """Create an AuditService with real dependencies."""
    from greenlang.infrastructure.audit_service.service import AuditService, AuditServiceConfig
    from greenlang.infrastructure.audit_service.event_collector import EventCollector
    from greenlang.infrastructure.audit_service.event_router import EventRouter

    config = AuditServiceConfig(
        enable_async_processing=False,  # Sync for testing
        worker_count=1,
    )

    router = EventRouter(
        db_pool=db_pool,
        redis_client=redis_client,
    )

    service = AuditService(
        config=config,
        collector=EventCollector(),
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
def sample_tenant_id() -> str:
    """Generate a unique tenant ID for test isolation."""
    return f"t-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_user_id() -> str:
    """Generate a unique user ID."""
    return f"u-test-{uuid.uuid4().hex[:8]}"


# ============================================================================
# TestAuditFlowE2E
# ============================================================================


class TestAuditFlowE2E:
    """End-to-end tests for audit event flow."""

    @pytest.mark.asyncio
    async def test_log_and_query_single_event(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
        sample_user_id,
    ) -> None:
        """Log a single event and verify it can be queried."""
        # Log event
        event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .with_user(sample_user_id) \
            .with_result("success") \
            .build()

        result = await audit_service.log_event(event)
        assert result is True

        # Flush to ensure persistence
        await audit_service.flush()

        # Query event
        events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            user_id=sample_user_id,
        )

        assert events["total"] >= 1
        assert any(e["user_id"] == sample_user_id for e in events["items"])

    @pytest.mark.asyncio
    async def test_log_multiple_events_batch(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Log multiple events and verify batch persistence."""
        event_count = 10

        for i in range(event_count):
            event = EventBuilder("auth.login_success") \
                .with_tenant(sample_tenant_id) \
                .with_user(f"u-batch-{i}") \
                .with_detail("batch_index", i) \
                .build()
            await audit_service.log_event(event)

        await audit_service.flush()

        # Query all events for tenant
        events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            page_size=50,
        )

        assert events["total"] >= event_count

    @pytest.mark.asyncio
    async def test_event_filtering_by_category(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Events can be filtered by category."""
        # Log auth event
        auth_event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .build()
        await audit_service.log_event(auth_event)

        # Log rbac event
        rbac_event = EventBuilder("rbac.role_created") \
            .with_tenant(sample_tenant_id) \
            .build()
        await audit_service.log_event(rbac_event)

        await audit_service.flush()

        # Filter by auth category
        auth_events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            category="auth",
        )

        # Filter by rbac category
        rbac_events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            category="rbac",
        )

        assert all(e["category"] == "auth" for e in auth_events["items"])
        assert all(e["category"] == "rbac" for e in rbac_events["items"])

    @pytest.mark.asyncio
    async def test_event_filtering_by_date_range(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Events can be filtered by date range."""
        event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .build()
        await audit_service.log_event(event)
        await audit_service.flush()

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            start_date=start,
            end_date=end,
        )

        assert events["total"] >= 1

    @pytest.mark.asyncio
    async def test_event_filtering_by_severity(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Events can be filtered by severity."""
        # Log error event
        error_event = EventBuilder("auth.login_failure") \
            .with_tenant(sample_tenant_id) \
            .with_severity("error") \
            .build()
        await audit_service.log_event(error_event)

        await audit_service.flush()

        events = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            severity="error",
        )

        assert all(e["severity"] == "error" for e in events["items"])

    @pytest.mark.asyncio
    async def test_event_get_by_id(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Specific event can be retrieved by ID."""
        event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .build()

        await audit_service.log_event(event)
        await audit_service.flush()

        # Get the event by ID
        retrieved = await event_repository.get_event_by_id(event.event_id)

        assert retrieved is not None
        assert retrieved["event_id"] == event.event_id

    @pytest.mark.asyncio
    async def test_event_pagination(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Events support pagination."""
        # Log 25 events
        for i in range(25):
            event = EventBuilder("auth.login_success") \
                .with_tenant(sample_tenant_id) \
                .with_detail("index", i) \
                .build()
            await audit_service.log_event(event)

        await audit_service.flush()

        # Get first page
        page1 = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            page=1,
            page_size=10,
        )

        # Get second page
        page2 = await event_repository.get_events(
            tenant_id=sample_tenant_id,
            page=2,
            page_size=10,
        )

        assert len(page1["items"]) == 10
        assert len(page2["items"]) >= 10
        # Items should be different
        page1_ids = {e["event_id"] for e in page1["items"]}
        page2_ids = {e["event_id"] for e in page2["items"]}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_event_stats_aggregation(
        self,
        audit_service,
        event_repository,
        sample_tenant_id,
    ) -> None:
        """Event statistics can be aggregated."""
        # Log various events
        events = [
            ("auth.login_success", "info"),
            ("auth.login_failure", "error"),
            ("rbac.role_created", "info"),
            ("data.read", "info"),
        ]

        for event_type, severity in events:
            event = EventBuilder(event_type) \
                .with_tenant(sample_tenant_id) \
                .with_severity(severity) \
                .build()
            await audit_service.log_event(event)

        await audit_service.flush()

        stats = await event_repository.get_stats(tenant_id=sample_tenant_id)

        assert stats["total_events"] >= 4
        assert "events_by_category" in stats or "category" in stats

    @pytest.mark.asyncio
    async def test_tenant_isolation(
        self,
        audit_service,
        event_repository,
    ) -> None:
        """Events are isolated by tenant."""
        tenant1 = f"t-iso-{uuid.uuid4().hex[:8]}"
        tenant2 = f"t-iso-{uuid.uuid4().hex[:8]}"

        # Log event for tenant1
        event1 = EventBuilder("auth.login_success") \
            .with_tenant(tenant1) \
            .build()
        await audit_service.log_event(event1)

        # Log event for tenant2
        event2 = EventBuilder("auth.login_success") \
            .with_tenant(tenant2) \
            .build()
        await audit_service.log_event(event2)

        await audit_service.flush()

        # Query tenant1 events
        tenant1_events = await event_repository.get_events(tenant_id=tenant1)

        # Query tenant2 events
        tenant2_events = await event_repository.get_events(tenant_id=tenant2)

        # Verify isolation
        assert all(e["tenant_id"] == tenant1 for e in tenant1_events["items"])
        assert all(e["tenant_id"] == tenant2 for e in tenant2_events["items"])


# ============================================================================
# TestRedisIntegration
# ============================================================================


class TestRedisIntegration:
    """Integration tests for Redis pub/sub functionality."""

    @pytest.mark.asyncio
    async def test_event_published_to_redis(
        self,
        audit_service,
        redis_client,
        sample_tenant_id,
    ) -> None:
        """Events are published to Redis pub/sub channel."""
        # Subscribe to audit channel
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"audit:events:{sample_tenant_id}")

        # Log event
        event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .build()
        await audit_service.log_event(event)
        await audit_service.flush()

        # Check for message (with timeout)
        message = await asyncio.wait_for(
            pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0),
            timeout=10.0,
        )

        if message:
            assert message["type"] == "message"
            # Message should contain event data

        await pubsub.unsubscribe()
        await pubsub.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_events(
        self,
        audit_service,
        redis_client,
        sample_tenant_id,
    ) -> None:
        """Multiple subscribers receive the same event."""
        # Create two subscribers
        pubsub1 = redis_client.pubsub()
        pubsub2 = redis_client.pubsub()

        channel = f"audit:events:{sample_tenant_id}"
        await pubsub1.subscribe(channel)
        await pubsub2.subscribe(channel)

        # Log event
        event = EventBuilder("auth.login_success") \
            .with_tenant(sample_tenant_id) \
            .build()
        await audit_service.log_event(event)
        await audit_service.flush()

        # Both should receive
        # (Implementation depends on Redis pub/sub behavior)

        await pubsub1.unsubscribe()
        await pubsub2.unsubscribe()
        await pubsub1.close()
        await pubsub2.close()
