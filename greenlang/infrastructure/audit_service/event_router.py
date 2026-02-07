# -*- coding: utf-8 -*-
"""
Audit Event Router - SEC-005: Centralized Audit Logging Service

Routes audit events to multiple destinations concurrently:
- PostgreSQL: Persistent storage for compliance queries
- Loki: Structured JSON logs for real-time monitoring
- Redis: Pub/sub for real-time alerting

**Design Principles:**
- Concurrent writes to all destinations (asyncio.gather)
- Batch inserts for PostgreSQL efficiency
- Fire-and-forget with retry for DB writes
- Structured JSON for Loki pipeline compatibility

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .event_model import UnifiedAuditEvent
from .event_types import AuditSeverity

logger = logging.getLogger(__name__)

# Dedicated logger for Loki-bound audit events
_loki_logger = logging.getLogger("greenlang.audit.unified")


# ---------------------------------------------------------------------------
# Router Configuration
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Configuration for the AuditEventRouter.

    Attributes:
        enable_postgres: Whether to write to PostgreSQL (default True).
        enable_loki: Whether to emit to Loki via logging (default True).
        enable_redis: Whether to publish to Redis pub/sub (default True).
        postgres_batch_size: Batch size for PostgreSQL inserts (default 50).
        postgres_max_retries: Max retry attempts for PostgreSQL (default 3).
        postgres_retry_delay: Base delay between retries in seconds (default 0.5).
        redis_channel: Redis pub/sub channel name (default "gl:audit:events").
        table_name: PostgreSQL table name (default "security.unified_audit_log").
    """

    enable_postgres: bool = True
    enable_loki: bool = True
    enable_redis: bool = True
    postgres_batch_size: int = 50
    postgres_max_retries: int = 3
    postgres_retry_delay: float = 0.5
    redis_channel: str = "gl:audit:events"
    table_name: str = "security.unified_audit_log"


# ---------------------------------------------------------------------------
# Router Metrics
# ---------------------------------------------------------------------------


@dataclass
class RouterMetrics:
    """Metrics for monitoring the event router.

    Attributes:
        events_routed: Total events routed.
        postgres_writes: Successful PostgreSQL writes.
        postgres_failures: Failed PostgreSQL writes.
        loki_writes: Successful Loki emissions.
        loki_failures: Failed Loki emissions.
        redis_publishes: Successful Redis publishes.
        redis_failures: Failed Redis publishes.
    """

    events_routed: int = 0
    postgres_writes: int = 0
    postgres_failures: int = 0
    loki_writes: int = 0
    loki_failures: int = 0
    redis_publishes: int = 0
    redis_failures: int = 0


# ---------------------------------------------------------------------------
# Audit Event Router
# ---------------------------------------------------------------------------


class AuditEventRouter:
    """Routes audit events to PostgreSQL, Loki, and Redis concurrently.

    Provides fire-and-forget routing with retry support for database writes.
    All destinations are written concurrently using asyncio.gather.

    Example:
        >>> router = AuditEventRouter(db_pool=pool, redis_client=redis)
        >>> await router.route(event)
        >>> await router.route_batch([event1, event2])
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        """Initialize the event router.

        Args:
            db_pool: Async PostgreSQL connection pool (psycopg_pool).
            redis_client: Async Redis client (redis.asyncio).
            config: Router configuration.
        """
        self._pool = db_pool
        self._redis = redis_client
        self._config = config or RouterConfig()
        self._metrics = RouterMetrics()

    @property
    def metrics(self) -> RouterMetrics:
        """Get current router metrics.

        Returns:
            Current metrics snapshot.
        """
        return self._metrics

    async def route(self, event: UnifiedAuditEvent) -> None:
        """Route a single audit event to all enabled destinations.

        Writes to PostgreSQL, Loki, and Redis concurrently.
        Failures in one destination do not affect others.

        Args:
            event: The audit event to route.
        """
        await self.route_batch([event])

    async def route_batch(self, events: List[UnifiedAuditEvent]) -> None:
        """Route a batch of audit events to all enabled destinations.

        Writes to PostgreSQL, Loki, and Redis concurrently.
        Failures in one destination do not affect others.

        Args:
            events: List of audit events to route.
        """
        if not events:
            return

        self._metrics.events_routed += len(events)

        # Build list of concurrent tasks
        tasks = []

        if self._config.enable_postgres and self._pool:
            tasks.append(self._write_to_postgres(events))

        if self._config.enable_loki:
            tasks.append(self._emit_to_loki(events))

        if self._config.enable_redis and self._redis:
            tasks.append(self._publish_to_redis(events))

        # Execute all tasks concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -------------------------------------------------------------------------
    # PostgreSQL Writer
    # -------------------------------------------------------------------------

    async def _write_to_postgres(self, events: List[UnifiedAuditEvent]) -> None:
        """Write events to PostgreSQL in batches with retry.

        Args:
            events: List of audit events to write.
        """
        for i in range(0, len(events), self._config.postgres_batch_size):
            batch = events[i : i + self._config.postgres_batch_size]
            await self._write_batch_with_retry(batch)

    async def _write_batch_with_retry(
        self, events: List[UnifiedAuditEvent]
    ) -> None:
        """Write a batch to PostgreSQL with exponential backoff retry.

        Args:
            events: Batch of events to write.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._config.postgres_max_retries):
            try:
                await self._insert_batch(events)
                self._metrics.postgres_writes += len(events)
                return
            except Exception as e:
                last_error = e
                delay = self._config.postgres_retry_delay * (2**attempt)
                logger.warning(
                    "PostgreSQL write failed (attempt %d/%d): %s, "
                    "retrying in %.2fs",
                    attempt + 1,
                    self._config.postgres_max_retries,
                    str(e),
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self._metrics.postgres_failures += len(events)
        logger.error(
            "PostgreSQL write failed after %d retries: %s",
            self._config.postgres_max_retries,
            last_error,
        )

    async def _insert_batch(self, events: List[UnifiedAuditEvent]) -> None:
        """Execute batch insert into PostgreSQL.

        Args:
            events: Batch of events to insert.
        """
        if not self._pool:
            return

        # Build insert query with parameterized values
        insert_sql = f"""
            INSERT INTO {self._config.table_name} (
                event_id, correlation_id, event_type, category, severity,
                user_id, tenant_id, session_id, client_ip, user_agent,
                geo_location, resource_type, resource_id, resource_name,
                action, result, error_message, request_path, request_method,
                response_status, duration_ms, metadata, tags,
                occurred_at, recorded_at
            ) VALUES (
                %(event_id)s, %(correlation_id)s, %(event_type)s,
                %(category)s, %(severity)s, %(user_id)s, %(tenant_id)s,
                %(session_id)s, %(client_ip)s, %(user_agent)s,
                %(geo_location)s, %(resource_type)s, %(resource_id)s,
                %(resource_name)s, %(action)s, %(result)s, %(error_message)s,
                %(request_path)s, %(request_method)s, %(response_status)s,
                %(duration_ms)s, %(metadata)s, %(tags)s,
                %(occurred_at)s, %(recorded_at)s
            )
        """

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for event in events:
                    params = self._event_to_params(event)
                    await cur.execute(insert_sql, params)
            await conn.commit()

    def _event_to_params(self, event: UnifiedAuditEvent) -> Dict[str, Any]:
        """Convert event to PostgreSQL parameter dictionary.

        Args:
            event: The audit event.

        Returns:
            Dictionary of parameter values.
        """
        return {
            "event_id": event.event_id,
            "correlation_id": event.correlation_id,
            "event_type": event.event_type.value,
            "category": event.category.value if event.category else None,
            "severity": event.severity.value if event.severity else None,
            "user_id": event.user_id,
            "tenant_id": event.tenant_id,
            "session_id": event.session_id,
            "client_ip": event.client_ip,
            "user_agent": event.user_agent,
            "geo_location": (
                json.dumps(event.geo_location) if event.geo_location else None
            ),
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "resource_name": event.resource_name,
            "action": event.action.value if event.action else None,
            "result": event.result.value,
            "error_message": event.error_message,
            "request_path": event.request_path,
            "request_method": event.request_method,
            "response_status": event.response_status,
            "duration_ms": event.duration_ms,
            "metadata": json.dumps(event.metadata, default=str),
            "tags": event.tags,
            "occurred_at": event.occurred_at,
            "recorded_at": event.recorded_at or datetime.now(timezone.utc),
        }

    # -------------------------------------------------------------------------
    # Loki Emitter
    # -------------------------------------------------------------------------

    async def _emit_to_loki(self, events: List[UnifiedAuditEvent]) -> None:
        """Emit events to Loki via structured JSON logging.

        Uses a dedicated logger that Alloy/Promtail pipelines can match on.

        Args:
            events: List of audit events to emit.
        """
        try:
            for event in events:
                self._emit_single_to_loki(event)
            self._metrics.loki_writes += len(events)
        except Exception as e:
            self._metrics.loki_failures += len(events)
            logger.error("Loki emission failed: %s", e)

    def _emit_single_to_loki(self, event: UnifiedAuditEvent) -> None:
        """Emit a single event to Loki.

        Args:
            event: The audit event to emit.
        """
        # Get log level from severity
        level_map = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        level = level_map.get(event.severity or AuditSeverity.INFO, logging.INFO)

        # Emit as JSON with extra fields for Loki labels
        _loki_logger.log(
            level,
            event.to_json(),
            extra={
                "event_type": event.event_type.value,
                "category": event.category.value if event.category else "",
                "tenant_id": event.tenant_id or "",
                "user_id": event.user_id or "",
                "result": event.result.value,
            },
        )

    # -------------------------------------------------------------------------
    # Redis Publisher
    # -------------------------------------------------------------------------

    async def _publish_to_redis(self, events: List[UnifiedAuditEvent]) -> None:
        """Publish events to Redis pub/sub channel.

        Enables real-time alerting and event streaming.

        Args:
            events: List of audit events to publish.
        """
        if not self._redis:
            return

        try:
            for event in events:
                message = event.to_json()
                await self._redis.publish(self._config.redis_channel, message)
            self._metrics.redis_publishes += len(events)
        except Exception as e:
            self._metrics.redis_failures += len(events)
            logger.warning("Redis publish failed: %s", e)

    # -------------------------------------------------------------------------
    # Single-Destination Methods
    # -------------------------------------------------------------------------

    async def write_to_postgres_only(
        self, events: List[UnifiedAuditEvent]
    ) -> None:
        """Write events to PostgreSQL only.

        Args:
            events: List of audit events to write.
        """
        if self._pool:
            await self._write_to_postgres(events)

    async def emit_to_loki_only(self, events: List[UnifiedAuditEvent]) -> None:
        """Emit events to Loki only.

        Args:
            events: List of audit events to emit.
        """
        await self._emit_to_loki(events)

    async def publish_to_redis_only(
        self, events: List[UnifiedAuditEvent]
    ) -> None:
        """Publish events to Redis only.

        Args:
            events: List of audit events to publish.
        """
        if self._redis:
            await self._publish_to_redis(events)


__all__ = [
    "AuditEventRouter",
    "RouterConfig",
    "RouterMetrics",
]
