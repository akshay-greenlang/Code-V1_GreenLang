"""
Billing Event Emitter - Agent Factory Metering (INFRA-010)

Emits billing events to a message bus for downstream accounting systems.
Supports batch emission with configurable intervals. Events include
execution completions, budget threshold alerts, quota violations,
and period summaries.

Classes:
    - BillingEventType: Event type enumeration.
    - BillingEvent: Single billing event dataclass.
    - BillingEventEmitter: Event emission service with batching.

Example:
    >>> emitter = BillingEventEmitter(redis_client)
    >>> await emitter.emit(BillingEvent(
    ...     event_type=BillingEventType.EXECUTION_COMPLETE,
    ...     agent_key="intake-agent",
    ...     tenant_id="acme",
    ...     amount_usd=0.0042,
    ... ))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------


class BillingEventType(str, Enum):
    """Types of billing events."""

    EXECUTION_COMPLETE = "execution_complete"
    """An agent execution completed and incurred costs."""

    BUDGET_THRESHOLD = "budget_threshold"
    """A budget warning or limit threshold was reached."""

    QUOTA_EXCEEDED = "quota_exceeded"
    """An execution quota was exceeded."""

    PERIOD_SUMMARY = "period_summary"
    """End-of-period cost summary for accounting."""


# ---------------------------------------------------------------------------
# Billing Event
# ---------------------------------------------------------------------------


@dataclass
class BillingEvent:
    """Single billing event for downstream accounting.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: The type of billing event.
        agent_key: Agent that generated the event.
        tenant_id: Tenant for billing attribution.
        amount_usd: Cost amount in USD (0.0 for non-cost events).
        category: Cost category (compute, tokens, storage, etc.).
        metadata: Additional context for the event.
        timestamp: When the event occurred (UTC).
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: BillingEventType = BillingEventType.EXECUTION_COMPLETE
    agent_key: str = ""
    tenant_id: str = ""
    amount_usd: float = 0.0
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the event to a dictionary.

        Returns:
            JSON-serialisable dictionary.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "agent_key": self.agent_key,
            "tenant_id": self.tenant_id,
            "amount_usd": self.amount_usd,
            "category": self.category,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Serialise the event to a JSON string.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Emitter Metrics
# ---------------------------------------------------------------------------


@dataclass
class EmitterMetrics:
    """Metrics for the billing event emitter.

    Attributes:
        total_emitted: Total events emitted.
        total_batches: Total batch flushes.
        total_errors: Total emission errors.
        last_flush_time: Timestamp of last successful flush.
    """

    total_emitted: int = 0
    total_batches: int = 0
    total_errors: int = 0
    last_flush_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Billing Event Emitter
# ---------------------------------------------------------------------------


class BillingEventEmitter:
    """Emits billing events to a Redis pub/sub channel.

    Accumulates events in a buffer and flushes them in batches at
    configurable intervals for efficient emission. Supports both
    pub/sub and list-based (reliable queue) delivery.

    Attributes:
        channel: Redis pub/sub channel for event delivery.
        flush_interval_s: Seconds between batch flushes.
        metrics: Observable metrics.
    """

    _DEFAULT_CHANNEL = "gl:billing:events"

    def __init__(
        self,
        redis_client: Any,
        channel: Optional[str] = None,
        flush_interval_s: float = 5.0,
        use_reliable_queue: bool = True,
    ) -> None:
        """Initialize the billing event emitter.

        Args:
            redis_client: Async Redis client (redis.asyncio).
            channel: Redis channel/key for event delivery.
            flush_interval_s: Seconds between batch flushes.
            use_reliable_queue: If True, use RPUSH (reliable queue).
                If False, use PUBLISH (pub/sub, fire-and-forget).
        """
        self._redis = redis_client
        self.channel = channel or self._DEFAULT_CHANNEL
        self.flush_interval_s = flush_interval_s
        self._use_reliable_queue = use_reliable_queue
        self._buffer: List[BillingEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self.metrics = EmitterMetrics()
        self._on_emit: Optional[Callable[[List[BillingEvent]], Any]] = None

        logger.info(
            "BillingEventEmitter initialised (channel=%s, flush=%.1fs, reliable=%s)",
            self.channel, flush_interval_s, use_reliable_queue,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def on_emit(
        self,
        callback: Callable[[List[BillingEvent]], Any],
    ) -> BillingEventEmitter:
        """Register a callback invoked after each batch flush.

        Args:
            callback: Function or coroutine receiving the flushed events.

        Returns:
            Self for fluent chaining.
        """
        self._on_emit = callback
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("BillingEventEmitter: background flush started")

    async def stop(self) -> None:
        """Stop the background flush task and emit remaining events."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()
        logger.info("BillingEventEmitter: stopped and flushed")

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    async def emit(self, event: BillingEvent) -> None:
        """Buffer a single billing event for batch emission.

        Args:
            event: The billing event to emit.
        """
        async with self._lock:
            self._buffer.append(event)
        logger.debug(
            "BillingEventEmitter: buffered event %s (type=%s, agent=%s)",
            event.event_id, event.event_type.value, event.agent_key,
        )

    async def emit_immediate(self, event: BillingEvent) -> None:
        """Emit a single event immediately, bypassing the buffer.

        Use this for high-priority events that cannot wait for the
        next flush cycle (e.g., budget exceeded alerts).

        Args:
            event: The billing event to emit immediately.
        """
        try:
            if self._use_reliable_queue:
                await self._redis.rpush(self.channel, event.to_json())
            else:
                await self._redis.publish(self.channel, event.to_json())
            self.metrics.total_emitted += 1
            logger.info(
                "BillingEventEmitter: immediate emit %s (type=%s)",
                event.event_id, event.event_type.value,
            )
        except Exception as exc:
            self.metrics.total_errors += 1
            logger.error(
                "BillingEventEmitter: immediate emit failed: %s", exc,
            )
            raise

    async def emit_batch(self, events: List[BillingEvent]) -> None:
        """Buffer multiple events for batch emission.

        Args:
            events: List of billing events to emit.
        """
        if not events:
            return
        async with self._lock:
            self._buffer.extend(events)

    # ------------------------------------------------------------------
    # Convenience Factories
    # ------------------------------------------------------------------

    def create_execution_event(
        self,
        agent_key: str,
        tenant_id: str,
        amount_usd: float,
        category: str = "compute",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BillingEvent:
        """Create a standard EXECUTION_COMPLETE event.

        Args:
            agent_key: Agent that executed.
            tenant_id: Tenant identifier.
            amount_usd: Execution cost.
            category: Cost category.
            metadata: Additional context.

        Returns:
            BillingEvent ready for emission.
        """
        return BillingEvent(
            event_type=BillingEventType.EXECUTION_COMPLETE,
            agent_key=agent_key,
            tenant_id=tenant_id,
            amount_usd=amount_usd,
            category=category,
            metadata=metadata or {},
        )

    def create_budget_alert_event(
        self,
        agent_key: str,
        tenant_id: str,
        spent_usd: float,
        budget_usd: float,
        utilisation_pct: float,
    ) -> BillingEvent:
        """Create a BUDGET_THRESHOLD event.

        Args:
            agent_key: Agent key.
            tenant_id: Tenant identifier.
            spent_usd: Current spend.
            budget_usd: Budget limit.
            utilisation_pct: Budget utilisation percentage.

        Returns:
            BillingEvent ready for emission.
        """
        return BillingEvent(
            event_type=BillingEventType.BUDGET_THRESHOLD,
            agent_key=agent_key,
            tenant_id=tenant_id,
            amount_usd=spent_usd,
            metadata={
                "budget_usd": budget_usd,
                "utilisation_pct": utilisation_pct,
            },
        )

    # ------------------------------------------------------------------
    # Background Flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Background task that flushes the buffer periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_s)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("BillingEventEmitter: flush loop error: %s", exc)

    async def _flush(self) -> None:
        """Flush all buffered events to the message bus."""
        async with self._lock:
            if not self._buffer:
                return
            events = list(self._buffer)
            self._buffer.clear()

        if not events:
            return

        try:
            if self._use_reliable_queue:
                # Use pipeline for batch RPUSH
                pipe = self._redis.pipeline()
                for event in events:
                    pipe.rpush(self.channel, event.to_json())
                await pipe.execute()
            else:
                for event in events:
                    await self._redis.publish(self.channel, event.to_json())

            self.metrics.total_emitted += len(events)
            self.metrics.total_batches += 1
            self.metrics.last_flush_time = time.monotonic()

            logger.info(
                "BillingEventEmitter: flushed %d events to '%s'",
                len(events), self.channel,
            )

            # Invoke callback
            if self._on_emit is not None:
                try:
                    result = self._on_emit(events)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.error(
                        "BillingEventEmitter: on_emit callback failed: %s", exc,
                    )

        except Exception as exc:
            self.metrics.total_errors += 1
            logger.error(
                "BillingEventEmitter: flush failed for %d events: %s",
                len(events), exc,
            )
            # Re-buffer on failure
            async with self._lock:
                self._buffer = events + self._buffer

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot.

        Returns:
            Dictionary with configuration and metrics.
        """
        return {
            "channel": self.channel,
            "buffer_size": len(self._buffer),
            "flush_interval_s": self.flush_interval_s,
            "use_reliable_queue": self._use_reliable_queue,
            "metrics": {
                "total_emitted": self.metrics.total_emitted,
                "total_batches": self.metrics.total_batches,
                "total_errors": self.metrics.total_errors,
            },
        }


__all__ = [
    "BillingEvent",
    "BillingEventEmitter",
    "BillingEventType",
]
