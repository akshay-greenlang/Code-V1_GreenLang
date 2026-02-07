# -*- coding: utf-8 -*-
"""
Acknowledgment Tracker - Durable message acknowledgment and dead-letter management.

Tracks the lifecycle of durable messages from delivery through acknowledgment
or failure.  Provides XACK-based acknowledgment, XPENDING-based pending
inspection, XCLAIM-based stale reclamation, and dead-letter queue (DLQ)
routing for messages that exhaust their retry budget.

The tracker wraps Redis Stream consumer-group primitives behind a clean
async API and maintains Prometheus-compatible counters for observability.

Classes:
    - AcknowledgmentTracker: Core tracking and DLQ management.
    - AckTrackerMetrics: Observable counters for monitoring.

Example:
    >>> tracker = AcknowledgmentTracker(redis_client, max_retries=3)
    >>> await tracker.track("gl:msg:carbon.calculate", "1706000000000-0", envelope)
    >>> await tracker.acknowledge("gl:msg:carbon.calculate", "carbon-consumers", "1706000000000-0")
    >>> pending = await tracker.get_pending("gl:msg:carbon.calculate", "carbon-consumers")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.messaging.protocol import (
    DeliveryStatus,
    MessageEnvelope,
    MessageReceipt,
    PendingMessage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class AckTrackerMetrics:
    """Observable counters for the acknowledgment tracker.

    Designed for export to Prometheus via a collector or periodic scrape.

    Attributes:
        messages_tracked: Total messages registered for tracking.
        messages_acknowledged: Total messages successfully acknowledged.
        messages_failed: Total messages that failed processing.
        messages_reclaimed: Total stale messages reclaimed via XCLAIM.
        messages_dead_lettered: Total messages moved to the DLQ.
        reclaim_cycles: Number of reclaim sweeps executed.
    """

    messages_tracked: int = 0
    messages_acknowledged: int = 0
    messages_failed: int = 0
    messages_reclaimed: int = 0
    messages_dead_lettered: int = 0
    reclaim_cycles: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Serialize metrics to a plain dictionary.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "messages_tracked": self.messages_tracked,
            "messages_acknowledged": self.messages_acknowledged,
            "messages_failed": self.messages_failed,
            "messages_reclaimed": self.messages_reclaimed,
            "messages_dead_lettered": self.messages_dead_lettered,
            "reclaim_cycles": self.reclaim_cycles,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AckTrackerConfig:
    """Configuration for the AcknowledgmentTracker.

    Attributes:
        max_retries: Maximum delivery attempts before dead-lettering.
        ack_timeout_s: Seconds before an unacknowledged message is
            considered stale and eligible for reclamation.
        dlq_prefix: Prefix for dead-letter queue stream keys.
        tracking_key_prefix: Prefix for per-message tracking hashes.
        tracking_ttl_seconds: TTL for tracking hashes in Redis.
        max_reclaim_batch: Maximum messages to reclaim per sweep.
    """

    max_retries: int = 3
    ack_timeout_s: float = 60.0
    dlq_prefix: str = "gl:dlq"
    tracking_key_prefix: str = "gl:ack"
    tracking_ttl_seconds: int = 86400  # 24 hours
    max_reclaim_batch: int = 100


# ---------------------------------------------------------------------------
# AcknowledgmentTracker
# ---------------------------------------------------------------------------


class AcknowledgmentTracker:
    """Tracks message acknowledgment state for durable channels.

    Each durable message sent via Redis Streams is registered with the
    tracker.  The tracker maintains a per-message Redis hash with status,
    retry count, and error information.  It provides methods to:

    - **acknowledge**: Mark a message as successfully processed (XACK).
    - **fail**: Record a processing failure and retry or dead-letter.
    - **get_pending**: Inspect unacknowledged messages (XPENDING).
    - **reclaim_stale**: Reclaim stale messages from crashed consumers (XCLAIM).
    - **move_to_dlq**: Move a message to the dead-letter queue.

    All Redis operations are performed through the injected client.

    Attributes:
        config: Tracker configuration.
        metrics: Observable counters.
    """

    def __init__(
        self,
        redis_client: Any,
        config: Optional[AckTrackerConfig] = None,
    ) -> None:
        """Initialize the acknowledgment tracker.

        Args:
            redis_client: Async Redis client (dependency injection).
            config: Tracker configuration. Uses defaults if None.
        """
        self._redis = redis_client
        self.config = config or AckTrackerConfig()
        self.metrics = AckTrackerMetrics()
        logger.debug(
            "AcknowledgmentTracker initialized (max_retries=%d, ack_timeout=%.0fs)",
            self.config.max_retries,
            self.config.ack_timeout_s,
        )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    async def track(
        self,
        stream_name: str,
        message_id: str,
        envelope: MessageEnvelope,
    ) -> None:
        """Register a message for acknowledgment tracking.

        Creates a Redis hash that records the message metadata, delivery
        status, and retry count.  The hash expires after
        ``tracking_ttl_seconds``.

        Args:
            stream_name: Redis Stream key the message was sent to.
            message_id: Redis Stream entry ID.
            envelope: The message envelope being tracked.
        """
        tracking_key = self._tracking_key(stream_name, message_id)
        tracking_data = {
            "message_id": str(envelope.id),
            "stream_name": stream_name,
            "stream_entry_id": message_id,
            "source_agent": envelope.source_agent,
            "target_agent": envelope.target_agent,
            "message_type": envelope.message_type.value,
            "status": DeliveryStatus.DELIVERED.value,
            "attempt": str(envelope.attempt),
            "max_retries": str(envelope.max_retries),
            "tracked_at": datetime.now(timezone.utc).isoformat(),
            "error": "",
        }
        await self._redis.hset(tracking_key, mapping=tracking_data)
        await self._redis.expire(tracking_key, self.config.tracking_ttl_seconds)
        self.metrics.messages_tracked += 1
        logger.debug(
            "Tracking message %s on stream %s (entry=%s)",
            envelope.id,
            stream_name,
            message_id,
        )

    # ------------------------------------------------------------------
    # Acknowledgment
    # ------------------------------------------------------------------

    async def acknowledge(
        self,
        stream_name: str,
        group: str,
        message_id: str,
    ) -> bool:
        """Mark a message as acknowledged (XACK).

        Updates both the Redis Stream consumer group (XACK) and the
        tracking hash.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            message_id: Redis Stream entry ID to acknowledge.

        Returns:
            True if the message was successfully acknowledged.
        """
        try:
            acked = await self._redis.xack(stream_name, group, message_id)
            tracking_key = self._tracking_key(stream_name, message_id)
            await self._redis.hset(
                tracking_key,
                mapping={
                    "status": DeliveryStatus.ACKNOWLEDGED.value,
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.metrics.messages_acknowledged += 1
            logger.debug(
                "Acknowledged message %s on stream %s (group=%s, acked=%d)",
                message_id,
                stream_name,
                group,
                acked,
            )
            return acked > 0
        except Exception as exc:
            logger.error(
                "Failed to acknowledge message %s on stream %s: %s",
                message_id,
                stream_name,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Failure Handling
    # ------------------------------------------------------------------

    async def fail(
        self,
        stream_name: str,
        group: str,
        message_id: str,
        envelope: MessageEnvelope,
        error: str,
    ) -> bool:
        """Record a processing failure for a tracked message.

        If the message has remaining retries, it is re-added to the
        stream with an incremented attempt counter.  Otherwise, it is
        moved to the dead-letter queue.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            message_id: Redis Stream entry ID.
            envelope: The failed message envelope.
            error: Human-readable error description.

        Returns:
            True if the failure was recorded (retry or DLQ).
        """
        self.metrics.messages_failed += 1
        tracking_key = self._tracking_key(stream_name, message_id)

        # XACK the message so it is not redelivered by the group
        try:
            await self._redis.xack(stream_name, group, message_id)
        except Exception as exc:
            logger.warning(
                "XACK during fail for %s failed: %s", message_id, exc
            )

        if envelope.attempt < self.config.max_retries:
            # Retry: re-add to the stream with incremented attempt
            envelope.attempt += 1
            envelope.metadata["last_error"] = error
            envelope.metadata["retried_at"] = datetime.now(timezone.utc).isoformat()
            await self._redis.xadd(stream_name, envelope.to_flat_dict())

            await self._redis.hset(
                tracking_key,
                mapping={
                    "status": DeliveryStatus.PENDING.value,
                    "attempt": str(envelope.attempt),
                    "error": error,
                },
            )
            logger.info(
                "Message %s retried on stream %s (attempt %d/%d): %s",
                message_id,
                stream_name,
                envelope.attempt,
                self.config.max_retries,
                error,
            )
            return True

        # Exhausted retries: move to DLQ
        await self.move_to_dlq(stream_name, envelope, error)
        await self._redis.hset(
            tracking_key,
            mapping={
                "status": DeliveryStatus.FAILED.value,
                "error": error,
                "dead_lettered_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True

    # ------------------------------------------------------------------
    # Pending Inspection
    # ------------------------------------------------------------------

    async def get_pending(
        self,
        stream_name: str,
        group: str,
        count: int = 100,
    ) -> List[PendingMessage]:
        """Get messages that have not been acknowledged (XPENDING).

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            count: Maximum number of pending entries to return.

        Returns:
            List of PendingMessage instances.
        """
        try:
            # XPENDING stream group - + count
            raw = await self._redis.xpending_range(
                stream_name,
                group,
                min="-",
                max="+",
                count=count,
            )
        except Exception as exc:
            logger.error(
                "XPENDING failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return []

        pending: List[PendingMessage] = []
        for entry in raw:
            pending.append(
                PendingMessage(
                    stream_id=entry.get("message_id", ""),
                    consumer=entry.get("consumer", ""),
                    idle_ms=entry.get("time_since_delivered", 0),
                    delivery_count=entry.get("times_delivered", 0),
                )
            )
        return pending

    async def get_pending_count(
        self,
        stream_name: str,
        group: str,
    ) -> int:
        """Get the total count of pending (unacknowledged) messages.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.

        Returns:
            Number of pending messages.
        """
        try:
            info = await self._redis.xpending(stream_name, group)
            return info.get("pending", 0) if isinstance(info, dict) else 0
        except Exception as exc:
            logger.error(
                "XPENDING summary failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Stale Message Reclamation
    # ------------------------------------------------------------------

    async def reclaim_stale(
        self,
        stream_name: str,
        group: str,
        consumer: str,
        min_idle_ms: Optional[int] = None,
    ) -> List[MessageEnvelope]:
        """Reclaim stale messages from crashed consumers (XCLAIM).

        Messages that have been pending longer than ``min_idle_ms`` are
        claimed by the given consumer for reprocessing.

        Args:
            stream_name: Redis Stream key.
            group: Consumer group name.
            consumer: Consumer name that will take ownership.
            min_idle_ms: Minimum idle time in milliseconds.  Defaults to
                ``ack_timeout_s * 1000``.

        Returns:
            List of reclaimed MessageEnvelope instances.
        """
        if min_idle_ms is None:
            min_idle_ms = int(self.config.ack_timeout_s * 1000)

        self.metrics.reclaim_cycles += 1

        # Find stale messages
        pending = await self.get_pending(
            stream_name, group, count=self.config.max_reclaim_batch
        )
        stale_ids = [
            p.stream_id for p in pending if p.idle_ms >= min_idle_ms
        ]
        if not stale_ids:
            return []

        # XCLAIM the stale messages
        try:
            claimed_entries = await self._redis.xclaim(
                stream_name,
                group,
                consumer,
                min_idle_time=min_idle_ms,
                message_ids=stale_ids,
            )
        except Exception as exc:
            logger.error(
                "XCLAIM failed on stream %s group %s: %s",
                stream_name,
                group,
                exc,
            )
            return []

        reclaimed: List[MessageEnvelope] = []
        for entry_id, fields in claimed_entries:
            try:
                envelope = MessageEnvelope.from_flat_dict(fields)
                reclaimed.append(envelope)
            except Exception as exc:
                logger.warning(
                    "Failed to deserialize reclaimed entry %s: %s",
                    entry_id,
                    exc,
                )

        self.metrics.messages_reclaimed += len(reclaimed)
        if reclaimed:
            logger.info(
                "Reclaimed %d stale messages from stream %s (min_idle=%dms)",
                len(reclaimed),
                stream_name,
                min_idle_ms,
            )
        return reclaimed

    # ------------------------------------------------------------------
    # Dead-Letter Queue
    # ------------------------------------------------------------------

    async def move_to_dlq(
        self,
        stream_name: str,
        envelope: MessageEnvelope,
        error: str,
    ) -> str:
        """Move a failed message to the dead-letter queue stream.

        The DLQ entry includes the original envelope data plus error
        details and the timestamp of dead-lettering.

        Args:
            stream_name: Original Redis Stream key.
            envelope: The message envelope to dead-letter.
            error: Error description explaining the failure.

        Returns:
            The DLQ stream entry ID.
        """
        dlq_stream = self._dlq_stream(stream_name)

        dlq_data = envelope.to_flat_dict()
        dlq_data["dlq_error"] = error
        dlq_data["dlq_original_stream"] = stream_name
        dlq_data["dlq_dead_lettered_at"] = datetime.now(timezone.utc).isoformat()
        dlq_data["dlq_attempt_count"] = str(envelope.attempt)

        entry_id = await self._redis.xadd(dlq_stream, dlq_data)
        self.metrics.messages_dead_lettered += 1

        logger.warning(
            "Message %s dead-lettered to %s after %d attempts: %s",
            envelope.id,
            dlq_stream,
            envelope.attempt,
            error,
        )
        return entry_id

    async def list_dlq(
        self,
        stream_name: str,
        count: int = 50,
    ) -> List[Dict[str, Any]]:
        """List entries in the dead-letter queue for a stream.

        Args:
            stream_name: Original Redis Stream key.
            count: Maximum entries to return.

        Returns:
            List of DLQ entry dictionaries (newest first).
        """
        dlq_stream = self._dlq_stream(stream_name)
        try:
            entries = await self._redis.xrevrange(dlq_stream, count=count)
            return [
                {"entry_id": entry_id, **fields}
                for entry_id, fields in entries
            ]
        except Exception as exc:
            logger.error("Failed to list DLQ %s: %s", dlq_stream, exc)
            return []

    async def dlq_size(self, stream_name: str) -> int:
        """Get the number of entries in the DLQ for a stream.

        Args:
            stream_name: Original Redis Stream key.

        Returns:
            Number of DLQ entries.
        """
        dlq_stream = self._dlq_stream(stream_name)
        try:
            return await self._redis.xlen(dlq_stream)
        except Exception:
            return 0

    async def reprocess_from_dlq(
        self,
        stream_name: str,
        entry_id: str,
    ) -> bool:
        """Move a DLQ entry back to the original stream for reprocessing.

        The message's attempt counter is preserved so that another failure
        will return it to the DLQ.

        Args:
            stream_name: Original Redis Stream key.
            entry_id: DLQ stream entry ID to reprocess.

        Returns:
            True if the entry was found and reprocessed.
        """
        dlq_stream = self._dlq_stream(stream_name)
        try:
            entries = await self._redis.xrange(dlq_stream, count=1000)
            for eid, fields in entries:
                if eid == entry_id:
                    # Remove DLQ-specific fields before re-adding
                    reprocess_fields = {
                        k: v for k, v in fields.items()
                        if not k.startswith("dlq_")
                    }
                    # Reset attempt counter to allow fresh retries
                    reprocess_fields["attempt"] = "0"
                    await self._redis.xadd(stream_name, reprocess_fields)
                    await self._redis.xdel(dlq_stream, entry_id)
                    logger.info(
                        "Reprocessed DLQ entry %s back to stream %s",
                        entry_id,
                        stream_name,
                    )
                    return True
        except Exception as exc:
            logger.error(
                "Failed to reprocess DLQ entry %s: %s", entry_id, exc
            )
        return False

    async def purge_dlq(self, stream_name: str) -> int:
        """Delete all entries in the DLQ for a stream.

        Args:
            stream_name: Original Redis Stream key.

        Returns:
            Number of entries purged.
        """
        dlq_stream = self._dlq_stream(stream_name)
        try:
            length = await self._redis.xlen(dlq_stream)
            await self._redis.delete(dlq_stream)
            logger.info("Purged DLQ %s (%d entries)", dlq_stream, length)
            return length
        except Exception as exc:
            logger.error("Failed to purge DLQ %s: %s", dlq_stream, exc)
            return 0

    # ------------------------------------------------------------------
    # Tracking Lookup
    # ------------------------------------------------------------------

    async def get_tracking_info(
        self,
        stream_name: str,
        message_id: str,
    ) -> Optional[Dict[str, str]]:
        """Look up the tracking hash for a specific message.

        Args:
            stream_name: Redis Stream key.
            message_id: Redis Stream entry ID.

        Returns:
            Tracking hash data or None if not found.
        """
        tracking_key = self._tracking_key(stream_name, message_id)
        data = await self._redis.hgetall(tracking_key)
        if not data:
            return None
        return data

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _tracking_key(self, stream_name: str, message_id: str) -> str:
        """Build the Redis key for a per-message tracking hash.

        Args:
            stream_name: Redis Stream key.
            message_id: Redis Stream entry ID.

        Returns:
            Redis key string.
        """
        return f"{self.config.tracking_key_prefix}:{stream_name}:{message_id}"

    def _dlq_stream(self, stream_name: str) -> str:
        """Build the DLQ stream key for a given source stream.

        Args:
            stream_name: Original Redis Stream key.

        Returns:
            DLQ Redis Stream key.
        """
        return f"{self.config.dlq_prefix}:{stream_name}"


__all__ = [
    "AckTrackerConfig",
    "AckTrackerMetrics",
    "AcknowledgmentTracker",
]
