"""
Dead Letter Queue Handler for GreenLang

This module provides dead letter queue handling for failed events,
enabling manual review, retry, and auditing of failed messages.

Features:
- Automatic DLQ routing
- Retry scheduling
- Failure analysis
- Manual review workflows
- Alerting integration

Example:
    >>> dlq = DeadLetterQueue(config)
    >>> await dlq.start()
    >>> await dlq.process_failed_events()
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.events.event_schema import BaseEvent, EventMetadata

logger = logging.getLogger(__name__)


class DLQStatus(str, Enum):
    """Dead letter queue entry status."""
    PENDING = "pending"
    REVIEWING = "reviewing"
    RETRYING = "retrying"
    RESOLVED = "resolved"
    DISCARDED = "discarded"
    ESCALATED = "escalated"


class FailureReason(str, Enum):
    """Standard failure reasons."""
    HANDLER_ERROR = "handler_error"
    HANDLER_TIMEOUT = "handler_timeout"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    VALIDATION_FAILED = "validation_failed"
    DESERIALIZATION_FAILED = "deserialization_failed"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


@dataclass
class DeadLetterQueueConfig:
    """Configuration for dead letter queue."""
    storage_backend: str = "memory"  # memory, redis, postgres
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://localhost/greenlang"
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60
    retention_days: int = 30
    alert_threshold: int = 100
    alert_webhook_url: Optional[str] = None
    enable_auto_retry: bool = False
    auto_retry_interval_seconds: int = 300


class DLQEntry(BaseModel):
    """Dead letter queue entry."""
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    event: BaseEvent = Field(..., description="Failed event")
    original_topic: str = Field(..., description="Original topic")
    failure_reason: FailureReason = Field(..., description="Reason for failure")
    error_message: Optional[str] = Field(default=None, description="Error details")
    error_stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    retry_count: int = Field(default=0, description="Number of retries")
    max_retries: int = Field(default=3, description="Maximum retries allowed")
    status: DLQStatus = Field(default=DLQStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    next_retry_at: Optional[datetime] = Field(default=None)
    resolved_at: Optional[datetime] = Field(default=None)
    resolved_by: Optional[str] = Field(default=None)
    resolution_notes: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_next_retry(self, base_delay_seconds: int) -> datetime:
        """Calculate next retry time with exponential backoff."""
        delay = base_delay_seconds * (2 ** self.retry_count)
        return datetime.utcnow() + timedelta(seconds=delay)


class DLQStats(BaseModel):
    """Dead letter queue statistics."""
    total_entries: int = Field(default=0)
    pending_count: int = Field(default=0)
    reviewing_count: int = Field(default=0)
    retrying_count: int = Field(default=0)
    resolved_count: int = Field(default=0)
    discarded_count: int = Field(default=0)
    escalated_count: int = Field(default=0)
    entries_by_reason: Dict[str, int] = Field(default_factory=dict)
    entries_by_topic: Dict[str, int] = Field(default_factory=dict)
    oldest_pending_age_hours: Optional[float] = Field(default=None)


class DLQStorageBackend:
    """Base class for DLQ storage backends."""

    async def save(self, entry: DLQEntry) -> None:
        """Save a DLQ entry."""
        raise NotImplementedError

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        """Get a DLQ entry by ID."""
        raise NotImplementedError

    async def update(self, entry: DLQEntry) -> None:
        """Update a DLQ entry."""
        raise NotImplementedError

    async def delete(self, entry_id: str) -> None:
        """Delete a DLQ entry."""
        raise NotImplementedError

    async def list_pending(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[DLQEntry]:
        """List pending entries."""
        raise NotImplementedError

    async def list_ready_for_retry(
        self,
        limit: int = 100
    ) -> List[DLQEntry]:
        """List entries ready for retry."""
        raise NotImplementedError

    async def get_stats(self) -> DLQStats:
        """Get queue statistics."""
        raise NotImplementedError

    async def cleanup_old_entries(self, retention_days: int) -> int:
        """Clean up old entries."""
        raise NotImplementedError


class MemoryDLQStorage(DLQStorageBackend):
    """In-memory DLQ storage for testing."""

    def __init__(self):
        """Initialize memory storage."""
        self._entries: Dict[str, DLQEntry] = {}

    async def save(self, entry: DLQEntry) -> None:
        """Save a DLQ entry."""
        self._entries[entry.entry_id] = entry

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        """Get a DLQ entry by ID."""
        return self._entries.get(entry_id)

    async def update(self, entry: DLQEntry) -> None:
        """Update a DLQ entry."""
        entry.updated_at = datetime.utcnow()
        self._entries[entry.entry_id] = entry

    async def delete(self, entry_id: str) -> None:
        """Delete a DLQ entry."""
        self._entries.pop(entry_id, None)

    async def list_pending(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[DLQEntry]:
        """List pending entries."""
        pending = [
            e for e in self._entries.values()
            if e.status == DLQStatus.PENDING
        ]
        pending.sort(key=lambda e: e.created_at)
        return pending[offset:offset + limit]

    async def list_ready_for_retry(
        self,
        limit: int = 100
    ) -> List[DLQEntry]:
        """List entries ready for retry."""
        now = datetime.utcnow()
        ready = [
            e for e in self._entries.values()
            if e.status == DLQStatus.PENDING
            and e.next_retry_at
            and e.next_retry_at <= now
            and e.retry_count < e.max_retries
        ]
        ready.sort(key=lambda e: e.next_retry_at)
        return ready[:limit]

    async def get_stats(self) -> DLQStats:
        """Get queue statistics."""
        stats = DLQStats()

        for entry in self._entries.values():
            stats.total_entries += 1

            if entry.status == DLQStatus.PENDING:
                stats.pending_count += 1
            elif entry.status == DLQStatus.REVIEWING:
                stats.reviewing_count += 1
            elif entry.status == DLQStatus.RETRYING:
                stats.retrying_count += 1
            elif entry.status == DLQStatus.RESOLVED:
                stats.resolved_count += 1
            elif entry.status == DLQStatus.DISCARDED:
                stats.discarded_count += 1
            elif entry.status == DLQStatus.ESCALATED:
                stats.escalated_count += 1

            reason = entry.failure_reason.value
            stats.entries_by_reason[reason] = stats.entries_by_reason.get(reason, 0) + 1

            topic = entry.original_topic
            stats.entries_by_topic[topic] = stats.entries_by_topic.get(topic, 0) + 1

        # Calculate oldest pending age
        pending = [e for e in self._entries.values() if e.status == DLQStatus.PENDING]
        if pending:
            oldest = min(pending, key=lambda e: e.created_at)
            age = (datetime.utcnow() - oldest.created_at).total_seconds() / 3600
            stats.oldest_pending_age_hours = age

        return stats

    async def cleanup_old_entries(self, retention_days: int) -> int:
        """Clean up old entries."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        old_ids = [
            eid for eid, entry in self._entries.items()
            if entry.status in [DLQStatus.RESOLVED, DLQStatus.DISCARDED]
            and entry.updated_at < cutoff
        ]
        for eid in old_ids:
            del self._entries[eid]
        return len(old_ids)


class DeadLetterQueue:
    """
    Dead Letter Queue handler for GreenLang.

    Manages failed events, enabling retry, manual review,
    and resolution tracking.

    Attributes:
        config: DLQ configuration
        storage: Storage backend

    Example:
        >>> config = DeadLetterQueueConfig(
        ...     enable_auto_retry=True,
        ...     max_retry_attempts=3
        ... )
        >>> dlq = DeadLetterQueue(config)
        >>> async with dlq:
        ...     await dlq.add_failed_event(event, "handler_error")
    """

    def __init__(self, config: DeadLetterQueueConfig):
        """
        Initialize dead letter queue.

        Args:
            config: DLQ configuration
        """
        self.config = config
        self._storage: Optional[DLQStorageBackend] = None
        self._started = False
        self._auto_retry_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._retry_handler: Optional[Callable] = None
        self._alert_callbacks: List[Callable] = []

        logger.info("DeadLetterQueue initialized")

    async def start(self) -> None:
        """
        Start the dead letter queue.

        Initializes storage and starts background tasks.
        """
        if self._started:
            logger.warning("DLQ already started")
            return

        try:
            # Initialize storage
            self._storage = self._create_storage()

            # Start auto-retry if enabled
            if self.config.enable_auto_retry:
                self._auto_retry_task = asyncio.create_task(
                    self._auto_retry_loop()
                )

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop()
            )

            self._started = True
            self._shutdown = False

            logger.info("Dead letter queue started")

        except Exception as e:
            logger.error(f"Failed to start DLQ: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the dead letter queue gracefully.
        """
        self._shutdown = True

        # Cancel tasks
        if self._auto_retry_task:
            self._auto_retry_task.cancel()
            try:
                await self._auto_retry_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._started = False
        logger.info("Dead letter queue stopped")

    def _create_storage(self) -> DLQStorageBackend:
        """Create storage backend."""
        if self.config.storage_backend == "memory":
            return MemoryDLQStorage()
        # Would add redis and postgres backends
        return MemoryDLQStorage()

    def set_retry_handler(
        self,
        handler: Callable[[DLQEntry], bool]
    ) -> None:
        """
        Set the retry handler function.

        Args:
            handler: Function that attempts to reprocess the event
        """
        self._retry_handler = handler
        logger.info("Retry handler set")

    def add_alert_callback(
        self,
        callback: Callable[[DLQStats], None]
    ) -> None:
        """
        Add an alert callback.

        Args:
            callback: Function called when alert threshold is reached
        """
        self._alert_callbacks.append(callback)

    async def add_failed_event(
        self,
        event: BaseEvent,
        original_topic: str,
        failure_reason: FailureReason,
        error_message: Optional[str] = None,
        error_stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DLQEntry:
        """
        Add a failed event to the dead letter queue.

        Args:
            event: Failed event
            original_topic: Topic the event was consumed from
            failure_reason: Reason for failure
            error_message: Error details
            error_stack_trace: Stack trace
            metadata: Additional metadata

        Returns:
            Created DLQ entry
        """
        self._ensure_started()

        entry = DLQEntry(
            event=event,
            original_topic=original_topic,
            failure_reason=failure_reason,
            error_message=error_message,
            error_stack_trace=error_stack_trace,
            max_retries=self.config.max_retry_attempts,
            metadata=metadata or {},
        )

        # Calculate next retry time
        entry.next_retry_at = entry.calculate_next_retry(
            self.config.retry_delay_seconds
        )

        await self._storage.save(entry)

        logger.info(
            f"Added to DLQ: {event.event_type} "
            f"reason: {failure_reason.value} "
            f"entry_id: {entry.entry_id}"
        )

        # Check alert threshold
        await self._check_alerts()

        return entry

    async def get_entry(self, entry_id: str) -> Optional[DLQEntry]:
        """
        Get a DLQ entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            DLQ entry or None
        """
        self._ensure_started()
        return await self._storage.get(entry_id)

    async def list_pending(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[DLQEntry]:
        """
        List pending DLQ entries.

        Args:
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of pending entries
        """
        self._ensure_started()
        return await self._storage.list_pending(limit, offset)

    async def retry_entry(self, entry_id: str) -> bool:
        """
        Manually retry a DLQ entry.

        Args:
            entry_id: Entry to retry

        Returns:
            True if retry succeeded
        """
        self._ensure_started()

        entry = await self._storage.get(entry_id)
        if not entry:
            logger.warning(f"DLQ entry not found: {entry_id}")
            return False

        if entry.status != DLQStatus.PENDING:
            logger.warning(f"Entry not pending: {entry_id} status: {entry.status}")
            return False

        entry.status = DLQStatus.RETRYING
        entry.retry_count += 1
        await self._storage.update(entry)

        success = await self._execute_retry(entry)

        if success:
            entry.status = DLQStatus.RESOLVED
            entry.resolved_at = datetime.utcnow()
            entry.resolution_notes = "Retry successful"
        else:
            if entry.retry_count >= entry.max_retries:
                entry.status = DLQStatus.ESCALATED
            else:
                entry.status = DLQStatus.PENDING
                entry.next_retry_at = entry.calculate_next_retry(
                    self.config.retry_delay_seconds
                )

        await self._storage.update(entry)
        return success

    async def _execute_retry(self, entry: DLQEntry) -> bool:
        """Execute retry for an entry."""
        if not self._retry_handler:
            logger.warning("No retry handler configured")
            return False

        try:
            if asyncio.iscoroutinefunction(self._retry_handler):
                return await self._retry_handler(entry)
            else:
                return self._retry_handler(entry)
        except Exception as e:
            logger.error(f"Retry handler failed: {e}")
            entry.error_message = str(e)
            return False

    async def resolve_entry(
        self,
        entry_id: str,
        resolved_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually resolve a DLQ entry.

        Args:
            entry_id: Entry to resolve
            resolved_by: User who resolved
            notes: Resolution notes

        Returns:
            True if resolved
        """
        self._ensure_started()

        entry = await self._storage.get(entry_id)
        if not entry:
            return False

        entry.status = DLQStatus.RESOLVED
        entry.resolved_at = datetime.utcnow()
        entry.resolved_by = resolved_by
        entry.resolution_notes = notes

        await self._storage.update(entry)
        logger.info(f"DLQ entry resolved: {entry_id} by {resolved_by}")
        return True

    async def discard_entry(
        self,
        entry_id: str,
        discarded_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Discard a DLQ entry.

        Args:
            entry_id: Entry to discard
            discarded_by: User who discarded
            notes: Discard notes

        Returns:
            True if discarded
        """
        self._ensure_started()

        entry = await self._storage.get(entry_id)
        if not entry:
            return False

        entry.status = DLQStatus.DISCARDED
        entry.resolved_at = datetime.utcnow()
        entry.resolved_by = discarded_by
        entry.resolution_notes = notes

        await self._storage.update(entry)
        logger.info(f"DLQ entry discarded: {entry_id} by {discarded_by}")
        return True

    async def get_stats(self) -> DLQStats:
        """
        Get DLQ statistics.

        Returns:
            Queue statistics
        """
        self._ensure_started()
        return await self._storage.get_stats()

    async def _auto_retry_loop(self) -> None:
        """Background loop for automatic retries."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.auto_retry_interval_seconds)

                ready = await self._storage.list_ready_for_retry(limit=100)
                for entry in ready:
                    await self.retry_entry(entry.entry_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-retry loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup."""
        while not self._shutdown:
            try:
                # Run cleanup once per day
                await asyncio.sleep(86400)

                count = await self._storage.cleanup_old_entries(
                    self.config.retention_days
                )
                logger.info(f"Cleaned up {count} old DLQ entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _check_alerts(self) -> None:
        """Check if alert threshold is reached."""
        stats = await self._storage.get_stats()

        if stats.pending_count >= self.config.alert_threshold:
            logger.warning(
                f"DLQ alert threshold reached: {stats.pending_count} pending"
            )

            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(stats)
                    else:
                        callback(stats)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def _ensure_started(self) -> None:
        """Ensure DLQ is started."""
        if not self._started:
            raise RuntimeError("DLQ not started")

    async def __aenter__(self) -> "DeadLetterQueue":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
