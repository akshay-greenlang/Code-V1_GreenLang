# -*- coding: utf-8 -*-
"""
Distributed Task Queue - Redis Streams-backed task queue for agent execution.

Provides a production-grade distributed task queue built on Redis Streams
(XADD / XREADGROUP / XACK).  Features include priority ordering, consumer
group management, idempotency-key deduplication, task TTL, visibility
timeouts, progress tracking, and automatic dead-lettering of failed tasks.

Example:
    >>> queue = DistributedTaskQueue(redis_url="redis://localhost:6379")
    >>> await queue.initialize()
    >>> task_id = await queue.enqueue(TaskItem(
    ...     agent_key="carbon-agent",
    ...     payload={"scope": 1},
    ...     priority=TaskPriority.HIGH,
    ... ))
    >>> item = await queue.dequeue("worker-1")
    >>> await queue.acknowledge(item.id)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis import with graceful fallback
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Status of a task within the queue."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"


class TaskPriority(int, Enum):
    """Priority levels mapped to integer weights (lower = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaskItem:
    """Representation of a single task in the queue.

    Attributes:
        id: Unique task identifier (auto-generated if omitted).
        agent_key: Target agent that should process this task.
        payload: Arbitrary JSON-serialisable task data.
        priority: Task priority level.
        created_at: UTC ISO-8601 creation timestamp.
        ttl: Time-to-live in seconds (0 = no expiry).
        idempotency_key: Optional deduplication key (24h window).
        metadata: Extra metadata for routing / tracing.
        status: Current task status.
        attempts: Number of delivery attempts.
        max_retries: Maximum number of retries before dead-lettering.
        error: Last error message (if any).
        progress: Completion percentage (0-100).
        worker_id: ID of the worker currently processing this task.
    """

    agent_key: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl: int = 0
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.QUEUED
    attempts: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    progress: int = 0
    worker_id: Optional[str] = None

    def is_expired(self) -> bool:
        """Check whether the task has exceeded its TTL."""
        if self.ttl <= 0:
            return False
        created = datetime.fromisoformat(self.created_at)
        age = (datetime.now(timezone.utc) - created).total_seconds()
        return age > self.ttl

    def to_dict(self) -> Dict[str, str]:
        """Serialize to a flat dict suitable for Redis XADD."""
        return {
            "id": self.id,
            "agent_key": self.agent_key,
            "payload": json.dumps(self.payload),
            "priority": str(self.priority.value),
            "created_at": self.created_at,
            "ttl": str(self.ttl),
            "idempotency_key": self.idempotency_key or "",
            "metadata": json.dumps(self.metadata),
            "status": self.status.value,
            "attempts": str(self.attempts),
            "max_retries": str(self.max_retries),
            "error": self.error or "",
            "progress": str(self.progress),
            "worker_id": self.worker_id or "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> TaskItem:
        """Deserialize from a Redis hash / stream entry."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            agent_key=data.get("agent_key", ""),
            payload=json.loads(data.get("payload", "{}")),
            priority=TaskPriority(int(data.get("priority", "2"))),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            ttl=int(data.get("ttl", "0")),
            idempotency_key=data.get("idempotency_key") or None,
            metadata=json.loads(data.get("metadata", "{}")),
            status=TaskStatus(data.get("status", "queued")),
            attempts=int(data.get("attempts", "0")),
            max_retries=int(data.get("max_retries", "3")),
            error=data.get("error") or None,
            progress=int(data.get("progress", "0")),
            worker_id=data.get("worker_id") or None,
        )


# ---------------------------------------------------------------------------
# Queue configuration
# ---------------------------------------------------------------------------

@dataclass
class TaskQueueConfig:
    """Configuration for the DistributedTaskQueue.

    Attributes:
        redis_url: Redis connection URL.
        stream_prefix: Prefix for Redis stream keys.
        consumer_group: Name of the consumer group.
        visibility_timeout: Seconds before an unacknowledged task is
            redelivered.
        idempotency_window: Seconds for idempotency-key dedup window.
        max_stream_length: Max entries per stream (approximate via MAXLEN ~).
        block_ms: XREADGROUP block duration in milliseconds.
    """

    redis_url: str = "redis://localhost:6379"
    stream_prefix: str = "gl:taskq"
    consumer_group: str = "gl-workers"
    visibility_timeout: int = 300
    idempotency_window: int = 86400  # 24 hours
    max_stream_length: int = 100_000
    block_ms: int = 2000


# ---------------------------------------------------------------------------
# DistributedTaskQueue
# ---------------------------------------------------------------------------

class DistributedTaskQueue:
    """Redis Streams-based distributed task queue.

    Each priority level maps to a separate Redis Stream so that workers
    can read from higher-priority streams first.

    Stream keys:  ``{prefix}:p{priority_value}``
    Status hash:  ``{prefix}:status:{task_id}``
    Idempotency:  ``{prefix}:idem:{key}``

    Lifecycle:
        1. ``enqueue()`` adds a task to the appropriate priority stream.
        2. ``dequeue()`` reads from streams in priority order.
        3. ``acknowledge()`` marks a task complete (XACK + status update).
        4. ``fail()`` increments attempts; moves to DLQ on exhaustion.
    """

    def __init__(
        self,
        config: Optional[TaskQueueConfig] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialize the queue.

        Args:
            config: Queue configuration.
            redis_client: Optional pre-existing async Redis client.
        """
        self._config = config or TaskQueueConfig()
        self._redis: Optional[Any] = redis_client
        self._initialized = False
        logger.debug(
            "DistributedTaskQueue constructed (prefix=%s)",
            self._config.stream_prefix,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Redis and create consumer groups for all priority streams."""
        if self._redis is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError(
                    "redis.asyncio is required but not installed"
                )
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )

        # Create consumer groups (idempotent)
        for priority in TaskPriority:
            stream_key = self._stream_key(priority)
            try:
                await self._redis.xgroup_create(
                    stream_key,
                    self._config.consumer_group,
                    id="0",
                    mkstream=True,
                )
            except Exception as exc:
                # BUSYGROUP means group already exists
                if "BUSYGROUP" not in str(exc):
                    raise

        self._initialized = True
        logger.info("DistributedTaskQueue initialized with %d priority streams", len(TaskPriority))

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    async def enqueue(self, task: TaskItem) -> str:
        """Add a task to the queue.

        Args:
            task: Task to enqueue.

        Returns:
            The task ID.

        Raises:
            ValueError: If the idempotency key is duplicated.
            RuntimeError: If the queue is not initialized.
        """
        self._ensure_initialized()

        # Idempotency check
        if task.idempotency_key:
            idem_key = self._idem_key(task.idempotency_key)
            already = await self._redis.get(idem_key)
            if already:
                logger.info(
                    "Duplicate idempotency key '%s', returning existing task %s",
                    task.idempotency_key,
                    already,
                )
                return already

        # Expire check
        if task.is_expired():
            logger.warning("Task %s is already expired, skipping enqueue", task.id)
            return task.id

        task.status = TaskStatus.QUEUED
        stream_key = self._stream_key(task.priority)

        # XADD
        await self._redis.xadd(
            stream_key,
            task.to_dict(),
            maxlen=self._config.max_stream_length,
        )

        # Status hash
        await self._save_status(task)

        # Idempotency marker
        if task.idempotency_key:
            await self._redis.set(
                self._idem_key(task.idempotency_key),
                task.id,
                ex=self._config.idempotency_window,
            )

        logger.debug(
            "Enqueued task %s (agent=%s, priority=%s)",
            task.id,
            task.agent_key,
            task.priority.name,
        )
        return task.id

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    async def dequeue(
        self,
        worker_id: str,
        count: int = 1,
    ) -> Optional[TaskItem]:
        """Read the next highest-priority task from the queue.

        Reads from priority streams in order (CRITICAL first). Uses
        XREADGROUP with consumer group semantics.

        Args:
            worker_id: Unique worker identifier (consumer name).
            count: Number of messages to read (currently returns first).

        Returns:
            A TaskItem if available, otherwise None.
        """
        self._ensure_initialized()

        for priority in TaskPriority:
            stream_key = self._stream_key(priority)
            try:
                messages = await self._redis.xreadgroup(
                    groupname=self._config.consumer_group,
                    consumername=worker_id,
                    streams={stream_key: ">"},
                    count=count,
                    block=self._config.block_ms,
                )
            except Exception as exc:
                logger.error(
                    "XREADGROUP error on stream %s: %s", stream_key, exc
                )
                continue

            if not messages:
                continue

            # messages = [(stream_key, [(entry_id, {fields})])]
            for _stream, entries in messages:
                for entry_id, fields in entries:
                    task = TaskItem.from_dict(fields)
                    if task.is_expired():
                        await self._redis.xack(
                            stream_key,
                            self._config.consumer_group,
                            entry_id,
                        )
                        task.status = TaskStatus.TIMEOUT
                        await self._save_status(task)
                        continue

                    task.status = TaskStatus.RUNNING
                    task.worker_id = worker_id
                    task.attempts += 1
                    task.metadata["stream_entry_id"] = entry_id
                    await self._save_status(task)
                    return task

        return None

    # ------------------------------------------------------------------
    # Acknowledge / Fail
    # ------------------------------------------------------------------

    async def acknowledge(self, task_id: str) -> bool:
        """Mark a task as successfully completed.

        Args:
            task_id: Task to acknowledge.

        Returns:
            True if acknowledged.
        """
        self._ensure_initialized()
        task = await self.get_task(task_id)
        if task is None:
            return False

        entry_id = task.metadata.get("stream_entry_id")
        if entry_id:
            stream_key = self._stream_key(task.priority)
            await self._redis.xack(
                stream_key,
                self._config.consumer_group,
                entry_id,
            )

        task.status = TaskStatus.COMPLETED
        task.progress = 100
        await self._save_status(task)
        logger.debug("Task %s acknowledged by %s", task_id, task.worker_id)
        return True

    async def fail(
        self,
        task_id: str,
        error: str,
    ) -> bool:
        """Mark a task as failed; retry or move to DLQ.

        Args:
            task_id: Task that failed.
            error: Error description.

        Returns:
            True if the task was processed (retried or dead-lettered).
        """
        self._ensure_initialized()
        task = await self.get_task(task_id)
        if task is None:
            return False

        # Acknowledge from the original stream so it is not redelivered
        entry_id = task.metadata.get("stream_entry_id")
        if entry_id:
            stream_key = self._stream_key(task.priority)
            await self._redis.xack(
                stream_key,
                self._config.consumer_group,
                entry_id,
            )

        task.error = error

        if task.attempts < task.max_retries:
            # Re-enqueue for retry
            task.status = TaskStatus.QUEUED
            task.worker_id = None
            task.metadata.pop("stream_entry_id", None)
            stream_key = self._stream_key(task.priority)
            await self._redis.xadd(
                stream_key,
                task.to_dict(),
                maxlen=self._config.max_stream_length,
            )
            await self._save_status(task)
            logger.info(
                "Task %s re-enqueued for retry (attempt %d/%d)",
                task_id,
                task.attempts,
                task.max_retries,
            )
        else:
            # Dead-letter
            task.status = TaskStatus.DEAD_LETTERED
            dlq_key = f"{self._config.stream_prefix}:dlq"
            await self._redis.xadd(
                dlq_key,
                task.to_dict(),
                maxlen=self._config.max_stream_length,
            )
            await self._save_status(task)
            logger.warning(
                "Task %s dead-lettered after %d attempts: %s",
                task_id,
                task.attempts,
                error,
            )

        return True

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    async def update_progress(self, task_id: str, progress: int) -> bool:
        """Update the progress percentage for a running task.

        Args:
            task_id: Task to update.
            progress: Completion percentage (0-100).

        Returns:
            True if the task was found and updated.
        """
        self._ensure_initialized()
        status_key = self._status_key(task_id)
        exists = await self._redis.exists(status_key)
        if not exists:
            return False
        await self._redis.hset(status_key, "progress", str(min(100, max(0, progress))))
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_task(self, task_id: str) -> Optional[TaskItem]:
        """Retrieve a task by its ID from the status hash.

        Args:
            task_id: Task ID.

        Returns:
            TaskItem or None.
        """
        self._ensure_initialized()
        status_key = self._status_key(task_id)
        data = await self._redis.hgetall(status_key)
        if not data:
            return None
        return TaskItem.from_dict(data)

    async def queue_length(self, priority: Optional[TaskPriority] = None) -> int:
        """Return the number of entries across priority streams.

        Args:
            priority: Optional filter by specific priority.

        Returns:
            Total number of pending entries.
        """
        self._ensure_initialized()
        total = 0
        priorities = [priority] if priority is not None else list(TaskPriority)
        for p in priorities:
            stream_key = self._stream_key(p)
            try:
                length = await self._redis.xlen(stream_key)
                total += length
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Consumer group management
    # ------------------------------------------------------------------

    async def add_consumer(self, worker_id: str) -> None:
        """Explicitly create a consumer in each priority stream group.

        Args:
            worker_id: Consumer / worker identifier.
        """
        self._ensure_initialized()
        for priority in TaskPriority:
            stream_key = self._stream_key(priority)
            try:
                await self._redis.xgroup_createconsumer(
                    stream_key,
                    self._config.consumer_group,
                    worker_id,
                )
            except Exception:
                pass  # consumer may already exist

    async def remove_consumer(self, worker_id: str) -> None:
        """Remove a consumer from all priority stream groups.

        Args:
            worker_id: Consumer / worker identifier.
        """
        self._ensure_initialized()
        for priority in TaskPriority:
            stream_key = self._stream_key(priority)
            try:
                await self._redis.xgroup_delconsumer(
                    stream_key,
                    self._config.consumer_group,
                    worker_id,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream_key(self, priority: TaskPriority) -> str:
        return f"{self._config.stream_prefix}:p{priority.value}"

    def _status_key(self, task_id: str) -> str:
        return f"{self._config.stream_prefix}:status:{task_id}"

    def _idem_key(self, key: str) -> str:
        return f"{self._config.stream_prefix}:idem:{key}"

    async def _save_status(self, task: TaskItem) -> None:
        """Persist task status to a Redis hash with TTL."""
        status_key = self._status_key(task.id)
        await self._redis.hset(status_key, mapping=task.to_dict())
        # Status hashes expire after 7 days
        await self._redis.expire(status_key, 604800)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "DistributedTaskQueue is not initialized; call initialize() first"
            )


__all__ = [
    "DistributedTaskQueue",
    "TaskItem",
    "TaskPriority",
    "TaskQueueConfig",
    "TaskStatus",
]
