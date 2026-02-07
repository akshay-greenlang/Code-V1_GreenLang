# -*- coding: utf-8 -*-
"""
Dead Letter Queue - Storage and reprocessing of permanently failed tasks.

Tasks that exhaust their retry budget are moved to a dedicated DLQ stream
in Redis.  This module provides inspection, search, metrics, and
reprocessing capabilities for dead-lettered tasks.

Example:
    >>> dlq = DeadLetterQueue(redis_client=redis)
    >>> count = await dlq.size()
    >>> tasks = await dlq.list_tasks(limit=10)
    >>> reprocessed = await dlq.reprocess(task_id, target_queue)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.queue.task_queue import (
    TaskItem,
    TaskPriority,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# Redis import with graceful fallback
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeadLetterQueueConfig:
    """Configuration for the DeadLetterQueue.

    Attributes:
        stream_prefix: Redis key prefix (must match the task queue).
        dlq_stream_suffix: Suffix appended to the prefix for the DLQ stream.
        retention_seconds: Maximum age of entries before trimming.
        max_length: Approximate maximum entries in the DLQ stream.
        reprocess_consumer_group: Consumer group name for reprocessing reads.
    """

    stream_prefix: str = "gl:taskq"
    dlq_stream_suffix: str = "dlq"
    retention_seconds: int = 604800  # 7 days
    max_length: int = 50_000
    reprocess_consumer_group: str = "gl-dlq-reprocessors"


# ---------------------------------------------------------------------------
# Metrics model
# ---------------------------------------------------------------------------

@dataclass
class DLQMetrics:
    """Metrics snapshot for the dead letter queue.

    Attributes:
        size: Number of entries currently in the DLQ.
        oldest_entry_age_seconds: Age of the oldest entry in seconds.
        newest_entry_age_seconds: Age of the newest entry in seconds.
        reprocessed_count: Lifetime count of tasks reprocessed.
        agents: Count of unique agent keys in the DLQ.
    """

    size: int = 0
    oldest_entry_age_seconds: float = 0.0
    newest_entry_age_seconds: float = 0.0
    reprocessed_count: int = 0
    agents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "size": self.size,
            "oldest_entry_age_seconds": round(self.oldest_entry_age_seconds, 1),
            "newest_entry_age_seconds": round(self.newest_entry_age_seconds, 1),
            "reprocessed_count": self.reprocessed_count,
            "agents": self.agents,
        }


# ---------------------------------------------------------------------------
# DeadLetterQueue
# ---------------------------------------------------------------------------

class DeadLetterQueue:
    """Inspection, metrics, and reprocessing for dead-lettered tasks.

    The DLQ shares the Redis instance and key prefix with the main
    DistributedTaskQueue.  Entries are stored in a dedicated Redis Stream
    at ``{prefix}:{dlq_suffix}``.
    """

    def __init__(
        self,
        redis_client: Any,
        config: Optional[DeadLetterQueueConfig] = None,
    ) -> None:
        """Initialize the DLQ.

        Args:
            redis_client: Async Redis client.
            config: DLQ configuration.
        """
        self._redis = redis_client
        self._config = config or DeadLetterQueueConfig()
        self._dlq_key = (
            f"{self._config.stream_prefix}:{self._config.dlq_stream_suffix}"
        )
        self._reprocessed_count: int = 0
        logger.debug("DeadLetterQueue initialized at %s", self._dlq_key)

    # ------------------------------------------------------------------
    # Size / metrics
    # ------------------------------------------------------------------

    async def size(self) -> int:
        """Return the number of entries in the DLQ stream."""
        try:
            return await self._redis.xlen(self._dlq_key)
        except Exception:
            return 0

    async def metrics(self) -> DLQMetrics:
        """Compute current DLQ metrics.

        Returns:
            DLQMetrics snapshot.
        """
        length = await self.size()
        oldest_age = 0.0
        newest_age = 0.0
        agent_keys: set[str] = set()

        now = datetime.now(timezone.utc)

        if length > 0:
            # Oldest entry
            oldest_entries = await self._redis.xrange(
                self._dlq_key, count=1
            )
            if oldest_entries:
                _, fields = oldest_entries[0]
                oldest_ts = fields.get("created_at", "")
                if oldest_ts:
                    try:
                        oldest_dt = datetime.fromisoformat(oldest_ts)
                        oldest_age = (now - oldest_dt).total_seconds()
                    except ValueError:
                        pass

            # Newest entry
            newest_entries = await self._redis.xrevrange(
                self._dlq_key, count=1
            )
            if newest_entries:
                _, fields = newest_entries[0]
                newest_ts = fields.get("created_at", "")
                if newest_ts:
                    try:
                        newest_dt = datetime.fromisoformat(newest_ts)
                        newest_age = (now - newest_dt).total_seconds()
                    except ValueError:
                        pass

            # Unique agents (sample up to 500)
            sample = await self._redis.xrange(self._dlq_key, count=500)
            for _, fields in sample:
                ak = fields.get("agent_key", "")
                if ak:
                    agent_keys.add(ak)

        return DLQMetrics(
            size=length,
            oldest_entry_age_seconds=oldest_age,
            newest_entry_age_seconds=newest_age,
            reprocessed_count=self._reprocessed_count,
            agents=len(agent_keys),
        )

    # ------------------------------------------------------------------
    # Listing / inspection
    # ------------------------------------------------------------------

    async def list_tasks(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskItem]:
        """List tasks in the DLQ (newest first).

        Args:
            limit: Maximum number of entries to return.
            offset: Number of entries to skip from the newest end.

        Returns:
            List of TaskItem instances.
        """
        # Fetch offset + limit from the tail, then slice
        fetch_count = offset + limit
        entries = await self._redis.xrevrange(
            self._dlq_key, count=fetch_count
        )
        sliced = entries[offset: offset + limit]
        return [TaskItem.from_dict(fields) for _, fields in sliced]

    async def peek(self, task_id: str) -> Optional[TaskItem]:
        """Look up a specific task in the DLQ by its ID.

        Args:
            task_id: Task identifier to search for.

        Returns:
            TaskItem or None if not found.
        """
        # Scan up to 1000 entries (acceptable for DLQ sizes)
        entries = await self._redis.xrange(self._dlq_key, count=1000)
        for _, fields in entries:
            if fields.get("id") == task_id:
                return TaskItem.from_dict(fields)
        return None

    async def search_by_agent(
        self,
        agent_key: str,
        limit: int = 50,
    ) -> List[TaskItem]:
        """Find DLQ entries for a specific agent.

        Args:
            agent_key: Agent key to search for.
            limit: Maximum results.

        Returns:
            Matching TaskItem list.
        """
        entries = await self._redis.xrange(self._dlq_key, count=5000)
        results: List[TaskItem] = []
        for _, fields in entries:
            if fields.get("agent_key") == agent_key:
                results.append(TaskItem.from_dict(fields))
                if len(results) >= limit:
                    break
        return results

    # ------------------------------------------------------------------
    # Reprocessing
    # ------------------------------------------------------------------

    async def reprocess(
        self,
        task_id: str,
        target_stream_prefix: Optional[str] = None,
    ) -> bool:
        """Move a dead-lettered task back to the main queue.

        The task's status is reset to QUEUED and its attempt counter is
        preserved so that the next failure will re-enter the DLQ.

        Args:
            task_id: Task to reprocess.
            target_stream_prefix: Override the target stream prefix.

        Returns:
            True if the task was found and reprocessed.
        """
        prefix = target_stream_prefix or self._config.stream_prefix

        entries = await self._redis.xrange(self._dlq_key, count=5000)
        for entry_id, fields in entries:
            if fields.get("id") == task_id:
                task = TaskItem.from_dict(fields)
                task.status = TaskStatus.QUEUED
                task.worker_id = None
                task.error = None
                task.metadata.pop("stream_entry_id", None)

                target_key = f"{prefix}:p{task.priority.value}"
                await self._redis.xadd(target_key, task.to_dict())
                await self._redis.xdel(self._dlq_key, entry_id)
                self._reprocessed_count += 1

                logger.info(
                    "Reprocessed task %s from DLQ to %s",
                    task_id,
                    target_key,
                )
                return True

        logger.warning("Task %s not found in DLQ for reprocessing", task_id)
        return False

    async def reprocess_all(
        self,
        agent_key: Optional[str] = None,
        limit: int = 100,
    ) -> int:
        """Bulk-reprocess DLQ entries back to main queues.

        Args:
            agent_key: Optional filter by agent.
            limit: Maximum number of tasks to reprocess.

        Returns:
            Number of tasks reprocessed.
        """
        entries = await self._redis.xrange(self._dlq_key, count=5000)
        count = 0

        for entry_id, fields in entries:
            if count >= limit:
                break
            if agent_key and fields.get("agent_key") != agent_key:
                continue

            task = TaskItem.from_dict(fields)
            task.status = TaskStatus.QUEUED
            task.worker_id = None
            task.error = None
            task.metadata.pop("stream_entry_id", None)

            target_key = f"{self._config.stream_prefix}:p{task.priority.value}"
            await self._redis.xadd(target_key, task.to_dict())
            await self._redis.xdel(self._dlq_key, entry_id)
            count += 1

        self._reprocessed_count += count
        logger.info("Bulk reprocessed %d tasks from DLQ", count)
        return count

    # ------------------------------------------------------------------
    # Purge
    # ------------------------------------------------------------------

    async def purge(self, agent_key: Optional[str] = None) -> int:
        """Delete entries from the DLQ.

        Args:
            agent_key: If provided, only purge entries for this agent.
                If None, purge the entire DLQ.

        Returns:
            Number of entries removed.
        """
        if agent_key is None:
            length = await self.size()
            await self._redis.delete(self._dlq_key)
            logger.info("Purged entire DLQ (%d entries)", length)
            return length

        entries = await self._redis.xrange(self._dlq_key, count=10000)
        ids_to_delete: List[str] = []
        for entry_id, fields in entries:
            if fields.get("agent_key") == agent_key:
                ids_to_delete.append(entry_id)

        if ids_to_delete:
            await self._redis.xdel(self._dlq_key, *ids_to_delete)
        logger.info(
            "Purged %d DLQ entries for agent %s",
            len(ids_to_delete),
            agent_key,
        )
        return len(ids_to_delete)


__all__ = [
    "DLQMetrics",
    "DeadLetterQueue",
    "DeadLetterQueueConfig",
]
