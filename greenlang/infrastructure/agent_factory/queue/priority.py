# -*- coding: utf-8 -*-
"""
Priority Scheduler - Fair scheduling with starvation prevention.

Builds on the per-priority Redis Streams created by DistributedTaskQueue
to implement weighted fair scheduling.  Higher-priority streams are polled
more frequently, while a starvation-prevention mechanism promotes aged
low-priority tasks to ensure eventual processing.

Example:
    >>> scheduler = PriorityScheduler(redis_client=redis)
    >>> task = await scheduler.next_task("worker-1")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
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
class PrioritySchedulerConfig:
    """Configuration for the PriorityScheduler.

    Attributes:
        stream_prefix: Redis key prefix matching the task queue.
        consumer_group: Consumer group name matching the task queue.
        weights: Mapping of TaskPriority to relative poll weight.
            Higher weight means the stream is read more often per
            scheduling cycle.
        starvation_threshold_seconds: Age in seconds after which a
            low-priority task is promoted one level up.
        promotion_check_interval: Seconds between starvation sweeps.
        block_ms: XREADGROUP block timeout in milliseconds.
    """

    stream_prefix: str = "gl:taskq"
    consumer_group: str = "gl-workers"
    weights: Dict[TaskPriority, int] = field(default_factory=lambda: {
        TaskPriority.CRITICAL: 8,
        TaskPriority.HIGH: 4,
        TaskPriority.NORMAL: 2,
        TaskPriority.LOW: 1,
        TaskPriority.BACKGROUND: 1,
    })
    starvation_threshold_seconds: float = 300.0
    promotion_check_interval: float = 60.0
    block_ms: int = 1000


# ---------------------------------------------------------------------------
# PriorityScheduler
# ---------------------------------------------------------------------------

class PriorityScheduler:
    """Weighted fair-share scheduler across priority streams.

    The scheduler cycles through priority levels according to their
    configured weights.  For example with weights CRITICAL=8, HIGH=4,
    NORMAL=2, LOW=1, BACKGROUND=1, the CRITICAL stream will be polled
    8 times for every 1 poll of the LOW stream.

    A background starvation-prevention sweep periodically moves tasks
    that have been waiting longer than ``starvation_threshold_seconds``
    into the next-higher priority stream.
    """

    def __init__(
        self,
        redis_client: Any,
        config: Optional[PrioritySchedulerConfig] = None,
    ) -> None:
        """Initialize the priority scheduler.

        Args:
            redis_client: Async Redis client.
            config: Scheduler configuration.
        """
        self._redis = redis_client
        self._config = config or PrioritySchedulerConfig()
        self._schedule: List[TaskPriority] = self._build_schedule()
        self._schedule_index: int = 0
        self._running: bool = False
        self._promotion_task: Optional[asyncio.Task[None]] = None
        logger.debug(
            "PriorityScheduler initialized (schedule length=%d)",
            len(self._schedule),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the starvation-prevention background loop."""
        if self._running:
            return
        self._running = True
        self._promotion_task = asyncio.create_task(
            self._starvation_prevention_loop()
        )
        logger.info("PriorityScheduler started")

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._promotion_task and not self._promotion_task.done():
            self._promotion_task.cancel()
            try:
                await self._promotion_task
            except asyncio.CancelledError:
                pass
        logger.info("PriorityScheduler stopped")

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    async def next_task(self, worker_id: str) -> Optional[TaskItem]:
        """Read the next task according to the weighted schedule.

        Args:
            worker_id: Worker / consumer identifier.

        Returns:
            A TaskItem or None if no work is available.
        """
        # Cycle through the full schedule once to find work
        for _ in range(len(self._schedule)):
            priority = self._schedule[self._schedule_index]
            self._schedule_index = (
                (self._schedule_index + 1) % len(self._schedule)
            )

            stream_key = self._stream_key(priority)
            try:
                messages = await self._redis.xreadgroup(
                    groupname=self._config.consumer_group,
                    consumername=worker_id,
                    streams={stream_key: ">"},
                    count=1,
                    block=self._config.block_ms,
                )
            except Exception as exc:
                logger.error("XREADGROUP error on %s: %s", stream_key, exc)
                continue

            if not messages:
                continue

            for _stream, entries in messages:
                for entry_id, fields in entries:
                    task = TaskItem.from_dict(fields)
                    if task.is_expired():
                        await self._ack_entry(stream_key, entry_id)
                        continue
                    task.status = TaskStatus.RUNNING
                    task.worker_id = worker_id
                    task.attempts += 1
                    task.metadata["stream_entry_id"] = entry_id
                    return task

        return None

    # ------------------------------------------------------------------
    # Starvation prevention
    # ------------------------------------------------------------------

    async def _starvation_prevention_loop(self) -> None:
        """Periodically promote aged tasks to higher-priority streams."""
        while self._running:
            try:
                await asyncio.sleep(self._config.promotion_check_interval)
                await self._promote_starved_tasks()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Starvation prevention error")

    async def _promote_starved_tasks(self) -> None:
        """Scan lower-priority streams and promote aged pending tasks."""
        now = datetime.now(timezone.utc)
        promotable = [
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
            TaskPriority.BACKGROUND,
        ]

        for priority in promotable:
            stream_key = self._stream_key(priority)
            try:
                # Read pending entries (those not yet delivered)
                pending = await self._redis.xpending_range(
                    stream_key,
                    self._config.consumer_group,
                    min="-",
                    max="+",
                    count=50,
                )
            except Exception:
                continue

            if not pending:
                continue

            for entry in pending:
                idle_ms = entry.get("time_since_delivered", 0)
                if idle_ms is None:
                    continue
                idle_seconds = idle_ms / 1000.0
                if idle_seconds < self._config.starvation_threshold_seconds:
                    continue

                # Promote to one level higher
                higher = self._higher_priority(priority)
                if higher is None:
                    continue

                entry_id = entry.get("message_id")
                if entry_id is None:
                    continue

                # Read the entry data
                msgs = await self._redis.xrange(
                    stream_key, min=entry_id, max=entry_id
                )
                if not msgs:
                    continue

                _, fields = msgs[0]
                # Add to higher-priority stream
                higher_key = self._stream_key(higher)
                await self._redis.xadd(higher_key, fields)
                # Acknowledge original
                await self._ack_entry(stream_key, entry_id)
                logger.info(
                    "Promoted task from %s to %s (idle %.1fs)",
                    priority.name,
                    higher.name,
                    idle_seconds,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_schedule(self) -> List[TaskPriority]:
        """Build a flat schedule list by repeating each priority by weight."""
        schedule: List[TaskPriority] = []
        for priority in TaskPriority:
            weight = self._config.weights.get(priority, 1)
            schedule.extend([priority] * weight)
        return schedule

    def _stream_key(self, priority: TaskPriority) -> str:
        return f"{self._config.stream_prefix}:p{priority.value}"

    async def _ack_entry(self, stream_key: str, entry_id: str) -> None:
        """Acknowledge a single stream entry."""
        try:
            await self._redis.xack(
                stream_key, self._config.consumer_group, entry_id
            )
        except Exception:
            logger.exception("Failed to ACK entry %s on %s", entry_id, stream_key)

    @staticmethod
    def _higher_priority(priority: TaskPriority) -> Optional[TaskPriority]:
        """Return the next-higher priority, or None if already CRITICAL."""
        mapping = {
            TaskPriority.BACKGROUND: TaskPriority.LOW,
            TaskPriority.LOW: TaskPriority.NORMAL,
            TaskPriority.NORMAL: TaskPriority.HIGH,
            TaskPriority.HIGH: TaskPriority.CRITICAL,
        }
        return mapping.get(priority)

    def get_schedule(self) -> List[str]:
        """Return the current weighted schedule as a list of priority names."""
        return [p.name for p in self._schedule]


__all__ = [
    "PriorityScheduler",
    "PrioritySchedulerConfig",
]
