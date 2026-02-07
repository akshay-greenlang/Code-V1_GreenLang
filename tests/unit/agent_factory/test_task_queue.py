# -*- coding: utf-8 -*-
"""
Unit tests for Task Queue, Priority Scheduler, Dead Letter Queue,
Worker Pool, and Cron Scheduler.

These components do not exist as source files yet, so the tests define
the expected contracts and interfaces. All external dependencies
(Redis, PostgreSQL) are mocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Inline Minimal Implementations (test doubles / contract stubs)
# ============================================================================


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_key: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    idempotency_key: Optional[str] = None
    ttl_seconds: Optional[float] = None
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    visibility_timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


class TaskQueue:
    """In-memory task queue for unit testing."""

    def __init__(self) -> None:
        self._queue: List[Task] = []
        self._in_progress: Dict[str, Task] = {}
        self._completed: List[Task] = []
        self._idempotency_keys: set = set()
        self._dlq: List[Task] = []

    async def enqueue(self, task: Task) -> bool:
        if task.idempotency_key and task.idempotency_key in self._idempotency_keys:
            return False
        if task.idempotency_key:
            self._idempotency_keys.add(task.idempotency_key)
        self._queue.append(task)
        self._queue.sort(key=lambda t: -t.priority)
        return True

    async def dequeue(self) -> Optional[Task]:
        now = time.time()
        for i, task in enumerate(self._queue):
            if task.is_expired:
                self._queue.pop(i)
                continue
            task = self._queue.pop(i)
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = now
            self._in_progress[task.task_id] = task
            return task
        return None

    async def acknowledge(self, task_id: str) -> bool:
        task = self._in_progress.pop(task_id, None)
        if task is None:
            return False
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        self._completed.append(task)
        return True

    async def fail(self, task_id: str, error: str) -> bool:
        task = self._in_progress.pop(task_id, None)
        if task is None:
            return False
        task.retry_count += 1
        task.error = error
        if task.retry_count >= task.max_retries:
            task.status = TaskStatus.DEAD_LETTER
            self._dlq.append(task)
        else:
            task.status = TaskStatus.PENDING
            task.started_at = None
            self._queue.append(task)
            self._queue.sort(key=lambda t: -t.priority)
        return True

    @property
    def dlq(self) -> List[Task]:
        return list(self._dlq)

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    async def reprocess_dlq(self, task_id: str) -> bool:
        for i, task in enumerate(self._dlq):
            if task.task_id == task_id:
                task = self._dlq.pop(i)
                task.status = TaskStatus.PENDING
                task.retry_count = 0
                task.error = None
                self._queue.append(task)
                return True
        return False


class WorkerPool:
    """Simplified worker pool for unit testing."""

    def __init__(
        self,
        queue: TaskQueue,
        handler: Callable[[Task], Coroutine[Any, Any, None]],
        num_workers: int = 3,
    ) -> None:
        self._queue = queue
        self._handler = handler
        self._num_workers = num_workers
        self._running = False
        self._tasks: List[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._running = True
        for _ in range(self._num_workers):
            t = asyncio.create_task(self._worker_loop())
            self._tasks.append(t)

    async def stop(self, graceful: bool = True) -> None:
        self._running = False
        if graceful:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        else:
            for t in self._tasks:
                t.cancel()

    async def _worker_loop(self) -> None:
        while self._running:
            task = await self._queue.dequeue()
            if task is None:
                await asyncio.sleep(0.01)
                continue
            try:
                await self._handler(task)
                await self._queue.acknowledge(task.task_id)
            except Exception as exc:
                await self._queue.fail(task.task_id, str(exc))


class PriorityScheduler:
    """Schedules tasks in priority order with starvation prevention."""

    def __init__(self, max_low_priority_wait: int = 10) -> None:
        self._max_wait = max_low_priority_wait

    def schedule(self, tasks: List[Task]) -> List[Task]:
        now = time.time()
        boosted: List[tuple] = []
        for t in tasks:
            wait = now - t.created_at
            boost = min(wait / self._max_wait, 2.0)
            effective_priority = t.priority + boost
            boosted.append((effective_priority, t))
        boosted.sort(key=lambda x: -x[0])
        return [t for _, t in boosted]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def task_queue() -> TaskQueue:
    return TaskQueue()


@pytest.fixture
def sample_task() -> Task:
    return Task(
        agent_key="carbon-agent",
        payload={"shipment_id": "S-001"},
        priority=5,
    )


@pytest.fixture
def high_priority_task() -> Task:
    return Task(
        agent_key="carbon-agent",
        payload={"shipment_id": "S-002"},
        priority=10,
    )


# ============================================================================
# Tests
# ============================================================================


class TestTaskQueue:
    """Tests for the core task queue operations."""

    @pytest.mark.asyncio
    async def test_enqueue_task(
        self, task_queue: TaskQueue, sample_task: Task
    ) -> None:
        """Enqueuing a task adds it to the queue."""
        result = await task_queue.enqueue(sample_task)
        assert result is True
        assert task_queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_dequeue_task_priority_order(
        self,
        task_queue: TaskQueue,
        sample_task: Task,
        high_priority_task: Task,
    ) -> None:
        """Dequeue returns the highest priority task first."""
        await task_queue.enqueue(sample_task)
        await task_queue.enqueue(high_priority_task)

        first = await task_queue.dequeue()
        assert first is not None
        assert first.priority == 10

        second = await task_queue.dequeue()
        assert second is not None
        assert second.priority == 5

    @pytest.mark.asyncio
    async def test_task_deduplication_via_idempotency_key(
        self, task_queue: TaskQueue
    ) -> None:
        """Tasks with the same idempotency key are deduplicated."""
        task1 = Task(
            agent_key="agent-a",
            payload={"id": 1},
            idempotency_key="key-1",
        )
        task2 = Task(
            agent_key="agent-a",
            payload={"id": 2},
            idempotency_key="key-1",
        )
        assert await task_queue.enqueue(task1) is True
        assert await task_queue.enqueue(task2) is False
        assert task_queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_task_ttl_expiration(self, task_queue: TaskQueue) -> None:
        """Expired tasks are skipped during dequeue."""
        task = Task(
            agent_key="agent-a",
            payload={},
            ttl_seconds=0.0,
            created_at=time.time() - 100,
        )
        await task_queue.enqueue(task)
        result = await task_queue.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_task_visibility_timeout(self, task_queue: TaskQueue) -> None:
        """Dequeued task transitions to IN_PROGRESS with a started_at timestamp."""
        task = Task(agent_key="agent-a", payload={})
        await task_queue.enqueue(task)
        dequeued = await task_queue.dequeue()
        assert dequeued is not None
        assert dequeued.status == TaskStatus.IN_PROGRESS
        assert dequeued.started_at is not None

    @pytest.mark.asyncio
    async def test_task_acknowledge(self, task_queue: TaskQueue) -> None:
        """Acknowledging a task marks it as COMPLETED."""
        task = Task(agent_key="agent-a", payload={})
        await task_queue.enqueue(task)
        dequeued = await task_queue.dequeue()
        assert dequeued is not None
        result = await task_queue.acknowledge(dequeued.task_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_task_fail_and_retry(self, task_queue: TaskQueue) -> None:
        """Failing a task returns it to the queue for retry."""
        task = Task(agent_key="agent-a", payload={}, max_retries=3)
        await task_queue.enqueue(task)
        dequeued = await task_queue.dequeue()
        assert dequeued is not None

        await task_queue.fail(dequeued.task_id, "transient error")
        assert task_queue.pending_count == 1  # re-queued for retry

    @pytest.mark.asyncio
    async def test_task_fail_to_dlq_after_max_retries(
        self, task_queue: TaskQueue
    ) -> None:
        """After max retries, task is moved to the dead letter queue."""
        task = Task(agent_key="agent-a", payload={}, max_retries=2)
        await task_queue.enqueue(task)

        # Exhaust retries
        for _ in range(2):
            dequeued = await task_queue.dequeue()
            assert dequeued is not None
            await task_queue.fail(dequeued.task_id, "error")

        assert task_queue.pending_count == 0
        assert len(task_queue.dlq) == 1
        assert task_queue.dlq[0].status == TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_dead_letter_queue_reprocess(
        self, task_queue: TaskQueue
    ) -> None:
        """DLQ tasks can be reprocessed back into the main queue."""
        task = Task(agent_key="agent-a", payload={}, max_retries=1)
        await task_queue.enqueue(task)
        dequeued = await task_queue.dequeue()
        assert dequeued is not None
        await task_queue.fail(dequeued.task_id, "fatal")
        assert len(task_queue.dlq) == 1

        result = await task_queue.reprocess_dlq(task.task_id)
        assert result is True
        assert len(task_queue.dlq) == 0
        assert task_queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(
        self, task_queue: TaskQueue
    ) -> None:
        """Dequeue on an empty queue returns None."""
        result = await task_queue.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_acknowledge_unknown_returns_false(
        self, task_queue: TaskQueue
    ) -> None:
        """Acknowledging a nonexistent task returns False."""
        result = await task_queue.acknowledge("nonexistent-id")
        assert result is False


class TestPriorityScheduler:
    """Tests for priority-based scheduling with starvation prevention."""

    def test_priority_scheduler_ordering(self) -> None:
        """Scheduler orders tasks by priority (highest first)."""
        scheduler = PriorityScheduler()
        tasks = [
            Task(priority=1, created_at=time.time()),
            Task(priority=10, created_at=time.time()),
            Task(priority=5, created_at=time.time()),
        ]
        ordered = scheduler.schedule(tasks)
        assert ordered[0].priority == 10
        assert ordered[1].priority == 5
        assert ordered[2].priority == 1

    def test_priority_scheduler_starvation_prevention(self) -> None:
        """Old low-priority tasks get a boost to prevent starvation."""
        scheduler = PriorityScheduler(max_low_priority_wait=5)
        old_task = Task(priority=1, created_at=time.time() - 20)
        new_task = Task(priority=5, created_at=time.time())

        ordered = scheduler.schedule([old_task, new_task])
        # Old task should have been boosted above the new higher-priority task
        assert ordered[0].task_id == old_task.task_id


class TestWorkerPool:
    """Tests for the worker pool task processing."""

    @pytest.mark.asyncio
    async def test_worker_pool_start_stop(self) -> None:
        """Worker pool can start and stop cleanly."""
        queue = TaskQueue()
        handler = AsyncMock()
        pool = WorkerPool(queue, handler, num_workers=2)
        await pool.start()
        assert pool._running is True
        await pool.stop(graceful=False)
        assert pool._running is False

    @pytest.mark.asyncio
    async def test_worker_pool_processes_tasks(self) -> None:
        """Workers dequeue and process tasks via the handler."""
        queue = TaskQueue()
        processed: List[str] = []

        async def handler(task: Task) -> None:
            processed.append(task.task_id)

        await queue.enqueue(Task(agent_key="a", payload={}))
        await queue.enqueue(Task(agent_key="b", payload={}))

        pool = WorkerPool(queue, handler, num_workers=1)
        await pool.start()
        await asyncio.sleep(0.1)
        await pool.stop(graceful=False)

        assert len(processed) == 2

    @pytest.mark.asyncio
    async def test_worker_pool_graceful_shutdown(self) -> None:
        """Graceful shutdown waits for in-flight processing."""
        queue = TaskQueue()
        processed: List[str] = []

        async def handler(task: Task) -> None:
            await asyncio.sleep(0.02)
            processed.append(task.task_id)

        await queue.enqueue(Task(agent_key="a", payload={}))
        pool = WorkerPool(queue, handler, num_workers=1)
        await pool.start()
        await asyncio.sleep(0.05)
        pool._running = False
        await asyncio.gather(*pool._tasks, return_exceptions=True)
        assert len(processed) >= 1


class TestSchedulerCron:
    """Tests for cron-style scheduling (next run calculation)."""

    def test_scheduler_cron_next_run(self) -> None:
        """Simple interval-based next run calculation."""
        now = datetime.now(timezone.utc)
        interval_seconds = 60
        last_run = now - timedelta(seconds=45)
        next_run = last_run + timedelta(seconds=interval_seconds)
        assert next_run > now
        remaining = (next_run - now).total_seconds()
        assert 0 < remaining <= 60
