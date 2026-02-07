# -*- coding: utf-8 -*-
"""
Integration tests for Task Queue system.

Tests the full enqueue-dequeue cycle, priority ordering, dead letter queue
integration, worker pool processing, retry mechanics, deduplication,
and recovery scenarios.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

# Re-use the queue implementations from unit tests
from tests.unit.agent_factory.test_task_queue import (
    Task,
    TaskQueue,
    TaskStatus,
    WorkerPool,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def queue() -> TaskQueue:
    return TaskQueue()


# ============================================================================
# Tests
# ============================================================================


class TestQueueIntegration:
    """Integration tests for the task queue system."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_enqueue_dequeue_full_cycle(
        self, queue: TaskQueue
    ) -> None:
        """Full cycle: enqueue -> dequeue -> process -> acknowledge."""
        task = Task(agent_key="calc-agent", payload={"value": 42})
        await queue.enqueue(task)
        assert queue.pending_count == 1

        dequeued = await queue.dequeue()
        assert dequeued is not None
        assert dequeued.status == TaskStatus.IN_PROGRESS

        await queue.acknowledge(dequeued.task_id)
        assert queue.pending_count == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_priority_ordering_across_queues(
        self, queue: TaskQueue
    ) -> None:
        """Tasks are dequeued in strict priority order."""
        priorities = [1, 5, 3, 10, 2, 8]
        for p in priorities:
            await queue.enqueue(Task(agent_key="a", payload={}, priority=p))

        dequeued_priorities: List[int] = []
        while True:
            task = await queue.dequeue()
            if task is None:
                break
            dequeued_priorities.append(task.priority)
            await queue.acknowledge(task.task_id)

        assert dequeued_priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dlq_integration(self, queue: TaskQueue) -> None:
        """Tasks exhausting retries end up in the DLQ."""
        task = Task(agent_key="a", payload={}, max_retries=2)
        await queue.enqueue(task)

        for _ in range(2):
            t = await queue.dequeue()
            assert t is not None
            await queue.fail(t.task_id, "processing error")

        assert queue.pending_count == 0
        assert len(queue.dlq) == 1
        assert queue.dlq[0].status == TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_worker_pool_with_mock_agents(self) -> None:
        """Worker pool processes tasks using mock agent handlers."""
        queue = TaskQueue()
        results: Dict[str, Any] = {}

        async def handler(task: Task) -> None:
            results[task.task_id] = task.payload

        for i in range(5):
            await queue.enqueue(Task(agent_key="a", payload={"idx": i}))

        pool = WorkerPool(queue, handler, num_workers=2)
        await pool.start()
        await asyncio.sleep(0.2)
        await pool.stop(graceful=False)

        assert len(results) == 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_task_retry_and_eventual_dlq(
        self, queue: TaskQueue
    ) -> None:
        """Task retries on failure and eventually goes to DLQ."""
        task = Task(agent_key="a", payload={}, max_retries=3)
        await queue.enqueue(task)

        for i in range(3):
            t = await queue.dequeue()
            assert t is not None
            assert t.retry_count == i
            await queue.fail(t.task_id, f"error-{i}")

        assert len(queue.dlq) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_workers_dequeue_correctly(self) -> None:
        """Multiple concurrent workers do not process the same task."""
        queue = TaskQueue()
        processed_ids: List[str] = []
        lock = asyncio.Lock()

        async def handler(task: Task) -> None:
            async with lock:
                processed_ids.append(task.task_id)
            await asyncio.sleep(0.01)

        for _ in range(10):
            await queue.enqueue(Task(agent_key="a", payload={}))

        pool = WorkerPool(queue, handler, num_workers=3)
        await pool.start()
        await asyncio.sleep(0.3)
        await pool.stop(graceful=False)

        # No duplicates
        assert len(processed_ids) == len(set(processed_ids))
        assert len(processed_ids) == 10

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_task_deduplication(self, queue: TaskQueue) -> None:
        """Tasks with the same idempotency key are deduplicated."""
        t1 = Task(agent_key="a", payload={"v": 1}, idempotency_key="dedup-1")
        t2 = Task(agent_key="a", payload={"v": 2}, idempotency_key="dedup-1")

        assert await queue.enqueue(t1) is True
        assert await queue.enqueue(t2) is False
        assert queue.pending_count == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dlq_reprocess_back_to_queue(
        self, queue: TaskQueue
    ) -> None:
        """DLQ tasks can be reprocessed and succeed."""
        task = Task(agent_key="a", payload={"recover": True}, max_retries=1)
        await queue.enqueue(task)

        t = await queue.dequeue()
        assert t is not None
        await queue.fail(t.task_id, "first-time-fail")
        assert len(queue.dlq) == 1

        await queue.reprocess_dlq(task.task_id)
        assert len(queue.dlq) == 0
        assert queue.pending_count == 1

        retried = await queue.dequeue()
        assert retried is not None
        await queue.acknowledge(retried.task_id)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_queue_metrics_tracking(self, queue: TaskQueue) -> None:
        """Queue tracks pending count correctly through operations."""
        for i in range(5):
            await queue.enqueue(Task(agent_key="a", payload={"i": i}))
        assert queue.pending_count == 5

        t = await queue.dequeue()
        assert queue.pending_count == 4

        if t:
            await queue.acknowledge(t.task_id)
        assert queue.pending_count == 4

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expired_tasks_skipped_on_dequeue(
        self, queue: TaskQueue
    ) -> None:
        """Expired tasks are discarded during dequeue."""
        expired = Task(
            agent_key="a", payload={},
            ttl_seconds=0.0,
            created_at=time.time() - 100,
        )
        valid = Task(agent_key="a", payload={"valid": True})
        await queue.enqueue(expired)
        await queue.enqueue(valid)

        t = await queue.dequeue()
        assert t is not None
        assert t.payload.get("valid") is True
