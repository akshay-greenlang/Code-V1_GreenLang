# -*- coding: utf-8 -*-
"""
Load and performance tests for the Agent Factory.

Tests throughput for task queue operations, message bus delivery,
concurrent worker processing, and lifecycle state transitions
under load.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from tests.unit.agent_factory.test_task_queue import (
    Task,
    TaskQueue,
    WorkerPool,
)
from tests.unit.agent_factory.test_messaging_protocol import (
    MessageEnvelope,
    MessageRouter,
    RoutingPattern,
)
from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
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


class TestQueueThroughput:
    """Performance tests for task queue operations."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_queue_enqueue_throughput(
        self, queue: TaskQueue
    ) -> None:
        """Target: 1000+ tasks/sec enqueue throughput."""
        num_tasks = 5000
        tasks = [
            Task(agent_key="bench-agent", payload={"idx": i}, priority=i % 10)
            for i in range(num_tasks)
        ]

        start = time.perf_counter()
        for task in tasks:
            await queue.enqueue(task)
        elapsed = time.perf_counter() - start

        throughput = num_tasks / elapsed
        assert queue.pending_count == num_tasks
        assert throughput >= 1000, (
            f"Enqueue throughput {throughput:.0f} tasks/sec below target 1000"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_queue_dequeue_throughput(
        self, queue: TaskQueue
    ) -> None:
        """Target: 500+ tasks/sec dequeue throughput."""
        num_tasks = 2000
        for i in range(num_tasks):
            await queue.enqueue(
                Task(agent_key="bench-agent", payload={"idx": i})
            )

        start = time.perf_counter()
        dequeued = 0
        while True:
            task = await queue.dequeue()
            if task is None:
                break
            dequeued += 1
        elapsed = time.perf_counter() - start

        throughput = dequeued / elapsed
        assert dequeued == num_tasks
        assert throughput >= 500, (
            f"Dequeue throughput {throughput:.0f} tasks/sec below target 500"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_workers_throughput(self) -> None:
        """Workers process tasks concurrently at target throughput."""
        queue = TaskQueue()
        num_tasks = 1000
        processed = {"count": 0}

        async def fast_handler(task: Task) -> None:
            processed["count"] += 1

        for i in range(num_tasks):
            await queue.enqueue(
                Task(agent_key="a", payload={"idx": i})
            )

        start = time.perf_counter()
        pool = WorkerPool(queue, fast_handler, num_workers=5)
        await pool.start()

        # Wait until all processed or timeout
        timeout = 10.0
        while processed["count"] < num_tasks:
            await asyncio.sleep(0.01)
            if time.perf_counter() - start > timeout:
                break

        await pool.stop(graceful=False)
        elapsed = time.perf_counter() - start

        throughput = processed["count"] / elapsed
        assert processed["count"] == num_tasks, (
            f"Only {processed['count']}/{num_tasks} tasks processed"
        )
        assert throughput >= 100, (
            f"Worker throughput {throughput:.0f} tasks/sec"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_message_bus_throughput(self) -> None:
        """Message bus delivers messages at target throughput."""
        router = MessageRouter()
        delivered = {"count": 0}

        async def handler(m: MessageEnvelope) -> None:
            delivered["count"] += 1

        router.register_agent("target", handler)

        num_messages = 5000
        messages = [
            MessageEnvelope(
                source="producer",
                destination="target",
                payload={"idx": i},
                pattern=RoutingPattern.POINT_TO_POINT,
            )
            for i in range(num_messages)
        ]

        start = time.perf_counter()
        for msg in messages:
            await router.route(msg)
        elapsed = time.perf_counter() - start

        throughput = delivered["count"] / elapsed
        assert delivered["count"] == num_messages
        assert throughput >= 1000, (
            f"Message throughput {throughput:.0f} msg/sec below target 1000"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_lifecycle_state_transitions_under_load(self) -> None:
        """State machine handles rapid transitions under load."""
        num_machines = 100
        transitions_per_machine = 10

        machines = [
            AgentStateMachine(initial_state=AgentState.RUNNING)
            for _ in range(num_machines)
        ]

        start = time.perf_counter()
        total_transitions = 0

        for sm in machines:
            for _ in range(transitions_per_machine // 2):
                sm.transition(AgentState.DEGRADED, reason="test", actor="load")
                sm.transition(AgentState.RUNNING, reason="test", actor="load")
                total_transitions += 2

        elapsed = time.perf_counter() - start

        throughput = total_transitions / elapsed
        assert total_transitions == num_machines * transitions_per_machine
        assert throughput >= 10000, (
            f"Transition throughput {throughput:.0f}/sec below target 10000"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_enqueue_dequeue_interleaved(self) -> None:
        """Interleaved enqueue and dequeue maintains queue integrity."""
        queue = TaskQueue()
        enqueued = 0
        dequeued = 0

        async def producer():
            nonlocal enqueued
            for i in range(500):
                await queue.enqueue(Task(agent_key="a", payload={"i": i}))
                enqueued += 1
                if i % 10 == 0:
                    await asyncio.sleep(0)

        async def consumer():
            nonlocal dequeued
            while dequeued < 500:
                task = await queue.dequeue()
                if task:
                    await queue.acknowledge(task.task_id)
                    dequeued += 1
                else:
                    await asyncio.sleep(0.001)

        start = time.perf_counter()
        await asyncio.gather(producer(), consumer())
        elapsed = time.perf_counter() - start

        assert enqueued == 500
        assert dequeued == 500
        assert elapsed < 5.0, f"Interleaved operations took {elapsed:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pub_sub_fan_out_throughput(self) -> None:
        """Pub/sub fan-out to many subscribers at target throughput."""
        router = MessageRouter()
        ch = router.create_channel("load-test")
        num_subscribers = 50
        received_counts: Dict[str, int] = {}

        for i in range(num_subscribers):
            name = f"sub-{i}"
            received_counts[name] = 0

            async def handler(m: MessageEnvelope, n=name) -> None:
                received_counts[n] += 1

            ch.subscribe(name, handler)

        num_messages = 100
        start = time.perf_counter()
        for i in range(num_messages):
            msg = MessageEnvelope(
                source="producer",
                topic="load-test",
                payload={"idx": i},
                pattern=RoutingPattern.PUB_SUB,
            )
            await router.route(msg)
        elapsed = time.perf_counter() - start

        total_deliveries = sum(received_counts.values())
        expected = num_messages * num_subscribers
        assert total_deliveries == expected

        throughput = total_deliveries / elapsed
        assert throughput >= 1000, (
            f"Fan-out throughput {throughput:.0f} deliveries/sec"
        )
