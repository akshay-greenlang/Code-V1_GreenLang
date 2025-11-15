"""
Messaging Pattern Tests

Tests for coordination patterns: request-reply, pub-sub, work queue, saga.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from ..redis_streams_broker import RedisStreamsBroker
from ..patterns import (
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    SagaPattern,
    CircuitBreakerPattern,
    SagaError,
    CircuitBreakerError,
)


@pytest.fixture
async def broker():
    """Create Redis broker."""
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()
    yield broker
    await broker.disconnect()


# Request-Reply Tests

@pytest.mark.asyncio
async def test_request_reply_success(broker):
    """Test successful request-reply."""
    pattern = RequestReplyPattern(broker)

    # Handler
    async def calculator(payload):
        a = payload["a"]
        b = payload["b"]
        return {"result": a + b}

    # Start handler
    handler_task = asyncio.create_task(
        pattern.handle_request("calc.add", calculator, "calculators")
    )

    await asyncio.sleep(0.5)  # Wait for handler

    # Send request
    response = await pattern.send_request(
        "calc.add",
        {"a": 10, "b": 32},
        timeout=5.0,
    )

    assert response is not None
    assert response.payload["result"] == 42

    handler_task.cancel()


@pytest.mark.asyncio
async def test_request_reply_timeout(broker):
    """Test request timeout."""
    pattern = RequestReplyPattern(broker)

    # Send request to non-existent handler
    response = await pattern.send_request(
        "nonexistent.service",
        {"data": "test"},
        timeout=1.0,
    )

    assert response is None  # Timeout


@pytest.mark.asyncio
async def test_request_reply_async_handler(broker):
    """Test with async handler."""
    pattern = RequestReplyPattern(broker)

    async def async_handler(payload):
        await asyncio.sleep(0.1)  # Simulate async work
        return {"processed": True}

    handler_task = asyncio.create_task(
        pattern.handle_request("async.service", async_handler, "async_handlers")
    )

    await asyncio.sleep(0.5)

    response = await pattern.send_request("async.service", {"data": "test"})
    assert response.payload["processed"] is True

    handler_task.cancel()


# Pub-Sub Tests

@pytest.mark.asyncio
async def test_pubsub_single_subscriber(broker):
    """Test pub-sub with single subscriber."""
    pattern = PubSubPattern(broker)
    received = []

    async def handler(message):
        received.append(message.payload)

    # Subscribe
    await pattern.subscribe("events.*", handler)
    await asyncio.sleep(0.5)

    # Publish
    await pattern.publish("events.test", {"event": "test_event"})
    await asyncio.sleep(0.5)

    assert len(received) > 0


@pytest.mark.asyncio
async def test_pubsub_multiple_subscribers(broker):
    """Test pub-sub with multiple subscribers."""
    pattern = PubSubPattern(broker)
    received1 = []
    received2 = []

    async def handler1(message):
        received1.append(message.payload)

    async def handler2(message):
        received2.append(message.payload)

    # Subscribe both
    await pattern.subscribe("multi.events.*", handler1)
    await pattern.subscribe("multi.events.*", handler2)
    await asyncio.sleep(0.5)

    # Publish
    await pattern.publish("multi.events.test", {"data": "broadcast"})
    await asyncio.sleep(0.5)

    # Both should receive
    assert len(received1) > 0
    assert len(received2) > 0


@pytest.mark.asyncio
async def test_pubsub_pattern_matching(broker):
    """Test pattern matching in pub-sub."""
    pattern = PubSubPattern(broker)
    received = []

    async def handler(message):
        received.append(message.payload["type"])

    # Subscribe to wildcard
    await pattern.subscribe("agent.*.status", handler)
    await asyncio.sleep(0.5)

    # Publish different topics
    await pattern.publish("agent.calculator.status", {"type": "calc"})
    await pattern.publish("agent.reporter.status", {"type": "report"})
    await asyncio.sleep(0.5)

    # Both should match pattern
    assert "calc" in received
    assert "report" in received


# Work Queue Tests

@pytest.mark.asyncio
async def test_work_queue_single_worker(broker):
    """Test work queue with single worker."""
    pattern = WorkQueuePattern(broker)
    processed = []

    async def worker_handler(payload):
        processed.append(payload["task_id"])

    # Submit tasks
    for i in range(5):
        await pattern.submit_task("tasks", {"task_id": i})

    # Process tasks
    async def run_worker():
        await pattern.process_tasks(
            "tasks",
            worker_handler,
            "workers",
            num_workers=1,
        )

    worker_task = asyncio.create_task(run_worker())
    await asyncio.sleep(2.0)
    worker_task.cancel()

    assert len(processed) == 5


@pytest.mark.asyncio
async def test_work_queue_multiple_workers(broker):
    """Test work queue with multiple workers."""
    pattern = WorkQueuePattern(broker)
    processed = []

    async def worker_handler(payload):
        await asyncio.sleep(0.1)  # Simulate work
        processed.append(payload["task_id"])

    # Submit batch
    await pattern.submit_batch(
        "parallel_tasks",
        [{"task_id": i} for i in range(20)],
    )

    # Process with 5 workers
    async def run_workers():
        await pattern.process_tasks(
            "parallel_tasks",
            worker_handler,
            "parallel_workers",
            num_workers=5,
        )

    worker_task = asyncio.create_task(run_workers())
    await asyncio.sleep(3.0)
    worker_task.cancel()

    # Multiple workers should process faster
    assert len(processed) >= 15  # Should process most tasks


@pytest.mark.asyncio
async def test_work_queue_priority(broker):
    """Test priority task handling."""
    pattern = WorkQueuePattern(broker)

    # Submit high priority task
    high_id = await pattern.submit_task(
        "priority_tasks",
        {"task": "urgent"},
        priority="high",
    )

    # Submit normal priority task
    normal_id = await pattern.submit_task(
        "priority_tasks",
        {"task": "regular"},
        priority="normal",
    )

    assert high_id != normal_id


# Saga Pattern Tests

@pytest.mark.asyncio
async def test_saga_success(broker):
    """Test successful saga execution."""
    pattern = SagaPattern(broker)
    execution_log = []

    def step1(data):
        execution_log.append("step1_execute")
        return {"step1_done": True}

    def step1_compensate(data):
        execution_log.append("step1_compensate")

    def step2(data):
        execution_log.append("step2_execute")
        return {"step2_done": True}

    def step2_compensate(data):
        execution_log.append("step2_compensate")

    pattern.add_step("step1", step1, step1_compensate)
    pattern.add_step("step2", step2, step2_compensate)

    result = await pattern.execute({"initial": "data"})

    assert "step1_execute" in execution_log
    assert "step2_execute" in execution_log
    assert "step1_done" in result
    assert "step2_done" in result


@pytest.mark.asyncio
async def test_saga_failure_compensation(broker):
    """Test saga compensation on failure."""
    pattern = SagaPattern(broker)
    execution_log = []

    def step1(data):
        execution_log.append("step1_execute")
        return {"step1_done": True}

    def step1_compensate(data):
        execution_log.append("step1_compensate")

    def step2(data):
        execution_log.append("step2_execute")
        raise ValueError("Step 2 failed!")

    def step2_compensate(data):
        execution_log.append("step2_compensate")

    pattern.add_step("step1", step1, step1_compensate)
    pattern.add_step("step2", step2, step2_compensate)

    with pytest.raises(SagaError):
        await pattern.execute({"initial": "data"})

    # Should execute step1, fail on step2, then compensate step1
    assert "step1_execute" in execution_log
    assert "step2_execute" in execution_log
    assert "step1_compensate" in execution_log


@pytest.mark.asyncio
async def test_saga_async_steps(broker):
    """Test saga with async steps."""
    pattern = SagaPattern(broker)

    async def async_step(data):
        await asyncio.sleep(0.1)
        return {"async_result": True}

    async def async_compensate(data):
        await asyncio.sleep(0.1)

    pattern.add_step("async", async_step, async_compensate)

    result = await pattern.execute({"data": "test"})
    assert result["async_result"] is True


# Circuit Breaker Tests

@pytest.mark.asyncio
async def test_circuit_breaker_closed(broker):
    """Test circuit breaker in CLOSED state."""
    breaker = CircuitBreakerPattern(failure_threshold=3)

    async def success_func():
        return "success"

    result = await breaker.call(success_func)
    assert result == "success"
    assert breaker.state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_opens(broker):
    """Test circuit breaker opens after failures."""
    breaker = CircuitBreakerPattern(failure_threshold=3)

    async def failing_func():
        raise ValueError("Operation failed")

    # Cause failures
    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(failing_func)

    # Circuit should be OPEN
    assert breaker.state == "OPEN"

    # Next call should be blocked
    with pytest.raises(CircuitBreakerError):
        await breaker.call(failing_func)


@pytest.mark.asyncio
async def test_circuit_breaker_half_open(broker):
    """Test circuit breaker HALF_OPEN state."""
    breaker = CircuitBreakerPattern(
        failure_threshold=2,
        timeout_seconds=1,
        half_open_max_calls=2,
    )

    call_count = [0]

    async def recovering_func():
        call_count[0] += 1
        if call_count[0] <= 2:
            raise ValueError("Still failing")
        return "recovered"

    # Open circuit
    for _ in range(2):
        with pytest.raises(ValueError):
            await breaker.call(recovering_func)

    assert breaker.state == "OPEN"

    # Wait for timeout
    await asyncio.sleep(1.5)

    # Should enter HALF_OPEN
    async def success_func():
        return "success"

    # Successful calls in HALF_OPEN
    for _ in range(2):
        await breaker.call(success_func)

    # Should close circuit
    assert breaker.state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_reset(broker):
    """Test manual circuit breaker reset."""
    breaker = CircuitBreakerPattern(failure_threshold=2)

    async def failing_func():
        raise ValueError("Failed")

    # Open circuit
    for _ in range(2):
        with pytest.raises(ValueError):
            await breaker.call(failing_func)

    assert breaker.state == "OPEN"

    # Manual reset
    breaker.reset()
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


# Integration Tests

@pytest.mark.asyncio
async def test_request_reply_with_circuit_breaker(broker):
    """Test request-reply pattern with circuit breaker."""
    pattern = RequestReplyPattern(broker)
    breaker = CircuitBreakerPattern(failure_threshold=3)

    async def handler(payload):
        return {"result": "success"}

    handler_task = asyncio.create_task(
        pattern.handle_request("cb.service", handler, "cb_handlers")
    )

    await asyncio.sleep(0.5)

    # Call through circuit breaker
    async def make_request():
        return await pattern.send_request("cb.service", {"data": "test"})

    response = await breaker.call(make_request)
    assert response is not None

    handler_task.cancel()


@pytest.mark.asyncio
async def test_work_queue_with_saga(broker):
    """Test work queue pattern with saga orchestration."""
    work_pattern = WorkQueuePattern(broker)
    saga = SagaPattern(broker)

    processed = []

    # Saga steps
    def validate(data):
        return {"validated": True}

    def compensate_validate(data):
        pass

    def process(data):
        processed.append(data)
        return {"processed": True}

    def compensate_process(data):
        processed.remove(data)

    saga.add_step("validate", validate, compensate_validate)
    saga.add_step("process", process, compensate_process)

    # Submit task that executes saga
    async def task_handler(payload):
        await saga.execute(payload)

    await work_pattern.submit_task("saga_tasks", {"data": "test"})

    async def run_worker():
        await work_pattern.process_tasks(
            "saga_tasks",
            task_handler,
            "saga_workers",
            num_workers=1,
        )

    worker_task = asyncio.create_task(run_worker())
    await asyncio.sleep(2.0)
    worker_task.cancel()

    assert len(processed) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
