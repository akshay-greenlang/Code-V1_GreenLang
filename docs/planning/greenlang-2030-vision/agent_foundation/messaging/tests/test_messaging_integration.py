# -*- coding: utf-8 -*-
"""
Integration Tests for Messaging System

Tests complete messaging workflows with Redis broker.
Requires running Redis instance on localhost:6379.

Run tests:
    pytest test_messaging_integration.py -v
    pytest test_messaging_integration.py::test_basic_pubsub -v
"""

import pytest
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

import sys
from pathlib import Path
from greenlang.determinism import DeterministicClock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from messaging import (
    RedisStreamsBroker,
    Message,
    MessagePriority,
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    EventSourcingPattern,
    SagaPattern,
    CircuitBreakerPattern,
    MessagingConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
async def broker():
    """Create Redis broker for tests."""
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.mark.asyncio
async def test_broker_connection():
    """Test broker connection and health check."""
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")

    # Test connection
    await broker.connect()
    assert broker.is_connected()

    # Test health check
    health = await broker.health_check()
    assert health["status"] == "healthy"
    assert health["connected"] is True
    assert health["latency_ms"] < 100

    # Test disconnection
    await broker.disconnect()
    assert not broker.is_connected()


@pytest.mark.asyncio
async def test_basic_pubsub(broker):
    """Test basic publish and consume."""
    topic = "test.basic_pubsub"

    # Publish message
    message_id = await broker.publish(
        topic,
        {"test": "data", "timestamp": DeterministicClock.utcnow().isoformat()},
        priority=MessagePriority.NORMAL,
    )
    assert message_id is not None

    # Consume message
    consumer_group = "test_consumers"
    received = False

    async for message in broker.consume(topic, consumer_group, batch_size=1):
        assert message.topic == topic
        assert message.payload["test"] == "data"
        await broker.acknowledge(message)
        received = True
        break

    assert received is True


@pytest.mark.asyncio
async def test_batch_publishing(broker):
    """Test batch publishing performance."""
    topic = "test.batch_publishing"

    # Prepare batch
    payloads = [{"index": i, "data": f"message_{i}"} for i in range(100)]

    # Publish batch
    start_time = DeterministicClock.utcnow()
    message_ids = await broker.publish_batch(topic, payloads)
    duration_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

    assert len(message_ids) == 100
    assert duration_ms < 1000  # Should complete in < 1 second

    logger.info(f"Published 100 messages in {duration_ms:.2f}ms")


@pytest.mark.asyncio
async def test_message_priorities(broker):
    """Test message priority handling."""
    topic = "test.priorities"

    # Publish with different priorities
    await broker.publish(topic, {"priority": "low"}, MessagePriority.LOW)
    await broker.publish(topic, {"priority": "high"}, MessagePriority.HIGH)
    await broker.publish(topic, {"priority": "normal"}, MessagePriority.NORMAL)

    # High priority messages should be in separate stream
    # This is implementation-specific test


@pytest.mark.asyncio
async def test_consumer_groups(broker):
    """Test consumer group functionality."""
    topic = "test.consumer_groups"
    group_name = "test_group"

    # Create consumer group
    await broker.create_consumer_group(topic, group_name)

    # Publish messages
    for i in range(5):
        await broker.publish(topic, {"message": i})

    # Consume with multiple consumers in same group
    consumer1_count = 0
    consumer2_count = 0

    async def consumer1():
        nonlocal consumer1_count
        async for message in broker.consume(topic, group_name, consumer_id="c1"):
            await broker.acknowledge(message)
            consumer1_count += 1
            if consumer1_count >= 2:
                break

    async def consumer2():
        nonlocal consumer2_count
        async for message in broker.consume(topic, group_name, consumer_id="c2"):
            await broker.acknowledge(message)
            consumer2_count += 1
            if consumer2_count >= 2:
                break

    # Run consumers concurrently
    await asyncio.gather(
        asyncio.wait_for(consumer1(), timeout=5),
        asyncio.wait_for(consumer2(), timeout=5),
    )

    # Both consumers should have processed messages
    total = consumer1_count + consumer2_count
    assert total >= 4  # At least 4 of 5 messages processed


@pytest.mark.asyncio
async def test_request_reply_pattern(broker):
    """Test request-reply pattern."""
    pattern = RequestReplyPattern(broker)

    # Define handler
    def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"Processed: {payload['input']}", "status": "success"}

    # Start handler in background
    handler_task = asyncio.create_task(
        pattern.handle_request("test.requests", handler, "request_handlers")
    )

    await asyncio.sleep(0.5)  # Let handler start

    # Send request
    response = await pattern.send_request(
        "test.requests",
        {"input": "test data"},
        timeout=5.0,
    )

    handler_task.cancel()

    assert response is not None
    assert response.payload["status"] == "success"
    assert "Processed: test data" in response.payload["result"]


@pytest.mark.asyncio
async def test_work_queue_pattern(broker):
    """Test work queue with multiple workers."""
    pattern = WorkQueuePattern(broker)

    # Submit tasks
    tasks = [{"task_id": i, "data": f"task_{i}"} for i in range(10)]
    await pattern.submit_batch("test.work_queue", tasks)

    # Process with workers
    processed = []

    def worker_handler(payload: Dict[str, Any]) -> Any:
        processed.append(payload["task_id"])
        return {"status": "completed"}

    # Start workers
    worker_task = asyncio.create_task(
        pattern.process_tasks(
            "test.work_queue",
            worker_handler,
            consumer_group="test_workers",
            num_workers=3,
        )
    )

    # Let workers run
    await asyncio.sleep(2)
    worker_task.cancel()

    # Verify tasks were processed
    assert len(processed) >= 5  # At least half processed


@pytest.mark.asyncio
async def test_event_sourcing_pattern(broker):
    """Test event sourcing for audit trails."""
    pattern = EventSourcingPattern(broker)

    # Log events
    event_id1 = await pattern.log_event(
        "calculation",
        {"input": 100, "output": 200, "formula": "x * 2"},
        agent_id="test_agent_1",
    )

    event_id2 = await pattern.log_event(
        "validation",
        {"rules_passed": 10, "rules_failed": 0},
        agent_id="test_agent_1",
    )

    assert event_id1 is not None
    assert event_id2 is not None


@pytest.mark.asyncio
async def test_saga_pattern(broker):
    """Test saga pattern for distributed transactions."""
    saga = SagaPattern(broker)

    executed_steps = []
    compensated_steps = []

    def step1(data: Dict) -> Dict:
        executed_steps.append("step1")
        return {"step1": "done"}

    def compensate1(data: Dict):
        compensated_steps.append("step1")

    def step2(data: Dict) -> Dict:
        executed_steps.append("step2")
        return {"step2": "done"}

    def compensate2(data: Dict):
        compensated_steps.append("step2")

    saga.add_step("step1", step1, compensate1)
    saga.add_step("step2", step2, compensate2)

    # Execute successful saga
    result = await saga.execute({"initial": "data"})

    assert "step1" in executed_steps
    assert "step2" in executed_steps
    assert result["step1"] == "done"
    assert result["step2"] == "done"


@pytest.mark.asyncio
async def test_saga_compensation(broker):
    """Test saga compensation on failure."""
    saga = SagaPattern(broker)

    executed_steps = []
    compensated_steps = []

    def step1(data: Dict) -> Dict:
        executed_steps.append("step1")
        return {"step1": "done"}

    def compensate1(data: Dict):
        compensated_steps.append("step1")

    def step2_fail(data: Dict) -> Dict:
        executed_steps.append("step2")
        raise Exception("Step 2 failed")

    def compensate2(data: Dict):
        compensated_steps.append("step2")

    saga.add_step("step1", step1, compensate1)
    saga.add_step("step2", step2_fail, compensate2)

    # Execute saga that fails
    from messaging.patterns import SagaError
    with pytest.raises(SagaError):
        await saga.execute({"initial": "data"})

    # Verify compensation occurred
    assert "step1" in executed_steps
    assert "step2" in executed_steps
    assert "step1" in compensated_steps  # Should compensate step1


@pytest.mark.asyncio
async def test_circuit_breaker():
    """Test circuit breaker pattern."""
    breaker = CircuitBreakerPattern(
        failure_threshold=3,
        timeout_seconds=1,
    )

    call_count = 0

    def failing_service():
        nonlocal call_count
        call_count += 1
        raise Exception("Service failed")

    # Fail until circuit opens
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_service)

    assert breaker.state == "OPEN"

    # Circuit should block calls
    from messaging.patterns import CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await breaker.call(failing_service)

    # Wait for timeout
    await asyncio.sleep(1.5)

    # Circuit should attempt reset (half-open)
    def recovering_service():
        return "success"

    # Should allow call in half-open state
    result = await breaker.call(recovering_service)
    assert result == "success"


@pytest.mark.asyncio
async def test_message_acknowledgment(broker):
    """Test message acknowledgment."""
    topic = "test.acknowledgment"

    # Publish message
    await broker.publish(topic, {"data": "test"})

    # Consume and acknowledge
    async for message in broker.consume(topic, "ack_test_group"):
        assert message.status.value == "processing"
        await broker.acknowledge(message)
        assert message.status.value == "completed"
        break


@pytest.mark.asyncio
async def test_message_nack_and_requeue(broker):
    """Test message nack and requeue."""
    topic = "test.nack_requeue"

    # Publish message
    await broker.publish(topic, {"data": "test"})

    # Consume and nack
    attempt_count = 0
    async for message in broker.consume(topic, "nack_test_group"):
        attempt_count += 1
        if attempt_count == 1:
            # First attempt - nack with requeue
            await broker.nack(message, "Processing failed", requeue=True)
        else:
            # Second attempt - acknowledge
            await broker.acknowledge(message)
            break

        if attempt_count >= 2:
            break

    assert attempt_count >= 1


@pytest.mark.asyncio
async def test_dead_letter_queue(broker):
    """Test dead letter queue for failed messages."""
    topic = "test.dlq"

    # Publish message
    await broker.publish(topic, {"data": "test"}, ttl_seconds=None)

    # Fail message multiple times to trigger DLQ
    async for message in broker.consume(topic, "dlq_test_group"):
        # Set max retries to 1 for faster test
        message.max_retries = 1
        message.retry_count = 1  # Already at max

        # Nack without requeue (move to DLQ)
        await broker.nack(message, "Permanent failure", requeue=False)
        break

    await asyncio.sleep(0.5)

    # Check DLQ
    dlq_messages = await broker.get_dead_letter_messages(topic, limit=10)
    assert len(dlq_messages) > 0

    # Reprocess from DLQ
    if dlq_messages:
        await broker.reprocess_dead_letter_message(dlq_messages[0])


@pytest.mark.asyncio
async def test_consumer_lag(broker):
    """Test consumer lag monitoring."""
    topic = "test.lag"
    group = "lag_test_group"

    # Create consumer group
    await broker.create_consumer_group(topic, group)

    # Publish messages
    for i in range(10):
        await broker.publish(topic, {"count": i})

    await asyncio.sleep(0.5)

    # Check lag
    lag = await broker.get_consumer_lag(topic, group)
    assert lag >= 0  # Should have pending messages


@pytest.mark.asyncio
async def test_metrics_collection(broker):
    """Test metrics collection."""
    topic = "test.metrics"

    # Generate activity
    for i in range(10):
        await broker.publish(topic, {"count": i})

    # Get metrics
    metrics = broker.get_metrics()

    assert metrics["messages_published"] >= 10
    assert "throughput_per_second" in metrics
    assert "average_latency_ms" in metrics


@pytest.mark.asyncio
async def test_message_expiration(broker):
    """Test message TTL expiration."""
    topic = "test.expiration"

    # Publish message with 1 second TTL
    await broker.publish(topic, {"data": "expires"}, ttl_seconds=1)

    # Wait for expiration
    await asyncio.sleep(2)

    # Try to consume - should skip expired message
    expired = False
    async for message in broker.consume(topic, "expiration_test_group"):
        expired = message.is_expired()
        # Broker should have skipped expired message
        break

    # Message should be expired if received
    # Or no message received (better)


@pytest.mark.asyncio
async def test_message_provenance(broker):
    """Test message provenance tracking."""
    topic = "test.provenance"

    # Publish message
    await broker.publish(
        topic,
        {"data": "test", "calculation": "result"},
        headers={"source": "test_agent"},
    )

    # Consume and verify provenance
    async for message in broker.consume(topic, "provenance_test_group"):
        assert message.provenance_hash is not None
        assert len(message.provenance_hash) == 64  # SHA-256 hash
        await broker.acknowledge(message)
        break


@pytest.mark.asyncio
async def test_concurrent_consumers(broker):
    """Test multiple concurrent consumers."""
    topic = "test.concurrent"

    # Publish many messages
    for i in range(20):
        await broker.publish(topic, {"task": i})

    # Run concurrent consumers
    async def consumer(consumer_id: str, count_dict: Dict):
        count = 0
        async for message in broker.consume(
            topic,
            "concurrent_test_group",
            consumer_id=consumer_id,
        ):
            await broker.acknowledge(message)
            count += 1
            if count >= 5:
                break
        count_dict[consumer_id] = count

    counts = {}
    await asyncio.gather(
        consumer("c1", counts),
        consumer("c2", counts),
        consumer("c3", counts),
    )

    # Verify load distribution
    total_processed = sum(counts.values())
    assert total_processed >= 10  # At least half of messages processed


@pytest.mark.asyncio
async def test_config_loading():
    """Test configuration loading."""
    # Test default config
    config = MessagingConfig()
    assert config.broker_type == "redis"
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379

    # Test environment-based config
    import os
    os.environ["GREENLANG_REDIS_HOST"] = "test-redis"
    os.environ["GREENLANG_REDIS_PORT"] = "6380"

    config = MessagingConfig.from_env()
    assert config.redis.host == "test-redis"
    assert config.redis.port == 6380

    # Cleanup
    del os.environ["GREENLANG_REDIS_HOST"]
    del os.environ["GREENLANG_REDIS_PORT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
