# -*- coding: utf-8 -*-
"""
Redis Streams Broker Tests

Comprehensive test suite for Redis Streams broker implementation.
Tests all features: publish, consume, patterns, DLQ, health monitoring.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from ..redis_streams_broker import RedisStreamsBroker
from ..message import Message, MessagePriority, MessageStatus
from ..consumer_group import ConsumerGroupManager
from greenlang.determinism import DeterministicClock
from ..patterns import (
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    SagaPattern,
)


# Fixtures

@pytest.fixture
async def redis_broker():
    """Create Redis broker instance."""
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.fixture
async def consumer_manager(redis_broker):
    """Create consumer group manager."""
    manager = ConsumerGroupManager(redis_broker)
    yield manager
    await manager.shutdown()


# Basic Operations Tests

@pytest.mark.asyncio
async def test_broker_connection(redis_broker):
    """Test broker connection and health check."""
    assert redis_broker.is_connected()

    health = await redis_broker.health_check()
    assert health["status"] == "healthy"
    assert health["connected"] is True
    assert health["latency_ms"] < 100


@pytest.mark.asyncio
async def test_publish_message(redis_broker):
    """Test publishing single message."""
    message_id = await redis_broker.publish(
        topic="test.messages",
        payload={"data": "test"},
        priority=MessagePriority.NORMAL,
    )

    assert message_id is not None
    assert isinstance(message_id, str)


@pytest.mark.asyncio
async def test_publish_batch(redis_broker):
    """Test batch publishing."""
    payloads = [{"index": i, "data": f"message_{i}"} for i in range(100)]

    message_ids = await redis_broker.publish_batch(
        topic="test.batch",
        payloads=payloads,
        priority=MessagePriority.NORMAL,
    )

    assert len(message_ids) == 100
    assert all(isinstance(mid, str) for mid in message_ids)


@pytest.mark.asyncio
async def test_consume_messages(redis_broker):
    """Test consuming messages."""
    # Publish test messages
    await redis_broker.publish(
        "test.consume",
        {"data": "message_1"},
    )
    await redis_broker.publish(
        "test.consume",
        {"data": "message_2"},
    )

    # Consume messages
    messages_received = []

    async def consume():
        async for message in redis_broker.consume(
            "test.consume",
            "test_group",
            batch_size=2,
            timeout_ms=1000,
        ):
            messages_received.append(message)
            await redis_broker.acknowledge(message)

            if len(messages_received) >= 2:
                break

    await asyncio.wait_for(consume(), timeout=5.0)

    assert len(messages_received) == 2
    assert all(isinstance(m, Message) for m in messages_received)


@pytest.mark.asyncio
async def test_message_acknowledgment(redis_broker):
    """Test message acknowledgment."""
    # Publish message
    await redis_broker.publish("test.ack", {"data": "test"})

    # Consume and acknowledge
    async for message in redis_broker.consume("test.ack", "ack_group"):
        assert message.status == MessageStatus.PROCESSING
        await redis_broker.acknowledge(message)
        assert message.status == MessageStatus.COMPLETED
        break


@pytest.mark.asyncio
async def test_message_nack_and_requeue(redis_broker):
    """Test negative acknowledgment and requeue."""
    # Publish message
    await redis_broker.publish("test.nack", {"data": "test"})

    # Consume and nack
    async for message in redis_broker.consume("test.nack", "nack_group"):
        await redis_broker.nack(
            message,
            error_message="Processing failed",
            requeue=True,
        )
        break

    # Message should be requeued
    assert message.retry_count == 1
    assert message.status == MessageStatus.FAILED


@pytest.mark.asyncio
async def test_dead_letter_queue(redis_broker):
    """Test DLQ for failed messages."""
    # Publish message with max retries
    message_data = {"data": "will_fail"}

    for _ in range(4):  # max_retries = 3, so 4th will go to DLQ
        await redis_broker.publish("test.dlq", message_data)

        async for message in redis_broker.consume("test.dlq", "dlq_group"):
            await redis_broker.nack(
                message,
                error_message="Processing failed",
                requeue=True,
            )
            break

    # Check DLQ
    dlq_messages = await redis_broker.get_dead_letter_messages("test.dlq")
    assert len(dlq_messages) > 0


@pytest.mark.asyncio
async def test_message_priority(redis_broker):
    """Test priority message handling."""
    # Publish high priority message
    high_id = await redis_broker.publish(
        "test.priority",
        {"data": "high"},
        priority=MessagePriority.HIGH,
    )

    # Publish normal priority message
    normal_id = await redis_broker.publish(
        "test.priority",
        {"data": "normal"},
        priority=MessagePriority.NORMAL,
    )

    assert high_id != normal_id


@pytest.mark.asyncio
async def test_message_ttl(redis_broker):
    """Test message time-to-live."""
    # Publish message with TTL
    await redis_broker.publish(
        "test.ttl",
        {"data": "expires"},
        ttl_seconds=1,
    )

    # Wait for expiration
    await asyncio.sleep(2)

    # Message should be expired when consumed
    messages_received = []

    async def consume():
        async for message in redis_broker.consume("test.ttl", "ttl_group"):
            if message.is_expired():
                # Expired messages should be skipped
                pass
            else:
                messages_received.append(message)

            if len(messages_received) >= 1:
                break

    try:
        await asyncio.wait_for(consume(), timeout=3.0)
    except asyncio.TimeoutError:
        pass  # Expected if message expired

    # Should receive no messages (expired)
    assert len(messages_received) == 0


# Consumer Group Tests

@pytest.mark.asyncio
async def test_create_consumer_group(redis_broker):
    """Test creating consumer group."""
    await redis_broker.create_consumer_group("test.group", "my_group")
    # Should not raise exception


@pytest.mark.asyncio
async def test_delete_consumer_group(redis_broker):
    """Test deleting consumer group."""
    await redis_broker.create_consumer_group("test.delete", "delete_group")
    await redis_broker.delete_consumer_group("test.delete", "delete_group")
    # Should not raise exception


@pytest.mark.asyncio
async def test_consumer_lag(redis_broker):
    """Test consumer lag tracking."""
    # Publish messages
    for i in range(10):
        await redis_broker.publish("test.lag", {"index": i})

    # Create consumer group
    await redis_broker.create_consumer_group("test.lag", "lag_group")

    # Get lag
    lag = await redis_broker.get_consumer_lag("test.lag", "lag_group")
    assert lag >= 0


@pytest.mark.asyncio
async def test_consumer_manager_scaling(redis_broker, consumer_manager):
    """Test consumer group scaling."""
    # Create group
    await consumer_manager.create_group("test.scaling", "workers")

    # Add handler
    processed = []

    async def handler(message):
        processed.append(message.id)

    # Scale to 5 consumers
    await consumer_manager.scale_consumers(
        "test.scaling",
        "workers",
        count=5,
        handler=handler,
    )

    # Check consumer count
    stats = await consumer_manager.get_group_stats("test.scaling", "workers")
    assert stats.consumer_count == 5

    # Scale down to 2
    await consumer_manager.scale_consumers(
        "test.scaling",
        "workers",
        count=2,
        handler=handler,
    )

    stats = await consumer_manager.get_group_stats("test.scaling", "workers")
    assert stats.consumer_count == 2


# Pattern Tests

@pytest.mark.asyncio
async def test_request_reply_pattern(redis_broker):
    """Test request-reply pattern."""
    pattern = RequestReplyPattern(redis_broker)

    # Start request handler
    async def handle_request(payload):
        return {"result": payload["value"] * 2}

    handler_task = asyncio.create_task(
        pattern.handle_request("test.rpc", handle_request, "rpc_handlers")
    )

    # Wait for handler to start
    await asyncio.sleep(0.5)

    # Send request
    response = await pattern.send_request(
        "test.rpc",
        {"value": 21},
        timeout=5.0,
    )

    assert response is not None
    assert response.payload["result"] == 42

    # Cleanup
    handler_task.cancel()


@pytest.mark.asyncio
async def test_pubsub_pattern(redis_broker):
    """Test pub-sub pattern."""
    pattern = PubSubPattern(redis_broker)

    received_messages = []

    async def handler(message):
        received_messages.append(message)

    # Subscribe
    await pattern.subscribe("test.events.*", handler)

    # Publish
    await pattern.publish("test.events.update", {"event": "data_updated"})

    # Wait for delivery
    await asyncio.sleep(0.5)

    assert len(received_messages) > 0


@pytest.mark.asyncio
async def test_work_queue_pattern(redis_broker):
    """Test work queue pattern."""
    pattern = WorkQueuePattern(redis_broker)

    # Submit tasks
    task_ids = await pattern.submit_batch(
        "test.tasks",
        [{"task_id": i} for i in range(10)],
        priority=MessagePriority.NORMAL,
    )

    assert len(task_ids) == 10

    # Process tasks
    processed = []

    async def handler(payload):
        processed.append(payload["task_id"])

    # Create workers
    async def run_workers():
        await pattern.process_tasks(
            "test.tasks",
            handler,
            consumer_group="task_workers",
            num_workers=3,
        )

    worker_task = asyncio.create_task(run_workers())

    # Wait for processing
    await asyncio.sleep(2.0)

    # Stop workers
    worker_task.cancel()

    assert len(processed) > 0


@pytest.mark.asyncio
async def test_saga_pattern(redis_broker):
    """Test saga pattern."""
    pattern = SagaPattern(redis_broker)

    # Track execution
    executed_steps = []
    compensated_steps = []

    # Add saga steps
    def step1(data):
        executed_steps.append("step1")
        return {"step1_result": "success"}

    def compensate1(data):
        compensated_steps.append("step1")

    def step2(data):
        executed_steps.append("step2")
        raise ValueError("Step 2 failed")

    def compensate2(data):
        compensated_steps.append("step2")

    pattern.add_step("step1", step1, compensate1)
    pattern.add_step("step2", step2, compensate2)

    # Execute saga (should fail and compensate)
    with pytest.raises(Exception):
        await pattern.execute({"initial": "data"})

    assert "step1" in executed_steps
    assert "step1" in compensated_steps


# Performance Tests

@pytest.mark.asyncio
async def test_high_throughput(redis_broker):
    """Test high message throughput."""
    message_count = 1000
    payloads = [{"index": i} for i in range(message_count)]

    # Publish batch
    start_time = DeterministicClock.utcnow()
    await redis_broker.publish_batch("test.throughput", payloads)
    publish_duration = (DeterministicClock.utcnow() - start_time).total_seconds()

    # Calculate throughput
    throughput = message_count / publish_duration
    print(f"Publish throughput: {throughput:.2f} msg/s")

    assert throughput > 100  # Should be able to publish 100+ msg/s


@pytest.mark.asyncio
async def test_low_latency(redis_broker):
    """Test message latency."""
    latencies = []

    for _ in range(100):
        start = DeterministicClock.utcnow()
        await redis_broker.publish("test.latency", {"data": "test"})
        latency_ms = (DeterministicClock.utcnow() - start).total_seconds() * 1000
        latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[94]  # 95th percentile

    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")

    assert avg_latency < 50  # Average should be < 50ms
    assert p95_latency < 100  # P95 should be < 100ms


# Error Handling Tests

@pytest.mark.asyncio
async def test_connection_failure_handling(redis_broker):
    """Test handling of connection failures."""
    # Disconnect
    await redis_broker.disconnect()

    # Publishing should fail
    with pytest.raises(ConnectionError):
        await redis_broker.publish("test.fail", {"data": "test"})

    # Reconnect
    await redis_broker.connect()
    assert redis_broker.is_connected()


@pytest.mark.asyncio
async def test_invalid_topic_handling(redis_broker):
    """Test handling of invalid topics."""
    # Should raise validation error
    with pytest.raises(ValueError):
        await redis_broker.publish(
            "invalid topic with spaces!",
            {"data": "test"}
        )


@pytest.mark.asyncio
async def test_message_processing_error_recovery(redis_broker):
    """Test error recovery during message processing."""
    # Publish message
    await redis_broker.publish("test.error", {"data": "test"})

    # Consume with error
    async for message in redis_broker.consume("test.error", "error_group"):
        try:
            raise ValueError("Simulated processing error")
        except Exception as e:
            await redis_broker.nack(message, str(e), requeue=True)
            break

    # Message should be requeued
    assert message.retry_count > 0


# Metrics Tests

@pytest.mark.asyncio
async def test_broker_metrics(redis_broker):
    """Test broker metrics collection."""
    # Perform operations
    await redis_broker.publish("test.metrics", {"data": "test"})

    async for message in redis_broker.consume("test.metrics", "metrics_group"):
        await redis_broker.acknowledge(message)
        break

    # Get metrics
    metrics = redis_broker.get_metrics()

    assert metrics["messages_published"] > 0
    assert metrics["messages_consumed"] > 0
    assert "throughput_per_second" in metrics
    assert "average_latency_ms" in metrics


@pytest.mark.asyncio
async def test_consumer_group_stats(redis_broker, consumer_manager):
    """Test consumer group statistics."""
    # Create group and consumers
    await consumer_manager.create_group("test.stats", "stat_workers")

    async def handler(message):
        await asyncio.sleep(0.1)

    await consumer_manager.add_consumer(
        "test.stats",
        "stat_workers",
        handler,
    )

    # Get stats
    stats = await consumer_manager.get_group_stats("test.stats", "stat_workers")

    assert stats is not None
    assert stats.consumer_count == 1
    assert stats.group_name == "stat_workers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
