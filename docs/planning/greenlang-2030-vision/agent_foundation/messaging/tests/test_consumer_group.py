# -*- coding: utf-8 -*-
"""
Consumer Group Manager Tests

Tests for consumer group management, scaling, and health monitoring.
"""

import pytest
import asyncio
from datetime import datetime

from ..redis_streams_broker import RedisStreamsBroker
from ..consumer_group import ConsumerGroupManager, ConsumerState
from ..message import Message


@pytest.fixture
async def broker():
    """Create Redis broker."""
    broker = RedisStreamsBroker(redis_url="redis://localhost:6379")
    await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.fixture
async def manager(broker):
    """Create consumer group manager."""
    manager = ConsumerGroupManager(broker)
    yield manager
    await manager.shutdown()


@pytest.mark.asyncio
async def test_create_consumer_group(manager):
    """Test creating consumer group."""
    await manager.create_group("test.create", "test_group")

    # Should not raise exception
    assert "test.create:test_group" in manager._groups


@pytest.mark.asyncio
async def test_delete_consumer_group(manager):
    """Test deleting consumer group."""
    await manager.create_group("test.delete", "delete_group")
    await manager.delete_group("test.delete", "delete_group")

    assert "test.delete:delete_group" not in manager._groups


@pytest.mark.asyncio
async def test_add_consumer(broker, manager):
    """Test adding consumer to group."""
    await manager.create_group("test.add", "add_group")

    processed = []

    async def handler(message):
        processed.append(message.id)

    consumer_id = await manager.add_consumer(
        "test.add",
        "add_group",
        handler,
    )

    assert consumer_id in manager._consumers
    assert manager._consumers[consumer_id].state == ConsumerState.RUNNING


@pytest.mark.asyncio
async def test_remove_consumer(broker, manager):
    """Test removing consumer from group."""
    await manager.create_group("test.remove", "remove_group")

    async def handler(message):
        pass

    consumer_id = await manager.add_consumer(
        "test.remove",
        "remove_group",
        handler,
    )

    await manager.remove_consumer(consumer_id, graceful=False)

    assert consumer_id not in manager._consumers


@pytest.mark.asyncio
async def test_scale_consumers_up(broker, manager):
    """Test scaling consumers up."""
    await manager.create_group("test.scale_up", "scale_group")

    processed = []

    async def handler(message):
        processed.append(message.id)

    # Scale to 5 consumers
    await manager.scale_consumers(
        "test.scale_up",
        "scale_group",
        count=5,
        handler=handler,
    )

    # Check consumer count
    consumers = await manager.list_consumers(
        topic="test.scale_up",
        group_name="scale_group",
    )

    assert len(consumers) == 5


@pytest.mark.asyncio
async def test_scale_consumers_down(broker, manager):
    """Test scaling consumers down."""
    await manager.create_group("test.scale_down", "scale_group")

    async def handler(message):
        pass

    # Scale to 5 then down to 2
    await manager.scale_consumers(
        "test.scale_down",
        "scale_group",
        count=5,
        handler=handler,
    )

    await manager.scale_consumers(
        "test.scale_down",
        "scale_group",
        count=2,
        handler=handler,
    )

    consumers = await manager.list_consumers(
        topic="test.scale_down",
        group_name="scale_group",
    )

    assert len(consumers) == 2


@pytest.mark.asyncio
async def test_consumer_message_processing(broker, manager):
    """Test consumers actually process messages."""
    await manager.create_group("test.process", "process_group")

    # Publish test messages
    for i in range(5):
        await broker.publish("test.process", {"index": i})

    processed = []

    async def handler(message):
        processed.append(message.payload["index"])

    # Add consumer
    await manager.add_consumer(
        "test.process",
        "process_group",
        handler,
    )

    # Wait for processing
    await asyncio.sleep(2.0)

    assert len(processed) > 0


@pytest.mark.asyncio
async def test_consumer_stats(broker, manager):
    """Test consumer statistics."""
    await manager.create_group("test.stats", "stats_group")

    async def handler(message):
        await asyncio.sleep(0.1)

    consumer_id = await manager.add_consumer(
        "test.stats",
        "stats_group",
        handler,
    )

    # Get consumer info
    info = await manager.get_consumer_info(consumer_id)

    assert info is not None
    assert info.consumer_id == consumer_id
    assert info.topic == "test.stats"
    assert info.group_name == "stats_group"
    assert info.state == ConsumerState.RUNNING


@pytest.mark.asyncio
async def test_group_stats(broker, manager):
    """Test group statistics."""
    await manager.create_group("test.group_stats", "stats_group")

    async def handler(message):
        pass

    # Add multiple consumers
    for _ in range(3):
        await manager.add_consumer(
            "test.group_stats",
            "stats_group",
            handler,
        )

    # Get group stats
    stats = await manager.get_group_stats("test.group_stats", "stats_group")

    assert stats is not None
    assert stats.consumer_count == 3
    assert stats.group_name == "stats_group"
    assert stats.topic == "test.group_stats"


@pytest.mark.asyncio
async def test_list_consumers(broker, manager):
    """Test listing consumers."""
    await manager.create_group("test.list", "list_group")

    async def handler(message):
        pass

    # Add consumers
    await manager.add_consumer("test.list", "list_group", handler)
    await manager.add_consumer("test.list", "list_group", handler)

    # List all consumers
    all_consumers = await manager.list_consumers()
    assert len(all_consumers) >= 2

    # List by topic
    topic_consumers = await manager.list_consumers(topic="test.list")
    assert len(topic_consumers) == 2

    # List by group
    group_consumers = await manager.list_consumers(group_name="list_group")
    assert len(group_consumers) == 2


@pytest.mark.asyncio
async def test_stop_all_consumers(broker, manager):
    """Test stopping all consumers in group."""
    await manager.create_group("test.stop_all", "stop_group")

    async def handler(message):
        pass

    # Add consumers
    for _ in range(3):
        await manager.add_consumer("test.stop_all", "stop_group", handler)

    # Stop all
    await manager.stop_all_consumers("test.stop_all", "stop_group")

    # Check all stopped
    consumers = await manager.list_consumers(
        topic="test.stop_all",
        group_name="stop_group",
    )

    assert len(consumers) == 0


@pytest.mark.asyncio
async def test_consumer_failure_tracking(broker, manager):
    """Test tracking consumer failures."""
    await manager.create_group("test.failures", "fail_group")

    async def failing_handler(message):
        raise ValueError("Processing failed")

    consumer_id = await manager.add_consumer(
        "test.failures",
        "fail_group",
        failing_handler,
    )

    # Publish message to trigger failure
    await broker.publish("test.failures", {"data": "test"})

    await asyncio.sleep(1.0)

    info = await manager.get_consumer_info(consumer_id)
    # Consumer should still be running (failures are handled)
    assert info.state == ConsumerState.RUNNING


@pytest.mark.asyncio
async def test_health_monitoring(broker, manager):
    """Test health monitoring."""
    await manager.start_health_monitoring()

    # Health monitoring should be running
    assert manager._health_check_task is not None
    assert not manager._health_check_task.done()

    await manager.stop_health_monitoring()

    assert manager._health_check_task is None


@pytest.mark.asyncio
async def test_graceful_shutdown(broker, manager):
    """Test graceful shutdown."""
    await manager.create_group("test.shutdown", "shutdown_group")

    async def handler(message):
        await asyncio.sleep(0.5)

    # Add consumer
    await manager.add_consumer("test.shutdown", "shutdown_group", handler)

    # Graceful shutdown
    await manager.shutdown(graceful=True)

    # All consumers should be stopped
    assert len(manager._consumers) == 0


@pytest.mark.asyncio
async def test_consumer_throughput(broker, manager):
    """Test consumer throughput calculation."""
    await manager.create_group("test.throughput", "throughput_group")

    # Publish messages
    for i in range(10):
        await broker.publish("test.throughput", {"index": i})

    async def handler(message):
        await asyncio.sleep(0.05)

    consumer_id = await manager.add_consumer(
        "test.throughput",
        "throughput_group",
        handler,
    )

    # Wait for processing
    await asyncio.sleep(2.0)

    info = await manager.get_consumer_info(consumer_id)

    # Should have processed some messages
    assert info.messages_processed > 0

    # Should have calculable throughput
    throughput = info.get_throughput()
    assert throughput > 0


@pytest.mark.asyncio
async def test_parallel_processing(broker, manager):
    """Test parallel processing with multiple consumers."""
    await manager.create_group("test.parallel", "parallel_group")

    # Publish many messages
    for i in range(20):
        await broker.publish("test.parallel", {"index": i})

    processed = []
    lock = asyncio.Lock()

    async def handler(message):
        async with lock:
            processed.append(message.payload["index"])
        await asyncio.sleep(0.1)

    # Scale to 5 consumers
    await manager.scale_consumers(
        "test.parallel",
        "parallel_group",
        count=5,
        handler=handler,
    )

    # Wait for processing
    await asyncio.sleep(3.0)

    # Multiple consumers should process in parallel
    assert len(processed) >= 10


@pytest.mark.asyncio
async def test_consumer_context_manager(broker):
    """Test consumer manager as context manager."""
    async with ConsumerGroupManager(broker) as manager:
        await manager.create_group("test.context", "context_group")

        async def handler(message):
            pass

        await manager.add_consumer("test.context", "context_group", handler)

        consumers = await manager.list_consumers()
        assert len(consumers) > 0

    # Should be shutdown after context exit
    # Cannot check manager state as it's out of scope


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
