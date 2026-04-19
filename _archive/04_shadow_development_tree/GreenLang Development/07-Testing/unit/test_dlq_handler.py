"""
Unit Tests for Dead Letter Queue Handler

Tests for DLQ handler functionality including:
- Message routing to DLQ topics
- Redis retry tracking
- Exponential backoff calculations
- Error categorization
- Batch processing
- Statistics and alerting
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from greenlang.infrastructure.events.dlq_handler import (
    DeadLetterQueueHandler,
    DLQHandlerConfig,
    DLQMessage,
    DLQStats,
    ErrorCategory,
    DLQMessageStatus,
)


class TestDLQMessage:
    """Test DLQMessage model."""

    def test_create_dlq_message(self):
        """Test creating a DLQ message."""
        msg = DLQMessage(
            original_queue="emissions",
            message_body={"value": 100},
            error_message="Connection timeout"
        )

        assert msg.original_queue == "emissions"
        assert msg.error_message == "Connection timeout"
        assert msg.retry_count == 0
        assert msg.status == DLQMessageStatus.PENDING
        assert msg.message_id is not None

    def test_dlq_message_validation(self):
        """Test DLQ message validation."""
        with pytest.raises(ValueError):
            DLQMessage(
                original_queue="",  # Invalid empty queue name
                message_body={},
                error_message="error"
            )

    def test_calculate_next_retry(self):
        """Test exponential backoff calculation."""
        msg = DLQMessage(
            original_queue="test",
            message_body={},
            error_message="error"
        )

        # First retry: 60 seconds
        msg.retry_count = 0
        retry1 = msg.calculate_next_retry(
            base_delay=60,
            multiplier=5.0,
            max_delay=3600
        )
        now = datetime.utcnow()
        assert 59 <= (retry1 - now).total_seconds() <= 61

        # Second retry: 300 seconds (60 * 5)
        msg.retry_count = 1
        retry2 = msg.calculate_next_retry(
            base_delay=60,
            multiplier=5.0,
            max_delay=3600
        )
        assert 299 <= (retry2 - now).total_seconds() <= 301

        # Third retry: 1500 seconds (300 * 5)
        msg.retry_count = 2
        retry3 = msg.calculate_next_retry(
            base_delay=60,
            multiplier=5.0,
            max_delay=3600
        )
        assert 1499 <= (retry3 - now).total_seconds() <= 1501

        # Exceeds max, should cap at max_delay
        msg.retry_count = 3
        retry4 = msg.calculate_next_retry(
            base_delay=60,
            multiplier=5.0,
            max_delay=3600
        )
        delay = (retry4 - now).total_seconds()
        assert 3599 <= delay <= 3601


class TestDLQHandlerConfig:
    """Test DLQHandlerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DLQHandlerConfig()

        assert config.max_retries == 3
        assert config.initial_backoff_seconds == 60
        assert config.backoff_multiplier == 5.0
        assert config.retention_days == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = DLQHandlerConfig(
            max_retries=5,
            initial_backoff_seconds=120,
            retention_days=60
        )

        assert config.max_retries == 5
        assert config.initial_backoff_seconds == 120
        assert config.retention_days == 60


# Async tests
@pytest.mark.asyncio
async def test_initialization():
    """Test handler initialization."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    handler = DeadLetterQueueHandler(config)

    assert handler.config == config
    assert not handler._started
    assert not handler._shutdown


@pytest.mark.asyncio
async def test_start_stop():
    """Test start and stop."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        max_retries=3,
        initial_backoff_seconds=10,
    )
    handler = DeadLetterQueueHandler(config)
    await handler.start()

    assert handler._started
    assert not handler._shutdown

    await handler.stop()
    assert not handler._started


@pytest.mark.asyncio
async def test_send_to_dlq():
    """Test sending message to DLQ."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        message = {"temp": 95.5, "pressure": 2.0}
        error = Exception("Processing failed")

        msg_id = await handler.send_to_dlq(
            message=message,
            error=error,
            original_queue="heat-processing",
            error_category=ErrorCategory.TRANSIENT
        )

        assert msg_id is not None
        assert msg_id in handler._local_storage

        stored_msg = handler._local_storage[msg_id]
        assert stored_msg.original_queue == "heat-processing"
        assert stored_msg.message_body == message
        assert stored_msg.error_category == ErrorCategory.TRANSIENT


@pytest.mark.asyncio
async def test_get_dlq_stats():
    """Test getting DLQ statistics."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        # Send messages to DLQ
        await handler.send_to_dlq(
            {"data": 1},
            Exception("error1"),
            "queue1"
        )
        await handler.send_to_dlq(
            {"data": 2},
            Exception("error2"),
            "queue2"
        )

        stats = await handler.get_dlq_stats()

        assert stats.total_pending == 2
        assert stats.total_messages_processed == 2
        assert "queue1" in stats.pending_by_queue
        assert "queue2" in stats.pending_by_queue


@pytest.mark.asyncio
async def test_process_dlq_with_handler():
    """Test processing DLQ messages with handler."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        initial_backoff_seconds=0  # No backoff for testing
    )
    async with DeadLetterQueueHandler(config) as handler:
        # Send a message
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        # Register handler that succeeds
        def success_handler(msg: DLQMessage) -> bool:
            return True

        handler.register_handler("emissions", success_handler)

        # Process messages immediately (message will be ready since backoff=0)
        processed = await handler.process_dlq(success_handler)

        assert processed == 1
        stored_msg = handler._local_storage[msg_id]
        assert stored_msg.status == DLQMessageStatus.RESOLVED


@pytest.mark.asyncio
async def test_process_dlq_with_failing_handler():
    """Test processing with handler that fails."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        initial_backoff_seconds=0  # No backoff for testing
    )
    async with DeadLetterQueueHandler(config) as handler:
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        # Handler that fails
        def failing_handler(msg: DLQMessage) -> bool:
            return False

        # Process messages (will attempt once since next_retry is now)
        processed = await handler.process_dlq(failing_handler)

        assert processed == 0
        stored_msg = handler._local_storage[msg_id]
        assert stored_msg.status == DLQMessageStatus.PENDING
        assert stored_msg.retry_count == 1


@pytest.mark.asyncio
async def test_max_retries_escalation():
    """Test message escalation after max retries."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        max_retries=2,
        initial_backoff_seconds=0  # No backoff so messages are ready immediately
    )
    async with DeadLetterQueueHandler(config) as handler:
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        # Handler that always fails
        def failing_handler(msg: DLQMessage) -> bool:
            return False

        # Process max_retries + 1 times (3 times since max_retries=2)
        for i in range(handler.config.max_retries + 1):
            await handler.process_dlq(failing_handler)

        stored_msg = handler._local_storage[msg_id]
        assert stored_msg.status == DLQMessageStatus.ESCALATED
        assert stored_msg.retry_count == handler.config.max_retries


@pytest.mark.asyncio
async def test_purge_old_messages():
    """Test purging old resolved messages."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        # Send message
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        # Manually resolve it
        msg = handler._local_storage[msg_id]
        msg.status = DLQMessageStatus.RESOLVED
        msg.resolved_at = datetime.utcnow() - timedelta(days=35)

        # Purge messages older than 30 days
        purged = await handler.purge_dlq(older_than_days=30)

        assert purged == 1
        assert msg_id not in handler._local_storage


@pytest.mark.asyncio
async def test_purge_keeps_recent_messages():
    """Test that recent messages are not purged."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        msg = handler._local_storage[msg_id]
        msg.status = DLQMessageStatus.RESOLVED
        msg.resolved_at = datetime.utcnow() - timedelta(days=5)

        purged = await handler.purge_dlq(older_than_days=30)

        assert purged == 0
        assert msg_id in handler._local_storage


@pytest.mark.asyncio
async def test_configure_dlq():
    """Test DLQ configuration."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        await handler.configure_dlq(
            "emissions",
            max_retries=5,
            retry_delay_seconds=120
        )
        assert handler._started


@pytest.mark.asyncio
async def test_alert_callback():
    """Test alert callbacks."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        dlq_depth_threshold=5
    )
    async with DeadLetterQueueHandler(config) as handler:
        alerts_triggered = []

        async def alert_callback(stats: DLQStats) -> None:
            alerts_triggered.append(stats)

        handler.add_alert_callback(alert_callback)

        # Send messages to exceed threshold
        for i in range(handler.config.dlq_depth_threshold + 1):
            await handler.send_to_dlq(
                {"num": i},
                Exception(f"error {i}"),
                "queue"
            )

        # Alert should be triggered
        assert len(alerts_triggered) > 0


@pytest.mark.asyncio
async def test_handler_registration():
    """Test handler registration."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        def custom_handler(msg: DLQMessage) -> bool:
            return True

        handler.register_handler("emissions", custom_handler)

        assert "emissions" in handler._retry_handlers
        assert handler._retry_handlers["emissions"] == custom_handler


@pytest.mark.asyncio
async def test_async_handler():
    """Test async handler support."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        initial_backoff_seconds=0  # No backoff
    )
    async with DeadLetterQueueHandler(config) as handler:
        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("test error"),
            "emissions"
        )

        # Async handler
        async def async_handler(msg: DLQMessage) -> bool:
            await asyncio.sleep(0.01)
            return True

        processed = await handler.process_dlq(async_handler)

        assert processed == 1
        assert handler._local_storage[msg_id].status == DLQMessageStatus.RESOLVED


@pytest.mark.asyncio
async def test_error_categorization():
    """Test error categorization."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        msg_id_transient = await handler.send_to_dlq(
            {"value": 1},
            Exception("timeout"),
            "emissions",
            error_category=ErrorCategory.TRANSIENT
        )

        msg_id_permanent = await handler.send_to_dlq(
            {"value": 2},
            Exception("validation error"),
            "emissions",
            error_category=ErrorCategory.PERMANENT
        )

        assert handler._local_storage[msg_id_transient].error_category == ErrorCategory.TRANSIENT
        assert handler._local_storage[msg_id_permanent].error_category == ErrorCategory.PERMANENT


@pytest.mark.asyncio
async def test_metadata_preservation():
    """Test metadata is preserved through DLQ."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        message = {"temp": 95.5}
        metadata = {"agent_id": "gl_010", "batch_id": "batch_123"}

        msg_id = await handler.send_to_dlq(
            message,
            Exception("error"),
            "heat",
            metadata=metadata
        )

        stored_msg = handler._local_storage[msg_id]
        assert stored_msg.metadata == metadata


@pytest.mark.asyncio
async def test_queue_name_isolation():
    """Test messages are isolated by queue."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)
    async with DeadLetterQueueHandler(config) as handler:
        await handler.send_to_dlq(
            {"value": 1},
            Exception("error1"),
            "queue1"
        )
        await handler.send_to_dlq(
            {"value": 2},
            Exception("error2"),
            "queue2"
        )

        stats = await handler.get_dlq_stats()
        assert stats.pending_by_queue["queue1"] == 1
        assert stats.pending_by_queue["queue2"] == 1


@pytest.mark.asyncio
async def test_context_manager():
    """Test context manager usage."""
    config = DLQHandlerConfig(kafka_enabled=False, redis_enabled=False)

    async with DeadLetterQueueHandler(config) as handler:
        assert handler._started

        msg_id = await handler.send_to_dlq(
            {"value": 100},
            Exception("error"),
            "test"
        )
        assert msg_id in handler._local_storage

    # Should be stopped after exiting context
    # Trying to use it should raise RuntimeError
    with pytest.raises(RuntimeError):
        await handler.send_to_dlq(
            {"value": 100},
            Exception("error"),
            "test"
        )


class TestDeadLetterQueueHandler:
    """Test class for compatibility (skipped)."""
    pass


class TestDLQStats:
    """Test DLQStats model."""

    def test_create_stats(self):
        """Test creating DLQ stats."""
        stats = DLQStats(
            total_pending=10,
            total_escalated=2,
            total_resolved=5
        )

        assert stats.total_pending == 10
        assert stats.total_escalated == 2
        assert stats.total_resolved == 5

    def test_stats_defaults(self):
        """Test stats default values."""
        stats = DLQStats()

        assert stats.total_pending == 0
        assert stats.total_escalated == 0
        assert stats.total_resolved == 0
        assert stats.oldest_pending_message_age_seconds is None


# Integration tests
@pytest.mark.asyncio
async def test_dlq_workflow():
    """Test complete DLQ workflow."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        max_retries=2,
        initial_backoff_seconds=0  # No backoff for testing
    )

    async with DeadLetterQueueHandler(config) as handler:
        # Send message to DLQ
        msg_id = await handler.send_to_dlq(
            {"temp": 95},
            Exception("sensor error"),
            "temperature",
            error_category=ErrorCategory.TRANSIENT
        )

        # Verify message in DLQ
        stats = await handler.get_dlq_stats()
        assert stats.total_pending == 1

        # First attempt fails (message is ready since backoff=0)
        attempt_count = [0]

        def handler_func(msg: DLQMessage) -> bool:
            attempt_count[0] += 1
            return attempt_count[0] >= 2  # Succeed on second attempt

        # First process attempt fails
        processed = await handler.process_dlq(handler_func, max_messages=10)
        assert processed == 0
        assert handler._local_storage[msg_id].retry_count == 1

        # Message is still ready for second attempt (with zero backoff)
        # Second process succeeds
        processed = await handler.process_dlq(handler_func, max_messages=10)
        assert processed == 1
        assert handler._local_storage[msg_id].status == DLQMessageStatus.RESOLVED


@pytest.mark.asyncio
async def test_dlq_batch_processing():
    """Test batch processing of DLQ messages."""
    config = DLQHandlerConfig(
        kafka_enabled=False,
        redis_enabled=False,
        max_retries=1,
        initial_backoff_seconds=0  # No backoff for testing
    )

    async with DeadLetterQueueHandler(config) as handler:
        # Send multiple messages
        msg_ids = []
        for i in range(5):
            msg_id = await handler.send_to_dlq(
                {"id": i},
                Exception(f"error {i}"),
                "batch-queue"
            )
            msg_ids.append(msg_id)

        # Process with limit (all are ready since backoff=0)
        def simple_handler(msg: DLQMessage) -> bool:
            return True

        processed = await handler.process_dlq(simple_handler, max_messages=3)
        assert processed == 3

        # Verify stats
        stats = await handler.get_dlq_stats()
        assert stats.total_resolved == 3
