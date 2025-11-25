# -*- coding: utf-8 -*-
"""
Message Model Tests

Tests for message data models, serialization, and validation.
"""

import pytest
import json
from datetime import datetime, timedelta

from greenlang.determinism import DeterministicClock
from ..message import (
    Message,
    MessageBatch,
    MessageAck,
    DeadLetterMessage,
    MessagePriority,
    MessageStatus,
)


class TestMessage:
    """Test Message model."""

    def test_create_message(self):
        """Test creating a message."""
        message = Message(
            topic="test.topic",
            payload={"key": "value"},
        )

        assert message.id is not None
        assert message.topic == "test.topic"
        assert message.payload == {"key": "value"}
        assert message.priority == MessagePriority.NORMAL
        assert message.status == MessageStatus.PENDING

    def test_message_with_priority(self):
        """Test message with different priorities."""
        high = Message(
            topic="test",
            payload={},
            priority=MessagePriority.HIGH,
        )
        assert high.priority == MessagePriority.HIGH

        critical = Message(
            topic="test",
            payload={},
            priority=MessagePriority.CRITICAL,
        )
        assert critical.priority == MessagePriority.CRITICAL

    def test_message_with_correlation_id(self):
        """Test message with correlation ID for request-reply."""
        message = Message(
            topic="test",
            payload={},
            correlation_id="corr-123",
            reply_to="reply.topic",
        )

        assert message.correlation_id == "corr-123"
        assert message.reply_to == "reply.topic"

    def test_message_with_headers(self):
        """Test message with custom headers."""
        message = Message(
            topic="test",
            payload={},
            headers={"custom": "header", "version": "1.0"},
        )

        assert message.headers["custom"] == "header"
        assert message.headers["version"] == "1.0"

    def test_message_provenance_hash(self):
        """Test provenance hash calculation."""
        message = Message(
            topic="test",
            payload={"data": "test"},
            source_agent="agent_1",
            target_agent="agent_2",
        )

        assert message.provenance_hash is not None
        assert len(message.provenance_hash) == 64  # SHA-256

    def test_message_ttl(self):
        """Test message time-to-live."""
        message = Message(
            topic="test",
            payload={},
            ttl_seconds=1,
        )

        # Should not be expired immediately
        assert not message.is_expired()

        # Manually set old timestamp
        message.timestamp = DeterministicClock.utcnow() - timedelta(seconds=2)
        assert message.is_expired()

    def test_message_retry(self):
        """Test message retry logic."""
        message = Message(
            topic="test",
            payload={},
            max_retries=3,
        )

        assert message.retry_count == 0
        assert message.can_retry()

        # Increment retries
        message.increment_retry()
        assert message.retry_count == 1
        assert message.can_retry()

        message.increment_retry()
        message.increment_retry()
        assert message.retry_count == 3
        assert not message.can_retry()  # At max

    def test_message_serialization_json(self):
        """Test JSON serialization."""
        message = Message(
            topic="test.serialize",
            payload={"data": "test", "number": 42},
            priority=MessagePriority.HIGH,
        )

        # Serialize
        serialized = message.serialize(format="json")
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = Message.deserialize(serialized, format="json")
        assert deserialized.topic == message.topic
        assert deserialized.payload == message.payload
        assert deserialized.priority == message.priority
        assert deserialized.id == message.id

    def test_message_serialization_msgpack(self):
        """Test MessagePack serialization."""
        pytest.importorskip("msgpack")  # Skip if msgpack not installed

        message = Message(
            topic="test.msgpack",
            payload={"data": "test"},
        )

        # Serialize
        serialized = message.serialize(format="msgpack")
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = Message.deserialize(serialized, format="msgpack")
        assert deserialized.topic == message.topic
        assert deserialized.payload == message.payload

    def test_message_validation(self):
        """Test message validation."""
        # Valid topic
        message = Message(topic="valid.topic", payload={})
        assert message.topic == "valid.topic"

        # Invalid topic
        with pytest.raises(ValueError):
            Message(topic="invalid topic with spaces", payload={})

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = Message(
            topic="test",
            payload={"key": "value"},
        )

        msg_dict = message.to_dict()

        assert isinstance(msg_dict, dict)
        assert msg_dict["topic"] == "test"
        assert msg_dict["payload"] == {"key": "value"}

    def test_message_repr(self):
        """Test message string representation."""
        message = Message(
            topic="test",
            payload={},
        )

        repr_str = repr(message)
        assert "Message" in repr_str
        assert message.id[:8] in repr_str
        assert "test" in repr_str


class TestMessageBatch:
    """Test MessageBatch model."""

    def test_create_batch(self):
        """Test creating message batch."""
        messages = [
            Message(topic="test", payload={"index": i})
            for i in range(5)
        ]

        batch = MessageBatch(messages=messages)

        assert batch.batch_id is not None
        assert len(batch) == 5
        assert batch.size() == 5

    def test_batch_serialization(self):
        """Test batch serialization."""
        messages = [
            Message(topic="test", payload={"index": i})
            for i in range(3)
        ]

        batch = MessageBatch(messages=messages)

        # Serialize all
        serialized = batch.serialize_all(format="json")

        assert len(serialized) == 3
        assert all(isinstance(s, bytes) for s in serialized)


class TestMessageAck:
    """Test MessageAck model."""

    def test_create_ack_success(self):
        """Test creating success acknowledgment."""
        ack = MessageAck(
            message_id="msg-123",
            status="success",
            processing_time_ms=50.5,
        )

        assert ack.message_id == "msg-123"
        assert ack.status == "success"
        assert ack.is_success()
        assert ack.processing_time_ms == 50.5

    def test_create_ack_failure(self):
        """Test creating failure acknowledgment."""
        ack = MessageAck(
            message_id="msg-456",
            status="failure",
            processing_time_ms=100.0,
            error_message="Processing failed",
        )

        assert ack.status == "failure"
        assert not ack.is_success()
        assert ack.error_message == "Processing failed"

    def test_ack_with_consumer_id(self):
        """Test acknowledgment with consumer ID."""
        ack = MessageAck(
            message_id="msg-789",
            status="success",
            processing_time_ms=25.0,
            consumer_id="consumer-001",
        )

        assert ack.consumer_id == "consumer-001"


class TestDeadLetterMessage:
    """Test DeadLetterMessage model."""

    def test_create_dlq_message(self):
        """Test creating dead letter message."""
        original = Message(
            topic="test",
            payload={"data": "failed"},
        )

        dlq_msg = DeadLetterMessage(
            original_message=original,
            failure_reason="Max retries exceeded",
            retry_history=[
                {"attempt": 1, "error": "Error 1"},
                {"attempt": 2, "error": "Error 2"},
            ],
            dlq_topic="dlq:test",
        )

        assert dlq_msg.original_message == original
        assert dlq_msg.failure_reason == "Max retries exceeded"
        assert len(dlq_msg.retry_history) == 2
        assert dlq_msg.dlq_topic == "dlq:test"

    def test_dlq_to_dict(self):
        """Test converting DLQ message to dictionary."""
        original = Message(
            topic="test",
            payload={"data": "test"},
        )

        dlq_msg = DeadLetterMessage(
            original_message=original,
            failure_reason="Failed",
            dlq_topic="dlq:test",
        )

        dlq_dict = dlq_msg.to_dict()

        assert isinstance(dlq_dict, dict)
        assert "original_message" in dlq_dict
        assert "failure_reason" in dlq_dict
        assert dlq_dict["failure_reason"] == "Failed"


class TestMessagePriority:
    """Test MessagePriority enum."""

    def test_priority_levels(self):
        """Test all priority levels."""
        assert MessagePriority.LOW.value == "low"
        assert MessagePriority.NORMAL.value == "normal"
        assert MessagePriority.HIGH.value == "high"
        assert MessagePriority.CRITICAL.value == "critical"

    def test_priority_comparison(self):
        """Test priority enum values."""
        priorities = [
            MessagePriority.LOW,
            MessagePriority.NORMAL,
            MessagePriority.HIGH,
            MessagePriority.CRITICAL,
        ]

        assert len(priorities) == 4
        assert all(isinstance(p, MessagePriority) for p in priorities)


class TestMessageStatus:
    """Test MessageStatus enum."""

    def test_status_values(self):
        """Test all status values."""
        assert MessageStatus.PENDING.value == "pending"
        assert MessageStatus.PROCESSING.value == "processing"
        assert MessageStatus.COMPLETED.value == "completed"
        assert MessageStatus.FAILED.value == "failed"
        assert MessageStatus.DEAD_LETTER.value == "dead_letter"


class TestMessageEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_payload(self):
        """Test message with empty payload."""
        message = Message(topic="test", payload={})
        assert message.payload == {}

    def test_large_payload(self):
        """Test message with large payload."""
        large_data = {"data": "x" * 10000}
        message = Message(topic="test", payload=large_data)

        # Should serialize successfully
        serialized = message.serialize()
        assert len(serialized) > 10000

    def test_nested_payload(self):
        """Test message with nested payload."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {"value": 42}
                }
            }
        }

        message = Message(topic="test", payload=nested)
        assert message.payload["level1"]["level2"]["level3"]["value"] == 42

    def test_special_characters_in_payload(self):
        """Test message with special characters."""
        special = {
            "emoji": "😀",
            "unicode": "日本語",
            "special": "@#$%^&*()",
        }

        message = Message(topic="test", payload=special)

        # Should serialize and deserialize correctly
        serialized = message.serialize()
        deserialized = Message.deserialize(serialized)

        assert deserialized.payload == special

    def test_negative_ttl(self):
        """Test that negative TTL is not allowed."""
        with pytest.raises(ValueError):
            Message(topic="test", payload={}, ttl_seconds=-1)

    def test_zero_max_retries(self):
        """Test message with zero max retries."""
        message = Message(topic="test", payload={}, max_retries=0)
        assert not message.can_retry()

    def test_message_immutability_after_hash(self):
        """Test provenance hash remains consistent."""
        message = Message(
            topic="test",
            payload={"data": "test"},
        )

        original_hash = message.provenance_hash

        # Changing retry count should not affect provenance hash
        message.increment_retry()

        assert message.provenance_hash == original_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
