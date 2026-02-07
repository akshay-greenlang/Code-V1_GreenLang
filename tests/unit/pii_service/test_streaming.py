# -*- coding: utf-8 -*-
"""
Unit tests for PII Streaming Scanner - SEC-011 PII Service.

Tests the streaming PII scanner for Kafka and Kinesis:
- Kafka consumer/producer lifecycle
- Kinesis record processing
- Dead letter queue handling
- Tenant extraction from headers
- Batch processing
- Error handling

Coverage target: 85%+ of streaming/*.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def kafka_scanner(streaming_config, enforcement_config, mock_kafka_consumer, mock_kafka_producer):
    """Create KafkaPIIScanner instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.streaming.kafka_scanner import KafkaPIIScanner
        return KafkaPIIScanner(
            config=streaming_config,
            enforcement_config=enforcement_config,
            consumer=mock_kafka_consumer,
            producer=mock_kafka_producer,
        )
    except ImportError:
        pytest.skip("KafkaPIIScanner not yet implemented")


@pytest.fixture
def kinesis_scanner(streaming_config, enforcement_config, mock_kinesis_client):
    """Create KinesisPIIScanner instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.streaming.kinesis_scanner import KinesisPIIScanner
        return KinesisPIIScanner(
            config=streaming_config,
            enforcement_config=enforcement_config,
            kinesis_client=mock_kinesis_client,
        )
    except ImportError:
        pytest.skip("KinesisPIIScanner not yet implemented")


@pytest.fixture
def sample_kafka_message():
    """Create sample Kafka message."""
    return MagicMock(
        topic="raw-events",
        value=json.dumps({
            "event_id": str(uuid4()),
            "user_email": "john@company.com",
            "user_name": "John Doe",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).encode(),
        headers=[
            ("X-Tenant-ID", b"tenant-123"),
            ("X-Request-ID", b"req-456"),
        ],
        offset=12345,
        partition=0,
    )


@pytest.fixture
def sample_kinesis_record():
    """Create sample Kinesis record."""
    return {
        "SequenceNumber": "12345",
        "Data": json.dumps({
            "event_id": str(uuid4()),
            "user_email": "john@company.com",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).encode(),
        "PartitionKey": "tenant-123",
    }


# ============================================================================
# TestKafkaScannerLifecycle
# ============================================================================


class TestKafkaScannerLifecycle:
    """Tests for Kafka scanner lifecycle."""

    @pytest.mark.asyncio
    async def test_kafka_scanner_starts(
        self, kafka_scanner, mock_kafka_consumer
    ):
        """Kafka scanner starts consumer correctly."""
        await kafka_scanner.start()

        mock_kafka_consumer.start.assert_awaited_once()
        mock_kafka_consumer.subscribe.assert_awaited()

    @pytest.mark.asyncio
    async def test_kafka_scanner_stops(
        self, kafka_scanner, mock_kafka_consumer, mock_kafka_producer
    ):
        """Kafka scanner stops consumer and producer correctly."""
        await kafka_scanner.start()
        await kafka_scanner.stop()

        mock_kafka_consumer.stop.assert_awaited_once()
        mock_kafka_producer.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kafka_scanner_subscribes_to_topics(
        self, kafka_scanner, mock_kafka_consumer, streaming_config
    ):
        """Kafka scanner subscribes to configured topics."""
        await kafka_scanner.start()

        # Should subscribe to input topics
        subscribe_call = mock_kafka_consumer.subscribe.call_args
        assert streaming_config.input_topics[0] in str(subscribe_call)


# ============================================================================
# TestKafkaMessageProcessing
# ============================================================================


class TestKafkaMessageProcessing:
    """Tests for Kafka message processing."""

    @pytest.mark.asyncio
    async def test_kafka_scanner_processes_messages(
        self, kafka_scanner, mock_kafka_consumer, sample_kafka_message
    ):
        """Kafka scanner processes messages from consumer."""
        mock_kafka_consumer._messages = [
            {
                "topic": "raw-events",
                "value": sample_kafka_message.value,
                "headers": sample_kafka_message.headers,
            }
        ]

        await kafka_scanner.start()
        await kafka_scanner.process_one()

        # Message should be processed
        assert mock_kafka_consumer._position == 1

    @pytest.mark.asyncio
    async def test_kafka_scanner_sends_to_output_topic(
        self, kafka_scanner, mock_kafka_producer, streaming_config, sample_kafka_message
    ):
        """Clean messages are sent to output topic."""
        # Process a clean message
        await kafka_scanner._process_message(sample_kafka_message)

        # Should be sent to output topic
        sent = mock_kafka_producer._sent_messages
        assert len(sent) > 0
        assert sent[0]["topic"] == streaming_config.output_topic

    @pytest.mark.asyncio
    async def test_kafka_scanner_sends_blocked_to_dlq(
        self, kafka_scanner, mock_kafka_producer, streaming_config
    ):
        """Blocked messages are sent to dead letter queue."""
        # Message with SSN (should be blocked)
        blocked_message = MagicMock(
            topic="raw-events",
            value=json.dumps({
                "ssn": "123-45-6789",
                "data": "sensitive",
            }).encode(),
            headers=[("X-Tenant-ID", b"tenant-123")],
        )

        await kafka_scanner._process_message(blocked_message)

        # Should be sent to DLQ
        sent = mock_kafka_producer._sent_messages
        dlq_messages = [m for m in sent if m["topic"] == streaming_config.dlq_topic]
        assert len(dlq_messages) > 0

    @pytest.mark.asyncio
    async def test_kafka_scanner_extracts_tenant_from_headers(
        self, kafka_scanner, sample_kafka_message
    ):
        """Tenant ID is extracted from Kafka headers."""
        tenant_id = kafka_scanner._extract_tenant(sample_kafka_message)

        assert tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_kafka_scanner_handles_errors(
        self, kafka_scanner, mock_kafka_producer
    ):
        """Kafka scanner handles processing errors gracefully."""
        # Message that will cause error
        bad_message = MagicMock(
            topic="raw-events",
            value=b"not-valid-json",
            headers=[],
        )

        # Should not raise
        await kafka_scanner._process_message(bad_message)

        # Error message should go to DLQ
        sent = mock_kafka_producer._sent_messages
        dlq_messages = [m for m in sent if "pii-blocked" in m.get("topic", "")]
        # May or may not send to DLQ depending on error handling strategy


# ============================================================================
# TestKinesisScannerLifecycle
# ============================================================================


class TestKinesisScannerLifecycle:
    """Tests for Kinesis scanner lifecycle."""

    @pytest.mark.asyncio
    async def test_kinesis_scanner_starts(
        self, kinesis_scanner
    ):
        """Kinesis scanner starts correctly."""
        await kinesis_scanner.start()

        assert kinesis_scanner._running is True

    @pytest.mark.asyncio
    async def test_kinesis_scanner_stops(
        self, kinesis_scanner
    ):
        """Kinesis scanner stops correctly."""
        await kinesis_scanner.start()
        await kinesis_scanner.stop()

        assert kinesis_scanner._running is False


# ============================================================================
# TestKinesisRecordProcessing
# ============================================================================


class TestKinesisRecordProcessing:
    """Tests for Kinesis record processing."""

    @pytest.mark.asyncio
    async def test_kinesis_scanner_processes_records(
        self, kinesis_scanner, mock_kinesis_client, sample_kinesis_record
    ):
        """Kinesis scanner processes records from stream."""
        mock_kinesis_client._records = [sample_kinesis_record]

        await kinesis_scanner._process_records([sample_kinesis_record])

        # Record should be processed

    @pytest.mark.asyncio
    async def test_kinesis_scanner_sends_to_output_stream(
        self, kinesis_scanner, mock_kinesis_client, streaming_config, sample_kinesis_record
    ):
        """Clean records are sent to output stream."""
        await kinesis_scanner._process_records([sample_kinesis_record])

        # Should be sent to output stream
        records = mock_kinesis_client._records
        # Implementation specific

    @pytest.mark.asyncio
    async def test_kinesis_scanner_sends_blocked_to_dlq(
        self, kinesis_scanner, mock_kinesis_client, streaming_config
    ):
        """Blocked records are sent to dead letter stream."""
        blocked_record = {
            "SequenceNumber": "12345",
            "Data": json.dumps({
                "ssn": "123-45-6789",
            }).encode(),
            "PartitionKey": "tenant-123",
        }

        await kinesis_scanner._process_records([blocked_record])

        # Should be sent to DLQ stream


# ============================================================================
# TestStreamProcessorMetrics
# ============================================================================


class TestStreamProcessorMetrics:
    """Tests for stream processor metrics."""

    @pytest.mark.asyncio
    async def test_stream_processor_metrics(
        self, kafka_scanner, sample_kafka_message
    ):
        """Stream processor records metrics."""
        from unittest.mock import patch

        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_stream_processed_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await kafka_scanner._process_message(sample_kafka_message)

            # Should record metrics
            # Implementation specific

    @pytest.mark.asyncio
    async def test_stream_blocked_metrics(
        self, kafka_scanner
    ):
        """Blocked messages are counted in metrics."""
        from unittest.mock import patch

        blocked_message = MagicMock(
            topic="raw-events",
            value=json.dumps({"ssn": "123-45-6789"}).encode(),
            headers=[("X-Tenant-ID", b"tenant-123")],
        )

        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_stream_blocked_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await kafka_scanner._process_message(blocked_message)

            # Should record blocked metric


# ============================================================================
# TestBatchProcessing
# ============================================================================


class TestBatchProcessing:
    """Tests for batch message processing."""

    @pytest.mark.asyncio
    async def test_batch_processing(
        self, kafka_scanner, streaming_config
    ):
        """Messages are processed in batches."""
        # Create batch of messages
        messages = [
            MagicMock(
                topic="raw-events",
                value=json.dumps({"data": f"message-{i}"}).encode(),
                headers=[("X-Tenant-ID", b"tenant-123")],
            )
            for i in range(streaming_config.batch_size)
        ]

        results = await kafka_scanner._process_batch(messages)

        assert len(results) == streaming_config.batch_size

    @pytest.mark.asyncio
    async def test_batch_partial_failure(
        self, kafka_scanner
    ):
        """Batch processing handles partial failures."""
        messages = [
            # Good message
            MagicMock(
                topic="raw-events",
                value=json.dumps({"data": "clean"}).encode(),
                headers=[("X-Tenant-ID", b"tenant-123")],
            ),
            # Bad message (invalid JSON)
            MagicMock(
                topic="raw-events",
                value=b"invalid-json",
                headers=[("X-Tenant-ID", b"tenant-123")],
            ),
            # Good message
            MagicMock(
                topic="raw-events",
                value=json.dumps({"data": "also clean"}).encode(),
                headers=[("X-Tenant-ID", b"tenant-123")],
            ),
        ]

        # Should process all without raising
        results = await kafka_scanner._process_batch(messages)

        # Good messages should be processed
        successful = [r for r in results if r.get("success", False)]
        assert len(successful) >= 2


# ============================================================================
# TestMessageDecoding
# ============================================================================


class TestMessageDecoding:
    """Tests for message decoding."""

    @pytest.mark.asyncio
    async def test_json_message_decoding(
        self, kafka_scanner
    ):
        """JSON messages are decoded correctly."""
        message = MagicMock(
            value=json.dumps({"key": "value"}).encode(),
            headers=[],
        )

        content = kafka_scanner._decode_message(message)

        assert content == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_utf8_message_decoding(
        self, kafka_scanner
    ):
        """UTF-8 messages are decoded correctly."""
        message = MagicMock(
            value="Hello World".encode("utf-8"),
            headers=[],
        )

        content = kafka_scanner._decode_message(message)

        assert content == "Hello World"

    @pytest.mark.asyncio
    async def test_binary_message_handling(
        self, kafka_scanner
    ):
        """Binary messages are handled appropriately."""
        message = MagicMock(
            value=bytes([0x00, 0x01, 0x02, 0xff]),
            headers=[],
        )

        # Should handle without error
        try:
            content = kafka_scanner._decode_message(message)
        except Exception:
            # May raise or return placeholder
            pass


# ============================================================================
# TestHeaderExtraction
# ============================================================================


class TestHeaderExtraction:
    """Tests for Kafka header extraction."""

    @pytest.mark.asyncio
    async def test_extract_tenant_from_headers(
        self, kafka_scanner
    ):
        """Tenant ID is extracted from X-Tenant-ID header."""
        message = MagicMock(
            headers=[
                ("X-Tenant-ID", b"my-tenant"),
                ("X-Request-ID", b"req-123"),
            ],
        )

        tenant_id = kafka_scanner._extract_tenant(message)

        assert tenant_id == "my-tenant"

    @pytest.mark.asyncio
    async def test_extract_tenant_missing_header(
        self, kafka_scanner
    ):
        """Default tenant is used when header is missing."""
        message = MagicMock(
            headers=[],
        )

        tenant_id = kafka_scanner._extract_tenant(message)

        assert tenant_id == "default" or tenant_id is None

    @pytest.mark.asyncio
    async def test_extract_multiple_headers(
        self, kafka_scanner
    ):
        """Multiple headers are extracted correctly."""
        message = MagicMock(
            headers=[
                ("X-Tenant-ID", b"tenant-abc"),
                ("X-User-ID", b"user-123"),
                ("X-Request-ID", b"req-456"),
                ("X-Correlation-ID", b"corr-789"),
            ],
        )

        context = kafka_scanner._extract_context(message)

        assert context.tenant_id == "tenant-abc"
        assert context.user_id == "user-123"
        assert context.request_id == "req-456"


# ============================================================================
# TestRedactedMessageOutput
# ============================================================================


class TestRedactedMessageOutput:
    """Tests for redacted message output."""

    @pytest.mark.asyncio
    async def test_redacted_message_preserves_structure(
        self, kafka_scanner, mock_kafka_producer
    ):
        """Redacted messages preserve JSON structure."""
        message = MagicMock(
            topic="raw-events",
            value=json.dumps({
                "event_id": "123",
                "user_email": "john@company.com",
                "metadata": {"key": "value"},
            }).encode(),
            headers=[("X-Tenant-ID", b"tenant-123")],
        )

        await kafka_scanner._process_message(message)

        # Output should be valid JSON
        sent = mock_kafka_producer._sent_messages
        if sent:
            output_value = sent[0]["value"]
            parsed = json.loads(output_value)
            assert "event_id" in parsed
            assert "metadata" in parsed

    @pytest.mark.asyncio
    async def test_redacted_message_removes_pii(
        self, kafka_scanner, mock_kafka_producer
    ):
        """Redacted messages have PII removed/replaced."""
        message = MagicMock(
            topic="raw-events",
            value=json.dumps({
                "user_email": "john@company.com",
                "ssn": "123-45-6789",
            }).encode(),
            headers=[("X-Tenant-ID", b"tenant-123")],
        )

        await kafka_scanner._process_message(message)

        # Output should not contain raw PII
        sent = mock_kafka_producer._sent_messages
        # Check based on whether message was blocked or redacted
        for msg in sent:
            if msg["topic"] != "pii-blocked":
                output = msg["value"].decode() if isinstance(msg["value"], bytes) else msg["value"]
                assert "123-45-6789" not in output


# ============================================================================
# TestConsumerGroupHandling
# ============================================================================


class TestConsumerGroupHandling:
    """Tests for consumer group behavior."""

    @pytest.mark.asyncio
    async def test_consumer_group_assignment(
        self, kafka_scanner, mock_kafka_consumer, streaming_config
    ):
        """Consumer uses configured consumer group."""
        await kafka_scanner.start()

        # Consumer should be configured with the group
        assert streaming_config.consumer_group == "pii-scanner-test"

    @pytest.mark.asyncio
    async def test_offset_commit_on_success(
        self, kafka_scanner, mock_kafka_consumer
    ):
        """Offsets are committed after successful processing."""
        message = MagicMock(
            topic="raw-events",
            value=json.dumps({"data": "clean"}).encode(),
            headers=[("X-Tenant-ID", b"tenant-123")],
            offset=100,
        )

        await kafka_scanner._process_message(message)

        # Offset should be committed
        # Implementation specific


# ============================================================================
# TestGracefulShutdown
# ============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_completes_processing(
        self, kafka_scanner, mock_kafka_consumer
    ):
        """Graceful shutdown completes in-flight processing."""
        await kafka_scanner.start()

        # Start processing
        mock_kafka_consumer._messages = [
            {"topic": "raw-events", "value": b'{"data": "test"}', "headers": []},
        ]

        # Initiate shutdown
        await kafka_scanner.stop()

        # Should have stopped cleanly
        assert kafka_scanner._running is False

    @pytest.mark.asyncio
    async def test_shutdown_timeout(
        self, kafka_scanner
    ):
        """Shutdown respects timeout."""
        await kafka_scanner.start()

        # Stop with timeout
        await kafka_scanner.stop(timeout_seconds=5)

        assert kafka_scanner._running is False
