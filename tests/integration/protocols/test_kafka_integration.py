# -*- coding: utf-8 -*-
"""
Kafka Integration Tests for Process Heat Agents
================================================

Comprehensive integration tests for Kafka producer/consumer functionality:
- Producer send (sync, async)
- Producer batch operations
- Consumer group management
- Offset commit/seek
- Transaction support
- Error handling
- Schema registry integration

Test Coverage Target: 85%+

References:
- greenlang/infrastructure/protocols/kafka_producer.py
- greenlang/infrastructure/protocols/kafka_consumer.py

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from tests.integration.protocols.conftest import (
    MockKafkaCluster,
    MockKafkaRecord,
)


# =============================================================================
# Test Class: Kafka Producer Tests
# =============================================================================


class TestKafkaProducer:
    """Test Kafka producer functionality."""

    @pytest.mark.asyncio
    async def test_produce_single_message(self, mock_kafka_cluster):
        """Test producing a single message."""
        cluster = mock_kafka_cluster

        record = await cluster.produce(
            topic="test-topic",
            value=b'{"temperature": 85.5}'
        )

        assert record.topic == "test-topic"
        assert record.offset >= 0
        assert record.partition >= 0

    @pytest.mark.asyncio
    async def test_produce_with_key(self, mock_kafka_cluster):
        """Test producing a message with key."""
        cluster = mock_kafka_cluster

        record = await cluster.produce(
            topic="test-topic",
            key=b"sensor-001",
            value=b'{"value": 85.5}'
        )

        assert record.key == b"sensor-001"

    @pytest.mark.asyncio
    async def test_produce_to_specific_partition(self, mock_kafka_cluster):
        """Test producing to a specific partition."""
        cluster = mock_kafka_cluster

        record = await cluster.produce(
            topic="test-topic",
            value=b"test",
            partition=1
        )

        assert record.partition == 1

    @pytest.mark.asyncio
    async def test_produce_with_headers(self, mock_kafka_cluster):
        """Test producing a message with headers."""
        cluster = mock_kafka_cluster

        headers = [
            ("source", b"boiler_1"),
            ("content-type", b"application/json")
        ]

        record = await cluster.produce(
            topic="test-topic",
            value=b'{"data": "test"}',
            headers=headers
        )

        # Check headers (includes provenance header added by mock)
        header_dict = {h[0]: h[1] for h in record.headers}
        assert header_dict["source"] == b"boiler_1"
        assert "provenance_hash" in header_dict

    @pytest.mark.asyncio
    async def test_produce_creates_topic_if_not_exists(self, mock_kafka_cluster):
        """Test producing to a non-existent topic creates it."""
        cluster = mock_kafka_cluster

        record = await cluster.produce(
            topic="new-topic",
            value=b"test"
        )

        assert "new-topic" in cluster.topics
        assert record.topic == "new-topic"

    @pytest.mark.asyncio
    async def test_produce_batch(self, mock_kafka_cluster):
        """Test producing multiple messages in a batch."""
        cluster = mock_kafka_cluster

        records_data = [
            {"topic": "test-topic", "value": b'{"id": 1}', "key": b"key1"},
            {"topic": "test-topic", "value": b'{"id": 2}', "key": b"key2"},
            {"topic": "test-topic", "value": b'{"id": 3}', "key": b"key3"},
        ]

        records = await cluster.produce_batch(records_data)

        assert len(records) == 3
        assert all(r.topic == "test-topic" for r in records)

    @pytest.mark.asyncio
    async def test_produce_when_cluster_down(self, mock_kafka_cluster):
        """Test producing fails when cluster is down."""
        cluster = mock_kafka_cluster
        cluster.simulate_cluster_down()

        with pytest.raises(ConnectionError, match="not running"):
            await cluster.produce("test-topic", b"test")

    @pytest.mark.asyncio
    async def test_produce_provenance_hash(self, mock_kafka_cluster):
        """Test provenance hash is added to messages."""
        cluster = mock_kafka_cluster

        record = await cluster.produce(
            topic="test-topic",
            value=b"test"
        )

        # Check provenance hash in headers
        header_dict = {h[0]: h[1] for h in record.headers}
        provenance = header_dict.get("provenance_hash")

        assert provenance is not None
        assert len(provenance.decode()) == 64  # SHA-256 hex string


# =============================================================================
# Test Class: Kafka Consumer Tests
# =============================================================================


class TestKafkaConsumer:
    """Test Kafka consumer functionality."""

    @pytest.mark.asyncio
    async def test_consume_messages(self, mock_kafka_with_data):
        """Test consuming messages from a topic."""
        cluster = mock_kafka_with_data

        records = await cluster.consume(
            group_id="test-group",
            topics=["test-topic"]
        )

        assert len(records) > 0

    @pytest.mark.asyncio
    async def test_consume_creates_consumer_group(self, mock_kafka_with_data):
        """Test consuming creates consumer group."""
        cluster = mock_kafka_with_data

        await cluster.consume(
            group_id="new-group",
            topics=["test-topic"]
        )

        assert "new-group" in cluster.consumer_groups

    @pytest.mark.asyncio
    async def test_consume_one(self, mock_kafka_with_data):
        """Test consuming a single message."""
        cluster = mock_kafka_with_data

        record = await cluster.consume_one(
            group_id="test-group",
            topics=["test-topic"]
        )

        assert record is not None
        assert record.topic == "test-topic"

    @pytest.mark.asyncio
    async def test_consume_with_max_records(self, mock_kafka_with_data):
        """Test consuming with max records limit."""
        cluster = mock_kafka_with_data

        records = await cluster.consume(
            group_id="test-group",
            topics=["test-topic"],
            max_records=2
        )

        assert len(records) <= 2

    @pytest.mark.asyncio
    async def test_consume_from_multiple_topics(self, mock_kafka_cluster):
        """Test consuming from multiple topics."""
        cluster = mock_kafka_cluster

        # Produce to multiple topics
        await cluster.produce("topic-a", b"message-a")
        await cluster.produce("topic-b", b"message-b")

        records = await cluster.consume(
            group_id="multi-topic-group",
            topics=["topic-a", "topic-b"]
        )

        topics = {r.topic for r in records}
        assert "topic-a" in topics or "topic-b" in topics

    @pytest.mark.asyncio
    async def test_consume_when_cluster_down(self, mock_kafka_cluster):
        """Test consuming fails when cluster is down."""
        cluster = mock_kafka_cluster
        cluster.simulate_cluster_down()

        with pytest.raises(ConnectionError, match="not running"):
            await cluster.consume("test-group", ["test-topic"])


# =============================================================================
# Test Class: Kafka Offset Management Tests
# =============================================================================


class TestKafkaOffsetManagement:
    """Test Kafka offset commit and seek functionality."""

    @pytest.mark.asyncio
    async def test_commit_offsets(self, mock_kafka_with_data):
        """Test committing offsets."""
        cluster = mock_kafka_with_data
        group_id = "commit-test-group"

        # Consume some messages
        records = await cluster.consume(
            group_id=group_id,
            topics=["test-topic"]
        )

        # Commit offsets
        offsets = {"test-topic:0": 3}
        await cluster.commit(group_id, offsets)

        assert cluster.committed_offsets[group_id]["test-topic:0"] == 3

    @pytest.mark.asyncio
    async def test_seek_to_offset(self, mock_kafka_with_data):
        """Test seeking to a specific offset."""
        cluster = mock_kafka_with_data
        group_id = "seek-test-group"

        await cluster.seek(
            group_id=group_id,
            topic="test-topic",
            partition=0,
            offset=2
        )

        assert cluster.committed_offsets[group_id]["test-topic:0"] == 2

    @pytest.mark.asyncio
    async def test_seek_to_beginning(self, mock_kafka_with_data):
        """Test seeking to the beginning of a partition."""
        cluster = mock_kafka_with_data
        group_id = "seek-begin-group"

        # First, set some offset
        await cluster.seek(group_id, "test-topic", 0, 5)

        # Seek to beginning
        await cluster.seek_to_beginning(group_id, "test-topic", 0)

        assert cluster.committed_offsets[group_id]["test-topic:0"] == 0

    @pytest.mark.asyncio
    async def test_seek_to_end(self, mock_kafka_with_data):
        """Test seeking to the end of a partition."""
        cluster = mock_kafka_with_data
        group_id = "seek-end-group"

        # Get current end offset
        partition_records = cluster.topics["test-topic"][0]
        expected_end = len(partition_records)

        await cluster.seek_to_end(group_id, "test-topic", 0)

        assert cluster.committed_offsets[group_id]["test-topic:0"] == expected_end

    @pytest.mark.asyncio
    async def test_consume_respects_committed_offset(self, mock_kafka_cluster):
        """Test consuming starts from committed offset."""
        cluster = mock_kafka_cluster
        group_id = "offset-respect-group"

        # Produce messages
        for i in range(5):
            await cluster.produce("test-topic", f'{{"id": {i}}}'.encode())

        # Commit offset at 3
        await cluster.commit(group_id, {"test-topic:0": 3})

        # Consume should start from offset 3
        records = await cluster.consume(group_id, ["test-topic"])

        # Should only get messages from offset 3 onwards
        offsets = [r.offset for r in records if r.partition == 0]
        assert all(o >= 3 for o in offsets) if offsets else True


# =============================================================================
# Test Class: Kafka Transaction Tests
# =============================================================================


class TestKafkaTransactions:
    """Test Kafka transaction support."""

    @pytest.mark.asyncio
    async def test_begin_transaction(self, mock_kafka_cluster):
        """Test beginning a transaction."""
        cluster = mock_kafka_cluster
        producer_id = "transactional-producer"

        await cluster.begin_transaction(producer_id)

        assert cluster._transaction_active[producer_id] is True

    @pytest.mark.asyncio
    async def test_commit_transaction(self, mock_kafka_cluster):
        """Test committing a transaction."""
        cluster = mock_kafka_cluster
        producer_id = "transactional-producer"

        await cluster.begin_transaction(producer_id)

        # Produce within transaction
        await cluster.produce(
            topic="test-topic",
            value=b"transactional-msg",
            producer_id=producer_id
        )

        # Message should not be visible yet
        pre_commit_count = cluster.get_topic_message_count("test-topic")

        # Commit transaction
        await cluster.commit_transaction(producer_id)

        # Message should now be visible
        post_commit_count = cluster.get_topic_message_count("test-topic")

        assert post_commit_count > pre_commit_count

    @pytest.mark.asyncio
    async def test_abort_transaction(self, mock_kafka_cluster):
        """Test aborting a transaction."""
        cluster = mock_kafka_cluster
        producer_id = "abort-producer"

        initial_count = cluster.get_topic_message_count("test-topic")

        await cluster.begin_transaction(producer_id)

        # Produce within transaction
        await cluster.produce(
            topic="test-topic",
            value=b"aborted-msg",
            producer_id=producer_id
        )

        # Abort transaction
        await cluster.abort_transaction(producer_id)

        # Message count should be unchanged
        final_count = cluster.get_topic_message_count("test-topic")
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_commit_without_active_transaction(self, mock_kafka_cluster):
        """Test committing without active transaction raises error."""
        cluster = mock_kafka_cluster

        with pytest.raises(RuntimeError, match="No active transaction"):
            await cluster.commit_transaction("non-transactional")


# =============================================================================
# Test Class: Kafka Schema Registry Tests
# =============================================================================


class TestKafkaSchemaRegistry:
    """Test Kafka schema registry functionality."""

    @pytest.mark.asyncio
    async def test_register_schema(self, mock_kafka_cluster):
        """Test registering an Avro schema."""
        cluster = mock_kafka_cluster

        schema = {
            "type": "record",
            "name": "ProcessHeatEvent",
            "fields": [
                {"name": "temperature", "type": "double"},
                {"name": "pressure", "type": "double"},
                {"name": "timestamp", "type": "string"}
            ]
        }

        schema_id = cluster.register_schema("process-heat-value", schema)

        assert schema_id > 0
        assert "process-heat-value" in cluster.schemas

    @pytest.mark.asyncio
    async def test_get_schema(self, mock_kafka_cluster):
        """Test retrieving a registered schema."""
        cluster = mock_kafka_cluster

        schema = {
            "type": "record",
            "name": "EmissionEvent",
            "fields": [
                {"name": "value_kg", "type": "double"},
                {"name": "source", "type": "string"}
            ]
        }

        cluster.register_schema("emission-value", schema)

        retrieved = cluster.get_schema("emission-value")

        assert retrieved == schema
        assert retrieved["name"] == "EmissionEvent"

    @pytest.mark.asyncio
    async def test_get_nonexistent_schema(self, mock_kafka_cluster):
        """Test retrieving a non-existent schema returns None."""
        cluster = mock_kafka_cluster

        result = cluster.get_schema("nonexistent-schema")

        assert result is None


# =============================================================================
# Test Class: Kafka Partition Tests
# =============================================================================


class TestKafkaPartitions:
    """Test Kafka partition management."""

    @pytest.mark.asyncio
    async def test_get_topic_partitions(self, mock_kafka_cluster):
        """Test getting partition IDs for a topic."""
        cluster = mock_kafka_cluster

        partitions = cluster.get_topic_partitions("test-topic")

        assert len(partitions) == 3  # Default 3 partitions
        assert partitions == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_key_based_partitioning(self, mock_kafka_cluster):
        """Test messages with same key go to same partition."""
        cluster = mock_kafka_cluster

        records = []
        for i in range(10):
            record = await cluster.produce(
                topic="test-topic",
                key=b"same-key",
                value=f'{{"id": {i}}}'.encode()
            )
            records.append(record)

        # All records with same key should be in same partition
        partitions = {r.partition for r in records}
        assert len(partitions) == 1

    @pytest.mark.asyncio
    async def test_round_robin_partitioning(self, mock_kafka_cluster):
        """Test messages without key are round-robin partitioned."""
        cluster = mock_kafka_cluster

        # Clear existing data for clean test
        cluster.topics["test-topic"] = {0: [], 1: [], 2: []}
        cluster._offset_counter = 0

        records = []
        for i in range(9):  # 9 messages across 3 partitions
            record = await cluster.produce(
                topic="test-topic",
                value=f'{{"id": {i}}}'.encode()
            )
            records.append(record)

        # Should have distributed across partitions
        partitions = [r.partition for r in records]
        unique_partitions = set(partitions)
        assert len(unique_partitions) >= 2  # At least 2 different partitions


# =============================================================================
# Test Class: Kafka Consumer Group Tests
# =============================================================================


class TestKafkaConsumerGroups:
    """Test Kafka consumer group management."""

    @pytest.mark.asyncio
    async def test_get_consumer_group_info(self, mock_kafka_with_data):
        """Test getting consumer group information."""
        cluster = mock_kafka_with_data
        group_id = "info-test-group"

        await cluster.consume(group_id, ["test-topic"])

        info = cluster.get_consumer_group_info(group_id)

        assert info is not None
        assert "topics" in info
        assert "created_at" in info

    @pytest.mark.asyncio
    async def test_multiple_consumer_groups(self, mock_kafka_with_data):
        """Test multiple consumer groups consuming same topic."""
        cluster = mock_kafka_with_data

        await cluster.consume("group-a", ["test-topic"])
        await cluster.consume("group-b", ["test-topic"])

        assert "group-a" in cluster.consumer_groups
        assert "group-b" in cluster.consumer_groups

    @pytest.mark.asyncio
    async def test_consumer_group_independent_offsets(self, mock_kafka_cluster):
        """Test consumer groups have independent offsets."""
        cluster = mock_kafka_cluster

        # Produce messages
        for i in range(5):
            await cluster.produce("test-topic", f'{{"id": {i}}}'.encode())

        # Commit different offsets for different groups
        await cluster.commit("group-a", {"test-topic:0": 2})
        await cluster.commit("group-b", {"test-topic:0": 4})

        assert cluster.committed_offsets["group-a"]["test-topic:0"] == 2
        assert cluster.committed_offsets["group-b"]["test-topic:0"] == 4


# =============================================================================
# Test Class: Kafka Statistics Tests
# =============================================================================


class TestKafkaStatistics:
    """Test Kafka cluster statistics."""

    @pytest.mark.asyncio
    async def test_cluster_statistics(self, mock_kafka_cluster):
        """Test cluster statistics are accurate."""
        cluster = mock_kafka_cluster

        stats = cluster.get_statistics()

        assert stats["running"] is True
        assert stats["topic_count"] >= 4  # Default topics created

    @pytest.mark.asyncio
    async def test_message_count_statistics(self, mock_kafka_cluster):
        """Test message count in statistics."""
        cluster = mock_kafka_cluster

        # Produce messages
        for _ in range(10):
            await cluster.produce("test-topic", b"test")

        count = cluster.get_topic_message_count("test-topic")
        assert count == 10

    @pytest.mark.asyncio
    async def test_statistics_update_on_consumer_group_creation(self, mock_kafka_cluster):
        """Test statistics update on consumer group creation."""
        cluster = mock_kafka_cluster

        initial_stats = cluster.get_statistics()
        initial_groups = initial_stats["consumer_group_count"]

        await cluster.consume("new-group", ["test-topic"])

        updated_stats = cluster.get_statistics()
        assert updated_stats["consumer_group_count"] == initial_groups + 1


# =============================================================================
# Test Class: Kafka Performance Tests
# =============================================================================


@pytest.mark.performance
class TestKafkaPerformance:
    """Performance tests for Kafka operations."""

    @pytest.mark.asyncio
    async def test_produce_throughput(self, mock_kafka_cluster, throughput_calculator):
        """Test producer throughput."""
        cluster = mock_kafka_cluster

        throughput_calculator.start()

        for i in range(1000):
            payload = f'{{"id": {i}, "value": {i * 1.5}}}'.encode()
            await cluster.produce("test-topic", payload)
            throughput_calculator.record_message(len(payload))

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 500

    @pytest.mark.asyncio
    async def test_batch_produce_throughput(self, mock_kafka_cluster, throughput_calculator):
        """Test batch producer throughput."""
        cluster = mock_kafka_cluster

        throughput_calculator.start()

        for _ in range(10):
            batch = [
                {"topic": "test-topic", "value": f'{{"id": {i}}}'.encode()}
                for i in range(100)
            ]
            records = await cluster.produce_batch(batch)
            for r in records:
                throughput_calculator.record_message(len(r.value))

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 1000

    @pytest.mark.asyncio
    async def test_consume_throughput(self, mock_kafka_cluster, throughput_calculator):
        """Test consumer throughput."""
        cluster = mock_kafka_cluster

        # First produce messages
        for i in range(500):
            await cluster.produce("test-topic", f'{{"id": {i}}}'.encode())

        throughput_calculator.start()

        records = await cluster.consume(
            group_id="perf-group",
            topics=["test-topic"],
            max_records=500
        )

        for r in records:
            throughput_calculator.record_message(len(r.value))

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 500


# =============================================================================
# Test Class: Kafka Process Heat Integration
# =============================================================================


class TestKafkaProcessHeatIntegration:
    """Integration tests for process heat data via Kafka."""

    @pytest.mark.asyncio
    async def test_produce_process_heat_event(
        self,
        mock_kafka_cluster,
        sample_process_heat_data
    ):
        """Test producing process heat event."""
        cluster = mock_kafka_cluster

        payload = json.dumps(sample_process_heat_data).encode()

        record = await cluster.produce(
            topic="process-heat-data",
            key=b"boiler_1",
            value=payload,
            headers=[("event_type", b"sensor_reading")]
        )

        assert record.topic == "process-heat-data"
        assert record.key == b"boiler_1"

    @pytest.mark.asyncio
    async def test_consume_emissions_events(
        self,
        mock_kafka_cluster,
        sample_emission_event
    ):
        """Test consuming emissions events."""
        cluster = mock_kafka_cluster

        # Produce emission events
        for i in range(5):
            event = sample_emission_event.copy()
            event["value_kg"] = 100.0 + i * 10
            await cluster.produce(
                topic="emissions-events",
                key=event["source"].encode(),
                value=json.dumps(event).encode()
            )

        # Consume events
        records = await cluster.consume(
            group_id="emissions-processor",
            topics=["emissions-events"]
        )

        assert len(records) == 5

        # Parse and validate
        events = [json.loads(r.value.decode()) for r in records]
        total_emissions = sum(e["value_kg"] for e in events)
        assert total_emissions == 600.0  # 100+110+120+130+140

    @pytest.mark.asyncio
    async def test_dead_letter_queue_workflow(self, mock_kafka_cluster):
        """Test dead letter queue for failed messages."""
        cluster = mock_kafka_cluster
        group_id = "dlq-processor"

        # Produce messages
        await cluster.produce("process-heat-data", b'{"valid": "data"}')
        await cluster.produce("process-heat-data", b'invalid-json')

        # Consume and process
        records = await cluster.consume(group_id, ["process-heat-data"])

        for record in records:
            try:
                data = json.loads(record.value.decode())
                # Process successful
            except json.JSONDecodeError:
                # Send to DLQ
                await cluster.produce(
                    topic="dlq-topic",
                    value=record.value,
                    headers=[
                        ("original_topic", record.topic.encode()),
                        ("error", b"invalid_json")
                    ]
                )

        # Check DLQ
        dlq_count = cluster.get_topic_message_count("dlq-topic")
        assert dlq_count == 1

    @pytest.mark.asyncio
    async def test_transactional_emissions_workflow(self, mock_kafka_cluster):
        """Test transactional emissions calculation workflow."""
        cluster = mock_kafka_cluster
        producer_id = "emissions-calculator"

        # Begin transaction
        await cluster.begin_transaction(producer_id)

        # Calculate emissions from multiple sources
        sources = [
            {"source": "boiler_1", "fuel_kg": 100, "ef": 2.68},
            {"source": "boiler_2", "fuel_kg": 150, "ef": 2.68},
            {"source": "heater_1", "fuel_kg": 75, "ef": 2.68},
        ]

        total_emissions = 0
        for src in sources:
            emissions = src["fuel_kg"] * src["ef"]
            total_emissions += emissions

            await cluster.produce(
                topic="emissions-events",
                key=src["source"].encode(),
                value=json.dumps({
                    "source": src["source"],
                    "emissions_kg": emissions
                }).encode(),
                producer_id=producer_id
            )

        # Produce aggregated result
        await cluster.produce(
            topic="emissions-events",
            key=b"aggregate",
            value=json.dumps({
                "type": "aggregate",
                "total_emissions_kg": total_emissions
            }).encode(),
            producer_id=producer_id
        )

        # Commit all atomically
        await cluster.commit_transaction(producer_id)

        # Verify all messages are visible
        count = cluster.get_topic_message_count("emissions-events")
        assert count >= 4  # 3 sources + 1 aggregate

    @pytest.mark.asyncio
    async def test_schema_evolution_workflow(self, mock_kafka_cluster):
        """Test schema evolution for process heat events."""
        cluster = mock_kafka_cluster

        # Register v1 schema
        schema_v1 = {
            "type": "record",
            "name": "ProcessHeatEvent",
            "fields": [
                {"name": "temperature", "type": "double"},
                {"name": "timestamp", "type": "string"}
            ]
        }
        cluster.register_schema("process-heat-value", schema_v1)

        # Register v2 schema (backward compatible - new optional field)
        schema_v2 = {
            "type": "record",
            "name": "ProcessHeatEvent",
            "fields": [
                {"name": "temperature", "type": "double"},
                {"name": "timestamp", "type": "string"},
                {"name": "efficiency", "type": ["null", "double"], "default": None}
            ]
        }
        cluster.register_schema("process-heat-value-v2", schema_v2)

        # Both schemas should be accessible
        assert cluster.get_schema("process-heat-value") is not None
        assert cluster.get_schema("process-heat-value-v2") is not None

    @pytest.mark.asyncio
    async def test_offset_reset_replay(self, mock_kafka_cluster):
        """Test replaying messages by resetting offset."""
        cluster = mock_kafka_cluster
        group_id = "replay-group"

        # Produce messages
        for i in range(10):
            await cluster.produce("test-topic", f'{{"seq": {i}}}'.encode())

        # Consume all messages
        await cluster.consume(group_id, ["test-topic"], max_records=10)

        # Commit offset at end
        await cluster.commit(group_id, {"test-topic:0": 10})

        # Reset to beginning
        await cluster.seek_to_beginning(group_id, "test-topic", 0)

        # Should be able to consume all again
        replayed = await cluster.consume(group_id, ["test-topic"], max_records=10)

        assert len(replayed) >= 1  # Can replay from beginning
