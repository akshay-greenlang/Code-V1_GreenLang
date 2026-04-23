"""
Tests for Kafka Streaming Module - GL-001 ThermalCommand

Comprehensive test coverage for Kafka producer and consumer.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..kafka_streaming import (
    # Configuration
    KafkaConfig,
    ProducerConfig,
    ConsumerConfig,
    TopicConfig,
    SecurityProtocol,
    SASLMechanism,
    CompressionType,
    AcksMode,
    AutoOffsetReset,
    PartitionStrategy,
    # Partitioners
    HashPartitioner,
    RoundRobinPartitioner,
    EquipmentPartitioner,
    # Producer/Consumer
    ThermalCommandProducer,
    ThermalCommandConsumer,
    DeliveryReport,
    ConsumedMessage,
    # Metrics
    ProducerMetrics,
    ConsumerMetrics,
)
from ..kafka_schemas import (
    TelemetryNormalizedEvent,
    TelemetryPoint,
    DispatchPlanEvent,
    LoadAllocation,
    ExpectedImpact,
    SafetyEvent,
    SafetyLevel,
    BoundaryViolation,
    AuditLogEvent,
    AuditAction,
    SolverStatus,
    UnitOfMeasure,
)


class TestKafkaConfig:
    """Tests for Kafka configuration models."""

    def test_default_kafka_config(self) -> None:
        """Test default Kafka configuration."""
        config = KafkaConfig()

        assert config.bootstrap_servers == "localhost:9092"
        assert config.client_id.startswith("gl001-thermalcommand-")
        assert config.security_protocol == SecurityProtocol.PLAINTEXT
        assert config.sasl_mechanism is None

    def test_kafka_config_with_sasl(self) -> None:
        """Test Kafka configuration with SASL."""
        config = KafkaConfig(
            bootstrap_servers="kafka.example.com:9093",
            security_protocol=SecurityProtocol.SASL_SSL,
            sasl_mechanism=SASLMechanism.SCRAM_SHA_256,
            sasl_username="user",
            sasl_password="pass",
            ssl_ca_location="/path/to/ca.pem",
        )

        assert config.security_protocol == SecurityProtocol.SASL_SSL
        assert config.sasl_mechanism == SASLMechanism.SCRAM_SHA_256

    def test_kafka_config_to_confluent(self) -> None:
        """Test conversion to confluent-kafka configuration."""
        config = KafkaConfig(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            client_id="test-client",
        )

        confluent_config = config.to_confluent_config()

        assert confluent_config["bootstrap.servers"] == "kafka1:9092,kafka2:9092"
        assert confluent_config["client.id"] == "test-client"


class TestProducerConfig:
    """Tests for producer configuration."""

    def test_default_producer_config(self) -> None:
        """Test default producer configuration."""
        config = ProducerConfig()

        assert config.acks == AcksMode.ALL
        assert config.compression_type == CompressionType.LZ4
        assert config.enable_idempotence is True
        assert config.batch_size == 16384

    def test_producer_config_transactional(self) -> None:
        """Test transactional producer configuration."""
        config = ProducerConfig(
            transactional_id="gl001-transactions",
            enable_idempotence=True,
        )

        assert config.transactional_id == "gl001-transactions"

    def test_producer_config_to_confluent(self) -> None:
        """Test conversion to confluent-kafka producer configuration."""
        config = ProducerConfig(
            acks=AcksMode.ALL,
            compression_type=CompressionType.SNAPPY,
            linger_ms=10,
        )

        confluent_config = config.to_confluent_config()

        assert confluent_config["acks"] == "all"
        assert confluent_config["compression.type"] == "snappy"
        assert confluent_config["linger.ms"] == 10


class TestConsumerConfig:
    """Tests for consumer configuration."""

    def test_consumer_config(self) -> None:
        """Test consumer configuration."""
        config = ConsumerConfig(
            group_id="test-consumer-group",
            auto_offset_reset=AutoOffsetReset.EARLIEST,
            enable_auto_commit=False,
        )

        assert config.group_id == "test-consumer-group"
        assert config.auto_offset_reset == AutoOffsetReset.EARLIEST
        assert config.enable_auto_commit is False
        assert config.isolation_level == "read_committed"

    def test_consumer_config_to_confluent(self) -> None:
        """Test conversion to confluent-kafka consumer configuration."""
        config = ConsumerConfig(
            group_id="test-group",
            max_poll_records=100,
        )

        confluent_config = config.to_confluent_config()

        assert confluent_config["group.id"] == "test-group"
        assert confluent_config["max.poll.records"] == 100


class TestTopicConfig:
    """Tests for topic configuration."""

    def test_topic_config(self) -> None:
        """Test topic configuration."""
        config = TopicConfig(
            name="gl001.telemetry.normalized",
            num_partitions=24,
            replication_factor=3,
            retention_ms=86400000,
        )

        assert config.name == "gl001.telemetry.normalized"
        assert config.num_partitions == 24
        assert config.schema_subject == "gl001.telemetry.normalized-value"

    def test_topic_config_validation(self) -> None:
        """Test topic name validation."""
        from pydantic import ValidationError

        # Valid topic name
        config = TopicConfig(name="gl001.safety.events")
        assert config.name == "gl001.safety.events"

        # Invalid topic name pattern
        with pytest.raises(ValidationError):
            TopicConfig(name="invalid.topic.name")


class TestPartitioners:
    """Tests for message partitioners."""

    def test_hash_partitioner(self) -> None:
        """Test hash-based partitioner."""
        partitioner = HashPartitioner()

        # Same key should always go to same partition
        partition1 = partitioner.partition(
            "test-topic",
            b"key-1",
            b"value",
            12,
        )
        partition2 = partitioner.partition(
            "test-topic",
            b"key-1",
            b"value",
            12,
        )

        assert partition1 == partition2
        assert 0 <= partition1 < 12

    def test_round_robin_partitioner(self) -> None:
        """Test round-robin partitioner."""
        partitioner = RoundRobinPartitioner()

        partitions = []
        for _ in range(12):
            partition = partitioner.partition(
                "test-topic",
                None,
                b"value",
                12,
            )
            partitions.append(partition)

        # Should have distributed across all partitions
        assert len(set(partitions)) == 12

    def test_equipment_partitioner(self) -> None:
        """Test equipment-based partitioner."""
        partitioner = EquipmentPartitioner()

        # Same equipment should go to same partition
        partition1 = partitioner.partition(
            "gl001.telemetry.normalized",
            b"boiler-01:sensor-001",
            b"value",
            12,
        )
        partition2 = partitioner.partition(
            "gl001.telemetry.normalized",
            b"boiler-01:sensor-002",
            b"value",
            12,
        )

        assert partition1 == partition2  # Same equipment prefix


class TestProducerMetrics:
    """Tests for producer metrics."""

    def test_metrics_record_success(self) -> None:
        """Test recording successful message production."""
        metrics = ProducerMetrics()

        metrics.record_success("gl001.telemetry.normalized", 1024, 5.5)
        metrics.record_success("gl001.telemetry.normalized", 2048, 6.5)

        assert metrics.messages_sent == 2
        assert metrics.bytes_sent == 3072
        assert metrics.messages_by_topic["gl001.telemetry.normalized"] == 2
        assert metrics.avg_latency_ms == 6.0  # (5.5 + 6.5) / 2

    def test_metrics_record_failure(self) -> None:
        """Test recording failed message production."""
        metrics = ProducerMetrics()

        metrics.record_failure(
            "gl001.telemetry.normalized",
            "KafkaError",
            "Connection refused",
        )

        assert metrics.messages_failed == 1
        assert metrics.errors_by_type["KafkaError"] == 1
        assert metrics.last_error == "Connection refused"

    def test_metrics_success_rate(self) -> None:
        """Test success rate calculation."""
        metrics = ProducerMetrics()

        metrics.record_success("topic", 100, 5.0)
        metrics.record_success("topic", 100, 5.0)
        metrics.record_failure("topic", "Error", "message")

        assert metrics.success_rate == pytest.approx(0.6667, rel=0.01)


class TestConsumerMetrics:
    """Tests for consumer metrics."""

    def test_metrics_record_consumed(self) -> None:
        """Test recording successful message consumption."""
        metrics = ConsumerMetrics()

        metrics.record_consumed("gl001.telemetry.normalized", 1024, 10.5)
        metrics.record_consumed("gl001.telemetry.normalized", 2048, 15.5)

        assert metrics.messages_consumed == 2
        assert metrics.messages_processed == 2
        assert metrics.bytes_consumed == 3072
        assert metrics.avg_processing_time_ms == 13.0

    def test_metrics_record_failure(self) -> None:
        """Test recording failed message processing."""
        metrics = ConsumerMetrics()

        metrics.record_failure("gl001.telemetry.normalized", "ParseError")

        assert metrics.messages_consumed == 1
        assert metrics.messages_failed == 1
        assert metrics.errors_by_type["ParseError"] == 1


class TestThermalCommandProducer:
    """Tests for ThermalCommand Kafka producer."""

    @pytest.fixture
    def producer(self) -> ThermalCommandProducer:
        """Create a test producer instance."""
        kafka_config = KafkaConfig(bootstrap_servers="localhost:9092")
        producer_config = ProducerConfig(enable_idempotence=True)
        return ThermalCommandProducer(kafka_config, producer_config)

    @pytest.mark.asyncio
    async def test_producer_start_stop(self, producer: ThermalCommandProducer) -> None:
        """Test producer start and stop lifecycle."""
        await producer.start()
        assert producer._running is True

        await producer.close()
        assert producer._running is False

    @pytest.mark.asyncio
    async def test_send_telemetry(self, producer: ThermalCommandProducer) -> None:
        """Test sending telemetry event."""
        await producer.start()

        try:
            now = datetime.now(timezone.utc)
            event = TelemetryNormalizedEvent(
                source_system="test-collector",
                points=[
                    TelemetryPoint(
                        tag_id="T-101",
                        value=450.5,
                        unit=UnitOfMeasure.CELSIUS,
                        timestamp=now,
                    )
                ],
                collection_timestamp=now,
                batch_id="test-batch-001",
                sequence_number=1,
            )

            report = await producer.send_telemetry(event, source="test")

            assert report.success is True
            assert report.topic == "gl001.telemetry.normalized"
            assert producer.metrics.messages_sent >= 1
        finally:
            await producer.close()

    @pytest.mark.asyncio
    async def test_send_dispatch_plan(self, producer: ThermalCommandProducer) -> None:
        """Test sending dispatch plan event."""
        await producer.start()

        try:
            now = datetime.now(timezone.utc)
            event = DispatchPlanEvent(
                plan_id="plan-test-001",
                horizon_start=now,
                horizon_end=now + timedelta(hours=24),
                allocations=[
                    LoadAllocation(
                        equipment_id="boiler-01",
                        load_mw=50.0,
                        min_load_mw=10.0,
                        max_load_mw=100.0,
                        efficiency_percent=92.0,
                        emissions_rate_kgco2_mwh=180.0,
                        fuel_type="natural_gas",
                        ramp_rate_mw_min=5.0,
                        marginal_cost_usd_mwh=45.0,
                    )
                ],
                solver_status=SolverStatus.OPTIMAL,
                solver_time_seconds=2.5,
                expected_impact=ExpectedImpact(
                    total_cost_usd=50000.0,
                    total_emissions_tco2=250.0,
                    average_efficiency_percent=92.0,
                ),
                demand_mw=50.0,
                created_by="test",
            )

            report = await producer.send_dispatch_plan(event, source="test")

            assert report.success is True
            assert report.topic == "gl001.plan.dispatch"
        finally:
            await producer.close()

    @pytest.mark.asyncio
    async def test_send_safety_event_high_priority(
        self, producer: ThermalCommandProducer
    ) -> None:
        """Test safety events are sent with high priority."""
        await producer.start()

        try:
            now = datetime.now(timezone.utc)
            event = SafetyEvent(
                event_id="safety-test-001",
                level=SafetyLevel.ALARM,
                event_timestamp=now,
                equipment_id="boiler-01",
                equipment_name="Main Boiler",
                boundary_violations=[
                    BoundaryViolation(
                        tag_id="T-101",
                        boundary_type="high",
                        limit_value=550.0,
                        actual_value=565.0,
                        deviation_percent=2.73,
                        unit=UnitOfMeasure.CELSIUS,
                    )
                ],
                operator_action_required=True,
            )

            report = await producer.send_safety_event(event, source="test")

            assert report.success is True
            assert report.topic == "gl001.safety.events"
        finally:
            await producer.close()

    @pytest.mark.asyncio
    async def test_send_audit_log(self, producer: ThermalCommandProducer) -> None:
        """Test sending audit log event."""
        await producer.start()

        try:
            now = datetime.now(timezone.utc)
            event = AuditLogEvent(
                audit_id="audit-test-001",
                action=AuditAction.CREATE,
                action_timestamp=now,
                actor_id="test-service",
                actor_type="service",
                resource_type="dispatch_plan",
                resource_id="plan-001",
                correlation_id="corr-test-001",
                outcome="success",
            )

            report = await producer.send_audit_log(event, source="test")

            assert report.success is True
            assert report.topic == "gl001.audit.log"
        finally:
            await producer.close()

    @pytest.mark.asyncio
    async def test_producer_not_running_error(
        self, producer: ThermalCommandProducer
    ) -> None:
        """Test that producing without starting raises error."""
        now = datetime.now(timezone.utc)
        event = TelemetryNormalizedEvent(
            source_system="test",
            points=[
                TelemetryPoint(
                    tag_id="T-101",
                    value=450.5,
                    unit=UnitOfMeasure.CELSIUS,
                    timestamp=now,
                )
            ],
            collection_timestamp=now,
            batch_id="batch-001",
            sequence_number=1,
        )

        with pytest.raises(RuntimeError, match="not running"):
            await producer.send_telemetry(event)

    @pytest.mark.asyncio
    async def test_producer_flush(self, producer: ThermalCommandProducer) -> None:
        """Test flushing pending messages."""
        await producer.start()

        try:
            count = await producer.flush()
            assert count >= 0
        finally:
            await producer.close()

    def test_producer_get_metrics(self, producer: ThermalCommandProducer) -> None:
        """Test getting producer metrics."""
        metrics = producer.get_metrics()

        assert isinstance(metrics, ProducerMetrics)
        assert metrics.messages_sent == 0


class TestThermalCommandConsumer:
    """Tests for ThermalCommand Kafka consumer."""

    @pytest.fixture
    def consumer(self) -> ThermalCommandConsumer:
        """Create a test consumer instance."""
        kafka_config = KafkaConfig(bootstrap_servers="localhost:9092")
        consumer_config = ConsumerConfig(group_id="test-consumer-group")
        return ThermalCommandConsumer(kafka_config, consumer_config)

    @pytest.mark.asyncio
    async def test_consumer_start_stop(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test consumer start and stop lifecycle."""
        await consumer.start()
        assert consumer._running is True

        await consumer.close()
        assert consumer._running is False

    @pytest.mark.asyncio
    async def test_consumer_subscribe(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test subscribing to topics."""
        await consumer.start()

        try:
            await consumer.subscribe([
                ThermalCommandConsumer.TOPIC_TELEMETRY,
                ThermalCommandConsumer.TOPIC_SAFETY,
            ])

            assert ThermalCommandConsumer.TOPIC_TELEMETRY in consumer._subscribed_topics
            assert ThermalCommandConsumer.TOPIC_SAFETY in consumer._subscribed_topics
        finally:
            await consumer.close()

    @pytest.mark.asyncio
    async def test_consumer_unsubscribe(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test unsubscribing from topics."""
        await consumer.start()

        try:
            await consumer.subscribe([ThermalCommandConsumer.TOPIC_TELEMETRY])
            await consumer.unsubscribe()

            assert len(consumer._subscribed_topics) == 0
        finally:
            await consumer.close()

    @pytest.mark.asyncio
    async def test_consumer_not_running_error(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test that consuming without starting raises error."""
        with pytest.raises(RuntimeError, match="not running"):
            async for _ in consumer.consume():
                pass

    def test_consumer_add_handler(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test adding message handler."""
        handler = AsyncMock()

        consumer.add_handler(ThermalCommandConsumer.TOPIC_TELEMETRY, handler)

        assert len(consumer._handlers[ThermalCommandConsumer.TOPIC_TELEMETRY]) == 1

    def test_consumer_get_metrics(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test getting consumer metrics."""
        metrics = consumer.get_metrics()

        assert isinstance(metrics, ConsumerMetrics)
        assert metrics.messages_consumed == 0

    @pytest.mark.asyncio
    async def test_consumer_commit(
        self, consumer: ThermalCommandConsumer
    ) -> None:
        """Test committing offsets."""
        await consumer.start()

        try:
            await consumer.subscribe([ThermalCommandConsumer.TOPIC_TELEMETRY])
            await consumer.commit()

            # Metrics should track commit
            # (In mock, no actual commit happens)
        finally:
            await consumer.close()


class TestTopicConstants:
    """Tests for topic constant definitions."""

    def test_producer_topic_constants(self) -> None:
        """Test producer topic constants match schema registry."""
        from ..kafka_schemas import TopicSchemaRegistry

        topics = TopicSchemaRegistry.list_topics()

        assert ThermalCommandProducer.TOPIC_TELEMETRY in topics
        assert ThermalCommandProducer.TOPIC_DISPATCH in topics
        assert ThermalCommandProducer.TOPIC_RECOMMENDATIONS in topics
        assert ThermalCommandProducer.TOPIC_SAFETY in topics
        assert ThermalCommandProducer.TOPIC_MAINTENANCE in topics
        assert ThermalCommandProducer.TOPIC_EXPLAINABILITY in topics
        assert ThermalCommandProducer.TOPIC_AUDIT in topics

    def test_consumer_topic_constants_match_producer(self) -> None:
        """Test consumer topic constants match producer."""
        assert (
            ThermalCommandConsumer.TOPIC_TELEMETRY
            == ThermalCommandProducer.TOPIC_TELEMETRY
        )
        assert (
            ThermalCommandConsumer.TOPIC_DISPATCH
            == ThermalCommandProducer.TOPIC_DISPATCH
        )
        assert (
            ThermalCommandConsumer.TOPIC_SAFETY
            == ThermalCommandProducer.TOPIC_SAFETY
        )
        assert (
            ThermalCommandConsumer.TOPIC_AUDIT
            == ThermalCommandProducer.TOPIC_AUDIT
        )
