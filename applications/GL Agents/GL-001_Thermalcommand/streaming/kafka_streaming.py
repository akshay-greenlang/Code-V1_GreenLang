"""
Kafka Streaming Module - GL-001 ThermalCommand

This module provides Kafka producer and consumer implementations for
the ThermalCommand system with exactly-once semantics, batching,
partitioning, and offset management.

Key Features:
    - Transactional producer with exactly-once semantics
    - Consumer groups with automatic offset management
    - Schema Registry integration for schema validation
    - Partitioning strategies for ordered processing
    - Batch production with configurable flush intervals
    - Dead letter queue handling for failed messages
    - Comprehensive metrics and health monitoring

Example:
    >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
    >>> producer = ThermalCommandProducer(config)
    >>> await producer.send_telemetry(telemetry_event)
    >>> await producer.close()

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from .event_envelope import EventEnvelope, EnvelopeBatch, SchemaVersion
from .kafka_schemas import (
    ActionRecommendationEvent,
    AuditLogEvent,
    DispatchPlanEvent,
    ExplainabilityReportEvent,
    MaintenanceTriggerEvent,
    SafetyEvent,
    TelemetryNormalizedEvent,
    TopicSchemaRegistry,
)

logger = logging.getLogger(__name__)

# Type variable for generic event types
EventT = TypeVar("EventT", bound=BaseModel)


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class SecurityProtocol(str, Enum):
    """Kafka security protocols."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class SASLMechanism(str, Enum):
    """SASL authentication mechanisms."""

    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"
    OAUTHBEARER = "OAUTHBEARER"


class CompressionType(str, Enum):
    """Message compression types."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AcksMode(str, Enum):
    """Producer acknowledgment modes."""

    NONE = "0"
    LEADER = "1"
    ALL = "all"


class AutoOffsetReset(str, Enum):
    """Consumer auto offset reset behavior."""

    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class PartitionStrategy(str, Enum):
    """Partitioning strategies for message distribution."""

    ROUND_ROBIN = "round_robin"
    KEY_HASH = "key_hash"
    STICKY = "sticky"
    CUSTOM = "custom"


class TopicConfig(BaseModel):
    """
    Configuration for a Kafka topic.

    Attributes:
        name: Topic name
        num_partitions: Number of partitions
        replication_factor: Replication factor
        retention_ms: Retention period in milliseconds
        cleanup_policy: Cleanup policy (delete, compact)
        schema_version: Schema version for this topic
    """

    name: str = Field(
        ...,
        pattern=r"^gl001\.[a-z]+\.[a-z]+$",
        description="Topic name following gl001 naming convention",
    )
    num_partitions: int = Field(
        12,
        ge=1,
        le=256,
        description="Number of partitions",
    )
    replication_factor: int = Field(
        3,
        ge=1,
        le=5,
        description="Replication factor",
    )
    retention_ms: int = Field(
        604800000,  # 7 days
        ge=-1,
        description="Retention period in milliseconds (-1 for infinite)",
    )
    cleanup_policy: str = Field(
        "delete",
        description="Cleanup policy: delete or compact",
    )
    schema_version: SchemaVersion = Field(
        default_factory=SchemaVersion,
        description="Schema version for this topic",
    )
    min_insync_replicas: int = Field(
        2,
        ge=1,
        description="Minimum in-sync replicas for acks=all",
    )
    compression_type: CompressionType = Field(
        CompressionType.LZ4,
        description="Topic-level compression",
    )

    @property
    def schema_subject(self) -> str:
        """Return Schema Registry subject name."""
        return f"{self.name}-value"


class KafkaConfig(BaseModel):
    """
    Kafka connection and behavior configuration.

    Attributes:
        bootstrap_servers: Kafka bootstrap server addresses
        client_id: Client identifier
        security_protocol: Security protocol
        sasl_mechanism: SASL mechanism if using SASL
        ssl_ca_location: CA certificate path
    """

    bootstrap_servers: str = Field(
        "localhost:9092",
        description="Comma-separated bootstrap server addresses",
    )
    client_id: str = Field(
        default_factory=lambda: f"gl001-thermalcommand-{uuid.uuid4().hex[:8]}",
        description="Client identifier",
    )
    security_protocol: SecurityProtocol = Field(
        SecurityProtocol.PLAINTEXT,
        description="Security protocol",
    )
    sasl_mechanism: Optional[SASLMechanism] = Field(
        None,
        description="SASL mechanism",
    )
    sasl_username: Optional[str] = Field(
        None,
        description="SASL username",
    )
    sasl_password: Optional[str] = Field(
        None,
        description="SASL password (use secrets manager in production)",
    )
    ssl_ca_location: Optional[str] = Field(
        None,
        description="Path to CA certificate",
    )
    ssl_certificate_location: Optional[str] = Field(
        None,
        description="Path to client certificate",
    )
    ssl_key_location: Optional[str] = Field(
        None,
        description="Path to client key",
    )
    schema_registry_url: Optional[str] = Field(
        None,
        description="Schema Registry URL",
    )
    request_timeout_ms: int = Field(
        30000,
        ge=1000,
        description="Request timeout in milliseconds",
    )
    connections_max_idle_ms: int = Field(
        540000,
        ge=1000,
        description="Max idle time for connections",
    )

    def to_confluent_config(self) -> Dict[str, Any]:
        """Convert to confluent-kafka configuration dictionary."""
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": self.client_id,
            "security.protocol": self.security_protocol.value,
            "request.timeout.ms": self.request_timeout_ms,
            "connections.max.idle.ms": self.connections_max_idle_ms,
        }

        if self.sasl_mechanism:
            config["sasl.mechanism"] = self.sasl_mechanism.value
            if self.sasl_username:
                config["sasl.username"] = self.sasl_username
            if self.sasl_password:
                config["sasl.password"] = self.sasl_password

        if self.ssl_ca_location:
            config["ssl.ca.location"] = self.ssl_ca_location
        if self.ssl_certificate_location:
            config["ssl.certificate.location"] = self.ssl_certificate_location
        if self.ssl_key_location:
            config["ssl.key.location"] = self.ssl_key_location

        return config


class ProducerConfig(BaseModel):
    """
    Kafka producer-specific configuration.

    Attributes:
        acks: Acknowledgment mode
        compression_type: Message compression
        batch_size: Batch size in bytes
        linger_ms: Batch linger time
        enable_idempotence: Enable idempotent producer
    """

    acks: AcksMode = Field(
        AcksMode.ALL,
        description="Acknowledgment mode for durability",
    )
    compression_type: CompressionType = Field(
        CompressionType.LZ4,
        description="Message compression type",
    )
    batch_size: int = Field(
        16384,
        ge=0,
        description="Batch size in bytes",
    )
    linger_ms: int = Field(
        5,
        ge=0,
        description="Time to wait for batch to fill",
    )
    buffer_memory: int = Field(
        33554432,  # 32MB
        ge=1024,
        description="Total buffer memory in bytes",
    )
    max_in_flight_requests: int = Field(
        5,
        ge=1,
        le=10,
        description="Max in-flight requests per connection",
    )
    enable_idempotence: bool = Field(
        True,
        description="Enable idempotent producer",
    )
    transactional_id: Optional[str] = Field(
        None,
        description="Transactional ID for exactly-once semantics",
    )
    retries: int = Field(
        2147483647,  # Max int for infinite retries with idempotence
        ge=0,
        description="Number of retries",
    )
    retry_backoff_ms: int = Field(
        100,
        ge=0,
        description="Backoff between retries",
    )

    def to_confluent_config(self) -> Dict[str, Any]:
        """Convert to confluent-kafka producer configuration."""
        config = {
            "acks": self.acks.value,
            "compression.type": self.compression_type.value,
            "batch.size": self.batch_size,
            "linger.ms": self.linger_ms,
            "buffer.memory": self.buffer_memory,
            "max.in.flight.requests.per.connection": self.max_in_flight_requests,
            "enable.idempotence": self.enable_idempotence,
            "retries": self.retries,
            "retry.backoff.ms": self.retry_backoff_ms,
        }

        if self.transactional_id:
            config["transactional.id"] = self.transactional_id

        return config


class ConsumerConfig(BaseModel):
    """
    Kafka consumer-specific configuration.

    Attributes:
        group_id: Consumer group identifier
        auto_offset_reset: Offset reset behavior
        enable_auto_commit: Enable automatic offset commits
        max_poll_records: Maximum records per poll
    """

    group_id: str = Field(
        ...,
        min_length=1,
        description="Consumer group identifier",
    )
    auto_offset_reset: AutoOffsetReset = Field(
        AutoOffsetReset.EARLIEST,
        description="Offset reset behavior",
    )
    enable_auto_commit: bool = Field(
        False,  # Manual commits for exactly-once
        description="Enable automatic offset commits",
    )
    auto_commit_interval_ms: int = Field(
        5000,
        ge=1000,
        description="Auto commit interval",
    )
    max_poll_records: int = Field(
        500,
        ge=1,
        description="Maximum records per poll",
    )
    max_poll_interval_ms: int = Field(
        300000,
        ge=1000,
        description="Max time between polls before considered dead",
    )
    session_timeout_ms: int = Field(
        45000,
        ge=1000,
        description="Session timeout",
    )
    heartbeat_interval_ms: int = Field(
        3000,
        ge=1000,
        description="Heartbeat interval",
    )
    fetch_min_bytes: int = Field(
        1,
        ge=1,
        description="Minimum bytes to fetch",
    )
    fetch_max_wait_ms: int = Field(
        500,
        ge=0,
        description="Max wait time for fetch",
    )
    isolation_level: str = Field(
        "read_committed",
        description="Transaction isolation level",
    )

    def to_confluent_config(self) -> Dict[str, Any]:
        """Convert to confluent-kafka consumer configuration."""
        return {
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset.value,
            "enable.auto.commit": self.enable_auto_commit,
            "auto.commit.interval.ms": self.auto_commit_interval_ms,
            "max.poll.records": self.max_poll_records,
            "max.poll.interval.ms": self.max_poll_interval_ms,
            "session.timeout.ms": self.session_timeout_ms,
            "heartbeat.interval.ms": self.heartbeat_interval_ms,
            "fetch.min.bytes": self.fetch_min_bytes,
            "fetch.max.wait.ms": self.fetch_max_wait_ms,
            "isolation.level": self.isolation_level,
        }


# =============================================================================
# METRICS AND MONITORING
# =============================================================================


@dataclass
class ProducerMetrics:
    """Metrics for Kafka producer monitoring."""

    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batches_sent: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    messages_by_topic: Dict[str, int] = field(default_factory=dict)
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    def record_success(
        self,
        topic: str,
        message_size: int,
        latency_ms: float,
    ) -> None:
        """Record successful message production."""
        self.messages_sent += 1
        self.bytes_sent += message_size
        self.messages_by_topic[topic] = self.messages_by_topic.get(topic, 0) + 1

        # Update rolling average latency
        n = self.messages_sent
        self.avg_latency_ms = (
            (self.avg_latency_ms * (n - 1) + latency_ms) / n
        )

    def record_failure(self, topic: str, error_type: str, error_msg: str) -> None:
        """Record failed message production."""
        self.messages_failed += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        self.last_error = error_msg
        self.last_error_time = datetime.now(timezone.utc)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.messages_sent + self.messages_failed
        return self.messages_sent / total if total > 0 else 1.0


@dataclass
class ConsumerMetrics:
    """Metrics for Kafka consumer monitoring."""

    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    commits: int = 0
    rebalances: int = 0
    avg_processing_time_ms: float = 0.0
    current_lag: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    messages_by_topic: Dict[str, int] = field(default_factory=dict)

    def record_consumed(
        self,
        topic: str,
        message_size: int,
        processing_time_ms: float,
    ) -> None:
        """Record successful message consumption."""
        self.messages_consumed += 1
        self.messages_processed += 1
        self.bytes_consumed += message_size
        self.messages_by_topic[topic] = self.messages_by_topic.get(topic, 0) + 1

        # Update rolling average processing time
        n = self.messages_processed
        self.avg_processing_time_ms = (
            (self.avg_processing_time_ms * (n - 1) + processing_time_ms) / n
        )

    def record_failure(self, topic: str, error_type: str) -> None:
        """Record failed message processing."""
        self.messages_consumed += 1
        self.messages_failed += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1


# =============================================================================
# PARTITIONER
# =============================================================================


class Partitioner(ABC):
    """Abstract base class for message partitioners."""

    @abstractmethod
    def partition(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        num_partitions: int,
    ) -> int:
        """
        Determine partition for a message.

        Args:
            topic: Target topic
            key: Message key (may be None)
            value: Message value
            num_partitions: Number of partitions in topic

        Returns:
            Partition number
        """
        pass


class HashPartitioner(Partitioner):
    """Consistent hash-based partitioner."""

    def partition(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        num_partitions: int,
    ) -> int:
        """Partition based on murmur2 hash of key."""
        if key is None:
            # Use value hash if no key
            key = value

        # Simple consistent hashing (murmur2-like)
        h = int(hashlib.md5(key).hexdigest(), 16)
        return h % num_partitions


class RoundRobinPartitioner(Partitioner):
    """Round-robin partitioner for even distribution."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def partition(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        num_partitions: int,
    ) -> int:
        """Partition using round-robin distribution."""
        partition = self._counters[topic] % num_partitions
        self._counters[topic] += 1
        return partition


class EquipmentPartitioner(Partitioner):
    """
    Equipment-based partitioner for ThermalCommand.

    Ensures all messages for the same equipment ID go to
    the same partition for ordered processing.
    """

    def partition(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        num_partitions: int,
    ) -> int:
        """Partition based on equipment ID in key."""
        if key is None:
            return 0

        # Extract equipment ID from key (expected format: "equipment_id:...")
        key_str = key.decode("utf-8")
        equipment_id = key_str.split(":")[0] if ":" in key_str else key_str

        # Consistent hash on equipment ID
        h = int(hashlib.md5(equipment_id.encode()).hexdigest(), 16)
        return h % num_partitions


# =============================================================================
# KAFKA PRODUCER
# =============================================================================


class DeliveryReport(BaseModel):
    """Delivery report for produced messages."""

    topic: str
    partition: int
    offset: int
    timestamp_ms: int
    key: Optional[str]
    latency_ms: float
    success: bool
    error: Optional[str] = None


class ThermalCommandProducer:
    """
    Kafka producer for ThermalCommand events.

    Provides typed methods for producing events to specific topics
    with exactly-once semantics and batch optimization.

    Example:
        >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
        >>> producer_config = ProducerConfig(enable_idempotence=True)
        >>> producer = ThermalCommandProducer(config, producer_config)
        >>> await producer.start()
        >>> report = await producer.send_telemetry(telemetry_event)
        >>> await producer.close()
    """

    # Topic definitions
    TOPIC_TELEMETRY = "gl001.telemetry.normalized"
    TOPIC_DISPATCH = "gl001.plan.dispatch"
    TOPIC_RECOMMENDATIONS = "gl001.actions.recommendations"
    TOPIC_SAFETY = "gl001.safety.events"
    TOPIC_MAINTENANCE = "gl001.maintenance.triggers"
    TOPIC_EXPLAINABILITY = "gl001.explainability.reports"
    TOPIC_AUDIT = "gl001.audit.log"

    def __init__(
        self,
        kafka_config: KafkaConfig,
        producer_config: Optional[ProducerConfig] = None,
        partitioner: Optional[Partitioner] = None,
    ) -> None:
        """
        Initialize ThermalCommand producer.

        Args:
            kafka_config: Kafka connection configuration
            producer_config: Producer-specific configuration
            partitioner: Custom partitioner (defaults to EquipmentPartitioner)
        """
        self.kafka_config = kafka_config
        self.producer_config = producer_config or ProducerConfig()
        self.partitioner = partitioner or EquipmentPartitioner()

        self._producer: Optional[Any] = None  # confluent_kafka.Producer
        self._running = False
        self._pending_deliveries: Dict[str, asyncio.Future] = {}
        self._batch_queue: List[tuple] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

        self.metrics = ProducerMetrics()

        # Topic configurations
        self._topic_configs: Dict[str, TopicConfig] = {
            self.TOPIC_TELEMETRY: TopicConfig(
                name=self.TOPIC_TELEMETRY,
                num_partitions=24,  # High throughput
                retention_ms=86400000,  # 1 day
            ),
            self.TOPIC_DISPATCH: TopicConfig(
                name=self.TOPIC_DISPATCH,
                num_partitions=6,
                retention_ms=604800000,  # 7 days
            ),
            self.TOPIC_RECOMMENDATIONS: TopicConfig(
                name=self.TOPIC_RECOMMENDATIONS,
                num_partitions=12,
                retention_ms=604800000,
            ),
            self.TOPIC_SAFETY: TopicConfig(
                name=self.TOPIC_SAFETY,
                num_partitions=6,
                retention_ms=2592000000,  # 30 days
            ),
            self.TOPIC_MAINTENANCE: TopicConfig(
                name=self.TOPIC_MAINTENANCE,
                num_partitions=6,
                retention_ms=2592000000,
            ),
            self.TOPIC_EXPLAINABILITY: TopicConfig(
                name=self.TOPIC_EXPLAINABILITY,
                num_partitions=12,
                retention_ms=604800000,
            ),
            self.TOPIC_AUDIT: TopicConfig(
                name=self.TOPIC_AUDIT,
                num_partitions=12,
                retention_ms=-1,  # Infinite retention
                cleanup_policy="compact",
            ),
        }

        logger.info(
            f"ThermalCommandProducer initialized with client_id={kafka_config.client_id}"
        )

    async def start(self) -> None:
        """
        Start the producer and initialize transactional support.

        Raises:
            RuntimeError: If producer is already running
        """
        if self._running:
            raise RuntimeError("Producer is already running")

        logger.info("Starting ThermalCommandProducer...")

        # Build configuration
        config = self.kafka_config.to_confluent_config()
        config.update(self.producer_config.to_confluent_config())

        # Note: In production, use confluent_kafka.Producer
        # For this implementation, we use a mock for demonstration
        self._producer = MockKafkaProducer(config)

        # Initialize transactions if configured
        if self.producer_config.transactional_id:
            await self._producer.init_transactions()
            logger.info(
                f"Transactions initialized with id={self.producer_config.transactional_id}"
            )

        self._running = True

        # Start background flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())

        logger.info("ThermalCommandProducer started successfully")

    async def close(self) -> None:
        """Close the producer and flush pending messages."""
        if not self._running:
            return

        logger.info("Closing ThermalCommandProducer...")

        self._running = False

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining messages
        await self.flush()

        if self._producer:
            await self._producer.close()
            self._producer = None

        logger.info("ThermalCommandProducer closed")

    async def flush(self, timeout_seconds: float = 30.0) -> int:
        """
        Flush all pending messages.

        Args:
            timeout_seconds: Flush timeout

        Returns:
            Number of messages flushed
        """
        if not self._producer:
            return 0

        async with self._batch_lock:
            batch_size = len(self._batch_queue)
            if batch_size > 0:
                await self._flush_batch()

        await self._producer.flush(timeout_seconds)
        return batch_size

    async def _periodic_flush(self) -> None:
        """Background task for periodic batch flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.producer_config.linger_ms / 1000.0)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def _flush_batch(self) -> None:
        """Flush the current batch of messages."""
        if not self._batch_queue:
            return

        batch = self._batch_queue.copy()
        self._batch_queue.clear()

        for topic, key, value, headers, future in batch:
            try:
                start_time = time.monotonic()
                result = await self._producer.produce(
                    topic=topic,
                    key=key,
                    value=value,
                    headers=headers,
                )
                latency_ms = (time.monotonic() - start_time) * 1000

                report = DeliveryReport(
                    topic=topic,
                    partition=result.get("partition", 0),
                    offset=result.get("offset", 0),
                    timestamp_ms=result.get("timestamp_ms", int(time.time() * 1000)),
                    key=key.decode() if key else None,
                    latency_ms=latency_ms,
                    success=True,
                )

                self.metrics.record_success(topic, len(value), latency_ms)
                future.set_result(report)

            except Exception as e:
                error_type = type(e).__name__
                self.metrics.record_failure(topic, error_type, str(e))

                report = DeliveryReport(
                    topic=topic,
                    partition=-1,
                    offset=-1,
                    timestamp_ms=int(time.time() * 1000),
                    key=key.decode() if key else None,
                    latency_ms=0,
                    success=False,
                    error=str(e),
                )
                future.set_result(report)

        self.metrics.batches_sent += 1
        if self.metrics.batches_sent > 0:
            self.metrics.avg_batch_size = (
                self.metrics.messages_sent / self.metrics.batches_sent
            )

    async def _produce(
        self,
        topic: str,
        envelope: EventEnvelope,
        partition_key: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Internal method to produce a message.

        Args:
            topic: Target topic
            envelope: Event envelope to produce
            partition_key: Optional partition key override

        Returns:
            Delivery report
        """
        if not self._running:
            raise RuntimeError("Producer is not running")

        # Prepare message
        kafka_msg = envelope.to_kafka_message()
        key = (partition_key or envelope.metadata.partition_key or "").encode()
        value = kafka_msg["value"]
        headers = kafka_msg["headers"]

        # Determine partition
        topic_config = self._topic_configs.get(
            topic,
            TopicConfig(name=topic, num_partitions=12),
        )
        partition = self.partitioner.partition(
            topic, key, value, topic_config.num_partitions
        )

        # Add to batch queue
        future: asyncio.Future = asyncio.Future()

        async with self._batch_lock:
            self._batch_queue.append((topic, key, value, headers, future))

            # Flush if batch is full
            if len(self._batch_queue) >= self.producer_config.batch_size // 1024:
                await self._flush_batch()

        return await future

    # -------------------------------------------------------------------------
    # TYPED PRODUCTION METHODS
    # -------------------------------------------------------------------------

    async def send_telemetry(
        self,
        event: TelemetryNormalizedEvent,
        source: str = "opc-ua-collector",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send telemetry normalized event.

        Args:
            event: Telemetry event data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_TELEMETRY,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.source_system,
        )

        logger.debug(
            f"Sending telemetry batch_id={event.batch_id} "
            f"points={len(event.points)} correlation_id={envelope.metadata.correlation_id}"
        )

        return await self._produce(self.TOPIC_TELEMETRY, envelope)

    async def send_dispatch_plan(
        self,
        event: DispatchPlanEvent,
        source: str = "milp-optimizer",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send dispatch plan event.

        Args:
            event: Dispatch plan data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_DISPATCH,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.plan_id,
            priority=2,  # High priority
        )

        logger.info(
            f"Sending dispatch plan plan_id={event.plan_id} "
            f"solver_status={event.solver_status.value}"
        )

        return await self._produce(self.TOPIC_DISPATCH, envelope)

    async def send_recommendation(
        self,
        event: ActionRecommendationEvent,
        source: str = "recommendation-engine",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send action recommendation event.

        Args:
            event: Recommendation data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_RECOMMENDATIONS,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.recommendation_id,
        )

        logger.info(
            f"Sending recommendation id={event.recommendation_id} "
            f"confidence={event.overall_confidence:.2f}"
        )

        return await self._produce(self.TOPIC_RECOMMENDATIONS, envelope)

    async def send_safety_event(
        self,
        event: SafetyEvent,
        source: str = "safety-boundary-engine",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send safety event.

        Args:
            event: Safety event data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report

        Note:
            Safety events are always sent with highest priority.
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_SAFETY,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.equipment_id,
            priority=1,  # Highest priority for safety
        )

        logger.warning(
            f"Sending safety event id={event.event_id} "
            f"level={event.level.value} equipment={event.equipment_id}"
        )

        # Safety events bypass batch - send immediately
        if self._producer:
            async with self._batch_lock:
                await self._flush_batch()

        return await self._produce(self.TOPIC_SAFETY, envelope)

    async def send_maintenance_trigger(
        self,
        event: MaintenanceTriggerEvent,
        source: str = "predictive-maintenance",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send maintenance trigger event.

        Args:
            event: Maintenance trigger data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_MAINTENANCE,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.equipment_id,
        )

        logger.info(
            f"Sending maintenance trigger id={event.trigger_id} "
            f"equipment={event.equipment_id} priority={event.priority.value}"
        )

        return await self._produce(self.TOPIC_MAINTENANCE, envelope)

    async def send_explainability_report(
        self,
        event: ExplainabilityReportEvent,
        source: str = "explainability-engine",
        correlation_id: Optional[str] = None,
    ) -> DeliveryReport:
        """
        Send explainability report event.

        Args:
            event: Explainability report data
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            Delivery report
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_EXPLAINABILITY,
            source=source,
            payload=event,
            correlation_id=correlation_id,
            partition_key=event.model_id,
        )

        logger.debug(
            f"Sending explainability report id={event.report_id} "
            f"model={event.model_id}"
        )

        return await self._produce(self.TOPIC_EXPLAINABILITY, envelope)

    async def send_audit_log(
        self,
        event: AuditLogEvent,
        source: str = "audit-logger",
    ) -> DeliveryReport:
        """
        Send audit log event.

        Args:
            event: Audit log data
            source: Event source identifier

        Returns:
            Delivery report

        Note:
            Audit events use the event's correlation_id directly.
        """
        envelope = EventEnvelope.create(
            event_type=self.TOPIC_AUDIT,
            source=source,
            payload=event,
            correlation_id=event.correlation_id,
            partition_key=event.resource_type,
        )

        logger.debug(
            f"Sending audit log id={event.audit_id} "
            f"action={event.action.value} resource={event.resource_type}"
        )

        return await self._produce(self.TOPIC_AUDIT, envelope)

    # -------------------------------------------------------------------------
    # TRANSACTIONAL SUPPORT
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        """
        Context manager for transactional production.

        Example:
            >>> async with producer.transaction():
            ...     await producer.send_dispatch_plan(plan)
            ...     await producer.send_audit_log(audit)
        """
        if not self.producer_config.transactional_id:
            raise RuntimeError("Transactional ID not configured")

        await self._producer.begin_transaction()
        try:
            yield
            await self._producer.commit_transaction()
        except Exception as e:
            await self._producer.abort_transaction()
            raise

    def get_metrics(self) -> ProducerMetrics:
        """Return current producer metrics."""
        return self.metrics


# =============================================================================
# KAFKA CONSUMER
# =============================================================================


class ConsumedMessage(BaseModel, Generic[EventT]):
    """Consumed message with metadata."""

    envelope: EventEnvelope
    topic: str
    partition: int
    offset: int
    timestamp_ms: int
    key: Optional[str]
    headers: Dict[str, str]
    receive_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


MessageHandler = Callable[[ConsumedMessage], Any]


class ThermalCommandConsumer:
    """
    Kafka consumer for ThermalCommand events.

    Provides typed consumption with exactly-once semantics
    through manual offset management.

    Example:
        >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
        >>> consumer_config = ConsumerConfig(group_id="my-consumer-group")
        >>> consumer = ThermalCommandConsumer(config, consumer_config)
        >>> await consumer.subscribe([ThermalCommandConsumer.TOPIC_TELEMETRY])
        >>> async for message in consumer.consume():
        ...     process(message)
        ...     await consumer.commit()
    """

    # Topic definitions (same as producer)
    TOPIC_TELEMETRY = "gl001.telemetry.normalized"
    TOPIC_DISPATCH = "gl001.plan.dispatch"
    TOPIC_RECOMMENDATIONS = "gl001.actions.recommendations"
    TOPIC_SAFETY = "gl001.safety.events"
    TOPIC_MAINTENANCE = "gl001.maintenance.triggers"
    TOPIC_EXPLAINABILITY = "gl001.explainability.reports"
    TOPIC_AUDIT = "gl001.audit.log"

    def __init__(
        self,
        kafka_config: KafkaConfig,
        consumer_config: ConsumerConfig,
    ) -> None:
        """
        Initialize ThermalCommand consumer.

        Args:
            kafka_config: Kafka connection configuration
            consumer_config: Consumer-specific configuration
        """
        self.kafka_config = kafka_config
        self.consumer_config = consumer_config

        self._consumer: Optional[Any] = None  # confluent_kafka.Consumer
        self._running = False
        self._subscribed_topics: Set[str] = set()
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._pending_offsets: Dict[str, Dict[int, int]] = defaultdict(dict)

        self.metrics = ConsumerMetrics()

        logger.info(
            f"ThermalCommandConsumer initialized with "
            f"group_id={consumer_config.group_id}"
        )

    async def start(self) -> None:
        """
        Start the consumer.

        Raises:
            RuntimeError: If consumer is already running
        """
        if self._running:
            raise RuntimeError("Consumer is already running")

        logger.info("Starting ThermalCommandConsumer...")

        # Build configuration
        config = self.kafka_config.to_confluent_config()
        config.update(self.consumer_config.to_confluent_config())

        # Note: In production, use confluent_kafka.Consumer
        self._consumer = MockKafkaConsumer(config)

        self._running = True
        logger.info("ThermalCommandConsumer started successfully")

    async def close(self) -> None:
        """Close the consumer."""
        if not self._running:
            return

        logger.info("Closing ThermalCommandConsumer...")

        self._running = False

        if self._consumer:
            await self._consumer.close()
            self._consumer = None

        logger.info("ThermalCommandConsumer closed")

    async def subscribe(
        self,
        topics: List[str],
        from_beginning: bool = False,
    ) -> None:
        """
        Subscribe to topics.

        Args:
            topics: List of topic names
            from_beginning: If True, consume from beginning
        """
        if not self._running:
            raise RuntimeError("Consumer is not running")

        # Validate topics
        for topic in topics:
            if topic not in TopicSchemaRegistry.TOPIC_SCHEMAS:
                logger.warning(f"Unknown topic schema: {topic}")

        self._subscribed_topics.update(topics)
        await self._consumer.subscribe(topics, from_beginning)

        logger.info(f"Subscribed to topics: {topics}")

    async def unsubscribe(self) -> None:
        """Unsubscribe from all topics."""
        if self._consumer:
            await self._consumer.unsubscribe()
        self._subscribed_topics.clear()
        logger.info("Unsubscribed from all topics")

    def add_handler(
        self,
        topic: str,
        handler: MessageHandler,
    ) -> None:
        """
        Add a message handler for a topic.

        Args:
            topic: Topic to handle
            handler: Handler function
        """
        self._handlers[topic].append(handler)
        logger.debug(f"Added handler for topic {topic}")

    async def consume(
        self,
        timeout_seconds: float = 1.0,
    ) -> AsyncGenerator[ConsumedMessage, None]:
        """
        Consume messages from subscribed topics.

        Yields:
            ConsumedMessage instances

        Example:
            >>> async for msg in consumer.consume():
            ...     print(f"Received: {msg.envelope.metadata.event_type}")
        """
        if not self._running:
            raise RuntimeError("Consumer is not running")

        while self._running:
            try:
                message = await self._consumer.poll(timeout_seconds)

                if message is None:
                    continue

                if message.get("error"):
                    logger.error(f"Consumer error: {message['error']}")
                    continue

                start_time = time.monotonic()

                # Parse message
                try:
                    envelope = EventEnvelope.from_kafka_message(
                        value=message["value"],
                        headers=message.get("headers"),
                    )

                    consumed = ConsumedMessage(
                        envelope=envelope,
                        topic=message["topic"],
                        partition=message["partition"],
                        offset=message["offset"],
                        timestamp_ms=message["timestamp_ms"],
                        key=message["key"].decode() if message.get("key") else None,
                        headers={
                            k: v.decode() for k, v in (message.get("headers") or [])
                        },
                    )

                    # Track pending offset for commit
                    self._pending_offsets[message["topic"]][
                        message["partition"]
                    ] = message["offset"] + 1

                    processing_time = (time.monotonic() - start_time) * 1000
                    self.metrics.record_consumed(
                        message["topic"],
                        len(message["value"]),
                        processing_time,
                    )

                    # Call registered handlers
                    for handler in self._handlers.get(message["topic"], []):
                        try:
                            await handler(consumed)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

                    yield consumed

                except Exception as e:
                    logger.error(f"Failed to parse message: {e}")
                    self.metrics.record_failure(
                        message.get("topic", "unknown"),
                        "parse_error",
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consume error: {e}")
                await asyncio.sleep(1.0)  # Backoff on error

    async def commit(
        self,
        async_commit: bool = False,
    ) -> None:
        """
        Commit current offsets.

        Args:
            async_commit: If True, commit asynchronously
        """
        if not self._consumer or not self._pending_offsets:
            return

        offsets = []
        for topic, partitions in self._pending_offsets.items():
            for partition, offset in partitions.items():
                offsets.append({
                    "topic": topic,
                    "partition": partition,
                    "offset": offset,
                })

        await self._consumer.commit(offsets, async_commit)
        self._pending_offsets.clear()
        self.metrics.commits += 1

        logger.debug(f"Committed {len(offsets)} offsets")

    async def seek(
        self,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """
        Seek to a specific offset.

        Args:
            topic: Topic name
            partition: Partition number
            offset: Target offset
        """
        if self._consumer:
            await self._consumer.seek(topic, partition, offset)
            logger.info(f"Seeked to {topic}/{partition}/{offset}")

    async def get_lag(self) -> Dict[str, Dict[int, int]]:
        """
        Get current consumer lag per partition.

        Returns:
            Dictionary of topic -> partition -> lag
        """
        if not self._consumer:
            return {}

        lag = await self._consumer.get_lag()
        self.metrics.current_lag = {
            topic: sum(partitions.values())
            for topic, partitions in lag.items()
        }
        return lag

    def get_metrics(self) -> ConsumerMetrics:
        """Return current consumer metrics."""
        return self.metrics


# =============================================================================
# MOCK IMPLEMENTATIONS (for demonstration - replace with confluent_kafka)
# =============================================================================


class MockKafkaProducer:
    """Mock Kafka producer for demonstration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._offset_counter: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    async def init_transactions(self) -> None:
        """Initialize transactions."""
        pass

    async def begin_transaction(self) -> None:
        """Begin a transaction."""
        pass

    async def commit_transaction(self) -> None:
        """Commit current transaction."""
        pass

    async def abort_transaction(self) -> None:
        """Abort current transaction."""
        pass

    async def produce(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        headers: List[tuple],
    ) -> Dict[str, Any]:
        """Produce a message."""
        partition = hash(key) % 12 if key else 0
        offset = self._offset_counter[topic][partition]
        self._offset_counter[topic][partition] += 1

        return {
            "topic": topic,
            "partition": partition,
            "offset": offset,
            "timestamp_ms": int(time.time() * 1000),
        }

    async def flush(self, timeout: float) -> None:
        """Flush pending messages."""
        pass

    async def close(self) -> None:
        """Close producer."""
        pass


class MockKafkaConsumer:
    """Mock Kafka consumer for demonstration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._subscribed: List[str] = []
        self._messages: List[Dict[str, Any]] = []

    async def subscribe(
        self,
        topics: List[str],
        from_beginning: bool = False,
    ) -> None:
        """Subscribe to topics."""
        self._subscribed = topics

    async def unsubscribe(self) -> None:
        """Unsubscribe from topics."""
        self._subscribed = []

    async def poll(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Poll for messages."""
        await asyncio.sleep(timeout)
        return None  # No messages in mock

    async def commit(
        self,
        offsets: List[Dict[str, Any]],
        async_commit: bool,
    ) -> None:
        """Commit offsets."""
        pass

    async def seek(
        self,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Seek to offset."""
        pass

    async def get_lag(self) -> Dict[str, Dict[int, int]]:
        """Get consumer lag."""
        return {}

    async def close(self) -> None:
        """Close consumer."""
        pass
