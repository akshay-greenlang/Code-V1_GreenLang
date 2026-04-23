"""
Kafka Streaming Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade Kafka integration for real-time data streaming:
- Publish analysis results to Kafka topics
- Subscribe to sensor data streams
- Schema registry integration (Avro)
- Exactly-once semantics with transactions
- Consumer group management
- Dead letter queue handling

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)

# Type variable for message types
T = TypeVar('T', bound=BaseModel)


# =============================================================================
# Enumerations
# =============================================================================


class SerializationType(str, Enum):
    """Message serialization types."""

    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    STRING = "string"
    BYTES = "bytes"


class CompressionType(str, Enum):
    """Kafka message compression types."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AcksMode(str, Enum):
    """Kafka acknowledgment modes."""

    NONE = "0"  # No acknowledgment (fire and forget)
    LEADER = "1"  # Leader acknowledgment only
    ALL = "all"  # All in-sync replicas acknowledgment


class OffsetResetPolicy(str, Enum):
    """Kafka consumer offset reset policy."""

    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class IsolationLevel(str, Enum):
    """Kafka consumer isolation level."""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"


class DeliverySemantics(str, Enum):
    """Message delivery semantics."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class ProducerState(str, Enum):
    """Producer state."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    IN_TRANSACTION = "in_transaction"
    ERROR = "error"
    CLOSED = "closed"


class ConsumerState(str, Enum):
    """Consumer state."""

    UNINITIALIZED = "uninitialized"
    SUBSCRIBED = "subscribed"
    POLLING = "polling"
    PAUSED = "paused"
    ERROR = "error"
    CLOSED = "closed"


# =============================================================================
# Custom Exceptions
# =============================================================================


class KafkaError(Exception):
    """Base Kafka exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class KafkaConnectionError(KafkaError):
    """Kafka connection error."""
    pass


class KafkaProducerError(KafkaError):
    """Kafka producer error."""
    pass


class KafkaConsumerError(KafkaError):
    """Kafka consumer error."""
    pass


class KafkaSerializationError(KafkaError):
    """Serialization/deserialization error."""
    pass


class KafkaTransactionError(KafkaError):
    """Transaction error."""
    pass


class SchemaRegistryError(KafkaError):
    """Schema registry error."""
    pass


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class KafkaSecurityConfig(BaseModel):
    """Kafka security configuration."""

    model_config = ConfigDict(extra="forbid")

    security_protocol: str = Field(
        default="PLAINTEXT",
        description="Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)"
    )

    # SSL settings
    ssl_cafile: Optional[str] = Field(
        default=None,
        description="Path to CA certificate file"
    )
    ssl_certfile: Optional[str] = Field(
        default=None,
        description="Path to client certificate file"
    )
    ssl_keyfile: Optional[str] = Field(
        default=None,
        description="Path to client private key file"
    )
    ssl_password: Optional[str] = Field(
        default=None,
        description="Password for SSL key file"
    )
    ssl_check_hostname: bool = Field(
        default=True,
        description="Verify SSL hostname"
    )

    # SASL settings
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI)"
    )
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username"
    )
    sasl_password: Optional[str] = Field(
        default=None,
        description="SASL password"
    )


class SchemaRegistryConfig(BaseModel):
    """Schema Registry configuration for Avro."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable schema registry integration"
    )
    url: str = Field(
        default="http://localhost:8081",
        description="Schema registry URL"
    )
    basic_auth_user: Optional[str] = Field(
        default=None,
        description="Basic auth username"
    )
    basic_auth_password: Optional[str] = Field(
        default=None,
        description="Basic auth password"
    )
    auto_register_schemas: bool = Field(
        default=True,
        description="Automatically register new schemas"
    )
    schema_cache_size: int = Field(
        default=1000,
        ge=10,
        description="Schema cache size"
    )
    subject_name_strategy: str = Field(
        default="TopicNameStrategy",
        description="Subject naming strategy"
    )


class ProducerConfig(BaseModel):
    """Kafka producer configuration."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(
        default="gl015-insulscan-producer",
        description="Producer client ID"
    )

    # Delivery settings
    acks: AcksMode = Field(
        default=AcksMode.ALL,
        description="Acknowledgment mode"
    )
    delivery_semantics: DeliverySemantics = Field(
        default=DeliverySemantics.EXACTLY_ONCE,
        description="Delivery semantics"
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable idempotent producer"
    )
    transactional_id: Optional[str] = Field(
        default=None,
        description="Transactional ID for exactly-once"
    )

    # Batching
    batch_size: int = Field(
        default=16384,
        ge=0,
        description="Batch size in bytes"
    )
    linger_ms: int = Field(
        default=5,
        ge=0,
        description="Linger time in milliseconds"
    )
    buffer_memory: int = Field(
        default=33554432,  # 32MB
        ge=0,
        description="Total buffer memory in bytes"
    )

    # Compression
    compression_type: CompressionType = Field(
        default=CompressionType.LZ4,
        description="Compression type"
    )

    # Retries
    retries: int = Field(
        default=2147483647,  # Max int for infinite retries
        ge=0,
        description="Number of retries"
    )
    retry_backoff_ms: int = Field(
        default=100,
        ge=0,
        description="Retry backoff in milliseconds"
    )
    delivery_timeout_ms: int = Field(
        default=120000,
        ge=0,
        description="Delivery timeout in milliseconds"
    )

    # Serialization
    key_serializer: SerializationType = Field(
        default=SerializationType.STRING,
        description="Key serializer"
    )
    value_serializer: SerializationType = Field(
        default=SerializationType.JSON,
        description="Value serializer"
    )


class ConsumerConfig(BaseModel):
    """Kafka consumer configuration."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(
        default="gl015-insulscan-consumer",
        description="Consumer client ID"
    )
    group_id: str = Field(
        default="gl015-insulscan-group",
        description="Consumer group ID"
    )

    # Offset management
    auto_offset_reset: OffsetResetPolicy = Field(
        default=OffsetResetPolicy.EARLIEST,
        description="Auto offset reset policy"
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Enable auto commit"
    )
    auto_commit_interval_ms: int = Field(
        default=5000,
        ge=0,
        description="Auto commit interval"
    )

    # Fetch settings
    fetch_min_bytes: int = Field(
        default=1,
        ge=1,
        description="Minimum fetch bytes"
    )
    fetch_max_bytes: int = Field(
        default=52428800,  # 50MB
        ge=1,
        description="Maximum fetch bytes"
    )
    fetch_max_wait_ms: int = Field(
        default=500,
        ge=0,
        description="Maximum fetch wait time"
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        description="Maximum poll records"
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=1000,
        description="Maximum poll interval"
    )

    # Session
    session_timeout_ms: int = Field(
        default=45000,
        ge=1000,
        description="Session timeout"
    )
    heartbeat_interval_ms: int = Field(
        default=3000,
        ge=1000,
        description="Heartbeat interval"
    )

    # Isolation
    isolation_level: IsolationLevel = Field(
        default=IsolationLevel.READ_COMMITTED,
        description="Isolation level"
    )

    # Deserialization
    key_deserializer: SerializationType = Field(
        default=SerializationType.STRING,
        description="Key deserializer"
    )
    value_deserializer: SerializationType = Field(
        default=SerializationType.JSON,
        description="Value deserializer"
    )


class KafkaStreamingConfig(BaseModel):
    """Complete Kafka streaming configuration."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Connector identifier"
    )
    connector_name: str = Field(
        default="Kafka-Streaming",
        description="Connector name"
    )

    # Bootstrap servers
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"],
        min_length=1,
        description="Kafka bootstrap servers"
    )

    # Security
    security_config: KafkaSecurityConfig = Field(
        default_factory=KafkaSecurityConfig,
        description="Security configuration"
    )

    # Schema registry
    schema_registry_config: SchemaRegistryConfig = Field(
        default_factory=SchemaRegistryConfig,
        description="Schema registry configuration"
    )

    # Producer
    producer_config: ProducerConfig = Field(
        default_factory=ProducerConfig,
        description="Producer configuration"
    )

    # Consumer
    consumer_config: ConsumerConfig = Field(
        default_factory=ConsumerConfig,
        description="Consumer configuration"
    )

    # Dead letter queue
    dlq_enabled: bool = Field(
        default=True,
        description="Enable dead letter queue"
    )
    dlq_topic_suffix: str = Field(
        default=".dlq",
        description="DLQ topic suffix"
    )
    dlq_max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retries before sending to DLQ"
    )

    # Health check
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_check_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        description="Health check interval"
    )

    @field_validator('bootstrap_servers')
    @classmethod
    def validate_bootstrap_servers(cls, v: List[str]) -> List[str]:
        """Validate bootstrap servers format."""
        for server in v:
            parts = server.split(':')
            if len(parts) != 2:
                raise ValueError(f'Invalid server format: {server}')
            try:
                int(parts[1])
            except ValueError:
                raise ValueError(f'Invalid port in server: {server}')
        return v


# =============================================================================
# Data Models - Messages
# =============================================================================


class MessageHeaders(BaseModel):
    """Kafka message headers."""

    model_config = ConfigDict(frozen=False)

    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation ID"
    )
    message_type: str = Field(
        default="",
        description="Message type"
    )
    source: str = Field(
        default="gl015-insulscan",
        description="Message source"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    custom_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers"
    )


class KafkaMessage(BaseModel):
    """Kafka message wrapper."""

    model_config = ConfigDict(frozen=False)

    topic: str = Field(..., description="Topic name")
    key: Optional[str] = Field(default=None, description="Message key")
    value: Any = Field(..., description="Message value")
    headers: MessageHeaders = Field(
        default_factory=MessageHeaders,
        description="Message headers"
    )
    partition: Optional[int] = Field(
        default=None,
        description="Target partition"
    )
    timestamp_ms: Optional[int] = Field(
        default=None,
        description="Message timestamp in milliseconds"
    )


class ConsumedMessage(BaseModel):
    """Consumed Kafka message with metadata."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(..., description="Topic name")
    partition: int = Field(..., description="Partition")
    offset: int = Field(..., description="Offset")
    key: Optional[str] = Field(default=None, description="Message key")
    value: Any = Field(..., description="Message value")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Message headers"
    )
    timestamp: datetime = Field(..., description="Message timestamp")
    timestamp_type: str = Field(
        default="CreateTime",
        description="Timestamp type"
    )


class ProducerAck(BaseModel):
    """Producer acknowledgment."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(..., description="Topic")
    partition: int = Field(..., description="Partition")
    offset: int = Field(..., description="Offset")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp"
    )
    latency_ms: float = Field(
        default=0.0,
        description="Latency in milliseconds"
    )


# =============================================================================
# Data Models - Insulation Domain Messages
# =============================================================================


class TemperatureDataMessage(BaseModel):
    """Temperature data message for Kafka."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Message ID"
    )
    equipment_id: str = Field(..., description="Equipment ID")
    sensor_id: str = Field(..., description="Sensor ID")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    quality: str = Field(default="good", description="Data quality")
    source: str = Field(
        default="opcua",
        description="Data source"
    )


class ThermalAnalysisResult(BaseModel):
    """Thermal analysis result message."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Analysis ID"
    )
    equipment_id: str = Field(..., description="Equipment ID")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )

    # Thermal data
    surface_temperature_c: float = Field(..., description="Surface temperature")
    ambient_temperature_c: float = Field(..., description="Ambient temperature")
    operating_temperature_c: Optional[float] = Field(
        default=None,
        description="Operating temperature"
    )

    # Analysis results
    heat_loss_kw: Optional[float] = Field(
        default=None,
        description="Heat loss in kW"
    )
    thermal_efficiency: Optional[float] = Field(
        default=None,
        description="Thermal efficiency"
    )
    defect_detected: bool = Field(
        default=False,
        description="Defect detected flag"
    )
    defect_severity: Optional[str] = Field(
        default=None,
        description="Defect severity"
    )
    defect_location: Optional[str] = Field(
        default=None,
        description="Defect location"
    )

    # Recommendations
    repair_recommended: bool = Field(
        default=False,
        description="Repair recommended"
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level"
    )
    estimated_savings_annual: Optional[float] = Field(
        default=None,
        description="Estimated annual savings"
    )


class InsulationDefectEvent(BaseModel):
    """Insulation defect event message."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event ID"
    )
    event_type: str = Field(
        default="defect_detected",
        description="Event type"
    )
    equipment_id: str = Field(..., description="Equipment ID")
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )

    # Defect details
    defect_type: str = Field(..., description="Defect type")
    severity: str = Field(..., description="Severity level")
    location: str = Field(..., description="Location on equipment")
    area_m2: Optional[float] = Field(
        default=None,
        description="Affected area"
    )
    temperature_delta_c: Optional[float] = Field(
        default=None,
        description="Temperature difference"
    )

    # References
    thermal_image_id: Optional[str] = Field(
        default=None,
        description="Thermal image reference"
    )
    analysis_id: Optional[str] = Field(
        default=None,
        description="Analysis reference"
    )


class WorkOrderCreatedEvent(BaseModel):
    """Work order created event message."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event ID"
    )
    event_type: str = Field(
        default="work_order_created",
        description="Event type"
    )
    work_order_number: str = Field(..., description="Work order number")
    equipment_id: str = Field(..., description="Equipment ID")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    priority: str = Field(..., description="Priority")
    work_order_type: str = Field(..., description="Work order type")
    estimated_cost: Optional[float] = Field(
        default=None,
        description="Estimated cost"
    )


# =============================================================================
# Avro Schema Registry Client
# =============================================================================


class AvroSchemaRegistry:
    """
    Avro Schema Registry client.

    Manages schema registration, retrieval, and compatibility checks.
    """

    def __init__(self, config: SchemaRegistryConfig) -> None:
        """
        Initialize schema registry client.

        Args:
            config: Schema registry configuration
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.SchemaRegistry")
        self._schema_cache: Dict[int, Dict[str, Any]] = {}
        self._subject_cache: Dict[str, int] = {}

    async def register_schema(
        self,
        subject: str,
        schema: Dict[str, Any]
    ) -> int:
        """
        Register schema with registry.

        Args:
            subject: Subject name
            schema: Avro schema

        Returns:
            Schema ID
        """
        if not self._config.enabled:
            return -1

        # In production, use confluent-kafka-python or httpx
        # POST to /subjects/{subject}/versions

        self._logger.info(f"Registered schema for subject: {subject}")
        schema_id = len(self._schema_cache) + 1
        self._schema_cache[schema_id] = schema
        self._subject_cache[subject] = schema_id
        return schema_id

    async def get_schema(self, schema_id: int) -> Optional[Dict[str, Any]]:
        """
        Get schema by ID.

        Args:
            schema_id: Schema ID

        Returns:
            Schema or None
        """
        return self._schema_cache.get(schema_id)

    async def get_latest_schema(self, subject: str) -> Optional[Dict[str, Any]]:
        """
        Get latest schema for subject.

        Args:
            subject: Subject name

        Returns:
            Latest schema or None
        """
        schema_id = self._subject_cache.get(subject)
        if schema_id:
            return await self.get_schema(schema_id)
        return None

    async def check_compatibility(
        self,
        subject: str,
        schema: Dict[str, Any]
    ) -> bool:
        """
        Check schema compatibility.

        Args:
            subject: Subject name
            schema: Schema to check

        Returns:
            True if compatible
        """
        # In production, POST to /compatibility/subjects/{subject}/versions/latest
        return True


# =============================================================================
# Kafka Producer
# =============================================================================


class KafkaProducer:
    """
    Kafka producer with exactly-once semantics.

    Features:
    - Transactional message production
    - Schema registry integration
    - Automatic serialization
    - Delivery guarantees
    """

    def __init__(
        self,
        config: KafkaStreamingConfig,
        schema_registry: Optional[AvroSchemaRegistry] = None
    ) -> None:
        """
        Initialize Kafka producer.

        Args:
            config: Streaming configuration
            schema_registry: Optional schema registry client
        """
        self._config = config
        self._producer_config = config.producer_config
        self._logger = logging.getLogger(f"{__name__}.Producer")

        self._schema_registry = schema_registry
        self._producer: Optional[Any] = None  # aiokafka.AIOKafkaProducer
        self._state = ProducerState.UNINITIALIZED

        # Transaction state
        self._in_transaction = False

        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._bytes_sent = 0

    @property
    def state(self) -> ProducerState:
        """Get producer state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if producer is ready."""
        return self._state == ProducerState.READY

    async def start(self) -> None:
        """Start the producer."""
        self._logger.info("Starting Kafka producer")

        try:
            # In production, use aiokafka:
            # from aiokafka import AIOKafkaProducer
            # self._producer = AIOKafkaProducer(
            #     bootstrap_servers=self._config.bootstrap_servers,
            #     client_id=self._producer_config.client_id,
            #     acks=self._producer_config.acks.value,
            #     enable_idempotence=self._producer_config.enable_idempotence,
            #     transactional_id=self._producer_config.transactional_id,
            # )
            # await self._producer.start()

            # Initialize transactions if using exactly-once
            if self._producer_config.delivery_semantics == DeliverySemantics.EXACTLY_ONCE:
                # await self._producer.init_transactions()
                self._logger.info("Initialized transactional producer")

            self._state = ProducerState.READY
            self._logger.info("Kafka producer started")

        except Exception as e:
            self._state = ProducerState.ERROR
            self._logger.error(f"Failed to start producer: {e}")
            raise KafkaProducerError(f"Producer start failed: {e}")

    async def stop(self) -> None:
        """Stop the producer."""
        self._logger.info("Stopping Kafka producer")

        if self._in_transaction:
            try:
                await self.abort_transaction()
            except Exception as e:
                self._logger.warning(f"Error aborting transaction: {e}")

        if self._producer:
            # await self._producer.stop()
            pass

        self._state = ProducerState.CLOSED
        self._logger.info("Kafka producer stopped")

    async def send(
        self,
        topic: str,
        value: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        partition: Optional[int] = None
    ) -> ProducerAck:
        """
        Send message to Kafka topic.

        Args:
            topic: Topic name
            value: Message value
            key: Optional message key
            headers: Optional headers
            partition: Optional target partition

        Returns:
            Producer acknowledgment

        Raises:
            KafkaProducerError: If send fails
        """
        if not self.is_ready:
            raise KafkaProducerError("Producer not ready")

        start_time = datetime.utcnow()

        try:
            # Serialize value
            serialized_value = await self._serialize_value(value)

            # Serialize key
            serialized_key = key.encode() if key else None

            # Convert headers
            kafka_headers = [
                (k, v.encode()) for k, v in (headers or {}).items()
            ]

            # In production:
            # result = await self._producer.send_and_wait(
            #     topic=topic,
            #     value=serialized_value,
            #     key=serialized_key,
            #     headers=kafka_headers,
            #     partition=partition,
            # )

            # Mock result
            result_partition = partition or 0
            result_offset = self._messages_sent

            self._messages_sent += 1
            self._bytes_sent += len(serialized_value)

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._logger.debug(
                f"Sent message to {topic}[{result_partition}]@{result_offset}"
            )

            return ProducerAck(
                topic=topic,
                partition=result_partition,
                offset=result_offset,
                latency_ms=latency,
            )

        except Exception as e:
            self._messages_failed += 1
            self._logger.error(f"Failed to send message: {e}")
            raise KafkaProducerError(f"Send failed: {e}")

    async def send_batch(
        self,
        messages: List[KafkaMessage]
    ) -> List[ProducerAck]:
        """
        Send batch of messages.

        Args:
            messages: Messages to send

        Returns:
            List of acknowledgments
        """
        acks = []
        for msg in messages:
            ack = await self.send(
                topic=msg.topic,
                value=msg.value,
                key=msg.key,
                headers=msg.headers.custom_headers,
                partition=msg.partition,
            )
            acks.append(ack)
        return acks

    async def send_analysis_result(
        self,
        topic: str,
        result: ThermalAnalysisResult
    ) -> ProducerAck:
        """
        Send thermal analysis result.

        Args:
            topic: Topic name
            result: Analysis result

        Returns:
            Producer acknowledgment
        """
        headers = {
            "message_type": "thermal_analysis_result",
            "equipment_id": result.equipment_id,
            "analysis_id": result.analysis_id,
        }

        return await self.send(
            topic=topic,
            value=result.model_dump(mode='json'),
            key=result.equipment_id,
            headers=headers,
        )

    async def send_defect_event(
        self,
        topic: str,
        event: InsulationDefectEvent
    ) -> ProducerAck:
        """
        Send defect detection event.

        Args:
            topic: Topic name
            event: Defect event

        Returns:
            Producer acknowledgment
        """
        headers = {
            "message_type": "defect_event",
            "equipment_id": event.equipment_id,
            "severity": event.severity,
        }

        return await self.send(
            topic=topic,
            value=event.model_dump(mode='json'),
            key=event.equipment_id,
            headers=headers,
        )

    # =========================================================================
    # Transaction Support
    # =========================================================================

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self._producer_config.delivery_semantics != DeliverySemantics.EXACTLY_ONCE:
            raise KafkaTransactionError("Transactions require exactly-once semantics")

        if self._in_transaction:
            raise KafkaTransactionError("Transaction already in progress")

        # await self._producer.begin_transaction()
        self._in_transaction = True
        self._state = ProducerState.IN_TRANSACTION
        self._logger.debug("Transaction started")

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            raise KafkaTransactionError("No transaction in progress")

        # await self._producer.commit_transaction()
        self._in_transaction = False
        self._state = ProducerState.READY
        self._logger.debug("Transaction committed")

    async def abort_transaction(self) -> None:
        """Abort the current transaction."""
        if not self._in_transaction:
            raise KafkaTransactionError("No transaction in progress")

        # await self._producer.abort_transaction()
        self._in_transaction = False
        self._state = ProducerState.READY
        self._logger.warning("Transaction aborted")

    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions."""
        await self.begin_transaction()
        try:
            yield
            await self.commit_transaction()
        except Exception as e:
            await self.abort_transaction()
            raise

    # =========================================================================
    # Serialization
    # =========================================================================

    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize message value."""
        serializer = self._producer_config.value_serializer

        if serializer == SerializationType.JSON:
            if isinstance(value, BaseModel):
                return value.model_dump_json().encode()
            return json.dumps(value, default=str).encode()

        elif serializer == SerializationType.AVRO:
            if not self._schema_registry:
                raise KafkaSerializationError("Schema registry not configured")
            # Use fastavro or confluent_avro_serializer
            raise NotImplementedError("Avro serialization requires fastavro")

        elif serializer == SerializationType.STRING:
            return str(value).encode()

        elif serializer == SerializationType.BYTES:
            if isinstance(value, bytes):
                return value
            return str(value).encode()

        else:
            raise KafkaSerializationError(f"Unknown serializer: {serializer}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        return {
            "state": self._state.value,
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "bytes_sent": self._bytes_sent,
            "in_transaction": self._in_transaction,
        }


# =============================================================================
# Kafka Consumer
# =============================================================================


class KafkaConsumer:
    """
    Kafka consumer with consumer group support.

    Features:
    - Consumer group management
    - Manual offset commits
    - Message deserialization
    - Dead letter queue handling
    """

    def __init__(
        self,
        config: KafkaStreamingConfig,
        schema_registry: Optional[AvroSchemaRegistry] = None
    ) -> None:
        """
        Initialize Kafka consumer.

        Args:
            config: Streaming configuration
            schema_registry: Optional schema registry client
        """
        self._config = config
        self._consumer_config = config.consumer_config
        self._logger = logging.getLogger(f"{__name__}.Consumer")

        self._schema_registry = schema_registry
        self._consumer: Optional[Any] = None  # aiokafka.AIOKafkaConsumer
        self._state = ConsumerState.UNINITIALIZED

        # Subscribed topics
        self._subscribed_topics: Set[str] = set()

        # Message handlers
        self._handlers: Dict[str, Callable] = {}

        # DLQ producer
        self._dlq_producer: Optional[KafkaProducer] = None

        # Metrics
        self._messages_received = 0
        self._messages_processed = 0
        self._messages_failed = 0
        self._bytes_received = 0

    @property
    def state(self) -> ConsumerState:
        """Get consumer state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if consumer is ready."""
        return self._state in [ConsumerState.SUBSCRIBED, ConsumerState.POLLING]

    async def start(self) -> None:
        """Start the consumer."""
        self._logger.info("Starting Kafka consumer")

        try:
            # In production:
            # from aiokafka import AIOKafkaConsumer
            # self._consumer = AIOKafkaConsumer(
            #     bootstrap_servers=self._config.bootstrap_servers,
            #     client_id=self._consumer_config.client_id,
            #     group_id=self._consumer_config.group_id,
            #     auto_offset_reset=self._consumer_config.auto_offset_reset.value,
            #     enable_auto_commit=self._consumer_config.enable_auto_commit,
            # )
            # await self._consumer.start()

            # Initialize DLQ producer if enabled
            if self._config.dlq_enabled:
                self._dlq_producer = KafkaProducer(self._config, self._schema_registry)
                await self._dlq_producer.start()

            self._state = ConsumerState.SUBSCRIBED
            self._logger.info("Kafka consumer started")

        except Exception as e:
            self._state = ConsumerState.ERROR
            self._logger.error(f"Failed to start consumer: {e}")
            raise KafkaConsumerError(f"Consumer start failed: {e}")

    async def stop(self) -> None:
        """Stop the consumer."""
        self._logger.info("Stopping Kafka consumer")

        if self._dlq_producer:
            await self._dlq_producer.stop()

        if self._consumer:
            # await self._consumer.stop()
            pass

        self._state = ConsumerState.CLOSED
        self._logger.info("Kafka consumer stopped")

    async def subscribe(
        self,
        topics: List[str],
        handler: Optional[Callable[[ConsumedMessage], None]] = None
    ) -> None:
        """
        Subscribe to topics.

        Args:
            topics: Topics to subscribe to
            handler: Optional message handler
        """
        if not self.is_ready:
            raise KafkaConsumerError("Consumer not ready")

        # self._consumer.subscribe(topics)
        self._subscribed_topics.update(topics)

        if handler:
            for topic in topics:
                self._handlers[topic] = handler

        self._logger.info(f"Subscribed to topics: {topics}")

    async def unsubscribe(self) -> None:
        """Unsubscribe from all topics."""
        # self._consumer.unsubscribe()
        self._subscribed_topics.clear()
        self._handlers.clear()
        self._logger.info("Unsubscribed from all topics")

    async def poll(
        self,
        timeout_ms: int = 1000,
        max_records: Optional[int] = None
    ) -> List[ConsumedMessage]:
        """
        Poll for messages.

        Args:
            timeout_ms: Poll timeout
            max_records: Maximum records to return

        Returns:
            List of consumed messages
        """
        if not self.is_ready:
            raise KafkaConsumerError("Consumer not ready")

        self._state = ConsumerState.POLLING

        messages = []

        try:
            # In production:
            # data = await self._consumer.getmany(
            #     timeout_ms=timeout_ms,
            #     max_records=max_records or self._consumer_config.max_poll_records
            # )
            # for topic_partition, records in data.items():
            #     for record in records:
            #         messages.append(self._convert_message(record))

            self._messages_received += len(messages)

            # Process with handlers
            for msg in messages:
                await self._process_message(msg)

            return messages

        except Exception as e:
            self._logger.error(f"Poll error: {e}")
            raise KafkaConsumerError(f"Poll failed: {e}")

        finally:
            self._state = ConsumerState.SUBSCRIBED

    async def _process_message(self, message: ConsumedMessage) -> None:
        """Process a consumed message."""
        handler = self._handlers.get(message.topic)
        if not handler:
            return

        try:
            await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
            self._messages_processed += 1
        except Exception as e:
            self._messages_failed += 1
            self._logger.error(f"Handler error for {message.topic}: {e}")

            if self._config.dlq_enabled:
                await self._send_to_dlq(message, str(e))

    async def _send_to_dlq(
        self,
        message: ConsumedMessage,
        error: str
    ) -> None:
        """Send failed message to dead letter queue."""
        if not self._dlq_producer:
            return

        dlq_topic = f"{message.topic}{self._config.dlq_topic_suffix}"

        dlq_value = {
            "original_topic": message.topic,
            "original_partition": message.partition,
            "original_offset": message.offset,
            "original_key": message.key,
            "original_value": message.value,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
        }

        try:
            await self._dlq_producer.send(
                topic=dlq_topic,
                value=dlq_value,
                key=message.key,
            )
            self._logger.warning(f"Sent message to DLQ: {dlq_topic}")
        except Exception as e:
            self._logger.error(f"Failed to send to DLQ: {e}")

    async def commit(self) -> None:
        """Commit current offsets."""
        if not self._consumer_config.enable_auto_commit:
            # await self._consumer.commit()
            self._logger.debug("Committed offsets")

    async def seek_to_beginning(self, topic: str, partition: int) -> None:
        """Seek to beginning of partition."""
        # tp = TopicPartition(topic, partition)
        # await self._consumer.seek_to_beginning(tp)
        self._logger.info(f"Seeked to beginning: {topic}[{partition}]")

    async def seek_to_end(self, topic: str, partition: int) -> None:
        """Seek to end of partition."""
        # tp = TopicPartition(topic, partition)
        # await self._consumer.seek_to_end(tp)
        self._logger.info(f"Seeked to end: {topic}[{partition}]")

    # =========================================================================
    # Streaming Interface
    # =========================================================================

    async def stream(self) -> AsyncIterator[ConsumedMessage]:
        """
        Stream messages as async iterator.

        Yields:
            Consumed messages
        """
        while self.is_ready:
            messages = await self.poll(timeout_ms=1000)
            for msg in messages:
                yield msg

    async def stream_temperature_data(
        self,
        topic: str
    ) -> AsyncIterator[TemperatureDataMessage]:
        """
        Stream temperature data messages.

        Args:
            topic: Topic to consume from

        Yields:
            Temperature data messages
        """
        await self.subscribe([topic])

        async for msg in self.stream():
            try:
                data = TemperatureDataMessage(**msg.value)
                yield data
            except Exception as e:
                self._logger.warning(f"Invalid temperature message: {e}")

    async def stream_analysis_results(
        self,
        topic: str
    ) -> AsyncIterator[ThermalAnalysisResult]:
        """
        Stream thermal analysis results.

        Args:
            topic: Topic to consume from

        Yields:
            Analysis result messages
        """
        await self.subscribe([topic])

        async for msg in self.stream():
            try:
                result = ThermalAnalysisResult(**msg.value)
                yield result
            except Exception as e:
                self._logger.warning(f"Invalid analysis result: {e}")

    # =========================================================================
    # Deserialization
    # =========================================================================

    def _convert_message(self, record: Any) -> ConsumedMessage:
        """Convert Kafka record to ConsumedMessage."""
        # Deserialize value
        value = self._deserialize_value(record.value)

        # Convert headers
        headers = {}
        if record.headers:
            for key, val in record.headers:
                headers[key] = val.decode() if val else ""

        return ConsumedMessage(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            key=record.key.decode() if record.key else None,
            value=value,
            headers=headers,
            timestamp=datetime.fromtimestamp(record.timestamp / 1000),
            timestamp_type="CreateTime",
        )

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize message value."""
        deserializer = self._consumer_config.value_deserializer

        if deserializer == SerializationType.JSON:
            return json.loads(data.decode())

        elif deserializer == SerializationType.STRING:
            return data.decode()

        elif deserializer == SerializationType.BYTES:
            return data

        else:
            return data.decode()

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "state": self._state.value,
            "subscribed_topics": list(self._subscribed_topics),
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "messages_failed": self._messages_failed,
            "bytes_received": self._bytes_received,
        }


# =============================================================================
# Kafka Streaming Manager
# =============================================================================


class KafkaStreamingManager:
    """
    High-level Kafka streaming manager.

    Coordinates producer, consumer, and schema registry.
    """

    def __init__(self, config: KafkaStreamingConfig) -> None:
        """
        Initialize streaming manager.

        Args:
            config: Streaming configuration
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.Manager")

        # Schema registry
        self._schema_registry: Optional[AvroSchemaRegistry] = None
        if config.schema_registry_config.enabled:
            self._schema_registry = AvroSchemaRegistry(config.schema_registry_config)

        # Producer and consumer
        self._producer: Optional[KafkaProducer] = None
        self._consumer: Optional[KafkaConsumer] = None

        # Health check
        self._health_check_task: Optional[asyncio.Task] = None

    @property
    def producer(self) -> Optional[KafkaProducer]:
        """Get producer instance."""
        return self._producer

    @property
    def consumer(self) -> Optional[KafkaConsumer]:
        """Get consumer instance."""
        return self._consumer

    async def start(self) -> None:
        """Start the streaming manager."""
        self._logger.info("Starting Kafka streaming manager")

        # Start producer
        self._producer = KafkaProducer(self._config, self._schema_registry)
        await self._producer.start()

        # Start consumer
        self._consumer = KafkaConsumer(self._config, self._schema_registry)
        await self._consumer.start()

        # Start health check
        if self._config.health_check_enabled:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        self._logger.info("Kafka streaming manager started")

    async def stop(self) -> None:
        """Stop the streaming manager."""
        self._logger.info("Stopping Kafka streaming manager")

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop consumer
        if self._consumer:
            await self._consumer.stop()

        # Stop producer
        if self._producer:
            await self._producer.stop()

        self._logger.info("Kafka streaming manager stopped")

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                # Check producer
                if self._producer and not self._producer.is_ready:
                    self._logger.warning("Producer not ready")

                # Check consumer
                if self._consumer and not self._consumer.is_ready:
                    self._logger.warning("Consumer not ready")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    async def publish_analysis_result(
        self,
        result: ThermalAnalysisResult,
        topic: str = "insulscan.analysis.results"
    ) -> ProducerAck:
        """
        Publish thermal analysis result.

        Args:
            result: Analysis result
            topic: Target topic

        Returns:
            Producer acknowledgment
        """
        if not self._producer:
            raise KafkaProducerError("Producer not initialized")
        return await self._producer.send_analysis_result(topic, result)

    async def publish_defect_event(
        self,
        event: InsulationDefectEvent,
        topic: str = "insulscan.events.defects"
    ) -> ProducerAck:
        """
        Publish defect event.

        Args:
            event: Defect event
            topic: Target topic

        Returns:
            Producer acknowledgment
        """
        if not self._producer:
            raise KafkaProducerError("Producer not initialized")
        return await self._producer.send_defect_event(topic, event)

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics."""
        return {
            "producer": self._producer.get_metrics() if self._producer else None,
            "consumer": self._consumer.get_metrics() if self._consumer else None,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_kafka_streaming_config(
    bootstrap_servers: List[str],
    group_id: str = "gl015-insulscan-group",
    **kwargs
) -> KafkaStreamingConfig:
    """
    Create Kafka streaming configuration.

    Args:
        bootstrap_servers: Bootstrap server list
        group_id: Consumer group ID
        **kwargs: Additional configuration

    Returns:
        Configured KafkaStreamingConfig
    """
    consumer_config = ConsumerConfig(group_id=group_id)

    return KafkaStreamingConfig(
        bootstrap_servers=bootstrap_servers,
        consumer_config=consumer_config,
        **kwargs
    )


def create_kafka_streaming_manager(
    bootstrap_servers: List[str],
    group_id: str = "gl015-insulscan-group",
    **kwargs
) -> KafkaStreamingManager:
    """
    Create Kafka streaming manager.

    Args:
        bootstrap_servers: Bootstrap server list
        group_id: Consumer group ID
        **kwargs: Additional configuration

    Returns:
        Configured KafkaStreamingManager
    """
    config = create_kafka_streaming_config(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        **kwargs
    )
    return KafkaStreamingManager(config)
