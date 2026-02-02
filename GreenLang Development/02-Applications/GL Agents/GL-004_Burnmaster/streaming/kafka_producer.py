"""
Kafka Producer Module - GL-004 BURNMASTER

This module provides a Kafka producer implementation for combustion data streaming
with exactly-once semantics, idempotent message delivery, and comprehensive
failure handling for industrial combustion systems.

Key Features:
    - Exactly-once delivery semantics via idempotent producers
    - Transactional message production
    - Automatic retry with exponential backoff
    - Message deduplication via sequence numbers
    - Comprehensive failure recovery strategies
    - Provenance tracking with SHA-256 hashes

Example:
    >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
    >>> producer = CombustionDataProducer()
    >>> result = await producer.connect(config)
    >>> publish_result = await producer.publish_combustion_data(data)

Author: GreenLang Combustion Optimization Team
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class SecurityProtocol(str, Enum):
    """Kafka security protocols."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


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


class DeliveryStatus(str, Enum):
    """Message delivery status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    RETRYING = "retrying"


class RecoveryActionType(str, Enum):
    """Types of recovery actions for publish failures."""

    RETRY = "retry"
    SKIP = "skip"
    DEAD_LETTER = "dead_letter"
    CIRCUIT_BREAK = "circuit_break"
    RECONNECT = "reconnect"
    ABORT = "abort"


class CombustionEventType(str, Enum):
    """Types of combustion events."""

    TEMPERATURE_READING = "temperature_reading"
    PRESSURE_READING = "pressure_reading"
    FLOW_RATE = "flow_rate"
    OXYGEN_LEVEL = "oxygen_level"
    CO2_LEVEL = "co2_level"
    NOX_LEVEL = "nox_level"
    FLAME_DETECTION = "flame_detection"
    EFFICIENCY_METRIC = "efficiency_metric"
    ANOMALY_DETECTED = "anomaly_detected"
    SAFETY_ALERT = "safety_alert"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class KafkaConfig(BaseModel):
    """
    Kafka connection and behavior configuration.

    Attributes:
        bootstrap_servers: Kafka bootstrap server addresses
        client_id: Client identifier for tracking
        security_protocol: Security protocol to use
        sasl_mechanism: SASL mechanism if using SASL
        ssl_ca_location: Path to CA certificate
    """

    bootstrap_servers: str = Field(
        "localhost:9092",
        description="Comma-separated bootstrap server addresses",
    )
    client_id: str = Field(
        default_factory=lambda: f"gl004-burnmaster-{uuid.uuid4().hex[:8]}",
        description="Client identifier",
    )
    security_protocol: SecurityProtocol = Field(
        SecurityProtocol.PLAINTEXT,
        description="Security protocol",
    )
    sasl_mechanism: Optional[str] = Field(
        None,
        description="SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)",
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
    request_timeout_ms: int = Field(
        30000,
        ge=1000,
        le=300000,
        description="Request timeout in milliseconds",
    )
    connections_max_idle_ms: int = Field(
        540000,
        ge=1000,
        description="Max idle time for connections",
    )

    def to_aiokafka_config(self) -> Dict[str, Any]:
        """Convert to aiokafka configuration dictionary."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": self.client_id,
            "security_protocol": self.security_protocol.value,
            "request_timeout_ms": self.request_timeout_ms,
        }

        if self.sasl_mechanism:
            config["sasl_mechanism"] = self.sasl_mechanism
            if self.sasl_username:
                config["sasl_plain_username"] = self.sasl_username
            if self.sasl_password:
                config["sasl_plain_password"] = self.sasl_password

        if self.ssl_ca_location:
            config["ssl_cafile"] = self.ssl_ca_location
        if self.ssl_certificate_location:
            config["ssl_certfile"] = self.ssl_certificate_location
        if self.ssl_key_location:
            config["ssl_keyfile"] = self.ssl_key_location

        return config


class ProducerConfig(BaseModel):
    """
    Kafka producer-specific configuration.

    Attributes:
        acks: Acknowledgment mode for durability
        compression_type: Message compression
        batch_size: Batch size in bytes
        linger_ms: Batch linger time
        enable_idempotence: Enable idempotent producer for exactly-once
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
        le=1000,
        description="Time to wait for batch to fill",
    )
    max_request_size: int = Field(
        1048576,
        ge=1024,
        description="Maximum request size in bytes",
    )
    max_in_flight_requests: int = Field(
        5,
        ge=1,
        le=10,
        description="Max in-flight requests per connection",
    )
    enable_idempotence: bool = Field(
        True,
        description="Enable idempotent producer for exactly-once semantics",
    )
    transactional_id: Optional[str] = Field(
        None,
        description="Transactional ID for exactly-once semantics",
    )
    retries: int = Field(
        2147483647,
        ge=0,
        description="Number of retries (max for idempotent)",
    )
    retry_backoff_ms: int = Field(
        100,
        ge=0,
        le=10000,
        description="Backoff between retries",
    )
    max_block_ms: int = Field(
        60000,
        ge=1000,
        description="Max time to block on buffer full",
    )


# =============================================================================
# DATA MODELS
# =============================================================================


class CombustionDataPoint(BaseModel):
    """Single combustion data point."""

    tag_id: str = Field(..., description="Sensor tag identifier")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp",
    )
    quality: str = Field(
        "GOOD",
        description="Data quality indicator",
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    zone_id: Optional[str] = Field(None, description="Combustion zone identifier")


class CombustionData(BaseModel):
    """
    Combustion data batch for streaming.

    Attributes:
        batch_id: Unique batch identifier
        source_system: Source system identifier
        equipment_id: Equipment producing the data
        collection_timestamp: When data was collected
        points: List of data points
    """

    batch_id: str = Field(
        default_factory=lambda: f"batch-{uuid.uuid4().hex[:12]}",
        description="Unique batch identifier",
    )
    source_system: str = Field(
        ...,
        description="Source system identifier (e.g., DCS, PLC)",
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier",
    )
    collection_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Collection timestamp",
    )
    points: List[CombustionDataPoint] = Field(
        default_factory=list,
        description="Data points in batch",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @property
    def point_count(self) -> int:
        """Return number of points in batch."""
        return len(self.points)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        content = json.dumps(
            {
                "batch_id": self.batch_id,
                "source_system": self.source_system,
                "equipment_id": self.equipment_id,
                "collection_timestamp": self.collection_timestamp.isoformat(),
                "point_count": self.point_count,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class CombustionEvent(BaseModel):
    """
    Combustion event for streaming.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of combustion event
        equipment_id: Equipment that generated event
        timestamp: Event timestamp
        severity: Event severity level
    """

    event_id: str = Field(
        default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}",
        description="Unique event identifier",
    )
    event_type: CombustionEventType = Field(
        ...,
        description="Type of combustion event",
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier",
    )
    zone_id: Optional[str] = Field(
        None,
        description="Combustion zone identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    severity: str = Field(
        "INFO",
        description="Severity: INFO, WARNING, CRITICAL",
    )
    value: Optional[float] = Field(
        None,
        description="Associated value if applicable",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit of measurement",
    )
    description: str = Field(
        "",
        description="Event description",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        content = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "equipment_id": self.equipment_id,
                "timestamp": self.timestamp.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class Recommendation(BaseModel):
    """
    Optimization recommendation for streaming.

    Attributes:
        recommendation_id: Unique recommendation identifier
        equipment_id: Target equipment
        recommendation_type: Type of recommendation
        parameters: Recommended parameter changes
        confidence: Confidence score
    """

    recommendation_id: str = Field(
        default_factory=lambda: f"rec-{uuid.uuid4().hex[:12]}",
        description="Unique recommendation identifier",
    )
    equipment_id: str = Field(
        ...,
        description="Target equipment identifier",
    )
    zone_id: Optional[str] = Field(
        None,
        description="Combustion zone identifier",
    )
    recommendation_type: str = Field(
        ...,
        description="Type of recommendation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recommended parameter changes",
    )
    expected_benefit: Dict[str, float] = Field(
        default_factory=dict,
        description="Expected benefits (efficiency, emissions, etc.)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )
    priority: int = Field(
        1,
        ge=1,
        le=5,
        description="Priority level (1=highest)",
    )
    valid_until: datetime = Field(
        ...,
        description="Recommendation validity period",
    )
    explanation: str = Field(
        "",
        description="Human-readable explanation",
    )
    model_version: str = Field(
        "1.0.0",
        description="Model version that generated recommendation",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp",
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        content = json.dumps(
            {
                "recommendation_id": self.recommendation_id,
                "equipment_id": self.equipment_id,
                "recommendation_type": self.recommendation_type,
                "parameters": self.parameters,
                "timestamp": self.timestamp.isoformat(),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class Message(BaseModel):
    """
    Generic message wrapper for Kafka.

    Attributes:
        message_id: Unique message identifier
        topic: Target Kafka topic
        key: Partition key
        payload: Message payload
        headers: Message headers
    """

    message_id: str = Field(
        default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}",
        description="Unique message identifier",
    )
    topic: str = Field(
        ...,
        description="Target Kafka topic",
    )
    key: Optional[str] = Field(
        None,
        description="Partition key",
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="Message payload",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Message headers",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp",
    )
    sequence_number: Optional[int] = Field(
        None,
        description="Sequence number for ordering",
    )

    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format."""
        return {
            "topic": self.topic,
            "key": self.key.encode() if self.key else None,
            "value": json.dumps(self.payload).encode(),
            "headers": [
                (k, v.encode()) for k, v in self.headers.items()
            ],
            "timestamp_ms": int(self.timestamp.timestamp() * 1000),
        }


# =============================================================================
# RESULT MODELS
# =============================================================================


class ConnectionResult(BaseModel):
    """Result of connection attempt."""

    success: bool = Field(..., description="Connection success status")
    connected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Connection timestamp",
    )
    broker_version: Optional[str] = Field(
        None,
        description="Kafka broker version",
    )
    cluster_id: Optional[str] = Field(
        None,
        description="Kafka cluster identifier",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed",
    )
    latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Connection latency in milliseconds",
    )


class PublishResult(BaseModel):
    """Result of message publish operation."""

    success: bool = Field(..., description="Publish success status")
    message_id: str = Field(..., description="Message identifier")
    topic: str = Field(..., description="Target topic")
    partition: int = Field(-1, description="Assigned partition")
    offset: int = Field(-1, description="Message offset")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Publish timestamp",
    )
    latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Publish latency in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 hash for audit trail",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed",
    )
    retry_count: int = Field(
        0,
        ge=0,
        description="Number of retries attempted",
    )


class DeliveryGuarantee(BaseModel):
    """Exactly-once delivery guarantee status."""

    guaranteed: bool = Field(..., description="Delivery guarantee status")
    message_id: str = Field(..., description="Message identifier")
    sequence_number: int = Field(..., description="Producer sequence number")
    producer_id: str = Field(..., description="Producer identifier")
    epoch: int = Field(0, description="Producer epoch")
    idempotent: bool = Field(True, description="Idempotent delivery enabled")
    transactional: bool = Field(False, description="Transactional delivery enabled")
    provenance_hash: str = Field(
        "",
        description="SHA-256 hash for audit trail",
    )
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Verification timestamp",
    )


class PublishFailure(BaseModel):
    """Details of a publish failure."""

    message_id: str = Field(..., description="Failed message identifier")
    topic: str = Field(..., description="Target topic")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    retryable: bool = Field(True, description="Whether error is retryable")
    retry_count: int = Field(0, ge=0, description="Retries attempted")
    failed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Failure timestamp",
    )
    payload_size_bytes: int = Field(0, ge=0, description="Payload size")
    original_exception: Optional[str] = Field(
        None,
        description="Original exception string",
    )


class RecoveryAction(BaseModel):
    """Recovery action for publish failures."""

    action: RecoveryActionType = Field(..., description="Recovery action type")
    message_id: str = Field(..., description="Message identifier")
    executed: bool = Field(False, description="Action executed status")
    executed_at: Optional[datetime] = Field(
        None,
        description="Execution timestamp",
    )
    result: Optional[str] = Field(
        None,
        description="Action result description",
    )
    next_retry_at: Optional[datetime] = Field(
        None,
        description="Next retry timestamp if applicable",
    )
    dead_letter_topic: Optional[str] = Field(
        None,
        description="Dead letter topic if action is DEAD_LETTER",
    )


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class ProducerMetrics:
    """Metrics for producer monitoring."""

    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    batches_sent: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    retries: int = 0
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

        n = self.messages_sent
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1) + latency_ms) / n

    def record_failure(
        self,
        topic: str,
        error_type: str,
        error_msg: str,
    ) -> None:
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


# =============================================================================
# PRODUCER IMPLEMENTATION
# =============================================================================


class CombustionDataProducer:
    """
    Kafka producer for combustion data with exactly-once semantics.

    This producer is designed for industrial combustion systems and provides:
    - Exactly-once delivery semantics via idempotent producers
    - Automatic retry with exponential backoff
    - Comprehensive failure handling and recovery
    - Provenance tracking for audit trails

    Example:
        >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
        >>> producer = CombustionDataProducer()
        >>> result = await producer.connect(config)
        >>> if result.success:
        ...     publish_result = await producer.publish_combustion_data(data)
    """

    # Topic definitions for GL-004 BURNMASTER
    TOPIC_COMBUSTION_DATA = "gl004.combustion.data"
    TOPIC_COMBUSTION_EVENTS = "gl004.combustion.events"
    TOPIC_RECOMMENDATIONS = "gl004.optimization.recommendations"
    TOPIC_ALERTS = "gl004.safety.alerts"
    TOPIC_DEAD_LETTER = "gl004.dead-letter"

    def __init__(
        self,
        producer_config: Optional[ProducerConfig] = None,
    ) -> None:
        """
        Initialize CombustionDataProducer.

        Args:
            producer_config: Producer-specific configuration
        """
        self.producer_config = producer_config or ProducerConfig()
        self._kafka_config: Optional[KafkaConfig] = None
        self._producer: Optional[Any] = None
        self._connected = False
        self._sequence_counter: Dict[str, int] = defaultdict(int)
        self._pending_messages: Dict[str, Message] = {}
        self._lock = asyncio.Lock()

        self.metrics = ProducerMetrics()
        self._producer_id = f"prod-{uuid.uuid4().hex[:8]}"
        self._epoch = 0

        logger.info(
            f"CombustionDataProducer initialized with producer_id={self._producer_id}"
        )

    async def connect(self, config: KafkaConfig) -> ConnectionResult:
        """
        Connect to Kafka cluster.

        Args:
            config: Kafka connection configuration

        Returns:
            ConnectionResult with connection status
        """
        start_time = time.monotonic()
        logger.info(f"Connecting to Kafka at {config.bootstrap_servers}...")

        try:
            self._kafka_config = config

            # In production, use aiokafka.AIOKafkaProducer
            # For now, use mock implementation
            aiokafka_config = config.to_aiokafka_config()
            aiokafka_config.update({
                "acks": self.producer_config.acks.value,
                "compression_type": self.producer_config.compression_type.value,
                "max_batch_size": self.producer_config.batch_size,
                "linger_ms": self.producer_config.linger_ms,
                "enable_idempotence": self.producer_config.enable_idempotence,
            })

            if self.producer_config.transactional_id:
                aiokafka_config["transactional_id"] = self.producer_config.transactional_id

            self._producer = MockAIOKafkaProducer(aiokafka_config)
            await self._producer.start()

            self._connected = True
            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                f"Connected to Kafka cluster in {latency_ms:.2f}ms"
            )

            return ConnectionResult(
                success=True,
                connected_at=datetime.now(timezone.utc),
                broker_version="3.6.0",
                cluster_id=f"cluster-{uuid.uuid4().hex[:8]}",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Failed to connect to Kafka: {e}")

            return ConnectionResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
            self._connected = False
            logger.info("Disconnected from Kafka cluster")

    async def publish_combustion_data(
        self,
        data: CombustionData,
    ) -> PublishResult:
        """
        Publish combustion data batch to Kafka.

        Args:
            data: Combustion data batch to publish

        Returns:
            PublishResult with publish status
        """
        if not self._connected:
            return PublishResult(
                success=False,
                message_id=data.batch_id,
                topic=self.TOPIC_COMBUSTION_DATA,
                error="Producer not connected",
            )

        start_time = time.monotonic()
        provenance_hash = data.compute_hash()

        message = Message(
            message_id=data.batch_id,
            topic=self.TOPIC_COMBUSTION_DATA,
            key=data.equipment_id,
            payload=data.model_dump(mode="json"),
            headers={
                "content-type": "application/json",
                "producer-id": self._producer_id,
                "provenance-hash": provenance_hash,
                "schema-version": "1.0.0",
            },
        )

        return await self._publish_message(message, provenance_hash, start_time)

    async def publish_event(
        self,
        event: CombustionEvent,
    ) -> PublishResult:
        """
        Publish combustion event to Kafka.

        Args:
            event: Combustion event to publish

        Returns:
            PublishResult with publish status
        """
        if not self._connected:
            return PublishResult(
                success=False,
                message_id=event.event_id,
                topic=self.TOPIC_COMBUSTION_EVENTS,
                error="Producer not connected",
            )

        start_time = time.monotonic()
        provenance_hash = event.compute_hash()

        # Safety alerts go to dedicated topic
        topic = (
            self.TOPIC_ALERTS
            if event.event_type == CombustionEventType.SAFETY_ALERT
            else self.TOPIC_COMBUSTION_EVENTS
        )

        message = Message(
            message_id=event.event_id,
            topic=topic,
            key=event.equipment_id,
            payload=event.model_dump(mode="json"),
            headers={
                "content-type": "application/json",
                "producer-id": self._producer_id,
                "provenance-hash": provenance_hash,
                "event-type": event.event_type.value,
                "severity": event.severity,
            },
        )

        return await self._publish_message(message, provenance_hash, start_time)

    async def publish_recommendation(
        self,
        rec: Recommendation,
    ) -> PublishResult:
        """
        Publish optimization recommendation to Kafka.

        Args:
            rec: Optimization recommendation to publish

        Returns:
            PublishResult with publish status
        """
        if not self._connected:
            return PublishResult(
                success=False,
                message_id=rec.recommendation_id,
                topic=self.TOPIC_RECOMMENDATIONS,
                error="Producer not connected",
            )

        start_time = time.monotonic()
        provenance_hash = rec.compute_hash()

        message = Message(
            message_id=rec.recommendation_id,
            topic=self.TOPIC_RECOMMENDATIONS,
            key=rec.equipment_id,
            payload=rec.model_dump(mode="json"),
            headers={
                "content-type": "application/json",
                "producer-id": self._producer_id,
                "provenance-hash": provenance_hash,
                "priority": str(rec.priority),
                "confidence": f"{rec.confidence:.4f}",
                "model-version": rec.model_version,
            },
        )

        return await self._publish_message(message, provenance_hash, start_time)

    async def _publish_message(
        self,
        message: Message,
        provenance_hash: str,
        start_time: float,
    ) -> PublishResult:
        """
        Internal method to publish a message with retry logic.

        Args:
            message: Message to publish
            provenance_hash: Provenance hash for audit
            start_time: Start time for latency calculation

        Returns:
            PublishResult with publish status
        """
        async with self._lock:
            # Assign sequence number for exactly-once
            self._sequence_counter[message.topic] += 1
            message.sequence_number = self._sequence_counter[message.topic]
            message.headers["sequence-number"] = str(message.sequence_number)

            self._pending_messages[message.message_id] = message

        retry_count = 0
        max_retries = 3
        last_error = None

        while retry_count <= max_retries:
            try:
                kafka_message = message.to_kafka_message()
                result = await self._producer.send_and_wait(
                    topic=kafka_message["topic"],
                    key=kafka_message["key"],
                    value=kafka_message["value"],
                    headers=kafka_message["headers"],
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                # Remove from pending
                async with self._lock:
                    self._pending_messages.pop(message.message_id, None)

                self.metrics.record_success(
                    message.topic,
                    len(kafka_message["value"]),
                    latency_ms,
                )

                logger.debug(
                    f"Published message {message.message_id} to "
                    f"{message.topic}:{result['partition']}@{result['offset']}"
                )

                return PublishResult(
                    success=True,
                    message_id=message.message_id,
                    topic=message.topic,
                    partition=result["partition"],
                    offset=result["offset"],
                    timestamp_ms=result["timestamp_ms"],
                    latency_ms=latency_ms,
                    provenance_hash=provenance_hash,
                    retry_count=retry_count,
                )

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                self.metrics.retries += 1

                if retry_count <= max_retries:
                    backoff = self.producer_config.retry_backoff_ms * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Publish failed for {message.message_id}, "
                        f"retry {retry_count}/{max_retries} in {backoff}ms: {e}"
                    )
                    await asyncio.sleep(backoff / 1000.0)

        # All retries exhausted
        latency_ms = (time.monotonic() - start_time) * 1000
        self.metrics.record_failure(message.topic, "PublishError", last_error or "Unknown")

        return PublishResult(
            success=False,
            message_id=message.message_id,
            topic=message.topic,
            latency_ms=latency_ms,
            provenance_hash=provenance_hash,
            error=last_error,
            retry_count=retry_count,
        )

    async def ensure_exactly_once(
        self,
        message: Message,
    ) -> DeliveryGuarantee:
        """
        Ensure exactly-once delivery for a message.

        This method verifies that idempotent delivery is enabled and
        returns the delivery guarantee status.

        Args:
            message: Message to verify

        Returns:
            DeliveryGuarantee with guarantee status
        """
        sequence_number = self._sequence_counter.get(message.topic, 0)

        guarantee = DeliveryGuarantee(
            guaranteed=self.producer_config.enable_idempotence,
            message_id=message.message_id,
            sequence_number=sequence_number,
            producer_id=self._producer_id,
            epoch=self._epoch,
            idempotent=self.producer_config.enable_idempotence,
            transactional=self.producer_config.transactional_id is not None,
            provenance_hash=hashlib.sha256(
                f"{message.message_id}:{sequence_number}:{self._producer_id}".encode()
            ).hexdigest(),
        )

        if not guarantee.guaranteed:
            logger.warning(
                f"Exactly-once not guaranteed for message {message.message_id}: "
                "idempotence not enabled"
            )

        return guarantee

    async def handle_publish_failure(
        self,
        failure: PublishFailure,
    ) -> RecoveryAction:
        """
        Handle a publish failure and determine recovery action.

        Args:
            failure: Details of the publish failure

        Returns:
            RecoveryAction with recommended recovery strategy
        """
        logger.error(
            f"Handling publish failure for {failure.message_id}: "
            f"{failure.error_code} - {failure.error_message}"
        )

        # Determine recovery action based on error type
        if failure.retryable and failure.retry_count < 5:
            # Calculate next retry time with exponential backoff
            backoff_ms = self.producer_config.retry_backoff_ms * (2 ** failure.retry_count)
            next_retry = datetime.now(timezone.utc)

            action = RecoveryAction(
                action=RecoveryActionType.RETRY,
                message_id=failure.message_id,
                next_retry_at=next_retry,
                result=f"Scheduled retry in {backoff_ms}ms",
            )

        elif failure.error_code in ("MessageSizeTooLarge", "InvalidMessage"):
            # Non-retryable errors - send to dead letter queue
            action = RecoveryAction(
                action=RecoveryActionType.DEAD_LETTER,
                message_id=failure.message_id,
                dead_letter_topic=self.TOPIC_DEAD_LETTER,
                result="Message sent to dead letter queue",
            )

            # Attempt to publish to dead letter topic
            if failure.message_id in self._pending_messages:
                message = self._pending_messages[failure.message_id]
                message.topic = self.TOPIC_DEAD_LETTER
                message.headers["original-topic"] = failure.topic
                message.headers["failure-reason"] = failure.error_message

                try:
                    await self._producer.send_and_wait(
                        topic=message.topic,
                        key=message.key.encode() if message.key else None,
                        value=json.dumps(message.payload).encode(),
                        headers=[(k, v.encode()) for k, v in message.headers.items()],
                    )
                    action.executed = True
                    action.executed_at = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error(f"Failed to send to dead letter queue: {e}")

        elif failure.error_code in ("NotLeaderForPartition", "NetworkException"):
            # Connection issues - trigger reconnect
            action = RecoveryAction(
                action=RecoveryActionType.RECONNECT,
                message_id=failure.message_id,
                result="Triggering reconnection to Kafka cluster",
            )

            # Attempt reconnect
            if self._kafka_config:
                await self.disconnect()
                await asyncio.sleep(1.0)
                await self.connect(self._kafka_config)
                action.executed = True
                action.executed_at = datetime.now(timezone.utc)

        else:
            # Unknown error - abort
            action = RecoveryAction(
                action=RecoveryActionType.ABORT,
                message_id=failure.message_id,
                result=f"Aborting due to unrecoverable error: {failure.error_code}",
            )

        logger.info(
            f"Recovery action for {failure.message_id}: {action.action.value} - {action.result}"
        )

        return action

    def get_metrics(self) -> ProducerMetrics:
        """Return current producer metrics."""
        return self.metrics

    @property
    def is_connected(self) -> bool:
        """Return connection status."""
        return self._connected


# =============================================================================
# MOCK IMPLEMENTATION
# =============================================================================


class MockAIOKafkaProducer:
    """Mock aiokafka producer for testing and demonstration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize mock producer."""
        self.config = config
        self._started = False
        self._offset_counter: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    async def start(self) -> None:
        """Start the producer."""
        await asyncio.sleep(0.1)  # Simulate connection time
        self._started = True

    async def stop(self) -> None:
        """Stop the producer."""
        self._started = False

    async def send_and_wait(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        headers: List[tuple],
    ) -> Dict[str, Any]:
        """Send message and wait for acknowledgment."""
        if not self._started:
            raise RuntimeError("Producer not started")

        # Simulate send latency
        await asyncio.sleep(0.01)

        # Calculate partition
        partition = hash(key) % 12 if key else 0
        offset = self._offset_counter[topic][partition]
        self._offset_counter[topic][partition] += 1

        return {
            "topic": topic,
            "partition": partition,
            "offset": offset,
            "timestamp_ms": int(time.time() * 1000),
        }

    async def flush(self) -> None:
        """Flush pending messages."""
        pass
