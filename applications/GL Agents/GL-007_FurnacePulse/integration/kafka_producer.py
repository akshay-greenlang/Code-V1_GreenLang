"""
Kafka Event Producer for FurnacePulse

Implements topic-based event streaming following the FurnacePulse playbook design:
- furnacepulse.<site>.<furnace_id>.telemetry - Real-time sensor data
- furnacepulse.<site>.<furnace_id>.events - Operational events
- furnacepulse.models.inference - ML model predictions
- furnacepulse.alerts - Alert notifications

Includes Schema Registry integration for backward-compatible schema evolution.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Message Schema Definitions (Appendix B compliance)
# =============================================================================

class MessageType(str, Enum):
    """Message types for FurnacePulse events."""
    TELEMETRY = "telemetry"
    EVENT = "event"
    INFERENCE = "inference"
    ALERT = "alert"
    COMMAND = "command"
    ACKNOWLEDGMENT = "ack"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MessageHeader(BaseModel):
    """
    Standard message header for all FurnacePulse messages.
    Per Appendix B specification.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    version: str = Field("1.0", description="Schema version for evolution")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    source: str = Field(..., description="Source system identifier")
    correlation_id: Optional[str] = Field(None, description="For request-response correlation")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")


class TelemetryMessage(BaseModel):
    """
    Telemetry message format per Appendix B.

    Contains sensor readings from furnace instrumentation.
    Published to: furnacepulse.<site>.<furnace_id>.telemetry
    """
    header: MessageHeader
    site_id: str = Field(..., description="Site identifier")
    furnace_id: str = Field(..., description="Furnace identifier")
    readings: List[Dict[str, Any]] = Field(..., description="Sensor readings")

    class Reading(BaseModel):
        """Individual sensor reading."""
        tag_id: str
        value: Union[float, int, bool, str]
        unit: str
        quality: DataQuality
        source_timestamp: str
        acquisition_timestamp: str

    @validator('readings')
    def validate_readings(cls, v):
        """Ensure readings have required fields."""
        required_fields = {'tag_id', 'value', 'quality', 'source_timestamp'}
        for reading in v:
            if not required_fields.issubset(reading.keys()):
                raise ValueError(f"Reading missing required fields: {required_fields - reading.keys()}")
        return v

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.dict(), default=str).encode('utf-8')


class EventMessage(BaseModel):
    """
    Operational event message format per Appendix B.

    Represents discrete events (start/stop, mode changes, faults).
    Published to: furnacepulse.<site>.<furnace_id>.events
    """
    header: MessageHeader
    site_id: str
    furnace_id: str
    event_type: str = Field(..., description="Event type code")
    event_name: str = Field(..., description="Human-readable event name")
    description: str = Field("", description="Event description")
    severity: AlertSeverity = Field(AlertSeverity.INFO)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    related_tags: List[str] = Field(default_factory=list)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.dict(), default=str).encode('utf-8')


class InferenceMessage(BaseModel):
    """
    Model inference result message per Appendix B.

    Contains predictions from ML models (RUL, anomaly detection, etc.)
    Published to: furnacepulse.models.inference
    """
    header: MessageHeader
    site_id: str
    furnace_id: str
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    inference_type: str = Field(..., description="Type: rul, anomaly, health_index, etc.")

    # Prediction results
    predictions: Dict[str, Any] = Field(..., description="Model predictions")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="SHAP values or similar")

    # Context
    input_window_start: str = Field(..., description="Start of input data window")
    input_window_end: str = Field(..., description="End of input data window")
    processing_time_ms: float = Field(..., description="Inference processing time")

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.dict(), default=str).encode('utf-8')


class AlertMessage(BaseModel):
    """
    Alert notification message per Appendix B.

    Published to: furnacepulse.alerts
    """
    header: MessageHeader
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str
    furnace_id: str
    component_id: Optional[str] = Field(None, description="Specific component if applicable")

    # Alert details
    alert_type: str = Field(..., description="Alert type code")
    alert_name: str = Field(..., description="Human-readable alert name")
    severity: AlertSeverity
    description: str
    recommended_actions: List[str] = Field(default_factory=list)

    # Supporting data
    trigger_value: Optional[float] = Field(None)
    threshold_value: Optional[float] = Field(None)
    related_tags: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list, description="URLs to plots, IR images")

    # State
    acknowledged: bool = Field(False)
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[str] = Field(None)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.dict(), default=str).encode('utf-8')


# =============================================================================
# Schema Registry Client
# =============================================================================

class SchemaRegistryConfig(BaseModel):
    """Schema Registry configuration."""
    url: str = Field(..., description="Schema Registry URL")
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)
    ssl_ca_location: Optional[str] = Field(None)


class SchemaEvolutionMode(str, Enum):
    """Schema evolution compatibility modes."""
    BACKWARD = "BACKWARD"  # New schema can read old data
    FORWARD = "FORWARD"  # Old schema can read new data
    FULL = "FULL"  # Both backward and forward compatible
    NONE = "NONE"  # No compatibility checking


class SchemaRegistryClient:
    """
    Client for Confluent Schema Registry.

    Handles schema registration, retrieval, and compatibility checking.
    Ensures backward-compatible schema evolution.
    """

    def __init__(self, config: SchemaRegistryConfig):
        """Initialize schema registry client."""
        self.config = config
        self._schemas: Dict[str, int] = {}  # subject -> schema_id
        self._schema_cache: Dict[int, dict] = {}  # schema_id -> schema

    async def register_schema(
        self,
        subject: str,
        schema: dict,
        schema_type: str = "JSON"
    ) -> int:
        """
        Register a schema with the registry.

        Args:
            subject: Schema subject (usually topic-value)
            schema: JSON schema definition
            schema_type: Schema type (JSON, AVRO, PROTOBUF)

        Returns:
            Schema ID
        """
        # In production, use confluent-kafka-python:
        # from confluent_kafka.schema_registry import SchemaRegistryClient
        # schema_registry_conf = {'url': self.config.url}
        # sr_client = SchemaRegistryClient(schema_registry_conf)
        # schema_id = sr_client.register_schema(subject, Schema(json.dumps(schema), schema_type))

        logger.info(f"Registering schema for subject: {subject}")

        # Mock implementation
        schema_id = hash(json.dumps(schema, sort_keys=True)) % 1000000
        self._schemas[subject] = schema_id
        self._schema_cache[schema_id] = schema

        return schema_id

    async def get_schema(self, schema_id: int) -> Optional[dict]:
        """Get schema by ID."""
        return self._schema_cache.get(schema_id)

    async def get_latest_schema(self, subject: str) -> Optional[dict]:
        """Get latest schema for a subject."""
        schema_id = self._schemas.get(subject)
        if schema_id:
            return self._schema_cache.get(schema_id)
        return None

    async def check_compatibility(
        self,
        subject: str,
        schema: dict,
        version: str = "latest"
    ) -> bool:
        """
        Check if schema is compatible with existing versions.

        Args:
            subject: Schema subject
            schema: New schema to check
            version: Version to check against

        Returns:
            True if compatible
        """
        # In production, use Schema Registry API
        logger.info(f"Checking compatibility for subject: {subject}")
        return True  # Mock - always compatible

    async def set_compatibility(
        self,
        subject: str,
        mode: SchemaEvolutionMode
    ) -> None:
        """Set compatibility mode for a subject."""
        logger.info(f"Setting compatibility mode {mode} for subject: {subject}")


# =============================================================================
# Kafka Producer Configuration
# =============================================================================

class KafkaProducerConfig(BaseModel):
    """Kafka producer configuration."""

    # Bootstrap servers
    bootstrap_servers: str = Field(..., description="Comma-separated list of brokers")

    # Security
    security_protocol: str = Field("SASL_SSL", description="Security protocol")
    sasl_mechanism: str = Field("PLAIN", description="SASL mechanism")
    sasl_username: Optional[str] = Field(None)
    sasl_password: Optional[str] = Field(None)  # From vault
    ssl_ca_location: Optional[str] = Field(None, description="CA certificate path")

    # Producer settings
    acks: str = Field("all", description="Acknowledgment level: 0, 1, all")
    retries: int = Field(3, description="Number of retries")
    retry_backoff_ms: int = Field(100, description="Retry backoff in ms")
    batch_size: int = Field(16384, description="Batch size in bytes")
    linger_ms: int = Field(5, description="Linger time for batching")
    buffer_memory: int = Field(33554432, description="Buffer memory in bytes")
    compression_type: str = Field("gzip", description="Compression: none, gzip, snappy, lz4")

    # Idempotence and transactions
    enable_idempotence: bool = Field(True, description="Enable exactly-once semantics")
    transactional_id: Optional[str] = Field(None, description="Transaction ID for exactly-once")

    # Schema registry
    schema_registry: Optional[SchemaRegistryConfig] = Field(None)


# =============================================================================
# Kafka Event Producer
# =============================================================================

class KafkaEventProducer:
    """
    Kafka producer for FurnacePulse event streaming.

    Implements the FurnacePulse topic design:
    - furnacepulse.<site>.<furnace_id>.telemetry
    - furnacepulse.<site>.<furnace_id>.events
    - furnacepulse.models.inference
    - furnacepulse.alerts

    Features:
    - Schema Registry integration for backward-compatible evolution
    - Exactly-once semantics with idempotence
    - Automatic batching and compression
    - Delivery confirmations and error handling

    Usage:
        config = KafkaProducerConfig(bootstrap_servers="kafka:9092")
        producer = KafkaEventProducer(config)
        await producer.start()

        # Publish telemetry
        await producer.publish_telemetry("site1", "furnace1", readings)

        # Publish alert
        await producer.publish_alert(alert_message)
    """

    # Topic name templates
    TOPIC_TELEMETRY = "furnacepulse.{site}.{furnace}.telemetry"
    TOPIC_EVENTS = "furnacepulse.{site}.{furnace}.events"
    TOPIC_INFERENCE = "furnacepulse.models.inference"
    TOPIC_ALERTS = "furnacepulse.alerts"

    def __init__(self, config: KafkaProducerConfig):
        """Initialize Kafka producer."""
        self.config = config
        self._producer = None
        self._schema_registry: Optional[SchemaRegistryClient] = None
        self._started = False

        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._bytes_sent = 0

        # Registered schemas
        self._schema_ids: Dict[str, int] = {}

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._started:
            return

        logger.info(f"Starting Kafka producer for {self.config.bootstrap_servers}")

        # Initialize Schema Registry client
        if self.config.schema_registry:
            self._schema_registry = SchemaRegistryClient(self.config.schema_registry)
            await self._register_schemas()

        # In production, use aiokafka or confluent-kafka:
        # from aiokafka import AIOKafkaProducer
        # self._producer = AIOKafkaProducer(
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     security_protocol=self.config.security_protocol,
        #     sasl_mechanism=self.config.sasl_mechanism,
        #     sasl_plain_username=self.config.sasl_username,
        #     sasl_plain_password=self.config.sasl_password,
        #     acks=self.config.acks,
        #     compression_type=self.config.compression_type,
        #     enable_idempotence=self.config.enable_idempotence,
        # )
        # await self._producer.start()

        self._started = True
        logger.info("Kafka producer started successfully")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if not self._started:
            return

        logger.info("Stopping Kafka producer")

        if self._producer:
            # Flush pending messages
            # await self._producer.flush()
            # await self._producer.stop()
            pass

        self._started = False
        logger.info(f"Kafka producer stopped. Sent: {self._messages_sent}, Failed: {self._messages_failed}")

    async def _register_schemas(self) -> None:
        """Register message schemas with Schema Registry."""
        if not self._schema_registry:
            return

        # Define JSON schemas for each message type
        schemas = {
            "telemetry": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["header", "site_id", "furnace_id", "readings"],
                "properties": {
                    "header": {"type": "object"},
                    "site_id": {"type": "string"},
                    "furnace_id": {"type": "string"},
                    "readings": {"type": "array"}
                }
            },
            "event": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["header", "site_id", "furnace_id", "event_type"],
                "properties": {
                    "header": {"type": "object"},
                    "site_id": {"type": "string"},
                    "furnace_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "event_name": {"type": "string"},
                    "severity": {"type": "string"}
                }
            },
            "inference": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["header", "model_id", "predictions", "confidence"],
                "properties": {
                    "header": {"type": "object"},
                    "model_id": {"type": "string"},
                    "model_version": {"type": "string"},
                    "inference_type": {"type": "string"},
                    "predictions": {"type": "object"},
                    "confidence": {"type": "number"}
                }
            },
            "alert": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["header", "alert_id", "alert_type", "severity"],
                "properties": {
                    "header": {"type": "object"},
                    "alert_id": {"type": "string"},
                    "alert_type": {"type": "string"},
                    "severity": {"type": "string"},
                    "description": {"type": "string"},
                    "recommended_actions": {"type": "array"}
                }
            }
        }

        for msg_type, schema in schemas.items():
            subject = f"furnacepulse-{msg_type}-value"
            try:
                schema_id = await self._schema_registry.register_schema(subject, schema)
                self._schema_ids[msg_type] = schema_id
                logger.info(f"Registered schema for {subject}: ID={schema_id}")
            except Exception as e:
                logger.error(f"Failed to register schema for {subject}: {e}")

    def _get_topic(self, topic_template: str, site: str = "", furnace: str = "") -> str:
        """Generate topic name from template."""
        return topic_template.format(site=site, furnace=furnace)

    async def _produce(
        self,
        topic: str,
        key: Optional[str],
        value: bytes,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Internal method to produce a message.

        Args:
            topic: Kafka topic
            key: Message key for partitioning
            value: Message value bytes
            headers: Optional message headers

        Returns:
            True if successful
        """
        if not self._started:
            raise RuntimeError("Producer not started")

        try:
            # Convert headers to Kafka format
            kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]

            # In production:
            # await self._producer.send_and_wait(
            #     topic,
            #     key=key.encode() if key else None,
            #     value=value,
            #     headers=kafka_headers
            # )

            self._messages_sent += 1
            self._bytes_sent += len(value)

            logger.debug(f"Produced message to {topic}, key={key}, size={len(value)}")
            return True

        except Exception as e:
            self._messages_failed += 1
            logger.error(f"Failed to produce message to {topic}: {e}")
            return False

    async def publish_telemetry(
        self,
        site_id: str,
        furnace_id: str,
        readings: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Publish telemetry data to Kafka.

        Args:
            site_id: Site identifier
            furnace_id: Furnace identifier
            readings: List of sensor readings
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if successful
        """
        topic = self._get_topic(self.TOPIC_TELEMETRY, site_id, furnace_id)

        message = TelemetryMessage(
            header=MessageHeader(
                message_type=MessageType.TELEMETRY,
                source=f"furnacepulse.{site_id}.{furnace_id}",
                correlation_id=correlation_id
            ),
            site_id=site_id,
            furnace_id=furnace_id,
            readings=readings
        )

        # Use furnace_id as key for partition affinity
        return await self._produce(
            topic=topic,
            key=furnace_id,
            value=message.to_kafka_message(),
            headers={
                "message_type": "telemetry",
                "schema_version": "1.0"
            }
        )

    async def publish_event(
        self,
        site_id: str,
        furnace_id: str,
        event_type: str,
        event_name: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        related_tags: Optional[List[str]] = None
    ) -> bool:
        """
        Publish operational event to Kafka.

        Args:
            site_id: Site identifier
            furnace_id: Furnace identifier
            event_type: Event type code
            event_name: Human-readable event name
            severity: Event severity
            description: Event description
            attributes: Additional event attributes
            related_tags: Related sensor tags

        Returns:
            True if successful
        """
        topic = self._get_topic(self.TOPIC_EVENTS, site_id, furnace_id)

        message = EventMessage(
            header=MessageHeader(
                message_type=MessageType.EVENT,
                source=f"furnacepulse.{site_id}.{furnace_id}"
            ),
            site_id=site_id,
            furnace_id=furnace_id,
            event_type=event_type,
            event_name=event_name,
            severity=severity,
            description=description,
            attributes=attributes or {},
            related_tags=related_tags or []
        )

        return await self._produce(
            topic=topic,
            key=furnace_id,
            value=message.to_kafka_message(),
            headers={
                "message_type": "event",
                "event_type": event_type,
                "severity": severity.value
            }
        )

    async def publish_inference(
        self,
        site_id: str,
        furnace_id: str,
        model_id: str,
        model_version: str,
        inference_type: str,
        predictions: Dict[str, Any],
        confidence: float,
        input_window_start: datetime,
        input_window_end: datetime,
        processing_time_ms: float,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Publish model inference result to Kafka.

        Args:
            site_id: Site identifier
            furnace_id: Furnace identifier
            model_id: Model identifier
            model_version: Model version
            inference_type: Type of inference (rul, anomaly, health_index)
            predictions: Prediction results
            confidence: Prediction confidence (0-1)
            input_window_start: Start of input data window
            input_window_end: End of input data window
            processing_time_ms: Inference processing time
            feature_importance: Optional SHAP values or feature importance

        Returns:
            True if successful
        """
        message = InferenceMessage(
            header=MessageHeader(
                message_type=MessageType.INFERENCE,
                source=f"furnacepulse.{site_id}.{furnace_id}.{model_id}"
            ),
            site_id=site_id,
            furnace_id=furnace_id,
            model_id=model_id,
            model_version=model_version,
            inference_type=inference_type,
            predictions=predictions,
            confidence=confidence,
            feature_importance=feature_importance,
            input_window_start=input_window_start.isoformat() + "Z",
            input_window_end=input_window_end.isoformat() + "Z",
            processing_time_ms=processing_time_ms
        )

        return await self._produce(
            topic=self.TOPIC_INFERENCE,
            key=f"{site_id}.{furnace_id}.{model_id}",
            value=message.to_kafka_message(),
            headers={
                "message_type": "inference",
                "model_id": model_id,
                "inference_type": inference_type
            }
        )

    async def publish_alert(self, alert: AlertMessage) -> bool:
        """
        Publish alert to Kafka.

        Args:
            alert: Alert message

        Returns:
            True if successful
        """
        # Ensure header is set correctly
        if not alert.header:
            alert.header = MessageHeader(
                message_type=MessageType.ALERT,
                source=f"furnacepulse.{alert.site_id}.{alert.furnace_id}"
            )

        return await self._produce(
            topic=self.TOPIC_ALERTS,
            key=alert.alert_id,
            value=alert.to_kafka_message(),
            headers={
                "message_type": "alert",
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "site_id": alert.site_id,
                "furnace_id": alert.furnace_id
            }
        )

    async def publish_batch_telemetry(
        self,
        site_id: str,
        furnace_id: str,
        readings_batch: List[List[Dict[str, Any]]]
    ) -> int:
        """
        Publish a batch of telemetry messages efficiently.

        Args:
            site_id: Site identifier
            furnace_id: Furnace identifier
            readings_batch: List of reading lists

        Returns:
            Number of successfully published messages
        """
        success_count = 0

        for readings in readings_batch:
            if await self.publish_telemetry(site_id, furnace_id, readings):
                success_count += 1

        logger.info(
            f"Batch telemetry: {success_count}/{len(readings_batch)} messages published"
        )
        return success_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        return {
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "bytes_sent": self._bytes_sent,
            "success_rate": (
                self._messages_sent / (self._messages_sent + self._messages_failed)
                if (self._messages_sent + self._messages_failed) > 0
                else 1.0
            )
        }
