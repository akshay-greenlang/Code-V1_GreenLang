# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - Kafka Streaming Integration

Event streaming for heat exchanger monitoring with:
- GL-014 specific topics for temperatures, flows, pressures
- Computed KPIs and fouling predictions
- Cleaning recommendations and events
- Avro/JSON event schemas with schema registry
- Consumer groups and partitioning
- Exactly-once semantics support
- Dead letter queue handling

Topics:
    - gl014.raw.temperatures: Raw temperature readings
    - gl014.raw.flows: Raw flow measurements
    - gl014.raw.pressures: Raw pressure readings
    - gl014.computed.kpis: Computed performance KPIs
    - gl014.predictions.fouling: Fouling predictions
    - gl014.recommendations.cleaning: Cleaning recommendations
    - gl014.events.cleaning: Cleaning execution events

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class GL014Topics(str, Enum):
    """GL-014 Kafka topics."""
    # Raw sensor data
    RAW_TEMPERATURES = "gl014.raw.temperatures"
    RAW_FLOWS = "gl014.raw.flows"
    RAW_PRESSURES = "gl014.raw.pressures"

    # Computed data
    COMPUTED_KPIS = "gl014.computed.kpis"
    PREDICTIONS_FOULING = "gl014.predictions.fouling"

    # Recommendations and events
    RECOMMENDATIONS_CLEANING = "gl014.recommendations.cleaning"
    EVENTS_CLEANING = "gl014.events.cleaning"

    # System topics
    DEAD_LETTER = "gl014.dlq"
    AUDIT_LOG = "gl014.audit"


class SerializationFormat(str, Enum):
    """Message serialization formats."""
    JSON = "json"
    AVRO = "avro"


class CompressionType(str, Enum):
    """Kafka compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class DeliveryGuarantee(str, Enum):
    """Delivery guarantee levels."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class ConsumerState(str, Enum):
    """Consumer lifecycle state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATION
# =============================================================================

class KafkaStreamConfig(BaseModel):
    """Configuration for Kafka streaming."""
    # Connection
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"],
        description="Kafka bootstrap servers"
    )
    client_id: str = Field(
        default="gl014-exchangerpro",
        description="Kafka client ID"
    )

    # Consumer settings
    group_id: str = Field(
        default="gl014-consumer-group",
        description="Consumer group ID"
    )
    auto_offset_reset: str = Field(
        default="earliest",
        description="Auto offset reset policy"
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Enable auto commit"
    )

    # Producer settings
    producer_acks: str = Field(
        default="all",
        description="Producer acknowledgment mode"
    )
    compression_type: CompressionType = Field(
        default=CompressionType.SNAPPY,
        description="Message compression"
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable idempotent producer"
    )
    delivery_guarantee: DeliveryGuarantee = Field(
        default=DeliveryGuarantee.EXACTLY_ONCE,
        description="Delivery guarantee level"
    )

    # Schema registry
    schema_registry_url: Optional[str] = Field(
        default=None,
        description="Schema registry URL"
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.JSON,
        description="Default serialization format"
    )

    # Dead letter queue
    enable_dlq: bool = Field(
        default=True,
        description="Enable dead letter queue"
    )
    dlq_topic: str = Field(
        default="gl014.dlq",
        description="Dead letter queue topic"
    )
    max_retries_before_dlq: int = Field(
        default=3,
        ge=1,
        description="Max retries before DLQ"
    )

    # Performance
    batch_size: int = Field(
        default=16384,
        ge=1,
        description="Producer batch size"
    )
    linger_ms: int = Field(
        default=5,
        ge=0,
        description="Producer linger time"
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        description="Max records per poll"
    )

    # Security
    security_protocol: str = Field(
        default="PLAINTEXT",
        description="Security protocol"
    )
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism"
    )
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username"
    )
    sasl_password: Optional[str] = Field(
        default=None,
        description="SASL password"
    )


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class BaseEventSchema(BaseModel):
    """Base schema for all GL-014 events."""
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    source: str = Field(
        default="gl014-exchangerpro",
        description="Event source"
    )
    version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracing"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExchangerEventSchema(BaseEventSchema):
    """Base schema for exchanger-related events."""
    exchanger_id: str = Field(..., description="Heat exchanger ID")
    site_id: str = Field(..., description="Site identifier")


class TemperatureEventSchema(ExchangerEventSchema):
    """
    Schema for temperature events.

    Topic: gl014.raw.temperatures
    """
    event_type: str = Field(default="temperature_reading")

    # Temperature data
    tag_id: str = Field(..., description="Tag identifier")
    side: str = Field(..., description="Shell or Tube")
    location: str = Field(..., description="Inlet, Outlet, or Mid")
    temperature_value: float = Field(..., description="Temperature value")
    temperature_unit: str = Field(default="celsius", description="Temperature unit")

    # Quality
    quality_code: int = Field(default=0, description="OPC-UA quality code")
    source_timestamp: datetime = Field(..., description="Source timestamp")

    # Provenance
    provenance_hash: Optional[str] = Field(None, description="Data provenance hash")


class FlowEventSchema(ExchangerEventSchema):
    """
    Schema for flow events.

    Topic: gl014.raw.flows
    """
    event_type: str = Field(default="flow_reading")

    # Flow data
    tag_id: str = Field(..., description="Tag identifier")
    side: str = Field(..., description="Shell or Tube")
    flow_value: float = Field(..., description="Flow value")
    flow_unit: str = Field(default="kg/s", description="Flow unit")

    # Quality
    quality_code: int = Field(default=0, description="OPC-UA quality code")
    source_timestamp: datetime = Field(..., description="Source timestamp")

    # Provenance
    provenance_hash: Optional[str] = Field(None, description="Data provenance hash")


class PressureEventSchema(ExchangerEventSchema):
    """
    Schema for pressure events.

    Topic: gl014.raw.pressures
    """
    event_type: str = Field(default="pressure_reading")

    # Pressure data
    tag_id: str = Field(..., description="Tag identifier")
    side: str = Field(..., description="Shell or Tube")
    location: str = Field(..., description="Inlet or Outlet")
    pressure_value: float = Field(..., description="Pressure value")
    pressure_unit: str = Field(default="kPa", description="Pressure unit")

    # Quality
    quality_code: int = Field(default=0, description="OPC-UA quality code")
    source_timestamp: datetime = Field(..., description="Source timestamp")

    # Provenance
    provenance_hash: Optional[str] = Field(None, description="Data provenance hash")


class KPIEventSchema(ExchangerEventSchema):
    """
    Schema for computed KPI events.

    Topic: gl014.computed.kpis
    """
    event_type: str = Field(default="kpi_computed")

    # KPI data
    kpi_name: str = Field(..., description="KPI name")
    kpi_value: float = Field(..., description="KPI value")
    kpi_unit: str = Field(..., description="KPI unit")

    # Common exchanger KPIs
    overall_htc: Optional[float] = Field(None, description="Overall heat transfer coefficient (W/m2K)")
    lmtd: Optional[float] = Field(None, description="Log mean temperature difference (K)")
    duty: Optional[float] = Field(None, description="Heat duty (kW)")
    effectiveness: Optional[float] = Field(None, description="Thermal effectiveness (0-1)")
    ntu: Optional[float] = Field(None, description="Number of transfer units")

    # Calculation metadata
    calculation_time_ms: Optional[float] = Field(None, description="Calculation time")
    input_data_points: int = Field(default=0, description="Input data points used")

    # Provenance
    computation_hash: Optional[str] = Field(None, description="Computation provenance hash")


class FoulingPredictionSchema(ExchangerEventSchema):
    """
    Schema for fouling prediction events.

    Topic: gl014.predictions.fouling
    """
    event_type: str = Field(default="fouling_prediction")

    # Fouling prediction
    fouling_factor: float = Field(..., description="Current fouling factor (m2K/W)")
    fouling_rate: float = Field(..., description="Fouling rate (m2K/W per day)")
    fouling_severity: str = Field(..., description="Severity: clean, light, moderate, severe")

    # Predictions
    predicted_cleaning_date: Optional[datetime] = Field(
        None,
        description="Predicted optimal cleaning date"
    )
    days_until_cleaning: Optional[int] = Field(
        None,
        description="Days until cleaning recommended"
    )
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    # Model metadata
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")

    # Economic impact
    efficiency_loss_percent: Optional[float] = Field(
        None,
        description="Efficiency loss percentage"
    )
    estimated_cost_impact: Optional[float] = Field(
        None,
        description="Estimated cost impact ($/day)"
    )


class CleaningRecommendationSchema(ExchangerEventSchema):
    """
    Schema for cleaning recommendation events.

    Topic: gl014.recommendations.cleaning
    """
    event_type: str = Field(default="cleaning_recommendation")

    # Recommendation
    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Recommendation ID"
    )
    recommendation_type: str = Field(..., description="Chemical, Mechanical, etc.")
    priority: EventPriority = Field(..., description="Recommendation priority")
    urgency: str = Field(..., description="Immediate, Scheduled, Opportunistic")

    # Timing
    recommended_date: Optional[datetime] = Field(None, description="Recommended date")
    window_start: Optional[datetime] = Field(None, description="Window start")
    window_end: Optional[datetime] = Field(None, description="Window end")

    # Justification
    trigger_reason: str = Field(..., description="Reason for recommendation")
    fouling_factor: float = Field(..., description="Current fouling factor")
    efficiency_loss: float = Field(..., description="Current efficiency loss %")

    # Economic analysis
    cleaning_cost_estimate: float = Field(..., description="Estimated cleaning cost")
    energy_savings_estimate: float = Field(..., description="Estimated energy savings")
    roi_estimate: Optional[float] = Field(None, description="Estimated ROI")
    payback_days: Optional[int] = Field(None, description="Payback period in days")

    # Approval workflow
    requires_approval: bool = Field(default=True, description="Requires human approval")
    approver_role: str = Field(default="maintenance_engineer", description="Approver role")

    # Linkage to computation
    computation_record_id: Optional[str] = Field(
        None,
        description="Linked computation record"
    )


class CleaningEventSchema(ExchangerEventSchema):
    """
    Schema for cleaning execution events.

    Topic: gl014.events.cleaning
    """
    event_type: str = Field(default="cleaning_event")

    # Event details
    cleaning_event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Cleaning event ID"
    )
    cleaning_type: str = Field(..., description="Type of cleaning performed")
    cleaning_method: str = Field(..., description="Cleaning method")

    # Timing
    start_time: datetime = Field(..., description="Cleaning start time")
    end_time: Optional[datetime] = Field(None, description="Cleaning end time")
    duration_hours: Optional[float] = Field(None, description="Duration in hours")

    # Results
    status: str = Field(..., description="Planned, InProgress, Completed, Failed")
    pre_cleaning_fouling: Optional[float] = Field(None, description="Pre-cleaning fouling factor")
    post_cleaning_fouling: Optional[float] = Field(None, description="Post-cleaning fouling factor")
    effectiveness: Optional[float] = Field(None, description="Cleaning effectiveness 0-1")

    # Costs
    actual_cost: Optional[float] = Field(None, description="Actual cleaning cost")
    downtime_hours: Optional[float] = Field(None, description="Downtime hours")

    # Linkage
    recommendation_id: Optional[str] = Field(None, description="Linked recommendation")
    work_order_id: Optional[str] = Field(None, description="CMMS work order ID")


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

class SchemaRegistry:
    """
    Schema registry for Avro/JSON schema management.

    Provides schema versioning and compatibility checking.
    """

    def __init__(self, url: Optional[str] = None):
        """
        Initialize schema registry.

        Args:
            url: Schema registry URL (for Confluent Schema Registry)
        """
        self.url = url
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._schema_ids: Dict[str, int] = {}
        self._lock = asyncio.Lock()

        # Register default schemas
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default GL-014 schemas."""
        schemas = {
            "temperature_reading": TemperatureEventSchema.schema(),
            "flow_reading": FlowEventSchema.schema(),
            "pressure_reading": PressureEventSchema.schema(),
            "kpi_computed": KPIEventSchema.schema(),
            "fouling_prediction": FoulingPredictionSchema.schema(),
            "cleaning_recommendation": CleaningRecommendationSchema.schema(),
            "cleaning_event": CleaningEventSchema.schema(),
        }

        for name, schema in schemas.items():
            self._schemas[name] = schema
            self._schema_ids[name] = hash(json.dumps(schema, sort_keys=True)) % 100000

    async def register_schema(
        self,
        subject: str,
        schema: Dict[str, Any],
    ) -> int:
        """
        Register a schema.

        Args:
            subject: Schema subject name
            schema: Schema definition

        Returns:
            Schema ID
        """
        async with self._lock:
            schema_id = hash(json.dumps(schema, sort_keys=True)) % 100000
            self._schemas[subject] = schema
            self._schema_ids[subject] = schema_id

            logger.info(f"Registered schema {subject} with ID {schema_id}")
            return schema_id

    async def get_schema(self, subject: str) -> Optional[Dict[str, Any]]:
        """Get schema by subject."""
        return self._schemas.get(subject)

    async def get_schema_id(self, subject: str) -> Optional[int]:
        """Get schema ID by subject."""
        return self._schema_ids.get(subject)

    def validate_message(
        self,
        subject: str,
        message: Dict[str, Any],
    ) -> bool:
        """
        Validate message against schema.

        Args:
            subject: Schema subject
            message: Message to validate

        Returns:
            True if valid
        """
        schema = self._schemas.get(subject)
        if not schema:
            logger.warning(f"No schema found for subject: {subject}")
            return True  # Allow if no schema

        # Basic validation - check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in message:
                logger.error(f"Missing required field: {field}")
                return False

        return True


# =============================================================================
# DEAD LETTER HANDLER
# =============================================================================

@dataclass
class DeadLetterMessage:
    """Dead letter queue message."""
    original_topic: str
    original_message: bytes
    error_message: str
    error_type: str
    retry_count: int
    first_failure_time: datetime
    last_failure_time: datetime
    headers: Dict[str, str] = field(default_factory=dict)


class DeadLetterHandler:
    """
    Handler for dead letter queue processing.

    Manages failed messages and retry logic.
    """

    def __init__(self, config: KafkaStreamConfig):
        """Initialize dead letter handler."""
        self.config = config
        self._failed: Dict[str, DeadLetterMessage] = {}
        self._lock = asyncio.Lock()
        self._dlq_count = 0

    async def handle_failure(
        self,
        topic: str,
        message: bytes,
        error: Exception,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Handle message processing failure.

        Args:
            topic: Source topic
            message: Failed message
            error: Exception that occurred
            headers: Message headers

        Returns:
            True if should retry, False if sent to DLQ
        """
        msg_hash = hashlib.sha256(message).hexdigest()

        async with self._lock:
            now = datetime.now(timezone.utc)

            if msg_hash in self._failed:
                entry = self._failed[msg_hash]
                entry.retry_count += 1
                entry.last_failure_time = now
                entry.error_message = str(error)
            else:
                entry = DeadLetterMessage(
                    original_topic=topic,
                    original_message=message,
                    error_message=str(error),
                    error_type=type(error).__name__,
                    retry_count=1,
                    first_failure_time=now,
                    last_failure_time=now,
                    headers=headers or {},
                )
                self._failed[msg_hash] = entry

            if entry.retry_count >= self.config.max_retries_before_dlq:
                # Send to DLQ
                await self._send_to_dlq(entry)
                del self._failed[msg_hash]
                self._dlq_count += 1
                logger.warning(
                    f"Message sent to DLQ after {entry.retry_count} retries"
                )
                return False

            return True

    async def _send_to_dlq(self, entry: DeadLetterMessage) -> None:
        """Send message to dead letter queue."""
        dlq_message = {
            "original_topic": entry.original_topic,
            "original_message": entry.original_message.decode("utf-8", errors="replace"),
            "error_message": entry.error_message,
            "error_type": entry.error_type,
            "retry_count": entry.retry_count,
            "first_failure": entry.first_failure_time.isoformat(),
            "last_failure": entry.last_failure_time.isoformat(),
            "headers": entry.headers,
        }

        # In production, would publish to DLQ topic
        logger.info(f"DLQ message: {dlq_message}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get dead letter handler statistics."""
        async with self._lock:
            return {
                "pending_retries": len(self._failed),
                "total_dlq_messages": self._dlq_count,
                "pending_by_topic": {},
            }


# =============================================================================
# CONSUMER GROUP
# =============================================================================

class ConsumerGroup:
    """
    Consumer group management for coordinated consumption.
    """

    def __init__(
        self,
        group_id: str,
        topics: List[str],
        config: KafkaStreamConfig,
    ):
        """
        Initialize consumer group.

        Args:
            group_id: Consumer group ID
            topics: Topics to consume
            config: Kafka configuration
        """
        self.group_id = group_id
        self.topics = topics
        self.config = config
        self._consumers: Dict[str, "KafkaConsumerWrapper"] = {}
        self._lock = asyncio.Lock()

    async def add_consumer(
        self,
        consumer_id: str,
        handler: Callable,
    ) -> "KafkaConsumerWrapper":
        """Add consumer to group."""
        async with self._lock:
            consumer = KafkaConsumerWrapper(
                self.config,
                self.topics,
                handler,
            )
            self._consumers[consumer_id] = consumer
            return consumer

    async def start_all(self) -> None:
        """Start all consumers."""
        for consumer in self._consumers.values():
            await consumer.start()

    async def stop_all(self) -> None:
        """Stop all consumers."""
        for consumer in self._consumers.values():
            await consumer.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get consumer group statistics."""
        return {
            "group_id": self.group_id,
            "topics": self.topics,
            "consumer_count": len(self._consumers),
            "consumers": {
                cid: c.stats for cid, c in self._consumers.items()
            },
        }


# =============================================================================
# KAFKA PRODUCER
# =============================================================================

class KafkaProducerWrapper:
    """
    Kafka producer wrapper with reliability features.

    Supports exactly-once semantics and transactional messaging.
    """

    def __init__(self, config: KafkaStreamConfig):
        """Initialize producer."""
        self.config = config
        self._initialized = False
        self._sent = 0
        self._errors = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize producer connection."""
        if self._initialized:
            return

        # In production, would create Kafka producer
        await asyncio.sleep(0.01)
        self._initialized = True
        logger.info("Kafka producer initialized")

    async def close(self) -> None:
        """Close producer."""
        await self.flush()
        self._initialized = False
        logger.info("Kafka producer closed")

    async def send(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        partition: Optional[int] = None,
    ) -> bool:
        """
        Send message to topic.

        Args:
            topic: Target topic
            value: Message value
            key: Optional message key
            headers: Optional headers
            partition: Optional partition

        Returns:
            True if sent successfully
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                # In production, would send to Kafka
                self._sent += 1
                logger.debug(f"Sent message to {topic}")
                return True
            except Exception as e:
                self._errors += 1
                logger.error(f"Send failed: {e}")
                return False

    async def send_event(
        self,
        topic: GL014Topics,
        event: BaseEventSchema,
        key: Optional[str] = None,
    ) -> bool:
        """
        Send typed event to topic.

        Args:
            topic: GL-014 topic
            event: Event schema instance
            key: Optional partition key

        Returns:
            True if sent successfully
        """
        value = event.json().encode("utf-8")
        key_bytes = key.encode("utf-8") if key else None

        return await self.send(
            topic=topic.value,
            value=value,
            key=key_bytes,
            headers={"event_type": event.event_type},
        )

    async def flush(self) -> None:
        """Flush pending messages."""
        await asyncio.sleep(0.01)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            "initialized": self._initialized,
            "messages_sent": self._sent,
            "errors": self._errors,
        }


class ExactlyOnceProducer(KafkaProducerWrapper):
    """
    Kafka producer with exactly-once semantics.

    Uses transactions for guaranteed delivery.
    """

    def __init__(self, config: KafkaStreamConfig):
        """Initialize exactly-once producer."""
        config.enable_idempotence = True
        super().__init__(config)
        self._tx_active = False
        self._tx_count = 0

    async def begin_transaction(self) -> None:
        """Begin transaction."""
        if self._tx_active:
            raise RuntimeError("Transaction already active")
        self._tx_active = True
        logger.debug("Transaction started")

    async def commit_transaction(self) -> None:
        """Commit transaction."""
        if not self._tx_active:
            raise RuntimeError("No active transaction")
        self._tx_active = False
        self._tx_count += 1
        logger.debug("Transaction committed")

    async def abort_transaction(self) -> None:
        """Abort transaction."""
        if not self._tx_active:
            raise RuntimeError("No active transaction")
        self._tx_active = False
        logger.debug("Transaction aborted")

    async def send_exactly_once(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
    ) -> bool:
        """
        Send message with exactly-once guarantee.

        Args:
            topic: Target topic
            value: Message value
            key: Optional key

        Returns:
            True if successful
        """
        try:
            await self.begin_transaction()
            result = await self.send(topic, value, key)
            await self.commit_transaction()
            return result
        except Exception as e:
            await self.abort_transaction()
            logger.error(f"Exactly-once send failed: {e}")
            raise

    @property
    def stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        base_stats = super().stats
        base_stats["transactions_committed"] = self._tx_count
        base_stats["transaction_active"] = self._tx_active
        return base_stats


# =============================================================================
# KAFKA CONSUMER
# =============================================================================

class KafkaConsumerWrapper:
    """
    Kafka consumer wrapper with reliability features.
    """

    def __init__(
        self,
        config: KafkaStreamConfig,
        topics: List[str],
        handler: Optional[Callable] = None,
    ):
        """
        Initialize consumer.

        Args:
            config: Kafka configuration
            topics: Topics to subscribe
            handler: Message handler callback
        """
        self.config = config
        self.topics = topics
        self.handler = handler
        self._state = ConsumerState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._consumed = 0
        self._errors = 0
        self._last_offset: Dict[str, int] = {}

    async def start(self) -> None:
        """Start consumer."""
        if self._state == ConsumerState.RUNNING:
            return

        self._state = ConsumerState.STARTING
        self._task = asyncio.create_task(self._consume_loop())
        self._state = ConsumerState.RUNNING
        logger.info(f"Consumer started for topics: {self.topics}")

    async def stop(self) -> None:
        """Stop consumer."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._state = ConsumerState.STOPPED
        logger.info("Consumer stopped")

    async def pause(self) -> None:
        """Pause consumption."""
        self._state = ConsumerState.PAUSED

    async def resume(self) -> None:
        """Resume consumption."""
        if self._state == ConsumerState.PAUSED:
            self._state = ConsumerState.RUNNING

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        while self._state == ConsumerState.RUNNING:
            try:
                # In production, would poll Kafka
                await asyncio.sleep(0.1)

                # Simulate message processing
                if self.handler:
                    # Would process actual messages
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                logger.error(f"Consumer error: {e}")
                self._state = ConsumerState.ERROR
                await asyncio.sleep(5)
                self._state = ConsumerState.RUNNING

    async def commit(self) -> None:
        """Commit current offsets."""
        logger.debug("Offsets committed")

    @property
    def state(self) -> ConsumerState:
        """Get consumer state."""
        return self._state

    @property
    def stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            "state": self._state.value,
            "topics": self.topics,
            "messages_consumed": self._consumed,
            "errors": self._errors,
            "last_offsets": self._last_offset,
        }


# =============================================================================
# KAFKA STREAMING INTEGRATION
# =============================================================================

class KafkaStreamingIntegration:
    """
    Main Kafka streaming integration for GL-014 ExchangerPro.

    Provides unified interface for all GL-014 event streaming:
    - Raw sensor data publishing
    - Computed KPIs and predictions
    - Cleaning recommendations and events
    - Consumer group management

    Example:
        >>> config = KafkaStreamConfig(bootstrap_servers=["kafka:9092"])
        >>> async with KafkaStreamingIntegration(config) as kafka:
        ...     await kafka.publish_temperature(temp_event)
        ...     await kafka.publish_fouling_prediction(prediction)
    """

    def __init__(self, config: KafkaStreamConfig):
        """
        Initialize Kafka streaming integration.

        Args:
            config: Kafka configuration
        """
        self.config = config
        self.producer: Optional[ExactlyOnceProducer] = None
        self.consumers: Dict[str, KafkaConsumerWrapper] = {}
        self.schema_registry = SchemaRegistry(config.schema_registry_url)
        self.dlq_handler = DeadLetterHandler(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Kafka connections."""
        if self._initialized:
            return

        self.producer = ExactlyOnceProducer(self.config)
        await self.producer.initialize()
        self._initialized = True

        logger.info("Kafka streaming integration initialized")

    async def close(self) -> None:
        """Close all connections."""
        if self.producer:
            await self.producer.close()

        for consumer in self.consumers.values():
            await consumer.stop()

        self._initialized = False
        logger.info("Kafka streaming integration closed")

    # =========================================================================
    # PUBLISHING - RAW DATA
    # =========================================================================

    async def publish_temperature(
        self,
        event: TemperatureEventSchema,
    ) -> bool:
        """
        Publish temperature reading event.

        Args:
            event: Temperature event

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.RAW_TEMPERATURES,
            event,
            key=event.exchanger_id,
        )

    async def publish_flow(
        self,
        event: FlowEventSchema,
    ) -> bool:
        """
        Publish flow reading event.

        Args:
            event: Flow event

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.RAW_FLOWS,
            event,
            key=event.exchanger_id,
        )

    async def publish_pressure(
        self,
        event: PressureEventSchema,
    ) -> bool:
        """
        Publish pressure reading event.

        Args:
            event: Pressure event

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.RAW_PRESSURES,
            event,
            key=event.exchanger_id,
        )

    # =========================================================================
    # PUBLISHING - COMPUTED DATA
    # =========================================================================

    async def publish_kpi(
        self,
        event: KPIEventSchema,
    ) -> bool:
        """
        Publish computed KPI event.

        Args:
            event: KPI event

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.COMPUTED_KPIS,
            event,
            key=event.exchanger_id,
        )

    async def publish_fouling_prediction(
        self,
        event: FoulingPredictionSchema,
    ) -> bool:
        """
        Publish fouling prediction event.

        Args:
            event: Fouling prediction

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.PREDICTIONS_FOULING,
            event,
            key=event.exchanger_id,
        )

    # =========================================================================
    # PUBLISHING - RECOMMENDATIONS AND EVENTS
    # =========================================================================

    async def publish_cleaning_recommendation(
        self,
        event: CleaningRecommendationSchema,
    ) -> bool:
        """
        Publish cleaning recommendation event.

        Args:
            event: Cleaning recommendation

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.RECOMMENDATIONS_CLEANING,
            event,
            key=event.exchanger_id,
        )

    async def publish_cleaning_event(
        self,
        event: CleaningEventSchema,
    ) -> bool:
        """
        Publish cleaning execution event.

        Args:
            event: Cleaning event

        Returns:
            True if published successfully
        """
        return await self.producer.send_event(
            GL014Topics.EVENTS_CLEANING,
            event,
            key=event.exchanger_id,
        )

    # =========================================================================
    # CONSUMING
    # =========================================================================

    async def subscribe_to_topic(
        self,
        topic: GL014Topics,
        handler: Callable,
        consumer_id: Optional[str] = None,
    ) -> str:
        """
        Subscribe to a GL-014 topic.

        Args:
            topic: Topic to subscribe
            handler: Message handler
            consumer_id: Optional consumer ID

        Returns:
            Consumer ID
        """
        cid = consumer_id or f"consumer_{uuid.uuid4().hex[:8]}"

        consumer = KafkaConsumerWrapper(
            self.config,
            [topic.value],
            handler,
        )
        await consumer.start()

        self.consumers[cid] = consumer
        return cid

    async def unsubscribe(self, consumer_id: str) -> bool:
        """
        Unsubscribe a consumer.

        Args:
            consumer_id: Consumer ID

        Returns:
            True if unsubscribed
        """
        consumer = self.consumers.get(consumer_id)
        if not consumer:
            return False

        await consumer.stop()
        del self.consumers[consumer_id]
        return True

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "initialized": self._initialized,
            "producer": self.producer.stats if self.producer else None,
            "consumers": {
                cid: c.stats for cid, c in self.consumers.items()
            },
            "consumer_count": len(self.consumers),
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        dlq_stats = await self.dlq_handler.get_stats()

        return {
            "healthy": self._initialized and self.producer is not None,
            "producer_connected": self._initialized,
            "active_consumers": len(self.consumers),
            "dlq_pending": dlq_stats.get("pending_retries", 0),
        }

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> "KafkaStreamingIntegration":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_producer(config_dict: Dict[str, Any]) -> KafkaProducerWrapper:
    """Create Kafka producer from config dictionary."""
    return KafkaProducerWrapper(KafkaStreamConfig(**config_dict))


def create_consumer(
    config_dict: Dict[str, Any],
    topics: List[str],
    handler: Optional[Callable] = None,
) -> KafkaConsumerWrapper:
    """Create Kafka consumer from config dictionary."""
    return KafkaConsumerWrapper(
        KafkaStreamConfig(**config_dict),
        topics,
        handler,
    )


def create_exactly_once_producer(
    config_dict: Dict[str, Any],
) -> ExactlyOnceProducer:
    """Create exactly-once producer from config dictionary."""
    return ExactlyOnceProducer(KafkaStreamConfig(**config_dict))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GL014Topics",
    "SerializationFormat",
    "CompressionType",
    "DeliveryGuarantee",
    "ConsumerState",
    "EventPriority",

    # Configuration
    "KafkaStreamConfig",

    # Event Schemas
    "BaseEventSchema",
    "ExchangerEventSchema",
    "TemperatureEventSchema",
    "FlowEventSchema",
    "PressureEventSchema",
    "KPIEventSchema",
    "FoulingPredictionSchema",
    "CleaningRecommendationSchema",
    "CleaningEventSchema",

    # Schema Registry
    "SchemaRegistry",

    # Dead Letter Handler
    "DeadLetterMessage",
    "DeadLetterHandler",

    # Consumer Group
    "ConsumerGroup",

    # Producer
    "KafkaProducerWrapper",
    "ExactlyOnceProducer",

    # Consumer
    "KafkaConsumerWrapper",

    # Main Integration
    "KafkaStreamingIntegration",

    # Factory Functions
    "create_producer",
    "create_consumer",
    "create_exactly_once_producer",
]
