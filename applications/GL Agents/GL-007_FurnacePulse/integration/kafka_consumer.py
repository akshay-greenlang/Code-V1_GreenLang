"""
Kafka Event Consumer for FurnacePulse

Consumes telemetry, command, and event topics with:
- Schema validation against registered schemas
- Error handling with dead letter queue
- Configurable message handlers
- Offset management and consumer group coordination
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Consumer Configuration
# =============================================================================

class OffsetResetPolicy(str, Enum):
    """Kafka offset reset policy."""
    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class KafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration."""

    # Bootstrap servers
    bootstrap_servers: str = Field(..., description="Comma-separated list of brokers")

    # Consumer group
    group_id: str = Field(..., description="Consumer group ID")
    client_id: Optional[str] = Field(None, description="Client ID for tracking")

    # Security
    security_protocol: str = Field("SASL_SSL", description="Security protocol")
    sasl_mechanism: str = Field("PLAIN", description="SASL mechanism")
    sasl_username: Optional[str] = Field(None)
    sasl_password: Optional[str] = Field(None)  # From vault
    ssl_ca_location: Optional[str] = Field(None)

    # Consumer settings
    auto_offset_reset: OffsetResetPolicy = Field(
        OffsetResetPolicy.EARLIEST,
        description="Offset reset policy"
    )
    enable_auto_commit: bool = Field(False, description="Enable auto commit (prefer manual)")
    auto_commit_interval_ms: int = Field(5000, description="Auto commit interval")
    max_poll_records: int = Field(500, description="Max records per poll")
    max_poll_interval_ms: int = Field(300000, description="Max poll interval")
    session_timeout_ms: int = Field(45000, description="Session timeout")
    heartbeat_interval_ms: int = Field(3000, description="Heartbeat interval")

    # Schema registry
    schema_registry_url: Optional[str] = Field(None)

    # Dead letter queue
    dlq_topic: str = Field("furnacepulse.dlq", description="Dead letter queue topic")
    enable_dlq: bool = Field(True, description="Enable dead letter queue")
    max_retry_attempts: int = Field(3, description="Max retry before DLQ")


# =============================================================================
# Message Handler Interface
# =============================================================================

class MessageHandler(ABC):
    """
    Abstract base class for message handlers.

    Implement this interface to handle specific message types.
    """

    @abstractmethod
    async def handle(self, message: Dict[str, Any]) -> bool:
        """
        Handle a consumed message.

        Args:
            message: Deserialized message payload

        Returns:
            True if message was processed successfully

        Raises:
            MessageProcessingError: If processing fails permanently
        """
        pass

    @abstractmethod
    def can_handle(self, message_type: str) -> bool:
        """
        Check if this handler can process a message type.

        Args:
            message_type: Message type from header

        Returns:
            True if this handler can process the message
        """
        pass


class TelemetryHandler(MessageHandler):
    """Handler for telemetry messages."""

    def __init__(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Initialize telemetry handler.

        Args:
            callback: Function to call with telemetry data
        """
        self.callback = callback

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process telemetry message."""
        try:
            site_id = message.get("site_id")
            furnace_id = message.get("furnace_id")
            readings = message.get("readings", [])

            logger.debug(
                f"Processing telemetry: {site_id}/{furnace_id}, "
                f"{len(readings)} readings"
            )

            self.callback(message)
            return True

        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "telemetry"


class CommandHandler(MessageHandler):
    """Handler for command messages."""

    def __init__(
        self,
        command_executor: Callable[[str, Dict[str, Any]], bool],
        allowed_commands: Optional[Set[str]] = None
    ):
        """
        Initialize command handler.

        Args:
            command_executor: Function to execute commands
            allowed_commands: Set of allowed command types
        """
        self.command_executor = command_executor
        self.allowed_commands = allowed_commands or set()

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process command message."""
        command_type = message.get("command_type")
        command_params = message.get("parameters", {})

        # Validate command is allowed
        if self.allowed_commands and command_type not in self.allowed_commands:
            logger.warning(f"Command type not allowed: {command_type}")
            return False

        try:
            logger.info(f"Executing command: {command_type}")
            result = self.command_executor(command_type, command_params)
            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "command"


class AlertAcknowledgmentHandler(MessageHandler):
    """Handler for alert acknowledgment messages."""

    def __init__(self, ack_callback: Callable[[str, str, datetime], None]):
        """
        Initialize acknowledgment handler.

        Args:
            ack_callback: Callback(alert_id, acknowledged_by, timestamp)
        """
        self.ack_callback = ack_callback

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process acknowledgment message."""
        try:
            alert_id = message.get("alert_id")
            acknowledged_by = message.get("acknowledged_by")
            timestamp = datetime.fromisoformat(
                message.get("timestamp", datetime.utcnow().isoformat())
            )

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            self.ack_callback(alert_id, acknowledged_by, timestamp)
            return True

        except Exception as e:
            logger.error(f"Error processing acknowledgment: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "ack" or message_type == "acknowledgment"


# =============================================================================
# Schema Validator
# =============================================================================

class SchemaValidator:
    """
    JSON Schema validator for Kafka messages.

    Validates incoming messages against registered schemas.
    """

    def __init__(self, schema_registry_url: Optional[str] = None):
        """Initialize schema validator."""
        self.schema_registry_url = schema_registry_url
        self._schemas: Dict[str, dict] = {}
        self._load_default_schemas()

    def _load_default_schemas(self) -> None:
        """Load default message schemas."""
        self._schemas = {
            "telemetry": {
                "type": "object",
                "required": ["header", "site_id", "furnace_id", "readings"],
                "properties": {
                    "header": {
                        "type": "object",
                        "required": ["message_id", "message_type", "timestamp"]
                    },
                    "site_id": {"type": "string"},
                    "furnace_id": {"type": "string"},
                    "readings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["tag_id", "value"]
                        }
                    }
                }
            },
            "event": {
                "type": "object",
                "required": ["header", "site_id", "furnace_id", "event_type"],
                "properties": {
                    "header": {"type": "object"},
                    "site_id": {"type": "string"},
                    "furnace_id": {"type": "string"},
                    "event_type": {"type": "string"}
                }
            },
            "command": {
                "type": "object",
                "required": ["header", "command_type"],
                "properties": {
                    "header": {"type": "object"},
                    "command_type": {"type": "string"},
                    "target_site": {"type": "string"},
                    "target_furnace": {"type": "string"},
                    "parameters": {"type": "object"}
                }
            },
            "inference": {
                "type": "object",
                "required": ["header", "model_id", "predictions", "confidence"],
                "properties": {
                    "header": {"type": "object"},
                    "model_id": {"type": "string"},
                    "predictions": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "alert": {
                "type": "object",
                "required": ["header", "alert_id", "alert_type", "severity"],
                "properties": {
                    "header": {"type": "object"},
                    "alert_id": {"type": "string"},
                    "alert_type": {"type": "string"},
                    "severity": {"type": "string"}
                }
            },
            "ack": {
                "type": "object",
                "required": ["alert_id", "acknowledged_by"],
                "properties": {
                    "alert_id": {"type": "string"},
                    "acknowledged_by": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        }

    def validate(self, message: Dict[str, Any], message_type: str) -> List[str]:
        """
        Validate message against schema.

        Args:
            message: Message to validate
            message_type: Type of message

        Returns:
            List of validation errors (empty if valid)
        """
        schema = self._schemas.get(message_type)
        if not schema:
            return [f"Unknown message type: {message_type}"]

        errors = []

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in message:
                errors.append(f"Missing required field: {field}")

        # Check property types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in message:
                value = message[field]
                expected_type = field_schema.get("type")

                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field {field} must be string, got {type(value).__name__}")
                elif expected_type == "object" and not isinstance(value, dict):
                    errors.append(f"Field {field} must be object, got {type(value).__name__}")
                elif expected_type == "array" and not isinstance(value, list):
                    errors.append(f"Field {field} must be array, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be number, got {type(value).__name__}")

                # Check number constraints
                if expected_type == "number" and isinstance(value, (int, float)):
                    if "minimum" in field_schema and value < field_schema["minimum"]:
                        errors.append(
                            f"Field {field} must be >= {field_schema['minimum']}"
                        )
                    if "maximum" in field_schema and value > field_schema["maximum"]:
                        errors.append(
                            f"Field {field} must be <= {field_schema['maximum']}"
                        )

        return errors


# =============================================================================
# Dead Letter Queue
# =============================================================================

@dataclass
class DeadLetterMessage:
    """Message sent to dead letter queue."""
    original_topic: str
    original_partition: int
    original_offset: int
    original_key: Optional[str]
    original_value: bytes
    error_message: str
    error_type: str
    retry_count: int
    failed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_topic": self.original_topic,
            "original_partition": self.original_partition,
            "original_offset": self.original_offset,
            "original_key": self.original_key,
            "original_value": self.original_value.decode('utf-8', errors='replace'),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "failed_at": self.failed_at
        }


class DeadLetterQueue:
    """
    Dead letter queue handler.

    Sends failed messages to DLQ topic for later analysis and reprocessing.
    """

    def __init__(self, producer, dlq_topic: str):
        """
        Initialize DLQ handler.

        Args:
            producer: Kafka producer instance
            dlq_topic: DLQ topic name
        """
        self.producer = producer
        self.dlq_topic = dlq_topic
        self._messages_sent = 0

    async def send(self, dlq_message: DeadLetterMessage) -> bool:
        """
        Send message to dead letter queue.

        Args:
            dlq_message: Dead letter message

        Returns:
            True if successful
        """
        try:
            value = json.dumps(dlq_message.to_dict()).encode('utf-8')

            # In production:
            # await self.producer.send_and_wait(
            #     self.dlq_topic,
            #     key=dlq_message.original_key.encode() if dlq_message.original_key else None,
            #     value=value,
            #     headers=[
            #         ("error_type", dlq_message.error_type.encode()),
            #         ("original_topic", dlq_message.original_topic.encode())
            #     ]
            # )

            self._messages_sent += 1
            logger.info(
                f"Message sent to DLQ: {dlq_message.original_topic} "
                f"offset={dlq_message.original_offset}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get DLQ statistics."""
        return {"messages_sent": self._messages_sent}


# =============================================================================
# Kafka Event Consumer
# =============================================================================

class KafkaEventConsumer:
    """
    Kafka consumer for FurnacePulse event streaming.

    Features:
    - Multiple topic subscription
    - Schema validation
    - Pluggable message handlers
    - Dead letter queue for failed messages
    - Manual offset management

    Usage:
        config = KafkaConsumerConfig(
            bootstrap_servers="kafka:9092",
            group_id="furnacepulse-processor"
        )

        consumer = KafkaEventConsumer(config)

        # Register handlers
        consumer.register_handler(TelemetryHandler(process_telemetry))
        consumer.register_handler(CommandHandler(execute_command))

        # Subscribe and start consuming
        await consumer.subscribe([
            "furnacepulse.site1.furnace1.telemetry",
            "furnacepulse.commands"
        ])
        await consumer.start()
    """

    def __init__(self, config: KafkaConsumerConfig):
        """Initialize Kafka consumer."""
        self.config = config
        self._consumer = None
        self._handlers: List[MessageHandler] = []
        self._validator = SchemaValidator(config.schema_registry_url)
        self._dlq: Optional[DeadLetterQueue] = None
        self._running = False
        self._subscribed_topics: List[str] = []

        # Metrics
        self._messages_consumed = 0
        self._messages_processed = 0
        self._messages_failed = 0
        self._validation_errors = 0

        # Retry tracking
        self._retry_counts: Dict[str, int] = {}  # message_id -> retry_count

    def register_handler(self, handler: MessageHandler) -> None:
        """
        Register a message handler.

        Args:
            handler: Handler to register
        """
        self._handlers.append(handler)
        logger.info(f"Registered handler: {handler.__class__.__name__}")

    async def subscribe(self, topics: List[str]) -> None:
        """
        Subscribe to Kafka topics.

        Args:
            topics: List of topic names or patterns
        """
        self._subscribed_topics = topics
        logger.info(f"Subscribed to topics: {topics}")

        # In production:
        # self._consumer.subscribe(topics)

    async def start(self) -> None:
        """Start consuming messages."""
        if self._running:
            return

        logger.info(
            f"Starting Kafka consumer for group {self.config.group_id}"
        )

        # In production, initialize consumer:
        # from aiokafka import AIOKafkaConsumer
        # self._consumer = AIOKafkaConsumer(
        #     *self._subscribed_topics,
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     group_id=self.config.group_id,
        #     auto_offset_reset=self.config.auto_offset_reset.value,
        #     enable_auto_commit=self.config.enable_auto_commit,
        # )
        # await self._consumer.start()

        # Initialize DLQ if enabled
        if self.config.enable_dlq:
            # self._dlq = DeadLetterQueue(producer, self.config.dlq_topic)
            pass

        self._running = True
        logger.info("Kafka consumer started")

        # Start consume loop
        await self._consume_loop()

    async def stop(self) -> None:
        """Stop consuming messages."""
        if not self._running:
            return

        logger.info("Stopping Kafka consumer")
        self._running = False

        if self._consumer:
            # await self._consumer.stop()
            pass

        logger.info(
            f"Kafka consumer stopped. "
            f"Consumed: {self._messages_consumed}, "
            f"Processed: {self._messages_processed}, "
            f"Failed: {self._messages_failed}"
        )

    async def _consume_loop(self) -> None:
        """Main consume loop."""
        while self._running:
            try:
                # In production:
                # async for msg in self._consumer:
                #     await self._process_message(msg)

                # Mock: wait and check for stop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def _process_message(self, raw_message: Any) -> None:
        """
        Process a consumed message.

        Args:
            raw_message: Raw Kafka message
        """
        self._messages_consumed += 1

        # Extract message metadata
        # topic = raw_message.topic
        # partition = raw_message.partition
        # offset = raw_message.offset
        # key = raw_message.key.decode() if raw_message.key else None
        # value = raw_message.value

        # Mock values for demonstration
        topic = "furnacepulse.site1.furnace1.telemetry"
        partition = 0
        offset = self._messages_consumed
        key = None
        value = b'{}'

        try:
            # Deserialize message
            message = json.loads(value.decode('utf-8'))

            # Extract message type from header
            header = message.get("header", {})
            message_type = header.get("message_type", "unknown")
            message_id = header.get("message_id", f"{topic}-{partition}-{offset}")

            # Schema validation
            validation_errors = self._validator.validate(message, message_type)
            if validation_errors:
                self._validation_errors += 1
                logger.warning(
                    f"Schema validation failed for {message_id}: {validation_errors}"
                )

                # Send to DLQ if validation fails
                if self.config.enable_dlq and self._dlq:
                    await self._send_to_dlq(
                        topic, partition, offset, key, value,
                        f"Validation errors: {validation_errors}",
                        "ValidationError"
                    )
                return

            # Find and execute handler
            handled = False
            for handler in self._handlers:
                if handler.can_handle(message_type):
                    try:
                        success = await handler.handle(message)
                        if success:
                            self._messages_processed += 1
                            handled = True
                            break
                        else:
                            # Handler returned False - might need retry
                            await self._handle_processing_failure(
                                message_id, topic, partition, offset, key, value,
                                "Handler returned False",
                                "ProcessingError"
                            )
                            return

                    except Exception as e:
                        await self._handle_processing_failure(
                            message_id, topic, partition, offset, key, value,
                            str(e),
                            type(e).__name__
                        )
                        return

            if not handled:
                logger.warning(f"No handler found for message type: {message_type}")
                self._messages_failed += 1

            # Commit offset (manual commit)
            if not self.config.enable_auto_commit:
                # await self._consumer.commit({
                #     TopicPartition(topic, partition): offset + 1
                # })
                pass

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize message: {e}")
            self._messages_failed += 1

            if self.config.enable_dlq and self._dlq:
                await self._send_to_dlq(
                    topic, partition, offset, key, value,
                    f"JSON decode error: {e}",
                    "DeserializationError"
                )

        except Exception as e:
            logger.error(f"Unexpected error processing message: {e}")
            self._messages_failed += 1

    async def _handle_processing_failure(
        self,
        message_id: str,
        topic: str,
        partition: int,
        offset: int,
        key: Optional[str],
        value: bytes,
        error_message: str,
        error_type: str
    ) -> None:
        """Handle message processing failure with retry logic."""
        # Track retry count
        retry_count = self._retry_counts.get(message_id, 0) + 1
        self._retry_counts[message_id] = retry_count

        if retry_count >= self.config.max_retry_attempts:
            # Max retries exceeded - send to DLQ
            logger.error(
                f"Max retries ({self.config.max_retry_attempts}) exceeded "
                f"for message {message_id}"
            )
            self._messages_failed += 1

            if self.config.enable_dlq and self._dlq:
                await self._send_to_dlq(
                    topic, partition, offset, key, value,
                    error_message, error_type, retry_count
                )

            # Clean up retry tracking
            del self._retry_counts[message_id]

        else:
            # Will retry on next poll
            logger.warning(
                f"Message {message_id} failed (attempt {retry_count}), "
                f"will retry: {error_message}"
            )

    async def _send_to_dlq(
        self,
        topic: str,
        partition: int,
        offset: int,
        key: Optional[str],
        value: bytes,
        error_message: str,
        error_type: str,
        retry_count: int = 0
    ) -> None:
        """Send failed message to dead letter queue."""
        if not self._dlq:
            return

        dlq_message = DeadLetterMessage(
            original_topic=topic,
            original_partition=partition,
            original_offset=offset,
            original_key=key,
            original_value=value,
            error_message=error_message,
            error_type=error_type,
            retry_count=retry_count
        )

        await self._dlq.send(dlq_message)

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "messages_consumed": self._messages_consumed,
            "messages_processed": self._messages_processed,
            "messages_failed": self._messages_failed,
            "validation_errors": self._validation_errors,
            "pending_retries": len(self._retry_counts),
            "handlers_registered": len(self._handlers),
            "subscribed_topics": self._subscribed_topics,
            "success_rate": (
                self._messages_processed / self._messages_consumed
                if self._messages_consumed > 0
                else 1.0
            )
        }

    async def seek_to_beginning(self, topic: str, partition: int) -> None:
        """Seek to beginning of a partition for replay."""
        logger.info(f"Seeking to beginning of {topic}-{partition}")
        # In production:
        # tp = TopicPartition(topic, partition)
        # await self._consumer.seek_to_beginning(tp)

    async def seek_to_end(self, topic: str, partition: int) -> None:
        """Seek to end of a partition."""
        logger.info(f"Seeking to end of {topic}-{partition}")
        # In production:
        # tp = TopicPartition(topic, partition)
        # await self._consumer.seek_to_end(tp)

    async def get_committed_offsets(self) -> Dict[str, int]:
        """Get committed offsets for all subscribed partitions."""
        # In production:
        # offsets = await self._consumer.committed(...)
        return {}
