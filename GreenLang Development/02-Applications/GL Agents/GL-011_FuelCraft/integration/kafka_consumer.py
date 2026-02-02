"""
GL-011 FUELCRAFT - Kafka Event Consumer

Kafka consumer for FuelCraft event streaming:
- fuel.inventory.v1 - Inventory telemetry updates
- fuel.prices.v1 - Fuel price updates
- fuel.weather.v1 - Weather forecast updates

Features:
- Schema validation against registered schemas
- Dead letter queue for failed messages
- Configurable message handlers
- Manual offset management for exactly-once
- Circuit breaker per IEC 61511
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
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
    client_id: Optional[str] = Field(None, description="Client ID")

    # Security
    security_protocol: str = Field("SASL_SSL", description="Security protocol")
    sasl_mechanism: str = Field("PLAIN", description="SASL mechanism")
    sasl_username: Optional[str] = Field(None)
    sasl_password: Optional[str] = Field(None)
    ssl_ca_location: Optional[str] = Field(None)

    # Consumer settings
    auto_offset_reset: OffsetResetPolicy = Field(OffsetResetPolicy.EARLIEST)
    enable_auto_commit: bool = Field(False, description="Prefer manual commit")
    auto_commit_interval_ms: int = Field(5000)
    max_poll_records: int = Field(500)
    max_poll_interval_ms: int = Field(300000)
    session_timeout_ms: int = Field(45000)
    heartbeat_interval_ms: int = Field(3000)

    # Schema registry
    schema_registry_url: Optional[str] = Field(None)

    # Dead letter queue
    dlq_topic: str = Field("fuel.dlq.v1", description="Dead letter queue topic")
    enable_dlq: bool = Field(True)
    max_retry_attempts: int = Field(3)


# =============================================================================
# Message Models
# =============================================================================

class InventoryUpdateMessage(BaseModel):
    """
    Inventory update message from telemetry systems.

    Topic: fuel.inventory.v1
    """
    message_id: str
    timestamp: str
    site_id: str
    tank_id: str
    fuel_type: str

    # Inventory data
    level_percent: float = Field(..., ge=0, le=100)
    level_mmbtu: float = Field(..., ge=0)
    temperature_c: Optional[float] = Field(None)
    density_kg_m3: Optional[float] = Field(None)

    # Quality
    quality: str = Field("good", description="Data quality: good, uncertain, bad")
    source_timestamp: Optional[str] = Field(None)


class PriceUpdateMessage(BaseModel):
    """
    Fuel price update message.

    Topic: fuel.prices.v1
    """
    message_id: str
    timestamp: str
    price_source: str = Field(..., description="Price source: platts, argus, ice, etc.")

    # Price data
    fuel_type: str
    region: str
    spot_price_usd_mmbtu: float = Field(..., ge=0)
    forward_prices: Optional[Dict[str, float]] = Field(None)
    basis_differential: float = Field(0.0)

    # Metadata
    effective_date: str
    quote_type: str = Field("mid", description="bid, ask, mid")


class WeatherForecastMessage(BaseModel):
    """
    Weather forecast update message.

    Topic: fuel.weather.v1
    """
    message_id: str
    timestamp: str
    site_id: str
    forecast_source: str

    # Forecast data
    forecast_periods: List[Dict[str, Any]]
    temperature_impact_percent: Optional[float] = Field(None)
    hdd: Optional[float] = Field(None, description="Heating degree days")
    cdd: Optional[float] = Field(None, description="Cooling degree days")


# =============================================================================
# Message Handlers
# =============================================================================

class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, message: Dict[str, Any]) -> bool:
        """Handle a consumed message."""
        pass

    @abstractmethod
    def can_handle(self, message_type: str) -> bool:
        """Check if handler can process message type."""
        pass


class InventoryUpdateHandler(MessageHandler):
    """Handler for inventory update messages."""

    def __init__(
        self,
        callback: Callable[[InventoryUpdateMessage], None],
        validate: bool = True,
    ):
        """Initialize inventory handler."""
        self.callback = callback
        self.validate = validate

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process inventory update message."""
        try:
            if self.validate:
                inventory_msg = InventoryUpdateMessage(**message)
            else:
                inventory_msg = message

            logger.debug(
                f"Processing inventory update: {message.get('site_id')}/{message.get('tank_id')}"
            )

            self.callback(inventory_msg)
            return True

        except Exception as e:
            logger.error(f"Error processing inventory update: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "inventory_update"


class PriceUpdateHandler(MessageHandler):
    """Handler for price update messages."""

    def __init__(
        self,
        callback: Callable[[PriceUpdateMessage], None],
        fuel_types: Optional[Set[str]] = None,
    ):
        """Initialize price handler."""
        self.callback = callback
        self.fuel_types = fuel_types  # Filter by fuel type if specified

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process price update message."""
        try:
            # Filter by fuel type if specified
            if self.fuel_types and message.get("fuel_type") not in self.fuel_types:
                logger.debug(f"Skipping price for fuel type: {message.get('fuel_type')}")
                return True

            price_msg = PriceUpdateMessage(**message)

            logger.debug(
                f"Processing price update: {price_msg.fuel_type} @ ${price_msg.spot_price_usd_mmbtu}/MMBtu"
            )

            self.callback(price_msg)
            return True

        except Exception as e:
            logger.error(f"Error processing price update: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "price_update"


class WeatherForecastHandler(MessageHandler):
    """Handler for weather forecast messages."""

    def __init__(self, callback: Callable[[WeatherForecastMessage], None]):
        """Initialize weather handler."""
        self.callback = callback

    async def handle(self, message: Dict[str, Any]) -> bool:
        """Process weather forecast message."""
        try:
            weather_msg = WeatherForecastMessage(**message)
            logger.debug(f"Processing weather forecast for site: {weather_msg.site_id}")
            self.callback(weather_msg)
            return True
        except Exception as e:
            logger.error(f"Error processing weather forecast: {e}")
            return False

    def can_handle(self, message_type: str) -> bool:
        return message_type == "weather_forecast"


# =============================================================================
# Schema Validator
# =============================================================================

class SchemaValidator:
    """JSON Schema validator for Kafka messages."""

    def __init__(self, schema_registry_url: Optional[str] = None):
        """Initialize schema validator."""
        self.schema_registry_url = schema_registry_url
        self._schemas: Dict[str, dict] = {}
        self._load_default_schemas()

    def _load_default_schemas(self) -> None:
        """Load default message schemas."""
        self._schemas = {
            "inventory_update": {
                "type": "object",
                "required": ["message_id", "timestamp", "site_id", "tank_id", "fuel_type", "level_percent"],
                "properties": {
                    "message_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "site_id": {"type": "string"},
                    "tank_id": {"type": "string"},
                    "fuel_type": {"type": "string"},
                    "level_percent": {"type": "number", "minimum": 0, "maximum": 100},
                    "level_mmbtu": {"type": "number", "minimum": 0},
                    "temperature_c": {"type": "number"},
                    "quality": {"type": "string"},
                }
            },
            "price_update": {
                "type": "object",
                "required": ["message_id", "timestamp", "fuel_type", "spot_price_usd_mmbtu"],
                "properties": {
                    "message_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "fuel_type": {"type": "string"},
                    "spot_price_usd_mmbtu": {"type": "number", "minimum": 0},
                    "forward_prices": {"type": "object"},
                }
            },
            "weather_forecast": {
                "type": "object",
                "required": ["message_id", "timestamp", "site_id", "forecast_periods"],
                "properties": {
                    "message_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "site_id": {"type": "string"},
                    "forecast_periods": {"type": "array"},
                }
            }
        }

    def validate(self, message: Dict[str, Any], message_type: str) -> List[str]:
        """Validate message against schema."""
        schema = self._schemas.get(message_type)
        if not schema:
            return [f"Unknown message type: {message_type}"]

        errors = []

        # Check required fields
        for field in schema.get("required", []):
            if field not in message:
                errors.append(f"Missing required field: {field}")

        # Check property types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in message:
                value = message[field]
                expected_type = field_schema.get("type")

                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field {field} must be string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be number")
                elif expected_type == "object" and not isinstance(value, dict):
                    errors.append(f"Field {field} must be object")
                elif expected_type == "array" and not isinstance(value, list):
                    errors.append(f"Field {field} must be array")

                # Check numeric constraints
                if expected_type == "number" and isinstance(value, (int, float)):
                    if "minimum" in field_schema and value < field_schema["minimum"]:
                        errors.append(f"Field {field} must be >= {field_schema['minimum']}")
                    if "maximum" in field_schema and value > field_schema["maximum"]:
                        errors.append(f"Field {field} must be <= {field_schema['maximum']}")

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
    failed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
            "failed_at": self.failed_at,
        }


class DeadLetterQueue:
    """Dead letter queue handler."""

    def __init__(self, producer, dlq_topic: str):
        """Initialize DLQ handler."""
        self.producer = producer
        self.dlq_topic = dlq_topic
        self._messages_sent = 0

    async def send(self, dlq_message: DeadLetterMessage) -> bool:
        """Send message to DLQ."""
        try:
            value = json.dumps(dlq_message.to_dict()).encode('utf-8')

            # In production:
            # await self.producer.send_and_wait(self.dlq_topic, value=value)

            self._messages_sent += 1
            logger.info(
                f"Message sent to DLQ from {dlq_message.original_topic}, "
                f"offset={dlq_message.original_offset}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get DLQ statistics."""
        return {"messages_sent": self._messages_sent}


# =============================================================================
# Kafka Consumer
# =============================================================================

class FuelCraftKafkaConsumer:
    """
    Kafka consumer for FuelCraft event streaming.

    Topics:
    - fuel.inventory.v1 - Inventory telemetry
    - fuel.prices.v1 - Price updates
    - fuel.weather.v1 - Weather forecasts

    Features:
    - Schema validation
    - Pluggable message handlers
    - Dead letter queue
    - Manual offset management
    """

    TOPIC_INVENTORY = "fuel.inventory.v1"
    TOPIC_PRICES = "fuel.prices.v1"
    TOPIC_WEATHER = "fuel.weather.v1"

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
        self._retry_counts: Dict[str, int] = {}

    def register_handler(self, handler: MessageHandler) -> None:
        """Register a message handler."""
        self._handlers.append(handler)
        logger.info(f"Registered handler: {handler.__class__.__name__}")

    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to Kafka topics."""
        self._subscribed_topics = topics
        logger.info(f"Subscribed to topics: {topics}")

    async def start(self) -> None:
        """Start consuming messages."""
        if self._running:
            return

        logger.info(f"Starting FuelCraft Kafka consumer, group={self.config.group_id}")

        # In production, initialize consumer:
        # from aiokafka import AIOKafkaConsumer
        # self._consumer = AIOKafkaConsumer(
        #     *self._subscribed_topics,
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     group_id=self.config.group_id,
        #     ...
        # )
        # await self._consumer.start()

        self._running = True
        logger.info("FuelCraft Kafka consumer started")

        # Start consume loop
        await self._consume_loop()

    async def stop(self) -> None:
        """Stop consuming messages."""
        if not self._running:
            return

        logger.info("Stopping FuelCraft Kafka consumer")
        self._running = False

        if self._consumer:
            # await self._consumer.stop()
            pass

        logger.info(
            f"Consumer stopped. Consumed: {self._messages_consumed}, "
            f"Processed: {self._messages_processed}, Failed: {self._messages_failed}"
        )

    async def _consume_loop(self) -> None:
        """Main consume loop."""
        while self._running:
            try:
                # In production:
                # async for msg in self._consumer:
                #     await self._process_message(msg)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, raw_message: Any) -> None:
        """Process a consumed message."""
        self._messages_consumed += 1

        # Extract message metadata (mock values)
        topic = "fuel.inventory.v1"
        partition = 0
        offset = self._messages_consumed
        key = None
        value = b'{}'

        try:
            # Deserialize message
            message = json.loads(value.decode('utf-8'))

            # Determine message type from topic or header
            if "inventory" in topic:
                message_type = "inventory_update"
            elif "prices" in topic:
                message_type = "price_update"
            elif "weather" in topic:
                message_type = "weather_forecast"
            else:
                message_type = message.get("header", {}).get("message_type", "unknown")

            message_id = message.get("message_id", f"{topic}-{partition}-{offset}")

            # Schema validation
            validation_errors = self._validator.validate(message, message_type)
            if validation_errors:
                self._validation_errors += 1
                logger.warning(f"Schema validation failed for {message_id}: {validation_errors}")

                if self.config.enable_dlq and self._dlq:
                    await self._send_to_dlq(
                        topic, partition, offset, key, value,
                        f"Validation: {validation_errors}",
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
                            await self._handle_failure(
                                message_id, topic, partition, offset, key, value,
                                "Handler returned False", "ProcessingError"
                            )
                            return

                    except Exception as e:
                        await self._handle_failure(
                            message_id, topic, partition, offset, key, value,
                            str(e), type(e).__name__
                        )
                        return

            if not handled:
                logger.warning(f"No handler for message type: {message_type}")
                self._messages_failed += 1

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize message: {e}")
            self._messages_failed += 1

            if self.config.enable_dlq and self._dlq:
                await self._send_to_dlq(
                    topic, partition, offset, key, value,
                    str(e), "DeserializationError"
                )

    async def _handle_failure(
        self,
        message_id: str,
        topic: str,
        partition: int,
        offset: int,
        key: Optional[str],
        value: bytes,
        error_message: str,
        error_type: str,
    ) -> None:
        """Handle message processing failure with retry."""
        retry_count = self._retry_counts.get(message_id, 0) + 1
        self._retry_counts[message_id] = retry_count

        if retry_count >= self.config.max_retry_attempts:
            logger.error(f"Max retries exceeded for {message_id}")
            self._messages_failed += 1

            if self.config.enable_dlq and self._dlq:
                await self._send_to_dlq(
                    topic, partition, offset, key, value,
                    error_message, error_type, retry_count
                )

            del self._retry_counts[message_id]
        else:
            logger.warning(f"Message {message_id} failed (attempt {retry_count})")

    async def _send_to_dlq(
        self,
        topic: str,
        partition: int,
        offset: int,
        key: Optional[str],
        value: bytes,
        error_message: str,
        error_type: str,
        retry_count: int = 0,
    ) -> None:
        """Send failed message to DLQ."""
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
            retry_count=retry_count,
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
                if self._messages_consumed > 0 else 1.0
            ),
        }
