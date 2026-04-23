"""
GL-009 ThermalIQ - Kafka Event Consumer

Consumes thermal analysis requests and sensor data for processing.

Subscribed Topics:
- thermaliq.analysis.requests - Analysis request submissions
- thermaliq.sensor.data - Real-time sensor data
- thermaliq.commands - Control commands

Features:
- Async message consumption
- Dead letter queue handling
- Message handler registration
- Automatic offset management
- Error recovery and retry logic
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ConfigDict

from .event_schemas import (
    EventType,
    AnalysisRequestedEvent,
    FluidPropertyUpdatedEvent,
    SensorDataEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Kafka Consumer Configuration
# =============================================================================

class KafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration."""
    bootstrap_servers: str = Field(..., description="Comma-separated list of brokers")
    group_id: str = Field(..., description="Consumer group ID")
    security_protocol: str = Field("SASL_SSL", description="Security protocol")
    sasl_mechanism: str = Field("PLAIN", description="SASL mechanism")
    sasl_username: Optional[str] = Field(None)
    sasl_password: Optional[str] = Field(None)
    ssl_ca_location: Optional[str] = Field(None)

    # Consumer settings
    auto_offset_reset: str = Field("latest", description="earliest or latest")
    enable_auto_commit: bool = Field(True, description="Auto commit offsets")
    auto_commit_interval_ms: int = Field(5000, description="Auto commit interval")
    max_poll_records: int = Field(100, description="Max records per poll")
    session_timeout_ms: int = Field(30000, description="Session timeout")
    heartbeat_interval_ms: int = Field(3000, description="Heartbeat interval")
    max_poll_interval_ms: int = Field(300000, description="Max poll interval")

    # Dead letter queue
    enable_dlq: bool = Field(True, description="Enable dead letter queue")
    dlq_topic_suffix: str = Field(".dlq", description="DLQ topic suffix")
    max_retries: int = Field(3, description="Max retries before DLQ")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Message Handler Type
# =============================================================================

MessageHandler = Callable[[Dict[str, Any], Dict[str, str]], None]
AsyncMessageHandler = Callable[[Dict[str, Any], Dict[str, str]], Any]


# =============================================================================
# ThermalIQ Kafka Consumer
# =============================================================================

class ThermalIQKafkaConsumer:
    """
    Kafka consumer for GL-009 ThermalIQ.

    Subscribes to:
    - thermaliq.analysis.requests - Analysis requests
    - thermaliq.sensor.data - Sensor data streams
    - thermaliq.commands - Control commands

    Triggers:
    - Thermal analysis on request
    - Fluid property calculations
    - Exergy analysis
    - Alert generation for anomalies
    """

    # Topic patterns
    TOPIC_ANALYSIS_REQUESTS = "thermaliq.analysis.requests"
    TOPIC_SENSOR_DATA = "thermaliq.sensor.data"
    TOPIC_COMMANDS = "thermaliq.commands"

    def __init__(self, config: KafkaConsumerConfig):
        """
        Initialize Kafka consumer.

        Args:
            config: Consumer configuration
        """
        self.config = config
        self._consumer = None
        self._started = False
        self._running = False
        self._handlers: Dict[str, AsyncMessageHandler] = {}
        self._subscribed_topics: Set[str] = set()

        # Metrics
        self._messages_received = 0
        self._messages_processed = 0
        self._processing_errors = 0
        self._dlq_messages = 0
        self._last_error: Optional[str] = None

        # Retry tracking
        self._retry_counts: Dict[str, int] = {}

    def register_handler(
        self,
        event_type: str,
        handler: AsyncMessageHandler
    ) -> None:
        """
        Register a message handler for a specific event type.

        Args:
            event_type: Event type to handle
            handler: Async handler function
        """
        self._handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")

    def subscribe(self, topics: List[str]) -> None:
        """
        Subscribe to topics.

        Args:
            topics: List of topic names or patterns
        """
        self._subscribed_topics.update(topics)
        logger.info(f"Subscribed to topics: {topics}")

    async def start(self) -> None:
        """Start the Kafka consumer."""
        if self._started:
            logger.warning("ThermalIQ Kafka consumer already started")
            return

        logger.info(
            f"Starting ThermalIQ Kafka consumer "
            f"(group: {self.config.group_id})"
        )

        try:
            # In production, initialize aiokafka consumer:
            # from aiokafka import AIOKafkaConsumer
            # self._consumer = AIOKafkaConsumer(
            #     *self._subscribed_topics,
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     group_id=self.config.group_id,
            #     security_protocol=self.config.security_protocol,
            #     sasl_mechanism=self.config.sasl_mechanism,
            #     sasl_plain_username=self.config.sasl_username,
            #     sasl_plain_password=self.config.sasl_password,
            #     auto_offset_reset=self.config.auto_offset_reset,
            #     enable_auto_commit=self.config.enable_auto_commit,
            #     max_poll_records=self.config.max_poll_records,
            # )
            # await self._consumer.start()

            # Subscribe to default topics if none specified
            if not self._subscribed_topics:
                self._subscribed_topics = {
                    self.TOPIC_ANALYSIS_REQUESTS,
                    self.TOPIC_SENSOR_DATA,
                    self.TOPIC_COMMANDS,
                }

            self._started = True
            logger.info("ThermalIQ Kafka consumer started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        if not self._started:
            return

        logger.info("Stopping ThermalIQ Kafka consumer")
        self._running = False

        try:
            if self._consumer:
                # await self._consumer.stop()
                pass

            self._started = False
            logger.info(
                f"Consumer stopped. Received: {self._messages_received}, "
                f"Processed: {self._messages_processed}, "
                f"Errors: {self._processing_errors}, "
                f"DLQ: {self._dlq_messages}"
            )

        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")

    async def consume(self) -> None:
        """
        Main consumption loop.

        Continuously polls for messages and dispatches to handlers.
        """
        if not self._started:
            raise RuntimeError("Consumer not started. Call start() first.")

        self._running = True
        logger.info("Starting message consumption loop")

        try:
            while self._running:
                # In production:
                # async for message in self._consumer:
                #     await self._process_message(message)

                # Mock implementation - poll for messages
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Consumption loop cancelled")
        except Exception as e:
            logger.error(f"Error in consumption loop: {e}")
            self._last_error = str(e)
            raise

    async def consume_analysis_requests(self) -> None:
        """
        Consume analysis request messages.

        Specifically handles thermaliq.analysis.requests topic.
        """
        if not self._started:
            raise RuntimeError("Consumer not started")

        logger.info("Starting analysis request consumption")

        try:
            while self._running:
                # In production:
                # async for message in self._consumer:
                #     if message.topic == self.TOPIC_ANALYSIS_REQUESTS:
                #         await self._handle_analysis_request(message)

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Analysis request consumption cancelled")
        except Exception as e:
            logger.error(f"Error consuming analysis requests: {e}")
            raise

    async def consume_sensor_data(self) -> None:
        """
        Consume sensor data messages.

        Handles real-time sensor data from thermaliq.sensor.data topic.
        """
        if not self._started:
            raise RuntimeError("Consumer not started")

        logger.info("Starting sensor data consumption")

        try:
            while self._running:
                # In production:
                # async for message in self._consumer:
                #     if message.topic == self.TOPIC_SENSOR_DATA:
                #         await self._handle_sensor_data(message)

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Sensor data consumption cancelled")
        except Exception as e:
            logger.error(f"Error consuming sensor data: {e}")
            raise

    async def _process_message(self, message: Any) -> None:
        """
        Process a single Kafka message.

        Args:
            message: Kafka message object
        """
        self._messages_received += 1
        message_id = None

        try:
            # Deserialize message
            value = json.loads(message.value.decode("utf-8"))
            headers = {
                k: v.decode("utf-8") for k, v in (message.headers or [])
            }

            message_id = value.get("header", {}).get("message_id", "unknown")
            event_type = headers.get("event_type", value.get("event_type", "unknown"))

            logger.debug(
                f"Processing message {message_id} of type {event_type} "
                f"from {message.topic}"
            )

            # Find and invoke handler
            handler = self._handlers.get(event_type)
            if handler:
                await handler(value, headers)
                self._messages_processed += 1

                # Clear retry count on success
                if message_id in self._retry_counts:
                    del self._retry_counts[message_id]

                logger.debug(f"Successfully processed message {message_id}")
            else:
                logger.warning(
                    f"No handler registered for event type: {event_type}"
                )

        except json.JSONDecodeError as e:
            self._processing_errors += 1
            logger.error(f"Failed to decode message: {e}")
            await self._send_to_dlq(message, str(e))

        except Exception as e:
            self._processing_errors += 1
            logger.error(f"Error processing message {message_id}: {e}")

            # Handle retry logic
            retry_count = self._retry_counts.get(message_id, 0) + 1
            self._retry_counts[message_id] = retry_count

            if retry_count >= self.config.max_retries:
                logger.error(
                    f"Message {message_id} exceeded max retries ({self.config.max_retries}), "
                    f"sending to DLQ"
                )
                await self._send_to_dlq(message, str(e))
                del self._retry_counts[message_id]
            else:
                logger.warning(
                    f"Message {message_id} will be retried (attempt {retry_count})"
                )

    async def _handle_analysis_request(self, message: Any) -> None:
        """
        Handle analysis request message.

        Args:
            message: Kafka message
        """
        value = json.loads(message.value.decode("utf-8"))
        request_id = value.get("request_id")

        logger.info(f"Received analysis request: {request_id}")

        # Dispatch to registered handler
        handler = self._handlers.get(EventType.ANALYSIS_REQUESTED.value)
        if handler:
            await handler(value, {})
        else:
            logger.warning("No handler for analysis requests")

    async def _handle_sensor_data(self, message: Any) -> None:
        """
        Handle sensor data message.

        Args:
            message: Kafka message
        """
        value = json.loads(message.value.decode("utf-8"))
        sensor_id = value.get("sensor_id")

        logger.debug(f"Received sensor data from: {sensor_id}")

        # Dispatch to registered handler
        handler = self._handlers.get(EventType.SENSOR_DATA.value)
        if handler:
            await handler(value, {})

    async def _send_to_dlq(self, message: Any, error: str) -> None:
        """
        Send failed message to dead letter queue.

        Args:
            message: Original message
            error: Error description
        """
        if not self.config.enable_dlq:
            return

        self._dlq_messages += 1
        dlq_topic = f"{message.topic}{self.config.dlq_topic_suffix}"

        # In production, produce to DLQ topic:
        # dlq_message = {
        #     "original_topic": message.topic,
        #     "original_partition": message.partition,
        #     "original_offset": message.offset,
        #     "original_value": message.value.decode("utf-8"),
        #     "error": error,
        #     "timestamp": datetime.now(timezone.utc).isoformat(),
        #     "retry_count": self._retry_counts.get(message_id, 0),
        # }
        # await self._dlq_producer.send(dlq_topic, dlq_message)

        logger.info(f"Message sent to DLQ: {dlq_topic}")

    async def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Public method to handle a message (for testing/manual processing).

        Args:
            message: Message dictionary
        """
        event_type = message.get("header", {}).get("event_type")
        if not event_type:
            event_type = message.get("event_type")

        handler = self._handlers.get(event_type)
        if handler:
            await handler(message, {})
        else:
            logger.warning(f"No handler for event type: {event_type}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "processing_errors": self._processing_errors,
            "dlq_messages": self._dlq_messages,
            "pending_retries": len(self._retry_counts),
            "error_rate": (
                self._processing_errors / self._messages_received
                if self._messages_received > 0 else 0.0
            ),
            "success_rate": (
                self._messages_processed / self._messages_received
                if self._messages_received > 0 else 1.0
            ),
            "last_error": self._last_error,
            "is_started": self._started,
            "is_running": self._running,
            "subscribed_topics": list(self._subscribed_topics),
            "registered_handlers": list(self._handlers.keys()),
        }

    async def commit(self) -> None:
        """Manually commit offsets."""
        if self._consumer and not self.config.enable_auto_commit:
            # await self._consumer.commit()
            pass
        logger.debug("Offsets committed")

    async def seek_to_beginning(self, topic: Optional[str] = None) -> None:
        """
        Seek to beginning of topic(s).

        Args:
            topic: Specific topic or all subscribed topics
        """
        if self._consumer:
            # partitions = self._consumer.assignment()
            # if topic:
            #     partitions = [p for p in partitions if p.topic == topic]
            # await self._consumer.seek_to_beginning(*partitions)
            pass
        logger.info(f"Seeked to beginning of {topic or 'all topics'}")

    async def seek_to_end(self, topic: Optional[str] = None) -> None:
        """
        Seek to end of topic(s).

        Args:
            topic: Specific topic or all subscribed topics
        """
        if self._consumer:
            # partitions = self._consumer.assignment()
            # if topic:
            #     partitions = [p for p in partitions if p.topic == topic]
            # await self._consumer.seek_to_end(*partitions)
            pass
        logger.info(f"Seeked to end of {topic or 'all topics'}")


# =============================================================================
# Consumer Group Manager
# =============================================================================

class ConsumerGroupManager:
    """
    Manager for multiple consumer instances in a group.

    Provides:
    - Parallel consumption across partitions
    - Graceful shutdown
    - Health monitoring
    """

    def __init__(
        self,
        config: KafkaConsumerConfig,
        num_consumers: int = 1
    ):
        """
        Initialize consumer group manager.

        Args:
            config: Consumer configuration
            num_consumers: Number of consumer instances
        """
        self.config = config
        self.num_consumers = num_consumers
        self._consumers: List[ThermalIQKafkaConsumer] = []
        self._tasks: List[asyncio.Task] = []

    async def start(self, topics: List[str]) -> None:
        """
        Start all consumers in the group.

        Args:
            topics: Topics to subscribe to
        """
        logger.info(
            f"Starting consumer group with {self.num_consumers} consumers"
        )

        for i in range(self.num_consumers):
            consumer = ThermalIQKafkaConsumer(self.config)
            consumer.subscribe(topics)
            await consumer.start()
            self._consumers.append(consumer)

            # Start consumption task
            task = asyncio.create_task(consumer.consume())
            self._tasks.append(task)

        logger.info(f"Started {len(self._consumers)} consumers")

    async def stop(self) -> None:
        """Stop all consumers."""
        logger.info("Stopping consumer group")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop all consumers
        for consumer in self._consumers:
            await consumer.stop()

        self._consumers.clear()
        self._tasks.clear()

        logger.info("Consumer group stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all consumers."""
        total_received = 0
        total_processed = 0
        total_errors = 0

        for consumer in self._consumers:
            metrics = consumer.get_metrics()
            total_received += metrics["messages_received"]
            total_processed += metrics["messages_processed"]
            total_errors += metrics["processing_errors"]

        return {
            "num_consumers": len(self._consumers),
            "total_messages_received": total_received,
            "total_messages_processed": total_processed,
            "total_processing_errors": total_errors,
            "overall_success_rate": (
                total_processed / total_received
                if total_received > 0 else 1.0
            ),
        }
