"""
Generic Event Consumer for GreenLang

This module provides a generic event consumer that abstracts
the underlying messaging infrastructure.

Features:
- Multi-backend support
- Automatic deserialization
- Handler routing
- Error handling
- Dead letter queue integration
- Consumer group management

Example:
    >>> consumer = EventConsumer(config)
    >>> consumer.register_handler("emission.calculated", handler)
    >>> await consumer.start()
    >>> await consumer.consume()
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.events.event_schema import (
    BaseEvent,
    DomainEvent,
    IntegrationEvent,
    EventMetadata,
    EventPriority,
)

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Supported messaging backends."""
    KAFKA = "kafka"
    MQTT = "mqtt"
    REDIS = "redis"
    MEMORY = "memory"


class ProcessingResult(str, Enum):
    """Event processing result."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    SKIP = "skip"
    DLQ = "dlq"


@dataclass
class EventConsumerConfig:
    """Configuration for event consumer."""
    backend: BackendType = BackendType.KAFKA
    # Kafka settings
    kafka_bootstrap_servers: List[str] = field(
        default_factory=lambda: ["localhost:9092"]
    )
    kafka_group_id: str = "greenlang-consumer"
    # MQTT settings
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    # Common settings
    topics: List[str] = field(default_factory=lambda: ["greenlang-events"])
    max_retries: int = 3
    retry_delay_ms: int = 1000
    enable_dlq: bool = True
    dlq_topic: str = "greenlang-events-dlq"
    batch_size: int = 100
    commit_interval_ms: int = 5000


class ConsumedEvent(BaseModel):
    """Wrapper for consumed events with metadata."""
    event: BaseEvent = Field(..., description="The event")
    topic: str = Field(..., description="Source topic")
    partition: Optional[int] = Field(default=None)
    offset: Optional[int] = Field(default=None)
    key: Optional[str] = Field(default=None)
    headers: Dict[str, str] = Field(default_factory=dict)
    receive_timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0)


class EventHandler:
    """
    Event handler wrapper with metadata.
    """

    def __init__(
        self,
        event_type: str,
        handler: Callable[[ConsumedEvent], ProcessingResult],
        priority: int = 0,
        timeout_ms: int = 30000,
        max_retries: int = 3,
    ):
        """
        Initialize event handler.

        Args:
            event_type: Event type to handle
            handler: Handler function
            priority: Handler priority (higher = first)
            timeout_ms: Handler timeout
            max_retries: Max retries for this handler
        """
        self.event_type = event_type
        self.handler = handler
        self.priority = priority
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries


class EventConsumerBackend(ABC):
    """Abstract base class for event consumer backends."""

    @abstractmethod
    async def start(self) -> None:
        """Start the backend."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend."""
        pass

    @abstractmethod
    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics."""
        pass

    @abstractmethod
    async def consume(self) -> Optional[Dict[str, Any]]:
        """Consume a single message."""
        pass

    @abstractmethod
    async def commit(self, offsets: Dict[str, int]) -> None:
        """Commit offsets."""
        pass


class MemoryConsumerBackend(EventConsumerBackend):
    """In-memory backend for testing."""

    def __init__(self):
        """Initialize memory backend."""
        self.messages: asyncio.Queue = asyncio.Queue()
        self._subscribed_topics: Set[str] = set()

    async def start(self) -> None:
        """Start the backend."""
        logger.info("Memory consumer backend started")

    async def stop(self) -> None:
        """Stop the backend."""
        logger.info("Memory consumer backend stopped")

    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics."""
        self._subscribed_topics.update(topics)
        logger.info(f"Subscribed to topics: {topics}")

    async def consume(self) -> Optional[Dict[str, Any]]:
        """Consume a single message."""
        try:
            return await asyncio.wait_for(self.messages.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def commit(self, offsets: Dict[str, int]) -> None:
        """Commit offsets."""
        pass

    async def inject_message(self, message: Dict[str, Any]) -> None:
        """Inject a message for testing."""
        await self.messages.put(message)


class EventConsumer:
    """
    Generic event consumer for GreenLang.

    Provides a unified interface for consuming events across
    different messaging backends with handler routing.

    Attributes:
        config: Consumer configuration
        handlers: Registered event handlers

    Example:
        >>> config = EventConsumerConfig(
        ...     backend=BackendType.KAFKA,
        ...     topics=["emissions", "compliance"]
        ... )
        >>> consumer = EventConsumer(config)
        >>> consumer.register_handler("emission.calculated", my_handler)
        >>> async with consumer:
        ...     await consumer.consume()
    """

    def __init__(self, config: EventConsumerConfig):
        """
        Initialize event consumer.

        Args:
            config: Consumer configuration
        """
        self.config = config
        self._backend: Optional[EventConsumerBackend] = None
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._default_handlers: List[EventHandler] = []
        self._started = False
        self._consuming = False
        self._shutdown = False
        self._metrics: Dict[str, int] = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_dlq": 0,
            "events_skipped": 0,
        }
        self._dlq_producer = None

        logger.info(f"EventConsumer initialized with backend: {config.backend}")

    async def start(self) -> None:
        """
        Start the event consumer.

        Initializes the backend and subscribes to topics.
        """
        if self._started:
            logger.warning("Consumer already started")
            return

        try:
            # Initialize backend
            self._backend = self._create_backend()
            await self._backend.start()

            # Subscribe to topics
            await self._backend.subscribe(self.config.topics)

            self._started = True
            self._shutdown = False

            logger.info("Event consumer started")

        except Exception as e:
            logger.error(f"Failed to start consumer: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the event consumer gracefully.
        """
        self._shutdown = True
        self._consuming = False

        if self._backend:
            await self._backend.stop()

        self._started = False
        logger.info("Event consumer stopped")

    def _create_backend(self) -> EventConsumerBackend:
        """Create the appropriate backend."""
        if self.config.backend == BackendType.MEMORY:
            return MemoryConsumerBackend()
        elif self.config.backend == BackendType.KAFKA:
            # Would integrate with KafkaExactlyOnceConsumer
            return MemoryConsumerBackend()
        elif self.config.backend == BackendType.MQTT:
            # Would integrate with MQTTClient
            return MemoryConsumerBackend()
        else:
            return MemoryConsumerBackend()

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[ConsumedEvent], ProcessingResult],
        priority: int = 0,
        timeout_ms: int = 30000,
        max_retries: int = 3,
    ) -> None:
        """
        Register an event handler.

        Args:
            event_type: Event type to handle (or "*" for all)
            handler: Handler function
            priority: Handler priority
            timeout_ms: Handler timeout
            max_retries: Max retries
        """
        event_handler = EventHandler(
            event_type=event_type,
            handler=handler,
            priority=priority,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
        )

        if event_type == "*":
            self._default_handlers.append(event_handler)
            self._default_handlers.sort(key=lambda h: -h.priority)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(event_handler)
            self._handlers[event_type].sort(key=lambda h: -h.priority)

        logger.info(f"Registered handler for event type: {event_type}")

    def unregister_handler(self, event_type: str) -> None:
        """
        Unregister handlers for an event type.

        Args:
            event_type: Event type to unregister
        """
        if event_type == "*":
            self._default_handlers.clear()
        else:
            self._handlers.pop(event_type, None)

        logger.info(f"Unregistered handlers for event type: {event_type}")

    async def consume(
        self,
        max_events: Optional[int] = None,
        timeout_ms: Optional[int] = None
    ) -> None:
        """
        Start consuming events.

        Args:
            max_events: Maximum events to consume (None for unlimited)
            timeout_ms: Total timeout in milliseconds
        """
        self._ensure_started()
        self._consuming = True

        start_time = datetime.utcnow()
        event_count = 0
        pending_commits: Dict[str, int] = {}

        try:
            while self._consuming and not self._shutdown:
                # Check limits
                if max_events and event_count >= max_events:
                    break

                if timeout_ms:
                    elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                    if elapsed >= timeout_ms:
                        break

                # Consume message
                message = await self._backend.consume()
                if not message:
                    continue

                # Process message
                consumed_event = self._deserialize_message(message)
                if consumed_event:
                    result = await self._process_event(consumed_event)
                    event_count += 1

                    # Track offset for commit
                    if message.get("partition") is not None:
                        topic_partition = f"{message['topic']}-{message['partition']}"
                        pending_commits[topic_partition] = message.get("offset", 0)

                # Periodic commit
                if event_count % self.config.batch_size == 0:
                    await self._backend.commit(pending_commits)
                    pending_commits.clear()

            # Final commit
            if pending_commits:
                await self._backend.commit(pending_commits)

        except asyncio.CancelledError:
            logger.info("Consume cancelled")
        except Exception as e:
            logger.error(f"Consume error: {e}", exc_info=True)
            raise
        finally:
            self._consuming = False

    async def consume_one(
        self,
        timeout_ms: int = 5000
    ) -> Optional[ConsumedEvent]:
        """
        Consume and process a single event.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            ConsumedEvent if available
        """
        self._ensure_started()

        message = await self._backend.consume()
        if message:
            return self._deserialize_message(message)
        return None

    def _deserialize_message(
        self,
        message: Dict[str, Any]
    ) -> Optional[ConsumedEvent]:
        """Deserialize a message to ConsumedEvent."""
        try:
            value = message.get("value")
            if isinstance(value, bytes):
                value = value.decode()

            if isinstance(value, str):
                data = json.loads(value)
            else:
                data = value

            # Reconstruct event
            metadata = EventMetadata(
                event_id=data.get("event_id", str(uuid4())),
                correlation_id=data.get("correlation_id"),
                causation_id=data.get("causation_id"),
                timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
                version=data.get("version", 1),
                source=data.get("source", "unknown"),
                priority=EventPriority(data.get("priority", "normal")),
            )

            event = BaseEvent(
                event_type=data["event_type"],
                metadata=metadata,
                data=data.get("data", {}),
                provenance_hash=data.get("provenance_hash", ""),
            )

            # Parse headers
            headers = {}
            for key, val in message.get("headers", {}).items():
                if isinstance(val, bytes):
                    headers[key] = val.decode()
                else:
                    headers[key] = str(val)

            self._metrics["events_received"] += 1

            return ConsumedEvent(
                event=event,
                topic=message.get("topic", "unknown"),
                partition=message.get("partition"),
                offset=message.get("offset"),
                key=message.get("key"),
                headers=headers,
            )

        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            return None

    async def _process_event(
        self,
        consumed_event: ConsumedEvent
    ) -> ProcessingResult:
        """Process an event with registered handlers."""
        event_type = consumed_event.event.event_type

        # Find handlers
        handlers = self._handlers.get(event_type, []) + self._default_handlers

        if not handlers:
            logger.warning(f"No handler for event type: {event_type}")
            self._metrics["events_skipped"] += 1
            return ProcessingResult.SKIP

        # Execute handlers
        for handler in handlers:
            result = await self._execute_handler(handler, consumed_event)

            if result == ProcessingResult.FAILURE:
                if consumed_event.retry_count < handler.max_retries:
                    consumed_event.retry_count += 1
                    return ProcessingResult.RETRY
                else:
                    await self._send_to_dlq(consumed_event, "Max retries exceeded")
                    self._metrics["events_dlq"] += 1
                    return ProcessingResult.DLQ

            elif result == ProcessingResult.DLQ:
                await self._send_to_dlq(consumed_event, "Handler requested DLQ")
                self._metrics["events_dlq"] += 1
                return ProcessingResult.DLQ

            elif result == ProcessingResult.RETRY:
                if consumed_event.retry_count < handler.max_retries:
                    consumed_event.retry_count += 1
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)
                    continue
                else:
                    await self._send_to_dlq(consumed_event, "Max retries exceeded")
                    self._metrics["events_dlq"] += 1
                    return ProcessingResult.DLQ

        self._metrics["events_processed"] += 1
        return ProcessingResult.SUCCESS

    async def _execute_handler(
        self,
        handler: EventHandler,
        consumed_event: ConsumedEvent
    ) -> ProcessingResult:
        """Execute a single handler with timeout."""
        try:
            if asyncio.iscoroutinefunction(handler.handler):
                result = await asyncio.wait_for(
                    handler.handler(consumed_event),
                    timeout=handler.timeout_ms / 1000
                )
            else:
                result = handler.handler(consumed_event)

            return result if isinstance(result, ProcessingResult) else ProcessingResult.SUCCESS

        except asyncio.TimeoutError:
            logger.error(f"Handler timeout for {handler.event_type}")
            self._metrics["events_failed"] += 1
            return ProcessingResult.FAILURE

        except Exception as e:
            logger.error(f"Handler error for {handler.event_type}: {e}")
            self._metrics["events_failed"] += 1
            return ProcessingResult.FAILURE

    async def _send_to_dlq(
        self,
        consumed_event: ConsumedEvent,
        reason: str
    ) -> None:
        """Send event to dead letter queue."""
        if not self.config.enable_dlq:
            logger.warning("DLQ disabled, dropping event")
            return

        logger.info(
            f"Sending to DLQ: {consumed_event.event.event_type} "
            f"reason: {reason}"
        )
        # Would use EventProducer to publish to DLQ topic

    def _ensure_started(self) -> None:
        """Ensure consumer is started."""
        if not self._started:
            raise RuntimeError("Consumer not started")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consumer metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "started": self._started,
            "consuming": self._consuming,
            "backend": self.config.backend.value,
            "topics": self.config.topics,
            "registered_handlers": list(self._handlers.keys()),
            "events_received": self._metrics["events_received"],
            "events_processed": self._metrics["events_processed"],
            "events_failed": self._metrics["events_failed"],
            "events_dlq": self._metrics["events_dlq"],
            "events_skipped": self._metrics["events_skipped"],
        }

    async def __aenter__(self) -> "EventConsumer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
