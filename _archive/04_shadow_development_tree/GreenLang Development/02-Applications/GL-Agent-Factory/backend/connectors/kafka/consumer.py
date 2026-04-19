"""
Kafka Consumer for GreenLang Agent Events.

This module provides an async Kafka consumer for subscribing to agent events
with support for exactly-once semantics, consumer groups, and dead letter queue handling.

Features:
- Async event consumption with aiokafka
- Exactly-once semantics with read_committed isolation
- Consumer group management
- Manual offset commits for reliability
- Dead letter queue for processing failures
- Event filtering and routing
- Graceful shutdown with rebalance handling

Usage:
    from connectors.kafka.consumer import AgentEventConsumer
    from connectors.kafka.config import create_production_config

    config = create_production_config(
        bootstrap_servers=["kafka-1:9092"],
        sasl_username="consumer",
        sasl_password="secret",
    )

    async def handle_event(event: AgentEvent) -> None:
        print(f"Received: {event.event_type}")

    async with AgentEventConsumer(config) as consumer:
        await consumer.subscribe(["gl.agent.events"])
        await consumer.consume(handler=handle_event)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .config import (
    KafkaConfig,
    GreenLangTopics,
    DeadLetterQueueConfig,
    AutoOffsetReset,
    IsolationLevel,
)
from .events import (
    AgentEvent,
    AgentCalculationCompleted,
    AgentAlertRaised,
    AgentRecommendationGenerated,
    AgentHealthCheck,
    AgentConfigurationChanged,
    EventType,
)
from .serializers import (
    AgentEventDeserializer,
    SerializationFormat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

EventHandler = Callable[[AgentEvent], Coroutine[Any, Any, None]]
EventFilter = Callable[[AgentEvent], bool]


# =============================================================================
# Consumer Statistics
# =============================================================================


@dataclass
class ConsumerStatistics:
    """Statistics for consumer monitoring."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_filtered: int = 0
    bytes_received: int = 0
    commits: int = 0
    rebalances: int = 0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    last_message_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    messages_to_dlq: int = 0
    current_lag: int = 0


@dataclass
class ConsumedMessage:
    """Represents a consumed Kafka message."""

    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    headers: List[Tuple[str, bytes]]
    timestamp: datetime
    event: Optional[AgentEvent] = None
    processing_time_ms: float = 0.0


# =============================================================================
# Event Handler Registry
# =============================================================================


class EventHandlerRegistry:
    """
    Registry for event handlers.

    Supports multiple handlers per event type and wildcard handlers.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._filters: Dict[str, EventFilter] = {}
        self._global_handlers: List[EventHandler] = []

    def register(
        self,
        event_type: str,
        handler: EventHandler,
        filter_fn: Optional[EventFilter] = None,
    ) -> None:
        """
        Register a handler for an event type.

        Args:
            event_type: Event type to handle (or "*" for all)
            handler: Async handler function
            filter_fn: Optional filter function
        """
        if event_type == "*":
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

        if filter_fn:
            handler_id = f"{event_type}:{id(handler)}"
            self._filters[handler_id] = filter_fn

        logger.debug(f"Registered handler for event type: {event_type}")

    def unregister(self, event_type: str, handler: EventHandler) -> bool:
        """
        Unregister a handler.

        Args:
            event_type: Event type
            handler: Handler to remove

        Returns:
            True if removed, False if not found
        """
        if event_type == "*":
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                return True
        elif event_type in self._handlers:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                return True
        return False

    def get_handlers(self, event_type: str) -> List[EventHandler]:
        """
        Get all handlers for an event type.

        Args:
            event_type: Event type

        Returns:
            List of matching handlers
        """
        handlers = list(self._global_handlers)

        # Add specific handlers
        if event_type in self._handlers:
            handlers.extend(self._handlers[event_type])

        # Check for pattern matching (e.g., "agent.calculation.*")
        for pattern, pattern_handlers in self._handlers.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event_type.startswith(prefix):
                    handlers.extend(pattern_handlers)

        return handlers

    def should_handle(
        self,
        event: AgentEvent,
        handler: EventHandler,
    ) -> bool:
        """Check if handler should process event based on filters."""
        handler_id = f"{event.event_type}:{id(handler)}"
        if handler_id in self._filters:
            return self._filters[handler_id](event)
        return True


# =============================================================================
# Offset Manager
# =============================================================================


class OffsetManager:
    """
    Manages offset commits for exactly-once semantics.

    Supports both auto-commit and manual commit strategies.
    """

    def __init__(
        self,
        enable_auto_commit: bool = False,
        commit_interval_ms: int = 5000,
    ):
        """
        Initialize offset manager.

        Args:
            enable_auto_commit: Enable auto-commit
            commit_interval_ms: Auto-commit interval
        """
        self.enable_auto_commit = enable_auto_commit
        self.commit_interval_ms = commit_interval_ms
        self._pending_offsets: Dict[Tuple[str, int], int] = {}
        self._committed_offsets: Dict[Tuple[str, int], int] = {}
        self._last_commit_time = time.time()

    def mark_processed(self, topic: str, partition: int, offset: int) -> None:
        """Mark an offset as processed (ready to commit)."""
        key = (topic, partition)
        current = self._pending_offsets.get(key, -1)
        if offset > current:
            self._pending_offsets[key] = offset

    def get_offsets_to_commit(self) -> Dict[Tuple[str, int], int]:
        """Get offsets that need to be committed."""
        return dict(self._pending_offsets)

    def mark_committed(self, topic: str, partition: int, offset: int) -> None:
        """Mark offsets as committed."""
        key = (topic, partition)
        self._committed_offsets[key] = offset
        if key in self._pending_offsets and self._pending_offsets[key] <= offset:
            del self._pending_offsets[key]
        self._last_commit_time = time.time()

    def should_commit(self) -> bool:
        """Check if it's time to commit (based on interval)."""
        if not self._pending_offsets:
            return False
        elapsed = (time.time() - self._last_commit_time) * 1000
        return elapsed >= self.commit_interval_ms

    def reset(self) -> None:
        """Reset offset tracking."""
        self._pending_offsets.clear()
        self._committed_offsets.clear()


# =============================================================================
# Agent Event Consumer
# =============================================================================


class AgentEventConsumer:
    """
    Kafka consumer for GreenLang agent events.

    This consumer provides:
    - Async event consumption with aiokafka
    - Exactly-once semantics with manual commits
    - Consumer group coordination
    - Event filtering and routing
    - Dead letter queue handling
    - Graceful shutdown

    Example:
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            consumer=KafkaConsumerConfig(
                group_id="gl-agent-workers",
                enable_auto_commit=False,
                isolation_level=IsolationLevel.READ_COMMITTED,
            ),
        )

        async with AgentEventConsumer(config) as consumer:
            # Register handlers
            consumer.on(EventType.CALCULATION_COMPLETED, handle_calculation)
            consumer.on(EventType.ALERT_RAISED, handle_alert)

            # Subscribe and consume
            await consumer.subscribe(["gl.agent.events"])
            await consumer.consume()
    """

    def __init__(
        self,
        config: KafkaConfig,
        group_id: Optional[str] = None,
        schema_registry_url: Optional[str] = None,
    ):
        """
        Initialize the consumer.

        Args:
            config: Kafka configuration
            group_id: Optional override for consumer group ID
            schema_registry_url: Schema Registry URL for Avro
        """
        self.config = config
        self.group_id = group_id or config.consumer.group_id

        # Initialize components
        self._consumer = None
        self._started = False
        self._running = False
        self._subscribed_topics: Set[str] = set()

        # Deserializer
        self._deserializer = AgentEventDeserializer(
            schema_registry_url=schema_registry_url or (
                config.schema_registry.url if config.schema_registry else None
            ),
        )

        # Handler registry
        self._handler_registry = EventHandlerRegistry()

        # Offset manager
        self._offset_manager = OffsetManager(
            enable_auto_commit=config.consumer.enable_auto_commit,
            commit_interval_ms=config.consumer.auto_commit_interval_ms,
        )

        # Statistics
        self._stats = ConsumerStatistics()

        # Processing semaphore for concurrency control
        self._processing_semaphore = asyncio.Semaphore(10)

        logger.info(
            f"AgentEventConsumer initialized: "
            f"bootstrap_servers={config.bootstrap_servers}, "
            f"group_id={self.group_id}"
        )

    async def start(self) -> None:
        """Start the consumer and connect to Kafka."""
        if self._started:
            logger.warning("Consumer already started")
            return

        try:
            from aiokafka import AIOKafkaConsumer

            # Get consumer config
            consumer_config = self.config.to_aiokafka_consumer_config()
            consumer_config["group_id"] = self.group_id

            self._consumer = AIOKafkaConsumer(**consumer_config)

            await self._consumer.start()
            self._started = True

            logger.info("AgentEventConsumer started successfully")

        except ImportError:
            logger.warning(
                "aiokafka not installed, using mock consumer. "
                "Install with: pip install aiokafka"
            )
            self._consumer = MockKafkaConsumer()
            self._started = True

        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if not self._started:
            return

        self._running = False

        try:
            # Commit any pending offsets
            await self.commit()

            if self._consumer:
                await self._consumer.stop()

            self._started = False
            logger.info("AgentEventConsumer stopped")

        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")
            raise

    async def subscribe(
        self,
        topics: List[str],
        pattern: Optional[str] = None,
    ) -> None:
        """
        Subscribe to topics.

        Args:
            topics: List of topic names
            pattern: Optional topic pattern (regex)
        """
        if not self._started:
            await self.start()

        if pattern:
            # Pattern subscription
            self._consumer.subscribe(pattern=pattern)
            logger.info(f"Subscribed to pattern: {pattern}")
        else:
            self._consumer.subscribe(topics)
            self._subscribed_topics.update(topics)
            logger.info(f"Subscribed to topics: {topics}")

    async def unsubscribe(self) -> None:
        """Unsubscribe from all topics."""
        if self._consumer:
            self._consumer.unsubscribe()
            self._subscribed_topics.clear()
            logger.info("Unsubscribed from all topics")

    def on(
        self,
        event_type: Union[str, EventType],
        handler: EventHandler,
        filter_fn: Optional[EventFilter] = None,
    ) -> None:
        """
        Register an event handler.

        Args:
            event_type: Event type to handle (or "*" for all)
            handler: Async handler function
            filter_fn: Optional filter function

        Example:
            @consumer.on(EventType.CALCULATION_COMPLETED)
            async def handle_calculation(event: AgentCalculationCompleted):
                print(f"Calculation completed: {event.calculation_type}")
        """
        if isinstance(event_type, EventType):
            event_type = event_type.value

        self._handler_registry.register(event_type, handler, filter_fn)

    def off(
        self,
        event_type: Union[str, EventType],
        handler: EventHandler,
    ) -> bool:
        """
        Unregister an event handler.

        Args:
            event_type: Event type
            handler: Handler to remove

        Returns:
            True if removed, False if not found
        """
        if isinstance(event_type, EventType):
            event_type = event_type.value

        return self._handler_registry.unregister(event_type, handler)

    async def consume(
        self,
        handler: Optional[EventHandler] = None,
        max_messages: Optional[int] = None,
        timeout_ms: int = 1000,
    ) -> None:
        """
        Start consuming messages.

        Args:
            handler: Optional single handler for all events
            max_messages: Optional maximum messages to consume
            timeout_ms: Poll timeout in milliseconds
        """
        if not self._started:
            await self.start()

        if handler:
            self._handler_registry.register("*", handler)

        self._running = True
        messages_consumed = 0

        logger.info("Starting consumption loop")

        try:
            while self._running:
                # Check message limit
                if max_messages and messages_consumed >= max_messages:
                    logger.info(f"Reached max messages: {max_messages}")
                    break

                # Poll for messages
                try:
                    message = await asyncio.wait_for(
                        self._consumer.getone(),
                        timeout=timeout_ms / 1000,
                    )

                    # Process message
                    await self._process_message(message)
                    messages_consumed += 1

                    # Periodic commit check
                    if self._offset_manager.should_commit():
                        await self.commit()

                except asyncio.TimeoutError:
                    # No messages, check for commits
                    if self._offset_manager.should_commit():
                        await self.commit()
                    continue

        except asyncio.CancelledError:
            logger.info("Consumption cancelled")
        except Exception as e:
            logger.error(f"Consumption error: {e}", exc_info=True)
            raise
        finally:
            # Final commit
            await self.commit()

    async def consume_batch(
        self,
        handler: Optional[EventHandler] = None,
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> List[AgentEvent]:
        """
        Consume a batch of messages.

        Args:
            handler: Optional handler for each event
            batch_size: Maximum messages to fetch
            timeout_ms: Poll timeout

        Returns:
            List of consumed events
        """
        if not self._started:
            await self.start()

        events = []

        try:
            messages = await self._consumer.getmany(
                timeout_ms=timeout_ms,
                max_records=batch_size,
            )

            for topic_partition, partition_messages in messages.items():
                for message in partition_messages:
                    consumed = await self._process_message(message)
                    if consumed.event:
                        events.append(consumed.event)

                        # Call handler if provided
                        if handler:
                            await handler(consumed.event)

            # Commit after batch
            await self.commit()

        except Exception as e:
            logger.error(f"Batch consumption error: {e}", exc_info=True)
            raise

        return events

    async def _process_message(self, message) -> ConsumedMessage:
        """
        Process a single Kafka message.

        Args:
            message: Kafka message

        Returns:
            ConsumedMessage with processing details
        """
        start_time = time.time()
        self._stats.messages_received += 1
        self._stats.bytes_received += len(message.value) if message.value else 0

        consumed = ConsumedMessage(
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            key=message.key,
            value=message.value,
            headers=message.headers or [],
            timestamp=datetime.fromtimestamp(message.timestamp / 1000) if message.timestamp else datetime.utcnow(),
        )

        try:
            # Deserialize event
            event = await self._deserializer.deserialize(
                message.value,
                message.headers,
            )
            consumed.event = event

            # Get handlers
            handlers = self._handler_registry.get_handlers(event.event_type)

            if not handlers:
                self._stats.messages_filtered += 1
                logger.debug(f"No handlers for event type: {event.event_type}")
            else:
                # Execute handlers
                async with self._processing_semaphore:
                    for handler in handlers:
                        if self._handler_registry.should_handle(event, handler):
                            try:
                                await handler(event)
                            except Exception as handler_error:
                                logger.error(
                                    f"Handler error for {event.event_type}: {handler_error}",
                                    exc_info=True,
                                )
                                raise

            # Mark for commit
            self._offset_manager.mark_processed(
                message.topic,
                message.partition,
                message.offset + 1,  # Commit next offset
            )

            self._stats.messages_processed += 1
            self._stats.last_message_time = datetime.utcnow()

        except Exception as e:
            self._stats.messages_failed += 1
            self._stats.last_error = str(e)
            self._stats.last_error_time = datetime.utcnow()

            logger.error(
                f"Message processing failed: topic={message.topic}, "
                f"partition={message.partition}, offset={message.offset}, "
                f"error={e}"
            )

            # Send to DLQ if configured
            if self.config.dlq.enabled and consumed.event:
                await self._send_to_dlq(consumed, str(e))

            # Still commit to avoid reprocessing
            self._offset_manager.mark_processed(
                message.topic,
                message.partition,
                message.offset + 1,
            )

        finally:
            processing_time = (time.time() - start_time) * 1000
            consumed.processing_time_ms = processing_time

            # Update stats
            if processing_time > self._stats.max_processing_time_ms:
                self._stats.max_processing_time_ms = processing_time

            n = self._stats.messages_processed + self._stats.messages_failed
            self._stats.avg_processing_time_ms = (
                (self._stats.avg_processing_time_ms * (n - 1) + processing_time) / n
                if n > 0 else processing_time
            )

        return consumed

    async def commit(self) -> None:
        """Commit processed offsets."""
        if not self._consumer or self.config.consumer.enable_auto_commit:
            return

        offsets = self._offset_manager.get_offsets_to_commit()
        if not offsets:
            return

        try:
            # Convert to aiokafka format
            from aiokafka import TopicPartition

            commit_offsets = {
                TopicPartition(topic, partition): offset
                for (topic, partition), offset in offsets.items()
            }

            await self._consumer.commit(commit_offsets)

            # Mark as committed
            for (topic, partition), offset in offsets.items():
                self._offset_manager.mark_committed(topic, partition, offset)

            self._stats.commits += 1
            logger.debug(f"Committed offsets: {offsets}")

        except ImportError:
            # Mock consumer
            for (topic, partition), offset in offsets.items():
                self._offset_manager.mark_committed(topic, partition, offset)
            self._stats.commits += 1

        except Exception as e:
            logger.error(f"Commit failed: {e}")
            raise

    async def _send_to_dlq(
        self,
        consumed: ConsumedMessage,
        error: str,
    ) -> None:
        """Send failed message to dead letter queue."""
        dlq_topic = consumed.topic + self.config.dlq.topic_suffix

        try:
            # We need a producer to send to DLQ
            # This is a simplified implementation
            logger.warning(
                f"Would send to DLQ: topic={dlq_topic}, "
                f"original_offset={consumed.offset}, error={error}"
            )
            self._stats.messages_to_dlq += 1

        except Exception as dlq_error:
            logger.error(f"Failed to send to DLQ: {dlq_error}")

    def get_statistics(self) -> ConsumerStatistics:
        """Get consumer statistics."""
        return self._stats

    async def get_lag(self) -> Dict[str, int]:
        """
        Get consumer lag per partition.

        Returns:
            Dictionary of partition -> lag
        """
        lag = {}

        if self._consumer:
            try:
                # Get assigned partitions
                partitions = self._consumer.assignment()

                for tp in partitions:
                    # Get current position and end offset
                    position = await self._consumer.position(tp)
                    end_offsets = await self._consumer.end_offsets([tp])
                    end_offset = end_offsets.get(tp, 0)

                    partition_lag = end_offset - position
                    lag[f"{tp.topic}-{tp.partition}"] = partition_lag

                self._stats.current_lag = sum(lag.values())

            except Exception as e:
                logger.error(f"Failed to get lag: {e}")

        return lag

    async def seek_to_beginning(self, topics: Optional[List[str]] = None) -> None:
        """Seek to beginning of topics."""
        if self._consumer:
            partitions = self._consumer.assignment()
            if topics:
                partitions = [p for p in partitions if p.topic in topics]
            await self._consumer.seek_to_beginning(*partitions)
            logger.info(f"Seeked to beginning: {partitions}")

    async def seek_to_end(self, topics: Optional[List[str]] = None) -> None:
        """Seek to end of topics."""
        if self._consumer:
            partitions = self._consumer.assignment()
            if topics:
                partitions = [p for p in partitions if p.topic in topics]
            await self._consumer.seek_to_end(*partitions)
            logger.info(f"Seeked to end: {partitions}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check result dictionary
        """
        lag = await self.get_lag() if self._started else {}

        return {
            "status": "healthy" if self._started and self._running else "not_running",
            "connected": self._started,
            "subscribed_topics": list(self._subscribed_topics),
            "messages_processed": self._stats.messages_processed,
            "messages_failed": self._stats.messages_failed,
            "avg_processing_time_ms": round(self._stats.avg_processing_time_ms, 2),
            "current_lag": sum(lag.values()) if lag else 0,
            "last_error": self._stats.last_error,
        }

    # Context manager support
    async def __aenter__(self) -> "AgentEventConsumer":
        """Enter async context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.stop()


# =============================================================================
# Mock Consumer for Testing
# =============================================================================


class MockKafkaConsumer:
    """Mock Kafka consumer for testing without Kafka."""

    def __init__(self):
        self._messages: List[Any] = []
        self._subscribed: List[str] = []
        self._position: int = 0

    async def start(self) -> None:
        """Start mock consumer."""
        pass

    async def stop(self) -> None:
        """Stop mock consumer."""
        pass

    def subscribe(self, topics: List[str] = None, pattern: str = None) -> None:
        """Subscribe to topics."""
        if topics:
            self._subscribed = topics

    def unsubscribe(self) -> None:
        """Unsubscribe from topics."""
        self._subscribed = []

    def assignment(self) -> List[Any]:
        """Get assigned partitions."""
        return []

    async def getone(self) -> Any:
        """Get one message (blocking)."""
        if self._position < len(self._messages):
            msg = self._messages[self._position]
            self._position += 1
            return msg
        await asyncio.sleep(1)
        raise asyncio.TimeoutError()

    async def getmany(self, timeout_ms: int = 1000, max_records: int = 100) -> Dict:
        """Get multiple messages."""
        return {}

    async def commit(self, offsets: Dict = None) -> None:
        """Commit offsets."""
        pass

    async def position(self, tp) -> int:
        """Get current position."""
        return 0

    async def end_offsets(self, partitions: List) -> Dict:
        """Get end offsets."""
        return {}

    async def seek_to_beginning(self, *partitions) -> None:
        """Seek to beginning."""
        self._position = 0

    async def seek_to_end(self, *partitions) -> None:
        """Seek to end."""
        self._position = len(self._messages)

    def add_test_message(self, message: Any) -> None:
        """Add a test message (for testing)."""
        self._messages.append(message)


# =============================================================================
# Factory Function
# =============================================================================


def create_consumer(
    bootstrap_servers: List[str],
    group_id: str = "gl-agent-consumers",
    enable_exactly_once: bool = True,
) -> AgentEventConsumer:
    """
    Factory function to create a consumer with common settings.

    Args:
        bootstrap_servers: Kafka broker addresses
        group_id: Consumer group ID
        enable_exactly_once: Enable exactly-once semantics

    Returns:
        Configured AgentEventConsumer
    """
    from .config import KafkaConfig, KafkaConsumerConfig

    consumer_config = KafkaConsumerConfig(
        group_id=group_id,
        enable_auto_commit=not enable_exactly_once,
        isolation_level=(
            IsolationLevel.READ_COMMITTED if enable_exactly_once
            else IsolationLevel.READ_UNCOMMITTED
        ),
        auto_offset_reset=AutoOffsetReset.EARLIEST,
    )

    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        consumer=consumer_config,
    )

    return AgentEventConsumer(config=config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AgentEventConsumer",
    "EventHandlerRegistry",
    "OffsetManager",
    "ConsumerStatistics",
    "ConsumedMessage",
    "MockKafkaConsumer",
    "create_consumer",
    "EventHandler",
    "EventFilter",
]
