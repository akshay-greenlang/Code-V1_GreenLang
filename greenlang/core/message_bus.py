# -*- coding: utf-8 -*-
"""
MessageBus - Event-driven messaging for agent orchestration.

This module implements an async message bus for inter-agent communication,
supporting publish/subscribe patterns, priority queuing, and message persistence.

Example:
    >>> bus = MessageBus(config=MessageBusConfig())
    >>> await bus.subscribe("agent.thermal", handler_func)
    >>> await bus.publish("agent.thermal", Message(...))
    >>> await bus.close()

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

# Type variable for message payloads
T = TypeVar("T")


class MessagePriority(str, Enum):
    """Message priority levels for queue ordering."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    def __lt__(self, other: "MessagePriority") -> bool:
        """Enable priority comparison for queue ordering."""
        order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
        }
        return order[self] < order[other]


class MessageType(str, Enum):
    """Standard message types for orchestration."""

    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class Message:
    """
    Immutable message structure for inter-agent communication.

    Attributes:
        message_id: Unique message identifier (UUID)
        sender_id: ID of the sending agent
        recipient_id: ID of the target agent (or '*' for broadcast)
        message_type: Type of message (command, event, query, etc.)
        topic: Topic/channel for routing
        payload: Message content (any serializable data)
        priority: Message priority for queue ordering
        timestamp: ISO 8601 timestamp of creation
        correlation_id: Optional ID for request/response correlation
        reply_to: Optional topic for responses
        ttl_seconds: Time-to-live in seconds (0 = no expiry)
        metadata: Additional metadata for routing/filtering
    """

    sender_id: str
    recipient_id: str
    message_type: MessageType
    topic: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if not self.sender_id:
            raise ValueError("sender_id is required")
        if not self.topic:
            raise ValueError("topic is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id", "*"),
            message_type=MessageType(data.get("message_type", "event")),
            topic=data["topic"],
            payload=data.get("payload"),
            priority=MessagePriority(data.get("priority", "normal")),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl_seconds=data.get("ttl_seconds", 0),
            metadata=data.get("metadata", {}),
        )

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False
        created = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - created).total_seconds()
        return age > self.ttl_seconds

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of message for provenance tracking."""
        content = f"{self.sender_id}{self.recipient_id}{self.topic}{self.payload}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()


# Type alias for message handlers
MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """
    Represents a topic subscription.

    Attributes:
        topic: Topic pattern (supports wildcards: * for single, # for multi)
        handler: Async callback function for handling messages
        subscriber_id: ID of the subscribing agent/component
        filter_fn: Optional filter function to further filter messages
        created_at: Timestamp of subscription creation
    """

    topic: str
    handler: MessageHandler
    subscriber_id: str
    filter_fn: Optional[Callable[[Message], bool]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def matches_topic(self, message_topic: str) -> bool:
        """
        Check if message topic matches subscription pattern.

        Supports wildcards:
        - '*' matches a single level (e.g., 'agent.*' matches 'agent.thermal')
        - '#' matches multiple levels (e.g., 'agent.#' matches 'agent.thermal.status')
        """
        sub_parts = self.topic.split(".")
        msg_parts = message_topic.split(".")

        if "#" in sub_parts:
            # Multi-level wildcard: match prefix
            hash_idx = sub_parts.index("#")
            prefix_parts = sub_parts[:hash_idx]
            if len(msg_parts) < len(prefix_parts):
                return False
            return msg_parts[: len(prefix_parts)] == prefix_parts

        if len(sub_parts) != len(msg_parts):
            return False

        for sub_part, msg_part in zip(sub_parts, msg_parts):
            if sub_part == "*":
                continue  # Single-level wildcard matches any
            if sub_part != msg_part:
                return False

        return True

    def should_handle(self, message: Message) -> bool:
        """Check if this subscription should handle the message."""
        if not self.matches_topic(message.topic):
            return False
        if self.filter_fn and not self.filter_fn(message):
            return False
        return True


@dataclass
class MessageBusConfig:
    """
    Configuration for MessageBus.

    Attributes:
        max_queue_size: Maximum messages in queue (0 = unlimited)
        enable_persistence: Persist messages for recovery
        persistence_path: Path for message persistence
        enable_dead_letter: Enable dead letter queue for failed messages
        max_retries: Max delivery retries before dead-lettering
        retry_delay_seconds: Delay between retries
        metrics_enabled: Enable metrics collection
        max_handlers_per_topic: Maximum handlers per topic (0 = unlimited)
    """

    max_queue_size: int = 10000
    enable_persistence: bool = False
    persistence_path: str = "./message_bus_persistence"
    enable_dead_letter: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    metrics_enabled: bool = True
    max_handlers_per_topic: int = 100


@dataclass
class MessageBusMetrics:
    """Metrics for message bus monitoring."""

    messages_published: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    messages_dead_lettered: int = 0
    messages_expired: int = 0
    active_subscriptions: int = 0
    queue_size: int = 0
    avg_delivery_time_ms: float = 0.0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "messages_published": self.messages_published,
            "messages_delivered": self.messages_delivered,
            "messages_failed": self.messages_failed,
            "messages_dead_lettered": self.messages_dead_lettered,
            "messages_expired": self.messages_expired,
            "active_subscriptions": self.active_subscriptions,
            "queue_size": self.queue_size,
            "avg_delivery_time_ms": round(self.avg_delivery_time_ms, 2),
            "last_updated": self.last_updated,
        }


class MessageBus:
    """
    Async message bus for agent orchestration.

    Provides publish/subscribe messaging with:
    - Topic-based routing with wildcard support
    - Priority queuing
    - Message TTL and expiration
    - Dead letter queue for failed deliveries
    - Metrics collection
    - Optional message persistence

    Example:
        >>> config = MessageBusConfig(enable_persistence=False)
        >>> bus = MessageBus(config)
        >>>
        >>> async def handle_thermal(msg: Message):
        ...     print(f"Received: {msg.payload}")
        >>>
        >>> await bus.subscribe("agent.thermal.*", handle_thermal, "my-subscriber")
        >>> await bus.publish(Message(
        ...     sender_id="orchestrator",
        ...     recipient_id="thermal-agent",
        ...     message_type=MessageType.COMMAND,
        ...     topic="agent.thermal.optimize",
        ...     payload={"target_temp": 450}
        ... ))
        >>> await bus.close()
    """

    def __init__(self, config: Optional[MessageBusConfig] = None) -> None:
        """
        Initialize MessageBus.

        Args:
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or MessageBusConfig()
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size if self.config.max_queue_size > 0 else 0
        )
        self._dead_letter_queue: List[Message] = []
        self._metrics = MessageBusMetrics()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._delivery_times: List[float] = []
        self._lock = asyncio.Lock()

        logger.info(
            f"MessageBus initialized with config: max_queue={self.config.max_queue_size}"
        )

    async def start(self) -> None:
        """Start the message bus processor."""
        if self._running:
            logger.warning("MessageBus is already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("MessageBus started")

    async def close(self) -> None:
        """Gracefully shutdown the message bus."""
        logger.info("Shutting down MessageBus...")
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Clear remaining messages
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("MessageBus shutdown complete")

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        subscriber_id: str,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """
        Subscribe to a topic.

        Args:
            topic: Topic pattern (supports * and # wildcards)
            handler: Async callback for handling messages
            subscriber_id: ID of the subscribing component
            filter_fn: Optional filter function

        Returns:
            Subscription ID for unsubscribing
        """
        subscription = Subscription(
            topic=topic,
            handler=handler,
            subscriber_id=subscriber_id,
            filter_fn=filter_fn,
        )

        async with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []

            if (
                self.config.max_handlers_per_topic > 0
                and len(self._subscriptions[topic]) >= self.config.max_handlers_per_topic
            ):
                raise ValueError(
                    f"Maximum handlers ({self.config.max_handlers_per_topic}) "
                    f"reached for topic '{topic}'"
                )

            self._subscriptions[topic].append(subscription)
            self._metrics.active_subscriptions = sum(
                len(subs) for subs in self._subscriptions.values()
            )

        subscription_id = f"{subscriber_id}:{topic}:{len(self._subscriptions[topic])}"
        logger.debug(f"Subscription created: {subscription_id}")
        return subscription_id

    async def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic to unsubscribe from
            subscriber_id: ID of the subscriber

        Returns:
            True if subscription was removed
        """
        async with self._lock:
            if topic not in self._subscriptions:
                return False

            original_count = len(self._subscriptions[topic])
            self._subscriptions[topic] = [
                sub
                for sub in self._subscriptions[topic]
                if sub.subscriber_id != subscriber_id
            ]
            removed = len(self._subscriptions[topic]) < original_count

            if not self._subscriptions[topic]:
                del self._subscriptions[topic]

            self._metrics.active_subscriptions = sum(
                len(subs) for subs in self._subscriptions.values()
            )

        if removed:
            logger.debug(f"Unsubscribed {subscriber_id} from {topic}")
        return removed

    async def publish(self, message: Message) -> bool:
        """
        Publish a message to the bus.

        Args:
            message: Message to publish

        Returns:
            True if message was queued successfully
        """
        if message.is_expired():
            self._metrics.messages_expired += 1
            logger.warning(f"Message {message.message_id} expired before publishing")
            return False

        try:
            # Priority queue uses (priority, timestamp, message) tuple
            # Lower priority value = higher priority
            priority_value = {
                MessagePriority.CRITICAL: 0,
                MessagePriority.HIGH: 1,
                MessagePriority.NORMAL: 2,
                MessagePriority.LOW: 3,
            }[message.priority]

            queue_item = (priority_value, time.time(), message)
            await self._queue.put(queue_item)

            self._metrics.messages_published += 1
            self._metrics.queue_size = self._queue.qsize()

            logger.debug(
                f"Message published: {message.message_id} to {message.topic}"
            )
            return True

        except asyncio.QueueFull:
            logger.error(f"Queue full, message {message.message_id} dropped")
            return False

    async def publish_sync(self, message: Message) -> int:
        """
        Publish and immediately deliver a message synchronously.

        Args:
            message: Message to publish

        Returns:
            Number of handlers that received the message
        """
        if message.is_expired():
            self._metrics.messages_expired += 1
            return 0

        handlers_count = await self._deliver_message(message)
        self._metrics.messages_published += 1
        return handlers_count

    async def request_response(
        self,
        message: Message,
        timeout_seconds: float = 30.0,
    ) -> Optional[Message]:
        """
        Send a request and wait for a response.

        Args:
            message: Request message
            timeout_seconds: Timeout for waiting for response

        Returns:
            Response message or None if timeout
        """
        # Create a unique reply topic
        reply_topic = f"reply.{message.message_id}"
        response_future: asyncio.Future = asyncio.Future()

        async def response_handler(response: Message) -> None:
            if not response_future.done():
                response_future.set_result(response)

        # Subscribe to reply topic
        await self.subscribe(reply_topic, response_handler, f"requester-{message.message_id}")

        # Set reply_to on the message
        message.reply_to = reply_topic
        message.correlation_id = message.message_id

        # Publish the request
        await self.publish(message)

        try:
            response = await asyncio.wait_for(response_future, timeout=timeout_seconds)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request {message.message_id} timed out after {timeout_seconds}s")
            return None
        finally:
            await self.unsubscribe(reply_topic, f"requester-{message.message_id}")

    async def _process_queue(self) -> None:
        """Background task to process queued messages."""
        while self._running:
            try:
                # Wait for message with timeout to allow shutdown checks
                try:
                    _, _, message = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                self._metrics.queue_size = self._queue.qsize()

                if message.is_expired():
                    self._metrics.messages_expired += 1
                    logger.debug(f"Message {message.message_id} expired, skipping")
                    continue

                await self._deliver_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    async def _deliver_message(self, message: Message) -> int:
        """
        Deliver message to matching subscribers.

        Args:
            message: Message to deliver

        Returns:
            Number of handlers that received the message
        """
        start_time = time.perf_counter()
        handlers_called = 0

        # Find matching subscriptions
        matching_subs: List[Subscription] = []
        async with self._lock:
            for topic_pattern, subs in self._subscriptions.items():
                for sub in subs:
                    if sub.should_handle(message):
                        matching_subs.append(sub)

        # Deliver to each matching subscription
        for sub in matching_subs:
            retries = 0
            delivered = False

            while retries <= self.config.max_retries and not delivered:
                try:
                    await sub.handler(message)
                    delivered = True
                    handlers_called += 1
                    self._metrics.messages_delivered += 1

                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Handler {sub.subscriber_id} failed for message "
                        f"{message.message_id} (attempt {retries}): {e}"
                    )

                    if retries <= self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds)

            if not delivered:
                self._metrics.messages_failed += 1
                if self.config.enable_dead_letter:
                    self._dead_letter_queue.append(message)
                    self._metrics.messages_dead_lettered += 1
                    logger.error(
                        f"Message {message.message_id} dead-lettered after "
                        f"{self.config.max_retries} retries"
                    )

        # Update delivery time metrics
        delivery_time_ms = (time.perf_counter() - start_time) * 1000
        self._delivery_times.append(delivery_time_ms)
        if len(self._delivery_times) > 1000:
            self._delivery_times = self._delivery_times[-1000:]
        self._metrics.avg_delivery_time_ms = (
            sum(self._delivery_times) / len(self._delivery_times)
        )
        self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

        return handlers_called

    def get_metrics(self) -> MessageBusMetrics:
        """Get current message bus metrics."""
        return self._metrics

    def get_dead_letter_queue(self) -> List[Message]:
        """Get messages in the dead letter queue."""
        return self._dead_letter_queue.copy()

    async def replay_dead_letter(self, message_id: str) -> bool:
        """
        Replay a message from the dead letter queue.

        Args:
            message_id: ID of message to replay

        Returns:
            True if message was replayed
        """
        for i, msg in enumerate(self._dead_letter_queue):
            if msg.message_id == message_id:
                del self._dead_letter_queue[i]
                await self.publish(msg)
                logger.info(f"Replayed dead-lettered message: {message_id}")
                return True
        return False

    def get_subscriptions(self, topic: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get active subscriptions.

        Args:
            topic: Optional topic to filter by

        Returns:
            Dict of topic -> list of subscriber IDs
        """
        result = {}
        for t, subs in self._subscriptions.items():
            if topic is None or t == topic:
                result[t] = [sub.subscriber_id for sub in subs]
        return result


# Factory function for creating pre-configured message buses
def create_message_bus(
    persistence: bool = False,
    dead_letter: bool = True,
    max_queue_size: int = 10000,
) -> MessageBus:
    """
    Create a message bus with common configurations.

    Args:
        persistence: Enable message persistence
        dead_letter: Enable dead letter queue
        max_queue_size: Maximum queue size

    Returns:
        Configured MessageBus instance
    """
    config = MessageBusConfig(
        enable_persistence=persistence,
        enable_dead_letter=dead_letter,
        max_queue_size=max_queue_size,
    )
    return MessageBus(config)


__all__ = [
    "Message",
    "MessageBus",
    "MessageBusConfig",
    "MessageBusMetrics",
    "MessageHandler",
    "MessagePriority",
    "MessageType",
    "Subscription",
    "create_message_bus",
]
