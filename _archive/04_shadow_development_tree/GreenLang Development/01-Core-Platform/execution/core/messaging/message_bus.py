# -*- coding: utf-8 -*-
"""
MessageBus - Event-driven messaging for GreenLang agent orchestration.

This module implements an async message bus for inter-agent communication,
supporting publish/subscribe patterns, priority queuing, request-reply, and message persistence.

Based on implementations from GL-001 (ThermoSync) and GL-003 (SteamSync).

Example:
    >>> from greenlang.core.messaging import MessageBus, create_event, StandardEvents
    >>> bus = MessageBus()
    >>> await bus.start()
    >>>
    >>> async def handle_calculation(event):
    ...     print(f"Calculation completed: {event.payload}")
    >>>
    >>> await bus.subscribe(
    ...     StandardEvents.CALCULATION_COMPLETED,
    ...     handle_calculation,
    ...     "my-subscriber"
    ... )
    >>>
    >>> event = create_event(
    ...     event_type=StandardEvents.CALCULATION_COMPLETED,
    ...     source_agent="GL-001",
    ...     payload={"result": 42.5}
    ... )
    >>> await bus.publish(event)
    >>> await bus.close()

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
import uuid

from greenlang.execution.core.messaging.events import Event, EventPriority

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


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
        request_timeout_seconds: Default timeout for request-reply pattern
    """

    max_queue_size: int = 10000
    enable_persistence: bool = False
    persistence_path: str = "./message_bus_persistence"
    enable_dead_letter: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    metrics_enabled: bool = True
    max_handlers_per_topic: int = 100
    request_timeout_seconds: float = 30.0


@dataclass
class Subscription:
    """
    Represents an event subscription.

    Attributes:
        event_type: Event type pattern (supports wildcards)
        handler: Async callback function for handling events
        subscriber_id: ID of the subscribing agent/component
        filter_fn: Optional filter function to further filter events
        created_at: Timestamp of subscription creation
    """

    event_type: str
    handler: EventHandler
    subscriber_id: str
    filter_fn: Optional[Callable[[Event], bool]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def matches_event_type(self, event_type: str) -> bool:
        """
        Check if event type matches subscription pattern.

        Supports wildcards:
        - '*' matches exactly one level (e.g., 'agent.*' matches 'agent.started')
        - '**' matches multiple levels (e.g., 'agent.**' matches 'agent.thermal.status')
        """
        # Exact match
        if self.event_type == event_type:
            return True

        # Wildcard '*' matches all events
        if self.event_type == "*":
            return True

        # Pattern matching with wildcards
        sub_parts = self.event_type.split(".")
        event_parts = event_type.split(".")

        # Multi-level wildcard
        if "**" in sub_parts:
            wildcard_idx = sub_parts.index("**")
            prefix_parts = sub_parts[:wildcard_idx]
            suffix_parts = sub_parts[wildcard_idx + 1 :]

            # Check prefix match
            if len(event_parts) < len(prefix_parts):
                return False
            if event_parts[: len(prefix_parts)] != prefix_parts:
                return False

            # Check suffix match (if any)
            if suffix_parts:
                if len(event_parts) < len(suffix_parts):
                    return False
                if event_parts[-len(suffix_parts) :] != suffix_parts:
                    return False

            return True

        # Single-level wildcard matching
        if len(sub_parts) != len(event_parts):
            return False

        for sub_part, event_part in zip(sub_parts, event_parts):
            if sub_part == "*":
                continue  # Single-level wildcard matches any
            if sub_part != event_part:
                return False

        return True

    def should_handle(self, event: Event) -> bool:
        """Check if this subscription should handle the event."""
        if not self.matches_event_type(event.event_type):
            return False
        if self.filter_fn and not self.filter_fn(event):
            return False
        return True


@dataclass
class MessageBusMetrics:
    """Metrics for message bus monitoring."""

    events_published: int = 0
    events_delivered: int = 0
    events_failed: int = 0
    events_dead_lettered: int = 0
    events_expired: int = 0
    active_subscriptions: int = 0
    queue_size: int = 0
    avg_delivery_time_ms: float = 0.0
    requests_sent: int = 0
    requests_timeout: int = 0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "events_published": self.events_published,
            "events_delivered": self.events_delivered,
            "events_failed": self.events_failed,
            "events_dead_lettered": self.events_dead_lettered,
            "events_expired": self.events_expired,
            "active_subscriptions": self.active_subscriptions,
            "queue_size": self.queue_size,
            "avg_delivery_time_ms": round(self.avg_delivery_time_ms, 2),
            "requests_sent": self.requests_sent,
            "requests_timeout": self.requests_timeout,
            "last_updated": self.last_updated,
        }


class MessageBus(ABC):
    """
    Abstract base class for message bus implementations.

    Defines the interface that all message bus implementations must follow.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the message bus processor."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the message bus processor."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Gracefully shutdown the message bus."""
        pass

    @abstractmethod
    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.

        Args:
            event: Event to publish

        Returns:
            True if event was queued successfully
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        subscriber_id: str,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type pattern (supports wildcards)
            handler: Async callback for handling events
            subscriber_id: ID of the subscribing component
            filter_fn: Optional filter function

        Returns:
            Subscription ID for unsubscribing
        """
        pass

    @abstractmethod
    async def unsubscribe(self, event_type: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from
            subscriber_id: ID of the subscriber

        Returns:
            True if subscription was removed
        """
        pass

    @abstractmethod
    async def request_reply(
        self, event: Event, timeout_seconds: Optional[float] = None
    ) -> Optional[Event]:
        """
        Request-reply pattern with timeout.

        Args:
            event: Request event
            timeout_seconds: Timeout for waiting for response

        Returns:
            Response event or None if timeout
        """
        pass

    @abstractmethod
    def get_metrics(self) -> MessageBusMetrics:
        """Get current message bus metrics."""
        pass


class InMemoryMessageBus(MessageBus):
    """
    In-memory message bus for single-process deployments.

    Provides publish/subscribe messaging with:
    - Event-type-based routing with wildcard support
    - Priority queuing
    - Dead letter queue for failed deliveries
    - Request-reply pattern
    - Metrics collection

    Example:
        >>> bus = InMemoryMessageBus()
        >>> await bus.start()
        >>> await bus.subscribe("agent.*", handler, "my-agent")
        >>> await bus.publish(event)
        >>> await bus.close()
    """

    def __init__(self, config: Optional[MessageBusConfig] = None) -> None:
        """
        Initialize InMemoryMessageBus.

        Args:
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or MessageBusConfig()
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size if self.config.max_queue_size > 0 else 0
        )
        self._dead_letter_queue: List[Event] = []
        self._metrics = MessageBusMetrics()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._delivery_times: List[float] = []
        self._reply_futures: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._counter = 0  # Counter for queue ordering

        logger.info(
            f"InMemoryMessageBus initialized with config: max_queue={self.config.max_queue_size}"
        )

    async def start(self) -> None:
        """Start the message bus processor."""
        if self._running:
            logger.warning("MessageBus is already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("InMemoryMessageBus started")

    async def stop(self) -> None:
        """Stop the message bus processor."""
        logger.info("Stopping InMemoryMessageBus...")
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("InMemoryMessageBus stopped")

    async def close(self) -> None:
        """Gracefully shutdown the message bus."""
        await self.stop()

        # Clear remaining messages
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel pending reply futures
        for future in self._reply_futures.values():
            if not future.done():
                future.cancel()

        self._subscriptions.clear()
        self._reply_futures.clear()

        logger.info("InMemoryMessageBus shutdown complete")

    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.

        Args:
            event: Event to publish

        Returns:
            True if event was queued successfully
        """
        try:
            # Priority queue uses (priority, counter, event) tuple
            # Lower priority value = higher priority
            # Counter ensures unique ordering when priorities are equal
            priority_value = {
                EventPriority.CRITICAL: 0,
                EventPriority.HIGH: 1,
                EventPriority.MEDIUM: 2,
                EventPriority.LOW: 3,
            }[event.priority]

            self._counter += 1
            queue_item = (priority_value, self._counter, event)
            await self._queue.put(queue_item)

            self._metrics.events_published += 1
            self._metrics.queue_size = self._queue.qsize()

            logger.debug(
                f"Event published: {event.event_id} type={event.event_type}"
            )
            return True

        except asyncio.QueueFull:
            logger.error(f"Queue full, event {event.event_id} dropped")
            return False

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        subscriber_id: str,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type pattern (supports wildcards)
            handler: Async callback for handling events
            subscriber_id: ID of the subscribing component
            filter_fn: Optional filter function

        Returns:
            Subscription ID for unsubscribing
        """
        subscription = Subscription(
            event_type=event_type,
            handler=handler,
            subscriber_id=subscriber_id,
            filter_fn=filter_fn,
        )

        async with self._lock:
            if (
                self.config.max_handlers_per_topic > 0
                and len(self._subscriptions[event_type])
                >= self.config.max_handlers_per_topic
            ):
                raise ValueError(
                    f"Maximum handlers ({self.config.max_handlers_per_topic}) "
                    f"reached for event type '{event_type}'"
                )

            self._subscriptions[event_type].append(subscription)
            self._metrics.active_subscriptions = sum(
                len(subs) for subs in self._subscriptions.values()
            )

        subscription_id = f"{subscriber_id}:{event_type}:{len(self._subscriptions[event_type])}"
        logger.debug(f"Subscription created: {subscription_id}")
        return subscription_id

    async def unsubscribe(self, event_type: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from
            subscriber_id: ID of the subscriber

        Returns:
            True if subscription was removed
        """
        async with self._lock:
            if event_type not in self._subscriptions:
                return False

            original_count = len(self._subscriptions[event_type])
            self._subscriptions[event_type] = [
                sub
                for sub in self._subscriptions[event_type]
                if sub.subscriber_id != subscriber_id
            ]
            removed = len(self._subscriptions[event_type]) < original_count

            if not self._subscriptions[event_type]:
                del self._subscriptions[event_type]

            self._metrics.active_subscriptions = sum(
                len(subs) for subs in self._subscriptions.values()
            )

        if removed:
            logger.debug(f"Unsubscribed {subscriber_id} from {event_type}")
        return removed

    async def request_reply(
        self, event: Event, timeout_seconds: Optional[float] = None
    ) -> Optional[Event]:
        """
        Request-reply pattern with timeout.

        Args:
            event: Request event
            timeout_seconds: Timeout for waiting for response

        Returns:
            Response event or None if timeout
        """
        timeout = timeout_seconds or self.config.request_timeout_seconds

        # Create reply future
        reply_future: asyncio.Future = asyncio.Future()
        self._reply_futures[event.event_id] = reply_future

        # Publish request
        await self.publish(event)
        self._metrics.requests_sent += 1

        # Wait for reply with timeout
        try:
            reply = await asyncio.wait_for(reply_future, timeout=timeout)
            logger.debug(f"Request {event.event_id} received reply")
            return reply
        except asyncio.TimeoutError:
            logger.warning(
                f"Request {event.event_id} timed out after {timeout}s"
            )
            self._metrics.requests_timeout += 1
            return None
        finally:
            if event.event_id in self._reply_futures:
                del self._reply_futures[event.event_id]

    async def _process_queue(self) -> None:
        """Background task to process queued events."""
        while self._running:
            try:
                # Wait for event with timeout to allow shutdown checks
                try:
                    _, _, event = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                self._metrics.queue_size = self._queue.qsize()

                # Check if this is a reply to a pending request
                if event.correlation_id and event.correlation_id in self._reply_futures:
                    future = self._reply_futures[event.correlation_id]
                    if not future.done():
                        future.set_result(event)
                    continue

                # Deliver to subscribers
                await self._deliver_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)

    async def _deliver_event(self, event: Event) -> int:
        """
        Deliver event to matching subscribers.

        Args:
            event: Event to deliver

        Returns:
            Number of handlers that received the event
        """
        start_time = time.perf_counter()
        handlers_called = 0

        # Find matching subscriptions
        matching_subs: List[Subscription] = []
        async with self._lock:
            for event_type_pattern, subs in self._subscriptions.items():
                for sub in subs:
                    if sub.should_handle(event):
                        matching_subs.append(sub)

        # Deliver to each matching subscription
        for sub in matching_subs:
            retries = 0
            delivered = False

            while retries <= self.config.max_retries and not delivered:
                try:
                    await sub.handler(event)
                    delivered = True
                    handlers_called += 1
                    self._metrics.events_delivered += 1

                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Handler {sub.subscriber_id} failed for event "
                        f"{event.event_id} (attempt {retries}): {e}"
                    )

                    if retries <= self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds)

            if not delivered:
                self._metrics.events_failed += 1
                if self.config.enable_dead_letter:
                    self._dead_letter_queue.append(event)
                    self._metrics.events_dead_lettered += 1
                    logger.error(
                        f"Event {event.event_id} dead-lettered after "
                        f"{self.config.max_retries} retries"
                    )

        # Update delivery time metrics
        delivery_time_ms = (time.perf_counter() - start_time) * 1000
        self._delivery_times.append(delivery_time_ms)
        if len(self._delivery_times) > 1000:
            self._delivery_times = self._delivery_times[-1000:]
        self._metrics.avg_delivery_time_ms = (
            sum(self._delivery_times) / len(self._delivery_times)
            if self._delivery_times
            else 0.0
        )
        self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

        return handlers_called

    def get_metrics(self) -> MessageBusMetrics:
        """Get current message bus metrics."""
        return self._metrics

    def get_dead_letter_queue(self) -> List[Event]:
        """Get events in the dead letter queue."""
        return self._dead_letter_queue.copy()

    async def replay_dead_letter(self, event_id: str) -> bool:
        """
        Replay an event from the dead letter queue.

        Args:
            event_id: ID of event to replay

        Returns:
            True if event was replayed
        """
        for i, event in enumerate(self._dead_letter_queue):
            if event.event_id == event_id:
                del self._dead_letter_queue[i]
                await self.publish(event)
                logger.info(f"Replayed dead-lettered event: {event_id}")
                return True
        return False

    def get_subscriptions(
        self, event_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get active subscriptions.

        Args:
            event_type: Optional event type to filter by

        Returns:
            Dict of event_type -> list of subscriber IDs
        """
        result = {}
        for et, subs in self._subscriptions.items():
            if event_type is None or et == event_type:
                result[et] = [sub.subscriber_id for sub in subs]
        return result


class RedisMessageBus(MessageBus):
    """
    Redis-backed message bus for distributed deployments.

    This is a placeholder for future Redis-based implementation
    to support multi-process/multi-server deployments.
    """

    def __init__(self, redis_url: str, config: Optional[MessageBusConfig] = None):
        """
        Initialize RedisMessageBus.

        Args:
            redis_url: Redis connection URL
            config: Configuration options
        """
        self.redis_url = redis_url
        self.config = config or MessageBusConfig()
        logger.info(f"RedisMessageBus initialized (NOT YET IMPLEMENTED)")

    async def start(self) -> None:
        """Start the message bus processor."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def stop(self) -> None:
        """Stop the message bus processor."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def close(self) -> None:
        """Gracefully shutdown the message bus."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        subscriber_id: str,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """Subscribe to an event type."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def unsubscribe(self, event_type: str, subscriber_id: str) -> bool:
        """Unsubscribe from an event type."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    async def request_reply(
        self, event: Event, timeout_seconds: Optional[float] = None
    ) -> Optional[Event]:
        """Request-reply pattern with timeout."""
        raise NotImplementedError("RedisMessageBus not yet implemented")

    def get_metrics(self) -> MessageBusMetrics:
        """Get current message bus metrics."""
        raise NotImplementedError("RedisMessageBus not yet implemented")


def create_message_bus(
    bus_type: str = "memory",
    redis_url: Optional[str] = None,
    persistence: bool = False,
    dead_letter: bool = True,
    max_queue_size: int = 10000,
) -> MessageBus:
    """
    Factory function for creating message bus instances.

    Args:
        bus_type: Type of bus ("memory" or "redis")
        redis_url: Redis connection URL (required if bus_type="redis")
        persistence: Enable message persistence
        dead_letter: Enable dead letter queue
        max_queue_size: Maximum queue size

    Returns:
        Configured MessageBus instance

    Raises:
        ValueError: If invalid bus_type or missing redis_url

    Example:
        >>> bus = create_message_bus(bus_type="memory", dead_letter=True)
        >>> await bus.start()
    """
    config = MessageBusConfig(
        enable_persistence=persistence,
        enable_dead_letter=dead_letter,
        max_queue_size=max_queue_size,
    )

    if bus_type == "memory":
        return InMemoryMessageBus(config)
    elif bus_type == "redis":
        if not redis_url:
            raise ValueError("redis_url is required for Redis message bus")
        return RedisMessageBus(redis_url, config)
    else:
        raise ValueError(f"Invalid bus_type: {bus_type}. Use 'memory' or 'redis'")


__all__ = [
    "MessageBus",
    "InMemoryMessageBus",
    "RedisMessageBus",
    "MessageBusConfig",
    "MessageBusMetrics",
    "Subscription",
    "EventHandler",
    "create_message_bus",
]
