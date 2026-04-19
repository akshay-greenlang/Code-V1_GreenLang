# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Event Bus

This module implements an async event bus with publish/subscribe pattern
for decoupled event handling. Supports multiple handlers per event type,
prioritized delivery, and error handling.

Design Principles:
    - Async/await for non-blocking event delivery
    - Multiple handlers per event type
    - Handler priority for ordered execution
    - Error isolation (one handler failure doesn't affect others)
    - Event filtering and transformation

Example:
    >>> bus = EventBus()
    >>> @bus.subscribe(ControlSetpointChanged)
    ... async def handle_setpoint_change(event):
    ...     print(f"Setpoint changed: {event.fuel_flow_setpoint}")
    >>> await bus.publish(event)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, Field

from core.events.base_event import DomainEvent

logger = logging.getLogger(__name__)

# Type for event handlers
TEvent = TypeVar("TEvent", bound=DomainEvent)
EventHandler = Callable[[TEvent], Awaitable[None]]


class DeliveryMode(str, Enum):
    """Event delivery mode."""
    SEQUENTIAL = "sequential"  # Handlers run one after another
    PARALLEL = "parallel"  # Handlers run concurrently
    FIRE_AND_FORGET = "fire_and_forget"  # Don't wait for handlers


class HandlerPriority(int, Enum):
    """Handler priority levels."""
    CRITICAL = 0  # Run first (safety handlers)
    HIGH = 100
    NORMAL = 500
    LOW = 900
    BACKGROUND = 1000  # Run last


@dataclass
class EventSubscription:
    """
    Represents a subscription to events.

    Attributes:
        subscription_id: Unique identifier
        event_type: Type of event subscribed to
        handler: Async handler function
        priority: Handler priority
        filter_fn: Optional filter function
        max_retries: Retry count on failure
        active: Whether subscription is active
    """

    subscription_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: Type[DomainEvent] = field(default=DomainEvent)
    handler: EventHandler = field(default=None)
    priority: HandlerPriority = field(default=HandlerPriority.NORMAL)
    filter_fn: Optional[Callable[[DomainEvent], bool]] = field(default=None)
    max_retries: int = field(default=0)
    active: bool = field(default=True)
    created_at: datetime = field(default_factory=datetime.utcnow)
    handler_name: str = field(default="")

    def __post_init__(self):
        """Set handler name from function."""
        if self.handler and not self.handler_name:
            self.handler_name = getattr(
                self.handler, "__name__", str(self.handler)
            )

    def matches(self, event: DomainEvent) -> bool:
        """Check if this subscription matches the event."""
        if not self.active:
            return False

        # Check event type
        if not isinstance(event, self.event_type):
            return False

        # Check filter
        if self.filter_fn and not self.filter_fn(event):
            return False

        return True


@dataclass
class DeliveryResult:
    """Result of event delivery to a handler."""

    subscription_id: str
    handler_name: str
    success: bool
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    retries_used: int = 0


class EventBusConfig(BaseModel):
    """Configuration for event bus."""

    delivery_mode: DeliveryMode = Field(
        default=DeliveryMode.SEQUENTIAL,
        description="Default event delivery mode"
    )
    max_handlers_per_event: int = Field(
        default=100,
        ge=1,
        description="Maximum handlers per event type"
    )
    handler_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Timeout for handler execution"
    )
    enable_dead_letter_queue: bool = Field(
        default=True,
        description="Queue failed events for retry"
    )
    dead_letter_queue_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum dead letter queue size"
    )
    log_all_events: bool = Field(
        default=False,
        description="Log all published events"
    )


class EventBus:
    """
    Async event bus with publish/subscribe pattern.

    The EventBus provides decoupled event delivery with features:
        - Multiple handlers per event type
        - Prioritized handler execution
        - Async/await for non-blocking delivery
        - Error isolation between handlers
        - Event filtering
        - Dead letter queue for failed events

    Thread Safety:
        The EventBus is safe for concurrent use from multiple coroutines.

    Example:
        >>> bus = EventBus()
        >>>
        >>> @bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.HIGH)
        ... async def handle_setpoint(event):
        ...     await process_setpoint_change(event)
        >>>
        >>> await bus.publish(event)
    """

    def __init__(self, config: Optional[EventBusConfig] = None):
        """
        Initialize event bus.

        Args:
            config: Bus configuration
        """
        self.config = config or EventBusConfig()

        # Subscriptions by event type
        self._subscriptions: Dict[Type[DomainEvent], List[EventSubscription]] = (
            defaultdict(list)
        )

        # Global subscriptions (receive all events)
        self._global_subscriptions: List[EventSubscription] = []

        # Dead letter queue
        self._dead_letter_queue: List[Tuple[DomainEvent, Exception]] = []

        # Metrics
        self._events_published = 0
        self._events_delivered = 0
        self._delivery_failures = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("Event bus initialized")

    def subscribe(
        self,
        event_type: Type[TEvent],
        priority: HandlerPriority = HandlerPriority.NORMAL,
        filter_fn: Optional[Callable[[TEvent], bool]] = None,
        max_retries: int = 0
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            priority: Handler priority
            filter_fn: Optional filter function
            max_retries: Retry count on failure

        Returns:
            Decorator function

        Example:
            >>> @bus.subscribe(ControlSetpointChanged)
            ... async def handle_setpoint(event):
            ...     print(f"New setpoint: {event.fuel_flow_setpoint}")
        """
        def decorator(handler: EventHandler) -> EventHandler:
            subscription = EventSubscription(
                event_type=event_type,
                handler=handler,
                priority=priority,
                filter_fn=filter_fn,
                max_retries=max_retries
            )
            self._add_subscription(subscription)
            return handler

        return decorator

    def subscribe_all(
        self,
        priority: HandlerPriority = HandlerPriority.NORMAL,
        filter_fn: Optional[Callable[[DomainEvent], bool]] = None
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to subscribe a handler to all events.

        Args:
            priority: Handler priority
            filter_fn: Optional filter function

        Returns:
            Decorator function
        """
        def decorator(handler: EventHandler) -> EventHandler:
            subscription = EventSubscription(
                event_type=DomainEvent,
                handler=handler,
                priority=priority,
                filter_fn=filter_fn
            )
            self._global_subscriptions.append(subscription)
            self._sort_subscriptions(self._global_subscriptions)
            logger.info(f"Global handler registered: {subscription.handler_name}")
            return handler

        return decorator

    def _add_subscription(self, subscription: EventSubscription) -> None:
        """Add a subscription to the registry."""
        event_type = subscription.event_type
        subscriptions = self._subscriptions[event_type]

        # Check max handlers
        if len(subscriptions) >= self.config.max_handlers_per_event:
            raise ValueError(
                f"Maximum handlers ({self.config.max_handlers_per_event}) "
                f"reached for {event_type.__name__}"
            )

        subscriptions.append(subscription)
        self._sort_subscriptions(subscriptions)

        logger.info(
            f"Handler '{subscription.handler_name}' subscribed to "
            f"{event_type.__name__} (priority={subscription.priority.name})"
        )

    def _sort_subscriptions(self, subscriptions: List[EventSubscription]) -> None:
        """Sort subscriptions by priority (lower value = higher priority)."""
        subscriptions.sort(key=lambda s: s.priority.value)

    def add_handler(
        self,
        event_type: Type[TEvent],
        handler: EventHandler,
        priority: HandlerPriority = HandlerPriority.NORMAL,
        filter_fn: Optional[Callable[[TEvent], bool]] = None,
        max_retries: int = 0
    ) -> str:
        """
        Programmatically add a handler.

        Args:
            event_type: Type of event to handle
            handler: Async handler function
            priority: Handler priority
            filter_fn: Optional filter function
            max_retries: Retry count on failure

        Returns:
            Subscription ID
        """
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
            max_retries=max_retries
        )
        self._add_subscription(subscription)
        return subscription.subscription_id

    def remove_handler(self, subscription_id: str) -> bool:
        """
        Remove a handler by subscription ID.

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if removed, False if not found
        """
        # Check global subscriptions
        for i, sub in enumerate(self._global_subscriptions):
            if sub.subscription_id == subscription_id:
                del self._global_subscriptions[i]
                logger.info(f"Global handler removed: {subscription_id}")
                return True

        # Check type-specific subscriptions
        for event_type, subscriptions in self._subscriptions.items():
            for i, sub in enumerate(subscriptions):
                if sub.subscription_id == subscription_id:
                    del subscriptions[i]
                    logger.info(
                        f"Handler removed from {event_type.__name__}: "
                        f"{subscription_id}"
                    )
                    return True

        return False

    async def publish(
        self,
        event: DomainEvent,
        delivery_mode: Optional[DeliveryMode] = None
    ) -> List[DeliveryResult]:
        """
        Publish an event to all matching handlers.

        Args:
            event: Event to publish
            delivery_mode: Override default delivery mode

        Returns:
            List of delivery results
        """
        mode = delivery_mode or self.config.delivery_mode

        if self.config.log_all_events:
            logger.debug(f"Publishing event: {event}")

        self._events_published += 1

        # Get matching handlers
        handlers = self._get_matching_handlers(event)

        if not handlers:
            logger.debug(f"No handlers for event: {event.event_type}")
            return []

        # Deliver based on mode
        if mode == DeliveryMode.SEQUENTIAL:
            results = await self._deliver_sequential(event, handlers)
        elif mode == DeliveryMode.PARALLEL:
            results = await self._deliver_parallel(event, handlers)
        elif mode == DeliveryMode.FIRE_AND_FORGET:
            asyncio.create_task(self._deliver_parallel(event, handlers))
            results = []
        else:
            raise ValueError(f"Unknown delivery mode: {mode}")

        # Track metrics
        for result in results:
            if result.success:
                self._events_delivered += 1
            else:
                self._delivery_failures += 1
                if self.config.enable_dead_letter_queue:
                    self._add_to_dead_letter_queue(event, result.error)

        return results

    def _get_matching_handlers(
        self,
        event: DomainEvent
    ) -> List[EventSubscription]:
        """Get all handlers matching an event."""
        handlers: List[EventSubscription] = []

        # Global handlers first
        for sub in self._global_subscriptions:
            if sub.matches(event):
                handlers.append(sub)

        # Type-specific handlers
        event_type = type(event)
        for sub in self._subscriptions.get(event_type, []):
            if sub.matches(event):
                handlers.append(sub)

        # Check parent types (for inheritance)
        for registered_type, subs in self._subscriptions.items():
            if registered_type != event_type and isinstance(event, registered_type):
                for sub in subs:
                    if sub.matches(event):
                        handlers.append(sub)

        # Sort by priority
        handlers.sort(key=lambda s: s.priority.value)

        return handlers

    async def _deliver_sequential(
        self,
        event: DomainEvent,
        handlers: List[EventSubscription]
    ) -> List[DeliveryResult]:
        """Deliver event to handlers sequentially."""
        results = []

        for sub in handlers:
            result = await self._invoke_handler(event, sub)
            results.append(result)

        return results

    async def _deliver_parallel(
        self,
        event: DomainEvent,
        handlers: List[EventSubscription]
    ) -> List[DeliveryResult]:
        """Deliver event to handlers in parallel."""
        tasks = [
            self._invoke_handler(event, sub)
            for sub in handlers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to DeliveryResult
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(DeliveryResult(
                    subscription_id=handlers[i].subscription_id,
                    handler_name=handlers[i].handler_name,
                    success=False,
                    error=result
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _invoke_handler(
        self,
        event: DomainEvent,
        subscription: EventSubscription
    ) -> DeliveryResult:
        """Invoke a single handler with retry logic."""
        import time
        start_time = time.perf_counter()
        retries = 0
        last_error: Optional[Exception] = None

        while retries <= subscription.max_retries:
            try:
                # Apply timeout
                await asyncio.wait_for(
                    subscription.handler(event),
                    timeout=self.config.handler_timeout_seconds
                )

                execution_time = (time.perf_counter() - start_time) * 1000

                return DeliveryResult(
                    subscription_id=subscription.subscription_id,
                    handler_name=subscription.handler_name,
                    success=True,
                    execution_time_ms=execution_time,
                    retries_used=retries
                )

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Handler '{subscription.handler_name}' timed out "
                    f"(attempt {retries + 1}/{subscription.max_retries + 1})"
                )
                retries += 1

            except Exception as e:
                last_error = e
                logger.error(
                    f"Handler '{subscription.handler_name}' failed: {e} "
                    f"(attempt {retries + 1}/{subscription.max_retries + 1})"
                )
                retries += 1

        execution_time = (time.perf_counter() - start_time) * 1000

        return DeliveryResult(
            subscription_id=subscription.subscription_id,
            handler_name=subscription.handler_name,
            success=False,
            error=last_error,
            execution_time_ms=execution_time,
            retries_used=retries - 1
        )

    def _add_to_dead_letter_queue(
        self,
        event: DomainEvent,
        error: Optional[Exception]
    ) -> None:
        """Add failed event to dead letter queue."""
        if len(self._dead_letter_queue) >= self.config.dead_letter_queue_size:
            # Remove oldest
            self._dead_letter_queue.pop(0)

        self._dead_letter_queue.append((event, error))
        logger.warning(
            f"Event added to dead letter queue: {event.event_type} "
            f"(queue size: {len(self._dead_letter_queue)})"
        )

    def get_dead_letter_queue(self) -> List[Tuple[DomainEvent, Exception]]:
        """Get the dead letter queue contents."""
        return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> int:
        """Clear the dead letter queue."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count

    async def replay_dead_letter_queue(self) -> List[DeliveryResult]:
        """Replay all events in the dead letter queue."""
        results = []
        events = list(self._dead_letter_queue)
        self._dead_letter_queue.clear()

        for event, _ in events:
            event_results = await self.publish(event)
            results.extend(event_results)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._events_published,
            "events_delivered": self._events_delivered,
            "delivery_failures": self._delivery_failures,
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "total_subscriptions": sum(
                len(subs) for subs in self._subscriptions.values()
            ) + len(self._global_subscriptions),
            "subscriptions_by_type": {
                event_type.__name__: len(subs)
                for event_type, subs in self._subscriptions.items()
            }
        }

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all subscriptions."""
        all_subs = []

        for sub in self._global_subscriptions:
            all_subs.append({
                "subscription_id": sub.subscription_id,
                "event_type": "* (global)",
                "handler_name": sub.handler_name,
                "priority": sub.priority.name,
                "active": sub.active
            })

        for event_type, subs in self._subscriptions.items():
            for sub in subs:
                all_subs.append({
                    "subscription_id": sub.subscription_id,
                    "event_type": event_type.__name__,
                    "handler_name": sub.handler_name,
                    "priority": sub.priority.name,
                    "active": sub.active
                })

        return all_subs

    async def close(self) -> None:
        """Close the event bus and cleanup resources."""
        # Wait for any pending deliveries
        await asyncio.sleep(0)

        logger.info(
            f"Event bus closed. Stats: "
            f"published={self._events_published}, "
            f"delivered={self._events_delivered}, "
            f"failures={self._delivery_failures}"
        )


# Convenience type for typing event handlers
from typing import Tuple
