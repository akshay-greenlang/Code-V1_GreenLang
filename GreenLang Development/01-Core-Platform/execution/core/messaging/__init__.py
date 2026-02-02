# -*- coding: utf-8 -*-
"""
GreenLang Messaging - Event-driven agent communication infrastructure.

This module provides standardized messaging capabilities for GreenLang agents,
including publish/subscribe patterns, request-reply, and event routing.

Example:
    >>> from greenlang.core.messaging import (
    ...     InMemoryMessageBus,
    ...     create_event,
    ...     StandardEvents,
    ...     EventPriority
    ... )
    >>>
    >>> # Create message bus
    >>> bus = InMemoryMessageBus()
    >>> await bus.start()
    >>>
    >>> # Subscribe to events
    >>> async def handle_calculation(event):
    ...     print(f"Result: {event.payload}")
    >>>
    >>> await bus.subscribe(
    ...     StandardEvents.CALCULATION_COMPLETED,
    ...     handle_calculation,
    ...     "my-agent"
    ... )
    >>>
    >>> # Publish event
    >>> event = create_event(
    ...     event_type=StandardEvents.CALCULATION_COMPLETED,
    ...     source_agent="GL-001",
    ...     payload={"result": 42.5, "units": "kW"},
    ...     priority=EventPriority.HIGH
    ... )
    >>> await bus.publish(event)
    >>>
    >>> # Cleanup
    >>> await bus.close()

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

from greenlang.execution.core.messaging.events import (
    Event,
    EventPriority,
    StandardEvents,
    create_event,
)
from greenlang.execution.core.messaging.message_bus import (
    EventHandler,
    InMemoryMessageBus,
    MessageBus,
    MessageBusConfig,
    MessageBusMetrics,
    RedisMessageBus,
    Subscription,
    create_message_bus,
)

__all__ = [
    # Events
    "Event",
    "EventPriority",
    "StandardEvents",
    "create_event",
    # Message Bus
    "MessageBus",
    "InMemoryMessageBus",
    "RedisMessageBus",
    "MessageBusConfig",
    "MessageBusMetrics",
    "Subscription",
    "EventHandler",
    "create_message_bus",
]

__version__ = "1.0.0"
