"""
Event Bus for Agent Pipeline Orchestration.

This module provides an event-driven architecture for orchestrating
agent workflows, notifications, and system events.

Features:
- Event publishing and subscription
- Handler registration with filters
- Async event processing
- Event replay and audit logging

Example:
    >>> bus = EventBus(queue)
    >>> bus.subscribe("agent.completed", on_agent_complete)
    >>> await bus.publish("agent.completed", {"execution_id": "123"})
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types in the system."""

    # Agent lifecycle events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_CERTIFIED = "agent.certified"
    AGENT_DEPRECATED = "agent.deprecated"

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_TIMEOUT = "execution.timeout"
    EXECUTION_CANCELLED = "execution.cancelled"

    # Validation events
    VALIDATION_PASSED = "validation.passed"
    VALIDATION_FAILED = "validation.failed"
    GOLDEN_TEST_PASSED = "goldentest.passed"
    GOLDEN_TEST_FAILED = "goldentest.failed"

    # Regulatory events
    COMPLIANCE_CHECK = "compliance.check"
    COMPLIANCE_ALERT = "compliance.alert"
    DEADLINE_WARNING = "deadline.warning"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "health.check"
    RATE_LIMIT_EXCEEDED = "ratelimit.exceeded"

    # Audit events
    AUDIT_LOG = "audit.log"
    PROVENANCE_VERIFIED = "provenance.verified"


class Event(BaseModel):
    """
    Event model for the event bus.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        source: Source of the event
        data: Event payload data
        timestamp: When the event occurred
        correlation_id: ID for tracing related events
        tenant_id: Tenant context
        user_id: User context
    """

    event_id: str = Field(..., description="Unique event ID")
    event_type: str = Field(..., description="Event type")
    source: str = Field(..., description="Event source")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    user_id: Optional[str] = Field(None, description="User ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Type alias for event handlers
EventHandler = Callable[[Event], Any]


class EventSubscription(BaseModel):
    """
    Subscription to an event type.

    Attributes:
        handler_id: Unique handler identifier
        event_pattern: Event type pattern (supports wildcards)
        handler: Handler function
        filters: Additional filters
    """

    handler_id: str
    event_pattern: str
    filters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class EventBus:
    """
    Event Bus for agent pipeline orchestration.

    This class provides:
    - Publish/subscribe for system events
    - Pattern-based event matching
    - Async event handlers
    - Event replay capabilities
    - Dead letter handling for failed events

    Attributes:
        queue: Redis stream queue for persistence
        subscriptions: Active event subscriptions
        handlers: Handler functions by ID

    Example:
        >>> bus = EventBus(queue)
        >>>
        >>> @bus.on("execution.completed")
        ... async def handle_completion(event: Event):
        ...     print(f"Completed: {event.data}")
        >>>
        >>> await bus.publish(EventType.EXECUTION_COMPLETED, {"id": "123"})
    """

    def __init__(
        self,
        queue=None,
        max_handlers_per_event: int = 100,
        handler_timeout_seconds: int = 30,
    ):
        """
        Initialize Event Bus.

        Args:
            queue: Optional Redis stream queue for persistence
            max_handlers_per_event: Max handlers per event type
            handler_timeout_seconds: Handler execution timeout
        """
        self.queue = queue
        self.max_handlers_per_event = max_handlers_per_event
        self.handler_timeout = handler_timeout_seconds

        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._handlers: Dict[str, EventHandler] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000

        logger.info("EventBus initialized")

    def subscribe(
        self,
        event_pattern: str,
        handler: EventHandler,
        handler_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_pattern: Event type or pattern (e.g., "execution.*")
            handler: Handler function (sync or async)
            handler_id: Optional handler ID
            filters: Additional filters for events

        Returns:
            Handler ID

        Example:
            >>> bus.subscribe("execution.*", handle_execution)
            >>> bus.subscribe("agent.certified", notify_team, filters={"priority": "high"})
        """
        import uuid

        handler_id = handler_id or str(uuid.uuid4())

        subscription = EventSubscription(
            handler_id=handler_id,
            event_pattern=event_pattern,
            filters=filters or {},
        )

        if event_pattern not in self._subscriptions:
            self._subscriptions[event_pattern] = []

        if len(self._subscriptions[event_pattern]) >= self.max_handlers_per_event:
            raise ValueError(f"Max handlers reached for pattern: {event_pattern}")

        self._subscriptions[event_pattern].append(subscription)
        self._handlers[handler_id] = handler

        logger.info(f"Subscribed to '{event_pattern}' with handler {handler_id}")

        return handler_id

    def unsubscribe(self, handler_id: str) -> bool:
        """
        Unsubscribe a handler.

        Args:
            handler_id: Handler ID to unsubscribe

        Returns:
            True if unsubscribed, False if not found
        """
        if handler_id not in self._handlers:
            return False

        del self._handlers[handler_id]

        # Remove from subscriptions
        for pattern, subs in self._subscriptions.items():
            self._subscriptions[pattern] = [
                s for s in subs if s.handler_id != handler_id
            ]

        logger.info(f"Unsubscribed handler {handler_id}")
        return True

    def on(self, event_pattern: str, filters: Optional[Dict[str, Any]] = None):
        """
        Decorator for subscribing to events.

        Args:
            event_pattern: Event type or pattern
            filters: Optional filters

        Example:
            >>> @bus.on("execution.completed")
            ... async def handle_completion(event: Event):
            ...     print(f"Completed: {event.data}")
        """
        def decorator(handler: EventHandler):
            self.subscribe(event_pattern, handler, filters=filters)
            return handler
        return decorator

    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "system",
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Event:
        """
        Publish an event.

        Args:
            event_type: Type of event
            data: Event data payload
            source: Event source
            correlation_id: Optional correlation ID
            tenant_id: Optional tenant ID
            user_id: Optional user ID

        Returns:
            Published event

        Example:
            >>> await bus.publish(
            ...     EventType.EXECUTION_COMPLETED,
            ...     {"execution_id": "123", "result": "success"}
            ... )
        """
        import uuid

        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type if isinstance(event_type, str) else event_type.value,
            source=source,
            data=data,
            correlation_id=correlation_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Persist to Redis if queue available
        if self.queue:
            await self.queue.publish_event(
                event_type=event.event_type,
                source=event.source,
                data=event.data,
                correlation_id=event.correlation_id,
            )

        # Dispatch to handlers
        await self._dispatch(event)

        logger.debug(f"Published event: type={event.event_type}, id={event.event_id}")

        return event

    async def _dispatch(self, event: Event) -> None:
        """
        Dispatch event to matching handlers.

        Args:
            event: Event to dispatch
        """
        matching_handlers: List[EventHandler] = []

        for pattern, subscriptions in self._subscriptions.items():
            if self._matches_pattern(event.event_type, pattern):
                for sub in subscriptions:
                    if self._matches_filters(event, sub.filters):
                        handler = self._handlers.get(sub.handler_id)
                        if handler:
                            matching_handlers.append(handler)

        if not matching_handlers:
            return

        # Execute handlers concurrently with timeout
        tasks = []
        for handler in matching_handlers:
            task = asyncio.create_task(self._execute_handler(handler, event))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(
        self,
        handler: EventHandler,
        event: Event,
    ) -> None:
        """
        Execute a handler with timeout and error handling.

        Args:
            handler: Handler function
            event: Event to handle
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await asyncio.wait_for(
                    handler(event),
                    timeout=self.handler_timeout,
                )
            else:
                handler(event)

        except asyncio.TimeoutError:
            logger.warning(
                f"Handler timeout for event {event.event_id}: {handler.__name__}"
            )
        except Exception as e:
            logger.error(
                f"Handler error for event {event.event_id}: {e}",
                exc_info=True,
            )

    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """
        Check if event type matches a pattern.

        Supports wildcards:
        - "*" matches any single segment
        - "**" matches any number of segments

        Args:
            event_type: Event type to check
            pattern: Pattern to match against

        Returns:
            True if matches
        """
        if pattern == "*" or pattern == "**":
            return True

        if "*" not in pattern:
            return event_type == pattern

        # Handle wildcard patterns
        pattern_parts = pattern.split(".")
        event_parts = event_type.split(".")

        if len(pattern_parts) != len(event_parts) and "**" not in pattern:
            return False

        for p_part, e_part in zip(pattern_parts, event_parts):
            if p_part == "*":
                continue
            if p_part == "**":
                return True
            if p_part != e_part:
                return False

        return True

    def _matches_filters(
        self,
        event: Event,
        filters: Dict[str, Any],
    ) -> bool:
        """
        Check if event matches subscription filters.

        Args:
            event: Event to check
            filters: Filters to apply

        Returns:
            True if all filters match
        """
        if not filters:
            return True

        for key, value in filters.items():
            event_value = event.data.get(key)
            if event_value != value:
                return False

        return True

    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get recent event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum events to return

        Returns:
            List of recent events
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    async def replay_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> int:
        """
        Replay historical events.

        Args:
            event_type: Optional filter by event type
            since: Replay events since this time

        Returns:
            Number of events replayed
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        for event in events:
            await self._dispatch(event)

        logger.info(f"Replayed {len(events)} events")
        return len(events)

    def get_subscriptions(self) -> Dict[str, int]:
        """
        Get subscription counts by pattern.

        Returns:
            Dictionary of pattern -> subscription count
        """
        return {
            pattern: len(subs)
            for pattern, subs in self._subscriptions.items()
        }


# Pre-configured event handlers for common scenarios

async def log_all_events(event: Event) -> None:
    """Handler that logs all events."""
    logger.info(
        f"Event: type={event.event_type}, source={event.source}, "
        f"id={event.event_id}, data={event.data}"
    )


async def execution_metrics_handler(event: Event) -> None:
    """Handler that tracks execution metrics."""
    if event.event_type == EventType.EXECUTION_COMPLETED.value:
        processing_time = event.data.get("processing_time_ms", 0)
        agent_id = event.data.get("agent_id", "unknown")
        logger.info(
            f"Execution metric: agent={agent_id}, time={processing_time}ms"
        )


async def compliance_alert_handler(event: Event) -> None:
    """Handler that processes compliance alerts."""
    if event.event_type == EventType.COMPLIANCE_ALERT.value:
        severity = event.data.get("severity", "info")
        message = event.data.get("message", "")
        logger.warning(f"Compliance Alert [{severity}]: {message}")
