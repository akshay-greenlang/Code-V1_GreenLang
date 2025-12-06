"""
Message Queue Module for GreenLang Agent Factory.

This package provides Redis Streams-based message queue functionality
for async agent execution, event publishing, and workflow orchestration.

Components:
- RedisStreamQueue: Core message queue using Redis Streams
- ExecutionWorker: Worker that processes agent executions
- EventBus: Event-driven architecture for pipeline orchestration

Example:
    >>> from queue import RedisStreamQueue, EventBus
    >>>
    >>> queue = RedisStreamQueue("redis://localhost:6379")
    >>> await queue.connect()
    >>>
    >>> bus = EventBus(queue)
    >>> bus.subscribe("execution.completed", handle_complete)
    >>>
    >>> await queue.publish_execution(...)
"""

from .redis_streams import (
    RedisStreamQueue,
    ExecutionWorker,
    StreamName,
    MessagePriority,
    ExecutionMessage,
    ResultMessage,
    EventMessage,
)
from .event_bus import (
    EventBus,
    Event,
    EventType,
    EventSubscription,
    log_all_events,
    execution_metrics_handler,
    compliance_alert_handler,
)

__all__ = [
    # Redis Streams
    "RedisStreamQueue",
    "ExecutionWorker",
    "StreamName",
    "MessagePriority",
    "ExecutionMessage",
    "ResultMessage",
    "EventMessage",
    # Event Bus
    "EventBus",
    "Event",
    "EventType",
    "EventSubscription",
    # Pre-built handlers
    "log_all_events",
    "execution_metrics_handler",
    "compliance_alert_handler",
]
