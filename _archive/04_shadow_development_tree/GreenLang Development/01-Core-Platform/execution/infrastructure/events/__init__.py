"""
GreenLang Infrastructure - Event-Driven Architecture

This module provides event-driven architecture components for GreenLang agents,
including event sourcing, saga orchestration, and dead letter queue handling.

All components follow:
- Exactly-once semantics
- Event sourcing patterns
- CQRS compatibility
- Provenance tracking
"""

from greenlang.infrastructure.events.event_schema import (
    EventSchema,
    EventType,
    BaseEvent,
    DomainEvent,
    IntegrationEvent,
)
from greenlang.infrastructure.events.event_producer import EventProducer
from greenlang.infrastructure.events.event_consumer import EventConsumer
from greenlang.infrastructure.events.dead_letter_queue import DeadLetterQueue
from greenlang.infrastructure.events.saga_orchestrator import SagaOrchestrator
from greenlang.infrastructure.events.event_sourcing import EventStore

__all__ = [
    "EventSchema",
    "EventType",
    "BaseEvent",
    "DomainEvent",
    "IntegrationEvent",
    "EventProducer",
    "EventConsumer",
    "DeadLetterQueue",
    "SagaOrchestrator",
    "EventStore",
]
