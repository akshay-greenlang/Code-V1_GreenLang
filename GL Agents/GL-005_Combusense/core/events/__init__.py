# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Event Sourcing Infrastructure

This module provides a complete event sourcing implementation for combustion
control state management with zero-hallucination guarantees.

Components:
    - DomainEvent: Base class for all domain events
    - EventStore: Append-only event storage with persistence
    - EventBus: Async event bus with pub/sub pattern
    - Aggregate: Base aggregate class with event replay
    - SnapshotManager: Periodic snapshot management for performance

Example:
    >>> from core.events import EventStore, EventBus, Aggregate
    >>> store = EventStore(backend="sqlite", connection_string="events.db")
    >>> bus = EventBus()
    >>> await store.initialize()
"""

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import for event sourcing components."""
    if name in ("DomainEvent", "EventMetadata"):
        from core.events.base_event import DomainEvent, EventMetadata
        return DomainEvent if name == "DomainEvent" else EventMetadata
    elif name in ("EventStore", "EventStoreBackend"):
        from core.events.event_store import EventStore, EventStoreBackend
        return EventStore if name == "EventStore" else EventStoreBackend
    elif name in ("EventBus", "EventHandler", "EventSubscription"):
        from core.events.event_bus import EventBus, EventHandler, EventSubscription
        if name == "EventBus":
            return EventBus
        elif name == "EventHandler":
            return EventHandler
        else:
            return EventSubscription
    elif name in ("Aggregate", "AggregateRoot"):
        from core.events.aggregates import Aggregate, AggregateRoot
        return Aggregate if name == "Aggregate" else AggregateRoot
    elif name in ("SnapshotManager", "Snapshot"):
        from core.events.snapshots import SnapshotManager, Snapshot
        return SnapshotManager if name == "SnapshotManager" else Snapshot
    elif name in (
        "ControlSetpointChanged",
        "SafetyInterventionTriggered",
        "OptimizationCompleted",
        "SensorReadingReceived",
        "AlarmTriggered",
        "SystemStateChanged",
    ):
        from core.events.domain_events import (
            ControlSetpointChanged,
            SafetyInterventionTriggered,
            OptimizationCompleted,
            SensorReadingReceived,
            AlarmTriggered,
            SystemStateChanged,
        )
        mapping = {
            "ControlSetpointChanged": ControlSetpointChanged,
            "SafetyInterventionTriggered": SafetyInterventionTriggered,
            "OptimizationCompleted": OptimizationCompleted,
            "SensorReadingReceived": SensorReadingReceived,
            "AlarmTriggered": AlarmTriggered,
            "SystemStateChanged": SystemStateChanged,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "DomainEvent",
    "EventMetadata",
    # Event store
    "EventStore",
    "EventStoreBackend",
    # Event bus
    "EventBus",
    "EventHandler",
    "EventSubscription",
    # Aggregates
    "Aggregate",
    "AggregateRoot",
    # Snapshots
    "SnapshotManager",
    "Snapshot",
    # Domain events
    "ControlSetpointChanged",
    "SafetyInterventionTriggered",
    "OptimizationCompleted",
    "SensorReadingReceived",
    "AlarmTriggered",
    "SystemStateChanged",
]
