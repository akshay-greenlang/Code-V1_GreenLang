# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Core Module

This module contains core infrastructure components for event sourcing,
state management, and domain-driven design patterns.
"""

from core.events import (
    DomainEvent,
    EventStore,
    EventBus,
    Aggregate,
    SnapshotManager,
)

__all__ = [
    "DomainEvent",
    "EventStore",
    "EventBus",
    "Aggregate",
    "SnapshotManager",
]
