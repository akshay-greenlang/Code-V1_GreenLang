# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Aggregate Classes

This module implements the aggregate pattern for event sourcing.
Aggregates are the primary building blocks that encapsulate domain logic
and maintain consistency boundaries through event replay.

Design Principles:
    - Aggregates are rebuilt from events (event sourcing)
    - All state changes emit events
    - Events are applied in order
    - Aggregates are consistency boundaries
    - State is deterministically reproducible

Pattern:
    1. Commands are received by the aggregate
    2. Business rules are validated
    3. If valid, events are emitted
    4. Events are applied to update state
    5. Events are persisted to event store

Example:
    >>> class CombustionAggregate(Aggregate):
    ...     def apply_control_setpoint_changed(self, event):
    ...         self._fuel_flow = event.fuel_flow_setpoint
    ...         self._air_flow = event.air_flow_setpoint
    >>>
    >>> aggregate = CombustionAggregate("burner-001")
    >>> events = await aggregate.load_from_store(event_store)
    >>> aggregate.apply_all(events)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from core.events.base_event import DomainEvent, EventMetadata
from core.events.event_store import EventStore

logger = logging.getLogger(__name__)

TAggregate = TypeVar("TAggregate", bound="Aggregate")


class AggregateState(BaseModel):
    """
    Base class for aggregate state snapshots.

    Subclass this to define state that can be snapshotted.
    """

    model_config = ConfigDict(frozen=False)

    version: int = Field(default=0, ge=0, description="Current version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Aggregate(ABC):
    """
    Base class for event-sourced aggregates.

    An aggregate is a cluster of domain objects that can be treated as a
    single unit for data changes. The aggregate root guarantees the
    consistency of changes being made within the aggregate.

    In event sourcing, aggregates are rebuilt by replaying their event
    history. All state changes are captured as events.

    Subclass Pattern:
        1. Define _apply_<event_type> methods for each event type
        2. Call _apply_event() or apply_all() to update state
        3. Use raise_event() to emit new events
        4. Override create_snapshot() and restore_from_snapshot()

    Attributes:
        aggregate_id: Unique identifier for this aggregate
        aggregate_type: Type name of the aggregate
        version: Current version (number of applied events)
        uncommitted_events: Events not yet persisted

    Example:
        >>> class OrderAggregate(Aggregate):
        ...     def __init__(self, order_id: str):
        ...         super().__init__(order_id, "Order")
        ...         self._items = []
        ...         self._status = "pending"
        ...
        ...     def apply_item_added(self, event):
        ...         self._items.append(event.item)
        ...
        ...     def add_item(self, item):
        ...         if self._status != "pending":
        ...             raise ValueError("Cannot modify completed order")
        ...         event = ItemAdded(aggregate_id=self.aggregate_id, item=item)
        ...         self.raise_event(event)
    """

    # Registry of apply methods by event type
    _apply_methods: Dict[str, Callable] = {}

    def __init__(
        self,
        aggregate_id: str,
        aggregate_type: str = "Aggregate"
    ):
        """
        Initialize the aggregate.

        Args:
            aggregate_id: Unique identifier
            aggregate_type: Type name
        """
        self._aggregate_id = aggregate_id
        self._aggregate_type = aggregate_type
        self._version = 0
        self._uncommitted_events: List[DomainEvent] = []
        self._event_history: Deque[DomainEvent] = deque(maxlen=1000)
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()

        # Build apply method registry
        self._build_apply_registry()

        logger.debug(f"Created aggregate {aggregate_type}:{aggregate_id}")

    def _build_apply_registry(self) -> None:
        """Build registry of apply methods."""
        self._apply_methods = {}

        for attr_name in dir(self):
            if attr_name.startswith("apply_"):
                # Extract event type from method name
                # apply_control_setpoint_changed -> ControlSetpointChanged
                event_name_parts = attr_name[6:].split("_")
                event_type = "".join(
                    part.capitalize() for part in event_name_parts
                )
                method = getattr(self, attr_name)
                if callable(method):
                    self._apply_methods[event_type] = method

    @property
    def aggregate_id(self) -> str:
        """Get aggregate ID."""
        return self._aggregate_id

    @property
    def aggregate_type(self) -> str:
        """Get aggregate type."""
        return self._aggregate_type

    @property
    def version(self) -> int:
        """Get current version."""
        return self._version

    @property
    def uncommitted_events(self) -> List[DomainEvent]:
        """Get uncommitted events."""
        return list(self._uncommitted_events)

    @property
    def is_new(self) -> bool:
        """Check if aggregate is new (no events)."""
        return self._version == 0

    def raise_event(self, event: DomainEvent) -> None:
        """
        Raise a new event (apply and mark as uncommitted).

        This is the primary method for changing aggregate state.
        The event is applied immediately and marked for persistence.

        Args:
            event: Event to raise
        """
        # Apply the event
        self._apply_event(event)

        # Mark as uncommitted
        self._uncommitted_events.append(event)

        logger.debug(
            f"Event raised: {event.event_type} "
            f"on {self._aggregate_type}:{self._aggregate_id}"
        )

    def _apply_event(self, event: DomainEvent) -> None:
        """
        Apply a single event to update aggregate state.

        Args:
            event: Event to apply
        """
        event_type = event.event_type

        # Look up apply method
        apply_method = self._apply_methods.get(event_type)

        if apply_method:
            apply_method(event)
        else:
            # Check for generic apply method
            if hasattr(self, "_apply_unknown"):
                self._apply_unknown(event)
            else:
                logger.warning(
                    f"No apply method for event type: {event_type}"
                )

        # Update version and timestamp
        self._version += 1
        self._updated_at = event.metadata.timestamp

        # Store in history
        self._event_history.append(event)

    def apply_all(self, events: List[DomainEvent]) -> None:
        """
        Apply multiple events in order.

        Used when loading aggregate from event store.

        Args:
            events: Events to apply
        """
        for event in events:
            self._apply_event(event)

        logger.debug(
            f"Applied {len(events)} events to "
            f"{self._aggregate_type}:{self._aggregate_id}, "
            f"version now {self._version}"
        )

    def mark_events_committed(self) -> List[DomainEvent]:
        """
        Mark uncommitted events as committed.

        Called after events are persisted to event store.

        Returns:
            The events that were committed
        """
        committed = self._uncommitted_events
        self._uncommitted_events = []
        return committed

    def clear_uncommitted_events(self) -> None:
        """Clear uncommitted events (e.g., on rollback)."""
        self._uncommitted_events.clear()

    def get_event_history(
        self,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """
        Get recent event history.

        Args:
            limit: Maximum events to return

        Returns:
            List of events
        """
        events = list(self._event_history)
        if limit:
            events = events[-limit:]
        return events

    async def load_from_store(
        self,
        event_store: EventStore,
        from_version: int = 0
    ) -> int:
        """
        Load and apply events from event store.

        Args:
            event_store: Event store instance
            from_version: Version to start from

        Returns:
            Number of events loaded
        """
        events = await event_store.load(
            aggregate_id=self._aggregate_id,
            aggregate_type=self._aggregate_type,
            from_version=from_version
        )

        self.apply_all(events)
        return len(events)

    async def save_to_store(
        self,
        event_store: EventStore,
        expected_version: Optional[int] = None
    ) -> int:
        """
        Save uncommitted events to event store.

        Args:
            event_store: Event store instance
            expected_version: Expected current version (for concurrency)

        Returns:
            New version number
        """
        if not self._uncommitted_events:
            return self._version

        # Get the stored version before uncommitted events
        # This is: current_version - uncommitted_count
        stored_version = self._version - len(self._uncommitted_events)

        # If expected_version not specified, use None to skip concurrency check
        # for new aggregates, or use stored version for existing ones
        if expected_version is None and stored_version > 0:
            expected_version = stored_version

        new_version = await event_store.append(
            aggregate_id=self._aggregate_id,
            events=self._uncommitted_events,
            aggregate_type=self._aggregate_type,
            expected_version=expected_version
        )

        self.mark_events_committed()
        return new_version

    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current aggregate state.

        Override in subclasses to include domain-specific state.

        Returns:
            Serializable state dictionary
        """
        return {
            "aggregate_id": self._aggregate_id,
            "aggregate_type": self._aggregate_type,
            "version": self._version,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
        }

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore aggregate state from a snapshot.

        Override in subclasses to restore domain-specific state.

        Args:
            snapshot: Snapshot dictionary
        """
        self._version = snapshot.get("version", 0)
        self._created_at = datetime.fromisoformat(
            snapshot.get("created_at", datetime.utcnow().isoformat())
        )
        self._updated_at = datetime.fromisoformat(
            snapshot.get("updated_at", datetime.utcnow().isoformat())
        )

    def get_provenance_hash(self) -> str:
        """
        Calculate provenance hash for current state.

        Returns:
            SHA-256 hash of state
        """
        state_str = json.dumps(self.create_snapshot(), sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def __str__(self) -> str:
        """String representation."""
        return f"{self._aggregate_type}({self._aggregate_id}, v{self._version})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"<{self.__class__.__name__} "
            f"id={self._aggregate_id} "
            f"version={self._version} "
            f"uncommitted={len(self._uncommitted_events)}>"
        )


class AggregateRoot(Aggregate):
    """
    Extended aggregate with additional root entity features.

    AggregateRoot provides additional functionality:
        - Child entity management
        - Invariant validation
        - Domain service integration

    Use this for aggregates that manage child entities.
    """

    def __init__(
        self,
        aggregate_id: str,
        aggregate_type: str = "AggregateRoot"
    ):
        """Initialize aggregate root."""
        super().__init__(aggregate_id, aggregate_type)
        self._child_entities: Dict[str, Any] = {}

    def add_child_entity(self, entity_id: str, entity: Any) -> None:
        """Add a child entity."""
        self._child_entities[entity_id] = entity

    def get_child_entity(self, entity_id: str) -> Optional[Any]:
        """Get a child entity by ID."""
        return self._child_entities.get(entity_id)

    def remove_child_entity(self, entity_id: str) -> bool:
        """Remove a child entity."""
        if entity_id in self._child_entities:
            del self._child_entities[entity_id]
            return True
        return False

    def validate_invariants(self) -> List[str]:
        """
        Validate aggregate invariants.

        Override in subclasses to add business rule validation.

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []

        # Check that version is non-negative
        if self._version < 0:
            violations.append("Version cannot be negative")

        return violations

    def ensure_invariants(self) -> None:
        """
        Ensure invariants are satisfied, raise if not.

        Raises:
            ValueError: If invariants are violated
        """
        violations = self.validate_invariants()
        if violations:
            raise ValueError(
                f"Aggregate invariants violated: {', '.join(violations)}"
            )


class AggregateRepository(Generic[TAggregate]):
    """
    Repository for loading and saving aggregates.

    Provides a clean interface for working with aggregates
    and their event streams.

    Example:
        >>> repo = AggregateRepository(CombustionAggregate, event_store)
        >>> aggregate = await repo.get("burner-001")
        >>> aggregate.change_setpoint(1000, 12500)
        >>> await repo.save(aggregate)
    """

    def __init__(
        self,
        aggregate_class: Type[TAggregate],
        event_store: EventStore,
        snapshot_threshold: int = 100
    ):
        """
        Initialize repository.

        Args:
            aggregate_class: Class of aggregate to manage
            event_store: Event store instance
            snapshot_threshold: Events before auto-snapshot
        """
        self._aggregate_class = aggregate_class
        self._event_store = event_store
        self._snapshot_threshold = snapshot_threshold
        self._cache: Dict[str, TAggregate] = {}

    async def get(
        self,
        aggregate_id: str,
        use_cache: bool = True
    ) -> TAggregate:
        """
        Load an aggregate by ID.

        Args:
            aggregate_id: Aggregate identifier
            use_cache: Use cached instance if available

        Returns:
            Loaded aggregate

        Raises:
            KeyError: If aggregate doesn't exist
        """
        # Check cache
        if use_cache and aggregate_id in self._cache:
            return self._cache[aggregate_id]

        # Check if exists
        exists = await self._event_store.exists(aggregate_id)
        if not exists:
            raise KeyError(f"Aggregate not found: {aggregate_id}")

        # Create new instance
        aggregate = self._aggregate_class(aggregate_id)

        # Try to load from snapshot first
        snapshot = await self._event_store.load_latest_snapshot(aggregate_id)
        from_version = 0

        if snapshot:
            version, snapshot_data = snapshot
            aggregate.restore_from_snapshot(snapshot_data)
            from_version = version + 1
            logger.debug(
                f"Restored {aggregate_id} from snapshot at version {version}"
            )

        # Load remaining events
        events_loaded = await aggregate.load_from_store(
            self._event_store,
            from_version=from_version
        )

        logger.debug(
            f"Loaded aggregate {aggregate_id}: "
            f"snapshot_version={from_version - 1 if snapshot else 'none'}, "
            f"events_applied={events_loaded}"
        )

        # Cache
        if use_cache:
            self._cache[aggregate_id] = aggregate

        return aggregate

    async def get_or_create(
        self,
        aggregate_id: str
    ) -> Tuple[TAggregate, bool]:
        """
        Get existing aggregate or create new one.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            Tuple of (aggregate, is_new)
        """
        exists = await self._event_store.exists(aggregate_id)

        if exists:
            aggregate = await self.get(aggregate_id)
            return (aggregate, False)
        else:
            aggregate = self._aggregate_class(aggregate_id)
            self._cache[aggregate_id] = aggregate
            return (aggregate, True)

    async def save(
        self,
        aggregate: TAggregate,
        expected_version: Optional[int] = None
    ) -> int:
        """
        Save aggregate to event store.

        Args:
            aggregate: Aggregate to save
            expected_version: Expected current version

        Returns:
            New version number
        """
        if not aggregate.uncommitted_events:
            return aggregate.version

        new_version = await aggregate.save_to_store(
            self._event_store,
            expected_version=expected_version
        )

        # Check if snapshot needed
        if new_version % self._snapshot_threshold == 0:
            await self._save_snapshot(aggregate)

        logger.debug(
            f"Saved aggregate {aggregate.aggregate_id} "
            f"at version {new_version}"
        )

        return new_version

    async def _save_snapshot(self, aggregate: TAggregate) -> None:
        """Save a snapshot of the aggregate."""
        snapshot_data = aggregate.create_snapshot()
        await self._event_store.save_snapshot(
            aggregate_id=aggregate.aggregate_id,
            version=aggregate.version,
            snapshot_data=snapshot_data,
            aggregate_type=aggregate.aggregate_type
        )
        logger.info(
            f"Saved snapshot for {aggregate.aggregate_id} "
            f"at version {aggregate.version}"
        )

    def evict_from_cache(self, aggregate_id: str) -> bool:
        """Remove aggregate from cache."""
        if aggregate_id in self._cache:
            del self._cache[aggregate_id]
            return True
        return False

    def clear_cache(self) -> int:
        """Clear all cached aggregates."""
        count = len(self._cache)
        self._cache.clear()
        return count
