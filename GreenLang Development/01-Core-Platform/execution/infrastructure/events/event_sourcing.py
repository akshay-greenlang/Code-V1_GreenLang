"""
Event Sourcing Implementation for GreenLang

This module provides event sourcing capabilities for audit trails
and aggregate state reconstruction.

Features:
- Event store with append-only writes
- Aggregate state reconstruction
- Snapshot support
- Projection management
- Audit trail compliance
- Provenance tracking

Example:
    >>> store = EventStore(config)
    >>> await store.append("emissions-2024", emission_event)
    >>> events = await store.read_stream("emissions-2024")
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.events.event_schema import (
    BaseEvent,
    DomainEvent,
    EventMetadata,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AggregateRoot")
E = TypeVar("E", bound=BaseEvent)


class StreamPosition(str, Enum):
    """Stream position markers."""
    START = "start"
    END = "end"


class ExpectedVersion(int, Enum):
    """Expected version constants."""
    ANY = -1
    NO_STREAM = 0
    STREAM_EXISTS = -2


@dataclass
class EventStoreConfig:
    """Configuration for event store."""
    storage_backend: str = "memory"
    postgres_url: str = "postgresql://localhost/greenlang_events"
    redis_url: str = "redis://localhost:6379"
    snapshot_frequency: int = 100
    enable_snapshots: bool = True
    enable_projections: bool = True
    max_events_per_read: int = 1000
    retention_days: int = 3650  # 10 years for audit


class StoredEvent(BaseModel):
    """Stored event with metadata."""
    event_id: str = Field(..., description="Event identifier")
    stream_id: str = Field(..., description="Stream identifier")
    version: int = Field(..., description="Event version in stream")
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(..., description="Event payload")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="Provenance hash")

    def to_domain_event(self) -> DomainEvent:
        """Convert to domain event."""
        return DomainEvent(
            event_type=self.event_type,
            aggregate_id=self.stream_id,
            aggregate_type=self.metadata.get("aggregate_type", "Unknown"),
            aggregate_version=self.version,
            data=self.event_data,
            metadata=EventMetadata(
                event_id=self.event_id,
                timestamp=self.timestamp,
                correlation_id=self.metadata.get("correlation_id"),
                causation_id=self.metadata.get("causation_id"),
            ),
            provenance_hash=self.provenance_hash,
        )


class Snapshot(BaseModel):
    """Aggregate snapshot."""
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    stream_id: str = Field(..., description="Stream identifier")
    version: int = Field(..., description="Version at snapshot")
    aggregate_type: str = Field(..., description="Aggregate type")
    state: Dict[str, Any] = Field(..., description="Aggregate state")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")


class StreamInfo(BaseModel):
    """Stream information."""
    stream_id: str = Field(..., description="Stream identifier")
    current_version: int = Field(default=0, description="Current version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    event_count: int = Field(default=0, description="Total events")
    snapshot_version: Optional[int] = Field(default=None)


class EventStoreBackend:
    """Base class for event store backends."""

    async def append(
        self,
        stream_id: str,
        events: List[StoredEvent],
        expected_version: int
    ) -> int:
        """Append events to stream."""
        raise NotImplementedError

    async def read_stream(
        self,
        stream_id: str,
        start_version: int = 0,
        count: int = 1000
    ) -> List[StoredEvent]:
        """Read events from stream."""
        raise NotImplementedError

    async def read_all(
        self,
        position: int = 0,
        count: int = 1000
    ) -> List[StoredEvent]:
        """Read all events across streams."""
        raise NotImplementedError

    async def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """Get stream information."""
        raise NotImplementedError

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save aggregate snapshot."""
        raise NotImplementedError

    async def get_snapshot(self, stream_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for stream."""
        raise NotImplementedError


class MemoryEventStoreBackend(EventStoreBackend):
    """In-memory event store backend for testing."""

    def __init__(self):
        """Initialize memory store."""
        self._streams: Dict[str, List[StoredEvent]] = {}
        self._all_events: List[StoredEvent] = []
        self._snapshots: Dict[str, Snapshot] = {}
        self._stream_info: Dict[str, StreamInfo] = {}
        self._lock = asyncio.Lock()

    async def append(
        self,
        stream_id: str,
        events: List[StoredEvent],
        expected_version: int
    ) -> int:
        """Append events to stream."""
        async with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = []
                self._stream_info[stream_id] = StreamInfo(
                    stream_id=stream_id,
                    current_version=0
                )

            current_version = self._stream_info[stream_id].current_version

            # Check expected version
            if expected_version >= 0 and expected_version != current_version:
                raise ValueError(
                    f"Version mismatch: expected {expected_version}, "
                    f"actual {current_version}"
                )

            # Append events
            for i, event in enumerate(events):
                event.version = current_version + i + 1
                self._streams[stream_id].append(event)
                self._all_events.append(event)

            # Update stream info
            new_version = current_version + len(events)
            self._stream_info[stream_id].current_version = new_version
            self._stream_info[stream_id].updated_at = datetime.utcnow()
            self._stream_info[stream_id].event_count += len(events)

            return new_version

    async def read_stream(
        self,
        stream_id: str,
        start_version: int = 0,
        count: int = 1000
    ) -> List[StoredEvent]:
        """Read events from stream."""
        if stream_id not in self._streams:
            return []

        events = self._streams[stream_id]
        filtered = [e for e in events if e.version >= start_version]
        return filtered[:count]

    async def read_all(
        self,
        position: int = 0,
        count: int = 1000
    ) -> List[StoredEvent]:
        """Read all events across streams."""
        return self._all_events[position:position + count]

    async def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """Get stream information."""
        return self._stream_info.get(stream_id)

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save aggregate snapshot."""
        self._snapshots[snapshot.stream_id] = snapshot
        if snapshot.stream_id in self._stream_info:
            self._stream_info[snapshot.stream_id].snapshot_version = snapshot.version

    async def get_snapshot(self, stream_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for stream."""
        return self._snapshots.get(stream_id)


class AggregateRoot(BaseModel):
    """
    Base class for event-sourced aggregates.

    Aggregates are rebuilt from events using the apply method.
    """
    aggregate_id: str = Field(..., description="Aggregate identifier")
    version: int = Field(default=0, description="Current version")
    uncommitted_events: List[DomainEvent] = Field(
        default_factory=list,
        exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    def apply(self, event: DomainEvent) -> None:
        """
        Apply an event to update aggregate state.

        Override in subclasses to handle specific events.
        """
        pass

    def raise_event(self, event: DomainEvent) -> None:
        """
        Raise a new domain event.

        Args:
            event: Event to raise
        """
        event.aggregate_version = self.version + 1
        self.apply(event)
        self.uncommitted_events.append(event)
        self.version += 1

    def load_from_history(self, events: List[DomainEvent]) -> None:
        """
        Load aggregate from event history.

        Args:
            events: Historical events
        """
        for event in events:
            self.apply(event)
            self.version = event.aggregate_version

    def clear_uncommitted_events(self) -> List[DomainEvent]:
        """
        Clear and return uncommitted events.

        Returns:
            List of uncommitted events
        """
        events = list(self.uncommitted_events)
        self.uncommitted_events.clear()
        return events

    def to_snapshot_state(self) -> Dict[str, Any]:
        """
        Get state for snapshotting.

        Override to customize snapshot data.
        """
        return self.dict(exclude={"uncommitted_events"})

    @classmethod
    def from_snapshot(cls: Type[T], state: Dict[str, Any]) -> T:
        """
        Reconstruct from snapshot state.

        Args:
            state: Snapshot state

        Returns:
            Aggregate instance
        """
        return cls(**state)


class Projection(BaseModel):
    """
    Base class for event projections.

    Projections build read models from events.
    """
    projection_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Projection name")
    position: int = Field(default=0, description="Current position")
    state: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def handle(self, event: StoredEvent) -> None:
        """
        Handle an event.

        Override in subclasses to update projection state.
        """
        pass


class EventStore:
    """
    Event store for event sourcing.

    Provides append-only event storage with aggregate reconstruction
    and snapshot support.

    Attributes:
        config: Store configuration
        backend: Storage backend

    Example:
        >>> store = EventStore(config)
        >>> await store.start()
        >>> await store.save_aggregate(emission_aggregate)
        >>> loaded = await store.load_aggregate("emission-001", EmissionAggregate)
    """

    def __init__(self, config: EventStoreConfig):
        """
        Initialize event store.

        Args:
            config: Store configuration
        """
        self.config = config
        self._backend: Optional[EventStoreBackend] = None
        self._started = False
        self._projections: Dict[str, Projection] = {}
        self._projection_handlers: Dict[str, Callable] = {}
        self._projection_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info("EventStore initialized")

    async def start(self) -> None:
        """
        Start the event store.

        Initializes backend and starts projection processing.
        """
        if self._started:
            logger.warning("Event store already started")
            return

        try:
            self._backend = self._create_backend()

            if self.config.enable_projections:
                self._projection_task = asyncio.create_task(
                    self._projection_loop()
                )

            self._started = True
            self._shutdown = False

            logger.info("Event store started")

        except Exception as e:
            logger.error(f"Failed to start event store: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the event store gracefully.
        """
        self._shutdown = True

        if self._projection_task:
            self._projection_task.cancel()
            try:
                await self._projection_task
            except asyncio.CancelledError:
                pass

        self._started = False
        logger.info("Event store stopped")

    def _create_backend(self) -> EventStoreBackend:
        """Create storage backend."""
        if self.config.storage_backend == "memory":
            return MemoryEventStoreBackend()
        return MemoryEventStoreBackend()

    async def append(
        self,
        stream_id: str,
        event: BaseEvent,
        expected_version: int = ExpectedVersion.ANY
    ) -> int:
        """
        Append a single event to a stream.

        Args:
            stream_id: Stream identifier
            event: Event to append
            expected_version: Expected stream version

        Returns:
            New stream version
        """
        self._ensure_started()

        stored_event = self._create_stored_event(stream_id, event, 0)
        new_version = await self._backend.append(
            stream_id,
            [stored_event],
            expected_version
        )

        logger.debug(f"Appended event to stream {stream_id}, version {new_version}")
        return new_version

    async def append_batch(
        self,
        stream_id: str,
        events: List[BaseEvent],
        expected_version: int = ExpectedVersion.ANY
    ) -> int:
        """
        Append multiple events to a stream.

        Args:
            stream_id: Stream identifier
            events: Events to append
            expected_version: Expected stream version

        Returns:
            New stream version
        """
        self._ensure_started()

        stored_events = [
            self._create_stored_event(stream_id, event, i)
            for i, event in enumerate(events)
        ]

        new_version = await self._backend.append(
            stream_id,
            stored_events,
            expected_version
        )

        logger.debug(
            f"Appended {len(events)} events to stream {stream_id}, "
            f"version {new_version}"
        )
        return new_version

    def _create_stored_event(
        self,
        stream_id: str,
        event: BaseEvent,
        order: int
    ) -> StoredEvent:
        """Create stored event from base event."""
        provenance_str = f"{stream_id}:{event.event_type}:{event.metadata.event_id}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return StoredEvent(
            event_id=event.metadata.event_id,
            stream_id=stream_id,
            version=0,  # Set by backend
            event_type=event.event_type,
            event_data=event.data,
            metadata={
                "correlation_id": event.metadata.correlation_id,
                "causation_id": event.metadata.causation_id,
                "source": event.metadata.source,
            },
            timestamp=event.metadata.timestamp,
            provenance_hash=provenance_hash,
        )

    async def read_stream(
        self,
        stream_id: str,
        start_version: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """
        Read events from a stream.

        Args:
            stream_id: Stream identifier
            start_version: Starting version
            count: Maximum events to read

        Returns:
            List of stored events
        """
        self._ensure_started()

        return await self._backend.read_stream(
            stream_id,
            start_version,
            count or self.config.max_events_per_read
        )

    async def read_all(
        self,
        position: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """
        Read all events across streams.

        Args:
            position: Starting position
            count: Maximum events to read

        Returns:
            List of stored events
        """
        self._ensure_started()

        return await self._backend.read_all(
            position,
            count or self.config.max_events_per_read
        )

    async def save_aggregate(
        self,
        aggregate: AggregateRoot,
        expected_version: Optional[int] = None
    ) -> int:
        """
        Save an aggregate's uncommitted events.

        Args:
            aggregate: Aggregate to save
            expected_version: Expected version

        Returns:
            New version
        """
        self._ensure_started()

        events = aggregate.clear_uncommitted_events()
        if not events:
            return aggregate.version

        version = expected_version if expected_version is not None else aggregate.version - len(events)

        # Convert domain events to base events
        base_events = [
            BaseEvent(
                event_type=e.event_type,
                metadata=e.metadata,
                data=e.data,
            )
            for e in events
        ]

        new_version = await self.append_batch(
            aggregate.aggregate_id,
            base_events,
            version
        )

        # Create snapshot if needed
        if self.config.enable_snapshots:
            if new_version % self.config.snapshot_frequency == 0:
                await self._create_snapshot(aggregate, new_version)

        return new_version

    async def load_aggregate(
        self,
        aggregate_id: str,
        aggregate_type: Type[T]
    ) -> Optional[T]:
        """
        Load an aggregate from events.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Aggregate class

        Returns:
            Loaded aggregate or None
        """
        self._ensure_started()

        # Try to load from snapshot first
        snapshot = None
        start_version = 0

        if self.config.enable_snapshots:
            snapshot = await self._backend.get_snapshot(aggregate_id)
            if snapshot and snapshot.aggregate_type == aggregate_type.__name__:
                start_version = snapshot.version + 1

        # Load events
        events = await self._backend.read_stream(aggregate_id, start_version)

        if not events and not snapshot:
            return None

        # Reconstruct aggregate
        if snapshot:
            aggregate = aggregate_type.from_snapshot(snapshot.state)
        else:
            aggregate = aggregate_type(aggregate_id=aggregate_id)

        # Apply events
        domain_events = [e.to_domain_event() for e in events]
        aggregate.load_from_history(domain_events)

        return aggregate

    async def _create_snapshot(
        self,
        aggregate: AggregateRoot,
        version: int
    ) -> None:
        """Create snapshot for aggregate."""
        state = aggregate.to_snapshot_state()
        provenance_str = f"{aggregate.aggregate_id}:{version}:{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        snapshot = Snapshot(
            stream_id=aggregate.aggregate_id,
            version=version,
            aggregate_type=type(aggregate).__name__,
            state=state,
            provenance_hash=provenance_hash,
        )

        await self._backend.save_snapshot(snapshot)
        logger.info(f"Created snapshot for {aggregate.aggregate_id} at version {version}")

    def register_projection(
        self,
        projection: Projection,
        handler: Callable[[Projection, StoredEvent], None]
    ) -> None:
        """
        Register a projection.

        Args:
            projection: Projection to register
            handler: Event handler function
        """
        self._projections[projection.name] = projection
        self._projection_handlers[projection.name] = handler
        logger.info(f"Registered projection: {projection.name}")

    async def _projection_loop(self) -> None:
        """Background loop for projection processing."""
        while not self._shutdown:
            try:
                await asyncio.sleep(1)  # Process projections every second

                for name, projection in self._projections.items():
                    handler = self._projection_handlers.get(name)
                    if not handler:
                        continue

                    # Read new events
                    events = await self._backend.read_all(
                        projection.position,
                        count=100
                    )

                    for event in events:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(projection, event)
                            else:
                                handler(projection, event)
                            projection.position += 1
                        except Exception as e:
                            logger.error(f"Projection handler error: {e}")

                    projection.last_updated = datetime.utcnow()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Projection loop error: {e}")

    async def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """
        Get stream information.

        Args:
            stream_id: Stream identifier

        Returns:
            Stream info or None
        """
        self._ensure_started()
        return await self._backend.get_stream_info(stream_id)

    def _ensure_started(self) -> None:
        """Ensure store is started."""
        if not self._started:
            raise RuntimeError("Event store not started")

    async def __aenter__(self) -> "EventStore":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
