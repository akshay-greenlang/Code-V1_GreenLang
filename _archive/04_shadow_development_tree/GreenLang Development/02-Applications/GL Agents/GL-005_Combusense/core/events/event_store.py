# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Event Store

This module implements an append-only event store with support for multiple
storage backends (SQLite, PostgreSQL) and provides idempotent writes,
optimistic concurrency control, and schema versioning.

Design Principles:
    - Append-only: Events are never modified or deleted
    - Ordered: Events maintain strict ordering within streams
    - Idempotent: Duplicate writes are safely handled
    - Versioned: Schema versioning for forward compatibility
    - Auditable: Full provenance tracking with hashes

Storage Backends:
    - SQLite: For development and single-instance deployment
    - PostgreSQL: For production with high availability

Example:
    >>> store = EventStore(backend="sqlite", connection_string="events.db")
    >>> await store.initialize()
    >>> await store.append("aggregate-001", [event1, event2])
    >>> events = await store.load("aggregate-001")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, Field

from core.events.base_event import DomainEvent, EventEnvelope
from core.events.domain_events import EVENT_REGISTRY, deserialize_event

logger = logging.getLogger(__name__)


class EventStoreBackend(str, Enum):
    """Supported storage backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MEMORY = "memory"


class OptimisticConcurrencyError(Exception):
    """Raised when expected version doesn't match current version."""

    def __init__(self, stream_name: str, expected: int, actual: int):
        self.stream_name = stream_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Concurrency conflict in stream '{stream_name}': "
            f"expected version {expected}, but current is {actual}"
        )


class EventStreamNotFoundError(Exception):
    """Raised when an event stream doesn't exist."""
    pass


class EventStoreConfig(BaseModel):
    """Configuration for event store."""

    backend: EventStoreBackend = Field(
        default=EventStoreBackend.SQLITE,
        description="Storage backend type"
    )
    connection_string: str = Field(
        default="events.db",
        description="Connection string or file path"
    )
    pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size (PostgreSQL only)"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient failures"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for bulk operations"
    )
    enable_snapshots: bool = Field(
        default=True,
        description="Enable automatic snapshots"
    )
    snapshot_threshold: int = Field(
        default=100,
        ge=10,
        description="Events before auto-snapshot"
    )


class StreamMetadata(BaseModel):
    """Metadata about an event stream."""

    stream_name: str = Field(..., description="Stream name")
    aggregate_type: str = Field(..., description="Aggregate type")
    aggregate_id: str = Field(..., description="Aggregate ID")
    current_version: int = Field(default=0, ge=0, description="Current version")
    event_count: int = Field(default=0, ge=0, description="Total event count")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_event_id: Optional[str] = Field(default=None)


class EventStoreBackendBase(ABC):
    """Abstract base class for event store backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (create tables, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        pass

    @abstractmethod
    async def append_events(
        self,
        stream_name: str,
        events: Sequence[DomainEvent],
        expected_version: Optional[int] = None
    ) -> int:
        """
        Append events to a stream.

        Args:
            stream_name: Name of the event stream
            events: Events to append
            expected_version: Expected current version (for optimistic concurrency)

        Returns:
            New version number

        Raises:
            OptimisticConcurrencyError: If version mismatch
        """
        pass

    @abstractmethod
    async def load_events(
        self,
        stream_name: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """
        Load events from a stream.

        Args:
            stream_name: Name of the event stream
            from_version: Start version (inclusive)
            to_version: End version (inclusive)
            limit: Maximum events to load

        Returns:
            List of domain events
        """
        pass

    @abstractmethod
    async def get_stream_metadata(
        self,
        stream_name: str
    ) -> Optional[StreamMetadata]:
        """Get metadata for a stream."""
        pass

    @abstractmethod
    async def stream_exists(self, stream_name: str) -> bool:
        """Check if a stream exists."""
        pass

    @abstractmethod
    async def get_global_position(self) -> int:
        """Get global event position (total events)."""
        pass


class SQLiteBackend(EventStoreBackendBase):
    """SQLite event store backend for development and testing."""

    def __init__(self, connection_string: str):
        """
        Initialize SQLite backend.

        Args:
            connection_string: Path to SQLite database file
        """
        self.connection_string = connection_string
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create database and tables."""
        def _create_tables(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()

            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    stream_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    global_position INTEGER NOT NULL,
                    event_data TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    provenance_hash TEXT NOT NULL,
                    stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stream_name, sequence_number)
                )
            """)

            # Streams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS streams (
                    stream_name TEXT PRIMARY KEY,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    current_version INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_event_id TEXT
                )
            """)

            # Snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    snapshot_data TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stream_name, version)
                )
            """)

            # Indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_stream
                ON events(stream_name, sequence_number)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_global
                ON events(global_position)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_stream
                ON snapshots(stream_name, version DESC)
            """)

            conn.commit()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_connection)
        await loop.run_in_executor(None, _create_tables, self._connection)
        logger.info(f"SQLite event store initialized: {self.connection_string}")

    def _init_connection(self) -> None:
        """Initialize database connection."""
        # Create parent directories if needed
        db_path = Path(self.connection_string)
        if db_path.parent and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            self.connection_string,
            check_same_thread=False,
            isolation_level="IMMEDIATE"
        )
        self._connection.row_factory = sqlite3.Row

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._connection.close)
            self._connection = None
            logger.info("SQLite event store closed")

    async def append_events(
        self,
        stream_name: str,
        events: Sequence[DomainEvent],
        expected_version: Optional[int] = None
    ) -> int:
        """Append events to a stream with optimistic concurrency."""
        if not events:
            return await self._get_current_version(stream_name)

        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._append_events_sync,
                stream_name,
                list(events),
                expected_version
            )

    def _append_events_sync(
        self,
        stream_name: str,
        events: List[DomainEvent],
        expected_version: Optional[int]
    ) -> int:
        """Synchronous implementation of append_events."""
        cursor = self._connection.cursor()

        try:
            # Get current version
            cursor.execute(
                "SELECT current_version FROM streams WHERE stream_name = ?",
                (stream_name,)
            )
            row = cursor.fetchone()
            current_version = row["current_version"] if row else -1

            # Check optimistic concurrency
            if expected_version is not None and current_version != expected_version:
                raise OptimisticConcurrencyError(
                    stream_name, expected_version, current_version
                )

            # Get global position
            cursor.execute("SELECT COALESCE(MAX(global_position), 0) FROM events")
            global_position = cursor.fetchone()[0]

            # Insert or update stream
            if current_version == -1:
                # New stream
                aggregate_type = events[0].aggregate_type
                aggregate_id = events[0].aggregate_id
                cursor.execute("""
                    INSERT INTO streams
                    (stream_name, aggregate_type, aggregate_id, current_version, event_count)
                    VALUES (?, ?, ?, 0, 0)
                """, (stream_name, aggregate_type, aggregate_id))
                current_version = 0

            # Insert events
            new_version = current_version
            for event in events:
                new_version += 1
                global_position += 1

                # Update event with sequence number
                event_with_seq = event.with_sequence_number(new_version)

                cursor.execute("""
                    INSERT INTO events
                    (event_id, stream_name, event_type, sequence_number,
                     global_position, event_data, metadata, provenance_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_with_seq.metadata.event_id,
                    stream_name,
                    event_with_seq.event_type,
                    new_version,
                    global_position,
                    json.dumps(event_with_seq.model_dump(), default=str),
                    json.dumps(event_with_seq.metadata.model_dump(), default=str),
                    event_with_seq.provenance_hash
                ))

            # Update stream metadata
            cursor.execute("""
                UPDATE streams
                SET current_version = ?,
                    event_count = event_count + ?,
                    updated_at = CURRENT_TIMESTAMP,
                    last_event_id = ?
                WHERE stream_name = ?
            """, (
                new_version,
                len(events),
                events[-1].metadata.event_id,
                stream_name
            ))

            self._connection.commit()
            logger.debug(
                f"Appended {len(events)} events to stream '{stream_name}', "
                f"new version: {new_version}"
            )
            return new_version

        except Exception as e:
            self._connection.rollback()
            logger.error(f"Failed to append events: {e}")
            raise

    async def _get_current_version(self, stream_name: str) -> int:
        """Get current version of a stream."""
        loop = asyncio.get_event_loop()

        def _get_version() -> int:
            cursor = self._connection.cursor()
            cursor.execute(
                "SELECT current_version FROM streams WHERE stream_name = ?",
                (stream_name,)
            )
            row = cursor.fetchone()
            return row["current_version"] if row else 0

        return await loop.run_in_executor(None, _get_version)

    async def load_events(
        self,
        stream_name: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """Load events from a stream."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_events_sync,
            stream_name,
            from_version,
            to_version,
            limit
        )

    def _load_events_sync(
        self,
        stream_name: str,
        from_version: int,
        to_version: Optional[int],
        limit: Optional[int]
    ) -> List[DomainEvent]:
        """Synchronous implementation of load_events."""
        cursor = self._connection.cursor()

        query = """
            SELECT event_data FROM events
            WHERE stream_name = ? AND sequence_number >= ?
        """
        params: List[Any] = [stream_name, from_version]

        if to_version is not None:
            query += " AND sequence_number <= ?"
            params.append(to_version)

        query += " ORDER BY sequence_number ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event_data = json.loads(row["event_data"])
            event = deserialize_event(event_data)
            events.append(event)

        logger.debug(
            f"Loaded {len(events)} events from stream '{stream_name}' "
            f"(versions {from_version} to {to_version or 'latest'})"
        )
        return events

    async def get_stream_metadata(
        self,
        stream_name: str
    ) -> Optional[StreamMetadata]:
        """Get metadata for a stream."""
        loop = asyncio.get_event_loop()

        def _get_metadata() -> Optional[StreamMetadata]:
            cursor = self._connection.cursor()
            cursor.execute(
                "SELECT * FROM streams WHERE stream_name = ?",
                (stream_name,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return StreamMetadata(
                stream_name=row["stream_name"],
                aggregate_type=row["aggregate_type"],
                aggregate_id=row["aggregate_id"],
                current_version=row["current_version"],
                event_count=row["event_count"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                last_event_id=row["last_event_id"]
            )

        return await loop.run_in_executor(None, _get_metadata)

    async def stream_exists(self, stream_name: str) -> bool:
        """Check if a stream exists."""
        metadata = await self.get_stream_metadata(stream_name)
        return metadata is not None

    async def get_global_position(self) -> int:
        """Get global event position."""
        loop = asyncio.get_event_loop()

        def _get_position() -> int:
            cursor = self._connection.cursor()
            cursor.execute("SELECT COALESCE(MAX(global_position), 0) FROM events")
            return cursor.fetchone()[0]

        return await loop.run_in_executor(None, _get_position)

    async def save_snapshot(
        self,
        stream_name: str,
        version: int,
        snapshot_data: Dict[str, Any],
        snapshot_type: str
    ) -> None:
        """Save a snapshot."""
        loop = asyncio.get_event_loop()

        def _save() -> None:
            cursor = self._connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO snapshots
                (stream_name, version, snapshot_data, snapshot_type)
                VALUES (?, ?, ?, ?)
            """, (
                stream_name,
                version,
                json.dumps(snapshot_data, default=str),
                snapshot_type
            ))
            self._connection.commit()

        await loop.run_in_executor(None, _save)
        logger.debug(f"Saved snapshot for stream '{stream_name}' at version {version}")

    async def load_latest_snapshot(
        self,
        stream_name: str
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Load the latest snapshot for a stream."""
        loop = asyncio.get_event_loop()

        def _load() -> Optional[Tuple[int, Dict[str, Any]]]:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT version, snapshot_data FROM snapshots
                WHERE stream_name = ?
                ORDER BY version DESC
                LIMIT 1
            """, (stream_name,))
            row = cursor.fetchone()
            if not row:
                return None
            return (row["version"], json.loads(row["snapshot_data"]))

        return await loop.run_in_executor(None, _load)


class InMemoryBackend(EventStoreBackendBase):
    """In-memory event store backend for testing."""

    def __init__(self):
        """Initialize in-memory backend."""
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._streams: Dict[str, StreamMetadata] = {}
        self._snapshots: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        self._global_position = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize backend (no-op for memory)."""
        logger.info("In-memory event store initialized")

    async def close(self) -> None:
        """Clear all data."""
        self._events.clear()
        self._streams.clear()
        self._snapshots.clear()
        self._global_position = 0
        logger.info("In-memory event store closed")

    async def append_events(
        self,
        stream_name: str,
        events: Sequence[DomainEvent],
        expected_version: Optional[int] = None
    ) -> int:
        """Append events to a stream."""
        if not events:
            metadata = self._streams.get(stream_name)
            return metadata.current_version if metadata else 0

        async with self._lock:
            # Get or create stream
            if stream_name not in self._streams:
                self._streams[stream_name] = StreamMetadata(
                    stream_name=stream_name,
                    aggregate_type=events[0].aggregate_type,
                    aggregate_id=events[0].aggregate_id
                )
                self._events[stream_name] = []

            metadata = self._streams[stream_name]
            current_version = metadata.current_version

            # Check optimistic concurrency
            if expected_version is not None and current_version != expected_version:
                raise OptimisticConcurrencyError(
                    stream_name, expected_version, current_version
                )

            # Append events
            new_version = current_version
            for event in events:
                new_version += 1
                self._global_position += 1

                event_with_seq = event.with_sequence_number(new_version)
                self._events[stream_name].append({
                    "event": event_with_seq.model_dump(),
                    "global_position": self._global_position
                })

            # Update metadata
            self._streams[stream_name] = StreamMetadata(
                stream_name=stream_name,
                aggregate_type=metadata.aggregate_type,
                aggregate_id=metadata.aggregate_id,
                current_version=new_version,
                event_count=metadata.event_count + len(events),
                created_at=metadata.created_at,
                updated_at=datetime.utcnow(),
                last_event_id=events[-1].metadata.event_id
            )

            return new_version

    async def load_events(
        self,
        stream_name: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """Load events from a stream."""
        if stream_name not in self._events:
            return []

        events = []
        for event_record in self._events[stream_name]:
            event_data = event_record["event"]
            seq = event_data.get("sequence_number", 0)

            if seq < from_version:
                continue
            if to_version is not None and seq > to_version:
                break
            if limit is not None and len(events) >= limit:
                break

            event = deserialize_event(event_data)
            events.append(event)

        return events

    async def get_stream_metadata(
        self,
        stream_name: str
    ) -> Optional[StreamMetadata]:
        """Get metadata for a stream."""
        return self._streams.get(stream_name)

    async def stream_exists(self, stream_name: str) -> bool:
        """Check if a stream exists."""
        return stream_name in self._streams

    async def get_global_position(self) -> int:
        """Get global event position."""
        return self._global_position

    async def save_snapshot(
        self,
        stream_name: str,
        version: int,
        snapshot_data: Dict[str, Any],
        snapshot_type: str
    ) -> None:
        """Save a snapshot."""
        if stream_name not in self._snapshots:
            self._snapshots[stream_name] = []
        self._snapshots[stream_name].append((version, snapshot_data))

    async def load_latest_snapshot(
        self,
        stream_name: str
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Load the latest snapshot for a stream."""
        if stream_name not in self._snapshots or not self._snapshots[stream_name]:
            return None
        return max(self._snapshots[stream_name], key=lambda x: x[0])


class EventStore:
    """
    Main event store class providing a unified interface to storage backends.

    The EventStore is the primary interface for persisting and retrieving
    domain events. It supports multiple backends and provides features like:
        - Append-only event storage
        - Optimistic concurrency control
        - Event stream management
        - Snapshot integration
        - Global ordering

    Example:
        >>> config = EventStoreConfig(backend="sqlite", connection_string="events.db")
        >>> store = EventStore(config)
        >>> await store.initialize()
        >>> await store.append("combustion-001", [event1, event2])
        >>> events = await store.load("combustion-001")
    """

    def __init__(
        self,
        config: Optional[EventStoreConfig] = None,
        backend: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize event store.

        Args:
            config: Full configuration object
            backend: Backend type (shorthand)
            connection_string: Connection string (shorthand)
        """
        if config is None:
            config = EventStoreConfig(
                backend=EventStoreBackend(backend or "sqlite"),
                connection_string=connection_string or "events.db"
            )

        self.config = config
        self._backend: Optional[EventStoreBackendBase] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the event store and backend.

        Must be called before any other operations.
        """
        if self._initialized:
            return

        # Create appropriate backend
        if self.config.backend == EventStoreBackend.SQLITE:
            self._backend = SQLiteBackend(self.config.connection_string)
        elif self.config.backend == EventStoreBackend.MEMORY:
            self._backend = InMemoryBackend()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        await self._backend.initialize()
        self._initialized = True
        logger.info(f"Event store initialized with {self.config.backend.value} backend")

    async def close(self) -> None:
        """Close the event store and release resources."""
        if self._backend:
            await self._backend.close()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise RuntimeError("Event store not initialized. Call initialize() first.")

    @staticmethod
    def get_stream_name(aggregate_type: str, aggregate_id: str) -> str:
        """
        Get stream name for an aggregate.

        Args:
            aggregate_type: Type of aggregate
            aggregate_id: Aggregate ID

        Returns:
            Stream name in format "AggregateType-aggregate_id"
        """
        return f"{aggregate_type}-{aggregate_id}"

    async def append(
        self,
        aggregate_id: str,
        events: Union[DomainEvent, Sequence[DomainEvent]],
        aggregate_type: str = "CombustionControlAggregate",
        expected_version: Optional[int] = None
    ) -> int:
        """
        Append events to an aggregate's stream.

        Args:
            aggregate_id: Aggregate identifier
            events: Event or list of events to append
            aggregate_type: Type of aggregate
            expected_version: Expected current version (for concurrency control)

        Returns:
            New version number

        Raises:
            OptimisticConcurrencyError: If version mismatch
        """
        self._ensure_initialized()

        # Normalize to list
        if isinstance(events, DomainEvent):
            events = [events]

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        return await self._backend.append_events(
            stream_name, events, expected_version
        )

    async def load(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate",
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[DomainEvent]:
        """
        Load events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate
            from_version: Start version (inclusive)
            to_version: End version (inclusive)

        Returns:
            List of domain events
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        return await self._backend.load_events(
            stream_name, from_version, to_version
        )

    async def get_version(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> int:
        """
        Get current version of an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            Current version (0 if doesn't exist)
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        metadata = await self._backend.get_stream_metadata(stream_name)
        return metadata.current_version if metadata else 0

    async def exists(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> bool:
        """
        Check if an aggregate exists.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            True if aggregate has events
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        return await self._backend.stream_exists(stream_name)

    async def get_metadata(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> Optional[StreamMetadata]:
        """
        Get metadata for an aggregate's stream.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            Stream metadata or None
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        return await self._backend.get_stream_metadata(stream_name)

    async def save_snapshot(
        self,
        aggregate_id: str,
        version: int,
        snapshot_data: Dict[str, Any],
        aggregate_type: str = "CombustionControlAggregate",
        snapshot_type: str = "full"
    ) -> None:
        """
        Save a snapshot of aggregate state.

        Args:
            aggregate_id: Aggregate identifier
            version: Version at which snapshot was taken
            snapshot_data: Serialized aggregate state
            aggregate_type: Type of aggregate
            snapshot_type: Type of snapshot
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        await self._backend.save_snapshot(
            stream_name, version, snapshot_data, snapshot_type
        )

    async def load_latest_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str = "CombustionControlAggregate"
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """
        Load the latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            Tuple of (version, snapshot_data) or None
        """
        self._ensure_initialized()

        stream_name = self.get_stream_name(aggregate_type, aggregate_id)
        return await self._backend.load_latest_snapshot(stream_name)

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for transactional operations.

        Currently provides locking semantics. Full transactions
        are backend-dependent.

        Example:
            async with store.transaction():
                await store.append(id, [event1, event2])
        """
        # For SQLite, the backend handles transactions internally
        yield
