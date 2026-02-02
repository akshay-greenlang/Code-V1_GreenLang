# -*- coding: utf-8 -*-
"""
Hash-Chained Audit Event Store for GL-FOUND-X-001

This module implements the append-only event store with tamper-evidence via
SHA-256 hash chaining. Every significant action in the orchestrator emits a
RunEvent that is cryptographically linked to the previous event, forming an
immutable audit trail.

Features:
- Append-only event storage with hash chaining
- SHA-256 cryptographic verification
- PostgreSQL persistence with asyncpg
- In-memory store for testing
- Audit package export with optional signing
- Chain integrity verification

Hash Chain Algorithm:
    event_hash = SHA256(event_id + prev_event_hash + event_type + timestamp + payload_json)
    First event: prev_event_hash = "genesis"

Example:
    >>> store = PostgresEventStore(database_url)
    >>> await store.initialize()
    >>>
    >>> event = await store.append(RunEvent(
    ...     event_id=str(uuid4()),
    ...     run_id="run-123",
    ...     event_type=EventType.RUN_SUBMITTED,
    ...     timestamp=datetime.now(timezone.utc),
    ...     payload={"workflow": "carbon-calc"},
    ...     prev_event_hash="genesis",
    ...     event_hash=""  # Will be computed
    ... ))
    >>>
    >>> is_valid = await store.verify_chain("run-123")
    >>> assert is_valid

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Hash-chained audit trail
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# Use canonical serialization for deterministic hashing
try:
    from greenlang.utilities.serialization.canonical import canonical_dumps
except ImportError:
    import json
    def canonical_dumps(obj: Any) -> str:
        """Fallback canonical JSON serialization."""
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)

# Use deterministic clock for consistent timestamps
try:
    from greenlang.utilities.determinism.clock import DeterministicClock
except ImportError:
    class DeterministicClock:
        """Fallback clock implementation."""
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime.now(tz or timezone.utc).replace(microsecond=0)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

GENESIS_HASH = "genesis"
HASH_ALGORITHM = "sha256"


# ============================================================================
# EVENT TYPE ENUMERATION
# ============================================================================

class EventType(str, Enum):
    """
    Enumeration of all audit event types in the orchestrator lifecycle.

    Events follow the run lifecycle:
    1. RUN_SUBMITTED - Run created and queued
    2. PLAN_COMPILED - DAG compiled from pipeline YAML
    3. POLICY_EVALUATED - Governance policies checked
    4. STEP_READY - Step dependencies satisfied
    5. STEP_STARTED - Step execution began
    6. STEP_RETRIED - Step retried after failure
    7. STEP_SUCCEEDED - Step completed successfully
    8. STEP_FAILED - Step failed (terminal)
    9. ARTIFACT_WRITTEN - Output artifact stored
    10. RUN_SUCCEEDED - All steps completed
    11. RUN_FAILED - Run failed (terminal)
    12. RUN_CANCELED - Run canceled by user
    """
    # Run lifecycle events
    RUN_SUBMITTED = "RUN_SUBMITTED"
    RUN_STARTED = "RUN_STARTED"
    RUN_SUCCEEDED = "RUN_SUCCEEDED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELED = "RUN_CANCELED"
    RUN_COMPLETED = "RUN_COMPLETED"

    # Planning events
    PLAN_COMPILED = "PLAN_COMPILED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    POLICY_VIOLATION = "POLICY_VIOLATION"

    # Step lifecycle events
    STEP_READY = "STEP_READY"
    STEP_STARTED = "STEP_STARTED"
    STEP_RETRIED = "STEP_RETRIED"
    STEP_SUCCEEDED = "STEP_SUCCEEDED"
    STEP_FAILED = "STEP_FAILED"
    STEP_COMPLETED = "STEP_COMPLETED"

    # Artifact events
    ARTIFACT_WRITTEN = "ARTIFACT_WRITTEN"
    ARTIFACT_READ = "ARTIFACT_READ"

    # Approval gate events (P1)
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECISION = "APPROVAL_DECISION"
    APPROVAL_TIMEOUT = "APPROVAL_TIMEOUT"

    # Fan-out events (P1)
    FANOUT_STARTED = "FANOUT_STARTED"
    FANOUT_ITEM_COMPLETED = "FANOUT_ITEM_COMPLETED"
    FANOUT_COMPLETED = "FANOUT_COMPLETED"

    # Concurrency events (P1)
    CONCURRENCY_SLOT_ACQUIRED = "CONCURRENCY_SLOT_ACQUIRED"
    CONCURRENCY_SLOT_RELEASED = "CONCURRENCY_SLOT_RELEASED"
    CONCURRENCY_QUEUE_TIMEOUT = "CONCURRENCY_QUEUE_TIMEOUT"


# ============================================================================
# EVENT MODEL
# ============================================================================

class RunEvent(BaseModel):
    """
    Immutable audit event with hash chain linkage.

    Each event contains a cryptographic link to the previous event,
    forming a tamper-evident chain. Any modification to a past event
    will invalidate all subsequent hashes.

    Attributes:
        event_id: Unique identifier (UUID)
        run_id: Associated run identifier
        step_id: Optional step identifier (for step-level events)
        event_type: Type of event from EventType enum
        timestamp: UTC timestamp of event creation
        payload: Event-specific data dictionary
        prev_event_hash: SHA-256 hash of previous event (or "genesis")
        event_hash: SHA-256 hash of this event (computed)

    Example:
        >>> event = RunEvent(
        ...     event_id="550e8400-e29b-41d4-a716-446655440000",
        ...     run_id="run-2024-001",
        ...     event_type=EventType.STEP_STARTED,
        ...     timestamp=datetime.now(timezone.utc),
        ...     payload={"agent": "CarbonCalculator", "inputs": {"fuel": "natural_gas"}},
        ...     prev_event_hash="abc123...",
        ...     event_hash=""  # Will be computed
        ... )
    """
    event_id: str = Field(
        ...,
        description="Unique event identifier (UUID)",
        min_length=1
    )
    run_id: str = Field(
        ...,
        description="Associated run identifier",
        min_length=1
    )
    step_id: Optional[str] = Field(
        None,
        description="Optional step identifier for step-level events"
    )
    event_type: EventType = Field(
        ...,
        description="Type of audit event"
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of event creation"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data dictionary"
    )
    prev_event_hash: str = Field(
        ...,
        description="SHA-256 hash of previous event or 'genesis'"
    )
    event_hash: str = Field(
        "",
        description="SHA-256 hash of this event (computed)"
    )

    model_config = {
        "frozen": False,  # Allow event_hash to be set after creation
        "extra": "forbid"
    }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC and has no microseconds for determinism."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.replace(microsecond=0)

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this event.

        Hash is computed from:
            event_id + prev_event_hash + event_type + timestamp_iso + payload_canonical_json

        Returns:
            SHA-256 hexadecimal hash string

        Example:
            >>> event.event_hash = event.compute_hash()
        """
        # Build hash input string deterministically
        hash_input = (
            f"{self.event_id}"
            f"{self.prev_event_hash}"
            f"{self.event_type.value}"
            f"{self.timestamp.isoformat()}"
            f"{canonical_dumps(self.payload)}"
        )
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def verify_hash(self) -> bool:
        """
        Verify that the stored event_hash matches the computed hash.

        Returns:
            True if hash is valid, False if tampered

        Example:
            >>> if not event.verify_hash():
            ...     raise TamperDetectedError("Event hash mismatch")
        """
        expected_hash = self.compute_hash()
        return self.event_hash == expected_hash


# ============================================================================
# AUDIT PACKAGE MODEL
# ============================================================================

class AuditPackage(BaseModel):
    """
    Exportable audit trail package for a run.

    Contains all events for a run with chain verification status
    and optional cryptographic signature.

    Attributes:
        run_id: Run identifier
        events: List of all events for the run
        chain_valid: Whether the hash chain is valid
        exported_at: UTC timestamp of export
        signature: Optional Ed25519/RSA signature
        metadata: Additional export metadata

    Example:
        >>> package = await store.export_audit_package("run-123")
        >>> print(f"Chain valid: {package.chain_valid}")
        >>> print(f"Events: {len(package.events)}")
    """
    run_id: str = Field(
        ...,
        description="Run identifier"
    )
    events: List[RunEvent] = Field(
        default_factory=list,
        description="All events for the run in chronological order"
    )
    chain_valid: bool = Field(
        ...,
        description="Whether the hash chain verification passed"
    )
    exported_at: datetime = Field(
        default_factory=lambda: DeterministicClock.now(timezone.utc),
        description="UTC timestamp of export"
    )
    signature: Optional[str] = Field(
        None,
        description="Optional cryptographic signature of the package"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional export metadata"
    )

    model_config = {
        "frozen": True,
        "extra": "forbid"
    }

    def compute_package_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire audit package.

        Returns:
            SHA-256 hexadecimal hash of all event hashes concatenated
        """
        if not self.events:
            return hashlib.sha256(b"empty").hexdigest()

        # Concatenate all event hashes
        combined = "".join(event.event_hash for event in self.events)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()


# ============================================================================
# EVENT STORE PROTOCOL
# ============================================================================

@runtime_checkable
class EventStore(Protocol):
    """
    Protocol defining the EventStore interface.

    All event store implementations must provide these methods for
    append-only event storage with hash chain verification.

    Methods:
        append: Append a new event to the store
        get_events: Retrieve all events for a run
        verify_chain: Verify the hash chain integrity
        export_audit_package: Export complete audit trail
    """

    async def append(self, event: RunEvent) -> str:
        """
        Append a new event to the store.

        The event_hash will be computed and set before storage.
        Returns the computed event_hash.

        Args:
            event: RunEvent to append

        Returns:
            Computed event_hash

        Raises:
            ChainIntegrityError: If prev_event_hash doesn't match
        """
        ...

    async def get_events(self, run_id: str) -> List[RunEvent]:
        """
        Retrieve all events for a run in chronological order.

        Args:
            run_id: Run identifier

        Returns:
            List of RunEvents ordered by timestamp
        """
        ...

    async def verify_chain(self, run_id: str) -> bool:
        """
        Verify the hash chain integrity for a run.

        Checks that:
        1. First event has prev_event_hash = "genesis"
        2. Each event's event_hash matches computed hash
        3. Each event's prev_event_hash matches previous event_hash

        Args:
            run_id: Run identifier

        Returns:
            True if chain is valid, False if tampered
        """
        ...

    async def export_audit_package(self, run_id: str) -> AuditPackage:
        """
        Export complete audit trail as AuditPackage.

        Args:
            run_id: Run identifier

        Returns:
            AuditPackage with all events and verification status
        """
        ...


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class EventStoreError(Exception):
    """Base exception for event store errors."""
    pass


class ChainIntegrityError(EventStoreError):
    """Raised when hash chain integrity is violated."""

    def __init__(
        self,
        message: str,
        run_id: Optional[str] = None,
        event_id: Optional[str] = None,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None
    ):
        super().__init__(message)
        self.run_id = run_id
        self.event_id = event_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class EventNotFoundError(EventStoreError):
    """Raised when requested event or run is not found."""
    pass


# ============================================================================
# IN-MEMORY EVENT STORE (Testing/Development)
# ============================================================================

class InMemoryEventStore:
    """
    In-memory event store for testing and development.

    Provides the same interface as PostgresEventStore but stores
    events in memory. Useful for unit tests and local development.

    Example:
        >>> store = InMemoryEventStore()
        >>> await store.append(event)
        >>> events = await store.get_events("run-123")
    """

    def __init__(self):
        """Initialize empty event store."""
        self._events: Dict[str, List[RunEvent]] = {}
        self._latest_hash: Dict[str, str] = {}
        logger.info("Initialized InMemoryEventStore")

    async def append(self, event: RunEvent) -> str:
        """
        Append event to in-memory store.

        Args:
            event: RunEvent to append

        Returns:
            Computed event_hash

        Raises:
            ChainIntegrityError: If prev_event_hash is invalid
        """
        run_id = event.run_id

        # Initialize run if first event
        if run_id not in self._events:
            self._events[run_id] = []
            self._latest_hash[run_id] = GENESIS_HASH

        # Verify chain linkage
        expected_prev_hash = self._latest_hash[run_id]
        if event.prev_event_hash != expected_prev_hash:
            raise ChainIntegrityError(
                f"Invalid prev_event_hash: expected {expected_prev_hash}, "
                f"got {event.prev_event_hash}",
                run_id=run_id,
                event_id=event.event_id,
                expected_hash=expected_prev_hash,
                actual_hash=event.prev_event_hash
            )

        # Compute and set event hash
        event.event_hash = event.compute_hash()

        # Store event
        self._events[run_id].append(event)
        self._latest_hash[run_id] = event.event_hash

        logger.debug(
            f"Appended event {event.event_id} to run {run_id}: "
            f"type={event.event_type.value}, hash={event.event_hash[:16]}..."
        )

        return event.event_hash

    async def get_events(self, run_id: str) -> List[RunEvent]:
        """
        Get all events for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of events in chronological order
        """
        return self._events.get(run_id, []).copy()

    async def get_latest_hash(self, run_id: str) -> str:
        """
        Get the latest event hash for a run.

        Args:
            run_id: Run identifier

        Returns:
            Latest event hash or "genesis" if no events
        """
        return self._latest_hash.get(run_id, GENESIS_HASH)

    async def verify_chain(self, run_id: str) -> bool:
        """
        Verify hash chain integrity for a run.

        Args:
            run_id: Run identifier

        Returns:
            True if chain is valid
        """
        events = await self.get_events(run_id)

        if not events:
            return True  # Empty chain is valid

        prev_hash = GENESIS_HASH

        for event in events:
            # Verify prev_event_hash linkage
            if event.prev_event_hash != prev_hash:
                logger.error(
                    f"Chain broken at event {event.event_id}: "
                    f"expected prev_hash {prev_hash}, got {event.prev_event_hash}"
                )
                return False

            # Verify event_hash computation
            if not event.verify_hash():
                logger.error(
                    f"Hash mismatch at event {event.event_id}: "
                    f"stored {event.event_hash}, computed {event.compute_hash()}"
                )
                return False

            prev_hash = event.event_hash

        logger.debug(f"Chain verification passed for run {run_id}: {len(events)} events")
        return True

    async def export_audit_package(self, run_id: str) -> AuditPackage:
        """
        Export audit package for a run.

        Args:
            run_id: Run identifier

        Returns:
            AuditPackage with events and verification status
        """
        events = await self.get_events(run_id)
        chain_valid = await self.verify_chain(run_id)

        return AuditPackage(
            run_id=run_id,
            events=events,
            chain_valid=chain_valid,
            exported_at=DeterministicClock.now(timezone.utc),
            metadata={
                "event_count": len(events),
                "store_type": "in_memory",
                "hash_algorithm": HASH_ALGORITHM
            }
        )

    async def clear(self, run_id: Optional[str] = None) -> None:
        """
        Clear events (for testing).

        Args:
            run_id: Optional run to clear, or all if None
        """
        if run_id:
            self._events.pop(run_id, None)
            self._latest_hash.pop(run_id, None)
        else:
            self._events.clear()
            self._latest_hash.clear()


# ============================================================================
# POSTGRESQL EVENT STORE (Production)
# ============================================================================

class PostgresEventStore:
    """
    PostgreSQL-backed event store for production use.

    Uses asyncpg for high-performance async PostgreSQL operations.
    Provides durability, concurrent access, and ACID guarantees.

    Table Schema:
        CREATE TABLE run_events (
            event_id VARCHAR(36) PRIMARY KEY,
            run_id VARCHAR(255) NOT NULL,
            step_id VARCHAR(255),
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}',
            prev_event_hash VARCHAR(64) NOT NULL,
            event_hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX idx_run_events_run_id ON run_events(run_id);
        CREATE INDEX idx_run_events_run_timestamp ON run_events(run_id, timestamp);
        CREATE INDEX idx_run_events_event_type ON run_events(event_type);

    Example:
        >>> store = PostgresEventStore(
        ...     database_url="postgresql+asyncpg://user:pass@localhost/greenlang"
        ... )
        >>> await store.initialize()
        >>>
        >>> event_hash = await store.append(event)
        >>> events = await store.get_events("run-123")
        >>> is_valid = await store.verify_chain("run-123")
        >>>
        >>> await store.close()
    """

    # SQL statements
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS run_events (
            event_id VARCHAR(36) PRIMARY KEY,
            run_id VARCHAR(255) NOT NULL,
            step_id VARCHAR(255),
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}',
            prev_event_hash VARCHAR(64) NOT NULL,
            event_hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """

    CREATE_INDEXES_SQL = """
        CREATE INDEX IF NOT EXISTS idx_run_events_run_id
            ON run_events(run_id);
        CREATE INDEX IF NOT EXISTS idx_run_events_run_timestamp
            ON run_events(run_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_run_events_event_type
            ON run_events(event_type);
    """

    INSERT_EVENT_SQL = """
        INSERT INTO run_events
            (event_id, run_id, step_id, event_type, timestamp, payload,
             prev_event_hash, event_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING event_hash;
    """

    SELECT_EVENTS_SQL = """
        SELECT event_id, run_id, step_id, event_type, timestamp,
               payload, prev_event_hash, event_hash
        FROM run_events
        WHERE run_id = $1
        ORDER BY timestamp ASC, event_id ASC;
    """

    SELECT_LATEST_HASH_SQL = """
        SELECT event_hash
        FROM run_events
        WHERE run_id = $1
        ORDER BY timestamp DESC, event_id DESC
        LIMIT 1;
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 5
    ):
        """
        Initialize PostgreSQL event store.

        Args:
            database_url: PostgreSQL connection URL (asyncpg format)
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool = None
        self._initialized = False

        logger.info(
            f"Created PostgresEventStore: pool_size={pool_size}, "
            f"max_overflow={max_overflow}"
        )

    async def initialize(self) -> None:
        """
        Initialize the connection pool and create tables.

        Must be called before using the store.

        Raises:
            ImportError: If asyncpg is not installed
            ConnectionError: If database connection fails
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgresEventStore. "
                "Install with: pip install asyncpg"
            )

        try:
            # Parse database URL for asyncpg
            # Remove 'postgresql+asyncpg://' prefix if present
            db_url = self._database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = "postgresql://" + db_url[21:]
            elif db_url.startswith("postgres://"):
                db_url = "postgresql://" + db_url[11:]

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                db_url,
                min_size=2,
                max_size=self._pool_size + self._max_overflow,
                command_timeout=60
            )

            # Create tables and indexes
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_TABLE_SQL)
                await conn.execute(self.CREATE_INDEXES_SQL)

            self._initialized = True
            logger.info("PostgresEventStore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgresEventStore: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PostgresEventStore closed")

    def _ensure_initialized(self) -> None:
        """Ensure store is initialized before operations."""
        if not self._initialized or not self._pool:
            raise EventStoreError(
                "PostgresEventStore not initialized. Call initialize() first."
            )

    async def append(self, event: RunEvent) -> str:
        """
        Append event to PostgreSQL store.

        Args:
            event: RunEvent to append

        Returns:
            Computed event_hash

        Raises:
            ChainIntegrityError: If prev_event_hash is invalid
            EventStoreError: If database operation fails
        """
        self._ensure_initialized()

        run_id = event.run_id

        async with self._pool.acquire() as conn:
            # Get latest hash for chain verification
            row = await conn.fetchrow(self.SELECT_LATEST_HASH_SQL, run_id)
            expected_prev_hash = row['event_hash'] if row else GENESIS_HASH

            # Verify chain linkage
            if event.prev_event_hash != expected_prev_hash:
                raise ChainIntegrityError(
                    f"Invalid prev_event_hash: expected {expected_prev_hash}, "
                    f"got {event.prev_event_hash}",
                    run_id=run_id,
                    event_id=event.event_id,
                    expected_hash=expected_prev_hash,
                    actual_hash=event.prev_event_hash
                )

            # Compute event hash
            event.event_hash = event.compute_hash()

            # Insert event
            try:
                import json
                payload_json = json.dumps(event.payload)

                await conn.execute(
                    self.INSERT_EVENT_SQL,
                    event.event_id,
                    event.run_id,
                    event.step_id,
                    event.event_type.value,
                    event.timestamp,
                    payload_json,
                    event.prev_event_hash,
                    event.event_hash
                )

                logger.debug(
                    f"Appended event {event.event_id} to run {run_id}: "
                    f"type={event.event_type.value}, hash={event.event_hash[:16]}..."
                )

                return event.event_hash

            except Exception as e:
                logger.error(f"Failed to insert event: {e}")
                raise EventStoreError(f"Database insert failed: {e}") from e

    async def get_events(self, run_id: str) -> List[RunEvent]:
        """
        Get all events for a run from PostgreSQL.

        Args:
            run_id: Run identifier

        Returns:
            List of events in chronological order
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self.SELECT_EVENTS_SQL, run_id)

            events = []
            for row in rows:
                events.append(RunEvent(
                    event_id=row['event_id'],
                    run_id=row['run_id'],
                    step_id=row['step_id'],
                    event_type=EventType(row['event_type']),
                    timestamp=row['timestamp'],
                    payload=dict(row['payload']) if row['payload'] else {},
                    prev_event_hash=row['prev_event_hash'],
                    event_hash=row['event_hash']
                ))

            return events

    async def get_latest_hash(self, run_id: str) -> str:
        """
        Get the latest event hash for a run.

        Args:
            run_id: Run identifier

        Returns:
            Latest event hash or "genesis" if no events
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(self.SELECT_LATEST_HASH_SQL, run_id)
            return row['event_hash'] if row else GENESIS_HASH

    async def verify_chain(self, run_id: str) -> bool:
        """
        Verify hash chain integrity for a run.

        Args:
            run_id: Run identifier

        Returns:
            True if chain is valid
        """
        events = await self.get_events(run_id)

        if not events:
            return True

        prev_hash = GENESIS_HASH

        for event in events:
            # Verify prev_event_hash linkage
            if event.prev_event_hash != prev_hash:
                logger.error(
                    f"Chain broken at event {event.event_id}: "
                    f"expected prev_hash {prev_hash}, got {event.prev_event_hash}"
                )
                return False

            # Verify event_hash computation
            if not event.verify_hash():
                logger.error(
                    f"Hash mismatch at event {event.event_id}: "
                    f"stored {event.event_hash}, computed {event.compute_hash()}"
                )
                return False

            prev_hash = event.event_hash

        logger.debug(f"Chain verification passed for run {run_id}: {len(events)} events")
        return True

    async def export_audit_package(self, run_id: str) -> AuditPackage:
        """
        Export audit package for a run.

        Args:
            run_id: Run identifier

        Returns:
            AuditPackage with events and verification status
        """
        events = await self.get_events(run_id)
        chain_valid = await self.verify_chain(run_id)

        return AuditPackage(
            run_id=run_id,
            events=events,
            chain_valid=chain_valid,
            exported_at=DeterministicClock.now(timezone.utc),
            metadata={
                "event_count": len(events),
                "store_type": "postgresql",
                "hash_algorithm": HASH_ALGORITHM
            }
        )


# ============================================================================
# EVENT FACTORY HELPER
# ============================================================================

class EventFactory:
    """
    Factory for creating RunEvents with proper hash chaining.

    Simplifies event creation by automatically handling:
    - UUID generation for event_id
    - Timestamp generation
    - Hash chain linkage

    Example:
        >>> factory = EventFactory(store)
        >>> event = await factory.create_event(
        ...     run_id="run-123",
        ...     event_type=EventType.RUN_SUBMITTED,
        ...     payload={"workflow": "carbon-calc"}
        ... )
        >>> event_hash = await store.append(event)
    """

    def __init__(self, store: EventStore):
        """
        Initialize event factory.

        Args:
            store: EventStore implementation for hash lookups
        """
        self._store = store

    async def create_event(
        self,
        run_id: str,
        event_type: EventType,
        payload: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> RunEvent:
        """
        Create a new RunEvent with proper hash chaining.

        Args:
            run_id: Run identifier
            event_type: Type of event
            payload: Event-specific data
            step_id: Optional step identifier
            event_id: Optional event ID (auto-generated if not provided)
            timestamp: Optional timestamp (auto-generated if not provided)

        Returns:
            RunEvent ready for appending to store
        """
        # Get previous hash for chain linkage
        prev_hash = await self._store.get_latest_hash(run_id)

        return RunEvent(
            event_id=event_id or str(uuid4()),
            run_id=run_id,
            step_id=step_id,
            event_type=event_type,
            timestamp=timestamp or DeterministicClock.now(timezone.utc),
            payload=payload or {},
            prev_event_hash=prev_hash,
            event_hash=""  # Will be computed on append
        )

    async def emit(
        self,
        run_id: str,
        event_type: EventType,
        payload: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None
    ) -> str:
        """
        Create and append an event in one operation.

        Args:
            run_id: Run identifier
            event_type: Type of event
            payload: Event-specific data
            step_id: Optional step identifier

        Returns:
            Event hash of the appended event
        """
        event = await self.create_event(
            run_id=run_id,
            event_type=event_type,
            payload=payload,
            step_id=step_id
        )
        return await self._store.append(event)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Event types
    "EventType",
    # Models
    "RunEvent",
    "AuditPackage",
    # Protocol
    "EventStore",
    # Implementations
    "InMemoryEventStore",
    "PostgresEventStore",
    # Factory
    "EventFactory",
    # Exceptions
    "EventStoreError",
    "ChainIntegrityError",
    "EventNotFoundError",
    # Constants
    "GENESIS_HASH",
    "HASH_ALGORITHM",
]
