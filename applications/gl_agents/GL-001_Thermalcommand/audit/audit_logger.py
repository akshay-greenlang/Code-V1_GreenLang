"""
Enhanced Audit Logger for GL-001 ThermalCommand

This module implements an enhanced audit logging system with:
    - Correlation ID propagation across services
    - Append-only storage support
    - Hash chaining for integrity verification
    - Multi-index queryability (asset, time, operator, event type, boundary)
    - Retention policy enforcement (7+ years)

The audit logger provides tamper-evident logging suitable for regulatory
compliance and forensic analysis.

Example:
    >>> logger = EnhancedAuditLogger(
    ...     storage_backend=FileStorageBackend("/audit/logs"),
    ...     retention_years=7
    ... )
    >>> logger.log_decision(decision_event)
    >>> events = logger.query(asset_id="boiler-001", event_type=EventType.DECISION)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .audit_events import (
    AuditEvent,
    BaseAuditEvent,
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    SystemAuditEvent,
    OverrideAuditEvent,
    EventType,
    create_event_from_dict,
)

# Module logger
logger = logging.getLogger(__name__)

# Context variable for correlation ID propagation
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    _correlation_id.set(correlation_id)


@contextmanager
def correlation_context(correlation_id: str) -> Generator[str, None, None]:
    """
    Context manager for correlation ID propagation.

    Args:
        correlation_id: Correlation ID to propagate

    Yields:
        The correlation ID

    Example:
        >>> with correlation_context("corr-12345") as corr_id:
        ...     # All operations within will use this correlation ID
        ...     process_data()
    """
    token = _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(token)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr-{uuid4().hex[:12]}-{int(datetime.now(timezone.utc).timestamp())}"


class HashChainEntry(BaseModel):
    """Entry in the hash chain for integrity verification."""

    sequence_number: int = Field(..., ge=0, description="Sequence number in chain")
    event_id: str = Field(..., description="Event ID")
    event_hash: str = Field(..., description="SHA-256 hash of event")
    previous_hash: str = Field(..., description="Hash of previous entry")
    chain_hash: str = Field(..., description="Cumulative chain hash")
    timestamp: datetime = Field(..., description="Entry timestamp")

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class RetentionPolicy(BaseModel):
    """Retention policy configuration."""

    retention_years: int = Field(7, ge=1, description="Retention period in years")
    archive_after_days: int = Field(365, ge=1, description="Archive after days")
    compress_after_days: int = Field(30, ge=1, description="Compress after days")
    delete_after_retention: bool = Field(True, description="Delete after retention period")

    def should_archive(self, event_time: datetime) -> bool:
        """Check if event should be archived."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.archive_after_days)
        return event_time < cutoff

    def should_delete(self, event_time: datetime) -> bool:
        """Check if event should be deleted (past retention)."""
        if not self.delete_after_retention:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_years * 365)
        return event_time < cutoff


class StorageBackend(ABC):
    """Abstract base class for audit log storage backends."""

    @abstractmethod
    def append(self, event: AuditEvent, chain_entry: HashChainEntry) -> str:
        """Append event to storage. Returns storage key."""
        pass

    @abstractmethod
    def get(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        pass

    @abstractmethod
    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        boundary_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query events with filters."""
        pass

    @abstractmethod
    def get_chain_entries(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> List[HashChainEntry]:
        """Get hash chain entries."""
        pass

    @abstractmethod
    def get_latest_chain_entry(self) -> Optional[HashChainEntry]:
        """Get the latest chain entry."""
        pass

    @abstractmethod
    def count(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count events matching filters."""
        pass


class FileStorageBackend(StorageBackend):
    """
    File-based append-only storage backend.

    Stores events in JSON files organized by date with append-only semantics.
    """

    def __init__(self, base_path: str):
        """
        Initialize file storage backend.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.base_path / "events").mkdir(exist_ok=True)
        (self.base_path / "chain").mkdir(exist_ok=True)
        (self.base_path / "indices").mkdir(exist_ok=True)

        # In-memory indices for fast querying
        self._event_index: Dict[str, str] = {}  # event_id -> file_path
        self._asset_index: Dict[str, Set[str]] = {}  # asset_id -> event_ids
        self._type_index: Dict[EventType, Set[str]] = {}  # type -> event_ids
        self._correlation_index: Dict[str, Set[str]] = {}  # correlation_id -> event_ids
        self._boundary_index: Dict[str, Set[str]] = {}  # boundary_id -> event_ids

        # Chain state
        self._chain: List[HashChainEntry] = []
        self._lock = threading.Lock()

        # Load existing indices
        self._load_indices()

    def _load_indices(self) -> None:
        """Load indices from disk."""
        index_file = self.base_path / "indices" / "event_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                self._event_index = data.get("events", {})
                self._asset_index = {k: set(v) for k, v in data.get("assets", {}).items()}
                self._type_index = {
                    EventType(k): set(v) for k, v in data.get("types", {}).items()
                }
                self._correlation_index = {
                    k: set(v) for k, v in data.get("correlations", {}).items()
                }
                self._boundary_index = {
                    k: set(v) for k, v in data.get("boundaries", {}).items()
                }
            except Exception as e:
                logger.warning(f"Failed to load indices: {e}")

        # Load chain
        chain_file = self.base_path / "chain" / "chain.json"
        if chain_file.exists():
            try:
                with open(chain_file, "r") as f:
                    chain_data = json.load(f)
                self._chain = [HashChainEntry(**e) for e in chain_data]
            except Exception as e:
                logger.warning(f"Failed to load chain: {e}")

    def _save_indices(self) -> None:
        """Save indices to disk."""
        index_file = self.base_path / "indices" / "event_index.json"
        data = {
            "events": self._event_index,
            "assets": {k: list(v) for k, v in self._asset_index.items()},
            "types": {k.value: list(v) for k, v in self._type_index.items()},
            "correlations": {k: list(v) for k, v in self._correlation_index.items()},
            "boundaries": {k: list(v) for k, v in self._boundary_index.items()},
        }
        with open(index_file, "w") as f:
            json.dump(data, f)

    def _save_chain(self) -> None:
        """Save chain to disk."""
        chain_file = self.base_path / "chain" / "chain.json"
        with open(chain_file, "w") as f:
            json.dump([e.dict() for e in self._chain], f, default=str)

    def _get_event_path(self, event: AuditEvent) -> Path:
        """Get file path for event."""
        date_str = event.timestamp.strftime("%Y/%m/%d")
        event_dir = self.base_path / "events" / date_str
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir / f"{event.event_id}.json"

    def _update_indices(self, event: AuditEvent, file_path: str) -> None:
        """Update in-memory indices."""
        event_id = str(event.event_id)

        self._event_index[event_id] = file_path

        # Asset index
        if event.asset_id not in self._asset_index:
            self._asset_index[event.asset_id] = set()
        self._asset_index[event.asset_id].add(event_id)

        # Type index
        if event.event_type not in self._type_index:
            self._type_index[event.event_type] = set()
        self._type_index[event.event_type].add(event_id)

        # Correlation index
        if event.correlation_id not in self._correlation_index:
            self._correlation_index[event.correlation_id] = set()
        self._correlation_index[event.correlation_id].add(event_id)

        # Boundary index (for safety events)
        if isinstance(event, SafetyAuditEvent):
            if event.boundary_id not in self._boundary_index:
                self._boundary_index[event.boundary_id] = set()
            self._boundary_index[event.boundary_id].add(event_id)

    def append(self, event: AuditEvent, chain_entry: HashChainEntry) -> str:
        """Append event to storage."""
        with self._lock:
            file_path = self._get_event_path(event)

            # Write event (append-only - never overwrite)
            if file_path.exists():
                raise ValueError(f"Event already exists: {event.event_id}")

            event_data = event.dict()
            event_data["_chain_entry"] = chain_entry.dict()

            with open(file_path, "w") as f:
                json.dump(event_data, f, default=str, indent=2)

            # Update indices
            self._update_indices(event, str(file_path))
            self._save_indices()

            # Update chain
            self._chain.append(chain_entry)
            self._save_chain()

            return str(file_path)

    def get(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        file_path = self._event_index.get(event_id)
        if not file_path:
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            data.pop("_chain_entry", None)
            return create_event_from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load event {event_id}: {e}")
            return None

    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        boundary_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query events with filters."""
        # Start with all event IDs or filtered sets
        candidate_ids: Optional[Set[str]] = None

        if asset_id:
            candidate_ids = self._asset_index.get(asset_id, set()).copy()

        if event_type:
            type_ids = self._type_index.get(event_type, set())
            candidate_ids = candidate_ids & type_ids if candidate_ids else type_ids.copy()

        if correlation_id:
            corr_ids = self._correlation_index.get(correlation_id, set())
            candidate_ids = candidate_ids & corr_ids if candidate_ids else corr_ids.copy()

        if boundary_id:
            bound_ids = self._boundary_index.get(boundary_id, set())
            candidate_ids = candidate_ids & bound_ids if candidate_ids else bound_ids.copy()

        # If no index filters, use all events
        if candidate_ids is None:
            candidate_ids = set(self._event_index.keys())

        # Load events and apply remaining filters
        events = []
        for event_id in candidate_ids:
            event = self.get(event_id)
            if not event:
                continue

            # Time filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            # Operator filter
            if operator_id and event.operator_id != operator_id:
                continue

            events.append(event)

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply pagination
        return events[offset:offset + limit]

    def get_chain_entries(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> List[HashChainEntry]:
        """Get hash chain entries."""
        if end_sequence is None:
            return self._chain[start_sequence:]
        return self._chain[start_sequence:end_sequence]

    def get_latest_chain_entry(self) -> Optional[HashChainEntry]:
        """Get the latest chain entry."""
        return self._chain[-1] if self._chain else None

    def count(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count events matching filters."""
        events = self.query(
            asset_id=asset_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=1000000,
        )
        return len(events)


class InMemoryStorageBackend(StorageBackend):
    """
    In-memory storage backend for testing.

    Not suitable for production use.
    """

    def __init__(self):
        self._events: Dict[str, AuditEvent] = {}
        self._chain: List[HashChainEntry] = []
        self._lock = threading.Lock()

    def append(self, event: AuditEvent, chain_entry: HashChainEntry) -> str:
        """Append event to storage."""
        with self._lock:
            event_id = str(event.event_id)
            if event_id in self._events:
                raise ValueError(f"Event already exists: {event_id}")

            self._events[event_id] = event
            self._chain.append(chain_entry)
            return event_id

    def get(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        return self._events.get(event_id)

    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        boundary_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query events with filters."""
        events = []
        for event in self._events.values():
            if asset_id and event.asset_id != asset_id:
                continue
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if operator_id and event.operator_id != operator_id:
                continue
            if correlation_id and event.correlation_id != correlation_id:
                continue
            if boundary_id and isinstance(event, SafetyAuditEvent):
                if event.boundary_id != boundary_id:
                    continue

            events.append(event)

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[offset:offset + limit]

    def get_chain_entries(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> List[HashChainEntry]:
        """Get hash chain entries."""
        if end_sequence is None:
            return self._chain[start_sequence:]
        return self._chain[start_sequence:end_sequence]

    def get_latest_chain_entry(self) -> Optional[HashChainEntry]:
        """Get the latest chain entry."""
        return self._chain[-1] if self._chain else None

    def count(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count events matching filters."""
        return len(self.query(
            asset_id=asset_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=1000000,
        ))


class EnhancedAuditLogger:
    """
    Enhanced audit logger with hash chaining and correlation ID propagation.

    This class provides comprehensive audit logging with:
    - Correlation ID propagation for distributed tracing
    - Append-only storage with hash chaining for integrity
    - Multi-index queries for efficient retrieval
    - Retention policy enforcement

    Attributes:
        storage: Storage backend
        retention_policy: Retention policy configuration

    Example:
        >>> logger = EnhancedAuditLogger(
        ...     storage_backend=FileStorageBackend("/audit/logs"),
        ...     retention_years=7
        ... )
        >>> with correlation_context("corr-12345"):
        ...     logger.log_decision(decision_event)
    """

    # Genesis hash for chain initialization
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        retention_years: int = 7,
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        """
        Initialize enhanced audit logger.

        Args:
            storage_backend: Storage backend (default: in-memory)
            retention_years: Retention period in years (default: 7)
            retention_policy: Custom retention policy
        """
        self.storage = storage_backend or InMemoryStorageBackend()
        self.retention_policy = retention_policy or RetentionPolicy(
            retention_years=retention_years
        )
        self._lock = threading.Lock()
        self._sequence = 0

        # Initialize sequence from existing chain
        latest = self.storage.get_latest_chain_entry()
        if latest:
            self._sequence = latest.sequence_number + 1

        logger.info(
            "EnhancedAuditLogger initialized",
            extra={
                "retention_years": self.retention_policy.retention_years,
                "sequence": self._sequence,
            }
        )

    def _compute_event_hash(self, event: AuditEvent) -> str:
        """Compute SHA-256 hash of event."""
        event_data = event.dict()
        json_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _get_previous_hash(self) -> str:
        """Get hash of previous chain entry."""
        latest = self.storage.get_latest_chain_entry()
        return latest.chain_hash if latest else self.GENESIS_HASH

    def _create_chain_entry(self, event: AuditEvent) -> HashChainEntry:
        """Create a new hash chain entry for event."""
        event_hash = self._compute_event_hash(event)
        previous_hash = self._get_previous_hash()

        # Chain hash = hash(event_hash + previous_hash)
        chain_input = event_hash + previous_hash
        chain_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        entry = HashChainEntry(
            sequence_number=self._sequence,
            event_id=str(event.event_id),
            event_hash=event_hash,
            previous_hash=previous_hash,
            chain_hash=chain_hash,
            timestamp=datetime.now(timezone.utc),
        )

        self._sequence += 1
        return entry

    def log_event(self, event: AuditEvent) -> str:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Storage key for the event

        Raises:
            ValueError: If event is invalid
        """
        with self._lock:
            start_time = datetime.now(timezone.utc)

            # Set correlation ID from context if not set
            if not event.correlation_id:
                ctx_corr_id = get_correlation_id()
                if ctx_corr_id:
                    # Need to create new event with correlation ID
                    event_data = event.dict()
                    event_data["correlation_id"] = ctx_corr_id
                    event = create_event_from_dict(event_data)

            # Create chain entry
            chain_entry = self._create_chain_entry(event)

            # Store event
            storage_key = self.storage.append(event, chain_entry)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info(
                f"Audit event logged: {event.event_type.value}",
                extra={
                    "event_id": str(event.event_id),
                    "correlation_id": event.correlation_id,
                    "asset_id": event.asset_id,
                    "sequence": chain_entry.sequence_number,
                    "processing_time_ms": processing_time,
                }
            )

            return storage_key

    def log_decision(self, event: DecisionAuditEvent) -> str:
        """Log a decision audit event."""
        return self.log_event(event)

    def log_action(self, event: ActionAuditEvent) -> str:
        """Log an action audit event."""
        return self.log_event(event)

    def log_safety(self, event: SafetyAuditEvent) -> str:
        """Log a safety audit event."""
        return self.log_event(event)

    def log_compliance(self, event: ComplianceAuditEvent) -> str:
        """Log a compliance audit event."""
        return self.log_event(event)

    def log_system(self, event: SystemAuditEvent) -> str:
        """Log a system audit event."""
        return self.log_event(event)

    def log_override(self, event: OverrideAuditEvent) -> str:
        """Log an override audit event."""
        return self.log_event(event)

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an event by ID.

        Args:
            event_id: Event ID

        Returns:
            The event if found, None otherwise
        """
        return self.storage.get(event_id)

    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        boundary_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            asset_id: Filter by asset ID
            event_type: Filter by event type
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (exclusive)
            operator_id: Filter by operator ID
            boundary_id: Filter by safety boundary ID
            correlation_id: Filter by correlation ID
            limit: Maximum events to return
            offset: Offset for pagination

        Returns:
            List of matching events
        """
        return self.storage.query(
            asset_id=asset_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            operator_id=operator_id,
            boundary_id=boundary_id,
            correlation_id=correlation_id,
            limit=limit,
            offset=offset,
        )

    def query_by_correlation(self, correlation_id: str) -> List[AuditEvent]:
        """
        Get all events for a correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            List of events sorted by timestamp
        """
        return self.query(correlation_id=correlation_id, limit=10000)

    def count(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Count events matching filters.

        Args:
            asset_id: Filter by asset ID
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Count of matching events
        """
        return self.storage.count(
            asset_id=asset_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
        )

    def verify_chain(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of the hash chain.

        Args:
            start_sequence: Starting sequence number
            end_sequence: Ending sequence number (exclusive)

        Returns:
            Tuple of (is_valid, error_message)
        """
        entries = self.storage.get_chain_entries(start_sequence, end_sequence)

        if not entries:
            return True, None

        prev_hash = self.GENESIS_HASH if start_sequence == 0 else None

        for i, entry in enumerate(entries):
            # For first entry after genesis, get previous from chain
            if prev_hash is None and i == 0:
                if start_sequence > 0:
                    prev_entries = self.storage.get_chain_entries(start_sequence - 1, start_sequence)
                    if prev_entries:
                        prev_hash = prev_entries[0].chain_hash
                    else:
                        return False, f"Cannot find previous entry for sequence {start_sequence}"

            # Verify previous hash linkage
            if entry.previous_hash != prev_hash:
                return False, f"Chain broken at sequence {entry.sequence_number}"

            # Verify chain hash computation
            expected_chain_hash = hashlib.sha256(
                (entry.event_hash + entry.previous_hash).encode("utf-8")
            ).hexdigest()

            if entry.chain_hash != expected_chain_hash:
                return False, f"Invalid chain hash at sequence {entry.sequence_number}"

            # Verify event hash
            event = self.storage.get(entry.event_id)
            if event:
                computed_hash = self._compute_event_hash(event)
                if computed_hash != entry.event_hash:
                    return False, f"Event hash mismatch at sequence {entry.sequence_number}"

            prev_hash = entry.chain_hash

        logger.info(
            f"Chain verified: sequences {start_sequence} to {end_sequence or 'latest'}",
            extra={"entries_verified": len(entries)}
        )
        return True, None

    def get_chain_statistics(self) -> Dict[str, Any]:
        """
        Get hash chain statistics.

        Returns:
            Dictionary of chain statistics
        """
        entries = self.storage.get_chain_entries()
        latest = self.storage.get_latest_chain_entry()

        return {
            "total_entries": len(entries),
            "latest_sequence": latest.sequence_number if latest else -1,
            "latest_timestamp": latest.timestamp.isoformat() if latest else None,
            "latest_chain_hash": latest.chain_hash if latest else self.GENESIS_HASH,
            "genesis_hash": self.GENESIS_HASH,
        }

    def enforce_retention(self) -> Dict[str, int]:
        """
        Enforce retention policy.

        Returns:
            Dictionary with counts of archived/deleted events
        """
        # This is a simplified implementation
        # Production would need more sophisticated handling
        now = datetime.now(timezone.utc)
        archive_cutoff = now - timedelta(days=self.retention_policy.archive_after_days)
        delete_cutoff = now - timedelta(days=self.retention_policy.retention_years * 365)

        archived = 0
        deleted = 0

        # Query old events
        events = self.query(end_time=archive_cutoff, limit=10000)

        for event in events:
            if self.retention_policy.should_delete(event.timestamp):
                # Would delete in production
                deleted += 1
            elif self.retention_policy.should_archive(event.timestamp):
                # Would archive in production
                archived += 1

        logger.info(
            f"Retention policy enforced",
            extra={"archived": archived, "deleted": deleted}
        )

        return {"archived": archived, "deleted": deleted}

    def get_audit_trail(
        self,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """
        Get complete audit trail for a correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            Complete audit trail with all related events
        """
        events = self.query_by_correlation(correlation_id)

        # Organize by event type
        decisions = []
        actions = []
        safety = []
        compliance = []
        system = []
        overrides = []

        for event in events:
            if event.event_type == EventType.DECISION:
                decisions.append(event.dict())
            elif event.event_type == EventType.ACTION:
                actions.append(event.dict())
            elif event.event_type == EventType.SAFETY:
                safety.append(event.dict())
            elif event.event_type == EventType.COMPLIANCE:
                compliance.append(event.dict())
            elif event.event_type == EventType.SYSTEM:
                system.append(event.dict())
            elif event.event_type == EventType.OVERRIDE:
                overrides.append(event.dict())

        return {
            "correlation_id": correlation_id,
            "total_events": len(events),
            "time_range": {
                "start": min(e.timestamp for e in events).isoformat() if events else None,
                "end": max(e.timestamp for e in events).isoformat() if events else None,
            },
            "decisions": decisions,
            "actions": actions,
            "safety_events": safety,
            "compliance_events": compliance,
            "system_events": system,
            "overrides": overrides,
        }
