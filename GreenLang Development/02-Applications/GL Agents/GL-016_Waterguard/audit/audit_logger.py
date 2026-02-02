"""
Waterguard Audit Logger for GL-016

This module implements an immutable audit logging system with:
    - Append-only storage with 7-year retention
    - Hash chaining for tamper resistance (SHA-256)
    - Multi-index queryability (asset, time, operator, event type)
    - Correlation ID propagation across services

The audit logger provides tamper-evident logging suitable for regulatory
compliance and forensic analysis of water chemistry decisions.

Example:
    >>> logger = WaterguardAuditLogger(
    ...     storage_backend=FileStorageBackend("/audit/logs"),
    ...     retention_years=7
    ... )
    >>> logger.log_recommendation(rec_id, inputs, outputs, explanation)
    >>> events = logger.query(asset_id="boiler-001", event_type=EventType.RECOMMENDATION_GENERATED)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .audit_events import (
    WaterguardAuditEvent,
    ChemistryCalculationEvent,
    RecommendationGeneratedEvent,
    CommandExecutedEvent,
    ConstraintViolationEvent,
    ConfigChangeEvent,
    OperatorActionEvent,
    EventType,
    RecommendationType,
    CommandType,
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
        ...     process_chemistry_data()
    """
    token = _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(token)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"wg-corr-{uuid4().hex[:12]}-{int(datetime.now(timezone.utc).timestamp())}"


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
    """Retention policy configuration for 7-year regulatory compliance."""

    retention_years: int = Field(7, ge=1, description="Retention period in years")
    archive_after_days: int = Field(365, ge=1, description="Archive after days")
    compress_after_days: int = Field(30, ge=1, description="Compress after days")
    delete_after_retention: bool = Field(False, description="Delete after retention period")

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

    def get_retention_end(self, event_time: datetime) -> datetime:
        """Get the date when retention expires for an event."""
        return event_time + timedelta(days=self.retention_years * 365)


class StorageBackend(ABC):
    """Abstract base class for audit log storage backends."""

    @abstractmethod
    def append(self, event: WaterguardAuditEvent, chain_entry: HashChainEntry) -> str:
        """Append event to storage. Returns storage key."""
        pass

    @abstractmethod
    def get(self, event_id: str) -> Optional[WaterguardAuditEvent]:
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
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WaterguardAuditEvent]:
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
    Suitable for production use with proper filesystem permissions.
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
        self._event_index: Dict[str, str] = {}
        self._asset_index: Dict[str, Set[str]] = {}
        self._type_index: Dict[EventType, Set[str]] = {}
        self._correlation_index: Dict[str, Set[str]] = {}
        self._operator_index: Dict[str, Set[str]] = {}

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
                self._operator_index = {
                    k: set(v) for k, v in data.get("operators", {}).items()
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
            "operators": {k: list(v) for k, v in self._operator_index.items()},
        }
        with open(index_file, "w") as f:
            json.dump(data, f)

    def _save_chain(self) -> None:
        """Save chain to disk."""
        chain_file = self.base_path / "chain" / "chain.json"
        with open(chain_file, "w") as f:
            json.dump([e.dict() for e in self._chain], f, default=str)

    def _get_event_path(self, event: WaterguardAuditEvent) -> Path:
        """Get file path for event."""
        date_str = event.timestamp.strftime("%Y/%m/%d")
        event_dir = self.base_path / "events" / date_str
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir / f"{event.event_id}.json"

    def _update_indices(self, event: WaterguardAuditEvent, file_path: str) -> None:
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

        # Operator index
        if event.operator_id:
            if event.operator_id not in self._operator_index:
                self._operator_index[event.operator_id] = set()
            self._operator_index[event.operator_id].add(event_id)

    def append(self, event: WaterguardAuditEvent, chain_entry: HashChainEntry) -> str:
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

    def get(self, event_id: str) -> Optional[WaterguardAuditEvent]:
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
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WaterguardAuditEvent]:
        """Query events with filters."""
        candidate_ids: Optional[Set[str]] = None

        if asset_id:
            candidate_ids = self._asset_index.get(asset_id, set()).copy()

        if event_type:
            type_ids = self._type_index.get(event_type, set())
            candidate_ids = candidate_ids & type_ids if candidate_ids else type_ids.copy()

        if correlation_id:
            corr_ids = self._correlation_index.get(correlation_id, set())
            candidate_ids = candidate_ids & corr_ids if candidate_ids else corr_ids.copy()

        if operator_id:
            op_ids = self._operator_index.get(operator_id, set())
            candidate_ids = candidate_ids & op_ids if candidate_ids else op_ids.copy()

        if candidate_ids is None:
            candidate_ids = set(self._event_index.keys())

        # Load events and apply remaining filters
        events = []
        for event_id in candidate_ids:
            event = self.get(event_id)
            if not event:
                continue

            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            events.append(event)

        # Sort by timestamp descending
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

    Not suitable for production use - data is lost on restart.
    """

    def __init__(self):
        self._events: Dict[str, WaterguardAuditEvent] = {}
        self._chain: List[HashChainEntry] = []
        self._lock = threading.Lock()

    def append(self, event: WaterguardAuditEvent, chain_entry: HashChainEntry) -> str:
        """Append event to storage."""
        with self._lock:
            event_id = str(event.event_id)
            if event_id in self._events:
                raise ValueError(f"Event already exists: {event_id}")

            self._events[event_id] = event
            self._chain.append(chain_entry)
            return event_id

    def get(self, event_id: str) -> Optional[WaterguardAuditEvent]:
        """Get event by ID."""
        return self._events.get(event_id)

    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WaterguardAuditEvent]:
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


class WaterguardAuditLogger:
    """
    Immutable audit logger for Waterguard with hash chaining.

    This class provides comprehensive audit logging with:
    - Append-only storage with 7-year retention
    - Tamper-resistant hash chain (SHA-256)
    - Multi-index queries for efficient retrieval
    - Correlation ID propagation

    Attributes:
        storage: Storage backend
        retention_policy: Retention policy configuration

    Example:
        >>> logger = WaterguardAuditLogger(
        ...     storage_backend=FileStorageBackend("/audit/logs"),
        ...     retention_years=7
        ... )
        >>> logger.log_recommendation(rec_id, inputs, outputs, explanation)
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        retention_years: int = 7,
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        """
        Initialize Waterguard audit logger.

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
            "WaterguardAuditLogger initialized",
            extra={
                "retention_years": self.retention_policy.retention_years,
                "sequence": self._sequence,
            }
        )

    def _compute_event_hash(self, event: WaterguardAuditEvent) -> str:
        """Compute SHA-256 hash of event."""
        event_data = event.dict()
        json_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _get_previous_hash(self) -> str:
        """Get hash of previous chain entry."""
        latest = self.storage.get_latest_chain_entry()
        return latest.chain_hash if latest else self.GENESIS_HASH

    def _create_chain_entry(self, event: WaterguardAuditEvent) -> HashChainEntry:
        """Create a new hash chain entry for event."""
        event_hash = self._compute_event_hash(event)
        previous_hash = self._get_previous_hash()

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

    def log_event(self, event: WaterguardAuditEvent) -> str:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Storage key for the event
        """
        with self._lock:
            start_time = datetime.now(timezone.utc)

            # Set correlation ID from context if not set
            if not event.correlation_id:
                ctx_corr_id = get_correlation_id()
                if ctx_corr_id:
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

    def log_recommendation(
        self,
        rec_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        explanation: str,
        asset_id: str,
        recommendation_type: RecommendationType,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Log a recommendation generated event.

        Args:
            rec_id: Recommendation ID
            inputs: Input data used
            outputs: Recommendation outputs
            explanation: Natural language explanation
            asset_id: Asset ID
            recommendation_type: Type of recommendation
            correlation_id: Optional correlation ID

        Returns:
            Storage key for the event
        """
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        event = RecommendationGeneratedEvent(
            correlation_id=correlation_id or generate_correlation_id(),
            asset_id=asset_id,
            recommendation_id=rec_id,
            recommendation_type=recommendation_type,
            chemistry_event_id=inputs.get("chemistry_event_id", "unknown"),
            input_data_hash=input_hash,
            current_value=outputs.get("current_value", 0.0),
            recommended_value=outputs.get("recommended_value", 0.0),
            unit=outputs.get("unit", ""),
            min_allowed=outputs.get("min_allowed", 0.0),
            max_allowed=outputs.get("max_allowed", 100.0),
            explanation=explanation,
            confidence_score=outputs.get("confidence", 0.8),
        )

        return self.log_event(event)

    def log_command(
        self,
        cmd_id: str,
        target: str,
        value: float,
        operator: str,
        approved_by: str,
        asset_id: str,
        command_type: CommandType,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Log a command executed event.

        Args:
            cmd_id: Command ID
            target: Target tag
            value: Target value
            operator: Operator ID
            approved_by: Approver ID
            asset_id: Asset ID
            command_type: Type of command
            correlation_id: Optional correlation ID

        Returns:
            Storage key for the event
        """
        now = datetime.now(timezone.utc)

        event = CommandExecutedEvent(
            correlation_id=correlation_id or generate_correlation_id(),
            asset_id=asset_id,
            operator_id=operator,
            command_id=cmd_id,
            command_type=command_type,
            target_tag=target,
            target_equipment=target.split(".")[0] if "." in target else target,
            target_value=value,
            unit="",
            approved_by=approved_by,
            approval_timestamp=now,
            execution_timestamp=now,
            execution_status="SUCCESS",
        )

        return self.log_event(event)

    def log_config_change(
        self,
        change_id: str,
        before: Any,
        after: Any,
        approved_by: str,
        asset_id: str,
        config_section: str,
        config_key: str,
        reason: str,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Log a configuration change event.

        Args:
            change_id: Change ID
            before: Value before change
            after: Value after change
            approved_by: Approver ID
            asset_id: Asset ID
            config_section: Configuration section
            config_key: Configuration key
            reason: Reason for change
            correlation_id: Optional correlation ID

        Returns:
            Storage key for the event
        """
        before_hash = hashlib.sha256(
            json.dumps(before, sort_keys=True, default=str).encode()
        ).hexdigest()
        after_hash = hashlib.sha256(
            json.dumps(after, sort_keys=True, default=str).encode()
        ).hexdigest()

        event = ConfigChangeEvent(
            correlation_id=correlation_id or generate_correlation_id(),
            asset_id=asset_id,
            change_id=change_id,
            config_section=config_section,
            config_key=config_key,
            before_value=before,
            after_value=after,
            approved_by=approved_by,
            approval_timestamp=datetime.now(timezone.utc),
            change_reason=reason,
            config_version_before="1.0.0",
            config_version_after="1.0.1",
            config_hash_before=before_hash,
            config_hash_after=after_hash,
        )

        return self.log_event(event)

    def get_event(self, event_id: str) -> Optional[WaterguardAuditEvent]:
        """Get an event by ID."""
        return self.storage.get(event_id)

    def query(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WaterguardAuditEvent]:
        """
        Query audit events.

        Args:
            asset_id: Filter by asset ID
            event_type: Filter by event type
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (exclusive)
            operator_id: Filter by operator ID
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
            correlation_id=correlation_id,
            limit=limit,
            offset=offset,
        )

    def query_by_correlation(self, correlation_id: str) -> List[WaterguardAuditEvent]:
        """Get all events for a correlation ID."""
        return self.query(correlation_id=correlation_id, limit=10000)

    def count(
        self,
        asset_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count events matching filters."""
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
            if prev_hash is None and i == 0:
                if start_sequence > 0:
                    prev_entries = self.storage.get_chain_entries(start_sequence - 1, start_sequence)
                    if prev_entries:
                        prev_hash = prev_entries[0].chain_hash
                    else:
                        return False, f"Cannot find previous entry for sequence {start_sequence}"

            if entry.previous_hash != prev_hash:
                return False, f"Chain broken at sequence {entry.sequence_number}"

            expected_chain_hash = hashlib.sha256(
                (entry.event_hash + entry.previous_hash).encode("utf-8")
            ).hexdigest()

            if entry.chain_hash != expected_chain_hash:
                return False, f"Invalid chain hash at sequence {entry.sequence_number}"

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
        """Get hash chain statistics."""
        entries = self.storage.get_chain_entries()
        latest = self.storage.get_latest_chain_entry()

        return {
            "total_entries": len(entries),
            "latest_sequence": latest.sequence_number if latest else -1,
            "latest_timestamp": latest.timestamp.isoformat() if latest else None,
            "latest_chain_hash": latest.chain_hash if latest else self.GENESIS_HASH,
            "genesis_hash": self.GENESIS_HASH,
            "retention_years": self.retention_policy.retention_years,
        }

    def get_audit_trail(self, correlation_id: str) -> Dict[str, Any]:
        """Get complete audit trail for a correlation ID."""
        events = self.query_by_correlation(correlation_id)

        chemistry = []
        recommendations = []
        commands = []
        violations = []
        config_changes = []
        operator_actions = []

        for event in events:
            if event.event_type == EventType.CHEMISTRY_CALCULATION:
                chemistry.append(event.dict())
            elif event.event_type == EventType.RECOMMENDATION_GENERATED:
                recommendations.append(event.dict())
            elif event.event_type == EventType.COMMAND_EXECUTED:
                commands.append(event.dict())
            elif event.event_type == EventType.CONSTRAINT_VIOLATION:
                violations.append(event.dict())
            elif event.event_type == EventType.CONFIG_CHANGE:
                config_changes.append(event.dict())
            elif event.event_type == EventType.OPERATOR_ACTION:
                operator_actions.append(event.dict())

        return {
            "correlation_id": correlation_id,
            "total_events": len(events),
            "time_range": {
                "start": min(e.timestamp for e in events).isoformat() if events else None,
                "end": max(e.timestamp for e in events).isoformat() if events else None,
            },
            "chemistry_calculations": chemistry,
            "recommendations": recommendations,
            "commands": commands,
            "violations": violations,
            "config_changes": config_changes,
            "operator_actions": operator_actions,
        }
