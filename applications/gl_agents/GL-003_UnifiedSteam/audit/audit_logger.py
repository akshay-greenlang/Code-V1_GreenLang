"""
Audit Logger for GL-003 UnifiedSteam SteamSystemOptimizer

This module implements comprehensive audit logging for steam system operations
with immutable audit entries, timestamps, and user attribution.

Key Features:
    - Immutable audit entries with SHA-256 hash chaining
    - User attribution for all actions
    - Calculation logging with formula tracking
    - Recommendation and setpoint change logging
    - Operator action tracking
    - Queryable audit log with time window support

Example:
    >>> logger = AuditLogger(storage_backend=FileAuditStorage("/audit/logs"))
    >>> entry = logger.log_calculation(
    ...     calc_type="steam_balance",
    ...     inputs={"flow_rate": 1000.0},
    ...     outputs={"enthalpy": 2800.0},
    ...     formula_id="STEAM_ENTHALPY_V1",
    ...     duration_ms=15.5
    ... )
    >>> print(entry.entry_hash)

Author: GreenLang Steam Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events in the steam system."""

    CALCULATION = "CALCULATION"
    RECOMMENDATION = "RECOMMENDATION"
    SETPOINT_CHANGE = "SETPOINT_CHANGE"
    OPERATOR_ACTION = "OPERATOR_ACTION"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    VALIDATION = "VALIDATION"
    OPTIMIZATION = "OPTIMIZATION"
    SAFETY_CHECK = "SAFETY_CHECK"
    DATA_INGESTION = "DATA_INGESTION"


class TimeWindow(BaseModel):
    """Time window for audit log queries."""

    start_time: datetime = Field(..., description="Start of time window (inclusive)")
    end_time: datetime = Field(..., description="End of time window (exclusive)")

    @validator("end_time")
    def end_after_start(cls, v, values):
        """Validate end_time is after start_time."""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v

    @classmethod
    def last_hours(cls, hours: int) -> "TimeWindow":
        """Create time window for last N hours."""
        now = datetime.now(timezone.utc)
        return cls(
            start_time=now - timedelta(hours=hours),
            end_time=now,
        )

    @classmethod
    def last_days(cls, days: int) -> "TimeWindow":
        """Create time window for last N days."""
        now = datetime.now(timezone.utc)
        return cls(
            start_time=now - timedelta(days=days),
            end_time=now,
        )


class AuditFilter(BaseModel):
    """Filter criteria for audit log queries."""

    event_types: Optional[List[AuditEventType]] = Field(
        None, description="Filter by event types"
    )
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    asset_id: Optional[str] = Field(None, description="Filter by asset ID")
    correlation_id: Optional[str] = Field(None, description="Filter by correlation ID")
    formula_id: Optional[str] = Field(None, description="Filter by formula ID")
    tag_id: Optional[str] = Field(None, description="Filter by tag ID")
    min_duration_ms: Optional[float] = Field(
        None, ge=0, description="Minimum duration in ms"
    )
    max_duration_ms: Optional[float] = Field(
        None, ge=0, description="Maximum duration in ms"
    )


class AuditEntry(BaseModel):
    """
    Immutable audit entry for steam system operations.

    All audit entries are timestamped, attributed, and include a SHA-256
    hash for integrity verification. Entries form a hash chain for
    tamper detection.
    """

    entry_id: UUID = Field(default_factory=uuid4, description="Unique entry identifier")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Entry creation timestamp"
    )

    # Attribution
    user_id: Optional[str] = Field(None, description="User who initiated the action")
    agent_id: str = Field(default="GL-003", description="Agent identifier")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for distributed tracing"
    )

    # Context
    asset_id: Optional[str] = Field(None, description="Asset/equipment identifier")
    subsystem: Optional[str] = Field(
        None, description="Subsystem (boiler, turbine, header, etc.)"
    )

    # Event data
    event_data: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )

    # Hash chain
    previous_hash: Optional[str] = Field(
        None, description="Hash of previous entry in chain"
    )
    sequence_number: int = Field(0, ge=0, description="Sequence number in chain")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def entry_hash(self) -> str:
        """
        Calculate SHA-256 hash of entry for integrity verification.

        Returns:
            Hex-encoded SHA-256 hash of entry content.
        """
        entry_data = self.dict(exclude={"entry_hash"})
        json_str = json.dumps(entry_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class AuditStorageBackend(ABC):
    """Abstract base class for audit log storage backends."""

    @abstractmethod
    def append(self, entry: AuditEntry) -> str:
        """Append entry to storage. Returns storage key."""
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID."""
        pass

    @abstractmethod
    def query(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query entries with filters."""
        pass

    @abstractmethod
    def get_latest_entry(self) -> Optional[AuditEntry]:
        """Get the most recent entry."""
        pass

    @abstractmethod
    def count(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
    ) -> int:
        """Count entries matching filters."""
        pass


class InMemoryAuditStorage(AuditStorageBackend):
    """
    In-memory audit storage backend for testing and development.

    Not suitable for production use due to volatility.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._entries: Dict[str, AuditEntry] = {}
        self._chain: List[str] = []  # Ordered entry IDs
        self._lock = threading.Lock()

    def append(self, entry: AuditEntry) -> str:
        """Append entry to storage."""
        with self._lock:
            entry_id = str(entry.entry_id)
            if entry_id in self._entries:
                raise ValueError(f"Entry already exists: {entry_id}")

            self._entries[entry_id] = entry
            self._chain.append(entry_id)
            return entry_id

    def get(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID."""
        return self._entries.get(entry_id)

    def query(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query entries with filters."""
        results = []

        for entry in self._entries.values():
            # Time window filter
            if time_window:
                if entry.timestamp < time_window.start_time:
                    continue
                if entry.timestamp >= time_window.end_time:
                    continue

            # Apply filters
            if filters:
                if filters.event_types and entry.event_type not in filters.event_types:
                    continue
                if filters.user_id and entry.user_id != filters.user_id:
                    continue
                if filters.asset_id and entry.asset_id != filters.asset_id:
                    continue
                if filters.correlation_id and entry.correlation_id != filters.correlation_id:
                    continue
                if filters.formula_id:
                    if entry.event_data.get("formula_id") != filters.formula_id:
                        continue
                if filters.tag_id:
                    if entry.event_data.get("tag") != filters.tag_id:
                        continue
                if filters.min_duration_ms is not None:
                    duration = entry.event_data.get("duration_ms", 0)
                    if duration < filters.min_duration_ms:
                        continue
                if filters.max_duration_ms is not None:
                    duration = entry.event_data.get("duration_ms", float("inf"))
                    if duration > filters.max_duration_ms:
                        continue

            results.append(entry)

        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply pagination
        return results[offset:offset + limit]

    def get_latest_entry(self) -> Optional[AuditEntry]:
        """Get the most recent entry."""
        if not self._chain:
            return None
        return self._entries.get(self._chain[-1])

    def count(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
    ) -> int:
        """Count entries matching filters."""
        return len(self.query(time_window, filters, limit=1000000))


class FileAuditStorage(AuditStorageBackend):
    """
    File-based audit storage backend with append-only semantics.

    Stores entries in JSON files organized by date.
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
        (self.base_path / "entries").mkdir(exist_ok=True)
        (self.base_path / "indices").mkdir(exist_ok=True)

        # In-memory index for fast lookup
        self._index: Dict[str, str] = {}  # entry_id -> file_path
        self._chain: List[str] = []  # Ordered entry IDs
        self._lock = threading.Lock()

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        index_file = self.base_path / "indices" / "entry_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                self._index = data.get("entries", {})
                self._chain = data.get("chain", [])
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save index to disk."""
        index_file = self.base_path / "indices" / "entry_index.json"
        data = {
            "entries": self._index,
            "chain": self._chain,
        }
        with open(index_file, "w") as f:
            json.dump(data, f)

    def _get_entry_path(self, entry: AuditEntry) -> Path:
        """Get file path for entry."""
        date_str = entry.timestamp.strftime("%Y/%m/%d")
        entry_dir = self.base_path / "entries" / date_str
        entry_dir.mkdir(parents=True, exist_ok=True)
        return entry_dir / f"{entry.entry_id}.json"

    def append(self, entry: AuditEntry) -> str:
        """Append entry to storage."""
        with self._lock:
            entry_id = str(entry.entry_id)
            file_path = self._get_entry_path(entry)

            # Append-only: never overwrite
            if file_path.exists():
                raise ValueError(f"Entry already exists: {entry_id}")

            # Write entry
            with open(file_path, "w") as f:
                json.dump(entry.dict(), f, indent=2, default=str)

            # Update index
            self._index[entry_id] = str(file_path)
            self._chain.append(entry_id)
            self._save_index()

            return entry_id

    def get(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID."""
        file_path = self._index.get(entry_id)
        if not file_path:
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return AuditEntry(**data)
        except Exception as e:
            logger.error(f"Failed to load entry {entry_id}: {e}")
            return None

    def query(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query entries with filters."""
        results = []

        for entry_id in self._chain:
            entry = self.get(entry_id)
            if not entry:
                continue

            # Time window filter
            if time_window:
                if entry.timestamp < time_window.start_time:
                    continue
                if entry.timestamp >= time_window.end_time:
                    continue

            # Apply filters
            if filters:
                if filters.event_types and entry.event_type not in filters.event_types:
                    continue
                if filters.user_id and entry.user_id != filters.user_id:
                    continue
                if filters.asset_id and entry.asset_id != filters.asset_id:
                    continue
                if filters.correlation_id and entry.correlation_id != filters.correlation_id:
                    continue

            results.append(entry)

        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[offset:offset + limit]

    def get_latest_entry(self) -> Optional[AuditEntry]:
        """Get the most recent entry."""
        if not self._chain:
            return None
        return self.get(self._chain[-1])

    def count(
        self,
        time_window: Optional[TimeWindow] = None,
        filters: Optional[AuditFilter] = None,
    ) -> int:
        """Count entries matching filters."""
        return len(self.query(time_window, filters, limit=1000000))


class AuditLogger:
    """
    Audit logger for steam system operations.

    Provides comprehensive logging of calculations, recommendations,
    setpoint changes, operator actions, and system events with
    immutable entries and hash chaining.

    Attributes:
        storage: Storage backend for audit entries
        agent_id: Agent identifier for entries

    Example:
        >>> logger = AuditLogger()
        >>> entry = logger.log_calculation(
        ...     calc_type="steam_balance",
        ...     inputs={"flow_rate": 1000.0},
        ...     outputs={"enthalpy": 2800.0},
        ...     formula_id="STEAM_ENTHALPY_V1",
        ...     duration_ms=15.5
        ... )
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        storage_backend: Optional[AuditStorageBackend] = None,
        agent_id: str = "GL-003",
    ):
        """
        Initialize audit logger.

        Args:
            storage_backend: Storage backend (default: in-memory)
            agent_id: Agent identifier for entries
        """
        self.storage = storage_backend or InMemoryAuditStorage()
        self.agent_id = agent_id
        self._lock = threading.Lock()
        self._sequence = 0

        # Initialize sequence from existing entries
        latest = self.storage.get_latest_entry()
        if latest:
            self._sequence = latest.sequence_number + 1

        logger.info(
            "AuditLogger initialized",
            extra={"agent_id": agent_id, "sequence": self._sequence}
        )

    def _get_previous_hash(self) -> str:
        """Get hash of previous entry in chain."""
        latest = self.storage.get_latest_entry()
        return latest.entry_hash if latest else self.GENESIS_HASH

    def _create_entry(
        self,
        event_type: AuditEventType,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        subsystem: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEntry:
        """Create a new audit entry with hash chain linking."""
        with self._lock:
            previous_hash = self._get_previous_hash()

            entry = AuditEntry(
                event_type=event_type,
                user_id=user_id,
                agent_id=self.agent_id,
                correlation_id=correlation_id,
                asset_id=asset_id,
                subsystem=subsystem,
                event_data=event_data,
                previous_hash=previous_hash,
                sequence_number=self._sequence,
            )

            # Store entry
            self.storage.append(entry)
            self._sequence += 1

            logger.info(
                f"Audit entry logged: {event_type.value}",
                extra={
                    "entry_id": str(entry.entry_id),
                    "sequence": entry.sequence_number,
                }
            )

            return entry

    def log_calculation(
        self,
        calc_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula_id: str,
        duration_ms: float,
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        subsystem: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log a calculation event.

        Args:
            calc_type: Type of calculation (e.g., "steam_balance", "enthalpy")
            inputs: Input values used in calculation
            outputs: Output values from calculation
            formula_id: Identifier of formula used
            duration_ms: Calculation duration in milliseconds
            user_id: User who initiated the calculation
            asset_id: Asset identifier
            subsystem: Subsystem identifier
            correlation_id: Correlation ID for tracing
            metadata: Additional metadata

        Returns:
            Created AuditEntry
        """
        # Compute input/output hashes for integrity
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()
        outputs_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        event_data = {
            "calc_type": calc_type,
            "inputs": inputs,
            "outputs": outputs,
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "formula_id": formula_id,
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }

        return self._create_entry(
            event_type=AuditEventType.CALCULATION,
            event_data=event_data,
            user_id=user_id,
            asset_id=asset_id,
            subsystem=subsystem,
            correlation_id=correlation_id,
        )

    def log_recommendation(
        self,
        recommendation: Dict[str, Any],
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log an optimization recommendation.

        Args:
            recommendation: Recommendation details (setpoints, actions)
            context: Context in which recommendation was made
            user_id: User who initiated the optimization
            asset_id: Asset identifier
            correlation_id: Correlation ID for tracing

        Returns:
            Created AuditEntry
        """
        event_data = {
            "recommendation": recommendation,
            "context": context,
            "recommendation_hash": hashlib.sha256(
                json.dumps(recommendation, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }

        return self._create_entry(
            event_type=AuditEventType.RECOMMENDATION,
            event_data=event_data,
            user_id=user_id,
            asset_id=asset_id,
            correlation_id=correlation_id,
        )

    def log_setpoint_change(
        self,
        tag: str,
        old_value: float,
        new_value: float,
        source: str,
        authorization: Optional[str] = None,
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log a setpoint change event.

        Args:
            tag: Control tag identifier
            old_value: Previous setpoint value
            new_value: New setpoint value
            source: Source of change (AGENT, OPERATOR, SYSTEM)
            authorization: Authorization reference if required
            user_id: User who made the change
            asset_id: Asset identifier
            correlation_id: Correlation ID for tracing
            reason: Reason for the change

        Returns:
            Created AuditEntry
        """
        event_data = {
            "tag": tag,
            "old_value": old_value,
            "new_value": new_value,
            "delta": new_value - old_value,
            "delta_pct": ((new_value - old_value) / old_value * 100) if old_value != 0 else None,
            "source": source,
            "authorization": authorization,
            "reason": reason,
        }

        return self._create_entry(
            event_type=AuditEventType.SETPOINT_CHANGE,
            event_data=event_data,
            user_id=user_id,
            asset_id=asset_id,
            correlation_id=correlation_id,
        )

    def log_operator_action(
        self,
        action_type: str,
        user_id: str,
        details: Dict[str, Any],
        asset_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        authorization_level: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log an operator action.

        Args:
            action_type: Type of action (APPROVE, REJECT, OVERRIDE, ACKNOWLEDGE)
            user_id: Operator user ID
            details: Action details
            asset_id: Asset identifier
            correlation_id: Correlation ID for tracing
            authorization_level: Required authorization level

        Returns:
            Created AuditEntry
        """
        event_data = {
            "action_type": action_type,
            "details": details,
            "authorization_level": authorization_level,
        }

        return self._create_entry(
            event_type=AuditEventType.OPERATOR_ACTION,
            event_data=event_data,
            user_id=user_id,
            asset_id=asset_id,
            correlation_id=correlation_id,
        )

    def log_system_event(
        self,
        event_type_detail: str,
        details: Dict[str, Any],
        asset_id: Optional[str] = None,
        subsystem: Optional[str] = None,
        correlation_id: Optional[str] = None,
        severity: str = "INFO",
    ) -> AuditEntry:
        """
        Log a system event.

        Args:
            event_type_detail: Specific type of system event
            details: Event details
            asset_id: Asset identifier
            subsystem: Subsystem identifier
            correlation_id: Correlation ID for tracing
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)

        Returns:
            Created AuditEntry
        """
        event_data = {
            "event_type_detail": event_type_detail,
            "details": details,
            "severity": severity,
        }

        return self._create_entry(
            event_type=AuditEventType.SYSTEM_EVENT,
            event_data=event_data,
            asset_id=asset_id,
            subsystem=subsystem,
            correlation_id=correlation_id,
        )

    def query_audit_log(
        self,
        filters: Optional[AuditFilter] = None,
        time_window: Optional[TimeWindow] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """
        Query the audit log.

        Args:
            filters: Filter criteria
            time_window: Time window for query
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of matching AuditEntry objects
        """
        return self.storage.query(
            time_window=time_window,
            filters=filters,
            limit=limit,
            offset=offset,
        )

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """
        Get a specific audit entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            AuditEntry if found, None otherwise
        """
        return self.storage.get(entry_id)

    def verify_chain(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify integrity of the audit log hash chain.

        Args:
            start_sequence: Starting sequence number
            end_sequence: Ending sequence number (None for all)

        Returns:
            Tuple of (is_valid, error_message)
        """
        entries = self.storage.query(limit=1000000)
        entries.sort(key=lambda e: e.sequence_number)

        # Filter by sequence range
        if end_sequence is not None:
            entries = [e for e in entries if start_sequence <= e.sequence_number <= end_sequence]
        else:
            entries = [e for e in entries if e.sequence_number >= start_sequence]

        if not entries:
            return True, None

        prev_hash = self.GENESIS_HASH if start_sequence == 0 else None

        for entry in entries:
            # For first entry after genesis, get previous hash from entry
            if prev_hash is None:
                if entry.sequence_number > 0:
                    # Find previous entry
                    all_entries = self.storage.query(limit=1000000)
                    prev_entries = [e for e in all_entries if e.sequence_number == entry.sequence_number - 1]
                    if prev_entries:
                        prev_hash = prev_entries[0].entry_hash
                    else:
                        return False, f"Cannot find previous entry for sequence {entry.sequence_number}"

            # Verify previous hash linkage
            if entry.previous_hash != prev_hash:
                return False, f"Chain broken at sequence {entry.sequence_number}: expected {prev_hash}, got {entry.previous_hash}"

            prev_hash = entry.entry_hash

        logger.info(
            f"Chain verified: sequences {start_sequence} to {end_sequence or 'latest'}",
            extra={"entries_verified": len(entries)}
        )
        return True, None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dictionary of statistics
        """
        all_entries = self.storage.query(limit=1000000)

        # Count by event type
        type_counts: Dict[str, int] = {}
        for entry in all_entries:
            event_type = entry.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Time range
        timestamps = [e.timestamp for e in all_entries]
        earliest = min(timestamps) if timestamps else None
        latest = max(timestamps) if timestamps else None

        return {
            "total_entries": len(all_entries),
            "current_sequence": self._sequence,
            "entries_by_type": type_counts,
            "time_range": {
                "earliest": earliest.isoformat() if earliest else None,
                "latest": latest.isoformat() if latest else None,
            },
            "genesis_hash": self.GENESIS_HASH,
        }
