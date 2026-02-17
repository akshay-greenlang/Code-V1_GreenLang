# -*- coding: utf-8 -*-
"""
Dataset Registry Engine - AGENT-DATA-016 Data Freshness Monitor

Pure-Python engine for managing a catalog of monitored datasets with
metadata, refresh tracking, grouping, and statistics. Provides dataset
registration, refresh event recording, dataset grouping, bulk
operations, and aggregate statistics for data freshness monitoring.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations. Statistics are
derived from in-memory data structures with explicit, auditable
formulas.

Engine 1 of 7 in the Data Freshness Monitor pipeline.

Example:
    >>> from greenlang.data_freshness_monitor.dataset_registry import (
    ...     DatasetRegistryEngine,
    ... )
    >>> engine = DatasetRegistryEngine()
    >>> ds = engine.register_dataset(
    ...     name="ERP Spend Data",
    ...     source_name="SAP",
    ...     source_type="erp",
    ...     owner="finance-team",
    ...     refresh_cadence="daily",
    ...     priority="high",
    ... )
    >>> print(ds.id, ds.status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.models import (
        DatasetDefinition,
        DatasetStatus,
        DatasetPriority,
        RefreshCadence,
        RefreshEvent,
        DatasetGroup,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.info(
        "data_freshness_monitor.models not available; "
        "using inline dataclass-style fallback models"
    )

    # ------------------------------------------------------------------
    # Inline fallback enums
    # ------------------------------------------------------------------
    from enum import Enum

    class DatasetStatus(str, Enum):  # type: ignore[no-redef]
        """Dataset lifecycle status."""

        ACTIVE = "active"
        INACTIVE = "inactive"
        STALE = "stale"
        ERROR = "error"
        PENDING = "pending"

    class DatasetPriority(str, Enum):  # type: ignore[no-redef]
        """Priority classification for dataset monitoring."""

        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    class RefreshCadence(str, Enum):  # type: ignore[no-redef]
        """Expected refresh frequency for a dataset."""

        REALTIME = "realtime"
        HOURLY = "hourly"
        DAILY = "daily"
        WEEKLY = "weekly"
        MONTHLY = "monthly"
        QUARTERLY = "quarterly"
        ANNUAL = "annual"

    class DatasetDefinition:  # type: ignore[no-redef]
        """Fallback dataset definition without Pydantic."""

        def __init__(
            self,
            id: str = "",
            name: str = "",
            source_name: str = "",
            source_type: str = "",
            owner: str = "",
            refresh_cadence: str = "daily",
            priority: str = "medium",
            status: str = "active",
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            created_at: Optional[datetime] = None,
            updated_at: Optional[datetime] = None,
            version: int = 1,
            provenance_hash: str = "",
        ) -> None:
            self.id = id or uuid4().hex[:12]
            self.name = name
            self.source_name = source_name
            self.source_type = source_type
            self.owner = owner
            self.refresh_cadence = refresh_cadence
            self.priority = priority
            self.status = status
            self.tags = tags or []
            self.metadata = metadata or {}
            now = _utcnow()
            self.created_at = created_at or now
            self.updated_at = updated_at or now
            self.version = version
            self.provenance_hash = provenance_hash

        def to_dict(self) -> Dict[str, Any]:
            """Serialise to dictionary."""
            return {
                "id": self.id,
                "name": self.name,
                "source_name": self.source_name,
                "source_type": self.source_type,
                "owner": self.owner,
                "refresh_cadence": self.refresh_cadence,
                "priority": self.priority,
                "status": self.status,
                "tags": list(self.tags),
                "metadata": dict(self.metadata),
                "created_at": self.created_at.isoformat()
                if self.created_at else None,
                "updated_at": self.updated_at.isoformat()
                if self.updated_at else None,
                "version": self.version,
                "provenance_hash": self.provenance_hash,
            }

    class RefreshEvent:  # type: ignore[no-redef]
        """Fallback refresh event without Pydantic."""

        def __init__(
            self,
            id: str = "",
            dataset_id: str = "",
            refreshed_at: Optional[datetime] = None,
            recorded_at: Optional[datetime] = None,
            data_size_bytes: Optional[int] = None,
            record_count: Optional[int] = None,
            source_info: Optional[Dict[str, Any]] = None,
            provenance_hash: str = "",
        ) -> None:
            self.id = id or uuid4().hex[:12]
            self.dataset_id = dataset_id
            self.refreshed_at = refreshed_at or _utcnow()
            self.recorded_at = recorded_at or _utcnow()
            self.data_size_bytes = data_size_bytes
            self.record_count = record_count
            self.source_info = source_info or {}
            self.provenance_hash = provenance_hash

        def to_dict(self) -> Dict[str, Any]:
            """Serialise to dictionary."""
            return {
                "id": self.id,
                "dataset_id": self.dataset_id,
                "refreshed_at": self.refreshed_at.isoformat()
                if self.refreshed_at else None,
                "recorded_at": self.recorded_at.isoformat()
                if self.recorded_at else None,
                "data_size_bytes": self.data_size_bytes,
                "record_count": self.record_count,
                "source_info": dict(self.source_info),
                "provenance_hash": self.provenance_hash,
            }

    class DatasetGroup:  # type: ignore[no-redef]
        """Fallback dataset group without Pydantic."""

        def __init__(
            self,
            id: str = "",
            name: str = "",
            description: str = "",
            dataset_ids: Optional[List[str]] = None,
            priority: str = "medium",
            sla_id: Optional[str] = None,
            created_at: Optional[datetime] = None,
            updated_at: Optional[datetime] = None,
        ) -> None:
            self.id = id or uuid4().hex[:12]
            self.name = name
            self.description = description
            self.dataset_ids = dataset_ids or []
            self.priority = priority
            self.sla_id = sla_id
            self.created_at = created_at or _utcnow()
            self.updated_at = updated_at or _utcnow()

        def to_dict(self) -> Dict[str, Any]:
            """Serialise to dictionary."""
            return {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "dataset_ids": list(self.dataset_ids),
                "priority": self.priority,
                "sla_id": self.sla_id,
                "created_at": self.created_at.isoformat()
                if self.created_at else None,
                "updated_at": self.updated_at.isoformat()
                if self.updated_at else None,
            }


# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.metrics import (
        record_dataset_registered as _record_dataset_registered,
        record_refresh_event as _record_refresh_event,
    )
    _METRICS_AVAILABLE = True

    def record_dataset_registered(status: str) -> None:
        """Delegate to metrics module record_dataset_registered."""
        _record_dataset_registered(status)

    def record_refresh_event(dataset_id: str) -> None:
        """Delegate to metrics module record_refresh_event."""
        _record_refresh_event(dataset_id)

except ImportError:
    _METRICS_AVAILABLE = False

    def record_dataset_registered(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def record_refresh_event(dataset_id: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    logger.info(
        "data_freshness_monitor.metrics not available; "
        "dataset registry metrics disabled"
    )


# ---------------------------------------------------------------------------
# Provenance import (graceful fallback with inline tracker)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "data_freshness_monitor.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-data-freshness-monitor-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            """Initialize with genesis hash."""
            self._last_chain_hash: str = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def hash_record(self, data: Dict[str, Any]) -> str:
            """Compute deterministic SHA-256 hash of a data record.

            Args:
                data: Dictionary to hash.

            Returns:
                Hex-encoded SHA-256 hash string.
            """
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        def add_to_chain(
            self,
            operation: str,
            input_hash: str,
            output_hash: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            """Add a chain link and return the new chain hash.

            Args:
                operation: Name of the operation performed.
                input_hash: SHA-256 hash of the operation input.
                output_hash: SHA-256 hash of the operation output.
                metadata: Optional additional metadata.

            Returns:
                New chain hash linking this entry to the previous.
            """
            timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
            with self._lock:
                combined = json.dumps({
                    "previous": self._last_chain_hash,
                    "input": input_hash,
                    "output": output_hash,
                    "operation": operation,
                    "timestamp": timestamp,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "operation": operation,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "chain_hash": chain_hash,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                })
                self._last_chain_hash = chain_hash

            return chain_hash

        def get_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain.

            Returns:
                List of provenance entries, oldest first.
            """
            with self._lock:
                return list(self._chain)

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _gen_id() -> str:
    """Generate a 12-character hex identifier from uuid4.

    Returns:
        12-character lowercase hex string.
    """
    return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Valid enum value sets for validation
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"active", "inactive", "stale", "error", "pending"}
_VALID_PRIORITIES = {"critical", "high", "medium", "low"}
_VALID_CADENCES = {
    "realtime", "hourly", "daily", "weekly",
    "monthly", "quarterly", "annual",
}


# ---------------------------------------------------------------------------
# DatasetRegistryEngine
# ---------------------------------------------------------------------------


class DatasetRegistryEngine:
    """Pure-Python engine for managing monitored datasets in data freshness monitoring.

    Manages the lifecycle of datasets including registration, refresh event
    recording, grouping, bulk operations, and aggregate statistics. Each
    mutation is tracked with SHA-256 provenance hashing for complete audit
    trails.

    The engine stores all state in-memory using dict-based storage with
    thread-safe access via a threading.Lock. For persistent storage, the
    service layer serialises state to PostgreSQL.

    Attributes:
        _datasets: Registered datasets keyed by dataset id.
        _refresh_history: Refresh events keyed by dataset id.
        _groups: Dataset groups keyed by group id.
        _operation_count: Total number of mutating operations performed.
        _provenance: SHA-256 provenance tracker.
        _lock: Thread-safety lock for concurrent access.

    Example:
        >>> engine = DatasetRegistryEngine()
        >>> ds = engine.register_dataset(
        ...     name="ERP Spend Data",
        ...     source_name="SAP",
        ...     source_type="erp",
        ...     owner="finance-team",
        ...     refresh_cadence="daily",
        ...     priority="high",
        ... )
        >>> assert ds.status in ("active", DatasetStatus.ACTIVE)
        >>> assert engine.get_dataset_count() == 1
    """

    def __init__(self) -> None:
        """Initialize DatasetRegistryEngine with empty state."""
        self._datasets: Dict[str, DatasetDefinition] = {}
        self._refresh_history: Dict[str, List[RefreshEvent]] = {}
        self._groups: Dict[str, DatasetGroup] = {}
        self._operation_count: int = 0
        self._provenance = ProvenanceTracker()
        self._lock = threading.Lock()

        logger.info("DatasetRegistryEngine initialized")

    # ------------------------------------------------------------------
    # 1. register_dataset
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        source_name: str,
        source_type: str = "other",
        owner: str = "",
        refresh_cadence: str = "daily",
        priority: str = "medium",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetDefinition:
        """Register a new dataset for freshness monitoring.

        Creates a DatasetDefinition with a unique ID, validates all
        input parameters, sets the status to active, and records
        provenance.

        Args:
            name: Human-readable name for the dataset.
            source_name: Name of the upstream data source
                (e.g. "SAP", "Snowflake", "S3 bucket").
            source_type: Source type classification. Common values:
                erp, database, api, file, stream, warehouse,
                spreadsheet, other. Defaults to "other".
            owner: Team or individual responsible for this dataset.
            refresh_cadence: Expected refresh frequency. Must be one of:
                realtime, hourly, daily, weekly, monthly, quarterly,
                annual. Defaults to "daily".
            priority: Monitoring priority. Must be one of: critical,
                high, medium, low. Defaults to "medium".
            tags: Optional list of tags for categorisation.
            metadata: Optional dictionary of additional metadata.

        Returns:
            DatasetDefinition with populated id, status, and timestamps.

        Raises:
            ValueError: If name is empty, refresh_cadence is invalid,
                or priority is invalid.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="ERP Spend Data",
            ...     source_name="SAP",
            ...     source_type="erp",
            ...     owner="finance-team",
            ... )
            >>> assert ds.id
            >>> assert ds.name == "ERP Spend Data"
        """
        start = time.time()

        # Validate name
        if not name or not name.strip():
            raise ValueError("Dataset name must not be empty")

        # Validate source_name
        if not source_name or not source_name.strip():
            raise ValueError("Source name must not be empty")

        # Validate refresh_cadence
        cadence_lower = refresh_cadence.lower().strip()
        if cadence_lower not in _VALID_CADENCES:
            raise ValueError(
                f"Invalid refresh_cadence: {refresh_cadence!r}. "
                f"Must be one of: {sorted(_VALID_CADENCES)}"
            )

        # Validate priority
        priority_lower = priority.lower().strip()
        if priority_lower not in _VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {priority!r}. "
                f"Must be one of: {sorted(_VALID_PRIORITIES)}"
            )

        dataset_id = _gen_id()
        now = _utcnow()

        dataset = DatasetDefinition(
            id=dataset_id,
            name=name.strip(),
            source_name=source_name.strip(),
            source_type=source_type.strip().lower() if source_type else "other",
            owner=owner.strip() if owner else "",
            refresh_cadence=cadence_lower,
            priority=priority_lower,
            status="active",
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
            registered_at=now,
            updated_at=now,
        )

        # Compute provenance hash
        input_hash = self._provenance.hash_record({
            "name": dataset.name,
            "source_name": dataset.source_name,
            "source_type": dataset.source_type,
            "owner": dataset.owner,
            "refresh_cadence": dataset.refresh_cadence,
            "priority": dataset.priority,
        })
        output_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "status": "active",
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="register_dataset",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "dataset_id": dataset_id,
                "name": dataset.name,
                "source_name": dataset.source_name,
            },
        )
        dataset.provenance_hash = provenance_hash

        # Store
        with self._lock:
            self._datasets[dataset_id] = dataset
            self._refresh_history[dataset_id] = []
            self._operation_count += 1

        elapsed = time.time() - start
        record_dataset_registered("registered")

        logger.info(
            "Dataset registered: id=%s, name=%s, source=%s, "
            "cadence=%s, priority=%s, %.3fms",
            dataset_id, dataset.name, dataset.source_name,
            dataset.refresh_cadence, dataset.priority,
            elapsed * 1000,
        )
        return dataset

    # ------------------------------------------------------------------
    # 2. get_dataset
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[DatasetDefinition]:
        """Retrieve a registered dataset by its ID.

        Args:
            dataset_id: Unique identifier of the dataset to retrieve.

        Returns:
            DatasetDefinition if found, None otherwise.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="S3", source_type="file",
            ... )
            >>> retrieved = engine.get_dataset(ds.id)
            >>> assert retrieved is not None
            >>> assert retrieved.name == "Test"
        """
        with self._lock:
            dataset = self._datasets.get(dataset_id)

        if dataset is None:
            logger.debug("Dataset not found: id=%s", dataset_id)
            return None

        logger.debug(
            "Dataset retrieved: id=%s, name=%s",
            dataset_id, dataset.name,
        )
        return dataset

    # ------------------------------------------------------------------
    # 3. list_datasets
    # ------------------------------------------------------------------

    def list_datasets(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        source_name: Optional[str] = None,
        cadence: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[DatasetDefinition]:
        """List registered datasets with optional filtering and pagination.

        Filters are combined with AND logic: a dataset must match all
        specified filters to be included in the result.

        Args:
            status: Filter by dataset status (e.g. "active", "stale").
                None to include all statuses.
            priority: Filter by priority (e.g. "high", "critical").
                None to include all priorities.
            source_name: Filter by source name (case-insensitive
                substring match). None to include all sources.
            cadence: Filter by refresh cadence (e.g. "daily", "hourly").
                None to include all cadences.
            limit: Maximum number of results to return. None for all.
            offset: Number of results to skip before returning.
                Defaults to 0.

        Returns:
            List of DatasetDefinition objects matching all filters,
            sorted by name ascending.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> engine.register_dataset(
            ...     name="A", source_name="SAP", priority="high",
            ... )
            >>> engine.register_dataset(
            ...     name="B", source_name="S3", priority="low",
            ... )
            >>> high = engine.list_datasets(priority="high")
            >>> assert len(high) == 1
        """
        with self._lock:
            results = list(self._datasets.values())

        # Apply filters
        if status is not None:
            status_lower = status.lower().strip()
            results = [
                d for d in results
                if self._get_status_value(d) == status_lower
            ]

        if priority is not None:
            priority_lower = priority.lower().strip()
            results = [
                d for d in results
                if self._get_priority_value(d) == priority_lower
            ]

        if source_name is not None:
            source_lower = source_name.lower().strip()
            results = [
                d for d in results
                if source_lower in d.source_name.lower()
            ]

        if cadence is not None:
            cadence_lower = cadence.lower().strip()
            results = [
                d for d in results
                if self._get_cadence_value(d) == cadence_lower
            ]

        # Sort by name ascending
        results.sort(key=lambda d: d.name.lower())

        # Apply pagination
        if offset > 0:
            results = results[offset:]

        if limit is not None and limit > 0:
            results = results[:limit]

        logger.debug(
            "Listed datasets: total=%d, filters=(status=%s, priority=%s, "
            "source=%s, cadence=%s, limit=%s, offset=%d)",
            len(results), status, priority, source_name, cadence,
            limit, offset,
        )
        return results

    # ------------------------------------------------------------------
    # 4. update_dataset
    # ------------------------------------------------------------------

    def update_dataset(
        self,
        dataset_id: str,
        **updates: Any,
    ) -> DatasetDefinition:
        """Update specified fields of an existing dataset.

        Only the fields provided in ``updates`` are updated. All other
        fields retain their current values. The internal version counter
        is incremented and a new provenance hash is computed.

        Supported fields:
            name, source_name, source_type, owner, refresh_cadence,
            priority, status, tags, metadata.

        Args:
            dataset_id: ID of the dataset to update.
            **updates: Field-name/value pairs to update.

        Returns:
            Updated DatasetDefinition.

        Raises:
            KeyError: If dataset_id is not found.
            ValueError: If updated values fail validation or
                unsupported fields are provided.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="SAP",
            ... )
            >>> updated = engine.update_dataset(ds.id, priority="high")
            >>> assert updated.priority in ("high", DatasetPriority.HIGH)
        """
        start = time.time()

        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")
            dataset = self._datasets[dataset_id]

        # Validate updatable fields
        updatable = {
            "name", "source_name", "source_type", "owner",
            "refresh_cadence", "priority", "status", "tags", "metadata",
        }
        invalid_keys = set(updates.keys()) - updatable
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable: {sorted(updatable)}"
            )

        # Track changes
        changes: Dict[str, Any] = {}

        for key, value in updates.items():
            # Validate specific fields
            if key == "name":
                if not value or not str(value).strip():
                    raise ValueError("Dataset name must not be empty")
                value = str(value).strip()

            if key == "source_name":
                if not value or not str(value).strip():
                    raise ValueError("Source name must not be empty")
                value = str(value).strip()

            if key == "refresh_cadence":
                cadence_lower = str(value).lower().strip()
                if cadence_lower not in _VALID_CADENCES:
                    raise ValueError(
                        f"Invalid refresh_cadence: {value!r}. "
                        f"Must be one of: {sorted(_VALID_CADENCES)}"
                    )
                value = cadence_lower

            if key == "priority":
                priority_lower = str(value).lower().strip()
                if priority_lower not in _VALID_PRIORITIES:
                    raise ValueError(
                        f"Invalid priority: {value!r}. "
                        f"Must be one of: {sorted(_VALID_PRIORITIES)}"
                    )
                value = priority_lower

            if key == "status":
                status_lower = str(value).lower().strip()
                if status_lower not in _VALID_STATUSES:
                    raise ValueError(
                        f"Invalid status: {value!r}. "
                        f"Must be one of: {sorted(_VALID_STATUSES)}"
                    )
                value = status_lower

            if key == "tags" and isinstance(value, list):
                value = list(value)

            if key == "metadata" and isinstance(value, dict):
                value = dict(value)

            old_val = getattr(dataset, key, None)
            if old_val != value:
                changes[key] = {"old": str(old_val), "new": str(value)}
                setattr(dataset, key, value)

        # Bump version and update timestamp
        dataset.version += 1
        dataset.updated_at = _utcnow()

        # Compute provenance
        input_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "changes": {k: v["new"] for k, v in changes.items()},
            "version": dataset.version,
        })
        output_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "status": self._get_status_value(dataset),
            "version": dataset.version,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="update_dataset",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "dataset_id": dataset_id,
                "changes": list(changes.keys()),
                "version": dataset.version,
            },
        )
        dataset.provenance_hash = provenance_hash

        with self._lock:
            self._datasets[dataset_id] = dataset
            self._operation_count += 1

        elapsed = time.time() - start

        logger.info(
            "Dataset updated: id=%s, fields=%s, version=%d, %.3fms",
            dataset_id, list(changes.keys()),
            dataset.version, elapsed * 1000,
        )
        return dataset

    # ------------------------------------------------------------------
    # 5. remove_dataset
    # ------------------------------------------------------------------

    def remove_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from the registry.

        Removes the dataset, its refresh history, and detaches it from
        any groups. Records provenance for the removal operation.

        Args:
            dataset_id: ID of the dataset to remove.

        Returns:
            True if the dataset was found and removed, False if the
            dataset was not found.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="S3",
            ... )
            >>> assert engine.remove_dataset(ds.id) is True
            >>> assert engine.get_dataset(ds.id) is None
        """
        start = time.time()

        with self._lock:
            if dataset_id not in self._datasets:
                logger.debug(
                    "Cannot remove: dataset not found id=%s", dataset_id,
                )
                return False

            dataset = self._datasets.pop(dataset_id)
            self._refresh_history.pop(dataset_id, None)

            # Remove from all groups
            for group in self._groups.values():
                if dataset_id in group.dataset_ids:
                    group.dataset_ids.remove(dataset_id)

            self._operation_count += 1

        # Compute provenance
        input_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "name": dataset.name,
        })
        output_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "action": "removed",
        })
        self._provenance.add_to_chain(
            operation="remove_dataset",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "dataset_id": dataset_id,
                "name": dataset.name,
            },
        )

        elapsed = time.time() - start

        logger.info(
            "Dataset removed: id=%s, name=%s, %.3fms",
            dataset_id, dataset.name, elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 6. record_refresh
    # ------------------------------------------------------------------

    def record_refresh(
        self,
        dataset_id: str,
        refreshed_at: Optional[datetime] = None,
        data_size_bytes: Optional[int] = None,
        record_count: Optional[int] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ) -> RefreshEvent:
        """Record a data refresh event for a dataset.

        Creates a RefreshEvent and appends it to the dataset's refresh
        history. Updates the dataset's updated_at timestamp.

        Args:
            dataset_id: ID of the dataset that was refreshed.
            refreshed_at: Timestamp when the refresh occurred. Defaults
                to current UTC time if not provided.
            data_size_bytes: Size of the refreshed data in bytes.
            record_count: Number of records in the refreshed data.
            source_info: Optional metadata about the refresh source
                (e.g. pipeline run id, triggering event).

        Returns:
            RefreshEvent with populated id and timestamps.

        Raises:
            KeyError: If dataset_id is not found.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="SAP",
            ... )
            >>> event = engine.record_refresh(
            ...     ds.id,
            ...     data_size_bytes=1024000,
            ...     record_count=5000,
            ... )
            >>> assert event.dataset_id == ds.id
        """
        start = time.time()

        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")

        now = _utcnow()
        event_id = _gen_id()

        event = RefreshEvent(
            id=event_id,
            dataset_id=dataset_id,
            refreshed_at=refreshed_at or now,
            recorded_at=now,
            data_size_bytes=data_size_bytes,
            record_count=record_count,
            source_info=dict(source_info) if source_info else {},
            provenance_hash="",
        )

        # Compute provenance for the event
        input_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "refreshed_at": str(event.refreshed_at),
            "data_size_bytes": data_size_bytes,
            "record_count": record_count,
        })
        output_hash = self._provenance.hash_record({
            "event_id": event_id,
            "dataset_id": dataset_id,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="record_refresh",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "dataset_id": dataset_id,
                "event_id": event_id,
            },
        )
        event.provenance_hash = provenance_hash

        # Store event and update dataset timestamp
        with self._lock:
            self._refresh_history.setdefault(dataset_id, []).append(event)
            if dataset_id in self._datasets:
                self._datasets[dataset_id].updated_at = now
            self._operation_count += 1

        elapsed = time.time() - start
        record_refresh_event(dataset_id)

        logger.info(
            "Refresh recorded: dataset=%s, event=%s, size=%s, "
            "records=%s, %.3fms",
            dataset_id, event_id,
            data_size_bytes, record_count,
            elapsed * 1000,
        )
        return event

    # ------------------------------------------------------------------
    # 7. get_refresh_history
    # ------------------------------------------------------------------

    def get_refresh_history(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
    ) -> List[RefreshEvent]:
        """Retrieve refresh history for a dataset.

        Returns events sorted by refreshed_at descending (most recent
        first).

        Args:
            dataset_id: ID of the dataset whose history to retrieve.
            limit: Maximum number of events to return. None for all.

        Returns:
            List of RefreshEvent objects, most recent first. Returns
            an empty list if no events exist or dataset is not found.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="SAP",
            ... )
            >>> engine.record_refresh(ds.id)
            >>> engine.record_refresh(ds.id)
            >>> history = engine.get_refresh_history(ds.id, limit=1)
            >>> assert len(history) == 1
        """
        with self._lock:
            events = list(self._refresh_history.get(dataset_id, []))

        # Sort by refreshed_at descending
        events.sort(key=lambda e: e.refreshed_at, reverse=True)

        if limit is not None and limit > 0:
            events = events[:limit]

        logger.debug(
            "Refresh history retrieved: dataset=%s, events=%d",
            dataset_id, len(events),
        )
        return events

    # ------------------------------------------------------------------
    # 8. get_last_refresh
    # ------------------------------------------------------------------

    def get_last_refresh(
        self,
        dataset_id: str,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent refresh for a dataset.

        Args:
            dataset_id: ID of the dataset to query.

        Returns:
            datetime of the most recent refresh, or None if no refresh
            events have been recorded for this dataset.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="Test", source_name="SAP",
            ... )
            >>> assert engine.get_last_refresh(ds.id) is None
            >>> engine.record_refresh(ds.id)
            >>> assert engine.get_last_refresh(ds.id) is not None
        """
        with self._lock:
            events = self._refresh_history.get(dataset_id, [])

        if not events:
            logger.debug(
                "No refresh events for dataset: %s", dataset_id,
            )
            return None

        # Find the most recent refreshed_at
        latest = max(events, key=lambda e: e.refreshed_at)

        logger.debug(
            "Last refresh for dataset=%s: %s",
            dataset_id, latest.refreshed_at.isoformat(),
        )
        return latest.refreshed_at

    # ------------------------------------------------------------------
    # 9. create_group
    # ------------------------------------------------------------------

    def create_group(
        self,
        name: str,
        description: str = "",
        dataset_ids: Optional[List[str]] = None,
        priority: str = "medium",
        sla_id: Optional[str] = None,
    ) -> DatasetGroup:
        """Create a new dataset group for batch monitoring.

        Groups allow related datasets to be monitored together with
        shared SLA definitions and priority levels.

        Args:
            name: Human-readable name for the group.
            description: Optional description of the group's purpose.
            dataset_ids: Optional list of dataset IDs to include.
                Non-existent IDs are silently filtered out.
            priority: Group priority level. Must be one of: critical,
                high, medium, low. Defaults to "medium".
            sla_id: Optional SLA definition ID to associate with
                this group.

        Returns:
            DatasetGroup with populated id and timestamps.

        Raises:
            ValueError: If name is empty or priority is invalid.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="A", source_name="SAP",
            ... )
            >>> group = engine.create_group(
            ...     name="Finance Datasets",
            ...     dataset_ids=[ds.id],
            ...     priority="high",
            ... )
            >>> assert ds.id in group.dataset_ids
        """
        start = time.time()

        # Validate name
        if not name or not name.strip():
            raise ValueError("Group name must not be empty")

        # Validate priority
        priority_lower = priority.lower().strip()
        if priority_lower not in _VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {priority!r}. "
                f"Must be one of: {sorted(_VALID_PRIORITIES)}"
            )

        # Filter dataset_ids to only existing datasets
        valid_ids: List[str] = []
        if dataset_ids:
            with self._lock:
                for did in dataset_ids:
                    if did in self._datasets:
                        valid_ids.append(did)
                    else:
                        logger.warning(
                            "Dataset %s not found; excluded from group",
                            did,
                        )

        group_id = _gen_id()
        now = _utcnow()

        group = DatasetGroup(
            id=group_id,
            name=name.strip(),
            description=description.strip() if description else "",
            dataset_ids=valid_ids,
            priority=priority_lower,
            sla_id=sla_id,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._groups[group_id] = group
            self._operation_count += 1

        elapsed = time.time() - start

        logger.info(
            "Group created: id=%s, name=%s, datasets=%d, "
            "priority=%s, %.3fms",
            group_id, group.name, len(valid_ids),
            group.priority, elapsed * 1000,
        )
        return group

    # ------------------------------------------------------------------
    # 10. get_group
    # ------------------------------------------------------------------

    def get_group(
        self,
        group_id: str,
    ) -> Optional[DatasetGroup]:
        """Retrieve a dataset group by its ID.

        Args:
            group_id: Unique identifier of the group to retrieve.

        Returns:
            DatasetGroup if found, None otherwise.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> group = engine.create_group(name="Test Group")
            >>> retrieved = engine.get_group(group.id)
            >>> assert retrieved is not None
            >>> assert retrieved.name == "Test Group"
        """
        with self._lock:
            group = self._groups.get(group_id)

        if group is None:
            logger.debug("Group not found: id=%s", group_id)
            return None

        logger.debug(
            "Group retrieved: id=%s, name=%s",
            group_id, group.name,
        )
        return group

    # ------------------------------------------------------------------
    # 11. list_groups
    # ------------------------------------------------------------------

    def list_groups(self) -> List[DatasetGroup]:
        """List all dataset groups.

        Returns:
            List of DatasetGroup objects sorted by name ascending.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> engine.create_group(name="B Group")
            >>> engine.create_group(name="A Group")
            >>> groups = engine.list_groups()
            >>> assert groups[0].name == "A Group"
        """
        with self._lock:
            groups = list(self._groups.values())

        groups.sort(key=lambda g: g.name.lower())

        logger.debug("Listed groups: total=%d", len(groups))
        return groups

    # ------------------------------------------------------------------
    # 12. add_to_group
    # ------------------------------------------------------------------

    def add_to_group(
        self,
        group_id: str,
        dataset_id: str,
    ) -> DatasetGroup:
        """Add a dataset to an existing group.

        If the dataset is already in the group, this is a no-op and
        the group is returned unchanged.

        Args:
            group_id: ID of the group to add the dataset to.
            dataset_id: ID of the dataset to add.

        Returns:
            Updated DatasetGroup.

        Raises:
            KeyError: If group_id or dataset_id is not found.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="A", source_name="SAP",
            ... )
            >>> group = engine.create_group(name="Test Group")
            >>> group = engine.add_to_group(group.id, ds.id)
            >>> assert ds.id in group.dataset_ids
        """
        with self._lock:
            if group_id not in self._groups:
                raise KeyError(f"Group not found: {group_id}")
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")

            group = self._groups[group_id]

            if dataset_id in group.dataset_ids:
                logger.debug(
                    "Dataset %s already in group %s",
                    dataset_id, group_id,
                )
                return group

            group.dataset_ids.append(dataset_id)
            self._operation_count += 1

        logger.info(
            "Dataset %s added to group %s (%s)",
            dataset_id, group_id, group.name,
        )
        return group

    # ------------------------------------------------------------------
    # 13. remove_from_group
    # ------------------------------------------------------------------

    def remove_from_group(
        self,
        group_id: str,
        dataset_id: str,
    ) -> DatasetGroup:
        """Remove a dataset from an existing group.

        If the dataset is not in the group, this is a no-op and
        the group is returned unchanged.

        Args:
            group_id: ID of the group to remove the dataset from.
            dataset_id: ID of the dataset to remove.

        Returns:
            Updated DatasetGroup.

        Raises:
            KeyError: If group_id is not found.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> ds = engine.register_dataset(
            ...     name="A", source_name="SAP",
            ... )
            >>> group = engine.create_group(
            ...     name="Test Group", dataset_ids=[ds.id],
            ... )
            >>> group = engine.remove_from_group(group.id, ds.id)
            >>> assert ds.id not in group.dataset_ids
        """
        with self._lock:
            if group_id not in self._groups:
                raise KeyError(f"Group not found: {group_id}")

            group = self._groups[group_id]

            if dataset_id not in group.dataset_ids:
                logger.debug(
                    "Dataset %s not in group %s",
                    dataset_id, group_id,
                )
                return group

            group.dataset_ids.remove(dataset_id)
            self._operation_count += 1

        logger.info(
            "Dataset %s removed from group %s (%s)",
            dataset_id, group_id, group.name,
        )
        return group

    # ------------------------------------------------------------------
    # 14. bulk_register
    # ------------------------------------------------------------------

    def bulk_register(
        self,
        datasets: List[Dict[str, Any]],
    ) -> List[DatasetDefinition]:
        """Register multiple datasets in a single batch operation.

        Each entry in the list is a dictionary with the same keys
        accepted by ``register_dataset``. Datasets that fail validation
        are skipped with a warning; successfully registered datasets
        are returned.

        Args:
            datasets: List of dictionaries, each containing dataset
                parameters (name, source_name, source_type, etc.).

        Returns:
            List of successfully registered DatasetDefinition objects.

        Raises:
            ValueError: If datasets list is empty.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> results = engine.bulk_register([
            ...     {"name": "A", "source_name": "SAP", "source_type": "erp"},
            ...     {"name": "B", "source_name": "S3", "source_type": "file"},
            ... ])
            >>> assert len(results) == 2
        """
        start = time.time()

        if not datasets:
            raise ValueError("Datasets list must not be empty")

        registered: List[DatasetDefinition] = []
        failed_count = 0

        for idx, ds_data in enumerate(datasets):
            try:
                dataset = self.register_dataset(
                    name=ds_data.get("name", ""),
                    source_name=ds_data.get("source_name", ""),
                    source_type=ds_data.get("source_type", "other"),
                    owner=ds_data.get("owner", ""),
                    refresh_cadence=ds_data.get("refresh_cadence", "daily"),
                    priority=ds_data.get("priority", "medium"),
                    tags=ds_data.get("tags"),
                    metadata=ds_data.get("metadata"),
                )
                registered.append(dataset)
            except (ValueError, TypeError) as exc:
                failed_count += 1
                logger.warning(
                    "Bulk register: entry %d failed: %s", idx, str(exc),
                )

        elapsed = time.time() - start

        logger.info(
            "Bulk register complete: registered=%d, failed=%d, %.3fms",
            len(registered), failed_count, elapsed * 1000,
        )
        return registered

    # ------------------------------------------------------------------
    # 15. get_datasets_by_source
    # ------------------------------------------------------------------

    def get_datasets_by_source(
        self,
        source_name: str,
    ) -> List[DatasetDefinition]:
        """Retrieve all datasets originating from a specific source.

        Performs case-insensitive exact matching on the source_name
        field.

        Args:
            source_name: The source name to filter by.

        Returns:
            List of DatasetDefinition objects with matching source_name,
            sorted by name ascending.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> engine.register_dataset(
            ...     name="A", source_name="SAP",
            ... )
            >>> engine.register_dataset(
            ...     name="B", source_name="SAP",
            ... )
            >>> engine.register_dataset(
            ...     name="C", source_name="S3",
            ... )
            >>> sap_datasets = engine.get_datasets_by_source("SAP")
            >>> assert len(sap_datasets) == 2
        """
        source_lower = source_name.lower().strip()

        with self._lock:
            results = [
                d for d in self._datasets.values()
                if d.source_name.lower() == source_lower
            ]

        results.sort(key=lambda d: d.name.lower())

        logger.debug(
            "Datasets by source=%s: count=%d",
            source_name, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # 16. get_dataset_count
    # ------------------------------------------------------------------

    def get_dataset_count(self) -> int:
        """Return the total number of registered datasets.

        Returns:
            Integer count of datasets in the registry.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> assert engine.get_dataset_count() == 0
            >>> engine.register_dataset(
            ...     name="A", source_name="SAP",
            ... )
            >>> assert engine.get_dataset_count() == 1
        """
        with self._lock:
            return len(self._datasets)

    # ------------------------------------------------------------------
    # 17. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all registered datasets.

        Returns counts grouped by status, priority, and refresh cadence,
        plus total dataset count, total refresh events, total groups,
        and total operations.

        Returns:
            Dictionary with the following structure:
            {
                "total_datasets": int,
                "total_refresh_events": int,
                "total_groups": int,
                "total_operations": int,
                "by_status": {"active": int, "stale": int, ...},
                "by_priority": {"critical": int, "high": int, ...},
                "by_cadence": {"daily": int, "hourly": int, ...},
                "provenance_chain_length": int,
            }

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> engine.register_dataset(
            ...     name="A", source_name="SAP", priority="high",
            ... )
            >>> stats = engine.get_statistics()
            >>> assert stats["total_datasets"] == 1
            >>> assert stats["by_priority"]["high"] == 1
        """
        with self._lock:
            datasets = list(self._datasets.values())
            total_events = sum(
                len(events)
                for events in self._refresh_history.values()
            )
            total_groups = len(self._groups)
            total_ops = self._operation_count

        # Count by status
        by_status: Dict[str, int] = {}
        for ds in datasets:
            status_val = self._get_status_value(ds)
            by_status[status_val] = by_status.get(status_val, 0) + 1

        # Count by priority
        by_priority: Dict[str, int] = {}
        for ds in datasets:
            priority_val = self._get_priority_value(ds)
            by_priority[priority_val] = by_priority.get(priority_val, 0) + 1

        # Count by cadence
        by_cadence: Dict[str, int] = {}
        for ds in datasets:
            cadence_val = self._get_cadence_value(ds)
            by_cadence[cadence_val] = by_cadence.get(cadence_val, 0) + 1

        stats = {
            "total_datasets": len(datasets),
            "total_refresh_events": total_events,
            "total_groups": total_groups,
            "total_operations": total_ops,
            "by_status": by_status,
            "by_priority": by_priority,
            "by_cadence": by_cadence,
            "provenance_chain_length": self._provenance.entry_count,
        }

        logger.debug(
            "Statistics computed: datasets=%d, events=%d, groups=%d",
            len(datasets), total_events, total_groups,
        )
        return stats

    # ------------------------------------------------------------------
    # 18. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all registry state.

        Removes all datasets, refresh history, groups, and resets the
        operation counter. The provenance tracker is also re-initialised.
        Primarily used for testing.

        Example:
            >>> engine = DatasetRegistryEngine()
            >>> engine.register_dataset(
            ...     name="Test", source_name="SAP",
            ... )
            >>> assert engine.get_dataset_count() == 1
            >>> engine.reset()
            >>> assert engine.get_dataset_count() == 0
        """
        with self._lock:
            self._datasets.clear()
            self._refresh_history.clear()
            self._groups.clear()
            self._operation_count = 0

        self._provenance = ProvenanceTracker()

        logger.info("DatasetRegistryEngine state reset")

    # ------------------------------------------------------------------
    # Introspection helpers (properties)
    # ------------------------------------------------------------------

    @property
    def dataset_count(self) -> int:
        """Return the total number of registered datasets."""
        with self._lock:
            return len(self._datasets)

    @property
    def group_count(self) -> int:
        """Return the total number of dataset groups."""
        with self._lock:
            return len(self._groups)

    @property
    def operation_count(self) -> int:
        """Return the total number of mutating operations performed."""
        with self._lock:
            return self._operation_count

    @property
    def provenance_chain_length(self) -> int:
        """Return the number of entries in the provenance chain."""
        return self._provenance.entry_count

    def get_all_dataset_ids(self) -> List[str]:
        """Return all registered dataset IDs.

        Returns:
            List of dataset ID strings, sorted alphabetically.
        """
        with self._lock:
            return sorted(self._datasets.keys())

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Return the full provenance chain for audit.

        Returns:
            List of provenance entries, oldest first.
        """
        return self._provenance.get_chain()

    # ------------------------------------------------------------------
    # Private helpers: enum value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_status_value(dataset: DatasetDefinition) -> str:
        """Extract status string from a DatasetDefinition.

        Handles both enum and plain string status attributes.

        Args:
            dataset: The dataset definition.

        Returns:
            Lowercase status string.
        """
        status = dataset.status
        if isinstance(status, str):
            return status.lower()
        if hasattr(status, "value"):
            return str(status.value).lower()
        return str(status).lower()

    @staticmethod
    def _get_priority_value(dataset: DatasetDefinition) -> str:
        """Extract priority string from a DatasetDefinition.

        Handles both enum and plain string priority attributes.

        Args:
            dataset: The dataset definition.

        Returns:
            Lowercase priority string.
        """
        priority = dataset.priority
        if isinstance(priority, str):
            return priority.lower()
        if hasattr(priority, "value"):
            return str(priority.value).lower()
        return str(priority).lower()

    @staticmethod
    def _get_cadence_value(dataset: DatasetDefinition) -> str:
        """Extract refresh_cadence string from a DatasetDefinition.

        Handles both enum and plain string cadence attributes.

        Args:
            dataset: The dataset definition.

        Returns:
            Lowercase cadence string.
        """
        cadence = dataset.refresh_cadence
        if isinstance(cadence, str):
            return cadence.lower()
        if hasattr(cadence, "value"):
            return str(cadence.value).lower()
        return str(cadence).lower()


# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = ["DatasetRegistryEngine"]
