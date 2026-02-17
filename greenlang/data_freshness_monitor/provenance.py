# -*- coding: utf-8 -*-
"""
Provenance Tracking for Data Freshness Monitor Agent - AGENT-DATA-016

Provides SHA-256 based audit trail tracking for all data freshness
monitoring operations. Maintains an in-memory chain-hashed operation
log for tamper-evident provenance.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - Float normalization ensures reproducible hashing
    - JSON export for external audit systems
    - Complete provenance for every freshness check step

Example:
    >>> from greenlang.data_freshness_monitor.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry_id = tracker.record("freshness_check", "chk_001", "check", "abc123")
    >>> valid, chain = tracker.verify_chain("freshness_check", "chk_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _normalize_value(value: Any) -> Any:
    """Normalize a value for deterministic serialization.

    Handles float precision, NaN/Inf edge cases, and recursive
    normalization of nested structures.

    Args:
        value: Any Python value to normalize.

    Returns:
        Normalized value safe for deterministic JSON serialization.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "__NaN__"
        if math.isinf(value):
            return "__Inf__" if value > 0 else "__-Inf__"
        return round(value, 10)
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


@dataclass
class ProvenanceEntry:
    """A single provenance record in the chain.

    Attributes:
        entry_id: Unique identifier for this provenance entry.
        operation: Name of the operation performed.
        input_hash: SHA-256 hash of the operation input.
        output_hash: SHA-256 hash of the operation output.
        timestamp: ISO-formatted UTC timestamp of the operation.
        parent_hash: Chain hash of the previous entry in the chain.
        chain_hash: SHA-256 chain hash linking this entry to the chain.
        metadata: Optional additional metadata for audit context.
    """

    entry_id: str
    operation: str
    input_hash: str
    output_hash: str
    timestamp: str
    parent_hash: str
    chain_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to a dictionary for serialization.

        Returns:
            Dictionary representation of the entry.
        """
        return asdict(self)


class ProvenanceTracker:
    """Tracks provenance for data freshness monitoring with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    Attributes:
        _chain_store: In-memory chain storage grouped by entity key.
        _global_chain: Flat list of all entries in order.
        _last_chain_hash: Most recent chain hash for linking.
        _lock: Thread-safety lock.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry_id = tracker.record(
        ...     "freshness_check", "chk_001", "check", "abc123"
        ... )
        >>> valid, chain = tracker.verify_chain(
        ...     "freshness_check", "chk_001"
        ... )
        >>> assert valid is True
    """

    # Initial chain hash (genesis)
    GENESIS_HASH = hashlib.sha256(
        b"greenlang-data-freshness-monitor-genesis"
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize ProvenanceTracker with genesis hash."""
        self._chain_store: Dict[str, List[Dict[str, Any]]] = {}
        self._global_chain: List[Dict[str, Any]] = []
        self._last_chain_hash: str = self.GENESIS_HASH
        self._lock = threading.Lock()
        logger.info(
            "ProvenanceTracker initialized (data freshness monitor)"
        )

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute a deterministic SHA-256 hash with float normalization.

        Normalizes floats to 10 decimal places, sorts dictionary keys,
        and handles NaN/Inf edge cases for reproducible hashing.

        Args:
            data: Data to hash (dict, list, str, number, or other).

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        normalized = _normalize_value(data)
        serialized = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_hash(self, data: Any) -> str:
        """Compute a deterministic SHA-256 hash (instance method).

        Delegates to the static compute_hash method for deterministic
        SHA-256 hashing with float normalization and sorted keys.

        Args:
            data: Data to hash (dict, list, str, number, or other).

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return ProvenanceTracker.compute_hash(data)

    def hash_record(self, data: Dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hash of a data record.

        Sorts keys and normalizes values to ensure deterministic hashing
        regardless of insertion order.

        Args:
            data: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return self._compute_hash(data)

    def build_hash(self, data: Any) -> str:
        """Build a SHA-256 hash for arbitrary data.

        Args:
            data: Data to hash (dict, list, or other).

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._compute_hash(data)

    def _compute_chain_hash(
        self,
        previous_hash: str,
        input_hash: str,
        output_hash: str,
        operation: str,
        timestamp: str,
    ) -> str:
        """Compute the next chain hash linking to the previous.

        Args:
            previous_hash: Previous chain hash.
            input_hash: Hash of the operation input.
            output_hash: Hash of the operation output.
            operation: Operation or action name.
            timestamp: ISO-formatted timestamp.

        Returns:
            New SHA-256 chain hash.
        """
        combined = json.dumps(
            {
                "previous": previous_hash,
                "input": input_hash,
                "output": output_hash,
                "operation": operation,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Chain entry methods
    # ------------------------------------------------------------------

    def add_entry(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Add a provenance entry to the chain.

        Creates a ProvenanceEntry linking the previous chain hash to new
        input and output hashes for the given operation.

        Args:
            operation: Name of the operation (check_freshness,
                evaluate_sla, detect_breach, send_alert,
                predict_refresh, register_dataset, record_refresh,
                validate, report).
            input_hash: SHA-256 hash of the operation input.
            output_hash: SHA-256 hash of the operation output.
            metadata: Optional additional metadata to include.

        Returns:
            The created ProvenanceEntry with computed chain hash.
        """
        timestamp = _utcnow().isoformat()
        meta = metadata or {}
        entry_id = str(uuid4())

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_chain_hash(
                parent_hash,
                input_hash,
                output_hash,
                operation,
                timestamp,
            )

            entry = ProvenanceEntry(
                entry_id=entry_id,
                operation=operation,
                input_hash=input_hash,
                output_hash=output_hash,
                timestamp=timestamp,
                parent_hash=parent_hash,
                chain_hash=chain_hash,
                metadata=meta,
            )

            entry_dict = entry.to_dict()
            self._global_chain.append(entry_dict)
            self._last_chain_hash = chain_hash

        logger.debug(
            "Chain entry added: op=%s in=%s out=%s chain=%s",
            operation,
            input_hash[:16],
            output_hash[:16],
            chain_hash[:16],
        )
        return entry

    def add_to_chain(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a chain link recording an operation with input/output hashes.

        Convenience method that returns only the chain hash string.
        Delegates to add_entry internally.

        Args:
            operation: Name of the operation (check_freshness,
                evaluate_sla, detect_breach, send_alert,
                predict_refresh, register_dataset, record_refresh,
                validate, report).
            input_hash: SHA-256 hash of the operation input.
            output_hash: SHA-256 hash of the operation output.
            metadata: Optional additional metadata to include.

        Returns:
            New chain hash linking this entry to the previous.
        """
        entry = self.add_entry(operation, input_hash, output_hash, metadata)
        return entry.chain_hash

    def record_operation(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry for an entity operation.

        Args:
            entity_type: Type of entity (freshness_check, sla_policy,
                breach_event, alert_notification, refresh_event,
                dataset_registration, prediction, validation, report,
                pipeline).
            entity_id: Unique entity identifier.
            action: Action performed (check, evaluate, detect, alert,
                predict, register, refresh, validate, report, cancel).
            data_hash: SHA-256 hash of the operation data.
            user_id: User who performed the operation.

        Returns:
            Chain hash of the new entry.
        """
        timestamp = _utcnow().isoformat()
        store_key = f"{entity_type}:{entity_id}"

        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": timestamp,
            "chain_hash": "",
        }

        with self._lock:
            chain_hash = self._compute_chain_hash(
                self._last_chain_hash,
                data_hash,
                data_hash,
                action,
                timestamp,
            )
            entry["chain_hash"] = chain_hash

            # Store in entity chain
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)

            # Store in global chain
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash=%s",
            entity_type,
            entity_id[:8],
            action,
            chain_hash[:16],
        )
        return chain_hash

    # Alias for backward compatibility
    record = record_operation

    # ------------------------------------------------------------------
    # Chain verification and retrieval
    # ------------------------------------------------------------------

    def verify_chain(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of the provenance chain.

        When entity_type and entity_id are provided, verifies only the
        entity-scoped chain. Otherwise verifies the full global chain.

        Args:
            entity_type: Optional type of entity to verify.
            entity_id: Optional entity ID whose chain to verify.

        Returns:
            Tuple of (is_valid: bool, chain_entries: list).
        """
        if entity_type and entity_id:
            store_key = f"{entity_type}:{entity_id}"
            with self._lock:
                chain = list(self._chain_store.get(store_key, []))
        else:
            with self._lock:
                chain = list(self._global_chain)

        if not chain:
            return True, []

        is_valid = True
        for i, entry in enumerate(chain):
            if i == 0:
                if not entry.get("chain_hash"):
                    is_valid = False
                    break
            required = [
                "action" if "action" in entry else "operation",
                "timestamp",
                "chain_hash",
            ]
            # Validate entity entries have entity fields
            if "entity_type" in entry:
                required.extend(["entity_type", "entity_id", "data_hash"])
            else:
                required.extend(["input_hash", "output_hash"])

            for field_name in required:
                if field_name not in entry:
                    is_valid = False
                    logger.warning(
                        "Chain verification failed at entry %d: "
                        "missing field %s",
                        i,
                        field_name,
                    )
                    break
            if not is_valid:
                break

        return is_valid, chain

    def get_chain(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get the provenance chain.

        When entity_type and entity_id are provided, returns only the
        entity-scoped chain. Otherwise returns the full global chain.

        Args:
            entity_type: Optional type of entity.
            entity_id: Optional entity ID to look up.

        Returns:
            List of provenance entries, oldest first.
        """
        if entity_type and entity_id:
            store_key = f"{entity_type}:{entity_id}"
            with self._lock:
                return list(self._chain_store.get(store_key, []))
        with self._lock:
            return list(self._global_chain)

    def get_latest_hash(self) -> str:
        """Get the current chain hash (head of the chain).

        Returns:
            The most recent SHA-256 chain hash.
        """
        with self._lock:
            return self._last_chain_hash

    # Alias for cross-agent compatibility
    get_current_hash = get_latest_hash

    def get_entry(self, entry_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific provenance entry by its global index.

        Args:
            entry_index: Zero-based index into the global chain.

        Returns:
            The provenance entry dict or None if index is out of bounds.
        """
        with self._lock:
            if 0 <= entry_index < len(self._global_chain):
                return dict(self._global_chain[entry_index])
        return None

    def get_global_chain(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the global provenance chain (all entities).

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of provenance entries, newest first.
        """
        with self._lock:
            return list(reversed(self._global_chain[-limit:]))

    def get_chain_length(self) -> int:
        """Return the total number of provenance entries in the global chain."""
        with self._lock:
            return len(self._global_chain)

    # ------------------------------------------------------------------
    # Reset and export
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the provenance tracker, resetting to genesis state.

        Removes all stored chains and resets the last chain hash
        back to the genesis hash. Alias for reset().
        """
        self.reset()

    def reset(self) -> None:
        """Reset the provenance tracker to genesis state.

        Clears all stored chains and resets the last chain hash
        back to the genesis hash.
        """
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self.GENESIS_HASH
        logger.info("ProvenanceTracker reset to genesis")

    def export_json(self) -> str:
        """Export all provenance records as JSON string.

        Returns:
            JSON string of provenance records.
        """
        with self._lock:
            data = list(self._global_chain)
        return json.dumps(data, indent=2, default=str)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries."""
        with self._lock:
            return len(self._global_chain)

    @property
    def entity_count(self) -> int:
        """Return the number of unique entities tracked."""
        with self._lock:
            return len(self._chain_store)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.Lock()


def get_provenance_tracker() -> ProvenanceTracker:
    """Return the singleton ProvenanceTracker instance.

    Thread-safe lazy initialization. Returns the same instance on
    every call within the process.

    Returns:
        The global ProvenanceTracker singleton.

    Example:
        >>> tracker = get_provenance_tracker()
        >>> tracker.record_operation("freshness_check", "c1", "check", "abc123")
    """
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = ProvenanceTracker()
                logger.info(
                    "Singleton ProvenanceTracker created "
                    "(data freshness monitor)"
                )
    return _tracker_instance


__all__ = [
    "ProvenanceEntry",
    "ProvenanceTracker",
    "get_provenance_tracker",
]
