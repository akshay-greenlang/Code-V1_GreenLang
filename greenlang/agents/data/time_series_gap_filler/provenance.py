# -*- coding: utf-8 -*-
"""
Provenance Tracking for Time Series Gap Filler Agent - AGENT-DATA-014

Thin shim that delegates core hashing to the shared
``greenlang.data_commons.provenance`` base class while preserving the
extended time-series-gap-filler API (``ProvenanceEntry`` dataclass,
``add_entry``, ``add_to_chain``, entity-scoped ``verify_chain``,
``get_entry``, ``clear``/``reset``, and singleton helper).

Example:
    >>> from greenlang.agents.data.time_series_gap_filler.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry_id = tracker.record("gap_fill_job", "job_001", "fill", "abc123")
    >>> valid, chain = tracker.verify_chain("gap_fill_job", "job_001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)


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
        """Convert entry to a dictionary for serialization."""
        return asdict(self)


class ProvenanceTracker(_BaseProvenanceTracker):
    """Provenance tracker for time series gap filling operations.

    Extends the shared base class with ``ProvenanceEntry`` dataclass
    support, ``add_entry``/``add_to_chain`` methods, and entity-scoped
    chain verification.
    """

    def __init__(self) -> None:
        """Initialize with time-series-gap-filler genesis hash."""
        super().__init__(agent_name="time-series-gap-filler")
        self._lock = threading.Lock()

    # -- Hashing -----------------------------------------------------------

    def hash_record(self, data: Dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hash of a data record.

        Args:
            data: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -- Chain entry methods -----------------------------------------------

    def add_entry(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Add a provenance entry to the chain.

        Args:
            operation: Name of the operation (detect_gaps, detect_frequency,
                select_strategy, fill_gaps, validate, report, pipeline).
            input_hash: SHA-256 hash of the operation input.
            output_hash: SHA-256 hash of the operation output.
            metadata: Optional additional metadata to include.

        Returns:
            The created ProvenanceEntry with computed chain hash.
        """
        timestamp = utcnow().isoformat()
        meta = metadata or {}
        entry_id = str(uuid4())

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_tsgf_chain_hash(
                parent_hash, input_hash, output_hash, operation, timestamp,
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
            self._global_chain.append(entry.to_dict())
            self._last_chain_hash = chain_hash

        logger.debug(
            "Chain entry added: op=%s in=%s out=%s chain=%s",
            operation, input_hash[:16], output_hash[:16], chain_hash[:16],
        )
        return entry

    def add_to_chain(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a chain link, returning only the chain hash string.

        Args:
            operation: Name of the operation.
            input_hash: SHA-256 hash of the operation input.
            output_hash: SHA-256 hash of the operation output.
            metadata: Optional additional metadata to include.

        Returns:
            New chain hash linking this entry to the previous.
        """
        entry = self.add_entry(operation, input_hash, output_hash, metadata)
        return entry.chain_hash

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry for an entity operation.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            action: Action performed.
            data_hash: SHA-256 hash of the operation data.
            user_id: User who performed the operation.

        Returns:
            Chain hash of the new entry.
        """
        timestamp = utcnow().isoformat()
        store_key = f"{entity_type}:{entity_id}"
        entry: Dict[str, Any] = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": timestamp,
            "chain_hash": "",
        }
        with self._lock:
            chain_hash = self._compute_tsgf_chain_hash(
                self._last_chain_hash, data_hash, data_hash,
                action, timestamp,
            )
            entry["chain_hash"] = chain_hash
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash
        return chain_hash

    # -- Verification / retrieval ------------------------------------------

    def verify_chain(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of the provenance chain.

        Args:
            entity_type: Optional type of entity to verify.
            entity_id: Optional entity ID whose chain to verify.

        Returns:
            Tuple of (is_valid, chain_entries).
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
            if i == 0 and not entry.get("chain_hash"):
                is_valid = False
                break
            required = [
                "action" if "action" in entry else "operation",
                "timestamp", "chain_hash",
            ]
            if "entity_type" in entry:
                required.extend(["entity_type", "entity_id", "data_hash"])
            else:
                required.extend(["input_hash", "output_hash"])
            for field_name in required:
                if field_name not in entry:
                    is_valid = False
                    logger.warning(
                        "Chain verification failed at entry %d: missing field %s",
                        i, field_name,
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
        """Get the provenance chain, optionally scoped to an entity.

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

    def clear(self) -> None:
        """Clear the provenance tracker. Alias for reset()."""
        self.reset()

    def reset(self) -> None:
        """Reset the provenance tracker to genesis state."""
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self._GENESIS_HASH
        logger.info("ProvenanceTracker reset to genesis")

    # -- Properties --------------------------------------------------------

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

    def export_json(self) -> str:
        """Export all provenance records as JSON string."""
        with self._lock:
            data = list(self._global_chain)
        return json.dumps(data, indent=2, default=str)

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _compute_tsgf_chain_hash(
        previous_hash: str,
        input_hash: str,
        output_hash: str,
        operation: str,
        timestamp: str,
    ) -> str:
        """Compute the next chain hash (5-arg variant for time series gap filler).

        Args:
            previous_hash: Previous chain hash.
            input_hash: Hash of the operation input.
            output_hash: Hash of the operation output.
            operation: Operation or action name.
            timestamp: ISO-formatted timestamp.

        Returns:
            New SHA-256 chain hash.
        """
        combined = json.dumps({
            "previous": previous_hash,
            "input": input_hash,
            "output": output_hash,
            "operation": operation,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.Lock()


def get_provenance_tracker() -> ProvenanceTracker:
    """Return the singleton ProvenanceTracker instance.

    Thread-safe lazy initialization.

    Returns:
        The global ProvenanceTracker singleton.
    """
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = ProvenanceTracker()
                logger.info(
                    "Singleton ProvenanceTracker created "
                    "(time series gap filler)"
                )
    return _tracker_instance


__all__ = [
    "ProvenanceEntry",
    "ProvenanceTracker",
    "get_provenance_tracker",
]
