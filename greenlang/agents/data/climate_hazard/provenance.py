# -*- coding: utf-8 -*-
"""
Provenance Tracking for Climate Hazard Connector - AGENT-DATA-020

Thin shim that delegates core chain-hashing to the shared
``greenlang.data_commons.provenance`` base class while preserving the
extended climate-hazard-specific API (``ProvenanceEntry`` dataclass,
thread-safe singleton helpers, ``verify_chain()`` returning ``bool``,
and ``record()`` returning a ``ProvenanceEntry``).

Example:
    >>> from greenlang.agents.data.climate_hazard.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("hazard_source", "register_source", "src_001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from greenlang.data_commons.provenance import ProvenanceTracker as _BaseProvenanceTracker
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceEntry:
    """A single tamper-evident provenance record for a climate hazard event."""

    entity_type: str
    entity_id: str
    action: str
    hash_value: str
    parent_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary."""
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "hash_value": self.hash_value,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    "hazard_source", "hazard_data", "risk_index", "scenario_projection",
    "asset", "exposure", "vulnerability", "compliance_report",
})

VALID_ACTIONS = frozenset({
    "register_source", "update_source", "delete_source",
    "ingest_data", "query_data", "aggregate_data",
    "calculate_risk", "calculate_multi_hazard", "calculate_compound",
    "rank_hazards", "project_scenario", "project_multi",
    "project_timeseries", "register_asset", "update_asset",
    "delete_asset", "assess_exposure", "assess_portfolio",
    "assess_supply_chain", "identify_hotspots", "score_vulnerability",
    "score_sector", "create_sensitivity", "create_adaptive",
    "calculate_residual", "rank_entities", "generate_report",
    "generate_tcfd", "generate_csrd", "generate_taxonomy",
    "run_pipeline", "run_batch", "clear_engine", "export_data",
    "import_data", "search_data",
})


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker(_BaseProvenanceTracker):
    """Climate hazard provenance tracker with extended dataclass-based API.

    Inherits genesis-hash derivation from the shared base class while
    maintaining the ``ProvenanceEntry`` chain, thread-safe state, and
    the full climate-hazard-specific public API.
    """

    def __init__(
        self, genesis_hash: str = "greenlang-climate-hazard-connector-genesis",
    ) -> None:
        """Initialize with climate-hazard genesis hash.

        Args:
            genesis_hash: String used to compute the immutable genesis hash.
        """
        # Compute genesis hash locally (matches original behavior using raw
        # genesis_hash string rather than the base class "greenlang-{name}-genesis"
        # pattern) to preserve backward-compatible chain hashes.
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        # Initialize base class internals without re-computing genesis
        self.agent_name: str = "climate-hazard"
        self._GENESIS_HASH: str = self._genesis_hash
        self._chain_store: Dict[str, List[ProvenanceEntry]] = {}
        self._global_chain: List[ProvenanceEntry] = []
        self._last_chain_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "ProvenanceTracker initialized with genesis hash prefix=%s",
            self._genesis_hash[:16],
        )

    # -- Public API --------------------------------------------------------

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry.

        Args:
            entity_type: Type of entity being tracked.
            action: Action performed.
            entity_id: Unique entity identifier.
            data: Optional serializable payload.
            metadata: Optional additional metadata.

        Returns:
            The newly created ProvenanceEntry.

        Raises:
            ValueError: If entity_type, action, or entity_id are empty.
        """
        if not entity_type:
            raise ValueError("entity_type must not be empty")
        if not action:
            raise ValueError("action must not be empty")
        if not entity_id:
            raise ValueError("entity_id must not be empty")

        timestamp = utcnow().isoformat()
        data_hash = self._hash_data(data)
        store_key = f"{entity_type}:{entity_id}"
        entry_metadata: Dict[str, Any] = {"data_hash": data_hash}
        if metadata:
            entry_metadata.update(metadata)

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_ch_chain_hash(
                parent_hash, data_hash, action, timestamp,
            )
            entry = ProvenanceEntry(
                entity_type=entity_type, entity_id=entity_id,
                action=action, hash_value=chain_hash,
                parent_hash=parent_hash, timestamp=timestamp,
                metadata=entry_metadata,
            )
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash_prefix=%s",
            entity_type, entity_id[:16], action, chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the global chain integrity.

        Returns:
            True if the chain is intact, False otherwise.
        """
        with self._lock:
            chain = list(self._global_chain)
        if not chain:
            return True
        required_fields = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp",
        }
        for i, entry in enumerate(chain):
            for fn in required_fields:
                if not getattr(entry, fn, None):
                    return False
            if i == 0 and entry.parent_hash != self._genesis_hash:
                return False
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                return False
        return True

    def get_chain(self) -> List[ProvenanceEntry]:
        """Return the full global provenance chain."""
        with self._lock:
            return list(self._global_chain)

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Return entries filtered by entity_type and/or action.

        Args:
            entity_type: Optional entity type filter.
            action: Optional action filter.
            limit: Optional max number of entries to return.

        Returns:
            Filtered list of ProvenanceEntry objects.
        """
        with self._lock:
            entries = list(self._global_chain)
        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]
        if action:
            entries = [e for e in entries if e.action == action]
        if limit and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]
        return entries

    def get_entries_by_entity(
        self, entity_type: str, entity_id: str,
    ) -> List[ProvenanceEntry]:
        """Return entries for a specific entity_type:entity_id.

        Args:
            entity_type: Entity type.
            entity_id: Entity identifier.

        Returns:
            List of ProvenanceEntry objects.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._lock:
            return list(self._chain_store.get(store_key, []))

    def get_entry_by_hash(self, hash_value: str) -> Optional[ProvenanceEntry]:
        """Look up a single entry by its chain hash.

        Args:
            hash_value: SHA-256 chain hash to search for.

        Returns:
            The matching ProvenanceEntry, or None.
        """
        if not hash_value:
            return None
        with self._lock:
            for entry in self._global_chain:
                if entry.hash_value == hash_value:
                    return entry
        return None

    def get_entity_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Return all entries for a given entity_id across all types.

        Args:
            entity_id: Entity identifier.

        Returns:
            List of ProvenanceEntry objects.
        """
        with self._lock:
            return [e for e in self._global_chain if e.entity_id == entity_id]

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the chain as a list of dictionaries."""
        with self._lock:
            return [entry.to_dict() for entry in self._global_chain]

    def export_json(self) -> str:
        """Export all provenance records as JSON string."""
        return json.dumps(self.export_chain(), indent=2, default=str)

    def reset(self) -> None:
        """Clear all provenance state and reset to genesis."""
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self._genesis_hash
        logger.info("ProvenanceTracker reset to genesis state")

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

    @property
    def genesis_hash(self) -> str:
        """Return the genesis hash."""
        return self._genesis_hash

    @property
    def last_chain_hash(self) -> str:
        """Return the most recent chain hash."""
        with self._lock:
            return self._last_chain_hash

    def __len__(self) -> int:
        return self.entry_count

    def __repr__(self) -> str:
        return (
            f"ProvenanceTracker(entries={self.entry_count}, "
            f"entities={self.entity_count}, "
            f"genesis_prefix={self._genesis_hash[:12]})"
        )

    # -- Internal ----------------------------------------------------------

    def _hash_data(self, data: Optional[Any]) -> str:
        """Compute SHA-256 hash for arbitrary data."""
        if data is None:
            serialized = "null"
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_ch_chain_hash(
        parent_hash: str, data_hash: str,
        action: str, timestamp: str,
    ) -> str:
        """Compute the next chain hash for climate hazard entries.

        Args:
            parent_hash: Previous chain hash.
            data_hash: SHA-256 hash of the data payload.
            action: Action label.
            timestamp: ISO-formatted timestamp.

        Returns:
            New SHA-256 chain hash.
        """
        combined = json.dumps({
            "action": action, "data_hash": data_hash,
            "parent_hash": parent_hash, "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def build_hash(self, data: Any) -> str:
        """Build a SHA-256 hash for arbitrary data."""
        return self._hash_data(data)


# ---------------------------------------------------------------------------
# Thread-safe singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Return the process-wide singleton ProvenanceTracker."""
    global _singleton_tracker
    if _singleton_tracker is None:
        with _singleton_lock:
            if _singleton_tracker is None:
                _singleton_tracker = ProvenanceTracker()
    return _singleton_tracker


def set_provenance_tracker(tracker: ProvenanceTracker) -> None:
    """Replace the singleton with a custom tracker.

    Args:
        tracker: The ProvenanceTracker instance to install.

    Raises:
        TypeError: If tracker is not a ProvenanceTracker.
    """
    if not isinstance(tracker, ProvenanceTracker):
        raise TypeError(
            f"tracker must be a ProvenanceTracker instance, got {type(tracker)}"
        )
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = tracker


def reset_provenance_tracker() -> None:
    """Destroy the current singleton."""
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None


__all__ = [
    "ProvenanceEntry",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "ProvenanceTracker",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
]
