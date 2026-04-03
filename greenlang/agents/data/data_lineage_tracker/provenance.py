# -*- coding: utf-8 -*-
"""
Provenance Tracking for Data Lineage Tracker - AGENT-DATA-018

Thin shim that delegates core chain-hashing to the shared
``greenlang.data_commons.provenance`` base class while preserving the
extended data-lineage-tracker API (``ProvenanceEntry`` dataclass,
``record()`` returning ``ProvenanceEntry``, ``verify_chain()`` returning
``bool``, singleton helpers, and ``reset``).

Example:
    >>> from greenlang.agents.data.data_lineage_tracker.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("lineage_asset", "asset_001", "asset_registered")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
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
    """A single tamper-evident provenance record for a data lineage event."""

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
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker(_BaseProvenanceTracker):
    """Data lineage provenance tracker with extended dataclass API.

    Inherits genesis-hash derivation from the shared base class while
    maintaining the ``ProvenanceEntry`` chain, thread-safe state, and
    the full data-lineage-tracker-specific public API.
    """

    def __init__(
        self, genesis_hash: str = "greenlang-data-lineage-genesis",
    ) -> None:
        """Initialize with data-lineage genesis hash.

        Args:
            genesis_hash: String used to compute the immutable genesis hash.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self.agent_name: str = "data-lineage-tracker"
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
        entity_id: str,
        action: str,
        metadata: Optional[Any] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry for a data lineage operation.

        Args:
            entity_type: Type of entity (lineage_asset, transformation,
                lineage_edge, impact_analysis, validation, report,
                change_event, quality_score).
            entity_id: Unique entity identifier.
            action: Action performed (asset_registered,
                transformation_captured, edge_created, impact_analyzed,
                validation_completed, report_generated, change_detected,
                quality_scored).
            metadata: Optional serializable payload.

        Returns:
            The newly created ProvenanceEntry.

        Raises:
            ValueError: If entity_type, entity_id, or action are empty.
        """
        if not entity_type:
            raise ValueError("entity_type must not be empty")
        if not entity_id:
            raise ValueError("entity_id must not be empty")
        if not action:
            raise ValueError("action must not be empty")

        timestamp = utcnow().isoformat()
        data_hash = self._hash_data(metadata)
        store_key = f"{entity_type}:{entity_id}"

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_dlt_chain_hash(
                parent_hash=parent_hash,
                data_hash=data_hash,
                action=action,
                timestamp=timestamp,
            )
            entry = ProvenanceEntry(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                hash_value=chain_hash,
                parent_hash=parent_hash,
                timestamp=timestamp,
                metadata={"data_hash": data_hash},
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
        """Verify the integrity of the entire global provenance chain.

        Returns:
            True if the chain is intact, False otherwise.
        """
        with self._lock:
            chain = list(self._global_chain)
        if not chain:
            logger.debug("verify_chain: chain is empty - trivially valid")
            return True
        required_fields = {
            "entity_type", "entity_id", "action",
            "hash_value", "parent_hash", "timestamp",
        }
        for i, entry in enumerate(chain):
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    logger.warning(
                        "verify_chain: entry[%d] missing or empty field '%s'",
                        i, field_name,
                    )
                    return False
            if i == 0 and entry.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: entry[0] parent_hash does not match genesis hash"
                )
                return False
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between entry[%d] and entry[%d]",
                    i - 1, i,
                )
                return False
        logger.debug("verify_chain: %d entries verified successfully", len(chain))
        return True

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries filtered by entity_type and/or entity_id.

        Args:
            entity_type: Optional entity type filter.
            entity_id: Optional entity ID filter.

        Returns:
            List of matching ProvenanceEntry objects.
        """
        with self._lock:
            if entity_type and entity_id:
                store_key = f"{entity_type}:{entity_id}"
                return list(self._chain_store.get(store_key, []))
            if entity_type:
                return [
                    entry for entry in self._global_chain
                    if entry.entity_type == entity_type
                ]
            return list(self._global_chain)

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

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Return all entries for a given entity_id across all types.

        Args:
            entity_id: Entity identifier.

        Returns:
            List of ProvenanceEntry objects.
        """
        with self._lock:
            return [
                entry for entry in self._global_chain
                if entry.entity_id == entity_id
            ]

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
        """Return the number of unique entity keys tracked."""
        with self._lock:
            return len(self._chain_store)

    def __len__(self) -> int:
        return self.entry_count

    # -- Internal ----------------------------------------------------------

    def _hash_data(self, data: Optional[Any]) -> str:
        """Compute SHA-256 hash for arbitrary data."""
        if data is None:
            serialized = "null"
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_dlt_chain_hash(
        parent_hash: str, data_hash: str,
        action: str, timestamp: str,
    ) -> str:
        """Compute the next chain hash for data lineage tracker entries.

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
    """Return the process-wide singleton ProvenanceTracker.

    Returns:
        The singleton ProvenanceTracker instance.
    """
    global _singleton_tracker
    if _singleton_tracker is None:
        with _singleton_lock:
            if _singleton_tracker is None:
                _singleton_tracker = ProvenanceTracker()
                logger.info("Data lineage singleton ProvenanceTracker created")
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
    logger.info("Data lineage ProvenanceTracker singleton replaced")


def reset_provenance_tracker() -> None:
    """Destroy the current singleton."""
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None
    logger.info("Data lineage ProvenanceTracker singleton reset to None")


__all__ = [
    "ProvenanceEntry",
    "ProvenanceTracker",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
]
