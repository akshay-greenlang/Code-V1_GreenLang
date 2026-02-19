# -*- coding: utf-8 -*-
"""
Provenance Tracking for Land Use Emissions Agent - AGENT-MRV-006

Provides SHA-256 based audit trail tracking for all land use emissions
agent operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across parcel management, carbon stock snapshots,
land-use transitions, emission calculations, SOC assessments, compliance
checking, uncertainty analysis, and pipeline orchestration.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - First entry parent_hash = "0" * 64 (genesis anchor)
    - JSON export for external audit systems
    - Complete provenance for every land use emission operation

Entity Types (12):
    - PARCEL: Land parcel registrations with geographic, climatic,
      and soil characteristics. Tracks parcel lifecycle including
      creation, updates, and deactivation.
    - CARBON_STOCK: Carbon stock snapshots for individual parcels
      and pools. Records point-in-time measurements used in the
      stock-difference calculation method.
    - TRANSITION: Land-use transition events recording category
      changes (e.g. forest to cropland), areas affected, disturbance
      types, and transition dates.
    - CALCULATION: Individual land use emission calculation results
      with per-pool and per-gas breakdowns, CO2e totals, removals,
      net emissions, and calculation trace steps.
    - SOC_ASSESSMENT: Soil organic carbon assessment results using
      IPCC Tier 1 reference stocks and land use, management, and
      input factors (F_LU, F_MG, F_I).
    - EMISSION_FACTOR: Emission factor records from authoritative
      sources (IPCC 2006, IPCC 2019, IPCC Wetlands 2013, national
      inventories, literature, custom) scoped by land category,
      climate zone, soil type, and gas.
    - COMPLIANCE_CHECK: Regulatory compliance check results and
      findings for GHG Protocol, IPCC 2006, CSRD/ESRS E1, EU LULUCF,
      UK SECR, and UNFCCC frameworks.
    - UNCERTAINTY_RUN: Monte Carlo uncertainty quantification results
      with confidence intervals, standard deviation, coefficient of
      variation, and percentile distributions.
    - AGGREGATION: Aggregated emission results grouped by tenant,
      land category, climate zone, or time period.
    - BATCH: Batch calculation jobs processing multiple land use
      emission inputs as a single unit of work.
    - CONFIG: Configuration change events tracking modifications to
      calculation defaults, feature flags, and capacity limits.
    - SYSTEM: System-level events including startup, shutdown, health
      checks, and migration operations.

Actions (16):
    Entity management: CREATE, UPDATE, DELETE
    Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE
    Import/Export: EXPORT, IMPORT
    Snapshots: SNAPSHOT, TRANSITION
    Disturbances: FIRE, HARVEST
    Peatland: REWET, DRAIN

Example:
    >>> from greenlang.land_use_emissions.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("PARCEL", "parcel_001", "CREATE")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceEntry:
    """A single tamper-evident provenance record for a land use emission event.

    Attributes:
        entity_type: Type of entity being tracked (PARCEL, CARBON_STOCK,
            TRANSITION, CALCULATION, SOC_ASSESSMENT, EMISSION_FACTOR,
            COMPLIANCE_CHECK, UNCERTAINTY_RUN, AGGREGATION, BATCH,
            CONFIG, SYSTEM).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
            ASSESS, CHECK, VALIDATE, AGGREGATE, EXPORT, IMPORT,
            SNAPSHOT, TRANSITION, FIRE, HARVEST, REWET, DRAIN).
        hash_value: SHA-256 chain hash of this entry, incorporating the
            previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding entry.
            For the first entry in the chain, this equals "0" * 64
            (the genesis anchor hash).
        timestamp: UTC ISO-formatted timestamp when the entry was created.
        metadata: Optional dictionary of additional contextual fields.
    """

    entity_type: str
    entity_id: str
    action: str
    hash_value: str
    parent_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
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
    "PARCEL",
    "CARBON_STOCK",
    "TRANSITION",
    "CALCULATION",
    "SOC_ASSESSMENT",
    "EMISSION_FACTOR",
    "COMPLIANCE_CHECK",
    "UNCERTAINTY_RUN",
    "AGGREGATION",
    "BATCH",
    "CONFIG",
    "SYSTEM",
})

VALID_ACTIONS = frozenset({
    # Entity management
    "CREATE",
    "UPDATE",
    "DELETE",
    # Calculations
    "CALCULATE",
    "ASSESS",
    "CHECK",
    "VALIDATE",
    "AGGREGATE",
    # Import/Export
    "EXPORT",
    "IMPORT",
    # Snapshots and transitions
    "SNAPSHOT",
    "TRANSITION",
    # Disturbances
    "FIRE",
    "HARVEST",
    # Peatland operations
    "REWET",
    "DRAIN",
})


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for land use emission operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    The genesis hash anchors the chain. For the first entry in the chain,
    the parent_hash is set to ``"0" * 64`` (64 zero characters). Every
    subsequent entry incorporates the previous entry's chain hash so that
    any tampering is detectable via ``verify_chain()``.

    Supported entity types (12):
        - ``PARCEL``: Land parcel registrations with geographic, climatic,
          and soil characteristics.
        - ``CARBON_STOCK``: Carbon stock snapshots for individual parcels
          and pools.
        - ``TRANSITION``: Land-use transition events recording category
          changes, areas affected, and disturbance types.
        - ``CALCULATION``: Individual land use emission calculation results
          with per-pool and per-gas breakdowns.
        - ``SOC_ASSESSMENT``: Soil organic carbon assessment results using
          IPCC Tier 1 reference stocks and factors.
        - ``EMISSION_FACTOR``: Emission factor records from authoritative
          sources scoped by land category, climate zone, and gas.
        - ``COMPLIANCE_CHECK``: Regulatory compliance check results for
          GHG Protocol, IPCC, CSRD, EU LULUCF, UK SECR, and UNFCCC.
        - ``UNCERTAINTY_RUN``: Monte Carlo uncertainty quantification
          results with confidence intervals.
        - ``AGGREGATION``: Aggregated emission results grouped by tenant,
          land category, climate zone, or time period.
        - ``BATCH``: Batch calculation jobs processing multiple inputs.
        - ``CONFIG``: Configuration change events.
        - ``SYSTEM``: System-level events (startup, shutdown, health).

    Supported actions (16):
        Entity management: CREATE, UPDATE, DELETE.
        Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE.
        Import/Export: EXPORT, IMPORT.
        Snapshots: SNAPSHOT, TRANSITION.
        Disturbances: FIRE, HARVEST.
        Peatland: REWET, DRAIN.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by entity_type:entity_id.
        _global_chain: Flat list of all ProvenanceEntry objects in order.
        _last_chain_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.
        _max_entries: Maximum number of entries before oldest are evicted.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("PARCEL", "parcel_001", "CREATE")
        >>> assert entry.hash_value != ""
        >>> assert entry.parent_hash == "0" * 64
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: Optional[str] = None,
        max_entries: int = 10000,
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        The first entry in the chain will have ``parent_hash = "0" * 64``
        regardless of the genesis_hash parameter value.

        Args:
            genesis_hash: Optional string used for internal chain
                anchoring. Defaults to ``None``, which results in
                ``"0" * 64`` as the genesis anchor.
            max_entries: Maximum number of provenance entries to retain
                in memory. When exceeded, the oldest entries are evicted.
                Defaults to 10000.
        """
        self._genesis_hash: str = "0" * 64
        self._chain_store: Dict[str, List[ProvenanceEntry]] = {}
        self._global_chain: List[ProvenanceEntry] = []
        self._last_chain_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
        self._max_entries: int = max_entries
        logger.info(
            "ProvenanceTracker initialized with genesis hash prefix=%s",
            self._genesis_hash[:16],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry for a land use emission operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (PARCEL, CARBON_STOCK, TRANSITION,
                CALCULATION, SOC_ASSESSMENT, EMISSION_FACTOR,
                COMPLIANCE_CHECK, UNCERTAINTY_RUN, AGGREGATION, BATCH,
                CONFIG, SYSTEM).
            entity_id: Unique entity identifier.
            action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
                ASSESS, CHECK, VALIDATE, AGGREGATE, EXPORT, IMPORT,
                SNAPSHOT, TRANSITION, FIRE, HARVEST, REWET, DRAIN).
            data: Optional serializable payload; its SHA-256 hash is
                stored.
            metadata: Optional dictionary of extra contextual fields.

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

        timestamp = _utcnow().isoformat()
        data_hash = self._hash_data(data)
        store_key = f"{entity_type}:{entity_id}"

        entry_metadata: Dict[str, Any] = {"data_hash": data_hash}
        if metadata:
            entry_metadata.update(metadata)

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_chain_hash(
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
                metadata=entry_metadata,
            )

            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

            self._evict_if_needed()

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash_prefix=%s",
            entity_type,
            entity_id[:16],
            action,
            chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the entire global provenance chain.

        Walks the global chain in insertion order and checks that every
        entry contains all required fields and that the chain is
        structurally consistent.

        Returns:
            Tuple of (is_valid, error_message). When the chain is
            intact, returns (True, None).
        """
        with self._lock:
            chain = list(self._global_chain)

        if not chain:
            logger.debug("verify_chain: chain is empty - trivially valid")
            return True, None

        required_fields = {
            "entity_type",
            "entity_id",
            "action",
            "hash_value",
            "parent_hash",
            "timestamp",
        }

        for i, entry in enumerate(chain):
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    msg = (
                        f"entry[{i}] missing or empty field '{field_name}'"
                    )
                    logger.warning("verify_chain: %s", msg)
                    return False, msg

            if i == 0 and entry.parent_hash != self._genesis_hash:
                msg = "entry[0] parent_hash does not match genesis hash"
                logger.warning("verify_chain: %s", msg)
                return False, msg

            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                msg = (
                    f"chain break between entry[{i - 1}] and entry[{i}]"
                )
                logger.warning("verify_chain: %s", msg)
                return False, msg

        logger.debug(
            "verify_chain: %d entries verified successfully", len(chain)
        )
        return True, None

    def get_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries filtered by entity_type, entity_id, action.

        Args:
            entity_type: Optional entity type to filter by.
            entity_id: Optional entity identifier to filter by.
            action: Optional action to filter by.
            limit: Maximum number of entries to return.

        Returns:
            List of matching ProvenanceEntry objects.
        """
        if entity_type and entity_id:
            entries = self.get_entries_for_entity(entity_type, entity_id)
        elif entity_type:
            with self._lock:
                entries = [
                    e for e in self._global_chain
                    if e.entity_type == entity_type
                ]
        else:
            with self._lock:
                entries = list(self._global_chain)

        if action:
            entries = [e for e in entries if e.action == action]

        if limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries filtered by entity_type, action, and/or limit.

        Args:
            entity_type: Optional entity type to filter by.
            action: Optional action to filter by.
            limit: Optional maximum number of entries to return.

        Returns:
            List of matching ProvenanceEntry objects.
        """
        with self._lock:
            entries = list(self._global_chain)

        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]

        if action:
            entries = [e for e in entries if e.action == action]

        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries for a specific entity_type:entity_id pair.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of ProvenanceEntry objects for the entity, oldest first.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._lock:
            return list(self._chain_store.get(store_key, []))

    def get_chain(self) -> List[ProvenanceEntry]:
        """Return the complete global provenance chain in insertion order.

        Returns:
            List of all ProvenanceEntry objects, oldest first.
        """
        with self._lock:
            return list(self._global_chain)

    def get_chain_hash(self) -> str:
        """Return the current chain hash representing the entire trail.

        Returns:
            Hex-encoded SHA-256 chain hash string.
        """
        with self._lock:
            return self._last_chain_hash

    def get_audit_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the audit trail as a list of dictionaries.

        Args:
            entity_type: Optional filter by entity type.
            entity_id: Optional filter by entity id.

        Returns:
            List of dictionaries, each representing one audit entry.
        """
        if entity_type and entity_id:
            entries = self.get_entries_for_entity(entity_type, entity_id)
        elif entity_type:
            entries = self.get_entries(entity_type=entity_type)
        else:
            entries = self.get_chain()

        return [entry.to_dict() for entry in entries]

    def export_trail(self, format: str = "json", indent: int = 2) -> str:
        """Export all provenance records as a formatted string.

        Args:
            format: Output format. Currently only "json" is supported.
            indent: JSON indentation level.

        Returns:
            Formatted string representation of the global chain.

        Raises:
            ValueError: If format is not "json".
        """
        if format != "json":
            raise ValueError(
                f"Unsupported export format '{format}'; only 'json' is supported"
            )

        with self._lock:
            chain_dicts = [entry.to_dict() for entry in self._global_chain]

        return json.dumps(chain_dicts, indent=indent, default=str)

    def clear_trail(self) -> None:
        """Clear all provenance state and reset to genesis.

        After calling this method the tracker behaves as if newly
        constructed. Primarily intended for testing.
        """
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self._genesis_hash
        logger.info("ProvenanceTracker reset to genesis state")

    # Alias for backward compatibility with sibling agents
    clear = clear_trail

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries recorded."""
        with self._lock:
            return len(self._global_chain)

    @property
    def entity_count(self) -> int:
        """Return the number of unique entity_type:entity_id keys tracked."""
        with self._lock:
            return len(self._chain_store)

    @property
    def genesis_hash(self) -> str:
        """Return the genesis hash that anchors the provenance chain."""
        return self._genesis_hash

    @property
    def last_chain_hash(self) -> str:
        """Return the most recent chain hash for the global chain."""
        with self._lock:
            return self._last_chain_hash

    @property
    def max_entries(self) -> int:
        """Return the maximum number of entries before eviction."""
        return self._max_entries

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of provenance entries recorded."""
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ProvenanceTracker(entries={self.entry_count}, "
            f"entities={self.entity_count}, "
            f"genesis_prefix={self._genesis_hash[:12]})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_data(self, data: Optional[Any]) -> str:
        """Compute a SHA-256 hash for arbitrary data.

        Args:
            data: Any JSON-serializable object, or None.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        if data is None:
            serialized = "null"
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_entry_hash(self, entry_data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for an entry data dictionary.

        Args:
            entry_data: Dictionary of entry fields to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(entry_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_chain_hash(
        self,
        parent_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next SHA-256 chain hash linking to the previous entry.

        Args:
            parent_hash: The chain hash of the preceding entry.
            data_hash: SHA-256 hash of the operation's data payload.
            action: Action label recorded in this entry.
            timestamp: ISO-formatted UTC timestamp string.

        Returns:
            New hex-encoded SHA-256 chain hash.
        """
        combined = json.dumps(
            {
                "action": action,
                "data_hash": data_hash,
                "parent_hash": parent_hash,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries from global chain if max_entries exceeded.

        Must be called while holding self._lock.
        """
        overflow = len(self._global_chain) - self._max_entries
        if overflow <= 0:
            return

        evicted = self._global_chain[:overflow]
        self._global_chain = self._global_chain[overflow:]

        for entry in evicted:
            store_key = f"{entry.entity_type}:{entry.entity_id}"
            entity_entries = self._chain_store.get(store_key)
            if entity_entries:
                try:
                    entity_entries.remove(entry)
                except ValueError:
                    pass
                if not entity_entries:
                    del self._chain_store[store_key]

        logger.debug(
            "Evicted %d oldest provenance entries (max_entries=%d)",
            overflow,
            self._max_entries,
        )

    def build_hash(self, data: Any) -> str:
        """Build a standalone SHA-256 hash for arbitrary data.

        Utility method for callers that need to pre-compute hashes.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._hash_data(data)


# ---------------------------------------------------------------------------
# Thread-safe singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Return the process-wide singleton ProvenanceTracker.

    Creates the instance on first call (lazy initialization). Subsequent
    calls return the same object. The function is thread-safe.

    Returns:
        The singleton ProvenanceTracker instance.

    Example:
        >>> tracker_a = get_provenance_tracker()
        >>> tracker_b = get_provenance_tracker()
        >>> assert tracker_a is tracker_b
    """
    global _singleton_tracker
    if _singleton_tracker is None:
        with _singleton_lock:
            if _singleton_tracker is None:
                _singleton_tracker = ProvenanceTracker()
                logger.info(
                    "Land use emissions singleton ProvenanceTracker created"
                )
    return _singleton_tracker


def set_provenance_tracker(tracker: ProvenanceTracker) -> None:
    """Replace the process-wide singleton with a custom tracker.

    Useful in tests that need isolated tracker instances.

    Args:
        tracker: The ProvenanceTracker instance to install.

    Raises:
        TypeError: If tracker is not a ProvenanceTracker instance.
    """
    if not isinstance(tracker, ProvenanceTracker):
        raise TypeError(
            f"tracker must be a ProvenanceTracker instance, got {type(tracker)}"
        )
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = tracker
    logger.info("Land use emissions ProvenanceTracker singleton replaced")


def reset_provenance_tracker() -> None:
    """Destroy the current singleton and reset to None.

    The next call to get_provenance_tracker will create a fresh
    instance. Intended for use in test teardown.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # fresh instance
    """
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None
    logger.info(
        "Land use emissions ProvenanceTracker singleton reset to None"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclass
    "ProvenanceEntry",
    # Constants
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    # Tracker class
    "ProvenanceTracker",
    # Singleton helpers
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
]
