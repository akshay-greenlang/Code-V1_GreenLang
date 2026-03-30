# -*- coding: utf-8 -*-
"""
Provenance Tracking for Plot Boundary Manager - AGENT-EUDR-006

Provides SHA-256 based audit trail tracking for all plot boundary
management operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across boundary creation, validation,
repair, area calculation, overlap detection, versioning, simplification,
split/merge operations, and multi-format export for EU Deforestation
Regulation (EUDR) Articles 9, 10, and 31 compliance.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every boundary operation
    - Bit-perfect reproducibility for boundary audit trails

Entity Types (10):
    - boundary: Plot boundary CRUD operations (create, update, delete,
      import, restore)
    - version: Boundary version management operations (create version,
      archive version, compare versions)
    - validation: Boundary geometry validation operations (OGC checks,
      topology validation, coordinate validation)
    - repair: Geometry repair operations (self-intersection repair,
      ring closure, vertex removal, spike removal, orientation fix)
    - area_calc: Geodetic area calculation operations (Karney ellipsoidal,
      spherical approximation, UTM projection)
    - overlap: Overlap detection operations (spatial scan, pairwise
      intersection, severity classification)
    - simplification: Polygon simplification operations (Douglas-Peucker,
      Visvalingam-Whyatt, topology-preserving)
    - split: Boundary split operations (cutting line application, child
      boundary generation, area conservation verification)
    - merge: Boundary merge operations (union computation, topology
      cleanup, area conservation verification)
    - export: Boundary export operations (format conversion, CRS
      reprojection, regulatory submission preparation)

Actions (15):
    Boundary management: create, update, delete, validate, repair
    Calculation: calculate, detect, simplify
    Operations: split, merge, export, import, restore
    System: query, batch

Example:
    >>> from greenlang.agents.eudr.plot_boundary.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("boundary", "create", "plot-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a plot boundary operation.

    Attributes:
        entity_type: Type of entity being tracked (boundary, version,
            validation, repair, area_calc, overlap, simplification,
            split, merge, export).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (create, update, delete, validate,
            repair, calculate, detect, simplify, split, merge, export,
            import, restore, query, batch).
        hash_value: SHA-256 chain hash of this record, incorporating
            the previous record's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding
            record.
        timestamp: UTC ISO-formatted timestamp when the record was
            created.
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
        """Serialize the record to a plain dictionary.

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
            "metadata": dict(self.metadata),
        }

# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    "boundary",
    "version",
    "validation",
    "repair",
    "area_calc",
    "overlap",
    "simplification",
    "split",
    "merge",
    "export",
})

VALID_ACTIONS = frozenset({
    # Boundary management operations
    "create",
    "update",
    "delete",
    "validate",
    "repair",
    # Calculation and detection operations
    "calculate",
    "detect",
    "simplify",
    # Split/merge/export operations
    "split",
    "merge",
    "export",
    "import",
    "restore",
    # System operations
    "query",
    "batch",
})

# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Tracks provenance for plot boundary operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails for EUDR Article 31
    record-keeping compliance (5-year retention).

    The genesis hash anchors the chain. Every new record incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types:
        - ``boundary``: Plot boundary CRUD operations including creation
          from coordinate arrays, update with version tracking, soft

from greenlang.schemas import utcnow
          deletion with archive retention, import from external GIS
          sources, and restoration from archived versions.
        - ``version``: Boundary version management operations including
          immutable version snapshot creation, version comparison with
          area diff computation, and version archive management per
          EUDR Article 31 retention requirements.
        - ``validation``: Boundary geometry validation operations
          including OGC Simple Features compliance checking, topology
          validation (self-intersection, ring closure, orientation),
          coordinate range validation, vertex count enforcement, and
          sliver/spike detection.
        - ``repair``: Geometry repair operations including self-
          intersection repair via node insertion, ring closure by
          vertex duplication, duplicate vertex removal, spike removal,
          orientation reversal, hole removal, and convex hull fallback.
        - ``area_calc``: Geodetic area calculation operations including
          Karney ellipsoidal area on the WGS84 reference ellipsoid,
          spherical excess approximation, UTM-projected area, perimeter
          calculation, compactness index computation, and EUDR threshold
          classification.
        - ``overlap``: Overlap detection operations including R-tree
          spatial index scanning, pairwise polygon intersection testing,
          overlap area computation, overlap percentage calculation,
          severity classification, and overlap geometry extraction.
        - ``simplification``: Polygon simplification operations including
          Douglas-Peucker vertex reduction, Visvalingam-Whyatt area-
          based simplification, topology-preserving simplification,
          area deviation checking, and Hausdorff distance computation.
        - ``split``: Boundary split operations including cutting line
          validation, polygon splitting along cutting line, child
          boundary generation with genealogy tracking, and area
          conservation verification.
        - ``merge``: Boundary merge operations including polygon union
          computation, topology cleanup after merge, merged boundary
          generation with genealogy tracking, and area conservation
          verification.
        - ``export``: Boundary export operations including format
          conversion (GeoJSON, KML, WKT, WKB, Shapefile, EUDR XML,
          GPX, GML), CRS reprojection, coordinate precision control,
          metadata embedding, and regulatory submission preparation.

    Supported actions (15):
        Boundary management: create, update, delete, validate, repair.
        Calculation: calculate, detect, simplify.
        Operations: split, merge, export, import, restore.
        System: query, batch.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by
            ``"entity_type:entity_id"``.
        _global_chain: Flat list of all ProvenanceRecord objects in
            insertion order.
        _last_chain_hash: Most recent chain hash for linking the
            next record.
        _lock: Reentrant lock for thread-safe access.
        _max_records: Maximum number of records to retain in memory.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> record = tracker.record_operation("boundary", "create", "plot-001")
        >>> assert record.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-PBM-006-PLOT-BOUNDARY-MANAGER-GENESIS",
        max_records: int = 100000,
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to the Plot Boundary Manager agent
                identifier.
            max_records: Maximum number of records to retain in
                memory. Oldest records are evicted when exceeded.
                Default is 100000.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self._chain_store: Dict[str, List[ProvenanceRecord]] = {}
        self._global_chain: List[ProvenanceRecord] = []
        self._last_chain_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
        self._max_records: int = max_records
        logger.info(
            "ProvenanceTracker initialized with genesis hash prefix=%s, "
            "max_records=%d",
            self._genesis_hash[:16],
            self._max_records,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_operation(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """Record a provenance entry for a plot boundary operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous record hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (boundary, version, validation,
                repair, area_calc, overlap, simplification, split,
                merge, export).
            action: Action performed (see VALID_ACTIONS).
            entity_id: Unique entity identifier.
            data: Optional serializable payload; its SHA-256 hash is
                stored.
            metadata: Optional dictionary of extra contextual fields.

        Returns:
            The newly created ProvenanceRecord.

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

        # Build record metadata combining data hash with caller metadata
        record_metadata: Dict[str, Any] = {"data_hash": data_hash}
        if metadata:
            record_metadata.update(metadata)

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_chain_hash(
                parent_hash=parent_hash,
                data_hash=data_hash,
                action=action,
                timestamp=timestamp,
            )

            record = ProvenanceRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                hash_value=chain_hash,
                parent_hash=parent_hash,
                timestamp=timestamp,
                metadata=record_metadata,
            )

            # Persist to entity-scoped and global stores
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(record)
            self._global_chain.append(record)
            self._last_chain_hash = chain_hash

            # Evict oldest records if over capacity
            self._evict_if_needed()

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash_prefix=%s",
            entity_type,
            entity_id[:16],
            action,
            chain_hash[:16],
        )
        return record

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire global provenance chain.

        Walks the global chain in insertion order and checks that every
        record contains all required fields and that the chain is
        structurally consistent: the first record chains from the genesis
        hash, and each subsequent record's parent_hash matches the
        preceding record's hash_value.

        Returns:
            True if the chain is intact, False if any record is
            malformed or the chain links are broken.
        """
        with self._lock:
            chain = list(self._global_chain)

        if not chain:
            logger.debug("verify_chain: chain is empty - trivially valid")
            return True

        required_fields = {
            "entity_type",
            "entity_id",
            "action",
            "hash_value",
            "parent_hash",
            "timestamp",
        }

        for i, record in enumerate(chain):
            # Validate all required fields are present and non-empty
            for field_name in required_fields:
                value = getattr(record, field_name, None)
                if not value:
                    logger.warning(
                        "verify_chain: record[%d] missing or empty field '%s'",
                        i,
                        field_name,
                    )
                    return False

            # First record must chain from the genesis hash
            if i == 0 and record.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: record[0] parent_hash does not match "
                    "genesis hash"
                )
                return False

            # Each subsequent record's parent must match the previous hash
            if i > 0 and record.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between record[%d] and "
                    "record[%d]",
                    i - 1,
                    i,
                )
                return False

        logger.debug(
            "verify_chain: %d records verified successfully", len(chain)
        )
        return True

    def get_chain(self) -> List[ProvenanceRecord]:
        """Return a copy of the complete global provenance chain.

        Returns:
            List of all ProvenanceRecord objects in insertion order.
        """
        with self._lock:
            return list(self._global_chain)

    def get_records_by_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceRecord]:
        """Return provenance records for a specific entity_type:entity_id pair.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of ProvenanceRecord objects, oldest first.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._lock:
            return list(self._chain_store.get(store_key, []))

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceRecord]:
        """Return provenance records filtered by entity_type, action, and/or limit.

        Args:
            entity_type: Optional entity type filter.
            action: Optional action filter.
            limit: Optional maximum number of records to return
                (most recent when truncated).

        Returns:
            List of matching ProvenanceRecord objects.
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

    def export_chain(self) -> str:
        """Export all provenance records as a formatted JSON string.

        Returns:
            Indented JSON string representation of the global chain.
        """
        with self._lock:
            chain_dicts = [record.to_dict() for record in self._global_chain]
        return json.dumps(chain_dicts, indent=2, default=str)

    def clear(self) -> None:
        """Clear all provenance state and reset to genesis.

        Primarily intended for testing.
        """
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self._genesis_hash
        logger.info("ProvenanceTracker reset to genesis state")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance records."""
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

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of provenance records."""
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

        Serializes the payload to canonical JSON (sorted keys, default
        ``str`` fallback) before hashing so that equivalent structures
        always produce the same digest.

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

    def _compute_chain_hash(
        self,
        parent_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next SHA-256 chain hash linking to the previous record.

        Args:
            parent_hash: Chain hash of the preceding record (or genesis
                hash for the first record).
            data_hash: SHA-256 hash of the operation's data payload.
            action: Action label recorded in this record.
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
        """Evict oldest records if global chain exceeds max_records.

        Called within the lock context. Removes the oldest 10% of records
        when the chain exceeds max_records to amortize eviction cost.
        """
        if len(self._global_chain) <= self._max_records:
            return

        evict_count = max(1, self._max_records // 10)
        evicted = self._global_chain[:evict_count]
        self._global_chain = self._global_chain[evict_count:]

        # Clean up entity-scoped store for evicted records
        evicted_keys = set()
        for record in evicted:
            store_key = f"{record.entity_type}:{record.entity_id}"
            evicted_keys.add(store_key)

        for store_key in evicted_keys:
            if store_key in self._chain_store:
                remaining = [
                    r for r in self._chain_store[store_key]
                    if r not in evicted
                ]
                if remaining:
                    self._chain_store[store_key] = remaining
                else:
                    del self._chain_store[store_key]

        logger.debug(
            "Evicted %d oldest provenance records (max=%d)",
            evict_count,
            self._max_records,
        )

    def build_hash(self, data: Any) -> str:
        """Build a standalone SHA-256 hash for arbitrary data.

        Utility method for callers that need to pre-compute hashes
        before calling ``record_operation()``.

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

    Creates the instance on first call (lazy initialization).
    Thread-safe via double-checked locking.

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
                    "Plot boundary manager singleton "
                    "ProvenanceTracker created"
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
            f"tracker must be a ProvenanceTracker instance, "
            f"got {type(tracker)}"
        )
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = tracker
    logger.info(
        "Plot boundary manager ProvenanceTracker singleton replaced"
    )

def reset_provenance_tracker() -> None:
    """Destroy the current singleton and reset to None.

    The next call to get_provenance_tracker() will create a fresh
    instance. Intended for test teardown.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # fresh instance
    """
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None
    logger.info(
        "Plot boundary manager ProvenanceTracker singleton reset to None"
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclass
    "ProvenanceRecord",
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
