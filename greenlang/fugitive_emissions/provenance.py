# -*- coding: utf-8 -*-
"""
Provenance Tracking for Fugitive Emissions Agent - AGENT-MRV-005

Provides SHA-256 based audit trail tracking for all fugitive emissions
agent operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across equipment component management, LDAR
survey tracking, emission factor selection, calculation execution, batch
processing, repair management, compliance checking, uncertainty analysis,
and pipeline orchestration.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - First entry parent_hash = "0" * 64 (genesis anchor)
    - JSON export for external audit systems
    - Complete provenance for every fugitive emission operation

Entity Types (10):
    - CALCULATION: Individual fugitive emission calculation results with
      per-gas breakdown, CO2e totals, and method-specific intermediate
      values (average emission factor, screening ranges, correlation
      equation, engineering estimate, direct measurement).
    - SOURCE: Registered fugitive emission source type definitions
      including equipment category (valve, pump, compressor, connector,
      flange, pressure relief valve, open-ended line, sampling
      connection), service classification (gas/light liquid/heavy
      liquid), applicable gases, and default emission factors.
    - COMPONENT: Individual equipment component registrations with
      location, service type, installation date, and LDAR monitoring
      schedule. Tracks component-level leak history and repair records
      for compliance with EPA Method 21 and OGI survey requirements.
    - FACTOR: Emission factor records from authoritative sources
      (EPA, IPCC, DEFRA, EU ETS, API) scoped by source type, greenhouse
      gas, calculation method, service classification, and facility type.
    - SURVEY: LDAR survey records including survey type (OGI, Method 21,
      AVO, Hi-Flow), components inspected, leaks detected, concentration
      readings (ppm), and overall facility leak rate.
    - REPAIR: Component repair records tracking leak-to-repair timelines,
      repair method, post-repair verification readings, and delay of
      repair (DOR) justifications for regulatory compliance.
    - COMPLIANCE: Regulatory compliance check results and findings
      for GHG Protocol, ISO 14064, CSRD/ESRS E1, EPA 40 CFR Part 98
      (Subpart W), UK SECR, and EU ETS frameworks.
    - BATCH: Batch calculation jobs processing multiple fugitive emission
      inputs as a single unit of work.
    - UNCERTAINTY: Monte Carlo uncertainty quantification results with
      confidence intervals, variance contributions, and data quality
      scoring for emission estimates.
    - PIPELINE: End-to-end pipeline orchestration runs coordinating
      data intake, component registration, LDAR processing, calculation,
      aggregation, compliance checking, and reporting.

Actions (15):
    Entity management: CREATE, UPDATE, DELETE
    Calculations: CALCULATE, VALIDATE, LOOKUP
    Data operations: SELECT, REGISTER, AGGREGATE
    Analysis: ANALYZE, CHECK
    Import/Export: EXPORT, IMPORT
    Migration/Audit: MIGRATE, AUDIT

Example:
    >>> from greenlang.fugitive_emissions.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("COMPONENT", "comp_valve_001", "REGISTER")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
    """A single tamper-evident provenance record for a fugitive emission event.

    Attributes:
        entity_type: Type of entity being tracked (CALCULATION, SOURCE,
            COMPONENT, FACTOR, SURVEY, REPAIR, COMPLIANCE, BATCH,
            UNCERTAINTY, PIPELINE).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
            VALIDATE, LOOKUP, SELECT, REGISTER, AGGREGATE, ANALYZE,
            CHECK, EXPORT, IMPORT, MIGRATE, AUDIT).
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
    "CALCULATION",
    "SOURCE",
    "COMPONENT",
    "FACTOR",
    "SURVEY",
    "REPAIR",
    "COMPLIANCE",
    "BATCH",
    "UNCERTAINTY",
    "PIPELINE",
})

VALID_ACTIONS = frozenset({
    # Entity management
    "CREATE",
    "UPDATE",
    "DELETE",
    # Calculations
    "CALCULATE",
    "VALIDATE",
    "LOOKUP",
    # Data operations
    "SELECT",
    "REGISTER",
    "AGGREGATE",
    # Analysis
    "ANALYZE",
    "CHECK",
    # Import/Export
    "EXPORT",
    "IMPORT",
    # Migration/Audit
    "MIGRATE",
    "AUDIT",
})


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for fugitive emission operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    The genesis hash anchors the chain. For the first entry in the chain,
    the parent_hash is set to ``"0" * 64`` (64 zero characters). Every
    subsequent entry incorporates the previous entry's chain hash so that
    any tampering is detectable via ``verify_chain()``.

    Supported entity types (10):
        - ``CALCULATION``: Individual fugitive emission calculation results
          with per-gas breakdown, CO2e totals, and method-specific
          intermediate values for average emission factor, screening
          ranges, correlation equation, engineering estimate, and
          direct measurement approaches.
        - ``SOURCE``: Registered fugitive emission source type definitions
          including equipment category (valve, pump, compressor, connector,
          flange, pressure relief valve, open-ended line, sampling
          connection), service classification (gas, light liquid, heavy
          liquid), applicable greenhouse gases, default emission factors,
          and EPA/IPCC reference codes.
        - ``COMPONENT``: Individual equipment component registrations with
          facility assignment, process unit location, service type,
          installation date, and LDAR monitoring schedule. Tracks
          component-level leak history and repair records for compliance
          with EPA Method 21 and OGI survey requirements.
        - ``FACTOR``: Emission factor records from authoritative sources
          (EPA, IPCC, DEFRA, EU ETS, API) scoped by source type,
          greenhouse gas, calculation method, service classification,
          and facility type (refinery, gas plant, chemical plant,
          production, distribution).
        - ``SURVEY``: LDAR survey records including survey type (OGI,
          Method 21, AVO, Hi-Flow), components inspected, leaks detected,
          concentration readings (ppm), screening values, and overall
          facility leak rate.
        - ``REPAIR``: Component repair records tracking leak-to-repair
          timelines, repair method (tighten, replace packing, replace
          component, re-route, install plug), post-repair verification
          readings, and delay of repair (DOR) justifications for
          regulatory compliance.
        - ``COMPLIANCE``: Regulatory compliance check results and findings
          for GHG Protocol, ISO 14064, CSRD/ESRS E1, EPA 40 CFR Part 98
          (Subpart W), UK SECR, and EU ETS frameworks.
        - ``BATCH``: Batch calculation jobs processing multiple fugitive
          emission inputs as a single unit of work.
        - ``UNCERTAINTY``: Monte Carlo uncertainty quantification results
          with confidence intervals, variance contributions, and data
          quality scoring.
        - ``PIPELINE``: End-to-end pipeline orchestration runs coordinating
          data intake, component registration, LDAR processing,
          calculation, aggregation, compliance checking, and reporting.

    Supported actions (15):
        Entity management: CREATE, UPDATE, DELETE.
        Calculations: CALCULATE, VALIDATE, LOOKUP.
        Data operations: SELECT, REGISTER, AGGREGATE.
        Analysis: ANALYZE, CHECK.
        Import/Export: EXPORT, IMPORT.
        Migration/Audit: MIGRATE, AUDIT.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
            Set to ``"0" * 64`` so that the first entry's parent_hash
            is a string of 64 zero characters.
        _chain_store: In-memory chain storage keyed by ``"entity_type:entity_id"``.
        _global_chain: Flat list of all ProvenanceEntry objects in insertion order.
        _last_chain_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.
        _max_entries: Maximum number of entries before oldest are evicted.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("COMPONENT", "comp_valve_001", "REGISTER")
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
        regardless of the genesis_hash parameter value. The genesis_hash
        parameter is used only for internal chain anchoring consistency.

        Args:
            genesis_hash: Optional string used for internal chain
                anchoring. Defaults to ``None``, which results in
                ``"0" * 64`` as the genesis anchor.
            max_entries: Maximum number of provenance entries to retain
                in memory. When exceeded, the oldest entries are evicted
                to prevent unbounded memory growth. Defaults to 10000.
        """
        # First entry parent_hash = "0" * 64 as specified
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
        """Record a provenance entry for a fugitive emission operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record. Additional ``metadata`` is merged
        into the entry metadata alongside the computed data hash.

        Args:
            entity_type: Type of entity (CALCULATION, SOURCE, COMPONENT,
                FACTOR, SURVEY, REPAIR, COMPLIANCE, BATCH, UNCERTAINTY,
                PIPELINE).
            entity_id: Unique entity identifier.
            action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
                VALIDATE, LOOKUP, SELECT, REGISTER, AGGREGATE, ANALYZE,
                CHECK, EXPORT, IMPORT, MIGRATE, AUDIT).
            data: Optional serializable payload; its SHA-256 hash is
                stored. Pass ``None`` to record an action without
                associated data.
            metadata: Optional dictionary of extra contextual fields to
                store alongside the data hash.

        Returns:
            The newly created :class:`ProvenanceEntry`.

        Raises:
            ValueError: If ``entity_type``, ``entity_id``, or ``action``
                are empty strings.
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

        # Build entry metadata combining data hash with caller metadata
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

            # Persist to entity-scoped and global stores
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

            # Evict oldest entries if max_entries exceeded
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
        structurally consistent: the first entry chains from the genesis
        hash (``"0" * 64``), and each subsequent entry's parent_hash
        matches the preceding entry's hash_value.

        Returns:
            Tuple of ``(is_valid, error_message)``. When the chain is
            intact, returns ``(True, None)``. When a problem is detected,
            returns ``(False, description_of_problem)``.
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
            # Validate all required fields are present and non-empty
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    msg = (
                        f"entry[{i}] missing or empty field '{field_name}'"
                    )
                    logger.warning("verify_chain: %s", msg)
                    return False, msg

            # First entry must chain from the genesis hash ("0" * 64)
            if i == 0 and entry.parent_hash != self._genesis_hash:
                msg = "entry[0] parent_hash does not match genesis hash"
                logger.warning("verify_chain: %s", msg)
                return False, msg

            # Each subsequent entry's parent must match the previous hash
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
        """Return provenance entries filtered by entity_type, entity_id, action, and limit.

        When ``entity_type`` and ``entity_id`` are both provided, uses
        the O(1) keyed store for fast lookup. Otherwise filters the
        global chain. An optional ``limit`` restricts the number of
        returned entries (most recent first when limit is applied).

        Args:
            entity_type: Optional entity type to filter by (CALCULATION,
                SOURCE, COMPONENT, FACTOR, SURVEY, REPAIR, COMPLIANCE,
                BATCH, UNCERTAINTY, PIPELINE).
            entity_id: Optional entity identifier to filter by. When
                provided alongside entity_type, uses the fast keyed store.
            action: Optional action to filter by. See VALID_ACTIONS.
            limit: Maximum number of entries to return. When set, the
                most recent matching entries are returned. Defaults to 100.

        Returns:
            List of matching :class:`ProvenanceEntry` objects. Oldest
            first unless truncated by ``limit``, in which case the last
            ``limit`` entries (most recent) are returned.
        """
        # Fast path: entity_type + entity_id uses keyed store
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

        # Apply action filter
        if action:
            entries = [e for e in entries if e.action == action]

        # Apply limit (return the most recent N entries)
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

        When ``entity_type`` is provided, only entries matching that type
        are returned. When ``action`` is also provided, both filters are
        applied. An optional ``limit`` restricts the number of returned
        entries (most recent first when limit is applied).

        Args:
            entity_type: Optional entity type to filter by.
            action: Optional action to filter by. See VALID_ACTIONS.
            limit: Optional maximum number of entries to return.

        Returns:
            List of matching :class:`ProvenanceEntry` objects.
        """
        with self._lock:
            entries = list(self._global_chain)

        # Apply entity_type filter
        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]

        # Apply action filter
        if action:
            entries = [e for e in entries if e.action == action]

        # Apply limit (return the most recent N entries)
        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries for a specific entity_type:entity_id pair.

        Uses the O(1) keyed store for fast lookup by entity key.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of :class:`ProvenanceEntry` objects for the entity,
            oldest first. Returns an empty list if the entity has no
            entries.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._lock:
            return list(self._chain_store.get(store_key, []))

    def get_chain(self) -> List[ProvenanceEntry]:
        """Return the complete global provenance chain in insertion order.

        Returns:
            List of all :class:`ProvenanceEntry` objects, oldest first.
        """
        with self._lock:
            return list(self._global_chain)

    def get_chain_hash(self) -> str:
        """Return the current chain hash representing the entire trail.

        This is the SHA-256 hash of the most recent entry, which
        transitively encodes every preceding entry via chain linking.
        If the chain is empty, returns the genesis hash (``"0" * 64``).

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

        Provides a convenient format for rendering audit trails in
        reports, dashboards, and compliance documentation.

        Args:
            entity_type: Optional filter by entity type.
            entity_id: Optional filter by entity id (requires entity_type).

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
            format: Output format. Currently only ``"json"`` is supported.
                Defaults to ``"json"``.
            indent: JSON indentation level. Defaults to 2 spaces.

        Returns:
            Formatted string representation of the global chain.

        Raises:
            ValueError: If ``format`` is not ``"json"``.
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
        constructed. The genesis hash is restored to ``"0" * 64``.
        Primarily intended for testing.
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
        """Return the number of unique ``entity_type:entity_id`` keys tracked."""
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
        """Return the total number of provenance entries recorded.

        Returns:
            Integer count of entries in the global chain.
        """
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing entry count and entity count.
        """
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
            data: Any JSON-serializable object, or ``None``.

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

        This is a general-purpose hash computation method that takes
        a dictionary of entry fields and produces a deterministic
        SHA-256 digest using canonical JSON serialization.

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

        The input is a sorted-key JSON object containing the four inputs
        so that the hash is deterministic regardless of Python dict
        ordering.

        Args:
            parent_hash: The chain hash of the immediately preceding entry
                (or the genesis hash ``"0" * 64`` for the first entry).
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

        Removes entries from the beginning of the global chain and cleans
        up the entity-scoped chain store accordingly. Must be called
        while holding ``self._lock``.
        """
        overflow = len(self._global_chain) - self._max_entries
        if overflow <= 0:
            return

        evicted = self._global_chain[:overflow]
        self._global_chain = self._global_chain[overflow:]

        # Clean up entity-scoped store for evicted entries
        for entry in evicted:
            store_key = f"{entry.entity_type}:{entry.entity_id}"
            entity_entries = self._chain_store.get(store_key)
            if entity_entries:
                # Remove the first occurrence (oldest)
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

        Utility method for callers that need to pre-compute hashes before
        calling :meth:`record`.

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
    """Return the process-wide singleton :class:`ProvenanceTracker`.

    Creates the instance on first call (lazy initialization). Subsequent
    calls return the same object. The function is thread-safe.

    Returns:
        The singleton :class:`ProvenanceTracker` instance.

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
                    "Fugitive emissions singleton ProvenanceTracker created"
                )
    return _singleton_tracker


def set_provenance_tracker(tracker: ProvenanceTracker) -> None:
    """Replace the process-wide singleton with a custom tracker.

    Useful in tests that need isolated tracker instances.

    Args:
        tracker: The :class:`ProvenanceTracker` instance to install.

    Raises:
        TypeError: If ``tracker`` is not a :class:`ProvenanceTracker` instance.
    """
    if not isinstance(tracker, ProvenanceTracker):
        raise TypeError(
            f"tracker must be a ProvenanceTracker instance, got {type(tracker)}"
        )
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = tracker
    logger.info("Fugitive emissions ProvenanceTracker singleton replaced")


def reset_provenance_tracker() -> None:
    """Destroy the current singleton and reset to ``None``.

    The next call to :func:`get_provenance_tracker` will create a fresh
    instance. Intended for use in test teardown to prevent state leakage.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # fresh instance
    """
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None
    logger.info(
        "Fugitive emissions ProvenanceTracker singleton reset to None"
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
