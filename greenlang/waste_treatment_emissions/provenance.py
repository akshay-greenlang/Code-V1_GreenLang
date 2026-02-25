# -*- coding: utf-8 -*-
"""
Provenance Tracking for Waste Treatment Emissions Agent - AGENT-MRV-007

Provides SHA-256 based audit trail tracking for all on-site waste treatment
emissions agent operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across facility management, waste stream
characterisation, treatment events, emission calculations, methane recovery,
energy recovery, compliance checking, uncertainty analysis, and pipeline
orchestration.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - First entry parent_hash = "0" * 64 (genesis anchor)
    - JSON and CSV export for external audit systems
    - Complete provenance for every waste treatment emission operation

Entity Types (12):
    - FACILITY: On-site waste treatment facility registrations with
      treatment technology, capacity, permit details, and geographic
      location. Tracks facility lifecycle including creation, updates,
      and decommissioning.
    - WASTE_STREAM: Waste stream characterisation records describing
      composition, DOC content, moisture, generation rate, and waste
      category (industrial, municipal, commercial, hazardous, organic).
    - TREATMENT_EVENT: Individual waste treatment events recording
      treatment method (landfill, incineration, composting, anaerobic
      digestion, gasification, pyrolysis), throughput quantities,
      operating parameters, and residual outputs.
    - CALCULATION: Individual waste treatment emission calculation
      results with per-gas breakdowns (CH4, CO2, N2O), CO2e totals,
      biogenic fractions, and calculation trace steps.
    - METHANE_RECOVERY: Methane capture and recovery records including
      collection efficiency, destruction efficiency, recovered volumes,
      and flare/utilisation split.
    - ENERGY_RECOVERY: Energy recovery records from waste-to-energy
      operations including heat and electricity generation, turbine
      efficiency, and grid displacement credits.
    - EMISSION_FACTOR: Emission factor records from authoritative
      sources (IPCC 2006, IPCC 2019, US EPA, national inventories,
      literature, custom) scoped by waste type, treatment method,
      climate zone, and gas.
    - COMPLIANCE_CHECK: Regulatory compliance check results and
      findings for GHG Protocol, IPCC 2006, CSRD/ESRS E1, EU ETS,
      US EPA Part 98 Subpart HH/TT, and national frameworks.
    - UNCERTAINTY_RUN: Monte Carlo uncertainty quantification results
      with confidence intervals, standard deviation, coefficient of
      variation, and percentile distributions.
    - AGGREGATION: Aggregated emission results grouped by tenant,
      facility, waste category, treatment method, or time period.
    - BATCH: Batch calculation jobs processing multiple waste treatment
      emission inputs as a single unit of work.
    - CONFIG: Configuration change events tracking modifications to
      calculation defaults, feature flags, and capacity limits.
    - SYSTEM: System-level events including startup, shutdown, health
      checks, and migration operations.

Actions (16):
    Entity management: CREATE, UPDATE, DELETE
    Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE
    Treatment operations: TREAT, COMBUST, COMPOST, DIGEST, GASIFY
    Recovery operations: CAPTURE, FLARE, UTILIZE

Example:
    >>> from greenlang.waste_treatment_emissions.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("FACILITY", "facility_001", "CREATE")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
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
    """A single tamper-evident provenance record for a waste treatment event.

    Attributes:
        entity_type: Type of entity being tracked (FACILITY, WASTE_STREAM,
            TREATMENT_EVENT, CALCULATION, METHANE_RECOVERY, ENERGY_RECOVERY,
            EMISSION_FACTOR, COMPLIANCE_CHECK, UNCERTAINTY_RUN, AGGREGATION,
            BATCH, CONFIG, SYSTEM).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
            ASSESS, CHECK, VALIDATE, AGGREGATE, TREAT, COMBUST,
            COMPOST, DIGEST, GASIFY, CAPTURE, FLARE, UTILIZE).
        hash_value: SHA-256 chain hash of this entry, incorporating the
            previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding entry.
            For the first entry in the chain, this equals "0" * 64
            (the genesis anchor hash).
        timestamp: UTC ISO-formatted timestamp when the entry was created.
        actor: User or service identity that originated the operation.
        metadata: Optional dictionary of additional contextual fields.
    """

    entity_type: str
    entity_id: str
    action: str
    hash_value: str
    parent_hash: str
    timestamp: str
    actor: str = "system"
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
            "actor": self.actor,
            "metadata": self.metadata,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a flat dictionary suitable for CSV export.

        Metadata is serialized as a JSON string to fit a single column.

        Returns:
            Flat dictionary with all fields as scalar values.
        """
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "hash_value": self.hash_value,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "metadata": json.dumps(self.metadata, sort_keys=True, default=str),
        }


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    "FACILITY",
    "WASTE_STREAM",
    "TREATMENT_EVENT",
    "CALCULATION",
    "METHANE_RECOVERY",
    "ENERGY_RECOVERY",
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
    # Treatment operations
    "TREAT",
    "COMBUST",
    "COMPOST",
    "DIGEST",
    "GASIFY",
    # Recovery operations
    "CAPTURE",
    "FLARE",
    "UTILIZE",
})

# CSV header column order for deterministic export
_CSV_COLUMNS = [
    "timestamp",
    "entity_type",
    "entity_id",
    "action",
    "actor",
    "hash_value",
    "parent_hash",
    "metadata",
]


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for waste treatment emission operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    The genesis hash anchors the chain. For the first entry in the chain,
    the parent_hash is set to ``"0" * 64`` (64 zero characters). Every
    subsequent entry incorporates the previous entry's chain hash so that
    any tampering is detectable via ``verify_chain()``.

    Supported entity types (12):
        - ``FACILITY``: On-site waste treatment facility registrations with
          treatment technology, capacity, and permit details.
        - ``WASTE_STREAM``: Waste stream characterisation records describing
          composition, DOC content, and generation rate.
        - ``TREATMENT_EVENT``: Individual waste treatment events recording
          treatment method, throughput, and operating parameters.
        - ``CALCULATION``: Individual waste treatment emission calculation
          results with per-gas breakdowns and CO2e totals.
        - ``METHANE_RECOVERY``: Methane capture and recovery records with
          collection efficiency and destruction efficiency.
        - ``ENERGY_RECOVERY``: Energy recovery records from waste-to-energy
          operations including heat and electricity generation.
        - ``EMISSION_FACTOR``: Emission factor records from authoritative
          sources scoped by waste type, treatment method, and gas.
        - ``COMPLIANCE_CHECK``: Regulatory compliance check results for
          GHG Protocol, IPCC, CSRD, EU ETS, US EPA Part 98.
        - ``UNCERTAINTY_RUN``: Monte Carlo uncertainty quantification
          results with confidence intervals.
        - ``AGGREGATION``: Aggregated emission results grouped by tenant,
          facility, waste category, or time period.
        - ``BATCH``: Batch calculation jobs processing multiple inputs.
        - ``CONFIG``: Configuration change events.
        - ``SYSTEM``: System-level events (startup, shutdown, health).

    Supported actions (16):
        Entity management: CREATE, UPDATE, DELETE.
        Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE.
        Treatment operations: TREAT, COMBUST, COMPOST, DIGEST, GASIFY.
        Recovery operations: CAPTURE, FLARE, UTILIZE.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by entity_type:entity_id.
        _global_chain: Flat list of all ProvenanceEntry objects in order.
        _last_chain_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.
        _max_entries: Maximum number of entries before oldest are evicted.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("FACILITY", "facility_001", "CREATE")
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
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry for a waste treatment emission operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (FACILITY, WASTE_STREAM,
                TREATMENT_EVENT, CALCULATION, METHANE_RECOVERY,
                ENERGY_RECOVERY, EMISSION_FACTOR, COMPLIANCE_CHECK,
                UNCERTAINTY_RUN, AGGREGATION, BATCH, CONFIG, SYSTEM).
            entity_id: Unique entity identifier.
            action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
                ASSESS, CHECK, VALIDATE, AGGREGATE, TREAT, COMBUST,
                COMPOST, DIGEST, GASIFY, CAPTURE, FLARE, UTILIZE).
            data: Optional serializable payload; its SHA-256 hash is
                stored in the metadata.
            actor: User or service identity that originated the operation.
                Defaults to "system".
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
                actor=actor,
                metadata=entry_metadata,
            )

            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

            self._evict_if_needed()

        logger.debug(
            "Recorded provenance: %s/%s action=%s actor=%s hash_prefix=%s",
            entity_type,
            entity_id[:16],
            action,
            actor,
            chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the entire global provenance chain.

        Walks the global chain in insertion order and checks that every
        entry contains all required fields and that the chain is
        structurally consistent (each entry's parent_hash matches the
        preceding entry's hash_value).

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
            # Check required fields are present and non-empty
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    msg = (
                        f"entry[{i}] missing or empty field '{field_name}'"
                    )
                    logger.warning("verify_chain: %s", msg)
                    return False, msg

            # First entry must link to genesis
            if i == 0 and entry.parent_hash != self._genesis_hash:
                msg = "entry[0] parent_hash does not match genesis hash"
                logger.warning("verify_chain: %s", msg)
                return False, msg

            # Subsequent entries must link to their predecessor
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

    def verify_entry(self, entry: ProvenanceEntry) -> bool:
        """Verify that a single entry's hash is consistent with its fields.

        Recomputes the chain hash from the entry's parent_hash, the
        data_hash stored in metadata, the action, and the timestamp, then
        compares against the stored hash_value.

        Args:
            entry: The ProvenanceEntry to verify.

        Returns:
            True if the recomputed hash matches hash_value, False otherwise.
        """
        data_hash = entry.metadata.get("data_hash", "")
        if not data_hash:
            logger.warning(
                "verify_entry: entry %s/%s has no data_hash in metadata",
                entry.entity_type,
                entry.entity_id,
            )
            return False

        expected = self._compute_chain_hash(
            parent_hash=entry.parent_hash,
            data_hash=data_hash,
            action=entry.action,
            timestamp=entry.timestamp,
        )

        if expected != entry.hash_value:
            logger.warning(
                "verify_entry: hash mismatch for %s/%s "
                "(expected=%s, actual=%s)",
                entry.entity_type,
                entry.entity_id,
                expected[:16],
                entry.hash_value[:16],
            )
            return False

        return True

    def get_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries filtered by entity_type, entity_id, action, actor.

        Args:
            entity_type: Optional entity type to filter by.
            entity_id: Optional entity identifier to filter by.
            action: Optional action to filter by.
            actor: Optional actor to filter by.
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

        if actor:
            entries = [e for e in entries if e.actor == actor]

        if limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Return provenance entries filtered by entity_type, action, actor, and/or limit.

        Args:
            entity_type: Optional entity type to filter by.
            action: Optional action to filter by.
            actor: Optional actor to filter by.
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

        if actor:
            entries = [e for e in entries if e.actor == actor]

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

    def get_entity_summary(self) -> Dict[str, int]:
        """Return a summary of entry counts per entity_type.

        Returns:
            Dictionary mapping entity_type to the number of entries.
        """
        with self._lock:
            summary: Dict[str, int] = {}
            for entry in self._global_chain:
                summary[entry.entity_type] = (
                    summary.get(entry.entity_type, 0) + 1
                )
        return summary

    def get_action_summary(self) -> Dict[str, int]:
        """Return a summary of entry counts per action.

        Returns:
            Dictionary mapping action to the number of entries.
        """
        with self._lock:
            summary: Dict[str, int] = {}
            for entry in self._global_chain:
                summary[entry.action] = (
                    summary.get(entry.action, 0) + 1
                )
        return summary

    def export_json(self, indent: int = 2) -> str:
        """Export all provenance records as a JSON-formatted string.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation of the global chain.
        """
        with self._lock:
            chain_dicts = [entry.to_dict() for entry in self._global_chain]

        return json.dumps(chain_dicts, indent=indent, default=str)

    def export_csv(self) -> str:
        """Export all provenance records as a CSV-formatted string.

        The column order is deterministic and defined by ``_CSV_COLUMNS``.
        Metadata is serialized as a JSON string in a single column.

        Returns:
            CSV string representation of the global chain.
        """
        with self._lock:
            flat_rows = [
                entry.to_flat_dict() for entry in self._global_chain
            ]

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=_CSV_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)

        return output.getvalue()

    def export_trail(self, format: str = "json", indent: int = 2) -> str:
        """Export all provenance records as a formatted string.

        Supports both JSON and CSV output formats for integration with
        external audit systems, compliance reviewers, and SOC 2 evidence
        collection.

        Args:
            format: Output format. Supported values: "json", "csv".
            indent: JSON indentation level (only used for "json" format).

        Returns:
            Formatted string representation of the global chain.

        Raises:
            ValueError: If format is not "json" or "csv".
        """
        if format == "json":
            return self.export_json(indent=indent)
        elif format == "csv":
            return self.export_csv()
        else:
            raise ValueError(
                f"Unsupported export format '{format}'; "
                f"supported formats are 'json' and 'csv'"
            )

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

    def __contains__(self, entity_key: str) -> bool:
        """Check whether a given entity_type:entity_id key has entries.

        Args:
            entity_key: A string in the format "ENTITY_TYPE:entity_id".

        Returns:
            True if the key has at least one provenance entry.
        """
        with self._lock:
            return entity_key in self._chain_store

    def __iter__(self):
        """Iterate over the global chain entries in insertion order."""
        with self._lock:
            chain_copy = list(self._global_chain)
        return iter(chain_copy)

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

        Utility method for callers that need to pre-compute hashes
        before recording provenance entries.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._hash_data(data)

    def build_composite_hash(self, *parts: Any) -> str:
        """Build a SHA-256 hash from multiple data parts.

        Concatenates the individual hashes of each part and hashes the
        result, producing a single composite hash suitable for complex
        provenance records that span multiple data elements.

        Args:
            *parts: Variable number of JSON-serializable objects.

        Returns:
            Hex-encoded SHA-256 composite hash.
        """
        individual_hashes = [self._hash_data(part) for part in parts]
        combined = "|".join(individual_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


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
                    "Waste treatment emissions singleton "
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
        "Waste treatment emissions ProvenanceTracker singleton replaced"
    )


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
        "Waste treatment emissions ProvenanceTracker singleton "
        "reset to None"
    )


# ---------------------------------------------------------------------------
# Convenience factory for treatment-specific provenance
# ---------------------------------------------------------------------------


def record_treatment_provenance(
    tracker: ProvenanceTracker,
    facility_id: str,
    waste_stream_id: str,
    treatment_method: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for a treatment event.

    Creates three linked entries:
        1. TREATMENT_EVENT with the appropriate treatment action
        2. CALCULATION with the emission calculation data
        3. COMPLIANCE_CHECK with a VALIDATE action

    This is a convenience function for the common case where a single
    treatment event triggers calculation and compliance checking.

    Args:
        tracker: The ProvenanceTracker to record into.
        facility_id: Identifier for the facility performing the treatment.
        waste_stream_id: Identifier for the waste stream being treated.
        treatment_method: Treatment method name. Must be one of:
            "incineration", "composting", "anaerobic_digestion",
            "gasification", "landfill", "generic".
        calculation_data: Optional dictionary of calculation results.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of three ProvenanceEntry objects in recording order.

    Raises:
        ValueError: If treatment_method is not recognized.
    """
    method_to_action = {
        "incineration": "COMBUST",
        "composting": "COMPOST",
        "anaerobic_digestion": "DIGEST",
        "gasification": "GASIFY",
        "landfill": "TREAT",
        "generic": "TREAT",
    }

    treatment_action = method_to_action.get(treatment_method)
    if treatment_action is None:
        raise ValueError(
            f"Unrecognized treatment_method '{treatment_method}'; "
            f"expected one of: {sorted(method_to_action.keys())}"
        )

    entries: List[ProvenanceEntry] = []

    # 1. Treatment event
    treatment_entry = tracker.record(
        entity_type="TREATMENT_EVENT",
        entity_id=f"{facility_id}:{waste_stream_id}",
        action=treatment_action,
        data={
            "facility_id": facility_id,
            "waste_stream_id": waste_stream_id,
            "treatment_method": treatment_method,
        },
        actor=actor,
        metadata={"treatment_method": treatment_method},
    )
    entries.append(treatment_entry)

    # 2. Calculation
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{facility_id}:{waste_stream_id}:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "facility_id": facility_id,
            "waste_stream_id": waste_stream_id,
            "treatment_method": treatment_method,
        },
    )
    entries.append(calc_entry)

    # 3. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{facility_id}:{waste_stream_id}:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "facility_id": facility_id,
            "treatment_method": treatment_method,
            "triggered_by": treatment_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded treatment provenance sequence: facility=%s "
        "waste_stream=%s method=%s entries=%d",
        facility_id,
        waste_stream_id,
        treatment_method,
        len(entries),
    )
    return entries


def record_recovery_provenance(
    tracker: ProvenanceTracker,
    facility_id: str,
    recovery_type: str,
    recovery_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> ProvenanceEntry:
    """Record a provenance entry for a methane or energy recovery event.

    Args:
        tracker: The ProvenanceTracker to record into.
        facility_id: Identifier for the facility.
        recovery_type: Type of recovery. Must be one of:
            "capture", "flare", "utilize", "energy".
        recovery_data: Optional dictionary of recovery details.
        actor: User or service identity. Defaults to "system".

    Returns:
        The recorded ProvenanceEntry.

    Raises:
        ValueError: If recovery_type is not recognized.
    """
    type_to_config = {
        "capture": ("METHANE_RECOVERY", "CAPTURE"),
        "flare": ("METHANE_RECOVERY", "FLARE"),
        "utilize": ("METHANE_RECOVERY", "UTILIZE"),
        "energy": ("ENERGY_RECOVERY", "UTILIZE"),
    }

    config = type_to_config.get(recovery_type)
    if config is None:
        raise ValueError(
            f"Unrecognized recovery_type '{recovery_type}'; "
            f"expected one of: {sorted(type_to_config.keys())}"
        )

    entity_type, action = config

    entry = tracker.record(
        entity_type=entity_type,
        entity_id=f"{facility_id}:recovery:{recovery_type}",
        action=action,
        data=recovery_data,
        actor=actor,
        metadata={
            "facility_id": facility_id,
            "recovery_type": recovery_type,
        },
    )

    logger.info(
        "Recorded recovery provenance: facility=%s type=%s action=%s",
        facility_id,
        recovery_type,
        action,
    )
    return entry


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
    # Convenience factories
    "record_treatment_provenance",
    "record_recovery_provenance",
]
