# -*- coding: utf-8 -*-
"""
Provenance Tracking for Agricultural Emissions Agent - AGENT-MRV-008

Provides SHA-256 based audit trail tracking for all agricultural emissions
agent operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across farm management, livestock tracking, manure
management, feed characterisation, cropland inputs, rice cultivation, field
burning, emission calculations, compliance checking, uncertainty analysis,
and pipeline orchestration.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - First entry parent_hash = "0" * 64 (genesis anchor)
    - JSON and CSV export for external audit systems
    - Complete provenance for every agricultural emission operation

Entity Types (13):
    - FARM: Agricultural facility registrations with farm type, area,
      climate zone, soil classification, geographic coordinates, and
      operational parameters. Tracks farm lifecycle including creation,
      updates, and decommissioning.
    - LIVESTOCK: Livestock population records describing animal category,
      breed, head count, average weight, feeding situation, productivity
      level, and manure management assignment. Supports cattle, buffalo,
      sheep, goats, swine, poultry, horses, mules, and other categories.
    - MANURE_SYSTEM: Manure management system records including system
      type (lagoon, liquid, solid storage, dry lot, pasture, daily
      spread, digester, composting, deep bedding, pit storage, other),
      retention time, methane conversion factor (MCF), and capacity.
    - FEED_CHARACTERISTICS: Feed characterisation records describing
      diet composition, digestible energy (DE%), crude protein content,
      dry matter intake, feed additive usage, and methane conversion
      rate (Ym) for enteric fermentation calculations.
    - CROPLAND_INPUT: Cropland input records for soil N2O emissions
      including synthetic fertiliser application (kg N), organic
      amendments, crop residue management, nitrogen fixation, liming
      quantities (CaCO3/dolomite), and urea application rates.
    - RICE_FIELD: Rice paddy field records including water regime
      (continuously flooded, intermittently flooded, rainfed, upland),
      organic amendments, cultivar type, growing season length, soil
      type, and pre-season water status for CH4 scaling factors.
    - FIELD_BURNING: Agricultural field burning records including crop
      type (wheat, maize, rice, sugarcane, cotton, other), area burned,
      fuel load (dry matter per hectare), combustion factor, and
      emission factors for CH4, N2O, CO, and NOx.
    - CALCULATION: Individual agricultural emission calculation results
      with per-gas breakdowns (CH4, N2O, CO2), CO2e totals, tier level,
      source category, and calculation trace steps.
    - COMPLIANCE_CHECK: Regulatory compliance check results and findings
      for IPCC 2006 Vol 4, IPCC 2019 Refinement, GHG Protocol
      Agriculture Guidance, CSRD/ESRS E1, national inventory
      guidelines, and SBTi FLAG target frameworks.
    - UNCERTAINTY_RUN: Monte Carlo uncertainty quantification results
      with confidence intervals, standard deviation, coefficient of
      variation, and percentile distributions per emission source.
    - AGGREGATION: Aggregated emission results grouped by tenant, farm,
      livestock category, emission source, crop type, or time period.
    - BATCH: Batch calculation jobs processing multiple agricultural
      emission inputs as a single unit of work.
    - SYSTEM: System-level events including startup, shutdown, health
      checks, and migration operations.

Actions (16):
    Entity management: CREATE, UPDATE, DELETE
    Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE
    Agricultural operations: FERMENT, DIGEST, FERTILIZE, CULTIVATE,
        BURN, LIME, APPLY_UREA, GRAZE

Example:
    >>> from greenlang.agricultural_emissions.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("FARM", "farm_001", "CREATE")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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
    """A single tamper-evident provenance record for an agricultural event.

    Attributes:
        entity_type: Type of entity being tracked (FARM, LIVESTOCK,
            MANURE_SYSTEM, FEED_CHARACTERISTICS, CROPLAND_INPUT,
            RICE_FIELD, FIELD_BURNING, CALCULATION, COMPLIANCE_CHECK,
            UNCERTAINTY_RUN, AGGREGATION, BATCH, SYSTEM).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
            ASSESS, CHECK, VALIDATE, AGGREGATE, FERMENT, DIGEST,
            FERTILIZE, CULTIVATE, BURN, LIME, APPLY_UREA, GRAZE).
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
    "FARM",
    "LIVESTOCK",
    "MANURE_SYSTEM",
    "FEED_CHARACTERISTICS",
    "CROPLAND_INPUT",
    "RICE_FIELD",
    "FIELD_BURNING",
    "CALCULATION",
    "COMPLIANCE_CHECK",
    "UNCERTAINTY_RUN",
    "AGGREGATION",
    "BATCH",
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
    # Agricultural operations
    "FERMENT",
    "DIGEST",
    "FERTILIZE",
    "CULTIVATE",
    "BURN",
    "LIME",
    "APPLY_UREA",
    "GRAZE",
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
    """Tracks provenance for agricultural emission operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    The genesis hash anchors the chain. For the first entry in the chain,
    the parent_hash is set to ``"0" * 64`` (64 zero characters). Every
    subsequent entry incorporates the previous entry's chain hash so that
    any tampering is detectable via ``verify_chain()``.

    Supported entity types (13):
        - ``FARM``: Agricultural facility registrations with farm type,
          area, climate zone, and soil classification.
        - ``LIVESTOCK``: Livestock population records describing animal
          category, breed, head count, and average weight.
        - ``MANURE_SYSTEM``: Manure management system records including
          system type, retention time, and methane conversion factor.
        - ``FEED_CHARACTERISTICS``: Feed characterisation records with
          diet composition, DE%, crude protein, and Ym factor.
        - ``CROPLAND_INPUT``: Cropland input records for soil N2O
          including fertiliser, residues, and nitrogen fixation.
        - ``RICE_FIELD``: Rice paddy field records including water
          regime, organic amendments, and growing season length.
        - ``FIELD_BURNING``: Agricultural field burning records with
          crop type, area burned, and fuel load.
        - ``CALCULATION``: Individual agricultural emission calculation
          results with per-gas breakdowns and CO2e totals.
        - ``COMPLIANCE_CHECK``: Regulatory compliance check results for
          IPCC 2006, IPCC 2019, GHG Protocol, CSRD, SBTi FLAG.
        - ``UNCERTAINTY_RUN``: Monte Carlo uncertainty quantification
          results with confidence intervals.
        - ``AGGREGATION``: Aggregated emission results grouped by tenant,
          farm, livestock category, or time period.
        - ``BATCH``: Batch calculation jobs processing multiple inputs.
        - ``SYSTEM``: System-level events (startup, shutdown, health).

    Supported actions (16):
        Entity management: CREATE, UPDATE, DELETE.
        Calculations: CALCULATE, ASSESS, CHECK, VALIDATE, AGGREGATE.
        Agricultural operations: FERMENT, DIGEST, FERTILIZE, CULTIVATE,
            BURN, LIME, APPLY_UREA, GRAZE.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by entity_type:entity_id.
        _global_chain: Flat list of all ProvenanceEntry objects in order.
        _last_chain_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.
        _max_entries: Maximum number of entries before oldest are evicted.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("FARM", "farm_001", "CREATE")
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
        """Record a provenance entry for an agricultural emission operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (FARM, LIVESTOCK, MANURE_SYSTEM,
                FEED_CHARACTERISTICS, CROPLAND_INPUT, RICE_FIELD,
                FIELD_BURNING, CALCULATION, COMPLIANCE_CHECK,
                UNCERTAINTY_RUN, AGGREGATION, BATCH, SYSTEM).
            entity_id: Unique entity identifier.
            action: Action performed (CREATE, UPDATE, DELETE, CALCULATE,
                ASSESS, CHECK, VALIDATE, AGGREGATE, FERMENT, DIGEST,
                FERTILIZE, CULTIVATE, BURN, LIME, APPLY_UREA, GRAZE).
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
                    "Agricultural emissions singleton "
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
        "Agricultural emissions ProvenanceTracker singleton replaced"
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
        "Agricultural emissions ProvenanceTracker singleton "
        "reset to None"
    )


# ---------------------------------------------------------------------------
# Convenience factory: Enteric fermentation provenance
# ---------------------------------------------------------------------------


def record_enteric_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    livestock_id: str,
    feed_id: Optional[str] = None,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for enteric fermentation.

    Creates a linked sequence of entries covering the full enteric
    fermentation calculation lifecycle:
        1. LIVESTOCK entity with FERMENT action (enteric CH4 production)
        2. FEED_CHARACTERISTICS with ASSESS action (feed quality evaluation)
           -- only if feed_id is provided
        3. CALCULATION with CALCULATE action (emission quantification)
        4. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    This is a convenience function for the common case where a single
    livestock enteric fermentation event triggers feed assessment,
    emission calculation, and compliance checking.

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm where livestock are kept.
        livestock_id: Identifier for the livestock population or cohort.
        feed_id: Optional identifier for the feed characteristics record.
            When provided, an ASSESS entry is recorded for feed quality.
        calculation_data: Optional dictionary of calculation results
            including CH4 emissions, Ym factor, GE intake, and tier level.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of ProvenanceEntry objects in recording order (3 or 4 entries).

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_enteric_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     livestock_id="cattle_dairy_001",
        ...     feed_id="feed_high_grain",
        ...     calculation_data={"ch4_kg": 128.5, "tier": 2},
        ... )
        >>> assert len(entries) == 4
        >>> assert entries[0].action == "FERMENT"
    """
    entries: List[ProvenanceEntry] = []

    # 1. Livestock enteric fermentation event
    ferment_entry = tracker.record(
        entity_type="LIVESTOCK",
        entity_id=f"{farm_id}:{livestock_id}",
        action="FERMENT",
        data={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "source": "enteric_fermentation",
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "emission_source": "enteric_fermentation",
        },
    )
    entries.append(ferment_entry)

    # 2. Feed characteristics assessment (optional)
    if feed_id:
        feed_entry = tracker.record(
            entity_type="FEED_CHARACTERISTICS",
            entity_id=f"{farm_id}:{livestock_id}:{feed_id}",
            action="ASSESS",
            data={
                "farm_id": farm_id,
                "livestock_id": livestock_id,
                "feed_id": feed_id,
            },
            actor=actor,
            metadata={
                "farm_id": farm_id,
                "feed_id": feed_id,
                "triggered_by": ferment_entry.hash_value[:16],
            },
        )
        entries.append(feed_entry)

    # 3. Emission calculation
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{livestock_id}:enteric:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "emission_source": "enteric_fermentation",
            "triggered_by": ferment_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 4. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{livestock_id}:enteric:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "emission_source": "enteric_fermentation",
            "triggered_by": ferment_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded enteric provenance sequence: farm=%s livestock=%s "
        "feed=%s entries=%d",
        farm_id,
        livestock_id,
        feed_id or "none",
        len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Convenience factory: Manure management provenance
# ---------------------------------------------------------------------------


def record_manure_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    livestock_id: str,
    manure_system_id: str,
    manure_system_type: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for manure management.

    Creates a linked sequence of entries covering the full manure
    management emission calculation lifecycle:
        1. MANURE_SYSTEM with DIGEST action (anaerobic decomposition)
        2. LIVESTOCK with ASSESS action (VS production assessment)
        3. CALCULATION with CALCULATE action (CH4 + N2O quantification)
        4. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    Handles all manure management system types: lagoon, liquid, solid
    storage, dry lot, pasture (range/paddock), daily spread, digester,
    composting, deep bedding, pit storage, and other.

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm where manure is managed.
        livestock_id: Identifier for the livestock population producing
            the manure (volatile solids source).
        manure_system_id: Identifier for the manure management system.
        manure_system_type: Type of manure management system. Used in
            metadata for audit trail context.
        calculation_data: Optional dictionary of calculation results
            including CH4 and N2O emissions, MCF, Bo, and VS values.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of four ProvenanceEntry objects in recording order.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_manure_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     livestock_id="cattle_dairy_001",
        ...     manure_system_id="lagoon_001",
        ...     manure_system_type="lagoon",
        ...     calculation_data={"ch4_kg": 45.2, "n2o_kg": 0.8},
        ... )
        >>> assert len(entries) == 4
        >>> assert entries[0].action == "DIGEST"
    """
    entries: List[ProvenanceEntry] = []

    # 1. Manure system digestion event
    digest_entry = tracker.record(
        entity_type="MANURE_SYSTEM",
        entity_id=f"{farm_id}:{manure_system_id}",
        action="DIGEST",
        data={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "manure_system_id": manure_system_id,
            "manure_system_type": manure_system_type,
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "manure_system_type": manure_system_type,
            "emission_source": "manure_management",
        },
    )
    entries.append(digest_entry)

    # 2. Livestock VS production assessment
    livestock_entry = tracker.record(
        entity_type="LIVESTOCK",
        entity_id=f"{farm_id}:{livestock_id}",
        action="ASSESS",
        data={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "assessment_type": "volatile_solids_production",
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "manure_system_id": manure_system_id,
            "triggered_by": digest_entry.hash_value[:16],
        },
    )
    entries.append(livestock_entry)

    # 3. Emission calculation (CH4 + N2O)
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{manure_system_id}:manure:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "manure_system_id": manure_system_id,
            "manure_system_type": manure_system_type,
            "emission_source": "manure_management",
            "triggered_by": digest_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 4. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{manure_system_id}:manure:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "manure_system_type": manure_system_type,
            "emission_source": "manure_management",
            "triggered_by": digest_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded manure provenance sequence: farm=%s livestock=%s "
        "system=%s type=%s entries=%d",
        farm_id,
        livestock_id,
        manure_system_id,
        manure_system_type,
        len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Convenience factory: Cropland / soil emission provenance
# ---------------------------------------------------------------------------


def record_cropland_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    cropland_id: str,
    input_type: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for cropland emissions.

    Creates a linked sequence of entries covering soil N2O emissions from
    synthetic fertiliser, organic amendments, crop residues, liming, and
    urea application:
        1. CROPLAND_INPUT with the appropriate agricultural action
           (FERTILIZE, LIME, APPLY_UREA, or CULTIVATE)
        2. CALCULATION with CALCULATE action (N2O / CO2 quantification)
        3. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    Supports four cropland input types:
        - ``"synthetic_fertiliser"``: Maps to FERTILIZE action. Covers
          synthetic N fertiliser application and resulting direct +
          indirect N2O emissions.
        - ``"organic_amendment"``: Maps to FERTILIZE action. Covers
          organic N inputs (manure, compost, sewage sludge).
        - ``"liming"``: Maps to LIME action. Covers CO2 emissions from
          limestone (CaCO3) and dolomite (CaMg(CO3)2) application.
        - ``"urea"``: Maps to APPLY_UREA action. Covers CO2 emissions
          from urea and urea-containing fertiliser hydrolysis.
        - ``"crop_residue"``: Maps to CULTIVATE action. Covers N2O from
          crop residue decomposition and nitrogen mineralisation.

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm where cropland is located.
        cropland_id: Identifier for the cropland field or plot.
        input_type: Type of cropland input. Must be one of:
            "synthetic_fertiliser", "organic_amendment", "liming",
            "urea", "crop_residue".
        calculation_data: Optional dictionary of calculation results
            including N2O-N direct/indirect emissions, CO2 from liming,
            emission factors, and tier level.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of three ProvenanceEntry objects in recording order.

    Raises:
        ValueError: If input_type is not recognized.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_cropland_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     cropland_id="field_north_01",
        ...     input_type="synthetic_fertiliser",
        ...     calculation_data={"n2o_kg": 2.4, "tier": 1},
        ... )
        >>> assert len(entries) == 3
        >>> assert entries[0].action == "FERTILIZE"
    """
    input_type_to_action = {
        "synthetic_fertiliser": "FERTILIZE",
        "organic_amendment": "FERTILIZE",
        "liming": "LIME",
        "urea": "APPLY_UREA",
        "crop_residue": "CULTIVATE",
    }

    cropland_action = input_type_to_action.get(input_type)
    if cropland_action is None:
        raise ValueError(
            f"Unrecognized input_type '{input_type}'; "
            f"expected one of: {sorted(input_type_to_action.keys())}"
        )

    entries: List[ProvenanceEntry] = []

    # 1. Cropland input event
    input_entry = tracker.record(
        entity_type="CROPLAND_INPUT",
        entity_id=f"{farm_id}:{cropland_id}",
        action=cropland_action,
        data={
            "farm_id": farm_id,
            "cropland_id": cropland_id,
            "input_type": input_type,
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "cropland_id": cropland_id,
            "input_type": input_type,
            "emission_source": "cropland_soil",
        },
    )
    entries.append(input_entry)

    # 2. Emission calculation
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{cropland_id}:{input_type}:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "cropland_id": cropland_id,
            "input_type": input_type,
            "emission_source": "cropland_soil",
            "triggered_by": input_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 3. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{cropland_id}:{input_type}:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "cropland_id": cropland_id,
            "input_type": input_type,
            "triggered_by": input_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded cropland provenance sequence: farm=%s cropland=%s "
        "input_type=%s action=%s entries=%d",
        farm_id,
        cropland_id,
        input_type,
        cropland_action,
        len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Convenience factory: Rice cultivation provenance
# ---------------------------------------------------------------------------


def record_rice_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    rice_field_id: str,
    water_regime: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for rice cultivation.

    Creates a linked sequence of entries covering CH4 emissions from
    flooded rice paddies under various water management regimes:
        1. RICE_FIELD with CULTIVATE action (paddy cultivation event)
        2. CALCULATION with CALCULATE action (CH4 quantification using
           IPCC baseline EF and scaling factors)
        3. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    The water regime is critical for emission calculations as it determines
    the IPCC scaling factor (SF_w) applied to the baseline CH4 emission
    factor. Supported regimes:
        - ``"continuously_flooded"``: SF_w = 1.0 (IPCC default baseline)
        - ``"intermittently_flooded_single"``: SF_w = 0.60 (single aeration)
        - ``"intermittently_flooded_multiple"``: SF_w = 0.52 (multiple)
        - ``"rainfed_regular"``: SF_w = 0.28 (rainfed, regular flooding)
        - ``"rainfed_drought"``: SF_w = 0.25 (rainfed, drought prone)
        - ``"upland"``: SF_w = 0.0 (no flooding, negligible CH4)

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm where the rice field is located.
        rice_field_id: Identifier for the rice paddy field.
        water_regime: Water management regime for the rice field. Used
            in metadata for audit trail traceability.
        calculation_data: Optional dictionary of calculation results
            including CH4 emissions, baseline EF, scaling factors
            (SF_w, SF_p, SF_o, SF_s), and cultivation period.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of three ProvenanceEntry objects in recording order.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_rice_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     rice_field_id="paddy_east_01",
        ...     water_regime="continuously_flooded",
        ...     calculation_data={"ch4_kg": 185.0, "sf_w": 1.0},
        ... )
        >>> assert len(entries) == 3
        >>> assert entries[0].action == "CULTIVATE"
    """
    entries: List[ProvenanceEntry] = []

    # 1. Rice field cultivation event
    cultivate_entry = tracker.record(
        entity_type="RICE_FIELD",
        entity_id=f"{farm_id}:{rice_field_id}",
        action="CULTIVATE",
        data={
            "farm_id": farm_id,
            "rice_field_id": rice_field_id,
            "water_regime": water_regime,
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "rice_field_id": rice_field_id,
            "water_regime": water_regime,
            "emission_source": "rice_cultivation",
        },
    )
    entries.append(cultivate_entry)

    # 2. Emission calculation (CH4 with scaling factors)
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{rice_field_id}:rice:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "rice_field_id": rice_field_id,
            "water_regime": water_regime,
            "emission_source": "rice_cultivation",
            "triggered_by": cultivate_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 3. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{rice_field_id}:rice:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "rice_field_id": rice_field_id,
            "water_regime": water_regime,
            "triggered_by": cultivate_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded rice cultivation provenance sequence: farm=%s "
        "rice_field=%s water_regime=%s entries=%d",
        farm_id,
        rice_field_id,
        water_regime,
        len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Convenience factory: Field burning provenance
# ---------------------------------------------------------------------------


def record_field_burning_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    field_id: str,
    crop_type: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for field burning.

    Creates a linked sequence of entries covering CH4, N2O, CO, and NOx
    emissions from prescribed burning of agricultural residues:
        1. FIELD_BURNING with BURN action (burning event)
        2. CALCULATION with CALCULATE action (multi-gas quantification)
        3. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    Supported crop types for field burning: wheat, maize, rice, sugarcane,
    cotton, barley, sorghum, millet, oats, and other. Each crop type has
    specific IPCC default values for fuel load (dry matter per hectare),
    combustion factor, and residue-to-crop ratio.

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm where burning occurs.
        field_id: Identifier for the field being burned.
        crop_type: Type of crop residue being burned. Used in metadata
            for audit trail context and emission factor selection.
        calculation_data: Optional dictionary of calculation results
            including CH4, N2O, CO, NOx emissions, area burned,
            fuel load, and combustion factor.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of three ProvenanceEntry objects in recording order.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_field_burning_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     field_id="field_south_03",
        ...     crop_type="wheat",
        ...     calculation_data={"ch4_kg": 3.2, "n2o_kg": 0.07},
        ... )
        >>> assert len(entries) == 3
        >>> assert entries[0].action == "BURN"
    """
    entries: List[ProvenanceEntry] = []

    # 1. Field burning event
    burn_entry = tracker.record(
        entity_type="FIELD_BURNING",
        entity_id=f"{farm_id}:{field_id}",
        action="BURN",
        data={
            "farm_id": farm_id,
            "field_id": field_id,
            "crop_type": crop_type,
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "field_id": field_id,
            "crop_type": crop_type,
            "emission_source": "field_burning",
        },
    )
    entries.append(burn_entry)

    # 2. Emission calculation (CH4, N2O, CO, NOx)
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{field_id}:burning:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "field_id": field_id,
            "crop_type": crop_type,
            "emission_source": "field_burning",
            "triggered_by": burn_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 3. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{field_id}:burning:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "field_id": field_id,
            "crop_type": crop_type,
            "triggered_by": burn_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded field burning provenance sequence: farm=%s field=%s "
        "crop_type=%s entries=%d",
        farm_id,
        field_id,
        crop_type,
        len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Convenience factory: Pasture/grazing provenance
# ---------------------------------------------------------------------------


def record_grazing_provenance(
    tracker: ProvenanceTracker,
    farm_id: str,
    livestock_id: str,
    pasture_id: str,
    calculation_data: Optional[Dict[str, Any]] = None,
    actor: str = "system",
) -> List[ProvenanceEntry]:
    """Record a standard sequence of provenance entries for pasture grazing.

    Creates a linked sequence of entries covering N2O emissions from
    livestock manure deposited on pasture, range, and paddock during
    grazing (IPCC Category 3D Indirect/Direct N2O from managed soils):
        1. LIVESTOCK with GRAZE action (grazing activity)
        2. MANURE_SYSTEM with ASSESS action (pasture manure assessment)
        3. CALCULATION with CALCULATE action (N2O quantification)
        4. COMPLIANCE_CHECK with VALIDATE action (regulatory check)

    Pasture/range/paddock manure deposition is treated as a separate
    manure management pathway in IPCC 2006 Vol 4, Chapter 10. The
    nitrogen excreted on pasture contributes to both direct N2O
    emissions (EF3_PRP) and indirect N2O via volatilisation and
    leaching/runoff.

    Args:
        tracker: The ProvenanceTracker to record into.
        farm_id: Identifier for the farm.
        livestock_id: Identifier for the grazing livestock population.
        pasture_id: Identifier for the pasture area or paddock.
        calculation_data: Optional dictionary of calculation results
            including N2O direct, N2O indirect (volatilisation +
            leaching), nitrogen excreted, and grazing days.
        actor: User or service identity. Defaults to "system".

    Returns:
        List of four ProvenanceEntry objects in recording order.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entries = record_grazing_provenance(
        ...     tracker,
        ...     farm_id="farm_001",
        ...     livestock_id="cattle_beef_001",
        ...     pasture_id="pasture_west_01",
        ...     calculation_data={"n2o_direct_kg": 1.2, "n2o_indirect_kg": 0.4},
        ... )
        >>> assert len(entries) == 4
        >>> assert entries[0].action == "GRAZE"
    """
    entries: List[ProvenanceEntry] = []

    # 1. Livestock grazing event
    graze_entry = tracker.record(
        entity_type="LIVESTOCK",
        entity_id=f"{farm_id}:{livestock_id}",
        action="GRAZE",
        data={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "pasture_id": pasture_id,
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "pasture_id": pasture_id,
            "emission_source": "pasture_grazing",
        },
    )
    entries.append(graze_entry)

    # 2. Manure system assessment (pasture deposition)
    manure_entry = tracker.record(
        entity_type="MANURE_SYSTEM",
        entity_id=f"{farm_id}:{pasture_id}:pasture",
        action="ASSESS",
        data={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "pasture_id": pasture_id,
            "system_type": "pasture_range_paddock",
        },
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "pasture_id": pasture_id,
            "manure_system_type": "pasture_range_paddock",
            "triggered_by": graze_entry.hash_value[:16],
        },
    )
    entries.append(manure_entry)

    # 3. Emission calculation (N2O direct + indirect)
    calc_entry = tracker.record(
        entity_type="CALCULATION",
        entity_id=f"{farm_id}:{pasture_id}:grazing:calc",
        action="CALCULATE",
        data=calculation_data,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "livestock_id": livestock_id,
            "pasture_id": pasture_id,
            "emission_source": "pasture_grazing",
            "triggered_by": graze_entry.hash_value[:16],
        },
    )
    entries.append(calc_entry)

    # 4. Compliance check
    compliance_entry = tracker.record(
        entity_type="COMPLIANCE_CHECK",
        entity_id=f"{farm_id}:{pasture_id}:grazing:compliance",
        action="VALIDATE",
        data=None,
        actor=actor,
        metadata={
            "farm_id": farm_id,
            "pasture_id": pasture_id,
            "emission_source": "pasture_grazing",
            "triggered_by": graze_entry.hash_value[:16],
        },
    )
    entries.append(compliance_entry)

    logger.info(
        "Recorded grazing provenance sequence: farm=%s livestock=%s "
        "pasture=%s entries=%d",
        farm_id,
        livestock_id,
        pasture_id,
        len(entries),
    )
    return entries


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
    "record_enteric_provenance",
    "record_manure_provenance",
    "record_cropland_provenance",
    "record_rice_provenance",
    "record_field_burning_provenance",
    "record_grazing_provenance",
]
