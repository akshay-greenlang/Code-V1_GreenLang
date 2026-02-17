# -*- coding: utf-8 -*-
"""
Provenance Tracking for Climate Hazard Connector - AGENT-DATA-020

Provides SHA-256 based audit trail tracking for all climate hazard
connector operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across hazard source management, risk
index calculation, scenario projection, exposure/vulnerability
assessment, and compliance reporting.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every climate hazard operation

Entity Types:
    - hazard_source: External climate hazard data providers and feeds
    - hazard_data: Raw and processed climate hazard observations
    - risk_index: Calculated climate risk indices and scores
    - scenario_projection: Future climate scenario projections (RCP/SSP)
    - asset: Physical assets, facilities, and infrastructure
    - exposure: Asset and portfolio exposure assessments
    - vulnerability: Vulnerability and adaptive capacity scores
    - compliance_report: TCFD, CSRD, EU Taxonomy compliance reports

Actions (36):
    Source management: register_source, update_source, delete_source
    Data operations: ingest_data, query_data, aggregate_data
    Risk calculation: calculate_risk, calculate_multi_hazard,
        calculate_compound, rank_hazards
    Scenario projection: project_scenario, project_multi,
        project_timeseries
    Asset management: register_asset, update_asset, delete_asset
    Exposure assessment: assess_exposure, assess_portfolio,
        assess_supply_chain, identify_hotspots
    Vulnerability scoring: score_vulnerability, score_sector,
        create_sensitivity, create_adaptive, calculate_residual,
        rank_entities
    Reporting: generate_report, generate_tcfd, generate_csrd,
        generate_taxonomy
    Pipeline: run_pipeline, run_batch
    Maintenance: clear_engine, export_data, import_data, search_data

Example:
    >>> from greenlang.climate_hazard.provenance import ProvenanceTracker
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceEntry:
    """A single tamper-evident provenance record for a climate hazard event.

    Attributes:
        entity_type: Type of entity being tracked (hazard_source,
            hazard_data, risk_index, scenario_projection, asset,
            exposure, vulnerability, compliance_report).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (register_source, ingest_data,
            calculate_risk, project_scenario, assess_exposure,
            score_vulnerability, generate_report, run_pipeline, etc.).
        hash_value: SHA-256 hash of the input data at the time of recording.
        parent_hash: SHA-256 chain hash of the immediately preceding entry.
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
    "hazard_source",
    "hazard_data",
    "risk_index",
    "scenario_projection",
    "asset",
    "exposure",
    "vulnerability",
    "compliance_report",
})

VALID_ACTIONS = frozenset({
    # hazard_source actions (source management)
    "register_source",
    "update_source",
    "delete_source",
    # hazard_data actions (data operations)
    "ingest_data",
    "query_data",
    "aggregate_data",
    # risk_index actions (risk calculation)
    "calculate_risk",
    "calculate_multi_hazard",
    "calculate_compound",
    "rank_hazards",
    # scenario_projection actions
    "project_scenario",
    "project_multi",
    "project_timeseries",
    # asset actions (asset management)
    "register_asset",
    "update_asset",
    "delete_asset",
    # exposure actions (exposure assessment)
    "assess_exposure",
    "assess_portfolio",
    "assess_supply_chain",
    "identify_hotspots",
    # vulnerability actions (vulnerability scoring)
    "score_vulnerability",
    "score_sector",
    "create_sensitivity",
    "create_adaptive",
    "calculate_residual",
    "rank_entities",
    # compliance_report actions (reporting)
    "generate_report",
    "generate_tcfd",
    "generate_csrd",
    "generate_taxonomy",
    # pipeline actions
    "run_pipeline",
    "run_batch",
    # maintenance actions
    "clear_engine",
    "export_data",
    "import_data",
    "search_data",
})


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for climate hazard connector operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    The genesis hash anchors the chain. Every new entry incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types:
        - ``hazard_source``: External climate hazard data providers and
          feeds (e.g., NOAA, Copernicus, World Bank Climate).
        - ``hazard_data``: Raw and processed climate hazard observations
          such as temperature extremes, precipitation, sea level rise,
          wildfire risk, and flood frequency data.
        - ``risk_index``: Calculated climate risk indices combining
          hazard probability, exposure magnitude, and vulnerability
          factors into composite scores.
        - ``scenario_projection``: Future climate scenario projections
          under RCP/SSP pathways with time-horizon modeling.
        - ``asset``: Physical assets, facilities, infrastructure, and
          supply chain nodes subject to climate hazard exposure.
        - ``exposure``: Asset-level and portfolio-level exposure
          assessments quantifying potential climate impact.
        - ``vulnerability``: Vulnerability scores capturing sensitivity,
          adaptive capacity, and residual risk after adaptation.
        - ``compliance_report``: Generated TCFD, CSRD/ESRS, and EU
          Taxonomy compliance reports and disclosures.

    Supported actions (36):
        Source management: register_source, update_source, delete_source.
        Data operations: ingest_data, query_data, aggregate_data.
        Risk calculation: calculate_risk, calculate_multi_hazard,
            calculate_compound, rank_hazards.
        Scenario projection: project_scenario, project_multi,
            project_timeseries.
        Asset management: register_asset, update_asset, delete_asset.
        Exposure assessment: assess_exposure, assess_portfolio,
            assess_supply_chain, identify_hotspots.
        Vulnerability scoring: score_vulnerability, score_sector,
            create_sensitivity, create_adaptive, calculate_residual,
            rank_entities.
        Reporting: generate_report, generate_tcfd, generate_csrd,
            generate_taxonomy.
        Pipeline: run_pipeline, run_batch.
        Maintenance: clear_engine, export_data, import_data, search_data.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by ``"entity_type:entity_id"``.
        _global_chain: Flat list of all ProvenanceEntry objects in insertion order.
        _last_chain_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("hazard_source", "register_source", "src_001")
        >>> assert entry.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self, genesis_hash: str = "greenlang-climate-hazard-connector-genesis"
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis hash.
                Defaults to ``"greenlang-climate-hazard-connector-genesis"``.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self._chain_store: Dict[str, List[ProvenanceEntry]] = {}
        self._global_chain: List[ProvenanceEntry] = []
        self._last_chain_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
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
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry for a climate hazard connector operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record. Additional ``metadata`` is merged
        into the entry metadata alongside the computed data hash.

        Args:
            entity_type: Type of entity (hazard_source, hazard_data,
                risk_index, scenario_projection, asset, exposure,
                vulnerability, compliance_report).
            action: Action performed (register_source, ingest_data,
                calculate_risk, project_scenario, assess_exposure,
                score_vulnerability, generate_report, run_pipeline,
                etc.). See VALID_ACTIONS for the complete set.
            entity_id: Unique entity identifier.
            data: Optional serializable payload; its SHA-256 hash is
                stored. Pass ``None`` to record an action without
                associated data.
            metadata: Optional dictionary of extra contextual fields to
                store alongside the data hash.

        Returns:
            The newly created :class:`ProvenanceEntry`.

        Raises:
            ValueError: If ``entity_type``, ``action``, or ``entity_id``
                are empty strings.
        """
        if not entity_type:
            raise ValueError("entity_type must not be empty")
        if not action:
            raise ValueError("action must not be empty")
        if not entity_id:
            raise ValueError("entity_id must not be empty")

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

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash_prefix=%s",
            entity_type,
            entity_id[:16],
            action,
            chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire global provenance chain.

        Walks the global chain in insertion order and checks that every
        entry contains all required fields and that the chain is
        structurally consistent: the first entry chains from the genesis
        hash, and each subsequent entry's parent_hash matches the
        preceding entry's hash_value.

        Returns:
            ``True`` if the chain is intact, ``False`` if any entry is
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

        for i, entry in enumerate(chain):
            # Validate all required fields are present and non-empty
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    logger.warning(
                        "verify_chain: entry[%d] missing or empty field '%s'",
                        i,
                        field_name,
                    )
                    return False

            # First entry must chain from the genesis hash
            if i == 0 and entry.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: entry[0] parent_hash does not match genesis hash"
                )
                return False

            # Each subsequent entry's parent must match the previous entry's hash
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between entry[%d] and entry[%d]",
                    i - 1,
                    i,
                )
                return False

        logger.debug("verify_chain: %d entries verified successfully", len(chain))
        return True

    def get_chain(self) -> List[ProvenanceEntry]:
        """Return the full global provenance chain in insertion order.

        Returns:
            List of all :class:`ProvenanceEntry` objects recorded, oldest
            first. Returns an empty list if no entries have been recorded.
        """
        with self._lock:
            return list(self._global_chain)

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
            entity_type: Optional entity type to filter by (hazard_source,
                hazard_data, risk_index, scenario_projection, asset,
                exposure, vulnerability, compliance_report).
            action: Optional action to filter by. See VALID_ACTIONS.
            limit: Optional maximum number of entries to return.
                When set, the most recent matching entries are returned.

        Returns:
            List of matching :class:`ProvenanceEntry` objects. Oldest
            first unless truncated by ``limit``, in which case the last
            ``limit`` entries (most recent) are returned.
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

    def get_entries_by_entity(
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

    def get_entry_by_hash(self, hash_value: str) -> Optional[ProvenanceEntry]:
        """Look up a single provenance entry by its SHA-256 chain hash.

        Performs a linear scan of the global chain to find the entry whose
        ``hash_value`` matches the provided value.

        Args:
            hash_value: The SHA-256 chain hash to search for.

        Returns:
            The matching :class:`ProvenanceEntry`, or ``None`` if no entry
            with the given hash exists.
        """
        if not hash_value:
            return None

        with self._lock:
            for entry in self._global_chain:
                if entry.hash_value == hash_value:
                    return entry
        return None

    def get_entity_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Return all provenance entries for a given entity_id across all types.

        Scans the global chain and collects every entry whose ``entity_id``
        matches, preserving insertion order.

        Args:
            entity_id: The entity identifier to look up.

        Returns:
            List of :class:`ProvenanceEntry` objects for the entity,
            oldest first. Returns an empty list if no entries match.
        """
        with self._lock:
            return [
                entry
                for entry in self._global_chain
                if entry.entity_id == entity_id
            ]

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the entire global provenance chain as a list of dictionaries.

        Suitable for serializing to JSON for external audit systems.

        Returns:
            List of entry dictionaries in insertion order (oldest first).
        """
        with self._lock:
            return [entry.to_dict() for entry in self._global_chain]

    def export_json(self) -> str:
        """Export all provenance records as a formatted JSON string.

        Returns:
            Indented JSON string representation of the global chain.
        """
        return json.dumps(self.export_chain(), indent=2, default=str)

    def reset(self) -> None:
        """Clear all provenance state and reset to genesis.

        After calling this method the tracker behaves as if newly
        constructed. Primarily intended for testing.
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
                (or the genesis hash for the first entry).
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
                    "Climate hazard connector singleton ProvenanceTracker created"
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
    logger.info("Climate hazard connector ProvenanceTracker singleton replaced")


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
        "Climate hazard connector ProvenanceTracker singleton reset to None"
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
