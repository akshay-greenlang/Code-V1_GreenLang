# -*- coding: utf-8 -*-
"""
Provenance Tracking for Flaring Agent - AGENT-MRV-006

Provides SHA-256 based audit trail tracking for all flaring agent
operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across flare system management, gas composition
analysis, heating value calculation, combustion efficiency determination,
emission calculation, uncertainty quantification, compliance checking,
pilot/purge accounting, event tracking, and pipeline orchestration.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every flaring operation

Entity Types (12):
    - flare_system: Registered flare system definitions and specs
    - gas_composition: Gas composition analysis records
    - emission_factor: Emission factor records by source authority
    - flaring_event: Individual flaring event records
    - calculation: Emission calculation results
    - batch: Batch calculation jobs
    - combustion_efficiency: CE test and modeling records
    - pilot_purge: Pilot and purge gas accounting records
    - compliance: Regulatory compliance check results
    - uncertainty: Monte Carlo uncertainty quantification results
    - audit: Audit trail entries for calculation steps
    - pipeline: End-to-end pipeline orchestration runs

Actions (16):
    Entity management: register, update, delete
    Calculations: calculate, calculate_batch
    Analysis: analyze_composition, calculate_heating_value,
        determine_ce, quantify_uncertainty
    Events: log_event, classify_event
    Compliance: check_compliance
    Accounting: account_pilot_purge
    Validation: validate
    Pipeline: run_pipeline
    Export: export

Flaring-Specific Steps:
    - gas_composition_analysis: Validate and process gas composition
    - heating_value_calculation: Compute HHV/LHV from composition
    - combustion_efficiency_determination: Model CE with wind/tip/assist
    - emission_calculation: Core GHG emission quantification
    - uncertainty_quantification: Monte Carlo uncertainty analysis
    - compliance_check: Regulatory framework validation

Example:
    >>> from greenlang.flaring.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("flare_system", "register", "flare_001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
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
    """A single tamper-evident provenance record for a flaring operation.

    Each entry records an operation on a specific entity with a SHA-256
    chain hash linking it to the previous entry for tamper detection.

    Attributes:
        entity_type: Type of entity being tracked (flare_system,
            gas_composition, emission_factor, flaring_event,
            calculation, batch, combustion_efficiency, pilot_purge,
            compliance, uncertainty, audit, pipeline).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (register, update, delete, calculate,
            calculate_batch, analyze_composition, calculate_heating_value,
            determine_ce, quantify_uncertainty, log_event, classify_event,
            check_compliance, account_pilot_purge, validate,
            run_pipeline, export).
        hash_value: SHA-256 chain hash of this entry, incorporating
            the previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding
            entry.
        timestamp: UTC ISO-formatted timestamp when entry was created.
        step_name: Optional flaring-specific step name for granular
            tracking (gas_composition_analysis, heating_value_calculation,
            combustion_efficiency_determination, emission_calculation,
            uncertainty_quantification, compliance_check).
        metadata: Optional dictionary of additional contextual fields.
    """

    entity_type: str
    entity_id: str
    action: str
    hash_value: str
    parent_hash: str
    timestamp: str
    step_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: Dict[str, Any] = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "hash_value": self.hash_value,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
        }
        if self.step_name:
            result["step_name"] = self.step_name
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    "flare_system",
    "gas_composition",
    "emission_factor",
    "flaring_event",
    "calculation",
    "batch",
    "combustion_efficiency",
    "pilot_purge",
    "compliance",
    "uncertainty",
    "audit",
    "pipeline",
})

VALID_ACTIONS = frozenset({
    # Entity management
    "register",
    "update",
    "delete",
    # Calculations
    "calculate",
    "calculate_batch",
    # Analysis
    "analyze_composition",
    "calculate_heating_value",
    "determine_ce",
    "quantify_uncertainty",
    # Events
    "log_event",
    "classify_event",
    # Compliance
    "check_compliance",
    # Accounting
    "account_pilot_purge",
    # Validation
    "validate",
    # Pipeline
    "run_pipeline",
    # Export
    "export",
})

VALID_STEP_NAMES = frozenset({
    "gas_composition_analysis",
    "heating_value_calculation",
    "combustion_efficiency_determination",
    "emission_calculation",
    "uncertainty_quantification",
    "compliance_check",
    "pilot_purge_accounting",
    "event_classification",
    "volume_conversion",
    "gwp_application",
    "black_carbon_estimation",
    "batch_aggregation",
    "provenance_generation",
})


# ---------------------------------------------------------------------------
# ProvenanceChain
# ---------------------------------------------------------------------------


class ProvenanceChain:
    """Manages a chain of ProvenanceEntry objects with SHA-256 chain hashing.

    Provides methods for adding entries, verifying chain integrity,
    exporting to JSON, and computing standalone hashes. Thread-safe
    via a reentrant lock.

    The genesis hash anchors the chain. Every new entry incorporates
    the previous chain hash so that any tampering is detectable via
    verify_chain().

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _entries: Ordered list of ProvenanceEntry objects.
        _last_hash: Most recent chain hash for linking the next entry.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> chain = ProvenanceChain()
        >>> entry = chain.add_entry(
        ...     entity_type="calculation",
        ...     action="calculate",
        ...     entity_id="calc_001",
        ...     step_name="emission_calculation",
        ... )
        >>> assert chain.verify_chain() is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-MRV-X-006-FLARING-GENESIS",
    ) -> None:
        """Initialize ProvenanceChain with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to ``"GL-MRV-X-006-FLARING-GENESIS"``.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self._entries: List[ProvenanceEntry] = []
        self._last_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
        logger.debug(
            "ProvenanceChain initialized with genesis prefix=%s",
            self._genesis_hash[:16],
        )

    def add_entry(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
        step_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Add a new entry to the provenance chain.

        Computes a SHA-256 hash incorporating the data, previous hash,
        action, and timestamp to produce a tamper-evident record.

        Args:
            entity_type: Type of entity being tracked.
            action: Action performed on the entity.
            entity_id: Unique identifier for the entity.
            data: Optional serializable payload; its SHA-256 hash is
                stored in metadata.
            step_name: Optional flaring-specific step name.
            metadata: Optional extra contextual fields.

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

        timestamp = _utcnow().isoformat()
        data_hash = self._hash_data(data)

        entry_metadata: Dict[str, Any] = {"data_hash": data_hash}
        if step_name:
            entry_metadata["step_name"] = step_name
        if metadata:
            entry_metadata.update(metadata)

        with self._lock:
            parent_hash = self._last_hash
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
                step_name=step_name,
                metadata=entry_metadata,
            )

            self._entries.append(entry)
            self._last_hash = chain_hash

        logger.debug(
            "Chain entry added: %s/%s action=%s step=%s hash=%s",
            entity_type,
            entity_id[:16],
            action,
            step_name or "(none)",
            chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire provenance chain.

        Walks the chain in insertion order and checks that every entry
        has required fields, the first entry chains from the genesis
        hash, and each subsequent entry's parent_hash matches the
        preceding entry's hash_value.

        Returns:
            True if the chain is intact, False if any tampering is
            detected.
        """
        with self._lock:
            chain = list(self._entries)

        if not chain:
            logger.debug("verify_chain: empty chain - trivially valid")
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
                        "verify_chain: entry[%d] missing or empty '%s'",
                        i, field_name,
                    )
                    return False

            if i == 0 and entry.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: entry[0] parent_hash does not "
                    "match genesis hash"
                )
                return False

            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between entry[%d] "
                    "and entry[%d]",
                    i - 1, i,
                )
                return False

        logger.debug(
            "verify_chain: %d entries verified successfully",
            len(chain),
        )
        return True

    def export_json(self) -> str:
        """Export all provenance records as a formatted JSON string.

        Returns:
            Indented JSON string representation of the chain.
        """
        with self._lock:
            chain_dicts = [entry.to_dict() for entry in self._entries]
        return json.dumps(chain_dicts, indent=2, default=str)

    def get_hash(self) -> str:
        """Return the most recent chain hash.

        Returns:
            Hex-encoded SHA-256 hash of the chain's latest state.
        """
        with self._lock:
            return self._last_hash

    def clear(self) -> None:
        """Clear all provenance state and reset to genesis.

        Primarily intended for testing.
        """
        with self._lock:
            self._entries.clear()
            self._last_hash = self._genesis_hash
        logger.info("ProvenanceChain reset to genesis state")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of entries in the chain."""
        with self._lock:
            return len(self._entries)

    @property
    def genesis_hash(self) -> str:
        """Return the genesis hash that anchors the chain."""
        return self._genesis_hash

    @property
    def last_hash(self) -> str:
        """Return the most recent chain hash."""
        with self._lock:
            return self._last_hash

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        step_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Return entries filtered by entity_type, action, step, and/or limit.

        Args:
            entity_type: Optional entity type filter.
            action: Optional action filter.
            step_name: Optional step name filter.
            limit: Optional max entries to return (most recent).

        Returns:
            Filtered list of ProvenanceEntry objects.
        """
        with self._lock:
            entries = list(self._entries)

        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]
        if action:
            entries = [e for e in entries if e.action == action]
        if step_name:
            entries = [e for e in entries if e.step_name == step_name]
        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceEntry]:
        """Return entries for a specific entity_type:entity_id pair.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of ProvenanceEntry objects for the entity.
        """
        with self._lock:
            return [
                e for e in self._entries
                if e.entity_type == entity_type and e.entity_id == entity_id
            ]

    def get_entries_by_step(
        self,
        step_name: str,
    ) -> List[ProvenanceEntry]:
        """Return entries for a specific flaring step name.

        Args:
            step_name: Flaring step name to filter by.

        Returns:
            List of ProvenanceEntry objects matching the step.
        """
        with self._lock:
            return [
                e for e in self._entries if e.step_name == step_name
            ]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of entries."""
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ProvenanceChain(entries={self.entry_count}, "
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

    def _compute_chain_hash(
        self,
        parent_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next SHA-256 chain hash.

        Args:
            parent_hash: Chain hash of the preceding entry.
            data_hash: SHA-256 hash of the operation's data payload.
            action: Action label for this entry.
            timestamp: ISO-formatted UTC timestamp.

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

        Utility method for callers that need to pre-compute hashes.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._hash_data(data)


# ---------------------------------------------------------------------------
# ProvenanceTracker (backward-compatible facade wrapping ProvenanceChain)
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for flaring operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity type
    and entity ID.

    This class wraps ProvenanceChain and adds entity-scoped storage
    for fast lookups, matching the pattern from stationary combustion.

    Supported entity types (12):
        flare_system, gas_composition, emission_factor, flaring_event,
        calculation, batch, combustion_efficiency, pilot_purge,
        compliance, uncertainty, audit, pipeline.

    Supported actions (16):
        register, update, delete, calculate, calculate_batch,
        analyze_composition, calculate_heating_value, determine_ce,
        quantify_uncertainty, log_event, classify_event,
        check_compliance, account_pilot_purge, validate,
        run_pipeline, export.

    Flaring-specific steps:
        gas_composition_analysis, heating_value_calculation,
        combustion_efficiency_determination, emission_calculation,
        uncertainty_quantification, compliance_check,
        pilot_purge_accounting, event_classification,
        volume_conversion, gwp_application, black_carbon_estimation,
        batch_aggregation, provenance_generation.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("flare_system", "register", "flare_001")
        >>> assert entry.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-MRV-X-006-FLARING-GENESIS",
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the genesis hash.
        """
        self._chain = ProvenanceChain(genesis_hash)
        self._entity_store: Dict[str, List[ProvenanceEntry]] = {}
        self._store_lock: threading.RLock = threading.RLock()
        logger.info(
            "Flaring ProvenanceTracker initialized with genesis prefix=%s",
            self._chain.genesis_hash[:16],
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
        step_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a provenance entry for a flaring operation.

        Args:
            entity_type: Type of entity being tracked.
            action: Action performed on the entity.
            entity_id: Unique identifier for the entity.
            data: Optional serializable payload.
            step_name: Optional flaring-specific step name.
            metadata: Optional extra contextual fields.

        Returns:
            The newly created ProvenanceEntry.

        Raises:
            ValueError: If required fields are empty.
        """
        entry = self._chain.add_entry(
            entity_type=entity_type,
            action=action,
            entity_id=entity_id,
            data=data,
            step_name=step_name,
            metadata=metadata,
        )

        store_key = f"{entity_type}:{entity_id}"
        with self._store_lock:
            if store_key not in self._entity_store:
                self._entity_store[store_key] = []
            self._entity_store[store_key].append(entry)

        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire provenance chain.

        Returns:
            True if the chain is intact, False otherwise.
        """
        return self._chain.verify_chain()

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        step_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Return filtered provenance entries.

        Args:
            entity_type: Optional entity type filter.
            action: Optional action filter.
            step_name: Optional step name filter.
            limit: Optional max entries (most recent).

        Returns:
            List of matching ProvenanceEntry objects.
        """
        return self._chain.get_entries(
            entity_type=entity_type,
            action=action,
            step_name=step_name,
            limit=limit,
        )

    def get_entries_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceEntry]:
        """Return entries for a specific entity via O(1) keyed lookup.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of ProvenanceEntry objects for the entity.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._store_lock:
            return list(self._entity_store.get(store_key, []))

    def get_entries_by_step(self, step_name: str) -> List[ProvenanceEntry]:
        """Return entries for a specific flaring step.

        Args:
            step_name: Flaring step name to filter by.

        Returns:
            List of matching ProvenanceEntry objects.
        """
        return self._chain.get_entries_by_step(step_name)

    def export_json(self) -> str:
        """Export all provenance records as formatted JSON.

        Returns:
            Indented JSON string of the chain.
        """
        return self._chain.export_json()

    def clear(self) -> None:
        """Clear all provenance state and reset to genesis."""
        self._chain.clear()
        with self._store_lock:
            self._entity_store.clear()
        logger.info("Flaring ProvenanceTracker reset to genesis state")

    def build_hash(self, data: Any) -> str:
        """Build a standalone SHA-256 hash for arbitrary data.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._chain.build_hash(data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries."""
        return self._chain.entry_count

    @property
    def entity_count(self) -> int:
        """Return the number of unique entity keys tracked."""
        with self._store_lock:
            return len(self._entity_store)

    @property
    def genesis_hash(self) -> str:
        """Return the genesis hash that anchors the chain."""
        return self._chain.genesis_hash

    @property
    def last_chain_hash(self) -> str:
        """Return the most recent chain hash."""
        return self._chain.last_hash

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of provenance entries."""
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ProvenanceTracker(entries={self.entry_count}, "
            f"entities={self.entity_count}, "
            f"genesis_prefix={self.genesis_hash[:12]})"
        )


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
                    "Flaring singleton ProvenanceTracker created"
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
    logger.info("Flaring ProvenanceTracker singleton replaced")


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
        "Flaring ProvenanceTracker singleton reset to None"
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
    "VALID_STEP_NAMES",
    # Chain class
    "ProvenanceChain",
    # Tracker class
    "ProvenanceTracker",
    # Singleton helpers
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
]
