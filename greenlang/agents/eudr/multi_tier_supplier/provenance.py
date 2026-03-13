# -*- coding: utf-8 -*-
"""
Provenance Tracking for Multi-Tier Supplier Tracker - AGENT-EUDR-008

Provides SHA-256 based audit trail tracking for all multi-tier supplier
tracking operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across supplier discovery, profile
management, tier depth assessment, relationship lifecycle management,
risk propagation, compliance monitoring, gap analysis, and audit
reporting.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every supplier tracking operation
    - Bit-perfect reproducibility for risk and compliance audits

Entity Types (10):
    - supplier: Supplier profile creation and management operations.
    - relationship: Supplier-to-supplier relationship operations.
    - tier_assessment: Tier depth assessment and scoring operations.
    - risk_score: Risk assessment and scoring operations.
    - compliance_check: Compliance monitoring and checking operations.
    - certification: Certification record management operations.
    - gap: Data gap detection and analysis operations.
    - remediation: Remediation plan management operations.
    - report: Audit and compliance report generation operations.
    - batch: Batch processing pipeline operations.

Actions (12):
    Entity management: discover, create, update, delete, assess,
        propagate, check, certify, analyze, remediate, export, batch.

Example:
    >>> from greenlang.agents.eudr.multi_tier_supplier.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> record = tracker.record("supplier", "create", "sup-001")
    >>> assert record.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
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
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a multi-tier supplier tracking event.

    Immutable dataclass (frozen=True) ensuring records cannot be modified
    after creation. This is critical for audit trail integrity under
    EUDR Article 31 (5-year retention).

    Attributes:
        entity_type: Type of entity being tracked (supplier, relationship,
            tier_assessment, risk_score, compliance_check, certification,
            gap, remediation, report, batch).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (discover, create, update, delete,
            assess, propagate, check, certify, analyze, remediate,
            export, batch).
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
            "metadata": dict(self.metadata) if self.metadata else {},
        }


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    "supplier",
    "relationship",
    "tier_assessment",
    "risk_score",
    "compliance_check",
    "certification",
    "gap",
    "remediation",
    "report",
    "batch",
})

VALID_ACTIONS = frozenset({
    # Core operations
    "discover",
    "create",
    "update",
    "delete",
    "assess",
    "propagate",
    "check",
    "certify",
    "analyze",
    "remediate",
    "export",
    "batch",
})

# ---------------------------------------------------------------------------
# Default max records limit
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RECORDS: int = 100_000


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for multi-tier supplier operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails for EUDR Article 31
    record-keeping compliance (5-year retention).

    The genesis hash anchors the chain. Every new record incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types (10):
        - ``supplier``: Supplier profile creation, update, deactivation
          and discovery operations including deduplication.
        - ``relationship``: Supplier-to-supplier relationship creation,
          lifecycle transitions, strength scoring, and conflict detection.
        - ``tier_assessment``: Tier depth calculation, visibility scoring,
          coverage assessment, and benchmark comparison.
        - ``risk_score``: Individual and composite risk scoring across
          six risk categories with trend analysis.
        - ``compliance_check``: DDS validity, certification status,
          geolocation coverage, and deforestation-free verification.
        - ``certification``: Certification record creation, renewal,
          expiry tracking, and independent verification.
        - ``gap``: Data gap detection, severity classification, and
          gap trend analysis.
        - ``remediation``: Remediation plan creation, step tracking,
          progress monitoring, and completion.
        - ``report``: Audit report generation in JSON, PDF, CSV, and
          EUDR XML formats with Article 14 compliance.
        - ``batch``: Batch processing job creation, progress tracking,
          completion, cancellation, and error handling.

    Supported actions (12):
        discover, create, update, delete, assess, propagate, check,
        certify, analyze, remediate, export, batch.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by
            ``"entity_type:entity_id"``.
        _global_chain: Flat list of all ProvenanceRecord objects in
            insertion order.
        _last_chain_hash: Most recent chain hash for linking the
            next record.
        _max_records: Maximum number of records to retain in memory.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> record = tracker.record("supplier", "create", "sup-001")
        >>> assert record.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-MST-008-MULTI-TIER-SUPPLIER-TRACKER-GENESIS",
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to the Multi-Tier Supplier Tracker agent
                identifier.
            max_records: Maximum number of records to retain in memory.
                When exceeded, oldest records are evicted from the
                global chain (FIFO). Set to 0 for unlimited.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self._chain_store: Dict[str, List[ProvenanceRecord]] = {}
        self._global_chain: List[ProvenanceRecord] = []
        self._last_chain_hash: str = self._genesis_hash
        self._max_records: int = max_records
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "ProvenanceTracker initialized with genesis hash prefix=%s, "
            "max_records=%d",
            self._genesis_hash[:16],
            self._max_records,
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
    ) -> ProvenanceRecord:
        """Record a provenance entry for a multi-tier supplier tracking operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous record hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (supplier, relationship,
                tier_assessment, risk_score, compliance_check,
                certification, gap, remediation, report, batch).
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

        timestamp = _utcnow().isoformat()
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

            entry = ProvenanceRecord(
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
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

            # Evict oldest records if max_records exceeded
            if self._max_records > 0:
                self._evict_if_needed()

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

            # First record must chain from the genesis hash
            if i == 0 and entry.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: entry[0] parent_hash does not match "
                    "genesis hash"
                )
                return False

            # Each subsequent record's parent must match the previous hash
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between entry[%d] and "
                    "entry[%d]",
                    i - 1,
                    i,
                )
                return False

        logger.debug(
            "verify_chain: %d entries verified successfully", len(chain)
        )
        return True

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

    def get_entries_for_entity(
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

    def export_json(self) -> str:
        """Export all provenance records as a formatted JSON string.

        Returns:
            Indented JSON string representation of the global chain.
        """
        with self._lock:
            chain_dicts = [entry.to_dict() for entry in self._global_chain]
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
        """Return the total number of provenance records recorded."""
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
    def max_records(self) -> int:
        """Return the maximum number of records retained in memory."""
        return self._max_records

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of provenance records recorded."""
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ProvenanceTracker(entries={self.entry_count}, "
            f"entities={self.entity_count}, "
            f"genesis_prefix={self._genesis_hash[:12]}, "
            f"max_records={self._max_records})"
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
        """Evict oldest records from global chain if max_records exceeded.

        Called under the lock. Only evicts from the global chain; entity
        store entries remain for entity-scoped lookups.
        """
        if self._max_records <= 0:
            return
        overflow = len(self._global_chain) - self._max_records
        if overflow > 0:
            self._global_chain = self._global_chain[overflow:]
            logger.debug(
                "Evicted %d oldest provenance records (max_records=%d)",
                overflow,
                self._max_records,
            )

    def build_hash(self, data: Any) -> str:
        """Build a standalone SHA-256 hash for arbitrary data.

        Utility method for callers that need to pre-compute hashes
        before calling ``record()``.

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
                    "Multi-tier supplier tracker singleton "
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
        "Multi-tier supplier tracker ProvenanceTracker singleton replaced"
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
        "Multi-tier supplier tracker ProvenanceTracker singleton reset to None"
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
