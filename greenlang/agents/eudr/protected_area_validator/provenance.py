# -*- coding: utf-8 -*-
"""
Provenance Tracking for Protected Area Validator - AGENT-EUDR-022

Provides SHA-256 based audit trail tracking for all protected area
validation operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across WDPA database management, spatial
overlap detection, buffer zone monitoring, designation validation,
proximity alert generation, compliance tracking, conservation assessment,
and compliance reporting.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every protected area operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 31 five-year audit trail compliance

Entity Types (14):
    - protected_area, overlap, buffer_zone, designation, proximity_alert,
      compliance_record, conservation, report, integration_event,
      batch_job, exemption, version, config_change, audit_log

Actions (14):
    - ingest, detect, analyze, validate, alert, track, assess, generate,
      publish, screen, grant, update, export, verify

Example:
    >>> from greenlang.agents.eudr.protected_area_validator.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("overlap", "detect", "overlap-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES: frozenset = frozenset({
    "protected_area",
    "overlap",
    "buffer_zone",
    "designation",
    "proximity_alert",
    "compliance_record",
    "conservation",
    "report",
    "integration_event",
    "batch_job",
    "exemption",
    "version",
    "config_change",
    "audit_log",
})

VALID_ACTIONS: frozenset = frozenset({
    "ingest",
    "detect",
    "analyze",
    "validate",
    "alert",
    "track",
    "assess",
    "generate",
    "publish",
    "screen",
    "grant",
    "update",
    "export",
    "verify",
})


# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record.

    Frozen (immutable) to guarantee that recorded provenance entries
    cannot be modified after creation, supporting EUDR Article 31
    audit trail integrity requirements.

    Attributes:
        entity_type: Type of entity (protected_area, overlap, etc.).
        action: Action performed (detect, analyze, validate, etc.).
        entity_id: Unique identifier for the entity.
        timestamp: ISO 8601 timestamp of the operation.
        actor: User or system identifier that performed the action.
        metadata: Optional additional metadata dictionary.
        previous_hash: SHA-256 hash of the previous record in the chain.
        hash_value: SHA-256 hash of this record (deterministic).
    """

    entity_type: str
    action: str
    entity_id: str
    timestamp: str
    actor: str
    metadata: Dict[str, Any]
    previous_hash: str
    hash_value: str


# ---------------------------------------------------------------------------
# ProvenanceTracker class
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Thread-safe provenance tracker for protected area validator operations.

    Maintains an in-memory chain of provenance records with SHA-256
    chain hashing. Each record is linked to the previous record via
    hash chaining, providing tamper-evident audit trail for all
    protected area validation operations per EUDR Article 31.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record(
        ...     "overlap", "detect", "overlap-001",
        ...     actor="spatial_overlap_engine",
        ...     metadata={"wdpa_id": 12345, "risk_score": "85.00"},
        ... )
        >>> assert entry.entity_type == "overlap"
    """

    _GENESIS_HASH: str = "GL-EUDR-PAV-022-PROTECTED-AREA-VALIDATOR-GENESIS"
    _VALID_ENTITY_TYPES: frozenset = VALID_ENTITY_TYPES
    _VALID_ACTIONS: frozenset = VALID_ACTIONS

    def __init__(self) -> None:
        """Initialize the provenance tracker with an empty chain."""
        self._chain: List[ProvenanceRecord] = []
        self._genesis_hash: str = self._GENESIS_HASH
        self._lock = threading.Lock()
        logger.info(
            f"ProvenanceTracker initialized with genesis hash: "
            f"{self._genesis_hash[:16]}..."
        )

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """Record a new provenance entry in the chain.

        Args:
            entity_type: Type of entity (protected_area, overlap, etc.).
            action: Action performed (detect, analyze, validate, etc.).
            entity_id: Unique identifier for the entity.
            actor: User or system identifier (default: "system").
            metadata: Optional additional metadata dictionary.

        Returns:
            ProvenanceRecord with computed hash_value.

        Raises:
            ValueError: If entity_type or action is not valid.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> entry = tracker.record("protected_area", "ingest", "pa-12345")
            >>> assert entry.hash_value is not None
        """
        if entity_type not in self._VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of {sorted(self._VALID_ENTITY_TYPES)}"
            )
        if action not in self._VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Must be one of {sorted(self._VALID_ACTIONS)}"
            )

        with self._lock:
            if not self._chain:
                previous_hash = self._genesis_hash
            else:
                previous_hash = self._chain[-1].hash_value

            timestamp = _utcnow().isoformat()
            meta = metadata or {}

            hash_value = self._compute_hash(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                timestamp=timestamp,
                actor=actor,
                metadata=meta,
                previous_hash=previous_hash,
            )

            record = ProvenanceRecord(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                timestamp=timestamp,
                actor=actor,
                metadata=meta,
                previous_hash=previous_hash,
                hash_value=hash_value,
            )

            self._chain.append(record)

            logger.debug(
                f"Provenance recorded: {entity_type}.{action} "
                f"entity_id={entity_id} hash={hash_value[:16]}..."
            )

            return record

    def _compute_hash(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        timestamp: str,
        actor: str,
        metadata: Dict[str, Any],
        previous_hash: str,
    ) -> str:
        """Compute deterministic SHA-256 hash for a provenance record.

        Returns:
            64-character hex SHA-256 hash string.
        """
        data = {
            "entity_type": entity_type,
            "action": action,
            "entity_id": entity_id,
            "timestamp": timestamp,
            "actor": actor,
            "metadata": metadata,
            "previous_hash": previous_hash,
        }
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        hash_bytes = hashlib.sha256(canonical_json.encode("utf-8")).digest()
        return hash_bytes.hex()

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire provenance chain.

        Returns:
            True if chain is valid, False otherwise.
        """
        with self._lock:
            if not self._chain:
                return True

            if self._chain[0].previous_hash != self._genesis_hash:
                logger.error("First record previous_hash does not match genesis")
                return False

            for i, record in enumerate(self._chain):
                expected_prev = (
                    self._genesis_hash if i == 0 else self._chain[i - 1].hash_value
                )
                if record.previous_hash != expected_prev:
                    logger.error(f"Record {i} previous_hash mismatch")
                    return False

                computed_hash = self._compute_hash(
                    entity_type=record.entity_type,
                    action=record.action,
                    entity_id=record.entity_id,
                    timestamp=record.timestamp,
                    actor=record.actor,
                    metadata=record.metadata,
                    previous_hash=record.previous_hash,
                )
                if record.hash_value != computed_hash:
                    logger.error(f"Record {i} hash mismatch")
                    return False

            logger.info(f"Provenance chain verified ({len(self._chain)} records)")
            return True

    def get_chain(self) -> List[ProvenanceRecord]:
        """Get a copy of the entire provenance chain."""
        with self._lock:
            return self._chain.copy()

    def get_records_by_entity(
        self, entity_type: str, entity_id: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific entity."""
        with self._lock:
            return [
                r for r in self._chain
                if r.entity_type == entity_type and r.entity_id == entity_id
            ]

    def get_records_by_action(self, action: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific action."""
        with self._lock:
            return [r for r in self._chain if r.action == action]

    def get_records_by_entity_type(
        self, entity_type: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific entity type."""
        with self._lock:
            return [r for r in self._chain if r.entity_type == entity_type]

    def export_json(self) -> str:
        """Export the entire provenance chain as JSON string."""
        with self._lock:
            records_data = [
                {
                    "entity_type": r.entity_type,
                    "action": r.action,
                    "entity_id": r.entity_id,
                    "timestamp": r.timestamp,
                    "actor": r.actor,
                    "metadata": r.metadata,
                    "previous_hash": r.previous_hash,
                    "hash_value": r.hash_value,
                }
                for r in self._chain
            ]
            return json.dumps(
                {
                    "agent_id": "GL-EUDR-PAV-022",
                    "genesis_hash": self._genesis_hash,
                    "record_count": len(self._chain),
                    "records": records_data,
                },
                indent=2,
            )

    def clear(self) -> None:
        """Clear the provenance chain (for testing only)."""
        with self._lock:
            self._chain.clear()
            logger.warning("Provenance chain cleared (testing only)")


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_global_tracker: Optional[ProvenanceTracker] = None


def get_tracker() -> ProvenanceTracker:
    """Get the global ProvenanceTracker singleton instance."""
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = ProvenanceTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global ProvenanceTracker singleton (for testing only)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
        logger.warning("Provenance tracker reset (testing only)")
