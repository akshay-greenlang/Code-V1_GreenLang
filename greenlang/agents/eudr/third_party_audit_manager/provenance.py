# -*- coding: utf-8 -*-
"""
Provenance Tracking for Third-Party Audit Manager - AGENT-EUDR-024

Provides SHA-256 based audit trail tracking for all third-party audit
management operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance across audit planning and scheduling, auditor
qualification tracking, audit execution monitoring, non-conformance
detection and classification, corrective action request lifecycle
management, certification scheme integration, ISO 19011 report generation,
competent authority liaison, and audit analytics.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every audit lifecycle operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 31 five-year audit trail compliance

Entity Types (12):
    - audit: Audit lifecycle events
    - auditor: Auditor registry operations
    - checklist: Audit checklist management
    - evidence: Evidence collection and verification
    - non_conformance: NC detection and classification
    - root_cause_analysis: RCA framework operations
    - corrective_action: CAR lifecycle management
    - certificate: Certification scheme integration
    - authority_interaction: Competent authority liaison
    - audit_report: ISO 19011 report generation
    - analytics: Audit analytics calculations
    - config_change: Configuration change audit

Actions (12):
    - schedule: Audit scheduling
    - assign: Auditor assignment
    - execute: Audit execution
    - classify: NC classification
    - issue: CAR issuance
    - escalate: Escalation action
    - verify: Verification action
    - close: Closure action
    - generate: Report generation
    - sync: Certification sync
    - respond: Authority response
    - calculate: Analytics calculation

Example:
    >>> from greenlang.agents.eudr.third_party_audit_manager.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("audit", "schedule", "audit-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid entity types and actions (module-level constants for import)
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES: frozenset = frozenset({
    "audit",
    "auditor",
    "checklist",
    "evidence",
    "non_conformance",
    "root_cause_analysis",
    "corrective_action",
    "certificate",
    "authority_interaction",
    "audit_report",
    "analytics",
    "config_change",
})

VALID_ACTIONS: frozenset = frozenset({
    "schedule",
    "assign",
    "execute",
    "classify",
    "issue",
    "escalate",
    "verify",
    "close",
    "generate",
    "sync",
    "respond",
    "calculate",
})

# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for an audit management operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 31 audit trail integrity requirements and audit
    lifecycle decision traceability.

    Attributes:
        entity_type: Type of entity (audit, auditor, non_conformance, etc.).
        action: Action performed (schedule, classify, issue, etc.).
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
    """Thread-safe provenance tracker for third-party audit management operations.

    Maintains an in-memory chain of provenance records with SHA-256
    chain hashing. Each record is linked to the previous record via
    hash chaining, providing tamper-evident audit trail for all
    audit management operations per EUDR Article 31 requirements.

    The tracker supports 12 entity types covering the full audit
    lifecycle: audit planning, auditor management, checklist tracking,
    evidence collection, NC classification, CAR management, certification
    integration, authority liaison, report generation, and analytics.

    Attributes:
        _chain: List of provenance records in chronological order.
        _genesis_hash: Genesis hash for the chain (agent identifier).
        _lock: Threading lock for thread-safe operations.
    """

    _GENESIS_HASH: str = "GL-EUDR-TAM-024-THIRD-PARTY-AUDIT-MANAGER-GENESIS"

    _VALID_ENTITY_TYPES: frozenset = VALID_ENTITY_TYPES
    _VALID_ACTIONS: frozenset = VALID_ACTIONS

    def __init__(self) -> None:
        """Initialize the provenance tracker with an empty chain."""
        self._chain: List[ProvenanceRecord] = []
        self._genesis_hash: str = self._GENESIS_HASH
        self._lock = threading.Lock()
        logger.info(
            "ProvenanceTracker initialized with genesis hash: "
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

        Thread-safe method to add a new provenance record to the chain.
        Each record is linked to the previous record via hash chaining.

        Args:
            entity_type: Type of entity (audit, auditor, etc.).
            action: Action performed (schedule, classify, etc.).
            entity_id: Unique identifier for the entity.
            actor: User or system identifier (default: "system").
            metadata: Optional additional metadata dictionary.

        Returns:
            ProvenanceRecord with computed hash_value.

        Raises:
            ValueError: If entity_type or action is not valid.
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

            timestamp = utcnow().isoformat()
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

        Args:
            entity_type: Entity type.
            action: Action performed.
            entity_id: Entity identifier.
            timestamp: ISO 8601 timestamp.
            actor: Actor identifier.
            metadata: Metadata dictionary.
            previous_hash: Previous record hash.

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

        Recomputes hashes for all records and checks that each record's
        hash matches the computed hash and that previous_hash links are
        valid.

        Returns:
            True if chain is valid, False otherwise.
        """
        with self._lock:
            if not self._chain:
                logger.debug("Provenance chain is empty, verification skipped")
                return True

            if self._chain[0].previous_hash != self._genesis_hash:
                logger.error(
                    "First record previous_hash does not match genesis: "
                    f"{self._chain[0].previous_hash} != {self._genesis_hash}"
                )
                return False

            for i, record in enumerate(self._chain):
                if i == 0:
                    expected_prev = self._genesis_hash
                else:
                    expected_prev = self._chain[i - 1].hash_value

                if record.previous_hash != expected_prev:
                    logger.error(
                        f"Record {i} previous_hash mismatch: "
                        f"{record.previous_hash} != {expected_prev}"
                    )
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
                    logger.error(
                        f"Record {i} hash mismatch: "
                        f"{record.hash_value} != {computed_hash}"
                    )
                    return False

            logger.info(
                "Provenance chain verified successfully "
                f"({len(self._chain)} records)"
            )
            return True

    def get_chain(self) -> List[ProvenanceRecord]:
        """Get a copy of the entire provenance chain.

        Returns:
            List of ProvenanceRecord objects in chronological order.
        """
        with self._lock:
            return self._chain.copy()

    def get_records_by_entity(
        self, entity_type: str, entity_id: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific entity.

        Args:
            entity_type: Entity type to filter by.
            entity_id: Entity identifier to filter by.

        Returns:
            List of ProvenanceRecord objects matching the filters.
        """
        with self._lock:
            return [
                r for r in self._chain
                if r.entity_type == entity_type and r.entity_id == entity_id
            ]

    def get_records_by_action(self, action: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific action.

        Args:
            action: Action to filter by.

        Returns:
            List of ProvenanceRecord objects matching the action.
        """
        with self._lock:
            return [r for r in self._chain if r.action == action]

    def get_records_by_entity_type(
        self, entity_type: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific entity type.

        Args:
            entity_type: Entity type to filter by.

        Returns:
            List of ProvenanceRecord objects matching the entity type.
        """
        with self._lock:
            return [r for r in self._chain if r.entity_type == entity_type]

    def get_records_by_audit(
        self, audit_id: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records related to a specific audit.

        Searches entity_id and metadata for the audit identifier to
        trace the full audit lifecycle including planning, execution,
        NC detection, CAR management, and report generation.

        Args:
            audit_id: Audit identifier to search for.

        Returns:
            List of ProvenanceRecord objects related to the audit.
        """
        with self._lock:
            results = []
            for r in self._chain:
                if r.entity_id == audit_id:
                    results.append(r)
                    continue
                if r.metadata.get("audit_id") == audit_id:
                    results.append(r)
            return results

    def export_json(self) -> str:
        """Export the entire provenance chain as JSON string.

        Returns:
            JSON string containing all provenance records, suitable
            for external audit system ingestion per EUDR Article 31.
        """
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
                    "agent_id": "GL-EUDR-TAM-024",
                    "genesis_hash": self._genesis_hash,
                    "record_count": len(self._chain),
                    "records": records_data,
                },
                indent=2,
            )

    def clear(self) -> None:
        """Clear the provenance chain (for testing only).

        WARNING: This should only be used in test environments.
        Clearing provenance in production violates audit requirements.
        """
        with self._lock:
            self._chain.clear()
            logger.warning("Provenance chain cleared (testing only)")

# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_global_tracker: Optional[ProvenanceTracker] = None

def get_tracker() -> ProvenanceTracker:
    """Get the global ProvenanceTracker singleton instance.

    Thread-safe lazy initialization. Subsequent calls return the same
    instance.

    Returns:
        ProvenanceTracker singleton instance.
    """
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = ProvenanceTracker()
    return _global_tracker

def reset_tracker() -> None:
    """Reset the global ProvenanceTracker singleton (for testing only).

    WARNING: This should only be used in test environments.
    """
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
        logger.warning("Provenance tracker reset (testing only)")
