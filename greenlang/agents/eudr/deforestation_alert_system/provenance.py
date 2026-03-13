# -*- coding: utf-8 -*-
"""
Provenance Tracking for Deforestation Alert System - AGENT-EUDR-020

Provides SHA-256 based audit trail tracking for all deforestation alert
system operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across satellite change detection, alert
generation, severity classification, spatial buffer monitoring, EUDR
cutoff date verification, historical baseline comparison, alert workflow
management, compliance impact assessment, configuration changes, and
batch processing operations.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every deforestation alert operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 31 five-year audit trail compliance
    - All severity scores, cutoff determinations, and compliance
      decisions are fully traceable from satellite detection through
      calculation to output

Entity Types (12):
    - satellite_detection: Satellite imagery change detection event
    - alert: Deforestation alert generation and lifecycle
    - severity_score: Severity classification and scoring
    - spatial_buffer: Buffer zone creation and monitoring
    - buffer_violation: Buffer zone violation detection
    - cutoff_verification: EUDR cutoff date temporal verification
    - historical_baseline: Historical reference baseline establishment
    - baseline_comparison: Current vs baseline comparison
    - workflow_state: Alert workflow state transitions
    - compliance_impact: Compliance impact assessment
    - config_change: Configuration change audit
    - batch_operation: Batch processing operation tracking

Actions (12):
    - detect: Satellite change detection
    - generate: Alert generation
    - classify: Severity classification
    - monitor: Buffer zone monitoring
    - verify: Cutoff date verification
    - establish: Baseline establishment
    - compare: Baseline comparison
    - triage: Alert triage
    - investigate: Alert investigation
    - resolve: Alert resolution
    - assess: Compliance impact assessment
    - export: Data export or report generation

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("satellite_detection", "detect", "det-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

    >>> entry2 = tracker.record(
    ...     "alert", "generate", "alert-001",
    ...     actor="satellite_change_detector",
    ...     metadata={"detection_id": "det-001", "severity": "HIGH"},
    ... )
    >>> assert len(tracker.get_chain()) == 2

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
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
# Valid entity types and actions (module-level constants for import)
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES: frozenset = frozenset({
    "satellite_detection",
    "alert",
    "severity_score",
    "spatial_buffer",
    "buffer_violation",
    "cutoff_verification",
    "historical_baseline",
    "baseline_comparison",
    "workflow_state",
    "compliance_impact",
    "config_change",
    "batch_operation",
})

VALID_ACTIONS: frozenset = frozenset({
    "detect",
    "generate",
    "classify",
    "monitor",
    "verify",
    "establish",
    "compare",
    "triage",
    "investigate",
    "resolve",
    "assess",
    "export",
})


# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a deforestation alert operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 31 audit trail integrity requirements and deforestation
    alert decision traceability from satellite detection through severity
    classification to compliance impact assessment.

    Attributes:
        entity_type: Type of entity (satellite_detection, alert, etc.).
        action: Action performed (detect, generate, classify, etc.).
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
    """Thread-safe provenance tracker for deforestation alert system operations.

    Maintains an in-memory chain of provenance records with SHA-256
    chain hashing. Each record is linked to the previous record via
    hash chaining, providing tamper-evident audit trail for all
    deforestation alert operations per EUDR Article 31 requirements.

    The tracker supports 12 entity types covering the full lifecycle
    of deforestation monitoring: satellite detection, alert generation,
    severity scoring, spatial buffer management, cutoff verification,
    baseline operations, workflow management, and compliance assessment.

    The tracker is implemented as a thread-safe singleton to ensure
    consistent provenance tracking across all agent operations in a
    single process.

    Attributes:
        _chain: List of provenance records in chronological order.
        _genesis_hash: Genesis hash for the chain (agent identifier).
        _lock: Threading lock for thread-safe operations.
    """

    # Genesis hash for the deforestation alert system agent
    _GENESIS_HASH: str = "GL-EUDR-DAS-020-DEFORESTATION-ALERT-SYSTEM-GENESIS"

    # Supported entity types (references module-level constant)
    _VALID_ENTITY_TYPES: frozenset = VALID_ENTITY_TYPES

    # Supported actions (references module-level constant)
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

        Thread-safe method to add a new provenance record to the chain.
        Each record is linked to the previous record via hash chaining,
        providing tamper-evident audit trail for deforestation alert
        operations.

        Args:
            entity_type: Type of entity (satellite_detection, alert, etc.).
            action: Action performed (detect, generate, classify, etc.).
            entity_id: Unique identifier for the entity.
            actor: User or system identifier (default: "system").
            metadata: Optional additional metadata dictionary.

        Returns:
            ProvenanceRecord with computed hash_value.

        Raises:
            ValueError: If entity_type or action is not valid.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> entry = tracker.record(
            ...     "satellite_detection",
            ...     "detect",
            ...     "det-sentinel2-2025-001",
            ...     actor="satellite_change_detector",
            ...     metadata={"source": "sentinel2", "area_ha": "12.5"},
            ... )
            >>> assert entry.entity_type == "satellite_detection"
            >>> assert entry.hash_value is not None
        """
        # Validate entity_type
        if entity_type not in self._VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of {sorted(self._VALID_ENTITY_TYPES)}"
            )

        # Validate action
        if action not in self._VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Must be one of {sorted(self._VALID_ACTIONS)}"
            )

        with self._lock:
            # Get previous hash (genesis or last record)
            if not self._chain:
                previous_hash = self._genesis_hash
            else:
                previous_hash = self._chain[-1].hash_value

            # Create timestamp
            timestamp = _utcnow().isoformat()

            # Prepare metadata
            meta = metadata or {}

            # Compute hash
            hash_value = self._compute_hash(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                timestamp=timestamp,
                actor=actor,
                metadata=meta,
                previous_hash=previous_hash,
            )

            # Create record
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

            # Append to chain
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
        # Create canonical JSON representation (sorted keys for determinism)
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
        valid. This ensures the chain has not been tampered with.

        Returns:
            True if chain is valid, False otherwise.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> assert tracker.verify_chain() is True
        """
        with self._lock:
            if not self._chain:
                logger.debug("Provenance chain is empty, verification skipped")
                return True

            # Check first record links to genesis
            if self._chain[0].previous_hash != self._genesis_hash:
                logger.error(
                    f"First record previous_hash does not match genesis: "
                    f"{self._chain[0].previous_hash} != {self._genesis_hash}"
                )
                return False

            # Verify each record
            for i, record in enumerate(self._chain):
                # Get expected previous hash
                if i == 0:
                    expected_prev = self._genesis_hash
                else:
                    expected_prev = self._chain[i - 1].hash_value

                # Check previous_hash link
                if record.previous_hash != expected_prev:
                    logger.error(
                        f"Record {i} previous_hash mismatch: "
                        f"{record.previous_hash} != {expected_prev}"
                    )
                    return False

                # Recompute hash
                computed_hash = self._compute_hash(
                    entity_type=record.entity_type,
                    action=record.action,
                    entity_id=record.entity_id,
                    timestamp=record.timestamp,
                    actor=record.actor,
                    metadata=record.metadata,
                    previous_hash=record.previous_hash,
                )

                # Check hash match
                if record.hash_value != computed_hash:
                    logger.error(
                        f"Record {i} hash mismatch: "
                        f"{record.hash_value} != {computed_hash}"
                    )
                    return False

            logger.info(
                f"Provenance chain verified successfully "
                f"({len(self._chain)} records)"
            )
            return True

    def get_chain(self) -> List[ProvenanceRecord]:
        """Get a copy of the entire provenance chain.

        Returns:
            List of ProvenanceRecord objects in chronological order.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> chain = tracker.get_chain()
            >>> assert len(chain) == 1
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

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("alert", "generate", "alert-001")
            >>> tracker.record("alert", "classify", "alert-001")
            >>> records = tracker.get_records_by_entity("alert", "alert-001")
            >>> assert len(records) == 2
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

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> tracker.record("satellite_detection", "detect", "det-002")
            >>> records = tracker.get_records_by_action("detect")
            >>> assert len(records) == 2
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

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("alert", "generate", "alert-001")
            >>> tracker.record("alert", "classify", "alert-002")
            >>> records = tracker.get_records_by_entity_type("alert")
            >>> assert len(records) == 2
        """
        with self._lock:
            return [r for r in self._chain if r.entity_type == entity_type]

    def get_records_by_detection(
        self, detection_id: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records related to a specific satellite detection.

        Searches entity_id and metadata for the detection identifier to
        trace the full lifecycle from detection through alert generation
        to compliance impact assessment.

        Args:
            detection_id: Detection identifier to search for.

        Returns:
            List of ProvenanceRecord objects related to the detection.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> tracker.record("alert", "generate", "alert-001",
            ...                metadata={"detection_id": "det-001"})
            >>> records = tracker.get_records_by_detection("det-001")
            >>> assert len(records) == 2
        """
        with self._lock:
            results = []
            for r in self._chain:
                # Check entity_id matches
                if r.entity_id == detection_id:
                    results.append(r)
                    continue
                # Check metadata for detection_id field
                if r.metadata.get("detection_id") == detection_id:
                    results.append(r)
            return results

    def get_records_by_alert(
        self, alert_id: str
    ) -> List[ProvenanceRecord]:
        """Get all provenance records related to a specific alert.

        Searches entity_id and metadata for the alert identifier to
        trace the full alert lifecycle including triage, investigation,
        resolution, and compliance assessment.

        Args:
            alert_id: Alert identifier to search for.

        Returns:
            List of ProvenanceRecord objects related to the alert.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("alert", "generate", "alert-001")
            >>> tracker.record("workflow_state", "triage", "wf-001",
            ...                metadata={"alert_id": "alert-001"})
            >>> records = tracker.get_records_by_alert("alert-001")
            >>> assert len(records) == 2
        """
        with self._lock:
            results = []
            for r in self._chain:
                if r.entity_id == alert_id:
                    results.append(r)
                    continue
                if r.metadata.get("alert_id") == alert_id:
                    results.append(r)
            return results

    def export_json(self) -> str:
        """Export the entire provenance chain as JSON string.

        Returns:
            JSON string containing all provenance records, suitable
            for external audit system ingestion per EUDR Article 31.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> json_str = tracker.export_json()
            >>> assert "satellite_detection" in json_str
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
                    "agent_id": "GL-EUDR-DAS-020",
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

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("satellite_detection", "detect", "det-001")
            >>> tracker.clear()
            >>> assert len(tracker.get_chain()) == 0
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
    instance. Uses double-checked locking to minimize contention.

    Returns:
        ProvenanceTracker singleton instance.

    Example:
        >>> tracker = get_tracker()
        >>> tracker2 = get_tracker()
        >>> assert tracker is tracker2  # Same instance
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

    Example:
        >>> reset_tracker()
        >>> # Next get_tracker() call will create a new instance
    """
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
        logger.warning("Provenance tracker reset (testing only)")
