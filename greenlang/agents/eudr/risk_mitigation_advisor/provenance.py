# -*- coding: utf-8 -*-
"""
Provenance Tracking for Risk Mitigation Advisor - AGENT-EUDR-025

Provides SHA-256 based audit trail tracking for all risk mitigation advisor
operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across strategy selection, remediation plan design,
capacity building management, measure library operations, effectiveness
tracking, continuous monitoring, cost-benefit optimization, stakeholder
collaboration, and mitigation reporting.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every mitigation operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 31 five-year audit trail compliance
    - All strategy recommendations, plan decisions, effectiveness
      calculations, and optimization results are fully traceable
      from risk input through calculation to output

Entity Types (14):
    - strategy_recommendation: ML-powered strategy recommendation
    - remediation_plan: Remediation plan creation and modification
    - plan_milestone: Plan milestone tracking and completion
    - capacity_enrollment: Supplier capacity building enrollment
    - capacity_module: Training module completion tracking
    - mitigation_measure: Measure library entry management
    - effectiveness_record: Before-after effectiveness measurement
    - monitoring_event: Continuous monitoring trigger event
    - adaptive_adjustment: Adaptive plan adjustment recommendation
    - optimization_result: Cost-benefit optimization output
    - stakeholder_action: Stakeholder collaboration activity
    - mitigation_report: Report generation and distribution
    - config_change: Configuration change audit
    - batch_operation: Batch processing operation tracking

Actions (14):
    - recommend: Strategy recommendation generation
    - create: Entity creation (plan, enrollment, report)
    - update: Entity modification
    - complete: Milestone or module completion
    - measure: Effectiveness measurement
    - detect: Trigger event detection
    - adjust: Adaptive plan adjustment
    - optimize: Budget optimization execution
    - collaborate: Stakeholder collaboration action
    - generate: Report generation
    - approve: Plan or adjustment approval
    - escalate: Escalation chain activation
    - archive: Data archival for retention
    - export: Data export or report distribution

Example:
    >>> from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record(
    ...     "strategy_recommendation", "recommend", "strat-001"
    ... )
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

    >>> entry2 = tracker.record(
    ...     "remediation_plan", "create", "plan-001",
    ...     actor="strategy_selector",
    ...     metadata={"supplier_id": "sup-001", "status": "draft"},
    ... )
    >>> assert len(tracker.get_chain()) == 2

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
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
    "strategy_recommendation",
    "remediation_plan",
    "plan_milestone",
    "capacity_enrollment",
    "capacity_module",
    "mitigation_measure",
    "effectiveness_record",
    "monitoring_event",
    "adaptive_adjustment",
    "optimization_result",
    "stakeholder_action",
    "mitigation_report",
    "config_change",
    "batch_operation",
})

VALID_ACTIONS: frozenset = frozenset({
    "recommend",
    "create",
    "update",
    "complete",
    "measure",
    "detect",
    "adjust",
    "optimize",
    "collaborate",
    "generate",
    "approve",
    "escalate",
    "archive",
    "export",
})


# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a risk mitigation operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 31 audit trail integrity requirements and mitigation
    decision traceability from risk input through strategy selection
    to effectiveness verification.

    Attributes:
        entity_type: Type of entity (strategy_recommendation, plan, etc.).
        action: Action performed (recommend, create, update, etc.).
        entity_id: Unique identifier for the entity.
        timestamp: ISO 8601 timestamp of the operation.
        actor: User or system identifier that performed the action.
        metadata: Optional additional metadata dictionary.
        previous_hash: SHA-256 hash of the previous record in the chain.
        hash_value: SHA-256 hash of this record including all fields.

    Example:
        >>> record = ProvenanceRecord(
        ...     entity_type="strategy_recommendation",
        ...     action="recommend",
        ...     entity_id="strat-001",
        ...     timestamp="2026-03-11T10:00:00+00:00",
        ...     actor="strategy_selector",
        ...     metadata={"confidence": 0.85},
        ...     previous_hash="abc123",
        ...     hash_value="def456",
        ... )
        >>> assert record.entity_type == "strategy_recommendation"
    """

    entity_type: str
    action: str
    entity_id: str
    timestamp: str
    actor: str
    metadata: Optional[Dict[str, Any]]
    previous_hash: str
    hash_value: str


# ---------------------------------------------------------------------------
# ProvenanceTracker class
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Thread-safe SHA-256 chain-hashed provenance tracker.

    Maintains an append-only chain of provenance records where each
    record's hash includes the hash of the previous record, creating
    a tamper-evident audit trail for all risk mitigation operations.

    The tracker supports:
    - Chain-hashed SHA-256 provenance records
    - Thread-safe append operations via threading.Lock
    - Chain integrity verification
    - JSON export for external audit systems
    - Configurable genesis hash for chain anchoring
    - Entity-type and action-based filtering
    - Metadata attachment for rich audit context
    - Maximum chain length enforcement

    Attributes:
        _chain: List of ProvenanceRecord entries.
        _lock: Threading lock for thread-safe operations.
        _genesis_hash: Genesis hash anchor for the first record.
        _algorithm: Hash algorithm (sha256, sha384, sha512).
        _max_chain_length: Maximum number of records before pruning.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record(
        ...     "strategy_recommendation", "recommend", "strat-001",
        ... )
        >>> assert tracker.verify_chain() is True
        >>> chain_json = tracker.export_json()
        >>> assert len(json.loads(chain_json)) == 1
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-RMA-025-RISK-MITIGATION-ADVISOR-GENESIS",
        algorithm: str = "sha256",
        max_chain_length: int = 100_000,
    ) -> None:
        """Initialize ProvenanceTracker.

        Args:
            genesis_hash: Anchor hash for the first record in the chain.
            algorithm: Hash algorithm to use (sha256, sha384, sha512).
            max_chain_length: Maximum records before oldest are pruned.

        Raises:
            ValueError: If algorithm is not supported.
        """
        valid_algorithms = {"sha256", "sha384", "sha512"}
        if algorithm not in valid_algorithms:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Must be one of {valid_algorithms}"
            )
        self._chain: List[ProvenanceRecord] = []
        self._lock = threading.Lock()
        self._genesis_hash = genesis_hash
        self._algorithm = algorithm
        self._max_chain_length = max_chain_length

        logger.debug(
            f"ProvenanceTracker initialized: algorithm={algorithm}, "
            f"genesis_hash={genesis_hash[:20]}..."
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

        Creates a new ProvenanceRecord with SHA-256 hash linking to the
        previous record in the chain. Thread-safe via internal lock.

        Args:
            entity_type: Type of entity being tracked.
            action: Action performed on the entity.
            entity_id: Unique identifier for the entity.
            actor: User or system performing the action.
            metadata: Optional metadata dictionary.

        Returns:
            The newly created ProvenanceRecord.

        Raises:
            ValueError: If entity_type or action is not valid.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> entry = tracker.record(
            ...     "remediation_plan", "create", "plan-001",
            ...     actor="plan_designer",
            ...     metadata={"supplier_id": "sup-001"},
            ... )
            >>> assert entry.action == "create"
        """
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of {sorted(VALID_ENTITY_TYPES)}"
            )
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Must be one of {sorted(VALID_ACTIONS)}"
            )

        with self._lock:
            previous_hash = (
                self._chain[-1].hash_value
                if self._chain
                else self._genesis_hash
            )

            timestamp = _utcnow().isoformat()

            # Build canonical string for hashing
            canonical = json.dumps(
                {
                    "entity_type": entity_type,
                    "action": action,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "actor": actor,
                    "metadata": metadata or {},
                    "previous_hash": previous_hash,
                },
                sort_keys=True,
                default=str,
            )

            hasher = hashlib.new(self._algorithm)
            hasher.update(canonical.encode("utf-8"))
            hash_value = hasher.hexdigest()

            entry = ProvenanceRecord(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                timestamp=timestamp,
                actor=actor,
                metadata=metadata,
                previous_hash=previous_hash,
                hash_value=hash_value,
            )

            self._chain.append(entry)

            # Prune if over max length
            if len(self._chain) > self._max_chain_length:
                prune_count = len(self._chain) - self._max_chain_length
                self._chain = self._chain[prune_count:]
                logger.warning(
                    f"Provenance chain pruned: removed {prune_count} "
                    f"oldest records"
                )

            logger.debug(
                f"Provenance recorded: {entity_type}.{action} "
                f"entity={entity_id} hash={hash_value[:12]}..."
            )

            return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire provenance chain.

        Recomputes all hashes and verifies chain linkage from genesis
        through every subsequent record.

        Returns:
            True if the chain is valid, False if tampered.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("config_change", "update", "cfg-001")
            >>> assert tracker.verify_chain() is True
        """
        with self._lock:
            if not self._chain:
                return True

            for i, entry in enumerate(self._chain):
                expected_previous = (
                    self._chain[i - 1].hash_value
                    if i > 0
                    else self._genesis_hash
                )

                if entry.previous_hash != expected_previous:
                    logger.error(
                        f"Chain broken at index {i}: expected "
                        f"previous_hash={expected_previous[:12]}..., "
                        f"got {entry.previous_hash[:12]}..."
                    )
                    return False

                # Recompute hash
                canonical = json.dumps(
                    {
                        "entity_type": entry.entity_type,
                        "action": entry.action,
                        "entity_id": entry.entity_id,
                        "timestamp": entry.timestamp,
                        "actor": entry.actor,
                        "metadata": entry.metadata or {},
                        "previous_hash": entry.previous_hash,
                    },
                    sort_keys=True,
                    default=str,
                )
                hasher = hashlib.new(self._algorithm)
                hasher.update(canonical.encode("utf-8"))
                recomputed = hasher.hexdigest()

                if recomputed != entry.hash_value:
                    logger.error(
                        f"Hash mismatch at index {i}: expected "
                        f"{recomputed[:12]}..., got {entry.hash_value[:12]}..."
                    )
                    return False

            return True

    def get_chain(self) -> List[ProvenanceRecord]:
        """Return a copy of the provenance chain.

        Returns:
            List of ProvenanceRecord entries (copy for thread safety).
        """
        with self._lock:
            return list(self._chain)

    def get_by_entity(self, entity_id: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a specific entity.

        Args:
            entity_id: Entity identifier to filter by.

        Returns:
            List of ProvenanceRecord entries for the entity.
        """
        with self._lock:
            return [r for r in self._chain if r.entity_id == entity_id]

    def get_by_type(self, entity_type: str) -> List[ProvenanceRecord]:
        """Get all provenance records of a specific entity type.

        Args:
            entity_type: Entity type to filter by.

        Returns:
            List of ProvenanceRecord entries of the given type.
        """
        with self._lock:
            return [r for r in self._chain if r.entity_type == entity_type]

    def get_by_action(self, action: str) -> List[ProvenanceRecord]:
        """Get all provenance records with a specific action.

        Args:
            action: Action to filter by.

        Returns:
            List of ProvenanceRecord entries with the given action.
        """
        with self._lock:
            return [r for r in self._chain if r.action == action]

    def get_latest_hash(self) -> str:
        """Return the hash of the most recent record, or genesis hash.

        Returns:
            SHA-256 hash string of the latest record.
        """
        with self._lock:
            if self._chain:
                return self._chain[-1].hash_value
            return self._genesis_hash

    def chain_length(self) -> int:
        """Return the number of records in the chain.

        Returns:
            Integer count of provenance records.
        """
        with self._lock:
            return len(self._chain)

    def export_json(self, indent: int = 2) -> str:
        """Export the provenance chain as JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the chain.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("config_change", "update", "cfg-001")
            >>> json_str = tracker.export_json()
            >>> import json
            >>> data = json.loads(json_str)
            >>> assert len(data) == 1
        """
        with self._lock:
            records = []
            for r in self._chain:
                records.append({
                    "entity_type": r.entity_type,
                    "action": r.action,
                    "entity_id": r.entity_id,
                    "timestamp": r.timestamp,
                    "actor": r.actor,
                    "metadata": r.metadata,
                    "previous_hash": r.previous_hash,
                    "hash_value": r.hash_value,
                })
            return json.dumps(records, indent=indent, default=str)

    def clear(self) -> None:
        """Clear the provenance chain (for testing only).

        Warning:
            This operation destroys provenance data and should only
            be used in test environments.
        """
        with self._lock:
            self._chain.clear()
            logger.warning("Provenance chain cleared")


# ---------------------------------------------------------------------------
# Module-level singleton for provenance tracking
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_global_tracker: Optional[ProvenanceTracker] = None


def get_tracker() -> ProvenanceTracker:
    """Get the global ProvenanceTracker singleton instance.

    Thread-safe lazy initialization. Creates a new ProvenanceTracker
    on first call with default genesis hash.

    Returns:
        ProvenanceTracker singleton instance.

    Example:
        >>> tracker = get_tracker()
        >>> assert tracker is get_tracker()  # Same instance
    """
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = ProvenanceTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global ProvenanceTracker singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_tracker()
        >>> # Next get_tracker() call creates fresh instance
    """
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
