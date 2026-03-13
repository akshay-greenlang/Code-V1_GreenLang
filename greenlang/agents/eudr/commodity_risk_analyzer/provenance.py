# -*- coding: utf-8 -*-
"""
Provenance Tracking for Commodity Risk Analyzer - AGENT-EUDR-018

Provides SHA-256 based audit trail tracking for all commodity risk
analysis operations. Maintains an in-memory chain-hashed operation
log for tamper-evident provenance across commodity profiling, derived
product analysis, price volatility monitoring, production forecasting,
substitution risk detection, regulatory compliance mapping, due
diligence workflow management, portfolio risk aggregation, batch
analysis operations, and configuration changes.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every commodity risk analysis operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 31 five-year audit trail compliance
    - All risk scores and classification decisions are fully
      traceable from input data through calculation to output

Entity Types (10):
    - commodity_profile: Commodity risk profiling and scoring
    - derived_product: Derived product analysis and traceability
    - price_volatility: Price volatility monitoring and classification
    - production_forecast: Production yield forecasting
    - substitution_event: Commodity substitution detection
    - regulatory_compliance: Regulatory requirement mapping
    - dd_workflow: Due diligence workflow management
    - portfolio_analysis: Portfolio risk aggregation
    - audit_log: Audit log entry recording
    - config_change: Configuration change audit

Actions (12):
    - profile: Profile commodity risk factors
    - analyze: Analyze derived products or portfolio
    - query: Query price or volatility data
    - forecast: Generate production forecasts
    - detect: Detect substitution events
    - check: Check regulatory compliance
    - initiate: Initiate DD workflow
    - aggregate: Aggregate portfolio risk
    - compare: Compare commodities
    - export: Export data or reports
    - archive: Archive historical records
    - update: Update profiles or configurations

Example:
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("commodity_profile", "profile", "cpf-cocoa01")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
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
    """A single tamper-evident provenance record for a commodity risk analysis operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 31 audit trail integrity requirements and commodity
    risk analysis decision traceability.

    Attributes:
        entity_type: Type of entity (commodity_profile, derived_product, etc.).
        action: Action performed (profile, analyze, query, etc.).
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
    """Thread-safe provenance tracker for commodity risk analyzer operations.

    Maintains an in-memory chain of provenance records with SHA-256
    chain hashing. Each record is linked to the previous record via
    hash chaining, providing tamper-evident audit trail for all
    commodity risk analysis operations per EUDR Article 31 requirements.

    The tracker is implemented as a thread-safe singleton to ensure
    consistent provenance tracking across all agent operations in a
    single process.

    Attributes:
        _chain: List of provenance records in chronological order.
        _genesis_hash: Genesis hash for the chain (agent identifier).
        _lock: Threading lock for thread-safe operations.
    """

    # Genesis hash for the commodity risk analyzer agent
    _GENESIS_HASH: str = "GL-EUDR-CRA-018-COMMODITY-RISK-ANALYZER-GENESIS"

    # Supported entity types
    _VALID_ENTITY_TYPES: frozenset = frozenset({
        "commodity_profile",
        "derived_product",
        "price_volatility",
        "production_forecast",
        "substitution_event",
        "regulatory_compliance",
        "dd_workflow",
        "portfolio_analysis",
        "audit_log",
        "config_change",
    })

    # Supported actions
    _VALID_ACTIONS: frozenset = frozenset({
        "profile",
        "analyze",
        "query",
        "forecast",
        "detect",
        "check",
        "initiate",
        "aggregate",
        "compare",
        "export",
        "archive",
        "update",
    })

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
        providing tamper-evident audit trail.

        Args:
            entity_type: Type of entity (commodity_profile, derived_product, etc.).
            action: Action performed (profile, analyze, query, etc.).
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
            ...     "commodity_profile",
            ...     "profile",
            ...     "cpf-cocoa01",
            ...     actor="user@example.com",
            ...     metadata={"risk_score": 68.5},
            ... )
            >>> assert entry.entity_type == "commodity_profile"
            >>> assert entry.hash_value is not None
        """
        # Validate entity_type
        if entity_type not in self._VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type: {entity_type}. "
                f"Must be one of {self._VALID_ENTITY_TYPES}"
            )

        # Validate action
        if action not in self._VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Must be one of {self._VALID_ACTIONS}"
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
            >>> tracker.record("commodity_profile", "update", "cpf-001")
            >>> records = tracker.get_records_by_entity("commodity_profile", "cpf-001")
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
            >>> tracker.record("derived_product", "profile", "dpd-001")
            >>> records = tracker.get_records_by_action("profile")
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
            >>> tracker.record("commodity_profile", "update", "cpf-002")
            >>> records = tracker.get_records_by_entity_type("commodity_profile")
            >>> assert len(records) == 2
        """
        with self._lock:
            return [r for r in self._chain if r.entity_type == entity_type]

    def export_json(self) -> str:
        """Export the entire provenance chain as JSON string.

        Returns:
            JSON string containing all provenance records.

        Example:
            >>> tracker = ProvenanceTracker()
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
            >>> json_str = tracker.export_json()
            >>> assert "commodity_profile" in json_str
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
            >>> tracker.record("commodity_profile", "profile", "cpf-001")
            >>> tracker.clear()
            >>> assert len(tracker.get_chain()) == 0
        """
        with self._lock:
            self._chain.clear()
            logger.warning("Provenance chain cleared (testing only)")


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_global_tracker: Optional[ProvenanceTracker] = None


def get_tracker() -> ProvenanceTracker:
    """Get the global ProvenanceTracker singleton instance.

    Thread-safe lazy initialization. Subsequent calls return the same instance.

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
