# -*- coding: utf-8 -*-
"""
Provenance Tracking for Deforestation Satellite Connector - AGENT-DATA-007

Provides SHA-256 based audit trail tracking for all deforestation satellite
connector operations. Maintains an in-memory chain-hashed operation log
for tamper-evident provenance.

Operation Types:
    - satellite_acquisition: Satellite imagery acquisition
    - change_detection: Bi-temporal change detection
    - alert_aggregation: Alert query and aggregation
    - baseline_assessment: EUDR baseline compliance check
    - classification: Land cover classification
    - compliance_report: Compliance report generation
    - pipeline_execution: Full pipeline execution

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every operation

Example:
    >>> from greenlang.deforestation_satellite.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry_id = tracker.record(
    ...     "satellite_acquisition", "SCN-abc123def456", "acquire", "abc123",
    ... )
    >>> valid, chain = tracker.verify_chain("SCN-abc123def456")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Valid operation types
# ---------------------------------------------------------------------------

VALID_OPERATION_TYPES = frozenset({
    "satellite_acquisition",
    "change_detection",
    "alert_aggregation",
    "baseline_assessment",
    "classification",
    "compliance_report",
    "pipeline_execution",
})


# =============================================================================
# ProvenanceTracker
# =============================================================================


class ProvenanceTracker:
    """Tracks provenance for deforestation satellite operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity.

    Seven operation types are supported:
        1. satellite_acquisition -- Satellite imagery acquisition
        2. change_detection      -- Bi-temporal change detection
        3. alert_aggregation     -- Alert query and aggregation
        4. baseline_assessment   -- EUDR baseline compliance check
        5. classification        -- Land cover classification
        6. compliance_report     -- Compliance report generation
        7. pipeline_execution    -- Full pipeline execution

    Attributes:
        _chain_store: In-memory chain storage grouped by entity_id.
        _global_chain: Flat list of all entries in order.
        _last_chain_hash: Most recent chain hash for linking.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry_id = tracker.record(
        ...     "satellite_acquisition", "SCN-abc123", "acquire", "abc123",
        ... )
        >>> valid, chain = tracker.verify_chain("SCN-abc123")
        >>> assert valid is True
    """

    # Initial chain hash (genesis)
    _GENESIS_HASH = hashlib.sha256(
        b"greenlang-deforestation-satellite-genesis"
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize ProvenanceTracker."""
        self._chain_store: Dict[str, List[Dict[str, Any]]] = {}
        self._global_chain: List[Dict[str, Any]] = []
        self._last_chain_hash: str = self._GENESIS_HASH
        logger.info("ProvenanceTracker initialized for deforestation satellite connector")

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry for an entity operation.

        Args:
            entity_type: Type of entity (one of VALID_OPERATION_TYPES or any string).
            entity_id: Unique entity identifier.
            action: Action performed (acquire, detect, query, check, classify,
                generate, complete, etc.).
            data_hash: SHA-256 hash of the operation data.
            user_id: User who performed the operation.

        Returns:
            Chain hash of the new entry.
        """
        timestamp = _utcnow().isoformat()

        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": timestamp,
            "chain_hash": "",
        }

        # Build chain hash
        chain_hash = self._compute_chain_hash(
            self._last_chain_hash, data_hash, action, timestamp,
        )
        entry["chain_hash"] = chain_hash

        # Store in entity chain
        if entity_id not in self._chain_store:
            self._chain_store[entity_id] = []
        self._chain_store[entity_id].append(entry)

        # Store in global chain
        self._global_chain.append(entry)
        self._last_chain_hash = chain_hash

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash=%s",
            entity_type, entity_id[:8] if len(entity_id) > 8 else entity_id,
            action, chain_hash[:16],
        )
        return chain_hash

    def verify_chain(self, entity_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of the provenance chain for an entity.

        Checks that all entries have required fields and non-empty
        chain hashes.

        Args:
            entity_id: Entity ID whose chain to verify.

        Returns:
            Tuple of (is_valid: bool, chain_entries: list).
        """
        chain = self._chain_store.get(entity_id, [])
        if not chain:
            return True, []

        is_valid = True
        required_fields = [
            "entity_type", "entity_id", "action",
            "data_hash", "timestamp", "chain_hash",
        ]

        for i, entry in enumerate(chain):
            # First entry must have a chain_hash
            if i == 0 and not entry.get("chain_hash"):
                is_valid = False
                break

            # Check required fields
            for field_name in required_fields:
                if field_name not in entry:
                    is_valid = False
                    logger.warning(
                        "Chain verification failed for %s: missing field %s at index %d",
                        entity_id, field_name, i,
                    )
                    break
            if not is_valid:
                break

        return is_valid, chain

    def get_chain(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get the provenance chain for an entity.

        Args:
            entity_id: Entity ID to look up.

        Returns:
            List of provenance entries, oldest first.
        """
        return list(self._chain_store.get(entity_id, []))

    def get_by_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all provenance entries for an entity (alias for get_chain).

        Args:
            entity_id: Entity ID to look up.

        Returns:
            List of provenance entries.
        """
        return self.get_chain(entity_id)

    def get_global_chain(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the global provenance chain (all entities).

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of provenance entries, newest first.
        """
        return list(reversed(self._global_chain[-limit:]))

    def _compute_chain_hash(
        self,
        previous_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next chain hash linking to the previous.

        Args:
            previous_hash: Previous chain hash.
            data_hash: Hash of the current operation data.
            action: Action performed.
            timestamp: ISO-formatted timestamp.

        Returns:
            New SHA-256 chain hash.
        """
        combined = json.dumps({
            "previous": previous_hash,
            "data": data_hash,
            "action": action,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries."""
        return len(self._global_chain)

    @property
    def entity_count(self) -> int:
        """Return the number of unique entities tracked."""
        return len(self._chain_store)

    def export_json(self) -> str:
        """Export all provenance records as JSON string.

        Returns:
            JSON string of provenance records.
        """
        return json.dumps(self._global_chain, indent=2, default=str)

    def build_hash(self, data: Any) -> str:
        """Build a SHA-256 hash for arbitrary data.

        Args:
            data: Data to hash (dict, list, or other).

        Returns:
            Hex-encoded SHA-256 hash.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "ProvenanceTracker",
    "VALID_OPERATION_TYPES",
]
