# -*- coding: utf-8 -*-
"""
Provenance Tracker - AGENT-FOUND-007: Agent Registry & Service Catalog

SHA-256 based audit trail tracking for all registry mutations.
Maintains an in-memory chain-hashed log for tamper-evident provenance.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links entries in sequence
    - Verification recomputes the full chain
    - JSON export for external audit systems

Example:
    >>> from greenlang.agent_registry.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry_id = tracker.record(
    ...     entity_type="agent",
    ...     entity_id="GL-MRV-X-001",
    ...     action="register",
    ...     data_hash="abc123...",
    ...     user_id="admin",
    ... )
    >>> chain = tracker.get_chain("GL-MRV-X-001")
    >>> assert tracker.verify_chain("GL-MRV-X-001")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# ProvenanceEntry model
# ---------------------------------------------------------------------------


class ProvenanceEntry(BaseModel):
    """A single provenance chain entry.

    Each entry records a mutation to an entity and is linked to the
    previous entry via a chain hash for tamper evidence.

    Attributes:
        entry_id: Unique entry identifier.
        entity_type: Type of entity (agent, capability, health, etc.).
        entity_id: Identifier of the affected entity.
        action: Action performed (register, unregister, update, reload, health_update).
        data_hash: SHA-256 hash of the entity data at this point.
        user_id: User who performed the action.
        timestamp: When the action occurred.
        chain_hash: SHA-256 hash linking to the previous entry.
        details: Additional context about the action.
    """

    entry_id: str = Field(
        default_factory=_new_uuid, description="Provenance entry ID",
    )
    entity_type: str = Field(
        ..., description="Entity type (agent, capability, health)",
    )
    entity_id: str = Field(
        ..., description="Identifier of the affected entity",
    )
    action: str = Field(
        ..., description="Action performed",
    )
    data_hash: str = Field(
        ..., description="SHA-256 hash of entity data at this point",
    )
    user_id: str = Field(
        default="system", description="User who performed the action",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Action timestamp",
    )
    chain_hash: str = Field(
        default="", description="Chain hash linking to previous entry",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context",
    )


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for agent registry entities with SHA-256 chain hashing.

    Maintains an ordered log of entity mutations with SHA-256 hashes
    that chain together to provide tamper-evident audit trails.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> eid = tracker.record("agent", "GL-MRV-X-001", "register", "hash...", "admin")
        >>> assert tracker.verify_chain("GL-MRV-X-001")
    """

    _GENESIS_HASH = hashlib.sha256(
        b"greenlang-agent-registry-genesis",
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize the ProvenanceTracker."""
        self._entries: List[ProvenanceEntry] = []
        self._last_chain_hash: str = self._GENESIS_HASH
        self._entity_index: Dict[str, List[int]] = {}
        logger.info("ProvenanceTracker initialized (agent-registry)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry for an entity mutation.

        Builds a chain hash linking this entry to the previous one,
        then appends to the log and updates the entity index.

        Args:
            entity_type: Type of entity (agent, capability, health).
            entity_id: Identifier of the affected entity.
            action: Action performed (register, unregister, update, etc.).
            data_hash: SHA-256 hash of the entity data.
            user_id: User who performed the action.
            details: Additional context.

        Returns:
            The chain_hash of the new provenance entry.
        """
        entry = ProvenanceEntry(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            data_hash=data_hash,
            user_id=user_id,
            details=details or {},
        )

        # Build chain hash
        entry_data = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": entry.timestamp.isoformat(),
        }
        entry_hash = self._hash_dict(entry_data)
        chain_hash = self._build_next_chain_hash(entry_hash)
        entry.chain_hash = chain_hash

        # Append and index
        idx = len(self._entries)
        self._entries.append(entry)
        self._last_chain_hash = chain_hash

        indices = self._entity_index.setdefault(entity_id, [])
        indices.append(idx)

        logger.debug(
            "Recorded provenance: %s %s %s (%s)",
            action, entity_type, entity_id, entry.entry_id,
        )
        return chain_hash

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Get the provenance chain for a specific entity.

        Args:
            entity_id: The entity to retrieve the chain for.

        Returns:
            List of ProvenanceEntry records in chronological order.
        """
        indices = self._entity_index.get(entity_id, [])
        return [self._entries[i] for i in indices]

    def verify_chain(self, entity_id: str) -> bool:
        """Verify the integrity of the provenance chain for an entity.

        Recomputes chain hashes from genesis and verifies they match
        the stored hashes.

        Args:
            entity_id: The entity to verify.

        Returns:
            True if chain is intact, False if tampered.
        """
        chain = self.get_chain(entity_id)
        if not chain:
            return True

        indices = self._entity_index.get(entity_id, [])
        if not indices:
            return True

        # Verify by replaying the global chain
        current_hash = self._GENESIS_HASH
        entity_entry_set = set(indices)

        for global_idx, entry in enumerate(self._entries):
            entry_data = {
                "entity_type": entry.entity_type,
                "entity_id": entry.entity_id,
                "action": entry.action,
                "data_hash": entry.data_hash,
                "user_id": entry.user_id,
                "timestamp": entry.timestamp.isoformat(),
            }
            entry_hash = self._hash_dict(entry_data)
            combined = f"{current_hash}:{entry_hash}"
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()

            if global_idx in entity_entry_set:
                if entry.chain_hash != expected_hash:
                    logger.warning(
                        "Chain verification failed at entry %s (index %d)",
                        entry.entry_id, global_idx,
                    )
                    return False

            current_hash = expected_hash

        return True

    def get_all_entries(self) -> List[ProvenanceEntry]:
        """Get all provenance entries in chronological order.

        Returns:
            List of all ProvenanceEntry records.
        """
        return list(self._entries)

    def export_json(self) -> str:
        """Export all provenance records as a JSON string.

        Returns:
            JSON string of provenance records.
        """
        records = [entry.model_dump(mode="json") for entry in self._entries]
        return json.dumps(records, indent=2, default=str)

    @property
    def entry_count(self) -> int:
        """Return the number of provenance entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_next_chain_hash(self, entry_hash: str) -> str:
        """Build the next chain hash linking to the previous.

        Args:
            entry_hash: Hash of the current entry data.

        Returns:
            New chain hash incorporating the previous.
        """
        combined = f"{self._last_chain_hash}:{entry_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary.

        Args:
            data: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


__all__ = [
    "ProvenanceEntry",
    "ProvenanceTracker",
]
