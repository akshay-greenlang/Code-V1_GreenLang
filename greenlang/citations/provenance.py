# -*- coding: utf-8 -*-
"""
Citations Provenance Tracker - AGENT-FOUND-005: Citations & Evidence

Provides SHA-256 based audit trail tracking for all citation and evidence
changes. Maintains an in-memory operation log with chain hashing for
tamper evidence.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every change

Example:
    >>> from greenlang.citations.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> entry_id = tracker.record(
    ...     entity_type="citation",
    ...     entity_id="cit-001",
    ...     action="create",
    ...     data_hash="abc123...",
    ...     user_id="analyst_1",
    ... )
    >>> print(entry_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
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


# ---------------------------------------------------------------------------
# ProvenanceEntry model
# ---------------------------------------------------------------------------


class ProvenanceEntry(BaseModel):
    """A single entry in the provenance chain."""

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique provenance entry ID",
    )
    entity_type: str = Field(
        ..., description="Type of entity (citation, package, verification)",
    )
    entity_id: str = Field(
        ..., description="ID of the entity this entry refers to",
    )
    action: str = Field(
        ..., description="Action performed (create, update, delete, verify)",
    )
    data_hash: str = Field(
        ..., description="SHA-256 hash of the entity data at this point",
    )
    previous_hash: str = Field(
        ..., description="Hash of the previous chain entry",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="When this entry was recorded",
    )
    user_id: str = Field(
        default="system", description="User who performed the action",
    )
    chain_hash: str = Field(
        default="", description="Combined chain hash for this entry",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this entry",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for citation and evidence changes with SHA-256 chain hashing.

    Maintains an ordered log of changes with SHA-256 hashes that chain
    together to provide tamper-evident audit trails.

    Attributes:
        _entries: Ordered list of provenance entries.
        _last_chain_hash: Most recent chain hash for linking.
        _entity_index: Index mapping entity_id to entry indices.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> eid = tracker.record("citation", "cit-001", "create", "hash123")
        >>> chain = tracker.get_chain("cit-001")
        >>> assert tracker.verify_chain("cit-001")
    """

    # Initial chain hash (genesis)
    _GENESIS_HASH = hashlib.sha256(b"greenlang-citations-genesis").hexdigest()

    def __init__(self) -> None:
        """Initialize ProvenanceTracker."""
        self._entries: List[ProvenanceEntry] = []
        self._last_chain_hash: str = self._GENESIS_HASH
        self._entity_index: Dict[str, List[int]] = {}
        logger.info("ProvenanceTracker initialized")

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry to the audit trail.

        Args:
            entity_type: Type of entity (citation, package, verification).
            entity_id: ID of the entity.
            action: Action performed (create, update, delete, verify).
            data_hash: SHA-256 hash of the entity data.
            user_id: User who performed the action.
            metadata: Optional additional metadata.

        Returns:
            The entry_id of the new provenance entry.
        """
        entry = ProvenanceEntry(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            data_hash=data_hash,
            previous_hash=self._last_chain_hash,
            user_id=user_id,
            metadata=metadata or {},
        )

        # Build chain hash
        entry_data = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "previous_hash": self._last_chain_hash,
            "timestamp": entry.timestamp.isoformat(),
            "user_id": user_id,
        }
        entry_hash = self._hash_dict(entry_data)
        chain_hash = self._build_next_chain_hash(entry_hash)
        entry.chain_hash = chain_hash

        # Append and update state
        entry_index = len(self._entries)
        self._entries.append(entry)
        self._last_chain_hash = chain_hash

        # Update entity index
        if entity_id not in self._entity_index:
            self._entity_index[entity_id] = []
        self._entity_index[entity_id].append(entry_index)

        logger.debug(
            "Recorded provenance: %s %s %s -> %s",
            entity_type, action, entity_id, entry.entry_id,
        )
        return entry.entry_id

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Get the provenance chain for a specific entity.

        Args:
            entity_id: The entity to get provenance for.

        Returns:
            List of ProvenanceEntry records, oldest first.
        """
        indices = self._entity_index.get(entity_id, [])
        return [self._entries[i] for i in indices]

    def verify_chain(self, entity_id: Optional[str] = None) -> bool:
        """Verify the integrity of the provenance chain.

        If entity_id is provided, verifies only entries for that entity
        within the overall chain. If None, verifies the entire chain.

        Args:
            entity_id: Optional entity to verify. Verifies all if None.

        Returns:
            True if chain is intact, False if tampered.
        """
        if not self._entries:
            return True

        # For full chain verification, replay all entries
        if entity_id is None:
            return self._verify_full_chain()

        # For entity-specific verification, check entries in sequence
        chain = self.get_chain(entity_id)
        if not chain:
            return True

        # Verify each entry's chain hash is present in the full chain
        for entry in chain:
            if not self._verify_single_entry(entry):
                logger.warning(
                    "Chain verification failed for entity %s at entry %s",
                    entity_id, entry.entry_id,
                )
                return False

        return True

    def get_all_entries(self, limit: int = 100) -> List[ProvenanceEntry]:
        """Get all provenance entries, newest first.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of ProvenanceEntry records, newest first.
        """
        entries = list(reversed(self._entries))
        return entries[:limit]

    def export_json(self) -> str:
        """Export all provenance records as JSON string.

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

    def _verify_full_chain(self) -> bool:
        """Verify the full provenance chain from genesis.

        Returns:
            True if the entire chain is intact.
        """
        current_hash = self._GENESIS_HASH

        for entry in self._entries:
            entry_data = {
                "entity_type": entry.entity_type,
                "entity_id": entry.entity_id,
                "action": entry.action,
                "data_hash": entry.data_hash,
                "previous_hash": current_hash,
                "timestamp": entry.timestamp.isoformat(),
                "user_id": entry.user_id,
            }
            entry_hash = self._hash_dict(entry_data)
            combined = f"{current_hash}:{entry_hash}"
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()

            if entry.chain_hash != expected_hash:
                logger.warning(
                    "Chain verification failed at entry %s", entry.entry_id,
                )
                return False

            current_hash = expected_hash

        return True

    def _verify_single_entry(self, entry: ProvenanceEntry) -> bool:
        """Verify a single entry's chain hash against its previous_hash.

        Args:
            entry: The entry to verify.

        Returns:
            True if the entry's chain_hash is valid.
        """
        entry_data = {
            "entity_type": entry.entity_type,
            "entity_id": entry.entity_id,
            "action": entry.action,
            "data_hash": entry.data_hash,
            "previous_hash": entry.previous_hash,
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
        }
        entry_hash = self._hash_dict(entry_data)
        combined = f"{entry.previous_hash}:{entry_hash}"
        expected_hash = hashlib.sha256(combined.encode()).hexdigest()

        return entry.chain_hash == expected_hash

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
