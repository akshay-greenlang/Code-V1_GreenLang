# -*- coding: utf-8 -*-
"""
Assumptions Provenance Tracker - AGENT-FOUND-004: Assumptions Registry

Provides SHA-256 based audit trail tracking for all assumption changes.
Maintains an in-memory operation log with chain hashing for tamper evidence.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every change

Example:
    >>> from greenlang.assumptions.provenance import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> log_id = tracker.record_change(
    ...     user_id="analyst_1",
    ...     change_type="create",
    ...     assumption_id="ef.electricity",
    ...     old_value=None,
    ...     new_value=0.42,
    ...     reason="Initial creation",
    ... )
    >>> print(log_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.assumptions.models import ChangeLogEntry, ChangeType

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class ProvenanceTracker:
    """Tracks provenance for assumption changes with SHA-256 chain hashing.

    Maintains an ordered log of changes with SHA-256 hashes that chain
    together to provide tamper-evident audit trails.

    Attributes:
        _entries: Ordered list of change log entries.
        _last_chain_hash: Most recent chain hash for linking.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> log_id = tracker.record_change("user1", "create", "ef.gas", None, 2.0, "init")
        >>> trail = tracker.get_audit_trail(assumption_id="ef.gas")
    """

    # Initial chain hash (genesis)
    _GENESIS_HASH = hashlib.sha256(b"greenlang-assumptions-genesis").hexdigest()

    def __init__(self) -> None:
        """Initialize ProvenanceTracker."""
        self._entries: List[ChangeLogEntry] = []
        self._last_chain_hash: str = self._GENESIS_HASH
        logger.info("ProvenanceTracker initialized")

    def record_change(
        self,
        user_id: str,
        change_type: str,
        assumption_id: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        scenario_id: Optional[str] = None,
    ) -> str:
        """Record a change to the provenance audit trail.

        Args:
            user_id: User who made the change.
            change_type: Type of change (create, update, delete, etc.).
            assumption_id: Affected assumption identifier.
            old_value: Previous value (None for creates).
            new_value: New value (None for deletes).
            reason: Reason for the change.
            scenario_id: Optional affected scenario identifier.

        Returns:
            The log_id of the new entry.
        """
        # Parse change_type to enum
        ct = ChangeType(change_type) if isinstance(change_type, str) else change_type

        entry = ChangeLogEntry(
            user_id=user_id,
            change_type=ct,
            assumption_id=assumption_id,
            scenario_id=scenario_id,
            old_value=old_value,
            new_value=new_value,
            change_reason=reason,
        )

        # Build chain hash
        entry_data = {
            "user": user_id,
            "type": ct.value,
            "assumption": assumption_id,
            "old": old_value,
            "new": new_value,
            "reason": reason,
            "timestamp": entry.timestamp.isoformat(),
        }
        entry_hash = self._hash_dict(entry_data)
        chain_hash = self._build_next_chain_hash(entry_hash)
        entry.provenance_hash = chain_hash

        self._entries.append(entry)
        self._last_chain_hash = chain_hash

        logger.debug(
            "Recorded provenance: %s %s %s",
            ct.value, assumption_id, entry.log_id,
        )
        return entry.log_id

    def get_audit_trail(
        self,
        assumption_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ChangeLogEntry]:
        """Get the audit trail, optionally filtered.

        Args:
            assumption_id: Optional filter by assumption ID.
            user_id: Optional filter by user ID.
            limit: Maximum number of entries to return.

        Returns:
            List of ChangeLogEntry records, newest first.
        """
        entries = list(self._entries)

        if assumption_id is not None:
            entries = [e for e in entries if e.assumption_id == assumption_id]

        if user_id is not None:
            entries = [e for e in entries if e.user_id == user_id]

        # Sort by timestamp descending (newest first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    def build_chain_hash(self, data: Any) -> str:
        """Build a SHA-256 hash for arbitrary data.

        Args:
            data: Data to hash (dict, list, or string).

        Returns:
            Hex-encoded SHA-256 hash.
        """
        if isinstance(data, dict):
            return self._hash_dict(data)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def verify_chain(self, entries: Optional[List[ChangeLogEntry]] = None) -> bool:
        """Verify the integrity of the provenance chain.

        Recomputes chain hashes from genesis and verifies they match
        the stored hashes in each entry.

        Args:
            entries: Entries to verify. Uses all entries if None.

        Returns:
            True if chain is intact, False if tampered.
        """
        check_entries = entries if entries is not None else self._entries
        if not check_entries:
            return True

        current_hash = self._GENESIS_HASH

        for entry in check_entries:
            entry_data = {
                "user": entry.user_id,
                "type": entry.change_type.value,
                "assumption": entry.assumption_id,
                "old": entry.old_value,
                "new": entry.new_value,
                "reason": entry.change_reason,
                "timestamp": entry.timestamp.isoformat(),
            }
            entry_hash = self._hash_dict(entry_data)
            combined = f"{current_hash}:{entry_hash}"
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()

            if entry.provenance_hash != expected_hash:
                logger.warning(
                    "Chain verification failed at entry %s", entry.log_id,
                )
                return False

            current_hash = expected_hash

        return True

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
    "ProvenanceTracker",
]
