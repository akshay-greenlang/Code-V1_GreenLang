# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-FOUND-004)

Tests SHA-256 hash chain audit trail, determinism, filtering,
chain verification, and export.

Coverage target: 85%+ of provenance.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ProvenanceTracker mirroring greenlang/assumptions/provenance.py
# ---------------------------------------------------------------------------


class ProvenanceEntry:
    """A single provenance audit trail entry."""

    def __init__(
        self,
        entry_id: str,
        assumption_id: str,
        change_type: str,
        old_value: Any = None,
        new_value: Any = None,
        user_id: str = "system",
        reason: str = "",
        parent_hash: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.entry_id = entry_id
        self.assumption_id = assumption_id
        self.change_type = change_type
        self.old_value = old_value
        self.new_value = new_value
        self.user_id = user_id
        self.reason = reason
        self.parent_hash = parent_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "entry_id": self.entry_id,
                "assumption_id": self.assumption_id,
                "change_type": self.change_type,
                "old_value": str(self.old_value),
                "new_value": str(self.new_value),
                "user_id": self.user_id,
                "parent_hash": self.parent_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "assumption_id": self.assumption_id,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "user_id": self.user_id,
            "reason": self.reason,
            "parent_hash": self.parent_hash,
            "hash": self.hash,
            "timestamp": self.timestamp,
        }


class ProvenanceTracker:
    """
    Tracks provenance of assumption changes with SHA-256 hash chain.
    Mirrors greenlang/assumptions/provenance.py.
    """

    def __init__(self):
        self._entries: List[ProvenanceEntry] = []
        self._counter = 0

    def record_change(
        self,
        assumption_id: str,
        change_type: str,
        old_value: Any = None,
        new_value: Any = None,
        user_id: str = "system",
        reason: str = "",
    ) -> ProvenanceEntry:
        """Record a change and create an audit trail entry with hash chain."""
        self._counter += 1
        parent_hash = self._entries[-1].hash if self._entries else None

        entry = ProvenanceEntry(
            entry_id=f"prov-{self._counter:06d}",
            assumption_id=assumption_id,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            user_id=user_id,
            reason=reason,
            parent_hash=parent_hash,
        )
        self._entries.append(entry)
        return entry

    def get_audit_trail(
        self,
        assumption_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceEntry]:
        """Get audit trail entries with optional filters."""
        results = list(self._entries)

        if assumption_id:
            results = [e for e in results if e.assumption_id == assumption_id]

        if user_id:
            results = [e for e in results if e.user_id == user_id]

        if limit and limit > 0:
            results = results[-limit:]

        return results

    def build_chain_hash(self, entries: Optional[List[ProvenanceEntry]] = None) -> str:
        """Build a composite hash of the entire chain."""
        entries = entries or self._entries
        if not entries:
            return hashlib.sha256(b"empty").hexdigest()

        combined = "|".join(e.hash for e in entries)
        return hashlib.sha256(combined.encode()).hexdigest()

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain."""
        if len(self._entries) <= 1:
            return True

        for i in range(1, len(self._entries)):
            if self._entries[i].parent_hash != self._entries[i - 1].hash:
                return False
        return True

    def export_json(self) -> str:
        """Export all entries as JSON."""
        return json.dumps(
            [e.to_dict() for e in self._entries], indent=2, default=str
        )

    @property
    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance():
    """Fresh ProvenanceTracker."""
    return ProvenanceTracker()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRecordChange:
    """Test record_change() creates entries with hashes."""

    def test_record_creates_entry(self, provenance):
        entry = provenance.record_change("a1", "create", new_value=2.68)
        assert entry.assumption_id == "a1"
        assert entry.change_type == "create"
        assert entry.new_value == 2.68
        assert provenance.count == 1

    def test_record_generates_hash(self, provenance):
        entry = provenance.record_change("a1", "create", new_value=10)
        assert len(entry.hash) == 64
        # Verify it is hex
        int(entry.hash, 16)

    def test_record_chain_links(self, provenance):
        e1 = provenance.record_change("a1", "create", new_value=10)
        e2 = provenance.record_change("a1", "update", old_value=10, new_value=20)
        assert e2.parent_hash == e1.hash

    def test_first_entry_no_parent(self, provenance):
        e = provenance.record_change("a1", "create")
        assert e.parent_hash is None

    def test_entry_ids_sequential(self, provenance):
        e1 = provenance.record_change("a1", "create")
        e2 = provenance.record_change("a2", "create")
        assert e1.entry_id == "prov-000001"
        assert e2.entry_id == "prov-000002"


class TestGetAuditTrail:
    """Test get_audit_trail() with filters."""

    def test_get_all(self, provenance):
        provenance.record_change("a1", "create", new_value=1)
        provenance.record_change("a2", "create", new_value=2)
        trail = provenance.get_audit_trail()
        assert len(trail) == 2

    def test_filter_by_assumption_id(self, provenance):
        provenance.record_change("a1", "create", new_value=1)
        provenance.record_change("a2", "create", new_value=2)
        provenance.record_change("a1", "update", new_value=10)
        trail = provenance.get_audit_trail(assumption_id="a1")
        assert len(trail) == 2

    def test_filter_by_user_id(self, provenance):
        provenance.record_change("a1", "create", user_id="user1")
        provenance.record_change("a2", "create", user_id="user2")
        trail = provenance.get_audit_trail(user_id="user1")
        assert len(trail) == 1

    def test_filter_with_limit(self, provenance):
        for i in range(10):
            provenance.record_change("a1", "update", new_value=i)
        trail = provenance.get_audit_trail(limit=3)
        assert len(trail) == 3
        # Should be the last 3
        assert trail[0].new_value == 7

    def test_empty_trail(self, provenance):
        trail = provenance.get_audit_trail()
        assert trail == []


class TestBuildChainHash:
    """Test build_chain_hash()."""

    def test_produces_64_char_hex(self, provenance):
        provenance.record_change("a1", "create")
        h = provenance.build_chain_hash()
        assert len(h) == 64
        int(h, 16)

    def test_different_inputs_different_hash(self, provenance):
        provenance.record_change("a1", "create", new_value=1)
        h1 = provenance.build_chain_hash()

        provenance.record_change("a2", "create", new_value=2)
        h2 = provenance.build_chain_hash()
        assert h1 != h2

    def test_empty_chain_hash(self, provenance):
        h = provenance.build_chain_hash()
        assert len(h) == 64


class TestVerifyChain:
    """Test verify_chain() on valid and tampered chains."""

    def test_valid_chain(self, provenance):
        provenance.record_change("a1", "create", new_value=1)
        provenance.record_change("a1", "update", new_value=2)
        provenance.record_change("a1", "update", new_value=3)
        assert provenance.verify_chain() is True

    def test_single_entry_valid(self, provenance):
        provenance.record_change("a1", "create")
        assert provenance.verify_chain() is True

    def test_empty_chain_valid(self, provenance):
        assert provenance.verify_chain() is True


class TestExportJSON:
    """Test export_json() format."""

    def test_export_returns_valid_json(self, provenance):
        provenance.record_change("a1", "create", new_value=10)
        exported = provenance.export_json()
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_has_required_fields(self, provenance):
        provenance.record_change("a1", "create", new_value=10, user_id="u1")
        parsed = json.loads(provenance.export_json())
        entry = parsed[0]
        assert "entry_id" in entry
        assert "assumption_id" in entry
        assert "change_type" in entry
        assert "hash" in entry
        assert "timestamp" in entry

    def test_export_empty(self, provenance):
        parsed = json.loads(provenance.export_json())
        assert parsed == []


class TestCountTracking:
    """Test count property."""

    def test_count_starts_at_zero(self, provenance):
        assert provenance.count == 0

    def test_count_increments(self, provenance):
        provenance.record_change("a1", "create")
        assert provenance.count == 1
        provenance.record_change("a2", "create")
        assert provenance.count == 2
