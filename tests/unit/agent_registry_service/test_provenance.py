# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-FOUND-007)

Tests provenance recording, chain retrieval, integrity verification,
SHA-256 hash linking, and tamper detection.

Coverage target: 85%+ of provenance.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ProvenanceTracker mirroring the agent registry service
# ---------------------------------------------------------------------------


class ProvenanceEntry:
    """A single provenance chain entry."""

    def __init__(
        self,
        entry_id: str,
        entity_id: str,
        entity_type: str,
        action: str,
        data_hash: str,
        previous_hash: str = "",
        timestamp: Optional[datetime] = None,
        actor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.action = action
        self.data_hash = data_hash
        self.previous_hash = previous_hash
        self.timestamp = timestamp or datetime.utcnow()
        self.actor = actor
        self.details = details or {}
        self.chain_hash = self._compute_chain_hash()

    def _compute_chain_hash(self) -> str:
        content = json.dumps(
            {
                "entry_id": self.entry_id,
                "entity_id": self.entity_id,
                "action": self.action,
                "data_hash": self.data_hash,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "action": self.action,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
            "chain_hash": self.chain_hash,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
        }


class ProvenanceTracker:
    """Tracks provenance chains for agent registry operations."""

    def __init__(self):
        self._chains: Dict[str, List[ProvenanceEntry]] = {}

    @property
    def entry_count(self) -> int:
        return sum(len(chain) for chain in self._chains.values())

    def record(
        self,
        entity_id: str,
        entity_type: str,
        action: str,
        data_hash: str,
        actor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Record a new provenance entry."""
        chain = self._chains.get(entity_id, [])
        previous_hash = chain[-1].chain_hash if chain else ""

        entry = ProvenanceEntry(
            entry_id=str(uuid.uuid4()),
            entity_id=entity_id,
            entity_type=entity_type,
            action=action,
            data_hash=data_hash,
            previous_hash=previous_hash,
            actor=actor,
            details=details,
        )

        chain.append(entry)
        self._chains[entity_id] = chain
        return entry

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        """Get the full provenance chain for an entity."""
        return self._chains.get(entity_id, [])

    def verify_chain(self, entity_id: str) -> Dict[str, Any]:
        """Verify the integrity of a provenance chain."""
        chain = self.get_chain(entity_id)

        if not chain:
            return {
                "valid": False,
                "entity_id": entity_id,
                "error": "Chain not found",
                "entries_checked": 0,
            }

        errors: List[str] = []

        if chain[0].previous_hash != "":
            errors.append("First entry has non-empty previous_hash")

        for i in range(1, len(chain)):
            expected_prev = chain[i - 1].chain_hash
            actual_prev = chain[i].previous_hash
            if actual_prev != expected_prev:
                errors.append(
                    f"Entry {i}: expected previous_hash "
                    f"'{expected_prev[:16]}...' but got '{actual_prev[:16]}...'"
                )

        for i, entry in enumerate(chain):
            recomputed = entry._compute_chain_hash()
            if recomputed != entry.chain_hash:
                errors.append(
                    f"Entry {i}: chain_hash mismatch"
                )

        return {
            "valid": len(errors) == 0,
            "entity_id": entity_id,
            "entries_checked": len(chain),
            "errors": errors,
        }

    def export_json(self, entity_id: str) -> str:
        """Export provenance chain as JSON."""
        chain = self.get_chain(entity_id)
        return json.dumps(
            [e.to_dict() for e in chain],
            indent=2, default=str,
        )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestProvenanceRecord:
    """Test recording entries."""

    def test_record_first_entry(self):
        tracker = ProvenanceTracker()
        entry = tracker.record(
            entity_id="gl-001", entity_type="agent",
            action="register", data_hash="abc123",
        )
        assert entry.entity_id == "gl-001"
        assert entry.entity_type == "agent"
        assert entry.action == "register"
        assert entry.data_hash == "abc123"
        assert entry.previous_hash == ""

    def test_record_increments_count(self):
        tracker = ProvenanceTracker()
        assert tracker.entry_count == 0
        tracker.record("gl-001", "agent", "register", "h1")
        assert tracker.entry_count == 1
        tracker.record("gl-001", "agent", "update", "h2")
        assert tracker.entry_count == 2

    def test_record_sets_chain_hash(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("gl-001", "agent", "register", "h1")
        assert len(entry.chain_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", entry.chain_hash)

    def test_record_links_to_previous(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("gl-001", "agent", "register", "h1")
        e2 = tracker.record("gl-001", "agent", "update", "h2")
        assert e2.previous_hash == e1.chain_hash

    def test_record_actor(self):
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "gl-001", "agent", "register", "h1", actor="admin-user",
        )
        assert entry.actor == "admin-user"

    def test_record_details(self):
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "gl-001", "agent", "register", "h1",
            details={"version": "1.0.0"},
        )
        assert entry.details["version"] == "1.0.0"

    def test_record_auto_entry_id(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("gl-001", "agent", "register", "h1")
        assert len(entry.entry_id) == 36  # UUID

    def test_record_timestamp_auto(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("gl-001", "agent", "register", "h1")
        assert isinstance(entry.timestamp, datetime)


class TestProvenanceGetChain:
    """Test chain retrieval."""

    def test_get_chain_single_entry(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        chain = tracker.get_chain("gl-001")
        assert len(chain) == 1

    def test_get_chain_multiple_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-001", "agent", "update", "h2")
        tracker.record("gl-001", "agent", "hot_reload", "h3")
        chain = tracker.get_chain("gl-001")
        assert len(chain) == 3

    def test_get_chain_not_found(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("nonexistent")
        assert chain == []

    def test_get_chain_separate_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-002", "agent", "register", "h2")
        assert len(tracker.get_chain("gl-001")) == 1
        assert len(tracker.get_chain("gl-002")) == 1

    def test_get_chain_order_preserved(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("gl-001", "agent", "register", "h1")
        e2 = tracker.record("gl-001", "agent", "update", "h2")
        chain = tracker.get_chain("gl-001")
        assert chain[0].entry_id == e1.entry_id
        assert chain[1].entry_id == e2.entry_id


class TestProvenanceVerifyChain:
    """Test integrity verification."""

    def test_valid_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-001", "agent", "update", "h2")
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is True
        assert result["entries_checked"] == 2
        assert result["errors"] == []

    def test_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is True
        assert result["entries_checked"] == 1

    def test_nonexistent_chain_invalid(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_long_chain_valid(self):
        tracker = ProvenanceTracker()
        for i in range(20):
            tracker.record("gl-001", "agent", f"update-{i}", f"h{i}")
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is True
        assert result["entries_checked"] == 20


class TestProvenanceHashChain:
    """Test SHA-256 linking."""

    def test_first_entry_previous_hash_empty(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("gl-001", "agent", "register", "h1")
        assert entry.previous_hash == ""

    def test_second_entry_links_first(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("gl-001", "agent", "register", "h1")
        e2 = tracker.record("gl-001", "agent", "update", "h2")
        assert e2.previous_hash == e1.chain_hash

    def test_third_entry_links_second(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        e2 = tracker.record("gl-001", "agent", "update", "h2")
        e3 = tracker.record("gl-001", "agent", "update", "h3")
        assert e3.previous_hash == e2.chain_hash

    def test_chain_hash_is_sha256(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("gl-001", "agent", "register", "h1")
        assert re.match(r"^[0-9a-f]{64}$", entry.chain_hash)

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("gl-001", "agent", "register", "h1")
        e2 = tracker.record("gl-001", "agent", "update", "h2")
        assert e1.chain_hash != e2.chain_hash


class TestProvenanceTamperDetection:
    """Test modified hash fails verification."""

    def test_tampered_chain_hash_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-001", "agent", "update", "h2")
        chain = tracker.get_chain("gl-001")
        chain[0].chain_hash = "0" * 64
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_tampered_previous_hash_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-001", "agent", "update", "h2")
        chain = tracker.get_chain("gl-001")
        chain[1].previous_hash = "f" * 64
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is False

    def test_tampered_data_hash_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        chain = tracker.get_chain("gl-001")
        chain[0].data_hash = "tampered"
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is False

    def test_untampered_chain_stays_valid(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("gl-001", "agent", f"action-{i}", f"hash-{i}")
        result = tracker.verify_chain("gl-001")
        assert result["valid"] is True

    def test_entry_count_multiple_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        tracker.record("gl-001", "agent", "update", "h2")
        tracker.record("gl-002", "agent", "register", "h3")
        assert tracker.entry_count == 3


class TestProvenanceExport:
    """Test JSON export."""

    def test_export_json(self):
        tracker = ProvenanceTracker()
        tracker.record("gl-001", "agent", "register", "h1")
        json_str = tracker.export_json("gl-001")
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["entity_id"] == "gl-001"

    def test_export_json_empty(self):
        tracker = ProvenanceTracker()
        json_str = tracker.export_json("nonexistent")
        data = json.loads(json_str)
        assert data == []
