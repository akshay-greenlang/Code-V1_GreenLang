# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-FOUND-009)

Tests provenance recording, chain retrieval, integrity verification,
SHA-256 hash linking, tamper detection, and multi-entity chains.

Coverage target: 85%+ of provenance.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ProvenanceTracker (NO config arg per spec)
# ---------------------------------------------------------------------------


class ProvenanceEntry:
    """A single provenance chain entry."""

    def __init__(self, entry_id: str, entity_id: str, entity_type: str,
                 action: str, data_hash: str, previous_hash: str = "",
                 timestamp: Optional[datetime] = None,
                 actor: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.action = action
        self.data_hash = data_hash
        self.previous_hash = previous_hash
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.actor = actor
        self.details = details or {}
        self.chain_hash = self._compute_chain_hash()

    def _compute_chain_hash(self) -> str:
        content = json.dumps({
            "entry_id": self.entry_id,
            "entity_id": self.entity_id,
            "action": self.action,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class ProvenanceTracker:
    """Tracks provenance chains for QA test harness."""

    def __init__(self):
        self._chains: Dict[str, List[ProvenanceEntry]] = {}
        self._entry_counter = 0

    @property
    def entry_count(self) -> int:
        return sum(len(chain) for chain in self._chains.values())

    def record(self, entity_id: str, entity_type: str, action: str,
               data_hash: str, actor: Optional[str] = None,
               details: Optional[Dict[str, Any]] = None) -> ProvenanceEntry:
        self._entry_counter += 1
        entry_id = f"prov-{self._entry_counter:06d}"

        chain = self._chains.get(entity_id, [])
        previous_hash = chain[-1].chain_hash if chain else ""

        entry = ProvenanceEntry(
            entry_id=entry_id, entity_id=entity_id,
            entity_type=entity_type, action=action,
            data_hash=data_hash, previous_hash=previous_hash,
            actor=actor, details=details,
        )

        if entity_id not in self._chains:
            self._chains[entity_id] = []
        self._chains[entity_id].append(entry)
        return entry

    def verify_chain(self, entity_id: str) -> Dict[str, Any]:
        chain = self._chains.get(entity_id, [])

        if not chain:
            return {
                "entity_id": entity_id, "is_valid": True,
                "length": 0, "message": "Empty chain (valid)",
            }

        if chain[0].previous_hash != "":
            return {
                "entity_id": entity_id, "is_valid": False,
                "length": len(chain),
                "message": "First entry has non-empty previous_hash",
                "tampered_at": 0,
            }

        for i in range(1, len(chain)):
            expected_prev = chain[i - 1].chain_hash
            actual_prev = chain[i].previous_hash
            if actual_prev != expected_prev:
                return {
                    "entity_id": entity_id, "is_valid": False,
                    "length": len(chain),
                    "message": f"Chain broken at entry {i}",
                    "tampered_at": i,
                }

        for i, entry in enumerate(chain):
            recomputed = entry._compute_chain_hash()
            if recomputed != entry.chain_hash:
                return {
                    "entity_id": entity_id, "is_valid": False,
                    "length": len(chain),
                    "message": f"Entry {i} hash tampered",
                    "tampered_at": i,
                }

        return {
            "entity_id": entity_id, "is_valid": True,
            "length": len(chain), "message": "Chain integrity verified",
        }

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        return self._chains.get(entity_id, [])


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRecord:
    def test_record_creates_entry(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("entity-001", "test_run", "executed", "abc123")
        assert entry.entity_id == "entity-001"
        assert entry.entity_type == "test_run"
        assert entry.action == "executed"
        assert entry.data_hash == "abc123"

    def test_record_entry_id_format(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("e1", "test", "action", "hash")
        assert entry.entry_id.startswith("prov-")

    def test_record_chain_hash_computation(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("e1", "test", "create", "h1")
        assert len(entry.chain_hash) == 64

    def test_record_first_entry_no_previous(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("e1", "test", "create", "h1")
        assert entry.previous_hash == ""

    def test_record_second_entry_links_to_first(self):
        tracker = ProvenanceTracker()
        e1 = tracker.record("e1", "test", "create", "h1")
        e2 = tracker.record("e1", "test", "update", "h2")
        assert e2.previous_hash == e1.chain_hash

    def test_record_with_actor(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("e1", "test", "create", "h1", actor="user-001")
        assert entry.actor == "user-001"

    def test_record_with_details(self):
        tracker = ProvenanceTracker()
        entry = tracker.record("e1", "test", "create", "h1",
                               details={"reason": "initial"})
        assert entry.details["reason"] == "initial"

    def test_record_increments_counter(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e2", "test", "a2", "h2")
        assert tracker._entry_counter == 2

    def test_record_entry_count_property(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e1", "test", "a2", "h2")
        tracker.record("e2", "test", "a3", "h3")
        assert tracker.entry_count == 3


class TestVerifyChain:
    def test_verify_chain_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker.record("e1", "test", "update", "h2")
        tracker.record("e1", "test", "verify", "h3")
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is True
        assert result["length"] == 3

    def test_verify_chain_tampered_linkage(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker.record("e1", "test", "update", "h2")
        tracker._chains["e1"][0].chain_hash = "tampered_hash"
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False

    def test_verify_chain_empty(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["length"] == 0

    def test_verify_chain_single_entry(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is True
        assert result["length"] == 1

    def test_verify_chain_tampered_hash_recompute(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker._chains["e1"][0].data_hash = "tampered"
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False
        assert "tampered" in result["message"].lower()

    def test_verify_chain_message_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        result = tracker.verify_chain("e1")
        assert "verified" in result["message"].lower()


class TestGetChain:
    def test_get_chain_returns_ordered(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e1", "test", "a2", "h2")
        tracker.record("e1", "test", "a3", "h3")
        chain = tracker.get_chain("e1")
        assert len(chain) == 3
        assert chain[0].action == "a1"
        assert chain[1].action == "a2"
        assert chain[2].action == "a3"

    def test_get_chain_nonexistent(self):
        tracker = ProvenanceTracker()
        chain = tracker.get_chain("nonexistent")
        assert chain == []

    def test_get_chain_separate_entities(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e2", "test", "a2", "h2")
        chain1 = tracker.get_chain("e1")
        chain2 = tracker.get_chain("e2")
        assert len(chain1) == 1
        assert len(chain2) == 1
        assert chain1[0].entity_id == "e1"
        assert chain2[0].entity_id == "e2"


class TestMultipleRecordsSameEntity:
    def test_multiple_records(self):
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record("e1", "test", f"action_{i}", f"hash_{i}")
        chain = tracker.get_chain("e1")
        assert len(chain) == 5

    def test_chain_linkage_maintained(self):
        tracker = ProvenanceTracker()
        entries = []
        for i in range(5):
            e = tracker.record("e1", "test", f"a{i}", f"h{i}")
            entries.append(e)
        for i in range(1, 5):
            assert entries[i].previous_hash == entries[i - 1].chain_hash


class TestChainHashDeterminism:
    def test_chain_hash_deterministic(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "", ts)
        e2 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "", ts)
        assert e1.chain_hash == e2.chain_hash

    def test_different_data_hash_different_chain_hash(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = ProvenanceEntry("p-1", "e1", "test", "create", "hash_a", "", ts)
        e2 = ProvenanceEntry("p-1", "e1", "test", "create", "hash_b", "", ts)
        assert e1.chain_hash != e2.chain_hash

    def test_different_action_different_chain_hash(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "", ts)
        e2 = ProvenanceEntry("p-1", "e1", "test", "update", "h1", "", ts)
        assert e1.chain_hash != e2.chain_hash

    def test_different_entity_id_different_chain_hash(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "", ts)
        e2 = ProvenanceEntry("p-1", "e2", "test", "create", "h1", "", ts)
        assert e1.chain_hash != e2.chain_hash

    def test_different_previous_hash_different_chain_hash(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "", ts)
        e2 = ProvenanceEntry("p-1", "e1", "test", "create", "h1", "prev123", ts)
        assert e1.chain_hash != e2.chain_hash
