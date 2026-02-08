# -*- coding: utf-8 -*-
"""
Integration Tests for Provenance Chain Integrity (AGENT-FOUND-009)

Tests end-to-end provenance chain creation, verification, tamper detection,
and multi-entity provenance tracking.

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
# Inline ProvenanceTracker (full integration version)
# ---------------------------------------------------------------------------


class ProvenanceEntry:
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
    def __init__(self):
        self._chains: Dict[str, List[ProvenanceEntry]] = {}
        self._counter = 0

    def record(self, entity_id: str, entity_type: str, action: str,
               data_hash: str, actor: Optional[str] = None,
               details: Optional[Dict[str, Any]] = None) -> ProvenanceEntry:
        self._counter += 1
        chain = self._chains.get(entity_id, [])
        prev = chain[-1].chain_hash if chain else ""
        entry = ProvenanceEntry(
            f"prov-{self._counter:06d}", entity_id, entity_type,
            action, data_hash, prev, actor=actor, details=details,
        )
        if entity_id not in self._chains:
            self._chains[entity_id] = []
        self._chains[entity_id].append(entry)
        return entry

    def verify_chain(self, entity_id: str) -> Dict[str, Any]:
        chain = self._chains.get(entity_id, [])
        if not chain:
            return {"is_valid": True, "length": 0, "message": "Empty chain"}
        if chain[0].previous_hash != "":
            return {"is_valid": False, "length": len(chain),
                    "message": "First entry has previous hash"}
        for i in range(1, len(chain)):
            if chain[i].previous_hash != chain[i - 1].chain_hash:
                return {"is_valid": False, "length": len(chain),
                        "message": f"Broken at {i}"}
        for i, e in enumerate(chain):
            if e._compute_chain_hash() != e.chain_hash:
                return {"is_valid": False, "length": len(chain),
                        "message": f"Tampered at {i}"}
        return {"is_valid": True, "length": len(chain),
                "message": "Verified"}

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        return self._chains.get(entity_id, [])


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def tracker():
    return ProvenanceTracker()


class TestProvenanceChainIntegrity:
    def test_single_entry_valid(self, tracker):
        tracker.record("test-1", "test_run", "executed", "abc123")
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is True
        assert result["length"] == 1

    def test_multi_entry_valid(self, tracker):
        tracker.record("test-1", "test_run", "created", "h1")
        tracker.record("test-1", "test_run", "executed", "h2")
        tracker.record("test-1", "test_run", "completed", "h3")
        tracker.record("test-1", "test_run", "reported", "h4")
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is True
        assert result["length"] == 4

    def test_chain_linkage_correct(self, tracker):
        entries = []
        for i in range(5):
            e = tracker.record("test-1", "run", f"step_{i}", f"hash_{i}")
            entries.append(e)
        for i in range(1, 5):
            assert entries[i].previous_hash == entries[i - 1].chain_hash

    def test_first_entry_no_previous(self, tracker):
        e = tracker.record("test-1", "run", "start", "h1")
        assert e.previous_hash == ""

    def test_chain_hash_sha256_length(self, tracker):
        e = tracker.record("test-1", "run", "start", "h1")
        assert len(e.chain_hash) == 64

    def test_long_chain_integrity(self, tracker):
        for i in range(100):
            tracker.record("test-1", "run", f"step_{i}", f"hash_{i}")
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is True
        assert result["length"] == 100


class TestProvenanceTamperDetection:
    def test_tamper_chain_hash(self, tracker):
        tracker.record("test-1", "run", "step1", "h1")
        tracker.record("test-1", "run", "step2", "h2")
        tracker._chains["test-1"][0].chain_hash = "tampered"
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is False

    def test_tamper_data_hash(self, tracker):
        tracker.record("test-1", "run", "step1", "h1")
        tracker._chains["test-1"][0].data_hash = "tampered"
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is False

    def test_tamper_previous_hash(self, tracker):
        tracker.record("test-1", "run", "step1", "h1")
        tracker.record("test-1", "run", "step2", "h2")
        tracker._chains["test-1"][1].previous_hash = "tampered"
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is False

    def test_tamper_action(self, tracker):
        tracker.record("test-1", "run", "step1", "h1")
        tracker._chains["test-1"][0].action = "tampered_action"
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is False

    def test_tamper_middle_of_chain(self, tracker):
        for i in range(5):
            tracker.record("test-1", "run", f"step_{i}", f"h{i}")
        tracker._chains["test-1"][2].data_hash = "tampered"
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is False

    def test_untampered_chain_valid(self, tracker):
        for i in range(10):
            tracker.record("test-1", "run", f"step_{i}", f"h{i}")
        result = tracker.verify_chain("test-1")
        assert result["is_valid"] is True


class TestProvenanceMultiEntity:
    def test_separate_entities_independent(self, tracker):
        tracker.record("entity-1", "test", "action1", "h1")
        tracker.record("entity-2", "test", "action2", "h2")
        tracker.record("entity-1", "test", "action3", "h3")

        chain1 = tracker.get_chain("entity-1")
        chain2 = tracker.get_chain("entity-2")
        assert len(chain1) == 2
        assert len(chain2) == 1

    def test_multi_entity_all_valid(self, tracker):
        for i in range(3):
            tracker.record("e1", "test", f"a{i}", f"h{i}")
        for i in range(5):
            tracker.record("e2", "test", f"b{i}", f"g{i}")

        r1 = tracker.verify_chain("e1")
        r2 = tracker.verify_chain("e2")
        assert r1["is_valid"] is True
        assert r2["is_valid"] is True

    def test_tamper_one_entity_other_valid(self, tracker):
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e1", "test", "a2", "h2")
        tracker.record("e2", "test", "b1", "g1")
        tracker.record("e2", "test", "b2", "g2")

        tracker._chains["e1"][0].data_hash = "tampered"

        r1 = tracker.verify_chain("e1")
        r2 = tracker.verify_chain("e2")
        assert r1["is_valid"] is False
        assert r2["is_valid"] is True

    def test_nonexistent_entity_valid(self, tracker):
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["length"] == 0

    def test_actors_tracked(self, tracker):
        tracker.record("e1", "test", "action", "h1", actor="user-1")
        tracker.record("e1", "test", "action", "h2", actor="user-2")
        chain = tracker.get_chain("e1")
        assert chain[0].actor == "user-1"
        assert chain[1].actor == "user-2"

    def test_details_tracked(self, tracker):
        tracker.record("e1", "test", "action", "h1",
                       details={"test_name": "test_calc", "category": "unit"})
        chain = tracker.get_chain("e1")
        assert chain[0].details["test_name"] == "test_calc"
