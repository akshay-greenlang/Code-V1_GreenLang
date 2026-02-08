# -*- coding: utf-8 -*-
"""
Integration Tests for Provenance Chain (AGENT-FOUND-008)

Tests provenance chain integrity, tamper detection, multiple entities,
and cross-component provenance tracking.

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
# Inline provenance tracker for integration testing
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
        entry_id = f"prov-{self._counter:06d}"
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
            return {"is_valid": True, "length": 0}
        if chain[0].previous_hash != "":
            return {"is_valid": False, "length": len(chain), "error": "genesis"}
        for i in range(1, len(chain)):
            if chain[i].previous_hash != chain[i - 1].chain_hash:
                return {"is_valid": False, "length": len(chain),
                        "error": f"break at {i}"}
        for i, entry in enumerate(chain):
            if entry._compute_chain_hash() != entry.chain_hash:
                return {"is_valid": False, "length": len(chain),
                        "error": f"tampered at {i}"}
        return {"is_valid": True, "length": len(chain)}

    def get_chain(self, entity_id: str) -> List[ProvenanceEntry]:
        return self._chains.get(entity_id, [])

    @property
    def total_entries(self) -> int:
        return sum(len(c) for c in self._chains.values())


def _content_hash(data: Dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestProvenanceChainIntegrity:
    """Test provenance chain integrity across multiple operations."""

    def test_single_entry_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "verification", "created", "hash_001")
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is True
        assert result["length"] == 1

    def test_multi_entry_chain_valid(self):
        tracker = ProvenanceTracker()
        for i in range(10):
            tracker.record("e1", "verification", f"step_{i}", f"hash_{i}")
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is True
        assert result["length"] == 10

    def test_chain_linkage_complete(self):
        tracker = ProvenanceTracker()
        entries = []
        for i in range(5):
            e = tracker.record("e1", "test", f"action_{i}", f"h_{i}")
            entries.append(e)
        # Verify each entry links to previous
        for i in range(1, len(entries)):
            assert entries[i].previous_hash == entries[i - 1].chain_hash

    def test_first_entry_has_empty_previous(self):
        tracker = ProvenanceTracker()
        e = tracker.record("e1", "test", "create", "h1")
        assert e.previous_hash == ""

    def test_long_chain_integrity(self):
        tracker = ProvenanceTracker()
        for i in range(100):
            tracker.record("entity-long", "test", f"op_{i}", f"h_{i}")
        result = tracker.verify_chain("entity-long")
        assert result["is_valid"] is True
        assert result["length"] == 100

    def test_empty_chain_valid(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["length"] == 0


class TestProvenanceTamperDetection:
    """Test tamper detection in provenance chains."""

    def test_detect_modified_data_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker.record("e1", "test", "update", "h2")
        # Tamper
        tracker._chains["e1"][0].data_hash = "TAMPERED"
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False

    def test_detect_modified_chain_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker.record("e1", "test", "update", "h2")
        # Tamper with chain hash
        tracker._chains["e1"][0].chain_hash = "TAMPERED"
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False

    def test_detect_modified_action(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        # Tamper with action
        tracker._chains["e1"][0].action = "TAMPERED"
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False

    def test_detect_reordered_entries(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e1", "test", "a2", "h2")
        tracker.record("e1", "test", "a3", "h3")
        # Swap entries 1 and 2
        chain = tracker._chains["e1"]
        chain[1], chain[2] = chain[2], chain[1]
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False

    def test_detect_inserted_entry(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "a1", "h1")
        tracker.record("e1", "test", "a2", "h2")
        # Insert a fake entry
        fake = ProvenanceEntry("fake", "e1", "test", "fake", "fake_hash", "",
                               datetime(2026, 1, 1, tzinfo=timezone.utc))
        tracker._chains["e1"].insert(1, fake)
        result = tracker.verify_chain("e1")
        assert result["is_valid"] is False


class TestProvenanceMultipleEntities:
    """Test provenance across multiple entities."""

    def test_independent_chains(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "verification", "create", "h1")
        tracker.record("e2", "drift", "detect", "h2")
        tracker.record("e1", "verification", "update", "h3")
        assert tracker.verify_chain("e1")["is_valid"] is True
        assert tracker.verify_chain("e2")["is_valid"] is True
        assert len(tracker.get_chain("e1")) == 2
        assert len(tracker.get_chain("e2")) == 1

    def test_many_entities(self):
        tracker = ProvenanceTracker()
        for entity in range(20):
            for op in range(5):
                tracker.record(f"entity-{entity}", "test", f"op_{op}", f"h_{entity}_{op}")
        assert tracker.total_entries == 100
        for entity in range(20):
            result = tracker.verify_chain(f"entity-{entity}")
            assert result["is_valid"] is True
            assert result["length"] == 5

    def test_entity_tampering_isolated(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1")
        tracker.record("e2", "test", "create", "h2")
        # Tamper e1 only
        tracker._chains["e1"][0].data_hash = "TAMPERED"
        assert tracker.verify_chain("e1")["is_valid"] is False
        assert tracker.verify_chain("e2")["is_valid"] is True


class TestCrossComponentProvenance:
    """Test provenance tracking across components."""

    def test_verification_to_drift_provenance(self):
        tracker = ProvenanceTracker()
        # Verification component records
        v_hash = _content_hash({"emissions": 100.0})
        tracker.record("exec-001", "verification", "input_hashed", v_hash)
        # Drift component records
        tracker.record("exec-001", "drift", "baseline_compared", v_hash)
        chain = tracker.get_chain("exec-001")
        assert len(chain) == 2
        assert chain[0].entity_type == "verification"
        assert chain[1].entity_type == "drift"

    def test_full_pipeline_provenance(self):
        tracker = ProvenanceTracker()
        exec_id = "exec-pipeline-001"
        # Step 1: Input hashing
        ih = _content_hash({"fuel": "diesel", "qty": 1000})
        tracker.record(exec_id, "hasher", "input_hash", ih)
        # Step 2: Verification
        tracker.record(exec_id, "verifier", "verified", ih)
        # Step 3: Drift check
        tracker.record(exec_id, "drift", "checked", ih)
        # Step 4: Report
        tracker.record(exec_id, "reporter", "generated", ih)
        chain = tracker.get_chain(exec_id)
        assert len(chain) == 4
        result = tracker.verify_chain(exec_id)
        assert result["is_valid"] is True

    def test_provenance_actors_tracked(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1", actor="agent-001")
        tracker.record("e1", "test", "verify", "h2", actor="verifier-001")
        chain = tracker.get_chain("e1")
        assert chain[0].actor == "agent-001"
        assert chain[1].actor == "verifier-001"

    def test_provenance_details_preserved(self):
        tracker = ProvenanceTracker()
        tracker.record("e1", "test", "create", "h1",
                        details={"source": "ERP", "batch_size": 100})
        chain = tracker.get_chain("e1")
        assert chain[0].details["source"] == "ERP"
        assert chain[0].details["batch_size"] == 100
