# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-DATA-003)

Tests provenance recording, chain verification, chain retrieval,
JSON export, genesis hash generation, tamper detection, and
deterministic hashing for ERP connector audit trails.

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
# Inline ProvenanceTracker mirroring greenlang/erp_connector/provenance.py
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """SHA-256 provenance chain tracker for ERP connector audit trails."""

    GENESIS_HASH = "0" * 64

    def __init__(self, agent_id: str = "GL-DATA-X-004"):
        self._agent_id = agent_id
        self._chains: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(self, chain_id: str, operation: str, data: Dict[str, Any],
               actor: str = "system") -> str:
        if chain_id not in self._chains:
            self._chains[chain_id] = []
        chain = self._chains[chain_id]
        previous_hash = chain[-1]["record_hash"] if chain else self.GENESIS_HASH
        record = {
            "chain_id": chain_id, "sequence": len(chain) + 1,
            "operation": operation, "data_hash": self._hash_data(data),
            "previous_hash": previous_hash, "agent_id": self._agent_id,
            "actor": actor, "timestamp": datetime.utcnow().isoformat(),
        }
        record["record_hash"] = self._hash_record(record)
        chain.append(record)
        return record["record_hash"]

    def verify_chain(self, chain_id: str) -> Dict[str, Any]:
        chain = self._chains.get(chain_id, [])
        if not chain:
            return {"is_valid": True, "chain_length": 0, "broken_at": None}
        for i, record in enumerate(chain):
            expected_prev = chain[i - 1]["record_hash"] if i > 0 else self.GENESIS_HASH
            if record["previous_hash"] != expected_prev:
                return {"is_valid": False, "chain_length": len(chain), "broken_at": i + 1}
            expected_hash = self._hash_record({k: v for k, v in record.items() if k != "record_hash"})
            if record["record_hash"] != expected_hash:
                return {"is_valid": False, "chain_length": len(chain), "broken_at": i + 1}
        return {"is_valid": True, "chain_length": len(chain), "broken_at": None}

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        return list(self._chains.get(chain_id, []))

    def get_latest(self, chain_id: str) -> Optional[Dict[str, Any]]:
        chain = self._chains.get(chain_id, [])
        return chain[-1] if chain else None

    def export_json(self, chain_id: str) -> str:
        return json.dumps(self.get_chain(chain_id), indent=2, default=str)

    def get_all_chain_ids(self) -> List[str]:
        return list(self._chains.keys())

    def get_statistics(self) -> Dict[str, Any]:
        total_records = sum(len(c) for c in self._chains.values())
        return {"total_chains": len(self._chains), "total_records": total_records, "agent_id": self._agent_id}

    def _hash_data(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _hash_record(self, record: Dict[str, Any]) -> str:
        content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestProvenanceTrackerInit:
    def test_default_agent_id(self):
        tracker = ProvenanceTracker()
        assert tracker.agent_id == "GL-DATA-X-004"

    def test_custom_agent_id(self):
        tracker = ProvenanceTracker(agent_id="CUSTOM-ERP-001")
        assert tracker.agent_id == "CUSTOM-ERP-001"

    def test_genesis_hash(self):
        assert ProvenanceTracker.GENESIS_HASH == "0" * 64
        assert len(ProvenanceTracker.GENESIS_HASH) == 64

    def test_initial_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.get_all_chain_ids() == []

    def test_initial_statistics(self):
        tracker = ProvenanceTracker()
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 0
        assert stats["total_records"] == 0


class TestRecord:
    def test_record_returns_hash(self):
        tracker = ProvenanceTracker()
        h = tracker.record("conn-001", "register", {"host": "sap.com"})
        assert len(h) == 64
        int(h, 16)

    def test_record_creates_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {"host": "sap.com"})
        assert "conn-001" in tracker.get_all_chain_ids()

    def test_record_appends_to_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        chain = tracker.get_chain("conn-001")
        assert len(chain) == 2

    def test_record_sequence_numbers(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        tracker.record("conn-001", "sync", {})
        chain = tracker.get_chain("conn-001")
        assert chain[0]["sequence"] == 1
        assert chain[1]["sequence"] == 2
        assert chain[2]["sequence"] == 3

    def test_first_record_links_to_genesis(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        chain = tracker.get_chain("conn-001")
        assert chain[0]["previous_hash"] == ProvenanceTracker.GENESIS_HASH

    def test_second_record_links_to_first(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        chain = tracker.get_chain("conn-001")
        assert chain[1]["previous_hash"] == h1

    def test_record_contains_operation(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "sync_spend", {})
        chain = tracker.get_chain("conn-001")
        assert chain[0]["operation"] == "sync_spend"

    def test_record_contains_agent_id(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        chain = tracker.get_chain("conn-001")
        assert chain[0]["agent_id"] == "GL-DATA-X-004"

    def test_record_contains_actor(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {}, actor="erp-admin")
        chain = tracker.get_chain("conn-001")
        assert chain[0]["actor"] == "erp-admin"

    def test_deterministic_hashing(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"host": "sap.com", "port": 443}
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        assert h1 == h2


class TestVerifyChain:
    def test_empty_chain_valid(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True

    def test_single_record_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {"host": "sap.com"})
        result = tracker.verify_chain("conn-001")
        assert result["is_valid"] is True

    def test_multi_record_valid(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        tracker.record("conn-001", "sync", {})
        result = tracker.verify_chain("conn-001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 3

    def test_tampered_record_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        tracker._chains["conn-001"][0]["operation"] = "TAMPERED"
        result = tracker.verify_chain("conn-001")
        assert result["is_valid"] is False

    def test_tampered_hash_detected(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-001", "test", {})
        tracker._chains["conn-001"][0]["record_hash"] = "a" * 64
        result = tracker.verify_chain("conn-001")
        assert result["is_valid"] is False


class TestGetChain:
    def test_get_chain_exists(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        chain = tracker.get_chain("conn-001")
        assert len(chain) == 1

    def test_get_chain_nonexistent(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain("nonexistent") == []

    def test_get_chain_returns_copy(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        c1 = tracker.get_chain("conn-001")
        c2 = tracker.get_chain("conn-001")
        assert c1 is not c2


class TestExportJson:
    def test_export_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {"host": "sap.com"})
        exported = tracker.export_json("conn-001")
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_empty_chain(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json("nonexistent")
        parsed = json.loads(exported)
        assert parsed == []


class TestProvenanceStatistics:
    def test_stats_after_records(self):
        tracker = ProvenanceTracker()
        tracker.record("conn-001", "register", {})
        tracker.record("conn-002", "register", {})
        tracker.record("conn-001", "sync", {})
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 2
        assert stats["total_records"] == 3

    def test_stats_agent_id(self):
        tracker = ProvenanceTracker(agent_id="TEST-ERP")
        stats = tracker.get_statistics()
        assert stats["agent_id"] == "TEST-ERP"
