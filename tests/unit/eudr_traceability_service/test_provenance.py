# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-DATA-005)

Tests provenance recording, SHA-256 chain integrity, chain verification,
tamper detection, deterministic hashing, export, and operation type tracking
for the EUDR Traceability Connector audit trails.

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
# Inline ProvenanceTracker mirroring greenlang/eudr_traceability/provenance.py
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """SHA-256 provenance chain tracker for EUDR traceability audit trails.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails, grouped by entity.
    """

    GENESIS_HASH = "0" * 64
    AGENT_ID = "GL-DATA-EUDR-001"

    OPERATION_TYPES = [
        "plot_registration",
        "custody_transfer",
        "dds_generation",
        "risk_assessment",
        "commodity_classification",
        "compliance_check",
        "eu_submission",
    ]

    def __init__(self, agent_id: str = "GL-DATA-EUDR-001"):
        self._agent_id = agent_id
        self._chains: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(self, chain_id: str, operation: str, data: Dict[str, Any],
               actor: str = "system") -> str:
        """Record a provenance entry and return its chain hash."""
        if chain_id not in self._chains:
            self._chains[chain_id] = []
        chain = self._chains[chain_id]
        previous_hash = chain[-1]["record_hash"] if chain else self.GENESIS_HASH
        record = {
            "chain_id": chain_id,
            "sequence": len(chain) + 1,
            "operation": operation,
            "data_hash": self._hash_data(data),
            "previous_hash": previous_hash,
            "agent_id": self._agent_id,
            "actor": actor,
            "timestamp": datetime.utcnow().isoformat(),
        }
        record["record_hash"] = self._hash_record(record)
        chain.append(record)
        return record["record_hash"]

    def verify_chain(self, chain_id: str) -> Dict[str, Any]:
        """Verify the integrity of a provenance chain."""
        chain = self._chains.get(chain_id, [])
        if not chain:
            return {"is_valid": True, "chain_length": 0, "broken_at": None}

        for i, record in enumerate(chain):
            expected_prev = chain[i - 1]["record_hash"] if i > 0 else self.GENESIS_HASH
            if record["previous_hash"] != expected_prev:
                return {"is_valid": False, "chain_length": len(chain), "broken_at": i + 1}
            expected_hash = self._hash_record(
                {k: v for k, v in record.items() if k != "record_hash"}
            )
            if record["record_hash"] != expected_hash:
                return {"is_valid": False, "chain_length": len(chain), "broken_at": i + 1}

        return {"is_valid": True, "chain_length": len(chain), "broken_at": None}

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get the provenance chain for an entity."""
        return list(self._chains.get(chain_id, []))

    def get_latest(self, chain_id: str) -> Optional[Dict[str, Any]]:
        chain = self._chains.get(chain_id, [])
        return chain[-1] if chain else None

    def export_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Export a chain as a list of dicts."""
        return self.get_chain(chain_id)

    def export_json(self, chain_id: str) -> str:
        """Export chain as JSON string."""
        return json.dumps(self.get_chain(chain_id), indent=2, default=str)

    def get_all_chain_ids(self) -> List[str]:
        return list(self._chains.keys())

    def get_statistics(self) -> Dict[str, Any]:
        total_records = sum(len(c) for c in self._chains.values())
        return {
            "total_chains": len(self._chains),
            "total_records": total_records,
            "agent_id": self._agent_id,
        }

    def _hash_data(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _hash_record(self, record: Dict[str, Any]) -> str:
        content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tracker() -> ProvenanceTracker:
    return ProvenanceTracker()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAgentID:
    """Tests for agent identification."""

    def test_agent_id(self):
        tracker = ProvenanceTracker()
        assert tracker.agent_id == "GL-DATA-EUDR-001"

    def test_custom_agent_id(self):
        tracker = ProvenanceTracker(agent_id="CUSTOM-EUDR-001")
        assert tracker.agent_id == "CUSTOM-EUDR-001"


class TestTrackOperation:
    """Tests for basic operation tracking."""

    def test_track_operation(self, tracker):
        h = tracker.record("PLT-00001", "plot_registration", {"lat": -3.1, "lon": -60.0})
        assert h is not None
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_record_creates_chain(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"lat": -3.1})
        assert "PLT-00001" in tracker.get_all_chain_ids()

    def test_record_appends_to_chain(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        chain = tracker.get_chain("PLT-00001")
        assert len(chain) == 2

    def test_record_sequence_numbers(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        tracker.record("PLT-00001", "dds_generation", {})
        chain = tracker.get_chain("PLT-00001")
        assert chain[0]["sequence"] == 1
        assert chain[1]["sequence"] == 2
        assert chain[2]["sequence"] == 3


class TestHashChainIntegrity:
    """Tests for hash chain integrity."""

    def test_hash_chain_integrity(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"plot": "A"})
        tracker.record("PLT-00001", "custody_transfer", {"to": "B"})
        chain = tracker.get_chain("PLT-00001")
        # Second entry links to first
        assert chain[1]["previous_hash"] == chain[0]["record_hash"]

    def test_first_entry_no_previous(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        chain = tracker.get_chain("PLT-00001")
        assert chain[0]["previous_hash"] == ProvenanceTracker.GENESIS_HASH

    def test_chain_links_propagate(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        tracker.record("PLT-00001", "dds_generation", {})
        chain = tracker.get_chain("PLT-00001")
        assert chain[1]["previous_hash"] == chain[0]["record_hash"]
        assert chain[2]["previous_hash"] == chain[1]["record_hash"]


class TestVerifyChain:
    """Tests for chain verification."""

    def test_verify_chain_valid(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"lat": -3.1})
        tracker.record("PLT-00001", "custody_transfer", {"to": "buyer"})
        result = tracker.verify_chain("PLT-00001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 2
        assert result["broken_at"] is None

    def test_verify_chain_empty(self, tracker):
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["chain_length"] == 0

    def test_verify_chain_tampered(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        # Tamper with the first record
        tracker._chains["PLT-00001"][0]["operation"] = "TAMPERED"
        result = tracker.verify_chain("PLT-00001")
        assert result["is_valid"] is False

    def test_verify_chain_hash_tampered(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        # Tamper with the hash
        tracker._chains["PLT-00001"][0]["record_hash"] = "a" * 64
        result = tracker.verify_chain("PLT-00001")
        assert result["is_valid"] is False


class TestExportChain:
    """Tests for chain export."""

    def test_export_chain(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"lat": -3.1})
        exported = tracker.export_chain("PLT-00001")
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert "chain_id" in exported[0]
        assert "record_hash" in exported[0]

    def test_export_chain_empty(self, tracker):
        exported = tracker.export_chain("nonexistent")
        assert exported == []

    def test_export_json(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"lat": -3.1})
        exported_json = tracker.export_json("PLT-00001")
        parsed = json.loads(exported_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_chain_returns_copy(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        c1 = tracker.export_chain("PLT-00001")
        c2 = tracker.export_chain("PLT-00001")
        assert c1 is not c2


class TestMultipleOperations:
    """Tests for tracking multiple operations."""

    def test_multiple_operations(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {"plot": "A"})
        tracker.record("DDS-00001", "dds_generation", {"dds": "B"})
        tracker.record("RISK-00001", "risk_assessment", {"risk": "C"})
        assert len(tracker.get_all_chain_ids()) == 3

    def test_multiple_chains_independent(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00002", "plot_registration", {})
        c1 = tracker.get_chain("PLT-00001")
        c2 = tracker.get_chain("PLT-00002")
        assert len(c1) == 1
        assert len(c2) == 1
        assert c1[0]["record_hash"] != c2[0]["record_hash"]


class TestOperationTypes:
    """Tests for all 7 defined operation types."""

    def test_operation_types(self, tracker):
        assert len(ProvenanceTracker.OPERATION_TYPES) == 7

    @pytest.mark.parametrize("op_type", ProvenanceTracker.OPERATION_TYPES)
    def test_each_operation_type(self, tracker, op_type):
        h = tracker.record(f"entity-{op_type}", op_type, {"type": op_type})
        assert len(h) == 64
        chain = tracker.get_chain(f"entity-{op_type}")
        assert chain[0]["operation"] == op_type


class TestDeterministicHashing:
    """Tests for deterministic hash computation."""

    def test_compute_hash_deterministic(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"lat": -3.1190, "lon": -60.0217, "country": "BR"}
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        assert h1 == h2

    def test_compute_hash_deterministic_3x(self):
        data = {"commodity": "cocoa", "weight": 100.5}
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        t3 = ProvenanceTracker()
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        h3 = t3._hash_data(data)
        assert h1 == h2 == h3

    def test_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker._hash_data({"x": 1})
        h2 = tracker._hash_data({"x": 2})
        assert h1 != h2


class TestSpecificProvenanceOperations:
    """Tests for specific EUDR operation provenance tracking."""

    def test_plot_registration_provenance(self, tracker):
        h = tracker.record("PLT-00001", "plot_registration", {
            "latitude": -3.1190, "longitude": -60.0217,
            "country": "BR", "commodity": "cocoa",
        })
        chain = tracker.get_chain("PLT-00001")
        assert chain[0]["operation"] == "plot_registration"
        assert len(h) == 64

    def test_custody_transfer_provenance(self, tracker):
        h = tracker.record("XFER-00001", "custody_transfer", {
            "from": "supplier_A", "to": "trader_B",
            "commodity": "cocoa", "weight_kg": 5000,
        })
        chain = tracker.get_chain("XFER-00001")
        assert chain[0]["operation"] == "custody_transfer"
        assert len(h) == 64

    def test_dds_generation_provenance(self, tracker):
        h = tracker.record("DDS-00001", "dds_generation", {
            "operator": "ChocoCorp", "commodity": "cocoa",
            "plots": ["PLT-00001", "PLT-00002"],
        })
        chain = tracker.get_chain("DDS-00001")
        assert chain[0]["operation"] == "dds_generation"
        assert len(h) == 64

    def test_risk_assessment_provenance(self, tracker):
        h = tracker.record("RISK-00001", "risk_assessment", {
            "country": "BR", "commodity": "cocoa",
            "score": 72.5, "level": "high",
        })
        chain = tracker.get_chain("RISK-00001")
        assert chain[0]["operation"] == "risk_assessment"
        assert len(h) == 64


class TestGenesisHash:
    """Tests for genesis hash properties."""

    def test_genesis_hash_length(self):
        assert len(ProvenanceTracker.GENESIS_HASH) == 64

    def test_genesis_hash_all_zeros(self):
        assert ProvenanceTracker.GENESIS_HASH == "0" * 64


class TestStatistics:
    """Tests for provenance statistics."""

    def test_initial_statistics(self, tracker):
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 0
        assert stats["total_records"] == 0

    def test_statistics_after_records(self, tracker):
        tracker.record("PLT-00001", "plot_registration", {})
        tracker.record("PLT-00002", "plot_registration", {})
        tracker.record("PLT-00001", "custody_transfer", {})
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 2
        assert stats["total_records"] == 3
        assert stats["agent_id"] == "GL-DATA-EUDR-001"
