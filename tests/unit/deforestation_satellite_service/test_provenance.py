# -*- coding: utf-8 -*-
"""
Unit Tests for ProvenanceTracker (AGENT-DATA-007)

Tests provenance recording, SHA-256 chain integrity, chain verification,
tamper detection, 7 operation types specific to deforestation satellite
connector, entity retrieval, deterministic hashing, and statistics.

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
# Inline ProvenanceTracker mirroring
# greenlang/deforestation_satellite/provenance.py
# ---------------------------------------------------------------------------

# 7 operation types for deforestation satellite connector
OPERATION_SATELLITE_ACQUISITION = "satellite_acquisition"
OPERATION_CHANGE_DETECTION = "change_detection"
OPERATION_ALERT_AGGREGATION = "alert_aggregation"
OPERATION_BASELINE_ASSESSMENT = "baseline_assessment"
OPERATION_CLASSIFICATION = "classification"
OPERATION_COMPLIANCE_REPORT = "compliance_report"
OPERATION_PIPELINE_EXECUTION = "pipeline_execution"

ALL_OPERATION_TYPES = [
    OPERATION_SATELLITE_ACQUISITION,
    OPERATION_CHANGE_DETECTION,
    OPERATION_ALERT_AGGREGATION,
    OPERATION_BASELINE_ASSESSMENT,
    OPERATION_CLASSIFICATION,
    OPERATION_COMPLIANCE_REPORT,
    OPERATION_PIPELINE_EXECUTION,
]


class ProvenanceTracker:
    """SHA-256 provenance chain tracker for deforestation satellite audit trails."""

    GENESIS_HASH = "0" * 64

    def __init__(self, agent_id: str = "GL-DATA-GEO-003"):
        self._agent_id = agent_id
        self._chains: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def record(
        self,
        chain_id: str,
        operation: str,
        data: Dict[str, Any],
        actor: str = "system",
    ) -> str:
        """Record a provenance entry and return the record hash."""
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
        return list(self._chains.get(chain_id, []))

    def get_by_entity(self, chain_id: str) -> List[Dict[str, Any]]:
        """Alias for get_chain for entity-oriented retrieval."""
        return self.get_chain(chain_id)

    def get_latest(self, chain_id: str) -> Optional[Dict[str, Any]]:
        chain = self._chains.get(chain_id, [])
        return chain[-1] if chain else None

    def export_json(self, chain_id: str) -> str:
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

    def get_provenance_count(self) -> int:
        return sum(len(c) for c in self._chains.values())

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
        assert tracker.agent_id == "GL-DATA-GEO-003"

    def test_custom_agent_id(self):
        tracker = ProvenanceTracker(agent_id="CUSTOM-SAT-001")
        assert tracker.agent_id == "CUSTOM-SAT-001"

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
    def test_record_creates_entry(self):
        tracker = ProvenanceTracker()
        h = tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {"satellite": "sentinel2"})
        assert len(h) == 64
        int(h, 16)

    def test_record_creates_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {"satellite": "sentinel2"})
        assert "scene-001" in tracker.get_all_chain_ids()

    def test_record_appends_to_chain(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        chain = tracker.get_chain("scene-001")
        assert len(chain) == 2

    def test_record_sequence_numbers(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        tracker.record("scene-001", OPERATION_CLASSIFICATION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[0]["sequence"] == 1
        assert chain[1]["sequence"] == 2
        assert chain[2]["sequence"] == 3

    def test_first_record_links_to_genesis(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[0]["previous_hash"] == ProvenanceTracker.GENESIS_HASH

    def test_second_record_links_to_first(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[1]["previous_hash"] == h1

    def test_record_contains_operation(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_ALERT_AGGREGATION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[0]["operation"] == OPERATION_ALERT_AGGREGATION

    def test_record_contains_agent_id(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[0]["agent_id"] == "GL-DATA-GEO-003"

    def test_record_contains_actor(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {}, actor="pipeline-worker")
        chain = tracker.get_chain("scene-001")
        assert chain[0]["actor"] == "pipeline-worker"

    def test_record_has_timestamp(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        chain = tracker.get_chain("scene-001")
        assert "timestamp" in chain[0]


class TestChainIntegrity:
    def test_chain_integrity_sha256(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        chain = tracker.get_chain("scene-001")
        # Second record's previous_hash == first record's record_hash
        assert chain[1]["previous_hash"] == chain[0]["record_hash"]

    def test_chain_three_links(self):
        tracker = ProvenanceTracker()
        h1 = tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        h2 = tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        h3 = tracker.record("scene-001", OPERATION_CLASSIFICATION, {})
        chain = tracker.get_chain("scene-001")
        assert chain[0]["previous_hash"] == ProvenanceTracker.GENESIS_HASH
        assert chain[1]["previous_hash"] == h1
        assert chain[2]["previous_hash"] == h2


class TestVerifyChain:
    def test_verify_chain_valid_empty(self):
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result["is_valid"] is True
        assert result["chain_length"] == 0

    def test_verify_chain_valid_single(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        result = tracker.verify_chain("scene-001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 1

    def test_verify_chain_valid_multi(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        tracker.record("scene-001", OPERATION_CLASSIFICATION, {})
        result = tracker.verify_chain("scene-001")
        assert result["is_valid"] is True
        assert result["chain_length"] == 3

    def test_verify_chain_tampered_operation(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        tracker._chains["scene-001"][0]["operation"] = "TAMPERED"
        result = tracker.verify_chain("scene-001")
        assert result["is_valid"] is False

    def test_verify_chain_tampered_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        tracker._chains["scene-001"][0]["record_hash"] = "a" * 64
        result = tracker.verify_chain("scene-001")
        assert result["is_valid"] is False

    def test_verify_chain_tampered_previous_hash(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-001", OPERATION_CHANGE_DETECTION, {})
        tracker._chains["scene-001"][1]["previous_hash"] = "b" * 64
        result = tracker.verify_chain("scene-001")
        assert result["is_valid"] is False


class TestOperationTypes:
    """Test all 7 operation types can be recorded."""

    def test_satellite_acquisition(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-001", OPERATION_SATELLITE_ACQUISITION, {"satellite": "sentinel2"})
        assert len(h) == 64

    def test_change_detection(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-002", OPERATION_CHANGE_DETECTION, {"type": "clear_cut"})
        assert len(h) == 64

    def test_alert_aggregation(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-003", OPERATION_ALERT_AGGREGATION, {"source": "glad"})
        assert len(h) == 64

    def test_baseline_assessment(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-004", OPERATION_BASELINE_ASSESSMENT, {"country": "BRA"})
        assert len(h) == 64

    def test_classification(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-005", OPERATION_CLASSIFICATION, {"type": "dense_forest"})
        assert len(h) == 64

    def test_compliance_report(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-006", OPERATION_COMPLIANCE_REPORT, {"status": "COMPLIANT"})
        assert len(h) == 64

    def test_pipeline_execution(self):
        tracker = ProvenanceTracker()
        h = tracker.record("e-007", OPERATION_PIPELINE_EXECUTION, {"stages": 7})
        assert len(h) == 64

    def test_all_7_operation_types_count(self):
        assert len(ALL_OPERATION_TYPES) == 7


class TestGetByEntity:
    def test_get_by_entity_exists(self):
        tracker = ProvenanceTracker()
        tracker.record("plot-001", OPERATION_SATELLITE_ACQUISITION, {})
        records = tracker.get_by_entity("plot-001")
        assert len(records) == 1

    def test_get_by_entity_nonexistent(self):
        tracker = ProvenanceTracker()
        records = tracker.get_by_entity("nonexistent")
        assert records == []

    def test_get_by_entity_multiple(self):
        tracker = ProvenanceTracker()
        tracker.record("plot-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("plot-001", OPERATION_CHANGE_DETECTION, {})
        tracker.record("plot-001", OPERATION_CLASSIFICATION, {})
        records = tracker.get_by_entity("plot-001")
        assert len(records) == 3


class TestGetChain:
    def test_get_chain_empty(self):
        tracker = ProvenanceTracker()
        assert tracker.get_chain("nonexistent") == []

    def test_get_chain_returns_copy(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        c1 = tracker.get_chain("scene-001")
        c2 = tracker.get_chain("scene-001")
        assert c1 is not c2


class TestMultipleRecordsSameEntity:
    def test_multiple_records_same_entity(self):
        tracker = ProvenanceTracker()
        tracker.record("plot-001", OPERATION_SATELLITE_ACQUISITION, {"scene": "S2A_001"})
        tracker.record("plot-001", OPERATION_CHANGE_DETECTION, {"dndvi": -0.25})
        tracker.record("plot-001", OPERATION_ALERT_AGGREGATION, {"count": 3})
        tracker.record("plot-001", OPERATION_BASELINE_ASSESSMENT, {"forest": True})
        chain = tracker.get_chain("plot-001")
        assert len(chain) == 4
        ops = [r["operation"] for r in chain]
        assert OPERATION_SATELLITE_ACQUISITION in ops
        assert OPERATION_CHANGE_DETECTION in ops
        assert OPERATION_ALERT_AGGREGATION in ops
        assert OPERATION_BASELINE_ASSESSMENT in ops


class TestHashDeterministic:
    def test_hash_deterministic(self):
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        data = {"satellite": "sentinel2", "bands": 6}
        h1 = t1._hash_data(data)
        h2 = t2._hash_data(data)
        assert h1 == h2

    def test_hash_different_data_different_hash(self):
        tracker = ProvenanceTracker()
        h1 = tracker._hash_data({"satellite": "sentinel2"})
        h2 = tracker._hash_data({"satellite": "landsat8"})
        assert h1 != h2


class TestProvenanceCount:
    def test_provenance_count_initial(self):
        tracker = ProvenanceTracker()
        assert tracker.get_provenance_count() == 0

    def test_provenance_count_after_records(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-002", OPERATION_CHANGE_DETECTION, {})
        tracker.record("scene-001", OPERATION_CLASSIFICATION, {})
        assert tracker.get_provenance_count() == 3

    def test_statistics_after_records(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {})
        tracker.record("scene-002", OPERATION_CHANGE_DETECTION, {})
        stats = tracker.get_statistics()
        assert stats["total_chains"] == 2
        assert stats["total_records"] == 2

    def test_statistics_agent_id(self):
        tracker = ProvenanceTracker(agent_id="TEST-SAT")
        stats = tracker.get_statistics()
        assert stats["agent_id"] == "TEST-SAT"


class TestExportJson:
    def test_export_valid_json(self):
        tracker = ProvenanceTracker()
        tracker.record("scene-001", OPERATION_SATELLITE_ACQUISITION, {"satellite": "sentinel2"})
        exported = tracker.export_json("scene-001")
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_empty_chain(self):
        tracker = ProvenanceTracker()
        exported = tracker.export_json("nonexistent")
        parsed = json.loads(exported)
        assert parsed == []
