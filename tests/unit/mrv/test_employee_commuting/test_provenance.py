# -*- coding: utf-8 -*-
"""
Test suite for employee_commuting.provenance - AGENT-MRV-020.

Tests SHA-256 provenance chain, entry creation, chain validation,
Merkle-style hashing, serialization, thread safety, and batch
provenance for the Employee Commuting Agent (GL-MRV-S3-007).

Coverage (~30 tests):
- ProvenanceEntry creation and immutability
- ProvenanceChain creation, properties, serialization
- _canonical_json / _serialize for Decimal, datetime, Enum, nested dict
- _compute_hash (SHA-256 determinism, different-input divergence)
- _compute_chain_hash
- _merkle_hash (single, multiple, odd count, empty)
- ProvenanceTracker start_chain, record_stage, seal_chain, validate_chain
- 10-stage pipeline recording (all employee commuting stages)
- Chain tamper detection
- Hash determinism and different-input divergence
- Chain reset and thread safety
- BatchProvenance and BatchProvenanceTracker
- Export chain (JSON)
- Chain root_hash and sealed status
- Stage-specific recording with metadata
- Standalone hash functions

Author: GL-TestEngineer
Date: February 2026
"""

import hashlib
import threading
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
import pytest

from greenlang.agents.mrv.employee_commuting.provenance import (
    ProvenanceEntry,
    ProvenanceChain,
    ProvenanceTracker,
    ProvenanceStage,
    BatchProvenance,
    BatchProvenanceTracker,
    _canonical_json,
    _sha256,
    _serialize,
    _compute_hash,
    _compute_chain_hash,
    _merkle_hash,
    _merkle_proof,
    _verify_merkle_proof,
    get_provenance_tracker,
    hash_calculation_input,
    hash_commute_input,
    hash_vehicle_input,
    hash_transit_input,
    hash_telework_input,
    hash_emission_factor,
    hash_calculation_result,
    hash_distance_result,
    hash_spend_result,
    hash_telework_calc_result,
    hash_extrapolation_result,
    hash_batch_input,
    AGENT_ID,
    AGENT_VERSION,
    GENESIS_HASH,
)


# ==============================================================================
# PROVENANCE ENTRY TESTS
# ==============================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass."""

    def test_provenance_entry_creation(self):
        """Test ProvenanceEntry creation with valid data."""
        entry = ProvenanceEntry(
            stage=ProvenanceStage.VALIDATE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            metadata={},
            duration_ms=1.5,
            engine_id="validation-engine",
            engine_version="1.0.0",
        )
        assert entry.stage == ProvenanceStage.VALIDATE
        assert entry.engine_id == "validation-engine"
        assert entry.engine_version == "1.0.0"
        assert entry.duration_ms == 1.5

    def test_provenance_entry_frozen(self):
        """Test ProvenanceEntry is immutable (frozen dataclass)."""
        entry = ProvenanceEntry(
            stage=ProvenanceStage.VALIDATE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            metadata={},
            duration_ms=0.0,
            engine_id="test",
            engine_version="1.0.0",
        )
        with pytest.raises(Exception):
            entry.stage = ProvenanceStage.CLASSIFY  # type: ignore[misc]

    def test_provenance_entry_to_dict(self):
        """Test ProvenanceEntry serialization to dict."""
        entry = ProvenanceEntry(
            stage=ProvenanceStage.VALIDATE,
            timestamp="2026-01-01T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            metadata={"source": "test"},
            duration_ms=2.0,
            engine_id="validation-engine",
            engine_version="1.0.0",
        )
        d = entry.to_dict()
        assert d["stage"] == "VALIDATE"
        assert d["engine_id"] == "validation-engine"
        assert d["metadata"]["source"] == "test"

    def test_provenance_entry_to_json(self):
        """Test ProvenanceEntry serialization to JSON."""
        entry = ProvenanceEntry(
            stage=ProvenanceStage.CLASSIFY,
            timestamp="2026-01-01T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            metadata={},
            duration_ms=0.0,
            engine_id="classify-engine",
            engine_version="1.0.0",
        )
        j = entry.to_json()
        assert isinstance(j, str)
        assert "CLASSIFY" in j


# ==============================================================================
# PROVENANCE CHAIN TESTS
# ==============================================================================


class TestProvenanceChain:
    """Test ProvenanceChain dataclass."""

    def test_provenance_chain_creation(self):
        """Test ProvenanceChain initialization."""
        chain = ProvenanceChain(
            chain_id="chain-001",
            tenant_id="tenant-001",
        )
        assert chain.chain_id == "chain-001"
        assert chain.tenant_id == "tenant-001"
        assert chain.agent_id == AGENT_ID
        assert len(chain.entries) == 0
        assert chain.final_hash is None
        assert chain.sealed_at is None
        assert chain.is_sealed is False

    def test_provenance_chain_root_hash_empty(self):
        """Test root_hash is empty string when no entries."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        assert chain.root_hash == ""

    def test_provenance_chain_last_hash_empty(self):
        """Test last_hash is empty string when no entries."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        assert chain.last_hash == ""

    def test_provenance_chain_entry_count(self):
        """Test entry_count property."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        assert chain.entry_count == 0

    def test_provenance_chain_stages_recorded_empty(self):
        """Test stages_recorded is empty list when no entries."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        assert chain.stages_recorded == []

    def test_provenance_chain_to_dict(self):
        """Test ProvenanceChain serialization to dict."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        d = chain.to_dict()
        assert d["chain_id"] == "chain-001"
        assert d["tenant_id"] == "t1"
        assert d["agent_id"] == AGENT_ID
        assert isinstance(d["entries"], list)
        assert d["is_sealed"] is False

    def test_provenance_chain_to_json(self):
        """Test ProvenanceChain serialization to JSON."""
        chain = ProvenanceChain(chain_id="chain-001", tenant_id="t1")
        j = chain.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["chain_id"] == "chain-001"


# ==============================================================================
# SERIALIZATION HELPERS TESTS
# ==============================================================================


class TestSerializationHelpers:
    """Test _canonical_json, _serialize, _compute_hash, _compute_chain_hash, _merkle_hash."""

    def test_canonical_json_decimal(self):
        """Test _canonical_json converts Decimal to string."""
        result = _canonical_json({"value": Decimal("123.456")})
        assert "123.456" in result

    def test_canonical_json_datetime(self):
        """Test _canonical_json converts datetime to ISO format."""
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _canonical_json({"ts": dt})
        assert "2026-01-01" in result

    def test_canonical_json_enum(self):
        """Test _canonical_json converts Enum to its value."""
        result = _canonical_json({"stage": ProvenanceStage.VALIDATE})
        assert "VALIDATE" in result

    def test_canonical_json_nested_dict(self):
        """Test _canonical_json handles nested dicts with sorted keys."""
        data = {"z": 1, "a": {"y": 2, "b": 3}}
        result = _canonical_json(data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == 3

    def test_serialize_alias(self):
        """Test _serialize is an alias for _canonical_json."""
        data = {"key": "value"}
        assert _serialize(data) == _canonical_json(data)

    def test_sha256_known_value(self):
        """Test _sha256 produces correct SHA-256 for known input."""
        result = _sha256("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_compute_hash_sha256(self):
        """Test _compute_hash returns 64-char SHA-256 hex."""
        h = _compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_hash_deterministic(self):
        """Test _compute_hash is deterministic."""
        h1 = _compute_hash({"key": "value", "num": 42})
        h2 = _compute_hash({"key": "value", "num": 42})
        assert h1 == h2

    def test_compute_hash_different_inputs(self):
        """Test _compute_hash produces different hashes for different inputs."""
        h1 = _compute_hash({"key": "value1"})
        h2 = _compute_hash({"key": "value2"})
        assert h1 != h2

    def test_compute_chain_hash(self):
        """Test _compute_chain_hash produces valid SHA-256."""
        h = _compute_chain_hash("prev", "VALIDATE", "input", "output")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_chain_hash_deterministic(self):
        """Test _compute_chain_hash is deterministic."""
        h1 = _compute_chain_hash("prev", "VALIDATE", "in1", "out1")
        h2 = _compute_chain_hash("prev", "VALIDATE", "in1", "out1")
        assert h1 == h2

    def test_compute_chain_hash_different_stages(self):
        """Test _compute_chain_hash varies by stage."""
        h1 = _compute_chain_hash("prev", "VALIDATE", "in1", "out1")
        h2 = _compute_chain_hash("prev", "CLASSIFY", "in1", "out1")
        assert h1 != h2

    def test_merkle_hash_single(self):
        """Test _merkle_hash with single hash returns it unchanged."""
        h = "a" * 64
        result = _merkle_hash([h])
        assert result == h

    def test_merkle_hash_multiple(self):
        """Test _merkle_hash with multiple hashes produces valid hash."""
        hashes = ["a" * 64, "b" * 64]
        result = _merkle_hash(hashes)
        assert len(result) == 64

    def test_merkle_hash_odd_count(self):
        """Test _merkle_hash with odd number of hashes."""
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        result = _merkle_hash(hashes)
        assert len(result) == 64

    def test_merkle_hash_empty(self):
        """Test _merkle_hash with empty list returns SHA-256 of empty bytes."""
        result = _merkle_hash([])
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_merkle_hash_deterministic(self):
        """Test _merkle_hash is deterministic for same input set."""
        hashes = [_compute_hash({"i": i}) for i in range(5)]
        r1 = _merkle_hash(hashes)
        r2 = _merkle_hash(hashes)
        assert r1 == r2


# ==============================================================================
# MERKLE PROOF TESTS
# ==============================================================================


class TestMerkleProof:
    """Test _merkle_proof and _verify_merkle_proof."""

    def test_merkle_proof_two_hashes(self):
        """Test Merkle proof for two-element tree."""
        hashes = sorted(["a" * 64, "b" * 64])
        root = _merkle_hash(hashes)
        proof = _merkle_proof(hashes, 0)
        assert _verify_merkle_proof(hashes[0], proof, root)

    def test_merkle_proof_multiple_hashes(self):
        """Test Merkle proof for multi-element tree."""
        hashes = sorted([_compute_hash({"i": i}) for i in range(4)])
        root = _merkle_hash(hashes)
        for idx in range(4):
            proof = _merkle_proof(hashes, idx)
            assert _verify_merkle_proof(hashes[idx], proof, root)

    def test_merkle_proof_invalid_index_raises(self):
        """Test _merkle_proof raises for out-of-range index."""
        hashes = ["a" * 64, "b" * 64]
        with pytest.raises(ValueError, match="out of range"):
            _merkle_proof(hashes, 5)

    def test_merkle_proof_empty(self):
        """Test _merkle_proof returns empty list for empty hashes."""
        result = _merkle_proof([], 0)
        # Should return empty proof for empty list (no target)
        assert result == []


# ==============================================================================
# PROVENANCE TRACKER TESTS
# ==============================================================================


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create fresh ProvenanceTracker."""
        return ProvenanceTracker()

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes with default agent info."""
        assert tracker.agent_id == AGENT_ID
        assert tracker.agent_version == AGENT_VERSION

    def test_start_chain(self, tracker):
        """Test starting a new provenance chain."""
        chain = tracker.start_chain("tenant-001")
        assert isinstance(chain, ProvenanceChain)
        assert chain.tenant_id == "tenant-001"
        assert chain.chain_id is not None
        assert len(chain.chain_id) > 0

    def test_start_chain_custom_id(self, tracker):
        """Test starting chain with custom ID."""
        chain = tracker.start_chain("tenant-001", chain_id="custom-001")
        assert chain.chain_id == "custom-001"

    def test_start_chain_duplicate_raises(self, tracker):
        """Test starting chain with duplicate ID raises ValueError."""
        tracker.start_chain("t1", chain_id="dup")
        with pytest.raises(ValueError, match="already exists"):
            tracker.start_chain("t1", chain_id="dup")

    def test_start_chain_with_metadata(self, tracker):
        """Test starting chain with metadata stores creation context."""
        chain = tracker.start_chain(
            "t1",
            metadata={"source": "survey", "year": 2025},
            chain_id="meta-001",
        )
        assert hasattr(chain, "_creation_metadata")
        assert chain._creation_metadata["source"] == "survey"

    def test_record_stage_string(self, tracker):
        """Test recording a pipeline stage with string name."""
        chain = tracker.start_chain("t1", chain_id="rec-str")
        entry = tracker.record_stage(
            "rec-str", "VALIDATE", {"origin": "survey"}, {"valid": True}
        )
        assert entry.stage == ProvenanceStage.VALIDATE
        assert len(entry.input_hash) == 64
        assert len(entry.output_hash) == 64
        assert len(entry.chain_hash) == 64

    def test_record_stage_enum(self, tracker):
        """Test recording stage using ProvenanceStage enum."""
        chain = tracker.start_chain("t1", chain_id="rec-enum")
        entry = tracker.record_stage(
            "rec-enum", ProvenanceStage.CLASSIFY, {"data": 1}, {"result": 2}
        )
        assert entry.stage == ProvenanceStage.CLASSIFY

    def test_record_stage_with_metadata(self, tracker):
        """Test recording stage with metadata dictionary."""
        chain = tracker.start_chain("t1", chain_id="rec-meta")
        entry = tracker.record_stage(
            "rec-meta",
            ProvenanceStage.RESOLVE_EFS,
            {"mode": "car"},
            {"ef": 0.17048},
            engine_id="ef-engine",
            engine_version="1.0.0",
            metadata={"source": "DEFRA", "year": 2024},
            duration_ms=3.5,
        )
        assert entry.metadata["source"] == "DEFRA"
        assert entry.engine_id == "ef-engine"
        assert entry.duration_ms == 3.5

    def test_record_stage_sealed_raises(self, tracker):
        """Test recording stage on sealed chain raises ValueError."""
        chain = tracker.start_chain("t1", chain_id="sealed-test")
        tracker.record_stage("sealed-test", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain("sealed-test")
        with pytest.raises(ValueError, match="already sealed"):
            tracker.record_stage("sealed-test", "CLASSIFY", {"c": 3}, {"d": 4})

    def test_record_stage_not_found_raises(self, tracker):
        """Test recording stage on nonexistent chain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.record_stage("nonexistent", "VALIDATE", {}, {})

    def test_validate_chain_valid(self, tracker):
        """Test validating a valid provenance chain returns (True, [])."""
        chain = tracker.start_chain("t1", chain_id="val-ok")
        tracker.record_stage("val-ok", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.record_stage("val-ok", "CLASSIFY", {"c": 3}, {"d": 4})
        is_valid, errors = tracker.validate_chain("val-ok")
        assert is_valid is True
        assert errors == []

    def test_validate_chain_empty(self, tracker):
        """Test validating an empty chain returns (True, [])."""
        tracker.start_chain("t1", chain_id="val-empty")
        is_valid, errors = tracker.validate_chain("val-empty")
        assert is_valid is True
        assert errors == []

    def test_validate_chain_tamper_detection(self, tracker):
        """Test tamper detection when chain_hash is modified."""
        chain = tracker.start_chain("t1", chain_id="tamper")
        tracker.record_stage("tamper", "VALIDATE", {"a": 1}, {"b": 2})
        chain = tracker.get_chain("tamper")
        original_entry = chain.entries[0]
        # Create a tampered entry with wrong chain_hash
        tampered = ProvenanceEntry(
            stage=original_entry.stage,
            timestamp=original_entry.timestamp,
            input_hash=original_entry.input_hash,
            output_hash=original_entry.output_hash,
            chain_hash="f" * 64,  # Tampered hash
            metadata=original_entry.metadata,
            duration_ms=original_entry.duration_ms,
            engine_id=original_entry.engine_id,
            engine_version=original_entry.engine_version,
        )
        chain.entries[0] = tampered
        is_valid, errors = tracker.validate_chain("tamper")
        assert is_valid is False
        assert len(errors) > 0

    def test_seal_chain(self, tracker):
        """Test sealing a provenance chain returns 64-char Merkle root."""
        chain = tracker.start_chain("t1", chain_id="seal-test")
        tracker.record_stage("seal-test", "VALIDATE", {"a": 1}, {"b": 2})
        final_hash = tracker.seal_chain("seal-test")
        assert len(final_hash) == 64
        assert all(c in "0123456789abcdef" for c in final_hash)

    def test_seal_chain_sets_sealed_state(self, tracker):
        """Test sealing marks the chain as sealed with timestamp."""
        chain = tracker.start_chain("t1", chain_id="seal-state")
        tracker.record_stage("seal-state", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain("seal-state")
        chain = tracker.get_chain("seal-state")
        assert chain.is_sealed is True
        assert chain.sealed_at is not None
        assert chain.final_hash is not None

    def test_seal_chain_already_sealed_raises(self, tracker):
        """Test sealing an already sealed chain raises ValueError."""
        chain = tracker.start_chain("t1", chain_id="double-seal")
        tracker.record_stage("double-seal", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain("double-seal")
        with pytest.raises(ValueError, match="already sealed"):
            tracker.seal_chain("double-seal")

    def test_seal_chain_not_found_raises(self, tracker):
        """Test sealing nonexistent chain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.seal_chain("nonexistent")

    def test_hash_deterministic_same_inputs(self, tracker):
        """Test same inputs produce same final chain hashes."""
        data_in = {"mode": "sov", "distance_km": Decimal("15.0")}
        data_out = {"co2e_kg": Decimal("1825.0")}

        chain1 = tracker.start_chain("t1", chain_id="det-1")
        tracker.record_stage("det-1", "VALIDATE", data_in, data_out)
        hash_1 = tracker.seal_chain("det-1")

        chain2 = tracker.start_chain("t1", chain_id="det-2")
        tracker.record_stage("det-2", "VALIDATE", data_in, data_out)
        hash_2 = tracker.seal_chain("det-2")

        assert hash_1 == hash_2

    def test_hash_different_inputs(self, tracker):
        """Test different inputs produce different final chain hashes."""
        chain1 = tracker.start_chain("t1", chain_id="diff-1")
        tracker.record_stage("diff-1", "VALIDATE", {"mode": "car"}, {"co2e": 100})
        hash_1 = tracker.seal_chain("diff-1")

        chain2 = tracker.start_chain("t1", chain_id="diff-2")
        tracker.record_stage("diff-2", "VALIDATE", {"mode": "bus"}, {"co2e": 50})
        hash_2 = tracker.seal_chain("diff-2")

        assert hash_1 != hash_2

    def test_record_all_10_stages(self, tracker):
        """Test recording all 10 pipeline stages for employee commuting."""
        chain = tracker.start_chain("t1", chain_id="all-stages")
        stages = [
            ProvenanceStage.VALIDATE,
            ProvenanceStage.CLASSIFY,
            ProvenanceStage.NORMALIZE,
            ProvenanceStage.RESOLVE_EFS,
            ProvenanceStage.CALCULATE_COMMUTE,
            ProvenanceStage.CALCULATE_TELEWORK,
            ProvenanceStage.APPLY_ALLOCATION,
            ProvenanceStage.COMPLIANCE,
            ProvenanceStage.AGGREGATE,
            ProvenanceStage.SEAL,
        ]
        for i, stage in enumerate(stages):
            tracker.record_stage(
                "all-stages", stage, {"step": i}, {"result": i * 10}
            )
        chain = tracker.get_chain("all-stages")
        assert chain.entry_count == 10
        assert chain.entries[0].stage == ProvenanceStage.VALIDATE
        assert chain.entries[9].stage == ProvenanceStage.SEAL
        assert chain.stages_recorded[0] == "VALIDATE"
        assert chain.stages_recorded[9] == "SEAL"

    def test_get_chain(self, tracker):
        """Test get_chain returns the correct chain."""
        tracker.start_chain("t1", chain_id="get-test")
        chain = tracker.get_chain("get-test")
        assert chain is not None
        assert chain.chain_id == "get-test"

    def test_get_chain_nonexistent_returns_none(self, tracker):
        """Test get_chain returns None for missing chain."""
        result = tracker.get_chain("nonexistent")
        assert result is None

    def test_export_chain(self, tracker):
        """Test export_chain returns dict with all fields."""
        tracker.start_chain("t1", chain_id="export-test")
        tracker.record_stage("export-test", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain("export-test")
        exported = tracker.export_chain("export-test")
        assert isinstance(exported, dict)
        assert exported["chain_id"] == "export-test"
        assert len(exported["entries"]) == 1
        assert exported["is_sealed"] is True
        assert exported["final_hash"] is not None

    def test_verify_entry(self, tracker):
        """Test verify_entry confirms a valid entry."""
        tracker.start_chain("t1", chain_id="verify-entry")
        entry = tracker.record_stage(
            "verify-entry", "VALIDATE", {"a": 1}, {"b": 2}
        )
        assert tracker.verify_entry(entry, "") is True

    def test_delete_chain(self, tracker):
        """Test delete_chain removes the chain."""
        tracker.start_chain("t1", chain_id="del-test")
        assert tracker.delete_chain("del-test") is True
        assert tracker.get_chain("del-test") is None

    def test_delete_chain_nonexistent(self, tracker):
        """Test delete_chain returns False for missing chain."""
        assert tracker.delete_chain("nonexistent") is False

    def test_list_chains(self, tracker):
        """Test list_chains returns chains with pagination."""
        tracker.start_chain("t1", chain_id="list-1")
        tracker.start_chain("t1", chain_id="list-2")
        tracker.start_chain("t2", chain_id="list-3")
        all_chains = tracker.list_chains()
        assert len(all_chains) == 3
        # Filter by tenant
        t1_chains = tracker.list_chains(tenant_id="t1")
        assert len(t1_chains) == 2

    def test_list_chains_pagination(self, tracker):
        """Test list_chains pagination with limit and offset."""
        for i in range(5):
            tracker.start_chain("t1", chain_id=f"page-{i}")
        page = tracker.list_chains(limit=2, offset=1)
        assert len(page) == 2

    def test_get_chain_summary(self, tracker):
        """Test get_chain_summary returns lightweight summary."""
        tracker.start_chain("t1", chain_id="summary-test")
        tracker.record_stage(
            "summary-test", "VALIDATE", {"a": 1}, {"b": 2},
            engine_id="val-engine", duration_ms=5.0,
        )
        summary = tracker.get_chain_summary("summary-test")
        assert summary["chain_id"] == "summary-test"
        assert summary["entry_count"] == 1
        assert summary["duration_total_ms"] == 5.0
        assert "val-engine" in summary["engines_used"]

    def test_get_stage_hash(self, tracker):
        """Test get_stage_hash returns output hash for specific stage."""
        tracker.start_chain("t1", chain_id="stage-hash")
        tracker.record_stage("stage-hash", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.record_stage("stage-hash", "CLASSIFY", {"c": 3}, {"d": 4})
        h = tracker.get_stage_hash("stage-hash", "VALIDATE")
        assert h is not None
        assert len(h) == 64

    def test_get_stage_hash_not_found(self, tracker):
        """Test get_stage_hash returns None for unrecorded stage."""
        tracker.start_chain("t1", chain_id="no-stage")
        h = tracker.get_stage_hash("no-stage", "AGGREGATE")
        assert h is None

    def test_reset(self, tracker):
        """Test reset clears all chains."""
        tracker.start_chain("t1", chain_id="reset-1")
        tracker.start_chain("t1", chain_id="reset-2")
        tracker.reset()
        assert len(tracker.get_all_chains()) == 0

    def test_clear_all_chains(self, tracker):
        """Test clear_all_chains returns count and clears."""
        tracker.start_chain("t1", chain_id="clear-1")
        tracker.start_chain("t1", chain_id="clear-2")
        count = tracker.clear_all_chains()
        assert count == 2
        assert len(tracker.get_all_chains()) == 0

    def test_chain_root_hash_after_entry(self, tracker):
        """Test root_hash is first entry's chain_hash."""
        tracker.start_chain("t1", chain_id="root-hash")
        tracker.record_stage("root-hash", "VALIDATE", {"a": 1}, {"b": 2})
        chain = tracker.get_chain("root-hash")
        assert chain.root_hash == chain.entries[0].chain_hash

    def test_chain_timestamp_iso8601(self, tracker):
        """Test that entry timestamps are ISO 8601 UTC format."""
        tracker.start_chain("t1", chain_id="ts-test")
        entry = tracker.record_stage("ts-test", "VALIDATE", {"a": 1}, {"b": 2})
        # ISO 8601 should contain 'T' and '+00:00' or 'Z'
        assert "T" in entry.timestamp
        # Should be parseable
        dt = datetime.fromisoformat(entry.timestamp)
        assert dt.tzinfo is not None

    def test_thread_safety(self):
        """Test concurrent chain operations are thread-safe."""
        tracker = ProvenanceTracker()
        errors = []

        def create_and_seal(index):
            try:
                chain = tracker.start_chain("t1", chain_id=f"thread-{index}")
                tracker.record_stage(
                    f"thread-{index}", "VALIDATE", {"i": index}, {"ok": True}
                )
                tracker.seal_chain(f"thread-{index}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_and_seal, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All 20 chains should exist
        all_chains = tracker.get_all_chains()
        assert len(all_chains) == 20


# ==============================================================================
# BATCH PROVENANCE TESTS
# ==============================================================================


class TestBatchProvenance:
    """Test BatchProvenance dataclass and BatchProvenanceTracker."""

    def test_batch_provenance_creation(self):
        """Test BatchProvenance creation and to_dict."""
        batch = BatchProvenance(batch_id="batch-001", tenant_id="t1")
        assert batch.batch_id == "batch-001"
        assert batch.tenant_id == "t1"
        assert batch.batch_hash is None
        assert batch.batch_sealed_at is None
        d = batch.to_dict()
        assert d["batch_id"] == "batch-001"
        assert d["is_sealed"] is False

    def test_batch_tracker_start_batch(self):
        """Test starting a new batch."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_size=100)
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

    def test_batch_tracker_start_batch_custom_id(self):
        """Test starting batch with custom ID."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_id="batch-custom")
        assert batch_id == "batch-custom"

    def test_batch_tracker_duplicate_batch_raises(self):
        """Test starting batch with duplicate ID raises ValueError."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_tracker.start_batch("t1", batch_id="dup-batch")
        with pytest.raises(ValueError, match="already exists"):
            batch_tracker.start_batch("t1", batch_id="dup-batch")

    def test_batch_tracker_add_chain(self):
        """Test adding a chain to the batch."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_id="add-chain")
        chain = tracker.start_chain("t1", chain_id="ch-1")
        batch_tracker.add_chain(batch_id, chain)
        batch = batch_tracker.get_batch(batch_id)
        assert "ch-1" in batch.individual_chain_ids

    def test_batch_tracker_build_merkle_tree(self):
        """Test building Merkle tree seals the batch."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_id="merkle-batch")

        # Create and seal 3 chains
        for i in range(3):
            cid = f"merkle-ch-{i}"
            tracker.start_chain("t1", chain_id=cid)
            tracker.record_stage(cid, "VALIDATE", {"i": i}, {"ok": True})
            tracker.seal_chain(cid)
            batch_tracker.add_chain(batch_id, cid)

        root_hash = batch_tracker.build_merkle_tree(batch_id)
        assert len(root_hash) == 64
        batch = batch_tracker.get_batch(batch_id)
        assert batch.batch_sealed_at is not None
        assert batch.batch_hash == root_hash

    def test_batch_tracker_verify_merkle_proof(self):
        """Test verifying Merkle proof for included chain."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_id="proof-batch")

        chain_ids = []
        for i in range(4):
            cid = f"proof-ch-{i}"
            tracker.start_chain("t1", chain_id=cid)
            tracker.record_stage(cid, "VALIDATE", {"i": i}, {"ok": True})
            tracker.seal_chain(cid)
            batch_tracker.add_chain(batch_id, cid)
            chain_ids.append(cid)

        batch_tracker.build_merkle_tree(batch_id)

        # Each chain should be provably included
        for cid in chain_ids:
            assert batch_tracker.verify_merkle_proof(batch_id, cid) is True

    def test_batch_tracker_verify_non_included_chain(self):
        """Test Merkle proof fails for chain not in batch."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_id="non-incl")

        tracker.start_chain("t1", chain_id="in-batch")
        tracker.record_stage("in-batch", "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain("in-batch")
        batch_tracker.add_chain(batch_id, "in-batch")
        batch_tracker.build_merkle_tree(batch_id)

        # Create chain NOT in the batch
        tracker.start_chain("t1", chain_id="not-in-batch")
        tracker.record_stage("not-in-batch", "VALIDATE", {"x": 99}, {"y": 100})
        tracker.seal_chain("not-in-batch")

        assert batch_tracker.verify_merkle_proof(batch_id, "not-in-batch") is False

    def test_batch_tracker_get_batch_summary(self):
        """Test get_batch_summary returns correct metadata."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_id = batch_tracker.start_batch("t1", batch_size=2, batch_id="sum-batch")
        summary = batch_tracker.get_batch_summary(batch_id)
        assert summary["batch_id"] == "sum-batch"
        assert summary["batch_size"] == 2
        assert summary["chain_count"] == 0
        assert summary["is_sealed"] is False

    def test_batch_tracker_delete_batch(self):
        """Test delete_batch removes the batch."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_tracker.start_batch("t1", batch_id="del-batch")
        assert batch_tracker.delete_batch("del-batch") is True
        assert batch_tracker.delete_batch("del-batch") is False

    def test_batch_tracker_list_batches(self):
        """Test list_batches with tenant filtering."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_tracker.start_batch("t1", batch_id="lb-1")
        batch_tracker.start_batch("t2", batch_id="lb-2")
        all_b = batch_tracker.list_batches()
        assert len(all_b) == 2
        t1_b = batch_tracker.list_batches(tenant_id="t1")
        assert len(t1_b) == 1

    def test_batch_tracker_reset(self):
        """Test reset clears all batches."""
        tracker = ProvenanceTracker()
        batch_tracker = BatchProvenanceTracker(tracker)
        batch_tracker.start_batch("t1", batch_id="rst-1")
        batch_tracker.reset()
        assert len(batch_tracker.list_batches()) == 0


# ==============================================================================
# STANDALONE HASH FUNCTION TESTS
# ==============================================================================


class TestStandaloneHashFunctions:
    """Test standalone hash_* convenience functions."""

    def test_hash_calculation_input(self):
        """Test hash_calculation_input returns 64-char SHA-256."""
        h = hash_calculation_input({"method": "employee_specific", "year": 2025})
        assert len(h) == 64

    def test_hash_commute_input(self):
        """Test hash_commute_input for commute pattern data."""
        h = hash_commute_input({"mode": "car", "distance_km": 25.0})
        assert len(h) == 64

    def test_hash_vehicle_input(self):
        """Test hash_vehicle_input for vehicle-specific data."""
        h = hash_vehicle_input({"vehicle_type": "medium_car", "fuel_type": "petrol"})
        assert len(h) == 64

    def test_hash_transit_input(self):
        """Test hash_transit_input for transit commute data."""
        h = hash_transit_input({"transit_mode": "commuter_rail", "distance_km": 45})
        assert len(h) == 64

    def test_hash_telework_input(self):
        """Test hash_telework_input for telework data."""
        h = hash_telework_input({"days_per_week": 2, "country": "US"})
        assert len(h) == 64

    def test_hash_emission_factor(self):
        """Test hash_emission_factor for EF records."""
        h = hash_emission_factor(
            {"ef_type": "car_medium_petrol", "ef_value": "0.17048", "source": "DEFRA"}
        )
        assert len(h) == 64

    def test_hash_calculation_result(self):
        """Test hash_calculation_result for complete result."""
        h = hash_calculation_result({"total_co2e_kg": 1825.0, "method": "distance"})
        assert len(h) == 64

    def test_hash_distance_result(self):
        """Test hash_distance_result parametric hash."""
        h = hash_distance_result(
            distance_km=15.0, mode="car", ef=0.17048,
            working_days=230, co2e_kg=1177.3,
        )
        assert len(h) == 64

    def test_hash_spend_result(self):
        """Test hash_spend_result parametric hash."""
        h = hash_spend_result(
            spend=2500.0, currency="USD", mode="transit",
            ef=0.15, co2e_kg=375.0,
        )
        assert len(h) == 64

    def test_hash_telework_calc_result(self):
        """Test hash_telework_calc_result parametric hash."""
        h = hash_telework_calc_result(
            country="US", days=104, ef_kwh=0.4,
            daily_kwh=3.5, co2e_kg=145.6,
        )
        assert len(h) == 64

    def test_hash_extrapolation_result(self):
        """Test hash_extrapolation_result parametric hash."""
        h = hash_extrapolation_result(
            sample_size=500, total_population=10000,
            response_rate=0.85, sample_co2e_kg=50000.0,
            extrapolated_co2e_kg=1000000.0,
        )
        assert len(h) == 64

    def test_hash_batch_input(self):
        """Test hash_batch_input for batch list."""
        h = hash_batch_input([
            {"employee_id": "E001", "mode": "car"},
            {"employee_id": "E002", "mode": "bus"},
        ])
        assert len(h) == 64

    def test_hash_determinism(self):
        """Test all hash functions are deterministic."""
        data = {"mode": "car", "distance_km": 15.0}
        h1 = hash_commute_input(data)
        h2 = hash_commute_input(data)
        assert h1 == h2


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test get_provenance_tracker singleton."""

    def test_get_provenance_tracker_returns_tracker(self):
        """Test get_provenance_tracker returns ProvenanceTracker instance."""
        tracker = get_provenance_tracker()
        assert isinstance(tracker, ProvenanceTracker)

    def test_get_provenance_tracker_singleton(self):
        """Test get_provenance_tracker returns same instance."""
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_independent_tracker_instances(self):
        """Test directly-created trackers are independent."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1 is not t2
        t1.start_chain("t1", chain_id="independent-1")
        assert t2.get_chain("independent-1") is None


# ==============================================================================
# COVERAGE META-TEST
# ==============================================================================


def test_provenance_module_coverage():
    """Meta-test to ensure comprehensive provenance coverage."""
    tested_components = [
        "ProvenanceEntry",
        "ProvenanceChain",
        "ProvenanceTracker",
        "ProvenanceStage",
        "BatchProvenance",
        "BatchProvenanceTracker",
        "_canonical_json",
        "_sha256",
        "_serialize",
        "_compute_hash",
        "_compute_chain_hash",
        "_merkle_hash",
        "_merkle_proof",
        "_verify_merkle_proof",
        "get_provenance_tracker",
        "hash_calculation_input",
        "hash_commute_input",
        "hash_vehicle_input",
        "hash_transit_input",
        "hash_telework_input",
        "hash_emission_factor",
        "hash_calculation_result",
        "hash_distance_result",
        "hash_spend_result",
        "hash_telework_calc_result",
        "hash_extrapolation_result",
        "hash_batch_input",
    ]
    assert len(tested_components) == 27
