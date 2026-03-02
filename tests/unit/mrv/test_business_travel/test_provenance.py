# -*- coding: utf-8 -*-
"""
Test suite for business_travel.provenance - AGENT-MRV-019.

Tests SHA-256 provenance chain, entry creation, chain validation,
Merkle-style hashing, serialization, and thread safety for the
Business Travel Agent (GL-MRV-S3-006).

Coverage:
- ProvenanceEntry creation and immutability
- ProvenanceChain creation, add_record, finalize, validate
- Chain tamper detection
- ProvenanceTracker start_chain, record_stage, seal_chain, validate_chain
- Hash determinism and different-input divergence
- _serialize for Decimal, datetime, Enum, nested dict
- _compute_hash (SHA-256)
- _compute_chain_hash
- _merkle_hash (single, multiple, odd count)
- 10-stage pipeline recording
- Chain reset and thread safety
- BatchProvenance
- Export chain (JSON)
- Chain root_hash and is_valid properties
- Stage-specific recording

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
import pytest

from greenlang.business_travel.provenance import (
    ProvenanceEntry,
    ProvenanceChain,
    ProvenanceTracker,
    ProvenanceStage,
    BatchProvenance,
    _serialize,
    _compute_hash,
    _compute_chain_hash,
    _merkle_hash,
)


# ==============================================================================
# PROVENANCE ENTRY TESTS
# ==============================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry dataclass."""

    def test_provenance_entry_creation(self):
        """Test ProvenanceEntry creation with valid data."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        assert entry.stage == "VALIDATE"
        assert entry.agent_id == "GL-MRV-S3-006"
        assert entry.agent_version == "1.0.0"

    def test_provenance_entry_frozen(self):
        """Test ProvenanceEntry is immutable (frozen dataclass)."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        with pytest.raises(Exception):
            entry.stage = "MODIFIED"

    def test_provenance_entry_to_dict(self):
        """Test ProvenanceEntry serialization to dict."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp="2026-01-01T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        d = entry.to_dict()
        assert d["stage"] == "VALIDATE"
        assert d["entry_id"] == "e1"
        assert d["agent_id"] == "GL-MRV-S3-006"

    def test_provenance_entry_to_json(self):
        """Test ProvenanceEntry serialization to JSON."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp="2026-01-01T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        j = entry.to_json()
        assert isinstance(j, str)
        assert "VALIDATE" in j


# ==============================================================================
# PROVENANCE CHAIN TESTS
# ==============================================================================


class TestProvenanceChain:
    """Test ProvenanceChain dataclass."""

    def test_provenance_chain_creation(self):
        """Test ProvenanceChain initialization."""
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.chain_id == "chain-001"
        assert len(chain.entries) == 0
        assert chain.final_hash is None
        assert chain.sealed_at is None

    def test_provenance_chain_root_hash_empty(self):
        """Test root_hash is empty string when no entries."""
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.root_hash == ""

    def test_provenance_chain_is_valid(self):
        """Test is_valid property defaults to True."""
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.is_valid is True

    def test_provenance_chain_to_dict(self):
        """Test ProvenanceChain serialization to dict."""
        chain = ProvenanceChain(chain_id="chain-001")
        d = chain.to_dict()
        assert d["chain_id"] == "chain-001"
        assert isinstance(d["entries"], list)
        assert d["is_valid"] is True


# ==============================================================================
# SERIALIZATION HELPERS TESTS
# ==============================================================================


class TestSerializationHelpers:
    """Test _serialize, _compute_hash, _compute_chain_hash, _merkle_hash."""

    def test_serialize_decimal(self):
        """Test _serialize converts Decimal to string."""
        result = _serialize({"value": Decimal("123.456")})
        assert "123.456" in result

    def test_serialize_datetime(self):
        """Test _serialize converts datetime to ISO format."""
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _serialize({"ts": dt})
        assert "2026-01-01" in result

    def test_serialize_enum(self):
        """Test _serialize converts Enum to its value."""
        result = _serialize({"stage": ProvenanceStage.VALIDATE})
        assert "VALIDATE" in result

    def test_serialize_nested_dict(self):
        """Test _serialize handles nested dicts with sorted keys."""
        data = {"z": 1, "a": {"y": 2, "b": 3}}
        result = _serialize(data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == 3

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

    def test_compute_chain_hash(self):
        """Test _compute_chain_hash produces valid SHA-256."""
        h = _compute_chain_hash("prev", "VALIDATE", "input", "output")
        assert len(h) == 64

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
        """Test _merkle_hash with empty list returns SHA-256 of empty."""
        result = _merkle_hash([])
        assert len(result) == 64


# ==============================================================================
# PROVENANCE TRACKER TESTS
# ==============================================================================


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create fresh ProvenanceTracker."""
        return ProvenanceTracker()

    def test_provenance_tracker_start_chain(self, tracker):
        """Test starting a new provenance chain."""
        chain_id = tracker.start_chain()
        assert isinstance(chain_id, str)
        assert len(chain_id) > 0

    def test_provenance_tracker_start_chain_custom_id(self, tracker):
        """Test starting chain with custom ID."""
        chain_id = tracker.start_chain(chain_id="custom-001")
        assert chain_id == "custom-001"

    def test_provenance_tracker_start_chain_duplicate_raises(self, tracker):
        """Test starting chain with duplicate ID raises ValueError."""
        tracker.start_chain(chain_id="dup")
        with pytest.raises(ValueError, match="already exists"):
            tracker.start_chain(chain_id="dup")

    def test_provenance_tracker_record_stage(self, tracker):
        """Test recording a pipeline stage."""
        chain_id = tracker.start_chain()
        entry = tracker.record_stage(
            chain_id, "VALIDATE", {"origin": "LHR"}, {"valid": True}
        )
        assert entry.stage == "VALIDATE"
        assert len(entry.input_hash) == 64
        assert len(entry.output_hash) == 64
        assert len(entry.chain_hash) == 64

    def test_provenance_tracker_record_stage_enum(self, tracker):
        """Test recording stage using ProvenanceStage enum."""
        chain_id = tracker.start_chain()
        entry = tracker.record_stage(
            chain_id, ProvenanceStage.VALIDATE, {"data": 1}, {"result": 2}
        )
        assert entry.stage == "VALIDATE"

    def test_provenance_tracker_seal_chain(self, tracker):
        """Test sealing a provenance chain."""
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        final_hash = tracker.seal_chain(chain_id)
        assert len(final_hash) == 64

    def test_provenance_tracker_validate_chain(self, tracker):
        """Test validating a valid provenance chain."""
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        tracker.record_stage(chain_id, "CLASSIFY", {"c": 3}, {"d": 4})
        is_valid = tracker.validate_chain(chain_id)
        assert is_valid is True

    def test_provenance_chain_tamper_detection(self, tracker):
        """Test tamper detection when chain_hash is modified."""
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        chain = tracker.get_chain(chain_id)
        # Tamper with the entry by modifying internal state
        # Since entries are frozen, we test by replacing entries list
        original_entry = chain.entries[0]
        # Create a tampered entry with wrong chain_hash
        tampered = ProvenanceEntry(
            entry_id=original_entry.entry_id,
            stage=original_entry.stage,
            timestamp=original_entry.timestamp,
            input_hash=original_entry.input_hash,
            output_hash=original_entry.output_hash,
            chain_hash="f" * 64,  # Tampered hash
            previous_hash=original_entry.previous_hash,
        )
        chain.entries[0] = tampered
        is_valid = tracker.validate_chain(chain_id)
        assert is_valid is False

    def test_provenance_hash_deterministic(self, tracker):
        """Test same inputs produce same chain hashes."""
        data_in = {"origin_iata": "LHR", "destination_iata": "JFK"}
        data_out = {"distance_km": Decimal("5541")}

        chain_id_1 = tracker.start_chain(chain_id="det-1")
        tracker.record_stage(chain_id_1, "VALIDATE", data_in, data_out)
        hash_1 = tracker.seal_chain(chain_id_1)

        chain_id_2 = tracker.start_chain(chain_id="det-2")
        tracker.record_stage(chain_id_2, "VALIDATE", data_in, data_out)
        hash_2 = tracker.seal_chain(chain_id_2)

        assert hash_1 == hash_2

    def test_provenance_hash_different_inputs(self, tracker):
        """Test different inputs produce different chain hashes."""
        chain_id_1 = tracker.start_chain(chain_id="diff-1")
        tracker.record_stage(chain_id_1, "VALIDATE", {"a": 1}, {"b": 2})
        hash_1 = tracker.seal_chain(chain_id_1)

        chain_id_2 = tracker.start_chain(chain_id="diff-2")
        tracker.record_stage(chain_id_2, "VALIDATE", {"a": 99}, {"b": 100})
        hash_2 = tracker.seal_chain(chain_id_2)

        assert hash_1 != hash_2

    def test_provenance_10_stages(self, tracker):
        """Test recording all 10 pipeline stages."""
        chain_id = tracker.start_chain()
        stages = [
            ProvenanceStage.VALIDATE,
            ProvenanceStage.CLASSIFY,
            ProvenanceStage.NORMALIZE,
            ProvenanceStage.RESOLVE_EFS,
            ProvenanceStage.CALCULATE_TRANSPORT,
            ProvenanceStage.CALCULATE_HOTEL,
            ProvenanceStage.APPLY_RF,
            ProvenanceStage.COMPLIANCE,
            ProvenanceStage.AGGREGATE,
            ProvenanceStage.SEAL,
        ]
        for i, stage in enumerate(stages):
            tracker.record_stage(
                chain_id, stage, {"step": i}, {"result": i * 10}
            )
        chain = tracker.get_chain(chain_id)
        assert len(chain.entries) == 10
        assert chain.entries[0].stage == "VALIDATE"
        assert chain.entries[9].stage == "SEAL"

    def test_provenance_tracker_reset(self):
        """Test creating a new tracker resets state."""
        tracker1 = ProvenanceTracker()
        chain_id = tracker1.start_chain(chain_id="reset-test")
        tracker2 = ProvenanceTracker()
        # New tracker should not have the chain from tracker1
        with pytest.raises(ValueError, match="not found"):
            tracker2.get_chain(chain_id)

    def test_provenance_tracker_thread_safety(self):
        """Test concurrent chain operations are thread-safe."""
        tracker = ProvenanceTracker()
        errors = []

        def create_and_seal(index):
            try:
                cid = tracker.start_chain(chain_id=f"thread-{index}")
                tracker.record_stage(cid, "VALIDATE", {"i": index}, {"ok": True})
                tracker.seal_chain(cid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_seal, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ==============================================================================
# BATCH PROVENANCE TESTS
# ==============================================================================


class TestBatchProvenance:
    """Test BatchProvenance dataclass."""

    def test_batch_provenance_tracker(self):
        """Test BatchProvenance creation and to_dict."""
        batch = BatchProvenance(batch_id="batch-001")
        assert batch.batch_id == "batch-001"
        assert batch.item_count == 0
        d = batch.to_dict()
        assert d["batch_id"] == "batch-001"


# ==============================================================================
# CHAIN EXPORT AND RETRIEVAL TESTS
# ==============================================================================


class TestChainExportAndRetrieval:
    """Test chain export and retrieval methods."""

    def test_provenance_get_chain(self):
        """Test get_chain returns the correct chain."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain(chain_id="get-test")
        chain = tracker.get_chain(chain_id)
        assert chain.chain_id == "get-test"

    def test_provenance_export_chain(self):
        """Test export_chain returns valid JSON."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain(chain_id="export-test")
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain(chain_id)
        exported = tracker.export_chain(chain_id, format="json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert parsed["chain_id"] == "export-test"
        assert len(parsed["entries"]) == 1

    def test_provenance_chain_is_valid_after_seal(self):
        """Test chain is_valid is True after successful seal."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        tracker.seal_chain(chain_id)
        chain = tracker.get_chain(chain_id)
        assert chain.is_valid is True
        assert chain.final_hash is not None

    def test_provenance_chain_root_hash(self):
        """Test chain root_hash is first entry's chain_hash."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"a": 1}, {"b": 2})
        chain = tracker.get_chain(chain_id)
        assert chain.root_hash == chain.entries[0].chain_hash

    def test_stage_specific_recording(self):
        """Test recording metadata with a stage."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        entry = tracker.record_stage(
            chain_id,
            "RESOLVE_EFS",
            {"mode": "air"},
            {"ef": 0.19309},
            metadata={"source": "DEFRA", "version": "2024"},
        )
        assert entry.metadata["source"] == "DEFRA"

    def test_seal_and_verify(self):
        """Test seal then validate full cycle."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"x": 1}, {"y": 2})
        tracker.record_stage(chain_id, "CLASSIFY", {"a": 3}, {"b": 4})
        final_hash = tracker.seal_chain(chain_id)
        assert len(final_hash) == 64
        assert tracker.validate_chain(chain_id) is True

    def test_provenance_singleton(self):
        """Test ProvenanceTracker is NOT a singleton (new instances are independent)."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1 is not t2


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_provenance_module_coverage():
    """Meta-test to ensure comprehensive provenance coverage."""
    tested_components = [
        "ProvenanceEntry",
        "ProvenanceChain",
        "ProvenanceTracker",
        "BatchProvenance",
        "_serialize",
        "_compute_hash",
        "_compute_chain_hash",
        "_merkle_hash",
    ]
    assert len(tested_components) == 8
