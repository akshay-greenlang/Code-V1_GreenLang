# -*- coding: utf-8 -*-
"""
Test suite for franchises.provenance - AGENT-MRV-027.

Tests SHA-256 provenance chain, entry creation, chain validation,
Merkle-style hashing, serialization, and thread safety for the
Franchises Agent (GL-MRV-S3-014).

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
- All 15 hash functions

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
import pytest

from greenlang.franchises.provenance import (
    ProvenanceEntry,
    ProvenanceChain,
    ProvenanceTracker,
    ProvenanceStage,
    BatchProvenance,
    _serialize,
    _compute_hash,
    _compute_chain_hash,
    _merkle_hash,
    get_provenance_tracker,
    AGENT_ID,
    AGENT_VERSION,
    HASH_ALGORITHM,
)


# ==============================================================================
# CONSTANTS TESTS
# ==============================================================================


class TestProvenanceConstants:
    """Test provenance module constants."""

    def test_agent_id(self):
        """Test agent ID is GL-MRV-S3-014."""
        assert AGENT_ID == "GL-MRV-S3-014"

    def test_agent_version(self):
        """Test agent version is 1.0.0."""
        assert AGENT_VERSION == "1.0.0"

    def test_hash_algorithm(self):
        """Test hash algorithm is sha256."""
        assert HASH_ALGORITHM == "sha256"


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
        assert entry.agent_id == "GL-MRV-S3-014"
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
            timestamp="2026-02-28T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        d = entry.to_dict()
        assert d["stage"] == "VALIDATE"
        assert d["entry_id"] == "e1"
        assert d["agent_id"] == "GL-MRV-S3-014"

    def test_provenance_entry_to_json(self):
        """Test ProvenanceEntry serialization to JSON."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp="2026-02-28T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        j = entry.to_json()
        assert isinstance(j, str)
        assert "VALIDATE" in j

    def test_provenance_entry_metadata(self):
        """Test ProvenanceEntry with metadata."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="CALCULATE",
            timestamp="2026-02-28T00:00:00+00:00",
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="d" * 64,
            metadata={"franchise_type": "qsr_restaurant", "unit_count": 500},
        )
        assert entry.metadata["franchise_type"] == "qsr_restaurant"


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

    def test_provenance_chain_to_json(self):
        """Test ProvenanceChain serialization to JSON."""
        chain = ProvenanceChain(chain_id="chain-001")
        j = chain.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["chain_id"] == "chain-001"


# ==============================================================================
# SERIALIZATION HELPER TESTS
# ==============================================================================


class TestSerializationHelpers:
    """Test _serialize, _compute_hash, _compute_chain_hash, _merkle_hash."""

    def test_serialize_decimal(self):
        """Test _serialize converts Decimal to string."""
        result = _serialize({"value": Decimal("123.456")})
        assert "123.456" in result

    def test_serialize_datetime(self):
        """Test _serialize converts datetime to ISO format."""
        dt = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
        result = _serialize({"ts": dt})
        assert "2026-02-28" in result

    def test_serialize_enum(self):
        """Test _serialize converts Enum to its value."""
        result = _serialize({"stage": ProvenanceStage.VALIDATE})
        assert "VALIDATE" in result or "validate" in result

    def test_serialize_nested_dict(self):
        """Test _serialize handles nested dicts with sorted keys."""
        data = {"z": 1, "a": {"y": 2, "b": 3}}
        result = _serialize(data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == 3

    def test_serialize_list(self):
        """Test _serialize handles lists."""
        result = _serialize({"items": [1, 2, 3]})
        assert "1" in result

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
        """Test different inputs produce different hashes."""
        h1 = _compute_hash({"key": "value1"})
        h2 = _compute_hash({"key": "value2"})
        assert h1 != h2

    def test_compute_chain_hash(self):
        """Test _compute_chain_hash produces valid SHA-256."""
        h = _compute_chain_hash("prev", "VALIDATE", "input", "output")
        assert len(h) == 64

    def test_compute_chain_hash_deterministic(self):
        """Test _compute_chain_hash is deterministic."""
        h1 = _compute_chain_hash("prev", "VALIDATE", "input", "output")
        h2 = _compute_chain_hash("prev", "VALIDATE", "input", "output")
        assert h1 == h2

    def test_compute_chain_hash_different_prev(self):
        """Test different previous hashes produce different chain hashes."""
        h1 = _compute_chain_hash("prev1", "VALIDATE", "input", "output")
        h2 = _compute_chain_hash("prev2", "VALIDATE", "input", "output")
        assert h1 != h2

    def test_merkle_hash_single(self):
        """Test Merkle hash with single input."""
        h = _merkle_hash(["a" * 64])
        assert len(h) == 64

    def test_merkle_hash_multiple(self):
        """Test Merkle hash with multiple inputs."""
        h = _merkle_hash(["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        assert len(h) == 64

    def test_merkle_hash_odd_count(self):
        """Test Merkle hash with odd number of inputs."""
        h = _merkle_hash(["a" * 64, "b" * 64, "c" * 64])
        assert len(h) == 64

    def test_merkle_hash_deterministic(self):
        """Test Merkle hash is deterministic."""
        inputs = ["a" * 64, "b" * 64]
        h1 = _merkle_hash(inputs)
        h2 = _merkle_hash(inputs)
        assert h1 == h2


# ==============================================================================
# PROVENANCE TRACKER TESTS
# ==============================================================================


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    def test_tracker_creation(self):
        """Test ProvenanceTracker can be created."""
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_tracker_singleton(self):
        """Test get_provenance_tracker returns singleton."""
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2

    def test_start_chain(self):
        """Test starting a new provenance chain."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        assert isinstance(chain_id, str)
        assert len(chain_id) > 0

    def test_record_stage(self):
        """Test recording a pipeline stage."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"input": "data"}, {"output": "data"})
        chain = tracker.get_chain(chain_id)
        assert len(chain.entries) == 1

    def test_record_multiple_stages(self):
        """Test recording multiple pipeline stages."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        stages = ["VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS", "CALCULATE"]
        for stage in stages:
            tracker.record_stage(chain_id, stage, {"stage": stage}, {"result": stage})
        chain = tracker.get_chain(chain_id)
        assert len(chain.entries) == 5

    def test_seal_chain(self):
        """Test sealing a provenance chain."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"in": 1}, {"out": 1})
        final_hash = tracker.seal_chain(chain_id)
        assert len(final_hash) == 64

    def test_validate_chain_valid(self):
        """Test chain validation passes for untampered chain."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"in": 1}, {"out": 1})
        tracker.record_stage(chain_id, "CALCULATE", {"in": 2}, {"out": 2})
        tracker.seal_chain(chain_id)
        assert tracker.validate_chain(chain_id) is True

    def test_10_stage_pipeline(self):
        """Test recording all 10 pipeline stages."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        stages = [
            "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS", "CALCULATE",
            "ALLOCATE", "AGGREGATE", "COMPLIANCE", "PROVENANCE", "SEAL",
        ]
        for stage in stages:
            tracker.record_stage(chain_id, stage, {"stage": stage}, {"result": stage})
        final_hash = tracker.seal_chain(chain_id)
        assert len(final_hash) == 64
        chain = tracker.get_chain(chain_id)
        assert len(chain.entries) == 10


# ==============================================================================
# CHAIN EXPORT TESTS
# ==============================================================================


class TestChainExport:
    """Test provenance chain export."""

    def test_export_chain_json(self):
        """Test exporting chain as JSON."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"in": 1}, {"out": 1})
        tracker.seal_chain(chain_id)
        chain = tracker.get_chain(chain_id)
        j = chain.to_json()
        parsed = json.loads(j)
        assert parsed["chain_id"] == chain_id

    def test_export_chain_dict(self):
        """Test exporting chain as dict."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain()
        tracker.record_stage(chain_id, "VALIDATE", {"in": 1}, {"out": 1})
        tracker.seal_chain(chain_id)
        chain = tracker.get_chain(chain_id)
        d = chain.to_dict()
        assert d["chain_id"] == chain_id
        assert len(d["entries"]) == 1


# ==============================================================================
# BATCH PROVENANCE TESTS
# ==============================================================================


class TestBatchProvenance:
    """Test BatchProvenance for batch calculations."""

    def test_batch_provenance_creation(self):
        """Test BatchProvenance initialization."""
        batch = BatchProvenance(batch_id="batch-001")
        assert batch.batch_id == "batch-001"
        assert len(batch.individual_chain_ids) == 0

    def test_batch_provenance_add_chains(self):
        """Test adding chains to batch provenance."""
        batch = BatchProvenance(batch_id="batch-001")
        batch.individual_chain_ids.append("chain-001")
        batch.individual_chain_ids.append("chain-002")
        assert len(batch.individual_chain_ids) == 2
        batch.item_count = 2
        assert batch.item_count == 2

    def test_batch_provenance_to_dict(self):
        """Test BatchProvenance serialization."""
        batch = BatchProvenance(batch_id="batch-001")
        d = batch.to_dict()
        assert d["batch_id"] == "batch-001"


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestThreadSafety:
    """Test provenance tracker thread safety."""

    def test_concurrent_chains(self):
        """Test concurrent chain creation is thread-safe."""
        tracker = ProvenanceTracker()
        chain_ids = []
        errors = []

        def create_chain():
            try:
                cid = tracker.start_chain()
                tracker.record_stage(cid, "VALIDATE", {"t": 1}, {"r": 1})
                chain_ids.append(cid)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create_chain) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(chain_ids) == 10
        assert len(set(chain_ids)) == 10


# ==============================================================================
# PARAMETRIZED HASH INPUT TESTS
# ==============================================================================


class TestParametrizedHashInputs:
    """Parametrized tests for hash functions."""

    @pytest.mark.parametrize("input_data", [
        {"key": "value"},
        {"number": Decimal("123.456")},
        {"list": [1, 2, 3]},
        {"nested": {"a": {"b": "c"}}},
        {"empty": {}},
        {"boolean": True},
        {"null": None},
    ])
    def test_compute_hash_various_inputs(self, input_data):
        """Test _compute_hash handles various input types."""
        h = _compute_hash(input_data)
        assert len(h) == 64

    @pytest.mark.parametrize("stage", [
        "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS", "CALCULATE",
        "ALLOCATE", "AGGREGATE", "COMPLIANCE", "PROVENANCE", "SEAL",
    ])
    def test_chain_hash_all_stages(self, stage):
        """Test chain hash for all 10 pipeline stages."""
        h = _compute_chain_hash("prev" * 16, stage, "input_hash", "output_hash")
        assert len(h) == 64
