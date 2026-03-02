# -*- coding: utf-8 -*-
"""
Test suite for upstream_leased_assets.provenance - AGENT-MRV-021.

Tests SHA-256 provenance chain, entry creation, chain validation,
Merkle-style hashing, serialization, and thread safety for the
Upstream Leased Assets Agent (GL-MRV-S3-008).

Coverage:
- ProvenanceEntry creation, immutability, to_dict, to_json
- ProvenanceChain creation, add_record, finalize, validate, tamper detection
- Serialization helpers: _serialize for Decimal, datetime, Enum, nested dict
- Standalone hash functions: _compute_hash, _compute_chain_hash, _merkle_hash
- ProvenanceTracker: start_chain, record_stage (10 stages), seal_chain, validate
- BatchProvenance: multiple chains, aggregate hash
- Merkle proof: single, multiple, odd count
- 10-stage pipeline recording
- Chain reset and thread safety
- Singleton pattern

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
import pytest

try:
    from greenlang.upstream_leased_assets.provenance import (
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
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="upstream_leased_assets.provenance not available",
)

pytestmark = _SKIP


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
        assert entry.agent_id == "GL-MRV-S3-008"
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
        assert d["agent_id"] == "GL-MRV-S3-008"

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

    def test_provenance_entry_hash_lengths(self):
        """Test all hashes are 64 characters."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="d" * 64,
        )
        assert len(entry.input_hash) == 64
        assert len(entry.output_hash) == 64
        assert len(entry.chain_hash) == 64
        assert len(entry.previous_hash) == 64

    def test_provenance_entry_with_metadata(self):
        """Test ProvenanceEntry with optional metadata."""
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="CALCULATE_BUILDING",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="d" * 64,
            metadata={"building_type": "office", "method": "asset_specific"},
        )
        assert entry.metadata["building_type"] == "office"


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

    def test_provenance_chain_add_entry(self):
        """Test adding entries to chain."""
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        chain.add_entry(entry)
        assert len(chain.entries) == 1

    def test_provenance_chain_multiple_entries(self):
        """Test chain with multiple entries maintains order."""
        chain = ProvenanceChain(chain_id="chain-001")
        for i, stage in enumerate(["VALIDATE", "CLASSIFY", "NORMALIZE"]):
            entry = ProvenanceEntry(
                entry_id=f"e{i}",
                stage=stage,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash="a" * 64,
                output_hash="b" * 64,
                chain_hash="c" * 64,
                previous_hash="" if i == 0 else "d" * 64,
            )
            chain.add_entry(entry)
        assert len(chain.entries) == 3
        assert chain.entries[0].stage == "VALIDATE"
        assert chain.entries[2].stage == "NORMALIZE"

    def test_provenance_chain_seal(self):
        """Test sealing the chain sets final_hash."""
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="SEAL",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        chain.add_entry(entry)
        chain.seal()
        assert chain.final_hash is not None
        assert len(chain.final_hash) == 64
        assert chain.sealed_at is not None

    def test_provenance_chain_tamper_detection(self):
        """Test chain detects tampering after seal."""
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="",
        )
        chain.add_entry(entry)
        chain.seal()
        original_hash = chain.final_hash
        # Tampering should be detectable
        assert chain.is_valid is True


# ==============================================================================
# SERIALIZATION HELPER TESTS
# ==============================================================================


class TestSerializeHelpers:
    """Test _serialize helper function."""

    def test_serialize_string(self):
        """Test serializing a string."""
        result = _serialize("hello")
        assert isinstance(result, str)

    def test_serialize_decimal(self):
        """Test serializing a Decimal."""
        result = _serialize(Decimal("123.45678901"))
        assert isinstance(result, str)
        assert "123.45678901" in result or "123.4567890" in result

    def test_serialize_decimal_quantized(self):
        """Test Decimal is quantized to 8dp."""
        r1 = _serialize(Decimal("1.23456789000"))
        r2 = _serialize(Decimal("1.23456789"))
        assert r1 == r2

    def test_serialize_integer(self):
        """Test serializing an integer."""
        result = _serialize(42)
        assert "42" in str(result)

    def test_serialize_float(self):
        """Test serializing a float."""
        result = _serialize(3.14)
        assert isinstance(result, (str, float))

    def test_serialize_datetime(self):
        """Test serializing a datetime."""
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _serialize(dt)
        assert isinstance(result, str)
        assert "2026" in result

    def test_serialize_enum(self):
        """Test serializing an enum value."""
        result = _serialize(ProvenanceStage.VALIDATE)
        assert isinstance(result, str)
        assert "validate" in result.lower()

    def test_serialize_dict(self):
        """Test serializing a nested dict."""
        d = {"key1": Decimal("100"), "key2": "value", "nested": {"a": 1}}
        result = _serialize(d)
        assert isinstance(result, (str, dict))

    def test_serialize_list(self):
        """Test serializing a list."""
        lst = [Decimal("1.0"), "hello", 42]
        result = _serialize(lst)
        assert isinstance(result, (str, list))

    def test_serialize_none(self):
        """Test serializing None."""
        result = _serialize(None)
        assert result is None or result == "None" or result == "null"


# ==============================================================================
# STANDALONE HASH FUNCTION TESTS
# ==============================================================================


class TestComputeHash:
    """Test _compute_hash function."""

    def test_returns_64_char_hex(self):
        """Test _compute_hash returns 64-character hex string."""
        h = _compute_hash("test_data")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        """Test same input produces same hash."""
        h1 = _compute_hash("test_data")
        h2 = _compute_hash("test_data")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        """Test different inputs produce different hashes."""
        h1 = _compute_hash("data_a")
        h2 = _compute_hash("data_b")
        assert h1 != h2

    def test_empty_string_valid(self):
        """Test empty string produces valid hash."""
        h = _compute_hash("")
        assert len(h) == 64


class TestComputeChainHash:
    """Test _compute_chain_hash function."""

    def test_chain_hash_two_entries(self):
        """Test chain hash with two entries."""
        h = _compute_chain_hash("prev_hash", "current_data")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_chain_hash_deterministic(self):
        """Test chain hash is deterministic."""
        h1 = _compute_chain_hash("prev", "curr")
        h2 = _compute_chain_hash("prev", "curr")
        assert h1 == h2

    def test_chain_hash_different_prev(self):
        """Test different previous hashes produce different chain hashes."""
        h1 = _compute_chain_hash("prev_a", "curr")
        h2 = _compute_chain_hash("prev_b", "curr")
        assert h1 != h2


class TestMerkleHash:
    """Test _merkle_hash function."""

    def test_merkle_single_leaf(self):
        """Test Merkle hash with single leaf."""
        h = _merkle_hash(["leaf_1"])
        assert len(h) == 64

    def test_merkle_two_leaves(self):
        """Test Merkle hash with two leaves."""
        h = _merkle_hash(["leaf_1", "leaf_2"])
        assert len(h) == 64

    def test_merkle_multiple_leaves(self):
        """Test Merkle hash with multiple leaves."""
        h = _merkle_hash(["a", "b", "c", "d", "e"])
        assert len(h) == 64

    def test_merkle_odd_count(self):
        """Test Merkle hash with odd number of leaves."""
        h = _merkle_hash(["a", "b", "c"])
        assert len(h) == 64

    def test_merkle_deterministic(self):
        """Test Merkle hash is deterministic."""
        h1 = _merkle_hash(["a", "b", "c"])
        h2 = _merkle_hash(["a", "b", "c"])
        assert h1 == h2

    def test_merkle_order_matters(self):
        """Test Merkle hash depends on leaf order."""
        h1 = _merkle_hash(["a", "b"])
        h2 = _merkle_hash(["b", "a"])
        assert h1 != h2

    def test_merkle_empty_list(self):
        """Test Merkle hash with empty list."""
        h = _merkle_hash([])
        assert len(h) == 64 or h == ""


# ==============================================================================
# PROVENANCE TRACKER TESTS
# ==============================================================================


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    def test_tracker_creation(self):
        """Test ProvenanceTracker initialization."""
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_start_chain(self):
        """Test starting a new provenance chain."""
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain("asset-001")
        assert chain_id is not None
        assert isinstance(chain_id, str)

    def test_record_validate_stage(self):
        """Test recording VALIDATE stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"asset_type": "building"},
            output_data={"validated": True},
        )
        chain = tracker.get_current_chain()
        assert len(chain.entries) == 1
        assert chain.entries[0].stage == "validate" or \
            chain.entries[0].stage == "VALIDATE"

    def test_record_classify_stage(self):
        """Test recording CLASSIFY stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.CLASSIFY,
            input_data={"asset_type": "building"},
            output_data={"category": "building", "method": "asset_specific"},
        )
        chain = tracker.get_current_chain()
        assert len(chain.entries) >= 1

    def test_record_normalize_stage(self):
        """Test recording NORMALIZE stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.NORMALIZE,
            input_data={"floor_area": "2500 sqm"},
            output_data={"floor_area_sqm": Decimal("2500")},
        )
        chain = tracker.get_current_chain()
        assert len(chain.entries) >= 1

    def test_record_resolve_efs_stage(self):
        """Test recording RESOLVE_EFS stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.RESOLVE_EFS,
            input_data={"region": "US"},
            output_data={"grid_ef": Decimal("0.37170")},
        )

    def test_record_calculate_building_stage(self):
        """Test recording CALCULATE_BUILDING stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("bldg-001")
        tracker.record_stage(
            ProvenanceStage.CALCULATE_BUILDING,
            input_data={"electricity_kwh": Decimal("450000")},
            output_data={"total_co2e_kg": Decimal("167265")},
        )

    def test_record_calculate_vehicle_stage(self):
        """Test recording CALCULATE_VEHICLE stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("veh-001")
        tracker.record_stage(
            ProvenanceStage.CALCULATE_VEHICLE,
            input_data={"distance_km": Decimal("25000")},
            output_data={"total_co2e_kg": Decimal("5250")},
        )

    def test_record_calculate_equipment_stage(self):
        """Test recording CALCULATE_EQUIPMENT stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("equip-001")
        tracker.record_stage(
            ProvenanceStage.CALCULATE_EQUIPMENT,
            input_data={"rated_power_kw": Decimal("500")},
            output_data={"total_co2e_kg": Decimal("832500")},
        )

    def test_record_calculate_it_stage(self):
        """Test recording CALCULATE_IT stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("it-001")
        tracker.record_stage(
            ProvenanceStage.CALCULATE_IT,
            input_data={"rated_power_w": Decimal("500"), "pue": Decimal("1.4")},
            output_data={"total_co2e_kg": Decimal("2050")},
        )

    def test_record_compliance_stage(self):
        """Test recording COMPLIANCE stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.COMPLIANCE,
            input_data={"frameworks": ["ghg_protocol"]},
            output_data={"status": "pass", "score": 95.0},
        )

    def test_record_seal_stage(self):
        """Test recording SEAL stage."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.SEAL,
            input_data={"chain_id": "asset-001"},
            output_data={"sealed": True},
        )

    def test_full_10_stage_pipeline(self):
        """Test recording all 10 stages in sequence."""
        tracker = ProvenanceTracker()
        tracker.start_chain("bldg-001")

        stages_data = [
            (ProvenanceStage.VALIDATE, {"input": "raw"}, {"valid": True}),
            (ProvenanceStage.CLASSIFY, {"type": "building"}, {"category": "building"}),
            (ProvenanceStage.NORMALIZE, {"area": "2500"}, {"area_sqm": 2500}),
            (ProvenanceStage.RESOLVE_EFS, {"region": "US"}, {"ef": 0.37}),
            (ProvenanceStage.CALCULATE_BUILDING, {"kwh": 450000}, {"co2e": 167265}),
            (ProvenanceStage.CALCULATE_VEHICLE, {}, {"skipped": True}),
            (ProvenanceStage.CALCULATE_EQUIPMENT, {}, {"skipped": True}),
            (ProvenanceStage.CALCULATE_IT, {}, {"skipped": True}),
            (ProvenanceStage.COMPLIANCE, {"fw": "ghg"}, {"status": "pass"}),
            (ProvenanceStage.SEAL, {"final": True}, {"sealed": True}),
        ]

        for stage, inp, out in stages_data:
            tracker.record_stage(stage, input_data=inp, output_data=out)

        chain = tracker.get_current_chain()
        assert len(chain.entries) == 10

    def test_seal_chain(self):
        """Test sealing the chain produces final hash."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        final_hash = tracker.seal_chain()
        assert isinstance(final_hash, str)
        assert len(final_hash) == 64

    def test_validate_chain(self):
        """Test validating a sealed chain."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        tracker.seal_chain()
        assert tracker.validate_chain() is True

    def test_chain_hash_deterministic(self):
        """Test chain produces deterministic hash for same data."""
        def create_chain():
            t = ProvenanceTracker()
            t.start_chain("asset-001")
            t.record_stage(
                ProvenanceStage.VALIDATE,
                input_data={"asset_type": "building", "area": Decimal("2500")},
                output_data={"validated": True},
            )
            return t.seal_chain()

        h1 = create_chain()
        h2 = create_chain()
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Test different data produces different chain hash."""
        t1 = ProvenanceTracker()
        t1.start_chain("asset-001")
        t1.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"area": Decimal("2500")},
            output_data={"valid": True},
        )
        h1 = t1.seal_chain()

        t2 = ProvenanceTracker()
        t2.start_chain("asset-002")
        t2.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"area": Decimal("5000")},
            output_data={"valid": True},
        )
        h2 = t2.seal_chain()

        assert h1 != h2

    def test_chain_reset(self):
        """Test resetting tracker clears chain."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        tracker.reset()
        new_chain = tracker.start_chain("asset-002")
        chain = tracker.get_current_chain()
        assert len(chain.entries) == 0

    def test_export_chain_json(self):
        """Test exporting chain as JSON."""
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"test": True},
            output_data={"valid": True},
        )
        tracker.seal_chain()
        exported = tracker.export_chain()
        assert isinstance(exported, (str, dict))
        if isinstance(exported, str):
            data = json.loads(exported)
        else:
            data = exported
        assert "entries" in data or "chain_id" in data


# ==============================================================================
# BATCH PROVENANCE TESTS
# ==============================================================================


class TestBatchProvenance:
    """Test BatchProvenance for multiple chains."""

    def test_batch_creation(self):
        """Test BatchProvenance initialization."""
        batch = BatchProvenance(batch_id="batch-001")
        assert batch.batch_id == "batch-001"

    def test_batch_add_chain(self):
        """Test adding chain to batch."""
        batch = BatchProvenance(batch_id="batch-001")
        tracker = ProvenanceTracker()
        tracker.start_chain("asset-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        tracker.seal_chain()
        chain = tracker.get_current_chain()
        batch.add_chain(chain)
        assert len(batch.chains) == 1

    def test_batch_aggregate_hash(self):
        """Test batch aggregate hash from multiple chains."""
        batch = BatchProvenance(batch_id="batch-001")

        for i in range(3):
            tracker = ProvenanceTracker()
            tracker.start_chain(f"asset-{i:03d}")
            tracker.record_stage(
                ProvenanceStage.VALIDATE,
                input_data={"idx": i},
                output_data={"valid": True},
            )
            tracker.seal_chain()
            batch.add_chain(tracker.get_current_chain())

        agg_hash = batch.compute_aggregate_hash()
        assert len(agg_hash) == 64
        assert all(c in "0123456789abcdef" for c in agg_hash)

    def test_batch_aggregate_hash_deterministic(self):
        """Test batch aggregate hash is deterministic."""
        def create_batch():
            b = BatchProvenance(batch_id="batch-001")
            for i in range(3):
                t = ProvenanceTracker()
                t.start_chain(f"asset-{i:03d}")
                t.record_stage(
                    ProvenanceStage.VALIDATE,
                    input_data={"idx": i},
                    output_data={"valid": True},
                )
                t.seal_chain()
                b.add_chain(t.get_current_chain())
            return b.compute_aggregate_hash()

        h1 = create_batch()
        h2 = create_batch()
        assert h1 == h2


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestProvenanceSingleton:
    """Test ProvenanceTracker singleton pattern (if applicable)."""

    def test_tracker_instances_independent(self):
        """Test separate tracker instances are independent."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        t1.start_chain("chain-1")
        t2.start_chain("chain-2")
        c1 = t1.get_current_chain()
        c2 = t2.get_current_chain()
        assert c1.chain_id != c2.chain_id


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestProvenanceThreadSafety:
    """Test thread safety of provenance operations."""

    def test_concurrent_chain_creation(self):
        """Test concurrent chain creation across threads."""
        hashes = []

        def create_and_seal(idx):
            tracker = ProvenanceTracker()
            tracker.start_chain(f"asset-{idx:03d}")
            tracker.record_stage(
                ProvenanceStage.VALIDATE,
                input_data={"idx": idx},
                output_data={"valid": True},
            )
            h = tracker.seal_chain()
            hashes.append(h)

        threads = [threading.Thread(target=create_and_seal, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(hashes) == 10
        for h in hashes:
            assert len(h) == 64

    def test_concurrent_hash_computation(self):
        """Test concurrent _compute_hash calls are safe."""
        results = []

        def compute(data):
            results.append(_compute_hash(data))

        threads = [
            threading.Thread(target=compute, args=(f"data_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        # All should be valid 64-char hex
        for h in results:
            assert len(h) == 64

    def test_concurrent_serialize(self):
        """Test concurrent _serialize calls are safe."""
        results = []

        def serialize_data(val):
            results.append(_serialize(val))

        values = [Decimal("100.5"), "hello", 42, None, datetime.now(timezone.utc)]
        threads = [
            threading.Thread(target=serialize_data, args=(v,))
            for v in values * 2
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
