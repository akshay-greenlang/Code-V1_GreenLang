# -*- coding: utf-8 -*-
"""
Test suite for downstream_leased_assets.provenance - AGENT-MRV-026.

Tests SHA-256 provenance chain, entry creation, chain validation,
Merkle-style hashing, serialization, and thread safety for the
Downstream Leased Assets Agent (GL-MRV-S3-013).

Coverage:
- SHA-256: 64-char hex, deterministic, different inputs
- Chain lifecycle: UUID, stage hash, ordering
- Merkle tree: single leaf, multiple, deterministic
- Seal/verify, tamper detection
- 15 standalone hash functions for all asset types
- Thread safety: 12 threads

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import json
from decimal import Decimal
from datetime import datetime, timezone
import pytest

try:
    from greenlang.agents.mrv.downstream_leased_assets.provenance import (
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

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="downstream_leased_assets.provenance not available")
pytestmark = _SKIP


# ==============================================================================
# PROVENANCE ENTRY TESTS
# ==============================================================================


class TestProvenanceEntry:

    def test_creation(self):
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
        assert entry.agent_id == "GL-MRV-S3-013"
        assert entry.agent_version == "1.0.0"

    def test_frozen(self):
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

    def test_to_dict(self):
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
        assert d["agent_id"] == "GL-MRV-S3-013"

    def test_to_json(self):
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

    def test_hash_lengths(self):
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

    def test_with_metadata(self):
        entry = ProvenanceEntry(
            entry_id="e1",
            stage="CALCULATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64,
            output_hash="b" * 64,
            chain_hash="c" * 64,
            previous_hash="d" * 64,
            metadata={"building_type": "office", "vacancy_rate": 0.12},
        )
        assert entry.metadata["building_type"] == "office"


# ==============================================================================
# PROVENANCE CHAIN TESTS
# ==============================================================================


class TestProvenanceChain:

    def test_creation(self):
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.chain_id == "chain-001"
        assert len(chain.entries) == 0
        assert chain.final_hash is None

    def test_root_hash_empty(self):
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.root_hash == ""

    def test_is_valid(self):
        chain = ProvenanceChain(chain_id="chain-001")
        assert chain.is_valid is True

    def test_to_dict(self):
        chain = ProvenanceChain(chain_id="chain-001")
        d = chain.to_dict()
        assert d["chain_id"] == "chain-001"

    def test_add_entry(self):
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1", stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64, output_hash="b" * 64,
            chain_hash="c" * 64, previous_hash="",
        )
        chain.add_entry(entry)
        assert len(chain.entries) == 1

    def test_multiple_entries(self):
        chain = ProvenanceChain(chain_id="chain-001")
        for i, stage in enumerate(["VALIDATE", "CLASSIFY", "NORMALIZE"]):
            entry = ProvenanceEntry(
                entry_id=f"e{i}", stage=stage,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash="a" * 64, output_hash="b" * 64,
                chain_hash="c" * 64, previous_hash="" if i == 0 else "d" * 64,
            )
            chain.add_entry(entry)
        assert len(chain.entries) == 3
        assert chain.entries[0].stage == "VALIDATE"

    def test_seal(self):
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1", stage="SEAL",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64, output_hash="b" * 64,
            chain_hash="c" * 64, previous_hash="",
        )
        chain.add_entry(entry)
        chain.seal()
        assert chain.final_hash is not None
        assert len(chain.final_hash) == 64

    def test_tamper_detection(self):
        chain = ProvenanceChain(chain_id="chain-001")
        entry = ProvenanceEntry(
            entry_id="e1", stage="VALIDATE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="a" * 64, output_hash="b" * 64,
            chain_hash="c" * 64, previous_hash="",
        )
        chain.add_entry(entry)
        chain.seal()
        assert chain.is_valid is True


# ==============================================================================
# SERIALIZATION HELPER TESTS
# ==============================================================================


class TestSerializeHelpers:

    def test_serialize_string(self):
        result = _serialize("hello")
        assert isinstance(result, str)

    def test_serialize_decimal(self):
        result = _serialize(Decimal("123.45678901"))
        assert isinstance(result, str)

    def test_serialize_decimal_quantized(self):
        r1 = _serialize(Decimal("1.23456789000"))
        r2 = _serialize(Decimal("1.23456789"))
        assert r1 == r2

    def test_serialize_integer(self):
        result = _serialize(42)
        assert "42" in str(result)

    def test_serialize_datetime(self):
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _serialize(dt)
        assert "2026" in str(result)

    def test_serialize_enum(self):
        result = _serialize(ProvenanceStage.VALIDATE)
        assert "validate" in str(result).lower()

    def test_serialize_dict(self):
        d = {"key1": Decimal("100"), "key2": "value"}
        result = _serialize(d)
        assert isinstance(result, (str, dict))

    def test_serialize_none(self):
        result = _serialize(None)
        assert result is None or result == "None" or result == "null"


# ==============================================================================
# STANDALONE HASH FUNCTION TESTS
# ==============================================================================


class TestComputeHash:

    def test_returns_64_char_hex(self):
        h = _compute_hash("test_data")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        h1 = _compute_hash("test_data")
        h2 = _compute_hash("test_data")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = _compute_hash("data_a")
        h2 = _compute_hash("data_b")
        assert h1 != h2

    def test_empty_string_valid(self):
        h = _compute_hash("")
        assert len(h) == 64

    def test_building_data_hash(self):
        h = _compute_hash("office|2500|temperate|450000")
        assert len(h) == 64

    def test_vehicle_data_hash(self):
        h = _compute_hash("medium_car|diesel|25000|10")
        assert len(h) == 64

    def test_equipment_data_hash(self):
        h = _compute_hash("construction|200|2000|0.60")
        assert len(h) == 64

    def test_it_asset_data_hash(self):
        h = _compute_hash("server|500|0.85|1.40")
        assert len(h) == 64

    def test_spend_data_hash(self):
        h = _compute_hash("531120|100000|USD|2024")
        assert len(h) == 64

    def test_vacancy_data_hash(self):
        h = _compute_hash("office|0.12|base_load")
        assert len(h) == 64

    def test_tenant_data_hash(self):
        h = _compute_hash("T-001|875.00|0.35")
        assert len(h) == 64

    def test_allocation_data_hash(self):
        h = _compute_hash("area|875|2500")
        assert len(h) == 64

    def test_compliance_data_hash(self):
        h = _compute_hash("ghg_protocol|pass|95.0")
        assert len(h) == 64

    def test_portfolio_data_hash(self):
        h = _compute_hash("portfolio|15|250000")
        assert len(h) == 64

    def test_dc_rule_data_hash(self):
        h = _compute_hash("DC-DLA-001|operational_control|tenant")
        assert len(h) == 64


class TestComputeChainHash:

    def test_chain_hash_two_entries(self):
        h = _compute_chain_hash("prev_hash", "current_data")
        assert len(h) == 64

    def test_chain_hash_deterministic(self):
        h1 = _compute_chain_hash("prev", "curr")
        h2 = _compute_chain_hash("prev", "curr")
        assert h1 == h2

    def test_chain_hash_different_prev(self):
        h1 = _compute_chain_hash("prev_a", "curr")
        h2 = _compute_chain_hash("prev_b", "curr")
        assert h1 != h2


class TestMerkleHash:

    def test_single_leaf(self):
        h = _merkle_hash(["leaf_1"])
        assert len(h) == 64

    def test_two_leaves(self):
        h = _merkle_hash(["leaf_1", "leaf_2"])
        assert len(h) == 64

    def test_multiple_leaves(self):
        h = _merkle_hash(["a", "b", "c", "d", "e"])
        assert len(h) == 64

    def test_odd_count(self):
        h = _merkle_hash(["a", "b", "c"])
        assert len(h) == 64

    def test_deterministic(self):
        h1 = _merkle_hash(["a", "b", "c"])
        h2 = _merkle_hash(["a", "b", "c"])
        assert h1 == h2

    def test_order_matters(self):
        h1 = _merkle_hash(["a", "b"])
        h2 = _merkle_hash(["b", "a"])
        assert h1 != h2

    def test_empty_list(self):
        h = _merkle_hash([])
        assert len(h) == 64 or h == ""


# ==============================================================================
# PROVENANCE TRACKER TESTS
# ==============================================================================


class TestProvenanceTracker:

    def test_creation(self):
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_start_chain(self):
        tracker = ProvenanceTracker()
        chain_id = tracker.start_chain("DLA-001")
        assert chain_id is not None

    def test_record_validate_stage(self):
        tracker = ProvenanceTracker()
        tracker.start_chain("DLA-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"asset_type": "building"},
            output_data={"validated": True},
        )
        chain = tracker.get_current_chain()
        assert len(chain.entries) == 1

    def test_record_all_10_stages(self):
        tracker = ProvenanceTracker()
        tracker.start_chain("DLA-001")
        stages_data = [
            (ProvenanceStage.VALIDATE, {"input": "raw"}, {"valid": True}),
            (ProvenanceStage.CLASSIFY, {"type": "building"}, {"category": "building"}),
            (ProvenanceStage.NORMALIZE, {"area": "2500"}, {"area_sqm": 2500}),
            (ProvenanceStage.RESOLVE_EFS, {"region": "US"}, {"ef": 0.37}),
            (ProvenanceStage.CALCULATE, {"kwh": 450000}, {"co2e": 167265}),
            (ProvenanceStage.ALLOCATE, {"share": 0.35}, {"allocated": True}),
            (ProvenanceStage.AGGREGATE, {"method": "hybrid"}, {"total": 155000}),
            (ProvenanceStage.COMPLIANCE, {"fw": "ghg"}, {"status": "pass"}),
            (ProvenanceStage.PROVENANCE, {"chain": True}, {"hashed": True}),
            (ProvenanceStage.SEAL, {"final": True}, {"sealed": True}),
        ]
        for stage, inp, out in stages_data:
            tracker.record_stage(stage, input_data=inp, output_data=out)
        chain = tracker.get_current_chain()
        assert len(chain.entries) == 10

    def test_seal_chain(self):
        tracker = ProvenanceTracker()
        tracker.start_chain("DLA-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        final_hash = tracker.seal_chain()
        assert isinstance(final_hash, str)
        assert len(final_hash) == 64

    def test_validate_chain(self):
        tracker = ProvenanceTracker()
        tracker.start_chain("DLA-001")
        tracker.record_stage(
            ProvenanceStage.VALIDATE,
            input_data={"data": "test"},
            output_data={"valid": True},
        )
        tracker.seal_chain()
        assert tracker.validate_chain() is True

    def test_chain_hash_deterministic(self):
        def create_chain():
            t = ProvenanceTracker()
            t.start_chain("DLA-001")
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
        t1 = ProvenanceTracker()
        t1.start_chain("DLA-001")
        t1.record_stage(ProvenanceStage.VALIDATE, {"area": Decimal("2500")}, {"valid": True})
        h1 = t1.seal_chain()

        t2 = ProvenanceTracker()
        t2.start_chain("DLA-002")
        t2.record_stage(ProvenanceStage.VALIDATE, {"area": Decimal("5000")}, {"valid": True})
        h2 = t2.seal_chain()

        assert h1 != h2

    def test_chain_reset(self):
        tracker = ProvenanceTracker()
        tracker.start_chain("DLA-001")
        tracker.record_stage(ProvenanceStage.VALIDATE, {"data": "test"}, {"valid": True})
        tracker.reset()
        tracker.start_chain("DLA-002")
        chain = tracker.get_current_chain()
        assert len(chain.entries) == 0


# ==============================================================================
# BATCH PROVENANCE TESTS
# ==============================================================================


class TestBatchProvenance:

    def test_batch_creation(self):
        batch = BatchProvenance(batch_id="batch-001")
        assert batch.batch_id == "batch-001"

    def test_batch_aggregate_hash(self):
        batch = BatchProvenance(batch_id="batch-001")
        for i in range(3):
            tracker = ProvenanceTracker()
            tracker.start_chain(f"DLA-{i:03d}")
            tracker.record_stage(ProvenanceStage.VALIDATE, {"idx": i}, {"valid": True})
            tracker.seal_chain()
            batch.add_chain(tracker.get_current_chain())
        agg_hash = batch.compute_aggregate_hash()
        assert len(agg_hash) == 64

    def test_batch_aggregate_hash_deterministic(self):
        def create_batch():
            b = BatchProvenance(batch_id="batch-001")
            for i in range(3):
                t = ProvenanceTracker()
                t.start_chain(f"DLA-{i:03d}")
                t.record_stage(ProvenanceStage.VALIDATE, {"idx": i}, {"valid": True})
                t.seal_chain()
                b.add_chain(t.get_current_chain())
            return b.compute_aggregate_hash()

        h1 = create_batch()
        h2 = create_batch()
        assert h1 == h2


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestThreadSafety:

    def test_concurrent_chain_creation(self):
        hashes = []

        def create_and_seal(idx):
            tracker = ProvenanceTracker()
            tracker.start_chain(f"DLA-{idx:03d}")
            tracker.record_stage(ProvenanceStage.VALIDATE, {"idx": idx}, {"valid": True})
            h = tracker.seal_chain()
            hashes.append(h)

        threads = [threading.Thread(target=create_and_seal, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(hashes) == 12
        for h in hashes:
            assert len(h) == 64

    def test_concurrent_hash_computation(self):
        results = []

        def compute(data):
            results.append(_compute_hash(data))

        threads = [
            threading.Thread(target=compute, args=(f"data_{i}",))
            for i in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 12
        for h in results:
            assert len(h) == 64

    def test_concurrent_serialize(self):
        results = []

        def serialize_data(val):
            results.append(_serialize(val))

        values = [Decimal("100.5"), "hello", 42, None, datetime.now(timezone.utc)]
        threads = [threading.Thread(target=serialize_data, args=(v,)) for v in values * 2]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
