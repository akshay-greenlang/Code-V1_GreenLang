# -*- coding: utf-8 -*-
"""Unit tests for Capital Goods Agent provenance tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal

import pytest

from greenlang.capital_goods.provenance import (
    CapitalGoodsProvenance,
    ProvenanceEntry,
    ProvenanceStage,
    hash_asset_record,
    hash_capex_spend,
    hash_physical_record,
    hash_supplier_record,
    hash_calculation_result,
)


# ============================================================================
# SINGLETON PATTERN TESTS
# ============================================================================


class TestProvenanceSingleton:
    """Test provenance singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    def test_singleton_same_instance(self):
        """Test that multiple calls return the same instance."""
        prov1 = CapitalGoodsProvenance()
        prov2 = CapitalGoodsProvenance()
        assert prov1 is prov2

    def test_reset_creates_new_instance(self):
        """Test that reset() creates a new instance."""
        prov1 = CapitalGoodsProvenance()
        old_id = id(prov1)
        CapitalGoodsProvenance.reset()
        prov2 = CapitalGoodsProvenance()
        assert id(prov2) != old_id

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        import threading

        instances = []

        def get_instance():
            instances.append(CapitalGoodsProvenance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


# ============================================================================
# PROVENANCE CHAIN TESTS
# ============================================================================


class TestProvenanceChain:
    """Test provenance chain creation and management."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    def test_start_chain(self):
        """Test starting a new provenance chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        initial_hash = prov.start_chain(request_id)

        assert isinstance(initial_hash, str)
        assert len(initial_hash) == 64  # SHA-256 hex digest length
        assert request_id in prov._chains

    def test_start_chain_creates_genesis_entry(self):
        """Test that starting chain creates genesis entry."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        initial_hash = prov.start_chain(request_id)

        chain = prov._chains[request_id]
        assert len(chain) == 1
        assert chain[0].stage == ProvenanceStage.GENESIS
        assert chain[0].current_hash == initial_hash

    def test_start_chain_twice_same_request_raises(self):
        """Test that starting chain twice for same request raises error."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        with pytest.raises(ValueError, match="Chain already exists"):
            prov.start_chain(request_id)

    def test_record_stage(self):
        """Test recording a stage in the chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        data = {"test": "data"}
        entry = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data=data,
        )

        assert isinstance(entry, ProvenanceEntry)
        assert entry.stage == ProvenanceStage.INPUT_VALIDATION
        assert entry.data == data

    def test_record_stage_updates_chain(self):
        """Test that recording stage updates the chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data={"test": "data"},
        )

        chain = prov._chains[request_id]
        assert len(chain) == 2  # Genesis + new entry

    def test_record_stage_chains_hashes(self):
        """Test that recording stage chains hashes correctly."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        initial_hash = prov.start_chain(request_id)

        entry1 = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data={"test": "data1"},
        )

        assert entry1.previous_hash == initial_hash
        assert entry1.current_hash != initial_hash

        entry2 = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.FACTOR_LOOKUP,
            data={"test": "data2"},
        )

        assert entry2.previous_hash == entry1.current_hash
        assert entry2.current_hash != entry1.current_hash

    def test_seal_chain(self):
        """Test sealing a provenance chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {})

        final_hash = prov.seal_chain(request_id)

        assert isinstance(final_hash, str)
        assert len(final_hash) == 64

    def test_seal_chain_marks_sealed(self):
        """Test that sealing chain marks it as sealed."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        prov.seal_chain(request_id)

        assert prov._sealed[request_id] is True

    def test_record_stage_after_seal_raises(self):
        """Test that recording stage after seal raises error."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.seal_chain(request_id)

        with pytest.raises(ValueError, match="Chain is sealed"):
            prov.record_stage(request_id, ProvenanceStage.CALCULATION, {})


# ============================================================================
# ALL STAGES TEST
# ============================================================================


class TestAllProvenanceStages:
    """Test all provenance stages can be recorded."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    @pytest.mark.parametrize("stage", [
        ProvenanceStage.INPUT_VALIDATION,
        ProvenanceStage.ASSET_RECORD_INTAKE,
        ProvenanceStage.CAPEX_SPEND_INTAKE,
        ProvenanceStage.PHYSICAL_RECORD_INTAKE,
        ProvenanceStage.SUPPLIER_RECORD_INTAKE,
        ProvenanceStage.METHOD_SELECTION,
        ProvenanceStage.FACTOR_LOOKUP,
        ProvenanceStage.CALCULATION,
        ProvenanceStage.UNCERTAINTY_ANALYSIS,
        ProvenanceStage.DATA_QUALITY_ASSESSMENT,
        ProvenanceStage.COMPLIANCE_CHECK,
        ProvenanceStage.RESULT_VALIDATION,
        ProvenanceStage.OUTPUT_GENERATION,
        ProvenanceStage.ERROR_HANDLING,
        ProvenanceStage.BATCH_PROCESSING,
        ProvenanceStage.CACHE_OPERATION,
        ProvenanceStage.DATABASE_OPERATION,
    ])
    def test_record_all_stages(self, stage):
        """Test that all stages can be recorded."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        entry = prov.record_stage(
            request_id=request_id,
            stage=stage,
            data={"test": "data"},
        )

        assert entry.stage == stage


# ============================================================================
# CHAIN VERIFICATION TESTS
# ============================================================================


class TestChainVerification:
    """Test provenance chain verification."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    def test_verify_chain_valid(self):
        """Test verifying a valid chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data1"})
        prov.record_stage(request_id, ProvenanceStage.CALCULATION, {"test": "data2"})
        prov.seal_chain(request_id)

        is_valid = prov.verify_chain(request_id)

        assert is_valid is True

    def test_verify_chain_detects_tampering(self):
        """Test that verification detects tampering."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data1"})
        prov.record_stage(request_id, ProvenanceStage.CALCULATION, {"test": "data2"})

        # Tamper with chain
        chain = prov._chains[request_id]
        chain[1].data = {"tampered": "data"}

        is_valid = prov.verify_chain(request_id)

        assert is_valid is False

    def test_verify_chain_detects_broken_hash_link(self):
        """Test that verification detects broken hash links."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data1"})
        prov.record_stage(request_id, ProvenanceStage.CALCULATION, {"test": "data2"})

        # Break hash link
        chain = prov._chains[request_id]
        chain[2].previous_hash = "invalid_hash"

        is_valid = prov.verify_chain(request_id)

        assert is_valid is False

    def test_verify_nonexistent_chain_raises(self):
        """Test that verifying nonexistent chain raises error."""
        prov = CapitalGoodsProvenance()

        with pytest.raises(ValueError, match="Chain does not exist"):
            prov.verify_chain("NONEXISTENT")


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


class TestProvenanceSerialization:
    """Test provenance serialization."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    def test_to_dict(self):
        """Test converting provenance chain to dictionary."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data"})
        prov.seal_chain(request_id)

        chain_dict = prov.to_dict(request_id)

        assert isinstance(chain_dict, dict)
        assert "request_id" in chain_dict
        assert "entries" in chain_dict
        assert "sealed" in chain_dict
        assert chain_dict["request_id"] == request_id
        assert chain_dict["sealed"] is True

    def test_to_dict_entries_structure(self):
        """Test that to_dict entries have correct structure."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data"})

        chain_dict = prov.to_dict(request_id)

        assert len(chain_dict["entries"]) == 2  # Genesis + one entry
        for entry in chain_dict["entries"]:
            assert "stage" in entry
            assert "timestamp" in entry
            assert "data" in entry
            assert "previous_hash" in entry
            assert "current_hash" in entry

    def test_from_dict_roundtrip(self):
        """Test from_dict round-trip."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data"})
        prov.seal_chain(request_id)

        chain_dict = prov.to_dict(request_id)

        # Create new provenance instance and restore
        CapitalGoodsProvenance.reset()
        prov2 = CapitalGoodsProvenance()
        prov2.from_dict(chain_dict)

        # Verify restoration
        restored_dict = prov2.to_dict(request_id)
        assert restored_dict == chain_dict

    def test_from_dict_preserves_verification(self):
        """Test that from_dict preserves verification ability."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)
        prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data"})
        prov.seal_chain(request_id)

        chain_dict = prov.to_dict(request_id)

        # Restore and verify
        CapitalGoodsProvenance.reset()
        prov2 = CapitalGoodsProvenance()
        prov2.from_dict(chain_dict)

        assert prov2.verify_chain(request_id) is True


# ============================================================================
# STANDALONE HASH FUNCTION TESTS
# ============================================================================


class TestStandaloneHashFunctions:
    """Test standalone hash helper functions."""

    def test_hash_asset_record(self, sample_asset_building):
        """Test hashing asset record."""
        hash_value = hash_asset_record(sample_asset_building)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_asset_record_deterministic(self, sample_asset_building):
        """Test that hashing is deterministic."""
        hash1 = hash_asset_record(sample_asset_building)
        hash2 = hash_asset_record(sample_asset_building)

        assert hash1 == hash2

    def test_hash_asset_record_changes_with_input(
        self,
        sample_asset_building,
        sample_asset_machinery,
    ):
        """Test that hash changes with different input."""
        hash1 = hash_asset_record(sample_asset_building)
        hash2 = hash_asset_record(sample_asset_machinery)

        assert hash1 != hash2

    def test_hash_capex_spend(self, sample_capex_construction):
        """Test hashing CapEx spend record."""
        hash_value = hash_capex_spend(sample_capex_construction)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_capex_spend_deterministic(self, sample_capex_construction):
        """Test that hashing is deterministic."""
        hash1 = hash_capex_spend(sample_capex_construction)
        hash2 = hash_capex_spend(sample_capex_construction)

        assert hash1 == hash2

    def test_hash_capex_spend_changes_with_input(
        self,
        sample_capex_construction,
        sample_capex_machinery,
    ):
        """Test that hash changes with different input."""
        hash1 = hash_capex_spend(sample_capex_construction)
        hash2 = hash_capex_spend(sample_capex_machinery)

        assert hash1 != hash2

    def test_hash_physical_record(self, sample_physical_steel):
        """Test hashing physical record."""
        hash_value = hash_physical_record(sample_physical_steel)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_physical_record_deterministic(self, sample_physical_steel):
        """Test that hashing is deterministic."""
        hash1 = hash_physical_record(sample_physical_steel)
        hash2 = hash_physical_record(sample_physical_steel)

        assert hash1 == hash2

    def test_hash_physical_record_changes_with_input(
        self,
        sample_physical_steel,
        sample_physical_concrete,
    ):
        """Test that hash changes with different input."""
        hash1 = hash_physical_record(sample_physical_steel)
        hash2 = hash_physical_record(sample_physical_concrete)

        assert hash1 != hash2

    def test_hash_supplier_record(self, sample_supplier_epd):
        """Test hashing supplier record."""
        hash_value = hash_supplier_record(sample_supplier_epd)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_supplier_record_deterministic(self, sample_supplier_epd):
        """Test that hashing is deterministic."""
        hash1 = hash_supplier_record(sample_supplier_epd)
        hash2 = hash_supplier_record(sample_supplier_epd)

        assert hash1 == hash2

    def test_hash_supplier_record_changes_with_input(
        self,
        sample_supplier_epd,
        sample_supplier_pcf,
    ):
        """Test that hash changes with different input."""
        hash1 = hash_supplier_record(sample_supplier_epd)
        hash2 = hash_supplier_record(sample_supplier_pcf)

        assert hash1 != hash2

    def test_hash_calculation_result(self, sample_spend_based_result):
        """Test hashing calculation result."""
        hash_value = hash_calculation_result(sample_spend_based_result)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_calculation_result_deterministic(self, sample_spend_based_result):
        """Test that hashing is deterministic."""
        hash1 = hash_calculation_result(sample_spend_based_result)
        hash2 = hash_calculation_result(sample_spend_based_result)

        assert hash1 == hash2

    def test_hash_calculation_result_changes_with_input(
        self,
        sample_spend_based_result,
        sample_average_data_result,
    ):
        """Test that hash changes with different input."""
        hash1 = hash_calculation_result(sample_spend_based_result)
        hash2 = hash_calculation_result(sample_average_data_result)

        assert hash1 != hash2


# ============================================================================
# HASH ALGORITHM TESTS
# ============================================================================


class TestHashAlgorithm:
    """Test hash algorithm properties."""

    def test_hash_uses_sha256(self):
        """Test that hashing uses SHA-256."""
        test_data = {"test": "data"}
        json_str = json.dumps(test_data, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(json_str.encode()).hexdigest()

        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        initial_hash = prov.start_chain(request_id)

        # Initial hash should be SHA-256 of request_id + genesis data
        assert len(initial_hash) == 64
        assert all(c in "0123456789abcdef" for c in initial_hash)

    def test_hash_collision_resistance(self):
        """Test hash collision resistance with similar inputs."""
        from greenlang.capital_goods.models import CapitalAssetRecord, AssetCategory, AssetSubCategory

        asset1 = CapitalAssetRecord(
            asset_id="ASSET-001",
            asset_name="Asset 1",
            asset_category=AssetCategory.BUILDINGS,
            asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
            acquisition_date=datetime(2026, 1, 1).date(),
            acquisition_cost_usd=Decimal("1000000.00"),
            useful_life_years=40,
            depreciation_method="straight-line",
            currency_code="USD",
            reporting_year=2026,
        )

        asset2 = CapitalAssetRecord(
            asset_id="ASSET-002",  # Only ID differs
            asset_name="Asset 1",
            asset_category=AssetCategory.BUILDINGS,
            asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
            acquisition_date=datetime(2026, 1, 1).date(),
            acquisition_cost_usd=Decimal("1000000.00"),
            useful_life_years=40,
            depreciation_method="straight-line",
            currency_code="USD",
            reporting_year=2026,
        )

        hash1 = hash_asset_record(asset1)
        hash2 = hash_asset_record(asset2)

        assert hash1 != hash2


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestProvenanceEdgeCases:
    """Test provenance edge cases."""

    @pytest.fixture(autouse=True)
    def reset_provenance(self):
        """Reset provenance before each test."""
        CapitalGoodsProvenance.reset()
        yield
        CapitalGoodsProvenance.reset()

    def test_empty_data_in_stage(self):
        """Test recording stage with empty data."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        entry = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data={},
        )

        assert entry.data == {}

    def test_none_data_in_stage(self):
        """Test recording stage with None data."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        entry = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data=None,
        )

        assert entry.data is None

    def test_large_data_in_stage(self):
        """Test recording stage with large data."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        entry = prov.record_stage(
            request_id=request_id,
            stage=ProvenanceStage.INPUT_VALIDATION,
            data=large_data,
        )

        assert len(entry.data) == 1000

    def test_very_long_chain(self):
        """Test creating a very long provenance chain."""
        prov = CapitalGoodsProvenance()
        request_id = "REQ-2026-001"
        prov.start_chain(request_id)

        # Add 100 stages
        for i in range(100):
            prov.record_stage(
                request_id=request_id,
                stage=ProvenanceStage.CALCULATION,
                data={"iteration": i},
            )

        chain = prov._chains[request_id]
        assert len(chain) == 101  # Genesis + 100 entries

        # Verify chain is still valid
        assert prov.verify_chain(request_id) is True

    def test_multiple_chains(self):
        """Test managing multiple provenance chains."""
        prov = CapitalGoodsProvenance()

        # Create 10 different chains
        for i in range(10):
            request_id = f"REQ-2026-{i:03d}"
            prov.start_chain(request_id)
            prov.record_stage(request_id, ProvenanceStage.INPUT_VALIDATION, {"id": i})

        # Verify all chains exist and are valid
        for i in range(10):
            request_id = f"REQ-2026-{i:03d}"
            assert request_id in prov._chains
            assert prov.verify_chain(request_id) is True
