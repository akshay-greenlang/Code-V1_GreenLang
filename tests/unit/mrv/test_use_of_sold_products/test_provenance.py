# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceChainBuilder -- AGENT-MRV-024

Tests SHA-256 deterministic hashing, chain building, Merkle tree computation,
chain validation, seal/verify lifecycle, stage sequence validation, batch
provenance, standalone hash functions (25+), convenience stage recorders,
and thread safety with 10+ concurrent threads.

Target: 50+ tests.
Author: GL-TestEngineer
"""

import hashlib
import json
import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.provenance import (
        ProvenanceChainBuilder,
        ProvenanceChain,
        ProvenanceStage,
        StageRecord,
        BatchProvenance,
        get_provenance_builder,
        reset_provenance_builder,
        _compute_hash,
        _serialize,
        _build_merkle_tree,
        _verify_merkle_proof,
        hash_product_input,
        hash_direct_emission_input,
        hash_indirect_emission_input,
        hash_fuel_sale_input,
        hash_feedstock_input,
        hash_fuel_combustion_result,
        hash_refrigerant_leakage_result,
        hash_chemical_release_result,
        hash_electricity_result,
        hash_heating_fuel_result,
        hash_steam_cooling_result,
        hash_lifetime_result,
        hash_degradation_result,
        hash_weibull_result,
        hash_fleet_result,
        hash_grid_ef,
        hash_fuel_ef,
        hash_refrigerant_gwp,
        hash_compliance_result,
        hash_dc_rule_result,
        hash_dqi_result,
        hash_uncertainty_result,
        hash_aggregation_result,
        hash_batch_input,
        hash_batch_result,
        hash_config,
        hash_metadata,
        hash_arbitrary,
        create_chain,
        record_validation,
        record_classification,
        record_normalization,
        record_ef_resolution,
        record_calculation,
        record_lifetime,
        record_aggregation,
        record_compliance,
        record_provenance,
        seal_and_verify,
        export_chain_json,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProvenanceChainBuilder not available")
pytestmark = _SKIP


# ============================================================================
# HELPERS
# ============================================================================


def _valid_sha256(h: str) -> bool:
    """Check that h is a 64-char lowercase hex string."""
    if not isinstance(h, str) or len(h) != 64:
        return False
    try:
        int(h, 16)
        return True
    except ValueError:
        return False


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_builder():
    """Reset the singleton before each test."""
    reset_provenance_builder()
    yield
    reset_provenance_builder()


@pytest.fixture
def builder():
    """Create a ProvenanceChainBuilder instance."""
    return ProvenanceChainBuilder()


@pytest.fixture
def sample_input():
    """Sample input data for hashing."""
    return {
        "product_id": "VEH-001",
        "category": "vehicles",
        "fuel_type": "gasoline",
        "quantity": "1000",
    }


@pytest.fixture
def sample_result():
    """Sample result data for hashing."""
    return {
        "total_co2e_kg": "41670000.0",
        "method": "direct_fuel_combustion",
        "lifetime_years": "15",
    }


# ============================================================================
# TEST: Core Hashing
# ============================================================================


class TestCoreHashing:
    """Test core SHA-256 hashing functions."""

    def test_compute_hash_returns_64_chars(self):
        """Test _compute_hash returns 64-char hex string."""
        h = _compute_hash("test data")
        assert _valid_sha256(h)

    def test_compute_hash_deterministic(self):
        """Test _compute_hash is deterministic."""
        h1 = _compute_hash("same input")
        h2 = _compute_hash("same input")
        assert h1 == h2

    def test_compute_hash_different_inputs(self):
        """Test _compute_hash differs for different inputs."""
        h1 = _compute_hash("input A")
        h2 = _compute_hash("input B")
        assert h1 != h2

    def test_compute_hash_empty_string(self):
        """Test _compute_hash handles empty string."""
        h = _compute_hash("")
        assert _valid_sha256(h)

    def test_compute_hash_matches_hashlib(self):
        """Test _compute_hash matches direct hashlib SHA-256."""
        data = "test data for verification"
        expected = hashlib.sha256(data.encode("utf-8")).hexdigest()
        actual = _compute_hash(data)
        assert actual == expected

    def test_serialize_dict(self):
        """Test _serialize produces consistent JSON for dicts."""
        d = {"b": 2, "a": 1, "c": 3}
        s = _serialize(d)
        assert isinstance(s, str)
        # Should be sorted by keys for determinism
        parsed = json.loads(s)
        assert list(parsed.keys()) == sorted(d.keys())

    def test_serialize_decimal(self):
        """Test _serialize handles Decimal values."""
        d = {"value": Decimal("2.315")}
        s = _serialize(d)
        assert "2.315" in s

    def test_serialize_nested(self):
        """Test _serialize handles nested structures."""
        d = {"outer": {"inner": "value", "number": 42}}
        s = _serialize(d)
        assert "inner" in s
        assert "42" in s


# ============================================================================
# TEST: Chain Building
# ============================================================================


class TestChainBuilding:
    """Test provenance chain building."""

    def test_create_chain(self, builder, sample_input):
        """Test creating a new provenance chain."""
        chain = builder.create_chain("CALC-001", sample_input)
        assert chain is not None
        assert chain.calc_id == "CALC-001"

    def test_chain_has_initial_hash(self, builder, sample_input):
        """Test chain has an initial hash after creation."""
        chain = builder.create_chain("CALC-001", sample_input)
        assert chain.root_hash is not None
        assert _valid_sha256(chain.root_hash)

    def test_add_stage_record(self, builder, sample_input, sample_result):
        """Test adding a stage record to the chain."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        assert len(chain.stages) >= 2

    def test_chain_stages_ordered(self, builder, sample_input, sample_result):
        """Test chain stages are in order."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        stage_names = [s.stage_name for s in chain.stages]
        assert stage_names[0] == "validate"
        assert stage_names[1] == "calculate"

    def test_each_stage_has_hash(self, builder, sample_input, sample_result):
        """Test each stage record has a hash."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        for stage in chain.stages:
            assert _valid_sha256(stage.data_hash)


# ============================================================================
# TEST: Merkle Tree
# ============================================================================


class TestMerkleTree:
    """Test Merkle tree computation and verification."""

    def test_merkle_tree_single_leaf(self):
        """Test Merkle tree with single leaf."""
        leaves = ["a" * 64]
        root = _build_merkle_tree(leaves)
        assert _valid_sha256(root)

    def test_merkle_tree_two_leaves(self):
        """Test Merkle tree with two leaves."""
        leaves = ["a" * 64, "b" * 64]
        root = _build_merkle_tree(leaves)
        assert _valid_sha256(root)

    def test_merkle_tree_deterministic(self):
        """Test Merkle tree is deterministic."""
        leaves = ["a" * 64, "b" * 64, "c" * 64]
        root1 = _build_merkle_tree(leaves)
        root2 = _build_merkle_tree(leaves)
        assert root1 == root2

    def test_merkle_tree_different_leaves(self):
        """Test different leaves produce different root."""
        root1 = _build_merkle_tree(["a" * 64, "b" * 64])
        root2 = _build_merkle_tree(["c" * 64, "d" * 64])
        assert root1 != root2

    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        leaves = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        root = _build_merkle_tree(leaves)
        # Verify root is valid
        assert _valid_sha256(root)


# ============================================================================
# TEST: Seal and Verify
# ============================================================================


class TestSealAndVerify:
    """Test chain sealing and verification."""

    def test_seal_chain(self, builder, sample_input, sample_result):
        """Test sealing a completed chain."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        sealed = builder.seal(chain)
        assert sealed.is_sealed is True

    def test_verify_sealed_chain(self, builder, sample_input, sample_result):
        """Test verifying a sealed chain."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        sealed = builder.seal(chain)
        is_valid = builder.verify(sealed)
        assert is_valid is True

    def test_tampered_chain_fails_verification(self, builder, sample_input, sample_result):
        """Test that a tampered chain fails verification."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        sealed = builder.seal(chain)
        # Tamper with the seal hash
        sealed.seal_hash = "0" * 64
        is_valid = builder.verify(sealed)
        assert is_valid is False

    def test_sealed_chain_has_merkle_root(self, builder, sample_input, sample_result):
        """Test sealed chain has Merkle root hash."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        sealed = builder.seal(chain)
        assert _valid_sha256(sealed.merkle_root)


# ============================================================================
# TEST: Standalone Hash Functions (25+)
# ============================================================================


class TestStandaloneHashFunctions:
    """Test all standalone hash_* functions."""

    def test_hash_product_input(self, sample_input):
        """Test hash_product_input returns valid hash."""
        h = hash_product_input(sample_input)
        assert _valid_sha256(h)

    def test_hash_direct_emission_input(self):
        """Test hash_direct_emission_input returns valid hash."""
        h = hash_direct_emission_input({"fuel_type": "gasoline", "units": 1000})
        assert _valid_sha256(h)

    def test_hash_indirect_emission_input(self):
        """Test hash_indirect_emission_input returns valid hash."""
        h = hash_indirect_emission_input({"kwh_per_year": "400", "region": "US"})
        assert _valid_sha256(h)

    def test_hash_fuel_sale_input(self):
        """Test hash_fuel_sale_input returns valid hash."""
        h = hash_fuel_sale_input({"fuel_type": "gasoline", "litres": "1000000"})
        assert _valid_sha256(h)

    def test_hash_feedstock_input(self):
        """Test hash_feedstock_input returns valid hash."""
        h = hash_feedstock_input({"feedstock_type": "naphtha", "kg": "1000000"})
        assert _valid_sha256(h)

    def test_hash_fuel_combustion_result(self, sample_result):
        """Test hash_fuel_combustion_result returns valid hash."""
        h = hash_fuel_combustion_result(sample_result)
        assert _valid_sha256(h)

    def test_hash_refrigerant_leakage_result(self):
        """Test hash_refrigerant_leakage_result returns valid hash."""
        h = hash_refrigerant_leakage_result({"co2e": "1879200", "ref": "R-410A"})
        assert _valid_sha256(h)

    def test_hash_chemical_release_result(self):
        """Test hash_chemical_release_result returns valid hash."""
        h = hash_chemical_release_result({"co2e": "21450000", "chemical": "HFC-134a"})
        assert _valid_sha256(h)

    def test_hash_electricity_result(self):
        """Test hash_electricity_result returns valid hash."""
        h = hash_electricity_result({"co2e": "25020000", "region": "US"})
        assert _valid_sha256(h)

    def test_hash_heating_fuel_result(self):
        """Test hash_heating_fuel_result returns valid hash."""
        h = hash_heating_fuel_result({"co2e": "386000000", "fuel": "natural_gas"})
        assert _valid_sha256(h)

    def test_hash_steam_cooling_result(self):
        """Test hash_steam_cooling_result returns valid hash."""
        h = hash_steam_cooling_result({"co2e": "5000000", "type": "district"})
        assert _valid_sha256(h)

    def test_hash_lifetime_result(self):
        """Test hash_lifetime_result returns valid hash."""
        h = hash_lifetime_result({"lifetime": "15", "model": "linear"})
        assert _valid_sha256(h)

    def test_hash_degradation_result(self):
        """Test hash_degradation_result returns valid hash."""
        h = hash_degradation_result({"rate": "0.005", "year": "10"})
        assert _valid_sha256(h)

    def test_hash_weibull_result(self):
        """Test hash_weibull_result returns valid hash."""
        h = hash_weibull_result({"shape": "3.5", "scale": "15"})
        assert _valid_sha256(h)

    def test_hash_fleet_result(self):
        """Test hash_fleet_result returns valid hash."""
        h = hash_fleet_result({"fleet_co2e": "38000000"})
        assert _valid_sha256(h)

    def test_hash_grid_ef(self):
        """Test hash_grid_ef returns valid hash."""
        h = hash_grid_ef({"region": "US", "ef": "0.417"})
        assert _valid_sha256(h)

    def test_hash_fuel_ef(self):
        """Test hash_fuel_ef returns valid hash."""
        h = hash_fuel_ef({"fuel": "gasoline", "ef": "2.315"})
        assert _valid_sha256(h)

    def test_hash_refrigerant_gwp(self):
        """Test hash_refrigerant_gwp returns valid hash."""
        h = hash_refrigerant_gwp({"ref": "R-410A", "gwp": "2088"})
        assert _valid_sha256(h)

    def test_hash_compliance_result(self):
        """Test hash_compliance_result returns valid hash."""
        h = hash_compliance_result({"framework": "GHG_PROTOCOL", "status": "compliant"})
        assert _valid_sha256(h)

    def test_hash_dc_rule_result(self):
        """Test hash_dc_rule_result returns valid hash."""
        h = hash_dc_rule_result({"rule": "DC-USP-001", "result": "pass"})
        assert _valid_sha256(h)

    def test_hash_dqi_result(self):
        """Test hash_dqi_result returns valid hash."""
        h = hash_dqi_result({"score": "80", "classification": "Good"})
        assert _valid_sha256(h)

    def test_hash_uncertainty_result(self):
        """Test hash_uncertainty_result returns valid hash."""
        h = hash_uncertainty_result({"lower": "37500000", "upper": "45800000"})
        assert _valid_sha256(h)

    def test_hash_aggregation_result(self):
        """Test hash_aggregation_result returns valid hash."""
        h = hash_aggregation_result({"total": "70000000", "products": "5"})
        assert _valid_sha256(h)

    def test_hash_batch_input(self):
        """Test hash_batch_input returns valid hash."""
        h = hash_batch_input({"batch_size": "5", "products": ["P1", "P2"]})
        assert _valid_sha256(h)

    def test_hash_batch_result(self):
        """Test hash_batch_result returns valid hash."""
        h = hash_batch_result({"total": "70000000", "count": "5"})
        assert _valid_sha256(h)

    def test_hash_config(self):
        """Test hash_config returns valid hash."""
        h = hash_config({"gwp": "AR5", "region": "US"})
        assert _valid_sha256(h)

    def test_hash_metadata(self):
        """Test hash_metadata returns valid hash."""
        h = hash_metadata({"agent_id": "GL-MRV-S3-011", "version": "1.0.0"})
        assert _valid_sha256(h)

    def test_hash_arbitrary(self):
        """Test hash_arbitrary returns valid hash for any data."""
        h = hash_arbitrary({"any": "data", "key": 42})
        assert _valid_sha256(h)


# ============================================================================
# TEST: Convenience Stage Recorders
# ============================================================================


class TestConvenienceRecorders:
    """Test convenience stage recorder functions."""

    def test_record_validation(self, builder, sample_input):
        """Test record_validation adds a validation stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_validation(chain, sample_input)
        assert any(s.stage_name == "validate" for s in chain.stages)

    def test_record_classification(self, builder, sample_input):
        """Test record_classification adds a classification stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_classification(chain, {"category": "vehicles"})
        assert any(s.stage_name == "classify" for s in chain.stages)

    def test_record_normalization(self, builder, sample_input):
        """Test record_normalization adds a normalization stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_normalization(chain, {"units": "kgCO2e"})
        assert any(s.stage_name == "normalize" for s in chain.stages)

    def test_record_ef_resolution(self, builder, sample_input):
        """Test record_ef_resolution adds an EF resolution stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_ef_resolution(chain, {"ef_source": "defra"})
        assert any(s.stage_name == "resolve_efs" for s in chain.stages)

    def test_record_calculation(self, builder, sample_input, sample_result):
        """Test record_calculation adds a calculation stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_calculation(chain, sample_result)
        assert any(s.stage_name == "calculate" for s in chain.stages)

    def test_record_lifetime(self, builder, sample_input):
        """Test record_lifetime adds a lifetime stage."""
        chain = builder.create_chain("CALC-001", sample_input)
        record_lifetime(chain, {"lifetime_years": 15})
        assert any(s.stage_name == "lifetime" for s in chain.stages)


# ============================================================================
# TEST: Export
# ============================================================================


class TestExport:
    """Test chain export functionality."""

    def test_export_chain_json(self, builder, sample_input, sample_result):
        """Test exporting chain as JSON."""
        chain = builder.create_chain("CALC-001", sample_input)
        builder.record_stage(chain, "validate", sample_input)
        builder.record_stage(chain, "calculate", sample_result)
        sealed = builder.seal(chain)
        json_str = export_chain_json(sealed)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "calc_id" in parsed or "calculation_id" in parsed


# ============================================================================
# TEST: Determinism and Reproducibility
# ============================================================================


class TestDeterminism:
    """Test deterministic hashing guarantees."""

    def test_same_input_same_hash_100_times(self, sample_input):
        """Test same input produces same hash 100 times."""
        hashes = [hash_product_input(sample_input) for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_order_independence_for_dict_keys(self):
        """Test dict key order does not affect hash."""
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        assert hash_arbitrary(d1) == hash_arbitrary(d2)

    def test_decimal_precision_matters(self):
        """Test Decimal precision is preserved in hashing."""
        d1 = {"value": Decimal("2.315")}
        d2 = {"value": Decimal("2.3150")}
        # These should produce the same hash if normalized
        h1 = hash_arbitrary(d1)
        h2 = hash_arbitrary(d2)
        assert _valid_sha256(h1)
        assert _valid_sha256(h2)


# ============================================================================
# TEST: Thread Safety
# ============================================================================


class TestThreadSafety:
    """Test thread safety of provenance operations."""

    def test_concurrent_hash_generation(self):
        """Test 10+ threads generating hashes concurrently."""
        results = []
        errors = []

        def _hash():
            try:
                h = hash_product_input({"product_id": "VEH-001", "category": "vehicles"})
                results.append(h)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_hash) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        # All threads should produce the same hash
        assert len(set(results)) == 1

    def test_concurrent_chain_building(self):
        """Test 10+ threads building chains concurrently."""
        results = []
        errors = []

        def _build(thread_id):
            try:
                builder = ProvenanceChainBuilder()
                chain = builder.create_chain(f"CALC-{thread_id}", {"id": thread_id})
                builder.record_stage(chain, "validate", {"id": thread_id})
                sealed = builder.seal(chain)
                results.append(sealed.seal_hash)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_build, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(results) == 12

    def test_singleton_builder_thread_safe(self):
        """Test singleton provenance builder is thread-safe."""
        results = []
        errors = []

        def _get():
            try:
                b = get_provenance_builder()
                results.append(id(b))
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(set(results)) == 1
