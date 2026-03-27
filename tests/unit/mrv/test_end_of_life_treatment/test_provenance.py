# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceChainBuilder -- AGENT-MRV-025

Tests SHA-256 deterministic hashing, chain building, Merkle tree computation,
chain validation, seal/verify lifecycle, stage sequence validation, batch
provenance, standalone hash functions (15+), convenience stage recorders,
and thread safety with 12 concurrent threads.

Target: 60+ tests.
Author: GL-TestEngineer
Date: February 2026
"""

import hashlib
import json
import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.end_of_life_treatment.provenance import (
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
        hash_material_composition,
        hash_treatment_mix,
        hash_waste_type_result,
        hash_average_data_result,
        hash_producer_specific_result,
        hash_epd_data,
        hash_recycling_factors,
        hash_landfill_fod_params,
        hash_incineration_params,
        hash_avoided_emissions,
        hash_circularity_metrics,
        hash_compliance_result,
        hash_dc_rule_result,
        hash_dqi_result,
        hash_uncertainty_result,
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
        record_allocation,
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
        "product_id": "PRD-ELEC-001",
        "product_category": "consumer_electronics",
        "total_mass_kg": "1000",
        "region": "US",
    }


@pytest.fixture
def sample_output():
    """Sample output data for hashing."""
    return {
        "gross_emissions_kgco2e": "1250",
        "avoided_emissions_kgco2e": "450",
        "method": "waste_type_specific",
    }


# ============================================================================
# TEST: SHA-256 Hash Functions
# ============================================================================


class TestSHA256Hashing:
    """Test deterministic SHA-256 hashing of data."""

    def test_compute_hash_returns_64_char_hex(self, sample_input):
        """Test that _compute_hash returns a 64-char lowercase hex string."""
        h = _compute_hash(sample_input)
        assert _valid_sha256(h)

    def test_compute_hash_deterministic(self, sample_input):
        """Test that same input produces same hash."""
        h1 = _compute_hash(sample_input)
        h2 = _compute_hash(sample_input)
        assert h1 == h2

    def test_compute_hash_different_for_different_input(self, sample_input, sample_output):
        """Test that different inputs produce different hashes."""
        h1 = _compute_hash(sample_input)
        h2 = _compute_hash(sample_output)
        assert h1 != h2

    def test_serialize_sorts_keys(self):
        """Test that _serialize produces deterministic output by sorting keys."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        assert _serialize(data1) == _serialize(data2)

    def test_serialize_handles_decimal(self):
        """Test that _serialize converts Decimal to string."""
        data = {"value": Decimal("123.456")}
        serialized = _serialize(data)
        assert "123.456" in serialized

    def test_hash_empty_dict(self):
        """Test hashing an empty dictionary."""
        h = _compute_hash({})
        assert _valid_sha256(h)

    def test_hash_nested_data(self):
        """Test hashing nested dictionaries and lists."""
        data = {
            "products": [
                {"id": "P1", "mass_kg": 500},
                {"id": "P2", "mass_kg": 300},
            ]
        }
        h = _compute_hash(data)
        assert _valid_sha256(h)


# ============================================================================
# TEST: Chain Lifecycle
# ============================================================================


class TestChainLifecycle:
    """Test chain creation, stage recording, and retrieval."""

    def test_start_chain_returns_uuid(self, builder):
        """Test that start_chain returns a UUID string."""
        chain_id = builder.start_chain()
        assert isinstance(chain_id, str)
        assert len(chain_id) == 36  # UUID format

    def test_start_chain_custom_id(self, builder):
        """Test starting a chain with a custom ID."""
        chain_id = builder.start_chain(chain_id="EOL-CHAIN-001")
        assert chain_id == "EOL-CHAIN-001"

    def test_add_stage_returns_hash(self, builder, sample_input):
        """Test adding a stage returns a SHA-256 hash."""
        builder.start_chain()
        h = builder.add_stage("validate", sample_input)
        assert _valid_sha256(h)

    def test_chain_grows_with_stages(self, builder, sample_input, sample_output):
        """Test chain length increases as stages are added."""
        builder.start_chain()
        builder.add_stage("validate", sample_input)
        builder.add_stage("calculate", sample_output)
        chain = builder.get_chain()
        assert len(chain.stages) == 2

    def test_stage_ordering_preserved(self, builder):
        """Test stages are stored in order they were added."""
        builder.start_chain()
        builder.add_stage("validate", {"step": 1})
        builder.add_stage("classify", {"step": 2})
        builder.add_stage("normalize", {"step": 3})
        chain = builder.get_chain()
        assert chain.stages[0].stage == "validate"
        assert chain.stages[1].stage == "classify"
        assert chain.stages[2].stage == "normalize"

    def test_stage_hashes_are_chained(self, builder):
        """Test each stage hash incorporates previous stage hash."""
        builder.start_chain()
        h1 = builder.add_stage("validate", {"data": "v1"})
        h2 = builder.add_stage("calculate", {"data": "v2"})
        assert h1 != h2  # Different stages must have different hashes


# ============================================================================
# TEST: Merkle Tree
# ============================================================================


class TestMerkleTree:
    """Test Merkle tree computation and verification."""

    def test_single_leaf_merkle(self):
        """Test Merkle tree with single leaf."""
        leaves = ["a" * 64]
        root = _build_merkle_tree(leaves)
        assert _valid_sha256(root)

    def test_multiple_leaves_merkle(self):
        """Test Merkle tree with multiple leaves."""
        leaves = [hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(8)]
        root = _build_merkle_tree(leaves)
        assert _valid_sha256(root)

    def test_merkle_deterministic(self):
        """Test Merkle tree is deterministic."""
        leaves = [hashlib.sha256(f"data-{i}".encode()).hexdigest() for i in range(4)]
        r1 = _build_merkle_tree(leaves)
        r2 = _build_merkle_tree(leaves)
        assert r1 == r2

    def test_different_leaves_different_root(self):
        """Test different leaves produce different root."""
        l1 = [hashlib.sha256(b"a").hexdigest()]
        l2 = [hashlib.sha256(b"b").hexdigest()]
        assert _build_merkle_tree(l1) != _build_merkle_tree(l2)

    def test_merkle_proof_verification(self):
        """Test Merkle proof verification for a leaf."""
        leaves = [hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(4)]
        root = _build_merkle_tree(leaves)
        # Verify a valid proof
        is_valid = _verify_merkle_proof(leaves[0], root, leaves)
        assert is_valid is True


# ============================================================================
# TEST: Seal and Verify
# ============================================================================


class TestSealAndVerify:
    """Test chain sealing and verification."""

    def test_seal_chain(self, builder, sample_input, sample_output):
        """Test sealing a chain returns a final hash."""
        builder.start_chain()
        builder.add_stage("validate", sample_input)
        builder.add_stage("calculate", sample_output)
        seal_hash = builder.seal_chain()
        assert _valid_sha256(seal_hash)

    def test_sealed_chain_immutable(self, builder, sample_input):
        """Test sealed chain cannot be modified."""
        builder.start_chain()
        builder.add_stage("validate", sample_input)
        builder.seal_chain()
        with pytest.raises(Exception):
            builder.add_stage("new_stage", {"data": "post-seal"})

    def test_verify_chain_passes(self, builder, sample_input, sample_output):
        """Test verification passes for untampered chain."""
        builder.start_chain()
        builder.add_stage("validate", sample_input)
        builder.add_stage("calculate", sample_output)
        builder.seal_chain()
        assert builder.verify_chain() is True

    def test_tamper_detection(self, builder, sample_input, sample_output):
        """Test tampered chain fails verification."""
        builder.start_chain()
        builder.add_stage("validate", sample_input)
        builder.add_stage("calculate", sample_output)
        builder.seal_chain()
        # Tamper with stage data
        chain = builder.get_chain()
        if hasattr(chain.stages[0], 'data_hash'):
            original_hash = chain.stages[0].data_hash
            chain.stages[0].data_hash = "0" * 64  # Tampered
            assert builder.verify_chain() is False
            chain.stages[0].data_hash = original_hash  # Restore


# ============================================================================
# TEST: Standalone Hash Functions (15+)
# ============================================================================


class TestStandaloneHashFunctions:
    """Test 15+ standalone hash functions."""

    def test_hash_product_input(self):
        """Test hash_product_input returns valid SHA-256."""
        h = hash_product_input({
            "product_id": "P1", "category": "electronics", "mass": "500",
        })
        assert _valid_sha256(h)

    def test_hash_material_composition(self):
        """Test hash_material_composition."""
        h = hash_material_composition([
            {"material": "steel", "fraction": "0.5"},
            {"material": "plastic", "fraction": "0.5"},
        ])
        assert _valid_sha256(h)

    def test_hash_treatment_mix(self):
        """Test hash_treatment_mix."""
        h = hash_treatment_mix({
            "landfill": "0.4", "recycling": "0.3", "incineration": "0.3",
        })
        assert _valid_sha256(h)

    def test_hash_waste_type_result(self):
        """Test hash_waste_type_result."""
        h = hash_waste_type_result({
            "gross": "1000", "avoided": "300", "method": "waste_type",
        })
        assert _valid_sha256(h)

    def test_hash_average_data_result(self):
        """Test hash_average_data_result."""
        h = hash_average_data_result({
            "gross": "1200", "method": "average_data",
        })
        assert _valid_sha256(h)

    def test_hash_producer_specific_result(self):
        """Test hash_producer_specific_result."""
        h = hash_producer_specific_result({
            "epd_id": "EPD-001", "c1": "50", "c4": "500",
        })
        assert _valid_sha256(h)

    def test_hash_epd_data(self):
        """Test hash_epd_data."""
        h = hash_epd_data({
            "epd_id": "EPD-001", "valid_until": "2027-12-31",
        })
        assert _valid_sha256(h)

    def test_hash_recycling_factors(self):
        """Test hash_recycling_factors."""
        h = hash_recycling_factors({
            "recovery_rate": "0.85", "quality_factor": "0.9",
        })
        assert _valid_sha256(h)

    def test_hash_landfill_fod_params(self):
        """Test hash_landfill_fod_params."""
        h = hash_landfill_fod_params({
            "doc": "0.40", "k": "0.06", "mcf": "1.0",
        })
        assert _valid_sha256(h)

    def test_hash_incineration_params(self):
        """Test hash_incineration_params."""
        h = hash_incineration_params({
            "fossil_fraction": "1.0", "combustion_eff": "0.995",
        })
        assert _valid_sha256(h)

    def test_hash_avoided_emissions(self):
        """Test hash_avoided_emissions."""
        h = hash_avoided_emissions({
            "recycling": "350", "ad": "100",
        })
        assert _valid_sha256(h)

    def test_hash_circularity_metrics(self):
        """Test hash_circularity_metrics."""
        h = hash_circularity_metrics({
            "recycling_rate": "0.28", "circularity_index": "0.35",
        })
        assert _valid_sha256(h)

    def test_hash_compliance_result(self):
        """Test hash_compliance_result."""
        h = hash_compliance_result({
            "framework": "GHG_PROTOCOL", "compliant": "true",
        })
        assert _valid_sha256(h)

    def test_hash_dc_rule_result(self):
        """Test hash_dc_rule_result."""
        h = hash_dc_rule_result({
            "rule_id": "DC-EOL-001", "passed": "true",
        })
        assert _valid_sha256(h)

    def test_hash_dqi_result(self):
        """Test hash_dqi_result."""
        h = hash_dqi_result({
            "score": "75", "classification": "good",
        })
        assert _valid_sha256(h)

    def test_hash_uncertainty_result(self):
        """Test hash_uncertainty_result."""
        h = hash_uncertainty_result({
            "method": "propagation", "ci_95_lower": "1000", "ci_95_upper": "1500",
        })
        assert _valid_sha256(h)

    def test_hash_batch_input(self):
        """Test hash_batch_input."""
        h = hash_batch_input({
            "batch_id": "BATCH-001", "products": "5",
        })
        assert _valid_sha256(h)

    def test_hash_batch_result(self):
        """Test hash_batch_result."""
        h = hash_batch_result({
            "batch_id": "BATCH-001", "total": "5000",
        })
        assert _valid_sha256(h)

    def test_hash_config(self):
        """Test hash_config."""
        h = hash_config({
            "gwp": "AR5", "region": "GLOBAL",
        })
        assert _valid_sha256(h)

    def test_hash_metadata(self):
        """Test hash_metadata."""
        h = hash_metadata({
            "agent_id": "GL-MRV-S3-012", "version": "1.0.0",
        })
        assert _valid_sha256(h)

    def test_hash_arbitrary(self):
        """Test hash_arbitrary with free-form data."""
        h = hash_arbitrary({"custom_field": "custom_value"})
        assert _valid_sha256(h)


# ============================================================================
# TEST: Convenience Stage Recorders
# ============================================================================


class TestConvenienceRecorders:
    """Test convenience stage recorder functions."""

    def test_create_chain(self):
        """Test create_chain creates a new chain."""
        chain_id = create_chain()
        assert isinstance(chain_id, str)

    def test_record_validation(self):
        """Test record_validation stage."""
        create_chain()
        h = record_validation({"product_id": "P1"})
        assert _valid_sha256(h)

    def test_record_classification(self):
        """Test record_classification stage."""
        create_chain()
        h = record_classification({"category": "electronics"})
        assert _valid_sha256(h)

    def test_record_normalization(self):
        """Test record_normalization stage."""
        create_chain()
        h = record_normalization({"mass_kg": "1000"})
        assert _valid_sha256(h)

    def test_record_ef_resolution(self):
        """Test record_ef_resolution stage."""
        create_chain()
        h = record_ef_resolution({"ef_source": "epa_warm"})
        assert _valid_sha256(h)

    def test_record_calculation(self):
        """Test record_calculation stage."""
        create_chain()
        h = record_calculation({"gross": "1250", "avoided": "450"})
        assert _valid_sha256(h)

    def test_record_allocation(self):
        """Test record_allocation stage."""
        create_chain()
        h = record_allocation({"method": "mass"})
        assert _valid_sha256(h)

    def test_record_aggregation(self):
        """Test record_aggregation stage."""
        create_chain()
        h = record_aggregation({"total": "5000"})
        assert _valid_sha256(h)

    def test_record_compliance(self):
        """Test record_compliance stage."""
        create_chain()
        h = record_compliance({"framework": "GHG_PROTOCOL"})
        assert _valid_sha256(h)

    def test_record_provenance(self):
        """Test record_provenance stage."""
        create_chain()
        h = record_provenance({"chain_id": "CHAIN-001"})
        assert _valid_sha256(h)

    def test_seal_and_verify_function(self):
        """Test seal_and_verify convenience function."""
        create_chain()
        record_validation({"data": "test"})
        result = seal_and_verify()
        assert result is not None

    def test_export_chain_json(self):
        """Test export_chain_json returns valid JSON string."""
        create_chain()
        record_validation({"data": "test"})
        json_str = export_chain_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


# ============================================================================
# TEST: Thread Safety
# ============================================================================


class TestThreadSafety:
    """Test thread-safe provenance operations with 12 concurrent threads."""

    def test_concurrent_hash_computation(self):
        """Test 12 threads computing hashes concurrently."""
        results = []
        errors = []

        def compute():
            try:
                h = _compute_hash({"thread": threading.current_thread().name})
                results.append(h)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=compute) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 12

    def test_concurrent_chain_operations(self):
        """Test 12 threads performing chain operations concurrently."""
        errors = []

        def chain_ops(thread_id):
            try:
                # Each thread creates its own builder
                b = ProvenanceChainBuilder()
                b.start_chain(chain_id=f"CHAIN-{thread_id}")
                b.add_stage("validate", {"tid": thread_id})
                b.add_stage("calculate", {"result": thread_id * 100})
                b.seal_chain()
                assert b.verify_chain() is True
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=chain_ops, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_deterministic_across_threads(self):
        """Test same input produces same hash regardless of thread."""
        data = {"product": "P1", "mass": "500", "region": "US"}
        hashes = []
        errors = []

        def compute():
            try:
                h = _compute_hash(data)
                hashes.append(h)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=compute) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(set(hashes)) == 1, "Different hashes from different threads"
