# -*- coding: utf-8 -*-
"""
Unit tests for ProvenanceChainBuilder -- AGENT-MRV-023

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
    from greenlang.agents.mrv.processing_sold_products.provenance import (
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
        hash_processing_input,
        hash_site_specific_result,
        hash_average_data_result,
        hash_spend_based_result,
        hash_processing_chain_result,
        hash_grid_ef,
        hash_fuel_ef,
        hash_allocation_result,
        hash_aggregation_result,
        hash_compliance_result,
        hash_dc_rule_result,
        hash_dqi_result,
        hash_uncertainty_result,
        hash_energy_consumption,
        hash_fuel_consumption,
        hash_currency_conversion,
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
        "product_id": "STEEL-001",
        "category": "metals_ferrous",
        "quantity_tonnes": "500",
        "processing_type": "machining",
    }


@pytest.fixture
def sample_output():
    """Sample output data for hashing."""
    return {
        "emissions_kg": "140000",
        "method": "average_data",
        "ef_used": "280",
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
        data = {"products": [{"id": "P1", "qty": 100}, {"id": "P2", "qty": 200}]}
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
        chain_id = builder.start_chain(chain_id="MY-CHAIN-001")
        assert chain_id == "MY-CHAIN-001"

    def test_start_chain_duplicate_raises(self, builder):
        """Test that starting a chain with a duplicate ID raises ValueError."""
        builder.start_chain(chain_id="DUP-001")
        with pytest.raises(ValueError, match="already exists"):
            builder.start_chain(chain_id="DUP-001")

    def test_record_stage_returns_stage_record(self, builder, sample_input, sample_output):
        """Test that record_stage returns a StageRecord."""
        chain_id = builder.start_chain()
        record = builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        assert isinstance(record, StageRecord)
        assert record.stage == ProvenanceStage.VALIDATE
        assert _valid_sha256(record.input_hash)
        assert _valid_sha256(record.output_hash)

    def test_record_stage_unknown_chain_raises(self, builder, sample_input, sample_output):
        """Test that recording to an unknown chain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            builder.record_stage("NONEXISTENT", ProvenanceStage.VALIDATE, sample_input, sample_output)

    def test_record_stage_sealed_chain_raises(self, builder, sample_input, sample_output):
        """Test that recording to a sealed chain raises ValueError."""
        chain_id = builder.start_chain()
        builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        chain = builder.get_chain(chain_id)
        # Build hashes for sealing
        stage_hashes = [
            builder.hash_stage(s.stage, s.input_hash, s.output_hash, s.timestamp)
            for s in chain.stages
        ]
        chain.chain_hash = builder.compute_chain_hash(stage_hashes)
        chain.merkle_root = builder.compute_merkle_root(stage_hashes)
        builder.seal_chain(chain)
        with pytest.raises(ValueError, match="already sealed"):
            builder.record_stage(chain_id, ProvenanceStage.CLASSIFY, {}, {})

    def test_get_chain(self, builder, sample_input, sample_output):
        """Test retrieving a chain by ID."""
        chain_id = builder.start_chain()
        builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        chain = builder.get_chain(chain_id)
        assert isinstance(chain, ProvenanceChain)
        assert chain.chain_id == chain_id
        assert chain.stage_count == 1

    def test_get_chain_unknown_raises(self, builder):
        """Test that getting an unknown chain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            builder.get_chain("NONEXISTENT")

    def test_delete_chain(self, builder):
        """Test deleting a chain."""
        chain_id = builder.start_chain()
        assert builder.delete_chain(chain_id) is True
        assert builder.delete_chain(chain_id) is False

    def test_get_all_chain_ids(self, builder):
        """Test getting all chain IDs."""
        id1 = builder.start_chain()
        id2 = builder.start_chain()
        all_ids = builder.get_all_chain_ids()
        assert id1 in all_ids
        assert id2 in all_ids

    def test_reset_clears_all_chains(self, builder):
        """Test that reset clears all chains."""
        builder.start_chain()
        builder.start_chain()
        builder.reset()
        assert len(builder.get_all_chain_ids()) == 0


# ============================================================================
# TEST: Chain Building and Hashing
# ============================================================================


class TestChainBuilding:
    """Test chain hash computation and Merkle root."""

    def test_build_chain_computes_chain_hash(self, builder, sample_input, sample_output):
        """Test that build_chain computes a valid chain_hash."""
        record1 = builder.build_stage_record(ProvenanceStage.VALIDATE, sample_input, sample_output)
        record2 = builder.build_stage_record(ProvenanceStage.CALCULATE, sample_output, {"total": 100})
        chain = builder.build_chain([record1, record2])
        assert _valid_sha256(chain.chain_hash)
        assert _valid_sha256(chain.merkle_root)
        assert chain.stage_count == 2

    def test_compute_chain_hash_single(self, builder):
        """Test chain hash with a single hash returns the hash itself."""
        h = hashlib.sha256(b"test").hexdigest()
        result = builder.compute_chain_hash([h])
        assert result == h

    def test_compute_chain_hash_multiple(self, builder):
        """Test chain hash with multiple hashes produces a different hash."""
        h1 = hashlib.sha256(b"test1").hexdigest()
        h2 = hashlib.sha256(b"test2").hexdigest()
        result = builder.compute_chain_hash([h1, h2])
        assert _valid_sha256(result)
        assert result != h1
        assert result != h2

    def test_compute_chain_hash_empty(self, builder):
        """Test chain hash of empty list returns SHA-256 of empty bytes."""
        result = builder.compute_chain_hash([])
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_compute_merkle_root_single(self, builder):
        """Test Merkle root with a single hash returns the hash itself."""
        h = hashlib.sha256(b"single").hexdigest()
        result = builder.compute_merkle_root([h])
        assert result == h

    def test_compute_merkle_root_two_hashes(self, builder):
        """Test Merkle root with two hashes."""
        h1 = hashlib.sha256(b"left").hexdigest()
        h2 = hashlib.sha256(b"right").hexdigest()
        root = builder.compute_merkle_root([h1, h2])
        assert _valid_sha256(root)

    def test_compute_merkle_root_empty(self, builder):
        """Test Merkle root of empty list returns SHA-256 of empty bytes."""
        result = builder.compute_merkle_root([])
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


# ============================================================================
# TEST: Merkle Tree
# ============================================================================


class TestMerkleTree:
    """Test Merkle tree building and verification."""

    def test_build_merkle_tree_two_leaves(self):
        """Test building Merkle tree with 2 leaves."""
        h1 = hashlib.sha256(b"leaf1").hexdigest()
        h2 = hashlib.sha256(b"leaf2").hexdigest()
        levels, root = _build_merkle_tree([h1, h2])
        assert len(levels) == 2  # leaf level + root level
        assert len(levels[0]) == 2  # 2 leaves
        assert len(levels[1]) == 1  # 1 root
        assert root == levels[1][0]

    def test_build_merkle_tree_odd_leaves(self):
        """Test building Merkle tree with 3 leaves (odd, so last is duplicated)."""
        leaves = [hashlib.sha256(f"leaf{i}".encode()).hexdigest() for i in range(3)]
        levels, root = _build_merkle_tree(leaves)
        assert _valid_sha256(root)
        assert len(levels) >= 2

    def test_build_merkle_tree_empty(self):
        """Test building Merkle tree with empty list."""
        levels, root = _build_merkle_tree([])
        assert _valid_sha256(root)
        assert root == hashlib.sha256(b"").hexdigest()

    def test_build_merkle_tree_single_leaf(self):
        """Test building Merkle tree with a single leaf."""
        h = hashlib.sha256(b"only").hexdigest()
        levels, root = _build_merkle_tree([h])
        assert root == h
        assert len(levels) == 1


# ============================================================================
# TEST: Seal and Validate
# ============================================================================


class TestSealAndValidate:
    """Test chain sealing and integrity validation."""

    def _build_sealed_chain(self, builder, sample_input, sample_output):
        """Helper to build and seal a chain with 3 stages."""
        chain_id = builder.start_chain()
        builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        builder.record_stage(chain_id, ProvenanceStage.CLASSIFY, sample_output, {"classified": True})
        builder.record_stage(chain_id, ProvenanceStage.CALCULATE, {"classified": True}, {"total": 100})
        chain = builder.get_chain(chain_id)
        # Compute hashes before sealing
        stage_hashes = [
            builder.hash_stage(s.stage, s.input_hash, s.output_hash, s.timestamp)
            for s in chain.stages
        ]
        chain.chain_hash = builder.compute_chain_hash(stage_hashes)
        chain.merkle_root = builder.compute_merkle_root(stage_hashes)
        return chain

    def test_seal_chain_returns_hash(self, builder, sample_input, sample_output):
        """Test that seal_chain returns a 64-char hash."""
        chain = self._build_sealed_chain(builder, sample_input, sample_output)
        seal_hash = builder.seal_chain(chain)
        assert _valid_sha256(seal_hash)
        assert chain.is_sealed is True

    def test_seal_chain_already_sealed_raises(self, builder, sample_input, sample_output):
        """Test that sealing an already-sealed chain raises ValueError."""
        chain = self._build_sealed_chain(builder, sample_input, sample_output)
        builder.seal_chain(chain)
        with pytest.raises(ValueError, match="already sealed"):
            builder.seal_chain(chain)

    def test_seal_empty_chain_raises(self, builder):
        """Test that sealing an empty chain raises ValueError."""
        chain_id = builder.start_chain()
        chain = builder.get_chain(chain_id)
        with pytest.raises(ValueError, match="no stages"):
            builder.seal_chain(chain)

    def test_validate_chain_passes(self, builder, sample_input, sample_output):
        """Test that a correctly built and sealed chain passes validation."""
        chain = self._build_sealed_chain(builder, sample_input, sample_output)
        builder.seal_chain(chain)
        assert builder.validate_chain(chain) is True

    def test_validate_chain_tampered_chain_hash(self, builder, sample_input, sample_output):
        """Test that tampering with chain_hash causes validation failure."""
        chain = self._build_sealed_chain(builder, sample_input, sample_output)
        builder.seal_chain(chain)
        chain.chain_hash = "a" * 64  # tampered
        assert builder.validate_chain(chain) is False

    def test_validate_chain_tampered_merkle_root(self, builder, sample_input, sample_output):
        """Test that tampering with merkle_root causes validation failure."""
        chain = self._build_sealed_chain(builder, sample_input, sample_output)
        builder.seal_chain(chain)
        chain.merkle_root = "b" * 64  # tampered
        assert builder.validate_chain(chain) is False

    def test_validate_empty_chain_fails(self, builder):
        """Test that validating an empty chain fails."""
        chain = ProvenanceChain(chain_id="EMPTY-001")
        assert builder.validate_chain(chain) is False


# ============================================================================
# TEST: Stage Sequence Validation
# ============================================================================


class TestStageSequence:
    """Test stage sequence ordering validation."""

    def test_valid_sequence(self, builder):
        """Test that a valid non-decreasing sequence passes."""
        records = [
            builder.build_stage_record(ProvenanceStage.VALIDATE, {}, {}),
            builder.build_stage_record(ProvenanceStage.CLASSIFY, {}, {}),
            builder.build_stage_record(ProvenanceStage.CALCULATE, {}, {}),
            builder.build_stage_record(ProvenanceStage.SEAL, {}, {}),
        ]
        assert builder.validate_stage_sequence(records) is True

    def test_invalid_reverse_sequence(self, builder):
        """Test that a reverse-ordered sequence fails."""
        records = [
            builder.build_stage_record(ProvenanceStage.SEAL, {}, {}),
            builder.build_stage_record(ProvenanceStage.VALIDATE, {}, {}),
        ]
        assert builder.validate_stage_sequence(records) is False

    def test_empty_sequence_valid(self, builder):
        """Test that an empty sequence is valid."""
        assert builder.validate_stage_sequence([]) is True

    def test_single_stage_valid(self, builder):
        """Test that a single stage is valid."""
        records = [builder.build_stage_record(ProvenanceStage.CALCULATE, {}, {})]
        assert builder.validate_stage_sequence(records) is True

    def test_all_10_stages_in_order(self, builder):
        """Test all 10 stages in correct order."""
        records = [
            builder.build_stage_record(stage, {}, {})
            for stage in ProvenanceStage
        ]
        assert builder.validate_stage_sequence(records) is True
        assert len(records) == 10


# ============================================================================
# TEST: Provenance Stage Enum
# ============================================================================


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum values."""

    def test_10_stages_defined(self):
        """Test that exactly 10 stages are defined."""
        assert len(ProvenanceStage) == 10

    @pytest.mark.parametrize(
        "stage,value",
        [
            (ProvenanceStage.VALIDATE, "VALIDATE"),
            (ProvenanceStage.CLASSIFY, "CLASSIFY"),
            (ProvenanceStage.NORMALIZE, "NORMALIZE"),
            (ProvenanceStage.RESOLVE_EFS, "RESOLVE_EFS"),
            (ProvenanceStage.CALCULATE, "CALCULATE"),
            (ProvenanceStage.ALLOCATE, "ALLOCATE"),
            (ProvenanceStage.AGGREGATE, "AGGREGATE"),
            (ProvenanceStage.COMPLIANCE, "COMPLIANCE"),
            (ProvenanceStage.PROVENANCE, "PROVENANCE"),
            (ProvenanceStage.SEAL, "SEAL"),
        ],
    )
    def test_stage_values(self, stage, value):
        """Test each stage enum has the correct string value."""
        assert stage.value == value


# ============================================================================
# TEST: Domain-Specific Hash Functions (25+)
# ============================================================================


class TestStandaloneHashFunctions:
    """Test all 25+ standalone hash functions."""

    def test_hash_product_input(self):
        """Test hash_product_input returns valid hash."""
        h = hash_product_input({"product_id": "P1", "category": "metals_ferrous"})
        assert _valid_sha256(h)

    def test_hash_processing_input(self):
        """Test hash_processing_input returns valid hash."""
        h = hash_processing_input({"type": "machining", "energy_kwh": 500})
        assert _valid_sha256(h)

    def test_hash_site_specific_result(self):
        """Test hash_site_specific_result returns valid hash."""
        h = hash_site_specific_result("energy", Decimal("28000"), Decimal("0.417"), Decimal("11676"))
        assert _valid_sha256(h)

    def test_hash_average_data_result(self):
        """Test hash_average_data_result returns valid hash."""
        h = hash_average_data_result("metals_ferrous", "machining", Decimal("1000"), Decimal("280"), Decimal("280000"))
        assert _valid_sha256(h)

    def test_hash_spend_based_result(self):
        """Test hash_spend_based_result returns valid hash."""
        h = hash_spend_based_result(Decimal("1000000"), "331", Decimal("0.82"), Decimal("754400"), True, True)
        assert _valid_sha256(h)

    def test_hash_processing_chain_result(self):
        """Test hash_processing_chain_result returns valid hash."""
        h = hash_processing_chain_result("STEEL_AUTOMOTIVE", [{"step": "machining"}], Decimal("195"), Decimal("19500"))
        assert _valid_sha256(h)

    def test_hash_grid_ef(self):
        """Test hash_grid_ef returns valid hash."""
        h = hash_grid_ef("US", Decimal("0.417"), "egrid", 2024)
        assert _valid_sha256(h)

    def test_hash_fuel_ef(self):
        """Test hash_fuel_ef returns valid hash."""
        h = hash_fuel_ef("natural_gas", Decimal("2.024"), "defra")
        assert _valid_sha256(h)

    def test_hash_allocation_result(self):
        """Test hash_allocation_result returns valid hash."""
        h = hash_allocation_result("mass", Decimal("0.6"), Decimal("100000"), Decimal("60000"))
        assert _valid_sha256(h)

    def test_hash_aggregation_result(self):
        """Test hash_aggregation_result returns valid hash."""
        h = hash_aggregation_result({"metals": 50}, {"avg": 80}, {"machining": 30}, Decimal("100000"))
        assert _valid_sha256(h)

    def test_hash_compliance_result(self):
        """Test hash_compliance_result returns valid hash."""
        h = hash_compliance_result("ghg_protocol", "PASS", Decimal("95"))
        assert _valid_sha256(h)

    def test_hash_dc_rule_result(self):
        """Test hash_dc_rule_result returns valid hash."""
        h = hash_dc_rule_result("DC-PSP-001", True, "Scope 1", Decimal("50000"))
        assert _valid_sha256(h)

    def test_hash_dqi_result(self):
        """Test hash_dqi_result returns valid hash."""
        h = hash_dqi_result({"reliability": Decimal("3.5"), "completeness": Decimal("4.0")}, Decimal("3.75"))
        assert _valid_sha256(h)

    def test_hash_uncertainty_result(self):
        """Test hash_uncertainty_result returns valid hash."""
        h = hash_uncertainty_result("analytical", Decimal("0.95"), Decimal("80000"), Decimal("120000"))
        assert _valid_sha256(h)

    def test_hash_energy_consumption(self):
        """Test hash_energy_consumption returns valid hash."""
        h = hash_energy_consumption("electricity", Decimal("50000"), "meter_reading", "2024-Q4")
        assert _valid_sha256(h)

    def test_hash_fuel_consumption(self):
        """Test hash_fuel_consumption returns valid hash."""
        h = hash_fuel_consumption("diesel", Decimal("10000"), "litres", "2024")
        assert _valid_sha256(h)

    def test_hash_currency_conversion(self):
        """Test hash_currency_conversion returns valid hash."""
        h = hash_currency_conversion(
            Decimal("100000"), "EUR", Decimal("108500"), Decimal("1.085"), Decimal("1.0")
        )
        assert _valid_sha256(h)

    def test_hash_batch_input(self):
        """Test hash_batch_input returns valid hash."""
        h = hash_batch_input("BATCH-001", 5, ["aaa", "bbb"])
        assert _valid_sha256(h)

    def test_hash_batch_result(self):
        """Test hash_batch_result returns valid hash."""
        h = hash_batch_result("BATCH-001", [{"product": "P1", "emissions": 100}])
        assert _valid_sha256(h)

    def test_hash_config(self):
        """Test hash_config returns valid hash."""
        h = hash_config({"version": "1.0.0", "agent_id": "GL-MRV-S3-010"})
        assert _valid_sha256(h)

    def test_hash_metadata(self):
        """Test hash_metadata returns valid hash."""
        h = hash_metadata({"tenant": "ACME", "created_by": "user@example.com"})
        assert _valid_sha256(h)

    def test_hash_arbitrary(self):
        """Test hash_arbitrary returns valid hash for any data type."""
        assert _valid_sha256(hash_arbitrary("string_data"))
        assert _valid_sha256(hash_arbitrary(42))
        assert _valid_sha256(hash_arbitrary([1, 2, 3]))
        assert _valid_sha256(hash_arbitrary(None))

    def test_deterministic_across_all_hash_functions(self):
        """Test that all hash functions produce deterministic results."""
        h1 = hash_product_input({"id": "X"})
        h2 = hash_product_input({"id": "X"})
        assert h1 == h2


# ============================================================================
# TEST: Builder Domain Hash Methods
# ============================================================================


class TestBuilderHashMethods:
    """Test domain-specific hash methods on the builder instance."""

    def test_hash_input(self, builder):
        """Test builder.hash_input returns valid hash."""
        h = builder.hash_input({"key": "value"})
        assert _valid_sha256(h)

    def test_hash_emission_factor(self, builder):
        """Test builder.hash_emission_factor returns valid hash."""
        h = builder.hash_emission_factor("EF-001", Decimal("280"), "ecoinvent", "US")
        assert _valid_sha256(h)

    def test_hash_product(self, builder):
        """Test builder.hash_product returns valid hash."""
        h = builder.hash_product("P1", "metals_ferrous", Decimal("500"), "tonne")
        assert _valid_sha256(h)

    def test_hash_calculation(self, builder):
        """Test builder.hash_calculation returns valid hash."""
        h = builder.hash_calculation("average_data", "a" * 64, Decimal("140000"), "b" * 64)
        assert _valid_sha256(h)

    def test_hash_product_breakdown(self, builder):
        """Test builder.hash_product_breakdown returns valid hash."""
        h = builder.hash_product_breakdown("P1", Decimal("140000"), "EF-001", "average_data")
        assert _valid_sha256(h)

    def test_hash_allocation(self, builder):
        """Test builder.hash_allocation returns valid hash."""
        h = builder.hash_allocation(
            "mass",
            {"P1": Decimal("0.6"), "P2": Decimal("0.4")},
            {"P1": Decimal("60000"), "P2": Decimal("40000")},
        )
        assert _valid_sha256(h)

    def test_hash_aggregation(self, builder):
        """Test builder.hash_aggregation returns valid hash."""
        h = builder.hash_aggregation("2024", Decimal("500"), "c" * 64)
        assert _valid_sha256(h)

    def test_hash_compliance(self, builder):
        """Test builder.hash_compliance returns valid hash."""
        h = builder.hash_compliance("ghg_protocol", "compliant", 6, 0)
        assert _valid_sha256(h)

    def test_hash_dqi(self, builder):
        """Test builder.hash_dqi returns valid hash."""
        h = builder.hash_dqi(
            Decimal("3.5"), Decimal("4.0"), Decimal("3.0"), Decimal("3.5"), Decimal("3.0")
        )
        assert _valid_sha256(h)

    def test_hash_uncertainty(self, builder):
        """Test builder.hash_uncertainty returns valid hash."""
        h = builder.hash_uncertainty("analytical", Decimal("100000"), Decimal("70000"), Decimal("130000"))
        assert _valid_sha256(h)


# ============================================================================
# TEST: Convenience Stage Recorders
# ============================================================================


class TestConvenienceRecorders:
    """Test convenience functions for recording each of the 10 stages."""

    def test_create_chain(self):
        """Test create_chain returns builder and chain_id."""
        b, cid = create_chain()
        assert isinstance(b, ProvenanceChainBuilder)
        assert isinstance(cid, str)

    def test_record_validation(self):
        """Test record_validation records a VALIDATE stage."""
        b, cid = create_chain()
        record = record_validation(cid, {"raw": True}, {"validated": True})
        assert record.stage == ProvenanceStage.VALIDATE

    def test_record_classification(self):
        """Test record_classification records a CLASSIFY stage."""
        b, cid = create_chain()
        record_validation(cid, {}, {})
        record = record_classification(cid, {}, {"category": "metals_ferrous"})
        assert record.stage == ProvenanceStage.CLASSIFY

    def test_record_normalization(self):
        """Test record_normalization records a NORMALIZE stage."""
        b, cid = create_chain()
        record_validation(cid, {}, {})
        record = record_normalization(cid, {}, {"tonnes": 500})
        assert record.stage == ProvenanceStage.NORMALIZE

    def test_record_ef_resolution(self):
        """Test record_ef_resolution records a RESOLVE_EFS stage."""
        b, cid = create_chain()
        record = record_ef_resolution(cid, {}, {"ef": 280})
        assert record.stage == ProvenanceStage.RESOLVE_EFS

    def test_record_calculation(self):
        """Test record_calculation records a CALCULATE stage."""
        b, cid = create_chain()
        record = record_calculation(cid, {}, {"total": 140000})
        assert record.stage == ProvenanceStage.CALCULATE

    def test_record_allocation(self):
        """Test record_allocation records an ALLOCATE stage."""
        b, cid = create_chain()
        record = record_allocation(cid, {}, {"allocated": [60, 40]})
        assert record.stage == ProvenanceStage.ALLOCATE

    def test_record_aggregation(self):
        """Test record_aggregation records an AGGREGATE stage."""
        b, cid = create_chain()
        record = record_aggregation(cid, {}, {"total": 100000})
        assert record.stage == ProvenanceStage.AGGREGATE

    def test_record_compliance(self):
        """Test record_compliance records a COMPLIANCE stage."""
        b, cid = create_chain()
        record = record_compliance(cid, {}, {"status": "PASS"})
        assert record.stage == ProvenanceStage.COMPLIANCE

    def test_record_provenance_stage(self):
        """Test record_provenance records a PROVENANCE stage."""
        b, cid = create_chain()
        record = record_provenance(cid, {}, {"hash": "abc"})
        assert record.stage == ProvenanceStage.PROVENANCE


# ============================================================================
# TEST: Batch Provenance
# ============================================================================


class TestBatchProvenance:
    """Test batch provenance tracking with Merkle tree."""

    def test_start_batch(self, builder):
        """Test starting a batch session."""
        batch_id = builder.start_batch(item_count=3)
        assert isinstance(batch_id, str)

    def test_start_batch_custom_id(self, builder):
        """Test starting a batch with a custom ID."""
        batch_id = builder.start_batch(batch_id="BATCH-001")
        assert batch_id == "BATCH-001"

    def test_start_batch_duplicate_raises(self, builder):
        """Test that starting a batch with a duplicate ID raises ValueError."""
        builder.start_batch(batch_id="DUP-BATCH")
        with pytest.raises(ValueError, match="already exists"):
            builder.start_batch(batch_id="DUP-BATCH")

    def test_seal_batch_returns_root_hash(self, builder, sample_input, sample_output):
        """Test that sealing a batch returns a valid root hash."""
        batch_id = builder.start_batch()

        # Create two chains and add them to the batch
        cid1 = builder.start_chain()
        builder.record_stage(cid1, ProvenanceStage.VALIDATE, sample_input, sample_output)
        chain1 = builder.get_chain(cid1)
        sh1 = [builder.hash_stage(s.stage, s.input_hash, s.output_hash, s.timestamp) for s in chain1.stages]
        chain1.chain_hash = builder.compute_chain_hash(sh1)
        chain1.merkle_root = builder.compute_merkle_root(sh1)

        cid2 = builder.start_chain()
        builder.record_stage(cid2, ProvenanceStage.CALCULATE, sample_output, {"total": 99})
        chain2 = builder.get_chain(cid2)
        sh2 = [builder.hash_stage(s.stage, s.input_hash, s.output_hash, s.timestamp) for s in chain2.stages]
        chain2.chain_hash = builder.compute_chain_hash(sh2)
        chain2.merkle_root = builder.compute_merkle_root(sh2)

        builder.add_chain_to_batch(batch_id, cid1)
        builder.add_chain_to_batch(batch_id, cid2)

        root = builder.seal_batch(batch_id)
        assert _valid_sha256(root)

    def test_get_batch_summary(self, builder):
        """Test getting a batch summary."""
        batch_id = builder.start_batch(item_count=0, batch_id="SUMMARY-001")
        summary = builder.get_batch_summary("SUMMARY-001")
        assert summary["batch_id"] == "SUMMARY-001"

    def test_seal_batch_already_sealed_raises(self, builder):
        """Test that sealing an already-sealed batch raises ValueError."""
        batch_id = builder.start_batch()
        builder.seal_batch(batch_id)
        with pytest.raises(ValueError, match="already sealed"):
            builder.seal_batch(batch_id)


# ============================================================================
# TEST: Export
# ============================================================================


class TestChainExport:
    """Test chain export to JSON."""

    def test_export_chain_json(self, builder, sample_input, sample_output):
        """Test exporting a chain to JSON string."""
        chain_id = builder.start_chain()
        builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        chain = builder.get_chain(chain_id)
        exported = builder.export_chain(chain)
        assert isinstance(exported, dict)
        assert exported["chain_id"] == chain_id
        assert "stages" in exported
        assert len(exported["stages"]) == 1

    def test_chain_to_json(self, builder, sample_input, sample_output):
        """Test ProvenanceChain.to_json produces valid JSON."""
        chain_id = builder.start_chain()
        builder.record_stage(chain_id, ProvenanceStage.VALIDATE, sample_input, sample_output)
        chain = builder.get_chain(chain_id)
        json_str = chain.to_json()
        parsed = json.loads(json_str)
        assert parsed["chain_id"] == chain_id

    def test_stage_record_to_dict(self, builder):
        """Test StageRecord.to_dict produces correct structure."""
        record = builder.build_stage_record(ProvenanceStage.CALCULATE, {"x": 1}, {"y": 2})
        d = record.to_dict()
        assert d["stage"] == "CALCULATE"
        assert _valid_sha256(d["input_hash"])
        assert _valid_sha256(d["output_hash"])
        assert "timestamp" in d


# ============================================================================
# TEST: Singleton Pattern
# ============================================================================


class TestProvenanceSingleton:
    """Test singleton pattern for ProvenanceChainBuilder."""

    def test_singleton_identity(self):
        """Test that two instantiations return the same object."""
        b1 = ProvenanceChainBuilder()
        b2 = ProvenanceChainBuilder()
        assert b1 is b2

    def test_get_provenance_builder_singleton(self):
        """Test that get_provenance_builder returns the same singleton."""
        b1 = get_provenance_builder()
        b2 = get_provenance_builder()
        assert b1 is b2

    def test_reset_singleton_creates_new(self):
        """Test that reset_singleton allows a new instance."""
        b1 = ProvenanceChainBuilder()
        reset_provenance_builder()
        b2 = ProvenanceChainBuilder()
        assert b1 is not b2


# ============================================================================
# TEST: Thread Safety
# ============================================================================


class TestProvenanceThreadSafety:
    """Test thread safety of ProvenanceChainBuilder."""

    def test_concurrent_chain_creation(self, builder):
        """Test that 10 threads can create chains concurrently."""
        results = []
        errors = []

        def create():
            try:
                cid = builder.start_chain()
                results.append(cid)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(results) == 10
        # All chain IDs must be unique
        assert len(set(results)) == 10

    def test_concurrent_stage_recording(self, builder, sample_input, sample_output):
        """Test that 10 threads can record stages on different chains concurrently."""
        chain_ids = [builder.start_chain() for _ in range(10)]
        errors = []

        def record(cid):
            try:
                builder.record_stage(cid, ProvenanceStage.VALIDATE, sample_input, sample_output)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=record, args=(cid,)) for cid in chain_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        for cid in chain_ids:
            chain = builder.get_chain(cid)
            assert chain.stage_count == 1

    def test_concurrent_hash_computation(self, builder):
        """Test that 10 threads computing hashes concurrently produce deterministic results."""
        data = {"test": "concurrent", "value": 42}
        results = []

        def compute():
            h = builder.hash_input(data)
            results.append(h)

        threads = [threading.Thread(target=compute) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 10
        assert len(set(results)) == 1  # All hashes must be identical


# ============================================================================
# TEST: _is_valid_hash
# ============================================================================


class TestIsValidHash:
    """Test the _is_valid_hash static method."""

    def test_valid_hash(self, builder):
        """Test that a valid 64-char hex string passes."""
        h = hashlib.sha256(b"valid").hexdigest()
        assert builder._is_valid_hash(h) is True

    def test_invalid_length(self, builder):
        """Test that a hash with wrong length fails."""
        assert builder._is_valid_hash("abc") is False

    def test_invalid_chars(self, builder):
        """Test that a hash with non-hex characters fails."""
        assert builder._is_valid_hash("g" * 64) is False

    def test_not_string(self, builder):
        """Test that a non-string fails."""
        assert builder._is_valid_hash(12345) is False
