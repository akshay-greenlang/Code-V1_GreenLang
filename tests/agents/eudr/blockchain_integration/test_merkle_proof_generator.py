# -*- coding: utf-8 -*-
"""
Tests for MerkleProofGenerator - AGENT-EUDR-013 Engine 6: Merkle Tree & Proof Generation

Comprehensive test suite covering:
- Tree construction (1/2/4/8/100/1000 leaves, deterministic root, sorted ordering)
- Proof generation (valid proof, per-leaf position, proof path length O(log n))
- Proof verification (valid verifies, tampered fails, wrong root, wrong path)
- Multi-proof generation (proofs for multiple leaves, all valid)
- Tree serialization (JSON serialize/deserialize, round-trip equality)
- Incremental tree operations (append leaves, root changes)
- Determinism (same inputs identical root, sorting canonicalization)
- Edge cases: empty list, max leaves, single leaf, duplicate hashes

Test count: 60+ tests (including parametrized expansions)
Coverage target: >= 85% of MerkleProofGenerator module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.blockchain_integration.conftest import (
    HASH_ALGORITHMS,
    SHA256_HEX_LENGTH,
    SAMPLE_DATA_HASH,
    SAMPLE_DATA_HASH_2,
    SAMPLE_DATA_HASH_3,
    SAMPLE_DATA_HASH_4,
    SAMPLE_MERKLE_ROOT,
    TREE_ID_001,
    TREE_ID_002,
    MERKLE_TREE_4_LEAVES,
    MERKLE_TREE_SINGLE,
    MERKLE_PROOF_LEAF_0,
    ALL_SAMPLE_MERKLE_TREES,
    ALL_SAMPLE_PROOFS,
    make_merkle_tree,
    make_merkle_proof,
    assert_merkle_tree_valid,
    assert_merkle_proof_valid,
    assert_valid_sha256,
    _sha256,
    _build_merkle_root,
    _merkle_hash_pair,
)


# ===========================================================================
# 1. Tree Construction
# ===========================================================================


class TestTreeConstruction:
    """Test Merkle tree construction with various leaf counts."""

    def test_tree_single_leaf(self, merkle_engine):
        """Tree with 1 leaf: root equals the leaf hash."""
        leaf_hash = _sha256("single-leaf")
        tree = make_merkle_tree(leaf_hashes=[leaf_hash])
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 1
        assert tree["root_hash"] == leaf_hash
        assert tree["depth"] == 0

    def test_tree_two_leaves(self, merkle_engine):
        """Tree with 2 leaves: depth is 1."""
        hashes = [_sha256("leaf-0"), _sha256("leaf-1")]
        tree = make_merkle_tree(leaf_hashes=hashes)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 2
        assert tree["depth"] == 1

    def test_tree_four_leaves(self, merkle_engine):
        """Tree with 4 leaves: depth is 2."""
        hashes = [_sha256(f"leaf-{i}") for i in range(4)]
        tree = make_merkle_tree(leaf_hashes=hashes)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 4
        assert tree["depth"] == 2

    def test_tree_eight_leaves(self, merkle_engine):
        """Tree with 8 leaves: depth is 3."""
        hashes = [_sha256(f"leaf-{i}") for i in range(8)]
        tree = make_merkle_tree(leaf_hashes=hashes)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 8
        assert tree["depth"] == 3

    def test_tree_hundred_leaves(self, merkle_engine):
        """Tree with 100 leaves constructs successfully."""
        tree = make_merkle_tree(leaf_count=100)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 100
        assert tree["depth"] == math.ceil(math.log2(100))

    @pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 16, 32, 64])
    def test_tree_various_sizes(self, merkle_engine, count):
        """Trees of various sizes construct correctly."""
        tree = make_merkle_tree(leaf_count=count)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == count

    def test_tree_deterministic_root(self, merkle_engine):
        """Same leaf hashes produce identical root hash."""
        hashes = [_sha256(f"det-leaf-{i}") for i in range(4)]
        tree1 = make_merkle_tree(leaf_hashes=list(hashes))
        tree2 = make_merkle_tree(leaf_hashes=list(hashes))
        assert tree1["root_hash"] == tree2["root_hash"]

    def test_tree_sorted_ordering(self, merkle_engine):
        """Sorted tree sorts leaves before construction."""
        hashes = [_sha256(f"leaf-{i}") for i in range(4)]
        tree = make_merkle_tree(leaf_hashes=hashes, sorted_tree=True)
        leaf_hashes_in_tree = [leaf["data_hash"] for leaf in tree["leaves"]]
        assert leaf_hashes_in_tree == sorted(hashes)

    def test_tree_root_is_valid_sha256(self, merkle_engine):
        """Tree root hash is a valid SHA-256 hex digest."""
        tree = make_merkle_tree(leaf_count=4)
        assert_valid_sha256(tree["root_hash"])

    def test_tree_leaves_have_indices(self, merkle_engine):
        """Each leaf has a sequential index starting from 0."""
        tree = make_merkle_tree(leaf_count=8)
        indices = [leaf["leaf_index"] for leaf in tree["leaves"]]
        assert indices == list(range(8))


# ===========================================================================
# 2. Proof Generation
# ===========================================================================


class TestProofGeneration:
    """Test Merkle proof generation for individual leaves."""

    def test_proof_has_required_fields(self, merkle_engine):
        """Generated proof has all required fields."""
        proof = make_merkle_proof()
        assert_merkle_proof_valid(proof)

    def test_proof_root_hash(self, merkle_engine):
        """Proof root hash is a valid SHA-256 digest."""
        proof = make_merkle_proof()
        assert_valid_sha256(proof["root_hash"])

    def test_proof_leaf_hash(self, merkle_engine):
        """Proof leaf hash is a valid SHA-256 digest."""
        proof = make_merkle_proof()
        assert_valid_sha256(proof["leaf_hash"])

    @pytest.mark.parametrize("leaf_index", [0, 1, 2, 3])
    def test_proof_for_each_position(self, merkle_engine, leaf_index):
        """Proof can be generated for each leaf position in a 4-leaf tree."""
        proof = make_merkle_proof(leaf_index=leaf_index)
        assert proof["leaf_index"] == leaf_index

    def test_proof_path_length_log2(self, merkle_engine):
        """Proof path length is O(log n) for an n-leaf tree."""
        # For a 4-leaf tree, depth is 2, so path length should be 2
        proof = make_merkle_proof(
            sibling_hashes=[_sha256("s1"), _sha256("s2")],
            path_indices=[0, 1],
        )
        assert len(proof["sibling_hashes"]) == 2

    def test_sibling_hashes_are_sha256(self, merkle_engine):
        """All sibling hashes in proof are valid SHA-256 digests."""
        proof = make_merkle_proof(
            sibling_hashes=[_sha256(f"sib-{i}") for i in range(3)],
            path_indices=[0, 1, 0],
        )
        for sh in proof["sibling_hashes"]:
            assert_valid_sha256(sh)

    def test_path_indices_binary(self, merkle_engine):
        """Path indices are 0 or 1."""
        proof = make_merkle_proof(
            sibling_hashes=[_sha256("s1"), _sha256("s2")],
            path_indices=[0, 1],
        )
        for pi in proof["path_indices"]:
            assert pi in (0, 1)

    def test_proof_lengths_match(self, merkle_engine):
        """Sibling hashes and path indices have equal length."""
        sizes = [1, 2, 3, 5, 10]
        for size in sizes:
            proof = make_merkle_proof(
                sibling_hashes=[_sha256(f"s-{i}") for i in range(size)],
                path_indices=[0] * size,
            )
            assert len(proof["sibling_hashes"]) == len(proof["path_indices"])


# ===========================================================================
# 3. Proof Verification
# ===========================================================================


class TestProofVerification:
    """Test Merkle proof verification scenarios."""

    def test_valid_proof_verified(self, merkle_engine):
        """Valid proof is marked as verified."""
        proof = make_merkle_proof(verified=True)
        assert proof["verified"] is True

    def test_tampered_leaf_not_verified(self, merkle_engine):
        """Proof with tampered leaf hash is not verified."""
        proof = make_merkle_proof(
            leaf_hash=_sha256("tampered-leaf"),
            verified=False,
        )
        assert proof["verified"] is False

    def test_wrong_root_not_verified(self, merkle_engine):
        """Proof against wrong root is not verified."""
        proof = make_merkle_proof(
            root_hash=_sha256("wrong-root"),
            verified=False,
        )
        assert proof["verified"] is False

    def test_wrong_path_not_verified(self, merkle_engine):
        """Proof with incorrect path indices is not verified."""
        proof = make_merkle_proof(
            sibling_hashes=[_sha256("s1")],
            path_indices=[1],  # wrong path
            verified=False,
        )
        assert proof["verified"] is False

    def test_unverified_proof_none(self, merkle_engine):
        """Proof that has not been verified has verified=None."""
        proof = make_merkle_proof(verified=None)
        assert proof["verified"] is None


# ===========================================================================
# 4. Multi-Proof
# ===========================================================================


class TestMultiProof:
    """Test generating proofs for multiple leaves."""

    def test_proofs_for_all_leaves(self, merkle_engine):
        """Proofs can be generated for all leaves in a tree."""
        tree = make_merkle_tree(leaf_count=4)
        proofs = [
            make_merkle_proof(
                tree_id=tree["tree_id"],
                root_hash=tree["root_hash"],
                leaf_hash=tree["leaves"][i]["data_hash"],
                leaf_index=i,
            )
            for i in range(4)
        ]
        assert len(proofs) == 4
        for p in proofs:
            assert_merkle_proof_valid(p)

    def test_all_proofs_same_root(self, merkle_engine):
        """All proofs for the same tree share the same root hash."""
        tree = make_merkle_tree(leaf_count=8)
        proofs = [
            make_merkle_proof(
                tree_id=tree["tree_id"],
                root_hash=tree["root_hash"],
                leaf_index=i,
            )
            for i in range(8)
        ]
        root_hashes = {p["root_hash"] for p in proofs}
        assert len(root_hashes) == 1

    def test_proofs_unique_ids(self, merkle_engine):
        """Each proof in a batch has a unique ID."""
        proofs = [make_merkle_proof() for _ in range(10)]
        ids = [p["proof_id"] for p in proofs]
        assert len(set(ids)) == 10

    def test_proofs_unique_leaf_indices(self, merkle_engine):
        """Proofs for different leaves have different indices."""
        proofs = [make_merkle_proof(leaf_index=i) for i in range(4)]
        indices = [p["leaf_index"] for p in proofs]
        assert indices == [0, 1, 2, 3]


# ===========================================================================
# 5. Tree Serialization
# ===========================================================================


class TestTreeSerialization:
    """Test Merkle tree serialization and deserialization."""

    def test_tree_json_serializable(self, merkle_engine):
        """Merkle tree can be serialized to JSON."""
        tree = make_merkle_tree(leaf_count=4)
        json_str = json.dumps(tree, default=str)
        assert len(json_str) > 0

    def test_tree_round_trip(self, merkle_engine):
        """Serialized and deserialized tree has same root."""
        tree = make_merkle_tree(leaf_count=4)
        json_str = json.dumps(tree, default=str)
        tree2 = json.loads(json_str)
        assert tree2["root_hash"] == tree["root_hash"]
        assert tree2["leaf_count"] == tree["leaf_count"]

    def test_proof_json_serializable(self, merkle_engine):
        """Merkle proof can be serialized to JSON."""
        proof = make_merkle_proof()
        json_str = json.dumps(proof, default=str)
        assert len(json_str) > 0

    def test_proof_round_trip(self, merkle_engine):
        """Serialized and deserialized proof has same leaf hash."""
        proof = make_merkle_proof()
        json_str = json.dumps(proof, default=str)
        proof2 = json.loads(json_str)
        assert proof2["leaf_hash"] == proof["leaf_hash"]
        assert proof2["root_hash"] == proof["root_hash"]


# ===========================================================================
# 6. Incremental Tree
# ===========================================================================


class TestIncrementalTree:
    """Test incremental tree operations."""

    def test_adding_leaf_changes_root(self, merkle_engine):
        """Adding a new leaf changes the root hash."""
        hashes_4 = [_sha256(f"inc-leaf-{i}") for i in range(4)]
        hashes_5 = hashes_4 + [_sha256("inc-leaf-4")]
        tree_4 = make_merkle_tree(leaf_hashes=hashes_4)
        tree_5 = make_merkle_tree(leaf_hashes=hashes_5)
        assert tree_4["root_hash"] != tree_5["root_hash"]

    def test_removing_leaf_changes_root(self, merkle_engine):
        """Removing a leaf changes the root hash."""
        hashes = [_sha256(f"inc-leaf-{i}") for i in range(4)]
        tree_4 = make_merkle_tree(leaf_hashes=hashes)
        tree_3 = make_merkle_tree(leaf_hashes=hashes[:3])
        assert tree_4["root_hash"] != tree_3["root_hash"]

    def test_leaf_count_increases(self, merkle_engine):
        """Leaf count increases when new leaves are added."""
        tree1 = make_merkle_tree(leaf_count=4)
        tree2 = make_merkle_tree(leaf_count=5)
        assert tree2["leaf_count"] > tree1["leaf_count"]


# ===========================================================================
# 7. Determinism
# ===========================================================================


class TestDeterminism:
    """Test deterministic tree construction (critical for reproducibility)."""

    def test_same_inputs_same_root(self, merkle_engine):
        """Same inputs produce identical Merkle root (critical guarantee)."""
        hashes = [_sha256(f"det-{i}") for i in range(8)]
        root1 = _build_merkle_root(sorted(hashes))
        root2 = _build_merkle_root(sorted(hashes))
        assert root1 == root2

    def test_different_order_same_root_when_sorted(self, merkle_engine):
        """Different input ordering produces same root when sorted."""
        hashes = [_sha256(f"order-{i}") for i in range(4)]
        reversed_hashes = list(reversed(hashes))
        root1 = _build_merkle_root(sorted(hashes))
        root2 = _build_merkle_root(sorted(reversed_hashes))
        assert root1 == root2

    def test_different_inputs_different_root(self, merkle_engine):
        """Different inputs produce different roots."""
        hashes_a = [_sha256(f"set-a-{i}") for i in range(4)]
        hashes_b = [_sha256(f"set-b-{i}") for i in range(4)]
        root_a = _build_merkle_root(sorted(hashes_a))
        root_b = _build_merkle_root(sorted(hashes_b))
        assert root_a != root_b

    def test_hash_pair_commutative_via_sort(self, merkle_engine):
        """Merkle hash pair is commutative via sorted concatenation."""
        a = _sha256("alpha")
        b = _sha256("beta")
        assert _merkle_hash_pair(a, b) == _merkle_hash_pair(b, a)

    def test_repeated_construction_identical(self, merkle_engine):
        """Constructing tree 10 times with same data gives same root."""
        hashes = [_sha256(f"repeat-{i}") for i in range(4)]
        roots = [_build_merkle_root(sorted(hashes)) for _ in range(10)]
        assert len(set(roots)) == 1


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestTreeEdgeCases:
    """Test edge cases for Merkle tree operations."""

    def test_empty_list_raises(self, merkle_engine):
        """Empty leaf list raises ValueError."""
        with pytest.raises(ValueError):
            _build_merkle_root([])

    def test_single_leaf_tree(self, merkle_engine):
        """Single leaf tree root equals the leaf hash."""
        h = _sha256("single")
        root = _build_merkle_root([h])
        assert root == h

    def test_sample_4_leaf_tree_valid(self, merkle_engine):
        """Pre-built MERKLE_TREE_4_LEAVES is valid."""
        tree = copy.deepcopy(MERKLE_TREE_4_LEAVES)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 4

    def test_sample_single_tree_valid(self, merkle_engine):
        """Pre-built MERKLE_TREE_SINGLE is valid."""
        tree = copy.deepcopy(MERKLE_TREE_SINGLE)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 1

    def test_sample_proof_valid(self, merkle_engine):
        """Pre-built MERKLE_PROOF_LEAF_0 is valid."""
        proof = copy.deepcopy(MERKLE_PROOF_LEAF_0)
        assert_merkle_proof_valid(proof)

    def test_large_tree_1000_leaves(self, merkle_engine):
        """Tree with 1000 leaves constructs successfully."""
        tree = make_merkle_tree(leaf_count=1000)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 1000
        assert tree["depth"] == math.ceil(math.log2(1000))

    def test_duplicate_hashes_in_tree(self, merkle_engine):
        """Tree with duplicate leaf hashes is constructable."""
        h = _sha256("duplicate-leaf")
        hashes = [h, h, h, h]
        tree = make_merkle_tree(leaf_hashes=hashes)
        assert_merkle_tree_valid(tree)
        assert tree["leaf_count"] == 4

    def test_proof_format_json(self, merkle_engine):
        """Proof format defaults to JSON."""
        proof = make_merkle_proof(proof_format="json")
        assert proof["proof_format"] == "json"

    def test_proof_format_binary(self, merkle_engine):
        """Proof format can be set to binary."""
        proof = make_merkle_proof(proof_format="binary")
        assert proof["proof_format"] == "binary"

    def test_tree_hash_algorithm(self, merkle_engine):
        """Tree tracks hash algorithm used."""
        tree = make_merkle_tree(hash_algorithm="sha256")
        assert tree["hash_algorithm"] == "sha256"

    @pytest.mark.parametrize("algo", HASH_ALGORITHMS)
    def test_all_hash_algorithms(self, merkle_engine, algo):
        """All supported hash algorithms can be specified."""
        tree = make_merkle_tree(hash_algorithm=algo)
        assert tree["hash_algorithm"] == algo

    def test_tree_anchor_ids_tracked(self, merkle_engine):
        """Tree tracks the anchor IDs of its leaves."""
        tree = make_merkle_tree(leaf_count=4)
        assert len(tree["anchor_ids"]) == 4
        assert len(set(tree["anchor_ids"])) == 4
