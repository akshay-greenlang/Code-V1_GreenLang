# -*- coding: utf-8 -*-
"""
Merkle Tree and Proof Generation - AGENT-EUDR-013 Engine 6

Deterministic Merkle tree construction and cryptographic proof generation
for tamper-evident anchoring of EUDR compliance data on-chain. Provides
O(log n) inclusion proofs, sorted tree construction for deterministic root
hashes, incremental tree updates, and multiple serialization formats.

Zero-Hallucination Guarantees:
    - All Merkle tree construction uses deterministic SHA-256 hashing
    - No ML/LLM used for any tree operation or proof computation
    - Sorted tree leaves guarantee identical root for identical inputs
    - Hash pair ordering is deterministic (lexicographic sort before concat)
    - Proof verification is standalone (only needs proof + hash + root)
    - All numeric operations use integer arithmetic only
    - Bit-perfect reproducibility: same inputs always produce same root
    - SHA-256 provenance hashes on every tree and proof operation

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligation data integrity
    - EU 2023/1115 (EUDR) Article 14: Five-year immutable record retention
    - EU 2023/1115 (EUDR) Article 10(2): Risk assessment data provenance
    - ISO 22095:2020: Chain of Custody - tamper-evident traceability
    - NIST SP 800-185: SHA-3 Derived Functions (Merkle tree hashing)

Performance Targets:
    - Build tree (100 leaves): <10ms
    - Build tree (10,000 leaves): <500ms
    - Generate proof: <1ms
    - Verify proof: <1ms
    - Serialize/deserialize tree (1000 leaves): <50ms

Tree Properties:
    - Leaf count: 1 to 10,000 (configurable via max_tree_leaves)
    - Proof path length: O(log2 n) where n = leaf count
    - Hash algorithm: SHA-256 (default), SHA-512, Keccak-256
    - Tree ordering: Sorted (deterministic) or insertion-order
    - Odd leaf handling: Duplicate last leaf to maintain binary structure

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
Agent ID: GL-EUDR-BCI-013
Engine: 6 of 8 (Merkle Tree and Proof Generation)
Status: Production Ready
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.blockchain_integration.config import (
    BlockchainIntegrationConfig,
    get_config,
)
from greenlang.agents.eudr.blockchain_integration.models import (
    MerkleLeaf,
    MerkleProof,
    MerkleTree,
    ProofFormat,
)
from greenlang.agents.eudr.blockchain_integration.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.blockchain_integration.metrics import (
    observe_merkle_build_duration,
    record_api_error,
    record_merkle_proof_generated,
    record_merkle_tree_built,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "MKL") -> str:
    """Generate a prefixed UUID4 string identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Prefixed UUID4 string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Hash algorithm registry
# ---------------------------------------------------------------------------

_HASH_ALGORITHMS: Dict[str, Any] = {
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}

# Keccak-256 support (requires pycryptodome or pysha3)
try:
    from Crypto.Hash import keccak as _keccak_mod  # type: ignore[import]

    def _keccak256_factory(data: bytes = b"") -> Any:
        """Create a Keccak-256 hash object with hashlib-compatible interface."""
        h = _keccak_mod.new(digest_bits=256)
        if data:
            h.update(data)
        return h

    _HASH_ALGORITHMS["keccak256"] = _keccak256_factory
except ImportError:
    try:
        import sha3 as _sha3_mod  # type: ignore[import]
        _HASH_ALGORITHMS["keccak256"] = _sha3_mod.keccak_256
    except ImportError:
        logger.debug("keccak256 not available; sha256 and sha512 only")


# ---------------------------------------------------------------------------
# Proof step data model
# ---------------------------------------------------------------------------


class ProofStep:
    """A single step in a Merkle proof authentication path.

    Attributes:
        sibling_hash: Hash of the sibling node at this tree level.
        direction: 0 if sibling is on the left, 1 if on the right.
        level: Tree level (0 = leaf level, depth = root level).
    """

    __slots__ = ("sibling_hash", "direction", "level")

    def __init__(
        self,
        sibling_hash: str,
        direction: int,
        level: int,
    ) -> None:
        """Initialize a ProofStep.

        Args:
            sibling_hash: Hash of the sibling node.
            direction: 0 for left sibling, 1 for right sibling.
            level: Tree level of this step.
        """
        self.sibling_hash = sibling_hash
        self.direction = direction
        self.level = level

    def to_dict(self) -> Dict[str, Any]:
        """Serialize proof step to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "sibling_hash": self.sibling_hash,
            "direction": self.direction,
            "level": self.level,
        }


# ---------------------------------------------------------------------------
# Internal tree representation
# ---------------------------------------------------------------------------


class InternalTree:
    """Internal representation of a complete Merkle tree with all levels.

    Stores all tree levels from leaves to root for efficient proof
    generation and tree inspection.

    Attributes:
        tree_id: Unique tree identifier.
        levels: List of hash lists, levels[0] = leaves, levels[-1] = [root].
        leaf_hashes: Original sorted leaf hashes.
        root_hash: Merkle root hash.
        depth: Tree depth.
        hash_algorithm: Hash algorithm used.
        is_sorted: Whether leaves were sorted before construction.
        anchor_ids: Anchor IDs associated with leaves.
        leaf_to_index: Mapping from leaf hash to leaf index.
        created_at: UTC creation timestamp.
    """

    __slots__ = (
        "tree_id",
        "levels",
        "leaf_hashes",
        "root_hash",
        "depth",
        "hash_algorithm",
        "is_sorted",
        "anchor_ids",
        "leaf_to_index",
        "created_at",
    )

    def __init__(
        self,
        tree_id: str,
        levels: List[List[str]],
        leaf_hashes: List[str],
        root_hash: str,
        depth: int,
        hash_algorithm: str,
        is_sorted: bool,
        anchor_ids: List[str],
    ) -> None:
        """Initialize an InternalTree.

        Args:
            tree_id: Unique tree identifier.
            levels: Complete tree levels.
            leaf_hashes: Sorted leaf hashes.
            root_hash: Computed root hash.
            depth: Tree depth.
            hash_algorithm: Hash algorithm used.
            is_sorted: Whether leaves were sorted.
            anchor_ids: Associated anchor IDs.
        """
        self.tree_id = tree_id
        self.levels = levels
        self.leaf_hashes = leaf_hashes
        self.root_hash = root_hash
        self.depth = depth
        self.hash_algorithm = hash_algorithm
        self.is_sorted = is_sorted
        self.anchor_ids = anchor_ids
        self.created_at = _utcnow()

        # Build reverse lookup: leaf_hash -> index
        self.leaf_to_index: Dict[str, int] = {
            h: i for i, h in enumerate(leaf_hashes)
        }


# ==========================================================================
# MerkleProofGenerator
# ==========================================================================


class MerkleProofGenerator:
    """Deterministic Merkle tree construction and cryptographic proof generation engine.

    Builds balanced binary Merkle trees from sets of record hashes,
    generates O(log n) inclusion proofs for individual leaves, and
    provides standalone proof verification. Supports sorted tree
    construction for deterministic root hashes, incremental tree
    updates, and multiple serialization formats (JSON, binary).

    CRITICAL DETERMINISM GUARANTEE: Given the same set of input hashes,
    this engine will always produce the identical root hash regardless
    of insertion order (when sorted_tree=True). This is achieved by
    lexicographic sorting of leaf hashes before tree construction and
    by lexicographic ordering of hash pairs at each internal node.

    Zero-Hallucination: All tree operations use deterministic
    cryptographic hashing. No ML/LLM involved in any computation.
    All arithmetic is integer-only. SHA-256 provenance hashes are
    recorded for every tree and proof operation.

    Thread Safety: All mutable state is protected by a reentrant lock.

    Attributes:
        _config: Blockchain integration configuration.
        _provenance: Provenance tracker for SHA-256 audit trails.
        _trees: Storage of built internal trees.
        _tree_models: Storage of Pydantic MerkleTree models.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> from greenlang.agents.eudr.blockchain_integration.merkle_proof_generator import (
        ...     MerkleProofGenerator,
        ... )
        >>> gen = MerkleProofGenerator()
        >>> tree = gen.build_tree(["aabb" * 16, "ccdd" * 16, "eeff" * 16])
        >>> proof = gen.generate_proof(tree.tree_id, "aabb" * 16)
        >>> valid = gen.verify_proof("aabb" * 16, proof.sibling_hashes,
        ...                          proof.path_indices, tree.root_hash)
        >>> assert valid is True
    """

    def __init__(
        self,
        config: Optional[BlockchainIntegrationConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the MerkleProofGenerator engine.

        Args:
            config: Optional configuration override. Uses get_config()
                singleton when None.
            provenance: Optional provenance tracker override. Uses
                get_provenance_tracker() singleton when None.
        """
        self._config = config or get_config()
        self._provenance = provenance or get_provenance_tracker()
        self._lock = threading.RLock()

        # Internal tree storage: tree_id -> InternalTree
        self._trees: Dict[str, InternalTree] = {}

        # Pydantic model storage: tree_id -> MerkleTree
        self._tree_models: Dict[str, MerkleTree] = {}

        # Statistics
        self._total_trees_built: int = 0
        self._total_proofs_generated: int = 0
        self._total_verifications: int = 0

        logger.info(
            "MerkleProofGenerator initialized: max_leaves=%d, "
            "sorted=%s, algorithm=%s",
            self._config.max_tree_leaves,
            self._config.sorted_tree,
            self._config.hash_algorithm,
        )

    # ------------------------------------------------------------------
    # Tree Construction
    # ------------------------------------------------------------------

    def build_tree(
        self,
        record_hashes: List[str],
        anchor_ids: Optional[List[str]] = None,
        hash_algorithm: Optional[str] = None,
        sorted_tree: Optional[bool] = None,
        tree_id: Optional[str] = None,
    ) -> MerkleTree:
        """Build a Merkle tree from a list of record hashes.

        Constructs a balanced binary Merkle tree by:
        1. Optionally sorting leaf hashes (lexicographic, deterministic)
        2. Hashing each leaf with domain separation prefix
        3. Building internal nodes by hashing pairs bottom-up
        4. Duplicating the last node at odd-count levels

        Args:
            record_hashes: List of hex-encoded hash strings (1 to
                max_tree_leaves entries). Each hash represents one
                EUDR compliance record.
            anchor_ids: Optional list of anchor IDs corresponding to
                each record hash. Must be same length as record_hashes
                when provided.
            hash_algorithm: Hash algorithm override (sha256, sha512,
                keccak256). Defaults to configured algorithm.
            sorted_tree: Whether to sort leaves. Defaults to configured
                value.
            tree_id: Optional tree ID override. Auto-generated when None.

        Returns:
            MerkleTree Pydantic model with root hash, leaves, and depth.

        Raises:
            ValueError: If record_hashes is empty or exceeds max_tree_leaves.
            ValueError: If anchor_ids length doesn't match record_hashes.
            ValueError: If hash_algorithm is not supported.
        """
        start_time = time.monotonic()

        # Validate inputs
        if not record_hashes:
            raise ValueError("record_hashes must not be empty")

        if len(record_hashes) > self._config.max_tree_leaves:
            raise ValueError(
                f"record_hashes count ({len(record_hashes)}) exceeds "
                f"max_tree_leaves ({self._config.max_tree_leaves})"
            )

        if anchor_ids is not None and len(anchor_ids) != len(record_hashes):
            raise ValueError(
                f"anchor_ids length ({len(anchor_ids)}) must match "
                f"record_hashes length ({len(record_hashes)})"
            )

        effective_algorithm = hash_algorithm or self._config.hash_algorithm
        if effective_algorithm not in _HASH_ALGORITHMS:
            raise ValueError(
                f"Unsupported hash algorithm: '{effective_algorithm}'. "
                f"Supported: {sorted(_HASH_ALGORITHMS.keys())}"
            )

        effective_sorted = (
            sorted_tree if sorted_tree is not None else self._config.sorted_tree
        )
        effective_tree_id = tree_id or str(uuid.uuid4())
        effective_anchor_ids = anchor_ids or [
            f"anon-{i}" for i in range(len(record_hashes))
        ]

        try:
            # Step 1: Sort leaves for deterministic root (if enabled)
            if effective_sorted:
                sorted_pairs = self._sort_leaves(
                    record_hashes, effective_anchor_ids
                )
                sorted_hashes = [p[0] for p in sorted_pairs]
                sorted_anchor_ids = [p[1] for p in sorted_pairs]
            else:
                sorted_hashes = list(record_hashes)
                sorted_anchor_ids = list(effective_anchor_ids)

            # Step 2: Compute leaf hashes with domain separation
            leaf_hashes = self._compute_leaf_hashes(
                sorted_hashes, effective_algorithm
            )

            # Step 3: Build internal nodes bottom-up
            levels = self._build_internal_nodes(
                leaf_hashes, effective_algorithm
            )

            # Root is the single element at the top level
            root_hash = levels[-1][0]
            depth = len(levels) - 1

            # Step 4: Create InternalTree
            internal_tree = InternalTree(
                tree_id=effective_tree_id,
                levels=levels,
                leaf_hashes=leaf_hashes,
                root_hash=root_hash,
                depth=depth,
                hash_algorithm=effective_algorithm,
                is_sorted=effective_sorted,
                anchor_ids=sorted_anchor_ids,
            )

            # Step 5: Create MerkleLeaf models
            leaves = []
            for i, (data_hash, leaf_hash) in enumerate(
                zip(sorted_hashes, leaf_hashes)
            ):
                leaf = MerkleLeaf(
                    leaf_index=i,
                    data_hash=data_hash,
                    anchor_id=sorted_anchor_ids[i],
                    leaf_hash=leaf_hash,
                )
                leaves.append(leaf)

            # Step 6: Create MerkleTree model
            tree_model = MerkleTree(
                tree_id=effective_tree_id,
                root_hash=root_hash,
                leaf_count=len(leaf_hashes),
                leaves=leaves,
                depth=depth,
                hash_algorithm=effective_algorithm,
                sorted=effective_sorted,
                anchor_ids=sorted_anchor_ids,
                created_at=_utcnow(),
            )

            # Step 7: Store tree
            with self._lock:
                self._trees[effective_tree_id] = internal_tree
                self._tree_models[effective_tree_id] = tree_model
                self._total_trees_built += 1

            # Step 8: Record provenance
            provenance_entry = self._provenance.record(
                entity_type="merkle_tree",
                action="create",
                entity_id=effective_tree_id,
                data={
                    "root_hash": root_hash,
                    "leaf_count": len(leaf_hashes),
                    "depth": depth,
                    "hash_algorithm": effective_algorithm,
                    "sorted": effective_sorted,
                },
                metadata={
                    "module_version": _MODULE_VERSION,
                    "operation": "build_tree",
                },
            )
            tree_model.provenance_hash = provenance_entry.hash_value

            # Step 9: Record metrics
            elapsed_s = time.monotonic() - start_time
            record_merkle_tree_built(effective_algorithm)
            observe_merkle_build_duration(elapsed_s)

            elapsed_ms = elapsed_s * 1000
            logger.info(
                "Merkle tree built: id=%s root=%s leaves=%d "
                "depth=%d algo=%s sorted=%s elapsed=%.1fms",
                effective_tree_id[:16],
                root_hash[:16],
                len(leaf_hashes),
                depth,
                effective_algorithm,
                effective_sorted,
                elapsed_ms,
            )
            return tree_model

        except Exception as exc:
            record_api_error("build_tree")
            logger.error(
                "Failed to build Merkle tree: %s", str(exc), exc_info=True
            )
            raise

    def get_tree(self, tree_id: str) -> Optional[MerkleTree]:
        """Retrieve a previously built Merkle tree by its identifier.

        Args:
            tree_id: Merkle tree identifier.

        Returns:
            MerkleTree model if found, None otherwise.

        Raises:
            ValueError: If tree_id is empty.
        """
        if not tree_id:
            raise ValueError("tree_id must not be empty")

        with self._lock:
            return self._tree_models.get(tree_id)

    # ------------------------------------------------------------------
    # Proof Generation
    # ------------------------------------------------------------------

    def generate_proof(
        self,
        tree_id: str,
        record_hash: str,
        proof_format: Optional[str] = None,
    ) -> MerkleProof:
        """Generate a Merkle inclusion proof for a specific record hash.

        Computes the authentication path from the leaf to the root,
        providing the sibling hashes and path direction indices needed
        to independently verify inclusion.

        Args:
            tree_id: Merkle tree identifier.
            record_hash: Hex-encoded hash of the record to prove.
            proof_format: Output format (json, binary). Defaults to
                configured format.

        Returns:
            MerkleProof with sibling hashes and path indices.

        Raises:
            ValueError: If tree_id or record_hash is empty.
            KeyError: If tree_id is not found.
            KeyError: If record_hash is not a leaf in the tree.
        """
        start_time = time.monotonic()

        if not tree_id:
            raise ValueError("tree_id must not be empty")
        if not record_hash:
            raise ValueError("record_hash must not be empty")

        with self._lock:
            internal = self._trees.get(tree_id)

        if internal is None:
            raise KeyError(f"Merkle tree not found: {tree_id}")

        # Compute the leaf hash for this record hash
        leaf_hash = self._hash_leaf(record_hash, internal.hash_algorithm)

        if leaf_hash not in internal.leaf_to_index:
            raise KeyError(
                f"Record hash not found in tree: {record_hash[:32]}..."
            )

        leaf_index = internal.leaf_to_index[leaf_hash]

        # Generate proof path
        proof_steps = self._get_proof_path(
            internal.levels, leaf_index
        )

        effective_format = proof_format or self._config.proof_format

        # Find the anchor_id for this leaf
        anchor_id = (
            internal.anchor_ids[leaf_index]
            if leaf_index < len(internal.anchor_ids)
            else None
        )

        # Create MerkleProof model
        proof = MerkleProof(
            tree_id=tree_id,
            root_hash=internal.root_hash,
            leaf_hash=leaf_hash,
            leaf_index=leaf_index,
            sibling_hashes=[step.sibling_hash for step in proof_steps],
            path_indices=[step.direction for step in proof_steps],
            hash_algorithm=internal.hash_algorithm,
            proof_format=effective_format,
            anchor_id=anchor_id,
            verified=None,
            created_at=_utcnow(),
        )

        # Verify the proof locally
        is_valid = self.verify_proof(
            record_hash,
            proof.sibling_hashes,
            proof.path_indices,
            internal.root_hash,
            internal.hash_algorithm,
        )
        proof.verified = is_valid

        with self._lock:
            self._total_proofs_generated += 1

        # Record provenance
        self._provenance.record(
            entity_type="merkle_tree",
            action="verify",
            entity_id=proof.proof_id,
            data={
                "tree_id": tree_id,
                "leaf_hash": leaf_hash[:32],
                "leaf_index": leaf_index,
                "path_length": len(proof_steps),
                "verified": is_valid,
            },
            metadata={
                "module_version": _MODULE_VERSION,
                "operation": "generate_proof",
            },
        )

        record_merkle_proof_generated()

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Merkle proof generated: tree=%s leaf_index=%d "
            "path_length=%d verified=%s elapsed=%.1fms",
            tree_id[:16],
            leaf_index,
            len(proof_steps),
            is_valid,
            elapsed_ms,
        )
        return proof

    def generate_multi_proof(
        self,
        tree_id: str,
        record_hashes: List[str],
        proof_format: Optional[str] = None,
    ) -> List[MerkleProof]:
        """Generate Merkle proofs for multiple record hashes in a tree.

        Convenience method that generates individual proofs for each
        record hash. Returns proofs in the same order as the input
        hashes.

        Args:
            tree_id: Merkle tree identifier.
            record_hashes: List of hex-encoded record hashes.
            proof_format: Output format override.

        Returns:
            List of MerkleProof objects, one per input hash.

        Raises:
            ValueError: If tree_id is empty or record_hashes is empty.
            KeyError: If tree_id is not found.
            KeyError: If any record_hash is not in the tree.
        """
        if not tree_id:
            raise ValueError("tree_id must not be empty")
        if not record_hashes:
            raise ValueError("record_hashes must not be empty")

        proofs: List[MerkleProof] = []
        for record_hash in record_hashes:
            proof = self.generate_proof(tree_id, record_hash, proof_format)
            proofs.append(proof)

        logger.info(
            "Multi-proof generated: tree=%s proofs=%d",
            tree_id[:16],
            len(proofs),
        )
        return proofs

    # ------------------------------------------------------------------
    # Proof Verification
    # ------------------------------------------------------------------

    def verify_proof(
        self,
        record_hash: str,
        sibling_hashes: List[str],
        path_indices: List[int],
        root_hash: str,
        hash_algorithm: Optional[str] = None,
    ) -> bool:
        """Verify a Merkle inclusion proof against a known root hash.

        This method is STANDALONE: it does not require access to the
        original tree. Only the proof data (sibling hashes, path indices),
        the record hash, and the expected root hash are needed.

        Verification process:
        1. Compute the leaf hash from the record hash
        2. Walk up the proof path, hashing with siblings at each level
        3. Compare the computed root with the expected root

        Args:
            record_hash: Hex-encoded hash of the record being verified.
            sibling_hashes: Ordered list of sibling hashes from the proof.
            path_indices: List of 0/1 values indicating sibling position
                (0 = sibling is left, 1 = sibling is right).
            root_hash: Expected Merkle root hash.
            hash_algorithm: Hash algorithm used in tree construction.
                Defaults to configured algorithm.

        Returns:
            True if the proof is valid and the record is included in
            the tree, False otherwise.

        Raises:
            ValueError: If sibling_hashes and path_indices lengths differ.
        """
        start_time = time.monotonic()

        if len(sibling_hashes) != len(path_indices):
            raise ValueError(
                f"sibling_hashes length ({len(sibling_hashes)}) "
                f"must equal path_indices length ({len(path_indices)})"
            )

        effective_algorithm = hash_algorithm or self._config.hash_algorithm

        try:
            # Compute the leaf hash
            current_hash = self._hash_leaf(record_hash, effective_algorithm)

            # Walk up the proof path
            for sibling_hash, direction in zip(sibling_hashes, path_indices):
                if direction == 0:
                    # Sibling is on the left
                    current_hash = self._hash_pair(
                        sibling_hash, current_hash, effective_algorithm
                    )
                else:
                    # Sibling is on the right
                    current_hash = self._hash_pair(
                        current_hash, sibling_hash, effective_algorithm
                    )

            # Compare computed root with expected root
            is_valid = current_hash == root_hash

            with self._lock:
                self._total_verifications += 1

            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.debug(
                "Proof verification: record=%s valid=%s elapsed=%.1fms",
                record_hash[:16],
                is_valid,
                elapsed_ms,
            )
            return is_valid

        except Exception as exc:
            logger.error(
                "Proof verification failed: %s", str(exc), exc_info=True
            )
            return False

    # ------------------------------------------------------------------
    # Tree Serialization
    # ------------------------------------------------------------------

    def serialize_tree(
        self,
        tree_id: str,
        fmt: str = "json",
    ) -> str:
        """Serialize a Merkle tree to a portable format.

        Supports JSON and binary serialization formats. JSON format
        is human-readable and suitable for API responses. Binary format
        is compact and suitable for on-chain storage.

        Args:
            tree_id: Merkle tree identifier.
            fmt: Output format: 'json' or 'binary'.

        Returns:
            Serialized tree as a string (JSON) or base64-encoded string
            (binary).

        Raises:
            ValueError: If tree_id is empty or fmt is unsupported.
            KeyError: If tree_id is not found.
        """
        if not tree_id:
            raise ValueError("tree_id must not be empty")
        if fmt not in ("json", "binary"):
            raise ValueError(
                f"Unsupported format: '{fmt}'. Use 'json' or 'binary'"
            )

        with self._lock:
            internal = self._trees.get(tree_id)
            model = self._tree_models.get(tree_id)

        if internal is None or model is None:
            raise KeyError(f"Merkle tree not found: {tree_id}")

        if fmt == "json":
            return self._serialize_json(internal, model)
        else:
            return self._serialize_binary(internal)

    def deserialize_tree(
        self,
        data: str,
        fmt: str = "json",
    ) -> MerkleTree:
        """Deserialize a Merkle tree from a portable format.

        Reconstructs a MerkleTree model from serialized data. The
        deserialized tree is stored internally for subsequent proof
        generation.

        Args:
            data: Serialized tree data (JSON string or base64 binary).
            fmt: Input format: 'json' or 'binary'.

        Returns:
            Deserialized MerkleTree model.

        Raises:
            ValueError: If data is empty or fmt is unsupported.
            ValueError: If deserialized tree fails validation.
        """
        if not data:
            raise ValueError("data must not be empty")
        if fmt not in ("json", "binary"):
            raise ValueError(
                f"Unsupported format: '{fmt}'. Use 'json' or 'binary'"
            )

        if fmt == "json":
            return self._deserialize_json(data)
        else:
            return self._deserialize_binary(data)

    # ------------------------------------------------------------------
    # Incremental Tree Update
    # ------------------------------------------------------------------

    def append_to_tree(
        self,
        tree_id: str,
        new_hashes: List[str],
        new_anchor_ids: Optional[List[str]] = None,
    ) -> MerkleTree:
        """Append new record hashes to an existing tree by rebuilding.

        Creates a new tree containing all original leaves plus the new
        hashes. The original tree is retained for historical reference.
        The new tree receives a new tree_id.

        Note: This is a full rebuild, not an incremental update. For
        trees with sorted leaves, the new tree will have a different
        root hash than one built from scratch with all leaves, because
        new leaves are inserted at their sorted position.

        Args:
            tree_id: Existing tree identifier.
            new_hashes: Additional record hashes to add.
            new_anchor_ids: Optional anchor IDs for new hashes.

        Returns:
            New MerkleTree model with all leaves (old + new).

        Raises:
            ValueError: If tree_id is empty or new_hashes is empty.
            KeyError: If tree_id is not found.
            ValueError: If total leaves would exceed max_tree_leaves.
        """
        if not tree_id:
            raise ValueError("tree_id must not be empty")
        if not new_hashes:
            raise ValueError("new_hashes must not be empty")

        with self._lock:
            internal = self._trees.get(tree_id)

        if internal is None:
            raise KeyError(f"Merkle tree not found: {tree_id}")

        # Collect original data hashes from leaves
        original_model = self._tree_models.get(tree_id)
        if original_model is None:
            raise KeyError(f"Merkle tree model not found: {tree_id}")

        original_data_hashes = [leaf.data_hash for leaf in original_model.leaves]
        original_anchor_ids = list(internal.anchor_ids)

        # Combine old and new
        combined_hashes = original_data_hashes + list(new_hashes)
        combined_anchor_ids = original_anchor_ids + (
            new_anchor_ids or [f"anon-{i}" for i in range(len(new_hashes))]
        )

        total_leaves = len(combined_hashes)
        if total_leaves > self._config.max_tree_leaves:
            raise ValueError(
                f"Combined leaf count ({total_leaves}) exceeds "
                f"max_tree_leaves ({self._config.max_tree_leaves})"
            )

        # Build new tree with combined data
        new_tree = self.build_tree(
            record_hashes=combined_hashes,
            anchor_ids=combined_anchor_ids,
            hash_algorithm=internal.hash_algorithm,
            sorted_tree=internal.is_sorted,
        )

        logger.info(
            "Tree appended: original=%s new_tree=%s "
            "original_leaves=%d added=%d total=%d",
            tree_id[:16],
            new_tree.tree_id[:16],
            len(original_data_hashes),
            len(new_hashes),
            total_leaves,
        )
        return new_tree

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return Merkle proof generator statistics.

        Returns:
            Dictionary of operational statistics.
        """
        with self._lock:
            return {
                "total_trees_built": self._total_trees_built,
                "total_proofs_generated": self._total_proofs_generated,
                "total_verifications": self._total_verifications,
                "trees_in_memory": len(self._trees),
                "max_tree_leaves": self._config.max_tree_leaves,
                "default_hash_algorithm": self._config.hash_algorithm,
                "default_sorted": self._config.sorted_tree,
                "supported_algorithms": sorted(_HASH_ALGORITHMS.keys()),
                "module_version": _MODULE_VERSION,
            }

    def get_tree_count(self) -> int:
        """Return the number of trees currently in memory.

        Returns:
            Tree count.
        """
        with self._lock:
            return len(self._trees)

    # ------------------------------------------------------------------
    # Reset / Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all tree state. Intended for testing teardown."""
        with self._lock:
            self._trees.clear()
            self._tree_models.clear()
            self._total_trees_built = 0
            self._total_proofs_generated = 0
            self._total_verifications = 0
        logger.info("MerkleProofGenerator state cleared")

    def remove_tree(self, tree_id: str) -> bool:
        """Remove a tree from memory.

        Args:
            tree_id: Tree identifier to remove.

        Returns:
            True if the tree was found and removed, False otherwise.
        """
        with self._lock:
            removed_internal = self._trees.pop(tree_id, None)
            removed_model = self._tree_models.pop(tree_id, None)
        return removed_internal is not None or removed_model is not None

    # ------------------------------------------------------------------
    # Internal: Hash Operations
    # ------------------------------------------------------------------

    def _hash_leaf(self, data_hash: str, algorithm: str) -> str:
        """Compute a leaf hash with domain separation prefix.

        Uses a 0x00 byte prefix to distinguish leaf hashes from
        internal node hashes, preventing second-preimage attacks.

        Args:
            data_hash: Hex-encoded data hash.
            algorithm: Hash algorithm name.

        Returns:
            Hex-encoded leaf hash.
        """
        hasher = _HASH_ALGORITHMS[algorithm]
        h = hasher(b"\x00" + bytes.fromhex(data_hash))
        return h.hexdigest()

    def _hash_pair(
        self,
        left: str,
        right: str,
        algorithm: str,
    ) -> str:
        """Hash a pair of node hashes to produce a parent hash.

        Uses a 0x01 byte prefix for internal nodes (domain separation).
        The pair is sorted lexicographically before concatenation to
        ensure deterministic hashing regardless of input order.

        Args:
            left: Hex-encoded left child hash.
            right: Hex-encoded right child hash.
            algorithm: Hash algorithm name.

        Returns:
            Hex-encoded parent hash.
        """
        hasher = _HASH_ALGORITHMS[algorithm]
        # Lexicographic sort for deterministic pairing
        if left > right:
            left, right = right, left
        h = hasher(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Internal: Tree Construction
    # ------------------------------------------------------------------

    def _sort_leaves(
        self,
        record_hashes: List[str],
        anchor_ids: List[str],
    ) -> List[Tuple[str, str]]:
        """Sort record hashes lexicographically with associated anchor IDs.

        Args:
            record_hashes: Unsorted record hashes.
            anchor_ids: Anchor IDs parallel to record_hashes.

        Returns:
            Sorted list of (record_hash, anchor_id) tuples.
        """
        pairs = list(zip(record_hashes, anchor_ids))
        pairs.sort(key=lambda p: p[0])
        return pairs

    def _compute_leaf_hashes(
        self,
        sorted_hashes: List[str],
        algorithm: str,
    ) -> List[str]:
        """Compute leaf hashes with domain separation for all data hashes.

        Args:
            sorted_hashes: Sorted data hashes.
            algorithm: Hash algorithm name.

        Returns:
            List of computed leaf hashes.
        """
        return [self._hash_leaf(h, algorithm) for h in sorted_hashes]

    def _build_internal_nodes(
        self,
        leaf_hashes: List[str],
        algorithm: str,
    ) -> List[List[str]]:
        """Build all internal tree levels from leaves to root.

        At each level, consecutive pairs of hashes are combined using
        _hash_pair(). If the level has an odd number of nodes, the last
        node is duplicated to maintain binary structure.

        Args:
            leaf_hashes: Computed leaf hashes (level 0).
            algorithm: Hash algorithm name.

        Returns:
            List of hash lists, where levels[0] = leaf_hashes and
            levels[-1] = [root_hash].
        """
        levels: List[List[str]] = [list(leaf_hashes)]

        current_level = list(leaf_hashes)
        while len(current_level) > 1:
            next_level: List[str] = []

            # Duplicate last node if odd count
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                parent = self._hash_pair(
                    current_level[i], current_level[i + 1], algorithm
                )
                next_level.append(parent)

            levels.append(next_level)
            current_level = next_level

        return levels

    def _get_proof_path(
        self,
        levels: List[List[str]],
        leaf_index: int,
    ) -> List[ProofStep]:
        """Extract the authentication path for a leaf at a given index.

        Walks from the leaf level to the root level, collecting the
        sibling hash at each level along with the direction indicator.

        Args:
            levels: Complete tree levels from _build_internal_nodes().
            leaf_index: Index of the target leaf in levels[0].

        Returns:
            List of ProofStep objects from leaf to root.
        """
        proof_steps: List[ProofStep] = []
        current_index = leaf_index

        for level_idx in range(len(levels) - 1):
            level = levels[level_idx]

            # Determine sibling index
            if current_index % 2 == 0:
                # Current is left child, sibling is right
                sibling_index = current_index + 1
                direction = 1  # sibling is on the right
            else:
                # Current is right child, sibling is left
                sibling_index = current_index - 1
                direction = 0  # sibling is on the left

            # Handle edge case: odd-length level with last node duplicated
            if sibling_index >= len(level):
                sibling_index = current_index  # Self-sibling (duplicated)

            sibling_hash = level[sibling_index]
            proof_steps.append(
                ProofStep(
                    sibling_hash=sibling_hash,
                    direction=direction,
                    level=level_idx,
                )
            )

            # Move to parent index
            current_index = current_index // 2

        return proof_steps

    # ------------------------------------------------------------------
    # Internal: Serialization
    # ------------------------------------------------------------------

    def _serialize_json(
        self,
        internal: InternalTree,
        model: MerkleTree,
    ) -> str:
        """Serialize a tree to JSON format.

        Args:
            internal: Internal tree representation.
            model: Pydantic MerkleTree model.

        Returns:
            JSON string.
        """
        data = {
            "tree_id": internal.tree_id,
            "root_hash": internal.root_hash,
            "leaf_count": len(internal.leaf_hashes),
            "depth": internal.depth,
            "hash_algorithm": internal.hash_algorithm,
            "sorted": internal.is_sorted,
            "anchor_ids": internal.anchor_ids,
            "leaves": [
                {
                    "leaf_index": leaf.leaf_index,
                    "data_hash": leaf.data_hash,
                    "anchor_id": leaf.anchor_id,
                    "leaf_hash": leaf.leaf_hash,
                }
                for leaf in model.leaves
            ],
            "levels": internal.levels,
            "created_at": internal.created_at.isoformat(),
            "module_version": _MODULE_VERSION,
        }
        return json.dumps(data, indent=2, default=str)

    def _serialize_binary(self, internal: InternalTree) -> str:
        """Serialize a tree to compact binary format (base64-encoded).

        Binary layout:
        - Header: version (1B) + algorithm_id (1B) + sorted (1B) +
          leaf_count (4B big-endian) + depth (2B big-endian)
        - Root hash (32B for SHA-256, 64B for SHA-512)
        - Leaf hashes (leaf_count * hash_size)

        Args:
            internal: Internal tree representation.

        Returns:
            Base64-encoded binary string.
        """
        algo_ids = {"sha256": 0, "sha512": 1, "keccak256": 2}
        hash_sizes = {"sha256": 32, "sha512": 64, "keccak256": 32}

        algo_id = algo_ids.get(internal.hash_algorithm, 0)
        hash_size = hash_sizes.get(internal.hash_algorithm, 32)

        parts: List[bytes] = []

        # Header
        header = struct.pack(
            ">BBBIh",
            1,  # version
            algo_id,
            1 if internal.is_sorted else 0,
            len(internal.leaf_hashes),
            internal.depth,
        )
        parts.append(header)

        # Root hash
        parts.append(bytes.fromhex(internal.root_hash)[:hash_size])

        # Leaf hashes
        for leaf_hash in internal.leaf_hashes:
            parts.append(bytes.fromhex(leaf_hash)[:hash_size])

        binary_data = b"".join(parts)
        return base64.b64encode(binary_data).decode("ascii")

    def _deserialize_json(self, data: str) -> MerkleTree:
        """Deserialize a tree from JSON format.

        Args:
            data: JSON string.

        Returns:
            MerkleTree model.
        """
        parsed = json.loads(data)

        tree_id = parsed["tree_id"]
        algorithm = parsed.get("hash_algorithm", "sha256")
        is_sorted = parsed.get("sorted", True)
        anchor_ids = parsed.get("anchor_ids", [])

        # Reconstruct leaves
        leaves = []
        data_hashes = []
        for leaf_data in parsed.get("leaves", []):
            leaf = MerkleLeaf(
                leaf_index=leaf_data["leaf_index"],
                data_hash=leaf_data["data_hash"],
                anchor_id=leaf_data["anchor_id"],
                leaf_hash=leaf_data["leaf_hash"],
            )
            leaves.append(leaf)
            data_hashes.append(leaf_data["data_hash"])

        # Rebuild tree from data hashes for full internal structure
        if data_hashes:
            rebuilt = self.build_tree(
                record_hashes=data_hashes,
                anchor_ids=anchor_ids,
                hash_algorithm=algorithm,
                sorted_tree=is_sorted,
                tree_id=tree_id,
            )
            return rebuilt

        # Empty tree fallback
        tree_model = MerkleTree(
            tree_id=tree_id,
            root_hash=parsed["root_hash"],
            leaf_count=parsed.get("leaf_count", 0),
            leaves=leaves,
            depth=parsed.get("depth", 0),
            hash_algorithm=algorithm,
            sorted=is_sorted,
            anchor_ids=anchor_ids,
            created_at=_utcnow(),
        )
        return tree_model

    def _deserialize_binary(self, data: str) -> MerkleTree:
        """Deserialize a tree from binary format (base64-encoded).

        Note: Binary deserialization reconstructs the tree model but
        requires the original data hashes for full proof generation.
        Only root hash and leaf hashes are preserved.

        Args:
            data: Base64-encoded binary string.

        Returns:
            MerkleTree model with root and leaf information.
        """
        binary_data = base64.b64decode(data)

        algo_names = {0: "sha256", 1: "sha512", 2: "keccak256"}
        hash_sizes = {0: 32, 1: 64, 2: 32}

        # Parse header (9 bytes: 1+1+1+4+2)
        version, algo_id, sorted_flag, leaf_count, depth = struct.unpack(
            ">BBBIh", binary_data[:9]
        )

        algorithm = algo_names.get(algo_id, "sha256")
        hash_size = hash_sizes.get(algo_id, 32)
        is_sorted = sorted_flag == 1

        offset = 9

        # Parse root hash
        root_hash = binary_data[offset: offset + hash_size].hex()
        offset += hash_size

        # Parse leaf hashes
        leaf_hashes: List[str] = []
        for _ in range(leaf_count):
            leaf_hash = binary_data[offset: offset + hash_size].hex()
            leaf_hashes.append(leaf_hash)
            offset += hash_size

        # Create minimal MerkleTree model
        leaves = [
            MerkleLeaf(
                leaf_index=i,
                data_hash=lh,  # leaf_hash used as data_hash placeholder
                anchor_id=f"binary-{i}",
                leaf_hash=lh,
            )
            for i, lh in enumerate(leaf_hashes)
        ]

        tree_id = str(uuid.uuid4())
        tree_model = MerkleTree(
            tree_id=tree_id,
            root_hash=root_hash,
            leaf_count=leaf_count,
            leaves=leaves,
            depth=depth,
            hash_algorithm=algorithm,
            sorted=is_sorted,
            created_at=_utcnow(),
        )

        with self._lock:
            self._tree_models[tree_id] = tree_model

        logger.info(
            "Tree deserialized from binary: id=%s root=%s leaves=%d",
            tree_id[:16],
            root_hash[:16],
            leaf_count,
        )
        return tree_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core class
    "MerkleProofGenerator",
    # Supporting classes
    "ProofStep",
    "InternalTree",
]
