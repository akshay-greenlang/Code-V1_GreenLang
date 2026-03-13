# -*- coding: utf-8 -*-
"""
Merkle Routes - AGENT-EUDR-013 Blockchain Integration API

Endpoints for Merkle tree construction, retrieval, proof generation,
and standalone proof verification. Supports SHA-256, SHA-512, and
Keccak-256 hash algorithms with optional sorted-tree mode.

Endpoints:
    POST   /merkle/build              - Build Merkle tree
    GET    /merkle/{tree_id}          - Get Merkle tree
    POST   /merkle/{tree_id}/proof    - Generate inclusion proof
    POST   /merkle/verify             - Verify Merkle proof

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013, Feature 2 (Merkle Tree Proof Engine)
Agent ID: GL-EUDR-BCI-013
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.blockchain_integration.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_blockchain_service,
    get_request_id,
    rate_limit_standard,
    rate_limit_verify,
    require_permission,
    validate_tree_id,
)
from greenlang.agents.eudr.blockchain_integration.api.schemas import (
    HashAlgorithmSchema,
    MerkleBuildRequest,
    MerkleProofRequest,
    MerkleProofResponse,
    MerkleTreeResponse,
    MerkleVerifyRequest,
    MerkleVerifyResponse,
    ProofFormatSchema,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Merkle Proofs"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_tree_store: Dict[str, Dict] = {}


def _get_tree_store() -> Dict[str, Dict]:
    """Return the Merkle tree store singleton."""
    return _tree_store


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash_pair(
    left: str,
    right: str,
    algorithm: str = "sha256",
) -> str:
    """Hash a pair of nodes using the specified algorithm.

    Zero hallucination: deterministic hash computation only.

    Args:
        left: Left node hash.
        right: Right node hash.
        algorithm: Hash algorithm to use.

    Returns:
        Hash of the concatenated pair.
    """
    combined = left + right
    if algorithm == "sha512":
        return hashlib.sha512(combined.encode("utf-8")).hexdigest()
    elif algorithm == "keccak256":
        # Use sha3_256 as Python standard library keccak equivalent
        return hashlib.sha3_256(combined.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _build_merkle_tree(
    leaves: List[str],
    algorithm: str = "sha256",
    sorted_tree: bool = True,
) -> Dict[str, Any]:
    """Build a Merkle tree from leaf hashes.

    Zero hallucination: deterministic tree construction using only
    cryptographic hash functions. No LLM involved.

    Args:
        leaves: List of leaf hashes.
        algorithm: Hash algorithm for internal nodes.
        sorted_tree: Whether to sort leaves before building.

    Returns:
        Dict with tree_id, root_hash, depth, and full tree layers.
    """
    if sorted_tree:
        leaves = sorted(leaves)

    # Build tree bottom-up
    current_layer = list(leaves)
    depth = 0

    while len(current_layer) > 1:
        next_layer: List[str] = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            parent = _hash_pair(left, right, algorithm)
            next_layer.append(parent)
        current_layer = next_layer
        depth += 1

    root_hash = current_layer[0] if current_layer else ""

    return {
        "root_hash": root_hash,
        "depth": depth,
        "leaf_count": len(leaves),
        "leaves": leaves,
    }


def _generate_proof(
    leaves: List[str],
    leaf_hash: str,
    algorithm: str = "sha256",
    sorted_tree: bool = True,
) -> Optional[Dict[str, Any]]:
    """Generate a Merkle inclusion proof for a specific leaf.

    Args:
        leaves: All leaf hashes in the tree.
        leaf_hash: Target leaf hash.
        algorithm: Hash algorithm.
        sorted_tree: Whether leaves were sorted.

    Returns:
        Proof dict with path, root, and index, or None if leaf not found.
    """
    if sorted_tree:
        leaves = sorted(leaves)

    if leaf_hash not in leaves:
        return None

    leaf_index = leaves.index(leaf_hash)
    proof_path: List[str] = []
    current_layer = list(leaves)
    current_index = leaf_index

    while len(current_layer) > 1:
        next_layer: List[str] = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left

            # If current index is in this pair, record sibling
            if i == current_index or i + 1 == current_index:
                if current_index % 2 == 0:
                    # Current is left, sibling is right
                    sibling = right if i + 1 < len(current_layer) else left
                else:
                    # Current is right, sibling is left
                    sibling = left
                proof_path.append(sibling)

            parent = _hash_pair(left, right, algorithm)
            next_layer.append(parent)

        current_layer = next_layer
        current_index = current_index // 2

    root_hash = current_layer[0] if current_layer else ""

    return {
        "leaf_index": leaf_index,
        "proof": proof_path,
        "root_hash": root_hash,
    }


# ---------------------------------------------------------------------------
# POST /merkle/build
# ---------------------------------------------------------------------------


@router.post(
    "/merkle/build",
    response_model=MerkleTreeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Build Merkle tree",
    description=(
        "Build a Merkle tree from a list of leaf hashes. Supports "
        "SHA-256, SHA-512, and Keccak-256 algorithms. Optionally "
        "sorts leaves before building for deterministic trees."
    ),
    responses={
        201: {"description": "Merkle tree built successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def build_merkle_tree(
    request: Request,
    body: MerkleBuildRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:merkle:build")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_blockchain_service),
) -> MerkleTreeResponse:
    """Build a Merkle tree from leaf hashes.

    Args:
        request: FastAPI request object.
        body: Merkle tree build parameters.
        user: Authenticated user with merkle:build permission.
        service: Blockchain integration service.

    Returns:
        MerkleTreeResponse with tree ID and root hash.
    """
    start = time.monotonic()
    try:
        tree_id = str(uuid.uuid4())
        now = _utcnow()

        # Deterministic tree construction (zero hallucination)
        tree_data = _build_merkle_tree(
            leaves=body.leaves,
            algorithm=body.algorithm.value,
            sorted_tree=body.sorted_tree,
        )

        provenance_hash = _compute_provenance_hash({
            "tree_id": tree_id,
            "root_hash": tree_data["root_hash"],
            "leaf_count": tree_data["leaf_count"],
            "algorithm": body.algorithm.value,
            "built_by": user.user_id,
        })

        # Only include leaves for small trees (< 1000)
        include_leaves = tree_data["leaf_count"] <= 1000

        record = {
            "tree_id": tree_id,
            "root_hash": tree_data["root_hash"],
            "algorithm": body.algorithm,
            "leaf_count": tree_data["leaf_count"],
            "depth": tree_data["depth"],
            "sorted_tree": body.sorted_tree,
            "leaves": tree_data["leaves"] if include_leaves else None,
            "created_at": now,
            "anchor_id": None,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
            "_all_leaves": tree_data["leaves"],  # Internal, for proof gen
        }

        store = _get_tree_store()
        store[tree_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Merkle tree built: id=%s root=%s leaves=%d depth=%d "
            "algorithm=%s elapsed_ms=%.1f",
            tree_id,
            tree_data["root_hash"][:16] + "...",
            tree_data["leaf_count"],
            tree_data["depth"],
            body.algorithm.value,
            elapsed_ms,
        )

        # Return without internal _all_leaves field
        response_data = {k: v for k, v in record.items() if not k.startswith("_")}
        return MerkleTreeResponse(**response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to build Merkle tree: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build Merkle tree",
        )


# ---------------------------------------------------------------------------
# GET /merkle/{tree_id}
# ---------------------------------------------------------------------------


@router.get(
    "/merkle/{tree_id}",
    response_model=MerkleTreeResponse,
    summary="Get Merkle tree",
    description="Retrieve details of a previously built Merkle tree.",
    responses={
        200: {"description": "Merkle tree retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Tree not found"},
    },
)
async def get_merkle_tree(
    request: Request,
    tree_id: str = Depends(validate_tree_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:merkle:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_blockchain_service),
) -> MerkleTreeResponse:
    """Get Merkle tree details.

    Args:
        request: FastAPI request object.
        tree_id: Merkle tree identifier.
        user: Authenticated user with merkle:read permission.
        service: Blockchain integration service.

    Returns:
        MerkleTreeResponse with tree details.

    Raises:
        HTTPException: 404 if tree not found.
    """
    try:
        store = _get_tree_store()
        record = store.get(tree_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Merkle tree {tree_id} not found",
            )

        response_data = {k: v for k, v in record.items() if not k.startswith("_")}
        return MerkleTreeResponse(**response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get Merkle tree %s: %s",
            tree_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve Merkle tree",
        )


# ---------------------------------------------------------------------------
# POST /merkle/{tree_id}/proof
# ---------------------------------------------------------------------------


@router.post(
    "/merkle/{tree_id}/proof",
    response_model=MerkleProofResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate inclusion proof",
    description=(
        "Generate a Merkle inclusion proof for a specific leaf hash "
        "within a previously built Merkle tree."
    ),
    responses={
        200: {"description": "Proof generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Tree or leaf not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_proof(
    request: Request,
    body: MerkleProofRequest,
    tree_id: str = Depends(validate_tree_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:merkle:proof")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_blockchain_service),
) -> MerkleProofResponse:
    """Generate a Merkle inclusion proof.

    Args:
        request: FastAPI request object.
        body: Proof generation request.
        tree_id: Merkle tree identifier.
        user: Authenticated user with merkle:proof permission.
        service: Blockchain integration service.

    Returns:
        MerkleProofResponse with proof path.

    Raises:
        HTTPException: 404 if tree not found or leaf not in tree.
    """
    start = time.monotonic()
    try:
        store = _get_tree_store()
        record = store.get(tree_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Merkle tree {tree_id} not found",
            )

        # Get all leaves (including internal storage)
        all_leaves = record.get("_all_leaves") or record.get("leaves") or []
        algorithm = record.get("algorithm", HashAlgorithmSchema.SHA256)
        sorted_tree = record.get("sorted_tree", True)

        # Handle enum values
        algo_value = algorithm.value if hasattr(algorithm, "value") else str(algorithm)

        proof_data = _generate_proof(
            leaves=all_leaves,
            leaf_hash=body.leaf_hash,
            algorithm=algo_value,
            sorted_tree=sorted_tree,
        )

        if proof_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Leaf hash {body.leaf_hash[:16]}... not found in tree {tree_id}",
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Merkle proof generated: tree=%s leaf=%s path_length=%d "
            "elapsed_ms=%.1f",
            tree_id,
            body.leaf_hash[:16] + "...",
            len(proof_data["proof"]),
            elapsed_ms,
        )

        return MerkleProofResponse(
            tree_id=tree_id,
            leaf_hash=body.leaf_hash,
            leaf_index=proof_data["leaf_index"],
            proof=proof_data["proof"],
            root_hash=proof_data["root_hash"],
            algorithm=algorithm,
            format=body.format,
            generated_at=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate proof for tree %s: %s",
            tree_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate Merkle proof",
        )


# ---------------------------------------------------------------------------
# POST /merkle/verify
# ---------------------------------------------------------------------------


@router.post(
    "/merkle/verify",
    response_model=MerkleVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify Merkle proof",
    description=(
        "Verify a standalone Merkle inclusion proof by recomputing "
        "the root hash from the leaf and proof path."
    ),
    responses={
        200: {"description": "Proof verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_merkle_proof(
    request: Request,
    body: MerkleVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:merkle:verify")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_blockchain_service),
) -> MerkleVerifyResponse:
    """Verify a Merkle inclusion proof.

    Zero hallucination: deterministic hash chain verification only.

    Args:
        request: FastAPI request object.
        body: Proof verification request.
        user: Authenticated user with merkle:verify permission.
        service: Blockchain integration service.

    Returns:
        MerkleVerifyResponse with verification result.
    """
    start = time.monotonic()
    try:
        algo_value = body.algorithm.value

        # Recompute root from leaf and proof path (deterministic)
        current_hash = body.leaf_hash
        for sibling_hash in body.proof:
            # Sorted tree: smaller hash always on left
            if current_hash <= sibling_hash:
                current_hash = _hash_pair(current_hash, sibling_hash, algo_value)
            else:
                current_hash = _hash_pair(sibling_hash, current_hash, algo_value)

        is_valid = current_hash == body.root_hash

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Merkle proof verified: leaf=%s valid=%s "
            "proof_length=%d elapsed_ms=%.1f",
            body.leaf_hash[:16] + "...",
            is_valid,
            len(body.proof),
            elapsed_ms,
        )

        return MerkleVerifyResponse(
            is_valid=is_valid,
            leaf_hash=body.leaf_hash,
            root_hash=body.root_hash,
            computed_root=current_hash,
            algorithm=body.algorithm,
            verified_at=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to verify Merkle proof: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify Merkle proof",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
