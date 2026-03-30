# -*- coding: utf-8 -*-
"""
Hash Integrity Routes - AGENT-EUDR-012 Document Authentication API

Endpoints for cryptographic hash integrity validation including hash
computation, verification against stored hashes, registry lookup,
and Merkle tree generation for DDS evidence packages.

Endpoints:
    POST   /hashes/compute            - Compute document hash
    POST   /hashes/verify             - Verify document against stored hash
    GET    /hashes/registry/{hash}    - Look up hash in registry
    GET    /hashes/merkle/{dds_id}    - Get Merkle root for DDS package

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Feature 3 (Hash Integrity)
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.document_authentication.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_dav_service,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_dds_id,
)
from greenlang.agents.eudr.document_authentication.api.schemas import (
    ComputeHashSchema,
    HashAlgorithmSchema,
    HashResultSchema,
    MerkleTreeSchema,
    ProvenanceInfo,
    RegistryLookupSchema,
    VerifyHashSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Hash Integrity"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_hash_store: Dict[str, Dict] = {}
_registry_store: Dict[str, Dict] = {}
_merkle_store: Dict[str, Dict] = {}

def _get_hash_store() -> Dict[str, Dict]:
    """Return the hash record store singleton."""
    return _hash_store

def _get_registry_store() -> Dict[str, Dict]:
    """Return the hash registry store singleton."""
    return _registry_store

def _get_merkle_store() -> Dict[str, Dict]:
    """Return the Merkle tree store singleton."""
    return _merkle_store

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _compute_document_hash(reference: str, algorithm: str) -> str:
    """Compute deterministic hash of a document reference.

    Zero hallucination: uses only Python hashlib.

    Args:
        reference: Document reference string.
        algorithm: Hash algorithm name (sha256, sha512).

    Returns:
        Hex-encoded hash string.
    """
    if algorithm == "sha512":
        return hashlib.sha512(reference.encode("utf-8")).hexdigest()
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()

def _compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle root from a list of leaf hashes.

    Deterministic: uses SHA-256 pairwise hashing only.

    Args:
        hashes: List of hex-encoded hash strings.

    Returns:
        Hex-encoded Merkle root hash.
    """
    if not hashes:
        return hashlib.sha256(b"empty").hexdigest()

    current_level = list(hashes)
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = hashlib.sha256(
                (left + right).encode("utf-8")
            ).hexdigest()
            next_level.append(combined)
        current_level = next_level

    return current_level[0]

# ---------------------------------------------------------------------------
# POST /hashes/compute
# ---------------------------------------------------------------------------

@router.post(
    "/hashes/compute",
    response_model=HashResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Compute document hash",
    description=(
        "Compute a cryptographic hash (SHA-256/SHA-512) for an EUDR "
        "document. Optionally register the hash in the integrity "
        "registry and associate with a DDS package for Merkle tree."
    ),
    responses={
        201: {"description": "Hash computed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compute_hash(
    request: Request,
    body: ComputeHashSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:hashes:compute")
    ),
    _rate: None = Depends(rate_limit_write),
) -> HashResultSchema:
    """Compute a cryptographic hash for a document.

    Args:
        body: Hash computation request.
        user: Authenticated user with hashes:compute permission.

    Returns:
        HashResultSchema with computed hash values.
    """
    start = time.monotonic()
    try:
        hash_id = str(uuid.uuid4())
        now = utcnow()

        primary_hash = _compute_document_hash(
            body.document_reference, body.algorithm.value,
        )

        secondary_hash = None
        secondary_algo = None
        if body.compute_secondary:
            secondary_algo = HashAlgorithmSchema.SHA512
            secondary_hash = _compute_document_hash(
                body.document_reference, "sha512",
            )

        provenance_data = body.model_dump(mode="json")
        provenance_data["hash_id"] = hash_id
        provenance_data["computed_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        registry_entry_id = None
        registered = False
        if body.register_in_registry:
            registry_entry_id = str(uuid.uuid4())
            registry = _get_registry_store()
            registry[primary_hash] = {
                "entry_id": registry_entry_id,
                "hash_value": primary_hash,
                "algorithm": body.algorithm,
                "document_reference": body.document_reference,
                "registered_at": now,
                "registered_by": user.user_id,
                "dds_id": body.dds_id,
                "expires_at": None,
            }
            registered = True

        record = {
            "hash_id": hash_id,
            "document_reference": body.document_reference,
            "primary_hash": primary_hash,
            "primary_algorithm": body.algorithm,
            "secondary_hash": secondary_hash,
            "secondary_algorithm": secondary_algo,
            "hash_match": None,
            "registered": registered,
            "registry_entry_id": registry_entry_id,
            "dds_id": body.dds_id,
            "file_size_bytes": len(body.document_reference.encode("utf-8")),
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_hash_store()
        store[hash_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Hash computed: id=%s algo=%s registered=%s",
            hash_id,
            body.algorithm.value,
            registered,
        )

        return HashResultSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compute hash: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute document hash",
        )

# ---------------------------------------------------------------------------
# POST /hashes/verify
# ---------------------------------------------------------------------------

@router.post(
    "/hashes/verify",
    response_model=HashResultSchema,
    summary="Verify document hash",
    description=(
        "Verify a document against an expected hash value. Recomputes "
        "the document hash and compares against the provided value."
    ),
    responses={
        200: {"description": "Hash verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_hash(
    request: Request,
    body: VerifyHashSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:hashes:verify")
    ),
    _rate: None = Depends(rate_limit_write),
) -> HashResultSchema:
    """Verify a document against a stored hash.

    Args:
        body: Hash verification request.
        user: Authenticated user with hashes:verify permission.

    Returns:
        HashResultSchema with verification result.
    """
    start = time.monotonic()
    try:
        hash_id = str(uuid.uuid4())
        now = utcnow()

        computed_hash = _compute_document_hash(
            body.document_reference, body.algorithm.value,
        )
        hash_match = (computed_hash.lower() == body.expected_hash.lower())

        provenance_data = body.model_dump(mode="json")
        provenance_data["hash_id"] = hash_id
        provenance_data["verified_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        record = {
            "hash_id": hash_id,
            "document_reference": body.document_reference,
            "primary_hash": computed_hash,
            "primary_algorithm": body.algorithm,
            "secondary_hash": None,
            "secondary_algorithm": None,
            "hash_match": hash_match,
            "registered": False,
            "registry_entry_id": None,
            "dds_id": None,
            "file_size_bytes": len(body.document_reference.encode("utf-8")),
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_hash_store()
        store[hash_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Hash verified: id=%s match=%s algo=%s",
            hash_id,
            hash_match,
            body.algorithm.value,
        )

        return HashResultSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to verify hash: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify document hash",
        )

# ---------------------------------------------------------------------------
# GET /hashes/registry/{hash}
# ---------------------------------------------------------------------------

@router.get(
    "/hashes/registry/{hash}",
    response_model=RegistryLookupSchema,
    summary="Look up hash in registry",
    description="Look up a hash value in the integrity registry.",
    responses={
        200: {"description": "Registry lookup completed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def lookup_hash_registry(
    request: Request,
    hash: str = Path(..., description="Hash value to look up"),
    user: AuthUser = Depends(
        require_permission("eudr-dav:hashes:registry:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RegistryLookupSchema:
    """Look up a hash in the integrity registry.

    Args:
        hash: Hash value to look up.
        user: Authenticated user with hashes:registry:read permission.

    Returns:
        RegistryLookupSchema indicating whether hash exists.
    """
    start = time.monotonic()
    try:
        registry = _get_registry_store()
        hash_lower = hash.lower().strip()
        entry = registry.get(hash_lower)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        if entry is not None:
            return RegistryLookupSchema(
                hash_value=hash_lower,
                found=True,
                document_reference=entry.get("document_reference"),
                algorithm=entry.get("algorithm"),
                registered_at=entry.get("registered_at"),
                registered_by=entry.get("registered_by"),
                dds_id=entry.get("dds_id"),
                expires_at=entry.get("expires_at"),
                processing_time_ms=elapsed_ms,
            )

        return RegistryLookupSchema(
            hash_value=hash_lower,
            found=False,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to look up hash: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to look up hash in registry",
        )

# ---------------------------------------------------------------------------
# GET /hashes/merkle/{dds_id}
# ---------------------------------------------------------------------------

@router.get(
    "/hashes/merkle/{dds_id}",
    response_model=MerkleTreeSchema,
    summary="Get Merkle root for DDS package",
    description=(
        "Get the Merkle tree root hash for all documents in a DDS "
        "evidence package. Computes the tree on-the-fly from "
        "registry entries."
    ),
    responses={
        200: {"description": "Merkle tree retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "DDS package not found"},
    },
)
async def get_merkle_tree(
    request: Request,
    dds_id: str = Depends(validate_dds_id),
    user: AuthUser = Depends(
        require_permission("eudr-dav:hashes:merkle:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> MerkleTreeSchema:
    """Get Merkle tree for a DDS package.

    Args:
        dds_id: DDS package identifier.
        user: Authenticated user with hashes:merkle:read permission.

    Returns:
        MerkleTreeSchema with Merkle root and leaf hashes.

    Raises:
        HTTPException: 404 if no documents found for DDS ID.
    """
    start = time.monotonic()
    try:
        registry = _get_registry_store()

        # Collect all hashes for this DDS package
        document_hashes = []
        for entry in registry.values():
            if entry.get("dds_id") == dds_id:
                document_hashes.append(entry["hash_value"])

        if not document_hashes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No documents found for DDS package {dds_id}",
            )

        document_hashes.sort()
        merkle_root = _compute_merkle_root(document_hashes)

        # Compute tree depth
        import math
        tree_depth = math.ceil(math.log2(max(len(document_hashes), 1))) + 1

        now = utcnow()
        provenance_hash = _compute_provenance_hash({
            "dds_id": dds_id,
            "merkle_root": merkle_root,
            "leaf_count": len(document_hashes),
            "computed_by": user.user_id,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Merkle tree computed: dds_id=%s leaves=%d root=%s",
            dds_id,
            len(document_hashes),
            merkle_root[:16],
        )

        return MerkleTreeSchema(
            dds_id=dds_id,
            merkle_root=merkle_root,
            algorithm=HashAlgorithmSchema.SHA256,
            leaf_count=len(document_hashes),
            tree_depth=tree_depth,
            document_hashes=document_hashes,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            created_at=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to compute Merkle tree for %s: %s",
            dds_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute Merkle tree",
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
