# -*- coding: utf-8 -*-
"""
Hash Integrity Validator Engine - AGENT-EUDR-012: Document Authentication (Engine 3)

SHA-256/SHA-512 hash-based tamper detection, document deduplication, and
Merkle tree construction engine for EUDR document authentication. Provides
an immutable hash registry with first-seen timestamps, duplicate detection,
modification tracking, incremental hashing for large documents, HMAC-based
integrity verification, and Merkle tree evidence packaging.

Zero-Hallucination Guarantees:
    - All hash computations use Python hashlib (SHA-256, SHA-512, HMAC)
    - No ML/LLM used for any integrity logic
    - Hash comparisons are constant-time (hmac.compare_digest)
    - Merkle tree construction is deterministic and reproducible
    - SHA-256 provenance hashes on every operation
    - Immutable hash registry for EUDR Article 14 retention

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligations
    - EU 2023/1115 (EUDR) Article 10: Document verification requirements
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - NIST FIPS 180-4: Secure Hash Standard (SHA-256, SHA-512)
    - NIST FIPS 198-1: HMAC specification

Performance Targets:
    - SHA-256 hash (1MB): <5ms
    - SHA-512 hash (1MB): <5ms
    - Incremental hash (100MB): <2 seconds
    - Duplicate lookup: <1ms
    - Merkle tree (100 leaves): <50ms
    - HMAC computation (1MB): <5ms

PRD Feature References:
    - PRD-AGENT-EUDR-012 Feature 3: Hash Integrity Validation
    - PRD-AGENT-EUDR-012 Feature 3.1: SHA-256/SHA-512 Dual Hashing
    - PRD-AGENT-EUDR-012 Feature 3.2: Immutable Hash Registry
    - PRD-AGENT-EUDR-012 Feature 3.3: Duplicate Detection
    - PRD-AGENT-EUDR-012 Feature 3.4: Modification Detection
    - PRD-AGENT-EUDR-012 Feature 3.5: Incremental Hashing
    - PRD-AGENT-EUDR-012 Feature 3.6: Merkle Tree Construction
    - PRD-AGENT-EUDR-012 Feature 3.7: HMAC Integrity
    - PRD-AGENT-EUDR-012 Feature 3.8: Registry Statistics

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.document_authentication.config import get_config
from greenlang.agents.eudr.document_authentication.metrics import (
    observe_verification_duration,
    record_api_error,
    record_duplicate_detected,
    record_hash_computed,
    record_tampering_detected,
)
from greenlang.agents.eudr.document_authentication.models import (
    HashAlgorithm,
    HashRecord,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
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


def _compute_provenance_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default incremental hash chunk size (1MB).
_DEFAULT_CHUNK_SIZE: int = 1_048_576

#: Large document threshold (100MB).
_LARGE_DOCUMENT_THRESHOLD: int = 104_857_600

#: Maximum HMAC key size in bytes.
_MAX_HMAC_KEY_SIZE: int = 1024

#: Supported hash algorithms for compute_hash.
_SUPPORTED_ALGORITHMS = frozenset({"sha256", "sha512"})


# ---------------------------------------------------------------------------
# Internal: Hash Registry Entry
# ---------------------------------------------------------------------------


class _RegistryEntry:
    """Internal hash registry entry for tracking document hashes.

    Attributes:
        hash_sha256: SHA-256 hash of the document.
        hash_sha512: SHA-512 hash of the document.
        document_id: ID of the document that first registered this hash.
        filename: Original filename of the first document.
        file_size_bytes: File size in bytes.
        first_seen_at: UTC timestamp when this hash was first seen.
        last_seen_at: UTC timestamp when this hash was last seen.
        seen_count: Number of times this hash has been encountered.
        parent_hash: Hash chain parent (for anchoring).
        expires_at: Registry entry expiration date.
    """

    __slots__ = (
        "hash_sha256", "hash_sha512", "document_id", "filename",
        "file_size_bytes", "first_seen_at", "last_seen_at",
        "seen_count", "parent_hash", "expires_at",
    )

    def __init__(
        self,
        hash_sha256: str,
        hash_sha512: str,
        document_id: str,
        filename: str,
        file_size_bytes: int,
        ttl_days: int,
        parent_hash: Optional[str] = None,
    ) -> None:
        """Initialize a registry entry.

        Args:
            hash_sha256: SHA-256 hash value.
            hash_sha512: SHA-512 hash value.
            document_id: Document that first registered.
            filename: Original filename.
            file_size_bytes: File size in bytes.
            ttl_days: Registry TTL in days.
            parent_hash: Optional parent hash for chain anchoring.
        """
        now = _utcnow()
        self.hash_sha256 = hash_sha256
        self.hash_sha512 = hash_sha512
        self.document_id = document_id
        self.filename = filename
        self.file_size_bytes = file_size_bytes
        self.first_seen_at = now
        self.last_seen_at = now
        self.seen_count = 1
        self.parent_hash = parent_hash
        self.expires_at = now + timedelta(days=ttl_days)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "hash_sha256": self.hash_sha256,
            "hash_sha512": self.hash_sha512,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "seen_count": self.seen_count,
            "parent_hash": self.parent_hash,
            "expires_at": self.expires_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Internal: Merkle Tree Node
# ---------------------------------------------------------------------------


class _MerkleNode:
    """Internal Merkle tree node.

    Attributes:
        hash_value: SHA-256 hash of this node.
        left: Left child node (or None for leaf).
        right: Right child node (or None for leaf).
        is_leaf: Whether this is a leaf node.
        data_index: Index of the source data (leaf only).
    """

    __slots__ = (
        "hash_value", "left", "right", "is_leaf", "data_index",
    )

    def __init__(
        self,
        hash_value: str,
        left: Optional[_MerkleNode] = None,
        right: Optional[_MerkleNode] = None,
        is_leaf: bool = False,
        data_index: int = -1,
    ) -> None:
        """Initialize a Merkle tree node.

        Args:
            hash_value: Hash of this node.
            left: Left child.
            right: Right child.
            is_leaf: Whether this is a leaf node.
            data_index: Source data index for leaf nodes.
        """
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.data_index = data_index

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        result: Dict[str, Any] = {
            "hash": self.hash_value,
            "is_leaf": self.is_leaf,
        }
        if self.is_leaf:
            result["data_index"] = self.data_index
        if self.left is not None:
            result["left"] = self.left.to_dict()
        if self.right is not None:
            result["right"] = self.right.to_dict()
        return result


# ---------------------------------------------------------------------------
# HashIntegrityValidator
# ---------------------------------------------------------------------------


class HashIntegrityValidator:
    """SHA-256/SHA-512 hash-based tamper detection and document deduplication engine.

    Provides comprehensive hash integrity validation for EUDR document
    authentication including dual SHA-256/SHA-512 hashing, an immutable
    hash registry with first-seen timestamps and EUDR Article 14 five-year
    retention, duplicate detection, modification tracking, incremental
    hashing for large documents, HMAC-based integrity verification, and
    Merkle tree construction for DDS evidence packages.

    All hash computations use Python hashlib. No ML or LLM is used.
    Hash comparisons use hmac.compare_digest for constant-time safety.
    Every operation produces a SHA-256 provenance hash for tamper-evident
    audit trails.

    Thread Safety:
        All public methods are thread-safe via reentrant locking.
        The hash registry is guarded by the lock.

    Attributes:
        _config: Document authentication configuration singleton.
        _provenance: ProvenanceTracker for audit trail hashing.
        _registry_sha256: SHA-256 -> _RegistryEntry mapping.
        _registry_sha512: SHA-512 -> _RegistryEntry mapping.
        _filename_hashes: filename -> list of SHA-256 hashes mapping.
        _document_hashes: document_id -> SHA-256 hash mapping.
        _operation_history: Ordered list of hash operations.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> validator = HashIntegrityValidator()
        >>> record = validator.compute_hash(b"document content", "doc-001", "test.pdf")
        >>> assert record.hash_value != ""
        >>> verified = validator.verify_hash(b"document content", record.hash_value)
        >>> assert verified is True
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize HashIntegrityValidator.

        Args:
            config: Optional DocumentAuthenticationConfig override.
                If None, uses the singleton from get_config().
            provenance: Optional ProvenanceTracker override. If None,
                creates a new instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker(
            genesis_hash=self._config.genesis_hash,
        )

        # -- Hash registry -----------------------------------------------
        self._registry_sha256: Dict[str, _RegistryEntry] = {}
        self._registry_sha512: Dict[str, _RegistryEntry] = {}
        self._filename_hashes: Dict[str, List[str]] = {}
        self._document_hashes: Dict[str, str] = {}

        # -- Operation history -------------------------------------------
        self._operation_history: List[Dict[str, Any]] = []

        # -- Thread safety -----------------------------------------------
        self._lock = threading.RLock()

        logger.info(
            "HashIntegrityValidator initialized: module_version=%s, "
            "primary=%s, secondary=%s, ttl=%dd",
            _MODULE_VERSION,
            self._config.hash_algorithm,
            self._config.secondary_hash,
            self._config.registry_ttl_days,
        )

    # ------------------------------------------------------------------
    # Public API: Compute Hash
    # ------------------------------------------------------------------

    def compute_hash(
        self,
        document_bytes: bytes,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        register: bool = True,
    ) -> HashRecord:
        """Compute SHA-256 and SHA-512 hashes for a document.

        Computes both primary (SHA-256) and secondary (SHA-512) hashes,
        checks the registry for duplicates, and optionally registers
        the hash for future lookups.

        For documents larger than 100MB, incremental hashing is used
        automatically for memory efficiency.

        Args:
            document_bytes: Raw document content in bytes.
            document_id: Optional document ID. Auto-generated if None.
            filename: Optional original filename for tracking.
            register: Whether to register the hash in the registry.

        Returns:
            HashRecord with computed hashes, duplicate status, and
            provenance hash.

        Raises:
            ValueError: If document_bytes is empty.
        """
        start_time = time.monotonic()
        doc_id = document_id or _generate_id()
        fname = filename or "unknown"

        try:
            if not document_bytes:
                raise ValueError("document_bytes must not be empty")

            # -- Compute hashes (incremental for large docs) ---------------
            if len(document_bytes) > _LARGE_DOCUMENT_THRESHOLD:
                hash_sha256 = self._incremental_hash(
                    document_bytes, "sha256",
                )
                hash_sha512 = self._incremental_hash(
                    document_bytes, "sha512",
                )
            else:
                hash_sha256 = hashlib.sha256(document_bytes).hexdigest()
                hash_sha512 = hashlib.sha512(document_bytes).hexdigest()

            # -- Check for duplicates --------------------------------------
            is_duplicate, duplicate_doc_id = self._check_duplicate_internal(
                hash_sha256,
            )

            # -- Register hash if requested --------------------------------
            registry_expires_at: Optional[datetime] = None
            if register:
                registry_expires_at = self._register_hash_internal(
                    hash_sha256=hash_sha256,
                    hash_sha512=hash_sha512,
                    document_id=doc_id,
                    filename=fname,
                    file_size_bytes=len(document_bytes),
                )

            # -- Build result ----------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            provenance_data = {
                "document_id": doc_id,
                "hash_sha256": hash_sha256,
                "hash_sha512": hash_sha512,
                "is_duplicate": is_duplicate,
                "file_size": len(document_bytes),
            }
            prov_hash = _compute_provenance_hash(provenance_data)

            record = HashRecord(
                document_id=doc_id,
                algorithm=HashAlgorithm.SHA256,
                hash_value=hash_sha256,
                secondary_algorithm=HashAlgorithm.SHA512,
                secondary_hash_value=hash_sha512,
                is_duplicate=is_duplicate,
                duplicate_document_id=duplicate_doc_id,
                registry_expires_at=registry_expires_at,
                provenance_hash=prov_hash,
            )

            # -- Record provenance -----------------------------------------
            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="hash",
                    action="compute_hash",
                    entity_id=doc_id,
                    data=provenance_data,
                    metadata={
                        "document_id": doc_id,
                        "is_duplicate": is_duplicate,
                    },
                )

            # -- Record metrics --------------------------------------------
            if self._config.enable_metrics:
                record_hash_computed("sha256")
                record_hash_computed("sha512")
                if is_duplicate:
                    record_duplicate_detected()
                observe_verification_duration(elapsed_ms / 1000.0)

            # -- Record history --------------------------------------------
            self._record_operation(
                "compute_hash", doc_id, hash_sha256,
                is_duplicate, elapsed_ms,
            )

            logger.info(
                "Hash computed: doc_id=%s sha256=%s..%s duplicate=%s "
                "size=%d time=%.1fms",
                doc_id[:12],
                hash_sha256[:8], hash_sha256[-8:],
                is_duplicate,
                len(document_bytes),
                elapsed_ms,
            )

            return record

        except ValueError:
            raise
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Hash computation failed for doc_id=%s: %s (%.1fms)",
                doc_id[:12], str(e), elapsed_ms,
                exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("compute_hash")
            raise

    # ------------------------------------------------------------------
    # Public API: Verify Hash
    # ------------------------------------------------------------------

    def verify_hash(
        self,
        document_bytes: bytes,
        expected_hash: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify a document against an expected hash value.

        Uses hmac.compare_digest for constant-time comparison to
        prevent timing attacks.

        Args:
            document_bytes: Raw document content.
            expected_hash: Expected hex-encoded hash value.
            algorithm: Hash algorithm ('sha256' or 'sha512').

        Returns:
            True if the computed hash matches the expected hash.

        Raises:
            ValueError: If document_bytes is empty or algorithm unsupported.
        """
        start_time = time.monotonic()

        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not expected_hash:
            raise ValueError("expected_hash must not be empty")

        algorithm_lower = algorithm.lower().strip()
        if algorithm_lower not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )

        # -- Compute hash --------------------------------------------------
        if algorithm_lower == "sha256":
            computed = hashlib.sha256(document_bytes).hexdigest()
        else:
            computed = hashlib.sha512(document_bytes).hexdigest()

        # -- Constant-time comparison --------------------------------------
        match = hmac.compare_digest(computed, expected_hash.lower())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # -- Record provenance ---------------------------------------------
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="hash",
                action="verify_hash",
                entity_id=expected_hash[:16],
                data={
                    "algorithm": algorithm_lower,
                    "match": match,
                    "expected_prefix": expected_hash[:16],
                },
            )

        if self._config.enable_metrics:
            record_hash_computed(algorithm_lower)
            if not match:
                record_tampering_detected()

        self._record_operation(
            "verify_hash", expected_hash[:16], computed,
            not match, elapsed_ms,
        )

        logger.info(
            "Hash verified: algorithm=%s match=%s time=%.1fms",
            algorithm_lower, match, elapsed_ms,
        )

        return match

    # ------------------------------------------------------------------
    # Public API: Register Hash
    # ------------------------------------------------------------------

    def register_hash(
        self,
        hash_sha256: str,
        hash_sha512: str,
        document_id: str,
        filename: str = "unknown",
        file_size_bytes: int = 0,
    ) -> Dict[str, Any]:
        """Register a pre-computed hash pair in the registry.

        Used when hashes have been computed externally and need to be
        registered for duplicate detection.

        Args:
            hash_sha256: SHA-256 hash value.
            hash_sha512: SHA-512 hash value.
            document_id: Document ID to associate.
            filename: Original filename.
            file_size_bytes: File size in bytes.

        Returns:
            Dictionary with registration status and details.

        Raises:
            ValueError: If hash values are empty or malformed.
        """
        if not hash_sha256 or len(hash_sha256) != 64:
            raise ValueError(
                "hash_sha256 must be a 64-character hex string"
            )
        if not hash_sha512 or len(hash_sha512) != 128:
            raise ValueError(
                "hash_sha512 must be a 128-character hex string"
            )
        if not document_id:
            raise ValueError("document_id must not be empty")

        expires_at = self._register_hash_internal(
            hash_sha256=hash_sha256,
            hash_sha512=hash_sha512,
            document_id=document_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
        )

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="hash",
                action="compute_hash",
                entity_id=document_id,
                data={
                    "hash_sha256": hash_sha256,
                    "hash_sha512": hash_sha512,
                },
                metadata={"action": "register"},
            )

        logger.info(
            "Hash registered: doc_id=%s sha256=%s..%s",
            document_id[:12], hash_sha256[:8], hash_sha256[-8:],
        )

        return {
            "document_id": document_id,
            "hash_sha256": hash_sha256,
            "hash_sha512": hash_sha512,
            "status": "registered",
            "expires_at": expires_at.isoformat() if expires_at else None,
        }

    # ------------------------------------------------------------------
    # Public API: Lookup Hash
    # ------------------------------------------------------------------

    def lookup_hash(
        self, hash_value: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up a hash in the registry.

        Args:
            hash_value: SHA-256 or SHA-512 hash to look up.

        Returns:
            Registry entry dictionary if found, None otherwise.
        """
        with self._lock:
            # Try SHA-256 first
            entry = self._registry_sha256.get(hash_value)
            if entry is not None:
                return entry.to_dict()

            # Try SHA-512
            entry = self._registry_sha512.get(hash_value)
            if entry is not None:
                return entry.to_dict()

        return None

    # ------------------------------------------------------------------
    # Public API: Check Duplicate
    # ------------------------------------------------------------------

    def check_duplicate(
        self, document_bytes: bytes,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a document is a duplicate based on its hash.

        Computes SHA-256 of the document and checks the registry.

        Args:
            document_bytes: Raw document content.

        Returns:
            Tuple of (is_duplicate, duplicate_document_id).

        Raises:
            ValueError: If document_bytes is empty.
        """
        if not document_bytes:
            raise ValueError("document_bytes must not be empty")

        hash_sha256 = hashlib.sha256(document_bytes).hexdigest()
        return self._check_duplicate_internal(hash_sha256)

    # ------------------------------------------------------------------
    # Public API: Detect Modification
    # ------------------------------------------------------------------

    def detect_modification(
        self,
        document_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Detect if a document with the same filename has been modified.

        Compares the current hash against previously registered hashes
        for the same filename. If the filename was seen before with a
        different hash, a modification is detected.

        Args:
            document_bytes: Raw document content.
            filename: Filename to check against.

        Returns:
            Dictionary with modification detection results:
                - modified (bool): Whether modification was detected
                - current_hash (str): Hash of the provided document
                - previous_hashes (list): Previously registered hashes
                - previous_document_ids (list): Documents with same name

        Raises:
            ValueError: If inputs are empty.
        """
        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not filename:
            raise ValueError("filename must not be empty")

        start_time = time.monotonic()

        current_hash = hashlib.sha256(document_bytes).hexdigest()

        with self._lock:
            previous_hashes = self._filename_hashes.get(filename, [])

        modified = False
        previous_doc_ids: List[str] = []

        if previous_hashes:
            for prev_hash in previous_hashes:
                if not hmac.compare_digest(current_hash, prev_hash):
                    modified = True
                    # Look up the document ID for the previous hash
                    with self._lock:
                        entry = self._registry_sha256.get(prev_hash)
                        if entry:
                            previous_doc_ids.append(entry.document_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        if modified and self._config.enable_metrics:
            record_tampering_detected()

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="hash",
                action="verify_hash",
                entity_id=filename,
                data={
                    "modified": modified,
                    "current_hash": current_hash,
                    "previous_count": len(previous_hashes),
                },
                metadata={"filename": filename},
            )

        self._record_operation(
            "detect_modification", filename, current_hash,
            modified, elapsed_ms,
        )

        if modified:
            logger.warning(
                "Document modification detected: filename=%s "
                "current=%s..%s previous_count=%d time=%.1fms",
                filename, current_hash[:8], current_hash[-8:],
                len(previous_hashes), elapsed_ms,
            )
        else:
            logger.debug(
                "No modification for filename=%s time=%.1fms",
                filename, elapsed_ms,
            )

        return {
            "modified": modified,
            "current_hash": current_hash,
            "previous_hashes": list(previous_hashes),
            "previous_document_ids": previous_doc_ids,
            "filename": filename,
        }

    # ------------------------------------------------------------------
    # Public API: Merkle Tree Construction
    # ------------------------------------------------------------------

    def build_merkle_tree(
        self,
        document_hashes: List[str],
    ) -> Dict[str, Any]:
        """Build a Merkle tree from a list of document hashes.

        Constructs a binary Merkle tree suitable for DDS evidence
        packages. The root hash provides a single tamper-evident
        digest for an entire set of documents.

        If the number of hashes is odd, the last hash is duplicated
        to make the count even at each level.

        Args:
            document_hashes: List of hex-encoded SHA-256 hashes.

        Returns:
            Dictionary containing:
                - root_hash (str): Merkle root hash
                - leaf_count (int): Number of leaf hashes
                - tree_depth (int): Depth of the tree
                - tree (dict): Full tree structure
                - provenance_hash (str): Provenance hash of the operation

        Raises:
            ValueError: If document_hashes is empty.
        """
        start_time = time.monotonic()

        if not document_hashes:
            raise ValueError("document_hashes must not be empty")

        # -- Build leaf nodes ----------------------------------------------
        leaves: List[_MerkleNode] = []
        for idx, h in enumerate(document_hashes):
            node = _MerkleNode(
                hash_value=h,
                is_leaf=True,
                data_index=idx,
            )
            leaves.append(node)

        # -- Build tree bottom-up ------------------------------------------
        current_level = leaves
        depth = 0

        while len(current_level) > 1:
            next_level: List[_MerkleNode] = []

            # Pad with duplicate of last if odd
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1]
                combined = left.hash_value + right.hash_value
                parent_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()
                parent = _MerkleNode(
                    hash_value=parent_hash,
                    left=left,
                    right=right,
                )
                next_level.append(parent)

            current_level = next_level
            depth += 1

        root = current_level[0]
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        provenance_data = {
            "root_hash": root.hash_value,
            "leaf_count": len(document_hashes),
            "tree_depth": depth,
        }
        prov_hash = _compute_provenance_hash(provenance_data)

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="hash",
                action="compute_hash",
                entity_id=root.hash_value[:16],
                data=provenance_data,
                metadata={"action": "build_merkle_tree"},
            )

        logger.info(
            "Merkle tree built: root=%s..%s leaves=%d depth=%d "
            "time=%.1fms",
            root.hash_value[:8], root.hash_value[-8:],
            len(document_hashes), depth, elapsed_ms,
        )

        return {
            "root_hash": root.hash_value,
            "leaf_count": len(document_hashes),
            "tree_depth": depth,
            "tree": root.to_dict(),
            "provenance_hash": prov_hash,
        }

    def get_merkle_root(
        self,
        document_hashes: List[str],
    ) -> str:
        """Compute only the Merkle root hash without full tree structure.

        Lightweight version of build_merkle_tree that returns only
        the root hash string.

        Args:
            document_hashes: List of hex-encoded SHA-256 hashes.

        Returns:
            Merkle root hash string.

        Raises:
            ValueError: If document_hashes is empty.
        """
        if not document_hashes:
            raise ValueError("document_hashes must not be empty")

        current_level = list(document_hashes)

        while len(current_level) > 1:
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                parent_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()
                next_level.append(parent_hash)

            current_level = next_level

        return current_level[0]

    # ------------------------------------------------------------------
    # Public API: HMAC Operations
    # ------------------------------------------------------------------

    def compute_hmac(
        self,
        document_bytes: bytes,
        key: bytes,
        algorithm: str = "sha256",
    ) -> str:
        """Compute HMAC for a document with a secret key.

        Args:
            document_bytes: Raw document content.
            key: Secret key bytes for HMAC computation.
            algorithm: Hash algorithm ('sha256' or 'sha512').

        Returns:
            Hex-encoded HMAC value.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not key:
            raise ValueError("key must not be empty")
        if len(key) > _MAX_HMAC_KEY_SIZE:
            raise ValueError(
                f"key exceeds maximum size of {_MAX_HMAC_KEY_SIZE} bytes"
            )

        algorithm_lower = algorithm.lower().strip()
        if algorithm_lower not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}"
            )

        hmac_value = hmac.new(
            key, document_bytes, algorithm_lower,
        ).hexdigest()

        if self._config.enable_metrics:
            record_hash_computed(f"hmac_{algorithm_lower}")

        logger.debug(
            "HMAC computed: algorithm=%s length=%d",
            algorithm_lower, len(document_bytes),
        )

        return hmac_value

    def verify_hmac(
        self,
        document_bytes: bytes,
        key: bytes,
        expected_hmac: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify an HMAC tag for a document.

        Uses hmac.compare_digest for constant-time comparison.

        Args:
            document_bytes: Raw document content.
            key: Secret key bytes.
            expected_hmac: Expected hex-encoded HMAC value.
            algorithm: Hash algorithm.

        Returns:
            True if the computed HMAC matches the expected value.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not key:
            raise ValueError("key must not be empty")
        if not expected_hmac:
            raise ValueError("expected_hmac must not be empty")

        computed = self.compute_hmac(document_bytes, key, algorithm)
        match = hmac.compare_digest(computed, expected_hmac.lower())

        if not match and self._config.enable_metrics:
            record_tampering_detected()

        logger.debug(
            "HMAC verified: algorithm=%s match=%s", algorithm, match,
        )

        return match

    # ------------------------------------------------------------------
    # Public API: Registry Statistics
    # ------------------------------------------------------------------

    def get_registry_stats(self) -> Dict[str, Any]:
        """Return hash registry statistics.

        Returns:
            Dictionary with:
                - total_entries: Number of unique hashes
                - total_duplicates_detected: Total duplicates found
                - total_operations: Total hash operations performed
                - oldest_entry: Oldest registry entry date
                - newest_entry: Newest registry entry date
                - expired_entries: Number of expired entries
                - active_entries: Number of active entries
                - total_bytes_hashed: Approximate bytes hashed
        """
        now = _utcnow()

        with self._lock:
            total_entries = len(self._registry_sha256)

            expired = 0
            active = 0
            oldest: Optional[datetime] = None
            newest: Optional[datetime] = None
            total_bytes = 0
            total_dupes = 0

            for entry in self._registry_sha256.values():
                total_bytes += entry.file_size_bytes
                if entry.seen_count > 1:
                    total_dupes += entry.seen_count - 1

                if entry.expires_at <= now:
                    expired += 1
                else:
                    active += 1

                if oldest is None or entry.first_seen_at < oldest:
                    oldest = entry.first_seen_at
                if newest is None or entry.first_seen_at > newest:
                    newest = entry.first_seen_at

            total_operations = len(self._operation_history)

        return {
            "total_entries": total_entries,
            "total_duplicates_detected": total_dupes,
            "total_operations": total_operations,
            "oldest_entry": oldest.isoformat() if oldest else None,
            "newest_entry": newest.isoformat() if newest else None,
            "expired_entries": expired,
            "active_entries": active,
            "total_bytes_hashed": total_bytes,
            "module_version": _MODULE_VERSION,
        }

    # ------------------------------------------------------------------
    # Public API: Operation History
    # ------------------------------------------------------------------

    def get_operation_history(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve hash operation history with optional filters.

        Args:
            operation_type: Filter by operation type (compute_hash,
                verify_hash, detect_modification).
            limit: Maximum records to return.

        Returns:
            List of operation dictionaries, newest first.
        """
        with self._lock:
            history = list(self._operation_history)

        if operation_type:
            history = [
                h for h in history
                if h["operation"] == operation_type
            ]

        history = history[-limit:] if len(history) > limit else history
        history.reverse()
        return history

    # ------------------------------------------------------------------
    # Internal: Duplicate checking
    # ------------------------------------------------------------------

    def _check_duplicate_internal(
        self,
        hash_sha256: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a SHA-256 hash exists in the registry.

        Args:
            hash_sha256: SHA-256 hash to check.

        Returns:
            Tuple of (is_duplicate, existing_document_id).
        """
        with self._lock:
            entry = self._registry_sha256.get(hash_sha256)
            if entry is not None:
                entry.seen_count += 1
                entry.last_seen_at = _utcnow()
                return True, entry.document_id

        return False, None

    # ------------------------------------------------------------------
    # Internal: Hash registration
    # ------------------------------------------------------------------

    def _register_hash_internal(
        self,
        hash_sha256: str,
        hash_sha512: str,
        document_id: str,
        filename: str,
        file_size_bytes: int,
    ) -> Optional[datetime]:
        """Register a hash pair in the internal registry.

        Args:
            hash_sha256: SHA-256 hash value.
            hash_sha512: SHA-512 hash value.
            document_id: Document ID.
            filename: Original filename.
            file_size_bytes: File size in bytes.

        Returns:
            Registry expiration datetime.
        """
        ttl_days = self._config.registry_ttl_days

        with self._lock:
            # Check if already registered
            if hash_sha256 in self._registry_sha256:
                existing = self._registry_sha256[hash_sha256]
                existing.seen_count += 1
                existing.last_seen_at = _utcnow()
                return existing.expires_at

            # Determine parent hash for chain anchoring
            parent_hash: Optional[str] = None
            if self._document_hashes:
                # Chain to most recently registered hash
                last_doc_id = list(self._document_hashes.keys())[-1]
                parent_hash = self._document_hashes.get(last_doc_id)

            entry = _RegistryEntry(
                hash_sha256=hash_sha256,
                hash_sha512=hash_sha512,
                document_id=document_id,
                filename=filename,
                file_size_bytes=file_size_bytes,
                ttl_days=ttl_days,
                parent_hash=parent_hash,
            )

            self._registry_sha256[hash_sha256] = entry
            self._registry_sha512[hash_sha512] = entry
            self._document_hashes[document_id] = hash_sha256

            # Track filename -> hashes mapping
            if filename not in self._filename_hashes:
                self._filename_hashes[filename] = []
            self._filename_hashes[filename].append(hash_sha256)

        return entry.expires_at

    # ------------------------------------------------------------------
    # Internal: Incremental hashing
    # ------------------------------------------------------------------

    def _incremental_hash(
        self,
        document_bytes: bytes,
        algorithm: str = "sha256",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> str:
        """Compute hash incrementally for large documents.

        Processes the document in chunks to limit memory usage for
        documents larger than 100MB.

        Args:
            document_bytes: Raw document content.
            algorithm: Hash algorithm ('sha256' or 'sha512').
            chunk_size: Size of each chunk in bytes.

        Returns:
            Hex-encoded hash value.
        """
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            hasher = hashlib.sha512()

        offset = 0
        total = len(document_bytes)

        while offset < total:
            end = min(offset + chunk_size, total)
            hasher.update(document_bytes[offset:end])
            offset = end

        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Internal: Operation history recording
    # ------------------------------------------------------------------

    def _record_operation(
        self,
        operation: str,
        entity_id: str,
        hash_value: str,
        anomaly_detected: bool,
        elapsed_ms: float,
    ) -> None:
        """Record a hash operation in the history log.

        Args:
            operation: Operation type string.
            entity_id: Entity identifier.
            hash_value: Computed or verified hash.
            anomaly_detected: Whether anomaly/duplicate was found.
            elapsed_ms: Processing time in milliseconds.
        """
        entry = {
            "operation": operation,
            "entity_id": entity_id,
            "hash_prefix": hash_value[:16] if hash_value else "",
            "anomaly_detected": anomaly_detected,
            "processing_time_ms": round(elapsed_ms, 2),
            "timestamp": _utcnow().isoformat(),
        }
        with self._lock:
            self._operation_history.append(entry)

    # ------------------------------------------------------------------
    # Public API: Batch Hash Computation
    # ------------------------------------------------------------------

    def batch_compute_hashes(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[HashRecord]:
        """Compute hashes for multiple documents in a single batch.

        Each document is hashed independently. Failures for individual
        documents are logged but do not abort the entire batch.

        Args:
            documents: List of document dictionaries, each containing:
                - document_bytes (bytes): Raw document content
                - document_id (str, optional): Document ID
                - filename (str, optional): Original filename
                - register (bool, optional): Register in registry

        Returns:
            List of HashRecord objects, one per document.

        Raises:
            ValueError: If documents list is empty or exceeds limit.
        """
        if not documents:
            raise ValueError("documents list must not be empty")

        max_size = self._config.batch_max_size
        if len(documents) > max_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum "
                f"of {max_size}"
            )

        start_time = time.monotonic()
        results: List[HashRecord] = []
        success_count = 0
        failure_count = 0

        for idx, doc in enumerate(documents):
            try:
                doc_bytes = doc.get("document_bytes", b"")
                doc_id = doc.get("document_id")
                fname = doc.get("filename")
                register = doc.get("register", True)

                record = self.compute_hash(
                    document_bytes=doc_bytes,
                    document_id=doc_id,
                    filename=fname,
                    register=register,
                )
                results.append(record)
                success_count += 1

            except Exception as e:
                failure_count += 1
                error_doc_id = doc.get("document_id", _generate_id())
                logger.warning(
                    "Batch hash failed for document[%d] "
                    "doc_id=%s: %s",
                    idx, str(error_doc_id)[:12], str(e),
                )
                error_record = HashRecord(
                    document_id=error_doc_id,
                    algorithm=HashAlgorithm.SHA256,
                    hash_value="0" * 64,
                    provenance_hash=_compute_provenance_hash({
                        "error": str(e),
                        "document_id": error_doc_id,
                    }),
                )
                results.append(error_record)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Batch hash computation complete: total=%d success=%d "
            "failure=%d time=%.1fms",
            len(documents), success_count, failure_count, elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Hash Chain Anchoring
    # ------------------------------------------------------------------

    def anchor_hash_chain(
        self,
        document_id: str,
        parent_document_id: str,
    ) -> Dict[str, Any]:
        """Link a document hash to a parent document hash for chain anchoring.

        Creates an explicit parent-child relationship between document
        hashes for provenance chain construction.

        Args:
            document_id: Child document ID.
            parent_document_id: Parent document ID.

        Returns:
            Dictionary with anchoring details.

        Raises:
            ValueError: If either document ID is not in the registry.
        """
        with self._lock:
            child_hash = self._document_hashes.get(document_id)
            parent_hash = self._document_hashes.get(parent_document_id)

        if child_hash is None:
            raise ValueError(
                f"Document not found in registry: {document_id}"
            )
        if parent_hash is None:
            raise ValueError(
                f"Parent document not found in registry: "
                f"{parent_document_id}"
            )

        # Update the child's parent reference
        with self._lock:
            entry = self._registry_sha256.get(child_hash)
            if entry:
                entry.parent_hash = parent_hash

        # Compute chain hash linking child to parent
        chain_data = f"{parent_hash}:{child_hash}"
        chain_hash = hashlib.sha256(
            chain_data.encode("utf-8"),
        ).hexdigest()

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="hash",
                action="compute_hash",
                entity_id=document_id,
                data={
                    "action": "anchor_chain",
                    "child_hash": child_hash,
                    "parent_hash": parent_hash,
                    "chain_hash": chain_hash,
                },
                metadata={
                    "child_document_id": document_id,
                    "parent_document_id": parent_document_id,
                },
            )

        logger.info(
            "Hash chain anchored: child=%s parent=%s chain=%s..%s",
            document_id[:12], parent_document_id[:12],
            chain_hash[:8], chain_hash[-8:],
        )

        return {
            "document_id": document_id,
            "parent_document_id": parent_document_id,
            "child_hash": child_hash,
            "parent_hash": parent_hash,
            "chain_hash": chain_hash,
            "status": "anchored",
        }

    # ------------------------------------------------------------------
    # Public API: Merkle Proof Generation
    # ------------------------------------------------------------------

    def generate_merkle_proof(
        self,
        document_hashes: List[str],
        target_index: int,
    ) -> Dict[str, Any]:
        """Generate a Merkle inclusion proof for a specific document.

        Creates the sibling hashes needed to reconstruct the Merkle
        root from a single leaf, enabling efficient verification
        that a document is included in an evidence package.

        Args:
            document_hashes: List of all document hashes in the tree.
            target_index: Index of the target document in the list.

        Returns:
            Dictionary containing:
                - target_hash (str): Hash of the target document
                - proof_hashes (list): Sibling hashes for proof path
                - proof_directions (list): 'left' or 'right' for each
                  sibling indicating its position
                - merkle_root (str): The Merkle root hash
                - provenance_hash (str): Provenance hash

        Raises:
            ValueError: If inputs are invalid.
        """
        if not document_hashes:
            raise ValueError("document_hashes must not be empty")
        if target_index < 0 or target_index >= len(document_hashes):
            raise ValueError(
                f"target_index {target_index} out of range "
                f"[0, {len(document_hashes)})"
            )

        start_time = time.monotonic()

        # Build leaf level
        current_level = list(document_hashes)
        proof_hashes: List[str] = []
        proof_directions: List[str] = []
        current_index = target_index

        while len(current_level) > 1:
            # Pad if odd
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            next_level: List[str] = []
            next_index = current_index // 2

            for i in range(0, len(current_level), 2):
                left_hash = current_level[i]
                right_hash = current_level[i + 1]
                combined = left_hash + right_hash
                parent_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()
                next_level.append(parent_hash)

                # Collect proof for target
                if i == current_index or i + 1 == current_index:
                    if current_index % 2 == 0:
                        # Target is left; sibling is right
                        proof_hashes.append(right_hash)
                        proof_directions.append("right")
                    else:
                        # Target is right; sibling is left
                        proof_hashes.append(left_hash)
                        proof_directions.append("left")

            current_level = next_level
            current_index = next_index

        merkle_root = current_level[0] if current_level else ""
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        provenance_data = {
            "target_index": target_index,
            "target_hash": document_hashes[target_index],
            "merkle_root": merkle_root,
            "proof_length": len(proof_hashes),
        }
        prov_hash = _compute_provenance_hash(provenance_data)

        logger.info(
            "Merkle proof generated: target_index=%d proof_len=%d "
            "root=%s..%s time=%.1fms",
            target_index, len(proof_hashes),
            merkle_root[:8], merkle_root[-8:],
            elapsed_ms,
        )

        return {
            "target_hash": document_hashes[target_index],
            "target_index": target_index,
            "proof_hashes": proof_hashes,
            "proof_directions": proof_directions,
            "merkle_root": merkle_root,
            "provenance_hash": prov_hash,
        }

    def verify_merkle_proof(
        self,
        target_hash: str,
        proof_hashes: List[str],
        proof_directions: List[str],
        expected_root: str,
    ) -> bool:
        """Verify a Merkle inclusion proof.

        Reconstructs the root hash from the target leaf and its
        proof path, then compares against the expected root.

        Args:
            target_hash: Hash of the target document.
            proof_hashes: Sibling hashes from the proof.
            proof_directions: Directions ('left' or 'right') for
                each sibling hash.
            expected_root: Expected Merkle root hash.

        Returns:
            True if the proof is valid.

        Raises:
            ValueError: If proof_hashes and proof_directions lengths
                do not match.
        """
        if len(proof_hashes) != len(proof_directions):
            raise ValueError(
                "proof_hashes and proof_directions must have "
                "the same length"
            )

        current = target_hash

        for sibling, direction in zip(proof_hashes, proof_directions):
            if direction == "left":
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha256(
                combined.encode("utf-8"),
            ).hexdigest()

        match = hmac.compare_digest(current, expected_root)

        logger.debug(
            "Merkle proof verification: match=%s", match,
        )

        return match

    # ------------------------------------------------------------------
    # Public API: Registry Cleanup
    # ------------------------------------------------------------------

    def cleanup_expired_entries(self) -> Dict[str, Any]:
        """Remove expired entries from the hash registry.

        Entries older than the configured registry_ttl_days are
        removed to manage memory usage while respecting EUDR
        Article 14 five-year retention requirements.

        Returns:
            Dictionary with cleanup statistics.
        """
        now = _utcnow()
        removed_count = 0
        retained_count = 0

        with self._lock:
            expired_sha256_keys: List[str] = []
            expired_sha512_keys: List[str] = []

            for key, entry in self._registry_sha256.items():
                if entry.expires_at <= now:
                    expired_sha256_keys.append(key)
                    expired_sha512_keys.append(entry.hash_sha512)
                else:
                    retained_count += 1

            for key in expired_sha256_keys:
                del self._registry_sha256[key]
                removed_count += 1

            for key in expired_sha512_keys:
                self._registry_sha512.pop(key, None)

        logger.info(
            "Registry cleanup: removed=%d retained=%d",
            removed_count, retained_count,
        )

        return {
            "removed": removed_count,
            "retained": retained_count,
            "cleanup_at": now.isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API: Export Registry
    # ------------------------------------------------------------------

    def export_registry(
        self,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Export the hash registry as a list of dictionaries.

        Args:
            include_expired: Whether to include expired entries.

        Returns:
            List of registry entry dictionaries.
        """
        now = _utcnow()

        with self._lock:
            entries = list(self._registry_sha256.values())

        if not include_expired:
            entries = [e for e in entries if e.expires_at > now]

        return [e.to_dict() for e in entries]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            registry_count = len(self._registry_sha256)
            operation_count = len(self._operation_history)
        return (
            f"HashIntegrityValidator("
            f"registry={registry_count}, "
            f"operations={operation_count})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "HashIntegrityValidator",
]
