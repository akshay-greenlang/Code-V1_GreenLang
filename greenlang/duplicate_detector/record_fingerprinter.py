# -*- coding: utf-8 -*-
"""
Record Fingerprinter Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Generates deterministic fingerprints for records using SHA-256, SimHash,
and MinHash algorithms. Supports type-aware field normalization for
strings, numbers, dates, booleans, and categorical fields.

Zero-Hallucination Guarantees:
    - All fingerprints use deterministic hash algorithms
    - Field normalization uses rule-based transforms only
    - SimHash uses character n-gram frequency vectors
    - MinHash uses universal hash family for signature generation
    - No ML/LLM calls in fingerprinting path
    - Provenance recorded for every fingerprint operation

Supported Algorithms:
    SHA-256: Cryptographic hash for exact-match deduplication
    SimHash: Locality-sensitive hash (64-bit) for near-duplicate detection
    MinHash: Signature arrays for Jaccard similarity estimation

Example:
    >>> from greenlang.duplicate_detector.record_fingerprinter import RecordFingerprinter
    >>> engine = RecordFingerprinter()
    >>> fp = engine.fingerprint_record(
    ...     record={"name": "Alice", "email": "alice@example.com"},
    ...     field_set=["name", "email"],
    ...     algorithm=FingerprintAlgorithm.SHA256,
    ... )
    >>> print(fp.fingerprint_hash)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.duplicate_detector.models import (
    FieldType,
    FingerprintAlgorithm,
    RecordFingerprint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RecordFingerprinter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "FP") -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a fingerprinting operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Large prime for MinHash universal hashing
_MINHASH_PRIME: int = 2_147_483_647  # 2^31 - 1 (Mersenne prime)
_MINHASH_MAX_HASH: int = 2**32 - 1

# SimHash default n-gram size for shingle generation
_SIMHASH_NGRAM_SIZE: int = 3

# Whitespace collapse pattern
_WHITESPACE_RE = re.compile(r"\s+")

# Phone normalization: keep only digits and leading +
_PHONE_RE = re.compile(r"[^\d+]")

# Email normalization
_EMAIL_RE = re.compile(r"^([^@]+)@(.+)$")


# =============================================================================
# RecordFingerprinter
# =============================================================================


class RecordFingerprinter:
    """Record fingerprinting engine for duplicate detection.

    Generates deterministic fingerprints for records using SHA-256,
    SimHash, or MinHash algorithms. Provides type-aware field
    normalization to ensure consistent fingerprints across variant
    representations of the same data.

    This engine follows GreenLang's zero-hallucination principle by
    using only deterministic hash algorithms and rule-based field
    normalization. No ML/LLM calls are made.

    Attributes:
        _stats_lock: Threading lock for stats updates.
        _invocations: Total invocation count.
        _successes: Total successful invocations.
        _failures: Total failed invocations.
        _total_duration_ms: Cumulative processing time.

    Example:
        >>> engine = RecordFingerprinter()
        >>> fp = engine.fingerprint_record(
        ...     {"name": "Alice", "age": 30},
        ...     ["name", "age"],
        ...     FingerprintAlgorithm.SHA256,
        ... )
        >>> assert fp.fingerprint_hash != ""
    """

    def __init__(self) -> None:
        """Initialize RecordFingerprinter with empty statistics."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None

        # Pre-generate hash coefficients for MinHash
        self._minhash_a: List[int] = []
        self._minhash_b: List[int] = []
        self._minhash_initialized: bool = False

        logger.info("RecordFingerprinter initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fingerprint_record(
        self,
        record: Dict[str, Any],
        field_set: List[str],
        algorithm: FingerprintAlgorithm,
        record_id: Optional[str] = None,
        field_types: Optional[Dict[str, FieldType]] = None,
        num_hashes: int = 128,
        hash_bits: int = 64,
    ) -> RecordFingerprint:
        """Generate a fingerprint for a single record.

        Normalizes the specified fields, concatenates them in sorted
        order, and computes the fingerprint using the chosen algorithm.

        Args:
            record: Input record as a dictionary.
            field_set: List of field names to include in fingerprint.
            algorithm: Fingerprinting algorithm to use.
            record_id: Optional record identifier (auto-generated if None).
            field_types: Optional mapping of field names to FieldType.
            num_hashes: Number of hash functions for MinHash.
            hash_bits: Bit width for SimHash.

        Returns:
            RecordFingerprint with computed hash and metadata.

        Raises:
            ValueError: If record or field_set is empty.
        """
        start_time = time.monotonic()
        rid = record_id or str(record.get("id", _generate_id("REC")))

        try:
            if not record:
                raise ValueError("record must not be empty")
            if not field_set:
                raise ValueError("field_set must not be empty")

            # Normalize fields in sorted order for determinism
            normalized = self._normalize_fields(
                record, sorted(field_set), field_types or {},
            )

            # Compute fingerprint based on algorithm
            fp_hash = self._compute_fingerprint(
                normalized, algorithm, num_hashes, hash_bits,
            )

            provenance = _compute_provenance(
                "fingerprint_record", f"{rid}:{algorithm.value}",
            )

            result = RecordFingerprint(
                record_id=str(rid),
                fingerprint_hash=fp_hash,
                algorithm=algorithm,
                field_set=sorted(field_set),
                normalized_fields=True,
                provenance_hash=provenance,
            )

            self._record_success(time.monotonic() - start_time)
            logger.debug(
                "Fingerprinted record %s with %s: %s",
                rid, algorithm.value, fp_hash[:16],
            )
            return result

        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error("Failed to fingerprint record %s: %s", rid, e)
            raise

    def fingerprint_batch(
        self,
        records: List[Dict[str, Any]],
        field_set: List[str],
        algorithm: FingerprintAlgorithm,
        id_field: str = "id",
        field_types: Optional[Dict[str, FieldType]] = None,
        num_hashes: int = 128,
        hash_bits: int = 64,
    ) -> List[RecordFingerprint]:
        """Generate fingerprints for a batch of records.

        Args:
            records: List of input records.
            field_set: List of field names to include.
            algorithm: Fingerprinting algorithm to use.
            id_field: Field name for record identifier.
            field_types: Optional mapping of field names to FieldType.
            num_hashes: Number of hash functions for MinHash.
            hash_bits: Bit width for SimHash.

        Returns:
            List of RecordFingerprint instances.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError("records list must not be empty")

        logger.info(
            "Fingerprinting batch of %d records with %s",
            len(records), algorithm.value,
        )

        results: List[RecordFingerprint] = []
        for idx, record in enumerate(records):
            rid = str(record.get(id_field, f"rec-{idx}"))
            fp = self.fingerprint_record(
                record=record,
                field_set=field_set,
                algorithm=algorithm,
                record_id=rid,
                field_types=field_types,
                num_hashes=num_hashes,
                hash_bits=hash_bits,
            )
            results.append(fp)

        logger.info(
            "Batch fingerprinting complete: %d records processed",
            len(results),
        )
        return results

    def normalize_field(
        self,
        value: Any,
        field_type: FieldType,
    ) -> str:
        """Normalize a single field value based on its type.

        Type-aware normalization rules:
        - STRING: lowercase, strip, collapse whitespace
        - NUMERIC: round to 6 decimals
        - DATE: ISO 8601 format
        - BOOLEAN: "true" or "false"
        - CATEGORICAL: lowercase, strip

        Args:
            value: The field value to normalize.
            field_type: The declared type of the field.

        Returns:
            Normalized string representation.
        """
        if value is None:
            return ""
        return self._normalize_by_type(value, field_type)

    def compute_sha256(self, normalized: str) -> str:
        """Compute SHA-256 hash of a normalized string.

        Args:
            normalized: Normalized string to hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_simhash(
        self,
        normalized: str,
        hash_bits: int = 64,
    ) -> int:
        """Compute SimHash of a normalized string using character n-grams.

        SimHash is a locality-sensitive hash that maps similar strings
        to similar hash values. It works by:
        1. Extracting character n-grams (shingles)
        2. Hashing each shingle to a fixed-width integer
        3. Building a weighted bit vector from all shingle hashes
        4. Converting the vector to a binary hash

        Args:
            normalized: Normalized string to hash.
            hash_bits: Number of bits in the output hash.

        Returns:
            SimHash as an integer.
        """
        if not normalized:
            return 0

        shingles = self._generate_ngrams(normalized, _SIMHASH_NGRAM_SIZE)
        if not shingles:
            return 0

        bit_vector: List[int] = [0] * hash_bits

        for shingle in shingles:
            shingle_hash = self._hash_to_int(shingle, hash_bits)
            for i in range(hash_bits):
                bit = (shingle_hash >> i) & 1
                if bit == 1:
                    bit_vector[i] += 1
                else:
                    bit_vector[i] -= 1

        simhash_value = 0
        for i in range(hash_bits):
            if bit_vector[i] > 0:
                simhash_value |= (1 << i)

        return simhash_value

    def compute_minhash(
        self,
        normalized: str,
        num_hashes: int = 128,
    ) -> List[int]:
        """Compute MinHash signature of a normalized string.

        MinHash generates a compact signature that preserves Jaccard
        similarity. It uses a family of universal hash functions
        h(x) = (a * x + b) mod p, where p is a large prime.

        Args:
            normalized: Normalized string to hash.
            num_hashes: Number of hash functions (signature length).

        Returns:
            List of MinHash signature values.
        """
        if not normalized:
            return [_MINHASH_MAX_HASH] * num_hashes

        self._ensure_minhash_coefficients(num_hashes)

        shingles = self._generate_ngrams(normalized, _SIMHASH_NGRAM_SIZE)
        if not shingles:
            return [_MINHASH_MAX_HASH] * num_hashes

        shingle_hashes: List[int] = []
        for shingle in shingles:
            h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
            shingle_hashes.append(h % _MINHASH_PRIME)

        signature: List[int] = [_MINHASH_MAX_HASH] * num_hashes
        for i in range(num_hashes):
            a = self._minhash_a[i]
            b = self._minhash_b[i]
            for sh in shingle_hashes:
                hash_val = (a * sh + b) % _MINHASH_PRIME
                if hash_val < signature[i]:
                    signature[i] = hash_val

        return signature

    def estimate_jaccard(
        self,
        sig_a: List[int],
        sig_b: List[int],
    ) -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        Args:
            sig_a: First MinHash signature.
            sig_b: Second MinHash signature.

        Returns:
            Estimated Jaccard similarity (0.0 to 1.0).

        Raises:
            ValueError: If signatures have different lengths.
        """
        if len(sig_a) != len(sig_b):
            raise ValueError(
                f"Signature lengths must match: {len(sig_a)} != {len(sig_b)}"
            )
        if not sig_a:
            return 0.0

        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics.

        Returns:
            Dictionary with invocation counts and timing.
        """
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations
            return {
                "engine_name": "RecordFingerprinter",
                "invocations": self._invocations,
                "successes": self._successes,
                "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": (
                    self._last_invoked_at.isoformat()
                    if self._last_invoked_at else None
                ),
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # ------------------------------------------------------------------
    # Private methods - Normalization
    # ------------------------------------------------------------------

    def _normalize_fields(
        self,
        record: Dict[str, Any],
        field_set: List[str],
        field_types: Dict[str, FieldType],
    ) -> str:
        """Normalize and concatenate fields in sorted order.

        Args:
            record: Input record dictionary.
            field_set: Sorted list of field names.
            field_types: Mapping of field names to FieldType.

        Returns:
            Pipe-delimited concatenation of normalized field values.
        """
        parts: List[str] = []
        for field_name in field_set:
            value = record.get(field_name)
            ftype = field_types.get(field_name, FieldType.STRING)
            normalized = self._normalize_by_type(value, ftype)
            parts.append(normalized)
        return "|".join(parts)

    def _normalize_by_type(self, value: Any, field_type: FieldType) -> str:
        """Normalize a single value based on its field type.

        Args:
            value: The value to normalize.
            field_type: The declared field type.

        Returns:
            Normalized string representation.
        """
        if value is None:
            return ""

        if field_type == FieldType.STRING:
            return self._normalize_string(value)
        elif field_type == FieldType.NUMERIC:
            return self._normalize_numeric(value)
        elif field_type == FieldType.DATE:
            return self._normalize_date(value)
        elif field_type == FieldType.BOOLEAN:
            return self._normalize_boolean(value)
        elif field_type == FieldType.CATEGORICAL:
            return self._normalize_categorical(value)
        else:
            return self._normalize_string(value)

    def _normalize_string(self, value: Any) -> str:
        """Normalize a string value: lowercase, strip, collapse whitespace."""
        s = str(value).strip().lower()
        return _WHITESPACE_RE.sub(" ", s)

    def _normalize_numeric(self, value: Any) -> str:
        """Normalize a numeric value: round to 6 decimal places."""
        try:
            return f"{float(value):.6f}"
        except (ValueError, TypeError):
            return str(value).strip()

    def _normalize_boolean(self, value: Any) -> str:
        """Normalize a boolean value to 'true' or 'false'."""
        if isinstance(value, bool):
            return "true" if value else "false"
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return "true"
        return "false"

    def _normalize_date(self, value: Any) -> str:
        """Normalize a date value to ISO 8601 (YYYY-MM-DD) format."""
        s = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S",
                     "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y",
                     "%Y-%m-%dT%H:%M:%S.%f",
                     "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(s[:26], fmt)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                continue
        return s.lower()

    def _normalize_categorical(self, value: Any) -> str:
        """Normalize a categorical value: lowercase, strip."""
        return str(value).strip().lower()

    # ------------------------------------------------------------------
    # Private methods - Hash computation
    # ------------------------------------------------------------------

    def _compute_fingerprint(
        self,
        normalized: str,
        algorithm: FingerprintAlgorithm,
        num_hashes: int,
        hash_bits: int,
    ) -> str:
        """Compute fingerprint hash string using the specified algorithm.

        Args:
            normalized: Normalized field string.
            algorithm: Algorithm to use.
            num_hashes: Number of MinHash functions.
            hash_bits: SimHash bit width.

        Returns:
            Hex-encoded hash string.
        """
        if algorithm == FingerprintAlgorithm.SHA256:
            return self.compute_sha256(normalized)

        elif algorithm == FingerprintAlgorithm.SIMHASH:
            simhash_val = self.compute_simhash(normalized, hash_bits)
            return format(simhash_val, f"0{hash_bits // 4}x")

        elif algorithm == FingerprintAlgorithm.MINHASH:
            signature = self.compute_minhash(normalized, num_hashes)
            sig_bytes = b"".join(
                struct.pack("<I", v & 0xFFFFFFFF) for v in signature[:8]
            )
            return hashlib.sha256(sig_bytes).hexdigest()

        return self.compute_sha256(normalized)

    def _generate_ngrams(self, text: str, n: int) -> List[str]:
        """Generate character n-grams from text.

        Args:
            text: Input text string.
            n: N-gram size.

        Returns:
            List of n-gram strings.
        """
        if len(text) < n:
            return [text] if text else []
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _hash_to_int(self, text: str, bits: int) -> int:
        """Hash a string to an integer of specified bit width.

        Args:
            text: Input text.
            bits: Output bit width.

        Returns:
            Integer hash value.
        """
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        full_int = int(digest, 16)
        mask = (1 << bits) - 1
        return full_int & mask

    def _ensure_minhash_coefficients(self, num_hashes: int) -> None:
        """Initialize MinHash hash function coefficients if needed.

        Uses deterministic seed-based generation for reproducibility.

        Args:
            num_hashes: Number of hash functions needed.
        """
        if self._minhash_initialized and len(self._minhash_a) >= num_hashes:
            return

        import random
        rng = random.Random(42)

        self._minhash_a = []
        self._minhash_b = []
        for _ in range(num_hashes):
            a = rng.randint(1, _MINHASH_PRIME - 1)
            b = rng.randint(0, _MINHASH_PRIME - 1)
            self._minhash_a.append(a)
            self._minhash_b.append(b)

        self._minhash_initialized = True

    # ------------------------------------------------------------------
    # Private methods - Stats tracking
    # ------------------------------------------------------------------

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        elapsed_ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += elapsed_ms
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        elapsed_ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += elapsed_ms
            self._last_invoked_at = _utcnow()
