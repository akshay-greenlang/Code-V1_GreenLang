# -*- coding: utf-8 -*-
"""
Deterministic Artifact Hashing Engine - AGENT-FOUND-008: Reproducibility Agent

Provides deterministic hashing of arbitrary data artifacts with float
normalization, sorted-key serialization, incremental hashing for large
datasets, and a TTL-based hash cache.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Float values are normalized to configurable decimal places
    - Dict keys are sorted for consistent serialization
    - Sets are converted to sorted lists before hashing
    - Decimal values are converted to string representation

Example:
    >>> from greenlang.reproducibility.artifact_hasher import ArtifactHasher
    >>> from greenlang.reproducibility.config import ReproducibilityConfig
    >>> hasher = ArtifactHasher(ReproducibilityConfig())
    >>> h = hasher.compute_hash({"emissions": 100.5, "scope": 1})
    >>> print(h)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from greenlang.reproducibility.config import ReproducibilityConfig
from greenlang.reproducibility.models import ArtifactHash
from greenlang.reproducibility.metrics import (
    record_hash_computation,
    record_hash_mismatch,
    record_cache_hit,
    record_cache_miss,
)

logger = logging.getLogger(__name__)


class ArtifactHasher:
    """Deterministic artifact hashing engine.

    Computes SHA-256 hashes of arbitrary data with float normalization
    and sorted-key serialization for consistent, reproducible results.

    Supports incremental hashing for large datasets, hash verification,
    batch hashing, and a TTL-based in-memory cache.

    Attributes:
        _config: Reproducibility configuration.
        _hash_cache: In-memory cache mapping cache keys to (hash, timestamp).
        _hash_history: In-memory history of computed hashes by artifact ID.

    Example:
        >>> hasher = ArtifactHasher(ReproducibilityConfig())
        >>> h = hasher.compute_hash({"value": 42.0})
        >>> match, actual = hasher.verify_hash({"value": 42.0}, h)
        >>> assert match is True
    """

    def __init__(self, config: ReproducibilityConfig) -> None:
        """Initialize ArtifactHasher.

        Args:
            config: Reproducibility configuration instance.
        """
        self._config = config
        self._hash_cache: Dict[str, Tuple[str, float]] = {}
        self._hash_history: Dict[str, List[ArtifactHash]] = {}
        logger.info(
            "ArtifactHasher initialized: algorithm=%s, decimals=%d, cache_ttl=%ds",
            config.hash_algorithm,
            config.float_normalization_decimals,
            config.hash_cache_ttl_seconds,
        )

    def compute_hash(
        self,
        data: Any,
        algorithm: str = "",
    ) -> str:
        """Compute deterministic hash of arbitrary data.

        Normalizes the data (float rounding, key sorting, set ordering)
        and produces a SHA-256 hex digest.

        Args:
            data: Data to hash.
            algorithm: Hash algorithm override (default uses config).

        Returns:
            Hex-encoded hash string.
        """
        algo = algorithm or self._config.hash_algorithm
        normalized = self._normalize_value(
            data, self._config.float_normalization_decimals,
        )
        serialized = self._serialize_deterministic(normalized)

        # Check cache
        cache_key = hashlib.md5(serialized).hexdigest()
        cached = self._get_cached(cache_key)
        if cached is not None:
            record_cache_hit()
            return cached

        record_cache_miss()

        if algo == "sha256":
            result = hashlib.sha256(serialized).hexdigest()
        elif algo == "sha512":
            result = hashlib.sha512(serialized).hexdigest()
        elif algo == "md5":
            result = hashlib.md5(serialized).hexdigest()
        else:
            result = hashlib.sha256(serialized).hexdigest()

        # Store in cache
        self._set_cached(cache_key, result)
        record_hash_computation("generic")

        return result

    def _normalize_value(self, value: Any, decimals: int = 15) -> Any:
        """Normalize a value for deterministic hashing.

        Handles Decimal, float, dict, list, set, tuple, datetime, bool,
        int, and str types recursively.

        Args:
            value: Value to normalize.
            decimals: Number of decimal places for float rounding.

        Returns:
            Normalized representation suitable for deterministic serialization.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            return value

        if isinstance(value, Decimal):
            return str(value.normalize())

        if isinstance(value, float):
            return round(value, decimals)

        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()

        if isinstance(value, dict):
            return {
                str(k): self._normalize_value(v, decimals)
                for k, v in sorted(value.items())
            }

        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item, decimals) for item in value]

        if isinstance(value, set):
            normalized_items = [self._normalize_value(item, decimals) for item in value]
            return sorted(normalized_items, key=lambda x: str(x))

        if isinstance(value, bytes):
            return value.hex()

        # Fallback: convert to string
        return str(value)

    def _serialize_deterministic(self, data: Any) -> bytes:
        """Serialize data deterministically to bytes.

        Uses json.dumps with sort_keys=True and default=str for
        consistent byte representation.

        Args:
            data: Normalized data to serialize.

        Returns:
            UTF-8 encoded bytes of the JSON representation.
        """
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
        return json_str.encode("utf-8")

    def compute_incremental_hash(self, chunks: List[Any]) -> str:
        """Compute hash incrementally over a list of data chunks.

        Useful for large datasets that should not be loaded into
        memory all at once.

        Args:
            chunks: List of data chunks to hash incrementally.

        Returns:
            Hex-encoded SHA-256 hash of the combined chunks.
        """
        hasher = hashlib.sha256()
        for chunk in chunks:
            normalized = self._normalize_value(
                chunk, self._config.float_normalization_decimals,
            )
            serialized = self._serialize_deterministic(normalized)
            hasher.update(serialized)
        record_hash_computation("incremental")
        return hasher.hexdigest()

    def verify_hash(
        self,
        data: Any,
        expected_hash: str,
        algorithm: str = "",
    ) -> Tuple[bool, str]:
        """Verify data matches an expected hash.

        Args:
            data: Data to hash and compare.
            expected_hash: Expected hash value.
            algorithm: Hash algorithm override.

        Returns:
            Tuple of (match: bool, actual_hash: str).
        """
        actual_hash = self.compute_hash(data, algorithm=algorithm)
        match = actual_hash == expected_hash
        if not match:
            record_hash_mismatch()
            logger.warning(
                "Hash mismatch: expected=%s, actual=%s",
                expected_hash[:16], actual_hash[:16],
            )
        return match, actual_hash

    def batch_hash(self, items: Dict[str, Any]) -> Dict[str, str]:
        """Compute hashes for multiple items in a batch.

        Args:
            items: Dictionary mapping item names to data.

        Returns:
            Dictionary mapping item names to computed hashes.
        """
        results: Dict[str, str] = {}
        for name, data in items.items():
            results[name] = self.compute_hash(data)
        record_hash_computation("batch")
        return results

    def store_hash(
        self,
        artifact_id: str,
        artifact_type: str,
        data_hash: str,
        provenance_hash: str = "",
    ) -> ArtifactHash:
        """Store a hash record for an artifact.

        Args:
            artifact_id: Unique artifact identifier.
            artifact_type: Type of the artifact.
            data_hash: Computed hash of the artifact.
            provenance_hash: Provenance chain hash.

        Returns:
            ArtifactHash record.
        """
        record = ArtifactHash(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            data_hash=data_hash,
            algorithm=self._config.hash_algorithm,
            normalization_applied=True,
            provenance_hash=provenance_hash,
        )

        if artifact_id not in self._hash_history:
            self._hash_history[artifact_id] = []
        self._hash_history[artifact_id].append(record)

        logger.debug(
            "Stored hash for artifact %s: %s", artifact_id, data_hash[:16],
        )
        return record

    def get_hash_history(self, artifact_id: str) -> List[ArtifactHash]:
        """Get the hash history for an artifact.

        Args:
            artifact_id: Unique artifact identifier.

        Returns:
            List of ArtifactHash records, newest first.
        """
        history = self._hash_history.get(artifact_id, [])
        return sorted(history, key=lambda h: h.computed_at, reverse=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Get a hash from the cache if still valid.

        Args:
            cache_key: Cache key (MD5 of serialized data).

        Returns:
            Cached hash or None if expired/missing.
        """
        entry = self._hash_cache.get(cache_key)
        if entry is None:
            return None

        cached_hash, cached_time = entry
        if time.time() - cached_time > self._config.hash_cache_ttl_seconds:
            del self._hash_cache[cache_key]
            return None

        return cached_hash

    def _set_cached(self, cache_key: str, hash_value: str) -> None:
        """Store a hash in the cache.

        Args:
            cache_key: Cache key (MD5 of serialized data).
            hash_value: Hash value to cache.
        """
        self._hash_cache[cache_key] = (hash_value, time.time())

    def clear_cache(self) -> int:
        """Clear the hash cache.

        Returns:
            Number of entries removed.
        """
        count = len(self._hash_cache)
        self._hash_cache.clear()
        logger.info("Hash cache cleared: %d entries removed", count)
        return count


__all__ = [
    "ArtifactHasher",
]
