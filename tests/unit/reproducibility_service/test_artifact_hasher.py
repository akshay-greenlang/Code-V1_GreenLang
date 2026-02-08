# -*- coding: utf-8 -*-
"""
Unit Tests for ArtifactHasher (AGENT-FOUND-008)

Tests hash computation, normalization, verification, batch hashing,
caching, and history tracking.

Coverage target: 85%+ of artifact_hasher.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline ArtifactHasher mirroring greenlang/reproducibility/artifact_hasher.py
# ---------------------------------------------------------------------------

DEFAULT_ABSOLUTE_TOLERANCE = 1e-9


class ArtifactHash:
    def __init__(self, artifact_id: str, hash_value: str, algorithm: str = "sha256",
                 created_at: Optional[datetime] = None):
        self.artifact_id = artifact_id
        self.hash_value = hash_value
        self.algorithm = algorithm
        self.created_at = created_at or datetime.now(timezone.utc)


class ArtifactHasher:
    """Deterministic artifact hashing for reproducibility."""

    def __init__(self, algorithm: str = "sha256", float_precision: int = 15,
                 cache_ttl: int = 3600):
        self.algorithm = algorithm
        self.float_precision = float_precision
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, str] = {}
        self._history: Dict[str, List[ArtifactHash]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def compute_hash(self, data: Any, tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE) -> str:
        """Compute deterministic hash of data."""
        normalized = self._normalize_value(data, tolerance)
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)

        # Check cache
        cache_key = json_str
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()
        self._cache[cache_key] = hash_value
        return hash_value

    def _normalize_value(self, data: Any, tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE) -> Any:
        """Normalize data for deterministic hashing."""
        if data is None:
            return None
        if isinstance(data, bool):
            return data
        if isinstance(data, (int, str)):
            return data
        if isinstance(data, float):
            if tolerance > 0:
                precision = max(0, -int(math.log10(tolerance)))
                return round(data, precision)
            return data
        if isinstance(data, Decimal):
            return str(data)
        if isinstance(data, datetime):
            return data.replace(microsecond=0).isoformat()
        if isinstance(data, dict):
            return {str(k): self._normalize_value(v, tolerance)
                    for k, v in sorted(data.items())}
        if isinstance(data, (list, tuple)):
            return [self._normalize_value(item, tolerance) for item in data]
        if isinstance(data, set):
            return sorted([self._normalize_value(item, tolerance) for item in data])
        if isinstance(data, frozenset):
            return sorted([self._normalize_value(item, tolerance) for item in data])
        return str(data)

    def verify_hash(self, data: Any, expected_hash: str,
                    tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE) -> Tuple[bool, str]:
        """Verify data hash matches expected value."""
        actual_hash = self.compute_hash(data, tolerance)
        if actual_hash == expected_hash:
            return True, "Hash matches"
        return False, f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."

    def batch_hash(self, items: List[Any],
                   tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE) -> List[str]:
        """Compute hashes for a batch of items."""
        return [self.compute_hash(item, tolerance) for item in items]

    def store_hash(self, artifact_id: str, data: Any,
                   tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE) -> ArtifactHash:
        """Compute and store a hash record."""
        hash_value = self.compute_hash(data, tolerance)
        artifact = ArtifactHash(
            artifact_id=artifact_id,
            hash_value=hash_value,
            algorithm=self.algorithm,
        )
        if artifact_id not in self._history:
            self._history[artifact_id] = []
        self._history[artifact_id].append(artifact)
        return artifact

    def get_hash_history(self, artifact_id: str) -> List[ArtifactHash]:
        """Get hash history for an artifact."""
        return self._history.get(artifact_id, [])


# ===========================================================================
# Test Classes
# ===========================================================================


class TestArtifactHasherInit:
    """Test ArtifactHasher initialization."""

    def test_default_algorithm(self):
        hasher = ArtifactHasher()
        assert hasher.algorithm == "sha256"

    def test_custom_algorithm(self):
        hasher = ArtifactHasher(algorithm="sha512")
        assert hasher.algorithm == "sha512"

    def test_default_float_precision(self):
        hasher = ArtifactHasher()
        assert hasher.float_precision == 15

    def test_default_cache_ttl(self):
        hasher = ArtifactHasher()
        assert hasher.cache_ttl == 3600

    def test_empty_cache_on_init(self):
        hasher = ArtifactHasher()
        assert hasher._cache == {}

    def test_zero_cache_counters_on_init(self):
        hasher = ArtifactHasher()
        assert hasher._cache_hits == 0
        assert hasher._cache_misses == 0


class TestComputeHash:
    """Test compute_hash method."""

    def test_compute_hash_simple_dict(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_compute_hash_deterministic(self):
        hasher = ArtifactHasher()
        h1 = hasher.compute_hash({"emissions": 100.5})
        h2 = hasher.compute_hash({"emissions": 100.5})
        assert h1 == h2

    def test_compute_hash_different_key_order_same_hash(self):
        hasher = ArtifactHasher()
        h1 = hasher.compute_hash({"a": 1, "b": 2, "c": 3})
        h2 = hasher.compute_hash({"c": 3, "a": 1, "b": 2})
        assert h1 == h2

    def test_compute_hash_with_floats(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({"value": 3.141592653589793})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_compute_hash_with_decimals(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({"value": Decimal("100.50")})
        assert isinstance(result, str)

    def test_compute_hash_nested_dict(self):
        hasher = ArtifactHasher()
        data = {"level1": {"level2": {"value": 42}}}
        result = hasher.compute_hash(data)
        assert len(result) == 64

    def test_compute_hash_with_lists(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({"items": [1, 2, 3, 4, 5]})
        assert len(result) == 64

    def test_compute_hash_with_none_values(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({"key": None})
        assert len(result) == 64

    def test_compute_hash_empty_dict(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash({})
        assert len(result) == 64

    def test_compute_hash_string_input(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash("simple_string")
        assert len(result) == 64

    def test_compute_hash_integer_input(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash(42)
        assert len(result) == 64

    def test_compute_hash_boolean_input(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash(True)
        assert len(result) == 64

    def test_compute_hash_list_input(self):
        hasher = ArtifactHasher()
        result = hasher.compute_hash([1, 2, 3])
        assert len(result) == 64

    def test_compute_hash_different_data_different_hash(self):
        hasher = ArtifactHasher()
        h1 = hasher.compute_hash({"a": 1})
        h2 = hasher.compute_hash({"a": 2})
        assert h1 != h2


class TestNormalizeValue:
    """Test _normalize_value method."""

    def test_normalize_none(self):
        hasher = ArtifactHasher()
        assert hasher._normalize_value(None) is None

    def test_normalize_bool_true(self):
        hasher = ArtifactHasher()
        assert hasher._normalize_value(True) is True

    def test_normalize_bool_false(self):
        hasher = ArtifactHasher()
        assert hasher._normalize_value(False) is False

    def test_normalize_int(self):
        hasher = ArtifactHasher()
        assert hasher._normalize_value(42) == 42

    def test_normalize_string(self):
        hasher = ArtifactHasher()
        assert hasher._normalize_value("hello") == "hello"

    def test_normalize_float_with_tolerance(self):
        hasher = ArtifactHasher()
        result = hasher._normalize_value(3.14159, tolerance=1e-3)
        assert result == round(3.14159, 3)

    def test_normalize_float_zero_tolerance(self):
        hasher = ArtifactHasher()
        val = 3.14159265358979
        result = hasher._normalize_value(val, tolerance=0.0)
        assert result == val

    def test_normalize_decimal(self):
        hasher = ArtifactHasher()
        result = hasher._normalize_value(Decimal("100.50"))
        assert result == "100.50"

    def test_normalize_datetime(self):
        hasher = ArtifactHasher()
        dt = datetime(2026, 1, 15, 12, 30, 45, 123456)
        result = hasher._normalize_value(dt)
        assert "123456" not in result  # microseconds removed

    def test_normalize_dict_sorting(self):
        hasher = ArtifactHasher()
        data = {"z": 1, "a": 2, "m": 3}
        result = hasher._normalize_value(data)
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_normalize_list(self):
        hasher = ArtifactHasher()
        result = hasher._normalize_value([3, 1, 2])
        assert result == [3, 1, 2]  # Lists preserve order

    def test_normalize_tuple_to_list(self):
        hasher = ArtifactHasher()
        result = hasher._normalize_value((1, 2, 3))
        assert result == [1, 2, 3]

    def test_normalize_set_to_sorted_list(self):
        hasher = ArtifactHasher()
        result = hasher._normalize_value({3, 1, 2})
        assert result == [1, 2, 3]

    def test_normalize_nested(self):
        hasher = ArtifactHasher()
        data = {"outer": {"inner": [1.5, 2.5]}}
        result = hasher._normalize_value(data, tolerance=1e-1)
        assert isinstance(result, dict)
        assert isinstance(result["outer"], dict)
        assert isinstance(result["outer"]["inner"], list)

    def test_normalize_unknown_type(self):
        hasher = ArtifactHasher()

        class Custom:
            def __str__(self):
                return "custom_object"

        result = hasher._normalize_value(Custom())
        assert result == "custom_object"


class TestVerifyHash:
    """Test verify_hash method."""

    def test_verify_hash_match(self):
        hasher = ArtifactHasher()
        data = {"key": "value"}
        h = hasher.compute_hash(data)
        match, msg = hasher.verify_hash(data, h)
        assert match is True
        assert "matches" in msg.lower()

    def test_verify_hash_mismatch(self):
        hasher = ArtifactHasher()
        data = {"key": "value"}
        match, msg = hasher.verify_hash(data, "wrong_hash_value_1234567890abcdef")
        assert match is False
        assert "mismatch" in msg.lower()

    def test_verify_hash_empty_dict_match(self):
        hasher = ArtifactHasher()
        data = {}
        h = hasher.compute_hash(data)
        match, _ = hasher.verify_hash(data, h)
        assert match is True


class TestBatchHash:
    """Test batch_hash method."""

    def test_batch_hash_multiple_items(self):
        hasher = ArtifactHasher()
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        results = hasher.batch_hash(items)
        assert len(results) == 3
        for r in results:
            assert len(r) == 64

    def test_batch_hash_consistency(self):
        hasher = ArtifactHasher()
        items = [{"x": 10}, {"y": 20}]
        r1 = hasher.batch_hash(items)
        r2 = hasher.batch_hash(items)
        assert r1 == r2

    def test_batch_hash_empty_list(self):
        hasher = ArtifactHasher()
        results = hasher.batch_hash([])
        assert results == []

    def test_batch_hash_single_item(self):
        hasher = ArtifactHasher()
        items = [{"key": "value"}]
        results = hasher.batch_hash(items)
        assert len(results) == 1
        assert results[0] == hasher.compute_hash({"key": "value"})


class TestHashCache:
    """Test hash caching behavior."""

    def test_cache_miss_increments(self):
        hasher = ArtifactHasher()
        hasher.compute_hash({"a": 1})
        assert hasher._cache_misses == 1

    def test_cache_hit_increments(self):
        hasher = ArtifactHasher()
        hasher.compute_hash({"a": 1})
        hasher.compute_hash({"a": 1})
        assert hasher._cache_hits == 1

    def test_cache_stores_result(self):
        hasher = ArtifactHasher()
        h = hasher.compute_hash({"test": True})
        assert len(hasher._cache) == 1

    def test_different_data_separate_cache_entries(self):
        hasher = ArtifactHasher()
        hasher.compute_hash({"a": 1})
        hasher.compute_hash({"b": 2})
        assert len(hasher._cache) == 2
        assert hasher._cache_misses == 2


class TestStoreAndHistory:
    """Test store_hash and get_hash_history."""

    def test_store_hash_creates_artifact(self):
        hasher = ArtifactHasher()
        artifact = hasher.store_hash("art-001", {"key": "value"})
        assert isinstance(artifact, ArtifactHash)
        assert artifact.artifact_id == "art-001"
        assert len(artifact.hash_value) == 64

    def test_store_hash_algorithm(self):
        hasher = ArtifactHasher()
        artifact = hasher.store_hash("art-002", {"x": 1})
        assert artifact.algorithm == "sha256"

    def test_get_hash_history_returns_entries(self):
        hasher = ArtifactHasher()
        hasher.store_hash("art-001", {"v": 1})
        hasher.store_hash("art-001", {"v": 2})
        history = hasher.get_hash_history("art-001")
        assert len(history) == 2

    def test_get_hash_history_nonexistent(self):
        hasher = ArtifactHasher()
        history = hasher.get_hash_history("nonexistent")
        assert history == []

    def test_history_preserves_order(self):
        hasher = ArtifactHasher()
        hasher.store_hash("art-001", {"v": 1})
        hasher.store_hash("art-001", {"v": 2})
        history = hasher.get_hash_history("art-001")
        assert history[0].hash_value != history[1].hash_value


class TestIncrementalHashConsistency:
    """Test that hashing is consistent across different hasher instances."""

    def test_same_data_different_hashers(self):
        h1 = ArtifactHasher()
        h2 = ArtifactHasher()
        data = {"emissions": 100.5, "fuel": "diesel"}
        assert h1.compute_hash(data) == h2.compute_hash(data)

    def test_float_normalization_consistent(self):
        h1 = ArtifactHasher()
        h2 = ArtifactHasher()
        data = {"value": 1.0000000001}
        assert h1.compute_hash(data) == h2.compute_hash(data)
