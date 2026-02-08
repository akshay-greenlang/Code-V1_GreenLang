# -*- coding: utf-8 -*-
"""
Integration Tests for Deterministic Hashing (AGENT-FOUND-008)

Tests hash consistency, float normalization roundtrips, decimal precision,
complex nested data, large datasets, and batch operations.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Inline hasher
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE = 1e-9


class DeterministicHasher:
    """Full deterministic hasher for integration testing."""

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        self.tolerance = tolerance

    def compute(self, data: Any) -> str:
        normalized = self._normalize(data)
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _normalize(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, bool):
            return data
        if isinstance(data, (int, str)):
            return data
        if isinstance(data, float):
            if self.tolerance > 0:
                precision = max(0, -int(math.log10(self.tolerance)))
                return round(data, precision)
            return data
        if isinstance(data, Decimal):
            return str(data)
        if isinstance(data, datetime):
            return data.replace(microsecond=0).isoformat()
        if isinstance(data, dict):
            return {str(k): self._normalize(v) for k, v in sorted(data.items())}
        if isinstance(data, (list, tuple)):
            return [self._normalize(item) for item in data]
        if isinstance(data, set):
            return sorted([self._normalize(item) for item in data])
        return str(data)

    def batch_compute(self, items: List[Any]) -> List[str]:
        return [self.compute(item) for item in items]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestHashConsistencyAcrossCalls:
    """Test hash consistency across multiple calls and instances."""

    def test_same_instance_same_hash(self):
        h = DeterministicHasher()
        data = {"emissions": 100.5, "fuel": "diesel"}
        assert h.compute(data) == h.compute(data)

    def test_different_instance_same_hash(self):
        h1 = DeterministicHasher()
        h2 = DeterministicHasher()
        data = {"emissions": 100.5, "fuel": "diesel"}
        assert h1.compute(data) == h2.compute(data)

    def test_key_order_independence(self):
        h = DeterministicHasher()
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        assert h.compute(d1) == h.compute(d2)

    def test_multiple_sequential_calls(self):
        h = DeterministicHasher()
        data = {"value": 42.0}
        hashes = [h.compute(data) for _ in range(100)]
        assert len(set(hashes)) == 1

    def test_consistency_with_nested_dicts(self):
        h = DeterministicHasher()
        data = {"l1": {"l2": {"l3": {"value": 1.0}}}}
        assert h.compute(data) == h.compute(data)


class TestFloatNormalizationRoundtrip:
    """Test float normalization produces consistent hashes."""

    def test_float_precision_collapse(self):
        h = DeterministicHasher(tolerance=1e-6)
        d1 = {"v": 1.0000001}
        d2 = {"v": 1.0000002}
        # These should normalize to the same value with 6 decimal places
        assert h.compute(d1) == h.compute(d2)

    def test_float_significant_difference(self):
        h = DeterministicHasher(tolerance=1e-9)
        d1 = {"v": 1.0}
        d2 = {"v": 2.0}
        assert h.compute(d1) != h.compute(d2)

    def test_float_zero(self):
        h = DeterministicHasher()
        d1 = {"v": 0.0}
        d2 = {"v": 0.0}
        assert h.compute(d1) == h.compute(d2)

    def test_float_negative(self):
        h = DeterministicHasher()
        d1 = {"v": -1.5}
        d2 = {"v": -1.5}
        assert h.compute(d1) == h.compute(d2)


class TestDecimalPrecision:
    """Test Decimal handling in hashing."""

    def test_decimal_hashed_as_string(self):
        h = DeterministicHasher()
        d = {"v": Decimal("100.50")}
        result = h.compute(d)
        assert len(result) == 64

    def test_decimal_deterministic(self):
        h = DeterministicHasher()
        d1 = {"v": Decimal("100.50")}
        d2 = {"v": Decimal("100.50")}
        assert h.compute(d1) == h.compute(d2)

    def test_decimal_different_precision_different_hash(self):
        h = DeterministicHasher()
        d1 = {"v": Decimal("100.5")}
        d2 = {"v": Decimal("100.50")}
        # "100.5" != "100.50" as strings
        assert h.compute(d1) != h.compute(d2)


class TestNestedComplexData:
    """Test hashing of complex nested structures."""

    def test_deeply_nested(self):
        h = DeterministicHasher()
        data = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        assert len(h.compute(data)) == 64

    def test_mixed_types(self):
        h = DeterministicHasher()
        data = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        assert h.compute(data) == h.compute(data)

    def test_list_of_dicts(self):
        h = DeterministicHasher()
        data = {"items": [{"id": 1, "v": 10}, {"id": 2, "v": 20}]}
        assert h.compute(data) == h.compute(data)

    def test_set_handling(self):
        h = DeterministicHasher()
        data1 = {"values": {3, 1, 2}}
        data2 = {"values": {1, 2, 3}}
        assert h.compute(data1) == h.compute(data2)


class TestLargeDataset:
    """Test hashing of large datasets."""

    def test_large_dict(self):
        h = DeterministicHasher()
        data = {f"field_{i}": float(i) for i in range(1000)}
        result = h.compute(data)
        assert len(result) == 64
        # Verify deterministic
        assert h.compute(data) == result

    def test_large_list(self):
        h = DeterministicHasher()
        data = {"values": list(range(10000))}
        result = h.compute(data)
        assert len(result) == 64


class TestBatchHashConsistency:
    """Test batch hashing consistency."""

    def test_batch_matches_individual(self):
        h = DeterministicHasher()
        items = [{"a": i} for i in range(10)]
        batch_results = h.batch_compute(items)
        individual_results = [h.compute(item) for item in items]
        assert batch_results == individual_results

    def test_batch_deterministic(self):
        h = DeterministicHasher()
        items = [{"x": 1}, {"x": 2}, {"x": 3}]
        r1 = h.batch_compute(items)
        r2 = h.batch_compute(items)
        assert r1 == r2

    def test_batch_empty(self):
        h = DeterministicHasher()
        assert h.batch_compute([]) == []

    def test_batch_1000_items(self):
        h = DeterministicHasher()
        items = [{"value": i * 0.1} for i in range(1000)]
        results = h.batch_compute(items)
        assert len(results) == 1000
        assert len(set(results)) == 1000  # All unique
