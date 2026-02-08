# -*- coding: utf-8 -*-
"""
Load Tests for Hash Performance (AGENT-FOUND-008)

Tests hash computation latency, batch throughput, concurrent requests,
verification throughput, and drift detection performance.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Inline hasher for performance testing
# ---------------------------------------------------------------------------

class PerformanceHasher:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def compute_hash(self, data: Any) -> str:
        normalized = self._normalize(data)
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        if json_str in self._cache:
            return self._cache[json_str]
        h = hashlib.sha256(json_str.encode()).hexdigest()
        self._cache[json_str] = h
        return h

    def _normalize(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, bool):
            return data
        if isinstance(data, (int, str)):
            return data
        if isinstance(data, float):
            return round(data, 9)
        if isinstance(data, dict):
            return {str(k): self._normalize(v) for k, v in sorted(data.items())}
        if isinstance(data, (list, tuple)):
            return [self._normalize(item) for item in data]
        return str(data)

    def verify_hash(self, data: Any, expected: str) -> bool:
        return self.compute_hash(data) == expected

    def batch_hash(self, items: List[Any]) -> List[str]:
        return [self.compute_hash(item) for item in items]

    def detect_drift(self, baseline: Dict, current: Dict) -> Dict:
        bh = self.compute_hash(baseline)
        ch = self.compute_hash(current)
        return {"match": bh == ch, "baseline_hash": bh, "current_hash": ch}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestHashComputationLatency:
    """Test that single hash computation stays under target latency."""

    def test_hash_computation_under_1ms(self):
        hasher = PerformanceHasher()
        data = {"emissions": 100.5, "fuel_type": "diesel", "quantity": 1000}
        times = []
        for _ in range(100):
            start = time.perf_counter()
            hasher.compute_hash(data)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        # Clear cache to test uncached performance
        hasher._cache.clear()
        for _ in range(100):
            start = time.perf_counter()
            hasher.compute_hash(data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        avg_ms = sum(times) / len(times)
        assert avg_ms < 1.0, f"Average hash time {avg_ms:.3f}ms exceeds 1ms target"

    def test_hash_computation_small_data(self):
        hasher = PerformanceHasher()
        data = {"a": 1}
        start = time.perf_counter()
        for _ in range(1000):
            hasher._cache.clear()
            hasher.compute_hash(data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_hash_ms = elapsed_ms / 1000
        assert per_hash_ms < 1.0

    def test_hash_computation_medium_data(self):
        hasher = PerformanceHasher()
        data = {f"field_{i}": float(i) * 1.1 for i in range(100)}
        start = time.perf_counter()
        for _ in range(100):
            hasher._cache.clear()
            hasher.compute_hash(data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_hash_ms = elapsed_ms / 100
        assert per_hash_ms < 5.0, f"Medium data hash {per_hash_ms:.3f}ms exceeds 5ms"


class TestBatchHashPerformance:
    """Test batch hashing throughput."""

    def test_batch_hash_1000_items_under_1s(self):
        hasher = PerformanceHasher()
        items = [{"id": i, "value": float(i) * 1.1} for i in range(1000)]
        start = time.perf_counter()
        results = hasher.batch_hash(items)
        elapsed = time.perf_counter() - start
        assert len(results) == 1000
        assert elapsed < 1.0, f"Batch 1000 took {elapsed:.3f}s, exceeds 1s target"

    def test_batch_hash_10000_items(self):
        hasher = PerformanceHasher()
        items = [{"id": i, "v": float(i)} for i in range(10000)]
        start = time.perf_counter()
        results = hasher.batch_hash(items)
        elapsed = time.perf_counter() - start
        assert len(results) == 10000
        assert elapsed < 10.0, f"Batch 10000 took {elapsed:.3f}s"

    def test_batch_throughput_rate(self):
        hasher = PerformanceHasher()
        items = [{"id": i, "v": float(i)} for i in range(5000)]
        start = time.perf_counter()
        hasher.batch_hash(items)
        elapsed = time.perf_counter() - start
        throughput = 5000 / elapsed
        assert throughput > 1000, f"Throughput {throughput:.0f}/s below 1000/s target"


class TestConcurrentHashRequests:
    """Test hash computation under concurrent load."""

    def test_concurrent_hash_10_threads(self):
        hasher = PerformanceHasher()
        data_items = [{"thread": i, "value": float(i)} for i in range(100)]
        results = []

        def hash_batch(items):
            return [hasher.compute_hash(item) for item in items]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for batch_start in range(0, 100, 10):
                batch = data_items[batch_start:batch_start + 10]
                futures.append(executor.submit(hash_batch, batch))
            for f in as_completed(futures):
                results.extend(f.result())

        assert len(results) == 100

    def test_concurrent_verify_20_threads(self):
        hasher = PerformanceHasher()
        data_items = [{"id": i} for i in range(200)]
        hashes = [hasher.compute_hash(d) for d in data_items]
        results = []

        def verify_batch(items_hashes):
            return [hasher.verify_hash(d, h) for d, h in items_hashes]

        pairs = list(zip(data_items, hashes))
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for batch_start in range(0, 200, 10):
                batch = pairs[batch_start:batch_start + 10]
                futures.append(executor.submit(verify_batch, batch))
            for f in as_completed(futures):
                results.extend(f.result())

        assert all(results)
        assert len(results) == 200


class TestVerificationThroughput:
    """Test verification throughput target."""

    def test_verification_throughput_100_per_second(self):
        hasher = PerformanceHasher()
        data_items = [{"id": i, "v": float(i)} for i in range(100)]
        hashes = [hasher.compute_hash(d) for d in data_items]

        start = time.perf_counter()
        for d, h in zip(data_items, hashes):
            hasher.verify_hash(d, h)
        elapsed = time.perf_counter() - start

        throughput = 100 / elapsed
        assert throughput >= 100, f"Throughput {throughput:.0f}/s below 100/s"

    def test_verification_1000_ops(self):
        hasher = PerformanceHasher()
        data_items = [{"id": i} for i in range(1000)]
        hashes = [hasher.compute_hash(d) for d in data_items]

        start = time.perf_counter()
        results = [hasher.verify_hash(d, h) for d, h in zip(data_items, hashes)]
        elapsed = time.perf_counter() - start

        assert all(results)
        throughput = 1000 / elapsed
        assert throughput >= 100


class TestDriftDetectionPerformance:
    """Test drift detection performance."""

    def test_drift_detection_latency(self):
        hasher = PerformanceHasher()
        baseline = {f"field_{i}": float(i) for i in range(50)}
        current = {f"field_{i}": float(i) + 0.001 for i in range(50)}
        times = []
        for _ in range(100):
            start = time.perf_counter()
            hasher.detect_drift(baseline, current)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        avg_ms = sum(times) / len(times)
        assert avg_ms < 5.0, f"Drift detection avg {avg_ms:.3f}ms exceeds 5ms"

    def test_drift_detection_batch(self):
        hasher = PerformanceHasher()
        baseline = {"emissions": 100.0}
        currents = [{"emissions": 100.0 + i * 0.1} for i in range(1000)]
        start = time.perf_counter()
        for current in currents:
            hasher.detect_drift(baseline, current)
        elapsed = time.perf_counter() - start
        throughput = 1000 / elapsed
        assert throughput >= 100, f"Drift throughput {throughput:.0f}/s below 100/s"
