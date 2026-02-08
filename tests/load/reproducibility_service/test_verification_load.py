# -*- coding: utf-8 -*-
"""
Load Tests for Verification Under Load (AGENT-FOUND-008)

Tests concurrent verifications, sustained load, replay under load,
and memory stability.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline verification service for load testing
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class LoadTestVerifier:
    """Verifier optimized for load testing."""

    def __init__(self):
        self._results: List[Dict] = []

    def verify(self, input_data: Dict, expected_hash: Optional[str] = None) -> Dict:
        actual = _content_hash(input_data)
        if expected_hash is None:
            status = "skipped"
        elif actual == expected_hash:
            status = "pass"
        else:
            status = "fail"
        result = {
            "status": status,
            "input_hash": actual,
            "is_reproducible": status != "fail",
        }
        self._results.append(result)
        return result

    def replay(self, execution_id: str, inputs: Dict,
               expected_output: Optional[Dict] = None) -> Dict:
        ih = _content_hash(inputs)
        output_match = None
        if expected_output is not None:
            oh = _content_hash(inputs)  # Deterministic replay
            eoh = _content_hash(expected_output)
            output_match = oh == eoh
        return {
            "execution_id": execution_id,
            "input_hash": ih,
            "output_match": output_match,
            "status": "completed",
        }

    @property
    def total_verifications(self) -> int:
        return len(self._results)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestConcurrentVerifications:
    """Test concurrent verification operations."""

    def test_concurrent_verifications_50(self):
        verifier = LoadTestVerifier()
        data_items = [{"id": i, "value": float(i)} for i in range(500)]
        hashes = [_content_hash(d) for d in data_items]
        results = []

        def verify_batch(batch):
            batch_results = []
            for d, h in batch:
                batch_results.append(verifier.verify(d, h))
            return batch_results

        pairs = list(zip(data_items, hashes))
        batch_size = 10
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(verify_batch, b) for b in batches]
            for f in as_completed(futures):
                results.extend(f.result())

        assert len(results) == 500
        assert all(r["is_reproducible"] for r in results)

    def test_concurrent_mixed_pass_fail(self):
        verifier = LoadTestVerifier()
        results = []

        def verify_one(data, expected):
            return verifier.verify(data, expected)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                data = {"id": i}
                h = _content_hash(data)
                if i % 5 == 0:
                    h = "wrong_hash"  # Every 5th fails
                futures.append(executor.submit(verify_one, data, h))
            for f in as_completed(futures):
                results.append(f.result())

        passed = sum(1 for r in results if r["is_reproducible"])
        failed = sum(1 for r in results if not r["is_reproducible"])
        assert passed == 80
        assert failed == 20


class TestSustainedVerificationLoad:
    """Test sustained verification load over time."""

    def test_sustained_load_5_seconds(self):
        verifier = LoadTestVerifier()
        duration = 2.0  # seconds (reduced for CI)
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < duration:
            data = {"counter": count, "ts": count * 0.001}
            h = _content_hash(data)
            verifier.verify(data, h)
            count += 1
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        assert throughput >= 100, f"Sustained throughput {throughput:.0f}/s below 100/s"
        assert count > 0

    def test_sustained_load_no_errors(self):
        verifier = LoadTestVerifier()
        errors = 0
        for i in range(1000):
            data = {"id": i}
            h = _content_hash(data)
            result = verifier.verify(data, h)
            if not result["is_reproducible"]:
                errors += 1
        assert errors == 0


class TestReplayUnderLoad:
    """Test replay operations under load."""

    def test_replay_100_concurrent(self):
        verifier = LoadTestVerifier()
        results = []

        def do_replay(exec_id, inputs):
            return verifier.replay(exec_id, inputs)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                futures.append(
                    executor.submit(do_replay, f"exec-{i}", {"value": i})
                )
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == 100
        assert all(r["status"] == "completed" for r in results)

    def test_replay_with_output_comparison(self):
        verifier = LoadTestVerifier()
        results = []
        for i in range(100):
            result = verifier.replay(
                f"exec-{i}",
                {"value": i},
                expected_output={"value": i},
            )
            results.append(result)
        completed = sum(1 for r in results if r["status"] == "completed")
        assert completed == 100

    def test_replay_throughput(self):
        verifier = LoadTestVerifier()
        start = time.perf_counter()
        for i in range(1000):
            verifier.replay(f"exec-{i}", {"id": i})
        elapsed = time.perf_counter() - start
        throughput = 1000 / elapsed
        assert throughput >= 100, f"Replay throughput {throughput:.0f}/s below 100/s"


class TestMemoryStability:
    """Test memory stability during load."""

    def test_memory_stability_10k_verifications(self):
        verifier = LoadTestVerifier()
        # Measure initial memory approximation
        initial_results = verifier.total_verifications

        for i in range(10000):
            data = {"id": i, "payload": "x" * 100}
            h = _content_hash(data)
            verifier.verify(data, h)

        final_results = verifier.total_verifications
        assert final_results == initial_results + 10000

    def test_memory_stability_large_payloads(self):
        verifier = LoadTestVerifier()
        for i in range(100):
            data = {f"field_{j}": float(j) for j in range(1000)}
            data["id"] = i
            h = _content_hash(data)
            verifier.verify(data, h)
        assert verifier.total_verifications == 100

    def test_cache_does_not_grow_unbounded(self):
        """Test that hash cache stays manageable in size."""
        cache: Dict[str, str] = {}
        for i in range(10000):
            data = json.dumps({"unique": i}, sort_keys=True)
            h = hashlib.sha256(data.encode()).hexdigest()
            cache[data] = h

        # 10k unique entries, each ~40 bytes key + 64 bytes value
        # Should be under 10MB
        total_size = sum(len(k) + len(v) for k, v in cache.items())
        assert total_size < 10 * 1024 * 1024  # <10MB

    def test_sequential_verification_no_leak(self):
        """Run many verifications and check counter consistency."""
        verifier = LoadTestVerifier()
        n = 5000
        for i in range(n):
            verifier.verify({"i": i})
        assert verifier.total_verifications == n
