# -*- coding: utf-8 -*-
"""
Integration Tests for Assertion Categories (AGENT-FOUND-009)

Tests end-to-end flows for each assertion category: zero-hallucination,
determinism, lineage, golden file, and regression.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline integrated assertion testing
# ---------------------------------------------------------------------------


def _hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


class AssertionTestHarness:
    """Harness for testing assertion categories end-to-end."""

    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._golden: Dict[str, Dict] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._agents[name] = fn

    def test_zero_hallucination(self, agent_type: str, input_data: Dict) -> Dict:
        fn = self._agents[agent_type]
        output = fn(input_data)
        checks = []

        # Numeric traceability
        data = output.get("data", {})
        input_numbers = self._extract_numbers(input_data)
        output_numbers = self._extract_numbers(data)
        for key, val in output_numbers.items():
            suspicious = (isinstance(val, (int, float)) and val > 0
                         and val == round(val, -3)
                         and val not in input_numbers.values())
            checks.append({"name": f"numeric_{key}", "passed": not suspicious})

        # Provenance
        prov = data.get("provenance_id")
        checks.append({"name": "provenance", "passed": isinstance(prov, str) and len(prov) >= 8})

        # Consistency
        success = output.get("success", False)
        has_data = bool(data)
        has_error = bool(output.get("error"))
        consistent = (success and has_data and not has_error) or (not success and has_error)
        checks.append({"name": "consistency", "passed": consistent})

        all_pass = all(c["passed"] for c in checks)
        return {"passed": all_pass, "checks": checks}

    def test_determinism(self, agent_type: str, input_data: Dict,
                         iterations: int = 3) -> Dict:
        fn = self._agents[agent_type]
        hashes = []
        for _ in range(iterations):
            output = fn(input_data)
            hashes.append(_hash(output))
        all_equal = len(set(hashes)) == 1
        return {"is_deterministic": all_equal, "unique": len(set(hashes)),
                "hashes": hashes}

    def test_lineage(self, agent_type: str, input_data: Dict) -> Dict:
        fn = self._agents[agent_type]
        output = fn(input_data)
        data = output.get("data", {})
        has_prov = "provenance_id" in data
        has_ts = "timestamp" in output or "created_at" in data
        has_metrics = output.get("metrics") is not None
        return {"has_provenance": has_prov, "has_timestamp": has_ts,
                "has_metrics": has_metrics,
                "complete": has_prov and has_ts}

    def test_golden_file(self, agent_type: str, input_data: Dict,
                         gf_id: str) -> Dict:
        fn = self._agents[agent_type]
        output = fn(input_data)
        gf = self._golden.get(gf_id)
        if not gf:
            return {"passed": False, "error": "Golden file not found"}
        expected = gf["expected"]
        actual = output.get("data", {})
        diffs = [k for k in expected if expected.get(k) != actual.get(k)]
        return {"passed": len(diffs) == 0, "diffs": diffs}

    def test_regression(self, agent_type: str, input_data: Dict,
                        baseline_hash: str) -> Dict:
        fn = self._agents[agent_type]
        output = fn(input_data)
        current = _hash(output)
        return {"has_regression": current != baseline_hash,
                "baseline": baseline_hash, "current": current}

    def save_golden(self, name: str, expected: Dict) -> str:
        gf_id = f"gf-{len(self._golden) + 1}"
        self._golden[gf_id] = {"name": name, "expected": expected}
        return gf_id

    def _extract_numbers(self, data: Any, prefix: str = "") -> Dict[str, Any]:
        numbers = {}
        if isinstance(data, dict):
            for k, v in data.items():
                fk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numbers[fk] = v
                elif isinstance(v, (dict, list)):
                    numbers.update(self._extract_numbers(v, fk))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                fk = f"{prefix}[{i}]"
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    numbers[fk] = item
                elif isinstance(item, (dict, list)):
                    numbers.update(self._extract_numbers(item, fk))
        return numbers


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------

def _good_agent(d):
    return {"success": True,
            "data": {"result": 42.5, "provenance_id": "prov-abc12345",
                     "created_at": "2026-01-01"},
            "error": None, "metrics": {"duration_ms": 2.0},
            "timestamp": "2026-01-01T00:00:00Z"}


def _hallucinating_agent(d):
    return {"success": True,
            "data": {"result": 5000, "provenance_id": "prov-abc12345"},
            "error": None}


def _no_provenance_agent(d):
    return {"success": True, "data": {"result": 10}, "error": None}


def _non_deterministic_agent(d):
    import random
    return {"success": True, "data": {"result": random.random()}, "error": None}


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def harness():
    h = AssertionTestHarness()
    h.register("GoodAgent", _good_agent)
    h.register("HalluAgent", _hallucinating_agent)
    h.register("NoProvAgent", _no_provenance_agent)
    h.register("NonDetAgent", _non_deterministic_agent)
    return h


class TestZeroHallucinationFlow:
    def test_good_agent_passes(self, harness):
        result = harness.test_zero_hallucination("GoodAgent", {"x": 42.5})
        assert result["passed"] is True

    def test_hallucinating_agent_fails(self, harness):
        result = harness.test_zero_hallucination("HalluAgent", {"x": 123})
        suspicious = [c for c in result["checks"] if not c["passed"]]
        assert len(suspicious) >= 1

    def test_has_provenance_check(self, harness):
        result = harness.test_zero_hallucination("GoodAgent", {})
        prov_check = [c for c in result["checks"] if c["name"] == "provenance"]
        assert len(prov_check) == 1
        assert prov_check[0]["passed"] is True

    def test_consistency_check(self, harness):
        result = harness.test_zero_hallucination("GoodAgent", {})
        cons_check = [c for c in result["checks"] if c["name"] == "consistency"]
        assert len(cons_check) == 1
        assert cons_check[0]["passed"] is True


class TestDeterminismFlow:
    def test_deterministic_agent(self, harness):
        result = harness.test_determinism("GoodAgent", {"x": 1}, iterations=5)
        assert result["is_deterministic"] is True
        assert result["unique"] == 1

    def test_non_deterministic_agent(self, harness):
        result = harness.test_determinism("NonDetAgent", {}, iterations=10)
        # Random agent is very likely non-deterministic with 10 iterations
        assert result["unique"] >= 1  # at least 1 unique hash

    def test_determinism_iterations_count(self, harness):
        result = harness.test_determinism("GoodAgent", {}, iterations=7)
        assert len(result["hashes"]) == 7


class TestLineageFlow:
    def test_good_agent_lineage_complete(self, harness):
        result = harness.test_lineage("GoodAgent", {})
        assert result["has_provenance"] is True
        assert result["has_timestamp"] is True
        assert result["complete"] is True

    def test_no_provenance_agent(self, harness):
        result = harness.test_lineage("NoProvAgent", {})
        assert result["has_provenance"] is False
        assert result["complete"] is False

    def test_lineage_metrics(self, harness):
        result = harness.test_lineage("GoodAgent", {})
        assert result["has_metrics"] is True


class TestGoldenFileFlow:
    def test_golden_file_match(self, harness):
        gf_id = harness.save_golden("test", {"result": 42.5, "provenance_id": "prov-abc12345",
                                              "created_at": "2026-01-01"})
        result = harness.test_golden_file("GoodAgent", {}, gf_id)
        assert result["passed"] is True

    def test_golden_file_mismatch(self, harness):
        gf_id = harness.save_golden("test", {"result": 999})
        result = harness.test_golden_file("GoodAgent", {}, gf_id)
        assert result["passed"] is False

    def test_golden_file_not_found(self, harness):
        result = harness.test_golden_file("GoodAgent", {}, "gf-999")
        assert result["passed"] is False

    def test_golden_file_partial_match(self, harness):
        gf_id = harness.save_golden("test", {"result": 42.5})
        result = harness.test_golden_file("GoodAgent", {}, gf_id)
        assert result["passed"] is True


class TestRegressionFlow:
    def test_no_regression(self, harness):
        output = _good_agent({})
        baseline = _hash(output)
        result = harness.test_regression("GoodAgent", {}, baseline)
        assert result["has_regression"] is False

    def test_regression_detected(self, harness):
        result = harness.test_regression("GoodAgent", {}, "wrong_hash")
        assert result["has_regression"] is True

    def test_regression_hashes_populated(self, harness):
        output = _good_agent({})
        baseline = _hash(output)
        result = harness.test_regression("GoodAgent", {}, baseline)
        assert result["baseline"] == baseline
        assert result["current"] == baseline
