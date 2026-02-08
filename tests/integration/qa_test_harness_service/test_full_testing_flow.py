# -*- coding: utf-8 -*-
"""
Integration Tests for Full Testing Flow (AGENT-FOUND-009)

Tests end-to-end flows combining test runner, assertion engine, golden file
manager, regression detector, performance benchmarker, coverage tracker,
and report generator working together.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline integrated components
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


def _hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


class IntegratedHarness:
    """Integrated QA test harness for end-to-end testing."""

    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._golden: Dict[str, Dict] = {}
        self._baselines: Dict[str, Dict] = {}
        self._history: List[Dict] = []
        self._coverage: Dict[str, Set[str]] = {}
        self._agent_methods: Dict[str, List[str]] = {}

    def register(self, name: str, fn: Callable,
                 methods: Optional[List[str]] = None) -> None:
        self._agents[name] = fn
        self._agent_methods[name] = methods or ["execute", "run"]
        self._coverage[name] = set()

    def run_test(self, agent_type: str, input_data: Dict,
                 expected: Optional[Dict] = None) -> Dict:
        fn = self._agents.get(agent_type)
        if fn is None:
            r = {"status": "error", "error": f"Not registered: {agent_type}"}
            self._history.append(r)
            return r
        try:
            output = fn(input_data)
            status = "passed" if output.get("success") else "failed"
            if expected:
                for k, v in expected.items():
                    if output.get("data", {}).get(k) != v:
                        status = "failed"
                        break
            self._coverage[agent_type].add("execute")
            self._coverage[agent_type].add("run")
        except Exception as e:
            output = {}
            status = "error"
        r = {"status": status, "output": output,
             "input_hash": _hash(input_data), "output_hash": _hash(output)}
        self._history.append(r)
        return r

    def run_suite(self, agent_type: str, cases: List[Dict]) -> Dict:
        results = [self.run_test(agent_type, c.get("input", {}),
                                 c.get("expected")) for c in cases]
        passed = sum(1 for r in results if r["status"] == "passed")
        total = len(results)
        return {"total": total, "passed": passed, "failed": total - passed,
                "pass_rate": round(passed / total * 100, 2) if total else 0,
                "results": results}

    def save_golden(self, agent_type: str, inp: Dict, out: Dict) -> str:
        gf_id = f"gf-{len(self._golden) + 1}"
        self._golden[gf_id] = {"agent": agent_type, "input": inp,
                                "expected": out, "hash": _hash(out)}
        return gf_id

    def compare_golden(self, gf_id: str, actual: Dict) -> Dict:
        gf = self._golden.get(gf_id)
        if not gf:
            return {"match": False, "error": "Not found"}
        exp = gf["expected"]
        diffs = [k for k in exp if exp[k] != actual.get(k)]
        return {"match": len(diffs) == 0, "diffs": diffs}

    def create_baseline(self, agent_type: str, inp: Dict, out: Dict) -> str:
        bl_id = f"bl-{len(self._baselines) + 1}"
        self._baselines[bl_id] = {"agent": agent_type, "input_hash": _hash(inp),
                                   "output_hash": _hash(out), "output": out}
        return bl_id

    def check_regression(self, bl_id: str, out: Dict) -> Dict:
        bl = self._baselines.get(bl_id)
        if not bl:
            return {"regression": False, "message": "No baseline"}
        current = _hash(out)
        return {"regression": current != bl["output_hash"],
                "baseline_hash": bl["output_hash"], "current_hash": current}

    def benchmark(self, agent_type: str, inp: Dict, iters: int = 5) -> Dict:
        fn = self._agents[agent_type]
        timings = []
        for _ in range(iters):
            s = time.perf_counter()
            fn(inp)
            timings.append((time.perf_counter() - s) * 1000)
        timings.sort()
        return {"min": round(timings[0], 3),
                "mean": round(sum(timings) / len(timings), 3),
                "max": round(timings[-1], 3)}

    def get_coverage(self, agent_type: str) -> Dict:
        total = self._agent_methods.get(agent_type, [])
        covered = self._coverage.get(agent_type, set())
        covered_count = len(covered.intersection(set(total)))
        pct = (covered_count / len(total) * 100) if total else 0
        return {"total": len(total), "covered": covered_count, "pct": round(pct, 2)}

    def generate_report(self, suite_result: Dict) -> str:
        return (f"Total: {suite_result.get('total', 0)}, "
                f"Passed: {suite_result.get('passed', 0)}, "
                f"Pass Rate: {suite_result.get('pass_rate', 0)}%")

    def get_stats(self) -> Dict:
        total = len(self._history)
        passed = sum(1 for r in self._history if r.get("status") == "passed")
        return {"total": total, "passed": passed, "failed": total - passed}


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------

def _good_agent(d):
    return {"success": True, "data": {"result": 42, "provenance_id": "prov-abc12345"},
            "error": None}


def _bad_agent(d):
    return {"success": False, "data": {}, "error": "Failed"}


def _varying_agent(d):
    """Agent that produces different output based on input."""
    val = d.get("x", 0) * 10
    return {"success": True, "data": {"result": val}, "error": None}


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def harness():
    h = IntegratedHarness()
    h.register("GoodAgent", _good_agent, ["execute", "run", "validate"])
    h.register("BadAgent", _bad_agent)
    h.register("VaryAgent", _varying_agent)
    return h


class TestEndToEndTestPass:
    def test_e2e_pass(self, harness):
        result = harness.run_test("GoodAgent", {"x": 1})
        assert result["status"] == "passed"
        assert result["output"]["success"] is True

    def test_e2e_pass_has_hashes(self, harness):
        result = harness.run_test("GoodAgent", {"x": 1})
        assert len(result["input_hash"]) == 16
        assert len(result["output_hash"]) == 16

    def test_e2e_pass_records_history(self, harness):
        harness.run_test("GoodAgent", {"x": 1})
        assert harness.get_stats()["total"] == 1


class TestEndToEndTestFail:
    def test_e2e_fail(self, harness):
        result = harness.run_test("BadAgent", {"x": 1})
        assert result["status"] == "failed"

    def test_e2e_fail_expected_mismatch(self, harness):
        result = harness.run_test("GoodAgent", {}, expected={"result": 999})
        assert result["status"] == "failed"


class TestSuiteExecutionWithResults:
    def test_suite_all_pass(self, harness):
        cases = [{"input": {"x": i}} for i in range(5)]
        result = harness.run_suite("GoodAgent", cases)
        assert result["total"] == 5
        assert result["passed"] == 5
        assert result["pass_rate"] == 100.0

    def test_suite_mixed(self, harness):
        cases = [{"input": {}}]
        r1 = harness.run_suite("GoodAgent", cases)
        r2 = harness.run_suite("BadAgent", cases)
        assert r1["passed"] == 1
        assert r2["failed"] == 1

    def test_suite_empty(self, harness):
        result = harness.run_suite("GoodAgent", [])
        assert result["total"] == 0


class TestGoldenFileSaveThenCompare:
    def test_save_then_match(self, harness):
        output = _good_agent({})
        gf_id = harness.save_golden(
            "GoodAgent", {}, output["data"])
        result = harness.compare_golden(gf_id, output["data"])
        assert result["match"] is True

    def test_save_then_mismatch(self, harness):
        gf_id = harness.save_golden("GoodAgent", {}, {"result": 42})
        result = harness.compare_golden(gf_id, {"result": 99})
        assert result["match"] is False
        assert "result" in result["diffs"]

    def test_compare_nonexistent(self, harness):
        result = harness.compare_golden("gf-999", {})
        assert result["match"] is False


class TestRegressionBaselineThenCheck:
    def test_baseline_then_match(self, harness):
        output = _good_agent({})
        bl_id = harness.create_baseline("GoodAgent", {}, output)
        result = harness.check_regression(bl_id, output)
        assert result["regression"] is False

    def test_baseline_then_regression(self, harness):
        output1 = _good_agent({})
        bl_id = harness.create_baseline("GoodAgent", {}, output1)
        output2 = {"different": "data"}
        result = harness.check_regression(bl_id, output2)
        assert result["regression"] is True

    def test_no_baseline(self, harness):
        result = harness.check_regression("bl-999", {})
        assert result["regression"] is False


class TestBenchmarkThenBaselineCompare:
    def test_benchmark_runs(self, harness):
        result = harness.benchmark("GoodAgent", {}, iters=5)
        assert result["min"] >= 0
        assert result["mean"] >= result["min"]
        assert result["max"] >= result["mean"]


class TestCoverageTrackThenReport:
    def test_coverage_after_test(self, harness):
        harness.run_test("GoodAgent", {})
        report = harness.get_coverage("GoodAgent")
        assert report["covered"] >= 2  # execute and run covered
        assert report["pct"] > 0

    def test_coverage_unregistered(self, harness):
        report = harness.get_coverage("Unknown")
        assert report["total"] == 0


class TestReportGenerationAfterSuite:
    def test_report_generation(self, harness):
        cases = [{"input": {"x": i}} for i in range(3)]
        suite_result = harness.run_suite("GoodAgent", cases)
        report = harness.generate_report(suite_result)
        assert "3" in report
        assert "100.0%" in report

    def test_report_with_failures(self, harness):
        suite_result = {"total": 10, "passed": 7, "pass_rate": 70.0}
        report = harness.generate_report(suite_result)
        assert "70.0%" in report


class TestStatisticsAfterMultipleOperations:
    def test_full_stats(self, harness):
        harness.run_test("GoodAgent", {})
        harness.run_test("BadAgent", {})
        harness.run_test("GoodAgent", {"x": 2})
        stats = harness.get_stats()
        assert stats["total"] == 3
        assert stats["passed"] == 2
        assert stats["failed"] == 1
