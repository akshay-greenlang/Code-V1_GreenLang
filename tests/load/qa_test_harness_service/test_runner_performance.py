# -*- coding: utf-8 -*-
"""
Load Tests for Test Runner Performance (AGENT-FOUND-009)

Tests single test latency, suite throughput, concurrent suite execution,
assertion throughput, large suite scalability, and hash computation
under load.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs (self-contained for load testing)
# ---------------------------------------------------------------------------


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestCategory(str, Enum):
    ZERO_HALLUCINATION = "zero_hallucination"
    DETERMINISM = "determinism"
    LINEAGE = "lineage"
    GOLDEN_FILE = "golden_file"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    COVERAGE = "coverage"
    INTEGRATION = "integration"
    UNIT = "unit"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestAssertion:
    def __init__(self, name, passed, message="", severity=SeverityLevel.HIGH):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity


class TestCaseInput:
    def __init__(self, name, agent_type, test_id="", category=TestCategory.UNIT,
                 input_data=None, expected_output=None, timeout_seconds=60,
                 tags=None, skip=False, skip_reason="", severity=SeverityLevel.HIGH):
        self.test_id = test_id or str(uuid.uuid4())
        self.name = name
        self.category = category
        self.agent_type = agent_type
        self.input_data = input_data or {}
        self.expected_output = expected_output
        self.timeout_seconds = timeout_seconds
        self.tags = tags or []
        self.skip = skip
        self.skip_reason = skip_reason
        self.severity = severity


class TestCaseResult:
    def __init__(self, test_id, name, category, status, assertions=None,
                 duration_ms=0.0, started_at=None, completed_at=None,
                 error_message=None, agent_result=None, input_hash="",
                 output_hash="", metadata=None):
        self.test_id = test_id
        self.name = name
        self.category = category
        self.status = status
        self.assertions = assertions or []
        self.duration_ms = duration_ms
        self.started_at = started_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.agent_result = agent_result
        self.input_hash = input_hash
        self.output_hash = output_hash
        self.metadata = metadata or {}


class TestSuiteInput:
    def __init__(self, name, suite_id="", test_cases=None, parallel=True,
                 max_workers=4, fail_fast=False, tags_include=None,
                 tags_exclude=None):
        self.suite_id = suite_id or str(uuid.uuid4())
        self.name = name
        self.test_cases = test_cases or []
        self.parallel = parallel
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.tags_include = tags_include or []
        self.tags_exclude = tags_exclude or []


class TestSuiteResult:
    def __init__(self, suite_id, name, status, test_results=None,
                 total_tests=0, passed=0, failed=0, skipped=0, errors=0,
                 duration_ms=0.0, pass_rate=0.0, provenance_hash=""):
        self.suite_id = suite_id
        self.name = name
        self.status = status
        self.test_results = test_results or []
        self.total_tests = total_tests
        self.passed = passed
        self.failed = failed
        self.skipped = skipped
        self.errors = errors
        self.duration_ms = duration_ms
        self.pass_rate = pass_rate
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline TestRunner for load testing
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class LoadTestRunner:
    """TestRunner optimized for load testing."""

    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._history: List[TestCaseResult] = []

    def register_agent(self, agent_type: str, agent_fn: Callable) -> None:
        self._agents[agent_type] = agent_fn

    def run_test(self, test_input: TestCaseInput) -> TestCaseResult:
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        input_hash = _compute_hash(test_input.input_data)

        if test_input.skip:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.SKIPPED,
                input_hash=input_hash,
                metadata={"skip_reason": test_input.skip_reason},
                started_at=started_at,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

        agent_fn = self._agents.get(test_input.agent_type)
        if agent_fn is None:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.ERROR,
                input_hash=input_hash,
                error_message=f"Agent type not registered: {test_input.agent_type}",
                started_at=started_at,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

        try:
            output = agent_fn(test_input.input_data)
            output_hash = _compute_hash(output)

            assertions = []
            success = output.get("success", False)
            assertions.append(TestAssertion(
                name="agent_success", passed=success,
                message="" if success else "Agent did not succeed",
            ))

            if test_input.expected_output:
                for key, expected in test_input.expected_output.items():
                    actual = output.get("data", {}).get(key)
                    match = actual == expected
                    assertions.append(TestAssertion(
                        name=f"expected_{key}", passed=match,
                        message=f"Expected {expected}, got {actual}" if not match else "",
                    ))

            all_pass = all(a.passed for a in assertions)
            status = TestStatus.PASSED if all_pass else TestStatus.FAILED

            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=status,
                assertions=assertions, input_hash=input_hash,
                output_hash=output_hash, agent_result=output,
                started_at=started_at,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.ERROR,
                input_hash=input_hash, error_message=str(e),
                started_at=started_at,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        self._history.append(result)
        return result

    def run_suite(self, suite_input: TestSuiteInput) -> TestSuiteResult:
        start_time = time.perf_counter()
        results = []
        for test in suite_input.test_cases:
            results.append(self.run_test(test))

        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        total = len(results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        overall_status = TestStatus.PASSED
        if failed > 0 or errors > 0:
            overall_status = TestStatus.FAILED
        elif skipped == total:
            overall_status = TestStatus.SKIPPED

        duration_ms = (time.perf_counter() - start_time) * 1000

        return TestSuiteResult(
            suite_id=suite_input.suite_id, name=suite_input.name,
            status=overall_status, test_results=results,
            total_tests=total, passed=passed, failed=failed,
            skipped=skipped, errors=errors, pass_rate=round(pass_rate, 2),
            duration_ms=round(duration_ms, 3),
        )

    @property
    def total_runs(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Inline AssertionEngine for assertion throughput tests
# ---------------------------------------------------------------------------


class LoadAssertionEngine:
    """Assertion engine optimized for load testing."""

    def check_zero_hallucination(self, input_data: Dict, output_data: Dict) -> List[Dict]:
        checks = []
        input_numbers = self._extract_numbers(input_data)
        output_numbers = self._extract_numbers(output_data)

        for key, val in output_numbers.items():
            suspicious = (
                isinstance(val, (int, float))
                and val > 0
                and val == round(val, -3)
                and val not in input_numbers.values()
            )
            checks.append({"name": f"numeric_{key}", "passed": not suspicious})

        prov = output_data.get("provenance_id")
        checks.append({
            "name": "provenance",
            "passed": isinstance(prov, str) and len(prov) >= 8,
        })

        return checks

    def check_determinism(self, agent_fn: Callable, input_data: Dict,
                          iterations: int = 3) -> Dict:
        hashes = []
        for _ in range(iterations):
            output = agent_fn(input_data)
            hashes.append(_compute_hash(output))
        all_equal = len(set(hashes)) == 1
        return {"is_deterministic": all_equal, "unique": len(set(hashes))}

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
# Mock agents for load testing
# ---------------------------------------------------------------------------


def _fast_agent(input_data):
    return {
        "success": True,
        "data": {"result": 42, "provenance_id": "prov-load-test-1"},
        "error": None,
    }


def _medium_agent(input_data):
    return {
        "success": True,
        "data": {
            "result": sum(v for v in input_data.values() if isinstance(v, (int, float))),
            "provenance_id": "prov-med-test",
            "region": input_data.get("region", "US"),
        },
        "error": None,
    }


def _heavy_agent(input_data):
    """Agent with heavier computation (larger output)."""
    return {
        "success": True,
        "data": {
            f"field_{i}": float(i) * 1.1
            for i in range(50)
        },
        "error": None,
    }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSingleTestLatency:
    """Test that single test execution stays under target latency."""

    def test_single_test_under_10ms(self):
        """Test that a single test run completes in under 10ms on average."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        times = []
        for i in range(200):
            tc = TestCaseInput(name=f"perf_test_{i}", agent_type="FastAgent",
                               input_data={"x": i})
            start = time.perf_counter()
            result = runner.run_test(tc)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert result.status == TestStatus.PASSED

        avg_ms = sum(times) / len(times)
        assert avg_ms < 10.0, f"Average single test time {avg_ms:.3f}ms exceeds 10ms target"

    def test_single_test_p95_under_15ms(self):
        """Test that p95 latency is under 15ms."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        times = []
        for i in range(500):
            tc = TestCaseInput(name=f"p95_test_{i}", agent_type="FastAgent",
                               input_data={"v": i})
            start = time.perf_counter()
            runner.run_test(tc)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        times.sort()
        p95_idx = int(len(times) * 0.95)
        p95_ms = times[p95_idx]
        assert p95_ms < 15.0, f"p95 latency {p95_ms:.3f}ms exceeds 15ms target"

    def test_single_test_with_assertions_under_15ms(self):
        """Test that a test with expected output assertions stays under 15ms."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        times = []
        for i in range(200):
            tc = TestCaseInput(
                name=f"assert_test_{i}", agent_type="FastAgent",
                input_data={"x": i},
                expected_output={"result": 42},
            )
            start = time.perf_counter()
            result = runner.run_test(tc)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert result.status == TestStatus.PASSED

        avg_ms = sum(times) / len(times)
        assert avg_ms < 15.0, f"Avg test-with-assertions {avg_ms:.3f}ms exceeds 15ms target"

    def test_skip_test_under_1ms(self):
        """Test that skipped tests resolve nearly instantly."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        times = []
        for i in range(500):
            tc = TestCaseInput(name=f"skip_{i}", agent_type="FastAgent",
                               skip=True, skip_reason="perf test")
            start = time.perf_counter()
            result = runner.run_test(tc)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert result.status == TestStatus.SKIPPED

        avg_ms = sum(times) / len(times)
        assert avg_ms < 1.0, f"Average skip time {avg_ms:.3f}ms exceeds 1ms target"


class TestSuiteThroughput:
    """Test suite execution throughput."""

    def test_suite_100_tests_under_5s(self):
        """Test that a 100-test suite completes in under 5 seconds."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        tests = [
            TestCaseInput(name=f"suite_test_{i}", agent_type="FastAgent",
                          input_data={"idx": i})
            for i in range(100)
        ]
        suite = TestSuiteInput(name="Load100", test_cases=tests, parallel=False)

        start = time.perf_counter()
        result = runner.run_suite(suite)
        elapsed_s = time.perf_counter() - start

        assert result.total_tests == 100
        assert result.passed == 100
        assert elapsed_s < 5.0, f"Suite of 100 took {elapsed_s:.3f}s, exceeds 5s target"

    def test_suite_500_tests_throughput(self):
        """Test throughput target: >100 tests/second for a 500-test suite."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        tests = [
            TestCaseInput(name=f"tp_test_{i}", agent_type="FastAgent",
                          input_data={"idx": i})
            for i in range(500)
        ]
        suite = TestSuiteInput(name="Load500", test_cases=tests, parallel=False)

        start = time.perf_counter()
        result = runner.run_suite(suite)
        elapsed_s = time.perf_counter() - start

        throughput = 500 / elapsed_s
        assert result.total_tests == 500
        assert throughput > 100, f"Throughput {throughput:.0f}/s below 100/s target"

    def test_suite_1000_tests_completes(self):
        """Test that a 1000-test suite completes within 30 seconds."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        tests = [
            TestCaseInput(name=f"big_test_{i}", agent_type="FastAgent",
                          input_data={"idx": i})
            for i in range(1000)
        ]
        suite = TestSuiteInput(name="Load1000", test_cases=tests, parallel=False)

        start = time.perf_counter()
        result = runner.run_suite(suite)
        elapsed_s = time.perf_counter() - start

        assert result.total_tests == 1000
        assert result.passed == 1000
        assert elapsed_s < 30.0, f"Suite of 1000 took {elapsed_s:.3f}s, exceeds 30s target"

    def test_suite_mixed_agents_throughput(self):
        """Test throughput with mixed agent types."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)
        runner.register_agent("MediumAgent", _medium_agent)
        runner.register_agent("HeavyAgent", _heavy_agent)

        agents = ["FastAgent", "MediumAgent", "HeavyAgent"]
        tests = [
            TestCaseInput(
                name=f"mixed_{i}", agent_type=agents[i % 3],
                input_data={"value": float(i)},
            )
            for i in range(300)
        ]
        suite = TestSuiteInput(name="Mixed300", test_cases=tests, parallel=False)

        start = time.perf_counter()
        result = runner.run_suite(suite)
        elapsed_s = time.perf_counter() - start

        throughput = 300 / elapsed_s
        assert result.total_tests == 300
        assert throughput > 50, f"Mixed throughput {throughput:.0f}/s below 50/s target"


class TestConcurrentSuiteExecution:
    """Test concurrent suite execution across threads."""

    def test_concurrent_suites_10_threads(self):
        """Test running 10 suites concurrently across threads."""
        results = []

        def run_suite_in_thread(suite_idx):
            runner = LoadTestRunner()
            runner.register_agent("FastAgent", _fast_agent)
            tests = [
                TestCaseInput(
                    name=f"s{suite_idx}_t{i}", agent_type="FastAgent",
                    input_data={"suite": suite_idx, "test": i},
                )
                for i in range(20)
            ]
            suite = TestSuiteInput(
                name=f"ConcurrentSuite_{suite_idx}",
                test_cases=tests, parallel=False,
            )
            return runner.run_suite(suite)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_suite_in_thread, i) for i in range(10)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 10
        total_tests = sum(r.total_tests for r in results)
        assert total_tests == 200
        assert all(r.passed == 20 for r in results)
        assert elapsed_s < 30.0, f"10 concurrent suites took {elapsed_s:.3f}s"

    def test_concurrent_individual_tests_50_threads(self):
        """Test running 50 individual tests concurrently."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)
        results = []

        def run_one(idx):
            tc = TestCaseInput(
                name=f"conc_test_{idx}", agent_type="FastAgent",
                input_data={"idx": idx},
            )
            return runner.run_test(tc)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(run_one, i) for i in range(200)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 200
        passed_count = sum(1 for r in results if r.status == TestStatus.PASSED)
        assert passed_count == 200
        assert elapsed_s < 30.0, f"200 concurrent tests took {elapsed_s:.3f}s"


class TestAssertionThroughput:
    """Test assertion checking throughput."""

    def test_zero_hallucination_throughput(self):
        """Test zero-hallucination checks at >500/second."""
        engine = LoadAssertionEngine()
        input_data = {"emissions": 100.5, "fuel_type": "diesel", "quantity": 1000}
        output_data = {
            "total_emissions": 2680.0,
            "unit": "kg_co2e",
            "provenance_id": "prov-abc12345",
        }

        start = time.perf_counter()
        for _ in range(1000):
            engine.check_zero_hallucination(input_data, output_data)
        elapsed_s = time.perf_counter() - start

        throughput = 1000 / elapsed_s
        assert throughput > 500, (
            f"Zero-hallucination throughput {throughput:.0f}/s below 500/s target"
        )

    def test_determinism_check_throughput(self):
        """Test determinism checks at >100/second."""
        engine = LoadAssertionEngine()

        start = time.perf_counter()
        for i in range(200):
            engine.check_determinism(
                _fast_agent, {"x": i}, iterations=3,
            )
        elapsed_s = time.perf_counter() - start

        throughput = 200 / elapsed_s
        assert throughput > 100, (
            f"Determinism throughput {throughput:.0f}/s below 100/s target"
        )

    def test_number_extraction_large_payload(self):
        """Test number extraction from large payloads stays performant."""
        engine = LoadAssertionEngine()
        large_data = {f"field_{i}": float(i) * 1.1 for i in range(500)}

        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine._extract_numbers(large_data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 5.0, f"Number extraction avg {avg_ms:.3f}ms exceeds 5ms target"


class TestHashComputationUnderLoad:
    """Test hash computation performance under load."""

    def test_hash_computation_1000_ops(self):
        """Test computing 1000 hashes in under 1 second."""
        data_items = [{"id": i, "value": float(i) * 1.1} for i in range(1000)]

        start = time.perf_counter()
        hashes = [_compute_hash(d) for d in data_items]
        elapsed_s = time.perf_counter() - start

        assert len(hashes) == 1000
        assert elapsed_s < 1.0, f"1000 hashes took {elapsed_s:.3f}s, exceeds 1s target"

    def test_hash_determinism_under_load(self):
        """Test that hash computation remains deterministic under load."""
        data = {"emissions": 100.5, "fuel_type": "diesel"}

        hashes = set()
        for _ in range(10000):
            hashes.add(_compute_hash(data))

        assert len(hashes) == 1, f"Hash produced {len(hashes)} unique values under load"

    def test_concurrent_hash_computation(self):
        """Test hash computation across 20 concurrent threads."""
        data_items = [{"id": i, "v": float(i)} for i in range(1000)]
        results = []

        def hash_batch(batch):
            return [_compute_hash(d) for d in batch]

        batch_size = 50
        batches = [data_items[i:i + batch_size]
                   for i in range(0, len(data_items), batch_size)]

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(hash_batch, b) for b in batches]
            for f in as_completed(futures):
                results.extend(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 1000
        assert elapsed_s < 5.0, f"Concurrent hashing took {elapsed_s:.3f}s"


class TestRunnerHistoryScalability:
    """Test that runner history tracking scales well."""

    def test_history_tracks_10000_runs(self):
        """Test that history correctly tracks 10000 test runs."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        for i in range(10000):
            tc = TestCaseInput(name=f"hist_test_{i}", agent_type="FastAgent",
                               input_data={"i": i})
            runner.run_test(tc)

        assert runner.total_runs == 10000

    def test_history_no_data_loss_under_load(self):
        """Test no data loss when running many tests quickly."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        expected_count = 5000
        for i in range(expected_count):
            tc = TestCaseInput(name=f"loss_test_{i}", agent_type="FastAgent")
            runner.run_test(tc)

        assert runner.total_runs == expected_count


class TestSuiteScalability:
    """Test suite execution at scale."""

    def test_large_suite_pass_rate_accuracy(self):
        """Test that pass rate is computed accurately in large suites."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        # 80% pass, 20% fail (unregistered agent causes ERROR)
        tests = []
        for i in range(500):
            if i % 5 == 0:
                tests.append(TestCaseInput(
                    name=f"fail_{i}", agent_type="UnknownAgent",
                    input_data={"i": i},
                ))
            else:
                tests.append(TestCaseInput(
                    name=f"pass_{i}", agent_type="FastAgent",
                    input_data={"i": i},
                ))

        suite = TestSuiteInput(name="ScaleTest", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)

        assert result.total_tests == 500
        assert result.passed == 400
        assert result.errors == 100
        assert result.pass_rate == 80.0

    def test_sustained_test_execution(self):
        """Test sustained test execution for 2 seconds."""
        runner = LoadTestRunner()
        runner.register_agent("FastAgent", _fast_agent)

        duration = 2.0  # seconds
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < duration:
            tc = TestCaseInput(
                name=f"sustained_{count}", agent_type="FastAgent",
                input_data={"c": count},
            )
            runner.run_test(tc)
            count += 1

        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        assert throughput > 100, f"Sustained throughput {throughput:.0f}/s below 100/s"
        assert count > 0
        assert runner.total_runs == count
