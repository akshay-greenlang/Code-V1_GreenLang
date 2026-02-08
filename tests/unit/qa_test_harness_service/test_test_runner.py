# -*- coding: utf-8 -*-
"""
Unit Tests for TestRunner (AGENT-FOUND-009)

Tests test execution, suite orchestration, parallel/sequential modes,
timeout handling, fail-fast, tag filtering, hash computation, and history.

Coverage target: 85%+ of test_runner.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs
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
    def __init__(self, name, passed, message="", severity=SeverityLevel.HIGH, **kw):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity


class TestCaseInput:
    def __init__(self, name, agent_type, test_id="", category=TestCategory.UNIT,
                 input_data=None, expected_output=None, golden_file_path=None,
                 timeout_seconds=60, tags=None, skip=False, skip_reason="",
                 severity=SeverityLevel.HIGH, **kw):
        self.test_id = test_id or str(uuid.uuid4())
        self.name = name
        self.category = category
        self.agent_type = agent_type
        self.input_data = input_data or {}
        self.expected_output = expected_output
        self.golden_file_path = golden_file_path
        self.timeout_seconds = timeout_seconds
        self.tags = tags or []
        self.skip = skip
        self.skip_reason = skip_reason
        self.severity = severity


class TestCaseResult:
    def __init__(self, test_id, name, category, status, assertions=None,
                 duration_ms=0.0, started_at=None, completed_at=None,
                 error_message=None, agent_result=None, input_hash="",
                 output_hash="", metadata=None, **kw):
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
                 tags_exclude=None, **kw):
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
                 duration_ms=0.0, pass_rate=0.0, provenance_hash="", **kw):
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
# Inline TestRunner mirroring greenlang/qa_test_harness/test_runner.py
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class TestRunner:
    """Simulates the QA test runner."""

    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._history: List[TestCaseResult] = []

    def register_agent(self, agent_type: str, agent_fn: Callable) -> None:
        self._agents[agent_type] = agent_fn

    def run_test(self, test_input: TestCaseInput) -> TestCaseResult:
        start_time = time.time()
        started_at = datetime.now(timezone.utc)
        input_hash = _compute_hash(test_input.input_data)

        # Skip
        if test_input.skip:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.SKIPPED,
                input_hash=input_hash, metadata={"skip_reason": test_input.skip_reason},
                started_at=started_at,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._history.append(result)
            return result

        # Unknown agent
        agent_fn = self._agents.get(test_input.agent_type)
        if agent_fn is None:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.ERROR,
                input_hash=input_hash,
                error_message=f"Agent type not registered: {test_input.agent_type}",
                started_at=started_at,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._history.append(result)
            return result

        # Execute
        try:
            # Simulate timeout
            if test_input.timeout_seconds <= 0:
                raise TimeoutError("Test timed out")

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
                duration_ms=(time.time() - start_time) * 1000,
            )

        except TimeoutError:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.TIMEOUT,
                input_hash=input_hash,
                error_message=f"Test timed out after {test_input.timeout_seconds}s",
                started_at=started_at,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            result = TestCaseResult(
                test_id=test_input.test_id, name=test_input.name,
                category=test_input.category, status=TestStatus.ERROR,
                input_hash=input_hash, error_message=str(e),
                started_at=started_at,
                duration_ms=(time.time() - start_time) * 1000,
            )

        self._history.append(result)
        return result

    def run_suite(self, suite_input: TestSuiteInput) -> TestSuiteResult:
        tests_to_run = self._filter_tests(suite_input)

        if suite_input.parallel and len(tests_to_run) > 1:
            results = self._run_parallel(tests_to_run, suite_input.max_workers,
                                         suite_input.fail_fast)
        else:
            results = self._run_sequential(tests_to_run, suite_input.fail_fast)

        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        total = len(results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        if failed > 0 or errors > 0:
            overall_status = TestStatus.FAILED
        elif skipped == total:
            overall_status = TestStatus.SKIPPED
        else:
            overall_status = TestStatus.PASSED

        return TestSuiteResult(
            suite_id=suite_input.suite_id, name=suite_input.name,
            status=overall_status, test_results=results,
            total_tests=total, passed=passed, failed=failed,
            skipped=skipped, errors=errors, pass_rate=round(pass_rate, 2),
        )

    def _filter_tests(self, suite_input: TestSuiteInput) -> List[TestCaseInput]:
        filtered = []
        for test in suite_input.test_cases:
            if suite_input.tags_include:
                if not any(t in test.tags for t in suite_input.tags_include):
                    continue
            if suite_input.tags_exclude:
                if any(t in test.tags for t in suite_input.tags_exclude):
                    continue
            filtered.append(test)
        return filtered

    def _run_sequential(self, tests, fail_fast):
        results = []
        for test in tests:
            result = self.run_test(test)
            results.append(result)
            if fail_fast and result.status in (TestStatus.FAILED, TestStatus.ERROR):
                for remaining in tests[tests.index(test) + 1:]:
                    results.append(TestCaseResult(
                        test_id=remaining.test_id, name=remaining.name,
                        category=remaining.category, status=TestStatus.SKIPPED,
                        metadata={"skip_reason": "fail_fast"},
                    ))
                break
        return results

    def _run_parallel(self, tests, max_workers, fail_fast):
        # Simulate parallel by running sequentially for testing
        return self._run_sequential(tests, fail_fast)

    def get_run_history(self) -> List[TestCaseResult]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Mock agent functions
# ---------------------------------------------------------------------------

def _success_agent(input_data):
    return {"success": True, "data": {"result": 42}, "error": None}


def _failure_agent(input_data):
    return {"success": False, "data": {}, "error": "Processing failed"}


def _error_agent(input_data):
    raise RuntimeError("Agent crashed")


def _slow_agent(input_data):
    time.sleep(0.01)
    return {"success": True, "data": {"result": "slow"}, "error": None}


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def runner():
    r = TestRunner()
    r.register_agent("SuccessAgent", _success_agent)
    r.register_agent("FailureAgent", _failure_agent)
    r.register_agent("ErrorAgent", _error_agent)
    r.register_agent("SlowAgent", _slow_agent)
    return r


class TestRunTestSuccess:
    def test_run_test_success(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert result.status == TestStatus.PASSED

    def test_run_test_has_assertions(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert len(result.assertions) >= 1

    def test_run_test_has_output_hash(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert result.output_hash != ""

    def test_run_test_has_input_hash(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert result.input_hash != ""

    def test_run_test_has_started_at(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert result.started_at is not None

    def test_run_test_duration_positive(self, runner):
        tc = TestCaseInput(name="test_success", agent_type="SuccessAgent")
        result = runner.run_test(tc)
        assert result.duration_ms >= 0


class TestRunTestFailure:
    def test_run_test_failure(self, runner):
        tc = TestCaseInput(name="test_failure", agent_type="FailureAgent")
        result = runner.run_test(tc)
        assert result.status == TestStatus.FAILED

    def test_run_test_failure_has_failed_assertion(self, runner):
        tc = TestCaseInput(name="test_failure", agent_type="FailureAgent")
        result = runner.run_test(tc)
        failed_assertions = [a for a in result.assertions if not a.passed]
        assert len(failed_assertions) >= 1


class TestRunTestSkip:
    def test_run_test_skip(self, runner):
        tc = TestCaseInput(name="test_skip", agent_type="SuccessAgent",
                           skip=True, skip_reason="Not ready")
        result = runner.run_test(tc)
        assert result.status == TestStatus.SKIPPED

    def test_run_test_skip_has_reason(self, runner):
        tc = TestCaseInput(name="test_skip", agent_type="SuccessAgent",
                           skip=True, skip_reason="Not ready")
        result = runner.run_test(tc)
        assert result.metadata.get("skip_reason") == "Not ready"


class TestRunTestTimeout:
    def test_run_test_timeout(self, runner):
        tc = TestCaseInput(name="test_timeout", agent_type="SlowAgent",
                           timeout_seconds=0)
        result = runner.run_test(tc)
        assert result.status == TestStatus.TIMEOUT

    def test_run_test_timeout_has_error_message(self, runner):
        tc = TestCaseInput(name="test_timeout", agent_type="SlowAgent",
                           timeout_seconds=0)
        result = runner.run_test(tc)
        assert "timed out" in result.error_message.lower()


class TestRunTestError:
    def test_run_test_error(self, runner):
        tc = TestCaseInput(name="test_error", agent_type="ErrorAgent")
        result = runner.run_test(tc)
        assert result.status == TestStatus.ERROR

    def test_run_test_error_has_message(self, runner):
        tc = TestCaseInput(name="test_error", agent_type="ErrorAgent")
        result = runner.run_test(tc)
        assert result.error_message is not None
        assert "crashed" in result.error_message.lower()


class TestRunTestUnknownAgent:
    def test_run_test_unknown_agent(self, runner):
        tc = TestCaseInput(name="test_unknown", agent_type="NonexistentAgent")
        result = runner.run_test(tc)
        assert result.status == TestStatus.ERROR

    def test_run_test_unknown_agent_error_message(self, runner):
        tc = TestCaseInput(name="test_unknown", agent_type="NonexistentAgent")
        result = runner.run_test(tc)
        assert "not registered" in result.error_message.lower()


class TestRunSuite:
    def test_run_suite_all_pass(self, runner):
        tests = [
            TestCaseInput(name=f"test_{i}", agent_type="SuccessAgent")
            for i in range(3)
        ]
        suite = TestSuiteInput(name="AllPass", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.status == TestStatus.PASSED
        assert result.passed == 3
        assert result.failed == 0

    def test_run_suite_with_failures(self, runner):
        tests = [
            TestCaseInput(name="pass_test", agent_type="SuccessAgent"),
            TestCaseInput(name="fail_test", agent_type="FailureAgent"),
        ]
        suite = TestSuiteInput(name="Mixed", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.status == TestStatus.FAILED
        assert result.passed == 1
        assert result.failed == 1

    def test_run_suite_pass_rate(self, runner):
        tests = [
            TestCaseInput(name="p1", agent_type="SuccessAgent"),
            TestCaseInput(name="p2", agent_type="SuccessAgent"),
            TestCaseInput(name="f1", agent_type="FailureAgent"),
        ]
        suite = TestSuiteInput(name="RateTest", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.pass_rate == pytest.approx(66.67, rel=0.01)

    def test_run_suite_total_tests(self, runner):
        tests = [
            TestCaseInput(name=f"test_{i}", agent_type="SuccessAgent")
            for i in range(5)
        ]
        suite = TestSuiteInput(name="TotalTest", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.total_tests == 5

    def test_run_suite_empty(self, runner):
        suite = TestSuiteInput(name="Empty", test_cases=[], parallel=False)
        result = runner.run_suite(suite)
        assert result.total_tests == 0

    def test_run_suite_all_skipped(self, runner):
        tests = [
            TestCaseInput(name=f"test_{i}", agent_type="SuccessAgent", skip=True)
            for i in range(3)
        ]
        suite = TestSuiteInput(name="AllSkipped", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.status == TestStatus.SKIPPED
        assert result.skipped == 3


class TestRunSuiteParallel:
    def test_run_suite_parallel(self, runner):
        tests = [
            TestCaseInput(name=f"test_{i}", agent_type="SuccessAgent")
            for i in range(4)
        ]
        suite = TestSuiteInput(name="Parallel", test_cases=tests, parallel=True,
                               max_workers=2)
        result = runner.run_suite(suite)
        assert result.passed == 4


class TestRunSuiteSequential:
    def test_run_suite_sequential(self, runner):
        tests = [
            TestCaseInput(name=f"test_{i}", agent_type="SuccessAgent")
            for i in range(3)
        ]
        suite = TestSuiteInput(name="Sequential", test_cases=tests, parallel=False)
        result = runner.run_suite(suite)
        assert result.passed == 3


class TestRunSuiteFailFast:
    def test_run_suite_fail_fast(self, runner):
        tests = [
            TestCaseInput(name="pass_first", agent_type="SuccessAgent"),
            TestCaseInput(name="fail_here", agent_type="FailureAgent"),
            TestCaseInput(name="skip_this", agent_type="SuccessAgent"),
        ]
        suite = TestSuiteInput(name="FailFast", test_cases=tests, parallel=False,
                               fail_fast=True)
        result = runner.run_suite(suite)
        assert result.failed >= 1
        skipped_results = [r for r in result.test_results if r.status == TestStatus.SKIPPED]
        assert len(skipped_results) >= 1

    def test_run_suite_fail_fast_stops_early(self, runner):
        tests = [
            TestCaseInput(name="error_first", agent_type="ErrorAgent"),
            TestCaseInput(name="never_runs", agent_type="SuccessAgent"),
            TestCaseInput(name="never_runs_2", agent_type="SuccessAgent"),
        ]
        suite = TestSuiteInput(name="FailFastError", test_cases=tests, parallel=False,
                               fail_fast=True)
        result = runner.run_suite(suite)
        assert result.total_tests == 3
        assert result.errors >= 1


class TestFilterTests:
    def test_filter_tests_include_tags(self, runner):
        tests = [
            TestCaseInput(name="smoke1", agent_type="SuccessAgent", tags=["smoke"]),
            TestCaseInput(name="unit1", agent_type="SuccessAgent", tags=["unit"]),
            TestCaseInput(name="smoke2", agent_type="SuccessAgent", tags=["smoke"]),
        ]
        suite = TestSuiteInput(name="FilterInclude", test_cases=tests,
                               tags_include=["smoke"], parallel=False)
        result = runner.run_suite(suite)
        assert result.total_tests == 2

    def test_filter_tests_exclude_tags(self, runner):
        tests = [
            TestCaseInput(name="smoke1", agent_type="SuccessAgent", tags=["smoke"]),
            TestCaseInput(name="slow1", agent_type="SuccessAgent", tags=["slow"]),
            TestCaseInput(name="smoke2", agent_type="SuccessAgent", tags=["smoke"]),
        ]
        suite = TestSuiteInput(name="FilterExclude", test_cases=tests,
                               tags_exclude=["slow"], parallel=False)
        result = runner.run_suite(suite)
        assert result.total_tests == 2

    def test_filter_tests_include_and_exclude(self, runner):
        tests = [
            TestCaseInput(name="t1", agent_type="SuccessAgent", tags=["smoke", "slow"]),
            TestCaseInput(name="t2", agent_type="SuccessAgent", tags=["smoke"]),
            TestCaseInput(name="t3", agent_type="SuccessAgent", tags=["unit"]),
        ]
        suite = TestSuiteInput(name="FilterBoth", test_cases=tests,
                               tags_include=["smoke"], tags_exclude=["slow"],
                               parallel=False)
        result = runner.run_suite(suite)
        assert result.total_tests == 1


class TestComputeHash:
    def test_compute_hash_deterministic(self):
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_compute_hash_different_data(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_length(self):
        h = _compute_hash({"key": "value"})
        assert len(h) == 16


class TestGetRunHistory:
    def test_get_run_history_empty(self, runner):
        assert runner.get_run_history() == []

    def test_get_run_history_after_runs(self, runner):
        tc1 = TestCaseInput(name="test_1", agent_type="SuccessAgent")
        tc2 = TestCaseInput(name="test_2", agent_type="FailureAgent")
        runner.run_test(tc1)
        runner.run_test(tc2)
        history = runner.get_run_history()
        assert len(history) == 2

    def test_get_run_history_includes_suite_tests(self, runner):
        tests = [
            TestCaseInput(name=f"t{i}", agent_type="SuccessAgent")
            for i in range(3)
        ]
        suite = TestSuiteInput(name="Suite", test_cases=tests, parallel=False)
        runner.run_suite(suite)
        history = runner.get_run_history()
        assert len(history) == 3


class TestRunTestWithExpectedOutput:
    def test_expected_output_match(self, runner):
        tc = TestCaseInput(name="test_match", agent_type="SuccessAgent",
                           expected_output={"result": 42})
        result = runner.run_test(tc)
        assert result.status == TestStatus.PASSED

    def test_expected_output_mismatch(self, runner):
        tc = TestCaseInput(name="test_mismatch", agent_type="SuccessAgent",
                           expected_output={"result": 999})
        result = runner.run_test(tc)
        assert result.status == TestStatus.FAILED
