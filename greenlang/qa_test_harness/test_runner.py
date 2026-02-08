# -*- coding: utf-8 -*-
"""
Test Execution Engine for QA Test Harness - AGENT-FOUND-009

Provides the core test execution engine that runs individual test cases
and test suites, with parallel and sequential execution modes, timeout
enforcement, and comprehensive result tracking.

Zero-Hallucination Guarantees:
    - All test results use deterministic comparison
    - No LLM-generated expected values
    - Complete provenance hash for every execution
    - SHA-256 hashing for input/output audit trails

Example:
    >>> from greenlang.qa_test_harness.test_runner import TestRunner
    >>> runner = TestRunner(config, assertion_engine)
    >>> result = runner.run_test(test_input, agent_registry)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from greenlang.qa_test_harness.config import QATestHarnessConfig
from greenlang.qa_test_harness.assertion_engine import AssertionEngine
from greenlang.qa_test_harness.models import (
    TestAssertion,
    TestCaseInput,
    TestCaseResult,
    TestSuiteInput,
    TestSuiteResult,
    TestRun,
    TestStatus,
    TestCategory,
)
from greenlang.qa_test_harness.metrics import (
    record_test_run,
    record_assertion,
    record_suite,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class TestRunner:
    """Test execution engine for individual tests and suites.

    Manages the lifecycle of test execution including agent instantiation,
    timeout enforcement, assertion evaluation, and result compilation.

    Attributes:
        config: QA test harness configuration.
        assertion_engine: Assertion engine for evaluating test assertions.
        _run_history: In-memory history of test runs.
        _test_count: Total number of tests executed.
        _pass_count: Number of tests passed.
        _fail_count: Number of tests failed.

    Example:
        >>> runner = TestRunner(config, assertion_engine)
        >>> result = runner.run_test(test_input, agent_registry)
    """

    def __init__(
        self,
        config: QATestHarnessConfig,
        assertion_engine: AssertionEngine,
    ) -> None:
        """Initialize TestRunner.

        Args:
            config: QA test harness configuration.
            assertion_engine: Assertion engine instance.
        """
        self.config = config
        self.assertion_engine = assertion_engine

        # Run history for regression detection and statistics
        self._run_history: List[TestRun] = []
        self._test_count: int = 0
        self._pass_count: int = 0
        self._fail_count: int = 0

        logger.info("TestRunner initialized")

    def run_test(
        self,
        test_input: TestCaseInput,
        agent_registry: Dict[str, Type[Any]],
    ) -> TestCaseResult:
        """Run a single test case.

        Args:
            test_input: Test case specification.
            agent_registry: Mapping of agent_type to agent class.

        Returns:
            TestCaseResult with test outcome.
        """
        self._test_count += 1
        started_at = _utcnow()
        start_time = time.time()

        result = TestCaseResult(
            test_id=test_input.test_id,
            name=test_input.name,
            category=test_input.category,
            status=TestStatus.RUNNING,
            started_at=started_at,
            input_hash=self._compute_hash(test_input.input_data),
        )

        logger.info(
            "Running test: %s [%s]",
            test_input.name, test_input.category.value,
        )

        # Handle skip
        if test_input.skip:
            result.status = TestStatus.SKIPPED
            result.completed_at = _utcnow()
            result.duration_ms = (time.time() - start_time) * 1000
            result.metadata["skip_reason"] = test_input.skip_reason
            self._record_run(test_input, result)
            return result

        try:
            # Get agent class
            agent_class = agent_registry.get(test_input.agent_type)
            if not agent_class:
                raise ValueError(
                    f"Agent type not registered: {test_input.agent_type}"
                )

            # Create agent instance
            agent = agent_class()

            # Execute with timeout
            timeout = test_input.timeout_seconds or self.config.default_timeout_seconds
            agent_result = self._execute_with_timeout(
                agent, test_input.input_data, timeout,
            )

            result.agent_result = (
                agent_result.model_dump() if agent_result else None
            )
            result.output_hash = self._compute_hash(
                agent_result.data if agent_result else {}
            )

            # Run category-specific assertions
            assertions = self.assertion_engine.run_assertions(
                test_input, agent_result, agent,
            )
            result.assertions = assertions

            # Record individual assertion metrics
            for a in assertions:
                record_assertion("passed" if a.passed else "failed")

            # Determine status
            if all(a.passed for a in assertions):
                result.status = TestStatus.PASSED
                self._pass_count += 1
            else:
                result.status = TestStatus.FAILED
                self._fail_count += 1
                failed_assertions = [a for a in assertions if not a.passed]
                result.error_message = (
                    f"{len(failed_assertions)} assertion(s) failed"
                )

        except TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error_message = (
                f"Test timed out after {test_input.timeout_seconds}s"
            )
            self._fail_count += 1

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            self._fail_count += 1
            logger.error("Test error: %s", e, exc_info=True)

        # Finalize timing
        result.completed_at = _utcnow()
        result.duration_ms = (time.time() - start_time) * 1000

        # Record metrics
        record_test_run(
            status=result.status.value,
            category=test_input.category.value,
            duration_seconds=result.duration_ms / 1000,
        )

        # Store in history
        self._record_run(test_input, result)

        logger.info(
            "Test completed: %s - %s (%.2fms)",
            test_input.name, result.status.value, result.duration_ms,
        )

        return result

    def run_suite(
        self,
        suite_input: TestSuiteInput,
        agent_registry: Dict[str, Type[Any]],
    ) -> TestSuiteResult:
        """Run a test suite.

        Args:
            suite_input: Test suite specification.
            agent_registry: Mapping of agent_type to agent class.

        Returns:
            TestSuiteResult with all test outcomes.
        """
        started_at = _utcnow()
        start_time = time.time()

        logger.info("Starting test suite: %s", suite_input.name)

        # Filter tests by tags
        tests_to_run = self._filter_tests(suite_input)

        # Run tests
        max_workers = suite_input.max_workers or self.config.max_parallel_workers
        fail_fast = suite_input.fail_fast or self.config.fail_fast

        if suite_input.parallel and len(tests_to_run) > 1:
            results = self._run_parallel(
                tests_to_run, agent_registry, max_workers, fail_fast,
            )
        else:
            results = self._run_sequential(
                tests_to_run, agent_registry, fail_fast,
            )

        # Calculate statistics
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(
            1 for r in results
            if r.status in (TestStatus.ERROR, TestStatus.TIMEOUT)
        )

        total = len(results)
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        # Determine overall status
        if failed > 0 or errors > 0:
            overall_status = TestStatus.FAILED
        elif skipped == total:
            overall_status = TestStatus.SKIPPED
        else:
            overall_status = TestStatus.PASSED

        completed_at = _utcnow()
        duration_ms = (time.time() - start_time) * 1000

        # Build result
        suite_result = TestSuiteResult(
            suite_id=suite_input.suite_id,
            name=suite_input.name,
            status=overall_status,
            test_results=results,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_ms=duration_ms,
            started_at=started_at,
            completed_at=completed_at,
            pass_rate=round(pass_rate, 2),
            provenance_hash=self._compute_provenance_hash(suite_input, results),
        )

        # Record suite metric
        record_suite()

        logger.info(
            "Suite completed: %s - %d/%d passed (%.1f%%) in %.2fms",
            suite_input.name, passed, total, pass_rate, duration_ms,
        )

        return suite_result

    def get_run_history(
        self,
        agent_type: Optional[str] = None,
    ) -> List[TestRun]:
        """Get test run history, optionally filtered by agent type.

        Args:
            agent_type: Optional agent type filter.

        Returns:
            List of test run records.
        """
        if agent_type:
            return [
                r for r in self._run_history
                if r.agent_type == agent_type
            ]
        return list(self._run_history)

    def get_run(self, run_id: str) -> Optional[TestRun]:
        """Get a specific test run by ID.

        Args:
            run_id: Test run identifier.

        Returns:
            TestRun if found, None otherwise.
        """
        for run in self._run_history:
            if run.run_id == run_id:
                return run
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        tests: List[TestCaseInput],
        agent_registry: Dict[str, Type[Any]],
        max_workers: int,
        fail_fast: bool,
    ) -> List[TestCaseResult]:
        """Run tests in parallel using thread pool.

        Args:
            tests: List of test cases to run.
            agent_registry: Mapping of agent_type to agent class.
            max_workers: Maximum number of parallel workers.
            fail_fast: Whether to stop on first failure.

        Returns:
            List of test case results.
        """
        results: List[TestCaseResult] = []
        failed = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(self.run_test, test, agent_registry): test
                for test in tests
            }

            for future in concurrent.futures.as_completed(future_to_test):
                if fail_fast and failed:
                    future.cancel()
                    continue

                try:
                    result = future.result()
                    results.append(result)

                    if result.status in (TestStatus.FAILED, TestStatus.ERROR):
                        failed = True

                except Exception as e:
                    test = future_to_test[future]
                    results.append(TestCaseResult(
                        test_id=test.test_id,
                        name=test.name,
                        category=test.category,
                        status=TestStatus.ERROR,
                        error_message=str(e),
                    ))
                    failed = True

        return results

    def _run_sequential(
        self,
        tests: List[TestCaseInput],
        agent_registry: Dict[str, Type[Any]],
        fail_fast: bool,
    ) -> List[TestCaseResult]:
        """Run tests sequentially.

        Args:
            tests: List of test cases to run.
            agent_registry: Mapping of agent_type to agent class.
            fail_fast: Whether to stop on first failure.

        Returns:
            List of test case results.
        """
        results: List[TestCaseResult] = []

        for i, test in enumerate(tests):
            result = self.run_test(test, agent_registry)
            results.append(result)

            if fail_fast and result.status in (TestStatus.FAILED, TestStatus.ERROR):
                # Skip remaining tests
                for remaining in tests[i + 1:]:
                    results.append(TestCaseResult(
                        test_id=remaining.test_id,
                        name=remaining.name,
                        category=remaining.category,
                        status=TestStatus.SKIPPED,
                        metadata={"skip_reason": "fail_fast"},
                    ))
                break

        return results

    def _execute_with_timeout(
        self,
        agent: Any,
        input_data: Dict[str, Any],
        timeout_seconds: int,
    ) -> Any:
        """Execute agent with timeout enforcement.

        Args:
            agent: Agent instance to execute.
            input_data: Input data for the agent.
            timeout_seconds: Timeout in seconds.

        Returns:
            AgentResult from the agent.

        Raises:
            TimeoutError: If agent execution exceeds timeout.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.run, input_data)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Agent execution timed out after {timeout_seconds}s"
                )

    def _filter_tests(
        self,
        suite_input: TestSuiteInput,
    ) -> List[TestCaseInput]:
        """Filter tests based on include/exclude tags.

        Args:
            suite_input: Suite specification with tag filters.

        Returns:
            Filtered list of test cases.
        """
        filtered: List[TestCaseInput] = []

        for test in suite_input.test_cases:
            # Check include tags
            if suite_input.tags_include:
                if not any(tag in test.tags for tag in suite_input.tags_include):
                    continue

            # Check exclude tags
            if suite_input.tags_exclude:
                if any(tag in test.tags for tag in suite_input.tags_exclude):
                    continue

            filtered.append(test)

        return filtered

    def _compute_hash(self, data: Any) -> str:
        """Compute deterministic SHA-256 hash of data.

        Args:
            data: Data to hash.

        Returns:
            Hex-encoded SHA-256 hash (first 16 chars).
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _compute_provenance_hash(
        self,
        suite_input: TestSuiteInput,
        results: List[TestCaseResult],
    ) -> str:
        """Compute provenance hash for audit trail.

        Args:
            suite_input: Suite specification.
            results: List of test case results.

        Returns:
            Full SHA-256 hex digest.
        """
        provenance_data = {
            "suite_id": suite_input.suite_id,
            "suite_name": suite_input.name,
            "test_count": len(results),
            "result_hashes": [r.output_hash for r in results],
            "timestamp": _utcnow().isoformat(),
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _record_run(
        self,
        test_input: TestCaseInput,
        result: TestCaseResult,
    ) -> None:
        """Record a test run in history.

        Args:
            test_input: Test case specification.
            result: Test case result.
        """
        run = TestRun(
            test_case_id=test_input.test_id,
            agent_type=test_input.agent_type,
            category=test_input.category,
            status=result.status,
            assertions=[
                a.model_dump() for a in result.assertions
            ],
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            duration_ms=result.duration_ms,
            error_message=result.error_message,
        )
        self._run_history.append(run)

    @property
    def test_count(self) -> int:
        """Return total number of tests executed."""
        return self._test_count

    @property
    def pass_count(self) -> int:
        """Return number of tests passed."""
        return self._pass_count

    @property
    def fail_count(self) -> int:
        """Return number of tests failed."""
        return self._fail_count


__all__ = [
    "TestRunner",
]
