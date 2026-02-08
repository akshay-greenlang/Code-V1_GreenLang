# -*- coding: utf-8 -*-
"""
QA Test Harness Service Setup - AGENT-FOUND-009: QA Test Harness

Provides ``configure_qa_test_harness(app)`` which wires up the
QA Test Harness SDK (test runner, assertion engine, golden file manager,
regression detector, performance benchmarker, coverage tracker, report
generator, provenance tracker) and mounts the REST API.

Also exposes ``get_qa_test_harness(app)`` for programmatic access
and the ``QATestHarnessService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.qa_test_harness.setup import configure_qa_test_harness
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_qa_test_harness(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from greenlang.qa_test_harness.config import QATestHarnessConfig, get_config
from greenlang.qa_test_harness.assertion_engine import AssertionEngine
from greenlang.qa_test_harness.test_runner import TestRunner
from greenlang.qa_test_harness.golden_file_manager import GoldenFileManager
from greenlang.qa_test_harness.regression_detector import RegressionDetector
from greenlang.qa_test_harness.performance_benchmarker import PerformanceBenchmarker
from greenlang.qa_test_harness.coverage_tracker import CoverageTracker
from greenlang.qa_test_harness.report_generator import ReportGenerator
from greenlang.qa_test_harness.provenance import ProvenanceTracker
from greenlang.qa_test_harness.models import (
    TestCaseInput,
    TestCaseResult,
    TestSuiteInput,
    TestSuiteResult,
    TestStatus,
    TestCategory,
    TestAssertion,
    GoldenFileEntry,
    PerformanceBenchmark,
    CoverageReport,
    QAStatistics,
)
from greenlang.qa_test_harness.metrics import (
    PROMETHEUS_AVAILABLE,
    record_test_run,
    update_pass_rate,
    record_regression,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# QATestHarnessService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["QATestHarnessService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class QATestHarnessService:
    """Unified facade over the QA Test Harness SDK.

    Aggregates all QA test harness engines (test runner, assertion engine,
    golden file manager, regression detector, performance benchmarker,
    coverage tracker, report generator, provenance tracker) through a
    single entry point with convenience methods for common operations.

    Attributes:
        config: QATestHarnessConfig instance.
        assertion_engine: AssertionEngine instance.
        runner: TestRunner instance.
        golden_file_manager: GoldenFileManager instance.
        regression_detector: RegressionDetector instance.
        benchmarker: PerformanceBenchmarker instance.
        coverage_tracker: CoverageTracker instance.
        report_generator: ReportGenerator instance.
        provenance: ProvenanceTracker instance.

    Example:
        >>> service = QATestHarnessService()
        >>> result = service.run_test(test_input)
        >>> print(result.status)
    """

    def __init__(
        self,
        config: Optional[QATestHarnessConfig] = None,
    ) -> None:
        """Initialize the QA Test Harness Service facade.

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Initialize all engines
        self.provenance = ProvenanceTracker()
        self.assertion_engine = AssertionEngine(self.config)
        self.runner = TestRunner(self.config, self.assertion_engine)
        self.golden_file_manager = GoldenFileManager(self.config)
        self.regression_detector = RegressionDetector(self.config)
        self.benchmarker = PerformanceBenchmarker(self.config)
        self.coverage_tracker = CoverageTracker(self.config)
        self.report_generator = ReportGenerator(self.config)

        # Agent registry maps agent_type to agent class
        self._agent_registry: Dict[str, Type[Any]] = {}

        # Statistics tracking
        self._stats = QAStatistics()
        self._started = False

        logger.info("QATestHarnessService facade created")

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_type: str,
        agent_class: Type[Any],
    ) -> None:
        """Register an agent type for testing.

        Args:
            agent_type: The agent type identifier.
            agent_class: The agent class to instantiate for tests.
        """
        self._agent_registry[agent_type] = agent_class
        logger.info("Registered agent for testing: %s", agent_type)

    # ------------------------------------------------------------------
    # Core test operations
    # ------------------------------------------------------------------

    def run_test(
        self,
        test_input: TestCaseInput,
    ) -> TestCaseResult:
        """Run a single test case.

        Args:
            test_input: Test case specification.

        Returns:
            TestCaseResult with test outcome.
        """
        result = self.runner.run_test(test_input, self._agent_registry)

        # Track coverage
        if self.config.enable_coverage_tracking:
            self.coverage_tracker.track(test_input.agent_type, test_input.name)

        # Record provenance
        self.provenance.record(
            entity_type="test_run",
            entity_id=result.test_id,
            action="execute",
            data_hash=result.output_hash or result.input_hash,
        )

        # Update statistics
        self._update_stats(result)

        return result

    def run_suite(
        self,
        suite_input: TestSuiteInput,
    ) -> TestSuiteResult:
        """Run a test suite.

        Args:
            suite_input: Test suite specification.

        Returns:
            TestSuiteResult with all test outcomes.
        """
        result = self.runner.run_suite(suite_input, self._agent_registry)

        # Track coverage for each test
        if self.config.enable_coverage_tracking:
            for test_result in result.test_results:
                matching = next(
                    (tc for tc in suite_input.test_cases
                     if tc.test_id == test_result.test_id),
                    None,
                )
                if matching:
                    self.coverage_tracker.track(
                        matching.agent_type, matching.name,
                    )

        # Record provenance
        self.provenance.record(
            entity_type="test_suite",
            entity_id=result.suite_id,
            action="execute",
            data_hash=result.provenance_hash,
        )

        # Update statistics from suite results
        for test_result in result.test_results:
            self._update_stats(test_result)

        return result

    def test_determinism(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        iterations: Optional[int] = None,
    ) -> TestCaseResult:
        """Run a determinism test for an agent.

        Args:
            agent_type: Agent type to test.
            input_data: Input data for the agent.
            iterations: Number of iterations (defaults to config value).

        Returns:
            TestCaseResult with determinism verification.

        Raises:
            ValueError: If agent type is not registered.
        """
        agent_class = self._agent_registry.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type not registered: {agent_type}")

        iters = iterations or self.config.determinism_iterations

        # Build test input
        test_input = TestCaseInput(
            name=f"determinism_{agent_type}",
            description=f"Verify {agent_type} produces deterministic outputs",
            category=TestCategory.DETERMINISM,
            agent_type=agent_type,
            input_data=input_data,
            tags=["determinism", "critical"],
        )

        # Run the basic test first
        result = self.run_test(test_input)

        # Add determinism-specific assertions
        det_assertions = self.assertion_engine.check_determinism(
            agent_class, input_data, iters,
        )
        result.assertions.extend(det_assertions)

        # Update status if new assertions failed
        if not all(a.passed for a in result.assertions):
            result.status = TestStatus.FAILED

        return result

    def test_zero_hallucination(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
    ) -> TestCaseResult:
        """Run a zero-hallucination test for an agent.

        Args:
            agent_type: Agent type to test.
            input_data: Input data for the agent.

        Returns:
            TestCaseResult with zero-hallucination verification.
        """
        test_input = TestCaseInput(
            name=f"zero_hallucination_{agent_type}",
            description=f"Verify {agent_type} produces no hallucinated data",
            category=TestCategory.ZERO_HALLUCINATION,
            agent_type=agent_type,
            input_data=input_data,
            tags=["zero_hallucination", "critical"],
        )

        return self.run_test(test_input)

    def test_lineage(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
    ) -> TestCaseResult:
        """Run a lineage completeness test for an agent.

        Args:
            agent_type: Agent type to test.
            input_data: Input data for the agent.

        Returns:
            TestCaseResult with lineage verification.
        """
        test_input = TestCaseInput(
            name=f"lineage_completeness_{agent_type}",
            description=f"Verify {agent_type} outputs have complete lineage",
            category=TestCategory.LINEAGE,
            agent_type=agent_type,
            input_data=input_data,
            tags=["lineage", "audit"],
        )

        return self.run_test(test_input)

    def test_regression(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        baseline_hash: Optional[str] = None,
    ) -> TestCaseResult:
        """Run a regression test for an agent.

        Args:
            agent_type: Agent type to test.
            input_data: Input data for the agent.
            baseline_hash: Optional explicit baseline hash.

        Returns:
            TestCaseResult with regression detection.
        """
        test_input = TestCaseInput(
            name=f"regression_{agent_type}",
            description=f"Detect regression in {agent_type}",
            category=TestCategory.REGRESSION,
            agent_type=agent_type,
            input_data=input_data,
            tags=["regression"],
        )

        result = self.run_test(test_input)

        # Run regression-specific check
        input_hash = result.input_hash
        output_hash = result.output_hash

        if output_hash:
            regression_assertion = self.regression_detector.check_regression(
                agent_type=agent_type,
                input_hash=input_hash,
                output_hash=output_hash,
                baseline_hash=baseline_hash,
            )
            result.assertions.append(regression_assertion)

            # Check historical consistency
            history_assertion = self.regression_detector.check_historical_consistency(
                agent_type=agent_type,
                input_hash=input_hash,
                output_hash=output_hash,
            )
            result.assertions.append(history_assertion)

            # Update status if regression detected
            if not all(a.passed for a in result.assertions):
                result.status = TestStatus.FAILED
                self._stats.regressions_detected += 1

        return result

    # ------------------------------------------------------------------
    # Golden file operations
    # ------------------------------------------------------------------

    def save_golden_file(
        self,
        agent_type: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> GoldenFileEntry:
        """Save a golden file.

        Args:
            agent_type: Type of agent.
            name: Golden file name.
            input_data: Input data used to generate output.
            output_data: Output data to save.

        Returns:
            GoldenFileEntry with metadata.
        """
        entry = self.golden_file_manager.save_golden_file(
            agent_type=agent_type,
            name=name,
            input_data=input_data,
            output_data=output_data,
        )

        # Record provenance
        self.provenance.record(
            entity_type="golden_file",
            entity_id=entry.file_id,
            action="create",
            data_hash=entry.content_hash,
        )

        return entry

    def compare_golden_file(
        self,
        file_id: str,
        agent_result: Any,
    ) -> List[TestAssertion]:
        """Compare an agent result with a golden file.

        Args:
            file_id: Golden file entry ID.
            agent_result: Agent execution result.

        Returns:
            List of assertion results.

        Raises:
            ValueError: If golden file entry is not found.
        """
        entry = self.golden_file_manager.get_golden_file(file_id)
        if entry is None:
            raise ValueError(f"Golden file entry not found: {file_id}")

        assertions = self.golden_file_manager.compare_with_golden(
            agent_result, entry,
        )

        # Track mismatches in stats
        if not all(a.passed for a in assertions):
            self._stats.golden_file_mismatches += 1

        return assertions

    # ------------------------------------------------------------------
    # Performance benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        iterations: Optional[int] = None,
        threshold_ms: Optional[float] = None,
    ) -> PerformanceBenchmark:
        """Run a performance benchmark.

        Args:
            agent_type: Agent type to benchmark.
            input_data: Input data for the agent.
            iterations: Number of iterations.
            threshold_ms: Performance threshold in ms.

        Returns:
            PerformanceBenchmark with timing statistics.

        Raises:
            ValueError: If agent type is not registered.
        """
        agent_class = self._agent_registry.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type not registered: {agent_type}")

        result = self.benchmarker.benchmark(
            agent_class=agent_class,
            input_data=input_data,
            iterations=iterations,
            threshold_ms=threshold_ms,
        )

        # Record provenance
        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent_type": agent_type,
                "iterations": iterations or self.config.performance_default_iterations,
                "p95_ms": result.p95_ms,
                "p99_ms": result.p99_ms,
            }, sort_keys=True, default=str).encode()
        ).hexdigest()

        self.provenance.record(
            entity_type="benchmark",
            entity_id=agent_type,
            action="execute",
            data_hash=provenance_hash,
        )

        return result

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def get_coverage(
        self,
        agent_type: str,
    ) -> CoverageReport:
        """Get coverage report for an agent type.

        Args:
            agent_type: Type of agent to report on.

        Returns:
            CoverageReport with coverage statistics.
        """
        agent_class = self._agent_registry.get(agent_type)
        return self.coverage_tracker.get_report(agent_type, agent_class)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        suite_result: TestSuiteResult,
        format: Optional[str] = None,
    ) -> str:
        """Generate a test report.

        Args:
            suite_result: Suite result to report on.
            format: Report format (text, json, markdown, html).

        Returns:
            Formatted report string.
        """
        return self.report_generator.generate(suite_result, format=format)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> QAStatistics:
        """Get aggregated QA statistics.

        Returns:
            QAStatistics summary.
        """
        # Update pass rate
        if self._stats.total_runs > 0:
            self._stats.pass_rate = round(
                self._stats.passed / self._stats.total_runs * 100, 2,
            )
            update_pass_rate(self._stats.pass_rate)

        # Update coverage
        all_reports = self.coverage_tracker.get_all_reports(self._agent_registry)
        if all_reports:
            total_coverage = sum(
                r.coverage_percent for r in all_reports.values()
            )
            self._stats.coverage_percent = round(
                total_coverage / len(all_reports), 2,
            )

        return self._stats

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_runner(self) -> TestRunner:
        """Get the TestRunner instance.

        Returns:
            TestRunner used by this service.
        """
        return self.runner

    def get_assertion_engine(self) -> AssertionEngine:
        """Get the AssertionEngine instance.

        Returns:
            AssertionEngine used by this service.
        """
        return self.assertion_engine

    def get_golden_file_manager(self) -> GoldenFileManager:
        """Get the GoldenFileManager instance.

        Returns:
            GoldenFileManager used by this service.
        """
        return self.golden_file_manager

    def get_regression_detector(self) -> RegressionDetector:
        """Get the RegressionDetector instance.

        Returns:
            RegressionDetector used by this service.
        """
        return self.regression_detector

    def get_benchmarker(self) -> PerformanceBenchmarker:
        """Get the PerformanceBenchmarker instance.

        Returns:
            PerformanceBenchmarker used by this service.
        """
        return self.benchmarker

    def get_coverage_tracker(self) -> CoverageTracker:
        """Get the CoverageTracker instance.

        Returns:
            CoverageTracker used by this service.
        """
        return self.coverage_tracker

    def get_provenance(self) -> ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            ProvenanceTracker used by this service.
        """
        return self.provenance

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get QA test harness service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_runs": self._stats.total_runs,
            "passed": self._stats.passed,
            "failed": self._stats.failed,
            "skipped": self._stats.skipped,
            "errors": self._stats.errors,
            "pass_rate": self._stats.pass_rate,
            "regressions_detected": self._stats.regressions_detected,
            "golden_file_mismatches": self._stats.golden_file_mismatches,
            "registered_agents": len(self._agent_registry),
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_stats(self, result: TestCaseResult) -> None:
        """Update statistics from a test case result.

        Args:
            result: Completed test case result.
        """
        self._stats.total_runs += 1

        if result.status == TestStatus.PASSED:
            self._stats.passed += 1
        elif result.status == TestStatus.FAILED:
            self._stats.failed += 1
        elif result.status == TestStatus.SKIPPED:
            self._stats.skipped += 1
        elif result.status in (TestStatus.ERROR, TestStatus.TIMEOUT):
            self._stats.errors += 1

        # Update average duration
        total = self._stats.total_runs
        prev_avg = self._stats.avg_duration_ms
        self._stats.avg_duration_ms = (
            (prev_avg * (total - 1) + result.duration_ms) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the QA test harness service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("QATestHarnessService already started; skipping")
            return

        logger.info("QATestHarnessService starting up...")
        self._started = True
        logger.info("QATestHarnessService startup complete")

    def shutdown(self) -> None:
        """Shutdown the QA test harness service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("QATestHarnessService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> QATestHarnessService:
    """Get or create the singleton QATestHarnessService instance.

    Returns:
        The singleton QATestHarnessService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = QATestHarnessService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_qa_test_harness(
    app: Any,
    config: Optional[QATestHarnessConfig] = None,
) -> QATestHarnessService:
    """Configure the QA Test Harness Service on a FastAPI application.

    Creates the QATestHarnessService, stores it in app.state, mounts
    the QA test harness API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional QA test harness config.

    Returns:
        QATestHarnessService instance.
    """
    global _singleton_instance

    service = QATestHarnessService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.qa_test_harness_service = service

    # Mount QA test harness API router
    try:
        from greenlang.qa_test_harness.api.router import router as qa_router
        if qa_router is not None:
            app.include_router(qa_router)
            logger.info("QA test harness service API router mounted")
    except ImportError:
        logger.warning("QA test harness router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("QA test harness service configured on app")
    return service


def get_qa_test_harness(app: Any) -> QATestHarnessService:
    """Get the QATestHarnessService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        QATestHarnessService instance.

    Raises:
        RuntimeError: If QA test harness service not configured.
    """
    service = getattr(app.state, "qa_test_harness_service", None)
    if service is None:
        raise RuntimeError(
            "QA test harness service not configured. "
            "Call configure_qa_test_harness(app) first."
        )
    return service


def get_router() -> Any:
    """Get the QA test harness API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.qa_test_harness.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "QATestHarnessService",
    "configure_qa_test_harness",
    "get_qa_test_harness",
    "get_router",
]
