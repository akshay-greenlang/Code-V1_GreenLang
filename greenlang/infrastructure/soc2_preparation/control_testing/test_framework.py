# -*- coding: utf-8 -*-
"""
Control Test Framework - SEC-009 Phase 4

Core test execution framework for SOC 2 Type II control testing. Provides
test registration, execution, result tracking, and export capabilities.

The framework supports:
    - Test registration with unique IDs and criterion mapping
    - Single test and suite execution with async support
    - Result aggregation and historical tracking
    - Export to multiple formats (JSON, CSV, PDF)

Example:
    >>> framework = ControlTestFramework()
    >>> test = ControlTest(
    ...     test_id="CC6.1.1",
    ...     criterion_id="CC6.1",
    ...     test_type=TestType.AUTOMATED,
    ...     description="Verify MFA is enforced for all users",
    ...     procedure="Query auth_service for MFA status",
    ...     expected_result="100% MFA enforcement",
    ... )
    >>> framework.register_test(test)
    >>> result = await framework.execute_test("CC6.1.1")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestType(str, Enum):
    """Type of control test execution method."""

    AUTOMATED = "automated"
    """Fully automated test with no human intervention required."""

    SEMI_AUTOMATED = "semi_automated"
    """Automated data collection with manual verification."""

    MANUAL = "manual"
    """Requires manual execution and evidence collection."""

    OBSERVATION = "observation"
    """Physical observation or walkthrough required."""

    INQUIRY = "inquiry"
    """Interview or inquiry-based test."""


class TestStatus(str, Enum):
    """Status of a control test execution."""

    PENDING = "pending"
    """Test is registered but has not been executed."""

    RUNNING = "running"
    """Test is currently executing."""

    PASSED = "passed"
    """Test executed successfully and control is operating effectively."""

    FAILED = "failed"
    """Test executed but control is not operating effectively."""

    ERROR = "error"
    """Test execution encountered an error."""

    SKIPPED = "skipped"
    """Test was skipped due to precondition failure or configuration."""

    NOT_APPLICABLE = "not_applicable"
    """Test is not applicable to this environment."""


class Severity(str, Enum):
    """Severity level for test failures."""

    CRITICAL = "critical"
    """Critical control failure requiring immediate remediation."""

    HIGH = "high"
    """High-severity control deficiency."""

    MEDIUM = "medium"
    """Medium-severity control weakness."""

    LOW = "low"
    """Low-severity or informational finding."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ControlTest(BaseModel):
    """Definition of a SOC 2 control test.

    Attributes:
        test_id: Unique identifier for the test (e.g., "CC6.1.1").
        criterion_id: SOC 2 criterion being tested (e.g., "CC6.1").
        test_type: Execution method (automated, manual, etc.).
        description: Human-readable description of what is tested.
        procedure: Step-by-step test procedure.
        expected_result: Expected outcome for a passing test.
        frequency: How often the test should run (daily, weekly, etc.).
        owner: Team or individual responsible for this test.
        tags: Labels for filtering and grouping tests.
        enabled: Whether this test is active.
        timeout_seconds: Maximum execution time before timeout.
        dependencies: List of test IDs that must pass first.
        metadata: Arbitrary key-value pairs for integrations.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    test_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique test identifier (e.g., CC6.1.1).",
    )
    criterion_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="SOC 2 criterion ID (e.g., CC6.1).",
    )
    test_type: TestType = Field(
        default=TestType.AUTOMATED,
        description="Test execution method.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Description of what is being tested.",
    )
    procedure: str = Field(
        default="",
        max_length=4096,
        description="Step-by-step test procedure.",
    )
    expected_result: str = Field(
        default="",
        max_length=2048,
        description="Expected outcome for a passing test.",
    )
    frequency: str = Field(
        default="quarterly",
        description="Test execution frequency (daily, weekly, monthly, quarterly).",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team or individual responsible for this test.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Labels for filtering and grouping.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this test is active.",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum execution time in seconds.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Test IDs that must pass before this test runs.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for integrations.",
    )

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, v: str) -> str:
        """Validate criterion ID format."""
        valid_prefixes = ("CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7", "CC8", "CC9", "A1", "A2", "C1", "PI1")
        if not any(v.upper().startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid criterion ID '{v}'. Must start with one of: {valid_prefixes}"
            )
        return v.upper()

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        """Validate and normalize frequency."""
        allowed = {"continuous", "daily", "weekly", "monthly", "quarterly", "annually", "on_demand"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid frequency '{v}'. Allowed: {sorted(allowed)}")
        return v_lower


class Evidence(BaseModel):
    """Evidence collected during test execution.

    Attributes:
        evidence_id: Unique identifier for this evidence item.
        evidence_type: Type of evidence (screenshot, log, config, etc.).
        description: Description of what this evidence shows.
        content: The actual evidence content (text, base64, URL).
        collected_at: When the evidence was collected.
        hash: SHA-256 hash for integrity verification.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evidence identifier.",
    )
    evidence_type: str = Field(
        ...,
        description="Type: screenshot, log, config, query_result, document, etc.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Description of what this evidence shows.",
    )
    content: str = Field(
        default="",
        description="Evidence content (text, base64 encoded, or URL).",
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Collection timestamp (UTC).",
    )
    hash: str = Field(
        default="",
        description="SHA-256 hash for integrity verification.",
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the evidence content."""
        content_bytes = self.content.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """Compute hash after initialization if not provided."""
        if not self.hash and self.content:
            object.__setattr__(self, "hash", self.compute_hash())


class TestResult(BaseModel):
    """Result of a single control test execution.

    Attributes:
        result_id: Unique identifier for this result.
        test_id: ID of the test that was executed.
        test_run_id: ID of the test run this result belongs to.
        status: Final test status (passed, failed, error, etc.).
        severity: Severity level if the test failed.
        actual_result: What was actually observed.
        evidence: List of evidence items collected.
        exceptions: List of exception/finding descriptions.
        error_message: Error message if status is ERROR.
        started_at: When the test started executing.
        completed_at: When the test finished.
        duration_ms: Execution duration in milliseconds.
        executed_by: Identity of the executor (user or automation).
        notes: Additional notes or observations.
    """

    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier.",
    )
    test_id: str = Field(
        ...,
        description="ID of the test that was executed.",
    )
    test_run_id: str = Field(
        default="",
        description="ID of the test run this belongs to.",
    )
    status: TestStatus = Field(
        default=TestStatus.PENDING,
        description="Final test status.",
    )
    severity: Optional[Severity] = Field(
        default=None,
        description="Severity level for failures.",
    )
    actual_result: str = Field(
        default="",
        max_length=4096,
        description="What was actually observed.",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence collected during test.",
    )
    exceptions: List[str] = Field(
        default_factory=list,
        description="Exception/finding descriptions.",
    )
    error_message: str = Field(
        default="",
        max_length=2048,
        description="Error message if status is ERROR.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Start timestamp (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp (UTC).",
    )
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Execution duration in milliseconds.",
    )
    executed_by: str = Field(
        default="automation",
        description="Identity of the executor.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Additional notes or observations.",
    )

    @property
    def passed(self) -> bool:
        """Return True if the test passed."""
        return self.status == TestStatus.PASSED


class TestRun(BaseModel):
    """A collection of test executions (a test suite run).

    Attributes:
        run_id: Unique identifier for this test run.
        name: Descriptive name for this run.
        criteria: SOC 2 criteria included in this run.
        started_at: When the run started.
        completed_at: When the run finished.
        status: Overall run status.
        total_tests: Total number of tests in the run.
        passed_count: Number of tests that passed.
        failed_count: Number of tests that failed.
        error_count: Number of tests that errored.
        skipped_count: Number of tests that were skipped.
        initiated_by: Who started this run.
        environment: Environment where tests were executed.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique run identifier.",
    )
    name: str = Field(
        default="",
        max_length=256,
        description="Descriptive name for this run.",
    )
    criteria: List[str] = Field(
        default_factory=list,
        description="SOC 2 criteria included in this run.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Start timestamp (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp (UTC).",
    )
    status: TestStatus = Field(
        default=TestStatus.PENDING,
        description="Overall run status.",
    )
    total_tests: int = Field(
        default=0,
        ge=0,
        description="Total number of tests.",
    )
    passed_count: int = Field(
        default=0,
        ge=0,
        description="Number of passed tests.",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed tests.",
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errored tests.",
    )
    skipped_count: int = Field(
        default=0,
        ge=0,
        description="Number of skipped tests.",
    )
    initiated_by: str = Field(
        default="system",
        description="Who initiated this run.",
    )
    environment: str = Field(
        default="production",
        description="Environment where tests ran.",
    )

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a percentage."""
        executed = self.passed_count + self.failed_count
        if executed == 0:
            return 0.0
        return (self.passed_count / executed) * 100.0


# ---------------------------------------------------------------------------
# Test Executor Type
# ---------------------------------------------------------------------------

TestExecutor = Callable[[ControlTest], Coroutine[Any, Any, TestResult]]


# ---------------------------------------------------------------------------
# Control Test Framework
# ---------------------------------------------------------------------------


class ControlTestFramework:
    """Core framework for SOC 2 control test management and execution.

    Manages test registration, execution scheduling, result tracking,
    and export of test results for auditor review.

    Attributes:
        _tests: Registry of control tests by test_id.
        _results: Historical test results by test_run_id.
        _runs: Test run metadata by run_id.
        _executors: Custom test executor functions by test_id.
    """

    def __init__(self) -> None:
        """Initialize the control test framework."""
        self._tests: Dict[str, ControlTest] = {}
        self._results: Dict[str, List[TestResult]] = {}
        self._runs: Dict[str, TestRun] = {}
        self._executors: Dict[str, TestExecutor] = {}
        logger.info("ControlTestFramework initialized")

    # ------------------------------------------------------------------
    # Test Registration
    # ------------------------------------------------------------------

    def register_test(self, test: ControlTest) -> None:
        """Register a control test with the framework.

        Args:
            test: The control test definition to register.

        Raises:
            ValueError: If a test with the same ID already exists.
        """
        if test.test_id in self._tests:
            raise ValueError(f"Test '{test.test_id}' is already registered")
        self._tests[test.test_id] = test
        logger.debug("Registered test: %s (%s)", test.test_id, test.criterion_id)

    def register_executor(
        self,
        test_id: str,
        executor: TestExecutor,
    ) -> None:
        """Register a custom executor function for a test.

        Args:
            test_id: The test ID to associate with this executor.
            executor: Async function that executes the test.
        """
        self._executors[test_id] = executor
        logger.debug("Registered executor for test: %s", test_id)

    def get_test(self, test_id: str) -> Optional[ControlTest]:
        """Get a test definition by ID.

        Args:
            test_id: The test identifier.

        Returns:
            ControlTest if found, None otherwise.
        """
        return self._tests.get(test_id)

    def list_tests(
        self,
        criterion_id: Optional[str] = None,
        test_type: Optional[TestType] = None,
        enabled_only: bool = True,
    ) -> List[ControlTest]:
        """List registered tests with optional filtering.

        Args:
            criterion_id: Filter by criterion (e.g., "CC6").
            test_type: Filter by test type.
            enabled_only: If True, only return enabled tests.

        Returns:
            List of matching control tests.
        """
        tests = list(self._tests.values())

        if enabled_only:
            tests = [t for t in tests if t.enabled]

        if criterion_id:
            criterion_upper = criterion_id.upper()
            tests = [t for t in tests if t.criterion_id.startswith(criterion_upper)]

        if test_type:
            tests = [t for t in tests if t.test_type == test_type]

        return sorted(tests, key=lambda t: t.test_id)

    # ------------------------------------------------------------------
    # Test Execution
    # ------------------------------------------------------------------

    async def execute_test(self, test_id: str) -> TestResult:
        """Execute a single control test.

        Args:
            test_id: The ID of the test to execute.

        Returns:
            TestResult with execution outcome.

        Raises:
            ValueError: If the test is not registered.
        """
        test = self._tests.get(test_id)
        if test is None:
            raise ValueError(f"Test '{test_id}' is not registered")

        if not test.enabled:
            return TestResult(
                test_id=test_id,
                status=TestStatus.SKIPPED,
                actual_result="Test is disabled",
                completed_at=datetime.now(timezone.utc),
            )

        start_time = datetime.now(timezone.utc)
        logger.info("Executing test: %s", test_id)

        try:
            # Check for custom executor
            if test_id in self._executors:
                result = await asyncio.wait_for(
                    self._executors[test_id](test),
                    timeout=test.timeout_seconds,
                )
            else:
                # Default executor - returns pending for manual tests
                result = await self._default_executor(test)

            result.started_at = start_time
            result.completed_at = datetime.now(timezone.utc)
            result.duration_ms = int(
                (result.completed_at - start_time).total_seconds() * 1000
            )

            logger.info(
                "Test %s completed: %s (duration=%dms)",
                test_id,
                result.status.value,
                result.duration_ms,
            )
            return result

        except asyncio.TimeoutError:
            return TestResult(
                test_id=test_id,
                status=TestStatus.ERROR,
                error_message=f"Test timed out after {test.timeout_seconds} seconds",
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.error("Test %s failed with error: %s", test_id, exc, exc_info=True)
            return TestResult(
                test_id=test_id,
                status=TestStatus.ERROR,
                error_message=str(exc),
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

    async def execute_suite(
        self,
        criteria: Optional[List[str]] = None,
        test_type: Optional[TestType] = None,
        parallel: bool = False,
        initiated_by: str = "system",
    ) -> List[TestResult]:
        """Execute a suite of control tests.

        Args:
            criteria: List of SOC 2 criteria to test (e.g., ["CC6", "CC7"]).
            test_type: Filter by test type.
            parallel: If True, execute tests in parallel.
            initiated_by: Identity of who initiated the run.

        Returns:
            List of TestResults for all executed tests.
        """
        # Create test run
        run = TestRun(
            name=f"Control Test Suite - {datetime.now(timezone.utc).isoformat()}",
            criteria=criteria or [],
            initiated_by=initiated_by,
            status=TestStatus.RUNNING,
        )
        self._runs[run.run_id] = run

        # Get tests matching criteria
        tests: List[ControlTest] = []
        if criteria:
            for criterion in criteria:
                tests.extend(self.list_tests(criterion_id=criterion, test_type=test_type))
        else:
            tests = self.list_tests(test_type=test_type)

        run.total_tests = len(tests)
        logger.info(
            "Starting test suite: %s (%d tests)",
            run.run_id,
            run.total_tests,
        )

        results: List[TestResult] = []

        if parallel:
            # Execute tests in parallel
            tasks = [self.execute_test(t.test_id) for t in tests]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # Execute tests sequentially
            for test in tests:
                result = await self.execute_test(test.test_id)
                result.test_run_id = run.run_id
                results.append(result)

        # Update run statistics
        for result in results:
            if result.status == TestStatus.PASSED:
                run.passed_count += 1
            elif result.status == TestStatus.FAILED:
                run.failed_count += 1
            elif result.status == TestStatus.ERROR:
                run.error_count += 1
            elif result.status == TestStatus.SKIPPED:
                run.skipped_count += 1

        run.completed_at = datetime.now(timezone.utc)
        run.status = (
            TestStatus.PASSED
            if run.failed_count == 0 and run.error_count == 0
            else TestStatus.FAILED
        )

        # Store results
        self._results[run.run_id] = results

        logger.info(
            "Test suite completed: %s (passed=%d, failed=%d, errors=%d)",
            run.run_id,
            run.passed_count,
            run.failed_count,
            run.error_count,
        )

        return results

    async def _default_executor(self, test: ControlTest) -> TestResult:
        """Default test executor for tests without custom executors.

        For manual and observation tests, returns PENDING requiring manual completion.
        For automated tests without executors, returns ERROR.

        Args:
            test: The test to execute.

        Returns:
            TestResult with appropriate status.
        """
        if test.test_type in (TestType.MANUAL, TestType.OBSERVATION, TestType.INQUIRY):
            return TestResult(
                test_id=test.test_id,
                status=TestStatus.PENDING,
                actual_result="Awaiting manual execution",
                notes="This test requires manual execution. Please complete and update results.",
            )
        else:
            return TestResult(
                test_id=test.test_id,
                status=TestStatus.ERROR,
                error_message="No executor registered for automated test",
            )

    # ------------------------------------------------------------------
    # Result Management
    # ------------------------------------------------------------------

    def get_results(self, test_run_id: str) -> List[TestResult]:
        """Get all results for a test run.

        Args:
            test_run_id: The test run identifier.

        Returns:
            List of TestResults for the run.
        """
        return self._results.get(test_run_id, [])

    def get_run(self, run_id: str) -> Optional[TestRun]:
        """Get test run metadata.

        Args:
            run_id: The run identifier.

        Returns:
            TestRun if found, None otherwise.
        """
        return self._runs.get(run_id)

    def list_runs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TestRun]:
        """List test runs with pagination.

        Args:
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.

        Returns:
            List of TestRun objects, newest first.
        """
        runs = sorted(
            self._runs.values(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return runs[offset : offset + limit]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(
        self,
        results: List[TestResult],
        format: str = "json",
    ) -> str:
        """Export test results to a specified format.

        Args:
            results: List of TestResults to export.
            format: Output format (json, csv, markdown).

        Returns:
            Formatted string containing the exported results.

        Raises:
            ValueError: If the format is not supported.
        """
        if format == "json":
            return self._export_json(results)
        elif format == "csv":
            return self._export_csv(results)
        elif format == "markdown":
            return self._export_markdown(results)
        else:
            raise ValueError(f"Unsupported format: {format}. Use json, csv, or markdown.")

    def _export_json(self, results: List[TestResult]) -> str:
        """Export results as JSON."""
        data = [r.model_dump(mode="json") for r in results]
        return json.dumps(data, indent=2, default=str)

    def _export_csv(self, results: List[TestResult]) -> str:
        """Export results as CSV."""
        lines = [
            "test_id,status,severity,started_at,completed_at,duration_ms,executed_by"
        ]
        for r in results:
            severity = r.severity.value if r.severity else ""
            completed = r.completed_at.isoformat() if r.completed_at else ""
            lines.append(
                f"{r.test_id},{r.status.value},{severity},"
                f"{r.started_at.isoformat()},{completed},{r.duration_ms},{r.executed_by}"
            )
        return "\n".join(lines)

    def _export_markdown(self, results: List[TestResult]) -> str:
        """Export results as Markdown table."""
        lines = [
            "| Test ID | Status | Severity | Duration (ms) | Notes |",
            "|---------|--------|----------|---------------|-------|",
        ]
        for r in results:
            severity = r.severity.value if r.severity else "-"
            notes = r.notes[:50] + "..." if len(r.notes) > 50 else r.notes
            lines.append(
                f"| {r.test_id} | {r.status.value} | {severity} | "
                f"{r.duration_ms} | {notes} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics.

        Returns:
            Dictionary with test counts, pass rates, and run history.
        """
        total_tests = len(self._tests)
        by_criterion: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for test in self._tests.values():
            criterion = test.criterion_id.split(".")[0]
            by_criterion[criterion] = by_criterion.get(criterion, 0) + 1
            by_type[test.test_type.value] = by_type.get(test.test_type.value, 0) + 1

        return {
            "total_tests": total_tests,
            "by_criterion": by_criterion,
            "by_type": by_type,
            "total_runs": len(self._runs),
            "enabled_tests": len([t for t in self._tests.values() if t.enabled]),
        }


__all__ = [
    "ControlTestFramework",
    "ControlTest",
    "TestResult",
    "TestRun",
    "TestStatus",
    "TestType",
    "Severity",
    "Evidence",
    "TestExecutor",
]
