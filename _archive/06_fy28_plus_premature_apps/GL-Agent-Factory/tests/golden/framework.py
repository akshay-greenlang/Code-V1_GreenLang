"""
GoldenTestRunner Framework

Provides the core testing infrastructure for golden tests that validate
zero-hallucination deterministic calculations across all GreenLang agents.

Key Features:
- Expected vs actual comparison with configurable tolerances
- Hash verification for provenance tracking
- Performance timing validation
- Detailed failure reporting for debugging
- Batch test execution with progress tracking

Example:
    runner = GoldenTestRunner()
    result = runner.run_single_test(test_case)
    assert result.passed, f"Test failed: {result.failure_reason}"
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys

logger = logging.getLogger(__name__)


class ComparisonMode(str, Enum):
    """Comparison mode for expected vs actual values."""

    EXACT = "exact"  # Bit-perfect match required
    RELATIVE = "relative"  # Relative tolerance (percentage)
    ABSOLUTE = "absolute"  # Absolute tolerance (fixed value)
    HYBRID = "hybrid"  # Use both relative and absolute


@dataclass
class ToleranceConfig:
    """Configuration for numeric comparison tolerances.

    Attributes:
        mode: Comparison mode to use
        relative_tolerance: Relative tolerance as fraction (e.g., 0.001 = 0.1%)
        absolute_tolerance: Absolute tolerance as fixed value
        decimal_places: Number of decimal places to round before comparison
    """

    mode: ComparisonMode = ComparisonMode.RELATIVE
    relative_tolerance: float = 1e-6  # Default: 0.0001%
    absolute_tolerance: float = 1e-9
    decimal_places: Optional[int] = None

    @classmethod
    def exact(cls) -> "ToleranceConfig":
        """Create exact match configuration."""
        return cls(mode=ComparisonMode.EXACT)

    @classmethod
    def regulatory(cls) -> "ToleranceConfig":
        """Create regulatory-grade tolerance (0.01% relative)."""
        return cls(
            mode=ComparisonMode.HYBRID,
            relative_tolerance=1e-4,  # 0.01%
            absolute_tolerance=1e-6,
        )

    @classmethod
    def standard(cls) -> "ToleranceConfig":
        """Create standard tolerance for general calculations."""
        return cls(
            mode=ComparisonMode.RELATIVE,
            relative_tolerance=1e-6,  # 0.0001%
        )


@dataclass
class ComparisonResult:
    """Result of comparing expected vs actual values.

    Attributes:
        passed: Whether the comparison passed
        expected: Expected value
        actual: Actual value
        difference: Absolute difference
        relative_difference: Relative difference as percentage
        field_path: Path to the field being compared
        tolerance_used: Tolerance configuration used
        failure_reason: Reason for failure if not passed
    """

    passed: bool
    expected: Any
    actual: Any
    difference: Optional[float] = None
    relative_difference: Optional[float] = None
    field_path: str = ""
    tolerance_used: Optional[ToleranceConfig] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "difference": self.difference,
            "relative_difference_pct": self.relative_difference,
            "field_path": self.field_path,
            "failure_reason": self.failure_reason,
        }


@dataclass
class GoldenTestResult:
    """Result of running a single golden test.

    Attributes:
        test_id: Unique test identifier
        test_name: Human-readable test name
        passed: Whether the test passed
        execution_time_ms: Time to execute in milliseconds
        comparisons: List of field comparison results
        provenance_verified: Whether provenance hash was verified
        expected_hash: Expected provenance hash
        actual_hash: Actual provenance hash
        failure_reasons: List of failure reasons
        metadata: Additional test metadata
    """

    test_id: str
    test_name: str
    passed: bool
    execution_time_ms: float
    comparisons: List[ComparisonResult] = field(default_factory=list)
    provenance_verified: bool = False
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "passed": self.passed,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "provenance_verified": self.provenance_verified,
            "failure_reasons": self.failure_reasons,
            "comparisons": [c.to_dict() for c in self.comparisons if not c.passed],
            "metadata": self.metadata,
        }


@dataclass
class GoldenTestSuite:
    """Collection of golden test results.

    Attributes:
        suite_name: Name of the test suite
        agent_id: Agent being tested
        agent_version: Version of the agent
        results: Individual test results
        start_time: When the suite started
        end_time: When the suite ended
        total_tests: Total number of tests
        passed_tests: Number of passing tests
        failed_tests: Number of failing tests
    """

    suite_name: str
    agent_id: str
    agent_version: str
    results: List[GoldenTestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        """Number of passing tests."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_tests(self) -> int:
        """Number of failing tests."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return self.failed_tests == 0

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "suite_name": self.suite_name,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "summary": {
                "total": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "pass_rate_pct": round(self.pass_rate, 2),
                "duration_ms": round(self.duration_ms, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_junit_xml(self) -> str:
        """Convert to JUnit XML format for CI/CD integration."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{self.suite_name}" tests="{self.total_tests}" '
            f'failures="{self.failed_tests}" time="{self.duration_ms / 1000:.3f}">',
        ]

        for result in self.results:
            if result.passed:
                lines.append(
                    f'  <testcase classname="{self.agent_id}" '
                    f'name="{result.test_name}" time="{result.execution_time_ms / 1000:.3f}"/>'
                )
            else:
                lines.append(
                    f'  <testcase classname="{self.agent_id}" '
                    f'name="{result.test_name}" time="{result.execution_time_ms / 1000:.3f}">'
                )
                for reason in result.failure_reasons:
                    lines.append(f'    <failure message="{reason}"/>')
                lines.append("  </testcase>")

        lines.append("</testsuite>")
        return "\n".join(lines)


class NumericComparator:
    """Compares numeric values with configurable tolerance.

    This class handles the comparison of numeric values (floats, ints, Decimals)
    with various tolerance modes to support regulatory-grade validation.
    """

    def __init__(self, tolerance: ToleranceConfig = None):
        """Initialize comparator with tolerance configuration."""
        self.tolerance = tolerance or ToleranceConfig.standard()

    def compare(
        self,
        expected: Union[int, float, Decimal],
        actual: Union[int, float, Decimal],
        field_path: str = "",
    ) -> ComparisonResult:
        """Compare two numeric values.

        Args:
            expected: Expected value
            actual: Actual value
            field_path: Path to the field for error reporting

        Returns:
            ComparisonResult with comparison details
        """
        # Handle None values
        if expected is None and actual is None:
            return ComparisonResult(
                passed=True,
                expected=expected,
                actual=actual,
                field_path=field_path,
            )

        if expected is None or actual is None:
            return ComparisonResult(
                passed=False,
                expected=expected,
                actual=actual,
                field_path=field_path,
                failure_reason=f"None mismatch: expected={expected}, actual={actual}",
            )

        # Convert to float for comparison
        exp_float = float(expected)
        act_float = float(actual)

        # Round if decimal places specified
        if self.tolerance.decimal_places is not None:
            exp_float = round(exp_float, self.tolerance.decimal_places)
            act_float = round(act_float, self.tolerance.decimal_places)

        # Calculate differences
        abs_diff = abs(exp_float - act_float)
        rel_diff = 0.0

        if exp_float != 0:
            rel_diff = abs_diff / abs(exp_float)

        # Check based on mode
        passed = False

        if self.tolerance.mode == ComparisonMode.EXACT:
            passed = exp_float == act_float

        elif self.tolerance.mode == ComparisonMode.RELATIVE:
            passed = rel_diff <= self.tolerance.relative_tolerance

        elif self.tolerance.mode == ComparisonMode.ABSOLUTE:
            passed = abs_diff <= self.tolerance.absolute_tolerance

        elif self.tolerance.mode == ComparisonMode.HYBRID:
            # Pass if either relative OR absolute tolerance is met
            passed = (
                rel_diff <= self.tolerance.relative_tolerance
                or abs_diff <= self.tolerance.absolute_tolerance
            )

        failure_reason = None
        if not passed:
            failure_reason = (
                f"Numeric mismatch at '{field_path}': "
                f"expected={exp_float}, actual={act_float}, "
                f"abs_diff={abs_diff}, rel_diff={rel_diff * 100:.6f}%"
            )

        return ComparisonResult(
            passed=passed,
            expected=exp_float,
            actual=act_float,
            difference=abs_diff,
            relative_difference=rel_diff * 100,
            field_path=field_path,
            tolerance_used=self.tolerance,
            failure_reason=failure_reason,
        )


class HashVerifier:
    """Verifies provenance hashes for audit trail validation.

    Ensures that:
    - Provenance hashes are 64-character SHA-256 hex strings
    - Same inputs produce same hashes (determinism)
    - Hashes match expected values in golden tests
    """

    @staticmethod
    def verify_format(hash_value: str) -> bool:
        """Verify hash is valid SHA-256 format."""
        if not hash_value or not isinstance(hash_value, str):
            return False

        if len(hash_value) != 64:
            return False

        try:
            int(hash_value, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def verify_match(expected: str, actual: str) -> ComparisonResult:
        """Verify two hashes match exactly."""
        if not HashVerifier.verify_format(expected):
            return ComparisonResult(
                passed=False,
                expected=expected,
                actual=actual,
                failure_reason=f"Invalid expected hash format: {expected}",
            )

        if not HashVerifier.verify_format(actual):
            return ComparisonResult(
                passed=False,
                expected=expected,
                actual=actual,
                failure_reason=f"Invalid actual hash format: {actual}",
            )

        passed = expected == actual

        return ComparisonResult(
            passed=passed,
            expected=expected,
            actual=actual,
            failure_reason=None if passed else f"Hash mismatch: expected={expected[:16]}..., actual={actual[:16]}...",
        )

    @staticmethod
    def compute_hash(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class GoldenTestRunner:
    """Runs golden tests against GreenLang agents.

    The runner executes test cases, compares outputs against expected values,
    and generates detailed reports for CI/CD integration.

    Example:
        runner = GoldenTestRunner()
        suite = runner.run_test_suite(
            tests=test_cases,
            agent=carbon_agent,
            suite_name="Carbon Emissions Golden Tests"
        )
        assert suite.all_passed, f"Failed tests: {suite.failed_tests}"
    """

    def __init__(
        self,
        tolerance: ToleranceConfig = None,
        verify_provenance: bool = True,
        performance_threshold_ms: float = 100.0,
    ):
        """Initialize the golden test runner.

        Args:
            tolerance: Tolerance configuration for numeric comparisons
            verify_provenance: Whether to verify provenance hashes
            performance_threshold_ms: Maximum allowed execution time
        """
        self.tolerance = tolerance or ToleranceConfig.regulatory()
        self.verify_provenance = verify_provenance
        self.performance_threshold_ms = performance_threshold_ms
        self.comparator = NumericComparator(self.tolerance)
        self.hash_verifier = HashVerifier()

    def run_single_test(
        self,
        test_case: "GoldenTestCase",
        agent: Any,
        input_class: Any,
    ) -> GoldenTestResult:
        """Run a single golden test case.

        Args:
            test_case: The test case to run
            agent: The agent instance to test
            input_class: The input model class for the agent

        Returns:
            GoldenTestResult with comparison details
        """
        comparisons: List[ComparisonResult] = []
        failure_reasons: List[str] = []

        # Time the execution
        start_time = time.perf_counter()

        try:
            # Create input from test case data
            input_data = input_class(**test_case.input_data)

            # Run the agent
            result = agent.run(input_data)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Compare output fields
            for field_name, expected_value in test_case.expected_output.items():
                if hasattr(result, field_name):
                    actual_value = getattr(result, field_name)

                    comparison = self._compare_values(
                        expected_value,
                        actual_value,
                        field_name,
                    )
                    comparisons.append(comparison)

                    if not comparison.passed:
                        failure_reasons.append(comparison.failure_reason)
                else:
                    failure_reasons.append(f"Missing field: {field_name}")
                    comparisons.append(
                        ComparisonResult(
                            passed=False,
                            expected=expected_value,
                            actual=None,
                            field_path=field_name,
                            failure_reason=f"Field '{field_name}' not in output",
                        )
                    )

            # Verify provenance hash
            provenance_verified = False
            expected_hash = test_case.expected_output.get("provenance_hash")
            actual_hash = getattr(result, "provenance_hash", None)

            if self.verify_provenance and expected_hash and actual_hash:
                hash_result = self.hash_verifier.verify_match(expected_hash, actual_hash)
                provenance_verified = hash_result.passed

                if not hash_result.passed:
                    # Note: Hash may differ due to timestamp, so this is informational
                    logger.debug(f"Provenance hash differs (expected in dynamic calculation)")

            # Check performance
            if execution_time_ms > self.performance_threshold_ms:
                failure_reasons.append(
                    f"Performance threshold exceeded: {execution_time_ms:.2f}ms > {self.performance_threshold_ms}ms"
                )

            passed = len(failure_reasons) == 0

            return GoldenTestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                passed=passed,
                execution_time_ms=execution_time_ms,
                comparisons=comparisons,
                provenance_verified=provenance_verified,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                failure_reasons=failure_reasons,
                metadata=test_case.metadata,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Test execution failed: {str(e)}")

            return GoldenTestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                passed=False,
                execution_time_ms=execution_time_ms,
                failure_reasons=[f"Exception during execution: {str(e)}"],
                metadata=test_case.metadata,
            )

    def run_test_suite(
        self,
        tests: List["GoldenTestCase"],
        agent: Any,
        input_class: Any,
        suite_name: str,
    ) -> GoldenTestSuite:
        """Run a complete test suite.

        Args:
            tests: List of test cases to run
            agent: The agent instance to test
            input_class: The input model class
            suite_name: Name of the test suite

        Returns:
            GoldenTestSuite with all results
        """
        suite = GoldenTestSuite(
            suite_name=suite_name,
            agent_id=getattr(agent, "AGENT_ID", "unknown"),
            agent_version=getattr(agent, "VERSION", "unknown"),
            start_time=datetime.utcnow(),
        )

        for test in tests:
            result = self.run_single_test(test, agent, input_class)
            suite.results.append(result)

            if result.passed:
                logger.info(f"PASS: {test.test_name}")
            else:
                logger.warning(f"FAIL: {test.test_name} - {result.failure_reasons}")

        suite.end_time = datetime.utcnow()

        logger.info(
            f"Suite '{suite_name}' complete: "
            f"{suite.passed_tests}/{suite.total_tests} passed "
            f"({suite.pass_rate:.1f}%) in {suite.duration_ms:.2f}ms"
        )

        return suite

    def _compare_values(
        self,
        expected: Any,
        actual: Any,
        field_path: str,
    ) -> ComparisonResult:
        """Compare expected and actual values.

        Handles different types: numeric, string, boolean, list, dict.
        """
        # Handle None
        if expected is None and actual is None:
            return ComparisonResult(passed=True, expected=expected, actual=actual, field_path=field_path)

        if expected is None or actual is None:
            return ComparisonResult(
                passed=False,
                expected=expected,
                actual=actual,
                field_path=field_path,
                failure_reason=f"None mismatch at '{field_path}'",
            )

        # Numeric comparison
        if isinstance(expected, (int, float, Decimal)):
            return self.comparator.compare(expected, actual, field_path)

        # String comparison
        if isinstance(expected, str):
            passed = expected == str(actual)
            return ComparisonResult(
                passed=passed,
                expected=expected,
                actual=actual,
                field_path=field_path,
                failure_reason=None if passed else f"String mismatch at '{field_path}': expected='{expected}', actual='{actual}'",
            )

        # Boolean comparison
        if isinstance(expected, bool):
            passed = expected == actual
            return ComparisonResult(
                passed=passed,
                expected=expected,
                actual=actual,
                field_path=field_path,
                failure_reason=None if passed else f"Boolean mismatch at '{field_path}'",
            )

        # List comparison
        if isinstance(expected, list):
            if not isinstance(actual, list):
                return ComparisonResult(
                    passed=False,
                    expected=expected,
                    actual=actual,
                    field_path=field_path,
                    failure_reason=f"Type mismatch at '{field_path}': expected list",
                )

            if len(expected) != len(actual):
                return ComparisonResult(
                    passed=False,
                    expected=expected,
                    actual=actual,
                    field_path=field_path,
                    failure_reason=f"List length mismatch at '{field_path}': expected {len(expected)}, got {len(actual)}",
                )

            for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                item_result = self._compare_values(exp_item, act_item, f"{field_path}[{i}]")
                if not item_result.passed:
                    return item_result

            return ComparisonResult(passed=True, expected=expected, actual=actual, field_path=field_path)

        # Dict comparison
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return ComparisonResult(
                    passed=False,
                    expected=expected,
                    actual=actual,
                    field_path=field_path,
                    failure_reason=f"Type mismatch at '{field_path}': expected dict",
                )

            for key, exp_val in expected.items():
                if key not in actual:
                    return ComparisonResult(
                        passed=False,
                        expected=expected,
                        actual=actual,
                        field_path=f"{field_path}.{key}",
                        failure_reason=f"Missing key '{key}' at '{field_path}'",
                    )

                val_result = self._compare_values(exp_val, actual[key], f"{field_path}.{key}")
                if not val_result.passed:
                    return val_result

            return ComparisonResult(passed=True, expected=expected, actual=actual, field_path=field_path)

        # Default: exact equality
        passed = expected == actual
        return ComparisonResult(
            passed=passed,
            expected=expected,
            actual=actual,
            field_path=field_path,
            failure_reason=None if passed else f"Value mismatch at '{field_path}'",
        )


# Utility functions for test generation
def generate_test_id(agent_id: str, test_category: str, index: int) -> str:
    """Generate a unique test ID."""
    return f"{agent_id}_{test_category}_{index:04d}"


def calculate_expected_emissions(quantity: float, emission_factor: float) -> float:
    """Calculate expected emissions for validation.

    Formula: emissions = quantity * emission_factor
    """
    return round(quantity * emission_factor, 6)


def validate_provenance_hash(hash_value: str) -> bool:
    """Validate that a provenance hash is properly formatted."""
    return HashVerifier.verify_format(hash_value)
