# -*- coding: utf-8 -*-
"""
Golden Test Runner Framework for GreenLang Agent Factory
=========================================================

A comprehensive framework for executing golden tests defined in AgentSpec YAML files.
Golden tests are expert-validated test scenarios with known-correct answers for
zero-hallucination compliance verification.

Features:
- Load tests from pack.yaml `tests.golden` section
- Tolerance-based numeric comparisons (relative and absolute)
- String contains/equals assertions
- Async test execution with concurrency control
- Rich console output with pass/fail status
- JSON and JUnit XML output formats
- Detailed test reports with provenance tracking

Author: GreenLang Testing Team
Date: December 2025
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    Awaitable,
)

import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# Type Definitions
# ==============================================================================

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

# Type alias for agent callable
AgentCallable = Union[
    Callable[[Dict[str, Any]], Dict[str, Any]],
    Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
]


# ==============================================================================
# Enums and Status Types
# ==============================================================================


class TestStatus(str, Enum):
    """Test result status enumeration."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    PENDING = "pending"


class ToleranceType(str, Enum):
    """Tolerance comparison type for numeric values."""

    RELATIVE = "relative"  # Percentage-based (e.g., 0.01 = 1%)
    ABSOLUTE = "absolute"  # Fixed value (e.g., 0.5 = +/- 0.5)


class AssertionType(str, Enum):
    """Type of assertion for expected values."""

    NUMERIC = "numeric"
    STRING_EQUALS = "string_equals"
    STRING_CONTAINS = "string_contains"
    BOOLEAN = "boolean"
    ARRAY_LENGTH = "array_length"
    OBJECT_HAS_KEY = "object_has_key"
    REGEX_MATCH = "regex_match"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ExpectedValue:
    """
    Expected value specification with tolerance configuration.

    Supports both numeric comparisons with tolerance and string assertions.

    Attributes:
        value: Expected value (numeric, string, boolean, etc.)
        tolerance: Tolerance for numeric comparisons (default: 0.01 = 1%)
        tolerance_type: Type of tolerance (relative or absolute)
        assertion_type: Type of assertion to perform
        case_sensitive: Whether string comparison is case-sensitive
    """

    value: Any
    tolerance: float = 0.01
    tolerance_type: ToleranceType = ToleranceType.RELATIVE
    assertion_type: Optional[AssertionType] = None
    case_sensitive: bool = True

    @classmethod
    def from_yaml(cls, data: Any) -> "ExpectedValue":
        """
        Create ExpectedValue from YAML data.

        Supports multiple formats:
        - Simple value: {"co2e_kg": 100.0}
        - With tolerance: {"co2e_kg": {"value": 100.0, "tol": 0.01}}
        - With full spec: {"co2e_kg": {"value": 100.0, "tolerance": 0.01, "type": "relative"}}

        Args:
            data: YAML data (value or dict with value and tolerance)

        Returns:
            ExpectedValue instance
        """
        if isinstance(data, dict):
            # Extract value
            value = data.get("value", data.get("val"))
            if value is None:
                raise ValueError(f"Expected value dict must have 'value' key: {data}")

            # Extract tolerance (support both 'tol' and 'tolerance')
            tolerance = data.get("tolerance", data.get("tol", 0.01))

            # Extract tolerance type
            tol_type_str = data.get("type", data.get("tolerance_type", "relative"))
            tolerance_type = ToleranceType(tol_type_str.lower())

            # Extract assertion type if specified
            assertion_type = None
            if "assertion" in data:
                assertion_type = AssertionType(data["assertion"])

            # Case sensitivity for strings
            case_sensitive = data.get("case_sensitive", True)

            return cls(
                value=value,
                tolerance=tolerance,
                tolerance_type=tolerance_type,
                assertion_type=assertion_type,
                case_sensitive=case_sensitive,
            )
        else:
            # Simple value - infer assertion type from value type
            return cls(value=data)

    def __repr__(self) -> str:
        if isinstance(self.value, (int, float)):
            tol_str = (
                f"{self.tolerance * 100:.1f}%"
                if self.tolerance_type == ToleranceType.RELATIVE
                else f"+/-{self.tolerance}"
            )
            return f"ExpectedValue({self.value} {tol_str})"
        return f"ExpectedValue({self.value!r})"


@dataclass
class GoldenTestCase:
    """
    A single golden test case with input, expected output, and metadata.

    Golden tests represent expert-validated scenarios with known-correct answers.

    Attributes:
        name: Unique test name
        description: Human-readable description
        input: Input data for the agent
        expect: Expected output values with tolerances
        tags: Optional tags for filtering
        skip: Whether to skip this test
        skip_reason: Reason for skipping
        timeout_seconds: Test-specific timeout
        expert_source: Who validated this test
        reference_standard: Standard this test validates (GHG Protocol, CBAM, etc.)
    """

    name: str
    input: Dict[str, Any]
    expect: Dict[str, ExpectedValue]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    skip: bool = False
    skip_reason: Optional[str] = None
    timeout_seconds: float = 30.0
    expert_source: Optional[str] = None
    reference_standard: Optional[str] = None

    @classmethod
    def from_yaml(cls, data: Dict[str, Any]) -> "GoldenTestCase":
        """
        Create GoldenTestCase from YAML data.

        Args:
            data: YAML dictionary with test definition

        Returns:
            GoldenTestCase instance
        """
        name = data.get("name", data.get("id", "unnamed_test"))
        description = data.get("description", "")
        input_data = data.get("input", {})

        # Parse expected values
        expect_raw = data.get("expect", data.get("expected", {}))
        expect: Dict[str, ExpectedValue] = {}
        for key, value in expect_raw.items():
            expect[key] = ExpectedValue.from_yaml(value)

        return cls(
            name=name,
            description=description,
            input=input_data,
            expect=expect,
            tags=data.get("tags", []),
            skip=data.get("skip", False),
            skip_reason=data.get("skip_reason"),
            timeout_seconds=data.get("timeout", data.get("timeout_seconds", 30.0)),
            expert_source=data.get("expert_source"),
            reference_standard=data.get("reference_standard"),
        )

    def __repr__(self) -> str:
        return f"GoldenTestCase(name={self.name!r}, inputs={len(self.input)}, expects={len(self.expect)})"


@dataclass
class AssertionResult:
    """
    Result of a single assertion within a test.

    Attributes:
        field: Field name being asserted
        passed: Whether assertion passed
        expected: Expected value
        actual: Actual value
        deviation: Absolute deviation (for numeric)
        deviation_pct: Percentage deviation (for numeric)
        message: Human-readable result message
        tolerance: Tolerance used
        tolerance_type: Type of tolerance used
    """

    field: str
    passed: bool
    expected: Any
    actual: Any
    deviation: Optional[float] = None
    deviation_pct: Optional[float] = None
    message: str = ""
    tolerance: float = 0.01
    tolerance_type: ToleranceType = ToleranceType.RELATIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field": self.field,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "deviation": self.deviation,
            "deviation_pct": self.deviation_pct,
            "message": self.message,
            "tolerance": self.tolerance,
            "tolerance_type": self.tolerance_type.value,
        }


@dataclass
class GoldenTestResult:
    """
    Result of executing a golden test.

    Attributes:
        test_name: Name of the test
        status: Test status (passed, failed, error, skipped)
        assertions: List of assertion results
        execution_time_ms: Test execution time in milliseconds
        error_message: Error message if test errored
        input_data: Input data used
        output_data: Actual output from agent
        provenance_hash: Hash of input/output for reproducibility
        timestamp: When test was executed
        metadata: Additional metadata
    """

    test_name: str
    status: TestStatus
    assertions: List[AssertionResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    @property
    def failed_assertions(self) -> List[AssertionResult]:
        """Get list of failed assertions."""
        return [a for a in self.assertions if not a.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "assertions": [a.to_dict() for a in self.assertions],
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TestSuiteResult:
    """
    Result of executing a full test suite.

    Attributes:
        pack_name: Name of the pack being tested
        pack_version: Version of the pack
        results: List of individual test results
        total_duration_ms: Total execution time
        timestamp: When suite was executed
        metadata: Additional metadata
    """

    pack_name: str
    pack_version: str
    results: List[GoldenTestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Number of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Number of errored tests."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Number of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "skipped": self.skipped,
                "pass_rate": self.pass_rate,
            },
            "results": [r.to_dict() for r in self.results],
            "total_duration_ms": self.total_duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ==============================================================================
# Assertion Engine
# ==============================================================================


class AssertionEngine:
    """
    Engine for comparing actual outputs against expected values.

    Supports numeric tolerance comparisons, string assertions,
    and complex object comparisons.
    """

    def __init__(self, default_tolerance: float = 0.01):
        """
        Initialize assertion engine.

        Args:
            default_tolerance: Default relative tolerance for numeric comparisons
        """
        self.default_tolerance = default_tolerance

    def compare(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """
        Compare actual value against expected value.

        Args:
            field: Field name for reporting
            expected: Expected value specification
            actual: Actual value from output

        Returns:
            AssertionResult with comparison details
        """
        # Handle None actual value
        if actual is None:
            return AssertionResult(
                field=field,
                passed=False,
                expected=expected.value,
                actual=None,
                message=f"Field '{field}' is missing from output",
                tolerance=expected.tolerance,
                tolerance_type=expected.tolerance_type,
            )

        # Determine assertion type
        assertion_type = expected.assertion_type
        if assertion_type is None:
            assertion_type = self._infer_assertion_type(expected.value)

        # Dispatch to appropriate comparison method
        if assertion_type == AssertionType.NUMERIC:
            return self._compare_numeric(field, expected, actual)
        elif assertion_type == AssertionType.STRING_EQUALS:
            return self._compare_string_equals(field, expected, actual)
        elif assertion_type == AssertionType.STRING_CONTAINS:
            return self._compare_string_contains(field, expected, actual)
        elif assertion_type == AssertionType.BOOLEAN:
            return self._compare_boolean(field, expected, actual)
        elif assertion_type == AssertionType.ARRAY_LENGTH:
            return self._compare_array_length(field, expected, actual)
        elif assertion_type == AssertionType.OBJECT_HAS_KEY:
            return self._compare_object_has_key(field, expected, actual)
        else:
            return self._compare_generic(field, expected, actual)

    def _infer_assertion_type(self, value: Any) -> AssertionType:
        """Infer assertion type from value type."""
        if isinstance(value, (int, float, Decimal)):
            return AssertionType.NUMERIC
        elif isinstance(value, bool):
            return AssertionType.BOOLEAN
        elif isinstance(value, str):
            return AssertionType.STRING_EQUALS
        else:
            return AssertionType.NUMERIC  # Default

    def _compare_numeric(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Compare numeric values with tolerance."""
        try:
            expected_val = float(expected.value)
            actual_val = float(actual)
        except (ValueError, TypeError) as e:
            return AssertionResult(
                field=field,
                passed=False,
                expected=expected.value,
                actual=actual,
                message=f"Cannot compare as numeric: {e}",
                tolerance=expected.tolerance,
                tolerance_type=expected.tolerance_type,
            )

        # Calculate deviation
        deviation = actual_val - expected_val

        # Calculate percentage deviation
        if expected_val != 0:
            deviation_pct = (deviation / expected_val) * 100
        else:
            deviation_pct = 0.0 if deviation == 0 else float("inf")

        # Determine if within tolerance
        tolerance = expected.tolerance
        if expected.tolerance_type == ToleranceType.RELATIVE:
            # Relative tolerance: |deviation| <= tolerance * |expected|
            tolerance_value = tolerance * abs(expected_val) if expected_val != 0 else tolerance
            within_tolerance = abs(deviation) <= tolerance_value
            tol_display = f"{tolerance * 100:.2f}%"
        else:
            # Absolute tolerance: |deviation| <= tolerance
            within_tolerance = abs(deviation) <= tolerance
            tol_display = f"+/-{tolerance}"

        if within_tolerance:
            message = (
                f"PASS: {field} = {actual_val:.6g} "
                f"(expected {expected_val:.6g}, deviation {deviation_pct:+.4f}%, within {tol_display})"
            )
        else:
            message = (
                f"FAIL: {field} = {actual_val:.6g} "
                f"(expected {expected_val:.6g}, deviation {deviation_pct:+.4f}%, exceeds {tol_display})"
            )

        return AssertionResult(
            field=field,
            passed=within_tolerance,
            expected=expected_val,
            actual=actual_val,
            deviation=deviation,
            deviation_pct=deviation_pct,
            message=message,
            tolerance=tolerance,
            tolerance_type=expected.tolerance_type,
        )

    def _compare_string_equals(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Compare string equality."""
        expected_str = str(expected.value)
        actual_str = str(actual)

        if expected.case_sensitive:
            passed = expected_str == actual_str
        else:
            passed = expected_str.lower() == actual_str.lower()

        if passed:
            message = f"PASS: {field} = '{actual_str}'"
        else:
            message = f"FAIL: {field} = '{actual_str}' (expected '{expected_str}')"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected_str,
            actual=actual_str,
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )

    def _compare_string_contains(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Check if string contains expected substring."""
        expected_str = str(expected.value)
        actual_str = str(actual)

        if expected.case_sensitive:
            passed = expected_str in actual_str
        else:
            passed = expected_str.lower() in actual_str.lower()

        if passed:
            message = f"PASS: {field} contains '{expected_str}'"
        else:
            message = f"FAIL: {field} does not contain '{expected_str}'"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected_str,
            actual=actual_str,
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )

    def _compare_boolean(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Compare boolean values."""
        expected_bool = bool(expected.value)
        actual_bool = bool(actual)
        passed = expected_bool == actual_bool

        if passed:
            message = f"PASS: {field} = {actual_bool}"
        else:
            message = f"FAIL: {field} = {actual_bool} (expected {expected_bool})"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected_bool,
            actual=actual_bool,
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )

    def _compare_array_length(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Compare array length."""
        if not isinstance(actual, (list, tuple)):
            return AssertionResult(
                field=field,
                passed=False,
                expected=expected.value,
                actual=actual,
                message=f"FAIL: {field} is not an array",
                tolerance=0.0,
                tolerance_type=ToleranceType.ABSOLUTE,
            )

        expected_len = int(expected.value)
        actual_len = len(actual)
        passed = expected_len == actual_len

        if passed:
            message = f"PASS: {field} length = {actual_len}"
        else:
            message = f"FAIL: {field} length = {actual_len} (expected {expected_len})"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected_len,
            actual=actual_len,
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )

    def _compare_object_has_key(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Check if object has expected key."""
        if not isinstance(actual, dict):
            return AssertionResult(
                field=field,
                passed=False,
                expected=expected.value,
                actual=actual,
                message=f"FAIL: {field} is not an object",
                tolerance=0.0,
                tolerance_type=ToleranceType.ABSOLUTE,
            )

        expected_key = str(expected.value)
        passed = expected_key in actual

        if passed:
            message = f"PASS: {field} has key '{expected_key}'"
        else:
            message = f"FAIL: {field} missing key '{expected_key}'"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected_key,
            actual=list(actual.keys()),
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )

    def _compare_generic(
        self,
        field: str,
        expected: ExpectedValue,
        actual: Any,
    ) -> AssertionResult:
        """Generic equality comparison."""
        passed = expected.value == actual

        if passed:
            message = f"PASS: {field} = {actual}"
        else:
            message = f"FAIL: {field} = {actual} (expected {expected.value})"

        return AssertionResult(
            field=field,
            passed=passed,
            expected=expected.value,
            actual=actual,
            message=message,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
        )


# ==============================================================================
# Output Formatters
# ==============================================================================


class OutputFormatter(ABC):
    """Base class for test result formatters."""

    @abstractmethod
    def format(self, suite_result: TestSuiteResult) -> str:
        """Format test suite result."""
        pass

    @abstractmethod
    def write(self, suite_result: TestSuiteResult, path: Path) -> None:
        """Write formatted result to file."""
        pass


class ConsoleFormatter(OutputFormatter):
    """Rich console output formatter with colored pass/fail status."""

    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, verbose: bool = False):
        """
        Initialize console formatter.

        Args:
            use_colors: Whether to use ANSI colors
            verbose: Whether to show detailed output
        """
        self.use_colors = use_colors
        self.verbose = verbose

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def format(self, suite_result: TestSuiteResult) -> str:
        """Format test suite result for console output."""
        lines = []

        # Header
        lines.append("")
        lines.append(self._color("=" * 78, self.BLUE))
        lines.append(self._color(f"GOLDEN TEST RESULTS: {suite_result.pack_name} v{suite_result.pack_version}", self.BOLD))
        lines.append(self._color("=" * 78, self.BLUE))

        # Individual test results
        for result in suite_result.results:
            lines.append("")
            status_icon = self._get_status_icon(result.status)
            status_color = self._get_status_color(result.status)

            lines.append(f"{self._color(status_icon, status_color)} {result.test_name}")

            if result.status == TestStatus.ERROR and result.error_message:
                lines.append(f"  {self._color('ERROR:', self.RED)} {result.error_message}")

            elif result.status == TestStatus.SKIPPED:
                lines.append(f"  {self._color('SKIPPED', self.YELLOW)}")

            else:
                for assertion in result.assertions:
                    if assertion.passed:
                        if self.verbose:
                            lines.append(f"  {self._color('[PASS]', self.GREEN)} {assertion.message}")
                    else:
                        lines.append(f"  {self._color('[FAIL]', self.RED)} {assertion.message}")

            if self.verbose:
                lines.append(f"  {self._color('Time:', self.GRAY)} {result.execution_time_ms:.2f}ms")

        # Summary
        lines.append("")
        lines.append(self._color("=" * 78, self.BLUE))
        lines.append(self._color("SUMMARY", self.BOLD))
        lines.append(self._color("=" * 78, self.BLUE))

        lines.append(f"Total tests:   {suite_result.total}")
        lines.append(f"Passed:        {self._color(str(suite_result.passed), self.GREEN)}")
        lines.append(f"Failed:        {self._color(str(suite_result.failed), self.RED if suite_result.failed > 0 else self.GREEN)}")
        lines.append(f"Errors:        {self._color(str(suite_result.errors), self.RED if suite_result.errors > 0 else self.GREEN)}")
        lines.append(f"Skipped:       {self._color(str(suite_result.skipped), self.YELLOW if suite_result.skipped > 0 else self.GRAY)}")
        lines.append(f"Pass rate:     {suite_result.pass_rate:.1f}%")
        lines.append(f"Duration:      {suite_result.total_duration_ms:.2f}ms")

        lines.append("")
        if suite_result.success:
            lines.append(self._color("ALL TESTS PASSED", self.GREEN + self.BOLD))
        else:
            lines.append(self._color("TESTS FAILED", self.RED + self.BOLD))

        lines.append(self._color("=" * 78, self.BLUE))
        lines.append("")

        return "\n".join(lines)

    def _get_status_icon(self, status: TestStatus) -> str:
        """Get status icon for display."""
        icons = {
            TestStatus.PASSED: "[PASS]",
            TestStatus.FAILED: "[FAIL]",
            TestStatus.ERROR: "[ERR ]",
            TestStatus.SKIPPED: "[SKIP]",
            TestStatus.PENDING: "[PEND]",
        }
        return icons.get(status, "[????]")

    def _get_status_color(self, status: TestStatus) -> str:
        """Get color for status."""
        colors = {
            TestStatus.PASSED: self.GREEN,
            TestStatus.FAILED: self.RED,
            TestStatus.ERROR: self.RED,
            TestStatus.SKIPPED: self.YELLOW,
            TestStatus.PENDING: self.GRAY,
        }
        return colors.get(status, self.RESET)

    def write(self, suite_result: TestSuiteResult, path: Path) -> None:
        """Write to file (strips colors)."""
        # Temporarily disable colors for file output
        original_colors = self.use_colors
        self.use_colors = False
        content = self.format(suite_result)
        self.use_colors = original_colors

        path.write_text(content)
        logger.info(f"Console output written to {path}")


class JSONFormatter(OutputFormatter):
    """JSON output formatter for programmatic consumption."""

    def __init__(self, indent: int = 2, include_details: bool = True):
        """
        Initialize JSON formatter.

        Args:
            indent: JSON indentation level
            include_details: Whether to include detailed assertion info
        """
        self.indent = indent
        self.include_details = include_details

    def format(self, suite_result: TestSuiteResult) -> str:
        """Format test suite result as JSON."""
        data = suite_result.to_dict()

        if not self.include_details:
            # Remove detailed assertion info
            for result in data.get("results", []):
                result.pop("assertions", None)
                result.pop("input_data", None)
                result.pop("output_data", None)

        return json.dumps(data, indent=self.indent, default=str)

    def write(self, suite_result: TestSuiteResult, path: Path) -> None:
        """Write JSON to file."""
        content = self.format(suite_result)
        path.write_text(content)
        logger.info(f"JSON report written to {path}")


class JUnitFormatter(OutputFormatter):
    """JUnit XML output formatter for CI/CD integration."""

    def format(self, suite_result: TestSuiteResult) -> str:
        """Format test suite result as JUnit XML."""
        # Create root element
        testsuites = ET.Element("testsuites")
        testsuites.set("name", f"Golden Tests: {suite_result.pack_name}")
        testsuites.set("tests", str(suite_result.total))
        testsuites.set("failures", str(suite_result.failed))
        testsuites.set("errors", str(suite_result.errors))
        testsuites.set("skipped", str(suite_result.skipped))
        testsuites.set("time", f"{suite_result.total_duration_ms / 1000:.3f}")
        testsuites.set("timestamp", suite_result.timestamp.isoformat())

        # Create testsuite element
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", suite_result.pack_name)
        testsuite.set("tests", str(suite_result.total))
        testsuite.set("failures", str(suite_result.failed))
        testsuite.set("errors", str(suite_result.errors))
        testsuite.set("skipped", str(suite_result.skipped))
        testsuite.set("time", f"{suite_result.total_duration_ms / 1000:.3f}")

        # Add test cases
        for result in suite_result.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", result.test_name)
            testcase.set("classname", f"{suite_result.pack_name}.GoldenTests")
            testcase.set("time", f"{result.execution_time_ms / 1000:.3f}")

            if result.status == TestStatus.FAILED:
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", "Assertions failed")
                failure_details = "\n".join(
                    a.message for a in result.assertions if not a.passed
                )
                failure.text = failure_details

            elif result.status == TestStatus.ERROR:
                error = ET.SubElement(testcase, "error")
                error.set("message", result.error_message or "Unknown error")
                error.text = result.error_message

            elif result.status == TestStatus.SKIPPED:
                skipped = ET.SubElement(testcase, "skipped")
                skipped.set("message", "Test skipped")

        # Convert to string
        return ET.tostring(testsuites, encoding="unicode", method="xml")

    def write(self, suite_result: TestSuiteResult, path: Path) -> None:
        """Write JUnit XML to file."""
        content = '<?xml version="1.0" encoding="UTF-8"?>\n' + self.format(suite_result)
        path.write_text(content)
        logger.info(f"JUnit XML report written to {path}")


# ==============================================================================
# Golden Test Runner
# ==============================================================================


class GoldenTestRunner:
    """
    Runner for executing golden tests from pack.yaml files.

    Loads tests from the `tests.golden` section of pack.yaml,
    executes them against the agent, and generates detailed reports.

    Features:
    - Async test execution with configurable concurrency
    - Tolerance-based numeric comparisons
    - Multiple output formats (console, JSON, JUnit)
    - Provenance tracking for reproducibility
    - Tag-based test filtering

    Example:
        >>> runner = GoldenTestRunner()
        >>> runner.load_tests_from_pack("path/to/pack.yaml")
        >>> result = await runner.run_all(agent_callable)
        >>> runner.print_results(result)
    """

    def __init__(
        self,
        default_tolerance: float = 0.01,
        default_timeout: float = 30.0,
        max_concurrent: int = 5,
    ):
        """
        Initialize Golden Test Runner.

        Args:
            default_tolerance: Default relative tolerance for numeric comparisons (0.01 = 1%)
            default_timeout: Default timeout per test in seconds
            max_concurrent: Maximum concurrent test executions
        """
        self.default_tolerance = default_tolerance
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent

        self.tests: List[GoldenTestCase] = []
        self.pack_name: str = ""
        self.pack_version: str = ""

        self.assertion_engine = AssertionEngine(default_tolerance=default_tolerance)

        # Output formatters
        self.console_formatter = ConsoleFormatter(use_colors=True, verbose=False)
        self.json_formatter = JSONFormatter()
        self.junit_formatter = JUnitFormatter()

        logger.info(
            f"GoldenTestRunner initialized (tolerance={default_tolerance}, "
            f"timeout={default_timeout}s, max_concurrent={max_concurrent})"
        )

    def load_tests_from_pack(self, pack_path: Union[str, Path]) -> int:
        """
        Load golden tests from pack.yaml file.

        Args:
            pack_path: Path to pack.yaml file

        Returns:
            Number of tests loaded

        Raises:
            FileNotFoundError: If pack file not found
            ValueError: If pack file is invalid
        """
        pack_path = Path(pack_path)
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack file not found: {pack_path}")

        with open(pack_path, "r") as f:
            pack_data = yaml.safe_load(f)

        # Extract pack metadata
        self.pack_name = pack_data.get("name", pack_path.parent.name)
        self.pack_version = pack_data.get("version", "0.0.0")

        # Extract golden tests
        tests_section = pack_data.get("tests", {})
        golden_tests = tests_section.get("golden", [])

        self.tests = []
        for test_data in golden_tests:
            try:
                test_case = GoldenTestCase.from_yaml(test_data)
                self.tests.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to parse test: {test_data.get('name', 'unknown')}: {e}")

        logger.info(f"Loaded {len(self.tests)} golden tests from {pack_path}")
        return len(self.tests)

    def load_tests_from_yaml(self, yaml_path: Union[str, Path]) -> int:
        """
        Load golden tests from a standalone YAML file.

        Expected format:
        ```yaml
        golden_tests:
          - name: test_1
            input: {...}
            expect: {...}
        ```

        Args:
            yaml_path: Path to YAML file

        Returns:
            Number of tests loaded
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        golden_tests = data.get("golden_tests", data.get("tests", []))

        self.tests = []
        for test_data in golden_tests:
            try:
                test_case = GoldenTestCase.from_yaml(test_data)
                self.tests.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to parse test: {test_data.get('name', 'unknown')}: {e}")

        logger.info(f"Loaded {len(self.tests)} tests from {yaml_path}")
        return len(self.tests)

    def add_test(self, test: GoldenTestCase) -> None:
        """
        Add a single test case.

        Args:
            test: GoldenTestCase to add
        """
        self.tests.append(test)
        logger.debug(f"Added test: {test.name}")

    def filter_tests(
        self,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
    ) -> List[GoldenTestCase]:
        """
        Filter tests by tags or name pattern.

        Args:
            tags: List of tags to filter by (tests must have at least one)
            name_pattern: Substring to match in test names

        Returns:
            Filtered list of tests
        """
        filtered = self.tests

        if tags:
            filtered = [t for t in filtered if any(tag in t.tags for tag in tags)]

        if name_pattern:
            filtered = [t for t in filtered if name_pattern.lower() in t.name.lower()]

        return filtered

    async def run_test(
        self,
        test: GoldenTestCase,
        agent_callable: AgentCallable,
    ) -> GoldenTestResult:
        """
        Run a single golden test.

        Args:
            test: Test case to run
            agent_callable: Callable that takes input dict and returns output dict

        Returns:
            GoldenTestResult with pass/fail status and details
        """
        start_time = time.time()

        # Check if test should be skipped
        if test.skip:
            return GoldenTestResult(
                test_name=test.name,
                status=TestStatus.SKIPPED,
                input_data=test.input,
                metadata={"skip_reason": test.skip_reason},
            )

        try:
            # Execute agent with timeout
            timeout = test.timeout_seconds or self.default_timeout

            if asyncio.iscoroutinefunction(agent_callable):
                output = await asyncio.wait_for(
                    agent_callable(test.input),
                    timeout=timeout,
                )
            else:
                # Wrap sync function in executor
                loop = asyncio.get_event_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, agent_callable, test.input),
                    timeout=timeout,
                )

            execution_time_ms = (time.time() - start_time) * 1000

            # Ensure output is a dict
            if not isinstance(output, dict):
                output = {"result": output}

            # Run assertions
            assertions = []
            all_passed = True

            for field, expected_value in test.expect.items():
                actual = self._get_nested_value(output, field)
                assertion_result = self.assertion_engine.compare(field, expected_value, actual)
                assertions.append(assertion_result)
                if not assertion_result.passed:
                    all_passed = False

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(test.input, output)

            status = TestStatus.PASSED if all_passed else TestStatus.FAILED

            return GoldenTestResult(
                test_name=test.name,
                status=status,
                assertions=assertions,
                execution_time_ms=execution_time_ms,
                input_data=test.input,
                output_data=output,
                provenance_hash=provenance_hash,
                metadata={
                    "description": test.description,
                    "expert_source": test.expert_source,
                    "reference_standard": test.reference_standard,
                },
            )

        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            return GoldenTestResult(
                test_name=test.name,
                status=TestStatus.ERROR,
                error_message=f"Test timed out after {timeout}s",
                execution_time_ms=execution_time_ms,
                input_data=test.input,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Test {test.name} failed with error: {e}")
            return GoldenTestResult(
                test_name=test.name,
                status=TestStatus.ERROR,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                input_data=test.input,
            )

    async def run_all(
        self,
        agent_callable: AgentCallable,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
    ) -> TestSuiteResult:
        """
        Run all golden tests (or filtered subset).

        Args:
            agent_callable: Callable that takes input dict and returns output dict
            tags: Optional tags to filter tests
            name_pattern: Optional name pattern to filter tests

        Returns:
            TestSuiteResult with all test results
        """
        start_time = time.time()

        # Filter tests if needed
        tests_to_run = self.filter_tests(tags=tags, name_pattern=name_pattern)

        if not tests_to_run:
            logger.warning("No tests to run after filtering")
            return TestSuiteResult(
                pack_name=self.pack_name,
                pack_version=self.pack_version,
            )

        logger.info(f"Running {len(tests_to_run)} golden tests...")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(test: GoldenTestCase) -> GoldenTestResult:
            async with semaphore:
                return await self.run_test(test, agent_callable)

        # Run tests concurrently
        results = await asyncio.gather(*[run_with_semaphore(t) for t in tests_to_run])

        total_duration_ms = (time.time() - start_time) * 1000

        suite_result = TestSuiteResult(
            pack_name=self.pack_name,
            pack_version=self.pack_version,
            results=list(results),
            total_duration_ms=total_duration_ms,
        )

        logger.info(
            f"Test run complete: {suite_result.passed}/{suite_result.total} passed "
            f"({suite_result.pass_rate:.1f}%)"
        )

        return suite_result

    def run_sync(
        self,
        agent_callable: AgentCallable,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
    ) -> TestSuiteResult:
        """
        Synchronous wrapper for run_all.

        Args:
            agent_callable: Callable that takes input dict and returns output dict
            tags: Optional tags to filter tests
            name_pattern: Optional name pattern to filter tests

        Returns:
            TestSuiteResult with all test results
        """
        return asyncio.run(self.run_all(agent_callable, tags=tags, name_pattern=name_pattern))

    def print_results(
        self,
        suite_result: TestSuiteResult,
        verbose: bool = False,
        use_colors: bool = True,
    ) -> None:
        """
        Print test results to console.

        Args:
            suite_result: Test suite result to print
            verbose: Whether to show detailed output
            use_colors: Whether to use ANSI colors
        """
        self.console_formatter.use_colors = use_colors
        self.console_formatter.verbose = verbose
        print(self.console_formatter.format(suite_result))

    def export_json(
        self,
        suite_result: TestSuiteResult,
        output_path: Union[str, Path],
        include_details: bool = True,
    ) -> None:
        """
        Export test results to JSON file.

        Args:
            suite_result: Test suite result to export
            output_path: Path for output file
            include_details: Whether to include detailed assertion info
        """
        self.json_formatter.include_details = include_details
        self.json_formatter.write(suite_result, Path(output_path))

    def export_junit(
        self,
        suite_result: TestSuiteResult,
        output_path: Union[str, Path],
    ) -> None:
        """
        Export test results to JUnit XML file.

        Args:
            suite_result: Test suite result to export
            output_path: Path for output file
        """
        self.junit_formatter.write(suite_result, Path(output_path))

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """
        Get nested value from dict using dot notation.

        Args:
            data: Dictionary to search
            key: Key with optional dot notation (e.g., "result.co2e_kg")

        Returns:
            Value at key or None if not found
        """
        if "." not in key:
            return data.get(key)

        parts = key.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 hash of input/output for reproducibility tracking.

        Args:
            input_data: Test input data
            output_data: Agent output data

        Returns:
            SHA-256 hash string
        """
        combined = {
            "input": input_data,
            "output": output_data,
        }
        json_str = json.dumps(combined, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# ==============================================================================
# Convenience Functions
# ==============================================================================


def run_golden_tests(
    pack_path: Union[str, Path],
    agent_callable: AgentCallable,
    *,
    tolerance: float = 0.01,
    verbose: bool = False,
    export_json: Optional[Union[str, Path]] = None,
    export_junit: Optional[Union[str, Path]] = None,
) -> TestSuiteResult:
    """
    Convenience function to run golden tests from a pack.yaml file.

    Args:
        pack_path: Path to pack.yaml file
        agent_callable: Agent callable (sync or async)
        tolerance: Default tolerance for numeric comparisons
        verbose: Whether to show verbose console output
        export_json: Optional path to export JSON report
        export_junit: Optional path to export JUnit XML report

    Returns:
        TestSuiteResult with all test results
    """
    runner = GoldenTestRunner(default_tolerance=tolerance)
    runner.load_tests_from_pack(pack_path)

    result = runner.run_sync(agent_callable)

    runner.print_results(result, verbose=verbose)

    if export_json:
        runner.export_json(result, export_json)

    if export_junit:
        runner.export_junit(result, export_junit)

    return result


async def run_golden_tests_async(
    pack_path: Union[str, Path],
    agent_callable: AgentCallable,
    *,
    tolerance: float = 0.01,
    max_concurrent: int = 5,
) -> TestSuiteResult:
    """
    Async convenience function to run golden tests.

    Args:
        pack_path: Path to pack.yaml file
        agent_callable: Agent callable (sync or async)
        tolerance: Default tolerance for numeric comparisons
        max_concurrent: Maximum concurrent test executions

    Returns:
        TestSuiteResult with all test results
    """
    runner = GoldenTestRunner(default_tolerance=tolerance, max_concurrent=max_concurrent)
    runner.load_tests_from_pack(pack_path)
    return await runner.run_all(agent_callable)
