"""
Golden Test Framework

Expert-validated test scenarios for climate calculations.
Each test has a known-correct answer validated by climate scientists.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from decimal import Decimal
from enum import Enum
import yaml
from pathlib import Path
import time


class TestStatus(str, Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class GoldenTest(BaseModel):
    """
    Golden test with expert-validated expected output.

    Each test represents a real-world calculation scenario with a
    known-correct answer validated by climate scientists or auditors.
    """
    test_id: str
    name: str
    description: str
    category: str  # e.g., "scope1", "cbam", "scope3_transport"

    # Input data
    inputs: Dict[str, Any]

    # Expected output (expert-validated)
    expected_output: float
    expected_unit: str

    # Validation
    tolerance: float = 0.01  # Default ±1% tolerance
    tolerance_type: str = "relative"  # "relative" or "absolute"

    # Metadata
    expert_source: Optional[str] = None  # Who validated this?
    reference_standard: Optional[str] = None  # GHG Protocol, CBAM, etc.
    tags: List[str] = Field(default_factory=list)

    def __str__(self):
        return f"[{self.test_id}] {self.name}"


class GoldenTestResult(BaseModel):
    """Result of running a golden test."""
    test_id: str
    test_name: str
    status: TestStatus

    actual_output: Optional[float] = None
    expected_output: float

    deviation: Optional[float] = None  # Actual deviation from expected
    deviation_pct: Optional[float] = None  # Percentage deviation

    tolerance: float
    tolerance_type: str

    execution_time_ms: float = 0.0

    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_within_tolerance(self) -> bool:
        """Check if result is within tolerance."""
        if self.status != TestStatus.PASSED:
            return False

        if self.tolerance_type == "relative":
            return abs(self.deviation_pct) <= self.tolerance * 100
        else:
            return abs(self.deviation) <= self.tolerance


class GoldenTestRunner:
    """
    Run golden tests and compare against expert-validated results.

    This framework ensures that calculations match known-correct answers
    within specified tolerance (typically ±1%).
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize test runner.

        Args:
            tolerance: Default relative tolerance (e.g., 0.01 = ±1%)
        """
        self.default_tolerance = tolerance
        self.tests: Dict[str, GoldenTest] = {}

    def load_tests_from_yaml(self, yaml_path: Path) -> int:
        """
        Load golden tests from YAML file.

        Args:
            yaml_path: Path to YAML file with test scenarios

        Returns:
            Number of tests loaded
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        tests_data = data.get('golden_tests', [])

        for test_data in tests_data:
            test = GoldenTest(**test_data)
            self.tests[test.test_id] = test

        return len(tests_data)

    def add_test(self, test: GoldenTest):
        """Add a golden test."""
        self.tests[test.test_id] = test

    def run_test(
        self,
        test: GoldenTest,
        calculation_func: Callable[[Dict[str, Any]], float]
    ) -> GoldenTestResult:
        """
        Run a single golden test.

        Args:
            test: Golden test to run
            calculation_func: Function that takes inputs and returns calculated value

        Returns:
            Test result with pass/fail status
        """
        start_time = time.time()

        try:
            # Run calculation
            actual_output = calculation_func(test.inputs)

            # Calculate deviation
            deviation = actual_output - test.expected_output

            if test.expected_output != 0:
                deviation_pct = (deviation / test.expected_output) * 100
            else:
                deviation_pct = 0 if deviation == 0 else float('inf')

            # Determine tolerance
            tolerance = test.tolerance if test.tolerance else self.default_tolerance
            tolerance_type = test.tolerance_type

            # Check if within tolerance
            if tolerance_type == "relative":
                within_tolerance = abs(deviation_pct) <= tolerance * 100
            else:  # absolute
                within_tolerance = abs(deviation) <= tolerance

            status = TestStatus.PASSED if within_tolerance else TestStatus.FAILED

            execution_time_ms = (time.time() - start_time) * 1000

            return GoldenTestResult(
                test_id=test.test_id,
                test_name=test.name,
                status=status,
                actual_output=actual_output,
                expected_output=test.expected_output,
                deviation=deviation,
                deviation_pct=deviation_pct,
                tolerance=tolerance,
                tolerance_type=tolerance_type,
                execution_time_ms=execution_time_ms,
                metadata={
                    'inputs': test.inputs,
                    'category': test.category,
                    'expert_source': test.expert_source
                }
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            return GoldenTestResult(
                test_id=test.test_id,
                test_name=test.name,
                status=TestStatus.ERROR,
                expected_output=test.expected_output,
                tolerance=test.tolerance or self.default_tolerance,
                tolerance_type=test.tolerance_type,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
                metadata={'inputs': test.inputs}
            )

    def run_all_tests(
        self,
        calculation_func: Callable[[Dict[str, Any]], float],
        category_filter: Optional[str] = None
    ) -> List[GoldenTestResult]:
        """
        Run all golden tests.

        Args:
            calculation_func: Function that takes inputs and returns calculated value
            category_filter: Optional category to filter tests

        Returns:
            List of test results
        """
        results = []

        for test in self.tests.values():
            # Apply category filter
            if category_filter and test.category != category_filter:
                continue

            result = self.run_test(test, calculation_func)
            results.append(result)

        return results

    def get_summary(self, results: List[GoldenTestResult]) -> Dict[str, Any]:
        """
        Get summary of test results.

        Args:
            results: List of test results

        Returns:
            Summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)

        avg_execution_time = sum(r.execution_time_ms for r in results) / total if total > 0 else 0

        max_deviation_pct = max(
            (abs(r.deviation_pct) for r in results if r.deviation_pct is not None),
            default=0
        )

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'avg_execution_time_ms': avg_execution_time,
            'max_deviation_pct': max_deviation_pct,
        }

    def print_results(self, results: List[GoldenTestResult], verbose: bool = False):
        """Print test results to console."""
        summary = self.get_summary(results)

        print("\n" + "="*70)
        print("GOLDEN TEST RESULTS")
        print("="*70)

        for result in results:
            status_symbol = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.ERROR: "[ERR ]",
                TestStatus.SKIPPED: "[SKIP]"
            }[result.status]

            print(f"\n{status_symbol} [{result.test_id}] {result.test_name}")

            if result.status == TestStatus.PASSED:
                print(f"  Expected: {result.expected_output:.6f}")
                print(f"  Actual:   {result.actual_output:.6f}")
                print(f"  Deviation: {result.deviation_pct:+.4f}% (within ±{result.tolerance*100}%)")

            elif result.status == TestStatus.FAILED:
                print(f"  Expected: {result.expected_output:.6f}")
                print(f"  Actual:   {result.actual_output:.6f}")
                print(f"  Deviation: {result.deviation_pct:+.4f}% (EXCEEDS ±{result.tolerance*100}%)")
                print(f"  FAILED: Result outside tolerance!")

            elif result.status == TestStatus.ERROR:
                print(f"  ERROR: {result.error_message}")

            if verbose:
                print(f"  Execution time: {result.execution_time_ms:.2f}ms")

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Avg execution time: {summary['avg_execution_time_ms']:.2f}ms")
        print(f"Max deviation: {summary['max_deviation_pct']:.4f}%")
        print("="*70 + "\n")

        return summary['passed'] == summary['total_tests']


class GoldenTestValidator:
    """Validate that golden tests themselves are correct."""

    @staticmethod
    def validate_test(test: GoldenTest) -> List[str]:
        """
        Validate that a golden test is well-formed.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if not test.test_id:
            errors.append("Missing test_id")

        if not test.name:
            errors.append("Missing test name")

        if not test.inputs:
            errors.append("Missing inputs")

        if test.expected_output is None:
            errors.append("Missing expected_output")

        if not test.expected_unit:
            errors.append("Missing expected_unit")

        # Check tolerance is reasonable
        if test.tolerance < 0:
            errors.append(f"Negative tolerance: {test.tolerance}")

        if test.tolerance > 0.10:  # >10%
            errors.append(f"Tolerance too large: {test.tolerance} (>10%)")

        # Check expected output is reasonable
        if test.expected_output < 0:
            errors.append(f"Negative expected output: {test.expected_output}")

        return errors
