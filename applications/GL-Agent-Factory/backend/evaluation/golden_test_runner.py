"""
Golden Test Runner for GreenLang Agents

Loads golden tests from pack.yaml and validates agent outputs against
expected values with configurable tolerance.

Features:
- Load golden tests from pack.yaml
- Run agent with test inputs
- Compare outputs with expected (numeric tolerance support)
- Generate detailed test reports
- Track pass/fail rates

Example:
    >>> runner = GoldenTestRunner()
    >>> result = runner.run_tests("path/to/pack.yaml", agent_instance)
    >>> print(f"Pass rate: {result.pass_rate}%")

"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GoldenTest:
    """Single golden test case."""
    test_id: str
    category: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: Optional[Dict[str, float]] = None  # Field-specific tolerances
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCaseResult:
    """Result of a single test case execution."""
    test_id: str
    passed: bool
    actual_output: Any
    expected_output: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    differences: List[str] = field(default_factory=list)


@dataclass
class GoldenTestResult:
    """Overall golden test execution result."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    test_results: List[TestCaseResult]
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.passed_tests == self.total_tests


class GoldenTestRunner:
    """
    Run golden tests from pack.yaml specification.

    This runner loads golden test cases from pack.yaml and executes
    them against an agent instance, validating outputs match expected
    values within tolerance.
    """

    def __init__(self, default_tolerance: float = 1e-9):
        """
        Initialize golden test runner.

        Args:
            default_tolerance: Default numeric tolerance for float comparisons
        """
        self.default_tolerance = default_tolerance
        logger.info(f"GoldenTestRunner initialized (tolerance={default_tolerance})")

    def load_golden_tests(self, pack_yaml_path: Union[str, Path]) -> List[GoldenTest]:
        """
        Load golden test cases from pack.yaml.

        Args:
            pack_yaml_path: Path to pack.yaml file

        Returns:
            List of GoldenTest objects

        Raises:
            FileNotFoundError: If pack.yaml not found
            ValueError: If pack.yaml invalid or missing golden tests
        """
        pack_yaml_path = Path(pack_yaml_path)
        if not pack_yaml_path.exists():
            raise FileNotFoundError(f"Pack file not found: {pack_yaml_path}")

        with open(pack_yaml_path, 'r', encoding='utf-8') as f:
            pack_spec = yaml.safe_load(f)

        # Try multiple locations for golden tests
        golden_tests_spec = pack_spec.get('tests', {}).get('golden', [])
        if not golden_tests_spec:
            golden_tests_spec = pack_spec.get('golden_tests', {})

        if not golden_tests_spec:
            logger.warning(f"No golden tests found in {pack_yaml_path}")
            return []

        # Load test cases
        test_cases = self._load_test_cases(pack_yaml_path.parent, golden_tests_spec)

        logger.info(f"Loaded {len(test_cases)} golden tests from {pack_yaml_path}")
        return test_cases

    def _load_test_cases(
        self,
        pack_dir: Path,
        golden_tests_spec: Any
    ) -> List[GoldenTest]:
        """
        Load test cases from specification.

        Args:
            pack_dir: Directory containing pack.yaml
            golden_tests_spec: Golden tests section from pack.yaml

        Returns:
            List of GoldenTest objects
        """
        test_cases = []

        # Handle list of test cases directly
        if isinstance(golden_tests_spec, list):
            for i, tc in enumerate(golden_tests_spec):
                test_cases.append(self._parse_test_case(tc, i))
            return test_cases

        # Handle dict with categories or test_cases
        if isinstance(golden_tests_spec, dict):
            inline_cases = golden_tests_spec.get('test_cases', [])
            if inline_cases:
                for i, tc in enumerate(inline_cases):
                    test_cases.append(self._parse_test_case(tc, i))

        return test_cases

    def _parse_test_case(self, tc: Dict[str, Any], index: int) -> GoldenTest:
        """Parse a test case dict into GoldenTest object."""
        return GoldenTest(
            test_id=tc.get('name', tc.get('id', f"test_{index:03d}")),
            category=tc.get('category', 'uncategorized'),
            description=tc.get('description', ''),
            input_data=tc.get('input', {}),
            expected_output=tc.get('expect', tc.get('expected_output', {})),
            tolerance=tc.get('tolerance'),
            metadata=tc.get('metadata', {}),
        )

    def run_tests(
        self,
        pack_yaml_path: Union[str, Path],
        agent: Any,
        verbose: bool = True,
    ) -> GoldenTestResult:
        """
        Run all golden tests for an agent.

        Args:
            pack_yaml_path: Path to pack.yaml
            agent: Agent instance with run() method
            verbose: Print detailed output

        Returns:
            GoldenTestResult with execution summary
        """
        start_time = datetime.utcnow()
        golden_tests = self.load_golden_tests(pack_yaml_path)

        if not golden_tests:
            logger.warning("No golden tests to run")
            return GoldenTestResult(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                pass_rate=0.0,
                test_results=[],
                execution_time_ms=0.0,
            )

        test_results = []
        for test in golden_tests:
            result = self._run_single_test(agent, test, verbose)
            test_results.append(result)

        # Calculate statistics
        passed = sum(1 for r in test_results if r.passed)
        failed = len(test_results) - passed
        pass_rate = (passed / len(test_results)) * 100 if test_results else 0.0

        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = GoldenTestResult(
            total_tests=len(test_results),
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=pass_rate,
            test_results=test_results,
            execution_time_ms=execution_time_ms,
        )

        if verbose:
            self._print_summary(result)

        return result

    def _run_single_test(
        self,
        agent: Any,
        test: GoldenTest,
        verbose: bool = True,
    ) -> TestCaseResult:
        """
        Run a single test case.

        Args:
            agent: Agent instance
            test: Golden test case
            verbose: Print output

        Returns:
            TestCaseResult
        """
        test_start = datetime.utcnow()

        try:
            # Execute agent
            actual_output = agent.run(test.input_data)

            # Compare outputs
            passed, differences = self._compare_outputs(
                actual_output,
                test.expected_output,
                test.tolerance,
            )

            execution_time_ms = (datetime.utcnow() - test_start).total_seconds() * 1000

            result = TestCaseResult(
                test_id=test.test_id,
                passed=passed,
                actual_output=actual_output,
                expected_output=test.expected_output,
                differences=differences,
                execution_time_ms=execution_time_ms,
            )

            if verbose:
                status = "PASS" if passed else "FAIL"
                logger.info(f"[{status}] {test.test_id}: {test.description}")
                if not passed and differences:
                    for diff in differences:
                        logger.warning(f"  - {diff}")

            return result

        except Exception as e:
            execution_time_ms = (datetime.utcnow() - test_start).total_seconds() * 1000

            logger.error(f"[ERROR] {test.test_id}: {str(e)}")

            return TestCaseResult(
                test_id=test.test_id,
                passed=False,
                actual_output=None,
                expected_output=test.expected_output,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
            )

    def _compare_outputs(
        self,
        actual: Any,
        expected: Any,
        tolerance: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Compare actual vs expected outputs with tolerance.

        Args:
            actual: Actual output from agent
            expected: Expected output
            tolerance: Field-specific tolerances

        Returns:
            Tuple of (passed, differences)
        """
        differences = []

        # Convert to dicts if they're Pydantic models
        if hasattr(actual, 'dict'):
            actual = actual.dict()
        if hasattr(expected, 'dict'):
            expected = expected.dict()

        # Compare as dicts
        if isinstance(expected, dict):
            for key, expected_value in expected.items():
                actual_value = None
                if isinstance(actual, dict):
                    actual_value = actual.get(key)
                elif hasattr(actual, key):
                    actual_value = getattr(actual, key)

                if actual_value is None and key not in (actual if isinstance(actual, dict) else {}):
                    differences.append(f"Missing field: {key}")
                    continue

                # Get tolerance for this field
                field_tolerance = (
                    tolerance.get(key, self.default_tolerance)
                    if tolerance
                    else self.default_tolerance
                )

                # Compare with tolerance
                if not self._values_equal(actual_value, expected_value, field_tolerance):
                    differences.append(
                        f"Field '{key}': expected {expected_value}, got {actual_value}"
                    )

        # Direct comparison
        else:
            if not self._values_equal(actual, expected, self.default_tolerance):
                differences.append(f"Expected {expected}, got {actual}")

        return len(differences) == 0, differences

    def _values_equal(
        self,
        actual: Any,
        expected: Any,
        tolerance: float,
    ) -> bool:
        """
        Check if two values are equal within tolerance.

        Args:
            actual: Actual value
            expected: Expected value
            tolerance: Numeric tolerance for floats

        Returns:
            True if values match within tolerance
        """
        # Numeric comparison with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected) <= tolerance

        # Exact match for non-numeric types
        if type(actual) != type(expected):
            return False

        # List comparison
        if isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            return all(
                self._values_equal(a, e, tolerance)
                for a, e in zip(actual, expected)
            )

        # Dict comparison
        if isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(
                self._values_equal(actual[k], expected[k], tolerance)
                for k in expected.keys()
            )

        # Direct equality
        return actual == expected

    def _print_summary(self, result: GoldenTestResult) -> None:
        """Print test execution summary."""
        print("\n" + "=" * 80)
        print("GOLDEN TEST RESULTS")
        print("=" * 80)
        print(f"Total tests:   {result.total_tests}")
        print(f"Passed:        {result.passed_tests}")
        print(f"Failed:        {result.failed_tests}")
        print(f"Pass rate:     {result.pass_rate:.2f}%")
        print(f"Execution:     {result.execution_time_ms:.2f}ms")
        print("=" * 80)

        if result.failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_result in result.test_results:
                if not test_result.passed:
                    print(f"\n  {test_result.test_id}:")
                    if test_result.error_message:
                        print(f"    Error: {test_result.error_message}")
                    for diff in test_result.differences:
                        print(f"    - {diff}")
            print()
