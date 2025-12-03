#!/usr/bin/env python
"""
GreenLang Golden Test Runner
============================

Executes golden tests for all agents and validates outputs against
expected reference data with tolerance checking and hash verification.

Usage:
    python tests/run_golden_tests.py                    # Run all tests
    python tests/run_golden_tests.py --suite eudr      # Run EUDR suite only
    python tests/run_golden_tests.py --verbose         # Verbose output
    python tests/run_golden_tests.py --update          # Update expected outputs

Exit Codes:
    0 - All tests passed
    1 - One or more tests failed
    2 - Configuration/setup error
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TOLERANCE = 0.01  # 1% relative tolerance for calculations
HASH_ALGORITHM = "sha256"
GOLDEN_TESTS_DIR = PROJECT_ROOT / "tests" / "golden"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GoldenTestCase:
    """A single golden test case."""
    test_id: str
    description: str
    tool_name: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: float = DEFAULT_TOLERANCE
    tags: List[str] = field(default_factory=list)
    skip: bool = False
    skip_reason: str = ""


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: str
    passed: bool
    actual_output: Optional[Dict[str, Any]] = None
    error_message: str = ""
    execution_time_ms: float = 0.0
    validation_details: List[str] = field(default_factory=list)


@dataclass
class SuiteResult:
    """Result of a test suite execution."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    execution_time_ms: float
    test_results: List[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


# =============================================================================
# Tolerance Checking
# =============================================================================

def compare_values(actual: Any, expected: Any, tolerance: float = DEFAULT_TOLERANCE) -> Tuple[bool, str]:
    """
    Compare two values with tolerance for numeric types.

    Args:
        actual: The actual value from the test
        expected: The expected value from golden data
        tolerance: Relative tolerance for numeric comparisons (default 1%)

    Returns:
        Tuple of (is_equal, message)
    """
    # Handle None cases
    if actual is None and expected is None:
        return True, "Both values are None"
    if actual is None or expected is None:
        return False, f"Value mismatch: actual={actual}, expected={expected}"

    # Handle numeric types
    if isinstance(expected, (int, float, Decimal)):
        if not isinstance(actual, (int, float, Decimal)):
            return False, f"Type mismatch: expected numeric, got {type(actual).__name__}"

        actual_float = float(actual)
        expected_float = float(expected)

        # Handle zero case
        if expected_float == 0:
            if actual_float == 0:
                return True, "Both values are zero"
            return abs(actual_float) < tolerance, f"Expected 0, got {actual_float}"

        # Relative comparison
        relative_error = abs(actual_float - expected_float) / abs(expected_float)
        if relative_error <= tolerance:
            return True, f"Within tolerance: {relative_error:.4%} <= {tolerance:.4%}"
        return False, f"Exceeds tolerance: {relative_error:.4%} > {tolerance:.4%} (actual={actual_float}, expected={expected_float})"

    # Handle strings
    if isinstance(expected, str):
        if actual == expected:
            return True, "Strings match exactly"
        return False, f"String mismatch: '{actual}' != '{expected}'"

    # Handle booleans
    if isinstance(expected, bool):
        if actual == expected:
            return True, "Booleans match"
        return False, f"Boolean mismatch: {actual} != {expected}"

    # Handle lists
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, f"Type mismatch: expected list, got {type(actual).__name__}"
        if len(actual) != len(expected):
            return False, f"List length mismatch: {len(actual)} != {len(expected)}"

        for i, (a, e) in enumerate(zip(actual, expected)):
            is_equal, msg = compare_values(a, e, tolerance)
            if not is_equal:
                return False, f"List item [{i}]: {msg}"
        return True, "Lists match"

    # Handle dicts
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch: expected dict, got {type(actual).__name__}"

        # Check expected keys exist in actual
        for key in expected:
            if key not in actual:
                return False, f"Missing key: '{key}'"
            is_equal, msg = compare_values(actual[key], expected[key], tolerance)
            if not is_equal:
                return False, f"Key '{key}': {msg}"
        return True, "Dicts match"

    # Default comparison
    if actual == expected:
        return True, "Values match exactly"
    return False, f"Value mismatch: {actual} != {expected}"


def verify_provenance_hash(output: Dict[str, Any], expected_hash: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verify provenance hash if present.

    Args:
        output: The output data containing a hash
        expected_hash: Optional expected hash value

    Returns:
        Tuple of (is_valid, message)
    """
    # Look for hash fields in various locations
    hash_keys = ["provenance_hash", "dds_hash", "trace_hash", "hash"]
    actual_hash = None

    for key in hash_keys:
        if key in output:
            actual_hash = output[key]
            break

    if actual_hash is None:
        return True, "No hash field found (not required)"

    # Validate hash format (SHA-256 = 64 hex characters)
    if not isinstance(actual_hash, str):
        return False, f"Hash must be string, got {type(actual_hash).__name__}"

    if len(actual_hash) != 64:
        return False, f"Hash length must be 64, got {len(actual_hash)}"

    try:
        int(actual_hash, 16)
    except ValueError:
        return False, "Hash must be hexadecimal"

    if expected_hash and actual_hash != expected_hash:
        return False, f"Hash mismatch: {actual_hash} != {expected_hash}"

    return True, f"Valid hash: {actual_hash[:16]}..."


# =============================================================================
# Test Loading
# =============================================================================

def load_test_cases(json_file: Path) -> List[GoldenTestCase]:
    """
    Load test cases from a JSON file.

    Args:
        json_file: Path to the JSON test file

    Returns:
        List of GoldenTestCase objects
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = []

    # Handle both single test and array formats
    tests = data.get("tests", [data]) if isinstance(data.get("tests"), list) else [data]

    for test in tests:
        if "test_id" not in test:
            continue

        test_case = GoldenTestCase(
            test_id=test["test_id"],
            description=test.get("description", ""),
            tool_name=test.get("tool_name", ""),
            input_data=test.get("input", {}),
            expected_output=test.get("expected_output", {}),
            tolerance=test.get("tolerance", DEFAULT_TOLERANCE),
            tags=test.get("tags", []),
            skip=test.get("skip", False),
            skip_reason=test.get("skip_reason", "")
        )
        test_cases.append(test_case)

    return test_cases


def discover_test_files(suite_dir: Path) -> List[Path]:
    """
    Discover all JSON test files in a suite directory.

    Args:
        suite_dir: Directory to search for test files

    Returns:
        List of paths to JSON test files
    """
    test_files = []

    for json_file in suite_dir.rglob("*.json"):
        # Skip non-test files
        if json_file.name.startswith("_") or json_file.name.startswith("."):
            continue
        test_files.append(json_file)

    return sorted(test_files)


# =============================================================================
# Tool Loading
# =============================================================================

# Cached tool instances
_tool_cache: Dict[str, Any] = {}

def get_tool_executor(tool_name: str) -> Optional[callable]:
    """
    Get the executor function for a tool.

    Args:
        tool_name: Name of the tool (e.g., "validate_geolocation")

    Returns:
        Callable function or None if not found
    """
    if tool_name in _tool_cache:
        return _tool_cache[tool_name]

    # Map tool names to their module locations
    tool_map = {
        "validate_geolocation": ("greenlang.tools.eudr", "validate_geolocation"),
        "classify_commodity": ("greenlang.tools.eudr", "classify_commodity"),
        "assess_country_risk": ("greenlang.tools.eudr", "assess_country_risk"),
        "trace_supply_chain": ("greenlang.tools.eudr", "trace_supply_chain"),
        "generate_dds_report": ("greenlang.tools.eudr", "generate_dds_report"),
    }

    if tool_name not in tool_map:
        return None

    module_name, func_name = tool_map[tool_name]

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        _tool_cache[tool_name] = func
        return func
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load tool '{tool_name}': {e}")
        return None


# =============================================================================
# Test Execution
# =============================================================================

def execute_test(test_case: GoldenTestCase, verbose: bool = False) -> TestResult:
    """
    Execute a single golden test case.

    Args:
        test_case: The test case to execute
        verbose: Enable verbose output

    Returns:
        TestResult object
    """
    import time

    result = TestResult(
        test_id=test_case.test_id,
        passed=False
    )

    # Check for skip
    if test_case.skip:
        result.passed = True
        result.error_message = f"SKIPPED: {test_case.skip_reason}"
        return result

    # Get the tool executor
    executor = get_tool_executor(test_case.tool_name)
    if executor is None:
        result.error_message = f"Tool not found: {test_case.tool_name}"
        return result

    # Execute the test
    start_time = time.perf_counter()
    try:
        actual_output = executor(**test_case.input_data)
        result.actual_output = actual_output
    except Exception as e:
        result.error_message = f"Execution error: {str(e)}"
        result.execution_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    result.execution_time_ms = (time.perf_counter() - start_time) * 1000

    # Compare outputs
    is_equal, message = compare_values(
        actual_output,
        test_case.expected_output,
        test_case.tolerance
    )

    result.validation_details.append(message)

    if not is_equal:
        result.error_message = message
        return result

    # Verify provenance hash if expected
    if "expected_hash" in test_case.expected_output:
        is_valid, hash_msg = verify_provenance_hash(
            actual_output,
            test_case.expected_output.get("expected_hash")
        )
        result.validation_details.append(hash_msg)

        if not is_valid:
            result.error_message = hash_msg
            return result

    result.passed = True
    return result


def run_suite(suite_dir: Path, verbose: bool = False) -> SuiteResult:
    """
    Run all tests in a suite directory.

    Args:
        suite_dir: Directory containing test files
        verbose: Enable verbose output

    Returns:
        SuiteResult object
    """
    import time

    suite_name = suite_dir.name
    test_files = discover_test_files(suite_dir)

    result = SuiteResult(
        suite_name=suite_name,
        total_tests=0,
        passed=0,
        failed=0,
        skipped=0,
        execution_time_ms=0.0
    )

    start_time = time.perf_counter()

    for test_file in test_files:
        try:
            test_cases = load_test_cases(test_file)
        except Exception as e:
            print(f"  Warning: Could not load {test_file}: {e}")
            continue

        for test_case in test_cases:
            result.total_tests += 1

            if verbose:
                print(f"  Running: {test_case.test_id}...", end=" ")

            test_result = execute_test(test_case, verbose)
            result.test_results.append(test_result)

            if test_case.skip:
                result.skipped += 1
                status = "SKIP"
            elif test_result.passed:
                result.passed += 1
                status = "PASS"
            else:
                result.failed += 1
                status = "FAIL"

            if verbose:
                print(f"[{status}]")
                if not test_result.passed and not test_case.skip:
                    print(f"    Error: {test_result.error_message}")

    result.execution_time_ms = (time.perf_counter() - start_time) * 1000
    return result


# =============================================================================
# Report Generation
# =============================================================================

def print_summary(results: List[SuiteResult]) -> None:
    """Print a summary of all test results."""
    print("\n" + "=" * 80)
    print(" GOLDEN TEST SUMMARY")
    print("=" * 80)

    total_tests = sum(r.total_tests for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_time = sum(r.execution_time_ms for r in results)

    print(f"\n{'Suite':<30} {'Tests':>8} {'Passed':>8} {'Failed':>8} {'Skip':>8} {'Rate':>8}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.suite_name:<30} "
            f"{result.total_tests:>8} "
            f"{result.passed:>8} "
            f"{result.failed:>8} "
            f"{result.skipped:>8} "
            f"{result.pass_rate:>7.1f}%"
        )

    print("-" * 80)
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(
        f"{'TOTAL':<30} "
        f"{total_tests:>8} "
        f"{total_passed:>8} "
        f"{total_failed:>8} "
        f"{total_skipped:>8} "
        f"{overall_rate:>7.1f}%"
    )

    print(f"\nTotal execution time: {total_time:.1f}ms")

    if total_failed == 0:
        print("\n[SUCCESS] All golden tests passed!")
    else:
        print(f"\n[FAILURE] {total_failed} test(s) failed")


def generate_json_report(results: List[SuiteResult], output_path: Path) -> None:
    """Generate a JSON report of test results."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_tests": sum(r.total_tests for r in results),
            "passed": sum(r.passed for r in results),
            "failed": sum(r.failed for r in results),
            "skipped": sum(r.skipped for r in results),
            "execution_time_ms": sum(r.execution_time_ms for r in results)
        },
        "suites": []
    }

    for result in results:
        suite_data = {
            "name": result.suite_name,
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "pass_rate": result.pass_rate,
            "execution_time_ms": result.execution_time_ms,
            "tests": []
        }

        for test_result in result.test_results:
            suite_data["tests"].append({
                "test_id": test_result.test_id,
                "passed": test_result.passed,
                "error_message": test_result.error_message,
                "execution_time_ms": test_result.execution_time_ms,
                "validation_details": test_result.validation_details
            })

        report["suites"].append(suite_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nJSON report saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the golden test runner."""
    parser = argparse.ArgumentParser(
        description="GreenLang Golden Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Run specific test suite (e.g., 'eudr_compliance', 'fuel_emissions')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Generate JSON report to specified file"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Relative tolerance for numeric comparisons (default: {DEFAULT_TOLERANCE})"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" GreenLang Golden Test Runner")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Tolerance: +/-{args.tolerance * 100:.1f}%")
    print("=" * 80)

    # Determine which suites to run
    if args.suite:
        suite_dirs = [GOLDEN_TESTS_DIR / args.suite]
        if not suite_dirs[0].exists():
            print(f"Error: Suite directory not found: {suite_dirs[0]}")
            return 2
    else:
        suite_dirs = [
            d for d in GOLDEN_TESTS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

    if not suite_dirs:
        print("No test suites found")
        return 2

    # Run all suites
    results = []
    for suite_dir in suite_dirs:
        print(f"\nRunning suite: {suite_dir.name}")
        result = run_suite(suite_dir, args.verbose)
        results.append(result)

    # Print summary
    print_summary(results)

    # Generate report if requested
    if args.report:
        generate_json_report(results, Path(args.report))

    # Return exit code
    total_failed = sum(r.failed for r in results)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
