"""
Dimension 04: Accuracy Verification

This dimension verifies that agents pass golden tests with 100% accuracy
and handle edge cases correctly.

Checks:
    - Golden test pass rate (100% required)
    - Tolerance validation
    - Edge case handling
    - Boundary condition testing

Example:
    >>> dimension = AccuracyDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class AccuracyDimension(BaseDimension):
    """
    Accuracy Dimension Evaluator (D04).

    Verifies that agents produce accurate results by running
    golden tests and validating outputs.

    Configuration:
        required_pass_rate: Required golden test pass rate (default: 100.0)
        default_tolerance: Default numeric tolerance (default: 1e-6)
        check_edge_cases: Whether to check edge cases (default: True)
    """

    DIMENSION_ID = "D04"
    DIMENSION_NAME = "Accuracy"
    DESCRIPTION = "Verifies golden test pass rate and edge case handling"
    WEIGHT = 1.5
    REQUIRED_FOR_CERTIFICATION = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize accuracy dimension evaluator."""
        super().__init__(config)

        self.required_pass_rate = self.config.get("required_pass_rate", 100.0)
        self.default_tolerance = self.config.get("default_tolerance", 1e-6)
        self.check_edge_cases = self.config.get("check_edge_cases", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate accuracy for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Agent instance with run() method
            sample_input: Optional sample input

        Returns:
            DimensionResult with accuracy evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting accuracy evaluation")

        if agent is None:
            agent = self._load_agent(agent_path)

        if agent is None:
            self._add_check(
                name="agent_load",
                passed=False,
                message="Failed to load agent instance",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Load golden tests
        golden_tests = self._load_golden_tests(agent_path)

        if not golden_tests:
            self._add_check(
                name="golden_tests_exist",
                passed=False,
                message="No golden tests found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        self._add_check(
            name="golden_tests_exist",
            passed=True,
            message=f"Found {len(golden_tests)} golden test(s)",
            severity="info",
        )

        # Run golden tests
        test_results = self._run_golden_tests(agent, golden_tests)

        passed_count = sum(1 for r in test_results if r["passed"])
        total_count = len(test_results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        self._add_check(
            name="golden_test_pass_rate",
            passed=pass_rate >= self.required_pass_rate,
            message=f"Golden tests: {passed_count}/{total_count} passed ({pass_rate:.1f}%)",
            severity="error" if pass_rate < self.required_pass_rate else "info",
            details={
                "passed": passed_count,
                "total": total_count,
                "pass_rate": pass_rate,
                "failures": [r for r in test_results if not r["passed"]][:5],
            },
        )

        # Check tolerance handling
        tolerance_check = self._check_tolerance_handling(test_results)
        self._add_check(
            name="tolerance_handling",
            passed=tolerance_check["all_within_tolerance"],
            message="All results within specified tolerances"
            if tolerance_check["all_within_tolerance"]
            else f"{tolerance_check['violations']} tolerance violation(s)",
            severity="error" if not tolerance_check["all_within_tolerance"] else "info",
            details=tolerance_check,
        )

        # Check edge cases if enabled
        if self.check_edge_cases:
            edge_case_results = self._check_edge_cases(agent, agent_path)
            passed_edge = sum(1 for r in edge_case_results if r["passed"])
            total_edge = len(edge_case_results)

            self._add_check(
                name="edge_case_handling",
                passed=passed_edge == total_edge or total_edge == 0,
                message=f"Edge cases: {passed_edge}/{total_edge} handled correctly"
                if total_edge > 0
                else "No edge case tests defined",
                severity="warning" if passed_edge < total_edge else "info",
                details={
                    "passed": passed_edge,
                    "total": total_edge,
                    "failures": [r for r in edge_case_results if not r["passed"]],
                },
            )

        # Check boundary conditions
        boundary_results = self._check_boundary_conditions(agent)
        self._add_check(
            name="boundary_conditions",
            passed=boundary_results["all_passed"],
            message="Boundary conditions handled correctly"
            if boundary_results["all_passed"]
            else f"{boundary_results['failures']} boundary condition failure(s)",
            severity="warning" if not boundary_results["all_passed"] else "info",
            details=boundary_results,
        )

        # Check numeric precision
        precision_check = self._check_numeric_precision(test_results)
        self._add_check(
            name="numeric_precision",
            passed=precision_check["consistent"],
            message="Numeric precision is consistent"
            if precision_check["consistent"]
            else "Numeric precision issues detected",
            severity="warning" if not precision_check["consistent"] else "info",
            details=precision_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "golden_tests_total": total_count,
                "golden_tests_passed": passed_count,
                "pass_rate": pass_rate,
                "tolerance_used": self.default_tolerance,
            },
        )

    def _load_agent(self, agent_path: Path) -> Optional[Any]:
        """Load agent from path."""
        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return None

            import importlib.util

            spec = importlib.util.spec_from_file_location("agent", agent_file)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and name.endswith("Agent")
                    and hasattr(obj, "run")
                ):
                    return obj()

            return None

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            return None

    def _load_golden_tests(self, agent_path: Path) -> List[Dict[str, Any]]:
        """
        Load golden tests from agent directory.

        Args:
            agent_path: Path to agent directory

        Returns:
            List of golden test cases
        """
        golden_tests = []

        try:
            # Try pack.yaml
            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                tests = pack_spec.get("tests", {}).get("golden", [])
                golden_tests.extend(tests)

            # Try golden_tests.yaml
            golden_file = agent_path / "golden_tests.yaml"
            if golden_file.exists():
                with open(golden_file, "r", encoding="utf-8") as f:
                    golden_spec = yaml.safe_load(f)

                tests = golden_spec.get("test_cases", [])
                golden_tests.extend(tests)

            # Try tests directory
            tests_dir = agent_path / "tests"
            if tests_dir.exists():
                for test_file in tests_dir.glob("golden*.yaml"):
                    with open(test_file, "r", encoding="utf-8") as f:
                        test_spec = yaml.safe_load(f)

                    if isinstance(test_spec, list):
                        golden_tests.extend(test_spec)
                    elif isinstance(test_spec, dict):
                        tests = test_spec.get("test_cases", test_spec.get("tests", []))
                        golden_tests.extend(tests)

        except Exception as e:
            logger.error(f"Failed to load golden tests: {str(e)}")

        return golden_tests

    def _run_golden_tests(
        self,
        agent: Any,
        golden_tests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run golden tests against agent.

        Args:
            agent: Agent instance
            golden_tests: List of test cases

        Returns:
            List of test results
        """
        results = []

        for i, test in enumerate(golden_tests):
            test_name = test.get("name", f"test_{i + 1}")
            test_input = test.get("input", {})
            expected_output = test.get("expected", test.get("output", {}))
            tolerance = test.get("tolerance", self.default_tolerance)

            result = {
                "name": test_name,
                "passed": False,
                "input": test_input,
                "expected": expected_output,
                "actual": None,
                "error": None,
            }

            try:
                # Convert input to appropriate format
                if hasattr(agent, "INPUT_MODEL"):
                    input_model = getattr(agent, "INPUT_MODEL")
                    test_input_obj = input_model(**test_input)
                else:
                    test_input_obj = test_input

                # Run agent
                actual_output = agent.run(test_input_obj)

                # Convert to dict for comparison
                if hasattr(actual_output, "dict"):
                    actual_dict = actual_output.dict()
                elif hasattr(actual_output, "model_dump"):
                    actual_dict = actual_output.model_dump()
                elif hasattr(actual_output, "__dict__"):
                    actual_dict = actual_output.__dict__
                else:
                    actual_dict = {"value": actual_output}

                result["actual"] = actual_dict

                # Compare outputs
                comparison = self._compare_outputs(expected_output, actual_dict, tolerance)
                result["passed"] = comparison["match"]
                result["comparison"] = comparison

            except Exception as e:
                result["error"] = str(e)
                logger.warning(f"Golden test '{test_name}' failed: {str(e)}")

            results.append(result)

        return results

    def _compare_outputs(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        tolerance: float,
    ) -> Dict[str, Any]:
        """
        Compare expected and actual outputs.

        Args:
            expected: Expected output dictionary
            actual: Actual output dictionary
            tolerance: Numeric tolerance

        Returns:
            Comparison result dictionary
        """
        comparison = {
            "match": True,
            "mismatches": [],
            "tolerance_used": tolerance,
        }

        for key, expected_value in expected.items():
            if key not in actual:
                comparison["match"] = False
                comparison["mismatches"].append({
                    "field": key,
                    "expected": expected_value,
                    "actual": "MISSING",
                })
                continue

            actual_value = actual[key]

            # Numeric comparison with tolerance
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                if not math.isclose(expected_value, actual_value, rel_tol=tolerance, abs_tol=tolerance):
                    comparison["match"] = False
                    comparison["mismatches"].append({
                        "field": key,
                        "expected": expected_value,
                        "actual": actual_value,
                        "difference": abs(expected_value - actual_value),
                    })
            # String comparison
            elif isinstance(expected_value, str) and isinstance(actual_value, str):
                if expected_value.lower() != actual_value.lower():
                    comparison["match"] = False
                    comparison["mismatches"].append({
                        "field": key,
                        "expected": expected_value,
                        "actual": actual_value,
                    })
            # Direct comparison for other types
            elif expected_value != actual_value:
                comparison["match"] = False
                comparison["mismatches"].append({
                    "field": key,
                    "expected": expected_value,
                    "actual": actual_value,
                })

        return comparison

    def _check_tolerance_handling(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if all results are within specified tolerances.

        Args:
            test_results: List of test results

        Returns:
            Tolerance check results
        """
        result = {
            "all_within_tolerance": True,
            "violations": 0,
            "details": [],
        }

        for test in test_results:
            if "comparison" in test and "mismatches" in test["comparison"]:
                for mismatch in test["comparison"]["mismatches"]:
                    if "difference" in mismatch:
                        result["violations"] += 1
                        result["all_within_tolerance"] = False
                        result["details"].append({
                            "test": test["name"],
                            "field": mismatch["field"],
                            "difference": mismatch["difference"],
                        })

        return result

    def _check_edge_cases(
        self,
        agent: Any,
        agent_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Check edge case handling.

        Args:
            agent: Agent instance
            agent_path: Path to agent directory

        Returns:
            List of edge case test results
        """
        results = []

        # Define standard edge cases
        edge_cases = [
            {
                "name": "zero_input",
                "description": "Zero quantity input",
                "input_modifier": lambda x: {**x, "quantity": 0},
                "expect_error": False,
            },
            {
                "name": "negative_input",
                "description": "Negative quantity input",
                "input_modifier": lambda x: {**x, "quantity": -1},
                "expect_error": True,
            },
            {
                "name": "very_large_input",
                "description": "Very large quantity input",
                "input_modifier": lambda x: {**x, "quantity": 1e12},
                "expect_error": False,
            },
            {
                "name": "very_small_input",
                "description": "Very small quantity input",
                "input_modifier": lambda x: {**x, "quantity": 1e-12},
                "expect_error": False,
            },
        ]

        # Get sample input
        sample_input = self._get_sample_input(agent_path)
        if not sample_input:
            return results

        for edge_case in edge_cases:
            result = {
                "name": edge_case["name"],
                "description": edge_case["description"],
                "passed": False,
            }

            try:
                modified_input = edge_case["input_modifier"](sample_input)

                try:
                    output = agent.run(modified_input)
                    result["passed"] = not edge_case["expect_error"]
                except (ValueError, AssertionError) as e:
                    result["passed"] = edge_case["expect_error"]
                    result["error"] = str(e)
                except Exception as e:
                    result["passed"] = False
                    result["error"] = str(e)

            except Exception as e:
                result["error"] = f"Failed to prepare edge case: {str(e)}"

            results.append(result)

        return results

    def _get_sample_input(self, agent_path: Path) -> Optional[Dict[str, Any]]:
        """Get sample input from golden tests."""
        golden_tests = self._load_golden_tests(agent_path)
        if golden_tests:
            return golden_tests[0].get("input", {})
        return None

    def _check_boundary_conditions(self, agent: Any) -> Dict[str, Any]:
        """
        Check boundary condition handling.

        Args:
            agent: Agent instance

        Returns:
            Boundary check results
        """
        result = {
            "all_passed": True,
            "failures": 0,
            "checks": [],
        }

        # Check if agent has boundary validation
        if hasattr(agent, "INPUT_MODEL"):
            # Pydantic model should have validators
            result["checks"].append({
                "name": "has_input_model",
                "passed": True,
            })
        else:
            result["checks"].append({
                "name": "has_input_model",
                "passed": False,
            })
            result["failures"] += 1
            result["all_passed"] = False

        return result

    def _check_numeric_precision(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check numeric precision consistency.

        Args:
            test_results: List of test results

        Returns:
            Precision check results
        """
        result = {
            "consistent": True,
            "precision_issues": [],
        }

        for test in test_results:
            if test["actual"]:
                for key, value in test["actual"].items():
                    if isinstance(value, float):
                        # Check for excessive precision
                        str_value = str(value)
                        if len(str_value.split(".")[-1] if "." in str_value else "") > 15:
                            result["precision_issues"].append({
                                "test": test["name"],
                                "field": key,
                                "value": value,
                            })

        if result["precision_issues"]:
            result["consistent"] = False

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_load": (
                "Ensure agent.py exists and contains a class ending with 'Agent' "
                "that has a run() method."
            ),
            "golden_tests_exist": (
                "Create golden tests in one of these locations:\n"
                "  - pack.yaml under tests.golden section\n"
                "  - golden_tests.yaml in agent directory\n"
                "  - tests/golden_*.yaml files"
            ),
            "golden_test_pass_rate": (
                "Fix failing golden tests:\n"
                "  1. Review expected vs actual output\n"
                "  2. Check calculation formulas\n"
                "  3. Verify emission factors are correct\n"
                "  4. Ensure unit conversions are accurate"
            ),
            "tolerance_handling": (
                "Adjust numeric precision:\n"
                "  - Use round(value, 6) for consistent output\n"
                "  - Consider using Decimal for financial calculations\n"
                "  - Document expected tolerance in tests"
            ),
            "edge_case_handling": (
                "Add edge case handling:\n"
                "  - Handle zero inputs gracefully\n"
                "  - Validate negative inputs\n"
                "  - Set reasonable bounds for large values"
            ),
            "boundary_conditions": (
                "Add input validation:\n"
                "  - Use Pydantic Field(ge=0) for non-negative\n"
                "  - Add @validator for custom bounds\n"
                "  - Document valid input ranges"
            ),
            "numeric_precision": (
                "Standardize numeric precision:\n"
                "  - Use round(value, 6) consistently\n"
                "  - Avoid excessive decimal places\n"
                "  - Use appropriate data types"
            ),
        }

        return remediation_map.get(check.name)
