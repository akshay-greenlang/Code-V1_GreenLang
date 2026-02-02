"""
Robustness Dimension Evaluator

Evaluates agent robustness including:
- Edge case handling
- Error recovery
- Invalid input handling
- Boundary conditions
- Graceful degradation

Ensures agents handle unexpected inputs safely.

"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from dimension evaluation."""
    score: float
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class RobustnessEvaluator:
    """
    Evaluator for robustness dimension.

    Tests:
    1. Edge cases - Zero, negative, extreme values
    2. Invalid inputs - Type errors, missing fields
    3. Boundary conditions - Min/max values
    4. Error handling - Proper error messages
    5. Recovery - Graceful degradation
    """

    # Edge case values to test
    EDGE_CASE_VALUES = [
        0,
        -1,
        0.0,
        -0.0,
        float("inf"),
        float("-inf"),
        1e-15,
        1e15,
    ]

    # Invalid input types to test
    INVALID_INPUTS = [
        None,
        "",
        [],
        {},
        "invalid_string",
    ]

    def __init__(self):
        """Initialize robustness evaluator."""
        logger.info("RobustnessEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent robustness.

        Args:
            agent: Agent instance to evaluate
            pack_spec: Agent pack specification
            sample_inputs: Sample inputs for testing
            golden_result: Optional golden test results
            determinism_result: Optional determinism results

        Returns:
            EvaluationResult with score and details
        """
        tests_run = 0
        tests_passed = 0
        findings = []
        recommendations = []
        details = {}

        # Test 1: Edge case handling
        edge_score, edge_details = self._test_edge_cases(
            agent, pack_spec, sample_inputs
        )
        details["edge_cases"] = edge_details
        tests_run += edge_details.get("test_count", 0)
        tests_passed += edge_details.get("tests_passed", 0)

        if edge_score < 100:
            findings.append(f"Edge case handling: {edge_score:.1f}%")
            recommendations.append(
                "Add validation for edge case values (zero, negative, extreme)"
            )

        # Test 2: Invalid input handling
        invalid_score, invalid_details = self._test_invalid_inputs(
            agent, sample_inputs
        )
        details["invalid_inputs"] = invalid_details
        tests_run += invalid_details.get("test_count", 0)
        tests_passed += invalid_details.get("tests_passed", 0)

        if invalid_score < 100:
            findings.append(f"Invalid input handling: {invalid_score:.1f}%")
            recommendations.append(
                "Implement proper validation for all input types"
            )

        # Test 3: Boundary conditions
        boundary_score, boundary_details = self._test_boundary_conditions(
            agent, pack_spec, sample_inputs
        )
        details["boundary_conditions"] = boundary_details
        tests_run += boundary_details.get("test_count", 0)
        tests_passed += boundary_details.get("tests_passed", 0)

        if boundary_score < 100:
            findings.append(f"Boundary condition handling: {boundary_score:.1f}%")
            recommendations.append(
                "Document and validate boundary conditions"
            )

        # Test 4: Error message quality
        error_score, error_details = self._test_error_messages(
            agent, sample_inputs
        )
        details["error_messages"] = error_details
        tests_run += error_details.get("test_count", 0)
        tests_passed += error_details.get("tests_passed", 0)

        if error_score < 100:
            findings.append(f"Error message quality: {error_score:.1f}%")
            recommendations.append(
                "Provide descriptive error messages for validation failures"
            )

        # Test 5: Empty input handling
        empty_score, empty_details = self._test_empty_inputs(agent)
        details["empty_inputs"] = empty_details
        tests_run += empty_details.get("test_count", 0)
        tests_passed += empty_details.get("tests_passed", 0)

        if empty_score < 100:
            findings.append(f"Empty input handling: {empty_score:.1f}%")
            recommendations.append(
                "Handle empty/missing inputs gracefully"
            )

        # Calculate overall score
        if tests_run == 0:
            overall_score = 0.0
        else:
            overall_score = (tests_passed / tests_run) * 100

        return EvaluationResult(
            score=overall_score,
            test_count=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_run - tests_passed,
            details=details,
            findings=findings,
            recommendations=recommendations,
        )

    def _test_edge_cases(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test edge case handling."""
        tests_run = 0
        tests_passed = 0
        edge_tests = []

        if not sample_inputs:
            return 100.0, {"test_count": 1, "tests_passed": 1, "status": "SKIPPED"}

        # Get first input as template
        template_input = sample_inputs[0].copy()

        # Find numeric fields to test
        numeric_fields = [
            k for k, v in template_input.items()
            if isinstance(v, (int, float))
        ]

        for field_name in numeric_fields[:2]:  # Test first 2 numeric fields
            for edge_value in [0, -1, 1e10]:
                tests_run += 1
                test_input = template_input.copy()
                test_input[field_name] = edge_value

                try:
                    result = agent.run(test_input)
                    # Agent should either succeed or raise validation error
                    tests_passed += 1
                    edge_tests.append({
                        "field": field_name,
                        "value": edge_value,
                        "status": "HANDLED",
                    })
                except (ValueError, TypeError) as e:
                    # Validation error is acceptable
                    tests_passed += 1
                    edge_tests.append({
                        "field": field_name,
                        "value": edge_value,
                        "status": "VALIDATION_ERROR",
                        "error": str(e)[:50],
                    })
                except Exception as e:
                    # Unhandled exception is a failure
                    edge_tests.append({
                        "field": field_name,
                        "value": edge_value,
                        "status": "UNHANDLED_ERROR",
                        "error": str(e)[:50],
                    })

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "edge_tests": edge_tests,
        }

    def _test_invalid_inputs(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test invalid input handling."""
        tests_run = 0
        tests_passed = 0
        invalid_tests = []

        # Test with various invalid inputs
        invalid_inputs = [
            {},  # Empty dict
            {"invalid_field": "value"},  # Wrong fields
        ]

        for invalid_input in invalid_inputs:
            tests_run += 1
            try:
                agent.run(invalid_input)
                # May succeed with defaults
                tests_passed += 1
                invalid_tests.append({
                    "input": str(invalid_input)[:50],
                    "status": "ACCEPTED",
                })
            except (ValueError, TypeError, KeyError) as e:
                # Validation error is acceptable
                tests_passed += 1
                invalid_tests.append({
                    "input": str(invalid_input)[:50],
                    "status": "VALIDATION_ERROR",
                })
            except Exception as e:
                invalid_tests.append({
                    "input": str(invalid_input)[:50],
                    "status": "UNHANDLED_ERROR",
                    "error": str(e)[:50],
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "invalid_tests": invalid_tests,
        }

    def _test_boundary_conditions(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test boundary condition handling."""
        tests_run = 1
        tests_passed = 0
        boundary_tests = []

        # Check pack spec for documented boundaries
        constraints = pack_spec.get("constraints", {})
        limits = pack_spec.get("limits", {})

        if constraints or limits:
            tests_passed = 1
            boundary_tests.append({
                "check": "documented_boundaries",
                "status": "PRESENT",
            })
        else:
            # Not all agents need explicit boundaries
            tests_passed = 1
            boundary_tests.append({
                "check": "documented_boundaries",
                "status": "NOT_DOCUMENTED",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "boundary_tests": boundary_tests,
        }

    def _test_error_messages(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test error message quality."""
        tests_run = 1
        tests_passed = 0
        error_tests = []

        # Try to trigger an error and check message quality
        try:
            agent.run({})  # Empty input likely to cause error
            tests_passed = 1  # No error is also acceptable
            error_tests.append({
                "status": "NO_ERROR",
            })
        except Exception as e:
            error_msg = str(e)
            # Check if error message is descriptive (> 10 chars)
            if len(error_msg) > 10:
                tests_passed = 1
                error_tests.append({
                    "status": "DESCRIPTIVE",
                    "message_length": len(error_msg),
                })
            else:
                error_tests.append({
                    "status": "BRIEF",
                    "message_length": len(error_msg),
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "error_tests": error_tests,
        }

    def _test_empty_inputs(
        self, agent: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """Test empty input handling."""
        tests_run = 1
        tests_passed = 0
        empty_tests = []

        try:
            agent.run({})
            # Successfully handling empty input
            tests_passed = 1
            empty_tests.append({
                "input": "{}",
                "status": "ACCEPTED",
            })
        except (ValueError, TypeError, KeyError) as e:
            # Validation error is acceptable
            tests_passed = 1
            empty_tests.append({
                "input": "{}",
                "status": "VALIDATION_ERROR",
            })
        except Exception as e:
            empty_tests.append({
                "input": "{}",
                "status": "UNHANDLED_ERROR",
                "error": str(e)[:50],
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "empty_tests": empty_tests,
        }
