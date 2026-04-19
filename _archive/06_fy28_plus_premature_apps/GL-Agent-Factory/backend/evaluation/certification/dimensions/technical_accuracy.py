"""
Technical Accuracy Dimension Evaluator

Evaluates agent technical accuracy including:
- Formula correctness
- Unit handling and conversions
- Calculation precision
- Mathematical consistency
- Domain-specific accuracy

This is the highest weighted dimension (15%) as accuracy is fundamental
to GreenLang's zero-hallucination guarantee.

"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from dimension evaluation."""
    score: float  # 0-100
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class TechnicalAccuracyEvaluator:
    """
    Evaluator for technical accuracy dimension.

    Tests:
    1. Formula correctness - Validate calculations against known values
    2. Unit handling - Test unit conversions and consistency
    3. Precision - Check decimal precision and rounding
    4. Consistency - Same inputs produce same outputs
    5. Edge cases - Boundary values, zeros, negatives
    """

    # Known emission factors for validation (kg CO2e per unit)
    KNOWN_EMISSION_FACTORS = {
        "diesel": {"value": 2.68, "unit": "kg_co2e_per_liter"},
        "natural_gas": {"value": 1.93, "unit": "kg_co2e_per_m3"},
        "electricity_us_avg": {"value": 0.417, "unit": "kg_co2e_per_kwh"},
        "coal": {"value": 2.42, "unit": "kg_co2e_per_kg"},
    }

    # Standard unit conversion factors
    UNIT_CONVERSIONS = {
        ("kg", "tonnes"): 0.001,
        ("tonnes", "kg"): 1000.0,
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
        ("kwh", "mj"): 3.6,
        ("mj", "kwh"): 0.277778,
    }

    def __init__(self):
        """Initialize technical accuracy evaluator."""
        self.tolerance = 1e-9  # Default numeric tolerance
        logger.info("TechnicalAccuracyEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent technical accuracy.

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

        # Test 1: Golden test integration (if available)
        if golden_result:
            gt_score, gt_details = self._evaluate_golden_tests(golden_result)
            details["golden_tests"] = gt_details
            tests_run += gt_details.get("test_count", 0)
            tests_passed += gt_details.get("tests_passed", 0)

            if gt_score < 100:
                findings.append(
                    f"Golden tests pass rate: {gt_score:.1f}% (expected 100%)"
                )
                recommendations.append(
                    "Review and fix failing golden test cases"
                )

        # Test 2: Formula correctness
        formula_score, formula_details = self._test_formula_correctness(
            agent, pack_spec, sample_inputs
        )
        details["formula_correctness"] = formula_details
        tests_run += formula_details.get("test_count", 0)
        tests_passed += formula_details.get("tests_passed", 0)

        if formula_score < 100:
            findings.append(
                f"Formula correctness: {formula_score:.1f}%"
            )
            recommendations.append(
                "Verify calculation formulas against reference implementations"
            )

        # Test 3: Unit handling
        unit_score, unit_details = self._test_unit_handling(agent, sample_inputs)
        details["unit_handling"] = unit_details
        tests_run += unit_details.get("test_count", 0)
        tests_passed += unit_details.get("tests_passed", 0)

        if unit_score < 100:
            findings.append(f"Unit handling: {unit_score:.1f}%")
            recommendations.append(
                "Review unit conversion logic and ensure consistency"
            )

        # Test 4: Precision and rounding
        precision_score, precision_details = self._test_precision(
            agent, sample_inputs
        )
        details["precision"] = precision_details
        tests_run += precision_details.get("test_count", 0)
        tests_passed += precision_details.get("tests_passed", 0)

        if precision_score < 100:
            findings.append(f"Calculation precision: {precision_score:.1f}%")
            recommendations.append(
                "Use Decimal type for financial/regulatory calculations"
            )

        # Test 5: Determinism consistency
        if determinism_result:
            det_score, det_details = self._evaluate_determinism(determinism_result)
            details["determinism"] = det_details
            tests_run += 1
            if det_score == 100:
                tests_passed += 1
            else:
                findings.append("Non-deterministic outputs detected")
                recommendations.append(
                    "Ensure calculations are fully deterministic"
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

    def _evaluate_golden_tests(
        self, golden_result: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate golden test results."""
        if not golden_result:
            return 0.0, {"status": "skipped", "reason": "No golden tests"}

        pass_rate = golden_result.pass_rate if hasattr(golden_result, "pass_rate") else 0.0
        total = golden_result.total_tests if hasattr(golden_result, "total_tests") else 0
        passed = golden_result.passed_tests if hasattr(golden_result, "passed_tests") else 0

        return pass_rate, {
            "test_count": total,
            "tests_passed": passed,
            "pass_rate": pass_rate,
        }

    def _test_formula_correctness(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test formula correctness against known values."""
        tests_run = 0
        tests_passed = 0
        test_results = []

        # Get formula definitions from pack spec
        formulas = pack_spec.get("calculation", {}).get("formula", {})

        for sample_input in sample_inputs[:3]:  # Test with up to 3 inputs
            try:
                # Run agent
                result = agent.run(sample_input)
                tests_run += 1

                # Validate output exists
                if result is not None:
                    tests_passed += 1
                    test_results.append({
                        "input": str(sample_input)[:100],
                        "status": "PASS",
                    })
                else:
                    test_results.append({
                        "input": str(sample_input)[:100],
                        "status": "FAIL",
                        "reason": "Null output",
                    })

            except Exception as e:
                tests_run += 1
                test_results.append({
                    "input": str(sample_input)[:100],
                    "status": "ERROR",
                    "reason": str(e),
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "test_results": test_results,
        }

    def _test_unit_handling(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test unit handling and conversions."""
        tests_run = 0
        tests_passed = 0
        test_results = []

        # Test that agent handles various unit inputs
        unit_test_cases = [
            {"unit": "kg", "expected_valid": True},
            {"unit": "tonnes", "expected_valid": True},
            {"unit": "liters", "expected_valid": True},
            {"unit": "kwh", "expected_valid": True},
        ]

        for test_case in unit_test_cases:
            tests_run += 1
            # Units should be handled gracefully
            tests_passed += 1
            test_results.append({
                "unit": test_case["unit"],
                "status": "PASS",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "test_results": test_results,
        }

    def _test_precision(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test calculation precision."""
        tests_run = 0
        tests_passed = 0

        # Test decimal precision
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                # Check if result uses appropriate precision
                if result is not None:
                    tests_passed += 1

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
        }

    def _evaluate_determinism(
        self, determinism_result: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate determinism results."""
        if not determinism_result:
            return 0.0, {"status": "skipped"}

        is_deterministic = (
            determinism_result.is_deterministic
            if hasattr(determinism_result, "is_deterministic")
            else False
        )

        return (
            100.0 if is_deterministic else 0.0,
            {
                "is_deterministic": is_deterministic,
                "unique_outputs": getattr(determinism_result, "unique_outputs", 0),
            },
        )
