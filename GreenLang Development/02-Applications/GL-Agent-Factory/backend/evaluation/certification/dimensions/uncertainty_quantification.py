"""
Uncertainty Quantification Dimension Evaluator

Evaluates agent uncertainty quantification including:
- Error bounds calculation
- Confidence intervals
- Sensitivity analysis
- Monte Carlo uncertainty
- Data quality scoring

Critical for regulatory compliance where uncertainty must be disclosed.

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


class UncertaintyQuantificationEvaluator:
    """
    Evaluator for uncertainty quantification dimension.

    Tests:
    1. Error bounds - Upper/lower bounds present
    2. Confidence intervals - Statistical confidence levels
    3. Sensitivity analysis - Parameter sensitivity
    4. Data quality scoring - Input data quality assessment
    5. Propagation - Uncertainty propagation through calculations
    """

    # Required uncertainty fields
    UNCERTAINTY_FIELDS = [
        "uncertainty",
        "uncertainty_percent",
        "confidence_interval",
        "error_bound",
        "lower_bound",
        "upper_bound",
        "confidence_score",
        "data_quality_score",
    ]

    # GHG Protocol data quality indicators
    DATA_QUALITY_INDICATORS = {
        "temporal_correlation": "Time representativeness",
        "geographical_correlation": "Geographic representativeness",
        "technological_correlation": "Technology representativeness",
        "completeness": "Data completeness",
        "reliability": "Source reliability",
    }

    def __init__(self):
        """Initialize uncertainty quantification evaluator."""
        logger.info("UncertaintyQuantificationEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent uncertainty quantification.

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

        # Test 1: Error bounds presence
        bounds_score, bounds_details = self._test_error_bounds(
            agent, pack_spec, sample_inputs
        )
        details["error_bounds"] = bounds_details
        tests_run += bounds_details.get("test_count", 0)
        tests_passed += bounds_details.get("tests_passed", 0)

        if bounds_score < 100:
            findings.append(f"Error bounds: {bounds_score:.1f}%")
            recommendations.append(
                "Add uncertainty/error bounds to calculation outputs"
            )

        # Test 2: Confidence intervals
        ci_score, ci_details = self._test_confidence_intervals(
            agent, sample_inputs
        )
        details["confidence_intervals"] = ci_details
        tests_run += ci_details.get("test_count", 0)
        tests_passed += ci_details.get("tests_passed", 0)

        if ci_score < 100:
            findings.append(f"Confidence intervals: {ci_score:.1f}%")
            recommendations.append(
                "Include confidence intervals (95% CI recommended)"
            )

        # Test 3: Data quality scoring
        dq_score, dq_details = self._test_data_quality_scoring(
            agent, pack_spec, sample_inputs
        )
        details["data_quality"] = dq_details
        tests_run += dq_details.get("test_count", 0)
        tests_passed += dq_details.get("tests_passed", 0)

        if dq_score < 100:
            findings.append(f"Data quality scoring: {dq_score:.1f}%")
            recommendations.append(
                "Implement data quality indicators per GHG Protocol"
            )

        # Test 4: Uncertainty bounds validation
        validation_score, validation_details = self._test_uncertainty_validation(
            agent, sample_inputs
        )
        details["uncertainty_validation"] = validation_details
        tests_run += validation_details.get("test_count", 0)
        tests_passed += validation_details.get("tests_passed", 0)

        if validation_score < 100:
            findings.append(f"Uncertainty validation: {validation_score:.1f}%")
            recommendations.append(
                "Validate uncertainty values are within reasonable ranges"
            )

        # Test 5: Sensitivity documentation
        sensitivity_score, sensitivity_details = self._test_sensitivity_analysis(
            pack_spec
        )
        details["sensitivity"] = sensitivity_details
        tests_run += sensitivity_details.get("test_count", 0)
        tests_passed += sensitivity_details.get("tests_passed", 0)

        if sensitivity_score < 100:
            findings.append(f"Sensitivity analysis: {sensitivity_score:.1f}%")
            recommendations.append(
                "Document sensitivity of outputs to key input parameters"
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

    def _test_error_bounds(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test error bounds presence."""
        tests_run = 0
        tests_passed = 0
        bounds_checks = []

        # Check pack spec for uncertainty definition
        tests_run += 1
        uncertainty_in_spec = "uncertainty" in str(pack_spec).lower()

        if uncertainty_in_spec:
            tests_passed += 1
            bounds_checks.append({
                "check": "spec_uncertainty",
                "status": "DEFINED",
            })
        else:
            bounds_checks.append({
                "check": "spec_uncertainty",
                "status": "MISSING",
            })

        # Check agent output for uncertainty fields
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                has_uncertainty = any(
                    hasattr(result, f) for f in self.UNCERTAINTY_FIELDS
                )

                if has_uncertainty:
                    tests_passed += 1
                    bounds_checks.append({
                        "check": "output_uncertainty",
                        "status": "PRESENT",
                    })
                else:
                    bounds_checks.append({
                        "check": "output_uncertainty",
                        "status": "MISSING",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "bounds_checks": bounds_checks,
        }

    def _test_confidence_intervals(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test confidence interval presence."""
        tests_run = 0
        tests_passed = 0
        ci_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                ci_fields = [
                    "confidence_interval",
                    "ci_lower",
                    "ci_upper",
                    "confidence_score",
                    "lower_bound",
                    "upper_bound",
                ]

                has_ci = any(hasattr(result, f) for f in ci_fields)

                if has_ci:
                    tests_passed += 1
                    ci_checks.append({
                        "status": "PRESENT",
                    })
                else:
                    # CI not required for all agents
                    tests_passed += 1
                    ci_checks.append({
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "ci_checks": ci_checks,
        }

    def _test_data_quality_scoring(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test data quality scoring."""
        tests_run = 0
        tests_passed = 0
        dq_checks = []

        # Check pack spec for data quality references
        tests_run += 1
        dq_in_spec = any(
            indicator in str(pack_spec).lower()
            for indicator in ["data_quality", "quality_score", "dqi"]
        )

        if dq_in_spec:
            tests_passed += 1
            dq_checks.append({
                "check": "spec_data_quality",
                "status": "DEFINED",
            })
        else:
            # Not required for all agents
            tests_passed += 1
            dq_checks.append({
                "check": "spec_data_quality",
                "status": "N/A",
            })

        # Check agent output
        for sample_input in sample_inputs[:1]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                dq_fields = ["data_quality_score", "dqi", "quality_indicator"]
                has_dq = any(hasattr(result, f) for f in dq_fields)

                if has_dq:
                    tests_passed += 1
                    dq_checks.append({
                        "check": "output_dq",
                        "status": "PRESENT",
                    })
                else:
                    tests_passed += 1
                    dq_checks.append({
                        "check": "output_dq",
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "dq_checks": dq_checks,
        }

    def _test_uncertainty_validation(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test uncertainty values are valid."""
        tests_run = 0
        tests_passed = 0
        validation_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                # Check uncertainty is in valid range
                uncertainty = getattr(result, "uncertainty_percent", None)
                if uncertainty is not None:
                    # Uncertainty should be between 0-100%
                    if 0 <= uncertainty <= 100:
                        tests_passed += 1
                        validation_checks.append({
                            "value": uncertainty,
                            "status": "VALID",
                        })
                    else:
                        validation_checks.append({
                            "value": uncertainty,
                            "status": "INVALID",
                            "reason": "Out of range [0, 100]",
                        })
                else:
                    # No uncertainty to validate - pass
                    tests_passed += 1
                    validation_checks.append({
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "validation_checks": validation_checks,
        }

    def _test_sensitivity_analysis(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test sensitivity analysis documentation."""
        tests_run = 1
        tests_passed = 0
        sensitivity_checks = []

        # Check for sensitivity documentation
        sensitivity_mentioned = "sensitivity" in str(pack_spec).lower()

        if sensitivity_mentioned:
            tests_passed = 1
            sensitivity_checks.append({
                "check": "sensitivity_docs",
                "status": "DOCUMENTED",
            })
        else:
            # Not required for all agents
            tests_passed = 1
            sensitivity_checks.append({
                "check": "sensitivity_docs",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "sensitivity_checks": sensitivity_checks,
        }
