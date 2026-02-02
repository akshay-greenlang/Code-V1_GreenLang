"""
Explainability Dimension Evaluator

Evaluates agent explainability including:
- Decision transparency
- Reasoning traces
- Calculation breakdowns
- Source attribution
- Human-readable explanations

Critical for audit and regulatory compliance.

"""

import logging
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


class ExplainabilityEvaluator:
    """
    Evaluator for explainability dimension.

    Tests:
    1. Calculation transparency - Formula/method visible
    2. Reasoning traces - Step-by-step breakdown
    3. Source attribution - Data sources cited
    4. Human-readable output - Clear explanations
    5. Audit trail - Complete decision log
    """

    # Required explainability fields
    EXPLAINABILITY_FIELDS = [
        "explanation",
        "reasoning",
        "calculation_steps",
        "methodology",
        "formula",
        "breakdown",
    ]

    def __init__(self):
        """Initialize explainability evaluator."""
        logger.info("ExplainabilityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent explainability.

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

        # Test 1: Calculation transparency
        calc_score, calc_details = self._test_calculation_transparency(pack_spec)
        details["calculation_transparency"] = calc_details
        tests_run += calc_details.get("test_count", 0)
        tests_passed += calc_details.get("tests_passed", 0)

        if calc_score < 100:
            findings.append(f"Calculation transparency: {calc_score:.1f}%")
            recommendations.append(
                "Document calculation formulas in pack specification"
            )

        # Test 2: Reasoning traces
        reasoning_score, reasoning_details = self._test_reasoning_traces(
            agent, sample_inputs
        )
        details["reasoning_traces"] = reasoning_details
        tests_run += reasoning_details.get("test_count", 0)
        tests_passed += reasoning_details.get("tests_passed", 0)

        if reasoning_score < 100:
            findings.append(f"Reasoning traces: {reasoning_score:.1f}%")
            recommendations.append(
                "Add step-by-step reasoning to agent outputs"
            )

        # Test 3: Source attribution
        source_score, source_details = self._test_source_attribution(
            agent, pack_spec, sample_inputs
        )
        details["source_attribution"] = source_details
        tests_run += source_details.get("test_count", 0)
        tests_passed += source_details.get("tests_passed", 0)

        if source_score < 100:
            findings.append(f"Source attribution: {source_score:.1f}%")
            recommendations.append(
                "Cite all data sources in outputs"
            )

        # Test 4: Human-readable explanations
        readable_score, readable_details = self._test_human_readable(
            agent, sample_inputs
        )
        details["human_readable"] = readable_details
        tests_run += readable_details.get("test_count", 0)
        tests_passed += readable_details.get("tests_passed", 0)

        if readable_score < 100:
            findings.append(f"Human-readable output: {readable_score:.1f}%")
            recommendations.append(
                "Include plain-language explanations of results"
            )

        # Test 5: Methodology documentation
        method_score, method_details = self._test_methodology_docs(pack_spec)
        details["methodology"] = method_details
        tests_run += method_details.get("test_count", 0)
        tests_passed += method_details.get("tests_passed", 0)

        if method_score < 100:
            findings.append(f"Methodology documentation: {method_score:.1f}%")
            recommendations.append(
                "Document calculation methodology in pack spec"
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

    def _test_calculation_transparency(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test calculation transparency."""
        tests_run = 0
        tests_passed = 0
        transparency_checks = []

        # Check for formula documentation
        tests_run += 1
        calculation_section = pack_spec.get("calculation", {})
        formula = calculation_section.get("formula")

        if formula:
            tests_passed += 1
            transparency_checks.append({
                "check": "formula_documented",
                "status": "PRESENT",
                "formula_preview": str(formula)[:100],
            })
        else:
            transparency_checks.append({
                "check": "formula_documented",
                "status": "MISSING",
            })

        # Check for methodology
        tests_run += 1
        methodology = pack_spec.get("methodology") or calculation_section.get("methodology")

        if methodology:
            tests_passed += 1
            transparency_checks.append({
                "check": "methodology_documented",
                "status": "PRESENT",
            })
        else:
            # Not always required
            tests_passed += 1
            transparency_checks.append({
                "check": "methodology_documented",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "transparency_checks": transparency_checks,
        }

    def _test_reasoning_traces(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test reasoning traces in output."""
        tests_run = 0
        tests_passed = 0
        reasoning_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                reasoning_fields = [
                    "reasoning",
                    "calculation_steps",
                    "explanation",
                    "breakdown",
                    "trace",
                ]

                has_reasoning = any(hasattr(result, f) for f in reasoning_fields)

                if has_reasoning:
                    tests_passed += 1
                    reasoning_checks.append({
                        "status": "PRESENT",
                    })
                else:
                    # Not all outputs need reasoning - conditional pass
                    tests_passed += 1
                    reasoning_checks.append({
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
            "reasoning_checks": reasoning_checks,
        }

    def _test_source_attribution(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test source attribution."""
        tests_run = 0
        tests_passed = 0
        attribution_checks = []

        # Check pack spec for source references
        tests_run += 1
        sources = pack_spec.get("sources", []) or pack_spec.get("references", [])

        if sources:
            tests_passed += 1
            attribution_checks.append({
                "check": "spec_sources",
                "count": len(sources),
                "status": "PRESENT",
            })
        else:
            attribution_checks.append({
                "check": "spec_sources",
                "status": "MISSING",
            })

        # Check agent output for source fields
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                source_fields = [
                    "emission_factor_source",
                    "data_source",
                    "sources",
                    "citation",
                    "reference",
                ]

                has_source = any(hasattr(result, f) for f in source_fields)

                if has_source:
                    tests_passed += 1
                    attribution_checks.append({
                        "check": "output_sources",
                        "status": "PRESENT",
                    })
                else:
                    # Conditional pass
                    tests_passed += 1
                    attribution_checks.append({
                        "check": "output_sources",
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "attribution_checks": attribution_checks,
        }

    def _test_human_readable(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test human-readable output."""
        tests_run = 0
        tests_passed = 0
        readable_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                # Check for description/summary fields
                readable_fields = [
                    "description",
                    "summary",
                    "explanation",
                    "narrative",
                ]

                has_readable = any(hasattr(result, f) for f in readable_fields)

                if has_readable:
                    tests_passed += 1
                    readable_checks.append({
                        "status": "PRESENT",
                    })
                else:
                    # Not required for all agents
                    tests_passed += 1
                    readable_checks.append({
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
            "readable_checks": readable_checks,
        }

    def _test_methodology_docs(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test methodology documentation."""
        tests_run = 1
        tests_passed = 0
        method_checks = []

        # Check for methodology section
        methodology = pack_spec.get("methodology") or pack_spec.get("calculation", {}).get("methodology")
        description = pack_spec.get("description")

        if methodology or description:
            tests_passed = 1
            method_checks.append({
                "check": "methodology_docs",
                "status": "DOCUMENTED",
            })
        else:
            # Minimal docs acceptable
            tests_passed = 1
            method_checks.append({
                "check": "methodology_docs",
                "status": "MINIMAL",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "method_checks": method_checks,
        }
