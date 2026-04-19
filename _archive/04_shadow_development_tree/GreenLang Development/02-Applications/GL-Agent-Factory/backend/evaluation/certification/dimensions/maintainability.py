"""
Maintainability Dimension Evaluator

Evaluates agent maintainability including:
- Code quality indicators
- Documentation completeness
- Test coverage
- Dependency management
- Configuration management

Ensures agents are maintainable over their lifecycle.

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


class MaintainabilityEvaluator:
    """
    Evaluator for maintainability dimension.

    Tests:
    1. Documentation - Pack spec completeness
    2. Test coverage - Golden test presence
    3. Schema definition - Input/output schemas
    4. Versioning - Semantic versioning
    5. Dependencies - Dependency documentation
    """

    # Required pack spec sections
    REQUIRED_SECTIONS = [
        "pack",
        "description",
        "input",
        "output",
    ]

    # Recommended sections
    RECOMMENDED_SECTIONS = [
        "tests",
        "calculation",
        "sources",
        "dependencies",
    ]

    def __init__(self):
        """Initialize maintainability evaluator."""
        logger.info("MaintainabilityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent maintainability.

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

        # Test 1: Pack spec completeness
        spec_score, spec_details = self._test_pack_spec_completeness(pack_spec)
        details["pack_spec"] = spec_details
        tests_run += spec_details.get("test_count", 0)
        tests_passed += spec_details.get("tests_passed", 0)

        if spec_score < 100:
            findings.append(f"Pack spec completeness: {spec_score:.1f}%")
            recommendations.append(
                "Complete all required sections in pack.yaml"
            )

        # Test 2: Documentation quality
        doc_score, doc_details = self._test_documentation(pack_spec)
        details["documentation"] = doc_details
        tests_run += doc_details.get("test_count", 0)
        tests_passed += doc_details.get("tests_passed", 0)

        if doc_score < 100:
            findings.append(f"Documentation quality: {doc_score:.1f}%")
            recommendations.append(
                "Add comprehensive description and methodology documentation"
            )

        # Test 3: Test presence
        test_score, test_details = self._test_test_presence(
            pack_spec, golden_result
        )
        details["test_presence"] = test_details
        tests_run += test_details.get("test_count", 0)
        tests_passed += test_details.get("tests_passed", 0)

        if test_score < 100:
            findings.append(f"Test presence: {test_score:.1f}%")
            recommendations.append(
                "Add golden tests for comprehensive coverage"
            )

        # Test 4: Schema definition
        schema_score, schema_details = self._test_schema_definition(pack_spec)
        details["schema"] = schema_details
        tests_run += schema_details.get("test_count", 0)
        tests_passed += schema_details.get("tests_passed", 0)

        if schema_score < 100:
            findings.append(f"Schema definition: {schema_score:.1f}%")
            recommendations.append(
                "Define complete input/output schemas"
            )

        # Test 5: Versioning
        version_score, version_details = self._test_versioning(pack_spec)
        details["versioning"] = version_details
        tests_run += version_details.get("test_count", 0)
        tests_passed += version_details.get("tests_passed", 0)

        if version_score < 100:
            findings.append(f"Versioning: {version_score:.1f}%")
            recommendations.append(
                "Use semantic versioning (major.minor.patch)"
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

    def _test_pack_spec_completeness(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test pack spec completeness."""
        tests_run = 0
        tests_passed = 0
        section_checks = []

        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            tests_run += 1
            if section in pack_spec:
                tests_passed += 1
                section_checks.append({
                    "section": section,
                    "status": "PRESENT",
                })
            else:
                section_checks.append({
                    "section": section,
                    "status": "MISSING",
                })

        # Check recommended sections (half weight)
        for section in self.RECOMMENDED_SECTIONS:
            tests_run += 1
            if section in pack_spec:
                tests_passed += 1
                section_checks.append({
                    "section": section,
                    "status": "PRESENT",
                })
            else:
                # Recommended but not required - partial pass
                tests_passed += 0.5
                section_checks.append({
                    "section": section,
                    "status": "RECOMMENDED",
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": int(tests_passed),
            "section_checks": section_checks,
        }

    def _test_documentation(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test documentation quality."""
        tests_run = 0
        tests_passed = 0
        doc_checks = []

        # Check description
        tests_run += 1
        description = pack_spec.get("description", "")

        if description and len(str(description)) > 50:
            tests_passed += 1
            doc_checks.append({
                "check": "description",
                "length": len(str(description)),
                "status": "ADEQUATE",
            })
        elif description:
            tests_passed += 0.5
            doc_checks.append({
                "check": "description",
                "length": len(str(description)),
                "status": "BRIEF",
            })
        else:
            doc_checks.append({
                "check": "description",
                "status": "MISSING",
            })

        # Check methodology documentation
        tests_run += 1
        methodology = pack_spec.get("methodology") or pack_spec.get("calculation", {}).get("methodology")

        if methodology:
            tests_passed += 1
            doc_checks.append({
                "check": "methodology",
                "status": "PRESENT",
            })
        else:
            # Not strictly required
            tests_passed += 0.5
            doc_checks.append({
                "check": "methodology",
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": int(tests_passed),
            "doc_checks": doc_checks,
        }

    def _test_test_presence(
        self,
        pack_spec: Dict[str, Any],
        golden_result: Optional[Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test test presence."""
        tests_run = 1
        tests_passed = 0
        test_checks = []

        # Check for golden tests in spec
        golden_tests = pack_spec.get("tests", {}).get("golden", [])
        if not golden_tests:
            golden_tests = pack_spec.get("golden_tests", {}).get("test_cases", [])

        if golden_tests:
            tests_passed = 1
            test_checks.append({
                "check": "golden_tests",
                "count": len(golden_tests),
                "status": "PRESENT",
            })
        elif golden_result and hasattr(golden_result, "total_tests"):
            if golden_result.total_tests > 0:
                tests_passed = 1
                test_checks.append({
                    "check": "golden_tests",
                    "count": golden_result.total_tests,
                    "status": "PRESENT",
                })
        else:
            test_checks.append({
                "check": "golden_tests",
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "test_checks": test_checks,
        }

    def _test_schema_definition(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test schema definition."""
        tests_run = 0
        tests_passed = 0
        schema_checks = []

        # Check input schema
        tests_run += 1
        input_schema = pack_spec.get("input", {})

        if input_schema:
            tests_passed += 1
            schema_checks.append({
                "check": "input_schema",
                "field_count": len(input_schema),
                "status": "DEFINED",
            })
        else:
            schema_checks.append({
                "check": "input_schema",
                "status": "MISSING",
            })

        # Check output schema
        tests_run += 1
        output_schema = pack_spec.get("output", {})

        if output_schema:
            tests_passed += 1
            schema_checks.append({
                "check": "output_schema",
                "field_count": len(output_schema),
                "status": "DEFINED",
            })
        else:
            schema_checks.append({
                "check": "output_schema",
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "schema_checks": schema_checks,
        }

    def _test_versioning(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test versioning."""
        tests_run = 1
        tests_passed = 0
        version_checks = []

        # Check version format
        version = pack_spec.get("pack", {}).get("version", "")

        if version:
            # Check for semantic versioning pattern
            import re
            semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"

            if re.match(semver_pattern, str(version)):
                tests_passed = 1
                version_checks.append({
                    "version": version,
                    "status": "SEMVER",
                })
            else:
                tests_passed = 0.5
                version_checks.append({
                    "version": version,
                    "status": "NON_STANDARD",
                })
        else:
            version_checks.append({
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": int(tests_passed),
            "version_checks": version_checks,
        }
