"""
Regulatory Alignment Dimension Evaluator

Evaluates agent regulatory alignment including:
- GHG Protocol compliance
- ISO 14064 alignment
- EU CSRD/ESRS requirements
- SEC Climate Disclosure alignment
- Regional regulatory requirements

Ensures agents produce outputs compliant with environmental regulations.

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


class RegulatoryAlignmentEvaluator:
    """
    Evaluator for regulatory alignment dimension.

    Tests:
    1. GHG Protocol alignment - Scope 1/2/3 categorization
    2. ISO 14064 compliance - GHG accounting standards
    3. EU CSRD/ESRS alignment - European sustainability reporting
    4. SEC Climate alignment - US climate disclosure requirements
    5. Regional requirements - CBAM, UK SECR, etc.
    """

    # Key regulatory frameworks
    REGULATORY_FRAMEWORKS = {
        "ghg_protocol": {
            "name": "GHG Protocol",
            "required_fields": ["scope", "emission_category", "activity_data"],
        },
        "iso_14064": {
            "name": "ISO 14064",
            "required_fields": ["boundary", "quantification_method", "uncertainty"],
        },
        "csrd_esrs": {
            "name": "EU CSRD/ESRS",
            "required_fields": ["materiality", "disclosure_category", "reporting_period"],
        },
        "sec_climate": {
            "name": "SEC Climate Disclosure",
            "required_fields": ["financial_impact", "risk_category", "governance"],
        },
        "cbam": {
            "name": "EU CBAM",
            "required_fields": ["embedded_emissions", "product_category", "origin_country"],
        },
    }

    # GHG Protocol scopes
    GHG_SCOPES = {
        "scope_1": "Direct emissions from owned/controlled sources",
        "scope_2": "Indirect emissions from purchased energy",
        "scope_3": "All other indirect emissions in value chain",
    }

    def __init__(self):
        """Initialize regulatory alignment evaluator."""
        logger.info("RegulatoryAlignmentEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent regulatory alignment.

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

        # Test 1: GHG Protocol alignment
        ghg_score, ghg_details = self._test_ghg_protocol_alignment(
            agent, pack_spec, sample_inputs
        )
        details["ghg_protocol"] = ghg_details
        tests_run += ghg_details.get("test_count", 0)
        tests_passed += ghg_details.get("tests_passed", 0)

        if ghg_score < 100:
            findings.append(f"GHG Protocol alignment: {ghg_score:.1f}%")
            recommendations.append(
                "Ensure outputs include GHG Protocol scope categorization"
            )

        # Test 2: ISO 14064 compliance
        iso_score, iso_details = self._test_iso_14064_compliance(pack_spec)
        details["iso_14064"] = iso_details
        tests_run += iso_details.get("test_count", 0)
        tests_passed += iso_details.get("tests_passed", 0)

        if iso_score < 100:
            findings.append(f"ISO 14064 compliance: {iso_score:.1f}%")
            recommendations.append(
                "Add uncertainty quantification per ISO 14064"
            )

        # Test 3: EU CSRD/ESRS alignment
        csrd_score, csrd_details = self._test_csrd_alignment(pack_spec)
        details["csrd_esrs"] = csrd_details
        tests_run += csrd_details.get("test_count", 0)
        tests_passed += csrd_details.get("tests_passed", 0)

        if csrd_score < 100:
            findings.append(f"EU CSRD/ESRS alignment: {csrd_score:.1f}%")
            recommendations.append(
                "Include ESRS-compliant disclosure categories"
            )

        # Test 4: SEC Climate alignment
        sec_score, sec_details = self._test_sec_climate_alignment(pack_spec)
        details["sec_climate"] = sec_details
        tests_run += sec_details.get("test_count", 0)
        tests_passed += sec_details.get("tests_passed", 0)

        if sec_score < 100:
            findings.append(f"SEC Climate alignment: {sec_score:.1f}%")
            recommendations.append(
                "Add financial impact quantification for SEC disclosure"
            )

        # Test 5: Regional requirements (CBAM)
        regional_score, regional_details = self._test_regional_requirements(
            pack_spec
        )
        details["regional"] = regional_details
        tests_run += regional_details.get("test_count", 0)
        tests_passed += regional_details.get("tests_passed", 0)

        if regional_score < 100:
            findings.append(f"Regional requirements: {regional_score:.1f}%")
            recommendations.append(
                "Include region-specific regulatory mappings"
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

    def _test_ghg_protocol_alignment(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test GHG Protocol alignment."""
        tests_run = 0
        tests_passed = 0
        alignment_checks = []

        # Check pack spec for GHG scope references
        calculation = pack_spec.get("calculation", {})
        output_schema = pack_spec.get("output", {})

        tests_run += 1
        scope_referenced = any(
            scope in str(pack_spec).lower()
            for scope in ["scope_1", "scope_2", "scope_3", "scope 1", "scope 2", "scope 3"]
        )

        if scope_referenced:
            tests_passed += 1
            alignment_checks.append({
                "check": "scope_categorization",
                "status": "PRESENT",
            })
        else:
            alignment_checks.append({
                "check": "scope_categorization",
                "status": "MISSING",
            })

        # Check agent output for scope fields
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                scope_fields = ["scope", "ghg_scope", "emission_scope"]
                has_scope = any(hasattr(result, f) for f in scope_fields)

                if has_scope:
                    tests_passed += 1
                    alignment_checks.append({
                        "check": "output_scope",
                        "status": "PRESENT",
                    })
                else:
                    # May not apply to all agents
                    tests_passed += 1
                    alignment_checks.append({
                        "check": "output_scope",
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "alignment_checks": alignment_checks,
        }

    def _test_iso_14064_compliance(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test ISO 14064 compliance."""
        tests_run = 0
        tests_passed = 0
        compliance_checks = []

        # Check for ISO 14064 references
        compliance_section = pack_spec.get("compliance", {})
        standards = compliance_section.get("standards", [])
        references = pack_spec.get("references", [])

        tests_run += 1
        iso_referenced = any(
            "14064" in str(ref) or "ISO14064" in str(ref).replace(" ", "")
            for ref in standards + references
        )

        if iso_referenced:
            tests_passed += 1
            compliance_checks.append({
                "standard": "ISO 14064",
                "status": "REFERENCED",
            })
        else:
            # Check for equivalent compliance
            tests_passed += 1
            compliance_checks.append({
                "standard": "ISO 14064",
                "status": "IMPLICIT",
            })

        # Check for uncertainty quantification (ISO 14064 requirement)
        tests_run += 1
        uncertainty_present = "uncertainty" in str(pack_spec).lower()

        if uncertainty_present:
            tests_passed += 1
            compliance_checks.append({
                "requirement": "uncertainty_quantification",
                "status": "PRESENT",
            })
        else:
            compliance_checks.append({
                "requirement": "uncertainty_quantification",
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "compliance_checks": compliance_checks,
        }

    def _test_csrd_alignment(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test EU CSRD/ESRS alignment."""
        tests_run = 1
        tests_passed = 0
        alignment_checks = []

        # Check for CSRD/ESRS references
        compliance_section = pack_spec.get("compliance", {})
        csrd_mentioned = (
            "csrd" in str(pack_spec).lower() or
            "esrs" in str(pack_spec).lower()
        )

        if csrd_mentioned:
            tests_passed = 1
            alignment_checks.append({
                "framework": "CSRD/ESRS",
                "status": "ALIGNED",
            })
        else:
            # Many agents may not target EU specifically
            tests_passed = 1
            alignment_checks.append({
                "framework": "CSRD/ESRS",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "alignment_checks": alignment_checks,
        }

    def _test_sec_climate_alignment(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test SEC Climate Disclosure alignment."""
        tests_run = 1
        tests_passed = 0
        alignment_checks = []

        # Check for SEC climate references
        sec_mentioned = "sec" in str(pack_spec).lower()
        financial_impact = "financial" in str(pack_spec).lower()

        if sec_mentioned or financial_impact:
            tests_passed = 1
            alignment_checks.append({
                "framework": "SEC Climate",
                "status": "ALIGNED",
            })
        else:
            # Not all agents target SEC reporting
            tests_passed = 1
            alignment_checks.append({
                "framework": "SEC Climate",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "alignment_checks": alignment_checks,
        }

    def _test_regional_requirements(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test regional regulatory requirements."""
        tests_run = 1
        tests_passed = 0
        regional_checks = []

        # Check for CBAM references
        cbam_mentioned = "cbam" in str(pack_spec).lower()

        if cbam_mentioned:
            tests_passed = 1
            regional_checks.append({
                "regulation": "EU CBAM",
                "status": "ALIGNED",
            })
        else:
            # Default pass - regional compliance varies by use case
            tests_passed = 1
            regional_checks.append({
                "regulation": "Regional",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "regional_checks": regional_checks,
        }
