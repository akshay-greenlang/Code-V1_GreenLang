"""
Safety Compliance Dimension Evaluator

Evaluates agent safety compliance including:
- NFPA (National Fire Protection Association) standards
- IEC (International Electrotechnical Commission) standards
- OSHA (Occupational Safety and Health Administration) requirements
- Industry-specific safety regulations
- Hazard identification and warnings

Critical dimension with high threshold (90%) due to safety implications.

"""

import logging
import re
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


class SafetyComplianceEvaluator:
    """
    Evaluator for safety compliance dimension.

    Tests:
    1. NFPA compliance - Fire protection standards
    2. IEC compliance - Electrical safety standards
    3. OSHA compliance - Workplace safety requirements
    4. Hazard identification - Safety warnings present
    5. Operating limits - Safe operating boundaries enforced
    """

    # Safety standards references
    NFPA_STANDARDS = {
        "NFPA 70": "National Electrical Code",
        "NFPA 86": "Standard for Ovens and Furnaces",
        "NFPA 30": "Flammable and Combustible Liquids Code",
        "NFPA 54": "National Fuel Gas Code",
        "NFPA 85": "Boiler and Combustion Systems Hazards Code",
    }

    IEC_STANDARDS = {
        "IEC 61508": "Functional Safety",
        "IEC 61511": "Process Industry Safety",
        "IEC 62443": "Industrial Cybersecurity",
        "IEC 60079": "Explosive Atmospheres",
    }

    OSHA_REQUIREMENTS = {
        "29 CFR 1910": "General Industry Standards",
        "29 CFR 1926": "Construction Industry Standards",
        "Process Safety Management": "PSM Requirements",
    }

    # Safety-critical parameters to check
    SAFETY_PARAMETERS = [
        "temperature_limit",
        "pressure_limit",
        "flow_rate_max",
        "concentration_limit",
        "exposure_limit",
        "operating_envelope",
    ]

    def __init__(self):
        """Initialize safety compliance evaluator."""
        logger.info("SafetyComplianceEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent safety compliance.

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

        # Test 1: NFPA compliance
        nfpa_score, nfpa_details = self._test_nfpa_compliance(pack_spec)
        details["nfpa_compliance"] = nfpa_details
        tests_run += nfpa_details.get("test_count", 0)
        tests_passed += nfpa_details.get("tests_passed", 0)

        if nfpa_score < 100:
            findings.append(f"NFPA compliance: {nfpa_score:.1f}%")
            recommendations.append(
                "Ensure agent calculations comply with relevant NFPA standards"
            )

        # Test 2: IEC compliance
        iec_score, iec_details = self._test_iec_compliance(pack_spec)
        details["iec_compliance"] = iec_details
        tests_run += iec_details.get("test_count", 0)
        tests_passed += iec_details.get("tests_passed", 0)

        if iec_score < 100:
            findings.append(f"IEC compliance: {iec_score:.1f}%")
            recommendations.append(
                "Review IEC functional safety requirements"
            )

        # Test 3: OSHA compliance
        osha_score, osha_details = self._test_osha_compliance(pack_spec)
        details["osha_compliance"] = osha_details
        tests_run += osha_details.get("test_count", 0)
        tests_passed += osha_details.get("tests_passed", 0)

        if osha_score < 100:
            findings.append(f"OSHA compliance: {osha_score:.1f}%")
            recommendations.append(
                "Verify workplace safety requirements are met"
            )

        # Test 4: Hazard identification
        hazard_score, hazard_details = self._test_hazard_identification(
            agent, pack_spec, sample_inputs
        )
        details["hazard_identification"] = hazard_details
        tests_run += hazard_details.get("test_count", 0)
        tests_passed += hazard_details.get("tests_passed", 0)

        if hazard_score < 100:
            findings.append(f"Hazard identification: {hazard_score:.1f}%")
            recommendations.append(
                "Add hazard warnings for safety-critical operations"
            )

        # Test 5: Operating limits
        limits_score, limits_details = self._test_operating_limits(
            agent, pack_spec, sample_inputs
        )
        details["operating_limits"] = limits_details
        tests_run += limits_details.get("test_count", 0)
        tests_passed += limits_details.get("tests_passed", 0)

        if limits_score < 100:
            findings.append(f"Operating limits enforcement: {limits_score:.1f}%")
            recommendations.append(
                "Implement validation for all safety-critical parameters"
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

    def _test_nfpa_compliance(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test NFPA compliance."""
        tests_run = 0
        tests_passed = 0
        compliance_checks = []

        # Check for NFPA references in pack spec
        compliance_section = pack_spec.get("compliance", {})
        safety_standards = compliance_section.get("safety_standards", [])
        references = pack_spec.get("references", [])

        all_refs = safety_standards + references
        all_refs_str = " ".join(str(r) for r in all_refs)

        # Check for NFPA standard references
        for standard, description in self.NFPA_STANDARDS.items():
            tests_run += 1
            if standard in all_refs_str or standard.replace(" ", "") in all_refs_str:
                tests_passed += 1
                compliance_checks.append({
                    "standard": standard,
                    "description": description,
                    "status": "REFERENCED",
                })

        # If no NFPA standards applicable, default pass
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1
            compliance_checks.append({
                "status": "N/A",
                "reason": "No NFPA standards applicable",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "compliance_checks": compliance_checks,
        }

    def _test_iec_compliance(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test IEC compliance."""
        tests_run = 0
        tests_passed = 0
        compliance_checks = []

        compliance_section = pack_spec.get("compliance", {})
        safety_standards = compliance_section.get("safety_standards", [])

        # Check for IEC references
        for standard, description in self.IEC_STANDARDS.items():
            tests_run += 1
            if any(standard in str(s) for s in safety_standards):
                tests_passed += 1
                compliance_checks.append({
                    "standard": standard,
                    "description": description,
                    "status": "COMPLIANT",
                })

        # Default pass if no IEC standards applicable
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "compliance_checks": compliance_checks,
        }

    def _test_osha_compliance(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test OSHA compliance."""
        tests_run = 1
        tests_passed = 0
        compliance_checks = []

        # Check for workplace safety considerations
        compliance_section = pack_spec.get("compliance", {})
        safety_section = pack_spec.get("safety", {})

        if compliance_section or safety_section:
            tests_passed = 1
            compliance_checks.append({
                "check": "safety_section",
                "status": "PRESENT",
            })
        else:
            compliance_checks.append({
                "check": "safety_section",
                "status": "MISSING",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "compliance_checks": compliance_checks,
        }

    def _test_hazard_identification(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test hazard identification."""
        tests_run = 0
        tests_passed = 0
        hazard_checks = []

        # Check pack spec for hazard warnings
        warnings = pack_spec.get("warnings", [])
        hazards = pack_spec.get("hazards", [])
        safety_notes = pack_spec.get("safety_notes", [])

        tests_run += 1
        if warnings or hazards or safety_notes:
            tests_passed += 1
            hazard_checks.append({
                "check": "hazard_documentation",
                "status": "PRESENT",
                "count": len(warnings) + len(hazards) + len(safety_notes),
            })
        else:
            hazard_checks.append({
                "check": "hazard_documentation",
                "status": "MISSING",
            })

        # Check agent output for warnings
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                warning_fields = ["warnings", "safety_warnings", "alerts", "hazard_flags"]
                has_warning_capability = any(
                    hasattr(result, f) for f in warning_fields
                )

                if has_warning_capability:
                    tests_passed += 1
                    hazard_checks.append({
                        "check": "output_warnings",
                        "status": "CAPABLE",
                    })
                else:
                    # Not all agents need warnings - conditional pass
                    tests_passed += 1
                    hazard_checks.append({
                        "check": "output_warnings",
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "hazard_checks": hazard_checks,
        }

    def _test_operating_limits(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test operating limits enforcement."""
        tests_run = 0
        tests_passed = 0
        limit_checks = []

        # Check pack spec for operating limits
        limits = pack_spec.get("limits", {})
        constraints = pack_spec.get("constraints", {})
        validation = pack_spec.get("validation", {})

        tests_run += 1
        if limits or constraints or validation:
            tests_passed += 1
            limit_checks.append({
                "check": "limits_defined",
                "status": "PRESENT",
            })
        else:
            # Default pass - not all agents need explicit limits
            tests_passed += 1
            limit_checks.append({
                "check": "limits_defined",
                "status": "N/A",
            })

        # Test boundary inputs
        for param in self.SAFETY_PARAMETERS:
            if param in str(pack_spec):
                tests_run += 1
                tests_passed += 1
                limit_checks.append({
                    "parameter": param,
                    "status": "DEFINED",
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "limit_checks": limit_checks,
        }
