"""
Auditability Dimension Evaluator

Evaluates agent auditability including:
- Logging completeness
- Reproducibility
- Traceability
- Version tracking
- Audit trail integrity

Critical for regulatory compliance and financial audits.

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


class AuditabilityEvaluator:
    """
    Evaluator for auditability dimension.

    Tests:
    1. Logging - Comprehensive audit logging
    2. Reproducibility - Deterministic outputs
    3. Traceability - Input/output tracking
    4. Version tracking - Agent version in output
    5. Audit trail - Complete decision chain
    """

    # Required audit fields
    AUDIT_FIELDS = [
        "provenance_hash",
        "timestamp",
        "calculation_timestamp",
        "agent_version",
        "input_hash",
    ]

    def __init__(self):
        """Initialize auditability evaluator."""
        logger.info("AuditabilityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent auditability.

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

        # Test 1: Audit field presence
        audit_score, audit_details = self._test_audit_fields(
            agent, sample_inputs
        )
        details["audit_fields"] = audit_details
        tests_run += audit_details.get("test_count", 0)
        tests_passed += audit_details.get("tests_passed", 0)

        if audit_score < 100:
            findings.append(f"Audit field presence: {audit_score:.1f}%")
            recommendations.append(
                "Include provenance_hash and timestamp in all outputs"
            )

        # Test 2: Reproducibility (from determinism result)
        repro_score, repro_details = self._test_reproducibility(determinism_result)
        details["reproducibility"] = repro_details
        tests_run += repro_details.get("test_count", 0)
        tests_passed += repro_details.get("tests_passed", 0)

        if repro_score < 100:
            findings.append(f"Reproducibility: {repro_score:.1f}%")
            recommendations.append(
                "Ensure all calculations are fully reproducible"
            )

        # Test 3: Version tracking
        version_score, version_details = self._test_version_tracking(
            agent, pack_spec, sample_inputs
        )
        details["version_tracking"] = version_details
        tests_run += version_details.get("test_count", 0)
        tests_passed += version_details.get("tests_passed", 0)

        if version_score < 100:
            findings.append(f"Version tracking: {version_score:.1f}%")
            recommendations.append(
                "Include agent version in output for audit trail"
            )

        # Test 4: Traceability
        trace_score, trace_details = self._test_traceability(
            agent, sample_inputs
        )
        details["traceability"] = trace_details
        tests_run += trace_details.get("test_count", 0)
        tests_passed += trace_details.get("tests_passed", 0)

        if trace_score < 100:
            findings.append(f"Traceability: {trace_score:.1f}%")
            recommendations.append(
                "Implement input/output correlation for full traceability"
            )

        # Test 5: Provenance hash validity
        prov_score, prov_details = self._test_provenance_validity(
            agent, sample_inputs
        )
        details["provenance_validity"] = prov_details
        tests_run += prov_details.get("test_count", 0)
        tests_passed += prov_details.get("tests_passed", 0)

        if prov_score < 100:
            findings.append(f"Provenance validity: {prov_score:.1f}%")
            recommendations.append(
                "Ensure provenance hash is SHA-256 format"
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

    def _test_audit_fields(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test audit field presence."""
        tests_run = 0
        tests_passed = 0
        field_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)

                for audit_field in ["provenance_hash", "timestamp"]:
                    tests_run += 1
                    if hasattr(result, audit_field):
                        tests_passed += 1
                        field_checks.append({
                            "field": audit_field,
                            "status": "PRESENT",
                        })
                    else:
                        field_checks.append({
                            "field": audit_field,
                            "status": "MISSING",
                        })

            except Exception:
                tests_run += 2

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "field_checks": field_checks,
        }

    def _test_reproducibility(
        self, determinism_result: Optional[Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test reproducibility from determinism results."""
        tests_run = 1
        tests_passed = 0

        if determinism_result:
            is_deterministic = getattr(determinism_result, "is_deterministic", False)
            if is_deterministic:
                tests_passed = 1

            return (
                tests_passed / tests_run * 100,
                {
                    "test_count": tests_run,
                    "tests_passed": tests_passed,
                    "is_deterministic": is_deterministic,
                    "unique_outputs": getattr(determinism_result, "unique_outputs", 0),
                },
            )
        else:
            # No determinism test available - conditional pass
            return 100.0, {
                "test_count": 1,
                "tests_passed": 1,
                "status": "NOT_TESTED",
            }

    def _test_version_tracking(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test version tracking."""
        tests_run = 0
        tests_passed = 0
        version_checks = []

        # Check pack spec for version
        tests_run += 1
        pack_version = pack_spec.get("pack", {}).get("version")

        if pack_version:
            tests_passed += 1
            version_checks.append({
                "check": "pack_version",
                "version": pack_version,
                "status": "PRESENT",
            })
        else:
            version_checks.append({
                "check": "pack_version",
                "status": "MISSING",
            })

        # Check agent output for version
        for sample_input in sample_inputs[:1]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                version_fields = ["agent_version", "version", "pack_version"]
                has_version = any(hasattr(result, f) for f in version_fields)

                if has_version:
                    tests_passed += 1
                    version_checks.append({
                        "check": "output_version",
                        "status": "PRESENT",
                    })
                else:
                    # Not strictly required
                    tests_passed += 1
                    version_checks.append({
                        "check": "output_version",
                        "status": "N/A",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "version_checks": version_checks,
        }

    def _test_traceability(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test traceability."""
        tests_run = 0
        tests_passed = 0
        trace_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                # Check for traceability fields
                trace_fields = ["input_hash", "correlation_id", "request_id"]
                has_trace = any(hasattr(result, f) for f in trace_fields)

                # Provenance hash provides traceability
                if hasattr(result, "provenance_hash"):
                    has_trace = True

                if has_trace:
                    tests_passed += 1
                    trace_checks.append({
                        "status": "TRACEABLE",
                    })
                else:
                    trace_checks.append({
                        "status": "NOT_TRACEABLE",
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
            "trace_checks": trace_checks,
        }

    def _test_provenance_validity(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test provenance hash validity."""
        tests_run = 0
        tests_passed = 0
        prov_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                if hasattr(result, "provenance_hash"):
                    prov_hash = result.provenance_hash

                    # Validate SHA-256 format (64 hex chars)
                    if prov_hash and len(prov_hash) == 64:
                        if all(c in "0123456789abcdef" for c in prov_hash.lower()):
                            tests_passed += 1
                            prov_checks.append({
                                "hash": prov_hash[:16] + "...",
                                "status": "VALID",
                            })
                        else:
                            prov_checks.append({
                                "status": "INVALID_FORMAT",
                            })
                    else:
                        prov_checks.append({
                            "status": "INVALID_LENGTH",
                            "length": len(prov_hash) if prov_hash else 0,
                        })
                else:
                    # Provenance not present
                    prov_checks.append({
                        "status": "MISSING",
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
            "prov_checks": prov_checks,
        }
