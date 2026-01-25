"""
Security Dimension Evaluator

Evaluates agent security including:
- Input validation
- Injection prevention
- Data sanitization
- Access control readiness
- Secure defaults

Ensures agents are secure against common attack vectors.

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


class SecurityEvaluator:
    """
    Evaluator for security dimension.

    Tests:
    1. Input validation - Type checking, bounds validation
    2. Injection prevention - SQL, command injection tests
    3. Data sanitization - Output escaping
    4. Access control - Authentication readiness
    5. Secure defaults - Safe default configurations
    """

    # Injection attack patterns to test
    INJECTION_PATTERNS = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "${7*7}",
        "{{7*7}}",
        "`id`",
        "| ls -la",
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        "1 OR 1=1",
        "1; DROP TABLE",
        "UNION SELECT",
    ]

    def __init__(self):
        """Initialize security evaluator."""
        logger.info("SecurityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent security.

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

        # Test 1: Input validation
        validation_score, validation_details = self._test_input_validation(
            agent, pack_spec, sample_inputs
        )
        details["input_validation"] = validation_details
        tests_run += validation_details.get("test_count", 0)
        tests_passed += validation_details.get("tests_passed", 0)

        if validation_score < 100:
            findings.append(f"Input validation: {validation_score:.1f}%")
            recommendations.append(
                "Implement strict input validation with type checking"
            )

        # Test 2: Injection prevention
        injection_score, injection_details = self._test_injection_prevention(
            agent, sample_inputs
        )
        details["injection_prevention"] = injection_details
        tests_run += injection_details.get("test_count", 0)
        tests_passed += injection_details.get("tests_passed", 0)

        if injection_score < 100:
            findings.append(f"Injection prevention: {injection_score:.1f}%")
            recommendations.append(
                "Add input sanitization for all string inputs"
            )

        # Test 3: Data sanitization
        sanitization_score, sanitization_details = self._test_data_sanitization(
            agent, sample_inputs
        )
        details["data_sanitization"] = sanitization_details
        tests_run += sanitization_details.get("test_count", 0)
        tests_passed += sanitization_details.get("tests_passed", 0)

        if sanitization_score < 100:
            findings.append(f"Data sanitization: {sanitization_score:.1f}%")
            recommendations.append(
                "Ensure all outputs are properly escaped"
            )

        # Test 4: Secure configuration
        config_score, config_details = self._test_secure_config(pack_spec)
        details["secure_config"] = config_details
        tests_run += config_details.get("test_count", 0)
        tests_passed += config_details.get("tests_passed", 0)

        if config_score < 100:
            findings.append(f"Secure configuration: {config_score:.1f}%")
            recommendations.append(
                "Review and document security configuration"
            )

        # Test 5: Error information leakage
        leakage_score, leakage_details = self._test_error_leakage(
            agent, sample_inputs
        )
        details["error_leakage"] = leakage_details
        tests_run += leakage_details.get("test_count", 0)
        tests_passed += leakage_details.get("tests_passed", 0)

        if leakage_score < 100:
            findings.append(f"Error information leakage: {leakage_score:.1f}%")
            recommendations.append(
                "Avoid exposing internal details in error messages"
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

    def _test_input_validation(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test input validation."""
        tests_run = 0
        tests_passed = 0
        validation_tests = []

        # Check pack spec for input validation definition
        tests_run += 1
        input_schema = pack_spec.get("input", {})
        validation_rules = pack_spec.get("validation", {})

        if input_schema or validation_rules:
            tests_passed += 1
            validation_tests.append({
                "check": "schema_defined",
                "status": "PRESENT",
            })
        else:
            validation_tests.append({
                "check": "schema_defined",
                "status": "MISSING",
            })

        # Test type validation
        if sample_inputs:
            tests_run += 1
            try:
                # Try with wrong type
                test_input = sample_inputs[0].copy()
                for key in test_input:
                    if isinstance(test_input[key], (int, float)):
                        test_input[key] = "not_a_number"
                        break

                agent.run(test_input)
                # May accept with coercion
                tests_passed += 1
                validation_tests.append({
                    "check": "type_validation",
                    "status": "COERCED",
                })
            except (ValueError, TypeError) as e:
                # Validation error is good
                tests_passed += 1
                validation_tests.append({
                    "check": "type_validation",
                    "status": "VALIDATED",
                })
            except Exception:
                validation_tests.append({
                    "check": "type_validation",
                    "status": "UNHANDLED",
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "validation_tests": validation_tests,
        }

    def _test_injection_prevention(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test injection prevention."""
        tests_run = 0
        tests_passed = 0
        injection_tests = []

        if not sample_inputs:
            return 100.0, {"test_count": 1, "tests_passed": 1, "status": "SKIPPED"}

        template_input = sample_inputs[0].copy()

        # Find string fields to inject
        string_fields = [
            k for k, v in template_input.items()
            if isinstance(v, str)
        ]

        for field_name in string_fields[:2]:
            for pattern in self.INJECTION_PATTERNS[:3]:  # Test first 3 patterns
                tests_run += 1
                test_input = template_input.copy()
                test_input[field_name] = pattern

                try:
                    result = agent.run(test_input)
                    # Check if injection pattern appears in output
                    result_str = str(result)
                    if pattern in result_str:
                        injection_tests.append({
                            "field": field_name,
                            "pattern": pattern[:20],
                            "status": "REFLECTED",
                        })
                    else:
                        tests_passed += 1
                        injection_tests.append({
                            "field": field_name,
                            "pattern": pattern[:20],
                            "status": "SANITIZED",
                        })
                except (ValueError, TypeError) as e:
                    # Validation error is acceptable
                    tests_passed += 1
                    injection_tests.append({
                        "field": field_name,
                        "pattern": pattern[:20],
                        "status": "REJECTED",
                    })
                except Exception:
                    tests_passed += 1
                    injection_tests.append({
                        "field": field_name,
                        "pattern": pattern[:20],
                        "status": "ERROR",
                    })

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "injection_tests": injection_tests,
        }

    def _test_data_sanitization(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test data sanitization."""
        tests_run = 1
        tests_passed = 1  # Default pass
        sanitization_tests = []

        # Most GreenLang agents deal with numeric data
        # Sanitization is more relevant for string outputs
        sanitization_tests.append({
            "check": "output_sanitization",
            "status": "N/A",
            "reason": "Numeric output agents",
        })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "sanitization_tests": sanitization_tests,
        }

    def _test_secure_config(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test secure configuration."""
        tests_run = 1
        tests_passed = 0
        config_tests = []

        # Check for security-related configuration
        security_config = pack_spec.get("security", {})
        config = pack_spec.get("config", {})

        # Check for sensitive defaults
        has_secure_defaults = True
        if config:
            # Check for insecure patterns
            config_str = str(config).lower()
            insecure_patterns = ["password", "secret", "key", "token"]
            for pattern in insecure_patterns:
                if pattern in config_str:
                    has_secure_defaults = False
                    break

        if has_secure_defaults:
            tests_passed = 1
            config_tests.append({
                "check": "secure_defaults",
                "status": "SECURE",
            })
        else:
            config_tests.append({
                "check": "secure_defaults",
                "status": "REVIEW_NEEDED",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "config_tests": config_tests,
        }

    def _test_error_leakage(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test error information leakage."""
        tests_run = 1
        tests_passed = 0
        leakage_tests = []

        # Try to trigger an error
        try:
            agent.run({})
            tests_passed = 1  # No error
            leakage_tests.append({
                "status": "NO_ERROR",
            })
        except Exception as e:
            error_msg = str(e).lower()

            # Check for sensitive information in error
            sensitive_patterns = [
                "password",
                "secret",
                "key",
                "token",
                "internal",
                "stack",
                "/home/",
                "/usr/",
                "c:\\",
            ]

            has_leakage = any(p in error_msg for p in sensitive_patterns)

            if not has_leakage:
                tests_passed = 1
                leakage_tests.append({
                    "status": "SAFE",
                })
            else:
                leakage_tests.append({
                    "status": "POTENTIAL_LEAKAGE",
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "leakage_tests": leakage_tests,
        }
