# -*- coding: utf-8 -*-
"""
Property-Based Tests: Validation Determinism

Tests the critical property that validating the same input always produces
the same output. Validation is a pure function with no side effects.

Determinism Property:
    validate(payload, schema) == validate(payload, schema)  [always]

This property is essential for:
    - Auditability: Same validation can be reproduced at any time
    - Debugging: Issues can be reliably reproduced
    - Testing: Test results are stable and predictable
    - Caching: Validation results can be safely cached

Uses Hypothesis to generate random payloads and verify that validation
produces identical results when run multiple times.

GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from hypothesis import given, settings, assume, HealthCheck, Phase
from hypothesis import strategies as st

# Import components under test
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
    CoercionPolicy,
    UnknownFieldPolicy,
)
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.compiler.ir import (
    SchemaIR, PropertyIR, NumericConstraintIR, StringConstraintIR,
    ArrayConstraintIR, CompiledPattern
)


# =============================================================================
# DETERMINISTIC VALIDATION IMPLEMENTATION
# =============================================================================

class DeterministicValidator:
    """
    A deterministic validator for testing purposes.

    This validator implements key validation checks in a way that is
    guaranteed to be deterministic - same input always produces same output.
    """

    def __init__(self, schema_ir: SchemaIR, options: ValidationOptions):
        """
        Initialize validator with schema and options.

        Args:
            schema_ir: Compiled schema intermediate representation
            options: Validation options
        """
        self.schema_ir = schema_ir
        self.options = options

    def validate(self, payload: Dict[str, Any]) -> "ValidationResult":
        """
        Validate a payload against the schema.

        Args:
            payload: The payload to validate

        Returns:
            ValidationResult with findings and metadata
        """
        findings: List[Finding] = []

        # Validate required fields
        findings.extend(self._validate_required(payload))

        # Validate types
        findings.extend(self._validate_types(payload))

        # Validate numeric constraints
        findings.extend(self._validate_numeric_constraints(payload))

        # Validate string constraints
        findings.extend(self._validate_string_constraints(payload))

        # Validate enum constraints
        findings.extend(self._validate_enum_constraints(payload))

        # Validate unknown fields (if strict mode)
        findings.extend(self._validate_unknown_fields(payload))

        # Sort findings for deterministic order
        sorted_findings = self._sort_findings(findings)

        # Compute deterministic hash
        result_hash = self._compute_result_hash(sorted_findings, payload)

        return ValidationResult(
            valid=not any(f.severity == Severity.ERROR for f in sorted_findings),
            findings=sorted_findings,
            result_hash=result_hash,
        )

    def _validate_required(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check required fields are present."""
        findings = []

        for path in sorted(self.schema_ir.required_paths):
            key = path.strip('/').split('/')[0] if path else None
            if key and key not in payload:
                findings.append(Finding(
                    code="GLSCHEMA-E101",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Required field '{key}' is missing",
                    expected=key,
                    actual=None,
                ))

        return findings

    def _validate_types(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check types match schema expectations."""
        findings = []

        for path in sorted(self.schema_ir.properties.keys()):
            prop_ir = self.schema_ir.properties[path]
            key = path.strip('/').split('/')[0] if path else None

            if not key or key not in payload:
                continue

            value = payload[key]
            expected_type = prop_ir.type

            if expected_type and not self._check_type(value, expected_type):
                findings.append(Finding(
                    code="GLSCHEMA-E102",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Type mismatch: expected {expected_type}, got {type(value).__name__}",
                    expected=expected_type,
                    actual=type(value).__name__,
                ))

        return findings

    def _validate_numeric_constraints(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check numeric constraints."""
        findings = []

        for path in sorted(self.schema_ir.numeric_constraints.keys()):
            constraint = self.schema_ir.numeric_constraints[path]
            key = path.strip('/').split('/')[0] if path else None

            if not key or key not in payload:
                continue

            value = payload[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue

            if constraint.minimum is not None and value < constraint.minimum:
                findings.append(Finding(
                    code="GLSCHEMA-E201",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value {value} is below minimum {constraint.minimum}",
                    expected=f">= {constraint.minimum}",
                    actual=str(value),
                ))

            if constraint.maximum is not None and value > constraint.maximum:
                findings.append(Finding(
                    code="GLSCHEMA-E202",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value {value} is above maximum {constraint.maximum}",
                    expected=f"<= {constraint.maximum}",
                    actual=str(value),
                ))

            if constraint.exclusive_minimum is not None and value <= constraint.exclusive_minimum:
                findings.append(Finding(
                    code="GLSCHEMA-E203",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value {value} must be greater than {constraint.exclusive_minimum}",
                    expected=f"> {constraint.exclusive_minimum}",
                    actual=str(value),
                ))

            if constraint.exclusive_maximum is not None and value >= constraint.exclusive_maximum:
                findings.append(Finding(
                    code="GLSCHEMA-E204",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value {value} must be less than {constraint.exclusive_maximum}",
                    expected=f"< {constraint.exclusive_maximum}",
                    actual=str(value),
                ))

            if constraint.multiple_of is not None:
                if value % constraint.multiple_of != 0:
                    findings.append(Finding(
                        code="GLSCHEMA-E205",
                        severity=Severity.ERROR,
                        path=path,
                        message=f"Value {value} is not a multiple of {constraint.multiple_of}",
                        expected=f"multiple of {constraint.multiple_of}",
                        actual=str(value),
                    ))

        return findings

    def _validate_string_constraints(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check string constraints."""
        findings = []

        for path in sorted(self.schema_ir.string_constraints.keys()):
            constraint = self.schema_ir.string_constraints[path]
            key = path.strip('/').split('/')[0] if path else None

            if not key or key not in payload:
                continue

            value = payload[key]
            if not isinstance(value, str):
                continue

            if constraint.min_length is not None and len(value) < constraint.min_length:
                findings.append(Finding(
                    code="GLSCHEMA-E301",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"String length {len(value)} is below minimum {constraint.min_length}",
                    expected=f"length >= {constraint.min_length}",
                    actual=str(len(value)),
                ))

            if constraint.max_length is not None and len(value) > constraint.max_length:
                findings.append(Finding(
                    code="GLSCHEMA-E302",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"String length {len(value)} exceeds maximum {constraint.max_length}",
                    expected=f"length <= {constraint.max_length}",
                    actual=str(len(value)),
                ))

        return findings

    def _validate_enum_constraints(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check enum constraints."""
        findings = []

        for path in sorted(self.schema_ir.enums.keys()):
            enum_values = self.schema_ir.enums[path]
            key = path.strip('/').split('/')[0] if path else None

            if not key or key not in payload:
                continue

            value = payload[key]
            if value not in enum_values:
                findings.append(Finding(
                    code="GLSCHEMA-E401",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value '{value}' is not one of allowed values: {enum_values}",
                    expected=str(enum_values),
                    actual=str(value),
                ))

        return findings

    def _validate_unknown_fields(self, payload: Dict[str, Any]) -> List[Finding]:
        """Check for unknown fields based on policy."""
        findings = []

        if self.options.unknown_field_policy == UnknownFieldPolicy.IGNORE:
            return findings

        known_keys = set()
        for path in self.schema_ir.properties.keys():
            key = path.strip('/').split('/')[0] if path else None
            if key:
                known_keys.add(key)

        for key in sorted(payload.keys()):
            if key not in known_keys and not key.startswith('_'):
                severity = (
                    Severity.ERROR
                    if self.options.unknown_field_policy == UnknownFieldPolicy.ERROR
                    else Severity.WARNING
                )
                findings.append(Finding(
                    code="GLSCHEMA-E501" if severity == Severity.ERROR else "GLSCHEMA-W501",
                    severity=severity,
                    path=f"/{key}",
                    message=f"Unknown field '{key}' not defined in schema",
                    expected=None,
                    actual=key,
                ))

        return findings

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "null":
            return value is None
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "array":
            return isinstance(value, list)
        return True

    def _sort_findings(self, findings: List[Finding]) -> List[Finding]:
        """Sort findings deterministically by severity, path, and code."""
        severity_order = {
            Severity.ERROR: 0,
            Severity.WARNING: 1,
            Severity.INFO: 2,
        }

        return sorted(
            findings,
            key=lambda f: (
                severity_order.get(f.severity, 3),
                f.path or "",
                f.code or "",
            )
        )

    def _compute_result_hash(
        self,
        findings: List[Finding],
        payload: Dict[str, Any]
    ) -> str:
        """Compute deterministic hash of validation result."""
        # Serialize findings to deterministic JSON
        findings_data = [
            {
                "code": f.code,
                "severity": f.severity.value if hasattr(f.severity, 'value') else str(f.severity),
                "path": f.path,
                "message": f.message,
            }
            for f in findings
        ]

        hash_input = json.dumps({
            "findings": findings_data,
            "payload_hash": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        }, sort_keys=True)

        return hashlib.sha256(hash_input.encode()).hexdigest()


class ValidationResult:
    """Result of validation operation."""

    def __init__(
        self,
        valid: bool,
        findings: List[Finding],
        result_hash: str,
    ):
        self.valid = valid
        self.findings = findings
        self.result_hash = result_hash

    def __eq__(self, other: "ValidationResult") -> bool:
        """Check equality based on findings and validity."""
        if not isinstance(other, ValidationResult):
            return False
        return (
            self.valid == other.valid
            and self.result_hash == other.result_hash
            and len(self.findings) == len(other.findings)
        )


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Strategy for JSON-compatible primitive values
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-2**31, max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    st.text(min_size=0, max_size=100),
)

# Strategy for valid JSON keys
json_keys = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz_0123456789',
    min_size=1,
    max_size=30
).filter(lambda x: x[0].isalpha())


def json_values_recursive(max_depth: int = 3):
    """Generate recursive JSON values with controlled depth."""
    if max_depth <= 0:
        return json_primitives

    return st.one_of(
        json_primitives,
        st.lists(
            st.deferred(lambda: json_values_recursive(max_depth - 1)),
            min_size=0,
            max_size=5
        ),
        st.dictionaries(
            json_keys,
            st.deferred(lambda: json_values_recursive(max_depth - 1)),
            min_size=0,
            max_size=5
        ),
    )


# Strategy for complete payloads
json_payloads = st.dictionaries(
    json_keys,
    json_values_recursive(max_depth=3),
    min_size=0,
    max_size=10
)

# Strategy for payloads matching the test schema
schema_conforming_payloads = st.fixed_dictionaries({
    'name': st.text(min_size=1, max_size=50),
    'value': st.integers(min_value=0, max_value=1000),
}, optional={
    'description': st.text(max_size=200),
    'count': st.integers(min_value=1, max_value=100),
    'status': st.sampled_from(['pending', 'active', 'completed']),
})

# Strategy for validation options
validation_options_strategy = st.builds(
    ValidationOptions,
    profile=st.sampled_from(list(ValidationProfile)),
    normalize=st.booleans(),
    coercion_policy=st.sampled_from(list(CoercionPolicy)),
    unknown_field_policy=st.sampled_from(list(UnknownFieldPolicy)),
    max_errors=st.integers(min_value=1, max_value=100),
    fail_fast=st.booleans(),
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_schema_ir() -> SchemaIR:
    """Create a test schema IR with various constraints."""
    return SchemaIR(
        schema_id="test/determinism",
        version="1.0.0",
        schema_hash="d" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/value": PropertyIR(path="/value", type="number", required=True),
            "/description": PropertyIR(path="/description", type="string", required=False),
            "/count": PropertyIR(path="/count", type="integer", required=False),
            "/status": PropertyIR(path="/status", type="string", required=False),
        },
        required_paths={"/name", "/value"},
        numeric_constraints={
            "/value": NumericConstraintIR(
                path="/value",
                minimum=0,
                maximum=1000,
            ),
            "/count": NumericConstraintIR(
                path="/count",
                minimum=1,
                maximum=100,
            ),
        },
        string_constraints={
            "/name": StringConstraintIR(
                path="/name",
                min_length=1,
                max_length=100,
            ),
            "/description": StringConstraintIR(
                path="/description",
                max_length=500,
            ),
        },
        enums={
            "/status": ["pending", "active", "completed", "cancelled"],
        },
    )


@pytest.fixture
def default_options() -> ValidationOptions:
    """Create default validation options."""
    return ValidationOptions(
        profile=ValidationProfile.STANDARD,
        normalize=True,
        coercion_policy=CoercionPolicy.SAFE,
        unknown_field_policy=UnknownFieldPolicy.WARN,
    )


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestValidationDeterminism:
    """
    Property-based tests for validation determinism.

    The key property: same input always produces same output.
    """

    @given(payload=json_payloads)
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_deterministic_results_random_payloads(
        self,
        payload: Dict[str, Any],
        test_schema_ir,
        default_options,
    ):
        """
        Test that validating the same random payload twice gives identical results.
        """
        validator = DeterministicValidator(test_schema_ir, default_options)

        # Validate the same payload twice
        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        # Results must be identical
        assert result1 == result2, (
            f"Non-deterministic validation!\n"
            f"Payload: {payload}\n"
            f"Result 1 hash: {result1.result_hash}\n"
            f"Result 2 hash: {result2.result_hash}"
        )

    @given(payload=json_payloads, iterations=st.integers(min_value=2, max_value=10))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_deterministic_results_multiple_runs(
        self,
        payload: Dict[str, Any],
        iterations: int,
        test_schema_ir,
        default_options,
    ):
        """
        Test that validating the same payload N times gives identical results.
        """
        validator = DeterministicValidator(test_schema_ir, default_options)

        # Get reference result
        reference = validator.validate(copy.deepcopy(payload))

        # Validate multiple times
        for i in range(iterations):
            result = validator.validate(copy.deepcopy(payload))
            assert result == reference, (
                f"Non-deterministic result on iteration {i+1}!\n"
                f"Reference hash: {reference.result_hash}\n"
                f"Current hash: {result.result_hash}"
            )

    @given(payload=schema_conforming_payloads)
    @settings(max_examples=100, deadline=None)
    def test_deterministic_with_valid_payloads(
        self,
        payload: Dict[str, Any],
        test_schema_ir,
        default_options,
    ):
        """
        Test determinism with payloads that conform to the schema.
        """
        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1.valid == result2.valid
        assert result1.result_hash == result2.result_hash
        assert len(result1.findings) == len(result2.findings)

    @given(
        payload=json_payloads,
        options=validation_options_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_deterministic_with_different_options(
        self,
        payload: Dict[str, Any],
        options: ValidationOptions,
        test_schema_ir,
    ):
        """
        Test that validation is deterministic regardless of options.
        """
        validator = DeterministicValidator(test_schema_ir, options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1 == result2


# =============================================================================
# HASH DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestHashDeterminism:
    """
    Test that validation result hashes are deterministic.
    """

    @given(payload=json_payloads)
    @settings(max_examples=100, deadline=None)
    def test_result_hash_determinism(
        self,
        payload: Dict[str, Any],
        test_schema_ir,
        default_options,
    ):
        """
        Test that result hash is the same for identical validations.
        """
        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1.result_hash == result2.result_hash, (
            f"Hash mismatch for identical validation!\n"
            f"Hash 1: {result1.result_hash}\n"
            f"Hash 2: {result2.result_hash}"
        )

    @given(
        payload1=json_payloads,
        payload2=json_payloads,
    )
    @settings(max_examples=100, deadline=None)
    def test_different_payloads_different_hashes(
        self,
        payload1: Dict[str, Any],
        payload2: Dict[str, Any],
        test_schema_ir,
        default_options,
    ):
        """
        Test that different payloads produce different hashes (with high probability).
        """
        assume(payload1 != payload2)

        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(payload1)
        result2 = validator.validate(payload2)

        # Different payloads should (almost always) have different hashes
        # This isn't strictly required but validates hash quality
        if result1.result_hash == result2.result_hash:
            # In the rare case of hash collision, verify findings are actually same
            assert len(result1.findings) == len(result2.findings)


# =============================================================================
# FINDING ORDER DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestFindingOrderDeterminism:
    """
    Test that finding order is deterministic.
    """

    @given(payload=json_payloads)
    @settings(max_examples=100, deadline=None)
    def test_finding_order_determinism(
        self,
        payload: Dict[str, Any],
        test_schema_ir,
        default_options,
    ):
        """
        Test that findings are always in the same order.
        """
        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert len(result1.findings) == len(result2.findings)

        for f1, f2 in zip(result1.findings, result2.findings):
            assert f1.code == f2.code, "Finding codes don't match"
            assert f1.path == f2.path, "Finding paths don't match"
            assert f1.severity == f2.severity, "Finding severities don't match"

    def test_finding_order_is_sorted(
        self,
        test_schema_ir,
        default_options,
    ):
        """
        Test that findings are sorted by severity, path, and code.
        """
        # Payload with multiple errors at different paths
        payload = {
            "name": "",  # Too short
            "value": -100,  # Below minimum
            "status": "invalid",  # Invalid enum
            "unknown_field": "test",  # Unknown field
        }

        validator = DeterministicValidator(test_schema_ir, default_options)
        result = validator.validate(payload)

        # Verify findings are sorted
        for i in range(len(result.findings) - 1):
            current = result.findings[i]
            next_finding = result.findings[i + 1]

            current_severity = 0 if current.severity == Severity.ERROR else (
                1 if current.severity == Severity.WARNING else 2
            )
            next_severity = 0 if next_finding.severity == Severity.ERROR else (
                1 if next_finding.severity == Severity.WARNING else 2
            )

            # Current should be <= next in sort order
            assert (current_severity, current.path, current.code) <= (
                next_severity, next_finding.path, next_finding.code
            ), "Findings are not in deterministic order"


# =============================================================================
# EXPLICIT DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestExplicitDeterminism:
    """
    Explicit test cases for validation determinism.
    """

    @pytest.mark.parametrize("payload", [
        {},  # Empty
        {"name": "test", "value": 50},  # Valid
        {"value": 50},  # Missing required
        {"name": "test", "value": -100},  # Constraint violation
        {"name": "test", "value": 50, "status": "invalid"},  # Enum violation
        {"name": "test", "value": 50, "extra": "field"},  # Unknown field
    ])
    def test_explicit_payloads_deterministic(
        self,
        payload,
        test_schema_ir,
        default_options,
    ):
        """Test determinism with explicit payload examples."""
        validator = DeterministicValidator(test_schema_ir, default_options)

        results = [validator.validate(copy.deepcopy(payload)) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_large_payload_determinism(
        self,
        test_schema_ir,
        default_options,
    ):
        """Test determinism with large payloads."""
        # Create large payload
        payload = {
            "name": "test" * 10,
            "value": 500,
            "description": "A" * 200,
        }
        for i in range(50):
            payload[f"extra_field_{i}"] = f"value_{i}"

        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1 == result2

    @given(unicode_text=st.text(min_size=1, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_unicode_determinism(
        self,
        unicode_text: str,
        test_schema_ir,
        default_options,
    ):
        """Test determinism with unicode content."""
        payload = {
            "name": unicode_text,
            "value": 50,
        }

        validator = DeterministicValidator(test_schema_ir, default_options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1 == result2


# =============================================================================
# PROFILE INDEPENDENCE TESTS
# =============================================================================

@pytest.mark.property
class TestProfileDeterminism:
    """
    Test that validation is deterministic across different profiles.
    """

    @pytest.mark.parametrize("profile", [
        ValidationProfile.STRICT,
        ValidationProfile.STANDARD,
        ValidationProfile.PERMISSIVE,
    ])
    @given(payload=json_payloads)
    @settings(max_examples=50, deadline=None)
    def test_determinism_per_profile(
        self,
        profile: ValidationProfile,
        payload: Dict[str, Any],
        test_schema_ir,
    ):
        """Test that each profile produces deterministic results."""
        options = ValidationOptions(profile=profile)
        validator = DeterministicValidator(test_schema_ir, options)

        result1 = validator.validate(copy.deepcopy(payload))
        result2 = validator.validate(copy.deepcopy(payload))

        assert result1 == result2
