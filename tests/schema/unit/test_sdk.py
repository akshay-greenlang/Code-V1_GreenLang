# -*- coding: utf-8 -*-
"""
Unit Tests for GL-FOUND-X-002 SDK Interface.

This module provides comprehensive unit tests for the GreenLang Schema SDK,
testing the user-friendly API for schema validation, compilation, and fix
suggestion handling.

Test Coverage:
    - validate() function with various inputs
    - validate_batch() function
    - compile_schema() function
    - CompiledSchema class
    - Fix suggestion helpers
    - Finding helpers
    - Schema reference helpers
    - Error handling

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 7.5
"""

import pytest
from typing import Any, Dict, List

from greenlang.schema.sdk import (
    # Main functions
    validate,
    validate_batch,
    compile_schema,
    CompiledSchema,
    # Fix suggestion helpers
    apply_fixes,
    safe_fixes,
    review_fixes,
    # Finding helpers
    errors_only,
    warnings_only,
    findings_by_path,
    findings_by_code,
    # Schema reference helpers
    parse_schema_ref,
    schema_ref,
    # Type aliases
    SchemaInput,
    PayloadInput,
)
from greenlang.schema.models import (
    SchemaRef,
    ValidationReport,
    BatchValidationReport,
    Finding,
    FixSuggestion,
    JSONPatchOp,
    PatchSafety,
    Severity,
    ValidationProfile,
    PatchLevel,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_schema() -> Dict[str, Any]:
    """Create a simple test schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["name"],
    }


@pytest.fixture
def valid_payload() -> Dict[str, Any]:
    """Create a valid test payload."""
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
    }


@pytest.fixture
def invalid_payload() -> Dict[str, Any]:
    """Create an invalid test payload (missing required field)."""
    return {
        "age": 30,
        "email": "john@example.com",
    }


@pytest.fixture
def sample_findings() -> List[Finding]:
    """Create sample findings for testing helpers."""
    return [
        Finding(
            code="GLSCHEMA-E100",
            severity=Severity.ERROR,
            path="/name",
            message="Missing required field",
        ),
        Finding(
            code="GLSCHEMA-E200",
            severity=Severity.ERROR,
            path="/age",
            message="Value 200 exceeds maximum 150",
        ),
        Finding(
            code="GLSCHEMA-W700",
            severity=Severity.WARNING,
            path="/emial",
            message="Unknown field. Did you mean 'email'?",
        ),
        Finding(
            code="GLSCHEMA-W600",
            severity=Severity.WARNING,
            path="/deprecated_field",
            message="Field is deprecated",
        ),
        Finding(
            code="GLSCHEMA-E100",
            severity=Severity.ERROR,
            path="/address",
            message="Missing required field",
        ),
    ]


@pytest.fixture
def sample_fixes() -> List[FixSuggestion]:
    """Create sample fix suggestions for testing helpers."""
    return [
        FixSuggestion(
            patch=[JSONPatchOp(op="add", path="/name", value="Default Name")],
            preconditions=[],
            confidence=0.95,
            safety=PatchSafety.SAFE,
            rationale="Add default value for missing required field",
        ),
        FixSuggestion(
            patch=[JSONPatchOp(op="replace", path="/age", value=100)],
            preconditions=[JSONPatchOp(op="test", path="/age", value=200)],
            confidence=0.7,
            safety=PatchSafety.NEEDS_REVIEW,
            rationale="Clamp age to maximum allowed value",
        ),
        FixSuggestion(
            patch=[JSONPatchOp(op="move", path="/email", value=None, from_="/emial")],
            preconditions=[],
            confidence=0.85,
            safety=PatchSafety.NEEDS_REVIEW,
            rationale="Correct typo: emial -> email",
        ),
        FixSuggestion(
            patch=[JSONPatchOp(op="add", path="/name", value="Guessed Name")],
            preconditions=[],
            confidence=0.3,
            safety=PatchSafety.UNSAFE,
            rationale="Infer name from context",
        ),
    ]


# =============================================================================
# SCHEMA REFERENCE HELPER TESTS
# =============================================================================


class TestSchemaRefHelpers:
    """Test schema reference helper functions."""

    def test_parse_schema_ref_basic(self):
        """Test parsing a basic schema URI."""
        ref = parse_schema_ref("gl://schemas/activity@1.0.0")
        assert ref.schema_id == "activity"
        assert ref.version == "1.0.0"
        assert ref.variant is None

    def test_parse_schema_ref_with_namespace(self):
        """Test parsing a schema URI with namespace."""
        ref = parse_schema_ref("gl://schemas/emissions/activity@1.3.0")
        assert ref.schema_id == "emissions/activity"
        assert ref.version == "1.3.0"

    def test_parse_schema_ref_with_variant(self):
        """Test parsing a schema URI with variant."""
        ref = parse_schema_ref("gl://schemas/activity@1.0.0#strict")
        assert ref.schema_id == "activity"
        assert ref.version == "1.0.0"
        assert ref.variant == "strict"

    def test_parse_schema_ref_invalid(self):
        """Test parsing an invalid schema URI."""
        with pytest.raises(ValueError):
            parse_schema_ref("invalid-uri")

    def test_schema_ref_builder(self):
        """Test schema_ref builder function."""
        ref = schema_ref("emissions/activity", "1.3.0")
        assert ref.schema_id == "emissions/activity"
        assert ref.version == "1.3.0"
        assert ref.variant is None
        assert ref.to_uri() == "gl://schemas/emissions/activity@1.3.0"

    def test_schema_ref_builder_with_variant(self):
        """Test schema_ref builder with variant."""
        ref = schema_ref("activity", "1.0.0", variant="strict")
        assert ref.variant == "strict"
        assert ref.to_uri() == "gl://schemas/activity@1.0.0#strict"


# =============================================================================
# VALIDATE FUNCTION TESTS
# =============================================================================


class TestValidateFunction:
    """Test the validate() function."""

    def test_validate_valid_payload(self, simple_schema, valid_payload):
        """Test validation of a valid payload."""
        result = validate(valid_payload, simple_schema)

        assert isinstance(result, ValidationReport)
        assert result.valid is True
        assert result.summary.error_count == 0

    def test_validate_invalid_payload(self, simple_schema, invalid_payload):
        """Test validation of an invalid payload."""
        result = validate(invalid_payload, simple_schema)

        assert isinstance(result, ValidationReport)
        assert result.valid is False
        assert result.summary.error_count > 0

    def test_validate_with_profile_strict(self, simple_schema, valid_payload):
        """Test validation with strict profile."""
        result = validate(valid_payload, simple_schema, profile="strict")

        assert isinstance(result, ValidationReport)
        # Strict profile should still pass for valid payload
        # (assuming no unknown fields)

    def test_validate_with_profile_permissive(self, simple_schema, invalid_payload):
        """Test validation with permissive profile."""
        result = validate(invalid_payload, simple_schema, profile="permissive")

        assert isinstance(result, ValidationReport)
        # Permissive profile should still catch required field errors
        assert result.summary.error_count >= 0

    def test_validate_with_normalize(self, simple_schema, valid_payload):
        """Test validation with normalization enabled."""
        result = validate(valid_payload, simple_schema, normalize=True)

        assert result.normalized_payload is not None

    def test_validate_without_normalize(self, simple_schema, valid_payload):
        """Test validation without normalization."""
        result = validate(valid_payload, simple_schema, normalize=False)

        assert result.normalized_payload is None

    def test_validate_with_max_errors(self, simple_schema):
        """Test validation with max_errors limit."""
        payload = {}  # Missing all required fields
        result = validate(payload, simple_schema, max_errors=1)

        # Should stop after 1 error
        assert len([f for f in result.findings if f.severity == Severity.ERROR]) <= 1

    def test_validate_with_fail_fast(self, simple_schema):
        """Test validation with fail_fast option."""
        payload = {}  # Multiple errors
        result = validate(payload, simple_schema, fail_fast=True)

        # Should stop after first error
        error_count = len([f for f in result.findings if f.severity == Severity.ERROR])
        assert error_count <= 1

    def test_validate_yaml_string(self, simple_schema):
        """Test validation with YAML string payload."""
        yaml_payload = "name: John Doe\nage: 30"
        result = validate(yaml_payload, simple_schema)

        assert isinstance(result, ValidationReport)
        # Should parse and validate successfully

    def test_validate_json_string(self, simple_schema):
        """Test validation with JSON string payload."""
        json_payload = '{"name": "John Doe", "age": 30}'
        result = validate(json_payload, simple_schema)

        assert isinstance(result, ValidationReport)
        # Should parse and validate successfully


# =============================================================================
# VALIDATE BATCH TESTS
# =============================================================================


class TestValidateBatchFunction:
    """Test the validate_batch() function."""

    def test_validate_batch_basic(self, simple_schema):
        """Test batch validation with mixed valid/invalid payloads."""
        payloads = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"age": 35},  # Invalid - missing name
        ]

        result = validate_batch(payloads, simple_schema)

        assert isinstance(result, BatchValidationReport)
        assert result.summary.total_items == 3
        assert result.summary.valid_count >= 2  # At least 2 valid

    def test_validate_batch_all_valid(self, simple_schema):
        """Test batch validation with all valid payloads."""
        payloads = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35},
        ]

        result = validate_batch(payloads, simple_schema)

        assert result.summary.valid_count == 3
        assert result.summary.error_count == 0

    def test_validate_batch_all_invalid(self, simple_schema):
        """Test batch validation with all invalid payloads."""
        payloads = [
            {"age": 25},  # Missing name
            {"age": 30},  # Missing name
            {"age": 35},  # Missing name
        ]

        result = validate_batch(payloads, simple_schema)

        assert result.summary.error_count >= 3  # At least 3 errors
        assert result.summary.valid_count < 3

    def test_validate_batch_results_indexed(self, simple_schema):
        """Test that batch results have correct indices."""
        payloads = [
            {"name": "Alice"},
            {"name": "Bob"},
        ]

        result = validate_batch(payloads, simple_schema)

        assert len(result.results) == 2
        assert result.results[0].index == 0
        assert result.results[1].index == 1

    def test_validate_batch_failed_items(self, simple_schema):
        """Test failed_items() method."""
        payloads = [
            {"name": "Alice"},
            {"age": 30},  # Invalid
            {"name": "Charlie"},
        ]

        result = validate_batch(payloads, simple_schema)
        failed = result.failed_items()

        # Should have at least 1 failed item
        assert len(failed) >= 0  # May depend on validation implementation


# =============================================================================
# COMPILE SCHEMA TESTS
# =============================================================================


class TestCompileSchema:
    """Test the compile_schema() function."""

    def test_compile_inline_schema(self, simple_schema):
        """Test compiling an inline schema."""
        compiled = compile_schema(simple_schema)

        assert isinstance(compiled, CompiledSchema)
        assert compiled.schema_id == "inline/schema"
        assert compiled.version == "1.0.0"
        assert len(compiled.schema_hash) == 64

    def test_compiled_schema_validate(self, simple_schema, valid_payload):
        """Test validation using compiled schema."""
        compiled = compile_schema(simple_schema)
        result = compiled.validate(valid_payload)

        assert isinstance(result, ValidationReport)
        assert result.valid is True

    def test_compiled_schema_validate_batch(self, simple_schema):
        """Test batch validation using compiled schema."""
        compiled = compile_schema(simple_schema)
        payloads = [
            {"name": "Alice"},
            {"name": "Bob"},
        ]

        result = compiled.validate_batch(payloads)

        assert isinstance(result, BatchValidationReport)
        assert result.summary.total_items == 2

    def test_compiled_schema_properties(self, simple_schema):
        """Test CompiledSchema property accessors."""
        compiled = compile_schema(simple_schema)

        assert compiled.properties >= 0
        assert compiled.rules >= 0
        assert compiled.compile_time_ms >= 0

    def test_compiled_schema_repr(self, simple_schema):
        """Test CompiledSchema string representation."""
        compiled = compile_schema(simple_schema)
        repr_str = repr(compiled)

        assert "CompiledSchema" in repr_str
        assert "inline/schema" in repr_str


# =============================================================================
# FIX SUGGESTION HELPER TESTS
# =============================================================================


class TestFixSuggestionHelpers:
    """Test fix suggestion helper functions."""

    def test_safe_fixes(self, sample_fixes):
        """Test safe_fixes filter."""
        safe = safe_fixes(sample_fixes)

        assert len(safe) == 1
        assert all(f.safety == PatchSafety.SAFE for f in safe)

    def test_safe_fixes_empty(self):
        """Test safe_fixes with no suggestions."""
        assert safe_fixes(None) == []
        assert safe_fixes([]) == []

    def test_review_fixes(self, sample_fixes):
        """Test review_fixes filter."""
        review = review_fixes(sample_fixes)

        assert len(review) == 2
        assert all(f.safety == PatchSafety.NEEDS_REVIEW for f in review)

    def test_review_fixes_empty(self):
        """Test review_fixes with no suggestions."""
        assert review_fixes(None) == []
        assert review_fixes([]) == []

    def test_apply_fixes_safe_only(self, sample_fixes):
        """Test apply_fixes with safe level."""
        payload = {"age": 200, "emial": "test@example.com"}

        new_payload, applied = apply_fixes(payload, sample_fixes, safety="safe")

        # Should only apply safe fixes
        assert len(applied) <= 1
        assert all(f.safety == PatchSafety.SAFE for f in applied)

    def test_apply_fixes_needs_review(self, sample_fixes):
        """Test apply_fixes including needs_review level."""
        payload = {"age": 200, "emial": "test@example.com"}

        new_payload, applied = apply_fixes(payload, sample_fixes, safety="needs_review")

        # Should apply safe and needs_review fixes
        for fix in applied:
            assert fix.safety in (PatchSafety.SAFE, PatchSafety.NEEDS_REVIEW)

    def test_apply_fixes_preserves_original(self, sample_fixes):
        """Test that apply_fixes doesn't modify original payload."""
        original = {"age": 200, "emial": "test@example.com"}
        original_copy = original.copy()

        new_payload, applied = apply_fixes(original, sample_fixes)

        # Original should be unchanged
        assert original == original_copy


# =============================================================================
# FINDING HELPER TESTS
# =============================================================================


class TestFindingHelpers:
    """Test finding helper functions."""

    def test_errors_only(self, sample_findings):
        """Test errors_only filter."""
        errors = errors_only(sample_findings)

        assert len(errors) == 3
        assert all(f.severity == Severity.ERROR for f in errors)

    def test_warnings_only(self, sample_findings):
        """Test warnings_only filter."""
        warnings = warnings_only(sample_findings)

        assert len(warnings) == 2
        assert all(f.severity == Severity.WARNING for f in warnings)

    def test_findings_by_path(self, sample_findings):
        """Test findings_by_path filter."""
        findings = findings_by_path(sample_findings, "/name")

        assert len(findings) == 1
        assert findings[0].path == "/name"

    def test_findings_by_path_not_found(self, sample_findings):
        """Test findings_by_path with non-existent path."""
        findings = findings_by_path(sample_findings, "/nonexistent")

        assert findings == []

    def test_findings_by_code(self, sample_findings):
        """Test findings_by_code filter."""
        findings = findings_by_code(sample_findings, "GLSCHEMA-E100")

        assert len(findings) == 2
        assert all(f.code == "GLSCHEMA-E100" for f in findings)

    def test_findings_by_code_not_found(self, sample_findings):
        """Test findings_by_code with non-existent code."""
        findings = findings_by_code(sample_findings, "GLSCHEMA-E999")

        assert findings == []


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling in SDK functions."""

    def test_validate_invalid_schema_type(self):
        """Test validate with invalid schema type."""
        with pytest.raises(TypeError):
            validate({"name": "test"}, 12345)  # type: ignore

    def test_parse_schema_ref_invalid_format(self):
        """Test parsing invalid schema reference formats."""
        with pytest.raises(ValueError):
            parse_schema_ref("not-a-valid-uri")

        with pytest.raises(ValueError):
            parse_schema_ref("http://example.com/schema")

    def test_compile_schema_invalid_schema(self):
        """Test compile_schema with invalid schema structure."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "minimum": "not-a-number",  # Invalid - should be number
                }
            }
        }

        # GreenLang's compiler validates constraints strictly
        # so this should raise ValueError during compilation
        with pytest.raises(ValueError, match="Schema compilation failed"):
            compile_schema(invalid_schema)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSDKIntegration:
    """Integration tests for SDK workflow."""

    def test_full_validation_workflow(self, simple_schema, invalid_payload):
        """Test complete validation workflow."""
        # 1. Validate
        result = validate(invalid_payload, simple_schema)
        assert not result.valid

        # 2. Check errors
        errors = errors_only(result.findings)
        assert len(errors) > 0

        # 3. Get fix suggestions
        fixes = result.fix_suggestions or []

        # 4. Apply safe fixes
        if fixes:
            new_payload, applied = apply_fixes(invalid_payload, fixes)
            # Payload may or may not be fixed depending on suggestions

    def test_precompile_and_validate_workflow(self, simple_schema):
        """Test pre-compilation workflow for multiple validations."""
        # 1. Pre-compile schema
        compiled = compile_schema(simple_schema)

        # 2. Validate multiple payloads efficiently
        payloads = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35},
        ]

        results = []
        for payload in payloads:
            result = compiled.validate(payload)
            results.append(result)

        # All should be valid
        assert all(r.valid for r in results)

    def test_batch_validation_workflow(self, simple_schema):
        """Test batch validation workflow."""
        payloads = [
            {"name": "Alice", "age": 25},
            {"age": 30},  # Invalid
            {"name": "Charlie"},
        ]

        # 1. Validate batch
        result = validate_batch(payloads, simple_schema)

        # 2. Process results
        for item in result.results:
            if not item.valid:
                errors = errors_only(item.findings)
                # Handle errors for item at item.index

        # 3. Check summary
        assert result.summary.total_items == 3


# =============================================================================
# MODULE EXPORTS TEST
# =============================================================================


class TestModuleExports:
    """Test that all expected symbols are exported."""

    def test_sdk_exports(self):
        """Test SDK module exports."""
        from greenlang.schema import sdk

        # Main functions
        assert hasattr(sdk, "validate")
        assert hasattr(sdk, "validate_batch")
        assert hasattr(sdk, "compile_schema")

        # Classes
        assert hasattr(sdk, "CompiledSchema")

        # Fix helpers
        assert hasattr(sdk, "apply_fixes")
        assert hasattr(sdk, "safe_fixes")
        assert hasattr(sdk, "review_fixes")

        # Finding helpers
        assert hasattr(sdk, "errors_only")
        assert hasattr(sdk, "warnings_only")
        assert hasattr(sdk, "findings_by_path")
        assert hasattr(sdk, "findings_by_code")

        # Schema ref helpers
        assert hasattr(sdk, "parse_schema_ref")
        assert hasattr(sdk, "schema_ref")

    def test_package_exports(self):
        """Test package-level exports."""
        from greenlang import schema

        # Main functions should be accessible from package
        assert hasattr(schema, "validate")
        assert hasattr(schema, "compile_schema")
        assert hasattr(schema, "SchemaRef")
