# -*- coding: utf-8 -*-
"""
Unit tests for GL-FOUND-X-002 Structural Validator.

This module tests the StructuralValidator class which validates payload
structure against compiled schema IR.

Tests cover:
    - Required field validation (GLSCHEMA-E100)
    - Unknown field handling (GLSCHEMA-E101)
    - Type validation (GLSCHEMA-E102)
    - Null handling (GLSCHEMA-E103)
    - Property count constraints (GLSCHEMA-E105)
    - Validation profile behavior (strict/standard/permissive)
    - Nested object validation
    - Array validation
    - Edge cases and error handling

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.1
"""

from datetime import datetime
from typing import Any, Dict

import pytest

from greenlang.schema.compiler.ir import PropertyIR, SchemaIR
from greenlang.schema.errors import ErrorCode
from greenlang.schema.models.config import (
    CoercionPolicy,
    PatchLevel,
    UnknownFieldPolicy,
    ValidationOptions,
    ValidationProfile,
)
from greenlang.schema.models.finding import Severity
from greenlang.schema.validator.structural import (
    PYTHON_TO_JSON_TYPE,
    TYPE_COMPATIBILITY,
    StructuralValidator,
    validate_structure,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_hash() -> str:
    """Generate a valid 64-character SHA-256 hash for testing."""
    return "a" * 64


@pytest.fixture
def basic_ir(sample_schema_hash) -> SchemaIR:
    """Create a basic SchemaIR for testing with simple properties."""
    return SchemaIR(
        schema_id="test/basic",
        version="1.0.0",
        schema_hash=sample_schema_hash,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/value": PropertyIR(path="/value", type="number", required=True),
            "/description": PropertyIR(path="/description", type="string", required=False),
        },
        required_paths={"/name", "/value"},
    )


@pytest.fixture
def nested_ir(sample_schema_hash) -> SchemaIR:
    """Create SchemaIR with nested object structure."""
    return SchemaIR(
        schema_id="test/nested",
        version="1.0.0",
        schema_hash=sample_schema_hash,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/metadata": PropertyIR(path="/metadata", type="object", required=True),
            "/metadata/created_at": PropertyIR(path="/metadata/created_at", type="string", required=True),
            "/metadata/version": PropertyIR(path="/metadata/version", type="integer", required=False),
        },
        required_paths={"/name", "/metadata", "/metadata/created_at"},
    )


@pytest.fixture
def array_ir(sample_schema_hash) -> SchemaIR:
    """Create SchemaIR with array properties."""
    return SchemaIR(
        schema_id="test/array",
        version="1.0.0",
        schema_hash=sample_schema_hash,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/items": PropertyIR(
                path="/items",
                type="array",
                required=True,
                gl_extensions={"items_type": "object"},
            ),
            "/tags": PropertyIR(
                path="/tags",
                type="array",
                required=False,
                gl_extensions={"items_type": "string"},
            ),
        },
        required_paths={"/items"},
    )


@pytest.fixture
def property_count_ir(sample_schema_hash) -> SchemaIR:
    """Create SchemaIR with property count constraints."""
    return SchemaIR(
        schema_id="test/property_count",
        version="1.0.0",
        schema_hash=sample_schema_hash,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/data": PropertyIR(
                path="/data",
                type="object",
                required=True,
                gl_extensions={"minProperties": 2, "maxProperties": 5},
            ),
        },
        required_paths={"/data"},
    )


@pytest.fixture
def default_options() -> ValidationOptions:
    """Create default validation options (standard profile)."""
    return ValidationOptions()


@pytest.fixture
def strict_options() -> ValidationOptions:
    """Create strict validation options."""
    return ValidationOptions.strict()


@pytest.fixture
def permissive_options() -> ValidationOptions:
    """Create permissive validation options."""
    return ValidationOptions.permissive()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_finding_by_code(findings, code: str):
    """Find first finding with given error code."""
    for f in findings:
        if f.code == code:
            return f
    return None


def find_findings_by_code(findings, code: str):
    """Find all findings with given error code."""
    return [f for f in findings if f.code == code]


def count_errors(findings):
    """Count error-level findings."""
    return sum(1 for f in findings if f.severity == Severity.ERROR)


def count_warnings(findings):
    """Count warning-level findings."""
    return sum(1 for f in findings if f.severity == Severity.WARNING)


# =============================================================================
# TEST: INITIALIZATION
# =============================================================================


class TestStructuralValidatorInitialization:
    """Tests for StructuralValidator initialization."""

    def test_init_with_valid_args(self, basic_ir, default_options):
        """Test initialization with valid arguments."""
        validator = StructuralValidator(basic_ir, default_options)
        assert validator.ir == basic_ir
        assert validator.options == default_options

    def test_init_with_none_ir_raises(self, default_options):
        """Test that None IR raises ValueError."""
        with pytest.raises(ValueError, match="SchemaIR cannot be None"):
            StructuralValidator(None, default_options)

    def test_init_with_none_options_raises(self, basic_ir):
        """Test that None options raises ValueError."""
        with pytest.raises(ValueError, match="ValidationOptions cannot be None"):
            StructuralValidator(basic_ir, None)


# =============================================================================
# TEST: REQUIRED FIELD VALIDATION (GLSCHEMA-E100)
# =============================================================================


class TestRequiredFieldValidation:
    """Tests for required field validation (GLSCHEMA-E100)."""

    def test_valid_payload_with_all_required(self, basic_ir, default_options):
        """Test valid payload with all required fields present."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": 42}
        findings = validator.validate(payload)

        # Should have no errors for required fields
        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 0

    def test_missing_required_field(self, basic_ir, default_options):
        """Test detection of missing required field."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test"}  # Missing 'value'
        findings = validator.validate(payload)

        # Should have one missing required error
        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 1
        assert missing_errors[0].path == "/value"
        assert missing_errors[0].severity == Severity.ERROR

    def test_multiple_missing_required_fields(self, basic_ir, default_options):
        """Test detection of multiple missing required fields."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {}  # Missing both 'name' and 'value'
        findings = validator.validate(payload)

        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 2

        paths = {e.path for e in missing_errors}
        assert "/name" in paths
        assert "/value" in paths

    def test_nested_required_field(self, nested_ir, default_options):
        """Test validation of nested required fields."""
        validator = StructuralValidator(nested_ir, default_options)
        payload = {
            "name": "test",
            "metadata": {}  # Missing nested 'created_at'
        }
        findings = validator.validate(payload)

        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 1
        assert missing_errors[0].path == "/metadata/created_at"

    def test_optional_field_not_required(self, basic_ir, default_options):
        """Test that optional fields don't generate errors when missing."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": 42}  # 'description' is optional
        findings = validator.validate(payload)

        # Should have no missing required errors
        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 0


# =============================================================================
# TEST: TYPE VALIDATION (GLSCHEMA-E102)
# =============================================================================


class TestTypeValidation:
    """Tests for type validation (GLSCHEMA-E102)."""

    def test_valid_string_type(self, basic_ir, default_options):
        """Test valid string type."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test_string", "value": 42}
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        # Filter for /name path
        name_errors = [e for e in type_errors if e.path == "/name"]
        assert len(name_errors) == 0

    def test_invalid_string_type_number_provided(self, basic_ir, default_options):
        """Test detection of type mismatch: number instead of string."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": 123, "value": 42}  # name should be string
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        name_errors = [e for e in type_errors if e.path == "/name"]
        assert len(name_errors) == 1
        assert "integer" in name_errors[0].actual.lower()

    def test_valid_number_type_accepts_integer(self, basic_ir, default_options):
        """Test that number type accepts integer values."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": 42}  # integer for number field
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        value_errors = [e for e in type_errors if e.path == "/value"]
        assert len(value_errors) == 0

    def test_valid_number_type_accepts_float(self, basic_ir, default_options):
        """Test that number type accepts float values."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": 3.14}
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        value_errors = [e for e in type_errors if e.path == "/value"]
        assert len(value_errors) == 0

    def test_invalid_number_type_string_provided(self, basic_ir, default_options):
        """Test detection of type mismatch: string instead of number."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": "not a number"}
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        value_errors = [e for e in type_errors if e.path == "/value"]
        assert len(value_errors) == 1

    def test_boolean_type_validation(self, sample_schema_hash, default_options):
        """Test boolean type validation."""
        ir = SchemaIR(
            schema_id="test/bool",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={"/active": PropertyIR(path="/active", type="boolean", required=True)},
            required_paths={"/active"},
        )
        validator = StructuralValidator(ir, default_options)

        # Valid boolean
        findings = validator.validate({"active": True})
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        assert len(type_errors) == 0

        # Invalid - string "true" is not boolean
        findings = validator.validate({"active": "true"})
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        assert len(type_errors) == 1

    def test_array_type_validation(self, array_ir, default_options):
        """Test array type validation."""
        validator = StructuralValidator(array_ir, default_options)

        # Valid array
        findings = validator.validate({"items": [{"a": 1}]})
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        items_errors = [e for e in type_errors if e.path == "/items"]
        assert len(items_errors) == 0

        # Invalid - object instead of array
        findings = validator.validate({"items": {"not": "array"}})
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        items_errors = [e for e in type_errors if e.path == "/items"]
        assert len(items_errors) == 1

    def test_object_type_validation(self, nested_ir, default_options):
        """Test object type validation."""
        validator = StructuralValidator(nested_ir, default_options)

        # Valid object
        findings = validator.validate({
            "name": "test",
            "metadata": {"created_at": "2024-01-01"}
        })
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        metadata_errors = [e for e in type_errors if e.path == "/metadata"]
        assert len(metadata_errors) == 0

        # Invalid - string instead of object
        findings = validator.validate({
            "name": "test",
            "metadata": "not an object"
        })
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        metadata_errors = [e for e in type_errors if e.path == "/metadata"]
        assert len(metadata_errors) == 1


# =============================================================================
# TEST: NULL HANDLING (GLSCHEMA-E103)
# =============================================================================


class TestNullHandling:
    """Tests for null value handling (GLSCHEMA-E103)."""

    def test_null_value_not_allowed(self, basic_ir, default_options):
        """Test detection of null value when not allowed."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": None, "value": 42}
        findings = validator.validate(payload)

        null_errors = find_findings_by_code(findings, ErrorCode.INVALID_NULL.value)
        assert len(null_errors) == 1
        assert null_errors[0].path == "/name"

    def test_null_type_explicitly_allowed(self, sample_schema_hash, default_options):
        """Test that null is allowed when type includes 'null'."""
        ir = SchemaIR(
            schema_id="test/nullable",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={"/nullable_field": PropertyIR(path="/nullable_field", type="null", required=True)},
            required_paths={"/nullable_field"},
        )
        validator = StructuralValidator(ir, default_options)

        findings = validator.validate({"nullable_field": None})
        null_errors = find_findings_by_code(findings, ErrorCode.INVALID_NULL.value)
        assert len(null_errors) == 0

    def test_null_in_array_items(self, sample_schema_hash, default_options):
        """Test null handling in array items."""
        ir = SchemaIR(
            schema_id="test/array_null",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={
                "/items": PropertyIR(
                    path="/items",
                    type="array",
                    required=True,
                    gl_extensions={"items_type": "string"},
                )
            },
            required_paths={"/items"},
        )
        validator = StructuralValidator(ir, default_options)

        findings = validator.validate({"items": ["a", None, "b"]})
        null_errors = find_findings_by_code(findings, ErrorCode.INVALID_NULL.value)
        assert len(null_errors) == 1
        assert "/items/1" in null_errors[0].path


# =============================================================================
# TEST: ADDITIONAL PROPERTIES / UNKNOWN FIELDS (GLSCHEMA-E101)
# =============================================================================


class TestUnknownFieldHandling:
    """Tests for unknown field handling (GLSCHEMA-E101)."""

    def test_unknown_field_warning_standard_profile(self, basic_ir, default_options):
        """Test unknown field generates warning in standard profile."""
        validator = StructuralValidator(basic_ir, default_options)
        payload = {"name": "test", "value": 42, "unknown_field": "extra"}
        findings = validator.validate(payload)

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert unknown_findings[0].severity == Severity.WARNING
        assert unknown_findings[0].path == "/unknown_field"

    def test_unknown_field_error_strict_profile(self, basic_ir, strict_options):
        """Test unknown field generates error in strict profile."""
        validator = StructuralValidator(basic_ir, strict_options)
        payload = {"name": "test", "value": 42, "unknown_field": "extra"}
        findings = validator.validate(payload)

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert unknown_findings[0].severity == Severity.ERROR

    def test_unknown_field_ignored_permissive_profile(self, basic_ir, permissive_options):
        """Test unknown field is ignored in permissive profile."""
        validator = StructuralValidator(basic_ir, permissive_options)
        payload = {"name": "test", "value": 42, "unknown_field": "extra"}
        findings = validator.validate(payload)

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 0

    def test_multiple_unknown_fields(self, basic_ir, strict_options):
        """Test detection of multiple unknown fields."""
        validator = StructuralValidator(basic_ir, strict_options)
        payload = {
            "name": "test",
            "value": 42,
            "unknown1": "a",
            "unknown2": "b",
            "unknown3": "c"
        }
        findings = validator.validate(payload)

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 3

    def test_unknown_field_policy_override(self, basic_ir):
        """Test that unknown_field_policy overrides default behavior."""
        # Standard profile with error policy
        options = ValidationOptions(
            profile=ValidationProfile.STANDARD,
            unknown_field_policy=UnknownFieldPolicy.ERROR
        )
        validator = StructuralValidator(basic_ir, options)
        payload = {"name": "test", "value": 42, "unknown": "x"}
        findings = validator.validate(payload)

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert unknown_findings[0].severity == Severity.ERROR


# =============================================================================
# TEST: PROPERTY COUNT CONSTRAINTS (GLSCHEMA-E105)
# =============================================================================


class TestPropertyCountConstraints:
    """Tests for property count validation (GLSCHEMA-E105)."""

    def test_valid_property_count(self, property_count_ir, default_options):
        """Test valid property count within bounds."""
        validator = StructuralValidator(property_count_ir, default_options)
        payload = {"data": {"a": 1, "b": 2, "c": 3}}  # 3 properties, within [2,5]
        findings = validator.validate(payload)

        count_errors = find_findings_by_code(findings, ErrorCode.PROPERTY_COUNT_VIOLATION.value)
        assert len(count_errors) == 0

    def test_min_properties_violation(self, property_count_ir, default_options):
        """Test detection of too few properties."""
        validator = StructuralValidator(property_count_ir, default_options)
        payload = {"data": {"a": 1}}  # Only 1 property, minimum is 2
        findings = validator.validate(payload)

        count_errors = find_findings_by_code(findings, ErrorCode.PROPERTY_COUNT_VIOLATION.value)
        assert len(count_errors) == 1
        assert "minimum" in count_errors[0].message.lower()

    def test_max_properties_violation(self, property_count_ir, default_options):
        """Test detection of too many properties."""
        validator = StructuralValidator(property_count_ir, default_options)
        payload = {"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}  # 6 > 5
        findings = validator.validate(payload)

        count_errors = find_findings_by_code(findings, ErrorCode.PROPERTY_COUNT_VIOLATION.value)
        assert len(count_errors) == 1
        assert "maximum" in count_errors[0].message.lower()


# =============================================================================
# TEST: NESTED OBJECT VALIDATION
# =============================================================================


class TestNestedObjectValidation:
    """Tests for nested object validation."""

    def test_valid_nested_object(self, nested_ir, default_options):
        """Test validation of valid nested object."""
        validator = StructuralValidator(nested_ir, default_options)
        payload = {
            "name": "test",
            "metadata": {
                "created_at": "2024-01-01",
                "version": 1
            }
        }
        findings = validator.validate(payload)

        # Should have no errors
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_nested_required_field_missing(self, nested_ir, default_options):
        """Test detection of missing nested required field."""
        validator = StructuralValidator(nested_ir, default_options)
        payload = {
            "name": "test",
            "metadata": {}  # Missing created_at
        }
        findings = validator.validate(payload)

        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        nested_errors = [e for e in missing_errors if "/metadata/" in e.path]
        assert len(nested_errors) == 1

    def test_nested_type_mismatch(self, nested_ir, default_options):
        """Test detection of type mismatch in nested field."""
        validator = StructuralValidator(nested_ir, default_options)
        payload = {
            "name": "test",
            "metadata": {
                "created_at": "2024-01-01",
                "version": "not_an_integer"
            }
        }
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        version_errors = [e for e in type_errors if "/metadata/version" in e.path]
        assert len(version_errors) == 1


# =============================================================================
# TEST: ARRAY VALIDATION
# =============================================================================


class TestArrayValidation:
    """Tests for array validation."""

    def test_valid_array_of_objects(self, array_ir, default_options):
        """Test valid array of objects."""
        validator = StructuralValidator(array_ir, default_options)
        payload = {
            "items": [
                {"id": 1},
                {"id": 2}
            ]
        }
        findings = validator.validate(payload)

        # Should have no type errors for array items
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        items_errors = [e for e in type_errors if "/items/" in e.path]
        assert len(items_errors) == 0

    def test_array_with_wrong_item_type(self, array_ir, default_options):
        """Test array with wrong item type."""
        validator = StructuralValidator(array_ir, default_options)
        payload = {
            "items": [
                "string_not_object",
                {"valid": "object"}
            ]
        }
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        items_errors = [e for e in type_errors if "/items/0" in e.path]
        assert len(items_errors) == 1

    def test_empty_array_valid(self, array_ir, default_options):
        """Test that empty array is valid when array is required."""
        validator = StructuralValidator(array_ir, default_options)
        payload = {"items": []}
        findings = validator.validate(payload)

        # Empty array should not cause type error
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        items_errors = [e for e in type_errors if e.path == "/items"]
        assert len(items_errors) == 0


# =============================================================================
# TEST: VALIDATION OPTIONS
# =============================================================================


class TestValidationOptions:
    """Tests for validation options behavior."""

    def test_fail_fast_stops_on_first_error(self, basic_ir):
        """Test fail_fast option stops validation on first error."""
        options = ValidationOptions(fail_fast=True)
        validator = StructuralValidator(basic_ir, options)
        payload = {}  # Missing both required fields
        findings = validator.validate(payload)

        # Should have only one error due to fail_fast
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 1

    def test_max_errors_limit(self, sample_schema_hash):
        """Test max_errors option limits number of errors."""
        # Create IR with many required fields
        properties = {f"/field{i}": PropertyIR(path=f"/field{i}", type="string", required=True)
                     for i in range(10)}
        required_paths = {f"/field{i}" for i in range(10)}

        ir = SchemaIR(
            schema_id="test/many_fields",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties=properties,
            required_paths=required_paths,
        )

        options = ValidationOptions(max_errors=3)
        validator = StructuralValidator(ir, options)
        payload = {}  # All fields missing
        findings = validator.validate(payload)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) <= 3


# =============================================================================
# TEST: ROOT PAYLOAD VALIDATION
# =============================================================================


class TestRootPayloadValidation:
    """Tests for root payload validation."""

    def test_root_must_be_object(self, basic_ir, default_options):
        """Test that root payload must be an object."""
        validator = StructuralValidator(basic_ir, default_options)

        # List is not a valid root
        findings = validator.validate([1, 2, 3])  # type: ignore
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        assert len(type_errors) == 1
        assert "object" in type_errors[0].message.lower()

    def test_root_string_rejected(self, basic_ir, default_options):
        """Test that string is rejected as root."""
        validator = StructuralValidator(basic_ir, default_options)

        findings = validator.validate("not an object")  # type: ignore
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        assert len(type_errors) == 1

    def test_root_null_rejected(self, basic_ir, default_options):
        """Test that null is rejected as root."""
        validator = StructuralValidator(basic_ir, default_options)

        findings = validator.validate(None)  # type: ignore
        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        assert len(type_errors) == 1


# =============================================================================
# TEST: CONVENIENCE FUNCTION
# =============================================================================


class TestConvenienceFunction:
    """Tests for validate_structure convenience function."""

    def test_validate_structure_basic(self, basic_ir):
        """Test validate_structure convenience function."""
        payload = {"name": "test", "value": 42}
        findings = validate_structure(payload, basic_ir)

        # Valid payload should have no errors
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_validate_structure_with_options(self, basic_ir, strict_options):
        """Test validate_structure with custom options."""
        payload = {"name": "test", "value": 42, "extra": "unknown"}
        findings = validate_structure(payload, basic_ir, strict_options)

        # Should have unknown field error in strict mode
        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert unknown_findings[0].severity == Severity.ERROR

    def test_validate_structure_uses_default_options(self, basic_ir):
        """Test validate_structure uses default options when none provided."""
        payload = {"name": "test", "value": 42, "extra": "unknown"}
        findings = validate_structure(payload, basic_ir)

        # Should have unknown field warning (not error) with default options
        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert unknown_findings[0].severity == Severity.WARNING


# =============================================================================
# TEST: TYPE COMPATIBILITY
# =============================================================================


class TestTypeCompatibility:
    """Tests for type compatibility constants."""

    def test_python_to_json_type_mapping(self):
        """Test Python to JSON type mapping."""
        assert PYTHON_TO_JSON_TYPE[type(None)] == "null"
        assert PYTHON_TO_JSON_TYPE[bool] == "boolean"
        assert PYTHON_TO_JSON_TYPE[int] == "integer"
        assert PYTHON_TO_JSON_TYPE[float] == "number"
        assert PYTHON_TO_JSON_TYPE[str] == "string"
        assert PYTHON_TO_JSON_TYPE[list] == "array"
        assert PYTHON_TO_JSON_TYPE[dict] == "object"

    def test_number_accepts_integer(self):
        """Test that number type accepts integer in compatibility rules."""
        assert "integer" in TYPE_COMPATIBILITY["number"]
        assert "number" in TYPE_COMPATIBILITY["number"]

    def test_integer_does_not_accept_float(self):
        """Test that integer type does not accept float."""
        assert "number" not in TYPE_COMPATIBILITY["integer"]
        assert "integer" in TYPE_COMPATIBILITY["integer"]


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_payload(self, basic_ir, default_options):
        """Test validation of empty payload."""
        validator = StructuralValidator(basic_ir, default_options)
        findings = validator.validate({})

        # Should report missing required fields
        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 2  # name and value

    def test_deeply_nested_objects(self, sample_schema_hash, default_options):
        """Test validation of deeply nested objects."""
        # Create a simple IR that allows nested structures
        ir = SchemaIR(
            schema_id="test/deep",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={
                "/root": PropertyIR(path="/root", type="object", required=True)
            },
            required_paths={"/root"},
        )
        validator = StructuralValidator(ir, default_options)

        # Create deeply nested payload
        nested = {"value": 1}
        for _ in range(10):
            nested = {"nested": nested}
        payload = {"root": nested}

        # Should validate without errors
        findings = validator.validate(payload)
        assert count_errors(findings) == 0

    def test_special_characters_in_field_names(self, sample_schema_hash, default_options):
        """Test handling of special characters in field names."""
        ir = SchemaIR(
            schema_id="test/special",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={
                "/field_with_underscore": PropertyIR(path="/field_with_underscore", type="string", required=True),
                "/field-with-dash": PropertyIR(path="/field-with-dash", type="string", required=True),
            },
            required_paths={"/field_with_underscore", "/field-with-dash"},
        )
        validator = StructuralValidator(ir, default_options)

        payload = {
            "field_with_underscore": "value1",
            "field-with-dash": "value2"
        }
        findings = validator.validate(payload)

        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_boolean_subclass_of_int(self, sample_schema_hash, default_options):
        """Test that booleans are correctly identified (not confused with int)."""
        ir = SchemaIR(
            schema_id="test/bool_int",
            version="1.0.0",
            schema_hash=sample_schema_hash,
            compiled_at=datetime.now(),
            properties={
                "/int_field": PropertyIR(path="/int_field", type="integer", required=True),
                "/bool_field": PropertyIR(path="/bool_field", type="boolean", required=True),
            },
            required_paths={"/int_field", "/bool_field"},
        )
        validator = StructuralValidator(ir, default_options)

        # Test that boolean is NOT accepted as integer
        payload = {"int_field": True, "bool_field": True}
        findings = validator.validate(payload)

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        int_field_errors = [e for e in type_errors if e.path == "/int_field"]
        assert len(int_field_errors) == 1  # True should fail for integer field


# =============================================================================
# TEST: FINDING MESSAGES
# =============================================================================


class TestFindingMessages:
    """Tests for finding message quality."""

    def test_missing_required_message_includes_field_name(self, basic_ir, default_options):
        """Test that missing required message includes field name."""
        validator = StructuralValidator(basic_ir, default_options)
        findings = validator.validate({"name": "test"})  # Missing value

        missing_errors = find_findings_by_code(findings, ErrorCode.MISSING_REQUIRED.value)
        assert len(missing_errors) == 1
        assert "value" in missing_errors[0].message

    def test_type_mismatch_message_includes_types(self, basic_ir, default_options):
        """Test that type mismatch message includes expected and actual types."""
        validator = StructuralValidator(basic_ir, default_options)
        findings = validator.validate({"name": 123, "value": 42})

        type_errors = find_findings_by_code(findings, ErrorCode.TYPE_MISMATCH.value)
        name_errors = [e for e in type_errors if e.path == "/name"]
        assert len(name_errors) == 1
        assert "string" in name_errors[0].message.lower()
        assert "integer" in name_errors[0].message.lower()

    def test_unknown_field_message_includes_field_name(self, basic_ir, strict_options):
        """Test that unknown field message includes field name."""
        validator = StructuralValidator(basic_ir, strict_options)
        findings = validator.validate({"name": "test", "value": 42, "mystery": "field"})

        unknown_findings = find_findings_by_code(findings, ErrorCode.UNKNOWN_FIELD.value)
        assert len(unknown_findings) == 1
        assert "mystery" in unknown_findings[0].message
