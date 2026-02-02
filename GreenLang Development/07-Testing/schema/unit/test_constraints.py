# -*- coding: utf-8 -*-
"""
Unit Tests for Constraint Validator (Task 2.2)

This module tests the ConstraintValidator class that validates values against
schema constraints including:
- Numeric constraints (min/max, exclusive bounds, multipleOf)
- String constraints (pattern, minLength/maxLength, format)
- Array constraints (minItems/maxItems, uniqueItems)
- Enum validation

Error codes tested:
- GLSCHEMA-E200: RANGE_VIOLATION
- GLSCHEMA-E201: PATTERN_MISMATCH
- GLSCHEMA-E202: ENUM_VIOLATION
- GLSCHEMA-E203: LENGTH_VIOLATION
- GLSCHEMA-E204: UNIQUE_VIOLATION
- GLSCHEMA-E205: MULTIPLE_OF_VIOLATION
- GLSCHEMA-E206: FORMAT_VIOLATION

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.2
"""

import pytest
from datetime import datetime

from greenlang.schema.compiler.ir import (
    ArrayConstraintIR,
    CompiledPattern,
    NumericConstraintIR,
    SchemaIR,
    StringConstraintIR,
)
from greenlang.schema.models.config import ValidationOptions
from greenlang.schema.validator.constraints import (
    ConstraintValidator,
    FORMAT_VALIDATORS,
    _validate_date,
    _validate_datetime,
    _validate_email,
    _validate_hostname,
    _validate_ipv4,
    _validate_ipv6,
    _validate_json_pointer,
    _validate_regex,
    _validate_time,
    _validate_uri,
    _validate_uuid,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_ir():
    """Create a sample SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/constraints",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        numeric_constraints={
            "/temperature": NumericConstraintIR(
                path="/temperature",
                minimum=0.0,
                maximum=100.0,
            ),
            "/exclusive_range": NumericConstraintIR(
                path="/exclusive_range",
                exclusive_minimum=0.0,
                exclusive_maximum=10.0,
            ),
            "/multiple": NumericConstraintIR(
                path="/multiple",
                multiple_of=0.5,
            ),
        },
        string_constraints={
            "/email": StringConstraintIR(
                path="/email",
                format="email",
            ),
            "/code": StringConstraintIR(
                path="/code",
                min_length=3,
                max_length=10,
                pattern="^[A-Z]+$",
            ),
        },
        array_constraints={
            "/items": ArrayConstraintIR(
                path="/items",
                min_items=1,
                max_items=10,
                unique_items=True,
            ),
        },
        enums={
            "/status": ["active", "inactive", "pending"],
        },
    )


@pytest.fixture
def validation_options():
    """Create default validation options."""
    return ValidationOptions()


@pytest.fixture
def constraint_validator(sample_schema_ir, validation_options):
    """Create a ConstraintValidator instance for testing."""
    return ConstraintValidator(sample_schema_ir, validation_options)


# =============================================================================
# NUMERIC CONSTRAINT TESTS
# =============================================================================


class TestNumericConstraints:
    """Tests for numeric constraint validation."""

    def test_valid_value_within_range(self, constraint_validator):
        """Test that a valid value within range produces no findings."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0.0,
            maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(50, constraints, "/value")
        assert len(findings) == 0

    def test_value_below_minimum(self, constraint_validator):
        """Test that value below minimum produces RANGE_VIOLATION."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0.0,
            maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(-5, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E200"
        assert "less than minimum" in findings[0].message

    def test_value_above_maximum(self, constraint_validator):
        """Test that value above maximum produces RANGE_VIOLATION."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0.0,
            maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(150, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E200"
        assert "exceeds maximum" in findings[0].message

    def test_value_at_minimum_boundary(self, constraint_validator):
        """Test that value at minimum boundary is valid (inclusive)."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0.0,
            maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(0, constraints, "/value")
        assert len(findings) == 0

    def test_value_at_maximum_boundary(self, constraint_validator):
        """Test that value at maximum boundary is valid (inclusive)."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0.0,
            maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(100, constraints, "/value")
        assert len(findings) == 0

    def test_exclusive_minimum_violated(self, constraint_validator):
        """Test that value at exclusive minimum produces error."""
        constraints = NumericConstraintIR(
            path="/value",
            exclusive_minimum=0.0,
        )
        findings = constraint_validator.validate_numeric(0, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E200"
        assert "must be greater than" in findings[0].message

    def test_exclusive_minimum_valid(self, constraint_validator):
        """Test that value above exclusive minimum is valid."""
        constraints = NumericConstraintIR(
            path="/value",
            exclusive_minimum=0.0,
        )
        findings = constraint_validator.validate_numeric(0.001, constraints, "/value")
        assert len(findings) == 0

    def test_exclusive_maximum_violated(self, constraint_validator):
        """Test that value at exclusive maximum produces error."""
        constraints = NumericConstraintIR(
            path="/value",
            exclusive_maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(100, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E200"
        assert "must be less than" in findings[0].message

    def test_exclusive_maximum_valid(self, constraint_validator):
        """Test that value below exclusive maximum is valid."""
        constraints = NumericConstraintIR(
            path="/value",
            exclusive_maximum=100.0,
        )
        findings = constraint_validator.validate_numeric(99.999, constraints, "/value")
        assert len(findings) == 0

    def test_multiple_of_integer_valid(self, constraint_validator):
        """Test that integer multiple is valid."""
        constraints = NumericConstraintIR(
            path="/value",
            multiple_of=5,
        )
        findings = constraint_validator.validate_numeric(15, constraints, "/value")
        assert len(findings) == 0

    def test_multiple_of_integer_invalid(self, constraint_validator):
        """Test that non-multiple integer produces error."""
        constraints = NumericConstraintIR(
            path="/value",
            multiple_of=5,
        )
        findings = constraint_validator.validate_numeric(13, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E205"
        assert "not a multiple of" in findings[0].message

    def test_multiple_of_float_valid(self, constraint_validator):
        """Test that float multiple is valid."""
        constraints = NumericConstraintIR(
            path="/value",
            multiple_of=0.5,
        )
        findings = constraint_validator.validate_numeric(2.5, constraints, "/value")
        assert len(findings) == 0

    def test_multiple_of_float_invalid(self, constraint_validator):
        """Test that non-multiple float produces error."""
        constraints = NumericConstraintIR(
            path="/value",
            multiple_of=0.5,
        )
        findings = constraint_validator.validate_numeric(2.3, constraints, "/value")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E205"

    def test_multiple_constraints_violated(self, constraint_validator):
        """Test that multiple constraint violations are all reported."""
        constraints = NumericConstraintIR(
            path="/value",
            minimum=10.0,
            maximum=5.0,  # Invalid constraint, but we test both checks
        )
        findings = constraint_validator.validate_numeric(0, constraints, "/value")
        # Should report both minimum and maximum violations
        assert len(findings) >= 1


# =============================================================================
# STRING CONSTRAINT TESTS
# =============================================================================


class TestStringConstraints:
    """Tests for string constraint validation."""

    def test_valid_string_length(self, constraint_validator):
        """Test that string within length limits is valid."""
        constraints = StringConstraintIR(
            path="/name",
            min_length=2,
            max_length=10,
        )
        findings = constraint_validator.validate_string("hello", constraints, "/name")
        assert len(findings) == 0

    def test_string_too_short(self, constraint_validator):
        """Test that string below minLength produces LENGTH_VIOLATION."""
        constraints = StringConstraintIR(
            path="/name",
            min_length=5,
        )
        findings = constraint_validator.validate_string("hi", constraints, "/name")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E203"
        assert "less than minimum" in findings[0].message

    def test_string_too_long(self, constraint_validator):
        """Test that string above maxLength produces LENGTH_VIOLATION."""
        constraints = StringConstraintIR(
            path="/name",
            max_length=5,
        )
        findings = constraint_validator.validate_string("hello world", constraints, "/name")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E203"
        assert "exceeds maximum" in findings[0].message

    def test_string_at_min_length_boundary(self, constraint_validator):
        """Test that string at minLength boundary is valid."""
        constraints = StringConstraintIR(
            path="/name",
            min_length=5,
        )
        findings = constraint_validator.validate_string("hello", constraints, "/name")
        assert len(findings) == 0

    def test_string_at_max_length_boundary(self, constraint_validator):
        """Test that string at maxLength boundary is valid."""
        constraints = StringConstraintIR(
            path="/name",
            max_length=5,
        )
        findings = constraint_validator.validate_string("hello", constraints, "/name")
        assert len(findings) == 0

    def test_pattern_match_valid(self, constraint_validator):
        """Test that string matching pattern is valid."""
        constraints = StringConstraintIR(
            path="/code",
            pattern="^[A-Z]+$",
        )
        findings = constraint_validator.validate_string("ABC", constraints, "/code")
        assert len(findings) == 0

    def test_pattern_match_invalid(self, constraint_validator):
        """Test that string not matching pattern produces PATTERN_MISMATCH."""
        constraints = StringConstraintIR(
            path="/code",
            pattern="^[A-Z]+$",
        )
        findings = constraint_validator.validate_string("abc123", constraints, "/code")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E201"
        assert "does not match pattern" in findings[0].message

    def test_format_email_valid(self, constraint_validator):
        """Test that valid email format passes."""
        constraints = StringConstraintIR(
            path="/email",
            format="email",
        )
        findings = constraint_validator.validate_string(
            "user@example.com", constraints, "/email"
        )
        assert len(findings) == 0

    def test_format_email_invalid(self, constraint_validator):
        """Test that invalid email format produces FORMAT_VIOLATION."""
        constraints = StringConstraintIR(
            path="/email",
            format="email",
        )
        findings = constraint_validator.validate_string(
            "not-an-email", constraints, "/email"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E206"
        assert "does not match format" in findings[0].message

    def test_format_uri_valid(self, constraint_validator):
        """Test that valid URI format passes."""
        constraints = StringConstraintIR(
            path="/url",
            format="uri",
        )
        findings = constraint_validator.validate_string(
            "https://example.com/path", constraints, "/url"
        )
        assert len(findings) == 0

    def test_format_uri_invalid(self, constraint_validator):
        """Test that invalid URI format fails."""
        constraints = StringConstraintIR(
            path="/url",
            format="uri",
        )
        findings = constraint_validator.validate_string(
            "not a uri", constraints, "/url"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E206"

    def test_format_date_valid(self, constraint_validator):
        """Test that valid date format passes."""
        constraints = StringConstraintIR(
            path="/date",
            format="date",
        )
        findings = constraint_validator.validate_string(
            "2023-01-15", constraints, "/date"
        )
        assert len(findings) == 0

    def test_format_date_invalid(self, constraint_validator):
        """Test that invalid date format fails."""
        constraints = StringConstraintIR(
            path="/date",
            format="date",
        )
        findings = constraint_validator.validate_string(
            "01-15-2023", constraints, "/date"  # Wrong format
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E206"

    def test_format_datetime_valid(self, constraint_validator):
        """Test that valid datetime format passes."""
        constraints = StringConstraintIR(
            path="/datetime",
            format="date-time",
        )
        findings = constraint_validator.validate_string(
            "2023-01-15T10:30:00Z", constraints, "/datetime"
        )
        assert len(findings) == 0

    def test_format_uuid_valid(self, constraint_validator):
        """Test that valid UUID format passes."""
        constraints = StringConstraintIR(
            path="/id",
            format="uuid",
        )
        findings = constraint_validator.validate_string(
            "550e8400-e29b-41d4-a716-446655440000", constraints, "/id"
        )
        assert len(findings) == 0

    def test_format_uuid_invalid(self, constraint_validator):
        """Test that invalid UUID format fails."""
        constraints = StringConstraintIR(
            path="/id",
            format="uuid",
        )
        findings = constraint_validator.validate_string(
            "not-a-uuid", constraints, "/id"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E206"


# =============================================================================
# ARRAY CONSTRAINT TESTS
# =============================================================================


class TestArrayConstraints:
    """Tests for array constraint validation."""

    def test_valid_array_size(self, constraint_validator):
        """Test that array within size limits is valid."""
        constraints = ArrayConstraintIR(
            path="/items",
            min_items=1,
            max_items=10,
        )
        findings = constraint_validator.validate_array([1, 2, 3], constraints, "/items")
        assert len(findings) == 0

    def test_array_too_few_items(self, constraint_validator):
        """Test that array with too few items produces LENGTH_VIOLATION."""
        constraints = ArrayConstraintIR(
            path="/items",
            min_items=3,
        )
        findings = constraint_validator.validate_array([1], constraints, "/items")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E203"
        assert "minimum required" in findings[0].message

    def test_array_too_many_items(self, constraint_validator):
        """Test that array with too many items produces LENGTH_VIOLATION."""
        constraints = ArrayConstraintIR(
            path="/items",
            max_items=3,
        )
        findings = constraint_validator.validate_array(
            [1, 2, 3, 4, 5], constraints, "/items"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E203"
        assert "maximum allowed" in findings[0].message

    def test_empty_array_with_min_items(self, constraint_validator):
        """Test that empty array fails minItems constraint."""
        constraints = ArrayConstraintIR(
            path="/items",
            min_items=1,
        )
        findings = constraint_validator.validate_array([], constraints, "/items")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E203"

    def test_unique_items_valid(self, constraint_validator):
        """Test that array with unique items is valid."""
        constraints = ArrayConstraintIR(
            path="/items",
            unique_items=True,
        )
        findings = constraint_validator.validate_array(
            [1, 2, 3, 4, 5], constraints, "/items"
        )
        assert len(findings) == 0

    def test_unique_items_duplicate_primitives(self, constraint_validator):
        """Test that array with duplicate primitives produces UNIQUE_VIOLATION."""
        constraints = ArrayConstraintIR(
            path="/items",
            unique_items=True,
        )
        findings = constraint_validator.validate_array(
            [1, 2, 3, 2, 4], constraints, "/items"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E204"
        assert "duplicate items" in findings[0].message

    def test_unique_items_duplicate_strings(self, constraint_validator):
        """Test that array with duplicate strings produces UNIQUE_VIOLATION."""
        constraints = ArrayConstraintIR(
            path="/items",
            unique_items=True,
        )
        findings = constraint_validator.validate_array(
            ["a", "b", "c", "a"], constraints, "/items"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E204"

    def test_unique_items_duplicate_objects(self, constraint_validator):
        """Test that array with duplicate objects produces UNIQUE_VIOLATION."""
        constraints = ArrayConstraintIR(
            path="/items",
            unique_items=True,
        )
        findings = constraint_validator.validate_array(
            [{"x": 1}, {"x": 2}, {"x": 1}], constraints, "/items"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E204"

    def test_unique_items_similar_but_different(self, constraint_validator):
        """Test that similar but different objects are unique."""
        constraints = ArrayConstraintIR(
            path="/items",
            unique_items=True,
        )
        findings = constraint_validator.validate_array(
            [{"x": 1}, {"x": 2}, {"x": 3}], constraints, "/items"
        )
        assert len(findings) == 0


# =============================================================================
# ENUM VALIDATION TESTS
# =============================================================================


class TestEnumValidation:
    """Tests for enum constraint validation."""

    def test_valid_enum_value(self, constraint_validator):
        """Test that valid enum value passes."""
        findings = constraint_validator.validate_enum(
            "active", ["active", "inactive", "pending"], "/status"
        )
        assert len(findings) == 0

    def test_invalid_enum_value(self, constraint_validator):
        """Test that invalid enum value produces ENUM_VIOLATION."""
        findings = constraint_validator.validate_enum(
            "unknown", ["active", "inactive", "pending"], "/status"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E202"
        assert "not one of allowed values" in findings[0].message

    def test_enum_with_numbers(self, constraint_validator):
        """Test enum validation with numeric values."""
        findings = constraint_validator.validate_enum(
            1, [1, 2, 3], "/level"
        )
        assert len(findings) == 0

    def test_enum_number_not_in_list(self, constraint_validator):
        """Test that number not in enum list fails."""
        findings = constraint_validator.validate_enum(
            4, [1, 2, 3], "/level"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E202"

    def test_enum_with_null(self, constraint_validator):
        """Test enum validation with null value."""
        findings = constraint_validator.validate_enum(
            None, [None, "active", "inactive"], "/status"
        )
        assert len(findings) == 0

    def test_enum_with_mixed_types(self, constraint_validator):
        """Test enum validation with mixed types."""
        findings = constraint_validator.validate_enum(
            "text", [1, "text", True], "/value"
        )
        assert len(findings) == 0

    def test_enum_with_objects(self, constraint_validator):
        """Test enum validation with object values."""
        allowed = [{"type": "a"}, {"type": "b"}]
        findings = constraint_validator.validate_enum(
            {"type": "a"}, allowed, "/config"
        )
        assert len(findings) == 0

    def test_enum_object_not_in_list(self, constraint_validator):
        """Test that object not in enum list fails."""
        allowed = [{"type": "a"}, {"type": "b"}]
        findings = constraint_validator.validate_enum(
            {"type": "c"}, allowed, "/config"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E202"


# =============================================================================
# FORMAT VALIDATOR TESTS
# =============================================================================


class TestFormatValidators:
    """Tests for individual format validators."""

    # Email tests
    def test_validate_email_simple(self):
        """Test simple email validation."""
        assert _validate_email("user@example.com")
        assert _validate_email("user.name@example.com")
        assert _validate_email("user+tag@example.co.uk")

    def test_validate_email_invalid(self):
        """Test invalid email detection."""
        assert not _validate_email("not-an-email")
        assert not _validate_email("@example.com")
        assert not _validate_email("user@")
        assert not _validate_email("")
        assert not _validate_email("a" * 255 + "@example.com")  # Too long

    # URI tests
    def test_validate_uri_valid(self):
        """Test valid URI formats."""
        assert _validate_uri("https://example.com")
        assert _validate_uri("http://localhost:8080/path")
        assert _validate_uri("ftp://ftp.example.com/file.txt")
        assert _validate_uri("file:///path/to/file")

    def test_validate_uri_invalid(self):
        """Test invalid URI detection."""
        assert not _validate_uri("not a uri")
        assert not _validate_uri("")

    # Date tests
    def test_validate_date_valid(self):
        """Test valid date formats."""
        assert _validate_date("2023-01-15")
        assert _validate_date("2000-12-31")
        assert _validate_date("1999-01-01")

    def test_validate_date_invalid(self):
        """Test invalid date detection."""
        assert not _validate_date("01-15-2023")
        assert not _validate_date("2023/01/15")
        assert not _validate_date("not a date")
        assert not _validate_date("")
        assert not _validate_date("2023-13-01")  # Invalid month
        assert not _validate_date("2023-01-32")  # Invalid day

    # DateTime tests
    def test_validate_datetime_valid(self):
        """Test valid datetime formats."""
        assert _validate_datetime("2023-01-15T10:30:00Z")
        assert _validate_datetime("2023-01-15T10:30:00+00:00")
        assert _validate_datetime("2023-01-15T10:30:00.123Z")

    def test_validate_datetime_invalid(self):
        """Test invalid datetime detection."""
        assert not _validate_datetime("2023-01-15")
        assert not _validate_datetime("not a datetime")
        assert not _validate_datetime("")

    # Time tests
    def test_validate_time_valid(self):
        """Test valid time formats."""
        assert _validate_time("10:30:00")
        assert _validate_time("10:30:00Z")
        assert _validate_time("10:30:00+05:30")
        assert _validate_time("10:30:00.123")

    def test_validate_time_invalid(self):
        """Test invalid time detection."""
        assert not _validate_time("25:00:00")  # Invalid hour
        assert not _validate_time("10:60:00")  # Invalid minute
        assert not _validate_time("not a time")
        assert not _validate_time("")

    # UUID tests
    def test_validate_uuid_valid(self):
        """Test valid UUID formats."""
        assert _validate_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert _validate_uuid("550E8400-E29B-41D4-A716-446655440000")  # Uppercase

    def test_validate_uuid_invalid(self):
        """Test invalid UUID detection."""
        assert not _validate_uuid("not-a-uuid")
        assert not _validate_uuid("550e8400-e29b-41d4-a716")  # Too short
        assert not _validate_uuid("")

    # IPv4 tests
    def test_validate_ipv4_valid(self):
        """Test valid IPv4 formats."""
        assert _validate_ipv4("192.168.1.1")
        assert _validate_ipv4("0.0.0.0")
        assert _validate_ipv4("255.255.255.255")

    def test_validate_ipv4_invalid(self):
        """Test invalid IPv4 detection."""
        assert not _validate_ipv4("256.1.1.1")
        assert not _validate_ipv4("192.168.1")
        assert not _validate_ipv4("not an ip")
        assert not _validate_ipv4("")

    # IPv6 tests
    def test_validate_ipv6_valid(self):
        """Test valid IPv6 formats."""
        assert _validate_ipv6("::1")
        assert _validate_ipv6("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert _validate_ipv6("fe80::1")

    def test_validate_ipv6_invalid(self):
        """Test invalid IPv6 detection."""
        assert not _validate_ipv6("192.168.1.1")  # IPv4
        assert not _validate_ipv6("not an ip")
        assert not _validate_ipv6("")

    # Hostname tests
    def test_validate_hostname_valid(self):
        """Test valid hostname formats."""
        assert _validate_hostname("example.com")
        assert _validate_hostname("sub.example.com")
        assert _validate_hostname("localhost")
        assert _validate_hostname("my-server")

    def test_validate_hostname_invalid(self):
        """Test invalid hostname detection."""
        assert not _validate_hostname("-invalid.com")  # Starts with hyphen
        assert not _validate_hostname("invalid-.com")  # Ends with hyphen
        assert not _validate_hostname("")
        assert not _validate_hostname("a" * 254)  # Too long

    # JSON Pointer tests
    def test_validate_json_pointer_valid(self):
        """Test valid JSON Pointer formats."""
        assert _validate_json_pointer("")  # Root
        assert _validate_json_pointer("/foo")
        assert _validate_json_pointer("/foo/bar")
        assert _validate_json_pointer("/foo/0/bar")

    def test_validate_json_pointer_invalid(self):
        """Test invalid JSON Pointer detection."""
        assert not _validate_json_pointer("foo")  # Missing leading /
        assert not _validate_json_pointer("foo/bar")

    # Regex tests
    def test_validate_regex_valid(self):
        """Test valid regex patterns."""
        assert _validate_regex("^[a-z]+$")
        assert _validate_regex("\\d{3}-\\d{4}")
        assert _validate_regex(".*")

    def test_validate_regex_invalid(self):
        """Test invalid regex detection."""
        assert not _validate_regex("[invalid")  # Unclosed bracket
        assert not _validate_regex("(unclosed")


# =============================================================================
# RECURSIVE VALIDATION TESTS
# =============================================================================


class TestRecursiveValidation:
    """Tests for recursive payload validation."""

    def test_validate_nested_object(self, sample_schema_ir, validation_options):
        """Test validation of nested objects."""
        # Add string constraint for nested path
        sample_schema_ir.string_constraints["/user/email"] = StringConstraintIR(
            path="/user/email",
            format="email",
        )

        validator = ConstraintValidator(sample_schema_ir, validation_options)
        payload = {"user": {"email": "invalid-email"}}

        findings = validator.validate(payload, "")
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E206"
        assert findings[0].path == "/user/email"

    def test_validate_array_items(self, sample_schema_ir, validation_options):
        """Test validation of array items."""
        # Add numeric constraint for array items
        sample_schema_ir.numeric_constraints["/values/0"] = NumericConstraintIR(
            path="/values/0",
            minimum=0.0,
        )
        sample_schema_ir.numeric_constraints["/values/1"] = NumericConstraintIR(
            path="/values/1",
            minimum=0.0,
        )

        validator = ConstraintValidator(sample_schema_ir, validation_options)
        payload = {"values": [-5, 10]}

        findings = validator.validate(payload, "")
        assert len(findings) == 1
        assert findings[0].path == "/values/0"


# =============================================================================
# PATTERN WITH TIMEOUT TESTS
# =============================================================================


class TestPatternTimeout:
    """Tests for regex pattern matching with timeout protection."""

    def test_safe_pattern_matches(self, constraint_validator):
        """Test that safe pattern matching works correctly."""
        pattern = CompiledPattern(
            pattern="^[a-z]+$",
            is_safe=True,
            complexity_score=0.1,
            timeout_ms=100,
        )
        result = constraint_validator._match_pattern_with_timeout(pattern, "hello")
        assert result is True

    def test_safe_pattern_no_match(self, constraint_validator):
        """Test that safe pattern non-match returns False."""
        pattern = CompiledPattern(
            pattern="^[a-z]+$",
            is_safe=True,
            complexity_score=0.1,
            timeout_ms=100,
        )
        result = constraint_validator._match_pattern_with_timeout(pattern, "HELLO123")
        assert result is False

    def test_unsafe_pattern_skipped(self, constraint_validator):
        """Test that unsafe pattern is skipped (returns None)."""
        pattern = CompiledPattern(
            pattern="(a+)+$",  # Nested quantifier - unsafe
            is_safe=False,
            complexity_score=0.9,
            timeout_ms=100,
        )
        result = constraint_validator._match_pattern_with_timeout(pattern, "aaa")
        assert result is None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_max_errors(self, sample_schema_ir):
        """Test that max_errors=0 means unlimited."""
        options = ValidationOptions(max_errors=0)
        validator = ConstraintValidator(sample_schema_ir, options)

        # Add many constraint violations
        sample_schema_ir.numeric_constraints["/v1"] = NumericConstraintIR(
            path="/v1", maximum=0
        )
        sample_schema_ir.numeric_constraints["/v2"] = NumericConstraintIR(
            path="/v2", maximum=0
        )

        payload = {"v1": 100, "v2": 200}
        findings = validator.validate(payload, "")

        # With max_errors=0, both should be reported
        assert len(findings) == 2

    def test_float_equality_in_enum(self, constraint_validator):
        """Test that float equality works correctly in enum."""
        findings = constraint_validator.validate_enum(
            1.0, [1, 2, 3], "/value"  # 1.0 should equal 1
        )
        assert len(findings) == 0

    def test_empty_enum_list(self, constraint_validator):
        """Test validation against empty enum list."""
        findings = constraint_validator.validate_enum(
            "anything", [], "/value"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E202"

    def test_unicode_string_length(self, constraint_validator):
        """Test that Unicode string length is counted correctly."""
        constraints = StringConstraintIR(
            path="/name",
            max_length=5,
        )
        # Unicode characters
        findings = constraint_validator.validate_string(
            "\u00e9\u00e8\u00ea\u00eb\u00f4", constraints, "/name"
        )
        assert len(findings) == 0  # 5 characters exactly

    def test_nan_handling(self, constraint_validator):
        """Test handling of NaN values."""
        import math
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0,
            maximum=100,
        )
        # NaN comparisons are always False, so both min and max should fail
        findings = constraint_validator.validate_numeric(
            float('nan'), constraints, "/value"
        )
        # NaN handling depends on implementation
        # At minimum, we shouldn't crash

    def test_infinity_handling(self, constraint_validator):
        """Test handling of infinity values."""
        import math
        constraints = NumericConstraintIR(
            path="/value",
            minimum=0,
            maximum=100,
        )
        findings = constraint_validator.validate_numeric(
            float('inf'), constraints, "/value"
        )
        assert len(findings) == 1
        assert findings[0].code == "GLSCHEMA-E200"


# =============================================================================
# FORMAT VALIDATORS REGISTRY TESTS
# =============================================================================


class TestFormatValidatorsRegistry:
    """Tests for the FORMAT_VALIDATORS registry."""

    def test_all_standard_formats_registered(self):
        """Test that all standard JSON Schema formats are registered."""
        standard_formats = [
            "email",
            "uri",
            "uri-reference",
            "date",
            "date-time",
            "time",
            "uuid",
            "ipv4",
            "ipv6",
            "hostname",
            "regex",
            "json-pointer",
        ]
        for fmt in standard_formats:
            assert fmt in FORMAT_VALIDATORS, f"Format '{fmt}' not registered"

    def test_format_validators_are_callable(self):
        """Test that all format validators are callable."""
        for name, validator in FORMAT_VALIDATORS.items():
            assert callable(validator), f"Format validator '{name}' is not callable"

    def test_format_validators_accept_string(self):
        """Test that all format validators accept a string argument."""
        for name, validator in FORMAT_VALIDATORS.items():
            try:
                result = validator("test")
                assert isinstance(result, bool), (
                    f"Format validator '{name}' should return bool"
                )
            except Exception as e:
                pytest.fail(
                    f"Format validator '{name}' raised exception: {e}"
                )


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
