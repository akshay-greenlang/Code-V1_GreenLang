# -*- coding: utf-8 -*-
"""
Unit Tests for Schema Self-Validator (Task 1.6).

This module tests the SchemaValidator that validates schema documents
themselves for governance compliance per PRD section 6.8.

Test Coverage:
    1. Reference resolution validation
    2. Cycle detection with trace
    3. Duplicate property key detection
    4. Deprecation metadata validation
    5. Unit metadata validation
    6. Constraint consistency validation
    7. Pattern regex safety validation
    8. Rule expression validation

Author: GreenLang Team
Date: 2026-01-29
"""

import pytest
from typing import Dict, Any

from greenlang.schema.compiler.schema_validator import (
    SchemaValidator,
    SchemaValidationFinding,
    SchemaValidationResult,
    RegexAnalyzer,
    RegexAnalysisResult,
    validate_schema,
    is_valid_semver,
    compare_semver,
)
from greenlang.schema.errors import ErrorCode


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def validator() -> SchemaValidator:
    """Create a SchemaValidator instance for testing."""
    return SchemaValidator(strict=True)


@pytest.fixture
def permissive_validator() -> SchemaValidator:
    """Create a permissive SchemaValidator instance."""
    return SchemaValidator(strict=False)


@pytest.fixture
def regex_analyzer() -> RegexAnalyzer:
    """Create a RegexAnalyzer instance for testing."""
    return RegexAnalyzer()


# ============================================================================
# BASIC VALIDATION TESTS
# ============================================================================


class TestBasicValidation:
    """Tests for basic schema validation."""

    def test_valid_simple_schema(self, validator: SchemaValidator) -> None:
        """Test validation of a simple valid schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }
        result = validator.validate(schema)

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.validation_time_ms > 0

    def test_valid_nested_schema(self, validator: SchemaValidator) -> None:
        """Test validation of a nested schema."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
        result = validator.validate(schema)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_array_schema(self, validator: SchemaValidator) -> None:
        """Test validation of array schema."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 100,
        }
        result = validator.validate(schema)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_schema_with_definitions(self, validator: SchemaValidator) -> None:
        """Test schema with $defs definitions."""
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                    },
                },
            },
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_invalid_type_keyword(self, validator: SchemaValidator) -> None:
        """Test detection of invalid type keyword."""
        schema = {
            "type": "invalid_type",
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == ErrorCode.SCHEMA_KEYWORD_INVALID.value


# ============================================================================
# REFERENCE VALIDATION TESTS
# ============================================================================


class TestReferenceValidation:
    """Tests for $ref reference validation."""

    def test_valid_local_ref(self, validator: SchemaValidator) -> None:
        """Test valid local $ref resolution."""
        schema = {
            "$defs": {
                "Name": {"type": "string", "minLength": 1},
            },
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/Name"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_ref_target(self, validator: SchemaValidator) -> None:
        """Test detection of missing $ref target."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/NonExistent"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.REF_RESOLUTION_FAILED.value for e in result.errors)

    def test_circular_ref_detection(self, validator: SchemaValidator) -> None:
        """Test detection of circular references."""
        schema = {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "type": "object",
            "properties": {
                "value": {"$ref": "#/$defs/A"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        circular_errors = [e for e in result.errors if e.code == ErrorCode.CIRCULAR_REF.value]
        assert len(circular_errors) >= 1
        # Should have cycle trace in details
        assert "cycle" in circular_errors[0].details

    def test_self_circular_ref(self, validator: SchemaValidator) -> None:
        """Test detection of self-referencing circular reference."""
        schema = {
            "$defs": {
                "Recursive": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Recursive"},
                    },
                },
            },
            "type": "object",
            "properties": {
                "root": {"$ref": "#/$defs/Recursive"},
            },
        }
        # Self-referencing schemas are actually valid in JSON Schema
        # (for recursive structures), so this should pass
        result = validator.validate(schema)
        # The validator should not report an error for valid recursive schemas
        # unless there's an actual infinite loop in the schema structure itself

    def test_deep_ref_chain(self, validator: SchemaValidator) -> None:
        """Test validation of deep $ref chains."""
        schema = {
            "$defs": {
                "Level1": {"$ref": "#/$defs/Level2"},
                "Level2": {"$ref": "#/$defs/Level3"},
                "Level3": {"type": "string"},
            },
            "type": "object",
            "properties": {
                "value": {"$ref": "#/$defs/Level1"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is True


# ============================================================================
# DUPLICATE KEY TESTS
# ============================================================================


class TestDuplicateKeyValidation:
    """Tests for duplicate key detection."""

    def test_alias_collision_with_property(self, validator: SchemaValidator) -> None:
        """Test detection of alias that collides with existing property."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "fullName": {"type": "string"},
            },
            "x-gl-aliases": {
                "name": "fullName",  # Alias points to another property
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.DUPLICATE_KEY.value for e in result.errors)

    def test_multiple_aliases_to_same_target(self, validator: SchemaValidator) -> None:
        """Test detection of multiple aliases pointing to same target."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "x-gl-aliases": {
                "fullName": "name",
                "completeName": "name",
            },
        }
        result = validator.validate(schema)

        # Should have a warning about multiple aliases
        warnings = [w for w in result.warnings if w.code == ErrorCode.DUPLICATE_KEY.value]
        assert len(warnings) >= 1


# ============================================================================
# DEPRECATION METADATA TESTS
# ============================================================================


class TestDeprecationMetadataValidation:
    """Tests for deprecation metadata validation."""

    def test_valid_deprecation_metadata(self, validator: SchemaValidator) -> None:
        """Test valid deprecation metadata."""
        schema = {
            "type": "object",
            "properties": {
                "oldField": {
                    "type": "string",
                    "deprecated": True,
                    "x-gl-deprecated": {
                        "since_version": "1.0.0",
                        "removal_version": "2.0.0",
                        "message": "Use newField instead",
                        "replacement": "newField",
                    },
                },
                "newField": {"type": "string"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is True

    def test_invalid_since_version(self, validator: SchemaValidator) -> None:
        """Test detection of invalid since_version."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "x-gl-deprecated": {
                        "since_version": "not-semver",
                    },
                },
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_KEYWORD_INVALID.value for e in result.errors)

    def test_removal_before_since(self, validator: SchemaValidator) -> None:
        """Test detection of removal_version < since_version."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "x-gl-deprecated": {
                        "since_version": "2.0.0",
                        "removal_version": "1.0.0",  # Before since!
                    },
                },
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value for e in result.errors)

    def test_deprecated_without_metadata(self, validator: SchemaValidator) -> None:
        """Test info finding for deprecated without x-gl-deprecated."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "deprecated": True,
                    # Missing x-gl-deprecated metadata
                },
            },
        }
        result = validator.validate(schema)

        # Should have an info-level finding
        assert len(result.info) >= 1


# ============================================================================
# CONSTRAINT CONSISTENCY TESTS
# ============================================================================


class TestConstraintConsistencyValidation:
    """Tests for constraint consistency validation."""

    def test_valid_numeric_constraints(self, validator: SchemaValidator) -> None:
        """Test valid numeric constraints."""
        schema = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": -1,
            "exclusiveMaximum": 101,
        }
        result = validator.validate(schema)

        assert result.valid is True

    def test_minimum_greater_than_maximum(self, validator: SchemaValidator) -> None:
        """Test detection of minimum > maximum."""
        schema = {
            "type": "number",
            "minimum": 100,
            "maximum": 50,
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value for e in result.errors)
        # Check error message mentions the values
        error = next(e for e in result.errors if e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value)
        assert "100" in error.message and "50" in error.message

    def test_exclusive_min_greater_than_exclusive_max(self, validator: SchemaValidator) -> None:
        """Test detection of exclusiveMinimum >= exclusiveMaximum."""
        schema = {
            "type": "number",
            "exclusiveMinimum": 100,
            "exclusiveMaximum": 50,
        }
        result = validator.validate(schema)

        assert result.valid is False

    def test_min_length_greater_than_max_length(self, validator: SchemaValidator) -> None:
        """Test detection of minLength > maxLength."""
        schema = {
            "type": "string",
            "minLength": 100,
            "maxLength": 10,
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value for e in result.errors)

    def test_min_items_greater_than_max_items(self, validator: SchemaValidator) -> None:
        """Test detection of minItems > maxItems."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 50,
            "maxItems": 10,
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value for e in result.errors)

    def test_min_properties_greater_than_max_properties(self, validator: SchemaValidator) -> None:
        """Test detection of minProperties > maxProperties."""
        schema = {
            "type": "object",
            "minProperties": 10,
            "maxProperties": 5,
        }
        result = validator.validate(schema)

        assert result.valid is False

    def test_enum_type_mismatch(self, validator: SchemaValidator) -> None:
        """Test detection of enum values not matching declared type."""
        schema = {
            "type": "integer",
            "enum": [1, 2, "three"],  # "three" is string, not integer
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value for e in result.errors)

    def test_const_type_mismatch(self, validator: SchemaValidator) -> None:
        """Test detection of const value not matching declared type."""
        schema = {
            "type": "number",
            "const": "not a number",
        }
        result = validator.validate(schema)

        assert result.valid is False


# ============================================================================
# PATTERN SAFETY TESTS
# ============================================================================


class TestPatternSafetyValidation:
    """Tests for regex pattern safety validation."""

    def test_valid_safe_pattern(self, validator: SchemaValidator) -> None:
        """Test validation of safe regex pattern."""
        schema = {
            "type": "string",
            "pattern": "^[a-zA-Z0-9]+$",
        }
        result = validator.validate(schema)

        assert result.valid is True

    def test_invalid_regex_syntax(self, validator: SchemaValidator) -> None:
        """Test detection of invalid regex syntax."""
        schema = {
            "type": "string",
            "pattern": "[invalid(regex",
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_KEYWORD_INVALID.value for e in result.errors)

    def test_nested_quantifier_detection(self, validator: SchemaValidator) -> None:
        """Test detection of nested quantifier (ReDoS risk)."""
        schema = {
            "type": "string",
            "pattern": "(a+)+",
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.REGEX_TOO_COMPLEX.value for e in result.errors)

    def test_pattern_in_pattern_properties(self, validator: SchemaValidator) -> None:
        """Test validation of patterns in patternProperties."""
        schema = {
            "type": "object",
            "patternProperties": {
                "^[a-z]+$": {"type": "string"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is True

    def test_unsafe_pattern_in_pattern_properties(self, validator: SchemaValidator) -> None:
        """Test detection of unsafe pattern in patternProperties."""
        schema = {
            "type": "object",
            "patternProperties": {
                "(a+)+": {"type": "string"},
            },
        }
        result = validator.validate(schema)

        assert result.valid is False


# ============================================================================
# UNIT METADATA TESTS
# ============================================================================


class TestUnitMetadataValidation:
    """Tests for unit metadata validation."""

    def test_valid_unit_metadata(self, validator: SchemaValidator) -> None:
        """Test valid unit metadata without catalog."""
        schema = {
            "type": "number",
            "x-gl-unit": {
                "dimension": "energy",
                "canonical": "kWh",
                "allowed": ["kWh", "Wh", "MWh"],
            },
        }
        result = validator.validate(schema)

        # Without unit catalog, validation passes
        assert result.valid is True

    def test_missing_dimension(self, validator: SchemaValidator) -> None:
        """Test detection of missing dimension in unit metadata."""
        schema = {
            "type": "number",
            "x-gl-unit": {
                "canonical": "kWh",
                # Missing "dimension"
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.DIMENSION_INVALID.value for e in result.errors)

    def test_missing_canonical(self, validator: SchemaValidator) -> None:
        """Test detection of missing canonical unit."""
        schema = {
            "type": "number",
            "x-gl-unit": {
                "dimension": "energy",
                # Missing "canonical"
            },
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.UNIT_MISSING.value for e in result.errors)

    def test_invalid_unit_metadata_type(self, validator: SchemaValidator) -> None:
        """Test detection of invalid x-gl-unit type."""
        schema = {
            "type": "number",
            "x-gl-unit": "not an object",
        }
        result = validator.validate(schema)

        assert result.valid is False


# ============================================================================
# RULE EXPRESSION TESTS
# ============================================================================


class TestRuleExpressionValidation:
    """Tests for rule expression validation."""

    def test_valid_rule(self, validator: SchemaValidator) -> None:
        """Test valid rule expression."""
        schema = {
            "type": "object",
            "properties": {
                "min": {"type": "number"},
                "max": {"type": "number"},
            },
            "x-gl-rules": [
                {
                    "id": "min_less_than_max",
                    "severity": "error",
                    "message": "min must be less than max",
                    "check": {
                        "lt": [{"path": "min"}, {"path": "max"}],
                    },
                },
            ],
        }
        result = validator.validate(schema)

        assert result.valid is True

    def test_rule_missing_id(self, validator: SchemaValidator) -> None:
        """Test detection of rule without id."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "x-gl-rules": [
                {
                    # Missing "id"
                    "check": {"gt": [{"path": "value"}, 0]},
                },
            ],
        }
        result = validator.validate(schema)

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_KEYWORD_INVALID.value for e in result.errors)

    def test_rule_missing_check(self, validator: SchemaValidator) -> None:
        """Test detection of rule without check expression."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "x-gl-rules": [
                {
                    "id": "test_rule",
                    # Missing "check"
                },
            ],
        }
        result = validator.validate(schema)

        assert result.valid is False

    def test_rule_invalid_severity(self, validator: SchemaValidator) -> None:
        """Test detection of invalid rule severity."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "x-gl-rules": [
                {
                    "id": "test_rule",
                    "severity": "invalid_severity",
                    "check": {"gt": [{"path": "value"}, 0]},
                },
            ],
        }
        result = validator.validate(schema)

        assert result.valid is False

    def test_rule_referencing_nonexistent_path(self, validator: SchemaValidator) -> None:
        """Test warning for rule referencing non-existent property."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "x-gl-rules": [
                {
                    "id": "test_rule",
                    "check": {"gt": [{"path": "nonexistent"}, 0]},
                },
            ],
        }
        result = validator.validate(schema)

        # Should have a warning
        assert any(w.code == ErrorCode.SCHEMA_KEYWORD_INVALID.value for w in result.warnings)


# ============================================================================
# REGEX ANALYZER TESTS
# ============================================================================


class TestRegexAnalyzer:
    """Tests for the RegexAnalyzer class."""

    def test_safe_pattern(self, regex_analyzer: RegexAnalyzer) -> None:
        """Test analysis of safe pattern."""
        result = regex_analyzer.analyze("^[a-zA-Z0-9]+$")

        assert result.is_safe is True
        assert result.complexity_score < 0.8
        assert result.vulnerability_type is None

    def test_nested_quantifier_detection(self, regex_analyzer: RegexAnalyzer) -> None:
        """Test detection of nested quantifiers."""
        result = regex_analyzer.analyze("(a+)+")

        assert result.is_safe is False
        assert result.vulnerability_type == "nested_quantifier"
        assert result.complexity_score > 0.8

    def test_overlapping_alternation_detection(self, regex_analyzer: RegexAnalyzer) -> None:
        """Test detection of overlapping alternations."""
        result = regex_analyzer.analyze("(a|a)+")

        assert result.is_safe is False
        assert result.vulnerability_type == "overlapping_alternation"

    def test_pattern_too_long(self, regex_analyzer: RegexAnalyzer) -> None:
        """Test detection of overly long patterns."""
        long_pattern = "a" * 2000
        result = regex_analyzer.analyze(long_pattern)

        assert result.is_safe is False
        assert result.vulnerability_type == "pattern_too_long"

    def test_is_valid_regex(self, regex_analyzer: RegexAnalyzer) -> None:
        """Test regex syntax validation."""
        valid, error = regex_analyzer.is_valid_regex("^[a-z]+$")
        assert valid is True
        assert error is None

        valid, error = regex_analyzer.is_valid_regex("[invalid(regex")
        assert valid is False
        assert error is not None


# ============================================================================
# SEMVER VALIDATION TESTS
# ============================================================================


class TestSemverValidation:
    """Tests for semver validation functions."""

    @pytest.mark.parametrize("version", [
        "1.0.0",
        "0.0.1",
        "10.20.30",
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-beta.2",
        "1.0.0+build.123",
        "1.0.0-alpha+build.456",
    ])
    def test_valid_semver(self, version: str) -> None:
        """Test valid semver versions."""
        assert is_valid_semver(version) is True

    @pytest.mark.parametrize("version", [
        "1.0",
        "v1.0.0",
        "1.0.0.0",
        "1",
        "not-a-version",
        "",
    ])
    def test_invalid_semver(self, version: str) -> None:
        """Test invalid semver versions."""
        assert is_valid_semver(version) is False

    def test_compare_semver(self) -> None:
        """Test semver comparison."""
        assert compare_semver("1.0.0", "1.0.0") == 0
        assert compare_semver("1.0.0", "2.0.0") == -1
        assert compare_semver("2.0.0", "1.0.0") == 1
        assert compare_semver("1.1.0", "1.0.0") == 1
        assert compare_semver("1.0.1", "1.0.0") == 1
        assert compare_semver("1.0.0", "1.0.0-alpha") == 1  # Release > prerelease
        assert compare_semver("1.0.0-alpha", "1.0.0-beta") == -1


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestValidateSchemaFunction:
    """Tests for the validate_schema convenience function."""

    def test_validate_dict_schema(self) -> None:
        """Test validating a dict schema."""
        result = validate_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })

        assert isinstance(result, SchemaValidationResult)
        assert result.valid is True

    def test_validate_json_string_schema(self) -> None:
        """Test validating a JSON string schema."""
        result = validate_schema("""
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        """)

        assert result.valid is True

    def test_validate_invalid_json(self) -> None:
        """Test handling of invalid JSON."""
        result = validate_schema("not valid json or yaml")

        assert result.valid is False
        assert any(e.code == ErrorCode.SCHEMA_PARSE_ERROR.value for e in result.errors)

    def test_strict_vs_permissive(self) -> None:
        """Test difference between strict and permissive modes."""
        schema_with_unsafe_pattern = {
            "type": "string",
            "pattern": "(a+)+",
        }

        strict_result = validate_schema(schema_with_unsafe_pattern, strict=True)
        assert strict_result.valid is False

        permissive_result = validate_schema(schema_with_unsafe_pattern, strict=False)
        # In permissive mode, ReDoS patterns become warnings
        assert len(permissive_result.warnings) > 0


# ============================================================================
# RESULT MODEL TESTS
# ============================================================================


class TestSchemaValidationResult:
    """Tests for SchemaValidationResult model."""

    def test_finding_count_property(self) -> None:
        """Test finding_count property."""
        result = SchemaValidationResult(
            valid=False,
            errors=[
                SchemaValidationFinding(
                    code="GLSCHEMA-E509",
                    severity="error",
                    path="/test",
                    message="Test error",
                ),
            ],
            warnings=[
                SchemaValidationFinding(
                    code="GLSCHEMA-W700",
                    severity="warning",
                    path="/test2",
                    message="Test warning",
                ),
            ],
        )

        assert result.finding_count == 2
        assert result.error_count == 1
        assert result.warning_count == 1

    def test_to_summary(self) -> None:
        """Test to_summary method."""
        result = SchemaValidationResult(
            valid=True,
            schema_id="test/schema",
            version="1.0.0",
        )

        summary = result.to_summary()
        assert "VALID" in summary
        assert "test/schema" in summary
        assert "1.0.0" in summary


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_schema(self, validator: SchemaValidator) -> None:
        """Test validation of empty schema."""
        result = validator.validate({})
        # Empty schema is valid (matches everything)
        assert result.valid is True

    def test_boolean_schema_true(self, validator: SchemaValidator) -> None:
        """Test validation handles boolean true schema concept."""
        # JSON Schema allows "true" as a schema that accepts anything
        # Our validator expects dicts, but should handle gracefully
        schema = {"type": "object", "additionalProperties": True}
        result = validator.validate(schema)
        assert result.valid is True

    def test_deeply_nested_schema(self, validator: SchemaValidator) -> None:
        """Test validation of deeply nested schema."""
        # Create a deeply nested schema
        schema: Dict[str, Any] = {"type": "string"}
        for _ in range(20):
            schema = {
                "type": "object",
                "properties": {"nested": schema},
            }

        result = validator.validate(schema)
        assert result.valid is True

    def test_schema_with_all_constraint_types(self, validator: SchemaValidator) -> None:
        """Test schema with all constraint types."""
        schema = {
            "type": "object",
            "properties": {
                "str_field": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-z]+$",
                },
                "num_field": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "exclusiveMinimum": -1,
                    "exclusiveMaximum": 101,
                    "multipleOf": 0.5,
                },
                "int_field": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "arr_field": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 10,
                    "uniqueItems": True,
                },
                "obj_field": {
                    "type": "object",
                    "minProperties": 0,
                    "maxProperties": 5,
                },
                "enum_field": {
                    "type": "string",
                    "enum": ["a", "b", "c"],
                },
                "const_field": {
                    "type": "string",
                    "const": "fixed",
                },
            },
            "required": ["str_field"],
        }

        result = validator.validate(schema)
        assert result.valid is True

    def test_multiple_errors_collected(self, validator: SchemaValidator) -> None:
        """Test that multiple errors are collected."""
        schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "type": "number",
                    "minimum": 100,
                    "maximum": 50,  # Error 1
                },
                "field2": {
                    "type": "string",
                    "minLength": 100,
                    "maxLength": 10,  # Error 2
                },
                "field3": {
                    "type": "array",
                    "minItems": 50,
                    "maxItems": 10,  # Error 3
                },
            },
        }

        result = validator.validate(schema)
        assert result.valid is False
        assert len(result.errors) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
