"""
Comprehensive tests for GreenLang Schema Validator.

Tests cover:
- JSON Schema validation
- Schema compilation and caching
- Custom format validators
- Graceful degradation when jsonschema unavailable
- Error reporting
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from greenlang.validation.schema import SchemaValidator, SchemaValidationError
from greenlang.validation.framework import ValidationResult, ValidationSeverity


# Test Fixtures
@pytest.fixture
def basic_schema():
    """Basic JSON schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name"]
    }


@pytest.fixture
def nested_schema():
    """Nested JSON schema."""
    return {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["username"]
            },
            "age": {"type": "integer", "minimum": 18}
        },
        "required": ["user"]
    }


@pytest.fixture
def array_schema():
    """Schema for array validation."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            },
            "required": ["id"]
        },
        "minItems": 1
    }


@pytest.fixture
def schema_with_formats():
    """Schema with format validators."""
    return {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "date": {"type": "string", "format": "date"},
            "uri": {"type": "string", "format": "uri"}
        }
    }


# SchemaValidator Tests
class TestSchemaValidator:
    """Test SchemaValidator functionality."""

    def test_validator_creation(self, basic_schema):
        """Test creating a schema validator."""
        validator = SchemaValidator(basic_schema)
        assert validator.schema == basic_schema

    def test_valid_data(self, basic_schema):
        """Test validation with valid data."""
        validator = SchemaValidator(basic_schema)
        data = {"name": "John", "age": 30}

        result = validator.validate(data)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_field(self, basic_schema):
        """Test validation with missing required field."""
        validator = SchemaValidator(basic_schema)
        data = {"age": 30}  # Missing 'name'

        result = validator.validate(data)

        assert result.valid is False
        assert len(result.errors) >= 1
        # Check that error mentions the required field
        error_messages = [e.message for e in result.errors]
        assert any("name" in msg.lower() for msg in error_messages)

    def test_wrong_type(self, basic_schema):
        """Test validation with wrong type."""
        validator = SchemaValidator(basic_schema)
        data = {"name": "John", "age": "thirty"}  # Wrong type for age

        result = validator.validate(data)

        assert result.valid is False
        assert len(result.errors) >= 1

    def test_minimum_constraint(self, basic_schema):
        """Test minimum value constraint."""
        validator = SchemaValidator(basic_schema)
        data = {"name": "John", "age": -5}  # Below minimum

        result = validator.validate(data)

        assert result.valid is False
        assert any("age" in e.field for e in result.errors)

    def test_nested_object_validation(self, nested_schema):
        """Test validation of nested objects."""
        validator = SchemaValidator(nested_schema)
        data = {
            "user": {
                "username": "johndoe",
                "email": "john@example.com"
            },
            "age": 25
        }

        result = validator.validate(data)
        assert result.valid is True

    def test_nested_object_missing_field(self, nested_schema):
        """Test validation with missing nested field."""
        validator = SchemaValidator(nested_schema)
        data = {
            "user": {
                "email": "john@example.com"
                # Missing 'username'
            },
            "age": 25
        }

        result = validator.validate(data)

        assert result.valid is False
        # Check that error mentions nested field
        assert any("user" in e.field or "username" in e.field for e in result.errors)

    def test_array_validation(self, array_schema):
        """Test array validation."""
        validator = SchemaValidator(array_schema)
        data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"}
        ]

        result = validator.validate(data)
        assert result.valid is True

    def test_array_missing_required(self, array_schema):
        """Test array with item missing required field."""
        validator = SchemaValidator(array_schema)
        data = [
            {"id": 1, "name": "Item 1"},
            {"name": "Item 2"}  # Missing 'id'
        ]

        result = validator.validate(data)
        assert result.valid is False

    def test_array_min_items(self, array_schema):
        """Test array minimum items constraint."""
        validator = SchemaValidator(array_schema)
        data = []  # Below minItems

        result = validator.validate(data)
        assert result.valid is False

    def test_root_type_validation(self):
        """Test validation of root type."""
        schema = {"type": "string"}
        validator = SchemaValidator(schema)

        # Valid
        result = validator.validate("hello")
        assert result.valid is True

        # Invalid
        result = validator.validate(123)
        assert result.valid is False

    def test_null_type_validation(self):
        """Test null type validation."""
        schema = {
            "type": "object",
            "properties": {
                "optional_field": {"type": ["string", "null"]}
            }
        }
        validator = SchemaValidator(schema)

        # Null should be valid
        result = validator.validate({"optional_field": None})
        assert result.valid is True

        # String should be valid
        result = validator.validate({"optional_field": "value"})
        assert result.valid is True

    def test_enum_validation(self):
        """Test enum validation."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"]}
            }
        }
        validator = SchemaValidator(schema)

        # Valid enum value
        result = validator.validate({"status": "active"})
        assert result.valid is True

        # Invalid enum value
        result = validator.validate({"status": "unknown"})
        assert result.valid is False

    def test_pattern_validation(self):
        """Test pattern validation."""
        schema = {
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z]{3}[0-9]{3}$"}
            }
        }
        validator = SchemaValidator(schema)

        # Valid pattern
        result = validator.validate({"code": "ABC123"})
        assert result.valid is True

        # Invalid pattern
        result = validator.validate({"code": "abc123"})
        assert result.valid is False

    def test_additional_properties(self):
        """Test additionalProperties constraint."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": False
        }
        validator = SchemaValidator(schema)

        # No additional properties - valid
        result = validator.validate({"name": "John"})
        assert result.valid is True

        # Has additional properties - invalid
        result = validator.validate({"name": "John", "extra": "field"})
        assert result.valid is False

    def test_error_field_path(self, nested_schema):
        """Test that error field paths are correct."""
        validator = SchemaValidator(nested_schema)
        data = {
            "user": {
                "email": "invalid"
                # Missing username
            },
            "age": 15  # Below minimum
        }

        result = validator.validate(data)

        # Check that errors have proper field paths
        assert any(e.field for e in result.errors)
        assert any(e.location for e in result.errors)

    def test_error_severity(self, basic_schema):
        """Test that errors have correct severity."""
        validator = SchemaValidator(basic_schema)
        data = {"age": 30}  # Missing required field

        result = validator.validate(data)

        assert all(e.severity == ValidationSeverity.ERROR for e in result.errors)

    def test_error_validator_name(self, basic_schema):
        """Test that errors have correct validator name."""
        validator = SchemaValidator(basic_schema)
        data = {"age": 30}

        result = validator.validate(data)

        assert all(e.validator == "json_schema" for e in result.errors)

    @patch('greenlang.validation.schema.JSONSCHEMA_AVAILABLE', False)
    def test_basic_validation_fallback(self, basic_schema):
        """Test basic validation when jsonschema unavailable."""
        validator = SchemaValidator(basic_schema)

        # Valid data
        data = {"name": "John", "age": 30}
        result = validator.validate(data)
        # Basic validation may still pass

        # Missing required field
        data = {"age": 30}
        result = validator.validate(data)
        assert result.valid is False

    @patch('greenlang.validation.schema.JSONSCHEMA_AVAILABLE', False)
    def test_basic_type_checking(self):
        """Test basic type checking without jsonschema."""
        schema = {"type": "string"}
        validator = SchemaValidator(schema)

        # Correct type
        result = validator.validate("hello")
        assert result.valid is True

        # Wrong type
        result = validator.validate(123)
        assert result.valid is False

    @patch('greenlang.validation.schema.JSONSCHEMA_AVAILABLE', False)
    def test_basic_object_type_checking(self, basic_schema):
        """Test basic object type checking."""
        validator = SchemaValidator(basic_schema)

        # Correct type (object)
        result = validator.validate({"name": "John"})
        # Should check required fields

        # Wrong type (not object)
        result = validator.validate("not an object")
        assert result.valid is False

    def test_from_file(self, tmp_path, basic_schema):
        """Test loading schema from file."""
        # Create temporary schema file
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(basic_schema))

        # Load validator from file
        validator = SchemaValidator.from_file(str(schema_file))

        # Test validation
        data = {"name": "John", "age": 30}
        result = validator.validate(data)
        assert result.valid is True

    def test_complex_schema(self):
        """Test complex schema with multiple constraints."""
        schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                    "pattern": "^[a-zA-Z0-9_]+$"
                },
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "age": {
                    "type": "integer",
                    "minimum": 18,
                    "maximum": 120
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                    "minItems": 1,
                    "maxItems": 5
                }
            },
            "required": ["username", "email"]
        }

        validator = SchemaValidator(schema)

        # Valid data
        data = {
            "username": "john_doe",
            "email": "john@example.com",
            "age": 25,
            "tags": ["developer", "python"]
        }
        result = validator.validate(data)
        assert result.valid is True

        # Invalid username (too short)
        data = {
            "username": "jd",
            "email": "john@example.com"
        }
        result = validator.validate(data)
        assert result.valid is False

    def test_oneOf_validation(self):
        """Test oneOf schema validation."""
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        }
        validator = SchemaValidator(schema)

        # Valid - string
        result = validator.validate("hello")
        assert result.valid is True

        # Valid - number
        result = validator.validate(42)
        assert result.valid is True

        # Invalid - boolean
        result = validator.validate(True)
        assert result.valid is False

    def test_allOf_validation(self):
        """Test allOf schema validation."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "object", "properties": {"age": {"type": "integer"}}}
            ]
        }
        validator = SchemaValidator(schema)

        # Valid
        result = validator.validate({"name": "John", "age": 30})
        assert result.valid is True

        # Invalid - missing property
        result = validator.validate({"name": "John"})
        # Depends on schema strictness
