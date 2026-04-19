# -*- coding: utf-8 -*-
"""
Unit tests for JSON schema validation

Tests JSON schema enforcement including:
- validate_json_payload() accepts valid JSON
- validate_json_payload() rejects invalid JSON syntax
- validate_json_payload() rejects schema violations
- validate_json_object() validates Python dicts
- extract_json_from_text() extracts from markdown
- normalize_tool_arguments() handles string JSON
- Error messages include helpful details
"""

import pytest
import json
from greenlang.intelligence.runtime.jsonio import (
    validate_json_payload,
    validate_json_object,
    extract_json_from_text,
    normalize_tool_arguments,
    JSONValidationError,
)


class TestValidateJsonPayload:
    """Test validate_json_payload() function"""

    def test_accepts_valid_json(self):
        """validate_json_payload() should accept valid JSON"""
        schema = {
            "type": "object",
            "properties": {
                "intensity": {"type": "number"},
                "unit": {"type": "string"}
            },
            "required": ["intensity", "unit"]
        }

        payload = '{"intensity": 450.3, "unit": "gCO2/kWh"}'
        obj, error = validate_json_payload(payload, schema)

        assert error is None
        assert obj is not None
        assert obj["intensity"] == 450.3
        assert obj["unit"] == "gCO2/kWh"

    def test_rejects_invalid_json_syntax(self):
        """validate_json_payload() should reject invalid JSON syntax"""
        schema = {"type": "object"}

        payload = '{invalid json syntax}'
        obj, error = validate_json_payload(payload, schema)

        assert obj is None
        assert error is not None
        assert "Invalid JSON syntax" in str(error)

    def test_rejects_schema_violations(self):
        """validate_json_payload() should reject schema violations"""
        schema = {
            "type": "object",
            "properties": {
                "intensity": {"type": "number"},
                "unit": {"type": "string"}
            },
            "required": ["intensity", "unit"]
        }

        # Missing required field
        payload = '{"intensity": 450}'
        obj, error = validate_json_payload(payload, schema)

        assert obj is not None  # JSON parsed successfully
        assert error is not None  # But schema validation failed
        assert "validation failed" in str(error).lower()

    def test_rejects_wrong_type(self):
        """validate_json_payload() should reject wrong types"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }

        payload = '{"count": "not a number"}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None
        assert "validation failed" in str(error).lower()

    def test_accepts_nested_objects(self):
        """validate_json_payload() should accept nested objects"""
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string"}
                    }
                }
            }
        }

        payload = '{"result": {"value": 450, "unit": "kg"}}'
        obj, error = validate_json_payload(payload, schema)

        assert error is None
        assert obj["result"]["value"] == 450

    def test_accepts_arrays(self):
        """validate_json_payload() should accept arrays"""
        schema = {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }

        payload = '{"values": [1, 2, 3, 4.5]}'
        obj, error = validate_json_payload(payload, schema)

        assert error is None
        assert obj["values"] == [1, 2, 3, 4.5]

    def test_rejects_array_type_mismatch(self):
        """validate_json_payload() should reject array type mismatches"""
        schema = {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }

        payload = '{"values": [1, "two", 3]}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None

    def test_empty_object_with_schema(self):
        """validate_json_payload() should validate empty objects"""
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

        payload = '{}'
        obj, error = validate_json_payload(payload, schema)

        assert error is None
        assert obj == {}


class TestValidateJsonObject:
    """Test validate_json_object() function"""

    def test_accepts_valid_dict(self):
        """validate_json_object() should accept valid Python dict"""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "string"}
            }
        }

        obj = {"x": 42, "y": "test"}
        error = validate_json_object(obj, schema)

        assert error is None

    def test_rejects_invalid_dict(self):
        """validate_json_object() should reject invalid dict"""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number"}
            }
        }

        obj = {"x": "not a number"}
        error = validate_json_object(obj, schema)

        assert error is not None
        assert "validation failed" in str(error).lower()

    def test_validates_required_fields(self):
        """validate_json_object() should enforce required fields"""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }

        # Missing required field
        obj = {}
        error = validate_json_object(obj, schema)

        assert error is not None

    def test_validates_additional_properties(self):
        """validate_json_object() should validate additionalProperties"""
        schema = {
            "type": "object",
            "properties": {
                "allowed": {"type": "string"}
            },
            "additionalProperties": False
        }

        # Has disallowed additional property
        obj = {"allowed": "ok", "extra": "not allowed"}
        error = validate_json_object(obj, schema)

        assert error is not None

    def test_validates_enums(self):
        """validate_json_object() should validate enum constraints"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            }
        }

        # Valid enum value
        obj1 = {"status": "active"}
        assert validate_json_object(obj1, schema) is None

        # Invalid enum value
        obj2 = {"status": "pending"}
        assert validate_json_object(obj2, schema) is not None

    def test_validates_numeric_constraints(self):
        """validate_json_object() should validate numeric constraints"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 0, "maximum": 100}
            }
        }

        # Valid
        obj1 = {"count": 50}
        assert validate_json_object(obj1, schema) is None

        # Below minimum
        obj2 = {"count": -5}
        assert validate_json_object(obj2, schema) is not None

        # Above maximum
        obj3 = {"count": 150}
        assert validate_json_object(obj3, schema) is not None


class TestExtractJsonFromText:
    """Test extract_json_from_text() function"""

    def test_extracts_plain_json(self):
        """extract_json_from_text() should extract plain JSON"""
        text = '{"x": 42, "y": "test"}'
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj["x"] == 42
        assert obj["y"] == "test"

    def test_extracts_from_markdown_json_block(self):
        """extract_json_from_text() should extract from ```json block"""
        text = '''
        Here is the result:
        ```json
        {"value": 450, "unit": "kg"}
        ```
        '''
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj["value"] == 450
        assert obj["unit"] == "kg"

    def test_extracts_from_markdown_generic_block(self):
        """extract_json_from_text() should extract from ``` block"""
        text = '''
        ```
        {"result": "success"}
        ```
        '''
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj["result"] == "success"

    def test_extracts_embedded_json_object(self):
        """extract_json_from_text() should extract embedded JSON object"""
        text = 'The result is {"status": "ok"} which means success.'
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj["status"] == "ok"

    def test_extracts_embedded_json_array(self):
        """extract_json_from_text() should extract embedded JSON array"""
        text = 'The values are [1, 2, 3] from the calculation.'
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj == [1, 2, 3]

    def test_returns_none_for_no_json(self):
        """extract_json_from_text() should return None when no JSON found"""
        text = 'This is just plain text without any JSON.'
        obj = extract_json_from_text(text)

        assert obj is None

    def test_handles_multiline_json(self):
        """extract_json_from_text() should handle multiline JSON"""
        text = '''
        ```json
        {
            "name": "test",
            "values": [
                1,
                2,
                3
            ]
        }
        ```
        '''
        obj = extract_json_from_text(text)

        assert obj is not None
        assert obj["name"] == "test"
        assert obj["values"] == [1, 2, 3]

    def test_prefers_first_valid_json(self):
        """extract_json_from_text() should return first valid JSON found"""
        text = '''
        First: {"a": 1}
        Second: {"b": 2}
        '''
        obj = extract_json_from_text(text)

        # Should find one of them (implementation dependent)
        assert obj is not None
        assert "a" in obj or "b" in obj


class TestNormalizeToolArguments:
    """Test normalize_tool_arguments() function"""

    def test_accepts_dict_arguments(self):
        """normalize_tool_arguments() should accept dict arguments"""
        schema = {
            "type": "object",
            "properties": {
                "region": {"type": "string"}
            }
        }

        args = {"region": "CA"}
        result = normalize_tool_arguments(args, schema)

        assert result == {"region": "CA"}

    def test_parses_string_json_arguments(self):
        """normalize_tool_arguments() should parse string JSON arguments"""
        schema = {
            "type": "object",
            "properties": {
                "region": {"type": "string"}
            }
        }

        args = '{"region": "CA"}'
        result = normalize_tool_arguments(args, schema)

        assert result == {"region": "CA"}

    def test_validates_against_schema(self):
        """normalize_tool_arguments() should validate against schema"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }

        # Invalid type
        args = {"count": "not a number"}

        with pytest.raises(JSONValidationError):
            normalize_tool_arguments(args, schema)

    def test_raises_on_invalid_json_string(self):
        """normalize_tool_arguments() should raise on invalid JSON string"""
        schema = {"type": "object"}

        args = '{invalid json}'

        with pytest.raises(JSONValidationError, match="not valid JSON"):
            normalize_tool_arguments(args, schema)

    def test_enforces_required_fields(self):
        """normalize_tool_arguments() should enforce required fields"""
        schema = {
            "type": "object",
            "properties": {
                "required_param": {"type": "string"}
            },
            "required": ["required_param"]
        }

        # Missing required field
        args = {}

        with pytest.raises(JSONValidationError):
            normalize_tool_arguments(args, schema)

    def test_handles_nested_objects(self):
        """normalize_tool_arguments() should handle nested objects"""
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"}
                    }
                }
            }
        }

        args = {"config": {"timeout": 30}}
        result = normalize_tool_arguments(args, schema)

        assert result["config"]["timeout"] == 30

    def test_handles_array_parameters(self):
        """normalize_tool_arguments() should handle array parameters"""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        args = {"items": ["a", "b", "c"]}
        result = normalize_tool_arguments(args, schema)

        assert result["items"] == ["a", "b", "c"]


class TestErrorMessages:
    """Test that error messages include helpful details"""

    def test_json_validation_error_includes_payload(self):
        """JSONValidationError should include payload"""
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}

        payload = '{"x": "wrong type"}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None
        assert error.payload is not None
        assert error.payload["x"] == "wrong type"

    def test_json_validation_error_includes_schema(self):
        """JSONValidationError should include schema"""
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}

        payload = '{"x": "wrong"}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None
        assert error.schema is not None
        assert error.schema["type"] == "object"

    def test_json_validation_error_includes_error_list(self):
        """JSONValidationError should include list of errors"""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }

        # Multiple violations
        payload = '{"a": "wrong"}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None
        assert len(error.errors) > 0

    def test_error_string_shows_first_errors(self):
        """Error string should show first few errors"""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number"}
            }
        }

        payload = '{"x": "wrong"}'
        obj, error = validate_json_payload(payload, schema)

        error_str = str(error)
        assert "Errors:" in error_str or "validation failed" in error_str.lower()

    def test_syntax_error_helpful_message(self):
        """Syntax error should have helpful message"""
        schema = {"type": "object"}

        payload = '{invalid: json, syntax}'
        obj, error = validate_json_payload(payload, schema)

        assert error is not None
        error_str = str(error)
        assert "Invalid JSON syntax" in error_str

    def test_validation_error_includes_message(self):
        """Validation error should include descriptive message"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 0}
            }
        }

        obj = {"count": -5}
        error = validate_json_object(obj, schema)

        assert error is not None
        assert error.message
        assert "validation failed" in error.message.lower()

    def test_multiple_errors_in_complex_schema(self):
        """Should capture multiple errors in complex schema"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age", "email"]
        }

        # Missing required, wrong type
        obj = {"name": 123, "age": "twenty"}

        error = validate_json_object(obj, schema)

        assert error is not None
        assert len(error.errors) > 1  # Multiple violations
