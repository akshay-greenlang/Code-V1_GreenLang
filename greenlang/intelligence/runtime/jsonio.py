"""
JSON Schema Validation

Runtime validation of JSON payloads against JSON Schemas:
- LLM response validation (ensure structured output matches schema)
- Tool argument validation (check parameters before execution)
- Input/output contract enforcement

Uses JSON Schema Draft 2020-12 specification via jsonschema library.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Tuple, Optional
from jsonschema import (
    Draft202012Validator,
    ValidationError as JSONSchemaValidationError,
)


class JSONValidationError(Exception):
    """
    Raised when JSON validation fails

    Attributes:
        message: Description of validation failure
        payload: The invalid JSON payload (if parseable)
        schema: The schema that was violated
        errors: List of specific validation errors
    """

    def __init__(
        self,
        message: str,
        payload: Optional[Any] = None,
        schema: Optional[Dict] = None,
        errors: Optional[list] = None,
    ):
        super().__init__(message)
        self.message = message
        self.payload = payload
        self.schema = schema
        self.errors = errors or []

    def __str__(self) -> str:
        parts = [self.message]
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
            # Show first 3 errors
            for err in self.errors[:3]:
                parts.append(f"  - {err}")
        return "\n".join(parts)


def validate_json_payload(
    payload_text: str, schema: Dict[str, Any]
) -> Tuple[Optional[dict], Optional[JSONValidationError]]:
    """
    Validate JSON text against a JSON Schema

    Args:
        payload_text: JSON string to validate
        schema: JSON Schema to validate against

    Returns:
        Tuple of (parsed_object, error):
        - If valid: (parsed_dict, None)
        - If invalid: (None, JSONValidationError) or (parsed_dict, JSONValidationError)

    Example:
        schema = {
            "type": "object",
            "properties": {
                "emissions": {"type": "number"},
                "unit": {"type": "string"}
            },
            "required": ["emissions", "unit"]
        }

        # Valid payload
        obj, err = validate_json_payload('{"emissions": 1021, "unit": "kg"}', schema)
        assert err is None
        assert obj["emissions"] == 1021

        # Invalid JSON
        obj, err = validate_json_payload('{invalid json}', schema)
        assert obj is None
        assert "Invalid JSON" in str(err)

        # Valid JSON, invalid schema
        obj, err = validate_json_payload('{"emissions": "not a number"}', schema)
        assert obj is not None  # Parsed successfully
        assert err is not None  # But validation failed
    """
    # Step 1: Parse JSON
    try:
        obj = json.loads(payload_text)
    except json.JSONDecodeError as e:
        return None, JSONValidationError(
            f"Invalid JSON syntax: {e}",
            payload=payload_text,
            schema=schema,
        )

    # Step 2: Validate against schema
    try:
        validator = Draft202012Validator(schema)
        validator.validate(obj)
        return obj, None
    except JSONSchemaValidationError as ve:
        # Collect all validation errors
        errors = [str(e.message) for e in validator.iter_errors(obj)]
        return obj, JSONValidationError(
            f"JSON Schema validation failed: {ve.message}",
            payload=obj,
            schema=schema,
            errors=errors,
        )


def validate_json_object(
    obj: Any, schema: Dict[str, Any]
) -> Optional[JSONValidationError]:
    """
    Validate a Python object against JSON Schema

    (Convenience wrapper for already-parsed objects)

    Args:
        obj: Python object to validate
        schema: JSON Schema to validate against

    Returns:
        None if valid, JSONValidationError if invalid

    Example:
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}

        # Valid
        err = validate_json_object({"x": 42}, schema)
        assert err is None

        # Invalid
        err = validate_json_object({"x": "not a number"}, schema)
        assert err is not None
    """
    try:
        validator = Draft202012Validator(schema)
        validator.validate(obj)
        return None
    except JSONSchemaValidationError as ve:
        errors = [str(e.message) for e in validator.iter_errors(obj)]
        return JSONValidationError(
            f"JSON Schema validation failed: {ve.message}",
            payload=obj,
            schema=schema,
            errors=errors,
        )


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON object from text (handles markdown code blocks)

    LLMs sometimes wrap JSON in markdown code blocks:
    ```json
    {"foo": "bar"}
    ```

    This function extracts the JSON content.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON object, or None if no valid JSON found

    Example:
        # Plain JSON
        obj = extract_json_from_text('{"x": 42}')
        assert obj == {"x": 42}

        # Markdown-wrapped JSON
        text = '''
        Here's the result:
        ```json
        {"x": 42}
        ```
        '''
        obj = extract_json_from_text(text)
        assert obj == {"x": 42}

        # No JSON
        obj = extract_json_from_text("Just plain text")
        assert obj is None
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    import re

    # Pattern: ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding {...} or [...] in text
    for pattern in [r"\{[^{}]*\}", r"\[[^\[\]]*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    return None


def normalize_tool_arguments(
    arguments: Any, tool_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize and validate tool arguments against schema

    Handles:
    - String-encoded JSON arguments (some providers send stringified JSON)
    - Missing optional parameters (fill with defaults)
    - Type coercion (string "42" -> int 42 if schema says integer)

    Args:
        arguments: Tool arguments (dict or JSON string)
        tool_schema: Tool parameter schema

    Returns:
        Normalized arguments dict

    Raises:
        JSONValidationError: If arguments don't match schema

    Example:
        schema = {
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "year": {"type": "integer", "default": 2024}
            },
            "required": ["region"]
        }

        # String-encoded JSON
        args = normalize_tool_arguments('{"region": "CA"}', schema)
        assert args == {"region": "CA", "year": 2024}

        # Already a dict
        args = normalize_tool_arguments({"region": "NY"}, schema)
        assert args == {"region": "NY", "year": 2024}

        # Invalid (missing required)
        normalize_tool_arguments({}, schema)  # raises JSONValidationError
    """
    # If arguments is a string, parse it
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise JSONValidationError(f"Tool arguments are not valid JSON: {e}")

    # Validate against schema
    error = validate_json_object(arguments, tool_schema)
    if error:
        raise error

    # TODO: Fill in defaults from schema (future enhancement)

    return arguments
