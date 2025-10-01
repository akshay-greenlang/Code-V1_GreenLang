"""
JSON Schema Type Aliases and Helpers

Provides type-safe JSON Schema handling for:
- Tool parameter validation
- LLM response schema enforcement
- Runtime schema validation

Uses JSON Schema Draft 2020-12 specification.
"""

from __future__ import annotations
from typing import Any, Dict, Union, List

# Type alias for JSON Schema objects
# Pragmatic approach: use Dict[str, Any] with runtime validation
# (more ergonomic than fully typed schema, which would be massive)
JSONSchema = Dict[str, Any]


def is_valid_schema(schema: Any) -> bool:
    """
    Quick validation that an object looks like a JSON Schema

    Checks for required top-level "type" field.
    More rigorous validation happens at runtime via jsonschema library.

    Args:
        schema: Potential JSON Schema object

    Returns:
        True if schema has basic JSON Schema structure

    Example:
        >>> is_valid_schema({"type": "object", "properties": {...}})
        True
        >>> is_valid_schema({"foo": "bar"})
        False
    """
    if not isinstance(schema, dict):
        return False
    return "type" in schema


def make_object_schema(
    properties: Dict[str, JSONSchema],
    required: List[str] | None = None,
    additional_properties: bool = False,
) -> JSONSchema:
    """
    Helper to construct object-type JSON Schemas

    Args:
        properties: Property name -> schema mapping
        required: List of required property names
        additional_properties: Allow extra properties?

    Returns:
        JSON Schema object

    Example:
        >>> make_object_schema(
        ...     properties={
        ...         "region": {"type": "string"},
        ...         "year": {"type": "integer", "minimum": 2000}
        ...     },
        ...     required=["region"]
        ... )
        {
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "year": {"type": "integer", "minimum": 2000}
            },
            "required": ["region"],
            "additionalProperties": False
        }
    """
    schema: JSONSchema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    return schema


def make_string_schema(
    description: str | None = None,
    enum: List[str] | None = None,
    pattern: str | None = None,
) -> JSONSchema:
    """
    Helper to construct string-type JSON Schemas

    Args:
        description: Human-readable description
        enum: Allowed values (for categorical strings)
        pattern: Regex pattern for validation

    Returns:
        JSON Schema for string type

    Example:
        >>> make_string_schema(
        ...     description="Region code",
        ...     enum=["CA", "NY", "TX"]
        ... )
        {
            "type": "string",
            "description": "Region code",
            "enum": ["CA", "NY", "TX"]
        }
    """
    schema: JSONSchema = {"type": "string"}
    if description:
        schema["description"] = description
    if enum:
        schema["enum"] = enum
    if pattern:
        schema["pattern"] = pattern
    return schema


def make_number_schema(
    description: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
    exclusive_maximum: float | None = None,
) -> JSONSchema:
    """
    Helper to construct number-type JSON Schemas

    Args:
        description: Human-readable description
        minimum: Minimum value (inclusive)
        maximum: Maximum value (inclusive)
        exclusive_minimum: Minimum value (exclusive)
        exclusive_maximum: Maximum value (exclusive)

    Returns:
        JSON Schema for number type

    Example:
        >>> make_number_schema(
        ...     description="Fuel amount in gallons",
        ...     minimum=0,
        ...     exclusive_minimum=0
        ... )
        {
            "type": "number",
            "description": "Fuel amount in gallons",
            "minimum": 0,
            "exclusiveMinimum": 0
        }
    """
    schema: JSONSchema = {"type": "number"}
    if description:
        schema["description"] = description
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    if exclusive_maximum is not None:
        schema["exclusiveMaximum"] = exclusive_maximum
    return schema


def make_integer_schema(
    description: str | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> JSONSchema:
    """
    Helper to construct integer-type JSON Schemas

    Args:
        description: Human-readable description
        minimum: Minimum value (inclusive)
        maximum: Maximum value (inclusive)

    Returns:
        JSON Schema for integer type

    Example:
        >>> make_integer_schema(
        ...     description="Year (2000-2030)",
        ...     minimum=2000,
        ...     maximum=2030
        ... )
        {
            "type": "integer",
            "description": "Year (2000-2030)",
            "minimum": 2000,
            "maximum": 2030
        }
    """
    schema: JSONSchema = {"type": "integer"}
    if description:
        schema["description"] = description
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def make_array_schema(
    items: JSONSchema,
    description: str | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> JSONSchema:
    """
    Helper to construct array-type JSON Schemas

    Args:
        items: Schema for array items
        description: Human-readable description
        min_items: Minimum array length
        max_items: Maximum array length

    Returns:
        JSON Schema for array type

    Example:
        >>> make_array_schema(
        ...     items={"type": "string"},
        ...     description="List of region codes",
        ...     min_items=1
        ... )
        {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of region codes",
            "minItems": 1
        }
    """
    schema: JSONSchema = {"type": "array", "items": items}
    if description:
        schema["description"] = description
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema
