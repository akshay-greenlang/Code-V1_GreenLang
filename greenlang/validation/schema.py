"""
GreenLang Schema Validator
JSON Schema-based validation with extended support.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import logging
import json

from .framework import ValidationResult, ValidationError, ValidationSeverity

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import Draft7Validator, validators
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not available, schema validation will be limited")


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors."""
    pass


class SchemaValidator:
    """
    JSON Schema validator with enhanced error reporting.

    Supports JSON Schema Draft 7 specification.

    Example:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }

        validator = SchemaValidator(schema)
        result = validator.validate({"name": "John", "age": 30})
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize schema validator.

        Args:
            schema: JSON Schema dictionary
        """
        self.schema = schema

        if JSONSCHEMA_AVAILABLE:
            # Create validator with format checking
            validator_cls = validators.create(
                meta_schema=Draft7Validator.META_SCHEMA,
                validators=Draft7Validator.VALIDATORS,
            )
            self.validator = validator_cls(schema)
        else:
            self.validator = None
            logger.warning("jsonschema not available, using basic validation")

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with errors if any
        """
        result = ValidationResult(valid=True)

        if not JSONSCHEMA_AVAILABLE:
            # Basic type checking without jsonschema
            result = self._basic_validation(data)
        else:
            # Full JSON Schema validation
            errors = list(self.validator.iter_errors(data))

            for error in errors:
                # Extract field path
                field_path = ".".join(str(p) for p in error.path) if error.path else "root"

                validation_error = ValidationError(
                    field=field_path,
                    message=error.message,
                    severity=ValidationSeverity.ERROR,
                    validator="json_schema",
                    value=error.instance if hasattr(error, 'instance') else None,
                    expected=error.schema if hasattr(error, 'schema') else None,
                    location=f"$.{field_path}"
                )
                result.add_error(validation_error)

        return result

    def _basic_validation(self, data: Any) -> ValidationResult:
        """
        Basic validation without jsonschema library.

        Args:
            data: Data to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        schema_type = self.schema.get("type")

        # Check type
        if schema_type:
            valid_type = self._check_type(data, schema_type)
            if not valid_type:
                error = ValidationError(
                    field="root",
                    message=f"Expected type {schema_type}, got {type(data).__name__}",
                    severity=ValidationSeverity.ERROR,
                    validator="basic_schema",
                    value=data,
                    expected=schema_type
                )
                result.add_error(error)
                return result

        # Check required properties for objects
        if schema_type == "object" and isinstance(data, dict):
            required = self.schema.get("required", [])
            for field in required:
                if field not in data:
                    error = ValidationError(
                        field=field,
                        message=f"Required field '{field}' is missing",
                        severity=ValidationSeverity.ERROR,
                        validator="basic_schema",
                        expected="required"
                    )
                    result.add_error(error)

        return result

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True

        return isinstance(value, expected_python_type)

    @classmethod
    def from_file(cls, schema_path: str) -> "SchemaValidator":
        """
        Create validator from JSON schema file.

        Args:
            schema_path: Path to JSON schema file

        Returns:
            SchemaValidator instance
        """
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return cls(schema)
