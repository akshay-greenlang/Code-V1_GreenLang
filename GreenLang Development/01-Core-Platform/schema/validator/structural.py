# -*- coding: utf-8 -*-
"""
Structural Validator for GL-FOUND-X-002.

This module implements structural validation (shape, types, required fields)
against compiled schema IR. It validates payload structure before constraint
validation occurs.

Validations:
    - Required field presence (including nested)
    - Type checking (string, number, integer, boolean, object, array, null)
    - Additional properties policy (error/warn/ignore based on profile)
    - Property count constraints (minProperties, maxProperties)
    - Null handling

Error Codes:
    - GLSCHEMA-E100: MISSING_REQUIRED - Required field missing
    - GLSCHEMA-E101: UNKNOWN_FIELD - Unknown field in strict mode
    - GLSCHEMA-E102: TYPE_MISMATCH - Type does not match schema
    - GLSCHEMA-E103: INVALID_NULL - Null not allowed
    - GLSCHEMA-E105: PROPERTY_COUNT_VIOLATION - Property count outside range

Example:
    >>> from greenlang.schema.validator.structural import StructuralValidator
    >>> from greenlang.schema.validator.structural import validate_structure
    >>> validator = StructuralValidator(ir, options)
    >>> findings = validator.validate(payload)
    >>> # Or use convenience function
    >>> findings = validate_structure(payload, ir)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.1
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from ..compiler.ir import PropertyIR, SchemaIR
from ..errors import ErrorCode
from ..models.config import (
    UnknownFieldPolicy,
    ValidationOptions,
    ValidationProfile,
)
from ..models.finding import Finding, FindingHint, Severity

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE MAPPING
# =============================================================================

# Python type to JSON Schema type mapping
PYTHON_TO_JSON_TYPE: Dict[type, str] = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}

# JSON Schema type compatibility rules
# "number" accepts both int and float, "integer" only accepts int
TYPE_COMPATIBILITY: Dict[str, Set[str]] = {
    "string": {"string"},
    "number": {"number", "integer"},  # number accepts integer
    "integer": {"integer"},
    "boolean": {"boolean"},
    "object": {"object"},
    "array": {"array"},
    "null": {"null"},
}


# =============================================================================
# STRUCTURAL VALIDATOR
# =============================================================================


class StructuralValidator:
    """
    Structural validator for payload shape and types.

    Validates that the payload matches the expected structure defined
    in the schema IR, including type checking, required fields, and
    additional properties handling.

    Follows the zero-hallucination principle by using deterministic
    validation logic without LLM inference for structural checks.

    Attributes:
        ir: Compiled schema Intermediate Representation
        options: Validation options controlling strictness
        _findings: Internal list of findings accumulated during validation

    Example:
        >>> from greenlang.schema.compiler.ir import SchemaIR
        >>> from greenlang.schema.models.config import ValidationOptions
        >>> ir = SchemaIR(schema_id="test", version="1.0", schema_hash="a" * 64,
        ...              compiled_at=datetime.now())
        >>> validator = StructuralValidator(ir, ValidationOptions())
        >>> findings = validator.validate({"energy": 100})
        >>> for f in findings:
        ...     print(f"{f.code}: {f.message}")
    """

    def __init__(self, ir: SchemaIR, options: ValidationOptions):
        """
        Initialize the structural validator.

        Args:
            ir: Compiled schema IR containing property definitions
            options: Validation options controlling behavior

        Raises:
            ValueError: If ir or options is None
        """
        if ir is None:
            raise ValueError("SchemaIR cannot be None")
        if options is None:
            raise ValueError("ValidationOptions cannot be None")

        self.ir = ir
        self.options = options
        self._findings: List[Finding] = []
        self._validated_paths: Set[str] = set()

        logger.debug(
            "StructuralValidator initialized with schema_id=%s, profile=%s",
            ir.schema_id,
            options.profile.value,
        )

    def validate(
        self,
        payload: Dict[str, Any],
        path: str = "",
    ) -> List[Finding]:
        """
        Validate payload structure against schema IR.

        Performs comprehensive structural validation including:
        - Required field presence (including nested objects)
        - Type checking for all JSON types
        - Additional properties policy enforcement
        - Property count constraints (minProperties/maxProperties)
        - Null value handling

        Args:
            payload: The payload to validate (must be a dict for root)
            path: Current JSON Pointer path (empty string for root)

        Returns:
            List of validation findings (errors and warnings)

        Raises:
            ValueError: If payload is not a dict at root level

        Example:
            >>> findings = validator.validate({"name": "test", "value": 42})
            >>> if any(f.is_error() for f in findings):
            ...     print("Validation failed")
        """
        start_time = datetime.now()
        self._findings = []
        self._validated_paths = set()

        logger.info(
            "Starting structural validation for schema=%s at path='%s'",
            self.ir.schema_id,
            path if path else "(root)",
        )

        # Root must be an object
        if not isinstance(payload, dict):
            self._add_finding(
                code=ErrorCode.TYPE_MISMATCH.value,
                severity=Severity.ERROR,
                path=path if path else "/",
                message=f"Root payload must be an object, got {self._get_type_name(payload)}",
                expected={"type": "object"},
                actual=self._get_type_name(payload),
            )
            return self._findings

        # Validate the root object
        self._validate_object(payload, path)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Structural validation complete: %d findings in %.2fms",
            len(self._findings),
            elapsed_ms,
        )

        return self._findings

    def _validate_object(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> None:
        """
        Validate an object value against schema IR.

        Performs the following checks in order:
        1. Property count constraints
        2. Required fields presence
        3. Type validation for each property
        4. Additional properties policy
        5. Recursive validation for nested objects/arrays

        Args:
            obj: The object to validate
            path: JSON Pointer path to this object
        """
        if not isinstance(obj, dict):
            self._add_finding(
                code=ErrorCode.TYPE_MISMATCH.value,
                severity=Severity.ERROR,
                path=path,
                message=f"Expected object but got {self._get_type_name(obj)}",
                expected={"type": "object"},
                actual=self._get_type_name(obj),
            )
            return

        # Check fail_fast option
        if self.options.fail_fast and self._has_errors():
            logger.debug("Fail-fast triggered at path='%s'", path)
            return

        # Check max_errors limit
        if self._exceeds_max_errors():
            return

        # Step 1: Validate property count constraints
        self._validate_property_count(obj, path)

        # Step 2: Validate required fields
        self._validate_required_fields(obj, path)

        # Step 3: Validate known properties and their types
        self._validate_known_properties(obj, path)

        # Step 4: Check for additional (unknown) properties
        self._validate_additional_properties(obj, path)

    def _validate_required_fields(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> None:
        """
        Check all required fields are present.

        Validates that all fields marked as required in the schema IR
        are present in the payload. Handles both root-level and nested
        required fields.

        Args:
            obj: The object to validate
            path: JSON Pointer path to this object
        """
        # Get required paths that are immediate children of current path
        for required_path in self.ir.required_paths:
            # Check if this required path is a direct child of current path
            if self._is_direct_child(path, required_path):
                field_name = self._get_field_name_from_path(required_path)

                if field_name not in obj:
                    # Get property info for better error messages
                    prop_ir = self.ir.get_property(required_path)
                    expected_type = prop_ir.type if prop_ir else "any"

                    self._add_finding(
                        code=ErrorCode.MISSING_REQUIRED.value,
                        severity=Severity.ERROR,
                        path=required_path,
                        message=f"Required field '{field_name}' is missing",
                        expected={"type": expected_type, "required": True},
                        actual=None,
                        hint=FindingHint(
                            category="missing_required",
                            suggested_values=[],
                            docs_url=None,
                        ),
                    )

                    if self.options.fail_fast:
                        return

    def _validate_known_properties(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> None:
        """
        Validate properties that are defined in the schema.

        For each property in the payload that has a corresponding
        definition in the schema IR, validates:
        - Type matches expected type
        - Null handling if value is None
        - Recursive validation for nested structures

        Args:
            obj: The object to validate
            path: JSON Pointer path to this object
        """
        for key, value in obj.items():
            if self._exceeds_max_errors():
                return

            prop_path = f"{path}/{key}" if path else f"/{key}"
            prop_ir = self.ir.get_property(prop_path)

            if prop_ir is not None:
                # This is a known property, validate its type
                self._validate_property_value(value, prop_ir, prop_path)

                # Mark this path as validated
                self._validated_paths.add(prop_path)

            # Recursively validate nested objects and arrays regardless
            # of whether they're in the schema (for deep validation)
            if isinstance(value, dict):
                self._validate_object(value, prop_path)
            elif isinstance(value, list):
                self._validate_array(value, prop_path)

    def _validate_property_value(
        self,
        value: Any,
        prop_ir: PropertyIR,
        path: str,
    ) -> None:
        """
        Validate a single property value against its PropertyIR.

        Args:
            value: The value to validate
            prop_ir: Property IR containing type information
            path: JSON Pointer path to this property
        """
        # Handle null values
        if value is None:
            # Check if null is allowed for this property
            if prop_ir.type and prop_ir.type != "null":
                self._add_finding(
                    code=ErrorCode.INVALID_NULL.value,
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Null value not allowed, expected {prop_ir.type}",
                    expected={"type": prop_ir.type, "nullable": False},
                    actual="null",
                    hint=FindingHint(
                        category="null_not_allowed",
                        suggested_values=[],
                        docs_url=None,
                    ),
                )
            return

        # Validate type
        if prop_ir.type:
            if not self._validate_type(value, prop_ir.type, path):
                return  # Type mismatch already recorded

    def _validate_type(
        self,
        value: Any,
        expected_type: str,
        path: str,
    ) -> bool:
        """
        Validate value matches expected type.

        Supports JSON Schema types:
        - string
        - number (accepts both int and float)
        - integer (accepts only int)
        - boolean
        - object
        - array
        - null

        Args:
            value: The value to check
            expected_type: Expected JSON Schema type
            path: JSON Pointer path for error reporting

        Returns:
            True if type matches, False otherwise
        """
        actual_type = self._get_type_name(value)

        # Handle union types (e.g., "string,null" or ["string", "null"])
        if "," in expected_type:
            allowed_types = [t.strip() for t in expected_type.split(",")]
        else:
            allowed_types = [expected_type]

        # Check if actual type is compatible with any allowed type
        for allowed in allowed_types:
            compatible_types = TYPE_COMPATIBILITY.get(allowed, {allowed})
            if actual_type in compatible_types:
                return True

        # Type mismatch
        self._add_finding(
            code=ErrorCode.TYPE_MISMATCH.value,
            severity=Severity.ERROR,
            path=path,
            message=f"Expected type '{expected_type}' but found '{actual_type}'",
            expected={"type": expected_type},
            actual=actual_type,
            hint=FindingHint(
                category="type_mismatch",
                suggested_values=[],
                docs_url=None,
            ),
        )
        return False

    def _validate_additional_properties(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> None:
        """
        Handle unknown/additional properties based on profile.

        Behavior depends on validation profile and unknown_field_policy:
        - STRICT profile: Always errors on unknown fields
        - STANDARD profile: Uses unknown_field_policy setting
        - PERMISSIVE profile: Ignores unknown fields

        Args:
            obj: The object to validate
            path: Current JSON Pointer path
        """
        # Get all known property names for this object level
        known_properties = self._get_known_properties_at_path(path)

        for key in obj.keys():
            prop_path = f"{path}/{key}" if path else f"/{key}"

            # Check if this property is known
            if key not in known_properties and prop_path not in self.ir.properties:
                self._handle_unknown_property(key, prop_path, known_properties)

    def _handle_unknown_property(
        self,
        key: str,
        path: str,
        known_properties: Set[str],
    ) -> None:
        """
        Handle an unknown property based on current policy.

        Args:
            key: The unknown property name
            path: JSON Pointer path to the property
            known_properties: Set of known property names at this level
        """
        # Determine effective policy based on profile
        if self.options.profile == ValidationProfile.STRICT:
            effective_policy = UnknownFieldPolicy.ERROR
        elif self.options.profile == ValidationProfile.PERMISSIVE:
            effective_policy = UnknownFieldPolicy.IGNORE
        else:
            effective_policy = self.options.unknown_field_policy

        if effective_policy == UnknownFieldPolicy.IGNORE:
            logger.debug("Ignoring unknown field '%s' at path='%s'", key, path)
            return

        severity = (
            Severity.ERROR
            if effective_policy == UnknownFieldPolicy.ERROR
            else Severity.WARNING
        )

        # Create finding for unknown field
        self._add_finding(
            code=ErrorCode.UNKNOWN_FIELD.value,
            severity=severity,
            path=path,
            message=f"Unknown field '{key}' is not defined in schema",
            expected={"known_fields": list(known_properties)[:10]},  # Limit list size
            actual=key,
            hint=FindingHint(
                category="unknown_field",
                suggested_values=list(known_properties)[:5],
                docs_url=None,
            ),
        )

    def _validate_property_count(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> None:
        """
        Validate object property count is within bounds.

        Checks minProperties and maxProperties constraints from the
        schema IR if defined for this path.

        Args:
            obj: The object to validate
            path: Current JSON Pointer path
        """
        prop_count = len(obj)

        # Check if there are property count constraints in the IR
        # We need to look for constraints at this object level
        # This is typically stored in the IR or derived from schema

        # For root level, check if schema defines minProperties/maxProperties
        # This would be stored in a dedicated field or derived from PropertyIR

        # Get the property IR for this path (if it represents an object type)
        prop_ir = self.ir.get_property(path) if path else None

        # Also check for root-level constraints
        min_props = None
        max_props = None

        if prop_ir and prop_ir.gl_extensions:
            min_props = prop_ir.gl_extensions.get("minProperties")
            max_props = prop_ir.gl_extensions.get("maxProperties")

        # Validate minProperties
        if min_props is not None and prop_count < min_props:
            self._add_finding(
                code=ErrorCode.PROPERTY_COUNT_VIOLATION.value,
                severity=Severity.ERROR,
                path=path if path else "/",
                message=f"Object has {prop_count} properties, minimum required is {min_props}",
                expected={"minProperties": min_props},
                actual=prop_count,
                hint=FindingHint(
                    category="property_count",
                    suggested_values=[],
                    docs_url=None,
                ),
            )

        # Validate maxProperties
        if max_props is not None and prop_count > max_props:
            self._add_finding(
                code=ErrorCode.PROPERTY_COUNT_VIOLATION.value,
                severity=Severity.ERROR,
                path=path if path else "/",
                message=f"Object has {prop_count} properties, maximum allowed is {max_props}",
                expected={"maxProperties": max_props},
                actual=prop_count,
                hint=FindingHint(
                    category="property_count",
                    suggested_values=[],
                    docs_url=None,
                ),
            )

    def _validate_array(
        self,
        arr: List[Any],
        path: str,
        items_type: Optional[str] = None,
    ) -> None:
        """
        Validate an array value.

        Validates array items recursively if they are objects or arrays.
        Item type validation is handled separately by the constraint validator.

        Args:
            arr: The array to validate
            path: JSON Pointer path to this array
            items_type: Optional type for array items
        """
        if not isinstance(arr, list):
            self._add_finding(
                code=ErrorCode.TYPE_MISMATCH.value,
                severity=Severity.ERROR,
                path=path,
                message=f"Expected array but got {self._get_type_name(arr)}",
                expected={"type": "array"},
                actual=self._get_type_name(arr),
            )
            return

        # Get property IR to check for items type constraint
        prop_ir = self.ir.get_property(path)
        effective_items_type = items_type

        if prop_ir and prop_ir.gl_extensions:
            effective_items_type = prop_ir.gl_extensions.get("items_type", items_type)

        # Validate each item
        for index, item in enumerate(arr):
            if self._exceeds_max_errors():
                return

            item_path = f"{path}/{index}"

            # Type check if items_type is specified
            if effective_items_type:
                self._validate_type(item, effective_items_type, item_path)

            # Handle null items
            if item is None:
                # Check if null is allowed in array items
                if effective_items_type and effective_items_type != "null":
                    self._add_finding(
                        code=ErrorCode.INVALID_NULL.value,
                        severity=Severity.ERROR,
                        path=item_path,
                        message="Null value not allowed in array items",
                        expected={"type": effective_items_type},
                        actual="null",
                    )
                continue

            # Recursively validate nested structures
            if isinstance(item, dict):
                self._validate_object(item, item_path)
            elif isinstance(item, list):
                self._validate_array(item, item_path)

    def _add_finding(
        self,
        code: str,
        severity: Severity,
        path: str,
        message: str,
        expected: Any = None,
        actual: Any = None,
        hint: Optional[FindingHint] = None,
    ) -> None:
        """
        Add a validation finding to the internal list.

        Args:
            code: Error code (GLSCHEMA-*)
            severity: Severity level (error/warning/info)
            path: JSON Pointer path where issue occurred
            message: Human-readable error message
            expected: What was expected (for debugging)
            actual: What was found (for debugging)
            hint: Optional hint for resolution
        """
        # Check if we've exceeded max errors
        if self._exceeds_max_errors():
            return

        # Convert expected to dict if not already
        expected_dict = None
        if expected is not None:
            if isinstance(expected, dict):
                expected_dict = expected
            else:
                expected_dict = {"value": expected}

        finding = Finding(
            code=code,
            severity=severity,
            path=path if path else "/",
            message=message,
            expected=expected_dict,
            actual=actual,
            hint=hint,
        )

        self._findings.append(finding)

        # Log the finding
        log_level = (
            logging.ERROR
            if severity == Severity.ERROR
            else logging.WARNING
            if severity == Severity.WARNING
            else logging.INFO
        )
        logger.log(log_level, "%s at %s: %s", code, path, message)

    def _get_type_name(self, value: Any) -> str:
        """
        Get JSON Schema type name for a Python value.

        Args:
            value: Python value to get type name for

        Returns:
            JSON Schema type name (string, number, integer, boolean, object, array, null)

        Examples:
            >>> validator._get_type_name("hello")
            'string'
            >>> validator._get_type_name(42)
            'integer'
            >>> validator._get_type_name(3.14)
            'number'
            >>> validator._get_type_name(None)
            'null'
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            # Must check bool before int because bool is subclass of int
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"

    def _get_known_properties_at_path(self, path: str) -> Set[str]:
        """
        Get set of known property names at a given path level.

        Args:
            path: JSON Pointer path to get properties for

        Returns:
            Set of property names known at this level
        """
        known = set()
        prefix = f"{path}/" if path else "/"

        for prop_path in self.ir.properties.keys():
            if prop_path.startswith(prefix):
                # Extract the immediate child property name
                remainder = prop_path[len(prefix):]
                if "/" not in remainder:
                    known.add(remainder)

        return known

    def _is_direct_child(self, parent_path: str, child_path: str) -> bool:
        """
        Check if child_path is a direct child of parent_path.

        Args:
            parent_path: Parent JSON Pointer path
            child_path: Potential child path

        Returns:
            True if child is a direct child of parent
        """
        if not parent_path:
            # Root level - child should be /name (one level)
            if child_path.startswith("/"):
                remainder = child_path[1:]
                return "/" not in remainder
            return False

        prefix = f"{parent_path}/"
        if child_path.startswith(prefix):
            remainder = child_path[len(prefix):]
            return "/" not in remainder
        return False

    def _get_field_name_from_path(self, path: str) -> str:
        """
        Extract field name from JSON Pointer path.

        Args:
            path: JSON Pointer path (e.g., "/parent/child")

        Returns:
            Field name (last component of path)
        """
        if not path:
            return ""
        parts = path.split("/")
        return parts[-1] if parts else ""

    def _has_errors(self) -> bool:
        """Check if any error-level findings exist."""
        return any(f.severity == Severity.ERROR for f in self._findings)

    def _exceeds_max_errors(self) -> bool:
        """Check if max_errors limit has been reached."""
        if self.options.max_errors == 0:
            return False  # Unlimited
        error_count = sum(1 for f in self._findings if f.severity == Severity.ERROR)
        return error_count >= self.options.max_errors


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def validate_structure(
    payload: Dict[str, Any],
    ir: SchemaIR,
    options: Optional[ValidationOptions] = None,
) -> List[Finding]:
    """
    Convenience function for structural validation.

    Creates a StructuralValidator and validates the payload against
    the provided schema IR.

    Args:
        payload: The payload to validate
        ir: Compiled schema IR
        options: Validation options (uses defaults if None)

    Returns:
        List of validation findings

    Example:
        >>> from greenlang.schema.validator.structural import validate_structure
        >>> ir = compile_schema(schema_source)
        >>> findings = validate_structure({"name": "test"}, ir)
        >>> for f in findings:
        ...     print(f"{f.code}: {f.message}")
    """
    if options is None:
        options = ValidationOptions()

    validator = StructuralValidator(ir, options)
    return validator.validate(payload)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "StructuralValidator",
    "validate_structure",
    "PYTHON_TO_JSON_TYPE",
    "TYPE_COMPATIBILITY",
]
