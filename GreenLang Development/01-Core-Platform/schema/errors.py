"""
GreenLang Schema Compiler & Validator Error Code Taxonomy.

This module defines all error codes used by GL-FOUND-X-002 (GreenLang Schema
Compiler & Validator). Error codes follow the GLSCHEMA-* prefix convention
and are organized into categories for easy reference and filtering.

Error Code Structure:
    GLSCHEMA-{SEVERITY}{CATEGORY}{NUMBER}

    Where:
    - SEVERITY: E=Error, W=Warning
    - CATEGORY: 1xx=Structural, 2xx=Constraint, 3xx=Unit, 4xx=Rule,
                5xx=Schema, 6xx=Deprecation, 7xx=Lint, 8xx=Limit
    - NUMBER: Specific error within category (00-99)

Example:
    >>> from greenlang.schema.errors import ErrorCode, get_error_info
    >>> info = get_error_info(ErrorCode.MISSING_REQUIRED)
    >>> print(f"{info.code}: {info.message_template}")
    GLSCHEMA-E100: Required field '{field}' is missing at path '{path}'

Usage:
    >>> from greenlang.schema.errors import format_error_message, ErrorCode
    >>> msg = format_error_message(
    ...     ErrorCode.TYPE_MISMATCH,
    ...     path="/data/value",
    ...     expected="integer",
    ...     actual="string"
    ... )
    >>> print(msg)
    Expected type 'integer' but found 'string' at path '/data/value'

Author: GreenLang Team
Date: 2026-01-28
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Severity(str, Enum):
    """Severity levels for validation findings.

    Attributes:
        ERROR: Validation failure that must be fixed. Causes validation to fail.
        WARNING: Potential issue that should be reviewed. Does not fail validation
            unless --fail-on-warnings is specified.
        INFO: Informational message for awareness. Never fails validation.
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(str, Enum):
    """Categories for organizing error codes.

    Each category represents a distinct type of validation concern with
    its own error code range.

    Attributes:
        STRUCTURAL: Shape and type validation (E1xx)
        CONSTRAINT: Value constraint validation (E2xx)
        UNIT: Unit and dimension validation (E3xx)
        RULE: Cross-field rule validation (E4xx)
        SCHEMA: Schema document validation (E5xx)
        DEPRECATION: Deprecation warnings (W6xx)
        LINT: Style and naming warnings (W7xx)
        LIMIT: Size and complexity limits (E8xx)
    """
    STRUCTURAL = "structural"
    CONSTRAINT = "constraint"
    UNIT = "unit"
    RULE = "rule"
    SCHEMA = "schema"
    DEPRECATION = "deprecation"
    LINT = "lint"
    LIMIT = "limit"


class ErrorCode(str, Enum):
    """All GLSCHEMA-* error codes for schema validation.

    Error codes are stable identifiers that should not change once released.
    Each code maps to a specific validation condition with a standardized
    message template.

    Structural Errors (GLSCHEMA-E1xx):
        Errors related to payload shape, types, and required fields.

    Constraint Errors (GLSCHEMA-E2xx):
        Errors related to value constraints like ranges, patterns, and enums.

    Unit Errors (GLSCHEMA-E3xx):
        Errors related to unit validation and dimensional analysis.

    Rule Errors (GLSCHEMA-E4xx):
        Errors related to cross-field validation rules.

    Schema Errors (GLSCHEMA-E5xx):
        Errors in the schema document itself.

    Deprecation Warnings (GLSCHEMA-W6xx):
        Warnings about deprecated or renamed fields.

    Lint Warnings (GLSCHEMA-W7xx):
        Style and naming convention warnings.

    Limit Errors (GLSCHEMA-E8xx):
        Errors when payload exceeds configured limits.
    """

    # =========================================================================
    # Structural Errors (GLSCHEMA-E1xx) - Shape and type validation
    # =========================================================================

    # E100: Required field is missing
    MISSING_REQUIRED = "GLSCHEMA-E100"

    # E101: Unknown field found in strict validation mode
    UNKNOWN_FIELD = "GLSCHEMA-E101"

    # E102: Value type does not match schema type
    TYPE_MISMATCH = "GLSCHEMA-E102"

    # E103: Null value where null is not allowed
    INVALID_NULL = "GLSCHEMA-E103"

    # E104: Object found where array expected, or vice versa
    CONTAINER_TYPE_MISMATCH = "GLSCHEMA-E104"

    # E105: Property count outside allowed range
    PROPERTY_COUNT_VIOLATION = "GLSCHEMA-E105"

    # E106: Required properties missing from object
    REQUIRED_PROPERTIES_MISSING = "GLSCHEMA-E106"

    # E107: Duplicate key in object (after alias resolution)
    DUPLICATE_KEY = "GLSCHEMA-E107"

    # =========================================================================
    # Constraint Errors (GLSCHEMA-E2xx) - Value constraints
    # =========================================================================

    # E200: Numeric value outside min/max range
    RANGE_VIOLATION = "GLSCHEMA-E200"

    # E201: String does not match required pattern
    PATTERN_MISMATCH = "GLSCHEMA-E201"

    # E202: Value not in allowed enum values
    ENUM_VIOLATION = "GLSCHEMA-E202"

    # E203: String or array length outside allowed range
    LENGTH_VIOLATION = "GLSCHEMA-E203"

    # E204: Array items are not unique when uniqueItems=true
    UNIQUE_VIOLATION = "GLSCHEMA-E204"

    # E205: Value not a multiple of required value
    MULTIPLE_OF_VIOLATION = "GLSCHEMA-E205"

    # E206: String format validation failed (date, email, uri, etc.)
    FORMAT_VIOLATION = "GLSCHEMA-E206"

    # E207: Const value mismatch
    CONST_VIOLATION = "GLSCHEMA-E207"

    # E208: Array does not contain required item
    CONTAINS_VIOLATION = "GLSCHEMA-E208"

    # E209: Property name does not match pattern
    PROPERTY_NAME_VIOLATION = "GLSCHEMA-E209"

    # =========================================================================
    # Unit Errors (GLSCHEMA-E3xx) - Unit validation
    # =========================================================================

    # E300: Required unit not provided with numeric value
    UNIT_MISSING = "GLSCHEMA-E300"

    # E301: Unit dimension does not match expected dimension
    UNIT_INCOMPATIBLE = "GLSCHEMA-E301"

    # E302: Unit is valid but not in canonical form
    UNIT_NONCANONICAL = "GLSCHEMA-E302"

    # E303: Unit not found in the unit catalog
    UNIT_UNKNOWN = "GLSCHEMA-E303"

    # E304: Unit conversion failed
    UNIT_CONVERSION_FAILED = "GLSCHEMA-E304"

    # E305: Value with unit has invalid format
    UNIT_FORMAT_INVALID = "GLSCHEMA-E305"

    # E306: Dimension specification invalid
    DIMENSION_INVALID = "GLSCHEMA-E306"

    # =========================================================================
    # Rule Errors (GLSCHEMA-E4xx) - Cross-field validation
    # =========================================================================

    # E400: Cross-field validation rule failed
    RULE_VIOLATION = "GLSCHEMA-E400"

    # E401: Conditional requirement not satisfied
    CONDITIONAL_REQUIRED = "GLSCHEMA-E401"

    # E402: Consistency check failed (e.g., sum != total)
    CONSISTENCY_ERROR = "GLSCHEMA-E402"

    # E403: Dependency requirement not met
    DEPENDENCY_VIOLATION = "GLSCHEMA-E403"

    # E404: Mutual exclusion constraint violated
    MUTUAL_EXCLUSION_VIOLATION = "GLSCHEMA-E404"

    # E405: OneOf constraint violated (multiple schemas match)
    ONE_OF_VIOLATION = "GLSCHEMA-E405"

    # E406: AnyOf constraint violated (no schemas match)
    ANY_OF_VIOLATION = "GLSCHEMA-E406"

    # E407: AllOf constraint violated (not all schemas match)
    ALL_OF_VIOLATION = "GLSCHEMA-E407"

    # E408: Not constraint violated (schema matches when it should not)
    NOT_VIOLATION = "GLSCHEMA-E408"

    # E409: If-then-else constraint violated
    IF_THEN_ELSE_VIOLATION = "GLSCHEMA-E409"

    # =========================================================================
    # Schema Errors (GLSCHEMA-E5xx) - Schema document validation
    # =========================================================================

    # E500: $ref target could not be resolved
    REF_RESOLUTION_FAILED = "GLSCHEMA-E500"

    # E501: Circular reference detected in schema
    CIRCULAR_REF = "GLSCHEMA-E501"

    # E502: Schema document itself is invalid
    SCHEMA_INVALID = "GLSCHEMA-E502"

    # E503: Schema version incompatible with validator
    SCHEMA_VERSION_MISMATCH = "GLSCHEMA-E503"

    # E504: Schema dialect not supported
    SCHEMA_DIALECT_UNSUPPORTED = "GLSCHEMA-E504"

    # E505: Schema definition not found
    SCHEMA_DEFINITION_NOT_FOUND = "GLSCHEMA-E505"

    # E506: Schema registry lookup failed
    SCHEMA_REGISTRY_ERROR = "GLSCHEMA-E506"

    # E507: Schema parsing failed
    SCHEMA_PARSE_ERROR = "GLSCHEMA-E507"

    # E508: Schema keyword is invalid or unsupported
    SCHEMA_KEYWORD_INVALID = "GLSCHEMA-E508"

    # E509: Schema constraints are inconsistent (e.g., min > max)
    SCHEMA_CONSTRAINT_INCONSISTENT = "GLSCHEMA-E509"

    # =========================================================================
    # Deprecation Warnings (GLSCHEMA-W6xx)
    # =========================================================================

    # W600: Using a field that has been deprecated
    DEPRECATED_FIELD = "GLSCHEMA-W600"

    # W601: Using old field name (field has been renamed)
    RENAMED_FIELD = "GLSCHEMA-W601"

    # W602: Using a field that will be removed in future version
    REMOVED_FIELD = "GLSCHEMA-W602"

    # W603: Using a deprecated schema version
    DEPRECATED_SCHEMA_VERSION = "GLSCHEMA-W603"

    # W604: Using a deprecated unit
    DEPRECATED_UNIT = "GLSCHEMA-W604"

    # W605: Using deprecated enum value
    DEPRECATED_ENUM_VALUE = "GLSCHEMA-W605"

    # =========================================================================
    # Lint Warnings (GLSCHEMA-W7xx) - Style and naming
    # =========================================================================

    # W700: Unknown field name is similar to a known field (possible typo)
    SUSPICIOUS_KEY = "GLSCHEMA-W700"

    # W701: Field name does not follow naming convention
    NONCOMPLIANT_CASING = "GLSCHEMA-W701"

    # W702: Unit format could be improved
    UNIT_FORMAT_STYLE = "GLSCHEMA-W702"

    # W703: Value could be more precise
    PRECISION_SUGGESTION = "GLSCHEMA-W703"

    # W704: Empty object or array (might be unintentional)
    EMPTY_CONTAINER = "GLSCHEMA-W704"

    # W705: Redundant or duplicate data
    REDUNDANT_DATA = "GLSCHEMA-W705"

    # W706: Value ordering suggestion (for readability)
    ORDERING_SUGGESTION = "GLSCHEMA-W706"

    # =========================================================================
    # Limit Errors (GLSCHEMA-E8xx) - Size and complexity limits
    # =========================================================================

    # E800: Payload exceeds maximum allowed size
    PAYLOAD_TOO_LARGE = "GLSCHEMA-E800"

    # E801: Object nesting exceeds maximum depth
    DEPTH_EXCEEDED = "GLSCHEMA-E801"

    # E802: Array contains too many items
    ITEMS_EXCEEDED = "GLSCHEMA-E802"

    # E803: Too many $ref expansions during resolution
    REFS_EXCEEDED = "GLSCHEMA-E803"

    # E804: Too many validation findings generated
    FINDINGS_EXCEEDED = "GLSCHEMA-E804"

    # E805: Total node count exceeds limit
    NODES_EXCEEDED = "GLSCHEMA-E805"

    # E806: Schema is too large
    SCHEMA_TOO_LARGE = "GLSCHEMA-E806"

    # E807: Regex pattern is too complex
    REGEX_TOO_COMPLEX = "GLSCHEMA-E807"

    # E808: Batch processing limit exceeded
    BATCH_LIMIT_EXCEEDED = "GLSCHEMA-E808"

    # E809: Processing timeout exceeded
    TIMEOUT_EXCEEDED = "GLSCHEMA-E809"


@dataclass(frozen=True)
class ErrorInfo:
    """Complete information about an error code.

    This dataclass provides all metadata needed to generate user-facing
    error messages with appropriate severity and context.

    Attributes:
        code: The GLSCHEMA-* error code string
        name: Human-readable name for the error
        category: Error category for grouping and filtering
        severity: Severity level (error, warning, info)
        message_template: Message template with {placeholder} variables
        hint_template: Optional hint for fixing the error
        documentation_url: Optional URL to detailed documentation

    Example:
        >>> info = ErrorInfo(
        ...     code="GLSCHEMA-E100",
        ...     name="MISSING_REQUIRED",
        ...     category=ErrorCategory.STRUCTURAL,
        ...     severity=Severity.ERROR,
        ...     message_template="Required field '{field}' is missing at path '{path}'",
        ...     hint_template="Add the required field with an appropriate value"
        ... )
    """
    code: str
    name: str
    category: ErrorCategory
    severity: Severity
    message_template: str
    hint_template: Optional[str] = None
    documentation_url: Optional[str] = None


# =============================================================================
# Error Registry - Maps ErrorCode to ErrorInfo
# =============================================================================

ERROR_REGISTRY: Dict[ErrorCode, ErrorInfo] = {
    # -------------------------------------------------------------------------
    # Structural Errors (GLSCHEMA-E1xx)
    # -------------------------------------------------------------------------
    ErrorCode.MISSING_REQUIRED: ErrorInfo(
        code="GLSCHEMA-E100",
        name="MISSING_REQUIRED",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Required field '{field}' is missing at path '{path}'",
        hint_template="Add the required field '{field}' with a value of type '{expected_type}'"
    ),

    ErrorCode.UNKNOWN_FIELD: ErrorInfo(
        code="GLSCHEMA-E101",
        name="UNKNOWN_FIELD",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Unknown field '{field}' at path '{path}' (strict mode enabled)",
        hint_template="Remove the unknown field or check for typos. Valid fields are: {valid_fields}"
    ),

    ErrorCode.TYPE_MISMATCH: ErrorInfo(
        code="GLSCHEMA-E102",
        name="TYPE_MISMATCH",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Expected type '{expected}' but found '{actual}' at path '{path}'",
        hint_template="Convert the value to type '{expected}' or check the input data"
    ),

    ErrorCode.INVALID_NULL: ErrorInfo(
        code="GLSCHEMA-E103",
        name="INVALID_NULL",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Null value not allowed at path '{path}'",
        hint_template="Provide a non-null value of type '{expected_type}'"
    ),

    ErrorCode.CONTAINER_TYPE_MISMATCH: ErrorInfo(
        code="GLSCHEMA-E104",
        name="CONTAINER_TYPE_MISMATCH",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Expected {expected} but found {actual} at path '{path}'",
        hint_template="Ensure the value is a {expected}"
    ),

    ErrorCode.PROPERTY_COUNT_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E105",
        name="PROPERTY_COUNT_VIOLATION",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Object at path '{path}' has {actual} properties, but {constraint} {limit} allowed",
        hint_template="Adjust the number of properties to be within the allowed range"
    ),

    ErrorCode.REQUIRED_PROPERTIES_MISSING: ErrorInfo(
        code="GLSCHEMA-E106",
        name="REQUIRED_PROPERTIES_MISSING",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Missing {count} required properties at path '{path}': {missing_fields}",
        hint_template="Add all required properties: {missing_fields}"
    ),

    ErrorCode.DUPLICATE_KEY: ErrorInfo(
        code="GLSCHEMA-E107",
        name="DUPLICATE_KEY",
        category=ErrorCategory.STRUCTURAL,
        severity=Severity.ERROR,
        message_template="Duplicate key '{key}' found at path '{path}' after alias resolution",
        hint_template="Remove the duplicate key or use a different name"
    ),

    # -------------------------------------------------------------------------
    # Constraint Errors (GLSCHEMA-E2xx)
    # -------------------------------------------------------------------------
    ErrorCode.RANGE_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E200",
        name="RANGE_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value {value} at path '{path}' is outside allowed range [{min}, {max}]",
        hint_template="Provide a value between {min} and {max}"
    ),

    ErrorCode.PATTERN_MISMATCH: ErrorInfo(
        code="GLSCHEMA-E201",
        name="PATTERN_MISMATCH",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value '{value}' at path '{path}' does not match pattern '{pattern}'",
        hint_template="Ensure the value matches the required pattern: {pattern}"
    ),

    ErrorCode.ENUM_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E202",
        name="ENUM_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value '{value}' at path '{path}' is not one of allowed values: {allowed}",
        hint_template="Use one of the allowed values: {allowed}"
    ),

    ErrorCode.LENGTH_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E203",
        name="LENGTH_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Length {actual_length} at path '{path}' violates constraint: {constraint}",
        hint_template="Adjust the length to satisfy: {constraint}"
    ),

    ErrorCode.UNIQUE_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E204",
        name="UNIQUE_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Array at path '{path}' contains duplicate items at indices {indices}",
        hint_template="Remove duplicate items to ensure all array elements are unique"
    ),

    ErrorCode.MULTIPLE_OF_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E205",
        name="MULTIPLE_OF_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value {value} at path '{path}' is not a multiple of {multiple_of}",
        hint_template="Provide a value that is a multiple of {multiple_of}"
    ),

    ErrorCode.FORMAT_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E206",
        name="FORMAT_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value '{value}' at path '{path}' does not match format '{format}'",
        hint_template="Provide a value in the correct '{format}' format"
    ),

    ErrorCode.CONST_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E207",
        name="CONST_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Value at path '{path}' must be exactly '{expected}', found '{actual}'",
        hint_template="Set the value to '{expected}'"
    ),

    ErrorCode.CONTAINS_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E208",
        name="CONTAINS_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Array at path '{path}' does not contain required item matching schema",
        hint_template="Add an item to the array that matches the required schema"
    ),

    ErrorCode.PROPERTY_NAME_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E209",
        name="PROPERTY_NAME_VIOLATION",
        category=ErrorCategory.CONSTRAINT,
        severity=Severity.ERROR,
        message_template="Property name '{property_name}' at path '{path}' does not match pattern '{pattern}'",
        hint_template="Rename the property to match pattern: {pattern}"
    ),

    # -------------------------------------------------------------------------
    # Unit Errors (GLSCHEMA-E3xx)
    # -------------------------------------------------------------------------
    ErrorCode.UNIT_MISSING: ErrorInfo(
        code="GLSCHEMA-E300",
        name="UNIT_MISSING",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Required unit not provided for value at path '{path}'",
        hint_template="Provide a unit from dimension '{dimension}'. Allowed units: {allowed_units}"
    ),

    ErrorCode.UNIT_INCOMPATIBLE: ErrorInfo(
        code="GLSCHEMA-E301",
        name="UNIT_INCOMPATIBLE",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Unit '{unit}' at path '{path}' is incompatible with dimension '{expected_dimension}'",
        hint_template="Use a unit from dimension '{expected_dimension}': {compatible_units}"
    ),

    ErrorCode.UNIT_NONCANONICAL: ErrorInfo(
        code="GLSCHEMA-E302",
        name="UNIT_NONCANONICAL",
        category=ErrorCategory.UNIT,
        severity=Severity.WARNING,
        message_template="Unit '{unit}' at path '{path}' is not in canonical form",
        hint_template="Use the canonical unit '{canonical_unit}' instead"
    ),

    ErrorCode.UNIT_UNKNOWN: ErrorInfo(
        code="GLSCHEMA-E303",
        name="UNIT_UNKNOWN",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Unit '{unit}' at path '{path}' is not recognized in the unit catalog",
        hint_template="Check spelling or register the unit. Known units for dimension '{dimension}': {known_units}"
    ),

    ErrorCode.UNIT_CONVERSION_FAILED: ErrorInfo(
        code="GLSCHEMA-E304",
        name="UNIT_CONVERSION_FAILED",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Failed to convert from '{from_unit}' to '{to_unit}' at path '{path}': {reason}",
        hint_template="Verify the unit conversion is possible between these units"
    ),

    ErrorCode.UNIT_FORMAT_INVALID: ErrorInfo(
        code="GLSCHEMA-E305",
        name="UNIT_FORMAT_INVALID",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Invalid unit format at path '{path}': {details}",
        hint_template="Use format {{ value: <number>, unit: '<unit>' }} or '<number> <unit>'"
    ),

    ErrorCode.DIMENSION_INVALID: ErrorInfo(
        code="GLSCHEMA-E306",
        name="DIMENSION_INVALID",
        category=ErrorCategory.UNIT,
        severity=Severity.ERROR,
        message_template="Invalid dimension '{dimension}' specified at path '{path}'",
        hint_template="Use a valid dimension from: {valid_dimensions}"
    ),

    # -------------------------------------------------------------------------
    # Rule Errors (GLSCHEMA-E4xx)
    # -------------------------------------------------------------------------
    ErrorCode.RULE_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E400",
        name="RULE_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Validation rule '{rule_id}' failed: {message}",
        hint_template="{hint}"
    ),

    ErrorCode.CONDITIONAL_REQUIRED: ErrorInfo(
        code="GLSCHEMA-E401",
        name="CONDITIONAL_REQUIRED",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Field '{field}' is required when {condition}",
        hint_template="Add field '{field}' or change the condition to not require it"
    ),

    ErrorCode.CONSISTENCY_ERROR: ErrorInfo(
        code="GLSCHEMA-E402",
        name="CONSISTENCY_ERROR",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Consistency check failed at path '{path}': {message}",
        hint_template="Ensure values are consistent: {expected_relationship}"
    ),

    ErrorCode.DEPENDENCY_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E403",
        name="DEPENDENCY_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Field '{field}' requires dependent fields: {dependencies}",
        hint_template="Add the required dependent fields: {dependencies}"
    ),

    ErrorCode.MUTUAL_EXCLUSION_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E404",
        name="MUTUAL_EXCLUSION_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Fields {fields} are mutually exclusive but multiple are present",
        hint_template="Provide only one of: {fields}"
    ),

    ErrorCode.ONE_OF_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E405",
        name="ONE_OF_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Value at path '{path}' matches {match_count} schemas but exactly one is required",
        hint_template="Modify the value to match exactly one of the allowed schemas"
    ),

    ErrorCode.ANY_OF_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E406",
        name="ANY_OF_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Value at path '{path}' does not match any of the allowed schemas",
        hint_template="Modify the value to match at least one of the allowed schemas"
    ),

    ErrorCode.ALL_OF_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E407",
        name="ALL_OF_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Value at path '{path}' does not satisfy all required schemas ({satisfied}/{total})",
        hint_template="Modify the value to satisfy all {total} required schemas"
    ),

    ErrorCode.NOT_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E408",
        name="NOT_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Value at path '{path}' matches a forbidden schema",
        hint_template="Modify the value so it does not match the forbidden schema"
    ),

    ErrorCode.IF_THEN_ELSE_VIOLATION: ErrorInfo(
        code="GLSCHEMA-E409",
        name="IF_THEN_ELSE_VIOLATION",
        category=ErrorCategory.RULE,
        severity=Severity.ERROR,
        message_template="Conditional validation failed at path '{path}': {message}",
        hint_template="Ensure the value satisfies the conditional requirement"
    ),

    # -------------------------------------------------------------------------
    # Schema Errors (GLSCHEMA-E5xx)
    # -------------------------------------------------------------------------
    ErrorCode.REF_RESOLUTION_FAILED: ErrorInfo(
        code="GLSCHEMA-E500",
        name="REF_RESOLUTION_FAILED",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Failed to resolve $ref '{ref}' at path '{path}'",
        hint_template="Check that the referenced schema exists: {ref}"
    ),

    ErrorCode.CIRCULAR_REF: ErrorInfo(
        code="GLSCHEMA-E501",
        name="CIRCULAR_REF",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Circular reference detected: {cycle}",
        hint_template="Break the circular reference chain: {cycle}"
    ),

    ErrorCode.SCHEMA_INVALID: ErrorInfo(
        code="GLSCHEMA-E502",
        name="SCHEMA_INVALID",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Schema is invalid: {reason}",
        hint_template="Fix the schema: {details}"
    ),

    ErrorCode.SCHEMA_VERSION_MISMATCH: ErrorInfo(
        code="GLSCHEMA-E503",
        name="SCHEMA_VERSION_MISMATCH",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Schema version '{version}' is incompatible with validator version '{validator_version}'",
        hint_template="Use a compatible schema version or upgrade the validator"
    ),

    ErrorCode.SCHEMA_DIALECT_UNSUPPORTED: ErrorInfo(
        code="GLSCHEMA-E504",
        name="SCHEMA_DIALECT_UNSUPPORTED",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Schema dialect '{dialect}' is not supported",
        hint_template="Use a supported dialect: {supported_dialects}"
    ),

    ErrorCode.SCHEMA_DEFINITION_NOT_FOUND: ErrorInfo(
        code="GLSCHEMA-E505",
        name="SCHEMA_DEFINITION_NOT_FOUND",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Schema definition '{definition}' not found in '{schema_id}'",
        hint_template="Check that the definition exists in the schema"
    ),

    ErrorCode.SCHEMA_REGISTRY_ERROR: ErrorInfo(
        code="GLSCHEMA-E506",
        name="SCHEMA_REGISTRY_ERROR",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Failed to fetch schema from registry: {error}",
        hint_template="Check registry connectivity and schema availability"
    ),

    ErrorCode.SCHEMA_PARSE_ERROR: ErrorInfo(
        code="GLSCHEMA-E507",
        name="SCHEMA_PARSE_ERROR",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Failed to parse schema: {error}",
        hint_template="Check schema syntax at line {line}, column {column}"
    ),

    ErrorCode.SCHEMA_KEYWORD_INVALID: ErrorInfo(
        code="GLSCHEMA-E508",
        name="SCHEMA_KEYWORD_INVALID",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Invalid or unsupported keyword '{keyword}' in schema",
        hint_template="Remove the invalid keyword or use a supported alternative"
    ),

    ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT: ErrorInfo(
        code="GLSCHEMA-E509",
        name="SCHEMA_CONSTRAINT_INCONSISTENT",
        category=ErrorCategory.SCHEMA,
        severity=Severity.ERROR,
        message_template="Schema constraints are inconsistent: {details}",
        hint_template="Fix the constraint: {suggestion}"
    ),

    # -------------------------------------------------------------------------
    # Deprecation Warnings (GLSCHEMA-W6xx)
    # -------------------------------------------------------------------------
    ErrorCode.DEPRECATED_FIELD: ErrorInfo(
        code="GLSCHEMA-W600",
        name="DEPRECATED_FIELD",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Field '{field}' at path '{path}' is deprecated since version {since_version}",
        hint_template="Use '{replacement}' instead. This field will be removed in version {removal_version}"
    ),

    ErrorCode.RENAMED_FIELD: ErrorInfo(
        code="GLSCHEMA-W601",
        name="RENAMED_FIELD",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Field '{old_name}' at path '{path}' has been renamed to '{new_name}'",
        hint_template="Update to use the new field name '{new_name}'"
    ),

    ErrorCode.REMOVED_FIELD: ErrorInfo(
        code="GLSCHEMA-W602",
        name="REMOVED_FIELD",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Field '{field}' at path '{path}' will be removed in version {removal_version}",
        hint_template="Remove usage of this field or migrate to '{replacement}'"
    ),

    ErrorCode.DEPRECATED_SCHEMA_VERSION: ErrorInfo(
        code="GLSCHEMA-W603",
        name="DEPRECATED_SCHEMA_VERSION",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Schema version '{version}' is deprecated",
        hint_template="Upgrade to schema version '{recommended_version}'"
    ),

    ErrorCode.DEPRECATED_UNIT: ErrorInfo(
        code="GLSCHEMA-W604",
        name="DEPRECATED_UNIT",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Unit '{unit}' at path '{path}' is deprecated",
        hint_template="Use '{replacement_unit}' instead"
    ),

    ErrorCode.DEPRECATED_ENUM_VALUE: ErrorInfo(
        code="GLSCHEMA-W605",
        name="DEPRECATED_ENUM_VALUE",
        category=ErrorCategory.DEPRECATION,
        severity=Severity.WARNING,
        message_template="Enum value '{value}' at path '{path}' is deprecated",
        hint_template="Use '{replacement_value}' instead"
    ),

    # -------------------------------------------------------------------------
    # Lint Warnings (GLSCHEMA-W7xx)
    # -------------------------------------------------------------------------
    ErrorCode.SUSPICIOUS_KEY: ErrorInfo(
        code="GLSCHEMA-W700",
        name="SUSPICIOUS_KEY",
        category=ErrorCategory.LINT,
        severity=Severity.WARNING,
        message_template="Unknown field '{field}' at path '{path}'. Did you mean '{suggestion}'?",
        hint_template="Check spelling. Similar known fields: {similar_fields}"
    ),

    ErrorCode.NONCOMPLIANT_CASING: ErrorInfo(
        code="GLSCHEMA-W701",
        name="NONCOMPLIANT_CASING",
        category=ErrorCategory.LINT,
        severity=Severity.WARNING,
        message_template="Field '{field}' at path '{path}' does not follow {convention} naming convention",
        hint_template="Rename to '{suggested_name}' to follow {convention} convention"
    ),

    ErrorCode.UNIT_FORMAT_STYLE: ErrorInfo(
        code="GLSCHEMA-W702",
        name="UNIT_FORMAT_STYLE",
        category=ErrorCategory.LINT,
        severity=Severity.WARNING,
        message_template="Unit format at path '{path}' could be improved",
        hint_template="Preferred format: {preferred_format}"
    ),

    ErrorCode.PRECISION_SUGGESTION: ErrorInfo(
        code="GLSCHEMA-W703",
        name="PRECISION_SUGGESTION",
        category=ErrorCategory.LINT,
        severity=Severity.INFO,
        message_template="Value at path '{path}' has unusual precision ({precision} decimal places)",
        hint_template="Consider using {suggested_precision} decimal places for this field"
    ),

    ErrorCode.EMPTY_CONTAINER: ErrorInfo(
        code="GLSCHEMA-W704",
        name="EMPTY_CONTAINER",
        category=ErrorCategory.LINT,
        severity=Severity.INFO,
        message_template="Empty {container_type} at path '{path}'",
        hint_template="Verify this empty {container_type} is intentional"
    ),

    ErrorCode.REDUNDANT_DATA: ErrorInfo(
        code="GLSCHEMA-W705",
        name="REDUNDANT_DATA",
        category=ErrorCategory.LINT,
        severity=Severity.INFO,
        message_template="Redundant data detected at path '{path}': {details}",
        hint_template="Consider removing the redundant data"
    ),

    ErrorCode.ORDERING_SUGGESTION: ErrorInfo(
        code="GLSCHEMA-W706",
        name="ORDERING_SUGGESTION",
        category=ErrorCategory.LINT,
        severity=Severity.INFO,
        message_template="Array at path '{path}' could benefit from ordering by '{order_key}'",
        hint_template="Consider sorting the array by '{order_key}' for consistency"
    ),

    # -------------------------------------------------------------------------
    # Limit Errors (GLSCHEMA-E8xx)
    # -------------------------------------------------------------------------
    ErrorCode.PAYLOAD_TOO_LARGE: ErrorInfo(
        code="GLSCHEMA-E800",
        name="PAYLOAD_TOO_LARGE",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Payload size {size_bytes} bytes exceeds maximum allowed {max_bytes} bytes",
        hint_template="Reduce payload size to under {max_bytes} bytes"
    ),

    ErrorCode.DEPTH_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E801",
        name="DEPTH_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Nesting depth {depth} at path '{path}' exceeds maximum {max_depth}",
        hint_template="Flatten the data structure to reduce nesting depth below {max_depth}"
    ),

    ErrorCode.ITEMS_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E802",
        name="ITEMS_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Array at path '{path}' has {count} items, exceeding maximum {max_items}",
        hint_template="Reduce array size to {max_items} items or split into multiple requests"
    ),

    ErrorCode.REFS_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E803",
        name="REFS_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="$ref expansion count {count} exceeds maximum {max_refs}",
        hint_template="Simplify schema structure to reduce $ref chain depth"
    ),

    ErrorCode.FINDINGS_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E804",
        name="FINDINGS_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Validation generated {count} findings, exceeding maximum {max_findings}. Additional findings truncated.",
        hint_template="Fix the most critical errors first and re-validate"
    ),

    ErrorCode.NODES_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E805",
        name="NODES_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Total node count {count} exceeds maximum {max_nodes}",
        hint_template="Reduce payload complexity to under {max_nodes} total nodes"
    ),

    ErrorCode.SCHEMA_TOO_LARGE: ErrorInfo(
        code="GLSCHEMA-E806",
        name="SCHEMA_TOO_LARGE",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Schema size {size_bytes} bytes exceeds maximum allowed {max_bytes} bytes",
        hint_template="Split schema into smaller components or reduce complexity"
    ),

    ErrorCode.REGEX_TOO_COMPLEX: ErrorInfo(
        code="GLSCHEMA-E807",
        name="REGEX_TOO_COMPLEX",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Regex pattern at path '{path}' has complexity score {score}, exceeding maximum {max_score}",
        hint_template="Simplify the regex pattern or use multiple simpler patterns"
    ),

    ErrorCode.BATCH_LIMIT_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E808",
        name="BATCH_LIMIT_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Batch contains {count} items, exceeding maximum {max_items}",
        hint_template="Split the batch into smaller batches of at most {max_items} items"
    ),

    ErrorCode.TIMEOUT_EXCEEDED: ErrorInfo(
        code="GLSCHEMA-E809",
        name="TIMEOUT_EXCEEDED",
        category=ErrorCategory.LIMIT,
        severity=Severity.ERROR,
        message_template="Validation timeout after {elapsed_ms}ms (limit: {timeout_ms}ms)",
        hint_template="Simplify the payload or increase the timeout limit"
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_error_info(code: Union[ErrorCode, str]) -> ErrorInfo:
    """Get complete information about an error code.

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string

    Returns:
        ErrorInfo with code, category, severity, and message template

    Raises:
        KeyError: If error code is not found in registry

    Example:
        >>> info = get_error_info(ErrorCode.MISSING_REQUIRED)
        >>> print(info.severity)
        Severity.ERROR
        >>> print(info.category)
        ErrorCategory.STRUCTURAL
    """
    if isinstance(code, str):
        # Convert string to ErrorCode
        for error_code in ErrorCode:
            if error_code.value == code:
                code = error_code
                break
        else:
            raise KeyError(f"Unknown error code: {code}")

    if code not in ERROR_REGISTRY:
        raise KeyError(f"Error code {code} not found in registry")

    return ERROR_REGISTRY[code]


def format_error_message(
    code: Union[ErrorCode, str],
    **kwargs: Any
) -> str:
    """Format an error message with context values.

    Substitutes placeholders in the message template with provided values.
    Unknown placeholders are left as-is.

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string
        **kwargs: Values to substitute in the message template

    Returns:
        Formatted error message string

    Example:
        >>> msg = format_error_message(
        ...     ErrorCode.TYPE_MISMATCH,
        ...     path="/data/value",
        ...     expected="integer",
        ...     actual="string"
        ... )
        >>> print(msg)
        Expected type 'integer' but found 'string' at path '/data/value'
    """
    info = get_error_info(code)
    try:
        return info.message_template.format(**kwargs)
    except KeyError:
        # Return template with available substitutions
        result = info.message_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


def format_error_hint(
    code: Union[ErrorCode, str],
    **kwargs: Any
) -> Optional[str]:
    """Format an error hint with context values.

    Substitutes placeholders in the hint template with provided values.
    Returns None if no hint is available for this error code.

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string
        **kwargs: Values to substitute in the hint template

    Returns:
        Formatted hint string or None if no hint available

    Example:
        >>> hint = format_error_hint(
        ...     ErrorCode.MISSING_REQUIRED,
        ...     field="energy",
        ...     expected_type="number"
        ... )
        >>> print(hint)
        Add the required field 'energy' with a value of type 'number'
    """
    info = get_error_info(code)
    if not info.hint_template:
        return None
    try:
        return info.hint_template.format(**kwargs)
    except KeyError:
        # Return template with available substitutions
        result = info.hint_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


def is_error(code: Union[ErrorCode, str]) -> bool:
    """Check if an error code represents an error (not warning/info).

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string

    Returns:
        True if the code is an error, False for warnings/info

    Example:
        >>> is_error(ErrorCode.MISSING_REQUIRED)
        True
        >>> is_error(ErrorCode.DEPRECATED_FIELD)
        False
    """
    info = get_error_info(code)
    return info.severity == Severity.ERROR


def is_warning(code: Union[ErrorCode, str]) -> bool:
    """Check if an error code represents a warning.

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string

    Returns:
        True if the code is a warning, False otherwise

    Example:
        >>> is_warning(ErrorCode.DEPRECATED_FIELD)
        True
        >>> is_warning(ErrorCode.MISSING_REQUIRED)
        False
    """
    info = get_error_info(code)
    return info.severity == Severity.WARNING


def is_info(code: Union[ErrorCode, str]) -> bool:
    """Check if an error code represents informational message.

    Args:
        code: ErrorCode enum value or GLSCHEMA-* string

    Returns:
        True if the code is info level, False otherwise

    Example:
        >>> is_info(ErrorCode.EMPTY_CONTAINER)
        True
        >>> is_info(ErrorCode.MISSING_REQUIRED)
        False
    """
    info = get_error_info(code)
    return info.severity == Severity.INFO


def get_codes_by_category(category: ErrorCategory) -> List[ErrorCode]:
    """Get all error codes in a specific category.

    Args:
        category: ErrorCategory to filter by

    Returns:
        List of ErrorCode values in the specified category

    Example:
        >>> structural_codes = get_codes_by_category(ErrorCategory.STRUCTURAL)
        >>> len(structural_codes) > 0
        True
    """
    return [
        code for code, info in ERROR_REGISTRY.items()
        if info.category == category
    ]


def get_codes_by_severity(severity: Severity) -> List[ErrorCode]:
    """Get all error codes with a specific severity.

    Args:
        severity: Severity level to filter by

    Returns:
        List of ErrorCode values with the specified severity

    Example:
        >>> errors = get_codes_by_severity(Severity.ERROR)
        >>> warnings = get_codes_by_severity(Severity.WARNING)
        >>> len(errors) > len(warnings)
        True
    """
    return [
        code for code, info in ERROR_REGISTRY.items()
        if info.severity == severity
    ]


def get_all_error_codes() -> List[str]:
    """Get all registered error code strings.

    Returns:
        List of all GLSCHEMA-* error code strings

    Example:
        >>> codes = get_all_error_codes()
        >>> "GLSCHEMA-E100" in codes
        True
    """
    return [info.code for info in ERROR_REGISTRY.values()]


def validate_error_code(code: str) -> bool:
    """Check if a string is a valid GLSCHEMA-* error code.

    Args:
        code: String to validate

    Returns:
        True if the code is valid, False otherwise

    Example:
        >>> validate_error_code("GLSCHEMA-E100")
        True
        >>> validate_error_code("INVALID-CODE")
        False
    """
    return code in get_all_error_codes()


def get_error_by_code(code: str) -> Optional[ErrorCode]:
    """Look up an ErrorCode enum by its string code.

    This is a convenience function for backward compatibility.

    Args:
        code: The error code string (e.g., "GLSCHEMA-E100")

    Returns:
        The ErrorCode enum value, or None if not found.

    Example:
        >>> error = get_error_by_code("GLSCHEMA-E100")
        >>> print(error)
        ErrorCode.MISSING_REQUIRED
    """
    for error_code in ErrorCode:
        if error_code.value == code:
            return error_code
    return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "Severity",
    "ErrorCategory",
    "ErrorCode",
    # Dataclass
    "ErrorInfo",
    # Registry
    "ERROR_REGISTRY",
    # Helper functions
    "get_error_info",
    "format_error_message",
    "format_error_hint",
    "is_error",
    "is_warning",
    "is_info",
    "get_codes_by_category",
    "get_codes_by_severity",
    "get_all_error_codes",
    "validate_error_code",
    "get_error_by_code",
]
