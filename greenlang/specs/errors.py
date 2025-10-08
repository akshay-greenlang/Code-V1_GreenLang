"""
GreenLang AgentSpec v2 Validation Errors

This module defines the error model for AgentSpec v2 validation.
All validation errors are mapped to stable error codes for tooling, CI/CD, and developer debugging.

Design Philosophy:
- Fail early, fail specific: Every failure maps to a stable code
- Strict by default: Unknown fields are errors
- Stable codes enable automation (CI checks, linting, code generation)

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-201 (AgentSpec v2 Schema + Validators)
"""

from enum import Enum
from typing import List, Optional, Any
from pydantic import ValidationError


class GLVErr(str, Enum):
    """
    Stable error codes for AgentSpec v2 validation.

    These codes are part of the public API and MUST NOT change between minor versions.
    Add new codes at the end to maintain backward compatibility.

    Error Code Convention:
    - GLValidationError.* prefix for all validation errors
    - Specific codes enable automated error handling (CI, linters, IDEs)
    """

    # Field-level errors
    MISSING_FIELD = "GLValidationError.MISSING_FIELD"
    """Required field is missing from specification"""

    UNKNOWN_FIELD = "GLValidationError.UNKNOWN_FIELD"
    """Unknown field detected (typo or unsupported field)"""

    # Format validation errors
    INVALID_SEMVER = "GLValidationError.INVALID_SEMVER"
    """Version string does not conform to Semantic Versioning 2.0.0"""

    INVALID_SLUG = "GLValidationError.INVALID_SLUG"
    """Agent ID slug does not match required pattern (lowercase, /, -, _)"""

    INVALID_URI = "GLValidationError.INVALID_URI"
    """URI scheme is invalid or malformed (python://, ef://, etc.)"""

    # Duplicate detection
    DUPLICATE_NAME = "GLValidationError.DUPLICATE_NAME"
    """Duplicate name detected across inputs/outputs/tools/connectors"""

    DUPLICATE_ID = "GLValidationError.DUPLICATE_ID"
    """Duplicate agent ID in registry"""

    # Unit and quantity errors
    UNIT_SYNTAX = "GLValidationError.UNIT_SYNTAX"
    """Unit string has invalid syntax (not parseable)"""

    UNIT_FORBIDDEN = "GLValidationError.UNIT_FORBIDDEN"
    """Non-dimensionless unit used where dimensionless ("1") required"""

    # Constraint violations
    CONSTRAINT = "GLValidationError.CONSTRAINT"
    """Value violates constraint (ge/gt/le/lt, enum, pattern, etc.)"""

    # Domain-specific errors
    FACTOR_UNRESOLVED = "GLValidationError.FACTOR_UNRESOLVED"
    """Emission factor reference cannot be resolved (ef:// URI not found)"""

    AI_SCHEMA_INVALID = "GLValidationError.AI_SCHEMA_INVALID"
    """AI tool schema_in or schema_out is not valid JSON Schema draft-2020-12"""

    BUDGET_INVALID = "GLValidationError.BUDGET_INVALID"
    """AI budget constraint is invalid or violated"""

    MODE_INVALID = "GLValidationError.MODE_INVALID"
    """Realtime mode is invalid (only 'replay' or 'live' allowed)"""

    CONNECTOR_INVALID = "GLValidationError.CONNECTOR_INVALID"
    """Realtime connector configuration is invalid"""

    PROVENANCE_INVALID = "GLValidationError.PROVENANCE_INVALID"
    """Provenance configuration is invalid (e.g., pin_ef=true but no factors)"""


class GLValidationError(ValueError):
    """
    GreenLang validation error with structured error code and path.

    This exception wraps Pydantic ValidationErrors and provides:
    - Stable error codes (GLVErr) for automation
    - Field path for precise error location
    - Human-readable error messages
    - Machine-parseable structure for CI/CD tools

    Example:
        >>> raise GLValidationError(
        ...     GLVErr.MISSING_FIELD,
        ...     "compute.inputs is required",
        ...     ["compute", "inputs"]
        ... )
        GLValidationError.MISSING_FIELD: compute/inputs: compute.inputs is required

    Attributes:
        code: Stable error code from GLVErr enum
        message: Human-readable error description
        path: Field path as list of strings (e.g., ["compute", "inputs", "fuel_volume"])
        context: Optional additional context (e.g., field value, constraint)
    """

    def __init__(
        self,
        code: GLVErr,
        message: str,
        path: Optional[List[str]] = None,
        context: Optional[Any] = None
    ):
        """
        Initialize GLValidationError.

        Args:
            code: Error code from GLVErr enum
            message: Human-readable error message
            path: Field path as list (e.g., ["compute", "inputs", "fuel"])
            context: Optional additional context (field value, constraint, etc.)
        """
        self.code = str(code)
        self.message = message
        self.path = path or []
        self.context = context

        # Format error message: CODE: path/to/field: message
        path_str = "/".join(self.path) if self.path else "(root)"
        full_message = f"{self.code}: {path_str}: {message}"

        super().__init__(full_message)

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with code, message, path, context fields
        """
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "context": self.context
        }

    @classmethod
    def from_pydantic(cls, error: ValidationError, context: Optional[str] = None) -> List['GLValidationError']:
        """
        Convert Pydantic ValidationError to list of GLValidationErrors.

        Maps Pydantic error types to stable GLVErr codes:
        - missing: MISSING_FIELD
        - extra_forbidden: UNKNOWN_FIELD
        - string_pattern_mismatch: INVALID_SEMVER, INVALID_SLUG, INVALID_URI (context-dependent)
        - greater_than_equal, less_than_equal, etc.: CONSTRAINT

        Args:
            error: Pydantic ValidationError
            context: Optional context string for error classification

        Returns:
            List of GLValidationErrors with appropriate codes
        """
        gl_errors = []

        for err in error.errors():
            err_type = err.get("type", "")
            loc = [str(l) for l in err.get("loc", [])]
            msg = err.get("msg", "")

            # Check if message contains a GLVErr code (from our custom validators)
            # Format: "Value error, GLVErr.CODE_NAME: ..."
            code = None
            if "GLVErr." in msg or "GLValidationError." in msg:
                # Extract the error code from the message
                for err_code in GLVErr:
                    if str(err_code) in msg:
                        code = err_code
                        break

            # If no code found in message, map Pydantic error types to GLVErr codes
            if code is None:
                if err_type == "missing":
                    code = GLVErr.MISSING_FIELD
                elif err_type in ("extra_forbidden", "unexpected_keyword_argument"):
                    code = GLVErr.UNKNOWN_FIELD
                elif err_type == "string_pattern_mismatch":
                    # Determine specific code based on field path
                    if "version" in loc:
                        code = GLVErr.INVALID_SEMVER
                    elif "id" in loc:
                        code = GLVErr.INVALID_SLUG
                    elif "uri" in loc[-1].lower() or "entrypoint" in loc or "ref" in loc or "impl" in loc:
                        code = GLVErr.INVALID_URI
                    else:
                        code = GLVErr.CONSTRAINT
                elif err_type in ("greater_than_equal", "less_than_equal", "greater_than", "less_than"):
                    code = GLVErr.CONSTRAINT
                elif err_type in ("enum", "literal_error"):
                    if "mode" in loc or "default_mode" in loc:
                        code = GLVErr.MODE_INVALID
                    else:
                        code = GLVErr.CONSTRAINT
                elif "unit" in str(loc):
                    code = GLVErr.UNIT_SYNTAX
                else:
                    # Default to CONSTRAINT for unknown types
                    code = GLVErr.CONSTRAINT

            gl_errors.append(cls(code, msg, loc, context=err.get("ctx")))

        return gl_errors


def raise_validation_error(code: GLVErr, message: str, path: Optional[List[str]] = None) -> None:
    """
    Convenience function to raise GLValidationError.

    Args:
        code: Error code from GLVErr enum
        message: Human-readable error message
        path: Optional field path

    Raises:
        GLValidationError
    """
    raise GLValidationError(code, message, path)
